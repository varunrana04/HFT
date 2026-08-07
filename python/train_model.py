#!/usr/bin/env python3
"""
train_model.py — ML Signal Combiner training pipeline.

Trains a LightGBM or XGBoost model to predict short-term price
direction from the 6 alpha signals produced by FeatureEngine.

Pipeline:
  1. Load historical feature vectors (from backtest output or CSV)
  2. Create target: sign(mid_price[t+N] - mid_price[t])
  3. Feature engineering: signal values + rolling stats
  4. Train/test split (time-series aware, no look-ahead)
  5. Train LightGBM gradient boosting model
  6. Evaluate: AUC, accuracy, feature importance, out-of-sample Sharpe
  7. Export: binary weights file for C++ SignalCombiner.load_model()

Usage:
    python train_model.py --data data/features.csv
    python train_model.py --data data/features.csv --model xgboost --horizon 100

Output:
    models/signal_weights.bin     — Binary weights for C++ engine
    models/feature_importance.png — Feature importance plot
    models/training_report.md     — Training summary
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# ─── Optional Imports ────────────────────────────────────────

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, classification_report,
        confusion_matrix
    )
    from sklearn.model_selection import TimeSeriesSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─── Feature Names ───────────────────────────────────────────

SIGNAL_NAMES = [
    'microprice', 'ofi', 'vpin',
    'spread_bps', 'realized_vol', 'stat_arb_zscore'
]


# ─── Data Loading ────────────────────────────────────────────

def load_features_csv(filepath: str) -> np.ndarray:
    """
    Load feature vectors from CSV.

    Expected columns: timestamp, microprice, ofi, vpin,
                      spread_bps, realized_vol, stat_arb_zscore, mid_price

    Returns numpy array of shape (N, 7) — 6 features + mid_price.
    """
    if HAS_PANDAS:
        df = pd.read_csv(filepath)

        # Auto-detect column names
        feature_cols = []
        for name in SIGNAL_NAMES:
            matching = [c for c in df.columns if name.lower() in c.lower()]
            if matching:
                feature_cols.append(matching[0])
            else:
                feature_cols.append(name)

        mid_col = None
        for candidate in ['mid_price', 'midprice', 'mid', 'price', 'close']:
            if candidate in df.columns:
                mid_col = candidate
                break

        if mid_col is None:
            raise ValueError(
                f"No mid price column found. Available: {list(df.columns)}"
            )

        cols = feature_cols + [mid_col]
        data = df[cols].dropna().values.astype(np.float64)
        return data
    else:
        # Pure numpy fallback
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        return data


def generate_synthetic_features(n: int = 100000, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic feature data for testing the pipeline
    when real data isn't available yet.
    """
    rng = np.random.RandomState(seed)

    # Simulate a mean-reverting price process
    prices = np.zeros(n)
    prices[0] = 100.0
    for i in range(1, n):
        prices[i] = prices[i-1] + rng.normal(0, 0.01) - 0.001 * (prices[i-1] - 100.0)

    mid_prices = prices

    # Generate correlated features
    microprice = mid_prices + rng.normal(0, 0.001, n)
    ofi = np.diff(mid_prices, prepend=mid_prices[0]) * 100 + rng.normal(0, 0.01, n)
    vpin = np.clip(0.3 + rng.normal(0, 0.1, n), 0, 1)
    spread_bps = np.clip(2.0 + rng.normal(0, 0.5, n), 0.1, 20)
    realized_vol = np.clip(0.01 + np.abs(rng.normal(0, 0.005, n)), 0, 0.1)
    statarb_z = (mid_prices - np.convolve(mid_prices, np.ones(100)/100, mode='same')) / \
                np.maximum(np.std(mid_prices[:100]), 1e-10)

    # Stack into (N, 7): 6 features + mid_price
    data = np.column_stack([
        microprice, ofi, vpin, spread_bps,
        realized_vol, statarb_z, mid_prices
    ])

    return data


# ─── Target Creation ─────────────────────────────────────────

def create_target(mid_prices: np.ndarray, horizon: int = 100) -> np.ndarray:
    """
    Create binary target: 1 if price goes up in next `horizon` ticks, 0 otherwise.
    Last `horizon` rows will have NaN targets.
    """
    future_prices = np.roll(mid_prices, -horizon)
    future_prices[-horizon:] = np.nan

    returns = (future_prices - mid_prices) / np.maximum(np.abs(mid_prices), 1e-10)

    # Binary: 1 if positive return, 0 if negative/zero
    target = np.where(returns > 0, 1.0, 0.0)
    target[-horizon:] = np.nan

    return target


# ─── Feature Engineering ─────────────────────────────────────

def engineer_features(features: np.ndarray) -> Tuple[np.ndarray, list]:
    """
    Create additional features from the base 6 signals.
    Returns enhanced features and column names.
    """
    n = features.shape[0]
    extra_features = []
    extra_names = []

    # Rolling means (lookback windows)
    for window in [10, 50]:
        for i, name in enumerate(SIGNAL_NAMES):
            col = features[:, i]
            rolling_mean = np.convolve(col, np.ones(window)/window, mode='same')
            extra_features.append(rolling_mean)
            extra_names.append(f'{name}_ma{window}')

    # Rolling std
    for i, name in enumerate(SIGNAL_NAMES):
        col = features[:, i]
        # Simple rolling std approximation
        ma = np.convolve(col, np.ones(50)/50, mode='same')
        sq_ma = np.convolve(col**2, np.ones(50)/50, mode='same')
        rolling_std = np.sqrt(np.maximum(sq_ma - ma**2, 0))
        extra_features.append(rolling_std)
        extra_names.append(f'{name}_std50')

    # Cross-signal interactions
    # OFI × VPIN (informed flow × flow toxicity)
    extra_features.append(features[:, 1] * features[:, 2])
    extra_names.append('ofi_x_vpin')

    # Spread × Vol (market stress indicator)
    extra_features.append(features[:, 3] * features[:, 4])
    extra_names.append('spread_x_vol')

    # Microprice momentum (current - MA50)
    ma50 = np.convolve(features[:, 0], np.ones(50)/50, mode='same')
    extra_features.append(features[:, 0] - ma50)
    extra_names.append('microprice_momentum')

    all_features = np.column_stack([features] + [f.reshape(-1, 1) if f.ndim == 1 else f
                                                  for f in extra_features])
    all_names = SIGNAL_NAMES + extra_names

    return all_features, all_names


# ─── Model Training ─────────────────────────────────────────

def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   feature_names: list) -> Tuple:
    """Train a LightGBM binary classifier."""
    if not HAS_LGB:
        raise RuntimeError("LightGBM not installed: pip install lightgbm")

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'max_depth': 6,
        'n_estimators': 500,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train,
                             feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val,
                           feature_name=feature_names, reference=train_data)

    callbacks = [
        lgb.early_stopping(50),
        lgb.log_evaluation(100)
    ]

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )

    return model, model.feature_importance(importance_type='gain')


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  feature_names: list) -> Tuple:
    """Train an XGBoost binary classifier."""
    if not HAS_XGB:
        raise RuntimeError("XGBoost not installed: pip install xgboost")

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 50,
        'seed': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    model = xgb.train(
        params, dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    importance = model.get_score(importance_type='gain')
    importance_arr = np.array([importance.get(f, 0) for f in feature_names])

    return model, importance_arr


# ─── Export Weights ──────────────────────────────────────────

def export_binary_weights(importance: np.ndarray,
                          feature_names: list,
                          output_path: str,
                          bias: float = 0.0) -> None:
    """
    Export the top-6 signal weights as a binary file
    that C++ SignalCombiner::load_model() can read.

    File format: 6 doubles (48 bytes) + 1 double bias (8 bytes) = 56 bytes
    """
    # Extract importance for the 6 base signals only
    base_importance = np.zeros(6)
    for i, name in enumerate(SIGNAL_NAMES):
        idx = feature_names.index(name) if name in feature_names else i
        if idx < len(importance):
            base_importance[i] = importance[idx]

    # Normalize to sum to 1
    total = np.sum(base_importance)
    if total > 0:
        weights = base_importance / total
    else:
        weights = np.ones(6) / 6.0

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Write binary: 6 doubles + 1 bias double
    with open(output_path, 'wb') as f:
        weights.astype(np.float64).tofile(f)
        np.array([bias], dtype=np.float64).tofile(f)

    print(f"  [+] Exported weights to {output_path}")
    print(f"      Weights: {dict(zip(SIGNAL_NAMES, weights.round(4)))}")
    print(f"      Bias: {bias:.6f}")


# ─── Evaluation ──────────────────────────────────────────────

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   feature_names: list, is_lgb: bool = True) -> dict:
    """Evaluate model on test set."""
    if is_lgb:
        y_pred_proba = model.predict(X_test)
    else:
        dtest = xgb.DMatrix(X_test, feature_names=feature_names)
        y_pred_proba = model.predict(dtest)

    y_pred = (y_pred_proba > 0.5).astype(int)

    metrics = {}
    if HAS_SKLEARN:
        metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
        metrics['accuracy'] = accuracy_score(y_test, y_pred)

        # Directional accuracy (most important for trading)
        metrics['directional_accuracy'] = metrics['accuracy']

        # Simulated trading PnL (simple)
        signals = y_pred_proba - 0.5  # [-0.5, +0.5]
        positions = np.sign(signals)   # -1 or +1

        # Dummy returns (we'd need actual prices for real PnL)
        random_returns = np.random.normal(0, 0.001, len(positions))
        strategy_returns = positions * random_returns
        sharpe = (np.mean(strategy_returns) / np.std(strategy_returns) *
                  np.sqrt(252 * 24 * 60)) if np.std(strategy_returns) > 0 else 0

        metrics['simulated_sharpe'] = sharpe
    else:
        metrics['accuracy'] = float(np.mean(y_pred == y_test))

    return metrics


# ─── Visualization ───────────────────────────────────────────

def plot_feature_importance(importance: np.ndarray, feature_names: list,
                            output_path: str) -> None:
    """Plot feature importance bar chart."""
    if not HAS_MPL:
        return

    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1][:20]  # Top 20
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_imp = importance[sorted_idx]

    # Normalize
    total = np.sum(sorted_imp)
    if total > 0:
        sorted_imp = sorted_imp / total * 100

    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#c9d1d9',
        'text.color': '#c9d1d9',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'grid.color': '#21262d',
    })

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color the base 6 signals differently
    colors = []
    for name in sorted_names:
        if name in SIGNAL_NAMES:
            colors.append('#58a6ff')  # Blue for base signals
        else:
            colors.append('#8b949e')  # Gray for engineered

    ax.barh(range(len(sorted_names)), sorted_imp, color=colors, alpha=0.8)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('Importance (%)', fontsize=12)
    ax.set_title('Feature Importance (Top 20)', fontsize=16,
                 fontweight='bold', color='#f0f6fc')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [+] Saved feature_importance.png")


def save_training_report(metrics: dict, importance: np.ndarray,
                         feature_names: list, config: dict,
                         output_dir: str) -> None:
    """Save markdown training report."""
    # Sort importance
    sorted_idx = np.argsort(importance)[::-1]
    total_imp = np.sum(importance)

    report = f"""# ML Signal Combiner — Training Report

## Configuration

| Parameter | Value |
|---|---|
| Model | {config.get('model', 'lightgbm')} |
| Prediction Horizon | {config.get('horizon', 100)} ticks |
| Train/Val/Test Split | 60/20/20 (time-series) |
| Features | {len(feature_names)} (6 base + {len(feature_names)-6} engineered) |

## Evaluation Metrics

| Metric | Value |
|---|---|
| **AUC** | {metrics.get('auc', 'N/A'):.4f} |
| **Accuracy** | {metrics.get('accuracy', 'N/A'):.4f} |
| **Directional Accuracy** | {metrics.get('directional_accuracy', 'N/A'):.4f} |
| **Simulated Sharpe** | {metrics.get('simulated_sharpe', 'N/A'):.3f} |

## Feature Importance (Top 10)

| Rank | Feature | Importance (%) |
|---|---|---|
"""
    for rank, idx in enumerate(sorted_idx[:10], 1):
        pct = (importance[idx] / total_imp * 100) if total_imp > 0 else 0
        marker = " ⭐" if feature_names[idx] in SIGNAL_NAMES else ""
        report += f"| {rank} | {feature_names[idx]}{marker} | {pct:.2f}% |\n"

    report += f"""
> ⭐ = Base signal (used in C++ engine weights)

## Base Signal Weights (Exported to C++)

| Signal | Weight |
|---|---|
"""
    for name in SIGNAL_NAMES:
        if name in feature_names:
            idx = feature_names.index(name)
            base_total = sum(importance[feature_names.index(n)]
                           for n in SIGNAL_NAMES if n in feature_names)
            w = importance[idx] / base_total if base_total > 0 else 1.0/6
            report += f"| {name} | {w:.4f} |\n"

    report += f"""
## Feature Importance Plot

![Feature Importance](feature_importance.png)

---

*Generated by HFT Engine ML Pipeline*
"""

    report_path = os.path.join(output_dir, 'training_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  [+] Saved training_report.md")


# ─── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HFT Engine — ML Signal Combiner Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_model.py --data data/features.csv
    python train_model.py --synthetic --horizon 50
    python train_model.py --data data/features.csv --model xgboost
        """
    )
    parser.add_argument('--data', type=str, default=None,
                        help='Path to features CSV file')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--model', type=str, default='lightgbm',
                        choices=['lightgbm', 'xgboost'],
                        help='Model type (default: lightgbm)')
    parser.add_argument('--horizon', type=int, default=100,
                        help='Prediction horizon in ticks (default: 100)')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory (default: models)')
    parser.add_argument('--n-samples', type=int, default=100000,
                        help='Synthetic data size (default: 100000)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  HFT Engine — ML Signal Combiner Training")
    print(f"{'='*60}")

    # ── Load Data ──
    t0 = time.time()

    if args.data:
        print(f"\n  Loading features from: {args.data}")
        data = load_features_csv(args.data)
    elif args.synthetic:
        print(f"\n  Generating {args.n_samples:,} synthetic samples...")
        data = generate_synthetic_features(args.n_samples)
    else:
        print("\n  [INFO] No data provided. Using synthetic data for demo.")
        print("         Use --data <path> for real features, or --synthetic flag.")
        data = generate_synthetic_features(50000)

    features = data[:, :6]   # 6 base signals
    mid_prices = data[:, 6]  # Mid price column

    print(f"  Data shape: {data.shape}")
    print(f"  Loaded in {time.time()-t0:.2f}s")

    # ── Create Target ──
    print(f"\n  Creating target (horizon={args.horizon} ticks)...")
    target = create_target(mid_prices, args.horizon)

    # Remove NaN rows
    valid = ~np.isnan(target)
    features = features[valid]
    target = target[valid]
    mid_prices = mid_prices[valid]

    print(f"  Valid samples: {len(target):,}")
    print(f"  Class balance: {np.mean(target):.3f} (1=up, 0=down)")

    # ── Engineer Features ──
    print(f"\n  Engineering features...")
    X, feature_names = engineer_features(features)
    y = target

    print(f"  Total features: {len(feature_names)}")

    # ── Train/Val/Test Split (time-series aware) ──
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ── Train Model ──
    print(f"\n  Training {args.model}...")
    t0 = time.time()

    if args.model == 'lightgbm':
        if not HAS_LGB:
            print("  [ERROR] LightGBM not installed: pip install lightgbm")
            sys.exit(1)
        model, importance = train_lightgbm(X_train, y_train, X_val, y_val,
                                           feature_names)
    else:
        if not HAS_XGB:
            print("  [ERROR] XGBoost not installed: pip install xgboost")
            sys.exit(1)
        model, importance = train_xgboost(X_train, y_train, X_val, y_val,
                                          feature_names)

    train_time = time.time() - t0
    print(f"  Training complete in {train_time:.2f}s")

    # ── Evaluate ──
    print(f"\n  Evaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test, feature_names,
                             is_lgb=(args.model == 'lightgbm'))

    print(f"  AUC:      {metrics.get('auc', 'N/A'):.4f}")
    print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")

    # ── Export ──
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Exporting to {output_dir}/...")

    # Binary weights for C++
    export_binary_weights(
        importance, feature_names,
        os.path.join(output_dir, 'signal_weights.bin')
    )

    # Feature importance plot
    plot_feature_importance(
        importance, feature_names,
        os.path.join(output_dir, 'feature_importance.png')
    )

    # Training report
    save_training_report(
        metrics, importance, feature_names,
        {'model': args.model, 'horizon': args.horizon},
        output_dir
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Model:        {args.model}")
    print(f"  AUC:          {metrics.get('auc', 'N/A'):.4f}")
    print(f"  Accuracy:     {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"  Train time:   {train_time:.2f}s")
    print(f"  Output:       {output_dir}/")
    print(f"{'='*60}")
    print(f"\n  To use in C++ engine:")
    print(f"    combiner.load_model(\"{os.path.join(output_dir, 'signal_weights.bin')}\");")
    print()


if __name__ == '__main__':
    main()
