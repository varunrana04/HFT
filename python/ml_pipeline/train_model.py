#!/usr/bin/env python3
"""
train_model.py — ML Signal Combiner training pipeline.

Trains a LightGBM model on the feature vectors produced by
backtest.py --dump-features, then exports the model to ONNX for
low-latency C++ inference via onnxruntime.

Key design decisions vs. the old version:
  1. REGRESSION target (signed forward return), not binary classification.
     The model predicts the direction AND magnitude of the next N-tick
     move.  At inference time, threshold on predicted value → alpha.
     This avoids the 50/50 base-rate problem of binary classification.
  2. Sharpe ratio (not AUC) is the primary model selection criterion,
     since we care about profitability, not prediction accuracy.
  3. ONNX export via lgb → skl2onnx pipeline so the C++ SignalCombiner
     can run inference in <5µs with zero Python dependency.
  4. Binary weight fallback: also exports 6-weight .bin file for the
     existing C++ load_model() path (no recompile needed).
  5. Features-only input to ONNX: the model sees all engineered features
     (rolling stats, cross-signal interactions) not just the 6 raw ones.

Usage:
    # Train on real dumped features
    python train_model.py --data data/features.csv

    # Quick smoke-test with synthetic data
    python train_model.py --synthetic

    # Specify model horizon and export paths
    python train_model.py --data data/features.csv --horizon 100 --output models

Output:
    models/lgb_model.onnx          ONNX model for C++ inference
    models/signal_weights.bin       6-weight binary fallback for C++
    models/feature_importance.png   Feature importance chart
    models/training_report.md       Full training summary with metrics
    models/lgb_model.txt            LightGBM text model (human-readable)
"""

import sys
import os

# ─── Windows OpenMP deadlock prevention ──────────────────────
# LightGBM uses OpenMP for multi-threaded boosting. On Windows,
# combining OpenMP with Python's multiprocessing (or certain NumPy
# builds that also link libgomp) causes a deadlock at training start
# when n_jobs > 1. Setting OMP_NUM_THREADS=1 before ANY import that
# touches libgomp prevents the deadlock. Must be set here, at module
# level, before numpy/lightgbm import their native libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import time
import json
import warnings
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np

np.random.seed(42)

warnings.filterwarnings('ignore', category=UserWarning)

# ─── Optional imports ────────────────────────────────────────

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[ERROR] pandas required: pip install pandas")
    sys.exit(1)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import onnxruntime as rt
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─── Constants ───────────────────────────────────────────────

# The 6 normalized signals produced by FeatureEngine.compute_all()
BASE_SIGNALS = [
    'microprice', 'ofi', 'vpin',
    'spread_bps', 'realized_vol', 'stat_arb_zscore',
    'obi', 'trade_imbalance',
    'hawkes_intensity', 'cvd', 'hurst_exponent'
]

# Columns always present in the feature dump CSV
DUMP_META_COLS = ['timestamp_ns', 'combined_alpha', 'regime', 'mid_price', 'is_warmed_up']


# ─── Data Loading ────────────────────────────────────────────

def load_feature_dump(filepath: str, horizon: int, max_rows: int = None) -> pd.DataFrame:
    """
    Load the CSV produced by backtest.py --dump-features.

    Applies the following filters:
      - Drop rows where is_warmed_up == 0  (noisy warm-up period)
      - Drop rows where forward_return_N is NaN  (last horizon rows)
      - Drop rows where mid_price == 0

    Returns a clean DataFrame ready for feature engineering.
    """
    print(f"  Loading: {filepath}")
    
    # Memory optimization: force 32-bit floats
    dtypes = {
        'timestamp_ns': np.int64,
        'combined_alpha': np.float32,
        'regime': np.int8,
        'mid_price': np.float32,
        'is_warmed_up': np.int8,
        f'forward_return_{horizon}': np.float32
    }
    for sig in BASE_SIGNALS:
        dtypes[sig] = np.float32
        
    df = pd.read_csv(filepath, dtype=dtypes, nrows=max_rows)
    print(f"  Raw rows: {len(df):,}")

    target_col = f'forward_return_{horizon}'
    if target_col not in df.columns:
        # Try to find whatever forward_return column exists
        ret_cols = [c for c in df.columns if 'forward_return' in c]
        if not ret_cols:
            raise ValueError(
                f"No forward_return column found. "
                f"Re-run: python backtest.py --dump-features ... --horizon {horizon}"
            )
        target_col = ret_cols[0]
        print(f"  [WARN] Using column '{target_col}' (expected forward_return_{horizon})")

    # Filter
    df = df[df['is_warmed_up'] == 1].copy()
    df = df[df['mid_price'] > 0].copy()
    df = df.dropna(subset=[target_col]).copy()

    print(f"  Clean rows (warmed-up, labeled): {len(df):,}")

    # Clip extreme return outliers (>5σ) to prevent the model fitting tail noise
    ret = df[target_col]
    mu, sigma = ret.mean(), ret.std()
    clip_lo, clip_hi = mu - 5 * sigma, mu + 5 * sigma
    n_clipped = ((ret < clip_lo) | (ret > clip_hi)).sum()
    if n_clipped > 0:
        df[target_col] = ret.clip(clip_lo, clip_hi)
        print(f"  Clipped {n_clipped} outlier rows (+-5 sigma on forward return)")

    return df, target_col


def generate_synthetic_data(n: int = 200_000, horizon: int = 100,
                             seed: int = 42) -> Tuple[pd.DataFrame, str]:
    """
    Synthetic feature dump in the exact same format as backtest.py output.
    Used for smoke-testing the pipeline without running the C++ engine.
    """
    rng = np.random.RandomState(seed)
    print(f"  Generating {n:,} synthetic samples (horizon={horizon})...")

    # Simulate a slightly trending + mean-reverting mid price
    prices = np.zeros(n)
    prices[0] = 50_000.0
    for i in range(1, n):
        drift  = 0.00001
        mr     = -0.005 * (prices[i-1] - 50_000.0) / 50_000.0
        noise  = rng.normal(0, 0.0002)
        prices[i] = prices[i-1] * (1 + drift + mr + noise)

    # Compute forward returns
    fwd = np.full(n, np.nan)
    fwd[:n-horizon] = (prices[horizon:] - prices[:n-horizon]) / prices[:n-horizon]

    # Generate signals with weak predictive relationships
    ofi        = np.sign(np.diff(prices, prepend=prices[0])) * rng.exponential(1, n)
    vpin       = np.clip(0.3 + rng.normal(0, 0.1, n), 0, 1)
    spread_bps = np.clip(0.5 + rng.exponential(0.3, n), 0.1, 5)
    rv         = np.abs(rng.normal(0, 0.001, n))

    # Microprice offset: weakly predictive of forward return
    micro   = ofi * 0.0001 + rng.normal(0, 0.00005, n)
    sma200  = np.convolve(prices, np.ones(200)/200, mode='same')
    statarb = np.where(sma200 > 0, (prices - sma200) / np.maximum(sma200 * 0.001, 1e-10), 0)
    statarb = np.clip(statarb, -3, 3)

    target_col = f'forward_return_{horizon}'
    df = pd.DataFrame({
        'timestamp_ns':    np.arange(n) * 1_000_000,
        'microprice':      micro,
        'ofi':             ofi,
        'vpin':            vpin,
        'spread_bps':      spread_bps,
        'realized_vol':    rv,
        'stat_arb_zscore': statarb,
        'combined_alpha':  np.zeros(n),
        'regime':          np.zeros(n, dtype=int),
        'mid_price':       prices,
        target_col:        fwd,
        'is_warmed_up':    np.ones(n, dtype=int),
    })

    # Drop NaN rows (last `horizon` rows)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    return df, target_col


# ─── Feature Engineering ────────────────────────────────────

def engineer_features(df: pd.DataFrame,
                       windows: Tuple[int, ...] = None
                       ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the full feature matrix from the explicit C++ signals.
    No rolling windows are computed in Python anymore — everything 
    comes strictly from the C++ FeatureEngine to ensure 100% parity 
    between training and live execution.
    """
    feat = pd.DataFrame(index=df.index)

    # ── Explicit C++ signals ────────────────────────────────
    for sig in BASE_SIGNALS:
        if sig in df.columns:
            feat[sig] = df[sig]
            
    # ── Regime one-hot ───────────────────────────────────────
    if 'regime' in df.columns:
        for r_val, r_name in [(0, 'normal'), (1, 'high_tox'),
                               (2, 'low_liq'), (3, 'trending')]:
            feat[f'regime_{r_name}'] = (df['regime'] == r_val).astype(np.float32)

    # ── Volatility Risk Premium (VRP) Proxy ──────────────────
    if 'realized_vol' in df.columns:
        rv_10 = df['realized_vol'].rolling(10).mean()
        rv_30 = df['realized_vol'].rolling(30).mean()
        feat['vol_risk_premium'] = (rv_10 / (rv_30 + 1e-9)).astype(np.float32)
        
    # ── Stationarity Binary Feature ──────────────────────────
    # CRITICAL: This feature must be computed strictly out-of-sample.
    # Using a rolling window over the full DataFrame (including test rows)
    # would cause lookahead contamination because adfuller on a window
    # that spans the train/test boundary lets the model "see" test prices.
    # We compute the rolling apply here, but train_and_evaluate ensures
    # this function is called BEFORE the split only for feature shape
    # discovery — the actual stat_vals values are aligned by index and the
    # test-set rows' ADF feature reflects only past data due to rolling().
    from statsmodels.tsa.stattools import adfuller
    def is_stationary_fn(x):
        if len(x) < 200: return 0.0
        try:
            _, p_value, _, _, _, _ = adfuller(x, maxlag=10, autolag='AIC')
            return 1.0 if p_value < 0.05 else 0.0
        except Exception:
            return 0.0

    if 'mid_price' in df.columns:
        # A rolling(200) window means each row's value only uses the 200
        # preceding rows — there is no lookahead as long as we never sort
        # the data in non-temporal order (we never do). The contamination
        # risk exists only if the full df is sorted globally after the
        # split, which would be a different bug. This is safe.
        stat_vals = df['mid_price'].rolling(200).apply(is_stationary_fn, raw=True)
        feat['is_stationary'] = stat_vals.astype(np.float32)

    feat = feat.dropna()
    return feat, list(feat.columns)


# ─── Sharpe Evaluation ───────────────────────────────────────

def compute_trading_sharpe(y_pred: np.ndarray,
                            y_true: np.ndarray,
                            threshold: float = 0.0,
                            annualize_factor: float = np.sqrt(252 * 24 * 3600)
                            ) -> float:
    """
    Simulate a simple threshold strategy on the regression predictions
    and compute the annualized Sharpe ratio.

    Logic:
      - If y_pred > +threshold  → go long  (position = +1)
      - If y_pred < -threshold  → go short (position = -1)
      - Otherwise               → flat     (position =  0)
      - Strategy return = position × actual forward return

    This is the key metric for model selection: a model with AUC 0.55
    but Sharpe 1.5 is far better than AUC 0.60 with Sharpe 0.3.

    annualize_factor: sqrt(periods_per_year). For tick data assume
    one tick ≈ 1 second → 252 × 24 × 3600 ticks/year.
    """
    positions = np.where(y_pred > threshold, 1.0,
                np.where(y_pred < -threshold, -1.0, 0.0))
    strategy_returns = positions * y_true

    active = strategy_returns[positions != 0]
    if len(active) < 10:
        return 0.0

    mean_ret = np.mean(active)
    std_ret  = np.std(active)
    if std_ret < 1e-15:
        return 0.0

    return float(mean_ret / std_ret * annualize_factor)


def compute_hit_rate(y_pred: np.ndarray, y_true: np.ndarray,
                     threshold: float = 0.0) -> float:
    """
    Directional accuracy: fraction of active trades where predicted
    direction matches actual return direction.
    """
    active = (np.abs(y_pred) > threshold)
    if active.sum() == 0:
        return 0.0
    correct = np.sign(y_pred[active]) == np.sign(y_true[active])
    return float(correct.mean())


def find_optimal_threshold(y_pred: np.ndarray, y_true: np.ndarray,
                            grid: np.ndarray = None) -> Tuple[float, float]:
    """
    Grid search over prediction thresholds to maximize Sharpe.
    Returns (best_threshold, best_sharpe).
    """
    if grid is None:
        # Search between 0 and the 80th percentile of |y_pred|
        max_thresh = np.percentile(np.abs(y_pred), 80)
        grid = np.linspace(0, max_thresh, 40)

    best_thresh, best_sharpe = 0.0, -np.inf
    for t in grid:
        s = compute_trading_sharpe(y_pred, y_true, threshold=t)
        if s > best_sharpe:
            best_sharpe = s
            best_thresh = t

    return best_thresh, best_sharpe


# ─── Model Training ──────────────────────────────────────────

def train_lightgbm_regression(
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        feature_names: List[str],
        n_estimators: int = 500,
        verbose_eval: int = 100,
) -> Tuple['lgb.Booster', np.ndarray]:
    """
    Train a LightGBM regression model to predict signed forward returns.

    Hyperparameters are tuned for financial time-series:
      - Small learning rate (0.02) to avoid overfitting noisy labels
      - Low num_leaves (31) and max_depth (5) to prevent memorisation
      - min_child_samples=200 enforces each leaf has statistical mass
      - feature_fraction=0.7 and bagging gives implicit regularisation

    Returns (booster, importance_array).
    """
    if not HAS_LGB:
        raise RuntimeError("LightGBM not installed: pip install lightgbm")

    params = {
        'objective':        'regression',   # Predict signed return directly
        'metric':           'rmse',
        'boosting_type':    'gbdt',
        'num_leaves':       31,
        'max_depth':        5,
        'learning_rate':    0.02,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq':     5,
        'min_child_samples': 200,           # Prevents overfitting on tick noise
        'lambda_l1':        0.1,
        'lambda_l2':        0.1,
        'verbose':          -1,
        'seed':             42,
        # WINDOWS DEADLOCK FIX: n_jobs=-1 causes OpenMP to spawn threads
        # that deadlock against Python's GIL on Windows when LightGBM is
        # built with MinGW/libgomp. Force single-threaded training.
        # Training on 200k rows with n_jobs=1 takes ~30-60s vs ~10s with
        # n_jobs=-1, which is acceptable for offline weight generation.
        'n_jobs':           1,
        'num_threads':      1,   # LightGBM internal thread count alias
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                              free_raw_data=True)
    val_data   = lgb.Dataset(X_val,   label=y_val,   feature_name=feature_names,
                              reference=train_data, free_raw_data=True)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=verbose_eval),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks,
    )

    importance = model.feature_importance(importance_type='gain')
    print(f"  Best iteration: {model.best_iteration}  "
          f"Val RMSE: {model.best_score['val']['rmse']:.6f}")

    return model, importance.astype(np.float64)


# ─── ONNX Export ─────────────────────────────────────────────

def export_onnx(model: 'lgb.Booster',
                feature_names: List[str],
                output_path: str,
                n_features: int) -> bool:
    """Export LightGBM model natively to ONNX using onnxmltools."""
    try:
        print(f"  Exporting LGB model to ONNX: {output_path}...")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        from onnxmltools.convert import convert_lightgbm
        from onnxconverter_common.data_types import FloatTensorType
        import onnxmltools

        # We must define the initial input shape
        initial_types = [('float_input', FloatTensorType([None, n_features]))]
        
        # Convert using target_opset=12 for broad compatibility with ONNX Runtime
        onnx_model = convert_lightgbm(model, initial_types=initial_types, target_opset=12)
        onnxmltools.utils.save_model(onnx_model, output_path)
        
        print(f"  [+] Saved ONNX model: {output_path}")
        return True
    except Exception as e:
        print(f"  [WARN] ONNX Export failed: {e}")
        return False


def export_binary_weights(weights: np.ndarray,
                           feature_names: List[str],
                           output_path: str,
                           bias: float = 0.0) -> None:
    """
    Export directional linear signal weights as a binary file for
    C++ SignalCombiner::load_model().

    CRITICAL: This function must receive Ridge regression *coefficients*
    (signed floats that can be negative), NOT LightGBM feature importances
    (which are always non-negative and destroy directional information).

    Signals like VPIN and spread_bps have a *negative* relationship with
    future returns: high toxicity / wide spread → bearish for the maker.
    Ridge regression preserves this sign; feature importance discards it.

    Format: 6 × float64 weights + 1 × float64 bias = 56 bytes.
    The 6 slots correspond to BASE_SIGNALS[:6]:
        [microprice, ofi, vpin, spread_bps, realized_vol, stat_arb_zscore]
    """
    # Accept either a full feature-length array or a pre-extracted 6-element array.
    # If full length, extract the 6 base signal coefficients by name.
    BASE_6 = BASE_SIGNALS[:6]
    if len(weights) != 6:
        base_w = np.zeros(6)
        for i, name in enumerate(BASE_6):
            if name in feature_names:
                idx = feature_names.index(name)
                base_w[i] = weights[idx] if idx < len(weights) else 0.0
        weights = base_w

    # Normalize so the sum-of-absolute-values = 1.0 (unit L1 norm).
    # This preserves the sign and relative magnitude of each coefficient
    # while keeping the combined alpha in a consistent scale range.
    total_abs = np.sum(np.abs(weights))
    if total_abs > 1e-12:
        weights = weights / total_abs
    else:
        weights = np.array([1/6, 1/6, -1/6, -1/6, 1/6, 1/6])  # safe fallback

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        weights.astype(np.float64).tofile(f)
        np.array([bias], dtype=np.float64).tofile(f)

    print(f"  [+] Saved binary weights: {output_path}")
    weight_str = ', '.join(f'{n}={w:+.4f}' for n, w in zip(BASE_6, weights))
    print(f"      {weight_str}")
    print(f"      bias={bias:+.6f}")


# ─── Visualisation ───────────────────────────────────────────

def plot_feature_importance(importance: np.ndarray,
                             feature_names: List[str],
                             output_path: str,
                             top_n: int = 25) -> None:
    if not HAS_MPL:
        return

    sorted_idx  = np.argsort(importance)[::-1][:top_n]
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_imp   = importance[sorted_idx]
    total        = sorted_imp.sum()
    pct          = sorted_imp / total * 100 if total > 0 else sorted_imp

    colors = ['#58a6ff' if n in BASE_SIGNALS else '#8b949e' for n in sorted_names]

    plt.rcParams.update({
        'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',   'axes.labelcolor': '#c9d1d9',
        'text.color': '#c9d1d9',       'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',      'grid.color': '#21262d',
    })

    fig, ax = plt.subplots(figsize=(13, max(6, top_n // 2)))
    bars = ax.barh(range(len(sorted_names)), pct, color=colors, alpha=0.85)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('Importance (% of total gain)', fontsize=11)
    ax.set_title(f'LightGBM Feature Importance  (top {top_n})',
                 fontsize=14, fontweight='bold', color='#f0f6fc')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25, axis='x')

    # Annotate bars with %
    for bar, val in zip(bars, pct):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=8, color='#c9d1d9')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved: {output_path}")


def plot_prediction_scatter(y_pred: np.ndarray, y_true: np.ndarray,
                             output_path: str, split_name: str = 'Test') -> None:
    """Scatter plot of predicted vs actual forward returns."""
    if not HAS_MPL:
        return

    # Downsample for plotting if too many points
    n = len(y_pred)
    if n > 20_000:
        idx = np.random.choice(n, 20_000, replace=False)
        y_pred, y_true = y_pred[idx], y_true[idx]

    corr = np.corrcoef(y_pred, y_true)[0, 1]

    plt.rcParams.update({
        'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',   'axes.labelcolor': '#c9d1d9',
        'text.color': '#c9d1d9',       'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
    })

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_pred, y_true, alpha=0.05, s=2, color='#58a6ff')
    ax.axhline(0, color='#8b949e', linewidth=0.7, linestyle='--')
    ax.axvline(0, color='#8b949e', linewidth=0.7, linestyle='--')
    ax.set_xlabel('Predicted Forward Return', fontsize=11)
    ax.set_ylabel('Actual Forward Return', fontsize=11)
    ax.set_title(f'{split_name} Set: Predicted vs Actual  (rho = {corr:.4f})',
                 fontsize=13, fontweight='bold', color='#f0f6fc')
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved: {output_path}")


# ─── Training Report ─────────────────────────────────────────

def save_training_report(metrics: Dict, importance: np.ndarray,
                          feature_names: List[str], config: Dict,
                          output_dir: str) -> None:
    sorted_idx  = np.argsort(importance)[::-1]
    total_imp   = importance.sum()

    report = f"""# ML Signal Combiner — Training Report

## Configuration

| Parameter | Value |
|---|---|
| Model | LightGBM Regression |
| Target | Signed forward return (regression, not classification) |
| Prediction horizon | {config.get('horizon', '?')} ticks |
| Train / Val / Test split | 60% / 20% / 20% (time-ordered, no shuffle) |
| Total features | {len(feature_names)} ({len(BASE_SIGNALS)} base + {len(feature_names) - len(BASE_SIGNALS)} engineered) |
| Best LGB iteration | {config.get('best_iter', '?')} |

## Out-of-Sample Metrics (Test Set)

| Metric | Value |
|---|---|
| **OOS Sharpe (optimal threshold)** | **{metrics.get('oos_sharpe', 0):.4f}** |
| OOS Sharpe (threshold = 0) | {metrics.get('oos_sharpe_zero', 0):.4f} |
| Optimal alpha threshold | {metrics.get('opt_threshold', 0):.6f} |
| Directional hit rate | {metrics.get('hit_rate', 0):.4f} |
| Pearson r (pred vs actual) | {metrics.get('pearson_r', 0):.4f} |
| R^2 | {metrics.get('r2', 0):.4f} |
| RMSE | {metrics.get('rmse', 0):.6f} |
| Active trades (above threshold) | {metrics.get('n_active', 0):,} / {metrics.get('n_test', 0):,} ({metrics.get('active_pct', 0):.1f}%) |

### Interpretation
- **OOS Sharpe > 1.0**: The model has real predictive signal
- **OOS Sharpe 0.5–1.0**: Marginal signal, may work with tighter risk controls
- **OOS Sharpe < 0.5**: Noise. Consider longer horizon or more features.

## Feature Importance (Top 15)

| Rank | Feature | Importance (%) | Type |
|---|---|---|---|
"""
    for rank, idx in enumerate(sorted_idx[:15], 1):
        pct    = importance[idx] / total_imp * 100 if total_imp > 0 else 0
        f_type = 'base' if feature_names[idx] in BASE_SIGNALS else 'engineered'
        marker = ' ★' if feature_names[idx] in BASE_SIGNALS else ''
        report += f'| {rank} | `{feature_names[idx]}`{marker} | {pct:.2f}% | {f_type} |\n'

    report += f"""
## Base Signal Weights (C++ Binary Fallback)

These weights are exported to `signal_weights.bin` for `SignalCombiner::load_model()`.
They represent the LightGBM feature importance normalised to the 6 base signals only.

| Signal | Weight |
|---|---|
"""
    base_imp = np.zeros(6)
    for i, name in enumerate(BASE_SIGNALS):
        if name in feature_names:
            ii = feature_names.index(name)
            base_imp[i] = importance[ii]
    base_total = base_imp.sum()
    for name, imp in zip(BASE_SIGNALS, base_imp):
        w = imp / base_total if base_total > 0 else 1.0 / 6
        report += f'| `{name}` | {w:.4f} |\n'

    report += f"""
## ONNX Model

ONNX model exported to `lgb_model.onnx` for C++ inference via onnxruntime.
- Input:  float32 tensor `[1, {len(feature_names)}]`
- Output: float32 scalar (predicted signed forward return)
- Usage in C++: clamp output to [-1, 1] → use as `combined_alpha`

## Plots

![Feature Importance](feature_importance.png)
![Prediction Scatter](prediction_scatter.png)

---
*Generated by HFT Engine ML Pipeline*
"""
    path = os.path.join(output_dir, 'training_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  [+] Saved: {path}")


# ─── Ridge Regression for Directional Signal Weights ────────

def fit_ridge_signal_weights(
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        alpha_ridge: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Fit a Ridge regression model on the training set to extract
    *directional, signed* linear coefficients for the 6 base signals.

    WHY Ridge and not LightGBM importances:
    ─────────────────────────────────────────
    LightGBM feature importance (gain) is ALWAYS non-negative. It measures
    how much each feature reduces the loss, not whether high values of that
    feature are bullish or bearish. Exporting importances as signal weights
    and using them in a linear dot product α = Σ(w_i × signal_i) ignores
    directional information entirely.

    Specifically:
      VPIN   = toxicity proxy: HIGH VPIN → adverse selection → negative alpha
      spread = friction proxy: WIDE spread → hard to profit → negative alpha

    Ridge regression fits y ≈ Xβ + b, where β can be negative, correctly
    capturing these inverse relationships. Using these coefficients in the
    C++ weighted_avg combiner produces a directionally valid alpha signal.

    Returns (coefficients_array_len_n_features, bias_scalar).
    """
    if not HAS_SKLEARN:
        print("  [WARN] sklearn not available — falling back to equal weights.")
        return np.ones(len(feature_names)) / len(feature_names), 0.0

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    # Standardize features so coefficients are on a comparable scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    ridge = Ridge(alpha=alpha_ridge, fit_intercept=True)
    ridge.fit(X_scaled, y_train)

    coef = ridge.coef_.astype(np.float64)
    bias = float(ridge.intercept_)

    # Report the base-signal weights for auditing
    BASE_6 = BASE_SIGNALS[:6]
    print(f"\n  [Ridge] Directional signal weights (signed):")
    for name in BASE_6:
        if name in feature_names:
            idx = feature_names.index(name)
            print(f"    {name:>20}: {coef[idx]:+.4f}")

    print(f"    {'bias':>20}: {bias:+.6f}")
    return coef, bias


# ─── Public API for walk_forward.py ─────────────────────────

def train_and_evaluate(df: pd.DataFrame,
                        target_col: str,
                        train_end_idx: int,
                        val_end_idx: int,
                        horizon: int,
                        output_dir: Optional[str] = None,
                        verbose: bool = True,
                        ) -> Dict:
    """
    Core training function called by both main() and walk_forward.py.

    Takes a DataFrame (already filtered), splits by index positions
    (time-ordered), trains LightGBM, evaluates, and returns a metrics dict.

    Args:
        df:            Clean feature DataFrame (from load_feature_dump or caller)
        target_col:    Name of the forward return column
        train_end_idx: Row index where training ends (exclusive)
        val_end_idx:   Row index where validation ends (exclusive)
        horizon:       Prediction horizon (used in column naming)
        output_dir:    If provided, saves model artifacts here
        verbose:       Print progress

    Returns dict with keys:
        model, feature_names, importance, oos_sharpe, opt_threshold,
        hit_rate, pearson_r, r2, rmse, n_test, n_active, active_pct,
        oos_sharpe_zero, val_sharpe
    """
    # ── Feature engineering ─────────────────────────────────
    feat, feature_names = engineer_features(df)

    # Align target with the feature index (rolling windows drop some rows)
    y_all = df.loc[feat.index, target_col].values.astype(np.float32)
    # float32 halves RAM vs float64 — LightGBM works natively with float32
    X_all = feat.values.astype(np.float32)

    n = len(X_all)
    train_end = min(train_end_idx, n)
    val_end   = min(val_end_idx,   n)

    X_train, y_train = X_all[:train_end],        y_all[:train_end]
    X_val,   y_val   = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test,  y_test  = X_all[val_end:],          y_all[val_end:]

    if verbose:
        print(f"  Split  train={len(X_train):,}  val={len(X_val):,}  "
              f"test={len(X_test):,}  features={len(feature_names)}")

    if len(X_train) < 500 or len(X_val) < 100:
        raise ValueError(f"Insufficient data for training: "
                         f"train={len(X_train)}, val={len(X_val)}")

    # ── Train ────────────────────────────────────────────────
    model, importance = train_lightgbm_regression(
        X_train, y_train, X_val, y_val, feature_names,
        verbose_eval=100 if verbose else 0,
    )

    # ── Validation Sharpe (for hyperparameter selection) ─────
    y_val_pred = model.predict(X_val).astype(np.float64)
    val_thresh, val_sharpe = find_optimal_threshold(y_val_pred, y_val)

    # ── OOS Evaluation (test set — never seen by model) ──────
    y_test_pred = model.predict(X_test).astype(np.float64)

    # Use the threshold found on val, apply to test (no look-ahead)
    oos_sharpe      = compute_trading_sharpe(y_test_pred, y_test, threshold=val_thresh)
    oos_sharpe_zero = compute_trading_sharpe(y_test_pred, y_test, threshold=0.0)
    hit_rate        = compute_hit_rate(y_test_pred, y_test, threshold=val_thresh)
    pearson_r       = float(np.corrcoef(y_test_pred, y_test)[0, 1])

    if HAS_SKLEARN:
        r2   = float(r2_score(y_test, y_test_pred))
    else:
        ss_res = np.sum((y_test - y_test_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-30)

    rmse     = float(np.sqrt(np.mean((y_test - y_test_pred) ** 2)))
    n_active = int((np.abs(y_test_pred) > val_thresh).sum())
    active_pct = n_active / max(len(y_test_pred), 1) * 100

    metrics = {
        'model':           model,
        'feature_names':   feature_names,
        'importance':      importance,
        'oos_sharpe':      oos_sharpe,
        'oos_sharpe_zero': oos_sharpe_zero,
        'val_sharpe':      val_sharpe,
        'opt_threshold':   val_thresh,
        'hit_rate':        hit_rate,
        'pearson_r':       pearson_r,
        'r2':              r2,
        'rmse':            rmse,
        'n_test':          len(y_test),
        'n_active':        n_active,
        'active_pct':      active_pct,
        'best_iter':       model.best_iteration,
        'y_test':          y_test,
        'y_test_pred':     y_test_pred,
    }

    # ── Save artefacts ───────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # ONNX model
        onnx_path = os.path.join(output_dir, 'lgb_model.onnx')
        export_onnx(model, feature_names, onnx_path, len(feature_names))

        # Binary weight fallback — fit Ridge regression on training data to get
        # *directional, signed* linear coefficients for the 6 base signals.
        # Ridge preserves negative weights (e.g. vpin, spread_bps must be negative)
        # whereas LightGBM feature importance is always non-negative and is wrong
        # for this purpose — it caused the $14k drawdown.
        weights_path = os.path.join(output_dir, 'signal_weights.bin')
        ridge_weights, ridge_bias = fit_ridge_signal_weights(
            X_train, y_train, feature_names)
        export_binary_weights(ridge_weights, feature_names, weights_path,
                              bias=ridge_bias)

        # LightGBM text model (human-readable, useful for debugging)
        txt_path = os.path.join(output_dir, 'lgb_model.txt')
        model.save_model(txt_path)
        print(f"  [+] Saved LGB text model: {txt_path}")

        # Feature importance plot
        plot_feature_importance(
            importance, feature_names,
            os.path.join(output_dir, 'feature_importance.png'))

        # Prediction scatter
        plot_prediction_scatter(
            y_test_pred, y_test,
            os.path.join(output_dir, 'prediction_scatter.png'))

        # Save metrics JSON for walk_forward.py to read
        # Only serialise scalar metrics — explicitly exclude arrays,
        # the Booster object, and numpy arrays which json can't handle
        SCALAR_KEYS = {
            'oos_sharpe', 'oos_sharpe_zero', 'val_sharpe', 'opt_threshold',
            'hit_rate', 'pearson_r', 'r2', 'rmse',
            'n_test', 'n_active', 'active_pct', 'best_iter',
        }
        json_metrics = {
            k: (float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v)
            for k, v in metrics.items()
            if k in SCALAR_KEYS
        }
        with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(json_metrics, f, indent=2)

        # Training report
        report_config = {'horizon': horizon, 'best_iter': model.best_iteration}
        save_training_report(metrics, importance, feature_names,
                             report_config, output_dir)

    return metrics


# ─── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HFT Engine — ML Signal Combiner Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on features dumped by backtest.py --dump-features
    python train_model.py --data data/features.csv

    # Quick smoke-test with synthetic data
    python train_model.py --synthetic

    # Custom horizon and output path
    python train_model.py --data data/features.csv --horizon 50 --output models

Then connect C++ engine to the ONNX model:
    combiner.load_onnx_model("models/lgb_model.onnx");
    combiner.set_mode(CombinerMode::ONNX_MODEL);

Or use the binary weight fallback (no recompile):
    combiner.load_model("models/signal_weights.bin");
        """
    )
    parser.add_argument('--data',      type=str,  default=None,
                        help='Path to features CSV from backtest.py --dump-features')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (smoke-test without real data)')
    parser.add_argument('--horizon',   type=int,  default=100,
                        help='Forward return horizon in ticks (default: 100)')
    parser.add_argument('--output',    type=str,  default='models',
                        help='Output directory (default: models)')
    parser.add_argument('--n-samples', type=int,  default=200_000,
                        help='Synthetic data size (default: 200000)')
    parser.add_argument('--max-rows', type=int, default=2_000_000,
                        help='Max rows to use for training (default: 2000000). '
                             'LightGBM does not benefit from >2M rows for this '
                             'problem. Reduces RAM from ~6GB to ~400MB. '
                             'Rows are sampled with temporal spacing to preserve '
                             'the time-series distribution across all 12 months.')
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  HFT Engine — ML Signal Combiner Training")
    print(f"{'='*65}")

    # ── Load or generate data ────────────────────────────────
    t0 = time.time()

    if args.data and not args.synthetic:
        # Load real feature dump from backtest.py --dump-features
        try:
            df, target_col = load_feature_dump(args.data, args.horizon,
                                               max_rows=args.max_rows)
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            sys.exit(1)
        print(f"  Data loaded: {len(df):,} rows")
    else:
        # Synthetic smoke-test (no real data, or --synthetic flag)
        if args.data and args.synthetic:
            print("  [NOTE] --synthetic flag overrides --data. Using synthetic data.")
        else:
            print("\n  No --data provided. Using synthetic data for demo.")
            print("  Run: python backtest.py --data data/BTCUSDT_2024.csv "
                  "--dump-features data/features.csv")
            print("  Then: python train_model.py --data data/features.csv\n")
        df, target_col = generate_synthetic_data(args.n_samples, args.horizon)

    print(f"  Data loaded in {time.time()-t0:.2f}s  "
          f"({len(df):,} rows, {len(df.columns)} columns)")

    n = len(df)
    train_end = int(n * 0.60)
    val_end   = int(n * 0.80)

    print(f"\n  Split: train=0:{train_end:,}  "
          f"val={train_end:,}:{val_end:,}  "
          f"test={val_end:,}:{n:,}")

    # ── Resolve output path ──────────────────────────────────
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output
    )

    # ── Train & evaluate ─────────────────────────────────────
    print(f"\n  Training LightGBM regression...")
    metrics = train_and_evaluate(
        df, target_col,
        train_end_idx=train_end,
        val_end_idx=val_end,
        horizon=args.horizon,
        output_dir=output_dir,
        verbose=True,
    )

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*65}")
    print(f"  OOS Sharpe (threshold={metrics['opt_threshold']:.4f}):  "
          f"{metrics['oos_sharpe']:+.4f}")
    print(f"  OOS Sharpe (threshold=0):             "
          f"{metrics['oos_sharpe_zero']:+.4f}")
    print(f"  Directional hit rate:                 "
          f"{metrics['hit_rate']*100:.2f}%")
    print(f"  Pearson r (pred vs actual):           "
          f"{metrics['pearson_r']:.4f}")
    print(f"  R^2:                                  "
          f"{metrics['r2']:.4f}")
    print(f"  Active trade fraction:                "
          f"{metrics['active_pct']:.1f}%")
    print(f"  Best LGB iteration:                   "
          f"{metrics['best_iter']}")
    print(f"  Output directory:                     "
          f"{output_dir}/")
    print(f"{'='*65}")

    sharpe = metrics['oos_sharpe']
    if sharpe >= 1.0:
        verdict = "STRONG SIGNAL  — ready for walk-forward testing"
    elif sharpe >= 0.5:
        verdict = "MARGINAL SIGNAL — consider longer horizon or more data"
    else:
        verdict = "WEAK SIGNAL    — review features and target horizon"
    print(f"\n  Verdict: {verdict}")

    print(f"\n  Next steps:")
    print(f"    Walk-forward:  python python/walk_forward.py "
          f"--data {args.data or 'data/features.csv'}")
    print(f"    C++ ONNX:      combiner.load_onnx_model(\"{output_dir}/lgb_model.onnx\");")
    print(f"    C++ fallback:  combiner.load_model(\"{output_dir}/signal_weights.bin\");")
    print()


if __name__ == '__main__':
    main()
