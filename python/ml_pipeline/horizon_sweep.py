#!/usr/bin/env python3
"""
horizon_sweep.py — Target Horizon Refinement

Sweeps over a range of prediction horizons to determine the optimal target
horizon for our HFT model. In HFT, predicting too far out introduces
random walk noise (Brownian motion), while predicting too close means the
price change is too small to overcome the bid-ask spread and latency.

Usage:
    python horizon_sweep.py --synthetic
    python horizon_sweep.py --data data/features.csv
"""

import sys
import os
import argparse
import time
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Try to import from train_model if in the same directory
try:
    from train_model import BASE_SIGNALS, compute_trading_sharpe, export_binary_weights
except ImportError:
    BASE_SIGNALS = [
        'microprice', 'ofi', 'vpin',
        'spread_bps', 'realized_vol', 'stat_arb_zscore',
        'obi', 'trade_imbalance', 'hawkes_intensity',
        'cvd', 'hurst_exponent'
    ]
    def compute_trading_sharpe(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.0) -> float:
        positions = np.where(y_pred > threshold, 1.0,
                    np.where(y_pred < -threshold, -1.0, 0.0))
        strategy_returns = positions * y_true
        active = strategy_returns[positions != 0]
        if len(active) < 10: return 0.0
        mean_ret = np.mean(active)
        std_ret  = np.std(active)
        if std_ret < 1e-15: return 0.0
        # Fix: Previously we scaled by sqrt(252*24*3600) assuming tick data was second data.
        # This massively inflated the Sharpe ratio to mathematically impossible levels.
        # We will now return the raw per-trade/tick Sharpe to reflect the true microstructure edge.
        return float(mean_ret / std_ret)
    def export_binary_weights(weights: np.ndarray, filepath: str, bias: float = 0.0):
        import struct
        assert len(weights) == 11, "Must export exactly 11 weights"
        with open(filepath, 'wb') as f:
            for w in weights:
                f.write(struct.pack('d', w))
            f.write(struct.pack('d', bias))
        print(f"  [+] Exported 96-byte model to {filepath}")


def generate_synthetic_data(n: int = 50_000) -> pd.DataFrame:
    """Generates base features and a mid_price."""
    rng = np.random.RandomState(42)
    print(f"  Generating {n:,} synthetic samples...")
    df = pd.DataFrame({'timestamp_ns': np.arange(n) * 1_000_000})
    for sig in BASE_SIGNALS:
        df[sig] = rng.normal(0, 1, n)
    
    # We want a mid_price where returns are partially predictable by features
    # but noise dominates over longer horizons.
    returns = df['ofi'].values * 0.00005 + df['microprice'].values * 0.00002 + rng.normal(0, 0.0005, n)
    df['mid_price'] = 50000.0 * np.exp(np.cumsum(returns))
    df['is_warmed_up'] = 1
    return df

def load_feature_dump(filepath: str) -> pd.DataFrame:
    print(f"  Loading: {filepath}")
    df = pd.read_csv(filepath)
    df = df[df['is_warmed_up'] == 1]
    return df

def evaluate_horizon(df: pd.DataFrame, horizon: int) -> Tuple[float, float]:
    """Runs a 5-fold TimeSeries Walk-Forward validation for the given horizon."""
    target_col = f'forward_return_{horizon}'
    
    if target_col not in df.columns:
        # Compute forward returns dynamically
        df[target_col] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1.0
        
    df_clean = df.dropna(subset=[target_col])
    if len(df_clean) < 1000:
        return 0.0, 0.0
        
    # Only train on features present in the dataset
    available_features = [f for f in BASE_SIGNALS if f in df_clean.columns]
    
    X = df_clean[available_features].values.astype(np.float64)
    y = df_clean[target_col].values.astype(np.float64)
    
    tscv = TimeSeriesSplit(n_splits=5)
    sharpes = []
    r2s = []
    
    for train_idx, test_idx in tscv.split(X):
        # Purge the last `horizon` samples from the training set to prevent lookahead overlap
        train_idx = train_idx[:-horizon]
        if len(train_idx) < 100:
            continue
            
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = Ridge(alpha=100.0, fit_intercept=True)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        sharpe = compute_trading_sharpe(y_pred, y_test, threshold=0.0)
        r2 = r2_score(y_test, y_pred)
        
        sharpes.append(sharpe)
        r2s.append(r2)
        
    # Re-train on all data to extract coefficients
    best_model = Ridge(alpha=100.0, fit_intercept=True).fit(X, y)
    
    # Map coefficients back to the 11-feature array
    weights_11 = np.zeros(11, dtype=np.float64)
    for i, feature in enumerate(available_features):
        idx = BASE_SIGNALS.index(feature)
        weights_11[idx] = best_model.coef_[i]
        
    # Scale coefficients
    total_abs = np.sum(np.abs(weights_11))
    bias = best_model.intercept_
    if total_abs > 1e-12:
        weights_11 = weights_11 * (0.33 / total_abs)
        bias = bias * (0.33 / total_abs)
        
    return np.mean(sharpes), np.mean(r2s), weights_11, bias

def plot_horizon_sweep(horizons: List[int], sharpes: List[float], r2s: List[float], output_path: str):
    plt.rcParams.update({
        'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',   'axes.labelcolor': '#c9d1d9',
        'text.color': '#c9d1d9',       'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',      'grid.color': '#21262d',
    })
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Sharpe
    ax1.plot(horizons, sharpes, marker='o', color='#58a6ff', linewidth=2)
    ax1.set_ylabel('OOS Sharpe Ratio', fontsize=12)
    ax1.set_title('Target Horizon Optimization Sweep', fontsize=14, fontweight='bold', color='#f0f6fc')
    ax1.grid(True, alpha=0.25)
    
    # Highlight best Sharpe
    best_idx = np.argmax(sharpes)
    ax1.scatter(horizons[best_idx], sharpes[best_idx], color='#3fb950', s=100, zorder=5, label=f'Best Sharpe ({horizons[best_idx]} ticks)')
    ax1.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d')
    
    # Plot R2
    ax2.plot(horizons, r2s, marker='s', color='#d2a8ff', linewidth=2)
    ax2.set_ylabel('Out-of-Sample R²', fontsize=12)
    ax2.set_xlabel('Prediction Horizon (ticks)', fontsize=12)
    ax2.grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [+] Saved horizon sweep plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', type=str, default='models')
    args = parser.parse_args()

    t0 = time.time()
    if args.synthetic or not args.data:
        df = generate_synthetic_data(n=100_000)
    else:
        df = load_feature_dump(args.data)
        
    horizons = [10, 25, 50, 100, 250, 500, 1000]
    sharpes = []
    r2s = []
    best_sharpe = -999.0
    best_weights = None
    best_bias = 0.0
    
    print("\n  [Target Horizon Sweep]")
    for h in horizons:
        print(f"    Evaluating horizon: {h:4d} ticks...", end='', flush=True)
        s, r2, w, b = evaluate_horizon(df, h)
        print(f" Sharpe = {s:+.2f} | R² = {r2:+.4f}")
        sharpes.append(s)
        r2s.append(r2)
        
        if s > best_sharpe:
            best_sharpe = s
            best_weights = w
            best_bias = b
            
    if best_weights is not None:
        print("\n  Final Optimal Coefficients (for best horizon):")
        for name, w in zip(BASE_SIGNALS, best_weights):
            print(f"    {name:>20}: {w:+.5f}")
        print(f"    {'bias':>20}: {best_bias:+.6f}")
        
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    if best_weights is not None:
        bin_path = os.path.join(output_dir, 'signal_weights.bin')
        export_binary_weights(best_weights, bin_path, best_bias)
        
    plot_path = os.path.join(output_dir, 'horizon_sweep.png')
    
    plot_horizon_sweep(horizons, sharpes, r2s, plot_path)
    print(f"\n  Done in {time.time()-t0:.2f}s")


if __name__ == '__main__':
    main()
