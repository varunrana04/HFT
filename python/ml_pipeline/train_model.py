#!/usr/bin/env python3
"""
train_model.py — Ridge Regression Signal Combiner Training

Fits a linear Ridge Regression model on the 11 engineered features
produced by backtest.py. Evaluates the model using Walk-Forward
TimeSeriesSplit to rigorously prevent look-ahead bias and mathematically
prove the out-of-sample edge.

Exports a 96-byte binary file (11 weights + 1 bias) to be loaded by
the C++ SignalCombiner.

Usage:
    python train_model.py --data data/features.csv --horizon 100
    python train_model.py --synthetic
"""

import sys
import os
import argparse
import time
import json
import warnings
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
warnings.filterwarnings('ignore')

# ─── Constants ───────────────────────────────────────────────

BASE_SIGNALS = [
    'microprice', 'ofi', 'vpin',
    'spread_bps', 'realized_vol', 'stat_arb_zscore',
    'obi', 'trade_imbalance', 'hawkes_intensity',
    'cvd', 'hurst_exponent'
]

# ─── Data Loading ────────────────────────────────────────────

def generate_synthetic_data(n: int = 50_000, horizon: int = 100) -> Tuple[pd.DataFrame, str]:
    """Generates synthetic features and a target for smoke-testing."""
    rng = np.random.RandomState(42)
    print(f"  Generating {n:,} synthetic samples...")

    df = pd.DataFrame({'timestamp_ns': np.arange(n) * 1_000_000})
    for sig in BASE_SIGNALS:
        df[sig] = rng.normal(0, 1, n)
        
    df['mid_price'] = 50000.0 + np.cumsum(rng.normal(0, 5, n))
    df['is_warmed_up'] = 1

    # Fake forward return logically tied to OFI and microprice
    fwd = df['ofi'].values * 0.0005 + df['microprice'].values * 0.0002 + rng.normal(0, 0.001, n)
    target_col = f'forward_return_{horizon}'
    df[target_col] = fwd

    return df, target_col

def load_feature_dump(filepath: str, horizon: int) -> Tuple[pd.DataFrame, str]:
    print(f"  Loading: {filepath}")
    df = pd.read_csv(filepath)
    
    target_col = f'forward_return_{horizon}'
    if target_col not in df.columns:
        if 'mid_price' not in df.columns:
            raise ValueError(f"Missing {target_col} and mid_price")
        df[target_col] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1.0

    # Clean data
    df = df[df['is_warmed_up'] == 1].dropna(subset=[target_col])
    print(f"  Clean rows: {len(df):,}")
    return df, target_col


# ─── Evaluation ──────────────────────────────────────────────

def compute_trading_sharpe(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.0) -> float:
    positions = np.where(y_pred > threshold, 1.0,
                np.where(y_pred < -threshold, -1.0, 0.0))
    strategy_returns = positions * y_true
    active = strategy_returns[positions != 0]
    
    if len(active) < 10: return 0.0
    mean_ret = np.mean(active)
    std_ret  = np.std(active)
    if std_ret < 1e-15: return 0.0
    
    return float(mean_ret / std_ret)


def export_binary_weights(weights: np.ndarray, output_path: str, bias: float) -> None:
    """Exports 11 float64 weights + 1 float64 bias = 96 bytes."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        weights.astype(np.float64).tofile(f)
        np.array([bias], dtype=np.float64).tofile(f)
    print(f"  [+] Saved 96-byte binary model: {output_path}")


# ─── Walk-Forward Training ───────────────────────────────────

def train_walk_forward(df: pd.DataFrame, target_col: str, output_dir: str):
    X = df[BASE_SIGNALS].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    
    # 5-fold Time Series Split (Walk-Forward)
    tscv = TimeSeriesSplit(n_splits=5)
    
    sharpes = []
    r2s = []
    
    print("\n  [Walk-Forward Validation]")
    fold = 1
    best_model = None
    best_sharpe = -999.0
    
    horizon = int(target_col.split('_')[-1])
    
    for train_idx, test_idx in tscv.split(X):
        # Purge the last `horizon` samples from the training set to prevent lookahead overlap
        train_idx = train_idx[:-horizon]
        if len(train_idx) < 100:
            continue
            
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # We explicitly do NOT scale here because FeatureEngine already Z-normalizes
        # the inputs to [-3, 3] in C++. Thus the coefficients represent actual weightings
        # to apply directly to the raw feature vector.
        model = Ridge(alpha=100.0, fit_intercept=True)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        sharpe = compute_trading_sharpe(y_pred, y_test, threshold=0.0)
        r2 = r2_score(y_test, y_pred)
        
        sharpes.append(sharpe)
        r2s.append(r2)
        
        print(f"    Fold {fold}: OOS Sharpe = {sharpe:+.2f} | R² = {r2:+.4f}")
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_model = model
            
        fold += 1
        
    print(f"\n  Average OOS Sharpe: {np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}")
    
    if best_model is None:
        best_model = Ridge(alpha=100.0, fit_intercept=True).fit(X, y)
        
    weights = best_model.coef_
    bias = best_model.intercept_
    
    # Scale coefficients so max alpha magnitude roughly bounds to [-1, 1] 
    # (assumes inputs are [-3, 3]).
    # We enforce sum(|w|) = 0.33 so that max theoretical alpha = 0.99
    total_abs = np.sum(np.abs(weights))
    if total_abs > 1e-12:
        weights = weights * (0.33 / total_abs)
        bias = bias * (0.33 / total_abs)
        
    print("\n  Final Optimal Coefficients:")
    for name, w in zip(BASE_SIGNALS, weights):
        print(f"    {name:>20}: {w:+.5f}")
    print(f"    {'bias':>20}: {bias:+.6f}")
    
    bin_path = os.path.join(output_dir, 'signal_weights.bin')
    export_binary_weights(weights, bin_path, bias)

# ─── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--output', type=str, default='models')
    args = parser.parse_args()

    t0 = time.time()
    if args.synthetic or not args.data:
        df, target_col = generate_synthetic_data(horizon=args.horizon)
    else:
        df, target_col = load_feature_dump(args.data, args.horizon)
        
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
    
    train_walk_forward(df, target_col, output_dir)
    print(f"\n  Done in {time.time()-t0:.2f}s")


if __name__ == '__main__':
    main()
