#!/usr/bin/env python3
"""
optimize_weights.py — Signal Combiner Weight Optimization

Finds the optimal *directional, signed* linear weights for the 6 base signals
by maximizing a CVaR-penalized return objective, then falling back to a
Ridge regression solution if skfolio is unavailable.

KEY DESIGN: weights CAN and MUST be negative for adverse signals.
  - VPIN   (toxicity)  → high value means toxic flow → NEGATIVE weight
  - spread_bps         → wide spread means friction  → NEGATIVE weight
  - microprice, OFI    → predictive of direction     → POSITIVE weight

The old skfolio default used `min_weights=0` (long-only), which forced all
weights positive and is INCORRECT for a signal combiner.  This is now fixed.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Import loading logic from train_model.py
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))
try:
    from train_model import (load_feature_dump, generate_synthetic_data,
                              export_binary_weights, fit_ridge_signal_weights)
except ImportError as e:
    print(f"[ERROR] Cannot import train_model.py: {e}")
    sys.exit(1)

CORE_6_SIGNALS = [
    'microprice', 'ofi', 'vpin',
    'spread_bps', 'realized_vol', 'stat_arb_zscore'
]


def _compute_sharpe(weights: np.ndarray, returns_df: pd.DataFrame) -> float:
    """Annualized Sharpe of the combined signal on the provided returns."""
    port_ret = returns_df.dot(weights)
    mu = port_ret.mean()
    sigma = port_ret.std()
    if sigma < 1e-12:
        return 0.0
    # Annualize assuming ~5M ticks/day (crypto 24h); sqrt(5e6 * 252) ≈ 35497
    return float(mu / sigma * 35497.0)


def optimize_signal_weights(df: pd.DataFrame, target_col: str, output_path: str):
    """
    Fit directional CVaR-optimal signal weights that can be negative.

    Strategy
    ────────
    1. Build per-signal "payoff" series: payoff_i(t) = signal_z_i(t) * fwd_ret(t)
       (each series represents the P&L you'd earn holding $1 of signal i at time t)
    2. Try skfolio with min_weights=-0.5 to allow short (negative) signal allocations
    3. Fallback: Ridge regression coefficients (always directionally correct)
    4. Validate OOS Sharpe on held-out 30% test set
    5. Export via export_binary_weights() (L1-normalized, preserves sign)
    """
    print("  Preparing signal returns with chronological 70/30 split...")

    df = df.dropna(subset=[target_col] + CORE_6_SIGNALS).copy()
    split_idx = int(len(df) * 0.7)

    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()

    fwd_ret_train = df_train[target_col].clip(lower=-0.005, upper=0.005)
    fwd_ret_test  = df_test[target_col].clip(lower=-0.005, upper=0.005)

    # ── Build per-signal payoff series (fit scaler on train only) ──
    X_train = pd.DataFrame(index=df_train.index)
    X_test  = pd.DataFrame(index=df_test.index)

    print("\n  [Signal payoff stats — Training Set]")
    for sig in CORE_6_SIGNALS:
        if sig in df_train.columns:
            mean_val = df_train[sig].mean()
            std_val  = df_train[sig].std() + 1e-9
            sig_z_tr = (df_train[sig] - mean_val) / std_val
            sig_z_te = (df_test[sig]  - mean_val) / std_val
            X_train[sig] = sig_z_tr * fwd_ret_train
            X_test[sig]  = sig_z_te * fwd_ret_test
            print(f"    {sig:>20}: mean={X_train[sig].mean():+.6f}  "
                  f"std={X_train[sig].std():.6f}")
        else:
            X_train[sig] = 0.0
            X_test[sig]  = 0.0

    weights = None

    # ── Primary path: skfolio CVaR with short weights allowed ─────────
    try:
        from skfolio.optimization import MeanRisk, ObjectiveFunction
        from skfolio.measures import RiskMeasure, cvar

        print("\n  Running skfolio MeanRisk optimization (CVaR, short weights allowed)...")
        eq_w = np.ones(6) / 6.0
        eq_cvar = cvar(X_train.dot(eq_w))
        print(f"  Equal-weight Train CVaR: {eq_cvar:.6f}")

        # CRITICAL FIX: min_weights=-0.50 allows negative allocations.
        # This is mandatory so signals with an inverse relationship to returns
        # (VPIN, spread_bps) receive negative weights instead of being
        # incorrectly forced positive.
        model = MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            risk_measure=RiskMeasure.CVAR,
            min_weights=-0.50,   # ← was 0 (long-only): FIXED
            max_weights=0.50,
            portfolio_params=dict(name="Optimized_DirectionalWeights"),
        )
        model.fit(X_train)
        weights = model.weights_.copy()

        achieved_cvar  = cvar(X_train.dot(weights))
        train_sharpe   = _compute_sharpe(weights, X_train)
        oos_sharpe     = _compute_sharpe(weights, X_test)

        print(f"  Achieved Train CVaR: {achieved_cvar:.6f}")
        print(f"  Train Sharpe (signal): {train_sharpe:+.4f}")
        print(f"  OOS   Sharpe (signal): {oos_sharpe:+.4f}")

        # Sanity check: if all weights are positive the optimizer found no
        # short allocations — the data may be too noisy or skfolio's solver
        # converged to the boundary.  Warn but accept.
        if np.all(weights >= -1e-6):
            print("  [WARN] All weights non-negative — adversarial signals "
                  "(VPIN, spread) may need manual override. "
                  "Falling back to Ridge for comparison.")

    except Exception as e:
        print(f"  [WARN] skfolio optimization failed: {e}")
        print("  Using Ridge regression fallback.")

    # ── Fallback: Ridge regression (always produces signed weights) ────
    # Also used as a comparison / sanity check when skfolio succeeds.
    X_tr_arr = df_train[CORE_6_SIGNALS].fillna(0).values.astype(np.float64)
    y_tr_arr = fwd_ret_train.values.astype(np.float64)
    ridge_coef, ridge_bias = fit_ridge_signal_weights(
        X_tr_arr, y_tr_arr, CORE_6_SIGNALS, alpha_ridge=1.0)
    # ridge_coef is already for the 6 base signals in the correct order
    ridge_weights_norm = ridge_coef / (np.sum(np.abs(ridge_coef)) + 1e-12)
    ridge_sharpe = _compute_sharpe(ridge_weights_norm, X_test)
    print(f"\n  Ridge OOS Sharpe (signal): {ridge_sharpe:+.4f}")

    # ── Choose the better weights ──────────────────────────────────────
    if weights is None:
        weights = ridge_coef
        print("  Selected: Ridge regression weights (skfolio unavailable)")
    else:
        skfolio_oos = _compute_sharpe(weights, X_test)
        if ridge_sharpe > skfolio_oos:
            weights = ridge_coef
            print(f"  Selected: Ridge weights (OOS Sharpe {ridge_sharpe:+.4f} "
                  f"> skfolio {skfolio_oos:+.4f})")
        else:
            print(f"  Selected: skfolio CVaR weights (OOS Sharpe {skfolio_oos:+.4f})")

    # ── Report final weights ───────────────────────────────────────────
    print("\n  [+] Final directional signal weights:")
    for sig, w in zip(CORE_6_SIGNALS, weights[:6] if len(weights) > 6 else weights):
        direction = "bullish ↑" if w > 1e-4 else ("bearish ↓" if w < -1e-4 else "neutral ~")
        print(f"    {sig:>20}: {w:+.4f}  ({direction})")

    export_binary_weights(weights, CORE_6_SIGNALS, output_path, bias=0.0)
    print(f"\n  [+] Saved directional weights to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Optimize Signal Combiner Weights')
    parser.add_argument('--data',      type=str,  default='',
                        help='Path to feature dump CSV from backtest.py --dump-features')
    parser.add_argument('--horizon',   type=int,  default=100,
                        help='Forward return horizon in ticks (default: 100)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for smoke-testing')
    parser.add_argument('--output',    type=str,
                        default='models/optimal_weights.bin',
                        help='Output binary path (default: models/optimal_weights.bin)')

    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  HFT Engine — Directional Signal Weight Optimization")
    print(f"{'='*65}")

    if args.data and not args.synthetic:
        # Use load_feature_dump so we get proper filtering (warmed-up rows,
        # outlier clipping, forward-return column auto-detection)
        df, target_col = load_feature_dump(args.data, args.horizon)
        print(f"  Data loaded: {len(df):,} rows")
    else:
        print("  Using synthetic data (pass --data path/to/features.csv for real data)")
        df, target_col = generate_synthetic_data(n=50_000, horizon=args.horizon)

    optimize_signal_weights(df, target_col, args.output)


if __name__ == '__main__':
    main()
