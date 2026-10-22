#!/usr/bin/env python3
"""
scripts/train_ridge_live.py
════════════════════════════════════════════════════════════════════════════════
Trains Ridge regression model on live recorded orderbook ticks from data/live/
and exports calibrated 56-byte binary models/signal_weights.bin and metadata.
"""

import os
import sys
import glob
import struct
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BIN = os.path.join(ROOT, "models", "signal_weights.bin")
OUT_BIN_PROD = os.path.join(ROOT, "models", "production", "signal_weights.bin")
OUT_META = os.path.join(ROOT, "models", "signal_weights_meta.json")

KEY_ORDER = ["w_obi", "w_vpin", "w_vol", "w_spread", "w_ofi", "w_microprice", "w_bias"]


def train_live_ridge():
    print("=" * 70)
    print("  HFT QUANT ENGINE: RIDGE SIGNAL WEIGHTS CALIBRATION (LIVE DATA)")
    print("=" * 70)

    tick_files = sorted(glob.glob(os.path.join(ROOT, "data", "live", "*", "ticks_*.csv")))
    if not tick_files:
        print("[ERROR] No tick files found in data/live/*/")
        return

    dfs = []
    for f in tick_files:
        df = pd.read_csv(f)
        if len(df) >= 20:
            df["forward_ret_10"] = (df["mid_price"].shift(-10) - df["mid_price"]) / df["mid_price"]
            dfs.append(df)

    if not dfs:
        print("[ERROR] Insufficient tick depth across tick files.")
        return

    raw_df = pd.concat(dfs, ignore_index=True).dropna(subset=["forward_ret_10"])
    print(f"[*] Loaded {len(raw_df):,} live ticks with valid 10-tick forward returns.")

    # Construct feature matrix aligned with StrategyEngine
    X_df = pd.DataFrame({
        "w_obi": raw_df["depth_imb"].fillna(0.0),
        "w_vpin": raw_df["vpin"].fillna(0.5) - 0.5,
        "w_vol": raw_df["hf_vol"].fillna(1.0),
        "w_spread": -(raw_df["spread_bps"].fillna(0.5) / 10.0),
        "w_ofi": raw_df["ofi"].fillna(0.0),
        "w_microprice": (raw_df["micro_bps"].fillna(0.0) / 10.0),
    })
    y = raw_df["forward_ret_10"].values.astype(np.float64)

    # 70/30 time-ordered train/val split
    split = int(len(X_df) * 0.70)
    X_train, X_val = X_df.iloc[:split], X_df.iloc[split:]
    y_train, y_val = y[:split], y[split:]

    # Standardize features for stable Ridge conditioning
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Fit Ridge
    ridge = Ridge(alpha=10.0, fit_intercept=True)
    ridge.fit(X_train_s, y_train)

    # Convert standardized coefficients back to unstandardized scale
    raw_coefs = ridge.coef_ / scaler.scale_
    raw_intercept = ridge.intercept_ - np.sum(raw_coefs * scaler.mean_)

    y_pred = ridge.predict(X_val_s)
    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    oos_r2 = float(1.0 - (ss_res / max(ss_tot, 1e-15)))
    pearson_r = float(np.corrcoef(y_pred, y_val)[0, 1]) if np.std(y_pred) > 1e-9 else 0.0
    hit_rate = float(np.mean(np.sign(y_pred) == np.sign(y_val)))

    strat_rets = np.sign(y_pred) * y_val
    sharpe_oos = float(np.mean(strat_rets) / (np.std(strat_rets) + 1e-15) * np.sqrt(252 * 24 * 3600)) if len(strat_rets) > 0 else 0.0

    print(f"\n[+] OOS Performance Metrics (n_val={len(X_val):,} ticks):")
    print(f"    • Information Coefficient (Pearson r): {pearson_r:+.4f}")
    print(f"    • Directional Hit Rate              : {hit_rate * 100:.2f}%")
    print(f"    • Out-of-Sample R²                  : {oos_r2:+.6f}")
    print(f"    • Annualized Strategy Sharpe        : {sharpe_oos:.2f}")

    # Build weights dict
    feature_cols = list(X_df.columns)
    weights = dict(zip(feature_cols, raw_coefs))
    weights["w_bias"] = float(raw_intercept)

    print("\n[+] Calibrated Alpha Coefficients:")
    for k in KEY_ORDER[:-1]:
        print(f"    {k:<15}: {weights[k]:+.8f}")
    print(f"    {'w_bias':<15}: {weights['w_bias']:+.8f}")

    vals = [float(weights.get(k, 0.0)) for k in KEY_ORDER]

    # Export to binary files
    os.makedirs(os.path.dirname(OUT_BIN), exist_ok=True)
    with open(OUT_BIN, "wb") as f:
        f.write(struct.pack(f"{len(vals)}d", *vals))

    os.makedirs(os.path.dirname(OUT_BIN_PROD), exist_ok=True)
    with open(OUT_BIN_PROD, "wb") as f:
        f.write(struct.pack(f"{len(vals)}d", *vals))

    meta = {
        "keys": KEY_ORDER,
        "values": vals,
        "oos_pearson_r": pearson_r,
        "oos_hit_rate": hit_rate,
        "oos_r2": oos_r2,
        "oos_sharpe": sharpe_oos,
        "n_train": split,
        "n_val": len(X_val),
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SUCCESS] Exported 56-byte binary weights:")
    print(f"  --> {OUT_BIN} ({os.path.getsize(OUT_BIN)} bytes)")
    print(f"  --> {OUT_BIN_PROD} ({os.path.getsize(OUT_BIN_PROD)} bytes)")
    print(f"  --> {OUT_META}")
    print("=" * 70)


if __name__ == "__main__":
    train_live_ridge()
