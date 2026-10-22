#!/usr/bin/env python3
"""
python/ml_pipeline/train_hmm_regime.py
════════════════════════════════════════════════════════════════════════════════
Continuous Probabilistic Regime Model Trainer (4 Regimes)
────────────────────────────────────────────────────────────────────────────────
Fits a 4-component Gaussian Mixture / HMM on High-Frequency Volatility and Spread
BPS to identify:
  0: MEAN_REVERTING (Low vol, tight spread)
  1: MOMENTUM_TREND (Moderate vol, directional flow)
  2: HIGH_VOL_WIDE  (Elevated vol, wide spread)
  3: TOXIC_CRISIS   (Extreme vol blowout, adverse selection risk)

Exports:
  - models/hmm_regime.pkl
  - models/hmm_regime_meta.json
  - python/models/hmm_regime.pkl
  - python/models/hmm_regime_meta.json
"""

import os
import sys
import glob
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIRS = [
    os.path.join(ROOT, "models"),
    os.path.join(ROOT, "python", "models")
]


def load_vol_and_spread_data() -> np.ndarray:
    live_ticks = sorted(glob.glob(os.path.join(DATA_DIR, "live", "*", "ticks_*.csv")))
    obs_list = []
    
    if live_ticks:
        print(f"[*] Extracting vol & spread from {len(live_ticks)} recorded tick files...")
        for f in live_ticks:
            try:
                df = pd.read_csv(f)
                vol_col = None
                for col in ["volatility_ema_100", "realized_vol", "vol"]:
                    if col in df.columns:
                        vol_col = col
                        break
                spread_col = None
                for col in ["spread_bps", "spread"]:
                    if col in df.columns:
                        spread_col = col
                        break
                        
                if vol_col and spread_col:
                    sub = df[[vol_col, spread_col]].dropna()
                    obs_list.append(sub.values)
            except Exception:
                continue
                
    if obs_list:
        combined = np.vstack(obs_list)
        if len(combined) > 1000:
            print(f"[+] Loaded {len(combined):,} live market observations.")
            return combined

    # Synthetic realistic microstructure distribution
    print("[*] Generating synthetic multi-regime distribution for calibration...")
    rng = np.random.RandomState(42)
    
    # 4 distinct regime clusters
    # 1. Calm Mean-Reverting: vol ~ 0.4-0.8, spread ~ 0.3-0.6 bps (50% of time)
    n1 = 50_000
    vol1 = rng.normal(0.6, 0.15, n1)
    sp1 = rng.normal(0.45, 0.10, n1)
    
    # 2. Trending Momentum: vol ~ 1.0-1.4, spread ~ 0.7-1.1 bps (30% of time)
    n2 = 30_000
    vol2 = rng.normal(1.2, 0.20, n2)
    sp2 = rng.normal(0.9, 0.18, n2)
    
    # 3. High Volatility: vol ~ 1.8-2.6, spread ~ 1.4-2.0 bps (15% of time)
    n3 = 15_000
    vol3 = rng.normal(2.1, 0.35, n3)
    sp3 = rng.normal(1.7, 0.30, n3)
    
    # 4. Toxic Crisis Blowout: vol ~ 3.5-6.0, spread ~ 2.8-5.0 bps (5% of time)
    n4 = 5_000
    vol4 = rng.normal(4.2, 0.8, n4)
    sp4 = rng.normal(3.5, 0.7, n4)
    
    vols = np.concatenate([vol1, vol2, vol3, vol4])
    sps = np.concatenate([sp1, sp2, sp3, sp4])
    obs = np.column_stack([np.clip(vols, 0.05, None), np.clip(sps, 0.1, None)])
    return obs


def train_regime_model():
    print("=" * 70)
    print("  HFT QUANT ENGINE: 4-REGIME PROBABILISTIC HMM/GMM TRAINING")
    print("=" * 70)
    
    obs = load_vol_and_spread_data()
    mean = np.mean(obs, axis=0)
    std = np.std(obs, axis=0) + 1e-8
    
    obs_scaled = (obs - mean) / std
    
    print(f"[*] Fitting Gaussian Mixture Model with 4 components...")
    gmm = GaussianMixture(
        n_components=4,
        covariance_type="full",
        max_iter=200,
        random_state=42,
        n_init=5
    )
    gmm.fit(obs_scaled)
    
    # Order components by ascending risk (volatility + spread mean)
    unscaled_means = gmm.means_ * std + mean
    risk_scores = unscaled_means[:, 0] * 1.5 + unscaled_means[:, 1]
    sorted_order = np.argsort(risk_scores)
    
    state_mapping = {}
    for new_idx, old_idx in enumerate(sorted_order):
        state_mapping[str(old_idx)] = int(new_idx)
        
    print("\n[+] Learned Regime Cluster Profiles (Unscaled Means):")
    regime_names = ["MEAN_REVERTING", "MOMENTUM_TREND", "HIGH_VOL_WIDE", "TOXIC_CRISIS"]
    for new_idx, old_idx in enumerate(sorted_order):
        v = unscaled_means[old_idx, 0]
        s = unscaled_means[old_idx, 1]
        weight = gmm.weights_[old_idx] * 100.0
        print(f"  Regime {new_idx} ({regime_names[new_idx]:<14}): Vol={v:.2f} | Spread={s:.2f} bps | Prior Weight={weight:.1f}%")
        
    bundle = {
        "model": gmm,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "state_mapping": state_mapping
    }
    
    meta = {
        "state_mapping": state_mapping,
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist()
    }
    
    for out_dir in OUTPUT_DIRS:
        os.makedirs(out_dir, exist_ok=True)
        pkl_path = os.path.join(out_dir, "hmm_regime.pkl")
        meta_path = os.path.join(out_dir, "hmm_regime_meta.json")
        
        joblib.dump(bundle, pkl_path)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[+] Saved model to {pkl_path} and {meta_path}")

    print("\n[+] HMM Regime Detector Calibration Complete.")


if __name__ == "__main__":
    train_regime_model()
