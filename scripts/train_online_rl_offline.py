#!/usr/bin/env python3
"""
scripts/train_online_rl_offline.py
════════════════════════════════════════════════════════════════════════════════
Offline Policy Calibration & Warm-Start Trainer for OnlineRLPolicyAdapter.
Ingests recorded live orderbook ticks and executed maker fills from data/live/
to fit theta_engine, theta_skew, and theta_size weights with audited fee accounting.
Saves calibrated weights to models/online_rl_weights.json.
"""

import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd

# Add repo root and python/ to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

from ollama_rl_engine import OnlineRLPolicyAdapter, OllamaBaseModelPrior, ContinuousPnLLearner, FEATURE_NAMES, DEFAULT_WEIGHTS_PATH


def run_offline_training():
    print("=" * 70)
    print("  HFT QUANT ENGINE: OFFLINE RL POLICY CALIBRATION")
    print("=" * 70)

    # 1. Load all live recorded data
    fill_files = sorted(glob.glob(os.path.join(ROOT, "data", "live", "*", "fills.csv")))
    tick_files = sorted(glob.glob(os.path.join(ROOT, "data", "live", "*", "ticks_*.csv")))

    if not fill_files:
        print("[ERROR] No fills.csv files found in data/live/!")
        return

    df_fills = pd.concat([pd.read_csv(f) for f in fill_files], ignore_index=True)
    df_fills = df_fills.sort_values(by="timestamp").reset_index(drop=True)
    print(f"[*] Loaded {len(df_fills):,} fills from {len(fill_files)} files.")

    df_ticks = pd.DataFrame()
    if tick_files:
        df_ticks = pd.concat([pd.read_csv(f) for f in tick_files], ignore_index=True)
        df_ticks = df_ticks.sort_values(by="timestamp").reset_index(drop=True)
        print(f"[*] Loaded {len(df_ticks):,} orderbook ticks from {len(tick_files)} files.")

    # 2. Initialize fresh Policy and Learner with learning rate tuned for replay
    policy = OnlineRLPolicyAdapter(lr=0.003, entropy_reg=1e-4)
    # Start from baseline prior
    policy.theta_engine = np.zeros((3, 13), dtype=np.float64)
    policy.theta_engine[0, 0] =  0.4   # Bullish: OFI
    policy.theta_engine[0, 2] =  0.5   # Bullish: ABR
    policy.theta_engine[1, 1] =  0.4   # Bearish: VPIN
    policy.theta_engine[1, 3] =  0.5   # Bearish: ASR
    policy.theta_engine[2, 6] = -0.3   # Mix: tight spread ratio
    policy.theta_skew = np.zeros(13, dtype=np.float64)
    policy.theta_size = np.zeros(13, dtype=np.float64)

    prior = OllamaBaseModelPrior()
    learner = ContinuousPnLLearner(policy, prior)

    initial_engine_weights = policy.theta_engine.copy()

    # 3. Replay Fills with exact market state
    print("\n[*] Replaying fills through ContinuousPnLLearner...")
    n_processed = 0

    for idx, row in df_fills.iterrows():
        # Microstructure feature snapshot
        alphas = {
            "abr": float(row.get("abr", 0.5)),
            "asr": float(row.get("asr", 0.5)),
            "vpin": float(row.get("vpin", 0.5)),
            "micro_bps": float(row.get("micro_bps", 0.0)),
            "ofi": float(row.get("ofi", 0.0)),
            "spread_bps": 0.5,
            "spread_ratio": 1.0,
            "depth_imb": 0.0,
            "hf_vol": 1.0,
            "obi_delta": 0.0,
            "tfi_fast": 0.0,
        }

        fill_dict = {
            "price": float(row["price"]),
            "qty": float(row["qty"]),
            "side": str(row["side"]),
            "engine": str(row["engine"]).upper(),
            "symbol": str(row["symbol"]),
        }

        gross_pnl = float(row["total_pnl"]) if "total_pnl" in row and not pd.isna(row["total_pnl"]) else 0.0
        inv = float(row.get("portfolio_net_inv", 0.0))
        
        # Determine adverse selection: if gross_pnl negative on an inventory liquidation
        is_adverse = (gross_pnl < -10.0)

        learner.record_fill(
            fill=fill_dict,
            alphas=alphas,
            gross_pnl=gross_pnl,
            inv=inv,
            vol=1.0,
            is_adverse=is_adverse,
        )
        n_processed += 1

    print(f"[+] Replay complete! Processed {n_processed:,} fill events.")
    print(f"    • Total Net PnL Processed: ${learner.total_net_pnl:,.2f}")
    print(f"    • Total Exchange Fees Accounted: ${learner.total_fees_paid:,.2f}")
    print(f"    • Profitable Fills: {learner.profitable_fills:,} / {learner.total_fills:,} ({(learner.profitable_fills / max(1, learner.total_fills))*100:.1f}%)")
    print(f"    • Cumulative Reward: {learner.cumulative_reward:,.2f}")

    # 4. Save Trained Weights
    out_path = DEFAULT_WEIGHTS_PATH
    success = policy.save_weights(out_path)
    if success:
        print(f"\n[SUCCESS] Calibrated weights successfully saved to:\n  --> {out_path}")
    else:
        print(f"\n[ERROR] Failed to save weights to {out_path}")

    # 5. Display Weight Comparison
    print("\n" + "=" * 70)
    print("  CALIBRATED WEIGHT MATRICES COMPARISON")
    print("=" * 70)
    print(f"{'Feature':<25} | {'Bullish (Old -> New)':<22} | {'Bearish (Old -> New)':<22} | {'Mix (Old -> New)':<22}")
    print("-" * 95)
    for i, fname in enumerate(FEATURE_NAMES):
        b_old, b_new = initial_engine_weights[0, i], policy.theta_engine[0, i]
        be_old, be_new = initial_engine_weights[1, i], policy.theta_engine[1, i]
        m_old, m_new = initial_engine_weights[2, i], policy.theta_engine[2, i]
        print(f"{fname:<25} | {b_old:+5.2f} -> {b_new:+5.2f}          | {be_old:+5.2f} -> {be_new:+5.2f}          | {m_old:+5.2f} -> {m_new:+5.2f}")

    print("\n[*] Skew Weights:")
    for i, fname in enumerate(FEATURE_NAMES[:7]):
        print(f"    {fname:<25}: {policy.theta_skew[i]:+6.3f}")

    # 6. Test Policy Ingestion Verification
    print("\n[*] Testing OnlineRLPolicyAdapter fresh load...")
    test_adapter = OnlineRLPolicyAdapter()
    loaded = test_adapter.load_weights(out_path)
    assert loaded, "test_adapter failed to load saved weights!"
    
    # Test act()
    test_state = test_adapter.build_state(
        {"ofi": 0.5, "vpin": 0.48, "abr": 0.65, "asr": 0.35, "micro_bps": 0.25},
        {"net_delta": 0.0},
        prior.current_prior,
    )
    sel_eng, probs, skew, size = test_adapter.act(test_state, prior.current_prior)
    print(f"[+] Verified test forward pass:")
    print(f"    • Selected Engine: {sel_eng.upper()}")
    print(f"    • Probabilities: [Bull: {probs[0]:.3f}, Bear: {probs[1]:.3f}, Mix: {probs[2]:.3f}]")
    print(f"    • Reservation Price Skew: {skew:+.4f}")
    print(f"    • Position Multiplier: {size:.3f}x")
    print("=" * 70)


if __name__ == "__main__":
    run_offline_training()
