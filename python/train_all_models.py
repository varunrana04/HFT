#!/usr/bin/env python3
"""
python/train_all_models.py
════════════════════════════════════════════════════════════════════════════════
Central Model Training & Calibration Orchestrator
────────────────────────────────────────────────────────────────────────────────
One-click CLI to train, calibrate, and validate all quantitative ML models:
  1. LightGBM Signal Booster  (8 L2 features, custom asymmetric loss)
  2. HMM Regime Detector      (4 Gaussian regimes: Mean-Rev, Trend, Vol, Crisis)
  3. Ridge Signal Combiner     (Walk-forward CVaR weights, 96-byte binary export)
  4. Online RL Policy Trainer  (Continuous PPO policy calibration on live fills)

Usage:
  python python/train_all_models.py --model all
  python python/train_all_models.py --model lgbm --rounds 150
  python python/train_all_models.py --model hmm
  python python/train_all_models.py --model ridge
  python python/train_all_models.py --model rl
"""

import os
import sys
import argparse
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_step(step_name: str, cmd: list) -> bool:
    print("\n" + "=" * 70)
    print(f"  RUNNING: {step_name}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 70)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if res.returncode == 0:
        print(f"[+] {step_name} completed successfully in {elapsed:.2f}s.")
        return True
    else:
        print(f"[!] {step_name} FAILED with exit code {res.returncode} in {elapsed:.2f}s.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Institutional HFT Model Training Orchestrator")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "lgbm", "hmm", "ridge", "rl"],
                        help="Model to train (all, lgbm, hmm, ridge, rl)")
    parser.add_argument("--rounds", type=int, default=150, help="Boosting rounds for LightGBM")
    args = parser.parse_args()

    print("=" * 70)
    print("  INSTITUTIONAL QUANT TRADING ENGINE — MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Target: {args.model.upper()}")
    print(f"  Root:   {ROOT}")

    py = sys.executable
    successes = []

    # 1. HMM Regime Detector
    if args.model in ["all", "hmm"]:
        cmd = [py, os.path.join(ROOT, "python", "ml_pipeline", "train_hmm_regime.py")]
        ok = run_step("HMM Regime Detector", cmd)
        successes.append(("HMM Regime Detector", ok))

    # 2. LightGBM Signal Model
    if args.model in ["all", "lgbm"]:
        cmd = [py, os.path.join(ROOT, "python", "ml_pipeline", "train_lightgbm.py"), "--rounds", str(args.rounds)]
        ok = run_step("LightGBM Microstructure Signal Model (8 Features)", cmd)
        successes.append(("LightGBM Signal Model", ok))

    # 3. Ridge Combiner
    if args.model in ["all", "ridge"]:
        cmd = [py, os.path.join(ROOT, "python", "ml_pipeline", "train_model.py"), "--synthetic"]
        ok = run_step("Ridge Signal Combiner (96-byte binary)", cmd)
        successes.append(("Ridge Combiner", ok))

    # 4. Online RL Policy
    if args.model in ["all", "rl"]:
        cmd = [py, os.path.join(ROOT, "scripts", "train_online_rl_offline.py")]
        ok = run_step("Online RL Policy Calibration", cmd)
        successes.append(("Online RL Policy", ok))

    print("\n" + "=" * 70)
    print("  MODEL TRAINING PIPELINE SUMMARY")
    print("=" * 70)
    all_ok = True
    for name, ok in successes:
        status = "[SUCCESS]" if ok else "[FAILED]"
        print(f"  {status:<10} : {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n[+] All requested models were trained and deployed to models/ successfully!")
    else:
        print("\n[!] One or more models encountered an issue. See logs above.")


if __name__ == "__main__":
    main()
