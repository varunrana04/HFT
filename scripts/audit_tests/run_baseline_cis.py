"""
run_baseline_cis.py — Baseline Bootstrap Confidence Interval computation.

Data source: real paper trade CSV journals (paper_trades_*.csv).

WHY NOT TARDIS REPLAY:
  The C++ feature engine (OFI, OBI, VPIN, microprice) requires consecutive
  streaming book-delta updates to compute non-zero features. Static Tardis
  book snapshot CSVs cannot seed these rolling state variables — the result
  is 0 trades regardless of threshold. This is an architectural constraint,
  not a bug: the engine was designed for live streaming, not batch replay.

  The correct baseline is from actual out-of-sample paper trade equity curves,
  which are the ground truth of realized performance.
"""
import sys
import os
import glob
import json

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def block_bootstrap_sharpe(pnls, num_bootstraps=1000, block_size=10):
    np.random.seed(42)
    n = len(pnls)
    if n < 2:
        return 0.0, 0.0, 0.0
    if n < block_size:
        block_size = max(1, n // 2)

    num_blocks = n // block_size + 1
    bootstrapped_sharpes = []

    for _ in range(num_bootstraps):
        start_indices = np.random.randint(0, n - block_size + 1, num_blocks)
        sampled_pnls = np.concatenate([pnls[i:i + block_size] for i in start_indices])[:n]
        mean_pnl, std_pnl = np.mean(sampled_pnls), np.std(sampled_pnls)
        if std_pnl > 0:
            sharpe = (mean_pnl / std_pnl) * np.sqrt(10000)
            bootstrapped_sharpes.append(sharpe)

    if not bootstrapped_sharpes:
        return 0.0, 0.0, 0.0
    bootstrapped_sharpes = np.array(bootstrapped_sharpes)
    return float(np.mean(bootstrapped_sharpes)), float(np.percentile(bootstrapped_sharpes, 2.5)), float(np.percentile(bootstrapped_sharpes, 97.5))


def load_paper_trade_sessions(root: str, min_rows: int = 50) -> dict:
    """Load all paper trade CSV sessions, returning per-session equity curves."""
    pattern = os.path.join(root, "paper_trades_*.csv")
    files = sorted(glob.glob(pattern))
    sessions = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            if "Equity" not in df.columns:
                if "PnL" in df.columns:
                    df["Equity"] = 100000.0 + df["PnL"].cumsum()
                else:
                    continue
            if len(df) < min_rows:
                continue
            
            if "Timestamp" in df.columns:
                df = df.sort_values("Timestamp").reset_index(drop=True)
            elif "TimestampNs" in df.columns:
                df = df.sort_values("TimestampNs").reset_index(drop=True)
                
            run_id = os.path.basename(f).replace("paper_trades_", "").replace(".csv", "")
            sessions[run_id] = df
            print(f"  Loaded session {run_id}: {len(df)} fills, "
                  f"equity ${df['Equity'].iloc[-1]:,.2f}")
        except Exception as e:
            print(f"  Skip {f}: {e}")
    return sessions


def compute_session_stats(df: pd.DataFrame) -> dict:
    """Compute per-fill PnL returns and key stats from a session DataFrame."""
    equity = df["Equity"].values.astype(float)
    initial = equity[0]
    final   = equity[-1]
    net_pnl = final - initial

    # Per-fill equity changes as returns
    returns = np.diff(equity)
    n_fills = len(df)
    n_wins  = int((returns > 0).sum())
    n_loss  = int((returns < 0).sum())

    max_dd = 0.0
    peak = equity[0]
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd

    mean_r = float(np.mean(returns)) if len(returns) > 0 else 0.0
    std_r  = float(np.std(returns))  if len(returns) > 0 else 1.0
    sharpe_raw = (mean_r / std_r * np.sqrt(len(returns))) if std_r > 0 else 0.0

    return {
        "n_fills":   n_fills,
        "net_pnl":   net_pnl,
        "win_rate":  n_wins / max(1, n_fills - 1),
        "max_dd":    max_dd,
        "sharpe_raw": sharpe_raw,
        "returns":   returns,
    }


def run_baseline_cis(root: str, num_bootstraps: int = 2000, block_size: int = 10) -> dict:
    """
    Main entry point.
    Loads all paper_trades_*.csv sessions, pools the per-fill returns,
    and computes block-bootstrap Sharpe 95% CIs.
    """
    print("=" * 60)
    print("  HFT Engine — Baseline Bootstrap Confidence Intervals")
    print("  Data: paper_trades_*.csv (realized out-of-sample fills)")
    print("=" * 60)

    sessions = load_paper_trade_sessions(root)
    if not sessions:
        print("ERROR: No qualifying paper trade sessions found.")
        return {}

    # --- Per-session stats ---
    all_returns = []
    session_summaries = []
    for run_id, df in sessions.items():
        stats = compute_session_stats(df)
        session_summaries.append({
            "run_id":   run_id,
            "n_fills":  stats["n_fills"],
            "net_pnl":  stats["net_pnl"],
            "win_rate": stats["win_rate"],
            "max_dd":   stats["max_dd"],
            "sharpe":   stats["sharpe_raw"],
        })
        all_returns.extend(stats["returns"].tolist())

    all_returns = np.array(all_returns, dtype=float)

    # --- Pooled bootstrap Sharpe ---
    mean_sharpe, ci_low, ci_high = block_bootstrap_sharpe(
        all_returns, num_bootstraps=num_bootstraps, block_size=block_size
    )

    # --- Pooled summary stats ---
    total_fills  = sum(s["n_fills"] for s in session_summaries)
    total_pnl    = sum(s["net_pnl"] for s in session_summaries)
    avg_win_rate = float(np.mean([s["win_rate"] for s in session_summaries]))
    avg_max_dd   = float(np.mean([s["max_dd"] for s in session_summaries]))
    pooled_pf    = float((all_returns[all_returns > 0].sum()) /
                         max(1e-9, abs(all_returns[all_returns < 0].sum())))

    print()
    print("=== Per-Session Summary ===")
    for s in session_summaries:
        print(f"  [{s['run_id']}]  fills={s['n_fills']:5d}  "
              f"net_pnl=${s['net_pnl']:+,.2f}  "
              f"win_rate={s['win_rate']*100:.1f}%  "
              f"max_dd={s['max_dd']*100:.2f}%  "
              f"sharpe={s['sharpe']:.3f}")

    print()
    print("=== Pooled Baseline Results ===")
    print(f"  Total fills analysed : {total_fills:,}")
    print(f"  Total net PnL        : ${total_pnl:+,.2f}")
    print(f"  Avg win rate         : {avg_win_rate*100:.2f}%")
    print(f"  Avg max drawdown     : {avg_max_dd*100:.2f}%")
    print(f"  Pooled profit factor : {pooled_pf:.4f}")
    print(f"  Bootstrap Sharpe     : Mean={mean_sharpe:.4f}")
    print(f"  95% CI               : [{ci_low:.4f}, {ci_high:.4f}]")
    print()

    if ci_low > 0:
        verdict = "[PASS] POSITIVE -- 95% CI entirely above zero. Edge is statistically real."
    elif ci_high < 0:
        verdict = "[FAIL] NEGATIVE -- 95% CI entirely below zero. Strategy is losing."
    else:
        verdict = "[WARN] STRADDLES ZERO -- More data needed for conclusive proof."
    print(f"  Verdict: {verdict}")
    print("=" * 60)

    results = {
        "sessions":        session_summaries,
        "total_fills":     total_fills,
        "total_pnl":       total_pnl,
        "avg_win_rate":    avg_win_rate,
        "avg_max_dd":      avg_max_dd,
        "profit_factor":   pooled_pf,
        "bootstrap_sharpe_mean": mean_sharpe,
        "bootstrap_sharpe_ci_low":  ci_low,
        "bootstrap_sharpe_ci_high": ci_high,
        "verdict": verdict,
    }

    # Save JSON artifact
    out_path = os.path.join(root, "reports", "data", "baseline_cis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Results saved -> {out_path}")

    return results


if __name__ == "__main__":
    run_baseline_cis(ROOT)
