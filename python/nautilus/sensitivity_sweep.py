"""
Sensitivity Sweep — Step 1 of Stress Testing
=============================================
Tests robustness of the LGBM edge across a grid of:
  - Confidence threshold (0.1 → 2.0 bps)
  - Taker fee         (0.05 → 0.5 bps)
  - Cooldown          (100 → 1000 ms)

KEY EFFICIENCY: LGBM predictions are computed ONCE per window,
then ALL threshold/fee/cooldown combinations are applied cheaply
via NumPy — no re-running the model.

Usage (from HFT root):
    python python/nautilus/sensitivity_sweep.py --period 1month
    python python/nautilus/sensitivity_sweep.py --period 1year   # takes ~20min
    python python/nautilus/sensitivity_sweep.py --period 1month --quick  # 3x3x2 grid
"""

import argparse, json, time
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb

HERE      = Path(__file__).parent
ML_DIR    = HERE.parent / "ml_pipeline" / "models"
LGBM_PATH = ML_DIR / "lgbm_trade_tape_model.txt"
CACHE_DIR = Path("nautilus_cache")
OUT_DIR   = Path("hft_nt_results")
OUT_DIR.mkdir(exist_ok=True)

FEATURES   = ["tfi_ema_50", "tfi_ema_200", "volatility_ema_100"]
WINDOW_DAYS = 7

# ── Parameter Grids ────────────────────────────────────────────────────────────
FULL_GRID = {
    "threshold": [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],  # bps
    "fee":       [0.05, 0.1, 0.2, 0.5],                   # bps round-trip
    "cooldown":  [100, 200, 500, 1000],                    # ms
}
QUICK_GRID = {
    "threshold": [0.3, 0.5, 1.0],
    "fee":       [0.05, 0.1, 0.2],
    "cooldown":  [100, 200],
}

# ── Data Loading ───────────────────────────────────────────────────────────────
def get_cached_dates():
    files = sorted(CACHE_DIR.glob("BTCUSDT-aggTrades-*.parquet"))
    return [date.fromisoformat(f.stem.split("-aggTrades-")[1]) for f in files]

def load_window(days: list[date]) -> pl.DataFrame:
    frames = []
    for d in days:
        p = CACHE_DIR / f"BTCUSDT-aggTrades-{d}.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p))
    if not frames:
        return pl.DataFrame()
    df = pl.concat(frames).sort("transact_time")
    df = df.with_columns([
        pl.when(pl.col("is_buyer_maker"))
          .then(-pl.col("quantity")).otherwise(pl.col("quantity"))
          .alias("signed_volume"),
    ])
    df = df.with_columns([
        pl.col("signed_volume").ewm_mean(half_life=50).alias("tfi_ema_50"),
        pl.col("signed_volume").ewm_mean(half_life=200).alias("tfi_ema_200"),
        pl.col("price").diff().abs().ewm_mean(half_life=100).alias("volatility_ema_100"),
    ])
    return df.drop_nulls(subset=FEATURES)

# ── Signal + Simulate with arbitrary params ────────────────────────────────────
def sweep_params(preds: np.ndarray, prices: np.ndarray,
                 times: np.ndarray, labels: np.ndarray,
                 threshold: float, fee_bps: float, cooldown_ms: int) -> dict:
    """Apply a single (threshold, fee, cooldown) combo to pre-computed predictions.

    VECTORIZED: the cooldown gate loops only over signal events (~1% of ticks)
    instead of all ticks — 100x faster for large windows.
    """
    # Step 1: apply threshold (pure numpy, instant)
    raw = np.where(preds > threshold, 1,
          np.where(preds < -threshold, -1, 0)).astype(np.int8)

    # Step 2: find only the indices where signal fires (tiny subset)
    fire_idx   = np.where(raw != 0)[0]
    if len(fire_idx) == 0:
        return None
    fire_times = times[fire_idx]
    fire_raw   = raw[fire_idx]

    # Step 3: apply cooldown — loop only over signal events (fast)
    keep   = np.zeros(len(fire_idx), dtype=bool)
    last_t = -np.inf
    for j in range(len(fire_idx)):
        if fire_times[j] - last_t >= cooldown_ms:
            keep[j] = True
            last_t  = fire_times[j]

    valid_idx    = fire_idx[keep]
    valid_sigs   = fire_raw[keep]
    valid_labels = labels[valid_idx]

    # Step 4: PnL calculation (numpy, instant)
    rets = valid_labels * valid_sigs - fee_bps
    if len(rets) == 0:
        return None

    pf     = rets[rets > 0].sum() / (abs(rets[rets < 0].sum()) + 1e-9)
    cumret = np.cumsum(rets)
    maxdd  = (cumret - np.maximum.accumulate(cumret)).min()

    return {
        "trades": len(rets),
        "pnl":    rets.sum(),
        "win":    (rets > 0).mean() * 100,
        "pf":     pf,
        "maxdd":  maxdd,
        "sharpe": rets.mean() / (rets.std() + 1e-9) * np.sqrt(len(rets) * 252),
    }

# ── Main Sweep ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="1month", choices=["1month", "1year"])
    parser.add_argument("--quick", action="store_true", help="Reduced grid (faster)")
    args = parser.parse_args()

    grid = QUICK_GRID if args.quick else FULL_GRID
    n_combos = len(grid["threshold"]) * len(grid["fee"]) * len(grid["cooldown"])

    print("="*70)
    print("  HFT Sensitivity Sweep")
    print("="*70)
    print(f"  Period  : {args.period}")
    print(f"  Grid    : {len(grid['threshold'])} thresholds × {len(grid['fee'])} fees"
          f" × {len(grid['cooldown'])} cooldowns = {n_combos} combos")
    print(f"  Mode    : {'QUICK' if args.quick else 'FULL'}")

    # Load model
    if not LGBM_PATH.exists():
        print(f"[ERROR] LGBM model not found: {LGBM_PATH}")
        return
    model = lgb.Booster(model_file=str(LGBM_PATH))
    print(f"  Model   : {model.num_trees()} trees")

    # Get windows
    all_dates = get_cached_dates()
    if not all_dates:
        print("[ERROR] No cache. Run download first.")
        return

    PERIOD_DAYS = {"1month": 30, "1year": 365}
    n_days = PERIOD_DAYS[args.period]
    dates  = all_dates[-n_days:] if len(all_dates) >= n_days else all_dates
    windows = []
    i = 0
    while i < len(dates):
        w = dates[i:i+WINDOW_DAYS]
        windows.append(w)
        i += WINDOW_DAYS
    print(f"  Windows : {len(windows)}\n")

    # Pre-compute predictions for each window (expensive, done once)
    print("  Pre-computing LGBM predictions for all windows...")
    window_data = []
    for wi, wdates in enumerate(windows):
        df = load_window(wdates)
        if df.is_empty() or len(df) < 100:
            continue
        X      = df.select(FEATURES).to_pandas().values.astype(np.float32)
        preds  = model.predict(X)
        prices = df["price"].to_numpy()
        times  = df["transact_time"].to_numpy()

        # Forward label (200ms) for PnL measurement
        df2 = df.with_columns((pl.col("transact_time")+200).alias("target_time"))
        future = (df2.select(["transact_time","price"])
                     .rename({"transact_time":"target_time","price":"future_price"})
                     .unique(subset=["target_time"],keep="last",maintain_order=True)
                     .sort("target_time"))
        df2 = df2.join_asof(future, on="target_time", strategy="backward")
        df2 = df2.with_columns(
            ((pl.col("future_price")-pl.col("price"))/pl.col("price")*10000.0)
            .alias("label_fwd_ret_bps")
        ).drop_nulls(subset=["label_fwd_ret_bps"])
        # Align preds with df2 (may have fewer rows after drop_nulls)
        n  = len(df2)
        preds_aligned  = preds[-n:]
        times_aligned  = df2["transact_time"].to_numpy()
        labels_aligned = df2["label_fwd_ret_bps"].to_numpy()
        window_data.append((f"{wdates[0]}:{wdates[-1]}", preds_aligned,
                            times_aligned, labels_aligned))
        print(f"    Window {wi+1}/{len(windows)}: {len(df2):,} ticks", flush=True)

    print(f"\n  Sweeping {n_combos} parameter combinations...")
    t0 = time.time()

    rows = []
    for threshold in grid["threshold"]:
        for fee in grid["fee"]:
            for cooldown in grid["cooldown"]:
                # Skip unprofitable-by-definition combos
                if threshold < fee * 2:
                    continue
                all_pnl, all_wr, all_pf, all_trades = [], [], [], []
                for name, preds, times, labels in window_data:
                    r = sweep_params(preds, None, times, labels,
                                     threshold, fee, cooldown)
                    if r and r["trades"] > 10:
                        all_pnl.append(r["pnl"])
                        all_wr.append(r["win"])
                        all_pf.append(r["pf"])
                        all_trades.append(r["trades"])

                if not all_pnl:
                    continue
                pnl_arr = np.array(all_pnl)
                rows.append({
                    "threshold_bps": threshold,
                    "fee_bps":       fee,
                    "cooldown_ms":   cooldown,
                    "n_windows":     len(all_pnl),
                    "profitable_pct": (np.array(all_pnl) > 0).mean() * 100,
                    "total_pnl":     pnl_arr.sum(),
                    "avg_pnl":       pnl_arr.mean(),
                    "avg_win_rate":  np.mean(all_wr),
                    "avg_pf":        np.mean(all_pf),
                    "total_trades":  sum(all_trades),
                    "sharpe_cross_windows": (pnl_arr.mean() /
                                             (pnl_arr.std() + 1e-9) *
                                             np.sqrt(len(pnl_arr))),
                })

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s\n")

    df_out = pd.DataFrame(rows).sort_values("sharpe_cross_windows", ascending=False)
    out_path = OUT_DIR / f"sensitivity_{args.period}.csv"
    df_out.to_csv(out_path, index=False)

    # Print top 20
    print("="*90)
    print(f"  TOP PARAMETER COMBINATIONS (sorted by cross-window Sharpe)")
    print("="*90)
    print(f"  {'Thresh':>7} {'Fee':>6} {'Cool':>6} | {'Win%':>6} {'PF':>6} "
          f"{'Sharpe':>8} {'Profitable%':>12} {'TotalPnL':>12}")
    print("  " + "-"*86)
    for _, r in df_out.head(20).iterrows():
        print(f"  {r.threshold_bps:>6.1f} {r.fee_bps:>6.2f} {r.cooldown_ms:>6.0f} |"
              f" {r.avg_win_rate:>6.1f} {r.avg_pf:>6.3f}"
              f" {r.sharpe_cross_windows:>8.3f} {r.profitable_pct:>11.0f}%"
              f" {r.total_pnl:>+12.0f}")

    # Break-even analysis
    print(f"\n  BREAK-EVEN FEE ANALYSIS (threshold=0.5 bps, cooldown=200ms):")
    baseline = df_out[(df_out.threshold_bps == 0.5) & (df_out.cooldown_ms == 200)]
    if not baseline.empty:
        for _, r in baseline.sort_values("fee_bps").iterrows():
            status = "✅ PROFITABLE" if r.total_pnl > 0 else "❌ LOSS"
            print(f"    Fee={r.fee_bps:.2f} bps | WR={r.avg_win_rate:.1f}% "
                  f"PF={r.avg_pf:.3f} PnL={r.total_pnl:+.0f} bps  {status}")

    print(f"\n  Saved: {out_path}")
    print("="*90)

if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
