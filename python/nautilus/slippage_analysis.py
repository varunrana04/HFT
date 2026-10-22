"""
Slippage & Market Impact Analysis — Step 3 of Stress Testing
=============================================================
Re-runs the walk-forward with realistic slippage models to find
the true break-even cost of execution.

Three slippage models:
  1. FLAT      : fixed additional cost per trade (0.1, 0.2, 0.5, 1.0 bps)
  2. ADAPTIVE  : base_fee + k * realized_vol (widens in volatile periods)
  3. IMPACT    : base_fee + sqrt(trade_size) * impact_factor (market impact)

At 1.28 bps avg PnL/trade with 0.1 bps fee, the gross edge is ~1.38 bps.
Break-even slippage = where net PnL per trade hits 0.

Usage (from HFT root):
    python python/nautilus/slippage_analysis.py --period 1month
    python python/nautilus/slippage_analysis.py --period 1year
"""

import argparse
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

FEATURES        = ["tfi_ema_50", "tfi_ema_200", "volatility_ema_100"]
CONFIDENCE_BAND = 0.5    # bps — keep fixed at optimal from sweep
COOLDOWN_MS     = 200
WINDOW_DAYS     = 7
SUBSAMPLE       = 5      # predict every Nth tick — EMA features are smooth,
                         # so subsampling barely changes signal distribution
                         # but gives 5x speedup on model.predict()

# ── Slippage models ────────────────────────────────────────────────────────────
FLAT_SLIPPAGE  = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]  # bps extra per side
BASE_FEE       = 0.1   # bps (taker fee already included in base)
VOL_COEFF      = 0.5   # adaptive: slip = BASE_FEE + VOL_COEFF * window_vol_bps
IMPACT_FACTOR  = 0.001 # impact: slip = BASE_FEE + IMPACT_FACTOR * sqrt(trade_vol)

def get_cached_dates():
    files = sorted(CACHE_DIR.glob("BTCUSDT-aggTrades-*.parquet"))
    return [date.fromisoformat(f.stem.split("-aggTrades-")[1]) for f in files]

def load_and_predict(days: list[date], model: lgb.Booster):
    """Load window, compute features, predict, return arrays."""
    frames = []
    for d in days:
        p = CACHE_DIR / f"BTCUSDT-aggTrades-{d}.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p))
    if not frames:
        return None

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
    ]).drop_nulls(subset=FEATURES)

    if len(df) < 100:
        return None

    # Predict on full data (no subsampling -- labels need full temporal resolution)
    X     = df.select(FEATURES).to_pandas().values.astype(np.float32)
    preds = model.predict(X)
    vol   = df["volatility_ema_100"].to_numpy()
    qty   = df["quantity"].to_numpy()

    # Forward label (200ms) on full data
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

    n = len(df2)
    return {
        "preds":  preds[-n:],
        "labels": df2["label_fwd_ret_bps"].to_numpy(),
        "times":  df2["transact_time"].to_numpy(),
        "vol":    vol[-n:],
        "qty":    qty[-n:],
        "window_vol_bps": float(np.diff(np.log(df["price"].to_numpy())).std() * 10000),
    }



def simulate_with_slippage(data: dict, extra_slip_bps: float,
                            slip_model: str = "flat") -> dict:
    """Run simulation with a given slippage model."""
    preds, labels, times = data["preds"], data["labels"], data["times"]
    vol, qty = data["vol"], data["qty"]
    win_vol  = data["window_vol_bps"]

    # Signal generation — vectorized cooldown (loop only over signal events)
    raw = np.where(preds > CONFIDENCE_BAND, 1,
          np.where(preds < -CONFIDENCE_BAND, -1, 0)).astype(np.int8)
    fire_idx   = np.where(raw != 0)[0]
    fire_times = times[fire_idx]
    fire_raw   = raw[fire_idx]
    keep   = np.zeros(len(fire_idx), dtype=bool)
    last_t = -np.inf
    for j in range(len(fire_idx)):
        if fire_times[j] - last_t >= COOLDOWN_MS:
            keep[j] = True
            last_t  = fire_times[j]
    valid_idx  = fire_idx[keep]
    valid_sigs = fire_raw[keep]

    pnl_list = []
    for j in range(len(valid_sigs)):
        # valid_idx[j] = tick position; labels indexed by signal rank j
        ret_bps = float(labels[valid_idx[j]]) * int(valid_sigs[j])

        # Slippage model
        if slip_model == "flat":
            slip = BASE_FEE + extra_slip_bps
        elif slip_model == "adaptive":
            slip = BASE_FEE + VOL_COEFF * win_vol * 0.01
        elif slip_model == "impact":
            slip = BASE_FEE + IMPACT_FACTOR * np.sqrt(float(qty[j])) * float(vol[j])
        else:
            slip = BASE_FEE

        net = ret_bps - slip
        pnl_list.append(net)

    if not pnl_list:
        return None
    arr = np.array(pnl_list)
    pf  = arr[arr > 0].sum() / (abs(arr[arr < 0].sum()) + 1e-9)
    # Gross edge = avg return before any fee (uses only the ticks where signals fired)
    gross = (labels[valid_idx[:len(arr)]] * valid_sigs[:len(arr)]).mean() if len(arr) else 0.0
    return {
        "trades":         len(arr),
        "pnl":            arr.sum(),
        "avg_pnl":        arr.mean(),
        "win":            (arr > 0).mean() * 100,
        "pf":             pf,
        "gross_edge_bps": float(gross),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="1month", choices=["1month", "1year"])
    args = parser.parse_args()

    print("="*70)
    print("  HFT Slippage & Market Impact Analysis")
    print("="*70)

    model = lgb.Booster(model_file=str(LGBM_PATH))
    all_dates = get_cached_dates()
    if not all_dates:
        print("[ERROR] No cache.")
        return

    PERIOD_DAYS = {"1month": 30, "1year": 365}
    n_days = PERIOD_DAYS[args.period]
    dates  = all_dates[-n_days:]
    windows = []
    i = 0
    while i < len(dates):
        windows.append(dates[i:i+WINDOW_DAYS])
        i += WINDOW_DAYS

    print(f"  Period  : {args.period} | {len(windows)} windows")
    print(f"  Baseline: threshold={CONFIDENCE_BAND} bps, fee={BASE_FEE} bps, cooldown={COOLDOWN_MS}ms\n")

    # Pre-compute predictions
    print("  Pre-computing predictions...")
    all_data = []
    for wi, wdates in enumerate(windows):
        d = load_and_predict(wdates, model)
        if d:
            all_data.append(d)
    print(f"  Loaded {len(all_data)} windows\n")

    # ── MODEL 1: FLAT SLIPPAGE SWEEP ──────────────────────────────────────
    print("="*70)
    print("  MODEL 1: FLAT SLIPPAGE SWEEP")
    print("  (Base fee=0.1 bps + extra slippage per side)")
    print("="*70)
    print(f"  {'Extra Slip':>12} {'Total Fee':>10} {'AvgPnL/trade':>14} "
          f"{'WR%':>6} {'PF':>6} {'Profitable%':>12} {'Verdict':>14}")
    print("  " + "-"*75)

    flat_rows = []
    for slip in FLAT_SLIPPAGE:
        results = [simulate_with_slippage(d, slip, "flat") for d in all_data]
        results = [r for r in results if r]
        if not results:
            continue
        avg_pnl   = np.mean([r["avg_pnl"] for r in results])
        avg_wr    = np.mean([r["win"] for r in results])
        avg_pf    = np.mean([r["pf"] for r in results])
        total_pnl = sum(r["pnl"] for r in results)
        prof_pct  = np.mean([r["pnl"] > 0 for r in results]) * 100
        total_fee = BASE_FEE + slip
        verdict   = "✅ PROFITABLE" if avg_pnl > 0 else "❌ LOSS"
        print(f"  {slip:>+11.2f} {total_fee:>10.2f} {avg_pnl:>+14.4f} "
              f"{avg_wr:>6.1f} {avg_pf:>6.3f} {prof_pct:>11.0f}%  {verdict}")
        flat_rows.append({"slip_bps": slip, "total_fee_bps": total_fee,
                          "avg_pnl_per_trade": avg_pnl, "win_rate": avg_wr,
                          "pf": avg_pf, "profitable_pct": prof_pct})

    # Break-even estimate
    pos = [r for r in flat_rows if r["avg_pnl_per_trade"] > 0]
    neg = [r for r in flat_rows if r["avg_pnl_per_trade"] <= 0]
    if pos and neg:
        be_lo = pos[-1]["slip_bps"]
        be_hi = neg[0]["slip_bps"]
        print(f"\n  ⚡ BREAK-EVEN SLIPPAGE: between +{be_lo:.2f} and +{be_hi:.2f} bps per side")
        print(f"     (Total break-even fee: {BASE_FEE + be_lo:.2f} → {BASE_FEE + be_hi:.2f} bps)")
        print(f"     Binance VIP0 taker = 0.1 bps → we have {be_lo:.2f} bps slippage budget")

    # ── MODEL 2: ADAPTIVE SLIPPAGE ────────────────────────────────────────
    print("\n" + "="*70)
    print("  MODEL 2: ADAPTIVE SLIPPAGE (fee scales with window volatility)")
    print(f"  slip = {BASE_FEE} + {VOL_COEFF} × window_realized_vol")
    print("="*70)
    results = [simulate_with_slippage(d, 0, "adaptive") for d in all_data]
    results = [r for r in results if r]
    if results:
        avg_pnl = np.mean([r["avg_pnl"] for r in results])
        avg_wr  = np.mean([r["win"] for r in results])
        avg_pf  = np.mean([r["pf"] for r in results])
        prof    = np.mean([r["pnl"] > 0 for r in results]) * 100
        print(f"  Avg PnL/trade : {avg_pnl:+.4f} bps")
        print(f"  Avg Win Rate  : {avg_wr:.1f}%")
        print(f"  Avg PF        : {avg_pf:.3f}")
        print(f"  Profitable    : {prof:.0f}% of windows")
        verdict = "✅ PROFITABLE" if avg_pnl > 0 else "❌ LOSS"
        print(f"  Verdict       : {verdict}")

    # ── MODEL 3: MARKET IMPACT ────────────────────────────────────────────
    print("\n" + "="*70)
    print("  MODEL 3: MARKET IMPACT (slip scales with sqrt(trade_volume))")
    print(f"  slip = {BASE_FEE} + {IMPACT_FACTOR} × √(qty) × vol")
    print("="*70)
    results = [simulate_with_slippage(d, 0, "impact") for d in all_data]
    results = [r for r in results if r]
    if results:
        avg_pnl = np.mean([r["avg_pnl"] for r in results])
        avg_wr  = np.mean([r["win"] for r in results])
        avg_pf  = np.mean([r["pf"] for r in results])
        prof    = np.mean([r["pnl"] > 0 for r in results]) * 100
        print(f"  Avg PnL/trade : {avg_pnl:+.4f} bps")
        print(f"  Avg Win Rate  : {avg_wr:.1f}%")
        print(f"  Avg PF        : {avg_pf:.3f}")
        print(f"  Profitable    : {prof:.0f}% of windows")
        verdict = "✅ PROFITABLE" if avg_pnl > 0 else "❌ LOSS"
        print(f"  Verdict       : {verdict}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  GROSS EDGE ESTIMATE (before all costs)")
    print("="*70)
    all_gross = []
    for d in all_data:
        preds, labels, times = d["preds"], d["labels"], d["times"]
        raw = np.where(preds > CONFIDENCE_BAND, 1,
              np.where(preds < -CONFIDENCE_BAND, -1, 0)).astype(np.int8)
        signals = np.zeros_like(raw)
        last_t  = -np.inf
        for i in range(len(raw)):
            if raw[i] != 0 and (times[i] - last_t) >= COOLDOWN_MS:
                signals[i] = raw[i]
                last_t = times[i]
        gross = labels[signals != 0] * signals[signals != 0]
        if len(gross):
            all_gross.append(gross.mean())

    if all_gross:
        gross_edge = np.mean(all_gross)
        print(f"  Gross edge (before fees) : {gross_edge:+.4f} bps/trade")
        print(f"  Base fee (Binance VIP0)  : -{BASE_FEE:.2f} bps/trade")
        print(f"  Net edge (base fee only) : {gross_edge - BASE_FEE:+.4f} bps/trade")
        print(f"  Slippage budget          : {gross_edge - BASE_FEE:.4f} bps before break-even")
        print(f"  Break-even total cost    : {gross_edge:.4f} bps/trade")

    # Save
    out_path = OUT_DIR / f"slippage_analysis_{args.period}.csv"
    pd.DataFrame(flat_rows).to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print("="*70)

if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
