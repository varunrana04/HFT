"""
Regime Analysis — Step 2 of Stress Testing
============================================
Reads the 1-year walk-forward results and classifies each weekly window
by market regime, then correlates regime type with model performance.

Answers:
  - What made Oct 4-10 the worst window (-838 bps MaxDD)?
  - What regimes does the model struggle with?
  - Is the edge regime-dependent or universal?

Usage (from HFT root):
    python python/nautilus/regime_analysis.py
    python python/nautilus/regime_analysis.py --results hft_nt_results/wf_lgbm_1year_windows.csv
"""

import argparse
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd

CACHE_DIR = Path("nautilus_cache")
OUT_DIR   = Path("hft_nt_results")

def load_price_stats(start_str: str, end_str: str) -> dict:
    """Load price data for a window and compute regime features."""
    from datetime import date, timedelta
    try:
        d0 = date.fromisoformat(start_str)
        d1 = date.fromisoformat(end_str)
    except Exception:
        return {}

    frames = []
    d = d0
    while d <= d1:
        p = CACHE_DIR / f"BTCUSDT-aggTrades-{d}.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p).select(["transact_time","price","quantity"]))
        d += timedelta(days=1)

    if not frames:
        return {}

    df = pl.concat(frames).sort("transact_time")
    prices = df["price"].to_numpy()
    n      = len(prices)

    # Compute regime features
    returns    = np.diff(np.log(prices))
    open_price = prices[0]
    close_price= prices[-1]
    price_ret  = (close_price - open_price) / open_price * 100   # % return

    # Realized volatility (annualized)
    rv         = returns.std() * np.sqrt(n * 252)

    # Price range as % of mean
    p_range    = (prices.max() - prices.min()) / prices.mean() * 100

    # Trend strength (R² of linear fit)
    x          = np.arange(n)
    coeffs     = np.polyfit(x, prices, 1)
    trend_line = np.polyval(coeffs, x)
    ss_res     = np.sum((prices - trend_line)**2)
    ss_tot     = np.sum((prices - prices.mean())**2)
    r_squared  = 1 - ss_res / (ss_tot + 1e-9)

    # Flash crash / spike detection: max single-candle move
    max_move   = np.abs(returns).max() * 10000  # bps

    # Regime label
    if rv > 2.0 and abs(price_ret) > 5:
        regime = "HIGH_VOL_TRENDING"
    elif rv > 2.0:
        regime = "HIGH_VOL_CHOPPY"
    elif abs(price_ret) > 3:
        regime = "LOW_VOL_TRENDING"
    else:
        regime = "LOW_VOL_SIDEWAYS"

    return {
        "open":      open_price,
        "close":     close_price,
        "price_ret": price_ret,
        "rv":        rv,
        "p_range":   p_range,
        "r_squared": r_squared,
        "max_move_bps": max_move,
        "regime":    regime,
        "n_ticks":   n,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="hft_nt_results/wf_lgbm_1year_windows.csv")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        # Try to find any 1-year windows CSV
        candidates = sorted(OUT_DIR.glob("wf_lgbm_1year_windows.csv"))
        if not candidates:
            print(f"[ERROR] No results file found. Run 1-year walk-forward first.")
            print(f"  Expected: {results_path}")
            return
        results_path = candidates[0]

    print("="*70)
    print("  HFT Regime Analysis")
    print("="*70)
    print(f"  Results : {results_path}")

    df = pd.read_csv(results_path)
    print(f"  Windows : {len(df)}")
    print(f"  Columns : {list(df.columns)}\n")

    # Parse date columns
    if "window" in df.columns:
        df["start"] = df["window"].str.split(":").str[0]
        df["end"]   = df["window"].str.split(":").str[1]
    elif "start_date" in df.columns:
        df["start"] = df["start_date"]
        df["end"]   = df["end_date"]
    else:
        # Try to parse from index or first column
        df["start"] = df.iloc[:, 0].str.split(":").str[0]
        df["end"]   = df.iloc[:, 0].str.split(":").str[1]

    # Load regime features for each window
    print("  Computing regime features for each window...")
    regime_rows = []
    for _, row in df.iterrows():
        stats = load_price_stats(str(row["start"]).strip(), str(row["end"]).strip())
        if stats:
            merged = dict(row) | stats
            regime_rows.append(merged)

    if not regime_rows:
        print("[ERROR] Could not load price data from cache.")
        print("  Make sure nautilus_cache/ has parquet files.")
        return

    rdf = pd.DataFrame(regime_rows)

    # Identify PnL and win rate columns — exact match first, then partial
    pnl_col = next((c for c in rdf.columns if c == "total_pnl"), None) or \
              next((c for c in rdf.columns if "pnl" in c.lower() and "pct" not in c.lower()), None)
    wr_col  = next((c for c in rdf.columns if c == "win_rate"), None) or \
              next((c for c in rdf.columns if c.startswith("win_")), None) or \
              next((c for c in rdf.columns if "win_rate" in c.lower()), None)
    dd_col  = next((c for c in rdf.columns if c == "max_dd"), None) or \
              next((c for c in rdf.columns if "max_dd" in c.lower() or "drawdown" in c.lower()), None)
    pf_col  = next((c for c in rdf.columns if c == "pf"), None) or \
              next((c for c in rdf.columns if "profit_factor" in c.lower()), None)

    print(f"\n  PnL col: {pnl_col} | WR col: {wr_col} | DD col: {dd_col}")

    # ── Regime Performance Summary ─────────────────────────────────────────
    print("\n" + "="*70)
    print("  PERFORMANCE BY REGIME")
    print("="*70)
    print(f"  {'Regime':<22} {'Windows':>7} {'AvgPnL':>10} {'AvgWR%':>8} "
          f"{'AvgPF':>7} {'WorstDD':>10}")
    print("  " + "-"*66)

    for regime in sorted(rdf["regime"].unique()):
        sub = rdf[rdf["regime"] == regime]
        avg_pnl = sub[pnl_col].mean() if pnl_col else float("nan")
        avg_wr  = sub[wr_col].mean()  if wr_col  else float("nan")
        avg_pf  = sub[pf_col].mean()  if pf_col  else float("nan")
        worst   = sub[dd_col].min()   if dd_col  else float("nan")
        print(f"  {regime:<22} {len(sub):>7} {avg_pnl:>+10.1f} {avg_wr:>8.1f} "
              f"{avg_pf:>7.3f} {worst:>10.1f}")

    # ── Worst 5 Windows ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  WORST 5 WINDOWS (by MaxDD)")
    print("="*70)
    print(f"  {'Window':<25} {'Regime':<22} {'PnL':>10} {'WR%':>6} "
          f"{'MaxDD':>8} {'RV':>6} {'PriceRet%':>10} {'MaxMove(bps)':>14}")
    print("  " + "-"*100)
    worst5 = rdf.nsmallest(5, dd_col) if dd_col else rdf.head(5)
    for _, r in worst5.iterrows():
        win_str = f"{str(r.get('start',''))[:10]}:{str(r.get('end',''))[:10]}"
        print(f"  {win_str:<25} {str(r.get('regime','?')):<22} "
              f"{r.get(pnl_col,0):>+10.1f} {r.get(wr_col,0):>6.1f} "
              f"{r.get(dd_col,0):>8.1f} {r.get('rv',0):>6.2f} "
              f"{r.get('price_ret',0):>+10.2f} {r.get('max_move_bps',0):>14.0f}")

    # ── Best 5 Windows ─────────────────────────────────────────────────────
    print("\n  BEST 5 WINDOWS (by PnL):")
    print(f"  {'Window':<25} {'Regime':<22} {'PnL':>10} {'WR%':>6} "
          f"{'MaxDD':>8} {'RV':>6} {'PriceRet%':>10}")
    print("  " + "-"*90)
    best5 = rdf.nlargest(5, pnl_col) if pnl_col else rdf.head(5)
    for _, r in best5.iterrows():
        win_str = f"{str(r.get('start',''))[:10]}:{str(r.get('end',''))[:10]}"
        print(f"  {win_str:<25} {str(r.get('regime','?')):<22} "
              f"{r.get(pnl_col,0):>+10.1f} {r.get(wr_col,0):>6.1f} "
              f"{r.get(dd_col,0):>8.1f} {r.get('rv',0):>6.2f} "
              f"{r.get('price_ret',0):>+10.2f}")

    # ── Correlation Analysis ───────────────────────────────────────────────
    if pnl_col:
        print("\n" + "="*70)
        print("  CORRELATION: REGIME FEATURES → PnL")
        print("="*70)
        feature_cols = ["rv", "p_range", "r_squared", "max_move_bps", "price_ret"]
        for fc in feature_cols:
            if fc in rdf.columns:
                corr = rdf[[fc, pnl_col]].corr().iloc[0, 1]
                bar  = "█" * int(abs(corr) * 20)
                sign = "+" if corr > 0 else "-"
                print(f"  {fc:<20}: {sign}{bar:<20} r={corr:+.3f}")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "regime_analysis.csv"
    rdf.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print("="*70)

if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
