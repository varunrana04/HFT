"""
HFT Extended Walk-Forward Simulation
=====================================
NO LOOK-AHEAD BIAS — honest OOS validation.

LGBM signal (correct):
    - Compute features from past ticks only (EMA)
    - Run model.predict(features) -> prediction in bps
    - Fire when |pred| > CONFIDENCE_BAND
    - PnL measured from ACTUAL future return (validation only, not signal)

RL signal (FIXED — was broken before):
    - Run policy.act_batch(features, inv=0) -> [bid_mult, ask_mult, gamma]
    - Signal fires when bid_mult < BID_MULT_THRESHOLD (high conviction = tight spread)
      AND inv_frac state allows entry
    - This uses ONLY current features + policy output — zero future data

Walk-forward:
    - Rolls weekly through cached data
    - Each window: 7 days OOS test (independent, like live trading)
    - Reports per-window metrics + aggregate + Monte Carlo

Usage:
    python python/nautilus/hft_extended_walkforward.py --period 1month
    python python/nautilus/hft_extended_walkforward.py --period 1year
    python python/nautilus/hft_extended_walkforward.py --period custom --start 2026-01-01 --end 2026-08-29
    python python/nautilus/hft_extended_walkforward.py --period 1month --model lgbm
    python python/nautilus/hft_extended_walkforward.py --period 1year  --model both

Run from HFT root directory.
"""

import argparse, json, sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
# torch imported lazily inside load_rl_model() -- only when --model rl/both
# This lets LGBM run even when PyTorch DLL is blocked by AppControl.

HERE      = Path(__file__).parent
ML_DIR    = HERE.parent / "ml_pipeline" / "models"
LGBM_PATH = ML_DIR / "lgbm_trade_tape_model.txt"
RL_PATH   = ML_DIR / "rl_policy.pt"
CACHE_DIR = Path("nautilus_cache")
OUT_DIR   = Path("hft_nt_results")
OUT_DIR.mkdir(exist_ok=True)

# ── Constants (match training) ─────────────────────────────────────────────────
FEATURES        = ["tfi_ema_50", "tfi_ema_200", "volatility_ema_100"]
N_FEATURES_RL   = 4          # 3 features + inv_frac
CONFIDENCE_BAND = 0.5        # bps — LGBM threshold
BID_MULT_THRESH = 0.8        # RL: bid_mult < this = high conviction long signal
                             # (policy willing to pay tight spread = confident)
TAKER_FEE_BPS   = 0.1
COOLDOWN_MS     = 200
MAX_POSITION    = 10
WINDOW_DAYS     = 7          # OOS test window size

# ─────────────────────────────────────────────────────────────────────────────
# RL NETWORK — defined lazily (torch imported only when RL is requested)
# ─────────────────────────────────────────────────────────────────────────────
def _build_rl_classes():
    """Import torch and define RL classes. Called once on first RL use."""
    import torch
    import torch.nn as nn

    class StateExtractor(nn.Module):
        def __init__(self, n_in=N_FEATURES_RL, latent_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in,256), nn.LayerNorm(256), nn.SiLU(),
                nn.Linear(256,256),  nn.LayerNorm(256), nn.SiLU(),
                nn.Linear(256,latent_dim), nn.SiLU(),
            )
        def forward(self, x): return self.net(x)

    class Actor(nn.Module):
        def __init__(self, latent_dim=128):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(latent_dim,128), nn.SiLU(), nn.Linear(128,3))
            self.log_std = nn.Parameter(torch.zeros(3))
            self.register_buffer("action_mins", torch.tensor([0.3,0.3,0.01]))
            self.register_buffer("action_maxs", torch.tensor([3.0,3.0,0.10]))
        def forward(self, lat):
            lo,hi = self.action_mins, self.action_maxs
            return lo + (hi-lo)*torch.sigmoid(self.net(lat))

    class PPOAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = StateExtractor()
            self.actor   = Actor()
        @torch.no_grad()
        def act_batch(self, obs_np: np.ndarray, chunk_size: int = 100_000) -> np.ndarray:
            """
            Chunked forward pass -- avoids OOM on large windows (16M+ ticks).
            16M x Linear(4->256) = 17GB in one shot; 100k chunks = ~100MB each.
            """
            results = []
            n = len(obs_np)
            for start in range(0, n, chunk_size):
                chunk = torch.tensor(obs_np[start:start+chunk_size], dtype=torch.float32)
                results.append(self.actor(self.encoder(chunk)).numpy())
            return np.vstack(results)

    return PPOAgent

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COMPUTATION (vectorized, no lookahead)
# ─────────────────────────────────────────────────────────────────────────────
def compute_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute features purely from past data (EMAs are causal — each value
    depends only on current and past ticks). No future data touched here.
    """
    df = df.sort("transact_time")
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
    # Forward label — used ONLY to measure realized PnL after signal fires.
    # NOT used in signal generation. Adding it here for convenience.
    df = df.with_columns((pl.col("transact_time")+200).alias("target_time"))
    future = (
        df.select(["transact_time","price"])
          .rename({"transact_time":"target_time","price":"future_price"})
          .unique(subset=["target_time"],keep="last",maintain_order=True)
          .sort("target_time")
    )
    df = df.join_asof(future, on="target_time", strategy="backward")
    df = df.with_columns([
        ((pl.col("future_price")-pl.col("price"))/pl.col("price")*10000.0)
        .alias("label_fwd_ret_bps")
    ])
    return df.drop_nulls(subset=FEATURES)

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATORS — ZERO LOOKAHEAD
# ─────────────────────────────────────────────────────────────────────────────
def lgbm_signals(df: pl.DataFrame, model: lgb.Booster) -> np.ndarray:
    """
    LGBM: predict from features only. Features are purely causal EMAs.
    Signal = model output exceeds confidence band.
    Cooldown gate applied post-hoc.
    NO FUTURE DATA USED.
    """
    X     = df.select(FEATURES).to_pandas().values.astype(np.float32)
    preds = model.predict(X)

    raw   = np.where(preds > CONFIDENCE_BAND, 1,
            np.where(preds < -CONFIDENCE_BAND, -1, 0)).astype(np.int8)

    times = df["transact_time"].to_numpy()
    out   = np.zeros_like(raw)
    last_t = -np.inf
    for i in range(len(raw)):
        if raw[i] != 0 and (times[i] - last_t) >= COOLDOWN_MS:
            out[i] = raw[i]
            last_t = times[i]
    return out, preds


def rl_signals_fixed(df: pl.DataFrame, agent) -> np.ndarray:
    """
    RL FIXED -- NO LOOK-AHEAD BIAS.

    Previous (BROKEN) version used labels[i] (future return) in the entry
    condition: `if labels[i] > half_sp` -- that is the market oracle.

    CORRECT version:
      1. Run ALL ticks through policy in ONE batch (fast, no Python loop)
         with inv_frac=0 (neutral inventory approximation for batch)
      2. Policy outputs [bid_mult, ask_mult, gamma] for each tick
      3. Signal fires based on bid_mult threshold:
            bid_mult < BID_MULT_THRESH  => LONG signal (policy wants to buy)
            ask_mult < BID_MULT_THRESH  => SHORT signal (policy wants to sell)
         This uses ONLY the policy's output -- zero future data.
      4. Cooldown and position limits enforced in NumPy state machine.

    Why bid_mult? The RL was trained to set bid_mult = how aggressively to
    bid relative to spread. Low bid_mult = willing to pay tight, means the
    policy is confident in a long move. This is the RL's "signal strength."
    """
    feats     = df.select(FEATURES).to_numpy().astype(np.float32)
    times     = df["transact_time"].to_numpy()
    n         = len(feats)

    # Build obs with inv_frac=0 (batch approximation)
    inv_zeros = np.zeros((n,1), dtype=np.float32)
    obs_batch = np.nan_to_num(np.hstack([feats, inv_zeros]), 0.0)

    print(f"    RL chunked inference: {n:,} ticks (100k chunks)...", flush=True)
    actions   = agent.act_batch(obs_batch, chunk_size=100_000)  # (N,3)

    bid_mults = actions[:, 0]  # low = aggressive long bid = long signal
    ask_mults = actions[:, 1]  # low = aggressive ask = short signal

    # Binary signals based on policy output
    long_raw  = bid_mults < BID_MULT_THRESH
    short_raw = ask_mults < BID_MULT_THRESH

    # Cooldown + position state machine in NumPy
    out   = np.zeros(n, dtype=np.int8)
    pos   = 0
    last_t = -np.inf

    for i in range(n):
        t = times[i]
        if t - last_t < COOLDOWN_MS:
            continue
        if long_raw[i] and pos < MAX_POSITION:
            out[i] = 1;  pos += 1;  last_t = t
        elif short_raw[i] and pos > -MAX_POSITION:
            out[i] = -1; pos -= 1;  last_t = t

    return out

# ─────────────────────────────────────────────────────────────────────────────
# FILL SIMULATOR (NT-equivalent, honest)
# ─────────────────────────────────────────────────────────────────────────────
def simulate_fills(df: pl.DataFrame, sigs: np.ndarray,
                   model_name: str, window_label: str):
    """
    Market order at tick price when signal fires.
    PnL computed from ACTUAL next prices (realized, not predicted).
    Taker fee deducted per leg.
    """
    if "label_fwd_ret_bps" not in df.columns:
        labels = np.zeros(len(df))
    else:
        labels = df["label_fwd_ret_bps"].to_numpy().astype(np.float32)

    prices = df["price"].cast(pl.Float32).to_numpy()
    times  = df["transact_time"].to_numpy()

    sig_idx  = np.where(sigs != 0)[0]
    sig_dirs = sigs[sig_idx]
    n_sig    = len(sig_idx)

    trades = []
    open_px, open_dir, open_t = None, None, None

    for k in range(n_sig):
        idx = sig_idx[k]
        d   = int(sig_dirs[k])
        px  = float(prices[idx])
        t   = int(times[idx])

        if open_px is None:
            open_px, open_dir, open_t = px, d, t
        elif d == -open_dir:
            # Reversal closes position
            ret_bps = open_dir * (px - open_px) / open_px * 10000.0
            trades.append({
                "pnl_bps":   ret_bps - TAKER_FEE_BPS * 2,
                "direction": open_dir,
                "entry_px":  open_px,
                "exit_px":   px,
                "hold_ms":   t - open_t,
            })
            open_px, open_dir, open_t = None, None, None

    if not trades:
        return {
            "window": window_label, "model": model_name,
            "signals": n_sig, "trades": 0,
            "total_pnl": 0.0, "win_rate": 0.0,
            "max_dd": 0.0, "sharpe": 0.0, "pf": 0.0,
            "avg_hold_ms": 0.0,
        }, []

    pnl      = np.array([t["pnl_bps"] for t in trades])
    hold_ms  = np.array([t["hold_ms"]  for t in trades])
    cum      = np.cumsum(pnl)
    wr       = float((pnl > 0).mean() * 100)
    dd       = float((cum - np.maximum.accumulate(cum)).min())
    sh       = float(pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(len(pnl) * 252))
    pf       = float(pnl[pnl>0].sum() / (abs(pnl[pnl<0].sum()) + 1e-9))

    return {
        "window":     window_label,
        "model":      model_name,
        "signals":    int(n_sig),
        "trades":     int(len(pnl)),
        "total_pnl":  float(cum[-1]),
        "win_rate":   wr,
        "max_dd":     dd,
        "sharpe":     sh,
        "pf":         pf,
        "avg_hold_ms": float(hold_ms.mean()),
    }, pnl.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_window(dates: list) -> pl.DataFrame:
    """Load and concatenate multiple days from cache."""
    frames = []
    for d in dates:
        pq = CACHE_DIR / f"BTCUSDT-aggTrades-{d}.parquet"
        if pq.exists():
            raw = pl.read_parquet(pq)
            raw = raw.rename({c: c.lower() for c in raw.columns})
            frames.append(raw)
    if not frames:
        return None
    df = pl.concat(frames).sort("transact_time")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def get_cached_dates() -> list:
    """Return sorted list of cached date strings."""
    files = sorted(CACHE_DIR.glob("BTCUSDT-aggTrades-*.parquet"))
    return [f.stem.replace("BTCUSDT-aggTrades-", "") for f in files]


def walk_forward(model_name: str, period: str,
                 start_override: str = None, end_override: str = None,
                 lgbm_model=None, rl_agent=None):
    all_dates = get_cached_dates()
    if not all_dates:
        print("[ERROR] No cached data found. Run download_walkforward_data.py first.")
        return [], np.array([])

    today = date.today()

    if period == "1month":
        cutoff = str(today - timedelta(days=30))
        window_dates = [d for d in all_dates if d >= cutoff]
    elif period == "1year":
        cutoff = str(today - timedelta(days=365))
        window_dates = [d for d in all_dates if d >= cutoff]
    elif period == "custom":
        window_dates = [d for d in all_dates
                        if start_override <= d <= end_override]
    else:
        window_dates = all_dates

    if not window_dates:
        print(f"[WARN] No cached data in range for period={period}.")
        print(f"  Cached: {all_dates[0]} to {all_dates[-1]}")
        print(f"  Run: python python/nautilus/download_walkforward_data.py --period {period}")
        return [], np.array([])

    print(f"\n{'='*65}")
    print(f"WALK-FORWARD: {model_name.upper()} | Period: {period}")
    print(f"Available days: {len(window_dates)} ({window_dates[0]} to {window_dates[-1]})")
    print(f"Window size: {WINDOW_DAYS} days | Step: {WINDOW_DAYS} days (non-overlapping)")
    print(f"{'='*65}")

    # Chunk dates into WINDOW_DAYS-size OOS windows
    windows = []
    for i in range(0, len(window_dates), WINDOW_DAYS):
        chunk = window_dates[i:i+WINDOW_DAYS]
        if chunk:
            windows.append(chunk)

    print(f"Total OOS windows: {len(windows)}")

    all_results = []
    all_pnl     = []

    for wi, w_dates in enumerate(windows):
        label = f"{w_dates[0]}:{w_dates[-1]}"
        n_days = len(w_dates)
        print(f"\n  Window {wi+1}/{len(windows)} [{label}] ({n_days} days)")

        raw_df = load_window(w_dates)
        if raw_df is None:
            print(f"    [SKIP] no data"); continue

        print(f"    Loading features for {len(raw_df):,} ticks...", flush=True)
        df = compute_features(raw_df)
        print(f"    Usable rows: {len(df):,}", flush=True)

        if len(df) < 1000:
            print(f"    [SKIP] too few rows"); continue

        if model_name == "lgbm":
            sigs, preds = lgbm_signals(df, lgbm_model)
            n_sig = int((sigs != 0).sum())
            print(f"    LGBM signals: {n_sig:,} "
                  f"| pred range [{preds.min():.2f}, {preds.max():.2f}] bps", flush=True)
        else:
            sigs  = rl_signals_fixed(df, rl_agent)
            n_sig = int((sigs != 0).sum())
            print(f"    RL signals: {n_sig:,}", flush=True)

        if n_sig == 0:
            print(f"    [WARN] 0 signals in this window")
            all_results.append({
                "window":label,"model":model_name,"signals":0,"trades":0,
                "total_pnl":0.0,"win_rate":0.0,"max_dd":0.0,
                "sharpe":0.0,"pf":0.0,"avg_hold_ms":0.0,
            })
            continue

        metrics, pnl_list = simulate_fills(df, sigs, model_name, label)
        all_results.append(metrics)
        all_pnl.extend(pnl_list)

        print(f"    Trades:  {metrics['trades']} round-trips")
        print(f"    PnL:     {metrics['total_pnl']:+.2f} bps")
        print(f"    Win:     {metrics['win_rate']:.1f}%")
        print(f"    MaxDD:   {metrics['max_dd']:.2f} bps")
        print(f"    Sharpe:  {metrics['sharpe']:.3f}")
        print(f"    PF:      {metrics['pf']:.3f}", flush=True)

    return all_results, np.array(all_pnl)

# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
def report(results, pnl_arr, model_name, period):
    df = pd.DataFrame(results)
    tag = f"{model_name}_{period}"

    print()
    print("="*70)
    print(f"COMBINED REPORT -- {model_name.upper()} | {period}")
    print(f"Threshold={CONFIDENCE_BAND}bps | Fee={TAKER_FEE_BPS}bps | Cooldown={COOLDOWN_MS}ms")
    print("="*70)

    if df.empty or len(pnl_arr) == 0:
        print("  No trades to report.")
        return {}

    print(f"  {'Window':<26} {'Sigs':>6} {'RT':>5} {'PnL':>9} "
          f"{'WR%':>7} {'Sharpe':>8} {'MaxDD':>8} {'PF':>6}")
    print("  " + "-"*74)
    for _, r in df.iterrows():
        pnl_str = f"{r['total_pnl']:+.1f}" if abs(r['total_pnl']) < 100000 else f"{r['total_pnl']/1000:+.1f}k"
        print(f"  {r['window']:<26} {r.get('signals',0):>6.0f} "
              f"{r['trades']:>5.0f} {pnl_str:>9} "
              f"{r['win_rate']:>7.1f} {r['sharpe']:>8.3f} "
              f"{r['max_dd']:>8.2f} {r.get('pf',0):>6.3f}")

    ttl   = df["total_pnl"].sum()
    wr    = df["win_rate"].mean()
    cons  = (df["total_pnl"] > 0).mean() * 100
    sh    = pnl_arr.mean() / (pnl_arr.std() + 1e-9) * np.sqrt(len(pnl_arr) * 252)
    pf    = pnl_arr[pnl_arr>0].sum() / (abs(pnl_arr[pnl_arr<0].sum()) + 1e-9)
    dd    = df["max_dd"].min()
    n_win = (df["total_pnl"] > 0).sum()
    n_tot = len(df[df["trades"] > 0])

    print("  " + "="*74)
    print(f"  AGGREGATE:")
    print(f"    Profitable windows   : {n_win}/{n_tot} ({cons:.0f}%)")
    print(f"    Total trades         : {len(pnl_arr):,}")
    print(f"    Total PnL            : {ttl:+,.2f} bps")
    print(f"    Avg win rate         : {wr:.1f}%")
    print(f"    Combined Sharpe(ann) : {sh:.3f}")
    print(f"    Profit factor        : {pf:.3f}")
    print(f"    Worst window MaxDD   : {dd:.2f} bps")
    print(f"    Avg PnL / trade      : {pnl_arr.mean():+.4f} bps")

    # Calmar: annualised return / max DD
    # Estimate ann return as (total_pnl / n_windows / 7) * 252 * 10000 bps/yr
    # (rough — for info only)
    n_windows_traded = max(1, n_tot)
    avg_daily_bps    = ttl / max(1, n_windows_traded * WINDOW_DAYS)
    ann_bps          = avg_daily_bps * 252
    calmar           = ann_bps / (abs(dd) + 1e-9)
    print(f"    Est ann PnL (bps/yr) : {ann_bps:+,.0f}")
    print(f"    Calmar ratio (ann)   : {calmar:.2f}")

    # Monte Carlo — memory-safe for large trade counts
    # 517k trades x 10k paths = 38 GB OOM. Cap at 10k trades/path, scale result.
    rng        = np.random.default_rng(42)
    N_PATHS    = 10_000
    MAX_SAMPLE = 10_000                          # max trades per MC path
    n_total    = len(pnl_arr)
    n_sample   = min(n_total, MAX_SAMPLE)
    scale      = n_total / n_sample              # scale factor for full PnL estimate

    # Sample n_sample trades per path (with replacement from the real PnL distribution)
    paths = rng.choice(pnl_arr, size=(N_PATHS, n_sample), replace=True).cumsum(axis=1)
    term  = paths[:, -1] * scale                 # scale terminal value to full count
    pct_pos = (term > 0).mean() * 100
    print(f"\n  MONTE CARLO ({N_PATHS:,} paths x {n_sample:,} trades, scaled x{scale:.1f}):")
    print(f"    Median terminal PnL  : {np.median(term):+,.2f} bps")
    print(f"    5th percentile       : {np.percentile(term,5):+,.2f} bps")
    print(f"    1st percentile       : {np.percentile(term,1):+,.2f} bps")
    print(f"    P(profitable path)   : {pct_pos:.1f}%")
    print("="*70)


    # Save
    df.to_csv(OUT_DIR / f"wf_{tag}_windows.csv", index=False)
    pd.DataFrame({"pnl_bps": pnl_arr}).to_csv(
        OUT_DIR / f"wf_{tag}_trades.csv", index=False)

    summary = {
        "model": model_name, "period": period,
        "n_windows": n_tot, "profitable_windows_pct": cons,
        "total_trades": len(pnl_arr),
        "total_pnl_bps": float(ttl),
        "avg_win_rate": float(wr),
        "sharpe_ann": float(sh),
        "profit_factor": float(pf),
        "worst_dd_bps": float(dd),
        "est_ann_pnl_bps": float(ann_bps),
        "calmar": float(calmar),
        "mc_median_bps": float(np.median(term)),
        "mc_5th_pct": float(np.percentile(term, 5)),
        "mc_p_profitable": float(pct_pos),
    }
    with open(OUT_DIR / f"wf_{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {OUT_DIR}/wf_{tag}_*.csv + summary.json")
    return summary

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=["lgbm","rl","both"], default="both")
    parser.add_argument("--period", choices=["1month","1year","custom"], default="1month")
    parser.add_argument("--start",  help="Start date YYYY-MM-DD (custom period)")
    parser.add_argument("--end",    help="End date YYYY-MM-DD (custom period)")
    args = parser.parse_args()

    print("="*70)
    print("HFT Extended Walk-Forward -- Honest OOS (no look-ahead bias)")
    print("="*70)
    print(f"  Model    : {args.model}")
    print(f"  Period   : {args.period}")
    print(f"  Window   : {WINDOW_DAYS} days OOS, step weekly")
    print(f"  Threshold: {CONFIDENCE_BAND} bps (LGBM) | BidMult<{BID_MULT_THRESH} (RL)")
    print(f"  Fee      : {TAKER_FEE_BPS} bps taker | Cooldown: {COOLDOWN_MS}ms")
    print("="*70)

    # Check cache
    all_dates = get_cached_dates()
    if all_dates:
        print(f"\n  Cache: {len(all_dates)} days ({all_dates[0]} to {all_dates[-1]})")
    else:
        print(f"\n  [ERROR] No data cached.")
        print(f"  Run first: python python/nautilus/download_walkforward_data.py --period {args.period}")
        sys.exit(1)

    lgbm_model = None
    rl_agent   = None

    if args.model in ("lgbm", "both"):
        if not LGBM_PATH.exists():
            print(f"[ERROR] LGBM model not found: {LGBM_PATH}")
        else:
            lgbm_model = lgb.Booster(model_file=str(LGBM_PATH))
            print(f"\nLGBM: {lgbm_model.num_trees()} trees | {lgbm_model.feature_name()}")

    if args.model in ("rl", "both"):
        if not RL_PATH.exists():
            print(f"[ERROR] RL policy not found: {RL_PATH}")
        else:
            try:
                import torch
                PPOAgent = _build_rl_classes()
                rl_agent = PPOAgent()
                ckpt     = torch.load(str(RL_PATH), map_location="cpu", weights_only=True)
                enc_w    = ckpt.get("encoder.net.0.weight")
                n_in     = enc_w.shape[1] if enc_w is not None else "?"
                print(f"\nRL: encoder input dim = {n_in}")
                if enc_w is not None and enc_w.shape[1] != N_FEATURES_RL:
                    print(f"  [WARN] Architecture mismatch: expected {N_FEATURES_RL}, got {n_in}")
                try:
                    rl_agent.load_state_dict(ckpt, strict=False)
                except Exception as e:
                    print(f"  [WARN] partial load: {e}")
                rl_agent.eval()
            except OSError as e:
                print(f"\n  [WARN] Cannot load PyTorch (AppControl policy blocks torch.dll).")
                print(f"  [WARN] Skipping RL model. Run --model lgbm only on this machine.")
                print(f"  [WARN] RL requires Kaggle/Linux to run without AppControl.")
                rl_agent = None

    all_summaries = {}

    if lgbm_model:
        res, pnl = walk_forward(
            "lgbm", args.period,
            start_override=args.start, end_override=args.end,
            lgbm_model=lgbm_model)
        s = report(res, pnl, "lgbm", args.period)
        all_summaries["lgbm"] = s

    if rl_agent:
        res, pnl = walk_forward(
            "rl", args.period,
            start_override=args.start, end_override=args.end,
            rl_agent=rl_agent)
        s = report(res, pnl, "rl", args.period)
        all_summaries["rl"] = s

    # Combined summary
    out_path = OUT_DIR / f"wf_{args.period}_combined.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll results saved to {OUT_DIR}/")
    print(f"Combined summary: {out_path}")


if __name__ == "__main__":
    main()
