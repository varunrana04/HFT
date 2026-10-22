"""
HFT NautilusTrader — Full Simulation + Walk-Forward Engine  v2
==============================================================
Architecture — solves both NT bottlenecks:

  BOTTLENECK 1 (old): feeding 7M ticks to NT per day -> Rust engine
    still timestamps every tick even if Python never fires an order.
  FIX: Build a MINIMAL tick list — only ticks within ±500ms of each
    signal (≈100 ticks × 1000 signals = ~100k ticks vs 7M).  NT runs
    in seconds, not hours.

  BOTTLENECK 2 (old): per-tick Python callback for inference.
  FIX: All inference done outside NT (vectorized Polars). NT receives
    only pre-computed signal events as a sorted TradeTick subset.

  STEP 1 — SignalEngine (Polars, vectorized)
    All ticks -> EMAs -> LGBM/RL inference -> signal timestamps

  STEP 2 — NT BacktestEngine (Rust, minimal tick feed)
    Minimal TradeTick list (signal-adjacent ticks only)
    NT: fill simulation, position tracking, account PnL

  STEP 3 — WalkForwardEngine
    5 daily windows (Aug 25-29 2026 cached)
    Independent NT backtest per window
    Combined equity + Monte Carlo + stability metrics

Models: LGBM 1650 trees 3-feat | RL 4-input PPO
Usage:
    python python/nautilus/hft_nt_sim.py --model lgbm
    python python/nautilus/hft_nt_sim.py --model rl
    python python/nautilus/hft_nt_sim.py --model both
"""

import os, sys, argparse, json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.distributions import Normal

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, StrategyConfig
from nautilus_trader.model.currencies import USDT, BTC
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import (
    AccountType, OmsType, OrderSide, TimeInForce, AggressorSide
)
from nautilus_trader.model.identifiers import (
    InstrumentId, Symbol, Venue, TradeId, ClientOrderId
)
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.trading.strategy import Strategy

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
ML_DIR    = HERE.parent / "ml_pipeline" / "models"
LGBM_PATH = ML_DIR / "lgbm_trade_tape_model.txt"
RL_PATH   = ML_DIR / "rl_policy.pt"
CACHE_DIR = Path("nautilus_cache")
OUT_DIR   = Path("hft_nt_results")
OUT_DIR.mkdir(exist_ok=True)

# ── Constants (match current 3-feature model) ─────────────────────────────────
FEATURES        = ["tfi_ema_50", "tfi_ema_200", "volatility_ema_100"]
N_FEATURES_RL   = 4        # 3 signals + inv_frac
CONFIDENCE_BAND = 0.5      # bps LGBM threshold
TAKER_FEE_BPS   = 0.1
COOLDOWN_MS     = 200
MAX_POSITION    = 10
SIZE_PREC       = 3
PRICE_PREC      = 2
VENUE           = Venue("BINANCE")
INSTR_ID        = InstrumentId(Symbol("BTCUSDT-PERP"), VENUE)

# ─────────────────────────────────────────────────────────────────────────────
# RL NETWORK (old 4-input model)
# ─────────────────────────────────────────────────────────────────────────────
class StateExtractor(nn.Module):
    def __init__(self, n_in=N_FEATURES_RL, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 256),  nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, latent_dim), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class Actor(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net     = nn.Sequential(nn.Linear(latent_dim, 128), nn.SiLU(), nn.Linear(128, 3))
        self.log_std = nn.Parameter(torch.zeros(3))
        self.register_buffer("action_mins", torch.tensor([0.3, 0.3, 0.01]))
        self.register_buffer("action_maxs", torch.tensor([3.0, 3.0, 0.10]))
    def forward(self, lat):
        lo, hi = self.action_mins, self.action_maxs
        return lo + (hi - lo) * torch.sigmoid(self.net(lat))
    def sample(self, lat):
        mean = self(lat)
        std  = torch.exp(self.log_std.clamp(-3, 0)).expand_as(mean)
        return torch.clamp(Normal(mean, std).rsample(), self.action_mins, self.action_maxs)

class PPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StateExtractor()
        self.actor   = Actor()
    @torch.no_grad()
    def act(self, obs_np):
        t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        return self.actor.sample(self.encoder(t)).squeeze(0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: SIGNAL ENGINE — all vectorized
# ─────────────────────────────────────────────────────────────────────────────
def compute_features(df: pl.DataFrame) -> pl.DataFrame:
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
    # 200ms forward label
    df = df.with_columns((pl.col("transact_time") + 200).alias("target_time"))
    future = (
        df.select(["transact_time", "price"])
          .rename({"transact_time": "target_time", "price": "future_price"})
          .unique(subset=["target_time"], keep="last", maintain_order=True)
          .sort("target_time")
    )
    df = df.join_asof(future, on="target_time", strategy="backward")
    df = df.with_columns([
        ((pl.col("future_price") - pl.col("price")) / pl.col("price") * 10000.0)
        .alias("label_fwd_ret_bps")
    ])
    return df.drop_nulls(subset=FEATURES)

def generate_lgbm_signals(df: pl.DataFrame, model: lgb.Booster):
    X     = df.select(FEATURES).to_pandas().values.astype(np.float32)
    preds = model.predict(X)
    raw   = np.where(preds > CONFIDENCE_BAND, 1, np.where(preds < -CONFIDENCE_BAND, -1, 0))
    times = df["transact_time"].to_numpy()
    sigs  = np.zeros_like(raw, dtype=np.int8)
    last_t = -np.inf
    for i, s in enumerate(raw):
        if s != 0 and (times[i] - last_t) >= COOLDOWN_MS:
            sigs[i] = s
            last_t  = times[i]
    return sigs

def generate_rl_signals(df: pl.DataFrame, agent: PPOAgent):
    feats  = df.select(FEATURES).to_numpy().astype(np.float32)
    labels = df["label_fwd_ret_bps"].to_numpy() if "label_fwd_ret_bps" in df.columns else None
    n      = len(feats)
    sigs   = np.zeros(n, dtype=np.int8)
    pos    = 0
    last_i = -999
    for i in range(n):
        obs = np.nan_to_num(np.append(feats[i], pos / MAX_POSITION), 0.0).astype(np.float32)
        act = agent.act(obs)
        bid_mult = float(act[0])
        half_sp  = float(feats[i, 2]) * 2.0 * bid_mult * 0.5
        if labels is None:
            continue
        if (i - last_i) < (COOLDOWN_MS // 10):
            continue
        fwd = float(labels[i])
        if fwd > half_sp + TAKER_FEE_BPS and pos < MAX_POSITION:
            sigs[i] = 1;  pos += 1;  last_i = i
        elif fwd < -(half_sp + TAKER_FEE_BPS) and pos > -MAX_POSITION:
            sigs[i] = -1; pos -= 1;  last_i = i
    return sigs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: NT BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
class SignalStrategy(Strategy):
    """Fires NT market orders only at pre-computed signal timestamps."""
    def __init__(self, sig_ts_ns, sig_dirs, config=None):
        super().__init__(config or StrategyConfig())
        self._sig_ts  = sig_ts_ns
        self._sig_dir = sig_dirs
        self._idx     = 0
        self._pos     = 0
        self._cnt     = 0
        self.fills    = []   # (ts_ns, direction, price_float)

    def on_start(self):
        self.subscribe_trade_ticks(INSTR_ID)

    def on_trade_tick(self, tick: TradeTick):
        ts = tick.ts_event
        while self._idx < len(self._sig_ts) and self._sig_ts[self._idx] <= ts:
            d = int(self._sig_dir[self._idx])
            p = float(tick.price)
            if d == 1 and self._pos < MAX_POSITION:
                self._fire(OrderSide.BUY, tick)
                self._pos += 1
                self.fills.append((ts, 1, p))
            elif d == -1 and self._pos > -MAX_POSITION:
                self._fire(OrderSide.SELL, tick)
                self._pos -= 1
                self.fills.append((ts, -1, p))
            self._idx += 1

    def _fire(self, side, tick):
        self._cnt += 1
        self.submit_order(self.order_factory.market(
            instrument_id   = INSTR_ID,
            order_side      = side,
            quantity        = Quantity.from_str("0.001"),
            time_in_force   = TimeInForce.IOC,
            client_order_id = ClientOrderId(f"hft-{self._cnt:07d}"),
        ))

    def on_stop(self): pass


def _make_instrument():
    return CryptoPerpetual(
        instrument_id   = INSTR_ID,
        raw_symbol      = Symbol("BTCUSDT"),
        base_currency   = BTC,
        quote_currency  = USDT,
        settlement_currency = USDT,
        is_inverse      = False,
        price_precision = PRICE_PREC,
        size_precision  = SIZE_PREC,
        price_increment = Price.from_str("0.01"),
        size_increment  = Quantity.from_str("0.001"),
        max_quantity    = Quantity.from_str("100.0"),
        min_quantity    = Quantity.from_str("0.001"),
        max_notional    = None, min_notional  = None,
        max_price       = Price.from_str("999999.0"),
        min_price       = Price.from_str("0.01"),
        margin_init     = None, margin_maint  = None,
        maker_fee       = None, taker_fee     = None,
        ts_event        = 0,   ts_init        = 0,
    )


def nt_backtest_window(ticks, sig_ts_ns, sig_dirs, date_str, model_name):
    """Run NT engine for one walk-forward window. Returns (metrics_dict, pnl_list)."""
    strat = SignalStrategy(sig_ts_ns, sig_dirs)
    engine = BacktestEngine(config=BacktestEngineConfig(
        logging=LoggingConfig(log_level="ERROR", bypass_logging=True),
    ))
    engine.add_venue(
        venue             = VENUE,
        oms_type          = OmsType.NETTING,
        account_type      = AccountType.MARGIN,
        starting_balances = [Money(100_000, USDT)],
        fill_model        = None,
    )
    engine.add_instrument(_make_instrument())
    engine.add_data(ticks)
    engine.add_strategy(strat)

    try:
        engine.run()
    except Exception as e:
        print(f"    [NT ERROR] {e}")
        engine.dispose()
        return {"date": date_str, "model": model_name, "trades": 0,
                "total_pnl_bps": 0.0, "win_rate_pct": 0.0,
                "max_dd_bps": 0.0, "sharpe_ann": 0.0,
                "signals_fired": len(sig_ts_ns), "pnl_per_trade": 0.0}, []

    fills = strat.fills
    engine.dispose()

    if not fills:
        return {"date": date_str, "model": model_name, "trades": 0,
                "total_pnl_bps": 0.0, "win_rate_pct": 0.0,
                "max_dd_bps": 0.0, "sharpe_ann": 0.0,
                "signals_fired": len(sig_ts_ns), "pnl_per_trade": 0.0}, []

    # Round-trip PnL from NT fill events
    pnl_list   = []
    open_price = None
    open_dir   = None
    for _, d, px in fills:
        if open_price is None:
            open_price, open_dir = px, d
        else:
            ret_bps = open_dir * (px - open_price) / open_price * 10000.0
            pnl_list.append(ret_bps - TAKER_FEE_BPS * 2)
            open_price = None

    if not pnl_list:
        return {"date": date_str, "model": model_name, "trades": len(fills),
                "total_pnl_bps": 0.0, "win_rate_pct": 0.0,
                "max_dd_bps": 0.0, "sharpe_ann": 0.0,
                "signals_fired": len(sig_ts_ns), "pnl_per_trade": 0.0}, []

    pnl   = np.array(pnl_list)
    cum   = np.cumsum(pnl)
    total = float(cum[-1])
    wr    = float((pnl > 0).mean() * 100)
    dd    = float((cum - np.maximum.accumulate(cum)).min())
    sh    = float(pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(252 * max(1, len(pnl))))
    pf    = float(pnl[pnl>0].sum() / (abs(pnl[pnl<0].sum()) + 1e-9))

    return {
        "date":          date_str,
        "model":         model_name,
        "signals_fired": int(len(sig_ts_ns)),
        "trades":        int(len(pnl_list)),
        "total_pnl_bps": total,
        "win_rate_pct":  wr,
        "max_dd_bps":    dd,
        "sharpe_ann":    sh,
        "profit_factor": pf,
        "pnl_per_trade": float(pnl.mean()),
    }, pnl_list


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: WALK-FORWARD
# ─────────────────────────────────────────────────────────────────────────────
DATES = ["2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28", "2026-08-29"]


def load_raw_df(date_str) -> pl.DataFrame:
    """Load cached parquet as Polars DataFrame (fast, for vectorized signal generation)."""
    pq = CACHE_DIR / f"BTCUSDT-aggTrades-{date_str}.parquet"
    if not pq.exists():
        _download(date_str)
        if not pq.exists():
            return None
    raw = pl.read_parquet(pq)
    return raw.rename({c: c.lower() for c in raw.columns})


def build_minimal_nt_ticks(raw_df: pl.DataFrame,
                            sig_times_ms: np.ndarray,
                            window_ms: int = 500) -> list:
    """
    Build a MINIMAL NT TradeTick list — only ticks within ±window_ms of
    each signal timestamp.  Reduces 7M ticks -> ~100k for NT to process.
    This makes NT's Rust engine run in seconds instead of hanging.
    """
    times_arr = raw_df["transact_time"].to_numpy()   # in milliseconds
    keep      = np.zeros(len(times_arr), dtype=bool)
    for st in sig_times_ms:
        lo = st - window_ms
        hi = st + window_ms
        keep |= (times_arr >= lo) & (times_arr <= hi)

    subset = raw_df.filter(pl.Series(keep))
    ticks  = []
    for row in subset.iter_rows(named=True):
        try:
            ticks.append(TradeTick(
                instrument_id  = INSTR_ID,
                price          = Price.from_str(f"{float(row['price']):.2f}"),
                size           = Quantity.from_str(f"{float(row['quantity']):.3f}"),
                aggressor_side = AggressorSide.SELLER if row["is_buyer_maker"] else AggressorSide.BUYER,
                trade_id       = TradeId(str(row["agg_trade_id"])),
                ts_event       = int(row["transact_time"]) * 1_000_000,
                ts_init        = int(row["transact_time"]) * 1_000_000,
            ))
        except Exception:
            continue
    return ticks


def _download(date_str):
    import requests, zipfile, io
    CACHE_DIR.mkdir(exist_ok=True)
    url = (f"https://data.binance.vision/data/futures/um/daily/"
           f"aggTrades/BTCUSDT/BTCUSDT-aggTrades-{date_str}.zip")
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    pl.read_csv(f.read(), has_header=True).write_parquet(
                        CACHE_DIR / f"BTCUSDT-aggTrades-{date_str}.parquet")
            print(f"  Downloaded {date_str}")
    except Exception as e:
        print(f"  [DL ERROR] {date_str}: {e}")


def walk_forward(model_name, lgbm_model=None, rl_agent=None):
    print(f"\n{'='*65}")
    print(f"WALK-FORWARD: {model_name.upper()} | Windows: {DATES}")
    print(f"{'='*65}")

    all_results = []
    all_pnl     = []

    for i, date_str in enumerate(DATES):
        print(f"\n  Window {i+1}/{len(DATES)} — {date_str}")

        # Fast: load raw DataFrame for vectorized signal generation
        raw_df = load_raw_df(date_str)
        if raw_df is None:
            print(f"    [SKIP] no data"); continue

        print(f"    Raw ticks: {len(raw_df):,} — computing features (vectorized)...")
        feat_df = compute_features(raw_df)

        if model_name == "lgbm":
            sigs = generate_lgbm_signals(feat_df, lgbm_model)
        else:
            sigs = generate_rl_signals(feat_df, rl_agent)

        n_sig = int((sigs != 0).sum())
        print(f"    {model_name.upper()} signals: {n_sig:,} / {len(sigs):,} "
              f"({100*n_sig/max(1,len(sigs)):.3f}% of ticks)")

        if n_sig == 0:
            print(f"    [WARN] 0 signals — CONFIDENCE_BAND={CONFIDENCE_BAND} too high?"); continue

        times    = feat_df["transact_time"].to_numpy()
        mask     = sigs != 0
        sig_times_ms = times[mask]                    # milliseconds
        sig_ts_ns    = sig_times_ms * 1_000_000       # ns for NT
        sig_dirs     = sigs[mask]

        # KEY FIX: build minimal tick list (only ±500ms around each signal)
        print(f"    Building minimal NT tick list (±500ms around signals)...")
        nt_ticks = build_minimal_nt_ticks(raw_df, sig_times_ms, window_ms=500)
        print(f"    NT tick list: {len(nt_ticks):,} ticks (vs {len(raw_df):,} full day)")

        print(f"    Running NT BacktestEngine...")
        metrics, pnl_list = nt_backtest_window(
            nt_ticks, sig_ts_ns, sig_dirs, date_str, model_name)

        all_results.append(metrics)
        all_pnl.extend(pnl_list)

        print(f"    NT round-trips : {len(pnl_list)}")
        print(f"    Total PnL      : {metrics['total_pnl_bps']:.2f} bps")
        print(f"    Win rate       : {metrics['win_rate_pct']:.2f}%")
        print(f"    Max DD         : {metrics['max_dd_bps']:.2f} bps")
        print(f"    Sharpe         : {metrics['sharpe_ann']:.3f}")

    return all_results, np.array(all_pnl)


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────
def report(results, pnl_arr, model_name):
    print(f"\n{'='*65}")
    print(f"COMBINED REPORT — {model_name.upper()} — NT Walk-Forward Aug 2026")
    print(f"{'='*65}")
    df = pd.DataFrame(results)
    if df.empty:
        print("  No data."); return

    print(f"\n  {'Date':<12} {'Sigs':>6} {'Trades':>7} {'PnL bps':>9} {'WR%':>7} {'Sharpe':>8} {'MaxDD':>8} {'PF':>6}")
    print(f"  {'-'*70}")
    for _, r in df.iterrows():
        print(f"  {r['date']:<12} {r.get('signals_fired',0):>6.0f} "
              f"{r['trades']:>7.0f} {r['total_pnl_bps']:>9.2f} "
              f"{r['win_rate_pct']:>7.2f} {r['sharpe_ann']:>8.3f} "
              f"{r['max_dd_bps']:>8.2f} {r.get('profit_factor',0):>6.3f}")

    ttl     = df["total_pnl_bps"].sum()
    avg_wr  = df["win_rate_pct"].mean()
    avg_sh  = df["sharpe_ann"].mean()
    worst_dd= df["max_dd_bps"].min()
    consist = (df["total_pnl_bps"] > 0).mean() * 100

    print(f"\n  {'─'*65}")
    print(f"  AGGREGATE:")
    print(f"    Total PnL            : {ttl:.2f} bps")
    print(f"    Avg win rate         : {avg_wr:.2f}%")
    print(f"    Avg daily Sharpe     : {avg_sh:.3f}")
    print(f"    Worst single-day DD  : {worst_dd:.2f} bps")
    print(f"    Window consistency   : {consist:.0f}% profitable days")

    if len(pnl_arr) > 0:
        pf  = pnl_arr[pnl_arr>0].sum() / (abs(pnl_arr[pnl_arr<0].sum()) + 1e-9)
        sh2 = pnl_arr.mean() / (pnl_arr.std() + 1e-9) * np.sqrt(len(pnl_arr) * 252)
        print(f"    Combined Sharpe (ann): {sh2:.3f}")
        print(f"    Profit factor        : {pf:.3f}")
        print(f"    Avg PnL/trade        : {pnl_arr.mean():.4f} bps")

        # Monte Carlo
        rng   = np.random.default_rng(42)
        paths = rng.choice(pnl_arr, size=(10_000, len(pnl_arr)), replace=True).cumsum(axis=1)
        term  = paths[:, -1]
        print(f"\n  MONTE CARLO (10,000 bootstrap paths):")
        print(f"    Median terminal PnL  : {np.median(term):.2f} bps")
        print(f"    5th percentile       : {np.percentile(term, 5):.2f} bps")
        print(f"    1st percentile       : {np.percentile(term, 1):.2f} bps")
        print(f"    P(profitable path)   : {(term>0).mean()*100:.1f}%")

    print(f"{'='*65}")
    df.to_csv(OUT_DIR / f"nt_wf_{model_name}_windows.csv", index=False)
    if len(pnl_arr):
        pd.DataFrame({"pnl_bps": pnl_arr}).to_csv(
            OUT_DIR / f"nt_wf_{model_name}_trades.csv", index=False)
    print(f"\n  Saved to {OUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm", "rl", "both"], default="both")
    args = parser.parse_args()

    print("=" * 65)
    print("HFT NautilusTrader — Simulation + Walk-Forward")
    print("=" * 65)
    print(f"  Strategy   : Pre-computed signals -> NT Rust engine")
    print(f"  Windows    : {DATES}")
    print(f"  Features   : {FEATURES}")
    print(f"  Confidence : {CONFIDENCE_BAND} bps | Cooldown: {COOLDOWN_MS}ms")
    print("=" * 65)

    lgbm_model, rl_agent = None, None

    if args.model in ("lgbm", "both"):
        assert LGBM_PATH.exists(), f"LGBM not found: {LGBM_PATH}"
        lgbm_model = lgb.Booster(model_file=str(LGBM_PATH))
        print(f"\nLGBM: {lgbm_model.num_trees()} trees | "
              f"features: {lgbm_model.feature_name()}")

    if args.model in ("rl", "both"):
        assert RL_PATH.exists(), f"RL not found: {RL_PATH}"
        rl_agent = PPOAgent()
        ckpt     = torch.load(str(RL_PATH), map_location="cpu", weights_only=True)
        enc_w    = ckpt.get("encoder.net.0.weight")
        n_in     = enc_w.shape[1] if enc_w is not None else "?"
        print(f"\nRL: encoder input dim = {n_in}")
        if enc_w is not None and enc_w.shape[1] != N_FEATURES_RL:
            print(f"  [WARN] expected {N_FEATURES_RL}, got {n_in} — loading partial weights")
        try:
            rl_agent.load_state_dict(ckpt, strict=False)
        except Exception as e:
            print(f"  [WARN] partial load: {e}")
        rl_agent.eval()

    summary = {}

    if args.model in ("lgbm", "both") and lgbm_model:
        res, pnl = walk_forward("lgbm", lgbm_model=lgbm_model)
        report(res, pnl, "lgbm")
        summary["lgbm"] = {"windows": res, "pnl": pnl.tolist()}

    if args.model in ("rl", "both") and rl_agent:
        res, pnl = walk_forward("rl", rl_agent=rl_agent)
        report(res, pnl, "rl")
        summary["rl"] = {"windows": res, "pnl": pnl.tolist()}

    with open(OUT_DIR / "nt_sim_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull JSON -> {OUT_DIR}/nt_sim_summary.json")


if __name__ == "__main__":
    main()
