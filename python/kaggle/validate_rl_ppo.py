"""
HFT RL PPO — Final Validation & Stress-Test (v4, L2 bookTicker)
================================================================
Requires: rl_policy.pt at /kaggle/working/models/
Downloads aggTrades + bookTicker (L2) for OOS period Aug 2026.
State: 8 L2 features + inv_frac = 9 inputs (must match train_rl_ppo.py v4).
Exports charts + metrics to hft_rl_validation.zip.

Paste entire cell into Kaggle and run. GPU optional.
"""

import os, io, requests, zipfile, gc, warnings, json, shutil
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.distributions import Normal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

MODEL_DIR  = os.path.join(BASE_DIR, "models")
EXPORT_DIR = os.path.join(BASE_DIR, "hft_rl_validation")
BASE_URL   = "https://data.binance.vision/data/futures/um/daily"
POLICY_PATH = os.path.join(MODEL_DIR, "rl_policy.pt")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

PRETRAINED = "/kaggle/input/models/varunrana3104/rl-dl-hft-model/other/default/1/rl_policy.pt"
if not os.path.exists(POLICY_PATH) and os.path.exists(PRETRAINED):
    shutil.copy(PRETRAINED, POLICY_PATH)
    print(f"Seeded from checkpoint: {POLICY_PATH}")

# ── Constants (must match train_rl_ppo.py v4) ─────────────────────────────────
FEATURES = [
    "tfi_ema_50", "tfi_ema_200",
    "volatility_ema_100", "volatility_ema_500",
    "book_imbalance", "spread_bps", "microprice", "spread_ema",
]
TARGET         = "label_fwd_ret_bps"
TAKER_FEE_BPS  = 0.1
MAX_POSITION   = 10
COOLDOWN_STEPS = 3
N_FEATURES     = 9   # 8 L2 signals + inv_frac
N_SIM          = 5_000

OOS_DATES = [
    "2026-08-25", "2026-08-26", "2026-08-27",
    "2026-08-28", "2026-08-29",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# NETWORK (identical to train_rl_ppo.py v4)
# ─────────────────────────────────────────────────────────────────────────────
class StateExtractor(nn.Module):
    def __init__(self, n_in=N_FEATURES, latent_dim=128):
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
    def forward(self, latent):
        lo, hi = self.action_mins, self.action_maxs
        mean   = lo + (hi - lo) * torch.sigmoid(self.net(latent))
        std    = torch.exp(self.log_std.clamp(-3, 0)).expand_as(mean)
        return mean, std
    def sample(self, latent):
        mean, std = self(latent)
        raw = Normal(mean, std).rsample()
        return torch.clamp(raw, self.action_mins, self.action_maxs)

class PPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StateExtractor()
        self.actor   = Actor()
    @torch.no_grad()
    def act(self, obs_np):
        t = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return self.actor.sample(self.encoder(t)).squeeze(0).cpu().numpy()

# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE (same as training)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_zip(url, timeout=60):
    r = requests.get(url, timeout=timeout)
    return r.content if r.status_code == 200 else None

def _process_day(symbol, date_str):
    trades_url = f"{BASE_URL}/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
    book_url   = f"{BASE_URL}/bookTicker/{symbol}/{symbol}-bookTicker-{date_str}.zip"
    with ThreadPoolExecutor(max_workers=2) as ex:
        trades_bytes = ex.submit(_fetch_zip, trades_url).result()
        book_bytes   = ex.submit(_fetch_zip, book_url).result()
    if trades_bytes is None:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(trades_bytes)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pl.read_csv(f.read(), has_header=True)
        df = df.rename({c: c.lower() for c in df.columns})
        df = df.with_columns(pl.col("price").cast(pl.Float32)).sort("transact_time")
        df = df.with_columns([
            pl.when(pl.col("is_buyer_maker"))
              .then(-pl.col("quantity")).otherwise(pl.col("quantity"))
              .alias("signed_volume"),
        ])
        df = df.with_columns([
            pl.col("signed_volume").ewm_mean(half_life=50).alias("tfi_ema_50"),
            pl.col("signed_volume").ewm_mean(half_life=200).alias("tfi_ema_200"),
            pl.col("price").diff().abs().ewm_mean(half_life=100).alias("volatility_ema_100"),
            pl.col("price").diff().abs().ewm_mean(half_life=500).alias("volatility_ema_500"),
        ])
        if book_bytes is not None:
            with zipfile.ZipFile(io.BytesIO(book_bytes)) as z:
                with z.open(z.namelist()[0]) as f:
                    book = pl.read_csv(f.read(), has_header=True)
            book = book.rename({c: c.lower() for c in book.columns})
            ts_col = "transaction_time" if "transaction_time" in book.columns else "event_time"
            book = book.select([
                pl.col(ts_col).alias("transact_time"),
                pl.col("best_bid_price").cast(pl.Float32).alias("bid_px"),
                pl.col("best_bid_qty").cast(pl.Float32).alias("bid_qty"),
                pl.col("best_ask_price").cast(pl.Float32).alias("ask_px"),
                pl.col("best_ask_qty").cast(pl.Float32).alias("ask_qty"),
            ]).sort("transact_time")
            df = df.join_asof(book, on="transact_time", strategy="backward")
            df = df.with_columns([((pl.col("bid_px")+pl.col("ask_px"))/2.0).alias("mid_px")])
            df = df.with_columns([
                ((pl.col("ask_px")-pl.col("bid_px"))/(pl.col("mid_px")+1e-9)*10000.0).alias("spread_bps"),
                ((pl.col("bid_qty")-pl.col("ask_qty"))/(pl.col("bid_qty")+pl.col("ask_qty")+1e-9)).alias("book_imbalance"),
                ((pl.col("bid_px")*pl.col("ask_qty")+pl.col("ask_px")*pl.col("bid_qty"))/(pl.col("bid_qty")+pl.col("ask_qty")+1e-9)).alias("microprice"),
            ])
            df = df.with_columns([pl.col("spread_bps").ewm_mean(half_life=50).alias("spread_ema")])
            df = df.drop(["bid_px","bid_qty","ask_px","ask_qty","mid_px"])
        else:
            df = df.with_columns([pl.col("signed_volume").abs().ewm_mean(half_life=50).alias("_abs_sv")])
            df = df.with_columns([
                (pl.col("signed_volume")/(pl.col("_abs_sv")+1e-9)).alias("book_imbalance"),
                (pl.col("volatility_ema_100")*2.0).alias("spread_bps"),
                pl.col("price").alias("microprice"),
            ])
            df = df.with_columns([pl.col("spread_bps").ewm_mean(half_life=50).alias("spread_ema")])
            df = df.drop("_abs_sv")
        # 200ms label
        df = df.with_columns((pl.col("transact_time")+200).alias("target_time"))
        future = (df.select(["transact_time","price"])
                    .rename({"transact_time":"target_time","price":"future_price"})
                    .unique(subset=["target_time"],keep="last",maintain_order=True)
                    .sort("target_time"))
        df = df.join_asof(future, on="target_time", strategy="backward")
        df = df.with_columns([
            ((pl.col("future_price")-pl.col("price"))/pl.col("price")*10000.0).alias("label_fwd_ret_bps")
        ])
        df = df.drop_nulls(subset=["label_fwd_ret_bps"])
        out = df.select(FEATURES + [TARGET])
        return out.collect().to_pandas() if hasattr(out, "collect") else out.to_pandas()
    except Exception as e:
        print(f"  [ERROR] {date_str}: {e}")
        return None

def load_oos_data():
    frames = []
    for d in OOS_DATES:
        print(f"  {d}...", end=" ", flush=True)
        f = _process_day("BTCUSDT", d)
        if f is not None:
            frames.append(f)
            print(f"{len(f):,} rows")
        else:
            print("skipped")
    df = pd.concat(frames, ignore_index=True)
    print(f"Total OOS: {len(df):,} rows")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# SEQUENTIAL BACKTEST WITH COOLDOWN
# ─────────────────────────────────────────────────────────────────────────────
def run_rl_backtest(df, agent):
    features_arr = df[FEATURES].values.astype(np.float32)
    labels       = df[TARGET].values.astype(np.float32)
    n            = len(features_arr)

    position       = 0
    last_fill_step = -COOLDOWN_STEPS - 1
    trades = []

    for i in range(n):
        inv_frac = position / MAX_POSITION
        obs = np.append(features_arr[i], inv_frac)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        actions = agent.act(obs)
        bid_spread_mult = float(actions[0])
        ask_spread_mult = float(actions[1])
        gamma           = float(actions[2])

        spread_bps_now = float(features_arr[i, FEATURES.index("spread_bps")])
        half_spread    = spread_bps_now * bid_spread_mult * 0.5
        fwd_ret        = float(labels[i])

        on_cooldown = (i - last_fill_step) < COOLDOWN_STEPS
        if on_cooldown:
            continue

        bid_filled = fwd_ret >  half_spread + TAKER_FEE_BPS
        ask_filled = fwd_ret < -(half_spread + TAKER_FEE_BPS)

        if bid_filled and position < MAX_POSITION:
            pnl = fwd_ret - TAKER_FEE_BPS
            position      += 1
            last_fill_step = i
            trades.append({"step":i, "dir":1, "pnl_bps":pnl,
                           "bid_mult":bid_spread_mult, "ask_mult":ask_spread_mult,
                           "gamma":gamma, "spread":spread_bps_now})
        elif ask_filled and position > -MAX_POSITION:
            pnl = -fwd_ret - TAKER_FEE_BPS
            position      -= 1
            last_fill_step = i
            trades.append({"step":i, "dir":-1, "pnl_bps":pnl,
                           "bid_mult":bid_spread_mult, "ask_mult":ask_spread_mult,
                           "gamma":gamma, "spread":spread_bps_now})

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def chart_equity_drawdown(trade_df):
    cum = trade_df["pnl_bps"].cumsum()
    dd  = cum - cum.cummax()
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["Cumulative PnL (bps)", "Drawdown (bps)"],
                        shared_xaxes=True)
    fig.add_trace(go.Scatter(y=cum.values, line=dict(color="#00d4ff", width=1.5),
                             name="PnL"), row=1, col=1)
    fig.add_trace(go.Scatter(y=dd.values, fill="tozeroy",
                             line=dict(color="#ff4444"), name="Drawdown"), row=2, col=1)
    fig.update_layout(template="plotly_dark", title="RL v4 L2 — OOS Equity", height=700)
    return fig

def chart_monte_carlo(trade_df):
    pnl_vec = trade_df["pnl_bps"].values
    n       = len(pnl_vec)
    rng     = np.random.default_rng(42)
    paths   = rng.choice(pnl_vec, size=(N_SIM, n), replace=True).cumsum(axis=1)
    fig = go.Figure()
    for p in paths[::max(1, N_SIM//200)]:
        fig.add_trace(go.Scatter(y=p, mode="lines",
                                 line=dict(width=0.3, color="rgba(0,212,255,0.06)"),
                                 showlegend=False))
    for q, nm, col in [(0.01,"1st pct","red"),(0.05,"5th pct","orange"),(0.50,"Median","lime")]:
        fig.add_trace(go.Scatter(y=np.quantile(paths,q,axis=0),
                                 line=dict(width=2,color=col),name=nm))
    fig.update_layout(template="plotly_dark", title=f"Monte Carlo ({N_SIM:,} paths)", height=600)
    return fig, paths

def chart_action_distributions(trade_df):
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Bid Spread Mult", "Ask Spread Mult", "Gamma"])
    fig.add_trace(go.Histogram(x=trade_df["bid_mult"], marker_color="#00d4ff", nbinsx=50), row=1, col=1)
    fig.add_trace(go.Histogram(x=trade_df["ask_mult"], marker_color="#ff9900", nbinsx=50), row=1, col=2)
    fig.add_trace(go.Histogram(x=trade_df["gamma"],    marker_color="#ff4444", nbinsx=50), row=1, col=3)
    fig.update_layout(template="plotly_dark", title="RL Action Distributions", height=400)
    return fig

def chart_pnl_heatmap(trade_df):
    df2 = trade_df.copy()
    df2["spread_bin"] = pd.cut(df2["spread"].clip(0, 5), bins=8)
    h = df2.groupby("spread_bin")["pnl_bps"].mean().reset_index()
    fig = go.Figure(go.Bar(x=[str(b) for b in h["spread_bin"]], y=h["pnl_bps"].values,
                            marker_color="#9b59b6"))
    fig.update_layout(template="plotly_dark",
                      title="Avg PnL by Spread Bucket", height=400,
                      xaxis_title="spread_bps bucket", yaxis_title="Avg PnL (bps)")
    return fig

def chart_gamma_vs_skew(trade_df):
    fig = go.Figure(go.Scatter(
        x=trade_df["gamma"],
        y=trade_df["pnl_bps"],
        mode="markers",
        marker=dict(size=3, color=trade_df["pnl_bps"],
                    colorscale="RdYlGn", showscale=True),
    ))
    fig.update_layout(template="plotly_dark",
                      title="Gamma vs PnL", height=450,
                      xaxis_title="gamma", yaxis_title="trade PnL (bps)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HFT RL v4 — OOS Validation (L2 bookTicker Features)")
    print(f"State: {N_FEATURES} inputs = {len(FEATURES)} L2 signals + inv_frac")
    print(f"Cooldown: {COOLDOWN_STEPS} steps | Device: {DEVICE}")
    print("=" * 60)

    # Load policy
    assert os.path.exists(POLICY_PATH), f"Policy not found: {POLICY_PATH}"
    agent = PPOAgent().to(DEVICE)
    ckpt  = torch.load(POLICY_PATH, map_location=DEVICE, weights_only=True)
    enc_w = ckpt.get("encoder.net.0.weight", None)
    if enc_w is not None and enc_w.shape[1] != N_FEATURES:
        raise ValueError(
            f"Policy has {enc_w.shape[1]} input features, expected {N_FEATURES}.\n"
            f"Retrain with train_rl_ppo.py v4."
        )
    agent.load_state_dict(ckpt, strict=False)
    agent.eval()
    print(f"Policy loaded from {POLICY_PATH}")

    # Load OOS data
    print("\nLoading OOS data...")
    oos_df = load_oos_data()

    # Run backtest
    print("\nRunning sequential backtest with cooldown gate...")
    trade_df = run_rl_backtest(oos_df, agent)

    if len(trade_df) == 0:
        raise RuntimeError("No trades executed — check policy or data.")

    pnl_vec   = trade_df["pnl_bps"].values
    total_pnl = pnl_vec.sum()
    win_rate  = (pnl_vec > 0).mean() * 100
    max_dd    = (trade_df["pnl_bps"].cumsum() - trade_df["pnl_bps"].cumsum().cummax()).min()
    sharpe    = (pnl_vec.mean() / (pnl_vec.std() + 1e-9)) * np.sqrt(len(pnl_vec))
    pf        = pnl_vec[pnl_vec>0].sum() / (abs(pnl_vec[pnl_vec<0].sum()) + 1e-9)

    print(f"\n{'='*60}")
    print(f"RESULTS — RL v4 L2  (OOS Aug 2026)")
    print(f"{'='*60}")
    print(f"  OOS rows         : {len(oos_df):,}")
    print(f"  Trades           : {len(trade_df):,}")
    print(f"  Win rate         : {win_rate:.2f}%")
    print(f"  Total PnL        : {total_pnl:,.2f} bps")
    print(f"  Avg PnL/trade    : {pnl_vec.mean():.4f} bps")
    print(f"  Max drawdown     : {max_dd:,.2f} bps")
    print(f"  Sharpe           : {sharpe:.3f}")
    print(f"  Profit factor    : {pf:.3f}")
    print(f"{'='*60}")

    # Monte Carlo
    print(f"\nRunning Monte Carlo ({N_SIM:,} paths)...")
    mc_fig, mc_paths = chart_monte_carlo(trade_df)
    terminal = mc_paths[:, -1]
    print(f"  Median: {np.median(terminal):,.2f} | 5th pct: {np.percentile(terminal,5):,.2f}")

    # Charts
    print("\nExporting charts...")
    charts = {
        "01_equity_and_drawdown":      chart_equity_drawdown(trade_df),
        "02_monte_carlo_cone":         mc_fig,
        "03_drawdown_distribution":    go.Figure(go.Histogram(x=pnl_vec, nbinsx=100,
                                           marker_color="#00d4ff")).update_layout(
                                               template="plotly_dark", title="Trade PnL Dist", height=450),
        "04_terminal_pnl_distribution": go.Figure(go.Histogram(x=terminal, nbinsx=100,
                                           marker_color="#ff9900")).update_layout(
                                               template="plotly_dark", title="MC Terminal PnL", height=450),
        "05_action_distributions":     chart_action_distributions(trade_df),
        "06_pnl_heatmap":              chart_pnl_heatmap(trade_df),
        "07_gamma_vs_pnl":             chart_gamma_vs_skew(trade_df),
    }
    for name, fig in charts.items():
        out = os.path.join(EXPORT_DIR, f"{name}.html")
        fig.write_html(out)
        print(f"  {out}")

    trade_df.to_csv(os.path.join(EXPORT_DIR, "oos_backtest_results.csv"), index=False)
    pd.DataFrame(mc_paths).to_csv(os.path.join(EXPORT_DIR, "monte_carlo_paths.csv"), index=False)

    metrics = {
        "total_trades":   len(trade_df),
        "total_pnl_bps":  total_pnl,
        "win_rate_pct":   win_rate,
        "max_dd_bps":     float(max_dd),
        "sharpe":         float(sharpe),
        "profit_factor":  float(pf),
        "mc_median_bps":  float(np.median(terminal)),
        "mc_5th_bps":     float(np.percentile(terminal, 5)),
        "mc_1st_bps":     float(np.percentile(terminal, 1)),
    }
    pd.DataFrame([metrics]).T.reset_index().rename(columns={"index":"metric",0:"value"}) \
      .to_csv(os.path.join(EXPORT_DIR, "summary_metrics.csv"), index=False)
    shutil.copy(POLICY_PATH, os.path.join(EXPORT_DIR, "rl_policy.pt"))

    zip_path = os.path.join(BASE_DIR, "hft_rl_validation.zip")
    import zipfile as zf
    with zf.ZipFile(zip_path, "w", zf.ZIP_DEFLATED) as z:
        for fn in os.listdir(EXPORT_DIR):
            z.write(os.path.join(EXPORT_DIR, fn), fn)
    print(f"\nExported: {zip_path}")
    print("Done. Download and review hft_rl_validation.zip.")

main()
