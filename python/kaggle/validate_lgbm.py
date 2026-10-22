"""
HFT LightGBM — Final Validation & Stress-Test (v4, L2 bookTicker)
===================================================================
Requires: lgbm_trade_tape_model.txt at /kaggle/working/models/
Downloads aggTrades + bookTicker (L2) for OOS period Aug 2026.
Computes 8 L2 microstructure features (must match train_lgbm_kaggle.py v4).
Exports charts + metrics to hft_lgbm_validation.zip.

Paste entire cell into Kaggle and run.
"""

import os, io, requests, zipfile, gc, warnings, json, shutil
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

MODEL_DIR  = os.path.join(BASE_DIR, "models")
EXPORT_DIR = os.path.join(BASE_DIR, "hft_lgbm_validation")
BASE_URL   = "https://data.binance.vision/data/futures/um/daily"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_trade_tape_model.txt")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Seed from Kaggle Model input if not already present
PRETRAINED = "/kaggle/input/models/varunrana3104/rl-dl-hft-model/other/default/1/lgbm_trade_tape_model.txt"
if not os.path.exists(MODEL_PATH) and os.path.exists(PRETRAINED):
    shutil.copy(PRETRAINED, MODEL_PATH)
    print(f"Seeded from checkpoint: {MODEL_PATH}")

# ── Feature set (must match train_lgbm_kaggle.py v4) ─────────────────────────
FEATURES = [
    "tfi_ema_50",
    "tfi_ema_200",
    "volatility_ema_100",
    "volatility_ema_500",
    "book_imbalance",   # REAL L2 OFI
    "spread_bps",       # REAL bid/ask spread
    "microprice",       # REAL weighted mid
    "spread_ema",       # EMA(spread_bps, 50)
]
TARGET          = "label_fwd_ret_bps"
CONFIDENCE_BAND = 0.5
TAKER_FEE_BPS   = 0.1
COOLDOWN_MS     = 200
N_SIM           = 10_000

# OOS dates (Aug 2026 — never seen during training)
OOS_DATES = [
    "2026-08-25", "2026-08-26", "2026-08-27",
    "2026-08-28", "2026-08-29",
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. L2 DATA PIPELINE (same as training)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_zip(url, timeout=60):
    r = requests.get(url, timeout=timeout)
    return r.content if r.status_code == 200 else None

def _process_day(symbol, date_str):
    """Download aggTrades + bookTicker, compute 8 L2 features, return DataFrame."""
    trades_url = f"{BASE_URL}/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
    book_url   = f"{BASE_URL}/bookTicker/{symbol}/{symbol}-bookTicker-{date_str}.zip"

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_t = ex.submit(_fetch_zip, trades_url)
        f_b = ex.submit(_fetch_zip, book_url)
        trades_bytes = f_t.result()
        book_bytes   = f_b.result()

    if trades_bytes is None:
        print(f"  [SKIP] {date_str} — aggTrades unavailable")
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
            book = (
                book.select([
                    pl.col(ts_col).alias("transact_time"),
                    pl.col("best_bid_price").cast(pl.Float32).alias("bid_px"),
                    pl.col("best_bid_qty").cast(pl.Float32).alias("bid_qty"),
                    pl.col("best_ask_price").cast(pl.Float32).alias("ask_px"),
                    pl.col("best_ask_qty").cast(pl.Float32).alias("ask_qty"),
                ]).sort("transact_time")
            )
            df = df.join_asof(book, on="transact_time", strategy="backward")
            df = df.with_columns([
                ((pl.col("bid_px") + pl.col("ask_px")) / 2.0).alias("mid_px"),
            ])
            df = df.with_columns([
                ((pl.col("ask_px") - pl.col("bid_px")) / (pl.col("mid_px") + 1e-9) * 10000.0)
                .alias("spread_bps"),
                ((pl.col("bid_qty") - pl.col("ask_qty")) /
                 (pl.col("bid_qty") + pl.col("ask_qty") + 1e-9))
                .alias("book_imbalance"),
                ((pl.col("bid_px") * pl.col("ask_qty") + pl.col("ask_px") * pl.col("bid_qty")) /
                 (pl.col("bid_qty") + pl.col("ask_qty") + 1e-9))
                .alias("microprice"),
            ])
            df = df.with_columns([
                pl.col("spread_bps").ewm_mean(half_life=50).alias("spread_ema"),
            ])
            df = df.drop(["bid_px", "bid_qty", "ask_px", "ask_qty", "mid_px"])
        else:
            print(f"  [WARN] {date_str} — bookTicker unavailable, using proxy")
            df = df.with_columns([
                pl.col("signed_volume").abs().ewm_mean(half_life=50).alias("_abs_sv"),
            ])
            df = df.with_columns([
                (pl.col("signed_volume") / (pl.col("_abs_sv") + 1e-9)).alias("book_imbalance"),
                (pl.col("volatility_ema_100") * 2.0).alias("spread_bps"),
                pl.col("price").alias("microprice"),
            ])
            df = df.with_columns([
                pl.col("spread_bps").ewm_mean(half_life=50).alias("spread_ema"),
            ])
            df = df.drop("_abs_sv")

        # Forward label 200ms
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
        df = df.drop_nulls(subset=["label_fwd_ret_bps"])
        result = df.select(FEATURES + [TARGET, "transact_time"]).collect() \
                   if hasattr(df, "collect") else df.select(FEATURES + [TARGET, "transact_time"])
        return result.to_pandas()
    except Exception as e:
        print(f"  [ERROR] {date_str}: {e}")
        return None

def load_oos_data(symbol="BTCUSDT"):
    print(f"Loading OOS data for {len(OOS_DATES)} days...")
    frames = []
    for d in OOS_DATES:
        print(f"  {d}...", end=" ")
        f = _process_day(symbol, d)
        if f is not None:
            frames.append(f)
            print(f"{len(f):,} rows")
        else:
            print("skipped")
    if not frames:
        raise RuntimeError("No OOS data loaded.")
    df = pd.concat(frames, ignore_index=True)
    print(f"Total OOS rows: {len(df):,}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. BACKTEST WITH COOLDOWN
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, preds):
    """
    Vectorised backtest with COOLDOWN_MS gate.
    Returns trade log DataFrame.
    """
    signal_mask = np.abs(preds) > CONFIDENCE_BAND
    times       = df["transact_time"].values
    labels      = df[TARGET].values
    directions  = np.sign(preds)

    trades = []
    last_trade_time = -np.inf

    for i in np.where(signal_mask)[0]:
        t = times[i]
        if t - last_trade_time < COOLDOWN_MS:
            continue
        direction = directions[i]
        pnl       = direction * labels[i] - TAKER_FEE_BPS
        trades.append({
            "time":      t,
            "pred_bps":  float(preds[i]),
            "fwd_bps":   float(labels[i]),
            "direction": int(direction),
            "pnl_bps":   float(pnl),
        })
        last_trade_time = t

    if not trades:
        raise RuntimeError("No trades fired — check CONFIDENCE_BAND or model features.")
    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def chart_equity_drawdown(trade_df):
    cum_pnl  = trade_df["pnl_bps"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown    = cum_pnl - running_max

    fig = make_subplots(rows=2, cols=1, subplot_titles=["Cumulative PnL (bps)", "Drawdown (bps)"],
                        shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=cum_pnl.values, mode="lines",
                             line=dict(color="#00d4ff", width=1.5), name="PnL"), row=1, col=1)
    fig.add_trace(go.Scatter(y=drawdown.values, mode="lines", fill="tozeroy",
                             line=dict(color="#ff4444", width=1), name="Drawdown"), row=2, col=1)
    fig.update_layout(template="plotly_dark", title="LGBM v4 L2 — OOS Equity Curve",
                      height=700, showlegend=True)
    return fig

def chart_monte_carlo(trade_df, n_sim=N_SIM):
    pnl_vec = trade_df["pnl_bps"].values
    n       = len(pnl_vec)
    rng     = np.random.default_rng(42)
    paths   = rng.choice(pnl_vec, size=(n_sim, n), replace=True).cumsum(axis=1)

    fig = go.Figure()
    for p in paths[::max(1, n_sim//200)]:
        fig.add_trace(go.Scatter(y=p, mode="lines",
                                 line=dict(width=0.3, color="rgba(0,212,255,0.08)"),
                                 showlegend=False))
    for q, name, color in [(0.01, "1st pct", "red"), (0.05, "5th pct", "orange"),
                            (0.50, "Median",  "lime")]:
        fig.add_trace(go.Scatter(y=np.quantile(paths, q, axis=0), mode="lines",
                                 line=dict(width=2, color=color), name=name))
    fig.update_layout(template="plotly_dark", title=f"Monte Carlo ({n_sim:,} paths)",
                      height=600)
    return fig, paths

def chart_pnl_distribution(trade_df):
    wins   = trade_df[trade_df["pnl_bps"] > 0]["pnl_bps"]
    losses = trade_df[trade_df["pnl_bps"] <= 0]["pnl_bps"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=wins.values,   name="Wins",   marker_color="#00ff88",
                               opacity=0.7, nbinsx=80))
    fig.add_trace(go.Histogram(x=losses.values, name="Losses", marker_color="#ff4444",
                               opacity=0.7, nbinsx=80))
    fig.update_layout(template="plotly_dark", barmode="overlay",
                      title="Trade PnL Distribution (bps)", height=450)
    return fig

def chart_feature_importance(model):
    imp   = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    idx   = np.argsort(imp)[::-1]
    fig   = go.Figure(go.Bar(x=[names[i] for i in idx], y=[imp[i] for i in idx],
                              marker_color="#00d4ff"))
    fig.update_layout(template="plotly_dark", title="Feature Importance (gain)",
                      height=400)
    return fig

def chart_pnl_heatmap(trade_df):
    df2 = trade_df.copy()
    df2["pred_bin"] = pd.cut(df2["pred_bps"].abs(), bins=10)
    hmap = df2.groupby("pred_bin")["pnl_bps"].mean().reset_index()
    fig  = go.Figure(go.Bar(x=[str(b) for b in hmap["pred_bin"]],
                             y=hmap["pnl_bps"].values, marker_color="#9b59b6"))
    fig.update_layout(template="plotly_dark",
                      title="Avg PnL by Prediction Magnitude Bucket", height=400,
                      xaxis_title="|pred| bin (bps)", yaxis_title="Avg net PnL (bps)")
    return fig

def chart_pred_distribution(preds, signal_mask):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=preds[~signal_mask], name="No signal", nbinsx=100,
                               marker_color="gray", opacity=0.6))
    fig.add_trace(go.Histogram(x=preds[signal_mask], name="Signal fired", nbinsx=80,
                               marker_color="#00d4ff", opacity=0.8))
    fig.add_vline(x=CONFIDENCE_BAND,  line_dash="dash", line_color="lime",
                  annotation_text=f"+{CONFIDENCE_BAND} bps")
    fig.add_vline(x=-CONFIDENCE_BAND, line_dash="dash", line_color="lime",
                  annotation_text=f"-{CONFIDENCE_BAND} bps")
    fig.update_layout(template="plotly_dark", barmode="overlay",
                      title="Prediction Distribution", height=450)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HFT LGBM v4 — OOS Validation (L2 bookTicker Features)")
    print(f"Features ({len(FEATURES)}): {FEATURES}")
    print(f"OOS dates: {OOS_DATES}")
    print("=" * 60)

    # Load model
    assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"\nModel loaded: {model.num_trees()} trees, {model.num_feature()} features")
    print(f"Model features: {model.feature_name()}")
    if model.num_feature() != len(FEATURES):
        raise ValueError(
            f"Model has {model.num_feature()} features but FEATURES list has {len(FEATURES)}.\n"
            f"Model: {model.feature_name()}\nExpected: {FEATURES}"
        )

    # Load OOS data
    oos_df = load_oos_data()
    X_oos  = oos_df[FEATURES].astype(np.float32)
    y_oos  = oos_df[TARGET].values.astype(np.float32)
    preds  = model.predict(X_oos.values)

    # Backtest
    print("\nRunning backtest with cooldown gate...")
    trade_df = run_backtest(oos_df, preds)
    signal_mask = np.abs(preds) > CONFIDENCE_BAND

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_total   = len(preds)
    n_signals = signal_mask.sum()
    pnl_vec   = trade_df["pnl_bps"].values
    total_pnl = pnl_vec.sum()
    win_rate  = (pnl_vec > 0).mean() * 100
    dir_acc   = (np.sign(preds[signal_mask]) == np.sign(y_oos[signal_mask])).mean() * 100
    max_dd    = (trade_df["pnl_bps"].cumsum() - trade_df["pnl_bps"].cumsum().cummax()).min()
    sharpe    = (pnl_vec.mean() / (pnl_vec.std() + 1e-9)) * np.sqrt(252 * 6.5 * 3600 / COOLDOWN_MS * 1000)
    pf        = pnl_vec[pnl_vec > 0].sum() / (abs(pnl_vec[pnl_vec < 0].sum()) + 1e-9)

    print(f"\n{'='*60}")
    print(f"RESULTS — LGBM v4 L2  (OOS Aug 2026)")
    print(f"{'='*60}")
    print(f"  Rows              : {n_total:,}")
    print(f"  Signals           : {n_signals:,}  ({100*n_signals/n_total:.2f}% of ticks)")
    print(f"  Trades (cooldown) : {len(trade_df):,}")
    print(f"  Win rate          : {win_rate:.2f}%")
    print(f"  Dir accuracy      : {dir_acc:.2f}%")
    print(f"  Total PnL         : {total_pnl:,.2f} bps")
    print(f"  Avg PnL/trade     : {pnl_vec.mean():.4f} bps")
    print(f"  Max drawdown      : {max_dd:,.2f} bps")
    print(f"  Sharpe (ann.)     : {sharpe:.3f}")
    print(f"  Profit factor     : {pf:.3f}")
    print(f"{'='*60}")

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    print(f"\nRunning Monte Carlo ({N_SIM:,} paths)...")
    mc_fig, mc_paths = chart_monte_carlo(trade_df)
    terminal = mc_paths[:, -1]
    print(f"  Median terminal PnL : {np.median(terminal):,.2f} bps")
    print(f"  5th pct             : {np.percentile(terminal, 5):,.2f} bps")
    print(f"  1st pct             : {np.percentile(terminal, 1):,.2f} bps")

    # ── Save charts ───────────────────────────────────────────────────────────
    print("\nExporting charts...")
    charts = {
        "01_equity_and_drawdown":      chart_equity_drawdown(trade_df),
        "02_monte_carlo_cone":         mc_fig,
        "03_drawdown_distribution":    chart_pnl_distribution(trade_df),
        "04_terminal_pnl_distribution": go.Figure(go.Histogram(
            x=terminal, nbinsx=100, marker_color="#00d4ff"
        )).update_layout(template="plotly_dark",
                         title="Monte Carlo Terminal PnL Distribution", height=450),
        "05_feature_importance":       chart_feature_importance(model),
        "06_pnl_heatmap":              chart_pnl_heatmap(trade_df),
        "07_pred_distribution":        chart_pred_distribution(preds, signal_mask),
    }
    for name, fig in charts.items():
        out = os.path.join(EXPORT_DIR, f"{name}.html")
        fig.write_html(out)
        print(f"  Saved {out}")

    # ── CSVs ──────────────────────────────────────────────────────────────────
    trade_df.to_csv(os.path.join(EXPORT_DIR, "oos_backtest_results.csv"), index=False)
    pd.DataFrame(mc_paths).to_csv(os.path.join(EXPORT_DIR, "monte_carlo_paths.csv"), index=False)

    metrics = {
        "total_trades":   len(trade_df),
        "total_pnl_bps":  total_pnl,
        "win_rate_pct":   win_rate,
        "dir_acc_pct":    dir_acc,
        "max_dd_bps":     float(max_dd),
        "sharpe":         float(sharpe),
        "profit_factor":  float(pf),
        "mc_median_bps":  float(np.median(terminal)),
        "mc_5th_bps":     float(np.percentile(terminal, 5)),
        "mc_1st_bps":     float(np.percentile(terminal, 1)),
    }
    pd.DataFrame([metrics]).T.reset_index().rename(columns={"index":"metric",0:"value"}) \
      .to_csv(os.path.join(EXPORT_DIR, "summary_metrics.csv"), index=False)

    # Copy model into export
    shutil.copy(MODEL_PATH, os.path.join(EXPORT_DIR, "lgbm_trade_tape_model.txt"))

    # ── Zip ───────────────────────────────────────────────────────────────────
    zip_path = os.path.join(BASE_DIR, "hft_lgbm_validation.zip")
    import zipfile as zf
    with zf.ZipFile(zip_path, "w", zf.ZIP_DEFLATED) as z:
        for fn in os.listdir(EXPORT_DIR):
            z.write(os.path.join(EXPORT_DIR, fn), fn)
    print(f"\nExported: {zip_path}")
    print("Done. Download and review hft_lgbm_validation.zip.")

main()
