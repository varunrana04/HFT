"""
HFT LightGBM Training Pipeline — Kaggle v4 (L2 bookTicker)
============================================================
Downloads aggTrades + bookTicker (L2) in-memory for each day.
Joins on timestamp to compute REAL microstructure features:
  - spread_bps      : actual bid/ask spread (not a proxy)
  - book_imbalance  : (bid_qty - ask_qty) / total  [Cont-Kukanov OBI]
  - microprice      : bid_px×ask_qty + ask_px×bid_qty / total
  - spread_ema      : EMA(spread_bps, 50)

Total: 8 features. Label: 200ms forward return in bps.
Output: /kaggle/working/models/lgbm_trade_tape_model.txt

Paste entire cell into Kaggle and run. GPU NOT required.
"""

import os, requests, zipfile, gc, warnings, io, shutil, json
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
warnings.filterwarnings("ignore")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

MODEL_DIR  = os.path.join(BASE_DIR, "models")
BASE_URL   = "https://data.binance.vision/data/futures/um/daily"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_trade_tape_model.txt")
PROG_PATH  = os.path.join(MODEL_DIR, "lgbm_progress.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Checkpoint seeding (Kaggle resume) ────────────────────────────────────────
PRETRAINED_LGBM = "/kaggle/input/models/varunrana3104/hft-1-0/other/default/1/lgbm_trade_tape_model.txt"
PRETRAINED_PROG = "/kaggle/input/models/varunrana3104/hft-1-0/other/default/1/lgbm_progress.json"
if not os.path.exists(MODEL_PATH) and os.path.exists(PRETRAINED_LGBM):
    shutil.copy(PRETRAINED_LGBM, MODEL_PATH)
    print(f"Seeded from checkpoint: {MODEL_PATH}")
if not os.path.exists(PROG_PATH) and os.path.exists(PRETRAINED_PROG):
    shutil.copy(PRETRAINED_PROG, PROG_PATH)

# ── Feature set (8 L2 microstructure signals) ─────────────────────────────────
FEATURES = [
    "tfi_ema_50",           # trade flow imbalance, fast EMA (aggTrades)
    "tfi_ema_200",          # trade flow imbalance, slow EMA (aggTrades)
    "volatility_ema_100",   # price-tick vol, fast EMA (aggTrades)
    "volatility_ema_500",   # price-tick vol, slow EMA (aggTrades)
    "book_imbalance",       # (bid_qty - ask_qty) / total — REAL L2 OFI
    "spread_bps",           # (ask - bid) / mid * 10000  — REAL bid/ask spread
    "microprice",           # bid_px*ask_qty + ask_px*bid_qty / total
    "spread_ema",           # EMA(spread_bps, 50) — regime-smoothed spread
]
TARGET          = "label_fwd_ret_bps"
CONFIDENCE_BAND = 0.5   # bps — threshold to classify as a signal
FP_PENALTY      = 1.8   # penalise false positives harder (adverse selection)
FN_PENALTY      = 1.0
TREES_FIRST     = 100
TREES_PER_MONTH = 50
# Memory guard — Binance Futures has 2-10M ticks/day; keeping 20% still gives
# ~400k-2M rows/day which is more than enough for tree training.
SUBSAMPLE_RATE  = 0.20   # fraction of ticks to keep per day
FETCH_WORKERS   = 2      # keep only 2 days in RAM at once (vs 8)

# ── Asymmetric loss (penalises false signals more than missed signals) ─────────
def _asymmetric_obj(y_pred, dataset):
    y_true      = dataset.get_label()
    residual    = y_pred - y_true
    true_signal = np.abs(y_true) > CONFIDENCE_BAND
    pred_signal = np.abs(y_pred) > CONFIDENCE_BAND
    false_pos   = pred_signal & ~true_signal
    grad = np.where(false_pos, FP_PENALTY * residual, FN_PENALTY * residual)
    hess = np.where(false_pos, FP_PENALTY,            FN_PENALTY)
    return grad, hess

def _asymmetric_eval(y_pred, dataset):
    y_true      = dataset.get_label()
    residual    = y_pred - y_true
    true_signal = np.abs(y_true) > CONFIDENCE_BAND
    pred_signal = np.abs(y_pred) > CONFIDENCE_BAND
    false_pos   = pred_signal & ~true_signal
    loss = np.mean(np.where(false_pos,
                             FP_PENALTY * residual ** 2,
                             FN_PENALTY * residual ** 2))
    return "asym_mse", loss, False

LGBM_PARAMS = {
    "learning_rate": 0.05,
    "max_depth":     6,
    "num_leaves":    63,
    "min_child_samples": 100,
    "subsample":     0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":     0.1,
    "reg_lambda":    0.1,
    "verbosity":     -1,
    "boosting_type": "gbdt",
    "objective":     _asymmetric_obj,
    "n_jobs":        -1,
}

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY L2 PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_zip(url, timeout=45):
    """Download a Binance Vision zip and return raw bytes. Returns None on fail."""
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        return None
    return r.content

def _process_day_in_memory(symbol, date_str):
    """
    Downloads aggTrades + bookTicker in parallel, joins on timestamp,
    computes 8 L2 microstructure features, labels with 200ms forward return.
    Returns a pandas DataFrame with FEATURES + TARGET columns.
    """
    base = f"{BASE_URL}"
    trades_url = f"{base}/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
    book_url   = f"{base}/bookTicker/{symbol}/{symbol}-bookTicker-{date_str}.zip"

    # Download both streams concurrently
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_trades = ex.submit(_fetch_zip, trades_url)
        f_book   = ex.submit(_fetch_zip, book_url)
        trades_bytes = f_trades.result()
        book_bytes   = f_book.result()

    if trades_bytes is None:
        return None

    try:
        # ── Parse aggTrades ────────────────────────────────────────────────────
        with zipfile.ZipFile(io.BytesIO(trades_bytes)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pl.read_csv(f.read(), has_header=True)
        df = df.rename({c: c.lower() for c in df.columns})
        df = df.with_columns(pl.col("price").cast(pl.Float32))
        df = df.sort("transact_time")

        df = df.with_columns([
            pl.when(pl.col("is_buyer_maker"))
              .then(-pl.col("quantity"))
              .otherwise(pl.col("quantity"))
              .alias("signed_volume"),
        ])

        # TFI + volatility from trade tape
        df = df.with_columns([
            pl.col("signed_volume").ewm_mean(half_life=50).alias("tfi_ema_50"),
            pl.col("signed_volume").ewm_mean(half_life=200).alias("tfi_ema_200"),
            pl.col("price").diff().abs().ewm_mean(half_life=100).alias("volatility_ema_100"),
            pl.col("price").diff().abs().ewm_mean(half_life=500).alias("volatility_ema_500"),
        ])

        # ── Parse bookTicker + join asof ──────────────────────────────────────
        if book_bytes is not None:
            with zipfile.ZipFile(io.BytesIO(book_bytes)) as z:
                with z.open(z.namelist()[0]) as f:
                    book = pl.read_csv(f.read(), has_header=True)
            book = book.rename({c: c.lower() for c in book.columns})

            # Key timestamp column differs between Binance formats
            ts_col = "transaction_time" if "transaction_time" in book.columns else "event_time"
            book = (
                book
                .select([
                    pl.col(ts_col).alias("transact_time"),
                    pl.col("best_bid_price").cast(pl.Float32).alias("bid_px"),
                    pl.col("best_bid_qty").cast(pl.Float32).alias("bid_qty"),
                    pl.col("best_ask_price").cast(pl.Float32).alias("ask_px"),
                    pl.col("best_ask_qty").cast(pl.Float32).alias("ask_qty"),
                ])
                .sort("transact_time")
            )

            df = df.join_asof(book, on="transact_time", strategy="backward")

            df = df.with_columns([
                ((pl.col("bid_px") + pl.col("ask_px")) / 2.0).alias("mid_px"),
            ])
            df = df.with_columns([
                # Real spread in bps
                ((pl.col("ask_px") - pl.col("bid_px")) / (pl.col("mid_px") + 1e-9) * 10000.0)
                .alias("spread_bps"),
                # Real book imbalance — Cont-Kukanov L1 OBI
                ((pl.col("bid_qty") - pl.col("ask_qty")) /
                 (pl.col("bid_qty") + pl.col("ask_qty") + 1e-9))
                .alias("book_imbalance"),
                # Microprice — inventory-weighted true mid
                ((pl.col("bid_px") * pl.col("ask_qty") + pl.col("ask_px") * pl.col("bid_qty")) /
                 (pl.col("bid_qty") + pl.col("ask_qty") + 1e-9))
                .alias("microprice"),
            ])
            df = df.with_columns([
                pl.col("spread_bps").ewm_mean(half_life=50).alias("spread_ema"),
            ])
            df = df.drop(["bid_px", "bid_qty", "ask_px", "ask_qty", "mid_px"])

        else:
            # Fallback if bookTicker unavailable: proxy from trades
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

        # ── Forward-return label (200ms horizon) ──────────────────────────────
        df = df.with_columns((pl.col("transact_time") + 200).alias("target_time"))
        future_prices = (
            df.select(["transact_time", "price"])
              .rename({"transact_time": "target_time", "price": "future_price"})
              .unique(subset=["target_time"], keep="last", maintain_order=True)
              .sort("target_time")
        )
        df = df.join_asof(future_prices, on="target_time", strategy="backward")
        df = df.with_columns([
            ((pl.col("future_price") - pl.col("price")) / pl.col("price") * 10000.0)
            .alias("label_fwd_ret_bps")
        ])
        df = df.drop_nulls(subset=["label_fwd_ret_bps"])
        out = df.select(FEATURES + [TARGET])

        # Random subsample to cap memory (deterministic seed per date)
        seed = int(date_str.replace("-", ""))
        n    = len(out)
        keep = max(1, int(n * SUBSAMPLE_RATE))
        if keep < n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=keep, replace=False)
            idx.sort()
            out = out[idx]

        return out.to_pandas().astype(np.float32)

    except Exception as e:
        print(f"  [ERROR] {date_str}: {e}")
        return None


def fetch_month_dataframe(symbol, year, month):
    start = datetime(year, month, 1)
    end   = (start + relativedelta(months=1)) - timedelta(days=1)
    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range((end - start).days + 1)]
    print(f"  Fetching {len(dates)} days (aggTrades+bookTicker, {FETCH_WORKERS} workers)...")
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
        frames = list(ex.map(lambda d: _process_day_in_memory(symbol, d), dates))
    valid = [f for f in frames if f is not None and len(f) > 0]
    del frames; gc.collect()
    if not valid:
        return None
    combined = pd.concat(valid, ignore_index=True)
    del valid; gc.collect()
    print(f"  Month total: {len(combined):,} rows (~{SUBSAMPLE_RATE*100:.0f}% sample) | features: {FEATURES}")
    return combined

def save_progress(year, month):
    with open(PROG_PATH, "w") as f:
        json.dump({"last_year": year, "last_month": month}, f)

def load_progress():
    if not os.path.exists(PROG_PATH):
        return None
    with open(PROG_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train_lgbm():
    print("=" * 60)
    print("HFT LightGBM v4  [L2 bookTicker Features]")
    print(f"Features ({len(FEATURES)}): {FEATURES}")
    print("Training : 2024-01 -> 2026-07")
    print("Holdout  : 2026-08 (OOS, never seen during training)")
    print("Loss     : Asymmetric MSE (FP_PENALTY=1.8)")
    print("=" * 60)

    # ── Fetch holdout once and cache ───────────────────────────────────────────
    holdout_cache = os.path.join(MODEL_DIR, "lgbm_holdout_l2.parquet")
    if os.path.exists(holdout_cache):
        print("\nLoading cached Aug-2026 holdout...")
        val_df = pd.read_parquet(holdout_cache)
    else:
        print("\nFetching Aug-2026 holdout (OOS)...")
        val_df = fetch_month_dataframe("BTCUSDT", 2026, 8)
        if val_df is None:
            raise RuntimeError("Failed to download Aug-2026 holdout.")
        val_df.to_parquet(holdout_cache)
        print(f"  Holdout cached: {holdout_cache} ({len(val_df):,} rows)")
        gc.collect()

    X_val   = val_df[FEATURES]
    y_val   = val_df[TARGET]
    val_set = lgb.Dataset(X_val, label=y_val, free_raw_data=False)
    print(f"  Holdout: {len(val_df):,} rows")

    # ── Load or init model ─────────────────────────────────────────────────────
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            existing = lgb.Booster(model_file=MODEL_PATH)
            if existing.num_feature() == len(FEATURES):
                print(f"\nResuming from: {MODEL_PATH} ({existing.num_trees()} trees)")
                model = existing
            else:
                print(f"\n[WARN] Existing model has {existing.num_feature()} features "
                      f"but we need {len(FEATURES)} — starting fresh.")
        except Exception:
            pass

    # ── Training loop ──────────────────────────────────────────────────────────
    current_date = datetime(2024, 1, 1)
    end_date     = datetime(2026, 7, 31)

    progress = load_progress()
    if progress:
        resume = datetime(progress["last_year"], progress["last_month"], 1) \
                 + relativedelta(months=1)
        print(f"Auto-resuming from {resume.year}-{resume.month:02d}")
        current_date = resume

    # Sequential fetch — no prefetching to avoid double-loading months into RAM
    while current_date <= end_date:
        year, month = current_date.year, current_date.month
        print(f"\n{'='*60}\n  Month: {year}-{month:02d}\n{'='*60}")

        train_df = fetch_month_dataframe("BTCUSDT", year, month)

        if train_df is None or len(train_df) < 1000:
            print("  Insufficient data, skipping...")
            current_date += relativedelta(months=1)
            continue

        X_train   = train_df[FEATURES]
        y_train   = train_df[TARGET]
        train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        n_rounds  = TREES_FIRST if model is None else TREES_PER_MONTH

        model = lgb.train(
            LGBM_PARAMS,
            train_set,
            num_boost_round       = n_rounds,
            valid_sets            = [val_set],
            valid_names           = ["aug2026_oos"],
            feval                 = _asymmetric_eval,
            init_model            = model,
            keep_training_booster = True,
            callbacks             = [lgb.log_evaluation(period=n_rounds)],
        )

        # Quick signal check on holdout
        preds = model.predict(X_val.values)
        sig   = np.abs(preds) > CONFIDENCE_BAND
        if sig.sum() > 0:
            wr = (np.sign(preds[sig]) == np.sign(y_val.values[sig])).mean() * 100
            print(f"  OOS check: {sig.sum():,} signals | win rate: {wr:.2f}%")

        model.save_model(MODEL_PATH)
        save_progress(year, month)
        print(f"  Saved -> {MODEL_PATH}")

        del train_df, X_train, y_train, train_set
        gc.collect()
        import ctypes
        try: ctypes.CDLL("libc.so.6").malloc_trim(0)   # return freed pages to OS
        except Exception: pass
        current_date += relativedelta(months=1)



    # ── Final OOS report ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL OOS REPORT — Aug 2026 Holdout")
    print("=" * 60)
    preds = model.predict(X_val.values)
    sig   = np.abs(preds) > CONFIDENCE_BAND
    n_sig = sig.sum()
    dir_acc = (np.sign(preds[sig]) == np.sign(y_val.values[sig])).mean() * 100
    pnl     = (np.sign(preds[sig]) * y_val.values[sig]) - 0.1  # net of 0.1bps taker fee
    print(f"  Total rows      : {len(preds):,}")
    print(f"  Signals fired   : {n_sig:,}  ({100*n_sig/len(preds):.2f}% of ticks)")
    print(f"  Directional acc : {dir_acc:.2f}%")
    print(f"  Win rate (net)  : {(pnl>0).mean()*100:.2f}%")
    print(f"  Avg PnL/trade   : {pnl.mean():.4f} bps")
    print(f"  Total PnL       : {pnl.sum():,.2f} bps")
    print(f"  Features used   : {model.feature_name()}")
    print(f"\n  Model -> {MODEL_PATH}")
    print("  Run validate_lgbm.py for full charts and Monte Carlo.")

train_lgbm()
