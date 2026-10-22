import os
import polars as pl

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data", "parquet")

_METAL_SYMBOLS = {"XAUUSDT", "XAGUSDT"}

def _is_metal(symbol):
    return symbol.upper() in _METAL_SYMBOLS


def build_feature_matrix(symbol, date_str, horizon_ms=200):
    """
    Builds a trade-tape feature matrix from aggTrades + bookTicker (L2).

    Features (8 signals):
        tfi_ema_50          — trade flow imbalance, fast EMA
        tfi_ema_200         — trade flow imbalance, slow EMA
        volatility_ema_100  — price-tick volatility, fast EMA
        volatility_ema_500  — price-tick volatility, slow EMA
        book_imbalance      — (bid_qty - ask_qty) / (bid_qty + ask_qty)  [REAL L2 OFI]
        spread_bps          — (ask_px - bid_px) / mid * 10000            [REAL L2 spread]
        microprice          — (bid_px*ask_qty + ask_px*bid_qty) / total  [REAL L2 mid]
        spread_ema          — EMA(spread_bps, half_life=50)               [smoothed spread]

    Label: label_fwd_ret_bps — forward return over horizon_ms (default 200ms).
    Falls back to trade-based proxies if bookTicker parquet is missing.
    Metal symbols (XAUUSDT, XAGUSDT) use 4-10x slower EMAs.
    """
    trades_path = os.path.join(DATA_DIR, "aggTrades",
                               f"{symbol}-aggTrades-{date_str}.parquet")
    book_path   = os.path.join(DATA_DIR, "bookTicker",
                               f"{symbol}-bookTicker-{date_str}.parquet")

    if not os.path.exists(trades_path):
        print(f"[WARN] aggTrades not found: {trades_path}")
        return None

    metal    = _is_metal(symbol)
    asset_id = 1 if metal else 0
    hl_fast  = 200  if metal else 50
    hl_slow  = 1000 if metal else 200
    hl_vol   = 500  if metal else 100
    hl_vol_s = 2000 if metal else 500

    has_l2 = os.path.exists(book_path)
    print(f"  Building: {symbol} {date_str}  "
          f"({'metal' if metal else 'crypto'}, horizon={horizon_ms}ms, "
          f"L2={'yes' if has_l2 else 'NO — using proxy fallback'})")

    # ── 1. Load aggTrades ──────────────────────────────────────────────────────
    df = pl.scan_parquet(trades_path).sort("transact_time")

    df = df.with_columns([
        pl.when(pl.col("is_buyer_maker"))
          .then(-pl.col("quantity"))
          .otherwise(pl.col("quantity"))
          .alias("signed_volume"),
        (pl.col("price") * pl.col("quantity")).alias("notional_value"),
    ])

    # ── 2. TFI EMAs + volatility EMAs (from trades) ───────────────────────────
    df = df.with_columns([
        pl.col("signed_volume").ewm_mean(half_life=hl_fast).alias("tfi_ema_50"),
        pl.col("signed_volume").ewm_mean(half_life=hl_slow).alias("tfi_ema_200"),
        pl.col("price").diff().abs().ewm_mean(half_life=hl_vol).alias("volatility_ema_100"),
        pl.col("price").diff().abs().ewm_mean(half_life=hl_vol_s).alias("volatility_ema_500"),
    ])

    # ── 3. L2 features — join bookTicker asof ─────────────────────────────────
    if has_l2:
        book_scan = pl.scan_parquet(book_path)
        book_cols = [c.lower() for c in book_scan.columns]
        book = (
            book_scan
              .rename({old: new for old, new in zip(book_scan.columns, book_cols)})
              .select([
                  pl.col("transaction_time").alias("transact_time"),
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
            ((pl.col("ask_px") - pl.col("bid_px")) / pl.col("mid_px") * 10000.0)
            .alias("spread_bps"),
            ((pl.col("bid_qty") - pl.col("ask_qty")) /
             (pl.col("bid_qty") + pl.col("ask_qty") + 1e-9))
            .alias("book_imbalance"),
            ((pl.col("bid_px") * pl.col("ask_qty") + pl.col("ask_px") * pl.col("bid_qty")) /
             (pl.col("bid_qty") + pl.col("ask_qty") + 1e-9))
            .alias("microprice"),
        ])
        df = df.with_columns([
            pl.col("spread_bps").ewm_mean(half_life=hl_fast).alias("spread_ema"),
        ])
        drop_l2_temp = ["bid_px", "bid_qty", "ask_px", "ask_qty", "mid_px"]
    else:
        # Fallback: proxy from trades only
        df = df.with_columns([
            pl.col("signed_volume").abs().ewm_mean(half_life=hl_fast).alias("_abs_sv"),
        ])
        df = df.with_columns([
            (pl.col("signed_volume") / (pl.col("_abs_sv") + 1e-9)).alias("book_imbalance"),
            (pl.col("volatility_ema_100") * 2.0).alias("spread_bps"),
            pl.col("price").alias("microprice"),
        ])
        df = df.with_columns([
            pl.col("spread_bps").ewm_mean(half_life=hl_fast).alias("spread_ema"),
        ])
        df = df.drop("_abs_sv")
        drop_l2_temp = []

    # ── 4. Asset type flag ─────────────────────────────────────────────────────
    df = df.with_columns([
        pl.lit(asset_id).cast(pl.Int8).alias("asset_type"),
    ])

    # ── 5. Forward-return label (200ms) ───────────────────────────────────────
    df = df.with_columns(
        (pl.col("transact_time") + horizon_ms).alias("target_time")
    )
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
    df = df.drop([c for c in (drop_l2_temp + ["target_time", "future_price"])
                  if c in df.columns])

    # ── 6. Save ────────────────────────────────────────────────────────────────
    final_df = df.collect()
    out_dir  = os.path.join(DATA_DIR, "features")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}-features-{date_str}.parquet")
    final_df.write_parquet(out_path)
    print(f"  Saved {final_df.height:,} rows -> {out_path}")
    return final_df


if __name__ == "__main__":
    from datetime import datetime, timedelta

    SYMBOLS    = ["BTCUSDT", "ETHUSDT", "XAUUSDT", "XAGUSDT"]
    start_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
    end_date   = datetime.strptime("2026-08-31", "%Y-%m-%d")

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        for sym in SYMBOLS:
            try:
                build_feature_matrix(sym, date_str)
            except Exception as e:
                print(f"  [ERROR] {sym} {date_str}: {e}")
        current += timedelta(days=1)

    print("Feature generation complete.")
