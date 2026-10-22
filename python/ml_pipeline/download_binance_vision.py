import os
import requests
import zipfile
import polars as pl
from datetime import datetime, timedelta
import concurrent.futures
import shutil

BASE_URL = "https://data.binance.vision/data/futures/um/daily"

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data")

# All symbols supported by this pipeline
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XAUUSDT", "XAGUSDT"]


def download_and_extract(data_type, symbol, date_str):
    url      = f"{BASE_URL}/{data_type}/{symbol}/{symbol}-{data_type}-{date_str}.zip"
    type_dir = os.path.join(DATA_DIR, "raw", data_type)
    os.makedirs(type_dir, exist_ok=True)
    zip_path = os.path.join(type_dir, f"{symbol}-{data_type}-{date_str}.zip")
    csv_path = os.path.join(type_dir, f"{symbol}-{data_type}-{date_str}.csv")

    if os.path.exists(csv_path):
        return csv_path

    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(type_dir)
            os.remove(zip_path)
            return csv_path
        else:
            print(f"  [SKIP] {symbol} {date_str} {data_type} HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"  [ERROR] {symbol} {date_str} {data_type}: {e}")
        return None


def convert_to_parquet(csv_path, data_type):
    if not csv_path or not os.path.exists(csv_path):
        return None
    parquet_dir  = os.path.join(DATA_DIR, "parquet", data_type)
    os.makedirs(parquet_dir, exist_ok=True)
    parquet_path = os.path.join(parquet_dir,
                                os.path.basename(csv_path).replace(".csv", ".parquet"))
    if os.path.exists(parquet_path):
        return parquet_path
    try:
        df = pl.read_csv(csv_path, has_header=True)
        df = df.rename({c: c.lower() for c in df.columns})
        # Cast numeric price/qty columns to Float32 for space efficiency
        if "price" in df.columns:
            df = df.with_columns(pl.col("price").cast(pl.Float32))
        for col in ["best_bid_price", "best_ask_price", "best_bid_qty", "best_ask_qty"]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float32))
        df.write_parquet(parquet_path)
        os.remove(csv_path)
        return parquet_path
    except Exception as e:
        print(f"  [ERROR] converting {csv_path}: {e}")
        return None


def fetch_day(symbol, date_str):
    """
    Download aggTrades + bookTicker (L2) in parallel, convert to parquet,
    build L2-enriched features, then clean up raw parquet to save disk.
    bookTicker gives real bid/ask spread and book imbalance (L2 OFI).
    """
    try:
        from build_features import build_feature_matrix
    except ImportError as e:
        print(f"  [CRITICAL] Cannot import build_features: {e}")
        return

    # ── Download both streams in parallel ─────────────────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_trades = ex.submit(download_and_extract, "aggTrades",  symbol, date_str)
        f_book   = ex.submit(download_and_extract, "bookTicker", symbol, date_str)
        trades_csv = f_trades.result()
        book_csv   = f_book.result()

    trades_pq = convert_to_parquet(trades_csv, "aggTrades")
    book_pq   = convert_to_parquet(book_csv,   "bookTicker")

    if trades_pq:
        try:
            build_feature_matrix(symbol, date_str)
        except Exception as e:
            print(f"  [ERROR] feature build {symbol} {date_str}: {e}")
        # Clean up raw parquet after features built — saves disk space
        for pq in [trades_pq, book_pq]:
            if pq and os.path.exists(pq):
                os.remove(pq)


def fetch_history(symbol, start_date, end_date, workers=4):
    start  = datetime.strptime(start_date, "%Y-%m-%d")
    end    = datetime.strptime(end_date,   "%Y-%m-%d")
    dates  = [(start + timedelta(days=i)).strftime("%Y-%m-%d")
              for i in range((end - start).days + 1)]
    print(f"Fetching {len(dates)} days for {symbol} (aggTrades + bookTicker L2)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_day, symbol, d) for d in dates]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"  Thread error: {e}")


if __name__ == "__main__":
    print("Unified Binance Vision Pipeline (aggTrades + bookTicker L2)")
    print(f"Symbols: {SYMBOLS}")
    START = "2024-01-01"
    END   = "2026-08-31"
    for sym in SYMBOLS:
        fetch_history(sym, START, END)
    print("Pipeline complete.")
