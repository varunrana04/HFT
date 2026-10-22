# ============================================================
# SCRIPT 1: download_walkforward_data.py
# Downloads daily aggTrades for 1 month or 1 year of BTCUSDT
# Usage:
#   python python/nautilus/download_walkforward_data.py --period 1month
#   python python/nautilus/download_walkforward_data.py --period 1year
#
# Run from HFT root directory. Saves to nautilus_cache/.
# Skips days already cached. Resumes on restart.
# ============================================================

import argparse, io, zipfile, requests
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE_DIR = Path("nautilus_cache")
BASE_URL  = "https://data.binance.vision/data/futures/um/daily"
SYMBOL    = "BTCUSDT"

try:
    import polars as pl
    USE_POLARS = True
except ImportError:
    import pandas as pd
    USE_POLARS = False


def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download_day(date_str: str, stream: str = "aggTrades") -> str:
    """Download one day of data, return status string."""
    out_path = CACHE_DIR / f"{SYMBOL}-{stream}-{date_str}.parquet"
    if out_path.exists():
        return f"  SKIP  {date_str} ({stream}) — already cached"

    url = f"{BASE_URL}/{stream}/{SYMBOL}/{SYMBOL}-{stream}-{date_str}.zip"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 404:
            return f"  MISS  {date_str} ({stream}) — not available yet"
        r.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                raw = f.read()

        if USE_POLARS:
            df = pl.read_csv(raw, has_header=True)
            df = df.rename({c: c.lower() for c in df.columns})
            df.write_parquet(out_path)
            nrows = len(df)
        else:
            import pandas as pd, io as _io
            df = pd.read_csv(_io.BytesIO(raw))
            df.columns = [c.lower() for c in df.columns]
            df.to_parquet(out_path, index=False)
            nrows = len(df)

        return f"  OK    {date_str} ({stream}) — {nrows:,} rows"
    except Exception as e:
        return f"  ERROR {date_str} ({stream}) — {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=["1month", "1year", "custom"], default="1month")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (for --period custom)")
    parser.add_argument("--end",   help="End date YYYY-MM-DD (for --period custom)")
    parser.add_argument("--workers", type=int, default=6, help="Parallel download threads")
    args = parser.parse_args()

    today = date.today()

    if args.period == "1month":
        start_d = today - timedelta(days=30)
        end_d   = today - timedelta(days=1)
    elif args.period == "1year":
        start_d = today - timedelta(days=365)
        end_d   = today - timedelta(days=1)
    else:
        start_d = date.fromisoformat(args.start)
        end_d   = date.fromisoformat(args.end)

    CACHE_DIR.mkdir(exist_ok=True)
    dates = [str(d) for d in date_range(start_d, end_d)]
    # Skip weekends (crypto trades 24/7 but Binance Vision has all days)
    print(f"Downloading {len(dates)} days: {dates[0]} -> {dates[-1]}")
    print(f"Workers: {args.workers} | Stream: aggTrades | Cache: {CACHE_DIR}/")
    print("-" * 60)

    tasks = []
    for d in dates:
        tasks.append(d)

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_day, d, "aggTrades"): d for d in tasks}
        for f in as_completed(futures):
            print(f.result())
            completed += 1
            if completed % 10 == 0:
                cached = len(list(CACHE_DIR.glob(f"{SYMBOL}-aggTrades-*.parquet")))
                print(f"  Progress: {completed}/{len(tasks)} | Cached total: {cached}")

    cached = sorted(CACHE_DIR.glob(f"{SYMBOL}-aggTrades-*.parquet"))
    print(f"\nDone. {len(cached)} days in cache ({CACHE_DIR}/)")
    print(f"First: {cached[0].name if cached else 'none'}")
    print(f"Last:  {cached[-1].name if cached else 'none'}")


if __name__ == "__main__":
    main()


# ============================================================
# SCRIPT 2: hft_extended_walkforward.py  (save separately)
# Full walk-forward: 1 month or 1 year
# ============================================================
# See hft_extended_walkforward.py
