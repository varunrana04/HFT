#!/usr/bin/env python3
"""
hf_data_loader.py — HuggingFace OHLCV-1m Dataset Adapter

Downloads and converts the mito0o852/OHLCV-1m dataset (Parquet)
into the format expected by our backtester and ML training pipeline.

Dataset: https://huggingface.co/datasets/mito0o852/OHLCV-1m
Format:  US Stock Market Minute-Level OHLCV (1992–2026)
Columns: timestamp, open, high, low, close, volume, symbol

Pipeline:
  1. Download Parquet files from HuggingFace Hub
  2. Filter by symbol (default: AAPL, SPY, QQQ, etc.)
  3. Convert OHLCV to synthetic trade/tick format
  4. Generate feature vectors for ML training
  5. Save as CSV compatible with backtest.py and train_model.py

Usage:
    python python/hf_data_loader.py --symbol AAPL --months 3
    python python/hf_data_loader.py --symbol SPY --start 2024-01 --end 2024-06
    python python/hf_data_loader.py --list-symbols     # Show available symbols

Output:
    data/AAPL_ohlcv.csv      — Raw OHLCV data
    data/AAPL_trades.csv     — Synthetic trade format (for backtest.py)
    data/AAPL_features.csv   — Feature vectors (for train_model.py)
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Optional, List

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[ERROR] pandas required: pip install pandas")
    sys.exit(1)

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HAS_HF = True
except ImportError:
    HAS_HF = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ─── Constants ───────────────────────────────────────────────

DATASET_REPO = "mito0o852/OHLCV-1m"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "hf_cache")

# Common stock symbols for testing
DEFAULT_SYMBOLS = ["AAPL", "SPY", "QQQ", "MSFT", "GOOGL", "AMZN", "TSLA", "META"]


# ─── Download ────────────────────────────────────────────────

def download_parquet_hf(month: str, cache_dir: str = CACHE_DIR) -> Optional[str]:
    """
    Download a monthly Parquet file from HuggingFace.

    Args:
        month: Month string like '2024-01'
        cache_dir: Local cache directory

    Returns:
        Path to downloaded file, or None on failure
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Try common filename patterns
    patterns = [
        f"data/{month}.parquet",
        f"{month}.parquet",
        f"data/ohlcv_{month}.parquet",
        f"ohlcv-1m-{month}.parquet",
    ]

    if HAS_HF:
        for pattern in patterns:
            try:
                path = hf_hub_download(
                    repo_id=DATASET_REPO,
                    filename=pattern,
                    repo_type="dataset",
                    cache_dir=cache_dir
                )
                print(f"  [+] Downloaded: {pattern}")
                return path
            except Exception:
                continue

        # If patterns fail, list files and find matching ones
        try:
            files = list_repo_files(DATASET_REPO, repo_type="dataset")
            parquet_files = [f for f in files if f.endswith('.parquet')]
            matching = [f for f in parquet_files if month in f]
            if matching:
                path = hf_hub_download(
                    repo_id=DATASET_REPO,
                    filename=matching[0],
                    repo_type="dataset",
                    cache_dir=cache_dir
                )
                print(f"  [+] Downloaded: {matching[0]}")
                return path
        except Exception as e:
            print(f"  [WARN] HuggingFace Hub error: {e}")

    # Fallback: Direct URL download
    if HAS_REQUESTS:
        for pattern in patterns:
            url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{pattern}"
            try:
                resp = requests.get(url, timeout=30, stream=True)
                if resp.status_code == 200:
                    local_path = os.path.join(cache_dir, f"{month}.parquet")
                    with open(local_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"  [+] Downloaded via URL: {pattern}")
                    return local_path
            except Exception:
                continue

    print(f"  [ERROR] Could not download data for {month}")
    return None


def list_available_files() -> List[str]:
    """List available Parquet files in the dataset."""
    if HAS_HF:
        try:
            files = list_repo_files(DATASET_REPO, repo_type="dataset")
            parquet_files = sorted([f for f in files if f.endswith('.parquet')])
            return parquet_files
        except Exception as e:
            print(f"  [ERROR] Could not list files: {e}")
    return []


# ─── Data Processing ────────────────────────────────────────

def load_and_filter(filepath: str, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Load a Parquet file and optionally filter by symbol.
    Handles various column name formats.
    """
    df = pd.read_parquet(filepath)

    # Standardize column names (lowercase)
    df.columns = [c.lower().strip() for c in df.columns]

    # Map common column name variants
    col_map = {}
    for col in df.columns:
        if 'time' in col or 'date' in col or col == 't':
            col_map[col] = 'timestamp'
        elif col in ['o', 'open_price']:
            col_map[col] = 'open'
        elif col in ['h', 'high_price']:
            col_map[col] = 'high'
        elif col in ['l', 'low_price']:
            col_map[col] = 'low'
        elif col in ['c', 'close_price', 'adj_close']:
            col_map[col] = 'close'
        elif col in ['v', 'vol']:
            col_map[col] = 'volume'
        elif col in ['s', 'sym', 'ticker', 'stock']:
            col_map[col] = 'symbol'

    df = df.rename(columns=col_map)

    # Filter by symbol if present
    if symbol and 'symbol' in df.columns:
        df = df[df['symbol'].str.upper() == symbol.upper()].copy()
        if len(df) == 0:
            available = df['symbol'].unique()[:20]
            print(f"  [WARN] Symbol '{symbol}' not found. Available: {list(available)}")
            return pd.DataFrame()

    # Sort by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def ohlcv_to_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OHLCV bars to synthetic trade-level data.

    For each 1-minute bar, generates 4 synthetic trades:
      1. Open price  (at bar start)
      2. High price  (at bar start + 15s)
      3. Low price   (at bar start + 30s)
      4. Close price (at bar start + 45s)

    This gives the backtester realistic price movement within each bar.
    """
    trades = []

    for _, row in df.iterrows():
        ts = row.get('timestamp', 0)

        # Convert timestamp to nanoseconds if needed
        if isinstance(ts, (int, float)):
            if ts < 1e12:  # Seconds
                ts_ns = int(ts * 1e9)
            elif ts < 1e15:  # Milliseconds
                ts_ns = int(ts * 1e6)
            elif ts < 1e18:  # Microseconds
                ts_ns = int(ts * 1e3)
            else:  # Already nanoseconds
                ts_ns = int(ts)
        else:
            # Try pandas Timestamp
            try:
                ts_ns = int(pd.Timestamp(ts).timestamp() * 1e9)
            except Exception:
                ts_ns = 0

        vol = row.get('volume', 100)
        quarter_vol = max(vol / 4.0, 1.0)

        # Generate 4 trades per bar (OHLC)
        prices = [
            ('open', row.get('open', 0), 0),
            ('high', row.get('high', 0), 15_000_000_000),   # +15s
            ('low',  row.get('low', 0),  30_000_000_000),   # +30s
            ('close', row.get('close', 0), 45_000_000_000), # +45s
        ]

        for label, price, offset in prices:
            if price > 0:
                trades.append({
                    'timestamp_ns': ts_ns + offset,
                    'price': price,
                    'quantity': quarter_vol,
                    'side': 1 if label in ('high', 'close') else 0,  # Synthetic side
                })

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values('timestamp_ns').reset_index(drop=True)
    return trades_df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 6 alpha signals from OHLCV data + mid_price column.

    This is a Python approximation of what FeatureEngine does in C++.
    Used to generate training data for train_model.py.
    """
    n = len(df)
    if n < 100:
        print(f"  [WARN] Not enough data ({n} rows) for feature computation")
        return pd.DataFrame()

    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    open_price = df['open'].values.astype(np.float64)

    mid_price = (high + low) / 2.0

    # 1. Microprice (approximation: VWAP-like)
    microprice = (high * volume + low * volume) / (2 * np.maximum(volume, 1))

    # 2. OFI (Order Flow Imbalance — approximated from OHLC)
    # Up bar → positive OFI, down bar → negative
    ofi = np.zeros(n)
    for i in range(1, n):
        price_change = close[i] - close[i-1]
        vol_factor = volume[i] / max(np.mean(volume[max(0,i-50):i+1]), 1)
        ofi[i] = np.sign(price_change) * vol_factor

    # 3. VPIN (Volume-Synchronized Probability of Informed Trading)
    bucket_size = 50
    n_buckets = 50
    vpin = np.zeros(n)
    buy_vol = np.zeros(n)
    sell_vol = np.zeros(n)

    for i in range(n):
        if close[i] > open_price[i]:
            buy_vol[i] = volume[i]
        else:
            sell_vol[i] = volume[i]

    for i in range(n_buckets, n):
        window_buy = np.sum(buy_vol[i-n_buckets:i])
        window_sell = np.sum(sell_vol[i-n_buckets:i])
        total = window_buy + window_sell
        if total > 0:
            vpin[i] = abs(window_buy - window_sell) / total

    # 4. Spread BPS (High-Low as proxy for bid-ask spread)
    spread_bps = np.zeros(n)
    for i in range(n):
        if mid_price[i] > 0:
            spread_bps[i] = (high[i] - low[i]) / mid_price[i] * 10000

    # 5. Realized Volatility (Welford's on returns)
    realized_vol = np.zeros(n)
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]

    window = 100
    for i in range(window, n):
        realized_vol[i] = np.std(returns[i-window:i])

    # 6. Stat-Arb Z-Score (mean reversion)
    lookback = 200
    stat_arb_z = np.zeros(n)
    for i in range(lookback, n):
        window_prices = mid_price[i-lookback:i]
        mean_p = np.mean(window_prices)
        std_p = np.std(window_prices)
        if std_p > 1e-10:
            stat_arb_z[i] = (mid_price[i] - mean_p) / std_p

    # Build features DataFrame
    features = pd.DataFrame({
        'timestamp': df.get('timestamp', range(n)),
        'microprice': microprice,
        'ofi': ofi,
        'vpin': vpin,
        'spread_bps': spread_bps,
        'realized_vol': realized_vol,
        'stat_arb_zscore': stat_arb_z,
        'mid_price': mid_price,
    })

    # Remove warmup period
    features = features.iloc[lookback:].reset_index(drop=True)

    return features


# ─── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HFT Engine — HuggingFace OHLCV-1m Data Loader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python python/hf_data_loader.py --symbol AAPL --months 3
    python python/hf_data_loader.py --symbol SPY --start 2024-01 --end 2024-06
    python python/hf_data_loader.py --list-files
    python python/hf_data_loader.py --local data/2024-01.parquet --symbol AAPL
        """
    )
    parser.add_argument('--symbol', type=str, default='AAPL',
                        help='Stock symbol to filter (default: AAPL)')
    parser.add_argument('--months', type=int, default=1,
                        help='Number of recent months to download (default: 1)')
    parser.add_argument('--start', type=str, default=None,
                        help='Start month (YYYY-MM format)')
    parser.add_argument('--end', type=str, default=None,
                        help='End month (YYYY-MM format)')
    parser.add_argument('--local', type=str, default=None,
                        help='Path to local Parquet file (skip download)')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory (default: data)')
    parser.add_argument('--list-files', action='store_true',
                        help='List available Parquet files in dataset')
    parser.add_argument('--list-symbols', action='store_true',
                        help='List available symbols in a file')
    parser.add_argument('--no-features', action='store_true',
                        help='Skip feature computation')
    parser.add_argument('--no-trades', action='store_true',
                        help='Skip synthetic trade generation')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  HFT Engine — HuggingFace Data Loader")
    print(f"{'='*60}")
    print(f"  Dataset: {DATASET_REPO}")
    print(f"  Symbol:  {args.symbol}")

    # ── List files mode ──
    if args.list_files:
        print(f"\n  Available files:")
        files = list_available_files()
        for f in files:
            print(f"    {f}")
        print(f"\n  Total: {len(files)} files")
        return

    # ── Output directory ──
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output
    )
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──
    t0 = time.time()
    all_data = []

    if args.local:
        # Load from local file
        print(f"\n  Loading from local file: {args.local}")
        df = load_and_filter(args.local, args.symbol)
        if len(df) > 0:
            all_data.append(df)
    else:
        # Download from HuggingFace
        if args.start and args.end:
            # Generate month range
            from datetime import datetime
            start = datetime.strptime(args.start, '%Y-%m')
            end = datetime.strptime(args.end, '%Y-%m')
            months = []
            current = start
            while current <= end:
                months.append(current.strftime('%Y-%m'))
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
        else:
            # Use recent months
            from datetime import datetime, timedelta
            now = datetime.now()
            months = []
            for i in range(args.months):
                d = now - timedelta(days=30 * i)
                months.append(d.strftime('%Y-%m'))
            months.reverse()

        print(f"\n  Downloading {len(months)} months: {months}")

        for month in months:
            filepath = download_parquet_hf(month)
            if filepath:
                df = load_and_filter(filepath, args.symbol)
                if len(df) > 0:
                    all_data.append(df)
                    print(f"    {month}: {len(df):,} rows")

    if not all_data:
        print("\n  [ERROR] No data loaded. Check symbol and date range.")
        print("  Try: python python/hf_data_loader.py --list-files")
        sys.exit(1)

    # Combine all months
    combined = pd.concat(all_data, ignore_index=True)
    load_time = time.time() - t0
    print(f"\n  Total rows: {len(combined):,}")
    print(f"  Load time:  {load_time:.2f}s")

    # ── List symbols mode ──
    if args.list_symbols:
        if 'symbol' in combined.columns:
            symbols = sorted(combined['symbol'].unique())
            print(f"\n  Available symbols ({len(symbols)}):")
            for s in symbols:
                count = len(combined[combined['symbol'] == s])
                print(f"    {s:10s}  {count:>8,} rows")
        return

    # ── Save raw OHLCV ──
    ohlcv_path = os.path.join(output_dir, f"{args.symbol}_ohlcv.csv")
    combined.to_csv(ohlcv_path, index=False)
    print(f"\n  [+] Saved OHLCV:    {ohlcv_path} ({len(combined):,} rows)")

    # ── Generate synthetic trades ──
    if not args.no_trades:
        print(f"\n  Generating synthetic trades...")
        trades_df = ohlcv_to_trades(combined)
        trades_path = os.path.join(output_dir, f"{args.symbol}_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"  [+] Saved trades:   {trades_path} ({len(trades_df):,} rows)")

    # ── Compute features ──
    if not args.no_features:
        print(f"\n  Computing features (6 alpha signals)...")
        features_df = compute_features(combined)
        if len(features_df) > 0:
            features_path = os.path.join(output_dir, f"{args.symbol}_features.csv")
            features_df.to_csv(features_path, index=False)
            print(f"  [+] Saved features: {features_path} ({len(features_df):,} rows)")
        else:
            print(f"  [WARN] Not enough data for feature computation")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  DATA LOADING COMPLETE")
    print(f"{'='*60}")
    print(f"  Symbol:     {args.symbol}")
    print(f"  OHLCV rows: {len(combined):,}")
    print(f"  Output:     {output_dir}/")
    print(f"{'='*60}")
    print(f"\n  Next steps:")
    print(f"    1. Backtest:  python python/backtest.py --data {ohlcv_path}")
    print(f"    2. ML Train:  python python/train_model.py --data {output_dir}/{args.symbol}_features.csv")
    print()


if __name__ == '__main__':
    main()
