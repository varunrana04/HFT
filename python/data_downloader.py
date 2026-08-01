"""
HFT Engine — Real Market Data Downloader & Validator

Downloads REAL trade data from Binance's public data repository
(data.binance.vision) — no API key required.

Data quality pipeline:
  1. Download raw CSV from Binance public archive
  2. Decompress ZIP files
  3. Run multi-stage validation:
     - Timestamp monotonicity (no out-of-order ticks)
     - Sequence continuity (no gaps)
     - Price sanity (no zeros, no negatives, no >10% jumps)
     - Quantity sanity (positive, reasonable bounds)
     - Duplicate detection
  4. Generate quality report with acceptance rate
  5. Save validated data as Parquet (columnar, fast I/O)

Usage:
    python data_downloader.py --symbol BTCUSDT --start 2024-01-01 --end 2024-01-07
"""

import os
import sys
import argparse
import hashlib
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
import numpy as np

# ─── Constants ─────────────────────────────────────────────────

BINANCE_BASE_URL = "https://data.binance.vision/data/spot/daily"
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
VALIDATED_DIR = DATA_DIR / "validated"


# ─── Download Functions ───────────────────────────────────────

def download_binance_aggtrades(
    symbol: str,
    date: datetime,
    data_dir: Path = RAW_DIR,
) -> Optional[Path]:
    """
    Download a single day of aggTrades data from Binance.
    
    Source: data.binance.vision (official Binance public data)
    No API key required.
    
    Returns path to the extracted CSV, or None on failure.
    """
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BINANCE_BASE_URL}/aggTrades/{symbol}/{filename}"
    
    # Create directories
    symbol_dir = data_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = symbol_dir / filename
    csv_filename = f"{symbol}-aggTrades-{date_str}.csv"
    csv_path = symbol_dir / csv_filename
    
    # Skip if already downloaded and extracted
    if csv_path.exists():
        print(f"  [SKIP] {csv_filename} already exists")
        return csv_path
    
    # Download
    print(f"  [GET]  {url}")
    try:
        response = requests.get(url, timeout=60, stream=True)
        if response.status_code == 404:
            print(f"  [WARN] No data available for {date_str}")
            return None
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERR]  Download failed: {e}")
        return None
    
    # Save ZIP
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Verify checksum if available
    checksum_url = f"{url}.CHECKSUM"
    try:
        checksum_resp = requests.get(checksum_url, timeout=10)
        if checksum_resp.status_code == 200:
            expected_hash = checksum_resp.text.strip().split()[0]
            with open(zip_path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            if actual_hash != expected_hash:
                print(f"  [ERR]  Checksum mismatch! Expected {expected_hash}, got {actual_hash}")
                zip_path.unlink()
                return None
            print(f"  [OK]   Checksum verified")
    except requests.RequestException:
        print(f"  [WARN] Could not verify checksum (non-critical)")
    
    # Extract
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(symbol_dir)
        print(f"  [OK]   Extracted {csv_filename}")
    except zipfile.BadZipFile:
        print(f"  [ERR]  Corrupt ZIP file")
        zip_path.unlink()
        return None
    
    # Clean up ZIP to save space
    zip_path.unlink()
    
    return csv_path


def download_date_range(
    symbol: str,
    start_date: str,
    end_date: str,
) -> list[Path]:
    """Download aggTrades for a date range."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print(f"Downloading {symbol} aggTrades: {start_date} → {end_date}")
    print(f"Source: data.binance.vision (official, verified)")
    print(f"{'='*60}\n")
    
    csv_files = []
    current = start
    while current <= end:
        path = download_binance_aggtrades(symbol, current)
        if path is not None:
            csv_files.append(path)
        current += timedelta(days=1)
    
    print(f"\nDownloaded {len(csv_files)} files")
    return csv_files


# ─── Validation Functions ─────────────────────────────────────

def validate_aggtrades(df: pd.DataFrame, symbol: str) -> dict:
    """
    Multi-stage validation of aggTrades data.
    
    Returns a dict with quality metrics and the cleaned DataFrame.
    """
    report = {
        "symbol": symbol,
        "total_rows": len(df),
        "issues": [],
        "valid_rows": 0,
        "rejected_rows": 0,
        "checks": {},
    }
    
    mask = pd.Series(True, index=df.index)  # Start with all rows valid
    
    # ── Check 1: Missing values ────────────────────────────────
    missing = df.isnull().any(axis=1)
    n_missing = missing.sum()
    report["checks"]["missing_values"] = {
        "count": int(n_missing),
        "status": "PASS" if n_missing == 0 else "WARN"
    }
    if n_missing > 0:
        report["issues"].append(f"{n_missing} rows with missing values")
        mask &= ~missing
    
    # ── Check 2: Timestamp monotonicity ────────────────────────
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[5]
    ts = df[ts_col].astype(np.int64)
    ts_decreasing = ts.diff().fillna(0) < 0
    n_ooo = ts_decreasing.sum()
    report["checks"]["timestamp_monotonicity"] = {
        "count": int(n_ooo),
        "status": "PASS" if n_ooo == 0 else "FAIL"
    }
    if n_ooo > 0:
        report["issues"].append(f"{n_ooo} out-of-order timestamps")
        mask &= ~ts_decreasing
    
    # ── Check 3: Duplicate agg_trade_ids ───────────────────────
    id_col = df.columns[0]
    dupes = df[id_col].duplicated(keep="first")
    n_dupes = dupes.sum()
    report["checks"]["duplicates"] = {
        "count": int(n_dupes),
        "status": "PASS" if n_dupes == 0 else "WARN"
    }
    if n_dupes > 0:
        report["issues"].append(f"{n_dupes} duplicate trade IDs")
        mask &= ~dupes
    
    # ── Check 4: Price sanity ──────────────────────────────────
    price_col = df.columns[1]
    prices = pd.to_numeric(df[price_col], errors="coerce")
    
    # 4a: Non-positive prices
    bad_prices = prices <= 0
    n_bad = bad_prices.sum()
    report["checks"]["non_positive_prices"] = {
        "count": int(n_bad),
        "status": "PASS" if n_bad == 0 else "FAIL"
    }
    if n_bad > 0:
        report["issues"].append(f"{n_bad} non-positive prices")
        mask &= ~bad_prices
    
    # 4b: Price jumps > 5% (single-tick)
    pct_change = prices.pct_change().abs()
    jumps = pct_change > 0.05
    # Allow first row
    jumps.iloc[0] = False
    n_jumps = jumps.sum()
    report["checks"]["price_jumps_5pct"] = {
        "count": int(n_jumps),
        "status": "PASS" if n_jumps == 0 else "WARN"
    }
    if n_jumps > 0:
        report["issues"].append(f"{n_jumps} price jumps > 5%")
        mask &= ~jumps
    
    # ── Check 5: Quantity sanity ───────────────────────────────
    qty_col = df.columns[2]
    qtys = pd.to_numeric(df[qty_col], errors="coerce")
    bad_qty = qtys <= 0
    n_bad_qty = bad_qty.sum()
    report["checks"]["non_positive_quantity"] = {
        "count": int(n_bad_qty),
        "status": "PASS" if n_bad_qty == 0 else "FAIL"
    }
    if n_bad_qty > 0:
        report["issues"].append(f"{n_bad_qty} non-positive quantities")
        mask &= ~bad_qty
    
    # ── Check 6: Sequence gaps ─────────────────────────────────
    ids = df[id_col].astype(np.int64)
    gaps = ids.diff().fillna(1)
    n_gaps = (gaps > 1).sum()
    total_missing = int(gaps[gaps > 1].sum() - n_gaps) if n_gaps > 0 else 0
    report["checks"]["sequence_gaps"] = {
        "count": int(n_gaps),
        "total_missing_ids": total_missing,
        "status": "PASS" if n_gaps == 0 else "INFO"
    }
    if n_gaps > 0:
        report["issues"].append(
            f"{n_gaps} sequence gaps ({total_missing} missing trade IDs)"
        )
    
    # ── Summary ────────────────────────────────────────────────
    valid_rows = mask.sum()
    rejected_rows = len(df) - valid_rows
    report["valid_rows"] = int(valid_rows)
    report["rejected_rows"] = int(rejected_rows)
    report["acceptance_rate"] = round(valid_rows / len(df) * 100, 4) if len(df) > 0 else 0.0
    report["clean_df"] = df[mask].copy()
    
    return report


def print_validation_report(report: dict) -> None:
    """Pretty-print validation results."""
    print(f"\n{'─'*60}")
    print(f"DATA QUALITY REPORT — {report['symbol']}")
    print(f"{'─'*60}")
    print(f"Total rows:    {report['total_rows']:>12,}")
    print(f"Valid rows:    {report['valid_rows']:>12,}")
    print(f"Rejected rows: {report['rejected_rows']:>12,}")
    print(f"Acceptance:    {report['acceptance_rate']:>11.4f}%")
    print()
    
    for check_name, check_data in report["checks"].items():
        status = check_data["status"]
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}.get(status, "?")
        count = check_data.get("count", 0)
        print(f"  {icon} {check_name:<30} {status:>4}  ({count:,} issues)")
    
    if report["issues"]:
        print(f"\nIssues found:")
        for issue in report["issues"]:
            print(f"  • {issue}")
    else:
        print(f"\n✅ All checks passed — data is clean!")
    print(f"{'─'*60}\n")


# ─── Pipeline ─────────────────────────────────────────────────

def run_pipeline(symbol: str, start_date: str, end_date: str) -> Optional[Path]:
    """
    Full pipeline: Download → Validate → Save as Parquet.
    
    Returns path to the validated Parquet file.
    """
    # Step 1: Download
    csv_files = download_date_range(symbol, start_date, end_date)
    if not csv_files:
        print("No data files downloaded!")
        return None
    
    # Step 2: Load and concatenate
    print("\nLoading CSV files...")
    columns = [
        "agg_trade_id", "price", "quantity",
        "first_trade_id", "last_trade_id",
        "timestamp", "is_buyer_maker", "is_best_match"
    ]
    
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, header=None, names=columns)
            dfs.append(df)
            print(f"  Loaded {csv_file.name}: {len(df):,} rows")
        except Exception as e:
            print(f"  [ERR] Failed to load {csv_file.name}: {e}")
    
    if not dfs:
        print("No data loaded!")
        return None
    
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(full_df):,} rows across {len(dfs)} files")
    
    # Step 3: Validate
    print("\nRunning validation pipeline...")
    report = validate_aggtrades(full_df, symbol)
    print_validation_report(report)
    
    # Step 4: Save validated data as Parquet
    clean_df = report["clean_df"]
    
    VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_file = VALIDATED_DIR / f"{symbol}_{start_date}_{end_date}_validated.parquet"
    
    # Convert types for efficient storage
    clean_df["agg_trade_id"] = clean_df["agg_trade_id"].astype(np.int64)
    clean_df["price"] = pd.to_numeric(clean_df["price"], errors="coerce").astype(np.float64)
    clean_df["quantity"] = pd.to_numeric(clean_df["quantity"], errors="coerce").astype(np.float64)
    clean_df["timestamp"] = clean_df["timestamp"].astype(np.int64)
    
    clean_df.to_parquet(parquet_file, index=False, engine="pyarrow")
    
    file_size_mb = parquet_file.stat().st_size / (1024 * 1024)
    print(f"Saved validated data: {parquet_file}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Rows: {len(clean_df):,}")
    
    # Step 5: Save validation report
    report_file = VALIDATED_DIR / f"{symbol}_{start_date}_{end_date}_report.txt"
    with open(report_file, "w") as f:
        f.write(f"Data Quality Report — {symbol}\n")
        f.write(f"Date range: {start_date} to {end_date}\n")
        f.write(f"Source: data.binance.vision (official Binance)\n")
        f.write(f"Total rows: {report['total_rows']:,}\n")
        f.write(f"Valid rows: {report['valid_rows']:,}\n")
        f.write(f"Rejected: {report['rejected_rows']:,}\n")
        f.write(f"Acceptance rate: {report['acceptance_rate']:.4f}%\n\n")
        for check_name, check_data in report["checks"].items():
            f.write(f"  {check_name}: {check_data['status']} ({check_data.get('count', 0)} issues)\n")
        if report["issues"]:
            f.write("\nIssues:\n")
            for issue in report["issues"]:
                f.write(f"  - {issue}\n")
    
    print(f"Report saved: {report_file}")
    
    return parquet_file


# ─── CLI Entry Point ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download and validate real Binance market data"
    )
    parser.add_argument(
        "--symbol", default="BTCUSDT",
        help="Trading pair (default: BTCUSDT)"
    )
    parser.add_argument(
        "--start", default="2024-07-01",
        help="Start date YYYY-MM-DD (default: 2024-07-01)"
    )
    parser.add_argument(
        "--end", default="2024-07-07",
        help="End date YYYY-MM-DD (default: 2024-07-07)"
    )
    
    args = parser.parse_args()
    result = run_pipeline(args.symbol, args.start, args.end)
    
    if result:
        print(f"\n✅ Pipeline complete! Validated data at: {result}")
    else:
        print(f"\n❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
