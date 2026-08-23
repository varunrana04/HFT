#!/usr/bin/env python3
"""
process_local_zips.py — Process local Binance aggTrade ZIP files.

Reads all BTCUSDT-aggTrades-2024-*.zip files in data/,
extracts only the 4 required columns (timestamp, price, quantity, is_buyer_maker)
directly into a single merged BTCUSDT_2024.csv file,
and deletes the ZIP files as it goes to save space.
"""

import os
import csv
import zipfile
import io
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def process_zips():
    output_path = DATA_DIR / "BTCUSDT_2024.csv"
    
    # Find all zip files for 2024
    zip_files = sorted([f for f in DATA_DIR.glob("BTCUSDT-aggTrades-2024-*.zip")])
    if not zip_files:
        print("[ERROR] No ZIP files found in data/")
        return

    print("=" * 62)
    print(f"  Processing {len(zip_files)} ZIP files directly into CSV")
    print(f"  Output: {output_path}")
    print("=" * 62)

    total_rows = 0
    t_start = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["timestamp", "price", "quantity", "is_buyer_maker"])

        for zip_path in zip_files:
            print(f"\n── {zip_path.name} {'─' * (40 - len(zip_path.name))}")
            rows_written = 0
            
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                    if not csv_names:
                        print("  [ERR ] No CSV found inside ZIP")
                        continue

                    print(f"  [STREAM] {csv_names[0]} → output CSV", end=" ", flush=True)

                    with zf.open(csv_names[0]) as csv_file:
                        reader = csv.reader(io.TextIOWrapper(csv_file, encoding="utf-8"))
                        first = next(reader, None)
                        if first is None:
                            continue

                        # Check header
                        try:
                            float(first[1])
                            writer.writerow([first[5], first[1], first[2], first[6]])
                            rows_written += 1
                        except (ValueError, IndexError):
                            pass

                        for row in reader:
                            if len(row) < 7:
                                continue
                            try:
                                writer.writerow([row[5], row[1], row[2], row[6]])
                                rows_written += 1
                            except IndexError:
                                continue
                                
            except zipfile.BadZipFile as e:
                print(f"  [ERR ] Bad ZIP: {e}")
                continue

            print(f"({rows_written:,} rows)")
            total_rows += rows_written
            out_f.flush()

            # Delete the ZIP file after processing it to save space
            print(f"  [CLEAN ] Deleting {zip_path.name}...")
            zip_path.unlink(missing_ok=True)

            size_mb = output_path.stat().st_size / 1e6
            elapsed = time.time() - t_start
            print(f"  Running total: {total_rows:,} rows | "
                  f"output: {size_mb:.0f} MB | {elapsed:.0f}s elapsed")

    size_mb = output_path.stat().st_size / 1e6
    elapsed = time.time() - t_start

    print(f"\n{'=' * 62}")
    print(f"  PROCESSING COMPLETE")
    print(f"  Total rows : {total_rows:,}")
    print(f"  File size  : {size_mb:.0f} MB")
    print(f"  Time       : {elapsed:.0f}s")
    print(f"\n  Run the backtest:")
    print(f"    python python\\backtest.py --data {output_path}")
    print("=" * 62)

if __name__ == "__main__":
    process_zips()
