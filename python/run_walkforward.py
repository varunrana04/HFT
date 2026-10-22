"""
HFT Walk-Forward Launcher
=========================
Auto-detects cached data. Downloads if missing. Then runs walk-forward.

Usage (from C:\\Users\\Varun\\Downloads\\HFT):
    python run_walkforward.py                          # 1-month, LGBM
    python run_walkforward.py --period 1month          # 1-month, LGBM
    python run_walkforward.py --period 1year           # 1-year, LGBM
    python run_walkforward.py --model both             # LGBM + RL
    python run_walkforward.py --period custom --start 2026-08-25 --end 2026-08-29
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_DIR   = Path("nautilus_cache")
DOWNLOADER  = Path("python/nautilus/download_walkforward_data.py")
SIM_SCRIPT  = Path("python/nautilus/hft_extended_walkforward.py")
WORKERS     = 6

PERIOD_DAYS = {
    "1month": 30,
    "1year":  365,
}

def count_cached(cache_dir: Path) -> tuple[int, str, str]:
    """Return (count, first_date, last_date) from cache."""
    files = sorted(cache_dir.glob("BTCUSDT-aggTrades-*.parquet"))
    if not files:
        return 0, "", ""
    dates = [f.stem.split("-aggTrades-")[1] for f in files]
    return len(files), dates[0], dates[-1]

def need_download(period: str, start: str, end: str, cached: int) -> bool:
    """Return True if we need to download more data."""
    if cached == 0:
        return True
    required = PERIOD_DAYS.get(period, 0)
    if period == "custom":
        d1 = datetime.strptime(start, "%Y-%m-%d")
        d2 = datetime.strptime(end,   "%Y-%m-%d")
        required = (d2 - d1).days + 1
    # Allow 10% slack (weekends / gaps)
    return cached < int(required * 0.9)

def run(cmd: list, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[WARN] Process exited with code {result.returncode}", flush=True)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="HFT Walk-Forward Launcher")
    parser.add_argument("--period", default="1month",
                        choices=["1month", "1year", "custom"],
                        help="Simulation period (default: 1month)")
    parser.add_argument("--model", default="lgbm",
                        choices=["lgbm", "rl", "both"],
                        help="Model to run (default: lgbm)")
    parser.add_argument("--start", default="",
                        help="Custom start date YYYY-MM-DD (--period custom)")
    parser.add_argument("--end", default="",
                        help="Custom end date YYYY-MM-DD (--period custom)")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="Download parallel workers (default: 6)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download check, go straight to sim")
    args = parser.parse_args()

    py = sys.executable

    # ── Check Cache ────────────────────────────────────────────────────────
    CACHE_DIR.mkdir(exist_ok=True)
    cached, first, last = count_cached(CACHE_DIR)

    print("\n" + "="*60)
    print("  HFT Walk-Forward Launcher")
    print("="*60)
    print(f"  Period  : {args.period}")
    print(f"  Model   : {args.model}")
    print(f"  Cache   : {cached} days", end="")
    if cached:
        print(f"  ({first} → {last})", end="")
    print()

    # ── Download if needed ─────────────────────────────────────────────────
    if not args.skip_download and need_download(args.period, args.start, args.end, cached):
        required = PERIOD_DAYS.get(args.period, 0)
        if args.period == "custom":
            d1 = datetime.strptime(args.start, "%Y-%m-%d")
            d2 = datetime.strptime(args.end,   "%Y-%m-%d")
            required = (d2 - d1).days + 1

        print(f"\n  Need {required} days, have {cached}. Starting download...")

        dl_cmd = [py, str(DOWNLOADER),
                  "--period", args.period,
                  "--workers", str(args.workers)]
        if args.period == "custom":
            dl_cmd += ["--start", args.start, "--end", args.end]

        rc = run(dl_cmd, f"DOWNLOAD: {args.period} BTCUSDT aggTrades")
        if rc != 0:
            print("\n[ERROR] Download failed. Aborting.")
            sys.exit(1)

        # Re-check after download
        cached, first, last = count_cached(CACHE_DIR)
        print(f"\n  Cache after download: {cached} days ({first} → {last})")
    else:
        print(f"\n  Cache sufficient ({cached} days). Skipping download.")

    # ── Run Walk-Forward Sim ───────────────────────────────────────────────
    sim_cmd = [py, "-u", str(SIM_SCRIPT),
               "--model", args.model,
               "--period", args.period]
    if args.period == "custom":
        sim_cmd += ["--start", args.start, "--end", args.end]

    rc = run(sim_cmd, f"WALK-FORWARD SIM: {args.model.upper()} | {args.period}")

    # ── Show saved results ─────────────────────────────────────────────────
    results_dir = Path("hft_nt_results")
    if results_dir.exists():
        csvs = sorted(results_dir.glob(f"wf_{args.model}_{args.period}*.csv"))
        summary = results_dir / f"wf_{args.model}_{args.period}_summary.json"
        print(f"\n{'='*60}")
        print("  SAVED RESULTS:")
        for f in csvs:
            size_kb = f.stat().st_size // 1024
            print(f"    {f.name}  ({size_kb} KB)")
        if summary.exists():
            import json
            with open(summary) as fh:
                data = json.load(fh)
            print(f"\n  SUMMARY:")
            for k, v in data.items():
                if isinstance(v, float):
                    print(f"    {k:30s}: {v:.4f}")
                else:
                    print(f"    {k:30s}: {v}")
        print("="*60)

    sys.exit(rc)

if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
