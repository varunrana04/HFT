#!/usr/bin/env python3
"""
analyze_paper.py — Post-run forensic analytics for paper trading sessions.

Usage:
    # Analyze the most recent (largest) paper trade file automatically:
    python scripts/analyze_paper.py

    # Analyze a specific file:
    python scripts/analyze_paper.py paper_trades_eb835e92.csv

    # Analyze all files matching a pattern:
    python scripts/analyze_paper.py --all

Output:
    - Per-session summary: PnL, Sharpe, win-rate, max drawdown
    - Fee breakdown: gross fees vs gross edge
    - Trade size analysis: average BTC per trade, notional
    - Equity curve statistics
    - Market impact estimate
    - Signal quality check (flag if too many trades cluster at same alpha)

BUG FIX (original script):
    start_equity was hardcoded to $1,000,000 but the live engine uses
    $10,000,000 — causing every PnL calculation to be wrong by $9M.
    This version auto-detects start_equity from the first CSV row's
    Equity column, so it works regardless of capital scale.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────
TAKER_FEE_BPS  = 1.5    # 1.5 bps per side for taker
MAKER_FEE_BPS  = -0.5   # -0.5 bps rebate for maker
BTC_ADV_USD    = 30_000 * 77_000  # ~$2.3B daily ADV on Binance BTC-USDT perp
ANNUALIZE_SQRT = 35_497.0         # sqrt(5e6 ticks/day * 252 days)


# ── File Discovery ─────────────────────────────────────────────────────────

def find_csv_files(root: str = None) -> list:
    """Find all paper_trades_*.csv files, sorted newest-first by size."""
    if root is None:
        root = os.path.join(os.path.dirname(__file__), '..')
    pattern = os.path.join(root, 'paper_trades*.csv')
    files = glob.glob(pattern)
    # Sort by file size descending so the richest file is first
    files.sort(key=lambda f: os.path.getsize(f), reverse=True)
    return files


def load_csv(path: str) -> pd.DataFrame:
    """Load and clean a paper_trades CSV. Auto-detects column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Normalise column names to lowercase for robustness
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)
    col_map = {c: c.lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    required = {'timestamp', 'side', 'price', 'qty', 'equity'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}  (got: {list(df.columns)})")

    df['price']     = pd.to_numeric(df['price'],     errors='coerce')
    df['qty']       = pd.to_numeric(df['qty'],       errors='coerce')
    df['equity']    = pd.to_numeric(df['equity'],    errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['price', 'qty', 'equity', 'timestamp']).copy()

    if df.empty:
        raise ValueError("No valid rows after parsing.")

    df['side'] = df['side'].str.upper().str.strip()
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


# ── Core Analytics ─────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame, filepath: str) -> dict:
    """Compute the full analytics suite for one trade log."""
    n = len(df)
    fname = os.path.basename(filepath)

    # ── Start / End equity (auto-detected, no hardcoding) ─────────────────
    start_equity = float(df['equity'].iloc[0])
    end_equity   = float(df['equity'].iloc[-1])
    net_pnl      = end_equity - start_equity
    net_pnl_bps  = net_pnl / start_equity * 10_000.0 if start_equity > 0 else 0.0

    # ── Duration ──────────────────────────────────────────────────────────
    t_start = df['timestamp'].iloc[0]
    t_end   = df['timestamp'].iloc[-1]
    duration_s = t_end - t_start
    duration_h = duration_s / 3600.0

    # ── Equity curve stats ────────────────────────────────────────────────
    equity_series = df['equity'].values.astype(np.float64)
    returns       = np.diff(equity_series) / np.maximum(equity_series[:-1], 1e-10)

    # Rolling max drawdown
    peak = np.maximum.accumulate(equity_series)
    dd   = (peak - equity_series) / np.maximum(peak, 1.0)
    max_drawdown_pct = float(np.max(dd)) * 100.0
    max_drawdown_usd = float(np.max(peak - equity_series))

    # Sharpe (tick-level, annualized)
    sharpe = 0.0
    if len(returns) > 2:
        mu  = np.mean(returns)
        sig = np.std(returns)
        if sig > 1e-15:
            sharpe = float(mu / sig * ANNUALIZE_SQRT)

    # ── Trade-level analytics ─────────────────────────────────────────────
    buys  = df[df['side'] == 'BUY']
    sells = df[df['side'] == 'SELL']

    avg_price    = float(df['price'].mean())
    avg_qty_btc  = float(df['qty'].mean())
    total_qty    = float(df['qty'].sum())
    avg_notional = avg_price * avg_qty_btc

    # ── Round-trip PnL per trade ──────────────────────────────────────────
    # Each row is one fill; compute per-row equity change as proxy for trade PnL
    df['equity_delta'] = df['equity'].diff().fillna(0.0)
    winners = df[df['equity_delta'] > 0]
    losers  = df[df['equity_delta'] < 0]
    flat    = df[df['equity_delta'] == 0]

    win_rate  = len(winners) / n if n > 0 else 0.0
    avg_win   = float(winners['equity_delta'].mean()) if len(winners) > 0 else 0.0
    avg_loss  = float(losers['equity_delta'].mean())  if len(losers)  > 0 else 0.0
    profit_factor = (
        abs(winners['equity_delta'].sum() / losers['equity_delta'].sum())
        if len(losers) > 0 and losers['equity_delta'].sum() != 0
        else float('inf')
    )

    # ── Fee estimate ──────────────────────────────────────────────────────
    # Assume taker fees (conservative) — actual fees depend on order type
    taker_fee_per_trade = avg_price * avg_qty_btc * (TAKER_FEE_BPS / 10_000.0)
    maker_fee_per_trade = avg_price * avg_qty_btc * (MAKER_FEE_BPS / 10_000.0)
    total_taker_fees    = taker_fee_per_trade * n
    total_maker_fees    = maker_fee_per_trade * n

    gross_edge = net_pnl + total_taker_fees  # Edge before paying fees

    # ── Market impact estimate (Almgren-Chriss sqrt law) ─────────────────
    # I(Q) = σ × η × sqrt(Q / ADV)
    # Use avg_qty in USD and BTC ADV in USD
    sigma_approx = float(np.std(np.diff(df['price'].values)) / avg_price) if n > 1 else 0.001
    eta          = 0.1
    avg_qty_usd  = avg_notional
    impact_bps   = sigma_approx * eta * np.sqrt(avg_qty_usd / BTC_ADV_USD) * 10_000.0
    impact_usd_per_trade = avg_qty_usd * impact_bps / 10_000.0

    # ── Consistency check: are trades clustering? ─────────────────────────
    price_changes = np.abs(np.diff(df['price'].values))
    pct_zero_move = float(np.mean(price_changes == 0.0)) * 100.0 if len(price_changes) > 0 else 0.0

    # ── Rate ──────────────────────────────────────────────────────────────
    trades_per_hour = n / max(duration_h, 1e-6)

    return {
        'file':                fname,
        'n_trades':            n,
        'n_buys':              len(buys),
        'n_sells':             len(sells),
        'duration_h':          round(duration_h, 2),
        'trades_per_hour':     round(trades_per_hour, 1),
        'start_equity':        round(start_equity, 2),
        'end_equity':          round(end_equity, 2),
        'net_pnl':             round(net_pnl, 2),
        'net_pnl_bps':         round(net_pnl_bps, 2),
        'max_drawdown_pct':    round(max_drawdown_pct, 4),
        'max_drawdown_usd':    round(max_drawdown_usd, 2),
        'sharpe':              round(sharpe, 3),
        'win_rate':            round(win_rate * 100, 1),
        'avg_win':             round(avg_win, 4),
        'avg_loss':            round(avg_loss, 4),
        'profit_factor':       round(profit_factor, 3),
        'avg_price':           round(avg_price, 2),
        'avg_qty_btc':         round(avg_qty_btc, 6),
        'avg_notional_usd':    round(avg_notional, 2),
        'total_volume_btc':    round(total_qty, 4),
        'est_taker_fees':      round(total_taker_fees, 2),
        'est_maker_fees':      round(total_maker_fees, 2),
        'gross_edge':          round(gross_edge, 2),
        'impact_bps':          round(impact_bps, 4),
        'impact_usd_per_trade': round(impact_usd_per_trade, 4),
        'pct_zero_price_move': round(pct_zero_move, 1),
    }


# ── Printing ───────────────────────────────────────────────────────────────

def print_report(r: dict):
    """Pretty-print one session analytics report."""
    line = '─' * 62
    print(f"\n{'═'*62}")
    print(f"  SESSION ANALYTICS  —  {r['file']}")
    print(f"{'═'*62}")

    # ── Overview ──────────────────────────────────────────────────────────
    print(f"\n  {'OVERVIEW':}")
    print(f"  {line}")
    print(f"  {'Trades':30}  {r['n_trades']:>10,}  ({r['n_buys']} BUY / {r['n_sells']} SELL)")
    print(f"  {'Duration':30}  {r['duration_h']:>10.1f}  hours")
    print(f"  {'Trade rate':30}  {r['trades_per_hour']:>10.1f}  trades/hour")

    # ── PnL ───────────────────────────────────────────────────────────────
    print(f"\n  {'P & L':}")
    print(f"  {line}")
    pnl_arrow = '▲' if r['net_pnl'] >= 0 else '▼'
    print(f"  {'Start equity':30}  ${r['start_equity']:>12,.2f}")
    print(f"  {'End equity':30}  ${r['end_equity']:>12,.2f}")
    print(f"  {'Net PnL':30}  {pnl_arrow} ${r['net_pnl']:>+11,.2f}  ({r['net_pnl_bps']:+.1f} bps)")
    print(f"  {'Max drawdown':30}  ${r['max_drawdown_usd']:>12,.2f}  ({r['max_drawdown_pct']:.4f}%)")
    print(f"  {'Sharpe (annualized)':30}  {r['sharpe']:>12.3f}")

    # ── Trade quality ─────────────────────────────────────────────────────
    print(f"\n  {'TRADE QUALITY':}")
    print(f"  {line}")
    print(f"  {'Win rate':30}  {r['win_rate']:>11.1f}%")
    print(f"  {'Avg win per fill':30}  ${r['avg_win']:>+12.4f}")
    print(f"  {'Avg loss per fill':30}  ${r['avg_loss']:>+12.4f}")
    print(f"  {'Profit factor':30}  {r['profit_factor']:>12.3f}")

    # ── Sizing ────────────────────────────────────────────────────────────
    print(f"\n  {'SIZING & MARKET IMPACT':}")
    print(f"  {line}")
    print(f"  {'Avg trade size':30}  {r['avg_qty_btc']:>10.6f}  BTC")
    print(f"  {'Avg notional':30}  ${r['avg_notional_usd']:>12,.2f}")
    print(f"  {'Total volume':30}  {r['total_volume_btc']:>10.4f}  BTC")
    print(f"  {'Est. impact (sqrt law)':30}  {r['impact_bps']:>10.4f}  bps / trade")
    print(f"  {'Impact cost / trade':30}  ${r['impact_usd_per_trade']:>12.4f}")

    # ── Fees ──────────────────────────────────────────────────────────────
    print(f"\n  {'FEES (ESTIMATED)':}")
    print(f"  {line}")
    print(f"  {'Total taker fees (1.5 bps)':30}  ${r['est_taker_fees']:>12,.2f}")
    print(f"  {'Total maker rebates (-0.5)':30}  ${r['est_maker_fees']:>12,.2f}")
    print(f"  {'Gross edge (PnL+fees)':30}  ${r['gross_edge']:>+12,.2f}")

    # ── Edge check ────────────────────────────────────────────────────────
    edge_per_trade_bps = (r['net_pnl'] / max(r['n_trades'], 1)) / r['avg_notional_usd'] * 10_000.0 if r['avg_notional_usd'] > 0 else 0.0
    theoretical_edge_bps = 3.5
    fee_hurdle_bps       = 3.0  # round-trip taker

    print(f"\n  {'EDGE DIAGNOSTICS':}")
    print(f"  {line}")
    print(f"  {'Net edge / trade':30}  {edge_per_trade_bps:>10.2f}  bps")
    print(f"  {'Theoretical edge':30}  {theoretical_edge_bps:>10.1f}  bps")
    print(f"  {'Fee hurdle (RT taker)':30}  {fee_hurdle_bps:>10.1f}  bps")
    print(f"  {'Remaining after fees':30}  {theoretical_edge_bps - fee_hurdle_bps:>10.1f}  bps")

    # ── Flags ─────────────────────────────────────────────────────────────
    flags = []
    if r['net_pnl'] < 0:
        flags.append(f"NEGATIVE PnL: ${r['net_pnl']:+,.2f}")
    if r['max_drawdown_pct'] > 2.0:
        flags.append(f"HIGH DRAWDOWN: {r['max_drawdown_pct']:.2f}%  (>2% threshold)")
    if r['win_rate'] < 40.0:
        flags.append(f"LOW WIN RATE: {r['win_rate']:.1f}%  (expected >50%)")
    if r['profit_factor'] < 1.0:
        flags.append("PROFIT FACTOR <1.0 — losing strategy")
    if r['avg_qty_btc'] > 5.0:
        flags.append(f"LARGE ORDER SIZE: {r['avg_qty_btc']:.4f} BTC avg — check market impact")
    if r['pct_zero_price_move'] > 30.0:
        flags.append(f"CLUSTERING: {r['pct_zero_price_move']:.0f}% of trades at same price — possible over-trading")
    if r['trades_per_hour'] > 5000:
        flags.append(f"HIGH FREQUENCY: {r['trades_per_hour']:.0f} trades/h — check for runaway loop")
    if r['impact_bps'] > r['net_pnl_bps'] and r['n_trades'] > 0:
        flags.append(f"MARKET IMPACT ({r['impact_bps']:.2f} bps) > NET EDGE ({r['net_pnl_bps']:.2f} bps)")

    if flags:
        print(f"\n  {'⚠  FLAGS':}")
        print(f"  {line}")
        for f in flags:
            print(f"  ⚠  {f}")
    else:
        print(f"\n  ✓  No critical flags detected.")

    print(f"\n{'═'*62}\n")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HFT Paper Trade Analytics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('file', nargs='?', default=None,
                        help='Path to paper_trades_*.csv (default: largest file found)')
    parser.add_argument('--all', action='store_true',
                        help='Analyze ALL paper_trades_*.csv files found')
    parser.add_argument('--root', type=str, default=None,
                        help='Search root directory (default: parent of scripts/)')
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    elif args.all:
        files = find_csv_files(args.root)
        if not files:
            print("[ERROR] No paper_trades_*.csv files found.")
            sys.exit(1)
        print(f"[INFO] Found {len(files)} paper trade files.")
        # For --all, skip tiny files (header-only = 1 line, no trades)
        files = [f for f in files if os.path.getsize(f) > 200]
    else:
        # Default: pick the most recent file with actual trades (>200 bytes)
        root = args.root or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        all_files = sorted(
            glob.glob(os.path.join(root, 'paper_trades*.csv')),
            key=os.path.getmtime, reverse=True
        )
        files = [f for f in all_files if os.path.getsize(f) > 200]
        if not files:
            print("[ERROR] No paper_trades_*.csv files with data found.")
            sys.exit(1)
        files = [files[0]]  # Most recent with data

    results = []
    for fpath in files:
        try:
            df = load_csv(fpath)
            r  = analyze(df, fpath)
            results.append(r)
            print_report(r)
        except Exception as e:
            print(f"[WARN] Could not analyze {os.path.basename(fpath)}: {e}")

    # ── Cross-session summary if multiple files ────────────────────────────
    if len(results) > 1:
        total_pnl    = sum(r['net_pnl']    for r in results)
        total_trades = sum(r['n_trades']   for r in results)
        total_fees   = sum(r['est_taker_fees'] for r in results)
        all_sharpes  = [r['sharpe'] for r in results if r['n_trades'] > 10]

        print(f"\n{'═'*62}")
        print(f"  CROSS-SESSION SUMMARY  ({len(results)} sessions)")
        print(f"{'═'*62}")
        print(f"  {'Total trades':30}  {total_trades:>10,}")
        print(f"  {'Total PnL':30}  ${total_pnl:>+12,.2f}")
        print(f"  {'Total taker fees (est.)':30}  ${total_fees:>12,.2f}")
        print(f"  {'Avg Sharpe across sessions':30}  {np.mean(all_sharpes):>12.3f}")
        print(f"{'═'*62}\n")


if __name__ == '__main__':
    main()
