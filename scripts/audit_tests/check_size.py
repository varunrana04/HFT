#!/usr/bin/env python3
"""
check_size.py — Position Size & Market Impact Audit Tool

Answers the key question before live deployment:
  "Is our intended trade size feasible at institutional scale,
   and what is the realistic all-in cost at each size tier?"

Usage:
    # Audit default $10M portfolio at current BTC price
    python scripts/check_size.py

    # Specific portfolio size and price
    python scripts/check_size.py --capital 10000000 --price 77000

    # Audit sizes extracted from an actual paper trade log
    python scripts/check_size.py --log paper_trades_eb835e92.csv

    # Full sweep: show cost table at multiple alpha levels
    python scripts/check_size.py --sweep

Output:
    - Per-size-tier table: notional, BTC qty, fees, market impact, net edge
    - Kelly-scaled size grid vs alpha strength
    - Warning flags for sizes that wipe out edge
    - Comparison vs actual trades in log file (if --log provided)

Background on the $14k drawdown:
    The original engine used flat 15% sizing (19.4 BTC at $10M / $77k BTC).
    At 1.5 bps taker fee = $231 per side = $462 round-trip.
    With a theoretical 3.5 bps edge = $539 round-trip gross edge.
    Net = $539 - $462 = $77 per trade.  But slippage on 19.4 BTC ≈ 1.2 bps
    additional cost → net edge goes to near-zero, any adverse tick → loss.

    With Kelly-scaled sizing at alpha=0.10:
        size_pct = 0.5 * 0.10 = 5%  →  $500k  →  6.5 BTC
        taker fee = $75 per side = $150 round-trip
        gross edge (3.5 bps * $500k) = $175
        net edge = $175 - $150 = $25 per trade  (positive, small but safe)
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_CAPITAL    = 10_000_000.0   # $10M paper portfolio
DEFAULT_BTC_PRICE  = 77_000.0       # Conservative BTC price for sizing
BTC_ADV_USD        = 2_300_000_000  # ~$2.3B daily Binance BTC-USDT perp ADV
SIGMA_1MIN         = 0.0003         # ~3 bps typical 1-min BTC vol
ETA                = 0.1            # Almgren-Chriss impact coefficient
TAKER_FEE_BPS      = 1.5
MAKER_FEE_BPS      = -0.5           # rebate
THEORETICAL_EDGE_BPS = 3.5          # known strategy edge before costs
MAX_POSITION_PCT   = 0.15           # hard cap
MIN_SIZE_PCT       = 0.005          # 0.5% floor


# ── Market Impact ──────────────────────────────────────────────────────────

def sqrt_impact_bps(qty_usd: float, sigma: float = SIGMA_1MIN,
                    eta: float = ETA, adv_usd: float = BTC_ADV_USD) -> float:
    """
    Almgren-Chriss square-root law:
        I(Q) = σ × η × sqrt(Q / ADV)  [in raw units, × 10000 = bps]
    """
    return sigma * eta * np.sqrt(qty_usd / adv_usd) * 10_000.0


# ── Kelly Sizing Grid ──────────────────────────────────────────────────────

def kelly_size_pct(alpha: float) -> float:
    """Half-Kelly position size fraction from signal strength."""
    return float(np.clip(0.5 * abs(alpha), MIN_SIZE_PCT, MAX_POSITION_PCT))


# ── Size Tier Analysis ─────────────────────────────────────────────────────

def analyze_size_tier(
        pct: float,
        capital: float,
        btc_price: float,
        fee_bps: float,
        label: str = '') -> dict:
    """Compute full cost breakdown for one position size."""
    notional     = capital * pct
    qty_btc      = notional / btc_price
    impact_bps   = sqrt_impact_bps(notional)
    fee_usd_one  = notional * fee_bps / 10_000.0
    fee_rt       = fee_usd_one * 2            # round-trip
    impact_usd   = notional * impact_bps / 10_000.0
    gross_edge   = notional * THEORETICAL_EDGE_BPS / 10_000.0  # per trade
    total_cost   = fee_rt + impact_usd
    net_edge     = gross_edge - total_cost
    net_edge_bps = net_edge / notional * 10_000.0 if notional > 0 else 0.0

    return {
        'label':          label,
        'pct':            pct * 100,
        'notional_usd':   round(notional),
        'qty_btc':        round(qty_btc, 4),
        'fee_per_side':   round(fee_usd_one, 2),
        'fee_rt':         round(fee_rt, 2),
        'impact_bps':     round(impact_bps, 4),
        'impact_usd':     round(impact_usd, 4),
        'gross_edge':     round(gross_edge, 2),
        'total_cost':     round(total_cost, 2),
        'net_edge':       round(net_edge, 2),
        'net_edge_bps':   round(net_edge_bps, 4),
        'viable':         net_edge > 0,
    }


def print_size_table(rows: list, title: str):
    """Print a formatted cost breakdown table."""
    print(f"\n{'═'*80}")
    print(f"  {title}")
    print(f"{'═'*80}")
    header = (f"  {'Label':22} {'%Portf':>6} {'Notional':>12} {'BTC Qty':>9} "
              f"{'Fee RT':>9} {'Impact':>8} {'Net Edge':>10} {'Status':>7}")
    print(header)
    print(f"  {'─'*76}")
    for r in rows:
        status = '✓ OK' if r['viable'] else '✗ LOSS'
        print(
            f"  {r['label']:22} {r['pct']:>5.1f}%"
            f" ${r['notional_usd']:>11,}"
            f" {r['qty_btc']:>9.4f}"
            f" ${r['fee_rt']:>8,.2f}"
            f" {r['impact_bps']:>7.2f}bp"
            f" ${r['net_edge']:>+9,.2f}"
            f"  {status}"
        )
    print(f"{'─'*80}")


def print_kelly_grid(capital: float, btc_price: float, fee_bps: float):
    """Show Kelly-scaled size vs alpha strength."""
    alphas = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00]
    print(f"\n{'═'*80}")
    print(f"  KELLY-SCALED SIZING GRID  (capital=${capital:,.0f}, BTC=${btc_price:,.0f})")
    print(f"{'═'*80}")
    header = (f"  {'Alpha':>8} {'Size%':>7} {'Notional':>13} {'BTC':>9} "
              f"{'Fee RT':>10} {'Impact':>9} {'Net Edge':>11} {'Status':>7}")
    print(header)
    print(f"  {'─'*76}")
    for a in alphas:
        pct = kelly_size_pct(a)
        r   = analyze_size_tier(pct, capital, btc_price, fee_bps)
        status = '✓' if r['viable'] else '✗'
        print(
            f"  α={a:>5.2f}  {pct*100:>5.1f}%"
            f"  ${r['notional_usd']:>12,}"
            f"  {r['qty_btc']:>8.4f} BTC"
            f"  ${r['fee_rt']:>8,.2f}"
            f"  {r['impact_bps']:>7.3f} bps"
            f"  ${r['net_edge']:>+9,.2f}"
            f"  {status}"
        )
    print(f"{'─'*80}")
    print(
        f"\n  Note: net_edge = gross_edge({THEORETICAL_EDGE_BPS} bps) "
        f"- round-trip_fee({fee_bps*2:.1f} bps) - market_impact\n"
    )


# ── Log File Audit ─────────────────────────────────────────────────────────

def audit_log(filepath: str, capital: float, btc_price: float, fee_bps: float):
    """Read an actual paper trade log and audit what sizes were actually traded."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['qty']   = pd.to_numeric(df['qty'],   errors='coerce')
    df = df.dropna(subset=['price', 'qty'])

    if df.empty:
        print(f"[WARN] No valid rows in {filepath}")
        return

    df['notional'] = df['price'] * df['qty']

    print(f"\n{'═'*80}")
    print(f"  LOG AUDIT  —  {os.path.basename(filepath)}")
    print(f"{'═'*80}")

    qty_stats  = df['qty'].describe()
    not_stats  = df['notional'].describe()

    print(f"\n  Trade Quantity (BTC):")
    print(f"    Mean   : {qty_stats['mean']:>12.6f}  BTC")
    print(f"    Median : {df['qty'].median():>12.6f}  BTC")
    print(f"    Std    : {qty_stats['std']:>12.6f}  BTC")
    print(f"    Min    : {qty_stats['min']:>12.6f}  BTC")
    print(f"    Max    : {qty_stats['max']:>12.6f}  BTC")
    print(f"    P95    : {df['qty'].quantile(0.95):>12.6f}  BTC")

    print(f"\n  Trade Notional (USD):")
    print(f"    Mean   : ${not_stats['mean']:>12,.2f}")
    print(f"    Median : ${df['notional'].median():>12,.2f}")
    print(f"    Max    : ${not_stats['max']:>12,.2f}")
    print(f"    P95    : ${df['notional'].quantile(0.95):>12,.2f}")

    mean_notional = float(not_stats['mean'])
    mean_pct      = mean_notional / capital * 100.0
    impact_bps    = sqrt_impact_bps(mean_notional)
    fee_rt        = mean_notional * fee_bps * 2 / 10_000.0
    gross_edge    = mean_notional * THEORETICAL_EDGE_BPS / 10_000.0
    net_edge      = gross_edge - fee_rt - mean_notional * impact_bps / 10_000.0

    print(f"\n  Cost Analysis at Mean Trade Size:")
    print(f"    Portfolio %   : {mean_pct:>8.2f}%")
    print(f"    Gross edge    : ${gross_edge:>+10.4f}")
    print(f"    RT fees       : ${fee_rt:>+10.4f}")
    print(f"    Market impact : {impact_bps:>8.4f} bps")
    print(f"    Net edge/trade: ${net_edge:>+10.4f}")

    if net_edge < 0:
        print(f"\n  ⚠  NEGATIVE NET EDGE at this trade size.")
        print(f"     To break even, need at least "
              f"{(fee_rt / mean_notional * 10_000):.2f} bps edge (current: {THEORETICAL_EDGE_BPS} bps available).")
    else:
        print(f"\n  ✓  Positive net edge: ${net_edge:.4f} / trade")

    print()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Position size and market impact audit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--capital', type=float, default=DEFAULT_CAPITAL,
                        help=f'Portfolio size in USD (default: ${DEFAULT_CAPITAL:,.0f})')
    parser.add_argument('--price',   type=float, default=DEFAULT_BTC_PRICE,
                        help=f'BTC price in USD (default: ${DEFAULT_BTC_PRICE:,.0f})')
    parser.add_argument('--log',     type=str,   default=None,
                        help='Path to paper_trades_*.csv to audit actual sizes')
    parser.add_argument('--sweep',   action='store_true',
                        help='Show full alpha sweep table')
    parser.add_argument('--maker',   action='store_true',
                        help='Use maker rebate fee model instead of taker')
    args = parser.parse_args()

    fee_bps = MAKER_FEE_BPS if args.maker else TAKER_FEE_BPS
    fee_label = 'MAKER (-0.5 bps rebate)' if args.maker else 'TAKER (+1.5 bps)'

    print(f"\n{'═'*80}")
    print(f"  HFT ENGINE — POSITION SIZE & MARKET IMPACT AUDIT")
    print(f"{'═'*80}")
    print(f"  Portfolio:    ${args.capital:>15,.0f}")
    print(f"  BTC price:    ${args.price:>15,.0f}")
    print(f"  Fee model:    {fee_label}")
    print(f"  Strategy edge: {THEORETICAL_EDGE_BPS} bps (estimated)")
    print(f"  ADV (Binance): ${BTC_ADV_USD:>15,.0f}")

    # ── Standard size tier table ───────────────────────────────────────────
    tiers = [
        (0.005, 'Floor (0.5%)'),
        (0.025, 'Low (2.5%)'),
        (0.050, 'Medium (5%)'),
        (0.075, 'Mid-high (7.5%)'),
        (0.100, 'High (10%)'),
        (0.150, 'Max/flat 15%'),
        (0.200, 'Over-sized (20%)'),
        (0.250, 'Extreme (25%)'),
    ]
    rows = [analyze_size_tier(pct, args.capital, args.price, fee_bps, lbl)
            for pct, lbl in tiers]
    print_size_table(rows, f'SIZE TIER ANALYSIS  (fee={fee_bps:+.1f} bps, capital=${args.capital:,.0f})')

    # ── Key insight box ────────────────────────────────────────────────────
    breakeven_row = next((r for r in rows if r['viable']), None)
    first_loss    = next((r for r in rows if not r['viable']), None)
    print(f"\n  KEY INSIGHT:")
    if breakeven_row:
        print(f"  ✓  Minimum viable size: {breakeven_row['pct']:.1f}%  "
              f"(${breakeven_row['notional_usd']:,.0f} / {breakeven_row['qty_btc']:.4f} BTC)")
    if first_loss:
        print(f"  ✗  Edge wiped out at:  {first_loss['pct']:.1f}%  "
              f"(${first_loss['notional_usd']:,.0f} / {first_loss['qty_btc']:.4f} BTC)")

    # ── Kelly grid ────────────────────────────────────────────────────────
    if args.sweep:
        print_kelly_grid(args.capital, args.price, fee_bps)
    else:
        # Mini Kelly grid: show the practically relevant alpha range
        print(f"\n  {'─'*78}")
        print(f"  KELLY SIZING PREVIEW  (run with --sweep for full grid)")
        for a in [0.05, 0.10, 0.20, 0.50]:
            pct = kelly_size_pct(a)
            r   = analyze_size_tier(pct, args.capital, args.price, fee_bps)
            flag = '✓' if r['viable'] else '✗'
            print(f"    α={a:.2f} → {pct*100:.1f}%  =  {r['qty_btc']:.4f} BTC"
                  f"  |  net edge ${r['net_edge']:+.2f}  {flag}")

    # ── Log audit ─────────────────────────────────────────────────────────
    if args.log:
        audit_log(args.log, args.capital, args.price, fee_bps)
    else:
        # Auto-find the largest paper trade file
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        all_csvs = sorted(
            glob.glob(os.path.join(root, 'paper_trades*.csv')),
            key=os.path.getsize, reverse=True
        )
        valid_csvs = [f for f in all_csvs if os.path.getsize(f) > 500]
        if valid_csvs:
            print(f"\n  Auto-detected log: {os.path.basename(valid_csvs[0])}")
            audit_log(valid_csvs[0], args.capital, args.price, fee_bps)


if __name__ == '__main__':
    main()
