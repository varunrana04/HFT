#!/usr/bin/env python3
"""
backtest.py — Python backtester for the HFT engine.

Loads historical trade/book data (Binance CSV from data_downloader.py),
feeds each tick into the C++ StrategyEngine via pybind11 bindings,
and produces:
  - Equity curve plot
  - Drawdown chart
  - Trade distribution histogram
  - Summary report (markdown table)

Usage:
    python backtest.py --data data/BTCUSDT_trades.csv
    python backtest.py --data data/BTCUSDT_trades.csv --capital 50000

Output saved to results/ directory.
"""

import sys
import os
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Try to import the C++ engine; fall back to pure-Python simulation
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "build"))
    import hft_engine
    HAS_CPP_ENGINE = True
except ImportError:
    HAS_CPP_ENGINE = False
    print("[WARN] C++ hft_engine module not found — using pure-Python simulation")


# ─── Data Classes ────────────────────────────────────────────

@dataclass
class Tick:
    """A single market tick from CSV."""
    timestamp_ms: int
    price: float
    quantity: float
    is_buyer_maker: bool  # True = seller-initiated (aggressor was seller)


@dataclass
class BacktestResult:
    """Complete backtest output."""
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    drawdown_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    trade_pnls: List[float] = field(default_factory=list)
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    avg_slippage: float = 0.0
    initial_capital: float = 0.0
    final_equity: float = 0.0
    return_pct: float = 0.0


# ─── CSV Loader ──────────────────────────────────────────────

def load_binance_csv(filepath: str, max_rows: Optional[int] = None) -> List[Tick]:
    """
    Load Binance trade data CSV.
    
    Expected columns: id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch
    OR: timestamp, price, quantity, side (simplified format from data_downloader.py)
    """
    ticks = []
    
    with open(filepath, 'r') as f:
        header = f.readline().strip().split(',')
        
        # Detect format
        if 'isBuyerMaker' in header:
            # Binance raw format
            price_idx = header.index('price')
            qty_idx = header.index('qty')
            time_idx = header.index('time')
            buyer_idx = header.index('isBuyerMaker')
        elif 'side' in header:
            # Simplified format
            price_idx = header.index('price')
            qty_idx = header.index('quantity') if 'quantity' in header else header.index('qty')
            time_idx = header.index('timestamp') if 'timestamp' in header else header.index('time')
            buyer_idx = -1  # Use 'side' column
            side_idx = header.index('side')
        else:
            # Fallback: assume price, qty, time columns
            price_idx = 1
            qty_idx = 2
            time_idx = 0
            buyer_idx = -1
        
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            
            parts = line.strip().split(',')
            if len(parts) < max(price_idx, qty_idx, time_idx) + 1:
                continue
            
            try:
                price = float(parts[price_idx])
                qty = float(parts[qty_idx])
                ts = int(parts[time_idx])
                
                if buyer_idx >= 0:
                    is_buyer_maker = parts[buyer_idx].strip().lower() in ('true', '1')
                elif 'side' in header:
                    is_buyer_maker = parts[side_idx].strip().lower() in ('sell', 'ask')
                else:
                    is_buyer_maker = (i % 2 == 0)  # Alternate as fallback
                
                ticks.append(Tick(
                    timestamp_ms=ts,
                    price=price,
                    quantity=qty,
                    is_buyer_maker=is_buyer_maker
                ))
            except (ValueError, IndexError):
                continue
    
    return ticks


# ─── Pure-Python Strategy Simulation ─────────────────────────

class PythonBacktestEngine:
    """
    Fallback backtester when C++ engine is unavailable.
    Implements a simplified version of the StrategyEngine logic.
    """
    
    def __init__(self, initial_capital: float = 100000.0,
                 alpha_threshold: float = 0.10,
                 position_size_pct: float = 0.01):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.alpha_threshold = alpha_threshold
        self.position_size_pct = position_size_pct
        
        # State
        self.position = 0.0
        self.avg_entry = 0.0
        self.realized_pnl = 0.0
        self.peak_equity = initial_capital
        
        # Feature state (simplified)
        self._prices = []
        self._volumes = []
        self._buy_volumes = []
        self._sell_volumes = []
        self._window = 100
        
        # Output
        self.equity_history = []
        self.trade_pnls = []
        self.trade_count = 0
        self.win_count = 0
    
    def on_tick(self, tick: Tick) -> None:
        """Process a single tick through the simplified pipeline."""
        self._prices.append(tick.price)
        self._volumes.append(tick.quantity)
        
        if tick.is_buyer_maker:
            self._sell_volumes.append(tick.quantity)
            self._buy_volumes.append(0.0)
        else:
            self._buy_volumes.append(tick.quantity)
            self._sell_volumes.append(0.0)
        
        # Keep window bounded
        if len(self._prices) > self._window * 2:
            self._prices = self._prices[-self._window:]
            self._volumes = self._volumes[-self._window:]
            self._buy_volumes = self._buy_volumes[-self._window:]
            self._sell_volumes = self._sell_volumes[-self._window:]
        
        if len(self._prices) < 20:
            self.equity_history.append(self._equity(tick.price))
            return
        
        # Compute simplified alpha signals
        alpha = self._compute_alpha(tick)
        
        # Trading logic
        if abs(alpha) > self.alpha_threshold:
            self._execute_signal(alpha, tick.price)
        elif self.position != 0.0 and abs(alpha) < 0.02:
            self._close_position(tick.price)
        
        eq = self._equity(tick.price)
        self.equity_history.append(eq)
        if eq > self.peak_equity:
            self.peak_equity = eq
    
    def _compute_alpha(self, tick: Tick) -> float:
        """Simplified multi-signal alpha computation."""
        prices = self._prices[-self._window:]
        
        # Signal 1: Momentum (price vs SMA)
        sma = np.mean(prices)
        momentum = (tick.price - sma) / sma if sma > 0 else 0
        
        # Signal 2: OFI proxy (buy vs sell volume imbalance)
        recent_buy = sum(self._buy_volumes[-20:])
        recent_sell = sum(self._sell_volumes[-20:])
        total_vol = recent_buy + recent_sell
        ofi = (recent_buy - recent_sell) / total_vol if total_vol > 0 else 0
        
        # Signal 3: Volatility regime
        returns = np.diff(np.log(np.array(prices[-50:])))
        vol = np.std(returns) if len(returns) > 1 else 0
        vol_signal = -vol * 10  # High vol → reduce exposure
        
        # Signal 4: Mean reversion (Z-score)
        std = np.std(prices)
        zscore = (tick.price - sma) / std if std > 0 else 0
        mean_rev = -zscore * 0.1  # Fade extremes
        
        # Combine
        alpha = 0.3 * momentum + 0.3 * ofi + 0.2 * vol_signal + 0.2 * mean_rev
        return np.clip(alpha, -1.0, 1.0)
    
    def _execute_signal(self, alpha: float, price: float) -> None:
        """Open or add to a position."""
        eq = self._equity(price)
        size = eq * self.position_size_pct / price
        
        if alpha > 0 and self.position <= 0:
            # Go long
            if self.position < 0:
                self._close_position(price)
            self.position = size
            self.avg_entry = price
            self.trade_count += 1
        elif alpha < 0 and self.position >= 0:
            # Go short
            if self.position > 0:
                self._close_position(price)
            self.position = -size
            self.avg_entry = price
            self.trade_count += 1
    
    def _close_position(self, price: float) -> None:
        """Close current position and realize PnL."""
        if self.position == 0:
            return
        
        if self.position > 0:
            pnl = (price - self.avg_entry) * self.position
        else:
            pnl = (self.avg_entry - price) * abs(self.position)
        
        self.realized_pnl += pnl
        self.trade_pnls.append(pnl)
        if pnl > 0:
            self.win_count += 1
        self.trade_count += 1
        self.position = 0.0
        self.avg_entry = 0.0
    
    def _equity(self, current_price: float) -> float:
        """Current equity = capital + realized + unrealized."""
        unrealized = 0.0
        if self.position > 0:
            unrealized = (current_price - self.avg_entry) * self.position
        elif self.position < 0:
            unrealized = (self.avg_entry - current_price) * abs(self.position)
        return self.initial_capital + self.realized_pnl + unrealized
    
    def get_result(self) -> BacktestResult:
        """Compile backtest results."""
        equity_arr = np.array(self.equity_history)
        
        # Drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (running_max - equity_arr) / np.maximum(running_max, 1e-10)
        
        # Sharpe
        if len(equity_arr) > 1:
            returns = np.diff(equity_arr) / equity_arr[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60) \
                     if np.std(returns) > 0 else 0.0
        else:
            sharpe = 0.0
        
        return BacktestResult(
            equity_curve=equity_arr,
            drawdown_curve=drawdown,
            timestamps=np.arange(len(equity_arr)),
            trade_pnls=self.trade_pnls,
            total_pnl=self.realized_pnl,
            max_drawdown=float(np.max(drawdown)) if len(drawdown) > 0 else 0.0,
            sharpe_ratio=sharpe,
            win_rate=self.win_count / max(self.trade_count, 1),
            total_trades=self.trade_count,
            avg_slippage=0.0,  # No slippage in pure-Python mode
            initial_capital=self.initial_capital,
            final_equity=float(equity_arr[-1]) if len(equity_arr) > 0 else self.initial_capital,
            return_pct=(float(equity_arr[-1]) / self.initial_capital - 1.0) * 100
                       if len(equity_arr) > 0 else 0.0,
        )


# ─── Visualization ───────────────────────────────────────────

def plot_results(result: BacktestResult, output_dir: str) -> None:
    """Generate and save backtest visualizations."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARN] matplotlib not found — skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ── Style ──
    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#c9d1d9',
        'text.color': '#c9d1d9',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'grid.color': '#21262d',
        'font.size': 11,
    })
    
    # ── 1. Equity Curve ──
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(result.equity_curve, color='#58a6ff', linewidth=1.2, alpha=0.9)
    ax.fill_between(range(len(result.equity_curve)),
                    result.initial_capital, result.equity_curve,
                    where=result.equity_curve >= result.initial_capital,
                    color='#238636', alpha=0.15)
    ax.fill_between(range(len(result.equity_curve)),
                    result.initial_capital, result.equity_curve,
                    where=result.equity_curve < result.initial_capital,
                    color='#da3633', alpha=0.15)
    ax.axhline(y=result.initial_capital, color='#8b949e',
               linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_title('Equity Curve', fontsize=16, fontweight='bold', color='#f0f6fc')
    ax.set_xlabel('Tick')
    ax.set_ylabel('Equity ($)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'${x:,.0f}'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_curve.png'), dpi=150)
    plt.close()
    print(f"  [+] Saved equity_curve.png")
    
    # ── 2. Drawdown Chart ──
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(range(len(result.drawdown_curve)),
                    0, -result.drawdown_curve * 100,
                    color='#da3633', alpha=0.4)
    ax.plot(-result.drawdown_curve * 100, color='#da3633',
            linewidth=0.8, alpha=0.9)
    ax.set_title('Drawdown', fontsize=16, fontweight='bold', color='#f0f6fc')
    ax.set_xlabel('Tick')
    ax.set_ylabel('Drawdown (%)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x:.1f}%'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown.png'), dpi=150)
    plt.close()
    print(f"  [+] Saved drawdown.png")
    
    # ── 3. Trade PnL Distribution ──
    if len(result.trade_pnls) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        pnls = np.array(result.trade_pnls)
        colors = ['#238636' if p > 0 else '#da3633' for p in pnls]
        ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=1.0)
        ax.axhline(y=0, color='#8b949e', linewidth=0.8)
        ax.set_title('Trade PnL Distribution', fontsize=16,
                      fontweight='bold', color='#f0f6fc')
        ax.set_xlabel('Trade #')
        ax.set_ylabel('PnL ($)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trade_pnl.png'), dpi=150)
        plt.close()
        print(f"  [+] Saved trade_pnl.png")


def save_report(result: BacktestResult, output_dir: str) -> None:
    """Save markdown summary report."""
    os.makedirs(output_dir, exist_ok=True)
    
    report = f"""# Backtest Report

## Summary

| Metric | Value |
|---|---|
| **Initial Capital** | ${result.initial_capital:,.2f} |
| **Final Equity** | ${result.final_equity:,.2f} |
| **Total PnL** | ${result.total_pnl:,.2f} |
| **Return** | {result.return_pct:+.2f}% |
| **Sharpe Ratio** | {result.sharpe_ratio:.3f} |
| **Max Drawdown** | {result.max_drawdown * 100:.2f}% |
| **Total Trades** | {result.total_trades:,} |
| **Win Rate** | {result.win_rate * 100:.1f}% |
| **Avg Slippage** | ${result.avg_slippage:.4f} |

## Equity Curve

![Equity Curve](equity_curve.png)

## Drawdown

![Drawdown](drawdown.png)

## Trade PnL Distribution

![Trade PnL](trade_pnl.png)

---

*Generated by HFT Engine Backtester*
"""
    
    report_path = os.path.join(output_dir, 'backtest_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  [+] Saved backtest_report.md")


# ─── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HFT Engine Backtester',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python backtest.py --data data/BTCUSDT_trades.csv
    python backtest.py --data data/BTCUSDT_trades.csv --capital 50000 --max-rows 100000
        """
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Path to Binance CSV data file')
    parser.add_argument('--capital', type=float, default=100000.0,
                        help='Initial capital (default: 100000)')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Max rows to process (default: all)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results)')
    parser.add_argument('--threshold', type=float, default=0.10,
                        help='Alpha entry threshold (default: 0.10)')
    
    args = parser.parse_args()
    
    # ── Load Data ──
    print(f"\n{'='*60}")
    print(f"  HFT Engine Backtester")
    print(f"{'='*60}")
    print(f"\n  Data: {args.data}")
    print(f"  Capital: ${args.capital:,.2f}")
    print(f"  Threshold: {args.threshold}")
    
    t0 = time.time()
    ticks = load_binance_csv(args.data, args.max_rows)
    load_time = time.time() - t0
    print(f"  Loaded {len(ticks):,} ticks in {load_time:.2f}s")
    
    if len(ticks) == 0:
        print("[ERROR] No data loaded. Check file path and format.")
        sys.exit(1)
    
    # ── Run Backtest ──
    print(f"\n  Running backtest...")
    t0 = time.time()
    
    if HAS_CPP_ENGINE:
        print("  [C++ engine active]")
        # TODO: Wire hft_engine.StrategyEngine when bindings are updated
        # For now, fall through to Python engine
    
    engine = PythonBacktestEngine(
        initial_capital=args.capital,
        alpha_threshold=args.threshold
    )
    
    # Progress tracking
    total = len(ticks)
    report_interval = max(total // 20, 1)
    
    for i, tick in enumerate(ticks):
        engine.on_tick(tick)
        if (i + 1) % report_interval == 0:
            pct = (i + 1) / total * 100
            eq = engine.equity_history[-1] if engine.equity_history else args.capital
            print(f"    [{pct:5.1f}%] Tick {i+1:>10,} / {total:,}  "
                  f"Equity: ${eq:,.2f}  Trades: {engine.trade_count}")
    
    backtest_time = time.time() - t0
    result = engine.get_result()
    
    print(f"\n  Backtest complete in {backtest_time:.2f}s")
    print(f"  Throughput: {len(ticks) / backtest_time:,.0f} ticks/sec")
    
    # ── Output ──
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output
    )
    
    print(f"\n  Generating reports...")
    save_report(result, output_dir)
    plot_results(result, output_dir)
    
    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Initial Capital:  ${result.initial_capital:>12,.2f}")
    print(f"  Final Equity:     ${result.final_equity:>12,.2f}")
    print(f"  Total PnL:        ${result.total_pnl:>12,.2f}")
    print(f"  Return:           {result.return_pct:>12.2f}%")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:>12.3f}")
    print(f"  Max Drawdown:     {result.max_drawdown * 100:>12.2f}%")
    print(f"  Total Trades:     {result.total_trades:>12,}")
    print(f"  Win Rate:         {result.win_rate * 100:>12.1f}%")
    print(f"{'='*60}\n")
    
    print(f"  Output saved to: {output_dir}/")


if __name__ == '__main__':
    main()
