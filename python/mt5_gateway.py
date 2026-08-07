#!/usr/bin/env python3
"""
mt5_gateway.py — MetaTrader 5 live demo execution gateway.

Connects to an MT5 demo account, streams live tick data into
the HFT strategy engine, and executes trades in real-time.

Requirements:
    pip install MetaTrader5 numpy

Usage:
    python mt5_gateway.py --symbol EURUSD --lot 0.01
    python mt5_gateway.py --symbol BTCUSD --lot 0.001 --magic 12345

The gateway:
    1. Initializes MT5 connection
    2. Streams tick data from the specified symbol
    3. Feeds ticks into the strategy engine (C++ or Python)
    4. Executes trades via MT5 order_send when signals fire
    5. Logs every order with timestamps and slippage
"""

import sys
import os
import time
import signal
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ─── Logging Setup ───────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('mt5_gateway')

# ─── MT5 Import ──────────────────────────────────────────────

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    logger.warning("MetaTrader5 package not found. Install with: pip install MetaTrader5")

# ─── C++ Engine Import ───────────────────────────────────────

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "build"))
    import hft_engine
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

# ─── Strategy Engine (Python Fallback) ───────────────────────

class SimpleMomentumStrategy:
    """
    Minimal momentum strategy for demo trading.
    Replace with C++ engine when bindings are updated.
    """
    
    def __init__(self, threshold: float = 0.0005, window: int = 50):
        self.threshold = threshold
        self.window = window
        self.prices = []
        self.last_signal = 0  # 1=buy, -1=sell, 0=flat
    
    def on_tick(self, bid: float, ask: float) -> int:
        """
        Process a tick and return signal.
        Returns: 1 (buy), -1 (sell), 0 (hold)
        """
        mid = (bid + ask) / 2
        self.prices.append(mid)
        
        if len(self.prices) > self.window * 2:
            self.prices = self.prices[-self.window:]
        
        if len(self.prices) < self.window:
            return 0
        
        prices = np.array(self.prices[-self.window:])
        sma = np.mean(prices)
        
        # Momentum signal
        momentum = (mid - sma) / sma
        
        # Mean reversion overlay
        std = np.std(prices)
        zscore = (mid - sma) / std if std > 0 else 0
        
        # Combined
        alpha = momentum - 0.1 * zscore
        
        if alpha > self.threshold and self.last_signal != 1:
            self.last_signal = 1
            return 1
        elif alpha < -self.threshold and self.last_signal != -1:
            self.last_signal = -1
            return -1
        
        return 0


# ─── MT5 Gateway ─────────────────────────────────────────────

class MT5Gateway:
    """
    Manages the connection to MetaTrader 5 and executes trades.
    """
    
    def __init__(self, symbol: str, lot_size: float = 0.01,
                 magic: int = 42000, slippage: int = 10):
        self.symbol = symbol
        self.lot_size = lot_size
        self.magic = magic
        self.slippage = slippage
        self.running = False
        
        # Stats
        self.tick_count = 0
        self.order_count = 0
        self.total_slippage = 0.0
        self.start_time = None
        
        # Trade log
        self.trade_log = []
    
    def connect(self) -> bool:
        """Initialize MT5 connection."""
        if not HAS_MT5:
            logger.error("MetaTrader5 not installed")
            return False
        
        if not mt5.initialize():
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False
        
        # Check symbol
        info = mt5.symbol_info(self.symbol)
        if info is None:
            logger.error(f"Symbol {self.symbol} not found")
            mt5.shutdown()
            return False
        
        if not info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logger.error(f"Failed to select {self.symbol}")
                mt5.shutdown()
                return False
        
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        
        logger.info(f"Connected to MT5")
        logger.info(f"  Terminal: {terminal.name}")
        logger.info(f"  Account: {account.login} ({account.server})")
        logger.info(f"  Balance: ${account.balance:,.2f}")
        logger.info(f"  Symbol:  {self.symbol}")
        logger.info(f"  Lot:     {self.lot_size}")
        
        return True
    
    def send_order(self, direction: int, price: float) -> Optional[dict]:
        """
        Send a market order to MT5.
        direction: 1 = buy, -1 = sell
        """
        if not HAS_MT5:
            return None
        
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": order_type,
            "price": price,
            "sl": 0.0,
            "tp": 0.0,
            "deviation": self.slippage,
            "magic": self.magic,
            "comment": f"HFT-Engine-{self.order_count}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        t0 = time.perf_counter_ns()
        result = mt5.order_send(request)
        latency_us = (time.perf_counter_ns() - t0) / 1000
        
        if result is None:
            logger.error(f"order_send returned None: {mt5.last_error()}")
            return None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning(f"Order rejected: code={result.retcode}, "
                         f"comment={result.comment}")
            return None
        
        # Calculate slippage
        actual_slippage = abs(result.price - price)
        self.total_slippage += actual_slippage
        self.order_count += 1
        
        trade_info = {
            'timestamp': datetime.now().isoformat(),
            'direction': 'BUY' if direction == 1 else 'SELL',
            'requested_price': price,
            'fill_price': result.price,
            'slippage': actual_slippage,
            'latency_us': latency_us,
            'order_id': result.order,
            'volume': result.volume,
        }
        self.trade_log.append(trade_info)
        
        logger.info(
            f"{'BUY ' if direction == 1 else 'SELL'} "
            f"{self.symbol} @ {result.price:.5f} "
            f"(req: {price:.5f}, slip: {actual_slippage:.5f}, "
            f"lat: {latency_us:.0f}µs)"
        )
        
        return trade_info
    
    def close_all(self) -> None:
        """Close all open positions for this magic number."""
        if not HAS_MT5:
            return
        
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return
        
        for pos in positions:
            if pos.magic != self.magic:
                continue
            
            close_type = (mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY
                         else mt5.ORDER_TYPE_BUY)
            tick = mt5.symbol_info_tick(self.symbol)
            price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": pos.ticket,
                "price": price,
                "deviation": self.slippage,
                "magic": self.magic,
                "comment": "HFT-Engine-Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Closed position #{pos.ticket}")
    
    def run(self, strategy, duration_sec: int = 3600) -> None:
        """
        Main event loop: stream ticks and execute signals.
        
        Args:
            strategy: Object with on_tick(bid, ask) -> int method
            duration_sec: How long to run (default: 1 hour)
        """
        self.running = True
        self.start_time = time.time()
        end_time = self.start_time + duration_sec
        
        logger.info(f"Starting live trading for {duration_sec}s...")
        logger.info(f"Press Ctrl+C to stop")
        
        # Signal handler for graceful shutdown
        def on_signal(sig, frame):
            self.running = False
            logger.info("Shutdown signal received")
        
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        
        last_tick_time = 0
        
        while self.running and time.time() < end_time:
            if not HAS_MT5:
                time.sleep(1)
                continue
            
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                time.sleep(0.001)
                continue
            
            # Skip duplicate ticks
            if tick.time_msc == last_tick_time:
                time.sleep(0.0001)  # 100µs sleep to avoid busy-wait
                continue
            
            last_tick_time = tick.time_msc
            self.tick_count += 1
            
            # Feed tick to strategy
            signal_val = strategy.on_tick(tick.bid, tick.ask)
            
            # Execute if signal
            if signal_val == 1:
                self.send_order(1, tick.ask)
            elif signal_val == -1:
                self.send_order(-1, tick.bid)
            
            # Log progress every 1000 ticks
            if self.tick_count % 1000 == 0:
                elapsed = time.time() - self.start_time
                logger.info(
                    f"Ticks: {self.tick_count:,}  "
                    f"Orders: {self.order_count}  "
                    f"Elapsed: {elapsed:.0f}s  "
                    f"Rate: {self.tick_count/elapsed:.0f} ticks/s"
                )
        
        # ── Shutdown ──
        logger.info("Stopping...")
        self.close_all()
        self.print_summary()
        
        if HAS_MT5:
            mt5.shutdown()
    
    def print_summary(self) -> None:
        """Print trading session summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_slippage = (self.total_slippage / self.order_count
                        if self.order_count > 0 else 0)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"  SESSION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"  Duration:      {elapsed:.0f}s")
        logger.info(f"  Ticks:         {self.tick_count:,}")
        logger.info(f"  Orders:        {self.order_count}")
        logger.info(f"  Avg Slippage:  {avg_slippage:.6f}")
        logger.info(f"  Tick Rate:     {self.tick_count/max(elapsed,1):.0f}/s")
        logger.info(f"{'='*50}")
    
    def save_log(self, filepath: str) -> None:
        """Save trade log to CSV."""
        if not self.trade_log:
            return
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            headers = self.trade_log[0].keys()
            f.write(','.join(headers) + '\n')
            for trade in self.trade_log:
                f.write(','.join(str(v) for v in trade.values()) + '\n')
        
        logger.info(f"Trade log saved to {filepath}")


# ─── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HFT Engine — MT5 Live Demo Gateway',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--symbol', type=str, default='EURUSD',
                        help='MT5 symbol (default: EURUSD)')
    parser.add_argument('--lot', type=float, default=0.01,
                        help='Lot size per trade (default: 0.01)')
    parser.add_argument('--magic', type=int, default=42000,
                        help='Magic number for trade tracking')
    parser.add_argument('--duration', type=int, default=3600,
                        help='Trading duration in seconds (default: 3600)')
    parser.add_argument('--threshold', type=float, default=0.0005,
                        help='Signal threshold (default: 0.0005)')
    parser.add_argument('--log', type=str, default='results/mt5_trades.csv',
                        help='Trade log output path')
    
    args = parser.parse_args()
    
    # ── Initialize Gateway ──
    gateway = MT5Gateway(
        symbol=args.symbol,
        lot_size=args.lot,
        magic=args.magic
    )
    
    if HAS_MT5:
        if not gateway.connect():
            sys.exit(1)
    else:
        logger.warning("Running in DRY-RUN mode (no MT5 connection)")
        logger.warning("Install MetaTrader5: pip install MetaTrader5")
    
    # ── Initialize Strategy ──
    strategy = SimpleMomentumStrategy(threshold=args.threshold)
    
    # ── Run ──
    try:
        gateway.run(strategy, duration_sec=args.duration)
    finally:
        gateway.save_log(args.log)


if __name__ == '__main__':
    main()
