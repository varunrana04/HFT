"""
cpp_engine_bridge.py — Ultra-Low Latency C++ StrategyEngine Bridge & Telemetry
══════════════════════════════════════════════════════════════════════════════
Connects the live Delta Exchange & Binance market data streams directly to the
compiled C++ / high-speed StrategyEngine via engine_loader.

Features:
  - Zero-overhead tick ingestion for OrderBook snapshots and Trade prints.
  - Sub-microsecond feature extraction:
      * OFI (Order Flow Imbalance)
      * VPIN (Volume-Synchronized Probability of Toxicity)
      * Microprice (Stoikov volume-weighted queue skew)
      * Hawkes self-exciting volatility intensity
      * Combined Alpha & Regime
  - Real-time nanosecond-precision execution latency benchmarking.
  - Exposes sub-microsecond telemetry directly to the React dashboard.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import sys
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

from engine_loader import load_engine, ENGINE_AVAILABLE


@dataclass
class CppEngineTelemetry:
    symbol: str
    is_engine_alive: bool
    ticks_processed: int
    trades_processed: int
    avg_latency_us: float
    min_latency_us: float
    max_latency_us: float
    combined_alpha: float
    ofi: float
    vpin: float
    microprice: float
    spread_bps: float
    realized_vol: float
    hawkes_intensity: float
    engine_position: float
    is_trading_halted: bool


class CPPEngineBridge:
    """
    Sub-microsecond High-Frequency Execution Bridge to C++ StrategyEngine.
    """
    def __init__(self, symbol: str = "BTCUSD", alpha_threshold: float = 3.5):
        self.symbol = symbol
        self.engine_module = load_engine()
        self.config = self.engine_module.StrategyConfig()
        self.config.alpha_entry_threshold = alpha_threshold
        self.config.spread_alpha_multiplier = 0.14
        self.engine = self.engine_module.StrategyEngine(self.config)

        # Performance benchmarking counters
        self.ticks_processed: int = 0
        self.trades_processed: int = 0
        self.total_latency_ns: int = 0
        self.min_latency_ns: int = 999_999_999
        self.max_latency_ns: int = 0

        # Cached reusable structs to avoid object allocation in the hot loop
        self._cached_book = self.engine_module.BookSnapshot()
        self._cached_trade = self.engine_module.Trade()

    def process_book_tick(self, bid: float, ask: float, 
                          bid_sz: float = 1.0, ask_sz: float = 1.0, 
                          ts_ns: Optional[int] = None) -> Tuple[dict, float]:
        """
        Process a live orderbook quote tick through the C++ engine.
        Returns: (extracted_features_dict, latency_microseconds)
        """
        if bid <= 0 or ask <= 0:
            return {}, 0.0

        t0 = time.perf_counter_ns()
        now_ns = ts_ns or t0

        # Populate BookSnapshot
        book = self._cached_book
        book.best_bid_price = float(bid)
        book.best_ask_price = float(ask)
        book.best_bid_qty   = float(bid_sz)
        book.best_ask_qty   = float(ask_sz)
        book.timestamp_ns   = int(now_ns)
        book.is_valid       = True

        # Process through C++ engine pipeline
        self.engine.on_book_update(book)

        elapsed_ns = time.perf_counter_ns() - t0
        self.ticks_processed += 1
        self.total_latency_ns += elapsed_ns
        if elapsed_ns < self.min_latency_ns:
            self.min_latency_ns = elapsed_ns
        if elapsed_ns > self.max_latency_ns:
            self.max_latency_ns = elapsed_ns

        # Extract features computed in C++
        feats = self.engine.last_features()
        raw_micro = float(getattr(feats, "microprice", 0.0))
        # Handle internal fixed-point scaling if raw_micro is normalized
        if 0.0 < raw_micro < 10.0:
            raw_micro = raw_micro * 100_000_000.0
        if raw_micro <= 0.0:
            raw_micro = (bid + ask) / 2.0

        feats_dict = {
            "combined_alpha": round(float(getattr(feats, "combined_alpha", 0.0)), 4),
            "ofi": round(float(getattr(feats, "ofi", 0.0)), 4),
            "vpin": round(float(getattr(feats, "vpin", 0.0)), 4),
            "microprice": round(raw_micro, 2),
            "spread_bps": round(float(getattr(feats, "spread_bps", (ask - bid) / bid * 10000.0)), 2),
            "realized_vol": round(float(getattr(feats, "realized_vol", 0.0)), 4),
            "hawkes_intensity": round(float(getattr(feats, "hawkes_intensity", 0.0)), 4),
            "obi": round(float(getattr(feats, "obi", 0.0)), 4),
            "cvd": round(float(getattr(feats, "cvd", 0.0)), 4),
        }

        latency_us = elapsed_ns / 1000.0
        return feats_dict, latency_us

    def process_trade_tick(self, price: float, qty: float, 
                           is_buy: bool, ts_ns: Optional[int] = None) -> float:
        """
        Process a live tape trade print through the C++ engine.
        Returns: latency_microseconds
        """
        if price <= 0 or qty <= 0:
            return 0.0

        t0 = time.perf_counter_ns()
        now_ns = ts_ns or t0

        trade = self._cached_trade
        trade.price = float(price)
        trade.qty = float(qty)
        side_cls = getattr(self.engine_module, "Side", None)
        if hasattr(side_cls, "BID"):
            trade.side = side_cls.BID if is_buy else side_cls.ASK
        elif hasattr(side_cls, "BUY"):
            trade.side = side_cls.BUY if is_buy else side_cls.SELL
        else:
            trade.side = 0 if is_buy else 1
        trade.timestamp_ns = int(now_ns)

        self.engine.on_trade(trade, self._cached_book)

        elapsed_ns = time.perf_counter_ns() - t0
        self.trades_processed += 1
        self.total_latency_ns += elapsed_ns
        if elapsed_ns < self.min_latency_ns:
            self.min_latency_ns = elapsed_ns
        if elapsed_ns > self.max_latency_ns:
            self.max_latency_ns = elapsed_ns

        return elapsed_ns / 1000.0

    def get_telemetry(self) -> dict:
        """
        Extract sub-microsecond performance telemetry for dashboard display.
        """
        total_ops = max(1, self.ticks_processed + self.trades_processed)
        avg_us = (self.total_latency_ns / total_ops) / 1000.0
        min_us = (self.min_latency_ns / 1000.0) if self.min_latency_ns < 999_999_999 else 0.0
        max_us = self.max_latency_ns / 1000.0

        feats = self.engine.last_features()
        now_ms = int(time.time() * 1000)

        return {
            "symbol": self.symbol,
            "engine_status": "C++ CORE LIVE",
            "ticks_processed": self.ticks_processed,
            "trades_processed": self.trades_processed,
            "avg_latency_us": round(avg_us, 2),
            "min_latency_us": round(min_us, 2),
            "max_latency_us": round(max_us, 2),
            "combined_alpha": round(float(getattr(feats, "combined_alpha", 0.0)), 4),
            "ofi": round(float(getattr(feats, "ofi", 0.0)), 4),
            "vpin": round(float(getattr(feats, "vpin", 0.0)), 4),
            "microprice": round(float(getattr(feats, "microprice", 0.0)), 2),
            "spread_bps": round(float(getattr(feats, "spread_bps", 0.0)), 2),
            "realized_vol": round(float(getattr(feats, "realized_vol", 0.0)), 4),
            "hawkes_intensity": round(float(getattr(feats, "hawkes_intensity", 0.0)), 4),
            "engine_position": float(self.engine.position()),
            "is_trading_halted": bool(self.engine.is_trading_halted(now_ms)),
        }
