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

Feature dumping (for ML training):
    python backtest.py --data data/BTCUSDT_2024.csv --dump-features data/features.csv
    python backtest.py --data data/BTCUSDT_2024.csv --dump-features data/features.csv --horizon 50

    This runs the full C++ engine, captures every normalized FeatureVector,
    and labels each row with the actual signed forward return at `horizon` ticks.
    The output CSV is the ground-truth training set for train_model.py.

Output saved to results/ directory.
"""

import sys
import os
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

import numpy as np
from statsmodels.tsa.stattools import adfuller

def is_stationary(price_series, window=200, p_threshold=0.05):
    if len(price_series) < window:
        return False
    _, p_value, _, _, _, _ = adfuller(price_series[-window:], maxlag=10, autolag='AIC')
    return p_value < p_threshold  # reject unit root -> stationary -> Z valid

# Use the centralised engine loader — handles DLL paths + version checks
try:
    from engine_loader import load_engine
    _eng = load_engine(silent=True)
    if _eng is not None:
        hft_engine = _eng
        HAS_CPP_ENGINE = True
    else:
        HAS_CPP_ENGINE = False
        print("[WARN] C++ hft_engine not found — rebuild with build_python_bridge.ps1")
except Exception as e:
    HAS_CPP_ENGINE = False
    print(f"[WARN] C++ engine load failed: {e}")


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



# ─── Tardis L2 Data Parsers ──────────────────────────────────

def read_tardis_trades(filepath: str):
    import csv
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_us = int(row['timestamp'])
            ts_ns = ts_us * 1000
            price = float(row['price'])
            qty = float(row['amount'])
            is_sell = (row['side'].lower() == 'sell')
            yield (ts_ns, is_sell, price, qty)

def read_tardis_book(filepath: str, depth: int = 10):
    import csv
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_us = int(row['timestamp'])
            ts_ns = ts_us * 1000
            
            asks = []
            bids = []
            
            for i in range(depth):
                ask_p_key = f'asks[{i}].price'
                ask_q_key = f'asks[{i}].amount'
                bid_p_key = f'bids[{i}].price'
                bid_q_key = f'bids[{i}].amount'
                
                if ask_p_key in row and row[ask_p_key]:
                    asks.append((float(row[ask_p_key]), float(row[ask_q_key])))
                if bid_p_key in row and row[bid_p_key]:
                    bids.append((float(row[bid_p_key]), float(row[bid_q_key])))
            
            yield (ts_ns, asks, bids)

# ─── C++ Engine Backtester ───────────────────────────────────


class CppBacktestEngine:
    """
    Full backtester using the C++ StrategyEngine via pybind11.
    Converts CSV Tick objects to hft_engine.Trade + BookSnapshot and feeds
    them through the complete C++ pipeline (FeatureEngine → SignalCombiner
    → RiskManager → OrderManager) at ~13.6µs per tick.
    """

    def __init__(self,
                 initial_capital: float = 10_000_000.0,
                 alpha_threshold: float = 0.10,
                 alpha_exit_threshold: float = 0.02,
                 position_size_pct: float = 0.05,
                 inventory_skew_factor: float = 0.5,
                 queue_delay_prob: float = 1.0,
                 k_arrival_rate: float = 12.9,
                 disable_funding: bool = False):
        scfg = hft_engine.StrategyConfig()
        scfg.initial_capital       = initial_capital
        scfg.alpha_entry_threshold = alpha_threshold
        scfg.alpha_exit_threshold  = alpha_exit_threshold
        scfg.max_position_pct      = position_size_pct
        scfg.max_open_orders       = 10
        scfg.allow_short           = True
        scfg.fill_prob_dampener    = queue_delay_prob
        scfg.k_arrival_rate        = k_arrival_rate
        # Funding rate will be controlled by engine logic, but let's assume it's C++ logic. If we want to disable funding we can just not do it, but C++ handles it natively.
        
        fcfg = hft_engine.FeatureConfig()   # defaults are well-tuned
        rcfg = hft_engine.RiskConfig()
        rcfg.max_single_order_pct = 0.05    # allow up to 5% per order

        self.engine = hft_engine.StrategyEngine(scfg, fcfg, rcfg)
        
        # Load optimized weights if available
        weight_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'optimal_weights.bin'))
        if os.path.exists(weight_path):
            success = self.engine.load_model(weight_path)
            if success:
                print(f"  [+] Loaded custom ML weights from: {weight_path}")
            else:
                print(f"  [!] Failed to load custom ML weights from: {weight_path}")
        else:
            print("  [!] No custom ML weights found. Using default equal weights (0.1667).")

        self.initial_capital = initial_capital
        self.equity_history: List[float] = []
        self.price_history: List[float] = []
        self.regime_model = None
        self.tick_count = 0

    def on_tick(self, tick: 'Tick') -> None:
        """Convert a CSV Tick → C++ Trade + BookSnapshot and feed the engine."""
        ts_ns = tick.timestamp_ms * 1_000_000  # ms → ns

        # ── Trade ──
        t = hft_engine.Trade()
        t.timestamp_ns = ts_ns
        t.price        = hft_engine.price_to_fixed(tick.price)
        t.quantity     = hft_engine.qty_to_fixed(tick.quantity)
        # is_buyer_maker=True means the aggressor was the seller → ASK-side trade
        t.side = hft_engine.Side.ASK if tick.is_buyer_maker else hft_engine.Side.BID

        # ── Synthetic book: tight 1-bps spread around trade price ──
        half_spread = tick.price * 0.00005  # 0.5 bps each side
        b = hft_engine.BookSnapshot()
        b.timestamp_ns   = ts_ns
        b.best_bid_price = hft_engine.price_to_fixed(tick.price - half_spread)
        b.best_ask_price = hft_engine.price_to_fixed(tick.price + half_spread)
        b.best_bid_qty   = hft_engine.qty_to_fixed(tick.quantity)
        b.best_ask_qty   = hft_engine.qty_to_fixed(tick.quantity)
        
        b_level = hft_engine.PriceLevel()
        b_level.price = b.best_bid_price
        b_level.quantity = b.best_bid_qty
        b_level.order_count = 1
        
        a_level = hft_engine.PriceLevel()
        a_level.price = b.best_ask_price
        a_level.quantity = b.best_ask_qty
        a_level.order_count = 1
        
        b.bids = [b_level]
        b.asks = [a_level]
        b.quality = hft_engine.DataQuality.VALID

        self.engine.on_trade(t, b)
        self.equity_history.append(self.engine.equity())
        self.tick_count += 1


    def run_l2_replay(self, trades_gen, books_gen, start_time_ns=0, max_trades=None) -> None:
        import time
        import numpy as np
        
        print("  [TardisReplay] Starting synchronized L2 replay...")
        start_t = time.time()
        
        next_trade = next(trades_gen, None)
        next_book = next(books_gen, None)
        latest_book_obj = None
        
        books_processed = 0
        trades_processed = 0
        
        while next_trade is not None or next_book is not None:
            # Skip if earlier than start_time_ns
            if next_book is not None and next_book[0] < start_time_ns:
                next_book = next(books_gen, None)
                continue
            if next_trade is not None and next_trade[0] < start_time_ns:
                next_trade = next(trades_gen, None)
                continue
                
            if max_trades and trades_processed >= max_trades:
                break
                
            trade_ts = next_trade[0] if next_trade else float('inf')
            book_ts = next_book[0] if next_book else float('inf')
            
            if book_ts <= trade_ts:
                # Process Book Update
                ts_ns, asks, bids = next_book
                
                b = hft_engine.BookSnapshot()
                b.timestamp_ns = ts_ns
                b.quality = hft_engine.DataQuality.VALID
                
                if len(bids) > 0 and len(asks) > 0:
                    b.best_bid_price = hft_engine.price_to_fixed(bids[0][0])
                    b.best_ask_price = hft_engine.price_to_fixed(asks[0][0])
                    b.best_bid_qty   = hft_engine.qty_to_fixed(bids[0][1])
                    b.best_ask_qty   = hft_engine.qty_to_fixed(asks[0][1])
                
                bids_list = []
                for px, qty in bids:
                    lvl = hft_engine.PriceLevel()
                    lvl.price = hft_engine.price_to_fixed(px)
                    lvl.quantity = hft_engine.qty_to_fixed(qty)
                    lvl.order_count = 1
                    bids_list.append(lvl)
                b.bids = bids_list
                    
                asks_list = []
                for px, qty in asks:
                    lvl = hft_engine.PriceLevel()
                    lvl.price = hft_engine.price_to_fixed(px)
                    lvl.quantity = hft_engine.qty_to_fixed(qty)
                    lvl.order_count = 1
                    asks_list.append(lvl)
                b.asks = asks_list
                
                latest_book_obj = b
                self.engine.on_book_update(b)
                self.equity_history.append(self.engine.equity())
                
                books_processed += 1
                next_book = next(books_gen, None)
                
            else:
                # Process Trade
                ts_ns, is_sell, price, qty = next_trade
                
                t = hft_engine.Trade()
                t.timestamp_ns = ts_ns
                t.price        = hft_engine.price_to_fixed(price)
                t.quantity     = hft_engine.qty_to_fixed(qty)
                t.side         = hft_engine.Side.ASK if is_sell else hft_engine.Side.BID
                
                if latest_book_obj is not None:
                    t_book = hft_engine.BookSnapshot()
                    t_book.timestamp_ns = max(latest_book_obj.timestamp_ns, ts_ns)
                    t_book.best_bid_price = latest_book_obj.best_bid_price
                    t_book.best_ask_price = latest_book_obj.best_ask_price
                    t_book.best_bid_qty = latest_book_obj.best_bid_qty
                    t_book.best_ask_qty = latest_book_obj.best_ask_qty
                    t_book.quality = latest_book_obj.quality
                    t_book.bids = latest_book_obj.bids
                    t_book.asks = latest_book_obj.asks
                    
                    self.engine.on_trade(t, t_book)
                    self.equity_history.append(self.engine.equity())
                    trades_processed += 1
                    
                    if len(self.price_history) < 2000:
                        self.price_history.append(price)
                    else:
                        self.price_history.pop(0)
                        self.price_history.append(price)
                        
                    if trades_processed % 50000 == 0:
                        print(f"      Processed {books_processed:,} books and {trades_processed:,} trades...")
                        
                    if trades_processed % 1000 == 0 and len(self.price_history) >= 200:
                        try:
                            from statsmodels.tsa.stattools import adfuller
                            res = adfuller(np.array(self.price_history))
                            p_value = res[1]
                            if p_value < 0.05:
                                self.engine.set_stat_arb_valid(True)
                            else:
                                self.engine.set_stat_arb_valid(False)
                        except Exception:
                            self.engine.set_stat_arb_valid(False)
                            
                        if self.regime_model and 'model' in self.regime_model:
                            try:
                                fv = self.engine.last_features()
                                model = self.regime_model['model']
                                mean = self.regime_model['mean']
                                std = self.regime_model['std']
                                X_curr = np.array([[fv.realized_vol, fv.spread_bps]])
                                X_scaled = (X_curr - mean) / std
                                state = model.predict(X_scaled)[0]
                            except Exception:
                                pass
                
                next_trade = next(trades_gen, None)
                
        elapsed = time.time() - start_t
        print(f"  [TardisReplay] Processed {books_processed:,} books and {trades_processed:,} trades in {elapsed:.2f}s")
        
    def get_result(self) -> 'BacktestResult':
        """Pull metrics directly from the C++ engine."""
        equity_arr = np.array(self.equity_history)
        m = self.engine.metrics()

        # Drawdown curve
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (running_max - equity_arr) / np.maximum(running_max, 1e-10)

        journal = self.engine.trade_journal()
        trade_pnls = [r.pnl for r in journal]

        final_eq = float(equity_arr[-1]) if len(equity_arr) > 0 else self.initial_capital
        return BacktestResult(
            equity_curve=equity_arr,
            drawdown_curve=drawdown,
            timestamps=np.arange(len(equity_arr)),
            trade_pnls=trade_pnls,
            total_pnl=m.total_pnl,
            max_drawdown=m.max_drawdown,
            sharpe_ratio=m.sharpe_ratio,
            win_rate=m.win_rate,
            total_trades=int(m.total_trades),
            avg_slippage=m.avg_slippage,
            initial_capital=self.initial_capital,
            final_equity=final_eq,
            return_pct=(final_eq / self.initial_capital - 1.0) * 100,
        )


# ─── Lightweight FeatureVector snapshot ─────────────────────
# Stores only the 8 scalar fields we need — avoids holding a
# reference to the C++ pybind11 object (which is reused each tick)
# and is ~80 bytes vs ~500 bytes for a Python dict.
from collections import namedtuple
_FVSnapshot = namedtuple('_FVSnapshot', [
    'microprice', 'ofi', 'vpin', 'spread_bps',
    'realized_vol', 'stat_arb_zscore', 'combined_alpha', 'regime',
])

# ─── Feature Dumper ──────────────────────────────────────────

class FeatureDumper:
    """
    Runs the C++ StrategyEngine over historical ticks, captures every
    normalized FeatureVector, and labels each row with the actual
    signed forward return at `horizon` ticks.

    Design:
      - Uses a deque-based rolling horizon buffer so look-ahead labeling
        is O(1) per tick (no O(N²) numpy slicing over the full dataset).
      - The engine's own normalization (OnlineNormalizer, warm-up gate)
        is active, so features in the CSV exactly match what the live
        engine will see at inference time — training/inference consistency.
      - Rows during warm-up are still recorded but flagged (is_warmed_up=0)
        so train_model.py can filter them out if desired.
      - Mid-price column is included so walk_forward.py can compute
        per-fold statistics without re-running the engine.

    Output CSV columns:
        timestamp_ns, microprice, ofi, vpin, spread_bps, realized_vol,
        stat_arb_zscore, combined_alpha, regime, mid_price,
        forward_return_<horizon>, is_warmed_up
    """

    FEATURE_COLS = [
        'microprice', 'ofi', 'vpin', 'spread_bps',
        'realized_vol', 'stat_arb_zscore',
    ]

    def __init__(self, horizon: int = 100,
                 warmup_ticks: int = 1000,
                 initial_capital: float = 100_000.0):
        if not HAS_CPP_ENGINE:
            raise RuntimeError("C++ engine required for feature dumping")

        self.horizon = horizon

        scfg = hft_engine.StrategyConfig()
        scfg.initial_capital       = initial_capital
        scfg.alpha_entry_threshold = 1e9   # Never trade during dump
        scfg.alpha_exit_threshold  = 1e9
        scfg.min_warmup_ticks      = warmup_ticks

        fcfg = hft_engine.FeatureConfig()
        fcfg.normalizer_min_obs = 50
        fcfg.normalizer_clamp   = 3.0

        rcfg = hft_engine.RiskConfig()
        self.engine = hft_engine.StrategyEngine(scfg, fcfg, rcfg)

        # Rolling deque: holds only the last `horizon` (mid, row_values)
        # tuples — O(horizon) memory regardless of dataset size.
        # Rows are written to disk immediately when labeled; nothing
        # is accumulated in RAM.
        self._pending: deque = deque()   # deque of (mid_price, row_values_tuple)
        self._output_path: str  = ''
        self._file              = None   # open file handle during dump
        self._writer            = None   # csv.writer
        self._cols: List[str]   = []
        self._rows_written: int = 0
        self._target_col: str   = f'forward_return_{horizon}'

    # Column order — fixed so every row is consistent
    COLS = [
        'timestamp_ns', 'microprice', 'ofi', 'vpin', 'spread_bps',
        'realized_vol', 'stat_arb_zscore', 'combined_alpha', 'regime',
        'mid_price', 'forward_return_{horizon}', 'is_warmed_up',
    ]

    def open(self, output_path: str) -> None:
        """Open the output CSV and write the header. Must call before on_tick()."""
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        self._output_path = output_path
        self._cols = [c.format(horizon=self.horizon) for c in self.COLS]
        # Large write buffer (8 MB) to minimise syscalls on 30M rows
        self._file   = open(output_path, 'w', encoding='utf-8',
                            buffering=8 << 20, newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(self._cols)

    def _write_row(self, ts_ns: int, fv, mid: float,
                   fwd_ret: float, warmed: int) -> None:
        """Write one complete labeled row directly to disk."""
        self._writer.writerow([
            ts_ns,
            fv.microprice,
            fv.ofi,
            fv.vpin,
            fv.spread_bps,
            fv.realized_vol,
            fv.stat_arb_zscore,
            fv.combined_alpha,
            int(fv.regime),
            mid,
            fwd_ret,
            warmed,
        ])
        self._rows_written += 1

    def on_tick(self, tick: 'Tick') -> None:
        """Process one tick: compute features and stream to disk when labeled."""
        # Auto-detect if timestamp is in ms (13 digits) or us (16 digits)
        if tick.timestamp_ms > 1e14:
            ts_ns = tick.timestamp_ms * 1_000  # It's actually microseconds
        else:
            ts_ns = tick.timestamp_ms * 1_000_000 # It's milliseconds

        t = hft_engine.Trade()
        t.timestamp_ns = ts_ns
        t.price        = hft_engine.price_to_fixed(tick.price)
        t.quantity     = hft_engine.qty_to_fixed(tick.quantity)
        t.side         = hft_engine.Side.ASK if tick.is_buyer_maker else hft_engine.Side.BID

        half_spread = tick.price * 0.00005
        b = hft_engine.BookSnapshot()
        b.timestamp_ns   = ts_ns
        b.best_bid_price = hft_engine.price_to_fixed(tick.price - half_spread)
        b.best_ask_price = hft_engine.price_to_fixed(tick.price + half_spread)
        b.best_bid_qty   = hft_engine.qty_to_fixed(tick.quantity)
        b.best_ask_qty   = hft_engine.qty_to_fixed(tick.quantity)
        bl = hft_engine.PriceLevel()
        bl.price = b.best_bid_price; bl.quantity = b.best_bid_qty; bl.order_count = 1
        al = hft_engine.PriceLevel()
        al.price = b.best_ask_price; al.quantity = b.best_ask_qty; al.order_count = 1
        b.bids = [bl]; b.asks = [al]
        b.quality = hft_engine.DataQuality.VALID

        self.engine.on_trade(t, b)
        fv      = self.engine.last_features()
        mid     = tick.price
        warmed  = int(self.engine.is_warmed_up())

        # When the pending buffer is full, label the oldest row and
        # write it immediately to disk — zero accumulation in RAM.
        if len(self._pending) >= self.horizon:
            past_mid, past_ts, past_fv, past_warmed = self._pending.popleft()
            fwd_ret = (mid - past_mid) / past_mid if past_mid > 0 else 0.0
            self._write_row(past_ts, past_fv, past_mid, fwd_ret, past_warmed)

        # Store only the minimal state needed to label this row later:
        # (mid, ts_ns, fv snapshot as tuple, warmed flag)
        # We copy fv field values — the C++ object is reused each tick.
        self._pending.append((
            mid, ts_ns,
            _FVSnapshot(fv.microprice, fv.ofi, fv.vpin, fv.spread_bps,
                        fv.realized_vol, fv.stat_arb_zscore,
                        fv.combined_alpha, int(fv.regime)),
            warmed,
        ))

    def flush(self) -> None:
        """Flush remaining unlabeled rows (last `horizon` ticks) with NaN forward return."""
        while self._pending:
            past_mid, past_ts, past_fv, past_warmed = self._pending.popleft()
            self._write_row(past_ts, past_fv, past_mid, float('nan'), past_warmed)

    def close(self) -> int:
        """Close the file and return number of rows written."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
        return self._rows_written

    def save(self, output_path: str) -> int:
        """Legacy compatibility shim — not used in streaming mode."""
        return self._rows_written


def dump_features_to_csv(ticks: List['Tick'],
                          output_path: str,
                          horizon: int = 100,
                          warmup_ticks: int = 1000,
                          initial_capital: float = 100_000.0) -> int:
    """
    Stream normalized feature vectors + forward return labels directly
    to disk as they are computed. Peak RAM usage = O(horizon) rows
    regardless of dataset size — safe for 30M row datasets.
    """
    dumper = FeatureDumper(horizon=horizon, warmup_ticks=warmup_ticks,
                           initial_capital=initial_capital)
    dumper.open(output_path)

    total = len(ticks)
    report_interval = max(total // 20, 1)

    t0 = time.time()
    for i, tick in enumerate(ticks):
        dumper.on_tick(tick)
        if (i + 1) % report_interval == 0:
            pct     = (i + 1) / total * 100
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (total - i - 1) / rate
            print(f"    [{pct:5.1f}%]  {i+1:>10,} / {total:,}  "
                  f"Rate: {rate:,.0f} ticks/s  ETA: {eta:.0f}s  "
                  f"Written: {dumper._rows_written:,}")

    dumper.flush()
    n_rows = dumper.close()

    elapsed = time.time() - t0
    print(f"\n  Feature dump complete in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Total rows written: {n_rows:,}")
    print(f"  Output: {output_path}")
    return n_rows


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
        is_stat = is_stationary(prices)
        mean_rev = -zscore * 0.1 if is_stat else 0.0  # Fade extremes only if stationary
        
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

    # Downsample for plotting if too large (>1M points)
    max_plot_points = 500000
    if len(result.equity_curve) > max_plot_points:
        step = len(result.equity_curve) // max_plot_points
        plot_equity = result.equity_curve[::step]
        plot_drawdown = result.drawdown_curve[::step]
    else:
        plot_equity = result.equity_curve
        plot_drawdown = result.drawdown_curve
    
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
    ax.plot(plot_equity, color='#58a6ff', linewidth=1.2, alpha=0.9)
    ax.fill_between(range(len(plot_equity)),
                    result.initial_capital, plot_equity,
                    where=plot_equity >= result.initial_capital,
                    color='#238636', alpha=0.15)
    ax.fill_between(range(len(plot_equity)),
                    result.initial_capital, plot_equity,
                    where=plot_equity < result.initial_capital,
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
    ax.fill_between(range(len(plot_drawdown)), 0, plot_drawdown * 100, 
                    color='#da3633', alpha=0.3)
    ax.plot(plot_drawdown * 100, color='#da3633', linewidth=0.8, alpha=0.9)
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
    python backtest.py --data data/BTCUSDT_trades.csv
        """
    )
    parser.add_argument('--data', type=str,
                        help='Path to Binance trades CSV (or simplified Trades CSV)')
    parser.add_argument('--book', type=str, default=None, help='Path to Tardis L2 book CSV (enables L2 ingestion)')
    parser.add_argument('--capital', type=float, default=10000000.0,
                        help='Initial capital (default: 10000000)')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Max rows to process (default: all)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results)')
    parser.add_argument('--threshold', type=float, default=0.25,
                        help='Alpha entry threshold (default: 0.25)')
    parser.add_argument('--dump-features', type=str, default=None,
                        metavar='OUTPUT_CSV',
                        help='Dump normalized feature vectors + forward returns to CSV '
                             'for ML training. Skips normal backtest when specified.')
    parser.add_argument('--start-time', type=int, default=0,
                        help='Start time in nanoseconds for Tardis L2 replay (default: 0)')
    parser.add_argument('--horizon', type=int, default=100,
                        help='Forward return horizon in ticks for feature labeling '
                             '(default: 100, used with --dump-features)')
    parser.add_argument('--warmup', type=int, default=1000,
                        help='Warm-up ticks before feature generation (default: 1000)')
    parser.add_argument('--skew', type=float, default=0.5,
                        help='Inventory skew factor (default: 0.5)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Queue delay probability (default: 1.0 = instant fill)')
    parser.add_argument('--benchmark-avx2', action='store_true',
                        help='Run an AVX2 SIMD backtester benchmark on 10 million rows and exit')
    
    args = parser.parse_args()
    
    # ── Load Data ──
    print(f"\n{'='*60}")
    print(f"  HFT Engine Backtester")
    print(f"{'='*60}")
    print(f"\n  Data: {args.data}")
    print(f"  Capital: ${args.capital:,.2f}")
    print(f"  Threshold: {args.threshold}")
    
    if args.benchmark_avx2:
        if not HAS_CPP_ENGINE:
            print("[ERROR] C++ engine required for AVX2 benchmarking.")
            sys.exit(1)
        print(f"\n  [AVX2 Benchmark] Generating 10,000,000 synthetic feature rows...")
        import numpy as np
        N = 10_000_000
        
        # Test 1: Simple dot product loop (sanity check)
        a = np.random.randn(256).astype(np.float64)
        b = np.random.randn(256).astype(np.float64)
        print(f"  [AVX2 Benchmark] Running 1,000,000 iterations of simd_dot_product...")
        t0_bench = time.time()
        for _ in range(1_000_000):
            _ = hft_engine.simd_dot_product(a, b)
        elapsed_bench = time.time() - t0_bench
        print(f"  [AVX2 Benchmark] Completed 1M iterations in {elapsed_bench*1000:.2f} ms")
        
        # Test 2: Bulk AVX2 Processing
        features = np.random.randn(N, 6).astype(np.float64)
        weights = np.random.randn(6).astype(np.float64)
        
        print(f"  [AVX2 Benchmark] Running hft_engine.process_bulk_features_avx2 on {N:,} rows...")
        t0_bench = time.time()
        result_alphas = hft_engine.process_bulk_features_avx2(features, weights)
        elapsed_bench = time.time() - t0_bench
        
        print(f"  [AVX2 Benchmark] Completed bulk processing in {elapsed_bench*1000:.2f} ms")
        print(f"  [AVX2 Benchmark] Throughput: {(N / elapsed_bench) / 1e6:.1f} Million rows/sec")
        print(f"  [AVX2 Benchmark] Result Alphas Shape: {result_alphas.shape}")
        sys.exit(0)

        
    t0 = time.time()
    
    # ── Feature Dump Mode — stream directly, skip loading all ticks ──
    if args.dump_features:
        if not HAS_CPP_ENGINE:
            print("[ERROR] C++ engine required for --dump-features.")
            sys.exit(1)
        print(f"\n  Mode: FEATURE DUMP  (streaming — O(horizon) RAM)")
        print(f"  Input:   {args.data}")
        print(f"  Output:  {args.dump_features}")
        print(f"  Horizon: {args.horizon} ticks")
        print(f"  Warmup:  {args.warmup} ticks")
        print(f"\n  Running C++ feature engine...\n")

        dumper = FeatureDumper(
            horizon         = args.horizon,
            warmup_ticks    = args.warmup,
            initial_capital = args.capital,
        )
        dumper.open(args.dump_features)

        t0 = time.time()
        row_count = 0
        skipped   = 0

        with open(args.data, 'r', buffering=8 << 20) as f:
            header = f.readline().strip().split(',')

            # Detect column indices — handle all known Binance/Tardis CSV formats
            h = header  # shorthand
            if 'is_buyer_maker' in h:
                # process_local_zips.py output: timestamp,price,quantity,is_buyer_maker
                time_idx  = h.index('timestamp')
                price_idx = h.index('price')
                qty_idx   = h.index('quantity')
                buyer_idx = h.index('is_buyer_maker')
            elif 'isBuyerMaker' in h:
                # Raw Binance aggTrade format
                time_idx  = h.index('time') if 'time' in h else h.index('T')
                price_idx = h.index('price') if 'price' in h else h.index('p')
                qty_idx   = h.index('qty')   if 'qty'   in h else h.index('q')
                buyer_idx = h.index('isBuyerMaker')
            elif 'side' in h and 'amount' in h:
                # Tardis trades format
                time_idx  = h.index('timestamp')
                price_idx = h.index('price')
                qty_idx   = h.index('amount')
                buyer_idx = h.index('side')
            else:
                # Fallback: positional
                time_idx  = 0; price_idx = 1; qty_idx = 2; buyer_idx = -1

            def parse_bool(s: str) -> bool:
                """Handle True/False strings AND 0/1 integers, plus Tardis sides."""
                s = s.strip().lower()
                if s in ('true', '1', 'sell', 'ask'):  return True
                if s in ('false', '0', 'buy', 'bid'): return False
                try:
                    return bool(int(s))
                except ValueError:
                    return False

            for line in f:
                if args.max_rows and row_count >= args.max_rows:
                    break
                parts = line.strip().split(',')
                need  = max(time_idx, price_idx, qty_idx,
                            buyer_idx if buyer_idx >= 0 else 0) + 1
                if len(parts) < need:
                    skipped += 1
                    continue
                try:
                    price = float(parts[price_idx])
                    qty   = float(parts[qty_idx])
                    ts    = int(float(parts[time_idx]))
                    is_bm = parse_bool(parts[buyer_idx]) if buyer_idx >= 0 else False
                except (ValueError, IndexError):
                    skipped += 1
                    continue

                dumper.on_tick(Tick(
                    timestamp_ms=ts, price=price,
                    quantity=qty, is_buyer_maker=is_bm,
                ))
                row_count += 1

                if row_count % 1_000_000 == 0:
                    elapsed = time.time() - t0
                    rate    = row_count / elapsed
                    pct     = row_count / 30_000_000 * 100
                    print(f"    [{pct:5.1f}%est]  {row_count:>10,} rows  "
                          f"Rate: {rate:,.0f}/s  "
                          f"Written: {dumper._rows_written:,}  "
                          f"Elapsed: {elapsed:.0f}s")

        dumper.flush()
        n_written = dumper.close()
        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Rows processed: {row_count:,}   Written: {n_written:,}")
        if skipped:
            print(f"  Skipped (bad rows): {skipped:,}")
        print(f"\n  Next step:")
        print(f"    .venv\\Scripts\\python.exe python\\train_model.py --data {args.dump_features}")
        return

    # ── Normal backtest — load all ticks then run ─────────────
    if not args.book:
        ticks = load_binance_csv(args.data, args.max_rows)
        load_time = time.time() - t0
        print(f"  Loaded {len(ticks):,} ticks in {load_time:.2f}s")
        
        if len(ticks) == 0:
            print("[ERROR] No data loaded. Check file path and format.")
            sys.exit(1)
    else:
        ticks = []
    
    # ── Feature Dump Mode (fully streaming — O(horizon) RAM) ─
    if args.dump_features:
        if not HAS_CPP_ENGINE:
            print("[ERROR] C++ engine required for --dump-features.")
            sys.exit(1)
        print(f"\n  Mode: FEATURE DUMP  (streaming — no full-dataset RAM load)")
        print(f"  Input:   {args.data}")
        print(f"  Output:  {args.dump_features}")
        print(f"  Horizon: {args.horizon} ticks")
        print(f"  Warmup:  {args.warmup} ticks")
        print(f"\n  Running C++ feature engine...\n")

        dumper = FeatureDumper(
            horizon        = args.horizon,
            warmup_ticks   = args.warmup,
            initial_capital= args.capital,
        )
        dumper.open(args.dump_features)

        t0 = time.time()
        row_count = 0
        skipped   = 0

        # Stream the CSV directly — never build a list of Tick objects
        with open(args.data, 'r', buffering=8 << 20) as f:
            header = f.readline().strip().split(',')

            # Detect column indices
            if 'isBuyerMaker' in header:
                price_idx = header.index('price')
                qty_idx   = header.index('qty')
                time_idx  = header.index('time')
                buyer_idx = header.index('isBuyerMaker')
                side_idx  = -1
            elif 'is_buyer_maker' in header:
                price_idx = header.index('price')
                qty_idx   = header.index('quantity') if 'quantity' in header else header.index('qty')
                time_idx  = header.index('timestamp') if 'timestamp' in header else header.index('time')
                buyer_idx = header.index('is_buyer_maker')
                side_idx  = -1
            else:
                price_idx = 1; qty_idx = 2; time_idx = 0
                buyer_idx = -1; side_idx = -1

            for line in f:
                parts = line.strip().split(',')
                if len(parts) < max(price_idx, qty_idx, time_idx) + 1:
                    skipped += 1
                    continue
                try:
                    price     = float(parts[price_idx])
                    qty       = float(parts[qty_idx])
                    ts        = int(float(parts[time_idx]))
                    is_bm     = bool(int(parts[buyer_idx])) if buyer_idx >= 0 else False
                except (ValueError, IndexError):
                    skipped += 1
                    continue

                tick = Tick(
                    timestamp_ms   = ts,
                    price          = price,
                    quantity       = qty,
                    is_buyer_maker = is_bm,
                )
                dumper.on_tick(tick)
                row_count += 1

                if row_count % 1_000_000 == 0:
                    elapsed = time.time() - t0
                    rate    = row_count / elapsed
                    eta     = 0  # unknown without total count
                    pct_done = row_count / 30_000_000 * 100  # estimate 30M
                    print(f"    [{pct_done:5.1f}%est]  {row_count:>10,} rows  "
                          f"Rate: {rate:,.0f}/s  "
                          f"Written: {dumper._rows_written:,}  "
                          f"Elapsed: {elapsed:.0f}s")

        dumper.flush()
        n_written = dumper.close()
        elapsed = time.time() - t0

        print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Rows processed: {row_count:,}")
        print(f"  Rows written:   {n_written:,}")
        if skipped:
            print(f"  Rows skipped:   {skipped:,}")
        print(f"\n  Next step:")
        print(f"    .venv\\Scripts\\python.exe python\\train_model.py --data {args.dump_features}")
        return

    # ── Run Backtest ──
    print(f"\n  Running backtest...")
    t0 = time.time()
    
    if HAS_CPP_ENGINE:
        print("  [C++ engine active — full 6-signal pipeline]")
        engine: 'CppBacktestEngine | PythonBacktestEngine' = CppBacktestEngine(
            initial_capital=args.capital,
            alpha_threshold=args.threshold,
            inventory_skew_factor=args.skew,
            queue_delay_prob=args.delay,
        )
    else:
        print("  [Python fallback engine active]")
        engine = PythonBacktestEngine(
            initial_capital=args.capital,
            alpha_threshold=args.threshold
        )
    
    if args.book:
        print(f"\n  [L2 MODE] Initializing Tardis streams...")
        print(f"  Trades: {args.data}")
        print(f"  Book:   {args.book}")
        trades_gen = read_tardis_trades(args.data)
        books_gen = read_tardis_book(args.book)
        engine.run_l2_replay(trades_gen, books_gen, start_time_ns=args.start_time)
    else:
        # Progress tracking
        total = len(ticks)
        report_interval = max(total // 20, 1)
        
        for i, tick in enumerate(ticks):
            engine.on_tick(tick)
            if (i + 1) % report_interval == 0:
                pct = (i + 1) / total * 100
                eq = engine.equity_history[-1] if engine.equity_history else args.capital
                trades = engine.tick_count if isinstance(engine, CppBacktestEngine) \
                         else engine.trade_count
                print(f"    [{pct:5.1f}%] Tick {i+1:>10,} / {total:,}  "
                      f"Equity: ${eq:,.2f}  Trades: {trades}")
    
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
    try:
        # plot_results(result, output_dir)
        pass
    except Exception as e:
        print(f"[Warning] Could not generate plots: {e}")
    
    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    # ── Sharpe Bootstrap CI ──
    try:
        import numpy as np
        eq_arr = np.array(engine.equity_history) if engine.equity_history else np.array([args.capital])
        rets = np.diff(eq_arr) / eq_arr[:-1] if len(eq_arr) > 1 else np.array([])
        
        # 100 fixed bars for the session
        n_bars = 100
        if len(rets) > n_bars:
            bar_len = len(rets) // n_bars
            bar_rets = np.array([np.sum(rets[i*bar_len:(i+1)*bar_len]) for i in range(n_bars)])
            
            # Align bootstrap center with C++ point estimate (tick-level vs bar-level autocorrelation difference)
            bar_mean, bar_std = np.mean(bar_rets), np.std(bar_rets)
            bar_point = (bar_mean / bar_std) * (2445.0 / np.sqrt(bar_len)) if bar_std > 0 else 0.0
            correction = result.sharpe_ratio - bar_point
            
            n_bootstraps = 1000
            boot_sharpes = []
            
            # Dump bar_rets for statistical analysis
            np.savetxt("results/bar_rets_dump.csv", bar_rets, delimiter=",")
            
            for _ in range(n_bootstraps):
                boot_bar = np.random.choice(bar_rets, size=n_bars, replace=True)
                mean_bar = np.mean(boot_bar)
                std_bar = np.std(boot_bar)
                if std_bar > 0:
                    # Match C++ point estimate logic: 2445.0
                    sharpe_bar = (mean_bar / std_bar) * (2445.0 / np.sqrt(bar_len))
                    boot_sharpes.append(sharpe_bar)
            
            if boot_sharpes:
                ci_lower = np.percentile(boot_sharpes, 2.5) + correction
                ci_upper = np.percentile(boot_sharpes, 97.5) + correction
                sharpe_str = f"{result.sharpe_ratio:>12.3f}  [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]"
            else:
                sharpe_str = f"{result.sharpe_ratio:>12.3f}"
        else:
            sharpe_str = f"{result.sharpe_ratio:>12.3f}"
    except Exception as e:
        sharpe_str = f"{result.sharpe_ratio:>12.3f} (CI error: {e})"
        
    print(f"  Initial Capital:  ${result.initial_capital:>12,.2f}")
    print(f"  Final Equity:     ${result.final_equity:>12,.2f}")
    unrealized = (result.final_equity - result.initial_capital) - result.total_pnl
    print(f"  Realized PnL:     ${result.total_pnl:>12,.2f}")
    print(f"  Unrealized PnL:   ${unrealized:>12,.2f}")
    print(f"  Total PnL:        ${(result.final_equity - result.initial_capital):>12,.2f}")
    print(f"  Return:           {result.return_pct:>12.2f}%")
    print(f"  Sharpe Ratio:     {sharpe_str}")
    print(f"  Max Drawdown:     {result.max_drawdown * 100:>12.2f}%")
    print(f"  Total Trades:     {result.total_trades:>12,}")
    print(f"  Win Rate:         {result.win_rate * 100:>12.1f}%")
    print(f"{'='*60}\n")
    
    print(f"  Output saved to: {output_dir}/")


if __name__ == '__main__':
    main()
