import os
import sys
import numpy as np
import pandas as pd
import time
from datetime import datetime

# Ensure we can load the engine
sys.path.insert(0, os.path.dirname(__file__))
from engine_loader import load_engine

hft_engine = load_engine()

def generate_synthetic_data(n):
    """Generate a synthetic order book/trades dataframe using GBM."""
    np.random.seed(42)
    prices = np.exp(np.cumsum(np.random.normal(0, 0.0001, n))) * 50000
    spreads = np.random.uniform(0.5, 2.0, n)
    
    df = pd.DataFrame({
        "timestamp_ns": np.arange(n) * 100000000,
        "best_bid": prices - spreads,
        "best_ask": prices + spreads,
        "bid_qty": np.random.uniform(0.1, 2.0, n),
        "ask_qty": np.random.uniform(0.1, 2.0, n),
        "trade_price": prices + np.random.normal(0, spreads/2, n),
        "trade_qty": np.random.uniform(0.01, 0.5, n),
        "is_buyer_maker": np.random.randint(0, 2, n).astype(bool)
    })
    return df

def run_backtest(data_path=None):
    print("=====================================================")
    print(" HFT Engine - C++ Deterministic Backtester")
    print("=====================================================")
    
    # Load data
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"[INFO] Loaded {len(df)} rows from {data_path}")
    else:
        print("[WARNING] No data path provided or file missing. Generating synthetic GBM data...")
        df = generate_synthetic_data(100000)
        
    config = hft_engine.StrategyConfig()
    config.initial_capital = 1000000.0
    config.min_warmup_ticks = 1000
    
    engine = hft_engine.StrategyEngine(config)
    engine.set_mode(hft_engine.EngineMode.BACKTEST)
    
    optimal_weights = [0.189, 0.006, -0.242, -0.238, 0.101, 0.200]
    engine.set_weights(optimal_weights)
    
    book = hft_engine.BookSnapshot()
    
    start_time = time.time()
    
    # Pre-extract arrays for speed
    best_bid_arr = df["best_bid"].values
    best_ask_arr = df["best_ask"].values
    bid_qty_arr = df["bid_qty"].values
    ask_qty_arr = df["ask_qty"].values
    trade_price_arr = df["trade_price"].values
    trade_qty_arr = df["trade_qty"].values
    is_buyer_maker_arr = df["is_buyer_maker"].values
    
    n_ticks = len(df)
    
    print(f"[INFO] Simulating {n_ticks} ticks...")
    
    for i in range(n_ticks):
        book.best_bid_price = int(best_bid_arr[i] * 1e8)
        book.best_ask_price = int(best_ask_arr[i] * 1e8)
        book.best_bid_qty = int(bid_qty_arr[i] * 1e8)
        book.best_ask_qty = int(ask_qty_arr[i] * 1e8)
        book.bid_count = 1
        book.ask_count = 1
        
        trade = hft_engine.Trade()
        trade.price = int(trade_price_arr[i] * 1e8)
        trade.quantity = int(trade_qty_arr[i] * 1e8)
        trade.side = hft_engine.Side.ASK if is_buyer_maker_arr[i] else hft_engine.Side.BID
        
        engine.on_trade(trade, book)
        
    elapsed = time.time() - start_time
    
    metrics = engine.metrics()
    
    print("\n=====================================================")
    print(" BACKTEST RESULTS")
    print("=====================================================")
    print(f"Ticks Processed : {n_ticks:,} ticks")
    print(f"Elapsed Time    : {elapsed:.2f} seconds")
    print(f"Processing Speed: {n_ticks / elapsed:,.0f} ticks/sec")
    print("-----------------------------------------------------")
    print(f"Initial Capital : ${config.initial_capital:,.2f}")
    print(f"Final Equity    : ${engine.equity():,.2f}")
    print(f"Total PnL       : ${metrics.total_pnl:,.2f}")
    print(f"Total Trades    : {metrics.total_trades}")
    print(f"Win Rate        : {metrics.win_rate * 100:.2f}%")
    print(f"Max Drawdown    : {metrics.max_drawdown * 100:.2f}%")
    print(f"Sharpe Ratio    : {metrics.sharpe_ratio:.2f}")
    print("=====================================================")
    
    # Save journal
    journal = engine.trade_journal()
    if journal:
        log_file = f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        import csv
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Side", "EntryPrice", "ExitPrice", "Qty", "PnL", "Slippage"])
            for rec in journal:
                side_str = "BUY" if rec.side == hft_engine.Side.BID else "SELL"
                writer.writerow([
                    rec.timestamp_ns, side_str, rec.entry_price/1e8, rec.exit_price/1e8,
                    abs(rec.quantity)/1e8, rec.pnl, rec.slippage
                ])
        print(f"[INFO] Saved trade journal to {log_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to historical tick CSV data")
    args = parser.parse_args()
    
    run_backtest(args.data)
