import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import asyncio
from datetime import datetime, timezone
import time
import csv
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import websockets
import collections
# from statsmodels.tsa.stattools import adfuller
import joblib
import numpy as np
import glob
import zipfile

# ─── Load Engine ───────────────────────────────────────────────
from engine_loader import load_engine
hft_engine = load_engine()
from binance_order_gateway import BinanceOrderGateway
print("DEBUG: Loaded C++ engine successfully")
# ─── Global State ──────────────────────────────────────────────
# ─── Process Lock & Global State ───────────────────────────────
import uuid
import subprocess

# LOCK_FILE = "live_paper_trade.lock"
# if os.path.exists(LOCK_FILE):
#     with open(LOCK_FILE, "r") as f:
#         try:
#             pid = int(f.read().strip())
#             # Check if PID exists using tasklist
#             output = subprocess.check_output(f'tasklist /FI "PID eq {pid}"', shell=True).decode()
#             if str(pid) in output:
#                 print(f"[FATAL] Another instance is running (PID {pid}). Aborting.")
#                 sys.exit(1)
#         except ValueError:
#             pass

# with open(LOCK_FILE, "w") as f:
#     f.write(str(os.getpid()))

print("DEBUG: After process lock")

class PaperState:
    run_id = str(uuid.uuid4())[:8]
    is_trading = True
    trade_log_file = f"paper_trades_{run_id}.csv"
    journal_idx = 0   # tracks how many records we've already written to CSV

with open(PaperState.trade_log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "RunID", "TimestampNs", "Side", "EntryPrice", "ExitPrice", "Qty", "PnL", "Slippage",
        "CombinedAlpha", "RealizedVol", "VPIN", "OFI", "OBI", "SpreadBps", "CVD", "Hawkes", "Regime"
    ])

def compress_old_logs(current_run_id):
    old_csvs = [f for f in glob.glob("paper_trades_*.csv") if current_run_id not in f]
    for csv_file in old_csvs:
        try:
            zip_name = csv_file.replace('.csv', '.zip')
            with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(csv_file)
            os.remove(csv_file)
            print(f"[INFO] Compressed old log {csv_file} -> {zip_name}")
        except Exception as e:
            print(f"[ERROR] Failed to compress {csv_file}: {e}")

compress_old_logs(PaperState.run_id)

# Global variable for sentiment model (legacy stub)
sentiment_model = None
# Global sentiment score [-1, +1] — updated by sentiment_loop every 5 min
sentiment_score: float = 0.0
# Global RL position multiplier [0.5, 1.5] — updated by ml_bridge_loop
rl_position_mult: float = 1.0
# Base order size from config (captured at boot; RL scales around this)
_base_order_size_btc: float = 1.0

from contextlib import asynccontextmanager

# ─── Latency Tracking (populated by websocket loop & execution loop) ───────
gateway_latency_ns: float = 0.0        # nanoseconds: last book-update engine call
execution_latency_ns: float = 0.0      # nanoseconds: last order-submit latency

async def python_binance_ws():
    """Pure-Python WebSocket fallback with sequence-gap detection and exponential backoff."""
    global gateway_latency_ns

    ws_base = os.environ.get('BINANCE_WS_BASE_URL', 'wss://fstream.binance.com')
    url = f"{ws_base}/stream?streams=btcusdt@trade/btcusdt@depth20@100ms"

    # Sequence gap tracking (Binance Futures depth stream)
    last_update_id: int = 0
    backoff: float = 1.0

    while True:
        try:
            print(f"[PythonWs] Connecting to {url}...")
            async with websockets.connect(url, ping_interval=30, ping_timeout=10, max_size=None) as ws:
                print("[PythonWs] Connected and subscribed.")
                backoff = 1.0  # reset on successful connect

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        if "data" not in data:
                            continue

                        d = data["data"]
                        stream = data.get("stream", "")

                        if "@trade" in stream:
                            trade = hft_engine.Trade()
                            trade.timestamp_ns = int(d["T"]) * 1_000_000
                            trade.price = int(float(d["p"]) * 1e8)
                            trade.qty   = int(float(d["q"]) * 1e8)  # must be .qty not .quantity
                            trade.side  = hft_engine.Side.ASK if d["m"] else hft_engine.Side.BID
                            engine.on_trade(trade, latest_book)

                        elif "@depth20" in stream:
                            u  = d.get("u", 0)
                            pu = d.get("pu", 0)

                            if last_update_id != 0 and pu != last_update_id:
                                print(
                                    f"[PythonWs] Sequence gap: expected pu={last_update_id} "
                                    f"got pu={pu}. Reconnecting..."
                                )
                                last_update_id = 0
                                break

                            last_update_id = u
                            latest_book.timestamp_ns = int(time.time() * 1e9)
                            bids = d.get("b", [])
                            asks = d.get("a", [])

                            if bids:
                                latest_book.best_bid_price = int(float(bids[0][0]) * 1e8)
                                latest_book.best_bid_qty   = int(float(bids[0][1]) * 1e8)
                                latest_book.bid_count      = len(bids)
                            if asks:
                                latest_book.best_ask_price = int(float(asks[0][0]) * 1e8)
                                latest_book.best_ask_qty   = int(float(asks[0][1]) * 1e8)
                                latest_book.ask_count      = len(asks)

                            if latest_book.is_valid():
                                # Weighted OBI from full 20-level ladder
                                # Weight = 1/(level+1) so top of book has most influence
                                bid_wvol = sum(
                                    float(b[1]) / (i + 1)
                                    for i, b in enumerate(bids[:20]) if float(b[1]) > 0
                                )
                                ask_wvol = sum(
                                    float(a[1]) / (i + 1)
                                    for i, a in enumerate(asks[:20]) if float(a[1]) > 0
                                )
                                ladder_obi = (bid_wvol - ask_wvol) / (bid_wvol + ask_wvol + 1e-8)

                                _t0 = time.perf_counter_ns()
                                engine.on_book_update(latest_book)
                                gateway_latency_ns = time.perf_counter_ns() - _t0

                                # Override single-level OBI with full-ladder OBI
                                engine._last_features.obi = ladder_obi

                    except Exception as loop_err:
                        print(f"[PythonWs] Message loop error: {loop_err}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            import math
            jitter = 0.5 * backoff * (1 + (time.time() % 1))  # ~0-100% jitter
            wait = backoff + jitter
            print(f"[PythonWs] Disconnected: {e}. Reconnecting in {wait:.1f}s...")
            await asyncio.sleep(wait)
            backoff = min(backoff * 2, 60.0)  # cap at 60 seconds

cpp_gateway = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cpp_gateway

    symbol = os.environ.get("BINANCE_SYMBOL", "BTCUSDT").upper()
    is_live = os.environ.get("BINANCE_MODE", "PAPER").upper() == "LIVE"

    print(f"[INFO] Reconciling {symbol} position directly with Binance API...")
    gateway = BinanceOrderGateway()
    await gateway.connect()
    
    if not gateway.api_key or not gateway.api_secret:
        if is_live:
            print("[FATAL] Binance API keys missing in LIVE mode. Aborting boot.")
            os._exit(1)
        else:
            print("[WARNING] Binance API keys missing in PAPER mode. Booting with flat position.")
            
    # Calculate UTC midnight timestamp for daily realized PnL
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    midnight_ms = int(midnight.timestamp() * 1000)

    try:
        pos_risk = await gateway.get_position_risk(symbol)
        realized_pnl = await gateway.get_realized_pnl(symbol, midnight_ms)
        
        if pos_risk is not None:
            binance_pos = float(pos_risk.get("positionAmt", 0.0))
            binance_pnl = float(pos_risk.get("unRealizedProfit", 0.0))
            binance_entry = float(pos_risk.get("entryPrice", 0.0))
            
            fixed_pos = int(binance_pos * 1e8)
            
            engine.set_position(fixed_pos)
            engine.set_avg_entry_price(binance_entry)
            engine.set_realized_pnl(realized_pnl)
            
            current_ts_ms = int(time.time() * 1000)
            engine.update_kill_switch_state(current_ts_ms)
            
            print(f"[INFO] Position reconciled. True Pos: {binance_pos} {symbol}, Entry: ${binance_entry}, UnrlPnL: ${binance_pnl}, DailyRealPnL: ${realized_pnl}")
        else:
            print("[FATAL] Could not retrieve position risk from Binance. Aborting boot.")
            os._exit(1)
    except Exception as e:
        print(f"[FATAL] Exception during position reconciliation: {e}")
        os._exit(1)
    finally:
        await gateway.close()

    use_cpp = os.environ.get("USE_CPP_GATEWAY", "0") == "1"
    cpp_gateway = None

    if use_cpp and hasattr(hft_engine, "BinanceWs"):
        print("[INFO] Initializing C++ Exchange Gateway (IXWebSocket)...")
        cpp_gateway = hft_engine.BinanceWs("btcusdt")
        cpp_gateway.start_live_feed(engine)
    else:
        print("[INFO] Using Python WebSocket gateway.")
        asyncio.create_task(python_binance_ws())

    asyncio.create_task(log_flusher_loop())
    asyncio.create_task(ml_bridge_loop())
    asyncio.create_task(execution_loop())
    asyncio.create_task(daily_reset_loop())
    asyncio.create_task(funding_rate_loop())
    asyncio.create_task(user_data_loop(gateway))
    asyncio.create_task(sentiment_loop())

    yield

    if cpp_gateway:
        cpp_gateway.stop()


app = FastAPI(title="HFT Quant Cockpit API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# load state
ACCOUNT_FILE = "paper_account.json"
loaded_capital = 10000000.0
loaded_position = 0
loaded_realized_pnl = 0.0

db_conn = None
db_url = os.environ.get("DATABASE_URL")
if db_url:
    try:
        import psycopg2
        print(f"[INFO] Connecting to PostgreSQL database...")
        db_conn = psycopg2.connect(db_url)
        db_conn.autocommit = True
        with db_conn.cursor() as cur:
            # Create tables if they don't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS account_state (
                    id SERIAL PRIMARY KEY,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    initial_capital FLOAT,
                    position BIGINT,
                    realized_pnl FLOAT
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(50),
                    timestamp_ns BIGINT,
                    side VARCHAR(10),
                    entry_price FLOAT,
                    exit_price FLOAT,
                    qty FLOAT,
                    pnl FLOAT,
                    slippage FLOAT,
                    alpha FLOAT,
                    volatility FLOAT,
                    vpin FLOAT,
                    ofi FLOAT,
                    obi FLOAT,
                    spread_bps FLOAT,
                    cvd FLOAT,
                    hawkes FLOAT,
                    regime INT
                );
            """)
            # Load state from PostgreSQL
            cur.execute("SELECT initial_capital, position, realized_pnl FROM account_state ORDER BY updated_at DESC LIMIT 1;")
            row = cur.fetchone()
            if row:
                loaded_capital, loaded_position, loaded_realized_pnl = row
                # Sanity-clamp: reject obviously corrupt PnL (> initial capital in magnitude)
                if abs(loaded_realized_pnl) > loaded_capital * 0.1:
                    print(f"[WARNING] DB PnL {loaded_realized_pnl:.2f} looks corrupted (>10% of capital). Resetting to 0.")
                    loaded_realized_pnl = 0.0
                print(f"[INFO] Loaded DB state: Capital={loaded_capital}, Pos={loaded_position/1e8}, PnL={loaded_realized_pnl}")
            else:
                print(f"[INFO] No existing DB state found. Using defaults.")
                cur.execute("INSERT INTO account_state (initial_capital, position, realized_pnl) VALUES (%s, %s, %s)",
                            (loaded_capital, loaded_position, loaded_realized_pnl))
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        db_conn = None

if not db_conn and os.path.exists(ACCOUNT_FILE):
    try:
        with open(ACCOUNT_FILE, "r") as f:
            state = json.load(f)
            loaded_capital = state.get("initial_capital", 10000000.0)
            loaded_position = state.get("position", 0)
            loaded_realized_pnl = state.get("realized_pnl", 0.0)
            print(f"[INFO] Loaded local state: Capital={loaded_capital}, Pos={loaded_position/1e8}, PnL={loaded_realized_pnl}")
    except Exception as e:
        print(f"[WARNING] Could not load local state: {e}")

print("DEBUG: Before StrategyConfig")
# Configure strategy
config = hft_engine.StrategyConfig()
config.initial_capital = loaded_capital
config.min_warmup_ticks = 1000  # Institutional: full warmup so all feature buffers are populated
config.max_position_pct = 0.10  # Reduced from 15% to 10% to minimize inventory skew risk in deployment

# ── Institutional Thresholds (Pre-Deployment Optimized) ──────────
# Lowered thresholds so that trades trigger frequently enough for live observation
config.alpha_entry_threshold = 0.03     # Lowered from 0.08 to increase trade frequency
config.alpha_short_multiplier = 1.1     # Lowered from 1.3
config.spread_alpha_multiplier = 0.05   # Lowered from 0.18 to allow trading in normal spreads
config.min_take_profit_bps    = 5.0     # Keep at 5 bps

# ── Futures Fee Model (USDM Perp, Binance VIP0) ─────────────────
# Maker: -0.5 bps (rebate), Taker: 1.5 bps. Strategy targets maker fills.
config.maker_fee_pct = -0.00005  # -0.5 bps maker rebate
taker_fee_global     =  0.00015  # 1.5 bps taker (used in mock engine)

print("DEBUG: Before StrategyEngine")
# The C++ Strategy Engine
engine = hft_engine.StrategyEngine(config)
engine.set_position(loaded_position)
engine.set_realized_pnl(loaded_realized_pnl)
# BACKTEST mode enables full trade journal (rich CSV output for audit/analysis)
engine.set_mode(hft_engine.EngineMode.BACKTEST)

print("DEBUG: After StrategyEngine")
# Load freshly trained Ridge directional weights from signal_weights.bin
# Fall back to hand-validated weights if the file isn't present
_weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'signal_weights.bin'))
if os.path.exists(_weights_path):
    try:
        engine.load_model(_weights_path)
        print(f"[INFO] Loaded ML Model from {_weights_path}")
    except Exception as _e:
        print(f"[WARNING] Could not load signal_weights.bin: {_e}")
        optimal_weights = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        engine.set_weights(optimal_weights)
        print(f"[INFO] Fallback weights (11 features): {optimal_weights}")
else:
    optimal_weights = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    engine.set_weights(optimal_weights)
    print(f"[INFO] No signal_weights.bin found. Using validated weights: {optimal_weights}")

print("DEBUG: Before BookSnapshot")
# We need the latest book to pass to on_trade
latest_book = hft_engine.BookSnapshot()
# BUG FIX: was -1, which passes the C++ is_valid() check (INVALID_PRICE = INT64_MIN ≠ -1)
# but gives wrong mid-price calculations until the first real depth update.
# 0 is correctly rejected by is_valid() (requires best_bid_price > 0).
latest_book.best_bid_price = 0
latest_book.best_ask_price = 0

# Global state
price_history   = collections.deque(maxlen=5000)   # Extended: 5000 for reliable ADF stationarity test
time_sampled_prices = collections.deque(maxlen=5000) # Time-sampled prices for volatility
funding_rate    = 0.0   # BTC-PERP 8h funding rate (used as carry signal)
mark_price      = 0.0   # Futures mark price

# ─── Shared order state (execution_loop ↔ user_data_loop) ───────────────────
# Both coroutines run concurrently in the same asyncio event loop.
# user_data_loop clears order_id when a fill or cancel arrives from Binance;
# execution_loop reads it before deciding whether to cancel a stale order.
import dataclasses

@dataclasses.dataclass
class OrderState:
    """Thread-safe (within asyncio) shared state for the currently live order."""
    order_id: int = -1          # −1 means no live order
    side: str = ""              # "BUY" or "SELL"
    price: float = 0.0
    qty: float = 0.0
    submitted_at: float = 0.0   # time.monotonic() at submit
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)

order_state = OrderState()

# ─── Binance Futures WebSocket Consumer ────────────────────────
# ─── Data Logger Loop (C++ Gateway handles network) ────────────
async def log_flusher_loop():
    print(f"[INFO] Background log flusher started.")
    
    while True:
        try:
            # Flush new journal records to CSV/DB
            journal = engine.trade_journal()
            new_records = journal[PaperState.journal_idx:]
            if new_records:
                # 1. Update Account State
                state_dict = {
                    "initial_capital": config.initial_capital,
                    "position": engine.position(),
                    "realized_pnl": engine.realized_pnl()
                }
                
                # Save to PostgreSQL
                if db_conn:
                    try:
                        with db_conn.cursor() as cur:
                            cur.execute("INSERT INTO account_state (initial_capital, position, realized_pnl) VALUES (%s, %s, %s)",
                                        (state_dict["initial_capital"], state_dict["position"], state_dict["realized_pnl"]))
                    except Exception as e:
                        print(f"[ERROR] Failed to save state to DB: {e}")
                
                # Save to local disk fallback
                try:
                    with open(ACCOUNT_FILE, "w") as sf:
                        json.dump(state_dict, sf)
                except Exception as e:
                    print(f"[WARNING] Failed to save local account state: {e}")

                # 2. Insert Trades
                for rec in new_records:
                    side_str = "BUY" if rec.side == hft_engine.Side.BID else "SELL"
                    fv = engine.last_features()
                    
                    # Save to DB
                    rec_pnl      = getattr(rec, 'pnl',      getattr(rec, 'total_pnl', 0.0))
                    rec_slippage = getattr(rec, 'slippage',  0.0)
                    if db_conn:
                        try:
                            with db_conn.cursor() as cur:
                                cur.execute("""
                                    INSERT INTO trades (
                                        run_id, timestamp_ns, side, entry_price, exit_price, qty, pnl, slippage,
                                        alpha, volatility, vpin, ofi, obi, spread_bps, cvd, hawkes, regime
                                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """, (
                                    PaperState.run_id, rec.timestamp_ns, side_str,
                                    rec.entry_price/1e8, rec.exit_price/1e8, abs(rec.quantity)/1e8,
                                    rec_pnl, rec_slippage, fv.combined_alpha, fv.realized_vol, fv.vpin,
                                    fv.ofi, fv.obi, fv.spread_bps, fv.cvd, fv.hawkes_intensity, int(fv.regime)
                                ))
                        except Exception as e:
                            print(f"[ERROR] Failed to insert trade to DB: {e}")

                    # Save to CSV
                    with open(PaperState.trade_log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            PaperState.run_id, rec.timestamp_ns, side_str,
                            rec.entry_price/1e8, rec.exit_price/1e8,
                            abs(rec.quantity)/1e8, rec_pnl, rec_slippage,
                            fv.combined_alpha, fv.realized_vol, fv.vpin, fv.ofi, fv.obi,
                            fv.spread_bps, fv.cvd, fv.hawkes_intensity, int(fv.regime)
                        ])
                PaperState.journal_idx = len(journal)
                
            await asyncio.sleep(1.0) # Flush at 1Hz
        except Exception as e:
            print(f"[ERROR] Log Flusher Exception: {e}")
            await asyncio.sleep(1)

# ─── API Endpoints ─────────────────────────────────────────────
class TradeControl(BaseModel):
    is_trading: bool

from fastapi.responses import FileResponse
from fastapi.background import BackgroundTasks
import shutil

@app.api_route("/", methods=["GET", "HEAD"])
async def get_dashboard():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "dashboard", "standalone.html"))

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "ok", "timestamp": time.time(), "is_trading": PaperState.is_trading}

@app.get("/api/export")
async def export_data(background_tasks: BackgroundTasks):
    """Zip all trade CSVs and account state and return for download."""
    zip_filename = f"hft_export_{PaperState.run_id}.zip"
    
    # Create a temporary directory to gather files
    temp_dir = f"temp_export_{PaperState.run_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy current logs and state
    if os.path.exists(ACCOUNT_FILE):
        shutil.copy(ACCOUNT_FILE, os.path.join(temp_dir, ACCOUNT_FILE))
        
    for f in glob.glob("paper_trades_*.csv"):
        shutil.copy(f, os.path.join(temp_dir, f))
        
    for f in glob.glob("paper_trades_*.zip"):
        shutil.copy(f, os.path.join(temp_dir, f))
        
    # Zip the directory
    shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', temp_dir)
    
    # Cleanup temp directory
    shutil.rmtree(temp_dir)
    
    # Send the zip file, and remove it after sending
    background_tasks.add_task(os.remove, zip_filename)
    return FileResponse(zip_filename, media_type="application/zip", filename=zip_filename)


@app.post("/api/trade_control")
async def trade_control(payload: TradeControl):
    PaperState.is_trading = payload.is_trading
    return {"status": "success", "is_trading": PaperState.is_trading}

@app.get("/api/trade_status")
async def get_trade_status():
    return {"is_trading": PaperState.is_trading}

@app.get("/api/trades")
async def get_recent_trades_api():
    trades = []
    if db_conn:
        try:
            with db_conn.cursor() as cur:
                cur.execute("SELECT timestamp_ns, side, entry_price, exit_price, qty, pnl, alpha, volatility, vpin, regime FROM trades ORDER BY timestamp_ns DESC LIMIT 50")
                for row in cur.fetchall():
                    trades.append({
                        "timestamp": row[0],
                        "side": row[1],
                        "entry_price": row[2],
                        "exit_price": row[3],
                        "qty": row[4],
                        "pnl": row[5],
                        "alpha": row[6],
                        "volatility": row[7],
                        "vpin": row[8],
                        "regime": row[9]
                    })
        except Exception as e:
            print(f"[ERROR] DB fetch trades failed: {e}")
    else:
        # Fallback to in-memory journal
        journal = engine.trade_journal()
        for rec in journal[-50:]:
            trades.append({
                "timestamp": rec.timestamp_ns,
                "side": "BUY" if rec.side == hft_engine.Side.BID else "SELL",
                "entry_price": rec.entry_price / 1e8,
                "exit_price": rec.exit_price / 1e8,
                "qty": abs(rec.quantity) / 1e8,
                "pnl": rec.pnl,
                "alpha": 0.0,
                "volatility": 0.0,
                "vpin": 0.0,
                "regime": 0
            })
        trades.reverse()
    return {"trades": trades}

# ─── Dashboard Telemetry Server ────────────────────────────────
chart_history = collections.deque(maxlen=100)

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[INFO] Dashboard connected.")
    
    # Send history on connect
    history_payload = {
        "type": "history",
        "data": list(chart_history)
    }
    await websocket.send_json(history_payload)
    
    try:
        while True:
            fv = engine.last_features()
            equity = engine.equity()
            pending = engine.pending_order()

            payload = {
                "type": "update",
                "timestamp": time.time(),
                "alpha": fv.combined_alpha,
                "vpin": fv.vpin,
                "book_imbalance": fv.obi,
                "volatility": fv.realized_vol,
                "regime": int(fv.regime),
                "cvd": fv.cvd,
                "hawkes": fv.hawkes_intensity,
                "sentiment": sentiment_score,
                "rl_mult": rl_position_mult,
                "best_bid": latest_book.best_bid_price / 1e8,
                "best_ask": latest_book.best_ask_price / 1e8,
                "equity": equity,
                "inventory": engine.position() / 1e8,
                "cash": 0,
                "funding_rate": funding_rate,
                "mark_price": mark_price,
                "pending_order": {
                    "active": pending.active,
                    "side": "BID" if pending.side == hft_engine.Side.BID else ("ASK" if pending.side == hft_engine.Side.ASK else "NONE"),
                    "price": pending.price / 1e8,
                    "qty": pending.qty / 1e8,
                    "queue_position": getattr(pending, 'queue_position', 0) / 1e8
                },
                # ─── Latency profiling (visible on dashboard) ─────────────
                "latency": {
                    "book_update_us": round(gateway_latency_ns / 1000.0, 2),
                    "order_submit_ms": round(execution_latency_ns / 1_000_000.0, 2)
                },
                # ─── Kill-switch state ──────────────────────────────────────
                # Reflects whether the C++ engine or ML bridge has halted trading.
                "kill_switch_halted": (
                    getattr(engine, '_halted', False) or
                    bool(
                        (lambda: (lambda ts: engine.is_trading_halted(ts) if callable(getattr(engine, 'is_trading_halted', None)) else False)(int(time.time()*1000)))()
                    )
                )
            }
            chart_history.append(payload)
            await websocket.send_json(payload)
            await asyncio.sleep(0.1)  # 10 Hz
    except (websockets.exceptions.ConnectionClosed, WebSocketDisconnect, RuntimeError):
        print("[INFO] Dashboard disconnected.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Telemetry loop exception: {e}")

# ─── User Data Stream Loop (Live Fill Tracking) ────────────────
async def user_data_loop(gateway: BinanceOrderGateway):
    """
    Connects to the Binance User Data Stream and processes execution reports.

    On FILLED / PARTIALLY_FILLED:
      - Syncs the real executed qty/price back to the C++ engine.
      - Clears order_state.order_id so execution_loop knows the order is gone.

    On CANCELED / EXPIRED:
      - Clears order_state.order_id (no fill to record).

    This is a no-op if API keys are not configured (paper mode).
    """
    if not gateway.api_key or not gateway.api_secret:
        print("[INFO] User Data Stream skipped (no API keys — paper mode).")
        return

    listen_key = await gateway.create_listen_key()
    if not listen_key:
        print("[WARNING] Could not obtain User Data Stream listen key. Fill tracking disabled.")
        return

    print("[INFO] User Data Stream listening for ORDER_TRADE_UPDATE events...")

    async for event in gateway.websocket_user_stream():
        try:
            event_type = event.get("e", "")

            if event_type == "ORDER_TRADE_UPDATE":
                o = event.get("o", {})
                symbol      = o.get("s", "")
                status      = o.get("X", "")    # FILLED, PARTIALLY_FILLED, CANCELED, EXPIRED
                side_raw    = o.get("S", "")    # BUY / SELL
                exec_qty    = float(o.get("l", 0.0))   # last-leg executed qty
                exec_px     = float(o.get("L", 0.0))   # last-leg executed price
                cum_qty     = float(o.get("z", 0.0))   # cumulative filled qty (P3-5)
                order_id    = int(o.get("i", -1))
                is_maker    = o.get("m", False)         # True = maker fill (lower fee)

                if status in ("FILLED", "PARTIALLY_FILLED") and exec_qty > 0:
                    fixed_qty = int(exec_qty * 1e8)
                    is_buy    = side_raw == "BUY"
                    side_enum = hft_engine.Side.BID if is_buy else hft_engine.Side.ASK

                    print(
                        f"[FILL] {status} orderId={order_id} {side_raw} "
                        f"{exec_qty} @ {exec_px} (cum={cum_qty}, symbol={symbol}, "
                        f"maker={'Y' if is_maker else 'N'})"
                    )

                    try:
                        engine.simulate_fill(side_enum, int(exec_px * 1e8), fixed_qty, is_maker)
                    except Exception as fill_err:
                        print(f"[WARNING] engine.simulate_fill failed: {fill_err}")

                    # ── Only clear order_state on full FILL; keep it alive on partial fills
                    #    so TTL sweep and stale-cancel logic still operate correctly ──────
                    if status == "FILLED":
                        async with order_state.lock:
                            if order_state.order_id == order_id:
                                order_state.order_id     = -1
                                order_state.submitted_at = 0.0
                                print(f"[FILL] order_state cleared — order {order_id} fully filled (cum={cum_qty})")



                elif status in ("CANCELED", "EXPIRED"):
                    print(f"[FILL] Order {order_id} {status} — no fill.")
                    async with order_state.lock:
                        if order_state.order_id == order_id:
                            order_state.order_id = -1

            elif event_type == "ACCOUNT_UPDATE":
                balances = event.get("a", {}).get("B", [])
                for b in balances:
                    if b.get("a") == "USDT":
                        print(f"[ACCOUNT] USDT Balance: {b.get('wb', '?')} wallet / {b.get('cw', '?')} cross-margin")

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[ERROR] user_data_loop event processing error: {e}")


# ─── Execution Loop ─────────────────────────────────────────────────────────
async def execution_loop():
    """
    Polls the C++ engine for pending limit orders and routes them to Binance.

    Safety gates (in order of precedence):
      1. Kill-switch: engine.is_trading_halted() — hard-blocks all order submission.
      2. Paper mode: no API keys → uses MOCK path (safe for local dev).
      3. Stale-order cancel: if a new engine signal arrives while an old order
         is still live (not yet filled/cancelled by the User Data Stream),
         we cancel the stale order before placing the new one.

    Shared state:
      Uses the module-level `order_state` (OrderState) guarded by asyncio.Lock
      so that user_data_loop can clear order_id on fills without races.
    """
    global execution_latency_ns

    print("[INFO] Starting Execution Loop (Live Trading)...")
    gateway = BinanceOrderGateway()
    await gateway.connect()

    last_executed_ts: int = 0
    SYMBOL           = os.environ.get("BINANCE_SYMBOL", "BTCUSDT").upper()
    STALE_ORDER_TTL_S = int(os.environ.get("STALE_ORDER_TTL_MS", "5000")) / 1000.0

    while True:
        try:
            # ── 1. Kill-switch gate ──────────────────────────────────────────
            current_ts_ms = int(time.time() * 1000)
            try:
                halted = engine.is_trading_halted(current_ts_ms)
            except Exception:
                halted = False

            if halted:
                async with order_state.lock:
                    if order_state.order_id != -1:
                        oid = order_state.order_id
                        print(f"[EXECUTION] Kill switch ACTIVE — cancelling live order {oid}")
                        try:
                            await gateway.cancel_order(SYMBOL, oid)
                        except Exception as ce:
                            print(f"[EXECUTION] Cancel failed: {ce}")
                        order_state.order_id = -1
                await asyncio.sleep(0.1)
                continue
            # ── 2. TTL-based stale-order sweep (ζ STALE_ORDER_TTL_MS) ─────────────
            # Cancel the live order even if no new signal has arrived,
            # if it has been sitting in the book longer than the TTL.
            # This prevents stale limit orders from being hit by adverse flow.
            async with order_state.lock:
                ttl_oid       = order_state.order_id
                ttl_submitted = order_state.submitted_at

            if ttl_oid != -1 and ttl_submitted > 0:
                age_s = time.monotonic() - ttl_submitted
                if age_s > STALE_ORDER_TTL_S:
                    print(
                        f"[EXECUTION] TTL expired ({age_s:.1f}s > {STALE_ORDER_TTL_S:.1f}s) "
                        f"— cancelling stale order {ttl_oid}"
                    )
                    try:
                        await gateway.cancel_order(SYMBOL, ttl_oid)
                    except Exception as ce:
                        print(f"[EXECUTION] TTL cancel failed: {ce}")
                    async with order_state.lock:
                        if order_state.order_id == ttl_oid:
                            order_state.order_id     = -1
                            order_state.submitted_at = 0.0

            # ── 3. Check for new signal ──────────────────────────────────────
            pending = engine.pending_order()

            if pending.active and pending.timestamp_ns > last_executed_ts:
                last_executed_ts = pending.timestamp_ns

                side_str = "BUY" if pending.side == hft_engine.Side.BID else "SELL"
                price    = pending.price / 1e8
                qty      = pending.qty   / 1e8

                # ── 3. Stale-order cancel ────────────────────────────────────
                async with order_state.lock:
                    stale_id = order_state.order_id

                if stale_id != -1:
                    print(f"[EXECUTION] New signal — cancelling stale order {stale_id}")
                    try:
                        await gateway.cancel_order(SYMBOL, stale_id)
                    except Exception as ce:
                        print(f"[EXECUTION] Stale cancel failed (may already be filled): {ce}")
                    async with order_state.lock:
                        if order_state.order_id == stale_id:
                            order_state.order_id = -1

                print(f"[EXECUTION] Firing Limit Order: {side_str} {qty} @ {price}")

                _t0 = time.perf_counter_ns()
                res = await gateway.place_limit_order(SYMBOL, side_str, qty, price)
                execution_latency_ns = time.perf_counter_ns() - _t0

                new_id = res.get("orderId", -1)
                async with order_state.lock:
                    order_state.order_id     = new_id
                    order_state.side         = side_str
                    order_state.price        = price
                    order_state.qty          = qty
                    order_state.submitted_at = time.monotonic()

                print(f"[EXECUTION] Gateway Response: {res}")

        except asyncio.CancelledError:
            # Graceful shutdown — cancel any live order first
            async with order_state.lock:
                shutdown_id = order_state.order_id
            if shutdown_id != -1:
                try:
                    await gateway.cancel_order(SYMBOL, shutdown_id)
                    print(f"[EXECUTION] Shutdown cancel: order {shutdown_id}")
                except Exception:
                    pass
            break
        except Exception as e:
            print(f"[ERROR] Execution loop exception: {e}")

        await asyncio.sleep(0.01)  # 10 ms polling

    await gateway.close()


# ─── ML Bridge Loop ────────────────────────────────────────────
async def ml_bridge_loop():
    global rl_position_mult, _base_order_size_btc
    print("[INFO] Starting ML Bridge Loop (ADF & True HMM)")
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'hmm_regime.pkl'))
    regime_model = None
    try:
        if os.path.exists(model_path):
            regime_model = joblib.load(model_path)
            print("[INFO] Loaded True HMM Regime Model")
    except Exception as e:
        print(f"[WARNING] Could not load regime model: {e}")

    # ── Online RL Policy ──────────────────────────────────────────────────
    try:
        from online_rl import OnlineRLPolicy
        rl_save = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'online_policy.npz'))
        rl_policy = OnlineRLPolicy(save_path=rl_save)
        _base_order_size_btc = engine.config.order_size_btc
        print("[INFO] Online RL Policy loaded.")
    except Exception as e:
        rl_policy = None
        print(f"[WARNING] Online RL not available: {e}")
    # ──────────────────────────────────────────────────────

    loop = asyncio.get_running_loop()
    prev_journal_len = 0

    while True:
        await asyncio.sleep(1.0)

        # Capture time-sampled price for volatility
        if latest_book.best_bid_price > 0:
            mid_price = (latest_book.best_bid_price + latest_book.best_ask_price) / 2.0 / 1e8
            time_sampled_prices.append(mid_price)

        # Compute realized volatility from buffered prices
        realized_vol_raw = 0.01
        if len(time_sampled_prices) >= 30:
            prices_arr = np.array(list(time_sampled_prices)[-30:])
            log_rets   = np.diff(np.log(prices_arr + 1e-10))
            realized_vol_raw = max(float(np.std(log_rets) * np.sqrt(len(log_rets) * 100)), 0.001)

        fv = engine.last_features()

        # ── Online RL: adapt position size from realized PnL feedback ─────────
        if rl_policy is not None:
            try:
                obs = rl_policy.obs_from_features(fv, realized_vol_raw, sentiment_score)
                mult = rl_policy.act(obs)
                rl_position_mult = mult
                engine.config.order_size_btc = _base_order_size_btc * mult

                # Update RL on every newly closed trade
                journal = engine.trade_journal()
                if len(journal) > prev_journal_len:
                    for rec in journal[prev_journal_len:]:
                        rec_pnl = getattr(rec, 'pnl', 0.0)
                        rl_policy.update(rec_pnl)
                    prev_journal_len = len(journal)
            except Exception:
                pass
        # ──────────────────────────────────────────────────────

        if regime_model and 'model' in regime_model:
            if len(time_sampled_prices) < 30:
                continue
            try:
                model = regime_model['model']
                mean  = regime_model['mean']
                std   = regime_model['std']

                X_curr   = np.array([[realized_vol_raw, fv.spread_bps]])
                X_scaled = (X_curr - mean) / (std + 1e-10)
                state_arr = await loop.run_in_executor(None, model.predict, X_scaled)
                state = int(state_arr[0])

                if state == 3:
                    if not getattr(engine, '_halted', False):
                        print(f"[ML BRIDGE] Regime State 3 (CRISIS) detected. Blocking new entries.")
                        try:
                            engine._halted = True
                        except AttributeError:
                            pass
                else:
                    daily_limit = getattr(getattr(engine, 'config', None), 'daily_loss_limit_usd', 500.0)
                    current_loss = engine.config.initial_capital - engine.equity()
                    if getattr(engine, '_halted', False) and current_loss < daily_limit * 0.5:
                        try:
                            engine._halted = False
                        except AttributeError:
                            pass

                print(f"[ML BRIDGE] HMM State {state} | Vol: {realized_vol_raw:.4f} | Alpha: {fv.combined_alpha:.4f} | VPIN: {fv.vpin:.3f} | RLmult: {rl_position_mult:.2f}")
            except asyncio.CancelledError:
                break
            except Exception:
                pass


# ─── Sentiment Loop (VADER + CryptoCompare, no API key) ────────────────
async def sentiment_loop():
    """Poll CryptoCompare crypto news every 5 min. Score with VADER. Update global sentiment_score."""
    global sentiment_score
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        print("[SENTIMENT] VADER sentiment loop started.")
    except ImportError:
        print("[SENTIMENT] vaderSentiment not installed — skipping sentiment loop.")
        return

    CC_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC&limit=10"
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(CC_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        articles = data.get("Data", [])[:10]
                        scores = []
                        for art in articles:
                            text = art.get("title", "") + " " + art.get("body", "")[:200]
                            scores.append(analyzer.polarity_scores(text)["compound"])
                        if scores:
                            sentiment_score = float(np.mean(scores))
                            print(f"[SENTIMENT] Score: {sentiment_score:+.3f} ({len(scores)} headlines)")
        except asyncio.CancelledError:
            break
        except Exception:
            pass
        await asyncio.sleep(300)  # poll every 5 minutes


# ─── Daily Reset Loop ──────────────────────────────────────────
async def daily_reset_loop():
    print("[INFO] Starting Daily Reset Loop (UTC Midnight Rollover)")

    current_day = datetime.now(timezone.utc).day

    while True:
        await asyncio.sleep(60.0)  # Check every minute
        now_utc = datetime.now(timezone.utc)

        if now_utc.day != current_day:
            print(f"[INFO] UTC Midnight Reached. Rolling over from day {current_day} to {now_utc.day}")
            current_day = now_utc.day

            # Reset engine metrics and rebase drawdown limit
            try:
                engine.new_trading_day()
            except AttributeError:
                pass  # If engine doesn't have it (though both now do)

            print("[INFO] Daily reset complete.")


# ─── Funding Rate & Mark Price Loop ─────────────────────────────
async def funding_rate_loop():
    """
    Polls Binance /fapi/v1/premiumIndex every 30 seconds to keep the global
    `funding_rate` and `mark_price` variables current.

    These values are:
      - Sent to the dashboard via the /ws/telemetry payload.
      - Available as a carry signal for the ML bridge regime logic.

    Falls back gracefully when API keys are absent (paper mode) or the
    network is unavailable.
    """
    global funding_rate, mark_price

    SYMBOL  = os.environ.get("BINANCE_SYMBOL", "BTCUSDT").upper()
    WS_BASE = os.environ.get("BINANCE_BASE_URL", "https://fapi.binance.com")
    url     = f"{WS_BASE}/fapi/v1/premiumIndex?symbol={SYMBOL}"
    POLL_S  = 30.0  # Binance updates funding rate info every ~5 s; 30 s is plenty

    print(f"[INFO] Starting Funding Rate Loop for {SYMBOL} (polling every {POLL_S:.0f}s)")

    import aiohttp as _aiohttp
    async with _aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(url, timeout=_aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        mark_price   = float(data.get("markPrice",   mark_price))
                        funding_rate = float(data.get("lastFundingRate", funding_rate))
                    else:
                        print(f"[FUNDING] HTTP {resp.status} from premiumIndex — keeping last values")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[FUNDING] Poll error: {e} — keeping last values")

            await asyncio.sleep(POLL_S)


if __name__ == "__main__":
    print("=====================================================")
    print(" HFT Engine - Live Paper Trading Node")
    print("=====================================================")
    
    try:
        pass
    except Exception:
        pass
        
    import asyncio
    import socket

    async def run_server():
        # Bind to 0.0.0.0 and use PORT environment variable for Railway compatibility
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", 8080))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            sock.close()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, 0))   # OS picks any free port
            port = sock.getsockname()[1]
            print(f"[INFO] Requested port unavailable, using fallback port {port}.")
        sock.listen(128)
        print(f"[INFO] Dashboard available at: http://{host}:{port}/")
        config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve(sockets=[sock])
            
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received. Shutting down gracefully...")

        # EOD Flatten: close any open position at best available price
        # if latest_book.best_bid_price > 0:
        #     engine.flatten(latest_book)

        # Session summary
        import pandas as pd
        if os.path.exists(PaperState.trade_log_file):
            df = pd.read_csv(PaperState.trade_log_file)
            n  = len(df)
            print(f"[INFO] Session recorded {n} trades.")
            if not df.empty and 'Equity' in df.columns:
                start_eq = df['Equity'].iloc[0]
                end_eq   = df['Equity'].iloc[-1]
                net_pnl  = end_eq - start_eq
                print(f"[INFO] Final Equity : ${end_eq:,.2f}")
                print(f"[INFO] Net PnL      : ${net_pnl:+,.2f}")
                if n > 0:
                    gross_fees = (df['Price'] * df['Qty'].div(1e8) * 0.00015).sum()
                    print(f"[INFO] Gross Fees   : ${gross_fees:,.2f}")
        print("[INFO] Live Paper Trading Node stopped.")
    finally:
        pass
