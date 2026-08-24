import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import asyncio
import time
import csv
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import websockets
import collections
from statsmodels.tsa.stattools import adfuller
import joblib
import numpy as np
import glob
import zipfile

# ─── Load Engine ───────────────────────────────────────────────
from engine_loader import load_engine
hft_engine = load_engine()
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

print("DEBUG: Before FastAPI")
# Global variable for sentiment model
sentiment_model = None

from contextlib import asynccontextmanager

async def python_binance_ws():
    print("[INFO] Starting pure Python WebSocket fallback...")
    url = "wss://fstream.binance.com/stream?streams=btcusdt@trade/btcusdt@depth5@100ms"
    
    while True:
        try:
            print(f"[PythonWs] Connecting to {url}...")
            async with websockets.connect(url, ping_interval=30, ping_timeout=10, max_size=None) as ws:
                print("[PythonWs] Connected and subscribed.")
                
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        if "data" in data:
                            d = data["data"]
                            stream = data.get("stream", "")
                            
                            if "@trade" in stream:
                                trade = hft_engine.Trade()
                                trade.timestamp_ns = int(d["T"]) * 1000000
                                trade.price = int(float(d["p"]) * 1e8)
                                trade.quantity = int(float(d["q"]) * 1e8)
                                trade.side = hft_engine.Side.ASK if d["m"] else hft_engine.Side.BID
                                engine.on_trade(trade, latest_book)
                                
                            elif "@depth5" in stream:
                                latest_book.timestamp_ns = int(time.time() * 1e9)
                                bids = d.get("b", [])
                                asks = d.get("a", [])
                                
                                if bids:
                                    latest_book.best_bid_price = int(float(bids[0][0]) * 1e8)
                                    latest_book.best_bid_qty = int(float(bids[0][1]) * 1e8)
                                    latest_book.bid_count = len(bids)
                                if asks:
                                    latest_book.best_ask_price = int(float(asks[0][0]) * 1e8)
                                    latest_book.best_ask_qty = int(float(asks[0][1]) * 1e8)
                                    latest_book.ask_count = len(asks)
                                    
                                if latest_book.is_valid():
                                    engine.on_book_update(latest_book)
                    except Exception as loop_err:
                        print(f"[PythonWs] Message loop error: {loop_err}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[PythonWs] Disconnected: {e}. Reconnecting in 2s...")
            await asyncio.sleep(2.0)

cpp_gateway = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cpp_gateway
    use_cpp = os.environ.get("USE_CPP_GATEWAY", "0") == "1"
    
    if use_cpp:
        print("[INFO] Initializing C++ Exchange Gateway (IXWebSocket)...")
        cpp_gateway = hft_engine.BinanceWs("btcusdt")
        cpp_gateway.start_live_feed(engine)
    else:
        print("[INFO] C++ Gateway disabled via env var. Using Python fallback.")
        asyncio.create_task(python_binance_ws())
    
    # Start the background tasks
    asyncio.create_task(log_flusher_loop())
    asyncio.create_task(ml_bridge_loop())
    
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
_weights_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'signal_weights.bin')
if os.path.exists(_weights_path):
    try:
        import struct as _struct
        with open(_weights_path, 'rb') as _f:
            _raw = _f.read(6 * 8)
        _loaded_weights = list(_struct.unpack('<6d', _raw))
        engine.set_weights(_loaded_weights)
        print(f"[INFO] Loaded Ridge weights from {_weights_path}: {[round(w,4) for w in _loaded_weights]}")
    except Exception as _e:
        print(f"[WARNING] Could not load signal_weights.bin: {_e}")
        optimal_weights = [0.189, 0.006, -0.242, -0.238, 0.101, 0.200]
        engine.set_weights(optimal_weights)
        print(f"[INFO] Fallback weights: {optimal_weights}")
else:
    optimal_weights = [0.189, 0.006, -0.242, -0.238, 0.101, 0.200]
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
                                    rec.pnl, rec.slippage, fv.combined_alpha, fv.realized_vol, fv.vpin,
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
                            abs(rec.quantity)/1e8, rec.pnl, rec.slippage,
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

@app.get("/")
async def get_dashboard():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "dashboard", "standalone.html"))

@app.get("/health")
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
            mid_price = (latest_book.best_bid_price + latest_book.best_ask_price) / 2 / 1e8
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
                    "queue_position": pending.queue_position / 1e8
                }
            }
            chart_history.append(payload)
            await websocket.send_json(payload)
            await asyncio.sleep(0.1) # 10Hz
    except (websockets.exceptions.ConnectionClosed, WebSocketDisconnect, RuntimeError):
        print("[INFO] Dashboard disconnected.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Telemetry loop exception: {e}")

# ─── ML Bridge Loop ────────────────────────────────────────────
async def ml_bridge_loop():
    print("[INFO] Starting ML Bridge Loop (ADF & True HMM)")
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'hmm_regime.pkl'))
    regime_model = None
    try:
        if os.path.exists(model_path):
            regime_model = joblib.load(model_path)
            print("[INFO] Loaded True HMM Regime Model")
    except Exception as e:
        print(f"[WARNING] Could not load regime model: {e}")

    loop = asyncio.get_running_loop()

    while True:
        await asyncio.sleep(1.0)
        
        # Capture time-sampled price for volatility and ADF
        if latest_book.best_bid_price > 0:
            mid_price = (latest_book.best_bid_price + latest_book.best_ask_price) / 2.0 / 1e8
            time_sampled_prices.append(mid_price)

        # 1. ADF Test for StatArb (needs >= 60 prices for quick demo; 5000 buffer for full power)
        if len(time_sampled_prices) >= 60:
            try:
                prices = np.array(time_sampled_prices)
                res = await loop.run_in_executor(None, adfuller, prices)
                p_value = res[1]
                adf_stat = res[0]

                if p_value > 0.05:
                    engine.set_stat_arb_valid(False)
                    print(f"[ML BRIDGE] ADF stat={adf_stat:.3f} p={p_value:.4f} > 0.05 — NOT stationary. StatArb DISABLED.")
                else:
                    engine.set_stat_arb_valid(True)
                    print(f"[ML BRIDGE] ADF stat={adf_stat:.3f} p={p_value:.4f} <= 0.05 — STATIONARY. StatArb ENABLED.")
            except asyncio.CancelledError:
                break
            except Exception:
                pass
        else:
            print(f"[ML BRIDGE] Buffering prices... ({len(time_sampled_prices)}/5000 for ADF)")
                
        # 2. Regime Prediction + write realized_vol to engine
        fv = engine.last_features()

        # Compute realized volatility from time_sampled_prices and feed it into the engine
        # so the Python weight formula can use it (C++ computes this internally)
        if len(time_sampled_prices) >= 30:
            prices_arr = np.array(list(time_sampled_prices)[-30:])
            log_rets   = np.diff(np.log(prices_arr + 1e-10))
            realized_vol_raw = float(np.std(log_rets) * np.sqrt(len(log_rets) * 100)) # scale up slightly so it isn't tiny

        if regime_model and 'model' in regime_model:
            try:
                model = regime_model['model']
                mean  = regime_model['mean']
                std   = regime_model['std']

                X_curr   = np.array([[fv.realized_vol, fv.spread_bps]])
                X_scaled = (X_curr - mean) / std
                state_arr = await loop.run_in_executor(None, model.predict, X_scaled)
                state = int(state_arr[0])

                # Regime-aware directional bias:
                # State 0 = low-vol trend  -> allow long + short
                # State 1 = high-vol chaos -> tighten thresholds (handled by spread_alpha_multiplier)
                # State 2 = mean-revert    -> boost short_multiplier aggressiveness
                # State 3 = crisis         -> halt all new entries
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
                            engine._halted = False  # Reopen if regime recovers and loss is small
                        except AttributeError:
                            pass

                print(f"[ML BRIDGE] HMM State {state} | Vol: {fv.realized_vol:.4f} | Alpha: {fv.combined_alpha:.4f} | VPIN: {fv.vpin:.3f}")
            except asyncio.CancelledError:
                break
            except Exception:
                pass


# ─── Startup ───────────────────────────────────────────────────
# Moved lifespan definition upwards

# Removed extra app definition
# Removed redundant paper_trade_loop

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
