"""
ensemble_launcher.py — 3-Engine Ensemble Launcher
===================================================
Runs three strategy engines simultaneously on a single live Binance
Futures WebSocket feed:

  Engine 1: Combined  (pure_python_engine — long + short, balanced)
  Engine 2: Bullish   (bullish_engine — long-side specialist)
  Engine 3: Bearish   (bearish_engine — short-side specialist)

Each engine maintains its own:
  - Signal weights       (directionally optimized)
  - Risk parameters      (position limits, daily loss limits)
  - Trade journal        (separate CSV per engine)
  - Equity tracking

Portfolio Aggregation:
  The ensemble reports a combined equity curve and aggregated PnL.
  Position limits per engine are set so the three combined cannot
  exceed the firm's 15% max-position rule:
    Combined: max 5 BTC (15% of $10M at $77k)
    Bullish:  max 3 BTC
    Bearish:  max 2 BTC
  Total max exposure: 10 BTC = ~$770k = 7.7% of portfolio (below 15%).

API Endpoints:
  GET  /api/ensemble/status     — equity + PnL per engine + aggregate
  GET  /api/ensemble/positions  — current positions per engine
  POST /api/ensemble/halt       — emergency halt all engines
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import asyncio
import time
import csv
import uuid
import struct

import uvicorn
import websockets
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# ── Engine imports ────────────────────────────────────────────
import pure_python_engine as combined_mod
from bullish_engine import BullishStrategyEngine, BullishConfig, EngineMode as BullishMode
from bearish_engine import BearishStrategyEngine, BearishConfig, EngineMode as BearishMode


# ── Run ID ────────────────────────────────────────────────────
RUN_ID = str(uuid.uuid4())[:8]

# ── Engine 1: Combined ───────────────────────────────────────
cfg_combined = combined_mod.StrategyConfig()
cfg_combined.initial_capital       = 10_000_000.0
cfg_combined.min_warmup_ticks      = 1000
cfg_combined.order_size_btc        = 1.0
cfg_combined.max_position_btc      = 5.0
cfg_combined.alpha_entry_threshold = 0.05
cfg_combined.alpha_short_multiplier= 1.2
cfg_combined.spread_alpha_multiplier= 0.14   # was 0.05 — red-zone fix
cfg_combined.min_take_profit_bps   = 5.0
cfg_combined.maker_fee_pct         = -0.00005
cfg_combined.daily_loss_limit_usd  = 20_000.0
cfg_combined.vpin_halt_threshold   = 0.60    # was 0.70 — tightened VPIN gate

engine_combined = combined_mod.StrategyEngine(cfg_combined)
engine_combined.set_mode(combined_mod.EngineMode.BACKTEST)

# Load signal weights from Ridge training
_weights_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'signal_weights.bin')
if os.path.exists(_weights_path):
    try:
        with open(_weights_path, 'rb') as _f:
            _raw = _f.read(6 * 8)
        _w = list(struct.unpack('<6d', _raw))
        # Map to pure_python_engine weight dict keys
        engine_combined._weights = {
            "w_obi":         _w[0],
            "w_vpin":        _w[2],
            "w_vol":         _w[4],
            "w_spread":      _w[3],
            "w_ofi":         _w[1],
            "w_microprice":  0.189,
            "w_bias":        0.0,
        }
        print(f"[COMBINED] Loaded Ridge weights")
    except Exception as e:
        print(f"[COMBINED] Weight load failed: {e}. Using defaults.")
else:
    print("[COMBINED] No signal_weights.bin found. Using built-in defaults.")

# ── Engine 2: Bullish ────────────────────────────────────────
cfg_bullish          = BullishConfig()
cfg_bullish.initial_capital = 10_000_000.0  # independent capital allocation
engine_bullish       = BullishStrategyEngine(cfg_bullish)
engine_bullish.set_mode(BullishMode.BACKTEST)

# ── Engine 3: Bearish ────────────────────────────────────────
cfg_bearish          = BearishConfig()
cfg_bearish.initial_capital = 10_000_000.0
engine_bearish       = BearishStrategyEngine(cfg_bearish)
engine_bearish.set_mode(BearishMode.BACKTEST)

# ── Shared book state ────────────────────────────────────────
latest_book = combined_mod.BookSnapshot()
latest_book.best_bid_price = 0
latest_book.best_ask_price = 0

# ── CSV Journal files ────────────────────────────────────────
def _open_journal(name):
    path = f"ensemble_{name}_{RUN_ID}.csv"
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ["RunID", "Engine", "Timestamp", "Side", "Price", "Qty", "PnL", "Equity"])
    return path

journal_paths = {
    "combined": _open_journal("combined"),
    "bullish":  _open_journal("bullish"),
    "bearish":  _open_journal("bearish"),
}
journal_idx = {"combined": 0, "bullish": 0, "bearish": 0}


def _flush_journal(engine_name, engine_obj):
    journal = engine_obj.trade_journal()
    new = journal[journal_idx[engine_name]:]
    if not new:
        return
    with open(journal_paths[engine_name], 'a', newline='') as f:
        w = csv.writer(f)
        for r in new:
            side_str = "BUY" if str(r.side).endswith("BID") else "SELL"
            px       = (r.exit_price or r.entry_price) / 1e8
            pnl_val  = getattr(r, 'pnl', 0.0)
            w.writerow([RUN_ID, engine_name.upper(), time.time(),
                        side_str, f"{px:.2f}", f"{getattr(r, 'quantity', 0):.4f}",
                        f"{pnl_val:.4f}", f"{engine_obj.equity():.2f}"])
    journal_idx[engine_name] = len(journal)


# ── Ensemble status ─────────────────────────────────────────
HALTED = False


def ensemble_pnl():
    """Aggregate PnL across all three engines."""
    c = engine_combined.equity() - cfg_combined.initial_capital
    b = engine_bullish.equity()  - cfg_bullish.initial_capital
    r = engine_bearish.equity()  - cfg_bearish.initial_capital
    return {"combined": round(c, 2), "bullish": round(b, 2),
            "bearish": round(r, 2), "total": round(c + b + r, 2)}


# ── WebSocket feed ───────────────────────────────────────────
async def binance_ws_loop():
    global HALTED
    url = "wss://fstream.binance.com/stream?streams=btcusdt@trade/btcusdt@depth5@100ms"
    print(f"[ENSEMBLE] Connecting: {url}")

    while True:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                print("[ENSEMBLE] Connected to Binance Futures.")
                async for message in ws:
                    if HALTED:
                        continue
                    outer = json.loads(message)
                    data  = outer.get('data', outer)
                    event = data.get('e', '')

                    if event in ('trade', 't'):
                        price = float(data['p'])
                        qty   = float(data['q'])
                        is_bm = data.get('m', False)

                        trade_c          = combined_mod.Trade()
                        trade_c.price    = int(price * 1e8)
                        trade_c.quantity = int(qty   * 1e8)
                        trade_c.qty      = int(qty   * 1e8)
                        trade_c.side     = (combined_mod.Side.ASK
                                            if is_bm else combined_mod.Side.BID)

                        # Feed all three engines
                        engine_combined.on_trade(trade_c, latest_book)
                        engine_bullish.on_trade(trade_c,  latest_book)
                        engine_bearish.on_trade(trade_c,  latest_book)

                        # Flush journals
                        _flush_journal("combined", engine_combined)
                        _flush_journal("bullish",  engine_bullish)
                        _flush_journal("bearish",  engine_bearish)

                    elif event == 'depthUpdate' or ('b' in data and 'a' in data):
                        bids = data.get('b', data.get('bids', []))
                        asks = data.get('a', data.get('asks', []))
                        if bids and asks:
                            bb_p = float(bids[0][0])
                            bb_q = float(bids[0][1])
                            ba_p = float(asks[0][0])
                            ba_q = float(asks[0][1])

                            latest_book.best_bid_price = int(bb_p * 1e8)
                            latest_book.best_ask_price = int(ba_p * 1e8)
                            latest_book.best_bid_qty   = int(bb_q * 1e8)
                            latest_book.best_ask_qty   = int(ba_q * 1e8)
                            latest_book.bid_count      = 1
                            latest_book.ask_count      = 1

                            engine_combined.on_book_update(latest_book)
                            engine_bullish.on_book_update(latest_book)
                            engine_bearish.on_book_update(latest_book)

        except Exception as e:
            print(f"[ENSEMBLE] WS error: {e}. Reconnecting in 2s...")
            await asyncio.sleep(2)


# ── FastAPI app ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(binance_ws_loop())
    yield

app = FastAPI(title="HFT Ensemble API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/ensemble/status")
async def status():
    pnl = ensemble_pnl()
    return {
        "run_id":    RUN_ID,
        "halted":    HALTED,
        "timestamp": time.time(),
        "engines": {
            "combined": {
                "equity":   round(engine_combined.equity(), 2),
                "position": round(engine_combined.position(), 6),
                "pnl":      pnl["combined"],
                "trades":   engine_combined._metrics.total_trades,
                "alpha":    round(engine_combined._last_features.combined_alpha, 4),
            },
            "bullish": {
                "equity":   round(engine_bullish.equity(), 2),
                "position": round(engine_bullish.position(), 6),
                "pnl":      pnl["bullish"],
                "trades":   engine_bullish._metrics.total_trades,
                "alpha":    round(engine_bullish._fv.combined_alpha, 4),
            },
            "bearish": {
                "equity":   round(engine_bearish.equity(), 2),
                "position": round(engine_bearish.position(), 6),
                "pnl":      pnl["bearish"],
                "trades":   engine_bearish._metrics.total_trades,
                "alpha":    round(engine_bearish._fv.combined_alpha, 4),
            },
        },
        "portfolio": {
            "total_equity": round(
                engine_combined.equity()
                + engine_bullish.equity()
                + engine_bearish.equity(), 2),
            "total_pnl": pnl["total"],
            "total_btc_long": round(
                max(engine_combined.position(), 0)
                + max(engine_bullish.position(), 0), 6),
            "total_btc_short": round(
                abs(min(engine_combined.position(), 0))
                + abs(min(engine_bearish.position(), 0)), 6),
        }
    }


@app.get("/api/ensemble/positions")
async def positions():
    return {
        "combined": engine_combined.position(),
        "bullish":  engine_bullish.position(),
        "bearish":  engine_bearish.position(),
        "net_btc": (engine_combined.position()
                    + engine_bullish.position()
                    + engine_bearish.position()),
    }


@app.post("/api/ensemble/halt")
async def halt_all():
    global HALTED
    HALTED = True
    engine_combined._halted = True
    engine_bullish._halted  = True
    engine_bearish._halted  = True

    # Emergency flatten all positions
    # if latest_book.best_bid_price > 0:
    #     engine_combined.flatten(latest_book)
    #     engine_bullish.flatten(latest_book)
    #     engine_bearish.flatten(latest_book)

    return {"status": "HALTED", "timestamp": time.time(),
            "final_pnl": ensemble_pnl()}


@app.post("/api/ensemble/resume")
async def resume():
    global HALTED
    HALTED = False
    engine_combined._halted = False
    engine_bullish._halted  = False
    engine_bearish._halted  = False
    return {"status": "RESUMED", "timestamp": time.time()}


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    import socket

    print("=" * 60)
    print(" HFT 3-Engine Ensemble Launcher")
    print(f" Run ID: {RUN_ID}")
    print(" Engines: Combined | Bullish | Bearish")
    print(f" Journals: {list(journal_paths.values())}")
    print("=" * 60)

    async def run_server():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', 8001))   # Port 8001 — separate from live_paper_trade.py
        except OSError:
            sock.bind(('127.0.0.1', 0))
        sock.listen(128)
        port = sock.getsockname()[1]
        print(f"[ENSEMBLE] Dashboard API at: http://127.0.0.1:{port}/api/ensemble/status")

        cfg = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(cfg)
        await server.serve(sockets=[sock])

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\n[ENSEMBLE] Shutdown requested.")
        # if latest_book.best_bid_price > 0:
        #     engine_combined.flatten(latest_book)
        #     engine_bullish.flatten(latest_book)
        #     engine_bearish.flatten(latest_book)
        pnl = ensemble_pnl()
        print(f"[ENSEMBLE] Final PnL — Combined: ${pnl['combined']:+,.2f} | "
              f"Bullish: ${pnl['bullish']:+,.2f} | "
              f"Bearish: ${pnl['bearish']:+,.2f} | "
              f"TOTAL: ${pnl['total']:+,.2f}")
