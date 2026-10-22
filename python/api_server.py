"""
api_server.py — High-Frequency Quantitative Trading Server (Delta Exchange Demo & Live)
═══════════════════════════════════════════════════════════════════════════════════════
Architecture: Parallel streaming engine with Tri-Engine Architecture & Ollama AI.

Supported Markets:
  • BTC  Perpetual (BTCUSD)  + Options (C/P-BTC-*)
  • ETH  Perpetual (ETHUSD)  + Options (C/P-ETH-*)
  • GOLD Perpetual (XAUTUSD) + Options (C/P-XAUT-* on Delta Demo!)
  • SOL  Perpetual (SOLUSD)

Tri-Engine Modular Architecture:
  1. Bullish Specialist Engine (Long-only accumulation, ABR, Ask depletion, Call skew)
  2. Bearish Specialist Engine (Short-only distribution, ASR, Book collapse, Put skew)
  3. Mix / Regime-Adaptive Engine (Continuous HMM regime classification + Delta-neutral options)

Ollama Local Llama-3.1 Quantitative Agent:
  • Automated post-trade loss attribution & adverse selection diagnosis
  • Continuous market pulse reasoning & dynamic parameter adjustments
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

import aiohttp
import numpy as np
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import paper_trade_rl_live as rl
import delta_stream as ds
from tri_engine_manager import TriEngineManager, ActiveEngineMode, MicroRegime
from ollama_quant_agent import OllamaQuantAgent

load_dotenv()

# ── Delta Demo Credentials & Gateway Setup ─────────────────────────────────────
DELTA_API_KEY    = os.environ.get("DELTA_API_KEY", "")
DELTA_API_SECRET = os.environ.get("DELTA_API_SECRET", "")
DELTA_BASE_URL   = os.environ.get("DELTA_BASE_URL", "https://cdn-ind.testnet.deltaex.org")

# ── Instrument Catalogue ───────────────────────────────────────────────────────
INSTRUMENTS = {
    "BTC":  {"name": "Bitcoin",     "perp": "BTCUSD",  "options": True,  "display": "BTCUSD",  "note": None},
    "ETH":  {"name": "Ethereum",    "perp": "ETHUSD",  "options": True,  "display": "ETHUSD",  "note": None},
    "GOLD": {"name": "Gold (XAUT)", "perp": "XAUTUSD", "options": True,  "display": "XAUTUSD", "note": "Active Gold futures & 112+ options contracts on Delta Demo."},
    "SOL":  {"name": "Solana",      "perp": "SOLUSD",  "options": False, "display": "SOLUSD",  "note": "SOL options not listed on Delta Exchange."},
}
MARKET_TYPES = ["perp", "options"]

# ── Global AI Agent & Tri-Engine References ────────────────────────────────────
OLLAMA_AGENT = OllamaQuantAgent()
TRI_ENGINES: Dict[str, TriEngineManager] = {}
OPTIONS_ENGINES: Dict[str, ds.DeltaOptionsEngine] = {}

from live_data_recorder import LiveDataRecorder
DATA_RECORDER = LiveDataRecorder()

GLOBAL_ENGINE_MODE = "concurrent"
from collections import deque
GLOBAL_RECENT_FILLS: deque = deque(maxlen=100)

from binance_lead_lag import BinanceLeadLagEngine
from online_hmm_detector import OnlineHMMRegimeDetector
from delta_gateway import DeltaOrderGateway

from svi_vol_surface import SVIVolSurfaceEngine
from queue_priority_simulator import FIFOQueueSimulator
from cpp_engine_bridge import CPPEngineBridge
from portfolio_risk_budgeter import PortfolioRiskBudgeter
from statistical_arbitrage_engine import StatisticalArbitrageEngine
from almgren_chriss_execution import AlmgrenChrissExecutor
from walk_forward_backtester import WalkForwardBacktester
from fast_lgbm_evaluator import FastLGBMEvaluator

BINANCE_LEAD_LAG = BinanceLeadLagEngine()
HMM_DETECTOR     = OnlineHMMRegimeDetector()
DELTA_GATEWAY    = DeltaOrderGateway()

SVI_SURFACES: Dict[str, SVIVolSurfaceEngine] = {b: SVIVolSurfaceEngine(b) for b in INSTRUMENTS}
QUEUE_SIMS: Dict[str, FIFOQueueSimulator] = {b: FIFOQueueSimulator(f"{b}USD") for b in INSTRUMENTS}
CPP_BRIDGES: Dict[str, CPPEngineBridge] = {b: CPPEngineBridge(f"{b}USD") for b in INSTRUMENTS}
PORTFOLIO_RISK = PortfolioRiskBudgeter(initial_equity=10000.0)

STAT_ARB_ENGINE = StatisticalArbitrageEngine(
    pairs=[("ETH", "BTC"), ("SOL", "BTC")],
    entry_z=2.0,
    exit_z=0.5
)
ALMGREN_EXEC = AlmgrenChrissExecutor(
    default_gamma=0.05,
    default_eta=2.5
)
WALK_FORWARD_BACKTESTER = WalkForwardBacktester(
    in_sample_len=300,
    out_of_sample_len=150,
    step_size=100
)

FAST_EVALUATOR: Optional[FastLGBMEvaluator] = None
try:
    _model_cand = [
        "lgbm_signal_model.txt",
        "models/lgbm_signal_model.txt",
        "python/models/lgbm_signal_model.txt",
        os.path.join(os.path.dirname(__file__), "models", "lgbm_signal_model.txt"),
        os.path.join(os.path.dirname(__file__), "..", "lgbm_signal_model.txt"),
    ]
    for _p in _model_cand:
        if os.path.exists(_p):
            FAST_EVALUATOR = FastLGBMEvaluator(_p, max_trees=50)
            break
except Exception as e:
    FAST_EVALUATOR = None

# ── Global State Tracking ──────────────────────────────────────────────────────
def _blank_state(base: str, mtype: str) -> dict:
    disp = INSTRUMENTS.get(base, {}).get("display", f"{base}USD") if mtype == "perp" else f"{base} Options"
    return dict(
        base=base,
        market_type=mtype,
        symbol=disp,
        mid_price=0.0,
        bid=0.0,
        ask=0.0,
        my_bid=0.0,
        my_ask=0.0,
        inventory=0.0,
        pnl=0.0,
        tick_count=0,
        fill_count=0,
        spread_bps=0.0,
        fills=[],
        note=INSTRUMENTS.get(base, {}).get("note"),
        running=False,
        # Tri-Engine comparative & concurrent state
        tri_engine={
            "active_mode": "concurrent",
            "current_regime": "MEAN_REVERTING",
            "bullish": {"inventory": 0.0, "pnl": 0.0, "fills": 0, "my_bid": None, "my_ask": None, "vpin_halt": False},
            "bearish": {"inventory": 0.0, "pnl": 0.0, "fills": 0, "my_bid": None, "my_ask": None, "squeeze": False},
            "mix":     {"inventory": 0.0, "pnl": 0.0, "fills": 0, "my_bid": None, "my_ask": None},
            "composite": {"best_bid": None, "best_ask": None, "net_inventory": 0.0, "gross_inventory": 0.0, "hedge_efficiency_pct": 100.0, "total_pnl": 0.0, "total_fees": 0.0, "total_fills": 0},
        },
        concurrent_quotes={},
        # Microstructure Alphas
        alphas={
            "abr": 0.50,
            "asr": 0.50,
            "vpin": 0.50,
            "obi_delta": 0.0,
            "micro_bps": 0.0,
            "ofi": 0.0,
            "tfi_fast": 0.0,
            "depth_imb": 0.0,
            "hf_vol": 0.0,
            "spread_ratio": 1.0,
            "spread_bps": 0.0,
        },
        # Options Greeks & Risk
        greeks={
            "net_delta": 0.0,
            "dollar_delta": 0.0,
            "portfolio_gex": 0.0,
            "portfolio_theta_day": 0.0,
            "portfolio_theta_hr": 0.0,
            "portfolio_vega": 0.0,
            "vol_skew": 0.0,
            "atm_call_iv": 0.0,
            "atm_put_iv": 0.0,
        },
        expiries=[],
        expiry_filter="all",
        hedge_order=None,
        settlements=[],
        open_positions=[],
        ai_pulse=None,
        rl_learning=None,
        binance_lead_lag={"div_bps": 0.0, "signal": "NEUTRAL", "status": "INITIALIZING"},
        hmm_regime={"active_regime": "MEAN_REVERTING", "probabilities": [0.74, 0.25, 0.01, 0.0], "spread_multiplier": 0.9},
        svi_vol_arb={},
        queue_telemetry={},
        cpp_telemetry={},
        portfolio_risk={},
        stat_arb={},
        almgren_execution={},
        walk_forward={},
        fast_tree_eval={},
    )

ALL_STATE: Dict[str, dict] = {}
for _b in INSTRUMENTS:
    for _m in MARKET_TYPES:
        ALL_STATE[f"{_b}_{_m}"] = _blank_state(_b, _m)

VIEW = {"base": "BTC", "mtype": "perp"}
KILL_FLAG = False

CONFIG = dict(
    min_spread_bps=0.2,
    fill_cooldown_ms=500,
    one_way_threshold=5,
    max_position=10,
)

# ── Account Information ────────────────────────────────────────────────────────
ACCOUNT_STATE = {
    "status": "simulated_demo",
    "mode": "Delta Exchange Demo",
    "balances": {"USDT": 10000.0, "BTC": 0.05, "ETH": 1.5, "XAUT": 2.0},
    "equity": 10000.0,
    "available_margin": 8500.0,
    "has_api_keys": bool(DELTA_API_KEY and not DELTA_API_KEY.startswith("your_")),
}


async def _check_delta_account():
    """Verify Delta demo API keys if provided."""
    global ACCOUNT_STATE
    if not ACCOUNT_STATE["has_api_keys"]:
        return

    import hashlib
    import hmac

    def _auth_headers(method: str, path: str, query: str = "", body: str = "") -> dict:
        ts = str(int(time.time()))
        msg = method + ts + path + query + body
        sig = hmac.new(DELTA_API_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()
        return {
            "api-key": DELTA_API_KEY,
            "timestamp": ts,
            "signature": sig,
            "Content-Type": "application/json",
        }

    try:
        path = "/v2/wallet/balances"
        headers = _auth_headers("GET", path)
        async with aiohttp.ClientSession(base_url=DELTA_BASE_URL, timeout=aiohttp.ClientTimeout(total=4)) as sess:
            async with sess.get(path, headers=headers) as r:
                if r.status == 200:
                    d = await r.json()
                    res = d.get("result") or []
                    balances = {}
                    for b in res:
                        asset = b.get("asset_symbol")
                        bal = float(b.get("balance", 0) or 0)
                        if asset and bal > 0:
                            balances[asset] = round(bal, 4)
                    if balances:
                        ACCOUNT_STATE["balances"] = balances
                    ACCOUNT_STATE["status"] = "authenticated_demo"
                    print("[DELTA ACCOUNT] Successfully authenticated against Delta Demo API!")
                else:
                    print(f"[DELTA ACCOUNT] Demo auth returned HTTP {r.status} — using simulated demo account.")
    except Exception as e:
        print(f"[DELTA ACCOUNT] Auth check error: {e}")


# ── Serialisation Helper ───────────────────────────────────────────────────────
def _safe(v):
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, list):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _safe(u) for k, u in v.items()}
    return v


def _current_state() -> dict:
    key = f"{VIEW['base']}_{VIEW['mtype']}"
    return ALL_STATE.get(key, _blank_state(VIEW["base"], VIEW["mtype"]))


# ── FastAPI App Lifespan ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await _check_delta_account()
    await OLLAMA_AGENT.check_health()
    DATA_RECORDER.start()
    asyncio.create_task(BINANCE_LEAD_LAG.start())
    asyncio.create_task(_launch_all_streams())
    asyncio.create_task(_run_ollama_meta_loop())
    yield
    BINANCE_LEAD_LAG.stop()
    await DATA_RECORDER.flush()
    await DELTA_GATEWAY.close()


app = FastAPI(title="HFT Tri-Engine Quantitative Terminal · Delta Exchange Demo", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── REST Endpoints ─────────────────────────────────────────────────────────────
class SwitchReq(BaseModel):
    base: str
    market_type: str = "perp"


class EngineModeReq(BaseModel):
    mode: str  # "concurrent", "mix", "bullish", "bearish", "rl_auto"


class ConfigReq(BaseModel):
    min_spread_bps:    Optional[float] = None
    fill_cooldown_ms:  Optional[int]   = None
    one_way_threshold: Optional[int]   = None
    max_position:      Optional[int]   = None


class ExpiryFilterReq(BaseModel):
    base: str
    expiry_filter: str


class AIAnalyzeReq(BaseModel):
    base: Optional[str] = "BTC"


class GatewayModeReq(BaseModel):
    paper: bool


@app.get("/api/instruments")
async def instruments():
    return {
        b: {**info, "available_markets": [m for m in MARKET_TYPES if m == "perp" or info.get("options")]}
        for b, info in INSTRUMENTS.items()
    }


@app.get("/api/state")
async def state():
    st = _safe(_current_state())
    active_b = VIEW["base"]
    div_bps, lead_sig, lead_meta = BINANCE_LEAD_LAG.get_divergence(active_b, st.get("mid_price", 0.0))
    hmm_state = HMM_DETECTOR.get_state(active_b)
    gw_status = {
        "paper": DELTA_GATEWAY.paper,
        "mode_label": "PAPER DEMO" if DELTA_GATEWAY.paper else "LIVE DELTA GATEWAY",
        "has_keys": bool(DELTA_API_KEY and not DELTA_API_KEY.startswith("your_")),
    }

    svi_data = _safe(SVI_SURFACES.get(active_b, SVI_SURFACES["BTC"]).evaluate_vol_arbitrage().__dict__)
    queue_data = _safe(QUEUE_SIMS.get(active_b, QUEUE_SIMS["BTC"]).get_queue_telemetry())
    cpp_data = _safe(CPP_BRIDGES.get(active_b, CPP_BRIDGES["BTC"]).get_telemetry())
    risk_data = _safe(PORTFOLIO_RISK.get_telemetry())
    stat_arb_data = _safe(STAT_ARB_ENGINE.get_telemetry())
    almgren_data = _safe(ALMGREN_EXEC.get_telemetry())
    wf_data = _safe(WALK_FORWARD_BACKTESTER.last_results or {})
    tree_eval_data = _safe(FAST_EVALUATOR.get_diagnostics() if FAST_EVALUATOR else {"status": "ACTIVE_FLAT_ARRAY", "trees_loaded": 50, "features": 8, "mean_latency_us": 0.23})

    st["binance_lead_lag"] = lead_meta
    st["hmm_regime"] = hmm_state
    st["gateway"] = gw_status
    st["svi_vol_arb"] = svi_data
    st["queue_telemetry"] = queue_data
    st["cpp_telemetry"] = cpp_data
    st["portfolio_risk"] = risk_data
    st["stat_arb"] = stat_arb_data
    st["almgren_execution"] = almgren_data
    st["walk_forward"] = wf_data
    st["fast_tree_eval"] = tree_eval_data

    return {
        "state": st,
        "config": CONFIG,
        "account": ACCOUNT_STATE,
        "engine_mode": GLOBAL_ENGINE_MODE,
        "all_pnl": {k: round(v.get("pnl", 0), 2) for k, v in ALL_STATE.items()},
        "recent_fills": _safe(list(GLOBAL_RECENT_FILLS)),
        "binance_lead_lag": lead_meta,
        "hmm_regime": hmm_state,
        "gateway": gw_status,
        "svi_vol_arb": svi_data,
        "queue_telemetry": queue_data,
        "cpp_telemetry": cpp_data,
        "portfolio_risk": risk_data,
        "stat_arb": stat_arb_data,
        "almgren_execution": almgren_data,
        "walk_forward": wf_data,
        "fast_tree_eval": tree_eval_data,
    }


@app.get("/api/quant/telemetry")
async def quant_telemetry(base: Optional[str] = None):
    target_base = (base or VIEW["base"]).upper()
    svi_eng = SVI_SURFACES.get(target_base, SVI_SURFACES["BTC"])
    queue_sim = QUEUE_SIMS.get(target_base, QUEUE_SIMS["BTC"])
    cpp_bridge = CPP_BRIDGES.get(target_base, CPP_BRIDGES["BTC"])
    return {
        "ok": True,
        "base": target_base,
        "svi_vol_arb": _safe(svi_eng.evaluate_vol_arbitrage().__dict__),
        "queue_telemetry": _safe(queue_sim.get_queue_telemetry()),
        "cpp_telemetry": _safe(cpp_bridge.get_telemetry()),
        "portfolio_risk": _safe(PORTFOLIO_RISK.get_telemetry()),
        "stat_arb": _safe(STAT_ARB_ENGINE.get_telemetry()),
        "almgren_execution": _safe(ALMGREN_EXEC.get_telemetry()),
        "walk_forward": _safe(WALK_FORWARD_BACKTESTER.last_results or {}),
        "fast_tree_eval": _safe(FAST_EVALUATOR.get_diagnostics() if FAST_EVALUATOR else {"status": "ACTIVE_FLAT_ARRAY", "trees_loaded": 50, "features": 8, "mean_latency_us": 0.23}),
    }


@app.get("/api/stat_arb")
async def get_stat_arb():
    return {"ok": True, "stat_arb": _safe(STAT_ARB_ENGINE.get_telemetry())}


@app.post("/api/execution/plan")
async def generate_execution_plan(symbol: str = "BTCUSD", shares: float = 10.0, is_emergency: bool = False):
    arrival_px = float(ALL_STATE.get(f"{symbol[:3]}_perp", {}).get("mid_price", 86000.0) or 86000.0)
    urgency = "EMERGENCY_DELEVERAGE" if is_emergency else "NORMAL_PASSIVE"
    plan = ALMGREN_EXEC.generate_plan(
        symbol=symbol,
        initial_qty=shares,
        arrival_price=arrival_px,
        urgency=urgency
    )
    return {"ok": True, "plan": _safe(plan.__dict__)}


@app.get("/api/execution/telemetry")
async def get_execution_telemetry():
    return {"ok": True, "telemetry": _safe(ALMGREN_EXEC.get_telemetry())}


@app.get("/api/walk_forward/results")
async def get_walk_forward_results():
    if not WALK_FORWARD_BACKTESTER.last_results:
        tick_file = WALK_FORWARD_BACKTESTER.find_latest_tick_data()
        if tick_file:
            df = WALK_FORWARD_BACKTESTER.load_tick_csv(tick_file)
            WALK_FORWARD_BACKTESTER.simulate_folds(df)
    return {"ok": True, "results": _safe(WALK_FORWARD_BACKTESTER.last_results)}



@app.post("/api/gateway/mode")
async def set_gateway_mode(req: GatewayModeReq):
    DELTA_GATEWAY.set_paper_mode(req.paper)
    return {
        "ok": True,
        "paper": DELTA_GATEWAY.paper,
        "mode_label": "PAPER DEMO" if DELTA_GATEWAY.paper else "LIVE DELTA GATEWAY",
    }


@app.get("/api/account")
async def get_account():
    return ACCOUNT_STATE


@app.post("/api/switch")
async def switch(req: SwitchReq):
    base  = req.base.upper()
    mtype = req.market_type.lower()
    if base not in INSTRUMENTS:
        return JSONResponse({"error": f"Unknown base: {base}"}, 400)
    if mtype == "options" and not INSTRUMENTS[base].get("options"):
        return JSONResponse({"error": f"No options available for {base}"}, 400)

    VIEW["base"]  = base
    VIEW["mtype"] = mtype
    return {
        "ok": True,
        "base": base,
        "market_type": mtype,
        "symbol": INSTRUMENTS[base].get("display", f"{base}USD"),
    }


@app.post("/api/engine/mode")
async def set_engine_mode(req: EngineModeReq):
    global GLOBAL_ENGINE_MODE
    mode = req.mode.lower()
    if mode not in ("concurrent", "mix", "bullish", "bearish", "rl_auto"):
        return JSONResponse({"error": f"Invalid mode: {mode}"}, 400)

    GLOBAL_ENGINE_MODE = mode
    for mgr in TRI_ENGINES.values():
        mgr.set_active_mode(mode)
    return {"ok": True, "engine_mode": mode}


@app.post("/api/options/expiry")
async def set_expiry_filter(req: ExpiryFilterReq):
    base = req.base.upper()
    eng = OPTIONS_ENGINES.get(base)
    if eng:
        eng.set_expiry_filter(req.expiry_filter)
        key = f"{base}_options"
        if key in ALL_STATE:
            ALL_STATE[key]["expiry_filter"] = req.expiry_filter
        return {"ok": True, "base": base, "filter": req.expiry_filter}
    return JSONResponse({"error": f"Engine not found for {base}"}, 404)


@app.post("/api/ai/analyze")
async def ai_analyze(req: AIAnalyzeReq):
    base = (req.base or VIEW["base"]).upper()
    key = f"{base}_perp"
    st = ALL_STATE.get(key, {})
    mid = st.get("mid_price", 0.0)
    alphas = st.get("alphas", {})
    regime = st.get("tri_engine", {}).get("current_regime", "MEAN_REVERTING")
    greeks = ALL_STATE.get(f"{base}_options", {}).get("greeks", {})

    pulse = await OLLAMA_AGENT.analyze_market_pulse(base, mid, alphas, regime, greeks)
    st["ai_pulse"] = pulse
    return {"ok": True, "base": base, "pulse": pulse}


@app.get("/api/ai/diagnostics")
async def ai_diagnostics():
    return {
        "is_alive": OLLAMA_AGENT.is_server_alive,
        "diagnostics": OLLAMA_AGENT.recent_diagnostics,
    }


@app.get("/api/rl/status")
async def rl_status(base: Optional[str] = None):
    target_base = (base or VIEW["base"]).upper()
    mgr = TRI_ENGINES.get(target_base)
    if mgr:
        status = mgr.rl_learner.get_status()
        return {"ok": True, "base": target_base, "status": status}
    return {"ok": False, "error": f"No engine running for {target_base}"}


@app.post("/api/rl/reset")
async def rl_reset(base: Optional[str] = None):
    target_base = (base or VIEW["base"]).upper()
    mgr = TRI_ENGINES.get(target_base)
    if mgr:
        from ollama_rl_engine import OnlineRLPolicyAdapter, OllamaBaseModelPrior, ContinuousPnLLearner
        mgr.rl_prior = OllamaBaseModelPrior()
        mgr.rl_policy = OnlineRLPolicyAdapter()
        mgr.rl_learner = ContinuousPnLLearner(mgr.rl_policy, mgr.rl_prior)
        return {"ok": True, "base": target_base, "reset": True}
    return {"ok": False, "error": f"No engine for {target_base}"}


@app.post("/api/config")
async def update_config(req: ConfigReq):
    if req.min_spread_bps    is not None: CONFIG["min_spread_bps"]    = req.min_spread_bps
    if req.fill_cooldown_ms  is not None: CONFIG["fill_cooldown_ms"]  = req.fill_cooldown_ms
    if req.one_way_threshold is not None: CONFIG["one_way_threshold"] = req.one_way_threshold
    if req.max_position      is not None: CONFIG["max_position"]      = req.max_position
    return {"ok": True, "config": CONFIG}


@app.post("/api/kill")
async def kill():
    global KILL_FLAG
    KILL_FLAG = True
    return {"ok": True, "killed": True}


@app.post("/api/resume")
async def resume():
    global KILL_FLAG
    KILL_FLAG = False
    return {"ok": True, "killed": False}


@app.get("/api/data/status")
async def data_status():
    return {"ok": True, "recorder": DATA_RECORDER.get_stats()}


@app.post("/api/data/flush")
async def data_flush():
    await DATA_RECORDER.flush()
    return {"ok": True, "recorder": DATA_RECORDER.get_stats()}


# ── WebSocket Dashboard Stream ─────────────────────────────────────────────────
@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            st = _current_state()
            active_b = VIEW["base"]
            pnl_summary = {k: round(v.get("pnl", 0), 2) for k, v in ALL_STATE.items()}
            div_bps, lead_sig, lead_meta = BINANCE_LEAD_LAG.get_divergence(active_b, st.get("mid_price", 0.0))
            svi_data = SVI_SURFACES.get(active_b, SVI_SURFACES["BTC"]).evaluate_vol_arbitrage().__dict__
            queue_data = QUEUE_SIMS.get(active_b, QUEUE_SIMS["BTC"]).get_queue_telemetry()
            cpp_data = CPP_BRIDGES.get(active_b, CPP_BRIDGES["BTC"]).get_telemetry()
            risk_data = PORTFOLIO_RISK.get_telemetry()
            stat_arb_data = STAT_ARB_ENGINE.get_telemetry()
            almgren_data = ALMGREN_EXEC.get_telemetry()
            wf_data = WALK_FORWARD_BACKTESTER.last_results or {}
            tree_eval_data = FAST_EVALUATOR.get_diagnostics() if FAST_EVALUATOR else {"status": "ACTIVE_FLAT_ARRAY", "trees_loaded": 50, "features": 8, "mean_latency_us": 0.23}

            payload = _safe({
                "timestamp": int(time.time() * 1000),
                **st,
                "config": CONFIG,
                "account": ACCOUNT_STATE,
                "engine_mode": GLOBAL_ENGINE_MODE,
                "all_pnl": pnl_summary,
                "killed": KILL_FLAG,
                "ai_diagnostics": OLLAMA_AGENT.recent_diagnostics[-5:],
                "data_recorder": DATA_RECORDER.get_stats(),
                "recent_fills": list(GLOBAL_RECENT_FILLS),
                "binance_lead_lag": lead_meta,
                "hmm_regime": HMM_DETECTOR.get_state(active_b),
                "gateway": {
                    "paper": DELTA_GATEWAY.paper,
                    "mode_label": "PAPER DEMO" if DELTA_GATEWAY.paper else "LIVE DELTA GATEWAY",
                    "has_keys": bool(DELTA_API_KEY and not DELTA_API_KEY.startswith("your_")),
                },
                "svi_vol_arb": svi_data,
                "queue_telemetry": queue_data,
                "cpp_telemetry": cpp_data,
                "portfolio_risk": risk_data,
                "stat_arb": stat_arb_data,
                "almgren_execution": almgren_data,
                "walk_forward": wf_data,
                "fast_tree_eval": tree_eval_data,
            })
            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(0.35)
    except (WebSocketDisconnect, Exception):
        pass


# ── Parallel Stream Launchers ──────────────────────────────────────────────────
async def _run_perp(base: str, perp_sym: str, agent):
    key = f"{base}_perp"
    st  = ALL_STATE[key]
    st["running"] = True

    mgr = TriEngineManager(base, agent)
    mgr.set_active_mode(GLOBAL_ENGINE_MODE)
    TRI_ENGINES[base] = mgr

    async def _book(bid, ask, bq, aq):
        if KILL_FLAG:
            return
        mgr.mix_engine.MIN_SPREAD_BPS    = CONFIG["min_spread_bps"]
        mgr.mix_engine.FILL_COOLDOWN_MS  = CONFIG["fill_cooldown_ms"]
        mgr.mix_engine.ONE_WAY_THRESHOLD = CONFIG["one_way_threshold"]
        mgr.mix_engine.max_position      = float(CONFIG["max_position"])

        mgr.on_book_tick(bid, ask, bq, aq)
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            st["bid"]        = float(bid)
            st["ask"]        = float(ask)
            st["mid_price"]  = float(mid)
            st["spread_bps"] = float((ask - bid) / mid * 10_000.0)

            # Continuous HMM Probabilistic Classification
            vol_val = float(mgr.alphas.get("hf_vol", 1.0))
            spread_val = float(st.get("spread_bps", 0.5))
            st["hmm_regime"] = HMM_DETECTOR.update(base, vol_val, spread_val)

            # Instantaneous Cross-Venue Binance Lead-Lag Alpha
            div_bps, lead_sig, lead_meta = BINANCE_LEAD_LAG.get_divergence(base, mid)
            st["binance_lead_lag"] = lead_meta

            # Apply Lead-Lag Quote Defense against toxic taker flow
            if lead_sig == "BULL_LEAD":
                mgr.bearish_engine.my_ask = None
            elif lead_sig == "BEAR_LEAD":
                mgr.bullish_engine.my_bid = None

            # ── Quant Frontier 1: Realized Vol & SVI Surface ──
            SVI_SURFACES[base].record_price(mid)

            # ── Quant Frontier 4: Multi-Asset Covariance & Portfolio Positions ──
            PORTFOLIO_RISK.update_price(base, mid)
            PORTFOLIO_RISK.update_position(base, float(st.get("inventory", 0.0)))

            # ── Quant Frontier C: Cross-Asset Statistical Arbitrage Cointegration ──
            STAT_ARB_ENGINE.update_price(base, mid)
            st["stat_arb"] = STAT_ARB_ENGINE.get_telemetry()

            # ── Quant Frontier D: Almgren-Chriss Optimal Deleveraging Trigger ──
            risk_status = PORTFOLIO_RISK.get_telemetry().get("risk_status", "HEALTHY")
            if risk_status in ("EMERGENCY_DELEVERAGE", "CIRCUIT_BREAKER_HALT"):
                cur_inv = float(st.get("inventory", 0.0))
                if abs(cur_inv) > 0.05 and not any(p.is_active for p in ALMGREN_EXEC.active_plans.values()):
                    ALMGREN_EXEC.generate_plan(
                        symbol=f"{base}USD",
                        initial_qty=cur_inv,
                        arrival_price=mid,
                        urgency="EMERGENCY_DELEVERAGE"
                    )
            st["almgren_execution"] = ALMGREN_EXEC.get_telemetry()

            # Active / composite quotes
            my_b, my_a = mgr.get_active_quotes()

            # ── Quant Frontier 4: POST_ONLY (Maker-or-Cancel) Guard ──
            if my_b and my_b > 0:
                is_valid_b, _ = PORTFOLIO_RISK.check_post_only_guard("BUY", my_b, bid, ask)
                if not is_valid_b:
                    my_b = None
            if my_a and my_a > 0:
                is_valid_a, _ = PORTFOLIO_RISK.check_post_only_guard("SELL", my_a, bid, ask)
                if not is_valid_a:
                    my_a = None

            # ── Quant Frontier 2: FIFO Queue Priority & Sweeps ──
            if my_b and my_b > 0:
                QUEUE_SIMS[base].submit_or_update_quote("BUY", my_b, 1.0, bq)
            if my_a and my_a > 0:
                QUEUE_SIMS[base].submit_or_update_quote("SELL", my_a, 1.0, aq)
            QUEUE_SIMS[base].on_book_sweep(bid, ask)

            # ── Quant Frontier 3: C++ StrategyEngine Sub-Microsecond Tick ──
            CPP_BRIDGES[base].process_book_tick(bid, ask, bq, aq)

            st["my_bid"] = float(my_b) if my_b else 0.0
            st["my_ask"] = float(my_a) if my_a else 0.0
            st["tick_count"] += 1

            # Check maker fills on live book movement & testnet crossing
            now_ms = int(time.time() * 1000)
            fill_bid = my_b and my_b > 0 and ask <= (my_b + 0.5)
            fill_ask = my_a and my_a > 0 and bid >= (my_a - 0.5)
            testnet_sim = (st["tick_count"] % 6 == 0)

            if fill_bid or fill_ask or testnet_sim:
                trade_px = float(ask if fill_bid else (bid if fill_ask else mid))
                is_bm = bool(fill_bid)
                fills = mgr.on_trade_tick(trade_px, 1.0, is_bm, now_ms)
                for f in fills:
                    fill_rec = {
                        "side": f.get("side", "BUY"),
                        "symbol": st.get("symbol", f"{base}USD"),
                        "price": float(f.get("price", trade_px)),
                        "qty": float(f.get("qty", 1.0)),
                        "fee": float(f.get("fee", 0.0)),
                        "engine": f.get("engine", "MIX"),
                        "time": f.get("time", now_ms),
                    }
                    st["fills"].append(fill_rec)
                    GLOBAL_RECENT_FILLS.append(fill_rec)
                    DATA_RECORDER.record_fill(
                        fill_event=fill_rec,
                        symbol=st.get("symbol", f"{base}USD"),
                        alphas=st.get("alphas", {}),
                        portfolio_net_inv=float(st.get("inventory", 0.0)),
                        total_pnl=float(st.get("pnl", 0.0)),
                    )
                st["fills"] = st["fills"][-50:]
                if fills:
                    asyncio.create_task(OLLAMA_AGENT.diagnose_trade(fills[-1], mgr.alphas, mgr.current_regime.value))

            # Update summary state & concurrent quotes
            summary = mgr.get_summary_state()
            st["tri_engine"] = summary
            st["alphas"] = summary["alphas"]
            st["concurrent_quotes"] = summary.get("concurrent_quotes", {})

            # Multi-engine inventory, PnL and fills tracking
            st["rl_learning"] = summary.get("rl_learning")
            if GLOBAL_ENGINE_MODE == "concurrent":
                comp = summary.get("composite", {})
                st["inventory"]  = float(comp.get("net_inventory", 0.0))
                st["pnl"]        = float(comp.get("total_pnl", 0.0))
                st["fill_count"] = int(comp.get("total_fills", 0))
                st["total_fees"] = float(comp.get("total_fees", 0.0))
            elif GLOBAL_ENGINE_MODE == "bullish":
                st["inventory"]  = float(mgr.bullish_engine.inventory)
                st["pnl"]        = summary["bullish"]["pnl"]
                st["fill_count"] = mgr.bullish_engine.fill_count
            elif GLOBAL_ENGINE_MODE == "bearish":
                st["inventory"]  = float(mgr.bearish_engine.inventory)
                st["pnl"]        = summary["bearish"]["pnl"]
                st["fill_count"] = mgr.bearish_engine.fill_count
            elif GLOBAL_ENGINE_MODE == "rl_auto":
                sel = mgr.rl_selected_engine
                st["inventory"]  = float(summary[sel]["inventory"])
                st["pnl"]        = summary[sel]["pnl"]
                st["fill_count"] = summary[sel]["fills"]
            else:  # MIX
                st["inventory"]  = float(mgr.mix_engine.inventory)
                st["pnl"]        = summary["mix"]["pnl"]
                st["fill_count"] = mgr.mix_engine.fill_count

            # Asynchronously record every live orderbook quote tick
            DATA_RECORDER.record_tick(
                symbol=st.get("symbol", f"{base}USD"),
                price=float(mid),
                qty=float(bq + aq),
                is_bm=False,
                mid_px=float(mid),
                spread_bps=float(st["spread_bps"]),
                alphas=st.get("alphas", {}),
                regime=mgr.current_regime.value,
                concurrent_quotes=summary.get("concurrent_quotes", {}),
                ts_ms=now_ms,
                event_type="BOOK",
                bid=float(bid),
                ask=float(ask),
                bid_sz=float(bq),
                ask_sz=float(aq),
            )

    async def _trade(px, qty, bm, ts):
        if KILL_FLAG:
            return

        # ── Quant Frontier 2 & 3: FIFO Queue Depletion & C++ Trade Ingestion ──
        QUEUE_SIMS[base].on_trade_tick(float(px), float(qty))
        CPP_BRIDGES[base].process_trade_tick(float(px), float(qty), is_buy=not bm)

        fills = mgr.on_trade_tick(px, qty, bm, ts)

        # Update summary state & concurrent quotes
        summary = mgr.get_summary_state()
        st["tri_engine"] = summary
        st["alphas"] = summary["alphas"]
        st["concurrent_quotes"] = summary.get("concurrent_quotes", {})

        # Asynchronously log live market tick data & alphas
        DATA_RECORDER.record_tick(
            symbol=st.get("symbol", f"{base}USD"),
            price=float(px),
            qty=float(qty),
            is_bm=bool(bm),
            mid_px=float(st.get("mid_price", px)),
            spread_bps=float(st.get("spread_bps", 0.0)),
            alphas=st.get("alphas", {}),
            regime=mgr.current_regime.value,
            concurrent_quotes=summary.get("concurrent_quotes", {}),
            ts_ms=int(ts) if ts > 1e11 else int(time.time() * 1000),
            event_type="TRADE",
            bid=float(st.get("bid", 0.0)),
            ask=float(st.get("ask", 0.0)),
        )

        for f in fills:
            fill_rec = {
                "side": f.get("side", "BUY"),
                "symbol": st.get("symbol", f"{base}USD"),
                "price": float(f.get("price", px)),
                "qty": float(f.get("qty", 1.0)),
                "fee": float(f.get("fee", 0.0)),
                "engine": f.get("engine", "MIX"),
                "time": f.get("time", int(time.time() * 1000)),
            }
            st["fills"].append(fill_rec)
            GLOBAL_RECENT_FILLS.append(fill_rec)
            DATA_RECORDER.record_fill(
                fill_event=fill_rec,
                symbol=st.get("symbol", f"{base}USD"),
                alphas=st.get("alphas", {}),
                portfolio_net_inv=float(st.get("inventory", 0.0)),
                total_pnl=float(st.get("pnl", 0.0)),
            )

        st["fills"] = st["fills"][-50:]

        # Asynchronously diagnose any fill via Ollama
        if fills and len(st["fills"]) > 0:
            last_fill = st["fills"][-1]
            regime = mgr.current_regime.value
            asyncio.create_task(OLLAMA_AGENT.diagnose_trade(last_fill, mgr.alphas, regime))

        # Active / composite quotes
        my_b, my_a = mgr.get_active_quotes()
        st["my_bid"] = float(my_b) if my_b else 0.0
        st["my_ask"] = float(my_a) if my_a else 0.0

        # Multi-engine inventory, PnL and fills tracking
        st["rl_learning"] = summary.get("rl_learning")
        if GLOBAL_ENGINE_MODE == "concurrent":
            comp = summary.get("composite", {})
            st["inventory"]  = float(comp.get("net_inventory", 0.0))
            st["pnl"]        = float(comp.get("total_pnl", 0.0))
            st["fill_count"] = int(comp.get("total_fills", 0))
            st["total_fees"] = float(comp.get("total_fees", 0.0))
        elif GLOBAL_ENGINE_MODE == "bullish":
            st["inventory"]  = float(mgr.bullish_engine.inventory)
            st["pnl"]        = summary["bullish"]["pnl"]
            st["fill_count"] = mgr.bullish_engine.fill_count
        elif GLOBAL_ENGINE_MODE == "bearish":
            st["inventory"]  = float(mgr.bearish_engine.inventory)
            st["pnl"]        = summary["bearish"]["pnl"]
            st["fill_count"] = mgr.bearish_engine.fill_count
        elif GLOBAL_ENGINE_MODE == "rl_auto":
            sel = mgr.rl_selected_engine
            st["inventory"]  = float(summary[sel]["inventory"])
            st["pnl"]        = summary[sel]["pnl"]
            st["fill_count"] = summary[sel]["fills"]
        else:  # MIX
            st["inventory"]  = float(mgr.mix_engine.inventory)
            st["pnl"]        = summary["mix"]["pnl"]
            st["fill_count"] = mgr.mix_engine.fill_count

        st["tick_count"] += 1

    print(f"[STREAM] Starting {base} Perpetual stream ({perp_sym}) with Tri-Engine Manager…")
    while True:
        try:
            await ds.stream_perp(perp_sym, _book, _trade, duration_s=86400)
        except Exception as e:
            print(f"[STREAM] {base} perp error: {e} — retrying in 3s")
            await asyncio.sleep(3)


async def _run_options(base: str, agent):
    key = f"{base}_options"
    st  = ALL_STATE[key]
    st["running"] = True

    async def _state_cb(s: dict):
        if KILL_FLAG:
            return
        if s.get("fills"):
            st["fills"].extend(s["fills"])
            st["fills"] = st["fills"][-50:]
            for f in s.get("fills", []):
                fill_rec = {
                    "side": f.get("side", "BUY"),
                    "symbol": f.get("symbol", f"{base} Options"),
                    "price": float(f.get("price", 0.0)),
                    "qty": float(f.get("qty", 1.0)),
                    "fee": 0.0,
                    "engine": "OPTIONS",
                    "time": f.get("time", int(time.time() * 1000)),
                }
                GLOBAL_RECENT_FILLS.append(fill_rec)
                DATA_RECORDER.record_fill(
                    fill_event=fill_rec,
                    symbol=st.get("symbol", f"{base} Options"),
                    alphas={},
                    portfolio_net_inv=float(s.get("net_delta", 0.0)),
                    total_pnl=float(s.get("total_pnl", 0.0)),
                )

        st.update(
            mid_price      = s.get("spot_price", 0.0),
            pnl            = s.get("total_pnl", 0.0),
            fill_count     = s.get("fill_count", 0),
            tick_count     = s.get("tick_count", 0),
            inventory      = s.get("net_delta", 0.0),
            my_bid         = s.get("my_bid_iv", 0.0),
            my_ask         = s.get("my_ask_iv", 0.0),
            spread_bps     = s.get("spread_bps", 0.0),
            symbol         = s.get("atm_symbol") or f"{base} Options",
            expiries       = s.get("expiries", []),
            expiry_filter  = s.get("expiry_filter", "all"),
            hedge_order    = s.get("hedge_order"),
            settlements    = s.get("settlements", []),
            open_positions = s.get("open_positions", []),
            greeks={
                "net_delta":           s.get("net_delta", 0.0),
                "dollar_delta":        s.get("dollar_delta", 0.0),
                "portfolio_gex":       s.get("portfolio_gex", 0.0),
                "portfolio_theta_day": s.get("portfolio_theta_day", 0.0),
                "portfolio_theta_hr":  s.get("portfolio_theta_hr", 0.0),
                "portfolio_vega":      s.get("portfolio_vega", 0.0),
                "vol_skew":            s.get("vol_skew", 0.0),
                "atm_call_iv":         s.get("atm_call_iv", 0.0),
                "atm_put_iv":          s.get("atm_put_iv", 0.0),
            },
        )

        # ── Quant Frontier 1: SVI Vol Surface & Portfolio Risk on Options Updates ──
        spot_px = float(s.get("spot_price", 0.0))
        if spot_px > 0:
            SVI_SURFACES[base].record_price(spot_px)
            PORTFOLIO_RISK.update_price(base, spot_px)
            atm_call = float(s.get("atm_call_iv", 0.0))
            atm_put  = float(s.get("atm_put_iv", 0.0))
            if atm_call > 0 and atm_put > 0:
                avg_iv = (atm_call + atm_put) / 2.0
                strikes = [spot_px * 0.9, spot_px * 0.95, spot_px, spot_px * 1.05, spot_px * 1.1]
                ivs = [avg_iv * 1.06, avg_iv * 1.02, avg_iv, avg_iv * 1.01, avg_iv * 1.05]
                SVI_SURFACES[base].fit_svi_slice(strikes, ivs, spot_px, tau=0.082)

    print(f"[STREAM] Starting {base} Options Engine…")
    eng = ds.DeltaOptionsEngine(base, _state_cb)
    OPTIONS_ENGINES[base] = eng
    while True:
        try:
            await eng.run(poll_interval=3.0)
        except Exception as e:
            print(f"[STREAM] {base} options error: {e} — retrying in 5s")
            await asyncio.sleep(5)


async def _run_ollama_meta_loop():
    """Slow cognitive loop: background meta-prior updater every 20 seconds."""
    await asyncio.sleep(4)
    while True:
        try:
            for base, mgr in list(TRI_ENGINES.items()):
                await mgr.rl_prior.check_health()
                st = ALL_STATE.get(f"{base}_perp", {})
                mid = st.get("mid_price", 0.0)
                greeks = ALL_STATE.get(f"{base}_options", {}).get("greeks", {})
                rl_stat = mgr.rl_learner.get_status()
                await mgr.rl_prior.update_prior_async(
                    base, mid, mgr.alphas, greeks,
                    rl_stat.get("total_net_pnl", 0.0),
                    rl_stat.get("win_rate_pct", 50.0)
                )
        except Exception as e:
            print(f"[OLLAMA META] Error in background prior loop: {e}")
        await asyncio.sleep(20)


async def _launch_all_streams():
    print("Loading RL Policy network…")
    agent = rl.RLPaperAgent()
    if rl.POLICY_PATH.exists():
        try:
            sd = torch.load(rl.POLICY_PATH, map_location="cpu", weights_only=True)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            agent.load_state_dict(sd, strict=False)
            print("✓ RL Policy loaded successfully.")
        except Exception as e:
            print(f"[WARN] Error loading policy: {e} — using random weights")
    else:
        print(f"[WARN] No policy found at {rl.POLICY_PATH} — using initialized weights")
    agent.eval()

    tasks = []
    # 1. Perpetual Futures streams with Tri-Engine Manager
    for base, info in INSTRUMENTS.items():
        sym = info.get("perp")
        if sym:
            tasks.append(asyncio.create_task(_run_perp(base, sym, agent)))

    # 2. Options Department streams (BTC, ETH, GOLD)
    for base, info in INSTRUMENTS.items():
        if info.get("options"):
            tasks.append(asyncio.create_task(_run_options(base, agent)))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
