"""
binance_lead_lag.py — Low-Latency Cross-Venue Lead-Lag Signal Engine
════════════════════════════════════════════════════════════════════════════════
Connects to Binance Futures public bookTicker WebSocket stream (<10ms broadcast).
Binance price discovery leads secondary venues (Delta Exchange) by 50–150ms.
Computes real-time microprice divergence to detect front-running opportunities
and protect maker quotes from adverse selection.
"""

import asyncio
import json
import math
import time
from typing import Dict, Optional, Tuple

import aiohttp

BINANCE_WS_URL = "wss://fstream.binance.com/ws/btcusdt@bookTicker/ethusdt@bookTicker/solusdt@bookTicker"

# Mapping from internal base to Binance symbol
SYMBOL_MAP = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "SOL":  "SOLUSDT",
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "SOLUSD": "SOLUSDT",
}


class BinanceLeadLagEngine:
    """
    Asynchronous cross-venue microprice divergence tracker.
    Provides sub-millisecond divergence lookups against Delta Exchange orderbooks.
    """
    def __init__(self):
        self._running: bool = False
        self._connected: bool = False
        self._last_msg_ts: float = 0.0

        # symbol -> {bid, ask, bid_sz, ask_sz, mid, micro, ts_ms}
        self.tickers: Dict[str, dict] = {}
        self.divergences: Dict[str, dict] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected and (time.time() - self._last_msg_ts < 10.0)

    async def start(self):
        """Start async WebSocket ingestion loop with robust backoff."""
        self._running = True
        backoff = 2
        while self._running:
            try:
                print(f"[BINANCE LEAD-LAG] Connecting to {BINANCE_WS_URL}…")
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.ws_connect(BINANCE_WS_URL, heartbeat=15) as ws:
                        self._connected = True
                        backoff = 2
                        print("[BINANCE LEAD-LAG] WebSocket connected! Streaming live bookTickers.")
                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self._on_message(msg.data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[BINANCE LEAD-LAG] Disconnected ({e}). Reconnecting in {backoff}s…")
                self._connected = False
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

        self._connected = False

    def stop(self):
        self._running = False

    def _on_message(self, raw_text: str):
        try:
            d = json.loads(raw_text)
            if "data" in d:
                d = d["data"]
            sym = d.get("s")  # e.g. "BTCUSDT"
            b_px = float(d.get("b", 0))
            a_px = float(d.get("a", 0))
            b_sz = float(d.get("B", 0))
            a_sz = float(d.get("A", 0))

            if b_px <= 0 or a_px <= 0 or (b_sz + a_sz) <= 0:
                return

            mid = (b_px + a_px) / 2.0
            # Stoikov volume-weighted microprice
            micro = (b_px * a_sz + a_px * b_sz) / (b_sz + a_sz)

            self._last_msg_ts = time.time()
            self.tickers[sym] = {
                "bid": b_px,
                "ask": a_px,
                "bid_sz": b_sz,
                "ask_sz": a_sz,
                "mid": mid,
                "micro": micro,
                "ts_ms": int(time.time() * 1000),
            }
        except Exception:
            pass

    def get_divergence(self, instrument: str, delta_mid: float) -> Tuple[float, str, dict]:
        """
        Calculate instantaneous lead-lag microprice divergence against Delta Exchange.
        Returns:
            divergence_bps: (Binance Micro - Delta Mid) / Delta Mid * 10,000
            signal: "BULL_LEAD", "BEAR_LEAD", or "NEUTRAL"
            meta: dictionary of Binance prices and latency
        """
        b_sym = SYMBOL_MAP.get(instrument.upper(), instrument.upper())
        b_data = self.tickers.get(b_sym)

        if not b_data:
            return 0.0, "NEUTRAL", {"status": "CONNECTING_BINANCE", "binance_sym": b_sym}

        b_micro = b_data["micro"]
        b_mid = b_data["mid"]
        d_mid = delta_mid if delta_mid > 0 else b_mid
        div_bps = ((b_micro - d_mid) / d_mid) * 10000.0 if d_mid > 0 else 0.0

        # Lead-lag thresholds:
        # > +2.0 bps: Binance is aggressively breaking upward (Delta will follow)
        # < -2.0 bps: Binance is violently cascading downward
        if div_bps >= 2.0:
            signal = "BULL_LEAD"
        elif div_bps <= -2.0:
            signal = "BEAR_LEAD"
        else:
            signal = "NEUTRAL"

        meta = {
            "binance_sym": b_sym,
            "binance_mid": round(b_mid, 2),
            "binance_micro": round(b_micro, 2),
            "delta_mid": round(delta_mid, 2) if delta_mid > 0 else round(b_mid, 2),
            "div_bps": round(div_bps, 2),
            "signal": signal,
            "latency_ms": int((time.time() * 1000) - b_data["ts_ms"]),
            "status": "LIVE",
        }
        self.divergences[instrument] = meta
        return div_bps, signal, meta

    def get_all_divergences(self) -> Dict[str, dict]:
        return dict(self.divergences)
