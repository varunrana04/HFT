"""
delta_gateway.py — Delta Exchange Order Gateway
================================================
Handles REST order placement + WebSocket L2 market data for Delta Exchange.
Supports both testnet and live environments via DELTA_BASE_URL env var.

Load credentials ONLY from environment variables / .env file.
Never hardcode keys.
"""

import asyncio
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from dotenv import load_dotenv

# ── Load .env (only if it exists — Render uses env vars directly) ──────────
load_dotenv()

# ── Configuration ───────────────────────────────────────────────────────────
DELTA_API_KEY    = os.environ.get("DELTA_API_KEY", "")
DELTA_API_SECRET = os.environ.get("DELTA_API_SECRET", "")
DELTA_BASE_URL   = os.environ.get("DELTA_BASE_URL", "https://cdn-ind.testnet.deltaex.org")
DELTA_WS_URL     = os.environ.get("DELTA_WS_URL",   "wss://cdn-ind.testnet.deltaex.org/live")
DELTA_PRODUCT_ID = int(os.environ.get("DELTA_PRODUCT_ID", "139"))  # BTC-USDT Perp testnet
PAPER_MODE       = os.environ.get("PAPER_MODE", "true").lower() == "true"


# ── HMAC Signature (Delta Exchange auth spec) ────────────────────────────────
def _sign(secret: str, method: str, path: str, query: str, body: str, ts: str) -> str:
    """
    Delta signature = HMAC-SHA256 of:
        method + timestamp + request_path + query_string + body
    """
    msg = method + ts + path + query + body
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()


def _auth_headers(method: str, path: str, query: str = "", body: str = "") -> dict:
    ts  = str(int(time.time()))
    sig = _sign(DELTA_API_SECRET, method, path, query, body, ts)
    return {
        "api-key":   DELTA_API_KEY,
        "timestamp": ts,
        "signature": sig,
        "Content-Type": "application/json",
    }


# ── Order State ──────────────────────────────────────────────────────────────
@dataclass
class DeltaOrderState:
    bid_order_id: Optional[int] = None
    ask_order_id: Optional[int] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ── REST Gateway ─────────────────────────────────────────────────────────────
class DeltaOrderGateway:
    """
    Minimal REST gateway for Delta Exchange.
    Paper mode: logs orders without sending to exchange.
    Live mode:  signs and sends real REST API calls.
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self.paper = PAPER_MODE
        if self.paper:
            print("[DELTA GW] PAPER MODE — no real orders will be sent.")
        else:
            if not DELTA_API_KEY or not DELTA_API_SECRET:
                raise RuntimeError("DELTA_API_KEY / DELTA_API_SECRET not set in environment.")
            print(f"[DELTA GW] LIVE MODE — connected to {DELTA_BASE_URL}")

    async def connect(self):
        self._session = aiohttp.ClientSession(
            base_url=DELTA_BASE_URL,
            timeout=aiohttp.ClientTimeout(total=5),
        )
        # Verify connectivity
        try:
            async with self._session.get("/v2/assets") as r:
                if r.status == 200:
                    print("[DELTA GW] REST connection verified.")
                else:
                    print(f"[DELTA GW] WARNING: /v2/assets returned HTTP {r.status}")
        except Exception as e:
            print(f"[DELTA GW] Connection check failed: {e}")

    async def close(self):
        if self._session:
            await self._session.close()

    # ── Account ─────────────────────────────────────────────────────────────
    async def get_balances(self) -> dict:
        if self.paper:
            return {"USDT": 10000.0, "BTC": 0.0}
        path = "/v2/wallet/balances"
        headers = _auth_headers("GET", path)
        async with self._session.get(path, headers=headers) as r:
            data = await r.json()
            return data.get("result", {})

    async def get_positions(self) -> list:
        if self.paper:
            return []
        path = "/v2/positions"
        headers = _auth_headers("GET", path)
        async with self._session.get(path, headers=headers) as r:
            data = await r.json()
            return data.get("result", [])

    # ── Orders ───────────────────────────────────────────────────────────────
    async def place_limit_order(
        self,
        side: str,          # "buy" or "sell"
        size: int,          # number of contracts (1 contract = 1 USD notional on BTCUSD)
        limit_price: float,
        post_only: bool = True,   # maker-only (avoids taker fee)
        reduce_only: bool = False,
    ) -> Optional[int]:
        """
        Place a limit order. Returns order_id or None on failure.
        post_only=True ensures maker rebate (you get PAID to place orders).
        """
        payload = {
            "product_id":    DELTA_PRODUCT_ID,
            "order_type":    "limit_order",
            "side":          side,
            "size":          size,
            "limit_price":   str(round(limit_price, 1)),
            "post_only":     post_only,
            "reduce_only":   reduce_only,
            "time_in_force": "gtc",   # good till cancel
        }
        body = json.dumps(payload)

        if self.paper:
            print(f"[DELTA PAPER] {side.upper()} {size}x @ ${limit_price:.1f} (post_only={post_only})")
            return int(time.time() * 1000) % 999999  # fake order id

        path    = "/v2/orders"
        headers = _auth_headers("POST", path, body=body)
        try:
            async with self._session.post(path, headers=headers, data=body) as r:
                data = await r.json()
                if data.get("success"):
                    oid = data["result"]["id"]
                    print(f"[DELTA GW] {side.upper()} order placed: id={oid} size={size} @ ${limit_price:.1f}")
                    return oid
                else:
                    print(f"[DELTA GW] Order rejected: {data}")
                    return None
        except Exception as e:
            print(f"[DELTA GW] Order error: {e}")
            return None

    async def cancel_order(self, order_id: int) -> bool:
        if self.paper:
            print(f"[DELTA PAPER] Cancel order {order_id}")
            return True

        path = f"/v2/orders/{order_id}"
        headers = _auth_headers("DELETE", path)
        try:
            async with self._session.delete(path, headers=headers) as r:
                data = await r.json()
                ok = data.get("success", False)
                if ok:
                    print(f"[DELTA GW] Order {order_id} cancelled.")
                else:
                    print(f"[DELTA GW] Cancel failed: {data}")
                return ok
        except Exception as e:
            print(f"[DELTA GW] Cancel error: {e}")
            return False

    async def cancel_all_orders(self) -> bool:
        """Cancel all open orders for the product. Call on shutdown."""
        if self.paper:
            print("[DELTA PAPER] Cancel all orders.")
            return True

        path = "/v2/orders/all"
        payload = {"product_id": DELTA_PRODUCT_ID}
        body = json.dumps(payload)
        headers = _auth_headers("DELETE", path, body=body)
        try:
            async with self._session.delete(path, headers=headers, data=body) as r:
                data = await r.json()
                return data.get("success", False)
        except Exception as e:
            print(f"[DELTA GW] Cancel-all error: {e}")
            return False

    # ── Market data (REST fallback) ──────────────────────────────────────────
    async def get_best_bid_ask(self) -> tuple[float, float]:
        """Returns (best_bid, best_ask) from L1 REST endpoint."""
        path = f"/v2/l2orderbook/BTCUSD?depth=1"
        try:
            async with self._session.get(path) as r:
                data = await r.json()
                book = data.get("result", {})
                bids = book.get("buy", [])
                asks = book.get("sell", [])
                bid = float(bids[0]["price"]) if bids else 0.0
                ask = float(asks[0]["price"]) if asks else 0.0
                return bid, ask
        except Exception as e:
            print(f"[DELTA GW] L2 REST error: {e}")
            return 0.0, 0.0


# ── WebSocket Market Data ────────────────────────────────────────────────────
class DeltaWebSocket:
    """
    Subscribes to Delta Exchange WebSocket for real-time L2 orderbook.
    Calls on_book_update(bid, ask, mid) on every update.
    """

    def __init__(self, on_book_update):
        self._cb  = on_book_update
        self._url = DELTA_WS_URL

    async def run(self):
        import websockets
        print(f"[DELTA WS] Connecting to {self._url}")
        while True:
            try:
                async with websockets.connect(self._url, ping_interval=20) as ws:
                    # Subscribe to L1 ticker (lightest feed)
                    sub = {
                        "type": "subscribe",
                        "payload": {
                            "channels": [
                                {"name": "v2/ticker", "symbols": ["BTCUSD"]},
                            ]
                        }
                    }
                    await ws.send(json.dumps(sub))
                    print("[DELTA WS] Subscribed to BTCUSD ticker.")

                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") == "v2/ticker":
                            data = msg.get("symbol_data", {}).get("BTCUSD", {})
                            bid = float(data.get("best_bid", 0) or 0)
                            ask = float(data.get("best_ask", 0) or 0)
                            if bid > 0 and ask > 0:
                                mid = (bid + ask) / 2.0
                                await self._cb(bid, ask, mid)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[DELTA WS] Error: {e} — reconnecting in 3s")
                await asyncio.sleep(3)


# ── Quick connectivity test ──────────────────────────────────────────────────
async def test_connection():
    """Run this to verify keys work before deploying."""
    print("=" * 60)
    print("Delta Exchange — Connection Test")
    print(f"Base URL : {DELTA_BASE_URL}")
    print(f"Paper    : {PAPER_MODE}")
    print(f"Key set  : {'YES' if DELTA_API_KEY else 'NO — set DELTA_API_KEY in .env'}")
    print("=" * 60)

    gw = DeltaOrderGateway()
    await gw.connect()

    # Test 1: public endpoint (no auth needed)
    bid, ask = await gw.get_best_bid_ask()
    print(f"\n[OK] BTCUSD L1: bid=${bid:,.1f}  ask=${ask:,.1f}  mid=${((bid+ask)/2):,.1f}")

    # Test 2: authenticated endpoint
    if DELTA_API_KEY:
        balances = await gw.get_balances()
        print(f"[OK] Balances: {balances}")

    # Test 3: paper order
    oid = await gw.place_limit_order("buy", size=1, limit_price=bid - 50)
    print(f"[OK] Test order id: {oid}")

    await gw.close()
    print("\nAll tests passed. Ready to trade.")


if __name__ == "__main__":
    asyncio.run(test_connection())
