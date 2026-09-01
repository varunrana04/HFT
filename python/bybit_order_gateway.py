import aiohttp
import asyncio
import time
import hmac
import hashlib
import json
import os
import websockets
import uuid
import urllib.parse
from typing import Dict, Any, Optional, AsyncIterator

class BybitOrderGateway:
    def __init__(self):
        # API Keys loaded from environment for security
        self.api_key = os.environ.get('BYBIT_API_KEY')
        self.api_secret = os.environ.get('BYBIT_API_SECRET')

        # Configurable: point at testnet or real via env var
        self.base_url = os.environ.get(
            'BYBIT_BASE_URL', 'https://api-testnet.bybit.com'
        )
        self.ws_base_url = os.environ.get(
            'BYBIT_WS_BASE_URL', 'wss://stream-testnet.bybit.com/v5/private'
        )

        if not self.api_key or not self.api_secret:
            print("[WARNING] Bybit API keys missing in environment. Paper mode only.")

        self.session: Optional[aiohttp.ClientSession] = None
        self.latency_stats: list = []
        self.recv_window = "5000"

    async def connect(self):
        """Initializes the aiohttp session."""
        self.session = aiohttp.ClientSession()
        print(f"[GATEWAY] Connected to {self.base_url}")

    async def close(self):
        """Closes the aiohttp session and cancels background tasks."""
        if self.session:
            await self.session.close()
            print("[GATEWAY] Session closed.")

    def _generate_signature(self, timestamp: str, payload: str) -> str:
        """Generates HMAC SHA256 signature for Bybit V5 API."""
        param_str = timestamp + self.api_key + self.recv_window + payload
        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_headers(self, timestamp: str, signature: str) -> dict:
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": self.recv_window,
            "Content-Type": "application/json"
        }

    # ─── Order Placement ────────────────────────────────────────────────────

    async def place_limit_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> Dict[str, Any]:
        """
        Submits an async Limit Order to Bybit V5 API.
        Returns the full API response including orderId for tracking.
        """
        if not self.api_key or not self.api_secret:
            return {"status": "MOCK_SUCCESS", "orderId": str(uuid.uuid4()), "latency_ms": 0}

        endpoint = "/v5/order/create"
        timestamp = str(int(time.time() * 1000))
        
        # Format price and quantity to avoid scientific notation
        price_str = f"{price:.2f}" if isinstance(price, float) else str(price)
        qty_str = f"{quantity:.4f}" if isinstance(quantity, float) else str(quantity)

        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side.capitalize(),
            "orderType": "Limit",
            "qty": qty_str,
            "price": price_str,
            "timeInForce": "GTC",
            "positionIdx": 0 # One-way mode
        }

        payload = json.dumps(params)
        signature = self._generate_signature(timestamp, payload)
        headers = self._get_headers(timestamp, signature)

        start_time = time.perf_counter_ns()

        async with self.session.post(
            f"{self.base_url}{endpoint}",
            headers=headers,
            data=payload
        ) as response:
            result = await response.json()

        latency_ms = (time.perf_counter_ns() - start_time) / 1_000_000.0
        self.latency_stats.append(latency_ms)
        
        ret_code = result.get("retCode", -1)
        ret_msg = result.get("retMsg", "")
        
        if ret_code == 0:
            order_id = result.get("result", {}).get("orderId", "?")
            return {"status": "NEW", "orderId": order_id, "latency_ms": latency_ms, "raw": result}
        else:
            print(f"[BybitRest] Order submission failed! Code: {ret_code} Msg: {ret_msg}")
            return {"status": "REJECTED", "orderId": -1, "latency_ms": latency_ms, "raw": result}

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancels an open order on Bybit."""
        if not self.api_key or not self.api_secret:
            return {"status": "MOCK_CANCELED"}

        endpoint = "/v5/order/cancel"
        timestamp = str(int(time.time() * 1000))
        
        params = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id
        }

        payload = json.dumps(params)
        signature = self._generate_signature(timestamp, payload)
        headers = self._get_headers(timestamp, signature)

        async with self.session.post(
            f"{self.base_url}{endpoint}",
            headers=headers,
            data=payload
        ) as response:
            return await response.json()

    # ─── Risk & Position Retrieval ──────────────────────────────────────────

    async def get_position_risk(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieves position risk for a specific symbol.
        Returns a dict containing 'size', 'entryPrice', 'unrealisedPnl'.
        """
        if not self.api_key or not self.api_secret:
            return {"size": 0.0, "entryPrice": 0.0, "unrealisedPnl": 0.0}

        endpoint = "/v5/position/list"
        timestamp = str(int(time.time() * 1000))
        
        params = {
            "category": "linear",
            "symbol": symbol
        }
        
        query_string = urllib.parse.urlencode(params)
        signature = self._generate_signature(timestamp, query_string)
        headers = self._get_headers(timestamp, signature)

        async with self.session.get(
            f"{self.base_url}{endpoint}?{query_string}",
            headers=headers
        ) as response:
            result = await response.json()
            
        ret_code = result.get("retCode", -1)
        if ret_code == 0:
            pos_list = result.get("result", {}).get("list", [])
            for p in pos_list:
                if p.get("symbol") == symbol:
                    # Bybit returns side, we make size negative if side is Sell
                    size = float(p.get("size", 0))
                    if p.get("side") == "Sell":
                        size = -size
                    return {
                        "size": size,
                        "entryPrice": float(p.get("avgPrice", 0)),
                        "unrealisedPnl": float(p.get("unrealisedPnl", 0))
                    }
        return {"size": 0.0, "entryPrice": 0.0, "unrealisedPnl": 0.0}

    async def get_realized_pnl(self, symbol: str, start_time: int) -> float:
        """
        Sums up realized PnL from closed positions after `start_time` (ms).
        """
        if not self.api_key or not self.api_secret:
            return 0.0

        endpoint = "/v5/position/closed-pnl"
        timestamp = str(int(time.time() * 1000))
        
        params = {
            "category": "linear",
            "symbol": symbol,
            "startTime": str(start_time),
            "limit": "100"
        }
        
        query_string = urllib.parse.urlencode(params)
        signature = self._generate_signature(timestamp, query_string)
        headers = self._get_headers(timestamp, signature)

        async with self.session.get(
            f"{self.base_url}{endpoint}?{query_string}",
            headers=headers
        ) as response:
            result = await response.json()

        total_pnl = 0.0
        ret_code = result.get("retCode", -1)
        if ret_code == 0:
            for pnl_item in result.get("result", {}).get("list", []):
                total_pnl += float(pnl_item.get("closedPnl", 0))

        return total_pnl

    # ─── User Data Stream (WebSocket) ───────────────────────────────────────

    async def stream_user_data(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Connects to Bybit V5 private websocket and yields execution reports.
        Handles authentication via the WS protocol automatically.
        """
        if not self.api_key or not self.api_secret:
            return

        expires = int(time.time() * 1000) + 10000
        signature = self._generate_signature(str(expires), "")

        async with websockets.connect(self.ws_base_url) as ws:
            # Authenticate
            auth_msg = {
                "op": "auth",
                "args": [self.api_key, expires, signature]
            }
            await ws.send(json.dumps(auth_msg))
            
            # Wait for auth response
            auth_resp = await ws.recv()
            print(f"[WS] Auth response: {auth_resp}")

            # Subscribe to execution reports and positions
            sub_msg = {
                "op": "subscribe",
                "args": ["execution", "position"]
            }
            await ws.send(json.dumps(sub_msg))

            while True:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    
                    # Yield raw execution reports to be parsed by the consumer
                    if data.get("topic") == "execution":
                        for exc in data.get("data", []):
                            yield exc

                except websockets.exceptions.ConnectionClosed:
                    print("[WS] Bybit private stream connection closed.")
                    break
                except Exception as e:
                    print(f"[WS] Error in Bybit stream: {e}")
                    break
