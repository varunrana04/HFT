import aiohttp
import asyncio
import time
import hmac
import hashlib
import json
import os
import websockets
from typing import Dict, Any, Optional, AsyncIterator

class BinanceOrderGateway:
    def __init__(self):
        # API Keys loaded from environment for security
        self.api_key = os.environ.get('BINANCE_API_KEY')
        self.api_secret = os.environ.get('BINANCE_API_SECRET')

        # Configurable: point at testnet or real via env var
        self.base_url = os.environ.get(
            'BINANCE_BASE_URL', 'https://testnet.binancefuture.com'
        )
        self.ws_base_url = os.environ.get(
            'BINANCE_WS_BASE_URL', 'wss://fstream.binance.com'
        )

        if not self.api_key or not self.api_secret:
            print("[WARNING] Binance API keys missing in environment. Paper mode only.")

        self.session: Optional[aiohttp.ClientSession] = None
        self.latency_stats: list = []
        self._listen_key: Optional[str] = None
        self._listen_key_keepalive_task: Optional[asyncio.Task] = None

    async def connect(self):
        """Initializes the aiohttp session."""
        self.session = aiohttp.ClientSession()
        print(f"[GATEWAY] Connected to {self.base_url}")

    async def close(self):
        """Closes the aiohttp session and cancels background tasks."""
        if self._listen_key_keepalive_task:
            self._listen_key_keepalive_task.cancel()
            try:
                await self._listen_key_keepalive_task
            except asyncio.CancelledError:
                pass
        if self.session:
            await self.session.close()
            print("[GATEWAY] Session closed.")

    def _generate_signature(self, query_string: str) -> str:
        """Generates HMAC SHA256 signature for Binance API."""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    # ─── Order Placement ────────────────────────────────────────────────────

    async def place_limit_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> Dict[str, Any]:
        """
        Submits an async Limit Order to Binance Futures.
        Returns the full Binance API response including orderId for tracking.
        """
        if not self.api_key or not self.api_secret:
            return {"status": "MOCK_SUCCESS", "orderId": -1, "latency_ms": 0}

        endpoint = "/fapi/v1/order"
        timestamp = int(time.time() * 1000)

        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp
        }

        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = self._generate_signature(query_string)
        headers = {"X-MBX-APIKEY": self.api_key}

        start_time = time.perf_counter_ns()

        async with self.session.post(
            f"{self.base_url}{endpoint}?{query_string}&signature={signature}",
            headers=headers
        ) as response:
            result = await response.json()

        latency_ms = (time.perf_counter_ns() - start_time) / 1_000_000.0
        self.latency_stats.append(latency_ms)
        result["latency_ms"] = latency_ms

        order_id = result.get("orderId", "?")
        status = result.get("status", "?")
        print(
            f"[GATEWAY] Order placed: {side} {quantity} @ {price} | "
            f"orderId={order_id} status={status} | Latency: {latency_ms:.2f} ms"
        )
        return result

    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Cancels an existing order."""
        if not self.api_key:
            return {"status": "MOCK_CANCEL"}

        endpoint = "/fapi/v1/order"
        timestamp = int(time.time() * 1000)

        params = {"symbol": symbol, "orderId": order_id, "timestamp": timestamp}
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = self._generate_signature(query_string)
        headers = {"X-MBX-APIKEY": self.api_key}

        start_time = time.perf_counter_ns()
        async with self.session.delete(
            f"{self.base_url}{endpoint}?{query_string}&signature={signature}",
            headers=headers
        ) as response:
            result = await response.json()

        latency_ms = (time.perf_counter_ns() - start_time) / 1_000_000.0
        print(f"[GATEWAY] Order canceled: {order_id} | Latency: {latency_ms:.2f} ms")
        return result

    # ─── Position & PnL Reconciliation ──────────────────────────────────────

    async def get_position_risk(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches the current position risk from Binance."""
        if not self.api_key or not self.api_secret:
            return {"positionAmt": 0.0, "entryPrice": 0.0, "unRealizedProfit": 0.0}

        endpoint = "/fapi/v2/positionRisk"
        timestamp = int(time.time() * 1000)
        params = {"symbol": symbol, "timestamp": timestamp}
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = self._generate_signature(query_string)
        headers = {"X-MBX-APIKEY": self.api_key}

        async with self.session.get(
            f"{self.base_url}{endpoint}?{query_string}&signature={signature}",
            headers=headers
        ) as response:
            result = await response.json()

        if isinstance(result, list):
            for pos in result:
                if pos.get("symbol") == symbol:
                    return pos
        return None

    async def get_realized_pnl(self, symbol: str, start_time_ms: int) -> float:
        """Fetches the realized PnL since the given start time."""
        if not self.api_key or not self.api_secret:
            return 0.0

        endpoint = "/fapi/v1/income"
        timestamp = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "incomeType": "REALIZED_PNL",
            "startTime": start_time_ms,
            "timestamp": timestamp
        }
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = self._generate_signature(query_string)
        headers = {"X-MBX-APIKEY": self.api_key}

        async with self.session.get(
            f"{self.base_url}{endpoint}?{query_string}&signature={signature}",
            headers=headers
        ) as response:
            result = await response.json()

        total_pnl = 0.0
        if isinstance(result, list):
            for item in result:
                total_pnl += float(item.get("income", 0.0))
        return total_pnl

    # ─── User Data Stream (Live Fill Tracking) ──────────────────────────────

    async def create_listen_key(self) -> Optional[str]:
        """
        POSTs to /fapi/v1/listenKey to obtain a Binance User Data Stream key.
        This key is valid for 60 minutes and must be kept alive with PUT every 30 mins.
        Returns the listenKey string, or None on failure.
        """
        if not self.api_key:
            return None

        endpoint = "/fapi/v1/listenKey"
        headers = {"X-MBX-APIKEY": self.api_key}

        try:
            async with self.session.post(
                f"{self.base_url}{endpoint}", headers=headers
            ) as response:
                result = await response.json()
                key = result.get("listenKey")
                if key:
                    self._listen_key = key
                    print(f"[GATEWAY] User Data Stream key obtained: {key[:8]}...")
                    # Spawn a background keepalive task
                    self._listen_key_keepalive_task = asyncio.create_task(
                        self._keepalive_listen_key_loop()
                    )
                return key
        except Exception as e:
            print(f"[GATEWAY] Failed to create listen key: {e}")
            return None

    async def _keepalive_listen_key_loop(self):
        """
        Sends a PUT to /fapi/v1/listenKey every 30 minutes to keep the stream alive.
        Binance invalidates the key after 60 minutes without a keepalive.
        """
        endpoint = "/fapi/v1/listenKey"
        headers = {"X-MBX-APIKEY": self.api_key}

        while True:
            await asyncio.sleep(1800)  # 30 minutes
            if not self._listen_key:
                break
            try:
                async with self.session.put(
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    params={"listenKey": self._listen_key}
                ) as response:
                    if response.status == 200:
                        print("[GATEWAY] Listen key keepalive sent successfully.")
                    else:
                        body = await response.text()
                        print(f"[GATEWAY][WARNING] Listen key keepalive failed ({response.status}): {body}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[GATEWAY] Listen key keepalive exception: {e}")

    async def websocket_user_stream(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator that connects to the Binance User Data Stream via WebSocket
        and yields parsed event dictionaries.

        Handles:
          - ORDER_TRADE_UPDATE: order fills, cancellations, expirations
          - ACCOUNT_UPDATE: margin/balance changes
          - LISTEN_KEY_EXPIRED: triggers key renewal

        Usage:
            async for event in gateway.websocket_user_stream():
                if event.get("e") == "ORDER_TRADE_UPDATE":
                    ...
        """
        if not self._listen_key:
            print("[GATEWAY] No listen key — call create_listen_key() first.")
            return

        backoff = 1.0
        while True:
            url = f"{self.ws_base_url}/ws/{self._listen_key}"
            try:
                print(f"[GATEWAY] Connecting to User Data Stream...")
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10
                ) as ws:
                    print("[GATEWAY] User Data Stream connected.")
                    backoff = 1.0  # reset on success
                    async for raw in ws:
                        try:
                            event = json.loads(raw)
                            # If the server signals key expiry, renew it
                            if event.get("e") == "listenKeyExpired":
                                print("[GATEWAY] Listen key expired — renewing...")
                                await self.create_listen_key()
                                break  # reconnect loop with new key
                            yield event
                        except json.JSONDecodeError:
                            pass
            except asyncio.CancelledError:
                return
            except Exception as e:
                print(
                    f"[GATEWAY] User Data Stream disconnected: {e}. "
                    f"Reconnecting in {backoff:.1f}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)  # exponential backoff, cap at 60s

    def avg_latency_ms(self) -> float:
        """Returns the rolling average order-submit latency in milliseconds."""
        if not self.latency_stats:
            return 0.0
        return sum(self.latency_stats[-100:]) / len(self.latency_stats[-100:])


async def main():
    gateway = BinanceOrderGateway()
    await gateway.connect()
    await gateway.place_limit_order("BTCUSDT", "BUY", 0.001, 60000.0)
    await asyncio.sleep(1)
    await gateway.close()


if __name__ == "__main__":
    asyncio.run(main())
