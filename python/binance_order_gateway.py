import aiohttp
import asyncio
import time
import hmac
import hashlib
import os
from typing import Dict, Any, Optional

class BinanceOrderGateway:
    def __init__(self):
        # API Keys loaded from environment for security
        self.api_key = os.environ.get('BINANCE_API_KEY')
        self.api_secret = os.environ.get('BINANCE_API_SECRET')
        self.base_url = "https://testnet.binancefuture.com"
        
        if not self.api_key or not self.api_secret:
            print("[WARNING] Binance API keys missing in environment. Paper mode only.")
            
        self.session: Optional[aiohttp.ClientSession] = None
        self.latency_stats = []

    async def connect(self):
        """Initializes the aiohttp session."""
        self.session = aiohttp.ClientSession()
        print(f"[GATEWAY] Connected to {self.base_url}")

    async def close(self):
        """Closes the aiohttp session."""
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

    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict[str, Any]:
        """
        Submits an asynchronous Limit Order to Binance Futures Testnet.
        Measures execution latency.
        """
        if not self.api_key or not self.api_secret:
            # Fallback for local testing without keys
            return {"status": "MOCK_SUCCESS", "latency_ms": 0}

        endpoint = "/fapi/v1/order"
        timestamp = int(time.time() * 1000)
        
        # Build query string
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
        
        headers = {
            "X-MBX-APIKEY": self.api_key
        }

        # Track latency
        start_time = time.perf_counter_ns()

        async with self.session.post(
            f"{self.base_url}{endpoint}?{query_string}&signature={signature}",
            headers=headers
        ) as response:
            result = await response.json()
            
            # Stop latency timer
            end_time = time.perf_counter_ns()
            latency_ms = (end_time - start_time) / 1_000_000.0
            self.latency_stats.append(latency_ms)
            
            result["latency_ms"] = latency_ms
            
            print(f"[GATEWAY] Order placed: {side} {quantity} @ {price} | Latency: {latency_ms:.2f} ms")
            return result

    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Cancels an existing order."""
        if not self.api_key:
            return {"status": "MOCK_CANCEL"}

        endpoint = "/fapi/v1/order"
        timestamp = int(time.time() * 1000)
        
        params = {
            "symbol": symbol,
            "orderId": order_id,
            "timestamp": timestamp
        }
        
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

async def main():
    gateway = BinanceOrderGateway()
    await gateway.connect()
    
    # Mock usage:
    await gateway.place_limit_order("BTCUSDT", "BUY", 0.001, 60000.0)
    await asyncio.sleep(1)
    
    await gateway.close()

if __name__ == "__main__":
    asyncio.run(main())
