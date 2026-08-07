#!/usr/bin/env python3
"""
tv_webhook.py — TradingView alert webhook server.

Receives TradingView alert webhooks, parses the signal, validates
through the risk engine, and forwards to a paper trading API
(Binance Testnet or Alpaca Paper Trading).

Requirements:
    pip install fastapi uvicorn httpx

Usage:
    python tv_webhook.py --port 8080
    python tv_webhook.py --port 8080 --broker alpaca

TradingView Alert Setup:
    1. Create an alert on your chart
    2. Set Webhook URL to: http://your-server:8080/webhook
    3. Set Alert Message (JSON):
        {
            "symbol": "{{ticker}}",
            "action": "{{strategy.order.action}}",
            "price": "{{close}}",
            "time": "{{time}}"
        }
"""

import sys
import os
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

# ─── Logging ─────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('tv_webhook')

# ─── FastAPI Import ──────────────────────────────────────────

try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.warning("FastAPI not installed. Install with: pip install fastapi uvicorn")

# ─── Risk Validator ──────────────────────────────────────────

class SimpleRiskValidator:
    """
    Lightweight risk checks before forwarding orders.
    Mirrors the C++ RiskManager logic in Python.
    """
    
    def __init__(self, max_position: float = 10.0,
                 max_daily_loss: float = 500.0,
                 max_orders_per_minute: int = 10,
                 cooldown_sec: float = 5.0):
        self.max_position = max_position
        self.max_daily_loss = max_daily_loss
        self.max_orders_per_minute = max_orders_per_minute
        self.cooldown_sec = cooldown_sec
        
        # State
        self.position = 0.0
        self.daily_pnl = 0.0
        self.recent_orders = []  # Timestamps
        self.last_order_time = 0.0
    
    def check(self, action: str, quantity: float) -> tuple:
        """
        Validate a proposed order.
        Returns (is_valid, reason).
        """
        now = time.time()
        
        # Cooldown check
        if now - self.last_order_time < self.cooldown_sec:
            return False, f"Cooldown active ({self.cooldown_sec}s)"
        
        # Rate limit check
        self.recent_orders = [t for t in self.recent_orders if now - t < 60]
        if len(self.recent_orders) >= self.max_orders_per_minute:
            return False, f"Rate limit ({self.max_orders_per_minute}/min)"
        
        # Position limit check
        new_position = self.position
        if action.upper() == 'BUY':
            new_position += quantity
        elif action.upper() == 'SELL':
            new_position -= quantity
        
        if abs(new_position) > self.max_position:
            return False, f"Position limit (max: {self.max_position})"
        
        # Daily loss check
        if self.daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss limit (${self.max_daily_loss})"
        
        return True, "OK"
    
    def record_fill(self, action: str, quantity: float, price: float) -> None:
        """Update state after a fill."""
        if action.upper() == 'BUY':
            self.position += quantity
        elif action.upper() == 'SELL':
            self.position -= quantity
        
        self.last_order_time = time.time()
        self.recent_orders.append(time.time())
    
    def reset_daily(self) -> None:
        """Reset daily counters."""
        self.daily_pnl = 0.0


# ─── Broker Adapters ─────────────────────────────────────────

class PaperBroker:
    """Base class for paper trading brokers."""
    
    def execute(self, symbol: str, action: str,
                quantity: float, price: float) -> Dict[str, Any]:
        """Execute a trade. Returns order result dict."""
        raise NotImplementedError


class SimulatedBroker(PaperBroker):
    """In-memory simulated broker for testing without any API."""
    
    def __init__(self):
        self.orders = []
        self.order_id = 0
    
    def execute(self, symbol: str, action: str,
                quantity: float, price: float) -> Dict[str, Any]:
        self.order_id += 1
        result = {
            'order_id': self.order_id,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'status': 'FILLED',
            'timestamp': datetime.now().isoformat(),
            'broker': 'simulated'
        }
        self.orders.append(result)
        logger.info(f"[SIM] {action} {quantity} {symbol} @ {price}")
        return result


class AlpacaBroker(PaperBroker):
    """Alpaca Paper Trading API adapter."""
    
    def __init__(self, api_key: str = '', api_secret: str = '',
                 base_url: str = 'https://paper-api.alpaca.markets'):
        self.api_key = api_key or os.getenv('ALPACA_API_KEY', '')
        self.api_secret = api_secret or os.getenv('ALPACA_API_SECRET', '')
        self.base_url = base_url
    
    def execute(self, symbol: str, action: str,
                quantity: float, price: float) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError:
            logger.error("httpx not installed: pip install httpx")
            return {'status': 'ERROR', 'reason': 'httpx not installed'}
        
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        }
        
        order_data = {
            'symbol': symbol,
            'qty': str(quantity),
            'side': action.lower(),
            'type': 'market',
            'time_in_force': 'ioc'
        }
        
        try:
            resp = httpx.post(
                f'{self.base_url}/v2/orders',
                json=order_data,
                headers=headers,
                timeout=5.0
            )
            resp.raise_for_status()
            result = resp.json()
            result['broker'] = 'alpaca'
            logger.info(f"[ALPACA] {action} {quantity} {symbol} → {result.get('status')}")
            return result
        except Exception as e:
            logger.error(f"Alpaca order failed: {e}")
            return {'status': 'ERROR', 'reason': str(e)}


# ─── Webhook Server ──────────────────────────────────────────

def create_app(broker: PaperBroker,
               risk: SimpleRiskValidator,
               default_qty: float = 0.01) -> 'FastAPI':
    """Create the FastAPI webhook application."""
    
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI not installed")
    
    app = FastAPI(
        title="HFT Engine — TradingView Webhook",
        description="Receives TradingView alerts and executes paper trades",
        version="1.0.0"
    )
    
    trade_log = []
    
    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "trades": len(trade_log),
            "position": risk.position,
            "uptime_s": time.time() - app.state.start_time
                        if hasattr(app.state, 'start_time') else 0
        }
    
    @app.post("/webhook")
    async def webhook(request: Request):
        """
        TradingView webhook endpoint.
        
        Expected JSON:
        {
            "symbol": "BTCUSDT",
            "action": "buy" | "sell",
            "price": 50000.0,
            "time": "2026-01-01T00:00:00Z",
            "quantity": 0.01  (optional)
        }
        """
        try:
            # Parse payload
            body = await request.body()
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                # Try form-encoded or plain text
                text = body.decode('utf-8').strip()
                payload = {"raw": text}
            
            symbol = payload.get('symbol', 'UNKNOWN')
            action = payload.get('action', '').upper()
            price = float(payload.get('price', 0))
            quantity = float(payload.get('quantity', default_qty))
            
            if action not in ('BUY', 'SELL'):
                raise HTTPException(400, f"Invalid action: {action}")
            
            if price <= 0:
                raise HTTPException(400, f"Invalid price: {price}")
            
            logger.info(f"Webhook: {action} {quantity} {symbol} @ {price}")
            
            # Risk check
            valid, reason = risk.check(action, quantity)
            if not valid:
                logger.warning(f"Risk rejected: {reason}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "REJECTED",
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # Execute
            result = broker.execute(symbol, action, quantity, price)
            risk.record_fill(action, quantity, price)
            
            # Log
            trade_entry = {
                **result,
                'webhook_time': datetime.now().isoformat(),
                'risk_position': risk.position,
            }
            trade_log.append(trade_entry)
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "EXECUTED",
                    "result": result,
                    "position": risk.position,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return JSONResponse(
                status_code=500,
                content={"status": "ERROR", "reason": str(e)}
            )
    
    @app.get("/trades")
    async def get_trades():
        """Get all executed trades."""
        return {"trades": trade_log, "count": len(trade_log)}
    
    @app.post("/reset")
    async def reset_daily():
        """Reset daily risk counters."""
        risk.reset_daily()
        return {"status": "ok", "message": "Daily counters reset"}
    
    @app.on_event("startup")
    async def startup():
        app.state.start_time = time.time()
        logger.info("TradingView webhook server started")
    
    return app


# ─── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HFT Engine — TradingView Webhook Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Brokers:
    sim     — In-memory simulation (default)
    alpaca  — Alpaca Paper Trading API (requires ALPACA_API_KEY env vars)

TradingView Alert JSON format:
    {"symbol":"{{ticker}}","action":"{{strategy.order.action}}","price":"{{close}}"}
        """
    )
    parser.add_argument('--port', type=int, default=8080,
                        help='Server port (default: 8080)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host (default: 0.0.0.0)')
    parser.add_argument('--broker', type=str, default='sim',
                        choices=['sim', 'alpaca'],
                        help='Broker backend (default: sim)')
    parser.add_argument('--qty', type=float, default=0.01,
                        help='Default order quantity (default: 0.01)')
    parser.add_argument('--max-position', type=float, default=10.0,
                        help='Max position size (default: 10.0)')
    parser.add_argument('--max-daily-loss', type=float, default=500.0,
                        help='Max daily loss in $ (default: 500)')
    
    args = parser.parse_args()
    
    if not HAS_FASTAPI:
        logger.error("FastAPI required: pip install fastapi uvicorn")
        sys.exit(1)
    
    # ── Initialize Broker ──
    if args.broker == 'alpaca':
        broker = AlpacaBroker()
        if not broker.api_key:
            logger.warning("ALPACA_API_KEY not set — orders will fail")
    else:
        broker = SimulatedBroker()
    
    # ── Initialize Risk ──
    risk = SimpleRiskValidator(
        max_position=args.max_position,
        max_daily_loss=args.max_daily_loss
    )
    
    # ── Create and Run App ──
    app = create_app(broker, risk, default_qty=args.qty)
    
    logger.info(f"Starting webhook server on {args.host}:{args.port}")
    logger.info(f"Broker: {args.broker}")
    logger.info(f"Webhook URL: http://{args.host}:{args.port}/webhook")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == '__main__':
    main()
