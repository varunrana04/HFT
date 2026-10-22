"""
queue_priority_simulator.py — Realistic L2/L3 FIFO Queue Position & Rate-Limit Simulator
════════════════════════════════════════════════════════════════════════════════════════
In institutional high-frequency market making, executions are determined by
queue priority (FIFO price-time priority) rather than instant touches.

Core Mechanics:
1. Virtual FIFO Order Queue:
   - When our maker order of size q is posted at price P:
     Q_0 = DepthAhead(P)
   - On trade ticks (px, qty) where px == P:
     Q_t = max(0, Q_{t-1} - qty)
   - When Q_{t-1} < qty, the excess volume fills our order:
     filled_qty = min(order_qty - cum_filled, qty - Q_{t-1})
   - Price-through execution:
     If best_ask <= my_bid or best_bid >= my_ask, all preceding depth was swept -> 100% fill.

2. Cancel-and-Replace Penalty (Queue Seniority Loss):
   - Every time a maker amends or shades their quote, the old order is canceled.
   - The new order goes to the BACK of the queue:
     Q_new = CurrentDepth(P_new)
   - This quantifies the exact adverse trade-off: shading quotes improves reservation
     price but resets queue priority to zero.

3. Delta Exchange Token-Bucket Rate Limiter:
   - Max 20 requests/second with burst capacity of 10.
   - Prevents HTTP 429 penalties while measuring queue wait latency.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class VirtualOrder:
    order_id: str
    symbol: str
    side: str            # "BUY" or "SELL"
    price: float
    qty: float
    filled_qty: float = 0.0
    queue_ahead: float = 0.0
    initial_queue: float = 0.0
    created_ts_ms: int = 0
    updated_ts_ms: int = 0
    active: bool = True

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    @property
    def queue_progress_pct(self) -> float:
        """Percentage of initial queue cleared ahead of us."""
        if self.initial_queue <= 0:
            return 100.0
        cleared = self.initial_queue - self.queue_ahead
        return max(0.0, min(100.0, (cleared / self.initial_queue) * 100.0))


class TokenBucketRateLimiter:
    """
    Delta Exchange Token-Bucket Rate Limiter.
    Standard rate: 20 msgs/second. Burst capacity: 10 msgs.
    """
    def __init__(self, rate_per_sec: float = 20.0, capacity: float = 10.0):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.total_throttled: int = 0
        self.total_allowed: int = 0

    def can_send(self, tokens_needed: float = 1.0) -> bool:
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        # Refill tokens
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

        if self.tokens >= tokens_needed:
            self.tokens -= tokens_needed
            self.total_allowed += 1
            return True
        else:
            self.total_throttled += 1
            return False

    def get_throttle_stats(self) -> dict:
        return {
            "tokens_available": round(self.tokens, 2),
            "capacity": self.capacity,
            "rate_per_sec": self.rate,
            "total_allowed": self.total_allowed,
            "total_throttled": self.total_throttled,
        }


class FIFOQueueSimulator:
    """
    High-Frequency L2/L3 Order Queue Position & Microstructural Fill Tracker.
    """
    def __init__(self, symbol: str = "BTCUSD"):
        self.symbol = symbol
        self.rate_limiter = TokenBucketRateLimiter(rate_per_sec=20.0, capacity=10.0)
        self.orders: Dict[str, VirtualOrder] = {}
        self.active_bid_id: Optional[str] = None
        self.active_ask_id: Optional[str] = None
        self._order_counter: int = 0
        self.total_fills: int = 0
        self.total_volume_filled: float = 0.0
        self.total_cancels_due_to_shading: int = 0

    def submit_or_update_quote(self, side: str, price: float, qty: float, 
                                book_depth_at_price: float) -> Tuple[Optional[VirtualOrder], bool]:
        """
        Post or update a quote. If price changes, applies the cancel-and-replace queue penalty.
        Returns (order, was_rate_limited).
        """
        if not self.rate_limiter.can_send(1.0):
            # Throttled by exchange rate limits
            active_id = self.active_bid_id if side == "BUY" else self.active_ask_id
            return (self.orders.get(active_id) if active_id else None), True

        active_id = self.active_bid_id if side == "BUY" else self.active_ask_id
        current_order = self.orders.get(active_id) if active_id else None
        now_ms = int(time.time() * 1000)

        # If existing quote at EXACT same price, keep our queue priority
        if current_order and current_order.active and abs(current_order.price - price) < 1e-4:
            return current_order, False

        # If price changed, cancel existing order (Seniority Loss!)
        if current_order and current_order.active:
            current_order.active = False
            self.total_cancels_due_to_shading += 1

        # Submit new order at the BACK of the queue
        self._order_counter += 1
        order_id = f"Q-{self.symbol}-{side[:1]}-{self._order_counter}"
        
        # Depth ahead of us is the existing liquidity at that price level
        depth_ahead = max(0.0, float(book_depth_at_price))

        new_order = VirtualOrder(
            order_id=order_id,
            symbol=self.symbol,
            side=side.upper(),
            price=round(price, 2),
            qty=qty,
            filled_qty=0.0,
            queue_ahead=depth_ahead,
            initial_queue=depth_ahead,
            created_ts_ms=now_ms,
            updated_ts_ms=now_ms,
            active=True,
        )
        self.orders[order_id] = new_order
        if side.upper() == "BUY":
            self.active_bid_id = order_id
        else:
            self.active_ask_id = order_id

        return new_order, False

    def on_trade_tick(self, trade_px: float, trade_qty: float) -> List[dict]:
        """
        Process an incoming tape execution tick against our active FIFO orders.
        Returns list of fill records.
        """
        fills = []
        now_ms = int(time.time() * 1000)

        for active_id in [self.active_bid_id, self.active_ask_id]:
            if not active_id:
                continue
            order = self.orders.get(active_id)
            if not order or not order.active:
                continue

            # Check if trade matches order price
            if abs(trade_px - order.price) < 0.5:
                # 1. Deplete queue ahead
                if order.queue_ahead > 0:
                    cleared = min(order.queue_ahead, trade_qty)
                    order.queue_ahead -= cleared
                    rem_trade = trade_qty - cleared
                else:
                    rem_trade = trade_qty

                # 2. Fill order if queue is exhausted
                if order.queue_ahead <= 0 and rem_trade > 0:
                    fill_amount = min(order.remaining_qty, rem_trade)
                    if fill_amount > 0:
                        order.filled_qty += fill_amount
                        self.total_fills += 1
                        self.total_volume_filled += fill_amount

                        fill_rec = {
                            "order_id": order.order_id,
                            "symbol": self.symbol,
                            "side": order.side,
                            "price": order.price,
                            "qty": fill_amount,
                            "queue_at_fill": 0.0,
                            "time": now_ms,
                        }
                        fills.append(fill_rec)

                        if order.remaining_qty <= 0:
                            order.active = False

        return fills

    def on_book_sweep(self, best_bid: float, best_ask: float) -> List[dict]:
        """
        Check if the market swept entirely through our quotes.
        If best_ask <= my_bid -> buyer crossed our passive bid.
        If best_bid >= my_ask -> seller crossed our passive ask.
        """
        fills = []
        now_ms = int(time.time() * 1000)

        # Check Bid cross
        if self.active_bid_id:
            bid_order = self.orders.get(self.active_bid_id)
            if bid_order and bid_order.active and best_ask > 0 and best_ask <= bid_order.price:
                fill_amt = bid_order.remaining_qty
                bid_order.filled_qty += fill_amt
                bid_order.active = False
                self.total_fills += 1
                self.total_volume_filled += fill_amt
                fills.append({
                    "order_id": bid_order.order_id,
                    "symbol": self.symbol,
                    "side": "BUY",
                    "price": bid_order.price,
                    "qty": fill_amt,
                    "queue_at_fill": 0.0,
                    "reason": "PRICE_SWEEP_THROUGH",
                    "time": now_ms,
                })

        # Check Ask cross
        if self.active_ask_id:
            ask_order = self.orders.get(self.active_ask_id)
            if ask_order and ask_order.active and best_bid > 0 and best_bid >= ask_order.price:
                fill_amt = ask_order.remaining_qty
                ask_order.filled_qty += fill_amt
                ask_order.active = False
                self.total_fills += 1
                self.total_volume_filled += fill_amt
                fills.append({
                    "order_id": ask_order.order_id,
                    "symbol": self.symbol,
                    "side": "SELL",
                    "price": ask_order.price,
                    "qty": fill_amt,
                    "queue_at_fill": 0.0,
                    "reason": "PRICE_SWEEP_THROUGH",
                    "time": now_ms,
                })

        return fills

    def get_queue_telemetry(self) -> dict:
        """Get live UI telemetry of our active quotes in the queue."""
        bid_order = self.orders.get(self.active_bid_id) if self.active_bid_id else None
        ask_order = self.orders.get(self.active_ask_id) if self.active_ask_id else None

        return {
            "symbol": self.symbol,
            "bid_quote": {
                "active": bool(bid_order and bid_order.active),
                "price": bid_order.price if bid_order else 0.0,
                "qty": bid_order.qty if bid_order else 0.0,
                "queue_ahead": round(bid_order.queue_ahead, 2) if bid_order else 0.0,
                "progress_pct": round(bid_order.queue_progress_pct, 1) if bid_order else 0.0,
            },
            "ask_quote": {
                "active": bool(ask_order and ask_order.active),
                "price": ask_order.price if ask_order else 0.0,
                "qty": ask_order.qty if ask_order else 0.0,
                "queue_ahead": round(ask_order.queue_ahead, 2) if ask_order else 0.0,
                "progress_pct": round(ask_order.queue_progress_pct, 1) if ask_order else 0.0,
            },
            "rate_limiter": self.rate_limiter.get_throttle_stats(),
            "total_cancels_due_to_shading": self.total_cancels_due_to_shading,
            "total_fills": self.total_fills,
            "total_volume_filled": round(self.total_volume_filled, 4),
        }
