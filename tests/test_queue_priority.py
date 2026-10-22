"""
test_queue_priority.py — Unit Tests for FIFO Queue Position & Rate-Limit Simulator
"""

import time
import unittest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "python"))

from queue_priority_simulator import FIFOQueueSimulator, TokenBucketRateLimiter


class TestFIFOQueueSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = FIFOQueueSimulator("BTCUSD")

    def test_queue_depletion_and_fill(self):
        """Test that an order only fills after the preceding queue is exhausted."""
        # Submit bid order at 84000.0 for 2 contracts with 5 contracts ahead in the book
        order, throttled = self.sim.submit_or_update_quote(
            side="BUY", price=84000.0, qty=2.0, book_depth_at_price=5.0
        )
        self.assertFalse(throttled)
        self.assertEqual(order.queue_ahead, 5.0)
        self.assertEqual(order.filled_qty, 0.0)

        # 1. Trade of 3 contracts at 84000.0 -> Queue drops from 5 to 2, 0 fills
        fills = self.sim.on_trade_tick(trade_px=84000.0, trade_qty=3.0)
        self.assertEqual(len(fills), 0)
        self.assertEqual(order.queue_ahead, 2.0)
        self.assertEqual(order.queue_progress_pct, 60.0)

        # 2. Trade of 3 contracts at 84000.0 -> 2 clears queue ahead, 1 fills our order!
        fills = self.sim.on_trade_tick(trade_px=84000.0, trade_qty=3.0)
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["qty"], 1.0)
        self.assertEqual(order.filled_qty, 1.0)
        self.assertEqual(order.queue_ahead, 0.0)
        self.assertTrue(order.active)

        # 3. Trade of 2 contracts -> Remaining 1 contract filled, order completed!
        fills = self.sim.on_trade_tick(trade_px=84000.0, trade_qty=2.0)
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["qty"], 1.0)
        self.assertEqual(order.filled_qty, 2.0)
        self.assertFalse(order.active)

    def test_cancel_and_replace_queue_seniority_loss(self):
        """Amending a quote price must cancel the old order and reset queue seniority to the back."""
        # Quote at 84000 with 10 contracts ahead
        order1, _ = self.sim.submit_or_update_quote("BUY", 84000.0, 1.0, 10.0)
        self.assertEqual(order1.queue_ahead, 10.0)

        # Trade clears 8 contracts ahead -> 2 ahead
        self.sim.on_trade_tick(84000.0, 8.0)
        self.assertEqual(order1.queue_ahead, 2.0)

        # Algorithm re-shades quote to 84001.0 where there are 15 contracts ahead
        order2, _ = self.sim.submit_or_update_quote("BUY", 84001.0, 1.0, 15.0)
        
        # Order 1 must be inactive, order 2 is at back of queue (15 contracts ahead)
        self.assertFalse(order1.active)
        self.assertTrue(order2.active)
        self.assertEqual(order2.queue_ahead, 15.0)
        self.assertEqual(self.sim.total_cancels_due_to_shading, 1)

    def test_price_sweep_through(self):
        """When the market sweeps entirely through our quote, execute 100% immediately."""
        self.sim.submit_or_update_quote("BUY", 84000.0, 3.0, 20.0)
        # Market best ask drops to 83999.0 (seller swept through 84000 bid)
        fills = self.sim.on_book_sweep(best_bid=83998.0, best_ask=83999.0)
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["qty"], 3.0)
        self.assertEqual(fills[0]["reason"], "PRICE_SWEEP_THROUGH")

    def test_token_bucket_rate_limiter(self):
        """Verify token bucket throttle under high-frequency burst conditions."""
        limiter = TokenBucketRateLimiter(rate_per_sec=20.0, capacity=5.0)
        # Should allow first 5 tokens
        for _ in range(5):
            self.assertTrue(limiter.can_send(1.0))
        # 6th should be throttled immediately
        self.assertFalse(limiter.can_send(1.0))
        
        stats = limiter.get_throttle_stats()
        self.assertEqual(stats["total_allowed"], 5)
        self.assertEqual(stats["total_throttled"], 1)


if __name__ == "__main__":
    unittest.main()
