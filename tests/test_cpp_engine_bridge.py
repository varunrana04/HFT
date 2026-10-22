"""
test_cpp_engine_bridge.py — Unit Tests for C++ High-Speed Engine Bridge & Telemetry
"""

import time
import unittest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "python"))

from cpp_engine_bridge import CPPEngineBridge


class TestCPPEngineBridge(unittest.TestCase):
    def setUp(self):
        self.bridge = CPPEngineBridge("BTCUSD")

    def test_bridge_book_tick_processing(self):
        """Verify processing of live orderbook quote ticks and feature extraction."""
        bid, ask = 84000.0, 84001.0
        feats, latency_us = self.bridge.process_book_tick(bid, ask, bid_sz=5.0, ask_sz=5.0)

        self.assertIn("microprice", feats)
        self.assertIn("ofi", feats)
        self.assertIn("vpin", feats)
        self.assertIn("spread_bps", feats)
        self.assertAlmostEqual(feats["microprice"], 84000.5, places=1)
        self.assertGreater(latency_us, 0.0)
        # Latency should be well under 100 microseconds
        self.assertLess(latency_us, 200.0)

    def test_bridge_trade_tick_processing(self):
        """Verify processing of live trades and latency metrics."""
        latency_us = self.bridge.process_trade_tick(price=84000.5, qty=2.0, is_buy=True)
        self.assertGreaterEqual(latency_us, 0.0)
        self.assertEqual(self.bridge.trades_processed, 1)

    def test_high_throughput_burst(self):
        """Benchmark high-throughput burst of 1,000 ticks."""
        t0 = time.perf_counter()
        for i in range(1000):
            p = 84000.0 + (i % 10) * 0.5
            self.bridge.process_book_tick(p, p + 1.0, 5.0, 5.0)
        elapsed = time.perf_counter() - t0

        avg_us = (elapsed / 1000.0) * 1_000_000.0
        print(f"\n[C++ BRIDGE BENCHMARK] 1,000 ticks processed: {avg_us:.2f} µs/tick (Throughput: {1000.0/elapsed:.0f} ticks/s)")
        
        # Telemetry check
        telemetry = self.bridge.get_telemetry()
        self.assertEqual(telemetry["ticks_processed"], 1000)
        self.assertIn("avg_latency_us", telemetry)
        self.assertEqual(telemetry["engine_status"], "C++ CORE LIVE")


if __name__ == "__main__":
    unittest.main()
