"""
test_svi_vol_surface.py — Unit Tests for SVI Volatility Surface & Vol Arbitrage Engine
"""

import math
import unittest
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "python"))

from svi_vol_surface import SVIParams, SVIVolSurfaceEngine, VolArbSignal


class TestSVIVolSurfaceEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SVIVolSurfaceEngine("BTC")

    def test_svi_total_variance_and_implied_vol(self):
        """Verify SVI total variance w(k) and implied volatility sigma(k) identities."""
        # a=0.04, b=0.10, rho=-0.20, m=0.0, sigma=0.10, tau=0.1
        params = SVIParams(a=0.04, b=0.10, rho=-0.20, m=0.0, sigma=0.10)
        tau = 0.1

        # ATM (k = 0):
        # w(0) = 0.04 + 0.10 * (-0.20 * 0 + sqrt(0 + 0.1^2)) = 0.04 + 0.10 * 0.10 = 0.05
        w_atm = params.total_variance(k=0.0)
        self.assertAlmostEqual(w_atm, 0.05, places=4)

        # Implied vol = sqrt(0.05 / 0.1) = sqrt(0.5) ≈ 0.7071 (70.71%)
        iv_atm = params.implied_vol(k=0.0, tau=tau)
        self.assertAlmostEqual(iv_atm, math.sqrt(0.5), places=4)

    def test_svi_arbitrage_free_bounds(self):
        """Validate Gatheral & Roger Lee arbitrage-free conditions."""
        # Valid arbitrage-free params
        valid_params = SVIParams(a=0.04, b=0.15, rho=-0.25, m=0.0, sigma=0.10)
        is_valid, msg = valid_params.is_arbitrage_free()
        self.assertTrue(is_valid, f"Expected valid parameters, got: {msg}")

        # Invalid: Negative vertex variance
        neg_params = SVIParams(a=-0.20, b=0.05, rho=0.0, m=0.0, sigma=0.05)
        is_valid, _ = neg_params.is_arbitrage_free()
        self.assertFalse(is_valid, "Should reject negative vertex variance")

        # Invalid: Violates Roger Lee's extreme strike wing slope bound (b * (1 + |rho|) >= 4/3)
        lee_violation = SVIParams(a=0.10, b=1.0, rho=0.8, m=0.0, sigma=0.10) # 1.0 * (1 + 0.8) = 1.8 > 1.333
        is_valid, msg = lee_violation.is_arbitrage_free()
        self.assertFalse(is_valid, f"Should reject Roger Lee wing slope violation: {msg}")

    def test_svi_slice_fitting(self):
        """Test calibration of SVI to a synthetic crypto options smile."""
        spot = 84000.0
        tau = 0.082
        strikes = [75000.0, 80000.0, 84000.0, 90000.0, 95000.0]
        # Realistic crypto skew: higher IV on downside puts
        ivs = [65.0, 58.0, 52.0, 50.0, 53.0]

        params = self.engine.fit_svi_slice(strikes, ivs, spot, tau)
        is_valid, msg = params.is_arbitrage_free()
        self.assertTrue(is_valid, f"Fitted parameters must be arbitrage-free: {msg}")

        # Verify that fitted IV is reasonable (between 30% and 80%)
        fitted_atm = params.implied_vol(0.0, tau) * 100.0
        self.assertTrue(40.0 <= fitted_atm <= 70.0, f"Fitted ATM IV {fitted_atm}% out of bounds")

    def test_vol_arbitrage_signals(self):
        """Test generation of Variance Risk Premium (VRP) arbitrage signals."""
        now = 1000.0
        # Scenario 1: Extremely stable underlying price -> low Realized Vol, high VRP
        for i in range(50):
            self.engine.record_price(84000.0 + (i % 2) * 2.0, ts=now + i)
        
        # SVI params giving high ATM IV ~ 70%
        self.engine.params["0.082"] = SVIParams(a=0.04, b=0.15, rho=-0.20, m=0.0, sigma=0.10)
        signal = self.engine.evaluate_vol_arbitrage(tau=0.082)
        
        self.assertIsInstance(signal, VolArbSignal)
        self.assertEqual(signal.signal, "HARVEST_VOL")
        self.assertGreater(signal.vrp_spread_pct, 4.0)

        # Scenario 2: Massive price shock -> high Realized Vol, negative VRP -> BUY_CONVEXITY
        for i in range(50):
            # 10% swings every tick
            self.engine.record_price(84000.0 * (1.10 if i % 2 == 0 else 0.90), ts=now + 50 + i)
        
        signal_conv = self.engine.evaluate_vol_arbitrage(tau=0.082)
        self.assertEqual(signal_conv.signal, "BUY_CONVEXITY")
        self.assertLess(signal_conv.vrp_spread_pct, -2.0)


if __name__ == "__main__":
    unittest.main()
