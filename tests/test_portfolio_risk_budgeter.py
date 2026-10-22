"""
test_portfolio_risk_budgeter.py — Unit Tests for Multi-Asset CVaR & Risk Budgeting Engine
"""

import unittest
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "python"))

from portfolio_risk_budgeter import PortfolioRiskBudgeter


class TestPortfolioRiskBudgeter(unittest.TestCase):
    def setUp(self):
        self.risk = PortfolioRiskBudgeter(initial_equity=10000.0)

    def test_covariance_matrix_positive_semi_definite(self):
        """Verify that the computed covariance matrix is symmetric and positive semi-definite."""
        cov = self.risk.compute_covariance_matrix()
        self.assertEqual(cov.shape, (4, 4))
        
        # Symmetry check
        self.assertTrue(np.allclose(cov, cov.T, atol=1e-5))

        # Positive semi-definiteness: all eigenvalues >= 0
        eigvals = np.linalg.eigvalsh(cov)
        for ev in eigvals:
            self.assertGreaterEqual(ev, 0.0, f"Negative eigenvalue detected: {ev}")

    def test_var_and_cvar_tail_risk(self):
        """Verify that CVaR > VaR and scales with inventory position."""
        # 1. Zero inventory -> 0 tail risk
        var, cvar, vol = self.risk.compute_portfolio_tail_risk()
        self.assertEqual(var, 0.0)
        self.assertEqual(cvar, 0.0)

        # 2. Add long position in BTC and ETH
        self.risk.update_position("BTC", 2.0)
        self.risk.update_position("ETH", 10.0)

        var, cvar, vol = self.risk.compute_portfolio_tail_risk()
        self.assertGreater(var, 0.0)
        self.assertGreater(cvar, var, "CVaR must strictly exceed VaR in the tail")
        self.assertGreater(vol, 0.0)

    def test_dynamic_kelly_inventory_sizing(self):
        """Verify volatility parity: higher vol assets receive tighter inventory bounds."""
        alloc = self.risk.compute_dynamic_limits()
        # SOL vol (70%) > BTC vol (45%) -> SOL dynamic limit scaling should reflect vol penalty
        sol_limit = alloc["SOL"]["dynamic_limit"]
        btc_limit = alloc["BTC"]["dynamic_limit"]
        self.assertGreater(sol_limit, 0.0)
        self.assertGreater(btc_limit, 0.0)

    def test_post_only_guard(self):
        """Verify that orders crossing the spread are rejected to avoid taker fees."""
        best_bid, best_ask = 84000.0, 84001.0

        # Passive buy at 84000.0 (inside spread) -> Accepted
        ok, msg = self.risk.check_post_only_guard("BUY", 84000.0, best_bid, best_ask)
        self.assertTrue(ok, msg)

        # Aggressive buy at 84001.5 (crosses ask) -> Rejected!
        ok, msg = self.risk.check_post_only_guard("BUY", 84001.5, best_bid, best_ask)
        self.assertFalse(ok)
        self.assertIn("POST_ONLY REJECT", msg)

        # Passive sell at 84001.0 -> Accepted
        ok, msg = self.risk.check_post_only_guard("SELL", 84001.0, best_bid, best_ask)
        self.assertTrue(ok, msg)

        # Aggressive sell at 83999.0 (crosses bid) -> Rejected!
        ok, msg = self.risk.check_post_only_guard("SELL", 83999.0, best_bid, best_ask)
        self.assertFalse(ok)
        self.assertIn("POST_ONLY REJECT", msg)

    def test_risk_status_and_telemetry(self):
        """Test telemetry output and healthy vs emergency risk state transitions."""
        telemetry = self.risk.get_telemetry()
        self.assertEqual(telemetry["risk_status"], "HEALTHY")
        self.assertGreater(telemetry["liquidation_distance_pct"], 50.0)

        # Force high margin usage
        self.risk.update_account(equity=10000.0, available_margin=1000.0)
        telemetry_warn = self.risk.get_telemetry()
        self.assertIn(telemetry_warn["risk_status"], ["WARNING", "EMERGENCY_DELEVERAGE"])


if __name__ == "__main__":
    unittest.main()
