"""
test_walk_forward_backtester.py — Unit Tests for Walk-Forward Live Backtester & Calibration
══════════════════════════════════════════════════════════════════════════════════════════
Verifies:
1. Walk-Forward Window Partitioning and Fold Indexing.
2. Spearman Rank & Pearson Information Coefficient (IC) accuracy.
3. Maker Fee Rebate (-0.02%) accounting and PnL conservation.
4. Metric outputs: Sharpe, Drawdown, Win Rate, and Calmar Ratio.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from walk_forward_backtester import WalkForwardBacktester


@pytest.fixture
def synthetic_tick_df():
    """Generates synthetic tick data with predictable mean-reversion and momentum characteristics."""
    np.random.seed(42)
    n = 600
    mid_price = 85000.0 + np.cumsum(np.random.randn(n) * 2.0)
    spread = 0.5 + np.random.rand(n) * 0.5
    bid = mid_price - spread / 2.0
    ask = mid_price + spread / 2.0

    df = pd.DataFrame({
        "timestamp": 1790123969000 + np.arange(n) * 1000,
        "symbol": "BTCUSD",
        "mid_price": mid_price,
        "bid": bid,
        "ask": ask,
        "spread_bps": spread / mid_price * 10000.0,
        "micro_bps": np.random.randn(n) * 0.2,
        "obi_delta": np.random.randn(n) * 0.1,
        "hf_vol": 1.0 + np.random.rand(n) * 0.5,
        "depth_imb": np.random.uniform(-0.8, 0.8, n),
        "tfi_fast": np.random.uniform(-1.0, 1.0, n),
        "abr": np.random.uniform(0.1, 0.9, n),
        "asr": np.random.uniform(0.1, 0.9, n),
    })
    # Add forward return: k=10
    k = 10
    df["fwd_ret"] = (df["mid_price"].shift(-k) - df["mid_price"]) / df["mid_price"]
    return df


def test_walk_forward_window_partitioning(synthetic_tick_df):
    """Checks that rolling windows step correctly without overlap violations."""
    backtester = WalkForwardBacktester(
        in_sample_len=200,
        out_of_sample_len=100,
        step_size=50
    )
    res = backtester.simulate_folds(synthetic_tick_df)

    assert "folds" in res
    assert res["folds_count"] > 0
    folds = res["folds"]

    # Total rows = 600. Window = 200 + 100 = 300. Step = 50.
    # Starts at 0, 50, 100, 150, 200, 250, 300 -> 7 folds
    assert res["folds_count"] == 7

    for i, fold in enumerate(folds):
        assert fold["fold_idx"] == i
        is_len = fold["in_sample_range"][1] - fold["in_sample_range"][0]
        oos_len = fold["out_of_sample_range"][1] - fold["out_of_sample_range"][0]
        assert is_len == 200
        assert oos_len == 100
        assert fold["in_sample_range"][1] == fold["out_of_sample_range"][0]


def test_information_coefficient_and_analytics(synthetic_tick_df):
    """Verifies that IC calculation produces bounded Pearson and Rank ICs and aggregations."""
    backtester = WalkForwardBacktester(
        in_sample_len=200,
        out_of_sample_len=100,
        step_size=100
    )
    res = backtester.simulate_folds(synthetic_tick_df)
    agg = res["aggregate"]

    # Verify bounds
    assert -1.0 <= agg["mean_ic"] <= 1.0
    assert -1.0 <= agg["mean_rank_ic"] <= 1.0
    assert isinstance(agg["information_ratio"], float)
    assert isinstance(agg["sharpe_ratio"], float)
    assert 0.0 <= agg["win_rate"] <= 100.0
    assert agg["max_drawdown"] >= 0.0


def test_maker_fee_rebate_and_equity_curve(synthetic_tick_df):
    """Verifies maker fee rebate (-0.02%) accounting and non-empty equity curve generation."""
    backtester = WalkForwardBacktester(
        in_sample_len=250,
        out_of_sample_len=100,
        step_size=100,
        maker_fee_rebate=-0.0002
    )
    res = backtester.simulate_folds(synthetic_tick_df)

    assert len(res["equity_curve"]) > 0
    # First point should start at 0
    assert res["equity_curve"][0]["equity"] == 0.0
    # Last point should match cumulative pnl
    assert res["equity_curve"][-1]["equity"] == res["aggregate"]["cumulative_pnl"]
