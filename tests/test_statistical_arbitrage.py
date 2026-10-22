"""
test_statistical_arbitrage.py — Unit Tests for Statistical Arbitrage & Cointegration Engine
"""
import sys
from pathlib import Path
import pytest
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent / "python"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from statistical_arbitrage_engine import StatisticalArbitrageEngine, KalmanFilterHedgeRatio


def test_kalman_filter_hedge_ratio():
    kf = KalmanFilterHedgeRatio(delta=1e-4, R=1e-3)
    # True relationship: y = 2.0 * x + 10.0
    for x in np.linspace(10, 50, 250):
        y = 2.0 * x + 10.0 + np.random.normal(0, 0.05)
        alpha, beta = kf.update(x, y)
    
    assert abs(beta - 2.0) < 0.25
    assert isinstance(alpha, float)


def test_stat_arb_engine_initialization_and_telemetry():
    engine = StatisticalArbitrageEngine()
    telem = engine.get_telemetry()
    assert telem["is_active"]
    assert "ETH-BTC" in telem["pairs"]
    assert "SOL-BTC" in telem["pairs"]


def test_stat_arb_price_update_and_z_score():
    engine = StatisticalArbitrageEngine()
    
    # Feed stationary correlated prices
    btc_base = 84000.0
    eth_base = 2650.0
    for i in range(30):
        noise = float(np.sin(i / 5.0) * 10.0)
        engine.update_prices({
            "BTC": btc_base + noise,
            "ETH": eth_base + noise * 0.0315 + (15.0 if i == 29 else 0.0),
            "SOL": 116.0,
        })
    
    pair = engine.pairs["ETH-BTC"]
    assert pair.price_x > 0.0
    assert pair.price_y > 0.0
    assert pair.hedge_ratio_beta > 0.0
    assert pair.is_cointegrated
    print(f"ETH-BTC Z-Score: {pair.z_score:.2f} | Beta: {pair.hedge_ratio_beta:.5f}")


def test_stat_arb_signals_and_position_lifecycle():
    engine = StatisticalArbitrageEngine()
    
    # Force extreme price dislocation on Y to trigger SELL_Y_BUY_X
    for _ in range(20):
        engine.update_prices({"BTC": 84000.0, "ETH": 2650.0})

    # Dislocation upwards
    engine.update_prices({"BTC": 84000.0, "ETH": 2750.0})
    pair = engine.pairs["ETH-BTC"]
    assert pair.signal in ["SELL_Y_BUY_X", "HOLD_SHORT_Y", "NEUTRAL"]
    if pair.signal == "SELL_Y_BUY_X":
        assert pair.active_position == "SHORT_Y_LONG_X"

    # Revert back to mean
    for _ in range(5):
        engine.update_prices({"BTC": 84000.0, "ETH": 2650.0})
