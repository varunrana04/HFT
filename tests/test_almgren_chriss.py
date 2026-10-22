"""
test_almgren_chriss.py — Unit Tests for Almgren-Chriss Optimal Execution Engine
"""
import sys
from pathlib import Path
import pytest
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent / "python"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from almgren_chriss_execution import AlmgrenChrissExecutor, LiquidationPlan


def test_almgren_chriss_plan_generation():
    executor = AlmgrenChrissExecutor()
    plan = executor.generate_plan(
        symbol="BTCUSD",
        initial_qty=10.0,
        arrival_price=84000.0,
        horizon_seconds=60.0,
        num_slices=10,
        annual_vol=0.45,
        urgency="NORMAL_PASSIVE",
    )

    assert plan.plan_id.startswith("AC-BTCUSD")
    assert plan.initial_qty == 10.0
    assert plan.num_slices == 10
    assert len(plan.slices) == 10
    assert plan.slices[-1].target_inventory == 0.0

    # Sum of slice sizes must equal initial qty
    total_unwound = sum(s.slice_qty for s in plan.slices)
    assert abs(total_unwound - 10.0) < 1e-3

    # Trajectory must be monotonically decreasing
    for i in range(1, len(plan.slices)):
        assert plan.slices[i].target_inventory <= plan.slices[i - 1].target_inventory


def test_almgren_chriss_emergency_deleveraging():
    executor = AlmgrenChrissExecutor()
    plan_normal = executor.generate_plan("BTCUSD", 10.0, 84000.0, 60.0, 10, urgency="NORMAL_PASSIVE")
    plan_emergency = executor.generate_plan("BTCUSD", 10.0, 84000.0, 60.0, 10, urgency="EMERGENCY_DELEVERAGE")

    # In emergency mode, kappa is higher (faster initial unwinding)
    assert plan_emergency.kappa > plan_normal.kappa
    # First slice in emergency mode should be strictly larger than normal mode
    assert plan_emergency.slices[0].slice_qty > plan_normal.slices[0].slice_qty
    print(f"Normal slice 1: {plan_normal.slices[0].slice_qty} | Emergency slice 1: {plan_emergency.slices[0].slice_qty}")


def test_almgren_chriss_fill_recording_and_telemetry():
    executor = AlmgrenChrissExecutor()
    plan = executor.generate_plan("ETHUSD", 50.0, 2650.0, 30.0, 5)
    
    # Fill step 1
    executor.record_slice_fill(plan.plan_id, step_index=1, filled_qty=plan.slices[0].slice_qty, fill_price=2649.50)
    assert plan.slices[0].is_completed
    assert plan.remaining_qty < 50.0

    telem = executor.get_telemetry()
    assert telem["is_active"]
    assert plan.plan_id in telem["active_plans"]
    assert telem["active_plans"][plan.plan_id]["progress_pct"] == 20.0
