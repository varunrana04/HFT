"""
almgren_chriss_execution.py — Institutional Almgren-Chriss (2000) Optimal Execution Engine
═══════════════════════════════════════════════════════════════════════════════════════════
Implements the canonical Almgren & Chriss (2000) optimal liquidation framework:
    min_x E[x] + lambda * V[x]
balancing market impact costs against inventory volatility risk.

Mathematical Foundation:
- Total position to unwind: X_0
- Liquidation horizon: T (seconds) over N discrete slices: tau = T / N
- Permanent price impact: gamma (linear price drift per unit traded)
- Temporary price impact: eta (instantaneous execution friction / adverse spread cross)
- Volatility: sigma (annualized asset volatility converted to per-second sigma_s)
- Trader Risk Aversion: lambda_risk

Closed-Form Optimal Trajectory:
    kappa_tilde = (1 / tau) * arccosh( (tau^2 * lambda * sigma_s^2) / (2 * eta) + 1 )
    x_j = (sinh(kappa_tilde * (T - t_j)) / sinh(kappa_tilde * T)) * X_0
    n_j = x_{j-1} - x_j  (trade slice at step j)

Execution Modes:
1. NORMAL_PASSIVE: Maker-only limit order slicing with queue seniority preservation.
2. EMERGENCY_DELEVERAGE: High urgency (10x lambda_risk) accelerating hyperbolic unwind.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ExecutionSlice:
    step_index: int
    target_time_s: float
    target_inventory: float
    slice_qty: float
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    slippage_bps: float = 0.0
    is_completed: bool = False


@dataclass
class LiquidationPlan:
    plan_id: str
    symbol: str
    initial_qty: float
    remaining_qty: float
    arrival_price: float
    horizon_seconds: float
    num_slices: int
    kappa: float
    urgency_mode: str          # "NORMAL_PASSIVE", "EMERGENCY_DELEVERAGE"
    slices: List[ExecutionSlice]
    start_time_s: float
    is_active: bool = True
    completed_time_s: Optional[float] = None
    total_realized_slippage_usd: float = 0.0
    expected_shortfall_usd: float = 0.0


class AlmgrenChrissExecutor:
    """
    Institutional Execution Algorithm Orchestrator.
    Generates non-linear optimal schedules and tracks execution progress.
    """
    def __init__(self, default_gamma: float = 0.05, default_eta: float = 2.5):
        self.gamma = default_gamma   # Permanent price impact parameter ($/BTC)
        self.eta = default_eta       # Temporary price impact parameter ($/(BTC/s))
        self.active_plans: Dict[str, LiquidationPlan] = {}
        self._plan_counter: int = 0

    def generate_plan(self, symbol: str, initial_qty: float, arrival_price: float,
                      horizon_seconds: float = 60.0, num_slices: int = 10,
                      annual_vol: float = 0.45, urgency: str = "NORMAL_PASSIVE") -> LiquidationPlan:
        """
        Derive the closed-form Almgren-Chriss liquidation trajectory.
        """
        if abs(initial_qty) < 1e-6 or horizon_seconds <= 0 or num_slices <= 0:
            num_slices = max(1, num_slices)

        self._plan_counter += 1
        plan_id = f"AC-{symbol}-{self._plan_counter}"
        now_s = time.time()

        # Risk aversion lambda: higher in emergency mode
        lambda_risk = 5.0e-4 if urgency == "EMERGENCY_DELEVERAGE" else 1.0e-4
        # Annualized volatility to per-second volatility: sigma_s = annual_vol / sqrt(31557600)
        sigma_s = (annual_vol / math.sqrt(31_557_600.0)) * arrival_price

        tau = horizon_seconds / num_slices
        # Dimensionless urgency parameter
        arg = (tau * tau * lambda_risk * sigma_s * sigma_s) / (2.0 * max(1e-8, self.eta)) + 1.0
        kappa_tilde = math.acosh(max(1.0001, arg)) / tau
        kappa_T = kappa_tilde * horizon_seconds

        # Guard against math overflow in sinh
        kappa_T = min(50.0, kappa_T)
        denom_sinh = math.sinh(kappa_T) if kappa_T > 1e-4 else horizon_seconds

        slices: List[ExecutionSlice] = []
        prev_x = initial_qty
        
        for j in range(1, num_slices + 1):
            t_j = j * tau
            arg_j = min(50.0, kappa_tilde * (horizon_seconds - t_j))
            num_sinh = math.sinh(arg_j) if arg_j > 1e-4 else (horizon_seconds - t_j)
            
            # Remaining target inventory at step j
            if j == num_slices:
                x_j = 0.0
            else:
                x_j = (num_sinh / denom_sinh) * initial_qty

            slice_size = prev_x - x_j
            slices.append(ExecutionSlice(
                step_index=j,
                target_time_s=now_s + t_j,
                target_inventory=round(x_j, 4),
                slice_qty=round(slice_size, 4),
                filled_qty=0.0,
            ))
            prev_x = x_j

        # Expected implementation shortfall (Almgren-Chriss E[x]):
        # E[x] = 0.5 * gamma * X_0^2 + eta * sum(n_j^2 / tau)
        sum_nj_sq = sum(s.slice_qty ** 2 for s in slices)
        expected_shortfall = (0.5 * self.gamma * (initial_qty ** 2) + 
                              self.eta * (sum_nj_sq / tau)) * arrival_price

        plan = LiquidationPlan(
            plan_id=plan_id,
            symbol=symbol,
            initial_qty=round(initial_qty, 4),
            remaining_qty=round(initial_qty, 4),
            arrival_price=round(arrival_price, 2),
            horizon_seconds=horizon_seconds,
            num_slices=num_slices,
            kappa=round(kappa_tilde, 4),
            urgency_mode=urgency,
            slices=slices,
            start_time_s=now_s,
            is_active=True,
            expected_shortfall_usd=round(expected_shortfall, 2),
        )

        self.active_plans[plan_id] = plan
        return plan

    def record_slice_fill(self, plan_id: str, step_index: int, filled_qty: float, fill_price: float):
        """
        Record actual market execution for a planned slice.
        """
        plan = self.active_plans.get(plan_id)
        if not plan or not plan.is_active:
            return

        for s in plan.slices:
            if s.step_index == step_index:
                s.filled_qty += filled_qty
                s.avg_fill_price = fill_price
                s.is_completed = True
                
                # Realized slippage vs arrival price
                price_diff = (fill_price - plan.arrival_price) if plan.initial_qty < 0 else (plan.arrival_price - fill_price)
                slippage_usd = price_diff * filled_qty
                plan.total_realized_slippage_usd += slippage_usd
                
                plan.remaining_qty = max(0.0, plan.remaining_qty - abs(filled_qty))
                if abs(plan.remaining_qty) < 1e-4:
                    plan.is_active = False
                    plan.completed_time_s = time.time()
                break

    def get_telemetry(self) -> dict:
        """
        Telemetry snapshot for Dashboard display.
        """
        plans_data = {}
        for pid, p in self.active_plans.items():
            completed_slices = sum(1 for s in p.slices if s.is_completed)
            progress_pct = round((completed_slices / max(1, p.num_slices)) * 100.0, 1)

            plans_data[pid] = {
                "plan_id": p.plan_id,
                "symbol": p.symbol,
                "initial_qty": p.initial_qty,
                "remaining_qty": p.remaining_qty,
                "arrival_price": p.arrival_price,
                "urgency_mode": p.urgency_mode,
                "is_active": p.is_active,
                "progress_pct": progress_pct,
                "expected_shortfall_usd": p.expected_shortfall_usd,
                "total_realized_slippage_usd": round(p.total_realized_slippage_usd, 2),
                "num_slices": p.num_slices,
                "slices": [
                    {
                        "step": s.step_index,
                        "target_inv": s.target_inventory,
                        "slice_qty": s.slice_qty,
                        "filled_qty": s.filled_qty,
                        "completed": s.is_completed,
                    }
                    for s in p.slices
                ],
            }

        return {
            "is_active": any(p.is_active for p in self.active_plans.values()),
            "total_plans_created": self._plan_counter,
            "active_plans": plans_data,
        }
