"""
paper_trade_rl_live.py — Low-Latency RL Market-Making Simulator
════════════════════════════════════════════════════════════════
Strategy: High-Frequency Reinforcement-Learning Market Maker with
Adaptive Microstructure Alphas and Dynamic Volatility Quoting.

FAST MICROSTRUCTURE ALPHAS (Eliminating lagging slow EMAs):
  1. alpha_micro_bps   — Instantaneous Microprice Deviation: ((Microprice - Mid) / Mid) * 10,000 (bps)
                         Zero-lag leading indicator of queue order pressure.
  2. alpha_ofi         — Real-Time Order Flow Imbalance (Cont-Kukanov formula)
                         Tracks dynamic top-of-book additions, cancellations, and aggressive consumption.
  3. alpha_tfi_fast    — Ultra-Fast 5-tick Trade Flow Imbalance: EMA(signed_vol, half_life=5)
                         Reacts immediately to trade bursts within 3-5 ticks.
  4. alpha_depth_imb   — Multi-Level Orderbook Depth Imbalance: (bid_depth - ask_depth) / total_depth
  5. alpha_hf_vol      — Fast 10-tick Realized Volatility: rolling std dev of log returns (annualized bps)
  6. alpha_spread_ratio— Instantaneous Spread Compression: current_spread / EMA(spread, 15)
  7. alpha_spread_bps  — Normalized market spread in basis points
  8. alpha_order_skew  — Net trade flow direction over last 20 ticks
  9. inventory_fraction— Current inventory position / max_position [-1.0, +1.0]

INVENTORY & ADAPTIVE RISK MANAGEMENT:
  • Microprice-shifted fair value: P_fair = P_mid + lambda * alpha_micro_bps
  • Volatility-adjusted quoting width: avoids adverse selection in high vol regimes
  • Soft inventory skew: long -> widen bid, tighten ask (prefer selling)
  • One-way unwind: |inventory| >= ONE_WAY_THRESHOLD -> quote only inventory-reducing side
  • Hard cap: inventory at max_position -> stop quoting that side

FILL SIMULATION:
  • Quotes placed INSIDE the market spread: my_bid = mid - offset, my_ask = mid + offset
  • Fill triggered when any market print crosses or touches within 1 tick of quote
  • 500ms fill cooldown prevents duplicate instantaneous fills
"""

import math
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

HERE        = Path(__file__).parent
ML_DIR      = HERE / "ml_pipeline" / "models"
POLICY_PATH = ML_DIR / "rl_policy.pt"


# ── Model Architecture (Matches train_rl_ppo.py) ──────────────────────────────
class StateExtractor(nn.Module):
    def __init__(self, n_in: int = 9, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 256),  nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, latent_dim), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 128), nn.SiLU(), nn.Linear(128, 3))
        self.log_std = nn.Parameter(torch.zeros(3))
        self.register_buffer("action_mins", torch.tensor([0.2, 0.2, 0.01]))
        self.register_buffer("action_maxs", torch.tensor([2.5, 2.5, 0.10]))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        lo, hi = self.action_mins, self.action_maxs
        return lo + (hi - lo) * torch.sigmoid(self.net(latent))


class RLPaperAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StateExtractor()
        self.actor   = Actor()

    @torch.no_grad()
    def get_action(self, state: torch.Tensor) -> np.ndarray:
        return self.actor(self.encoder(state)).squeeze(0).numpy()


# ── Incremental Online Filter ──────────────────────────────────────────────────
class FastOnlineEMA:
    """Ultra-low latency exponential moving average with configurable half-life."""
    def __init__(self, half_life_ticks: float):
        self.alpha = 1.0 - (2.0 ** (-1.0 / max(1.0, half_life_ticks)))
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1.0 - self.alpha) * self.value
        return self.value


# ── RL Limit Simulator with Fast Microstructure Alphas ────────────────────────
class RLLimitSimulator:
    def __init__(self, agent: RLPaperAgent):
        self.agent = agent

        # Fast non-lagging filters
        self.ema_tfi_fast   = FastOnlineEMA(5.0)    # 5-tick ultra-fast trade flow
        self.ema_tfi_med    = FastOnlineEMA(15.0)   # 15-tick fast trade flow
        self.ema_spread_fast= FastOnlineEMA(15.0)   # 15-tick spread baseline
        self.ema_vol_ticks  = FastOnlineEMA(20.0)   # 20-tick realized volatility

        # Rolling buffers for high-frequency volatility & momentum
        self.recent_returns  = deque(maxlen=10)     # 10-tick returns
        self.recent_trades   = deque(maxlen=20)     # 20-tick trade signs (-1 or +1)

        # Previous book state for Cont-Kukanov Order Flow Imbalance (OFI)
        self.prev_bid_px: Optional[float] = None
        self.prev_ask_px: Optional[float] = None
        self.prev_bid_qty: float = 0.0
        self.prev_ask_qty: float = 0.0

        # Current book
        self.bid_px: Optional[float] = None
        self.ask_px: Optional[float] = None
        self.bid_qty: float = 0.0
        self.ask_qty: float = 0.0
        self.last_price: Optional[float] = None

        # Position & PnL
        self.inventory: float = 0.0
        self.max_position: float = 10.0
        self.pnl: float = 0.0
        self.tick_count: int = 0
        self.fill_count: int = 0
        self.last_fill_ts: int = 0

        # Risk parameters
        self.FILL_COOLDOWN_MS: int = 500
        self.MIN_SPREAD_BPS: float = 0.2
        self.ONE_WAY_THRESHOLD: float = 5.0

        # Quotes
        self.my_bid: Optional[float] = None
        self.my_ask: Optional[float] = None

        # Latest computed alphas dictionary for telemetry
        self.latest_alphas = {
            "micro_bps": 0.0,
            "ofi": 0.0,
            "tfi_fast": 0.0,
            "depth_imb": 0.0,
            "hf_vol": 0.0,
            "spread_ratio": 1.0,
            "spread_bps": 0.0,
        }

    def on_book_tick(self, b_px: float, a_px: float, b_qty: float, a_qty: float):
        """Update top-of-book and compute Order Flow Imbalance (OFI)."""
        if b_px <= 0 or a_px <= 0:
            return

        # ── Cont-Kukanov Order Flow Imbalance (OFI) ──────────────────────────
        ofi = 0.0
        if self.prev_bid_px is not None and self.prev_ask_px is not None:
            # Bid side flow
            if b_px > self.prev_bid_px:
                delta_bid = b_qty
            elif b_px == self.prev_bid_px:
                delta_bid = b_qty - self.prev_bid_qty
            else:
                delta_bid = -self.prev_bid_qty

            # Ask side flow
            if a_px < self.prev_ask_px:
                delta_ask = a_qty
            elif a_px == self.prev_ask_px:
                delta_ask = a_qty - self.prev_ask_qty
            else:
                delta_ask = -self.prev_ask_qty

            ofi = delta_bid - delta_ask

        # Save previous state
        self.prev_bid_px = b_px
        self.prev_ask_px = a_px
        self.prev_bid_qty = b_qty
        self.prev_ask_qty = a_qty

        self.bid_px  = b_px
        self.ask_px  = a_px
        self.bid_qty = max(b_qty, 0.001)
        self.ask_qty = max(a_qty, 0.001)

        # Update OFI alpha normalized
        denom = (b_qty + a_qty + 1e-6)
        self.latest_alphas["ofi"] = float(np.clip(ofi / denom, -5.0, 5.0))

    def on_trade_tick(self, price: float, qty: float, is_bm: bool, ts_ms: int):
        """Process real-time trade tick, compute fast alphas, check fills, update quotes."""
        if price <= 0:
            return

        # 1. Log return for high-frequency realized volatility
        if self.last_price and self.last_price > 0:
            ret = math.log(price / self.last_price)
            self.recent_returns.append(ret)

        self.last_price = price
        signed_vol = -qty if is_bm else qty
        self.recent_trades.append(-1.0 if is_bm else 1.0)

        # 2. Fast Trade Flow Momentum (5-tick and 15-tick EMAs)
        f_tfi_fast = self.ema_tfi_fast.update(signed_vol)
        f_tfi_med  = self.ema_tfi_med.update(signed_vol)

        if None in (self.bid_px, self.ask_px) or self.bid_px <= 0 or self.ask_px <= 0:
            # Fallback if book tick hasn't arrived yet
            self.bid_px = price * 0.9998
            self.ask_px = price * 1.0002
            self.bid_qty = self.ask_qty = 1.0

        mid_px = (self.bid_px + self.ask_px) / 2.0
        if mid_px <= 0:
            return

        # 3. High-Frequency Realized Volatility (10-tick rolling std dev in bps)
        if len(self.recent_returns) >= 3:
            hf_vol_bps = float(np.std(self.recent_returns) * 10_000.0)
        else:
            hf_vol_bps = 1.0
        smoothed_vol = self.ema_vol_ticks.update(hf_vol_bps)

        # 4. Instantaneous Microprice Deviation (Zero-lag queue conviction)
        total_qty = self.bid_qty + self.ask_qty + 1e-9
        microprice = (self.bid_px * self.ask_qty + self.ask_px * self.bid_qty) / total_qty
        alpha_micro_bps = (microprice - mid_px) / mid_px * 10_000.0

        # 5. Orderbook Depth Imbalance
        book_imb = (self.bid_qty - self.ask_qty) / total_qty

        # 6. Instantaneous Spread Dynamics
        raw_spread_bps = (self.ask_px - self.bid_px) / mid_px * 10_000.0
        ema_spread = self.ema_spread_fast.update(raw_spread_bps)
        spread_ratio = raw_spread_bps / (ema_spread + 1e-6)

        # 7. Multi-tick trade flow direction
        order_skew = float(np.mean(self.recent_trades)) if self.recent_trades else 0.0

        # Store latest alphas for telemetry
        self.latest_alphas.update({
            "micro_bps": round(float(alpha_micro_bps), 3),
            "ofi": round(float(self.latest_alphas["ofi"]), 3),
            "tfi_fast": round(float(f_tfi_fast), 3),
            "depth_imb": round(float(book_imb), 3),
            "hf_vol": round(float(smoothed_vol), 2),
            "spread_ratio": round(float(spread_ratio), 3),
            "spread_bps": round(float(raw_spread_bps), 2),
        })

        tick_tolerance = max(mid_px * 0.00005, 0.01)

        # ── 8. Check Fills on Current Quotes ──────────────────────────────────
        on_cooldown = (ts_ms - self.last_fill_ts) < self.FILL_COOLDOWN_MS

        if not on_cooldown and self.my_bid is not None:
            if price <= self.my_bid + tick_tolerance and self.inventory < self.max_position:
                self.inventory += 1.0
                self.pnl -= float(self.my_bid)
                self.fill_count += 1
                self.last_fill_ts = ts_ms
                self.my_bid = None
                print(f"  [FILL] BOUGHT 1 @ {price:.2f} | Inv: {self.inventory:+.0f} | "
                      f"PnL: ${self.pnl + self.inventory * price:.2f}")

        if not on_cooldown and self.my_ask is not None:
            if price >= self.my_ask - tick_tolerance and self.inventory > -self.max_position:
                self.inventory -= 1.0
                self.pnl += float(self.my_ask)
                self.fill_count += 1
                self.last_fill_ts = ts_ms
                self.my_ask = None
                print(f"  [FILL] SOLD  1 @ {price:.2f} | Inv: {self.inventory:+.0f} | "
                      f"PnL: ${self.pnl + self.inventory * price:.2f}")

        # ── 9. Policy Decision with Scale-Invariant Features ───────────────────
        inv_frac = float(self.inventory / self.max_position)

        # Zero-centered, bounded feature vector (all ~[-3, +3])
        f1 = float(np.clip(f_tfi_fast / (total_qty + 1e-6), -3.0, 3.0))
        f2 = float(np.clip(f_tfi_med  / (total_qty + 1e-6), -3.0, 3.0))
        f3 = float(np.clip((smoothed_vol - 5.0) / 10.0, -3.0, 3.0))
        f4 = float(np.clip(self.latest_alphas["ofi"], -3.0, 3.0))
        f5 = float(np.clip(book_imb, -1.0, 1.0))
        f6 = float(np.clip((raw_spread_bps - 2.0) / 5.0, -2.0, 5.0))
        f7 = float(np.clip(alpha_micro_bps / 5.0, -3.0, 3.0))   # Normalized micro deviation
        f8 = float(np.clip(spread_ratio - 1.0, -2.0, 3.0))
        f9 = float(np.clip(inv_frac, -1.0, 1.0))
        state = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9], dtype=np.float32)

        if self.agent is not None:
            act = self.agent.get_action(torch.from_numpy(state))
            bid_mult, ask_mult, _ = act
        else:
            bid_mult, ask_mult = 1.0, 1.0

        # ── 10. Adaptive Quote Calculation ────────────────────────────────────
        half_spread = max((self.ask_px - self.bid_px) / 2.0, mid_px * 0.0001)
        min_offset  = self.MIN_SPREAD_BPS / 10_000.0 * mid_px

        # Microprice Fair Value Shift: skew midpoint towards microprice
        fair_shift = (alpha_micro_bps / 10_000.0) * mid_px * 0.3
        p_fair = mid_px + fair_shift

        # Volatility scaling: wider quotes when volatility expands
        vol_factor = max(0.8, min(2.5, smoothed_vol / 5.0))
        bid_offset = max(min_offset, min(half_spread * bid_mult * vol_factor, half_spread * 0.92))
        ask_offset = max(min_offset, min(half_spread * ask_mult * vol_factor, half_spread * 0.92))

        # Soft inventory skew
        skew = abs(inv_frac)
        if inv_frac > 0:    # Long -> widen bid, tighten ask (encourage selling)
            bid_offset *= (1.0 + skew * 2.0)
            ask_offset *= max(0.25, 1.0 - skew * 0.6)
        elif inv_frac < 0:  # Short -> tighten bid, widen ask (encourage buying)
            ask_offset *= (1.0 + skew * 2.0)
            bid_offset *= max(0.25, 1.0 - skew * 0.6)

        # One-way mode enforcement
        one_way_sell = self.inventory >= self.ONE_WAY_THRESHOLD
        one_way_buy  = self.inventory <= -self.ONE_WAY_THRESHOLD
        at_max_long  = self.inventory >= self.max_position
        at_max_short = self.inventory <= -self.max_position

        if at_max_long or one_way_sell:
            self.my_bid = None
            self.my_ask = p_fair + ask_offset
        elif at_max_short or one_way_buy:
            self.my_ask = None
            self.my_bid = p_fair - bid_offset
        else:
            self.my_bid = p_fair - bid_offset
            self.my_ask = p_fair + ask_offset

        self.tick_count += 1
