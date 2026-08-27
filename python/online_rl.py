"""
online_rl.py — Lightweight Online Policy Gradient for HFT Position Sizing
==========================================================================
No PyTorch / stable-baselines3 required. Pure numpy.

Algorithm: REINFORCE with a linear policy (tanh output).
  - State:  8-dim feature vector [obi, vpin, vol, spread, ofi, alpha, cvd, hawkes]
  - Action: position_multiplier ∈ [0.5, 1.5]  (scales engine.config.order_size_btc)
  - Reward: realized PnL from each closed trade
  - Update: θ += lr * action_logit * reward  (policy gradient)

The policy falls back to multiplier=1.0 when fewer than MIN_SAMPLES updates have
been collected (cold-start safety).
"""

import math
import os
import numpy as np

N_FEATURES = 8        # [obi, vpin, vol, spread, ofi, alpha, cvd, hawkes]
MIN_SAMPLES = 20      # minimum updates before policy is trusted
LR_INIT     = 3e-5    # learning rate (conservative for HFT)
ENTROPY_REG = 1e-4    # L2 regularization to prevent weight explosion


class OnlineRLPolicy:
    """
    Linear stochastic policy for adaptive position sizing.

    Usage:
        policy = OnlineRLPolicy()
        # On each tick where a new order may be placed:
        mult = policy.act(obs_vector)               # → float in [0.5, 1.5]
        # After a trade closes with realized_pnl:
        policy.update(realized_pnl)
    """

    def __init__(self, save_path: str = None):
        self.theta       = np.zeros(N_FEATURES, dtype=np.float64)
        self.lr          = LR_INIT
        self.n_updates   = 0
        self._last_obs   = None
        self._last_logit = 0.0
        self._save_path  = save_path

        if save_path and os.path.exists(save_path):
            self._load(save_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def act(self, obs: np.ndarray) -> float:
        """
        Returns position_multiplier ∈ [0.5, 1.5].
        Falls back to 1.0 during cold start.
        """
        obs = self._normalize(obs)
        logit = float(np.dot(self.theta, obs))
        self._last_obs   = obs
        self._last_logit = logit

        if self.n_updates < MIN_SAMPLES:
            return 1.0  # neutral until enough data

        action = math.tanh(logit)          # [-1, 1]
        return 1.0 + 0.5 * action          # [0.5, 1.5]

    def update(self, realized_pnl: float):
        """REINFORCE gradient update on closed-trade PnL."""
        if self._last_obs is None:
            return

        # Policy gradient: ∇θ log π(a|s) ≈ tanh'(logit) * obs
        tanh_deriv = 1.0 - math.tanh(self._last_logit) ** 2
        grad = tanh_deriv * self._last_obs

        # Normalize reward to reduce variance
        reward = math.tanh(realized_pnl / 100.0)  # clip large PnL jumps

        self.theta += self.lr * reward * grad
        # L2 regularization (prevents weight explosion in live trading)
        self.theta -= ENTROPY_REG * self.theta
        # Clip to prevent runaway weights
        np.clip(self.theta, -2.0, 2.0, out=self.theta)

        self.n_updates += 1

        # Periodic save
        if self._save_path and self.n_updates % 50 == 0:
            self._save(self._save_path)

    def obs_from_features(self, fv, realized_vol_raw: float, sentiment: float = 0.0) -> np.ndarray:
        """Build observation vector from a FeatureVector + extras."""
        return np.array([
            fv.obi,
            fv.vpin - 0.5,          # center around 0
            realized_vol_raw,
            (fv.spread_bps - 2.0),  # center around typical spread
            fv.ofi,
            fv.combined_alpha,
            fv.cvd,
            fv.hawkes_intensity,
        ], dtype=np.float64)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, path: str):
        try:
            np.savez(path, theta=self.theta, n_updates=np.array([self.n_updates]))
        except Exception:
            pass

    def _load(self, path: str):
        try:
            data = np.load(path)
            self.theta     = data["theta"]
            self.n_updates = int(data["n_updates"][0])
            print(f"[RL] Loaded online policy from {path} (n_updates={self.n_updates})")
        except Exception as e:
            print(f"[RL] Could not load policy: {e}")

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(obs: np.ndarray) -> np.ndarray:
        """Clip + unit-norm to prevent gradient explosion."""
        obs = np.clip(obs, -10.0, 10.0)
        norm = np.linalg.norm(obs)
        return obs / (norm + 1e-8)
