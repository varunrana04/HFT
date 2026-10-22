"""
online_hmm_detector.py — Continuous Probabilistic Regime Detector
════════════════════════════════════════════════════════════════════════════════
Loads Gaussian Mixture / HMM model (models/hmm_regime.pkl) trained on realized
volatility and spread. Computes continuous state posterior probabilities:
  P(Regime_k | Observation_t)
Enables dynamic quote width scaling and smooth transition between specialists.
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "hmm_regime.pkl")

REGIME_NAMES = [
    "MEAN_REVERTING",  # Low volatility, tight spread
    "MOMENTUM_TREND",  # Moderate volatility, directional flow
    "HIGH_VOL_WIDE",   # Elevated volatility, widening spread
    "TOXIC_CRISIS",    # Extreme volatility & spread blowout
]


class OnlineHMMRegimeDetector:
    """
    Sub-0.05ms probabilistic regime classifier.
    Transforms instantaneous [hf_vol, spread_bps] into continuous regime probabilities.
    """
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.mean = None
        self.std = None
        self.state_mapping = None
        self._loaded = False
        self.load()

    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            print(f"[HMM DETECTOR] Warning: {self.model_path} not found. Using empirical fallback.")
            return False
        try:
            bundle = joblib.load(self.model_path)
            if isinstance(bundle, dict):
                self.model = bundle["model"]
                self.mean = np.array(bundle["mean"], dtype=np.float64)
                self.std = np.array(bundle["std"], dtype=np.float64) + 1e-8
                self.state_mapping = bundle.get("state_mapping", {})
            else:
                self.model = bundle
                self.mean = np.array([0.01, 1.5])
                self.std = np.array([0.01, 1.5])
            self._loaded = True
            print(f"[HMM DETECTOR] Loaded {type(self.model).__name__} regime model from {self.model_path}")
            return True
        except Exception as e:
            print(f"[HMM DETECTOR] Failed to load model: {e}")
            return False

    def predict_regime(self, hf_vol: float, spread_bps: float) -> Tuple[str, List[float], float]:
        """
        Classify market micro-regime.
        Returns:
            active_regime: str name
            probabilities: list of 4 float probabilities summing to 1.0
            spread_multiplier: dynamic quote expansion factor [0.8x - 2.5x]
        """
        # Scaled observation
        obs = np.array([[max(0.001, float(hf_vol)), max(0.05, float(spread_bps))]], dtype=np.float64)

        if not self._loaded or self.model is None:
            # Empirical fallback logic
            if spread_bps >= 2.0 or hf_vol >= 2.5:
                return "TOXIC_CRISIS", [0.05, 0.10, 0.25, 0.60], 2.2
            elif hf_vol >= 1.4:
                return "HIGH_VOL_WIDE", [0.10, 0.20, 0.60, 0.10], 1.5
            elif spread_bps <= 0.6 and hf_vol <= 1.0:
                return "MEAN_REVERTING", [0.70, 0.20, 0.08, 0.02], 1.0
            else:
                return "MOMENTUM_TREND", [0.20, 0.60, 0.15, 0.05], 1.2

        try:
            obs_scaled = (obs - self.mean) / self.std
            raw_probs = self.model.predict_proba(obs_scaled)[0]

            # Remap cluster indices to ordered regime definitions
            n_comp = len(raw_probs)
            probs = [0.0] * 4
            for comp_idx, p in enumerate(raw_probs):
                target_idx = int(self.state_mapping.get(str(comp_idx), comp_idx)) % 4
                probs[target_idx] += float(p)

            # Normalize to guarantee exact sum = 1.0
            tot = sum(probs) or 1.0
            probs = [round(p / tot, 4) for p in probs]

            max_idx = int(np.argmax(probs))
            active_regime = REGIME_NAMES[max_idx]

            # Dynamic spread multiplier: weighted expectation of quote expansion
            # Calm = 0.9x, Trend = 1.1x, Vol = 1.5x, Crisis = 2.4x
            weights = [0.9, 1.1, 1.5, 2.4]
            spread_multiplier = round(sum(p * w for p, w in zip(probs, weights)), 2)

            return active_regime, probs, spread_multiplier
        except Exception:
            return "MEAN_REVERTING", [0.70, 0.20, 0.08, 0.02], 1.0

    def update(self, symbol: str, hf_vol: float, spread_bps: float) -> dict:
        """Update and cache regime state for a specific instrument."""
        if not hasattr(self, "_latest_states"):
            self._latest_states = {}
        regime, probs, mult = self.predict_regime(hf_vol, spread_bps)
        st = {
            "symbol": symbol,
            "active_regime": regime,
            "probabilities": probs,
            "spread_multiplier": mult,
        }
        self._latest_states[symbol] = st
        return st

    def get_state(self, symbol: str) -> dict:
        """Retrieve latest cached regime state for an instrument."""
        if not hasattr(self, "_latest_states"):
            self._latest_states = {}
        if symbol in self._latest_states:
            return self._latest_states[symbol]
        return {
            "symbol": symbol,
            "active_regime": "MEAN_REVERTING",
            "probabilities": [0.74, 0.25, 0.01, 0.00],
            "spread_multiplier": 0.95,
        }

