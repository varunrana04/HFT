"""
ollama_rl_engine.py — Hybrid Foundation LLM + Online Continuous RL Engine
════════════════════════════════════════════════════════════════════════
Architecture:
  1. SLOW COGNITIVE LOOP (OllamaBaseModelPrior):
     • Local Llama 3.1 on http://127.0.0.1:11434 acts as the Foundation Base Model.
     • Ingests macro features, order flow summaries, and live PnL history.
     • Maintains an in-context memory replay buffer:
         - Alpha Playbook Buffer: Recent market setups that generated positive Net PnL.
         - Toxic Pattern Buffer: Recent setups that suffered adverse selection or fee drag.
     • Emits structured Bayesian priors: p_bull, p_bear, p_mix, spread_multiplier, confidence.

  2. FAST REACTIVE LOOP (OnlineRLPolicyAdapter):
     • Pure NumPy zero-latency policy head operating synchronously on ticks (< 0.05ms).
     • State S_t (13-dim): Combines instantaneous microstructure alphas with Ollama priors.
     • Action A_t: Engine routing probabilities, reservation price skew, dynamic position size.

  3. CONTINUOUS PnL LEARNING LOOP (ContinuousPnLLearner):
     • Reward R_t = ΔNetPnL - FeePaid - λ_inv|q|σ - λ_as(Toxic).
     • Executes continuous online policy gradient updates on every fill and MTM checkpoint.
     • Gradient clipping and entropy regularization guarantee mathematical stability.
"""

import asyncio
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import aiohttp
import numpy as np

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")

N_FEATURES = 13  # 13-dim state vector
MAKER_FEE_RATE = 0.0002  # 0.02% (2 bps) on Delta Exchange
TAKER_FEE_RATE = 0.0005  # 0.05% (5 bps)
DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "online_rl_weights.json",
)


# ── Feature Names for Weight Interpretability Heatmap ─────────────────────────
FEATURE_NAMES = [
    "OFI (Order Flow Imb)",
    "VPIN (Toxicity Bias)",
    "ABR (Buyer Aggression)",
    "ASR (Seller Aggression)",
    "dOBI/dt (Book Collapse)",
    "Microprice Bias (bps)",
    "Spread Ratio",
    "Options Net Delta",
    "Ollama Bull Prior",
    "Ollama Bear Prior",
    "Ollama Mix Prior",
    "Ollama Spread Rec",
    "Ollama Confidence",
]


@dataclass
class TradeMemory:
    timestamp: int
    symbol: str
    side: str
    price: float
    engine: str
    net_pnl: float
    fee_paid: float
    is_adverse: bool
    alphas: Dict[str, float]


# ── 1. Slow Cognitive Loop: Ollama Foundation Base Model ───────────────────────
class OllamaBaseModelPrior:
    """
    Asynchronous Foundation Model agent powered by local Llama 3.1.
    Maintains in-context trade memory and generates macro Bayesian priors.
    """
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        self.host = host
        self.model = model
        self.is_alive = False

        # In-context memory buffers
        self.alpha_playbook: deque = deque(maxlen=8)    # Top winning trade setups
        self.toxic_patterns: deque = deque(maxlen=8)    # Top losing/adverse setups

        # Current Bayesian prior
        self.current_prior = {
            "p_bull": 0.333,
            "p_bear": 0.333,
            "p_mix": 0.334,
            "spread_multiplier": 1.0,
            "confidence": 0.70,
            "reasoning": "Initial neutral Bayesian prior over specialist engines.",
            "last_updated": int(time.time() * 1000),
        }
        self._is_thinking = False

    async def check_health(self) -> bool:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.get(f"{self.host}/api/tags") as r:
                    if r.status == 200:
                        self.is_alive = True
                        return True
        except Exception:
            self.is_alive = False
        return False

    def record_trade_memory(self, memory: TradeMemory):
        """Stores trade in the appropriate experiential memory buffer."""
        if memory.net_pnl > 0 and not memory.is_adverse:
            self.alpha_playbook.append(memory)
        elif memory.net_pnl < 0 or memory.is_adverse:
            self.toxic_patterns.append(memory)

    async def update_prior_async(self, base: str, mid_px: float, alphas: dict,
                                 greeks: dict, recent_net_pnl: float, win_rate: float):
        """Asynchronously queries Ollama with dynamic in-context memory replay."""
        if self._is_thinking:
            return  # Prevent overlapping LLM inferences
        self._is_thinking = True

        try:
            # Build dynamic in-context memory string
            playbook_str = ""
            if self.alpha_playbook:
                playbook_str = "Recent Profitable Setups (Reinforce These):\n"
                for m in list(self.alpha_playbook)[-4:]:
                    playbook_str += (f" • {m.engine} {m.side} @ ${m.price:.2f} -> Net PnL +${m.net_pnl:.2f} "
                                     f"(ABR={m.alphas.get('abr', 0.5):.2f}, VPIN={m.alphas.get('vpin', 0.5):.2f}, "
                                     f"OFI={m.alphas.get('ofi', 0):.2f})\n")

            toxic_str = ""
            if self.toxic_patterns:
                toxic_str = "Recent Toxic Losses / Adverse Selection (Avoid These):\n"
                for m in list(self.toxic_patterns)[-4:]:
                    adv_flag = "[ADVERSE SELECTION]" if m.is_adverse else "[FEE DRAG]"
                    toxic_str += (f" • {adv_flag} {m.engine} {m.side} @ ${m.price:.2f} -> Net PnL -${abs(m.net_pnl):.2f} "
                                  f"(ABR={m.alphas.get('abr', 0.5):.2f}, VPIN={m.alphas.get('vpin', 0.5):.2f}, "
                                  f"OFI={m.alphas.get('ofi', 0):.2f})\n")

            prompt = f"""You are a Lead Quant Researcher and Portfolio Manager.
Analyze this live microstructure state for {base}:
- Spot Price: ${mid_px:,.2f}
- Current Net PnL: ${recent_net_pnl:+.2f} | Win Rate: {win_rate:.1f}%
- OFI (Order Flow Imbalance): {alphas.get('ofi', 0):.2f}
- VPIN (Flow Toxicity): {alphas.get('vpin', 0.5):.2f}
- Buyer Aggression (ABR): {alphas.get('abr', 0.5):.2f}
- Seller Aggression (ASR): {alphas.get('asr', 0.5):.2f}
- Microprice Bias: {alphas.get('micro_bps', 0):.2f} bps
- Options Net Delta: {greeks.get('net_delta', 0):.3f}
- Options GEX: ${greeks.get('portfolio_gex', 0):.1f}

{playbook_str}
{toxic_str}

Task:
Synthesize an updated Bayesian Prior distribution over our 3 specialist engines:
1. BULLISH (Long accumulator)
2. BEARISH (Short distributor)
3. MIX (Regime-adaptive MM & Delta-neutral options)

Respond ONLY in valid JSON matching this schema:
{{
  "p_bull": float between 0.05 and 0.90,
  "p_bear": float between 0.05 and 0.90,
  "p_mix": float between 0.05 and 0.90,
  "spread_multiplier": float between 0.8 and 2.5,
  "confidence": float between 0.5 and 1.0,
  "reasoning": "1 concise sentence explaining the quantitative edge."
}}
Note: p_bull + p_bear + p_mix must sum to 1.0."""

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.15,
                    "top_p": 0.85,
                    "num_predict": 160,
                },
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4)) as session:
                async with session.post(f"{self.host}/api/generate", json=payload) as r:
                    if r.status == 200:
                        data = await r.json()
                        raw_json = json.loads(data.get("response", "{}").strip())
                        p_bull = float(raw_json.get("p_bull", 0.333))
                        p_bear = float(raw_json.get("p_bear", 0.333))
                        p_mix  = float(raw_json.get("p_mix", 0.334))
                        tot = p_bull + p_bear + p_mix + 1e-6
                        p_bull, p_bear, p_mix = p_bull / tot, p_bear / tot, p_mix / tot

                        self.current_prior = {
                            "p_bull": round(p_bull, 3),
                            "p_bear": round(p_bear, 3),
                            "p_mix":  round(p_mix, 3),
                            "spread_multiplier": round(float(raw_json.get("spread_multiplier", 1.0)), 2),
                            "confidence": round(float(raw_json.get("confidence", 0.7)), 2),
                            "reasoning": raw_json.get("reasoning", "Llama 3.1 Bayesian Prior updated."),
                            "last_updated": int(time.time() * 1000),
                        }
        except Exception:
            # Deterministic quantitative fallback when Ollama is offline/slow
            self._rule_based_prior(alphas, greeks)
        finally:
            self._is_thinking = False

    def _rule_based_prior(self, alphas: dict, greeks: dict):
        abr  = alphas.get("abr", 0.5)
        asr  = alphas.get("asr", 0.5)
        vpin = alphas.get("vpin", 0.5)
        ofi  = alphas.get("ofi", 0.0)

        if abr > 0.62 or ofi > 0.25:
            p_bull, p_bear, p_mix = 0.65, 0.10, 0.25
            reason = f"Buyer conviction dominant (ABR={abr:.2f}, OFI={ofi:+.2f}). Bullish specialist preferred."
        elif asr > 0.62 or (0.55 <= vpin <= 0.80):
            p_bull, p_bear, p_mix = 0.10, 0.65, 0.25
            reason = f"Seller aggression and toxicity elevated (ASR={asr:.2f}, VPIN={vpin:.2f}). Bearish specialist preferred."
        else:
            p_bull, p_bear, p_mix = 0.20, 0.20, 0.60
            reason = f"Balanced two-sided flow (ABR={abr:.2f}, ASR={asr:.2f}). Mix engine capturing spread."

        self.current_prior = {
            "p_bull": p_bull,
            "p_bear": p_bear,
            "p_mix":  p_mix,
            "spread_multiplier": 1.4 if vpin > 0.65 else 1.0,
            "confidence": 0.75,
            "reasoning": reason,
            "last_updated": int(time.time() * 1000),
        }


# ── 2. Fast Reactive Loop: Online RL Policy Adapter ────────────────────────────
class OnlineRLPolicyAdapter:
    """
    Sub-0.05ms Pure NumPy Policy Head operating directly on the live tick stream.
    Combines 13-dim microstructure signals with Ollama Foundation Model priors.
    """
    def __init__(self, lr: float = 2e-3, entropy_reg: float = 1e-5, auto_load: bool = True):
        self.lr = lr
        self.entropy_reg = entropy_reg

        # Weight matrices:
        # θ_engine: [3, 13] -> Softmax logits over [BULLISH, BEARISH, MIX]
        # θ_skew:   [13]    -> Reservation price skew in [-1, +1]
        # θ_size:   [13]    -> Position sizing multiplier in [0.5, 1.5]
        self.theta_engine = np.zeros((3, N_FEATURES), dtype=np.float64)
        self.theta_skew   = np.zeros(N_FEATURES, dtype=np.float64)
        self.theta_size   = np.zeros(N_FEATURES, dtype=np.float64)

        self.last_state: Optional[np.ndarray] = None
        self.last_engine_probs: np.ndarray = np.array([0.333, 0.333, 0.334])
        self.last_skew_logit: float = 0.0
        self.last_size_logit: float = 0.0
        self.n_updates: int = 0

        # Attempt to warm-start from persisted weights; fall back to cold-start priors
        if not (auto_load and self.load_weights()):
            # Baseline priors for cold start (Bullish prefers OFI/ABR, Bearish prefers ASR/VPIN)
            self.theta_engine[0, 0] =  0.4   # Bullish likes positive OFI
            self.theta_engine[0, 2] =  0.5   # Bullish likes high ABR
            self.theta_engine[1, 1] =  0.4   # Bearish likes high VPIN
            self.theta_engine[1, 3] =  0.5   # Bearish likes high ASR
            self.theta_engine[2, 6] = -0.3   # Mix prefers tight spread ratio

    def save_weights(self, filepath: Optional[str] = None) -> bool:
        """Persist learned theta matrices and update count to disk."""
        path = filepath or DEFAULT_WEIGHTS_PATH
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            payload = {
                "theta_engine": self.theta_engine.tolist(),
                "theta_skew": self.theta_skew.tolist(),
                "theta_size": self.theta_size.tolist(),
                "n_updates": int(self.n_updates),
                "last_updated": int(time.time() * 1000),
                "feature_names": FEATURE_NAMES,
            }
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            return True
        except Exception as e:
            print(f"[RLPolicy] Failed to save weights to {path}: {e}")
            return False

    def load_weights(self, filepath: Optional[str] = None) -> bool:
        """Load trained weights if available for immediate warm-start."""
        path = filepath or DEFAULT_WEIGHTS_PATH
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            self.theta_engine = np.array(payload["theta_engine"], dtype=np.float64)
            self.theta_skew   = np.array(payload["theta_skew"], dtype=np.float64)
            self.theta_size   = np.array(payload["theta_size"], dtype=np.float64)
            self.n_updates    = int(payload.get("n_updates", 0))
            return True
        except Exception as e:
            print(f"[RLPolicy] Failed to load weights from {path}: {e}")
            return False

    def build_state(self, alphas: dict, greeks: dict, prior: dict) -> np.ndarray:
        """Constructs scale-invariant 13-dim state vector."""
        s = np.zeros(N_FEATURES, dtype=np.float64)
        s[0]  = np.clip(alphas.get("ofi", 0.0), -3.0, 3.0)
        s[1]  = np.clip((alphas.get("vpin", 0.5) - 0.5) * 2.0, -1.0, 1.0)
        s[2]  = np.clip((alphas.get("abr", 0.5) - 0.5) * 2.0, -1.0, 1.0)
        s[3]  = np.clip((alphas.get("asr", 0.5) - 0.5) * 2.0, -1.0, 1.0)
        s[4]  = np.clip(alphas.get("obi_delta", 0.0), -2.0, 2.0)
        s[5]  = np.clip(alphas.get("micro_bps", 0.0) / 5.0, -2.0, 2.0)
        s[6]  = np.clip(alphas.get("spread_ratio", 1.0) - 1.0, -1.0, 3.0)
        s[7]  = np.clip(greeks.get("net_delta", 0.0) / 2.0, -2.0, 2.0)
        s[8]  = float(prior.get("p_bull", 0.333))
        s[9]  = float(prior.get("p_bear", 0.333))
        s[10] = float(prior.get("p_mix", 0.334))
        s[11] = float(prior.get("spread_multiplier", 1.0) - 1.0)
        s[12] = float(prior.get("confidence", 0.7))
        return s

    def act(self, s: np.ndarray, prior: dict) -> Tuple[str, np.ndarray, float, float]:
        """
        Fast Policy Forward Pass (< 0.05ms):
        Returns: (selected_engine, engine_probabilities, quote_skew, position_multiplier)
        """
        self.last_state = s

        # 1. Engine selection logits: RL Head + Bayesian LLM log-prior
        rl_logits = np.dot(self.theta_engine, s)
        prior_probs = np.array([
            max(1e-4, prior.get("p_bull", 0.333)),
            max(1e-4, prior.get("p_bear", 0.333)),
            max(1e-4, prior.get("p_mix", 0.334)),
        ])
        log_prior = np.log(prior_probs)

        # Blended logits (Bayesian posterior)
        combined_logits = rl_logits + log_prior
        # Numerically stable softmax
        exp_logits = np.exp(combined_logits - np.max(combined_logits))
        engine_probs = exp_logits / np.sum(exp_logits)
        self.last_engine_probs = engine_probs

        engine_idx = int(np.argmax(engine_probs))
        engine_map = ["bullish", "bearish", "mix"]
        selected_engine = engine_map[engine_idx]

        # 2. Reservation Price Skew: [-1.0, +1.0]
        self.last_skew_logit = float(np.dot(self.theta_skew, s))
        quote_skew = math.tanh(self.last_skew_logit)

        # 3. Dynamic Position Multiplier: [0.5, 1.5]
        self.last_size_logit = float(np.dot(self.theta_size, s))
        pos_multiplier = 1.0 + 0.5 * math.tanh(self.last_size_logit)

        return selected_engine, engine_probs, quote_skew, pos_multiplier


# ── 3. Continuous PnL Learning & Online Policy Gradient ────────────────────────
class ContinuousPnLLearner:
    """
    Updates the Online RL Policy Adapter weights in real time using audited Net PnL.
    Enforces exchange maker/taker fee accounting and adverse selection penalties.
    """
    def __init__(self, policy: OnlineRLPolicyAdapter, ollama_prior: OllamaBaseModelPrior):
        self.policy = policy
        self.ollama = ollama_prior

        # Performance counters
        self.total_net_pnl: float = 0.0
        self.total_fees_paid: float = 0.0
        self.total_fills: int = 0
        self.profitable_fills: int = 0
        self.adverse_fills: int = 0
        self.cumulative_reward: float = 0.0
        self.reward_history: deque = deque(maxlen=60)
        self.recent_trades: deque = deque(maxlen=30)

    def record_fill(self, fill: dict, alphas: dict, gross_pnl: float,
                    inv: float, vol: float, is_adverse: bool = False):
        """
        Called on every fill or closed trade round:
        1. Deducts exchange transaction fee.
        2. Computes audited net reward R_t.
        3. Updates RL policy gradient.
        4. Ingests trade into Ollama in-context memory.
        """
        price = float(fill.get("price", 0.0))
        qty   = float(fill.get("qty", 1.0))
        side  = fill.get("side", "BUY")
        engine = fill.get("engine", "MIX")

        # 1. Exact Delta Exchange Maker Fee
        fee_paid = price * qty * MAKER_FEE_RATE
        net_pnl = gross_pnl - fee_paid
        self.total_fees_paid += fee_paid
        self.total_net_pnl += net_pnl
        self.total_fills += 1

        if net_pnl > 0:
            self.profitable_fills += 1
        if is_adverse:
            self.adverse_fills += 1

        # 2. Audited Reward Calculation:
        # R_t = tanh(NetPnL / scale) - fee_penalty - inv_penalty - adverse_penalty
        pnl_component = math.tanh(net_pnl / 20.0)  # bounded [-1, 1]
        fee_penalty   = (fee_paid / max(1.0, price * 0.0005)) * 0.2
        inv_penalty   = 0.05 * abs(inv) * (vol / 5.0)
        adverse_pen   = 0.50 if is_adverse else 0.0

        reward = float(pnl_component - fee_penalty - inv_penalty - adverse_pen)
        reward = float(np.clip(reward, -2.0, 2.0))
        self.cumulative_reward += reward
        self.reward_history.append(round(reward, 3))

        # 3. Policy Gradient Update (θ ← θ + η * R_t * ∇ log π)
        s = self.policy.last_state
        if s is None:
            s = self.policy.build_state(alphas, {}, self.ollama.current_prior)
            self.policy.act(s, self.ollama.current_prior)

        # Engine policy gradient: ∇θ log π(a|s) = (e_a - π) ⊗ s
        eng_idx = 0 if engine == "BULLISH" else (1 if engine == "BEARISH" else 2)
        grad_engine = -self.policy.last_engine_probs.copy()
        grad_engine[eng_idx] += 1.0
        # Outer product [3, 13]
        self.policy.theta_engine += self.policy.lr * reward * np.outer(grad_engine, s)

        # Skew gradient: ∇θ log π(skew|s) ≈ (1 - tanh^2(logit)) * s
        skew_deriv = 1.0 - math.tanh(self.policy.last_skew_logit) ** 2
        self.policy.theta_skew += self.policy.lr * reward * skew_deriv * s

        # Sizing gradient
        size_deriv = 1.0 - math.tanh(self.policy.last_size_logit) ** 2
        self.policy.theta_size += self.policy.lr * reward * size_deriv * s

        # Weight regularization (prevents runaway explosion)
        self.policy.theta_engine *= (1.0 - self.policy.entropy_reg)
        self.policy.theta_skew   *= (1.0 - self.policy.entropy_reg)
        self.policy.theta_size   *= (1.0 - self.policy.entropy_reg)

        np.clip(self.policy.theta_engine, -2.5, 2.5, out=self.policy.theta_engine)
        np.clip(self.policy.theta_skew,   -2.5, 2.5, out=self.policy.theta_skew)
        np.clip(self.policy.theta_size,   -2.5, 2.5, out=self.policy.theta_size)

        self.policy.n_updates += 1

        # Periodic weight checkpoint (every 50 online fill updates)
        if self.policy.n_updates % 50 == 0:
            self.policy.save_weights()

        # 4. Feed trade memory into Ollama in-context buffers
        mem = TradeMemory(
            timestamp=int(time.time() * 1000),
            symbol=fill.get("symbol", "BTCUSD"),
            side=side,
            price=price,
            engine=engine,
            net_pnl=round(net_pnl, 2),
            fee_paid=round(fee_paid, 2),
            is_adverse=is_adverse,
            alphas=dict(alphas),
        )
        self.ollama.record_trade_memory(mem)
        self.recent_trades.append(mem)

    def get_status(self) -> dict:
        """Returns comprehensive telemetry for API and Dashboard."""
        win_rate = (self.profitable_fills / max(1, self.total_fills)) * 100.0
        adv_rate = (self.adverse_fills / max(1, self.total_fills)) * 100.0

        # Extract top weights for heatmap
        engine_weights_summary = {
            "bullish_weights": [round(float(w), 3) for w in self.policy.theta_engine[0]],
            "bearish_weights": [round(float(w), 3) for w in self.policy.theta_engine[1]],
            "mix_weights":     [round(float(w), 3) for w in self.policy.theta_engine[2]],
            "skew_weights":    [round(float(w), 3) for w in self.policy.theta_skew],
        }

        # Format playbooks
        playbook_list = [
            {
                "symbol": m.symbol, "side": m.side, "engine": m.engine,
                "net_pnl": m.net_pnl, "price": m.price, "time": m.timestamp,
                "vpin": m.alphas.get("vpin", 0.5), "abr": m.alphas.get("abr", 0.5),
            }
            for m in list(self.ollama.alpha_playbook)[-5:]
        ]
        toxic_list = [
            {
                "symbol": m.symbol, "side": m.side, "engine": m.engine,
                "net_pnl": m.net_pnl, "price": m.price, "time": m.timestamp,
                "vpin": m.alphas.get("vpin", 0.5), "asr": m.alphas.get("asr", 0.5),
                "is_adverse": m.is_adverse,
            }
            for m in list(self.ollama.toxic_patterns)[-5:]
        ]

        return {
            "is_alive":            self.ollama.is_alive,
            "n_updates":           self.policy.n_updates,
            "cumulative_reward":   round(self.cumulative_reward, 2),
            "recent_rewards":      list(self.reward_history),
            "total_net_pnl":       round(self.total_net_pnl, 2),
            "total_fees_paid":     round(self.total_fees_paid, 2),
            "total_fills":         self.total_fills,
            "win_rate_pct":        round(win_rate, 1),
            "adverse_rate_pct":    round(adv_rate, 1),
            "current_prior":       self.ollama.current_prior,
            "last_engine_probs":   [round(float(p), 3) for p in self.policy.last_engine_probs],
            "feature_names":       FEATURE_NAMES,
            "weights":             engine_weights_summary,
            "alpha_playbook":      playbook_list,
            "toxic_patterns":      toxic_list,
        }
