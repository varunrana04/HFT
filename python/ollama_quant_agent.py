"""
ollama_quant_agent.py — Local AI Quantitative Diagnostic Agent (Llama 3.1)
═════════════════════════════════════════════════════════════════════════
Integrates local Ollama (Llama 3.1) on the user's PC for:
1. Post-Trade Loss Attribution & Adverse Selection Diagnosis
2. Real-Time Microstructure Tape Analysis
3. Adaptive Parameter Recommendations (VPIN thresholds, spread multipliers)
4. Fallback Rule-Based Quant Diagnostic Engine for zero-latency resilience
"""

import asyncio
import json
import os
import subprocess
import time
from typing import Dict, Any, Optional

import aiohttp

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_BIN   = r"C:\Users\Varun\AppData\Local\Programs\Ollama\ollama.exe"


class OllamaQuantAgent:
    """
    Quantitative AI Agent powered by local Llama 3.1.
    Evaluates execution quality, attributes adverse selection, and provides
    parameter adjustments.
    """
    def __init__(self):
        self.model = OLLAMA_MODEL
        self.host  = OLLAMA_HOST
        self.is_server_alive = False
        self.recent_diagnostics = []

    async def check_health(self) -> bool:
        """Check if local Ollama server is responsive."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.get(f"{self.host}/api/tags") as r:
                    if r.status == 200:
                        self.is_server_alive = True
                        return True
        except Exception:
            self.is_server_alive = False
        return False

    async def diagnose_trade(self, trade: dict, alphas: dict, regime: str) -> dict:
        """
        Diagnoses a trade execution, identifying whether a loss was due to
        Adverse Selection, Queue Starvation, Regime Mismatch, or Normal Friction.
        """
        prompt = f"""You are a senior quantitative risk manager and market microstructure researcher.
Analyze this cryptocurrency/gold algorithmic trade:

Trade Details:
- Symbol: {trade.get('symbol', 'BTCUSD')}
- Side: {trade.get('side', 'BUY')}
- Fill Price: ${trade.get('price', 0):.2f}
- Engine: {trade.get('engine', 'MIX')}
- Market Regime: {regime}

Microstructure Alphas at Fill:
- OFI (Order Flow Imbalance): {alphas.get('ofi', 0):.3f}
- Microprice Bias: {alphas.get('micro_bps', 0):.3f} bps
- VPIN (Flow Toxicity): {alphas.get('vpin', 0.5):.3f}
- Aggressive Buyer Ratio (ABR): {alphas.get('abr', 0.5):.3f}
- Aggressive Seller Ratio (ASR): {alphas.get('asr', 0.5):.3f}
- Book Collapse Rate (dOBI): {alphas.get('obi_delta', 0):.3f}
- Spread Compression: {alphas.get('spread_ratio', 1.0):.2f}x

Task:
1. Classify execution failure/risk: ADVERSE_SELECTION, QUEUE_STARVATION, REGIME_MISMATCH, or CLEAN_EXECUTION.
2. Provide a 1-sentence diagnostic explanation.
3. Recommend 1 specific parameter adjustment.

Respond ONLY in valid JSON matching this schema:
{{
  "classification": "ADVERSE_SELECTION" | "QUEUE_STARVATION" | "REGIME_MISMATCH" | "CLEAN_EXECUTION",
  "explanation": "...",
  "recommendation": "...",
  "optimal_engine": "BULLISH" | "BEARISH" | "MIX"
}}
"""
        response_json = await self._call_llm(prompt)
        if not response_json:
            # Deterministic rule-based fallback
            response_json = self._rule_based_diagnosis(trade, alphas, regime)

        record = {
            "timestamp": int(time.time() * 1000),
            "trade": trade,
            "regime": regime,
            **response_json
        }
        self.recent_diagnostics.append(record)
        self.recent_diagnostics = self.recent_diagnostics[-25:]
        return record

    async def analyze_market_pulse(self, base: str, mid: float, alphas: dict, regime: str, greeks: dict) -> dict:
        """Generate high-level quant commentary on current market conditions."""
        prompt = f"""You are a senior HFT quant trader. Evaluate this live market snapshot for {base}:
- Mid Price: ${mid:,.2f}
- Micro-Regime: {regime}
- OFI: {alphas.get('ofi', 0):.2f}
- VPIN (Flow Toxicity): {alphas.get('vpin', 0.5):.2f}
- Buyer Aggression (ABR): {alphas.get('abr', 0.5):.2f}
- Seller Aggression (ASR): {alphas.get('asr', 0.5):.2f}
- Net Options Delta: {greeks.get('net_delta', 0):.3f}
- Portfolio GEX: ${greeks.get('portfolio_gex', 0):.1f}

Provide a 2-sentence market pulse and advise which of the 3 engines (BULLISH, BEARISH, or MIX) has the mathematical edge right now.
Respond in JSON:
{{
  "edge_engine": "BULLISH" | "BEARISH" | "MIX",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "market_pulse": "..."
}}
"""
        res = await self._call_llm(prompt)
        if not res:
            res = self._rule_based_market_pulse(alphas, regime)
        return res

    async def _call_llm(self, prompt: str) -> Optional[dict]:
        """Calls local Ollama API."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 180,
                }
            }
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4)) as session:
                async with session.post(f"{self.host}/api/generate", json=payload) as r:
                    if r.status == 200:
                        data = await r.json()
                        raw_text = data.get("response", "").strip()
                        return json.loads(raw_text)
        except Exception:
            pass
        return None

    def _rule_based_diagnosis(self, trade: dict, alphas: dict, regime: str) -> dict:
        """Deterministic quantitative attribution when LLM is offline."""
        side = trade.get("side", "BUY")
        vpin = alphas.get("vpin", 0.5)
        ofi = alphas.get("ofi", 0.0)
        micro_bps = alphas.get("micro_bps", 0.0)
        asr = alphas.get("asr", 0.5)
        abr = alphas.get("abr", 0.5)

        if side == "BUY" and (vpin > 0.65 or asr > 0.65 or ofi < -0.3):
            return {
                "classification": "ADVERSE_SELECTION",
                "explanation": f"Bought into toxic seller distribution (VPIN={vpin:.2f}, ASR={asr:.2f}). Informed sellers offloaded into quote.",
                "recommendation": "Tighten Bullish VPIN kill gate to 0.52 and require ABR > 0.60 for buy fills.",
                "optimal_engine": "BEARISH"
            }
        elif side == "SELL" and (abr > 0.65 or micro_bps > 0.25):
            return {
                "classification": "ADVERSE_SELECTION",
                "explanation": f"Sold into aggressive buyer sweep (ABR={abr:.2f}, Micro Bias=+{micro_bps:.2f} bps). Risk of short squeeze.",
                "recommendation": "Increase Bearish squeeze halt threshold and step back ask quote by 1 tick.",
                "optimal_engine": "BULLISH"
            }
        elif regime == "TOXIC_CRISIS":
            return {
                "classification": "REGIME_MISMATCH",
                "explanation": "Market in high-volatility crisis mode with wide spreads. Quoting without wider offsets caused friction loss.",
                "recommendation": "Switch to MIX engine with 2.5x spread expansion multiplier.",
                "optimal_engine": "MIX"
            }
        else:
            return {
                "classification": "CLEAN_EXECUTION",
                "explanation": f"Normal microstructure fill within acceptable variance (Regime={regime}).",
                "recommendation": "Maintain current risk parameters and let edge compound.",
                "optimal_engine": "MIX"
            }

    def _rule_based_market_pulse(self, alphas: dict, regime: str) -> dict:
        abr = alphas.get("abr", 0.5)
        asr = alphas.get("asr", 0.5)
        ofi = alphas.get("ofi", 0.0)
        vpin = alphas.get("vpin", 0.5)

        if abr > 0.60 or ofi > 0.25:
            return {
                "edge_engine": "BULLISH",
                "confidence": "HIGH" if abr > 0.70 else "MEDIUM",
                "market_pulse": f"Buyer aggression dominates (ABR={abr:.1%}, OFI=+{ofi:.2f}). Bullish specialist has positive expectancy on dip fills."
            }
        elif asr > 0.60 or ofi < -0.25 or (0.60 <= vpin <= 0.80):
            return {
                "edge_engine": "BEARISH",
                "confidence": "HIGH" if asr > 0.70 else "MEDIUM",
                "market_pulse": f"Seller pressure and informed dumping detected (ASR={asr:.1%}, VPIN={vpin:.2f}). Bearish specialist has positive expectancy."
            }
        else:
            return {
                "edge_engine": "MIX",
                "confidence": "HIGH",
                "market_pulse": f"Market balanced in two-sided chop (ABR={abr:.1%}, ASR={asr:.1%}). Mix engine capturing bid-ask spread and Theta decay."
            }
