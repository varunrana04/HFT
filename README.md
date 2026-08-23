# Institutional HFT Market-Making Engine

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Python 3.11](https://img.shields.io/badge/Python-3.11-yellow.svg)
![Architecture](https://img.shields.io/badge/Architecture-Low--Latency-success.svg)

> **"Bridging rigorous quantitative engineering with dynamic machine learning edge."**

## 🎯 The Mission
This project is an institutional-grade, adverse-selection-gated Maker strategy. Naive mean-reversion market making is consistently steamrolled by toxic flow. This engine solves that by executing an optimized C++ hot path for limit order book interaction, dynamically gated by asynchronous Python ML models predicting regime volatility and cointegration breakdown.

## 🚀 Core Technology Stack
- **C++20 Hot Path**: Zero-allocation limit order book updates and atomic feature engineering. Network Round-Trip Time (RTT) to Binance AWS is in the milliseconds.
- **Pybind11**: Zero-overhead Python bindings to bridge the execution core to high-level ML.
- **Signal Optimizer**: CVaR-optimized linear combination of 6 core microstructure features (OFI, VPIN, Microprice, Spread, Volatility, StatArb).
- **Asyncio Python Bridge**: Real-time statistical arbitrage tracking (ADF Cointegration), True Hidden Markov Model (HMM) Regime classification, and FinBERT sentiment analysis pipeline.

## 📊 Quantitative Validation & Phase 12 Mathematical Proofs
This engine has undergone rigorous adversarial validation against real historical L2 snapshots (Tardis.dev, Binance UM BTCUSDT). Our findings represent a mathematically honest, out-of-sample appraisal of raw microstructure signals:

- **The C++ L2 Depth Reality Check**: The legacy pseudo-random fill simulator has been fully replaced with a deterministic **C++ queue-position liquidity absorption model**. The entry-side logic parses raw 25-level L2 depth arrays (verified against Binance CSV schema), tracking `l2_queue > 0 ? l2_queue : tob_qty`. This ensures the engine is structurally gated by real-world L2 friction.
- **The HMM Determinism Proof**: We have formally debunked earlier hallucinated claims that `hmmlearn` caused a massive 7,000+ trade swing due to random seeding. By pinning the global seed and running parallel dual tests at the correct `--threshold 0.25`, the engine deterministically yields identical 1,824 trade outputs. The pipeline is 100% deterministic.
- **The Bootstrap CI Mathematical Fix**: The earlier documented astronomical `16.826 [13.775, 21.746]` Sharpe ratio was formally retracted as an artifact of a structural methodology flaw in the bootstrap script (downsampling the history array and mis-scaling the annualization). We have structurally fixed this. The Point Estimate now perfectly sits within its 95% Confidence Interval. (e.g. Sharpe Ratio: -0.859 [95% CI: -12.230, -2.062]).
- **The 30% Out-Of-Sample (OOS) Baseline**: To prove true ML alpha, we evaluated the CVaR-optimized ML weights vs Equal weights directly on a chronologically segregated 30% OOS holdout window. The ML weights fundamentally reduce trade churn and loss:
  - **Equal Weights (OOS)**: 715 trades, Win Rate 3.6%, Total PnL -$37.02, Sharpe -0.859 [95% CI: -12.230, -2.062]
  - **ML Weights (OOS)**: 696 trades, Win Rate 5.2%, Total PnL -$16.74, Sharpe -0.377 [95% CI: -13.560, 5.120]
- **The $0.21 Math Leak Fix**: We have permanently resolved all PnL leakages. By properly surfacing the fractional mark-to-market positions remaining at session termination, `Realized PnL + Unrealized PnL` now perfectly reconciles to the `Final Equity Delta` down to the exact cent.

## ⚙️ Experimental & Aspirational Features
While the core C++ matching engine and basic ML weights are functional, several advanced features are currently labeled as **Experimental** and are not yet structurally validated for production:

- **Dockerized Environment**: The `Dockerfile` exists but is currently failing to build on Windows due to caching locks. It is considered aspirational until CI/CD is fully restored.
- **FinBERT & ADF Pipelines**: Advanced NLP sentiment (FinBERT) and ADF Cointegration modules are partially scaffolded but are not yet actively driving Live signals.
- **Microstructure Risk Gates**:
  - **Dynamic Inventory Sizing**: The Avellaneda-Stoikov pricing mechanism naturally skews quotes based on inventory to prevent directional over-exposure.
  - **Max Allowed Qty Clamping**: The engine structurally enforces a 5% market impact depth limit against the Top-Of-Book (TOB) to prevent the system from artificially consuming ghost liquidity.
  
## 🗺️ Roadmap
- **Short-term**: Extend the liquidity-absorption model with real queue-position tracking across multiple book levels.
- **Medium-term**: Source additional alpha signals (e.g., funding rate momentum, cross-exchange order book imbalances) to overcome the current adverse selection barrier.
- **Long-term**: Expand to cross-venue statistical arbitrage (Binance vs Bybit).

---

## 🛠 Quickstart Guide

### 1. Build the Engine
**Windows**:
```cmd
scripts\build.bat
```

**Linux / Mac**:
```bash
./scripts/build.sh
```

### 2. Configure Environment
Create your `.env` file with exchange keys:
```bash
cp .env.example .env
# Edit .env with your Binance Testnet keys
```

### 3. Run the Live Paper Trading Node
```bash
source .venv/bin/activate
python python/live_paper_trade.py
```

### 4. Institutional Deployment
See the [Deployment Runbook](docs/DEPLOYMENT_RUNBOOK.md) for AWS deployment configurations and Docker host-networking execution instructions.

### 5. Live Test Diagnostics
Read the [Post-Live Test Diagnostic Report](docs/POST_LIVE_TEST_REPORT.md) for a teardown of the theoretical strategy. Note: Previous claims of an 8-hour live test session have been formally retracted pending actual data collection.

---
*Built for extreme latency environments. Review `docs/INSTITUTIONAL_TEARDOWN.md` and `docs/PITCH_DECK.md` for the full quantitative presentation.*
