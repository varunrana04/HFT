# HFT Engine — Project Session Summary
### As of: August 16, 2026 | Role: QR / QT / QA / DS / MLE / AIE

---

## What This Project Is

A **sub-microsecond HFT market-making engine** targeting BTCUSDT perpetual on Binance. Covers the full institutional quant stack: C++ execution core, ML alpha generation, Python research pipeline, React dashboard, and institutional Monte Carlo risk simulation.

> **Aiden's (BU Quant Trader / Market Maker) key directive:**
> *"Single-asset specialization, low-risk high-reward repeated edge, institutional Monte Carlo to understand real trading scenarios, and strong Sharpe ratio analysis."*

---

## Full Architecture

```
Binance WebSocket (aggTrade + bookTicker)
        ↓
DataValidator → OrderBook (L2, 20 levels, O(1))
        ↓
FeatureEngine [6 alpha signals — O(1) Welford accumulators — ~248ns p50]
        ↓
SignalCombiner [3 parallel models]
  ├── WeightedAverage          (20ns)
  ├── ML_MODEL binary          (~10ns, 56-byte decision tree)
  └── ONNX LightGBM            (~2.1µs, 52 features)
        ↓
RiskManager [5 sequential pre-trade gates — <50ns]
        ↓
StrategyEngine → pybind11 → FastAPI WebSocket → React Dashboard
```

---

## What Has Been Built

### 1. C++ Core Engine (`cpp/core/`) — 21 files

**Six Alpha Signals (all O(1) via Welford accumulators):**

| Signal | Formula | Edge |
|---|---|---|
| Microprice | `(Qa·Pb + Qb·Pa)/(Qa+Qb)` | True fair value |
| OFI | `ΔBID_qty − ΔASK_qty` | Directional pressure |
| VPIN | `\|buy−sell\|/total` (128-bucket ring) | Toxic flow detection |
| Spread BPS | `(ask−bid)/mid × 10k` | Liquidity regime |
| Realized Vol | Welford σ, 4096-tick ring | Volatility regime |
| StatArb Z | `(mid−μ)/σ`, Welford 4096-tick | Mean-reversion |

**Trading Lifecycle:**
- Entry: `|α| ≥ 0.10` AND inventory skew OK AND all 5 risk gates pass
- Sizing: Fractional Kelly — `0.5%–5% AUM` scaled by `|α|`
- Market impact cap: max 5% of top-of-book qty per order
- Exit: `|α| < 0.04` → maker limit exit (capture half-spread)
- Hard stop: unrealized PnL < −2% equity → taker market exit
- Circuit Breaker: 60s halt on drawdown/daily-loss breach

**Latency:** 248ns p50 feature extraction | 2.1µs ONNX inference | 35ns risk gates

**5 Risk Gates (ordered cheapest → most expensive):**
1. Circuit breaker active?
2. `|new_position| > max_position_pct × AUM`?
3. Current drawdown > 5%?
4. Daily loss > 3%?
5. This single order > 2% AUM?

---

### 2. ML Pipeline

- **52 features**: 6 base signals + Z-scores at [10,50,200] ticks + lag-1 values + 4 cross-interactions + 4 regime one-hot flags
- **Model**: LightGBM Regressor → signed forward return target (avoids classification base-rate problem)
- **Export**: ONNX via `skl2onnx` → sub-3µs C++ inference
- **Validation**: 6-fold walk-forward OOS (`walk_forward.py`)

---

### 3. Python Research Pipeline (10 scripts)

| Script | Purpose |
|---|---|
| `backtest.py` | Historical replay of 30M BTCUSDT ticks (2024) via pybind11 |
| `train_model.py` | Feature dump → LightGBM → ONNX export |
| `walk_forward.py` | 6-fold OOS walk-forward, IC scoring per fold |
| `live_paper_trade.py` | Live Binance WS → paper trading |
| `generate_report_charts.py` | Equity curve, IC heatmap, trade distribution |
| `monte_carlo.py` | **NEW** — Institutional Monte Carlo (see below) |
| `optimize.py` | Parameter sweep (entry threshold, Kelly fraction) |

---

### 4. Monte Carlo Simulation — NEW (`python/monte_carlo.py`)

Built per Aiden's recommendation. Models:

| Layer | Implementation |
|---|---|
| Price | GBM + Merton jump-diffusion + 3-state regime HMM (calm/trending/volatile) |
| Spread | Regime-dependent: 2 bps calm → 20 bps volatile |
| Fill probability | 70% (queue-position model) |
| Adverse selection | 18% of fills hit by informed traders (VPIN-based), costs ~$21/BTC |
| Sizing | Fractional Kelly: 0.5%–5% AUM — exact mirror of C++ engine |
| Risk | Same drawdown/daily-loss/circuit-breaker thresholds as live engine |

**6 Output Charts (`results/monte_carlo/`):**
1. Equity path fan chart (5th/25th/50th/75th/95th percentile bands)
2. Terminal PnL distribution + VaR/CVaR markers
3. Annualised Sharpe + Calmar ratio distributions
4. Max drawdown distribution + circuit breaker threshold
5. Risk/Reward scatter (Sharpe vs Return, coloured by max DD)
6. VaR/CVaR waterfall + survival curve

```bash
# Run commands
python python/monte_carlo.py --simulations 5000 --horizon 2000 --capital 100000
python python/monte_carlo.py --regime volatile --simulations 5000
python python/monte_carlo.py --regime calm --simulations 5000
python python/monte_carlo.py --regime trending --simulations 5000
```

---

### 5. React Dashboard (`dashboard/`)
Live equity curve, PnL, inventory, L2 order book view, start/stop controls. WebSocket at 100ms.

---

### 6. External Resource Intelligence (`resource_intelligence_report.md`)

Evaluated 7 resources for integration:

| Resource | Key Takeaway | Priority |
|---|---|---|
| GS Quant | IC/Sharpe validation | P0 |
| LSE API | Options chain, IV rank signal | P0 |
| HuggingFace `varunrana04` | FinBERT sentiment, model upload | P1 |
| Awesome Systematic Trading | 220+ strategies; Short-Term Reversal Sharpe=0.816 | P1 |
| Google Doc (4 Projects) | StatArb needs cointegration test first — **bug identified** | P0 |
| NautilusTrader | Tearsheet visualisation, Tardis data adapter | P1 |
| GS Developer Portal | Marquee institutional data (long-term) | P3 |

---

## Aiden's Call — What He Told Us

1. **Role fit**: Quant Dev is the best fit for this project. Quant Researcher also achievable via BU MSDS.
2. **Monte Carlo is mandatory**: Not optional — a QR will immediately ask "how does this perform across 5000 market scenarios?"
3. **Market-making economics**:
   - Small edge per trade: $0.1–$0.9/unit
   - Large quantity × small edge = large absolute PnL
   - Risk: $0.1 | Reward: $1.5–$2 per unit → 10:1–20:1 R:R
   - Repeat thousands of times per day
4. **Single-asset depth**: Aiden trades one instrument deeply — we stay on BTCUSDT only
5. **Sell-side context**: Aiden works at a sell-side firm, with hedge funds and banks as counterparties who also market-make and bid/hedge large blocks
6. **Sharpe matters**: Not just point Sharpe — distribution across regimes and scenarios

---

## What To Do Next

### From MC Results (Immediate)
- [ ] Read `results/monte_carlo/mc_summary_report.txt` — check Median Sharpe, VaR 99%, P(Profit)
- [ ] Run `--regime volatile` stress test — this is what Aiden cares about most
- [ ] Compare regime-specific Sharpe distributions: volatile vs calm vs trending
- [ ] Check: does circuit breaker fire too often in volatile regime? (indicates over-tightening)

### Code Fixes (This Week)
- [ ] **StatArb cointegration pre-filter**: add `statsmodels.tsa.stattools.coint()` before Z-score — this is currently a bug (identified from Google Doc review)
- [ ] **Volatility Risk Premium feature**: `realized_vol_10d / implied_vol` → 53rd feature
- [ ] **Overnight anomaly feature**: `close_to_open_return` → 54th feature (3 lines)

### Short-Term (Next 2 Weeks)
- [ ] Sign up for LSE API key → pull BTCUSDT IV rank
- [ ] Run Short-Term Reversal strategy from `awesome-systematic-trading` on BTCUSDT data → benchmark against our StatArb Z-score (published Sharpe: 0.816)
- [ ] Upload ONNX model to HuggingFace `varunrana04` with model card
- [ ] Benchmark HMM regime detection vs our rule-based flags (`hmmlearn`)

### Medium-Term (1 Month)
- [ ] Extend MC to regime transition sequences (calm→volatile→calm)
- [ ] Options-based regime signals (IV rank + put/call ratio from Deribit)
- [ ] FinBERT sentiment as 55th feature
- [ ] Add Transaction Cost Analysis (TCA) module
- [ ] Run 10,000+ MC paths for publication-grade statistical confidence

---

## Future Perspectives

### Engine Evolution Roadmap

```
Phase 1 (Now)      : Single-asset, rule-based + LightGBM + Monte Carlo
Phase 2 (3 months) : IV rank + FinBERT + HMM regimes → 55-feature vector
Phase 3 (6 months) : BTC/ETH cross-asset StatArb, Deribit options hedging
Phase 4 (1 year)   : Deep Q-Network for queue position, GPU CUDA features
```

### What Quant Firms Will See

| Signal | Have | Need |
|---|---|---|
| Low-latency C++ (248ns) | ✅ | More unit tests |
| ML alpha (LightGBM ONNX) | ✅ | HMM + FinBERT comparison |
| Risk framework (5 gates) | ✅ | TCA, slippage model |
| Monte Carlo (5000 paths) | ✅ | 10k+ paths, regime transitions |
| Market microstructure | ✅ | VPIN paper replication |
| Published alpha benchmarks | ❌ | awesome-systematic-trading runs |
| Public portfolio (HF) | ❌ | ONNX model card upload |

### The One-Sentence Edge Statement (for interviews)
> *"We capture the bid-ask spread on BTCUSDT by posting maker limit orders when our OFI + LightGBM signal predicts directional flow, while VPIN screens out informed-trader adverse selection, operating at 248ns with fractional Kelly sizing and 5 sequential risk gates."*

---

## Project Structure

```
HFT/
├── cpp/core/           ← C++ hot path (21 files, ~248ns)
├── python/             ← Research pipeline (10 scripts)
│   └── monte_carlo.py  ← NEW: Institutional MC simulation
├── dashboard/          ← React/Vite live dashboard
├── data/               ← 30M BTCUSDT ticks (2024)
├── models/             ← ONNX model artefacts
├── results/
│   ├── monte_carlo/    ← MC outputs (NEW)
│   ├── walk_forward/   ← 6-fold OOS results
│   └── report_charts/  ← Backtest charts
├── docs/               ← Architecture, design, API docs
│   └── SESSION_SUMMARY.md ← This file
└── config/             ← Strategy/risk YAML configs
```
