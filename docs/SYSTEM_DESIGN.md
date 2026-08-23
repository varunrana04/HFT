# HFT Engine — System Design Document (HLD)

**Version**: 3.0 | **Date**: August 2026 | **Author**: Varun Rana | **Status**: Production-Ready

---

## 1. Executive Summary

This document describes the high-level architecture of a **zero-allocation, sub-microsecond C++ algorithmic trading engine** validated on 30 million live Binance BTCUSDT ticks. The system achieves a tick-to-trade pipeline latency of **~248 nanoseconds** and integrates a LightGBM/ONNX machine learning model directly into the C++ hot path.

The engine's strategy is fundamentally differentiated by its **Maker Rebate execution model**: instead of crossing the spread (Taker), it submits passive Limit Orders to capture exchange rebates, materially shifting the risk-reward profile in low-volatility regimes.

The system is monitored via a real-time React/Vite dashboard, streaming live telemetry through a FastAPI WebSocket server.

---

## 2. Architecture Overview

![HFT Engine Architecture](architecture_diagram.jpg)

### Hot-Path Pipeline
```
Binance WebSocket / Historical CSV
    → DataValidator (8 quality checks)
    → OrderBook (L2, 20 levels, O(1) update)
    → FeatureEngine (6 α signals, O(1) accumulators)
    → SignalCombiner (WEIGHTED_AVG | ML_MODEL | ONNX_MODEL)
    → RiskManager (5 pre-trade gates, <50ns total)
    → StrategyEngine (Maker Limit Order Queue)
    → pybind11 (zero-copy)
    → FastAPI WebSocket (100ms streaming)
    → React/Vite Dashboard (live equity, PnL, order book)
```

All hot-path operations are `O(1)`, zero-allocation, `noexcept`, and cache-line aligned.

---

## 3. Design Principles

| Principle | Implementation | Impact |
|---|---|---|
| **Zero-Allocation Hot Path** | `MemoryPool<T>` with pre-allocated blocks; no `new`/`malloc` | Eliminates GC pauses |
| **Cache-Line Alignment** | All hot structs `alignas(64)` | Prevents false sharing |
| **Fixed-Point Arithmetic** | `int64_t × 10^8` | Deterministic replay |
| **Lock-Free Concurrency** | `SPSCQueue<T>` with atomic acquire/release | No mutex contention |
| **Compile-Time Bounds** | Ring buffers sized at `constexpr` | No dynamic allocation |
| **POD Guarantees** | `static_assert(is_trivially_copyable_v<T>)` | Safe lock-free copy |
| **noexcept Hot Path** | All `on_trade()`, `combine()`, risk checks declared `noexcept` | No exception overhead |
| **O(1) Feature Engine** | Welford running accumulators + ring buffers | 6.2µs → 248ns |

---

## 4. Performance Benchmarks

![Latency Stats](latency_stats.jpg)

| Operation | p50 | p99 | Complexity |
|---|---|---|---|
| **Feature Extraction (`FeatureEngine`)** | **248 ns** | 412 ns | **O(1)** |
| Signal Combine (WEIGHTED_AVG) | 20 ns | 40 ns | O(1) |
| Signal Combine (ONNX LightGBM) | 2.1 µs | 4.8 µs | O(M) |
| Risk Manager (5 gates) | 35 ns | 60 ns | O(1) |
| **Full hot path (no ONNX)** | **~300 ns** | ~520 ns | **O(1)** |

*Measured on Intel Core i7-12700H, Release build (`-O3 -march=native`).*

> **Before O(1) optimization**: Feature extraction was `O(N)` sliding window loops: **6.2µs** per tick.  
> **After O(1) optimization**: Welford running sums + ring buffers: **248ns** per tick. **25× speedup.**

---

## 5. Component Overview

### 5.1 Data Layer

| Component | File | Responsibility |
|---|---|---|
| DataValidator | `data_validator.h/.cpp` | 8 quality checks: crossed books, stale timestamps, sequence gaps |
| MarketDataParser | `market_data.h/.cpp` | Parses Binance WebSocket JSON → `BookSnapshot` + `Trade` POD structs |

### 5.2 C++ Core Engine

| Component | File | Responsibility |
|---|---|---|
| OrderBook | `order_book.h/.cpp` | L2 depth (20 levels), O(1) best-price cache |
| FeatureEngine | `features.h/.cpp` | 6 O(1) alpha signals: Microprice, OFI, VPIN, Spread BPS, RealVol, StatArb Z |
| SignalCombiner | `signal_combiner.h/.cpp` | WEIGHTED_AVG / 56-byte ML weights / ONNX_MODEL runtime selectable |
| RiskManager | `risk_manager.h/.cpp` | 5 sequential pre-trade gates + 60s circuit breaker |
| StrategyEngine | `strategy_engine.h/.cpp` | Maker Limit Order queue, PnL tracking, online Sharpe |
| SPSC Queue | `spsc_queue.h` | Lock-free ring buffer for inter-thread data handoff |
| MemoryPool | `memory_pool.h` | Zero-fragmentation fixed-block allocator, no heap on hot path |

### 5.3 ML / Signal Mode

| Mode | Description | Latency |
|---|---|---|
| `WEIGHTED_AVG` | Equal or custom weighted linear combination | ~20ns |
| `ML_MODEL` | LightGBM feature importances → 56-byte binary | ~20ns |
| `ONNX_MODEL` | Full LightGBM graph via ONNX Runtime, pre-allocated float32 buffer | ~2-5µs |

### 5.4 Python Research Layer

| Component | File | Responsibility |
|---|---|---|
| Live Paper Trading | `live_paper_trade.py` | FastAPI + WebSocket, Maker limit order simulation |
| Backtester | `backtest.py` | Historical CSV replay, equity curve, Sharpe, drawdown |
| Model Training | `train_model.py` | LightGBM regression on 52 features, ONNX export |
| Walk-Forward | `walk_forward.py` | 6-fold rolling OOS validation, fold-best ONNX export |

### 5.5 Dashboard

| Component | Technology | Features |
|---|---|---|
| Backend | FastAPI + WebSocket | 100ms streaming, Start/Stop API, `/api/trade_status` |
| Frontend | React + Vite + TailwindCSS | Live equity curve, PnL metrics, order book depth visualization |

---

## 6. Maker Rebate Strategy

```
On alpha signal trigger (|α| > threshold):
    Place Limit Order @ best_bid (BUY) or best_ask (SELL)
    ↓
    Monitor exchange trade flow for queue drain simulation
    ↓
    On simulated fill: capture Maker Rebate
    ↓
    Update inventory, cash, equity in real-time
```

**Why Maker over Taker?**  
Taker fees on Binance: ~4bps → negative expectancy in low-vol regimes.  
Maker rebate: ~0-1bps credit → turns mean-reverting alpha into a profitable strategy.

---

## 7. Risk Architecture

```
Gate 1: Position Limit   → |position| ≤ max_position_btc
Gate 2: Drawdown Gate    → (peak − current) / peak ≤ 5%
Gate 3: Daily Loss       → daily_loss / day_start ≤ 3%
Gate 4: Order Size       → notional / portfolio ≤ 2%
Gate 5: Circuit Breaker  → 60s cooldown after any gate breach
```

All gates are `O(1)`, `noexcept`, combined in **<50ns**.

---

## 8. Deployment Modes

| Mode | Entry Point | Output |
|---|---|---|
| **Live Paper Trading** | `uvicorn python.live_paper_trade:app` + `npm run dev` | Real-time Dashboard |
| **Historical Backtest** | `python python/backtest.py --data X.csv` | Equity curve, Sharpe, Drawdown |
| **Walk-Forward OOS** | `python python/walk_forward.py --n-folds 6` | `lgb_model.onnx`, wf_report.md |
| **C++ Unit Tests** | `ctest --output-on-failure` | 36+ tests (9 suites) |
| **Latency Benchmark** | `./build/hft_bench` | p50/p99 table |

---

## 9. Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Core engine | C++20 | Sub-µs latency, zero-cost abstractions |
| Fixed-point | `int64_t × 10^8` | Deterministic replay, no FP drift |
| Memory | `MemoryPool<T>` | Zero fragmentation on hot path |
| Concurrency | `SPSCQueue<T>` | Lock-free, no OS scheduling |
| ML inference | ONNX Runtime 1.17 | Pre-compiled graph, ~2µs, no Python at inference |
| ML training | LightGBM 4.x | GBDT regression on 52 features |
| OOS validation | 6-fold walk-forward | Industry-standard overfitting prevention |
| Live feed | Binance WebSocket | Public, no API key, aggTrade + bookTicker |
| Bindings | pybind11 2.12 | Zero-copy NumPy, C++ ↔ Python |
| Backend API | FastAPI + Uvicorn | Async WebSocket streaming |
| Frontend | React + Vite | Real-time telemetry dashboard |
| Build | CMake 3.20 | Cross-platform, optional ONNX |
