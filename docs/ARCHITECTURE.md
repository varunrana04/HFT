# HFT Engine — Architecture Document

**Version**: 3.0 | **Last Updated**: August 2026

---

## 1. Full System Context

```mermaid
graph TB
    subgraph "External Data Sources"
        BINANCE["🔵 Binance WebSocket<br/>(aggTrade + bookTicker)"]
        CSV["📁 Historical CSV<br/>(30M Ticks, Jan–Dec 2024)"]
    end

    subgraph "C++ Core Engine — Zero-Allocation Hot Path (~248ns)"
        DV["DataValidator<br/>(8 quality checks)"]
        OB["OrderBook<br/>(L2, 20 levels, O(1))"]
        FE["FeatureEngine<br/>(6 α signals, O(1) Accumulators)"]
        SC["SignalCombiner"]
        WA["WEIGHTED_AVG<br/>(20ns)"]
        ML["ML_MODEL<br/>(56-byte binary)"]
        ON["ONNX_MODEL<br/>(LightGBM, ~2µs)"]
        RM["RiskManager<br/>(5 Gates, <50ns)"]
        SE["StrategyEngine<br/>(Maker Limit Order Queue)"]
    end

    subgraph "Python Layer"
        PB["pybind11<br/>(Zero-Copy Bridge)"]
        FA["FastAPI + WebSocket<br/>(live_paper_trade.py)"]
        BT["backtest.py<br/>(Historical Replay)"]
        WF["walk_forward.py<br/>(6-Fold OOS)"]
    end

    subgraph "React / Vite Dashboard"
        EC["📈 Equity Curve"]
        PNL["💰 Live PnL / Inventory"]
        OBV["📊 Order Book View"]
        CTL["▶ Start / Stop Trading"]
    end

    BINANCE --> DV
    CSV --> DV
    DV --> OB --> FE --> SC
    SC --> WA
    SC --> ML
    SC --> ON
    WA --> RM
    ML --> RM
    ON --> RM
    RM --> SE
    SE --> PB
    PB --> FA
    PB --> BT
    PB --> WF
    FA -->|"WebSocket JSON 100ms"| EC
    FA --> PNL
    FA --> OBV
    FA --> CTL
```

---

## 2. Layered Architecture Diagram

![Architecture Diagram](architecture_diagram.jpg)

---

## 3. Hot-Path Sequence Diagram (Tick-to-Trade)

```mermaid
sequenceDiagram
    participant WS as Binance WebSocket
    participant DV as DataValidator
    participant OB as OrderBook
    participant FE as FeatureEngine (O1)
    participant SC as SignalCombiner
    participant RM as RiskManager
    participant SE as StrategyEngine
    participant FA as FastAPI WS

    WS->>DV: Raw aggTrade / bookTicker JSON
    DV->>OB: Validated BookSnapshot + Trade
    OB->>FE: Updated L2 snapshot
    
    Note over FE: O(1) Accumulators<br/>Microprice, OFI, VPIN<br/>SpreadBPS, RealVol, StatArb Z<br/>~248ns total
    
    FE->>SC: FeatureVector (6 doubles)
    SC->>RM: combined_alpha ∈ [-1, +1]
    
    alt |α| >= entry_threshold
        RM->>SE: All 5 gates PASS
        SE->>SE: Queue Maker Limit Order @ best_bid/ask
        SE->>FA: Telemetry (equity, pnl, inventory)
        FA-->>FA: Broadcast via WebSocket
    else gate FAIL
        RM-->>SE: Reject + increment counter
    end
```

---

## 4. O(1) Feature Engine Deep-Dive

All 6 signals run in constant time via **running accumulators** — no sliding window loops.

```mermaid
graph LR
    subgraph "Stateless (No Memory)"
        MP["Microprice<br/>(Qa·Pb + Qb·Pa)/(Qa+Qb)"]
        SB["Spread BPS<br/>(ask-bid)/mid × 10000"]
    end

    subgraph "O(1) Stateful (Ring Buffers + Welford)"
        OFI["OFI<br/>ΔBID_qty − ΔASK_qty<br/>prev book cached"]
        VPIN["VPIN<br/>128-bucket ring<br/>|buy-sell|/total"]
        RV["Realized Vol<br/>Welford online std_dev<br/>4096-tick ring"]
        SAZ["Stat-Arb Z<br/>(mid − μ) / σ<br/>Welford 4096-tick ring"]
    end

    subgraph "Normalization"
        ZNORM["Online Z-normalize<br/>(Welford per signal)<br/>→ bounded α ∈ [-1,+1]"]
    end

    MP --> ZNORM
    SB --> ZNORM
    OFI --> ZNORM
    VPIN --> ZNORM
    RV --> ZNORM
    SAZ --> ZNORM
```

**Key insight:** Before this optimization, all 6 signals used `O(N)` loops over the history window on each tick. Replacing with Welford's incremental algorithm reduced feature extraction from **6,200ns → 248ns** (25× speedup).

---

## 5. ML Signal Pipeline

```mermaid
graph LR
    subgraph "Training (Offline Python)"
        RAW["52 Feature Vectors<br/>(base signals + lags<br/>+ cross-interactions)"]
        LGB["LightGBM Regressor<br/>(signed forward return target)"]
        OON["ONNX Export<br/>(skl2onnx)"]
    end

    subgraph "Inference (Online C++)"
        BUF["Pre-allocated float32[52]<br/>(zero heap)"]
        ORT["onnxruntime::Session::Run()<br/>(~2.1µs p50)"]
        PRED["α ∈ [-1, +1]"]
    end

    RAW --> LGB --> OON --> BUF --> ORT --> PRED
```

**Feature engineering:** 6 base signals + rolling Z-scores at [10, 50, 200] ticks + lag-1 values + 4 cross-signal interactions + 4 regime one-hot flags = **52 features**.

**Target:** Signed forward return at horizon N ticks (regression). Avoids 50/50 base-rate problem of classification.

---

## 6. Risk Architecture

```mermaid
graph TB
    ORDER["📋 Proposed Order"] --> G1

    subgraph "5 Sequential Pre-Trade Gates"
        G1["Gate 1: Position Limit<br/>|new_pos| ≤ max_pos"]
        G2["Gate 2: Drawdown<br/>(peak-curr)/peak ≤ 5%"]
        G3["Gate 3: Daily Loss<br/>daily_loss/start ≤ 3%"]
        G4["Gate 4: Order Size<br/>notional/portfolio ≤ 2%"]
        G5["Gate 5: Circuit Breaker<br/>now > cooldown_until"]
    end

    G1 -->|PASS| G2 -->|PASS| G3 -->|PASS| G4 -->|PASS| G5
    G5 -->|ALL PASS| EMIT["✅ Order Emitted"]

    G1 -->|FAIL| REJ["❌ Rejected"]
    G2 -->|FAIL| REJ
    G3 -->|FAIL| TRIP["❌ + 60s Circuit Breaker"]
    G4 -->|FAIL| REJ
    G5 -->|FAIL| REJ
    TRIP --> G5
```

---

## 7. Performance Stats

![Latency Stats Dashboard](latency_stats.jpg)

| Operation | p50 | p99 | Complexity |
|---|---|---|---|
| **Feature Extraction** | **248 ns** | 412 ns | **O(1)** |
| Signal Combine (Weights) | 20 ns | 40 ns | O(1) |
| ONNX ML Inference | 2.1 µs | 4.8 µs | O(M) |
| Risk Gate Check | 35 ns | 60 ns | O(1) |
| Full Hot Path (no ONNX) | ~300 ns | ~520 ns | O(1) |

---

## 8. Deployment Architecture

```mermaid
graph LR
    subgraph "Development"
        SRC["GitHub Repo"]
        CMAKE["CMake Build"]
        TESTS["36+ Unit Tests<br/>GoogleTest"]
    end

    subgraph "Research Pipeline"
        BT["backtest.py<br/>Historical Replay"]
        TR["train_model.py<br/>LightGBM → ONNX"]
        WF["walk_forward.py<br/>6-Fold OOS"]
    end

    subgraph "Live Paper Trading"
        FA["FastAPI Backend<br/>:8000"]
        DASH["React Dashboard<br/>:5173"]
        BINANCE["Binance WS"]
    end

    SRC --> CMAKE --> TESTS
    CMAKE -->|pybind11| BT
    BT --> TR --> WF
    WF -->|best_fold/lgb_model.onnx| FA
    BINANCE -->|WebSocket| FA
    FA -->|WebSocket JSON| DASH
```
