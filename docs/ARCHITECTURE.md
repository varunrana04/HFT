# HFT Engine — Architecture Overview

**Version**: 2.0 | **Last Updated**: August 2026

---

## 1. System Context

```mermaid
graph TB
    subgraph "External"
        BINANCE["Binance API<br/>(Historical Data)"]
        HF["HuggingFace<br/>(Datasets)"]
        MT5E["MT5 Terminal<br/>(Live Demo)"]
        TVE["TradingView<br/>(Alerts)"]
        ALPACA["Alpaca<br/>(Paper Trading)"]
    end

    subgraph "HFT Engine"
        ENGINE["Core C++20 Engine<br/>(Sub-µs Latency)"]
        PYTHON["Python Layer<br/>(Research + Execution)"]
        DOCS["Documentation<br/>(HLD, LLD, API Ref)"]
    end

    BINANCE -->|REST CSV| ENGINE
    HF -->|Parquet/CSV| ENGINE
    ENGINE <-->|pybind11| PYTHON
    PYTHON <-->|Tick Stream| MT5E
    PYTHON <-->|Webhooks| TVE
    PYTHON <-->|REST API| ALPACA
```

---

## 2. Component Architecture

```mermaid
graph LR
    subgraph "Layer 1: Data Ingestion"
        DD["data_downloader.py<br/>Downloads historical data"]
        MD["MarketDataParser<br/>CSV → Trade/BookSnapshot"]
        DV["DataValidator<br/>8 quality checks"]
    end

    subgraph "Layer 2: Core Processing (Hot Path)"
        OB["OrderBook<br/>L2, 20 levels, O(1)"]
        FE["FeatureEngine<br/>6 alpha signals"]
        SC["SignalCombiner<br/>Weighted avg / ML"]
    end

    subgraph "Layer 3: Decision & Execution"
        SE["StrategyEngine<br/>Event loop + PnL"]
        RM["RiskManager<br/>5 pre-trade gates"]
        OM["OrderManager<br/>Create + track orders"]
    end

    subgraph "Layer 4: Output & Analytics"
        BT["Backtester<br/>Equity, Sharpe, DD"]
        MT5["MT5Gateway<br/>Live demo execution"]
        TV["TVWebhook<br/>FastAPI server"]
        VIZ["Visualizations<br/>matplotlib plots"]
    end

    DD --> MD --> DV --> OB --> FE --> SC --> SE
    SE --> RM --> OM
    SE --> BT --> VIZ
    SE --> MT5
    SE --> TV
```

---

## 3. Hot-Path Data Flow (Tick-to-Trade)

This is the critical path that must execute in **< 1 µs**.

```mermaid
sequenceDiagram
    participant Tick as Raw Tick Data
    participant DV as DataValidator
    participant OB as OrderBook
    participant FE as FeatureEngine
    participant SC as SignalCombiner
    participant SE as StrategyEngine
    participant RM as RiskManager
    participant OM as OrderManager

    Tick->>DV: Validate (timestamp, price, seq)
    DV->>OB: Update L2 book
    OB->>FE: BookSnapshot + Trade
    
    Note over FE: Compute 6 signals:<br/>Microprice, OFI, VPIN,<br/>Spread BPS, RealVol, StatArb Z
    
    FE->>SC: FeatureVector (6 doubles)
    SC->>SE: combined_alpha ∈ [-1, 1]
    
    alt |alpha| >= entry_threshold
        SE->>RM: check_order(proposed_order)
        
        alt All 5 gates PASS
            RM->>OM: create_order(side, price, qty)
            OM-->>SE: Order emitted
            Note over SE: Update PnL, metrics, journal
        else Any gate FAILS
            RM-->>SE: RiskVerdict (rejection reason)
            Note over SE: Increment risk_rejections counter
        end
    else |alpha| < exit_threshold AND position != 0
        SE->>RM: check_order(close_order)
        RM->>OM: Close position
    end
```

---

## 4. Infrastructure Layer

### 4.1 Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Stack-Allocated (No Heap on Hot Path)                    │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ MemoryPool   │  │ SPSCQueue    │  │ Ring Buffers  │  │
│  │ Fixed blocks │  │ 2^16 slots   │  │ VPIN: 128     │  │
│  │ O(1) alloc   │  │ Lock-free    │  │ Vol:  4096    │  │
│  │ 0 fragments  │  │ Atomic fences│  │ StatArb: 4096 │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  Cache Line: 64 bytes │ All structs: alignas(64)        │
│  POD Types Only       │ static_assert(trivially_copyable)│
└─────────────────────────────────────────────────────────┘
```

### 4.2 Fixed-Point Number System

```
Price:    int64_t × PRICE_SCALE (10^8)
          100.50 USD → 10,050,000,000

Quantity: int64_t × QTY_SCALE   (10^8)
          0.001 BTC → 100,000

Rationale:
  ✅ Deterministic replay (no FP rounding drift)
  ✅ Exact PnL calculations
  ✅ Compatible with exchange native formats
  ✅ Faster than double on integer ALU
```

---

## 5. Alpha Signal Pipeline

```mermaid
graph TB
    subgraph "Input"
        BOOK["BookSnapshot<br/>(bid/ask levels)"]
        TRADE["Trade<br/>(price, qty, side)"]
    end

    subgraph "Stateless Signals"
        MP["Microprice<br/>Vol-weighted fair value"]
        SP["Spread BPS<br/>Bid-ask spread"]
    end

    subgraph "Stateful Signals (Ring Buffers)"
        OFI["OFI<br/>Order Flow Imbalance<br/>Δ top-of-book"]
        VPIN["VPIN<br/>Informed Trading Prob<br/>128-bucket ring"]
        RV["Realized Vol<br/>Welford's algorithm<br/>4096-tick ring"]
        SAZ["Stat-Arb Z-Score<br/>Mean-reversion<br/>4096-tick ring"]
    end

    subgraph "Combination"
        COMB["SignalCombiner<br/>Σ w_i × signal_i<br/>or ML model"]
        ALPHA["α ∈ [-1, +1]"]
    end

    BOOK --> MP
    BOOK --> SP
    BOOK --> OFI
    TRADE --> VPIN
    TRADE --> RV
    BOOK --> SAZ

    MP --> COMB
    SP --> COMB
    OFI --> COMB
    VPIN --> COMB
    RV --> COMB
    SAZ --> COMB
    COMB --> ALPHA
```

---

## 6. Risk Management Architecture

```mermaid
graph TB
    ORDER["Proposed Order"] --> G1

    subgraph "Sequential Risk Gates"
        G1["Gate 1: Position Limit<br/>|new_pos| ≤ max_position"]
        G2["Gate 2: Drawdown<br/>(peak - current) / peak ≤ 5%"]
        G3["Gate 3: Daily Loss<br/>(day_start - current) / day_start ≤ 3%"]
        G4["Gate 4: Order Size<br/>notional / portfolio ≤ 2%"]
        G5["Gate 5: Circuit Breaker<br/>now > cooldown_until"]
    end

    G1 -->|PASS| G2
    G2 -->|PASS| G3
    G3 -->|PASS| G4
    G4 -->|PASS| G5
    G5 -->|PASS| EMIT["✅ Order Emitted"]

    G1 -->|FAIL| REJ["❌ Rejected + Stats"]
    G2 -->|FAIL| REJ
    G3 -->|FAIL| TRIP["❌ Rejected + Circuit Breaker Trip"]
    G4 -->|FAIL| REJ
    G5 -->|FAIL| REJ

    TRIP -->|60s cooldown| G5
```

---

## 7. Python Integration Architecture

```mermaid
graph LR
    subgraph "C++ (libhft_core.a)"
        CPP_SE["StrategyEngine"]
        CPP_FE["FeatureEngine"]
        CPP_RM["RiskManager"]
        CPP_OB["OrderBook"]
    end

    subgraph "pybind11 Bridge"
        BIND["py_engine.cpp<br/>Zero-copy NumPy<br/>Buffer protocol"]
    end

    subgraph "Python Applications"
        BT["backtest.py<br/>Historical replay<br/>Equity curve + Sharpe"]
        MT5G["mt5_gateway.py<br/>MT5 Demo account<br/>Live tick stream"]
        TVW["tv_webhook.py<br/>FastAPI server<br/>TradingView alerts"]
        TM["train_model.py<br/>LightGBM training<br/>(Phase 4)"]
    end

    CPP_SE <--> BIND
    CPP_FE <--> BIND
    CPP_RM <--> BIND
    CPP_OB <--> BIND
    BIND <--> BT
    BIND <--> MT5G
    BIND <--> TVW
    BIND <--> TM
```

---

## 8. Deployment Modes

| Mode | Entry Point | Data Source | Execution | Output |
|---|---|---|---|---|
| **Backtest** | `python backtest.py --data X.csv` | Binance CSV / HuggingFace | Simulated fills | Equity curve, Sharpe, drawdown |
| **MT5 Demo** | `python mt5_gateway.py --symbol EURUSD` | MT5 tick stream | MT5 `order_send` | Trade log CSV, slippage stats |
| **TV Webhook** | `python tv_webhook.py --port 8080` | TradingView alerts | Simulated / Alpaca Paper | REST API responses |
| **C++ Tests** | `./hft_tests` | Synthetic data | Assertions | Pass/Fail (36+ tests) |
| **Benchmark** | `./hft_bench` | Synthetic ticks | Latency timing | p50/p99 table |

---

## 9. Directory Structure

```
HFT/
├── cpp/
│   ├── core/                    # 20 files — The engine
│   │   ├── types.h              # POD types, fixed-point
│   │   ├── clock.h              # Nanosecond timestamps
│   │   ├── spsc_queue.h         # Lock-free ring buffer
│   │   ├── memory_pool.h        # Zero-alloc block pool
│   │   ├── order_book.h/.cpp    # L2 order book
│   │   ├── market_data.h/.cpp   # Parser
│   │   ├── data_validator.h/.cpp # Validation
│   │   ├── features.h/.cpp      # 6 alpha signals
│   │   ├── signal_combiner.h/.cpp # Signal aggregation
│   │   ├── risk_manager.h/.cpp  # 5 risk gates
│   │   ├── order_manager.h/.cpp # Order lifecycle
│   │   └── strategy_engine.h/.cpp # Central orchestrator
│   ├── tests/                   # 9 files — 36+ unit tests
│   │   ├── test_main.cpp
│   │   ├── test_types.cpp
│   │   ├── test_spsc_queue.cpp
│   │   ├── test_memory_pool.cpp
│   │   ├── test_order_book.cpp
│   │   ├── test_data_validator.cpp
│   │   ├── test_features.cpp
│   │   ├── test_risk_manager.cpp
│   │   └── test_strategy_engine.cpp
│   ├── bindings/
│   │   └── py_engine.cpp        # pybind11 + NumPy
│   └── bench/
│       └── latency_profiler.cpp # Tick-to-trade benchmarks
├── python/
│   ├── data_downloader.py       # Binance data acquisition
│   ├── backtest.py              # Historical replay engine
│   ├── mt5_gateway.py           # MT5 live execution
│   └── tv_webhook.py            # TradingView FastAPI
├── config/
│   └── default_config.yaml      # All tuning parameters
├── docs/
│   ├── SYSTEM_DESIGN.md         # High-Level Design
│   ├── LOW_LEVEL_DESIGN.md      # Implementation details
│   ├── ARCHITECTURE.md          # This file
│   └── API_REFERENCE.md         # Class/method reference
├── CMakeLists.txt               # Build system
├── requirements.txt             # Python dependencies
└── README.md                    # Portfolio presentation
```
