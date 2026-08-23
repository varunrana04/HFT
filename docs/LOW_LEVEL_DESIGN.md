# HFT Engine — Low-Level Design Document (LLD)

**Version**: 3.0 | **Date**: August 2026 | **Author**: Varun Rana

---

## 1. Core Data Types (`types.h`)

### 1.1 Fixed-Point Arithmetic

All prices and quantities use `int64_t` scaled by `10^8` to avoid floating-point non-determinism.

```cpp
static constexpr int64_t PRICE_SCALE = 100'000'000LL;  // 10^8
static constexpr int64_t QTY_SCALE   = 100'000'000LL;

// 62,845.12 USD  →  6'284'512'000'000
// 0.001 BTC      →  100'000
inline int64_t price_to_fixed(double p) { return static_cast<int64_t>(p * PRICE_SCALE + 0.5); }
inline double  fixed_to_price(int64_t f) { return static_cast<double>(f) / PRICE_SCALE; }
```

**Rationale:** IEEE 754 double has ~15-16 significant digits. For HFT, integer arithmetic eliminates rounding errors in PnL and ensures deterministic replay across platforms.

### 1.2 Struct Layout & Alignment

Every hot-path struct is `alignas(64)` (cache-line aligned) with `static_assert` verification:

| Struct | Size (bytes) | Alignment | Trivially Copyable |
|---|---|---|---|
| `PriceLevel` | 24 | natural | ✅ `static_assert` |
| `BookSnapshot` | ~1056 | 64 | ✅ `static_assert` |
| `Trade` | 64 | 64 | ✅ `static_assert` |
| `FeatureVector` | 128 | 64 | ✅ `static_assert` |
| `Order` | 128 | 64 | ✅ `static_assert` |

### 1.3 Enums (uint8_t to minimize struct padding)

```cpp
enum class Side        : uint8_t { BID = 0, ASK = 1, NONE = 255 };
enum class OrderType   : uint8_t { LIMIT = 0, MARKET = 1, IOC = 2 };
enum class OrderState  : uint8_t { NEW, SENT, PARTIAL, FILLED, CANCELLED, REJECTED };
enum class Regime      : uint8_t { NORMAL, HIGH_TOXICITY, LOW_LIQUIDITY, TRENDING };
enum class CombinerMode: uint8_t { WEIGHTED_AVG, ML_MODEL, ONNX_MODEL };
```

---

## 2. Order Book (`order_book.h/.cpp`)

### 2.1 Data Structure

```
BookSnapshot {
    timestamp_ns:  int64_t
    sequence_num:  int64_t
    bids[20]:      PriceLevel[]   // Descending price — best bid at [0]
    asks[20]:      PriceLevel[]   // Ascending price  — best ask at [0]
    bid_count:     int32_t
    ask_count:     int32_t
    best_bid_price:int64_t        // O(1) cached
    best_ask_price:int64_t        // O(1) cached
    best_bid_qty:  int64_t
    best_ask_qty:  int64_t
    quality:       DataQuality
}
```

### 2.2 Update Algorithm — O(1) amortized

```
on_snapshot(levels[]):
    for each level in levels:
        if side == BID: insert into bids[] descending by price
        else:           insert into asks[] ascending by price
    update cached best_bid / best_ask
    validate: best_bid < best_ask (reject crossed books → DataQuality::CROSSED_BOOK)
```

Memory: **No heap allocation** — fixed array of 20 levels per side.

---

## 3. Feature Engine (`features.h/.cpp`) — O(1) Accumulators

All 6 alpha signals are constant-time. No `for` loops over history windows.

### 3.1 Signal Summary

| Signal | Implementation | Complexity |
|---|---|---|
| **Microprice** | `(Qa·Pb + Qb·Pa) / (Qa+Qb)` — direct from current book | O(1) stateless |
| **OFI** | `ΔBID_qty − ΔASK_qty` — cached prev book state | O(1) stateful |
| **VPIN** | 128-bucket ring, `Σ|buy-sell|/total` | O(1) amortized |
| **Spread BPS** | `(ask-bid)/mid × 10000` — direct | O(1) stateless |
| **Realized Vol** | Welford online `std_dev(log_returns)` over 4096-tick ring | O(1) stateful |
| **Stat-Arb Z-Score** | Welford online `(mid-μ)/σ` over 4096-tick ring | O(1) stateful |

### 3.2 Welford's Online Algorithm (Key to O(1))

```cpp
// Called on each new tick — no loop, no window scan
void update_welford(double new_value) {
    ++count_;
    double delta  = new_value - mean_;
    mean_         += delta / count_;
    double delta2 = new_value - mean_;
    M2_           += delta * delta2;    // Running sum of squared deviations
}

double variance() const { return (count_ > 1) ? M2_ / (count_ - 1) : 0.0; }
double std_dev()  const { return std::sqrt(variance()); }
```

**Before optimization:** `std::accumulate` over N=4096 values = **O(N) = 6,200ns**  
**After optimization:** Welford incremental update = **O(1) = 248ns**

### 3.3 Ring Buffer Implementation

```cpp
// All stateful signals use this pattern — zero heap allocation
double buffer_[MAX_SIZE] = {};   // Stack-allocated at compile time
int32_t head_  = 0;
int32_t count_ = 0;

void push(double v) {
    int32_t idx = (head_ + count_) % MAX_SIZE;
    if (count_ < MAX_SIZE) { buffer_[idx] = v; ++count_; }
    else { buffer_[head_] = v; head_ = (head_ + 1) % MAX_SIZE; }
}
```

- No heap: entire buffer lives on the stack
- O(1) push: single modulo + store
- Cache-friendly: contiguous memory

### 3.4 VPIN — Volume-Synchronized Probability of Informed Trading

```
For each trade:
    Classify as buy (ask-side aggressor) or sell
    Accumulate into current bucket (size: 50 BTC)
    When bucket fills:
        store (buy_vol, sell_vol) in 128-slot ring buffer
        advance ring head
VPIN = Σ|buy_i - sell_i| / Σ(buy_i + sell_i)   over last 128 buckets
```

VPIN ∈ [0,1]: values near 1.0 indicate high informed/toxic order flow.

---

## 4. Signal Combiner (`signal_combiner.h/.cpp`)

### 4.1 Three Runtime Modes

```cpp
enum class CombinerMode : uint8_t { WEIGHTED_AVG, ML_MODEL, ONNX_MODEL };

// Mode 1: Weighted average (~20ns)
double combine_weighted(const FeatureVector& fv) {
    double sum = 0;
    for (int i = 0; i < 6; ++i)
        sum += weights_[i] * raw[i];
    return std::clamp(sum, -1.0, 1.0);
}

// Mode 2: 56-byte binary ML weights (~20ns, feature importances from LightGBM)
void load_model(const char* path);   // Reads 6 × float64 from .bin file

// Mode 3: ONNX Runtime full graph (~2.1µs, zero heap allocation at inference)
void load_onnx_model(const char* path, int n_features);
// Pre-allocates float32[52] input tensor on construction
// Calls onnxruntime::Session::Run() directly on hot path
```

### 4.2 ONNX Zero-Allocation Inference

```cpp
// Pre-allocated tensor — set once at startup
std::vector<float> onnx_input_buffer_;  // float32[52], allocated ONCE
Ort::Value input_tensor_ = ...;         // Wraps buffer (zero-copy)

// Hot-path: zero allocation
float predict_onnx(const FeatureVector& fv) {
    // 1. Copy 52 doubles → float32 input buffer (~10ns)
    populate_input(fv);
    // 2. Run model (~2µs post-JIT warmup)
    auto output = session_->Run(run_options_, ..., &input_tensor_, 1, ...);
    return output[0].GetTensorData<float>()[0];
}
```

---

## 5. Risk Manager (`risk_manager.h/.cpp`)

### 5.1 Five Sequential Pre-Trade Gates

```
Gate 1: POSITION_LIMIT
    new_pos = current_position + order_qty (signed)
    REJECT if |new_pos| > max_position_btc

Gate 2: DRAWDOWN_LIMIT
    dd = (peak_equity - current_equity) / peak_equity
    REJECT if dd > 0.05  (5%)

Gate 3: DAILY_LOSS_LIMIT
    daily_loss = (day_start_equity - current_equity) / day_start_equity
    REJECT if daily_loss > 0.03  (3%)
    TRIP circuit breaker if triggered

Gate 4: ORDER_SIZE_LIMIT
    order_pct = order_notional / portfolio_value
    REJECT if order_pct > 0.02  (2%)

Gate 5: CIRCUIT_BREAKER
    REJECT if now_ns < circuit_breaker_until_ns
    (60-second cooldown after any Gate 3 breach)
```

All 5 gates combined: **<50ns**, `noexcept`, zero allocation.

### 5.2 State

```cpp
class RiskManager {
    double peak_equity_;
    double day_start_equity_;
    double current_equity_;
    int64_t circuit_breaker_until_ns_;   // Cooldown end timestamp
    RiskStats stats_;                    // Rejection counters per gate
};
```

---

## 6. Strategy Engine — Maker Limit Order Logic

### 6.1 Event Loop (per tick)

```
on_trade(trade, book):
    1. features_.compute_all(book, trade)  → FeatureVector [~248ns]
    2. combiner_.combine(fv)               → alpha ∈ [-1, +1]
    3. record_return() for online Sharpe

    4. EXIT CHECK:
       if position != 0 AND |alpha| < exit_threshold:
           close_order = build_close_order()
           if risk_mgr_.check_order(close_order) == PASS:
               simulate_maker_fill(close_order)

    5. ENTRY CHECK:
       if |alpha| >= entry_threshold:
           side = (alpha > 0) ? BUY : SELL
           price = (side == BUY) ? best_bid : best_ask  // Maker pricing
           qty = compute_order_size(alpha, portfolio_value)
           if risk_mgr_.check_order(entry_order) == PASS:
               queue_limit_order(side, price, qty)

    6. QUEUE DRAIN CHECK:
       for each pending limit order:
           if trade.price crosses limit_price and trade.qty sufficient:
               simulate_fill(order)  // Capture Maker Rebate
               update_inventory / equity / cash
```

### 6.2 Maker vs Taker Cost Model

```
Taker order (crossing spread):
    cost = spread_bps/2 + taker_fee + market_impact
         = 2bps + 4bps + ~1bps = ~7bps per round-trip

Maker order (limit at best bid/ask):
    cost = -maker_rebate + queue_risk_premium
         = 0bps + ~1bps = ~1bps per round-trip

Net edge improvement: ~6bps per round-trip (x25 for BTC at $63k notional)
```

### 6.3 Position Sizing

```
order_qty = portfolio_value × position_size_pct × alpha_scale / price

alpha_scale = clamp(|alpha| × 10, 0.5, 2.0)
```

Stronger alpha → larger position, clamped to [0.5×, 2×] base.

### 6.4 Online Sharpe Ratio

```cpp
// Per tick — O(1), no stored returns array
void record_return(double equity) {
    double r = (equity - prev_equity_) / prev_equity_;
    sum_r_  += r;
    sum_r2_ += r * r;
    ++count_;
    prev_equity_ = equity;
}

double sharpe() const {
    double mean = sum_r_ / count_;
    double var  = sum_r2_ / count_ - mean * mean;
    return (var > 0) ? (mean / std::sqrt(var)) * std::sqrt(252 * 86400) : 0.0;
}
```

---

## 7. SPSC Queue (`spsc_queue.h`)

Lock-free single-producer single-consumer ring buffer:

```cpp
template<typename T, size_t N = 65536>
class SPSCQueue {
    static_assert((N & (N-1)) == 0, "Must be power of 2");

    alignas(64) std::atomic<size_t> write_pos_{0};
    alignas(64) std::atomic<size_t> read_pos_{0};
    alignas(64) T buffer_[N];

    bool try_push(const T& item) {
        size_t wp   = write_pos_.load(std::memory_order_relaxed);
        size_t next = (wp + 1) & (N - 1);
        if (next == read_pos_.load(std::memory_order_acquire)) return false; // Full
        buffer_[wp] = item;
        write_pos_.store(next, std::memory_order_release);
        return true;
    }

    bool try_pop(T& item) {
        size_t rp = read_pos_.load(std::memory_order_relaxed);
        if (rp == write_pos_.load(std::memory_order_acquire)) return false; // Empty
        item = buffer_[rp];
        read_pos_.store((rp + 1) & (N - 1), std::memory_order_release);
        return true;
    }
};
```

- Producer: `relaxed` load on `write_pos_`, `acquire` load on `read_pos_`, `release` store
- Consumer: `relaxed` load on `read_pos_`, `acquire` load on `write_pos_`, `release` store
- No CAS, no mutex: SPSC = zero contention by design

---

## 8. Memory Pool (`memory_pool.h`)

```cpp
template<typename T, size_t BlockSize = 64>
class MemoryPool {
    struct Block { alignas(T) uint8_t data[sizeof(T)]; Block* next; };
    Block blocks_[BlockSize];   // Pre-allocated on construction
    Block* free_list_;

    T* allocate() {
        if (!free_list_) return nullptr;
        Block* b = free_list_;
        free_list_ = b->next;
        return reinterpret_cast<T*>(b->data);
    }

    void deallocate(T* ptr) {
        Block* b = reinterpret_cast<Block*>(ptr);
        b->next  = free_list_;
        free_list_ = b;
    }
};
```

- O(1) alloc/dealloc: single pointer manipulation
- Zero fragmentation: fixed-size blocks
- Stack-allocated: entire pool lives as class member, no heap

---

## 9. Python Bindings (`py_engine.cpp`)

### 9.1 Zero-Copy Interface

```cpp
py::class_<FeatureVector>(m, "FeatureVector")
    .def_readonly("microprice",     &FeatureVector::microprice)
    .def_readonly("ofi",            &FeatureVector::ofi)
    .def_readonly("vpin",           &FeatureVector::vpin)
    .def_readonly("spread_bps",     &FeatureVector::spread_bps)
    .def_readonly("realized_vol",   &FeatureVector::realized_vol)
    .def_readonly("stat_arb_zscore",&FeatureVector::stat_arb_zscore);

// Zero-copy NumPy array wrapping
m.def("features_to_numpy", [](const FeatureVector& fv) {
    return py::array_t<double>({6}, {sizeof(double)},
        &fv.microprice, py::capsule(&fv));
});
```

### 9.2 Exposed API

| C++ Class | Python Name | Key Methods |
|---|---|---|
| `StrategyEngine` | `hft_engine.StrategyEngine` | `on_trade()`, `metrics()`, `reset()` |
| `FeatureEngine` | `hft_engine.FeatureEngine` | `compute_all(book, trade)` |
| `SignalCombiner` | `hft_engine.SignalCombiner` | `combine(fv)`, `set_mode()`, `load_model()`, `load_onnx_model()` |
| `RiskManager` | `hft_engine.RiskManager` | `check_order()`, `update_equity()`, `stats()` |
| `OrderBook` | `hft_engine.OrderBook` | `on_snapshot()`, `snapshot()` |

---

## 10. FastAPI WebSocket Backend (`live_paper_trade.py`)

```
Endpoints:
    GET  /api/trade_status      → { is_trading, equity, pnl, inventory }
    POST /api/start_trading     → Enables live order queue
    POST /api/stop_trading      → Disables order queue, holds position
    WS   /ws                   → Streams TelemetryData every 100ms

TelemetryData {
    timestamp: float
    is_trading: bool
    equity: float
    cash: float
    inventory: float         # BTC position
    pnl: float
    total_trades: int
    win_rate: float
    sharpe: float
    alpha: float             # Current combined α signal
    bid: float               # Best bid from live book
    ask: float               # Best ask from live book
    features: FeatureVector  # All 6 signals
}
```

---

## 11. File Inventory

| # | File | Language | Purpose |
|---|---|---|---|
| 1 | `types.h` | C++ | POD structs, fixed-point, enums |
| 2 | `clock.h` | C++ | Nanosecond wall-clock |
| 3 | `spsc_queue.h` | C++ | Lock-free ring buffer |
| 4 | `memory_pool.h` | C++ | Pre-allocated block pool |
| 5-6 | `order_book.h/.cpp` | C++ | L2 order book, O(1) update |
| 7-8 | `market_data.h/.cpp` | C++ | JSON → POD parser |
| 9-10 | `data_validator.h/.cpp` | C++ | 8 data quality checks |
| 11-12 | `features.h/.cpp` | C++ | 6 O(1) alpha signal accumulators |
| 13-14 | `signal_combiner.h/.cpp` | C++ | 3-mode combiner, ONNX bridge |
| 15-16 | `risk_manager.h/.cpp` | C++ | 5 pre-trade gates + circuit breaker |
| 17-18 | `order_manager.h/.cpp` | C++ | Order lifecycle tracking |
| 19-20 | `strategy_engine.h/.cpp` | C++ | Maker queue, PnL, online Sharpe |
| 21 | `py_engine.cpp` | C++ | pybind11 zero-copy bindings |
| 22 | `live_paper_trade.py` | Python | FastAPI WebSocket server |
| 23 | `backtest.py` | Python | Historical CSV replay |
| 24 | `train_model.py` | Python | LightGBM → ONNX export |
| 25 | `walk_forward.py` | Python | 6-fold rolling OOS validation |
| 26-34 | `test_*.cpp` (9 files) | C++ | 36+ GoogleTest unit tests |
| 35 | `dashboard/src/App.tsx` | TypeScript/React | Real-time trading dashboard |
