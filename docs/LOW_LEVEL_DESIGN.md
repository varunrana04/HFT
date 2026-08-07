# HFT Engine — Low-Level Design Document (LLD)

**Version**: 2.0  
**Date**: August 2026  
**Author**: Varun  

---

## 1. Core Data Types (`types.h`)

### 1.1 Fixed-Point Arithmetic

All prices and quantities use `int64_t` with a scale factor of `10^8` to avoid floating-point non-determinism on the hot path.

```cpp
static constexpr int64_t PRICE_SCALE = 100'000'000LL;
static constexpr int64_t QTY_SCALE   = 100'000'000LL;

// Conversion functions
int64_t price_to_fixed(double price);   // 100.50 → 10050000000
double  fixed_to_price(int64_t fixed);  // 10050000000 → 100.50
```

**Rationale**: IEEE 754 double has ~15-16 significant decimal digits. For HFT, exact integer arithmetic eliminates rounding errors in PnL calculations and ensures deterministic replay.

### 1.2 Struct Layout & Alignment

Every hot-path struct is `alignas(64)` (cache-line aligned) and verified with `static_assert`:

| Struct | Size (bytes) | Alignment | Trivially Copyable |
|---|---|---|---|
| `PriceLevel` | 24 | natural | ✅ `static_assert` |
| `BookSnapshot` | ~1056 | 64 | ✅ `static_assert` |
| `Trade` | 64 | 64 | ✅ `static_assert` |
| `FeatureVector` | 128 | 64 | ✅ `static_assert` |
| `Order` | 128 | 64 | ✅ `static_assert` |

### 1.3 Enum Design

All enums use `uint8_t` underlying type to minimize struct padding:

```cpp
enum class Side : uint8_t { BID = 0, ASK = 1, NONE = 255 };
enum class OrderType : uint8_t { LIMIT = 0, MARKET = 1, IOC = 2, FOK = 3 };
enum class OrderState : uint8_t { NEW, SENT, PARTIAL, FILLED, CANCELLED, REJECTED };
enum class Regime : uint8_t { NORMAL, HIGH_TOXICITY, LOW_LIQUIDITY, TRENDING };
enum class DataQuality : uint8_t { VALID, STALE_TIMESTAMP, OUT_OF_SEQUENCE, ... };
```

---

## 2. Order Book (`order_book.h/.cpp`)

### 2.1 Data Structure

```
BookSnapshot {
    timestamp_ns: i64
    sequence_num: i64
    bids[20]: PriceLevel[]   // Descending price order
    asks[20]: PriceLevel[]   // Ascending price order
    bid_count, ask_count: i32
    best_bid_price, best_ask_price: i64  // O(1) cached
    best_bid_qty, best_ask_qty: i64
    quality: DataQuality
}
```

### 2.2 Update Algorithm

```
on_snapshot(levels[]):
    for each level in levels:
        if level.side == BID:
            insert into bids[] maintaining descending price order
        else:
            insert into asks[] maintaining ascending price order
    update best_bid/ask cache
    validate: best_bid < best_ask (reject crossed books)
```

- **Complexity**: O(MAX_BOOK_LEVELS) = O(20) = O(1) per update
- **Memory**: No heap allocation — fixed array of 20 levels per side

### 2.3 Key Methods

```cpp
class OrderBook {
    void on_snapshot(const PriceLevel* bids, int bid_count,
                     const PriceLevel* asks, int ask_count,
                     int64_t ts, int64_t seq);
    const BookSnapshot& snapshot() const;
    void reset();
};
```

---

## 3. Feature Engine (`features.h/.cpp`)

### 3.1 Signal Computation Pipeline

```
compute_all(book, trade) → FeatureVector:
    1. microprice    = compute_microprice(book)      // Stateless
    2. ofi           = compute_ofi(book)             // Δ top-of-book
    3. update_vpin(trade); vpin = compute_vpin()      // Rolling buckets
    4. spread_bps    = compute_spread_bps(book)      // Stateless
    5. update_realized_vol(price); vol = compute_rv() // Ring buffer
    6. update_statarb(mid); z = compute_statarb_z()  // Ring buffer
    7. regime = classify_regime(vpin, spread, vol, ofi)
```

### 3.2 Signal Details

#### Microprice (Stateless)
```
microprice = (ask_price × bid_qty + bid_price × ask_qty) / (bid_qty + ask_qty)
```
- Volume-weighted fair value estimate
- More accurate than simple mid-price when book is imbalanced

#### OFI — Order Flow Imbalance (Stateful)
```
Δbid = bid_qty_change × I(bid_price unchanged or improved)
Δask = ask_qty_change × I(ask_price unchanged or improved)
OFI = Δbid - Δask
```
- Tracks net buying/selling pressure
- Requires previous book state (stored internally)

#### VPIN — Volume-Synchronized Probability of Informed Trading (Stateful)
```
Bucket volume = vpin_bucket_size (default: 50 units)
For each trade:
    Classify as buy/sell (by aggressor side)
    Accumulate into current bucket
    When bucket full:
        store (buy_vol, sell_vol) in ring buffer
        advance ring head
VPIN = Σ|buy_i - sell_i| / (Σ(buy_i + sell_i)) over N buckets
```
- Ring buffer: `double[MAX_VPIN_BUCKETS]` (128 max)
- Detects informed trading flow

#### Realized Volatility (Stateful)
```
Ring buffer of last N trade prices (default: 100)
log_returns[i] = ln(price[i] / price[i-1])
realized_vol = Welford's online std_dev(log_returns)
```
- Uses Welford's algorithm for numerically stable online variance
- O(1) per update with ring buffer

#### Stat-Arb Z-Score (Stateful)
```
Ring buffer of last N mid-prices (default: 1000)
mean = rolling_mean(mids)
std  = rolling_std(mids)
z_score = (current_mid - mean) / std
```
- Identifies mean-reverting opportunities
- Entry when |z| > 2.0, exit when |z| < 0.5

### 3.3 Ring Buffer Implementation

All stateful signals use a fixed-size circular buffer pattern:

```cpp
double buffer_[MAX_SIZE] = {};  // Stack-allocated
int32_t head_  = 0;             // Oldest element index
int32_t count_ = 0;             // Number of elements

void push(double value) {
    int32_t idx = (head_ + count_) % MAX_SIZE;
    if (count_ < MAX_SIZE) {
        buffer_[idx] = value;
        ++count_;
    } else {
        buffer_[head_] = value;  // Overwrite oldest
        head_ = (head_ + 1) % MAX_SIZE;
    }
}
```

- **No heap allocation**: Entire buffer on the stack
- **O(1) push/pop**: Single modulo + store
- **Cache friendly**: Contiguous memory access

---

## 4. Signal Combiner (`signal_combiner.h/.cpp`)

### 4.1 Weighted Average Mode (Default)

```cpp
double combine(const FeatureVector& fv) {
    double raw[6] = { fv.microprice, fv.ofi, fv.vpin,
                      fv.spread_bps, fv.realized_vol, fv.stat_arb_zscore };
    double sum = 0;
    for (int i = 0; i < 6; ++i)
        sum += weights_[i] * raw[i];
    return std::clamp(sum, -1.0, 1.0);
}
```

Default weights: `{1/6, 1/6, 1/6, 1/6, 1/6, 1/6}` (equal)

### 4.2 ML Mode (Phase 4 — Planned)

```cpp
// Phase 4: Load LightGBM or ONNX model
void load_model(const char* path);
double combine_ml(const FeatureVector& fv);
```

---

## 5. Risk Manager (`risk_manager.h/.cpp`)

### 5.1 Five Risk Gates

Every order passes through ALL 5 checks sequentially. First failure → immediate reject.

```
Gate 1: POSITION_LIMIT
    new_position = current_position + order_qty (signed)
    REJECT if |new_position| > max_position

Gate 2: DRAWDOWN_LIMIT
    drawdown = (peak_equity - current_equity) / peak_equity
    REJECT if drawdown > max_drawdown_pct (5%)

Gate 3: DAILY_LOSS_LIMIT
    daily_loss = (day_start_equity - current_equity) / day_start_equity
    REJECT if daily_loss > max_daily_loss_pct (3%)

Gate 4: ORDER_SIZE_LIMIT
    order_pct = order_notional / portfolio_value
    REJECT if order_pct > max_single_order_pct (2%)

Gate 5: CIRCUIT_BREAKER
    REJECT if now_ns < circuit_breaker_until_ns
    (60-second cooldown after any breach)
```

### 5.2 State Tracking

```cpp
class RiskManager {
    double peak_equity_;           // High-water mark
    double day_start_equity_;      // Equity at start of day
    double current_equity_;        // Latest equity
    int64_t circuit_breaker_until_ns_;  // Cooldown end time
    RiskStats stats_;              // Rejection counters
};
```

---

## 6. Strategy Engine (`strategy_engine.h/.cpp`)

### 6.1 Event Loop (per tick)

```
on_trade(trade, book):
    1. Update last_mid_price from book
    2. features_.compute_all(book, trade) → FeatureVector
    3. combiner_.combine(fv) → alpha ∈ [-1, 1]
    4. record_return() for Sharpe calculation
    
    5. EXIT CHECK:
       if position != 0 AND |alpha| < exit_threshold:
           Build close order
           risk_mgr_.check_order() → if PASS → simulate_fill(close)
    
    6. ENTRY CHECK:
       if |alpha| >= entry_threshold:
           Determine side (alpha > 0 → BUY, alpha < 0 → SELL)
           compute_order_size(alpha, equity)
           Build entry order at market (take liquidity)
           risk_mgr_.check_order() → if PASS → simulate_fill(entry)
```

### 6.2 Position Sizing

```
order_qty = portfolio_value × position_size_pct × alpha_scale / price

where:
    alpha_scale = clamp(|alpha| × 10, 0.5, 2.0)
```

Stronger signals → larger positions, but clamped to [0.5×, 2×] of base size.

### 6.3 PnL & Metrics Tracking

```
simulate_fill(side, price, quantity):
    if closing a position:
        realized_pnl += (exit_price - entry_price) × quantity  [long]
        realized_pnl += (entry_price - exit_price) × quantity  [short]
    if opening/adding:
        Update VWAP entry price
    
    Journal entry: {timestamp, price, qty, pnl, slippage, side}
    Update metrics: total_trades, win_rate, max_drawdown, Sharpe

unrealized_pnl():
    (current_price - avg_entry) × position  [long]
    (avg_entry - current_price) × position  [short]

equity():
    initial_capital + realized_pnl + unrealized_pnl
```

### 6.4 Sharpe Ratio (Online)

```
For each tick:
    return_i = (equity_i - equity_{i-1}) / equity_{i-1}
    sum_returns += return_i
    sum_returns_sq += return_i²
    count++

Sharpe = (mean_return / std_return) × √252
where:
    mean_return = sum_returns / count
    variance = sum_returns_sq / count - mean²
    std_return = √variance
```

---

## 7. SPSC Queue (`spsc_queue.h`)

### 7.1 Implementation

Lock-free single-producer single-consumer queue using a power-of-2 ring buffer:

```cpp
template<typename T, size_t N = DEFAULT_QUEUE_CAPACITY>
class SPSCQueue {
    static_assert((N & (N-1)) == 0, "Capacity must be power of 2");
    
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> write_pos_{0};
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> read_pos_{0};
    alignas(CACHE_LINE_SIZE) T buffer_[N];

    bool try_push(const T& item) {
        size_t wp = write_pos_.load(relaxed);
        size_t next = (wp + 1) & (N - 1);  // Bitwise mod
        if (next == read_pos_.load(acquire)) return false;  // Full
        buffer_[wp] = item;
        write_pos_.store(next, release);
        return true;
    }

    bool try_pop(T& item) {
        size_t rp = read_pos_.load(relaxed);
        if (rp == write_pos_.load(acquire)) return false;  // Empty
        item = buffer_[rp];
        read_pos_.store((rp + 1) & (N - 1), release);
        return true;
    }
};
```

### 7.2 Memory Ordering

- **Producer**: `relaxed` load on `write_pos_`, `acquire` load on `read_pos_`, `release` store on `write_pos_`
- **Consumer**: `relaxed` load on `read_pos_`, `acquire` load on `write_pos_`, `release` store on `read_pos_`
- **No CAS/mutex needed**: Single producer + single consumer = no contention

---

## 8. Memory Pool (`memory_pool.h`)

### 8.1 Design

```cpp
template<typename T, size_t BlockSize = 64>
class MemoryPool {
    struct Block {
        alignas(T) uint8_t data[sizeof(T)];
        Block* next;
    };
    
    Block blocks_[BlockSize];  // Pre-allocated
    Block* free_list_;         // Singly-linked free list
    
    T* allocate() {
        if (!free_list_) return nullptr;
        Block* b = free_list_;
        free_list_ = b->next;
        return reinterpret_cast<T*>(b->data);
    }
    
    void deallocate(T* ptr) {
        Block* b = reinterpret_cast<Block*>(ptr);
        b->next = free_list_;
        free_list_ = b;
    }
};
```

- **O(1) alloc/dealloc**: Just pointer manipulation on free list
- **Zero fragmentation**: Fixed-size blocks
- **Stack-allocated**: Entire pool lives on the stack (or as class member)

---

## 9. Python Bindings (`py_engine.cpp`)

### 9.1 Zero-Copy NumPy Interface

```cpp
py::class_<FeatureVector>(m, "FeatureVector")
    .def_readonly("microprice", &FeatureVector::microprice)
    .def_readonly("ofi", &FeatureVector::ofi)
    .def_readonly("vpin", &FeatureVector::vpin)
    // ... all fields exposed as read-only Python attributes
    ;

// NumPy array wrapping (zero-copy)
m.def("features_to_numpy", [](const FeatureVector& fv) {
    return py::array_t<double>({6}, {sizeof(double)},
        &fv.microprice, py::capsule(&fv));
});
```

### 9.2 Exposed Classes

| C++ Class | Python Module | Methods |
|---|---|---|
| `FeatureEngine` | `hft_engine.FeatureEngine` | `compute_all(book, trade)` |
| `RiskManager` | `hft_engine.RiskManager` | `check_order(...)` |
| `SignalCombiner` | `hft_engine.SignalCombiner` | `combine(fv)`, `set_weights(...)` |
| `StrategyEngine` | `hft_engine.StrategyEngine` | `on_trade(...)`, `metrics()`, `reset()` |
| `OrderBook` | `hft_engine.OrderBook` | `on_snapshot(...)`, `snapshot()` |

---

## 10. File Reference

### 10.1 Complete File Inventory

| # | File | Language | Lines | Size | Purpose |
|---|---|---|---|---|---|
| 1 | `types.h` | C++ | 294 | 12 KB | Core data types, fixed-point, POD structs |
| 2 | `clock.h` | C++ | ~100 | 4 KB | Nanosecond clock abstraction |
| 3 | `spsc_queue.h` | C++ | ~150 | 6 KB | Lock-free ring buffer |
| 4 | `memory_pool.h` | C++ | ~100 | 4 KB | Pre-allocated block pool |
| 5 | `order_book.h/.cpp` | C++ | ~400 | 16 KB | L2 order book |
| 6 | `market_data.h/.cpp` | C++ | ~250 | 9 KB | Market data parser |
| 7 | `data_validator.h/.cpp` | C++ | ~350 | 15 KB | Data quality validation |
| 8 | `features.h/.cpp` | C++ | ~450 | 18 KB | 6 alpha signal engine |
| 9 | `signal_combiner.h/.cpp` | C++ | ~30 | 1 KB | Signal aggregation |
| 10 | `risk_manager.h/.cpp` | C++ | ~350 | 11 KB | Pre-trade risk gates |
| 11 | `order_manager.h/.cpp` | C++ | ~50 | 2 KB | Order lifecycle |
| 12 | `strategy_engine.h/.cpp` | C++ | ~400 | 22 KB | Central orchestrator |
| 13 | `py_engine.cpp` | C++ | ~200 | 8 KB | pybind11 bindings |
| 14 | `backtest.py` | Python | ~400 | 20 KB | Historical replay |
| 15 | `mt5_gateway.py` | Python | ~350 | 14 KB | MT5 live execution |
| 16 | `tv_webhook.py` | Python | ~350 | 14 KB | TradingView webhook |
| 17 | `data_downloader.py` | Python | ~350 | 15 KB | Data acquisition |
| 18 | `CMakeLists.txt` | CMake | 73 | 3 KB | Build configuration |
| 19–27 | `test_*.cpp` (9 files) | C++ | ~700 | 70 KB | 36+ unit tests |
