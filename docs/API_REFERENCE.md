# HFT Engine — API Reference

**Version**: 2.0 | **Last Updated**: August 2026

---

## Table of Contents

1. [Core Types (`types.h`)](#1-core-types)
2. [Clock (`clock.h`)](#2-clock)
3. [SPSC Queue (`spsc_queue.h`)](#3-spsc-queue)
4. [Memory Pool (`memory_pool.h`)](#4-memory-pool)
5. [Order Book (`order_book.h`)](#5-order-book)
6. [Market Data Parser (`market_data.h`)](#6-market-data-parser)
7. [Data Validator (`data_validator.h`)](#7-data-validator)
8. [Feature Engine (`features.h`)](#8-feature-engine)
9. [Signal Combiner (`signal_combiner.h`)](#9-signal-combiner)
10. [Risk Manager (`risk_manager.h`)](#10-risk-manager)
11. [Order Manager (`order_manager.h`)](#11-order-manager)
12. [Strategy Engine (`strategy_engine.h`)](#12-strategy-engine)
13. [Python Modules](#13-python-modules)

---

## 1. Core Types

**Header**: `cpp/core/types.h`  
**Namespace**: `hft`

### Constants

| Name | Type | Value | Description |
|---|---|---|---|
| `CACHE_LINE_SIZE` | `size_t` | `64` | CPU cache line size in bytes |
| `PRICE_SCALE` | `int64_t` | `100'000'000` | Fixed-point price multiplier (10^8) |
| `QTY_SCALE` | `int64_t` | `100'000'000` | Fixed-point quantity multiplier (10^8) |
| `MAX_BOOK_LEVELS` | `int32_t` | `20` | Price levels per book side |
| `MAX_INSTRUMENTS` | `int32_t` | `16` | Max concurrent instruments |
| `DEFAULT_QUEUE_CAPACITY` | `size_t` | `65536` | SPSC queue slots (2^16) |
| `INVALID_PRICE` | `int64_t` | `INT64_MIN` | Sentinel for invalid prices |

### Conversion Functions

```cpp
int64_t price_to_fixed(double price);     // 100.50 → 10050000000
double  fixed_to_price(int64_t fixed);    // 10050000000 → 100.50
int64_t qty_to_fixed(double qty);         // 0.001 → 100000
double  fixed_to_qty(int64_t fixed);      // 100000 → 0.001
```

### Enumerations

#### `Side : uint8_t`
| Value | Code | Description |
|---|---|---|
| `BID` | `0` | Buy side |
| `ASK` | `1` | Sell side |
| `NONE` | `255` | Unspecified |

#### `OrderType : uint8_t`
| Value | Description |
|---|---|
| `LIMIT` | Limit order |
| `MARKET` | Market order |
| `IOC` | Immediate-Or-Cancel |
| `FOK` | Fill-Or-Kill |

#### `OrderState : uint8_t`
| Value | Description |
|---|---|
| `NEW` | Created, not yet sent |
| `SENT` | Sent to exchange |
| `PARTIAL` | Partially filled |
| `FILLED` | Fully filled |
| `CANCELLED` | Cancelled |
| `REJECTED` | Rejected by exchange/risk |

#### `Regime : uint8_t`
| Value | Description |
|---|---|
| `NORMAL` | Standard market conditions |
| `HIGH_TOXICITY` | High informed flow (VPIN > threshold) |
| `LOW_LIQUIDITY` | Wide spreads, thin book |
| `TRENDING` | Strong directional move |

#### `DataQuality : uint8_t`
| Value | Description |
|---|---|
| `VALID` | Data passed all checks |
| `STALE_TIMESTAMP` | Timestamp too old |
| `OUT_OF_SEQUENCE` | Sequence gap detected |
| `INVALID_PRICE` | Price out of range |
| `CROSSED_BOOK` | bid ≥ ask |
| `ZERO_QUANTITY` | Quantity = 0 |
| `DUPLICATE` | Duplicate sequence |
| `FUTURE_TIMESTAMP` | Timestamp in the future |

### Data Structures

#### `PriceLevel`
```cpp
struct PriceLevel {
    int64_t price;        // Fixed-point price
    int64_t quantity;     // Fixed-point aggregated quantity
    int32_t order_count;  // Number of orders at this level
};
```

#### `BookSnapshot` — `alignas(64)`
```cpp
struct alignas(CACHE_LINE_SIZE) BookSnapshot {
    int64_t    timestamp_ns;
    int64_t    sequence_num;
    PriceLevel bids[MAX_BOOK_LEVELS];
    PriceLevel asks[MAX_BOOK_LEVELS];
    int32_t    bid_count, ask_count;
    int64_t    best_bid_price, best_ask_price;
    int64_t    best_bid_qty, best_ask_qty;
    DataQuality quality;

    bool    is_valid() const noexcept;
    int64_t mid_price() const noexcept;   // (best_bid + best_ask) / 2
    int64_t spread() const noexcept;      // best_ask - best_bid
};
```

#### `Trade` — `alignas(64)`
```cpp
struct alignas(CACHE_LINE_SIZE) Trade {
    int64_t     timestamp_ns;
    int64_t     sequence_num;
    int64_t     price;          // Fixed-point
    int64_t     quantity;       // Fixed-point
    int32_t     instrument_id;
    Side        side;           // Aggressor side
    DataQuality quality;
};
```

#### `FeatureVector` — `alignas(64)`
```cpp
struct alignas(CACHE_LINE_SIZE) FeatureVector {
    double  microprice;       // Volume-weighted fair value
    double  ofi;              // Order flow imbalance
    double  vpin;             // Prob. of informed trading [0,1]
    double  spread_bps;       // Bid-ask spread in basis points
    double  realized_vol;     // Tick-level volatility
    double  stat_arb_zscore;  // Mean-reversion Z-score
    double  combined_alpha;   // Combined signal [-1, +1]
    int64_t timestamp_ns;
    Regime  regime;           // Market regime classification
};
```

#### `Order` — `alignas(64)`
```cpp
struct alignas(CACHE_LINE_SIZE) Order {
    uint64_t   id;
    int64_t    price;            // Fixed-point
    int64_t    quantity;         // Fixed-point
    int64_t    filled_qty;       // Fixed-point
    int64_t    expected_price;   // For slippage calculation
    int64_t    create_time_ns;
    int32_t    instrument_id;
    Side       side;
    OrderType  type;
    OrderState state;
};
```

---

## 2. Clock

**Header**: `cpp/core/clock.h`  
**Namespace**: `hft`

```cpp
int64_t now_ns();          // Current time in nanoseconds (TSC or steady_clock)
int64_t elapsed_ns(int64_t start);  // Nanoseconds since start
```

---

## 3. SPSC Queue

**Header**: `cpp/core/spsc_queue.h`  
**Namespace**: `hft`

```cpp
template<typename T, size_t N = DEFAULT_QUEUE_CAPACITY>
class SPSCQueue {
    bool try_push(const T& item) noexcept;   // Returns false if full
    bool try_pop(T& item) noexcept;          // Returns false if empty
    size_t size() const noexcept;            // Approximate size
    bool empty() const noexcept;
    void reset() noexcept;                   // Clear all elements
};
```

**Requirements**: `T` must be trivially copyable. `N` must be power of 2.

---

## 4. Memory Pool

**Header**: `cpp/core/memory_pool.h`  
**Namespace**: `hft`

```cpp
template<typename T, size_t BlockSize = 64>
class MemoryPool {
    T*   allocate() noexcept;       // O(1), returns nullptr if exhausted
    void deallocate(T* ptr) noexcept;  // O(1), returns block to free list
    size_t available() const noexcept;
    void reset() noexcept;          // Return all blocks to free list
};
```

---

## 5. Order Book

**Header**: `cpp/core/order_book.h`  
**Namespace**: `hft`

```cpp
class OrderBook {
public:
    OrderBook() noexcept;

    // Update book with new price levels
    void on_snapshot(const PriceLevel* bids, int bid_count,
                     const PriceLevel* asks, int ask_count,
                     int64_t timestamp_ns, int64_t sequence_num) noexcept;

    // Access current state
    const BookSnapshot& snapshot() const noexcept;

    // Reset to empty
    void reset() noexcept;
};
```

---

## 6. Market Data Parser

**Header**: `cpp/core/market_data.h`  
**Namespace**: `hft`

```cpp
class MarketDataParser {
public:
    // Parse a CSV line into a Trade struct
    bool parse_trade(const char* line, Trade& out) noexcept;

    // Parse a CSV line into book levels
    bool parse_book_update(const char* line,
                           PriceLevel* bids, int& bid_count,
                           PriceLevel* asks, int& ask_count) noexcept;
};
```

---

## 7. Data Validator

**Header**: `cpp/core/data_validator.h`  
**Namespace**: `hft`

```cpp
struct ValidationConfig {
    int64_t max_staleness_ns;      // Default: 5 seconds
    int64_t max_future_ns;         // Default: 1 second
    double  max_price_change_pct;  // Default: 0.05 (5%)
    int64_t max_quantity;          // Default: 1B
    int32_t min_book_depth;        // Default: 1
    int64_t max_sequence_gap;      // Default: 10
};

class DataValidator {
public:
    explicit DataValidator(const ValidationConfig& cfg = {}) noexcept;

    DataQuality validate_trade(const Trade& trade) noexcept;
    DataQuality validate_book(const BookSnapshot& book) noexcept;

    // Statistics
    const ValidationStats& stats() const noexcept;
    void reset() noexcept;
};
```

---

## 8. Feature Engine

**Header**: `cpp/core/features.h`  
**Namespace**: `hft`

```cpp
struct FeatureConfig {
    double  vpin_bucket_size    = 50.0;    // Volume per VPIN bucket
    int32_t vpin_n_buckets      = 50;      // Rolling bucket window
    int32_t vol_window_ticks    = 100;     // Realized vol window
    double  stat_arb_zscore_entry = 2.0;
    double  stat_arb_zscore_exit  = 0.5;
    int32_t stat_arb_lookback     = 1000;
    int32_t stat_arb_half_life_max = 500;
};

class FeatureEngine {
public:
    explicit FeatureEngine(const FeatureConfig& cfg = {}) noexcept;

    // Compute all 6 signals in one call
    FeatureVector compute_all(const BookSnapshot& book,
                              const Trade& trade) noexcept;

    // Individual signal access
    double compute_microprice(const BookSnapshot& book) const noexcept;
    double compute_ofi(const BookSnapshot& book) noexcept;
    double compute_vpin() const noexcept;
    double compute_spread_bps(const BookSnapshot& book) const noexcept;
    double compute_realized_vol() const noexcept;
    double compute_statarb_zscore() const noexcept;

    // State update (called internally by compute_all)
    void update_vpin(const Trade& trade) noexcept;
    void update_realized_vol(double price) noexcept;
    void update_statarb(double mid_price) noexcept;

    // Regime classification
    Regime classify_regime(double vpin, double spread_bps,
                           double vol, double ofi) const noexcept;

    void reset() noexcept;
};
```

---

## 9. Signal Combiner

**Header**: `cpp/core/signal_combiner.h`  
**Namespace**: `hft`

```cpp
class SignalCombiner {
public:
    SignalCombiner() noexcept;  // Default: equal weights (1/6 each)

    // Weighted average of 6 signals → clamped to [-1, +1]
    double combine(const FeatureVector& fv) noexcept;

    // Set custom weights (array of 6 doubles)
    void set_weights(const double* weights, size_t count) noexcept;

    // Phase 4: ML model loading (planned)
    // void load_model(const char* path);
    // double combine_ml(const FeatureVector& fv);
};
```

---

## 10. Risk Manager

**Header**: `cpp/core/risk_manager.h`  
**Namespace**: `hft`

```cpp
enum class RiskVerdict : uint8_t {
    PASS              = 0,
    POSITION_LIMIT    = 1,
    DRAWDOWN_LIMIT    = 2,
    DAILY_LOSS_LIMIT  = 3,
    ORDER_SIZE_LIMIT  = 4,
    CIRCUIT_BREAKER   = 5
};

struct RiskConfig {
    int64_t max_position                 = 100 * QTY_SCALE;
    double  max_drawdown_pct             = 0.05;   // 5%
    double  max_single_order_pct         = 0.02;   // 2%
    double  max_daily_loss_pct           = 0.03;   // 3%
    int64_t circuit_breaker_cooldown_ns  = 60LL * 1'000'000'000LL;  // 60s
};

struct RiskStats {
    uint64_t orders_checked, orders_passed;
    uint64_t rejected_position, rejected_drawdown;
    uint64_t rejected_daily_loss, rejected_order_size;
    uint64_t rejected_circuit_brk, circuit_breaker_trips;

    double pass_rate() const noexcept;
    void reset() noexcept;
};

class RiskManager {
public:
    explicit RiskManager(const RiskConfig& config = {}) noexcept;

    // Full risk check (returns verdict)
    RiskVerdict check_order(const Order& order,
                            int64_t current_position,
                            double current_pnl,
                            double portfolio_value) noexcept;

    // Legacy overload (uses stored equity)
    bool check_order(const Order& order,
                     int64_t current_position,
                     double current_pnl) noexcept;

    // Equity tracking
    void update_equity(double current_equity) noexcept;
    void new_trading_day() noexcept;
    void reset() noexcept;

    // Queries
    const RiskStats& stats() const noexcept;
    bool   is_circuit_breaker_active() const noexcept;
    double current_drawdown() const noexcept;
    double current_daily_loss() const noexcept;
};
```

---

## 11. Order Manager

**Header**: `cpp/core/order_manager.h`  
**Namespace**: `hft`

```cpp
class OrderManager {
public:
    OrderManager() noexcept;

    // Create a new order with auto-incrementing ID
    Order create_order(Side side, int64_t price, int64_t quantity,
                       OrderType type, int64_t expected_price) noexcept;

    // Process a fill event
    void on_fill(Order& order, int64_t fill_price,
                 int64_t fill_qty) noexcept;

    // Cancel an order
    void cancel(Order& order) noexcept;
};
```

---

## 12. Strategy Engine

**Header**: `cpp/core/strategy_engine.h`  
**Namespace**: `hft`

```cpp
enum class EngineMode : uint8_t {
    BACKTEST = 0,  // Deterministic replay
    LIVE     = 1   // Real-time execution
};

struct StrategyConfig {
    double  alpha_entry_threshold = 0.10;   // Min |α| to enter
    double  alpha_exit_threshold  = 0.02;   // |α| below this → exit
    double  position_size_pct    = 0.01;    // Size as % of portfolio
    double  initial_capital      = 100000.0;
    int64_t max_open_orders      = 5;
    bool    allow_short          = true;
};

struct TradeRecord {
    int64_t timestamp_ns;
    int64_t entry_price, exit_price;
    int64_t quantity;
    double  pnl, slippage;
    Side    side;
};

struct PerformanceMetrics {
    double  total_pnl, max_drawdown, peak_equity;
    double  sharpe_ratio, win_rate;
    double  avg_trade_pnl, avg_slippage;
    int64_t total_trades, winning_trades, losing_trades;
    int64_t risk_rejections, signals_generated;

    void update_sharpe() noexcept;  // Annualized (√252)
};

class StrategyEngine {
public:
    StrategyEngine(const StrategyConfig& = {},
                   const FeatureConfig& = {},
                   const RiskConfig& = {}) noexcept;

    // ── Event Handlers ──
    void on_trade(const Trade& trade,
                  const BookSnapshot& book) noexcept;
    void on_book_update(const BookSnapshot& book) noexcept;

    // ── Queries ──
    int64_t position() const noexcept;          // Net position (signed)
    double  unrealized_pnl() const noexcept;
    double  realized_pnl() const noexcept;
    double  equity() const noexcept;            // capital + realized + unrealized

    const FeatureVector&            last_features() const noexcept;
    const PerformanceMetrics&       metrics() const noexcept;
    const RiskStats&                risk_stats() const noexcept;
    const std::vector<TradeRecord>& trade_journal() const noexcept;

    // ── Control ──
    void set_mode(EngineMode mode) noexcept;
    void set_weights(const double* weights, size_t count) noexcept;
    void reset() noexcept;
    void new_trading_day() noexcept;
};
```

---

## 13. Python Modules

### 13.1 `backtest.py`

```
Usage: python backtest.py --data <CSV_PATH> [OPTIONS]

Options:
    --data PATH         Path to Binance CSV (required)
    --capital FLOAT     Initial capital (default: 100000)
    --max-rows INT      Limit rows (default: all)
    --output DIR        Output directory (default: results)
    --threshold FLOAT   Alpha entry threshold (default: 0.10)

Output:
    results/equity_curve.png     Equity curve plot
    results/drawdown.png         Drawdown chart
    results/trade_pnl.png        Trade PnL distribution
    results/backtest_report.md   Summary table
```

### 13.2 `live_paper_trade.py` — FastAPI WebSocket Server

```
Usage: python -m uvicorn python.live_paper_trade:app --reload --port 8000

REST Endpoints:
    GET  /api/trade_status      Returns current trading state, equity, pnl, inventory
    POST /api/start_trading     Activates live Maker limit order queue
    POST /api/stop_trading      Disables order queue (holds open positions)

WebSocket:
    WS   /ws                    Streams TelemetryData JSON at 100ms intervals

TelemetryData schema:
    {
        "timestamp":    float,       # Unix epoch seconds
        "is_trading":   bool,        # Whether engine is active
        "equity":       float,       # Total portfolio value
        "cash":         float,       # USD cash balance
        "inventory":    float,       # BTC position (signed)
        "pnl":          float,       # Realized + unrealized PnL
        "total_trades": int,         # Fill count
        "win_rate":     float,       # Fraction of winning trades
        "sharpe":       float,       # Online annualized Sharpe ratio
        "alpha":        float,       # Current combined alpha signal [-1, +1]
        "bid":          float,       # Best bid price
        "ask":          float,       # Best ask price
        "features":     object       # All 6 raw alpha signals
    }

Requires: pip install fastapi uvicorn websockets
```

### 13.3 `process_local_zips.py`

```
Usage: python python/process_local_zips.py

Merges 12 monthly Binance BTCUSDT ZIP archives into a single CSV.

Options:
    --input-dir DIR     Directory containing ZIP files (default: data/)
    --output PATH       Output CSV path (default: data/BTCUSDT_2024.csv)

Output: data/BTCUSDT_2024.csv  (~30M rows)

Requires: pip install pandas
```

---

## 14. Build Commands

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build everything
cmake --build build --config Release

# Run tests
cd build && ctest --output-on-failure

# Run benchmarks
./build/hft_bench

# Run backtest
python python/backtest.py --data data/BTCUSDT_2024.csv

# Start live paper trading backend
python -m uvicorn python.live_paper_trade:app --reload --port 8000

# Start React dashboard (separate terminal)
cd dashboard && npm run dev
# Navigate to http://localhost:5173
```
