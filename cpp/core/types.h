#pragma once
/**
 * @file types.h
 * @brief Core type definitions for the HFT engine.
 *
 * All types on the hot path are Plain Old Data (POD), cache-line aligned,
 * and use fixed-point arithmetic to avoid floating-point non-determinism.
 *
 * Fixed-point convention:
 *   Price:    int64_t, scaled by PRICE_SCALE (10^8) — supports 8 decimal places
 *   Quantity: int64_t, scaled by QTY_SCALE   (10^8) — supports 8 decimal places
 *
 * Cache alignment:
 *   All hot structs use alignas(CACHE_LINE_SIZE) to prevent false sharing
 *   between CPU cores and maximize L1/L2 cache utilization.
 */

#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace hft {

// ─── Hardware Constants ───────────────────────────────────────
/// CPU cache line size in bytes (64 bytes on x86-64, ARM Cortex-A)
static constexpr size_t CACHE_LINE_SIZE = 64;

// ─── Fixed-Point Scaling ──────────────────────────────────────
/// Price scale factor: 1 unit = 10^-8 of base currency
static constexpr int64_t PRICE_SCALE = 100'000'000LL;

/// Quantity scale factor: 1 unit = 10^-8 of base asset
static constexpr int64_t QTY_SCALE = 100'000'000LL;

// ─── Order Book Limits ────────────────────────────────────────
/// Maximum number of price levels tracked per side (bid/ask)
static constexpr int32_t MAX_BOOK_LEVELS = 20;

/// Maximum number of instruments tracked simultaneously
static constexpr int32_t MAX_INSTRUMENTS = 16;

// ─── SPSC Queue ───────────────────────────────────────────────
/// Default SPSC queue capacity (must be power of 2)
static constexpr size_t DEFAULT_QUEUE_CAPACITY = 65536; // 2^16

// ─── Sentinel Values ──────────────────────────────────────────
static constexpr int64_t INVALID_PRICE = std::numeric_limits<int64_t>::min();
static constexpr int64_t INVALID_QTY   = -1;
static constexpr int64_t INVALID_TS    = -1;

// ─── Enums ────────────────────────────────────────────────────

/// Side of an order or trade
enum class Side : uint8_t {
    BID  = 0,  ///< Buy side
    ASK  = 1,  ///< Sell side
    NONE = 255
};

/// Order type
enum class OrderType : uint8_t {
    LIMIT  = 0,
    MARKET = 1,
    IOC    = 2,  ///< Immediate-Or-Cancel
    FOK    = 3   ///< Fill-Or-Kill
};

/// Order lifecycle state
enum class OrderState : uint8_t {
    NEW       = 0,
    SENT      = 1,
    PARTIAL   = 2,
    FILLED    = 3,
    CANCELLED = 4,
    REJECTED  = 5
};

/// Market regime classification
enum class Regime : uint8_t {
    NORMAL        = 0,  ///< Healthy liquidity, normal spread
    HIGH_TOXICITY = 1,  ///< High VPIN, informed flow detected
    LOW_LIQUIDITY = 2,  ///< Wide spreads, thin book
    TRENDING      = 3,  ///< Persistent directional movement
    UNKNOWN       = 255
};

/// Data validation status
enum class DataQuality : uint8_t {
    VALID           = 0,
    STALE_TIMESTAMP = 1,  ///< Timestamp older than threshold
    OUT_OF_SEQUENCE = 2,  ///< Sequence number gap or reversal
    PRICE_ANOMALY   = 3,  ///< Price outside acceptable range
    QTY_ANOMALY     = 4,  ///< Negative or absurdly large quantity
    CROSSED_BOOK    = 5,  ///< Best bid >= best ask (corrupted)
    MISSING_LEVELS  = 6,  ///< Fewer levels than expected
    DUPLICATE       = 7,  ///< Duplicate message detected
    INVALID_SYMBOL  = 8   ///< Unrecognized instrument
};

// ─── Core Data Structures ─────────────────────────────────────

/**
 * @brief A single price level in the order book.
 *
 * Designed to fit snugly within cache lines when arrayed.
 * 24 bytes per level × 20 levels = 480 bytes per side.
 */
struct PriceLevel {
    int64_t price;        ///< Fixed-point price (× PRICE_SCALE)
    int64_t quantity;     ///< Total quantity at this level (× QTY_SCALE)
    int32_t order_count;  ///< Number of individual orders
    int32_t _pad;         ///< Padding for 8-byte alignment

    /// Check if this level contains valid data
    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return price != INVALID_PRICE && quantity > 0;
    }
};

// Verify PriceLevel is trivially copyable (required for lock-free ops)
static_assert(std::is_trivially_copyable_v<PriceLevel>,
    "PriceLevel must be trivially copyable for lock-free operations");
static_assert(sizeof(PriceLevel) == 24,
    "PriceLevel size must be exactly 24 bytes");

/**
 * @brief Complete snapshot of the order book at a point in time.
 *
 * Cache-line aligned to prevent false sharing when accessed from
 * multiple threads (e.g., producer writes, consumer reads via SPSC).
 */
struct BookSnapshot {
    int64_t    timestamp_ns;                ///< Nanosecond-precision timestamp
    int64_t    sequence_num;                ///< Exchange sequence number
    PriceLevel bids[MAX_BOOK_LEVELS];       ///< Bid levels (descending price)
    PriceLevel asks[MAX_BOOK_LEVELS];       ///< Ask levels (ascending price)
    int32_t    bid_count;                   ///< Actual number of bid levels
    int32_t    ask_count;                   ///< Actual number of ask levels
    int64_t    best_bid_price;              ///< Cached best bid for O(1) access
    int64_t    best_ask_price;              ///< Cached best ask for O(1) access
    int64_t    best_bid_qty;                ///< Quantity at best bid
    int64_t    best_ask_qty;                ///< Quantity at best ask
    uint32_t   instrument_id;              ///< Instrument identifier
    DataQuality quality;                    ///< Data validation status
    uint8_t    _pad[3];                     ///< Padding

    /// Compute mid price in fixed-point (overflow-safe)
    [[nodiscard]] constexpr int64_t mid_price() const noexcept {
        if (best_bid_price == INVALID_PRICE || best_ask_price == INVALID_PRICE) {
            return INVALID_PRICE;
        }
        // Use subtraction form to prevent int64 overflow on pathological data
        return best_bid_price + (best_ask_price - best_bid_price) / 2;
    }

    /// Compute spread in fixed-point
    [[nodiscard]] constexpr int64_t spread() const noexcept {
        if (best_bid_price == INVALID_PRICE || best_ask_price == INVALID_PRICE) {
            return INVALID_PRICE;
        }
        return best_ask_price - best_bid_price;
    }

    /// Check if the book is in a valid state (not crossed)
    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return quality == DataQuality::VALID
            && best_bid_price != INVALID_PRICE
            && best_ask_price != INVALID_PRICE
            && best_bid_price < best_ask_price
            && bid_count > 0
            && ask_count > 0;
    }
};

static_assert(std::is_trivially_copyable_v<BookSnapshot>,
    "BookSnapshot must be trivially copyable for lock-free operations");

/**
 * @brief A single trade/tick event.
 */
struct Trade {
    int64_t  timestamp_ns;   ///< Nanosecond timestamp
    int64_t  sequence_num;   ///< Exchange sequence number
    int64_t  price;          ///< Trade price (fixed-point)
    int64_t  quantity;       ///< Trade quantity (fixed-point)
    uint32_t instrument_id;  ///< Instrument identifier
    Side     side;           ///< Aggressor side (BID = buyer-initiated)
    DataQuality quality;     ///< Data validation status
    uint8_t  _pad[2];

    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return quality == DataQuality::VALID
            && price > 0
            && quantity > 0
            && side != Side::NONE;
    }
};

static_assert(std::is_trivially_copyable_v<Trade>,
    "Trade must be trivially copyable for lock-free operations");

/**
 * @brief Feature vector output — all alpha signals in one struct.
 *
 * Produced by the C++ feature engine, consumed by the signal combiner
 * and optionally exported to Python (via pybind11 zero-copy to NumPy).
 * Uses double for features since they feed into statistical models.
 */
struct FeatureVector {
    int64_t timestamp_ns;        ///< Timestamp of the source data
    double  microprice;          ///< Volume-weighted fair value
    double  ofi;                 ///< Order Flow Imbalance
    double  vpin;                ///< Volume-synchronized toxicity [0,1]
    double  spread_bps;          ///< Bid-ask spread in basis points
    double  realized_vol;        ///< Tick-level realized volatility
    double  stat_arb_zscore;     ///< Z-score of spread (pairs trading)
    double  vrp;                 ///< Volatility Risk Premium
    
    // ── Engineered Features ──
    double  obi;                 ///< Order Book Imbalance
    double  trade_imbalance;     ///< Buy/Sell trade volume imbalance
    double  microprice_z10;      ///< 10-tick rolling Z-score of microprice
    double  microprice_z50;      ///< 50-tick rolling Z-score of microprice
    double  ofi_z10;             ///< 10-tick rolling Z-score of OFI
    double  ofi_z50;             ///< 50-tick rolling Z-score of OFI
    double  obi_z10;             ///< 10-tick rolling Z-score of OBI
    double  obi_z50;             ///< 50-tick rolling Z-score of OBI
    
    double  combined_alpha;      ///< Weighted signal combination [-1, 1]
    Regime  regime;              ///< Current market regime
    uint8_t _pad[7];

    [[nodiscard]] constexpr bool has_valid_alpha() const noexcept {
        return combined_alpha >= -1.0 && combined_alpha <= 1.0;
    }
};

static_assert(std::is_trivially_copyable_v<FeatureVector>,
    "FeatureVector must be trivially copyable for lock-free operations");

/**
 * @brief Order representation sent to the exchange.
 */
struct Order {
    int64_t    timestamp_ns;      ///< Order creation time
    int64_t    price;             ///< Limit price (fixed-point), 0 for market
    int64_t    quantity;          ///< Order quantity (fixed-point)
    int64_t    filled_quantity;   ///< Quantity filled so far
    int64_t    avg_fill_price;    ///< Volume-weighted avg fill price
    int64_t    expected_price;    ///< Price at signal time (for slippage calc)
    uint64_t   order_id;          ///< Unique order identifier
    uint32_t   instrument_id;     ///< Instrument
    Side       side;              ///< BID or ASK
    OrderType  type;              ///< LIMIT, MARKET, IOC, FOK
    OrderState state;             ///< Current lifecycle state
    uint8_t    _pad;

    /// Compute slippage in fixed-point ticks
    [[nodiscard]] constexpr int64_t slippage() const noexcept {
        if (filled_quantity == 0 || avg_fill_price == 0) return 0;
        if (side == Side::BID) {
            return avg_fill_price - expected_price;  // Positive = unfavorable
        }
        return expected_price - avg_fill_price;  // Positive = unfavorable
    }

    /// Check if order is fully filled
    [[nodiscard]] constexpr bool is_filled() const noexcept {
        return state == OrderState::FILLED;
    }

    /// Check if order is terminal (no more state transitions expected)
    [[nodiscard]] constexpr bool is_terminal() const noexcept {
        return state == OrderState::FILLED
            || state == OrderState::CANCELLED
            || state == OrderState::REJECTED;
    }
};

static_assert(std::is_trivially_copyable_v<Order>,
    "Order must be trivially copyable for lock-free operations");

// ─── Utility Functions ────────────────────────────────────────

/// Convert a floating-point price to fixed-point
[[nodiscard]] constexpr int64_t price_to_fixed(double price) noexcept {
    return static_cast<int64_t>(price * static_cast<double>(PRICE_SCALE));
}

/// Convert a fixed-point price back to floating-point
[[nodiscard]] constexpr double fixed_to_price(int64_t fixed) noexcept {
    return static_cast<double>(fixed) / static_cast<double>(PRICE_SCALE);
}

/// Convert a floating-point quantity to fixed-point
[[nodiscard]] constexpr int64_t qty_to_fixed(double qty) noexcept {
    return static_cast<int64_t>(qty * static_cast<double>(QTY_SCALE));
}

/// Convert a fixed-point quantity back to floating-point
[[nodiscard]] constexpr double fixed_to_qty(int64_t fixed) noexcept {
    return static_cast<double>(fixed) / static_cast<double>(QTY_SCALE);
}

} // namespace hft
