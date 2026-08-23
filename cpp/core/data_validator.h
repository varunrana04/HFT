#pragma once
/**
 * @file data_validator.h
 * @brief Comprehensive data quality validation for market data.
 *
 * This is the GATEKEEPER — no tick enters the engine without passing
 * through validation. Every piece of data is checked for:
 *   - Timestamp sanity (not stale, not in the future, not out of order)
 *   - Sequence continuity (no gaps, no duplicates, no reversals)
 *   - Price reasonableness (within N% of last known price)
 *   - Quantity validity (positive, within sane bounds)
 *   - Book integrity (not crossed, sufficient depth)
 *   - Duplicate detection (same sequence number seen twice)
 *
 * Philosophy: It's better to REJECT a valid tick than to ACCEPT a corrupt one.
 * Corrupt data causes phantom signals that look real in backtests but
 * generate catastrophic losses in live trading.
 *
 * Usage:
 *   DataValidator validator(config);
 *   DataQuality result = validator.validate_trade(trade);
 *   if (result != DataQuality::VALID) { reject(trade); log(result); }
 */

#include "types.h"
#include <cstdint>
#include <cmath>

namespace hft {

/**
 * @brief Configuration for data validation thresholds.
 *
 * All thresholds are intentionally conservative (strict) by default.
 * Better to false-positive (reject good data) than false-negative
 * (accept bad data that corrupts signals).
 */
struct DataValidationConfig {
    /// Maximum age of a tick before it's considered stale (nanoseconds)
    /// Default: 5 seconds — anything older is likely a replay/glitch
    int64_t max_staleness_ns = 5'000'000'000LL;

    /// Maximum allowed timestamp jump into the "future" (nanoseconds)
    /// Default: 1 second — clock skew tolerance
    int64_t max_future_ns = 1'000'000'000LL;

    /// Maximum allowed price change per tick as a fraction
    /// Default: 0.05 (5%) — catches fat-finger errors and data corruption
    double max_price_change_pct = 0.05;

    /// Maximum allowed quantity per order/trade (fixed-point)
    /// Default: 1B units — catches data encoding errors
    int64_t max_quantity = 1'000'000'000LL * QTY_SCALE;

    /// Minimum expected bid or ask depth levels
    /// Default: 1 — at least best bid and best ask must exist
    int32_t min_book_depth = 1;

    /// Maximum allowed sequence gap before flagging
    /// Default: 10 — small gaps can happen during reconnects
    int64_t max_sequence_gap = 10;

    /// Enable/disable specific checks (for debugging)
    bool check_timestamps = true;
    bool check_sequences  = true;
    bool check_prices     = true;
    bool check_quantities = true;
    bool check_book_state = true;
};

/**
 * @brief Validation statistics — tracks data quality over time.
 *
 * Monitor these counters to detect systemic data issues early.
 * A sudden spike in rejections indicates a feed problem.
 */
struct ValidationStats {
    uint64_t total_ticks_seen     = 0;
    uint64_t valid_ticks          = 0;
    uint64_t stale_timestamps     = 0;
    uint64_t out_of_sequence      = 0;
    uint64_t price_anomalies      = 0;
    uint64_t qty_anomalies        = 0;
    uint64_t crossed_books        = 0;
    uint64_t missing_levels       = 0;
    uint64_t duplicates           = 0;
    uint64_t future_timestamps    = 0;

    /// Acceptance rate as a fraction [0.0, 1.0]
    [[nodiscard]] double acceptance_rate() const noexcept {
        if (total_ticks_seen == 0) return 1.0;
        return static_cast<double>(valid_ticks) /
               static_cast<double>(total_ticks_seen);
    }

    /// Reset all counters
    void reset() noexcept {
        total_ticks_seen = valid_ticks = stale_timestamps = 0;
        out_of_sequence = price_anomalies = qty_anomalies = 0;
        crossed_books = missing_levels = duplicates = future_timestamps = 0;
    }
};

/**
 * @brief Market data gatekeeper — validates every tick before processing.
 *
 * Maintains rolling state (last timestamp, last sequence, last price)
 * to detect anomalies relative to the data stream context.
 */
class DataValidator {
public:
    explicit DataValidator(const DataValidationConfig& config = {}) noexcept;

    /**
     * @brief Validate a trade event.
     * @param trade The trade to validate
     * @param reference_time Current wall-clock time in nanoseconds
     * @return DataQuality::VALID if the trade passes all checks
     */
    DataQuality validate_trade(const Trade& trade,
                               int64_t reference_time) noexcept;

    /**
     * @brief Validate a book snapshot.
     * @param book The book snapshot to validate
     * @param reference_time Current wall-clock time in nanoseconds
     * @return DataQuality::VALID if the book passes all checks
     */
    DataQuality validate_book(const BookSnapshot& book,
                              int64_t reference_time) noexcept;

    /// Get current validation statistics
    [[nodiscard]] const ValidationStats& stats() const noexcept {
        return stats_;
    }

    /// Reset all state (use when switching instruments or after reconnect)
    void reset() noexcept;

    /// Update the reference price (call with a known-good price)
    void set_reference_price(int64_t price) noexcept {
        last_trade_price_ = price;
        has_reference_price_ = true;
    }

private:
    DataValidationConfig config_;
    ValidationStats      stats_;

    // Rolling state for contextual validation
    int64_t last_trade_ts_       = INVALID_TS;
    int64_t last_book_ts_        = INVALID_TS;
    int64_t last_trade_seq_      = -1;
    int64_t last_book_seq_       = -1;
    int64_t last_trade_price_    = INVALID_PRICE;
    bool    has_reference_price_  = false;

    // Internal check methods
    DataQuality check_timestamp(int64_t ts, int64_t reference_time,
                                int64_t& last_ts) noexcept;
    DataQuality check_sequence(int64_t seq, int64_t& last_seq) noexcept;
    DataQuality check_trade_price(int64_t price) noexcept;
    DataQuality check_quantity(int64_t qty) noexcept;
    DataQuality check_book_integrity(const BookSnapshot& book) noexcept;
};

} // namespace hft
