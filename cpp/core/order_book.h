#pragma once
/**
 * @file order_book.h
 * @brief Ultra-low-latency L2 order book engine.
 *
 * Maintains a real-time limit order book (LOB) from price-level updates.
 * Designed for microsecond-level performance:
 *   - Uses sorted arrays (not std::map) for cache locality
 *   - O(1) best bid/ask access via maintained indices
 *   - O(N) insert/delete where N = number of levels (N ≤ 20, so effectively O(1))
 *   - Outputs validated BookSnapshot structs for downstream consumption
 *   - Integrated with DataValidator for real-time quality gating
 *
 * Supports L2 (price-level aggregated) feeds. Price levels are maintained
 * in sorted order: bids descending, asks ascending.
 */

#include "types.h"
#include "data_validator.h"
#include <algorithm>
#include <cstring>

namespace hft {

/**
 * @brief Update to a single price level (from exchange feed).
 *
 * This is the input format: the exchange tells us "at price X,
 * the total quantity is now Y with Z orders". If quantity is 0,
 * the level should be removed.
 */
struct LevelUpdate {
    int64_t timestamp_ns;
    int64_t sequence_num;
    int64_t price;         ///< Fixed-point price
    int64_t quantity;      ///< New total quantity at this price (0 = remove)
    int32_t order_count;   ///< Number of orders (0 if unknown)
    Side    side;          ///< BID or ASK
    uint8_t _pad[3];
};

/**
 * @brief High-performance L2 order book with integrated validation.
 *
 * Usage:
 *   OrderBook book(instrument_id, validator_config);
 *   book.apply_update(update);                    // Apply exchange updates
 *   const BookSnapshot& snap = book.snapshot();   // Get current state
 */
class OrderBook {
public:
    /**
     * @param instrument_id  Unique identifier for this instrument
     * @param val_config     Data validation configuration
     */
    explicit OrderBook(uint32_t instrument_id = 0,
                       const DataValidationConfig& val_config = {}) noexcept;

    /**
     * @brief Apply a single level update from the exchange feed.
     *
     * @param update The price level update
     * @return DataQuality::VALID if the update was accepted and applied
     *
     * If the update fails validation, the book state is NOT modified.
     * This ensures corrupt data never pollutes the book.
     */
    DataQuality apply_update(const LevelUpdate& update) noexcept;

    /**
     * @brief Apply a full snapshot replacement (used for initial sync).
     *
     * Replaces the entire book state with the provided snapshot.
     * Still validates the snapshot before accepting.
     *
     * @param book The complete book snapshot from the exchange
     * @return DataQuality::VALID if accepted
     */
    DataQuality apply_snapshot(const BookSnapshot& book) noexcept;

    /**
     * @brief Apply a trade event (reduces quantity at the traded price).
     *
     * @param trade The trade event
     * @return DataQuality::VALID if accepted
     */
    DataQuality apply_trade(const Trade& trade) noexcept;

    /// Get the current book snapshot (const reference, zero-copy)
    [[nodiscard]] const BookSnapshot& snapshot() const noexcept {
        return snapshot_;
    }

    /// Get the best bid price (INVALID_PRICE if no bids)
    [[nodiscard]] int64_t best_bid() const noexcept {
        return snapshot_.best_bid_price;
    }

    /// Get the best ask price (INVALID_PRICE if no asks)
    [[nodiscard]] int64_t best_ask() const noexcept {
        return snapshot_.best_ask_price;
    }

    /// Get the mid price (INVALID_PRICE if book is incomplete)
    [[nodiscard]] int64_t mid_price() const noexcept {
        return snapshot_.mid_price();
    }

    /// Get the spread (INVALID_PRICE if book is incomplete)
    [[nodiscard]] int64_t spread() const noexcept {
        return snapshot_.spread();
    }

    /// Check if the book is in a valid, tradeable state
    [[nodiscard]] bool is_valid() const noexcept {
        return snapshot_.is_valid();
    }

    /// Get validation statistics
    [[nodiscard]] const ValidationStats& validation_stats() const noexcept {
        return validator_.stats();
    }

    /// Reset the entire book state
    void reset() noexcept;

    /// Set reference price for validation (e.g., from a known-good trade)
    void set_reference_price(int64_t price) noexcept {
        validator_.set_reference_price(price);
    }

private:
    BookSnapshot   snapshot_;       ///< Current book state
    DataValidator  validator_;      ///< Integrated data quality gate
    uint32_t       instrument_id_;  ///< Instrument identifier

    /// Insert or update a bid level, maintaining descending price order
    void upsert_bid(int64_t price, int64_t quantity, int32_t count) noexcept;

    /// Insert or update an ask level, maintaining ascending price order
    void upsert_ask(int64_t price, int64_t quantity, int32_t count) noexcept;

    /// Remove a bid level by price
    void remove_bid(int64_t price) noexcept;

    /// Remove an ask level by price
    void remove_ask(int64_t price) noexcept;

    /// Refresh cached best bid/ask after any modification
    void refresh_top_of_book() noexcept;
};

} // namespace hft
