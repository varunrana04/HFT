/**
 * @file data_validator.cpp
 * @brief Implementation of market data validation logic.
 *
 * Every check is designed to be FAST (no allocations, no branches where
 * possible) and STRICT (false positives preferred over false negatives).
 */

#include "data_validator.h"
#include <cmath>

namespace hft {

DataValidator::DataValidator(const DataValidationConfig& config) noexcept
    : config_(config) {}

void DataValidator::reset() noexcept {
    stats_.reset();
    last_trade_ts_      = INVALID_TS;
    last_book_ts_       = INVALID_TS;
    last_trade_seq_     = -1;
    last_book_seq_      = -1;
    last_trade_price_   = INVALID_PRICE;
    has_reference_price_ = false;
}

// ─── Trade Validation ─────────────────────────────────────────

DataQuality DataValidator::validate_trade(const Trade& trade,
                                          int64_t reference_time) noexcept {
    ++stats_.total_ticks_seen;

    // 1. Timestamp check
    if (config_.check_timestamps) {
        DataQuality ts_result = check_timestamp(
            trade.timestamp_ns, reference_time, last_trade_ts_
        );
        if (ts_result != DataQuality::VALID) return ts_result;
    }

    // 2. Sequence check
    if (config_.check_sequences) {
        DataQuality seq_result = check_sequence(
            trade.sequence_num, last_trade_seq_
        );
        if (seq_result != DataQuality::VALID) return seq_result;
    }

    // 3. Price reasonableness
    if (config_.check_prices) {
        DataQuality price_result = check_trade_price(trade.price);
        if (price_result != DataQuality::VALID) return price_result;
    }

    // 4. Quantity validity
    if (config_.check_quantities) {
        DataQuality qty_result = check_quantity(trade.quantity);
        if (qty_result != DataQuality::VALID) return qty_result;
    }

    // 5. Side must be defined
    if (trade.side == Side::NONE) {
        return DataQuality::PRICE_ANOMALY; // Reuse — side is critical
    }

    // All checks passed — update rolling state
    last_trade_ts_      = trade.timestamp_ns;
    last_trade_seq_     = trade.sequence_num;
    last_trade_price_   = trade.price;
    has_reference_price_ = true;

    ++stats_.valid_ticks;
    return DataQuality::VALID;
}

// ─── Book Validation ──────────────────────────────────────────

DataQuality DataValidator::validate_book(const BookSnapshot& book,
                                         int64_t reference_time) noexcept {
    ++stats_.total_ticks_seen;

    // 1. Timestamp check
    if (config_.check_timestamps) {
        DataQuality ts_result = check_timestamp(
            book.timestamp_ns, reference_time, last_book_ts_
        );
        if (ts_result != DataQuality::VALID) return ts_result;
    }

    // 2. Sequence check
    if (config_.check_sequences) {
        DataQuality seq_result = check_sequence(
            book.sequence_num, last_book_seq_
        );
        if (seq_result != DataQuality::VALID) return seq_result;
    }

    // 3. Book integrity (crossed check, depth check)
    if (config_.check_book_state) {
        DataQuality book_result = check_book_integrity(book);
        if (book_result != DataQuality::VALID) return book_result;
    }

    // 4. Price reasonableness on best bid/ask
    if (config_.check_prices && has_reference_price_) {
        // Check best bid against last known trade price
        DataQuality bid_check = check_trade_price(book.best_bid_price);
        if (bid_check != DataQuality::VALID) return bid_check;

        DataQuality ask_check = check_trade_price(book.best_ask_price);
        if (ask_check != DataQuality::VALID) return ask_check;
    }

    // All checks passed
    last_book_ts_  = book.timestamp_ns;
    last_book_seq_ = book.sequence_num;

    // Update reference price from mid if no trades yet
    if (!has_reference_price_ && book.best_bid_price != INVALID_PRICE
        && book.best_ask_price != INVALID_PRICE) {
        last_trade_price_ = (book.best_bid_price + book.best_ask_price) / 2;
        has_reference_price_ = true;
    }

    ++stats_.valid_ticks;
    return DataQuality::VALID;
}

// ─── Internal Check Methods ──────────────────────────────────

DataQuality DataValidator::check_timestamp(int64_t ts,
                                           int64_t reference_time,
                                           int64_t& last_ts) noexcept {
    // Reject if timestamp is in the future beyond tolerance
    if (reference_time > 0 && ts > reference_time + config_.max_future_ns) {
        ++stats_.future_timestamps;
        return DataQuality::STALE_TIMESTAMP;
    }

    // Reject if timestamp is too old (stale data)
    if (reference_time > 0
        && (reference_time - ts) > config_.max_staleness_ns) {
        ++stats_.stale_timestamps;
        return DataQuality::STALE_TIMESTAMP;
    }

    // Reject if timestamp goes backwards (out-of-order)
    // Allow equal timestamps (multiple events in same nanosecond)
    if (last_ts != INVALID_TS && ts < last_ts) {
        ++stats_.out_of_sequence;
        return DataQuality::OUT_OF_SEQUENCE;
    }

    return DataQuality::VALID;
}

DataQuality DataValidator::check_sequence(int64_t seq,
                                          int64_t& last_seq) noexcept {
    if (last_seq < 0) {
        // First message — accept any sequence
        return DataQuality::VALID;
    }

    // Detect duplicate
    if (seq == last_seq) {
        ++stats_.duplicates;
        return DataQuality::DUPLICATE;
    }

    // Detect reversal (sequence went backwards)
    if (seq < last_seq) {
        ++stats_.out_of_sequence;
        return DataQuality::OUT_OF_SEQUENCE;
    }

    // Detect gap (missing messages)
    if ((seq - last_seq) > config_.max_sequence_gap) {
        ++stats_.out_of_sequence;
        return DataQuality::OUT_OF_SEQUENCE;
    }

    return DataQuality::VALID;
}

DataQuality DataValidator::check_trade_price(int64_t price) noexcept {
    // Price must be positive
    if (price <= 0) {
        ++stats_.price_anomalies;
        return DataQuality::PRICE_ANOMALY;
    }

    // If we have a reference price, check for unreasonable moves
    if (has_reference_price_ && last_trade_price_ > 0) {
        double ref = static_cast<double>(last_trade_price_);
        double cur = static_cast<double>(price);
        double change = std::abs(cur - ref) / ref;

        if (change > config_.max_price_change_pct) {
            ++stats_.price_anomalies;
            return DataQuality::PRICE_ANOMALY;
        }
    }

    return DataQuality::VALID;
}

DataQuality DataValidator::check_quantity(int64_t qty) noexcept {
    if (qty <= 0) {
        ++stats_.qty_anomalies;
        return DataQuality::QTY_ANOMALY;
    }

    if (qty > config_.max_quantity) {
        ++stats_.qty_anomalies;
        return DataQuality::QTY_ANOMALY;
    }

    return DataQuality::VALID;
}

DataQuality DataValidator::check_book_integrity(
    const BookSnapshot& book) noexcept {

    // Must have minimum depth
    if (book.bid_count < config_.min_book_depth
        || book.ask_count < config_.min_book_depth) {
        ++stats_.missing_levels;
        return DataQuality::MISSING_LEVELS;
    }

    // Best bid and ask must be valid
    if (book.best_bid_price == INVALID_PRICE
        || book.best_ask_price == INVALID_PRICE) {
        ++stats_.missing_levels;
        return DataQuality::MISSING_LEVELS;
    }

    // Book must NOT be crossed (best bid < best ask)
    if (book.best_bid_price >= book.best_ask_price) {
        ++stats_.crossed_books;
        return DataQuality::CROSSED_BOOK;
    }

    // Validate that bid levels are in descending price order
    for (int32_t i = 1; i < book.bid_count; ++i) {
        if (book.bids[i].price >= book.bids[i - 1].price) {
            ++stats_.price_anomalies;
            return DataQuality::PRICE_ANOMALY;
        }
    }

    // Validate that ask levels are in ascending price order
    for (int32_t i = 1; i < book.ask_count; ++i) {
        if (book.asks[i].price <= book.asks[i - 1].price) {
            ++stats_.price_anomalies;
            return DataQuality::PRICE_ANOMALY;
        }
    }

    // Validate quantities at each level
    for (int32_t i = 0; i < book.bid_count; ++i) {
        if (book.bids[i].quantity <= 0) {
            ++stats_.qty_anomalies;
            return DataQuality::QTY_ANOMALY;
        }
    }
    for (int32_t i = 0; i < book.ask_count; ++i) {
        if (book.asks[i].quantity <= 0) {
            ++stats_.qty_anomalies;
            return DataQuality::QTY_ANOMALY;
        }
    }

    return DataQuality::VALID;
}

} // namespace hft
