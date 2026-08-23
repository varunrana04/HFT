/**
 * @file order_book.cpp
 * @brief Implementation of the L2 order book engine.
 *
 * Uses memmove for array shifting — this is deliberate. With MAX_LEVELS=20,
 * the arrays are small enough (480 bytes per side) to fit entirely in L1
 * cache, making memmove extremely fast (~10-50ns for these sizes).
 */

#include "order_book.h"
#include "clock.h"
#include <cstring>
#include <algorithm>

namespace hft {

OrderBook::OrderBook(uint32_t instrument_id,
                     const DataValidationConfig& val_config) noexcept
    : validator_(val_config)
    , instrument_id_(instrument_id)
{
    reset();
}

void OrderBook::reset() noexcept {
    std::memset(&snapshot_, 0, sizeof(BookSnapshot));
    snapshot_.timestamp_ns   = INVALID_TS;
    snapshot_.sequence_num   = -1;
    snapshot_.best_bid_price = INVALID_PRICE;
    snapshot_.best_ask_price = INVALID_PRICE;
    snapshot_.best_bid_qty   = 0;
    snapshot_.best_ask_qty   = 0;
    snapshot_.bid_count      = 0;
    snapshot_.ask_count      = 0;
    snapshot_.instrument_id  = instrument_id_;
    snapshot_.quality        = DataQuality::VALID;

    // Initialize all levels to invalid
    for (int32_t i = 0; i < MAX_BOOK_LEVELS; ++i) {
        snapshot_.bids[i] = {INVALID_PRICE, 0, 0, 0};
        snapshot_.asks[i] = {INVALID_PRICE, 0, 0, 0};
    }

    validator_.reset();
}

// ─── Apply Level Update ──────────────────────────────────────

DataQuality OrderBook::apply_update(const LevelUpdate& update) noexcept {
    // Build a minimal BookSnapshot for validation timestamp/sequence checks
    // We validate the update's metadata, not the full book yet
    int64_t ref_time = now_ns();

    // Basic sanity: price must be positive for non-removal updates
    if (update.quantity > 0 && update.price <= 0) {
        return DataQuality::PRICE_ANOMALY;
    }

    if (update.side == Side::NONE) {
        return DataQuality::PRICE_ANOMALY;
    }

    // Apply the update
    if (update.quantity <= 0) {
        // Remove this price level
        if (update.side == Side::BID) {
            remove_bid(update.price);
        } else {
            remove_ask(update.price);
        }
    } else {
        // Insert or update this price level
        if (update.side == Side::BID) {
            upsert_bid(update.price, update.quantity, update.order_count);
        } else {
            upsert_ask(update.price, update.quantity, update.order_count);
        }
    }

    // Update metadata
    snapshot_.timestamp_ns  = update.timestamp_ns;
    snapshot_.sequence_num  = update.sequence_num;
    snapshot_.instrument_id = instrument_id_;

    // Refresh top-of-book cache
    refresh_top_of_book();

    // Post-update validation: is the book in a consistent state?
    if (snapshot_.bid_count > 0 && snapshot_.ask_count > 0) {
        if (snapshot_.best_bid_price >= snapshot_.best_ask_price) {
            snapshot_.quality = DataQuality::CROSSED_BOOK;
            return DataQuality::CROSSED_BOOK;
        }
    }

    snapshot_.quality = DataQuality::VALID;
    return DataQuality::VALID;
}

// ─── Apply Full Snapshot ─────────────────────────────────────

DataQuality OrderBook::apply_snapshot(const BookSnapshot& book) noexcept {
    int64_t ref_time = now_ns();

    // Validate the incoming snapshot
    DataQuality quality = validator_.validate_book(book, ref_time);
    if (quality != DataQuality::VALID) {
        return quality;
    }

    // Replace entire book state
    snapshot_ = book;
    snapshot_.instrument_id = instrument_id_;
    snapshot_.quality = DataQuality::VALID;

    return DataQuality::VALID;
}

// ─── Apply Trade ─────────────────────────────────────────────

DataQuality OrderBook::apply_trade(const Trade& trade) noexcept {
    int64_t ref_time = now_ns();

    // Validate the trade
    DataQuality quality = validator_.validate_trade(trade, ref_time);
    if (quality != DataQuality::VALID) {
        return quality;
    }

    // A trade at a price reduces the resting quantity at that level
    // If the aggressor is a buyer (Side::BID), they hit the ask side
    // If the aggressor is a seller (Side::ASK), they hit the bid side
    if (trade.side == Side::BID) {
        // Buyer aggressed — reduce ask quantity at trade price
        for (int32_t i = 0; i < snapshot_.ask_count; ++i) {
            if (snapshot_.asks[i].price == trade.price) {
                snapshot_.asks[i].quantity -= trade.quantity;
                if (snapshot_.asks[i].quantity <= 0) {
                    remove_ask(trade.price);
                }
                break;
            }
        }
    } else {
        // Seller aggressed — reduce bid quantity at trade price
        for (int32_t i = 0; i < snapshot_.bid_count; ++i) {
            if (snapshot_.bids[i].price == trade.price) {
                snapshot_.bids[i].quantity -= trade.quantity;
                if (snapshot_.bids[i].quantity <= 0) {
                    remove_bid(trade.price);
                }
                break;
            }
        }
    }

    snapshot_.timestamp_ns = trade.timestamp_ns;
    refresh_top_of_book();

    return DataQuality::VALID;
}

// ─── Bid Operations ──────────────────────────────────────────

void OrderBook::upsert_bid(int64_t price, int64_t quantity,
                           int32_t count) noexcept {
    // Search for existing level at this price
    for (int32_t i = 0; i < snapshot_.bid_count; ++i) {
        if (snapshot_.bids[i].price == price) {
            // Update existing level
            snapshot_.bids[i].quantity    = quantity;
            snapshot_.bids[i].order_count = count;
            return;
        }
    }

    // Not found — insert new level in sorted position (descending)
    if (snapshot_.bid_count >= MAX_BOOK_LEVELS) {
        // Book is full — only insert if this price is better than worst
        if (price <= snapshot_.bids[snapshot_.bid_count - 1].price) {
            return; // Price is worse than all tracked levels
        }
        // Replace the worst level
        --snapshot_.bid_count;
    }

    // Find insertion point (descending order: highest price first)
    int32_t insert_pos = snapshot_.bid_count;
    for (int32_t i = 0; i < snapshot_.bid_count; ++i) {
        if (price > snapshot_.bids[i].price) {
            insert_pos = i;
            break;
        }
    }

    // Shift elements right to make room
    if (insert_pos < snapshot_.bid_count) {
        std::memmove(
            &snapshot_.bids[insert_pos + 1],
            &snapshot_.bids[insert_pos],
            static_cast<size_t>(snapshot_.bid_count - insert_pos)
                * sizeof(PriceLevel)
        );
    }

    // Insert the new level
    snapshot_.bids[insert_pos] = {price, quantity, count, 0};
    ++snapshot_.bid_count;
}

void OrderBook::remove_bid(int64_t price) noexcept {
    for (int32_t i = 0; i < snapshot_.bid_count; ++i) {
        if (snapshot_.bids[i].price == price) {
            // Shift elements left to fill the gap
            if (i < snapshot_.bid_count - 1) {
                std::memmove(
                    &snapshot_.bids[i],
                    &snapshot_.bids[i + 1],
                    static_cast<size_t>(snapshot_.bid_count - i - 1)
                        * sizeof(PriceLevel)
                );
            }
            --snapshot_.bid_count;

            // Clear the now-unused last slot
            snapshot_.bids[snapshot_.bid_count] = {INVALID_PRICE, 0, 0, 0};
            return;
        }
    }
}

// ─── Ask Operations ──────────────────────────────────────────

void OrderBook::upsert_ask(int64_t price, int64_t quantity,
                           int32_t count) noexcept {
    // Search for existing level at this price
    for (int32_t i = 0; i < snapshot_.ask_count; ++i) {
        if (snapshot_.asks[i].price == price) {
            snapshot_.asks[i].quantity    = quantity;
            snapshot_.asks[i].order_count = count;
            return;
        }
    }

    // Not found — insert new level in sorted position (ascending)
    if (snapshot_.ask_count >= MAX_BOOK_LEVELS) {
        if (price >= snapshot_.asks[snapshot_.ask_count - 1].price) {
            return; // Price is worse than all tracked levels
        }
        --snapshot_.ask_count;
    }

    // Find insertion point (ascending order: lowest price first)
    int32_t insert_pos = snapshot_.ask_count;
    for (int32_t i = 0; i < snapshot_.ask_count; ++i) {
        if (price < snapshot_.asks[i].price) {
            insert_pos = i;
            break;
        }
    }

    // Shift elements right
    if (insert_pos < snapshot_.ask_count) {
        std::memmove(
            &snapshot_.asks[insert_pos + 1],
            &snapshot_.asks[insert_pos],
            static_cast<size_t>(snapshot_.ask_count - insert_pos)
                * sizeof(PriceLevel)
        );
    }

    snapshot_.asks[insert_pos] = {price, quantity, count, 0};
    ++snapshot_.ask_count;
}

void OrderBook::remove_ask(int64_t price) noexcept {
    for (int32_t i = 0; i < snapshot_.ask_count; ++i) {
        if (snapshot_.asks[i].price == price) {
            if (i < snapshot_.ask_count - 1) {
                std::memmove(
                    &snapshot_.asks[i],
                    &snapshot_.asks[i + 1],
                    static_cast<size_t>(snapshot_.ask_count - i - 1)
                        * sizeof(PriceLevel)
                );
            }
            --snapshot_.ask_count;
            snapshot_.asks[snapshot_.ask_count] = {INVALID_PRICE, 0, 0, 0};
            return;
        }
    }
}

// ─── Refresh Top of Book ─────────────────────────────────────

void OrderBook::refresh_top_of_book() noexcept {
    if (snapshot_.bid_count > 0) {
        snapshot_.best_bid_price = snapshot_.bids[0].price;
        snapshot_.best_bid_qty   = snapshot_.bids[0].quantity;
    } else {
        snapshot_.best_bid_price = INVALID_PRICE;
        snapshot_.best_bid_qty   = 0;
    }

    if (snapshot_.ask_count > 0) {
        snapshot_.best_ask_price = snapshot_.asks[0].price;
        snapshot_.best_ask_qty   = snapshot_.asks[0].quantity;
    } else {
        snapshot_.best_ask_price = INVALID_PRICE;
        snapshot_.best_ask_qty   = 0;
    }
}

} // namespace hft
