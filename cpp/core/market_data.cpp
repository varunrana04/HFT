/**
 * @file market_data.cpp
 * @brief Implementation of market data parsers.
 *
 * Parsers are designed for SPEED:
 *   - No std::string allocation (works on string_view)
 *   - No exceptions (returns bool success)
 *   - Custom fast_parse_fixed avoids stod/strtod overhead
 *   - Direct conversion from decimal string → fixed-point int64
 */

#include "market_data.h"
#include <charconv>
#include <cstring>

namespace hft {

// ─── Helper: Split CSV by comma ──────────────────────────────

/// Find the Nth comma-separated field in a CSV line.
/// Returns empty string_view if field not found.
static std::string_view get_csv_field(std::string_view line,
                                      size_t field_index) noexcept {
    size_t start = 0;
    size_t field = 0;

    for (size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            if (field == field_index) {
                return line.substr(start, i - start);
            }
            ++field;
            start = i + 1;
        }
    }

    return {};  // Field not found
}

// ─── Fast Integer Parsing ────────────────────────────────────

int64_t fast_parse_int(std::string_view sv) noexcept {
    if (sv.empty()) return -1;

    int64_t result = 0;
    bool negative = false;
    size_t i = 0;

    if (sv[0] == '-') {
        negative = true;
        i = 1;
    }

    for (; i < sv.size(); ++i) {
        char c = sv[i];
        if (c < '0' || c > '9') return -1;  // Invalid character
        result = result * 10 + (c - '0');
    }

    return negative ? -result : result;
}

// ─── Fast Fixed-Point Parsing ────────────────────────────────

int64_t fast_parse_fixed(std::string_view sv) noexcept {
    if (sv.empty()) return INVALID_PRICE;

    bool negative = false;
    size_t i = 0;

    if (sv[0] == '-') {
        negative = true;
        i = 1;
    }

    int64_t integer_part = 0;
    int64_t decimal_part = 0;
    int64_t decimal_scale = PRICE_SCALE;
    bool in_decimal = false;
    int decimal_digits = 0;

    for (; i < sv.size(); ++i) {
        char c = sv[i];

        if (c == '.') {
            in_decimal = true;
            continue;
        }

        if (c < '0' || c > '9') {
            // Skip whitespace/newline at end
            if (c == '\r' || c == '\n' || c == ' ') break;
            return INVALID_PRICE;
        }

        if (!in_decimal) {
            integer_part = integer_part * 10 + (c - '0');
        } else {
            ++decimal_digits;
            if (decimal_digits <= 8) {  // Max 8 decimal places
                decimal_scale /= 10;
                decimal_part += (c - '0') * decimal_scale;
            }
            // Ignore digits beyond 8 decimal places
        }
    }

    int64_t result = integer_part * PRICE_SCALE + decimal_part;
    return negative ? -result : result;
}

// ─── Binance aggTrades Parser ────────────────────────────────

bool parse_binance_agg_trade(std::string_view csv_line,
                             Trade& trade) noexcept {
    // Format: agg_trade_id, price, quantity, first_trade_id,
    //         last_trade_id, timestamp_ms, is_buyer_maker, is_best_match
    //
    // Example: 123456,50000.12,0.001,100,105,1704067200000,false,true

    if (csv_line.empty()) return false;

    // Skip header lines
    if (csv_line[0] < '0' || csv_line[0] > '9') {
        // Check if first char is not a digit (header row)
        if (csv_line[0] != '-') return false;
    }

    // Field 0: agg_trade_id → sequence_num
    std::string_view f_id = get_csv_field(csv_line, 0);
    int64_t seq = fast_parse_int(f_id);
    if (seq < 0) return false;

    // Field 1: price
    std::string_view f_price = get_csv_field(csv_line, 1);
    int64_t price = fast_parse_fixed(f_price);
    if (price <= 0) return false;

    // Field 2: quantity
    std::string_view f_qty = get_csv_field(csv_line, 2);
    int64_t quantity = fast_parse_fixed(f_qty);
    if (quantity <= 0) return false;

    // Field 5: timestamp_ms
    std::string_view f_ts = get_csv_field(csv_line, 5);
    int64_t ts_ms = fast_parse_int(f_ts);
    if (ts_ms < 0) return false;

    // Field 6: is_buyer_maker
    // If is_buyer_maker == "true", the MAKER was the buyer,
    // meaning the TAKER (aggressor) was the seller → Side::ASK
    // If is_buyer_maker == "false", the aggressor was the buyer → Side::BID
    std::string_view f_side = get_csv_field(csv_line, 6);
    Side side = Side::NONE;
    if (!f_side.empty()) {
        // "true" or "True" → maker was buyer → aggressor is seller
        if (f_side[0] == 't' || f_side[0] == 'T') {
            side = Side::ASK;  // Sell aggressor
        } else {
            side = Side::BID;  // Buy aggressor
        }
    }

    // Populate the trade struct
    trade.timestamp_ns  = ts_ms * 1'000'000LL;  // ms → ns
    trade.sequence_num  = seq;
    trade.price         = price;
    trade.quantity       = quantity;
    trade.instrument_id  = 0;  // Set by caller
    trade.side           = side;
    trade.quality        = DataQuality::VALID;
    std::memset(trade._pad, 0, sizeof(trade._pad));

    return true;
}

// ─── Binance Raw Trades Parser ───────────────────────────────

bool parse_binance_trade(std::string_view csv_line, Trade& trade) noexcept {
    // Format: trade_id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch
    //
    // Example: 789,50001.23,0.005,250.00615,1704067200123,true,true

    if (csv_line.empty()) return false;
    if (csv_line[0] < '0' || csv_line[0] > '9') {
        if (csv_line[0] != '-') return false;
    }

    std::string_view f_id = get_csv_field(csv_line, 0);
    int64_t seq = fast_parse_int(f_id);
    if (seq < 0) return false;

    std::string_view f_price = get_csv_field(csv_line, 1);
    int64_t price = fast_parse_fixed(f_price);
    if (price <= 0) return false;

    std::string_view f_qty = get_csv_field(csv_line, 2);
    int64_t quantity = fast_parse_fixed(f_qty);
    if (quantity <= 0) return false;

    // Field 4: timestamp_ms
    std::string_view f_ts = get_csv_field(csv_line, 4);
    int64_t ts_ms = fast_parse_int(f_ts);
    if (ts_ms < 0) return false;

    // Field 5: isBuyerMaker
    std::string_view f_side = get_csv_field(csv_line, 5);
    Side side = Side::NONE;
    if (!f_side.empty()) {
        side = (f_side[0] == 't' || f_side[0] == 'T')
             ? Side::ASK : Side::BID;
    }

    trade.timestamp_ns   = ts_ms * 1'000'000LL;
    trade.sequence_num   = seq;
    trade.price          = price;
    trade.quantity        = quantity;
    trade.instrument_id   = 0;
    trade.side            = side;
    trade.quality         = DataQuality::VALID;
    std::memset(trade._pad, 0, sizeof(trade._pad));

    return true;
}

} // namespace hft
