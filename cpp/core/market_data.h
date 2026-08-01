#pragma once
/**
 * @file market_data.h
 * @brief Market data parsing and normalization layer.
 *
 * Converts raw exchange data (JSON, CSV rows, binary) into the
 * engine's internal POD types (Trade, LevelUpdate, BookSnapshot).
 * This is the entry point for ALL data — both live feeds and
 * historical file replay.
 *
 * For backtesting with real Binance data (from data.binance.vision),
 * this module provides parsers for:
 *   - aggTrades CSV format
 *   - trades CSV format
 *   - Order book depth snapshots
 */

#include "types.h"
#include <cstdint>
#include <string_view>

namespace hft {

/**
 * @brief Parse a Binance aggTrades CSV row into a Trade struct.
 *
 * Binance aggTrades CSV format (from data.binance.vision):
 *   agg_trade_id, price, quantity, first_trade_id, last_trade_id,
 *   timestamp_ms, is_buyer_maker, is_best_match
 *
 * @param csv_line  A single CSV line (no newline)
 * @param trade     Output trade struct
 * @return true if parsing succeeded, false on malformed data
 */
bool parse_binance_agg_trade(std::string_view csv_line, Trade& trade) noexcept;

/**
 * @brief Parse a Binance raw trades CSV row into a Trade struct.
 *
 * Binance trades CSV format:
 *   trade_id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch
 *
 * @param csv_line  A single CSV line (no newline)
 * @param trade     Output trade struct
 * @return true if parsing succeeded
 */
bool parse_binance_trade(std::string_view csv_line, Trade& trade) noexcept;

/**
 * @brief Fast string-to-double conversion for price/quantity fields.
 *
 * Avoids std::stod overhead by doing fixed-point conversion directly.
 * Handles up to 8 decimal places (matching PRICE_SCALE / QTY_SCALE).
 *
 * @param sv  String containing a decimal number
 * @return Fixed-point int64_t value (scaled by PRICE_SCALE)
 */
[[nodiscard]] int64_t fast_parse_fixed(std::string_view sv) noexcept;

/**
 * @brief Fast string-to-int64 conversion.
 *
 * @param sv  String containing an integer
 * @return The parsed integer, or -1 on failure
 */
[[nodiscard]] int64_t fast_parse_int(std::string_view sv) noexcept;

} // namespace hft
