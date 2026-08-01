/**
 * @file py_engine.cpp
 * @brief pybind11 Python bindings for the HFT engine.
 *
 * Exposes the entire C++ core to Python for backtesting and analytics:
 *   - All enums (Side, OrderType, OrderState, Regime, DataQuality)
 *   - All POD structs (PriceLevel, BookSnapshot, Trade, FeatureVector, Order)
 *   - All engine classes (OrderBook, FeatureEngine, RiskManager, etc.)
 *   - Utility functions (price_to_fixed, parse_binance_agg_trade, now_ns)
 *
 * FeatureVector provides a .to_numpy() method for zero-copy export of
 * signal values to a NumPy float64 array.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "types.h"
#include "clock.h"
#include "order_book.h"
#include "features.h"
#include "signal_combiner.h"
#include "risk_manager.h"
#include "order_manager.h"
#include "data_validator.h"
#include "market_data.h"

namespace py = pybind11;

PYBIND11_MODULE(hft_engine, m) {
    m.doc() = "HFT Engine — Ultra-low-latency trading engine (C++ core)";

    // ═══════════════════════════════════════════════════════════
    //  Constants
    // ═══════════════════════════════════════════════════════════
    m.attr("PRICE_SCALE")     = hft::PRICE_SCALE;
    m.attr("QTY_SCALE")       = hft::QTY_SCALE;
    m.attr("MAX_BOOK_LEVELS") = hft::MAX_BOOK_LEVELS;
    m.attr("INVALID_PRICE")   = hft::INVALID_PRICE;
    m.attr("CACHE_LINE_SIZE") = hft::CACHE_LINE_SIZE;

    // ═══════════════════════════════════════════════════════════
    //  Enums
    // ═══════════════════════════════════════════════════════════
    py::enum_<hft::Side>(m, "Side")
        .value("BID",  hft::Side::BID)
        .value("ASK",  hft::Side::ASK)
        .value("NONE", hft::Side::NONE)
        .export_values();

    py::enum_<hft::OrderType>(m, "OrderType")
        .value("LIMIT",  hft::OrderType::LIMIT)
        .value("MARKET", hft::OrderType::MARKET)
        .value("IOC",    hft::OrderType::IOC)
        .value("FOK",    hft::OrderType::FOK)
        .export_values();

    py::enum_<hft::OrderState>(m, "OrderState")
        .value("NEW",       hft::OrderState::NEW)
        .value("SENT",      hft::OrderState::SENT)
        .value("PARTIAL",   hft::OrderState::PARTIAL)
        .value("FILLED",    hft::OrderState::FILLED)
        .value("CANCELLED", hft::OrderState::CANCELLED)
        .value("REJECTED",  hft::OrderState::REJECTED)
        .export_values();

    py::enum_<hft::Regime>(m, "Regime")
        .value("NORMAL",        hft::Regime::NORMAL)
        .value("HIGH_TOXICITY", hft::Regime::HIGH_TOXICITY)
        .value("LOW_LIQUIDITY", hft::Regime::LOW_LIQUIDITY)
        .value("TRENDING",      hft::Regime::TRENDING)
        .value("UNKNOWN",       hft::Regime::UNKNOWN)
        .export_values();

    py::enum_<hft::DataQuality>(m, "DataQuality")
        .value("VALID",           hft::DataQuality::VALID)
        .value("STALE_TIMESTAMP", hft::DataQuality::STALE_TIMESTAMP)
        .value("OUT_OF_SEQUENCE", hft::DataQuality::OUT_OF_SEQUENCE)
        .value("PRICE_ANOMALY",   hft::DataQuality::PRICE_ANOMALY)
        .value("QTY_ANOMALY",     hft::DataQuality::QTY_ANOMALY)
        .value("CROSSED_BOOK",    hft::DataQuality::CROSSED_BOOK)
        .value("MISSING_LEVELS",  hft::DataQuality::MISSING_LEVELS)
        .value("DUPLICATE",       hft::DataQuality::DUPLICATE)
        .value("INVALID_SYMBOL",  hft::DataQuality::INVALID_SYMBOL)
        .export_values();

    py::enum_<hft::RiskVerdict>(m, "RiskVerdict")
        .value("PASS",             hft::RiskVerdict::PASS)
        .value("POSITION_LIMIT",   hft::RiskVerdict::POSITION_LIMIT)
        .value("DRAWDOWN_LIMIT",   hft::RiskVerdict::DRAWDOWN_LIMIT)
        .value("DAILY_LOSS_LIMIT", hft::RiskVerdict::DAILY_LOSS_LIMIT)
        .value("ORDER_SIZE_LIMIT", hft::RiskVerdict::ORDER_SIZE_LIMIT)
        .value("CIRCUIT_BREAKER",  hft::RiskVerdict::CIRCUIT_BREAKER)
        .export_values();

    // ═══════════════════════════════════════════════════════════
    //  Structs
    // ═══════════════════════════════════════════════════════════

    // ── PriceLevel ───────────────────────────────────────────
    py::class_<hft::PriceLevel>(m, "PriceLevel")
        .def(py::init<>())
        .def_readwrite("price",       &hft::PriceLevel::price)
        .def_readwrite("quantity",    &hft::PriceLevel::quantity)
        .def_readwrite("order_count", &hft::PriceLevel::order_count)
        .def("is_valid", &hft::PriceLevel::is_valid)
        .def("__repr__", [](const hft::PriceLevel& pl) {
            return "<PriceLevel price=" +
                   std::to_string(hft::fixed_to_price(pl.price)) +
                   " qty=" +
                   std::to_string(hft::fixed_to_qty(pl.quantity)) +
                   " orders=" + std::to_string(pl.order_count) + ">";
        });

    // ── BookSnapshot ─────────────────────────────────────────
    py::class_<hft::BookSnapshot>(m, "BookSnapshot")
        .def(py::init<>())
        .def_readwrite("timestamp_ns",   &hft::BookSnapshot::timestamp_ns)
        .def_readwrite("sequence_num",   &hft::BookSnapshot::sequence_num)
        .def_readwrite("bid_count",      &hft::BookSnapshot::bid_count)
        .def_readwrite("ask_count",      &hft::BookSnapshot::ask_count)
        .def_readwrite("best_bid_price", &hft::BookSnapshot::best_bid_price)
        .def_readwrite("best_ask_price", &hft::BookSnapshot::best_ask_price)
        .def_readwrite("best_bid_qty",   &hft::BookSnapshot::best_bid_qty)
        .def_readwrite("best_ask_qty",   &hft::BookSnapshot::best_ask_qty)
        .def_readwrite("instrument_id",  &hft::BookSnapshot::instrument_id)
        .def_readwrite("quality",        &hft::BookSnapshot::quality)
        .def("mid_price", &hft::BookSnapshot::mid_price)
        .def("spread",    &hft::BookSnapshot::spread)
        .def("is_valid",  &hft::BookSnapshot::is_valid)
        .def("get_bid", [](const hft::BookSnapshot& b, int i) -> hft::PriceLevel {
            if (i < 0 || i >= hft::MAX_BOOK_LEVELS) throw py::index_error();
            return b.bids[i];
        }, py::arg("index"))
        .def("get_ask", [](const hft::BookSnapshot& b, int i) -> hft::PriceLevel {
            if (i < 0 || i >= hft::MAX_BOOK_LEVELS) throw py::index_error();
            return b.asks[i];
        }, py::arg("index"))
        .def("__repr__", [](const hft::BookSnapshot& b) {
            return "<BookSnapshot bid=" +
                   std::to_string(hft::fixed_to_price(b.best_bid_price)) +
                   " ask=" +
                   std::to_string(hft::fixed_to_price(b.best_ask_price)) +
                   " levels=" + std::to_string(b.bid_count) + "/" +
                   std::to_string(b.ask_count) + ">";
        });

    // ── Trade ────────────────────────────────────────────────
    py::class_<hft::Trade>(m, "Trade")
        .def(py::init<>())
        .def_readwrite("timestamp_ns",  &hft::Trade::timestamp_ns)
        .def_readwrite("sequence_num",  &hft::Trade::sequence_num)
        .def_readwrite("price",         &hft::Trade::price)
        .def_readwrite("quantity",      &hft::Trade::quantity)
        .def_readwrite("instrument_id", &hft::Trade::instrument_id)
        .def_readwrite("side",          &hft::Trade::side)
        .def_readwrite("quality",       &hft::Trade::quality)
        .def("is_valid", &hft::Trade::is_valid)
        .def("__repr__", [](const hft::Trade& t) {
            return "<Trade price=" +
                   std::to_string(hft::fixed_to_price(t.price)) +
                   " qty=" +
                   std::to_string(hft::fixed_to_qty(t.quantity)) +
                   " side=" +
                   std::string(t.side == hft::Side::BID ? "BID" : "ASK") +
                   ">";
        });

    // ── LevelUpdate ──────────────────────────────────────────
    py::class_<hft::LevelUpdate>(m, "LevelUpdate")
        .def(py::init<>())
        .def_readwrite("timestamp_ns", &hft::LevelUpdate::timestamp_ns)
        .def_readwrite("sequence_num", &hft::LevelUpdate::sequence_num)
        .def_readwrite("price",        &hft::LevelUpdate::price)
        .def_readwrite("quantity",     &hft::LevelUpdate::quantity)
        .def_readwrite("order_count",  &hft::LevelUpdate::order_count)
        .def_readwrite("side",         &hft::LevelUpdate::side);

    // ── FeatureVector ────────────────────────────────────────
    py::class_<hft::FeatureVector>(m, "FeatureVector")
        .def(py::init<>())
        .def_readwrite("timestamp_ns",    &hft::FeatureVector::timestamp_ns)
        .def_readwrite("microprice",      &hft::FeatureVector::microprice)
        .def_readwrite("ofi",             &hft::FeatureVector::ofi)
        .def_readwrite("vpin",            &hft::FeatureVector::vpin)
        .def_readwrite("spread_bps",      &hft::FeatureVector::spread_bps)
        .def_readwrite("realized_vol",    &hft::FeatureVector::realized_vol)
        .def_readwrite("stat_arb_zscore", &hft::FeatureVector::stat_arb_zscore)
        .def_readwrite("combined_alpha",  &hft::FeatureVector::combined_alpha)
        .def_readwrite("regime",          &hft::FeatureVector::regime)
        .def("has_valid_alpha", &hft::FeatureVector::has_valid_alpha)
        .def("to_numpy", [](const hft::FeatureVector& fv) {
            // Return a 1-D float64 array of the 6 signal values
            py::array_t<double> arr(6);
            auto buf = arr.mutable_unchecked<1>();
            buf(0) = fv.microprice;
            buf(1) = fv.ofi;
            buf(2) = fv.vpin;
            buf(3) = fv.spread_bps;
            buf(4) = fv.realized_vol;
            buf(5) = fv.stat_arb_zscore;
            return arr;
        }, "Return the 6 signal values as a NumPy float64 array "
           "[microprice, ofi, vpin, spread_bps, realized_vol, stat_arb_zscore]")
        .def("__repr__", [](const hft::FeatureVector& fv) {
            return "<FeatureVector micro=" +
                   std::to_string(fv.microprice) +
                   " ofi=" + std::to_string(fv.ofi) +
                   " vpin=" + std::to_string(fv.vpin) +
                   " spread=" + std::to_string(fv.spread_bps) +
                   " vol=" + std::to_string(fv.realized_vol) +
                   " z=" + std::to_string(fv.stat_arb_zscore) +
                   ">";
        });

    // ── Order ────────────────────────────────────────────────
    py::class_<hft::Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("timestamp_ns",    &hft::Order::timestamp_ns)
        .def_readwrite("price",           &hft::Order::price)
        .def_readwrite("quantity",        &hft::Order::quantity)
        .def_readwrite("filled_quantity", &hft::Order::filled_quantity)
        .def_readwrite("avg_fill_price",  &hft::Order::avg_fill_price)
        .def_readwrite("expected_price",  &hft::Order::expected_price)
        .def_readwrite("order_id",        &hft::Order::order_id)
        .def_readwrite("instrument_id",   &hft::Order::instrument_id)
        .def_readwrite("side",            &hft::Order::side)
        .def_readwrite("type",            &hft::Order::type)
        .def_readwrite("state",           &hft::Order::state)
        .def("slippage",    &hft::Order::slippage)
        .def("is_filled",   &hft::Order::is_filled)
        .def("is_terminal", &hft::Order::is_terminal)
        .def("__repr__", [](const hft::Order& o) {
            return "<Order id=" + std::to_string(o.order_id) +
                   " price=" + std::to_string(hft::fixed_to_price(o.price)) +
                   " qty=" + std::to_string(hft::fixed_to_qty(o.quantity)) +
                   " filled=" + std::to_string(hft::fixed_to_qty(o.filled_quantity)) +
                   ">";
        });

    // ═══════════════════════════════════════════════════════════
    //  Config Structs
    // ═══════════════════════════════════════════════════════════

    py::class_<hft::DataValidationConfig>(m, "DataValidationConfig")
        .def(py::init<>())
        .def_readwrite("max_staleness_ns",    &hft::DataValidationConfig::max_staleness_ns)
        .def_readwrite("max_future_ns",       &hft::DataValidationConfig::max_future_ns)
        .def_readwrite("max_price_change_pct",&hft::DataValidationConfig::max_price_change_pct)
        .def_readwrite("max_quantity",        &hft::DataValidationConfig::max_quantity)
        .def_readwrite("min_book_depth",      &hft::DataValidationConfig::min_book_depth)
        .def_readwrite("max_sequence_gap",    &hft::DataValidationConfig::max_sequence_gap);

    py::class_<hft::FeatureConfig>(m, "FeatureConfig")
        .def(py::init<>())
        .def_readwrite("vpin_bucket_size",      &hft::FeatureConfig::vpin_bucket_size)
        .def_readwrite("vpin_n_buckets",        &hft::FeatureConfig::vpin_n_buckets)
        .def_readwrite("vol_window_ticks",      &hft::FeatureConfig::vol_window_ticks)
        .def_readwrite("stat_arb_zscore_entry", &hft::FeatureConfig::stat_arb_zscore_entry)
        .def_readwrite("stat_arb_zscore_exit",  &hft::FeatureConfig::stat_arb_zscore_exit)
        .def_readwrite("stat_arb_lookback",     &hft::FeatureConfig::stat_arb_lookback)
        .def_readwrite("stat_arb_half_life_max",&hft::FeatureConfig::stat_arb_half_life_max);

    py::class_<hft::RiskConfig>(m, "RiskConfig")
        .def(py::init<>())
        .def_readwrite("max_position",            &hft::RiskConfig::max_position)
        .def_readwrite("max_drawdown_pct",        &hft::RiskConfig::max_drawdown_pct)
        .def_readwrite("max_single_order_pct",    &hft::RiskConfig::max_single_order_pct)
        .def_readwrite("max_daily_loss_pct",      &hft::RiskConfig::max_daily_loss_pct)
        .def_readwrite("circuit_breaker_cooldown_ns",
                        &hft::RiskConfig::circuit_breaker_cooldown_ns);

    // ── Stats Structs ────────────────────────────────────────
    py::class_<hft::ValidationStats>(m, "ValidationStats")
        .def(py::init<>())
        .def_readonly("total_ticks_seen",  &hft::ValidationStats::total_ticks_seen)
        .def_readonly("valid_ticks",       &hft::ValidationStats::valid_ticks)
        .def_readonly("stale_timestamps",  &hft::ValidationStats::stale_timestamps)
        .def_readonly("out_of_sequence",   &hft::ValidationStats::out_of_sequence)
        .def_readonly("price_anomalies",   &hft::ValidationStats::price_anomalies)
        .def_readonly("qty_anomalies",     &hft::ValidationStats::qty_anomalies)
        .def_readonly("crossed_books",     &hft::ValidationStats::crossed_books)
        .def_readonly("duplicates",        &hft::ValidationStats::duplicates)
        .def("acceptance_rate", &hft::ValidationStats::acceptance_rate);

    py::class_<hft::RiskStats>(m, "RiskStats")
        .def(py::init<>())
        .def_readonly("orders_checked",       &hft::RiskStats::orders_checked)
        .def_readonly("orders_passed",        &hft::RiskStats::orders_passed)
        .def_readonly("rejected_position",    &hft::RiskStats::rejected_position)
        .def_readonly("rejected_drawdown",    &hft::RiskStats::rejected_drawdown)
        .def_readonly("rejected_daily_loss",  &hft::RiskStats::rejected_daily_loss)
        .def_readonly("rejected_order_size",  &hft::RiskStats::rejected_order_size)
        .def_readonly("rejected_circuit_brk", &hft::RiskStats::rejected_circuit_brk)
        .def_readonly("circuit_breaker_trips",&hft::RiskStats::circuit_breaker_trips)
        .def("pass_rate", &hft::RiskStats::pass_rate);

    // ═══════════════════════════════════════════════════════════
    //  Engine Classes
    // ═══════════════════════════════════════════════════════════

    // ── OrderBook ────────────────────────────────────────────
    py::class_<hft::OrderBook>(m, "OrderBook")
        .def(py::init<uint32_t, const hft::DataValidationConfig&>(),
             py::arg("instrument_id") = 0,
             py::arg("config") = hft::DataValidationConfig{})
        .def("apply_update",   &hft::OrderBook::apply_update)
        .def("apply_snapshot", &hft::OrderBook::apply_snapshot)
        .def("apply_trade",    &hft::OrderBook::apply_trade)
        .def("snapshot",       &hft::OrderBook::snapshot,
             py::return_value_policy::reference_internal)
        .def("best_bid",       &hft::OrderBook::best_bid)
        .def("best_ask",       &hft::OrderBook::best_ask)
        .def("mid_price",      &hft::OrderBook::mid_price)
        .def("spread",         &hft::OrderBook::spread)
        .def("is_valid",       &hft::OrderBook::is_valid)
        .def("reset",          &hft::OrderBook::reset)
        .def("set_reference_price", &hft::OrderBook::set_reference_price)
        .def("validation_stats",    &hft::OrderBook::validation_stats,
             py::return_value_policy::reference_internal);

    // ── FeatureEngine ────────────────────────────────────────
    py::class_<hft::FeatureEngine>(m, "FeatureEngine")
        .def(py::init<const hft::FeatureConfig&>(),
             py::arg("config") = hft::FeatureConfig{})
        .def("compute_all", &hft::FeatureEngine::compute_all)
        .def("reset",       &hft::FeatureEngine::reset)
        .def("config",      &hft::FeatureEngine::config,
             py::return_value_policy::reference_internal);

    // ── SignalCombiner ───────────────────────────────────────
    py::class_<hft::SignalCombiner>(m, "SignalCombiner")
        .def(py::init<>())
        .def("combine", &hft::SignalCombiner::combine)
        .def("set_weights", [](hft::SignalCombiner& sc,
                               py::array_t<double> weights) {
            auto buf = weights.unchecked<1>();
            sc.set_weights(buf.data(0),
                           static_cast<size_t>(buf.shape(0)));
        }, py::arg("weights"));

    // ── RiskManager ──────────────────────────────────────────
    py::class_<hft::RiskManager>(m, "RiskManager")
        .def(py::init<const hft::RiskConfig&>(),
             py::arg("config") = hft::RiskConfig{})
        .def("check_order",
             py::overload_cast<const hft::Order&, int64_t, double, double>(
                 &hft::RiskManager::check_order),
             py::arg("order"), py::arg("current_position"),
             py::arg("current_pnl"), py::arg("portfolio_value"))
        .def("update_equity",    &hft::RiskManager::update_equity)
        .def("new_trading_day",  &hft::RiskManager::new_trading_day)
        .def("reset",            &hft::RiskManager::reset)
        .def("stats",            &hft::RiskManager::stats,
             py::return_value_policy::reference_internal)
        .def("is_circuit_breaker_active",
             &hft::RiskManager::is_circuit_breaker_active)
        .def("current_drawdown", &hft::RiskManager::current_drawdown)
        .def("current_daily_loss", &hft::RiskManager::current_daily_loss);

    // ── OrderManager ─────────────────────────────────────────
    py::class_<hft::OrderManager>(m, "OrderManager")
        .def(py::init<>())
        .def("create_order", &hft::OrderManager::create_order)
        .def("on_fill",      &hft::OrderManager::on_fill)
        .def("cancel",       &hft::OrderManager::cancel);

    // ── DataValidator ────────────────────────────────────────
    py::class_<hft::DataValidator>(m, "DataValidator")
        .def(py::init<const hft::DataValidationConfig&>(),
             py::arg("config") = hft::DataValidationConfig{})
        .def("validate_trade", &hft::DataValidator::validate_trade)
        .def("validate_book",  &hft::DataValidator::validate_book)
        .def("stats",          &hft::DataValidator::stats,
             py::return_value_policy::reference_internal)
        .def("reset",          &hft::DataValidator::reset)
        .def("set_reference_price", &hft::DataValidator::set_reference_price);

    // ═══════════════════════════════════════════════════════════
    //  Free Functions
    // ═══════════════════════════════════════════════════════════

    m.def("price_to_fixed", &hft::price_to_fixed,
          "Convert float price to fixed-point int64");
    m.def("fixed_to_price", &hft::fixed_to_price,
          "Convert fixed-point int64 to float price");
    m.def("qty_to_fixed",   &hft::qty_to_fixed,
          "Convert float quantity to fixed-point int64");
    m.def("fixed_to_qty",   &hft::fixed_to_qty,
          "Convert fixed-point int64 to float quantity");
    m.def("now_ns",         &hft::now_ns,
          "Get current timestamp in nanoseconds (monotonic)");

    m.def("parse_binance_agg_trade",
          [](const std::string& csv_line) -> py::object {
              hft::Trade trade{};
              bool ok = hft::parse_binance_agg_trade(csv_line, trade);
              if (!ok) return py::none();
              return py::cast(trade);
          },
          py::arg("csv_line"),
          "Parse a Binance aggTrades CSV line into a Trade (or None)");

    m.def("parse_binance_trade",
          [](const std::string& csv_line) -> py::object {
              hft::Trade trade{};
              bool ok = hft::parse_binance_trade(csv_line, trade);
              if (!ok) return py::none();
              return py::cast(trade);
          },
          py::arg("csv_line"),
          "Parse a Binance raw trades CSV line into a Trade (or None)");
}
