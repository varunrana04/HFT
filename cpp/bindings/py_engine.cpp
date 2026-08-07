/**
 * @file py_engine.cpp
 * @brief Python bindings for the HFT Engine using pybind11.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector, std::string
#include "strategy_engine.h"
#include "types.h"

namespace py = pybind11;
using namespace hft;

PYBIND11_MODULE(hft_engine, m) {
    m.doc() = "High-Frequency Trading Engine C++ Core Extension";

    // ─── Enums ───────────────────────────────────────────────────────────
    py::enum_<Side>(m, "Side")
        .value("BID", Side::BID)
        .value("ASK", Side::ASK)
        .export_values();

    py::enum_<EngineMode>(m, "EngineMode")
        .value("BACKTEST", EngineMode::BACKTEST)
        .value("PAPER", EngineMode::PAPER)
        .value("LIVE", EngineMode::LIVE)
        .export_values();

    py::enum_<Regime>(m, "Regime")
        .value("NORMAL", Regime::NORMAL)
        .value("HIGH_TOXICITY", Regime::HIGH_TOXICITY)
        .value("LOW_LIQUIDITY", Regime::LOW_LIQUIDITY)
        .value("TRENDING", Regime::TRENDING)
        .value("UNKNOWN", Regime::UNKNOWN)
        .export_values();

    // ─── Config Structs ──────────────────────────────────────────────────
    py::class_<StrategyConfig>(m, "StrategyConfig")
        .def(py::init<>())
        .def_readwrite("initial_capital", &StrategyConfig::initial_capital)
        .def_readwrite("alpha_entry_threshold", &StrategyConfig::alpha_entry_threshold)
        .def_readwrite("alpha_exit_threshold", &StrategyConfig::alpha_exit_threshold)
        .def_readwrite("position_size_pct", &StrategyConfig::position_size_pct)
        .def_readwrite("maker_fee", &StrategyConfig::maker_fee)
        .def_readwrite("taker_fee", &StrategyConfig::taker_fee);

    py::class_<FeatureConfig>(m, "FeatureConfig")
        .def(py::init<>())
        .def_readwrite("vpin_bucket_size", &FeatureConfig::vpin_bucket_size)
        .def_readwrite("vpin_n_buckets", &FeatureConfig::vpin_n_buckets)
        .def_readwrite("vol_window_ticks", &FeatureConfig::vol_window_ticks)
        .def_readwrite("stat_arb_lookback", &FeatureConfig::stat_arb_lookback);

    py::class_<RiskConfig>(m, "RiskConfig")
        .def(py::init<>())
        .def_readwrite("max_position", &RiskConfig::max_position)
        .def_readwrite("max_drawdown_pct", &RiskConfig::max_drawdown_pct)
        .def_readwrite("max_daily_loss_pct", &RiskConfig::max_daily_loss_pct)
        .def_readwrite("circuit_breaker_cooldown_ns", &RiskConfig::circuit_breaker_cooldown_ns);

    // ─── Data Types ──────────────────────────────────────────────────────
    py::class_<Trade>(m, "Trade")
        .def(py::init<>())
        .def_readwrite("timestamp_ns", &Trade::timestamp_ns)
        .def_readwrite("price", &Trade::price)
        .def_readwrite("quantity", &Trade::quantity)
        .def_readwrite("side", &Trade::side);

    // Provide some helper functions to map double values to fixed-point for Python
    m.def("price_to_fixed", &price_to_fixed, "Convert float price to internal fixed-point representation");
    m.def("qty_to_fixed", &qty_to_fixed, "Convert float quantity to internal fixed-point representation");
    m.def("fixed_to_price", &fixed_to_price, "Convert internal fixed-point price back to float");
    m.def("fixed_to_qty", &fixed_to_qty, "Convert internal fixed-point quantity back to float");

    // ─── FeatureVector Output ────────────────────────────────────────────
    py::class_<FeatureVector>(m, "FeatureVector")
        .def(py::init<>())
        .def_readwrite("microprice", &FeatureVector::microprice)
        .def_readwrite("ofi", &FeatureVector::ofi)
        .def_readwrite("vpin", &FeatureVector::vpin)
        .def_readwrite("spread_bps", &FeatureVector::spread_bps)
        .def_readwrite("realized_vol", &FeatureVector::realized_vol)
        .def_readwrite("stat_arb_zscore", &FeatureVector::stat_arb_zscore)
        .def_readwrite("regime", &FeatureVector::regime)
        .def_readwrite("combined_alpha", &FeatureVector::combined_alpha);

    // ─── PerformanceMetrics ──────────────────────────────────────────────
    py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
        .def_readwrite("total_trades", &PerformanceMetrics::total_trades)
        .def_readwrite("winning_trades", &PerformanceMetrics::winning_trades)
        .def_readwrite("sharpe_ratio", &PerformanceMetrics::sharpe_ratio)
        .def_readwrite("max_drawdown", &PerformanceMetrics::max_drawdown);

    // ─── Strategy Engine (Core Orchestrator) ─────────────────────────────
    py::class_<StrategyEngine>(m, "StrategyEngine")
        .def(py::init<const StrategyConfig&, const FeatureConfig&, const RiskConfig&>(),
             py::arg("strategy_cfg") = StrategyConfig(),
             py::arg("feature_cfg") = FeatureConfig(),
             py::arg("risk_cfg") = RiskConfig())
        .def("on_trade", &StrategyEngine::on_trade, "Process a new trade tick and optionally place orders")
        .def("equity", &StrategyEngine::equity, "Get current total equity")
        .def("position", &StrategyEngine::position, "Get current net position")
        .def("last_features", &StrategyEngine::last_features, "Get the last calculated feature vector")
        .def("metrics", &StrategyEngine::metrics, "Get backtest/live performance metrics")
        .def("reset", &StrategyEngine::reset, "Reset engine state");
}
