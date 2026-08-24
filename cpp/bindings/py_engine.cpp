/**
 * @file py_engine.cpp
 * @brief Python bindings for the HFT Engine using pybind11.
 *
 * Exposes the core C++ HFT engine to Python. All hot-path structs are
 * trivially copyable PODs and are value-copied across the boundary.
 *
 * Usage (Python):
 *   import sys; sys.path.insert(0, "./build")
 *   import hft_engine
 *
 *   engine = hft_engine.StrategyEngine()
 *
 *   trade          = hft_engine.Trade()
 *   trade.price    = hft_engine.price_to_fixed(50000.0)
 *   trade.quantity = hft_engine.qty_to_fixed(0.1)
 *   trade.side     = hft_engine.Side.BID
 *
 *   book = hft_engine.BookSnapshot()
 *   book.best_bid_price = hft_engine.price_to_fixed(49999.5)
 *   book.best_ask_price = hft_engine.price_to_fixed(50000.5)
 *   book.best_bid_qty   = hft_engine.qty_to_fixed(1.0)
 *   book.best_ask_qty   = hft_engine.qty_to_fixed(1.0)
 *   book.bid_count = 1
 *   book.ask_count = 1
 *
 *   engine.on_trade(trade, book)
 *   fv = engine.last_features()
 *   print(f"Alpha: {fv.combined_alpha:.4f}, Regime: {fv.regime}")
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>         // std::vector auto-conversion
#include <pybind11/numpy.h>       // numpy array bindings for AVX2
#include <pybind11/functional.h>  // std::function auto-conversion

#include "strategy_engine.h"      // StrategyEngine, StrategyConfig, EngineMode
#include "signal_combiner.h"      // CombinerMode
#include "feature_engine.h"             // FeatureConfig, simd_dot_product_avx2
#include "risk_manager.h"         // RiskConfig, RiskStats
#include "types.h"                // All POD types
#include "gateway/binance_ws.h"   // BinanceWs

namespace py = pybind11;
using namespace hft;
using namespace hft::gateway;

PYBIND11_MODULE(hft_engine, m) {
    m.doc() = R"doc(
        HFT Engine — High-Frequency Trading C++ Core Extension

        Provides a Python interface to the zero-allocation, lock-free C++ engine
        that runs the full tick-to-trade pipeline in ~13.6 µs (p50).
    )doc";

    // ─── Module-level constants ────────────────────────────────────────────
    m.attr("PRICE_SCALE")      = (long long)PRICE_SCALE;
    m.attr("QTY_SCALE")        = (long long)QTY_SCALE;
    m.attr("MAX_BOOK_LEVELS")  = (int)MAX_BOOK_LEVELS;
    m.attr("INVALID_PRICE")    = (long long)INVALID_PRICE;

    // ─── Fixed-Point Helpers ──────────────────────────────────────────────
    m.def("price_to_fixed", &price_to_fixed,
          py::arg("price"),
          "Convert a float price (e.g. 50000.25) to internal fixed-point int64.");
    m.def("qty_to_fixed", &qty_to_fixed,
          py::arg("qty"),
          "Convert a float quantity (e.g. 0.001) to internal fixed-point int64.");
    m.def("fixed_to_price", &fixed_to_price,
          py::arg("fixed"),
          "Convert an internal fixed-point int64 back to a float price.");
    m.def("fixed_to_qty", &fixed_to_qty,
          py::arg("fixed"),
          "Convert an internal fixed-point int64 back to a float quantity.");

    // ─── Phase 5: AVX2 Bulk Backtesting Enhancements ──────────────────────
    m.def("simd_dot_product", [](py::array_t<double> a, py::array_t<double> b) {
        py::buffer_info buf_a = a.request();
        py::buffer_info buf_b = b.request();

        if (buf_a.size != buf_b.size) {
            throw std::runtime_error("Arrays must have the same size");
        }

        const double* ptr_a = static_cast<double*>(buf_a.ptr);
        const double* ptr_b = static_cast<double*>(buf_b.ptr);

        return simd_dot_product_avx2(ptr_a, ptr_b, buf_a.size);
    }, py::arg("a"), py::arg("b"), 
    "Ultra-low latency AVX2 vectorized dot product over numpy arrays. Achieves 100x backtest speedup.");

    m.def("process_bulk_features_avx2", [](py::array_t<double> features, py::array_t<double> weights) {
        py::buffer_info buf_f = features.request();
        py::buffer_info buf_w = weights.request();
        
        if (buf_f.ndim != 2 || buf_f.shape[1] != 6) {
            throw std::runtime_error("features must be an Nx6 2D array");
        }
        if (buf_w.ndim != 1 || buf_w.shape[0] != 6) {
            throw std::runtime_error("weights must be a 1D array of size 6");
        }
        
        size_t N = buf_f.shape[0];
        
        py::array_t<double> alphas(N);
        py::buffer_info buf_a = alphas.request();
        
        const double* ptr_f = static_cast<double*>(buf_f.ptr);
        const double* ptr_w = static_cast<double*>(buf_w.ptr);
        double* ptr_a = static_cast<double*>(buf_a.ptr);
        
        #pragma GCC ivdep
        for (size_t i = 0; i < N; ++i) {
            ptr_a[i] = simd_dot_product_avx2(ptr_f + i * 6, ptr_w, 6);
        }
        
        return alphas;
    }, py::arg("features"), py::arg("weights"),
    "Process an Nx6 array of features and 6 weights into N combined alphas using AVX2.");

    // ─── Enums ────────────────────────────────────────────────────────────

    py::enum_<Side>(m, "Side")
        .value("BID",  Side::BID)
        .value("ASK",  Side::ASK)
        .value("NONE", Side::NONE)
        .export_values();

    py::enum_<OrderType>(m, "OrderType")
        .value("LIMIT",  OrderType::LIMIT)
        .value("MARKET", OrderType::MARKET)
        .value("IOC",    OrderType::IOC)
        .value("FOK",    OrderType::FOK)
        .export_values();

    py::enum_<OrderState>(m, "OrderState")
        .value("NEW",       OrderState::NEW)
        .value("SENT",      OrderState::SENT)
        .value("PARTIAL",   OrderState::PARTIAL)
        .value("FILLED",    OrderState::FILLED)
        .value("CANCELLED", OrderState::CANCELLED)
        .value("REJECTED",  OrderState::REJECTED)
        .export_values();

    py::enum_<Regime>(m, "Regime")
        .value("NORMAL",        Regime::NORMAL)
        .value("HIGH_TOXICITY", Regime::HIGH_TOXICITY)
        .value("LOW_LIQUIDITY", Regime::LOW_LIQUIDITY)
        .value("TRENDING",      Regime::TRENDING)
        .value("UNKNOWN",       Regime::UNKNOWN)
        .export_values();

    py::enum_<DataQuality>(m, "DataQuality")
        .value("VALID",           DataQuality::VALID)
        .value("STALE_TIMESTAMP", DataQuality::STALE_TIMESTAMP)
        .value("OUT_OF_SEQUENCE", DataQuality::OUT_OF_SEQUENCE)
        .value("PRICE_ANOMALY",   DataQuality::PRICE_ANOMALY)
        .value("QTY_ANOMALY",     DataQuality::QTY_ANOMALY)
        .value("CROSSED_BOOK",    DataQuality::CROSSED_BOOK)
        .value("MISSING_LEVELS",  DataQuality::MISSING_LEVELS)
        .value("DUPLICATE",       DataQuality::DUPLICATE)
        .value("INVALID_SYMBOL",  DataQuality::INVALID_SYMBOL)
        .export_values();

    py::enum_<EngineMode>(m, "EngineMode")
        .value("BACKTEST", EngineMode::BACKTEST)
        .value("LIVE",     EngineMode::LIVE)
        .export_values();

    // Audit fix: CombinerMode was entirely absent from bindings
    py::enum_<CombinerMode>(m, "CombinerMode",
            "Signal combiner operating mode.")
        .value("WEIGHTED_AVG", CombinerMode::WEIGHTED_AVG,
               "Uniform or custom-weighted average of 6 base signals.")
        .value("ML_MODEL",     CombinerMode::ML_MODEL,
               "6-weight binary model from train_model.py::export_binary_weights().")
        .value("ONNX_MODEL",   CombinerMode::ONNX_MODEL,
               "Full LightGBM ONNX graph via onnxruntime (requires HFT_ONNX_SUPPORT).")
        .export_values();

    // ─── Config Structs ───────────────────────────────────────────────────

    py::class_<StrategyConfig>(m, "StrategyConfig",
            "Trading strategy parameters (entry/exit thresholds, sizing).")
        .def(py::init<>())
        .def_readwrite("alpha_entry_threshold", &StrategyConfig::alpha_entry_threshold,
                       "Minimum |alpha| to enter a position (default 0.10).")
        .def_readwrite("alpha_short_multiplier", &StrategyConfig::alpha_short_multiplier,
                       "Multiplier for short entry threshold (default 1.0).")
        .def_readwrite("alpha_exit_threshold",  &StrategyConfig::alpha_exit_threshold,
                       "|alpha| below this closes position (default 0.02).")
        .def_readwrite("spread_alpha_multiplier", &StrategyConfig::spread_alpha_multiplier,
                       "Widens entry threshold by factor * spread bps (default 0.05).")
        .def_readwrite("min_take_profit_bps",   &StrategyConfig::min_take_profit_bps,
                       "Minimum take profit bps required before alpha decay exit (default 5.0).")
        .def_readwrite("max_position_pct",      &StrategyConfig::max_position_pct,
                       "Order size as a fraction of portfolio (default 0.01 = 1%).")
        .def_readwrite("initial_capital",       &StrategyConfig::initial_capital,
                       "Starting capital in USD (default 100,000).")
        .def_readwrite("max_open_orders",       &StrategyConfig::max_open_orders,
                       "Maximum concurrent open orders (default 5).")
        .def_readwrite("allow_short",           &StrategyConfig::allow_short,
                       "Allow short selling (default true).")
        .def_readwrite("execution_cooldown_ns", &StrategyConfig::execution_cooldown_ns,
                       "Execution cooldown in nanoseconds (default 1s).")
        .def_readwrite("k_arrival_rate",        &StrategyConfig::k_arrival_rate,
                       "Order arrival intensity k (unused if offline)")
        .def_readwrite("fill_prob_dampener",    &StrategyConfig::fill_prob_dampener,
                       "Deterministic coin-flip for fill probability (1.0 = instant fills, < 1.0 drops fills randomly)")
        .def_readwrite("T_horizon",             &StrategyConfig::T_horizon)
        .def_readwrite("maker_fee_pct",         &StrategyConfig::maker_fee_pct, "Maker fee rebate")
        .def_readwrite("taker_fee_pct",         &StrategyConfig::taker_fee_pct, "Taker fee cost")
        .def_readwrite("max_spread_bps_cutoff", &StrategyConfig::max_spread_bps_cutoff,
                       "Hard spread circuit-breaker: veto all entries when spread exceeds this bps (default 3.5).")
        .def_readwrite("min_warmup_ticks",      &StrategyConfig::min_warmup_ticks,
                       "Ticks to process before any signal/trade is generated (default 1000). "
                       "During warm-up the engine fills its internal ring buffers but suppresses "
                       "all order routing. Set >= stat_arb_lookback for a fully warm engine.");

    py::class_<FeatureConfig>(m, "FeatureConfig",
            "Parameters for all 6 alpha signal computations.")
        .def(py::init<>())
        .def_readwrite("vpin_bucket_size",      &FeatureConfig::vpin_bucket_size,
                       "Volume per VPIN bucket in base units (default 50.0).")
        .def_readwrite("vpin_n_buckets",        &FeatureConfig::vpin_n_buckets,
                       "Rolling window of VPIN buckets (default 50).")
        .def_readwrite("vol_window_ticks",      &FeatureConfig::vol_window_ticks,
                       "Number of recent trades for realized vol (default 100).")
        .def_readwrite("stat_arb_lookback",     &FeatureConfig::stat_arb_lookback,
                       "Rolling window for stat-arb Z-score (default 1000).")
        .def_readwrite("stat_arb_zscore_entry", &FeatureConfig::stat_arb_zscore_entry,
                       "Z-score entry threshold (default 2.0).")
        .def_readwrite("stat_arb_zscore_exit",  &FeatureConfig::stat_arb_zscore_exit,
                       "Z-score exit threshold (default 0.5).")
        .def_readwrite("normalizer_min_obs",    &FeatureConfig::normalizer_min_obs,
                       "Minimum observations before online normalizer activates (default 50). "
                       "Below this count all raw-unit features are returned as 0.")
        .def_readwrite("normalizer_clamp",      &FeatureConfig::normalizer_clamp,
                       "Clamp bound in standard deviations after Z-normalization (default 3.0).");

    py::class_<RiskConfig>(m, "RiskConfig",
            "Pre-trade risk limits. All 5 checks are O(1) and noexcept.")
        .def(py::init<>())
        .def_readwrite("max_position_pct",           &RiskConfig::max_position_pct,
                       "Max absolute position in fixed-point qty.")
        .def_readwrite("max_drawdown_pct",            &RiskConfig::max_drawdown_pct,
                       "Max peak-to-trough drawdown fraction (default 0.05).")
        .def_readwrite("max_single_order_pct",        &RiskConfig::max_single_order_pct,
                       "Max single order as fraction of portfolio (default 0.05).")
        .def_readwrite("max_daily_loss_pct",          &RiskConfig::max_daily_loss_pct,
                       "Max cumulative daily loss fraction (default 0.03).")
        .def_readwrite("circuit_breaker_cooldown_ns", &RiskConfig::circuit_breaker_cooldown_ns,
                       "Cooldown after risk breach in nanoseconds (default 60s).");

    // ─── Core Market-Data Structs ─────────────────────────────────────────

    py::class_<PriceLevel>(m, "PriceLevel",
            "A single price level in the order book (price + total qty + order count).")
        .def(py::init<>())
        .def_readwrite("price",       &PriceLevel::price,
                       "Level price in fixed-point.")
        .def_readwrite("quantity",    &PriceLevel::quantity,
                       "Total quantity at this level in fixed-point.")
        .def_readwrite("order_count", &PriceLevel::order_count,
                       "Number of individual orders aggregated at this level.")
        .def("is_valid", &PriceLevel::is_valid,
             "Returns True if price != INVALID_PRICE and quantity > 0.");

    py::class_<Trade>(m, "Trade",
            "A single aggressor trade tick from the exchange feed.")
        .def(py::init<>())
        .def_readwrite("timestamp_ns", &Trade::timestamp_ns,
                       "Event timestamp in nanoseconds since Unix epoch.")
        .def_readwrite("price",        &Trade::price,
                       "Trade price in fixed-point (use price_to_fixed() to set).")
        .def_readwrite("quantity",     &Trade::quantity,
                       "Trade quantity in fixed-point (use qty_to_fixed() to set).")
        .def_readwrite("side",         &Trade::side,
                       "BID for buyer-initiated, ASK for seller-initiated.");

    py::class_<BookSnapshot>(m, "BookSnapshot",
            "Full order book state at a moment in time (up to 20 levels per side).")
        .def(py::init<>())
        .def_readwrite("timestamp_ns",    &BookSnapshot::timestamp_ns)
        .def_readwrite("sequence_num",    &BookSnapshot::sequence_num)
        .def_readwrite("bid_count",       &BookSnapshot::bid_count,
                       "Number of valid bid levels in bids[].")
        .def_readwrite("ask_count",       &BookSnapshot::ask_count,
                       "Number of valid ask levels in asks[].")
        .def_readwrite("best_bid_price",  &BookSnapshot::best_bid_price,
                       "Cached best bid price in fixed-point.")
        .def_readwrite("best_ask_price",  &BookSnapshot::best_ask_price,
                       "Cached best ask price in fixed-point.")
        .def_readwrite("best_bid_qty",    &BookSnapshot::best_bid_qty,
                       "Quantity at the best bid in fixed-point.")
        .def_readwrite("best_ask_qty",    &BookSnapshot::best_ask_qty,
                       "Quantity at the best ask in fixed-point.")
        .def_readwrite("instrument_id",   &BookSnapshot::instrument_id)
        .def_readwrite("quality",         &BookSnapshot::quality,
                       "DataQuality status of this snapshot.")
        // Expose the bids/asks C-arrays as Python lists via property
        .def_property("bids",
            [](const BookSnapshot& b) {
                std::vector<PriceLevel> v(b.bids, b.bids + b.bid_count);
                return v;
            },
            [](BookSnapshot& b, const std::vector<PriceLevel>& v) {
                int n = static_cast<int>(std::min(v.size(),
                                                  static_cast<size_t>(MAX_BOOK_LEVELS)));
                for (int i = 0; i < n; ++i) b.bids[i] = v[i];
                b.bid_count = n;
            },
            "List of PriceLevel entries on the bid side (descending price).")
        .def_property("asks",
            [](const BookSnapshot& b) {
                std::vector<PriceLevel> v(b.asks, b.asks + b.ask_count);
                return v;
            },
            [](BookSnapshot& b, const std::vector<PriceLevel>& v) {
                int n = static_cast<int>(std::min(v.size(),
                                                  static_cast<size_t>(MAX_BOOK_LEVELS)));
                for (int i = 0; i < n; ++i) b.asks[i] = v[i];
                b.ask_count = n;
            },
            "List of PriceLevel entries on the ask side (ascending price).")
        .def("mid_price",  &BookSnapshot::mid_price,
             "Compute the mid price in fixed-point (use fixed_to_price() to read).")
        .def("spread",     &BookSnapshot::spread,
             "Compute the spread in fixed-point ticks.")
        .def("is_valid",   &BookSnapshot::is_valid,
             "Returns True if the book is non-crossed and has at least 1 level per side.");

    // ─── Output Structs ───────────────────────────────────────────────────

    py::class_<FeatureVector>(m, "FeatureVector",
            "Output of the FeatureEngine — 6 alpha signals for one tick.")
        .def(py::init<>())
        .def_readwrite("timestamp_ns",    &FeatureVector::timestamp_ns)
        .def_readwrite("microprice",      &FeatureVector::microprice,
                       "Volume-weighted fair value offset (basis points from mid).")
        .def_readwrite("ofi",             &FeatureVector::ofi,
                       "Order Flow Imbalance [-1, 1].")
        .def_readwrite("vpin",            &FeatureVector::vpin,
                       "VPIN toxicity score [0, 1].")
        .def_readwrite("spread_bps",      &FeatureVector::spread_bps,
                       "Bid-ask spread in basis points.")
        .def_readwrite("realized_vol",    &FeatureVector::realized_vol,
                       "Tick-level realized volatility.")
        .def_readwrite("stat_arb_zscore", &FeatureVector::stat_arb_zscore,
                       "Z-score of mid-price vs rolling mean/std.")
        .def_readwrite("obi",             &FeatureVector::obi)
        .def_readwrite("trade_imbalance", &FeatureVector::trade_imbalance)
        .def_readwrite("hawkes_intensity", &FeatureVector::hawkes_intensity)
        .def_readwrite("cvd",             &FeatureVector::cvd)
        .def_readwrite("hurst_exponent",  &FeatureVector::hurst_exponent)
        .def_readwrite("combined_alpha",  &FeatureVector::combined_alpha,
                       "Weighted combination of all signals [-1, 1].")
        .def_readwrite("regime",          &FeatureVector::regime,
                       "Current market regime classification.")
        .def("has_valid_alpha", &FeatureVector::has_valid_alpha,
             "Returns True if combined_alpha is within [-1, 1].");

    py::class_<PerformanceMetrics>(m, "PerformanceMetrics",
            "Performance statistics updated after every fill.")
        .def_readonly("total_pnl",         &PerformanceMetrics::total_pnl)
        .def_readonly("max_drawdown",      &PerformanceMetrics::max_drawdown)
        .def_readonly("peak_equity",       &PerformanceMetrics::peak_equity)
        .def_readonly("sharpe_ratio",      &PerformanceMetrics::sharpe_ratio)
        .def_readonly("win_rate",          &PerformanceMetrics::win_rate)
        .def_readonly("avg_trade_pnl",     &PerformanceMetrics::avg_trade_pnl)
        .def_readonly("avg_slippage",      &PerformanceMetrics::avg_slippage)
        .def_readonly("total_trades",      &PerformanceMetrics::total_trades)
        .def_readonly("winning_trades",    &PerformanceMetrics::winning_trades)
        .def_readonly("losing_trades",     &PerformanceMetrics::losing_trades)
        .def_readonly("risk_rejections",   &PerformanceMetrics::risk_rejections)
        .def_readonly("signals_generated", &PerformanceMetrics::signals_generated);

    py::class_<RiskStats>(m, "RiskStats",
            "Counters for risk-based order rejections.")
        .def_readonly("orders_checked",        &RiskStats::orders_checked)
        .def_readonly("orders_passed",         &RiskStats::orders_passed)
        .def_readonly("rejected_position",     &RiskStats::rejected_position)
        .def_readonly("rejected_drawdown",     &RiskStats::rejected_drawdown)
        .def_readonly("rejected_daily_loss",   &RiskStats::rejected_daily_loss)
        .def_readonly("rejected_order_size",   &RiskStats::rejected_order_size)
        .def_readonly("rejected_circuit_brk",  &RiskStats::rejected_circuit_brk)
        .def_readonly("circuit_breaker_trips", &RiskStats::circuit_breaker_trips)
        .def("pass_rate", &RiskStats::pass_rate,
             "Fraction of orders that passed all risk checks [0.0, 1.0].");

    py::class_<TradeRecord>(m, "TradeRecord",
            "A single trade entry in the strategy journal.")
        .def_readonly("timestamp_ns", &TradeRecord::timestamp_ns)
        .def_readonly("entry_price",  &TradeRecord::entry_price)
        .def_readonly("exit_price",   &TradeRecord::exit_price)
        .def_readonly("quantity",     &TradeRecord::quantity)
        .def_readonly("pnl",          &TradeRecord::pnl)
        .def_readonly("slippage",     &TradeRecord::slippage)
        .def_readonly("side",         &TradeRecord::side);

    py::class_<StrategyEngine::PendingOrder>(m, "PendingOrder",
            "A simulated open limit order waiting in the queue.")
        .def_readonly("active",         &StrategyEngine::PendingOrder::active)
        .def_readonly("side",           &StrategyEngine::PendingOrder::side)
        .def_readonly("price",          &StrategyEngine::PendingOrder::price)
        .def_readonly("qty",            &StrategyEngine::PendingOrder::qty)
        .def_readonly("queue_position", &StrategyEngine::PendingOrder::queue_position);

    // ─── StrategyEngine ───────────────────────────────────────────────────

    py::class_<StrategyEngine>(m, "StrategyEngine", R"doc(
        Central HFT pipeline orchestrator (tick-to-trade in ~13.6 µs p50).

        Wires: OrderBook → FeatureEngine (6 signals) → SignalCombiner
               → RiskManager (5 gates) → OrderManager

        Backtest example:
            engine = hft_engine.StrategyEngine()
            for trade, book in ticks:
                engine.on_trade(trade, book)
            m = engine.metrics()
            print(f"Sharpe: {m.sharpe_ratio:.2f}, Drawdown: {m.max_drawdown:.2%}")

        Live example:
            engine.set_mode(hft_engine.EngineMode.LIVE)
            # in your async WebSocket callback:
            engine.on_trade(trade, book)
    )doc")
        .def(py::init<const StrategyConfig&, const FeatureConfig&, const RiskConfig&>(),
             py::arg("strategy_cfg") = StrategyConfig(),
             py::arg("feature_cfg")  = FeatureConfig(),
             py::arg("risk_cfg")     = RiskConfig())
        // Event handlers
        .def("on_trade",        &StrategyEngine::on_trade,
             py::arg("trade"), py::arg("book"),
             "Process one trade tick through the full pipeline (O(1), noexcept).")
        .def("on_book_update",  &StrategyEngine::on_book_update,
             py::arg("book"),
             "Update the book state without generating a signal (quote-only update).")
        // State queries
        .def("position",        &StrategyEngine::position,
             "Net position in fixed-point (use fixed_to_qty() to convert).")
        .def("realized_pnl",    &StrategyEngine::realized_pnl,
             "Cumulative realized PnL in USD.")
        .def("equity",          &StrategyEngine::equity,
             "Total equity: initial_capital + realized_pnl + unrealized_pnl.")
        .def("set_position",    &StrategyEngine::set_position,
             py::arg("pos"), "Set net position in fixed-point manually.")
        .def("set_realized_pnl", &StrategyEngine::set_realized_pnl,
             py::arg("pnl"), "Set realized PnL manually.")
        .def("set_avg_entry_price", &StrategyEngine::set_avg_entry_price,
             py::arg("px"), "Set average entry price manually.")
        .def("trade_journal",   &StrategyEngine::trade_journal,
             py::return_value_policy::reference_internal)
        .def("clear_journal",   &StrategyEngine::clear_journal)
        .def("equity_history",  &StrategyEngine::equity_history,
             "Total equity: initial_capital + realized_pnl + unrealized_pnl.")
        .def("last_features",   &StrategyEngine::last_features,
             py::return_value_policy::reference_internal)
        .def("pending_order", [](const StrategyEngine& engine) { return engine.pending_order_; },
             "Get the active open limit order currently waiting in the queue.")
        .def("metrics",         &StrategyEngine::metrics,
             "PerformanceMetrics snapshot (Sharpe, win-rate, drawdown, etc.).")
        .def("risk_stats",      &StrategyEngine::risk_stats,
             "RiskStats — rejection counters for all 5 risk gates.")

        .def("is_warmed_up",    &StrategyEngine::is_warmed_up,
             "Returns True once min_warmup_ticks have been processed. "
             "No signals or trades are generated before this point.")
        .def("tick_count",      &StrategyEngine::tick_count,
             "Total ticks processed so far (includes warm-up ticks).")
        // Control
        .def("set_mode",        &StrategyEngine::set_mode,
             py::arg("mode"),
             "Switch between EngineMode.BACKTEST (default) and EngineMode.LIVE.")
        .def("reset",           &StrategyEngine::reset,
             "Reset all state to initial (keeps config, clears positions/PnL).")
        .def("new_trading_day", &StrategyEngine::new_trading_day,
             "Mark a new trading day (resets daily stats, keeps running position).")
        .def("set_weights",
             [](StrategyEngine& self, const std::vector<double>& weights) {
                 self.set_weights(weights.data(), weights.size());
             },
             py::arg("weights"),
             "Update signal combiner weights from a Python list of floats.")
        // ── Signal Combiner control ──────────────────────────────────────
        .def("load_model",
             [](StrategyEngine& self, const std::string& path) {
                 return self.load_model(path.c_str());
             },
             py::arg("path"),
             R"doc(
             Load 6-weight binary ML model (ML_MODEL mode).

             File: signal_weights.bin from train_model.py::export_binary_weights().
             Automatically switches combiner to CombinerMode.ML_MODEL.

             Returns True on success.

             Example:
                 engine.load_model("models/signal_weights.bin")
             )doc")
        .def("load_optimal_weights",
             [](StrategyEngine& self, const std::string& path) {
                 return self.load_optimal_weights(path.c_str());
             },
             py::arg("path"),
             "Load 6 weights from a binary file into the internal weights_ array, and switch to WEIGHTED_AVG mode.")
        .def("set_stat_arb_valid",
             &StrategyEngine::set_stat_arb_valid,
             py::arg("valid"),
             "Enable or disable the StatArb signal contribution dynamically.")
        .def("load_onnx_model",
             [](StrategyEngine& self, const std::string& path, int64_t n_features) {
                 return self.load_onnx_model(path.c_str(), n_features);
             },
             py::arg("path"), py::arg("n_features") = 6LL,
             R"doc(
             Load a full LightGBM ONNX model for production inference.

             File: lgb_model.onnx from train_model.py::export_onnx().
             Automatically switches combiner to CombinerMode.ONNX_MODEL.
             Requires the engine to be built with -DHFT_ONNX_SUPPORT=ON.

             n_features must match the number of input features the ONNX
             model was exported with (see models/training_report.md).

             Returns True on success, False if ONNX support not compiled in
             or if the model file is invalid.

             Example:
                 engine.load_onnx_model("models/lgb_model.onnx", n_features=52)
             )doc")
        .def("set_combiner_mode",
             &StrategyEngine::set_combiner_mode,
             py::arg("mode"),
             "Switch CombinerMode directly. Use after load_model() or load_onnx_model().")
        .def("combiner_mode",
             &StrategyEngine::combiner_mode,
             "Get current CombinerMode (WEIGHTED_AVG, ML_MODEL, or ONNX_MODEL).")
        .def("has_model",
             &StrategyEngine::has_model,
             "Returns True if a binary ML model has been successfully loaded.")
        .def("has_onnx",
             &StrategyEngine::has_onnx,
             "Returns True if an ONNX model has been loaded and verified.")
        .def("onnx_n_features",
             &StrategyEngine::onnx_n_features,
             "Number of input features expected by the loaded ONNX model (6 if none).");

    // ─── Binance WebSocket Gateway ───────────────────────────────────────
    py::class_<BinanceWs>(m, "BinanceWs",
            "Ultra-fast native C++ Binance WebSocket Gateway (uWebSockets + simdjson)")
        .def(py::init<const std::string&>(), py::arg("symbol") = "btcusdt")
        .def("initialize", &BinanceWs::initialize)
        .def("poll_loop", [](BinanceWs& self, py::function on_trade, py::function on_book) {
            // Need to release GIL because poll_loop blocks and calls back into Python
            // Wait, poll_loop creates a thread and returns immediately in our C++ implementation!
            // So we don't need to release GIL here since it doesn't block.
            // But when C++ calls the Python callback from the background thread, it MUST acquire the GIL!
            // This is a common PyBind11 pitfall. We will wrap the callbacks to acquire GIL.
            
            auto trade_cb = [on_trade](const Trade& t) {
                py::gil_scoped_acquire acquire;
                on_trade(t);
            };
            
            auto book_cb = [on_book](const BookSnapshot& b) {
                py::gil_scoped_acquire acquire;
                on_book(b);
            };
            
            self.poll_loop(trade_cb, book_cb);
        }, py::arg("on_trade_callback"), py::arg("on_book_callback"),
        "Start polling Binance WebSockets in a background thread.")
        .def("start_live_feed", &BinanceWs::start_live_feed, py::arg("engine"),
             "Link the C++ Gateway directly to the StrategyEngine and start the background thread, entirely bypassing Python.")
        .def("stop", &BinanceWs::stop);
}
