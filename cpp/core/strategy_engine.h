#pragma once
/**
 * @file strategy_engine.h
 * @brief Central orchestrator — wires OrderBook, FeatureEngine,
 *        SignalCombiner, RiskManager, and OrderManager into a
 *        deterministic event-driven trading pipeline.
 *
 * The StrategyEngine receives raw market events (trades and book
 * snapshots) and produces trading decisions. It operates in two modes:
 *   - BACKTEST: Deterministic replay of historical data
 *   - LIVE:     Real-time event processing with wall-clock timestamps
 *
 * Pipeline per tick:
 *   Trade/Book → OrderBook update → FeatureEngine → SignalCombiner
 *   → α > threshold? → RiskManager gate → OrderManager → emit Order
 *
 * All hot-path operations are O(1), noexcept, and zero-allocation.
 */

#include "types.h"
#include "order_book.h"
#include "features.h"
#include "signal_combiner.h"
#include "risk_manager.h"
#include "order_manager.h"
#include "clock.h"

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

namespace hft {

// ─── Engine Mode ─────────────────────────────────────────────
enum class EngineMode : uint8_t {
    BACKTEST = 0,  ///< Deterministic replay, no wall-clock
    LIVE     = 1   ///< Real-time, uses now_ns() for timestamps
};

// ─── Strategy Configuration ─────────────────────────────────
struct StrategyConfig {
    double  alpha_entry_threshold = 0.10;  ///< Min |alpha| to enter
    double  alpha_exit_threshold  = 0.02;  ///< |alpha| below this → exit
    double  position_size_pct    = 0.01;   ///< Size as % of portfolio
    double  initial_capital      = 100000.0; ///< Starting capital (USD)
    int64_t max_open_orders      = 5;      ///< Max concurrent orders
    bool    allow_short          = true;    ///< Allow short selling
};

// ─── Trade Record (for journaling) ──────────────────────────
struct TradeRecord {
    int64_t timestamp_ns;
    int64_t entry_price;      ///< Fixed-point
    int64_t exit_price;       ///< Fixed-point (0 if still open)
    int64_t quantity;          ///< Fixed-point (signed: + = long, - = short)
    double  pnl;              ///< Realized PnL for this trade
    double  slippage;          ///< In price units
    Side    side;
};

// ─── Performance Metrics ────────────────────────────────────
struct PerformanceMetrics {
    double  total_pnl           = 0.0;
    double  max_drawdown        = 0.0;
    double  peak_equity         = 0.0;
    double  sharpe_ratio        = 0.0;
    double  win_rate            = 0.0;
    double  avg_trade_pnl       = 0.0;
    double  avg_slippage        = 0.0;
    int64_t total_trades        = 0;
    int64_t winning_trades      = 0;
    int64_t losing_trades       = 0;
    int64_t risk_rejections     = 0;
    int64_t signals_generated   = 0;

    // For Sharpe calculation (running)
    double  sum_returns         = 0.0;
    double  sum_returns_sq      = 0.0;
    int64_t return_count        = 0;

    void update_sharpe() noexcept {
        if (return_count < 2) { sharpe_ratio = 0.0; return; }
        double mean = sum_returns / static_cast<double>(return_count);
        double var  = (sum_returns_sq / static_cast<double>(return_count))
                      - (mean * mean);
        double std_dev = std::sqrt(std::max(var, 1e-15));
        // Annualized: assume 252 trading days, many ticks per day
        sharpe_ratio = mean / std_dev * std::sqrt(252.0);
    }
};

// ─── Strategy Engine ────────────────────────────────────────
/**
 * @brief Central trading pipeline orchestrator.
 *
 * Usage (backtest):
 *   StrategyEngine engine(strategy_cfg, feature_cfg, risk_cfg);
 *   for (auto& [book, trade] : historical_data) {
 *       engine.on_trade(trade, book);
 *   }
 *   auto metrics = engine.metrics();
 *
 * Usage (live via Python):
 *   engine.set_mode(EngineMode::LIVE);
 *   // called from mt5_gateway.py on each tick
 *   engine.on_trade(trade, book);
 */
class StrategyEngine {
public:
    StrategyEngine(const StrategyConfig& strategy_cfg = {},
                   const FeatureConfig& feature_cfg = {},
                   const RiskConfig& risk_cfg = {}) noexcept;

    // ── Event Handlers ──────────────────────────────────────
    /**
     * @brief Process a new trade tick through the full pipeline.
     *
     * This is the main entry point. It:
     *   1. Updates the order book with the trade
     *   2. Computes all 6 alpha signals
     *   3. Combines signals into a single alpha score
     *   4. If |alpha| > threshold, checks risk and emits order
     *   5. Updates PnL and performance metrics
     *
     * @param trade  Latest trade event
     * @param book   Current order book snapshot
     */
    void on_trade(const Trade& trade, const BookSnapshot& book) noexcept;

    /**
     * @brief Process a book-only update (no trade).
     *
     * Updates order book state and feature engine, but does NOT
     * generate trading signals (to avoid double-counting).
     */
    void on_book_update(const BookSnapshot& book) noexcept;

    // ── State Queries ───────────────────────────────────────
    /// Current net position (signed, fixed-point)
    [[nodiscard]] int64_t position() const noexcept { return position_; }

    /// Current unrealized PnL
    [[nodiscard]] double unrealized_pnl() const noexcept;

    /// Current realized PnL
    [[nodiscard]] double realized_pnl() const noexcept { return realized_pnl_; }

    /// Total equity (capital + realized + unrealized)
    [[nodiscard]] double equity() const noexcept;

    /// Latest computed feature vector
    [[nodiscard]] const FeatureVector& last_features() const noexcept {
        return last_fv_;
    }

    /// Performance metrics snapshot
    [[nodiscard]] const PerformanceMetrics& metrics() const noexcept {
        return metrics_;
    }

    /// Risk statistics
    [[nodiscard]] const RiskStats& risk_stats() const noexcept {
        return risk_mgr_.stats();
    }

    /// Trade journal
    [[nodiscard]] const std::vector<TradeRecord>& trade_journal() const noexcept {
        return journal_;
    }

    // ── Control ─────────────────────────────────────────────
    /// Set engine mode (BACKTEST or LIVE)
    void set_mode(EngineMode mode) noexcept { mode_ = mode; }

    /// Reset all state to initial
    void reset() noexcept;

    /// Reset for a new trading day (keeps position, resets daily stats)
    void new_trading_day() noexcept;

    /// Set signal combiner weights
    void set_weights(const double* weights, size_t count) noexcept {
        combiner_.set_weights(weights, count);
    }

private:
    // ── Configuration ───────────────────────────────────────
    StrategyConfig strategy_;
    EngineMode     mode_ = EngineMode::BACKTEST;

    // ── Sub-Engines ─────────────────────────────────────────
    FeatureEngine  features_;
    SignalCombiner combiner_;
    RiskManager    risk_mgr_;
    OrderManager   order_mgr_;

    // ── State ───────────────────────────────────────────────
    int64_t  position_       = 0;      ///< Net position (signed, fixed-point)
    double   realized_pnl_   = 0.0;    ///< Cumulative realized PnL
    double   avg_entry_price_ = 0.0;   ///< VWAP of current position
    int64_t  last_mid_price_ = 0;      ///< Latest mid price (fixed-point)
    int64_t  tick_count_     = 0;      ///< Total ticks processed
    double   prev_equity_    = 0.0;    ///< Previous tick's equity (for returns)

    // ── Outputs ─────────────────────────────────────────────
    FeatureVector            last_fv_  = {};
    PerformanceMetrics       metrics_  = {};
    std::vector<TradeRecord> journal_;

    // ── Internal Methods ────────────────────────────────────
    /// Compute order size based on alpha strength and portfolio value
    [[nodiscard]] int64_t compute_order_size(
        double alpha, double portfolio_value) const noexcept;

    /// Execute a simulated fill (backtest mode)
    void simulate_fill(Side side, int64_t price,
                       int64_t quantity) noexcept;

    /// Update performance metrics after a fill
    void update_metrics(double trade_pnl, double slippage) noexcept;

    /// Record equity return for Sharpe calculation
    void record_return() noexcept;
};

} // namespace hft
