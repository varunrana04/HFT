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
    double  alpha_entry_threshold = 3.5;   ///< Min |alpha| to enter
    double  alpha_short_multiplier= 1.0;   ///< Multiplier for short entry threshold (asymmetry)
    double  alpha_exit_threshold  = 0.02;  ///< |alpha| below this → exit

    /**
     * @brief Spread penalty multiplier — the primary "red zone" gate.
     *
     * Entry threshold is raised by (spread_bps × spread_alpha_multiplier).
     * Analysis of the 3D Alpha Surface shows the adverse-selection zone
     * clusters at spread_bps > 1.5. With multiplier = 0.14:
     *   spread = 1.0 bps → +0.14 added to threshold (mild penalty)
     *   spread = 2.0 bps → +0.28 added to threshold (moderate)
     *   spread = 3.0 bps → +0.42 added to threshold (near double)
     *   spread = 5.0 bps → +0.70 added — engine is effectively halted
     * Previous value: 0.05 (too lenient, allowed trading deep in red zone).
     */
    double  spread_alpha_multiplier = 0.14;

    /**
     * @brief Hard spread circuit-breaker. Any tick where the bid-ask
     * spread exceeds this value (in basis points) is vetoed outright —
     * no alpha calculation, no order submission.
     * Set to 3.5 bps: above this threshold the maker rebate (-0.5 bps)
     * cannot compensate for adverse selection risk identified in the
     * 3D surface red zone (spread > 3 bps + high vol).
     */
    double  max_spread_bps_cutoff  = 3.5;

    double  min_take_profit_bps   = 5.0;   ///< Minimum take profit bps required before alpha decay exit
    double  max_position_pct      = 0.15;   ///< Unified Max Position Size as % of portfolio
    double  initial_capital       = 1000000.0; ///< Starting capital ($1M Paper/Testnet Capital)
    double  maker_fee_pct         = -0.00005; ///< Maker fee (-0.5 bps institutional rebate)
    double  taker_fee_pct         = 0.00015;  ///< Taker fee (1.5 bps institutional taker)
    int64_t max_open_orders       = 5;      ///< Max concurrent orders
    bool    allow_short           = true;   ///< Allow short selling
    int64_t execution_cooldown_ns = 1000000000LL; ///< 1 second execution cooldown
    
    // Avellaneda-Stoikov Inventory Model Parameters
    double  gamma_by_regime[4]    = {0.1, 0.5, 0.1, 0.2}; ///< Risk aversion by Regime (NORMAL, HIGH_TOXICITY, LOW_LIQUIDITY, TRENDING)
    double  k_arrival_rate        = 12.9;   ///< Order arrival intensity k (First-pass k, full-session average. Rolling estimate deferred to backlog)
    
    // Advanced Optimization Toggles
    double  fill_prob_dampener    = 1.0;   ///< Fill probability for limit orders (1.0 = instant, 0.5 = 50% delay)
    double  T_horizon             = 1.0;   ///< AS Model Horizon (time remaining, e.g. 1 trading day)

    /**
     * @brief Minimum ticks to process before any signal or trade is generated.
     *
     * During this warm-up window the engine feeds data into the feature
     * engine's ring buffers (VPIN buckets, vol window, stat-arb lookback)
     * so that all signals are computed from a statistically meaningful
     * sample.  Setting this below the longest lookback (stat_arb_lookback,
     * default 1000) means some features may still be partially warm, but
     * the normalized output will be within a safe range.
     *
     * Recommended minimum: max(vol_window_ticks, vpin_n_buckets * ~10,
     *                          stat_arb_lookback)  → 1000 for defaults.
     */
    int64_t min_warmup_ticks = 1000;
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
        // Annualization: tick-level returns need sqrt(ticks_per_year).
        // BTC trades ~5M ticks/day × 365 days ≈ 1.8B ticks/year.
        // We use sqrt(252 * 6.5 * 3600) ≈ 2445 as a conservative equity-
        // market equivalent (adjust to sqrt(252*24*3600) for 24h crypto).
        // This is stored as a relative comparison metric; for a properly
        // annualized Sharpe, aggregate returns to daily first.
        static constexpr double ANNUALIZE = 2445.0;  // sqrt(252 * 6.5hr * 3600s)
        sharpe_ratio = mean / std_dev * ANNUALIZE;
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

    [[nodiscard]] const std::vector<TradeRecord>& trade_journal() const noexcept {
        return journal_;
    }

    void clear_journal() noexcept {
        journal_.clear();
    }

    [[nodiscard]] const std::vector<double>& equity_history() const noexcept {
        return equity_history_;
    }

    [[nodiscard]] const FeatureVector& last_features() const noexcept {
        return last_fv_;
    }

    /// Performance metrics snapshot
    [[nodiscard]] PerformanceMetrics metrics() const noexcept {
        PerformanceMetrics snap = metrics_;
        snap.total_pnl = realized_pnl_ + unrealized_pnl();
        return snap;
    }

    /// Risk statistics
    [[nodiscard]] const RiskStats& risk_stats() const noexcept {
        return risk_mgr_.stats();
    }



    // ── Warm-Up Query ───────────────────────────────────────
    /**
     * @brief Returns true once enough ticks have been processed that
     *        all feature buffers are meaningfully populated.
     *
     * No trades or signals are generated while this returns false.
     */
    [[nodiscard]] bool is_warmed_up() const noexcept {
        return tick_count_ >= strategy_.min_warmup_ticks;
    }

    /// Ticks processed so far (useful for monitoring warm-up progress)
    [[nodiscard]] int64_t tick_count() const noexcept { return tick_count_; }

    // ── Open Order State ──
    struct PendingOrder {
        bool active = false;
        Side side = Side::NONE;
        int64_t price = 0;
        int64_t qty = 0;
        int64_t queue_position = 0;
        int64_t timestamp_ns = 0;
        int64_t max_allowed_wait_ns = 2500000000LL; // 2.5 seconds
    };
    PendingOrder pending_order_;

    // ── Processing ─────────────────────────────────────────────
    /// Set engine mode (BACKTEST or LIVE)
    void set_mode(EngineMode mode) noexcept { mode_ = mode; }

    /// Reset all state to initial
    void reset() noexcept;

    /// Reset for a new trading day (keeps position, resets daily stats)
    void new_trading_day() noexcept;

    /// Set signal combiner weights (6-element array)
    void set_weights(const double* weights, size_t count) noexcept {
        combiner_.set_weights(weights, count);
    }

    // ── Signal Combiner forwarding ───────────────────────────
    /**
     * @brief Load 6-weight binary ML model (ML_MODEL mode).
     * @param path  Path to signal_weights.bin from train_model.py
     * @return true on success
     */
    [[nodiscard]] bool load_model(const char* path) noexcept {
        return combiner_.load_model(path);
    }

    [[nodiscard]] bool load_optimal_weights(const char* path) noexcept {
        return combiner_.load_optimal_weights(path);
    }
    
    void set_stat_arb_valid(bool valid) noexcept {
        combiner_.set_stat_arb_valid(valid);
    }

    /**
     * @brief Load a full LightGBM ONNX model for production inference.
     *
     * Switches combiner to ONNX_MODEL mode automatically.
     * n_features must match the number of input features the model
     * was exported with (check training_report.md for the count).
     *
     * Requires -DHFT_ONNX_SUPPORT at compile time.
     *
     * @param path       Path to lgb_model.onnx
     * @param n_features Number of input features (default 6 = base only)
     * @return true on success
     */
    [[nodiscard]] bool load_onnx_model(const char* path,
                                        int64_t n_features = 6) noexcept {
        return combiner_.load_onnx_model(path, n_features);
    }

    /// Switch combiner mode directly
    void set_combiner_mode(CombinerMode m) noexcept { combiner_.set_mode(m); }

    /// Query current combiner mode
    [[nodiscard]] CombinerMode combiner_mode() const noexcept {
        return combiner_.mode();
    }

    /// True if a binary ML model has been loaded
    [[nodiscard]] bool has_model() const noexcept { return combiner_.has_model(); }

    /// True if an ONNX model has been loaded and verified
    [[nodiscard]] bool has_onnx() const noexcept { return combiner_.has_onnx(); }

    /// Number of features expected by the loaded ONNX model (6 if none loaded)
    [[nodiscard]] int64_t onnx_n_features() const noexcept {
        return combiner_.onnx_n_features();
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
    int64_t  session_start_ns_ = 0;    ///< First tick timestamp of the day
    int64_t  last_funding_ts_ns_ = 0;  ///< Last absolute funding epoch boundary cleared
    int64_t  last_trade_ns_  = 0;      ///< Timestamp of last trade execution

    // ── Last seen L2 book (for book-sweep simulation in simulate_fill) ──
    BookSnapshot last_book_  = {};     ///< Snapshot of the most recent valid book

    // ── Outputs ─────────────────────────────────────────────
    FeatureVector            last_fv_  = {};
    PerformanceMetrics       metrics_  = {};
    std::vector<TradeRecord> journal_;
    std::vector<double>      equity_history_;

    // ── Internal Methods ────────────────────────────────────
    /// Compute order size based on alpha strength and portfolio value
    [[nodiscard]] int64_t compute_order_size(
        double alpha, double portfolio_value) const noexcept;

    /// Execute a simulated fill (backtest mode)
    void simulate_fill(Side side, int64_t price,
                       int64_t quantity, bool is_taker = false) noexcept;

    /// Update performance metrics after a fill
    void update_metrics(double trade_pnl, double slippage) noexcept;

    /// Record equity return for Sharpe calculation
    void record_return() noexcept;
};

} // namespace hft
