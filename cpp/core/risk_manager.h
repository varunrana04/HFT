#pragma once
/**
 * @file risk_manager.h
 * @brief Comprehensive pre-trade risk management.
 *
 * Enforces the following risk limits on every order:
 *   1. Position limit  — max absolute position size
 *   2. Drawdown gate   — halt if peak-to-current drawdown exceeds threshold
 *   3. Daily loss limit — halt if cumulative daily loss exceeds threshold
 *   4. Single order size — reject orders exceeding % of portfolio
 *   5. Circuit breaker  — cooldown period after any risk breach
 *
 * All checks are O(1) and noexcept. The circuit breaker uses
 * nanosecond timestamps from clock.h for precision.
 */

#include "types.h"
#include "clock.h"
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace hft {

// ─── Risk Statistics ─────────────────────────────────────────
/**
 * @brief Counters for risk-based order rejections.
 *
 * Monitor these to detect when the system is being risk-gated.
 * A spike in rejections may indicate a volatile market or
 * a strategy that needs recalibration.
 */
struct RiskStats {
    uint64_t orders_checked        = 0;
    uint64_t orders_passed         = 0;
    uint64_t rejected_position     = 0;  ///< Position limit breaches
    uint64_t rejected_drawdown     = 0;  ///< Drawdown gate triggers
    uint64_t rejected_daily_loss   = 0;  ///< Daily loss limit triggers
    uint64_t rejected_order_size   = 0;  ///< Single order size limit
    uint64_t rejected_circuit_brk  = 0;  ///< Circuit breaker cooldown
    uint64_t circuit_breaker_trips = 0;  ///< Total circuit breaker activations

    [[nodiscard]] double pass_rate() const noexcept {
        if (orders_checked == 0) return 1.0;
        return static_cast<double>(orders_passed) /
               static_cast<double>(orders_checked);
    }

    void reset() noexcept {
        orders_checked = orders_passed = 0;
        rejected_position = rejected_drawdown = rejected_daily_loss = 0;
        rejected_order_size = rejected_circuit_brk = 0;
        circuit_breaker_trips = 0;
    }
};

// ─── Risk Configuration ──────────────────────────────────────
/**
 * @brief All risk thresholds. Defaults match config/default_config.yaml.
 */
struct RiskConfig {
    int64_t max_position           = 100 * QTY_SCALE;  ///< Max abs position
    double  max_drawdown_pct       = 0.05;   ///< 5% max peak-to-trough
    double  max_single_order_pct   = 0.02;   ///< 2% of portfolio per order
    double  max_daily_loss_pct     = 0.03;   ///< 3% max daily loss
    int64_t circuit_breaker_cooldown_ns =
        60LL * 1'000'000'000LL;              ///< 60 seconds in nanoseconds
};

// ─── Risk Check Result ───────────────────────────────────────
/**
 * @brief Reason an order was rejected (or passed).
 */
enum class RiskVerdict : uint8_t {
    PASS              = 0,
    POSITION_LIMIT    = 1,
    DRAWDOWN_LIMIT    = 2,
    DAILY_LOSS_LIMIT  = 3,
    ORDER_SIZE_LIMIT  = 4,
    CIRCUIT_BREAKER   = 5
};

// ─── Risk Manager ────────────────────────────────────────────
/**
 * @brief Pre-trade risk gatekeeper.
 *
 * Call check_order() before every order submission. If it returns
 * anything other than PASS, the order MUST be rejected.
 *
 * Call update_pnl() after every fill to keep equity tracking current.
 * Call new_trading_day() at the start of each day to reset daily loss.
 */
class RiskManager {
public:
    explicit RiskManager(const RiskConfig& config = {}) noexcept;

    /**
     * @brief Run all risk checks on a proposed order.
     *
     * @param order            The order to validate
     * @param current_position Current net position (signed, fixed-point)
     * @param current_pnl      Current unrealized + realized PnL
     * @param portfolio_value  Current total portfolio value
     * @return RiskVerdict::PASS if order is allowed
     */
    RiskVerdict check_order(const Order& order,
                            int64_t current_position,
                            double current_pnl,
                            double portfolio_value) noexcept;

    /**
     * @brief Legacy check_order signature for backward compatibility.
     *
     * Uses stored equity for drawdown/daily loss calculations.
     * Portfolio value defaults to peak_equity_ for order size checks.
     */
    bool check_order(const Order& order,
                     int64_t current_position,
                     double current_pnl) noexcept;

    /**
     * @brief Update PnL tracking after a fill or mark-to-market.
     *
     * @param current_equity Current total equity (capital + unrealized PnL)
     */
    void update_equity(double current_equity) noexcept;

    /**
     * @brief Reset daily loss tracking (call at start of each trading day).
     */
    void new_trading_day() noexcept;

    /**
     * @brief Reset all state (position, equity, circuit breaker).
     */
    void reset() noexcept;

    /// Get risk statistics
    [[nodiscard]] const RiskStats& stats() const noexcept {
        return stats_;
    }

    /// Check if circuit breaker is currently active
    [[nodiscard]] bool is_circuit_breaker_active() const noexcept;

    /// Get current drawdown as a fraction [0, 1]
    [[nodiscard]] double current_drawdown() const noexcept;

    /// Get current daily loss as a fraction [0, 1]
    [[nodiscard]] double current_daily_loss() const noexcept;

private:
    RiskConfig config_;
    RiskStats  stats_;

    // Equity tracking
    double peak_equity_      = 0.0;  ///< High-water mark for drawdown
    double day_start_equity_ = 0.0;  ///< Equity at start of trading day
    double current_equity_   = 0.0;  ///< Latest known equity

    // Circuit breaker state
    int64_t circuit_breaker_until_ns_ = 0;  ///< Timestamp when cooldown ends

    /// Activate the circuit breaker for the configured cooldown
    void trip_circuit_breaker() noexcept;
};

} // namespace hft
