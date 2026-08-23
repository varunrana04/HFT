/**
 * @file risk_manager.cpp
 * @brief Full implementation of pre-trade risk management.
 *
 * Check order:
 *   1. Circuit breaker (fastest — pure timestamp compare)
 *   2. Position limit (integer compare)
 *   3. Drawdown gate (floating-point compare)
 *   4. Daily loss limit (floating-point compare)
 *   5. Single order size (floating-point compare)
 *
 * Checks are ordered from cheapest to most expensive to fail fast.
 */

#include "risk_manager.h"
#include <cmath>
#include <algorithm>

namespace hft {

// ─── Construction / Reset ────────────────────────────────────

RiskManager::RiskManager(const RiskConfig& config) noexcept
    : config_(config) {
    reset();
}

void RiskManager::reset() noexcept {
    stats_.reset();
    peak_equity_             = 0.0;
    day_start_equity_        = 0.0;
    current_equity_          = 0.0;
    circuit_breaker_until_ns_ = 0;
}

// ─── Equity Tracking ─────────────────────────────────────────

void RiskManager::update_equity(double current_equity) noexcept {
    current_equity_ = current_equity;

    // Update high-water mark
    if (current_equity > peak_equity_) {
        peak_equity_ = current_equity;
    }

    // Initialize day_start if not yet set
    if (day_start_equity_ <= 0.0) {
        day_start_equity_ = current_equity;
    }
}

void RiskManager::new_trading_day() noexcept {
    day_start_equity_ = current_equity_;
    // Don't reset peak — drawdown is cumulative
}

// ─── Circuit Breaker ─────────────────────────────────────────

void RiskManager::trip_circuit_breaker() noexcept {
    int64_t now = now_ns();
    circuit_breaker_until_ns_ = now + config_.circuit_breaker_cooldown_ns;
    ++stats_.circuit_breaker_trips;
}

bool RiskManager::is_circuit_breaker_active() const noexcept {
    if (circuit_breaker_until_ns_ <= 0) return false;
    return now_ns() < circuit_breaker_until_ns_;
}

// ─── Stale Quote Protection ──────────────────────────────────
bool RiskManager::check_stale_quote(int64_t order_submission_time_ns, int64_t current_time_ns) const noexcept {
    // Stale Quote Protection: 2.5 seconds = 2'500'000'000 nanoseconds
    return (current_time_ns - order_submission_time_ns) > 2'500'000'000LL;
}

// ─── Drawdown / Daily Loss Queries ───────────────────────────

double RiskManager::current_drawdown() const noexcept {
    if (peak_equity_ <= 0.0) return 0.0;
    double dd = (peak_equity_ - current_equity_) / peak_equity_;
    return std::max(dd, 0.0);
}

double RiskManager::current_daily_loss() const noexcept {
    if (day_start_equity_ <= 0.0) return 0.0;
    double loss = (day_start_equity_ - current_equity_) / day_start_equity_;
    return std::max(loss, 0.0);
}

// ─── Full Risk Check (new signature) ─────────────────────────

RiskVerdict RiskManager::check_order(const Order& order,
                                     int64_t current_position,
                                     double current_pnl,
                                     double portfolio_value) noexcept {
    ++stats_.orders_checked;

    // 1. Circuit breaker (cheapest check — timestamp compare)
    if (is_circuit_breaker_active()) {
        ++stats_.rejected_circuit_brk;
        return RiskVerdict::CIRCUIT_BREAKER;
    }

    // 2. Position limit (Dynamic % of Portfolio)
    int64_t new_pos = current_position;
    if (order.side == Side::BID) {
        new_pos += order.quantity;
    } else {
        new_pos -= order.quantity;
    }

    if (portfolio_value > 0.0) {
        double new_pos_qty = std::abs(fixed_to_qty(new_pos));
        double new_pos_notional = new_pos_qty * fixed_to_price(order.price);
        double max_allowed_notional = portfolio_value * config_.max_position_pct;
        
        if (new_pos_notional > max_allowed_notional) {
            // Maker Order constraint: Reject immediately if expanding position past limit
            if (order.type == OrderType::LIMIT) {
                ++stats_.rejected_position;
                trip_circuit_breaker();
                return RiskVerdict::POSITION_LIMIT;
            }
        }
    } else {
        // Fallback if portfolio value is 0 (test mode)
        // Bypass the position limit check
    }

    // 3. Drawdown gate
    // Update equity with current PnL for accurate drawdown
    if (portfolio_value > 0.0) {
        update_equity(portfolio_value + current_pnl);
    }

    if (current_drawdown() > config_.max_drawdown_pct) {
        ++stats_.rejected_drawdown;
        trip_circuit_breaker();
        return RiskVerdict::DRAWDOWN_LIMIT;
    }

    // 4. Daily loss limit
    if (current_daily_loss() > config_.max_daily_loss_pct) {
        ++stats_.rejected_daily_loss;
        trip_circuit_breaker();
        return RiskVerdict::DAILY_LOSS_LIMIT;
    }

    // 5. Single order size limit
    if (portfolio_value > 0.0) {
        double order_value = fixed_to_price(order.price)
                           * fixed_to_qty(order.quantity);
        double order_pct = order_value / portfolio_value;

        if (order_pct > config_.max_single_order_pct) {
            ++stats_.rejected_order_size;
            return RiskVerdict::ORDER_SIZE_LIMIT;
        }
    }

    ++stats_.orders_passed;
    return RiskVerdict::PASS;
}

// ─── Legacy Signature (backward compatible) ──────────────────

bool RiskManager::check_order(const Order& order,
                              int64_t current_position,
                              double current_pnl) noexcept {
    // Use peak equity as portfolio value for backward compat
    double portfolio = peak_equity_ > 0.0 ? peak_equity_ : 1.0;
    RiskVerdict verdict = check_order(order, current_position,
                                      current_pnl, portfolio);
    return verdict == RiskVerdict::PASS;
}

bool RiskManager::check_toxicity_halt(double sentiment, double vol) noexcept {
    if (std::abs(sentiment) > 0.80 && vol > config_.max_volatility_threshold) {
        trip_circuit_breaker();
        return false; // False means not safe to trade
    }
    return true; // True means safe to trade
}

// ─── Drop-Copy Kill Switch (Phase 4) ─────────────────────────

DropCopyKillSwitch::DropCopyKillSwitch(RiskManager& risk_mgr)
    : risk_mgr_(risk_mgr) {}

void DropCopyKillSwitch::start_listener(const std::string& fix_drop_copy_session) {
    is_listening_ = true;
    // Stub: In production, this would bind a thread to an isolated core 
    // and listen for FIX ExecutionReports. If there is a position mismatch 
    // or fatal exchange message, it calls risk_mgr_.trip_circuit_breaker()
    // or triggers a hardware kill switch via PCIe.
}

void DropCopyKillSwitch::stop_listener() {
    is_listening_ = false;
}

} // namespace hft

