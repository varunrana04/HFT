/**
 * @file strategy_engine.cpp
 * @brief Implementation of the central StrategyEngine orchestrator.
 *
 * Pipeline per tick:
 *   Trade → FeatureEngine.compute_all() → SignalCombiner.combine()
 *   → alpha check → RiskManager.check_order() → simulate_fill()
 *   → update PnL, metrics, journal
 */

#include "strategy_engine.h"
#include <cstring>

namespace hft {

// ─── Constructor ────────────────────────────────────────────
StrategyEngine::StrategyEngine(const StrategyConfig& strategy_cfg,
                               const FeatureConfig& feature_cfg,
                               const RiskConfig& risk_cfg) noexcept
    : strategy_(strategy_cfg),
      features_(feature_cfg),
      combiner_(),
      risk_mgr_(risk_cfg),
      order_mgr_() {
    journal_.reserve(4096);  // Pre-allocate for backtest
    prev_equity_ = strategy_.initial_capital;
    risk_mgr_.update_equity(strategy_.initial_capital);
}

// ─── Main Event Handler ─────────────────────────────────────
void StrategyEngine::on_trade(const Trade& trade,
                              const BookSnapshot& book) noexcept {
    ++tick_count_;

    // 1. Update cached mid price
    if (book.is_valid()) {
        last_mid_price_ = book.mid_price();
    }

    // 2. Compute all 6 alpha signals
    last_fv_ = features_.compute_all(book, trade);

    // 3. Combine signals into single alpha score
    last_fv_.combined_alpha = combiner_.combine(last_fv_);
    double alpha = last_fv_.combined_alpha;

    // 4. Record equity return for Sharpe calculation
    record_return();

    // 5. Determine if we should trade
    double abs_alpha = std::abs(alpha);

    // ── EXIT LOGIC: Close position if alpha flipped or decayed ──
    if (position_ != 0 && abs_alpha < strategy_.alpha_exit_threshold) {
        // Close the position
        Side close_side = (position_ > 0) ? Side::ASK : Side::BID;
        int64_t close_qty = std::abs(position_);

        // Build exit order
        int64_t exit_price = (close_side == Side::ASK)
            ? book.best_bid_price   // Sell at bid
            : book.best_ask_price;  // Buy at ask

        if (exit_price != INVALID_PRICE && exit_price > 0) {
            Order exit_order = order_mgr_.create_order(
                close_side, exit_price, close_qty,
                OrderType::MARKET, exit_price);

            // Risk check (exits are almost always allowed)
            double current_eq = equity();
            RiskVerdict verdict = risk_mgr_.check_order(
                exit_order, position_, realized_pnl_, current_eq);

            if (verdict == RiskVerdict::PASS) {
                simulate_fill(close_side, exit_price, close_qty);
            }
            // Even if risk rejects exit, we don't count it as rejection
        }
        return;
    }

    // ── ENTRY LOGIC: Open or add to position if alpha is strong ──
    if (abs_alpha >= strategy_.alpha_entry_threshold) {
        ++metrics_.signals_generated;

        // Direction: positive alpha → BUY, negative alpha → SELL
        Side entry_side = (alpha > 0) ? Side::BID : Side::ASK;

        // Check: don't add to a position in the same direction
        // if we're already at size, or in opposite direction if shorts
        // are disabled
        if (!strategy_.allow_short && entry_side == Side::ASK && position_ <= 0) {
            return;  // Can't short
        }

        // Compute order size
        double current_eq = equity();
        int64_t order_qty = compute_order_size(alpha, current_eq);
        if (order_qty <= 0) return;

        // Execution price: market-take at the opposite side
        int64_t exec_price = (entry_side == Side::BID)
            ? book.best_ask_price   // Buy at ask (taking liquidity)
            : book.best_bid_price;  // Sell at bid (taking liquidity)

        if (exec_price == INVALID_PRICE || exec_price <= 0) return;

        // Build entry order
        Order entry_order = order_mgr_.create_order(
            entry_side, exec_price, order_qty,
            OrderType::MARKET, last_mid_price_);

        // Risk gate
        RiskVerdict verdict = risk_mgr_.check_order(
            entry_order, position_, realized_pnl_, current_eq);

        if (verdict == RiskVerdict::PASS) {
            simulate_fill(entry_side, exec_price, order_qty);
        } else {
            ++metrics_.risk_rejections;
        }
    }
}

// ─── Book-Only Update ───────────────────────────────────────
void StrategyEngine::on_book_update(const BookSnapshot& book) noexcept {
    if (book.is_valid()) {
        last_mid_price_ = book.mid_price();
    }
    // Update equity tracking with new mark-to-market
    risk_mgr_.update_equity(equity());
}

// ─── State Queries ──────────────────────────────────────────
double StrategyEngine::unrealized_pnl() const noexcept {
    if (position_ == 0 || last_mid_price_ == 0) return 0.0;
    double current_price = fixed_to_price(last_mid_price_);
    double pos_qty = fixed_to_qty(std::abs(position_));

    if (position_ > 0) {
        // Long: profit if price went up
        return (current_price - avg_entry_price_) * pos_qty;
    } else {
        // Short: profit if price went down
        return (avg_entry_price_ - current_price) * pos_qty;
    }
}

double StrategyEngine::equity() const noexcept {
    return strategy_.initial_capital + realized_pnl_ + unrealized_pnl();
}

// ─── Compute Order Size ─────────────────────────────────────
int64_t StrategyEngine::compute_order_size(
    double alpha, double portfolio_value) const noexcept {
    if (portfolio_value <= 0.0 || last_mid_price_ == 0) return 0;

    double price = fixed_to_price(last_mid_price_);
    if (price <= 0.0) return 0;

    // Scale position size by alpha strength (stronger signal → bigger size)
    // Clamped to [0.5, 2.0] × base size to avoid extreme bets
    double alpha_scale = std::clamp(std::abs(alpha) * 10.0, 0.5, 2.0);
    double notional = portfolio_value * strategy_.position_size_pct * alpha_scale;
    double qty = notional / price;

    return qty_to_fixed(qty);
}

// ─── Simulate Fill (Backtest) ───────────────────────────────
void StrategyEngine::simulate_fill(Side side, int64_t price,
                                   int64_t quantity) noexcept {
    double fill_price = fixed_to_price(price);
    double fill_qty   = fixed_to_qty(quantity);
    double trade_pnl  = 0.0;
    double slippage   = 0.0;

    // Calculate slippage vs mid price
    if (last_mid_price_ > 0) {
        double mid = fixed_to_price(last_mid_price_);
        if (side == Side::BID) {
            slippage = fill_price - mid;  // Positive = overpaid
        } else {
            slippage = mid - fill_price;  // Positive = undersold
        }
    }

    if (side == Side::BID) {
        // ── BUYING ──
        if (position_ < 0) {
            // Closing/reducing short position → realize PnL
            int64_t close_qty = std::min(quantity, -position_);
            double close_qty_d = fixed_to_qty(close_qty);
            trade_pnl = (avg_entry_price_ - fill_price) * close_qty_d;
            realized_pnl_ += trade_pnl;

            position_ += close_qty;

            // If we bought more than the short, the rest opens a long
            int64_t remaining = quantity - close_qty;
            if (remaining > 0) {
                avg_entry_price_ = fill_price;
                position_ += remaining;
            } else if (position_ == 0) {
                avg_entry_price_ = 0.0;
            }
        } else {
            // Opening/adding to long position → update VWAP
            double old_notional = avg_entry_price_ * fixed_to_qty(position_);
            double new_notional = fill_price * fill_qty;
            position_ += quantity;
            if (position_ > 0) {
                avg_entry_price_ = (old_notional + new_notional)
                                   / fixed_to_qty(position_);
            }
        }
    } else {
        // ── SELLING ──
        if (position_ > 0) {
            // Closing/reducing long position → realize PnL
            int64_t close_qty = std::min(quantity, position_);
            double close_qty_d = fixed_to_qty(close_qty);
            trade_pnl = (fill_price - avg_entry_price_) * close_qty_d;
            realized_pnl_ += trade_pnl;

            position_ -= close_qty;

            // If we sold more than the long, the rest opens a short
            int64_t remaining = quantity - close_qty;
            if (remaining > 0) {
                avg_entry_price_ = fill_price;
                position_ -= remaining;
            } else if (position_ == 0) {
                avg_entry_price_ = 0.0;
            }
        } else {
            // Opening/adding to short position → update VWAP
            double old_notional = avg_entry_price_ * fixed_to_qty(-position_);
            double new_notional = fill_price * fill_qty;
            position_ -= quantity;
            if (position_ < 0) {
                avg_entry_price_ = (old_notional + new_notional)
                                   / fixed_to_qty(-position_);
            }
        }
    }

    // ── Journal Entry ──
    TradeRecord record{};
    record.timestamp_ns = last_fv_.timestamp_ns;
    record.entry_price  = price;
    record.exit_price   = 0;  // Filled at entry
    record.quantity     = (side == Side::BID) ? quantity : -quantity;
    record.pnl          = trade_pnl;
    record.slippage     = slippage;
    record.side         = side;
    journal_.push_back(record);

    // ── Update Metrics ──
    update_metrics(trade_pnl, slippage);

    // ── Update Risk Manager ──
    risk_mgr_.update_equity(equity());
}

// ─── Update Performance Metrics ─────────────────────────────
void StrategyEngine::update_metrics(double trade_pnl,
                                    double slippage) noexcept {
    ++metrics_.total_trades;
    metrics_.total_pnl = realized_pnl_ + unrealized_pnl();

    if (trade_pnl > 0.0) ++metrics_.winning_trades;
    else if (trade_pnl < 0.0) ++metrics_.losing_trades;

    if (metrics_.total_trades > 0) {
        metrics_.win_rate = static_cast<double>(metrics_.winning_trades)
                            / static_cast<double>(metrics_.total_trades);
        metrics_.avg_trade_pnl = realized_pnl_
                                 / static_cast<double>(metrics_.total_trades);
    }

    // Drawdown tracking
    double eq = equity();
    if (eq > metrics_.peak_equity) metrics_.peak_equity = eq;
    double dd = (metrics_.peak_equity > 0.0)
        ? (metrics_.peak_equity - eq) / metrics_.peak_equity
        : 0.0;
    if (dd > metrics_.max_drawdown) metrics_.max_drawdown = dd;

    // Slippage tracking (exponential moving average)
    double n = static_cast<double>(metrics_.total_trades);
    metrics_.avg_slippage = metrics_.avg_slippage * ((n - 1.0) / n)
                            + slippage * (1.0 / n);
}

// ─── Record Return (for Sharpe) ─────────────────────────────
void StrategyEngine::record_return() noexcept {
    double eq = equity();
    if (prev_equity_ > 0.0 && tick_count_ > 1) {
        double ret = (eq - prev_equity_) / prev_equity_;
        metrics_.sum_returns    += ret;
        metrics_.sum_returns_sq += ret * ret;
        ++metrics_.return_count;
        metrics_.update_sharpe();
    }
    prev_equity_ = eq;
}

// ─── Reset ──────────────────────────────────────────────────
void StrategyEngine::reset() noexcept {
    features_.reset();
    risk_mgr_.reset();
    position_        = 0;
    realized_pnl_    = 0.0;
    avg_entry_price_ = 0.0;
    last_mid_price_  = 0;
    tick_count_      = 0;
    prev_equity_     = strategy_.initial_capital;
    last_fv_         = {};
    metrics_         = {};
    journal_.clear();
    risk_mgr_.update_equity(strategy_.initial_capital);
}

// ─── New Trading Day ────────────────────────────────────────
void StrategyEngine::new_trading_day() noexcept {
    risk_mgr_.new_trading_day();
}

} // namespace hft
