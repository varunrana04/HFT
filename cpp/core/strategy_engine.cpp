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
#include <typeinfo>

namespace hft {

// ─── Constructor ────────────────────────────────────────────
StrategyEngine::StrategyEngine(const StrategyConfig& strategy_cfg,
                               const FeatureConfig& feature_cfg) noexcept
    : strategy_(strategy_cfg),
      features_(feature_cfg),
      combiner_(),
      kill_switch_(strategy_cfg.initial_capital, strategy_cfg.max_position_pct, 20.0, 5000, "kill_switch_state.txt"),
      order_mgr_(&kill_switch_) {
    journal_.reserve(4096);  // Pre-allocate for backtest
    prev_equity_ = strategy_.initial_capital;
    kill_switch_.update_state(0.0, 0.0, 0.0, 0); // initial state
}

// ─── Main Event Handler ─────────────────────────────────────
void StrategyEngine::on_trade(const Trade& trade,
                              const BookSnapshot& book) noexcept {
    ++tick_count_;

    // ── Stale Quote Protection ──
    if (pending_order_.active) {
        if (trade.timestamp_ns - pending_order_.timestamp_ns > pending_order_.max_allowed_wait_ns) {
            pending_order_.active = false;
        }
    }

    // ── Funding Rate Settlement (8-hour epoch boundaries) ──
    constexpr int64_t FUNDING_INTERVAL_NS = 28'800'000'000'000LL;
    if (last_funding_ts_ns_ == 0 && trade.timestamp_ns > 0) {
        last_funding_ts_ns_ = trade.timestamp_ns - (trade.timestamp_ns % FUNDING_INTERVAL_NS);
    }
    while (last_funding_ts_ns_ > 0 && (trade.timestamp_ns - last_funding_ts_ns_) >= FUNDING_INTERVAL_NS) {
        if (position_ != 0 && last_mid_price_ > 0) {
            // Placeholder: 0.01% flat fee on absolute notional
            double notional = std::abs(fixed_to_qty(position_)) * fixed_to_price(last_mid_price_);
            double funding_fee = notional * 0.0001; 
            // Deduct fee from realized PnL directly
            realized_pnl_ -= funding_fee;
            
            // Log funding to journal as a synthetic trade
            TradeRecord record{};
            record.timestamp_ns = last_funding_ts_ns_ + FUNDING_INTERVAL_NS;
            record.pnl = -funding_fee; // Negative PnL
            record.side = Side::NONE;
            record.quantity = 0; // pure fee
            if (mode_ == EngineMode::BACKTEST) {
                journal_.push_back(record);
            }
        }
        last_funding_ts_ns_ += FUNDING_INTERVAL_NS;
    }

    // ── Liquidity Absorption Fill Check ──
    if (pending_order_.active) {
        bool hit = false;
        // Taker SELL hits our BID
        if (trade.side == Side::ASK && pending_order_.side == Side::BID && trade.price <= pending_order_.price) {
            hit = true;
        }
        // Taker BUY hits our ASK
        else if (trade.side == Side::BID && pending_order_.side == Side::ASK && trade.price >= pending_order_.price) {
            hit = true;
        }
        
        if (hit) {
            pending_order_.queue_position -= trade.quantity;
            if (pending_order_.queue_position <= 0) {
                simulate_fill(pending_order_.side, pending_order_.price, pending_order_.qty, false);
                pending_order_.active = false;
            }
        }
    }

    // 1. Update cached mid price
    if (book.is_valid()) {
        last_mid_price_ = book.mid_price();
        last_book_      = book;          // cache for L2 sweep in simulate_fill
    }

    // 2. Compute all 6 alpha signals (always, so buffers fill during warm-up)
    last_fv_ = features_.compute_all(book, trade);

    // 3. Combine signals into single alpha score
    last_fv_.combined_alpha = combiner_.combine(last_fv_);

    // ── WARM-UP GATE: do NOT trade until buffers are populated ──────────
    // During warm-up the feature ring buffers (VPIN, vol, stat-arb) are
    // not yet full, so signals are unreliable and potentially unnormalized.
    // We continue computing features (to fill the buffers) but suppress
    // all signal generation and order routing.
    // record_return() is intentionally NOT called here — including warm-up
    // ticks in the Sharpe accumulator would pollute the variance with the
    // zero-signal period and understate the true post-warmup Sharpe.
    if (!is_warmed_up()) {
        return;
    }

    double alpha = last_fv_.combined_alpha;

    // 4. Record equity return for Sharpe calculation
    record_return();

    // 5. Determine if we should trade
    double abs_alpha = std::abs(alpha);

    // ── ACTIVE CIRCUIT BREAKER ──
    if (kill_switch_.is_trading_halted(last_fv_.timestamp_ns / 1000000)) {
        if (position_ != 0 && kill_switch_.should_flatten()) {
            // Panic liquidate all positions
            Side close_side = (position_ > 0) ? Side::ASK : Side::BID; 
            int64_t close_qty = std::abs(position_);
            // Taker hits the opposite side of the book: SELL hits BID, BUY hits ASK
            int64_t exit_price = (close_side == Side::ASK) ? book.best_bid_price : book.best_ask_price;
            
            if (exit_price != INVALID_PRICE && exit_price > 0) {
                (void)order_mgr_.create_order(
                    close_side, exit_price, close_qty, OrderType::MARKET, exit_price);
                simulate_fill(close_side, exit_price, close_qty, true); // true = is_taker
            }
        }
        return; // Block all new trading
    }

    // ── HARD STOP-LOSS & TRAILING STOP LOGIC ──
    if (position_ != 0) {
        double unrl_pnl = unrealized_pnl();
        double current_eq = equity();
        
        if (unrl_pnl > position_peak_unrealized_pnl_) {
            position_peak_unrealized_pnl_ = unrl_pnl;
        }

        // Hard Stop-Loss: If open position drags account down by > 2%
        bool hard_stop = (current_eq > 0.0 && (unrl_pnl / current_eq) < -0.02);
        
        // Volatility-Adjusted Trailing Stop
        // We use spread + realized volatility to determine a dynamic trailing buffer.
        double spread_bps = 0.0;
        if (book.best_ask_price > 0 && book.best_bid_price > 0) {
            spread_bps = static_cast<double>(book.best_ask_price - book.best_bid_price) / static_cast<double>(book.best_bid_price) * 10000.0;
        }
        // Base buffer is minimum take profit bps + 2x spread + vol scaler
        double trailing_buffer_bps = strategy_.min_take_profit_bps + (spread_bps * 2.0) + (last_fv_.realized_vol * 100.0);
        
        double notional = std::abs(fixed_to_qty(position_)) * fixed_to_price(last_mid_price_);
        double trailing_buffer_pnl = (trailing_buffer_bps / 10000.0) * notional;
        
        // Trigger trailing stop if we reached a decent profit peak, and gave back the buffer
        bool trailing_stop = false;
        if (position_peak_unrealized_pnl_ > trailing_buffer_pnl * 1.5) { // Ensure we actually had a good peak
            if (unrl_pnl < position_peak_unrealized_pnl_ - trailing_buffer_pnl) {
                trailing_stop = true;
            }
        }

        if (hard_stop || trailing_stop) {
            Side close_side = (position_ > 0) ? Side::ASK : Side::BID;
            int64_t close_qty = std::abs(position_);
            // Taker hits the opposite side of the book: SELL hits BID, BUY hits ASK
            int64_t exit_price = (close_side == Side::ASK) ? book.best_bid_price : book.best_ask_price;
            
            if (exit_price != INVALID_PRICE && exit_price > 0) {
                (void)order_mgr_.create_order(
                    close_side, exit_price, close_qty, OrderType::MARKET, exit_price);
                simulate_fill(close_side, exit_price, close_qty, true); // is_taker
            }
            return;
        }
    }

    // ── EXIT LOGIC: Close position if alpha flipped or decayed ──
    if (position_ != 0 && abs_alpha < strategy_.alpha_exit_threshold) {
        
        // Steamroller Fix: Ensure we don't exit for pennies. We must either hit the stop-loss (handled above)
        // or achieve the minimum take-profit BPS before closing due to alpha decay.
        double unrl_pnl = unrealized_pnl();
        double current_eq = equity();
        double pnl_bps = 0.0;
        if (current_eq > 0.0) {
            double notional = std::abs(fixed_to_qty(position_)) * fixed_to_price(last_mid_price_);
            if (notional > 0) pnl_bps = (unrl_pnl / notional) * 10000.0;
        }

        // If the position is profitable but hasn't reached the minimum take profit, let it run.
        if (pnl_bps >= 0.0 && pnl_bps < strategy_.min_take_profit_bps) {
            return; // Hold position
        }

        // Close the position
        Side close_side = (position_ > 0) ? Side::ASK : Side::BID;
        int64_t close_qty = std::abs(position_);

        // Build exit order (MAKER execution to capture spread)
        int64_t exit_price = (close_side == Side::ASK)
            ? book.best_ask_price   // Sell at our own ask (post limit)
            : book.best_bid_price;  // Buy at our own bid (post limit)

        if (exit_price != INVALID_PRICE && exit_price > 0) {
            Order exit_order = order_mgr_.create_order(
                close_side, exit_price, close_qty,
                OrderType::LIMIT, exit_price);

            // Risk check (handled inside create_order now)
            if (exit_order.state != OrderState::REJECTED) {
                if (!pending_order_.active || pending_order_.price != exit_price || pending_order_.side != close_side) {
                    pending_order_.active = true;
                    pending_order_.side = close_side;
                    pending_order_.price = exit_price;
                    pending_order_.qty = close_qty;
                    
                    int64_t l2_queue = 0;
                    if (close_side == Side::BID) {
                        for (int32_t i = 0; i < book.bid_count; ++i) {
                            if (book.bids[i].price >= exit_price) l2_queue += book.bids[i].quantity;
                            else break;
                        }
                    } else {
                        for (int32_t i = 0; i < book.ask_count; ++i) {
                            if (book.asks[i].price <= exit_price) l2_queue += book.asks[i].quantity;
                            else break;
                        }
                    }
                    pending_order_.queue_position = (l2_queue > 0) ? l2_queue : ((close_side == Side::BID) ? book.best_bid_qty : book.best_ask_qty);
                    
                    pending_order_.timestamp_ns = last_fv_.timestamp_ns;
                }
            }
        }
        return;
    }

    // ── ENTRY LOGIC: Open or add to position if alpha is strong ──
    // We use alpha strictly as an informational directional gate.
    
    if (last_fv_.timestamp_ns - last_trade_ns_ < strategy_.execution_cooldown_ns) {
        return; // Wait for cooldown before entering new trades
    }
    
    // Calculate dynamic threshold based on spread
    double spread_bps = 0.0;
    if (book.best_ask_price > 0 && book.best_bid_price > 0 && book.best_bid_price != INVALID_PRICE) {
        spread_bps = static_cast<double>(book.best_ask_price - book.best_bid_price) / static_cast<double>(book.best_bid_price) * 10000.0;
    }

    // ── HARD SPREAD CIRCUIT-BREAKER ──────────────────────────────────────────
    // 3D Alpha Surface analysis shows adverse-selection (red zone) clusters
    // sharply above 3.5 bps spread. Veto ALL entries unconditionally when
    // the spread exceeds this threshold — no alpha is worth the execution cost.
    if (strategy_.max_spread_bps_cutoff > 0.0 && spread_bps > strategy_.max_spread_bps_cutoff) {
        return;
    }

    double dynamic_long_threshold  = strategy_.alpha_entry_threshold + (spread_bps * strategy_.spread_alpha_multiplier);
    double dynamic_short_threshold = (strategy_.alpha_entry_threshold * strategy_.alpha_short_multiplier) + (spread_bps * strategy_.spread_alpha_multiplier);

    bool signal_buy = (alpha >= dynamic_long_threshold);
    bool signal_sell = (alpha <= -dynamic_short_threshold);

    if (signal_buy || signal_sell) {
        ++metrics_.signals_generated;

        // Direction
        Side entry_side = signal_buy ? Side::BID : Side::ASK;

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

        // Session start tracking for AS elapsed time (t)
        if (session_start_ns_ == 0 && last_fv_.timestamp_ns > 0) {
            session_start_ns_ = last_fv_.timestamp_ns;
        }

        // ── LIGHTGBM DRIVEN AVELLANEDA-STOIKOV ────────────────────
        // The LightGBM model combined the non-linear features into a single alpha [-1, 1].
        // We use this alpha to dynamically skew our reservation price.
        
        double lambda_intensity = std::tanh(last_fv_.hawkes_intensity);
        
        // Base spread (1 bps) + volatility expansion
        double base_spread_bps = 1.0 + (last_fv_.realized_vol * 10.0); 
        
        // Widen spread in high toxicity regimes
        if (static_cast<uint8_t>(last_fv_.regime) == static_cast<uint8_t>(Regime::HIGH_TOXICITY) || lambda_intensity > 0.6) {
            base_spread_bps += 2.0; 
        }

        double optimal_spread = fixed_to_price(last_mid_price_) * (base_spread_bps / 10000.0); 
        
        // Skew reservation price using LightGBM alpha. 
        // If alpha > 0, we expect price to rise, so we shift reservation up to buy higher and sell higher.
        // alpha is in [-1, 1], so we map it to a max skew of 2 bps.
        double max_skew = fixed_to_price(last_mid_price_) * (2.0 / 10000.0);
        double reservation_price = fixed_to_price(last_mid_price_) + (alpha * max_skew);
        
        int64_t lgbm_bid = price_to_fixed(reservation_price - optimal_spread / 2.0);
        int64_t lgbm_ask = price_to_fixed(reservation_price + optimal_spread / 2.0);

        // ── QUEUE-REACTIVE EXECUTION ────────────────────
        
        // True Hawkes clustering intensity [0, 1]
        // Since lambda can grow arbitrarily large during cascades, we squash it using tanh 
        // to keep it within [0, 1] for the skew penalty multiplier.
        // (lambda_intensity was computed above)
        
        int64_t exec_price = 0;
        if (entry_side == Side::BID) {
            // We want to buy. Start at the LightGBM policy bid.
            int64_t base_price = lgbm_bid;
            // Cap at Best Bid so we remain a Maker
            if (base_price > book.best_bid_price) base_price = book.best_bid_price; 
            
            // Push quote deeper into L2/L3 based on intensity
            int64_t skew_penalty = price_to_fixed(optimal_spread * lambda_intensity * 0.5);
            exec_price = base_price - skew_penalty;
        } else {
            // We want to sell. Start at the LightGBM policy ask.
            int64_t base_price = lgbm_ask;
            // Cap at Best Ask so we remain a Maker
            if (base_price < book.best_ask_price) base_price = book.best_ask_price; 
            
            // Push quote deeper into L2/L3 based on intensity
            int64_t skew_penalty = price_to_fixed(optimal_spread * lambda_intensity * 0.5);
            exec_price = base_price + skew_penalty;
        }

        if (exec_price == INVALID_PRICE || exec_price <= 0) return;

        // Market Impact Depth Check: Cap at 5% of Top-of-Book
        int64_t tob_qty = (entry_side == Side::BID) ? book.best_bid_qty : book.best_ask_qty;
        int64_t max_allowed_qty = static_cast<int64_t>(static_cast<double>(tob_qty) * 0.05);
        if (order_qty > max_allowed_qty) {
            order_qty = max_allowed_qty; // Scale down instead of outright rejection to capture edge
        }
        if (order_qty <= 0) return;

        // Build entry order
        Order entry_order = order_mgr_.create_order(
            entry_side, exec_price, order_qty,
            OrderType::LIMIT, last_mid_price_);

        // Risk gate
        if (entry_order.state != OrderState::REJECTED) {
            if (!pending_order_.active || pending_order_.price != exec_price || pending_order_.side != entry_side) {
                pending_order_.active = true;
                pending_order_.side = entry_side;
                pending_order_.price = exec_price;
                pending_order_.qty = order_qty;
                
                int64_t l2_queue = 0;
                if (entry_side == Side::BID) {
                    for (int32_t i = 0; i < book.bid_count; ++i) {
                        if (book.bids[i].price >= exec_price) l2_queue += book.bids[i].quantity;
                        else break;
                    }
                } else {
                    for (int32_t i = 0; i < book.ask_count; ++i) {
                        if (book.asks[i].price <= exec_price) l2_queue += book.asks[i].quantity;
                        else break;
                    }
                }
                pending_order_.queue_position = (l2_queue > 0) ? l2_queue : tob_qty;
                
                pending_order_.timestamp_ns = last_fv_.timestamp_ns;
            }
        } else {
            ++metrics_.risk_rejections;
        }
    }
}

// ─── Book-Only Update ───────────────────────────────────────
void StrategyEngine::on_book_update(const BookSnapshot& book) noexcept {
    if (book.is_valid()) {
        last_mid_price_ = book.mid_price();
        last_book_      = book;          // cache for L2 sweep in simulate_fill
    }

    // ── Stale Quote Protection ──
    if (pending_order_.active) {
        if (book.timestamp_ns - pending_order_.timestamp_ns > pending_order_.max_allowed_wait_ns) {
            pending_order_.active = false;
        }
    }

    // NOTE: Funding rate settlement is intentionally NOT duplicated here.
    // on_trade() is always called on the same tick as on_book_update() in
    // practice (the WebSocket combined stream delivers both together), so
    // processing funding in both handlers would double-charge the fee.
    // Settlement lives exclusively in on_trade() where the trade timestamp
    // drives the epoch boundary check.

    // Update equity tracking with new mark-to-market
    double eq = equity();
    kill_switch_.update_state(fixed_to_qty(position_), unrealized_pnl(), realized_pnl_, book.timestamp_ns / 1000000);

    // Continuous tick-by-tick drawdown tracking
    if (eq > metrics_.peak_equity) {
        metrics_.peak_equity = eq;
    }
    double dd = (metrics_.peak_equity > 0.0)
        ? (metrics_.peak_equity - eq) / metrics_.peak_equity
        : 0.0;
    if (dd > metrics_.max_drawdown) {
        metrics_.max_drawdown = dd;
    }
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

void StrategyEngine::update_kill_switch_state(int64_t timestamp_ms) noexcept {
    kill_switch_.update_state(fixed_to_qty(position_), unrealized_pnl(), realized_pnl_, timestamp_ms);
}

// ─── Compute Order Size ─────────────────────────────────────
int64_t StrategyEngine::compute_order_size(
    double alpha, double portfolio_value) const noexcept {
    if (portfolio_value <= 0.0 || last_mid_price_ == 0) return 0;

    double price = fixed_to_price(last_mid_price_);
    if (price <= 0.0) return 0;

    // ── Alpha-Scaled Fractional Kelly Sizing ─────────────────────────
    //
    // Rationale: flat 15% sizing ignores signal strength completely — a
    // weak alpha=0.06 trades the same size as a strong alpha=0.95, which
    // wastes Kelly-optimal edge and inflates position risk on weak signals.
    //
    // We use a half-Kelly approach:
    //   kelly_f   = |alpha|               (alpha ∈ [0,1] acts as an
    //                                      approximation for edge/odds)
    //   half_kelly = 0.5 * kelly_f        (half-Kelly reduces variance)
    //   position  = clamp(half_kelly, min_pct, max_position_pct)
    //
    // This naturally:
    //   - Scales size up when alpha is strong (high-confidence signal)
    //   - Scales size DOWN when alpha is weak (near-threshold marginal signal)
    //   - Hard-caps at max_position_pct (default 15%) enforced by RiskManager
    //   - Enforces a minimum of 0.5% so the smallest quote is still meaningful
    //
    // With a $10M portfolio and BTC at $77k:
    //   alpha=0.10 → 5.0% → $500k → 6.5 BTC (manageable, maker-friendly)
    //   alpha=0.50 → 15%  → $1.5M → 19.5 BTC (max, only on very strong signal)
    //   alpha=0.05 → 2.5% → $250k → 3.2 BTC  (small, low conviction)
    //
    static constexpr double MIN_SIZE_PCT = 0.005;  // 0.5% floor
    double abs_alpha  = std::abs(alpha);
    double half_kelly = 0.5 * abs_alpha;
    double size_pct   = std::clamp(half_kelly,
                                   MIN_SIZE_PCT,
                                   strategy_.max_position_pct);

    double notional = portfolio_value * size_pct;
    double qty      = notional / price;

    return qty_to_fixed(qty);
}

// ─── Simulate Fill ──────────────────────────────────────────
//
// Execution Model
// ───────────────
// MAKER fills (is_taker=false):
//   Posted at best_bid or best_ask. Filled at the posted limit price.
//   Fee = maker_fee_pct = -0.00005 (-0.5 bps rebate). No market impact.
//
// TAKER fills (is_taker=true, used for stop-loss / circuit-breaker exits):
//   True L2 book sweep: consume liquidity level-by-level from the cached
//   last_book_ snapshot until the full quantity is filled or book is
//   exhausted. Compute a quantity-weighted average fill price (VWAP)
//   across all consumed levels. Any residual quantity beyond book depth
//   is filled at the last available level price (conservative assumption).
//   Fee = taker_fee_pct = +0.00015 (1.5 bps cost).
//
void StrategyEngine::simulate_fill(Side side, int64_t price,
                                   int64_t quantity, bool is_taker) noexcept {

    // ── Determine effective fill price ───────────────────────────────
    int64_t effective_price = price;
    double  total_impact    = 0.0;

    if (is_taker && last_mid_price_ > 0) {
        // ── True L2 Book Sweep ────────────────────────────────────────
        // Walk the relevant side of last_book_ level-by-level.
        // BUY  taker → consume ASK levels (ascending price)
        // SELL taker → consume BID levels (descending price)
        //
        // Each level: consume min(level_qty, remaining_qty) at level_price.
        // Accumulate notional and quantity for VWAP calculation.
        // If book is exhausted before full fill, use the last level price
        // for the remainder (represents crossing into next tick / wider
        // spread in a thin book — conservative and realistic).

        int64_t remaining     = quantity;
        double  total_notional = 0.0;
        double  total_filled_d = 0.0;
        int64_t last_level_price = price; // fallback if book has no levels

        const PriceLevel* levels     = nullptr;
        int32_t           level_count = 0;

        if (side == Side::BID) {
            // Buying aggressively → sweep asks (ascending price)
            levels      = last_book_.asks;
            level_count = last_book_.ask_count;
        } else {
            // Selling aggressively → sweep bids (descending price)
            levels      = last_book_.bids;
            level_count = last_book_.bid_count;
        }

        if (level_count > 0 && levels != nullptr) {
            for (int32_t i = 0; i < level_count && remaining > 0; ++i) {
                if (!levels[i].is_valid()) break;

                last_level_price      = levels[i].price;
                int64_t available     = levels[i].quantity;
                int64_t fill_at_level = std::min(remaining, available);

                total_notional  += static_cast<double>(last_level_price)
                                  * static_cast<double>(fill_at_level);
                total_filled_d  += static_cast<double>(fill_at_level);
                remaining       -= fill_at_level;
            }

            // Residual beyond book depth: fill at last level price
            if (remaining > 0) {
                total_notional += static_cast<double>(last_level_price)
                                 * static_cast<double>(remaining);
                total_filled_d += static_cast<double>(remaining);
            }

            if (total_filled_d > 0.0) {
                effective_price = static_cast<int64_t>(total_notional / total_filled_d);
            }
        }

        // Market impact = VWAP deviation from the top-of-book price
        double mid_d   = fixed_to_price(last_mid_price_);
        double vwap_d  = fixed_to_price(effective_price);
        double qty_d   = fixed_to_qty(quantity);
        if (side == Side::BID) {
            total_impact = std::max(0.0, (vwap_d - mid_d) * qty_d);
        } else {
            total_impact = std::max(0.0, (mid_d - vwap_d) * qty_d);
        }
    }

    double fill_price = fixed_to_price(effective_price);
    double fill_qty   = fixed_to_qty(quantity);
    double trade_pnl  = 0.0;
    double avg_entry_price_prev = avg_entry_price_;

    // ── Taker vs Maker fees ──────────────────────────────────────────
    double fee_pct = is_taker ? strategy_.taker_fee_pct : strategy_.maker_fee_pct;
    double fee     = fill_price * fill_qty * fee_pct;
    realized_pnl_ -= fee;

    // ── Slippage vs mid price ────────────────────────────────────────
    // Positive slippage = paid more than mid (bad for buys, bad for sells)
    double slippage = 0.0;
    if (last_mid_price_ > 0) {
        double mid = fixed_to_price(last_mid_price_);
        slippage = (side == Side::BID)
            ? (fill_price - mid)   // Positive = overpaid vs mid
            : (mid - fill_price);  // Positive = undersold vs mid
    }
    // Add market impact cost to slippage so metrics accurately reflect it
    if (is_taker && fill_qty > 0.0) {
        slippage += total_impact / fill_qty;
    }

    if (side == Side::BID) {
        // ── BUYING ──
        if (position_ < 0) {
            // Closing/reducing short position → realize PnL
            int64_t close_qty   = std::min(quantity, -position_);
            double  close_qty_d = fixed_to_qty(close_qty);
            trade_pnl = (avg_entry_price_ - fill_price) * close_qty_d;
            realized_pnl_ += trade_pnl;
            position_ += close_qty;

            int64_t remaining = quantity - close_qty;
            if (remaining > 0) {
                avg_entry_price_ = fill_price;
                position_ += remaining;
                position_peak_unrealized_pnl_ = 0.0;
            } else if (position_ == 0) {
                avg_entry_price_ = 0.0;
                position_peak_unrealized_pnl_ = 0.0;
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
            int64_t close_qty   = std::min(quantity, position_);
            double  close_qty_d = fixed_to_qty(close_qty);
            trade_pnl = (fill_price - avg_entry_price_) * close_qty_d;
            realized_pnl_ += trade_pnl;
            position_ -= close_qty;

            int64_t remaining = quantity - close_qty;
            if (remaining > 0) {
                avg_entry_price_ = fill_price;
                position_ -= remaining;
                position_peak_unrealized_pnl_ = 0.0;
            } else if (position_ == 0) {
                avg_entry_price_ = 0.0;
                position_peak_unrealized_pnl_ = 0.0;
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

    // ── Journal Entry ────────────────────────────────────────────────
    if (mode_ == EngineMode::BACKTEST) {
        TradeRecord record{};
        record.timestamp_ns = last_fv_.timestamp_ns;

        if (trade_pnl != 0.0) {
            record.entry_price = price_to_fixed(avg_entry_price_prev);
            record.exit_price  = effective_price;
        } else {
            record.entry_price = effective_price;
            record.exit_price  = 0;
        }

        record.quantity  = (side == Side::BID) ? quantity : -quantity;
        record.pnl       = trade_pnl;
        record.slippage  = slippage;
        record.side      = side;
        journal_.push_back(record);
    }

    last_trade_ns_ = last_fv_.timestamp_ns;
    update_metrics(trade_pnl, slippage);
    kill_switch_.update_state(fixed_to_qty(position_), unrealized_pnl(), realized_pnl_, last_fv_.timestamp_ns / 1000000);
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

    // Drawdown tracking is now handled in on_book_update for tick-level precision

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
    
    if (mode_ == EngineMode::BACKTEST) {
        equity_history_.push_back(eq);
    }
}

// ─── Reset ──────────────────────────────────────────────────
void StrategyEngine::reset() noexcept {
    features_.reset();
    kill_switch_.manual_reset("Engine reset", now_ns() / 1000000);
    position_        = 0;
    realized_pnl_    = 0.0;
    avg_entry_price_ = 0.0;
    last_mid_price_  = 0;
    tick_count_      = 0;
    prev_equity_     = strategy_.initial_capital;
    session_start_ns_ = 0;
    position_peak_unrealized_pnl_ = 0.0;
    last_funding_ts_ns_ = 0;
    last_fv_         = {};
    last_book_       = {};
    metrics_         = {};
    journal_.clear();
    equity_history_.clear();
    kill_switch_.update_state(0.0, 0.0, 0.0, now_ns() / 1000000);
}

// ─── New Trading Day ────────────────────────────────────────
void StrategyEngine::new_trading_day() noexcept {
    // kill_switch handles daily reset based on timestamp automatically
    session_start_ns_ = 0; // Reset session start for new day
}

} // namespace hft
