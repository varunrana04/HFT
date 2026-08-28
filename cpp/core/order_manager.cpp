#include "order_manager.h"
#include "clock.h"
namespace hft {

Order OrderManager::create_order(Side side, int64_t price, int64_t quantity,
                                  OrderType type, int64_t expected_price) noexcept {
    if (kill_switch_) {
        int64_t current_ts_ms = now_ns() / 1000000;
        double req_qty = fixed_to_qty(quantity);
        if (side == Side::ASK) req_qty = -req_qty;
        
        if (!kill_switch_->check_order_allowed(req_qty, current_ts_ms)) {
            // Reject order
            Order o{};
            o.state = OrderState::REJECTED;
            return o;
        }
    }

    Order o{};
    o.timestamp_ns    = now_ns();
    o.price           = price;
    o.quantity        = quantity;
    o.filled_quantity = 0;
    o.avg_fill_price  = 0;
    o.expected_price  = expected_price;
    o.order_id        = next_id_++;
    o.instrument_id   = 0;
    o.side            = side;
    o.type            = type;
    o.state           = OrderState::NEW;
    return o;
}

void OrderManager::on_fill(Order& order, int64_t fill_price,
                            int64_t fill_qty) noexcept {
    double prev_notional  = static_cast<double>(order.avg_fill_price)
                          * static_cast<double>(order.filled_quantity);
    double new_notional   = static_cast<double>(fill_price)
                          * static_cast<double>(fill_qty);

    order.filled_quantity += fill_qty;

    if (order.filled_quantity > 0) {
        double new_avg = (prev_notional + new_notional)
                       / static_cast<double>(order.filled_quantity);
        order.avg_fill_price = static_cast<int64_t>(new_avg);
    }

    order.state = (order.filled_quantity >= order.quantity)
                  ? OrderState::FILLED : OrderState::PARTIAL;

    // 2. Update global inventory and realized PnL
    int64_t signed_fill_qty = (order.side == Side::BID) ? fill_qty : -fill_qty;
    
    // Check if we are flat, or increasing current position
    if (current_inventory_ == 0) {
        current_inventory_ = signed_fill_qty;
        avg_entry_price_ = fill_price;
    } else if ((current_inventory_ > 0 && signed_fill_qty > 0) || 
               (current_inventory_ < 0 && signed_fill_qty < 0)) {
        // Increasing position size -> VWAP the global average entry price
        double total_qty = static_cast<double>(std::abs(current_inventory_)) + static_cast<double>(fill_qty);
        double prev_inv_notional = static_cast<double>(avg_entry_price_) * static_cast<double>(std::abs(current_inventory_));
        double new_inv_notional = static_cast<double>(fill_price) * static_cast<double>(fill_qty);
        avg_entry_price_ = static_cast<int64_t>((prev_inv_notional + new_inv_notional) / total_qty);
        current_inventory_ += signed_fill_qty;
    } else {
        // Decreasing position size -> Realizing PnL
        int64_t abs_inv = std::abs(current_inventory_);
        if (fill_qty <= abs_inv) {
            // Partial or full close
            double pnl_per_unit = (current_inventory_ > 0) 
                                  ? static_cast<double>(fill_price - avg_entry_price_) 
                                  : static_cast<double>(avg_entry_price_ - fill_price);
            realized_pnl_ += static_cast<int64_t>((pnl_per_unit * static_cast<double>(fill_qty)) / static_cast<double>(QTY_SCALE));
            
            current_inventory_ += signed_fill_qty;
            if (current_inventory_ == 0) {
                avg_entry_price_ = 0;
            }
        } else {
            // Overshoot: Close existing, open new
            int64_t close_qty = abs_inv;
            int64_t open_qty = fill_qty - abs_inv;
            
            double pnl_per_unit = (current_inventory_ > 0) 
                                  ? static_cast<double>(fill_price - avg_entry_price_) 
                                  : static_cast<double>(avg_entry_price_ - fill_price);
            realized_pnl_ += static_cast<int64_t>((pnl_per_unit * static_cast<double>(close_qty)) / static_cast<double>(QTY_SCALE));
            
            current_inventory_ = (order.side == Side::BID) ? open_qty : -open_qty;
            avg_entry_price_ = fill_price;
        }
    }
}

void OrderManager::cancel(Order& order) noexcept {
    if (!order.is_terminal()) order.state = OrderState::CANCELLED;
}

bool OrderManager::check_inventory_limits(Side side, int64_t quantity) const noexcept {
    if (current_inventory_ == 0) return quantity <= max_inventory_;
    
    if (current_inventory_ > 0) {
        if (side == Side::BID) {
            return (current_inventory_ + quantity) <= max_inventory_;
        }
        return true; 
    } else {
        if (side == Side::ASK) {
            return (std::abs(current_inventory_) + quantity) <= max_inventory_;
        }
        return true;
    }
}

bool OrderManager::evaluate_exits(int64_t current_mid_price, Order& exit_order_out) noexcept {
    if (current_inventory_ == 0 || avg_entry_price_ == 0 || current_mid_price <= 0) return false;

    bool triggered = false;
    Side exit_side = Side::NONE;

    if (current_inventory_ > 0) { // LONG
        if (current_mid_price >= avg_entry_price_ + take_profit_ticks_) {
            triggered = true;
        } else if (current_mid_price <= avg_entry_price_ - stop_loss_ticks_) {
            triggered = true;
        }
        exit_side = Side::ASK;
    } else { // SHORT
        if (current_mid_price <= avg_entry_price_ - take_profit_ticks_) {
            triggered = true;
        } else if (current_mid_price >= avg_entry_price_ + stop_loss_ticks_) {
            triggered = true;
        }
        exit_side = Side::BID;
    }

    if (triggered) {
        exit_order_out = create_order(exit_side, 0, std::abs(current_inventory_), OrderType::MARKET, current_mid_price);
        return true;
    }
    return false;
}

bool OrderManager::is_trading_halted(int64_t current_timestamp_ms) noexcept {
    if (kill_switch_) {
        return kill_switch_->is_trading_halted(current_timestamp_ms);
    }
    return false;
}

bool OrderManager::should_flatten() const noexcept {
    if (kill_switch_) {
        return kill_switch_->should_flatten();
    }
    return false;
}

void OrderManager::update_kill_switch(int64_t current_timestamp_ms) noexcept {
    // Left empty here - it is expected that StrategyEngine updates KillSwitch
    // directly, as it has access to open_pnl, realized_pnl, and equity.
}

} // namespace hft
