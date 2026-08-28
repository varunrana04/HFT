#pragma once
#include "types.h"
#include "kill_switch.h"
#include <cstdint>
namespace hft {
class OrderManager {
public:
    OrderManager(core::KillSwitch* kill_switch = nullptr) noexcept : kill_switch_(kill_switch) {}
    
    Order create_order(Side side, int64_t price, int64_t quantity,
                       OrderType type, int64_t expected_price) noexcept;
    void on_fill(Order& order, int64_t fill_price, int64_t fill_qty) noexcept;
    void cancel(Order& order) noexcept;

    // --- Risk & Inventory ---
    bool check_inventory_limits(Side side, int64_t quantity) const noexcept;
    
    // Evaluates TP/SL. Returns true if an exit order was generated.
    bool evaluate_exits(int64_t current_mid_price, Order& exit_order_out) noexcept;
    
    // KillSwitch Checks
    bool is_trading_halted(int64_t current_timestamp_ms) noexcept;
    bool should_flatten() const noexcept;
    void update_kill_switch(int64_t current_timestamp_ms) noexcept;

    // Accessors
    int64_t current_inventory() const noexcept { return current_inventory_; }
    int64_t realized_pnl() const noexcept { return realized_pnl_; }
private:
    core::KillSwitch* kill_switch_;
    uint64_t next_id_ = 1;

    // --- State ---
    int64_t current_inventory_ = 0;  // Positive = Long, Negative = Short (fixed-point)
    int64_t avg_entry_price_ = 0;    // VWAP of current inventory (fixed-point)
    int64_t realized_pnl_ = 0;       // Cumulative realized PnL (fixed-point)

    // --- Hardcoded Risk Parameters (can be made configurable later) ---
    // 1 tick = 1 unit of price precision. Let's assume BTC/USDT where spread is typically 0.1 or 1.
    // Actually, price_to_fixed(15.0) is safe.
    int64_t take_profit_ticks_ = 1500000000; // 15.0 * 10^8
    int64_t stop_loss_ticks_   = 500000000;  // 5.0 * 10^8
    int64_t max_inventory_     = 10000000;   // 0.1 * 10^8
};
} // namespace hft

