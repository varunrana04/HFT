#include "order_manager.h"
#include "clock.h"
namespace hft {

Order OrderManager::create_order(Side side, int64_t price, int64_t quantity,
                                  OrderType type, int64_t expected_price) noexcept {
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
    // ── VWAP in floating-point to avoid int64 overflow ────────────────
    //
    // BUG FIX: The original code computed:
    //   prev_value = avg_fill_price * filled_quantity
    // Both fields are fixed-point int64 scaled by 10^8.
    // For BTC at $64,000:  avg_fill_price ≈ 6.4×10^12
    // For qty = 20 BTC:    filled_quantity ≈ 2.0×10^9
    // Product ≈ 1.28×10^22 → OVERFLOWS int64 (max ≈ 9.2×10^18).
    //
    // Fix: cast to double before multiplication. The precision loss is
    // negligible: double has 53-bit mantissa (~16 decimal digits), and
    // we only need 10 significant figures for price/qty in fixed-point.
    //
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
}

void OrderManager::cancel(Order& order) noexcept {
    if (!order.is_terminal()) order.state = OrderState::CANCELLED;
}

} // namespace hft
