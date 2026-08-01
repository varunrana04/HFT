#include "order_manager.h"
#include "clock.h"
namespace hft {
Order OrderManager::create_order(Side side, int64_t price, int64_t quantity,
                                  OrderType type, int64_t expected_price) noexcept {
    Order o{};
    o.timestamp_ns    = now_ns();
    o.price           = price;
    o.quantity         = quantity;
    o.filled_quantity  = 0;
    o.avg_fill_price   = 0;
    o.expected_price   = expected_price;
    o.order_id         = next_id_++;
    o.instrument_id    = 0;
    o.side             = side;
    o.type             = type;
    o.state            = OrderState::NEW;
    return o;
}
void OrderManager::on_fill(Order& order, int64_t fill_price,
                            int64_t fill_qty) noexcept {
    int64_t prev_value = order.avg_fill_price * order.filled_quantity;
    order.filled_quantity += fill_qty;
    if (order.filled_quantity > 0) {
        order.avg_fill_price = (prev_value + fill_price * fill_qty)
                               / order.filled_quantity;
    }
    order.state = (order.filled_quantity >= order.quantity)
                  ? OrderState::FILLED : OrderState::PARTIAL;
}
void OrderManager::cancel(Order& order) noexcept {
    if (!order.is_terminal()) order.state = OrderState::CANCELLED;
}
} // namespace hft
