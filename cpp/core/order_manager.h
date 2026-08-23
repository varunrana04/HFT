#pragma once
#include "types.h"
#include <cstdint>
namespace hft {
class OrderManager {
public:
    OrderManager() noexcept = default;
    Order create_order(Side side, int64_t price, int64_t quantity,
                       OrderType type, int64_t expected_price) noexcept;
    void on_fill(Order& order, int64_t fill_price, int64_t fill_qty) noexcept;
    void cancel(Order& order) noexcept;
private:
    uint64_t next_id_ = 1;
};
} // namespace hft
