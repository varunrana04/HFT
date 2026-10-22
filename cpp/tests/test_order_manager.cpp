#include <gtest/gtest.h>
#include "order_manager.h"
#include "types.h"

using namespace hft;

class OrderManagerTest : public ::testing::Test {
protected:
    OrderManager om;
};

TEST_F(OrderManagerTest, TestInventoryTracking) {
    EXPECT_EQ(om.current_inventory(), 0);
    EXPECT_EQ(om.realized_pnl(), 0);

    // Enter Long 0.5 at 100
    Order o1 = om.create_order(Side::BID, price_to_fixed(100.0), qty_to_fixed(0.5), OrderType::LIMIT, price_to_fixed(100.0));
    om.on_fill(o1, price_to_fixed(100.0), qty_to_fixed(0.5));

    EXPECT_EQ(om.current_inventory(), qty_to_fixed(0.5));

    // Sell 0.2 at 110 (Partial Close)
    Order o2 = om.create_order(Side::ASK, price_to_fixed(110.0), qty_to_fixed(0.2), OrderType::LIMIT, price_to_fixed(110.0));
    om.on_fill(o2, price_to_fixed(110.0), qty_to_fixed(0.2));

    EXPECT_EQ(om.current_inventory(), qty_to_fixed(0.3));
    // Realized PnL: 0.2 * (110 - 100) = 2.0
    EXPECT_EQ(om.realized_pnl(), price_to_fixed(2.0));

    // Sell 0.5 at 110 (Overshoot: close 0.3, open short 0.2)
    Order o3 = om.create_order(Side::ASK, price_to_fixed(110.0), qty_to_fixed(0.5), OrderType::LIMIT, price_to_fixed(110.0));
    om.on_fill(o3, price_to_fixed(110.0), qty_to_fixed(0.5));

    EXPECT_EQ(om.current_inventory(), -qty_to_fixed(0.2));
    // Realized PnL: previous 2.0 + (0.3 * (110 - 100)) = 2.0 + 3.0 = 5.0
    EXPECT_EQ(om.realized_pnl(), price_to_fixed(5.0));
}

TEST_F(OrderManagerTest, TestTakeProfit) {
    Order o1 = om.create_order(Side::BID, price_to_fixed(100.0), qty_to_fixed(0.5), OrderType::LIMIT, price_to_fixed(100.0));
    om.on_fill(o1, price_to_fixed(100.0), qty_to_fixed(0.5));

    Order exit_order;
    // Mid price 114 -> no exit (TP is 15 ticks)
    EXPECT_FALSE(om.evaluate_exits(price_to_fixed(114.0), exit_order));
    
    // Mid price 115 -> Trigger TP
    EXPECT_TRUE(om.evaluate_exits(price_to_fixed(115.0), exit_order));
    EXPECT_EQ(exit_order.side, Side::ASK);
    EXPECT_EQ(exit_order.quantity, qty_to_fixed(0.5));
    EXPECT_EQ(exit_order.type, OrderType::MARKET);
}

TEST_F(OrderManagerTest, TestStopLoss) {
    Order o1 = om.create_order(Side::ASK, price_to_fixed(100.0), qty_to_fixed(0.5), OrderType::LIMIT, price_to_fixed(100.0));
    om.on_fill(o1, price_to_fixed(100.0), qty_to_fixed(0.5));

    Order exit_order;
    // Short stop loss is if price rises. Default SL is 5 ticks.
    EXPECT_FALSE(om.evaluate_exits(price_to_fixed(104.0), exit_order));
    EXPECT_TRUE(om.evaluate_exits(price_to_fixed(105.0), exit_order));
    EXPECT_EQ(exit_order.side, Side::BID);
    EXPECT_EQ(exit_order.quantity, qty_to_fixed(0.5));
}
