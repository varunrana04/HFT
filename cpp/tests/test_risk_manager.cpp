/**
 * @file test_risk_manager.cpp
 * @brief Unit tests for the RiskManager (all 5 risk checks).
 *
 * IMPORTANT: All price/qty values are chosen so that
 *   order_notional = price * qty   fits within max_position_pct * portfolio.
 *   Price is in USD-per-coin (e.g. 100.0 for a $100 coin).
 *   Qty is in coins.
 *   So: order_notional = 100.0 * 1.0 = $100, which is 0.1% of a $100k portfolio.
 */

#include "risk_manager.h"
#include "order_manager.h"
#include "clock.h"
#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <thread>
#include <chrono>

// ─── Test harness (shared with test_main.cpp via linker) ─────
struct TestRegistrar {
    TestRegistrar(const char* name, std::function<bool()> func);
};
struct TestCase2R { std::string name; std::function<bool()> func; };
extern std::vector<TestCase2R>& get_tests();

#define ASSERT_TRUE(expr) do { if (!(expr)) { std::cerr << "  FAIL: " << #expr << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { std::cerr << "  FAIL: " << #a << " == " << #b << " (" << static_cast<int>(a) << " != " << static_cast<int>(b) << ") [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_NE(a, b) do { if ((a) == (b)) { std::cerr << "  FAIL: " << #a << " != " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_GT(a, b) do { if (!((a) > (b))) { std::cerr << "  FAIL: " << #a << " > " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_LT(a, b) do { if (!((a) < (b))) { std::cerr << "  FAIL: " << #a << " < " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)

#define TEST(suite, name)                                         \
    [[maybe_unused]] static bool test_##suite##_##name();         \
    [[maybe_unused]] static TestRegistrar reg_##suite##_##name(   \
        #suite "::" #name, test_##suite##_##name);                \
    static bool test_##suite##_##name()

// ─── Helper: create a test order ─────────────────────────────
// price: USD per coin (fixed-point via price_to_fixed)
// qty:   number of coins (fixed-point via qty_to_fixed)

static hft::Order make_order(hft::Side side, double price, double qty) {
    hft::Order o{};
    o.timestamp_ns   = hft::now_ns();
    o.price          = hft::price_to_fixed(price);
    o.quantity       = hft::qty_to_fixed(qty);
    o.filled_quantity = 0;
    o.avg_fill_price = 0;
    o.expected_price = o.price;
    o.order_id       = 1;
    o.instrument_id  = 0;
    o.side           = side;
    o.type           = hft::OrderType::LIMIT;
    o.state          = hft::OrderState::NEW;
    return o;
}

// ─── Test: Position Limit Pass ───────────────────────────────
// Portfolio = $100k, max_position_pct = 25%  → max notional = $25k
// Order: 1 coin × $100 = $100 notional → PASS

TEST(Risk, PositionLimitPass) {
    hft::RiskConfig config;
    config.max_position_pct = 0.25;
    hft::RiskManager rm(config);
    rm.update_equity(100000.0);

    // 1 coin at $100 = $100 notional, well within $25k limit
    auto order = make_order(hft::Side::BID, 100.0, 1.0);
    auto verdict = rm.check_order(order, 0, 0.0, 100000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::PASS);

    return true;
}

// ─── Test: Position Limit Reject ─────────────────────────────
// Portfolio = $100k, max_position_pct = 0.01% → max notional = $10
// Order: 1 coin × $100 = $100 → REJECT

TEST(Risk, PositionLimitReject) {
    hft::RiskConfig config;
    config.max_position_pct = 0.0001;   // 0.01% of portfolio
    config.circuit_breaker_cooldown_ns = 1;
    hft::RiskManager rm(config);
    rm.update_equity(100000.0);

    // 1 coin × $100 = $100, max allowed = $100k × 0.01% = $10 → REJECT
    auto order = make_order(hft::Side::BID, 100.0, 1.0);
    auto verdict = rm.check_order(order, 0, 0.0, 100000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::POSITION_LIMIT);
    ASSERT_EQ(rm.stats().rejected_position, 1u);

    return true;
}

// ─── Test: Drawdown Gate ─────────────────────────────────────
// Peak = $100k, current = $94k → 6% drawdown > 5% limit → REJECT

TEST(Risk, DrawdownGate) {
    hft::RiskConfig config;
    config.max_drawdown_pct = 0.05;       // 5% drawdown limit
    config.max_position_pct = 1.0;        // allow large positions
    config.max_single_order_pct = 1.0;    // allow large orders
    config.circuit_breaker_cooldown_ns = 1;
    hft::RiskManager rm(config);

    rm.update_equity(100000.0);   // sets peak = 100k
    rm.update_equity(94000.0);    // current = 94k, drawdown = 6%

    // Pass current_equity as portfolio_value so check_order uses the depressed value
    // current_pnl = 0 so update_equity inside check_order will call update_equity(94000),
    // which does NOT raise the peak (94k < 100k peak), so drawdown stays at 6%.
    auto order = make_order(hft::Side::BID, 100.0, 1.0);
    auto verdict = rm.check_order(order, 0, 0.0, 94000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::DRAWDOWN_LIMIT);
    ASSERT_EQ(rm.stats().rejected_drawdown, 1u);

    return true;
}

// ─── Test: Drawdown Below Threshold Passes ───────────────────
// Peak = $100k, current = $97k → 3% drawdown < 5% limit → PASS

TEST(Risk, DrawdownPass) {
    hft::RiskConfig config;
    config.max_drawdown_pct = 0.05;
    config.max_position_pct = 1.0;
    config.max_single_order_pct = 1.0;
    hft::RiskManager rm(config);

    rm.update_equity(100000.0);
    rm.update_equity(97000.0);  // 3% drawdown < 5%

    auto order = make_order(hft::Side::BID, 100.0, 1.0);
    auto verdict = rm.check_order(order, 0, 0.0, 97000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::PASS);

    return true;
}

// ─── Test: Daily Loss Limit ──────────────────────────────────
// Day starts at $100k, drops to $96k → 4% daily loss > 3% limit → REJECT

TEST(Risk, DailyLossLimit) {
    hft::RiskConfig config;
    config.max_daily_loss_pct = 0.03;    // 3%
    config.max_drawdown_pct   = 1.0;     // disable drawdown (won't trigger first)
    config.max_position_pct   = 1.0;
    config.max_single_order_pct = 1.0;
    config.circuit_breaker_cooldown_ns = 1;
    hft::RiskManager rm(config);

    rm.update_equity(100000.0);
    rm.new_trading_day();         // day_start_equity = 100k

    rm.update_equity(96000.0);    // 4% daily loss

    auto order = make_order(hft::Side::BID, 100.0, 1.0);
    // Pass 96k as portfolio_value; update_equity inside will call update_equity(96k+0)=96k
    // day_start = 100k, daily_loss = (100k-96k)/100k = 4% > 3% → DAILY_LOSS_LIMIT
    auto verdict = rm.check_order(order, 0, 0.0, 96000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::DAILY_LOSS_LIMIT);

    return true;
}

// ─── Test: Single Order Size Limit ───────────────────────────
// max_single_order_pct = 2%, portfolio = $100k → max order = $2k
// Order: 1 coin × $5000 = $5000 (5%) → REJECT

TEST(Risk, OrderSizeLimit) {
    hft::RiskConfig config;
    config.max_single_order_pct = 0.02;  // 2% max per order
    config.max_drawdown_pct     = 1.0;   // disable drawdown
    config.max_daily_loss_pct   = 1.0;   // disable daily loss
    config.max_position_pct     = 1.0;   // allow large position
    hft::RiskManager rm(config);

    rm.update_equity(100000.0);

    // 1 coin × $5000 = $5000 = 5% of portfolio → exceeds 2% limit
    auto order = make_order(hft::Side::BID, 5000.0, 1.0);
    auto verdict = rm.check_order(order, 0, 0.0, 100000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::ORDER_SIZE_LIMIT);

    return true;
}

// ─── Test: Small Order Passes Size Limit ─────────────────────
// max_single_order_pct = 2%, portfolio = $100k → max order = $2k
// Order: 0.01 coin × $100 = $1 (0.001%) → PASS

TEST(Risk, OrderSizePass) {
    hft::RiskConfig config;
    config.max_single_order_pct = 0.02;
    config.max_drawdown_pct     = 1.0;
    config.max_daily_loss_pct   = 1.0;
    config.max_position_pct     = 1.0;
    hft::RiskManager rm(config);

    rm.update_equity(100000.0);

    auto order = make_order(hft::Side::BID, 100.0, 0.01);
    auto verdict = rm.check_order(order, 0, 0.0, 100000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::PASS);

    return true;
}

// ─── Test: Circuit Breaker Cooldown ──────────────────────────

TEST(Risk, CircuitBreakerCooldown) {
    hft::RiskConfig config;
    config.max_position_pct = 0.0001;   // tiny limit so we breach easily
    config.circuit_breaker_cooldown_ns = 100'000'000LL;  // 100ms
    hft::RiskManager rm(config);
    rm.update_equity(100000.0);

    // Trigger breach → circuit breaker activates
    auto bad_order = make_order(hft::Side::BID, 100.0, 1.0);
    auto verdict = rm.check_order(bad_order, 0, 0.0, 100000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::POSITION_LIMIT);
    ASSERT_TRUE(rm.is_circuit_breaker_active());

    // A valid (small) order should still be rejected during cooldown
    auto good_order = make_order(hft::Side::BID, 1.0, 0.001);
    verdict = rm.check_order(good_order, 0, 0.0, 100000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::CIRCUIT_BREAKER);

    return true;
}

// ─── Test: Legacy bool interface ─────────────────────────────

TEST(Risk, LegacyBoolInterface) {
    hft::RiskConfig config;
    config.max_position_pct = 1.0;
    config.max_drawdown_pct = 1.0;
    config.max_daily_loss_pct = 1.0;
    config.max_single_order_pct = 1.0;
    hft::RiskManager rm(config);
    rm.update_equity(100000.0);

    // Small order, all limits disabled → true
    auto order = make_order(hft::Side::BID, 100.0, 0.01);
    bool ok = rm.check_order(order, 0, 0.0);
    ASSERT_TRUE(ok);

    return true;
}

// ─── Test: Reset clears all state ────────────────────────────

TEST(Risk, Reset) {
    hft::RiskConfig config;
    config.max_position_pct = 0.0001;
    config.circuit_breaker_cooldown_ns = 100'000'000'000LL; // 100s
    hft::RiskManager rm(config);
    rm.update_equity(100000.0);

    // Trip circuit breaker
    auto bad_order = make_order(hft::Side::BID, 100.0, 1.0);
    rm.check_order(bad_order, 0, 0.0, 100000.0);
    ASSERT_TRUE(rm.is_circuit_breaker_active());

    rm.reset();
    ASSERT_FALSE(rm.is_circuit_breaker_active());
    ASSERT_EQ(rm.stats().orders_checked, 0u);

    return true;
}

// ─── Test: Statistics Tracking ───────────────────────────────

TEST(Risk, StatsTracking) {
    hft::RiskConfig config;
    config.max_position_pct    = 1.0;
    config.max_drawdown_pct    = 1.0;
    config.max_daily_loss_pct  = 1.0;
    config.max_single_order_pct = 1.0;
    config.circuit_breaker_cooldown_ns = 1;
    hft::RiskManager rm(config);
    rm.update_equity(100000.0);

    for (int i = 0; i < 3; ++i) {
        auto order = make_order(hft::Side::BID, 100.0, 0.01);
        rm.check_order(order, 0, 0.0, 100000.0);
    }

    ASSERT_EQ(rm.stats().orders_checked, 3u);
    ASSERT_EQ(rm.stats().orders_passed, 3u);
    ASSERT_TRUE(std::abs(rm.stats().pass_rate() - 1.0) < 0.001);

    return true;
}

// ─── Test: New Trading Day resets daily loss ──────────────────

TEST(Risk, NewTradingDay) {
    hft::RiskConfig config;
    config.max_daily_loss_pct  = 0.03;
    config.max_drawdown_pct    = 1.0;
    config.circuit_breaker_cooldown_ns = 1;
    hft::RiskManager rm(config);

    // Day 1: start at 100k, lose to 96k → 4% daily loss
    rm.update_equity(100000.0);
    rm.new_trading_day();
    rm.update_equity(96000.0);
    ASSERT_GT(rm.current_daily_loss(), 0.03);

    // New day: daily tracking resets from current equity
    rm.new_trading_day();
    ASSERT_TRUE(rm.current_daily_loss() < 0.001);

    return true;
}
