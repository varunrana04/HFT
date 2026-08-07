/**
 * @file test_risk_manager.cpp
 * @brief Unit tests for the RiskManager (all 5 risk checks).
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

TEST(Risk, PositionLimitPass) {
    hft::RiskConfig config;
    config.max_position = hft::qty_to_fixed(100.0);
    hft::RiskManager rm(config);
    rm.update_equity(100000.0);

    // Buy 50 units — should pass (50 < 100)
    auto order = make_order(hft::Side::BID, 50000.0, 50.0);
    auto verdict = rm.check_order(order, 0, 0.0, 100000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::PASS);

    return true;
}

// ─── Test: Position Limit Reject ─────────────────────────────

TEST(Risk, PositionLimitReject) {
    hft::RiskConfig config;
    config.max_position = hft::qty_to_fixed(100.0);
    // Set very short cooldown so circuit breaker clears quickly
    config.circuit_breaker_cooldown_ns = 1;
    hft::RiskManager rm(config);
    rm.update_equity(100000.0);

    // Already hold 90, trying to buy 20 more → 110 > 100
    auto order = make_order(hft::Side::BID, 50000.0, 20.0);
    int64_t current_pos = hft::qty_to_fixed(90.0);
    auto verdict = rm.check_order(order, current_pos, 0.0, 100000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::POSITION_LIMIT);

    // Stats should reflect the rejection
    ASSERT_EQ(rm.stats().rejected_position, 1u);

    return true;
}

// ─── Test: Drawdown Gate ─────────────────────────────────────

TEST(Risk, DrawdownGate) {
    hft::RiskConfig config;
    config.max_drawdown_pct = 0.05; // 5% max drawdown
    config.circuit_breaker_cooldown_ns = 1;
    hft::RiskManager rm(config);

    // Set peak equity at 100k
    rm.update_equity(100000.0);

    // Equity drops to 94k → 6% drawdown > 5% limit
    rm.update_equity(94000.0);

    auto order = make_order(hft::Side::BID, 50000.0, 1.0);
    auto verdict = rm.check_order(order, 0, 0.0, 94000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::DRAWDOWN_LIMIT);
    ASSERT_EQ(rm.stats().rejected_drawdown, 1u);

    return true;
}

// ─── Test: Drawdown Below Threshold Passes ───────────────────

TEST(Risk, DrawdownPass) {
    hft::RiskConfig config;
    config.max_drawdown_pct = 0.05;
    hft::RiskManager rm(config);

    rm.update_equity(100000.0);
    // Small dip — 3% drawdown < 5% limit
    rm.update_equity(97000.0);

    auto order = make_order(hft::Side::BID, 50000.0, 1.0);
    auto verdict = rm.check_order(order, 0, 0.0, 97000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::PASS);

    return true;
}

// ─── Test: Daily Loss Limit ──────────────────────────────────

TEST(Risk, DailyLossLimit) {
    hft::RiskConfig config;
    config.max_daily_loss_pct = 0.03; // 3%
    config.max_drawdown_pct   = 0.10; // 10% (won't trigger first)
    config.circuit_breaker_cooldown_ns = 1;
    hft::RiskManager rm(config);

    // Day starts at 100k
    rm.update_equity(100000.0);
    rm.new_trading_day();

    // Loss to 96k → 4% daily loss > 3% limit
    rm.update_equity(96000.0);

    auto order = make_order(hft::Side::BID, 50000.0, 1.0);
    auto verdict = rm.check_order(order, 0, 0.0, 96000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::DAILY_LOSS_LIMIT);

    return true;
}

// ─── Test: Single Order Size Limit ───────────────────────────

TEST(Risk, OrderSizeLimit) {
    hft::RiskConfig config;
    config.max_single_order_pct = 0.02; // 2% max per order
    config.max_drawdown_pct     = 1.0;  // Disable drawdown for this test
    config.max_daily_loss_pct   = 1.0;  // Disable daily loss
    hft::RiskManager rm(config);

    rm.update_equity(100000.0);

    // Order value = 50000 * 1.0 = 50000, portfolio = 100000
    // That's 50% — way over 2%
    auto order = make_order(hft::Side::BID, 50000.0, 1.0);
    auto verdict = rm.check_order(order, 0, 0.0, 100000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::ORDER_SIZE_LIMIT);

    return true;
}

// ─── Test: Small Order Passes Size Limit ─────────────────────

TEST(Risk, OrderSizePass) {
    hft::RiskConfig config;
    config.max_single_order_pct = 0.02; // 2% max per order
    config.max_drawdown_pct     = 1.0;
    config.max_daily_loss_pct   = 1.0;
    hft::RiskManager rm(config);

    rm.update_equity(100000.0);

    // Order value = 100 * 0.01 = 1.0, portfolio = 100000
    // That's 0.001% — well under 2%
    auto order = make_order(hft::Side::BID, 100.0, 0.01);
    auto verdict = rm.check_order(order, 0, 0.0, 100000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::PASS);

    return true;
}

// ─── Test: Circuit Breaker Cooldown ──────────────────────────

TEST(Risk, CircuitBreakerCooldown) {
    hft::RiskConfig config;
    config.max_position = hft::qty_to_fixed(10.0);
    // 100ms cooldown for testing
    config.circuit_breaker_cooldown_ns = 100'000'000LL;
    hft::RiskManager rm(config);
    rm.update_equity(100000.0);

    // Trigger a position limit breach → circuit breaker activates
    auto bad_order = make_order(hft::Side::BID, 50000.0, 20.0);
    auto verdict = rm.check_order(bad_order, 0, 0.0, 100000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::POSITION_LIMIT);

    // Circuit breaker should now be active
    ASSERT_TRUE(rm.is_circuit_breaker_active());

    // Even a small valid order should be rejected during cooldown
    auto good_order = make_order(hft::Side::BID, 50000.0, 1.0);
    verdict = rm.check_order(good_order, 0, 0.0, 100000.0);
    ASSERT_EQ(verdict, hft::RiskVerdict::CIRCUIT_BREAKER);

    return true;
}

// ─── Test: Legacy bool interface ─────────────────────────────

TEST(Risk, LegacyBoolInterface) {
    hft::RiskConfig config;
    config.max_position = hft::qty_to_fixed(100.0);
    hft::RiskManager rm(config);
    rm.update_equity(100000.0);

    auto order = make_order(hft::Side::BID, 50000.0, 50.0);
    bool ok = rm.check_order(order, 0, 0.0);
    ASSERT_TRUE(ok);

    return true;
}

// ─── Test: Reset clears all state ────────────────────────────

TEST(Risk, Reset) {
    hft::RiskConfig config;
    config.max_position = hft::qty_to_fixed(10.0);
    config.circuit_breaker_cooldown_ns = 100'000'000'000LL; // 100s
    hft::RiskManager rm(config);
    rm.update_equity(100000.0);

    // Trip circuit breaker
    auto bad_order = make_order(hft::Side::BID, 50000.0, 20.0);
    rm.check_order(bad_order, 0, 0.0, 100000.0);
    ASSERT_TRUE(rm.is_circuit_breaker_active());

    // Reset
    rm.reset();
    ASSERT_FALSE(rm.is_circuit_breaker_active());
    ASSERT_EQ(rm.stats().orders_checked, 0u);

    return true;
}

// ─── Test: Statistics Tracking ───────────────────────────────

TEST(Risk, StatsTracking) {
    hft::RiskConfig config;
    config.max_position = hft::qty_to_fixed(100.0);
    config.max_drawdown_pct = 1.0;
    config.max_daily_loss_pct = 1.0;
    config.circuit_breaker_cooldown_ns = 1;
    hft::RiskManager rm(config);
    rm.update_equity(100000.0);

    // Pass 3 orders
    for (int i = 0; i < 3; ++i) {
        auto order = make_order(hft::Side::BID, 100.0, 1.0);
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
    config.max_daily_loss_pct = 0.03;
    config.max_drawdown_pct   = 1.0;
    config.circuit_breaker_cooldown_ns = 1;
    hft::RiskManager rm(config);

    // Day 1: start at 100k, lose to 96k → 4% daily loss
    rm.update_equity(100000.0);
    rm.new_trading_day();
    rm.update_equity(96000.0);
    ASSERT_GT(rm.current_daily_loss(), 0.03);

    // New day resets daily tracking
    rm.new_trading_day();
    // After new day, daily loss restarts from current equity
    ASSERT_TRUE(rm.current_daily_loss() < 0.001);

    return true;
}
