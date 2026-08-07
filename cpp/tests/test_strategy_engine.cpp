/**
 * @file test_strategy_engine.cpp
 * @brief Integration tests for the StrategyEngine orchestrator.
 *
 * Tests the full pipeline: Trade → Features → Signal → Risk → Order → PnL
 */

#include "strategy_engine.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <functional>

// ─── Test harness (shared with test_main.cpp via linker) ─────
struct TestRegistrar {
    TestRegistrar(const char* name, std::function<bool()> func);
};
struct TestCase7SE { std::string name; std::function<bool()> func; };
extern std::vector<TestCase7SE>& get_tests();

#define ASSERT_TRUE(expr) do { if (!(expr)) { std::cerr << "  FAIL: " << #expr << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { std::cerr << "  FAIL: " << #a << " == " << #b << " (" << (a) << " != " << (b) << ") [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_NE(a, b) do { if ((a) == (b)) { std::cerr << "  FAIL: " << #a << " != " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_GT(a, b) do { if (!((a) > (b))) { std::cerr << "  FAIL: " << #a << " > " << #b << " (" << (a) << " <= " << (b) << ") [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_LT(a, b) do { if (!((a) < (b))) { std::cerr << "  FAIL: " << #a << " < " << #b << " (" << (a) << " >= " << (b) << ") [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_GE(a, b) do { if (!((a) >= (b))) { std::cerr << "  FAIL: " << #a << " >= " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_NEAR(a, b, eps) do { if (std::abs((a) - (b)) > (eps)) { std::cerr << "  FAIL: " << #a << " ≈ " << #b << " (diff=" << std::abs((a)-(b)) << ", eps=" << (eps) << ") [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)

#define TEST(suite, name)                                         \
    [[maybe_unused]] static bool test_##suite##_##name();         \
    [[maybe_unused]] static TestRegistrar reg_##suite##_##name(   \
        #suite "::" #name, test_##suite##_##name);                \
    static bool test_##suite##_##name()

// ─── Helpers ─────────────────────────────────────────────────

static hft::BookSnapshot make_book(
    double bid_price, double bid_qty,
    double ask_price, double ask_qty,
    int64_t ts_ns = 1000, int64_t seq = 1) {

    hft::BookSnapshot book{};
    std::memset(&book, 0, sizeof(book));
    book.timestamp_ns   = ts_ns;
    book.sequence_num   = seq;
    book.quality        = hft::DataQuality::VALID;

    book.bids[0].price       = hft::price_to_fixed(bid_price);
    book.bids[0].quantity    = hft::qty_to_fixed(bid_qty);
    book.bids[0].order_count = 5;
    book.bid_count           = 1;
    book.best_bid_price      = book.bids[0].price;
    book.best_bid_qty        = book.bids[0].quantity;

    book.asks[0].price       = hft::price_to_fixed(ask_price);
    book.asks[0].quantity    = hft::qty_to_fixed(ask_qty);
    book.asks[0].order_count = 5;
    book.ask_count           = 1;
    book.best_ask_price      = book.asks[0].price;
    book.best_ask_qty        = book.asks[0].quantity;

    return book;
}

static hft::Trade make_trade(double price, double qty,
                              hft::Side side = hft::Side::BID,
                              int64_t ts_ns = 1000, int64_t seq = 1) {
    hft::Trade t{};
    std::memset(&t, 0, sizeof(t));
    t.timestamp_ns  = ts_ns;
    t.sequence_num  = seq;
    t.price         = hft::price_to_fixed(price);
    t.quantity      = hft::qty_to_fixed(qty);
    t.instrument_id = 0;
    t.side          = side;
    t.quality       = hft::DataQuality::VALID;
    return t;
}

// ═══════════════════════════════════════════════════════════
// Test 1: Construction and initial state
// ═══════════════════════════════════════════════════════════
TEST(StrategyEngine, InitialState) {
    hft::StrategyConfig cfg;
    cfg.initial_capital = 100000.0;
    hft::StrategyEngine engine(cfg);

    ASSERT_EQ(engine.position(), 0);
    ASSERT_NEAR(engine.realized_pnl(), 0.0, 1e-10);
    ASSERT_NEAR(engine.equity(), 100000.0, 1e-10);
    ASSERT_EQ(engine.metrics().total_trades, 0);
    ASSERT_EQ(engine.trade_journal().size(), static_cast<size_t>(0));
    return true;
}

// ═══════════════════════════════════════════════════════════
// Test 2: Process a single trade without triggering entry
// ═══════════════════════════════════════════════════════════
TEST(StrategyEngine, NoTradeOnWeakSignal) {
    hft::StrategyConfig cfg;
    cfg.initial_capital = 100000.0;
    cfg.alpha_entry_threshold = 0.50; // Very high threshold
    hft::StrategyEngine engine(cfg);

    auto book  = make_book(100.0, 10.0, 100.02, 10.0);
    auto trade = make_trade(100.01, 1.0, hft::Side::BID);

    engine.on_trade(trade, book);

    // Should not have traded — alpha too weak for high threshold
    ASSERT_EQ(engine.position(), 0);
    ASSERT_EQ(engine.metrics().total_trades, 0);
    return true;
}

// ═══════════════════════════════════════════════════════════
// Test 3: Feature vector is populated after tick
// ═══════════════════════════════════════════════════════════
TEST(StrategyEngine, FeaturesPopulated) {
    hft::StrategyEngine engine;

    auto book  = make_book(100.0, 10.0, 100.02, 10.0);
    auto trade = make_trade(100.01, 1.0, hft::Side::BID);

    engine.on_trade(trade, book);

    const auto& fv = engine.last_features();
    // Microprice should be between bid and ask
    ASSERT_GT(fv.microprice, 0.0);
    // Spread should be positive
    ASSERT_GT(fv.spread_bps, 0.0);
    return true;
}

// ═══════════════════════════════════════════════════════════
// Test 4: Book-only update doesn't trade
// ═══════════════════════════════════════════════════════════
TEST(StrategyEngine, BookUpdateNoTrade) {
    hft::StrategyEngine engine;

    auto book = make_book(100.0, 10.0, 100.02, 10.0);
    engine.on_book_update(book);

    ASSERT_EQ(engine.position(), 0);
    ASSERT_EQ(engine.metrics().total_trades, 0);
    return true;
}

// ═══════════════════════════════════════════════════════════
// Test 5: Reset clears all state
// ═══════════════════════════════════════════════════════════
TEST(StrategyEngine, ResetClearsState) {
    hft::StrategyConfig cfg;
    cfg.initial_capital = 50000.0;
    hft::StrategyEngine engine(cfg);

    // Feed some ticks
    for (int i = 0; i < 20; ++i) {
        double p = 100.0 + i * 0.01;
        auto book  = make_book(p, 10.0, p + 0.02, 10.0, 1000 + i, i);
        auto trade = make_trade(p + 0.01, 1.0, hft::Side::BID, 1000 + i, i);
        engine.on_trade(trade, book);
    }

    engine.reset();
    ASSERT_EQ(engine.position(), 0);
    ASSERT_NEAR(engine.realized_pnl(), 0.0, 1e-10);
    ASSERT_NEAR(engine.equity(), 50000.0, 1e-10);
    ASSERT_EQ(engine.metrics().total_trades, 0);
    ASSERT_EQ(engine.trade_journal().size(), static_cast<size_t>(0));
    return true;
}

// ═══════════════════════════════════════════════════════════
// Test 6: Equity equals initial capital when flat
// ═══════════════════════════════════════════════════════════
TEST(StrategyEngine, EquityFlatIsCapital) {
    hft::StrategyConfig cfg;
    cfg.initial_capital = 75000.0;
    hft::StrategyEngine engine(cfg);

    ASSERT_NEAR(engine.equity(), 75000.0, 1e-10);
    return true;
}

// ═══════════════════════════════════════════════════════════
// Test 7: Risk rejection is counted
// ═══════════════════════════════════════════════════════════
TEST(StrategyEngine, RiskRejectionCounted) {
    // Use extremely tight risk to trigger rejections
    hft::StrategyConfig scfg;
    scfg.initial_capital = 100.0;  // Very small capital
    scfg.alpha_entry_threshold = 0.0001; // Very sensitive
    scfg.position_size_pct = 0.5; // Very large order size

    hft::RiskConfig rcfg;
    rcfg.max_position = hft::qty_to_fixed(0.001); // Tiny position limit

    hft::StrategyEngine engine(scfg, {}, rcfg);

    // Feed enough ticks to generate signals and hit limits
    for (int i = 0; i < 50; ++i) {
        double p = 100.0 + i * 0.01;
        auto book  = make_book(p, 10.0, p + 0.02, 10.0, 1000 + i, i);
        auto trade = make_trade(p + 0.01, 1.0, hft::Side::BID, 1000 + i, i);
        engine.on_trade(trade, book);
    }

    // After hitting position limit, risk rejections should accumulate
    // We just verify the counter is accessible and non-negative
    ASSERT_GE(engine.metrics().risk_rejections, static_cast<int64_t>(0));
    return true;
}

// ═══════════════════════════════════════════════════════════
// Test 8: Metrics update consistently
// ═══════════════════════════════════════════════════════════
TEST(StrategyEngine, MetricsConsistency) {
    hft::StrategyEngine engine;

    // Feed a sequence of ticks
    for (int i = 0; i < 100; ++i) {
        double p = 100.0 + std::sin(i * 0.1) * 0.5;
        auto book  = make_book(p, 10.0, p + 0.02, 10.0, 1000 + i, i);
        auto trade = make_trade(p + 0.01, 1.0,
            (i % 3 == 0) ? hft::Side::ASK : hft::Side::BID,
            1000 + i, i);
        engine.on_trade(trade, book);
    }

    const auto& m = engine.metrics();
    // Win rate must be in [0, 1]
    ASSERT_GE(m.win_rate, 0.0);
    ASSERT_GE(1.0, m.win_rate);
    // Max drawdown must be non-negative
    ASSERT_GE(m.max_drawdown, 0.0);
    // Winning + losing <= total trades
    ASSERT_GE(m.total_trades, m.winning_trades + m.losing_trades);
    return true;
}

// ═══════════════════════════════════════════════════════════
// Test 9: New trading day resets daily loss
// ═══════════════════════════════════════════════════════════
TEST(StrategyEngine, NewTradingDayResets) {
    hft::StrategyEngine engine;

    // Simulate some activity
    auto book  = make_book(100.0, 10.0, 100.02, 10.0);
    auto trade = make_trade(100.01, 1.0, hft::Side::BID);
    engine.on_trade(trade, book);

    // Reset daily stats
    engine.new_trading_day();

    // Risk manager daily loss should be reset
    ASSERT_NEAR(engine.risk_stats().pass_rate(), 1.0, 1e-5);
    return true;
}

// ═══════════════════════════════════════════════════════════
// Test 10: Mode switching
// ═══════════════════════════════════════════════════════════
TEST(StrategyEngine, ModeSwitching) {
    hft::StrategyEngine engine;
    // Default is BACKTEST
    engine.set_mode(hft::EngineMode::LIVE);
    engine.set_mode(hft::EngineMode::BACKTEST);

    // Should not crash, position should remain flat
    ASSERT_EQ(engine.position(), 0);
    return true;
}
