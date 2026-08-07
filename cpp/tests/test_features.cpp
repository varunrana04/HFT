/**
 * @file test_features.cpp
 * @brief Unit tests for the FeatureEngine (all 6 alpha signals).
 */

#include "features.h"
#include "order_book.h"
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
struct TestCase2F { std::string name; std::function<bool()> func; };
extern std::vector<TestCase2F>& get_tests();

#define ASSERT_TRUE(expr) do { if (!(expr)) { std::cerr << "  FAIL: " << #expr << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { std::cerr << "  FAIL: " << #a << " == " << #b << " (" << static_cast<int>(a) << " != " << static_cast<int>(b) << ") [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_NE(a, b) do { if ((a) == (b)) { std::cerr << "  FAIL: " << #a << " != " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_GT(a, b) do { if (!((a) > (b))) { std::cerr << "  FAIL: " << #a << " > " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_LT(a, b) do { if (!((a) < (b))) { std::cerr << "  FAIL: " << #a << " < " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_GE(a, b) do { if (!((a) >= (b))) { std::cerr << "  FAIL: " << #a << " >= " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)

#define TEST(suite, name)                                         \
    [[maybe_unused]] static bool test_##suite##_##name();         \
    [[maybe_unused]] static TestRegistrar reg_##suite##_##name(   \
        #suite "::" #name, test_##suite##_##name);                \
    static bool test_##suite##_##name()

// ─── Helpers ─────────────────────────────────────────────────

static hft::BookSnapshot make_book(
    double bid_price, double bid_qty,
    double ask_price, double ask_qty,
    int bid_levels = 1, int ask_levels = 1) {

    hft::BookSnapshot book{};
    std::memset(&book, 0, sizeof(book));
    book.timestamp_ns   = 1000;
    book.sequence_num   = 1;
    book.quality        = hft::DataQuality::VALID;

    book.bids[0].price       = hft::price_to_fixed(bid_price);
    book.bids[0].quantity    = hft::qty_to_fixed(bid_qty);
    book.bids[0].order_count = 5;
    book.bid_count           = bid_levels;
    book.best_bid_price      = book.bids[0].price;
    book.best_bid_qty        = book.bids[0].quantity;

    for (int i = 1; i < bid_levels && i < hft::MAX_BOOK_LEVELS; ++i) {
        book.bids[i].price       = hft::price_to_fixed(bid_price - i * 0.01);
        book.bids[i].quantity    = hft::qty_to_fixed(bid_qty);
        book.bids[i].order_count = 3;
    }

    book.asks[0].price       = hft::price_to_fixed(ask_price);
    book.asks[0].quantity    = hft::qty_to_fixed(ask_qty);
    book.asks[0].order_count = 5;
    book.ask_count           = ask_levels;
    book.best_ask_price      = book.asks[0].price;
    book.best_ask_qty        = book.asks[0].quantity;

    for (int i = 1; i < ask_levels && i < hft::MAX_BOOK_LEVELS; ++i) {
        book.asks[i].price       = hft::price_to_fixed(ask_price + i * 0.01);
        book.asks[i].quantity    = hft::qty_to_fixed(ask_qty);
        book.asks[i].order_count = 3;
    }

    for (int i = bid_levels; i < hft::MAX_BOOK_LEVELS; ++i)
        book.bids[i] = {hft::INVALID_PRICE, 0, 0, 0};
    for (int i = ask_levels; i < hft::MAX_BOOK_LEVELS; ++i)
        book.asks[i] = {hft::INVALID_PRICE, 0, 0, 0};

    return book;
}

static hft::Trade make_trade(double price, double qty, hft::Side side,
                              int64_t seq = 1) {
    hft::Trade t{};
    t.timestamp_ns  = 1000;
    t.sequence_num  = seq;
    t.price         = hft::price_to_fixed(price);
    t.quantity      = hft::qty_to_fixed(qty);
    t.instrument_id = 0;
    t.side          = side;
    t.quality       = hft::DataQuality::VALID;
    return t;
}

// ─── Microprice Tests ────────────────────────────────────────

TEST(Features, MicropriceSym) {
    hft::FeatureEngine engine;
    auto book  = make_book(100.0, 10.0, 101.0, 10.0);
    auto trade = make_trade(100.5, 1.0, hft::Side::BID);
    hft::FeatureVector fv = engine.compute_all(book, trade);

    double expected_mid = (100.0 + 101.0) / 2.0;
    ASSERT_TRUE(std::abs(fv.microprice - expected_mid) < 0.01);
    return true;
}

TEST(Features, MicropriceAsym) {
    hft::FeatureEngine engine;
    auto book  = make_book(100.0, 90.0, 101.0, 10.0);
    auto trade = make_trade(100.5, 1.0, hft::Side::BID);
    hft::FeatureVector fv = engine.compute_all(book, trade);

    // microprice = (10*100 + 90*101) / 100 = 100.9
    double expected = (10.0 * 100.0 + 90.0 * 101.0) / 100.0;
    ASSERT_TRUE(std::abs(fv.microprice - expected) < 0.01);
    ASSERT_GT(fv.microprice, 100.5);
    return true;
}

// ─── OFI Tests ───────────────────────────────────────────────

TEST(Features, OFIBidIncrease) {
    hft::FeatureEngine engine;
    auto book1  = make_book(100.0, 10.0, 101.0, 10.0);
    auto trade1 = make_trade(100.5, 1.0, hft::Side::BID);
    engine.compute_all(book1, trade1);

    auto book2  = make_book(100.0, 20.0, 101.0, 10.0);
    auto trade2 = make_trade(100.5, 1.0, hft::Side::BID, 2);
    hft::FeatureVector fv = engine.compute_all(book2, trade2);

    // OFI = (20-10) - (10-10) = 10
    ASSERT_GT(fv.ofi, 0.0);
    ASSERT_TRUE(std::abs(fv.ofi - 10.0) < 0.01);
    return true;
}

TEST(Features, OFIAskDecrease) {
    hft::FeatureEngine engine;
    auto book1  = make_book(100.0, 10.0, 101.0, 20.0);
    auto trade1 = make_trade(100.5, 1.0, hft::Side::BID);
    engine.compute_all(book1, trade1);

    auto book2  = make_book(100.0, 10.0, 101.0, 5.0);
    auto trade2 = make_trade(100.5, 1.0, hft::Side::BID, 2);
    hft::FeatureVector fv = engine.compute_all(book2, trade2);

    // OFI = 0 - (5-20) = 15
    ASSERT_GT(fv.ofi, 0.0);
    ASSERT_TRUE(std::abs(fv.ofi - 15.0) < 0.01);
    return true;
}

// ─── Spread BPS Tests ────────────────────────────────────────

TEST(Features, SpreadBPS) {
    hft::FeatureEngine engine;
    auto book  = make_book(100.0, 10.0, 101.0, 10.0);
    auto trade = make_trade(100.5, 1.0, hft::Side::BID);
    hft::FeatureVector fv = engine.compute_all(book, trade);

    double expected_bps = 1.0 / 100.5 * 10000.0;
    ASSERT_TRUE(std::abs(fv.spread_bps - expected_bps) < 0.1);
    ASSERT_GT(fv.spread_bps, 0.0);
    return true;
}

TEST(Features, SpreadBPSTight) {
    hft::FeatureEngine engine;
    auto book  = make_book(50000.00, 10.0, 50000.01, 10.0);
    auto trade = make_trade(50000.005, 1.0, hft::Side::BID);
    hft::FeatureVector fv = engine.compute_all(book, trade);

    ASSERT_GT(fv.spread_bps, 0.0);
    ASSERT_LT(fv.spread_bps, 1.0);
    return true;
}

// ─── VPIN Tests ──────────────────────────────────────────────

TEST(Features, VPINPureBuy) {
    hft::FeatureConfig config;
    config.vpin_bucket_size = 10.0;
    config.vpin_n_buckets   = 5;
    hft::FeatureEngine engine(config);
    auto book = make_book(100.0, 10.0, 101.0, 10.0);

    hft::FeatureVector fv{};
    for (int i = 0; i < 100; ++i) {
        auto trade = make_trade(100.5, 2.0, hft::Side::BID,
                                static_cast<int64_t>(i + 1));
        fv = engine.compute_all(book, trade);
    }
    ASSERT_GT(fv.vpin, 0.8);
    return true;
}

TEST(Features, VPINBalanced) {
    hft::FeatureConfig config;
    config.vpin_bucket_size = 10.0;
    config.vpin_n_buckets   = 5;
    hft::FeatureEngine engine(config);
    auto book = make_book(100.0, 10.0, 101.0, 10.0);

    hft::FeatureVector fv{};
    for (int i = 0; i < 100; ++i) {
        hft::Side side = (i % 2 == 0) ? hft::Side::BID : hft::Side::ASK;
        auto trade = make_trade(100.5, 2.0, side,
                                static_cast<int64_t>(i + 1));
        fv = engine.compute_all(book, trade);
    }
    ASSERT_LT(fv.vpin, 0.3);
    return true;
}

// ─── Realized Volatility Tests ───────────────────────────────

TEST(Features, RealizedVolConstant) {
    hft::FeatureConfig config;
    config.vol_window_ticks = 50;
    hft::FeatureEngine engine(config);
    auto book = make_book(100.0, 10.0, 101.0, 10.0);

    hft::FeatureVector fv{};
    for (int i = 0; i < 60; ++i) {
        auto trade = make_trade(100.0, 1.0, hft::Side::BID,
                                static_cast<int64_t>(i + 1));
        fv = engine.compute_all(book, trade);
    }
    ASSERT_TRUE(fv.realized_vol < 1e-10);
    return true;
}

TEST(Features, RealizedVolMoving) {
    hft::FeatureConfig config;
    config.vol_window_ticks = 50;
    hft::FeatureEngine engine(config);
    auto book = make_book(100.0, 10.0, 101.0, 10.0);

    hft::FeatureVector fv{};
    for (int i = 0; i < 60; ++i) {
        double price = 100.0 + (i % 2 == 0 ? 0.5 : -0.5);
        auto trade = make_trade(price, 1.0, hft::Side::BID,
                                static_cast<int64_t>(i + 1));
        fv = engine.compute_all(book, trade);
    }
    ASSERT_GT(fv.realized_vol, 0.0);
    return true;
}

// ─── Stat-Arb Z-Score Tests ─────────────────────────────────

TEST(Features, StatArbZScore) {
    hft::FeatureConfig config;
    config.stat_arb_lookback = 100;
    hft::FeatureEngine engine(config);

    hft::FeatureVector fv{};
    for (int i = 0; i < 100; ++i) {
        auto book  = make_book(100.0, 10.0, 101.0, 10.0);
        auto trade = make_trade(100.5, 1.0, hft::Side::BID,
                                static_cast<int64_t>(i + 1));
        fv = engine.compute_all(book, trade);
    }
    // Z near zero for constant mid
    ASSERT_TRUE(std::abs(fv.stat_arb_zscore) < 0.5);

    // Spike the price
    auto book_spike  = make_book(110.0, 10.0, 111.0, 10.0);
    auto trade_spike = make_trade(110.5, 1.0, hft::Side::BID, 101);
    fv = engine.compute_all(book_spike, trade_spike);

    ASSERT_GT(fv.stat_arb_zscore, 1.0);
    return true;
}

// ─── Edge Cases ──────────────────────────────────────────────

TEST(Features, InvalidBook) {
    hft::FeatureEngine engine;
    hft::BookSnapshot book{};
    std::memset(&book, 0, sizeof(book));
    book.quality = hft::DataQuality::CROSSED_BOOK;
    book.best_bid_price = hft::INVALID_PRICE;
    book.best_ask_price = hft::INVALID_PRICE;

    auto trade = make_trade(100.0, 1.0, hft::Side::BID);
    hft::FeatureVector fv = engine.compute_all(book, trade);

    ASSERT_EQ(fv.regime, hft::Regime::UNKNOWN);
    ASSERT_TRUE(fv.microprice == 0.0);
    return true;
}

TEST(Features, Reset) {
    hft::FeatureEngine engine;
    auto book  = make_book(100.0, 10.0, 101.0, 10.0);
    auto trade = make_trade(100.5, 1.0, hft::Side::BID);
    engine.compute_all(book, trade);
    engine.reset();

    auto fv = engine.compute_all(book, trade);
    ASSERT_TRUE(std::abs(fv.ofi) < 0.01);
    return true;
}
