/**
 * @file test_data_validator.cpp
 * @brief Tests for the data quality validation layer.
 *
 * Tests ensure corrupt/noisy data is REJECTED before it can
 * pollute signals. This is critical for backtesting integrity.
 */
#include "data_validator.h"
#include "clock.h"
#include <iostream>
#include <vector>
#include <string>
#include <functional>

struct TestCase;
extern std::vector<TestCase>& get_tests();

struct TestRegistrar {
    TestRegistrar(const char* name, std::function<bool()> func);
};

#define ASSERT_TRUE(expr) do { if (!(expr)) { std::cerr << "  FAIL: " << #expr << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) do { if (static_cast<int>(a) != static_cast<int>(b)) { std::cerr << "  FAIL: " << #a << " == " << #b << " (" << static_cast<int>(a) << " != " << static_cast<int>(b) << ") [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_GT(a, b) do { if (!(static_cast<int>(a) > static_cast<int>(b))) { std::cerr << "  FAIL: " << #a << " > " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)

#define TEST(suite, name) \
    [[maybe_unused]] static bool test_##suite##_##name(); \
    [[maybe_unused]] static TestRegistrar reg_##suite##_##name(#suite "::" #name, test_##suite##_##name); \
    static bool test_##suite##_##name()

static hft::Trade make_valid_trade(int64_t ts, int64_t seq, double price,
                                   double qty, hft::Side side) {
    hft::Trade t{};
    t.timestamp_ns = ts;
    t.sequence_num = seq;
    t.price = hft::price_to_fixed(price);
    t.quantity = hft::qty_to_fixed(qty);
    t.side = side;
    t.quality = hft::DataQuality::VALID;
    return t;
}

TEST(DataValidator, ValidTrade_Accepted) {
    hft::DataValidator validator;
    int64_t now = hft::now_ns();
    auto t = make_valid_trade(now, 1, 50000.0, 0.1, hft::Side::BID);
    ASSERT_EQ(validator.validate_trade(t, now), hft::DataQuality::VALID);
    ASSERT_EQ(validator.stats().valid_ticks, uint64_t{1});
    return true;
}

TEST(DataValidator, StaleTrade_Rejected) {
    hft::DataValidator validator;
    int64_t now = hft::now_ns();
    // Trade from 10 seconds ago (threshold is 5s)
    int64_t stale_ts = now - 10'000'000'000LL;
    auto t = make_valid_trade(stale_ts, 1, 50000.0, 0.1, hft::Side::BID);
    ASSERT_EQ(validator.validate_trade(t, now),
              hft::DataQuality::STALE_TIMESTAMP);
    ASSERT_EQ(validator.stats().stale_timestamps, uint64_t{1});
    return true;
}

TEST(DataValidator, FutureTimestamp_Rejected) {
    hft::DataValidator validator;
    int64_t now = hft::now_ns();
    // Trade 5 seconds in the future (threshold is 1s)
    int64_t future_ts = now + 5'000'000'000LL;
    auto t = make_valid_trade(future_ts, 1, 50000.0, 0.1, hft::Side::BID);
    ASSERT_EQ(validator.validate_trade(t, now),
              hft::DataQuality::STALE_TIMESTAMP);
    return true;
}

TEST(DataValidator, OutOfSequence_Rejected) {
    hft::DataValidator validator;
    int64_t now = hft::now_ns();
    auto t1 = make_valid_trade(now, 10, 50000.0, 0.1, hft::Side::BID);
    ASSERT_EQ(validator.validate_trade(t1, now), hft::DataQuality::VALID);

    // Sequence goes backwards
    auto t2 = make_valid_trade(now + 1, 5, 50000.0, 0.1, hft::Side::BID);
    ASSERT_EQ(validator.validate_trade(t2, now + 1),
              hft::DataQuality::OUT_OF_SEQUENCE);
    return true;
}

TEST(DataValidator, Duplicate_Rejected) {
    hft::DataValidator validator;
    int64_t now = hft::now_ns();
    auto t1 = make_valid_trade(now, 10, 50000.0, 0.1, hft::Side::BID);
    ASSERT_EQ(validator.validate_trade(t1, now), hft::DataQuality::VALID);

    // Same sequence number = duplicate
    auto t2 = make_valid_trade(now + 1, 10, 50000.0, 0.1, hft::Side::BID);
    ASSERT_EQ(validator.validate_trade(t2, now + 1),
              hft::DataQuality::DUPLICATE);
    ASSERT_EQ(validator.stats().duplicates, uint64_t{1});
    return true;
}

TEST(DataValidator, PriceAnomaly_Rejected) {
    hft::DataValidator validator;
    int64_t now = hft::now_ns();

    // First trade establishes reference price
    auto t1 = make_valid_trade(now, 1, 50000.0, 0.1, hft::Side::BID);
    ASSERT_EQ(validator.validate_trade(t1, now), hft::DataQuality::VALID);

    // 10% price jump (threshold is 5%)
    auto t2 = make_valid_trade(now + 1, 2, 55001.0, 0.1, hft::Side::BID);
    ASSERT_EQ(validator.validate_trade(t2, now + 1),
              hft::DataQuality::PRICE_ANOMALY);
    ASSERT_EQ(validator.stats().price_anomalies, uint64_t{1});
    return true;
}

TEST(DataValidator, NegativePrice_Rejected) {
    hft::DataValidator validator;
    int64_t now = hft::now_ns();
    hft::Trade t{};
    t.timestamp_ns = now;
    t.sequence_num = 1;
    t.price = -100;
    t.quantity = hft::qty_to_fixed(0.1);
    t.side = hft::Side::BID;
    ASSERT_EQ(validator.validate_trade(t, now),
              hft::DataQuality::PRICE_ANOMALY);
    return true;
}

TEST(DataValidator, NegativeQuantity_Rejected) {
    hft::DataValidator validator;
    int64_t now = hft::now_ns();
    hft::Trade t{};
    t.timestamp_ns = now;
    t.sequence_num = 1;
    t.price = hft::price_to_fixed(50000.0);
    t.quantity = -1;
    t.side = hft::Side::BID;
    ASSERT_EQ(validator.validate_trade(t, now),
              hft::DataQuality::QTY_ANOMALY);
    return true;
}

TEST(DataValidator, CrossedBook_Rejected) {
    hft::DataValidator validator;
    int64_t now = hft::now_ns();
    hft::BookSnapshot book{};
    book.timestamp_ns = now;
    book.sequence_num = 1;
    book.best_bid_price = hft::price_to_fixed(101.0);
    book.best_ask_price = hft::price_to_fixed(100.0);  // Crossed!
    book.best_bid_qty = hft::qty_to_fixed(1.0);
    book.best_ask_qty = hft::qty_to_fixed(1.0);
    book.bid_count = 1;
    book.ask_count = 1;
    book.bids[0] = {hft::price_to_fixed(101.0), hft::qty_to_fixed(1.0), 1, 0};
    book.asks[0] = {hft::price_to_fixed(100.0), hft::qty_to_fixed(1.0), 1, 0};

    ASSERT_EQ(validator.validate_book(book, now),
              hft::DataQuality::CROSSED_BOOK);
    return true;
}

TEST(DataValidator, AcceptanceRate) {
    hft::DataValidator validator;
    int64_t now = hft::now_ns();

    // 3 valid trades
    for (int i = 1; i <= 3; ++i) {
        auto t = make_valid_trade(now + i, i, 50000.0, 0.1, hft::Side::BID);
        validator.validate_trade(t, now + i);
    }

    // 1 invalid trade (negative price)
    hft::Trade bad{};
    bad.timestamp_ns = now + 4;
    bad.sequence_num = 4;
    bad.price = -1;
    bad.quantity = hft::qty_to_fixed(0.1);
    bad.side = hft::Side::BID;
    validator.validate_trade(bad, now + 4);

    ASSERT_EQ(validator.stats().total_ticks_seen, uint64_t{4});
    ASSERT_EQ(validator.stats().valid_ticks, uint64_t{3});
    // Acceptance rate = 3/4 = 0.75
    ASSERT_TRUE(validator.stats().acceptance_rate() > 0.74);
    ASSERT_TRUE(validator.stats().acceptance_rate() < 0.76);
    return true;
}

TEST(DataValidator, Reset_ClearsState) {
    hft::DataValidator validator;
    int64_t now = hft::now_ns();
    auto t = make_valid_trade(now, 100, 50000.0, 0.1, hft::Side::BID);
    validator.validate_trade(t, now);
    ASSERT_GT(validator.stats().total_ticks_seen, uint64_t{0});

    validator.reset();
    ASSERT_EQ(validator.stats().total_ticks_seen, uint64_t{0});
    ASSERT_EQ(validator.stats().valid_ticks, uint64_t{0});
    return true;
}
