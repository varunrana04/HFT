/**
 * @file test_order_book.cpp
 * @brief Tests for the L2 order book engine with data validation.
 */
#include "order_book.h"
#include <iostream>
#include <vector>
#include <string>
#include <functional>

struct TestRegistrar {
    TestRegistrar(const char* name, std::function<bool()> func);
};

#define ASSERT_TRUE(expr) do { if (!(expr)) { std::cerr << "  FAIL: " << #expr << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { std::cerr << "  FAIL: " << #a << " == " << #b << " (" << static_cast<int>(a) << " != " << static_cast<int>(b) << ") [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_NE(a, b) do { if ((a) == (b)) { std::cerr << "  FAIL: " << #a << " != " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_GT(a, b) do { if (!((a) > (b))) { std::cerr << "  FAIL: " << #a << " > " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_LT(a, b) do { if (!((a) < (b))) { std::cerr << "  FAIL: " << #a << " < " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)

#define TEST(suite, name) \
    [[maybe_unused]] static bool test_##suite##_##name(); \
    [[maybe_unused]] static TestRegistrar reg_##suite##_##name(#suite "::" #name, test_##suite##_##name); \
    static bool test_##suite##_##name()

TEST(OrderBook, EmptyOnConstruction) {
    hft::OrderBook book(1);
    ASSERT_FALSE(book.is_valid());
    ASSERT_EQ(book.best_bid(), hft::INVALID_PRICE);
    ASSERT_EQ(book.best_ask(), hft::INVALID_PRICE);
    return true;
}

TEST(OrderBook, SingleBidAsk) {
    hft::OrderBook book(1);
    hft::LevelUpdate bid{};
    bid.timestamp_ns = 1000;
    bid.sequence_num = 1;
    bid.price = hft::price_to_fixed(100.0);
    bid.quantity = hft::qty_to_fixed(10.0);
    bid.order_count = 5;
    bid.side = hft::Side::BID;

    hft::LevelUpdate ask{};
    ask.timestamp_ns = 1001;
    ask.sequence_num = 2;
    ask.price = hft::price_to_fixed(100.05);
    ask.quantity = hft::qty_to_fixed(8.0);
    ask.order_count = 3;
    ask.side = hft::Side::ASK;

    ASSERT_EQ(book.apply_update(bid), hft::DataQuality::VALID);
    ASSERT_EQ(book.apply_update(ask), hft::DataQuality::VALID);
    ASSERT_TRUE(book.is_valid());
    ASSERT_EQ(book.best_bid(), hft::price_to_fixed(100.0));
    ASSERT_EQ(book.best_ask(), hft::price_to_fixed(100.05));
    return true;
}

TEST(OrderBook, MultipleBidLevels_SortedDescending) {
    hft::OrderBook book(1);
    int64_t prices[] = {
        hft::price_to_fixed(99.0),
        hft::price_to_fixed(100.0),
        hft::price_to_fixed(98.5),
        hft::price_to_fixed(99.5)
    };

    for (int i = 0; i < 4; ++i) {
        hft::LevelUpdate u{};
        u.timestamp_ns = 1000 + i;
        u.sequence_num = static_cast<int64_t>(i + 1);
        u.price = prices[i];
        u.quantity = hft::qty_to_fixed(1.0);
        u.order_count = 1;
        u.side = hft::Side::BID;
        book.apply_update(u);
    }

    // Add an ask so the book is valid
    hft::LevelUpdate ask{};
    ask.timestamp_ns = 2000;
    ask.sequence_num = 10;
    ask.price = hft::price_to_fixed(101.0);
    ask.quantity = hft::qty_to_fixed(1.0);
    ask.order_count = 1;
    ask.side = hft::Side::ASK;
    book.apply_update(ask);

    // Best bid should be 100.0 (highest)
    ASSERT_EQ(book.best_bid(), hft::price_to_fixed(100.0));

    // Verify bids are sorted descending
    const auto& snap = book.snapshot();
    for (int32_t i = 1; i < snap.bid_count; ++i) {
        ASSERT_GT(snap.bids[i - 1].price, snap.bids[i].price);
    }
    return true;
}

TEST(OrderBook, MultipleAskLevels_SortedAscending) {
    hft::OrderBook book(1);
    int64_t prices[] = {
        hft::price_to_fixed(101.0),
        hft::price_to_fixed(100.5),
        hft::price_to_fixed(102.0),
        hft::price_to_fixed(100.1)
    };

    // Add a bid first
    hft::LevelUpdate bid{};
    bid.timestamp_ns = 999;
    bid.sequence_num = 0;
    bid.price = hft::price_to_fixed(99.0);
    bid.quantity = hft::qty_to_fixed(1.0);
    bid.order_count = 1;
    bid.side = hft::Side::BID;
    book.apply_update(bid);

    for (int i = 0; i < 4; ++i) {
        hft::LevelUpdate u{};
        u.timestamp_ns = 1000 + i;
        u.sequence_num = static_cast<int64_t>(i + 1);
        u.price = prices[i];
        u.quantity = hft::qty_to_fixed(1.0);
        u.order_count = 1;
        u.side = hft::Side::ASK;
        book.apply_update(u);
    }

    // Best ask should be 100.1 (lowest)
    ASSERT_EQ(book.best_ask(), hft::price_to_fixed(100.1));

    // Verify asks are sorted ascending
    const auto& snap = book.snapshot();
    for (int32_t i = 1; i < snap.ask_count; ++i) {
        ASSERT_LT(snap.asks[i - 1].price, snap.asks[i].price);
    }
    return true;
}

TEST(OrderBook, RemoveLevel) {
    hft::OrderBook book(1);

    // Add two bid levels
    hft::LevelUpdate b1{};
    b1.timestamp_ns = 1000; b1.sequence_num = 1;
    b1.price = hft::price_to_fixed(100.0);
    b1.quantity = hft::qty_to_fixed(5.0);
    b1.order_count = 1; b1.side = hft::Side::BID;
    book.apply_update(b1);

    hft::LevelUpdate b2{};
    b2.timestamp_ns = 1001; b2.sequence_num = 2;
    b2.price = hft::price_to_fixed(99.0);
    b2.quantity = hft::qty_to_fixed(3.0);
    b2.order_count = 1; b2.side = hft::Side::BID;
    book.apply_update(b2);

    ASSERT_EQ(book.snapshot().bid_count, 2);

    // Remove best bid (quantity = 0)
    hft::LevelUpdate rm{};
    rm.timestamp_ns = 1002; rm.sequence_num = 3;
    rm.price = hft::price_to_fixed(100.0);
    rm.quantity = 0;
    rm.side = hft::Side::BID;
    book.apply_update(rm);

    ASSERT_EQ(book.snapshot().bid_count, 1);
    ASSERT_EQ(book.best_bid(), hft::price_to_fixed(99.0));
    return true;
}

TEST(OrderBook, UpdateExistingLevel) {
    hft::OrderBook book(1);

    hft::LevelUpdate u1{};
    u1.timestamp_ns = 1000; u1.sequence_num = 1;
    u1.price = hft::price_to_fixed(100.0);
    u1.quantity = hft::qty_to_fixed(5.0);
    u1.order_count = 3; u1.side = hft::Side::BID;
    book.apply_update(u1);

    // Update same price level with new quantity
    hft::LevelUpdate u2{};
    u2.timestamp_ns = 1001; u2.sequence_num = 2;
    u2.price = hft::price_to_fixed(100.0);
    u2.quantity = hft::qty_to_fixed(10.0);
    u2.order_count = 7; u2.side = hft::Side::BID;
    book.apply_update(u2);

    // Should still be 1 level, but updated quantity
    ASSERT_EQ(book.snapshot().bid_count, 1);
    ASSERT_EQ(book.snapshot().bids[0].quantity, hft::qty_to_fixed(10.0));
    ASSERT_EQ(book.snapshot().bids[0].order_count, 7);
    return true;
}

TEST(OrderBook, CrossedBook_Rejected) {
    hft::OrderBook book(1);

    // Add ask at 100
    hft::LevelUpdate ask{};
    ask.timestamp_ns = 1000; ask.sequence_num = 1;
    ask.price = hft::price_to_fixed(100.0);
    ask.quantity = hft::qty_to_fixed(1.0);
    ask.order_count = 1; ask.side = hft::Side::ASK;
    book.apply_update(ask);

    // Add bid at 101 (higher than ask = crossed!)
    hft::LevelUpdate bid{};
    bid.timestamp_ns = 1001; bid.sequence_num = 2;
    bid.price = hft::price_to_fixed(101.0);
    bid.quantity = hft::qty_to_fixed(1.0);
    bid.order_count = 1; bid.side = hft::Side::BID;

    auto result = book.apply_update(bid);
    ASSERT_EQ(result, hft::DataQuality::CROSSED_BOOK);
    return true;
}

TEST(OrderBook, Spread_Calculation) {
    hft::OrderBook book(1);

    hft::LevelUpdate bid{};
    bid.timestamp_ns = 1000; bid.sequence_num = 1;
    bid.price = hft::price_to_fixed(100.0);
    bid.quantity = hft::qty_to_fixed(1.0);
    bid.order_count = 1; bid.side = hft::Side::BID;
    book.apply_update(bid);

    hft::LevelUpdate ask{};
    ask.timestamp_ns = 1001; ask.sequence_num = 2;
    ask.price = hft::price_to_fixed(100.10);
    ask.quantity = hft::qty_to_fixed(1.0);
    ask.order_count = 1; ask.side = hft::Side::ASK;
    book.apply_update(ask);

    double spread_f = hft::fixed_to_price(book.spread());
    ASSERT_TRUE(spread_f > 0.099 && spread_f < 0.101);
    return true;
}
