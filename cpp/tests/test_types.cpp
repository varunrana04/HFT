/**
 * @file test_types.cpp
 * @brief Tests for core types, fixed-point conversion, and struct layouts.
 */

#include "types.h"
#include "market_data.h"

// Include test macros from test_main.cpp (they're in the same TU via linker)
extern struct TestRegistrar;

#include <iostream>
#include <vector>
#include <string>
#include <functional>

// Redefine test macros for this TU
struct TestCase2 { std::string name; std::function<bool()> func; };
extern std::vector<TestCase>& get_tests();

// Re-include assertion macros
#define ASSERT_TRUE(expr) do { if (!(expr)) { std::cerr << "  FAIL: " << #expr << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { std::cerr << "  FAIL: " << #a << " == " << #b << " (" << (a) << " != " << (b) << ") [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_NE(a, b) do { if ((a) == (b)) { std::cerr << "  FAIL: " << #a << " != " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_GT(a, b) do { if (!((a) > (b))) { std::cerr << "  FAIL: " << #a << " > " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_LT(a, b) do { if (!((a) < (b))) { std::cerr << "  FAIL: " << #a << " < " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)

#define TEST(suite, name)                                         \
    static bool test_##suite##_##name();                          \
    static TestRegistrar reg_##suite##_##name(                    \
        #suite "::" #name, test_##suite##_##name);                \
    static bool test_##suite##_##name()

// ─── Fixed-Point Conversion Tests ────────────────────────────

TEST(Types, PriceToFixed_WholeNumber) {
    int64_t fixed = hft::price_to_fixed(50000.0);
    ASSERT_EQ(fixed, 50000LL * hft::PRICE_SCALE);
    return true;
}

TEST(Types, PriceToFixed_Decimal) {
    int64_t fixed = hft::price_to_fixed(50000.12345678);
    double back = hft::fixed_to_price(fixed);
    // Should be accurate to 8 decimal places
    double diff = back - 50000.12345678;
    ASSERT_TRUE(diff < 0.000000015 && diff > -0.000000015);
    return true;
}

TEST(Types, PriceRoundTrip) {
    double prices[] = {0.00000001, 0.001, 1.0, 100.5, 50000.0, 99999.99999999};
    for (double p : prices) {
        int64_t fixed = hft::price_to_fixed(p);
        double back = hft::fixed_to_price(fixed);
        double diff = back - p;
        ASSERT_TRUE(diff < 0.00000002 && diff > -0.00000002);
    }
    return true;
}

TEST(Types, PriceLevelIsValid) {
    hft::PriceLevel valid{hft::price_to_fixed(100.0), hft::qty_to_fixed(1.0), 1, 0};
    ASSERT_TRUE(valid.is_valid());

    hft::PriceLevel invalid_price{hft::INVALID_PRICE, hft::qty_to_fixed(1.0), 1, 0};
    ASSERT_FALSE(invalid_price.is_valid());

    hft::PriceLevel zero_qty{hft::price_to_fixed(100.0), 0, 1, 0};
    ASSERT_FALSE(zero_qty.is_valid());
    return true;
}

TEST(Types, BookSnapshotMidPrice) {
    hft::BookSnapshot book{};
    book.best_bid_price = hft::price_to_fixed(100.0);
    book.best_ask_price = hft::price_to_fixed(101.0);
    int64_t mid = book.mid_price();
    double mid_f = hft::fixed_to_price(mid);
    ASSERT_TRUE(mid_f > 100.49 && mid_f < 100.51);
    return true;
}

TEST(Types, BookSnapshotSpread) {
    hft::BookSnapshot book{};
    book.best_bid_price = hft::price_to_fixed(100.0);
    book.best_ask_price = hft::price_to_fixed(100.05);
    int64_t spread = book.spread();
    double spread_f = hft::fixed_to_price(spread);
    ASSERT_TRUE(spread_f > 0.049 && spread_f < 0.051);
    return true;
}

TEST(Types, OrderSlippage_Buy) {
    hft::Order order{};
    order.side = hft::Side::BID;
    order.expected_price = hft::price_to_fixed(100.0);
    order.avg_fill_price = hft::price_to_fixed(100.05);
    order.filled_quantity = hft::qty_to_fixed(1.0);
    // For buy: slippage = fill - expected (positive = unfavorable)
    int64_t slip = order.slippage();
    ASSERT_GT(slip, 0);  // Paid more than expected
    return true;
}

TEST(Types, OrderSlippage_Sell) {
    hft::Order order{};
    order.side = hft::Side::ASK;
    order.expected_price = hft::price_to_fixed(100.0);
    order.avg_fill_price = hft::price_to_fixed(99.95);
    order.filled_quantity = hft::qty_to_fixed(1.0);
    // For sell: slippage = expected - fill (positive = unfavorable)
    int64_t slip = order.slippage();
    ASSERT_GT(slip, 0);  // Received less than expected
    return true;
}

// ─── Fast Parser Tests ───────────────────────────────────────

TEST(Parser, FastParseInt_Normal) {
    ASSERT_EQ(hft::fast_parse_int("12345"), 12345);
    ASSERT_EQ(hft::fast_parse_int("0"), 0);
    ASSERT_EQ(hft::fast_parse_int("1704067200000"), 1704067200000LL);
    return true;
}

TEST(Parser, FastParseFixed_Decimal) {
    int64_t val = hft::fast_parse_fixed("50000.12345678");
    double back = hft::fixed_to_price(val);
    double diff = back - 50000.12345678;
    ASSERT_TRUE(diff < 0.000000015 && diff > -0.000000015);
    return true;
}

TEST(Parser, FastParseFixed_WholeNumber) {
    int64_t val = hft::fast_parse_fixed("50000");
    ASSERT_EQ(val, 50000LL * hft::PRICE_SCALE);
    return true;
}

TEST(Parser, FastParseFixed_SmallDecimal) {
    int64_t val = hft::fast_parse_fixed("0.00000001");
    ASSERT_EQ(val, 1);  // 1 unit of fixed-point
    return true;
}

TEST(Parser, BinanceAggTrade_ValidLine) {
    std::string line = "123456,50000.12,0.001,100,105,1704067200000,false,true";
    hft::Trade trade{};
    bool ok = hft::parse_binance_agg_trade(line, trade);
    ASSERT_TRUE(ok);
    ASSERT_EQ(trade.sequence_num, 123456);
    ASSERT_EQ(trade.side, hft::Side::BID);  // is_buyer_maker=false → buyer aggressor
    ASSERT_GT(trade.price, 0);
    ASSERT_GT(trade.quantity, 0);
    ASSERT_EQ(trade.timestamp_ns, 1704067200000LL * 1000000LL);
    return true;
}

TEST(Parser, BinanceAggTrade_SellerAggressor) {
    std::string line = "789,50001.0,0.5,200,210,1704067200100,true,true";
    hft::Trade trade{};
    bool ok = hft::parse_binance_agg_trade(line, trade);
    ASSERT_TRUE(ok);
    ASSERT_EQ(trade.side, hft::Side::ASK);  // is_buyer_maker=true → seller aggressor
    return true;
}

TEST(Parser, BinanceAggTrade_HeaderLine) {
    std::string header = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match";
    hft::Trade trade{};
    bool ok = hft::parse_binance_agg_trade(header, trade);
    ASSERT_FALSE(ok);  // Should reject header lines
    return true;
}
