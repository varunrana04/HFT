/**
 * @file test_spsc_queue.cpp
 * @brief Tests for the lock-free SPSC ring buffer.
 */
#include "spsc_queue.h"
#include <iostream>
#include <vector>
#include <string>
#include <functional>

struct TestCase;
extern std::vector<TestCase>& get_tests();

#define ASSERT_TRUE(expr) do { if (!(expr)) { std::cerr << "  FAIL: " << #expr << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { std::cerr << "  FAIL: " << #a << " == " << #b << " (" << (a) << " != " << (b) << ") [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)
#define ASSERT_GT(a, b) do { if (!((a) > (b))) { std::cerr << "  FAIL: " << #a << " > " << #b << " [" << __FILE__ << ":" << __LINE__ << "]" << std::endl; return false; } } while(0)

#define TEST(suite, name)                                         \
    static bool test_##suite##_##name();                          \
    static TestRegistrar reg_##suite##_##name(                    \
        #suite "::" #name, test_##suite##_##name);                \
    static bool test_##suite##_##name()

TEST(SPSCQueue, EmptyOnConstruction) {
    hft::SPSCQueue<int, 16> q;
    ASSERT_TRUE(q.empty());
    ASSERT_FALSE(q.full());
    ASSERT_EQ(q.size_approx(), size_t{0});
    return true;
}

TEST(SPSCQueue, PushPop_SingleItem) {
    hft::SPSCQueue<int64_t, 16> q;
    ASSERT_TRUE(q.try_push(42));
    ASSERT_FALSE(q.empty());

    int64_t val = 0;
    ASSERT_TRUE(q.try_pop(val));
    ASSERT_EQ(val, 42);
    ASSERT_TRUE(q.empty());
    return true;
}

TEST(SPSCQueue, PushPop_MultipleItems) {
    hft::SPSCQueue<int, 16> q;
    for (int i = 0; i < 15; ++i) {  // Capacity - 1
        ASSERT_TRUE(q.try_push(i * 10));
    }
    ASSERT_EQ(q.capacity(), size_t{15});

    for (int i = 0; i < 15; ++i) {
        int val = -1;
        ASSERT_TRUE(q.try_pop(val));
        ASSERT_EQ(val, i * 10);
    }
    ASSERT_TRUE(q.empty());
    return true;
}

TEST(SPSCQueue, Full_RejectsPush) {
    hft::SPSCQueue<int, 4> q;  // Capacity = 3 (power of 2 minus 1)
    ASSERT_TRUE(q.try_push(1));
    ASSERT_TRUE(q.try_push(2));
    ASSERT_TRUE(q.try_push(3));
    ASSERT_FALSE(q.try_push(4));  // Should fail — full
    return true;
}

TEST(SPSCQueue, Empty_RejectsPop) {
    hft::SPSCQueue<int, 4> q;
    int val = -1;
    ASSERT_FALSE(q.try_pop(val));  // Should fail — empty
    ASSERT_EQ(val, -1);  // val unchanged
    return true;
}

TEST(SPSCQueue, Peek_DoesNotRemove) {
    hft::SPSCQueue<int, 16> q;
    q.try_push(99);
    int val = 0;
    ASSERT_TRUE(q.try_peek(val));
    ASSERT_EQ(val, 99);
    ASSERT_FALSE(q.empty());  // Item still there
    return true;
}

TEST(SPSCQueue, WrapAround) {
    hft::SPSCQueue<int, 4> q;
    // Push 3, pop 3, push 3, pop 3 — tests wrap-around
    for (int round = 0; round < 5; ++round) {
        for (int i = 0; i < 3; ++i) {
            ASSERT_TRUE(q.try_push(round * 10 + i));
        }
        for (int i = 0; i < 3; ++i) {
            int val = -1;
            ASSERT_TRUE(q.try_pop(val));
            ASSERT_EQ(val, round * 10 + i);
        }
    }
    return true;
}

TEST(SPSCQueue, TradeStruct) {
    // Test with our actual Trade struct
    hft::SPSCQueue<hft::Trade, 64> q;
    hft::Trade t{};
    t.timestamp_ns = 1000;
    t.price = hft::price_to_fixed(50000.0);
    t.quantity = hft::qty_to_fixed(0.1);
    t.side = hft::Side::BID;

    ASSERT_TRUE(q.try_push(t));

    hft::Trade out{};
    ASSERT_TRUE(q.try_pop(out));
    ASSERT_EQ(out.timestamp_ns, 1000);
    ASSERT_EQ(out.price, t.price);
    ASSERT_EQ(out.side, hft::Side::BID);
    return true;
}
