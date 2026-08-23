/**
 * @file test_memory_pool.cpp
 * @brief Tests for the pre-allocated memory pool.
 */
#include "memory_pool.h"
#include "types.h"
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

#define TEST(suite, name) \
    [[maybe_unused]] static bool test_##suite##_##name(); \
    [[maybe_unused]] static TestRegistrar reg_##suite##_##name(#suite "::" #name, test_##suite##_##name); \
    static bool test_##suite##_##name()

TEST(MemoryPool, InitialState) {
    hft::MemoryPool<hft::Trade, 64> pool;
    ASSERT_TRUE(pool.is_empty());
    ASSERT_FALSE(pool.is_full());
    ASSERT_EQ(pool.allocated_count(), size_t{0});
    ASSERT_EQ(pool.available_count(), size_t{64});
    ASSERT_EQ(pool.capacity(), size_t{64});
    return true;
}

TEST(MemoryPool, AllocateSingle) {
    hft::MemoryPool<hft::Trade, 64> pool;
    hft::Trade* t = pool.allocate();
    ASSERT_TRUE(t != nullptr);
    ASSERT_EQ(pool.allocated_count(), size_t{1});
    ASSERT_EQ(pool.available_count(), size_t{63});

    t->price = hft::price_to_fixed(50000.0);
    t->side = hft::Side::BID;
    ASSERT_EQ(t->price, hft::price_to_fixed(50000.0));
    return true;
}

TEST(MemoryPool, AllocateAll) {
    hft::MemoryPool<hft::PriceLevel, 8> pool;
    hft::PriceLevel* ptrs[8];
    for (size_t i = 0; i < 8; ++i) {
        ptrs[i] = pool.allocate();
        ASSERT_TRUE(ptrs[i] != nullptr);
    }
    ASSERT_TRUE(pool.is_full());
    ASSERT_TRUE(pool.allocate() == nullptr);  // Should fail
    return true;
}

TEST(MemoryPool, DeallocateAndReuse) {
    hft::MemoryPool<hft::PriceLevel, 4> pool;
    hft::PriceLevel* a = pool.allocate();
    hft::PriceLevel* b = pool.allocate();
    ASSERT_EQ(pool.allocated_count(), size_t{2});

    pool.deallocate(a);
    ASSERT_EQ(pool.allocated_count(), size_t{1});

    // Should be able to allocate again
    hft::PriceLevel* c = pool.allocate();
    ASSERT_TRUE(c != nullptr);
    ASSERT_EQ(pool.allocated_count(), size_t{2});
    return true;
}

TEST(MemoryPool, Reset) {
    hft::MemoryPool<hft::Trade, 16> pool;
    for (size_t i = 0; i < 10; ++i) {
        hft::Trade* t = pool.allocate();
        ASSERT_TRUE(t != nullptr);
    }
    ASSERT_EQ(pool.allocated_count(), size_t{10});

    pool.reset();
    ASSERT_TRUE(pool.is_empty());
    ASSERT_EQ(pool.allocated_count(), size_t{0});
    ASSERT_EQ(pool.available_count(), size_t{16});
    return true;
}
