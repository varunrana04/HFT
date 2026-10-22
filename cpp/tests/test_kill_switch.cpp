#include "../core/kill_switch.h"
#include <fstream>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace hft::core;

// Custom test macros are assumed to be provided by test_main.cpp
#include <iostream>
#include <cmath>
#include <functional>

struct TestRegistrar {
    TestRegistrar(const char* name, std::function<bool()> func);
};

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
extern std::string test_state_file;
std::string test_state_file = "test_kill_switch_state.txt";

static void cleanup_state() {
    std::remove(test_state_file.c_str());
}

static void write_corrupted_state() {
    std::ofstream outfile(test_state_file);
    outfile << "SOME_GARBAGE_DATA\n";
    outfile.close();
}

// Test 1: Position limit rejection (Long)
TEST(KillSwitchTest, RejectsExcessiveLongPosition) {
    cleanup_state();
    KillSwitch ks(100000.0, 0.05, 2.0, 5000, test_state_file);
    ks.update_state(1.5, 0.0, 0.0, 1000);
    
    // Requesting +1.0 would push position to 2.5 (violates max 2.0)
    ASSERT_FALSE(ks.check_order_allowed(1.0, 1000));
    
    // Requesting +0.5 is exactly 2.0 (allowed)
    ASSERT_TRUE(ks.check_order_allowed(0.5, 1000));
    cleanup_state();
    return true;
}

// Test 2: Short-side symmetry
TEST(KillSwitchTest, RejectsExcessiveShortPosition) {
    cleanup_state();
    KillSwitch ks(100000.0, 0.05, 2.0, 5000, test_state_file);
    ks.update_state(-1.5, 0.0, 0.0, 1000);
    
    // Requesting -1.0 would push position to -2.5 (violates max 2.0)
    ASSERT_FALSE(ks.check_order_allowed(-1.0, 1000));
    
    // Requesting -0.5 is exactly -2.0 (allowed)
    ASSERT_TRUE(ks.check_order_allowed(-0.5, 1000));
    cleanup_state();
    return true;
}

// Test 3: Exact-boundary behavior
TEST(KillSwitchTest, ExactBoundaryIsAllowed) {
    cleanup_state();
    KillSwitch ks(100000.0, 0.05, 2.0, 5000, test_state_file);
    ks.update_state(0.0, 0.0, 0.0, 1000);
    
    ASSERT_TRUE(ks.check_order_allowed(2.0, 1000));
    ASSERT_FALSE(ks.check_order_allowed(2.0001, 1000));
    cleanup_state();
    return true;
}

// Test 4: Circuit breaker trigger (Drawdown)
TEST(KillSwitchTest, TriggersOnDrawdown) {
    cleanup_state();
    KillSwitch ks(100000.0, 0.05, 2.0, 5000, test_state_file); // Max drawdown is -5000
    
    ks.update_state(1.0, -1000.0, -3000.0, 1000); // Total -4000 (Safe)
    ASSERT_FALSE(ks.is_trading_halted(1000));
    
    ks.update_state(1.0, -1000.0, -4500.0, 2000); // Total -5500 (Trip!)
    ASSERT_TRUE(ks.is_trading_halted(2000));
    ASSERT_TRUE(ks.should_flatten()); // Circuit breaker requires flattening
    cleanup_state();
    return true;
}

// Test 5: Post-kill call behavior
TEST(KillSwitchTest, RemainsKilledAfterProfitRecovery) {
    cleanup_state();
    KillSwitch ks(100000.0, 0.05, 2.0, 5000, test_state_file);
    
    ks.update_state(1.0, -6000.0, 0.0, 1000); // Trips
    ASSERT_TRUE(ks.is_trading_halted(1000));
    
    // PnL recovers, but switch must remain tripped
    ks.update_state(1.0, 10000.0, 0.0, 2000);
    ASSERT_TRUE(ks.is_trading_halted(2000));
    cleanup_state();
    return true;
}

// Test 6: Stale data fail-closed (Pull-based)
TEST(KillSwitchTest, HaltsOnStaleData) {
    cleanup_state();
    KillSwitch ks(100000.0, 0.05, 2.0, 5000, test_state_file);
    
    ks.update_state(0.0, 0.0, 0.0, 1000); // Last update at 1000
    
    ASSERT_FALSE(ks.is_trading_halted(2000)); // 1000ms passed (Safe)
    
    // 6000ms passed, > 5000ms timeout
    ASSERT_TRUE(ks.is_trading_halted(7000)); 
    ASSERT_FALSE(ks.should_flatten()); // Staleness shouldn't market panic
    
    // New data arrives, staleness resolves immediately
    ks.update_state(0.0, 0.0, 0.0, 7500);
    ASSERT_FALSE(ks.is_trading_halted(7500));
    cleanup_state();
    return true;
}

// Test 7: Persistence round-trip
TEST(KillSwitchTest, PersistenceRoundTrip) {
    cleanup_state();
    {
        KillSwitch ks(100000.0, 0.05, 2.0, 5000, test_state_file);
        ks.manual_trigger_kill("TEST_KILL", true, 1000);
    } // ks is destroyed, but state is on disk

    {
        // Fresh boot
        KillSwitch ks2(100000.0, 0.05, 2.0, 5000, test_state_file);
        
        // Feed it fresh current timestamp, should still be halted from disk
        ASSERT_TRUE(ks2.is_trading_halted(2000));
        ASSERT_TRUE(ks2.should_flatten());
    }
    cleanup_state();
    return true;
}

// Test 8: Cold boot staleness
TEST(KillSwitchTest, ColdBootIsStale) {
    cleanup_state();
    KillSwitch ks(100000.0, 0.05, 2.0, 5000, test_state_file);
    // Never called update_state
    ASSERT_TRUE(ks.is_trading_halted(1000));
    cleanup_state();
    return true;
}

// Test 9: Corrupted/truncated state file (Fail-safe)
TEST(KillSwitchTest, FailsSafeOnCorruptedState) {
    cleanup_state();
    write_corrupted_state();
    
    KillSwitch ks(100000.0, 0.05, 2.0, 5000, test_state_file);
    
    // Should boot up halted, but NOT request flattening
    ASSERT_TRUE(ks.is_trading_halted(1000));
    ASSERT_FALSE(ks.should_flatten());
    cleanup_state();
    return true;
}

// Test 10: Reset path
TEST(KillSwitchTest, ManualResetClearsState) {
    cleanup_state();
    KillSwitch ks(100000.0, 0.05, 2.0, 5000, test_state_file);
    ks.manual_trigger_kill("TEST", false, 1000);
    ASSERT_TRUE(ks.is_trading_halted(1000));
    
    ks.manual_reset("Operator cleared for trading", 2000);
    
    // Update state to prevent staleness trigger
    ks.update_state(0.0, 0.0, 0.0, 2001);
    
    ASSERT_FALSE(ks.is_trading_halted(2001));
    cleanup_state();
    return true;
}

// Test 11: Alert-failure isolation
TEST(KillSwitchTest, AlertFailureDoesNotBlockHalt) {
    cleanup_state();
    bool alert_fired = false;
    auto bad_alert = [&](const std::string& msg) {
        alert_fired = true;
        throw std::runtime_error("Webhook disconnected!");
    };

    KillSwitch ks(100000.0, 0.05, 2.0, 5000, test_state_file, bad_alert);
    
    ks.update_state(1.0, -10000.0, 0.0, 1000); // Massive drawdown
    
    // Exception was swallowed, state still updated
    ASSERT_TRUE(alert_fired);
    ASSERT_TRUE(ks.is_trading_halted(1000));
    cleanup_state();
    return true;
}
