#include "kill_switch.h"
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>

using namespace hft::core;

void run_tests() {
    std::string state_file = "test_kill_switch_state.tmp";
    std::remove(state_file.c_str()); // Ensure clean slate

    std::vector<std::string> alerts;
    auto alert_cb = [&](const std::string& msg) {
        alerts.push_back(msg);
        std::cout << "[ALERT CALLBACK] " << msg << std::endl;
    };

    double capital = 100000.0;
    double max_drawdown = 0.05; // 5% = 5000 max loss
    double max_pos = 10.0;
    long timeout_ms = 5000;

    std::cout << "--- Starting KillSwitch Tests ---" << std::endl;

    {
        // Test 1: Normal operations
        KillSwitch ks(capital, max_drawdown, max_pos, timeout_ms, state_file, alert_cb);
        long ts = 1000;
        
        // Cold boot: staleness triggers immediately if not updated
        assert(ks.is_trading_halted(ts) == true);

        // Update state
        ks.update_state(0.0, 0.0, 0.0, ts);
        assert(ks.is_trading_halted(ts) == false);

        // Check order limits
        assert(ks.check_order_allowed(5.0, ts) == true);
        assert(ks.check_order_allowed(11.0, ts) == false); // Exceeds max_pos
        assert(ks.check_order_allowed(-11.0, ts) == false); // Exceeds max_pos short

        // PnL updates
        ks.update_state(5.0, 1000.0, 2000.0, ts + 10);
        assert(ks.is_trading_halted(ts + 10) == false);
        assert(alerts.empty());
    }

    {
        // Test 2: Drawdown Violation
        KillSwitch ks(capital, max_drawdown, max_pos, timeout_ms, state_file, alert_cb);
        long ts = 2000;
        ks.update_state(5.0, 0.0, 0.0, ts);

        // -5001 total PnL exceeds 5000 max loss
        ks.update_state(5.0, -3000.0, -2001.0, ts + 10);
        
        assert(ks.is_trading_halted(ts + 10) == true);
        assert(ks.should_flatten() == true);
        assert(alerts.size() == 1);
        assert(alerts.back().find("Daily Drawdown Exceeded") != std::string::npos);
        
        // Even if PnL recovers, we remain halted
        ks.update_state(5.0, 1000.0, 1000.0, ts + 20);
        assert(ks.is_trading_halted(ts + 20) == true);
    }

    {
        // Test 3: Persistence (Loading state after crash)
        alerts.clear();
        KillSwitch ks(capital, max_drawdown, max_pos, timeout_ms, state_file, alert_cb);
        long ts = 3000;
        
        // Should boot up halted because of the state file written in Test 2
        assert(ks.is_trading_halted(ts) == true);
        assert(ks.should_flatten() == true);
        
        // Manual reset clears it
        ks.manual_reset("Operator Override", ts);
        assert(ks.is_trading_halted(ts) == false);
        assert(ks.should_flatten() == false);
        
        // Staleness timeout
        assert(ks.is_trading_halted(ts + timeout_ms + 1) == true); // Timeout triggered
        assert(ks.should_flatten() == false); // Staleness doesn't trigger flattening
    }

    std::remove(state_file.c_str()); // Cleanup
    std::cout << "--- All Tests Passed! ---" << std::endl;
}

int main() {
    run_tests();
    return 0;
}
