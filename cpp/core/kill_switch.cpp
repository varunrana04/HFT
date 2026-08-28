#include "kill_switch.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <system_error>

namespace hft {
namespace core {

KillSwitch::KillSwitch(double allocated_capital, 
                       double max_drawdown_pct, 
                       double max_position_size, 
                       long stale_timeout_ms, 
                       const std::string& state_file_path,
                       AlertCallback alert_cb)
    : allocated_capital_(allocated_capital),
      max_drawdown_pct_(max_drawdown_pct),
      max_position_size_(max_position_size),
      stale_timeout_ms_(stale_timeout_ms),
      state_file_path_(state_file_path),
      alert_cb_(std::move(alert_cb)) 
{
    load_persisted_state();
}

void KillSwitch::update_state(double current_pos, double realized_pnl, double unrealized_pnl, long current_ts) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    current_position_.store(current_pos);
    last_update_ts_.store(current_ts);

    if (is_killed_.load()) {
        return; // Already halted
    }

    double total_pnl = realized_pnl + unrealized_pnl;
    double max_allowed_drawdown = -(allocated_capital_ * max_drawdown_pct_);

    if (total_pnl <= max_allowed_drawdown) {
        trigger_kill("Daily Drawdown Exceeded", true, current_ts);
    }
}

bool KillSwitch::check_order_allowed(double requested_size_delta, long current_ts) {
    if (is_trading_halted(current_ts)) {
        return false;
    }

    // Short-side symmetry checked with abs()
    double new_position = current_position_.load() + requested_size_delta;
    if (std::abs(new_position) > max_position_size_) {
        return false;
    }

    return true;
}

bool KillSwitch::is_trading_halted(long current_ts) {
    if (is_killed_.load()) {
        return true;
    }
    return check_staleness(current_ts);
}

bool KillSwitch::should_flatten() const {
    return should_flatten_.load();
}

void KillSwitch::confirm_flattened([[maybe_unused]] long current_ts) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    // Log flattening confirmed
    should_flatten_.store(false);
}

void KillSwitch::manual_reset(const std::string& operator_note, long current_ts) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Explicitly delete the state file
    if (std::remove(state_file_path_.c_str()) != 0) {
        // If file doesn't exist, it's fine. If there's a permission error, we should log it.
        // For now, assume removal is successful or file wasn't there.
    }

    is_killed_.store(false);
    should_flatten_.store(false);
    trip_reason_ = "";
    last_update_ts_.store(current_ts); // Reset staleness timer immediately

    std::cout << "[KILL_SWITCH] Manual Reset by Operator: " << operator_note << std::endl;
}

void KillSwitch::manual_trigger_kill(const std::string& reason, bool require_flattening, long current_ts) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!is_killed_.load()) {
        trigger_kill(reason, require_flattening, current_ts);
    }
}

// Private methods

void KillSwitch::trigger_kill(const std::string& reason, bool require_flattening, long current_ts) {
    is_killed_.store(true);
    if (require_flattening) {
        should_flatten_.store(true);
    }
    trip_reason_ = reason;

    // 1. Synchronous state persistence
    persist_state_atomic(reason, current_ts);

    // 2. Asynchronous / Best-effort alerting (Decoupled from halt)
    if (alert_cb_) {
        std::string alert_msg = "[CRITICAL] Kill Switch Triggered! Reason: " + reason;
        try {
            alert_cb_(alert_msg);
        } catch (...) {
            // Alert failure must not block the halt
            std::cerr << "Failed to send alert webhook, but system is safely halted." << std::endl;
        }
    }
}

bool KillSwitch::check_staleness(long current_ts) {
    long last_ts = last_update_ts_.load();
    if (last_ts == 0) {
        return true; // Cold boot, never updated
    }
    if ((current_ts - last_ts) > stale_timeout_ms_) {
        // Note: Staleness halts trading, but does NOT set should_flatten().
        // We cannot market-panic when we don't have accurate pricing.
        return true; 
    }
    return false;
}

void KillSwitch::load_persisted_state() {
    std::ifstream infile(state_file_path_);
    if (!infile.good()) {
        return; // File doesn't exist, start fresh
    }

    std::string line;
    if (std::getline(infile, line)) {
        if (line == "KILLED") {
            is_killed_.store(true);
            std::getline(infile, trip_reason_);
            
            std::string flatten_line;
            if (std::getline(infile, flatten_line)) {
                if (flatten_line == "FLATTEN_REQUIRED") {
                    should_flatten_.store(true);
                }
            }
            return;
        } else {
            // File exists but is garbage/truncated.
            // Contract: Fail safe. Default to halted with critical alert.
            is_killed_.store(true);
            trip_reason_ = "CORRUPTED_STATE_FILE";
            should_flatten_.store(false); // Don't flatten on corrupt data
            if (alert_cb_) {
                try {
                    alert_cb_("[CRITICAL] Kill Switch State File Corrupted. Booting in HALTED state.");
                } catch(...) {}
            }
            return;
        }
    }
    
    // Empty file? Treat as corrupted.
    is_killed_.store(true);
    trip_reason_ = "EMPTY_STATE_FILE";
    should_flatten_.store(false);
}

void KillSwitch::persist_state_atomic(const std::string& reason, [[maybe_unused]] long current_ts) {
    std::string temp_file = state_file_path_ + ".tmp";
    std::ofstream outfile(temp_file);
    if (outfile.is_open()) {
        outfile << "KILLED\n";
        outfile << reason << "\n";
        if (should_flatten_.load()) {
            outfile << "FLATTEN_REQUIRED\n";
        }
        outfile.flush();
        outfile.close();

        // Atomic rename (POSIX standard, std::rename works well here)
        if (std::rename(temp_file.c_str(), state_file_path_.c_str()) != 0) {
            std::cerr << "Failed to rename temp state file!" << std::endl;
        }
    } else {
        std::cerr << "Failed to open temp state file for writing!" << std::endl;
    }
}

} // namespace core
} // namespace hft
