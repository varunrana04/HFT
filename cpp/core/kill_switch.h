#pragma once

#include <string>
#include <mutex>
#include <atomic>
#include <functional>

namespace hft {
namespace core {

class KillSwitch {
public:
    // Define an alert callback type (best-effort, non-blocking)
    using AlertCallback = std::function<void(const std::string& message)>;

    // Constructor parameters:
    // allocated_capital: Starting capital to base % drawdown on
    // max_drawdown_pct: Positive percentage (e.g. 0.05 for 5%)
    // max_position_size: Max absolute position size
    // stale_timeout_ms: Milliseconds before fail-closing on staleness
    // state_file_path: Render Persistent Disk path (e.g. /data/.kill_switch_state)
    KillSwitch(double allocated_capital, 
               double max_drawdown_pct, 
               double max_position_size, 
               long stale_timeout_ms, 
               const std::string& state_file_path,
               AlertCallback alert_cb = nullptr);

    ~KillSwitch() = default;

    // Disallow copy/move
    KillSwitch(const KillSwitch&) = delete;
    KillSwitch& operator=(const KillSwitch&) = delete;

    // Update the PnL state. OrderManager owns the average-entry math and passes PnL here.
    // DOES NOT trigger staleness check internally.
    void update_state(double current_pos, double realized_pnl, double unrealized_pnl, long current_ts);

    // Checks if a requested delta size violates the max position. Evaluates staleness lazily.
    bool check_order_allowed(double requested_size_delta, long current_ts);

    // Evaluates the overall halt status lazily, checking staleness using current_ts.
    bool is_trading_halted(long current_ts);

    // A flag to indicate if OrderManager should aggressively flatten existing exposure.
    // Does NOT trigger on staleness (we don't panic-sell when we don't know the price).
    bool should_flatten() const;

    // An interface for OrderManager to call when the flattening is complete.
    void confirm_flattened(long current_ts);

    // Manual reset. Deletes the state file and logs the operator note.
    void manual_reset(const std::string& operator_note, long current_ts);

    // Allow manual/external triggering of the kill switch
    void manual_trigger_kill(const std::string& reason, bool require_flattening, long current_ts);

private:
    void trigger_kill(const std::string& reason, bool require_flattening, long current_ts);
    bool check_staleness(long current_ts);
    void load_persisted_state();
    void persist_state_atomic(const std::string& reason, long current_ts);

    std::mutex state_mutex_;
    
    double allocated_capital_;
    double max_drawdown_pct_;
    double max_position_size_;
    long stale_timeout_ms_;
    std::string state_file_path_;
    AlertCallback alert_cb_;

    std::atomic<double> current_position_{0.0};
    std::atomic<long> last_update_ts_{0}; // 0 = never updated (staleness triggers immediately)

    // Halt State
    std::atomic<bool> is_killed_{false};
    std::atomic<bool> should_flatten_{false};
    std::string trip_reason_ = "";
};

} // namespace core
} // namespace hft
