#pragma once

#include <string>
#include <memory>
#include <cstdint>
#include "../core/types.h"

namespace hft {
namespace net {

// Forward declare QuickFIX application state
struct FixEngineState;

/**
 * @brief FIX Protocol Engine for institutional routing (Phase 3).
 * 
 * In production, this integrates with QuickFIX/C++ to send NewOrderSingle
 * (35=D) messages and receive ExecutionReports (35=8) directly from the
 * exchange matching engine (e.g. CME, Nasdaq, Binance VIP).
 */
class FixEngine {
public:
    explicit FixEngine(const std::string& config_file);
    ~FixEngine();

    // Start the FIX initiator
    void start();

    // Stop the FIX initiator
    void stop();

    // Send a NewOrderSingle message
    bool send_order(const hft::Order& order);

    // Cancel an existing order
    bool cancel_order(uint64_t order_id);

private:
    std::unique_ptr<FixEngineState> state_;
};

} // namespace net
} // namespace hft
