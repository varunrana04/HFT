#include "fix_engine.h"
#include <iostream>

namespace hft {
namespace net {

struct FixEngineState {
    bool is_started = false;
    // In production, this holds a QuickFIX::SocketInitiator and QuickFIX::Application
};

FixEngine::FixEngine(const std::string& config_file) {
    state_ = std::make_unique<FixEngineState>();
    std::cout << "[FIX-ENGINE] Initialized with config: " << config_file << "\n";
}

FixEngine::~FixEngine() {
    stop();
}

void FixEngine::start() {
    if (state_ && !state_->is_started) {
        state_->is_started = true;
        std::cout << "[FIX-ENGINE] Starting QuickFIX/C++ initiator stub...\n";
    }
}

void FixEngine::stop() {
    if (state_ && state_->is_started) {
        state_->is_started = false;
        std::cout << "[FIX-ENGINE] Stopping QuickFIX/C++ initiator.\n";
    }
}

bool FixEngine::send_order(const hft::Order& order) {
    if (!state_ || !state_->is_started) return false;
    
    // Simulate formatting a FIX 4.4 NewOrderSingle (35=D)
    std::cout << "[FIX-ENGINE] OUT -> 35=D | ClOrdID=" << order.order_id 
              << " | Side=" << (order.side == Side::BID ? "1" : "2")
              << " | Price=" << order.price 
              << " | OrderQty=" << order.quantity << "\n";
              
    return true;
}

bool FixEngine::cancel_order(uint64_t order_id) {
    if (!state_ || !state_->is_started) return false;
    
    // Simulate formatting a FIX 4.4 OrderCancelRequest (35=F)
    std::cout << "[FIX-ENGINE] OUT -> 35=F | OrigClOrdID=" << order_id << "\n";
    
    return true;
}

} // namespace net
} // namespace hft
