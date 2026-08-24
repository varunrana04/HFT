#pragma once

#include "inetwork.h"
#include <string>
#include <atomic>
#include <functional>
#include <thread>
#include <memory>

// Forward declare StrategyEngine
namespace hft { class StrategyEngine; }

namespace hft {
namespace gateway {

class BinanceWs : public net::INetworkRx {
public:
    BinanceWs(const std::string& symbol = "btcusdt");
    ~BinanceWs() override;

    bool initialize() override;
    
    // Legacy poll_loop interface
    void poll_loop(
        std::function<void(const Trade&)> on_trade_callback,
        std::function<void(const BookSnapshot&)> on_book_callback
    ) override;

    // Direct engine integration
    void start_live_feed(hft::StrategyEngine* engine);

    void stop() override;

private:
    std::string symbol_;
    std::atomic<bool> running_{false};
    
    // Callbacks
    std::function<void(const Trade&)> on_trade_callback_;
    std::function<void(const BookSnapshot&)> on_book_callback_;

    // Target Engine
    hft::StrategyEngine* engine_{nullptr};
    
    // Latest book state
    BookSnapshot latest_book_{};

    // PIMPL to avoid exposing IXWebSocket/simdjson to the rest of the app
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gateway
} // namespace hft
