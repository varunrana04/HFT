#pragma once

#include "inetwork.h"
#include <string>
#include <atomic>
#include <functional>
#include <thread>
#include <memory>
#include <simdjson/simdjson.h>

// Forward declare StrategyEngine
namespace hft { class StrategyEngine; }

namespace hft {
namespace gateway {

class BybitWs : public net::INetworkRx {
public:
    BybitWs(const std::string& symbol = "BTCUSDT");
    ~BybitWs() override;

    bool initialize() override;
    
    void poll_loop(
        std::function<void(const Trade&)> on_trade_callback,
        std::function<void(const BookSnapshot&)> on_book_callback
    ) override;

    void start_live_feed(hft::StrategyEngine* engine);

    void stop() override;

private:
    std::string symbol_;
    std::atomic<bool> running_{false};
    
    std::function<void(const Trade&)> on_trade_callback_;
    std::function<void(const BookSnapshot&)> on_book_callback_;

    hft::StrategyEngine* engine_{nullptr};
    
    BookSnapshot latest_book_{};

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gateway
} // namespace hft
