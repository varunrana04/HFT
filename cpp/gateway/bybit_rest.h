#pragma once

#include "types.h"
#include <string>
#include <memory>
#include <future>

namespace hft {
namespace gateway {

class BybitRest {
public:
    BybitRest(const std::string& api_key, const std::string& api_secret);
    ~BybitRest();

    /**
     * Submit an order via POST /v5/order/create
     */
    bool submit_order(const Order& order, const std::string& symbol = "BTCUSDT");

    /**
     * Cancel an order via POST /v5/order/cancel
     */
    bool cancel_order(const std::string& order_id, const std::string& symbol = "BTCUSDT");

    /**
     * Cancel all open orders via POST /v5/order/cancel-all
     */
    bool cancel_all_orders(const std::string& symbol = "BTCUSDT");

private:
    std::string api_key_;
    std::string api_secret_;
    std::string base_url_;

    std::string generate_signature(const std::string& timestamp, const std::string& payload, const std::string& recv_window = "5000") const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gateway
} // namespace hft
