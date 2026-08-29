#pragma once

#include "types.h"
#include <string>
#include <memory>
#include <future>

namespace hft {
namespace gateway {

class DeltaRest {
public:
    DeltaRest(const std::string& api_key, const std::string& api_secret);
    ~DeltaRest();

    /**
     * Submit an order via POST /v2/orders
     */
    bool submit_order(const Order& order, const std::string& symbol = "BTCUSD");

    /**
     * Cancel an order via DELETE /v2/orders/{id}
     */
    bool cancel_order(const std::string& order_id, const std::string& symbol = "BTCUSD");

private:
    std::string api_key_;
    std::string api_secret_;
    std::string base_url_;

    std::string generate_signature(const std::string& method, const std::string& endpoint, const std::string& payload, const std::string& timestamp) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gateway
} // namespace hft
