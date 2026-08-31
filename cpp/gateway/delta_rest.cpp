#include "delta_rest.h"
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <vector>

// OpenSSL for HMAC-SHA256
#include <openssl/hmac.h>
#include <openssl/sha.h>

// IXWebSocket HTTP Client
#include <ixwebsocket/IXHttpClient.h>

namespace hft {
namespace gateway {

struct DeltaRest::Impl {
    ix::HttpClient httpClient;
    
    Impl() {
        // HttpClient initialization if needed
    }
};

DeltaRest::DeltaRest(const std::string& api_key, const std::string& api_secret) 
    : api_key_(api_key), api_secret_(api_secret), impl_(std::make_unique<Impl>()) {
    const char* env_url = std::getenv("DELTA_BASE_URL");
    base_url_ = env_url ? env_url : "https://api.delta.exchange";
    
    const char* env_pid = std::getenv("DELTA_PRODUCT_ID");
    if (env_pid) {
        product_id_ = std::stoi(env_pid);
    }
    std::cout << "[DeltaRest] REST base URL: " << base_url_ << " | Product ID: " << product_id_ << "\n";
}

DeltaRest::~DeltaRest() = default;

std::string DeltaRest::generate_signature(const std::string& method, const std::string& endpoint, const std::string& payload, const std::string& timestamp) const {
    std::string data = method + timestamp + endpoint + payload;
    
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len;
    
    HMAC(EVP_sha256(), api_secret_.c_str(), static_cast<int>(api_secret_.length()),
         reinterpret_cast<const unsigned char*>(data.c_str()), static_cast<int>(data.length()),
         hash, &hash_len);
         
    std::stringstream ss;
    for (unsigned int i = 0; i < hash_len; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    
    return ss.str();
}

bool DeltaRest::submit_order(const Order& order, const std::string& symbol) {
    (void)symbol;
    if (api_key_.empty() || api_secret_.empty()) {
        std::cerr << "[DeltaRest] Missing API keys. Cannot submit order." << std::endl;
        return false;
    }

    std::string endpoint = "/v2/orders";
    std::string url = base_url_ + endpoint;
    
    // Construct JSON payload
    std::string side_str = (order.side == Side::BID) ? "buy" : "sell";
    
    // Convert generic HFT fixed precision back to strings for Delta
    double price = fixed_to_price(order.price);
    double qty = fixed_to_qty(order.quantity);

    std::stringstream ss;
    ss << "{\"product_id\":" << product_id_ << ",\"order_type\":\"limit_order\",\"size\":" << (int)qty 
       << ",\"side\":\"" << side_str << "\",\"limit_price\":\"" << std::fixed << std::setprecision(1) << price 
       << "\",\"time_in_force\":\"gtc\"}";
    
    std::string payload = ss.str();
    
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count());
    
    std::string signature = generate_signature("POST", endpoint, payload, timestamp);

    ix::HttpRequestArgsPtr args = impl_->httpClient.createRequest();
    args->extraHeaders["api-key"] = api_key_;
    args->extraHeaders["signature"] = signature;
    args->extraHeaders["timestamp"] = timestamp;
    args->extraHeaders["Content-Type"] = "application/json";

    auto response = impl_->httpClient.post(url, payload, args);
    
    if (response->statusCode == 200 || response->statusCode == 201) {
        std::cout << "[DeltaRest] Order submitted successfully: " << response->body << std::endl;
        return true;
    } else {
        std::cerr << "[DeltaRest] Order submission failed! Status: " << response->statusCode << " Body: " << response->body << std::endl;
        return false;
    }
}

bool DeltaRest::cancel_order(const std::string& order_id, const std::string& /*symbol*/) {
    if (api_key_.empty() || api_secret_.empty()) {
        std::cerr << "[DeltaRest] Missing API keys. Cannot cancel order." << std::endl;
        return false;
    }

    std::string endpoint = "/v2/orders/" + order_id;
    std::string url = base_url_ + endpoint;

    auto now = std::chrono::system_clock::now();
    std::string timestamp = std::to_string(
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count());

    // Delta DELETE /v2/orders/{id} — no request body
    std::string payload = "";
    std::string signature = generate_signature("DELETE", endpoint, payload, timestamp);

    ix::HttpRequestArgsPtr args = impl_->httpClient.createRequest();
    args->extraHeaders["api-key"]      = api_key_;
    args->extraHeaders["signature"]    = signature;
    args->extraHeaders["timestamp"]    = timestamp;
    args->extraHeaders["Content-Type"] = "application/json";

    auto response = impl_->httpClient.Delete(url, args);

    if (response->statusCode == 200 || response->statusCode == 204) {
        std::cout << "[DeltaRest] Order " << order_id << " cancelled successfully." << std::endl;
        return true;
    } else {
        std::cerr << "[DeltaRest] Cancel failed! Status: " << response->statusCode
                  << " Body: " << response->body << std::endl;
        return false;
    }
}

} // namespace gateway
} // namespace hft
