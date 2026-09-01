#include "bybit_rest.h"
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

struct BybitRest::Impl {
    ix::HttpClient httpClient;
    
    Impl() {
        // HttpClient initialization if needed
    }
};

BybitRest::BybitRest(const std::string& api_key, const std::string& api_secret) 
    : api_key_(api_key), api_secret_(api_secret), impl_(std::make_unique<Impl>()) {
    const char* env_url = std::getenv("BYBIT_BASE_URL");
    base_url_ = env_url ? env_url : "https://api-testnet.bybit.com";
    
    std::cout << "[BybitRest] REST base URL: " << base_url_ << "\n";
}

BybitRest::~BybitRest() = default;

std::string BybitRest::generate_signature(const std::string& timestamp, const std::string& payload, const std::string& recv_window) const {
    std::string data = timestamp + api_key_ + recv_window + payload;
    
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

bool BybitRest::submit_order(const Order& order, const std::string& symbol) {
    if (api_key_.empty() || api_secret_.empty()) {
        std::cerr << "[BybitRest] Missing API keys. Cannot submit order." << std::endl;
        return false;
    }

    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch()).count();
    std::string timestamp = std::to_string(now);

    std::string side_str = (order.side == Side::BID) ? "Buy" : "Sell";
    
    double price = static_cast<double>(order.price) / 1e8;
    double qty = static_cast<double>(order.quantity) / 1e8;
    
    std::stringstream price_ss, qty_ss;
    price_ss << std::fixed << std::setprecision(1) << price;
    qty_ss << std::fixed << std::setprecision(3) << qty; // Ensure precision aligns with symbol lot size if needed

    std::stringstream payload_ss;
    payload_ss << "{\"category\":\"linear\",\"symbol\":\"" << symbol 
               << "\",\"side\":\"" << side_str 
               << "\",\"orderType\":\"Limit\",\"qty\":\"" << qty_ss.str() 
               << "\",\"price\":\"" << price_ss.str() 
               << "\",\"timeInForce\":\"GTC\",\"positionIdx\":0}";
    std::string payload = payload_ss.str();

    std::string signature = generate_signature(timestamp, payload);

    ix::HttpRequestArgsPtr args = impl_->httpClient.createRequest(base_url_ + "/v5/order/create");
    args->extraHeaders["X-BAPI-API-KEY"] = api_key_;
    args->extraHeaders["X-BAPI-SIGN"] = signature;
    args->extraHeaders["X-BAPI-SIGN-TYPE"] = "2";
    args->extraHeaders["X-BAPI-TIMESTAMP"] = timestamp;
    args->extraHeaders["X-BAPI-RECV-WINDOW"] = "5000";
    args->extraHeaders["Content-Type"] = "application/json";
    args->body = payload;

    auto response = impl_->httpClient.post(base_url_ + "/v5/order/create", args);

    if (response && response->statusCode == 200) {
        std::cout << "[BybitRest] Order submitted successfully: " << response->body << std::endl;
        return true;
    }
    
    std::cerr << "[BybitRest] Order submission failed! Status: " << (response ? std::to_string(response->statusCode) : "N/A") << " Body: " << (response ? response->body : "N/A") << std::endl;
    return false;
}

bool BybitRest::cancel_order(const std::string& order_id, const std::string& symbol) {
    if (api_key_.empty() || api_secret_.empty()) {
        std::cerr << "[BybitRest] Missing API keys. Cannot cancel order." << std::endl;
        return false;
    }

    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch()).count();
    std::string timestamp = std::to_string(now);

    std::stringstream payload_ss;
    payload_ss << "{\"category\":\"linear\",\"symbol\":\"" << symbol << "\",\"orderId\":\"" << order_id << "\"}";
    std::string payload = payload_ss.str();

    std::string signature = generate_signature(timestamp, payload);

    ix::HttpRequestArgsPtr args = impl_->httpClient.createRequest(base_url_ + "/v5/order/cancel");
    args->extraHeaders["X-BAPI-API-KEY"] = api_key_;
    args->extraHeaders["X-BAPI-SIGN"] = signature;
    args->extraHeaders["X-BAPI-SIGN-TYPE"] = "2";
    args->extraHeaders["X-BAPI-TIMESTAMP"] = timestamp;
    args->extraHeaders["X-BAPI-RECV-WINDOW"] = "5000";
    args->extraHeaders["Content-Type"] = "application/json";
    args->body = payload;

    auto response = impl_->httpClient.post(base_url_ + "/v5/order/cancel", args);

    if (response && response->statusCode == 200) {
        std::cout << "[BybitRest] Order " << order_id << " cancelled successfully." << std::endl;
        return true;
    }
    
    std::cerr << "[BybitRest] Cancel failed! Status: " << (response ? std::to_string(response->statusCode) : "N/A") 
              << " Body: " << (response ? response->body : "N/A") << std::endl;
    return false;
}

bool BybitRest::cancel_all_orders(const std::string& symbol) {
    if (api_key_.empty() || api_secret_.empty()) {
        std::cerr << "[BybitRest] Missing API keys. Cannot cancel all orders." << std::endl;
        return false;
    }

    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch()).count();
    std::string timestamp = std::to_string(now);

    std::stringstream payload_ss;
    payload_ss << "{\"category\":\"linear\",\"symbol\":\"" << symbol << "\"}";
    std::string payload = payload_ss.str();

    std::string signature = generate_signature(timestamp, payload);

    ix::HttpRequestArgsPtr args = impl_->httpClient.createRequest(base_url_ + "/v5/order/cancel-all");
    args->extraHeaders["X-BAPI-API-KEY"] = api_key_;
    args->extraHeaders["X-BAPI-SIGN"] = signature;
    args->extraHeaders["X-BAPI-SIGN-TYPE"] = "2";
    args->extraHeaders["X-BAPI-TIMESTAMP"] = timestamp;
    args->extraHeaders["X-BAPI-RECV-WINDOW"] = "5000";
    args->extraHeaders["Content-Type"] = "application/json";
    args->body = payload;

    auto response = impl_->httpClient.post(base_url_ + "/v5/order/cancel-all", args);

    if (response && response->statusCode == 200) {
        std::cout << "[BybitRest] Cancel All successful: " << response->body << std::endl;
        return true;
    }
    
    std::cerr << "[BybitRest] Cancel All failed! Status: " << (response ? std::to_string(response->statusCode) : "N/A") 
              << " Body: " << (response ? response->body : "N/A") << std::endl;
    return false;
}

} // namespace gateway
} // namespace hft
