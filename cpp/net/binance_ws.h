#pragma once

#include <string>
#include <functional>
#include <memory>
#include <cstdint>

namespace hft {
namespace net {

// Forward declaration of internal Boost.Beast state
struct BinanceWsState;

/**
 * @brief Direct C++ WebSocket Connector for Binance L2 Order Book.
 * 
 * Replaces the Python Python-to-C++ bridge. This uses Boost.Asio
 * and Boost.Beast to achieve microsecond-latency network reads directly
 * into the C++ memory space.
 */
class BinanceWsClient {
public:
    BinanceWsClient(const std::string& host, const std::string& port, const std::string& stream_name);
    ~BinanceWsClient();

    // Callback type for incoming L2 updates (raw JSON string for now, will be parsed with simdjson)
    using MessageCallback = std::function<void(const char* data, size_t len)>;

    void set_on_message(MessageCallback cb);

    // Run the Asio event loop (blocks until stopped)
    void run();

    // Stop the event loop safely
    void stop();

private:
    std::unique_ptr<BinanceWsState> state_;
    std::string host_;
    std::string port_;
    std::string stream_name_;
};

} // namespace net
} // namespace hft
