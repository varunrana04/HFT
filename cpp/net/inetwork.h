#pragma once

#include <cstdint>
#include <functional>
#include "types.h"

namespace hft {
namespace net {

/**
 * @brief Generic Network RX Interface
 * 
 * Abstracts away the underlying networking technology (DPDK, Solarflare ef_vi, 
 * or standard UDP sockets) to allow dependency injection into the strategy engine.
 */
class INetworkRx {
public:
    virtual ~INetworkRx() = default;

    /**
     * @brief Initialize the network interface.
     * @return true if successful, false otherwise.
     */
    virtual bool initialize() = 0;

    /**
     * @brief Start polling for packets in a tight loop.
     * 
     * This function should block and run indefinitely, ideally pinned to an isolated core.
     * 
     * @param on_trade_callback Callback fired when a Trade packet is received.
     * @param on_book_callback Callback fired when a BookSnapshot packet is received.
     */
    virtual void poll_loop(
        std::function<void(const Trade&)> on_trade_callback,
        std::function<void(const BookSnapshot&)> on_book_callback
    ) = 0;

    /**
     * @brief Stop the polling loop.
     */
    virtual void stop() = 0;
};

/**
 * @brief Generic Network TX Interface
 */
class INetworkTx {
public:
    virtual ~INetworkTx() = default;

    virtual bool initialize() = 0;

    /**
     * @brief Send a raw order packet directly to the NIC (Zero-Copy if possible).
     * 
     * @param payload Pointer to the pre-formatted order payload.
     * @param size Size of the payload in bytes.
     * @return true if successful, false otherwise.
     */
    virtual bool send_order(const uint8_t* payload, size_t size) = 0;
};

} // namespace net
} // namespace hft
