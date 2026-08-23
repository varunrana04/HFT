#pragma once

#include "inetwork.h"
#include <string>
#include <atomic>
#include <iostream>

#ifdef HFT_USE_DPDK
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#endif

namespace hft {
namespace net {

class DpdkRx : public INetworkRx {
public:
    DpdkRx(const std::string& pci_addr, int core_id);
    ~DpdkRx() override;

    bool initialize() override;
    
    void poll_loop(
        std::function<void(const Trade&)> on_trade_callback,
        std::function<void(const BookSnapshot&)> on_book_callback
    ) override;

    void stop() override;

private:
    std::string pci_addr_;
    int core_id_;
    std::atomic<bool> running_{false};

#ifdef HFT_USE_DPDK
    uint16_t port_id_ = 0;
    struct rte_mempool* mbuf_pool_ = nullptr;
#endif
};

} // namespace net
} // namespace hft
