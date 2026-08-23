#pragma once

#include "inetwork.h"
#include <string>

#ifdef HFT_USE_DPDK
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#endif

namespace hft {
namespace net {

class DpdkTx : public INetworkTx {
public:
    DpdkTx(uint16_t port_id, struct rte_mempool* tx_pool);
    ~DpdkTx() override = default;

    bool initialize() override;
    
    bool send_order(const uint8_t* payload, size_t size) override;

private:
    uint16_t port_id_;
#ifdef HFT_USE_DPDK
    struct rte_mempool* tx_pool_ = nullptr;
#endif
};

} // namespace net
} // namespace hft
