#include "dpdk_tx.h"
#include <iostream>

#ifdef HFT_USE_DPDK
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
#endif

namespace hft {
namespace net {

#ifdef HFT_USE_DPDK
DpdkTx::DpdkTx(uint16_t port_id, struct rte_mempool* tx_pool)
    : port_id_(port_id), tx_pool_(tx_pool) {}
#else
// Mock constructor
DpdkTx::DpdkTx(uint16_t port_id, struct rte_mempool* tx_pool)
    : port_id_(port_id) { (void)tx_pool; }
#endif

bool DpdkTx::initialize() {
    std::cout << "[INFO] DpdkTx initialized on port " << port_id_ << "\n";
    return true;
}

bool DpdkTx::send_order(const uint8_t* payload, size_t size) {
#ifndef HFT_USE_DPDK
    // Mock TX
    return true;
#else
    if (unlikely(tx_pool_ == nullptr)) return false;

    // Allocate an mbuf for transmission
    struct rte_mbuf* m = rte_pktmbuf_alloc(tx_pool_);
    if (unlikely(m == nullptr)) {
        return false;
    }

    // In a real zero-copy architecture, we would have pre-formatted 
    // Ethernet + IP + UDP headers already in the mbuf, and we would 
    // just write the payload directly.
    char* data = rte_pktmbuf_append(m, size);
    if (unlikely(data == nullptr)) {
        rte_pktmbuf_free(m);
        return false;
    }

    rte_memcpy(data, payload, size);

    // Blast it out directly to the NIC TX Queue
    const uint16_t nb_tx = rte_eth_tx_burst(port_id_, 0, &m, 1);
    
    if (unlikely(nb_tx == 0)) {
        // Queue full, free it
        rte_pktmbuf_free(m);
        return false;
    }
    
    return true;
#endif
}

} // namespace net
} // namespace hft
