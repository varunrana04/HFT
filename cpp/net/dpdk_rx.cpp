#include "dpdk_rx.h"
#include "../core/thread_utils.h"

#ifdef HFT_USE_DPDK
#include <rte_ethdev.h>
#include <rte_eal.h>
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_udp.h>
#endif

namespace hft {
namespace net {

#define RX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32

DpdkRx::DpdkRx(const std::string& pci_addr, int core_id)
    : pci_addr_(pci_addr), core_id_(core_id) {}

DpdkRx::~DpdkRx() {
    stop();
}

bool DpdkRx::initialize() {
#ifndef HFT_USE_DPDK
    std::cerr << "[WARN] HFT_USE_DPDK is not defined. DpdkRx is mocked.\n";
    return true;
#else
    // EAL initialization requires argc/argv. In a real environment, 
    // these would be passed from main(). Mocking them here.
    char* argv[] = { (char*)"hft_engine", (char*)"-c", (char*)"1", (char*)"-n", (char*)"4" };
    int argc = 5;

    int ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        std::cerr << "[ERROR] DPDK EAL initialization failed\n";
        return false;
    }

    mbuf_pool_ = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS,
        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());

    if (mbuf_pool_ == nullptr) {
        std::cerr << "[ERROR] Cannot create mbuf pool\n";
        return false;
    }

    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0) {
        std::cerr << "[ERROR] No Ethernet ports found\n";
        return false;
    }

    // In a real setup, we would match pci_addr_ to the port_id.
    port_id_ = 0; 

    struct rte_eth_conf port_conf = {};
    port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_NONE;

    if (rte_eth_dev_configure(port_id_, 1, 1, &port_conf) < 0) {
        std::cerr << "[ERROR] Cannot configure device\n";
        return false;
    }

    if (rte_eth_rx_queue_setup(port_id_, 0, RX_RING_SIZE,
                               rte_eth_dev_socket_id(port_id_), NULL, mbuf_pool_) < 0) {
        std::cerr << "[ERROR] Cannot setup RX queue\n";
        return false;
    }

    if (rte_eth_dev_start(port_id_) < 0) {
        std::cerr << "[ERROR] Cannot start device\n";
        return false;
    }

    rte_eth_promiscuous_enable(port_id_);
    std::cout << "[INFO] DPDK Port " << port_id_ << " initialized.\n";
    return true;
#endif
}

void DpdkRx::poll_loop(
    std::function<void(const Trade&)> on_trade_callback,
    std::function<void(const BookSnapshot&)> on_book_callback
) {
    if (!utils::pin_thread_to_core(core_id_)) {
        std::cerr << "[WARN] DpdkRx could not pin to core " << core_id_ << "\n";
    }
    
    // For extreme low latency, run at max priority
    utils::set_thread_realtime_priority();

    running_ = true;

#ifndef HFT_USE_DPDK
    std::cout << "[INFO] Mock DPDK PMD Loop running on core " << core_id_ << "...\n";
    while (running_) {
        // Yield heavily in mock mode to not fry CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#else
    struct rte_mbuf* bufs[BURST_SIZE];

    std::cout << "[INFO] Starting DPDK PMD Loop on core " << core_id_ << "...\n";

    while (running_) {
        const uint16_t nb_rx = rte_eth_rx_burst(port_id_, 0, bufs, BURST_SIZE);
        if (unlikely(nb_rx == 0)) {
            continue; // Spin immediately (Zero-Copy polling)
        }

        for (uint16_t i = 0; i < nb_rx; i++) {
            // 1. Prefetch next packet to L1 Cache
            if (i < nb_rx - 1) {
                rte_prefetch0(rte_pktmbuf_mtod(bufs[i+1], void*));
            }

            // 2. Decode Ethernet -> IPv4 -> UDP
            struct rte_ether_hdr* eth_hdr = rte_pktmbuf_mtod(bufs[i], struct rte_ether_hdr*);
            if (eth_hdr->ether_type == rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)) {
                
                struct rte_ipv4_hdr* ip_hdr = (struct rte_ipv4_hdr*)(eth_hdr + 1);
                if (ip_hdr->next_proto_id == IPPROTO_UDP) {
                    
                    struct rte_udp_hdr* udp_hdr = (struct rte_udp_hdr*)(ip_hdr + 1);
                    uint8_t* payload = (uint8_t*)(udp_hdr + 1);
                    
                    // 3. Application level decoding (Mocked out for architecture demo)
                    // if (payload[0] == MSG_TYPE_TRADE) {
                    //     Trade t; ... on_trade_callback(t);
                    // }
                }
            }
            
            // 4. Free the mbuf back to the ring
            rte_pktmbuf_free(bufs[i]);
        }
    }
#endif
}

void DpdkRx::stop() {
    running_ = false;
}

} // namespace net
} // namespace hft
