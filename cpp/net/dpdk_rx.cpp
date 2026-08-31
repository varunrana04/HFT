#include "dpdk_rx.h"
#include "../core/thread_utils.h"
#include <immintrin.h>

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

namespace {
    #pragma pack(push, 1)
    struct WireTrade {
        uint8_t  msg_type;       // 1 = Trade
        uint64_t timestamp_ns;
        uint64_t sequence_num;
        int64_t  price;
        int64_t  quantity;
        uint32_t instrument_id;
        uint8_t  side;
    };

    struct WireBook {
        uint8_t  msg_type;       // 2 = Book
        uint64_t timestamp_ns;
        uint64_t sequence_num;
        uint32_t instrument_id;
        int64_t  best_bid_price;
        int64_t  best_ask_price;
        int64_t  best_bid_qty;
        int64_t  best_ask_qty;
        uint8_t  bid_count;
        uint8_t  ask_count;
        // Followed by raw PriceLevel arrays
    };
    #pragma pack(pop)
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
    (void)on_trade_callback;
    (void)on_book_callback;
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
                    
                    // 3. True Zero-Copy Application Level Decoding
                    if (payload[0] == 1) {
                        const WireTrade* wt = reinterpret_cast<const WireTrade*>(payload);
                        Trade t;
                        t.timestamp_ns  = wt->timestamp_ns;
                        t.sequence_num  = wt->sequence_num;
                        t.price         = wt->price;
                        t.quantity      = wt->quantity;
                        t.instrument_id = wt->instrument_id;
                        t.side          = static_cast<Side>(wt->side);
                        t.quality       = DataQuality::VALID;
                        on_trade_callback(t);
                    } 
                    else if (payload[0] == 2) {
                        const WireBook* wb = reinterpret_cast<const WireBook*>(payload);
                        BookSnapshot b;
                        b.timestamp_ns   = wb->timestamp_ns;
                        b.sequence_num   = wb->sequence_num;
                        b.instrument_id  = wb->instrument_id;
                        b.best_bid_price = wb->best_bid_price;
                        b.best_ask_price = wb->best_ask_price;
                        b.best_bid_qty   = wb->best_bid_qty;
                        b.best_ask_qty   = wb->best_ask_qty;
                        b.bid_count      = wb->bid_count;
                        b.ask_count      = wb->ask_count;
                        b.quality        = DataQuality::VALID;
                        
                        const PriceLevel* levels = reinterpret_cast<const PriceLevel*>(payload + sizeof(WireBook));
                        
                        // SIMD Vectorization: Parse L2 order book levels using AVX-512
                        // A 512-bit register holds 64 bytes. PriceLevel is 24 bytes.
                        // We copy 64 bytes at a time (overlapping is safe for plain data)
                        const char* src_ptr = reinterpret_cast<const char*>(levels);
                        char* dest_bids = reinterpret_cast<char*>(b.bids);
                        
                        size_t bid_bytes = b.bid_count * sizeof(PriceLevel);
                        size_t offset = 0;
                        while (offset + 64 <= bid_bytes) {
                            __m512i vec = _mm512_loadu_si512(src_ptr + offset);
                            _mm512_storeu_si512(dest_bids + offset, vec);
                            offset += 64;
                        }
                        // Remainder loop
                        for (size_t k = offset; k < bid_bytes; ++k) {
                            dest_bids[k] = src_ptr[k];
                        }

                        const char* src_asks = reinterpret_cast<const char*>(levels + b.bid_count);
                        char* dest_asks = reinterpret_cast<char*>(b.asks);
                        
                        size_t ask_bytes = b.ask_count * sizeof(PriceLevel);
                        offset = 0;
                        while (offset + 64 <= ask_bytes) {
                            __m512i vec = _mm512_loadu_si512(src_asks + offset);
                            _mm512_storeu_si512(dest_asks + offset, vec);
                            offset += 64;
                        }
                        for (size_t k = offset; k < ask_bytes; ++k) {
                            dest_asks[k] = src_asks[k];
                        }
                        
                        on_book_callback(b);
                    }
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
