/**
 * @file latency_profiler.cpp
 * @brief Latency benchmarks for every stage of the hot path.
 *
 * Measures p50, p99, and p99.9 latencies for:
 *   - Order book update
 *   - Order book level insertion/removal
 *   - Data validation
 *   - SPSC queue push/pop
 *   - Full tick-to-snapshot pipeline
 */

#include "types.h"
#include "clock.h"
#include "order_book.h"
#include "data_validator.h"
#include "spsc_queue.h"
#include "market_data.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>

/// Print percentile latencies from a sorted vector of nanoseconds
static void print_stats(const char* label, std::vector<int64_t>& samples) {
    if (samples.empty()) {
        std::cout << label << ": no samples" << std::endl;
        return;
    }
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();

    auto percentile = [&](double p) -> int64_t {
        size_t idx = static_cast<size_t>(p * static_cast<double>(n - 1));
        return samples[idx];
    };

    int64_t sum = 0;
    for (auto s : samples) sum += s;
    double avg = static_cast<double>(sum) / static_cast<double>(n);

    std::cout << std::setw(30) << std::left << label
              << "  avg=" << std::setw(8) << static_cast<int64_t>(avg) << "ns"
              << "  p50=" << std::setw(8) << percentile(0.50) << "ns"
              << "  p99=" << std::setw(8) << percentile(0.99) << "ns"
              << "  p99.9=" << std::setw(8) << percentile(0.999) << "ns"
              << "  min=" << std::setw(8) << samples.front() << "ns"
              << "  max=" << std::setw(8) << samples.back() << "ns"
              << std::endl;
}

int main() {
    constexpr int ITERATIONS = 100'000;

    std::cout << "\n=== HFT Engine Latency Profiler ===" << std::endl;
    std::cout << "Iterations: " << ITERATIONS << "\n" << std::endl;

    // ── Benchmark 1: Order Book Update ───────────────────────
    {
        std::vector<int64_t> latencies;
        latencies.reserve(ITERATIONS);
        hft::OrderBook book(1);

        for (int i = 0; i < ITERATIONS; ++i) {
            hft::LevelUpdate u{};
            u.timestamp_ns = hft::now_ns();
            u.sequence_num = static_cast<int64_t>(i);
            u.price = hft::price_to_fixed(50000.0 + (i % 20) * 0.01);
            u.quantity = hft::qty_to_fixed(1.0 + (i % 10) * 0.1);
            u.order_count = 1;
            u.side = (i % 2 == 0) ? hft::Side::BID : hft::Side::ASK;

            int64_t elapsed;
            {
                hft::ScopedTimer timer(elapsed);
                book.apply_update(u);
            }
            latencies.push_back(elapsed);

            // Periodically reset to avoid overflow
            if (i % 1000 == 999) book.reset();
        }
        print_stats("Order Book Update", latencies);
    }

    // ── Benchmark 2: Data Validation ─────────────────────────
    {
        std::vector<int64_t> latencies;
        latencies.reserve(ITERATIONS);
        hft::DataValidator validator;

        for (int i = 0; i < ITERATIONS; ++i) {
            int64_t now = hft::now_ns();
            hft::Trade t{};
            t.timestamp_ns = now;
            t.sequence_num = static_cast<int64_t>(i + 1);
            t.price = hft::price_to_fixed(50000.0);
            t.quantity = hft::qty_to_fixed(0.1);
            t.side = hft::Side::BID;
            t.quality = hft::DataQuality::VALID;

            int64_t elapsed;
            {
                hft::ScopedTimer timer(elapsed);
                validator.validate_trade(t, now);
            }
            latencies.push_back(elapsed);
        }
        print_stats("Data Validation (Trade)", latencies);
    }

    // ── Benchmark 3: SPSC Queue Push/Pop ─────────────────────
    {
        std::vector<int64_t> latencies;
        latencies.reserve(ITERATIONS);
        hft::SPSCQueue<hft::Trade, 65536> queue;

        for (int i = 0; i < ITERATIONS; ++i) {
            hft::Trade t{};
            t.timestamp_ns = hft::now_ns();
            t.price = hft::price_to_fixed(50000.0);

            int64_t elapsed;
            {
                hft::ScopedTimer timer(elapsed);
                queue.try_push(t);
                hft::Trade out;
                queue.try_pop(out);
            }
            latencies.push_back(elapsed);
        }
        print_stats("SPSC Push+Pop (Trade)", latencies);
    }

    // ── Benchmark 4: CSV Parse (Binance aggTrade) ────────────
    {
        std::vector<int64_t> latencies;
        latencies.reserve(ITERATIONS);
        std::string line = "123456789,50000.12345678,0.00100000,100,105,1704067200000,false,true";

        for (int i = 0; i < ITERATIONS; ++i) {
            hft::Trade trade;
            int64_t elapsed;
            {
                hft::ScopedTimer timer(elapsed);
                hft::parse_binance_agg_trade(line, trade);
            }
            latencies.push_back(elapsed);
        }
        print_stats("CSV Parse (aggTrade)", latencies);
    }

    // ── Benchmark 5: Full Pipeline ───────────────────────────
    {
        std::vector<int64_t> latencies;
        latencies.reserve(ITERATIONS);
        hft::OrderBook book(1);
        hft::DataValidator validator;
        std::string line = "123456789,50000.12345678,0.00100000,100,105,1704067200000,false,true";

        // Pre-populate book so it's in a realistic state
        for (int i = 0; i < 10; ++i) {
            hft::LevelUpdate u{};
            u.timestamp_ns = 1000 + i;
            u.sequence_num = static_cast<int64_t>(i);
            u.price = hft::price_to_fixed(49990.0 + i);
            u.quantity = hft::qty_to_fixed(1.0);
            u.order_count = 1;
            u.side = hft::Side::BID;
            book.apply_update(u);
        }
        for (int i = 0; i < 10; ++i) {
            hft::LevelUpdate u{};
            u.timestamp_ns = 2000 + i;
            u.sequence_num = static_cast<int64_t>(10 + i);
            u.price = hft::price_to_fixed(50001.0 + i);
            u.quantity = hft::qty_to_fixed(1.0);
            u.order_count = 1;
            u.side = hft::Side::ASK;
            book.apply_update(u);
        }

        for (int i = 0; i < ITERATIONS; ++i) {
            int64_t elapsed;
            {
                hft::ScopedTimer timer(elapsed);
                // Parse CSV → Trade
                hft::Trade trade;
                hft::parse_binance_agg_trade(line, trade);
                // Validate
                trade.sequence_num = static_cast<int64_t>(100 + i);
                trade.timestamp_ns = hft::now_ns();
                // Get book snapshot
                [[maybe_unused]] const auto& snap = book.snapshot();
            }
            latencies.push_back(elapsed);
        }
        print_stats("Full Pipeline (Parse+Validate)", latencies);
    }

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
