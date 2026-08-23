/**
 * @file latency_profiler.cpp
 * @brief Full-pipeline latency benchmarks for the HFT engine.
 *
 * Measures p50, p99, and p99.9 latencies for every stage:
 *   1. Order Book Update           — Single level insertion
 *   2. Data Validation             — Trade validation (8 checks)
 *   3. SPSC Queue Push/Pop         — Lock-free ring buffer roundtrip
 *   4. CSV Parse                   — Binance aggTrade line parse
 *   5. Feature Computation         — All 6 alpha signals
 *   6. Signal Combination          — Weighted average
 *   7. Risk Check                  — 5 pre-trade gates
 *   8. Strategy Engine (Full)      — Complete tick-to-trade pipeline
 *
 * Compile:
 *   cmake --build build --config Release --target hft_bench
 *
 * Run:
 *   ./build/hft_bench
 *
 * Output: Formatted markdown table suitable for README.md
 */

#include "types.h"
#include "clock.h"
#include "order_book.h"
#include "data_validator.h"
#include "spsc_queue.h"
#include "market_data.h"
#include "features.h"
#include "signal_combiner.h"
#include "risk_manager.h"
#include "order_manager.h"
#include "strategy_engine.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <string>

// ─── Benchmark Result ───────────────────────────────────────

struct BenchResult {
    std::string name;
    int64_t p50;
    int64_t p99;
    int64_t p999;
    int64_t min_ns;
    int64_t max_ns;
    double  avg;
};

// ─── Statistics Helper ──────────────────────────────────────

static BenchResult compute_stats(const char* label,
                                  std::vector<int64_t>& samples) {
    BenchResult r;
    r.name = label;

    if (samples.empty()) {
        r.p50 = r.p99 = r.p999 = r.min_ns = r.max_ns = 0;
        r.avg = 0;
        return r;
    }

    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();

    auto percentile = [&](double p) -> int64_t {
        size_t idx = static_cast<size_t>(p * static_cast<double>(n - 1));
        return samples[idx];
    };

    int64_t sum = 0;
    for (auto s : samples) sum += s;

    r.avg    = static_cast<double>(sum) / static_cast<double>(n);
    r.p50    = percentile(0.50);
    r.p99    = percentile(0.99);
    r.p999   = percentile(0.999);
    r.min_ns = samples.front();
    r.max_ns = samples.back();

    return r;
}

// ─── Print Helpers ──────────────────────────────────────────

static void print_header() {
    std::cout << "\n";
    std::cout << "| Stage | avg (ns) | p50 (ns) | p99 (ns) | p99.9 (ns) | min (ns) | max (ns) |\n";
    std::cout << "|---|---|---|---|---|---|---|\n";
}

static void print_row(const BenchResult& r) {
    std::cout << "| " << std::setw(35) << std::left << r.name
              << " | " << std::setw(8) << static_cast<int64_t>(r.avg)
              << " | " << std::setw(8) << r.p50
              << " | " << std::setw(8) << r.p99
              << " | " << std::setw(10) << r.p999
              << " | " << std::setw(8) << r.min_ns
              << " | " << std::setw(8) << r.max_ns
              << " |" << std::endl;
}

static void print_console_row(const BenchResult& r) {
    std::cout << "  " << std::setw(35) << std::left << r.name
              << "  avg=" << std::setw(8) << static_cast<int64_t>(r.avg) << "ns"
              << "  p50=" << std::setw(8) << r.p50 << "ns"
              << "  p99=" << std::setw(8) << r.p99 << "ns"
              << "  p99.9=" << std::setw(8) << r.p999 << "ns"
              << std::endl;
}

// ─── Create Realistic Test Data ─────────────────────────────

static hft::BookSnapshot make_book(int64_t seq) {
    hft::BookSnapshot book{};
    book.timestamp_ns = hft::now_ns();
    book.sequence_num = seq;
    book.bid_count = 5;
    book.ask_count = 5;

    for (int i = 0; i < 5; ++i) {
        book.bids[i].price = hft::price_to_fixed(50000.0 - i * 0.01);
        book.bids[i].quantity = hft::qty_to_fixed(1.0 + i * 0.5);
        book.bids[i].order_count = 10 + i;

        book.asks[i].price = hft::price_to_fixed(50000.01 + i * 0.01);
        book.asks[i].quantity = hft::qty_to_fixed(1.0 + i * 0.5);
        book.asks[i].order_count = 10 + i;
    }

    book.best_bid_price = book.bids[0].price;
    book.best_ask_price = book.asks[0].price;
    book.best_bid_qty = book.bids[0].quantity;
    book.best_ask_qty = book.asks[0].quantity;
    book.quality = hft::DataQuality::VALID;

    return book;
}

static hft::Trade make_trade(int64_t seq) {
    hft::Trade t{};
    t.timestamp_ns = hft::now_ns();
    t.sequence_num = seq;
    t.price = hft::price_to_fixed(50000.005);
    t.quantity = hft::qty_to_fixed(0.1);
    t.side = (seq % 2 == 0) ? hft::Side::BID : hft::Side::ASK;
    t.instrument_id = 0;
    t.quality = hft::DataQuality::VALID;
    return t;
}

// ─── Main ───────────────────────────────────────────────────

int main() {
    constexpr int WARMUP = 10'000;
    constexpr int ITERATIONS = 100'000;

    std::cout << "\n╔══════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     HFT Engine — Full Pipeline Latency Profiler  ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Warmup:     " << std::setw(8) << WARMUP
              << " iterations                  ║" << std::endl;
    std::cout << "║  Benchmark:  " << std::setw(8) << ITERATIONS
              << " iterations                  ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════╝\n" << std::endl;

    std::vector<BenchResult> results;

    // ── 1. Order Book Update ────────────────────────────────
    {
        std::vector<int64_t> latencies;
        latencies.reserve(ITERATIONS);
        auto book = std::make_unique<hft::OrderBook>(1);

        // Warmup
        for (int i = 0; i < WARMUP; ++i) {
            hft::LevelUpdate u{};
            u.timestamp_ns = hft::now_ns();
            u.sequence_num = static_cast<int64_t>(i);
            u.price = hft::price_to_fixed(50000.0 + (i % 20) * 0.01);
            u.quantity = hft::qty_to_fixed(1.0);
            u.order_count = 1;
            u.side = (i % 2 == 0) ? hft::Side::BID : hft::Side::ASK;
            book->apply_update(u);
            if (i % 1000 == 999) book->reset();
        }
        book->reset();

        // Benchmark
        for (int i = 0; i < ITERATIONS; ++i) {
            hft::LevelUpdate u{};
            u.timestamp_ns = hft::now_ns();
            u.sequence_num = static_cast<int64_t>(i);
            u.price = hft::price_to_fixed(50000.0 + (i % 20) * 0.01);
            u.quantity = hft::qty_to_fixed(1.0 + (i % 10) * 0.1);
            u.order_count = 1;
            u.side = (i % 2 == 0) ? hft::Side::BID : hft::Side::ASK;

            int64_t elapsed;
            { hft::ScopedTimer timer(elapsed); book->apply_update(u); }
            latencies.push_back(elapsed);
            if (i % 1000 == 999) book->reset();
        }
        results.push_back(compute_stats("Order Book Update", latencies));
    }

    // ── 2. Data Validation ──────────────────────────────────
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
            { hft::ScopedTimer timer(elapsed); validator.validate_trade(t, now); }
            latencies.push_back(elapsed);
        }
        results.push_back(compute_stats("Data Validation (Trade)", latencies));
    }

    // ── 3. SPSC Queue Push/Pop ──────────────────────────────
    {
        std::vector<int64_t> latencies;
        latencies.reserve(ITERATIONS);
        auto queue = std::make_unique<hft::SPSCQueue<hft::Trade, 65536>>();

        for (int i = 0; i < ITERATIONS; ++i) {
            hft::Trade t{};
            t.timestamp_ns = hft::now_ns();
            t.price = hft::price_to_fixed(50000.0);

            int64_t elapsed;
            {
                hft::ScopedTimer timer(elapsed);
                [[maybe_unused]] bool pushed = queue->try_push(t);
                hft::Trade out;
                [[maybe_unused]] bool popped = queue->try_pop(out);
            }
            latencies.push_back(elapsed);
        }
        results.push_back(compute_stats("SPSC Push+Pop", latencies));
    }

    // ── 4. CSV Parse ────────────────────────────────────────
    {
        std::vector<int64_t> latencies;
        latencies.reserve(ITERATIONS);
        std::string line = "123456789,50000.12345678,0.00100000,100,105,1704067200000,false,true";

        for (int i = 0; i < ITERATIONS; ++i) {
            hft::Trade trade;
            int64_t elapsed;
            { hft::ScopedTimer timer(elapsed); hft::parse_binance_agg_trade(line, trade); }
            latencies.push_back(elapsed);
        }
        results.push_back(compute_stats("CSV Parse (aggTrade)", latencies));
    }

    // ── 5. Feature Engine (All 6 Signals) ───────────────────
    {
        std::vector<int64_t> latencies;
        latencies.reserve(ITERATIONS);
        auto engine = std::make_unique<hft::FeatureEngine>();

        // Warmup with synthetic data
        for (int i = 0; i < WARMUP; ++i) {
            auto book = make_book(i);
            auto trade = make_trade(i);
            engine->compute_all(book, trade);
        }

        // Benchmark
        for (int i = 0; i < ITERATIONS; ++i) {
            auto book = make_book(WARMUP + i);
            auto trade = make_trade(WARMUP + i);

            int64_t elapsed;
            { hft::ScopedTimer timer(elapsed); engine->compute_all(book, trade); }
            latencies.push_back(elapsed);
        }
        results.push_back(compute_stats("Feature Engine (6 signals)", latencies));
    }

    // ── 6. Signal Combiner ──────────────────────────────────
    {
        std::vector<int64_t> latencies;
        latencies.reserve(ITERATIONS);
        hft::SignalCombiner combiner;

        hft::FeatureVector fv{};
        fv.microprice = 0.001;
        fv.ofi = 0.5;
        fv.vpin = 0.3;
        fv.spread_bps = 2.0;
        fv.realized_vol = 0.01;
        fv.stat_arb_zscore = -1.5;

        for (int i = 0; i < ITERATIONS; ++i) {
            fv.ofi = 0.5 * ((i % 100) - 50) / 50.0;

            int64_t elapsed;
            { hft::ScopedTimer timer(elapsed); combiner.combine(fv); }
            latencies.push_back(elapsed);
        }
        results.push_back(compute_stats("Signal Combiner", latencies));
    }

    // ── 7. Risk Manager ─────────────────────────────────────
    {
        std::vector<int64_t> latencies;
        latencies.reserve(ITERATIONS);
        hft::RiskConfig rcfg;
        hft::RiskManager risk(rcfg);
        risk.update_equity(100000.0);

        for (int i = 0; i < ITERATIONS; ++i) {
            hft::Order order{};
            order.order_id = static_cast<uint64_t>(i);
            order.price = hft::price_to_fixed(50000.0);
            order.quantity = hft::qty_to_fixed(0.01);
            order.side = hft::Side::BID;
            order.type = hft::OrderType::LIMIT;

            int64_t elapsed;
            {
                hft::ScopedTimer timer(elapsed);
                risk.check_order(order, 0, 0.0, 100000.0);
            }
            latencies.push_back(elapsed);
        }
        results.push_back(compute_stats("Risk Manager (5 gates)", latencies));
    }

    // ── 8. Full Pipeline: tick → features → signal → risk ──
    {
        std::vector<int64_t> latencies;
        latencies.reserve(ITERATIONS);

        hft::StrategyConfig scfg;
        scfg.initial_capital = 100000.0;
        scfg.alpha_entry_threshold = 0.10;
        hft::FeatureConfig fcfg;
        hft::RiskConfig rcfg;

        auto engine = std::make_unique<hft::StrategyEngine>(scfg, fcfg, rcfg);

        // Warmup
        for (int i = 0; i < WARMUP; ++i) {
            auto book = make_book(i);
            auto trade = make_trade(i);
            engine->on_trade(trade, book);
        }
        engine->reset();

        // Benchmark: full tick-to-trade
        for (int i = 0; i < ITERATIONS; ++i) {
            auto book = make_book(WARMUP + i);
            auto trade = make_trade(WARMUP + i);

            int64_t elapsed;
            { hft::ScopedTimer timer(elapsed); engine->on_trade(trade, book); }
            latencies.push_back(elapsed);
        }
        results.push_back(compute_stats("Full Pipeline (Tick-to-Trade)", latencies));
    }

    // ── Print Results ───────────────────────────────────────

    std::cout << "=== Console Output ===\n" << std::endl;
    for (const auto& r : results) {
        print_console_row(r);
    }

    std::cout << "\n\n=== Markdown Table (copy to README.md) ===\n";
    print_header();
    for (const auto& r : results) {
        print_row(r);
    }

    // ── Summary ─────────────────────────────────────────────

    auto& full = results.back();
    std::cout << "\n╔══════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  FULL PIPELINE SUMMARY (Tick-to-Trade)           ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  p50:    " << std::setw(8) << full.p50
              << " ns                                ║" << std::endl;
    std::cout << "║  p99:    " << std::setw(8) << full.p99
              << " ns                                ║" << std::endl;
    std::cout << "║  p99.9:  " << std::setw(8) << full.p999
              << " ns                                ║" << std::endl;
    std::cout << "║  avg:    " << std::setw(8) << static_cast<int64_t>(full.avg)
              << " ns                                ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════╝" << std::endl;

    std::cout << "\n=== Done ===" << std::endl;
    return 0;
}
