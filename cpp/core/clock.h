#pragma once
/**
 * @file clock.h
 * @brief High-resolution timing utilities for latency measurement.
 *
 * Provides nanosecond-precision timestamps using the best available
 * hardware source:
 *   - Windows: QueryPerformanceCounter (QPC)
 *   - Linux:   clock_gettime(CLOCK_MONOTONIC) or RDTSC
 *
 * All timing functions are marked noexcept and inline for zero
 * overhead on the hot path. The ScopedTimer class measures elapsed
 * time for any code block — use it to profile each stage of the
 * tick-to-trade pipeline.
 */

#include <cstdint>
#include <chrono>

#ifdef _WIN32
#include <intrin.h>   // __rdtsc on MSVC
#else
#include <x86intrin.h> // __rdtsc on GCC/Clang
#include <time.h>
#endif

namespace hft {

/**
 * @brief Get current timestamp in nanoseconds (monotonic clock).
 *
 * Uses the highest-resolution monotonic clock available on the platform.
 * This is the primary timestamp for all latency measurements.
 *
 * @return Nanoseconds since an arbitrary epoch (monotonic, never decreases)
 */
[[nodiscard]] inline int64_t now_ns() noexcept {
    auto tp = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        tp.time_since_epoch()
    );
    return ns.count();
}

/**
 * @brief Read the CPU timestamp counter (RDTSC) for ultra-fine timing.
 *
 * RDTSC gives cycle-level precision (~0.3ns per cycle on a 3GHz CPU).
 * Use this for micro-benchmarking individual functions on the hot path.
 * Do NOT use for wall-clock time — the counter frequency varies with
 * CPU frequency scaling unless the invariant TSC feature is present.
 *
 * @return Raw CPU cycle count
 */
[[nodiscard]] inline uint64_t rdtsc() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    return __rdtsc();
#else
    // Fallback for non-x86 (ARM, etc.) — use steady_clock
    return static_cast<uint64_t>(now_ns());
#endif
}

/**
 * @brief Serializing RDTSC — forces all prior instructions to complete.
 *
 * Use rdtscp() instead of rdtsc() when you need to ensure that the
 * timestamp is taken AFTER all preceding instructions have retired.
 * This prevents out-of-order execution from skewing measurements.
 *
 * @return Raw CPU cycle count (serialized)
 */
[[nodiscard]] inline uint64_t rdtscp() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int aux;
    return __rdtscp(&aux);
#elif defined(__i386__) || defined(_M_IX86)
    unsigned int aux;
    return __rdtscp(&aux);
#else
    return static_cast<uint64_t>(now_ns());
#endif
}

inline double& tsc_to_ns_ref() noexcept {
    static double val = 1.0; // Defaults to 1 cycle = 1 ns if uncalibrated
    return val;
}

/**
 * @brief Calibrates the TSC (Time Stamp Counter) frequency against the steady clock.
 * Must be called once during application startup on the main thread.
 */
inline void calibrate_tsc() noexcept {
    auto t0 = std::chrono::steady_clock::now();
    auto c0 = rdtscp();
    // Busy wait for 10ms to get a good sample
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(10)) {}
    auto c1 = rdtscp();
    auto t1 = std::chrono::steady_clock::now();
    
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    tsc_to_ns_ref() = static_cast<double>(ns) / static_cast<double>(c1 - c0);
}

/**
 * @brief RAII timer that measures elapsed nanoseconds for a scope.
 * Uses TSC (rdtscp) for ultra-low overhead (~20 cycles vs ~100+ for clock_gettime).
 */
class ScopedTimer {
public:
    explicit ScopedTimer(int64_t& output) noexcept
        : output_(output), start_(rdtscp()) {}

    ~ScopedTimer() noexcept {
        uint64_t end = rdtscp();
        output_ = static_cast<int64_t>(static_cast<double>(end - start_) * tsc_to_ns_ref());
    }

    // Non-copyable, non-movable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

private:
    int64_t& output_;
    uint64_t start_;
};

/**
 * @brief Cycle-based RAII timer for ultra-fine profiling.
 *
 * Same as ScopedTimer but uses RDTSC cycles instead of nanoseconds.
 * Convert cycles to nanoseconds using your CPU frequency.
 */
class ScopedCycleTimer {
public:
    explicit ScopedCycleTimer(uint64_t& output) noexcept
        : output_(output), start_(rdtscp()) {}

    ~ScopedCycleTimer() noexcept {
        output_ = rdtscp() - start_;
    }

    ScopedCycleTimer(const ScopedCycleTimer&) = delete;
    ScopedCycleTimer& operator=(const ScopedCycleTimer&) = delete;
    ScopedCycleTimer(ScopedCycleTimer&&) = delete;
    ScopedCycleTimer& operator=(ScopedCycleTimer&&) = delete;

private:
    uint64_t& output_;
    uint64_t  start_;
};

} // namespace hft
