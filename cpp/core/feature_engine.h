#pragma once
/**
 * @file features.h
 * @brief Feature computation engine — all 6 alpha signals.
 *
 * Computes the following signals from order book and trade data:
 *   1. Microprice    — volume-weighted fair value (stateless from book)
 *   2. OFI           — order flow imbalance (delta in top-of-book qty)
 *   3. VPIN          — volume-synchronized probability of informed trading
 *   4. Spread BPS    — bid-ask spread in basis points
 *   5. Realized Vol  — tick-level realized volatility (ring buffer)
 *   6. Stat-Arb Z    — Z-score of mid-price vs rolling mean/std
 *
 * All computation is noexcept, zero-allocation, and cache-friendly.
 * Internal state uses fixed-size ring buffers sized at compile time.
 *
 * Normalization:
 *   Signals 1 (microprice offset), 2 (OFI), 4 (spread_bps), and
 *   5 (realized_vol) are in raw/absolute units and are Z-score
 *   normalized using Welford's online algorithm before being returned
 *   in the FeatureVector.  Signals 3 (VPIN) and 6 (stat_arb_zscore)
 *   are already bounded: VPIN ∈ [0,1], stat-arb Z is a z-score.
 *   All six normalized outputs are then clamped to [-3, +3] to prevent
 *   outliers from dominating the SignalCombiner.
 */

#include "types.h"
#include "order_book.h"

#include <cstdint>
#include <cmath>
#include <algorithm>

namespace hft {

// ─── Compile-Time Limits ──────────────────────────────────────
/// Maximum VPIN buckets tracked (runtime n_buckets <= this)
static constexpr int32_t MAX_VPIN_BUCKETS = 128;

/// Maximum trade prices tracked for realized vol
static constexpr int32_t MAX_VOL_WINDOW = 4096;

/// Maximum mid prices tracked for stat-arb
static constexpr int32_t MAX_STATARB_WINDOW = 4096;

// ─── Feature Configuration ───────────────────────────────────
/**
 * @brief Tuning parameters for all alpha signals.
 *
 * Defaults match config/default_config.yaml.
 */
struct FeatureConfig {
    // VPIN
    double vpin_bucket_size   = 50.0;   ///< Volume per bucket (base units)
    int32_t vpin_n_buckets    = 50;     ///< Rolling window of buckets

    // Realized volatility
    int32_t vol_window_ticks  = 100;    ///< Trade count for vol estimation

    // Statistical arbitrage
    double  stat_arb_zscore_entry = 2.0;
    double  stat_arb_zscore_exit  = 0.5;
    int32_t stat_arb_lookback     = 1000; ///< Window for mean/std
    int32_t stat_arb_half_life_max = 500;

    /**
     * @brief Minimum observations before normalization is applied.
     *
     * Before this many ticks the normalizer lacks reliable statistics, so
     * raw values (possibly scaled to zero) are passed through.  Should be
     * less than or equal to StrategyConfig::min_warmup_ticks.
     */
    int32_t normalizer_min_obs = 50;

    /**
     * @brief Clamp bound (in standard deviations) after normalization.
     *
     * Prevents fat-tail prints from dominating the SignalCombiner.
     * e.g. 3.0 → any z-score beyond ±3σ is clipped to ±3.
     */
    double normalizer_clamp = 3.0;

    double rvr_ratio = 1.0; ///< Realized Volatility Regime Ratio
};



struct HawkesProcess {
    double mu = 0.1;      // Baseline intensity
    double alpha = 0.5;   // Jump size per trade
    double beta = 1.0;    // Decay rate (per second)
    
    double current_lambda = 0.1;
    int64_t last_update_ns = 0;

    void update(int64_t timestamp_ns, double trade_qty) noexcept {
        if (last_update_ns == 0) {
            current_lambda = mu + alpha * trade_qty;
            last_update_ns = timestamp_ns;
            return;
        }
        
        // Time difference in seconds
        double dt = static_cast<double>(timestamp_ns - last_update_ns) / 1e9;
        if (dt < 0) dt = 0;
        
        // Exponential decay
        current_lambda = mu + (current_lambda - mu) * std::exp(-beta * dt) + alpha * trade_qty;
        last_update_ns = timestamp_ns;
    }

    void reset() noexcept {
        current_lambda = mu;
        last_update_ns = 0;
    }
};

struct Lag1 {
    double prev = 0.0;
    bool has_prev = false;
    double update(double x) noexcept {
        double ret = has_prev ? prev : 0.0;
        prev = x;
        has_prev = true;
        return ret;
    }
    void reset() noexcept { prev = 0.0; has_prev = false; }
};

// ─── Feature Engine ──────────────────────────────────────────
/**
 * @brief Computes all 6 alpha signals from book and trade data.
 *
 * Usage:
 *   FeatureEngine engine(config);
 *   FeatureVector fv = engine.compute_all(book, trade);
 *
 * Must be called once per tick, in chronological order.
 * Not thread-safe — use one instance per thread.
 */
class FeatureEngine {
public:
    explicit FeatureEngine(const FeatureConfig& config = {}) noexcept;

    /**
     * @brief Compute all 6 features from the current book and latest trade.
     *
     * @param book   Current order book snapshot
     * @param trade  Latest trade event
     * @return FeatureVector with all signals normalized and populated
     */
    FeatureVector compute_all(const BookSnapshot& book,
                              const Trade& trade) noexcept;

    /// Reset all internal state (e.g., on instrument switch)
    void reset() noexcept;

    /// Get the current configuration
    [[nodiscard]] const FeatureConfig& config() const noexcept {
        return config_;
    }

    // ── Raw Signal Computation (Public for Testing) ──────────
    [[nodiscard]] double compute_microprice(const BookSnapshot& book) const noexcept;
    [[nodiscard]] double compute_ofi(const BookSnapshot& book) noexcept;
    [[nodiscard]] double compute_obi(const BookSnapshot& book) const noexcept;
    [[nodiscard]] double compute_spread_bps(const BookSnapshot& book) const noexcept;

private:
    FeatureConfig config_;

    // ── OFI state ────────────────────────────────────────────
    int64_t prev_bid_prices_[10] = {0};
    int64_t prev_ask_prices_[10] = {0};
    int64_t prev_bid_qtys_[10]   = {0};
    int64_t prev_ask_qtys_[10]   = {0};
    int32_t prev_bid_count_      = 0;
    int32_t prev_ask_count_      = 0;
    bool    has_prev_book_       = false;

    // ── VPIN state ───────────────────────────────────────────
    /// Per-bucket buy and sell volume (fixed-point, scaled by QTY_SCALE)
    double vpin_buy_vol_[MAX_VPIN_BUCKETS]  = {};
    double vpin_sell_vol_[MAX_VPIN_BUCKETS] = {};
    int32_t vpin_head_          = 0;   ///< Oldest bucket index (ring)
    int32_t vpin_count_         = 0;   ///< Filled bucket count
    double  vpin_current_buy_   = 0.0; ///< Accumulating current bucket buy vol
    double  vpin_current_sell_  = 0.0; ///< Accumulating current bucket sell vol
    double  vpin_running_abs_diff_ = 0.0; ///< O(1) tracker
    double  vpin_running_total_    = 0.0; ///< O(1) tracker

    // ── Realized Volatility & Hurst state ────────────────────
    double vol_log_returns_[MAX_VOL_WINDOW] = {};     ///< Ring of raw log returns
    double vol_log_returns_sq_[MAX_VOL_WINDOW] = {};  ///< Ring of squared log returns
    double vol_sum_ = 0.0;
    int32_t vol_head_   = 0;
    int32_t vol_count_  = 0;
    double vol_last_price_ = 0.0;
    double vol_sum_sq_ = 0.0;

    // ── Stat-Arb state ───────────────────────────────────────
    double statarb_mids_[MAX_STATARB_WINDOW] = {};  ///< Ring of mid prices
    int32_t statarb_head_  = 0;
    int32_t statarb_count_ = 0;
    double statarb_sum_ = 0.0;
    double statarb_sum_sq_ = 0.0;

    // ── Online Normalizers (Removed for ONNX/TransLOB tensor construction) ──

    // ── Hawkes Process State ─────────────────────────────────
    HawkesProcess hawkes_;

    // ── Trade Imbalance & CVD State ──────────────────────────
    double recent_buy_vol_  = 0.0;
    double recent_sell_vol_ = 0.0;
    double cvd_ = 0.0;

    // ── Signal Computation ───────────────────────────────────
    void update_vpin(const Trade& trade) noexcept;
    [[nodiscard]] double compute_vpin() const noexcept;

    void update_realized_vol(double price) noexcept;
    [[nodiscard]] double compute_realized_vol() const noexcept;
    [[nodiscard]] double compute_hurst_exponent() const noexcept;

    void update_statarb(double mid_price) noexcept;
    [[nodiscard]] double compute_statarb_zscore() const noexcept;

    /// Classify market regime from the computed signals
    [[nodiscard]] Regime classify_regime(
        double vpin, double spread_bps,
        double realized_vol, double ofi, double hurst) const noexcept;
};

// ─── SIMD Acceleration (Phase 2) ─────────────────────────────
/**
 * @brief Ultra-low latency vectorized dot product.
 * Compilers will auto-vectorize this to AVX2/FMA when compiled with -mavx2.
 */
inline double simd_dot_product_avx2(const double* a, const double* b, size_t n) noexcept {
    double total = 0.0;
    
    // Hint to compiler that pointers don't alias and it can be vectorized
    #pragma GCC ivdep
    for (size_t i = 0; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

} // namespace hft

