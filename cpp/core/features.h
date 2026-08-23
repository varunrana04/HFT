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

// ─── Online Feature Normalizer ────────────────────────────────
/**
 * @brief Welford's one-pass online mean/variance estimator for a single
 *        scalar signal.
 *
 * Used to Z-normalize features that live in raw/absolute units
 * (microprice offset from mid, OFI in qty units, spread in bps,
 * realized vol in log-return units).
 *
 * All operations are O(1) and noexcept.
 */
struct OnlineNormalizer {
    double   mean     = 0.0;
    double   M2       = 0.0;   ///< Welford's accumulator
    int64_t  count    = 0;

    /// Push a new observation and return its Z-score (or 0 if too few obs).
    double update_and_normalize(double x, int32_t min_obs,
                                double clamp) noexcept {
        ++count;
        double delta  = x - mean;
        mean         += delta / static_cast<double>(count);
        double delta2 = x - mean;
        M2           += delta * delta2;

        // Audit fix: require at least max(2, min_obs) to avoid divide-by-zero
        // in the variance formula (count-1 == 0 when count == 1)
        if (count < static_cast<int64_t>(std::max(2, min_obs))) return 0.0;

        double variance = M2 / static_cast<double>(count - 1);
        double sd       = std::sqrt(std::max(variance, 1e-30));
        double z        = (x - mean) / sd;

        return std::clamp(z, -clamp, clamp);
    }

    void reset() noexcept { mean = 0.0; M2 = 0.0; count = 0; }
};

struct WelfordVariance {
    double mean = 0.0;
    double M2 = 0.0;
    int64_t count = 0;

    void update(double x) noexcept {
        count++;
        double delta = x - mean;
        mean += delta / count;
        double delta2 = x - mean;
        M2 += delta * delta2;
    }
    double variance() const noexcept {
        if (count < 2) return 0.0;
        return M2 / (count - 1);
    }
    void reset() noexcept { mean = 0.0; M2 = 0.0; count = 0; }
};

template<int W>
struct RollingZScore {
    double buf[W] = {0};
    int head = 0;
    int count = 0;
    double sum = 0.0;
    double sum_sq = 0.0;

    double update(double x) noexcept {
        double oldest = buf[head];
        buf[head] = x;
        head = (head + 1) % W;
        
        if (count < W) {
            count++;
            sum += x;
            sum_sq += x * x;
        } else {
            sum += x - oldest;
            sum_sq += (x * x) - (oldest * oldest);
        }
        
        if (count < 2) return 0.0;
        
        double mean = sum / count;
        // variance = (E[X^2] - E[X]^2) * count / (count - 1) for sample variance
        // Or simply: (sum_sq - (sum * sum) / count) / (count - 1)
        double var_sum = sum_sq - (sum * sum) / count;
        
        // Due to floating point inaccuracies, var_sum can rarely become slightly negative.
        if (var_sum < 0.0) var_sum = 0.0;
        
        double variance = var_sum / (count - 1);
        double sd = std::sqrt(std::max(variance, 1e-30));
        double z = (x - mean) / sd;
        return std::clamp(z, -4.0, 4.0);
    }

    void reset() noexcept {
        head = 0; count = 0; sum = 0.0; sum_sq = 0.0;
        std::memset(buf, 0, sizeof(buf));
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
    int64_t prev_best_bid_price_ = INVALID_PRICE;
    int64_t prev_best_ask_price_ = INVALID_PRICE;
    int64_t prev_best_bid_qty_   = 0;
    int64_t prev_best_ask_qty_   = 0;
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

    // ── Realized Volatility state ────────────────────────────
    double vol_log_returns_sq_[MAX_VOL_WINDOW] = {};  ///< Ring of squared log returns
    int32_t vol_head_   = 0;
    int32_t vol_count_  = 0;
    double vol_last_price_ = 0.0;
    double vol_sum_sq_ = 0.0;
    WelfordVariance long_vol_tracker_;

    // ── Stat-Arb state ───────────────────────────────────────
    double statarb_mids_[MAX_STATARB_WINDOW] = {};  ///< Ring of mid prices
    int32_t statarb_head_  = 0;
    int32_t statarb_count_ = 0;
    double statarb_sum_ = 0.0;
    double statarb_sum_sq_ = 0.0;

    // ── Online Normalizers (one per raw-unit signal) ─────────
    // Signal index mapping:
    //   [0] microprice offset (= microprice - mid, in price units)
    //   [1] ofi              (in qty units, can be large)
    //   [2] spread_bps       (always positive, mean ~0.5–2 bps for BTC)
    //   [3] realized_vol     (small positive number, ~0.0001–0.005)
    OnlineNormalizer norm_microprice_;
    OnlineNormalizer norm_ofi_;
    OnlineNormalizer norm_spread_bps_;
    OnlineNormalizer norm_realized_vol_;
    OnlineNormalizer norm_obi_;

    // ── Engineered Features State ────────────────────────────
    RollingZScore<10> z10_microprice_;
    RollingZScore<50> z50_microprice_;
    RollingZScore<10> z10_ofi_;
    RollingZScore<50> z50_ofi_;
    RollingZScore<10> z10_obi_;
    RollingZScore<50> z50_obi_;

    // ── Trade Imbalance State ────────────────────────────────
    double recent_buy_vol_  = 0.0;
    double recent_sell_vol_ = 0.0;

    // ── Signal Computation ───────────────────────────────────
    void update_vpin(const Trade& trade) noexcept;
    [[nodiscard]] double compute_vpin() const noexcept;

    void update_realized_vol(double price) noexcept;
    [[nodiscard]] double compute_realized_vol() const noexcept;

    void update_statarb(double mid_price) noexcept;
    [[nodiscard]] double compute_statarb_zscore() const noexcept;

    /// Classify market regime from the computed signals
    [[nodiscard]] Regime classify_regime(
        double vpin, double spread_bps,
        double realized_vol, double ofi) const noexcept;
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

