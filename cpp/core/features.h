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
     * @return FeatureVector with all signals populated
     */
    FeatureVector compute_all(const BookSnapshot& book,
                              const Trade& trade) noexcept;

    /// Reset all internal state (e.g., on instrument switch)
    void reset() noexcept;

    /// Get the current configuration
    [[nodiscard]] const FeatureConfig& config() const noexcept {
        return config_;
    }

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

    // ── Realized Volatility state ────────────────────────────
    double vol_prices_[MAX_VOL_WINDOW] = {};  ///< Ring of trade prices
    int32_t vol_head_   = 0;
    int32_t vol_count_  = 0;

    // ── Stat-Arb state ───────────────────────────────────────
    double statarb_mids_[MAX_STATARB_WINDOW] = {};  ///< Ring of mid prices
    int32_t statarb_head_  = 0;
    int32_t statarb_count_ = 0;

    // ── Signal Computation ───────────────────────────────────
    [[nodiscard]] double compute_microprice(
        const BookSnapshot& book) const noexcept;

    [[nodiscard]] double compute_ofi(
        const BookSnapshot& book) noexcept;

    void update_vpin(const Trade& trade) noexcept;
    [[nodiscard]] double compute_vpin() const noexcept;

    [[nodiscard]] double compute_spread_bps(
        const BookSnapshot& book) const noexcept;

    void update_realized_vol(double price) noexcept;
    [[nodiscard]] double compute_realized_vol() const noexcept;

    void update_statarb(double mid_price) noexcept;
    [[nodiscard]] double compute_statarb_zscore() const noexcept;

    /// Classify market regime from the computed signals
    [[nodiscard]] Regime classify_regime(
        double vpin, double spread_bps,
        double realized_vol, double ofi) const noexcept;
};

} // namespace hft
