/**
 * @file features.cpp
 * @brief Implementation of all 6 alpha signals.
 *
 * Signal formulas:
 *
 * 1. Microprice = (Qa * Pb + Qb * Pa) / (Qb + Qa)
 *    Where Pb/Pa = best bid/ask price, Qb/Qa = best bid/ask qty
 *
 * 2. OFI = delta_bid_qty - delta_ask_qty  (at best level)
 *    Positive OFI → buying pressure → price likely to rise
 *
 * 3. VPIN = sum(|V_buy - V_sell|) / sum(V_total) over N buckets
 *    Each bucket fills when cumulative volume reaches bucket_size
 *    Range: [0, 1] — higher = more informed/toxic flow
 *
 * 4. Spread BPS = (Pa - Pb) / mid * 10000
 *
 * 5. Realized Vol = sqrt( sum(r_i^2) / (N-1) )  where r_i = ln(p_i / p_{i-1})
 *    Computed over a rolling window of trade prices
 *
 * 6. Stat-Arb Z = (mid - mean) / std  over a rolling lookback window
 *    Used for mean-reversion signals
 */

#include "features.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace hft {

// ─── Construction / Reset ────────────────────────────────────

FeatureEngine::FeatureEngine(const FeatureConfig& config) noexcept
    : config_(config)
{
    // Clamp runtime params to compile-time limits
    config_.vpin_n_buckets = std::min(config_.vpin_n_buckets,
                                     MAX_VPIN_BUCKETS);
    config_.vol_window_ticks = std::min(config_.vol_window_ticks,
                                       MAX_VOL_WINDOW);
    config_.stat_arb_lookback = std::min(config_.stat_arb_lookback,
                                        MAX_STATARB_WINDOW);

    reset();
}

void FeatureEngine::reset() noexcept {
    prev_best_bid_price_ = INVALID_PRICE;
    prev_best_ask_price_ = INVALID_PRICE;
    prev_best_bid_qty_   = 0;
    prev_best_ask_qty_   = 0;
    has_prev_book_       = false;

    std::memset(vpin_buy_vol_, 0, sizeof(vpin_buy_vol_));
    std::memset(vpin_sell_vol_, 0, sizeof(vpin_sell_vol_));
    vpin_head_         = 0;
    vpin_count_        = 0;
    vpin_current_buy_  = 0.0;
    vpin_current_sell_ = 0.0;

    std::memset(vol_prices_, 0, sizeof(vol_prices_));
    vol_head_  = 0;
    vol_count_ = 0;

    std::memset(statarb_mids_, 0, sizeof(statarb_mids_));
    statarb_head_  = 0;
    statarb_count_ = 0;
}

// ─── Main Entry Point ────────────────────────────────────────

FeatureVector FeatureEngine::compute_all(const BookSnapshot& book,
                                         const Trade& trade) noexcept {
    FeatureVector fv{};
    fv.timestamp_ns = trade.timestamp_ns;

    // Only compute if the book is in a valid state
    if (!book.is_valid()) {
        fv.regime = Regime::UNKNOWN;
        return fv;
    }

    // 1. Microprice (stateless)
    fv.microprice = compute_microprice(book);

    // 2. OFI (needs previous book state)
    fv.ofi = compute_ofi(book);

    // 3. VPIN (update buckets, then compute)
    update_vpin(trade);
    fv.vpin = compute_vpin();

    // 4. Spread in basis points (stateless)
    fv.spread_bps = compute_spread_bps(book);

    // 5. Realized volatility (update ring buffer, then compute)
    double trade_price = fixed_to_price(trade.price);
    update_realized_vol(trade_price);
    fv.realized_vol = compute_realized_vol();

    // 6. Stat-Arb Z-score (update ring buffer, then compute)
    double mid = fixed_to_price(book.mid_price());
    update_statarb(mid);
    fv.stat_arb_zscore = compute_statarb_zscore();

    // Classify regime from the computed signals
    fv.regime = classify_regime(fv.vpin, fv.spread_bps,
                                fv.realized_vol, fv.ofi);

    return fv;
}

// ─── Signal 1: Microprice ────────────────────────────────────

double FeatureEngine::compute_microprice(
    const BookSnapshot& book) const noexcept {

    double bid_qty = fixed_to_qty(book.best_bid_qty);
    double ask_qty = fixed_to_qty(book.best_ask_qty);
    double total   = bid_qty + ask_qty;

    if (total <= 0.0) return 0.0;

    double bid_price = fixed_to_price(book.best_bid_price);
    double ask_price = fixed_to_price(book.best_ask_price);

    // Microprice: weighted average where each side's weight is the
    // OPPOSITE side's quantity (ask qty weights bid price and vice versa)
    return (ask_qty * bid_price + bid_qty * ask_price) / total;
}

// ─── Signal 2: Order Flow Imbalance ──────────────────────────

double FeatureEngine::compute_ofi(const BookSnapshot& book) noexcept {
    double ofi = 0.0;

    if (has_prev_book_) {
        // Delta bid quantity at the touch
        double delta_bid = 0.0;
        if (book.best_bid_price == prev_best_bid_price_) {
            // Same price level — quantity change is meaningful
            delta_bid = fixed_to_qty(book.best_bid_qty)
                      - fixed_to_qty(prev_best_bid_qty_);
        } else if (book.best_bid_price > prev_best_bid_price_) {
            // New higher bid appeared — all its quantity is "added"
            delta_bid = fixed_to_qty(book.best_bid_qty);
        } else {
            // Best bid dropped — all previous quantity is "removed"
            delta_bid = -fixed_to_qty(prev_best_bid_qty_);
        }

        // Delta ask quantity at the touch
        double delta_ask = 0.0;
        if (book.best_ask_price == prev_best_ask_price_) {
            delta_ask = fixed_to_qty(book.best_ask_qty)
                      - fixed_to_qty(prev_best_ask_qty_);
        } else if (book.best_ask_price < prev_best_ask_price_) {
            // New lower ask appeared — all its quantity is "added"
            delta_ask = fixed_to_qty(book.best_ask_qty);
        } else {
            // Best ask lifted — all previous quantity is "removed"
            delta_ask = -fixed_to_qty(prev_best_ask_qty_);
        }

        // OFI = bid pressure - ask pressure
        // Positive → net buying → bullish signal
        ofi = delta_bid - delta_ask;
    }

    // Save current state for next call
    prev_best_bid_price_ = book.best_bid_price;
    prev_best_ask_price_ = book.best_ask_price;
    prev_best_bid_qty_   = book.best_bid_qty;
    prev_best_ask_qty_   = book.best_ask_qty;
    has_prev_book_       = true;

    return ofi;
}

// ─── Signal 3: VPIN ──────────────────────────────────────────

void FeatureEngine::update_vpin(const Trade& trade) noexcept {
    double qty = fixed_to_qty(trade.quantity);

    // Classify volume as buy or sell based on aggressor side
    if (trade.side == Side::BID) {
        vpin_current_buy_ += qty;
    } else {
        vpin_current_sell_ += qty;
    }

    // Check if current bucket is full
    double bucket_vol = vpin_current_buy_ + vpin_current_sell_;
    if (bucket_vol < config_.vpin_bucket_size) {
        return; // Bucket not yet full
    }

    // Bucket is full — commit it to the ring buffer
    int32_t n = config_.vpin_n_buckets;

    if (vpin_count_ < n) {
        // Still filling up the initial window
        int32_t idx = (vpin_head_ + vpin_count_) % n;
        vpin_buy_vol_[idx]  = vpin_current_buy_;
        vpin_sell_vol_[idx] = vpin_current_sell_;
        ++vpin_count_;
    } else {
        // Window is full — overwrite oldest bucket
        vpin_buy_vol_[vpin_head_]  = vpin_current_buy_;
        vpin_sell_vol_[vpin_head_] = vpin_current_sell_;
        vpin_head_ = (vpin_head_ + 1) % n;
    }

    // Reset accumulators for next bucket
    vpin_current_buy_  = 0.0;
    vpin_current_sell_ = 0.0;
}

double FeatureEngine::compute_vpin() const noexcept {
    if (vpin_count_ == 0) return 0.0;

    int32_t n = config_.vpin_n_buckets;
    double sum_abs_diff = 0.0;
    double sum_total    = 0.0;

    for (int32_t i = 0; i < vpin_count_; ++i) {
        int32_t idx = (vpin_head_ + i) % n;
        double buy  = vpin_buy_vol_[idx];
        double sell = vpin_sell_vol_[idx];
        sum_abs_diff += std::abs(buy - sell);
        sum_total    += buy + sell;
    }

    if (sum_total <= 0.0) return 0.0;
    return std::clamp(sum_abs_diff / sum_total, 0.0, 1.0);
}

// ─── Signal 4: Spread BPS ────────────────────────────────────

double FeatureEngine::compute_spread_bps(
    const BookSnapshot& book) const noexcept {

    double bid = fixed_to_price(book.best_bid_price);
    double ask = fixed_to_price(book.best_ask_price);
    double mid = (bid + ask) / 2.0;

    if (mid <= 0.0) return 0.0;

    return (ask - bid) / mid * 10000.0;
}

// ─── Signal 5: Realized Volatility ───────────────────────────

void FeatureEngine::update_realized_vol(double price) noexcept {
    if (price <= 0.0) return;

    int32_t w = config_.vol_window_ticks;

    if (vol_count_ < w) {
        vol_prices_[(vol_head_ + vol_count_) % w] = price;
        ++vol_count_;
    } else {
        vol_prices_[vol_head_] = price;
        vol_head_ = (vol_head_ + 1) % w;
    }
}

double FeatureEngine::compute_realized_vol() const noexcept {
    // Need at least 2 prices to compute returns
    if (vol_count_ < 2) return 0.0;

    int32_t w = config_.vol_window_ticks;
    int32_t n = vol_count_;
    double sum_sq = 0.0;

    for (int32_t i = 1; i < n; ++i) {
        int32_t idx_prev = (vol_head_ + i - 1) % w;
        int32_t idx_curr = (vol_head_ + i) % w;
        double p_prev = vol_prices_[idx_prev];
        double p_curr = vol_prices_[idx_curr];

        if (p_prev <= 0.0) continue;

        double log_ret = std::log(p_curr / p_prev);
        sum_sq += log_ret * log_ret;
    }

    // Realized vol = sqrt( sum(r^2) / (N-1) )
    double variance = sum_sq / static_cast<double>(n - 1);
    return std::sqrt(variance);
}

// ─── Signal 6: Stat-Arb Z-Score ──────────────────────────────

void FeatureEngine::update_statarb(double mid_price) noexcept {
    if (mid_price <= 0.0) return;

    int32_t w = config_.stat_arb_lookback;

    if (statarb_count_ < w) {
        statarb_mids_[(statarb_head_ + statarb_count_) % w] = mid_price;
        ++statarb_count_;
    } else {
        statarb_mids_[statarb_head_] = mid_price;
        statarb_head_ = (statarb_head_ + 1) % w;
    }
}

double FeatureEngine::compute_statarb_zscore() const noexcept {
    // Need sufficient data for meaningful statistics
    if (statarb_count_ < 10) return 0.0;

    int32_t w = config_.stat_arb_lookback;
    int32_t n = statarb_count_;

    // Compute mean
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        sum += statarb_mids_[(statarb_head_ + i) % w];
    }
    double mean = sum / static_cast<double>(n);

    // Compute standard deviation
    double sum_sq_diff = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        double diff = statarb_mids_[(statarb_head_ + i) % w] - mean;
        sum_sq_diff += diff * diff;
    }
    double std_dev = std::sqrt(sum_sq_diff / static_cast<double>(n - 1));

    if (std_dev < 1e-12) return 0.0; // Avoid division by zero

    // Z-score of the most recent mid price
    int32_t latest_idx = (statarb_head_ + n - 1) % w;
    double latest_mid  = statarb_mids_[latest_idx];

    return (latest_mid - mean) / std_dev;
}

// ─── Regime Classification ───────────────────────────────────

Regime FeatureEngine::classify_regime(
    double vpin, double spread_bps,
    double realized_vol, double ofi) const noexcept {

    // High VPIN indicates informed/toxic flow
    if (vpin > 0.7) {
        return Regime::HIGH_TOXICITY;
    }

    // Wide spread and low realized vol → illiquid market
    if (spread_bps > 50.0) {
        return Regime::LOW_LIQUIDITY;
    }

    // Persistent directional pressure with low vol → trending
    // Strong OFI with moderate volatility suggests directional move
    if (std::abs(ofi) > 10.0 && realized_vol > 0.0 && realized_vol < 0.005) {
        return Regime::TRENDING;
    }

    return Regime::NORMAL;
}

} // namespace hft
