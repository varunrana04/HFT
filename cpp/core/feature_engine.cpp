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

#include "feature_engine.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace hft {

// ─── Construction / Reset ────────────────────────────────────

FeatureEngine::FeatureEngine(const FeatureConfig& config) noexcept
    : config_(config)
{
    // Clamp runtime params to compile-time limits and prevent divide by zero (must be >= 1)
    config_.vpin_n_buckets = std::max(1, std::min(config_.vpin_n_buckets, MAX_VPIN_BUCKETS));
    config_.vol_window_ticks = std::max(1, std::min(config_.vol_window_ticks, MAX_VOL_WINDOW));
    config_.stat_arb_lookback = std::max(1, std::min(config_.stat_arb_lookback, MAX_STATARB_WINDOW));

    reset();
}

void FeatureEngine::reset() noexcept {
    std::memset(prev_bid_prices_, 0, sizeof(prev_bid_prices_));
    std::memset(prev_ask_prices_, 0, sizeof(prev_ask_prices_));
    std::memset(prev_bid_qtys_, 0, sizeof(prev_bid_qtys_));
    std::memset(prev_ask_qtys_, 0, sizeof(prev_ask_qtys_));
    prev_bid_count_ = 0;
    prev_ask_count_ = 0;
    has_prev_book_       = false;

    std::memset(vpin_buy_vol_, 0, sizeof(vpin_buy_vol_));
    std::memset(vpin_sell_vol_, 0, sizeof(vpin_sell_vol_));
    vpin_head_         = 0;
    vpin_count_        = 0;
    vpin_current_buy_  = 0.0;
    vpin_current_sell_ = 0.0;
    vpin_running_abs_diff_ = 0.0;
    vpin_running_total_    = 0.0;

    std::memset(vol_log_returns_, 0, sizeof(vol_log_returns_));
    std::memset(vol_log_returns_sq_, 0, sizeof(vol_log_returns_sq_));
    vol_head_  = 0;
    vol_count_ = 0;
    vol_last_price_ = 0.0;
    vol_sum_ = 0.0;
    vol_sum_sq_ = 0.0;

    std::memset(statarb_mids_, 0, sizeof(statarb_mids_));
    statarb_head_  = 0;
    statarb_count_ = 0;
    statarb_sum_ = 0.0;
    statarb_sum_sq_ = 0.0;

    hawkes_.reset();
    norm_microprice_.reset();
    norm_ofi_.reset();
    norm_spread_.reset();
    norm_vol_.reset();

    recent_buy_vol_  = 0.0;
    recent_sell_vol_ = 0.0;
    cvd_ = 0.0;
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
    double raw_microprice = compute_microprice(book);
    double mid_price_f    = fixed_to_price(book.mid_price());
    double spread_f = (book.best_ask_price > book.best_bid_price) ? fixed_to_price(book.best_ask_price - book.best_bid_price) : 0.01;
    
    // Spatial Normalization: scale offset by the current spread
    fv.microprice = (mid_price_f > 0.0 && spread_f > 0.0)
                    ? (raw_microprice - mid_price_f) / spread_f
                    : 0.0;

    // 2. OFI
    double raw_ofi = compute_ofi(book);
    double tob_vol = fixed_to_qty(book.best_bid_qty + book.best_ask_qty);
    // Spatial Normalization: scale by current Top of Book volume
    fv.ofi = (tob_vol > 0.0) ? (raw_ofi / tob_vol) : 0.0;

    // 3. VPIN (update buckets, then compute) — already in [0, 1]
    update_vpin(trade);
    fv.vpin = compute_vpin(); 

    // 4. Spread in basis points
    fv.spread_bps = compute_spread_bps(book);

    // 5. Realized volatility
    double trade_price = fixed_to_price(trade.price);
    update_realized_vol(trade_price);
    fv.realized_vol = compute_realized_vol();
    
    // VRP: Not yet implemented (requires long_vol_tracker_)
    double long_vol = 1e-9; 
    fv.vrp = 0.0;

    // 6. Stat-Arb Z-score
    update_statarb(mid_price_f);
    fv.stat_arb_zscore = compute_statarb_zscore();

    // 7. OBI (Order Book Imbalance) - naturally [-1, 1]
    fv.obi = compute_obi(book);
    


    // 8. Trade Imbalance (EMA of buy vs sell volume)
    double trade_qty = fixed_to_qty(trade.quantity);
    if (trade.side == Side::BID) {
        recent_buy_vol_ = recent_buy_vol_ * 0.99 + trade_qty;
        recent_sell_vol_ = recent_sell_vol_ * 0.99;
        cvd_ += trade_qty;
    } else {
        recent_sell_vol_ = recent_sell_vol_ * 0.99 + trade_qty;
        recent_buy_vol_ = recent_buy_vol_ * 0.99;
        cvd_ -= trade_qty;
    }
    double total_recent_vol = recent_buy_vol_ + recent_sell_vol_;
    fv.trade_imbalance = (total_recent_vol > 0.0) 
        ? (recent_buy_vol_ - recent_sell_vol_) / total_recent_vol 
        : 0.0;
    fv.cvd = cvd_;
    // 9. Hawkes Process Intensity
    hawkes_.update(trade.timestamp_ns, fixed_to_qty(trade.quantity));
    fv.hawkes_intensity = hawkes_.current_lambda;

    // 10. Hurst Exponent
    fv.hurst_exponent = compute_hurst_exponent();

    // Regime classification uses the RAW (physical) values so that
    // physical thresholds (e.g. spread_bps > 50) remain meaningful.
    fv.regime = classify_regime(fv.vpin, fv.spread_bps,
                                fv.realized_vol, raw_ofi, fv.hurst_exponent);

    // ── Online Normalization ─────────────────────────────────
    // Signals 1, 2, 4, 5 need Z-score normalization. 
    // VPIN (3) is already [0,1], Stat-Arb (6) is already a Z-score.
    int32_t min_obs = config_.normalizer_min_obs;
    double clamp    = config_.normalizer_clamp;

    fv.microprice   = norm_microprice_.update_and_normalize(fv.microprice, min_obs, clamp);
    fv.ofi          = norm_ofi_.update_and_normalize(fv.ofi, min_obs, clamp);
    fv.spread_bps   = norm_spread_.update_and_normalize(fv.spread_bps, min_obs, clamp);
    fv.realized_vol = norm_vol_.update_and_normalize(fv.realized_vol, min_obs, clamp);

    return fv;
}

// ─── Signal 1: Microprice ────────────────────────────────────

double FeatureEngine::compute_microprice(
    const BookSnapshot& book) const noexcept {

    double ask_qty_decayed = 0.0;
    double bid_qty_decayed = 0.0;

    // We only go up to 10 levels max
    int32_t bid_iters = std::min(book.bid_count, 10);
    int32_t ask_iters = std::min(book.ask_count, 10);

    for (int i = 0; i < bid_iters; ++i) {
        double weight = std::exp(-0.5 * i);
        bid_qty_decayed += fixed_to_qty(book.bids[i].quantity) * weight;
    }
    
    for (int i = 0; i < ask_iters; ++i) {
        double weight = std::exp(-0.5 * i);
        ask_qty_decayed += fixed_to_qty(book.asks[i].quantity) * weight;
    }

    double total = bid_qty_decayed + ask_qty_decayed;
    if (total <= 0.0) return 0.0;

    double bid_price = fixed_to_price(book.best_bid_price);
    double ask_price = fixed_to_price(book.best_ask_price);

    // Microprice: weighted average where each side's weight is the
    // OPPOSITE side's quantity (ask qty weights bid price and vice versa)
    return (ask_qty_decayed * bid_price + bid_qty_decayed * ask_price) / total;
}

// ─── Order Book Imbalance (OBI) ──────────────────────────────
double FeatureEngine::compute_obi(const BookSnapshot& book) const noexcept {
    double bid_qty = fixed_to_qty(book.best_bid_qty);
    double ask_qty = fixed_to_qty(book.best_ask_qty);
    double total = bid_qty + ask_qty;
    if (total <= 0.0) return 0.0;
    return (bid_qty - ask_qty) / total;
}

// ─── Signal 2: Order Flow Imbalance ──────────────────────────

double FeatureEngine::compute_ofi(const BookSnapshot& book) noexcept {
    double total_ofi = 0.0;

    if (has_prev_book_) {
        // Multi-level OFI with spatial decay
        int32_t bid_iters = std::min(book.bid_count, 10);
        for (int i = 0; i < bid_iters; ++i) {
            double weight = std::exp(-0.5 * i);
            double delta_bid = 0.0;
            
            if (i < prev_bid_count_ && book.bids[i].price == prev_bid_prices_[i]) {
                delta_bid = fixed_to_qty(book.bids[i].quantity) - fixed_to_qty(prev_bid_qtys_[i]);
            } else if (i >= prev_bid_count_ || book.bids[i].price > prev_bid_prices_[i]) {
                delta_bid = fixed_to_qty(book.bids[i].quantity);
            } else {
                delta_bid = -fixed_to_qty(prev_bid_qtys_[i]);
            }
            total_ofi += delta_bid * weight;
        }

        int32_t ask_iters = std::min(book.ask_count, 10);
        for (int i = 0; i < ask_iters; ++i) {
            double weight = std::exp(-0.5 * i);
            double delta_ask = 0.0;
            
            if (i < prev_ask_count_ && book.asks[i].price == prev_ask_prices_[i]) {
                delta_ask = fixed_to_qty(book.asks[i].quantity) - fixed_to_qty(prev_ask_qtys_[i]);
            } else if (i >= prev_ask_count_ || book.asks[i].price < prev_ask_prices_[i]) {
                delta_ask = fixed_to_qty(book.asks[i].quantity);
            } else {
                delta_ask = -fixed_to_qty(prev_ask_qtys_[i]);
            }
            total_ofi -= delta_ask * weight; // OFI = bids - asks
        }
    }

    // Save current state for next call up to 10 levels
    prev_bid_count_ = std::min(book.bid_count, 10);
    for (int i = 0; i < prev_bid_count_; ++i) {
        prev_bid_prices_[i] = book.bids[i].price;
        prev_bid_qtys_[i]   = book.bids[i].quantity;
    }
    prev_ask_count_ = std::min(book.ask_count, 10);
    for (int i = 0; i < prev_ask_count_; ++i) {
        prev_ask_prices_[i] = book.asks[i].price;
        prev_ask_qtys_[i]   = book.asks[i].quantity;
    }
    has_prev_book_ = true;

    return total_ofi;
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
    int32_t idx = 0;

    if (vpin_count_ < n) {
        // Still filling up the initial window
        idx = (vpin_head_ + vpin_count_) % n;
        ++vpin_count_;
    } else {
        // Window is full — subtract oldest bucket from running totals
        idx = vpin_head_;
        double old_buy = vpin_buy_vol_[idx];
        double old_sell = vpin_sell_vol_[idx];
        vpin_running_abs_diff_ -= std::abs(old_buy - old_sell);
        vpin_running_total_    -= (old_buy + old_sell);
        vpin_head_ = (vpin_head_ + 1) % n;
    }

    // Add new bucket to running totals
    vpin_buy_vol_[idx]  = vpin_current_buy_;
    vpin_sell_vol_[idx] = vpin_current_sell_;
    vpin_running_abs_diff_ += std::abs(vpin_current_buy_ - vpin_current_sell_);
    vpin_running_total_    += (vpin_current_buy_ + vpin_current_sell_);

    // Reset accumulators for next bucket
    vpin_current_buy_  = 0.0;
    vpin_current_sell_ = 0.0;
}

double FeatureEngine::compute_vpin() const noexcept {
    if (vpin_count_ == 0 || vpin_running_total_ <= 0.0) return 0.0;
    return std::clamp(vpin_running_abs_diff_ / vpin_running_total_, 0.0, 1.0);
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

    if (vol_last_price_ <= 0.0) {
        vol_last_price_ = price;
        return; // Need 2 prices to compute a return
    }

    double log_ret = std::log(price / vol_last_price_);
    double log_ret_sq = log_ret * log_ret;
    vol_last_price_ = price;

    int32_t w = config_.vol_window_ticks;
    int32_t idx = 0;

    if (vol_count_ < w) {
        idx = (vol_head_ + vol_count_) % w;
        ++vol_count_;
    } else {
        idx = vol_head_;
        vol_sum_sq_ -= vol_log_returns_sq_[idx];
        vol_sum_ -= vol_log_returns_[idx];
        vol_head_ = (vol_head_ + 1) % w;
    }

    vol_log_returns_sq_[idx] = log_ret_sq;
    vol_log_returns_[idx] = log_ret;
    vol_sum_sq_ += log_ret_sq;
    vol_sum_ += log_ret;
}

double FeatureEngine::compute_realized_vol() const noexcept {
    if (vol_count_ <= 1) return 0.0;
    
    double variance = vol_sum_sq_ / (vol_count_ - 1);
    if (variance < 0.0) variance = 0.0;
    
    return std::sqrt(variance);
}

double FeatureEngine::compute_hurst_exponent() const noexcept {
    if (vol_count_ < 10) return 0.5; // Default random walk for insufficient data

    double mean = vol_sum_ / vol_count_;
    double cumulative_deviation = 0.0;
    double max_dev = 0.0;
    double min_dev = 0.0;

    // Rescaled Range (R/S) Calculation
    for (int i = 0; i < vol_count_; ++i) {
        int idx = (vol_head_ + i) % config_.vol_window_ticks;
        cumulative_deviation += (vol_log_returns_[idx] - mean);
        if (cumulative_deviation > max_dev) max_dev = cumulative_deviation;
        if (cumulative_deviation < min_dev) min_dev = cumulative_deviation;
    }

    double R = max_dev - min_dev;
    if (R == 0.0) return 0.5;

    // Standard deviation S
    double variance = vol_sum_sq_ / vol_count_ - (mean * mean);
    if (variance <= 0.0) return 0.5;
    double S = std::sqrt(variance);

    // H = log(R/S) / log(N)
    return std::log(R / S) / std::log(static_cast<double>(vol_count_));
}

// ─── Signal 6: Stat-Arb Z-Score ──────────────────────────────

void FeatureEngine::update_statarb(double mid_price) noexcept {
    if (mid_price <= 0.0) return;

    int32_t w = config_.stat_arb_lookback;
    int32_t idx = 0;

    if (statarb_count_ < w) {
        idx = (statarb_head_ + statarb_count_) % w;
        ++statarb_count_;
    } else {
        idx = statarb_head_;
        double old_val = statarb_mids_[idx];
        statarb_sum_ -= old_val;
        statarb_sum_sq_ -= (old_val * old_val);
        statarb_head_ = (statarb_head_ + 1) % w;
    }

    statarb_mids_[idx] = mid_price;
    statarb_sum_ += mid_price;
    statarb_sum_sq_ += (mid_price * mid_price);
}

double FeatureEngine::compute_statarb_zscore() const noexcept {
    // Need sufficient data for meaningful statistics
    if (statarb_count_ < 10) return 0.0;

    int32_t n = statarb_count_;
    double mean = statarb_sum_ / static_cast<double>(n);

    // Compute variance: (sum_sq - (sum * sum) / n) / (n - 1)
    double var_sum = statarb_sum_sq_ - (statarb_sum_ * statarb_sum_) / static_cast<double>(n);
    if (var_sum < 0.0) var_sum = 0.0;
    
    double std_dev = std::sqrt(var_sum / static_cast<double>(n - 1));

    if (std_dev < 1e-12) return 0.0; // Avoid division by zero

    // Z-score of the most recent mid price
    int32_t w = config_.stat_arb_lookback;
    int32_t latest_idx = (statarb_head_ + n - 1) % w;
    double latest_mid  = statarb_mids_[latest_idx];

    return (latest_mid - mean) / std_dev;
}

// ─── Regime Classification ───────────────────────────────────

Regime FeatureEngine::classify_regime(
    double vpin, double spread_bps,
    double realized_vol, double ofi, double hurst) const noexcept {

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
    // Hurst > 0.65 confirms strong trending behavior
    if (std::abs(ofi) > 10.0 && realized_vol > 0.0 && realized_vol < 0.005 && hurst > 0.65) {
        return Regime::TRENDING;
    }

    return Regime::NORMAL;
}

} // namespace hft
