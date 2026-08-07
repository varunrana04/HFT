#pragma once
/**
 * @file signal_combiner.h
 * @brief Combines 6 alpha signals into a single trading signal.
 *
 * Supports two modes:
 *   1. WEIGHTED_AVG — Static or dynamic weighted average (default)
 *   2. ML_MODEL     — Binary model weights loaded from disk
 *
 * The ML mode loads a simple binary weight file exported by
 * train_model.py (LightGBM feature importances → normalized weights).
 * For full ONNX inference, link against onnxruntime separately.
 *
 * All methods are noexcept and O(1).
 */

#include "types.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>

namespace hft {

// ─── Combiner Mode ──────────────────────────────────────────
enum class CombinerMode : uint8_t {
    WEIGHTED_AVG = 0,  ///< Static weighted average (default)
    ML_MODEL     = 1   ///< ML-trained weights from file
};

// ─── Signal Statistics (for normalization) ──────────────────
struct SignalStats {
    double mean[6]   = {};   ///< Running mean per signal
    double var[6]    = {};   ///< Running variance per signal
    int64_t count    = 0;    ///< Total observations

    /// Update running stats with new observation (Welford's)
    void update(const double signals[6]) noexcept {
        ++count;
        double n = static_cast<double>(count);
        for (int i = 0; i < 6; ++i) {
            double delta = signals[i] - mean[i];
            mean[i] += delta / n;
            double delta2 = signals[i] - mean[i];
            var[i] += delta * delta2;
        }
    }

    /// Get standard deviation for signal i
    [[nodiscard]] double std_dev(int i) const noexcept {
        if (count < 2) return 1.0;
        return std::sqrt(var[i] / static_cast<double>(count - 1));
    }

    /// Z-normalize a signal
    [[nodiscard]] double normalize(int i, double value) const noexcept {
        double sd = std_dev(i);
        if (sd < 1e-15) return 0.0;
        return (value - mean[i]) / sd;
    }
};

// ─── ML Model Weights (binary file format) ──────────────────
/**
 * Binary format (48 bytes):
 *   - 6 × double (little-endian): weights for each signal
 *
 * Created by python/train_model.py:
 *   weights = model.feature_importances_ / sum(importances)
 *   weights.tofile("model_weights.bin")
 */
struct MLModelWeights {
    double weights[6] = {};
    double bias        = 0.0;
    bool   loaded      = false;
};

// ─── Signal Combiner ────────────────────────────────────────
class SignalCombiner {
public:
    SignalCombiner() noexcept = default;

    /**
     * @brief Combine 6 signals into α ∈ [-1, +1].
     *
     * In WEIGHTED_AVG mode: α = Σ w_i × signal_i
     * In ML_MODEL mode:     α = Σ w_i × z_normalize(signal_i) + bias
     */
    double combine(const FeatureVector& fv) noexcept;

    /**
     * @brief Set custom weights (array of up to 6 doubles).
     */
    void set_weights(const double* weights, size_t count) noexcept;

    /**
     * @brief Load ML model weights from a binary file.
     *
     * File format: 6 doubles (48 bytes) = normalized feature weights
     * Optionally followed by 1 double (8 bytes) = bias term
     *
     * @param path Path to binary weights file
     * @return true if loaded successfully
     */
    bool load_model(const char* path) noexcept;

    /**
     * @brief Switch combiner mode.
     */
    void set_mode(CombinerMode mode) noexcept { mode_ = mode; }

    /**
     * @brief Get current mode.
     */
    [[nodiscard]] CombinerMode mode() const noexcept { return mode_; }

    /**
     * @brief Check if ML model is loaded.
     */
    [[nodiscard]] bool has_model() const noexcept { return ml_weights_.loaded; }

    /**
     * @brief Get current weights (for inspection/logging).
     */
    [[nodiscard]] const double* weights() const noexcept { return weights_; }

    /**
     * @brief Enable/disable signal normalization.
     *
     * When enabled, signals are Z-normalized using running stats
     * before combining. Recommended for ML mode.
     */
    void set_normalize(bool enable) noexcept { normalize_ = enable; }

    /**
     * @brief Reset running statistics.
     */
    void reset_stats() noexcept { stats_ = {}; }

private:
    double        weights_[6] = {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6};
    CombinerMode  mode_       = CombinerMode::WEIGHTED_AVG;
    bool          normalize_  = false;
    MLModelWeights ml_weights_ = {};
    SignalStats    stats_      = {};
};

} // namespace hft
