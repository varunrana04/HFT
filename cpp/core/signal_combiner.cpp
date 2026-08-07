/**
 * @file signal_combiner.cpp
 * @brief Implementation of the ML-capable signal combiner.
 */

#include "signal_combiner.h"
#include <cstring>
#include <algorithm>
#include <cstdio>

namespace hft {

// ─── Combine ────────────────────────────────────────────────
double SignalCombiner::combine(const FeatureVector& fv) noexcept {
    double signals[6] = {
        fv.microprice, fv.ofi, fv.vpin,
        fv.spread_bps, fv.realized_vol, fv.stat_arb_zscore
    };

    // Update running stats (always, even in weighted mode)
    stats_.update(signals);

    // Select weights based on mode
    const double* w = weights_;
    double bias = 0.0;

    if (mode_ == CombinerMode::ML_MODEL && ml_weights_.loaded) {
        w = ml_weights_.weights;
        bias = ml_weights_.bias;
    }

    // Optionally normalize signals
    double processed[6];
    if (normalize_ && stats_.count > 10) {
        for (int i = 0; i < 6; ++i) {
            processed[i] = stats_.normalize(i, signals[i]);
        }
    } else {
        std::memcpy(processed, signals, sizeof(processed));
    }

    // Weighted sum
    double alpha = bias;
    for (int i = 0; i < 6; ++i) {
        alpha += w[i] * processed[i];
    }

    return std::clamp(alpha, -1.0, 1.0);
}

// ─── Set Weights ────────────────────────────────────────────
void SignalCombiner::set_weights(const double* weights, size_t count) noexcept {
    size_t n = std::min(count, size_t{6});
    std::memcpy(weights_, weights, n * sizeof(double));
}

// ─── Load Model ─────────────────────────────────────────────
bool SignalCombiner::load_model(const char* path) noexcept {
    if (!path) return false;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    // Read 6 weight doubles
    file.read(reinterpret_cast<char*>(ml_weights_.weights),
              6 * sizeof(double));
    if (!file.good()) return false;

    // Try to read optional bias term
    ml_weights_.bias = 0.0;
    file.read(reinterpret_cast<char*>(&ml_weights_.bias), sizeof(double));
    // If bias read fails, that's OK — bias stays 0.0

    ml_weights_.loaded = true;
    mode_ = CombinerMode::ML_MODEL;
    normalize_ = true;  // Enable normalization by default for ML

    return true;
}

} // namespace hft
