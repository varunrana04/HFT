#pragma once
/**
 * @file signal_combiner.h
 * @brief Combines the 6 normalized alpha signals into a single α ∈ [-1, +1].
 *
 * Three modes are supported:
 *
 *   WEIGHTED_AVG  Static or dynamic weighted average of the 6 signals.
 *                 Weights are uniform by default (1/6 each); call
 *                 set_weights() to override. Signals are already
 *                 Z-normalized by FeatureEngine, so no additional
 *                 normalization is applied in this mode.
 *
 *   ML_MODEL      Loads 11 feature-importance weights + bias from a
 *                 96-byte binary file exported by train_model.py.
 *                 Uses the same 11 base signals, no internal re-
 *                 normalization (features arrive pre-normalized).
 *
 * Audit fixes applied (2025-08):
 *   - <fstream> moved to .cpp (was polluting every TU and conflicting
 *     with ONNX Runtime Windows headers)
 *   - load_model() wrapped in try/catch so noexcept is safe
 *   - set_weights() guarded against null pointer
 *   - Double-normalization bug removed: SignalStats no longer re-
 *     normalizes signals that FeatureEngine already Z-normalized
 *   - SignalStats retained for ML diagnostics / stat tracking only
 *
 * All hot-path methods are noexcept and O(1).
 */

#include "types.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <vector>



namespace hft {

// ─── Combiner Mode ───────────────────────────────────────────
enum class CombinerMode : uint8_t {
    WEIGHTED_AVG = 0,  ///< Uniform/custom weights (default, zero dependencies)
    ML_MODEL     = 1,  ///< 6-weight binary file from train_model.py
    LGBM_MODEL   = 2,  ///< Flat-array LightGBM tree evaluator
};

// ─── ML Model Weights (binary file format) ───────────────────
/**
 * Binary file layout written by train_model.py::export_binary_weights():
 *   bytes  0–87 : 11 × float64  — normalized feature weights
 *   bytes 88–95 : 1 × float64  — bias term (0.0 if absent)
 * Total: 96 bytes.
 */
struct MLModelWeights {
    double weights[11] = {};
    double bias        = 0.0;
    bool   loaded      = false;
};

// ─── LightGBM Model (Flat-Array Evaluator) ───────────────────
struct LGBMNode {
    int split_feature = -1;
    double threshold = 0.0;
    double leaf_value = 0.0;
    int left_child = -1;
    int right_child = -1;
    bool is_leaf = true;
};

struct LGBMTree {
    std::vector<LGBMNode> nodes;
};

struct LGBMModel {
    std::vector<LGBMTree> trees;
    bool loaded = false;
};



// ─── Signal Combiner ─────────────────────────────────────────
class SignalCombiner {
public:
    SignalCombiner() noexcept = default;

    SignalCombiner(const SignalCombiner&)            = delete;
    SignalCombiner& operator=(const SignalCombiner&) = delete;
    SignalCombiner(SignalCombiner&&)                 = default;
    SignalCombiner& operator=(SignalCombiner&&)      = default;

    /**
     * @brief Combine signals into α ∈ [-1, +1].
     *
     * WEIGHTED_AVG: α = clamp( Σ w_i × fv.signal_i , -1, +1 )
     * ML_MODEL:     α = clamp( Σ w_i × fv.signal_i + bias , -1, +1 )
     *
     * In all modes the 6 signals from FeatureVector are already
     * Z-normalized by FeatureEngine — no secondary normalization here.
     */
    [[nodiscard]] double combine(const FeatureVector& fv) noexcept;

    /**
     * @brief Override the 11 base-signal weights.
     * @param weights Pointer to an array of doubles (non-null, count > 0)
     * @param count   Number of weights to copy (clamped to 11)
     */
    void set_weights(const double* weights, size_t count) noexcept;

    [[nodiscard]] bool load_optimal_weights(const char* path) noexcept;
    void set_stat_arb_valid(bool valid) noexcept { stat_arb_valid_.store(valid, std::memory_order_relaxed); }

    /**
     * @brief Load 11-weight binary model from disk (ML_MODEL mode).
     *
     * File: 11 × float64 weights + optional 1 × float64 bias (96 bytes).
     * On success, switches mode to ML_MODEL automatically.
     *
     * Wrapped in try/catch so the noexcept contract holds even if the
     * ifstream internals throw std::bad_alloc.
     *
     * @return true on success, false on any error
     */
    [[nodiscard]] bool load_model(const char* path) noexcept;

    /**
     * @brief Load LightGBM text model from disk (LGBM_MODEL mode).
     * Parses the lgbm_signal_model.txt dump.
     */
    [[nodiscard]] bool load_lgbm_model(const char* path) noexcept;



    // ── Accessors ───────────────────────────────────────────
    void        set_mode(CombinerMode m) noexcept { mode_ = m; }
    CombinerMode mode()       const noexcept { return mode_; }
    bool         has_model()  const noexcept { return ml_weights_.loaded; }
    const double* weights()   const noexcept { return weights_; }

private:
    // ── Weights ─────────────────────────────────────────────
    double        weights_[11] = {
        1.0/11, 1.0/11, 1.0/11, 1.0/11, 1.0/11, 1.0/11,
        1.0/11, 1.0/11, 1.0/11, 1.0/11, 1.0/11
    };
    CombinerMode  mode_       = CombinerMode::WEIGHTED_AVG;
    MLModelWeights ml_weights_ = {};
    LGBMModel      lgbm_model_ = {};
    std::atomic<bool> stat_arb_valid_{true};



    // ── Internal helpers ────────────────────────────────────
    [[nodiscard]] double combine_weighted(const double signals[11]) const noexcept;
    [[nodiscard]] double combine_ml(const double signals[11])       const noexcept;
    [[nodiscard]] double combine_lgbm(const double signals[11])     const noexcept;
};

} // namespace hft
