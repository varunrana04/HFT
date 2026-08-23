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
 *   ML_MODEL      Loads 6 feature-importance weights + bias from a
 *                 56-byte binary file exported by train_model.py.
 *                 Uses the same 6 base signals, no internal re-
 *                 normalization (features arrive pre-normalized).
 *
 *   ONNX_MODEL    Full LightGBM model exported to ONNX by train_model.py.
 *                 Builds a flat feature tensor of n_features (6 base +
 *                 engineered) and runs onnxruntime inference (<5 µs).
 *                 Enabled at compile time with -DHFT_ONNX_SUPPORT.
 *                 When HFT_ONNX_SUPPORT is not defined, ONNX_MODEL
 *                 silently falls back to ML_MODEL mode.
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
 * All hot-path methods are noexcept and O(1) (except ONNX session.Run
 * which is ~2–5 µs, still well within the 13 µs tick budget).
 */

#include "types.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <atomic>

// ─── ONNX Runtime (optional) ──────────────────────────────
#if defined(HFT_ONNX_SUPPORT)
#  include <onnxruntime_cxx_api.h>
#  include <memory>
#  include <vector>
#  include <string>
#endif

namespace hft {

// ─── Combiner Mode ───────────────────────────────────────────
enum class CombinerMode : uint8_t {
    WEIGHTED_AVG = 0,  ///< Uniform/custom weights (default, zero dependencies)
    ML_MODEL     = 1,  ///< 6-weight binary file from train_model.py
    ONNX_MODEL   = 2,  ///< Full LightGBM ONNX graph via onnxruntime
};

// ─── ML Model Weights (binary file format) ───────────────────
/**
 * Binary file layout written by train_model.py::export_binary_weights():
 *   bytes  0–47 : 6 × float64  — normalized feature weights
 *   bytes 48–55 : 1 × float64  — bias term (0.0 if absent)
 * Total: 56 bytes.
 */
struct MLModelWeights {
    double weights[6] = {};
    double bias        = 0.0;
    bool   loaded      = false;
};

// ─── ONNX Session State ──────────────────────────────────────
#if defined(HFT_ONNX_SUPPORT)
/**
 * Holds the onnxruntime session and pre-allocated input/output buffers.
 * Allocated on first load_onnx_model() call, then reused every tick.
 * All inference happens through a single pre-warmed session — zero
 * allocation on the hot path.
 */
struct OnnxState {
    Ort::Env                     env{ORT_LOGGING_LEVEL_WARNING, "hft_combiner"};
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions           session_opts;
    Ort::AllocatorWithDefaultOptions alloc;

    std::vector<float>       input_buf;   ///< Flat feature tensor, reused each tick
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int64_t>     input_shape; ///< [1, n_features]
    int64_t                  n_features = 0;

    bool loaded = false;
};
#endif

// ─── Signal Combiner ─────────────────────────────────────────
class SignalCombiner {
public:
    SignalCombiner() noexcept = default;

    // No copy — OnnxState is non-copyable (unique_ptr + Ort handles)
    SignalCombiner(const SignalCombiner&)            = delete;
    SignalCombiner& operator=(const SignalCombiner&) = delete;
    SignalCombiner(SignalCombiner&&)                 = default;
    SignalCombiner& operator=(SignalCombiner&&)      = default;

    /**
     * @brief Combine signals into α ∈ [-1, +1].
     *
     * WEIGHTED_AVG: α = clamp( Σ w_i × fv.signal_i , -1, +1 )
     * ML_MODEL:     α = clamp( Σ w_i × fv.signal_i + bias , -1, +1 )
     * ONNX_MODEL:   α = clamp( ort_session.Run(feature_tensor)[0] , -1, +1 )
     *
     * In all modes the 6 signals from FeatureVector are already
     * Z-normalized by FeatureEngine — no secondary normalization here.
     *
     * ONNX_MODEL builds a [1, n_features] input tensor. The first 6
     * elements are the base signals; elements 7–N are engineered
     * features. When ONNX isn't loaded, falls back to ML_MODEL.
     */
    [[nodiscard]] double combine(const FeatureVector& fv) noexcept;

    /**
     * @brief Override the 6 base-signal weights.
     * @param weights Pointer to an array of doubles (non-null, count > 0)
     * @param count   Number of weights to copy (clamped to 6)
     */
    void set_weights(const double* weights, size_t count) noexcept;

    [[nodiscard]] bool load_optimal_weights(const char* path) noexcept;
    void set_stat_arb_valid(bool valid) noexcept { stat_arb_valid_.store(valid, std::memory_order_relaxed); }

    /**
     * @brief Load 6-weight binary model from disk (ML_MODEL mode).
     *
     * File: 6 × float64 weights + optional 1 × float64 bias (56 bytes).
     * On success, switches mode to ML_MODEL automatically.
     *
     * Wrapped in try/catch so the noexcept contract holds even if the
     * ifstream internals throw std::bad_alloc.
     *
     * @return true on success, false on any error
     */
    [[nodiscard]] bool load_model(const char* path) noexcept;

    /**
     * @brief Load an ONNX model for full LightGBM graph inference.
     *
     * On success, switches mode to ONNX_MODEL automatically.
     * If HFT_ONNX_SUPPORT is not defined, always returns false.
     *
     * @param path       Path to lgb_model.onnx
     * @param n_features Number of input features (must match model)
     * @return true on success
     */
    [[nodiscard]] bool load_onnx_model(const char* path,
                                        int64_t n_features) noexcept;

    // ── Accessors ───────────────────────────────────────────
    void        set_mode(CombinerMode m) noexcept { mode_ = m; }
    CombinerMode mode()       const noexcept { return mode_; }
    bool         has_model()  const noexcept { return ml_weights_.loaded; }
    bool         has_onnx()   const noexcept;
    const double* weights()   const noexcept { return weights_; }

    /**
     * @brief Number of engineered features expected by the ONNX model.
     * Returns 6 when no ONNX model is loaded.
     */
    [[nodiscard]] int64_t onnx_n_features() const noexcept;

private:
    // ── Weights ─────────────────────────────────────────────
    double        weights_[6] = {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6};
    CombinerMode  mode_       = CombinerMode::WEIGHTED_AVG;
    MLModelWeights ml_weights_ = {};
    std::atomic<bool> stat_arb_valid_{true};

#if defined(HFT_ONNX_SUPPORT)
    std::unique_ptr<OnnxState> onnx_;  ///< null until load_onnx_model() succeeds
#endif

    // ── Internal helpers ────────────────────────────────────
    [[nodiscard]] double combine_weighted(const double signals[6]) const noexcept;
    [[nodiscard]] double combine_ml(const double signals[6])       const noexcept;

#if defined(HFT_ONNX_SUPPORT)
    [[nodiscard]] double combine_onnx(const FeatureVector& fv) noexcept;
#endif
};

} // namespace hft
