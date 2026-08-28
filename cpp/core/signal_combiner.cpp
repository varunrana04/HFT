/**
 * @file signal_combiner.cpp
 * @brief Implementation of the ML-capable signal combiner.
 *
 * Audit fixes applied vs. original:
 *   1. <fstream> moved here from header (was polluting every TU)
 *   2. load_model() wrapped in try/catch (ifstream can throw bad_alloc)
 *   3. set_weights() null-pointer guard added
 *   4. Double-normalization removed: signals from FeatureEngine are
 *      already Z-normalized; SignalStats no longer re-normalizes them
 *   5. combine() split into private helpers for clarity and testability
 *   5. combine() split into private helpers for clarity and testability
 */

#include "signal_combiner.h"
#include <fstream>   // moved from header
#include <iostream>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif
#include <sstream>
#include <string>

namespace hft {

// ─── combine() ───────────────────────────────────────────────

double SignalCombiner::combine(const FeatureVector& fv) noexcept {
    const double signals[11] = {
        fv.microprice, fv.ofi, fv.vpin,
        fv.spread_bps, fv.realized_vol, fv.stat_arb_zscore,
        fv.obi, fv.trade_imbalance, fv.hawkes_intensity,
        fv.cvd, fv.hurst_exponent
    };

    // Note: signals are already Z-normalized and clamped by FeatureEngine.
    // Do NOT apply secondary normalization here — that was the bug in the
    // original SignalStats re-normalization path.



    if (mode_ == CombinerMode::LGBM_MODEL && lgbm_model_.loaded) {
        return combine_lgbm(signals);
    }

    if (mode_ == CombinerMode::ML_MODEL && ml_weights_.loaded) {
        return combine_ml(signals);
    }

    return combine_weighted(signals);
}

// ─── combine_weighted() ──────────────────────────────────────

double SignalCombiner::combine_weighted(const double signals[11]) const noexcept {
    double alpha = 0.0;
    bool stat_arb = stat_arb_valid_.load(std::memory_order_relaxed);
    for (int i = 0; i < 11; ++i) {
        if (i == 5 && !stat_arb) {
            continue; // Skip StatArb (index 5)
        }
        alpha += weights_[i] * signals[i];
    }
    return std::clamp(alpha, -1.0, 1.0);
}

// ─── combine_ml() ────────────────────────────────────────────

double SignalCombiner::combine_ml(const double signals[11]) const noexcept {
    double alpha = ml_weights_.bias;
    for (int i = 0; i < 11; ++i) {
        alpha += ml_weights_.weights[i] * signals[i];
    }
    return std::clamp(alpha, -1.0, 1.0);
}

// ─── combine_lgbm() ──────────────────────────────────────────

double SignalCombiner::combine_lgbm(const double signals[11]) const noexcept {
    double sum = 0.0;
    
    // Evaluate all trees in the ensemble
    for (const auto& tree : lgbm_model_.trees) {
        if (tree.nodes.empty()) continue;
        
        int node_idx = 0;
        while (!tree.nodes[node_idx].is_leaf) {
            const auto& node = tree.nodes[node_idx];
            if (signals[node.split_feature] <= node.threshold) {
                node_idx = node.left_child;
            } else {
                node_idx = node.right_child;
            }
        }
        sum += tree.nodes[node_idx].leaf_value;
    }
    
    // LightGBM outputs margin, we don't apply sigmoid if it was trained as Regressor
    // However, our alphas need to be in [-1, 1], and the LGBMRegressor was trained on targets in that range.
    return std::clamp(sum, -1.0, 1.0);
}

// ─── set_weights() ───────────────────────────────────────────

void SignalCombiner::set_weights(const double* weights, size_t count) noexcept {
    // Audit fix: guard against null pointer (UB in original)
    if (!weights || count == 0) return;
    const size_t n = std::min(count, size_t{11});
    std::memcpy(weights_, weights, n * sizeof(double));
}

bool SignalCombiner::load_optimal_weights(const char* path) noexcept {
    if (!path) return false;
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            // Fallback to equal weights
            for (int i = 0; i < 11; ++i) weights_[i] = 1.0 / 11.0;
            return false;
        }

        // Read 11 weights
        file.read(reinterpret_cast<char*>(weights_), 11 * sizeof(double));
        if (!file.good()) {
            for (int i = 0; i < 11; ++i) weights_[i] = 1.0 / 11.0;
            return false;
        }
        
        // Optional bias reading (ignore it for weighted avg)
        double bias = 0.0;
        file.read(reinterpret_cast<char*>(&bias), sizeof(double));
        
        mode_ = CombinerMode::WEIGHTED_AVG;
        return true;
    } catch (...) {
        for (int i = 0; i < 11; ++i) weights_[i] = 1.0 / 11.0;
        return false;
    }
}

// ─── load_model() ────────────────────────────────────────────

bool SignalCombiner::load_model(const char* path) noexcept {
    if (!path) return false;

    // Use standard C FILE* to avoid MinGW std::ifstream ABI crashes across python boundaries
    FILE* file = fopen(path, "rb");
    if (!file) return false;

    size_t read_bytes = fread(ml_weights_.weights, 1, 11 * sizeof(double), file);
    if (read_bytes != 11 * sizeof(double)) {
        fclose(file);
        ml_weights_ = {};
        return false;
    }

    // Bias is optional — if the file ends here, bias stays 0.0
    ml_weights_.bias = 0.0;
    fread(&ml_weights_.bias, 1, sizeof(double), file);

    fclose(file);
    ml_weights_.loaded = true;
    mode_ = CombinerMode::ML_MODEL;
    return true;
}


// ─── load_lgbm_model() ───────────────────────────────────────

bool SignalCombiner::load_lgbm_model(const char* path) noexcept {
    if (!path) return false;

    std::ifstream file(path);
    if (!file.is_open()) return false;

    lgbm_model_.trees.clear();
    std::string line;
    LGBMTree current_tree;
    bool parsing_tree = false;

    auto parse_ints = [](const std::string& str) {
        std::vector<int> vals;
        std::istringstream iss(str);
        int v;
        while (iss >> v) vals.push_back(v);
        return vals;
    };

    auto parse_doubles = [](const std::string& str) {
        std::vector<double> vals;
        std::istringstream iss(str);
        double v;
        while (iss >> v) vals.push_back(v);
        return vals;
    };

    try {
        while (std::getline(file, line)) {
            if (line.rfind("Tree=", 0) == 0) {
                if (parsing_tree) {
                    lgbm_model_.trees.push_back(current_tree);
                }
                current_tree = LGBMTree{};
                parsing_tree = true;
            } else if (parsing_tree && line.rfind("num_leaves=", 0) == 0) {
                int num_leaves = std::stoi(line.substr(11));
                // LightGBM internal nodes = num_leaves - 1
                // Total nodes (internal + leaves) can be indexed differently.
                // LightGBM node indices:
                // > 0 means internal node (index)
                // < 0 means leaf node (~index)
            } else if (parsing_tree && line.rfind("split_feature=", 0) == 0) {
                auto splits = parse_ints(line.substr(14));
                current_tree.nodes.resize(splits.size()); // internal nodes
                for (size_t i = 0; i < splits.size(); ++i) {
                    current_tree.nodes[i].split_feature = splits[i];
                    current_tree.nodes[i].is_leaf = false;
                }
            } else if (parsing_tree && line.rfind("threshold=", 0) == 0) {
                auto thresholds = parse_doubles(line.substr(10));
                for (size_t i = 0; i < thresholds.size(); ++i) {
                    current_tree.nodes[i].threshold = thresholds[i];
                }
            } else if (parsing_tree && line.rfind("left_child=", 0) == 0) {
                auto lefts = parse_ints(line.substr(11));
                for (size_t i = 0; i < lefts.size(); ++i) {
                    current_tree.nodes[i].left_child = lefts[i];
                }
            } else if (parsing_tree && line.rfind("right_child=", 0) == 0) {
                auto rights = parse_ints(line.substr(12));
                for (size_t i = 0; i < rights.size(); ++i) {
                    current_tree.nodes[i].right_child = rights[i];
                }
            } else if (parsing_tree && line.rfind("leaf_value=", 0) == 0) {
                auto leaves = parse_doubles(line.substr(11));
                // Add leaf nodes to the array and fix up indices
                int leaf_offset = current_tree.nodes.size();
                current_tree.nodes.resize(leaf_offset + leaves.size());
                
                for (size_t i = 0; i < leaves.size(); ++i) {
                    current_tree.nodes[leaf_offset + i].is_leaf = true;
                    current_tree.nodes[leaf_offset + i].leaf_value = leaves[i];
                }
                
                // Fix up child pointers: LightGBM leaf indices are represented as ~idx
                for (int i = 0; i < leaf_offset; ++i) {
                    if (current_tree.nodes[i].left_child < 0) {
                        current_tree.nodes[i].left_child = leaf_offset + ~current_tree.nodes[i].left_child;
                    }
                    if (current_tree.nodes[i].right_child < 0) {
                        current_tree.nodes[i].right_child = leaf_offset + ~current_tree.nodes[i].right_child;
                    }
                }
            }
        }
        if (parsing_tree) {
            lgbm_model_.trees.push_back(current_tree);
        }
        
        if (!lgbm_model_.trees.empty()) {
            lgbm_model_.loaded = true;
            mode_ = CombinerMode::LGBM_MODEL;
            return true;
        }
    } catch (...) {
        return false;
    }
    return false;
}

} // namespace hft
