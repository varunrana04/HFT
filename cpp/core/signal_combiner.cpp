#include "signal_combiner.h"
#include <cstring>
#include <algorithm>
namespace hft {
double SignalCombiner::combine(const FeatureVector& fv) noexcept {
    double signals[6] = {fv.microprice, fv.ofi, fv.vpin,
                         fv.spread_bps, fv.realized_vol, fv.stat_arb_zscore};
    double alpha = 0.0;
    for (size_t i = 0; i < 6; ++i) { alpha += weights_[i] * signals[i]; }
    return std::clamp(alpha, -1.0, 1.0);
}
void SignalCombiner::set_weights(const double* weights, size_t count) noexcept {
    size_t n = std::min(count, size_t{6});
    std::memcpy(weights_, weights, n * sizeof(double));
}
} // namespace hft
