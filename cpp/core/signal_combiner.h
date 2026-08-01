#pragma once
#include "types.h"
namespace hft {
class SignalCombiner {
public:
    SignalCombiner() noexcept = default;
    double combine(const FeatureVector& fv) noexcept;
    void set_weights(const double* weights, size_t count) noexcept;
private:
    double weights_[6] = {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6};
};
} // namespace hft
