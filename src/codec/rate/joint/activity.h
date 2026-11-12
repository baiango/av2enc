// Source activity measurement for the per-CTU lambda.
//
// The measure is the mean absolute deviation from the block mean, in Q8 units
// (sample units * 256).  It is integer-only and deterministic, which matters
// because the CTU lambda must reproduce exactly on CPU and GPU.  A caller that
// does not want adaptive lambda passes strength 0 and never reads this value.
//
// This is a rate-control hint, not a quality metric.

#pragma once

#include <cstdint>

namespace av2enc::still::rdo_loop {

// Mean absolute deviation from the block mean, Q8.  Reads only the visible
// w x h region.  Returns 0 for an empty region.
std::uint32_t luma_activity_q8(const std::uint8_t* plane, int stride, int x,
                               int y, int w, int h) noexcept;

// Frame mean of a list of CTU activities, rounded to nearest.  `count` must be
// > 0.  Kept as a helper so the adapter and the experiment agree on the
// denominator.
std::uint32_t mean_activity_q8(const std::uint32_t* activities, int count)
    noexcept;

}  // namespace av2enc::still::rdo_loop
