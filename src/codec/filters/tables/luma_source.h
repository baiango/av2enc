// §7.20.2 get source sample process (luma).
//
// Both §7.20.4 (PC-Wiener) and §7.20.5 (GDF) read luma through this process.
// Samples inside the current 64-line stripe come from CdefFrame (post
// CDEF/CCSO); samples outside come from CurrFrame (pre CDEF/CCSO) with the
// two-line clamp.

#ifndef AV2ENC_ENCODER_FILTERS_SPEC_TABLES_LUMA_SOURCE_H_
#define AV2ENC_ENCODER_FILTERS_SPEC_TABLES_LUMA_SOURCE_H_

#include <cstdint>

namespace av2enc::filters::spec {

// The §7.20.1 variables that §7.20.2 needs, restricted to plane 0.
struct LumaSource {
  const std::uint16_t* curr = nullptr;  // CurrFrame[0]
  const std::uint16_t* cdef = nullptr;  // CdefFrame[0]
  int curr_stride = 0;
  int cdef_stride = 0;
  int luma_start_x = 0;
  int luma_end_x = 0;
  int luma_start_y = 0;
  int luma_end_y = 0;
  int luma_stripe_start_y = 0;
  int luma_stripe_end_y = 0;
};

// §7.20.2 get_source_sample(0, x, y).
std::uint16_t luma_source_sample(const LumaSource& src, int x, int y);

}  // namespace av2enc::filters::spec

#endif  // AV2ENC_ENCODER_FILTERS_SPEC_TABLES_LUMA_SOURCE_H_
