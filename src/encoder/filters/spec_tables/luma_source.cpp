#include "encoder/filters/spec_tables/luma_source.h"

#include <cstddef>

#include "encoder/filters/spec_tables/spec_math.h"

namespace av2enc::filters::spec {

std::uint16_t luma_source_sample(const LumaSource& src, int x, int y) {
  x = Clip3(src.luma_start_x, src.luma_end_x, x);
  y = Clip3(src.luma_start_y, src.luma_end_y, y);
  if (y < src.luma_stripe_start_y) {
    y = src.luma_stripe_start_y - 2 > y ? src.luma_stripe_start_y - 2 : y;
    return src.curr[static_cast<std::ptrdiff_t>(y) * src.curr_stride + x];
  }
  if (y > src.luma_stripe_end_y) {
    y = src.luma_stripe_end_y + 2 < y ? src.luma_stripe_end_y + 2 : y;
    return src.curr[static_cast<std::ptrdiff_t>(y) * src.curr_stride + x];
  }
  return src.cdef[static_cast<std::ptrdiff_t>(y) * src.cdef_stride + x];
}

}  // namespace av2enc::filters::spec
