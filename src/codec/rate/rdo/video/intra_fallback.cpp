#include "codec/rate/rdo/video/intra_fallback.h"

#include "codec/rate/decide/transform_stream.h"
#include "pipeline/video/video_syntax.h"

#include <array>
#include <cstddef>

namespace av2enc::video::rdo {

bool intra_coefficient_bits(std::span<const IntraPlaneRateInput> planes,
                            double* bits, std::string* error) {
  if (bits == nullptr) {
    if (error != nullptr) *error = "null bit output";
    return false;
  }
  *bits = 0.0;
  if (planes.empty()) {
    if (error != nullptr) *error = "no intra planes";
    return false;
  }

  constexpr int kMaxArea = av2enc::still::rdo_decide::kMaxTransformArea;
  std::array<std::int32_t, kMaxArea> levels_scan{};
  std::array<std::uint16_t, kMaxArea> scan{};

  double total = 0.0;
  for (const IntraPlaneRateInput& plane : planes) {
    const int area = plane.adj_w * plane.adj_h;
    if (area <= 0 || area > kMaxArea) {
      if (error != nullptr) *error = "intra plane transform area is invalid";
      return false;
    }
    if (plane.levels_raster.size() <
        static_cast<std::size_t>(area)) {
      if (error != nullptr) *error = "intra plane level span is truncated";
      return false;
    }
    if (plane.eob < 0 || plane.eob > area) {
      if (error != nullptr) *error = "intra plane EOB is invalid";
      return false;
    }
    av2enc::video::syntax::build_scan(
        plane.adj_w, plane.adj_h,
        std::span<std::uint16_t>(scan.data(), static_cast<std::size_t>(area)));
    for (int c = 0; c < area; ++c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      levels_scan[static_cast<std::size_t>(c)] =
          plane.levels_raster[static_cast<std::size_t>(pos)];
    }

    av2enc::still::rdo_decide::TransformStreamBlock block;
    block.plane = plane.plane;
    block.tx_size = plane.tx_size;
    block.w = plane.adj_w;
    block.h = plane.adj_h;
    block.txb_skip_context = 0;
    block.dc_sign_context = 0;
    block.levels_scan =
        std::span<const std::int32_t>(levels_scan.data(),
                                      static_cast<std::size_t>(area));
    block.eob = plane.eob;

    av2enc::still::rdo_decide::TransformStreamRate stream;
    if (!stream.init(block, error)) return false;
    total += stream.block_bits();
  }

  *bits = total;
  return true;
}

}  // namespace av2enc::video::rdo
