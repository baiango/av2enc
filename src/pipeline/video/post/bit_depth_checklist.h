// M2 10-bit geometry/header checklist (8-bit and 10-bit only; no 12-bit).
//
// This is a planning/validation layer, not a 10-bit M2 codec.  AV2 v1.0.0
// signals exactly two bit depths: bit_depth_idc 1 = 8-bit and 0 = 10-bit
// (§6.4.1 Table 6.3).  A value above 1 is reserved, so no conforming stream may
// signal 12-bit; fmt::BitDepth::k12 exists only for internal buffers.
//
// What is ready in this directory:
//   * sequence/frame header bit depth fields (bit_depth_idc, f(9) base_q_idx);
//   * 10-bit geometry and tile-layout checks that share the exact MI/SB
//     formulas the 8-bit writer uses;
//   * an explicit list of the parts that are NOT 10-bit capable yet, so an
//     integrator cannot accidentally claim a 10-bit M2 stream.
//
// What is not ready: the M2 tile syntax writer, transform/quantizer and
// reconstruction are 8-bit (they reuse the M1 8-bit tables).  M10
// (src/encoder/m10/) is the only verified 10-bit bitstream path and is a still
// picture.  Do not route M2 tile payloads through the 10-bit header and call the
// result a 10-bit encoder.
#pragma once

#include "formats/format.h"

#include <string>
#include <vector>

namespace av2enc::video::post {

enum class VideoChecklistStatus : std::uint8_t {
  kPass = 0,
  kFail,
  kNotImplemented,
  kInfo,
};

const char* checklist_status_name(VideoChecklistStatus status) noexcept;

struct VideoChecklistItem {
  std::string id;
  VideoChecklistStatus status = VideoChecklistStatus::kInfo;
  std::string detail;
};

struct VideoBitDepthChecklist {
  fmt::BitDepth depth = fmt::BitDepth::k8;
  int width = 0;
  int height = 0;
  int sb_size = 64;
  int base_q_idx = 90;
  int mi_cols = 0;
  int mi_rows = 0;
  int sb_cols = 0;
  int sb_rows = 0;
  int bit_depth_idc = -1;
  int base_q_idx_bits = 0;
  int max_quantizer = 0;
  std::vector<VideoChecklistItem> items;

  bool ok_for_header() const noexcept;
  bool ok_for_8bit_video_tile_path() const noexcept;
};

// §5.18.4.4 geometry: mi_cols = 2 * ((width + 7) >> 3), same for rows, so the
// coded picture is rounded up to 8 pixels per axis.  This 8-bit 4:2:0 writer
// only requires even dimensions; the MI-aligned padded region is reconstructed
// and cropped back to the visible size on output.
// Returns false with `reason` on a geometry the current writer cannot code.
bool video_geometry_supported(int width, int height, int sb_size,
                           std::string* reason = nullptr);

// Build the checklist from the frame geometry, format and frame qindex.
// `depth == k12` is always a failure (reserved in AV2 v1.0.0).
VideoBitDepthChecklist make_video_bit_depth_checklist(int width, int height,
                                                int sb_size,
                                                fmt::BitDepth depth,
                                                int base_q_idx);

}  // namespace av2enc::video::post
