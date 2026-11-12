#include "pipeline/video/post/bit_depth_checklist.h"

#include "pipeline/video/post/rc_adapter.h"

#include <sstream>

namespace av2enc::video::post {
namespace {

VideoChecklistItem item(const char* id, VideoChecklistStatus status,
                     const std::string& detail) {
  VideoChecklistItem out;
  out.id = id;
  out.status = status;
  out.detail = detail;
  return out;
}

}  // namespace

const char* checklist_status_name(VideoChecklistStatus status) noexcept {
  switch (status) {
    case VideoChecklistStatus::kPass:
      return "PASS";
    case VideoChecklistStatus::kFail:
      return "FAIL";
    case VideoChecklistStatus::kNotImplemented:
      return "NOT IMPLEMENTED";
    case VideoChecklistStatus::kInfo:
      return "INFO";
  }
  return "INFO";
}

bool VideoBitDepthChecklist::ok_for_header() const noexcept {
  for (const VideoChecklistItem& i : items) {
    if (i.status == VideoChecklistStatus::kFail) return false;
  }
  return true;
}

bool VideoBitDepthChecklist::ok_for_8bit_video_tile_path() const noexcept {
  for (const VideoChecklistItem& i : items) {
    if (i.status == VideoChecklistStatus::kFail ||
        i.status == VideoChecklistStatus::kNotImplemented) {
      return false;
    }
  }
  return true;
}

bool video_geometry_supported(int width, int height, int sb_size,
                           std::string* reason) {
  const auto fail = [reason](const std::string& text) {
    if (reason != nullptr) *reason = text;
    return false;
  };
  if (width <= 0 || height <= 0) return fail("frame dimensions must be > 0");
  if (width > 65535 || height > 65535) {
    return fail("frame dimensions exceed the uvlc range");
  }
  if (sb_size != 64 && sb_size != 128) {
    return fail("M2 supports 64x64 or 128x128 superblocks");
  }
  if ((width & 1) != 0 || (height & 1) != 0) {
    return fail("M2 4:2:0 requires even dimensions");
  }
  if ((width & 7) != 0 || (height & 7) != 0) {
    return fail("M2 baseline requires dimensions divisible by 8");
  }
  if (reason != nullptr) reason->clear();
  return true;
}

VideoBitDepthChecklist make_video_bit_depth_checklist(int width, int height,
                                                int sb_size,
                                                fmt::BitDepth depth,
                                                int base_q_idx) {
  VideoBitDepthChecklist out;
  out.depth = depth;
  out.width = width;
  out.height = height;
  out.sb_size = sb_size;
  out.base_q_idx = base_q_idx;
  out.mi_cols = 2 * ((width + 7) >> 3);
  out.mi_rows = 2 * ((height + 7) >> 3);
  const int sb_mi = sb_size / 4;
  out.sb_cols = sb_mi > 0 ? (out.mi_cols + sb_mi - 1) / sb_mi : 0;
  out.sb_rows = sb_mi > 0 ? (out.mi_rows + sb_mi - 1) / sb_mi : 0;
  out.bit_depth_idc = fmt::bit_depth_idc(depth);
  out.base_q_idx_bits = video_base_q_idx_bits(depth);
  out.max_quantizer = video_max_quantizer(depth);

  // --- bit depth ---------------------------------------------------------
  if (depth == fmt::BitDepth::k12) {
    out.items.push_back(item(
        "bit_depth.signalable", VideoChecklistStatus::kFail,
        "AV2 v1.0.0 bit_depth_idc is one bit: 0 = 10-bit, 1 = 8-bit; "
        "12-bit is reserved. No 12-bit stream may be emitted."));
  } else {
    std::ostringstream text;
    text << "bit_depth_idc = " << out.bit_depth_idc << " ("
         << (depth == fmt::BitDepth::k10 ? "10-bit" : "8-bit") << ")";
    out.items.push_back(
        item("bit_depth.signalable", VideoChecklistStatus::kPass, text.str()));
  }

  if (out.max_quantizer > 0) {
    std::ostringstream text;
    text << "MaxQ = " << out.max_quantizer;
    out.items.push_back(item("quantizer.maxq",
                             base_q_idx >= 1 && base_q_idx <= out.max_quantizer
                                 ? VideoChecklistStatus::kPass
                                 : VideoChecklistStatus::kFail,
                             text.str()));
  } else {
    out.items.push_back(item("quantizer.maxq", VideoChecklistStatus::kFail,
                             "no conforming MaxQ for this bit depth"));
  }

  if (out.base_q_idx_bits > 0) {
    std::ostringstream text;
    text << "base_q_idx is f(" << out.base_q_idx_bits << ")";
    out.items.push_back(item(
        "header.base_q_idx_bits",
        base_q_idx >= 1 && (1 << out.base_q_idx_bits) > base_q_idx
            ? VideoChecklistStatus::kPass
            : VideoChecklistStatus::kFail,
        text.str()));
  } else {
    out.items.push_back(item("header.base_q_idx_bits",
                             VideoChecklistStatus::kFail,
                             "no conforming base_q_idx descriptor"));
  }

  // --- geometry ----------------------------------------------------------
  std::string reason;
  const bool geom_ok = video_geometry_supported(width, height, sb_size, &reason);
  std::ostringstream geom;
  if (geom_ok) {
    geom << width << "x" << height << ", sb " << sb_size << "x" << sb_size
         << ", mi " << out.mi_cols << "x" << out.mi_rows << ", sb grid "
         << out.sb_cols << "x" << out.sb_rows
         << " (mi = 2*((dim+7)>>3), §5.18.4.4)";
  } else {
    geom << reason;
  }
  out.items.push_back(item("geometry.video_baseline",
                           geom_ok ? VideoChecklistStatus::kPass
                                   : VideoChecklistStatus::kFail,
                           geom.str()));

  // --- chroma ------------------------------------------------------------
  out.items.push_back(item(
      "format.chroma", VideoChecklistStatus::kPass,
      "4:2:0 only (chroma_format_idc = 0); 4:2:2/4:4:4 are out of scope"));

  // --- codec path --------------------------------------------------------
  if (depth == fmt::BitDepth::k8) {
    out.items.push_back(item(
        "codec.tile_syntax", VideoChecklistStatus::kPass,
        "M2 syntax/transform/recon path is 8-bit"));
    out.items.push_back(item(
        "filters.bit_depth", VideoChecklistStatus::kPass,
        "deblock/CDEF/CCSO CPU implementations support 8-bit 4:2:0"));
  } else {
    out.items.push_back(item(
        "codec.tile_syntax", VideoChecklistStatus::kNotImplemented,
        "M2 tile syntax, transform/quantizer and reconstruction are 8-bit; "
        "10-bit M2 needs hbd transform/quant/recon (see src/encoder/m10/ for "
        "the verified 10-bit still path)"));
    out.items.push_back(item(
        "filters.bit_depth", VideoChecklistStatus::kNotImplemented,
        "deblock/CDEF/CCSO are 8-bit only; a 10-bit M2 frame must keep all "
        "loop-filter flags off until each filter has a 10-bit gate"));
    out.items.push_back(item(
        "recon.samples", VideoChecklistStatus::kNotImplemented,
        "10-bit recon needs uint16 planes with Clip1 = 0..1023, DC bias 512, "
        "and the §7.14.3/§7.15.2.1 high-bit-depth clamps"));
  }

  out.items.push_back(item(
      "cdf.slice", video_coeff_cdf_slice_available(
                           rc::coeff_cdf_qctx(base_q_idx))
                       ? VideoChecklistStatus::kPass
                       : VideoChecklistStatus::kNotImplemented,
      "M2 uses the shared four-slice coefficient-CDF tables (§8.3.1 qctx from "
      "raw base_q_idx)"));
  out.items.push_back(item(
      "reserved.no_12bit", depth == fmt::BitDepth::k12
                               ? VideoChecklistStatus::kFail
                               : VideoChecklistStatus::kPass,
      "12-bit cannot be signalled in AV2 v1.0.0"));
  return out;
}

}  // namespace av2enc::video::post
