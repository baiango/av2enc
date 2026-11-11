#include "encoder/m2/post/bit_depth_checklist.h"

#include "encoder/m2/post/rc_adapter.h"

#include <sstream>

namespace av2enc::m2::post {
namespace {

M2ChecklistItem item(const char* id, M2ChecklistStatus status,
                     const std::string& detail) {
  M2ChecklistItem out;
  out.id = id;
  out.status = status;
  out.detail = detail;
  return out;
}

}  // namespace

const char* checklist_status_name(M2ChecklistStatus status) noexcept {
  switch (status) {
    case M2ChecklistStatus::kPass:
      return "PASS";
    case M2ChecklistStatus::kFail:
      return "FAIL";
    case M2ChecklistStatus::kNotImplemented:
      return "NOT IMPLEMENTED";
    case M2ChecklistStatus::kInfo:
      return "INFO";
  }
  return "INFO";
}

bool M2BitDepthChecklist::ok_for_header() const noexcept {
  for (const M2ChecklistItem& i : items) {
    if (i.status == M2ChecklistStatus::kFail) return false;
  }
  return true;
}

bool M2BitDepthChecklist::ok_for_8bit_m2_tile_path() const noexcept {
  for (const M2ChecklistItem& i : items) {
    if (i.status == M2ChecklistStatus::kFail ||
        i.status == M2ChecklistStatus::kNotImplemented) {
      return false;
    }
  }
  return true;
}

bool m2_geometry_supported(int width, int height, int sb_size,
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

M2BitDepthChecklist make_m2_bit_depth_checklist(int width, int height,
                                                int sb_size,
                                                fmt::BitDepth depth,
                                                int base_q_idx) {
  M2BitDepthChecklist out;
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
  out.base_q_idx_bits = m2_base_q_idx_bits(depth);
  out.max_quantizer = m2_max_quantizer(depth);

  // --- bit depth ---------------------------------------------------------
  if (depth == fmt::BitDepth::k12) {
    out.items.push_back(item(
        "bit_depth.signalable", M2ChecklistStatus::kFail,
        "AV2 v1.0.0 bit_depth_idc is one bit: 0 = 10-bit, 1 = 8-bit; "
        "12-bit is reserved. No 12-bit stream may be emitted."));
  } else {
    std::ostringstream text;
    text << "bit_depth_idc = " << out.bit_depth_idc << " ("
         << (depth == fmt::BitDepth::k10 ? "10-bit" : "8-bit") << ")";
    out.items.push_back(
        item("bit_depth.signalable", M2ChecklistStatus::kPass, text.str()));
  }

  if (out.max_quantizer > 0) {
    std::ostringstream text;
    text << "MaxQ = " << out.max_quantizer;
    out.items.push_back(item("quantizer.maxq",
                             base_q_idx >= 1 && base_q_idx <= out.max_quantizer
                                 ? M2ChecklistStatus::kPass
                                 : M2ChecklistStatus::kFail,
                             text.str()));
  } else {
    out.items.push_back(item("quantizer.maxq", M2ChecklistStatus::kFail,
                             "no conforming MaxQ for this bit depth"));
  }

  if (out.base_q_idx_bits > 0) {
    std::ostringstream text;
    text << "base_q_idx is f(" << out.base_q_idx_bits << ")";
    out.items.push_back(item(
        "header.base_q_idx_bits",
        base_q_idx >= 1 && (1 << out.base_q_idx_bits) > base_q_idx
            ? M2ChecklistStatus::kPass
            : M2ChecklistStatus::kFail,
        text.str()));
  } else {
    out.items.push_back(item("header.base_q_idx_bits",
                             M2ChecklistStatus::kFail,
                             "no conforming base_q_idx descriptor"));
  }

  // --- geometry ----------------------------------------------------------
  std::string reason;
  const bool geom_ok = m2_geometry_supported(width, height, sb_size, &reason);
  std::ostringstream geom;
  if (geom_ok) {
    geom << width << "x" << height << ", sb " << sb_size << "x" << sb_size
         << ", mi " << out.mi_cols << "x" << out.mi_rows << ", sb grid "
         << out.sb_cols << "x" << out.sb_rows
         << " (mi = 2*((dim+7)>>3), §5.18.4.4)";
  } else {
    geom << reason;
  }
  out.items.push_back(item("geometry.m2_baseline",
                           geom_ok ? M2ChecklistStatus::kPass
                                   : M2ChecklistStatus::kFail,
                           geom.str()));

  // --- chroma ------------------------------------------------------------
  out.items.push_back(item(
      "format.chroma", M2ChecklistStatus::kPass,
      "4:2:0 only (chroma_format_idc = 0); 4:2:2/4:4:4 are out of scope"));

  // --- codec path --------------------------------------------------------
  if (depth == fmt::BitDepth::k8) {
    out.items.push_back(item(
        "codec.tile_syntax", M2ChecklistStatus::kPass,
        "M2 syntax/transform/recon path is 8-bit"));
    out.items.push_back(item(
        "filters.bit_depth", M2ChecklistStatus::kPass,
        "deblock/CDEF/CCSO CPU implementations support 8-bit 4:2:0"));
  } else {
    out.items.push_back(item(
        "codec.tile_syntax", M2ChecklistStatus::kNotImplemented,
        "M2 tile syntax, transform/quantizer and reconstruction are 8-bit; "
        "10-bit M2 needs hbd transform/quant/recon (see src/encoder/m10/ for "
        "the verified 10-bit still path)"));
    out.items.push_back(item(
        "filters.bit_depth", M2ChecklistStatus::kNotImplemented,
        "deblock/CDEF/CCSO are 8-bit only; a 10-bit M2 frame must keep all "
        "loop-filter flags off until each filter has a 10-bit gate"));
    out.items.push_back(item(
        "recon.samples", M2ChecklistStatus::kNotImplemented,
        "10-bit recon needs uint16 planes with Clip1 = 0..1023, DC bias 512, "
        "and the §7.14.3/§7.15.2.1 high-bit-depth clamps"));
  }

  out.items.push_back(item(
      "cdf.slice", m2_coeff_cdf_slice_available(
                           rc::coeff_cdf_qctx(base_q_idx))
                       ? M2ChecklistStatus::kPass
                       : M2ChecklistStatus::kNotImplemented,
      "M2 ships coefficient-CDF slice 0 only (base_q_idx <= 90)"));
  out.items.push_back(item(
      "reserved.no_12bit", depth == fmt::BitDepth::k12
                               ? M2ChecklistStatus::kFail
                               : M2ChecklistStatus::kPass,
      "12-bit cannot be signalled in AV2 v1.0.0"));
  return out;
}

}  // namespace av2enc::m2::post
