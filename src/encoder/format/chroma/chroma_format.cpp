#include "encoder/format/chroma/chroma_format.h"

#include "encoder/format/format.h"

#include <algorithm>
#include <sstream>

namespace av2enc::chroma {
namespace {

constexpr int kMiSize = 4;
constexpr int kMaxDimension = static_cast<int>(fmt::kMaxDimension);

bool valid_chroma(ChromaFormat chroma) {
  switch (chroma) {
    case ChromaFormat::k400:
    case ChromaFormat::k420:
    case ChromaFormat::k422:
    case ChromaFormat::k444:
      return true;
  }
  return false;
}

std::string dimension_list(int width, int height) {
  return std::to_string(width) + "x" + std::to_string(height);
}

}  // namespace

// ---------------------------------------------------------------------------
// Table 6.2
// ---------------------------------------------------------------------------

ChromaFormatInfo chroma_format_info(ChromaFormat format) {
  ChromaFormatInfo info;
  info.format = format;
  info.idc = fmt::chroma_format_idc(format);
  info.name = fmt::chroma_format_name(format);
  info.subsampling_x = fmt::subsampling_x(format);
  info.subsampling_y = fmt::subsampling_y(format);
  info.monochrome = format == ChromaFormat::k400;
  info.num_planes = fmt::num_planes(format);

  // Annex A.2: 4:0:0 and 4:2:0 are Main_420_10_IP0 (profile 0), 4:2:2 is
  // Main_422_10_IP1 (profile 3) and 4:4:4 is Main_444_10_IP1 (profile 4).
  switch (format) {
    case ChromaFormat::k400:
    case ChromaFormat::k420:
      info.profile_idc = 0;
      break;
    case ChromaFormat::k422:
      info.profile_idc = 3;
      break;
    case ChromaFormat::k444:
      info.profile_idc = 4;
      break;
  }

  // 8-bit stream: MaxQ = MAXQ_8_BITS (255) and base_q_idx is f(8).
  info.max_quantizer = fmt::max_quantizer(fmt::BitDepth::k8);
  info.base_q_idx_bits = fmt::base_q_idx_bits(fmt::BitDepth::k8);
  return info;
}

std::optional<ChromaFormatInfo> chroma_format_info_from_idc(int idc) {
  const auto format = fmt::chroma_format_from_idc(idc);
  if (!format) {
    return std::nullopt;
  }
  return chroma_format_info(*format);
}

std::optional<ChromaFormat> chroma_format_from_idc(int idc) {
  return fmt::chroma_format_from_idc(idc);
}

std::optional<ChromaFormat> chroma_format_from_name(std::string_view name) {
  if (name == "400" || name == "4:0:0" || name == "mono" ||
      name == "monochrome") {
    return ChromaFormat::k400;
  }
  if (name == "420" || name == "4:2:0") {
    return ChromaFormat::k420;
  }
  if (name == "422" || name == "4:2:2") {
    return ChromaFormat::k422;
  }
  if (name == "444" || name == "4:4:4") {
    return ChromaFormat::k444;
  }
  return std::nullopt;
}

int chroma_format_idc(ChromaFormat format) { return fmt::chroma_format_idc(format); }
int subsampling_x(ChromaFormat format) { return fmt::subsampling_x(format); }
int subsampling_y(ChromaFormat format) { return fmt::subsampling_y(format); }
bool is_monochrome(ChromaFormat format) { return format == ChromaFormat::k400; }
int num_planes(ChromaFormat format) { return fmt::num_planes(format); }

bool validate_encoder_format(const Format& format, std::string& error) {
  if (!valid_chroma(format.chroma)) {
    error = "invalid chroma format";
    return false;
  }
  if (format.depth != fmt::BitDepth::k8) {
    error = "chroma component encodes 8-bit streams only; use "
            "src/encoder/format/ + src/cuda/m1/hbd/ for high bit depth";
    return false;
  }
  if (format.width == 0 || format.height == 0) {
    error = "width and height must be non-zero";
    return false;
  }
  if (format.width > static_cast<std::uint32_t>(kMaxDimension) ||
      format.height > static_cast<std::uint32_t>(kMaxDimension)) {
    error = "dimensions exceed the implementation limit of " +
            std::to_string(kMaxDimension);
    return false;
  }
  error.clear();
  return true;
}

// ---------------------------------------------------------------------------
// ChromaMiSize
// ---------------------------------------------------------------------------

std::optional<BlockSize> chroma_mi_size(BlockSize luma, ChromaFormat chroma) {
  return fmt::chroma_residual_size(luma, chroma);
}

std::optional<BlockSize> chroma_mi_size(int luma_width, int luma_height,
                                        ChromaFormat chroma) {
  return fmt::chroma_residual_size(luma_width, luma_height, chroma);
}

bool is_chroma_offset_for_subsize(BlockSize luma, ChromaFormat chroma) {
  // §5.20.3.2 is_chroma_offset_for_subsize(subSize):
  //   if (SubsamplingY && Mi_Height_Log2[subSize] == 0) return 1
  //   if (SubsamplingX && Mi_Width_Log2[subSize] == 0) return 1
  // A 4-sample dimension with subsampling would leave a 2-sample chroma block.
  if (!valid_chroma(chroma) || is_monochrome(chroma)) {
    return false;
  }
  const int w4 = fmt::block_width(luma) / kMiSize;
  const int h4 = fmt::block_height(luma) / kMiSize;
  if (fmt::subsampling_y(chroma) != 0 && h4 == 1) {
    return true;
  }
  if (fmt::subsampling_x(chroma) != 0 && w4 == 1) {
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

int plane_origin_x(int luma_mi_col, int subsampling_x_value) {
  return (luma_mi_col >> subsampling_x_value) * kMiSize;
}

int plane_origin_y(int luma_mi_row, int subsampling_y_value) {
  return (luma_mi_row >> subsampling_y_value) * kMiSize;
}

FrameShape make_frame_shape(int width, int height, ChromaFormat chroma) {
  FrameShape shape;
  shape.width = width;
  shape.height = height;
  shape.chroma = chroma;
  shape.info = chroma_format_info(chroma);
  shape.monochrome = shape.info.monochrome;
  shape.num_planes = shape.info.num_planes;

  // §5.18.4.4 compute_image_size().
  shape.mi_cols = 2 * ((width + 7) >> 3);
  shape.mi_rows = 2 * ((height + 7) >> 3);
  shape.coded_width = shape.mi_cols * kMiSize;
  shape.coded_height = shape.mi_rows * kMiSize;
  shape.padded_width = ((width + 63) / 64) * 64;
  shape.padded_height = ((height + 63) / 64) * 64;

  for (int plane = 0; plane < 3; ++plane) {
    PlaneShape& p = shape.planes[static_cast<std::size_t>(plane)];
    p.present = plane == 0 || !shape.monochrome;
    if (!p.present) {
      continue;
    }
    p.subsampling_x = plane == 0 ? 0 : shape.info.subsampling_x;
    p.subsampling_y = plane == 0 ? 0 : shape.info.subsampling_y;
    if (plane == 0) {
      p.visible_width = width;
      p.visible_height = height;
      p.coded_width = shape.coded_width;
      p.coded_height = shape.coded_height;
      p.padded_width = shape.padded_width;
      p.padded_height = shape.padded_height;
    } else {
      // §7.21.2: visible chroma is ceil(width / 2^subX), not the coded size.
      p.visible_width =
          (width + ((1 << p.subsampling_x) - 1)) >> p.subsampling_x;
      p.visible_height =
          (height + ((1 << p.subsampling_y) - 1)) >> p.subsampling_y;
      p.coded_width = shape.coded_width >> p.subsampling_x;
      p.coded_height = shape.coded_height >> p.subsampling_y;
      p.padded_width = shape.padded_width >> p.subsampling_x;
      p.padded_height = shape.padded_height >> p.subsampling_y;
    }
  }
  return shape;
}

std::optional<FrameShape> make_frame_shape_checked(int width, int height,
                                                   ChromaFormat chroma,
                                                   std::string& error) {
  Format format;
  format.width = static_cast<std::uint32_t>(width);
  format.height = static_cast<std::uint32_t>(height);
  format.chroma = chroma;
  format.depth = fmt::BitDepth::k8;
  if (!validate_encoder_format(format, error)) {
    return std::nullopt;
  }
  return make_frame_shape(width, height, chroma);
}

bool supports_64x64_none(const FrameShape& shape, std::string& error) {
  if (shape.width <= 0 || shape.height <= 0) {
    error = "empty frame";
    return false;
  }
  if (shape.monochrome) {
    // The 64x64 leaf has no chroma residual; nothing else to check.
  } else {
    if (!chroma_mi_size(BlockSize::k64x64, shape.chroma)) {
      error = std::string("Subsampled_Size has no legal chroma residual for "
                          "BLOCK_64X64 in ") +
              shape.info.name;
      return false;
    }
  }
  const int sb_cols = (shape.mi_cols + 15) / 16;
  const int sb_rows = (shape.mi_rows + 15) / 16;
  for (int sb_row = 0; sb_row < sb_rows; ++sb_row) {
    for (int sb_col = 0; sb_col < sb_cols; ++sb_col) {
      const int mi_row = sb_row * 16;
      const int mi_col = sb_col * 16;
      // §5.20.3.2: PARTITION_NONE at 64x64 needs the half-block tests.
      if (mi_row + 8 >= shape.mi_rows || mi_col + 8 >= shape.mi_cols) {
        std::ostringstream message;
        message << "PARTITION_NONE at BLOCK_64X64 is not allowed at mi ("
                << mi_row << "," << mi_col << ") for "
                << dimension_list(shape.width, shape.height)
                << " (MiCols=" << shape.mi_cols << ", MiRows=" << shape.mi_rows
                << "); use a 64-aligned height or the M1 partition tree";
        error = message.str();
        return false;
      }
    }
  }
  error.clear();
  return true;
}

// ---------------------------------------------------------------------------
// Tool legality
// ---------------------------------------------------------------------------

ToolLegality tool_legality(ChromaFormat format) {
  const ChromaFormatInfo info = chroma_format_info(format);
  ToolLegality tools;
  tools.format = format;
  tools.monochrome = info.monochrome;
  tools.num_planes = info.num_planes;
  tools.subsampling_x = info.subsampling_x;
  tools.subsampling_y = info.subsampling_y;

  // §5.4.3: Monochrome has no chroma plane and no SDP (the bit is not even
  // coded).  A shared-partition SDP tree is meaningless without chroma.
  tools.enable_sdp = false;
  tools.sdp_syntax_present = !info.monochrome;

  // §5.4.5: cfl_ds_filter_index is absent for Monochrome; CfL is a chroma
  // predictor.  MHCCP is also chroma-only.
  tools.enable_cfl_intra = false;
  tools.cfl_available = !info.monochrome;
  tools.cfl_ds_filter_index_present = !info.monochrome;
  tools.enable_mhccp = false;
  tools.mhccp_available = !info.monochrome;

  // §5.4.8: enable_chroma_dctonly, enable_cctx, separate_uv_delta_q and the
  // whole UV delta-q block are absent when Monochrome.
  tools.enable_chroma_dctonly = false;
  tools.chroma_dctonly_syntax_present = !info.monochrome;
  tools.enable_cctx = false;
  tools.cctx_syntax_present = !info.monochrome;
  tools.separate_uv_delta_q = false;
  tools.separate_uv_delta_q_syntax_present = !info.monochrome;
  tools.uv_delta_q_syntax_present = !info.monochrome;

  // §5.4.10: the sequence-level enable_ccso bit exists for every format, but
  // per-plane chroma CCSO parameters do not exist for Monochrome.  CDEF UV
  // strengths, loop-restoration UV tools and the U/V deblock planes are
  // likewise chroma-only.
  tools.enable_ccso = false;
  tools.ccso_chroma_planes_available = !info.monochrome;
  tools.enable_cdef = false;
  tools.cdef_uv_strengths_present = !info.monochrome;
  tools.enable_restoration = false;
  tools.restoration_uv_present = !info.monochrome;
  tools.deblock_chroma_planes_present = !info.monochrome;

  // §5.20.5.3: HasChroma == NumPlanes > 1 && TreeType != LUMA_PART.  In the
  // non-SDP tree HasChroma is false exactly for Monochrome, so read_intra_uv_mode
  // (and the whole chroma residual) is absent.
  tools.uv_mode_syntax_present = !info.monochrome;
  tools.chroma_residual_present = !info.monochrome;
  tools.palette_allowed = !info.monochrome;
  tools.intrabc_allowed = !info.monochrome;
  return tools;
}

std::vector<std::string> tool_legality_report(ChromaFormat format) {
  const ChromaFormatInfo info = chroma_format_info(format);
  const ToolLegality tools = tool_legality(format);
  std::vector<std::string> lines;

  auto add = [&lines](const std::string& text) { lines.push_back(text); };

  add(std::string("format: ") + info.name + " chroma_format_idc=" +
      std::to_string(info.idc) + " SubsamplingX=" +
      std::to_string(info.subsampling_x) + " SubsamplingY=" +
      std::to_string(info.subsampling_y));
  add(std::string("Monochrome=") + (info.monochrome ? "1" : "0") +
      " NumPlanes=" + std::to_string(info.num_planes) +
      " seq_profile_idc=" + std::to_string(info.profile_idc));
  add(std::string("enable_sdp=0") +
      (tools.sdp_syntax_present ? " (bit coded)" : " (forced, bit absent)"));
  add(std::string("enable_cfl_intra=0; cfl_ds_filter_index ") +
      (tools.cfl_ds_filter_index_present ? "coded" : "forced 0, absent"));
  add(std::string("enable_mhccp=0") +
      (tools.mhccp_available ? "" : "; unavailable (no chroma plane)"));
  add(std::string("enable_chroma_dctonly=") +
      (tools.enable_chroma_dctonly ? "1" : "0") +
      (tools.chroma_dctonly_syntax_present ? " (bit coded)"
                                           : " (forced 0, bit absent)"));
  add(std::string("enable_cctx=0") +
      (tools.cctx_syntax_present ? " (bit coded)" : " (forced 0, bit absent)"));
  add(std::string("separate_uv_delta_q=0") +
      (tools.separate_uv_delta_q_syntax_present ? " (bit coded)"
                                                : " (forced 0, bit absent)"));
  add(std::string("UV delta-q signalling ") +
      (tools.uv_delta_q_syntax_present ? "present" : "absent (Monochrome)"));
  add(std::string("CCSO chroma planes ") +
      (tools.ccso_chroma_planes_available ? "may be enabled"
                                          : "forced off (Monochrome)"));
  add(std::string("CDEF UV strengths ") +
      (tools.cdef_uv_strengths_present ? "present" : "absent (Monochrome)"));
  add(std::string("loop-restoration UV tools ") +
      (tools.restoration_uv_present ? "present" : "absent (Monochrome)"));
  add(std::string("deblock U/V planes ") +
      (tools.deblock_chroma_planes_present ? "present" : "absent (Monochrome)"));
  add(std::string("read_intra_uv_mode() ") +
      (tools.uv_mode_syntax_present ? "coded" : "absent (HasChroma=0)"));
  add(std::string("chroma residual ") +
      (tools.chroma_residual_present ? "present" : "absent (HasChroma=0)"));
  return lines;
}

// ---------------------------------------------------------------------------
// Checklist
// ---------------------------------------------------------------------------

std::vector<ChecklistItem> chroma_checklist(ChromaFormat format) {
  const ChromaFormatInfo info = chroma_format_info(format);
  const ToolLegality tools = tool_legality(format);
  std::vector<ChecklistItem> items;

  auto add = [&items](std::string area, std::string id, std::string spec,
                      std::string requirement) {
    items.push_back({std::move(area), std::move(id), std::move(spec),
                     std::move(requirement)});
  };

  add("header", "chroma_format_idc", "§5.4.1, §6.4.1 Table 6.2",
      "write uvlc(" + std::to_string(info.idc) + ") for " + info.name +
          "; do not cast the semantic enum (4:4:4 = 2, 4:2:2 = 3)");
  add("header", "set_chroma_format_and_bit_depth", "§5.4.1",
      "derive SubsamplingX=" + std::to_string(info.subsampling_x) +
          ", SubsamplingY=" + std::to_string(info.subsampling_y) +
          ", Monochrome=" + (info.monochrome ? "1" : "0") +
          ", NumPlanes=" + std::to_string(info.num_planes));
  add("header", "seq_profile_idc", "Annex A.2",
      "seq_profile_idc = " + std::to_string(info.profile_idc));
  add("header", "Monochrome conditionals", "§5.4.3, §5.4.5, §5.4.8",
      info.monochrome
          ? "omit enable_sdp, cfl_ds_filter_index, enable_chroma_dctonly, "
            "enable_cctx, separate_uv_delta_q and the UV delta-q fields; "
            "Monochrome is inferred from chroma_format_idc == 1"
          : "the Monochrome-only fields are present (write zero)");
  add("header", "enable_sdp", "§5.4.3",
      tools.sdp_syntax_present ? "enable_sdp bit is coded (write 0)"
                               : "Monochrome: enable_sdp is inferred 0");
  add("header", "cfl/mhccp", "§5.4.5",
      "enable_cfl_intra and enable_mhccp have no effect without a chroma "
      "plane; keep both 0 for Monochrome");
  add("geometry", "plane count", "§6.4.1",
      "NumPlanes = " + std::to_string(info.num_planes));
  add("geometry", "mi grid", "§5.18.4.4",
      "MiCols = 2*((width+7)>>3), MiRows likewise; coded luma = MiCols*4 by "
      "MiRows*4");
  add("geometry", "visible chroma", "§7.21.2",
      "visible chroma is ceil(width / 2^SubsamplingX) by "
      "ceil(height / 2^SubsamplingY)");
  add("geometry", "ChromaMiSize", "§5.20.3.2",
      "BLOCK_64X64 -> " +
          (info.monochrome
               ? std::string("BLOCK_INVALID (no chroma)")
               : std::string(fmt::block_size_name(*chroma_mi_size(
                     BlockSize::k64x64, format)))) +
          "; ChromaMiSize != MiSize only when a subsampled plane would fall "
          "below 4 samples");
  add("geometry", "4:2:2 residual", "§5.20.7.26 Table 6.2",
      "4:2:2 halves width only (SubsamplingY=0); 4:4:4 is full resolution; "
      "monochrome has no residual plane");
  add("prediction", "uv mode syntax", "§5.20.5.3",
      info.monochrome ? "read_intra_uv_mode()/CfL alphas are absent"
                      : "read_intra_uv_mode() is coded after the luma mode");
  add("prediction", "DC bias", "§7.13.2.10",
      "8-bit neutral value is 128; chroma prediction uses the plane's own "
      "subsampled origin (ChromaMiRow>>subY)*4");
  add("residual", "TX size", "§5.20.7.25",
      "get_tx_size(plane>0) = Max_Tx_Size_Rect[get_plane_residual_size("
      "ChromaMiSize, plane)]; 32x64/64x64 use Adjusted_Tx_Size 32x32");
  add("residual", "dequant clip", "§7.14.3",
      "8-bit dequant clip is ±(1<<15)-1 (shared with 4:2:0)");
  add("residual", "inverse transform clip", "§7.15.4",
      "row clip ±(1<<15)-1, column clip -256..255; Transform_Shift is "
      "bit-depth independent");
  add("entropy", "coefficient alphabet", "§5.20.7.27",
      "plane/tx-size dependent CDFs are the same for 4:2:2/4:4:4; only the "
      "plane dimensions and txSzCtx change");
  add("entropy", "txb skip context", "§8.3.2",
      "chroma maxX4 = MiCols>>subX, maxY4 = MiRows>>subY; 4:2:2 halves only "
      "the column bound");
  add("filter", "chroma planes", "§7.17-§7.20",
      info.monochrome ? "no U/V deblock/CDEF/CCSO/restoration planes"
                      : "filter chroma at its subsampled origin; 4:2:2 has "
                        "full-height chroma, 4:4:4 has full-resolution chroma");
  return items;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const char* block_size_name(BlockSize size) {
  return fmt::block_size_name(size);
}
int block_width(BlockSize size) { return fmt::block_width(size); }
int block_height(BlockSize size) { return fmt::block_height(size); }
bool is_valid_block_size(BlockSize size) {
  const int index = static_cast<int>(size);
  return index >= 0 && index < 29;
}

}  // namespace av2enc::chroma
