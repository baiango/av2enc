// Chroma-format model for the AV2 encoder: 4:2:0, 4:2:2, 4:4:4 and 4:0:0.
//
// Spec-only reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (commit 38f28e7).  The clauses that constrain this file are:
//
//   §5.4.1        sequence_header_obu(): chroma_format_idc, bit_depth_idc,
//                 set_chroma_format_and_bit_depth()
//   §6.4.1 Table 6.2  chroma_format_idc -> SubsamplingX/Y, Monochrome
//   §5.4.3        sequence_partition_config(): Monochrome forces enable_sdp = 0
//   §5.4.5        sequence_intra_config(): Monochrome removes
//                 cfl_ds_filter_index; CfL/MHCCP have no chroma plane
//   §5.4.8        sequence_transform_quant_entropy_config(): Monochrome removes
//                 enable_chroma_dctonly, enable_cctx, separate_uv_delta_q and
//                 the UV delta-q signalling
//   §5.20.3.2     ChromaMiSize, is_chroma_offset_for_subsize()
//   §5.20.5.3     HasChroma: intra_frame_mode_info() reads read_intra_uv_mode()
//                 only when a chroma plane exists
//   §5.20.7.25/26 get_tx_size() / get_plane_residual_size(): Subsampled_Size
//   §7.21.2       visible chroma = ceil(width / 2^SubsamplingX)
//
// The numeric chroma_format_idc values are deliberately not the enum order:
//   0 = 4:2:0, 1 = 4:0:0 (monochrome), 2 = 4:4:4, 3 = 4:2:2.
//
// This component owns *format/geometry/tool legality*.  It does not modify the
// M1/M2 cores and it does not write the bitstream by itself; `chroma_headers.h`
// and `chroma_still.h` do that, reusing the existing bit writer, OBU framing,
// entropy coder and transform kernels.
#pragma once

#include "formats/format.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace av2enc::chroma {

// Semantic aliases.  `fmt::ChromaFormat` is the existing semantic enum
// (k400/k420/k422/k444); the spec idc mapping is in `chroma_format_idc()`.
using Format = fmt::Format;
using ChromaFormat = fmt::ChromaFormat;
using BlockSize = fmt::BlockSize;

// ---------------------------------------------------------------------------
// Table 6.2 / set_chroma_format_and_bit_depth()
// ---------------------------------------------------------------------------

struct ChromaFormatInfo {
  ChromaFormat format = ChromaFormat::k420;
  int idc = 0;                 // ChromaFormatIdc (not the enum ordinal)
  const char* name = "4:2:0";
  int subsampling_x = 1;       // SubsamplingX
  int subsampling_y = 1;       // SubsamplingY
  bool monochrome = false;     // Monochrome
  int num_planes = 3;          // NumPlanes
  int profile_idc = 0;         // Annex A.2 named profile
  int max_quantizer = 255;     // MaxQ, 8-bit stream
  int base_q_idx_bits = 8;     // §5.18.6.1
};

ChromaFormatInfo chroma_format_info(ChromaFormat format);
std::optional<ChromaFormatInfo> chroma_format_info_from_idc(int idc);
std::optional<ChromaFormat> chroma_format_from_idc(int idc);
std::optional<ChromaFormat> chroma_format_from_name(std::string_view name);

int chroma_format_idc(ChromaFormat format);
int subsampling_x(ChromaFormat format);
int subsampling_y(ChromaFormat format);
bool is_monochrome(ChromaFormat format);
int num_planes(ChromaFormat format);

// Semantic validation for the encoder.  This component encodes 8-bit only
// (the 10-bit/12-bit path lives in `src/encoder/format/` and `src/cuda/m1/hbd/`).
bool validate_encoder_format(const Format& format, std::string& error);

// ---------------------------------------------------------------------------
// ChromaMiSize and ChromaMiRow/Col
// ---------------------------------------------------------------------------
//
// The core is ChromaMiSize (§5.20.3.2): the luma block size whose residual is
// shared by the chroma block.  It differs from MiSize only when a subsampled
// dimension would shrink below 4 samples (is_chroma_offset_for_subsize()).
// All blocks this component writes are square/rectangular leaves whose chroma
// residual is a legal `Subsampled_Size` entry, so ChromaMiSize == luma size.
//
// `chroma_mi_size()` returns nullopt for:
//   * monochrome (no chroma residual), or
//   * a BLOCK_INVALID Subsampled_Size entry (for example 4x8 luma in 4:2:2).
std::optional<BlockSize> chroma_mi_size(BlockSize luma, ChromaFormat chroma);
std::optional<BlockSize> chroma_mi_size(int luma_width, int luma_height,
                                        ChromaFormat chroma);

// True when the chroma residual would be smaller than 4x4 in a subsampled
// direction, i.e. is_chroma_offset_for_subsize(subSize) == 1.  For the 64x64
// leaf used by the minimal still writer this is always false.
bool is_chroma_offset_for_subsize(BlockSize luma, ChromaFormat chroma);

// ---------------------------------------------------------------------------
// Plane / frame geometry
// ---------------------------------------------------------------------------
//
// Three sizes are reported:
//   visible  the cropped output rectangle.  Chroma is ceil(width / 2^subX).
//   coded    MiCols*4 / MiRows*4 for luma, shifted right by the subsampling
//            factors for chroma (§5.18.4.4, §7.21.2).
//   padded   64x64-superblock aligned.  The minimal still writer codes one
//            PARTITION_NONE 64x64 block per superblock, so its recon/source
//            buffers are padded to this size and edge reads clamp to visible.
struct PlaneShape {
  bool present = false;
  int visible_width = 0;
  int visible_height = 0;
  int coded_width = 0;
  int coded_height = 0;
  int padded_width = 0;
  int padded_height = 0;
  int subsampling_x = 0;
  int subsampling_y = 0;
};

struct FrameShape {
  int width = 0;   // visible luma
  int height = 0;
  ChromaFormat chroma = ChromaFormat::k420;
  ChromaFormatInfo info{};
  int mi_cols = 0;
  int mi_rows = 0;
  int coded_width = 0;
  int coded_height = 0;
  int padded_width = 0;
  int padded_height = 0;
  int num_planes = 3;
  bool monochrome = false;
  std::array<PlaneShape, 3> planes{};
};

FrameShape make_frame_shape(int width, int height, ChromaFormat chroma);
std::optional<FrameShape> make_frame_shape_checked(int width, int height,
                                                   ChromaFormat chroma,
                                                   std::string& error);

// The minimal still writer codes PARTITION_NONE at 64x64 for every superblock.
// §5.20.3.2 allows that only when the half-block tests stay inside the frame:
//   has_rows = mi_row + 8 < MiRows,  has_cols = mi_col + 8 < MiCols
// and §5.20.3.2 requires a legal Subsampled_Size for the chroma residual.
// 64x64 and 640x360 both satisfy it; geometries that do not are rejected with
// a precise reason instead of emitting an invalid partition.
bool supports_64x64_none(const FrameShape& shape, std::string& error);

// Sample position of a luma MI coordinate in a plane (ChromaMiRow>>subX)*4.
int plane_origin_x(int luma_mi_col, int subsampling_x);
int plane_origin_y(int luma_mi_row, int subsampling_y);

// ---------------------------------------------------------------------------
// Tool legality per chroma format
// ---------------------------------------------------------------------------
//
// Everything below is derived from the §5.4 syntax conditions.  "syntax
// present" means a bit/field exists in the sequence header for that format;
// "available" means the tool can affect a chroma plane.
//
// The minimal still writer disables every optional tool, so the values here
// are also the defaults it feeds to the header writer.  They are exposed
// separately because the M1/M2 integration will need them.
struct ToolLegality {
  ChromaFormat format = ChromaFormat::k420;
  bool monochrome = false;
  int num_planes = 3;
  int subsampling_x = 1;
  int subsampling_y = 1;

  // §5.4.3 sequence_partition_config().
  bool enable_sdp = false;                    // forced 0 for Monochrome
  bool sdp_syntax_present = true;             // absent for Monochrome
  bool enable_extended_sdp = false;
  bool enable_ext_partitions = false;
  bool reduce_pb_aspect_ratio = false;

  // §5.4.5 sequence_intra_config().
  bool enable_cfl_intra = false;              // no chroma plane in Monochrome
  bool cfl_available = true;
  bool cfl_ds_filter_index_present = true;    // absent for Monochrome
  bool enable_mhccp = false;                  // chroma-only tool
  bool mhccp_available = true;
  bool enable_ibp = false;

  // §5.4.8 sequence_transform_quant_entropy_config().
  bool enable_chroma_dctonly = false;
  bool chroma_dctonly_syntax_present = true;  // absent for Monochrome
  bool enable_cctx = false;
  bool cctx_syntax_present = true;            // absent for Monochrome
  bool separate_uv_delta_q = false;
  bool separate_uv_delta_q_syntax_present = true;  // absent for Monochrome
  bool uv_delta_q_syntax_present = true;      // absent for Monochrome

  // §5.4.10 sequence_filter_config() and per-plane tool parameters.
  bool enable_ccso = false;
  bool ccso_chroma_planes_available = true;   // false for Monochrome
  bool enable_cdef = false;
  bool cdef_uv_strengths_present = true;      // false for Monochrome
  bool enable_restoration = false;
  bool restoration_uv_present = true;         // false for Monochrome
  bool deblock_chroma_planes_present = true;  // false for Monochrome

  // §5.20.5.3 mode_info(): read_intra_uv_mode() is coded only when HasChroma.
  bool uv_mode_syntax_present = true;         // false for Monochrome
  bool chroma_residual_present = true;        // false for Monochrome
  bool palette_allowed = true;                // no chroma planes in Monochrome
  bool intrabc_allowed = true;
};

ToolLegality tool_legality(ChromaFormat format);

// Human-readable legality table, one line per constraint.  Tests pin the
// Monochrome-forced entries; the README prints this table.
std::vector<std::string> tool_legality_report(ChromaFormat format);

// ---------------------------------------------------------------------------
// Checklist (documentation as data, mirroring fmt::format_checklist)
// ---------------------------------------------------------------------------

struct ChecklistItem {
  std::string area;       // "header", "geometry", "prediction", "residual",
                          // "entropy", "filter"
  std::string id;
  std::string spec;       // clause reference
  std::string requirement;
};

std::vector<ChecklistItem> chroma_checklist(ChromaFormat format);

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const char* block_size_name(BlockSize size);
int block_width(BlockSize size);
int block_height(BlockSize size);
bool is_valid_block_size(BlockSize size);

// Number of 4x4 units in a plane dimension.
inline int mi_units(int pixels) { return pixels >> 2; }

}  // namespace av2enc::chroma
