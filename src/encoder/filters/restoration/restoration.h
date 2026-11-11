// AV2 in-loop restoration — CPU reference API (CCSO + loop restoration).
//
// Scope: 8-bit 4:2:0.  This is a correctness oracle, deliberately scalar and
// single-threaded (docs/architecture.md).  No bitstream syntax is parsed here;
// the caller supplies the values that §5.18.7.11 / §5.18.7.12 and
// §5.20.10.2–§5.20.10.6 derive.
//
// Implemented:
//   * CCSO (§7.19, §7.19.1), all planes, band-offset-only and edge-classified
//     modes, 7 extension filters, tile clipping, lossless segments, BRU
//     inactive regions.
//   * Loop restoration framework (§7.20–§7.20.2): 4x4 block walk, stripes
//     offset up by 8 luma lines, CurrFrame/CdefFrame sample sourcing, unit and
//     tile bookkeeping.
//   * Non-separable Wiener filter (§7.20.3), luma 32-tap and chroma 12-tap +
//     chroma-from-luma taps.
//
// Not implemented (see README.md for the normative-table blocker):
//   * PC-Wiener (§7.20.4), both the pixel classification and the filtering.
//   * GDF (§7.20.5).
//   * Film grain synthesis (§7.21.7).
//
// Output buffers never alias inputs; the reference writes LrFrame, a separate
// plane pair, exactly as the spec does.

#ifndef AV2ENC_ENCODER_FILTERS_RESTORATION_RESTORATION_H_
#define AV2ENC_ENCODER_FILTERS_RESTORATION_RESTORATION_H_

#include <cstddef>
#include <cstdint>

#include "encoder/filters/restoration/ccso_core.h"
#include "encoder/filters/restoration/wiener_core.h"

namespace av2enc::restoration {

struct RestorationPlane {
  std::uint8_t* data = nullptr;
  int stride = 0;
};

// Full-frame description shared by CCSO and loop restoration.
//
// `cdef` holds the post-CDEF samples (CCSO modifies it in place; loop
// restoration reads it).  `curr` holds the deblocked samples before CDEF.
// `lr` is the loop-restoration output (may equal `cdef` only for CCSO-only
// use; loop restoration requires distinct buffers).
struct RestorationFrame {
  RestorationPlane curr[3];
  RestorationPlane cdef[3];
  RestorationPlane lr[3];
  int mi_cols = 0;
  int mi_rows = 0;
  int subsampling_x = 1;
  int subsampling_y = 1;
  int bit_depth = 8;

  // Per-segment metadata, luma MI grid (mi_cols x mi_rows).
  const std::uint8_t* segment_ids = nullptr;
  const std::uint8_t* chroma_segment_ids = nullptr;
  const std::uint8_t* lossless = nullptr;  // kMaxSegments entries

  // Optional MiColStart/End, MiRowStart/End grids (luma MI units).  Null means
  // "single tile / whole frame".
  const std::uint16_t* mi_col_start = nullptr;
  const std::uint16_t* mi_col_end = nullptr;
  const std::uint16_t* mi_row_start = nullptr;
  const std::uint16_t* mi_row_end = nullptr;
  int disable_loopfilters_across_tiles = 0;

  // BRU: `bru_modes` is the luma MI-grid bru_mode array; `use_bru` gates the
  // loop-restoration skip (CCSO always consults the array when non-null,
  // matching §7.19.1).
  int use_bru = 0;
  const std::uint8_t* bru_modes = nullptr;

  // Superblock geometry, in luma samples and Mi_Width_Log2[SbSize].
  int sb_width = 64;
  int sb_height = 64;
  int sb_shift = 4;

  // CfL downsampling filter index used by §7.20.3 get_luma_sample().
  int cfl_ds_filter_index = 0;
};

// Result codes.  Errors are reported before any pixel is modified whenever the
// problem is knowable up front.
enum class RestorationStatus {
  kOk = 0,
  kInvalidArgument,
  kUnsupportedBitDepth,
  kUnsupportedSubsampling,
  kUnsupportedPcWiener,   // needs tables that are not in the spec text
  kUnsupportedGdf,        // needs tables that are not in the spec text
  kUnsupportedCcsoUnitSize,  // CCSO unit smaller than the superblock
  kMissingSubclassGrid,   // frame_filters_on without a §7.20.4 class map
};

const char* restoration_status_string(RestorationStatus status);

// Required plane extent helpers (derived from the MI grid, as in §7.19.1).
int restoration_plane_width(const RestorationFrame& frame, int plane);
int restoration_plane_height(const RestorationFrame& frame, int plane);

// §5.20.10.4 count_units_in_frame().
int restoration_count_units(int unit_size, int frame_size);

// ---------------------------------------------------------------------------
// CCSO
// ---------------------------------------------------------------------------

// One plane's parsed §5.18.7.12 state.
struct CcsoParams {
  bool enabled = false;
  bool bo_only = false;
  int scale_idx = 0;
  int quant_idx = 0;
  int ext_filter = 0;
  int edge_clf = 0;
  int max_band_log2 = 0;
  // CcsoFilterOffset[plane][band][cls0][cls1], flattened.  Use
  // ccso_set_offsets() to fill it from the ccso_offset_idx values.
  std::int16_t offsets[kCcsoOffsetEntries] = {};
};

struct CcsoFrameParams {
  CcsoParams plane[3];
  int ccso_luma_size_log2 = kCcsoLumaSizeLog2;
  // CcsoBlks[plane]: one byte per luma CCSO unit, row-major.
  const std::uint8_t* blks[3] = {nullptr, nullptr, nullptr};
  int blk_cols[3] = {0, 0, 0};
  int blk_rows[3] = {0, 0, 0};
};

// Number of ccso_offset_idx values §5.18.7.12 reads for one plane:
//   bo_only:             (1 << max_band_log2) * 1 * 1
//   otherwise:           (1 << max_band_log2) * maxEdgeInterval^2
// with maxEdgeInterval = 3 - edge_clf, capped at 1 for bo_only.
int ccso_offset_count(const CcsoParams& params);

// Fills `params.offsets` from the 7-bit ccso_offset_idx values in the order the
// syntax reads them.  Returns false on bad indices/counts.
bool ccso_set_offsets(CcsoParams& params, const std::uint8_t* indices,
                      std::size_t count);

// Applies §7.19 / §7.19.1 in place to `frame.cdef`.
RestorationStatus ccso_frame(const RestorationFrame& frame,
                             const CcsoFrameParams& params);

// Runs CCSO for one plane only (diagnostics/tests).  Same validation.
RestorationStatus ccso_plane_frame(const RestorationFrame& frame,
                                   const CcsoFrameParams& params, int plane);

// ---------------------------------------------------------------------------
// Loop restoration (non-separable Wiener)
// ---------------------------------------------------------------------------

struct LoopRestorationParams {
  RestorationType frame_type[3] = {kRestoreNone, kRestoreNone, kRestoreNone};
  // LoopRestorationSize[plane], in plane samples.
  int unit_size[3] = {64, 64, 64};

  // LrType[plane]: unit_rows x unit_cols bytes, frame-global unit indexing.
  // The grid is required when frame_type[plane] != RESTORE_NONE.
  const std::uint8_t* unit_type[3] = {nullptr, nullptr, nullptr};
  int unit_rows[3] = {0, 0, 0};
  int unit_cols[3] = {0, 0, 0};

  // LrWienerNs[plane][unitRow][unitCol][coeff], stride 18 per unit.
  const std::int16_t* unit_coeffs[3] = {nullptr, nullptr, nullptr};

  // Frame-level Wiener filters (§5.18.7.11 frame_filters_on).  When set, the
  // per-unit coefficient arrays are ignored.  `subclass_grid` is the luma MI
  // grid of FilterClass/SubclassLookup results; producing it requires §7.20.4
  // pixel classification, whose LUT is not in the spec text, so it must be
  // supplied by the caller.
  int frame_filters_on[3] = {0, 0, 0};
  int num_filter_classes = 1;
  const std::uint8_t* subclass_grid = nullptr;
  const std::int16_t* frame_coeffs[3] = {nullptr, nullptr, nullptr};

  // GDF is not implemented; a nonzero request fails with kUnsupportedGdf.
  int gdf_frame_enable = 0;
};

// Applies loop restoration: copies CdefFrame to LrFrame, then replaces samples
// with Wiener-filtered values where LrType selects RESTORE_WIENER_NONSEP.
RestorationStatus loop_restoration_frame(const RestorationFrame& frame,
                                         const LoopRestorationParams& params);

// Runs the block loop for one plane into `frame.lr` (the copy step included).
RestorationStatus loop_restoration_plane(const RestorationFrame& frame,
                                         const LoopRestorationParams& params,
                                         int plane);

// Builds the shared host/device core views.  Exposed for tests and for the
// CUDA launcher, which uses the same conversion.
CcsoCoreFrame ccso_to_core(const RestorationFrame& frame,
                           const CcsoFrameParams& params);
LrCoreFrame loop_restoration_to_core(const RestorationFrame& frame,
                                     const LoopRestorationParams& params);

}  // namespace av2enc::restoration

#endif  // AV2ENC_ENCODER_FILTERS_RESTORATION_RESTORATION_H_
