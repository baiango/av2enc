// AV2 §7.18 CDEF — CPU reference API.
//
// Implements the normative Constrained Directional Enhancement Filter for
// 8-bit 4:2:0 reconstructed frames.  Spec-only: this component filters planes
// and reports decisions.  It performs no bitstream I/O and is not wired into
// the encoder pipeline; a later integration pass owns that.
//
// The caller owns all memory.  Source and destination are separate planes:
// CDEF reads every tap from the unfiltered source frame (`CurrFrame`) and writes
// only the filtered frame (`CdefFrame`), so a block never observes another
// block's output.  This also means the component can be run in any block order,
// which the CUDA kernel relies on.

#ifndef AV2ENC_ENCODER_FILTERS_CDEF_CDEF_H_
#define AV2ENC_ENCODER_FILTERS_CDEF_CDEF_H_

#include <cstdint>

#include "encoder/filters/cdef/cdef_core.h"

namespace av2enc::cdef {

enum class CdefStatus {
  kOk,
  kInvalidArgument,
  kUnsupportedBitDepth,
  kUnsupportedSubsampling,
};

const char* cdef_status_string(CdefStatus status);

// One CDEF strength set (cdef_y_pri_strength[i] etc., §5.18.7.10).
//   y_pri / uv_pri: 0..15
//   y_sec / uv_sec: 0, 1, 2 or 4 (the syntax value 3 is mapped to 4)
struct CdefStrength {
  int y_pri = 0;
  int y_sec = 0;
  int uv_pri = 0;
  int uv_sec = 0;
};

struct CdefOptions {
  // CdefDamping = cdef_damping_minus_3 + 3, range 3..6.
  int damping = 3;
  // CdefStrengths = cdef_strengths_minus_1 + 1, range 1..8.
  int num_strengths = 1;
  CdefStrength strengths[kMaxStrengths] = {};
  // cdef_on_skip_txfm_frame_enable: when false, an 8x8 block whose four MI
  // cells are all skipped is not filtered.
  bool on_skip_txfm = false;
  // disable_loopfilters_across_tiles: when true, taps outside the block's tile
  // are unavailable (is_inside_filter_region uses the tile bounds).
  bool disable_loopfilters_across_tiles = false;
  // Only 8 is supported; coeffShift = bit_depth - 8 is therefore zero.
  int bit_depth = 8;
};

struct CdefPlane {
  const std::uint8_t* src = nullptr;  // CurrFrame (unfiltered)
  std::uint8_t* dst = nullptr;        // CdefFrame (filtered / copied)
  int stride = 0;
};

// Per-frame CDEF inputs.  All grids are in luma MI (4x4) units and indexed
// row-major as `row * mi_cols + col`.
struct CdefFrame {
  CdefPlane planes[3];
  int mi_cols = 0;  // must be even
  int mi_rows = 0;  // must be even
  int subsampling_x = 1;
  int subsampling_y = 1;

  // Skips[mi] (§7.18.1).  Required.
  const std::uint8_t* skip = nullptr;
  // LosslessArray[SegmentIds[mi]] for luma.  Required.
  const std::uint8_t* lossless = nullptr;
  // LosslessArray[ChromaSegmentIds[mi]] on the luma MI grid.  Optional; when
  // null no chroma cell is treated as lossless.
  const std::uint8_t* lossless_chroma = nullptr;

  // cdef_idx per 64x64 luma block: value in [-1, num_strengths).  -1 means the
  // block is copied without filtering.  Required.
  const std::int8_t* cdef_idx = nullptr;
  // Grid size for cdef_idx.  Both zero means "derive ceil(mi/16)".
  int n64_cols = 0;
  int n64_rows = 0;

  // Tile starts in luma MI units.  num_* == 0 means one implicit tile.
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

// Decision report for one 8x8 CDEF block.  Filled by cdef_inspect_block()
// without writing pixels.
struct CdefBlockInfo {
  int r = 0;  // block origin in luma MI units
  int c = 0;
  int idx = -1;
  bool skip = false;        // luma copy-only (skip/lossless)
  bool skip_chroma = false; // chroma copy-only (lossless)
  bool filtered = false;    // luma filtering runs
  int y_dir = 0;            // detected direction 0..7
  std::int64_t var = 0;     // scaled variance from §7.18.2
  int var_str = 0;          // derived variance strength 0..12
  int y_pri = 0;            // final luma strengths/dir/damping
  int y_sec = 0;
  int y_dir_used = 0;
  int y_damping = 0;
  int uv_pri = 0;           // final chroma strengths/dir/damping
  int uv_sec = 0;
  int uv_dir = 0;
  int uv_damping = 0;
  CdefRegion region{};      // filter region (tile or frame)
};

int cdef_plane_width(const CdefFrame& frame, int plane);
int cdef_plane_height(const CdefFrame& frame, int plane);

// Validates geometry, options and pointers.  Returns false and sets `status`
// on failure.
bool cdef_supported(const CdefFrame& frame, const CdefOptions& options,
                    CdefStatus* status);

// Runs CDEF over the whole frame.  Every 8x8 luma block (4x4 chroma) is
// visited exactly once; the destination plane is fully written.
CdefStatus cdef_frame(const CdefFrame& frame, const CdefOptions& options);

// Runs the decision path for one block without writing pixels.  `r`/`c` are in
// luma MI units and must address an 8x8 block origin (multiples of 2).
CdefStatus cdef_inspect_block(const CdefFrame& frame,
                              const CdefOptions& options, int r, int c,
                              CdefBlockInfo* info);

}  // namespace av2enc::cdef

#endif  // AV2ENC_ENCODER_FILTERS_CDEF_CDEF_H_
