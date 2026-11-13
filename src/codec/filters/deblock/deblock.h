// AV2 §7.17 deblocking filter — CPU reference API.
//
// Scope: 8-bit 4:2:0 planar.  This is a correctness oracle; it is deliberately
// scalar and single-threaded, per docs/architecture.md.
//
// Geometry requirement: the luma MI grid must be even in both directions
// (frame dimensions padded to a multiple of 8).  That makes the 4:2:0 chroma
// grid integral and guarantees the chroma filter reach stays inside the
// MI-aligned plane.  Standard encoder/decoder frames satisfy this.
//
// The filter is applied in the normative order described in §7.17.1:
//
//   for plane in {Y,U,V}:
//     for pass in {vertical, horizontal}:
//       if apply_deblocking_filter[plane==0 ? pass : plane+1]:
//         for every 4x4-luma edge on the plane's grid:
//           edge_deblocking_filter(plane, pass, row, col)
//
// The caller supplies reconstructed planes plus metadata derived from the
// partition tree / transform block parsing.  Bitstream syntax is not handled
// here; see the README for the integration contract.

#ifndef AV2ENC_ENCODER_FILTERS_DEBLOCK_DEBLOCK_H_
#define AV2ENC_ENCODER_FILTERS_DEBLOCK_DEBLOCK_H_

#include <cstdint>

#include "codec/filters/deblock/deblock_core.h"

namespace av2enc::deblock {

// A mutable reconstructed plane.  `stride` is in samples.  The buffer must be
// at least deblock_plane_height() rows of stride samples.  For 4:2:0 the
// required visible extent is (mi_cols*4)>>sub_x by (mi_rows*4)>>sub_y, i.e. the
// frame rounded up to the MI grid, matching the spec's planeWidth/planeHeight.
struct DeblockPlane {
  std::uint8_t* data = nullptr;
  int stride = 0;
};

// Full frame description for one deblocking pass.
struct DeblockFrame {
  DeblockPlane planes[3];
  // One metadata grid per plane, mi_cols*mi_rows entries, row-major.  Chroma
  // grids are indexed on the full luma MI grid (an entry is replicated over
  // the luma extent of the chroma block).
  const DeblockMi* mi[3] = {nullptr, nullptr, nullptr};
  int mi_cols = 0;
  int mi_rows = 0;
  int subsampling_x = 1;
  int subsampling_y = 1;
  // Tile starts in luma MI units (MiColStarts / MiRowStarts).  Passing
  // num == 0 means "single tile / no tile boundaries".
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

// 10/12-bit planes use the same public description with a uint16_t container.
// `options.bit_depth` selects the §4.8 Clip1 range and the §7.17 qindex offset;
// `num_planes` is 1 for 4:0:0 and 3 for a chroma format.
struct DeblockPlane16 {
  std::uint16_t* data = nullptr;
  int stride = 0;
};

struct DeblockFrame16 {
  DeblockPlane16 planes[3];
  const DeblockMi* mi[3] = {nullptr, nullptr, nullptr};
  int mi_cols = 0;
  int mi_rows = 0;
  int subsampling_x = 1;
  int subsampling_y = 1;
  int num_planes = 3;
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

enum class DeblockStatus {
  kOk = 0,
  kInvalidArgument,
  kUnsupportedBitDepth,
  kUnsupportedSubsampling,
};

const char* deblock_status_string(DeblockStatus status);

// Required plane extents, in samples, derived from the MI grid.
int deblock_plane_width(const DeblockFrame& frame, int plane);
int deblock_plane_height(const DeblockFrame& frame, int plane);

// Validates the frame/options and applies the filter in place.
DeblockStatus deblock_frame(DeblockFrame& frame, const DeblockOptions& options);

// 10/12-bit variant (uint16_t sample container).  Supports 4:2:0, 4:2:2,
// 4:4:4 and 4:0:0; `options.bit_depth` must be 10 or 12.
bool deblock_supported16(const DeblockFrame16& frame,
                         const DeblockOptions& options, DeblockStatus* status);
DeblockCoreFrame16 deblock_to_core16(const DeblockFrame16& frame);
DeblockStatus deblock_frame16(DeblockFrame16& frame,
                              const DeblockOptions& options);
int deblock_plane_width16(const DeblockFrame16& frame, int plane);
int deblock_plane_height16(const DeblockFrame16& frame, int plane);

// Runs the edge decision/sample-filter arithmetic for one edge without
// modifying pixels.  Useful for integration diagnostics and for tests that
// compare CPU and CUDA decisions.  Returns a zeroed record (on_screen == 0)
// for off-screen edges and for invalid plane/pass/row/col.
DeblockEdgeInfo deblock_inspect_edge(const DeblockFrame& frame,
                                     const DeblockOptions& options, int plane,
                                     int pass, int row, int col);

// Converts the public frame description to the shared host/device core form.
DeblockCoreFrame deblock_to_core(const DeblockFrame& frame);

// True when `options` + plane geometry are inside the supported subset.
bool deblock_supported(const DeblockFrame& frame, const DeblockOptions& options,
                       DeblockStatus* status);

}  // namespace av2enc::deblock

#endif  // AV2ENC_ENCODER_FILTERS_DEBLOCK_DEBLOCK_H_
