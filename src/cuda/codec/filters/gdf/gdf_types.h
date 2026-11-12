// Plain-old-data descriptions shared by the AV2 GDF device core, the host
// launcher API and the host-emulation parity test.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//   §5.18.7.9  gdf_params() / GdfBlkSize
//   §5.20.10.3 read_gdf() per-unit use_gdf
//   §7.20.1    table-index derivation (qpIdx, refDstIdx)
//   §7.20.2    get_gdf_sample() luma source
//   §7.20.5    apply_gdf_filter()
//
// This header contains no CUDA and no allocation so the same descriptions are
// usable from .cu kernels, host launchers and the host-emulation test.

#ifndef AV2ENC_CUDA_GDF_GDF_TYPES_H_
#define AV2ENC_CUDA_GDF_GDF_TYPES_H_

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::gdf {

// §9.8 table extents.  Unscoped enum constants (not objects) are safe to use
// from device code: they need no host storage and cannot be "undefined in
// device code" the way a namespace-scope constexpr object can.
enum : int {
  kGdfNumRefDst = 6,
  kGdfNumQpIdx = 6,
  kGdfNumTaps = 22,  // 18 spatial + 4 gradient
  kGdfNumClasses = 4,
  kGdfNumComponents = 3,
  kGdfIntraPositions = 4096,  // 16^3
  kGdfInterPositions = 1000,  // 10^3
  // GdfBlkSize is 64 or 128 per §5.18.7.9; the core additionally accepts any
  // even size in [2, 128] so the filter can be exercised as small blocks.
  kGdfMinUnitSize = 2,
  kGdfMaxUnitSize = 128,
  // A 128x128 unit has a 136-sample vertical span (the -8 stripe offset plus
  // the last-row clamp), which can intersect three 64-line stripe windows.
  kGdfMaxRegionsPerUnit = 3,
};

// §7.20.1 derivation inputs.  `order_hints` and `frame_distance` are *device*
// pointers to the first Min(NumTotalRefs, 2) entries; they may be null for an
// intra frame, where refDstIdx is 0 and the arrays are not read.
struct GdfIndexParams {
  int frame_is_intra = 1;
  int base_q_idx = 0;
  int bit_depth = 8;
  int gdf_pic_qc_idx = 0;  // f(2), 0..3
  const int* order_hints = nullptr;
  const int* frame_distance = nullptr;
  int num_total_refs = 0;
  int restricted_oh = 0;
};

// §7.20.5 Gdf_Coords[18]: {dy, dx} per spatial tap.  Same layout as the CPU
// table, copied into device memory once by gdf_buffers_init.
struct GdfCoord {
  std::int8_t dy;
  std::int8_t dx;
};

// Device pointers to the normative §9.8 tables.  gdf_buffers_init allocates
// and uploads these once; the kernels receive this descriptor by value, so no
// namespace-scope host constexpr object is referenced from device code.
struct GdfTables {
  const std::uint16_t* alpha = nullptr;       // [6][6][22][4]
  const std::int16_t* weight = nullptr;       // [6][6][3][22][4]
  const std::int32_t* bias = nullptr;         // [6][6][3]
  const std::int16_t* intra_error = nullptr;  // [6][4096]
  const std::int16_t* inter_error = nullptr;  // [5][6][1000]
  const GdfCoord* coords = nullptr;           // [18]
};

// §7.20.1 result: indices into Gdf_Alpha / Gdf_Weight / Gdf_Bias and the
// Gdf_Intra_Error / Gdf_Inter_Error banks.
struct GdfTableIndices {
  int qp_idx = 0;       // 0..5
  int ref_dst_idx = 0;  // 0..5, 0 = intra
};

// §7.20.2 get_gdf_sample() luma source.  All pointers and strides are device
// memory.  The rectangle [start_x, end_x] x [start_y, end_y] is the visible
// luma plane; samples below/above the 64-line stripe come from `curr`, the
// two-line clamp included.
struct GdfFrameSource {
  const std::uint16_t* curr = nullptr;
  int curr_stride = 0;
  const std::uint16_t* cdef = nullptr;
  int cdef_stride = 0;
  int start_x = 0;
  int end_x = 0;
  int start_y = 0;
  int end_y = 0;
  int stripe_start_y = 0;
  int stripe_end_y = 0;
};

// One GDF unit.  `lr` is the device LrFrame[0] plane; `x`/`y` are the unit's
// top-left luma sample, `w`/`h` its even extent.  `use_gdf` is the §5.20.10.3
// per-unit decision (GdfBlks), computed on the CPU; a zero value makes the
// kernel a no-op for this unit.  `bit_depth` is 8, 10 or 12; `pix_scale` is
// GdfPixScale (1..4).
//
// A unit that spans a 64-line stripe must be filtered once per stripe region
// (§7.20.2 clamps source reads to the current stripe; the stripes are offset
// by -8 luma samples).  Set stripe_start_y/stripe_end_y to that region's
// bounds; a negative stripe_start_y means "use GdfFrameSource's stripe".
struct GdfUnitJob {
  std::uint16_t* lr = nullptr;
  int lr_stride = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int qp_idx = 0;
  int ref_dst_idx = 0;
  int pix_scale = 1;
  int bit_depth = 8;
  int use_gdf = 1;
  int stripe_start_y = -1;
  int stripe_end_y = -1;
};

// One stripe-bounded part of a GDF unit.  Mirrors
// av2enc::filters::gdf::GdfStripeRegion in the CPU decision module.
struct GdfRegion {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int stripe_start_y = 0;
  int stripe_end_y = 0;
};

// Job slots needed by gdf_apply_grid: one per potential stripe region.
inline constexpr int gdf_grid_slots(int cols, int rows) noexcept {
  return cols > 0 && rows > 0 ? cols * rows * kGdfMaxRegionsPerUnit : 0;
}

// §5.18.7.9 unit grid form.  `cols`/`rows` are the GdfBlks dimensions,
// `unit_size` is GdfBlkSize and `width`/`height` are the (padded) luma plane
// extents used to clip the last row/column of units.  `mi_rows` (= MiRows of
// the padded luma plane) is needed for the §7.20.1 last-unit-row clamp.  The
// device `use_gdf` mask has cols*rows entries, row-major, and is the CPU
// decision applied on the GPU.
struct GdfFrameGrid {
  GdfFrameSource src{};
  std::uint16_t* lr = nullptr;
  int lr_stride = 0;
  int bit_depth = 8;
  int pix_scale = 1;
  int unit_size = 64;
  int cols = 0;
  int rows = 0;
  int width = 0;
  int height = 0;
  int mi_rows = 0;
  GdfIndexParams index{};
};

// Dynamic shared-memory bytes needed by one unit block: four uint16 gradient
// taps plus one class byte per 2x2 cell.
inline constexpr std::size_t gdf_unit_shared_bytes(int max_unit_size) noexcept {
  const std::size_t cells_per_side =
      static_cast<std::size_t>(max_unit_size / 2);
  return cells_per_side * cells_per_side *
         (4 * sizeof(std::uint16_t) + sizeof(std::uint8_t));
}

}  // namespace av2enc::cuda::gdf

#endif  // AV2ENC_CUDA_GDF_GDF_TYPES_H_
