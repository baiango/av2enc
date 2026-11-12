// AV2 v1.0.0 §7.13.3.15 — save subpu size process, shared host/device core.
//
// This header is deliberately pure integer C++ with no CUDA intrinsics, so it
// is compiled
//   * by nvcc as `__host__ __device__` code for the real kernels, and
//   * by the host compiler (no __CUDACC__) for the no-nvcc emulation parity
//     test.
//
// The normative process (§7.13.3.15, p. 484):
//
//   subX = (plane > 0) ? SubsamplingX : 0
//   subY = (plane > 0) ? SubsamplingY : 0
//   subPuSz = find_tx_size(w, h)
//   lumaRow = y >> (2 - subY)
//   lumaCol = x >> (2 - subX)
//   for (r = 0; r < h >> (MI_SIZE_LOG2 - subY); r++)
//     for (c = 0; c < w >> (MI_SIZE_LOG2 - subX); c++) {
//       SubPuColBase[plane > 0][lumaRow + r][lumaCol + c] = lumaCol
//       SubPuRowBase[plane > 0][lumaRow + r][lumaCol + c] = lumaRow
//       SubPuSize[plane > 0][lumaRow + r][lumaCol + c] = subPuSz
//     }
//
// §7.13.3.14 early-terminates when the subblock is the whole prediction unit:
//
//   if (w == PuWidth && h == PuHeight) return;
//
// The early return is *after* the argument validation the CPU reference
// performs, and it writes nothing.
//
// The arrays are indexed on the full luma MI grid for both planes.  `plane`
// is 0 (luma), 1 (U) or 2 (V); the storage slot is `plane > 0 ? 1 : 0`.
#ifndef AV2ENC_CUDA_SUBPU_SUBPU_DEVICE_CUH_
#define AV2ENC_CUDA_SUBPU_SUBPU_DEVICE_CUH_

#include <cstdint>

#if defined(__CUDACC__)
#define AV2ENC_SUBPU_HD __host__ __device__
#else
#define AV2ENC_SUBPU_HD
#endif

namespace av2enc::cuda::subpu {

// §3 MI_SIZE_LOG2 / MI_SIZE.  Enumerators (not variables) are compile-time
// integral constants in both host and device code and need no device symbol.
enum : int {
  kMiSizeLog2 = 2,
  kMiSize = 1 << kMiSizeLog2,
  kTxSizeCount = 25,
  kTxInvalid = -1,
  // The grid has two storage planes: luma and "chroma" (U and V share the
  // chroma slot because the process is invoked once per plane; the reference
  // grid uses `plane > 0`).
  kSubPuPlaneCount = 2,
};

// AV2 TX_SIZE enum in TX_SIZES_ALL order (§6.19.6.1).  The integer values are
// the indices into Tx_Width / Tx_Height and into the per-MI SubPuSize array.
enum class TxSize : std::uint8_t {
  k4x4 = 0,
  k8x8 = 1,
  k16x16 = 2,
  k32x32 = 3,
  k64x64 = 4,
  k4x8 = 5,
  k8x4 = 6,
  k8x16 = 7,
  k16x8 = 8,
  k16x32 = 9,
  k32x16 = 10,
  k32x64 = 11,
  k64x32 = 12,
  k4x16 = 13,
  k16x4 = 14,
  k8x32 = 15,
  k32x8 = 16,
  k16x64 = 17,
  k64x16 = 18,
  k4x32 = 19,
  k32x4 = 20,
  k8x64 = 21,
  k64x8 = 22,
  k4x64 = 23,
  k64x4 = 24,
  kCount = 25,
};

// Tx_Width[TX_SIZES_ALL] / Tx_Height[TX_SIZES_ALL] (§7.16.2, p. 614).  The
// tables are function-local to the __host__ __device__ accessors below: a
// namespace-scope host `constexpr` array is not a device symbol, so kernels
// must not reference one.  The emulation test cross-checks both tables against
// `encoder/m1/missing_bits/save_subpu.h`.
AV2ENC_SUBPU_HD constexpr int tx_width(int tx_size) noexcept {
  constexpr int kWidths[kTxSizeCount] = {
      4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
      4, 32, 8, 64, 4, 64};
  return (tx_size >= 0 && tx_size < kTxSizeCount) ? kWidths[tx_size] : 0;
}

AV2ENC_SUBPU_HD constexpr int tx_height(int tx_size) noexcept {
  constexpr int kHeights[kTxSizeCount] = {
      4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
      32, 4, 64, 8, 64, 4};
  return (tx_size >= 0 && tx_size < kTxSizeCount) ? kHeights[tx_size] : 0;
}

// §5.20.6.3 find_tx_size(): the TX_SIZE whose Tx_Width/Tx_Height match exactly,
// or kTxInvalid.  Linear search, exactly like the CPU reference.  Both tables
// are function-local here too.
AV2ENC_SUBPU_HD constexpr int find_tx_size(int w, int h) noexcept {
  constexpr int kWidths[kTxSizeCount] = {
      4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64,
      4, 32, 8, 64, 4, 64};
  constexpr int kHeights[kTxSizeCount] = {
      4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16,
      32, 4, 64, 8, 64, 4};
  for (int index = 0; index < kTxSizeCount; ++index) {
    if (kWidths[index] == w && kHeights[index] == h) {
      return index;
    }
  }
  return kTxInvalid;
}

// Outcome of one save_subpu invocation.  The CPU reference collapses all
// rejections to `false`; keeping distinct codes lets the tests prove that
// device and host reject the same cases for the same reason.
enum SubPuCode : int {
  kSubPuOk = 0,
  kSubPuInvalidPlane = 1,
  kSubPuInvalidRegion = 2,
  kSubPuInvalidPuSize = 3,
  kSubPuNoTxSize = 4,
  kSubPuRegionTooSmall = 5,
  kSubPuOutOfBounds = 6,
  kSubPuInvalidGrid = 7,
};

// Device view of the three per-plane arrays.  Both slots are `mi_rows x
// mi_cols` in *luma* MI units; row-major with row stride `mi_cols`.
struct SubPuGridView {
  std::uint8_t* size[kSubPuPlaneCount] = {nullptr, nullptr};
  std::int32_t* col_base[kSubPuPlaneCount] = {nullptr, nullptr};
  std::int32_t* row_base[kSubPuPlaneCount] = {nullptr, nullptr};
  int mi_rows = 0;
  int mi_cols = 0;
};

AV2ENC_SUBPU_HD inline bool grid_slot_ready(const SubPuGridView& grid,
                                            int plane_index) noexcept {
  return plane_index >= 0 && plane_index < kSubPuPlaneCount &&
         grid.size[plane_index] != nullptr &&
         grid.col_base[plane_index] != nullptr &&
         grid.row_base[plane_index] != nullptr && grid.mi_rows > 0 &&
         grid.mi_cols > 0;
}

// Clears one MI cell of one storage slot.  The reset kernel strides this over
// the grid; the host emulation calls reset_grid() below.
AV2ENC_SUBPU_HD inline void reset_grid_cell(const SubPuGridView& grid,
                                            int plane_index,
                                            int index) noexcept {
  if (!grid_slot_ready(grid, plane_index) || index < 0) {
    return;
  }
  grid.size[plane_index][index] = 0;
  grid.col_base[plane_index][index] = 0;
  grid.row_base[plane_index][index] = 0;
}

// Host/device clear of the whole grid: SubPuSize/ColBase/RowBase for both
// storage slots become zero, matching SubPuGrid::reset().
AV2ENC_SUBPU_HD inline void reset_grid(const SubPuGridView& grid) noexcept {
  const long long cells =
      static_cast<long long>(grid.mi_rows) * static_cast<long long>(grid.mi_cols);
  if (cells <= 0) {
    return;
  }
  for (int plane_index = 0; plane_index < kSubPuPlaneCount; ++plane_index) {
    if (!grid_slot_ready(grid, plane_index)) {
      continue;
    }
    for (long long index = 0; index < cells; ++index) {
      reset_grid_cell(grid, plane_index, static_cast<int>(index));
    }
  }
}

// The full §7.13.3.15 process.  `plane` 0/1/2, `x,y,w,h` are in that plane's
// samples, `pu_width/pu_height` in the same plane's samples.  Returns a
// SubPuCode; writes nothing unless the return is kSubPuOk.
AV2ENC_SUBPU_HD inline int save_subpu(int plane, int x, int y, int w, int h,
                                      int subsampling_x, int subsampling_y,
                                      int pu_width, int pu_height,
                                      const SubPuGridView& grid) noexcept {
  if (plane < 0 || plane > 2) {
    return kSubPuInvalidPlane;
  }
  if (x < 0 || y < 0 || w <= 0 || h <= 0) {
    return kSubPuInvalidRegion;
  }
  if (pu_width < 0 || pu_height < 0) {
    return kSubPuInvalidPuSize;
  }

  // §7.13.3.15 / §7.13.3.14 early termination: the region is the whole PU, so
  // the caller has nothing to record.  Nothing is written.
  if (w == pu_width && h == pu_height) {
    return kSubPuOk;
  }

  const int sub_x = (plane > 0) ? (subsampling_x != 0 ? 1 : 0) : 0;
  const int sub_y = (plane > 0) ? (subsampling_y != 0 ? 1 : 0) : 0;

  const int sub_pu_size = find_tx_size(w, h);
  if (sub_pu_size == kTxInvalid) {
    return kSubPuNoTxSize;
  }

  const int luma_row = y >> (kMiSizeLog2 - sub_y);
  const int luma_col = x >> (kMiSizeLog2 - sub_x);
  const int rows = h >> (kMiSizeLog2 - sub_y);
  const int cols = w >> (kMiSizeLog2 - sub_x);
  if (rows <= 0 || cols <= 0) {
    return kSubPuRegionTooSmall;
  }

  const int plane_index = plane > 0 ? 1 : 0;
  if (!grid_slot_ready(grid, plane_index)) {
    return kSubPuInvalidGrid;
  }
  if (luma_row < 0 || luma_col < 0 || luma_row + rows > grid.mi_rows ||
      luma_col + cols > grid.mi_cols) {
    return kSubPuOutOfBounds;
  }

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int index = (luma_row + r) * grid.mi_cols + (luma_col + c);
      grid.size[plane_index][index] = static_cast<std::uint8_t>(sub_pu_size);
      grid.col_base[plane_index][index] = luma_col;
      grid.row_base[plane_index][index] = luma_row;
    }
  }
  return kSubPuOk;
}

// Device copy of the Tx_Width/Tx_Height/find_tx_size relation, one row per
// TX_SIZE.  The GPU test compares this table against the CPU reference.
struct SubPuTxEntry {
  int size = 0;
  int width = 0;
  int height = 0;
  int roundtrip = kTxInvalid;
};

AV2ENC_SUBPU_HD inline SubPuTxEntry make_tx_entry(int size) noexcept {
  SubPuTxEntry entry;
  entry.size = size;
  entry.width = tx_width(size);
  entry.height = tx_height(size);
  entry.roundtrip = find_tx_size(entry.width, entry.height);
  return entry;
}

}  // namespace av2enc::cuda::subpu

#endif  // AV2ENC_CUDA_SUBPU_SUBPU_DEVICE_CUH_
