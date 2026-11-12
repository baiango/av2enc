// AV2 v1.0.0 §5.20.6 transform-partition geometry and scan generation for the
// CUDA TX_MODE_SELECT path.
//
// This header is deliberately pure integer C++ with no CUDA intrinsics, so it
// is compiled
//   * by nvcc as __host__ __device__ code for the real kernels, and
//   * by the host compiler (no __CUDACC__) for the no-nvcc emulation parity
//     test.
//
// It contains the device-side authority for
//   * `find_tx_size()` / `set_tx_size()` and the §5.20.6.3 partition geometry,
//   * §5.20.7.30 scans for all 25 TX sizes and all three TX_CLASS values
//     (including rectangular sizes),
//   * the LumaTxSizes / LumaTxMiddle / LumaTxScanOrder grid writer,
//   * plan resolution: a partition decision + origin -> one `TxBlockJob` per
//     TX part, which is what `src/cuda/tx_variants/` executes.
//
// The §9.2 tables are the shipped `encoder/m2/tx_partition/tx_partition_tables.h`
// arrays, so device and CPU cannot drift.  Build the real kernels with
// `--expt-relaxed-constexpr` (as `src/cuda/dip/` does) so device code may read
// those namespace-scope constexpr arrays.
#pragma once

#include <cstdint>

#include "cuda/codec/transform/variants/tx_variants.h"
#include "codec/transform/tx_partition/tx_partition_tables.h"
#include "codec/transform/variants/tx_variants.h"

#if defined(__CUDACC__)
#define TXP_HD __host__ __device__
#else
#define TXP_HD
#endif

namespace av2enc::cuda::txpart {

namespace cpu = av2enc::video::txpartition;

using av2enc::txv::TxParams;

inline constexpr int kMiSizePx = 4;
inline constexpr int kBlockCount = 29;
inline constexpr int kTxSizeCount = 25;
inline constexpr int kTxTypeCount = 16;
inline constexpr int kMaxParts = 4;
inline constexpr int kMaxAdjDim = 32;
inline constexpr int kMaxScanArea = kMaxAdjDim * kMaxAdjDim;
inline constexpr int kScanClasses = 3;
inline constexpr int kScanTableCount = kTxSizeCount * kScanClasses;
inline constexpr int kScanInvalid = 0xFFFF;

// Table 6.23 txPartition values (int form for device use; the CPU enum is
// checked against these constants in tx_partition_plan.cpp).
inline constexpr int kPartNone = 0;
inline constexpr int kPartSplit = 1;
inline constexpr int kPartHorz = 2;
inline constexpr int kPartVert = 3;
inline constexpr int kPartHorz4 = 4;
inline constexpr int kPartVert4 = 5;
inline constexpr int kPartHorz5 = 6;
inline constexpr int kPartVert5 = 7;

// §5.20.7.30 TX_CLASS_* values.
inline constexpr int kScan2D = 0;
inline constexpr int kScanHoriz = 1;
inline constexpr int kScanVert = 2;

// ---------------------------------------------------------------------------
// §9.2 geometry tables
// ---------------------------------------------------------------------------
//
// CUDA cannot odr-use a namespace-scope host `constexpr` array from device
// code, so the shared header carries its own copies.  In the device pass of
// nvcc (`__CUDA_ARCH__` defined) they are `__device__ constexpr`; in every host
// pass and in the emulation build they are plain `constexpr`.  A host-only
// `static_assert` below pins every entry to the shipped CPU §9.2 tables, so the
// copies cannot drift.
#if defined(__CUDA_ARCH__)
#define TXP_CONST_TABLE static __device__ constexpr
#else
#define TXP_CONST_TABLE static constexpr
#endif

TXP_CONST_TABLE int kBlockWidthTable[kBlockCount] = {
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32,
    32, 32, 64, 64, 1, 4, 2, 8, 4, 16, 1, 8, 2, 16,
};
TXP_CONST_TABLE int kBlockHeightTable[kBlockCount] = {
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16,
    32, 64, 32, 64, 4, 1, 8, 2, 16, 4, 8, 1, 16, 2,
};
TXP_CONST_TABLE int kMaxTxSizeRectTable[kBlockCount] = {
    0, 5, 6, 1, 7, 8, 2, 9, 10, 3, 11, 12, 4, 4, 4,
    4, 4, 4, 4, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
};
TXP_CONST_TABLE int kGroupVertOrHorzTable[kBlockCount] = {
    0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 1, 2, 255, 255, 255, 255, 1, 2, 255, 255,
};
TXP_CONST_TABLE int kTxWidthTable[kTxSizeCount] = {
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64,
    4, 16, 8, 32, 16, 64, 4, 32, 8, 64, 4, 64,
};
TXP_CONST_TABLE int kTxHeightTable[kTxSizeCount] = {
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32,
    16, 4, 32, 8, 64, 16, 32, 4, 64, 8, 64, 4,
};
TXP_CONST_TABLE int kTxWidthLog2Table[kTxSizeCount] = {
    2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 5, 5, 6,
    2, 4, 3, 5, 4, 6, 2, 5, 3, 6, 2, 6,
};
TXP_CONST_TABLE int kTxHeightLog2Table[kTxSizeCount] = {
    2, 3, 4, 5, 6, 3, 2, 4, 3, 5, 4, 6, 5,
    4, 2, 5, 3, 6, 4, 5, 2, 6, 3, 6, 2,
};
TXP_CONST_TABLE int kAdjustedTxSizeTable[kTxSizeCount] = {
    0, 1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 3, 3,
    13, 14, 15, 16, 9, 10, 19, 20, 15, 16, 19, 20,
};

#if !defined(__CUDA_ARCH__)
namespace detail {

constexpr bool geometry_tables_match_cpu() {
  for (int i = 0; i < kBlockCount; ++i) {
    if (kBlockWidthTable[i] != cpu::kTxPartitionBlockWidth[i]) return false;
    if (kBlockHeightTable[i] != cpu::kTxPartitionBlockHeight[i]) return false;
    if (kMaxTxSizeRectTable[i] != cpu::kTxPartitionMaxTxSizeRect[i]) {
      return false;
    }
    if (kGroupVertOrHorzTable[i] !=
        cpu::kTxPartitionTypeGroupVertOrHorz[i]) {
      return false;
    }
  }
  for (int i = 0; i < kTxSizeCount; ++i) {
    if (kTxWidthTable[i] != cpu::kTxPartitionTxWidth[i]) return false;
    if (kTxHeightTable[i] != cpu::kTxPartitionTxHeight[i]) return false;
    if (kTxWidthLog2Table[i] != cpu::kTxPartitionTxWidthLog2[i]) return false;
    if (kTxHeightLog2Table[i] != cpu::kTxPartitionTxHeightLog2[i]) {
      return false;
    }
    if (kAdjustedTxSizeTable[i] != cpu::kTxPartitionAdjustedTxSize[i]) {
      return false;
    }
  }
  return true;
}

static_assert(geometry_tables_match_cpu(),
              "tx_partition device geometry tables must match the CPU §9.2 "
              "tables in encoder/m2/tx_partition/tx_partition_tables.h");

}  // namespace detail
#endif

#undef TXP_CONST_TABLE

TXP_HD constexpr int block_width_mi(int mi_size) {
  return (mi_size >= 0 && mi_size < kBlockCount)
             ? kBlockWidthTable[mi_size]
             : 0;
}
TXP_HD constexpr int block_height_mi(int mi_size) {
  return (mi_size >= 0 && mi_size < kBlockCount)
             ? kBlockHeightTable[mi_size]
             : 0;
}
TXP_HD constexpr int block_width_px(int mi_size) {
  return block_width_mi(mi_size) * kMiSizePx;
}
TXP_HD constexpr int block_height_px(int mi_size) {
  return block_height_mi(mi_size) * kMiSizePx;
}
TXP_HD constexpr int max_tx_size_rect(int mi_size) {
  return (mi_size >= 0 && mi_size < kBlockCount)
             ? kMaxTxSizeRectTable[mi_size]
             : -1;
}
TXP_HD constexpr int tx_group_vert_or_horz(int mi_size) {
  return (mi_size >= 0 && mi_size < kBlockCount)
             ? kGroupVertOrHorzTable[mi_size]
             : 255;
}
TXP_HD constexpr int tx_width(int tx_size) {
  return (tx_size >= 0 && tx_size < kTxSizeCount)
             ? kTxWidthTable[tx_size]
             : 0;
}
TXP_HD constexpr int tx_height(int tx_size) {
  return (tx_size >= 0 && tx_size < kTxSizeCount)
             ? kTxHeightTable[tx_size]
             : 0;
}
TXP_HD constexpr int tx_width_log2(int tx_size) {
  return (tx_size >= 0 && tx_size < kTxSizeCount)
             ? kTxWidthLog2Table[tx_size]
             : 0;
}
TXP_HD constexpr int tx_height_log2(int tx_size) {
  return (tx_size >= 0 && tx_size < kTxSizeCount)
             ? kTxHeightLog2Table[tx_size]
             : 0;
}
TXP_HD constexpr int adjusted_tx_size(int tx_size) {
  return (tx_size >= 0 && tx_size < kTxSizeCount)
             ? kAdjustedTxSizeTable[tx_size]
             : -1;
}

// §5.20.6.3 find_tx_size(w, h).  -1 == TX_INVALID.
TXP_HD constexpr int find_tx_size(int w, int h) {
  for (int tx = 0; tx < kTxSizeCount; ++tx) {
    if (tx_width(tx) == w && tx_height(tx) == h) return tx;
  }
  return -1;
}

// §5.20.6.2/§5.20.6.3 64x64 chunking helpers.
TXP_HD constexpr int tx_chunk_cols(int mi_size) {
  const int chunks = block_width_px(mi_size) >> 6;
  return chunks > 1 ? chunks : 1;
}
TXP_HD constexpr int tx_chunk_rows(int mi_size) {
  const int chunks = block_height_px(mi_size) >> 6;
  return chunks > 1 ? chunks : 1;
}

// ---------------------------------------------------------------------------
// Legal partition enumeration (§5.20.6.3)
// ---------------------------------------------------------------------------

struct LegalSplits {
  int applicable = 0;
  int chunked = 0;
  int allow_horz = 0;
  int allow_vert = 0;
  int has_partition_type = 0;
  int has_2or3 = 0;
  int reduced_tx_part_set = 0;
  int option_count = 0;
  int options[8] = {};
};

TXP_HD inline int legal_option_contains(const LegalSplits& legal, int part) {
  for (int i = 0; i < legal.option_count; ++i) {
    if (legal.options[i] == part) return 1;
  }
  return 0;
}

TXP_HD inline LegalSplits legal_splits(int mi_size, int reduced_tx_part_set) {
  LegalSplits out;
  out.reduced_tx_part_set = reduced_tx_part_set ? 1 : 0;
  if (mi_size < 0 || mi_size >= kBlockCount || mi_size == 0) return out;
  const int bw = block_width_px(mi_size);
  const int bh = block_height_px(mi_size);
  out.chunked = (bw > 64 || bh > 64) ? 1 : 0;
  if (out.chunked) return out;

  const int tx = max_tx_size_rect(mi_size);
  if (tx < 0) return out;
  const int tw = tx_width(tx);
  const int th = tx_height(tx);
  out.applicable = 1;
  out.allow_horz = find_tx_size(tw, th / 2) >= 0 ? 1 : 0;
  out.allow_vert = find_tx_size(tw / 2, th) >= 0 ? 1 : 0;
  out.has_partition_type = out.allow_horz && out.allow_vert ? 1 : 0;

  const auto add = [&out](int part) {
    if (out.option_count < 8) out.options[out.option_count++] = part;
  };
  add(kPartNone);
  if (out.has_partition_type != 0) {
    add(kPartSplit);
    add(kPartHorz);
    add(kPartVert);
  } else if (tx_group_vert_or_horz(mi_size) > 0) {
    out.has_2or3 = 1;
    if (out.allow_horz != 0) {
      add(kPartHorz);
      if (reduced_tx_part_set == 0) add(kPartHorz4);
    } else {
      add(kPartVert);
      if (reduced_tx_part_set == 0) add(kPartVert4);
    }
  } else if (out.allow_horz != 0 || out.allow_vert != 0) {
    add(out.allow_horz != 0 ? kPartHorz : kPartVert);
  }
  return out;
}

TXP_HD inline int partition_signallable(int mi_size, int part) {
  const LegalSplits legal = legal_splits(mi_size, 0);
  return legal.applicable != 0 && legal_option_contains(legal, part) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// §5.20.6.3 partition geometry
// ---------------------------------------------------------------------------

struct PlanPart {
  int mi_row = 0;
  int mi_col = 0;
  int tx_size = -1;
};

struct PlanGeometry {
  int count = 0;
  PlanPart parts[kMaxParts] = {};
};

// Resolves one partition decision into concrete TX parts in decode order.
// Returns false when the partition is not signallable for this block size.
TXP_HD inline bool plan_geometry(int mi_size, int partition, int mi_row,
                                 int mi_col, PlanGeometry* out) {
  if (out == nullptr) return false;
  *out = PlanGeometry{};
  if (mi_size < 0 || mi_size >= kBlockCount) return false;
  const int tx = max_tx_size_rect(mi_size);
  if (tx < 0) return false;
  if (partition_signallable(mi_size, partition) == 0) return false;

  const int w4 = tx_width(tx) / kMiSizePx;
  const int h4 = tx_height(tx) / kMiSizePx;
  const auto push = [out](int row, int col, int size) {
    if (out->count < kMaxParts) {
      out->parts[out->count].mi_row = row;
      out->parts[out->count].mi_col = col;
      out->parts[out->count].tx_size = size;
      ++out->count;
    }
  };

  switch (partition) {
    case kPartNone:
      push(mi_row, mi_col, tx);
      break;
    case kPartSplit: {
      const int sub = find_tx_size(tx_width(tx) / 2, tx_height(tx) / 2);
      if (sub < 0) return false;
      const int hw = w4 / 2;
      const int hh = h4 / 2;
      push(mi_row, mi_col, sub);
      push(mi_row, mi_col + hw, sub);
      push(mi_row + hh, mi_col, sub);
      push(mi_row + hh, mi_col + hw, sub);
      break;
    }
    case kPartHorz: {
      const int sub = find_tx_size(tx_width(tx), tx_height(tx) / 2);
      if (sub < 0) return false;
      const int hh = h4 / 2;
      push(mi_row, mi_col, sub);
      push(mi_row + hh, mi_col, sub);
      break;
    }
    case kPartVert: {
      const int sub = find_tx_size(tx_width(tx) / 2, tx_height(tx));
      if (sub < 0) return false;
      const int hw = w4 / 2;
      push(mi_row, mi_col, sub);
      push(mi_row, mi_col + hw, sub);
      break;
    }
    case kPartHorz4: {
      const int sub = find_tx_size(tx_width(tx), tx_height(tx) / 4);
      if (sub < 0) return false;
      const int q = h4 / 4;
      for (int i = 0; i < 4; ++i) push(mi_row + i * q, mi_col, sub);
      break;
    }
    case kPartVert4: {
      const int sub = find_tx_size(tx_width(tx) / 4, tx_height(tx));
      if (sub < 0) return false;
      const int q = w4 / 4;
      for (int i = 0; i < 4; ++i) push(mi_row, mi_col + i * q, sub);
      break;
    }
    default:
      // HORZ5 / VERT5 have no syntax path.
      return false;
  }
  return out->count > 0;
}

// ---------------------------------------------------------------------------
// §5.20.7.30 scans: 25 sizes x 3 classes, rectangular included
// ---------------------------------------------------------------------------

TXP_HD constexpr int scan_width(int tx_size) {
  const int w = tx_width(tx_size);
  return w < kMaxAdjDim ? w : kMaxAdjDim;
}
TXP_HD constexpr int scan_height(int tx_size) {
  const int h = tx_height(tx_size);
  return h < kMaxAdjDim ? h : kMaxAdjDim;
}
TXP_HD constexpr int scan_area(int tx_size) {
  return scan_width(tx_size) * scan_height(tx_size);
}

// Fills one scan (w*h entries) in §5.20.7.30 order.  TX_CLASS_VERT is the
// row-major walk and TX_CLASS_HORIZ the column-major walk; that is the literal
// spec loop nesting, not a naming intuition.
TXP_HD inline void scan_fill(int tx_size, int cls, std::uint16_t* out) {
  if (out == nullptr) return;
  const int w = scan_width(tx_size);
  const int h = scan_height(tx_size);
  const int area = w * h;
  if (area <= 0) return;
  int c = 0;
  if (cls == kScanVert) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        out[c++] = static_cast<std::uint16_t>(y * w + x);
      }
    }
    return;
  }
  if (cls == kScanHoriz) {
    for (int x = 0; x < w; ++x) {
      for (int y = 0; y < h; ++y) {
        out[c++] = static_cast<std::uint16_t>(y * w + x);
      }
    }
    return;
  }
  int x = 0;
  int y = 0;
  for (int i = 0; i < area; ++i) {
    out[c++] = static_cast<std::uint16_t>(y * w + x);
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int max_s = h - 1 - y;
      const int s = x < max_s ? x : max_s;
      x -= s;
      y += s;
    }
  }
}

// Device view of the 25x3 scan table.
//   data[(tx_size * 3 + cls) * kMaxScanArea + pos]
//   length[tx_size * 3 + cls] == scan_area(tx_size)
struct ScanTableView {
  std::uint16_t* data = nullptr;
  std::uint16_t* length = nullptr;
};

TXP_HD inline int scan_table_index(int tx_size, int cls) {
  return tx_size * kScanClasses + cls;
}

TXP_HD inline bool scan_table_fill_one(const ScanTableView& view, int tx_size,
                                       int cls) {
  if (view.data == nullptr || view.length == nullptr) return false;
  if (tx_size < 0 || tx_size >= kTxSizeCount) return false;
  if (cls < 0 || cls >= kScanClasses) return false;
  const int area = scan_area(tx_size);
  if (area <= 0) return false;
  const int index = scan_table_index(tx_size, cls);
  scan_fill(tx_size, cls, view.data + index * kMaxScanArea);
  view.length[index] = static_cast<std::uint16_t>(area);
  return true;
}

TXP_HD inline std::uint16_t scan_table_at(const ScanTableView& view,
                                          int tx_size, int cls, int pos) {
  if (view.data == nullptr) return kScanInvalid;
  if (tx_size < 0 || tx_size >= kTxSizeCount) return kScanInvalid;
  if (cls < 0 || cls >= kScanClasses) return kScanInvalid;
  if (pos < 0 || pos >= scan_area(tx_size)) return kScanInvalid;
  return view.data[scan_table_index(tx_size, cls) * kMaxScanArea + pos];
}

// ---------------------------------------------------------------------------
// LumaTxSizes / LumaTxMiddle / LumaTxScanOrder writer
// ---------------------------------------------------------------------------

struct TxGridView {
  std::uint8_t* tx_size = nullptr;
  std::uint8_t* middle = nullptr;
  std::uint8_t* scan_order = nullptr;
  int mi_rows = 0;
  int mi_cols = 0;
};

TXP_HD inline bool grid_contains(const TxGridView& grid, int row, int col) {
  return row >= 0 && col >= 0 && row < grid.mi_rows && col < grid.mi_cols;
}

TXP_HD inline void grid_reset(TxGridView& grid, int rows, int cols) {
  grid.mi_rows = rows > 0 ? rows : 0;
  grid.mi_cols = cols > 0 ? cols : 0;
  const int n = grid.mi_rows * grid.mi_cols;
  for (int i = 0; i < n; ++i) {
    if (grid.tx_size != nullptr) grid.tx_size[i] = 0;
    if (grid.middle != nullptr) grid.middle[i] = 0;
    if (grid.scan_order != nullptr) grid.scan_order[i] = 0;
  }
}

TXP_HD inline void grid_set(const TxGridView& grid, int row, int col, int h4,
                            int w4, int tx_size_value, int mid, int order) {
  if (grid.tx_size == nullptr) return;
  for (int i = 0; i < h4; ++i) {
    for (int j = 0; j < w4; ++j) {
      const int r = row + i;
      const int c = col + j;
      if (!grid_contains(grid, r, c)) continue;
      const int idx = r * grid.mi_cols + c;
      grid.tx_size[idx] = static_cast<std::uint8_t>(tx_size_value);
      if (grid.middle != nullptr) {
        grid.middle[idx] = static_cast<std::uint8_t>(mid);
      }
      if (grid.scan_order != nullptr) {
        grid.scan_order[idx] = static_cast<std::uint8_t>(order);
      }
    }
  }
}

// §5.20.6.3 set_tx_size().  h4/w4 are in 4x4 units.
TXP_HD inline bool set_tx_size(const TxGridView& grid, int row, int col,
                               int h4, int w4, int mid, int order) {
  const int sub = find_tx_size(w4 * kMiSizePx, h4 * kMiSizePx);
  if (sub < 0) return false;
  grid_set(grid, row, col, h4, w4, sub, mid, order);
  return true;
}

// ---------------------------------------------------------------------------
// Plan descriptors and per-part job resolution
// ---------------------------------------------------------------------------

// One transform-partition plan: a single-level §5.20.6.3 decision over a
// <= 64x64 region.  Blocks larger than 64x64 are chunked by the caller (one
// kPartNone plan per 64x64 chunk, using mi_size == BLOCK_64X64).
struct PlanDesc {
  int mi_size = -1;
  int partition = kPartNone;
  int mi_row = 0;
  int mi_col = 0;
  int tx_type = 0;
  int part_base = 0;  // index of the first part in the batch
  int in_base = 0;    // adjusted-residual element offset of the first part
  int out_base = 0;   // coded-residual element offset of the first part
};

// One resolved TX part.  `parts[]` is indexed by `plan.part_base + i`.
struct ResolvedPart {
  int mi_row = 0;
  int mi_col = 0;
  int tx_size = -1;
};

// `params` is [tx_type][tx_size] (tx_type * 25 + tx_size), the same layout the
// host wrapper uploads.  Writes one TxBlockJob per part, with explicit element
// offsets, ready for the `_multi` entry points of src/cuda/tx_variants/.
TXP_HD inline bool plan_resolve(const TxParams* params, const PlanDesc& plan,
                                TxBlockJob* jobs, int job_capacity,
                                ResolvedPart* parts, int part_capacity) {
  if (params == nullptr || jobs == nullptr) return false;
  if (plan.tx_type < 0 || plan.tx_type >= kTxTypeCount) return false;
  PlanGeometry geom;
  if (!plan_geometry(plan.mi_size, plan.partition, plan.mi_row, plan.mi_col,
                     &geom)) {
    return false;
  }
  if (plan.part_base < 0) return false;
  if (plan.part_base + geom.count > job_capacity) return false;
  if (parts != nullptr && plan.part_base + geom.count > part_capacity) {
    return false;
  }
  int adj_offset = 0;
  int coded_offset = 0;
  for (int i = 0; i < geom.count; ++i) {
    const PlanPart& part = geom.parts[i];
    if (part.tx_size < 0 || part.tx_size >= kTxSizeCount) return false;
    const TxParams& p = params[plan.tx_type * kTxSizeCount + part.tx_size];
    if (p.w <= 0 || p.h <= 0 || p.coded_w <= 0 || p.coded_h <= 0) {
      return false;
    }
    TxBlockJob& job = jobs[plan.part_base + i];
    job.params = p;
    job.in_offset = static_cast<std::int32_t>(plan.in_base + adj_offset);
    job.out_offset = static_cast<std::int32_t>(plan.out_base + coded_offset);
    if (parts != nullptr) {
      ResolvedPart& out = parts[plan.part_base + i];
      out.mi_row = part.mi_row;
      out.mi_col = part.mi_col;
      out.tx_size = part.tx_size;
    }
    adj_offset += p.w * p.h;
    coded_offset += p.coded_w * p.coded_h;
  }
  return true;
}

}  // namespace av2enc::cuda::txpart
