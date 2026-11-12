// AV2 §7.17 deblocking filter — shared host/device core.
//
// This header contains the normative arithmetic of the AV2 deblocking filter
// in a form that can be compiled both as ordinary host C++ and as CUDA device
// code.  The CPU reference (deblock.cpp) and the CUDA kernel (src/cuda/deblock)
// call the *same* inline functions, so CPU/GPU results are bit-exact by
// construction when the metadata and plane contents match.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7), §7.17.
//
// Scope of this implementation: 8-bit 4:2:0 planar.  The algorithm itself is
// written for the general case where the caller supplies the per-4x4-block
// metadata (mode-info base, transform sizes, skip, qindex, lossless flags and
// sub-PU bases) that the decoder derives while parsing the partition tree.
//
// Filter lengths: AV2 §7.17.3 replaces the AV1 "4/8/14-tap" naming with
// maxWidthPos/maxWidthNeg:
//
//   filterSize <= 4   -> maxWidthPos = 1        (4-tap class)
//   filterSize == 8   -> maxWidthPos = 3        (8-tap class)
//   filterSize == 16  -> 4 (chroma) / 6 (luma)  (14-tap class)
//   filterSize > 16   -> 4 (chroma) / 8 (luma)  (wide class)
//
// An `sbEdge` (a vertical tile edge, or a horizontal edge on the 64x64 grid)
// clamps maxWidthNeg to min(maxWidthPos, 2 chroma / 6 luma).

#ifndef AV2ENC_ENCODER_FILTERS_DEBLOCK_DEBLOCK_CORE_H_
#define AV2ENC_ENCODER_FILTERS_DEBLOCK_DEBLOCK_CORE_H_

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define AV2ENC_DEBLOCK_HD __host__ __device__ __forceinline__
// During the device compilation pass the normative tables must live in device
// memory; the host pass sees ordinary inline constexpr arrays.
#if defined(__CUDA_ARCH__)
#define AV2ENC_DEBLOCK_TABLE __device__ constexpr
#else
#define AV2ENC_DEBLOCK_TABLE inline constexpr
#endif
#else
#define AV2ENC_DEBLOCK_HD inline
#define AV2ENC_DEBLOCK_TABLE inline constexpr
#endif

namespace av2enc::deblock {

// Smallest coding unit used by deblocking metadata, in luma samples.
constexpr int kMiSize = 4;

// AV2 TxSize enumeration (TX_SIZES_ALL order, §7.15.4 / spec tables).
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

inline constexpr int kTxSizesAll = 25;

// Tx_Width[TX_SIZES_ALL] / Tx_Height[TX_SIZES_ALL] from the AV2 spec tables.
AV2ENC_DEBLOCK_TABLE int kTxWidth[kTxSizesAll] = {
    4, 8, 16, 32, 64, 4, 8, 8, 16, 16, 32, 32, 64, 4, 16, 8, 32, 16, 64, 4,
    32, 8, 64, 4, 64};
AV2ENC_DEBLOCK_TABLE int kTxHeight[kTxSizesAll] = {
    4, 8, 16, 32, 64, 8, 4, 16, 8, 32, 16, 64, 32, 16, 4, 32, 8, 64, 16, 32,
    4, 64, 8, 64, 4};

AV2ENC_DEBLOCK_HD constexpr int tx_width(TxSize tx) {
  return kTxWidth[static_cast<int>(tx)];
}
AV2ENC_DEBLOCK_HD constexpr int tx_height(TxSize tx) {
  return kTxHeight[static_cast<int>(tx)];
}
AV2ENC_DEBLOCK_HD constexpr bool tx_size_valid(TxSize tx) {
  return static_cast<int>(tx) < kTxSizesAll;
}

// ---------------------------------------------------------------------------
// Per-4x4-luma-block metadata (one entry per luma MI cell for every plane).
//
// The decoder keeps several aligned arrays; this component folds them into one
// POD.  Chroma arrays are indexed on the full luma MI grid: a chroma block's
// entry is replicated over the luma extent it covers, matching the spec's
// "chroma information is held in the bottom-right mode info" rule (§7.17.2).
// ---------------------------------------------------------------------------
struct DeblockMi {
  // MiColBase / MiRowBase: origin of the coding block, in luma MI units.
  std::uint16_t block_col_base = 0;
  std::uint16_t block_row_base = 0;
  // Num_4x4_Blocks_Wide/High[MiSize].
  std::uint8_t block_w4 = 1;
  std::uint8_t block_h4 = 1;
  // SubPuSize: Max_Tx_Size_Rect[MiSize] in the decoder.
  TxSize sub_pu_size = TxSize::k4x4;
  // DeblockingTxSizes[plane][row][col].
  TxSize tx_size = TxSize::k4x4;
  // TxColBase / TxRowBase, in luma MI units.
  std::uint16_t tx_col_base = 0;
  std::uint16_t tx_row_base = 0;
  // SubPuColBase / SubPuRowBase, in luma MI units.
  std::uint16_t sub_pu_col_base = 0;
  std::uint16_t sub_pu_row_base = 0;
  // Skips[row][col] (0 for intra region / SDP, per §7.17.2).
  std::uint8_t skip = 0;
  // RegionTypes[row][col] == INTRA_REGION.
  std::uint8_t region_intra = 1;
  // LosslessArray[segment].
  std::uint8_t lossless = 0;
  // Effective deblocking quantizer index for the cell: get_qindex(0, segment),
  // i.e. LumaQIndex / ChromaQIndex.  For M1 (no delta-Q, no segmentation) this
  // is simply base_q_idx.
  std::uint8_t qindex = 0;
};

// Frame-level deblocking state.  Defaults are the M1 intra configuration.
struct DeblockOptions {
  // apply_deblocking_filter[0..3]: Y vertical, Y horizontal, U, V.
  bool apply_deblocking_filter[4] = {true, true, true, true};
  // §5.4.10 / §7.17.2: skip filtering across tile boundaries.
  bool disable_loopfilters_across_tiles = false;
  // allow_df_sub_pu: enable sub-prediction-unit edges (inter tool; harmless
  // for intra where the sub-PU bases equal the block base).
  bool allow_df_sub_pu = false;
  // Used for `if (RegionTypes == INTRA_REGION || (FrameIsIntra && enable_sdp))`.
  bool frame_is_intra = true;
  bool enable_sdp = false;
  // Chroma AC quantizer deltas used by §7.17.6 (M1: all zero).
  int delta_q_u_ac = 0;
  int delta_q_v_ac = 0;
  int base_uv_ac_delta_q = 0;
  // DfDeltaQ[0..3]: luma pass 0 / luma pass 1 / U / V.
  int df_delta_q[4] = {0, 0, 0, 0};
  // BitDepth; this component supports 8 only.
  int bit_depth = 8;
};

// Device/host-visible frame description.  All pointers refer to caller-owned
// memory; the core never allocates or synchronises.
struct DeblockCoreFrame {
  // Plane sample pointers and strides (samples per row).  For 4:2:0 the chroma
  // planes are (mi_cols*4)>>sub_x wide and (mi_rows*4)>>sub_y high.
  std::uint8_t* plane[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};
  // Per-plane metadata grids, each mi_cols*mi_rows entries, row-major.
  const DeblockMi* mi[3] = {nullptr, nullptr, nullptr};
  int mi_cols = 0;
  int mi_rows = 0;
  int sub_x = 1;
  int sub_y = 1;
  // Tile starts in luma MI units (MiColStarts / MiRowStarts).  `num_*` may be
  // zero, meaning "no tile boundaries" (equivalently a single implicit tile).
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

// Diagnostic result for one edge call.  Populated even when the edge is not
// filtered, so tests and integration code can compare decisions.
struct DeblockEdgeInfo {
  int on_screen = 0;
  int tile_vert_edge = 0;
  int tile_horz_edge = 0;
  int sb_edge = 0;
  int is_block_edge = 0;
  int is_tx_edge = 0;
  int is_sub_pu_edge = 0;
  int skip = 0;
  int curr_lossless = 0;
  int prev_lossless = 0;
  int curr_q = 0;
  int prev_q = 0;
  int curr_side = 0;
  int prev_side = 0;
  int q_thr = 0;
  int side = 0;
  int filter_size = 0;
  int max_width_pos = 0;
  int max_width_neg = 0;
  int width = 0;
  int apply_filter = 0;
};

// ---------------------------------------------------------------------------
// Normative tables (§7.14.2, §7.17.6, §7.17.7).
// ---------------------------------------------------------------------------
AV2ENC_DEBLOCK_TABLE int kAcQlookup[25] = {
    64, 40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 54, 55,
    57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78};

AV2ENC_DEBLOCK_TABLE int kSideThresholds[296] = {
    -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -14, -13, -11, -10, -9, -7, -6, -4,
    -3, -2, 0, 0, 2, 3, 5, 6, 7, 9, 10, 12, 13,
    15, 16, 17, 19, 20, 22, 23, 24, 26, 27, 29, 30, 32,
    33, 34, 36, 37, 39, 40, 42, 43, 44, 46, 47, 49, 50,
    51, 53, 54, 56, 57, 59, 60, 61, 63, 64, 66, 67, 69,
    70, 71, 73, 74, 76, 77, 78, 80, 81, 83, 84, 86, 87,
    88, 90, 91, 93, 94, 96, 101, 111, 120, 130, 140, 150, 160,
    170, 180, 190, 200, 210, 220, 230, 240, 249, 259, 269, 279, 289,
    299, 309, 319, 329, 339, 349, 359, 368, 378, 388, 398, 408, 418,
    428, 438, 448, 458, 468, 478, 488, 497, 507, 517, 527, 537, 547,
    557, 567, 577, 587, 597, 607, 616, 626, 636, 646, 656, 666, 676,
    686, 696, 706, 716, 726, 736, 745, 755, 765, 775, 785, 795, 805,
    815, 825, 835, 845, 855, 864, 874, 884, 894, 904, 914, 924, 934,
    944, 954, 964, 974, 984, 993, 1003, 1013, 1023, 1033, 1043, 1053, 1063,
    1073, 1083, 1093, 1103, 1112, 1122, 1132, 1142, 1152, 1162, 1172, 1182, 1192,
    1202, 1212, 1222, 1232, 1241, 1251, 1261, 1271, 1281, 1291, 1301, 1311, 1321,
    1331, 1341, 1351, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450,
    1460, 1470, 1480, 1489, 1499, 1509, 1519, 1529, 1539, 1549, 1559, 1569, 1579,
    1589, 1599, 1608, 1618, 1628, 1638, 1648, 1658, 1668, 1678};

AV2ENC_DEBLOCK_TABLE int kWMult[8] = {85, 51, 37, 28, 23, 20, 17, 15};
AV2ENC_DEBLOCK_TABLE int kQThreshMults[8] = {32, 25, 19, 19, 18, 18, 17, 17};
AV2ENC_DEBLOCK_TABLE int kQFirst[9] = {45, 43, 40, 35, 32, 32, 32, 32, 32};

static_assert(sizeof(kTxWidth) / sizeof(kTxWidth[0]) == kTxSizesAll,
              "Tx_Width table size");
static_assert(sizeof(kTxHeight) / sizeof(kTxHeight[0]) == kTxSizesAll,
              "Tx_Height table size");
static_assert(sizeof(kAcQlookup) / sizeof(kAcQlookup[0]) == 25,
              "Ac_Qlookup table size");
static_assert(sizeof(kSideThresholds) / sizeof(kSideThresholds[0]) == 296,
              "Side_Thresholds table size");
static_assert(sizeof(kWMult) / sizeof(kWMult[0]) == 8, "W_Mult table size");
static_assert(sizeof(kQThreshMults) / sizeof(kQThreshMults[0]) == 8,
              "Q_Thresh_Mults table size");
static_assert(sizeof(kQFirst) / sizeof(kQFirst[0]) == 9, "Q_First table size");

inline constexpr int kDfShift = 8;       // DF_SHIFT
inline constexpr int kDfDeltaScale = 8;  // DF_DELTA_SCALE
inline constexpr int kMaxQ = 255;        // MAXQ_8_BITS
inline constexpr int kMaxSideTable = 296;

// ---------------------------------------------------------------------------
// Integer helpers.  AV2 Round2 is (x + 2^(n-1)) >> n with an arithmetic shift
// (C++20 defines right shift of a negative value as arithmetic).
// ---------------------------------------------------------------------------
AV2ENC_DEBLOCK_HD constexpr int clip3(int lo, int hi, int v) {
  return v < lo ? lo : (v > hi ? hi : v);
}
AV2ENC_DEBLOCK_HD constexpr int clip1_8(int v) { return clip3(0, 255, v); }
AV2ENC_DEBLOCK_HD constexpr int round2(int x, int n) {
  return n == 0 ? x : ((x + (1 << (n - 1))) >> n);
}
AV2ENC_DEBLOCK_HD constexpr int iabs(int x) { return x < 0 ? -x : x; }
AV2ENC_DEBLOCK_HD constexpr int imin(int a, int b) { return a < b ? a : b; }
AV2ENC_DEBLOCK_HD constexpr int imax(int a, int b) { return a > b ? a : b; }

// qlookup() / get_q() from §7.14.2.
AV2ENC_DEBLOCK_HD int qlookup(int q) {
  if (q < 25) {
    return kAcQlookup[q];
  }
  return kAcQlookup[((q - 1) % 24) + 1] << ((q - 1) / 24);
}

AV2ENC_DEBLOCK_HD int get_q(int qindex, int delta) {
  if (qindex == 0 && delta <= 0) {
    return kAcQlookup[0];
  }
  const int qc = clip3(1, kMaxQ, qindex + delta);
  return qlookup(qc);
}

AV2ENC_DEBLOCK_HD int q_clamped(int qindex, int delta) {
  if (qindex == 0 && delta <= 0) {
    return 0;
  }
  return clip3(1, kMaxQ, qindex + delta);
}

// §7.17.6 adaptive filter strength selection.  Returns lvlSeg; also derives the
// §7.17.5 qThr/side pair.
AV2ENC_DEBLOCK_HD void adaptive_filter_strength(const DeblockOptions& opts,
                                                int plane, int pass,
                                                int qindex, int* q_thr,
                                                int* side) {
  int delta = 0;
  if (plane == 1) {
    delta = opts.delta_q_u_ac + opts.base_uv_ac_delta_q;
  } else if (plane == 2) {
    delta = opts.delta_q_v_ac + opts.base_uv_ac_delta_q;
  }
  const int qc = q_clamped(qindex, delta);
  const int i = (plane == 0) ? pass : (plane + 1);
  const int lvl = qc + kDfDeltaScale * opts.df_delta_q[i];

  *q_thr = round2(get_q(lvl, 0), 3) >> 6;
  const int q_ind =
      clip3(0, kMaxSideTable - 1, lvl - 24 * (opts.bit_depth - 8));
  *side = imax(kSideThresholds[q_ind] + (1 << (12 - opts.bit_depth)), 0) >>
          (13 - opts.bit_depth);
}

// §7.17.3 filter maximum width.
AV2ENC_DEBLOCK_HD void filter_max_width(int plane, int filter_size, int sb_edge,
                                        int* max_width_pos,
                                        int* max_width_neg) {
  if (filter_size <= 4) {
    *max_width_pos = 1;
  } else if (filter_size == 8) {
    *max_width_pos = 3;
  } else if (filter_size == 16) {
    *max_width_pos = (plane != 0) ? 4 : 6;
  } else {
    *max_width_pos = (plane != 0) ? 4 : 8;
  }
  if (sb_edge) {
    *max_width_neg = imin(*max_width_pos, (plane != 0) ? 2 : 6);
  } else {
    *max_width_neg = *max_width_pos;
  }
}

// §7.17.2 filt_max_size().
AV2ENC_DEBLOCK_HD TxSize filt_max_size(int pass, bool is_tx_edge, TxSize tx_sz,
                                       TxSize sub_pu_size,
                                       bool* is_sub_pu_edge) {
  *is_sub_pu_edge = true;
  const bool use_width = (pass == 0);
  const int cur = use_width ? tx_width(tx_sz) : tx_height(tx_sz);
  const int sub = use_width ? tx_width(sub_pu_size) : tx_height(sub_pu_size);
  if (cur < sub) {
    *is_sub_pu_edge = false;
    return tx_sz;
  }
  if (!is_tx_edge && cur == 8) {
    return TxSize::k4x4;
  }
  if (!is_tx_edge && cur == 16 && sub == 16) {
    return TxSize::k8x8;
  }
  return sub_pu_size;
}

AV2ENC_DEBLOCK_HD bool is_tile_start(const std::uint16_t* starts, int n,
                                     int value) {
  if (starts == nullptr) {
    return false;
  }
  for (int i = 0; i < n; ++i) {
    if (static_cast<int>(starts[i]) == value) {
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Pixel helpers.
// ---------------------------------------------------------------------------
AV2ENC_DEBLOCK_HD int sample_at(const std::uint8_t* plane, int stride, int x,
                                int y) {
  return static_cast<int>(plane[static_cast<std::ptrdiff_t>(y) * stride + x]);
}

AV2ENC_DEBLOCK_HD void write_sample(std::uint8_t* plane, int stride, int x,
                                    int y, int value) {
  plane[static_cast<std::ptrdiff_t>(y) * stride + x] =
      static_cast<std::uint8_t>(clip1_8(value));
}

// §7.17.7.2 filter choice.  Returns the chosen filter width (samples on each
// side of the boundary, capped by maxWidthPos/maxWidthNeg), or 0.
AV2ENC_DEBLOCK_HD int filter_choice(const std::uint8_t* plane, int stride,
                                    int x, int y, int q_thr, int side_thr,
                                    int dx, int dy, int max_width_neg,
                                    int max_width_pos) {
  if (q_thr == 0 || side_thr == 0) {
    return 0;
  }
  const int count = kMiSize;
  const int max_samples_pos = clip3(3, 8, max_width_pos + 1);
  const int max_samples_neg = clip3(3, 8, max_width_neg + 1);

  int s[16];
  int t[16];
  constexpr int kOff = 8;

  const int x2 = x + (count - 1) * dy;
  const int y2 = y + (count - 1) * dx;

  for (int dist = 0; dist < max_samples_pos; ++dist) {
    s[kOff + dist] =
        sample_at(plane, stride, x + dist * dx, y + dist * dy);
    t[kOff + dist] =
        sample_at(plane, stride, x2 + dist * dx, y2 + dist * dy);
  }
  for (int dist = 0; dist < max_samples_neg; ++dist) {
    s[kOff - dist - 1] =
        sample_at(plane, stride, x - (dist + 1) * dx, y - (dist + 1) * dy);
    t[kOff - dist - 1] =
        sample_at(plane, stride, x2 - (dist + 1) * dx, y2 - (dist + 1) * dy);
  }

  int second_deriv[4];  // indices -2..1 stored at second_deriv[dist + 2]
  for (int dist = -2; dist < 2; ++dist) {
    const int sp0 = s[kOff + dist - 1];
    const int sq0 = s[kOff + dist];
    const int sq1 = s[kOff + dist + 1];
    const int tp0 = t[kOff + dist - 1];
    const int tq0 = t[kOff + dist];
    const int tq1 = t[kOff + dist + 1];
    const int deriv_s = iabs(sp0 - (sq0 << 1) + sq1);
    const int deriv_t = iabs(tp0 - (tq0 << 1) + tq1);
    second_deriv[dist + 2] = (deriv_s + deriv_t + 1) >> 1;
  }
  const int sd_m2 = second_deriv[0];
  const int sd_m1 = second_deriv[1];
  const int sd_0 = second_deriv[2];
  const int sd_1 = second_deriv[3];

  if (sd_m2 > side_thr || sd_1 > side_thr) {
    return 0;
  }
  if (max_width_pos == 1) {
    return 1;
  }

  const int side_thr2 = side_thr >> 2;
  if (sd_m2 > side_thr2 || sd_1 > side_thr2) {
    return 1;
  }
  if (sd_m1 + sd_0 > q_thr * 4) {
    return 1;
  }

  const int side_thr3 = side_thr >> 3;
  if (sd_m2 > side_thr3 || sd_1 > side_thr3) {
    return 2;
  }
  if (sd_m1 + sd_0 > q_thr * 3) {
    return 2;
  }

  const int end_thr = (side_thr * 3) >> 4;
  if (max_width_neg > 2) {
    const int deriv_s =
        iabs(s[kOff - 1] - s[kOff - 4] -
             3 * (s[kOff - 1] - s[kOff - 2]));
    const int deriv_t =
        iabs(t[kOff - 1] - t[kOff - 4] -
             3 * (t[kOff - 1] - t[kOff - 2]));
    if (((deriv_s + deriv_t + 1) >> 1) > end_thr) {
      return 2;
    }
  }
  {
    const int deriv_s =
        iabs(s[kOff] - s[kOff + 3] - 3 * (s[kOff] - s[kOff + 1]));
    const int deriv_t =
        iabs(t[kOff] - t[kOff + 3] - 3 * (t[kOff] - t[kOff + 1]));
    if (((deriv_s + deriv_t + 1) >> 1) > end_thr) {
      return 2;
    }
  }

  if (max_width_pos == 3) {
    return 3;
  }

  const int transition = (sd_m1 + sd_0) << 4;
  int prev_dist = 3;
  for (int dist = 4; dist <= max_width_pos; dist += 2) {
    const int q_thr4 = q_thr * kQFirst[dist - 4];
    const int end_thr4 = (side_thr * dist) >> 4;
    if (transition > q_thr4) {
      return prev_dist;
    }
    const int dist2 = imin(7, dist);
    if (max_width_neg >= dist2) {
      const int deriv_s = iabs(s[kOff - 1] - s[kOff - dist2 - 1] -
                               dist2 * (s[kOff - 1] - s[kOff - 2]));
      const int deriv_t = iabs(t[kOff - 1] - t[kOff - dist2 - 1] -
                               dist2 * (t[kOff - 1] - t[kOff - 2]));
      if (((deriv_s + deriv_t + 1) >> 1) > end_thr4) {
        return prev_dist;
      }
    }
    {
      const int deriv_s = iabs(s[kOff] - s[kOff + dist2] -
                               dist2 * (s[kOff] - s[kOff + 1]));
      const int deriv_t = iabs(t[kOff] - t[kOff + dist2] -
                               dist2 * (t[kOff] - t[kOff + 1]));
      if (((deriv_s + deriv_t + 1) >> 1) > end_thr4) {
        return prev_dist;
      }
    }
    prev_dist = dist;
  }
  return max_width_pos;
}

// §7.17.7.1 sample filtering.
AV2ENC_DEBLOCK_HD void sample_filtering(std::uint8_t* plane, int stride, int x,
                                        int y, int q_thr, int dx, int dy,
                                        int max_width_neg, int max_width_pos,
                                        int prev_lossless,
                                        int curr_lossless) {
  const int width = imax(max_width_neg, max_width_pos);
  const int q0 = sample_at(plane, stride, x, y);
  const int q1 = sample_at(plane, stride, x + dx, y + dy);
  const int p0 = sample_at(plane, stride, x - dx, y - dy);
  const int p1 = sample_at(plane, stride, x - 2 * dx, y - 2 * dy);

  const int q_thr_clamp = q_thr * kQThreshMults[width - 1];
  int delta_m2 = p1 - q1 + 3 * (q0 - p0);
  delta_m2 *= 4;
  delta_m2 = clip3(-q_thr_clamp, q_thr_clamp, delta_m2);

  const int delta_video_neg = delta_m2 * kWMult[max_width_neg - 1];
  const int delta_video_pos = delta_m2 * kWMult[max_width_pos - 1];

  for (int i = 0; i < width; ++i) {
    const int diff_neg = round2(delta_video_neg * (max_width_neg - i),
                                3 + kDfShift);
    const int diff_pos = round2(delta_video_pos * (max_width_pos - i),
                                3 + kDfShift);
    const int qy = y + i * dy;
    const int qx = x + i * dx;
    if (!curr_lossless) {
      write_sample(plane, stride, qx, qy,
                   sample_at(plane, stride, qx, qy) - diff_pos);
    }
    if (i < max_width_neg && !prev_lossless) {
      const int pi = -i - 1;
      const int py = y + pi * dy;
      const int px = x + pi * dx;
      write_sample(plane, stride, px, py,
                   sample_at(plane, stride, px, py) + diff_neg);
    }
  }
}

// ---------------------------------------------------------------------------
// One edge call: §7.17.2 edge deblocking filter process.
//
// `Apply` selects between the modifying path and a pure inspection path used
// by tests.  The function returns the decision metadata either way.
// ---------------------------------------------------------------------------
template <bool Apply>
AV2ENC_DEBLOCK_HD DeblockEdgeInfo deblock_edge_core(
    const DeblockCoreFrame& frame, const DeblockOptions& opts, int plane,
    int pass, int row, int col) {
  DeblockEdgeInfo info{};

  const int sub_x = (plane != 0) ? frame.sub_x : 0;
  const int sub_y = (plane != 0) ? frame.sub_y : 0;
  const int dx = (pass == 0) ? 1 : 0;
  const int dy = (pass == 0) ? 0 : 1;
  const int x = col * kMiSize;
  const int y = row * kMiSize;

  info.tile_vert_edge =
      (pass == 0) && is_tile_start(frame.tile_col_starts, frame.num_tile_cols,
                                   col);
  info.tile_horz_edge =
      (pass == 1) && is_tile_start(frame.tile_row_starts, frame.num_tile_rows,
                                   row);
  const int horz64_edge = (pass == 1) && ((y % 64) == 0);
  info.sb_edge = horz64_edge || info.tile_vert_edge;

  if (opts.disable_loopfilters_across_tiles) {
    if (pass == 0 && info.tile_vert_edge) {
      return info;
    }
    if (pass == 1 && info.tile_horz_edge) {
      return info;
    }
  }
  if (pass == 0 && x == 0) {
    return info;
  }
  if (pass == 1 && y == 0) {
    return info;
  }
  info.on_screen = 1;

  const int x_p = x >> sub_x;
  const int y_p = y >> sub_y;
  const int prev_row = row - (dy << sub_y);
  const int prev_col = col - (dx << sub_x);

  const DeblockMi* mi = frame.mi[plane];
  const std::ptrdiff_t grid = static_cast<std::ptrdiff_t>(frame.mi_cols);
  const DeblockMi& cur = mi[static_cast<std::ptrdiff_t>(row) * grid + col];
  const DeblockMi& prev =
      mi[static_cast<std::ptrdiff_t>(prev_row) * grid + prev_col];

  const int base_row = cur.block_row_base;
  const int base_col = cur.block_col_base;
  const int base_y = (base_row * kMiSize) >> sub_y;
  const int base_x = (base_col * kMiSize) >> sub_x;

  bool is_sub_pu_edge =
      opts.allow_df_sub_pu &&
      (cur.sub_pu_col_base != prev.sub_pu_col_base ||
       cur.sub_pu_row_base != prev.sub_pu_row_base);

  TxSize tx_sz = cur.tx_size;
  TxSize prev_tx_sz = prev.tx_size;
  const TxSize sub_pu_size = cur.sub_pu_size;
  const TxSize prev_sub_pu_size = prev.sub_pu_size;

  // Chroma information is held in the bottom-right mode info.
  int adj_row = row;
  int adj_col = col;
  if (plane > 0) {
    adj_row = base_row + static_cast<int>(cur.block_h4) - 1;
    adj_col = base_col + static_cast<int>(cur.block_w4) - 1;
  }
  const DeblockMi& adj =
      mi[static_cast<std::ptrdiff_t>(adj_row) * grid + adj_col];

  int skip = adj.skip;
  if (adj.region_intra || (opts.frame_is_intra && opts.enable_sdp)) {
    skip = 0;
  }

  const int x_r = x_p - base_x;
  const int y_r = y_p - base_y;
  const bool is_block_edge =
      (pass == 0 && x_r == 0) || (pass == 1 && y_r == 0);
  const bool is_tx_edge =
      (cur.tx_col_base != prev.tx_col_base) ||
      (cur.tx_row_base != prev.tx_row_base);

  if (is_sub_pu_edge) {
    bool sub_edge = false;
    tx_sz = filt_max_size(pass, is_tx_edge, tx_sz, sub_pu_size, &sub_edge);
    is_sub_pu_edge = sub_edge;
    prev_tx_sz = filt_max_size(pass, is_tx_edge, prev_tx_sz, prev_sub_pu_size,
                               &sub_edge);
  }
  if (is_block_edge) {
    is_sub_pu_edge = false;
  }

  int curr_q = 0;
  int curr_side = 0;
  adaptive_filter_strength(opts, plane, pass, adj.qindex, &curr_q, &curr_side);
  int prev_q = 0;
  int prev_side = 0;
  adaptive_filter_strength(opts, plane, pass, prev.qindex, &prev_q,
                           &prev_side);

  const int curr_lossless = adj.lossless;
  const int prev_lossless = prev.lossless;

  int apply_filter = 0;
  const bool curr_ok = (curr_q != 0 && curr_side != 0);
  const bool prev_ok = (prev_q != 0 && prev_side != 0);
  if (!is_tx_edge && !is_sub_pu_edge) {
    apply_filter = 0;
  } else if (!curr_ok && !prev_ok) {
    apply_filter = 0;
  } else if (is_block_edge || skip == 0 || is_sub_pu_edge) {
    apply_filter = 1;
  } else {
    apply_filter = 0;
  }

  info.is_block_edge = is_block_edge ? 1 : 0;
  info.is_tx_edge = is_tx_edge ? 1 : 0;
  info.is_sub_pu_edge = is_sub_pu_edge ? 1 : 0;
  info.skip = skip;
  info.curr_lossless = curr_lossless;
  info.prev_lossless = prev_lossless;
  info.curr_q = curr_q;
  info.prev_q = prev_q;
  info.curr_side = curr_side;
  info.prev_side = prev_side;
  info.apply_filter = apply_filter;
  if (!apply_filter) {
    return info;
  }

  int filter_size = 0;
  if (pass == 0) {
    filter_size = imin(tx_width(prev_tx_sz), tx_width(tx_sz));
  } else {
    filter_size = imin(tx_height(prev_tx_sz), tx_height(tx_sz));
  }

  const int plane_w = (frame.mi_cols * kMiSize) >> sub_x;
  const int plane_h = (frame.mi_rows * kMiSize) >> sub_y;
  if (plane == 0) {
    if (x_p + dx * 16 > plane_w || y_p + dy * 16 > plane_h) {
      filter_size = imin(filter_size, 16);
    }
  } else {
    if (x_p + dx * 8 > plane_w || y_p + dy * 8 > plane_h) {
      filter_size = imin(filter_size, 8);
    }
  }

  if (prev_lossless && curr_lossless) {
    return info;
  }

  int q_thr = 0;
  int side = 0;
  if (curr_q && prev_q) {
    q_thr = (curr_q + prev_q + 1) >> 1;
  } else {
    q_thr = imax(curr_q, prev_q);
  }
  if (curr_side && prev_side) {
    side = (curr_side + prev_side + 1) >> 1;
  } else {
    side = imax(curr_side, prev_side);
  }
  if (is_sub_pu_edge && !is_tx_edge) {
    q_thr >>= 3;
    side >>= 3;
  }

  int max_width_pos = 0;
  int max_width_neg = 0;
  filter_max_width(plane, filter_size, info.sb_edge, &max_width_pos,
                   &max_width_neg);

  std::uint8_t* data = frame.plane[plane];
  const int stride = frame.stride[plane];
  const int width =
      filter_choice(data, stride, x_p, y_p, q_thr, side, dx, dy,
                    max_width_neg, max_width_pos);

  info.filter_size = filter_size;
  info.max_width_pos = max_width_pos;
  info.max_width_neg = max_width_neg;
  info.q_thr = q_thr;
  info.side = side;
  info.width = width;
  if (width == 0) {
    return info;
  }

  if (Apply) {
    const int neg = imin(width, max_width_neg);
    const int pos = imin(width, max_width_pos);
    for (int i = 0; i < kMiSize; ++i) {
      sample_filtering(data, stride, x_p + dy * i, y_p + dx * i, q_thr, dx, dy,
                       neg, pos, prev_lossless, curr_lossless);
    }
  }
  return info;
}

// Full normative traversal for the CPU reference:
//   for plane, for pass, for row, for col -> edge
// (vertical boundaries precede horizontal boundaries within each plane).
AV2ENC_DEBLOCK_HD void deblock_frame_core(const DeblockCoreFrame& frame,
                                          const DeblockOptions& opts) {
  for (int plane = 0; plane < 3; ++plane) {
    const int row_step = (plane == 0) ? 1 : (1 << frame.sub_y);
    const int col_step = (plane == 0) ? 1 : (1 << frame.sub_x);
    for (int pass = 0; pass < 2; ++pass) {
      const int flag_index = (plane == 0) ? pass : (plane + 1);
      if (!opts.apply_deblocking_filter[flag_index]) {
        continue;
      }
      for (int row = 0; row < frame.mi_rows; row += row_step) {
        for (int col = 0; col < frame.mi_cols; col += col_step) {
          (void)deblock_edge_core<true>(frame, opts, plane, pass, row, col);
        }
      }
    }
  }
}

}  // namespace av2enc::deblock

#endif  // AV2ENC_ENCODER_FILTERS_DEBLOCK_DEBLOCK_CORE_H_
