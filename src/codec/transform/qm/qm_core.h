// Scalar AV2 §7.14.4 dequantization math shared by the CPU reference and the
// CUDA kernel.  Keeping one `__host__ __device__` implementation makes the
// integer pipeline bit-exact by construction; the tests still check it against
// independently written formula expectations.
//
// The only non-integer operations are absent: every step is the normative
// Round2() on integers.  No floating point is involved, so -fmad / FP
// contraction cannot change the result.
#pragma once

#include <cstdint>

#include "codec/transform/qm/qm_constants.h"

#if defined(__CUDACC__)
#define AV2ENC_QM_HD __host__ __device__
#else
#define AV2ENC_QM_HD
#endif

namespace av2enc::qm::core {

// AV2 TX sizes in enum order 0..24.
AV2ENC_QM_HD inline bool valid_tx_size(int tx_size) {
  return tx_size >= 0 && tx_size < kNumTxSizes;
}

AV2ENC_QM_HD inline int coded_width(int tx_size) {
  switch (tx_size) {
    case 0: return 4;
    case 1: return 8;
    case 2: return 16;
    case 3: return 32;
    case 4: return 64;
    case 5: return 4;
    case 6: return 8;
    case 7: return 8;
    case 8: return 16;
    case 9: return 16;
    case 10: return 32;
    case 11: return 32;
    case 12: return 64;
    case 13: return 4;
    case 14: return 16;
    case 15: return 8;
    case 16: return 32;
    case 17: return 16;
    case 18: return 64;
    case 19: return 4;
    case 20: return 32;
    case 21: return 8;
    case 22: return 64;
    case 23: return 4;
    case 24: return 64;
    default: return 0;
  }
}

AV2ENC_QM_HD inline int coded_height(int tx_size) {
  switch (tx_size) {
    case 0: return 4;
    case 1: return 8;
    case 2: return 16;
    case 3: return 32;
    case 4: return 64;
    case 5: return 8;
    case 6: return 4;
    case 7: return 16;
    case 8: return 8;
    case 9: return 32;
    case 10: return 16;
    case 11: return 64;
    case 12: return 32;
    case 13: return 16;
    case 14: return 4;
    case 15: return 32;
    case 16: return 8;
    case 17: return 64;
    case 18: return 16;
    case 19: return 32;
    case 20: return 4;
    case 21: return 64;
    case 22: return 8;
    case 23: return 64;
    case 24: return 4;
    default: return 0;
  }
}

AV2ENC_QM_HD inline int adjusted_width(int tx_size) {
  const int w = coded_width(tx_size);
  return w < 32 ? w : 32;
}

AV2ENC_QM_HD inline int adjusted_height(int tx_size) {
  const int h = coded_height(tx_size);
  return h < 32 ? h : 32;
}

// §4.8 Round2: (value + (1 << (n-1))) >> n.  n <= 0 is a pass-through.
AV2ENC_QM_HD inline std::int32_t round2(std::int32_t value, int n) {
  if (n <= 0) return value;
  return (value + (1 << (n - 1))) >> n;
}

// §7.14.4 `PlaneTxType < IDTX`: only the 2-D transform types use a QM.  IDTX
// (9) and the six directional 1-D types do not.
AV2ENC_QM_HD inline bool tx_uses_qm(int tx_type) { return tx_type < 9; }

// §7.14.4 get_tx_class(PlaneTxType) == TX_CLASS_2D.  This is *not* the same
// predicate as tx_uses_qm(): get_tx_class() returns VERT/HORIZ only for the
// six V_*/H_* types (10..15), so IDTX (9) is TX_CLASS_2D and gets the TCQ
// dqDenom shift even though it does not use a quantizer matrix.
AV2ENC_QM_HD inline bool tx_class_is_2d(int tx_type) {
  switch (tx_type) {
    case 10:  // V_DCT
    case 11:  // H_DCT
    case 12:  // V_ADST
    case 13:  // H_ADST
    case 14:  // V_FLIPADST
    case 15:  // H_FLIPADST
      return false;
    default:
      return true;
  }
}

AV2ENC_QM_HD inline std::int32_t clip3(std::int32_t low, std::int32_t high,
                                       std::int32_t value) {
  return value < low ? low : (value > high ? high : value);
}

// Qm_Offset as a device-safe switch (a namespace-scope array indexed by a
// runtime value is a host variable and cannot be read from device code without
// extra compiler flags).
AV2ENC_QM_HD inline int offset_value(int tx_size) {
  switch (tx_size) {
    case 0: return 0;
    case 1: return 16;
    case 2: return 80;
    case 3: return 336;
    case 4: return 336;
    case 5: return 1360;
    case 6: return 1392;
    case 7: return 1424;
    case 8: return 1552;
    case 9: return 1680;
    case 10: return 2192;
    case 11: return 336;
    case 12: return 336;
    case 13: return 2704;
    case 14: return 2768;
    case 15: return 2832;
    case 16: return 3088;
    case 17: return 1680;
    case 18: return 2192;
    case 19: return 3344;
    case 20: return 3472;
    case 21: return 2832;
    case 22: return 3088;
    case 23: return 3344;
    case 24: return 3472;
    default: return -1;
  }
}

// One dequantization job in the compact form both the CPU and GPU consume.
// `qtable` points at kQuantizerMatrix; `user_qm` is the flattened
// [level][t][plane][i][j] array (15 * 3 * 3 * 8 * 8 bytes), or null.
struct CoeffParams {
  int tx_size = 0;
  int plane = 0;        // 0 = Y, 1 = U, 2 = V
  int tx_type = 0;      // AV2 TX_TYPE; < 9 uses a QM (IDTX is still 2-D)
  int seg_qm_level = kNoQmLevel;  // SegQMLevel[plane][segment_id]
  int qm_level = kNoQmLevel;      // qm_y[0] / qm_u[0] / qm_v[0] for >8x8
  int dc_quant = 0;     // get_dc_quant(plane)
  int ac_quant = 0;     // get_ac_quant(plane)
  int bit_depth = 8;    // BitDepth for the §7.14.4 Clip3 bound
  bool using_qmatrix = false;
  bool qm_data_present = false;  // QmDataPresent[segLvl]
  bool allow_tcq = false;
  bool lossless = false;
  bool use_fsc = false;
  const std::uint8_t* qtable = nullptr;
  const std::uint8_t* user_qm = nullptr;
};

AV2ENC_QM_HD inline int user_qm_index(int level, int t, int plane, int i,
                                      int j) {
  return ((((level * 3 + t) * 3 + plane) * 8 + i) * 8 + j);
}

// `tw > 8 || th > 8` selects the frame-level qm_y/qm_u/qm_v[0] entries;
// otherwise the per-segment SegQMLevel applies (§7.14.4).
AV2ENC_QM_HD inline int resolve_seg_level(const CoeffParams& p) {
  const int tw = adjusted_width(p.tx_size);
  const int th = adjusted_height(p.tx_size);
  if (tw > 8 || th > 8) {
    if (p.plane == 0) return p.qm_level;
    // qm_u for U, qm_v for V.  The caller supplies the correct one in
    // qm_level for planes 1 and 2.
    return p.qm_level;
  }
  return p.seg_qm_level;
}

AV2ENC_QM_HD inline bool use_qm(const CoeffParams& p, int seg_level) {
  return p.using_qmatrix && tx_uses_qm(p.tx_type) &&
         seg_level < kNoQmLevel;
}

AV2ENC_QM_HD inline bool use_user_qm(const CoeffParams& p, int seg_level) {
  const int tw = adjusted_width(p.tx_size);
  const int th = adjusted_height(p.tx_size);
  return use_qm(p, seg_level) && tw <= 8 && th <= 8 && p.qm_data_present &&
         p.user_qm != nullptr;
}

// The matrix coefficient m for adjusted position (i, j), per §7.14.4.
AV2ENC_QM_HD inline int matrix_value(const CoeffParams& p, int seg_level,
                                     int i, int j) {
  const int tw = adjusted_width(p.tx_size);
  const int th = adjusted_height(p.tx_size);
  if (use_user_qm(p, seg_level)) {
    if (tw < th) {
      return p.user_qm[user_qm_index(seg_level, 2, p.plane, i, j)];
    }
    if (tw > th) {
      return p.user_qm[user_qm_index(seg_level, 1, p.plane, i, j)];
    }
    const int qi = i * 8 / th;
    const int qj = j * 8 / tw;
    return p.user_qm[user_qm_index(seg_level, 0, p.plane, qi, qj)];
  }
  if (p.qtable == nullptr) return 0;
  const int plane_type = p.plane > 0 ? 1 : 0;
  const int index = offset_value(p.tx_size) + i * tw + j;
  return p.qtable[(seg_level * kQmPlaneTypes + plane_type) *
                      kQmTableEntries +
                  index];
}

// q2 = Round2(q * m, 5) when QM is active, else q.
AV2ENC_QM_HD inline std::int32_t quant_value(const CoeffParams& p,
                                             int seg_level, int i, int j) {
  const std::int32_t q = (i == 0 && j == 0) ? p.dc_quant : p.ac_quant;
  if (!use_qm(p, seg_level)) return q;
  const std::int32_t m = matrix_value(p, seg_level, i, j);
  return round2(q * m, 5);
}

// §7.14.4 dqDenom.  `pels` is the *coded* size product; `allow_tcq` adds the
// extra luma 2-D shift.  FSC is not implemented by this component, but the
// input is kept so a future caller cannot silently diverge.
AV2ENC_QM_HD inline int dequant_denom(const CoeffParams& p) {
  const int pels = coded_width(p.tx_size) * coded_height(p.tx_size);
  int shift = (pels > 256 ? 1 : 0) + (pels > 1024 ? 1 : 0);
  if (p.allow_tcq && p.plane == 0 && !p.lossless &&
      tx_class_is_2d(p.tx_type) && !p.use_fsc) {
    ++shift;
  }
  return 1 << shift;
}

// One dequantized coefficient from a signed level and its q2.  Step 8 of
// §7.14.4 clips the result to [-(1 << (7 + BitDepth)),
// (1 << (7 + BitDepth)) - 1] before it is stored in Dequant.
AV2ENC_QM_HD inline std::int32_t dequant_coeff(std::int32_t level,
                                               std::int32_t q2, int denom,
                                               int bit_depth) {
  const std::int32_t sign = level < 0 ? -1 : 1;
  const std::int64_t magnitude =
      level < 0 ? -static_cast<std::int64_t>(level)
                : static_cast<std::int64_t>(level);
  const std::int64_t high = magnitude * static_cast<std::int64_t>(q2);
  const std::int32_t masked =
      static_cast<std::int32_t>(high & 0xFFFFFFLL);
  const std::int32_t dq = round2(masked, 3);  // QUANT_TABLE_BITS
  const std::int32_t value = sign * (dq / denom);
  const std::int32_t bound = 1 << (7 + bit_depth);
  return clip3(-bound, bound - 1, value);
}

// Shared block drivers.  The CPU reference and the CUDA kernel both call
// these, so a divergence is impossible.
AV2ENC_QM_HD inline void build_q2_block(const CoeffParams& p,
                                        std::int32_t* q2) {
  const int tw = adjusted_width(p.tx_size);
  const int th = adjusted_height(p.tx_size);
  const int seg_level = resolve_seg_level(p);
  for (int i = 0; i < th; ++i) {
    for (int j = 0; j < tw; ++j) {
      q2[i * tw + j] = quant_value(p, seg_level, i, j);
    }
  }
}

AV2ENC_QM_HD inline void dequantize_block(const CoeffParams& p,
                                          const std::int32_t* levels,
                                          std::int32_t* dequant) {
  const int tw = adjusted_width(p.tx_size);
  const int th = adjusted_height(p.tx_size);
  const int seg_level = resolve_seg_level(p);
  const int denom = dequant_denom(p);
  for (int i = 0; i < th; ++i) {
    for (int j = 0; j < tw; ++j) {
      const int index = i * tw + j;
      const std::int32_t q2 = quant_value(p, seg_level, i, j);
      dequant[index] =
          dequant_coeff(levels[index], q2, denom, p.bit_depth);
    }
  }
}

}  // namespace av2enc::qm::core
