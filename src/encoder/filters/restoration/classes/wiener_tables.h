// Encoder-side Wiener loop-restoration constants and shared helpers.
//
// Everything here is transcribed from the AV2 v1.0.0 specification text
// (commit 38f28e7), not from an encoder implementation:
//
//   §5.18.7.11  Decode_Num_Filter_Classes / Encode_Num_Filter_Classes
//   §5.20.10.6  Wiener_Ns_Taps_Present, Shuffled_Index, predict_group()
//   §7.14.2     get_q(qindex, 0) (values generated into wiener_qlookup.h)
//   §7.20.3     Wiener_Ns_Config_Y / Wiener_Ns_Config_Uv (in restoration_common)
//   §7.20.4     Pc_Wiener_Normalizer, Mode_Weights, Mode_Offsets, lead/lag
//
// The Pc_Wiener_Lut_To_Class, Pc_Wiener_Sub_Classify and
// Pc_Wiener_Sub_Classify2 tables are *not* present in the specification text
// (see §9.8: "defined in the ... header file").  They are never transcribed
// here; callers supply them through PcWienerTables.

#ifndef AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_TABLES_H_
#define AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_TABLES_H_

#include <cstdint>

#include "encoder/filters/restoration/restoration_common.h"
#include "encoder/filters/restoration/classes/wiener_qlookup.h"

namespace av2enc::restoration::classes {

// §3 constants used by the encoder-side search.
constexpr int kLrBankSize = 4;                 // LR_BANK_SIZE
constexpr int kNumPcWienerFilters = 64;        // NUM_PC_WIENER_FILTERS
constexpr int kNumPcWienerLutClasses = 256;    // NUM_PC_WIENER_LUT_CLASSES
constexpr int kPcWienerLutEntries = 4096;      // 4 features x 3 bits each
constexpr int kPcWienerCoeffs = 13;            // PC_WIENER_COEFFS
constexpr int kPcWienerPrecBits = 7;           // PC_WIENER_PREC_BITS
constexpr int kPcWienerPrecFeature = 14;       // PC_WIENER_PREC_FEATURE
constexpr int kPcWienerLead = 1;               // PC_WIENER_LEAD
constexpr int kPcWienerLag = 4;                // PC_WIENER_LAG
constexpr int kMaxLrRefs = 16;                 // NUM_REF_FRAMES slots
constexpr int kRestrictedOrderHint = -1;       // RESTRICTED_OH

// §5.18.7.11 Decode_Num_Filter_Classes[8] lives in restoration_common.h as
// kDecodeNumFilterClasses.  The inverse mapping is Encode_Num_Filter_Classes.
inline constexpr std::int8_t kEncodeNumFilterClasses[17] = {
    0, 0, 1, 2, 3, 0, 4, 0, 5, 0, 0, 0, 6, 0, 0, 0, 7};

inline bool valid_num_filter_classes(int n) {
  for (int i = 0; i < 8; ++i) {
    if (kDecodeNumFilterClasses[i] == n) {
      return true;
    }
  }
  return false;
}

inline int encode_num_filter_classes_idx(int num_classes) {
  if (!valid_num_filter_classes(num_classes)) {
    return -1;
  }
  return kEncodeNumFilterClasses[num_classes];
}

inline int decode_num_filter_classes(int idx) {
  if (idx < 0 || idx > 7) {
    return 0;
  }
  return kDecodeNumFilterClasses[idx];
}

// §5.20.10.6 Wiener_Ns_Taps_Present[2][4][WIENER_NS_CHROMA_COEFFS].
inline constexpr std::int8_t kWienerNsTapsPresent[2][4]
                                                 [kWienerNsChromaCoeffs] = {
    {{1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
     {1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0}},
    {{1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
};

// Number of coefficient groups (design-matrix columns) for a plane.
inline int wiener_n_coeffs(int plane) {
  return plane == 0 ? kWienerNsLumaCoeffs : kWienerNsChromaCoeffs;
}

// Number of wiener_ns_length values (subset 0..n-1) for a plane.
inline int wiener_num_subsets(int plane) { return plane == 0 ? 4 : 3; }

inline bool wiener_tap_present(int plane, int subset, int j) {
  if (plane < 0 || plane > 2 || subset < 0 || subset > 3 || j < 0 ||
      j >= kWienerNsChromaCoeffs) {
    return false;
  }
  return kWienerNsTapsPresent[plane == 0 ? 0 : 1][subset][j] != 0;
}

// §5.18.7.11 num_filter_classes_idx is only signalled for luma frame filters.
// Chroma frame filters always use class 0 (§7.20.3).
inline int wiener_filter_set_index(int base_q_idx) {
  return lr_filter_set_index(base_q_idx);
}

// §7.20.4 Pc_Wiener_Normalizer[PC_WIENER_NUM_FEATURES + 1].
inline constexpr std::int32_t kPcWienerNormalizer[5] = {0, 3739, 3273, 3074,
                                                        7};

// §7.20.4 Mode_Weights[PC_WIENER_NUM_FEATURES][3].
inline constexpr std::int32_t kModeWeights[4][3] = {
    {-527, 15325, 321},
    {26436, -17705, 17905},
    {366, -147, -194},
    {202, -267, -179},
};

// §7.20.4 Mode_Offsets[PC_WIENER_NUM_FEATURES].
inline constexpr std::int32_t kModeOffsets[4] = {-547, -21565, -573, -680};

// §5.20.10.6 Shuffled_Index[64].  Only needed to translate PC-Wiener filters
// into Wiener-NS candidates; the Pc_Wiener_Filters table itself is absent.
inline constexpr std::int8_t kShuffledIndex[64] = {
    16, 7,  58, 21, 12, 61, 26, 38, 18, 30, 50, 45, 23, 49, 43, 62,
    42, 54, 27, 36, 17, 44, 32, 34, 4,  24, 52, 31, 37, 11, 33, 19,
    35, 6,  22, 53, 63, 25, 41, 47, 1,  59, 0,  28, 40, 55, 48, 8,
    5,  51, 9,  46, 56, 60, 15, 2,  13, 14, 57, 29, 3,  20, 39, 10};

// §7.14.2 get_q(qindex, 0) for 8-bit.  MaxQ is 255 for 8-bit streams.
inline int wiener_get_q(int qindex, int delta = 0, int max_q = 255) {
  if (qindex == 0 && delta <= 0) {
    return kAcQlookup[0];
  }
  int q_clamped = clip3(1, max_q, qindex + delta);
  if (q_clamped < 0 || q_clamped > 255) {
    return 0;
  }
  return kQlookupQ0To255[q_clamped];
}

// §4.8 Round2Signed, used by §7.20.4.
inline int wiener_round2_signed(int x, int n) {
  return x >= 0 ? round2(x, n) : -round2(-x, n);
}

// §7.20.4 get_qval_given_tskip(qindex, tskip, i), 8-bit specialisation.
// QUANT_TABLE_BITS is 3, so qstepShift is 13 and BitDepth - 8 is 0.
inline std::int64_t wiener_qval_given_tskip(int qindex, int tskip, int i) {
  if (i < 0 || i > 3) {
    return 0;
  }
  const int qstep_unshifted = wiener_get_q(qindex, 0);
  const int qstep = round2(qstep_unshifted, 0);
  const int qstep_shift = 3 + 10;
  const int diff_shift = qstep_shift - 8;
  const int prod = round2(tskip * qstep, 8);
  const std::int64_t qval =
      static_cast<std::int64_t>(kModeWeights[i][0]) * (tskip << diff_shift) +
      static_cast<std::int64_t>(kModeWeights[i][1]) * qstep +
      static_cast<std::int64_t>(kModeWeights[i][2]) * prod;
  const int rounded = wiener_round2_signed(static_cast<int>(qval),
                                           qstep_shift);
  return static_cast<std::int64_t>(255) * (kModeOffsets[i] + rounded);
}

// §5.20.10.6 predict_group().
inline int wiener_predict_group(const int counts[3]) {
  int pred = 0;
  for (int i = 1; i <= 2; ++i) {
    if (counts[i] > counts[pred]) {
      pred = i;
    }
  }
  return pred;
}

}  // namespace av2enc::restoration::classes

#endif  // AV2ENC_ENCODER_FILTERS_RESTORATION_CLASSES_WIENER_TABLES_H_
