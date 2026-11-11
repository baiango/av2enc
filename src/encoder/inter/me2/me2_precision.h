// AV2 MV precision model for the extended motion estimator (me2).
//
// This header is the single host/device description of:
//   * the AV2 MV precision enum (MV_PRECISION_EIGHT_PEL .. EIGHTH_PEL),
//   * the normative lower_mv_precision() rounding rule (§5.20.7.13),
//   * conversion between the ME-internal 1/16-sample domain and the
//     bitstream's 1/8-sample MV domain (§7.13.3.14 uses 2*mv for prediction),
//   * the frame-header -> FrameMvPrecision / UsePerBlockMvPrecision mapping
//     (§5.18.2, use_qtr_precision_mv / allow_high_precision_mv /
//     enable_flex_mvres),
//   * the inverse of the per-block pb_mv_precision derivation
//     (§5.20.7.11 / §5.20.7.13) so the encoder can emit a target precision,
//   * a deterministic MVD shell-code bin-count used as a rate proxy.
//
// The AV2 bitstream can signal at most MV_PRECISION_EIGHTH_PEL (1/8 luma
// sample).  The ME's 1/16-sample refinement is an *encoder-side search*
// granularity; every returned MV is lowered to the selected signaled
// precision before it is costed or reported as `mv_x`/`mv_y`.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7).  Constants: MV_LOW = -65536, MV_UPP = 65536
// (MV_IN_USE_BITS = 16).

#ifndef AV2ENC_ENCODER_INTER_ME2_ME2_PRECISION_H_
#define AV2ENC_ENCODER_INTER_ME2_ME2_PRECISION_H_

#include <cstdint>

#include "encoder/inter/me/me_core.h"

namespace av2enc::me2 {

// §7.13.3.14 / §5.4.6 MV_PRECISION_* values, verbatim.
enum MvPrecision {
  kMvPrecisionEightPel = 0,    // 8 luma samples
  kMvPrecisionFourPel = 1,     // 4
  kMvPrecisionTwoPel = 2,      // 2
  kMvPrecisionOnePel = 3,      // 1
  kMvPrecisionHalfPel = 4,     // 1/2
  kMvPrecisionQuarterPel = 5,  // 1/4
  kMvPrecisionEighthPel = 6,   // 1/8 (finest signalable)
  kMvPrecisionCount = 7,
};

constexpr std::int32_t kMvLow = -65536;  // exclusive lower bound, 1/8 units
constexpr std::int32_t kMvUpp = 65536;   // exclusive upper bound, 1/8 units

AV2ENC_ME_HD bool me2_precision_valid(int p) {
  return p >= kMvPrecisionEightPel && p <= kMvPrecisionEighthPel;
}

// Number of low bits removed when lowering an eighth-pel MV to `p`.
AV2ENC_ME_HD int me2_precision_shift(int p) {
  return kMvPrecisionEighthPel - p;
}

// Step, in 1/16-sample units, of an MV at `p`.  Eighth = 2, quarter = 4,
// half = 8, one = 16, ... eight = 128.
AV2ENC_ME_HD int me2_precision_quantum_1_16(int p) {
  return 1 << (kMvPrecisionEighthPel - p + 1);
}

// Floor((x + 2^(n-1)) / 2^n), i.e. Round2 with ties toward -inf.  Implemented
// without relying on implementation-defined right shift of negatives.
AV2ENC_ME_HD std::int32_t me2_round2_floor(std::int32_t x, int n) {
  if (n <= 0) {
    return x;
  }
  const std::int32_t half = static_cast<std::int32_t>(1) << (n - 1);
  const std::int32_t v = x + half;
  const std::int32_t d = static_cast<std::int32_t>(1) << n;
  return v >= 0 ? v / d : -((-v + d - 1) / d);
}

// Spec Round2Signed: round half away from zero.
AV2ENC_ME_HD std::int32_t me2_round2_signed(std::int32_t x, int n) {
  if (n <= 0) {
    return x;
  }
  const std::int32_t a = x < 0 ? -x : x;
  const std::int32_t r = (a + (static_cast<std::int32_t>(1) << (n - 1))) >> n;
  return x < 0 ? -r : r;
}

// ME 1/16-sample component -> bitstream 1/8-sample component.  This is exactly
// mc::mv_from_me_1_16() / AV2's `(2 * mv)` relationship.
AV2ENC_ME_HD std::int32_t me2_1_16_to_eighth(std::int32_t mv_1_16) {
  return me2_round2_signed(mv_1_16, 1);
}

AV2ENC_ME_HD std::int32_t me2_eighth_to_1_16(std::int32_t mv_eighth) {
  return mv_eighth * 2;
}

// Normative lower_mv_precision() for one component (§5.20.7.13).  Input and
// output are 1/8-sample units.  Ties round down (the a-1 trick), and a value
// that is not representable at `p` is clipped to MV_LOW/UPP +/- radix.
AV2ENC_ME_HD std::int32_t me2_lower_mv_precision_component(
    std::int32_t mv_eighth, int precision) {
  if (!me2_precision_valid(precision) ||
      precision == kMvPrecisionEighthPel) {
    return mv_eighth;
  }
  const int bits = me2_precision_shift(precision);
  const std::int32_t radix = static_cast<std::int32_t>(1) << bits;
  const std::int32_t a = mv_eighth < 0 ? -mv_eighth : mv_eighth;
  const std::int32_t a_int = me2_round2_floor(a - 1, bits);
  std::int32_t out = a_int << bits;
  if (mv_eighth < 0) {
    out = -out;
  }
  if ((a_int << bits) != a) {
    out = me::me_iclamp(kMvLow + radix, out, kMvUpp - radix);
  }
  return out;
}

AV2ENC_ME_HD void me2_lower_mv_precision(std::int32_t* mv_y_eighth,
                                         std::int32_t* mv_x_eighth,
                                         int precision) {
  *mv_y_eighth =
      me2_lower_mv_precision_component(*mv_y_eighth, precision);
  *mv_x_eighth =
      me2_lower_mv_precision_component(*mv_x_eighth, precision);
}

// Quantize a 1/16-sample ME MV to the signaled precision, returned in 1/16
// units.  The eighth-pel step uses AV2's 1/16 -> 1/8 conversion (ties away
// from zero); coarser steps then apply the normative lower_mv_precision()
// rule in the 1/8 domain.
AV2ENC_ME_HD std::int32_t me2_quantize_mv_component_1_16(
    std::int32_t mv_1_16, int precision) {
  const std::int32_t e = me2_1_16_to_eighth(mv_1_16);
  const std::int32_t q = me2_lower_mv_precision_component(e, precision);
  return me2_eighth_to_1_16(q);
}

AV2ENC_ME_HD bool me2_mv_is_at_precision(std::int32_t mv_1_16,
                                         int precision) {
  const std::int32_t q = me2_precision_quantum_1_16(precision);
  return (mv_1_16 % q) == 0;
}

// --- frame header (§5.18.2) -------------------------------------------------
// Mirrors:
//   if ( force_integer_mv ) {
//       FrameMvPrecision = MV_PRECISION_ONE_PEL; UsePerBlockMvPrecision = 0;
//   } else {
//       read use_qtr_precision_mv;
//       if ( use_qtr_precision_mv ) FrameMvPrecision = QUARTER;
//       else { read allow_high_precision_mv;
//              FrameMvPrecision = allow_high ? EIGHTH : HALF; }
//       UsePerBlockMvPrecision = enable_flex_mvres;
//   }
struct FramePrecisionSyntax {
  bool valid = false;
  int frame_precision = kMvPrecisionEighthPel;
  int use_qtr_precision_mv = 0;     // coded bit (absent when force_integer_mv)
  int allow_high_precision_mv = 0;  // coded only when !use_qtr && !force
  int coded_bits = 0;               // 0, 1 or 2 bits among the two flags
  int use_per_block_mv_precision = 0;
};

// Decoder-side derivation from the two header bits.
AV2ENC_ME_HD FramePrecisionSyntax me2_frame_precision_from_bits(
    int force_integer_mv, int use_qtr_precision_mv,
    int allow_high_precision_mv, int enable_flex_mvres) {
  FramePrecisionSyntax s;
  s.valid = true;
  if (force_integer_mv != 0) {
    s.frame_precision = kMvPrecisionOnePel;
    s.use_per_block_mv_precision = 0;
    s.coded_bits = 0;
    return s;
  }
  s.use_qtr_precision_mv = use_qtr_precision_mv != 0 ? 1 : 0;
  if (s.use_qtr_precision_mv != 0) {
    s.frame_precision = kMvPrecisionQuarterPel;
    s.coded_bits = 1;
  } else {
    s.allow_high_precision_mv = allow_high_precision_mv != 0 ? 1 : 0;
    s.frame_precision = s.allow_high_precision_mv != 0
                            ? kMvPrecisionEighthPel
                            : kMvPrecisionHalfPel;
    s.coded_bits = 2;
  }
  s.use_per_block_mv_precision = enable_flex_mvres != 0 ? 1 : 0;
  return s;
}

// Encoder-side inverse: the exact header bits for a target frame precision.
// ONE_PEL is only reachable through force_integer_mv (screen-content tools),
// which also disables per-block precision (UsePerBlockMvPrecision = 0).
AV2ENC_ME_HD bool me2_frame_precision_to_bits(int target_precision,
                                               int enable_flex_mvres,
                                               int force_integer_mv,
                                               FramePrecisionSyntax* out) {
  if (out == nullptr || !me2_precision_valid(target_precision)) {
    return false;
  }
  FramePrecisionSyntax s;
  s.valid = true;
  s.frame_precision = target_precision;
  if (target_precision == kMvPrecisionOnePel) {
    if (force_integer_mv == 0) {
      return false;  // not signallable in a non-screen-content frame
    }
    s.coded_bits = 0;
    s.use_per_block_mv_precision = 0;
    *out = s;
    return true;
  }
  if (force_integer_mv != 0) {
    return false;  // force_integer_mv pins FrameMvPrecision to ONE_PEL
  }
  if (target_precision == kMvPrecisionQuarterPel) {
    s.use_qtr_precision_mv = 1;
    s.coded_bits = 1;
  } else if (target_precision == kMvPrecisionEighthPel ||
             target_precision == kMvPrecisionHalfPel) {
    s.use_qtr_precision_mv = 0;
    s.allow_high_precision_mv =
        target_precision == kMvPrecisionEighthPel ? 1 : 0;
    s.coded_bits = 2;
  } else {
    return false;  // HALF/QUARTER/EIGHTH/ONE only
  }
  s.use_per_block_mv_precision = enable_flex_mvres != 0 ? 1 : 0;
  *out = s;
  return true;
}

// --- per-block pb_mv_precision (§5.20.7.11 / §5.20.7.13) --------------------
// Forward derivation used by the decoder:
//   adjusted = Max(MV_PRECISION_ONE_PEL, FrameMvPrecision - 2) - pb_mv_precision
//   MvPrecision = adjusted <= MV_PRECISION_TWO_PEL ? adjusted - 1 : adjusted
AV2ENC_ME_HD int me2_pb_precision_derive(int frame_precision,
                                         int pb_mv_precision) {
  if (!me2_precision_valid(frame_precision) || pb_mv_precision < 0) {
    return -1;
  }
  const int base = me::me_imax(static_cast<int>(kMvPrecisionOnePel),
                               frame_precision - 2);
  const int adjusted = base - pb_mv_precision;
  return adjusted <= kMvPrecisionTwoPel ? adjusted - 1 : adjusted;
}

// Encoder-side inverse.  Returns false when `target` cannot be signalled with
// this frame precision (finer than the frame precision, or a two-pel gap).
AV2ENC_ME_HD bool me2_block_precision_to_syntax(int frame_precision,
                                                int target_precision,
                                                int* use_most_probable,
                                                int* pb_mv_precision) {
  if (use_most_probable == nullptr || pb_mv_precision == nullptr ||
      !me2_precision_valid(frame_precision) ||
      !me2_precision_valid(target_precision)) {
    return false;
  }
  if (target_precision == frame_precision) {
    *use_most_probable = 1;
    *pb_mv_precision = -1;
    return true;
  }
  if (target_precision > frame_precision) {
    return false;  // per-block precision can only get coarser
  }
  for (int pb = 0; pb <= 7; ++pb) {
    if (me2_pb_precision_derive(frame_precision, pb) == target_precision) {
      *use_most_probable = 0;
      *pb_mv_precision = pb;
      return true;
    }
  }
  return false;
}

// True when `target` is signallable for a NEWMV block under this frame
// precision and flex setting.
AV2ENC_ME_HD bool me2_precision_reachable(int frame_precision, int target,
                                          int enable_flex_mvres) {
  if (!me2_precision_valid(frame_precision) ||
      !me2_precision_valid(target)) {
    return false;
  }
  if (target == frame_precision) {
    return true;
  }
  if (enable_flex_mvres == 0) {
    return false;
  }
  // UsePerBlockMvPrecision is forced to 0 when force_integer_mv pins the
  // frame to ONE_PEL; frame precision below HALF is also never produced by
  // the non-screen-content header path.
  if (frame_precision < kMvPrecisionHalfPel) {
    return false;
  }
  int um = 0;
  int pb = -1;
  return me2_block_precision_to_syntax(frame_precision, target, &um, &pb);
}

// Fill `out` with the signallable precisions in a fixed coarse-to-fine order.
// `mask` is a bitmask indexed by MvPrecision; 0 means "all reachable".
// The frame precision is always present, so a valid selection exists.
AV2ENC_ME_HD int me2_precision_candidates(int frame_precision,
                                          int enable_flex_mvres, int mask,
                                          int* out, int max_out) {
  if (out == nullptr || max_out <= 0 ||
      !me2_precision_valid(frame_precision)) {
    return 0;
  }
  int n = 0;
  for (int p = kMvPrecisionEightPel; p <= kMvPrecisionEighthPel; ++p) {
    if (n >= max_out) {
      break;
    }
    const bool reachable =
        me2_precision_reachable(frame_precision, p, enable_flex_mvres);
    const bool allowed = (mask == 0) || ((mask & (1 << p)) != 0);
    if (p == frame_precision || (reachable && allowed)) {
      out[n++] = p;
    }
  }
  if (n == 0) {
    out[0] = frame_precision;
    n = 1;
  }
  return n;
}

// Entropy-context helpers (§9.3 / §5.20.7.13).  `n` neighbour entries.
AV2ENC_ME_HD int me2_ctx_use_most_probable(const int* neighbour_use_most,
                                           int n) {
  int ctx = 0;
  for (int i = 0; i < n; ++i) {
    if (neighbour_use_most != nullptr && neighbour_use_most[i] != 0) {
      ++ctx;
    }
  }
  return ctx;
}

AV2ENC_ME_HD int me2_ctx_pb_precision(const int* neighbour_precisions, int n,
                                      int frame_precision) {
  int ctx = 0;
  for (int i = 0; i < n; ++i) {
    if (neighbour_precisions != nullptr &&
        neighbour_precisions[i] < frame_precision) {
      ctx = 1;
    }
  }
  return ctx;
}

// CDF row index for TilePbMvPrecisionCdf[ctx][row].
AV2ENC_ME_HD int me2_pb_precision_cdf_row(int frame_precision) {
  return frame_precision - kMvPrecisionHalfPel;
}

// --- rate proxy -------------------------------------------------------------
AV2ENC_ME_HD int me2_floor_log2(std::int32_t v) {
  int n = 0;
  while (v > 1) {
    v >>= 1;
    ++n;
  }
  return n;
}

// Count of shell-code symbols for one MVD at `precision`.  Inputs are 1/8
// sample units; each S()/L(1) decision counts as one unit.  This is not an
// entropy coder and deliberately excludes CDF skew: it is a deterministic
// rate proxy shared by CPU and GPU.  Returns -1 if the MVD is not
// representable at `precision` or exceeds MV_* bounds.
AV2ENC_ME_HD int me2_mvd_shell_bins(std::int32_t dy_eighth,
                                    std::int32_t dx_eighth, int precision) {
  if (!me2_precision_valid(precision)) {
    return -1;
  }
  const int shift = me2_precision_shift(precision);
  const std::int32_t radix = static_cast<std::int32_t>(1) << shift;
  const bool neg_y = dy_eighth < 0;
  const bool neg_x = dx_eighth < 0;
  const std::int32_t ay8 = neg_y ? -dy_eighth : dy_eighth;
  const std::int32_t ax8 = neg_x ? -dx_eighth : dx_eighth;
  if ((ay8 % radix) != 0 || (ax8 % radix) != 0) {
    return -1;
  }
  if (ay8 > 2 * static_cast<std::int32_t>(kMvUpp) ||
      ax8 > 2 * static_cast<std::int32_t>(kMvUpp)) {
    return -1;
  }
  const std::int32_t ay = ay8 / radix;
  const std::int32_t ax = ax8 / radix;
  const std::int32_t shell = ax + ay;
  const int cls = shell == 0 ? 0 : me2_floor_log2(shell);
  if (cls >= 17) {
    return -1;  // read_mv() rejects shells >= 2^17
  }
  const std::int32_t off = shell - (cls == 0 ? 0 : (static_cast<std::int32_t>(1) << cls));

  int bins = 1;  // shell_set
  bins += 1;     // shell_class (one 8-ary symbol)
  const bool shell_set = cls >= 8;
  if (shell_set && cls >= 15 && precision == kMvPrecisionEighthPel) {
    bins += 1;  // joint_shell_last_two_classes (classes 15 and 16)
  }

  if (cls < 2) {
    bins += 1;  // shell_offset_low_class
  } else if (cls == 2) {
    bins += off == 0 ? 1 : (off == 1 ? 2 : 3);
  } else {
    bins += cls;
  }

  if (shell > 0) {
    const std::int32_t max_pair = shell >> 1;
    bool skip_col_bit = false;
    std::int32_t col = 0;
    if (ax == max_pair && ((shell & 1) == 0)) {
      skip_col_bit = true;
      col = max_pair;
    } else if (ax <= max_pair) {
      col = ax;
    } else {
      col = ay;
    }

    if (max_pair > 0) {
      const int max_bits = me::me_imin(static_cast<int>(max_pair), 2);
      const std::int32_t prefix_col = me::me_imin(col, 2);
      for (int i = 0; i < max_bits; ++i) {
        ++bins;  // col_mv_greater
        if (prefix_col <= i) {
          break;
        }
      }
      if (max_pair > 2 && prefix_col == 2) {
        ++bins;  // col_remainder (NS)
      }
    }
    if (!skip_col_bit) {
      ++bins;  // col_mv_index
    }
  }

  if (dy_eighth != 0) {
    ++bins;  // sign
  }
  if (dx_eighth != 0) {
    ++bins;
  }
  return bins;
}

// Side-information bins for signalling the per-block precision itself.  These
// are emitted only for NEWMV when flex_mvres/UsePerBlockMvPrecision are on.
AV2ENC_ME_HD int me2_precision_side_bins(int frame_precision, int target,
                                         int enable_flex_mvres,
                                         bool has_newmv) {
  if (enable_flex_mvres == 0 || !has_newmv ||
      !me2_precision_valid(frame_precision) ||
      !me2_precision_valid(target)) {
    return 0;
  }
  // Frame precision below HALF only comes from force_integer_mv, which forces
  // UsePerBlockMvPrecision = 0, so neither precision symbol is coded.
  if (frame_precision < kMvPrecisionHalfPel) {
    return 0;
  }
  if (target == frame_precision) {
    return 1;  // use_most_probable_precision == 1
  }
  return 2;    // use_most_probable_precision == 0 + pb_mv_precision symbol
}

}  // namespace av2enc::me2

#endif  // AV2ENC_ENCODER_INTER_ME2_ME2_PRECISION_H_
