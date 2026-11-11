// §7.14.2 quantizer derivation and §7.14.4 forward/inverse quantization with
// DC/AC split, QM and TCQ.
#include "encoder/quant/dcdeltaq/dcdeltaq.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace av2enc::dcdeltaq {
namespace {

// Ac_Qlookup[25] from §7.14.2.
constexpr int kAcQlookup[25] = {
    64, 40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 54, 55,
    57, 59, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78};

void set_error(std::string* error, const char* what) {
  if (error != nullptr) *error = what;
}

int clip3(int lo, int hi, int value) {
  return value < lo ? lo : (value > hi ? hi : value);
}

int tx_class_2d(int tx_type) {
  // §5.20.7.29 get_tx_class(): V_* and H_* are the 1D classes; every other
  // type (including IDTX) is TX_CLASS_2D.
  return tx_type >= 10 ? 0 : 1;
}

int extra_tcq_shift(const QuantBlock& block) {
  return (block.allow_tcq && block.plane == 0 && !block.lossless &&
          tx_class_2d(block.tx_type) != 0 && !block.use_fsc)
             ? 1
             : 0;
}

}  // namespace

int qlookup(int q) {
  if (q < 25) return kAcQlookup[q < 0 ? 0 : q];
  return kAcQlookup[((q - 1) % 24) + 1] << ((q - 1) / 24);
}

int get_q(int qindex, int delta, int max_q) {
  if (qindex == 0 && delta <= 0) return kAcQlookup[0];
  const int clamped = clip3(1, max_q, qindex + delta);
  return qlookup(clamped);
}

ResolvedDeltaQ resolve_delta_q(const SequenceDeltaQ& seq_in,
                               const FrameDeltaQ& frame) {
  const SequenceDeltaQ seq = normalized_sequence_delta_q(seq_in);
  ResolvedDeltaQ out;
  out.y_dc = frame.delta_q_y_dc + seq.base_y_dc_delta_q;
  out.u_dc = frame.delta_q_u_dc + seq.base_uv_dc_delta_q;
  out.v_dc = frame.delta_q_v_dc + seq.base_uv_dc_delta_q;
  out.u_ac = frame.delta_q_u_ac + seq.base_uv_ac_delta_q;
  out.v_ac = frame.delta_q_v_ac + seq.base_uv_ac_delta_q;
  return out;
}

int get_qindex(const QuantFrameInputs& in, bool ignore_delta_q,
               int segment_id) {
  if (segment_id < 0 || segment_id >= in.num_segments) return in.base_q_idx;
  if (in.alt_q_enabled[segment_id]) {
    int qindex = in.base_q_idx + in.alt_q[segment_id];
    if (!ignore_delta_q && in.delta_q_present) {
      qindex = in.current_q_index + in.alt_q[segment_id];
    }
    return clip3(0, kMaxQ, qindex);
  }
  if (!ignore_delta_q && in.delta_q_present) return in.current_q_index;
  return in.base_q_idx;
}

QuantFrameState derive_quant(const QuantFrameInputs& in, int segment_id) {
  QuantFrameState out;
  out.delta = resolve_delta_q(in.seq, in.frame);
  out.qindex = get_qindex(in, false, segment_id);
  out.plane[0].dc = get_q(out.qindex, out.delta.y_dc);
  out.plane[0].ac = get_q(out.qindex, 0);
  out.plane[1].dc = get_q(out.qindex, out.delta.u_dc);
  out.plane[1].ac = get_q(out.qindex, out.delta.u_ac);
  out.plane[2].dc = get_q(out.qindex, out.delta.v_dc);
  out.plane[2].ac = get_q(out.qindex, out.delta.v_ac);

  const int ignore_qindex = get_qindex(in, true, segment_id);
  out.lossless = ignore_qindex == 0 && !in.delta_q_present &&
                 out.delta.y_dc <= 0 && out.delta.u_dc <= 0 &&
                 out.delta.v_dc <= 0 && out.delta.u_ac <= 0 &&
                 out.delta.v_ac <= 0;
  return out;
}

LosslessFlags derive_lossless_flags(const QuantFrameInputs& in) {
  LosslessFlags out;
  out.num_segments = std::max(1, std::min(in.num_segments, kMaxSegments));
  out.coded_lossless = true;
  for (int s = 0; s < out.num_segments; ++s) {
    const QuantFrameState q = derive_quant(in, s);
    out.lossless[s] = q.lossless;
    if (q.lossless) {
      out.has_lossless_segment = true;
    } else {
      out.coded_lossless = false;
    }
  }
  return out;
}

std::int32_t round2(std::int32_t value, int n) {
  if (n <= 0) return value;
  return static_cast<std::int32_t>(
      (static_cast<std::int64_t>(value) + (1LL << (n - 1))) >> n);
}

int adjusted_width(int coded_w) {
  return std::min(kMaxAdjustedDim, std::max(1, coded_w));
}

int adjusted_height(int coded_h) {
  return std::min(kMaxAdjustedDim, std::max(1, coded_h));
}

int dequant_shift(const QuantBlock& block) {
  const int pels = block.coded_w * block.coded_h;
  return (pels > 256 ? 1 : 0) + (pels > 1024 ? 1 : 0) +
         extra_tcq_shift(block);
}

bool build_q2_matrix(const QuantBlock& block, std::int32_t* q2,
                     std::string* error) {
  if (q2 == nullptr) {
    set_error(error, "build_q2_matrix: null q2");
    return false;
  }
  if (block.plane < 0 || block.plane > 2) {
    set_error(error, "build_q2_matrix: bad plane");
    return false;
  }
  if (block.coded_w <= 0 || block.coded_h <= 0) {
    set_error(error, "build_q2_matrix: bad coded size");
    return false;
  }
  const int tw = adjusted_width(block.coded_w);
  const int th = adjusted_height(block.coded_h);

  // §7.14.4 useQm = using_qmatrix == 1 && PlaneTxType < IDTX && segLvl < 15.
  bool use_qm = block.using_qmatrix && block.tx_type < 9 &&
                block.qm_state != nullptr;
  int seg_level = qm::kNoQmLevel;
  if (use_qm) {
    if (!qm::valid_tx_size(block.tx_size)) {
      set_error(error, "build_q2_matrix: QM requires a valid tx_size");
      return false;
    }
    seg_level =
        qm::resolve_seg_level(*block.qm_state, block.plane, block.tx_size,
                              block.segment_id);
    use_qm = seg_level >= 0 && seg_level < qm::kNumQmLevels;
  }

  for (int i = 0; i < th; ++i) {
    for (int j = 0; j < tw; ++j) {
      const int q = (i == 0 && j == 0) ? block.q.dc : block.q.ac;
      std::int32_t value = q;
      if (use_qm) {
        const int m = qm::matrix_value(*block.qm_state, block.plane,
                                       block.tx_size, seg_level, i, j);
        value = round2(q * m, 5);
      }
      q2[i * tw + j] = value;
    }
  }
  return true;
}

bool quantize_block(const double* coeff, const QuantBlock& block,
                    std::int32_t* levels, std::string* error) {
  if (coeff == nullptr || levels == nullptr) {
    set_error(error, "quantize_block: null buffer");
    return false;
  }
  const int tw = adjusted_width(block.coded_w);
  const int th = adjusted_height(block.coded_h);
  const int n = tw * th;
  if (block.lossless) {
    for (int i = 0; i < n; ++i) {
      levels[i] = static_cast<std::int32_t>(std::llround(coeff[i]));
    }
    return true;
  }
  std::int32_t q2[kMaxQuantElems];
  if (!build_q2_matrix(block, q2, error)) return false;
  // The decoder divides by 2^shift (dqDenom); the forward quantizer therefore
  // multiplies the target level by the same factor.  `coefficient_scale` is
  // the forward transform's coefficient per unit residual at q2 = 528.
  const double factor =
      block.coefficient_scale * static_cast<double>(528) *
      static_cast<double>(1 << extra_tcq_shift(block));
  for (int i = 0; i < n; ++i) {
    const double denom = static_cast<double>(q2[i]);
    if (denom <= 0.0) {
      set_error(error, "quantize_block: non-positive q2");
      return false;
    }
    levels[i] = static_cast<std::int32_t>(
        std::llround(coeff[i] * factor / denom));
  }
  return true;
}

bool dequantize_block(const std::int32_t* levels, const QuantBlock& block,
                      std::int32_t* dequant, std::string* error) {
  if (levels == nullptr || dequant == nullptr) {
    set_error(error, "dequantize_block: null buffer");
    return false;
  }
  const int tw = adjusted_width(block.coded_w);
  const int th = adjusted_height(block.coded_h);
  const int n = tw * th;
  if (block.lossless) {
    for (int i = 0; i < n; ++i) dequant[i] = levels[i];
    return true;
  }
  std::int32_t q2[kMaxQuantElems];
  if (!build_q2_matrix(block, q2, error)) return false;
  const int shift = dequant_shift(block);
  const std::int32_t denom = static_cast<std::int32_t>(1 << shift);
  const std::int32_t lo = -(1 << 15);
  const std::int32_t hi = (1 << 15) - 1;
  for (int i = 0; i < n; ++i) {
    const std::int32_t level = levels[i];
    const std::int64_t abs_level =
        level < 0 ? -static_cast<std::int64_t>(level) : level;
    const std::int64_t high = (abs_level * q2[i]) & 0xFFFFFF;
    const std::int32_t dq = round2(static_cast<std::int32_t>(high),
                                   kQuantTableBits) /
                            denom;
    const std::int32_t signed_dq = level < 0 ? -dq : dq;
    dequant[i] = clip3(lo, hi, signed_dq);
  }
  return true;
}

int tcq_state_q0(int state) { return (state >> 1) & 1; }

int tcq_syntax_level(int full_level, int pre_max) {
  if (pre_max < 2) return full_level;
  const int sign = full_level < 0 ? -1 : 1;
  const int magnitude = full_level < 0 ? -full_level : full_level;
  if (magnitude < pre_max - 1) return sign * magnitude;
  const int level =
      ((magnitude ^ pre_max) & 1) != 0 ? pre_max - 1 : pre_max;
  return sign * level;
}

int tcq_full_level(int syntax_level, int state) {
  const int sign = syntax_level < 0 ? -1 : 1;
  const int magnitude = syntax_level < 0 ? -syntax_level : syntax_level;
  if (magnitude == 0) return 0;
  return sign * (2 * magnitude - tcq_state_q0(state));
}

}  // namespace av2enc::dcdeltaq
