// AV2 intra bi-prediction (IBP) — CPU reference implementation.
// See ibp.h and README.md for the normative contract.

#include "codec/intra/ibp/ibp.h"

#include "codec/intra/ibp/ibp_tables.h"

#include <algorithm>
#include <cstdlib>

namespace av2enc::ibp {
namespace {

// ---------------------------------------------------------------------------
// Normative tables (single source: ibp_tables.h)
// ---------------------------------------------------------------------------

constexpr int kDrIntraDerivativeCount = 90;
constexpr int kDrIntraDerivative[kDrIntraDerivativeCount] = {
    AV2ENC_IBP_DR_INTRA_DERIVATIVE_TABLE};

constexpr int kDrInterpFilter[32][4] = {AV2ENC_IBP_DR_INTERP_FILTER_TABLE};

constexpr int kDivLutBits = 7;       // DIV_LUT_BITS
constexpr int kDivLutPrecBits = 9;   // DIV_LUT_PREC_BITS
constexpr int kDivLutNum = 129;      // DIV_LUT_NUM
constexpr int kDivLut[kDivLutNum] = {AV2ENC_IBP_DIV_LUT_TABLE};

constexpr int kIbpWeights[kNumIbpWeightRows][kWeightSize] = {
    AV2ENC_IBP_WEIGHTS_TABLE};

constexpr int kIiWeights1d[kIiWeights1dSize] = {
    AV2ENC_IBP_II_WEIGHTS_1D_TABLE};

static_assert(sizeof(kIbpWeights) / sizeof(kIbpWeights[0]) ==
              kNumIbpWeightRows);

// ---------------------------------------------------------------------------
// Scalar arithmetic
// ---------------------------------------------------------------------------

int round2(int value, int shift) noexcept {
  if (shift <= 0) {
    return value;
  }
  return (value + (1 << (shift - 1))) >> shift;
}

int floor_log2(int value) noexcept {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

int clip3(int low, int high, int value) noexcept {
  return value < low ? low : (value > high ? high : value);
}

int clip1(int value, int bit_depth) noexcept {
  return clip3(0, (1 << bit_depth) - 1, value);
}

bool is_power_of_two(int value) noexcept {
  return value > 0 && (value & (value - 1)) == 0;
}

bool valid_block_size(int w, int h) noexcept {
  return w >= intra::kMinBlockDim && w <= intra::kMaxBlockDim &&
         h >= intra::kMinBlockDim && h <= intra::kMaxBlockDim &&
         is_power_of_two(w) && is_power_of_two(h);
}

// §7.13.3.22 resolve_divisor.  `shift` includes DIV_LUT_PREC_BITS.
void resolve_divisor(int divisor, int& shift, int& factor) noexcept {
  const int magnitude = std::abs(divisor);
  const int n = floor_log2(magnitude);
  const int e = magnitude - (1 << n);
  const int f =
      n > kDivLutBits ? round2(e, n - kDivLutBits) : (e << (kDivLutBits - n));
  shift = n + kDivLutPrecBits;
  factor = kDivLut[f] * (divisor < 0 ? -1 : 1);
}

// §7.13.2.14 filter_corner process.
void filter_corner(intra::EdgeRow& above, intra::EdgeRow& left) noexcept {
  const int sum = left[0] * 5 + above[-1] * 6 + above[0] * 5;
  const int value = round2(sum, 4);
  left[-1] = static_cast<std::int16_t>(value);
  above[-1] = static_cast<std::int16_t>(value);
}

// ---------------------------------------------------------------------------
// §7.13.2.8 single directional prediction
// ---------------------------------------------------------------------------
//
// The edges are mutable copies: §7.13.2.8 pads them before reading.  The
// padding is idempotent for the two angles of an IBP pair (both lie outside
// 90..180, so they take the same else branch), but the function is written as
// the spec writes it so an independent call sequence stays faithful.
void single_directional(const IntraEdges& edges, int p_angle,
                        intra::EdgeRow& above, intra::EdgeRow& left,
                        std::uint8_t* dst, std::size_t dst_stride) noexcept {
  const int w = edges.w;
  const int h = edges.h;
  const int enable_idif = edges.is_luma ? 1 : 0;

  constexpr int kMinBase = -1;  // MrlIndex = 0
  const int max_base = w + h - 1;
  if (p_angle > 90 && p_angle < 180) {
    left[h] = left[h - 1];
    above[w] = above[w - 1];
    left[h + 1] = left[h - 1];
    above[w + 1] = above[w - 1];
  } else {
    left[max_base + 1] = left[max_base];
    above[max_base + 1] = above[max_base];
    left[max_base + 2] = left[max_base];
    above[max_base + 2] = above[max_base];
  }
  left[kMinBase - 1] = left[kMinBase];
  above[kMinBase - 1] = above[kMinBase];

  if (p_angle < 90) {
    const int dx = kDrIntraDerivative[p_angle];
    const int max_base_x = w + h - 1;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const int idx = (i + 1) * dx;
        const int base = (idx >> 6) + j;
        const int shift = (idx >> 1) & 0x1F;
        int value = 0;
        if (base < max_base_x + enable_idif) {
          if (enable_idif != 0) {
            const int* taps = kDrInterpFilter[shift];
            int sum = 0;
            for (int t = 0; t < 4; ++t) {
              sum += taps[t] * above[base + t - 1];
            }
            value = clip1(round2(sum, 7), edges.bit_depth);
          } else {
            value = round2(above[base] * (32 - shift) +
                               above[base + 1] * shift,
                           5);
          }
        } else {
          value = above[max_base_x];
        }
        dst[static_cast<std::size_t>(i) * dst_stride +
            static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
      }
    }
    return;
  }

  if (p_angle > 90 && p_angle < 180) {
    const int dx = kDrIntraDerivative[180 - p_angle];
    const int dy = kDrIntraDerivative[p_angle - 90];
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        int idx = (j << 6) - (i + 1) * dx;
        int base = idx >> 6;
        int value = 0;
        if (base >= kMinBase) {
          const int shift = (idx >> 1) & 0x1F;
          if (enable_idif != 0) {
            const int* taps = kDrInterpFilter[shift];
            int sum = 0;
            for (int t = 0; t < 4; ++t) {
              sum += taps[t] * above[base + t - 1];
            }
            value = clip1(round2(sum, 7), edges.bit_depth);
          } else {
            value = round2(above[base] * (32 - shift) +
                               above[base + 1] * shift,
                           5);
          }
        } else {
          idx = (i << 6) - (j + 1) * dy;
          base = idx >> 6;
          const int shift = (idx >> 1) & 0x1F;
          if (enable_idif != 0) {
            const int* taps = kDrInterpFilter[shift];
            int sum = 0;
            for (int t = 0; t < 4; ++t) {
              sum += taps[t] * left[base + t - 1];
            }
            value = clip1(round2(sum, 7), edges.bit_depth);
          } else {
            value = round2(left[base] * (32 - shift) +
                               left[base + 1] * shift,
                           5);
          }
        }
        dst[static_cast<std::size_t>(i) * dst_stride +
            static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
      }
    }
    return;
  }

  if (p_angle > 180) {
    const int dy = kDrIntraDerivative[270 - p_angle];
    const int max_base_y = w + h - 1;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        const int idx = (j + 1) * dy;
        const int base = (idx >> 6) + i;
        const int shift = (idx >> 1) & 0x1F;
        int value = 0;
        if (base < max_base_y + enable_idif) {
          if (enable_idif != 0) {
            const int* taps = kDrInterpFilter[shift];
            int sum = 0;
            for (int t = 0; t < 4; ++t) {
              sum += taps[t] * left[base + t - 1];
            }
            value = clip1(round2(sum, 7), edges.bit_depth);
          } else {
            value = round2(left[base] * (32 - shift) +
                               left[base + 1] * shift,
                           5);
          }
        } else {
          value = left[max_base_y];
        }
        dst[static_cast<std::size_t>(i) * dst_stride +
            static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
      }
    }
    return;
  }

  // p_angle == 90 or 180.
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int value = p_angle == 90 ? above[j] : left[i];
      dst[static_cast<std::size_t>(i) * dst_stride +
          static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(value);
    }
  }
}

// §7.13.2.7 reference-sample preparation when IBP is enabled (or when the
// ordinary path is used; the two differ only in the branches below).
void filter_reference_samples(const IntraEdges& edges,
                              const PredictParams& params, int p_angle,
                              bool apply_ibp, intra::EdgeRow& above,
                              intra::EdgeRow& left) noexcept {
  if (!params.enable_edge_filter || p_angle == 90 || p_angle == 180) {
    return;
  }

  int filter_type_above = 0;
  int filter_type_left = 0;
  int angle_above = p_angle - 90;
  int angle_left = p_angle - 180;
  int need_right = p_angle < 90 ? 1 : 0;
  int need_bottom = p_angle > 180 ? 1 : 0;

  if (edges.have_above && params.above_neighbor_is_smooth) {
    filter_type_above = 1;
  }
  if (edges.have_left && params.left_neighbor_is_smooth) {
    filter_type_left = 1;
  }

  if (apply_ibp) {
    if (p_angle > 180) {
      need_right = 1;
    }
    if (p_angle < 90) {
      need_bottom = 1;
    }
    if (angle_above > 90) {
      angle_above -= 180;
    }
    if (angle_left < -90) {
      angle_left += 180;
    }
  } else {
    const int filter_type = filter_type_above | filter_type_left;
    filter_type_above = filter_type;
    filter_type_left = filter_type;
  }

  if ((apply_ibp || (p_angle > 90 && p_angle < 180)) &&
      (edges.w + edges.h) >= 24) {
    filter_corner(above, left);
  }

  if (edges.have_above) {
    const int strength = intra::edge_filter_strength(
        edges.w, edges.h, filter_type_above != 0, angle_above);
    const int num_px = std::min(edges.w, edges.valid_above_px) +
                       (need_right != 0 ? edges.h : 0) + 1;
    intra::intra_edge_filter(above, num_px, strength);
  }
  if (edges.have_left) {
    const int strength = intra::edge_filter_strength(
        edges.w, edges.h, filter_type_left != 0, angle_left);
    const int num_px = std::min(edges.h, edges.valid_left_px) +
                       (need_bottom != 0 ? edges.w : 0) + 1;
    intra::intra_edge_filter(left, num_px, strength);
  }
}

// §7.13.2.12 IBP DC process.  Modifies a DC prediction in place.
void apply_ibp_dc(const IntraEdges& edges, std::uint8_t* dst,
                  std::size_t dst_stride) noexcept {
  const int w = edges.w;
  const int h = edges.h;
  const int log2_w = floor_log2(w);
  const int log2_h = floor_log2(h);

  if (edges.have_above) {
    const int c_begin = (w < h && edges.have_left) ? (w >> 2) : 0;
    for (int r = 0; r < (h >> 2); ++r) {
      const int s = ibp_dc_weight(log2_h - 2, r);
      for (int c = c_begin; c < w; ++c) {
        const int old = dst[static_cast<std::size_t>(r) * dst_stride +
                            static_cast<std::size_t>(c)];
        const int value = round2(edges.above[c] * (kWeightMax - s) +
                                     old * s,
                                 kWeightShift);
        dst[static_cast<std::size_t>(r) * dst_stride +
            static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(value);
      }
    }
  }

  if (edges.have_left) {
    const int r_begin = (w >= h && edges.have_above) ? (h >> 2) : 0;
    for (int r = r_begin; r < h; ++r) {
      for (int c = 0; c < (w >> 2); ++c) {
        const int s = ibp_dc_weight(log2_w - 2, c);
        const int old = dst[static_cast<std::size_t>(r) * dst_stride +
                            static_cast<std::size_t>(c)];
        const int value = round2(edges.left[r] * (kWeightMax - s) +
                                     old * s,
                                 kWeightShift);
        dst[static_cast<std::size_t>(r) * dst_stride +
            static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(value);
      }
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// §7.13.2.9 weights
// ---------------------------------------------------------------------------

WeightMatrix ibp_weights(int p_angle) noexcept {
  WeightMatrix result;
  const int angle = std::min(89, std::max(39, p_angle));
  const int dy = kDrIntraDerivative[90 - angle];
  for (int r = 0; r < kWeightSize; ++r) {
    int y = dy;
    for (int c = 0; c < kWeightSize; ++c) {
      const int dist = ((r + 1) << 6) + y;
      int shift = 0;
      int factor = 0;
      resolve_divisor(dist, shift, factor);
      shift -= kDivLutBits;
      result.at(r, c) = round2(y * factor, shift);
      y += dy;
    }
  }
  return result;
}

int ibp_dc_weight(int log2_extent, int index) noexcept {
  if (log2_extent < 0 || log2_extent >= kNumIbpWeightRows || index < 0 ||
      index >= kWeightSize) {
    return 0;
  }
  return kIbpWeights[log2_extent][index];
}

// ---------------------------------------------------------------------------
// §7.13.3.29 inter-intra mask (not IBP)
// ---------------------------------------------------------------------------

int ii_weights_1d(int index) noexcept {
  if (index < 0 || index >= kIiWeights1dSize) {
    return 0;
  }
  return kIiWeights1d[index];
}

bool build_inter_intra_mask(int w, int h, InterIntraMode mode,
                            std::span<int> mask) noexcept {
  if (w < intra::kMinBlockDim || w > intra::kMaxBlockDim ||
      h < intra::kMinBlockDim || h > intra::kMaxBlockDim ||
      mask.size() < static_cast<std::size_t>(w) * static_cast<std::size_t>(h)) {
    return false;
  }
  const int scale = 128 / std::max(h, w);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      int value = 32;
      if (mode == InterIntraMode::kV) {
        value = kIiWeights1d[i * scale];
      } else if (mode == InterIntraMode::kH) {
        value = kIiWeights1d[j * scale];
      } else if (mode == InterIntraMode::kSmooth) {
        value = kIiWeights1d[std::min(i, j) * scale];
      }
      mask[static_cast<std::size_t>(i) * static_cast<std::size_t>(w) +
           static_cast<std::size_t>(j)] = value;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Decision API
// ---------------------------------------------------------------------------

Decision decide(const DecisionParams& params) noexcept {
  Decision result;
  if (params.w < intra::kMinBlockDim || params.w > intra::kMaxBlockDim ||
      params.h < intra::kMinBlockDim || params.h > intra::kMaxBlockDim) {
    return result;
  }
  if (!is_power_of_two(params.w) || !is_power_of_two(params.h)) {
    return result;
  }
  if (static_cast<unsigned>(params.mode) >=
      static_cast<unsigned>(intra::kNumModes)) {
    return result;
  }
  if (params.plane < 0 || params.mrl_index != 0) {
    // This component implements MrlIndex = 0 only.  MRL changes pAngle and
    // the edge-filter extent, so silently ignoring it would be wrong.
    return result;
  }
  if (params.angle_delta < -intra::kMaxAngleDelta ||
      params.angle_delta > intra::kMaxAngleDelta) {
    return result;
  }
  if (!intra::is_directional(params.mode)) {
    if (params.angle_delta != 0) {
      return result;
    }
    result.valid = true;
    result.p_angle = 0;
    result.apply_ibp_edge_rules = false;
    result.use_ibp = false;
    return result;
  }

  int p_angle = 0;
  if (params.is_inter) {
    p_angle = intra::mode_to_angle(params.mode) +
              params.angle_delta * kAngleStep;
  } else {
    p_angle = intra::map_angle(params.mode, params.angle_delta, params.w,
                               params.h)
                  .p_angle;
  }
  if (p_angle <= 0 || p_angle >= 270) {
    return result;
  }

  const bool not4x4 = !(params.w == 4 && params.h == 4);
  const bool apply_ibp = params.enable_ibp && not4x4;
  const bool even_delta = (params.angle_delta & 1) == 0;

  result.valid = true;
  result.p_angle = p_angle;
  result.apply_ibp_edge_rules = apply_ibp;
  result.use_ibp = apply_ibp && even_delta && params.plane == 0 &&
                   (p_angle < 90 || p_angle > 180) &&
                   params.mrl_index == 0;
  if (result.use_ibp) {
    result.second_angle = p_angle < 90 ? p_angle + 180 : p_angle - 180;
  }
  return result;
}

bool ibp_dc_applicable(bool enable_ibp, bool use_dip, Mode mode, int w, int h,
                       int plane, bool uv_is_cfl) noexcept {
  return enable_ibp && !use_dip && mode == Mode::kDc &&
         !(w == 4 && h == 4) && (plane == 0 || !uv_is_cfl);
}

int build_ibp_candidates(std::span<ModeCandidate> out, int w, int h,
                         int plane, int mrl_index) noexcept {
  if (plane != 0 || mrl_index != 0) {
    return 0;
  }
  int count = 0;
  for (int m = static_cast<int>(Mode::kV);
       m <= static_cast<int>(Mode::kD67); ++m) {
    for (int delta = -2; delta <= 2; delta += 2) {
      DecisionParams params;
      params.enable_ibp = true;
      params.mode = static_cast<Mode>(m);
      params.angle_delta = delta;
      params.w = w;
      params.h = h;
      params.plane = plane;
      params.mrl_index = mrl_index;
      const Decision decision = decide(params);
      if (!decision.use_ibp) {
        continue;
      }
      if (out.size() < static_cast<std::size_t>(count + 1)) {
        return -1;
      }
      out[static_cast<std::size_t>(count)].mode = static_cast<Mode>(m);
      out[static_cast<std::size_t>(count)].angle_delta =
          static_cast<std::int8_t>(delta);
      ++count;
    }
  }
  return count;
}

// ---------------------------------------------------------------------------
// Prediction API
// ---------------------------------------------------------------------------

IntraStatus predict(const IntraEdges& edges, const PredictParams& params,
                    std::uint8_t* dst, std::size_t dst_stride,
                    PredictResult* result) noexcept {
  if (dst == nullptr || !valid_block_size(edges.w, edges.h)) {
    return IntraStatus::kInvalidArgument;
  }
  if (edges.bit_depth < 8 || edges.bit_depth > 12) {
    return IntraStatus::kInvalidArgument;
  }
  if (params.angle_delta < -intra::kMaxAngleDelta ||
      params.angle_delta > intra::kMaxAngleDelta) {
    return IntraStatus::kInvalidArgument;
  }
  if (!intra::is_directional(params.mode) && params.angle_delta != 0) {
    return IntraStatus::kInvalidArgument;
  }
  if (dst_stride < static_cast<std::size_t>(edges.w)) {
    return IntraStatus::kInvalidArgument;
  }

  DecisionParams dp;
  dp.enable_ibp = params.enable_ibp;
  dp.is_inter = params.is_inter;
  dp.mode = params.mode;
  dp.angle_delta = params.angle_delta;
  dp.w = edges.w;
  dp.h = edges.h;
  dp.plane = edges.is_luma ? 0 : 1;
  dp.mrl_index = 0;
  const Decision decision = decide(dp);
  if (!decision.valid) {
    return IntraStatus::kInvalidArgument;
  }

  PredictResult local;
  local.p_angle = decision.p_angle;
  local.second_angle = decision.second_angle;

  if (params.mode == Mode::kDc) {
    intra::PredictParams dc;
    dc.mode = Mode::kDc;
    dc.angle_delta = 0;
    dc.enable_edge_filter = false;
    const IntraStatus status =
        intra::predict_intra(edges, dc, dst, dst_stride);
    if (status != IntraStatus::kOk) {
      return status;
    }
    if (ibp_dc_applicable(params.enable_ibp, params.use_dip, params.mode,
                          edges.w, edges.h, edges.is_luma ? 0 : 1,
                          params.uv_is_cfl)) {
      apply_ibp_dc(edges, dst, dst_stride);
      local.dc_modified = true;
    }
    if (result != nullptr) {
      *result = local;
    }
    return IntraStatus::kOk;
  }

  if (!intra::is_directional(params.mode)) {
    intra::PredictParams plain;
    plain.mode = params.mode;
    plain.angle_delta = params.angle_delta;
    plain.enable_edge_filter = params.enable_edge_filter;
    plain.above_neighbor_is_smooth = params.above_neighbor_is_smooth;
    plain.left_neighbor_is_smooth = params.left_neighbor_is_smooth;
    const IntraStatus status =
        intra::predict_intra(edges, plain, dst, dst_stride);
    if (status == IntraStatus::kOk && result != nullptr) {
      *result = local;
    }
    return status;
  }

  // Directional prediction.  The spec mutates AboveRow/LeftCol, so work on
  // copies and leave the caller's edges reusable for RDO.
  intra::EdgeRow above = edges.above;
  intra::EdgeRow left = edges.left;
  filter_reference_samples(edges, params, decision.p_angle,
                           decision.apply_ibp_edge_rules, above, left);

  single_directional(edges, decision.p_angle, above, left, dst, dst_stride);

  if (decision.use_ibp) {
    // The §7.13.2.7 blend maps pAngle and secondAngle to the same weight
    // matrix; ibp_weights() receives the angle in 0..90.
    const int weight_angle =
        decision.p_angle < 90 ? decision.p_angle : 270 - decision.p_angle;
    const WeightMatrix weights = ibp_weights(weight_angle);

    std::array<std::uint8_t, intra::kMaxBlockDim * intra::kMaxBlockDim>
        second_pred{};
    single_directional(edges, decision.second_angle, above, left,
                       second_pred.data(),
                       static_cast<std::size_t>(edges.w));

    const int c_shift = edges.w >> (kWeightSizeLog2 + 1);
    const int r_shift = edges.h >> (kWeightSizeLog2 + 1);
    for (int r = 0; r < edges.h; ++r) {
      for (int c = 0; c < edges.w; ++c) {
        const int s = decision.p_angle < 90
                          ? weights.at(r >> r_shift, c >> c_shift)
                          : weights.at(c >> c_shift, r >> r_shift);
        const int first = dst[static_cast<std::size_t>(r) * dst_stride +
                              static_cast<std::size_t>(c)];
        const int second =
            second_pred[static_cast<std::size_t>(r) *
                           static_cast<std::size_t>(edges.w) +
                       static_cast<std::size_t>(c)];
        const int value =
            round2(first * s + second * (kWeightMax - s), kWeightShift);
        dst[static_cast<std::size_t>(r) * dst_stride +
            static_cast<std::size_t>(c)] = static_cast<std::uint8_t>(value);
      }
    }
    local.blended = true;
  }

  if (result != nullptr) {
    *result = local;
  }
  return IntraStatus::kOk;
}

IntraStatus predict_block(const NeighborRequest& request, int bit_depth,
                          const PredictParams& params, std::uint8_t* dst,
                          std::size_t dst_stride,
                          PredictResult* result) noexcept {
  IntraEdges edges;
  const IntraStatus status = intra::prepare_edges(request, bit_depth, edges);
  if (status != IntraStatus::kOk) {
    return status;
  }
  return predict(edges, params, dst, dst_stride, result);
}

// ---------------------------------------------------------------------------
// §5.4.5 sequence_intra_config syntax
// ---------------------------------------------------------------------------

void BitWriter::put_bit(bool value) {
  const int index = bit_position_ >> 3;
  if ((bit_position_ & 7) == 0) {
    bytes_.push_back(0);
  }
  if (value) {
    bytes_[static_cast<std::size_t>(index)] |=
        static_cast<std::uint8_t>(0x80U >> (bit_position_ & 7));
  }
  ++bit_position_;
}

void BitWriter::put_bits(std::uint32_t value, int count) {
  for (int i = count - 1; i >= 0; --i) {
    put_bit(((value >> i) & 1U) != 0U);
  }
}

bool BitReader::read_bit(bool& value) {
  if (bit_position_ >= static_cast<int>(bytes_.size()) * 8) {
    return false;
  }
  const std::uint8_t byte = bytes_[static_cast<std::size_t>(bit_position_ >> 3)];
  value = ((byte >> (7 - (bit_position_ & 7))) & 1U) != 0U;
  ++bit_position_;
  return true;
}

bool BitReader::read_bits(int count, std::uint32_t& value) {
  if (count < 0 || count > 32) {
    return false;
  }
  std::uint32_t result = 0;
  for (int i = 0; i < count; ++i) {
    bool bit = false;
    if (!read_bit(bit)) {
      return false;
    }
    result = (result << 1) | (bit ? 1U : 0U);
  }
  value = result;
  return true;
}

void write_sequence_intra_config(const SequenceIntraConfig& config,
                                 bool monochrome, BitWriter& writer) {
  writer.put_bit(config.enable_dip);
  writer.put_bit(config.enable_intra_edge_filter);
  writer.put_bit(config.enable_mrls);
  writer.put_bit(config.enable_cfl_intra);
  if (!monochrome) {
    writer.put_bits(static_cast<std::uint32_t>(config.cfl_ds_filter_index & 3),
                    2);
  }
  writer.put_bit(config.enable_mhccp);
  writer.put_bit(config.enable_ibp);
}

bool read_sequence_intra_config(bool monochrome, BitReader& reader,
                                SequenceIntraConfig& config) {
  bool dip = false;
  bool edge = false;
  bool mrls = false;
  bool cfl_intra = false;
  bool mhccp = false;
  bool ibp = false;
  if (!reader.read_bit(dip) || !reader.read_bit(edge) ||
      !reader.read_bit(mrls) || !reader.read_bit(cfl_intra)) {
    return false;
  }
  int cfl_index = 0;
  if (!monochrome) {
    std::uint32_t value = 0;
    if (!reader.read_bits(2, value)) {
      return false;
    }
    cfl_index = static_cast<int>(value);
  }
  if (!reader.read_bit(mhccp) || !reader.read_bit(ibp)) {
    return false;
  }
  config.enable_dip = dip;
  config.enable_intra_edge_filter = edge;
  config.enable_mrls = mrls;
  config.enable_cfl_intra = cfl_intra;
  config.cfl_ds_filter_index = monochrome ? 0 : cfl_index;
  config.enable_mhccp = mhccp;
  config.enable_ibp = ibp;
  return true;
}

}  // namespace av2enc::ibp
