// AV2 v1.0.0 DIP CPU reference.  See dip.h for the normative section map.

#include "encoder/dip/dip.h"

#include "encoder/dip/dip_weights.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace av2enc::dip {
namespace {

constexpr int kRoundShift = 10;
constexpr int kRoundOffset = 1 << (kRoundShift - 1);

int clip3(int low, int high, int value) noexcept {
  return value < low ? low : (value > high ? high : value);
}

int clip1(int value, int bit_depth) noexcept {
  return clip3(0, (1 << bit_depth) - 1, value);
}

int round2_10(int value) noexcept {
  // The spec's Round2(x, 10) is an arithmetic shift; C++ signed >> is
  // arithmetic on every supported host and on CUDA.
  return (value + kRoundOffset) >> kRoundShift;
}

int read_sample(const PlaneView& plane, int x, int y) noexcept {
  if (plane.bytes_per_sample == 1) {
    const auto* data = static_cast<const std::uint8_t*>(plane.data);
    return static_cast<int>(
        data[static_cast<std::size_t>(y) * static_cast<std::size_t>(plane.stride) +
             static_cast<std::size_t>(x)]);
  }
  const auto* data = static_cast<const std::uint16_t*>(plane.data);
  return static_cast<int>(
      data[static_cast<std::size_t>(y) * static_cast<std::size_t>(plane.stride) +
           static_cast<std::size_t>(x)]);
}

bool valid_plane(const PlaneView& plane) noexcept {
  return plane.data != nullptr && plane.stride > 0 &&
         (plane.bytes_per_sample == 1 || plane.bytes_per_sample == 2);
}

}  // namespace

bool valid_block_dim(int v) noexcept {
  return v == 4 || v == 8 || v == 16 || v == 32 || v == 64;
}

bool dip_block_available(int w, int h) noexcept {
  return valid_block_dim(w) && valid_block_dim(h) && w > 4 && h > 4 &&
         static_cast<std::int64_t>(w) * static_cast<std::int64_t>(h) >= 128;
}

const char* status_string(Status status) noexcept {
  switch (status) {
    case Status::kOk:
      return "ok";
    case Status::kInvalidArgument:
      return "invalid argument";
    case Status::kUnavailable:
      return "unavailable";
  }
  return "unknown";
}

Status prepare_edges(const Block& block, int bit_depth, Edges& edges) noexcept {
  if (bit_depth < 8 || bit_depth > 12) return Status::kInvalidArgument;
  if (!dip_block_available(block.w, block.h)) return Status::kUnavailable;
  if (!valid_plane(block.recon)) return Status::kInvalidArgument;
  if (block.x < 0 || block.y < 0 || block.max_x < block.x ||
      block.max_y < block.y) {
    return Status::kInvalidArgument;
  }
  if (block.recon.stride < block.max_x + 1) return Status::kInvalidArgument;
  if (block.have_left && block.x == 0) return Status::kInvalidArgument;
  if (block.have_above && block.y == 0) return Status::kInvalidArgument;
  if (block.num4_above_right < 0 || block.num4_below_left < 0) {
    return Status::kInvalidArgument;
  }

  edges.above.clear();
  edges.left.clear();
  edges.w = block.w;
  edges.h = block.h;
  edges.bit_depth = bit_depth;
  edges.have_left = block.have_left;
  edges.have_above = block.have_above;

  // §7.13.2.1: useDip sets numAboveNeeded = w + (w >> 2) and
  // numLeftNeeded = h + (h >> 2).  MrlIndex is 0 for DIP (DC_PRED cannot
  // signal MRL), so aboveMrlIndex is 0 as well.
  const int num_above = block.w + (block.w >> 2);
  const int num_left = block.h + (block.h >> 2);
  const int mid = 1 << (bit_depth - 1);

  for (int i = 0; i < num_left; ++i) {
    if (!block.have_left && block.have_above) {
      // Left unavailable, above available: replicate the sample above the
      // block's top-left corner (after the MRL offset, which is 0 here).
      edges.left[i] = static_cast<std::int16_t>(
          read_sample(block.recon, block.x, block.y - 1));
    } else if (!block.have_left) {
      edges.left[i] = static_cast<std::int16_t>(mid + 1);
    } else {
      const int limit =
          std::min(block.max_y,
                   block.y + block.h + 4 * block.num4_below_left - 1);
      const int row = std::min(limit, block.y + i);
      edges.left[i] = static_cast<std::int16_t>(
          read_sample(block.recon, block.x - 1, row));
    }
  }

  for (int i = 0; i < num_above; ++i) {
    if (!block.have_above && block.have_left) {
      edges.above[i] = static_cast<std::int16_t>(
          read_sample(block.recon, block.x - 1, block.y));
    } else if (!block.have_above) {
      edges.above[i] = static_cast<std::int16_t>(mid - 1);
    } else {
      const int limit =
          std::min(block.max_x,
                   block.x + block.w + 4 * block.num4_above_right - 1);
      const int col = std::min(limit, block.x + i);
      edges.above[i] = static_cast<std::int16_t>(
          read_sample(block.recon, col, block.y - 1));
    }
  }

  // The DIP feature f[0] is AboveRow[-1].  For MrlIndex = 0 the spec's loop
  // runs for i = 1 only.
  if (block.have_above && block.have_left) {
    const int corner =
        read_sample(block.recon, block.x - 1, block.y - 1);
    edges.above[-1] = static_cast<std::int16_t>(corner);
    edges.left[-1] = static_cast<std::int16_t>(corner);
  } else if (block.have_above) {
    const int corner = read_sample(block.recon, block.x, block.y - 1);
    edges.above[-1] = static_cast<std::int16_t>(corner);
    edges.left[-1] = static_cast<std::int16_t>(corner);
  } else if (block.have_left) {
    const int corner = read_sample(block.recon, block.x - 1, block.y);
    edges.above[-1] = static_cast<std::int16_t>(corner);
    edges.left[-1] = static_cast<std::int16_t>(corner);
  } else {
    edges.above[-1] = static_cast<std::int16_t>(mid);
    edges.left[-1] = static_cast<std::int16_t>(mid);
  }

  return Status::kOk;
}

Status dip_avg(const EdgeArray& row, int n,
               std::array<int, 5>& out) noexcept {
  const int down = n >> 2;
  if (down <= 0 || n > kMaxBlockDim) return Status::kInvalidArgument;
  // The largest index read is 4 * down + down - 1 = 5 * down - 1 <= 79.
  if (5 * down - 1 > kMaxEdgeSamples - 1) return Status::kInvalidArgument;
  for (int i = 0; i < 5; ++i) {
    int t = 0;
    for (int j = 0; j < down; ++j) {
      t += row[i * down + j];
    }
    out[static_cast<std::size_t>(i)] = (t + (down >> 1)) / down;
  }
  return Status::kOk;
}

Status extract_features(const Edges& edges, bool transpose,
                        std::array<int, kNumFeatures>& f) noexcept {
  if (!dip_block_available(edges.w, edges.h)) return Status::kUnavailable;
  std::array<int, 5> above_avg{};
  std::array<int, 5> left_avg{};
  Status status = dip_avg(edges.above, edges.w, above_avg);
  if (status != Status::kOk) return status;
  status = dip_avg(edges.left, edges.h, left_avg);
  if (status != Status::kOk) return status;

  f[0] = edges.above[-1];
  for (int i = 0; i < 4; ++i) {
    f[static_cast<std::size_t>(i + 1)] =
        transpose ? left_avg[static_cast<std::size_t>(i)]
                  : above_avg[static_cast<std::size_t>(i)];
    f[static_cast<std::size_t>(i + 5)] =
        transpose ? above_avg[static_cast<std::size_t>(i)]
                  : left_avg[static_cast<std::size_t>(i)];
  }
  f[9] = transpose ? left_avg[4] : above_avg[4];
  f[10] = transpose ? above_avg[4] : left_avg[4];
  return Status::kOk;
}

void transform(const std::array<int, kNumFeatures>& f, int mode, bool transpose,
               int bit_depth,
               std::array<std::uint16_t, kTransformSamples>& pred) noexcept {
  if (mode < 0 || mode >= kNumModes || bit_depth < 8 || bit_depth > 12) {
    pred.fill(0);
    return;
  }
  const int max_value = (1 << bit_depth) - 1;
  for (int i = 0; i < kTransformSize; ++i) {
    for (int j = 0; j < kTransformSize; ++j) {
      std::int32_t c = 0;
      for (int k = 0; k < kNumFeatures; ++k) {
        c += static_cast<std::int32_t>(
                 kDipWeights[mode][i * kTransformSize + j][k]) *
             f[static_cast<std::size_t>(k)];
      }
      const int v = clip3(0, max_value, round2_10(c));
      const int dst_index =
          transpose ? j * kTransformSize + i : i * kTransformSize + j;
      pred[static_cast<std::size_t>(dst_index)] =
          static_cast<std::uint16_t>(v);
    }
  }
}

void resample(const Edges& edges,
              const std::array<std::uint16_t, kTransformSamples>& dip_pred,
              std::uint16_t* dst, std::size_t dst_stride) noexcept {
  const int w = edges.w;
  const int h = edges.h;
  if (dst == nullptr || !dip_block_available(w, h) ||
      dst_stride < static_cast<std::size_t>(w)) {
    return;
  }
  const int upx = std::max(1, w / kTransformSize);
  const int upy = std::max(1, h / kTransformSize);
  const int downx = std::max(1, kTransformSize / w);
  const int downy = std::max(1, kTransformSize / h);

  int horz[kTransformSize][kMaxBlockDim]{};
  const int rows = std::min(h, kTransformSize);
  const int cols = std::min(w, kTransformSize);

  for (int i = 0; i < rows; ++i) {
    const int y = (i + 1) * upy - 1;
    for (int j = 0; j < cols; ++j) {
      const int p0 = j == 0
                         ? edges.left[y]
                         : dip_pred[static_cast<std::size_t>(i * downy) *
                                        kTransformSize +
                                    static_cast<std::size_t>((j - 1) * downx)];
      const int p1 =
          dip_pred[static_cast<std::size_t>(i * downy) * kTransformSize +
                   static_cast<std::size_t>(j * downx)];
      for (int k = 0; k < upx; ++k) {
        const int x = j * upx + k;
        const int w1 = k + 1;
        horz[i][x] = ((upx - w1) * p0 + w1 * p1) / upx;
      }
    }
  }

  for (int x = 0; x < w; ++x) {
    for (int i = 0; i < rows; ++i) {
      const int p0 = i == 0 ? edges.above[x] : horz[i - 1][x];
      const int p1 = horz[i][x];
      for (int k = 0; k < upy; ++k) {
        const int y = i * upy + k;
        const int w1 = k + 1;
        dst[static_cast<std::size_t>(y) * dst_stride +
            static_cast<std::size_t>(x)] =
            static_cast<std::uint16_t>(((upy - w1) * p0 + w1 * p1) / upy);
      }
    }
  }
}

Status validate_params(const Edges& edges, const Params& params) noexcept {
  if (params.mode < 0 || params.mode >= kNumModes) {
    return Status::kInvalidArgument;
  }
  if (!dip_block_available(edges.w, edges.h)) return Status::kUnavailable;
  if (edges.bit_depth < 8 || edges.bit_depth > 12) {
    return Status::kInvalidArgument;
  }
  return Status::kOk;
}

Status predict(const Edges& edges, const Params& params, std::uint16_t* dst,
               std::size_t dst_stride) noexcept {
  if (dst == nullptr || dst_stride < static_cast<std::size_t>(edges.w)) {
    return Status::kInvalidArgument;
  }
  Status status = validate_params(edges, params);
  if (status != Status::kOk) return status;

  std::array<int, kNumFeatures> f{};
  status = extract_features(edges, params.transpose, f);
  if (status != Status::kOk) return status;

  std::array<std::uint16_t, kTransformSamples> dip_pred{};
  transform(f, params.mode, params.transpose, edges.bit_depth, dip_pred);
  resample(edges, dip_pred, dst, dst_stride);
  return Status::kOk;
}

Status predict8(const Edges& edges, const Params& params, std::uint8_t* dst,
                std::size_t dst_stride) noexcept {
  if (edges.bit_depth != 8) return Status::kInvalidArgument;
  if (dst == nullptr || dst_stride < static_cast<std::size_t>(edges.w)) {
    return Status::kInvalidArgument;
  }
  std::array<std::uint16_t,
             static_cast<std::size_t>(kMaxBlockDim) * kMaxBlockDim>
      temp{};
  const Status status = predict(edges, params, temp.data(),
                               static_cast<std::size_t>(edges.w));
  if (status != Status::kOk) return status;
  for (int y = 0; y < edges.h; ++y) {
    for (int x = 0; x < edges.w; ++x) {
      dst[static_cast<std::size_t>(y) * dst_stride +
          static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(
          clip1(temp[static_cast<std::size_t>(y) *
                         static_cast<std::size_t>(edges.w) +
                     static_cast<std::size_t>(x)],
                8));
    }
  }
  return Status::kOk;
}

Status predict_block(const Block& block, int bit_depth, const Params& params,
                     std::uint16_t* dst, std::size_t dst_stride) noexcept {
  if (!dip_block_available(block.w, block.h)) return Status::kUnavailable;
  if (params.mode < 0 || params.mode >= kNumModes) {
    return Status::kInvalidArgument;
  }
  if (dst == nullptr || dst_stride < static_cast<std::size_t>(block.w)) {
    return Status::kInvalidArgument;
  }
  Edges edges;
  const Status status = prepare_edges(block, bit_depth, edges);
  if (status != Status::kOk) return status;
  return predict(edges, params, dst, dst_stride);
}

Status predict_block8(const Block& block, int bit_depth, const Params& params,
                      std::uint8_t* dst, std::size_t dst_stride) noexcept {
  if (bit_depth != 8) return Status::kInvalidArgument;
  Edges edges;
  const Status status = prepare_edges(block, bit_depth, edges);
  if (status != Status::kOk) return status;
  return predict8(edges, params, dst, dst_stride);
}

PredictionCost prediction_cost(const std::uint16_t* src,
                               std::size_t src_stride,
                               const std::uint16_t* pred,
                               std::size_t pred_stride, int w,
                               int h) noexcept {
  PredictionCost cost;
  if (src == nullptr || pred == nullptr || w <= 0 || h <= 0) return cost;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::int64_t diff =
          static_cast<std::int64_t>(
              src[static_cast<std::size_t>(y) * src_stride +
                  static_cast<std::size_t>(x)]) -
          static_cast<std::int64_t>(
              pred[static_cast<std::size_t>(y) * pred_stride +
                   static_cast<std::size_t>(x)]);
      const std::uint64_t mag =
          static_cast<std::uint64_t>(diff < 0 ? -diff : diff);
      cost.sad += mag;
      cost.sse += static_cast<std::uint64_t>(diff * diff);
    }
  }
  return cost;
}

PredictionCost prediction_cost8(const std::uint8_t* src,
                                std::size_t src_stride,
                                const std::uint8_t* pred,
                                std::size_t pred_stride, int w,
                                int h) noexcept {
  PredictionCost cost;
  if (src == nullptr || pred == nullptr || w <= 0 || h <= 0) return cost;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::int64_t diff =
          static_cast<std::int64_t>(
              src[static_cast<std::size_t>(y) * src_stride +
                  static_cast<std::size_t>(x)]) -
          static_cast<std::int64_t>(
              pred[static_cast<std::size_t>(y) * pred_stride +
                   static_cast<std::size_t>(x)]);
      const std::uint64_t mag =
          static_cast<std::uint64_t>(diff < 0 ? -diff : diff);
      cost.sad += mag;
      cost.sse += static_cast<std::uint64_t>(diff * diff);
    }
  }
  return cost;
}

Status decide(const Edges& edges, const std::uint8_t* src,
              std::size_t src_stride, const DecisionParams& params,
              Decision& decision) noexcept {
  decision = Decision{};
  SyntaxContext context;
  context.enable_dip = params.enable_dip;
  context.y_mode = params.y_mode;
  context.palette_size_y = params.palette_size_y;
  context.tree_is_chroma_part = params.tree_is_chroma_part;
  context.w = edges.w;
  context.h = edges.h;
  if (params.metric != CostMetric::kSad && params.metric != CostMetric::kSse) {
    return Status::kInvalidArgument;
  }
  if (!dip_mode_info_present(context)) return Status::kUnavailable;
  if (src == nullptr || src_stride < static_cast<std::size_t>(edges.w)) {
    return Status::kInvalidArgument;
  }
  if (edges.bit_depth != 8) return Status::kInvalidArgument;

  std::array<std::uint8_t,
             static_cast<std::size_t>(kMaxBlockDim) * kMaxBlockDim>
      pred{};
  bool have_best = false;
  PredictionCost best{};
  int best_mode = 0;
  bool best_transpose = false;

  for (int mode = 0; mode < kNumModes; ++mode) {
    for (int transpose = 0; transpose < 2; ++transpose) {
      Params candidate;
      candidate.mode = mode;
      candidate.transpose = transpose != 0;
      const Status status = predict8(edges, candidate, pred.data(),
                                     static_cast<std::size_t>(edges.w));
      if (status != Status::kOk) return status;
      const PredictionCost cost =
          prediction_cost8(src, src_stride, pred.data(),
                           static_cast<std::size_t>(edges.w), edges.w,
                           edges.h);
      const std::uint64_t value =
          params.metric == CostMetric::kSad ? cost.sad : cost.sse;
      const std::uint64_t best_value =
          params.metric == CostMetric::kSad ? best.sad : best.sse;
      if (!have_best || value < best_value) {
        have_best = true;
        best = cost;
        best_mode = mode;
        best_transpose = transpose != 0;
      }
    }
  }

  decision.available = true;
  decision.mode = best_mode;
  decision.transpose = best_transpose;
  decision.cost = best;
  return Status::kOk;
}

bool dip_mode_info_present(const SyntaxContext& context) noexcept {
  return context.enable_dip && !context.tree_is_chroma_part &&
         context.y_mode == 0 && context.palette_size_y == 0 &&
         dip_block_available(context.w, context.h);
}

bool validate_dip_mode_info(const SyntaxContext& context,
                            const DipModeInfo& info,
                            std::string* error) noexcept {
  const bool present = dip_mode_info_present(context);
  if (!present) {
    if (info.use_dip) {
      if (error != nullptr) {
        *error = "use_dip is not signallable for this block";
      }
      return false;
    }
    if (info.dip_transpose || info.dip_mode != 0) {
      if (error != nullptr) {
        *error = "dip_transpose/dip_mode present without use_dip";
      }
      return false;
    }
    return true;
  }
  if (!info.use_dip) {
    if (info.dip_transpose || info.dip_mode != 0) {
      if (error != nullptr) {
        *error = "dip_transpose/dip_mode present without use_dip";
      }
      return false;
    }
    return true;
  }
  if (info.dip_mode < 0 || info.dip_mode >= kNumModes) {
    if (error != nullptr) {
      *error = "dip_mode outside 0..5";
    }
    return false;
  }
  return true;
}

int use_dip_context(bool above_use_dip, bool left_use_dip) noexcept {
  return (above_use_dip ? 1 : 0) + (left_use_dip ? 1 : 0);
}

std::int16_t dip_weight(int mode, int position, int feature) noexcept {
  if (mode < 0 || mode >= kNumModes || position < 0 ||
      position >= kTransformSamples || feature < 0 ||
      feature >= kNumFeatures) {
    return 0;
  }
  return kDipWeights[mode][position][feature];
}

}  // namespace av2enc::dip
