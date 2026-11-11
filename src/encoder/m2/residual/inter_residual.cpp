#include "encoder/m2/residual/inter_residual.h"

#include "encoder/m2/m2_syntax.h"
#include "encoder/rdo/rdo.h"
#include "encoder/rdoq/rdoq.h"
#include "encoder/rdoq/rdoq_av2_model.h"
#include "encoder/tx_variants/tx_variants.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace av2enc::m2::residual {
namespace {

using txv::TxParams;
using txv::TxSize;
using txv::TxType;

constexpr int kPlaneCount = kResidualPlaneCount;
constexpr double kSkipFlagBits = 1.0;

int clip1(int value) { return value < 0 ? 0 : (value > 255 ? 255 : value); }

int floor_log2_int(int value) {
  int log = 0;
  while (value > 1) {
    value >>= 1;
    ++log;
  }
  return log;
}

int plane_width(const BlockGeometry& geometry, int plane) {
  return plane == 0 ? geometry.luma_w : geometry.chroma_w;
}
int plane_height(const BlockGeometry& geometry, int plane) {
  return plane == 0 ? geometry.luma_h : geometry.chroma_h;
}

bool read_block(const PlaneView& view, int width, int height, int plane,
                const BlockGeometry& geometry,
                std::vector<std::uint8_t>* out, std::string& error) {
  const int w = plane_width(geometry, plane);
  const int h = plane_height(geometry, plane);
  const int stride = view.stride > 0 ? view.stride : width;
  if (width < w || height < h || stride < w) {
    error = "plane " + std::to_string(plane) + " view is smaller than the block";
    return false;
  }
  const std::size_t need = static_cast<std::size_t>(h - 1) *
                               static_cast<std::size_t>(stride) +
                           static_cast<std::size_t>(w);
  if (view.data.size() < need) {
    error = "plane " + std::to_string(plane) + " view is truncated";
    return false;
  }
  out->resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  for (int y = 0; y < h; ++y) {
    std::memcpy(out->data() + static_cast<std::size_t>(y) *
                                  static_cast<std::size_t>(w),
                view.data.data() +
                    static_cast<std::size_t>(y) * static_cast<std::size_t>(stride),
                static_cast<std::size_t>(w));
  }
  return true;
}

// -log2 P(symbol) from a §9.3 CDF row (ascending cumulative convention; see
// av2_entropy.h compute_boundaries()).
double cdf_cost(const std::uint16_t* cdf, int n, int symbol) {
  if (n <= 1) return 0.0;
  double probability = 0.0;
  const double scale = 32768.0;
  if (symbol == 0) {
    probability = static_cast<double>(cdf[0]) / scale;
  } else if (symbol < n - 1) {
    probability = static_cast<double>(cdf[symbol] - cdf[symbol - 1]) / scale;
  } else {
    probability = (scale - static_cast<double>(cdf[n - 2])) / scale;
  }
  if (probability <= 0.0) return 16.0;  // zero-probability symbol
  return -std::log2(probability);
}

// Bit-accounting sink: charges -log2 P(symbol) from the same §9.3 CDFs the
// writer uses, so the skip-vs-residual decision sees the real cost of the
// coefficients it is about to emit (escapes and signs included).
struct BitCounter {
  double bits = 0.0;
};

bool count_symbol(void* context, std::uint16_t* cdf, int n, int symbol) {
  auto* counter = static_cast<BitCounter*>(context);
  counter->bits += cdf_cost(cdf, n, symbol);
  return true;
}

bool count_bool(void* context, bool /*bit*/) {
  auto* counter = static_cast<BitCounter*>(context);
  counter->bits += 1.0;
  return true;
}

// Exact syntax cost of one residual block under a neutral (all-zero) neighbour
// context.  The real stream shares a frame-level context, so this is still an
// estimate, but it is the writer's own symbol sequence rather than a model.
bool measure_block_bits(const InterResidualBlock& block,
                        std::span<const std::int32_t> levels, double* bits,
                        std::string* error) {
  const int luma_cols = block.mi_col + block.luma_w / 4 + 1;
  const int luma_rows = block.mi_row + block.luma_h / 4 + 1;
  const int chroma_cols = block.mi_col / 2 + block.chroma_w / 4 + 1;
  const int chroma_rows = block.mi_row / 2 + block.chroma_h / 4 + 1;
  const int cols[3] = {luma_cols + kCoeffContextPad,
                       chroma_cols + kCoeffContextPad,
                       chroma_cols + kCoeffContextPad};
  const int rows[3] = {luma_rows + kCoeffContextPad,
                       chroma_rows + kCoeffContextPad,
                       chroma_rows + kCoeffContextPad};
  std::vector<std::uint8_t> above_level[3];
  std::vector<std::uint8_t> left_level[3];
  std::vector<std::uint8_t> above_dc[3];
  std::vector<std::uint8_t> left_dc[3];
  CoeffContextView contexts;
  contexts.mi_cols = luma_cols;
  contexts.mi_rows = luma_rows;
  int eob_u = 0;
  contexts.eob_u = &eob_u;
  for (int plane = 0; plane < 3; ++plane) {
    above_level[plane].assign(static_cast<std::size_t>(cols[plane]), 0);
    left_level[plane].assign(static_cast<std::size_t>(rows[plane]), 0);
    above_dc[plane].assign(static_cast<std::size_t>(cols[plane]), 0);
    left_dc[plane].assign(static_cast<std::size_t>(rows[plane]), 0);
    contexts.above_level[plane] = above_level[plane].data();
    contexts.left_level[plane] = left_level[plane].data();
    contexts.above_dc[plane] = above_dc[plane].data();
    contexts.left_dc[plane] = left_dc[plane].data();
  }
  InterResidualCdfs cdfs;
  init_inter_residual_cdfs(cdfs);
  BitCounter counter;
  SymbolSink sink;
  sink.context = &counter;
  sink.encode_symbol = &count_symbol;
  sink.encode_bool = &count_bool;
  if (!write_inter_residual(sink, contexts, cdfs, block, levels, *error)) {
    return false;
  }
  *bits = counter.bits;
  return true;
}

int eob_from_levels_scan(const std::vector<std::int32_t>& levels_raster,
                         int adj_w, int adj_h) {
  std::vector<std::uint16_t> scan(
      static_cast<std::size_t>(adj_w * adj_h));
  av2enc::m2::syntax::build_scan(adj_w, adj_h, scan);
  for (int c = adj_w * adj_h - 1; c >= 0; --c) {
    const int pos = scan[static_cast<std::size_t>(c)];
    if (levels_raster[static_cast<std::size_t>(pos)] != 0) return c + 1;
  }
  return 0;
}

std::vector<std::int32_t> scan_levels(const PlaneResidual& plane) {
  const int area = plane.adj_w * plane.adj_h;
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  av2enc::m2::syntax::build_scan(plane.adj_w, plane.adj_h, scan);
  std::vector<std::int32_t> out(static_cast<std::size_t>(plane.eob));
  for (int c = 0; c < plane.eob; ++c) {
    out[static_cast<std::size_t>(c)] =
        plane.levels[static_cast<std::size_t>(
            scan[static_cast<std::size_t>(c)])];
  }
  return out;
}

double tx_type_symbol_bits(TxSize tx_size, TxType type, int eob) {
  const InterTxSet set = inter_tx_set(tx_size, /*reduced_tx_set=*/0);
  if (set == InterTxSet::kDctOnly) return 0.0;
  InterTxCandidate candidate;
  if (!inter_tx_symbol(tx_size, /*reduced_tx_set=*/0,
                       /*enable_inter_ddt=*/false, type, &candidate)) {
    return 16.0;
  }
  static const InterResidualCdfs* cdfs = [] {
    static InterResidualCdfs storage;
    init_inter_residual_cdfs(storage);
    return &storage;
  }();
  const int ctx = std::clamp(inter_tx_type_context(eob, tx_size), 0, 2);
  const int sqr_sz = std::clamp(tx_size_sqr(tx_size), 0, 3);
  switch (set) {
    case InterTxSet::kInter1:
      return cdf_cost(&cdfs->inter_tx_type_set1[ctx][sqr_sz][0], 2,
                      candidate.symbol) +
             cdf_cost(candidate.symbol == 0
                          ? &cdfs->inter_tx_type_index_set1[ctx][0]
                          : &cdfs->inter_tx_type_offset_set1[ctx][0],
                      8, candidate.offset_symbol);
    case InterTxSet::kInter2: {
      const int offset_ctx =
          std::clamp(inter_tx_type_offset_context(eob, tx_size), 0, 2);
      return cdf_cost(&cdfs->inter_tx_type_set2[ctx][0], 2,
                      candidate.symbol) +
             cdf_cost(candidate.symbol == 0
                          ? &cdfs->inter_tx_type_index_set2[offset_ctx][0]
                          : &cdfs->inter_tx_type_offset_set2[offset_ctx][0],
                      candidate.symbol == 0 ? 8 : 4, candidate.offset_symbol);
    }
    case InterTxSet::kDctIdtx:
      return cdf_cost(&cdfs->inter_tx_type_set3[ctx][sqr_sz][0], 2,
                      candidate.symbol);
    case InterTxSet::kDctIdtxIddct:
      return cdf_cost(&cdfs->inter_tx_type_set4[ctx][sqr_sz][0], 4,
                      candidate.symbol);
    case InterTxSet::kWide64:
    case InterTxSet::kHigh64:
    case InterTxSet::kWide32:
    case InterTxSet::kHigh32: {
      double bits = cdf_cost(&cdfs->inter_tx_type_long[ctx][sqr_sz][0], 4,
                             candidate.symbol);
      if (candidate.codes_long_flag) {
        const int flag = candidate.is_long_side_dct ? 1 : 0;
        bits += cdf_cost(&cdfs->is_long_side_dct[flag][0], 2, flag);
      }
      return bits;
    }
    case InterTxSet::kDctOnly:
      break;
  }
  return 0.0;
}

// Full transform -> quantise -> (RDOQ) -> inverse -> reconstruct for one plane
// under one transform type.
bool evaluate_plane(int plane, const BlockGeometry& geometry,
                    const std::vector<std::uint8_t>& source,
                    const std::vector<std::uint8_t>& prediction,
                    TxSize tx_size, TxType tx_type,
                    const DecisionOptions& options, PlaneResidual* out,
                    double* distortion, double* rate_bits,
                    std::string& error) {
  const auto params_opt = txv::make_tx_params(tx_size, tx_type);
  if (!params_opt.has_value()) {
    error = "unsupported transform size/type";
    return false;
  }
  const TxParams& params = *params_opt;
  const int w = plane_width(geometry, plane);
  const int h = plane_height(geometry, plane);
  if (params.coded_w != w || params.coded_h != h) {
    error = "transform size does not match the plane residual size";
    return false;
  }
  const int coded_pels = w * h;
  std::vector<std::int16_t> residual_coded(
      static_cast<std::size_t>(coded_pels));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int src = source[static_cast<std::size_t>(y * w + x)];
      const int pred = prediction[static_cast<std::size_t>(y * w + x)];
      residual_coded[static_cast<std::size_t>(y * w + x)] =
          static_cast<std::int16_t>(src - pred);
    }
  }
  const int area = params.w * params.h;
  std::vector<std::int16_t> residual_adj(static_cast<std::size_t>(area));
  txv::downsample_residual(residual_coded.data(), params,
                           residual_adj.data());
  std::vector<double> coeff(static_cast<std::size_t>(area));
  std::vector<std::int32_t> levels(static_cast<std::size_t>(area), 0);
  txv::forward_transform(residual_adj.data(), params, coeff.data(),
                         levels.data());

  if (options.rdoq) {
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    av2enc::m2::syntax::build_scan(static_cast<int>(params.w),
                                   static_cast<int>(params.h), scan);
    std::vector<std::int32_t> scalar_scan(static_cast<std::size_t>(area));
    std::vector<av2enc::rdoq::Coefficient> input(
        static_cast<std::size_t>(area));
    for (int c = 0; c < area; ++c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      scalar_scan[static_cast<std::size_t>(c)] = levels[static_cast<std::size_t>(pos)];
      input[static_cast<std::size_t>(c)].original = coeff[static_cast<std::size_t>(pos)];
      input[static_cast<std::size_t>(c)].scalar_level = levels[static_cast<std::size_t>(pos)];
      input[static_cast<std::size_t>(c)].level_context = 0;
    }
    av2enc::rdoq::av2::BlockInfo info;
    info.plane = plane;
    info.params = params;
    info.txb_skip_context = 0;
    info.dc_sign_context = 0;
    av2enc::rdoq::av2::Model model;
    std::string rdoq_error;
    if (!model.init(info, scalar_scan, &rdoq_error)) {
      error = "rdoq model init failed: " + rdoq_error;
      return false;
    }
    av2enc::rdoq::Config config;
    config.quant_step = 1.0 / params.quant_factor;
    config.lambda = av2enc::rdo::lambda_sse_from_quant_step(
        config.quant_step, options.rdoq_sse_scale);
    config.max_coefficients = area;
    av2enc::rdoq::RdoqResult result;
    if (!av2enc::rdoq::rdoq_quantize(input, config, model.rate_model(), result,
                                     &rdoq_error)) {
      error = "rdoq quantize failed: " + rdoq_error;
      return false;
    }
    std::fill(levels.begin(), levels.end(), 0);
    for (int c = 0; c < result.eob; ++c) {
      levels[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])] =
          result.levels[static_cast<std::size_t>(c)];
    }
  }

  out->tx_size = static_cast<int>(tx_size);
  out->tx_type = static_cast<int>(tx_type);
  out->coded_w = w;
  out->coded_h = h;
  out->adj_w = params.w;
  out->adj_h = params.h;
  out->levels = std::move(levels);
  out->eob = eob_from_levels_scan(out->levels, params.w, params.h);

  std::vector<std::int32_t> dequant(static_cast<std::size_t>(area));
  std::vector<std::int16_t> recon_residual(
      static_cast<std::size_t>(coded_pels));
  txv::inverse_transform(out->levels.data(), params, dequant.data(),
                         recon_residual.data());
  out->recon.resize(static_cast<std::size_t>(coded_pels));
  double sse = 0.0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t index = static_cast<std::size_t>(y * w + x);
      const int pred = prediction[index];
      const int value = clip1(pred + recon_residual[index]);
      out->recon[index] = static_cast<std::uint8_t>(value);
      const int d = static_cast<int>(source[index]) - value;
      sse += static_cast<double>(d) * static_cast<double>(d);
    }
  }
  *distortion = sse;
  *rate_bits = estimate_plane_rate_bits(*out, options);
  return true;
}

double plane_sse(const std::vector<std::uint8_t>& a,
                 const std::vector<std::uint8_t>& b) {
  double sse = 0.0;
  const std::size_t count = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < count; ++i) {
    const int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
    sse += static_cast<double>(d) * static_cast<double>(d);
  }
  return sse;
}

}  // namespace

bool inter_tx_sizes(const BlockGeometry& geometry, int* luma_tx_size,
                    int* chroma_tx_size) {
  if (luma_tx_size != nullptr) {
    *luma_tx_size = txv::tx_size_index(geometry.luma_w, geometry.luma_h);
    if (*luma_tx_size < 0) return false;
  }
  if (chroma_tx_size != nullptr) {
    *chroma_tx_size = txv::tx_size_index(geometry.chroma_w, geometry.chroma_h);
    if (*chroma_tx_size < 0) return false;
  }
  return true;
}

double estimate_plane_rate_bits(const PlaneResidual& plane,
                                const DecisionOptions& options) {
  (void)options;
  if (plane.eob == 0) {
    // txb_skip "all zero" symbol: about one bit at the default CDF.
    return 1.2;
  }
  const auto tx_size = static_cast<TxSize>(plane.tx_size);
  const auto tx_type = static_cast<TxType>(plane.tx_type);
  double bits = 0.0;
  // EOB: eob_pt symbol + eob_extra + literal bits.
  int eob_pt = 1;
  if (plane.eob == 2) {
    eob_pt = 2;
  } else if (plane.eob >= 3) {
    eob_pt = floor_log2_int(plane.eob - 1) + 2;
  }
  const int multisize = std::min(5, floor_log2_int(plane.coded_w)) +
                        std::min(5, floor_log2_int(plane.coded_h)) - 4;
  int cdf_n = 8;
  if (multisize == 0) {
    cdf_n = 5;
  } else if (multisize == 1) {
    cdf_n = 6;
  } else if (multisize == 2) {
    cdf_n = 7;
  }
  bits += std::log2(static_cast<double>(cdf_n));
  bits += 1.0 + 0.5 * static_cast<double>(eob_pt);
  // Transform type symbol (luma only; chroma codes none).  The set is empty
  // for TX_SET_DCTONLY, where tx_type_symbol_bits() returns 0.
  bits += tx_type_symbol_bits(tx_size, tx_type, plane.eob);
  const std::vector<std::int32_t> levels = scan_levels(plane);
  // Coefficient magnitudes and signs.  This is the documented bring-up model:
  // it charges a neutral-context estimate instead of re-deriving the §8.3.2
  // neighbour contexts (the writer does that exactly when it codes the
  // stream).  It is only used for the skip-vs-residual cost.
  for (int c = 0; c < plane.eob; ++c) {
    const int level = levels[static_cast<std::size_t>(c)];
    const int abs_level = level < 0 ? -level : level;
    if (abs_level == 0) {
      bits += 0.9;
      continue;
    }
    bits += 1.2;  // base symbol
    bits += 1.0;  // sign
    if (abs_level > 2) {
      bits += 1.0 + 0.4 * static_cast<double>(abs_level - 3);
    }
  }
  return bits;
}

bool decide_inter_block(const BlockInput& input, const DecisionOptions& options,
                        BlockDecision& decision, std::string& error) {
  error.clear();
  decision = BlockDecision{};
  const BlockGeometry& geometry = input.geometry;
  if (geometry.luma_w <= 0 || geometry.luma_h <= 0 || geometry.chroma_w <= 0 ||
      geometry.chroma_h <= 0) {
    error = "invalid block geometry";
    return false;
  }
  int luma_tx = 0;
  int chroma_tx = 0;
  if (!inter_tx_sizes(geometry, &luma_tx, &chroma_tx)) {
    error = "block size is not a codable transform size";
    return false;
  }

  std::vector<std::uint8_t> source[kPlaneCount];
  std::vector<std::uint8_t> prediction[kPlaneCount];
  for (int plane = 0; plane < kPlaneCount; ++plane) {
    if (!read_block(input.source[plane], plane_width(geometry, plane),
                    plane_height(geometry, plane), plane, geometry,
                    &source[plane], error)) {
      return false;
    }
    if (!read_block(input.prediction[plane], plane_width(geometry, plane),
                    plane_height(geometry, plane), plane, geometry,
                    &prediction[plane], error)) {
      return false;
    }
  }

  decision.skip_distortion = 0.0;
  for (int plane = 0; plane < kPlaneCount; ++plane) {
    decision.skip_distortion += plane_sse(source[plane], prediction[plane]);
  }
  decision.skip_rate_bits = kSkipFlagBits;
  const double quant_step = 1.0 / txv::make_tx_params(
                                      static_cast<TxSize>(luma_tx),
                                      TxType::kDctDct)
                                      ->quant_factor;
  const double lambda =
      options.lambda > 0.0
          ? options.lambda
          : av2enc::rdo::lambda_sse_from_quant_step(quant_step,
                                                    options.lambda_sse_scale);
  decision.skip_cost =
      decision.skip_distortion + lambda * decision.skip_rate_bits;

  if (!options.allow_residual) {
    decision.skip = true;
    decision.cost = decision.skip_cost;
    decision.distortion = decision.skip_distortion;
    decision.rate_bits = decision.skip_rate_bits;
    // Keep the reconstruction contract unconditional: a skip block reconstructs
    // to the prediction, so callers never need a branch.
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      PlaneResidual& residual = decision.plane[plane];
      const auto params = txv::make_tx_params(
          static_cast<TxSize>(plane == 0 ? luma_tx : chroma_tx),
          TxType::kDctDct);
      residual.tx_size = plane == 0 ? luma_tx : chroma_tx;
      residual.tx_type = static_cast<int>(TxType::kDctDct);
      residual.coded_w = params->coded_w;
      residual.coded_h = params->coded_h;
      residual.adj_w = params->w;
      residual.adj_h = params->h;
      residual.eob = 0;
      residual.levels.assign(static_cast<std::size_t>(params->w * params->h),
                            0);
      residual.recon = prediction[plane];
    }
    return true;
  }

  decision.residual_considered = true;
  double total_distortion = 0.0;
  double total_rate = 0.0;
  for (int plane = 0; plane < kPlaneCount; ++plane) {
    const auto tx_size =
        static_cast<TxSize>(plane == 0 ? luma_tx : chroma_tx);
    TxType best_type = TxType::kDctDct;
    PlaneResidual best;
    double best_distortion = 0.0;
    double best_rate = 0.0;
    double best_cost = std::numeric_limits<double>::infinity();
    bool have_best = false;

    InterTxSetInfo candidates;
    // Diagnostic: force one luma TX type (codable sizes only).  Used by the
    // decoder gate to isolate a single transform; not a decision knob.
    const char* forced_type = plane == 0
                                  ? std::getenv("AV2ENC_M2_RESIDUAL_TX_TYPE")
                                  : nullptr;
    if (plane == 0 && options.try_inter_tx_types) {
      if (!inter_tx_candidates(tx_size, options.reduced_tx_set,
                               options.enable_inter_ddt, &candidates)) {
        error = "inter TX type candidate enumeration failed";
        return false;
      }
    } else {
      candidates.count = 1;
      candidates.set = InterTxSet::kDctOnly;
      candidates.candidates[0].type = TxType::kDctDct;
    }

    if (forced_type != nullptr) {
      const int wanted = std::atoi(forced_type);
      int kept = 0;
      for (int i = 0; i < candidates.count; ++i) {
        if (static_cast<int>(
                candidates.candidates[static_cast<std::size_t>(i)].type) ==
            wanted) {
          candidates.candidates[static_cast<std::size_t>(kept++)] =
              candidates.candidates[static_cast<std::size_t>(i)];
        }
      }
      if (kept == 0) {
        // The forced type is not codable for this TX size; fall back to DCT.
        int dct_index = 0;
        for (int i = 0; i < candidates.count; ++i) {
          if (candidates.candidates[static_cast<std::size_t>(i)].type ==
              TxType::kDctDct) {
            dct_index = i;
            break;
          }
        }
        candidates.candidates[0] =
            candidates.candidates[static_cast<std::size_t>(dct_index)];
        kept = 1;
      }
      candidates.count = kept;
    }
    for (int i = 0; i < candidates.count; ++i) {
      const TxType type = candidates.candidates[static_cast<std::size_t>(i)].type;
      PlaneResidual candidate;
      double distortion = 0.0;
      double rate = 0.0;
      if (!evaluate_plane(plane, geometry, source[plane], prediction[plane],
                          tx_size, type, options, &candidate, &distortion,
                          &rate, error)) {
        return false;
      }
      const double cost = distortion + lambda * rate;
      const bool better =
          !have_best || cost < best_cost ||
          (cost == best_cost && distortion < best_distortion) ||
          (cost == best_cost && distortion == best_distortion &&
           static_cast<int>(type) < static_cast<int>(best_type));
      if (better) {
        have_best = true;
        best = std::move(candidate);
        best_type = type;
        best_distortion = distortion;
        best_rate = rate;
        best_cost = cost;
      }
    }
    if (!have_best) {
      error = "no inter TX type candidate";
      return false;
    }
    decision.plane[plane] = std::move(best);
    total_distortion += best_distortion;
    total_rate += best_rate;
  }

  bool zero_residual = true;
  for (int plane = 0; plane < kPlaneCount; ++plane) {
    if (decision.plane[plane].eob != 0) zero_residual = false;
  }
  decision.zero_residual = zero_residual;

  // A residual that quantises to all-zero is exactly the skip reconstruction;
  // never pay the skip_flag = 0 syntax for it.
  if (zero_residual) {
    decision.skip = true;
    decision.distortion = decision.skip_distortion;
    decision.rate_bits = decision.skip_rate_bits;
    decision.cost = decision.skip_cost;
    return true;
  }

  decision.distortion = total_distortion;
  decision.rate_bits = total_rate;
  decision.cost = total_distortion + lambda * total_rate;

  // Replace the per-plane estimate with the writer's own symbol cost: escapes
  // and BR codes dominate large inter residuals and the estimate alone makes
  // the decision code too many residuals.
  decision.skip = false;  // draft: the residual syntax is built below
  InterResidualBlock syntax;
  std::vector<std::int32_t> block_levels;
  if (!make_syntax_block(geometry, decision, &syntax, &block_levels, error)) {
    return false;
  }
  double measured_bits = 0.0;
  if (!measure_block_bits(syntax, block_levels, &measured_bits, &error)) {
    return false;
  }
  // + skip_flag = 0 versus skip_flag = 1.
  decision.rate_bits = measured_bits + 1.0;
  decision.cost = decision.distortion + lambda * decision.rate_bits;
  decision.skip = decision.cost >= decision.skip_cost;
  return true;
}

void append_scan_levels(const PlaneResidual& plane,
                        std::vector<std::int32_t>* out) {
  if (out == nullptr) return;
  const int area = plane.adj_w * plane.adj_h;
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  av2enc::m2::syntax::build_scan(plane.adj_w, plane.adj_h, scan);
  for (int c = 0; c < plane.eob; ++c) {
    out->push_back(plane.levels[static_cast<std::size_t>(
        scan[static_cast<std::size_t>(c)])]);
  }
}

bool make_syntax_block(const BlockGeometry& geometry,
                       const BlockDecision& decision,
                       InterResidualBlock* block,
                       std::vector<std::int32_t>* levels_out,
                       std::string& error) {
  error.clear();
  if (block == nullptr) {
    error = "null syntax block";
    return false;
  }
  if (decision.skip) {
    error = "cannot build residual syntax for a skip decision";
    return false;
  }
  *block = InterResidualBlock{};
  block->mi_row = geometry.mi_row;
  block->mi_col = geometry.mi_col;
  block->luma_w = geometry.luma_w;
  block->luma_h = geometry.luma_h;
  block->chroma_w = geometry.chroma_w;
  block->chroma_h = geometry.chroma_h;
  std::size_t offset =
      levels_out == nullptr ? 0 : levels_out->size();
  for (int plane = 0; plane < kPlaneCount; ++plane) {
    const PlaneResidual& plane_result = decision.plane[plane];
    if (plane_result.tx_size < 0 || plane_result.tx_size > 24) {
      error = "invalid tx_size in the decision";
      return false;
    }
    if (plane_result.coded_w != plane_width(geometry, plane) ||
        plane_result.coded_h != plane_height(geometry, plane)) {
      error = "decision plane size does not match the block geometry";
      return false;
    }
    block->plane[plane].tx_size =
        static_cast<std::uint8_t>(plane_result.tx_size);
    block->plane[plane].tx_type =
        static_cast<std::uint8_t>(plane_result.tx_type);
    block->plane[plane].eob =
        static_cast<std::uint32_t>(plane_result.eob);
    block->plane[plane].levels_offset = static_cast<std::uint32_t>(offset);
    if (levels_out != nullptr) {
      append_scan_levels(plane_result, levels_out);
      offset = levels_out->size();
    } else {
      offset += static_cast<std::size_t>(plane_result.eob);
    }
  }
  return true;
}

bool decide_inter_batch(const BatchInput& input, BatchOutput& output,
                        std::string* error) {
  if (error != nullptr) error->clear();
  const std::size_t count = input.blocks.size();
  if (output.status.size() < count || output.skip.size() < count ||
      output.distortion.size() < count || output.cost.size() < count ||
      output.eob.size() < count * kPlaneCount ||
      output.tx_size.size() < count * kPlaneCount ||
      output.tx_type.size() < count * kPlaneCount) {
    if (error != nullptr) *error = "batch output spans are too small";
    return false;
  }
  bool all_ok = true;
  for (std::size_t b = 0; b < count; ++b) {
    const BatchBlock& item = input.blocks[b];
    BlockInput block;
    block.geometry.mi_row = item.mi_row;
    block.geometry.mi_col = item.mi_col;
    block.geometry.luma_w = item.luma_w;
    block.geometry.luma_h = item.luma_h;
    block.geometry.chroma_w = item.chroma_w;
    block.geometry.chroma_h = item.chroma_h;
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      const std::size_t plane_area =
          static_cast<std::size_t>(plane_width(block.geometry, plane)) *
          static_cast<std::size_t>(plane_height(block.geometry, plane));
      if (item.src_offset[plane] < 0 || item.pred_offset[plane] < 0 ||
          static_cast<std::size_t>(item.src_offset[plane]) + plane_area >
              input.source[plane].size() ||
          static_cast<std::size_t>(item.pred_offset[plane]) + plane_area >
              input.prediction[plane].size()) {
        output.status[b] = -1;
        all_ok = false;
        continue;
      }
      // The view covers the rest of the plane from the block origin, so the
      // caller's row stride is honoured instead of assuming a tight crop.
      block.source[plane].data =
          input.source[plane].subspan(static_cast<std::size_t>(
              item.src_offset[plane]));
      block.source[plane].stride =
          item.src_stride[plane] > 0 ? item.src_stride[plane]
                                     : plane_width(block.geometry, plane);
      block.prediction[plane].data =
          input.prediction[plane].subspan(
              static_cast<std::size_t>(item.pred_offset[plane]));
      block.prediction[plane].stride =
          item.pred_stride[plane] > 0 ? item.pred_stride[plane]
                                      : plane_width(block.geometry, plane);
    }
    DecisionOptions options = input.options;
    options.allow_residual =
        (item.flags & kBatchAllowResidual) != 0 && options.allow_residual;
    if ((item.flags & kBatchTryTxTypes) != 0) options.try_inter_tx_types = true;
    if ((item.flags & kBatchRdoq) != 0) options.rdoq = true;

    BlockDecision decision;
    std::string block_error;
    if (!decide_inter_block(block, options, decision, block_error)) {
      output.status[b] = -1;
      if (error != nullptr && error->empty()) {
        *error = "block " + std::to_string(b) + ": " + block_error;
      }
      all_ok = false;
      continue;
    }
    output.status[b] = 0;
    output.skip[b] = decision.skip ? 1 : 0;
    output.distortion[b] = decision.distortion;
    output.cost[b] = decision.cost;
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      const PlaneResidual& residual = decision.plane[plane];
      const std::size_t slot =
          b * static_cast<std::size_t>(kPlaneCount) +
          static_cast<std::size_t>(plane);
      output.eob[slot] = residual.eob;
      output.tx_size[slot] = residual.tx_size;
      output.tx_type[slot] = residual.tx_type;
      if (item.levels_offset[plane] < 0 || item.recon_offset[plane] < 0) continue;
      const std::size_t level_area =
          static_cast<std::size_t>(residual.adj_w) *
          static_cast<std::size_t>(residual.adj_h);
      const std::size_t recon_area =
          static_cast<std::size_t>(residual.coded_w) *
          static_cast<std::size_t>(residual.coded_h);
      if (static_cast<std::size_t>(item.levels_offset[plane]) + level_area >
              output.levels.size() ||
          static_cast<std::size_t>(item.recon_offset[plane]) + recon_area >
              output.recon.size()) {
        output.status[b] = -1;
        if (error != nullptr && error->empty()) {
          *error = "block " + std::to_string(b) + ": output buffer overflow";
        }
        all_ok = false;
        break;
      }
      std::copy(residual.levels.begin(), residual.levels.end(),
                output.levels.begin() + item.levels_offset[plane]);
      std::copy(residual.recon.begin(), residual.recon.end(),
                output.recon.begin() + item.recon_offset[plane]);
    }
  }
  return all_ok;
}

}  // namespace av2enc::m2::residual
