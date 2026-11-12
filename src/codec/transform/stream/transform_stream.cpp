#include "codec/transform/stream/transform_stream.h"

#include "codec/rate/rdo/rdo.h"
#include "codec/rate/rdoq/rdoq_av2_model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::still::ts {
namespace {

using txtype::TxType;
using txtype::TxTypeCandidate;
using txtype::TxTypeQuery;
using txtype::TxTypeSet;
using txtype::TxSet;

bool finite_non_negative(double value) {
  return std::isfinite(value) && value >= 0.0;
}

int abs_i32(std::int32_t value) { return value < 0 ? -value : value; }

double coefficient_sse_scan(std::span<const double> original,
                            std::span<const std::int32_t> levels,
                            double quant_step) {
  double sse = 0.0;
  const std::size_t count = std::min(original.size(), levels.size());
  for (std::size_t i = 0; i < count; ++i) {
    const double reconstructed =
        static_cast<double>(levels[i]) * quant_step;
    const double difference = original[i] - reconstructed;
    sse += difference * difference;
  }
  return sse;
}

// Builds a scan-order rate model using the caller's hook, or the AV2 §9.3 CDF
// model when no hook is supplied.  `default_model` must outlive `model`.
bool build_rate_model(const Request& request, const RateModelRequest& info,
                      rdoq::av2::Model* default_model, rdoq::RateModel* model,
                      bool* custom, std::string* error) {
  *custom = false;
  *model = rdoq::RateModel{};
  if (request.hooks != nullptr && request.hooks->build != nullptr) {
    std::string build_error;
    if (!request.hooks->build(request.hooks->user, info, model,
                              &build_error)) {
      if (error != nullptr) {
        *error = build_error.empty() ? "rate-model hook failed" : build_error;
      }
      return false;
    }
    if (model->coefficient_rate == nullptr || model->eob_rate == nullptr) {
      if (error != nullptr) {
        *error = "rate-model hook returned an incomplete model";
      }
      return false;
    }
    *custom = true;
    return true;
  }

  rdoq::av2::BlockInfo av2_info;
  av2_info.plane = info.plane;
  av2_info.params = info.params;
  av2_info.txb_skip_context = info.txb_skip_context;
  av2_info.dc_sign_context = info.dc_sign_context;
  std::string init_error;
  if (!default_model->init(av2_info, info.scalar_levels_scan, &init_error)) {
    if (error != nullptr) {
      *error = init_error.empty() ? "AV2 rate-model init failed" : init_error;
    }
    return false;
  }
  *model = default_model->rate_model();
  return true;
}

// Scalar path bits under the same callback contract the trellis uses.  This is
// what rdoq::av2::Model::scalar_bits() computes, but it also works for custom
// models.
double scalar_rate_bits(const rdoq::RateModel& model,
                        std::span<const std::int32_t> levels_scan,
                        std::int32_t eob, std::int32_t area,
                        std::int32_t prev_level_cap) {
  if (eob < 0) return 0.0;
  double bits = model.eob_rate(model.context, eob, area);
  std::int32_t prev_state = 0;
  for (std::int32_t c = 0; c < eob; ++c) {
    const std::int32_t level = levels_scan[static_cast<std::size_t>(c)];
    const std::int32_t is_eob = (c + 1 == eob) ? 1 : 0;
    bits += model.coefficient_rate(model.context, c, prev_state, level, c,
                                   is_eob);
    const std::int32_t magnitude = abs_i32(level);
    prev_state = std::min(magnitude, prev_level_cap);
  }
  return bits;
}

struct CandidateEval {
  bool ok = false;
  bool codable = false;
  TxTypeCandidate candidate{};
  TxTypeSymbol symbol{};
  std::int32_t eob = 0;
  std::int32_t nonzero = 0;
  double sse = 0.0;
  double coeff_rate_bits = 0.0;
  double type_bits = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  double rdoq_distortion = 0.0;
  double rdoq_rate_bits = 0.0;
  double rdoq_cost = 0.0;
  std::int32_t max_level_searched = 0;
  bool level_range_truncated = false;
  std::vector<std::int32_t> levels_scan;
  std::vector<std::int16_t> recon_coded;
};

bool evaluate_candidate(const Request& request, const TxTypeQuery& query,
                        const TxTypeCandidate& candidate, bool run_trellis,
                        CandidateEval* eval, std::string* error) {
  const txv::TxParams& params = candidate.params;
  const int area = params.w * params.h;
  const int coded_area = params.coded_w * params.coded_h;
  if (area <= 0 || coded_area <= 0) {
    if (error != nullptr) *error = "invalid candidate geometry";
    return false;
  }

  std::vector<std::int16_t> residual_adj(static_cast<std::size_t>(area));
  std::vector<double> coeff_raster(static_cast<std::size_t>(area));
  std::vector<std::int32_t> scalar_raster(static_cast<std::size_t>(area));
  txv::downsample_residual(request.residual_coded, params,
                           residual_adj.data());
  txv::forward_transform(residual_adj.data(), params, coeff_raster.data(),
                         scalar_raster.data());

  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  txtype::diagonal_scan(params.w, params.h, scan.data());

  std::vector<double> coeff_scan(static_cast<std::size_t>(area));
  std::vector<std::int32_t> scalar_scan(static_cast<std::size_t>(area));
  std::int32_t scalar_eob = 0;
  for (int c = 0; c < area; ++c) {
    const std::size_t pos =
        static_cast<std::size_t>(scan[static_cast<std::size_t>(c)]);
    coeff_scan[static_cast<std::size_t>(c)] = coeff_raster[pos];
    scalar_scan[static_cast<std::size_t>(c)] = scalar_raster[pos];
    if (scalar_raster[pos] != 0) scalar_eob = c + 1;
  }

  const double quant_step = 1.0 / params.quant_factor;

  RateModelRequest model_info;
  model_info.params = params;
  model_info.plane = request.plane;
  model_info.txb_skip_context = request.txb_skip_context;
  model_info.dc_sign_context = request.dc_sign_context;
  model_info.scalar_levels_scan = scalar_scan;
  model_info.scalar_eob = scalar_eob;
  model_info.quant_step = quant_step;
  model_info.lambda = request.lambda;

  rdoq::av2::Model default_model;
  rdoq::RateModel rate_model;
  bool custom_model = false;
  if (!build_rate_model(request, model_info, &default_model, &rate_model,
                        &custom_model, error)) {
    return false;
  }

  std::vector<std::int32_t> levels_scan;
  std::int32_t eob = 0;
  double rdoq_distortion = 0.0;
  double rdoq_rate = 0.0;
  double rdoq_cost = 0.0;
  std::int32_t max_level_searched = 0;
  bool truncated = false;

  if (run_trellis) {
    std::vector<rdoq::Coefficient> coefficients(
        static_cast<std::size_t>(area));
    for (int c = 0; c < area; ++c) {
      coefficients[static_cast<std::size_t>(c)].original =
          coeff_scan[static_cast<std::size_t>(c)];
      coefficients[static_cast<std::size_t>(c)].scalar_level =
          scalar_scan[static_cast<std::size_t>(c)];
      coefficients[static_cast<std::size_t>(c)].level_context = c;
    }
    rdoq::Config config;
    config.lambda = request.lambda;
    config.quant_step = quant_step;
    config.prev_level_cap = request.prev_level_cap;
    config.level_cap = request.level_cap;
    config.allow_zero_to_one = request.allow_zero_to_one;
    config.max_coefficients = 4096;
    rdoq::RdoqResult result;
    std::string rdoq_error;
    const bool ok = rdoq::rdoq_quantize(coefficients, config, rate_model,
                                        result, &rdoq_error);
    if (custom_model && request.hooks != nullptr &&
        request.hooks->release != nullptr) {
      request.hooks->release(request.hooks->user, &rate_model);
    }
    if (!ok) {
      if (error != nullptr) {
        *error = rdoq_error.empty() ? "RDOQ failed" : rdoq_error;
      }
      return false;
    }
    levels_scan = std::move(result.levels);
    eob = result.eob;
    rdoq_distortion = result.distortion;
    rdoq_rate = result.rate_bits;
    rdoq_cost = result.cost;
    max_level_searched = result.max_level_searched;
    truncated = result.level_range_truncated;
  } else {
    levels_scan = scalar_scan;
    eob = scalar_eob;
    rdoq_distortion =
        coefficient_sse_scan(coeff_scan, levels_scan, quant_step);
    rdoq_rate = scalar_rate_bits(rate_model, levels_scan, eob, area,
                                 request.prev_level_cap);
    rdoq_cost = rdoq_distortion + request.lambda * rdoq_rate;
    if (custom_model && request.hooks != nullptr &&
        request.hooks->release != nullptr) {
      request.hooks->release(request.hooks->user, &rate_model);
    }
  }

  if (static_cast<int>(levels_scan.size()) != area) {
    if (error != nullptr) *error = "RDOQ returned the wrong level count";
    return false;
  }

  eval->symbol = classify_tx_type(query, candidate.type, eob);
  eval->codable = eval->symbol.valid &&
                  txtype::candidate_codable(candidate.type, eob);
  eval->candidate = candidate;

  // Reconstruct even when the candidate is not codable: the caller may want
  // the metrics, and it keeps the hot path branch-light.
  std::vector<std::int32_t> raster(static_cast<std::size_t>(area), 0);
  for (int c = 0; c < eob && c < area; ++c) {
    raster[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])] =
        levels_scan[static_cast<std::size_t>(c)];
  }
  std::vector<std::int32_t> dequant(static_cast<std::size_t>(area));
  eval->recon_coded.assign(static_cast<std::size_t>(coded_area), 0);
  txv::inverse_transform(raster.data(), params, dequant.data(),
                         eval->recon_coded.data());

  std::int32_t nonzero = 0;
  for (int c = 0; c < area; ++c) {
    if (levels_scan[static_cast<std::size_t>(c)] != 0) ++nonzero;
  }
  std::int64_t sse = 0;
  for (int i = 0; i < coded_area; ++i) {
    const std::int64_t d = static_cast<std::int64_t>(eval->recon_coded[
                               static_cast<std::size_t>(i)]) -
                           request.residual_coded[i];
    sse += d * d;
  }

  double type_bits = 0.0;
  if (eval->symbol.valid &&
      eval->symbol.kind != TxTypeSyntaxKind::kNone) {
    if (request.hooks != nullptr && request.hooks->type_bits != nullptr) {
      type_bits = request.hooks->type_bits(request.hooks->user, query,
                                           candidate);
    } else {
      type_bits = eval->symbol.bits;
    }
  }

  eval->eob = eob;
  eval->nonzero = nonzero;
  eval->sse = static_cast<double>(sse);
  eval->coeff_rate_bits = rdoq_rate;
  eval->type_bits = type_bits;
  eval->rate_bits = rdoq_rate + type_bits;
  eval->cost = eval->sse + request.lambda * eval->rate_bits;
  eval->rdoq_distortion = rdoq_distortion;
  eval->rdoq_rate_bits = rdoq_rate;
  eval->rdoq_cost = rdoq_cost;
  eval->max_level_searched = max_level_searched;
  eval->level_range_truncated = truncated;
  eval->levels_scan = std::move(levels_scan);
  eval->ok = true;
  return true;
}

// Validated, symbol-ordered candidate list.  `subset` may be empty.
bool collect_candidates(const Request& request, const TxTypeQuery& query,
                        std::vector<TxTypeCandidate>* candidates,
                        std::string* error) {
  TxTypeSet set;
  if (!txtype::intra_tx_candidates(query, &set)) {
    if (error != nullptr) *error = "unsupported TX size, mode or TX set";
    return false;
  }
  candidates->clear();
  if (request.candidate_types.empty()) {
    candidates->assign(set.candidates, set.candidates + set.count);
  } else {
    for (const TxType type : request.candidate_types) {
      bool found = false;
      for (int i = 0; i < set.count; ++i) {
        if (set.candidates[i].type == type) {
          candidates->push_back(set.candidates[i]);
          found = true;
          break;
        }
      }
      if (!found) {
        if (error != nullptr) {
          *error = "candidate TX type is not legal for this block";
        }
        return false;
      }
    }
  }
  if (!request.enable_idtx_intra) {
    std::vector<TxTypeCandidate> filtered;
    filtered.reserve(candidates->size());
    for (const TxTypeCandidate& candidate : *candidates) {
      txv::Tx1dType row = txv::Tx1dType::kDct;
      txv::Tx1dType col = txv::Tx1dType::kDct;
      txv::tx_1d_types(candidate.type, &row, &col);
      if (row == txv::Tx1dType::kIdt || col == txv::Tx1dType::kIdt) continue;
      filtered.push_back(candidate);
    }
    *candidates = std::move(filtered);
  }
  if (candidates->empty()) {
    if (error != nullptr) *error = "empty candidate list";
    return false;
  }
  return true;
}

bool prepare(const Request& request, TxTypeQuery* query,
             std::vector<TxTypeCandidate>* candidates, std::string* error) {
  if (request.residual_coded == nullptr) {
    if (error != nullptr) *error = "null residual";
    return false;
  }
  if (!txv::tx_size_supported(request.coded_w, request.coded_h)) {
    if (error != nullptr) *error = "unsupported coded TX size";
    return false;
  }
  if (!finite_non_negative(request.lambda)) {
    if (error != nullptr) *error = "lambda must be finite and >= 0";
    return false;
  }
  if (request.prev_level_cap < 0 || request.prev_level_cap > 63) {
    if (error != nullptr) *error = "prev_level_cap must be in [0, 63]";
    return false;
  }
  if (request.level_cap < 0) {
    if (error != nullptr) *error = "level_cap must be >= 0";
    return false;
  }
  if (request.reduced_tx_set < 0 || request.reduced_tx_set > 3) {
    if (error != nullptr) *error = "reduced_tx_set must be in [0, 3]";
    return false;
  }
  if (request.plane < 0 || request.plane > 2) {
    if (error != nullptr) *error = "plane must be 0, 1 or 2";
    return false;
  }

  const int mapped_mode = txtype::map_intra_mode(
      request.intra_mode, request.angle_delta, request.coded_w,
      request.coded_h);
  if (mapped_mode < 0) {
    if (error != nullptr) *error = "invalid intra mode, angle delta or size";
    return false;
  }

  query->coded_w = request.coded_w;
  query->coded_h = request.coded_h;
  query->mapped_mode = mapped_mode;
  query->reduced_tx_set = request.reduced_tx_set;
  query->dct_only = request.dct_only;
  return collect_candidates(request, *query, candidates, error);
}

void fill_decision(const CandidateEval& eval, Decision* decision) {
  decision->valid = eval.ok;
  decision->codable = eval.codable;
  decision->type = eval.candidate.type;
  decision->params = eval.candidate.params;
  decision->symbol = eval.symbol;
  decision->eob = eval.eob;
  decision->nonzero = eval.nonzero;
  decision->sse = eval.sse;
  decision->coeff_rate_bits = eval.coeff_rate_bits;
  decision->type_bits = eval.type_bits;
  decision->rate_bits = eval.rate_bits;
  decision->cost = eval.cost;
  decision->rdoq_distortion = eval.rdoq_distortion;
  decision->rdoq_rate_bits = eval.rdoq_rate_bits;
  decision->rdoq_cost = eval.rdoq_cost;
  decision->max_level_searched = eval.max_level_searched;
  decision->level_range_truncated = eval.level_range_truncated;
  decision->levels = eval.levels_scan;
  decision->recon_coded = eval.recon_coded;
}

bool decide(const Request& request, bool run_trellis, Decision* decision,
            std::string* error) {
  if (decision == nullptr) {
    if (error != nullptr) *error = "null decision output";
    return false;
  }
  *decision = Decision{};

  TxTypeQuery query;
  std::vector<TxTypeCandidate> candidates;
  if (!prepare(request, &query, &candidates, error)) {
    return false;
  }

  CandidateEval best;
  int considered = 0;
  for (const TxTypeCandidate& candidate : candidates) {
    CandidateEval eval;
    if (!evaluate_candidate(request, query, candidate, run_trellis, &eval,
                            error)) {
      return false;
    }
    ++considered;
    if (!eval.codable) continue;
    bool better = !best.ok;
    if (!better) {
      if (eval.cost != best.cost) {
        better = eval.cost < best.cost;
      } else if (eval.sse != best.sse) {
        better = eval.sse < best.sse;
      } else {
        better = static_cast<int>(eval.candidate.type) <
                 static_cast<int>(best.candidate.type);
      }
    }
    if (better) best = std::move(eval);
  }

  if (!best.ok) {
    if (error != nullptr) *error = "no codable TX-type candidate";
    return false;
  }

  fill_decision(best, decision);
  decision->valid = true;
  decision->codable = true;
  decision->candidates_considered = considered;
  return true;
}

}  // namespace

bool evaluate_candidates(const Request& request,
                         std::vector<Decision>* decisions,
                         std::string* error) {
  if (decisions == nullptr) {
    if (error != nullptr) *error = "null candidate output";
    return false;
  }
  decisions->clear();
  TxTypeQuery query;
  std::vector<TxTypeCandidate> candidates;
  if (!prepare(request, &query, &candidates, error)) {
    return false;
  }
  decisions->resize(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    CandidateEval eval;
    if (!evaluate_candidate(request, query, candidates[i],
                            /*run_trellis=*/true, &eval, error)) {
      decisions->clear();
      return false;
    }
    fill_decision(eval, &(*decisions)[i]);
    (*decisions)[i].candidates_considered = 1;
  }
  return true;
}

TxTypeSymbol classify_tx_type(const txtype::TxTypeQuery& query,
                              txtype::TxType type, int eob) {
  TxTypeSymbol out;
  if (query.dct_only || query.reduced_tx_set == 2 || eob <= 1) {
    out.kind = TxTypeSyntaxKind::kNone;
    out.valid = type == TxType::kDctDct;
    out.is_long_side_dct = false;
    out.has_long_flag = false;
    out.symbol = -1;
    out.cdf_symbols = 0;
    out.bits = 0.0;
    return out;
  }

  TxSet set = TxSet::kDctOnly;
  if (!txtype::intra_tx_set(query, &set)) {
    return out;
  }

  TxTypeSyntaxKind kind = TxTypeSyntaxKind::kNone;
  int cdf_symbols = 0;
  bool has_long_flag = false;
  bool implied_long = false;
  switch (set) {
    case TxSet::kIntra1:
      kind = TxTypeSyntaxKind::kIntraSet1;
      cdf_symbols = 7;
      break;
    case TxSet::kIntra2:
      kind = TxTypeSyntaxKind::kIntraSet2;
      cdf_symbols = 2;
      break;
    case TxSet::kWide32:
    case TxSet::kHigh32:
      kind = TxTypeSyntaxKind::kLong32;
      cdf_symbols = 4;
      has_long_flag = true;
      break;
    case TxSet::kWide64:
    case TxSet::kHigh64:
      kind = TxTypeSyntaxKind::kLong64;
      cdf_symbols = 4;
      implied_long = true;
      break;
    case TxSet::kDctOnly:
      out.kind = TxTypeSyntaxKind::kNone;
      out.valid = type == TxType::kDctDct;
      return out;
    case TxSet::kInter1:
    case TxSet::kInter2:
    case TxSet::kDctIdtx:
    case TxSet::kDctIdtxIddct:
      return out;
  }

  int symbol = -1;
  bool is_long_side_dct = false;
  if (!txtype::tx_type_symbol(query, type, &symbol, &is_long_side_dct)) {
    return out;
  }

  const auto params_opt =
      txv::make_tx_params(query.coded_w, query.coded_h, type);
  if (!params_opt.has_value()) return out;

  TxTypeCandidate candidate;
  candidate.type = type;
  candidate.symbol = static_cast<std::int8_t>(symbol);
  candidate.is_long_side_dct = is_long_side_dct;
  candidate.codes_long_flag = has_long_flag;
  candidate.params = *params_opt;

  out.kind = kind;
  out.symbol = static_cast<std::int8_t>(symbol);
  out.has_long_flag = has_long_flag;
  out.is_long_side_dct = implied_long ? true : is_long_side_dct;
  out.cdf_symbols = static_cast<std::int8_t>(cdf_symbols);
  out.bits = txtype::static_type_bits(query, candidate);
  out.valid = true;
  return out;
}

bool quantize_block(const Request& request, Decision* decision,
                    std::string* error) {
  return decide(request, /*run_trellis=*/true, decision, error);
}

bool quantize_block_scalar(const Request& request, Decision* decision,
                           std::string* error) {
  return decide(request, /*run_trellis=*/false, decision, error);
}

double default_lambda(const txv::TxParams& params, double sse_scale) {
  if (!std::isfinite(params.quant_factor) || params.quant_factor <= 0.0) {
    return 0.0;
  }
  const double quant_step = 1.0 / params.quant_factor;
  return rdo::lambda_sse_from_quant_step(quant_step, sse_scale);
}

}  // namespace av2enc::still::ts
