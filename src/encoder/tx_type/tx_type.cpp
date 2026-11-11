// AV2 intra transform-type selection — CPU reference implementation.
// See tx_type.h for the numeric/spec contract.
#include "encoder/tx_type/tx_type.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "encoder/intra_modes/intra_modes.h"
#include "encoder/tx_type/tx_type_tables.h"
#include "encoder/tx_variants/tx_tables.h"

namespace av2enc::txtype {
namespace {

constexpr int kMaxAdjTxElems = txv::kMaxAdjTxElems;
constexpr int kMaxCodedTxElems = txv::kMaxCodedTxDim * txv::kMaxCodedTxDim;

// §9.3 CDF layout: entries [0 .. nsym-2) are cumulative probabilities scaled to
// 32768, the next entry is the adaptation rate and the last is the counter.  The
// final symbol occupies the remainder.
double symbol_bits(const std::uint16_t* cdf, int nsym, int symbol) {
  if (symbol < 0 || symbol >= nsym) {
    return 0.0;
  }
  const int lo = symbol == 0 ? 0 : static_cast<int>(cdf[symbol - 1]);
  const int hi = symbol == nsym - 1 ? 32768 : static_cast<int>(cdf[symbol]);
  const int range = hi > lo ? hi - lo : 1;
  return -std::log2(static_cast<double>(range) / 32768.0);
}

double long_side_flag_bits(bool is_long_side_dct) {
  const std::uint16_t* cdf = kIsLongSideDctCdf[0];  // is_inter = 0
  return symbol_bits(cdf, 2, is_long_side_dct ? 1 : 0);
}

bool add_candidate(const TxTypeQuery& query, TxType type, int symbol,
                   bool is_long_side_dct, bool codes_long_flag,
                   TxTypeSet* out) {
  const auto params = txv::make_tx_params(query.coded_w, query.coded_h, type);
  if (!params.has_value() || out->count >= kMaxTxTypeCandidates) {
    return false;
  }
  TxTypeCandidate& c = out->candidates[out->count++];
  c.type = type;
  c.symbol = static_cast<std::int8_t>(symbol);
  c.is_long_side_dct = is_long_side_dct;
  c.codes_long_flag = codes_long_flag;
  c.params = *params;
  return true;
}

}  // namespace

const char* tx_set_name(TxSet set) {
  switch (set) {
    case TxSet::kDctOnly:
      return "TX_SET_DCTONLY";
    case TxSet::kWide64:
      return "TX_SET_WIDE_64";
    case TxSet::kHigh64:
      return "TX_SET_HIGH_64";
    case TxSet::kWide32:
      return "TX_SET_WIDE_32";
    case TxSet::kHigh32:
      return "TX_SET_HIGH_32";
    case TxSet::kIntra1:
      return "TX_SET_INTRA_1";
    case TxSet::kIntra2:
      return "TX_SET_INTRA_2";
    case TxSet::kInter1:
      return "TX_SET_INTER_1";
    case TxSet::kInter2:
      return "TX_SET_INTER_2";
    case TxSet::kDctIdtx:
      return "TX_SET_DCT_IDTX";
    case TxSet::kDctIdtxIddct:
      return "TX_SET_DCT_IDTX_IDDCT";
  }
  return "?";
}

int map_intra_mode(int mode, int angle_delta, int coded_w, int coded_h) {
  if (mode < 0 || mode >= intra::kNumModes) {
    return -1;
  }
  if (angle_delta < -intra::kMaxAngleDelta ||
      angle_delta > intra::kMaxAngleDelta) {
    return -1;
  }
  if (!txv::tx_size_supported(coded_w, coded_h)) {
    return -1;
  }
  const intra::MappedAngle mapped = intra::map_angle(
      static_cast<intra::Mode>(mode), angle_delta, coded_w, coded_h);
  return static_cast<int>(mapped.mode);
}

bool intra_tx_set(const TxTypeQuery& query, TxSet* set) {
  if (set == nullptr) {
    return false;
  }
  const int idx = txv::tx_size_index(query.coded_w, query.coded_h);
  if (idx < 0 || query.reduced_tx_set < 0 || query.reduced_tx_set > 3) {
    return false;
  }
  const int sqr = kTxSizeSqr[idx];
  const int sqr_up = kTxSizeSqrUp[idx];
  // §5.20.8.3, with TX_32X32 = 3 and TX_64X64 = 4.
  if (sqr_up > 3) {
    if (sqr >= 3) {
      *set = TxSet::kDctOnly;
    } else {
      *set = query.coded_w > query.coded_h ? TxSet::kWide64 : TxSet::kHigh64;
    }
    return true;
  }
  if (sqr_up == 3 && sqr != 3) {
    *set = query.coded_w > query.coded_h ? TxSet::kWide32 : TxSet::kHigh32;
    return true;
  }
  if (sqr_up == 3) {  // !is_inter
    *set = TxSet::kDctOnly;
    return true;
  }
  // reducedTxSet = plane == 0 ? reduced_tx_set : enable_chroma_dctonly.
  const int reduced = query.dct_only ? 1 : query.reduced_tx_set;
  if (reduced == 1) {
    *set = TxSet::kIntra2;
  } else if (reduced == 2) {
    *set = TxSet::kDctIdtx;
  } else if (reduced == 3) {
    *set = TxSet::kIntra2;
  } else {
    *set = TxSet::kIntra1;
  }
  return true;
}

bool intra_tx_candidates(const TxTypeQuery& query, TxTypeSet* out) {
  if (out == nullptr) {
    return false;
  }
  if (query.mapped_mode < 0 || query.mapped_mode >= kNumIntraModes) {
    return false;
  }
  if (!intra_tx_set(query, &out->set)) {
    return false;
  }
  out->count = 0;
  const int idx = txv::tx_size_index(query.coded_w, query.coded_h);

  // transform_type(): these two conditions bypass the type syntax entirely.
  if (query.dct_only || query.reduced_tx_set == 2) {
    return add_candidate(query, TxType::kDctDct, -1, false, false, out);
  }

  switch (out->set) {
    case TxSet::kDctOnly:
      return add_candidate(query, TxType::kDctDct, -1, false, false, out);

    case TxSet::kWide64:
    case TxSet::kHigh64: {
      const int orientation = out->set == TxSet::kWide64 ? 0 : 1;
      for (int symbol = 0; symbol < 4; ++symbol) {
        const auto type = static_cast<TxType>(kTxTypeInvLong[1][orientation][symbol]);
        if (!add_candidate(query, type, symbol, true, false, out)) {
          return false;
        }
      }
      return true;
    }

    case TxSet::kWide32:
    case TxSet::kHigh32: {
      const int orientation = out->set == TxSet::kWide32 ? 0 : 1;
      // is_long_side_dct = 0 branch, then the signalled flag = 1 branch.
      for (int flag = 0; flag < 2; ++flag) {
        for (int symbol = 0; symbol < 4; ++symbol) {
          const auto type =
              static_cast<TxType>(kTxTypeInvLong[flag][orientation][symbol]);
          if (!add_candidate(query, type, symbol, flag == 1, true, out)) {
            return false;
          }
        }
      }
      return true;
    }

    case TxSet::kIntra1:
    case TxSet::kIntra2: {
      const int n = out->set == TxSet::kIntra1 ? 7 : 2;
      const int size_class = kSizeClass[idx];
      for (int symbol = 0; symbol < n; ++symbol) {
        const auto type = static_cast<TxType>(
            kMdIdxToType[size_class][query.mapped_mode][symbol]);
        if (!add_candidate(query, type, symbol, false, false, out)) {
          return false;
        }
      }
      return true;
    }

    case TxSet::kInter1:
    case TxSet::kInter2:
    case TxSet::kDctIdtx:
    case TxSet::kDctIdtxIddct:
      break;
  }
  // Inter-only sets are unreachable for an intra block; the DCT_IDTX set is
  // forced to DCT_DCT by the transform_type() override handled above.
  return add_candidate(query, TxType::kDctDct, -1, false, false, out);
}

int intra_tx_types_for_size(int coded_w, int coded_h, int reduced_tx_set,
                            bool dct_only, TxType* out, int cap) {
  if (out == nullptr || cap <= 0) {
    return -1;
  }
  std::uint8_t seen[16] = {};
  int count = 0;
  for (int mode = 0; mode < kNumIntraModes; ++mode) {
    TxTypeQuery query;
    query.coded_w = coded_w;
    query.coded_h = coded_h;
    query.mapped_mode = mode;
    query.reduced_tx_set = reduced_tx_set;
    query.dct_only = dct_only;
    TxTypeSet set;
    if (!intra_tx_candidates(query, &set)) {
      return -1;
    }
    for (int i = 0; i < set.count; ++i) {
      const int value = static_cast<int>(set.candidates[i].type);
      if (seen[value] == 0) {
        if (count >= cap) {
          return -1;
        }
        seen[value] = 1;
        out[count++] = set.candidates[i].type;
      }
    }
  }
  std::sort(out, out + count, [](TxType a, TxType b) {
    return static_cast<int>(a) < static_cast<int>(b);
  });
  return count;
}

bool tx_type_symbol(const TxTypeQuery& query, TxType type, int* symbol,
                    bool* is_long_side_dct) {
  if (symbol == nullptr || is_long_side_dct == nullptr) {
    return false;
  }
  TxTypeSet set;
  if (!intra_tx_candidates(query, &set)) {
    return false;
  }
  for (int i = 0; i < set.count; ++i) {
    if (set.candidates[i].type == type) {
      *symbol = set.candidates[i].symbol;
      *is_long_side_dct = set.candidates[i].is_long_side_dct;
      return true;
    }
  }
  return false;
}

double static_type_bits(TxSet set, int tx_size_index,
                        const TxTypeCandidate& candidate) {
  if (tx_size_index < 0 || tx_size_index >= 25) {
    return 0.0;
  }
  const int symbol = candidate.symbol;
  const int sqr = kTxSizeSqr[tx_size_index];
  switch (set) {
    case TxSet::kIntra1:
      return symbol_bits(kIntraTxTypeSet1Cdf[sqr], 7, symbol);
    case TxSet::kIntra2:
      return symbol_bits(kIntraTxTypeSet2Cdf[sqr], 2, symbol);
    case TxSet::kWide64:
    case TxSet::kHigh64:
      return symbol_bits(kIntraTxTypeLongCdf[sqr], 4, symbol);
    case TxSet::kWide32:
    case TxSet::kHigh32:
      return symbol_bits(kIntraTxTypeLongCdf[sqr], 4, symbol) +
             long_side_flag_bits(candidate.is_long_side_dct);
    case TxSet::kDctOnly:
    case TxSet::kInter1:
    case TxSet::kInter2:
    case TxSet::kDctIdtx:
    case TxSet::kDctIdtxIddct:
      return 0.0;
  }
  return 0.0;
}

double static_type_bits(const TxTypeQuery& query,
                        const TxTypeCandidate& candidate) {
  TxSet set;
  if (!intra_tx_set(query, &set)) {
    return 0.0;
  }
  if (query.dct_only || query.reduced_tx_set == 2) {
    return 0.0;  // no type syntax
  }
  const int idx = txv::tx_size_index(query.coded_w, query.coded_h);
  return static_type_bits(set, idx, candidate);
}

double default_rate_bits(const TxTypeRateInputs& inputs, void* /*user*/) {
  return inputs.type_bits + static_cast<double>(inputs.nonzero);
}

double tx_type_rate_bits(const TxTypeRateInputs& inputs,
                         const TxTypeCostConfig& config) {
  if (config.rate_fn != nullptr) {
    return config.rate_fn(inputs, config.rate_user);
  }
  double bits =
      config.coefficient_bits * static_cast<double>(inputs.nonzero);
  if (config.use_static_type_cost) {
    bits += inputs.type_bits;
  }
  return bits;
}

void diagonal_scan(int w, int h, std::uint16_t* scan) {
  int x = 0;
  int y = 0;
  for (int c = 0; c < w * h; ++c) {
    scan[c] = static_cast<std::uint16_t>(y * w + x);
    ++x;
    --y;
    if (y < 0 || x >= w) {
      ++x;
      const int s = std::min(x, h - 1 - y);
      x -= s;
      y += s;
    }
  }
}

int eob_from_levels(const std::int32_t* levels, int w, int h) {
  std::uint16_t scan[kMaxAdjTxElems];
  diagonal_scan(w, h, scan);
  for (int c = w * h - 1; c >= 0; --c) {
    if (levels[scan[c]] != 0) {
      return c + 1;
    }
  }
  return 0;
}

bool evaluate_intra_tx_type(const std::int16_t* residual_coded,
                            const TxTypeQuery& query,
                            const TxTypeCandidate& candidate,
                            const TxTypeCostConfig& config, TxTypeEval* out,
                            std::int32_t* levels_out, std::int16_t* recon_out) {
  if (residual_coded == nullptr || out == nullptr) {
    return false;
  }
  const TxParams& p = candidate.params;
  if (p.coded_w != query.coded_w || p.coded_h != query.coded_h) {
    return false;
  }
  if (!txv::tx_size_supported(p.coded_w, p.coded_h)) {
    return false;
  }
  const int adj_n = p.w * p.h;
  const int coded_n = p.coded_w * p.coded_h;

  std::int16_t residual_adj[kMaxAdjTxElems];
  std::int16_t recon_coded[kMaxCodedTxElems];
  std::int32_t levels[kMaxAdjTxElems];
  std::int32_t dequant[kMaxAdjTxElems];
  double coeff[kMaxAdjTxElems];

  txv::downsample_residual(residual_coded, p, residual_adj);
  txv::forward_transform(residual_adj, p, coeff, levels);
  txv::inverse_transform(levels, p, dequant, recon_coded);

  std::int32_t nonzero = 0;
  std::int64_t sum_abs = 0;
  for (int i = 0; i < adj_n; ++i) {
    const std::int32_t level = levels[i];
    if (level != 0) {
      ++nonzero;
      sum_abs += level < 0 ? -static_cast<std::int64_t>(level) : level;
    }
  }
  std::int64_t error = 0;
  for (int i = 0; i < coded_n; ++i) {
    const std::int64_t d =
        static_cast<std::int64_t>(recon_coded[i]) - residual_coded[i];
    error += d * d;
  }

  out->candidate = candidate;
  out->eob = eob_from_levels(levels, p.w, p.h);
  out->nonzero = nonzero;
  out->sum_abs = sum_abs;
  out->sse = static_cast<double>(error);
  out->codable = candidate_codable(candidate.type, out->eob);
  out->type_bits = static_type_bits(query, candidate);

  TxTypeRateInputs inputs;
  inputs.query = query;
  inputs.candidate = candidate;
  inputs.eob = out->eob;
  inputs.nonzero = out->nonzero;
  inputs.sum_abs = out->sum_abs;
  inputs.sse = out->sse;
  inputs.type_bits = out->type_bits;
  out->rate_bits = tx_type_rate_bits(inputs, config);
  out->cost = out->sse + config.lambda * out->rate_bits;

  if (levels_out != nullptr) {
    std::memcpy(levels_out, levels,
                static_cast<std::size_t>(adj_n) * sizeof(std::int32_t));
  }
  if (recon_out != nullptr) {
    std::memcpy(recon_out, recon_coded,
                static_cast<std::size_t>(coded_n) * sizeof(std::int16_t));
  }
  return true;
}

int evaluate_intra_tx_candidates(const std::int16_t* residual_coded,
                                 const TxTypeQuery& query,
                                 const TxTypeCostConfig& config, TxTypeEval* out,
                                 int cap) {
  if (residual_coded == nullptr || out == nullptr || cap <= 0) {
    return -1;
  }
  TxTypeSet set;
  if (!intra_tx_candidates(query, &set)) {
    return -1;
  }
  if (set.count > cap) {
    return -1;
  }
  for (int i = 0; i < set.count; ++i) {
    if (!evaluate_intra_tx_type(residual_coded, query, set.candidates[i],
                                config, &out[i])) {
      return -1;
    }
  }
  return set.count;
}

bool choose_intra_tx_type(const std::int16_t* residual_coded,
                          const TxTypeQuery& query,
                          const TxTypeCostConfig& config, TxTypeChoice* out,
                          std::int32_t* levels_out, std::int16_t* recon_out) {
  if (residual_coded == nullptr || out == nullptr) {
    return false;
  }
  TxTypeSet set;
  if (!intra_tx_candidates(query, &set)) {
    return false;
  }
  return choose_tx_type_from_candidates(residual_coded, query,
                                        set.candidates, set.count, config, out,
                                        levels_out, recon_out);
}

bool choose_tx_type_from_candidates(const std::int16_t* residual_coded,
                                    const TxTypeQuery& query,
                                    const TxTypeCandidate* candidates, int count,
                                    const TxTypeCostConfig& config,
                                    TxTypeChoice* out,
                                    std::int32_t* levels_out,
                                    std::int16_t* recon_out) {
  if (residual_coded == nullptr || candidates == nullptr || out == nullptr ||
      count <= 0) {
    return false;
  }
  bool have = false;
  TxTypeEval best;
  int best_index = -1;
  for (int i = 0; i < count; ++i) {
    TxTypeEval eval;
    if (!evaluate_intra_tx_type(residual_coded, query, candidates[i], config,
                                &eval)) {
      return false;
    }
    if (!eval.codable) {
      continue;
    }
    const bool better =
        !have || eval.cost < best.cost ||
        (eval.cost == best.cost &&
         (eval.sse < best.sse ||
          (eval.sse == best.sse &&
           static_cast<int>(eval.candidate.type) <
               static_cast<int>(best.candidate.type))));
    if (better) {
      have = true;
      best = eval;
      best_index = i;
    }
  }
  if (!have) {
    return false;
  }
  out->valid = true;
  out->candidate = best.candidate;
  out->eval = best;
  out->candidate_index = best_index;
  out->candidates_considered = count;
  if (levels_out != nullptr || recon_out != nullptr) {
    TxTypeEval ignored;
    if (!evaluate_intra_tx_type(residual_coded, query, best.candidate, config,
                                &ignored, levels_out, recon_out)) {
      return false;
    }
  }
  return true;
}

}  // namespace av2enc::txtype
