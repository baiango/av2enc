#include "pipeline/video/video_tx_search.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "codec/rate/rc/rc_tables.h"
#include "codec/transform/inter_residual/inter_tx_type.h"
#include "codec/transform/inter_residual/video_residual_cdf.h"
#include "codec/transform/tx_partition/tx_partition.h"
#include "codec/transform/tx_partition/tx_scan.h"
#include "codec/transform/tx_type/tx_type.h"
#include "codec/transform/variants/tx_variants.h"

namespace av2enc::video::txtsearch {
namespace {

namespace txv = av2enc::txv;
namespace txtype = av2enc::txtype;
namespace txpart = av2enc::video::txpartition;
namespace residual = av2enc::video::residual;

// Partition bit indices, identical to VideoTxPartitionBit.
constexpr std::int32_t kPartBitNone = 1 << 0;
constexpr std::int32_t kPartBitSplit = 1 << 1;
constexpr std::int32_t kPartBitHorz = 1 << 2;
constexpr std::int32_t kPartBitVert = 1 << 3;
constexpr std::int32_t kPartBitHorz4 = 1 << 4;
constexpr std::int32_t kPartBitVert4 = 1 << 5;

// §9.3 CDF layout: entries [0 .. nsym-2) are cumulative probabilities scaled to
// 32768; the next entry is the adaptation rate and the last the counter.
double cdf_symbol_bits(const std::uint16_t* cdf, int nsym, int symbol) {
  if (symbol < 0 || symbol >= nsym) return 0.0;
  const int lo = symbol == 0 ? 0 : static_cast<int>(cdf[symbol - 1]);
  const int hi = symbol == nsym - 1 ? 32768 : static_cast<int>(cdf[symbol]);
  const int range = hi > lo ? hi - lo : 1;
  return -std::log2(static_cast<double>(range) / 32768.0);
}

// §8.3.2 `inter_tx_type` context from the decoded EOB.  Mirrors
// `inter_ctx_dev()` in `src/cuda/pipeline/video/tx/video_tx.cu`.
std::int32_t inter_ctx(int eob, int coded_w, int coded_h) {
  if (eob <= 0) return 0;
  int lw = 0;
  while ((1 << lw) < coded_w) ++lw;
  const int bwl = lw < 5 ? lw : 5;
  const int eoby = (eob - 1) >> bwl;
  const int eobx = (eob - 1) - (eoby << bwl);
  const int diag = eobx + eoby;
  if (diag < 2) return 1;
  const int limit = std::min(coded_w, 32) + std::min(coded_h, 32) - 4;
  if (diag > limit) return 2;
  return 0;
}

int mi_size_for_px(int w, int h) {
  for (int mi = 0; mi < txpart::kBlockCount; ++mi) {
    if (txpart::block_width_px(mi) == w && txpart::block_height_px(mi) == h) {
      return mi;
    }
  }
  return -1;
}

int partition_bit(int partition) {
  switch (static_cast<txpart::TxPartition>(partition)) {
    case txpart::TxPartition::kNone:
      return kPartBitNone;
    case txpart::TxPartition::kSplit:
      return kPartBitSplit;
    case txpart::TxPartition::kHorz:
      return kPartBitHorz;
    case txpart::TxPartition::kVert:
      return kPartBitVert;
    case txpart::TxPartition::kHorz4:
      return kPartBitHorz4;
    case txpart::TxPartition::kVert4:
      return kPartBitVert4;
    default:
      return 0;
  }
}

// Estimated §5.20.6.3 signalling cost from the §9.3 default CDFs.
double partition_bits_for(int mi_size, bool is_inter,
                          bool reduced_tx_part_set, int partition,
                          const txpart::LegalSplits& legal) {
  txpart::TxPartitionCdfs cdfs;
  cdfs.reset(reduced_tx_part_set);
  const bool fsc = false;
  std::uint16_t* do_cdf = cdfs.do_partition_cdf(fsc, is_inter, mi_size);
  if (do_cdf == nullptr) return 0.0;
  const bool do_split = partition != 0;
  double bits = cdf_symbol_bits(do_cdf, 2, do_split ? 1 : 0);
  if (!do_split) return bits;
  if (legal.has_partition_type) {
    std::uint16_t* type_cdf =
        cdfs.partition_type_cdf(fsc, is_inter, mi_size);
    if (type_cdf == nullptr) return bits;
    int symbol = 0;
    if (partition == static_cast<int>(txpart::TxPartition::kSplit)) {
      symbol = 0;
    } else if (partition == static_cast<int>(txpart::TxPartition::kHorz)) {
      symbol = 1;
    } else if (partition == static_cast<int>(txpart::TxPartition::kVert)) {
      symbol = 2;
    } else {
      return bits;
    }
    bits += cdf_symbol_bits(type_cdf, txpart::kTxPartitionTypeNum, symbol);
    return bits;
  }
  if (legal.has_2or3) {
    const bool horizontal = legal.allow_horz;
    const int symbol =
        ((horizontal ? partition == static_cast<int>(txpart::TxPartition::kHorz4)
                     : partition == static_cast<int>(txpart::TxPartition::kVert4)))
            ? 1
            : 0;
    if (!legal.reduced_tx_part_set) {
      std::uint16_t* cdf = cdfs.partition_2or3_cdf(fsc, is_inter, mi_size);
      if (cdf != nullptr) bits += cdf_symbol_bits(cdf, 2, symbol);
    }
  }
  return bits;
}

// ---------------------------------------------------------------------------
// TX-type candidate enumeration (§5.20.8.3) and symbol cost
// ---------------------------------------------------------------------------

struct TypeEntry {
  std::int32_t type = 0;
  double bits[3] = {0.0, 0.0, 0.0};
  std::int32_t codes_type = 1;
};

// Mirrors `inter_type_bits_ctx()` in the device module (including the
// documented bring-up choice of using the type context for INTER_2's offset
// term).
double inter_type_bits_ctx(residual::InterTxSet set, txv::TxSize ts,
                           const residual::InterTxCandidate& c, int ctx) {
  using namespace residual;
  if (set == InterTxSet::kDctOnly) return 0.0;
  const int sqr = std::clamp(tx_size_sqr(ts), 0, 3);
  switch (set) {
    case InterTxSet::kInter1:
      return cdf_symbol_bits(&kInterTxTypeSet1Cdf[ctx][sqr][0], 2,
                             c.symbol) +
             cdf_symbol_bits(c.symbol == 0
                                 ? &kInterTxTypeIndexSet1Cdf[ctx][0]
                                 : &kInterTxTypeOffsetSet1Cdf[ctx][0],
                             8, c.offset_symbol);
    case InterTxSet::kInter2: {
      const int offset_ctx = ctx;
      return cdf_symbol_bits(&kInterTxTypeSet2Cdf[ctx][0], 2, c.symbol) +
             cdf_symbol_bits(c.symbol == 0
                                 ? &kInterTxTypeIndexSet2Cdf[offset_ctx][0]
                                 : &kInterTxTypeOffsetSet2Cdf[offset_ctx][0],
                             c.symbol == 0 ? 8 : 4, c.offset_symbol);
    }
    case InterTxSet::kDctIdtx:
      return cdf_symbol_bits(&kInterTxTypeSet3Cdf[ctx][sqr][0], 2, c.symbol);
    case InterTxSet::kDctIdtxIddct:
      return cdf_symbol_bits(&kInterTxTypeSet4Cdf[ctx][sqr][0], 4, c.symbol);
    case InterTxSet::kWide64:
    case InterTxSet::kHigh64:
    case InterTxSet::kWide32:
    case InterTxSet::kHigh32: {
      double bits =
          cdf_symbol_bits(&kInterTxTypeLongCdf[ctx][sqr][0], 4, c.symbol);
      if (c.codes_long_flag) {
        const int flag = c.is_long_side_dct ? 1 : 0;
        bits += cdf_symbol_bits(&kIsLongSideDctCdf[flag][0], 2, flag);
      }
      return bits;
    }
    case InterTxSet::kDctOnly:
      break;
  }
  return 0.0;
}

// §5.20.8.2 canonical inverse tables (aliases are never emitted).
constexpr std::int32_t kInterInvSet1[16] = {
    9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 4, 5, 3, 6, 7, 8};
constexpr std::int32_t kInterInvSet2[12] = {
    9, 10, 11, 0, 1, 2, 4, 5, 3, 6, 7, 8};
constexpr std::int32_t kInterInvSet3[2] = {9, 0};
constexpr std::int32_t kInterInvSet4[4] = {0, 10, 11, 9};
constexpr std::int32_t kInterInvLong[2][2][4] = {
    {{10, 12, 14, 9}, {11, 13, 15, 9}},
    {{0, 1, 4, 11}, {0, 2, 5, 10}},
};

bool inter_leaf_types(std::int32_t coded_w, std::int32_t coded_h,
                      const Options& options, std::vector<TypeEntry>* out,
                      std::string* error) {
  using residual::InterTxCandidate;
  using residual::InterTxSet;
  const auto ts = static_cast<txv::TxSize>(
      txv::tx_size_index(coded_w, coded_h));
  const InterTxSet set =
      residual::inter_tx_set(ts, options.reduced_tx_set);
  out->clear();
  if (set == InterTxSet::kDctOnly) {
    InterTxCandidate c;
    if (!residual::inter_tx_symbol(ts, options.reduced_tx_set,
                                   options.enable_inter_ddt,
                                   txv::TxType::kDctDct, &c)) {
      if (error != nullptr) *error = "DCT_DCT missing from a DCTONLY set";
      return false;
    }
    TypeEntry e;
    e.type = 0;
    e.codes_type = 0;
    out->push_back(e);
    return true;
  }

  std::vector<std::int32_t> canonical;
  if (set == InterTxSet::kInter1) {
    canonical.assign(kInterInvSet1, kInterInvSet1 + 16);
  } else if (set == InterTxSet::kInter2) {
    canonical.assign(kInterInvSet2, kInterInvSet2 + 12);
  } else if (set == InterTxSet::kDctIdtx) {
    canonical.assign(kInterInvSet3, kInterInvSet3 + 2);
  } else if (set == InterTxSet::kDctIdtxIddct) {
    canonical.assign(kInterInvSet4, kInterInvSet4 + 4);
  } else {
    const int orientation =
        (set == InterTxSet::kWide64 || set == InterTxSet::kWide32) ? 0 : 1;
    const bool codes_flag =
        set == InterTxSet::kWide32 || set == InterTxSet::kHigh32;
    const int long_values = codes_flag ? 2 : 1;
    for (int flag = 0; flag < long_values; ++flag) {
      const int is_long = codes_flag ? flag : 1;
      for (int symbol = 0; symbol < 4; ++symbol) {
        canonical.push_back(kInterInvLong[is_long][orientation][symbol]);
      }
    }
  }

  for (std::int32_t type : canonical) {
    if (!options.include_1d_types && type >= 10) continue;
    if (options.enable_inter_ddt &&
        residual::inter_tx_type_is_ddt(static_cast<txv::TxType>(type))) {
      continue;
    }
    if (!residual::inter_tx_type_in_set(set,
                                        static_cast<txv::TxType>(type))) {
      continue;
    }
    InterTxCandidate c;
    if (!residual::inter_tx_symbol(ts, options.reduced_tx_set,
                                   options.enable_inter_ddt,
                                   static_cast<txv::TxType>(type), &c)) {
      continue;
    }
    TypeEntry e;
    e.type = type;
    e.codes_type = 1;
    for (int ctx = 0; ctx < 3; ++ctx) {
      e.bits[ctx] = inter_type_bits_ctx(set, ts, c, ctx);
    }
    out->push_back(e);
  }
  if (out->empty()) {
    if (error != nullptr) *error = "empty inter TX-type set";
    return false;
  }
  return true;
}

bool intra_leaf_types(std::int32_t coded_w, std::int32_t coded_h,
                      const Options& options, std::vector<TypeEntry>* out,
                      std::string* error) {
  txtype::TxTypeQuery q;
  q.coded_w = coded_w;
  q.coded_h = coded_h;
  q.mapped_mode = options.intra_mapped_mode;
  q.reduced_tx_set = options.reduced_tx_set;
  q.dct_only = options.dct_only;
  txtype::TxTypeSet set;
  if (!txtype::intra_tx_candidates(q, &set)) {
    if (error != nullptr) *error = "intra TX-type candidate enumeration failed";
    return false;
  }
  out->clear();
  for (int i = 0; i < set.count; ++i) {
    const int type = static_cast<int>(set.candidates[i].type);
    if (!options.include_1d_types && type >= 10) continue;
    TypeEntry e;
    e.type = type;
    e.codes_type = set.candidates[i].symbol >= 0 ? 1 : 0;
    const double bits =
        e.codes_type != 0 ? txtype::static_type_bits(q, set.candidates[i])
                          : 0.0;
    e.bits[0] = bits;
    e.bits[1] = bits;
    e.bits[2] = bits;
    out->push_back(e);
  }
  if (out->empty()) {
    if (error != nullptr) *error = "empty intra TX-type set";
    return false;
  }
  return true;
}

bool build_leaf_types(std::int32_t coded_w, std::int32_t coded_h, Mode mode,
                      const Options& options, std::vector<TypeEntry>* out,
                      std::string* error) {
  if (mode == Mode::kIntra) {
    return intra_leaf_types(coded_w, coded_h, options, out, error);
  }
  return inter_leaf_types(coded_w, coded_h, options, out, error);
}

// §7.14.2 get_q(): the same expression as the device module.
int get_q(std::int32_t qindex, std::int32_t delta) {
  if (qindex == 0 && delta <= 0) return rc::kQlookupQ0To255[0];
  const int clamped = std::clamp<std::int32_t>(qindex + delta, 1, 255);
  return rc::kQlookupQ0To255[clamped];
}

struct LeafCandidate {
  bool valid = false;
  std::int32_t tx_size = 0;
  std::int32_t tx_type = 0;
  int eob = 0;
  int nonzero = 0;
  double sse = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  std::vector<std::int32_t> levels;   // scan order
  std::vector<std::int16_t> residual; // coded size
};

// Full forward -> quantise -> dequantise -> inverse path for one candidate,
// with the same coded-domain SSE and rate model as the device kernel.
bool evaluate_candidate(const std::int16_t* residual, int block_w,
                        int block_h, int offset_x, int offset_y, int leaf_w,
                        int leaf_h, std::int32_t tx_size, std::int32_t tx_type,
                        const TypeEntry& type, bool is_inter, int qindex,
                        int delta_dc, int delta_ac, const Options& options,
                        LeafCandidate* out, std::string* error) {
  const auto params_opt = txv::make_tx_params(
      static_cast<txv::TxSize>(tx_size), static_cast<txv::TxType>(tx_type));
  if (!params_opt.has_value()) {
    if (error != nullptr) *error = "unsupported transform size/type";
    return false;
  }
  const txv::TxParams& params = *params_opt;
  if (params.coded_w != leaf_w || params.coded_h != leaf_h) {
    if (error != nullptr) *error = "candidate size does not match leaf";
    return false;
  }
  const int area = params.w * params.h;
  const int coded_n = leaf_w * leaf_h;

  std::vector<std::int16_t> coded(static_cast<std::size_t>(coded_n));
  for (int y = 0; y < leaf_h; ++y) {
    for (int x = 0; x < leaf_w; ++x) {
      coded[static_cast<std::size_t>(y) * leaf_w + x] =
          residual[static_cast<std::size_t>(offset_y + y) * block_w + offset_x +
                   x];
    }
  }
  std::vector<std::int16_t> adj(static_cast<std::size_t>(area));
  txv::downsample_residual(coded.data(), params, adj.data());
  std::vector<double> coeff(static_cast<std::size_t>(area));
  txv::forward_block(adj.data(), params, coeff.data());

  const int dcq = get_q(qindex, delta_dc);
  const int acq = get_q(qindex, delta_ac);
  const double scaled = params.quant_factor * 528.0;
  std::vector<std::int32_t> levels(static_cast<std::size_t>(area));
  std::vector<std::int32_t> dequant(static_cast<std::size_t>(area));
  for (int i = 0; i < area; ++i) {
    const int q = i == 0 ? dcq : acq;
    const double factor = scaled / static_cast<double>(q);
    const std::int32_t level = static_cast<std::int32_t>(
        std::lround(coeff[static_cast<std::size_t>(i)] * factor));
    levels[static_cast<std::size_t>(i)] = level;
    const std::int32_t aq = level < 0 ? -level : level;
    const std::int32_t high = (aq * q) & 0xFFFFFF;
    const std::int32_t dq =
        txv::round2(high, txv::kQuantTableBits) / params.dq_denom;
    dequant[static_cast<std::size_t>(i)] =
        std::max(-(1 << 15),
                 std::min((1 << 15) - 1, level < 0 ? -dq : dq));
  }
  std::vector<std::int16_t> recon(static_cast<std::size_t>(coded_n));
  txv::inverse_block(dequant.data(), params, recon.data());

  std::int64_t sse_i = 0;
  for (int i = 0; i < coded_n; ++i) {
    const std::int64_t d =
        static_cast<std::int64_t>(recon[static_cast<std::size_t>(i)]) -
        coded[static_cast<std::size_t>(i)];
    sse_i += d * d;
  }

  const txpart::TxScanClass cls =
      txpart::tx_class_of(static_cast<int>(tx_type));
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  if (!txpart::get_scan(tx_size, cls, scan)) {
    if (error != nullptr) *error = "scan table generation failed";
    return false;
  }
  int eob = 0;
  int nonzero = 0;
  for (int c = 0; c < area; ++c) {
    if (levels[scan[static_cast<std::size_t>(c)]] != 0) {
      ++nonzero;
      eob = c + 1;
    }
  }
  const double type_bits =
      options.use_type_bits && type.codes_type != 0
          ? type.bits[is_inter ? inter_ctx(eob, leaf_w, leaf_h) : 0]
          : 0.0;
  const double rate =
      eob == 0
          ? options.skip_bits
          : type_bits + options.coefficient_bits * static_cast<double>(nonzero);

  out->valid = true;
  out->tx_size = tx_size;
  out->tx_type = tx_type;
  out->eob = eob;
  out->nonzero = nonzero;
  out->sse = static_cast<double>(sse_i);
  out->rate_bits = rate;
  out->cost = out->sse + options.lambda * rate;
  out->residual = std::move(recon);
  out->levels.assign(static_cast<std::size_t>(area), 0);
  for (int c = 0; c < area; ++c) {
    out->levels[static_cast<std::size_t>(c)] = levels[scan[c]];
  }
  return true;
}

// §5.20.8.2 codability: intra `eob <= 1` forces DCT_DCT; inter codes the type
// even at eob == 1.
bool codable(std::int32_t tx_type, int eob, bool is_inter) {
  if (is_inter) return eob == 0 ? tx_type == 0 : true;
  return eob >= 2 || tx_type == 0;
}

struct PlanCandidate {
  bool valid = false;
  std::int32_t partition = 0;
  double sse = 0.0;
  double rate = 0.0;
  double cost = 0.0;
  txpart::SplitGeometry geometry{};
  LeafCandidate leaves[kMaxLeaves];
};

}  // namespace

bool search(const std::int16_t* residual_coded, std::int32_t coded_w,
            std::int32_t coded_h, Mode mode, std::int32_t qindex,
            std::int32_t delta_dc, std::int32_t delta_ac,
            const std::uint8_t* prediction, const Options& options,
            Result* out, std::string* error) {
  if (out == nullptr) return false;
  *out = Result{};
  if (residual_coded == nullptr || coded_w < 4 || coded_w > 64 ||
      coded_h < 4 || coded_h > 64) {
    if (error != nullptr) *error = "bad TX-search block";
    return false;
  }
  if (txv::tx_size_index(coded_w, coded_h) < 0) {
    if (error != nullptr) {
      *error = "block size is not one of the 25 AV2 transform sizes";
    }
    return false;
  }
  const bool is_inter = mode == Mode::kInter;
  const int mi_size = mi_size_for_px(coded_w, coded_h);
  const bool partition_search = options.search_partition && mi_size >= 0;
  const txpart::LegalSplits legal =
      partition_search
          ? txpart::legal_splits(mi_size, options.reduced_tx_part_set)
          : txpart::LegalSplits{};
  const bool can_partition = partition_search && legal.applicable;

  std::vector<PlanCandidate> plans;
  plans.reserve(4);

  const auto evaluate_plan = [&](std::int32_t partition,
                                 const txpart::SplitGeometry& geometry,
                                 double partition_bits) -> bool {
    PlanCandidate plan;
    plan.partition = partition;
    plan.geometry = geometry;
    plan.rate = partition_bits;
    plan.cost = partition_bits;
    plan.valid = geometry.count > 0 && geometry.count <= kMaxLeaves;
    for (int k = 0; k < geometry.count && plan.valid; ++k) {
      const int tx_size = geometry.parts[k].tx_size;
      const int leaf_w = txpart::tx_width(tx_size);
      const int leaf_h = txpart::tx_height(tx_size);
      if (tx_size < 0 || tx_size >= 25 || leaf_w > options.max_tx_size ||
          leaf_h > options.max_tx_size) {
        if (error != nullptr) *error = "partition leaf exceeds max_tx_size";
        return false;
      }
      std::vector<TypeEntry> types;
      if (!build_leaf_types(leaf_w, leaf_h, mode, options, &types, error)) {
        return false;
      }
      LeafCandidate best;
      bool added = false;
      for (const TypeEntry& entry : types) {
        if (!options.search_type && entry.type != 0) continue;
        LeafCandidate cand;
        if (!evaluate_candidate(
                residual_coded, coded_w, coded_h, geometry.parts[k].mi_col * 4,
                geometry.parts[k].mi_row * 4, leaf_w, leaf_h, tx_size,
                entry.type, entry, is_inter, qindex, delta_dc, delta_ac,
                options, &cand, error)) {
          return false;
        }
        added = true;
        if (!codable(cand.tx_type, cand.eob, is_inter)) continue;
        if (!best.valid || cand.cost < best.cost ||
            (cand.cost == best.cost &&
             (cand.sse < best.sse ||
              (cand.sse == best.sse && cand.tx_type < best.tx_type)))) {
          best = std::move(cand);
        }
      }
      if (!added) {
        LeafCandidate cand;
        TypeEntry dct;
        dct.type = 0;
        dct.codes_type = 0;
        if (!evaluate_candidate(
                residual_coded, coded_w, coded_h, geometry.parts[k].mi_col * 4,
                geometry.parts[k].mi_row * 4, leaf_w, leaf_h, tx_size, 0, dct,
                is_inter, qindex, delta_dc, delta_ac, options, &cand, error)) {
          return false;
        }
        if (!codable(cand.tx_type, cand.eob, is_inter)) {
          plan.valid = false;
          break;
        }
        best = std::move(cand);
      } else if (!best.valid) {
        plan.valid = false;
        break;
      }
      plan.leaves[k] = std::move(best);
      plan.cost += plan.leaves[k].cost;
      plan.sse += plan.leaves[k].sse;
      plan.rate += plan.leaves[k].rate_bits;
    }
    if (plan.valid) plans.push_back(std::move(plan));
    return true;
  };

  if (options.search_none && (options.partition_mask & kPartBitNone) != 0) {
    txpart::SplitGeometry geometry{};
    geometry.count = 1;
    geometry.parts[0] = txpart::SplitPart{
        0, 0, txv::tx_size_index(coded_w, coded_h)};
    const double bits =
        can_partition
            ? partition_bits_for(mi_size, is_inter, options.reduced_tx_part_set,
                                 0, legal)
            : 0.0;
    if (!evaluate_plan(0, geometry, bits)) return false;
  }

  if (can_partition) {
    for (int oi = 0; oi < legal.option_count; ++oi) {
      const int partition = static_cast<int>(legal.options[oi]);
      if (partition == 0) continue;  // NONE handled above
      const int bit = partition_bit(partition);
      if (bit == 0 || (options.partition_mask & bit) == 0) continue;
      txpart::SplitGeometry geometry{};
      if (!txpart::tx_partition_geometry(mi_size, 0, 0, legal.options[oi],
                                         &geometry, error)) {
        return false;
      }
      const double bits = partition_bits_for(
          mi_size, is_inter, options.reduced_tx_part_set, partition, legal);
      if (!evaluate_plan(partition, geometry, bits)) return false;
    }
  }

  if (plans.empty()) {
    if (error != nullptr) *error = "no candidate plan selected";
    return false;
  }

  const PlanCandidate* best = nullptr;
  for (const PlanCandidate& plan : plans) {
    if (!plan.valid) continue;
    if (best == nullptr || plan.cost < best->cost ||
        (plan.cost == best->cost && plan.sse < best->sse)) {
      best = &plan;
    }
  }
  if (best == nullptr) {
    if (error != nullptr) *error = "no codable TX plan";
    return false;
  }

  out->ok = true;
  out->partition = best->partition;
  out->count = best->geometry.count;
  out->sse = best->sse;
  out->rate_bits = best->rate;
  out->cost = best->cost;
  const int coded_n = coded_w * coded_h;
  if (prediction != nullptr) {
    out->recon.resize(static_cast<std::size_t>(coded_n));
  }
  for (int k = 0; k < best->geometry.count; ++k) {
    Leaf& leaf = out->leaves[k];
    const LeafCandidate& src = best->leaves[k];
    leaf.x = best->geometry.parts[k].mi_col * 4;
    leaf.y = best->geometry.parts[k].mi_row * 4;
    leaf.coded_w = txpart::tx_width(best->geometry.parts[k].tx_size);
    leaf.coded_h = txpart::tx_height(best->geometry.parts[k].tx_size);
    leaf.tx_size = src.tx_size;
    leaf.tx_type = src.tx_type;
    leaf.mi_row = best->geometry.parts[k].mi_row;
    leaf.mi_col = best->geometry.parts[k].mi_col;
    leaf.eob = src.eob;
    leaf.levels = src.levels;
    leaf.residual = src.residual;
    if (prediction != nullptr) {
      for (int y = 0; y < leaf.coded_h; ++y) {
        for (int x = 0; x < leaf.coded_w; ++x) {
          const int px = leaf.x + x;
          const int py = leaf.y + y;
          const int p = prediction[static_cast<std::size_t>(py) * coded_w + px];
          const int r =
              src.residual[static_cast<std::size_t>(y) * leaf.coded_w + x];
          out->recon[static_cast<std::size_t>(py) * coded_w + px] =
              static_cast<std::uint8_t>(std::clamp(p + r, 0, 255));
        }
      }
    }
  }
  return true;
}

}  // namespace av2enc::video::txtsearch
