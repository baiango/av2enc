#include "pipeline/video/video_tx_search.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "codec/rate/rc/rc_tables.h"
#include "codec/rate/rdo/rdo.h"
#include "codec/rate/rdoq/rdoq.h"
#include "codec/rate/rdoq/rdoq_av2_model.h"
#include "codec/transform/tcq/tcq.h"
#include "codec/transform/tcq/tcq_av2_model.h"
#include "codec/transform/inter_residual/inter_tx_type.h"
#include "codec/transform/inter_residual/video_residual_cdf.h"
#include "codec/transform/ddt/ddt.h"
#include "codec/transform/tx_partition/tx_partition.h"
#include "codec/transform/tx_partition/tx_scan.h"
#include "codec/transform/tx_type/tx_type.h"
#include "codec/transform/variants/tx_variants.h"
#include "core/parallel/parallel_for.h"
#include "pipeline/video/video_tx_scratch.h"

namespace av2enc::video::txtsearch {
namespace {

namespace txv = av2enc::txv;
namespace txtype = av2enc::txtype;
namespace txpart = av2enc::video::txpartition;
namespace residual = av2enc::video::residual;

// ---------------------------------------------------------------------------
// AV2ENC_DZ_DUMP: aggregate of what the scalar quantiser rounds away, measured
// in level units before any trellis touches the result.  Per-trial output
// would flood stderr (the search evaluates many candidates per leaf), so the
// totals accumulate and print once at exit.  Diagnostic only.
// ---------------------------------------------------------------------------
struct DeadZoneDump {
  long trials = 0;
  long coeff_total = 0;
  long kept = 0;
  long dropped = 0;
  // Energy (in level^2) of the residual the quantiser threw away vs kept.
  double dropped_energy = 0.0;
  double kept_energy = 0.0;
  // Dropped coefficients bucketed by their PRE-rounding magnitude in level
  // units.  Everything dropped sits in [0, 0.5), because |x| >= 0.5 rounds to
  // a non-zero level and is kept; the buckets show how close to the decision
  // the discarded residual sits.
  long dropped_mag[4] = {0, 0, 0, 0};  // <0.125, <0.25, <0.375, <0.5
  // Coefficients dropped while carrying more than a quarter level of magnitude.
  long dropped_material = 0;
};

inline DeadZoneDump& dead_zone_dump() {
  static DeadZoneDump d;
  return d;
}

inline void dead_zone_dump_register() {
  static const bool once = [] {
    std::atexit([] {
      const DeadZoneDump& d = dead_zone_dump();
      if (d.coeff_total == 0) return;
      std::fprintf(stderr,
                   "dz-scalar coeffs=%ld kept=%ld (%.1f%%) dropped=%ld "
                   "dropped_material=%ld (%.1f%% of dropped)\n"
                   "dz-energy kept=%.0f dropped=%.0f -> dropped share %.2f%%\n"
                   "dz-dropped-by-magnitude <.125=%ld <.25=%ld <.375=%ld <.5=%ld\n",
                   d.coeff_total, d.kept,
                   100.0 * static_cast<double>(d.kept) /
                       static_cast<double>(d.coeff_total),
                   d.dropped, d.dropped_material,
                   d.dropped ? 100.0 * static_cast<double>(d.dropped_material) /
                                   static_cast<double>(d.dropped)
                             : 0.0,
                   d.kept_energy, d.dropped_energy,
                   100.0 * d.dropped_energy /
                       (d.kept_energy + d.dropped_energy),
                   d.dropped_mag[0], d.dropped_mag[1], d.dropped_mag[2],
                   d.dropped_mag[3]);
    });
    return true;
  }();
  (void)once;
}

inline const bool g_dump_dead_zone = [] {
  const bool on = std::getenv("AV2ENC_DZ_DUMP") != nullptr;
  if (on) dead_zone_dump_register();
  return on;
}();
namespace ddt = av2enc::video::ddt;

// Partition bit indices, identical to VideoTxPartitionBit.
constexpr std::int32_t kPartBitNone = 1 << 0;
constexpr std::int32_t kPartBitSplit = 1 << 1;
constexpr std::int32_t kPartBitHorz = 1 << 2;
constexpr std::int32_t kPartBitVert = 1 << 3;
constexpr std::int32_t kPartBitHorz4 = 1 << 4;
constexpr std::int32_t kPartBitVert4 = 1 << 5;

// Minimum candidate count before a per-block loop hands work to the worker
// pool.  A fork-join round trip wakes `participants` threads through a
// condition variable (a few microseconds each), so a 2-3 candidate loop --
// the common DCT-only or kDctIdtx case -- is faster serially.  Measured on
// 240p QP106 and 720p QP217 with --cpu-threads 8: 2 candidates lose, 4
// candidates are a wash, 8+ win (see docs/video-candidate-parallel.md).
constexpr std::size_t kTypeCandidateParallelMin = 4;

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
                      std::string& error) {
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
      error = "DCT_DCT missing from a DCTONLY set";
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
    // §7.15.4.1: enable_inter_ddt does not remove or add candidates; it only
    // swaps the ADST/FDST 1-D kernels at reconstruction time.  The historical
    // `inter_tx_type_is_ddt()` filter (which also removed the identity types)
    // must not run: the encoder applies the DDT kernels instead.
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
    error = "empty inter TX-type set";
    return false;
  }
  return true;
}

bool intra_leaf_types(std::int32_t coded_w, std::int32_t coded_h,
                      const Options& options, std::vector<TypeEntry>* out,
                      std::string& error) {
  txtype::TxTypeQuery q;
  q.coded_w = coded_w;
  q.coded_h = coded_h;
  q.mapped_mode = options.intra_mapped_mode;
  q.reduced_tx_set = options.reduced_tx_set;
  q.dct_only = options.dct_only;
  txtype::TxTypeSet set;
  if (!txtype::intra_tx_candidates(q, &set)) {
    error = "intra TX-type candidate enumeration failed";
    return false;
  }
  out->clear();
  for (int i = 0; i < set.count; ++i) {
    const int type = static_cast<int>(set.candidates[i].type);
    if (!options.include_1d_types && type >= 10) continue;
    if (type == 9 && !options.allow_idtx) continue;  // §5.4.8
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
    error = "empty intra TX-type set";
    return false;
  }
  return true;
}

bool build_leaf_types(std::int32_t coded_w, std::int32_t coded_h, Mode mode,
                      const Options& options, std::vector<TypeEntry>* out,
                      std::string& error) {
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
  TypeEntry type;
  int eob = 0;
  int nonzero = 0;
  double sse = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  std::vector<std::int32_t> levels;   // scan order
  std::vector<std::int16_t> residual; // coded size
};

// One evaluated candidate: the scalar decision inputs plus spans into the
// payload ring.  Candidates that lose the comparator never own heap storage;
// the winner is copied into a LeafCandidate by `materialize_candidate()`.
struct CandidateSlot {
  bool valid = false;
  std::int32_t tx_size = 0;
  std::int32_t tx_type = 0;
  TypeEntry type;
  int eob = 0;
  int nonzero = 0;
  double sse = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  std::span<std::int32_t> levels;   // scan order, area entries
  std::span<std::int16_t> residual; // coded size
};

// Largest candidate list any leaf can produce: the inter SET_1 inverse table
// has 16 entries, and every other set is shorter.  The array is doubled so a
// future set enlargement cannot overflow the stack slots silently.
constexpr std::size_t kMaxTypeSlots = 32;

// Copies a scored candidate's payload into the winner storage.  Identical
// bytes to the previous `best = std::move(cand)`, but the candidate itself
// stays in the ring.
void materialize_candidate(const CandidateSlot& src, LeafCandidate* dst) {
  dst->valid = true;
  dst->tx_size = src.tx_size;
  dst->tx_type = src.tx_type;
  dst->type = src.type;
  dst->eob = src.eob;
  dst->nonzero = src.nonzero;
  dst->sse = src.sse;
  dst->rate_bits = src.rate_bits;
  dst->cost = src.cost;
  dst->levels.assign(src.levels.begin(), src.levels.end());
  dst->residual.assign(src.residual.begin(), src.residual.end());
}

// §8.3.2 coefficient syntax cost for one candidate leaf, from the static
// §9.3 CDF slice.  This is the same syntax the writer emits (`txb_skip`, the
// EOB prefix/extra bits, `coeff_base`/`coeff_base_eob`, `coeff_br`, the
// `read_quant` escape and the sign), so candidate comparison stops pretending
// every nonzero costs one bit.  `levels_raster` is the quantised candidate in
// raster order and `scan` is its §5.20.7.30 scan; `tcq` selects the TCQ
// syntax variant (escape at maxLevel - 1) whose base levels this candidate
// carries.  Contexts that depend on neighbouring coded blocks are not
// available inside the search: `txb_skip_ctx`/`dc_sign_ctx` are fixed
// estimates (0 is the writer's "same size as the coding block" context).
// Returns false when the static model rejects the block; the caller then
// keeps the historical heuristic.
// Under `--video-rdoq` the static §8.3 model prices the default (non-adaptive)
// CDF tables while the tile coder adapts its CDFs, so the real syntax cost is
// roughly this fraction of the estimate.  The RDOQ cost model therefore scales
// the static bits by `Options::model_rate_scale` (0.4, the measured
// calibration from the `--video-rdoq` integration; an exact-rate search sets
// 1.0).

bool block_syntax_bits(const txv::TxParams& params,
                       std::span<const std::int32_t> levels_raster,
                       std::span<const std::uint16_t> scan, int qctx,
                       bool is_inter, bool tcq, int txb_skip_ctx,
                       int dc_sign_ctx, int plane, TxScratch& scratch,
                       const av2enc::still::qctx::CoeffCdfSlice* cdf_slice,
                       double* out_bits) {
  const int area = params.w * params.h;
  if (area <= 0 ||
      levels_raster.size() < static_cast<std::size_t>(area) ||
      scan.size() < static_cast<std::size_t>(area)) {
    return false;
  }
  const std::span<std::int32_t> scan_levels(
      scratch.syntax_scan, static_cast<std::size_t>(area));
  for (int c = 0; c < area; ++c) {
    scan_levels[static_cast<std::size_t>(c)] =
        levels_raster[scan[static_cast<std::size_t>(c)]];
  }
  std::string error;
  if (tcq) {
    av2enc::tcq::av2::BlockInfo info;
    info.plane = plane;
    info.params = params;
    info.txb_skip_context = std::clamp(txb_skip_ctx, 0, 9);
    info.dc_sign_context = std::clamp(dc_sign_ctx, 0, 2);
    info.qctx = qctx;
    info.is_inter = is_inter;
    info.tx_class_2d = true;
    av2enc::tcq::av2::Model& model = scratch.tcq_model;
    if (!model.init(info, scan_levels, error)) return false;
    *out_bits = model.reference_bits();
    return true;
  }
  av2enc::rdoq::av2::BlockInfo info;
  info.plane = plane;
  info.params = params;
  info.txb_skip_context = std::clamp(txb_skip_ctx, 0, 9);
  info.dc_sign_context = std::clamp(dc_sign_ctx, 0, 2);
  info.qctx = qctx;
  info.is_inter = is_inter;
  info.cdf_override = cdf_slice;
  av2enc::rdoq::av2::Model& model = scratch.rdoq_model;
  if (!model.init(info, scan_levels, error)) return false;
  *out_bits = model.scalar_bits();
  return true;
}

// Full forward -> quantise -> dequantise -> inverse path for one candidate,
// with the same coded-domain SSE and rate model as the device kernel.  All
// buffers come from the per-thread scratch and the candidate's payload spans;
// the arithmetic is unchanged.
bool evaluate_candidate(const std::int16_t* residual, int block_w,
                        int /*block_h*/, int offset_x, int offset_y, int leaf_w,
                        int leaf_h, std::int32_t tx_size, std::int32_t tx_type,
                        const TypeEntry& type, bool is_inter, int qindex,
                        int delta_dc, int delta_ac, const Options& options,
                        bool rdoq, CandidateSlot* out, std::string& error) {
  const auto params_opt = txv::make_tx_params(
      static_cast<txv::TxSize>(tx_size), static_cast<txv::TxType>(tx_type));
  if (!params_opt.has_value()) {
    error = "unsupported transform size/type";
    return false;
  }
  const txv::TxParams& params = *params_opt;
  if (params.coded_w != leaf_w || params.coded_h != leaf_h) {
    error = "candidate size does not match leaf";
    return false;
  }
  const int area = params.w * params.h;
  const int coded_n = leaf_w * leaf_h;
  // The ring is sized from tx_scan_area(tx_size), which is exactly the
  // adjusted grid params.w * params.h for all 25 sizes.
  assert(out->levels.size() == static_cast<std::size_t>(area));
  assert(out->residual.size() == static_cast<std::size_t>(coded_n));
  TxScratch& scratch = tx_scratch();

  std::int16_t* coded = scratch.coded;
  for (int y = 0; y < leaf_h; ++y) {
    for (int x = 0; x < leaf_w; ++x) {
      coded[static_cast<std::size_t>(y) * static_cast<std::size_t>(leaf_w) +
            static_cast<std::size_t>(x)] =
          residual[static_cast<std::size_t>(offset_y + y) *
                       static_cast<std::size_t>(block_w) +
                   static_cast<std::size_t>(offset_x) +
                   static_cast<std::size_t>(x)];
    }
  }
  std::int16_t* adj = scratch.adj;
  txv::downsample_residual(coded, params, adj);
  // §7.15.4.1: when enable_inter_ddt derives DDTX/FDDT for this block, the
  // decoder reconstructs with the DDT kernels, so the search must forward and
  // inverse-transform with the same kernels.  Non-DDT candidates keep the
  // historical txv path byte-for-byte.
  ddt::Context ddt_ctx;
  ddt_ctx.enable_inter_ddt = options.enable_inter_ddt;
  ddt_ctx.is_inter = is_inter;
  ddt_ctx.use_intrabc = false;
  ddt_ctx.bit_depth = 8;
  const bool use_ddt = ddt::plan_has_ddt(params, ddt_ctx);
  double* coeff = scratch.coeff;
  std::int32_t* coeff_i = scratch.coeff_i;
  double scaled = params.quant_factor * 528.0;
  if (use_ddt) {
    std::fill(coeff_i, coeff_i + area, 0);
    (void)ddt::forward_block(adj, params, ddt_ctx, coeff_i);
    scaled = ddt::quant_factor(params) * 528.0;
  } else {
    std::fill(coeff, coeff + area, 0.0);
    txv::forward_block(adj, params, coeff);
  }

  const int dcq = get_q(qindex, delta_dc);
  const int acq = get_q(qindex, delta_ac);
  std::int32_t* levels = scratch.levels;
  for (int i = 0; i < area; ++i) {
    const int q = i == 0 ? dcq : acq;
    const double factor = scaled / static_cast<double>(q);
    const double source =
        use_ddt ? static_cast<double>(coeff_i[static_cast<std::size_t>(i)])
                : coeff[static_cast<std::size_t>(i)];
    const double x = source * factor;
    const std::int32_t level = static_cast<std::int32_t>(std::lround(x));
    levels[static_cast<std::size_t>(i)] = level;
    if (g_dump_dead_zone) {
      DeadZoneDump& d = dead_zone_dump();
      d.coeff_total++;
      if (level != 0) {
        d.kept++;
        d.kept_energy += x * x;
      } else {
        d.dropped++;
        d.dropped_energy += x * x;
        const double a = x < 0.0 ? -x : x;
        if (a >= 0.25) d.dropped_material++;
        const int b = a < 0.125 ? 0 : a < 0.25 ? 1 : a < 0.375 ? 2 : 3;
        d.dropped_mag[b]++;
      }
    }
  }

  const txpart::TxScanClass cls =
      txpart::tx_class_of(static_cast<int>(tx_type));
  bool scan_ok = false;
  const std::span<const std::uint16_t> scan =
      scratch.scan_table(tx_size, cls, &scan_ok);
  if (!scan_ok || scan.size() < static_cast<std::size_t>(area)) {
    error = "scan table generation failed";
    return false;
  }

  // §5.20.7.27 TCQ.  Luma-only (the search is luma-only), no FSC/lossless, and
  // TX_CLASS_2D only.  The trellis replaces the scalar levels with the site
  // syntax base levels and returns the mapped Quant[] the dequant consumes.
  std::int32_t* dequant_levels = scratch.dequant_levels;
  const bool use_tcq = options.tcq && dcq == acq && tx_type <= 9;
  if (use_tcq) {
    const std::span<av2enc::tcq::Coefficient> input(
        scratch.tcq_input, static_cast<std::size_t>(area));
    const std::span<std::int32_t> reference(
        scratch.aux, static_cast<std::size_t>(area));
    for (int c = 0; c < area; ++c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      const int row = pos / params.w;
      const int col = pos % params.w;
      input[static_cast<std::size_t>(c)].original =
          use_ddt ? static_cast<double>(coeff_i[static_cast<std::size_t>(pos)])
                  : coeff[static_cast<std::size_t>(pos)];
      input[static_cast<std::size_t>(c)].level_context = 0;
      input[static_cast<std::size_t>(c)].pre_max = (row + col) < 4 ? 8 : 6;
      const std::int32_t scalar = levels[static_cast<std::size_t>(pos)];
      const std::int32_t magnitude = std::abs(scalar);
      const std::int32_t base = (magnitude + 1) / 2;
      reference[static_cast<std::size_t>(c)] = scalar < 0 ? -base : base;
    }
    av2enc::tcq::av2::BlockInfo info;
    info.plane = 0;
    info.params = params;
    info.txb_skip_context = 0;
    info.dc_sign_context = 0;
    info.qctx = options.tcq_qctx;
    info.is_inter = is_inter;
    info.tx_class_2d = true;
    av2enc::tcq::av2::Model& model = scratch.tcq_model;
    std::string tcq_error;
    if (!model.init(info, reference, tcq_error)) {
      error = "tcq model init failed: " + tcq_error;
      return false;
    }
    av2enc::tcq::Config config;
    config.quant_step = static_cast<double>(acq) / scaled;
    config.lambda = av2enc::rdo::lambda_sse_from_quant_step(
        config.quant_step, /*sse_scale=*/0.125);
    config.max_coefficients = area;
    av2enc::tcq::Result& result = scratch.tcq_result;
    if (!av2enc::tcq::quantize(input, config, model.rate_model(), result,
                               tcq_error)) {
      error = "tcq quantize failed: " + tcq_error;
      return false;
    }
    std::fill(levels, levels + area, 0);
    std::fill(dequant_levels, dequant_levels + area, 0);
    for (int c = 0; c < result.eob; ++c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      levels[static_cast<std::size_t>(pos)] =
          result.base_levels[static_cast<std::size_t>(c)];
      dequant_levels[static_cast<std::size_t>(pos)] =
          result.quant_levels[static_cast<std::size_t>(c)];
    }
  }

  // Opt-in coefficient-level RDOQ.  Inter luma only, scalar DC == AC step, no
  // DDT (the trellis forward coefficients come from `txv::forward_block`).
  // The trellis returns syntax levels, so the reconstruction below and the
  // syntax writer see exactly the same coefficients.
  double rdoq_level_bits = -1.0;
  const bool intra_rdoq_ok = is_inter || options.intra_rdoq;
  const bool use_rdoq = rdoq && !use_tcq && options.rdoq && intra_rdoq_ok &&
                        dcq == acq && tx_type <= 9 && !use_ddt;
  if (use_rdoq) {
    const std::span<std::int32_t> scalar_scan(
        scratch.aux, static_cast<std::size_t>(area));
    const std::span<av2enc::rdoq::Coefficient> input(
        scratch.rdoq_input, static_cast<std::size_t>(area));
    for (int c = 0; c < area; ++c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      scalar_scan[static_cast<std::size_t>(c)] =
          levels[static_cast<std::size_t>(pos)];
      input[static_cast<std::size_t>(c)].original =
          coeff[static_cast<std::size_t>(pos)];
      input[static_cast<std::size_t>(c)].scalar_level =
          levels[static_cast<std::size_t>(pos)];
      input[static_cast<std::size_t>(c)].level_context = 0;
    }
    av2enc::rdoq::av2::BlockInfo info;
    info.plane = 0;
    info.params = params;
    info.txb_skip_context = 0;
    info.dc_sign_context = 0;
    info.qctx = options.rdoq_qctx;
    info.is_inter = is_inter;
    av2enc::rdoq::av2::Model& model = scratch.rdoq_model;
    std::string rdoq_error;
    if (!model.init(info, scalar_scan, rdoq_error)) {
      error = "rdoq model init failed: " + rdoq_error;
      return false;
    }
    av2enc::rdoq::Config config;
    config.quant_step = static_cast<double>(acq) / scaled;
    config.lambda = av2enc::rdo::lambda_sse_from_quant_step(
        config.quant_step, options.rdoq_sse_scale);
    config.max_coefficients = area;
    av2enc::rdoq::RdoqResult& result = scratch.rdoq_result;
    if (!av2enc::rdoq::rdoq_quantize(input, config, model.rate_model(), result,
                                     rdoq_error)) {
      error = "rdoq quantize failed: " + rdoq_error;
      return false;
    }
    std::fill(levels, levels + area, 0);
    for (int c = 0; c < result.eob; ++c) {
      const int pos = scan[static_cast<std::size_t>(c)];
      levels[static_cast<std::size_t>(pos)] =
          result.levels[static_cast<std::size_t>(c)];
    }
    rdoq_level_bits = result.rate_bits;
  }

  // Real §8.3 coefficient rate for candidate selection under --video-rdoq.
  // The historical `coefficient_bits * nonzero` placeholder cannot see level
  // magnitudes, EOB or the skip symbol; when RDOQ is on the search prices
  // every inter luma candidate with the same static model the trellis uses.
  const bool model_rate =
      options.rdoq && intra_rdoq_ok && dcq == acq && tx_type <= 9 &&
      !use_ddt && !use_tcq;

  std::int32_t* dequant = scratch.dequant;
  const int dq_denom = use_tcq ? (params.dq_denom << 1) : params.dq_denom;
  for (int i = 0; i < area; ++i) {
    const int q = i == 0 ? dcq : acq;
    const std::int32_t level =
        use_tcq ? dequant_levels[static_cast<std::size_t>(i)]
                : levels[static_cast<std::size_t>(i)];
    const std::int32_t aq = level < 0 ? -level : level;
    const std::int32_t high = (aq * q) & 0xFFFFFF;
    const std::int32_t dq = txv::round2(high, txv::kQuantTableBits) / dq_denom;
    dequant[static_cast<std::size_t>(i)] =
        std::max(-(1 << 15),
                 std::min((1 << 15) - 1, level < 0 ? -dq : dq));
  }
  std::int16_t* recon = scratch.recon;
  if (use_ddt) {
    (void)ddt::inverse_block(dequant, params, ddt_ctx, recon);
  } else {
    txv::inverse_block(dequant, params, recon);
  }

  std::int64_t sse_i = 0;
  for (int i = 0; i < coded_n; ++i) {
    const std::int64_t d =
        static_cast<std::int64_t>(recon[static_cast<std::size_t>(i)]) -
        coded[static_cast<std::size_t>(i)];
    sse_i += d * d;
  }
  // AV2ENC_PIPE_PERC: apply the source-driven per-sample weights to the
  // candidate distortion.  The plan's SSE is what the leaf comparison and the
  // plan-vs-plan tie-break read, so the TX choice and the leaf J use the same
  // number.  Null keeps the historical integer SSE.
  double sse_value = static_cast<double>(sse_i);
  if (options.distortion_weight != nullptr || options.lowpass_gain > 0.0 ||
      options.lowpass2_gain > 0.0 || options.detail_band_gain > 0.0) {
    double weighted = 0.0;
    for (int i = 0; i < coded_n; ++i) {
      const std::int64_t d =
          static_cast<std::int64_t>(recon[static_cast<std::size_t>(i)]) -
          coded[static_cast<std::size_t>(i)];
      const double weight =
          options.distortion_weight != nullptr
              ? static_cast<double>(
                    options.distortion_weight[static_cast<std::size_t>(i)])
              : 256.0;
      double value = static_cast<double>(d * d);
      if (options.lowpass_gain > 0.0 || options.lowpass2_gain > 0.0 ||
          options.detail_band_gain > 0.0) {
        const int x = i % leaf_w;
        const int y = i / leaf_w;
        // AV2ENC_TX_DETAIL_SEARCH_TAIL: 2-4 px error band, the difference of
        // an aligned 2x2 mean and the centered 5x5 mean (both clipped to the
        // transform leaf).  A 16x16 -> four 8x8 split changes how much of
        // this band the quantiser can localise, so the price is applied to the
        // reconstruction error the leaf decision already sees.
        if (options.detail_band_gain > 0.0) {
          const int cell_x = x & ~1;
          const int cell_y = y & ~1;
          std::int64_t sum2 = 0;
          int used2 = 0;
          for (int dy = 0; dy < 2; ++dy) {
            const int py = cell_y + dy;
            if (py >= leaf_h) break;
            for (int dx = 0; dx < 2; ++dx) {
              const int px = cell_x + dx;
              if (px >= leaf_w) break;
              const int j = py * leaf_w + px;
              sum2 += static_cast<std::int64_t>(
                          recon[static_cast<std::size_t>(j)]) -
                      coded[static_cast<std::size_t>(j)];
              ++used2;
            }
          }
          std::int64_t sum5 = 0;
          int used5 = 0;
          for (int dy = -2; dy <= 2; ++dy) {
            const int py = y + dy;
            if (py < 0 || py >= leaf_h) continue;
            for (int dx = -2; dx <= 2; ++dx) {
              const int px = x + dx;
              if (px < 0 || px >= leaf_w) continue;
              const int j = py * leaf_w + px;
              sum5 += static_cast<std::int64_t>(
                          recon[static_cast<std::size_t>(j)]) -
                      coded[static_cast<std::size_t>(j)];
              ++used5;
            }
          }
          const double mean2 = static_cast<double>(sum2) /
                               static_cast<double>(used2);
          const double mean5 = static_cast<double>(sum5) /
                               static_cast<double>(used5);
          const double band = mean2 - mean5;
          value += options.detail_band_gain * band * band;
        }
        if (options.lowpass_gain > 0.0 || options.lowpass2_gain > 0.0) {
          std::int64_t low = 0;
          int used = 0;
          for (int dy = -1; dy <= 1; ++dy) {
            const int py = y + dy;
            if (py < 0 || py >= leaf_h) continue;
            for (int dx = -1; dx <= 1; ++dx) {
              const int px = x + dx;
              if (px < 0 || px >= leaf_w) continue;
              const int j = py * leaf_w + px;
              low += static_cast<std::int64_t>(
                         recon[static_cast<std::size_t>(j)]) -
                     coded[static_cast<std::size_t>(j)];
              ++used;
            }
          }
          const double mean_low = static_cast<double>(low) / used;
          value += options.lowpass_gain * mean_low * mean_low;
          if (options.lowpass2_gain > 0.0) {
            std::int64_t low2 = 0;
            int used2 = 0;
            for (int dy = -3; dy <= 3; ++dy) {
              const int py = y + dy;
              if (py < 0 || py >= leaf_h) continue;
              for (int dx = -3; dx <= 3; ++dx) {
                const int px = x + dx;
                if (px < 0 || px >= leaf_w) continue;
                const int j = py * leaf_w + px;
                low2 += static_cast<std::int64_t>(
                            recon[static_cast<std::size_t>(j)]) -
                        coded[static_cast<std::size_t>(j)];
                ++used2;
              }
            }
            const double mean_low2 = static_cast<double>(low2) / used2;
            value += options.lowpass2_gain * mean_low2 * mean_low2;
          }
        }
      }
      weighted += weight * value;
    }
    sse_value = weighted / 256.0;
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
  double rate = eob == 0
                    ? options.skip_bits
                    : type_bits + options.coefficient_bits *
                                      static_cast<double>(nonzero);
  const auto tx_class =
      txpart::tx_class_of(static_cast<int>(tx_type));
  const std::span<const std::int32_t> levels_span(
      levels, static_cast<std::size_t>(area));
  if (model_rate && tx_class == txpart::TxScanClass::k2D) {
    // The RDOQ trellis already priced its own levels with the static §8.3
    // model; scalar candidates are priced here.  `rdoq_qctx` and `tcq_qctx`
    // are both the frame base_q_idx slice on the paths that enable them.
    double syntax_bits = rdoq_level_bits;
    if (!use_rdoq) syntax_bits = 0.0;
    if (use_rdoq || block_syntax_bits(
                        params, levels_span, scan, options.rdoq_qctx, is_inter,
                        use_tcq, /*txb_skip_ctx=*/0, /*dc_sign_ctx=*/0,
                        /*plane=*/0, scratch, options.cdf_slice,
                        &syntax_bits)) {
      rate = syntax_bits * options.model_rate_scale + type_bits;
    }
  } else if (options.use_context_rate &&
             tx_class == txpart::TxScanClass::k2D) {
    // The measured `--video-tx-context-rate` path prices the static §8.3
    // model directly (its own calibration lived in the TX lambda).
    double syntax_bits = 0.0;
    if (block_syntax_bits(params, levels_span, scan, options.tcq_qctx, is_inter,
                          use_tcq, /*txb_skip_ctx=*/0, /*dc_sign_ctx=*/0,
                          /*plane=*/0, scratch, options.cdf_slice,
                          &syntax_bits)) {
      rate = syntax_bits + type_bits;
    }
  }

  out->valid = true;
  out->tx_size = tx_size;
  out->tx_type = tx_type;
  out->type = type;
  out->eob = eob;
  out->nonzero = nonzero;
  out->sse = sse_value;
  out->rate_bits = rate;
  out->cost = out->sse + options.lambda * rate;
  std::copy(recon, recon + coded_n, out->residual.begin());
  for (int c = 0; c < area; ++c) {
    out->levels[static_cast<std::size_t>(c)] =
        levels[scan[static_cast<std::size_t>(c)]];
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
            Result* out, std::string& error) {
  if (out == nullptr) return false;
  *out = Result{};
  if (residual_coded == nullptr || coded_w < 4 || coded_w > 64 ||
      coded_h < 4 || coded_h > 64) {
    error = "bad TX-search block";
    return false;
  }
  if (txv::tx_size_index(coded_w, coded_h) < 0) {
error = "block size is not one of the 25 AV2 transform sizes";
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
        error = "partition leaf exceeds max_tx_size";
        return false;
      }
      std::vector<TypeEntry> types;
      if (!build_leaf_types(leaf_w, leaf_h, mode, options, &types, error)) {
        return false;
      }
      // Candidate list in evaluation order.  With `search_type` off only
      // DCT_DCT survives, so the loop below is a single candidate.
      std::vector<std::size_t> order;
      order.reserve(types.size());
      for (std::size_t t = 0; t < types.size(); ++t) {
        if (!options.search_type && types[t].type != 0) continue;
        order.push_back(t);
      }
      LeafCandidate best;
      bool added = false;
      // Selection with the historical comparator, unchanged: it tie-breaks on
      // (cost, sse, tx_type) and the caller feeds it in candidate order, so
      // the first candidate wins a full tie exactly as the serial loop did.
      const auto consider = [&](const CandidateSlot& cand) {
        added = true;
        if (!codable(cand.tx_type, cand.eob, is_inter)) return;
        if (!best.valid || cand.cost < best.cost ||
            (cand.cost == best.cost &&
             (cand.sse < best.sse ||
              (cand.sse == best.sse && cand.tx_type < best.tx_type)))) {
          materialize_candidate(cand, &best);
        }
      };
      // Payload slots: every candidate of this leaf gets its own span while
      // the (sequential) selection pass decides, so no candidate heap-allocs.
      // The serial loop only ever needs one slot.
      const bool pooled = parallel::pool_active() &&
                          order.size() >= kTypeCandidateParallelMin;
      CandidatePayloadRing& ring = candidate_ring();
      ring.prepare(pooled ? order.size() : 1, txpart::tx_scan_area(tx_size),
                   leaf_w * leaf_h);
      const std::size_t n_candidates = order.size();
      assert(n_candidates <= kMaxTypeSlots);
      CandidateSlot slots[kMaxTypeSlots];
      for (std::size_t i = 0; i < n_candidates; ++i) {
        slots[i].levels = ring.levels(i);
        slots[i].residual = ring.residual(i);
      }
      if (!pooled) {
        for (std::size_t t : order) {
          if (!evaluate_candidate(
                  residual_coded, coded_w, coded_h,
                  geometry.parts[k].mi_col * 4,
                  geometry.parts[k].mi_row * 4, leaf_w, leaf_h, tx_size,
                  types[t].type, types[t], is_inter, qindex, delta_dc, delta_ac,
                  options, /*rdoq=*/false, &slots[0], error)) {
            return false;
          }
          consider(slots[0]);
        }
      } else {
        // Every candidate is independent: evaluate_candidate owns all of its
        // work scratch, and each slot is written by one thread.  No shared
        // state, no reduction (the cost is a double and must not be summed in
        // a different order).
        std::string cand_errors[kMaxTypeSlots];
        parallel::parallel_for_pooled(
            n_candidates, kTypeCandidateParallelMin, [&](std::size_t i) {
              const TypeEntry& entry = types[order[i]];
              evaluate_candidate(residual_coded, coded_w, coded_h,
                                 geometry.parts[k].mi_col * 4,
                                 geometry.parts[k].mi_row * 4, leaf_w, leaf_h,
                                 tx_size, entry.type, entry, is_inter, qindex,
                                 delta_dc, delta_ac, options, /*rdoq=*/false,
                                 &slots[i], cand_errors[i]);
            });
        for (std::size_t i = 0; i < n_candidates; ++i) {
          if (!slots[i].valid) {
            error = cand_errors[i];
            return false;
          }
          consider(slots[i]);
        }
      }
      if (!added) {
        TypeEntry dct;
        dct.type = 0;
        dct.codes_type = 0;
        if (!evaluate_candidate(
                residual_coded, coded_w, coded_h,
                geometry.parts[k].mi_col * 4, geometry.parts[k].mi_row * 4,
                leaf_w, leaf_h, tx_size, 0, dct, is_inter, qindex, delta_dc,
                delta_ac, options, /*rdoq=*/false, &slots[0], error)) {
          return false;
        }
        if (!codable(slots[0].tx_type, slots[0].eob, is_inter)) {
          plan.valid = false;
          break;
        }
        materialize_candidate(slots[0], &best);
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
    error = "no candidate plan selected";
    return false;
  }

  PlanCandidate* best = nullptr;
  for (PlanCandidate& plan : plans) {
    if (!plan.valid) continue;
    if (best == nullptr || plan.cost < best->cost ||
        (plan.cost == best->cost && plan.sse < best->sse)) {
      best = &plan;
    }
  }
  if (best == nullptr) {
    error = "no codable TX plan";
    return false;
  }

  // Winner-only RDOQ: the scalar plan search decides size/type/partition, then
  // every coded leaf of the winning plan is re-quantised with the trellis.  A
  // leaf whose refined levels would make the block non-codable (an all-zero
  // inter leaf still codes its TX_TYPE; an intra leaf with eob <= 1 infers
  // DCT_DCT) keeps its scalar candidate.  `intra_rdoq` extends the same
  // contract to intra leaves (AV2ENC_RD_KEY_TX).
  if (options.rdoq && (is_inter || options.intra_rdoq)) {
    double scalar_leaf_bits = 0.0;
    for (int k = 0; k < best->geometry.count; ++k) {
      scalar_leaf_bits += best->leaves[k].rate_bits;
    }
    const double partition_bits = best->rate - scalar_leaf_bits;
    bool changed = false;
    for (int k = 0; k < best->geometry.count; ++k) {
      const LeafCandidate& scalar = best->leaves[k];
      const int leaf_w = txpart::tx_width(best->geometry.parts[k].tx_size);
      const int leaf_h = txpart::tx_height(best->geometry.parts[k].tx_size);
      CandidatePayloadRing& ring = candidate_ring();
      ring.prepare(1, txpart::tx_scan_area(scalar.tx_size), leaf_w * leaf_h);
      CandidateSlot refined;
      refined.levels = ring.levels(0);
      refined.residual = ring.residual(0);
      if (!evaluate_candidate(
              residual_coded, coded_w, coded_h,
              best->geometry.parts[k].mi_col * 4,
              best->geometry.parts[k].mi_row * 4, leaf_w, leaf_h,
              scalar.tx_size, scalar.tx_type, scalar.type, is_inter, qindex,
              delta_dc, delta_ac, options, /*rdoq=*/true, &refined, error)) {
        return false;
      }
      const LeafCandidate& incumbent = best->leaves[k];
      if (refined.valid && codable(refined.tx_type, refined.eob, is_inter) &&
          refined.cost <= incumbent.cost) {
        materialize_candidate(refined, &best->leaves[k]);
        changed = true;
      }
    }
    if (changed) {
      best->sse = 0.0;
      best->rate = partition_bits;
      best->cost = partition_bits;
      for (int k = 0; k < best->geometry.count; ++k) {
        best->sse += best->leaves[k].sse;
        best->rate += best->leaves[k].rate_bits;
        best->cost += best->leaves[k].cost;
      }
    }
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
          const int p = prediction[static_cast<std::size_t>(py) *
                                       static_cast<std::size_t>(coded_w) +
                                   static_cast<std::size_t>(px)];
          const int r = src.residual[static_cast<std::size_t>(y) *
                                         static_cast<std::size_t>(leaf.coded_w) +
                                     static_cast<std::size_t>(x)];
          out->recon[static_cast<std::size_t>(py) *
                         static_cast<std::size_t>(coded_w) +
                     static_cast<std::size_t>(px)] =
              static_cast<std::uint8_t>(std::clamp(p + r, 0, 255));
        }
      }
    }
  }
  return true;
}

bool scan_coefficient_bits(std::int32_t tx_size, std::int32_t tx_type,
                           std::int32_t plane,
                           const std::int32_t* levels_scan,
                           std::int32_t area, std::int32_t qctx,
                           bool is_inter, double* out_bits) {
  return scan_coefficient_bits(tx_size, tx_type, plane, levels_scan, area,
                               qctx, is_inter, /*slice=*/nullptr, out_bits);
}

bool scan_coefficient_bits(std::int32_t tx_size, std::int32_t tx_type,
                           std::int32_t plane,
                           const std::int32_t* levels_scan,
                           std::int32_t area, std::int32_t qctx,
                           bool is_inter,
                           const av2enc::still::qctx::CoeffCdfSlice* slice,
                           double* out_bits) {
  if (levels_scan == nullptr || out_bits == nullptr || area <= 0) return false;
  const auto params_opt = txv::make_tx_params(
      static_cast<txv::TxSize>(tx_size), static_cast<txv::TxType>(tx_type));
  if (!params_opt.has_value()) return false;
  const txv::TxParams& params = *params_opt;
  if (txpart::tx_class_of(tx_type) != txpart::TxScanClass::k2D) return false;
  const int scan_area = txpart::tx_scan_area(tx_size);
  if (area != scan_area ||
      area > params.w * params.h) {
    return false;
  }
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  if (!txpart::get_scan(tx_size, txpart::TxScanClass::k2D, scan)) {
    return false;
  }
  std::vector<std::int32_t> raster(
      static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h),
      0);
  for (int c = 0; c < area; ++c) {
    raster[scan[static_cast<std::size_t>(c)]] =
        levels_scan[static_cast<std::size_t>(c)];
  }
  TxScratch scratch;
  return block_syntax_bits(params, raster, scan, qctx, is_inter,
                           /*tcq=*/false, /*txb_skip_ctx=*/0,
                           /*dc_sign_ctx=*/0, plane, scratch, slice,
                           out_bits);
}

}  // namespace av2enc::video::txtsearch
