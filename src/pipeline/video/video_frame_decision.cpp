#include "pipeline/video/video_frame_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "codec/transform/tx_partition/tx_scan.h"
#include "pipeline/video/video_interp_filter.h"
#include "pipeline/video/video_horizon_port.h"
#include "pipeline/video/video_perceptual.h"
#include "pipeline/video/video_p_drift.h"
#include "pipeline/video/video_pipe_perc.h"
#include "pipeline/video/video_skip_drift.h"
#include "pipeline/video/video_uv_rd.h"
#include "pipeline/video/video_uv_search.h"
#include "pipeline/video/video_rd_engine.h"
#include "pipeline/video/video_seq_rd.h"

namespace av2enc::video {

namespace {

// AV2ENC_PIPE_PERC coefficient price for one already-quantised plan.  The
// inter luma TX search already filled `rate_bits` from the same static §8.3.2
// context model, so that value is reused (it also carries the partition and
// TX-type symbols the search priced).  Every other plan (intra luma, chroma)
// is priced here from its scan-order levels; a model failure falls back to the
// historical 8 + 2.5*nz + 0.5*eob estimate for consistency.
double pipe_plan_coefficient_bits(
    const TxPlan& plan, int plane, bool is_inter, int qctx,
    const still::qctx::CoeffCdfSlice* cdf_slice = nullptr) {
  if (plan.searched && is_inter && plane == 0 && cdf_slice == nullptr) {
    return plan.rate_bits;
  }
  const auto part_bits = [&](int tx_size, int tx_type,
                             const std::int32_t* levels_scan, int area,
                             int eob) {
    if (eob == 0) {
      std::vector<std::int32_t> zeros(static_cast<std::size_t>(area), 0);
      double bits = 0.0;
      if (area > 0 &&
          txtsearch::scan_coefficient_bits(tx_size, tx_type, plane,
                                           zeros.data(), area, qctx, is_inter,
                                           cdf_slice, &bits)) {
        return bits;
      }
      return 1.0;
    }
    double bits = 0.0;
    if (area > 0 &&
        txtsearch::scan_coefficient_bits(tx_size, tx_type, plane, levels_scan,
                                         area, qctx, is_inter, cdf_slice,
                                         &bits)) {
      return bits;
    }
    int nonzero = 0;
    for (int c = 0; c < eob; ++c) {
      if (levels_scan[c] != 0) ++nonzero;
    }
    return 8.0 + 2.5 * static_cast<double>(nonzero) +
           0.5 * static_cast<double>(eob);
  };
  if (!plan.searched) {
    const int area = plan.adj_w * plan.adj_h;
    if (area <= 0) return plan.eob == 0 ? 1.0 : 0.0;
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    if (!::av2enc::video::txpartition::get_scan(
            plan.tx_size, ::av2enc::video::txpartition::TxScanClass::k2D,
            scan)) {
      return plan.eob == 0 ? 1.0 : 0.0;
    }
    std::vector<std::int32_t> scan_levels(static_cast<std::size_t>(area), 0);
    for (int c = 0; c < area; ++c) {
      const std::size_t pos =
          static_cast<std::size_t>(c) < plan.levels.size()
              ? static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])
              : 0U;
      scan_levels[static_cast<std::size_t>(c)] =
          pos < plan.levels.size() ? plan.levels[pos] : 0;
    }
    double bits = part_bits(plan.tx_size, plan.tx_type, scan_levels.data(),
                            area, plan.eob);
    if (is_inter && plane == 0 && plan.eob > 0) bits += 4.0;
    return bits;
  }
  double bits = plan.parts.size() > 1 ? 3.0 : 0.0;
  for (const TxPlan::Part& part : plan.parts) {
    const int area =
        ::av2enc::video::txpartition::tx_scan_area(part.tx_size);
    bits += part_bits(part.tx_size, part.tx_type, part.levels.data(), area,
                      part.eob);
    if (is_inter && plane == 0 && part.eob > 0) bits += 4.0;
  }
  return bits;
}

}  // namespace

// ---------------------------------------------------------------------------
// AV2ENC_PRED_DUMP: split the luma error into what intra prediction already got
// wrong and what the residual coding left behind.
//
// encode_leaf runs on interior candidate nodes too (a --frames 1 run counted
// 11.6x the frame area), and it may run several times per final leaf, so
// counting calls is meaningless.  Instead the values are written into
// full-frame arrays on every call and the LAST write per pixel wins; the
// coverage mask counts each pixel once and is asserted against the frame area
// at exit.  Recorded recon can then be compared with the recon file, which is
// what makes the numbers checkable rather than merely plausible.
// ---------------------------------------------------------------------------
struct PredDump {
  long frame_pixels = 0;
  long covered = 0;
  double pred_sse = 0.0;
  double recon_sse = 0.0;
  bool allocated = false;
  std::vector<float> src;
  std::vector<float> pred;
  std::vector<float> recon;
  std::vector<unsigned char> hit;
};

inline PredDump& pred_dump() {
  static PredDump d;
  return d;
}

inline bool pred_dump_enabled() {
  // One getenv on the whole run: this is called per leaf, so the flag must be
  // a one-time static rather than a lookup each time.
  static const bool on = [] {
    if (std::getenv("AV2ENC_PRED_DUMP") == nullptr) return false;
    // Construct the accumulator BEFORE registering the handler: destruction
    // runs in reverse construction order, so a handler registered first would
    // otherwise see the object already destroyed and print nothing.
    (void)pred_dump();
    std::atexit([] {
        const PredDump& q = pred_dump();
        const std::size_t n = q.src.size();
        double pe = 0.0;
        double re = 0.0;
        long cov = 0;
        for (std::size_t i = 0; i < n; ++i) {
          if (q.hit.empty() || !q.hit[i]) continue;
          const double dp = static_cast<double>(q.src[i]) -
                            static_cast<double>(q.pred[i]);
          const double dr = static_cast<double>(q.src[i]) -
                            static_cast<double>(q.recon[i]);
          pe += dp * dp;
          re += dr * dr;
          ++cov;
        }
        if (cov == 0) return;
        const bool ok = q.frame_pixels > 0 && cov == q.frame_pixels;
        const double px = static_cast<double>(cov);
        std::fprintf(stderr,
                     "pred-dump leaves_covered=%ld pixels=%ld "
                     "frame_pixels=%ld coverage=%s pred_rmse=%.4f "
                     "recon_rmse=%.4f residual_keeps=%.1f%%\n",
                     cov, cov, q.frame_pixels, ok ? "OK" : "MISMATCH",
                     std::sqrt(pe / px), std::sqrt(re / px),
                     100.0 * std::sqrt(re / pe));
        if (!ok) {
          std::fprintf(stderr,
                       "pred-dump WARNING: coverage != frame area; these "
                       "numbers are not trustworthy\n");
        }
      });
    return true;
  }();
  return on;
}

// Records one leaf's (source, prediction, reconstruction) for every pixel it
// covers; later calls overwrite, so the last write per pixel wins.
inline void pred_dump_record(const VideoInputFrame& frame, int x0, int y0,
                             int w, int h, const std::vector<std::uint8_t>& pred,
                             const std::vector<std::uint8_t>& recon) {
  PredDump& d = pred_dump();
  const std::size_t n =
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  if (pred.size() != n || recon.size() != n) return;
  const int width = static_cast<int>(frame.width);
  const int height = static_cast<int>(frame.height);
  const std::size_t area =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (!d.allocated) {
    d.allocated = true;
    d.frame_pixels = static_cast<long>(area);
    d.src.assign(area, 0.0f);
    d.pred.assign(area, 0.0f);
    d.recon.assign(area, 0.0f);
    d.hit.assign(area, 0);
  }
  if (d.src.size() != area) return;
  for (int yy = 0; yy < h; ++yy) {
    const int y = y0 + yy;
    if (y < 0 || y >= height) continue;
    for (int xx = 0; xx < w; ++xx) {
      const int x = x0 + xx;
      if (x < 0 || x >= width) continue;
      const std::size_t i =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(x);
      d.src[i] = static_cast<float>(source_sample(frame, 0, x, y));
      d.pred[i] = static_cast<float>(pred[static_cast<std::size_t>(yy) *
                                               static_cast<std::size_t>(w) +
                                           static_cast<std::size_t>(xx)]);
      d.recon[i] = static_cast<float>(recon[static_cast<std::size_t>(yy) *
                                             static_cast<std::size_t>(w) +
                                         static_cast<std::size_t>(xx)]);
      if (!d.hit[i]) d.hit[i] = 1;
    }
  }
}

namespace {

// AV2ENC_RD_UVFULL: port the still product's full chroma intra mode search
// (`--still-uv-full`, `Encoder::choose_uv_mode()`) to the video key path.
// Env only, default off: unset (or 0/false/off) leaves every decision and every
// pinned stream byte-identical.
bool rd_uvfull_enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_RD_UVFULL");
    if (env == nullptr || *env == '\0') return false;
    if (std::strcmp(env, "0") == 0) return false;
    const auto equal = [env](const char* word) {
      for (std::size_t i = 0; word[i] != '\0'; ++i) {
        if (std::tolower(static_cast<unsigned char>(env[i])) != word[i]) {
          return false;
        }
      }
      return env[std::strlen(word)] == '\0';
    };
    return !(equal("false") || equal("off") || equal("no"));
  }();
  return value;
}

// AV2ENC_RD_UVFULL_TOPN: chroma candidates re-scored on the real transform
// (default 3).  0 keeps the prediction-only ranking; clamped to 0..8.
int rd_uvfull_top_n() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_RD_UVFULL_TOPN");
    if (env == nullptr || *env == '\0') return 3;
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed < 0 || parsed > 8) return 3;
    return static_cast<int>(parsed);
  }();
  return value;
}

// AV2ENC_COMPOUND_NEWMV enables the §5.20.7.6 compound NEWMV carrier:
// NEAR_NEWMV (list-0 stack + list-1 MVD), NEW_NEARMV (mirror) and
// JOINT_NEWMV (one joint MVD, §5.20.7.13 projMv).  It also makes the exact
// rate probe price the compound symbol set.  Default off: every pinned stream
// is byte-identical.
bool compound_newmv_enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_COMPOUND_NEWMV");
    if (env == nullptr || *env == '\0') return false;
    if (std::strcmp(env, "0") == 0) return false;
    const auto equal = [env](const char* word) {
      for (std::size_t i = 0; word[i] != '\0'; ++i) {
        if (std::tolower(static_cast<unsigned char>(env[i])) != word[i]) {
          return false;
        }
      }
      return env[std::strlen(word)] == '\0';
    };
    return !(equal("false") || equal("off") || equal("no"));
  }();
  return value;
}

// Bitmask of the modes the carrier searches: 1 = NEAR_NEWMV, 2 = NEW_NEARMV,
// 4 = JOINT_NEWMV.  0 keeps the depth-arm candidate set while the exact
// compound pricing stays on.  Default 7.
int compound_newmv_modes() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_COMPOUND_NEWMV_MODES");
    if (env == nullptr || *env == '\0') return 7;
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed < 0 || parsed > 7) return 7;
    return static_cast<int>(parsed);
  }();
  return value;
}

// AV2ENC_COMPOUND_EXACT_RATE: make the SyntaxProbe price the compound symbol
// set exactly (comp_mode / is_joint / compound_mode_non_joint /
// jmvd_scale_mode / DRL / MVD / comp_group_idx / cwp_idx / interp_filter).
// Default on with AV2ENC_COMPOUND_NEWMV (that carrier must be priced
// exactly), otherwise off so every pinned stream is byte-identical.
bool compound_exact_rate_enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_COMPOUND_EXACT_RATE");
    if (env != nullptr) {
      return !(env[0] == '0' && env[1] == '\0');
    }
    return compound_newmv_enabled();
  }();
  return value;
}

// Largest refinement offset in 1/8 luma samples (default 8 = one pel).  The
// search always tries 1/8, 1/4, 1/2 and one pel on each axis up to this bound.
int compound_newmv_radius() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_COMPOUND_NEWMV_RADIUS");
    if (env == nullptr || *env == '\0') return 8;
    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || parsed < 0 || parsed > 16) return 8;
    return static_cast<int>(parsed);
  }();
  return value;
}

// AV2ENC_COMPOUND_NEWMV_REFINE=0 keeps the raw plan ME MV and disables the
// per-ref refinement (ablation).
bool compound_newmv_refine() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_COMPOUND_NEWMV_REFINE");
    if (env == nullptr || *env == '\0') return true;
    return !(env[0] == '0' && env[1] == '\0');
  }();
  return value;
}

// AV2ENC_RD_UVFULL_STATS: sum the winning key-picture chroma coefficient
// information bits (static §8.3 model) and the UVMode histogram.  Diagnostic
// only; it never feeds a decision and is printed at exit.
struct UvFullStats {
  double u_bits = 0.0;
  double v_bits = 0.0;
  int leaves = 0;
  int modes[13] = {};
  int cfl = 0;
};

UvFullStats& uvfull_stats() {
  static UvFullStats stats;
  return stats;
}

void uvfull_print() {
  const UvFullStats& stats = uvfull_stats();
  std::fprintf(stderr,
               "uv-full key leaves=%d cfl=%d u_coef_bits=%.1f "
               "v_coef_bits=%.1f modes=",
               stats.leaves, stats.cfl, stats.u_bits, stats.v_bits);
  for (int i = 0; i < 13; ++i) {
    std::fprintf(stderr, "%s%d", i == 0 ? "" : ",", stats.modes[i]);
  }
  std::fprintf(stderr, "\n");
}

bool uvfull_stats_enabled() {
  static const bool value = [] {
    const char* env = std::getenv("AV2ENC_RD_UVFULL_STATS");
    if (env == nullptr || *env == '\0') return false;
    if (std::strcmp(env, "0") == 0) return false;
    std::atexit(&uvfull_print);
    return true;
  }();
  return value;
}

}  // namespace

bool InterFrameEncoder::decide_sdp(std::string& error) {
  sdpp::Tools tools;
  tools.enable_sdp = true;
  tools.enable_extended_sdp = false;
  tools.frame_is_intra = true;
  tools.single_picture_header_flag = false;

  sdpp::DecisionOptions sopt;
  sopt.min_block_size_px = options_.min_block_size;
  sopt.max_depth = options_.max_partition_depth;
  sopt.allow_rect = true;
  sopt.allow_split = false;
  sopt.allow_3way = false;
  sopt.allow_uneven4 = false;
  sopt.allow_sdp = true;
  sopt.allow_extended_sdp = false;

  const double lambda =
      0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
  const double signal_cost = lambda * 2.0;
  const auto sdp_cost = [this, signal_cost](const sdpp::NodeInfo& info) {
    sdpp::NodeCosts costs;
    if (info.tree_type == sdpp::TreeType::kChromaPart) {
      // Decoder-gated policy: one chroma PARTITION_NONE block per SB.
      costs.leaf = 0.0;
    } else {
      partition::PartitionNodeInfo pinfo;
      pinfo.mi_row = info.mi_row;
      pinfo.mi_col = info.mi_col;
      pinfo.size = info.size;
      pinfo.depth = info.depth;
      pinfo.has_chroma = info.has_chroma;
      pinfo.chroma_offset = info.chroma_offset;
      costs.leaf = static_cast<double>(block_variance_cost_at(pinfo));
    }
    costs.none_signal = 0.0;
    costs.split_signal = sdpp::kInfiniteCost;
    costs.horz_signal = signal_cost;
    costs.vert_signal = signal_cost;
    costs.horz3_signal = sdpp::kInfiniteCost;
    costs.vert3_signal = sdpp::kInfiniteCost;
    costs.horz4a_signal = sdpp::kInfiniteCost;
    costs.horz4b_signal = sdpp::kInfiniteCost;
    costs.vert4a_signal = sdpp::kInfiniteCost;
    costs.vert4b_signal = sdpp::kInfiniteCost;
    costs.region_type[0] = sdpp::kInfiniteCost;
    costs.region_type[1] = sdpp::kInfiniteCost;
    return costs;
  };

  sdp_frame_ = sdpp::decide_frame(geometry_, tools, sdp_cost, sopt);
  if (sdp_frame_.status != sdpp::DecisionStatus::kOk) {
    error = "SDP partition decision failed: " + sdp_frame_.error;
    return false;
  }
  if (!sdpp::validate_frame(geometry_, tools, sdp_frame_, error)) {
    error = "SDP partition tree invalid: " + error;
    return false;
  }

  plan_sdp_pass();
  sb_sinks_.resize(sdp_frame_.superblocks.size());
  for (std::size_t sb = 0; sb < sdp_frame_.superblocks.size(); ++sb) {
    if (!run_one_sb(sb_sinks_[sb], sdp_frame_.superblocks[sb], error)) {
      return false;
    }
  }
  merge_sinks();
  return true;
}

bool InterFrameEncoder::decide(std::string& error) {
  // AV2ENC_TX_DETAIL_SEARCH_TAIL: one picture ordinal per input frame.  The
  // key picture is picture 0, every P picture advances it, the TIP second pass
  // re-decides the same picture, and tiled encodes (tile_w_ < the frame width)
  // leave the state alone.
  const bool whole_picture =
      tile_w_ <= 0 || tile_w_ >= static_cast<int>(source_.width);
  if (!tip_leaf_armed_ && whole_picture) tx_detail::begin_picture();
  // Chained B: the band-gated descent arms on display order, not coding order.
  // -1 on the flat chain keeps the historical picture ordinal.
  tx_detail::display_ordinal() = display_ordinal_;
  if (options_.trace && whole_picture && !tip_leaf_armed_) {
    std::cerr << "  tx-detail picture " << tx_detail::picture_ordinal()
              << " tail=" << (tx_detail::tail_armed() ? 1 : 0)
              << " min=" << tx_detail::config().min_share << "\n";
  }
  if (keyframe_ && options_.enable_sdp) {
    if (!decide_sdp(error)) return false;
    dump_leaf_decisions();
    return true;
  }
  // §5.20.3.1 sub-8 leaves keep the intra keyframe only.  A P frame's inter
  // leaves must own a leaf-sized chroma block, so its search floor stays at
  // 8x8 even when --video-min-block 4 lowers the keyframe floor.
  const int search_min_block =
      keyframe_ ? options_.min_block_size
                : std::max(8, options_.min_block_size);
  partition::DecisionOptions poptions;
  poptions.min_block_size_px = search_min_block;
  poptions.max_depth = options_.max_partition_depth;
  poptions.allow_rect = true;
  // PARTITION_SPLIT only exists at a 128x128 (or 256x256) node.
  poptions.allow_split = sb_size_ > 64;
  if (options_.force_single_block) {
    // One 64x64 leaf per superblock: allow the root split chain but stop any
    // optional split that would create a block below 64x64.
    poptions.max_depth = 8;
    poptions.min_block_size_px = 64;
  }

  // Historical partition cost, kept byte-identical when the opt-in rate-aware
  // tree is off: a source-variance leaf distortion (SSE units) plus a
  // SAD-domain signal cost.  The two do not share a scale, which is the
  // reason the default tree never takes an optional split at high QP.
  const double legacy_lambda =
      0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
  // Rate-aware partition fallback.  It keeps the source variance (a
  // QP-independent proxy for the energy a leaf cannot predict) and charges a
  // rate term in the same SSE units:
  //   leaf  = variance + price * leaf_bits       (mode/MV/skip syntax)
  //   split = price * 2.0                        (partition symbol bits)
  // This is the whole cost for the keyframe and for P-frame leaves the
  // measured probe cannot model (sub-8, right/bottom edges, ME failure), and
  // it is the only cost when `partition_rd` is off (with the historical
  // SAD-domain price).  `price` is one estimated syntax bit in SSE units;
  // 700 with the 10-bit inter leaf term and 2-bit partition symbol gives a
  // ~8,400-SSE split penalty, calibrated from the measured local RD slope of
  // this clip at the ~100 kbps point (4,000-6,300 SSE/bit over QP 168..190;
  // see docs/video-rd-preset.md).  The measured probe supersedes it on the
  // P-frame path.
  constexpr double kPartitionSsePerSyntaxBit = 700.0;
  constexpr double kPartitionSignalBits = 2.0;
  constexpr double kPartitionLeafBitsInter = 10.0;
  constexpr double kPartitionLeafBitsIntra = 6.0;
  double syntax_bit_price =
      options_.partition_rd
          ? (options_.qp_lambda_scale
                 ? partition_model_price_at(options_, current_qindex_)
                 : kPartitionSsePerSyntaxBit)
          : legacy_lambda;
  const double signal_bits = kPartitionSignalBits;
  double leaf_bits = options_.partition_rd
                         ? (keyframe_ ? kPartitionLeafBitsIntra
                                      : kPartitionLeafBitsInter)
                         : 0.0;
  // Measured partition probe.  It replaces the variance leaf proxy with one
  // NEWMV prediction plus one quantised luma residual for every candidate
  // node, and is bounded to the first split level.  It prices leaves only,
  // so it is independent of which solver reaches them: the extended tree
  // gets exactly the cost the rect tree charges a same-shaped child.
  const bool partition_probe =
      options_.partition_rd && !keyframe_ && !options_.force_single_block &&
      !options_.enable_segmentation &&
      width_ == static_cast<int>(source_.width) &&
      height_ == static_cast<int>(source_.height);
  std::unordered_map<std::uint64_t, double> probe_costs;
  if (partition_probe) {
    syntax_bit_price = probe_syntax_price(0, 0);
    // The probe's bits are real (mode/skip syntax plus the residual rate),
    // so the leaf-level syntax estimate is inside the measured cost.
    leaf_bits = 0.0;
    poptions.max_depth =
        std::min(poptions.max_depth,
                 std::clamp(options_.partition_probe_depth, 1, 3));
    // AV2ENC_TX_DETAIL_SEARCH_TAIL: the measured probe clamps the P tree to
    // its first-level bound, which cannot reach a 16x16 coding block.  The
    // armed tail opens the four rect levels (64 -> 32 -> 16) it needs; the
    // per-node band gate below decides which nodes actually descend.
    if (tx_detail::band_descent_armed()) {
      poptions.max_depth =
          std::min(options_.max_partition_depth, 4);
    }
    // AV2ENC_RD_TAIL_ACT: one level deeper than the band gate's bound.  A
    // rect-path 16x16 coding block can sit at depth 4 (e.g. 64 -> 64x32 ->
    // 32x32 -> 32x16 -> 16x16), so the sub-16 descent needs depth 5 or it can
    // never fire on the blocks it is meant for.  Still below the CLI cap's
    // legal range and below the encoder's 8x8 inter floor.
    if (!keyframe_ && tail_tx::tail_armed(arming_ordinal()) &&
        tail_tx::config().act > 0.0) {
      poptions.max_depth = std::max(
          poptions.max_depth,
          std::min(options_.max_partition_depth + 1, 6));
    }
    probe_costs = build_partition_probe_costs(
        poptions.max_depth,
        std::max(poptions.min_block_size_px, partition::kMinBlockPixels));
  }
  const double signal_cost = syntax_bit_price * signal_bits;
  // AV2ENC_EXACT_RD partition refinement: when non-null, a node's leaf cost
  // comes from the measured (D, R) of the mode the leaf encoder actually chose
  // in the first pass.  Nodes the first pass did not encode (the alternative
  // side of a depth-1 split) keep the NEWMV probe as the early bound.  The
  // pointer is only ever set while a refinement solve runs, so the historical
  // probe-only path is bit-for-bit unchanged when the gate is off.
  const std::unordered_map<std::uint64_t, double>* exact_costs = nullptr;
  // AV2ENC_TX_DETAIL_SEARCH_TAIL: the band-gated descent.  `detail_descent`
  // is only true on an armed picture (the tail window, or the head window under
  // AV2ENC_RD_HEAD_PART); `detail_max_depth` is the depth the probe was just
  // built for, and `detail_min_share` the swept threshold.
  const bool detail_descent = !keyframe_ && tx_detail::band_descent_armed();
  const int detail_max_depth = poptions.max_depth;
  const double detail_min_share = tx_detail::config().min_share;
  // AV2ENC_RD_TAIL_ACT: the tail residual's sub-16 coding-block gate.  When
  // armed, a 16x16 node whose source high-pass activity clears the threshold
  // may not stay a coding block, so its residual can reach the legal 8x8 /
  // 8x4 / 4x4 transform partitions of an 8x8-class leaf.  The driver's
  // AV2ENC_RD_DRIVER_ACT is the same screen on the key picture.
  const bool tail_act_descent =
      !keyframe_ && tail_tx::tail_armed(arming_ordinal()) &&
      tail_tx::config().act > 0.0;
  const double tail_act = tail_tx::config().act;
  // One leaf model for both solvers.  It is exactly the historical keyframe
  // / P-frame estimate (variance + price * leaf_bits) or, when the measured
  // probe is active, the probe's measured residual RD on the node's own
  // geometry.  A 64x16 / 16x64 extended leaf is therefore priced by the same
  // code path that prices a NONE/HORZ/VERT/SPLIT child of the same shape,
  // including its leaf coder's transform size and chroma ownership.
  const auto partition_leaf_cost =
      [this, syntax_bit_price, leaf_bits, partition_probe, &probe_costs,
       &exact_costs, detail_descent, detail_max_depth,
       detail_min_share, tail_act_descent, tail_act](
          const partition::PartitionNodeInfo& info) {
        // The leaf coder handles coding blocks up to 64x64.  A larger node
        // must split (the 128/256 root and its rectangular children); the
        // spec never implies NONE at those sizes, so the solver always has a
        // partition.
        const int bw = partition::block_width(info.size);
        const int bh = partition::block_height(info.size);
        if (tail_act_descent && bw == 16 && bh == 16) {
          ++tail_tx::counters().act_nodes16;
        }
        if (bw > 64 || bh > 64) return partition::kInfiniteCost;
        // AV2ENC_TX_DETAIL_SEARCH_TAIL: in the armed tail, a node that is
        // strictly larger than 16x16 on at least one axis and carries the
        // 2-4 px source band is not allowed to stay a coding block.  The rect
        // solver then has to split it, so a band-rich region descends to the
        // 16x16 block whose legal §5.20.6.3 SPLIT is four 8x8 transforms.  The
        // gate is skipped when the remaining depth cannot place a 16x16 leaf
        // or when the spec allows no rect partition, so it can never make a
        // node unsolvable.
        if (detail_descent && bw >= 16 && bh >= 16 &&
            (bw > 16 || bh > 16) && info.depth + 1 <= detail_max_depth &&
            (info.allowed.is_allowed(partition::PartitionType::kHorz) ||
             info.allowed.is_allowed(partition::PartitionType::kVert))) {
          const partition::BlockRect rect = partition::block_rect(
              geometry_, info.mi_row, info.mi_col, info.size);
          if (rect.visible_width > 0 && rect.visible_height > 0 &&
              tx_detail::source_band_share(source_, tile_x_ + rect.x,
                                           tile_y_ + rect.y,
                                           rect.visible_width,
                                           rect.visible_height) >=
                  detail_min_share) {
            return partition::kInfiniteCost;
          }
        }
        // AV2ENC_RD_TAIL_ACT: the same forced descent one level deeper.  A
        // 16x16 node with real high-pass activity should not stay a coding
        // block, which is what makes an 8x8-class leaf (and therefore a legal
        // 4x4/8x4 transform partition) reachable in the tail.  Flat sky keeps
        // the 16x16 block: the screen is the source, not the measured D.  The
        // NONE candidate is priced with a large finite penalty rather than
        // kInfiniteCost: a depth-5 16x8/8x16 child can be outside what the
        // leaf coder accepts at the picture edge, and dropping NONE entirely
        // then leaves the node with no partition at all.
        bool tail_force_split = false;
        if (tail_act_descent && bw == 16 && bh == 16 &&
            info.depth + 1 <= detail_max_depth &&
            (info.allowed.is_allowed(partition::PartitionType::kHorz) ||
             info.allowed.is_allowed(partition::PartitionType::kVert))) {
          ++tail_tx::counters().act_nodes;
          const partition::BlockRect rect = partition::block_rect(
              geometry_, info.mi_row, info.mi_col, info.size);
          if (rect.visible_width > 0 && rect.visible_height > 0 &&
              tail_tx::source_activity(source_.y.data(),
                                       static_cast<int>(source_.width),
                                       static_cast<int>(source_.height),
                                       static_cast<int>(source_.width),
                                       tile_x_ + rect.x, tile_y_ + rect.y,
                                       rect.visible_width,
                                       rect.visible_height) >= tail_act) {
            ++tail_tx::counters().act_fired;
            tail_force_split = true;
          }
        }
        const double force_penalty =
            tail_force_split ? 1.0e9 : 0.0;
        double leaf = static_cast<double>(block_variance_cost_at(info)) +
                      syntax_bit_price * leaf_bits + force_penalty;
        if (partition_probe) {
          const std::uint64_t key =
              partition_probe_key(info.mi_row, info.mi_col, info.size);
          if (exact_costs != nullptr) {
            const auto exact = exact_costs->find(key);
            if (exact != exact_costs->end()) {
              return exact->second + force_penalty;
            }
          }
          const auto it = probe_costs.find(key);
          if (it != probe_costs.end()) {
            leaf = it->second + force_penalty;
          } else {
            double measured = 0.0;
            if (partition_probe_leaf_cost(
                    info, probe_syntax_price(info.mi_row, info.mi_col),
                    measured)) {
              leaf = measured + force_penalty;
            }
          }
        }
        return leaf;
      };
  const auto cost_fn = [partition_leaf_cost, signal_cost](
                           const partition::PartitionNodeInfo& info) {
    partition::NodeCosts costs;
    costs.leaf = partition_leaf_cost(info);
    costs.none_signal = 0.0;
    costs.split_signal = signal_cost;
    costs.horz_signal = signal_cost;
    costs.vert_signal = signal_cost;
    return costs;
  };

  std::vector<PartitionNode> base_superblocks;
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
  std::vector<pe::PartitionNode> ext_superblocks;
#endif
  // Re-runnable so the exact partition refinement can solve the same tree
  // with measured leaf costs.  Gate off: called exactly once, as before.
  const auto solve_partition = [&]() -> bool {
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
  if (options_.ext_partitions) {
    pe::Tools tools;
    tools.enable_ext_partitions = true;
    tools.enable_uneven_4way_partitions = true;
    pe::DecisionOptions eopt;
    // Mirror the rect solver's bound exactly.  With --video-partition-rd the
    // probe clamps `poptions.max_depth` to the measured first level; the
    // extended tree must search the same depth or it would compare a deeper
    // tree against the shallow rect one and win on tiny leaves alone.
    eopt.min_block_size_px = poptions.min_block_size_px;
    eopt.max_depth = poptions.max_depth;
    eopt.allow_rect = true;
    eopt.allow_split = poptions.allow_split;
    eopt.allow_3way = true;
    eopt.allow_uneven4 = true;
    eopt.tie_break = pe::TieBreakOrder::kRectCompatOrder;
    const auto ext_cost_fn = [this, partition_leaf_cost, signal_cost](
                                 const pe::NodeInfo& info) {
      pe::NodeCosts costs;
      const int bw = partition::block_width(info.size);
      const int bh = partition::block_height(info.size);
      partition::PartitionNodeInfo binfo;
      binfo.mi_row = info.mi_row;
      binfo.mi_col = info.mi_col;
      binfo.size = static_cast<partition::BlockSize>(info.size);
      // Same leaf model the rect solver uses for a NONE/HORZ/VERT/SPLIT
      // child of this geometry: the syntax estimate (variance + leaf rate)
      // or the measured probe RD, selected by the same predicate.
      costs.leaf = (bw <= 64 && bh <= 64)
                       ? partition_leaf_cost(binfo)
                       : pe::kInfiniteCost;
      costs.none_signal = 0.0;
      costs.split_signal = signal_cost;
      costs.horz_signal = signal_cost;
      costs.vert_signal = signal_cost;
      // Keep every child >= 8x8 so the video leaf coder keeps owning its
      // chroma block (no shared ChromaMi).  HORZ_4A/B and VERT_4A/B emit an
      // 8x16 / 16x8 minimum slice, so they need the 64-pixel axis.
      costs.horz3_signal =
          (bh >= 32 && bw >= 8 && !(bw == 8 && bh == 32) &&
           !pe::chroma_offset_for_partition(
               geometry_, pe::PartitionType::kHorz3, info.size))
              ? signal_cost
              : pe::kInfiniteCost;
      costs.vert3_signal =
          (bw >= 32 && bh >= 8 && !(bw == 32 && bh == 8) &&
           !pe::chroma_offset_for_partition(
               geometry_, pe::PartitionType::kVert3, info.size))
              ? signal_cost
              : pe::kInfiniteCost;
      const double h4 = (bh >= 64 && bw >= 8) ? signal_cost
                                              : pe::kInfiniteCost;
      const double v4 = (bw >= 64 && bh >= 8) ? signal_cost
                                              : pe::kInfiniteCost;
      costs.horz4a_signal = h4;
      costs.horz4b_signal = h4;
      costs.vert4a_signal = v4;
      costs.vert4b_signal = v4;
      return costs;
    };
    const pe::FrameDecision frame =
        pe::decide_frame(geometry_, ext_cost_fn, tools, eopt);
    if (frame.status != pe::DecisionStatus::kOk) {
      error = "extended partition decision failed: " + frame.error;
      return false;
    }
    if (!pe::validate_frame(geometry_, tools, frame, error)) {
      error = "extended partition tree invalid: " + error;
      return false;
    }
    ext_superblocks = frame.superblocks;
    if (FILE* tree_file = part_tree_file()) {
      for (const auto& sb : ext_superblocks) {
        dump_partition_tree_ext(sb, tree_file);
      }
      std::fflush(tree_file);
    }
  } else
#endif
  {
    const partition::FrameDecision frame =
        partition::decide_frame(geometry_, cost_fn, poptions);
    if (frame.status != partition::DecisionStatus::kOk) {
      error = "partition decision failed: " + frame.error;
      return false;
    }
    if (!partition::validate_frame(geometry_, frame, error)) {
      error = "partition tree invalid: " + error;
      return false;
    }
    base_superblocks = frame.superblocks;
    if (FILE* tree_file = part_tree_file()) {
      for (const auto& sb : base_superblocks) {
        dump_partition_tree(sb, tree_file);
      }
      std::fflush(tree_file);
    }
  }
  return true;
  };
  if (!solve_partition()) return false;

  // The partition tree is final, so the pure NEWMV searches can run ahead
  // of the coding-order walk on the host worker pool.  Serial at 1 thread.
  if (!base_superblocks.empty()) {
    joint_plan::ScopedTimer me_timer(&joint_plan::profile().me);
    precompute_leaf_me(base_superblocks, options_.cpu_threads);
  }

#if defined(AV2ENC_VIDEO_DELTAQ)
  if (!prepare_qp_plan(error)) return false;
#endif
  // One decision pass: frame-level plan, then one sink per superblock, then
  // the offset rebase.  `run_superblocks()` is the only place the schedule
  // lives: raster serial at --cpu-threads 1, the skew-2 wavefront above.
  const auto collect_frame = [&]() -> bool {
    plan_pass();
    // AV2ENC_RD_DRIVER: one superblock at a time on the key picture.  The
    // frame-wide tree above is only a fallback/reference; here every
    // superblock's layout is re-decided against the reconstruction the
    // previous superblocks actually built, then committed through the normal
    // collect_node()/encode_leaf() walk so recon + §6.19.2.3 BlockDecoded
    // advance before the next decision.
    // AV2ENC_RD_JOINT_P: the measured P-frame joint tree + TX/levels solve.
    // Same surface restrictions as the key/tail sequential walks; it is
    // mutually exclusive with the tail floor and the diagnostic, which it
    // subsumes (it prices those nodes with the same objective and can also
    // collapse ancestors).
    bool joint_p =
        !keyframe_ && !joint_plan_off_ && joint_plan::armed(arming_ordinal()) &&
        !options_.enable_sdp && !options_.force_single_block &&
        !palette_enabled_ && !ibc_enabled_ && !qp_tools_active() &&
        !options_.enable_tcq && !options_.tip_block_ref &&
        (tile_w_ <= 0 || tile_w_ >= static_cast<int>(source_.width)) &&
        (tile_h_ <= 0 || tile_h_ >= static_cast<int>(source_.height)) &&
        pipe_probe_ready_;
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    joint_p = joint_p && !options_.ext_partitions;
#endif
    bool sequential_key =
        keyframe_ && rd_driver::enabled() && !options_.enable_sdp &&
        !options_.force_single_block && !palette_enabled_ && !ibc_enabled_ &&
        !qp_tools_active();
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    // The driver walks the rect tree; the extended solver has no
    // base_superblocks to drive.
    sequential_key = sequential_key && !options_.ext_partitions;
#endif
    // AV2ENC_RD_TAIL_FLOOR: the tail P pictures, whole-picture serial walk.
    // The same surface restrictions as the key driver, because the trial
    // snapshots restore the reconstruction, the §6.19.2.3 maps and the probe
    // state but deliberately not the palette/IBC/segment grids the nightly
    // recipe keeps off.
    bool sequential_tail =
        !keyframe_ && tail_floor::armed(tx_detail::picture_ordinal()) &&
        !options_.enable_sdp && !options_.force_single_block &&
        !palette_enabled_ && !ibc_enabled_ && !qp_tools_active() &&
        (tile_w_ <= 0 || tile_w_ >= static_cast<int>(source_.width)) &&
        (tile_h_ <= 0 || tile_h_ >= static_cast<int>(source_.height)) &&
        pipe_probe_ready_;
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    sequential_tail = sequential_tail && !options_.ext_partitions;
#endif
    // AV2ENC_PART_DIAG: the same sequential walk, used only to measure both
    // sides of each P leaf.  It commits the reference tree, so the stream is
    // unchanged; it exists to size the P-partition pool before a driver is
    // built.  Mutually exclusive with the tail floor's measured descent.
    bool sequential_diag =
        !keyframe_ && part_diag::armed(tx_detail::picture_ordinal()) &&
        !sequential_tail && !options_.enable_sdp &&
        !options_.force_single_block && !palette_enabled_ && !ibc_enabled_ &&
        !qp_tools_active() &&
        (tile_w_ <= 0 || tile_w_ >= static_cast<int>(source_.width)) &&
        (tile_h_ <= 0 || tile_h_ >= static_cast<int>(source_.height)) &&
        pipe_probe_ready_;
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    sequential_diag = sequential_diag && !options_.ext_partitions;
#endif
    tail_floor_active_ = sequential_tail;
    tail_floor_trial_ = false;
    part_diag_active_ = sequential_diag;
    part_diag_trial_ = false;
    if (joint_p) {
      tail_floor_active_ = false;
      part_diag_active_ = false;
      if (joint_plan::config().batch > 0) {
        joint_plan::ScopedTimer timer(&joint_plan::profile().solve);
        return joint_p_batch_frame(base_superblocks, error);
      }
      joint_p_active_ = true;
      sb_sinks_.resize(base_superblocks.size());
      for (std::size_t sb = 0; sb < base_superblocks.size(); ++sb) {
        sb_sinks_[sb].reset();
        seed_unit_state(sb_sinks_[sb]);
        PartitionNode chosen;
        {
          joint_plan::ScopedTimer timer(&joint_plan::profile().solve);
          if (!joint_p_decide_sb(sb, base_superblocks[sb], &chosen, error)) {
            joint_p_active_ = false;
            return false;
          }
        }
        base_superblocks[sb] = std::move(chosen);
        {
          const std::size_t before = sb_sinks_[sb].blocks.size();
          joint_plan::ScopedTimer timer(&joint_plan::profile().commit);
          if (!collect_node(sb_sinks_[sb], base_superblocks[sb], error)) {
            joint_p_active_ = false;
            return false;
          }
          for (std::size_t i = before; i < sb_sinks_[sb].blocks.size(); ++i) {
            const syntax::BlockSyntax& entry = sb_sinks_[sb].blocks[i];
            if (entry.partition != static_cast<std::uint8_t>(
                                       PartitionType::kNone) ||
                entry.subtree != 0) {
              continue;
            }
            joint_plan::profile_bump(&joint_plan::profile().committed_leaves);
            if (entry.is_inter == 0) {
              joint_plan::profile_bump(&joint_plan::profile().committed_intra);
            }
          }
        }
      }
      joint_p_active_ = false;
      merge_sinks();
      return true;
    }
    if (sequential_key) {
      sb_sinks_.resize(base_superblocks.size());
      for (std::size_t sb = 0; sb < base_superblocks.size(); ++sb) {
        PartitionNode chosen;
        if (!seq_driver_decide_sb(sb, base_superblocks[sb], &chosen, error)) {
          return false;
        }
        base_superblocks[sb] = std::move(chosen);
        sb_sinks_[sb].reset();
        seed_unit_state(sb_sinks_[sb]);
        if (!collect_node(sb_sinks_[sb], base_superblocks[sb], error)) {
          return false;
        }
      }
      merge_sinks();
      return true;
    }
    // AV2ENC_RD_TAIL_FLOOR: the same sequential walk on the armed tail
    // pictures, with the measured 16x16 -> 8x8-class coding-block descent.
    // Every superblock is committed (recon + §6.19.2.3 BlockDecoded + the
    // probe state) before the next one is decided, so a candidate is measured
    // in the reconstruction its own layout and every earlier coded block
    // built.  Nothing here runs on the key picture or outside pictures
    // 33-59.
    if (sequential_tail || sequential_diag) {
      sb_sinks_.resize(base_superblocks.size());
      for (std::size_t sb = 0; sb < base_superblocks.size(); ++sb) {
        sb_sinks_[sb].reset();
        seed_unit_state(sb_sinks_[sb]);
        PartitionNode chosen;
        tail_floor_sb_root_ = &base_superblocks[sb];
        part_diag_sb_root_ = &base_superblocks[sb];
        const bool ok = collect_node(sb_sinks_[sb], base_superblocks[sb],
                                     error, ChromaInfo{}, &chosen);
        tail_floor_sb_root_ = nullptr;
        part_diag_sb_root_ = nullptr;
        if (!ok) {
          tail_floor_active_ = false;
          part_diag_active_ = false;
          return false;
        }
        base_superblocks[sb] = std::move(chosen);
      }
      tail_floor_active_ = false;
      tail_floor_trial_ = false;
      part_diag_active_ = false;
      part_diag_trial_ = false;
      merge_sinks();
      return true;
    }
    tail_floor_active_ = false;
    part_diag_active_ = false;
#if defined(AV2ENC_VIDEO_EXT_PARTITIONS)
    if (options_.ext_partitions) {
      sb_sinks_.resize(ext_superblocks.size());
      if (!run_superblocks(ext_superblocks, error)) return false;
    } else
#endif
    {
      sb_sinks_.resize(base_superblocks.size());
      if (!run_superblocks(base_superblocks, error)) return false;
    }
    merge_sinks();
    return true;
  };

  // AV2ENC_EXACT_RD partition refinement (one pass).  The historical probe
  // prices a node with one NEWMV + quantised residual before the mode search.
  // The first encode pass now records the real (D, R) of the mode each leaf
  // actually chose (encode_leaf -> sink.leaf_costs).  Re-solve the tree with
  // those exact costs where the node was encoded, keeping the probe only for
  // the alternative geometry the first pass did not visit (the early bound).
  // If the tree changes, encode once more with it.  This is deliberately not
  // iterated: a full exact cost per partition node would need every candidate
  // layout encoded, and the one refinement pass captures the dominant error
  // (the probe's NEWMV-only residual vs the leaf's chosen mode) at <=2x cost.
  const auto leaf_signature =
      [](const std::vector<PartitionNode>& superblocks) {
        std::vector<std::uint64_t> leaves;
        const auto walk = [&leaves](auto&& self,
                                    const PartitionNode& node) -> void {
          if (node.partition != partition::PartitionType::kNone) {
            for (const PartitionNode& child : node.children) {
              self(self, child);
            }
            return;
          }
          leaves.push_back(
              (static_cast<std::uint64_t>(node.mi_row) & 0xFFFFU) |
              ((static_cast<std::uint64_t>(node.mi_col) & 0xFFFFU) << 16U) |
              ((static_cast<std::uint64_t>(node.size) & 0xFFU) << 32U) |
              ((static_cast<std::uint64_t>(node.depth) & 0xFFU) << 40U));
        };
        for (const PartitionNode& sb : superblocks) walk(walk, sb);
        return leaves;
      };
  bool exact_partition_done = false;
  const auto collect_frame_refined = [&]() -> bool {
    if (!collect_frame()) return false;
    if (!exact_rd::partition_refine() || !partition_probe ||
        exact_partition_done || options_.ext_partitions ||
        qp_tools_active()) {
      return true;
    }
    exact_partition_done = true;
    std::unordered_map<std::uint64_t, double> measured;
    for (const SbSink& sink : sb_sinks_) {
      const std::size_t count = std::min(sink.blocks.size(),
                                         sink.leaf_costs.size());
      for (std::size_t i = 0; i < count; ++i) {
        const syntax::BlockSyntax& block = sink.blocks[i];
        const exact_rd::LeafCost& leaf = sink.leaf_costs[i];
        if (!leaf.valid) continue;
        if (block.partition !=
            static_cast<std::uint8_t>(partition::PartitionType::kNone)) {
          continue;
        }
        if (block.subtree != 0) continue;
        const partition::BlockSize size =
            static_cast<partition::BlockSize>(block.bsize);
        const double lambda =
            probe_syntax_price(block.mi_row, block.mi_col);
        measured[partition_probe_key(block.mi_row, block.mi_col, size)] =
            leaf.distortion + lambda * leaf.bits;
      }
    }
    if (measured.empty()) return true;
    ++exact_rd::counters().partition_refine_runs;
    exact_rd::counters().partition_overrides += measured.size();
    std::vector<PartitionNode> probe_tree = base_superblocks;
    exact_costs = &measured;
    const bool solved = solve_partition();
    exact_costs = nullptr;
    if (!solved) return false;
    if (leaf_signature(probe_tree) == leaf_signature(base_superblocks)) {
      base_superblocks = std::move(probe_tree);
      return true;
    }
    ++exact_rd::counters().partition_tree_changed;
    if (!base_superblocks.empty()) {
      precompute_leaf_me(base_superblocks, options_.cpu_threads);
    }
    return collect_frame();
  };
#if defined(AV2ENC_VIDEO_DELTAQ)
  if (qp_tools_active()) {
    if (delta_q_enabled_) {
      std::vector<int> effective(
          static_cast<std::size_t>(qp_sb_count()), base_q_index());
      bool converged = false;
      for (int pass = 0; pass < 8; ++pass) {
        sb_qindex_ = effective;
        if (!collect_frame_refined()) return false;
        std::vector<int> actual;
        if (!plan_delta_q_steps(actual, error)) return false;
        if (actual == effective) {
          converged = true;
          break;
        }
        effective = std::move(actual);
      }
      if (!converged) {
        // A quantizer-dependent skip decision can oscillate for a full-SB
        // leaf (the delta is only codable while the block is not a full-SB
        // skip).  Fall back to the neutral ramp: every step is zero, so the
        // stream stays exact whatever the skip decisions are.
        sb_target_q_.assign(static_cast<std::size_t>(qp_sb_count()),
                            base_q_index());
        sb_qindex_.assign(static_cast<std::size_t>(qp_sb_count()),
                          base_q_index());
        if (!collect_frame_refined()) return false;
        std::vector<int> actual;
        if (!plan_delta_q_steps(actual, error)) return false;
        if (actual != sb_qindex_) {
          error = "delta-Q plan: neutral fallback did not produce a zero "
                  "ramp";
          return false;
        }
      }
    } else if (!collect_frame_refined()) {
      return false;
    }
  } else if (!collect_frame_refined()) {
    return false;
  }
#else
  if (!collect_frame_refined()) return false;
#endif
  // --video-skip-collapse: rewrite all-static superblocks to one skip leaf
  // and re-collect.  Default off (returns immediately).
  if (!collapse_static_superblocks(base_superblocks, poptions, collect_frame,
                                   error)) {
    return false;
  }
  dump_leaf_decisions();
  return true;
}

bool InterFrameEncoder::plan_sdp_symbols(const sdpp::Node& node, syntax::BlockSyntax& entry,
                      std::string& error) {
  sdpp::Node info;
  info.mi_row = node.mi_row;
  info.mi_col = node.mi_col;
  info.size = node.size;
  info.tree_type = node.tree_type;
  info.partition = node.partition;
  info.has_chroma = node.has_chroma;
  info.chroma_offset = node.chroma_offset;
  info.chroma_partition_known = node.chroma_partition_known;
  info.implied_luma_partition = node.implied_luma_partition;
  sdpp::SymbolContexts contexts;
  const sdpp::MiSizeGrid& grid =
      node.tree_type == sdpp::TreeType::kChromaPart ? sdp_chroma_sizes_
                                                    : sdp_luma_sizes_;
  contexts = sdpp::node_contexts(geometry_, info, grid);
  const sdpp::SymbolPlan plan = sdpp::plan_partition_symbols(
      node.boundary, node.allowed, node.size, node.tree_type, node.partition,
      contexts, /*bru_active=*/true);
  if (!plan.ok) {
    error = "SDP symbol plan at " + std::to_string(node.mi_row) + "," +
            std::to_string(node.mi_col) + ": " + plan.error;
    return false;
  }
  if (plan.symbols.size() > 4) {
    error = "SDP node codes more than four partition symbols";
    return false;
  }
  entry.sdp_symbol_count = static_cast<std::uint8_t>(plan.symbols.size());
  for (std::size_t i = 0; i < plan.symbols.size(); ++i) {
    const sdpp::Symbol& symbol = plan.symbols[i];
    entry.sdp_symbols[i].kind = static_cast<std::uint8_t>(symbol.kind);
    entry.sdp_symbols[i].plane = static_cast<std::uint8_t>(symbol.plane);
    entry.sdp_symbols[i].context =
        static_cast<std::uint8_t>(symbol.context);
    entry.sdp_symbols[i].value = static_cast<std::uint8_t>(symbol.value);
  }
  return true;
}

bool InterFrameEncoder::collect_sdp_node(SbSink& sink, const sdpp::Node& node,
                      std::string& error) {
  SbSink::State& st = sink.state;
  if (node.depth == 0) {
    // §6.19.2.3 clear_block_decoded_flags(): superblock-local.
    st.decoded[0].reset(mi_cols_, mi_rows_);
    st.decoded[1].reset(mi_cols_ / 2, mi_rows_ / 2);
    st.decoded[2].reset(mi_cols_ / 2, mi_rows_ / 2);
    reset_refmvbank_for_sb(node.mi_row, node.mi_col);
  }
  const std::size_t index = sink.blocks.size();
  sink.blocks.emplace_back();
  syntax::BlockSyntax& entry = sink.blocks.back();
  entry.levels_offset = kNoLevels;
  entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
  entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
  entry.bsize = static_cast<std::uint8_t>(node.size);
  entry.partition = static_cast<std::uint8_t>(node.partition);
  entry.tree_type = static_cast<std::uint8_t>(node.tree_type);
  entry.has_chroma = node.codes_chroma ? 1 : 0;
  entry.chroma_offset = node.chroma_offset ? 1 : 0;
  entry.chroma_bsize =
      node.chroma.size == BlockSize::kInvalid
          ? 0
          : static_cast<std::uint8_t>(node.chroma.size);
  entry.chroma_mi_row = static_cast<std::uint16_t>(node.chroma.mi_row);
  entry.chroma_mi_col = static_cast<std::uint16_t>(node.chroma.mi_col);
  entry.chroma_partition_known = node.chroma_partition_known ? 1 : 0;
  entry.implied_luma_partition =
      static_cast<std::uint8_t>(node.implied_luma_partition);
  entry.flags |= syntax::kFlagHasChromaInfo;
  std::uint16_t mask = 0;
  for (int p = 0; p < sdpp::kPartitionCount; ++p) {
    if (node.allowed.allowed[p]) {
      mask = static_cast<std::uint16_t>(mask | (1U << p));
    }
  }
  entry.allowed_mask = mask;
  if (node.forced) entry.flags |= syntax::kFlagForced;
  if (node.depth == 0) entry.flags |= syntax::kFlagSbRoot;
  if (node.sdp_split) entry.flags |= syntax::kFlagSdpSplit;

  if (node.sdp_split) {
    if (node.children.size() != 2) {
      error = "SDP split node does not have exactly two tree children";
      return false;
    }
    for (const sdpp::Node& child : node.children) {
      if (!collect_sdp_node(sink, child, error)) return false;
    }
  } else if (node.partition == sdpp::PartitionType::kNone) {
    if (!plan_sdp_symbols(node, entry, error)) return false;
    PartitionNode pnode;
    pnode.mi_row = node.mi_row;
    pnode.mi_col = node.mi_col;
    pnode.size = node.size;
    pnode.partition = PartitionType::kNone;
    pnode.depth = node.depth;
    pnode.has_chroma = node.has_chroma;
    pnode.chroma_offset = node.chroma_offset;
    const std::size_t dump_level_begin = sink.levels.size();
    if (!encode_leaf(
            sink, pnode, index, error, static_cast<int>(node.tree_type),
            node.codes_chroma, node.chroma.mi_row, node.chroma.mi_col,
            node.chroma.size == BlockSize::kInvalid
                ? -1
                : static_cast<int>(node.chroma.size))) {
      return false;
    }
    record_leaf_dump(sink, sink.blocks[index], dump_level_begin);
    update_refmvbank_leaf(sink.blocks[index]);
    if (node.tree_type != sdpp::TreeType::kChromaPart) {
      sdp_luma_sizes_.fill_block(node.mi_row, node.mi_col, node.size);
    }
    if (node.codes_chroma) {
      sdp_chroma_sizes_.fill_block(node.chroma.mi_row, node.chroma.mi_col,
                                   node.chroma.size);
    }
  } else {
    if (!plan_sdp_symbols(node, entry, error)) return false;
    for (const sdpp::Node& child : node.children) {
      if (!collect_sdp_node(sink, child, error)) return false;
    }
  }
  const std::size_t subtree = sink.blocks.size() - index - 1;
  if (subtree > 0xFFFFU) {
    error = "partition subtree too large";
    return false;
  }
  sink.blocks[index].subtree = static_cast<std::uint16_t>(subtree);
  return true;
}

void InterFrameEncoder::record_leaf_dump(SbSink& sink, const syntax::BlockSyntax& entry,
                      std::size_t level_begin) {
  if (!block_dump_enabled()) return;
  LeafDumpRecord record;
  record.mi_row = entry.mi_row;
  record.mi_col = entry.mi_col;
  record.bsize = entry.bsize;
  record.partition = entry.partition;
  record.is_inter = entry.is_inter;
  record.mode = entry.is_inter ? entry.inter_mode : entry.y_mode;
  record.ref = entry.ref_index;
  record.drl_index = entry.drl_index;
  record.mv_x = entry.mv_x;
  record.mv_y = entry.mv_y;
  record.skip = entry.skip_flag;
  for (int plane = 0; plane < 3; ++plane) record.eob[plane] = entry.eob[plane];
  record.levels_digest = hash_levels(sink.levels, level_begin,
                                     sink.levels.size());
  record.palette_index = entry.palette_index;
  if (entry.palette_index != syntax::BlockSyntax::kNoPalette &&
      entry.palette_index < sink.palettes.size()) {
    const palette::PaletteResult& palette =
        sink.palettes[entry.palette_index];
    record.palette_size = palette.has_palette ? palette.size : 0;
    record.palette_direction = palette.direction;
    for (int i = 0; i < record.palette_size && i < 8; ++i) {
      record.palette_colors[static_cast<std::size_t>(i)] =
          palette.colors[static_cast<std::size_t>(i)];
    }
  }
  sink.leaf_dump.push_back(record);
}

void InterFrameEncoder::dump_leaf_decisions() {
  FILE* file = block_dump_file();
  if (file == nullptr) return;
  // Pre-order leaf order is raster order for a quadtree, so this sort is a
  // no-op on the serial walk and makes the text independent of the
  // completion order of a future wavefront.
  std::stable_sort(leaf_dump_.begin(), leaf_dump_.end(),
                   [](const LeafDumpRecord& a, const LeafDumpRecord& b) {
                     if (a.mi_row != b.mi_row) return a.mi_row < b.mi_row;
                     return a.mi_col < b.mi_col;
                   });
  for (const LeafDumpRecord& record : leaf_dump_) {
    std::fprintf(file, "%d %d %d %d %d %d %d %d %d %d %d %u %u %u %016llx",
                 record.mi_row, record.mi_col, record.bsize,
                 record.partition, record.is_inter, record.mode, record.ref,
                 record.drl_index, record.mv_x, record.mv_y, record.skip,
                 record.eob[0], record.eob[1], record.eob[2],
                 static_cast<unsigned long long>(record.levels_digest));
    if (record.palette_index != syntax::BlockSyntax::kNoPalette) {
      std::fprintf(file, " pal=%d,%d,", record.palette_size,
                   record.palette_direction);
      for (int i = 0; i < record.palette_size; ++i) {
        std::fprintf(file, "%s%02x", i == 0 ? "" : ",",
                     static_cast<unsigned>(
                         record.palette_colors[static_cast<std::size_t>(i)]));
      }
    }
    std::fputc('\n', file);
  }
  std::fflush(file);
  leaf_dump_.clear();
}

void InterFrameEncoder::merge_sinks() {
  std::size_t level_base = 0;
  // plan_pass() clears palettes_, but the SDP pass historically did not;
  // starting from the current size reproduces both.
  std::size_t palette_base = palettes_.size();
  for (SbSink& sink : sb_sinks_) {
    for (syntax::BlockSyntax& entry : sink.blocks) {
      if (entry.levels_offset == kNoLevels) {
        entry.levels_offset = 0;
      } else {
        entry.levels_offset += static_cast<std::uint32_t>(level_base);
      }
      for (int part = 0; part < entry.tx_part_count; ++part) {
        entry.tx_part_levels_offset[part] +=
            static_cast<std::uint32_t>(level_base);
      }
      if (entry.palette_index != syntax::BlockSyntax::kNoPalette) {
        entry.palette_index += static_cast<std::uint32_t>(palette_base);
      }
    }
    blocks_.insert(blocks_.end(), sink.blocks.begin(), sink.blocks.end());
    levels_.insert(levels_.end(), sink.levels.begin(), sink.levels.end());
    const std::size_t palette_count = sink.palettes.size();
    for (palette::PaletteResult& palette : sink.palettes) {
      palettes_.push_back(std::move(palette));
    }
    sub_pus_.insert(sub_pus_.end(), sink.sub_pus.begin(),
                    sink.sub_pus.end());
    leaf_dump_.insert(leaf_dump_.end(), sink.leaf_dump.begin(),
                      sink.leaf_dump.end());
    level_base += sink.levels.size();
    palette_base += palette_count;
    // Fold the unit-local counters back in.  Raster order is the historical
    // accumulation order; addition is associative, so a wavefront that
    // completes in any order still lands on the same totals.
    const SbSink::State& st = sink.state;
    inter_blocks_ += st.inter_blocks;
    intra_blocks_ += st.intra_blocks;
    temporal_added_ += st.temporal_added;
    temporal_selected_ += st.temporal_selected;
    same_ref_blocks_ += st.same_ref_blocks;
    refmvbank_added_ += st.refmvbank_added;
    refmvbank_selected_ += st.refmvbank_selected;
    mv_traj_derived_ += st.mv_traj_derived;
    amvd_selected_ += st.amvd_selected;
    flex_precision_selected_ += st.flex_precision_selected;
    derived_sign_selected_ += st.derived_sign_selected;
    tip_blocks_ += st.tip_blocks;
  }
}

bool InterFrameEncoder::run_one_sb(SbSink& sink, const PartitionNode& superblock,
                std::string& error) {
  sink.reset();
  seed_unit_state(sink);
  return collect_node(sink, superblock, error);
}

bool InterFrameEncoder::run_one_sb(SbSink& sink, const pe::PartitionNode& superblock,
                std::string& error) {
  sink.reset();
  seed_unit_state(sink);
  return collect_node_ext(sink, superblock, error);
}

bool InterFrameEncoder::run_one_sb(SbSink& sink, const sdpp::Node& superblock,
                std::string& error) {
  sink.reset();
  seed_unit_state(sink);
  return collect_sdp_node(sink, superblock, error);
}

void InterFrameEncoder::seed_unit_state(SbSink& sink) {
  sink.state.current_qindex = current_qindex_;
  sink.state.current_segment_id = 0;
  sink.state.quant_step = quant_step_;
  for (int plane = 0; plane < 3; ++plane) {
    sink.state.plane_quant[plane] = plane_quant_[plane];
  }
}

bool InterFrameEncoder::wavefront_diagonals(std::size_t count,
                         std::vector<std::vector<std::size_t>>& out) const {
  const int cols = partition::sb_cols(geometry_);
  const int rows = partition::sb_rows(geometry_);
  if (cols <= 0 || rows <= 0 ||
      static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows) !=
          count) {
    return false;
  }
  const int diagonals = 2 * (rows - 1) + cols;
  out.assign(static_cast<std::size_t>(diagonals), {});
  for (int sbr = rows - 1; sbr >= 0; --sbr) {
    for (int sbc = 0; sbc < cols; ++sbc) {
      const std::size_t index =
          static_cast<std::size_t>(sbr) * static_cast<std::size_t>(cols) +
          static_cast<std::size_t>(sbc);
      out[static_cast<std::size_t>(2 * sbr + sbc)].push_back(index);
    }
  }
  return true;
}

bool InterFrameEncoder::wavefront_allowed() const {
  if (keyframe_) return false;
  // AV2ENC_PIPE_PERC / AV2ENC_EFF_PERC / AV2ENC_PIPE_BA: the exact syntax probe
  // advances one shared CDF/grid state in coded order; concurrent superblocks
  // would race it.
  if (pipe_perc::enabled() || eff_perc::enabled() || pipe_ba::enabled()) return false;
  // R09: the syntax probe advances one shared CDF/grid state in coded order;
  // concurrent superblocks would race it.  Serial decisions keep the probe
  // deterministic and match the writer's serial tile walk.
  if (rate_hybrid::reporting()) return false;
#if defined(AV2ENC_VIDEO_DELTAQ)
  if (seg_enabled_ || delta_q_enabled_) return false;
#endif
  if (options_.refmvbank) return false;
  // --video-trace writes one std::cerr line per leaf from inside run_one_sb.
  // Concurrent superblocks interleave those insertions, so lines are lost or
  // split and any gate that counts them becomes flaky: gate_video_edge_mask
  // observed 56 then 53 leaves on the same binary.  Tracing is diagnostic, so
  // keeping the superblock walk serial there restores deterministic output at
  // no cost to the measured paths.
  if (options_.trace) return false;
  return true;
}

double InterFrameEncoder::wavefront_width() const {
  const int cols = partition::sb_cols(geometry_);
  const int rows = partition::sb_rows(geometry_);
  if (cols <= 0 || rows <= 0) return 0.0;
  const double diagonals = static_cast<double>(2 * (rows - 1) + cols);
  return (static_cast<double>(rows) * static_cast<double>(cols)) / diagonals;
}

int InterFrameEncoder::wavefront_override() {
  static const int value = [] {
    const char* env = std::getenv("AV2ENC_WAVEFRONT");
    if (env == nullptr) return -1;
    return std::string(env) == "0" ? 0 : 1;
  }();
  return value;
}

void InterFrameEncoder::plan_pass() {
  sb_alloc_frame_activity_ = 0.0;
  sb_alloc_sbs_ = 0;
  blocks_.clear();
  levels_.clear();
  palettes_.clear();
  leaf_dump_.clear();
  if (palette_enabled_) palette_state_.reset(mi_cols_, mi_rows_);
  // The merge stack reads the above-right / below-left MI grid, so stale
  // entries from the previous pass would make the encoder's stack differ
  // from the writer's.
  mv_grid_.assign(mv_grid_.size(), inter::mc::MiInfo{});
  intra_joint_.assign(intra_joint_.size(), 0);
  uv_smooth_.assign(uv_smooth_.size(), 0);
  // AV2ENC_PIPE_BA: the Butteraugli-shaped frame mask (local-contrast masking
  // blended with the tool weight).  Built once per picture, key included, so
  // every surface that prices distortion reads one map.  With the tool weight
  // at 0 the map is neutral (256 everywhere) and the pipeline still runs its
  // exact-rate decisions: that is the gate's base arm.
  if (pipe_ba::enabled()) {
    pipe_ba::init();
    pipe_ba::build_frame_mask(source_.y.data(), static_cast<int>(source_.width),
                              static_cast<int>(source_.width),
                              static_cast<int>(source_.height), &pipe_ba_mask_);
    if (options_.trace) {
      std::cerr << "  pipe-ba mask " << pipe_ba_mask_.width << "x"
                << pipe_ba_mask_.height << " weight=" << pipe_ba::weight()
                << " low_band=" << pipe_ba::config().low_band_gain << "\n";
    }
  }
  // AV2ENC_PIPE_PERC / AV2ENC_EFF_PERC: source-driven detail weights for this
  // frame and a fresh exact-syntax probe seeded from the same CDF image the
  // writer starts from.  Both are skipped when the gates are off, so the
  // historical pass does no work.
  if ((pipe_perc::enabled() || eff_perc::enabled() || pipe_ba::enabled()) &&
      !keyframe_) {
    if (eff_perc::enabled()) {
      eff_perc::init();
      eff_perc::build_frame_weights(
          source_.y.data(), static_cast<int>(source_.width),
          static_cast<int>(source_.width), static_cast<int>(source_.height),
          &eff_weights_);
    }
    if (pipe_perc::enabled()) {
      pipe_perc::init();
      pipe_perc::build_frame_weights(
          source_.y.data(), static_cast<int>(source_.width),
          static_cast<int>(source_.width), static_cast<int>(source_.height),
          &pipe_weights_);
    }
    pipe_probe_ready_ = false;
    rate_hybrid::ProbeConfig config;
    config.base_q_idx = static_cast<int>(options_.base_q_idx);
    config.mi_cols = mi_cols_;
    config.mi_rows = mi_rows_;
    config.num_refs = num_total_refs_;
    config.max_drl_bits_minus_1 = 4;
    for (int i = 0; i < 7; ++i) {
      config.frame_distance[i] = frame_distance_[static_cast<std::size_t>(i)];
    }
    config.reference_select = options_.compound && num_total_refs_ > 1;
    config.allow_bawp =
        (options_.enable_bawp || options_.enable_morph) && !keyframe_;
    config.enable_interintra = options_.enable_interintra && !keyframe_;
    config.enable_adaptive_mvd = options_.enable_adaptive_mvd;
    config.enable_flex_mvres = options_.enable_flex_mvres;
    config.enable_mvd_sign_derive = options_.enable_mvd_sign_derive;
    config.compound_exact = compound_exact_rate_enabled();
    config.enable_masked_compound =
        options_.compound &&
        (options_.compound_wedge || options_.compound_diffwtd);
    config.enable_cwp = options_.enable_cwp_weighted;
    config.num_same_ref_compound = options_.same_ref_compound ? 1 : 0;
    config.mv_precision =
        options_.seq_force_integer_mv ? 3 : options_.mv_precision;
    config.disable_cdf_update = !options_.adaptive_cdf;
    pipe_probe_.reset(config, cdf_seed_);
    pipe_probe_ready_ = true;
  }
  // R09: every decision pass starts from the same frame/tile CDF seed and an
  // empty neighbour grid, exactly like the writer does.
  rate_probe_ready_ = false;
  if (rate_hybrid::reporting() && !keyframe_) {
    rate_hybrid::ProbeConfig config;
    config.base_q_idx = static_cast<int>(options_.base_q_idx);
    config.mi_cols = mi_cols_;
    config.mi_rows = mi_rows_;
    config.num_refs = num_total_refs_;
    config.max_drl_bits_minus_1 = 4;
    for (int i = 0; i < 7; ++i) {
      config.frame_distance[i] = frame_distance_[static_cast<std::size_t>(i)];
    }
    config.reference_select = options_.compound && num_total_refs_ > 1;
    config.allow_bawp =
        (options_.enable_bawp || options_.enable_morph) && !keyframe_;
    config.enable_interintra = options_.enable_interintra && !keyframe_;
    config.enable_adaptive_mvd = options_.enable_adaptive_mvd;
    config.enable_flex_mvres = options_.enable_flex_mvres;
    config.enable_mvd_sign_derive = options_.enable_mvd_sign_derive;
    config.compound_exact = compound_exact_rate_enabled();
    config.enable_masked_compound =
        options_.compound &&
        (options_.compound_wedge || options_.compound_diffwtd);
    config.enable_cwp = options_.enable_cwp_weighted;
    config.num_same_ref_compound = options_.same_ref_compound ? 1 : 0;
    config.mv_precision =
        options_.seq_force_integer_mv ? 3 : options_.mv_precision;
    config.disable_cdf_update = !options_.adaptive_cdf;
    rate_probe_.reset(config, cdf_seed_);
    rate_probe_ready_ = true;
  }
  for (int plane = 0; plane < 3; ++plane) {
    std::fill(recon_[plane].samples.begin(), recon_[plane].samples.end(),
              static_cast<std::uint8_t>(0));
  }
  inter_blocks_ = 0;
  intra_blocks_ = 0;
#if defined(AV2ENC_VIDEO_DELTAQ)
  if (seg_enabled_) {
    deltaq::clear_above_context(enc_seg_grid_);
    deltaq::clear_left_context(enc_seg_grid_);
  }
#endif
}

void InterFrameEncoder::plan_sdp_pass() {
  blocks_.clear();
  levels_.clear();
  leaf_dump_.clear();
  sdp_luma_sizes_.reset(geometry_);
  sdp_chroma_sizes_.reset(geometry_);
  mv_grid_.assign(mv_grid_.size(), inter::mc::MiInfo{});
  intra_joint_.assign(intra_joint_.size(), 0);
  uv_smooth_.assign(uv_smooth_.size(), 0);
  for (int plane = 0; plane < 3; ++plane) {
    std::fill(recon_[plane].samples.begin(), recon_[plane].samples.end(),
              static_cast<std::uint8_t>(0));
  }
  inter_blocks_ = 0;
  intra_blocks_ = 0;
}

bool InterFrameEncoder::collect_node(SbSink& sink, const PartitionNode& node,
                  std::string& error,
                  const ChromaInfo& inherited_chroma,
                  PartitionNode* emitted) {
  // AV2ENC_RD_TAIL_FLOOR: a 16x16 tail leaf is the one place the measured
  // coding-block descent runs.  The hook is inert unless collect_frame() armed
  // the gate for this picture; `emitted` lets the sequential walk carry the
  // chosen layout back into the frame tree.  Trials re-enter this function
  // with `tail_floor_trial_` set, so they never recurse into the hook.
  if (part_diag_active_ && !part_diag_trial_ && !keyframe_ &&
      !tail_floor_active_ && part_diag_candidate(node)) {
    return part_diag_collect_node(sink, node, inherited_chroma, emitted,
                                  error);
  }
  if (tail_floor_active_ && !tail_floor_trial_ && !keyframe_ &&
      tail_floor_candidate(node)) {
    return tail_floor_collect_node(sink, node, inherited_chroma, emitted,
                                   error);
  }
  SbSink::State& st = sink.state;
  const ChromaInfo chroma = chroma_step(node, inherited_chroma);
  if (node.depth == 0) {
    // §6.19.2.3 clear_block_decoded_flags(): the map is superblock-local.
    st.decoded[0].reset(mi_cols_, mi_rows_);
    st.decoded[1].reset(mi_cols_ / 2, mi_rows_ / 2);
    st.decoded[2].reset(mi_cols_ / 2, mi_rows_ / 2);
    reset_refmvbank_for_sb(node.mi_row, node.mi_col);
#if defined(AV2ENC_VIDEO_DELTAQ)
    // §7.14.2 get_qindex() and §5.20.5.7 SegmentIds are superblock state for
    // this encoder: the syntax writer signals the same id and the delta at
    // the first leaf of the superblock.
    if (qp_tools_active()) {
      const int sb = (node.mi_row / qp_sb_mi()) * qp_sb_cols() +
                     node.mi_col / qp_sb_mi();
      if (sb < 0 || static_cast<std::size_t>(sb) >= sb_qindex_.size()) {
        error = "delta-Q: superblock index outside the plan";
        return false;
      }
      st.current_qindex = sb_qindex_[static_cast<std::size_t>(sb)];
      st.current_segment_id =
          seg_enabled_
              ? static_cast<int>(sb_segment_[static_cast<std::size_t>(sb)])
              : 0;
      st.quant_step =
          rc::kQlookupQ0To255[std::clamp(st.current_qindex, 0, 255)];
      update_plane_quant(sink.state);
      if (seg_enabled_ && node.mi_col == 0) {
        deltaq::clear_left_context(enc_seg_grid_);
      }
      if (trace_ && seg_enabled_) {
        std::cerr << "segq sb=" << sb
                  << " segment=" << st.current_segment_id
                  << " qindex=" << st.current_qindex << "\n";
      }
    }
#endif
  }
  const std::size_t index = sink.blocks.size();
  sink.blocks.emplace_back();
  syntax::BlockSyntax& entry = sink.blocks.back();
  entry.levels_offset = kNoLevels;
  entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
  entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
  entry.bsize = static_cast<std::uint8_t>(node.size);
  entry.partition = static_cast<std::uint8_t>(node.partition);
  std::uint8_t mask = 0;
  for (int p = 0; p < partition::kPartitionTypeCount; ++p) {
    if (node.allowed.allowed[p]) {
      mask = static_cast<std::uint8_t>(mask | (1U << p));
    }
  }
  entry.allowed_mask = mask;
  if (node.forced) entry.flags |= syntax::kFlagForced;
  if (node.depth == 0) entry.flags |= syntax::kFlagSbRoot;
  // Carry the decoded ownership arguments and the inherited ChromaMi block
  // so the writer codes chroma exactly once for a sub-8 group.  Every leaf
  // at 8x8 or above has ChromaMi == itself, which reproduces the historical
  // leaf-derived path.
  entry.flags |= syntax::kFlagHasChromaInfo;
  entry.has_chroma = node.has_chroma ? 1 : 0;
  entry.chroma_offset = node.chroma_offset ? 1 : 0;
  entry.chroma_bsize =
      chroma.valid ? static_cast<std::uint8_t>(chroma.size) : 0;
  entry.chroma_mi_row = static_cast<std::uint16_t>(chroma.mi_row);
  entry.chroma_mi_col = static_cast<std::uint16_t>(chroma.mi_col);

  if (node.partition == PartitionType::kNone) {
    const std::size_t dump_level_begin = sink.levels.size();
    if (!encode_leaf(sink, node, index, error, /*tree_type=*/0,
                     node.has_chroma, chroma.mi_row, chroma.mi_col,
                     chroma.valid ? static_cast<int>(chroma.size) : -1)) {
      return false;
    }
    record_leaf_dump(sink, sink.blocks[index], dump_level_begin);
    update_refmvbank_leaf(sink.blocks[index]);
    if (emitted != nullptr) *emitted = node;
  } else {
    std::vector<PartitionNode> emitted_children;
    if (emitted != nullptr) emitted_children.reserve(node.children.size());
    for (const PartitionNode& child : node.children) {
      PartitionNode child_out;
      if (!collect_node(sink, child, error, chroma,
                        emitted != nullptr ? &child_out : nullptr)) {
        return false;
      }
      if (emitted != nullptr) emitted_children.push_back(std::move(child_out));
    }
    if (emitted != nullptr) {
      *emitted = node;
      emitted->children = std::move(emitted_children);
    }
  }
  const std::size_t subtree = sink.blocks.size() - index - 1;
  if (subtree > 0xFFFFU) {
    error = "partition subtree too large";
    return false;
  }
  sink.blocks[index].subtree = static_cast<std::uint16_t>(subtree);
  return true;
}

bool InterFrameEncoder::collect_node_ext(SbSink& sink, const pe::PartitionNode& node,
                      std::string& error) {
  SbSink::State& st = sink.state;
  if (node.depth == 0) {
    st.decoded[0].reset(mi_cols_, mi_rows_);
    st.decoded[1].reset(mi_cols_ / 2, mi_rows_ / 2);
    st.decoded[2].reset(mi_cols_ / 2, mi_rows_ / 2);
    reset_refmvbank_for_sb(node.mi_row, node.mi_col);
  }
  const std::size_t index = sink.blocks.size();
  sink.blocks.emplace_back();
  syntax::BlockSyntax& entry = sink.blocks.back();
  entry.levels_offset = kNoLevels;
  entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
  entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
  entry.bsize = static_cast<std::uint8_t>(node.size);
  entry.partition = static_cast<std::uint8_t>(node.partition);
  std::uint16_t mask = 0;
  for (int p = 0; p < pe::kPartitionCount; ++p) {
    if (node.allowed.allowed[p]) {
      mask = static_cast<std::uint16_t>(mask | (1U << p));
    }
  }
  entry.allowed_mask = mask;
  if (node.forced) entry.flags |= syntax::kFlagForced;
  if (node.depth == 0) entry.flags |= syntax::kFlagSbRoot;
  entry.flags |= syntax::kFlagHasChromaInfo;
  entry.has_chroma = node.has_chroma ? 1 : 0;
  entry.chroma_offset = node.chroma_offset ? 1 : 0;
  entry.chroma_bsize =
      node.chroma.size == BlockSize::kInvalid
          ? 0
          : static_cast<std::uint8_t>(node.chroma.size);
  entry.chroma_mi_row = static_cast<std::uint16_t>(node.chroma.mi_row);
  entry.chroma_mi_col = static_cast<std::uint16_t>(node.chroma.mi_col);

  if (node.partition == pe::PartitionType::kNone) {
    // encode_leaf() only reads the origin/size, so a stack view is enough.
    PartitionNode leaf;
    leaf.mi_row = node.mi_row;
    leaf.mi_col = node.mi_col;
    leaf.size = static_cast<partition::BlockSize>(node.size);
    const std::size_t dump_level_begin = sink.levels.size();
    if (!encode_leaf(
            sink, leaf, index, error, /*tree_type=*/0, node.has_chroma,
            node.chroma.mi_row, node.chroma.mi_col,
            node.chroma.size == BlockSize::kInvalid
                ? -1
                : static_cast<int>(node.chroma.size))) {
      return false;
    }
    record_leaf_dump(sink, sink.blocks[index], dump_level_begin);
    update_refmvbank_leaf(sink.blocks[index]);
  } else {
    for (const pe::PartitionNode& child : node.children) {
      if (!collect_node_ext(sink, child, error)) return false;
    }
  }
  const std::size_t subtree = sink.blocks.size() - index - 1;
  if (subtree > 0xFFFFU) {
    error = "partition subtree too large";
    return false;
  }
  sink.blocks[index].subtree = static_cast<std::uint16_t>(subtree);
  return true;
}

bool InterFrameEncoder::encode_leaf(SbSink& sink, const PartitionNode& node, std::size_t index,
                 std::string& error, int tree_type,
                 bool leaf_has_chroma, int chroma_mi_row,
                 int chroma_mi_col, int chroma_bsize) {
  SbSink::State& st = sink.state;
  const int r = node.mi_row;
  const int c = node.mi_col;
  if (tip_leaf_armed_ && tree_type == 0 && r == tip_leaf_r_ && c == tip_leaf_c_) {
    return encode_tip_leaf(sink, node, index, error);
  }
  const int w = partition::block_width(node.size);
  const int h = partition::block_height(node.size);
  const int w4 = w >> 2;
  const int h4 = h >> 2;
  const int x0 = c * 4;
  const int y0 = r * 4;
  // §5.20.3.1 chroma ownership: an SDP leaf carries the decoded HasChroma
  // and the ChromaMi block it owns.  A CHROMA_PART leaf codes no luma; a
  // LUMA_PART leaf codes no chroma.
  const bool chroma_part = tree_type == 2;
  const bool luma_part = tree_type == 1;
  const bool owns_chroma = leaf_has_chroma && !luma_part;
  BlockSize chroma_size = node.size;
  if (chroma_bsize >= 0) {
    chroma_size = static_cast<BlockSize>(chroma_bsize);
  }
  // §5.20.3.1: a sub-8 leaf group shares one inherited ChromaMi block.  Its
  // chroma plane lives at the ChromaMi origin, not at this leaf's origin,
  // and is larger than leaf_w/2 x leaf_h/2.  Inter prediction (which derives
  // chroma from the luma leaf) and the screen-content tools are not wired
  // for that geometry, so such leaves stay intra.
  const bool shared_chroma =
      owns_chroma &&
      (chroma_mi_row != r || chroma_mi_col != c ||
       chroma_size != node.size);
  int cw = 0;
  int ch = 0;
  if (chroma_bsize >= 0) {
    const BlockSize chroma_residual =
        partition::chroma_plane_residual_size(geometry_, chroma_size);
    cw = partition::block_width(chroma_residual);
    ch = partition::block_height(chroma_residual);
  } else {
    cw = w / 2;
    ch = h / 2;
  }
  const int cx0 = chroma_bsize >= 0 ? (chroma_mi_col * 4) / 2 : x0 / 2;
  const int cy0 = chroma_bsize >= 0 ? (chroma_mi_row * 4) / 2 : y0 / 2;
  // Absolute luma/chroma coordinates for source reads and MC.  The coded
  // geometry and reconstruction stay tile-local.
  const int ax0 = x0 + tile_x_;
  const int ay0 = y0 + tile_y_;
  const int acx0 = cx0 + tile_x_ / 2;
  const int acy0 = cy0 + tile_y_ / 2;
  // --video-edge-mask: if this leaf's coded extent crosses the visible
  // bottom edge, the tail rows are decoder-invisible.  Both TX entry points
  // use the prediction as the source view there (zero residual) so no
  // levels are spent on samples the decoder discards after filtering.
  const int luma_mask_row = visible_rows(0, y0, h);
  const int chroma_mask_row = visible_rows(1, cy0, ch);
  // AV2ENC_PERCEPTUAL_RD: opt-in detail-aware TX distortion.  The weight map
  // is built from the source luma of this leaf's extent; strength 0 leaves it
  // invalid and both TX lambdas below pass through unchanged.  The proxy is
  // confined to the TX search: weighting the leaf-level candidate/skip SSE on
  // top of it measured worse, so only the TX cost is re-weighted (by dividing
  // lambda, which is the same as weighting the TX search's SSE).  A
  // CHROMA_PART leaf codes no luma, so it has no luma activity to score.  See
  // `video_perceptual.h` for the model and why the frame mean normalises it.
  const double perceptual_strength = perceptual::rd_strength();
  perceptual::LeafWeights perceptual_weights;
  if (perceptual_strength > 0.0 && !chroma_part &&
      source_.y.size() >= static_cast<std::size_t>(source_.width) *
                               static_cast<std::size_t>(source_.height)) {
    perceptual_weights = perceptual::make_leaf_weights(
        source_.y.data(), static_cast<int>(source_.width),
        static_cast<int>(source_.width), static_cast<int>(source_.height),
        ax0, ay0, w, h, perceptual_strength);
  }
  // --video-edge-mask: the decision metrics score only the rows the decoder
  // can see.  Removing the hidden-row bits is not enough on its own: the
  // skip/residual and mode comparisons would still charge a coded residual
  // for hidden-row SSD it can never fix (the source there is the clamped
  // last visible row, the prediction comes from the edge-padded reference),
  // which biases every bottom leaf toward skip.
  const auto sse_vis = [&](int plane, int sx, int sy, int sw, int sh,
                           const std::vector<std::uint8_t>& samples) {
    return sse_against_source(source_, plane, sx, sy, sw, sh, samples,
                              plane == 0 ? luma_mask_row : chroma_mask_row);
  };
  // AV2ENC_PIPE_PERC / AV2ENC_EFF_PERC: one objective for the leaf and the TX
  // search.
  //   D  = local perceptual luma SSE + weighted chroma SSE
  //   R  = exact inter syntax probe + coefficient model
  //   J  = D + lambda * R
  // `pipe_leaf_weights` is the per-pixel map handed to the TX candidate
  // search, so the TX size/type/partition choice minimizes the same J.
  // AV2ENC_PIPE_BA: the Butteraugli-shaped model replaces the CSF on the leaf
  // and TX distortion.  `ba_on` is the tool (weight > 0, mask built); the
  // pipeline itself (`pipe_on`) stays on at weight 0 so the gate's base arm
  // keeps the exact-rate decisions with plain SSE.
  const bool ba_on =
      pipe_ba::tool_on() && !keyframe_ && pipe_probe_ready_ && pipe_ba_mask_.valid();
  // PIPE_BA owns the distortion whenever it is on; the eff-perc branch only
  // runs for the EFF_PERC gate itself.
  const bool eff_on =
      !pipe_ba::enabled() && eff_perc::enabled() && !keyframe_ && pipe_probe_ready_;
  const bool pipe_on =
      (pipe_perc::enabled() || eff_perc::enabled() || pipe_ba::enabled()) &&
      !keyframe_ && pipe_probe_ready_;
  const double pipe_lambda =
      pipe_on
          ? leaf_rd_lambda(st.current_qindex, options_) *
                (pipe_ba::enabled() ? pipe_ba::config().lambda
                                    : (eff_on ? eff_perc::lambda_scale()
                                              : pipe_perc::lambda_scale()))
          : 0.0;
  const int pipe_qctx = ::av2enc::still::qctx::slice_for_base_q_idx(
      static_cast<int>(options_.base_q_idx));
  std::vector<std::uint16_t> pipe_leaf_weights;
  std::vector<std::uint16_t> pipe_chroma_weights;
  if (pipe_on && !chroma_part) {
    if (ba_on) {
      pipe_ba::fill_leaf_weights(pipe_ba_mask_, ax0, ay0, w, h,
                                 &pipe_leaf_weights);
      ++pipe_ba::counters().luma_leaves;
    } else if (eff_on) {
      eff_perc::fill_leaf_weights(eff_weights_, ax0, ay0, w, h,
                                  &pipe_leaf_weights);
    } else {
      pipe_perc::fill_leaf_weights(pipe_weights_, ax0, ay0, w, h,
                                   &pipe_leaf_weights);
    }
  }
  if (pipe_on && ba_on && !luma_part) {
    pipe_ba::fill_chroma_leaf_weights(pipe_ba_mask_, acx0, acy0, cw, ch,
                                      &pipe_chroma_weights);
    ++pipe_ba::counters().chroma_leaves;
  }
  const auto sse_pipe = [&](int plane, int sx, int sy, int sw, int sh,
                            const std::vector<std::uint8_t>& samples) {
    if (ba_on) {
      if (plane == 0) {
        return pipe_ba::plane_distortion(
            pipe_leaf_weights.empty() ? nullptr : pipe_leaf_weights.data(), sw,
            sh, source_.y.data(), static_cast<int>(source_.width), sx, sy,
            samples.data(), sw, 0, 0, luma_mask_row,
            pipe_ba::config().low_band_gain);
      }
      const std::span<const std::uint8_t> src_plane =
          plane == 1 ? source_.u : source_.v;
      if (src_plane.empty()) {
        return static_cast<double>(sse_vis(plane, sx, sy, sw, sh, samples));
      }
      return pipe_ba::plane_distortion(
                 pipe_chroma_weights.empty() ? nullptr
                                             : pipe_chroma_weights.data(),
                 sw, sh, src_plane.data(), static_cast<int>(source_.width) / 2,
                 sx, sy, samples.data(), sw, 0, 0, chroma_mask_row,
                 pipe_ba::config().chroma_low_band_gain) *
             pipe_ba::config().chroma_scale;
    }
    if (!pipe_on || pipe_leaf_weights.empty()) {
      return static_cast<double>(sse_vis(plane, sx, sy, sw, sh, samples));
    }
    if (plane != 0) {
      if (eff_on && eff_perc::csf_gain() > 0.0) {
        const std::span<const std::uint8_t> src_plane =
            plane == 1 ? source_.u : source_.v;
        if (!src_plane.empty()) {
          const double csf = eff_perc::weighted_sse(
              nullptr, sw, sh, src_plane.data(),
              static_cast<int>(source_.width) / 2, sx, sy, samples.data(), sw,
              0, 0, chroma_mask_row, eff_perc::csf_gain(),
              eff_perc::csf2_gain());
          return csf * eff_perc::chroma_weight();
        }
      }
      const double plain =
          static_cast<double>(sse_vis(plane, sx, sy, sw, sh, samples));
      return eff_on ? plain * eff_perc::chroma_weight() : plain;
    }
    if (eff_on) {
      return eff_perc::weighted_sse(
          pipe_leaf_weights.data(), sw, sh, source_.y.data(),
          static_cast<int>(source_.width), sx, sy, samples.data(), sw, 0, 0,
          luma_mask_row, eff_perc::csf_gain(), eff_perc::csf2_gain());
    }
    return pipe_perc::weighted_sse(
        pipe_leaf_weights.data(), sw, sh, source_.y.data(),
        static_cast<int>(source_.width), sx, sy, samples.data(), sw, 0, 0,
        luma_mask_row);
  };
  // AV2ENC_EFF_PERC: coefficient term selection.  Rate 1 prices the emitted
  // plan with the R19 interval width against the live frame-start CDF image;
  // rate 0 keeps the static §8.3 context model pipe-perc shipped with.
  // AV2ENC_PIPE_BA inherits the same front end at every tool weight, so the
  // base arm and the knob arms differ only in D.  The key picture keeps the
  // historical estimate in both gates (its probe is not seeded).
  const bool pipe_rate_on =
      eff_perc::enabled() && !keyframe_ && pipe_probe_ready_;
  const bool eff_interval =
      pipe_rate_on && (eff_perc::rate_mode() & 1) != 0;
  const bool eff_live_cdf =
      pipe_rate_on && (eff_perc::rate_mode() & 2) != 0;
  const still::qctx::CoeffCdfSlice* eff_slice =
      eff_live_cdf ? eff_cdf_slice() : nullptr;
  const auto coeff_bits = [&](const TxPlan& plan, int plane, bool is_inter) {
    if (eff_interval) {
      ++eff_perc::counters().rate_interval;
      return plan_rate_bits(plan, plane, is_inter,
                            static_cast<int>(node.size));
    }
    return pipe_plan_coefficient_bits(plan, plane, is_inter, pipe_qctx,
                                      eff_slice);
  };
  if (ibc_enabled_) {
    // §5.20.4.1 decode_block(): update_ibc_buffers() runs at the start of
    // every block, before its mode info builds the BV stack.
    ibcs::begin_ibc_block(ibc_buffer_, r, c);
  }

  syntax::BlockSyntax& entry = sink.blocks[index];

  // The CUDA pipeline evaluates inter candidates only when the luma block
  // fits inside the reconstructed reference extent.  A bottom/right edge node
  // can straddle the MI-aligned plane; both paths must make the same choice,
  // so the whole-frame CPU reference applies the same test.  Tiled coding has
  // its own clipped geometry and keeps the historical path.
  //
  // With --video-edge-inter the bottom edge is reconstructed into the
  // SB-aligned padded plane and the reference carries the same edge-extended
  // rows the decoder pads, so a leaf whose visible part is inside the picture
  // keeps the inter search.
  const bool inter_fits =
      width_ != static_cast<int>(source_.width) ||
      height_ != static_cast<int>(source_.height) ||
      (x0 + w <= mi_cols_ * 4 &&
       (y0 + h <= mi_rows_ * 4 ||
        (edge_pad_rows_ > 0 && y0 < mi_rows_ * 4)));

  InterCandidate best;
  InterCandidate same_ref_best;
  // Lowest-SSE signallable non-GLOBALMV candidate.  §5.20.7.6 makes BAWP
  // unavailable on a GLOBALMV leaf, so when the plain search picks GLOBALMV
  // (every zero-MV block of a static clip does) this carrier is the same MV
  // expressed as NEARMV/NEWMV and can carry use_bawp.
  InterCandidate bawp_carrier;
  InterCandidate global_cand;
  bool global_cand_valid = false;
#if defined(AV2ENC_VIDEO_DELTAQ)
  // SEG_LVL_SKIP / SEG_LVL_GLOBALMV force YMode = GLOBALMV and use
  // SkipSegFrame for RefFrame[0] on inter blocks.  The encoded block must
  // therefore keep the first reference and the §7.12.2.2 global MV.
  const bool seg_skip_leaf =
      !keyframe_ && seg_enabled_ && skip_seg_id_ >= 0 &&
      st.current_segment_id == skip_seg_id_;
  const bool seg_globalmv_leaf =
      !keyframe_ && seg_enabled_ && globalmv_seg_id_ >= 0 &&
      st.current_segment_id == globalmv_seg_id_;
  const bool seg_forced_inter = seg_skip_leaf || seg_globalmv_leaf;
#else
  const bool seg_skip_leaf = false;
  const bool seg_globalmv_leaf = false;
  const bool seg_forced_inter = false;
#endif
  // --video-leaf-rd prices the inter candidates with the same cost the leaf
  // decision uses, so a NEWMV that wins by a handful of SSE but costs ten
  // MVD bits no longer shadows a NEARMV/GLOBALMV that codes zero MVD.
  const double inter_lambda =
      options_.enable_leaf_rd
          ? leaf_rd_lambda(st.current_qindex, options_)
          : 0.0;
  const bool exact_on = exact_rd::enabled();
  // R09 (AV2ENC_RATE_HYBRID_S): when the model drives the decision, the
  // candidate ranking and the skip/residual bar use the exact CDF probe and
  // the Laplacian coefficient rate.  The measure switch keeps the historical
  // decisions and only records the model's prediction error.
  const bool rate_model_on = rate_hybrid::enabled() &&
                            rate_hybrid::syntax_enabled() &&
                            rate_probe_ready_;
  const bool rate_reporting = rate_hybrid::reporting() && rate_probe_ready_;
  // Minimal BlockSyntax view of one candidate, for the syntax probe.  The
  // probe only reads the mode/ref/DRL/MVD fields and the block geometry.
  const auto probe_node_for = [&](const InterCandidate& cand) {
    syntax::BlockSyntax probe{};
    probe.mi_row = static_cast<std::uint16_t>(r);
    probe.mi_col = static_cast<std::uint16_t>(c);
    probe.bsize = static_cast<std::uint8_t>(node.size);
    probe.is_inter = 1;
    probe.inter_mode = static_cast<std::uint8_t>(cand.mode);
    probe.ref_index = static_cast<std::uint8_t>(cand.ref_index);
    probe.drl_index = static_cast<std::uint8_t>(cand.drl_index);
    probe.mvd_y = cand.mvd.y;
    probe.mvd_x = cand.mvd.x;
    probe.compound = cand.compound ? 1 : 0;
    probe.compound_joint = cand.compound ? (cand.compound_joint ? 1 : 0) : 0;
    probe.compound_mode = static_cast<std::uint8_t>(cand.compound_mode);
    probe.ref_index1 = static_cast<std::uint8_t>(cand.ref_index1);
    probe.drl_index1 = static_cast<std::uint8_t>(cand.drl_index1);
    probe.mv_y = cand.mv.y;
    probe.mv_x = cand.mv.x;
    probe.mv1_y = cand.mv1.y;
    probe.mv1_x = cand.mv1.x;
    probe.compound_masked = cand.compound ? (cand.compound_type != 2 ? 1 : 0) : 0;
    probe.compound_type =
        probe.compound_masked ? static_cast<std::uint8_t>(cand.compound_type)
                              : 0;
    probe.cwp_index = static_cast<std::uint8_t>(cand.cwp_index);
    probe.cwp_weight = static_cast<std::int8_t>(cand.cwp_weight);
    probe.use_optflow = cand.use_optflow ? 1 : 0;
    probe.use_refinemv = cand.use_refinemv ? 1 : 0;
    probe.use_amvd = cand.use_amvd ? 1 : 0;
    probe.block_mv_precision = static_cast<std::int8_t>(cand.block_mv_precision);
    probe.mvd_sign_derived = cand.mvd_sign_derived ? 1 : 0;
    return probe;
  };
  // AV2ENC_COMPOUND_RATE: the historical proxy prices a compound block with
  // the single-prediction YMode/DRL formula, so comp_mode/is_joint/
  // compound_mode/comp_group_idx, the second DRL index and the compound
  // interp-filter symbol are free.  Charge the §5.20.7.6/.8/.16 symbol set so
  // the weighted pair search is not biased toward the most expensive syntax.
  const bool compound_rate_on =
      std::getenv("AV2ENC_COMPOUND_RATE") != nullptr;
  const double compound_rate_scale = [] {
    const char* value = std::getenv("AV2ENC_COMPOUND_RATE_SCALE");
    return value != nullptr ? std::atof(value) : 1.0;
  }();
  const auto compound_syntax_bits = [&](const InterCandidate& cand) {
    if (!compound_rate_on || !cand.compound) return 0.0;
    const bool joint = cand.compound_joint;
    const bool near_near = !joint && cand.compound_mode == 0;
    const bool near_new = !joint && cand.compound_mode == 1;
    const bool new_near = !joint && cand.compound_mode == 2;
    if (joint || near_new || new_near) {
      // Fallback for the compound NEWMV carriers when the exact probe is not
      // available: comp_mode + is_joint, then either compound_mode_non_joint
      // or jmvd_scale_mode, the DRL chain(s), the coded MVD and (joint /
      // NEAR_NEAR only) cwp_idx.
      double bits = 1.0;  // comp_mode
      bits += 1.0;        // is_joint
      if (joint) {
        bits += 2.3;      // jmvd_scale_mode (5 symbols)
        bits += 0.7 * static_cast<double>(cand.drl_index);
      } else {
        bits += 4.0;      // compound_mode_non_joint (5 symbols)
        if (near_new) {
          bits += 0.7 * static_cast<double>(cand.drl_index) +
                  0.7 * static_cast<double>(cand.drl_index1);
        } else {
          bits += 0.7 * static_cast<double>(cand.drl_index);
        }
      }
      const int shell = std::abs(cand.mvd.y) + std::abs(cand.mvd.x);
      if (shell > 0) {
        bits += 4.0 + std::floor(std::log2(static_cast<double>(shell)));
      }
      if (near_near || joint) {
        bits += 1.0;  // comp_group_idx (AVERAGE; enable_masked_compound)
        bits += 0.7 * static_cast<double>(cand.cwp_index);
      }
      if (interp_block_enabled()) bits += 1.585;  // interp_filter symbol
      return bits * compound_rate_scale;
    }
    multiref::CompoundModeParams params;
    params.type = cand.compound_mode == 0
                      ? inter::adv::CompoundType::kAverage
                      : (cand.compound_type == 0
                             ? inter::adv::CompoundType::kWedge
                             : (cand.compound_type == 1
                                    ? inter::adv::CompoundType::kDiffWtd
                                    : inter::adv::CompoundType::kAverage));
    params.cwp_weight = cand.cwp_weight;
    params.cwp_index = cand.cwp_index;
    params.use_imp_mask = cand.use_imp_mask;
    params.wedge_index = cand.wedge_index;
    params.wedge_sign = cand.wedge_sign;
    params.mask_type = cand.mask_type;
    const bool same_side = (frame_distance_[0] >= 0) ==
                           (frame_distance_[1] >= 0);
    double bits = multiref::compound_mode_bits(params, same_side, true);
    if (near_near) {
      // §5.20.7.8 read_drl_idx() twice; the second list starts at 0 for the
      // opposite-reference pair.
      bits += 0.7 * static_cast<double>(cand.drl_index) +
              0.7 * static_cast<double>(cand.drl_index1);
    }
    if (interp_block_enabled()) bits += 1.585;  // interp_filter symbol
    return bits * compound_rate_scale;
  };
  // Exact mode/ref/DRL/MVD price, or the historical constant formula when the
  // probe cannot model this candidate.  `modeled` reports whether the exact
  // probe covered the candidate (including the compound symbol set).
  const auto base_candidate_syntax_bits = [&](const InterCandidate& cand,
                                              bool* modeled) {
    *modeled = false;
    if (pipe_on) {
      rate_hybrid::Terms terms;
      if (pipe_probe_.price_mode(probe_node_for(cand), &terms)) {
        *modeled = true;
        return terms.mode_info();
      }
      ++(eff_on ? eff_perc::counters().fallbacks
                : pipe_perc::counters().fallbacks);
    }
    if (rate_model_on) {
      rate_hybrid::Terms terms;
      if (rate_probe_.price_mode(probe_node_for(cand), &terms)) {
        *modeled = true;
        return terms.mode_info();
      }
    }
    return inter_syntax_bits(cand.mode, cand.drl_index, cand.mvd.y,
                             cand.mvd.x, cand.use_amvd);
  };
  const auto candidate_syntax_bits = [&](const InterCandidate& cand) {
    bool modeled = false;
    double bits = base_candidate_syntax_bits(cand, &modeled);
    if (!modeled) bits += compound_syntax_bits(cand);
    return bits;
  };
  const auto inter_cost = [&](const InterCandidate& cand) {
    return static_cast<double>(cand.sse) +
           (pipe_on ? pipe_lambda : inter_lambda) *
               candidate_syntax_bits(cand);
  };
  // AV2ENC_EXACT_RD: the candidates the cheap search considered, with the
  // exact cost key the proxy used.  `order` makes the ranking total: equal
  // costs keep the historical search order.  Gate off leaves this empty.
  struct RankedCandidate {
    InterCandidate cand;
    double cost = 0.0;
    std::size_t order = 0;
  };
  std::vector<RankedCandidate> ranked_candidates;
  // The historical ranking, kept in parallel under the R09 gate so the report
  // can count how many winners the model changed.
  InterCandidate best_hist;
  const auto consider = [&](InterCandidate cand) {
    if (!cand.valid) return;
    if (exact_on || pipe_on) {
      ranked_candidates.push_back(
          RankedCandidate{cand, inter_cost(cand), ranked_candidates.size()});
    }
    if (rate_reporting) {
      if (!best_hist.valid) {
        best_hist = cand;
      } else if (inter_lambda > 0.0
                     ? static_cast<double>(cand.sse) +
                               inter_lambda *
                                   inter_syntax_bits(cand.mode, cand.drl_index,
                                                     cand.mvd.y, cand.mvd.x,
                                                     cand.use_amvd) <
                           static_cast<double>(best_hist.sse) +
                               inter_lambda *
                                   inter_syntax_bits(
                                       best_hist.mode, best_hist.drl_index,
                                       best_hist.mvd.y, best_hist.mvd.x,
                                       best_hist.use_amvd)
                     : cand.sse < best_hist.sse) {
        best_hist = cand;
      }
    }
    if (!best.valid) {
      best = cand;
    } else if (inter_lambda > 0.0 ? inter_cost(cand) < inter_cost(best)
                                   : cand.sse < best.sse) {
      best = cand;
    }
    // §5.20.7.6 BAWP is coded only for a non-GLOBALMV single-prediction
    // leaf.  Track the best carrier so a GLOBALMV winner can still use the
    // fitted predictor when the candidate set contains the same MV.
    if (options_.bawp_carrier && !cand.compound &&
        cand.mode != syntax::kInterGlobalMv &&
        (!bawp_carrier.valid || cand.sse < bawp_carrier.sse)) {
      bawp_carrier = cand;
    }
  };

  // §5.20.3.1 sub-8 luma leaves and shared-ChromaMi owners keep the intra
  // path: inter MC derives chroma from the leaf rectangle, which is not the
  // owned chroma block.  Leaves at 8x8 and above keep the historical inter
  // search, so every inter leaf still owns a leaf-sized chroma block.
  const bool inter_allowed =
      !shared_chroma && w >= 8 && h >= 8;
  if (!keyframe_ && inter_fits && inter_allowed) {
    for (int ref_index = 0; ref_index < num_total_refs_; ++ref_index) {
      merge::StackParams stack_params;
      merge::MvStack stack;
      const bool have_stack =
          load_stack(r, c, w4, h4, ref_index, sink.state, stack_params,
                     stack);

      // GLOBALMV (§5.20.7.6 / §7.13.3.14).  With an accepted non-identity
      // model the decoder either warps the block (§7.13.3.19/.20, blocks at
      // least 8x8) or applies the §7.12.2.2 projected centre MV as ordinary
      // translational MC.  Without a model this is the historical zero MV.
      {
        bool gm_use_warp = false;
        inter::mc::Mv gm_mv{};
        global_mv_for_block(ref_index, r, c, w4, h4, gm_use_warp, gm_mv);
        std::vector<std::uint8_t> luma;
        bool predicted = false;
#if defined(AV2ENC_VIDEO_GME)
        if (gm_use_warp && gm_plan_ != nullptr &&
            ref_index < gm_plan_->num_refs) {
          std::string warp_error;
          predicted = warp_predict_plane(
              gm_plan_->params[static_cast<std::size_t>(ref_index)],
              references_[static_cast<std::size_t>(ref_index)], 0, ax0, ay0, w,
              h, r, c, w4, h4, luma, warp_error);
        }
#endif
        if (!predicted) {
          gm_use_warp = false;
          predicted = predict_luma(ref_index, gm_mv, ax0, ay0, w, h, luma);
        }
        if (predicted) {
          InterCandidate cand;
          cand.valid = true;
          cand.use_warp = gm_use_warp;
          cand.mode = syntax::kInterGlobalMv;
          cand.ref_index = ref_index;
          cand.mv = gm_mv;
          cand.sse = sse_vis(0, ax0, ay0, w, h, luma);
          if (trace_) {
            std::cerr << "  cand ref=" << ref_index << " global mv=("
                      << gm_mv.y << "," << gm_mv.x << ") warp="
                      << (gm_use_warp ? 1 : 0) << " sse=" << cand.sse
                      << "\n";
          }
          consider(cand);
          if (!global_cand_valid || cand.sse < global_cand.sse) {
            global_cand = cand;
            global_cand_valid = true;
          }
        }
      }
      if (have_stack && !options_.force_global_mv) {
        if (options_.enable_nearmv) {
          // §7.12.2.20 inserts MVP-combination candidates only when both
          // dimensions exceed 32.  merge::build_stack_checked already applies
          // that gate.
          // The historical search truncates small blocks to the first two
          // candidates.  With temporal MVs enabled the §7.12.2.7/.8 entries
          // can sit past that point (sorting by weight), so search the whole
          // stack; every index is codable and the writer validates it.
          const int near_limit =
              (w > 32 && h > 32) || options_.enable_ref_frame_mvs ||
                      options_.refmvbank
                  ? stack.count
                  : std::min(stack.count, 2);
          for (int i = 0; i < near_limit; ++i) {
            merge::ResolvedMode resolved;
            if (merge::resolve_single_mode(
                    stack, stack_params, merge::SingleMode::kNearMv, i,
                    inter::mc::Mv{0, 0}, &resolved) !=
                merge::Status::kOk) {
              continue;
            }
            std::vector<std::uint8_t> luma;
            const inter::mc::Mv near_mv = resolved.mv[0];
            if (!predict_luma(ref_index, near_mv, ax0, ay0, w, h, luma)) {
              continue;
            }
            InterCandidate cand;
            cand.valid = true;
            cand.mode = syntax::kInterNearMv;
            cand.ref_index = ref_index;
            cand.drl_index = i;
            cand.mv = near_mv;
            cand.from_temporal = stack.cand[i].from_temporal;
            cand.from_refmvbank = stack.cand[i].from_refmvbank;
            cand.sse = sse_vis(0, ax0, ay0, w, h, luma);
            if (options_.ref_mvs_stress && cand.from_temporal) {
              cand.sse = -1;
            }
            if (trace_) {
              std::cerr << "  cand ref=" << ref_index << " near drl=" << i
                        << " mv=(" << near_mv.y << "," << near_mv.x
                        << ") bank=" << (cand.from_refmvbank ? 1 : 0)
                        << " sse=" << cand.sse << "\n";
            }
            consider(cand);
          }
        }

        if (options_.enable_newmv) {
          // seq_force_integer_mv pins FrameMvPrecision to one pel for the
          // whole sequence, so the inter search codes one-pel MVDs.
          const int mv_precision =
              options_.seq_force_integer_mv ? 3 : options_.mv_precision;
          inter::mc::Mv me_mv{};
          if (run_me(ref_index, ax0, ay0, w, h, me_mv, error)) {
            me_mv = quantize_mv(me_mv, mv_precision);
            inter::mc::Mv pred = stack.cand[0].mv[0];
            // §5.20.7.13 assign_mv(): for a frame precision coarser than
            // 1/2 pel (MV_PRECISION_ONE_PEL) the decoder lowers the predictor
            // before adding DiffMv.  The quarter/half frames keep it as-is.
            if (mv_precision == 3) {
              me2::me2_lower_mv_precision(&pred.y, &pred.x,
                                          me2::kMvPrecisionOnePel);
            }
            // §5.20.7.13 only lowers the predictor when MvPrecision is
            // coarser than 1/2 pel.  Our quarter/half frames keep the
            // predictor as-is; the MVD itself is coded in precision units.
            const inter::mc::Mv delta{me_mv.y - pred.y, me_mv.x - pred.x};
            inter::mc::Mv mvd =
                mv_precision == 0
                    ? delta
                    : quantize_mv(delta, mv_precision);
            const int frame_prec =
                mv_precision == 0
                    ? me2::kMvPrecisionEighthPel
                    : (mv_precision == 1
                           ? me2::kMvPrecisionQuarterPel
                           : (mv_precision == 2
                                  ? me2::kMvPrecisionHalfPel
                                  : me2::kMvPrecisionOnePel));
            // §5.20.7.6 AMVD: the MVD magnitudes must be exactly
            // representable by Amvd_Index_To_Mvd.  When they are, the
            // index form codes fewer bits and reconstructs the same MV.
            bool use_amvd = false;
            int target = frame_prec;
            if (options_.enable_adaptive_mvd && (mvd.y != 0 || mvd.x != 0)) {
              int amvd_index = 0;
              const bool row_ok =
                  mvd.y == 0 ||
                  inter::mvprec::amvd_index_for_magnitude(mvd.y, &amvd_index);
              const bool col_ok =
                  mvd.x == 0 ||
                  inter::mvprec::amvd_index_for_magnitude(mvd.x, &amvd_index);
              use_amvd = row_ok && col_ok;
            }
            // §5.20.7.6 flexible MV resolution: pick the coarsest
            // signallable precision that represents the chosen MVD exactly.
            // From a 1/8-pel frame the reachable coarser step is 1/2 pel
            // (pb_mv_precision = 0); from a 1/4- or 1/2-pel frame it is
            // one pel.  An exact multiple of the coarser radix reconstructs
            // the identical MV with a smaller shell code.
            if (!use_amvd && options_.enable_flex_mvres) {
              const int coarse = frame_prec == me2::kMvPrecisionEighthPel
                                     ? me2::kMvPrecisionHalfPel
                                     : me2::kMvPrecisionOnePel;
              const int radix =
                  1 << (me2::kMvPrecisionEighthPel - coarse);
              if ((mvd.y % radix) == 0 && (mvd.x % radix) == 0) {
                target = coarse;
              }
            }
            // §5.20.7.13 mv_sign derivation.  The decoder derives the last
            // nonzero sign from the parity of the coded magnitudes, so the
            // encoder forces that sign before costing the candidate.
            inter::mvprec::MvdSignConfig sign_config;
            sign_config.enable_mvd_sign_derive =
                options_.enable_mvd_sign_derive;
            sign_config.frame_precision = frame_prec;
            sign_config.mv_precision = use_amvd ? frame_prec : target;
            sign_config.y_mode = inter::mvprec::kYModeNewMv;
            sign_config.is_adaptive_mvd = use_amvd;
            sign_config.is_compound = false;
            sign_config.ref_mv_idx = 0;
            bool sign_derived = false;
            if (inter::mvprec::mvd_sign_derive_allowed(sign_config)) {
              const int shift =
                  me2::kMvPrecisionEighthPel - sign_config.mv_precision;
              std::int32_t values[2] = {mvd.y, mvd.x};
              int last_sign = 0;
              int last_comp = -1;
              for (int comp = 0; comp < 2; ++comp) {
                if (values[comp] == 0) continue;
                const std::int32_t magnitude =
                    values[comp] < 0 ? -values[comp] : values[comp];
                last_sign += magnitude >> shift;
                last_comp = comp;
              }
              if (last_comp >= 0) {
                const bool negative = (last_sign & 1) != 0;
                std::int32_t& component = last_comp == 0 ? mvd.y : mvd.x;
                const std::int32_t magnitude =
                    component < 0 ? -component : component;
                component = negative ? -magnitude : magnitude;
                sign_derived = true;
              }
            }
            merge::ResolvedMode resolved;
            const bool resolved_ok =
                mv_precision == 0
                    ? merge::resolve_single_mode(
                          stack, stack_params, merge::SingleMode::kNewMv, 0,
                          mvd, &resolved) == merge::Status::kOk
                    : true;
            if (resolved_ok) {
              std::vector<std::uint8_t> luma;
              const inter::mc::Mv new_mv =
                  mv_precision == 0
                      ? resolved.mv[0]
                      : inter::mc::Mv{pred.y + mvd.y, pred.x + mvd.x};
              if (predict_luma(ref_index, new_mv, ax0, ay0, w, h, luma)) {
                InterCandidate cand;
                cand.valid = true;
                cand.mode = syntax::kInterNewMv;
                cand.ref_index = ref_index;
                cand.drl_index = 0;
                cand.mv = new_mv;
                cand.mvd = mv_precision == 0 ? resolved.mvd[0] : mvd;
                cand.use_amvd = use_amvd;
                cand.block_mv_precision =
                    target == frame_prec ? -1 : target;
                cand.mvd_sign_derived = sign_derived;
                cand.from_temporal = stack.cand[0].from_temporal;
                cand.sse =
                    sse_vis(0, ax0, ay0, w, h, luma);
                if (trace_) {
                  std::cerr << "  cand ref=" << ref_index << " new me=("
                            << me_mv.y << "," << me_mv.x << ") res=("
                            << new_mv.y << "," << new_mv.x << ") mvd=("
                            << cand.mvd.y << "," << cand.mvd.x
                            << ") amvd=" << (cand.use_amvd ? 1 : 0)
                            << " prec=" << cand.block_mv_precision
                            << " derive=" << (cand.mvd_sign_derived ? 1 : 0)
                            << " sse=" << cand.sse << "\n";
                }
                consider(cand);
              }
            }
          }
        }
      }
    }
  }

  // Compound candidate (§5.20.7.10 comp_mode + §7.13.3.18 AVERAGE blend).
  // The gated subset is non-joint GLOBAL_GLOBALMV over two references, so
  // both MVs are zero and the blend is the CWP_EQUAL average.  The blender
  // addresses frame coordinates, so it writes into per-frame scratch planes
  // and the block is copied out afterwards.
#if defined(AV2ENC_VIDEO_FRAMES)
  if (!keyframe_ && options_.compound && num_total_refs_ >= 2 &&
      w >= 8 && h >= 8 && !options_.force_intra && inter_fits) {
    const inter::mc::BlockRect rect{x0, y0, w, h};
    multiref::CompoundPrediction prediction;
    std::string comp_error;
    if (multiref::build_compound_prediction(
            references_[0], references_[1], rect,
            static_cast<int>(node.size), inter::mc::Mv{0, 0},
            inter::mc::Mv{0, 0}, leaf_interp_filter(),
            mi_cols_, mi_rows_, prediction, comp_error)) {
      std::uint8_t* dst[3] = {compound_pred_[0].samples.data(),
                              compound_pred_[1].samples.data(),
                              compound_pred_[2].samples.data()};
      int dst_stride[3] = {compound_pred_[0].stride,
                           compound_pred_[1].stride,
                           compound_pred_[2].stride};
      multiref::CompoundModeParams params;
      params.type = inter::adv::CompoundType::kAverage;
      params.cwp_weight = multiref::kCwpEqual;
      std::int64_t compound_sse = -1;
      if (options_.compound_wedge || options_.compound_diffwtd) {
        // §7.13.3.27-.30 masked compound decision over the block-local 1/16
        // predictions.  The search order (AVERAGE, wedge sign 0 then 1 over
        // signalable indices, then DIFFWTD mask_type 0/1) is the same
        // enumeration the CUDA backend uses, so both paths pick the same
        // block on an SSE tie.
        std::vector<std::uint8_t> search_scratch(
            static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
        inter::adv::CompoundSearchInput in{};
        in.bsize = static_cast<int>(node.size);
        in.rect = inter::adv::BlockRect{0, 0, w, h};
        in.p0 = prediction.pred[0][0].samples.data();
        in.p0_stride = prediction.pred[0][0].stride;
        in.p1 = prediction.pred[1][0].samples.data();
        in.p1_stride = prediction.pred[1][0].stride;
        in.source = source_.y.data() +
                    static_cast<std::size_t>(y0) *
                        static_cast<std::size_t>(source_.width) +
                    static_cast<std::size_t>(x0);
        in.source_stride = static_cast<int>(source_.width);
        in.scratch = search_scratch.data();
        in.scratch_stride = w;
        in.enable_average = true;
        in.enable_cwp = false;
        in.enable_wedge = options_.compound_wedge;
        in.enable_diffwtd = options_.compound_diffwtd;
        in.enable_imp_mask = false;
        const std::optional<inter::adv::CompoundDecision> decision =
            inter::adv::search_compound(in);
        if (!decision.has_value()) {
          error = "compound search failed";
          return false;
        }
        params.type = decision->type;
        params.cwp_weight = decision->cwp_weight;
        params.wedge_index = decision->wedge_index;
        params.wedge_sign = decision->wedge_sign;
        params.mask_type = decision->mask_type;
        compound_sse = decision->sse;
        if (trace_) {
          std::cerr << "  cand compound search type="
                    << static_cast<int>(decision->type)
                    << " wedge=" << decision->wedge_index
                    << " sign=" << decision->wedge_sign
                    << " mask_type=" << decision->mask_type
                    << " sse=" << decision->sse << "\n";
        }
      }
      if (multiref::apply_compound_prediction(prediction, params, dst,
                                              dst_stride, comp_error)) {
        sink.compound_y.resize(static_cast<std::size_t>(w) *
                               static_cast<std::size_t>(h));
        sink.compound_u.resize(static_cast<std::size_t>(cw) *
                               static_cast<std::size_t>(ch));
        sink.compound_v.resize(static_cast<std::size_t>(cw) *
                               static_cast<std::size_t>(ch));
        for (int yy = 0; yy < h; ++yy) {
          std::memcpy(sink.compound_y.data() +
                          static_cast<std::size_t>(yy) *
                              static_cast<std::size_t>(w),
                      compound_pred_[0].samples.data() +
                          static_cast<std::size_t>(y0 + yy) *
                              static_cast<std::size_t>(
                                  compound_pred_[0].stride) +
                          x0,
                      static_cast<std::size_t>(w));
        }
        for (int yy = 0; yy < ch; ++yy) {
          std::memcpy(sink.compound_u.data() +
                          static_cast<std::size_t>(yy) *
                              static_cast<std::size_t>(cw),
                      compound_pred_[1].samples.data() +
                          static_cast<std::size_t>(cy0 + yy) *
                              static_cast<std::size_t>(
                                  compound_pred_[1].stride) +
                          cx0,
                      static_cast<std::size_t>(cw));
          std::memcpy(sink.compound_v.data() +
                          static_cast<std::size_t>(yy) *
                              static_cast<std::size_t>(cw),
                      compound_pred_[2].samples.data() +
                          static_cast<std::size_t>(cy0 + yy) *
                              static_cast<std::size_t>(
                                  compound_pred_[2].stride) +
                          cx0,
                      static_cast<std::size_t>(cw));
        }
        InterCandidate cand;
        cand.valid = true;
        cand.compound = true;
        cand.mode = syntax::kInterGlobalMv;
        cand.ref_index = 0;
        cand.mv = inter::mc::Mv{0, 0};
        cand.sse = compound_sse >= 0
                       ? compound_sse
                       : sse_vis(0, x0, y0, w, h,
                                            sink.compound_y);
        cand.compound_type =
            params.type == inter::adv::CompoundType::kWedge
                ? 0
                : (params.type == inter::adv::CompoundType::kDiffWtd ? 1 : 2);
        cand.wedge_index = params.wedge_index;
        cand.wedge_sign = params.wedge_sign;
        cand.mask_type = params.mask_type;
        if (trace_) {
          std::cerr << "  cand compound global-global sse=" << cand.sse
                    << "\n";
        }
        consider(cand);
      }
    }
  }

  // §5.4.6 num_same_ref_compound > 0.  The same-reference pair (0,0) may be
  // coded as GLOBAL_GLOBALMV with zero MVs; the AVERAGE blend of the two
  // identical predictions is the §7.13.3.18 result for that pair.  This is
  // evaluated independently of the (0,1) carrier above so the flag produces
  // a real block decision (the gate counts the leaves it selects).
  if (!keyframe_ && options_.same_ref_compound && options_.compound &&
      num_total_refs_ >= 2 && w >= 8 && h >= 8 && !options_.force_intra &&
      inter_fits) {
    const inter::mc::Mv zeros[2] = {inter::mc::Mv{0, 0},
                                    inter::mc::Mv{0, 0}};
    multiref::CompoundModeParams params;
    params.type = inter::adv::CompoundType::kAverage;
    params.cwp_weight = multiref::kCwpEqual;
    std::string same_error;
    if (compound_plain(zeros, static_cast<int>(node.size), ax0, ay0, w, h,
                       sink.compound_y, sink.compound_u, sink.compound_v,
                       same_error, &params, /*ref0=*/0, /*ref1=*/0)) {
      InterCandidate cand;
      cand.valid = true;
      cand.compound = true;
      cand.same_ref = true;
      cand.mode = syntax::kInterGlobalMv;
      cand.ref_index = 0;
      cand.ref_index1 = 0;
      cand.compound_mode = 3;
      cand.compound_type = 2;  // AVERAGE (no comp_group_idx symbol)
      cand.mv = inter::mc::Mv{0, 0};
      cand.mv1 = inter::mc::Mv{0, 0};
      cand.sse = sse_vis(0, x0, y0, w, h, sink.compound_y);
      same_ref_best = cand;
    }
  }

  // §5.20.7.6 NEAR_NEARMV compound candidates (--video-opfl /
  // --video-refine-mv).  The two lists are merged independently; with
  // use_optflow the block instead codes a single RefMvIdx into the two-list
  // stack and runs §7.13.3.8-.12 on the prediction.
  if (!keyframe_ && options_.compound && options_.compound_near_near &&
      !options_.force_global_mv && num_total_refs_ >= 2 && w >= 8 && h >= 8 &&
      !options_.force_intra && inter_fits) {
    const int m = 5;  // max_drl_bits_minus_1 + 1
    merge::StackParams p0;
    merge::StackParams p1;
    merge::MvStack l0;
    merge::MvStack l1;
    const bool have0 = load_stack(r, c, w4, h4, 0, sink.state, p0, l0);
    const bool have1 = load_stack(r, c, w4, h4, 1, sink.state, p1, l1);
    const int d0 = frame_distance_[0];
    const int d1 = frame_distance_[1];
    const bool optflow_ok =
        options_.opfl_refine_type != 0 &&
        temporal::opfl_allowed_for_refs(d0, d1, false, false, false);
    // §5.20.7.17: a NEAR_NEARMV compound block is not switchable, so a
    // frame with enable_refinemv derives use_refinemv for every eligible
    // block.  The decision must match the prediction, so it is per block.
    bool block_refine = false;
    if (options_.enable_refinemv) {
      temporal::RefineMvConfig rc;
      rc.enable_refinemv = true;
      rc.opfl_refine_type =
          static_cast<temporal::RefineType>(options_.opfl_refine_type);
      rc.frame_is_switch = false;
      rc.is_compound = true;
      rc.skip_mode = false;
      rc.use_optflow = false;
      rc.is_global_global_mv = false;
      rc.motion_mode_simple = true;
      rc.has_newmv = false;
      rc.y_mode_near_near_mv = true;
      rc.y_mode_joint_new_mv = false;
      rc.block_width = w;
      rc.block_height = h;
      rc.order_hint = 64;
      rc.ref0_order_hint = 64 - d0;
      rc.ref1_order_hint = 64 - d1;
      temporal::RefineMvSyntax syntax;
      if (temporal::refine_mv_syntax(rc, &syntax) ==
          temporal::TemporalStatus::kOk) {
        block_refine = syntax.use_refinemv;
      }
    }
    std::vector<std::pair<inter::mc::Mv, inter::mc::Mv>> seen;
    const auto already = [&seen](inter::mc::Mv a, inter::mc::Mv b) {
      for (const auto& s : seen) {
        if (s.first.y == a.y && s.first.x == a.x && s.second.y == b.y &&
            s.second.x == b.x) {
          return true;
        }
      }
      seen.emplace_back(a, b);
      return false;
    };
    const auto evaluate = [&](inter::mc::Mv mv0, inter::mc::Mv mv1,
                              int drl0, int drl1, bool use_optflow) {
      if (already(mv0, mv1)) return;
      bool ok = false;
      const inter::mc::Mv mvs[2] = {mv0, mv1};
      multiref::CompoundModeParams chosen_params;
      std::int64_t chosen_sse = -1;
      if (use_optflow) {
        ok = predict_compound_optflow(sink, mvs, ax0, ay0, w, h, error);
      } else if (block_refine) {
        ok = predict_compound_refinemv(sink, mvs, ax0, ay0, w, h, error);
      } else if (options_.enable_cwp_weighted ||
                 options_.enable_imp_msk_bld) {
        ok = compound_search_average(
            mvs, static_cast<int>(node.size), ax0, ay0, w, h,
            sink.compound_y, sink.compound_u, sink.compound_v, chosen_params,
            chosen_sse, error);
      } else {
        ok = compound_plain(mvs, static_cast<int>(node.size), ax0, ay0, w, h,
                            sink.compound_y, sink.compound_u, sink.compound_v,
                            error);
      }
      if (!ok) return;
      InterCandidate cand;
      cand.valid = true;
      cand.compound = true;
      cand.mode = syntax::kInterNearMv;
      cand.ref_index = 0;
      cand.mv = mv0;
      cand.mv1 = mv1;
      cand.drl_index = drl0;
      cand.drl_index1 = drl1;
      cand.compound_mode = 0;  // NEAR_NEARMV
      cand.use_optflow = use_optflow;
      cand.use_refinemv = block_refine && !use_optflow;
      cand.cwp_index = chosen_params.cwp_index;
      cand.cwp_weight = chosen_params.cwp_weight;
      cand.use_imp_mask = chosen_params.use_imp_mask;
      cand.sse = chosen_sse >= 0
                     ? chosen_sse
                     : sse_vis(0, ax0, ay0, w, h,
                                          sink.compound_y);
      if (trace_) {
        std::cerr << "  cand compound near-near mv0=(" << mv0.y << ","
                  << mv0.x << ") mv1=(" << mv1.y << "," << mv1.x
                  << ") drl=(" << drl0 << "," << drl1 << ") optflow="
                  << (use_optflow ? 1 : 0) << " refinemv="
                  << (cand.use_refinemv ? 1 : 0) << " cwp="
                  << cand.cwp_index << " imp="
                  << (cand.use_imp_mask ? 1 : 0) << " sse=" << cand.sse
                  << "\n";
      }
      consider(cand);
    };
    if (optflow_ok && !block_refine) {
      merge::StackParams cparams =
          make_stack_params(r, c, w4, h4, 0, sink.state);
      cparams.ref_frame[1] = 1;
      cparams.is_compound = true;
      bool gm_warp = false;
      inter::mc::Mv gm_mv1{};
      global_mv_for_block(1, r, c, w4, h4, gm_warp, gm_mv1);
      cparams.global_mv[1] = gm_mv1;
      merge::MvStack compound;
      if (merge::build_stack_checked(cparams, &compound) ==
          merge::Status::kOk) {
        const int limit = std::min(compound.count, m);
        for (int i = 0; i < limit; ++i) {
          const inter::mc::Mv mv0 =
              merge::clamp_mv_to_integer(compound.cand[i].mv[0]);
          const inter::mc::Mv mv1 =
              merge::clamp_mv_to_integer(compound.cand[i].mv[1]);
          evaluate(mv0, mv1, i, 0, /*use_optflow=*/true);
        }
      }
    }
    if (have0 && have1) {
      int pair_limit = block_refine ? 2 : m;
      if (const char* limit = std::getenv("AV2ENC_COMPOUND_PAIR_LIMIT")) {
        const int value = std::atoi(limit);
        if (value > 0) pair_limit = std::min(pair_limit, value);
      }
      const int limit0 = std::min(l0.count, pair_limit);
      const int limit1 = std::min(l1.count, pair_limit);
      for (int i = 0; i < limit0; ++i) {
        for (int j = 0; j < limit1; ++j) {
          merge::MvStack pair[2] = {l0, l1};
          inter::mc::Mv resolve[2];
          if (merge::resolve_compound_near_near(pair, i, j, resolve) !=
              merge::Status::kOk) {
            continue;
          }
          evaluate(resolve[0], resolve[1], i, j, /*use_optflow=*/false);
        }
      }
    }
  }

  // §5.20.7.6 compound NEWMV carrier (AV2ENC_COMPOUND_NEWMV).  The depth arm
  // could only code the per-list stack pair because NEAR_NEARMV has no MVD
  // symbol.  These modes code the plan search MV:
  //   NEAR_NEWMV  list-0 stack entry + list-1 MVD (two DRL indices),
  //   NEW_NEARMV  list-0 MVD + list-1 two-list stack entry (one DRL index),
  //   JOINT_NEWMV one MVD, §5.20.7.13 projMv derives the other list.
  // The search is the plan ME MV per reference plus a small on-axis
  // ±1/8..1-pel refinement.  Every candidate is priced by the same leaf RD
  // as the NEAR_NEARMV carrier.
  if (!keyframe_ && options_.compound && options_.compound_near_near &&
      compound_newmv_enabled() && !options_.force_global_mv &&
      num_total_refs_ >= 2 && w >= 8 && h >= 8 && !options_.force_intra &&
      inter_fits && options_.mv_precision == 0 &&
      !options_.seq_force_integer_mv && !options_.enable_adaptive_mvd &&
      !options_.enable_flex_mvres && !options_.enable_mvd_sign_derive &&
      !options_.enable_refinemv && options_.opfl_refine_type == 0) {
    merge::StackParams p0;
    merge::StackParams p1;
    merge::MvStack l0;
    merge::MvStack l1;
    const bool have0 = load_stack(r, c, w4, h4, 0, sink.state, p0, l0);
    const bool have1 = load_stack(r, c, w4, h4, 1, sink.state, p1, l1);
    // The two-list stack the decoder builds for the single-DRL compound modes.
    merge::StackParams cparams = make_stack_params(r, c, w4, h4, 0, sink.state);
    cparams.ref_frame[1] = 1;
    cparams.is_compound = true;
    bool gm_warp = false;
    inter::mc::Mv gm_mv1{};
    global_mv_for_block(1, r, c, w4, h4, gm_warp, gm_mv1);
    cparams.global_mv[1] = gm_mv1;
    merge::MvStack compound_stack;
    const bool have_compound =
        merge::build_stack_checked(cparams, &compound_stack) ==
            merge::Status::kOk &&
        compound_stack.count > 0;
    // The plan ME MV for each reference is already cached by the single-ref
    // NEWMV search.  A missing/failed search simply drops that reference's
    // NEWMV candidates.
    inter::mc::Mv me[2];
    bool have_me[2] = {false, false};
    for (int ref = 0; ref < 2; ++ref) {
      if (run_me(ref, ax0, ay0, w, h, me[ref], error)) {
        me[ref] = quantize_mv(me[ref], options_.mv_precision);
        have_me[ref] = true;
      }
    }
    // Compound-coupled refinement.  The near list is fixed by the stack, so
    // the coded new MV is searched against the actual pairwise blend rather
    // than each reference's single-prediction SSE.  The seed is the better of
    // the plan ME MV and the zero-delta stack predictor; the on-axis
    // ±1/8..1-pel offsets around it are the small refinement.
    const int radius = compound_newmv_radius();
    std::vector<inter::mc::Mv> offsets;
    offsets.push_back(inter::mc::Mv{0, 0});
    if (compound_newmv_refine()) {
      for (int step : {1, 2, 4, 8}) {
        if (step > radius) continue;
        offsets.push_back(inter::mc::Mv{step, 0});
        offsets.push_back(inter::mc::Mv{-step, 0});
        offsets.push_back(inter::mc::Mv{0, step});
        offsets.push_back(inter::mc::Mv{0, -step});
      }
    }
    const bool imp_mask_on = options_.enable_imp_msk_bld;
    const auto blend_sse = [&](inter::mc::Mv a, inter::mc::Mv b,
                               std::int64_t& sse) -> bool {
      const inter::mc::Mv mvs[2] = {a, b};
      multiref::CompoundModeParams eq;
      eq.type = inter::adv::CompoundType::kAverage;
      eq.cwp_index = 0;
      eq.cwp_weight = multiref::kCwpEqual;
      eq.use_imp_mask =
          imp_mask_on &&
          multiref::implicit_mask_eligible(
              /*seq_enable_imp_msk_bld=*/true, eq,
              /*global_global_mv=*/false, /*refs_scaled=*/false,
              /*compound_warp=*/false);
      if (!compound_plain(mvs, static_cast<int>(node.size), ax0, ay0, w, h,
                          sink.compound_y, sink.compound_u, sink.compound_v,
                          error, &eq, 0, 1)) {
        return false;
      }
      sse = sse_vis(0, ax0, ay0, w, h, sink.compound_y);
      return true;
    };
    const auto mv_shell = [](inter::mc::Mv a, inter::mc::Mv b) {
      return std::abs(a.y - b.y) + std::abs(a.x - b.x);
    };
    // Searches the coded list's MV against a fixed partner.  `new_is_list1`
    // selects NEAR_NEWMV (list 0 fixed) or NEW_NEARMV (list 1 fixed).
    const auto refine_pair = [&](bool new_is_list1, inter::mc::Mv fixed0,
                                 inter::mc::Mv fixed1,
                                 inter::mc::Mv me_target,
                                 inter::mc::Mv pred_target,
                                 inter::mc::Mv* best_target) {
      std::int64_t sse_me = -1;
      std::int64_t sse_pred = -1;
      const bool have_me =
          blend_sse(new_is_list1 ? fixed0 : me_target,
                    new_is_list1 ? me_target : fixed1, sse_me);
      const bool have_pred =
          blend_sse(new_is_list1 ? fixed0 : pred_target,
                    new_is_list1 ? pred_target : fixed1, sse_pred);
      if (!have_me && !have_pred) return false;
      inter::mc::Mv center = pred_target;
      inter::mc::Mv best = pred_target;
      std::int64_t best_sse = sse_pred;
      if (have_me && (!have_pred || sse_me < sse_pred)) {
        center = me_target;
        best = me_target;
        best_sse = sse_me;
      }
      const auto consider_target = [&](inter::mc::Mv target) {
        std::int64_t sse = 0;
        if (!blend_sse(new_is_list1 ? fixed0 : target,
                       new_is_list1 ? target : fixed1, sse)) {
          return;
        }
        if (sse < best_sse ||
            (sse == best_sse &&
             mv_shell(target, pred_target) <
                 mv_shell(best, pred_target))) {
          best_sse = sse;
          best = target;
        }
      };
      for (const inter::mc::Mv& off : offsets) {
        consider_target(inter::mc::Mv{center.y + off.y, center.x + off.x});
      }
      *best_target = best;
      return true;
    };
    struct SeenPair {
      int mode;
      bool joint;
      inter::mc::Mv a;
      inter::mc::Mv b;
    };
    std::vector<SeenPair> seen_pairs;
    const auto already_seen = [&](int mode, bool joint, inter::mc::Mv a,
                                  inter::mc::Mv b) {
      for (const SeenPair& s : seen_pairs) {
        if (s.mode == mode && s.joint == joint && s.a.y == a.y &&
            s.a.x == a.x && s.b.y == b.y && s.b.x == b.x) {
          return true;
        }
      }
      seen_pairs.push_back(SeenPair{mode, joint, a, b});
      return false;
    };
    const auto add_new = [&](int mode, bool joint, inter::mc::Mv mv0,
                             inter::mc::Mv mv1, inter::mc::Mv mvd, int drl0,
                             int drl1) {
      if (already_seen(mode, joint, mv0, mv1)) return;
      const inter::mc::Mv mvs[2] = {mv0, mv1};
      multiref::CompoundModeParams chosen;
      std::int64_t chosen_sse = -1;
      bool ok = false;
      if (joint && (options_.enable_cwp_weighted ||
                    options_.enable_imp_msk_bld)) {
        ok = compound_search_average(
            mvs, static_cast<int>(node.size), ax0, ay0, w, h,
            sink.compound_y, sink.compound_u, sink.compound_v, chosen,
            chosen_sse, error);
      } else {
        // NEAR_NEWMV / NEW_NEARMV never code cwp_idx, so the only prediction
        // is the equal-weight AVERAGE blend (or its derived implicit mask).
        chosen.type = inter::adv::CompoundType::kAverage;
        chosen.cwp_index = 0;
        chosen.cwp_weight = multiref::kCwpEqual;
        chosen.use_imp_mask =
            imp_mask_on &&
            multiref::implicit_mask_eligible(
                /*seq_enable_imp_msk_bld=*/true, chosen,
                /*global_global_mv=*/false, /*refs_scaled=*/false,
                /*compound_warp=*/false);
        ok = compound_plain(mvs, static_cast<int>(node.size), ax0, ay0, w, h,
                            sink.compound_y, sink.compound_u, sink.compound_v,
                            error, &chosen, 0, 1);
      }
      if (!ok) return;
      InterCandidate cand;
      cand.valid = true;
      cand.compound = true;
      cand.mode = syntax::kInterNearMv;
      cand.ref_index = 0;
      cand.ref_index1 = 1;
      cand.mv = mv0;
      cand.mv1 = mv1;
      cand.mvd = mvd;
      cand.drl_index = drl0;
      cand.drl_index1 = drl1;
      cand.compound_mode = mode;
      cand.compound_joint = joint;
      cand.compound_type = 2;  // AVERAGE
      cand.cwp_index = chosen.cwp_index;
      cand.cwp_weight = chosen.cwp_weight;
      cand.use_imp_mask = chosen.use_imp_mask;
      cand.sse = chosen_sse >= 0
                     ? chosen_sse
                     : sse_vis(0, ax0, ay0, w, h, sink.compound_y);
      if (trace_) {
        std::cerr << "  cand compound new mode=" << mode
                  << " joint=" << (joint ? 1 : 0) << " mv0=(" << mv0.y << ","
                  << mv0.x << ") mv1=(" << mv1.y << "," << mv1.x
                  << ") mvd=(" << mvd.y << "," << mvd.x << ") drl=(" << drl0
                  << "," << drl1 << ") cwp=" << cand.cwp_index << " sse="
                  << cand.sse << "\n";
      }
      consider(cand);
    };
    const int mode_mask = compound_newmv_modes();
    // NEAR_NEWMV: list 0 is the first per-list stack entry; the new MV is
    // list 1 and its predictor is that list's first stack entry.
    if ((mode_mask & 1) && have0 && have1 && l0.count > 0 && l1.count > 0 &&
        have_me[1]) {
      const inter::mc::Mv near0 =
          merge::clamp_mv_to_integer(l0.cand[0].mv[0]);
      const inter::mc::Mv pred1 =
          merge::clamp_mv_to_integer(l1.cand[0].mv[0]);
      inter::mc::Mv best_target = pred1;
      if (refine_pair(/*new_is_list1=*/true, near0, pred1, me[1], pred1,
                      &best_target)) {
        const auto add_target = [&](inter::mc::Mv target) {
          const inter::mc::Mv actual = merge::clamp_mv_to_integer(
              inter::mc::Mv{target.y, target.x});
          add_new(1, false, near0, actual,
                  inter::mc::Mv{actual.y - pred1.y, actual.x - pred1.x},
                  /*drl0=*/0, /*drl1=*/0);
        };
        add_target(pred1);
        add_target(me[1]);
        add_target(best_target);
      }
    }
    // NEW_NEARMV / JOINT_NEWMV read the two-list stack at one DRL index.
    if ((mode_mask & (2 | 4)) && have_compound &&
        compound_stack.count > 0) {
      const inter::mc::Mv pred0 =
          merge::clamp_mv_to_integer(compound_stack.cand[0].mv[0]);
      const inter::mc::Mv pred1 =
          merge::clamp_mv_to_integer(compound_stack.cand[0].mv[1]);
      if ((mode_mask & 2) && have_me[0]) {
        inter::mc::Mv best_target = pred0;
        if (refine_pair(/*new_is_list1=*/false, pred0, pred1, me[0], pred0,
                        &best_target)) {
          const auto add_target = [&](inter::mc::Mv target) {
            const inter::mc::Mv actual =
                merge::clamp_mv_to_integer(inter::mc::Mv{target.y, target.x});
            add_new(2, false, actual, pred1,
                    inter::mc::Mv{actual.y - pred0.y, actual.x - pred0.x},
                    /*drl0=*/0, /*drl1=*/0);
          };
          add_target(pred0);
          add_target(me[0]);
          add_target(best_target);
        }
      }
      if ((mode_mask & 4) && have_me[0] && have_me[1]) {
        const int d0 = frame_distance_[0];
        const int d1 = frame_distance_[1];
        const inter::mvprec::JointMvdPlan plan =
            inter::mvprec::joint_mvd_plan(-d0, -d1, false, false);
        inter::mc::Mv pred[2] = {pred0, pred1};
        const int base = plan.base_list;
        const auto joint_pair = [&](inter::mc::Mv target,
                                    inter::mc::Mv pair[2]) {
          const inter::mc::Mv diff{target.y - pred[base].y,
                                   target.x - pred[base].x};
          inter::mvprec::assign_joint_mvd(pred, diff, plan,
                                          /*jmvd_scale_mode=*/0,
                                          /*use_amvd=*/false, pair);
          return diff;
        };
        std::int64_t sse_me = -1;
        std::int64_t sse_pred = -1;
        inter::mc::Mv pair_me[2];
        inter::mc::Mv pair_pred[2];
        joint_pair(me[base], pair_me);
        joint_pair(pred[base], pair_pred);
        const bool have_me_sse = blend_sse(pair_me[0], pair_me[1], sse_me);
        const bool have_pred_sse =
            blend_sse(pair_pred[0], pair_pred[1], sse_pred);
        if (have_me_sse || have_pred_sse) {
          inter::mc::Mv center = pred[base];
          inter::mc::Mv best_target = pred[base];
          std::int64_t best_sse = sse_pred;
          if (have_me_sse && (!have_pred_sse || sse_me < sse_pred)) {
            center = me[base];
            best_target = me[base];
            best_sse = sse_me;
          }
          for (const inter::mc::Mv& off : offsets) {
            const inter::mc::Mv target{center.y + off.y, center.x + off.x};
            inter::mc::Mv pair[2];
            joint_pair(target, pair);
            std::int64_t sse = 0;
            if (!blend_sse(pair[0], pair[1], sse)) continue;
            if (sse < best_sse ||
                (sse == best_sse &&
                 mv_shell(target, pred[base]) <
                     mv_shell(best_target, pred[base]))) {
              best_sse = sse;
              best_target = target;
            }
          }
          const auto add_joint = [&](inter::mc::Mv target) {
            inter::mc::Mv jpair[2];
            const inter::mc::Mv diff = joint_pair(target, jpair);
            add_new(0, true, jpair[0], jpair[1], diff, /*drl0=*/0,
                    /*drl1=*/0);
          };
          add_joint(pred[base]);
          add_joint(me[base]);
          add_joint(best_target);
        }
      }
    }
  }
  // Prefer the same-reference compound carrier on an SSE tie with the best
  // ordinary candidate.  Both predictions are decoder-exact; the preference
  // makes --video-same-ref-compound actually exercise the §5.20.7.10
  // compound_mode_same_refs syntax instead of being shadowed by GLOBALMV.
  if (same_ref_best.valid &&
      (!best.valid || same_ref_best.sse <= best.sse)) {
    best = same_ref_best;
  }
#endif  // AV2ENC_VIDEO_FRAMES

  // Intra fallback: DC prediction + DCT residual, the M1 transform path.
  // With --video-intra-full the luma leaf runs the still product's
  // 61-candidate search (see src/pipeline/video/video_intra_search.h);
  // chroma stays DC_PRED because the M2 syntax cannot signal more.
  bool have_intra = false;
  std::int64_t intra_sse = 0;
  std::vector<std::uint8_t> intra_y_pred;
  std::vector<std::uint8_t> intra_u_pred;
  std::vector<std::uint8_t> intra_v_pred;
  TxPlan intra_y;
  TxPlan intra_u;
  TxPlan intra_v;
  // AV2ENC_RD_UVLEVEL: the chroma search scored these plans (including the
  // winning coefficient-level variant), so the leaf emits them directly
  // instead of re-running compute_tx and coding a different coefficient set.
  bool intra_chroma_ready = false;
  int intra_cctx_type = 0;
  std::uint8_t intra_y_mode = 0;  // AV2 YMode (0 = DC_PRED)
  std::int8_t intra_y_angle_delta = 0;
  std::uint8_t intra_mrl_index = 0;
  bool intra_mrl_sec_index = false;
  bool intra_use_dip = false;
  int intra_dip_mode = 0;
  bool intra_dip_transpose = false;
  bool intra_is_cfl = false;
  int intra_cfl_index = 1;
  int intra_cfl_mh_dir = 0;
  // AV2ENC_RD_UVFULL: the coded UVMode (DC unless the chroma search wins with a
  // directional/smooth/Paeth mode).  CfL keeps DC_PRED as its base mode.
  int intra_uv_mode = 0;
  // The reconstruction planes are tile-local, so plain bounds give the
  // decoder's §5.20.9.1 availability at every tile edge.  The bottom bound
  // is the coded MI extent, not the SB-aligned --video-edge-inter buffer.
  const int luma_max_x = recon_[0].width - 1;
  const int luma_max_y = mi_rows_ * 4 - 1;
  const bool luma_left = x0 > 0;
  const bool luma_above = y0 > 0;
  const int chroma_max_x = recon_[1].width - 1;
  const int chroma_max_y = mi_rows_ * 2 - 1;
  const bool chroma_left = cx0 > 0;
  const bool chroma_above = cy0 > 0;
  if (keyframe_ || options_.enable_intra_fallback || options_.force_intra) {
    bool luma_ok = chroma_part;
    if (!chroma_part) {
    const intra_search::Availability luma_avail = intra_search::availability(
        st.decoded[0], x0 / 4, y0 / 4, w / 4, h / 4, mi_cols_, mi_rows_,
        /*sb_mi=*/sb_size_ / 4);
    if (options_.enable_intra_full) {
      bool above_smooth = false;
      bool left_smooth = false;
      luma_neighbor_smooth(r, c, above_smooth, left_smooth);
      intra_search::LumaChoice choice;
      intra_y_pred.resize(static_cast<std::size_t>(w) *
                          static_cast<std::size_t>(h));
      luma_ok = intra_search::search_luma(
          recon_[0].samples.data(), recon_[0].stride, x0, y0, w, h,
          luma_avail,
          above_smooth, left_smooth, /*enable_edge_filter=*/true,
          source_.y.data(), static_cast<int>(source_.width),
          static_cast<int>(source_.width) - 1,
          static_cast<int>(source_.height) - 1, intra_y_pred.data(), choice,
          options_.enable_ibp);
      if (luma_ok) {
        intra_y_mode = static_cast<std::uint8_t>(choice.mode);
        intra_y_angle_delta = static_cast<std::int8_t>(choice.angle_delta);
        if (options_.enable_mrl) {
          // §5.20.5.5 MRL: only directional candidates can signal
          // mrl_index.  Re-use the still encoder's candidate order and
          // strict `<` tie-break; the baseline search already holds the
          // mrl_index 0 winner.
          const int sec_allowed = !(w == 4 && h == 4);
          std::int64_t best_sse = choice.sse;
          std::array<intra::ModeCandidate, intra::kMaxModeCandidates>
              candidates;
          const int count = intra::build_mode_candidates(candidates, true);
          for (int i = 0; i < count; ++i) {
            const intra::ModeCandidate& candidate =
                candidates[static_cast<std::size_t>(i)];
            if (!intra::is_directional(candidate.mode)) continue;
            for (int mrl_index = 1; mrl_index <= mrl::kMaxMrlIndex;
                 ++mrl_index) {
              for (int sec = 0; sec <= (sec_allowed ? 1 : 0); ++sec) {
                std::vector<std::uint8_t> mrl_pred;
                if (!predict_luma_mrl(
                        recon_[0], luma_avail, x0, y0, w, h, above_smooth,
                        left_smooth, static_cast<int>(candidate.mode),
                        candidate.angle_delta, mrl_index, sec != 0,
                        /*sb_height=*/64, mrl_pred, error)) {
                  return false;
                }
                const std::int64_t mrl_sse = sse_against_source(
                    source_, 0, x0, y0, w, h, mrl_pred);
                if (mrl_sse < best_sse) {
                  best_sse = mrl_sse;
                  intra_y_mode =
                      static_cast<std::uint8_t>(candidate.mode);
                  intra_y_angle_delta =
                      static_cast<std::int8_t>(candidate.angle_delta);
                  intra_mrl_index = static_cast<std::uint8_t>(mrl_index);
                  intra_mrl_sec_index = sec != 0;
                  intra_y_pred = std::move(mrl_pred);
                }
              }
            }
          }
          if (intra_mrl_index == 0) intra_mrl_sec_index = false;
        }
      }
    } else {
      luma_ok = intra_dc_predict(recon_[0], source_, 0, x0, y0, w, h,
                                 luma_max_x, luma_max_y, luma_left,
                                 luma_above, options_.enable_ibp,
                                 intra_y_pred);
    }
    // §5.20.7.9/§7.13.2.3: DIP is an alternative to ordinary DC at the same
    // YMode on the intra keyframe, not a new mode.  Evaluate the normative
    // candidate set and keep it only when its prediction SSE beats the
    // incumbent.
    if (options_.enable_dip && keyframe_ && luma_ok &&
        dip::dip_block_available(w, h)) {
      dip::Block dip_block;
      dip_block.recon = dip::PlaneView::from_u8(
          recon_[0].samples.data(), recon_[0].width);
      dip_block.x = x0;
      dip_block.y = y0;
      dip_block.w = w;
      dip_block.h = h;
      dip_block.max_x = luma_max_x;
      dip_block.max_y = luma_max_y;
      dip_block.have_left = luma_left;
      dip_block.have_above = luma_above;
      dip_block.num4_above_right = luma_avail.num4_above_right;
      dip_block.num4_below_left = luma_avail.num4_below_left;
      dip::Edges dip_edges;
      if (dip::prepare_edges(dip_block, 8, dip_edges) != dip::Status::kOk) {
        error = "dip::prepare_edges failed";
        return false;
      }
      std::vector<std::uint8_t> dip_src(
          static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
      for (int yy = 0; yy < h; ++yy) {
        for (int xx = 0; xx < w; ++xx) {
          dip_src[static_cast<std::size_t>(yy) *
                      static_cast<std::size_t>(w) +
                  static_cast<std::size_t>(xx)] =
              static_cast<std::uint8_t>(
                  source_sample(source_, 0, ax0 + xx, ay0 + yy));
        }
      }
      dip::DecisionParams dip_params;
      dip_params.enable_dip = true;
      dip_params.y_mode = 0;  // DC_PRED
      dip_params.palette_size_y = 0;
      dip_params.tree_is_chroma_part = false;
      dip_params.metric = dip::CostMetric::kSse;
      dip::Decision dip_decision;
      if (dip::decide(dip_edges, dip_src.data(),
                      static_cast<std::size_t>(w), dip_params,
                      dip_decision) != dip::Status::kOk ||
          !dip_decision.available) {
        error = "dip::decide failed";
        return false;
      }
      const std::int64_t incumbent_sse =
          sse_vis(0, ax0, ay0, w, h, intra_y_pred);
      if (static_cast<std::int64_t>(dip_decision.cost.sse) <
          incumbent_sse) {
        dip::Params dip_pred_params;
        dip_pred_params.mode = dip_decision.mode;
        dip_pred_params.transpose = dip_decision.transpose;
        std::vector<std::uint8_t> dip_pred(
            static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
        if (dip::predict_block8(dip_block, 8, dip_pred_params,
                                dip_pred.data(),
                                static_cast<std::size_t>(w)) !=
            dip::Status::kOk) {
          error = "dip::predict_block8 failed";
          return false;
        }
        intra_y_pred = std::move(dip_pred);
        intra_y_mode = 0;
        intra_y_angle_delta = 0;
        intra_mrl_index = 0;
        intra_mrl_sec_index = false;
        intra_use_dip = true;
        intra_dip_mode = dip_decision.mode;
        intra_dip_transpose = dip_decision.transpose;
      }
    }
    // --video-key-intra-txpart: an intra luma partition needs per-part
    // prediction, so it replaces the single-predictor TX search here.  DIP is
    // a whole-block alternative prediction and stays TX_PARTITION_NONE.
    const bool intra_tx_partition_search =
        options_.enable_key_intra_tx_part && options_.enable_tx_search &&
        keyframe_ && !chroma_part && !intra_use_dip &&
        intra_y_pred.size() ==
            static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    bool part_above_smooth = false;
    bool part_left_smooth = false;
    if (intra_tx_partition_search) {
      luma_neighbor_smooth(r, c, part_above_smooth, part_left_smooth);
    }
    if (!luma_ok ||
        (options_.enable_tx_search
             ? (intra_tx_partition_search
                    ? !encode_intra_luma_tx_partition(
                          r, c, w, h, node.size, intra_y_pred, intra_y_mode,
                          intra_y_angle_delta, intra_mrl_index,
                          intra_mrl_sec_index, part_above_smooth,
                          part_left_smooth, luma_mask_row, sink.state,
                          &intra_y, error)
                    : !search_luma_tx(
                          source_, ax0, ay0, w, h, intra_y_pred,
                          st.current_qindex, txtsearch::Mode::kIntra,
                          key_tx_rd_lambda(options_, st.current_qindex) *
                              perceptual::tx_lambda_scale(perceptual_weights),
                          options_.enable_idtx_intra,
                          std::max(0, mrl_stream::mapped_intra_mode_for_mrl(
                                          intra_y_mode, intra_y_angle_delta,
                                          intra_mrl_index, w, h)),
                          options_.enable_tcq,
                          ::av2enc::still::qctx::slice_for_base_q_idx(
                              options_.tx_context_rate ? base_q_index()
                                                       : st.current_qindex),
                          /*enable_inter_ddt=*/false,
                          /*search_partition=*/false,
                          /*context_rate=*/options_.tx_context_rate,
                          options_.trace, /*allow_rdoq=*/false,
                          /*rdoq_qctx=*/0,
                          rdoq_sse_scale_at(options_, st.current_qindex),
                          intra_y, error, luma_mask_row))
             : !compute_tx(source_, 0, ax0, ay0, w, h, intra_y_pred,
                           st.plane_quant[0], intra_y, error,
                           qm_enabled_ ? &qm_state_ : nullptr,
                           options_.enable_tcq,
                           ::av2enc::still::qctx::slice_for_base_q_idx(
                               st.current_qindex),
                           /*is_inter=*/false, /*allow_rdoq=*/false,
                           /*rdoq_qctx=*/0,
                           rdoq_sse_scale_at(options_, st.current_qindex),
                           luma_mask_row))) {
      error = "intra prediction failed";
      return false;
    }
    // §7.15.3 luma IST refinement (--video-ist).  Applies the still
    // encoder's analysis projection to the quantised luma levels and keeps
    // the candidate that minimises SSE + lambda * symbol bits.
    if (options_.enable_ist &&
        !refine_luma_ist(source_, ax0, ay0, w, h, intra_y_pred,
                         st.quant_step, intra_y_mode, intra_y_angle_delta,
                         intra_mrl_index, intra_y, error)) {
      return false;
    }
    // §7.13.5 CfL reads the reconstructed luma of this block, so the luma
    // reconstruction must be in the plane before the chroma decision.  A
    // losing intra candidate is overwritten by the inter reconstruction.
    write_plane_block(recon_[0], x0, y0, w, h, intra_y.recon);
    if (pred_dump_enabled() && keyframe_) {
      pred_dump_record(source_, x0, y0, w, h, intra_y_pred, intra_y.recon);
    }
    }  // !chroma_part
    if (owns_chroma) {
    if (!intra_dc_predict(recon_[1], source_, 1, cx0, cy0, cw, ch,
                          chroma_max_x, chroma_max_y, chroma_left,
                          chroma_above, options_.enable_ibp,
                          intra_u_pred) ||
        !intra_dc_predict(recon_[2], source_, 2, cx0, cy0, cw, ch,
                          chroma_max_x, chroma_max_y, chroma_left,
                          chroma_above, options_.enable_ibp,
                          intra_v_pred)) {
      error = "intra prediction failed";
      return false;
    }
    // IBP changes the DC candidate (§7.13.2.12 post-filter) and the CLI already
    // rejects --video-ibp with --video-cfl, so the search does not run there.
    const bool uv_full = rd_uvfull_enabled() && !options_.enable_ibp;
    if (options_.enable_cfl || uv_full) {
      // Opt-in CfL chroma decision (§7.13.5).  Without AV2ENC_RD_UVFULL the
      // candidate set is the M1 still product's default {DC, CfL} with derived
      // alpha; with it, CfL is one candidate of the full UV search below.
      const std::int64_t dc_sse_u =
          sse_vis(1, acx0, acy0, cw, ch, intra_u_pred);
      const int frame_chroma_width = static_cast<int>((source_.width + 1U) / 2U);
      const int frame_chroma_height =
          static_cast<int>((source_.height + 1U) / 2U);
      chroma_search::PlaneView u_plane;
      u_plane.recon = recon_[1].samples.data();
      u_plane.recon_stride = recon_[1].stride;
      u_plane.recon_width = recon_[1].width;
      u_plane.recon_height = mi_rows_ * 2;
      // The helper indexes the source with the block origin, so the pointer
      // carries the tile origin only (coded coordinates stay tile-local).
      const int tile_cx = tile_x_ / 2;
      const int tile_cy = tile_y_ / 2;
      u_plane.source = source_.u.data() +
                       static_cast<std::size_t>(tile_cy) *
                           static_cast<std::size_t>(frame_chroma_width) +
                       static_cast<std::size_t>(tile_cx);
      u_plane.source_stride = frame_chroma_width;
      u_plane.source_max_x = frame_chroma_width - 1 - tile_cx;
      u_plane.source_max_y = frame_chroma_height - 1 - tile_cy;
      chroma_search::PlaneView v_plane = u_plane;
      v_plane.recon = recon_[2].samples.data();
      v_plane.recon_stride = recon_[2].stride;
      v_plane.source = source_.v.data() +
                       static_cast<std::size_t>(tile_cy) *
                           static_cast<std::size_t>(frame_chroma_width) +
                       static_cast<std::size_t>(tile_cx);
      const intra_search::Availability chroma_avail =
          intra_search::availability(st.decoded[1], cx0 / 4, cy0 / 4, cw / 4,
                                     ch / 4, mi_cols_ / 2, mi_rows_ / 2,
                                     /*sb_mi=*/sb_size_ / 8);
      chroma_search::Options chroma_options;
      chroma_options.enable_cfl = true;
      chroma_options.ds_filter = 0;
      chroma_options.enable_mhccp = options_.enable_mhccp;
      chroma_search::Choice choice;
      // AV2ENC_RD_UVFULL needs the CfL prediction even when the helper's own
      // U-only SSE test would prefer plain DC: the full search ranks every
      // candidate on one perceptual-RD scale, so the CfL hypothesis is forced
      // into the returned `choice` by an unreachable DC score.
      const std::int64_t choose_dc_sse =
          uv_full ? std::numeric_limits<std::int64_t>::max() / 4 : dc_sse_u;
      if (!chroma_search::choose(
              u_plane, v_plane, recon_[0].samples.data(), recon_[0].stride,
              mi_cols_ * 4 - 1, mi_rows_ * 4 - 1, (cy0 * 2 / 64) * 64, cx0,
              cy0, cw, ch, mi_cols_ * 2, mi_rows_ * 2,
              chroma_avail.num4_above_right, chroma_avail.num4_below_left,
              intra_u_pred, intra_v_pred, choose_dc_sse, chroma_options,
              choice, error)) {
        return false;
      }
      intra_is_cfl = choice.is_cfl;
      intra_cfl_index = choice.cfl_index;
      intra_cfl_mh_dir = choice.cfl_mh_dir;
      intra_u_pred = choice.u;
      intra_v_pred = choice.v;
      if (uv_full) {
        // AV2ENC_RD_UVFULL: the 13 §5.20.5.6 base modes plus CfL, ranked by the
        // eff-perc perceptual D and the §8.2 symbol information bits
        // (src/pipeline/video/video_uv_search.h).
        uv_search::Params up;
        up.x = cx0;
        up.y = cy0;
        up.w = cw;
        up.h = ch;
        // §5.20.5.6 CTX_UV: the derived AngleDeltaUV and the uv_mode context
        // read the *coded* YMode, which is DC on a CHROMA_PART leaf.
        up.y_mode = chroma_part ? 0 : static_cast<int>(intra_y_mode);
        up.y_angle_delta = chroma_part ? 0
                                       : static_cast<int>(intra_y_angle_delta);
        up.avail = chroma_avail;
        chroma_neighbor_smooth(cy0 / 4, cx0 / 4, up.above_smooth,
                               up.left_smooth);
        up.enable_edge_filter = true;  // §5.4.8 always signals 1 here
        up.recon[0] = recon_[1].samples.data();
        up.recon_stride[0] = recon_[1].stride;
        up.recon[1] = recon_[2].samples.data();
        up.recon_stride[1] = recon_[2].stride;
        up.src[0] = uv_search::PlaneSrc{
            source_.u.data() +
                static_cast<std::size_t>(tile_cy) *
                    static_cast<std::size_t>(frame_chroma_width) +
                static_cast<std::size_t>(tile_cx),
            frame_chroma_width, frame_chroma_width - 1 - tile_cx,
            frame_chroma_height - 1 - tile_cy};
        up.src[1] = uv_search::PlaneSrc{
            source_.v.data() +
                static_cast<std::size_t>(tile_cy) *
                    static_cast<std::size_t>(frame_chroma_width) +
                static_cast<std::size_t>(tile_cx),
            frame_chroma_width, frame_chroma_width - 1 - tile_cx,
            frame_chroma_height - 1 - tile_cy};
        up.csf_gain = eff_perc::csf_gain();
        up.csf2_gain = eff_perc::csf2_gain();
        up.chroma_weight = eff_perc::chroma_weight();
        up.mask_row = chroma_mask_row;
        // One lambda for the chroma mode decision: the leaf RD weight the
        // eff-perc pipeline uses for its D + lambda * R objective.
        // AV2ENC_RD_UVLAMBDA is the chroma mode-site calibration multiplier;
        // unset it is 1.0, so every M4 result keeps its bytes.
        up.lambda = leaf_rd_lambda(st.current_qindex, options_) *
                    (eff_perc::enabled() ? eff_perc::lambda_scale() : 1.0) *
                    uv_rd::lambda_multiplier();
        up.level_lambda = up.lambda * uv_rd::level_lambda_multiplier();
        up.top_n = rd_uvfull_top_n();
        if (uv_rd::enabled() && uv_rd::trellis_variants() > 0) {
          // The level stage can make a lower-ranked mode win, so re-score the
          // wider measured candidate set.
          up.top_n = std::max(up.top_n, uv_rd::level_top_n());
        }
        if (choice.is_cfl) {
          up.have_cfl = true;
          up.cfl_u = choice.u;
          up.cfl_v = choice.v;
          up.cfl_index = choice.cfl_index;
          up.cfl_mh_dir = choice.cfl_mh_dir;
        }
        const int uv_qctx = ::av2enc::still::qctx::slice_for_base_q_idx(
            static_cast<int>(options_.base_q_idx));
        const auto uv_tx = [&](int plane,
                               const std::vector<std::uint8_t>& pred,
                               TxPlan& plan) {
          return compute_tx(source_, plane + 1, acx0, acy0, cw, ch, pred,
                            st.plane_quant[plane + 1], plan, error,
                            qm_enabled_ ? &qm_state_ : nullptr,
                            /*allow_tcq=*/false, /*qctx_slice=*/0,
                            /*is_inter=*/false, chroma_mask_row);
        };
        const auto uv_bits = [&](int plane, const TxPlan& plan, double& bits) {
          bits = pipe_plan_coefficient_bits(plan, plane + 1,
                                            /*is_inter=*/false, uv_qctx);
          return true;
        };
        // AV2ENC_RD_UVLEVEL: chroma coefficient-level RD.  The trellis needs
        // the scalar quantiser (QM and split DC/AC are excluded) and the
        // per-plane prediction in a tightly packed block.  The candidate is
        // re-scored on the exact J inside the search, so only the winning
        // plan is emitted.
        const bool uv_level_ok =
            uv_rd::enabled() && uv_rd::trellis_variants() > 0 &&
            !options_.enable_cctx && !qm_enabled_ &&
            st.plane_quant[1].dc == st.plane_quant[1].ac &&
            st.plane_quant[2].dc == st.plane_quant[2].ac &&
            st.plane_quant[1].ac > 0 && st.plane_quant[2].ac > 0;
        int level_variants = 0;
        std::vector<std::uint8_t> level_src[2];
        uv_rd::LevelParams level_params;
        if (uv_level_ok) {
          (void)uv_rd::stats_enabled();
          level_variants = uv_rd::trellis_variants();
          level_params.w = cw;
          level_params.h = ch;
          level_params.qctx = uv_qctx;
          level_params.mask_row = chroma_mask_row;
          for (int plane = 0; plane < 2; ++plane) {
            level_src[plane].resize(static_cast<std::size_t>(cw) *
                                    static_cast<std::size_t>(ch));
            for (int y = 0; y < ch; ++y) {
              for (int x = 0; x < cw; ++x) {
                level_src[plane][static_cast<std::size_t>(y) *
                                     static_cast<std::size_t>(cw) +
                                 static_cast<std::size_t>(x)] =
                    static_cast<std::uint8_t>(
                        up.src[plane].sample(up.x + x, up.y + y));
              }
            }
          }
        }
        up.level_variants = level_variants;
        const auto uv_level = [&](int plane,
                                  const std::vector<std::uint8_t>& pred,
                                  const TxPlan& /*scalar*/, int variant,
                                  TxPlan& plan) {
          if (!uv_level_ok) return false;
          level_params.plane = plane + 1;
          level_params.dc = st.plane_quant[plane + 1].dc;
          level_params.ac = st.plane_quant[plane + 1].ac;
          std::string level_error;
          const bool ok = uv_rd::trellis_plan(
              level_params, level_src[plane].data(), pred.data(),
              uv_rd::trellis_scale() * uv_rd::variant_weight(variant), plan,
              level_error);
          ++uv_rd::counters().plans;
          if (ok) {
            ++uv_rd::counters().trellis_runs;
          } else {
            ++uv_rd::counters().trellis_failures;
          }
          return ok;
        };
        uv_search::Result uv_result;
        if (!uv_search::search_with_levels(up, uv_tx, uv_bits, uv_level,
                                           uv_result, error)) {
          return false;
        }
        if (level_variants > 0 && uv_result.have_plans) {
          intra_u = uv_result.u_plan;
          intra_v = uv_result.v_plan;
          intra_chroma_ready = true;
          ++uv_rd::counters().blocks;
          uv_rd::counters().flips +=
              static_cast<unsigned long long>(uv_result.level_wins);
        }
        intra_uv_mode = uv_result.mode;
        intra_is_cfl = uv_result.is_cfl;
        intra_cfl_index = uv_result.cfl_index;
        intra_cfl_mh_dir = uv_result.cfl_mh_dir;
        intra_u_pred = uv_result.u;
        intra_v_pred = uv_result.v;
        if (trace_) {
          std::cerr << "  uv-full mode=" << intra_uv_mode
                    << " cfl=" << (intra_is_cfl ? 1 : 0)
                    << " cand=" << uv_result.candidates
                    << " rescored=" << uv_result.tx_rescored
                    << " D=" << uv_result.distortion
                    << " R=" << uv_result.bits << "\n";
        }
      }
    }
    if (intra_chroma_ready) {
      // AV2ENC_RD_UVLEVEL already emitted the scored plans; re-running
      // compute_tx here would code a coefficient set the RD did not choose.
    } else if (options_.enable_cctx) {
      // §5.4.8 enable_cctx / §7.14.6: joint U/V search.  is_cctx_allowed()
      // is true for 4:2:0; the CLI rejects the other formats with CCTX.
      if (!compute_chroma_cctx(source_, acx0, acy0, cw, ch, intra_u_pred,
                               intra_v_pred, st.quant_step,
                               /*is_inter=*/false,
                               intra_u, intra_v, &intra_cctx_type, error,
                               chroma_mask_row)) {
        return false;
      }
    } else if (!compute_tx(source_, 1, acx0, acy0, cw, ch, intra_u_pred,
                           st.plane_quant[1], intra_u, error,
                           qm_enabled_ ? &qm_state_ : nullptr,
                           /*allow_tcq=*/false, /*qctx_slice=*/0,
                           /*is_inter=*/false, chroma_mask_row) ||
               !compute_tx(source_, 2, acx0, acy0, cw, ch, intra_v_pred,
                           st.plane_quant[2], intra_v, error,
                           qm_enabled_ ? &qm_state_ : nullptr,
                           /*allow_tcq=*/false, /*qctx_slice=*/0,
                           /*is_inter=*/false, chroma_mask_row)) {
      return false;
    }
    if (keyframe_ && uvfull_stats_enabled()) {
      // AV2ENC_RD_UVFULL_STATS: key-picture chroma split and mode histogram.
      UvFullStats& stats = uvfull_stats();
      ++stats.leaves;
      if (intra_is_cfl) {
        ++stats.cfl;
      } else if (intra_uv_mode >= 0 && intra_uv_mode < 13) {
        ++stats.modes[intra_uv_mode];
      }
      if (!options_.enable_cctx) {
        const int stats_qctx = ::av2enc::still::qctx::slice_for_base_q_idx(
            static_cast<int>(options_.base_q_idx));
        stats.u_bits +=
            pipe_plan_coefficient_bits(intra_u, 1, /*is_inter=*/false,
                                       stats_qctx);
        stats.v_bits +=
            pipe_plan_coefficient_bits(intra_v, 2, /*is_inter=*/false,
                                       stats_qctx);
      }
    }
    }  // owns_chroma
    intra_sse = chroma_part
                    ? 0
                    : sse_vis(0, ax0, ay0, w, h,
                                         intra_y.recon);
    if (trace_) {
      std::cerr << "  cand intra sse=" << intra_sse;
      if (options_.enable_cctx) {
        std::cerr << " cctx=" << intra_cctx_type;
      }
      std::cerr << "\n";
    }
    have_intra = true;
  }

  // §5.20.8 luma palette (--video-palette).  The palette codes eob=0 on all
  // three planes, so its prediction is compared with the SSE of the ordinary
  // intra *prediction*, not the residual-coded reconstruction.
  PaletteChoice palette_choice;
  bool use_palette = false;
  std::int64_t intra_pred_sse = intra_sse;
  if (keyframe_ && palette_enabled_ && have_intra) {
    intra_pred_sse =
        sse_vis(0, ax0, ay0, w, h, intra_y_pred);
    if (!choose_palette_leaf(r, c, static_cast<int>(node.size), w, h,
                             intra_pred_sse, palette_choice, error)) {
      return false;
    }
    use_palette = palette_choice.found;
    if (trace_) {
      std::cerr << "  cand palette applied=" << (use_palette ? 1 : 0)
                << " pred_sse=" << intra_pred_sse
                << " palette_sse=" << palette_choice.sse << "\n";
    }
  }

  // SEG_LVL_SKIP / SEG_LVL_GLOBALMV: the decoder does not read is_inter or
  // any mode/ref syntax and forces YMode = GLOBALMV with SkipSegFrame.  Keep
  // the first-reference global candidate whenever one exists.
  if (seg_forced_inter && global_cand_valid) best = global_cand;

  // Inter residual (feature 1).  The predictor comes from the winner of the
  // motion search; when enabled, a quantised residual is coded for Y/U/V and
  // the reconstructed block is pred + dequantised residual.  The decision is
  // an SSE + lambda * estimated-bits comparison against skip_flag = 1.
  // One fully evaluated inter candidate.  The proxy search ranks candidates by
  // prediction SSE + static mode syntax; this structure holds what the real
  // predict -> TX -> quantise -> reconstruct -> rate path produced for one of
  // them, including the measured D and R used by AV2ENC_EXACT_RD.
  struct InterLeafResult {
    InterCandidate cand;
    std::vector<std::uint8_t> y;
    std::vector<std::uint8_t> u;
    std::vector<std::uint8_t> v;
    bool residual = false;
    TxPlan res_y;
    TxPlan res_u;
    TxPlan res_v;
    int cctx = 0;
    bool bawp = false;
    inter::adv::BawpFit fit{};
    bool interintra = false;
    int ii_mode = 0;
    bool ii_wedge = false;
    int ii_index = 0;
    std::int64_t ii_sse = 0;
    std::int64_t luma_sse = 0;
    // Measured objective terms.  Only filled when AV2ENC_EXACT_RD is on.
    double distortion = 0.0;
    double bits = 0.0;
    // AV2ENC_PIPE_PERC objective terms of the emitted form.  `pipe_bits`
    // includes the exact syntax (is_inter/skip/mode/ref/DRL/MVD) and the
    // context-coded residual; `pipe_distortion` is the local weighted D.
    double pipe_distortion = 0.0;
    double pipe_bits = 0.0;
    bool pipe_modeled = false;
    // §5.20.7.6 switchable interp_filter (0..2) carried by this evaluation.
    int filter = 0;
    bool filter_coded = false;
  };

  // Real leaf path for one candidate.  This is the historical winner-only
  // block, parameterised by the candidate instead of `best`: predict, apply
  // BAWP / inter-intra, run the shared luma TX search and chroma transform /
  // quantise / reconstruct, and decide skip vs residual.  With the gate off it
  // runs exactly once for the proxy winner, so every historical stream keeps
  // its bytes.
  const auto evaluate_inter_leaf =
      [&](const InterCandidate& input, InterLeafResult& out) -> bool {
    out = InterLeafResult{};
    out.cand = input;
    out.luma_sse = input.sse;
    if (out.cand.compound) {
#if defined(AV2ENC_VIDEO_FRAMES)
      // Re-run the candidate: the compound scratch planes hold the last
      // evaluated candidate, not necessarily this one.
      const inter::mc::Mv mvs[2] = {out.cand.mv, out.cand.mv1};
      bool predicted = false;
      if (out.cand.use_optflow) {
        predicted =
            predict_compound_optflow(sink, mvs, ax0, ay0, w, h, error);
      } else if (out.cand.use_refinemv) {
        predicted =
            predict_compound_refinemv(sink, mvs, ax0, ay0, w, h, error);
      } else {
        multiref::CompoundModeParams wparams;
        wparams.cwp_weight = out.cand.cwp_weight;
        wparams.cwp_index = static_cast<int>(out.cand.cwp_index);
        wparams.use_imp_mask = out.cand.use_imp_mask;
        wparams.type = out.cand.compound_type == 0
                           ? inter::adv::CompoundType::kWedge
                           : (out.cand.compound_type == 1
                                  ? inter::adv::CompoundType::kDiffWtd
                                  : inter::adv::CompoundType::kAverage);
        if (out.cand.compound_mode == 0) {
          wparams.type = inter::adv::CompoundType::kAverage;
        }
        wparams.wedge_index = out.cand.wedge_index;
        wparams.wedge_sign = out.cand.wedge_sign;
        wparams.mask_type = out.cand.mask_type;
        predicted = compound_plain(
            mvs, static_cast<int>(node.size), ax0, ay0, w, h,
            sink.compound_y, sink.compound_u, sink.compound_v, error,
            &wparams, 0, out.cand.same_ref ? 0 : 1);
      }
      if (!predicted) {
        error = "compound prediction failed";
        return false;
      }
#endif
      out.y = sink.compound_y;
      out.u = sink.compound_u;
      out.v = sink.compound_v;
    } else if (out.cand.use_warp) {
#if defined(AV2ENC_VIDEO_GME)
      std::string warp_error;
      if (gm_plan_ == nullptr || out.cand.ref_index >= gm_plan_->num_refs ||
          !warp_predict_420(
              gm_plan_->params[static_cast<std::size_t>(out.cand.ref_index)],
              references_[static_cast<std::size_t>(out.cand.ref_index)], ax0,
              ay0, w, h, r, c, w4, h4, out.y, out.u, out.v, warp_error)) {
        error = "inter warp prediction failed: " + warp_error;
        return false;
      }
#else
      error = "inter warp prediction is not linked in this build";
      return false;
#endif
    } else if (!predict_all(out.cand.ref_index, out.cand.mv, ax0, ay0, w, h,
                            out.y, out.u, out.v)) {
      error = "inter prediction failed";
      return false;
    }
    out.luma_sse = sse_vis(0, ax0, ay0, w, h, out.y);
    // §5.20.7.6 / §7.13.3.25: evaluate the fitted BAWP predictor for an
    // eligible single-prediction block (>= 8x8, not GLOBALMV, no warp).
    // The decoder applies the same fit before the residual, so a selected
    // block carries use_bawp = 1 and the adapted prediction.
    //
    // A GLOBALMV winner may still take BAWP through a NEARMV/NEWMV carrier
    // at the same MV: §5.20.7.6 does not code use_bawp on GLOBALMV, so the
    // encoder re-expresses an identical prediction in a mode that can.  The
    // swap never changes the MV or reference.  Segment-forced GLOBALMV
    // leaves keep their forced mode.
    const bool bawp_carrier_ok =
        out.cand.mode != syntax::kInterGlobalMv ||
        (options_.bawp_carrier && !seg_forced_inter &&
         bawp_carrier.valid &&
         bawp_carrier.ref_index == out.cand.ref_index &&
         bawp_carrier.mv == out.cand.mv);
    if (options_.enable_bawp && options_.tile_cols == 0 && !out.cand.compound &&
        !out.cand.use_warp && bawp_carrier_ok && w >= 8 && h >= 8 &&
        mi_cols_ * 4 == static_cast<int>(source_.width) &&
        mi_rows_ * 4 == static_cast<int>(source_.height)) {
      std::vector<std::uint8_t> adapted = out.y;
      inter::adv::BawpFit fit{};
      const inter::mc::PlaneView& ref_y =
          references_[static_cast<std::size_t>(out.cand.ref_index)].plane[0];
      if (bawp_adapt_luma(recon_[0].samples.data(), recon_[0].stride,
                          ref_y.data, ref_y.stride, mi_cols_, mi_rows_, r,
                          c, x0, y0, w, h, out.cand.mv.y, out.cand.mv.x,
                          adapted, &fit)) {
        // Keep the fit even when BAWP is not selected: the fade gate uses it
        // as the per-leaf illumination-ramp evidence.
        out.fit = fit;
        const std::int64_t adapted_sse =
            sse_vis(0, ax0, ay0, w, h, adapted);
        if (adapted_sse < out.luma_sse) {
          if (out.cand.mode == syntax::kInterGlobalMv) {
            out.cand = bawp_carrier;
          }
          out.bawp = true;
          out.fit = fit;
          out.y = adapted;
          out.luma_sse = adapted_sse;
        }
      }
      if (trace_) {
        std::cerr << "  cand bawp alpha=" << out.fit.alpha
                  << " count=" << out.fit.count
                  << " carrier=" << static_cast<int>(out.cand.mode)
                  << " applied=" << (out.bawp ? 1 : 0) << "\n";
      }
    }
    // §5.20.7.14/.15 + §7.13.3.29/.30 single-reference inter-intra.  The
    // blended block is skip-only (no residual) in this gated surface, and it
    // is evaluated after BAWP so the §5.20.7.14 use_bawp early return can
    // never be violated.
    if (options_.enable_interintra && !out.cand.compound &&
        !out.cand.use_warp && !out.bawp &&
        out.cand.mode != syntax::kInterGlobalMv && w >= 8 && h >= 8 &&
        w <= 64 && h <= 64 && !options_.force_inter_residual) {
      std::vector<std::uint8_t> ii_y;
      std::vector<std::uint8_t> ii_u;
      std::vector<std::uint8_t> ii_v;
      if (!evaluate_inter_intra(static_cast<int>(node.size), x0, y0, w, h,
                                out.y, out.u, out.v, ii_y, ii_u, ii_v,
                                out.ii_mode, out.ii_wedge, out.ii_index,
                                out.ii_sse, sink.state, error)) {
        return false;
      }
      const std::int64_t plain_sse =
          sse_vis(0, ax0, ay0, w, h, out.y) +
          sse_vis(1, acx0, acy0, cw, ch, out.u) +
          sse_vis(2, acx0, acy0, cw, ch, out.v);
      if (out.ii_sse < plain_sse) {
        out.interintra = true;
        out.y = ii_y;
        out.u = ii_u;
        out.v = ii_v;
        out.luma_sse = sse_vis(0, ax0, ay0, w, h, out.y);
      }
      if (trace_) {
        std::cerr << "  cand interintra applied="
                  << (out.interintra ? 1 : 0)
                  << " mode=" << out.ii_mode
                  << " wedge=" << (out.ii_wedge ? 1 : 0)
                  << " index=" << out.ii_index
                  << " sse=" << out.ii_sse
                  << " plain=" << plain_sse << "\n";
      }
    }
    // A compound leaf is coded skip-only in the gated subset: the writer
    // forces skip_flag = 1 below and never codes the residual.  Running the
    // residual search here would replace the prediction with
    // pred + dequantised residual in the encoder recon while the decoder
    // reconstructs the plain compound prediction, so the recon drifts from
    // the bitstream.  Leave compound leaves prediction-only, as emitted.
    if (options_.enable_inter_residual && !out.cand.compound &&
        !out.interintra && !seg_skip_leaf) {
      if (ba_on) ++pipe_ba::counters().tx_calls;
      const double pred_total =
          sse_pipe(0, ax0, ay0, w, h, out.y) +
          sse_pipe(1, acx0, acy0, cw, ch, out.u) +
          sse_pipe(2, acx0, acy0, cw, ch, out.v);
      if (((options_.enable_tx_search || options_.enable_inter_ddt)
               ? !search_luma_tx(source_, ax0, ay0, w, h, out.y,
                                 st.current_qindex,
                                 txtsearch::Mode::kInter,
                                 (pipe_on
                                      ? pipe_lambda
                                      : tx_rd_lambda(options_,
                                                     st.current_qindex) *
                                            perceptual::tx_lambda_scale(
                                                perceptual_weights)),
                                 /*allow_idtx=*/false,
                                 /*intra_mapped_mode=*/0,
                                 options_.enable_tcq,
                                 ::av2enc::still::qctx::slice_for_base_q_idx(
                                     (pipe_on || options_.tx_context_rate)
                                         ? base_q_index()
                                         : st.current_qindex),
                                 options_.enable_inter_ddt,
                                 /*search_partition=*/
                                 options_.enable_tx_search,
                                 /*context_rate=*/
                                 pipe_on || options_.tx_context_rate,
                                 options_.trace, options_.enable_rdoq,
                                 ::av2enc::still::qctx::slice_for_base_q_idx(
                                     static_cast<int>(options_.base_q_idx)),
                                 rdoq_sse_scale_at(options_,
                                                   st.current_qindex),
                                 out.res_y, error, luma_mask_row,
                                 pipe_on ? pipe_leaf_weights.data() : nullptr,
                                 eff_slice,
                                 ba_on ? pipe_ba::config().low_band_gain
                                       : (eff_on ? eff_perc::csf_gain() : 0.0),
                                 ba_on ? 0.0
                                       : (eff_on ? eff_perc::csf2_gain() : 0.0))
               : !compute_tx(source_, 0, ax0, ay0, w, h, out.y,
                             st.plane_quant[0], out.res_y, error,
                             qm_enabled_ ? &qm_state_ : nullptr,
                             options_.enable_tcq,
                             ::av2enc::still::qctx::slice_for_base_q_idx(
                                 st.current_qindex),
                             /*is_inter=*/true, options_.enable_rdoq,
                             ::av2enc::still::qctx::slice_for_base_q_idx(
                                 static_cast<int>(options_.base_q_idx)),
                             rdoq_sse_scale_at(options_, st.current_qindex),
                             luma_mask_row)) ||
          (options_.enable_cctx
               ? !compute_chroma_cctx(source_, acx0, acy0, cw, ch, out.u,
                                      out.v, st.quant_step,
                                      /*is_inter=*/true,
                                      out.res_u, out.res_v, &out.cctx, error,
                                      chroma_mask_row)
               : (!compute_tx(source_, 1, acx0, acy0, cw, ch, out.u,
                              st.plane_quant[1], out.res_u, error,
                              qm_enabled_ ? &qm_state_ : nullptr,
                              /*allow_tcq=*/false, /*qctx_slice=*/0,
                              /*is_inter=*/true, /*allow_rdoq=*/false,
                              /*rdoq_qctx=*/0, 0.0, chroma_mask_row) ||
                  !compute_tx(source_, 2, acx0, acy0, cw, ch, out.v,
                              st.plane_quant[2], out.res_v, error,
                              qm_enabled_ ? &qm_state_ : nullptr,
                              /*allow_tcq=*/false, /*qctx_slice=*/0,
                              /*is_inter=*/true, /*allow_rdoq=*/false,
                              /*rdoq_qctx=*/0, 0.0, chroma_mask_row)))) {
        return false;
      }
      // AV2ENC_RD_UVLEVEL: coefficient-level RD on the inter chroma residual.
      // The scalar plan is one candidate; the trellis proposals are scored on
      // the same J = D_perc + lambda * exact_bits the inter leaf already uses
      // for its skip/residual decision.  The winner replaces out.res_u/v, so
      // the emitted levels are exactly the scored ones.
      const bool inter_level_ok =
          uv_rd::enabled() && uv_rd::inter_enabled() &&
          !options_.enable_cctx && !qm_enabled_ &&
          st.plane_quant[1].dc == st.plane_quant[1].ac &&
          st.plane_quant[2].dc == st.plane_quant[2].ac &&
          st.plane_quant[1].ac > 0 && st.plane_quant[2].ac > 0;
      if (inter_level_ok) {
        (void)uv_rd::stats_enabled();
        const double level_lambda =
            (pipe_on ? pipe_lambda
                     : (options_.enable_leaf_rd
                            ? leaf_rd_lambda(st.current_qindex, options_)
                            : options_.inter_rd_lambda)) *
            uv_rd::level_lambda_multiplier();
        uv_rd::LevelParams lp;
        lp.w = cw;
        lp.h = ch;
        lp.qctx = pipe_qctx;
        lp.is_inter = true;
        lp.mask_row = chroma_mask_row;
        for (int plane = 1; plane <= 2; ++plane) {
          TxPlan& scalar = plane == 1 ? out.res_u : out.res_v;
          const std::vector<std::uint8_t>& pred =
              plane == 1 ? out.u : out.v;
          std::vector<std::uint8_t> src(static_cast<std::size_t>(cw) *
                                        static_cast<std::size_t>(ch));
          for (int y = 0; y < ch; ++y) {
            for (int x = 0; x < cw; ++x) {
              src[static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(cw) +
                  static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(
                  source_sample(source_, plane, acx0 + x, acy0 + y));
            }
          }
          lp.plane = plane;
          lp.dc = st.plane_quant[plane].dc;
          lp.ac = st.plane_quant[plane].ac;
          TxPlan best = scalar;
          const double best_d =
              sse_pipe(plane, acx0, acy0, cw, ch, scalar.recon);
          const double scalar_r = coeff_bits(scalar, plane, true);
          double best_cost = best_d + level_lambda * scalar_r;
          const int tries = uv_rd::trellis_variants();
          for (int variant = 1; variant <= tries; ++variant) {
            TxPlan proposal;
            std::string level_error;
            const bool ok = uv_rd::trellis_plan(
                lp, src.data(), pred.data(),
                uv_rd::trellis_scale() * uv_rd::variant_weight(variant),
                proposal, level_error);
            ++uv_rd::counters().plans;
            if (!ok) {
              ++uv_rd::counters().trellis_failures;
              continue;
            }
            ++uv_rd::counters().trellis_runs;
            const double alt_d =
                sse_pipe(plane, acx0, acy0, cw, ch, proposal.recon);
            const double alt_bits = coeff_bits(proposal, plane, true);
            const double cost = alt_d + level_lambda * alt_bits;
            if (cost < best_cost) {
              best_cost = cost;
              best = std::move(proposal);
              ++uv_rd::counters().flips;
            }
          }
          scalar = std::move(best);
        }
        ++uv_rd::counters().blocks;
      }
      if (trace_ && options_.enable_cctx) {
        std::cerr << "  cand inter cctx=" << out.cctx << "\n";
      }
      // AV2ENC_RD_TAIL_TX (default off): the L3 exact-rate mechanism on the
      // tail's inter luma residual.  The screen above ranked its plans with
      // the RDOQ-calibrated §8.3 model and a per-leaf distortion; this stage
      // re-searches with the uncalibrated model and the coefficient trellis,
      // then scores both plans with the driver's assembled-block BA-shaped D
      // and the live §8.2.6 interval rate.  Everything is inside the gate.
      if (tail_tx::tail_armed(arming_ordinal()) && pipe_on &&
          !keyframe_ && !chroma_part && out.res_y.searched &&
          !out.cand.compound && !out.interintra && !seg_skip_leaf) {
        const tail_tx::Config& tcfg = tail_tx::config();
        tail_tx::Counters& tc = tail_tx::counters();
        ++tc.leaves;
        // One J for the plan: D is the driver's BA-shaped leaf distortion on
        // the assembled block (causal top/left neighbours included so the
        // block-edge term is the error the decoder will show), R is the live
        // interval rate through the §5.20.7.30 scan order.
        const int rx = x0 > 0 ? 1 : 0;
        const int ry = y0 > 0 ? 1 : 0;
        const int pw = w + rx;
        const int ph = h + ry;
        const double csf =
            ba_on ? pipe_ba::config().low_band_gain
                  : (eff_on ? eff_perc::csf_gain() : 0.0);
        const double csf2 =
            ba_on ? 0.0 : (eff_on ? eff_perc::csf2_gain() : 0.0);
        std::vector<std::uint8_t> padded(
            static_cast<std::size_t>(pw) * static_cast<std::size_t>(ph), 0);
        const auto score_plan = [&](const TxPlan& plan, double* d_out,
                                    double* r_out) {
          for (int yy = 0; yy < h; ++yy) {
            std::memcpy(
                &padded[static_cast<std::size_t>(yy + ry) *
                            static_cast<std::size_t>(pw) +
                        static_cast<std::size_t>(rx)],
                &plan.recon[static_cast<std::size_t>(yy) *
                            static_cast<std::size_t>(w)],
                static_cast<std::size_t>(w));
          }
          if (ry > 0) {
            for (int xx = 0; xx < w; ++xx) {
              padded[static_cast<std::size_t>(rx + xx)] =
                  recon_[0].at(x0 + xx, y0 - 1);
            }
          }
          if (rx > 0) {
            for (int yy = 0; yy < h; ++yy) {
              padded[static_cast<std::size_t>(yy + ry) *
                         static_cast<std::size_t>(pw)] =
                  recon_[0].at(x0 - 1, y0 + yy);
            }
          }
          if (tcfg.ba_d) {
            *d_out = seq_rd::leaf_distortion_gains(
                source_.y.data(), static_cast<int>(source_.width),
                static_cast<int>(source_.height),
                static_cast<int>(source_.width), ax0, ay0, w, h,
                luma_mask_row, padded.data(), pw, ph, rx, ry, csf, csf2,
                tcfg.hf, tcfg.edge);
          } else {
            *d_out = static_cast<double>(sse_against_source(
                source_, 0, ax0, ay0, w, h, plan.recon, luma_mask_row));
          }
          *r_out = tcfg.live_rate
                       ? plan_rate_bits(plan, 0, true,
                                        static_cast<int>(node.size))
                       : plan.rate_bits;
        };
        double screen_d = 0.0;
        double screen_r = 0.0;
        const double screen_j = [&] {
          score_plan(out.res_y, &screen_d, &screen_r);
          return screen_d + pipe_lambda * tcfg.lambda * screen_r;
        }();
        // Exact re-search: uncalibrated §8.3 candidate rate (model_rate_scale
        // 1.0) and the L3 coefficient trellis, same lambda as the leaf.
        const double level_scale =
            rdoq_sse_scale_at(options_, st.current_qindex) *
            (tcfg.level ? tcfg.level_scale : 1.0);
        TxPlan exact_y;
        std::string exact_error;
        const bool exact_ok = search_luma_tx(
            source_, ax0, ay0, w, h, out.y, st.current_qindex,
            txtsearch::Mode::kInter, pipe_lambda * tcfg.lambda,
            /*allow_idtx=*/false, /*intra_mapped_mode=*/0, options_.enable_tcq,
            ::av2enc::still::qctx::slice_for_base_q_idx(
                (pipe_on || options_.tx_context_rate) ? base_q_index()
                                                     : st.current_qindex),
            options_.enable_inter_ddt,
            /*search_partition=*/options_.enable_tx_search,
            /*context_rate=*/true, options_.trace,
            /*allow_rdoq=*/tcfg.level && options_.enable_rdoq,
            ::av2enc::still::qctx::slice_for_base_q_idx(
                static_cast<int>(options_.base_q_idx)),
            level_scale, exact_y, exact_error, luma_mask_row,
            pipe_leaf_weights.empty() ? nullptr : pipe_leaf_weights.data(),
            eff_slice, csf, csf2, /*intra_rdoq=*/false,
            /*model_rate_scale=*/1.0);
        if (exact_ok) {
          ++tc.exact_ok;
          double exact_d = 0.0;
          double exact_r = 0.0;
          score_plan(exact_y, &exact_d, &exact_r);
          const double exact_j = exact_d + pipe_lambda * tcfg.lambda * exact_r;
          tc.screen_d += screen_d;
          tc.screen_r += screen_r;
          tc.exact_d += exact_d;
          tc.exact_r += exact_r;
          if (exact_j < screen_j) {
            if (exact_y.partition != out.res_y.partition ||
                exact_y.parts.size() != out.res_y.parts.size()) {
              ++tc.plan_changed;
            }
            out.res_y = std::move(exact_y);
            ++tc.kept_exact;
            if (trace_) {
              std::cerr << "  tail-tx exact kept parts="
                        << out.res_y.parts.size() << " D=" << exact_d
                        << " R=" << exact_r << " screen D=" << screen_d
                        << " R=" << screen_r << "\n";
            }
          }
        } else {
          ++tc.exact_failed;
        }
      }
      const double res_total =
          sse_pipe(0, ax0, ay0, w, h, out.res_y.recon) +
          sse_pipe(1, acx0, acy0, cw, ch, out.res_u.recon) +
          sse_pipe(2, acx0, acy0, cw, ch, out.res_v.recon);
      const bool all_zero =
          luma_plan_all_zero(out.res_y) && out.res_u.eob == 0 &&
          out.res_v.eob == 0;
      // AV2ENC_PORT_HORIZON (default off): price the error the skip and the
      // residual arm install in the next K real source pictures, not only the
      // current one.  Same decaying aggregation as the lane-2 driver
      // (w_k = 0.5^k); the prediction IS the skipped reconstruction, so the
      // two arms are measured against the same future frames.  Zero when the
      // gate is off, which leaves every decision below untouched.
      double horizon_skip = 0.0;
      double horizon_res = 0.0;
      if (horizon_port::enabled() && !horizon_future_.empty()) {
        const int horizon = horizon_port::horizon_k();
        horizon_skip = horizon_port::future_distortion(
            horizon_future_, horizon, ax0, ay0, w, h, out.y.data(),
            out.u.data(), out.v.data());
        horizon_res = horizon_port::future_distortion(
            horizon_future_, horizon, ax0, ay0, w, h, out.res_y.recon.data(),
            out.res_u.recon.data(), out.res_v.recon.data());
      }
      // AV2ENC_P_DRIFT (default off): head-only drift-aware skip/reference
      // cost.  A skipped leaf installs its prediction error in the reference,
      // where the following pictures copy it through fractional-pel MC.  The
      // term prices the error each arm installs in the next K real source
      // pictures, in the same eff-perc D units as the current-frame term (the
      // merged horizon port added a plain SSE and lost on this recipe,
      // docs/port-lane2.md).  Head pictures only; the drift-refresh schedule
      // code and thresholds are untouched.
      double pdrift_skip = 0.0;
      double pdrift_res = 0.0;
      if (p_drift::armed(keyframe_, tx_detail::picture_ordinal()) &&
          !horizon_future_.empty()) {
        const p_drift::Config& pcfg = p_drift::config();
        std::vector<double> drift_scratch;
        const double drift_csf1 = eff_on ? eff_perc::csf_gain() : 0.0;
        const double drift_csf2 = eff_on ? eff_perc::csf2_gain() : 0.0;
        const double drift_chroma = eff_perc::chroma_weight();
        pdrift_skip = pcfg.alpha * p_drift::future_distortion_perc(
                                     horizon_future_, pcfg.k, ax0, ay0, w, h,
                                     out.y.data(), out.u.data(), out.v.data(),
                                     cw, ch, luma_mask_row, chroma_mask_row,
                                     drift_csf1, drift_csf2, drift_chroma,
                                     pipe_leaf_weights.empty()
                                         ? nullptr
                                         : pipe_leaf_weights.data(),
                                     &drift_scratch);
        pdrift_res = pcfg.alpha * p_drift::future_distortion_perc(
                                    horizon_future_, pcfg.k, ax0, ay0, w, h,
                                    out.res_y.recon.data(),
                                    out.res_u.recon.data(),
                                    out.res_v.recon.data(), cw, ch,
                                    luma_mask_row, chroma_mask_row, drift_csf1,
                                    drift_csf2, drift_chroma,
                                    pipe_leaf_weights.empty()
                                        ? nullptr
                                        : pipe_leaf_weights.data(),
                                    &drift_scratch);
        ++p_drift::counters().leaves;
        p_drift::counters().future_skip += pdrift_skip;
        p_drift::counters().future_res += pdrift_res;
        p_drift::counters().now_skip += pred_total;
        p_drift::counters().now_res += res_total;
      }
      const double skip_distortion = pred_total + horizon_skip + pdrift_skip;
      const double res_distortion = res_total + horizon_res + pdrift_res;
      // Reconstruction step is qlookup/8 (66 at QP 90); sse_scale 0.5 is
      // the M1 calibrated value.  --video-leaf-rd swaps in the same
      // QP-derived lambda the intra/inter decision uses.
      const double kLambda =
          options_.enable_leaf_rd
              ? leaf_rd_lambda(st.current_qindex, options_)
              : options_.inter_rd_lambda;
      // Historical bar: 3.0 bits of residual syntax vs a 2.0-bit skip.
      const double hist_res_bits =
          3.0 + luma_plan_bits(out.res_y, true) +
          estimate_tx_bits(out.res_u, false) +
          estimate_tx_bits(out.res_v, false);
      // AV2ENC_SKIP_DRIFT: a skip leaf does not just take this frame's
      // prediction error, it installs that error in the reference and the
      // following frames copy it through subpel MC.  Charge the frozen error
      // in the skip arm of every comparison below.  Off by default; the
      // zero-MV skip is an exact copy, so its factor can be held at 1.
      const bool zero_mv_skip = out.cand.mv.y == 0 && out.cand.mv.x == 0;
      double drift_mv_factor =
          zero_mv_skip ? skip_drift::zero_mv_factor()
                       : skip_drift::nz_mv_factor();
      if (drift_mv_factor == 0.0 && zero_mv_skip && skip_drift::fade_enabled()) {
        // Illumination-ramp evidence: a zero-MV skip on a fade freezes a
        // signed DC bias in the reference.  Static content and sensor noise
        // have a near-zero leaf mean error, a fade does not.
        std::int64_t signed_sum = 0;
        for (int yy = 0; yy < h; ++yy) {
          for (int xx = 0; xx < w; ++xx) {
            const int s = source_sample(source_, 0, ax0 + xx, ay0 + yy);
            signed_sum += s - static_cast<int>(
                                  out.y[static_cast<std::size_t>(yy) *
                                            static_cast<std::size_t>(w) +
                                        static_cast<std::size_t>(xx)]);
          }
        }
        const double bias =
            static_cast<double>(signed_sum) / static_cast<double>(w * h);
        if (std::abs(bias) >= skip_drift::fade_threshold()) {
          drift_mv_factor = 1.0;
        }
      }
      if (drift_mv_factor == 0.0 && zero_mv_skip && illum_active_) {
        // AV2ENC_SKIP_DRIFT_ILLUM: the whole frame is moving under an
        // illumination ramp, so a zero-MV skip freezes the previous frame's
        // illumination even where the leaf's own mean error is below the
        // per-leaf bias bar.
        drift_mv_factor = 1.0;
      }
      if (drift_mv_factor == 0.0 && zero_mv_skip && ref_deficit_active_) {
        // AV2ENC_SKIP_DRIFT_REFDEF: the primary reference has already lost
        // more than the threshold against the key anchor, so the zero-MV skip
        // would freeze chain-added error, not the key's priced error.
        drift_mv_factor = 1.0;
      }
      const double drift_factor =
          skip_drift::enabled() ? skip_drift::alpha() * drift_mv_factor : 0.0;
      const double skip_drift_cost = drift_factor * pred_total;
      const bool hist_want =
          options_.force_inter_residual ||
          (!all_zero &&
           res_distortion + kLambda * hist_res_bits <
               skip_distortion + skip_drift_cost + kLambda * 2.0);
      // R09 bar: skip_flag = 0 pays the coefficient rate, skip_flag = 1 pays
      // the exact -log2 P(skip=1); the difference replaces the fixed 2.0/3.0
      // pair.  Both candidates pay the same is_inter/mode syntax, so it
      // cancels at this comparison.
      double res_syntax = 3.0;
      double skip_syntax = 2.0;
      if (rate_model_on) {
        double delta = 0.0;
        if (rate_probe_.skip_delta(probe_node_for(out.cand), &delta)) {
          res_syntax = 0.0;
          skip_syntax = delta;
        }
      }
      const double res_bits = res_syntax +
                              decision_coeff_bits(out.res_y, true) +
                              decision_coeff_bits(out.res_u, false) +
                              decision_coeff_bits(out.res_v, false);
      bool want =
          options_.force_inter_residual ||
          (!all_zero &&
           res_distortion + kLambda * res_bits <
               skip_distortion + skip_drift_cost + kLambda * skip_syntax);
      if (pipe_on) {
        // Joint skip/residual choice with the pipeline objective.  Both forms
        // pay the exact is_inter/skip/mode syntax; only the residual form
        // adds the context-coded coefficients.  This happens inside every
        // re-scored candidate, not after the candidate has been chosen.
        syntax::BlockSyntax probe = probe_node_for(out.cand);
        probe.use_bawp = out.bawp ? 1 : 0;
        probe.motion_mode = out.interintra ? 1 : 0;
        probe.skip_flag = 1;
        rate_hybrid::Terms skip_terms;
        const bool skip_ok = pipe_probe_.price_full(probe, &skip_terms);
        probe.skip_flag = 0;
        rate_hybrid::Terms res_terms;
        const bool res_ok = pipe_probe_.price_full(probe, &res_terms);
        if (skip_ok && res_ok) {
          const double full_res_bits =
              res_terms.total() +
              coeff_bits(out.res_y, 0, true) +
              coeff_bits(out.res_u, 1, true) +
              coeff_bits(out.res_v, 2, true);
          const double full_skip_bits = skip_terms.total();
          const double skip_cost =
              skip_distortion + skip_drift_cost + pipe_lambda * full_skip_bits;
          const double exact_res_cost =
              res_distortion + pipe_lambda * full_res_bits;
          want = options_.force_inter_residual ||
                 (!all_zero && exact_res_cost < skip_cost);
          if (want != hist_want) {
            ++(eff_on ? eff_perc::counters().skip_flips
                      : pipe_perc::counters().skip_flips);
            if (p_drift::armed(keyframe_, tx_detail::picture_ordinal())) {
              if (want) {
                ++p_drift::counters().skip_to_residual;
              } else {
                ++p_drift::counters().residual_to_skip;
              }
            }
          }
          out.pipe_modeled = true;
          out.pipe_distortion = want ? res_total : pred_total;
          out.pipe_bits = want ? full_res_bits : full_skip_bits;
        } else {
          ++(eff_on ? eff_perc::counters().fallbacks
                : pipe_perc::counters().fallbacks);
        }
      }
      if (rate_hybrid::enabled() && rate_reporting && want != hist_want) {
        rate_hybrid::report::count_decision("skip");
      }
      if (trace_) {
        std::cerr << "  ileaf " << w << "x" << h << " at " << x0 << ","
                  << y0 << " mv=" << out.cand.mv.y << "," << out.cand.mv.x
                  << " ref=" << static_cast<int>(out.cand.ref_index)
                  << " pred=" << pred_total << " res=" << res_total
                  << " rbits=" << res_bits
                  << " parts=" << out.res_y.parts.size()
                  << " eob=" << out.res_y.eob
                  << " zero=" << (all_zero ? 1 : 0)
                  << " want=" << (want ? 1 : 0);
        for (const TxPlan::Part& part : out.res_y.parts) {
          std::cerr << " [" << part.coded_w << "x" << part.coded_h << "/e"
                    << part.eob << "]";
        }
        std::cerr << "\n";
      }
      if (want) {
        out.residual = true;
        out.y = out.res_y.recon;
        out.u = out.res_u.recon;
        out.v = out.res_v.recon;
        out.luma_sse = sse_vis(0, ax0, ay0, w, h, out.y);
        if (trace_ && options_.enable_inter_ddt) {
          for (const TxPlan::Part& part : out.res_y.parts) {
            const auto part_params = txv::make_tx_params(
                static_cast<txv::TxSize>(part.tx_size),
                static_cast<txv::TxType>(part.tx_type));
            av2enc::video::ddt::Context ddt_ctx;
            ddt_ctx.enable_inter_ddt = true;
            ddt_ctx.is_inter = true;
            if (part_params.has_value() &&
                av2enc::video::ddt::plan_has_ddt(*part_params, ddt_ctx)) {
              std::cerr << "m2 inter ddt residual r=" << r << " c=" << c
                        << " size=" << part.coded_w << "x" << part.coded_h
                        << " type=" << part.tx_type << "\n";
            }
          }
        }
      }
    }
    if (trace_) {
      std::cerr << "  cand inter final sse=" << out.luma_sse
                << " residual=" << (out.residual ? 1 : 0) << "\n";
    }
    // Measured objective terms for AV2ENC_EXACT_RD.  D is the reconstruction
    // distortion of the block the leaf would actually emit (all owned
    // planes); R is the same syntax + coefficient estimate the leaf decision
    // charges, with the mode the candidate ended up carrying after the BAWP
    // carrier swap and the skip-only inter-intra / compound forms.  Gate off
    // does not compute this, so the historical path is untouched.
    if (exact_rd::enabled()) {
      out.distortion = 0.0;
      if (!chroma_part) {
        out.distortion +=
            static_cast<double>(sse_vis(0, ax0, ay0, w, h, out.y));
      }
      if (owns_chroma) {
        out.distortion +=
            static_cast<double>(sse_vis(1, acx0, acy0, cw, ch, out.u));
        out.distortion +=
            static_cast<double>(sse_vis(2, acx0, acy0, cw, ch, out.v));
      }
      out.bits = 2.0 + candidate_syntax_bits(out.cand);
      if (rate_model_on) {
        syntax::BlockSyntax probe = probe_node_for(out.cand);
        probe.skip_flag = out.residual ? 0 : 1;
        probe.use_bawp = out.bawp ? 1 : 0;
        probe.motion_mode = out.interintra ? 1 : 0;
        rate_hybrid::Terms terms;
        if (rate_probe_.price_full(probe, &terms)) out.bits = terms.total();
      }
      if (out.bawp && !rate_model_on) out.bits += 1.0;          // use_bawp
      if (out.interintra && !rate_model_on) out.bits += 2.0;    // motion_mode
      if (out.residual) {
        out.bits += decision_coeff_bits(out.res_y, true) +
                    decision_coeff_bits(out.res_u, false) +
                    decision_coeff_bits(out.res_v, false);
      }
    }
    return true;
  };

  // §5.20.7.6 switchable interp_filter selection.  One candidate is evaluated
  // under each of the three TileInterpFilterCdf symbols and the perceptual RD
  // rule picks the winner:
  //   D = plain SSE + w * (perceptual SSE - plain SSE)
  //   J = D + lambda * (syntax/coefficients + filter symbol)
  // `w` is AV2ENC_INTERP_BLOCK_CSF; w = 1 is exactly the eff-perc D the leaf
  // and TX search use, and larger values price low/mid-band error harder.
  // Gate off: one evaluation, filter 0, byte-identical to the historical path.
  constexpr double kInterpFilterSymbolBits = 1.585;  // -log2(1/3)
  const auto evaluate_inter_leaf_filtered =
      [&](const InterCandidate& input, InterLeafResult& out) -> bool {
    out = InterLeafResult{};
    const auto filter_y_mode = [](int mode) {
      return mode == syntax::kInterGlobalMv
                 ? inter::mvprec::kYModeGlobalMv
                 : (mode == syntax::kInterNearMv
                        ? inter::mvprec::kYModeNearMv
                        : inter::mvprec::kYModeNewMv);
    };
    const bool eligible =
        interp_block_enabled() && !input.compound && !input.use_warp &&
        inter::mvprec::needs_interp_filter(w >= 8 && h >= 8,
                                           filter_y_mode(input.mode),
                                           /*motion_mode=*/0);
    if (!eligible) {
      if (!evaluate_inter_leaf(input, out)) return false;
      out.filter = 0;
      return true;
    }
    const double lambda =
        pipe_on ? pipe_lambda
                : (exact_on ? leaf_rd_lambda(st.current_qindex, options_)
                            : inter_lambda);
    const double csf_weight = interp_block_csf_weight();
    const auto filter_cost = [&](const InterLeafResult& result) {
      double plain = 0.0;
      if (!chroma_part) {
        plain += static_cast<double>(sse_vis(0, ax0, ay0, w, h, result.y));
      }
      if (owns_chroma) {
        plain += static_cast<double>(sse_vis(1, acx0, acy0, cw, ch, result.u));
        plain += static_cast<double>(sse_vis(2, acx0, acy0, cw, ch, result.v));
      }
      double perceptual = plain;
      if (pipe_on) {
        perceptual = sse_pipe(0, ax0, ay0, w, h, result.y) +
                     sse_pipe(1, acx0, acy0, cw, ch, result.u) +
                     sse_pipe(2, acx0, acy0, cw, ch, result.v);
      }
      const double distortion = plain + csf_weight * (perceptual - plain);
      double bits = 0.0;
      if (pipe_on && result.pipe_modeled) {
        bits = result.pipe_bits;
      } else if (exact_on) {
        bits = result.bits;
      } else {
        bits = inter_syntax_bits(result.cand.mode, result.cand.drl_index,
                                 result.cand.mvd.y, result.cand.mvd.x,
                                 result.cand.use_amvd) +
               (result.bawp ? 1.0 : 0.0) +
               (result.interintra ? 2.0 : 0.0) +
               (result.residual
                    ? (luma_plan_bits(result.res_y, true) +
                       estimate_tx_bits(result.res_u, false) +
                       estimate_tx_bits(result.res_v, false))
                    : 2.0);
      }
      return distortion + lambda * (bits + kInterpFilterSymbolBits);
    };
    const int saved_override = interp_block_filter_override();
    // AV2ENC_INTERP_BLOCK_SKIP: EIGHTTAP's skip/residual decision is the
    // historical one.  Evaluate it first so the guard can preserve that
    // decision; the non-zero filters may then refine a leaf that already
    // codes residual, but they cannot introduce a filtered skip.
    const int skip_guard = interp_block_skip_guard();
    interp_block_filter_override() = 0;
    InterLeafResult chosen;
    if (!evaluate_inter_leaf(input, chosen)) {
      interp_block_filter_override() = saved_override;
      return false;
    }
    chosen.filter = 0;
    double chosen_cost = filter_cost(chosen);
    const bool base_residual = chosen.residual;
    const bool base_zero_mv = input.mv.y == 0 && input.mv.x == 0;
    const bool guard_this_leaf =
        skip_guard == 1 || (skip_guard == 2 && !base_zero_mv);
    if (!(guard_this_leaf && !base_residual)) {
      for (int filter = 1; filter <= 2; ++filter) {
        interp_block_filter_override() = filter;
        InterLeafResult result;
        if (!evaluate_inter_leaf(input, result)) {
          interp_block_filter_override() = saved_override;
          return false;
        }
        const double cost = filter_cost(result);
        if (cost < chosen_cost) {
          chosen = std::move(result);
          chosen.filter = filter;
          chosen_cost = cost;
        }
      }
      if (guard_this_leaf && base_residual && !chosen.residual) {
        // A non-zero filter would have turned EIGHTTAP's residual leaf into
        // a skip.  Re-evaluate EIGHTTAP and keep its uncorrected path; the
        // filter must not reshape the reference decision.
        interp_block_filter_override() = 0;
        InterLeafResult fallback;
        if (!evaluate_inter_leaf(input, fallback)) {
          interp_block_filter_override() = saved_override;
          return false;
        }
        fallback.filter = 0;
        chosen = std::move(fallback);
      }
    }
    interp_block_filter_override() = saved_override;
    out = std::move(chosen);
    out.filter_coded = true;
    out.pipe_bits += kInterpFilterSymbolBits;
    out.bits += kInterpFilterSymbolBits;
    return true;
  };

  // The gate-off path evaluates exactly the proxy winner.  With the gate on,
  // the cheap ranking is re-scored: the proxy winner first (preserving the
  // historical strict-< tie-break), then the next cheapest candidates in the
  // same order.  N=all would therefore visit a deterministic total order.
  std::vector<const InterCandidate*> exact_eval_order;
  if ((exact_on || pipe_on) && !ranked_candidates.empty()) {
    std::stable_sort(
        ranked_candidates.begin(), ranked_candidates.end(),
        [](const RankedCandidate& a, const RankedCandidate& b) {
          if (a.cost != b.cost) return a.cost < b.cost;
          return a.order < b.order;
        });
    const auto same_candidate = [](const InterCandidate& a,
                                   const InterCandidate& b) {
      return a.mode == b.mode && a.ref_index == b.ref_index &&
             a.mv.y == b.mv.y && a.mv.x == b.mv.x &&
             a.compound == b.compound && a.same_ref == b.same_ref &&
             a.compound_mode == b.compound_mode && a.compound_joint == b.compound_joint && a.drl_index == b.drl_index &&
             a.mv1.y == b.mv1.y && a.mv1.x == b.mv1.x &&
             a.use_warp == b.use_warp;
    };
    if (best.valid) exact_eval_order.push_back(&best);
    const std::size_t top_n = static_cast<std::size_t>(
        eff_on ? eff_perc::top_n()
               : (pipe_on ? pipe_perc::top_n() : exact_rd::top_n()));
    for (const RankedCandidate& ranked : ranked_candidates) {
      if (exact_eval_order.size() >= top_n) break;
      bool duplicate = false;
      for (const InterCandidate* already : exact_eval_order) {
        if (same_candidate(*already, ranked.cand)) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) exact_eval_order.push_back(&ranked.cand);
    }
  } else if (best.valid) {
    exact_eval_order.push_back(&best);
  }

  std::vector<std::uint8_t> inter_y;
  std::vector<std::uint8_t> inter_u;
  std::vector<std::uint8_t> inter_v;
  int inter_filter = 0;
  bool inter_residual = false;
  TxPlan res_y;
  TxPlan res_u;
  TxPlan res_v;
  int inter_cctx_type = 0;
  bool bawp_applied = false;
  inter::adv::BawpFit bawp_fit{};
  bool inter_intra_applied = false;
  int inter_intra_mode = 0;
  bool inter_intra_wedge = false;
  int inter_intra_index = 0;
  std::int64_t inter_luma_sse = best.sse;
  // Exact objective of the selected inter candidate.  Only read when the gate
  // is on (the intra/inter comparison below).
  double exact_inter_distortion = 0.0;
  double exact_inter_bits = 0.0;
  // Pipeline objective of the selected inter candidate (D and R are the
  // emitted form, so the intra/inter comparison below uses one J).
  double pipe_inter_distortion = 0.0;
  double pipe_inter_bits = 0.0;
  bool pipe_inter_valid = false;
  // R09 report: the model's candidate ranking vs the historical constants.
  // Both rankings are over the same candidate set; a changed winner is one of
  // the decisions the model moves.
  if (rate_model_on && rate_reporting && best.valid && best_hist.valid) {
    const auto same_candidate = [](const InterCandidate& a,
                                   const InterCandidate& b) {
      return a.mode == b.mode && a.ref_index == b.ref_index &&
             a.mv.y == b.mv.y && a.mv.x == b.mv.x &&
             a.compound == b.compound && a.same_ref == b.same_ref &&
             a.compound_mode == b.compound_mode && a.compound_joint == b.compound_joint &&
             a.drl_index == b.drl_index && a.mv1.y == b.mv1.y &&
             a.mv1.x == b.mv1.x && a.use_warp == b.use_warp;
    };
    if (!same_candidate(best, best_hist)) {
      rate_hybrid::report::count_decision("candidate");
    }
  }
  if (!keyframe_ && best.valid && !exact_eval_order.empty()) {
    InterLeafResult chosen;
    double chosen_cost = 0.0;
    bool have_chosen = false;
    // The proxy winner after its own evaluation (it may have swapped to the
    // BAWP carrier).  This is what the historical path would have emitted, so
    // the "exact differs from proxy" counter compares against it rather than
    // against the pre-evaluation candidate.
    InterCandidate proxy_after_eval;
    const double exact_lambda =
        leaf_rd_lambda(st.current_qindex, options_);
    for (const InterCandidate* candidate : exact_eval_order) {
      InterLeafResult result;
      if (!evaluate_inter_leaf_filtered(*candidate, result)) return false;
      if (pipe_on) {
        if (!result.pipe_modeled) {
          // The probe cannot model this carrier (compound / TIP / AMVD /
          // flex precision).  Price the emitted form with the local D and the
          // historical syntax estimate, and count the fallback.
          result.pipe_distortion =
              sse_pipe(0, ax0, ay0, w, h, result.y) +
              sse_pipe(1, acx0, acy0, cw, ch, result.u) +
              sse_pipe(2, acx0, acy0, cw, ch, result.v);
          result.pipe_bits =
              inter_syntax_bits(result.cand.mode, result.cand.drl_index,
                                result.cand.mvd.y, result.cand.mvd.x,
                                result.cand.use_amvd) +
              (result.bawp ? 1.0 : 0.0) +
              (result.interintra ? 2.0 : 0.0) +
              (result.residual
                   ? (luma_plan_bits(result.res_y, true) +
                      estimate_tx_bits(result.res_u, false) +
                      estimate_tx_bits(result.res_v, false))
                   : 2.0) +
              (result.filter_coded ? kInterpFilterSymbolBits : 0.0) +
              compound_syntax_bits(result.cand);
        }
        const double cost =
            result.pipe_distortion + pipe_lambda * result.pipe_bits;
        ++(eff_on ? eff_perc::counters().rescored
                  : pipe_perc::counters().rescored);
        if (!have_chosen) {
          proxy_after_eval = result.cand;
        }
        if (!have_chosen || cost < chosen_cost) {
          chosen = std::move(result);
          chosen_cost = cost;
          have_chosen = true;
        }
      } else if (exact_on) {
        const double cost =
            result.distortion + exact_lambda * result.bits;
        ++exact_rd::counters().rescored;
        if (!have_chosen) {
          proxy_after_eval = result.cand;
          if (result.residual) ++exact_rd::counters().proxy_residual;
        }
        if (!have_chosen || cost < chosen_cost) {
          chosen = std::move(result);
          chosen_cost = cost;
          have_chosen = true;
        }
      } else {
        chosen = std::move(result);
        chosen_cost = 0.0;
        have_chosen = true;
      }
      if (!exact_on && !pipe_on) break;
    }
    if (!have_chosen) {
      error = "exact RD produced no inter candidate";
      return false;
    }
    if (exact_on) {
      ++exact_rd::counters().leaves;
      const auto same_candidate = [](const InterCandidate& a,
                                     const InterCandidate& b) {
        return a.mode == b.mode && a.ref_index == b.ref_index &&
               a.mv.y == b.mv.y && a.mv.x == b.mv.x &&
               a.compound == b.compound && a.same_ref == b.same_ref &&
               a.compound_mode == b.compound_mode && a.compound_joint == b.compound_joint &&
               a.drl_index == b.drl_index && a.mv1.y == b.mv1.y &&
               a.mv1.x == b.mv1.x && a.use_warp == b.use_warp;
      };
      if (!same_candidate(chosen.cand, proxy_after_eval)) {
        ++exact_rd::counters().inter_winner_changed;
      }
      if (chosen.residual) ++exact_rd::counters().exact_residual;
      exact_inter_distortion = chosen.distortion;
      exact_inter_bits = chosen.bits;
    }
    if (pipe_on) {
      ++(eff_on ? eff_perc::counters().leaves
                : pipe_perc::counters().leaves);
      const auto same_candidate = [](const InterCandidate& a,
                                     const InterCandidate& b) {
        return a.mode == b.mode && a.ref_index == b.ref_index &&
               a.mv.y == b.mv.y && a.mv.x == b.mv.x &&
               a.compound == b.compound && a.same_ref == b.same_ref &&
               a.compound_mode == b.compound_mode && a.compound_joint == b.compound_joint &&
               a.drl_index == b.drl_index && a.mv1.y == b.mv1.y &&
               a.mv1.x == b.mv1.x && a.use_warp == b.use_warp;
      };
      if (!same_candidate(chosen.cand, proxy_after_eval)) {
        ++(eff_on ? eff_perc::counters().inter_winner_changed
                  : pipe_perc::counters().inter_winner_changed);
      }
      if (chosen.residual) {
        ++(eff_on ? eff_perc::counters().residual_leaves
                  : pipe_perc::counters().residual_leaves);
      }
      pipe_inter_distortion = chosen.pipe_distortion;
      pipe_inter_bits = chosen.pipe_bits;
      pipe_inter_valid = true;
    }
    best = chosen.cand;
    inter_y = std::move(chosen.y);
    inter_u = std::move(chosen.u);
    inter_v = std::move(chosen.v);
    inter_filter = chosen.filter;
    inter_residual = chosen.residual;
    res_y = std::move(chosen.res_y);
    res_u = std::move(chosen.res_u);
    res_v = std::move(chosen.res_v);
    inter_cctx_type = chosen.cctx;
    bawp_applied = chosen.bawp;
    bawp_fit = chosen.fit;
    inter_intra_applied = chosen.interintra;
    inter_intra_mode = chosen.ii_mode;
    inter_intra_wedge = chosen.ii_wedge;
    inter_intra_index = chosen.ii_index;
    inter_luma_sse = chosen.luma_sse;
  }

  bool use_intra = keyframe_ || options_.force_intra || !best.valid;
  if (seg_forced_inter && best.valid && !keyframe_) use_intra = false;
  // §5.20.5.3/.4 IBC + §7.13.3.26 morph on the intra keyframe.  The copy is
  // compared with the ordinary intra leaf (or a winning palette leaf) by
  // luma SSE; a win selects the IBC skip block (no residual) and, when it
  // helps, morph_pred = 1.
  IbcChoice ibc_choice;
  bool use_ibc = false;
  if (keyframe_ && ibc_enabled_ && have_intra && !shared_chroma &&
      w >= 8 && h >= 8) {
    const std::int64_t intra_distortion =
        use_palette ? palette_choice.sse : intra_sse;
    if (!choose_ibc_leaf(r, c, static_cast<int>(node.size),
                         intra_distortion, ibc_choice, error)) {
      return false;
    }
    use_ibc = ibc_choice.found;
  }
  if (use_ibc) {
    use_intra = false;
    use_palette = false;
  }
  if (!use_intra && have_intra && !seg_forced_inter) {
    if (pipe_on && pipe_inter_valid) {
      // Joint intra/inter comparison with the pipeline objective.  The intra
      // candidate pays its own is_inter symbol, mode/MRL/DIP/CfL syntax and
      // the context-coded coefficients; the inter side's terms were recorded
      // by evaluate_inter_leaf() for the emitted skip or residual form.
      syntax::BlockSyntax intra_probe = probe_node_for(best);
      intra_probe.is_inter = 0;
      intra_probe.skip_flag = 0;
      double intra_is_inter = 2.0;
      double unused_skip = 0.0;
      if (!pipe_probe_.price_leaf_flags(intra_probe, &intra_is_inter,
                                        &unused_skip)) {
        intra_is_inter = 2.0;
        ++(eff_on ? eff_perc::counters().fallbacks
                : pipe_perc::counters().fallbacks);
      }
      const double intra_bits =
          intra_is_inter +
          intra_luma_syntax_bits(intra_y_mode, intra_y_angle_delta,
                                 intra_mrl_index, intra_mrl_sec_index,
                                 intra_use_dip) +
          intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx) +
          coeff_bits(intra_y, 0, false) +
          coeff_bits(intra_u, 1, false) +
          coeff_bits(intra_v, 2, false);
      double intra_distortion = 0.0;
      if (!chroma_part) {
        intra_distortion += sse_pipe(0, ax0, ay0, w, h, intra_y.recon);
      }
      if (owns_chroma) {
        intra_distortion += sse_pipe(1, acx0, acy0, cw, ch, intra_u.recon);
        intra_distortion += sse_pipe(2, acx0, acy0, cw, ch, intra_v.recon);
      }
      use_intra =
          intra_distortion + pipe_lambda * intra_bits <
          pipe_inter_distortion + pipe_lambda * pipe_inter_bits;
      if (trace_) {
        std::cerr << "  pipe leafrd intra=" << intra_distortion << "+"
                  << pipe_lambda << "*" << intra_bits << " inter="
                  << pipe_inter_distortion << "+" << pipe_lambda << "*"
                  << pipe_inter_bits << " use_intra="
                  << (use_intra ? 1 : 0) << "\n";
      }
      {
        // Decision-change count against the historical constants computed on
        // exactly the same intra and inter plans.
        const double lambda = leaf_rd_lambda(st.current_qindex, options_);
        const double hist_intra_bits =
            2.0 + luma_plan_bits(intra_y, false) +
            estimate_tx_bits(intra_u, false) +
            estimate_tx_bits(intra_v, false) +
            intra_luma_syntax_bits(intra_y_mode, intra_y_angle_delta,
                                   intra_mrl_index, intra_mrl_sec_index,
                                   intra_use_dip) +
            intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx);
        double hist_inter_bits =
            2.0 + inter_syntax_bits(best.mode, best.drl_index, best.mvd.y,
                                    best.mvd.x, best.use_amvd) +
            (bawp_applied ? 1.0 : 0.0) +
            (inter_intra_applied ? 2.0 : 0.0);
        if (inter_residual) {
          hist_inter_bits += luma_plan_bits(res_y, true) +
                             estimate_tx_bits(res_u, false) +
                             estimate_tx_bits(res_v, false);
        }
        const bool use_intra_hist =
            static_cast<double>(intra_sse) + lambda * hist_intra_bits <
            static_cast<double>(inter_luma_sse) +
                lambda * hist_inter_bits;
        if (use_intra != use_intra_hist) {
          ++(eff_on ? eff_perc::counters().intra_flips
                    : pipe_perc::counters().intra_flips);
        }
      }
    } else if (exact_on) {
      // Exact intra/inter comparison: both sides are priced with the same
      // QP-derived leaf lambda, the reconstruction distortion of every plane
      // the leaf owns and the same syntax + coefficient estimate the leaf
      // decision charges.  The inter side's terms were recorded by
      // evaluate_inter_leaf(), including the residual plan it chose.
      const double lambda = leaf_rd_lambda(st.current_qindex, options_);
      double intra_distortion = 0.0;
      if (!chroma_part) {
        intra_distortion +=
            static_cast<double>(sse_vis(0, ax0, ay0, w, h, intra_y.recon));
      }
      if (owns_chroma) {
        intra_distortion += static_cast<double>(
            sse_vis(1, acx0, acy0, cw, ch, intra_u.recon));
        intra_distortion += static_cast<double>(
            sse_vis(2, acx0, acy0, cw, ch, intra_v.recon));
      }
      const double intra_bits =
          2.0 + decision_coeff_bits(intra_y, false) +
          decision_coeff_bits(intra_u, false) +
          decision_coeff_bits(intra_v, false) +
          intra_luma_syntax_bits(intra_y_mode, intra_y_angle_delta,
                                 intra_mrl_index, intra_mrl_sec_index,
                                 intra_use_dip) +
          intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx);
      use_intra =
          intra_distortion + lambda * intra_bits <
          exact_inter_distortion + lambda * exact_inter_bits;
      if (use_intra) ++exact_rd::counters().intra_flipped;
      if (trace_) {
        std::cerr << "  leafrd exact intra=" << intra_distortion << "+"
                  << lambda << "*" << intra_bits << " inter="
                  << exact_inter_distortion << "+" << lambda << "*"
                  << exact_inter_bits << " use_intra="
                  << (use_intra ? 1 : 0) << "\n";
      }
    } else if (options_.enable_leaf_rd) {
      // One leaf cost for the intra/inter decision: coded SSE plus one
      // QP-derived lambda times the real-ish syntax rate.  Both sides pay
      // the same is_inter/skip base; each adds its own mode syntax and its
      // residual estimate, so a mode that is cheap to signal can win even
      // when its prediction SSE is slightly worse.
      const double lambda = leaf_rd_lambda(st.current_qindex, options_);
      double intra_bits =
          2.0 + decision_coeff_bits(intra_y, false) +
          decision_coeff_bits(intra_u, false) +
          decision_coeff_bits(intra_v, false) +
          intra_luma_syntax_bits(intra_y_mode, intra_y_angle_delta,
                                 intra_mrl_index, intra_mrl_sec_index,
                                 intra_use_dip) +
          intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx);
      double inter_bits =
          2.0 + inter_syntax_bits(best.mode, best.drl_index, best.mvd.y,
                                  best.mvd.x, best.use_amvd);
      if (inter_residual) {
        inter_bits += decision_coeff_bits(res_y, true) +
                      decision_coeff_bits(res_u, false) +
                      decision_coeff_bits(res_v, false);
      }
      const double hist_intra_bits =
          2.0 + luma_plan_bits(intra_y, false) +
          estimate_tx_bits(intra_u, false) +
          estimate_tx_bits(intra_v, false) +
          intra_luma_syntax_bits(intra_y_mode, intra_y_angle_delta,
                                 intra_mrl_index, intra_mrl_sec_index,
                                 intra_use_dip) +
          intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx);
      double hist_inter_bits =
          2.0 + inter_syntax_bits(best.mode, best.drl_index, best.mvd.y,
                                  best.mvd.x, best.use_amvd);
      if (inter_residual) {
        hist_inter_bits += luma_plan_bits(res_y, true) +
                           estimate_tx_bits(res_u, false) +
                           estimate_tx_bits(res_v, false);
      }
      const bool intra_hist =
          static_cast<double>(intra_sse) + lambda * hist_intra_bits <
          static_cast<double>(inter_luma_sse) + lambda * hist_inter_bits;
      if (rate_model_on) {
        // is_inter is coded on both sides: the inter leaf pays -log2 P(1)
        // (plus skip), the intra leaf -log2 P(0).  The mode information and
        // the residual then use the exact probe / Laplacian.
        syntax::BlockSyntax probe = probe_node_for(best);
        probe.skip_flag = inter_residual ? 0 : 1;
        probe.use_bawp = bawp_applied ? 1 : 0;
        probe.motion_mode = inter_intra_applied ? 1 : 0;
        rate_hybrid::Terms inter_terms;
        if (rate_probe_.price_full(probe, &inter_terms)) {
          inter_bits = inter_terms.total();
          if (inter_residual) {
            inter_bits += decision_coeff_bits(res_y, true) +
                          decision_coeff_bits(res_u, false) +
                          decision_coeff_bits(res_v, false);
          }
        }
        double intra_is_inter = 2.0;
        double unused_skip = 0.0;
        probe.is_inter = 0;
        if (rate_probe_.price_leaf_flags(probe, &intra_is_inter,
                                         &unused_skip)) {
          intra_bits = intra_is_inter + decision_coeff_bits(intra_y, false) +
                       decision_coeff_bits(intra_u, false) +
                       decision_coeff_bits(intra_v, false) +
                       intra_luma_syntax_bits(
                           intra_y_mode, intra_y_angle_delta, intra_mrl_index,
                           intra_mrl_sec_index, intra_use_dip) +
                       intra_chroma_syntax_bits(intra_is_cfl,
                                                options_.enable_cctx);
        }
      }
      use_intra = static_cast<double>(intra_sse) + lambda * intra_bits <
                  static_cast<double>(inter_luma_sse) + lambda * inter_bits;
      if (rate_hybrid::enabled() && rate_reporting &&
          use_intra != intra_hist) {
        rate_hybrid::report::count_decision("intra");
      }
      if (trace_) {
        std::cerr << "  leafrd intra=" << intra_sse << "+" << lambda << "*"
                  << intra_bits << " inter=" << inter_luma_sse << "+"
                  << lambda << "*" << inter_bits << " use_intra="
                  << (use_intra ? 1 : 0) << "\n";
      }
    } else if (options_.enable_inter_residual) {
      // Rate-aware decision on the opt-in residual path.  The historical
      // path compared raw SSE only, so a fine quantiser made intra win even
      // when it cost two orders of magnitude more bits.
      const double lambda = options_.inter_rd_lambda;
      const double intra_bits =
          3.0 + luma_plan_bits(intra_y, false) +
          estimate_tx_bits(intra_u, false) +
          estimate_tx_bits(intra_v, false);
      double inter_bits = 8.0;
      if (inter_residual) {
        inter_bits += 3.0 + luma_plan_bits(res_y, true) +
                      estimate_tx_bits(res_u, false) +
                      estimate_tx_bits(res_v, false);
      }
      use_intra = static_cast<double>(intra_sse) + lambda * intra_bits <
                  static_cast<double>(inter_luma_sse) + lambda * inter_bits;
    } else {
      use_intra = intra_sse < inter_luma_sse;
    }
  }

  if (use_ibc) {
    entry.is_inter = 0;
    entry.skip_flag = 0;
    entry.y_mode = 0;  // DC_PRED (read_intrabc_info forces it)
    entry.flags |= syntax::kFlagIsIbc;
    entry.ibc_skip = 1;
    entry.ibc_mode = ibc_choice.stack_index >= 0 ? 1 : 0;
    entry.ibc_ref_mv_idx =
        ibc_choice.stack_index >= 0
            ? static_cast<std::uint8_t>(ibc_choice.stack_index)
            : 0;
    entry.ibc_bv_row = ibc_choice.bv.row;
    entry.ibc_bv_col = ibc_choice.bv.col;
    entry.morph_pred = ibc_choice.morph ? 1 : 0;
    write_plane_block(recon_[0], x0, y0, w, h, ibc_choice.luma);
    write_plane_block(recon_[1], cx0, cy0, cw, ch, ibc_choice.u);
    write_plane_block(recon_[2], cx0, cy0, cw, ch, ibc_choice.v);
    update_mv_grid(r, c, w4, h4, 0, 0, inter::mc::Mv{0, 0});
    set_intra_joint(r, c, w4, h4, 0);
    ++st.intra_blocks;
    if (trace_) {
      std::cerr << "m2 ibc r=" << r << " c=" << c << " " << w << "x" << h
                << " bv=(" << ibc_choice.bv.row << "," << ibc_choice.bv.col
                << ") mode=" << static_cast<int>(entry.ibc_mode)
                << " morph=" << static_cast<int>(entry.morph_pred)
                << " sse=" << ibc_choice.sse << "\n";
    }
  } else if (use_palette) {
    if (!have_intra) {
      error = "palette fallback unavailable";
      return false;
    }
    // §5.20.8: palette is luma only and forces YMode = DC_PRED.  The block
    // codes eob=0 on all three planes, so the reconstruction is exactly the
    // palette prediction plus the ordinary (recomputed) chroma path.  CfL is
    // not signalled: its derived alpha would read the palette-predicted
    // luma, so recompute chroma DC from the palette reconstruction.
    entry.is_inter = 0;
    entry.skip_flag = 0;
    entry.y_mode = 0;
    entry.y_angle_delta = 0;
    entry.uv_mode = 0;
    entry.palette_index = static_cast<std::uint32_t>(sink.palettes.size());
    sink.palettes.push_back(std::move(palette_choice.result));
    // Update the encoder-side §5.20.8.2 cache source only now: an IBC leaf
    // also seen this block would have cleared the palette decision.
    palette::store_palette_state(palette_state_, r, c, w, h,
                                 sink.palettes.back());
    const std::optional<txv::TxParams> params_opt =
        txv::make_tx_params(w, h, txv::TxType::kDctDct);
    if (!params_opt.has_value()) {
      error = "unsupported palette block transform size";
      return false;
    }
    intra_y = TxPlan{};
    intra_y.tx_size = params_opt->tx_size;
    intra_y.adj_w = params_opt->w;
    intra_y.adj_h = params_opt->h;
    intra_y.eob = 0;
    intra_y.recon = palette_choice.luma;
    write_plane_block(recon_[0], x0, y0, w, h, intra_y.recon);
    if (!intra_dc_predict(recon_[1], source_, 1, cx0, cy0, cw, ch,
                          chroma_max_x, chroma_max_y, chroma_left,
                          chroma_above, options_.enable_ibp,
                          intra_u_pred) ||
        !intra_dc_predict(recon_[2], source_, 2, cx0, cy0, cw, ch,
                          chroma_max_x, chroma_max_y, chroma_left,
                          chroma_above, options_.enable_ibp,
                          intra_v_pred) ||
        !compute_tx(source_, 1, acx0, acy0, cw, ch, intra_u_pred,
                    st.plane_quant[1], intra_u, error,
                    qm_enabled_ ? &qm_state_ : nullptr) ||
        !compute_tx(source_, 2, acx0, acy0, cw, ch, intra_v_pred,
                    st.plane_quant[2], intra_v, error,
                    qm_enabled_ ? &qm_state_ : nullptr)) {
      error = "palette chroma prediction failed";
      return false;
    }
    write_plane_block(recon_[1], cx0, cy0, cw, ch, intra_u.recon);
    write_plane_block(recon_[2], cx0, cy0, cw, ch, intra_v.recon);
    entry.levels_offset = static_cast<std::uint32_t>(sink.levels.size());
    write_luma_plan(entry, intra_y, sink.levels);
    {
      const TxPlan* plans[2] = {&intra_u, &intra_v};
      for (int plane = 1; plane < 3; ++plane) {
        entry.tx_size[plane] =
            static_cast<std::uint8_t>(plans[plane - 1]->tx_size);
        entry.eob[plane] = static_cast<std::uint32_t>(plans[plane - 1]->eob);
        const int area = plans[plane - 1]->adj_w * plans[plane - 1]->adj_h;
        std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
        syntax::build_scan(plans[plane - 1]->adj_w, plans[plane - 1]->adj_h,
                           scan);
        for (int c2 = 0; c2 < plans[plane - 1]->eob; ++c2) {
          sink.levels.push_back(plans[plane - 1]->levels[scan[
              static_cast<std::size_t>(c2)]]);
        }
      }
    }
    update_mv_grid(r, c, w4, h4, 0, 0, inter::mc::Mv{0, 0});
    set_intra_joint(r, c, w4, h4, 0);
    ++st.intra_blocks;
    if (trace_) {
      std::cerr << "m2 palette r=" << r << " c=" << c << " " << w << "x" << h
                << " size=" << sink.palettes.back().size
                << " direction=" << sink.palettes.back().direction
                << " sse=" << palette_choice.sse << "\n";
    }
  } else if (use_intra) {
    if (!have_intra) {
      error = "intra fallback unavailable";
      return false;
    }
    entry.is_inter = 0;
    entry.skip_flag = 0;
    entry.y_mode = chroma_part ? 0 : intra_y_mode;
    entry.y_angle_delta = intra_y_angle_delta;
    entry.mrl_index = intra_mrl_index;
    entry.mrl_sec_index = intra_mrl_sec_index ? 1 : 0;
    entry.use_dip = intra_use_dip ? 1 : 0;
    entry.dip_transpose = intra_dip_transpose ? 1 : 0;
    entry.dip_mode = static_cast<std::uint8_t>(intra_dip_mode);
    // DC_PRED unless the AV2ENC_RD_UVFULL chroma search picked a directional /
    // smooth / Paeth mode; CfL keeps DC_PRED as its base mode.
    entry.uv_mode = static_cast<std::uint8_t>(intra_uv_mode);
    if (intra_is_cfl) {
      entry.flags |= syntax::kFlagIsCfl;
      entry.cfl_index = static_cast<std::uint8_t>(intra_cfl_index);
      entry.cfl_mh_dir = static_cast<std::uint8_t>(intra_cfl_mh_dir);
    }
    entry.levels_offset = static_cast<std::uint32_t>(sink.levels.size());
    if (!chroma_part) {
      write_plane_block(recon_[0], x0, y0, w, h, intra_y.recon);
      write_luma_plan(entry, intra_y, sink.levels);
    }
    entry.cctx_type = static_cast<std::uint8_t>(intra_cctx_type);
    if (owns_chroma) {
      write_plane_block(recon_[1], cx0, cy0, cw, ch, intra_u.recon);
      write_plane_block(recon_[2], cx0, cy0, cw, ch, intra_v.recon);
      const TxPlan* plans[2] = {&intra_u, &intra_v};
      for (int plane = 1; plane < 3; ++plane) {
        entry.tx_size[plane] =
            static_cast<std::uint8_t>(plans[plane - 1]->tx_size);
        entry.eob[plane] = static_cast<std::uint32_t>(plans[plane - 1]->eob);
        const int area = plans[plane - 1]->adj_w * plans[plane - 1]->adj_h;
        std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
        syntax::build_scan(plans[plane - 1]->adj_w, plans[plane - 1]->adj_h,
                           scan);
        for (int c2 = 0; c2 < plans[plane - 1]->eob; ++c2) {
          sink.levels.push_back(plans[plane - 1]->levels[scan[
              static_cast<std::size_t>(c2)]]);
        }
      }
    }
    if (!chroma_part) {
      update_mv_grid(r, c, w4, h4, 0, 0, inter::mc::Mv{0, 0});
      set_intra_joint(r, c, w4, h4,
                      syntax::intra_joint_mode(intra_y_mode,
                                               intra_y_angle_delta));
    }
    if (owns_chroma) {
      // §7.13.2.15/.16: record this block's UVMode for the following leaves'
      // chroma filterTypeAbove/Left.  Inert (all zero) unless AV2ENC_RD_UVFULL
      // codes a smooth chroma mode.
      set_uv_smooth(cy0 / 4, cx0 / 4, cw / 4, ch / 4, intra_uv_mode,
                    intra_is_cfl);
    }
    ++st.intra_blocks;
    if (trace_) {
      int trace_ist_sec = intra_y.sec_tx_type;
      int trace_ist_set = intra_y.stx_set;
      if (intra_y.searched && intra_y.parts.size() == 1) {
        trace_ist_sec = intra_y.parts[0].sec_tx_type;
        trace_ist_set = intra_y.parts[0].stx_set;
      }
      std::cerr << "m2 intra r=" << r << " c=" << c << " " << w << "x" << h
                << " mode=" << static_cast<int>(intra_y_mode)
                << " delta=" << static_cast<int>(intra_y_angle_delta)
                << " mrl=" << static_cast<int>(intra_mrl_index)
                << " sec=" << (intra_mrl_sec_index ? 1 : 0)
                << (intra_use_dip
                        ? " dip=" + std::to_string(intra_dip_mode) + "/" +
                              std::to_string(intra_dip_transpose ? 1 : 0)
                        : std::string());
      if (options_.enable_ibp && !chroma_part) {
        // §5.4.5/§7.13.2: report the decision the decoder derives from the
        // signalled (mode, angle_delta).  The gate greps ibp_blend=1 to
        // prove the two-angle path is live.
        ibp::DecisionParams dp;
        dp.enable_ibp = true;
        dp.mode = static_cast<intra::Mode>(intra_y_mode);
        dp.angle_delta = intra_y_angle_delta;
        dp.w = w;
        dp.h = h;
        dp.plane = 0;
        dp.mrl_index = intra_mrl_index;
        const ibp::Decision dec = ibp::decide(dp);
        std::cerr << " ibp_apply=" << (dec.apply_ibp_edge_rules ? 1 : 0)
                  << " ibp_blend=" << (dec.use_ibp ? 1 : 0);
      }
      std::cerr << " ist=" << trace_ist_sec << "/" << trace_ist_set
                << " sse=" << intra_sse << "\n";
    }
  } else {
    entry.is_inter = 1;
    entry.skip_flag = inter_residual ? 0 : 1;
    if (best.from_temporal) ++st.temporal_selected;
    if (best.from_refmvbank) ++st.refmvbank_selected;
    if (best.same_ref) ++st.same_ref_blocks;
    if (best.use_amvd) ++st.amvd_selected;
    if (best.block_mv_precision >= 0) ++st.flex_precision_selected;
    if (best.mvd_sign_derived) ++st.derived_sign_selected;
    entry.inter_mode = static_cast<std::uint8_t>(best.mode);
    entry.drl_index = static_cast<std::uint8_t>(best.drl_index);
    entry.ref_index = static_cast<std::uint8_t>(best.ref_index);
    entry.compound = best.compound ? 1 : 0;
    entry.ref_index1 = static_cast<std::uint8_t>(
        best.compound ? best.ref_index1 : 1);
    entry.compound_joint = best.compound && best.compound_joint ? 1 : 0;
    entry.compound_mode = static_cast<std::uint8_t>(best.compound_mode);
    entry.drl_index1 = static_cast<std::uint8_t>(best.drl_index1);
    if (best.compound) {
      // A compound leaf never carries a residual in the gated subset.
      entry.skip_flag = 1;
      inter_residual = false;
      // §5.20.7.16 comp_group_idx: only an explicitly masked candidate sets
      // it.  AVERAGE (compound_type 2) and the new compound modes code 0.
      entry.compound_masked =
          best.compound_type == 0 || best.compound_type == 1 ? 1 : 0;
      // §5.20.7.16: without comp_group_idx = 1 there is no compound_type
      // symbol, so the AVERAGE sentinel is 0 for the writer.
      entry.compound_type = static_cast<std::uint8_t>(
          entry.compound_masked ? best.compound_type : 0);
      entry.wedge_index = static_cast<std::uint8_t>(best.wedge_index);
      entry.wedge_sign = static_cast<std::uint8_t>(best.wedge_sign);
      entry.mask_type = static_cast<std::uint8_t>(best.mask_type);
      entry.cwp_index = static_cast<std::uint8_t>(best.cwp_index);
      entry.cwp_weight = static_cast<std::int8_t>(best.cwp_weight);
      entry.use_optflow = best.use_optflow ? 1 : 0;
      entry.use_refinemv = best.use_refinemv ? 1 : 0;
      if (options_.enable_df_sub_pu &&
          (best.use_optflow || best.use_refinemv)) {
        collect_sub_pus(sink, x0, y0, w, h, best.use_optflow,
                        best.use_refinemv);
      }
    }
    if (best.compound) {
      entry.mv_y = best.mv.y;
      entry.mv_x = best.mv.x;
      entry.mv1_y = best.mv1.y;
      entry.mv1_x = best.mv1.x;
    } else {
      entry.mv_y = best.mode == syntax::kInterGlobalMv ? 0 : best.mv.y;
      entry.mv_x = best.mode == syntax::kInterGlobalMv ? 0 : best.mv.x;
      entry.mv1_y = 0;
      entry.mv1_x = 0;
    }
    entry.mvd_y = best.mvd.y;
    entry.mvd_x = best.mvd.x;
    entry.use_amvd = best.use_amvd ? 1 : 0;
    entry.block_mv_precision =
        static_cast<std::int8_t>(best.block_mv_precision);
    entry.mvd_sign_derived = best.mvd_sign_derived ? 1 : 0;
    // §5.20.7.6 switchable interp_filter carried on the inter leaf.  The
    // writer codes it where needs_interp_filter() is true; for every other
    // carrier (compound, warp, large GLOBALMV) it is 0 = EIGHTTAP and no
    // symbol is coded.  The intra-only mrl fields are unused here.
    entry.mrl_index = static_cast<std::uint8_t>(inter_filter);
    entry.use_bawp = bawp_applied ? 1 : 0;
    entry.bawp_alpha = static_cast<std::int16_t>(bawp_fit.alpha);
    if (inter_intra_applied) {
      // §5.20.7.14 INTERINTRA: the block is skip-only, RefFrame[1] is
      // INTRA_FRAME and wedge_sign is forced 0.
      entry.skip_flag = 1;
      inter_residual = false;
      entry.motion_mode = 1;
      entry.interintra_mode = static_cast<std::uint8_t>(inter_intra_mode);
      entry.wedge_interintra = inter_intra_wedge ? 1 : 0;
      entry.wedge_index = static_cast<std::uint8_t>(inter_intra_index);
    }
    write_plane_block(recon_[0], x0, y0, w, h, inter_y);
    write_plane_block(recon_[1], cx0, cy0, cw, ch, inter_u);
    write_plane_block(recon_[2], cx0, cy0, cw, ch, inter_v);
    if (inter_residual) {
      entry.levels_offset = static_cast<std::uint32_t>(sink.levels.size());
      write_luma_plan(entry, res_y, sink.levels);
      entry.cctx_type = static_cast<std::uint8_t>(inter_cctx_type);
      const TxPlan* plans[2] = {&res_u, &res_v};
      for (int plane = 1; plane < 3; ++plane) {
        entry.tx_size[plane] =
            static_cast<std::uint8_t>(plans[plane - 1]->tx_size);
        entry.eob[plane] =
            static_cast<std::uint32_t>(plans[plane - 1]->eob);
        const int area = plans[plane - 1]->adj_w * plans[plane - 1]->adj_h;
        std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
        syntax::build_scan(plans[plane - 1]->adj_w, plans[plane - 1]->adj_h,
                           scan);
        for (int c2 = 0; c2 < plans[plane - 1]->eob; ++c2) {
          sink.levels.push_back(plans[plane - 1]->levels[scan[
              static_cast<std::size_t>(c2)]]);
        }
      }
    }
    update_mv_grid(r, c, w4, h4, 1, best.ref_index, best.mv,
                   best.compound ? best.ref_index1 : merge::kRefNone,
                   best.compound ? &best.mv1 : nullptr,
                   /*warp0=*/best.use_warp, /*warp1=*/false,
                   /*is_wedge=*/best.compound && best.compound_type == 0,
                   best.wedge_index, best.wedge_sign,
                   static_cast<int>(node.size),
                   best.compound ? best.cwp_index : 0);
    set_intra_joint(r, c, w4, h4, 0);  // inter_block_mode_info(): DC
    ++st.inter_blocks;
    if (trace_) {
      std::cerr << "m2 inter r=" << r << " c=" << c << " " << w << "x" << h
                << " mode=" << static_cast<int>(best.mode)
                << " drl=" << best.drl_index
                << " ref=" << best.ref_index << " mv=(" << best.mv.y << ","
                << best.mv.x << ") mv1=(" << best.mv1.y << ","
                << best.mv1.x << ") sse=" << inter_luma_sse
                << " skip=" << static_cast<int>(entry.skip_flag)
                << " comp=" << static_cast<int>(entry.compound)
                << " cm=" << static_cast<int>(entry.compound_mode)
                << " cj=" << static_cast<int>(entry.compound_joint)
                << " ifilter=" << static_cast<int>(entry.mrl_index)
                << " optflow=" << static_cast<int>(entry.use_optflow)
                << " refinemv=" << static_cast<int>(entry.use_refinemv)
                << " ct=" << static_cast<int>(entry.compound_type)
                << " wedge=" << static_cast<int>(entry.wedge_index)
                << " sign=" << static_cast<int>(entry.wedge_sign)
                << " mask=" << static_cast<int>(entry.mask_type) << "\n";
    }
  }

#if defined(AV2ENC_VIDEO_DELTAQ)
  if (qp_tools_active()) {
    // §5.20.7.2: a skipped inter leaf codes no segment id when the frame has
    // no lossless segment; the decoder infers the §5.20.7.4 neighbour
    // prediction.  Replay it here so the deblock metadata uses the same
    // segment id (and therefore the same effective qindex) the decoder will.
    int segment_id = seg_enabled_ ? st.current_segment_id : 0;
    // A SEG_LVL_GLOBALMV segment cannot hold an intra leaf: §5.20.7.3
    // forces is_inter = 1 when the feature is active.  If no inter candidate
    // was available, fall back to the ALT_Q-free baseline segment (the
    // table gives every feature segment ALT_Q 0 in that composition).
    if (seg_enabled_ && !keyframe_ && entry.is_inter == 0 &&
        ((globalmv_seg_id_ >= 0 && segment_id == globalmv_seg_id_) ||
         (skip_seg_id_ >= 0 && segment_id == skip_seg_id_))) {
      segment_id = 0;
    }
    bool inferred = false;
    if (seg_enabled_ && !keyframe_ && entry.is_inter != 0 &&
        entry.skip_flag != 0 && !enc_seg_.has_lossless_segment &&
        !enc_seg_.seg_id_pre_skip) {
      segment_id = deltaq::segment_prediction(enc_seg_grid_, r, c);
      inferred = true;
    }
    int qindex = st.current_qindex;
    if (seg_enabled_) {
      std::string qp_error;
      if (!deltaq::effective_qindex(
              enc_seg_, base_q_index(), st.current_qindex, delta_q_enabled_,
              segment_id, /*ignore_delta_q=*/false, qindex, qp_error)) {
        error = "get_qindex: " + qp_error;
        return false;
      }
    }
    if (qindex < 0 || qindex > 255) {
      error = "get_qindex produced an out-of-range quantizer";
      return false;
    }
    entry.segment_id = static_cast<std::uint8_t>(segment_id);
    entry.qindex = static_cast<std::int16_t>(qindex);
    if (trace_ && inferred) {
      std::cerr << "segq inferred r=" << r << " c=" << c
                << " segment=" << segment_id << " qindex=" << qindex
                << "\n";
    }
    if (seg_enabled_) {
      deltaq::SegmentBlock segment_block;
      segment_block.mi_row = r;
      segment_block.mi_col = c;
      segment_block.bw4 = w4;
      segment_block.bh4 = h4;
      std::string seg_error;
      if (!deltaq::commit_segment_block(enc_seg_grid_, segment_block,
                                        segment_id, seg_error)) {
        error = "segment map: " + seg_error;
        return false;
      }
    }
  }
#endif

  // AV2ENC_EXACT_RD partition input: record the measured (D, R) of the leaf
  // the encoder will actually emit.  D is the reconstruction SSE of every
  // plane the leaf owns; R is the same syntax + coefficient estimate the leaf
  // decision charges.  The partition refinement later prices these with the
  // node's own partition lambda and compares them with the NEWMV probe on
  // nodes that were not encoded in the first pass.  Gate off: no record, no
  // extra work.
  if (exact_rd::enabled()) {
    exact_rd::LeafCost measured;
    if (use_ibc) {
      if (!chroma_part) {
        measured.distortion += static_cast<double>(
            sse_vis(0, ax0, ay0, w, h, ibc_choice.luma));
      }
      if (owns_chroma) {
        measured.distortion += static_cast<double>(
            sse_vis(1, acx0, acy0, cw, ch, ibc_choice.u));
        measured.distortion += static_cast<double>(
            sse_vis(2, acx0, acy0, cw, ch, ibc_choice.v));
      }
      // is_inter/is_ibc + mode/BV syntax: coarse but consistent with the
      // probe's constant mode/skip terms.  IBC is keyframe-only, so this never
      // enters the P-frame partition refinement.
      measured.bits = 8.0;
    } else if (use_palette) {
      if (!chroma_part) {
        measured.distortion += static_cast<double>(
            sse_vis(0, ax0, ay0, w, h, palette_choice.luma));
      }
      if (owns_chroma) {
        measured.distortion += static_cast<double>(
            sse_vis(1, acx0, acy0, cw, ch, intra_u.recon));
        measured.distortion += static_cast<double>(
            sse_vis(2, acx0, acy0, cw, ch, intra_v.recon));
      }
      measured.bits =
          2.0 + 4.0 + estimate_tx_bits(intra_y, false) +
          estimate_tx_bits(intra_u, false) +
          estimate_tx_bits(intra_v, false) +
          intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx);
    } else if (use_intra) {
      if (!chroma_part) {
        measured.distortion += static_cast<double>(
            sse_vis(0, ax0, ay0, w, h, intra_y.recon));
      }
      if (owns_chroma) {
        measured.distortion += static_cast<double>(
            sse_vis(1, acx0, acy0, cw, ch, intra_u.recon));
        measured.distortion += static_cast<double>(
            sse_vis(2, acx0, acy0, cw, ch, intra_v.recon));
      }
      measured.bits =
          2.0 + luma_plan_bits(intra_y, false) +
          estimate_tx_bits(intra_u, false) +
          estimate_tx_bits(intra_v, false) +
          intra_luma_syntax_bits(intra_y_mode, intra_y_angle_delta,
                                 intra_mrl_index, intra_mrl_sec_index,
                                 intra_use_dip) +
          intra_chroma_syntax_bits(intra_is_cfl, options_.enable_cctx);
    } else {
      if (!chroma_part) {
        measured.distortion +=
            static_cast<double>(sse_vis(0, ax0, ay0, w, h, inter_y));
      }
      if (owns_chroma) {
        measured.distortion += static_cast<double>(
            sse_vis(1, acx0, acy0, cw, ch, inter_u));
        measured.distortion += static_cast<double>(
            sse_vis(2, acx0, acy0, cw, ch, inter_v));
      }
      measured.bits =
          2.0 + inter_syntax_bits(best.mode, best.drl_index, best.mvd.y,
                                  best.mvd.x, best.use_amvd);
      if (bawp_applied) measured.bits += 1.0;
      if (inter_intra_applied) measured.bits += 2.0;
      if (inter_residual) {
        measured.bits += luma_plan_bits(res_y, true) +
                         estimate_tx_bits(res_u, false) +
                         estimate_tx_bits(res_v, false);
      }
    }
    measured.valid = true;
    if (sink.leaf_costs.size() < sink.blocks.size()) {
      sink.leaf_costs.resize(sink.blocks.size());
    }
    sink.leaf_costs[index] = measured;
  }

  // R09 (AV2ENC_RATE_HYBRID_S): record the decision-side rate for this leaf
  // and advance the probe.  Measurement runs (gate off, measure on) also come
  // here, so the report covers the historical decisions byte-identically.
  if (rate_reporting) {
    syntax::BlockSyntax probe = entry;
    rate_hybrid::Terms terms;
    const bool priced = rate_probe_.price_full(probe, &terms);
    if (entry.is_inter) {
      syntax::BlockSyntax::RateDiag& diag = entry.rate_diag;
      double old_syntax =
          2.0 + inter_syntax_bits(entry.inter_mode, entry.drl_index,
                                  entry.mvd_y, entry.mvd_x,
                                  entry.use_amvd != 0);
      if (entry.use_bawp) old_syntax += 1.0;
      if (entry.motion_mode == 1) old_syntax += 2.0;
      diag.syntax_old = static_cast<float>(old_syntax);
      diag.syntax_new =
          static_cast<float>(priced ? terms.total() : old_syntax);
      diag.term[0] = static_cast<float>(terms.is_inter);
      diag.term[1] = static_cast<float>(terms.skip);
      diag.term[2] = static_cast<float>(terms.mode);
      diag.term[3] = static_cast<float>(terms.ref + terms.comp_mode);
      diag.term[4] = static_cast<float>(terms.drl);
      diag.term[5] = static_cast<float>(terms.mvd);
      diag.bawp = static_cast<float>(terms.bawp);
      if (entry.skip_flag == 0) {
        float eob_sum = 0.0f;
        float nz_sum = 0.0f;
        float abs_sum = 0.0f;
        const auto accumulate = [&](const TxPlan& plan) {
          if (!plan.searched) {
            eob_sum += static_cast<float>(plan.eob);
            std::vector<std::uint16_t> scan(
                static_cast<std::size_t>(plan.adj_w * plan.adj_h));
            syntax::build_scan(plan.adj_w, plan.adj_h, scan);
            for (int c2 = 0; c2 < plan.eob; ++c2) {
              const std::int32_t v =
                  plan.levels[scan[static_cast<std::size_t>(c2)]];
              if (v != 0) {
                nz_sum += 1.0f;
                abs_sum += static_cast<float>(v < 0 ? -v : v);
              }
            }
            return;
          }
          for (const TxPlan::Part& part : plan.parts) {
            eob_sum += static_cast<float>(part.eob);
            for (int c2 = 0; c2 < part.eob; ++c2) {
              const std::int32_t v = part.levels[static_cast<std::size_t>(c2)];
              if (v != 0) {
                nz_sum += 1.0f;
                abs_sum += static_cast<float>(v < 0 ? -v : v);
              }
            }
          }
        };
        accumulate(res_y);
        accumulate(res_u);
        accumulate(res_v);
        diag.eob = eob_sum;
        diag.nonzero = nz_sum;
        diag.sum_abs = abs_sum;
        diag.coeff_old =
            static_cast<float>(luma_plan_bits(res_y, true) +
                               estimate_tx_bits(res_u, false) +
                               estimate_tx_bits(res_v, false));
        diag.coeff_new =
            static_cast<float>(laplacian_plan_bits(res_y, true) +
                               laplacian_plan_bits(res_u, false) +
                               laplacian_plan_bits(res_v, false));
      }
      diag.valid = 1;
      diag.fallback = priced ? 0 : 1;
      if (!priced) rate_hybrid::report::count_fallback();
    }
    rate_probe_.commit(entry);
  }

  // AV2ENC_RD_TAIL_FLOOR: stash the live inter-syntax price of the leaf the
  // writer will code, measured against the probe state this leaf starts from.
  // The measured coding-block decision prices every candidate leaf with it, so
  // a 16x16 leaf and its 8x8-class children are compared in the same coded
  // context.  Inert unless the gate armed this picture.
  if ((tail_floor_active_ || part_diag_active_ || joint_p_active_) &&
      pipe_on) {
    rate_hybrid::Terms tail_terms;
    const bool priced = pipe_probe_.price_full(entry, &tail_terms);
    if (tail_floor_leaf_bits_.size() <= index) {
      tail_floor_leaf_bits_.resize(index + 1, 0.0);
      tail_floor_leaf_bits_ok_.resize(index + 1, 0);
    }
    tail_floor_leaf_bits_[index] = priced ? tail_terms.total() : 0.0;
    tail_floor_leaf_bits_ok_[index] = priced ? 1 : 0;
    if (!priced) ++tail_floor::counters().fallback;
  }

  // AV2ENC_PIPE_PERC: advance the exact-syntax probe with the leaf the writer
  // will code, in coded order, so the next leaf's contexts match the writer.
  if (pipe_on && !chroma_part) pipe_probe_.commit(entry);

  // §6.19.2.3: this leaf's cells are now decoded for the availability of the
  // following leaves in this superblock.
  if (!chroma_part) st.decoded[0].fill(c, r, w4, h4);
  if (owns_chroma) {
    st.decoded[1].fill(cx0 / 4, cy0 / 4, cw / 4, ch / 4);
    st.decoded[2].fill(cx0 / 4, cy0 / 4, cw / 4, ch / 4);
  }
  if (ibc_enabled_) {
    ibcs::finish_ibc_block(ibc_buffer_, ibc_neighbours_, r, c,
                           static_cast<int>(node.size), use_ibc,
                           use_ibc ? ibc_choice.bv : ibc::BlockVector{});
  }
  (void)node;
  return true;
}

bool InterFrameEncoder::tip_block_ref_active() const {
  return options_.tip_block_ref && !keyframe_ && temporal_ctx_.tip_as_ref;
}

temporal::MvPair InterFrameEncoder::tip_unit_mvs(int unit_mi_row, int unit_mi_col, inter::mc::Mv block_mv,
                              const temporal::TipConfig& tip) const {
  temporal::TipCandInput in;
  in.cand_row = unit_mi_row;
  in.cand_col = unit_mi_col;
  in.base_row = unit_mi_row;
  in.base_col = unit_mi_col;
  in.is_16x16 = false;
  in.base_mv = temporal::Mv{block_mv.y, block_mv.x};
  in.field = temporal_ctx_.field;
  return temporal::get_tip_cand(in, tip);
}

bool InterFrameEncoder::predict_tip_leaf(int x0, int y0, int w, int h, int unit, inter::mc::Mv block_mv,
                      const temporal::TipConfig& tip, std::vector<temporal::MvPair>* unit_mvs,
                      std::string& error) {
  const int units_x = (w + unit - 1) / unit;
  const int units_y = (h + unit - 1) / unit;
  unit_mvs->assign(static_cast<std::size_t>(units_x) *
                       static_cast<std::size_t>(units_y),
                   temporal::MvPair{});
  std::vector<std::int16_t> scratch(2U * 16U * 16U, 0);
  std::vector<std::uint8_t> block(16U * 16U, 0);
  const int frame_w = mi_cols_ * 4;
  const int frame_h = mi_rows_ * 4;
  for (int uy = 0; uy < units_y; ++uy) {
    for (int ux = 0; ux < units_x; ++ux) {
      const int px = x0 + ux * unit;
      const int py = y0 + uy * unit;
      const int unit_mi_row = py / 4;
      const int unit_mi_col = px / 4;
      const temporal::MvPair mvs = tip_unit_mvs(unit_mi_row, unit_mi_col, block_mv, tip);
      (*unit_mvs)[static_cast<std::size_t>(uy) *
                      static_cast<std::size_t>(units_x) +
                  static_cast<std::size_t>(ux)] = mvs;
      for (int plane = 0; plane < 3; ++plane) {
        const int sub_x = plane == 0 ? 0 : 1;
        const int sub_y = plane == 0 ? 0 : 1;
        const int bw = unit >> sub_x;
        const int bh = unit >> sub_y;
        const int bx = px >> sub_x;
        const int by = py >> sub_y;
        // §7.13.3.2: the decoder skips a prediction unit whose top-left is
        // outside the frame.
        if (bx >= recon_[plane].width || by >= recon_[plane].height) continue;
        temporal::TipPredictInput in;
        in.ref[0] = temporal_plane(references_[0], plane);
        in.ref[1] = temporal_plane(references_[1], plane);
        in.num_refs = 2;
        in.x = bx;
        in.y = by;
        in.w = bw;
        in.h = bh;
        in.sub_x = sub_x;
        in.sub_y = sub_y;
        in.frame_w = frame_w;
        in.frame_h = frame_h;
        in.mv[0] = mvs.mv[0];
        in.mv[1] = mvs.mv[1];
        in.cwp_weight = temporal::kTipWeightingFactor[tip.tip_global_wtd_index];
        in.filter = tip.interp_filter;
        in.use_ref_area = false;
        in.dst = block.data();
        in.dst_stride = bw;
        in.scratch0 = scratch.data();
        in.scratch1 = scratch.data() + 16 * 16;
        in.scratch_stride = bw;
        if (temporal::tip_predict_block(in) != temporal::TemporalStatus::kOk) {
          error = "TIP block prediction failed";
          return false;
        }
        const int vis_w = std::min(bw, recon_[plane].width - bx);
        const int vis_h = std::min(bh, recon_[plane].height - by);
        for (int r = 0; r < vis_h; ++r) {
          std::copy_n(block.data() + static_cast<std::size_t>(r) *
                                          static_cast<std::size_t>(bw),
                      vis_w,
                      recon_[plane].samples.data() +
                          static_cast<std::ptrdiff_t>(by + r) * recon_[plane].stride + bx);
        }
      }
    }
  }
  return true;
}

temporal::TipConfig InterFrameEncoder::tip_block_config() const {
  temporal::TipConfig tip;
  tip.mode = temporal::TipFrameMode::kAsReference;
  tip.tip_global_wtd_index = temporal_ctx_.tip_weight_index;
  tip.tip_global_mv = temporal::Mv{0, 0};
  tip.interp_filter = temporal::InterpFilter::kEighttapSharp;
  tip.enable_tip_refinemv = false;
  tip.enable_refinemv = false;
  tip.opfl_refine_type = temporal::RefineType::kNone;
  tip.allow_tip_hole_fill = temporal_ctx_.tip_allow_hole_fill;
  tip.num_past_refs = temporal_ctx_.num_past_refs;
  tip.num_future_refs = temporal_ctx_.num_future_refs;
  tip.order_hint = temporal_ctx_.order_hint;
  tip.closest_past_order_hint = temporal_ctx_.tip_closest_past_order_hint;
  tip.closest_future_order_hint = temporal_ctx_.tip_closest_future_order_hint;
  return tip;
}

bool InterFrameEncoder::tip_ref_stack_ok(int r, int c, int w4, int h4, std::string& error) const {
  merge::StackParams params;
  params.mi_row = r;
  params.mi_col = c;
  params.bw4 = w4;
  params.bh4 = h4;
  params.sb_h4 = 16;
  params.mi_cols = mi_cols_;
  params.mi_rows = mi_rows_;
  params.mi = mv_grid_.data();
  params.ref_frame[0] = merge::kTipFrame;
  params.ref_frame[1] = merge::kRefNone;
  params.is_compound = false;
  params.global_mv[0] = inter::mc::Mv{0, 0};
  params.global_mv[1] = inter::mc::Mv{0, 0};
  params.reorder = merge::DrlReorder::kDisabled;
  params.max_drl_bits_minus_1 = 4;
  params.num_refs = num_total_refs_;
  params.frame_distance = frame_distance_.data();
  params.closest_past = temporal_ctx_.closest_past;
  params.closest_future = temporal_ctx_.closest_future;
  merge::MvStack stack;
  if (merge::build_stack_checked(params, &stack) != merge::Status::kOk) {
    error =
        "TIP block reference: TIP-ref MV stack needs an unimplemented "
        "§7.12.2.15/.17/.18 candidate";
    return false;
  }
  if (stack.count != 1 || stack.cand[0].mv[0].y != 0 || stack.cand[0].mv[0].x != 0) {
    error =
        "TIP block reference: expected the single zero GlobalMvs "
        "candidate";
    return false;
  }
  return true;
}

bool InterFrameEncoder::locate_tip_leaf(std::string& error) {
  tip_leaf_r_ = -1;
  tip_leaf_c_ = -1;
  tip_leaf_mv_ = inter::mc::Mv{0, 0};
  const std::uint8_t leaf_partition = static_cast<std::uint8_t>(partition::PartitionType::kNone);
  int last_inter = -1;
  for (int i = static_cast<int>(blocks_.size()) - 1; i >= 0; --i) {
    if (blocks_[static_cast<std::size_t>(i)].is_inter != 0) {
      last_inter = i;
      break;
    }
  }
  if (last_inter < 0) {
    error = "TIP block reference: no trailing inter leaf is TIP-eligible";
    return false;
  }
  for (int i = last_inter + 1; i < static_cast<int>(blocks_.size()); ++i) {
    if (blocks_[static_cast<std::size_t>(i)].is_inter != 0) {
      error =
          "TIP block reference: an inter leaf is decoded after the "
          "candidate TIP leaf";
      return false;
    }
  }
  const syntax::BlockSyntax& node = blocks_[static_cast<std::size_t>(last_inter)];
  if (node.partition != leaf_partition || node.tree_type != 0) {
    error = "TIP block reference: no trailing inter leaf is TIP-eligible";
    return false;
  }
  const int w = partition::block_width(static_cast<partition::BlockSize>(node.bsize));
  const int h = partition::block_height(static_cast<partition::BlockSize>(node.bsize));
  if (w < 8 || h < 8) {
    error = "TIP block reference: last leaf is smaller than 8x8";
    return false;
  }
  if (!tip_ref_stack_ok(node.mi_row, node.mi_col, w / 4, h / 4, error)) {
    return false;
  }
  inter::mc::Mv mv{0, 0};
  if (node.inter_mode != syntax::kInterGlobalMv) {
    mv = inter::mc::Mv{node.mv_y, node.mv_x};
  }
  if (mv.y <= inter::mc::kMvLow + 8 || mv.y >= inter::mc::kMvUpp - 8 ||
      mv.x <= inter::mc::kMvLow + 8 || mv.x >= inter::mc::kMvUpp - 8) {
    mv = inter::mc::Mv{0, 0};
  }
  tip_leaf_r_ = node.mi_row;
  tip_leaf_c_ = node.mi_col;
  tip_leaf_mv_ = mv;
  return true;
}

bool InterFrameEncoder::encode_tip_leaf(SbSink& sink, const PartitionNode& node,
                     std::size_t index, std::string& error) {
  SbSink::State& st = sink.state;
  const int r = node.mi_row;
  const int c = node.mi_col;
  const int w = partition::block_width(node.size);
  const int h = partition::block_height(node.size);
  const int w4 = w / 4;
  const int h4 = h / 4;
  const int x0 = c * 4;
  const int y0 = r * 4;
  if (w < 8 || h < 8) {
    error = "TIP block reference: target leaf is smaller than 8x8";
    return false;
  }
  if (!tip_ref_stack_ok(r, c, w4, h4, error)) return false;
  const int unit = (w >= 16 && h >= 16) ? 16 : 8;
  const temporal::TipConfig tip = tip_block_config();
  std::vector<temporal::MvPair> unit_mvs;
  if (!predict_tip_leaf(x0, y0, w, h, unit, tip_leaf_mv_, tip, &unit_mvs, error)) {
    return false;
  }
  std::int64_t sse = 0;
  {
    const int vis_w = std::min(w, static_cast<int>(source_.width) - x0);
    const int vis_h = std::min(h, static_cast<int>(source_.height) - y0);
    for (int yy = 0; yy < vis_h; ++yy) {
      for (int xx = 0; xx < vis_w; ++xx) {
        const int d = static_cast<int>(recon_[0].at(x0 + xx, y0 + yy)) -
                      static_cast<int>(
                          source_.y[static_cast<std::size_t>(y0 + yy) *
                                        static_cast<std::size_t>(
                                            source_.width) +
                                    static_cast<std::size_t>(x0 + xx)]);
        sse += d * d;
      }
    }
  }

  syntax::BlockSyntax& entry = sink.blocks[index];
  entry.is_inter = 1;
  entry.skip_flag = 1;
  entry.tip_mode = 1;
  entry.tip_pred_mode = 1;  // NEWMV
  entry.inter_mode = syntax::kInterNewMv;
  entry.drl_index = 0;
  entry.ref_index = 0;
  entry.ref_index1 = 1;
  entry.compound = 0;
  entry.motion_mode = 0;
  entry.use_bawp = 0;
  entry.explicit_bawp = 0;
  entry.explicit_bawp_scale = 0;
  entry.use_bawp_chroma = 0;
  entry.mv_y = tip_leaf_mv_.y;
  entry.mv_x = tip_leaf_mv_.x;
  entry.mv1_y = 0;
  entry.mv1_x = 0;
  entry.mvd_y = tip_leaf_mv_.y;
  entry.mvd_x = tip_leaf_mv_.x;
  entry.use_amvd = 0;
  entry.block_mv_precision = -1;
  entry.mvd_sign_derived = 0;
  // A TIP leaf codes no residual; keep the sink-local "no levels" marker so
  // merge_sinks() maps it to the writer's 0 rather than to a rebased offset.
  entry.levels_offset = kNoLevels;
  entry.tx_part_count = 0;
  entry.cctx_type = 0;
  for (int plane = 0; plane < 3; ++plane) {
    entry.eob[plane] = 0;
  }

  // §7.22 field: the decoder stores the per-8x8 get_tip_mv() pair with
  // RefFrame[0]/[1] = the TIP pair, so mirror it in the MI grid.
  const int units_x = (w + unit - 1) / unit;
  for (int yy = 0; yy < h4 && r + yy < mi_rows_; ++yy) {
    for (int xx = 0; xx < w4 && c + xx < mi_cols_; ++xx) {
      const int ux = (xx * 4) / unit;
      const int uy = (yy * 4) / unit;
      const temporal::MvPair& mvs =
          unit_mvs[static_cast<std::size_t>(uy) *
                       static_cast<std::size_t>(units_x) +
                   static_cast<std::size_t>(ux)];
      inter::mc::MiInfo& mi =
          mv_grid_[static_cast<std::size_t>(r + yy) * static_cast<std::size_t>(mi_cols_) +
                   static_cast<std::size_t>(c + xx)];
      mi = inter::mc::MiInfo{};
      mi.ref[0] = temporal_ctx_.closest_past >= 0
                      ? static_cast<std::int8_t>(temporal_ctx_.closest_past)
                      : merge::kRefNone;
      mi.ref[1] = temporal_ctx_.closest_future >= 0
                      ? static_cast<std::int8_t>(temporal_ctx_.closest_future)
                      : merge::kRefNone;
      mi.mv[0][0] = mvs.mv[0].row;
      mi.mv[0][1] = mvs.mv[0].col;
      mi.mv[1][0] = mvs.mv[1].row;
      mi.mv[1][1] = mvs.mv[1].col;
      mi.is_inter = 1;
      mi.w4 = static_cast<std::int8_t>(w4);
      mi.h4 = static_cast<std::int8_t>(h4);
      mi.col_base = static_cast<std::int16_t>(c);
      mi.row_base = static_cast<std::int16_t>(r);
      mi.bsize = static_cast<std::int16_t>(node.size);
    }
  }
  set_intra_joint(r, c, w4, h4, 0);  // inter_block_mode_info(): DC
  // §6.19.2.3: the leaf is now decoded for the availability of later leaves.
  st.decoded[0].fill(c, r, w4, h4);
  st.decoded[1].fill(c / 2, r / 2, w4 / 2, h4 / 2);
  st.decoded[2].fill(c / 2, r / 2, w4 / 2, h4 / 2);
  ++st.inter_blocks;
  ++st.tip_blocks;
  if (trace_) {
    const temporal::MvPair first = unit_mvs.empty() ? temporal::MvPair{} : unit_mvs.front();
    std::cerr << "m2 tip_block r=" << r << " c=" << c << " " << w << "x" << h
              << " ref=TIP pred=NEWMV drl=0 mv=(" << tip_leaf_mv_.y << "," << tip_leaf_mv_.x
              << ") unit=" << unit << " tipmv0=(" << first.mv[0].row << "," << first.mv[0].col
              << ")/(" << first.mv[1].row << "," << first.mv[1].col
              << ") weight=" << temporal::kTipWeightingFactor[tip.tip_global_wtd_index]
              << " holefill=" << (tip.allow_tip_hole_fill ? 1 : 0) << " sse=" << sse << "\n";
  }
  return true;
}

bool InterFrameEncoder::apply_tip_block_ref(std::string& error) {
  if (!tip_block_ref_active()) return true;
  if (blocks_.empty()) {
    error = "TIP block reference: frame has no coded leaves";
    return false;
  }
  if (!locate_tip_leaf(error)) return false;
  if (tip_leaf_r_ < 0 || tip_leaf_c_ < 0) {
    error = "TIP block reference: no TIP-eligible trailing inter leaf";
    return false;
  }
  tip_blocks_ = 0;
  tip_leaf_armed_ = true;
  if (!decide(error)) {
    tip_leaf_armed_ = false;
    return false;
  }
  tip_leaf_armed_ = false;
  if (tip_blocks_ != 1) {
    error =
        "TIP block reference: the second decision pass coded no TIP "
        "leaf";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// AV2ENC_RD_DRIVER: sequential key-frame driver
// ---------------------------------------------------------------------------
namespace {

struct SeqChildSlot {
  int mi_row = 0;
  int mi_col = 0;
  partition::BlockSize size = partition::BlockSize::kInvalid;
  bool has_chroma = false;
  bool chroma_offset = false;
};

// Decode-order children of one partition candidate.  Mirrors the spec-only
// rules of codec/partition/tree/partition.cpp (own tree, not third-party
// code): SPLIT is four children with chromaOffset = 0, HORZ/VERT are two and
// hand the ChromaMi offset to the first child only.
int seq_child_slots(const partition::FrameGeometry& geometry, int mi_row,
                    int mi_col, partition::BlockSize size,
                    partition::PartitionType partition, bool has_chroma,
                    bool chroma_offset, SeqChildSlot slots[4]) {
  if (partition == partition::PartitionType::kNone) return 0;
  const partition::BlockSize subsize =
      partition::partition_subsize(partition, size);
  if (!partition::is_valid_block_size(subsize)) return 0;
  const int w4 = partition::num_4x4_blocks_wide(size);
  const int h4 = partition::num_4x4_blocks_high(size);
  const int half_w = w4 >> 1;
  const int half_h = h4 >> 1;
  bool next_offset = chroma_offset;
  if (has_chroma && !next_offset) {
    next_offset =
        partition::is_chroma_offset_for_partition(geometry, partition, size);
  }
  switch (partition) {
    case partition::PartitionType::kSplit:
      slots[0] = {mi_row, mi_col, subsize, has_chroma, false};
      slots[1] = {mi_row, mi_col + half_w, subsize, has_chroma, false};
      slots[2] = {mi_row + half_h, mi_col, subsize, has_chroma, false};
      slots[3] = {mi_row + half_h, mi_col + half_w, subsize, has_chroma,
                  false};
      return 4;
    case partition::PartitionType::kHorz:
      slots[0] = {mi_row, mi_col, subsize, has_chroma && !next_offset,
                  next_offset};
      slots[1] = {mi_row + half_h, mi_col, subsize, has_chroma, next_offset};
      return 2;
    case partition::PartitionType::kVert:
      slots[0] = {mi_row, mi_col, subsize, has_chroma && !next_offset,
                  next_offset};
      slots[1] = {mi_row, mi_col + half_w, subsize, has_chroma, next_offset};
      return 2;
    case partition::PartitionType::kNone:
      break;
  }
  return 0;
}

bool seq_child_coded(const partition::FrameGeometry& geometry,
                     const SeqChildSlot& slot) {
  return slot.mi_row < partition::mi_rows(geometry) &&
         slot.mi_col < partition::mi_cols(geometry);
}

// Mean squared high-pass energy of a source block: the same (d - box3x3(d))^2
// term the BA-shaped leaf D uses, but on the *source*, not the error.  It is
// the activity screen for the sub-16 gate: a flat sky does not need 4x4/8x4
// coding blocks just because the measured D says the split is cheap.
double seq_source_activity(const std::uint8_t* src, int width, int height,
                           int stride, int sx, int sy, int bw, int bh) {
  if (src == nullptr || bw <= 0 || bh <= 0 || width <= 0 || height <= 0) {
    return 0.0;
  }
  const auto sample = [&](int x, int y) -> int {
    const int px = std::clamp(sx + x, 0, width - 1);
    const int py = std::clamp(sy + y, 0, height - 1);
    return static_cast<int>(
        src[static_cast<std::size_t>(py) * static_cast<std::size_t>(stride) +
            static_cast<std::size_t>(px)]);
  };
  double sum = 0.0;
  for (int y = 0; y < bh; ++y) {
    for (int x = 0; x < bw; ++x) {
      int low = 0;
      int used = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          low += sample(x + dx, y + dy);
          ++used;
        }
      }
      const double d =
          static_cast<double>(sample(x, y)) -
          static_cast<double>(low) / static_cast<double>(used);
      sum += d * d;
    }
  }
  return sum / static_cast<double>(bw * bh);
}

// Build a uniform layout: split each node along its long axis until the block
// fits `target` in both dimensions.  A square block first becomes two wide
// rectangles, which then split into the two square children, so an 8x8 target
// is reached in two partition depths per halving.  Spec-forced partitions are
// honoured, so a boundary superblock can never produce an invalid tree.
partition::PartitionNode seq_build_uniform(
    const partition::FrameGeometry& geometry, int mi_row, int mi_col,
    partition::BlockSize size, int depth, bool has_chroma, bool chroma_offset,
    int target, int max_depth, int min_px,
    const std::function<bool(int, int, partition::BlockSize)>& can_split) {
  partition::PartitionNode node;
  node.mi_row = mi_row;
  node.mi_col = mi_col;
  node.size = size;
  node.depth = depth;
  node.has_chroma = has_chroma;
  node.chroma_offset = chroma_offset;
  node.boundary = partition::partition_implied(geometry, mi_row, mi_col, size);
  node.allowed = partition::init_allowed_partitions(
      geometry, mi_row, mi_col, size, has_chroma, chroma_offset);
  bool forced = false;
  partition::PartitionType forced_partition =
      partition::PartitionType::kNone;
  if (node.boundary.implied &&
      node.allowed.is_allowed(node.boundary.implied_partition)) {
    forced = true;
    forced_partition = node.boundary.implied_partition;
  } else if (node.allowed.count == 1) {
    forced = true;
    for (int index = 0; index < partition::kPartitionTypeCount; ++index) {
      if (node.allowed.allowed[index]) {
        forced_partition = static_cast<partition::PartitionType>(index);
        break;
      }
    }
  }
  const auto policy = [&](partition::PartitionType partition) {
    if (partition == partition::PartitionType::kNone) return true;
    if (depth + 1 > max_depth) return false;
    const partition::BlockSize sub =
        partition::partition_subsize(partition, size);
    if (!partition::is_valid_block_size(sub)) return false;
    return partition::block_width(sub) >= min_px &&
           partition::block_height(sub) >= min_px;
  };
  partition::PartitionType chosen = partition::PartitionType::kNone;
  if (forced) {
    chosen = forced_partition;
  } else {
    const int bw = partition::block_width(size);
    const int bh = partition::block_height(size);
    if (bw <= target && bh <= target) {
      chosen = partition::PartitionType::kNone;
    } else if (can_split && !can_split(mi_row, mi_col, size)) {
      chosen = partition::PartitionType::kNone;
    } else {
      const partition::PartitionType prefer =
          bw > bh ? partition::PartitionType::kVert
                  : partition::PartitionType::kHorz;
      const partition::PartitionType other =
          prefer == partition::PartitionType::kVert
              ? partition::PartitionType::kHorz
              : partition::PartitionType::kVert;
      if (node.allowed.is_allowed(prefer) && policy(prefer)) {
        chosen = prefer;
      } else if (node.allowed.is_allowed(other) && policy(other)) {
        chosen = other;
      } else if (node.allowed.is_allowed(partition::PartitionType::kNone)) {
        chosen = partition::PartitionType::kNone;
      } else {
        for (int index = 0; index < partition::kPartitionTypeCount; ++index) {
          if (node.allowed.allowed[index] && policy(
                  static_cast<partition::PartitionType>(index))) {
            chosen = static_cast<partition::PartitionType>(index);
            break;
          }
        }
      }
    }
  }
  node.partition = chosen;
  node.forced = forced;
  if (chosen != partition::PartitionType::kNone) {
    SeqChildSlot slots[4];
    const int count = seq_child_slots(geometry, mi_row, mi_col, size, chosen,
                                      has_chroma, chroma_offset, slots);
    for (int i = 0; i < count; ++i) {
      if (!seq_child_coded(geometry, slots[i])) continue;
      node.children.push_back(seq_build_uniform(
          geometry, slots[i].mi_row, slots[i].mi_col, slots[i].size,
          depth + 1, slots[i].has_chroma, slots[i].chroma_offset, target,
          max_depth, min_px, can_split));
    }
    if (node.children.empty()) {
      // A policy/geometry disagreement must not leave a childless split.
      node.partition = partition::PartitionType::kNone;
    }
  }
  return node;
}

int seq_depth_for_target(int target) {
  int depth = 0;
  int size = 64;
  while (size > target && depth < 16) {
    size >>= 1;
    depth += 2;
  }
  return depth;
}

// Exact §8.3 information rate of one emitted leaf.  The entry carries the
// writer's levels in scan order; `rd_engine::plan_rate()` prices exactly those
// levels with the §9.3 static model, and `apply_calibration()` maps the model
// onto the live §8.3 cost (docs/rd-lane1.md §3).  A block the model cannot
// price contributes no bits, which only under-prices an unsupported TX class
// (IDTX is off in this recipe).
double seq_entry_rate(const syntax::BlockSyntax& entry,
                      const std::vector<std::int32_t>& levels, int qctx) {
  std::vector<std::vector<std::int32_t>> storage;
  storage.reserve(8);
  std::vector<rd_engine::BlockPlan> blocks;
  blocks.reserve(8);
  const auto add = [&](int plane, int tx_size, int tx_type, int eob,
                       std::size_t offset) {
    const int area = txpartition::tx_scan_area(tx_size);
    if (area <= 0 || eob <= 0) return;
    storage.emplace_back(static_cast<std::size_t>(area), 0);
    std::vector<std::int32_t>& scan_levels = storage.back();
    for (int i = 0; i < eob && i < area; ++i) {
      if (offset + static_cast<std::size_t>(i) < levels.size()) {
        scan_levels[static_cast<std::size_t>(i)] =
            levels[offset + static_cast<std::size_t>(i)];
      }
    }
    rd_engine::BlockPlan block;
    block.tx_size = tx_size;
    block.tx_type = tx_type;
    block.plane = plane;
    block.area = area;
    block.levels_scan = scan_levels.data();
    blocks.push_back(block);
  };
  if (entry.tx_part_count > 0) {
    const int count = std::min<int>(entry.tx_part_count, 4);
    for (int i = 0; i < count; ++i) {
      add(0, entry.tx_part_tx_size[i], entry.tx_part_tx_type[i],
          static_cast<int>(entry.tx_part_eob[i]),
          entry.tx_part_levels_offset[i]);
    }
  } else {
    add(0, entry.tx_size[0], entry.tx_type[0],
        static_cast<int>(entry.eob[0]), entry.levels_offset);
  }
  std::size_t chroma_offset =
      entry.levels_offset + static_cast<std::size_t>(entry.eob[0]);
  add(1, entry.tx_size[1], entry.tx_type[1],
      static_cast<int>(entry.eob[1]), chroma_offset);
  chroma_offset += static_cast<std::size_t>(entry.eob[1]);
  add(2, entry.tx_size[2], entry.tx_type[2],
      static_cast<int>(entry.eob[2]), chroma_offset);
  rd_engine::RateBreakdown rate =
      rd_engine::plan_rate(blocks, /*is_inter=*/false, qctx,
                           /*cdf=*/nullptr);
  rd_engine::apply_calibration(&rate);
  return rate.total();
}

}  // namespace

bool InterFrameEncoder::seq_driver_measure_leaf(const PartitionNode& node,
                                                const ChromaInfo& chroma,
                                                SbSink& sink, double* cost,
                                                std::string& error) {
  if (cost == nullptr || node.partition != PartitionType::kNone) return false;
  const int w = partition::block_width(node.size);
  const int h = partition::block_height(node.size);
  const int x0 = node.mi_col * 4;
  const int y0 = node.mi_row * 4;
  const int ax0 = x0 + tile_x_;
  const int ay0 = y0 + tile_y_;
  const bool owns_chroma = node.has_chroma;
  int cw = w / 2;
  int ch = h / 2;
  int cx0 = x0 / 2;
  int cy0 = y0 / 2;
  if (chroma.valid) {
    const partition::BlockSize residual =
        partition::chroma_plane_residual_size(geometry_, chroma.size);
    cw = partition::block_width(residual);
    ch = partition::block_height(residual);
    cx0 = (chroma.mi_col * 4) / 2;
    cy0 = (chroma.mi_row * 4) / 2;
  }
  const int acx0 = cx0 + tile_x_ / 2;
  const int acy0 = cy0 + tile_y_ / 2;
  const int luma_mask = visible_rows(0, y0, h);
  const int chroma_mask = visible_rows(1, cy0, ch);

  const std::size_t index = sink.blocks.size();
  sink.blocks.emplace_back();
  syntax::BlockSyntax& entry = sink.blocks.back();
  entry.levels_offset = kNoLevels;
  entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
  entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
  entry.bsize = static_cast<std::uint8_t>(node.size);
  entry.partition = static_cast<std::uint8_t>(node.partition);
  entry.flags |= syntax::kFlagHasChromaInfo;
  entry.has_chroma = node.has_chroma ? 1 : 0;
  entry.chroma_offset = node.chroma_offset ? 1 : 0;
  entry.chroma_bsize =
      chroma.valid ? static_cast<std::uint8_t>(chroma.size) : 0;
  entry.chroma_mi_row = static_cast<std::uint16_t>(chroma.mi_row);
  entry.chroma_mi_col = static_cast<std::uint16_t>(chroma.mi_col);
  if (!encode_leaf(sink, node, index, error, /*tree_type=*/0,
                   node.has_chroma, chroma.mi_row, chroma.mi_col,
                   chroma.valid ? static_cast<int>(chroma.size) : -1)) {
    return false;
  }
  const syntax::BlockSyntax& emitted = sink.blocks[index];
  const double key_csf =
      eff_perc::enabled() ? eff_perc::key_csf_gain() : 0.0;
  const double key_csf2 =
      eff_perc::enabled() ? eff_perc::csf2_gain() : 0.0;
  const double chroma_csf = eff_perc::enabled() ? eff_perc::csf_gain() : 0.0;
  const double chroma_csf2 = eff_perc::enabled() ? eff_perc::csf2_gain() : 0.0;
  const double chroma_weight =
      eff_perc::enabled() ? eff_perc::chroma_weight() : 1.0;
  // BA-shaped luma D: eff-perc key low-band distortion plus the high-pass and
  // block-edge terms from `seq_rd::leaf_distortion()`, measured on this leaf's
  // own reconstruction.  The low-band key CSF alone is blind to the error a
  // partition boundary leaves behind, which is the term Butteraugli charges.
  double distortion = seq_rd::leaf_distortion(
      source_.y.data(), static_cast<int>(source_.width),
      static_cast<int>(source_.height), static_cast<int>(source_.width), ax0,
      ay0, w, h, luma_mask, recon_[0].samples.data(), recon_[0].stride,
      recon_[0].height, x0, y0, key_csf, key_csf2);
  if (owns_chroma) {
    const double u = eff_perc::weighted_sse(
        /*weights=*/nullptr, cw, ch, source_.u.data(),
        static_cast<int>(source_.width) / 2, acx0, acy0,
        recon_[1].samples.data(), recon_[1].stride, cx0, cy0, chroma_mask,
        chroma_csf, chroma_csf2);
    const double v = eff_perc::weighted_sse(
        /*weights=*/nullptr, cw, ch, source_.v.data(),
        static_cast<int>(source_.width) / 2, acx0, acy0,
        recon_[2].samples.data(), recon_[2].stride, cx0, cy0, chroma_mask,
        chroma_csf, chroma_csf2);
    distortion += (u + v) * chroma_weight;
  }
  const int qctx = ::av2enc::still::qctx::slice_for_base_q_idx(
      options_.tx_context_rate ? base_q_index() : current_qindex_);
  double rate = seq_entry_rate(emitted, sink.levels, qctx);
  rate += intra_luma_syntax_bits(emitted.y_mode, emitted.y_angle_delta,
                                 emitted.mrl_index,
                                 emitted.mrl_sec_index != 0,
                                 emitted.use_dip != 0);
  rate += intra_chroma_syntax_bits(
      (emitted.flags & syntax::kFlagIsCfl) != 0, emitted.cctx_type != 0);
  if (emitted.tx_part_count > 0) rate += 2.0;
  const double lambda =
      key_tx_rd_lambda(options_, current_qindex_) *
      rd_driver::env_positive("AV2ENC_RD_KEY_TX_SCALE", 0.25) *
      rd_driver::env_positive("AV2ENC_RD_DRIVER_LAMBDA", 1.0) *
      sb_alloc_scale_;
  *cost = distortion + lambda * rate;
  ++rd_driver::counters().measured_leaves;
  return true;
}

bool InterFrameEncoder::seq_driver_decide_sb(std::size_t sb_index,
                                             const PartitionNode& reference,
                                             PartitionNode* out,
                                             std::string& error) {
  if (out == nullptr) return false;
  rd_driver::Counters& stats = rd_driver::counters();
  ++stats.superblocks;

  // AV2ENC_RD_DRIVER_SBALLOC (default 0 = off): content-adaptive per-superblock
  // rate allocation on the key tree.  The driver's tree lambda is scaled by
  // this superblock's own 1-px high-pass energy relative to the frame mean, so
  // a superblock whose source carries detail in the band SSIMULACRA2 charges
  // for (docs/band-d.md: local SS2 tracks the 1-px error at r = -0.82) is
  // priced cheaper per bit and spends more rate, while a smooth superblock is
  // priced dearer and coarsens.  strength 0 (the default) is exactly 1.0 for
  // every superblock, so unset is byte-identical.
  sb_alloc_scale_ = 1.0;
  if (std::getenv("AV2ENC_RD_DRIVER_SBALLOC") != nullptr) {
    const int cols = qp_sb_cols();
    const int mi = qp_sb_mi();
    const int sb = static_cast<int>(sb_index);
    const int sx = (cols > 0 ? sb % cols : 0) * mi * 4 + tile_x_;
    const int sy = (cols > 0 ? sb / cols : 0) * mi * 4 + tile_y_;
    const int n = mi * 4;
    const double act = seq_source_activity(
        source_.y.data(), static_cast<int>(source_.width),
        static_cast<int>(source_.height), static_cast<int>(source_.width), sx,
        sy, n, n);
    sb_alloc_frame_activity_ += act;
    ++sb_alloc_sbs_;
    const double mean = sb_alloc_frame_activity_ /
                        static_cast<double>(std::max<long long>(sb_alloc_sbs_, 1));
    if (mean > 0.0) {
      // Signed read: a negative strength is the control arm (spend where the
      // source is *smooth*), which distinguishes "mis-allocated" from
      // "allocation does nothing".
      const double strength = [] {
        const char* v = std::getenv("AV2ENC_RD_DRIVER_SBALLOC");
        if (v == nullptr || *v == '\0') return 0.0;
        char* end = nullptr;
        const double parsed = std::strtod(v, &end);
        if (end == v || !std::isfinite(parsed)) return 0.0;
        return parsed;
      }();
      const double cap =
          rd_driver::env_positive("AV2ENC_RD_DRIVER_SBALLOC_MAX", 4.0);
      if (strength != 0.0) {
        sb_alloc_scale_ =
            std::clamp(std::pow(std::max(act, 1e-6) / mean, -strength),
                       1.0 / cap, cap);
      }
    }
  }

  // Expand the historical layout to the driver's deepest candidate leaf so
  // every collapse point is available to the prune.  Every split in the
  // expanded tree is a real §5.20.3.2 partition for the geometry.
  const int target = rd_driver::prune_target();
  const int max_depth = std::max(options_.max_partition_depth,
                                 seq_depth_for_target(target));
  // High-activity gate for sub-16 leaves (AV2ENC_RD_DRIVER_ACT, default 0 =
  // off).  A 16x16 or smaller node may only descend to 8x8/4x4 when the source
  // high-pass energy clears the threshold, so flat regions keep the larger
  // coding block even when the measured D would call the split cheap.
  const double activity_threshold =
      rd_driver::env_positive("AV2ENC_RD_DRIVER_ACT", 0.0);
  const auto can_split = [&](int mi_row, int mi_col,
                             partition::BlockSize size) {
    if (activity_threshold <= 0.0) return true;
    const int bw = partition::block_width(size);
    const int bh = partition::block_height(size);
    if (bw > 16 || bh > 16) return true;
    const int sx = mi_col * 4 + tile_x_;
    const int sy = mi_row * 4 + tile_y_;
    return seq_source_activity(source_.y.data(),
                               static_cast<int>(source_.width),
                               static_cast<int>(source_.height),
                               static_cast<int>(source_.width), sx, sy, bw,
                               bh) >= activity_threshold;
  };
  std::function<PartitionNode(const PartitionNode&, const ChromaInfo&)> expand;
  expand = [&](const PartitionNode& node,
               const ChromaInfo& inherited) -> PartitionNode {
    const ChromaInfo chroma = chroma_step(node, inherited);
    if (node.partition == PartitionType::kNone) {
      const int bw = partition::block_width(node.size);
      const int bh = partition::block_height(node.size);
      if (bw <= target && bh <= target) return node;
      return seq_build_uniform(geometry_, node.mi_row, node.mi_col, node.size,
                               node.depth, node.has_chroma, node.chroma_offset,
                               target, max_depth, partition::kMinBlockPixels,
                               can_split);
    }
    PartitionNode grown = node;
    grown.children.clear();
    for (const PartitionNode& child : node.children) {
      grown.children.push_back(expand(child, chroma));
    }
    return grown;
  };
  PartitionNode expanded = expand(reference, ChromaInfo{});
  std::string validate_error;
  if (!partition::validate_tree(geometry_, expanded, validate_error)) {
    ++stats.layout_failures;
    ++stats.fallback_reference;
    *out = reference;
    return true;
  }

  // Snapshot every surface the leaf coder mutates.  The prune runs entirely on
  // a scratch sink and always restores the frame state before returning, so
  // the committed pass in collect_frame() is the only writer of the stream.
  struct Snap {
    std::array<std::vector<std::uint8_t>, 3> samples;
    std::array<int, 3> stride{};
    std::array<int, 3> width{};
    std::array<int, 3> height{};
    std::vector<std::uint8_t> joint;
    std::vector<std::uint8_t> uv;
    std::vector<inter::mc::MiInfo> mv;
    SbSink::State state;
    std::size_t blocks = 0;
    std::size_t levels = 0;
    std::size_t palettes = 0;
    std::size_t sub_pus = 0;
    std::size_t leaf_dump = 0;
    std::size_t leaf_costs = 0;
  };
  SbSink sink;
  sink.reset();
  seed_unit_state(sink);
  sink.state.decoded[0].reset(mi_cols_, mi_rows_);
  sink.state.decoded[1].reset(mi_cols_ / 2, mi_rows_ / 2);
  sink.state.decoded[2].reset(mi_cols_ / 2, mi_rows_ / 2);
  const auto capture = [&]() {
    Snap snap;
    for (int plane = 0; plane < 3; ++plane) {
      const std::size_t p = static_cast<std::size_t>(plane);
      snap.samples[p] = recon_[p].samples;
      snap.stride[p] = recon_[p].stride;
      snap.width[p] = recon_[p].width;
      snap.height[p] = recon_[p].height;
    }
    snap.joint = intra_joint_;
    snap.uv = uv_smooth_;
    snap.mv = mv_grid_;
    snap.state = sink.state;
    snap.blocks = sink.blocks.size();
    snap.levels = sink.levels.size();
    snap.palettes = sink.palettes.size();
    snap.sub_pus = sink.sub_pus.size();
    snap.leaf_dump = sink.leaf_dump.size();
    snap.leaf_costs = sink.leaf_costs.size();
    return snap;
  };
  const auto apply = [&](const Snap& snap) {
    for (int plane = 0; plane < 3; ++plane) {
      const std::size_t p = static_cast<std::size_t>(plane);
      recon_[p].samples = snap.samples[p];
      recon_[p].stride = snap.stride[p];
      recon_[p].width = snap.width[p];
      recon_[p].height = snap.height[p];
    }
    intra_joint_ = snap.joint;
    uv_smooth_ = snap.uv;
    mv_grid_ = snap.mv;
    sink.state = snap.state;
    sink.blocks.resize(snap.blocks);
    sink.levels.resize(snap.levels);
    sink.palettes.resize(snap.palettes);
    sink.sub_pus.resize(snap.sub_pus);
    sink.leaf_dump.resize(snap.leaf_dump);
    sink.leaf_costs.resize(snap.leaf_costs);
  };

  struct PruneResult {
    PartitionNode node;
    double cost = partition::kInfiniteCost;
    bool ok = false;
  };
  const double lambda =
      key_tx_rd_lambda(options_, current_qindex_) *
      rd_driver::env_positive("AV2ENC_RD_KEY_TX_SCALE", 0.25) *
      rd_driver::env_positive("AV2ENC_RD_DRIVER_LAMBDA", 1.0) *
      sb_alloc_scale_;
  // Sequential prune: at every node the split children are measured in the
  // reconstruction of the already-chosen preceding children, so a candidate
  // is always priced in the recon it builds.  The NONE trial is restored
  // before the split trial; the winner is re-encoded to keep the state
  // coherent for the following sibling.
  std::function<PruneResult(const PartitionNode&, const ChromaInfo&)> prune;
  prune = [&](const PartitionNode& node,
              const ChromaInfo& inherited) -> PruneResult {
    PruneResult result;
    const ChromaInfo chroma = chroma_step(node, inherited);
    if (node.partition == PartitionType::kNone) {
      double cost = 0.0;
      if (!seq_driver_measure_leaf(node, chroma, sink, &cost, error)) {
        result.node = node;
        return result;
      }
      result.node = node;
      result.cost = cost;
      result.ok = true;
      return result;
    }
    const Snap start = capture();
    // A spec-forced split has no NONE candidate: the decoder derives the
    // partition without syntax, so collapsing it would be an invalid tree.
    const bool can_leaf =
        !node.forced && node.allowed.is_allowed(PartitionType::kNone);
    PartitionNode leaf = node;
    leaf.partition = PartitionType::kNone;
    leaf.forced = false;
    leaf.children.clear();
    leaf.cost = 0.0;
    double leaf_cost = partition::kInfiniteCost;
    bool leaf_ok = false;
    if (can_leaf) {
      leaf_ok = seq_driver_measure_leaf(leaf, chroma, sink, &leaf_cost, error);
    }
    apply(start);

    double split_cost = lambda * 2.0;
    bool split_ok = true;
    std::vector<PartitionNode> kids;
    for (const PartitionNode& child : node.children) {
      PruneResult child_result = prune(child, chroma);
      if (!child_result.ok) {
        split_ok = false;
        break;
      }
      split_cost += child_result.cost;
      kids.push_back(std::move(child_result.node));
    }
    if (split_ok && leaf_ok && leaf_cost <= split_cost) {
      // Collapse: drop the measured children and commit the leaf instead.
      apply(start);
      double committed = 0.0;
      if (!seq_driver_measure_leaf(leaf, chroma, sink, &committed, error)) {
        result.node = node;
        return result;
      }
      result.node = std::move(leaf);
      result.cost = committed;
      result.ok = true;
      return result;
    }
    if (split_ok) {
      PartitionNode chosen = node;
      chosen.children = std::move(kids);
      chosen.cost = split_cost;
      result.node = std::move(chosen);
      result.cost = split_cost;
      result.ok = true;
      return result;
    }
    // The split could not be measured.  Keep the leaf when it could.
    apply(start);
    if (!leaf_ok) {
      result.node = node;
      return result;
    }
    double committed = 0.0;
    if (!seq_driver_measure_leaf(leaf, chroma, sink, &committed, error)) {
      result.node = node;
      return result;
    }
    result.node = std::move(leaf);
    result.cost = committed;
    result.ok = true;
    return result;
  };

  const Snap frame_start = capture();
  ++stats.layouts;
  rd_driver::measuring_flag() = true;
  PruneResult chosen = prune(expanded, ChromaInfo{});
  rd_driver::measuring_flag() = false;
  // Discard the scratch reconstruction: collect_node() is the commit pass and
  // the only writer of the emitted syntax.
  apply(frame_start);
  if (!chosen.ok || !std::isfinite(chosen.cost)) {
    ++stats.fallback_reference;
    *out = reference;
    return true;
  }
  std::string chosen_error;
  if (!partition::validate_tree(geometry_, chosen.node, chosen_error)) {
    ++stats.layout_failures;
    ++stats.fallback_reference;
    *out = reference;
    return true;
  }
  stats.chosen_cost += chosen.cost;
  *out = std::move(chosen.node);
  return true;
}

// ---------------------------------------------------------------------------
// AV2ENC_RD_TAIL_FLOOR: measured P-frame coding-block floor (tail pictures)
//
// The reference tree the measured partition probe solves is bounded to depth 4,
// so a 16x16 node is the coding-block floor on this clip (docs/rd-tail.md §7).
// This stage decides the one level the reference tree cannot reach, with the
// same objective the leaf and the TX search use:
//
//   screen  the existing per-block partition probe prices NONE against the
//           legal HORZ/VERT layouts and their 8x8-class descent.  A node whose
//           probe prefers NONE never reaches the exact stage.
//   exact   both layouts are encoded with encode_leaf() into the live
//           reconstruction, in raster order and after every earlier
//           superblock was committed, and scored with
//             J = D_BA(recon the leaf emitted) + lambda * (R_exact + 2*part)
//
// A candidate that cannot be encoded or priced keeps the reference 16x16 leaf,
// so the gate can never make a node unsolvable.
// ---------------------------------------------------------------------------
namespace {

struct TailFloorScreen {
  partition::BlockSize size = partition::BlockSize::kInvalid;
  int mi_row = 0;
  int mi_col = 0;
  int level = 0;
  bool has_chroma = false;
  bool chroma_offset = false;
  partition::PartitionType partition = partition::PartitionType::kNone;
  std::vector<TailFloorScreen> children;
  double leaf_cost = 0.0;  // probe cost of this node as one coding block
  double cost = 0.0;       // probe cost of the chosen layout
  bool ok = false;
};

int tail_floor_leaves(const partition::PartitionNode& node, int* n16,
                      int* n8, int* n4) {
  if (node.partition != partition::PartitionType::kNone) {
    int count = 0;
    for (const partition::PartitionNode& child : node.children) {
      count += tail_floor_leaves(child, n16, n8, n4);
    }
    return count;
  }
  const int w = partition::block_width(node.size);
  const int h = partition::block_height(node.size);
  const int side = w < h ? w : h;
  if (n16 != nullptr) {
    if (side >= 16) ++*n16;
    else if (side >= 8) ++*n8;
    else ++*n4;
  }
  return 1;
}

// Replace the node (mi_row, mi_col, size) inside a copy of the superblock tree
// with `replacement`.  Used to validate a candidate layout against the real
// §5.20.3 tree rules before a single symbol is written.
bool tail_floor_replace(partition::PartitionNode& node, int mi_row, int mi_col,
                        partition::BlockSize size,
                        const partition::PartitionNode& replacement,
                        bool* done) {
  if (*done) return true;
  if (node.mi_row == mi_row && node.mi_col == mi_col && node.size == size) {
    node = replacement;
    *done = true;
    return true;
  }
  for (partition::PartitionNode& child : node.children) {
    if (!tail_floor_replace(child, mi_row, mi_col, size, replacement, done)) {
      return false;
    }
    if (*done) return true;
  }
  return true;
}

}  // namespace

bool InterFrameEncoder::tail_floor_candidate(const PartitionNode& node) const {
  if (node.partition != partition::PartitionType::kNone) return false;
  if (node.size != partition::BlockSize::k16x16) return false;
  if (node.forced) return false;
  if (tail_floor::config().min_px > 8) return false;
  return node.allowed.is_allowed(partition::PartitionType::kHorz) ||
         node.allowed.is_allowed(partition::PartitionType::kVert);
}

bool InterFrameEncoder::tail_floor_leaf_cost(SbSink& sink, std::size_t index,
                                             double lambda, double hf,
                                             double edge, double* d_out,
                                             double* r_out,
                                             std::string& error) {
  (void)lambda;
  (void)error;
  if (d_out == nullptr || r_out == nullptr) return false;
  if (index >= sink.blocks.size()) {
    if (joint_p_active_) ++joint_plan::counters().fail_entry;
    return false;
  }
  if (index >= tail_floor_leaf_bits_ok_.size() ||
      tail_floor_leaf_bits_ok_[index] == 0) {
    if (joint_p_active_) ++joint_plan::counters().fail_entry;
    return false;
  }
  const syntax::BlockSyntax& entry = sink.blocks[index];
  if (entry.partition !=
          static_cast<std::uint8_t>(partition::PartitionType::kNone) ||
      entry.subtree != 0) {
    if (joint_p_active_) ++joint_plan::counters().fail_entry;
    return false;
  }
  const auto size = static_cast<partition::BlockSize>(entry.bsize);
  const int w = partition::block_width(size);
  const int h = partition::block_height(size);
  if (w <= 0 || h <= 0) return false;
  const int x0 = entry.mi_col * 4;
  const int y0 = entry.mi_row * 4;
  const int ax0 = x0 + tile_x_;
  const int ay0 = y0 + tile_y_;
  const int rows = visible_rows(0, y0, h);
  // One D for every candidate: the eff-perc inter low-band CSF (the gain the
  // leaf and its TX search minimise) with the driver's optional high-pass and
  // block-edge terms, measured on the reconstruction this layout built.
  const double csf = pipe_ba::enabled()
                         ? pipe_ba::config().low_band_gain
                         : (eff_perc::enabled() ? eff_perc::csf_gain() : 0.0);
  const double csf2 =
      pipe_ba::enabled()
          ? 0.0
          : (eff_perc::enabled() ? eff_perc::csf2_gain() : 0.0);
  double d = 0.0;
  {
  joint_plan::ScopedTimer d_timer(&joint_plan::profile().cost_d);
  d = seq_rd::leaf_distortion_gains(
      source_.y.data(), static_cast<int>(source_.width),
      static_cast<int>(source_.height), static_cast<int>(source_.width), ax0,
      ay0, w, h, rows, recon_[0].samples.data(), recon_[0].stride,
      recon_[0].height, x0, y0, csf, csf2, hf, edge);
  if (entry.has_chroma != 0 && entry.chroma_bsize != 0) {
    const auto chroma_size =
        static_cast<partition::BlockSize>(entry.chroma_bsize);
    const partition::BlockSize residual =
        partition::chroma_plane_residual_size(geometry_, chroma_size);
    const int cw = partition::block_width(residual);
    const int ch = partition::block_height(residual);
    const int cx0 = (entry.chroma_mi_col * 4) / 2;
    const int cy0 = (entry.chroma_mi_row * 4) / 2;
    const int acx0 = cx0 + tile_x_ / 2;
    const int acy0 = cy0 + tile_y_ / 2;
    const int crow = visible_rows(1, cy0, ch);
    const double chroma_csf =
        eff_perc::enabled() ? eff_perc::csf_gain() : 0.0;
    const double chroma_csf2 =
        eff_perc::enabled() ? eff_perc::csf2_gain() : 0.0;
    const double u = eff_perc::weighted_sse(
        /*weights=*/nullptr, cw, ch, source_.u.data(),
        static_cast<int>(source_.width) / 2, acx0, acy0,
        recon_[1].samples.data(), recon_[1].stride, cx0, cy0, crow, chroma_csf,
        chroma_csf2);
    const double v = eff_perc::weighted_sse(
        /*weights=*/nullptr, cw, ch, source_.v.data(),
        static_cast<int>(source_.width) / 2, acx0, acy0,
        recon_[2].samples.data(), recon_[2].stride, cx0, cy0, crow, chroma_csf,
        chroma_csf2);
    const double chroma_weight =
        eff_perc::enabled() ? eff_perc::chroma_weight() : 1.0;
    d += (u + v) * chroma_weight;
  }
  }

  // Exact residual rate: the emitted §8.3 symbol sequence through the live
  // §8.2.6 interval model against the frame-start CDF image (the same call the
  // leaf's own plan decision uses), plus the live inter-syntax group the probe
  // stashed as this leaf was written.
  double r = 0.0;
  const bool skip = entry.is_inter != 0 && entry.skip_flag != 0;
  if (!skip) {
    if (options_.enable_tcq || entry.tree_type != 0) {
      if (joint_p_active_) ++joint_plan::counters().fail_entry;
      return false;
    }
    rate_interval::LeafRateInput in;
    rate_interval::LeafLevelStorage storage;
    {
      joint_plan::ScopedTimer build_timer(&joint_plan::profile().cost_build);
      if (!rate_interval::build_from_syntax(
              entry, sink.levels, options_.enable_tx_search,
              options_.adaptive_cdf, static_cast<int>(options_.base_q_idx), &in,
              &storage, /*mode_less_inter=*/true)) {
        if (joint_p_active_) ++joint_plan::counters().fail_build;
        return false;
      }
    }
    const rate_interval::Mode mode =
        (rate_interval::mode() == rate_interval::Mode::kOff &&
         eff_perc::enabled() && (eff_perc::rate_mode() & 1) != 0)
            ? rate_interval::Mode::kInterval
            : rate_interval::mode();
    if (mode == rate_interval::Mode::kOff) {
      if (joint_p_active_) ++joint_plan::counters().fail_entry;
      return false;
    }
    rate_interval::LeafRate rate;
    {
      joint_plan::ScopedTimer estimate_timer(&joint_plan::profile().cost_estimate);
      rate = rate_interval::estimate(in, rate_interval_cdfs_, mode);
    }
    if (rate.failed ||
        !(rate.residual_bits < rate_interval::detail::kImpossibleBits)) {
      if (joint_p_active_) ++joint_plan::counters().fail_estimate;
      return false;
    }
    r += rate.residual_bits;
  }
  r += tail_floor_leaf_bits_[index];
  if (entry.is_inter == 0) {
    r += intra_luma_syntax_bits(entry.y_mode, entry.y_angle_delta,
                                entry.mrl_index, entry.mrl_sec_index != 0,
                                entry.use_dip != 0);
    r += intra_chroma_syntax_bits((entry.flags & syntax::kFlagIsCfl) != 0,
                                  entry.cctx_type != 0);
    if (entry.tx_part_count > 0) r += 2.0;
  }
  *d_out = d;
  *r_out = r;
  return true;
}

bool InterFrameEncoder::tail_floor_trial_cost(SbSink& sink, std::size_t begin,
                                              double lambda, int splits,
                                              double* d_out, double* r_out,
                                              std::string& error) {
  if (d_out == nullptr || r_out == nullptr) return false;
  double d_sum = 0.0;
  double r_sum = 0.0;
  for (std::size_t i = begin; i < sink.blocks.size(); ++i) {
    const syntax::BlockSyntax& entry = sink.blocks[i];
    if (entry.partition !=
            static_cast<std::uint8_t>(partition::PartitionType::kNone) ||
        entry.subtree != 0) {
      continue;  // internal node: no leaf cost
    }
    double d = 0.0;
    double r = 0.0;
    if (!tail_floor_leaf_cost(sink, i, lambda, tail_floor::config().hf,
                              tail_floor::config().edge, &d, &r, error)) {
      return false;
    }
    d_sum += d;
    r_sum += r;
  }
  r_sum += 2.0 * static_cast<double>(splits);
  *d_out = d_sum;
  *r_out = r_sum;
  return true;
}

bool InterFrameEncoder::tail_floor_collect_node(SbSink& sink,
                                                const PartitionNode& node,
                                                const ChromaInfo& inherited,
                                                PartitionNode* emitted,
                                                std::string& error) {
  tail_floor::Counters& tc = tail_floor::counters();
  const tail_floor::Config& cfg = tail_floor::config();
  ++tc.nodes;

  // The reference node is always a legal fallback: commit it unchanged.
  const auto commit_reference = [&]() -> bool {
    tail_floor_trial_ = true;
    const bool ok = collect_node(sink, node, error, inherited, emitted);
    tail_floor_trial_ = false;
    return ok;
  };
  if (!cfg.on || emitted == nullptr || node.forced ||
      node.size != partition::BlockSize::k16x16 ||
      tail_floor_sb_root_ == nullptr) {
    ++tc.leaves_16;
    ++tc.fallback;
    return commit_reference();
  }
  const bool eff_on =
      !pipe_ba::enabled() && eff_perc::enabled() && pipe_probe_ready_;
  const bool pipe_on =
      (pipe_perc::enabled() || eff_perc::enabled() || pipe_ba::enabled()) &&
      pipe_probe_ready_;
  if (!pipe_on) {
    ++tc.leaves_16;
    ++tc.fallback;
    return commit_reference();
  }
  const double pipe_lambda =
      leaf_rd_lambda(sink.state.current_qindex, options_) *
      (pipe_ba::enabled() ? pipe_ba::config().lambda
                          : (eff_on ? eff_perc::lambda_scale()
                                    : pipe_perc::lambda_scale()));
  const double lambda = pipe_lambda * cfg.lambda;
  if (!(lambda > 0.0)) {
    ++tc.leaves_16;
    ++tc.fallback;
    return commit_reference();
  }

  // --- Screen: the per-block partition probe on this node.  It is the same
  // measured NEWMV + quantised-residual cost the reference tree uses, so the
  // screen is a measured early bound, not a source-activity heuristic.
  std::function<TailFloorScreen(int, int, partition::BlockSize, int, bool,
                                bool)>
      screen = [&](int mi_row, int mi_col, partition::BlockSize size,
                   int level, bool has_chroma,
                   bool chroma_offset) -> TailFloorScreen {
    TailFloorScreen out;
    out.size = size;
    out.mi_row = mi_row;
    out.mi_col = mi_col;
    out.level = level;
    out.has_chroma = has_chroma;
    out.chroma_offset = chroma_offset;
    partition::PartitionNodeInfo info;
    info.mi_row = mi_row;
    info.mi_col = mi_col;
    info.size = size;
    info.depth = node.depth + level;
    info.has_chroma = has_chroma;
    info.chroma_offset = chroma_offset;
    info.boundary =
        partition::partition_implied(geometry_, mi_row, mi_col, size);
    info.allowed = partition::init_allowed_partitions(
        geometry_, mi_row, mi_col, size, has_chroma, chroma_offset);
    // A boundary-forced node has no NONE candidate: it is not the optional
    // split this stage measures.
    const bool forced =
        info.boundary.implied &&
        info.allowed.is_allowed(info.boundary.implied_partition);
    double leaf = 0.0;
    if (!partition_probe_leaf_cost(info, probe_syntax_price(mi_row, mi_col),
                                   leaf)) {
      return out;
    }
    out.leaf_cost = leaf;
    out.cost = leaf;
    out.ok = true;
    if (forced || level >= cfg.depth) return out;
    for (int trial = 0; trial < 2; ++trial) {
      const partition::PartitionType part =
          trial == 0 ? partition::PartitionType::kHorz
                     : partition::PartitionType::kVert;
      if (!info.allowed.is_allowed(part)) continue;
      const partition::BlockSize sub =
          partition::partition_subsize(part, size);
      if (!partition::is_valid_block_size(sub)) continue;
      if (partition::block_width(sub) < cfg.min_px ||
          partition::block_height(sub) < cfg.min_px) {
        continue;
      }
      SeqChildSlot slots[4];
      const int count = seq_child_slots(geometry_, mi_row, mi_col, size, part,
                                        has_chroma, chroma_offset, slots);
      if (count <= 0) continue;
      double sum = probe_syntax_price(mi_row, mi_col) * 2.0;
      std::vector<TailFloorScreen> kids;
      bool ok = true;
      for (int i = 0; i < count; ++i) {
        if (!seq_child_coded(geometry_, slots[i])) continue;
        TailFloorScreen kid =
            screen(slots[i].mi_row, slots[i].mi_col, slots[i].size, level + 1,
                   slots[i].has_chroma, slots[i].chroma_offset);
        if (!kid.ok) {
          ok = false;
          break;
        }
        sum += kid.cost;
        kids.push_back(std::move(kid));
      }
      if (!ok || kids.size() < 2) continue;
      if (sum < out.cost) {
        out.cost = sum;
        out.partition = part;
        out.children = std::move(kids);
      }
    }
    return out;
  };
  const double probe_price = probe_syntax_price(node.mi_row, node.mi_col);
  const TailFloorScreen screen_root =
      screen(node.mi_row, node.mi_col, node.size, 0, node.has_chroma,
             node.chroma_offset);
  if (!screen_root.ok) {
    ++tc.leaves_16;
    ++tc.fallback;
    return commit_reference();
  }
  const double screen_gain_bits =
      (screen_root.leaf_cost - screen_root.cost) / std::max(probe_price, 1e-9);
  tc.screen_bits += screen_gain_bits;
  if (cfg.screen) {
    ++tc.screened;
    const double margin = cfg.screen_margin * probe_price;
    if (screen_root.partition == partition::PartitionType::kNone ||
        screen_root.cost > screen_root.leaf_cost - margin) {
      ++tc.leaves_16;
      return commit_reference();
    }
  }
  ++tc.flagged;

  // --- Snapshot every surface encode_leaf() mutates.  The trials restore all
  // of it, so the committed encode is the only one that advances the frame.
  struct TailSnap {
    std::array<std::vector<std::uint8_t>, 3> samples;
    std::array<int, 3> stride{};
    std::array<int, 3> width{};
    std::array<int, 3> height{};
    std::vector<std::uint8_t> joint;
    std::vector<std::uint8_t> uv;
    std::vector<inter::mc::MiInfo> mv;
    SbSink::State state;
    std::vector<syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    std::vector<palette::PaletteResult> palettes;
    std::vector<post::VideoSubPuRect> sub_pus;
    std::vector<LeafDumpRecord> leaf_dump;
    std::vector<exact_rd::LeafCost> leaf_costs;
    std::vector<double> leaf_bits;
    std::vector<char> leaf_bits_ok;
    rate_hybrid::SyntaxProbe probe;
    rate_hybrid::SyntaxProbe rate_probe;
  };
  const auto capture = [&]() {
    TailSnap snap;
    for (int plane = 0; plane < 3; ++plane) {
      const std::size_t p = static_cast<std::size_t>(plane);
      snap.samples[p] = recon_[p].samples;
      snap.stride[p] = recon_[p].stride;
      snap.width[p] = recon_[p].width;
      snap.height[p] = recon_[p].height;
    }
    snap.joint = intra_joint_;
    snap.uv = uv_smooth_;
    snap.mv = mv_grid_;
    snap.state = sink.state;
    snap.blocks = sink.blocks;
    snap.levels = sink.levels;
    snap.palettes = sink.palettes;
    snap.sub_pus = sink.sub_pus;
    snap.leaf_dump = sink.leaf_dump;
    snap.leaf_costs = sink.leaf_costs;
    snap.leaf_bits = tail_floor_leaf_bits_;
    snap.leaf_bits_ok = tail_floor_leaf_bits_ok_;
    snap.probe = pipe_probe_;
    snap.rate_probe = rate_probe_;
    return snap;
  };
  const auto restore = [&](const TailSnap& snap) {
    for (int plane = 0; plane < 3; ++plane) {
      const std::size_t p = static_cast<std::size_t>(plane);
      recon_[p].samples = snap.samples[p];
      recon_[p].stride = snap.stride[p];
      recon_[p].width = snap.width[p];
      recon_[p].height = snap.height[p];
    }
    intra_joint_ = snap.joint;
    uv_smooth_ = snap.uv;
    mv_grid_ = snap.mv;
    sink.state = snap.state;
    sink.blocks = snap.blocks;
    sink.levels = snap.levels;
    sink.palettes = snap.palettes;
    sink.sub_pus = snap.sub_pus;
    sink.leaf_dump = snap.leaf_dump;
    sink.leaf_costs = snap.leaf_costs;
    tail_floor_leaf_bits_ = snap.leaf_bits;
    tail_floor_leaf_bits_ok_ = snap.leaf_bits_ok;
    pipe_probe_ = snap.probe;
    rate_probe_ = snap.rate_probe;
  };

  PartitionNode leaf = node;
  leaf.partition = partition::PartitionType::kNone;
  leaf.forced = false;
  leaf.children.clear();
  leaf.cost = 0.0;

  const TailSnap start = capture();
  const std::size_t begin = sink.blocks.size();

  // --- Trial 1: the reference 16x16 coding block, encoded in place.
  tail_floor_trial_ = true;
  if (!collect_node(sink, leaf, error, inherited, nullptr)) {
    tail_floor_trial_ = false;
    restore(start);
    ++tc.trial_failed;
    ++tc.fallback;
    ++tc.leaves_16;
    error.clear();
    return commit_reference();
  }
  double d_none = 0.0;
  double r_none = 0.0;
  if (!tail_floor_trial_cost(sink, begin, lambda, 0, &d_none, &r_none,
                             error)) {
    tail_floor_trial_ = false;
    restore(start);
    ++tc.trial_failed;
    ++tc.fallback;
    ++tc.leaves_16;
    error.clear();
    return commit_reference();
  }
  const double j_none = d_none + lambda * r_none;
  const TailSnap none_state = capture();
  restore(start);

  // --- Build the split candidate from the screen's hierarchical shape.
  int splits = 0;
  std::function<bool(const TailFloorScreen&, PartitionNode*)> build =
      [&](const TailFloorScreen& src, PartitionNode* out) -> bool {
    out->mi_row = src.mi_row;
    out->mi_col = src.mi_col;
    out->size = src.size;
    out->partition = src.partition;
    out->depth = node.depth + src.level;
    out->has_chroma = src.has_chroma;
    out->chroma_offset = src.chroma_offset;
    out->boundary = partition::partition_implied(geometry_, src.mi_row,
                                                 src.mi_col, src.size);
    out->allowed = partition::init_allowed_partitions(
        geometry_, src.mi_row, src.mi_col, src.size, src.has_chroma,
        src.chroma_offset);
    out->forced = false;
    out->cost = 0.0;
    out->children.clear();
    if (src.partition == partition::PartitionType::kNone) return true;
    ++splits;
    SeqChildSlot slots[4];
    const int count =
        seq_child_slots(geometry_, src.mi_row, src.mi_col, src.size,
                        src.partition, src.has_chroma, src.chroma_offset,
                        slots);
    if (count <= 0 || static_cast<std::size_t>(count) != src.children.size()) {
      return false;
    }
    out->children.resize(src.children.size());
    for (std::size_t i = 0; i < src.children.size(); ++i) {
      if (!build(src.children[i], &out->children[i])) return false;
    }
    return true;
  };
  PartitionNode split;
  std::string build_error;
  bool split_ok =
      screen_root.partition != partition::PartitionType::kNone &&
      build(screen_root, &split);
  if (split_ok) {
    // Validate the candidate in the real superblock tree before encoding it.
    PartitionNode probe_root = *tail_floor_sb_root_;
    bool replaced = false;
    if (!tail_floor_replace(probe_root, node.mi_row, node.mi_col, node.size,
                            split, &replaced) ||
        !replaced ||
        !partition::validate_tree(geometry_, probe_root, build_error)) {
      split_ok = false;
    }
  }
  if (!split_ok) {
    ++tc.leaves_16;
    ++tc.fallback;
    restore(none_state);
    tail_floor_trial_ = false;
    *emitted = leaf;
    return true;
  }

  // --- Trial 2: the screened split layout, encoded in place from the same
  // reconstruction the 16x16 trial started from.
  ++tc.trials;
  if (!collect_node(sink, split, error, inherited, nullptr)) {
    tail_floor_trial_ = false;
    restore(none_state);
    ++tc.trial_failed;
    ++tc.fallback;
    ++tc.leaves_16;
    error.clear();
    *emitted = leaf;
    return true;
  }
  double d_split = 0.0;
  double r_split = 0.0;
  if (!tail_floor_trial_cost(sink, begin, lambda, splits, &d_split, &r_split,
                             error)) {
    tail_floor_trial_ = false;
    restore(none_state);
    ++tc.trial_failed;
    ++tc.fallback;
    ++tc.leaves_16;
    error.clear();
    *emitted = leaf;
    return true;
  }
  tail_floor_trial_ = false;
  const double j_split = d_split + lambda * r_split;
  tc.d_none += d_none;
  tc.r_none += r_none;
  tc.d_split += d_split;
  tc.r_split += r_split;
  tc.j_delta += j_split - j_none;
  if (j_split < j_none) {
    // Keep the split encode: it is the committed state, no re-encode.
    int n16 = 0;
    int n8 = 0;
    int n4 = 0;
    const int leaves = tail_floor_leaves(split, &n16, &n8, &n4);
    ++tc.split_kept;
    tc.split_leaves += leaves;
    tc.leaves_16 += n16;
    tc.leaves_8 += n8;
    tc.leaves_4 += n4;
    *emitted = split;
    return true;
  }
  restore(none_state);
  ++tc.none_kept;
  ++tc.leaves_16;
  *emitted = leaf;
  return true;
}

// AV2ENC_PART_DIAG -----------------------------------------------------------
//
// Measure one emitted P leaf against each legal one-level split and commit the
// reference.  Both sides use the tail floor's exact objective, so the only
// difference between the two numbers is the coder's own leaf decision:
//
//   J = D_BA(leaf recon) + lambda * (R_exact + 2 bits per partition symbol)
//
// The stream is unchanged; the counters printed at exit answer whether the
// probe's depth-1 clamp is leaving splits on the table.  Prediction for an
// inter leaf comes from the reference picture, so a trial is exact for it;
// an intra leaf inside P reads the current reconstruction and is the
// approximation this diagnostic accepts.
bool InterFrameEncoder::part_diag_candidate(const PartitionNode& node) const {
  if (node.partition != partition::PartitionType::kNone) return false;
  if (node.forced || node.depth == 0) return false;
  const int w = partition::block_width(node.size);
  const int h = partition::block_height(node.size);
  if (w <= 0 || h <= 0 || w > 64 || h > 64) return false;
  if ((w < h ? w : h) < part_diag::config().min_px) return false;
  return node.allowed.is_allowed(partition::PartitionType::kHorz) ||
         node.allowed.is_allowed(partition::PartitionType::kVert) ||
         node.allowed.is_allowed(partition::PartitionType::kSplit);
}

bool InterFrameEncoder::part_diag_collect_node(SbSink& sink,
                                               const PartitionNode& node,
                                               const ChromaInfo& inherited,
                                               PartitionNode* emitted,
                                               std::string& error) {
  part_diag::Counters& c = part_diag::counters();
  const part_diag::Config& cfg = part_diag::config();
  const auto commit_reference = [&]() -> bool {
    part_diag_trial_ = true;
    const bool ok = collect_node(sink, node, error, inherited, emitted);
    part_diag_trial_ = false;
    return ok;
  };
  static bool cap_logged = false;
  if (cfg.max_nodes > 0 &&
      c.total.nodes + c.fails >= cfg.max_nodes) {
    if (!cap_logged) {
      ++c.capped;
      cap_logged = true;
    }
    return commit_reference();
  }

  const bool eff_on =
      !pipe_ba::enabled() && eff_perc::enabled() && pipe_probe_ready_;
  const bool pipe_on =
      (pipe_perc::enabled() || eff_perc::enabled() || pipe_ba::enabled()) &&
      pipe_probe_ready_;
  if (!pipe_on) return commit_reference();
  const double pipe_lambda =
      leaf_rd_lambda(sink.state.current_qindex, options_) *
      (pipe_ba::enabled() ? pipe_ba::config().lambda
                          : (eff_on ? eff_perc::lambda_scale()
                                    : pipe_perc::lambda_scale()));
  const double lambda = pipe_lambda * cfg.lambda;
  if (!(lambda > 0.0)) return commit_reference();
  c.lambda_ = lambda;

  // Snapshot every surface encode_leaf() mutates.  Only the reconstruction
  // and the two probes need their bytes back; the sink vectors only grow, so
  // restoring their sizes is enough.
  struct DiagSnap {
    std::array<std::vector<std::uint8_t>, 3> samples;
    std::array<int, 3> stride{};
    std::array<int, 3> width{};
    std::array<int, 3> height{};
    std::vector<std::uint8_t> joint;
    std::vector<std::uint8_t> uv;
    std::vector<inter::mc::MiInfo> mv;
    SbSink::State state;
    std::size_t blocks = 0;
    std::size_t levels = 0;
    std::size_t palettes = 0;
    std::size_t sub_pus = 0;
    std::size_t leaf_dump = 0;
    std::size_t leaf_costs = 0;
    std::size_t leaf_bits = 0;
    std::size_t leaf_bits_ok = 0;
    rate_hybrid::SyntaxProbe probe;
    rate_hybrid::SyntaxProbe rate_probe;
  };
  const auto capture = [&]() {
    DiagSnap s;
    for (int plane = 0; plane < 3; ++plane) {
      const std::size_t p = static_cast<std::size_t>(plane);
      s.samples[p] = recon_[p].samples;
      s.stride[p] = recon_[p].stride;
      s.width[p] = recon_[p].width;
      s.height[p] = recon_[p].height;
    }
    s.joint = intra_joint_;
    s.uv = uv_smooth_;
    s.mv = mv_grid_;
    s.state = sink.state;
    s.blocks = sink.blocks.size();
    s.levels = sink.levels.size();
    s.palettes = sink.palettes.size();
    s.sub_pus = sink.sub_pus.size();
    s.leaf_dump = sink.leaf_dump.size();
    s.leaf_costs = sink.leaf_costs.size();
    s.leaf_bits = tail_floor_leaf_bits_.size();
    s.leaf_bits_ok = tail_floor_leaf_bits_ok_.size();
    s.probe = pipe_probe_;
    s.rate_probe = rate_probe_;
    return s;
  };
  const auto restore = [&](const DiagSnap& s) {
    for (int plane = 0; plane < 3; ++plane) {
      const std::size_t p = static_cast<std::size_t>(plane);
      recon_[p].samples = s.samples[p];
      recon_[p].stride = s.stride[p];
      recon_[p].width = s.width[p];
      recon_[p].height = s.height[p];
    }
    intra_joint_ = s.joint;
    uv_smooth_ = s.uv;
    mv_grid_ = s.mv;
    sink.state = s.state;
    sink.blocks.resize(s.blocks);
    sink.levels.resize(s.levels);
    sink.palettes.resize(s.palettes);
    sink.sub_pus.resize(s.sub_pus);
    sink.leaf_dump.resize(s.leaf_dump);
    sink.leaf_costs.resize(s.leaf_costs);
    tail_floor_leaf_bits_.resize(s.leaf_bits);
    tail_floor_leaf_bits_ok_.resize(s.leaf_bits_ok);
    pipe_probe_ = s.probe;
    rate_probe_ = s.rate_probe;
  };

  PartitionNode leaf = node;
  leaf.partition = partition::PartitionType::kNone;
  leaf.forced = false;
  leaf.children.clear();
  leaf.cost = 0.0;

  const DiagSnap start = capture();
  std::size_t begin = sink.blocks.size();
  part_diag_trial_ = true;
  const bool leaf_encoded = collect_node(sink, leaf, error, inherited, nullptr);
  part_diag_trial_ = false;
  double d_leaf = 0.0;
  double r_leaf = 0.0;
  const bool leaf_priced =
      leaf_encoded && tail_floor_trial_cost(sink, begin, lambda, 0, &d_leaf,
                                            &r_leaf, error);
  restore(start);
  if (!leaf_priced) {
    ++c.fails;
    error.clear();
    return commit_reference();
  }

  double best_j = d_leaf + lambda * r_leaf;
  double best_d = d_leaf;
  double best_r = r_leaf;
  int best_part = 0;
  PartitionNode best_split;
  bool best_split_ok = false;
  static int trace_left = -1;
  if (trace_left < 0) {
    trace_left = std::getenv("AV2ENC_PART_DIAG_TRACE") != nullptr ? 12 : 0;
  }
  for (int trial = 0; trial < 3; ++trial) {
    const partition::PartitionType part =
        trial == 0 ? partition::PartitionType::kHorz
        : trial == 1 ? partition::PartitionType::kVert
                     : partition::PartitionType::kSplit;
    const auto trace = [&](const char* why) {
      if (trace_left <= 0) return;
      --trace_left;
      std::fprintf(stderr,
                   "rd-part-diag trace: %dx%d depth=%d part=%d allowed=%d "
                   "why=%s\n",
                   partition::block_width(node.size),
                   partition::block_height(node.size), node.depth,
                   static_cast<int>(part),
                   node.allowed.is_allowed(part) ? 1 : 0, why);
    };
    if (!node.allowed.is_allowed(part)) {
      trace("not-allowed");
      continue;
    }
    const partition::BlockSize sub =
        partition::partition_subsize(part, node.size);
    if (!partition::is_valid_block_size(sub)) {
      trace("bad-subsize");
      continue;
    }
    SeqChildSlot slots[4];
    const int count = seq_child_slots(geometry_, node.mi_row, node.mi_col,
                                      node.size, part, node.has_chroma,
                                      node.chroma_offset, slots);
    if (count < 2) {
      trace("no-slots");
      continue;
    }
    PartitionNode split;
    split.mi_row = node.mi_row;
    split.mi_col = node.mi_col;
    split.size = node.size;
    split.partition = part;
    split.depth = node.depth;
    split.has_chroma = node.has_chroma;
    split.chroma_offset = node.chroma_offset;
    split.boundary =
        partition::partition_implied(geometry_, node.mi_row, node.mi_col,
                                     node.size);
    split.allowed = node.allowed;
    split.forced = false;
    split.cost = 0.0;
    bool built = true;
    for (int i = 0; i < count; ++i) {
      if (!seq_child_coded(geometry_, slots[i])) continue;
      PartitionNode child;
      child.mi_row = slots[i].mi_row;
      child.mi_col = slots[i].mi_col;
      child.size = slots[i].size;
      child.depth = node.depth + 1;
      child.has_chroma = slots[i].has_chroma;
      child.chroma_offset = slots[i].chroma_offset;
      child.boundary = partition::partition_implied(
          geometry_, child.mi_row, child.mi_col, child.size);
      child.allowed = partition::init_allowed_partitions(
          geometry_, child.mi_row, child.mi_col, child.size,
          child.has_chroma, child.chroma_offset);
      child.forced = false;
      child.partition = partition::PartitionType::kNone;
      child.cost = 0.0;
      split.children.push_back(std::move(child));
    }
    if (split.children.size() < 2) {
      built = false;
      trace("children<2");
    }
    if (built && part_diag_sb_root_ != nullptr) {
      PartitionNode probe_root = *part_diag_sb_root_;
      bool replaced = false;
      std::string build_error;
      if (!tail_floor_replace(probe_root, node.mi_row, node.mi_col,
                              node.size, split, &replaced) ||
          !replaced) {
        built = false;
        trace("not-found");
      } else if (!partition::validate_tree(geometry_, probe_root,
                                           build_error)) {
        built = false;
        if (trace_left > 0) {
          --trace_left;
          std::fprintf(stderr,
                       "rd-part-diag trace: %dx%d depth=%d part=%d "
                       "invalid-tree: %s\n",
                       partition::block_width(node.size),
                       partition::block_height(node.size), node.depth,
                       static_cast<int>(part), build_error.c_str());
        }
      }
    }
    if (!built) {
      trace("build-fail");
      continue;
    }

    ++c.trials;
    begin = sink.blocks.size();
    part_diag_trial_ = true;
    const bool split_encoded = collect_node(sink, split, error, inherited,
                                            nullptr);
    part_diag_trial_ = false;
    double d_split = 0.0;
    double r_split = 0.0;
    const bool split_priced =
        split_encoded && tail_floor_trial_cost(sink, begin, lambda, 1,
                                               &d_split, &r_split, error);
    restore(start);
    if (!split_priced) {
      ++c.fails;
      error.clear();
      continue;
    }
    const double j = d_split + lambda * r_split;
    if (j < best_j) {
      best_j = j;
      best_d = d_split;
      best_r = r_split;
      best_part = trial + 1;
      best_split = split;
      best_split_ok = true;
    }
  }

  part_diag::Sample sample;
  sample.shape = part_diag::shape_of(partition::block_width(node.size),
                                     partition::block_height(node.size));
  sample.d_leaf = d_leaf;
  sample.r_leaf = r_leaf;
  sample.d_split = best_d;
  sample.r_split = best_r;
  sample.part = best_part;
  sample.ok = true;
  c.add(sample);
  error.clear();
  if (cfg.commit && best_split_ok) {
    part_diag_trial_ = true;
    const bool ok = collect_node(sink, best_split, error, inherited, emitted);
    part_diag_trial_ = false;
    return ok;
  }
  return commit_reference();
}

InterFrameEncoder::JointTrialSnap InterFrameEncoder::joint_p_capture(
    const SbSink& sink) const {
  JointTrialSnap snap;
  for (int plane = 0; plane < 3; ++plane) {
    const std::size_t p = static_cast<std::size_t>(plane);
    snap.samples[p] = recon_[p].samples;
    snap.stride[p] = recon_[p].stride;
    snap.width[p] = recon_[p].width;
    snap.height[p] = recon_[p].height;
  }
  snap.joint = intra_joint_;
  snap.uv = uv_smooth_;
  snap.mv = mv_grid_;
  snap.state = sink.state;
  snap.blocks = sink.blocks.size();
  snap.levels = sink.levels.size();
  snap.palettes = sink.palettes.size();
  snap.sub_pus = sink.sub_pus.size();
  snap.leaf_dump = sink.leaf_dump.size();
  snap.leaf_costs = sink.leaf_costs.size();
  snap.leaf_bits = tail_floor_leaf_bits_.size();
  snap.leaf_bits_ok = tail_floor_leaf_bits_ok_.size();
  snap.probe = pipe_probe_;
  snap.rate_probe = rate_probe_;
  return snap;
}

void InterFrameEncoder::joint_p_apply(SbSink& sink,
                                      const JointTrialSnap& snap) {
  for (int plane = 0; plane < 3; ++plane) {
    const std::size_t p = static_cast<std::size_t>(plane);
    recon_[p].samples = snap.samples[p];
    recon_[p].stride = snap.stride[p];
    recon_[p].width = snap.width[p];
    recon_[p].height = snap.height[p];
  }
  intra_joint_ = snap.joint;
  uv_smooth_ = snap.uv;
  mv_grid_ = snap.mv;
  sink.state = snap.state;
  sink.blocks.resize(snap.blocks);
  sink.levels.resize(snap.levels);
  sink.palettes.resize(snap.palettes);
  sink.sub_pus.resize(snap.sub_pus);
  sink.leaf_dump.resize(snap.leaf_dump);
  sink.leaf_costs.resize(snap.leaf_costs);
  tail_floor_leaf_bits_.resize(snap.leaf_bits);
  tail_floor_leaf_bits_ok_.resize(snap.leaf_bits_ok);
  pipe_probe_ = snap.probe;
  rate_probe_ = snap.rate_probe;
}

bool InterFrameEncoder::joint_p_same_layout(const PartitionNode& a,
                                            const PartitionNode& b) {
  // Everything `collect_node` reads to decide which partition symbols are
  // coded, not just the geometry: `forced`/`boundary`/`allowed` change the
  // bitstream, so two trees that differ there are not the same layout even
  // when they hold the same leaves.
  if (a.mi_row != b.mi_row || a.mi_col != b.mi_col || a.size != b.size ||
      a.partition != b.partition || a.depth != b.depth ||
      a.has_chroma != b.has_chroma || a.chroma_offset != b.chroma_offset ||
      a.forced != b.forced || a.boundary.has_rows != b.boundary.has_rows ||
      a.boundary.has_cols != b.boundary.has_cols ||
      a.boundary.implied != b.boundary.implied ||
      a.boundary.implied_partition != b.boundary.implied_partition ||
      a.allowed.count != b.allowed.count) {
    return false;
  }
  for (int p = 0; p < partition::kPartitionTypeCount; ++p) {
    if (a.allowed.allowed[p] != b.allowed.allowed[p]) return false;
  }
  if (a.children.size() != b.children.size()) return false;
  for (std::size_t i = 0; i < a.children.size(); ++i) {
    if (!joint_p_same_layout(a.children[i], b.children[i])) return false;
  }
  return true;
}

bool InterFrameEncoder::joint_p_measure_tree(const PartitionNode& root,
                                             SbSink& sink, double lambda,
                                             double* cost,
                                             std::string& error) {
  if (cost == nullptr) return false;
  double total = 0.0;
  std::function<bool(const PartitionNode&, const ChromaInfo&)> walk;
  walk = [&](const PartitionNode& node,
             const ChromaInfo& inherited) -> bool {
    const ChromaInfo chroma = chroma_step(node, inherited);
    if (node.partition == PartitionType::kNone) {
      double leaf_cost = 0.0;
      if (!joint_p_measure_leaf(node, inherited, sink, lambda, &leaf_cost,
                                error)) {
        return false;
      }
      total += leaf_cost;
      return true;
    }
    total += lambda * 2.0;
    for (const PartitionNode& child : node.children) {
      if (!walk(child, chroma)) return false;
    }
    return true;
  };
  if (!walk(root, ChromaInfo{})) return false;
  *cost = total;
  return true;
}

void InterFrameEncoder::joint_p_copy_frame_state(const InterFrameEncoder& src) {
  // The frame-start decision state pass 1 must see.  Everything here is either
  // rebuilt identically by plan_pass() (the source-driven weights, the masks,
  // the probes) or is genuinely raster state (the reconstruction and the
  // neighbour maps).  Copying the reconstruction is what lets a worker price a
  // leaf whose prediction or BAWP fit reads a neighbour it will not code.
  for (int plane = 0; plane < 3; ++plane) {
    const std::size_t p = static_cast<std::size_t>(plane);
    recon_[p] = src.recon_[p];
    compound_pred_[p] = src.compound_pred_[p];
  }
  intra_joint_ = src.intra_joint_;
  uv_smooth_ = src.uv_smooth_;
  mv_grid_ = src.mv_grid_;
  me_cache_ = src.me_cache_;
  qm_state_ = src.qm_state_;
  quant_step_ = src.quant_step_;
  for (int plane = 0; plane < 3; ++plane) {
    plane_quant_[plane] = src.plane_quant_[plane];
  }
  current_qindex_ = src.current_qindex_;
#if defined(AV2ENC_VIDEO_DELTAQ)
  sb_target_q_ = src.sb_target_q_;
  sb_qindex_ = src.sb_qindex_;
#endif
  palettes_.clear();
  blocks_.clear();
  levels_.clear();
}

std::unique_ptr<InterFrameEncoder> InterFrameEncoder::joint_p_trial_clone()
    const {
  auto clone = std::make_unique<InterFrameEncoder>(
      source_, references_, frame_distance_, num_total_refs_, options_,
      /*trace=*/false, keyframe_, tile_x_, tile_y_, tile_w_, tile_h_,
      cdf_seed_, /*cdf_final=*/nullptr, gm_plan_, ref_order_hints_,
      primary_ref_is_none_, &temporal_ctx_, prev_segment_ids_, illum_delta_,
      ref_deficit_db_, census_frame_, horizon_future_);
  clone->set_display_ordinal(display_ordinal_);
  if (joint_plan_off_) clone->set_joint_plan_off();
  // plan_pass() rebuilds the deterministic per-frame surfaces (weights, BA
  // mask, exact-syntax probe, frame-start CDF image) from the source and the
  // options; the copy below only carries the state that is raster-dependent.
  clone->plan_pass();
  clone->joint_p_copy_frame_state(*this);
  return clone;
}

bool InterFrameEncoder::joint_p_batch_frame(
    std::vector<PartitionNode>& superblocks, std::string& error) {
  const joint_plan::Config& cfg = joint_plan::config();
  const std::size_t count = superblocks.size();
  if (count == 0) return true;
  // --cpu-threads owns the worker count; the gate only selects the batch.
  const unsigned workers = std::max<unsigned>(
      1U, std::min<unsigned>(
              static_cast<unsigned>(cfg.batch),
              parallel::resolve_workers(options_.cpu_threads)));
  const std::size_t chunks =
      std::max<std::size_t>(1, std::min<std::size_t>(workers, count));
  const bool chain = cfg.batch_chain;
  const bool eff_on =
      !pipe_ba::enabled() && eff_perc::enabled() && pipe_probe_ready_;
  const double pipe_lambda =
      leaf_rd_lambda(current_qindex_, options_) *
      (pipe_ba::enabled() ? pipe_ba::config().lambda
                          : (eff_on ? eff_perc::lambda_scale()
                                    : pipe_perc::lambda_scale()));
  const double lambda = pipe_lambda * cfg.lambda;

  const auto fresh_sink = [&](SbSink& sink) {
    sink.reset();
    seed_unit_state(sink);
    sink.state.decoded[0].reset(mi_cols_, mi_rows_);
    sink.state.decoded[1].reset(mi_cols_ / 2, mi_rows_ / 2);
    sink.state.decoded[2].reset(mi_cols_ / 2, mi_rows_ / 2);
  };
  SbSink scratch;
  fresh_sink(scratch);
  const JointTrialSnap frame_start = joint_p_capture(scratch);

  joint_p_active_ = true;
  const auto fail = [&]() {
    joint_p_active_ = false;
    return false;
  };

  // Optional pass 0 (AV2ENC_RD_JOINT_BATCH_SEED=1): commit the reference
  // layouts serially and record the frame state every superblock starts from,
  // so pass 1 searches in the raster context the reference tree builds instead
  // of the frame-start one.
  std::vector<JointTrialSnap> starts;
  if (cfg.batch_seed_ref) {
    starts.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      starts[i] = joint_p_capture(scratch);
      if (!collect_node(scratch, superblocks[i], error)) return fail();
    }
    joint_p_apply(scratch, frame_start);
    blocks_.clear();
    levels_.clear();
    palettes_.clear();
  }

  // The gate-off tree is the anchor of every round: it is what the serial walk
  // passes to `joint_p_decide_sb` (the expansion start and the reference
  // guard), and holding it fixed is what makes the iteration a fixed point of
  // the same map instead of a re-solve of the previous round's answer.
  const std::vector<PartitionNode> reference = superblocks;
  std::vector<PartitionNode> candidate(count);
  std::vector<char> candidate_ok(count, 0);
  std::vector<JointTrialSnap> committed_starts;
  sb_sinks_.resize(count);
  long long changed_total = 0;

  for (int round = 0; round < cfg.batch_rounds; ++round) {
    // Pass 1: candidate layout per superblock, searched on a private clone
    // whose only shared inputs are the source and the reference frames.
    for (std::size_t i = 0; i < count; ++i) {
      candidate[i] = reference[i];
      candidate_ok[i] = 0;
    }
    const std::vector<JointTrialSnap>* seeds =
        starts.empty() ? nullptr : &starts;
    const auto run_chunk = [&](std::size_t chunk) {
      std::unique_ptr<InterFrameEncoder> worker = joint_p_trial_clone();
      joint_plan::trial_worker() = true;
      worker->joint_p_active_ = true;
      SbSink seed_sink;
      seed_sink.reset();
      seed_sink.state.decoded[0].reset(mi_cols_, mi_rows_);
      seed_sink.state.decoded[1].reset(mi_cols_ / 2, mi_rows_ / 2);
      seed_sink.state.decoded[2].reset(mi_cols_ / 2, mi_rows_ / 2);
      // Chained range: contiguous in raster order, so the worker's own
      // committed state is the serial walk's state for every superblock after
      // the first one.
      const std::size_t chain_begin = chunks == 0 ? 0 : chunk * count / chunks;
      const std::size_t chain_end =
          chunks == 0 ? count : (chunk + 1) * count / chunks;
      if (chain && seeds != nullptr) {
        worker->joint_p_apply(seed_sink, (*seeds)[chain_begin]);
      }
      for (std::size_t step = 0;; ++step) {
        const std::size_t index =
            chain ? chain_begin + step : chunk + step * chunks;
        if (chain ? index >= chain_end : index >= count) break;
        if (!chain && seeds != nullptr) {
          worker->joint_p_apply(seed_sink, (*seeds)[index]);
        }
        PartitionNode chosen;
        std::string chunk_error;
        if (!worker->joint_p_decide_sb(index, reference[index], &chosen,
                                       chunk_error)) {
          continue;
        }
        if (chain) {
          // Advance the worker to the state this layout builds, exactly what
          // pass 2 will commit: collect_node() reads the quantiser/segment
          // state off the sink, so the sink has to be seeded the way
          // collect_frame() seeds it before the commit.
          seed_sink.reset();
          worker->seed_unit_state(seed_sink);
          if (worker->collect_node(seed_sink, chosen, chunk_error)) {
            candidate[index] = std::move(chosen);
            candidate_ok[index] = 1;
          }
        } else {
          candidate[index] = std::move(chosen);
          candidate_ok[index] = 1;
        }
      }
      joint_plan::trial_worker() = false;
    };
    {
      joint_plan::ScopedTimer timer(&joint_plan::profile().pass1);
      if (chunks <= 1) {
        run_chunk(0);
      } else {
        parallel::parallel_for(chunks, workers, run_chunk);
      }
    }

    // Pass 2: exact re-price in raster order, then commit the winner.
    joint_plan::ScopedTimer pass2_timer(&joint_plan::profile().pass2);
    committed_starts.resize(count);
    long long changed_round = 0;
    for (std::size_t i = 0; i < count; ++i) {
      SbSink& sink = sb_sinks_[i];
      fresh_sink(sink);
      committed_starts[i] = joint_p_capture(sink);
      PartitionNode committed = reference[i];
      if (candidate_ok[i] != 0 && lambda > 0.0) {
        if (joint_p_same_layout(candidate[i], reference[i])) {
          // The batch reproduced the reference layout: commit it as the
          // serial walk would, with no extra measurement.
          if (joint_plan::counting()) {
            ++joint_plan::counters().batch_candidate_kept;
          }
        } else {
          const JointTrialSnap start = joint_p_capture(sink);
          std::string measure_error;
          double candidate_cost = partition::kInfiniteCost;
          const bool priced = joint_p_measure_tree(
              candidate[i], sink, lambda, &candidate_cost, measure_error);
          joint_p_apply(sink, start);
          if (priced) {
            double reference_cost = partition::kInfiniteCost;
            const bool reference_priced = joint_p_measure_tree(
                reference[i], sink, lambda, &reference_cost, measure_error);
            joint_p_apply(sink, start);
            if (reference_priced && std::isfinite(reference_cost)) {
              if (joint_plan::counting()) {
                joint_plan::counters().ref_j += reference_cost;
                ++joint_plan::counters().batch_reference_priced;
              }
              // The serial guard: keep the reference layout whenever the
              // exact objective does not prefer the candidate.
              if (candidate_cost < reference_cost) {
                committed = std::move(candidate[i]);
                if (joint_plan::counting()) {
                  ++joint_plan::counters().batch_candidate_kept;
                  joint_plan::counters().chosen_j += candidate_cost;
                }
              } else if (joint_plan::counting()) {
                ++joint_plan::counters().batch_reference_wins;
              }
            }
          }
        }
      }
      std::string validate_error;
      if (committed.partition != PartitionType::kNone &&
          !partition::validate_tree(geometry_, committed, validate_error)) {
        committed = reference[i];
      }
      if (!joint_p_same_layout(committed, reference[i])) ++changed_round;
      superblocks[i] = std::move(committed);
      if (!collect_node(sink, superblocks[i], error)) return fail();
    }
    changed_total = changed_round;
    if (joint_plan::counting()) {
      joint_plan::counters().layouts_changed += changed_round;
    }
    if (round + 1 < cfg.batch_rounds) {
      // Reset the frame and re-seed pass 1 from the state this round actually
      // committed at each superblock start (Jacobi iteration on the raster
      // recurrence).
      joint_p_apply(scratch, frame_start);
      blocks_.clear();
      levels_.clear();
      palettes_.clear();
      starts = std::move(committed_starts);
    }
  }
  if (joint_plan::counting()) {
    joint_plan::counters().batch_changed = changed_total;
  }
  joint_p_active_ = false;
  merge_sinks();
  return true;
}

// AV2ENC_RD_JOINT_P ----------------------------------------------------------
//
// The P-frame port of the key driver (`seq_driver_decide_sb`).  Every leaf is
// encoded with the coder's own field (mode search, TX/levels, coeff trellis)
// through collect_node(), priced with the P leaf objective (eff-perc inter CSF
// + exact §8.3 interval rate + live inter-syntax price), and every partition
// node is decided with one J.  Unlike the key driver the leaves are measured
// through collect_node() (not encode_leaf() directly) because the P path owns
// the inter syntax probe, §6.19.2.3 maps and the rate stash the exact cost
// reads; the sequential raster walk is what makes an intra leaf inside P
// priced in the reconstruction the decoder will see.
bool InterFrameEncoder::joint_p_measure_leaf(const PartitionNode& node,
                                             const ChromaInfo& inherited,
                                             SbSink& sink, double lambda,
                                             double* cost,
                                             std::string& error) {
  if (cost == nullptr || node.partition != PartitionType::kNone) return false;
  joint_plan::ScopedTimer leaf_timer(&joint_plan::profile().leaf);
  std::size_t index = sink.blocks.size();
  bool ok = false;
  {
    joint_plan::ScopedTimer timer(&joint_plan::profile().leaf_encode);
    ok = collect_node(sink, node, error, inherited, nullptr);
  }
  if (!ok) {
    ++joint_plan::counters().fail_collect;
    error.clear();
    return false;
  }
  // A coding-block candidate must write exactly one syntax entry; anything
  // else (a chroma tree, a palette/SDP split) is outside this objective.
  if (sink.blocks.size() != index + 1) {
    ++joint_plan::counters().fail_blocks;
    error.clear();
    return false;
  }
  {
    const syntax::BlockSyntax& probe = sink.blocks[index];
    joint_plan::profile_bump(&joint_plan::profile().trials);
    if (probe.is_inter == 0) {
      joint_plan::profile_bump(&joint_plan::profile().trials_intra);
    } else {
      joint_plan::profile_bump(&joint_plan::profile().trials_inter);
      if (probe.skip_flag != 0) {
        joint_plan::profile_bump(&joint_plan::profile().trials_skip);
      }
    }
  }
  double d = 0.0;
  double r = 0.0;
  {
    joint_plan::ScopedTimer timer(&joint_plan::profile().leaf_cost);
    if (!tail_floor_leaf_cost(sink, index, lambda, joint_plan::config().hf,
                              joint_plan::config().edge, &d, &r, error)) {
      ++joint_plan::counters().fail_cost;
      error.clear();
      return false;
    }
  }
  ++joint_plan::counters().leaves_measured;
  joint_plan::counters().d_measured += d;
  joint_plan::counters().r_measured += r;
  *cost = d + lambda * r;
  return true;
}

bool InterFrameEncoder::joint_p_decide_sb(std::size_t sb_index,
                                          const PartitionNode& reference,
                                          PartitionNode* out,
                                          std::string& error) {
  if (out == nullptr) return false;
  joint_plan::Counters& stats = joint_plan::counters();
  const joint_plan::Config& cfg = joint_plan::config();
  ++stats.superblocks;
  (void)sb_index;

  const int target = std::max(cfg.target, 8);
  const int max_depth = std::max(options_.max_partition_depth,
                                 seq_depth_for_target(target));
  // Expand the reference layout to the driver's deepest candidate leaf.  For
  // a reference leaf larger than `target` this is the uniform split; every
  // split in the expanded tree is a real §5.20.3.2 partition for the geometry.
  std::function<PartitionNode(const PartitionNode&, const ChromaInfo&)> expand;
  expand = [&](const PartitionNode& node,
               const ChromaInfo& inherited) -> PartitionNode {
    const ChromaInfo chroma = chroma_step(node, inherited);
    if (node.partition == PartitionType::kNone) {
      const int bw = partition::block_width(node.size);
      const int bh = partition::block_height(node.size);
      if (bw <= target && bh <= target) return node;
      return seq_build_uniform(geometry_, node.mi_row, node.mi_col, node.size,
                               node.depth, node.has_chroma, node.chroma_offset,
                               target, max_depth, target, nullptr);
    }
    PartitionNode grown = node;
    grown.children.clear();
    for (const PartitionNode& child : node.children) {
      grown.children.push_back(expand(child, chroma));
    }
    return grown;
  };
  PartitionNode expanded = expand(reference, ChromaInfo{});
  std::string validate_error;
  if (!partition::validate_tree(geometry_, expanded, validate_error)) {
    ++stats.failed;
    *out = reference;
    return true;
  }

  // Snapshot every surface the leaf coder mutates; every trial restores it and
  // only the committed collect_node() in collect_frame() advances the frame.
  struct Snap {
    std::array<std::vector<std::uint8_t>, 3> samples;
    std::array<int, 3> stride{};
    std::array<int, 3> width{};
    std::array<int, 3> height{};
    std::vector<std::uint8_t> joint;
    std::vector<std::uint8_t> uv;
    std::vector<inter::mc::MiInfo> mv;
    SbSink::State state;
    std::size_t blocks = 0;
    std::size_t levels = 0;
    std::size_t palettes = 0;
    std::size_t sub_pus = 0;
    std::size_t leaf_dump = 0;
    std::size_t leaf_costs = 0;
    std::size_t leaf_bits = 0;
    std::size_t leaf_bits_ok = 0;
    rate_hybrid::SyntaxProbe probe;
    rate_hybrid::SyntaxProbe rate_probe;
  };
  SbSink sink;
  sink.reset();
  seed_unit_state(sink);
  sink.state.decoded[0].reset(mi_cols_, mi_rows_);
  sink.state.decoded[1].reset(mi_cols_ / 2, mi_rows_ / 2);
  sink.state.decoded[2].reset(mi_cols_ / 2, mi_rows_ / 2);
  const auto capture = [&]() {
    Snap snap;
    for (int plane = 0; plane < 3; ++plane) {
      const std::size_t p = static_cast<std::size_t>(plane);
      snap.samples[p] = recon_[p].samples;
      snap.stride[p] = recon_[p].stride;
      snap.width[p] = recon_[p].width;
      snap.height[p] = recon_[p].height;
    }
    snap.joint = intra_joint_;
    snap.uv = uv_smooth_;
    snap.mv = mv_grid_;
    snap.state = sink.state;
    snap.blocks = sink.blocks.size();
    snap.levels = sink.levels.size();
    snap.palettes = sink.palettes.size();
    snap.sub_pus = sink.sub_pus.size();
    snap.leaf_dump = sink.leaf_dump.size();
    snap.leaf_costs = sink.leaf_costs.size();
    snap.leaf_bits = tail_floor_leaf_bits_.size();
    snap.leaf_bits_ok = tail_floor_leaf_bits_ok_.size();
    snap.probe = pipe_probe_;
    snap.rate_probe = rate_probe_;
    return snap;
  };
  const auto apply = [&](const Snap& snap) {
    for (int plane = 0; plane < 3; ++plane) {
      const std::size_t p = static_cast<std::size_t>(plane);
      recon_[p].samples = snap.samples[p];
      recon_[p].stride = snap.stride[p];
      recon_[p].width = snap.width[p];
      recon_[p].height = snap.height[p];
    }
    intra_joint_ = snap.joint;
    uv_smooth_ = snap.uv;
    mv_grid_ = snap.mv;
    sink.state = snap.state;
    sink.blocks.resize(snap.blocks);
    sink.levels.resize(snap.levels);
    sink.palettes.resize(snap.palettes);
    sink.sub_pus.resize(snap.sub_pus);
    sink.leaf_dump.resize(snap.leaf_dump);
    sink.leaf_costs.resize(snap.leaf_costs);
    tail_floor_leaf_bits_.resize(snap.leaf_bits);
    tail_floor_leaf_bits_ok_.resize(snap.leaf_bits_ok);
    pipe_probe_ = snap.probe;
    rate_probe_ = snap.rate_probe;
  };

  const bool eff_on =
      !pipe_ba::enabled() && eff_perc::enabled() && pipe_probe_ready_;
  const double pipe_lambda =
      leaf_rd_lambda(current_qindex_, options_) *
      (pipe_ba::enabled() ? pipe_ba::config().lambda
                          : (eff_on ? eff_perc::lambda_scale()
                                    : pipe_perc::lambda_scale()));
  const double lambda = pipe_lambda * cfg.lambda;
  if (!(lambda > 0.0)) {
    ++stats.failed;
    *out = reference;
    return true;
  }

  struct PruneResult {
    PartitionNode node;
    double cost = partition::kInfiniteCost;
    bool ok = false;
  };
  // Measure the reference tree itself with the same objective (it is the
  // gate-off stream's own layout).  The search keeps it whenever it wins, so
  // the gate can never make the tree worse under J; a changed tree is then a
  // measured improvement, not a side effect of the expanded candidate set.
  std::function<bool(const PartitionNode&, const ChromaInfo&, double*)>
      measure_ref;
  measure_ref = [&](const PartitionNode& node, const ChromaInfo& inherited,
                    double* cost) -> bool {
    const ChromaInfo chroma = chroma_step(node, inherited);
    if (node.partition == PartitionType::kNone) {
      ++stats.layouts_measured;
      double leaf_cost = 0.0;
      if (!joint_p_measure_leaf(node, inherited, sink, lambda, &leaf_cost,
                                error)) {
        ++stats.failed;
        return false;
      }
      *cost = leaf_cost;
      return true;
    }
    double sum = lambda * 2.0;
    for (const PartitionNode& child : node.children) {
      double child_cost = 0.0;
      if (!measure_ref(child, chroma, &child_cost)) return false;
      sum += child_cost;
    }
    *cost = sum;
    return true;
  };
  const Snap ref_start = capture();
  double ref_cost = partition::kInfiniteCost;
  bool ref_ok = false;
  {
    joint_plan::ScopedTimer timer(&joint_plan::profile().ref_walk);
    ref_ok = measure_ref(reference, ChromaInfo{}, &ref_cost);
  }
  apply(ref_start);
  // Sequential prune: at every node the split children are measured in the
  // reconstruction the already-chosen preceding children built, so a
  // candidate is always priced in the recon it produces.  The NONE trial is
  // restored before the split trial; the winner is re-encoded only when it is
  // the leaf (the split trial's state is already the committed layout).
  std::function<PruneResult(const PartitionNode&, const ChromaInfo&)> prune;
  prune = [&](const PartitionNode& node,
              const ChromaInfo& inherited) -> PruneResult {
    PruneResult result;
    ++stats.nodes;
    const ChromaInfo chroma = chroma_step(node, inherited);
    if (node.partition == PartitionType::kNone) {
      double cost = 0.0;
      ++stats.layouts_measured;
      if (!joint_p_measure_leaf(node, inherited, sink, lambda, &cost,
                                error)) {
        ++stats.failed;
        result.node = node;
        return result;
      }
      result.node = node;
      result.cost = cost;
      result.ok = true;
      return result;
    }
    const Snap start = capture();
    const bool can_leaf =
        !node.forced && node.allowed.is_allowed(PartitionType::kNone);
    PartitionNode leaf = node;
    leaf.partition = PartitionType::kNone;
    leaf.forced = false;
    leaf.children.clear();
    leaf.cost = 0.0;
    double leaf_cost = partition::kInfiniteCost;
    bool leaf_ok = false;
    if (can_leaf) {
      ++stats.layouts_measured;
      leaf_ok =
          joint_p_measure_leaf(leaf, inherited, sink, lambda, &leaf_cost,
                               error);
      if (!leaf_ok) ++stats.failed;
    }
    apply(start);

    double split_cost = lambda * 2.0;
    bool split_ok = true;
    std::vector<PartitionNode> kids;
    for (const PartitionNode& child : node.children) {
      PruneResult child_result = prune(child, chroma);
      if (!child_result.ok) {
        split_ok = false;
        break;
      }
      split_cost += child_result.cost;
      kids.push_back(std::move(child_result.node));
    }
    if (split_ok) ++stats.split_trials;
    if (split_ok && (!leaf_ok || split_cost < leaf_cost)) {
      PartitionNode chosen = node;
      chosen.children = std::move(kids);
      chosen.cost = split_cost;
      ++stats.split_kept;
      result.node = std::move(chosen);
      result.cost = split_cost;
      result.ok = true;
      return result;
    }
    if (leaf_ok) {
      // Collapse: drop the measured children and commit the leaf instead.
      apply(start);
      double committed = 0.0;
      if (!joint_p_measure_leaf(leaf, inherited, sink, lambda, &committed,
                                error)) {
        ++stats.failed;
        result.node = node;
        return result;
      }
      ++stats.leaf_kept;
      result.node = std::move(leaf);
      result.cost = committed;
      result.ok = true;
      return result;
    }
    apply(start);
    ++stats.reference_kept;
    result.node = node;
    return result;
  };

  const Snap frame_start = capture();
  rd_driver::measuring_flag() = true;
  PruneResult chosen;
  {
    joint_plan::ScopedTimer timer(&joint_plan::profile().prune);
    chosen = prune(expanded, ChromaInfo{});
  }
  rd_driver::measuring_flag() = false;
  apply(frame_start);
  if (ref_ok) stats.ref_j += ref_cost;
  if (!ref_ok) {
    // The reference layout could not be measured (a compound/TIP carrier the
    // exact leaf cost does not cover).  Keep it; an unmeasured tree is not a
    // decision.
    ++stats.reference_kept;
    *out = reference;
    return true;
  }
  if (!chosen.ok || ref_cost <= chosen.cost) {
    // The gate-off layout is at least as good under the measured objective.
    ++stats.reference_wins;
    stats.chosen_j += ref_cost;
    *out = reference;
    return true;
  }
  if (!chosen.ok || !std::isfinite(chosen.cost)) {
    ++stats.reference_kept;
    *out = reference;
    return true;
  }
  std::string chosen_error;
  if (!partition::validate_tree(geometry_, chosen.node, chosen_error)) {
    ++stats.failed;
    *out = reference;
    return true;
  }
  // Reference/chosen leaf histograms for the Phase-1 tree report.
  const auto count_leaves = [](auto&& self, const PartitionNode& node,
                               long long (&out)[5]) -> void {
    if (node.partition != PartitionType::kNone) {
      for (const PartitionNode& child : node.children) {
        self(self, child, out);
      }
      return;
    }
    ++out[joint_plan::bucket(partition::block_width(node.size),
                             partition::block_height(node.size))];
  };
  long long ref_hist[5] = {0, 0, 0, 0, 0};
  long long chosen_hist[5] = {0, 0, 0, 0, 0};
  count_leaves(count_leaves, reference, ref_hist);
  count_leaves(count_leaves, chosen.node, chosen_hist);
  const bool head = arming_ordinal() <= 32;
  bool changed = false;
  for (int i = 0; i < 5; ++i) {
    stats.ref_leaves[i] += ref_hist[i];
    stats.chosen_leaves[i] += chosen_hist[i];
    (head ? stats.head_ref : stats.tail_ref)[i] += ref_hist[i];
    (head ? stats.head_chosen : stats.tail_chosen)[i] += chosen_hist[i];
    if (ref_hist[i] != chosen_hist[i]) changed = true;
  }
  if (changed) ++stats.layouts_changed;
  stats.chosen_j += chosen.cost;
  *out = std::move(chosen.node);
  return true;
}

}  // namespace av2enc::video
