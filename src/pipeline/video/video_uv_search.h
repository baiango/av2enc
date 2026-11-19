// Full chroma intra mode search for the M2 video key picture (AV2ENC_RD_UVFULL).
//
// The historical M2 video key picture codes every chroma leaf as DC_PRED plus
// the opt-in CfL hypothesis (`chroma_search::choose()`), while the M1 still
// product has a full UV search behind `--still-uv-full`
// (`Encoder::choose_uv_mode()` in src/pipeline/still/still_encoder_decision.cpp).
// At 240p30 / 100 kbps the key picture is ~76% of the bits and its chroma is
// DC-only, so this header ports that candidate set to the video key path.
//
// Candidate set (§5.20.5.6 UVMode_CFL_NOT_ALLOWED / §5.20.7.32):
//   * the 13 base modes DC, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH, V, H, D45,
//     D135, D67, D113, D157, D203.  AngleDeltaUV is *not* a separate symbol: it
//     is derived as AngleDeltaY only when UVMode == YMode and YMode is
//     directional, so each candidate is evaluated with that derived delta and
//     there is no UV angle-delta search (§6.19.5.6).
//   * the CfL hypothesis (CFL_DERIVED_ALPHA) is supplied by the caller as
//     ready-made predictions, because its derived-alpha derivation lives in
//     `chroma_search::choose()` (shared with the CUDA emulation path).
//
// Decision:
//   stage 1 prices every candidate by the eff-perc perceptual distortion of the
//   prediction (same weighted-CSF D the eff-perc leaf/TX surfaces use, no SSE
//   mixing) plus lambda * the uv_mode/is_cfl/cfl_index symbol information bits
//   (§8.2 -log2 p under the static §9.3 CDF rows the writer starts from).
//   stage 2 re-scores the cheapest `top_n` candidates on the real chroma
//   transform: D is the same perceptual distortion of the quantised
//   reconstruction and R is the static §8.3 coefficient information bits of
//   that plan plus the mode symbol bits.  Only the winner is returned; the
//   caller re-runs its own chroma TX on the winning prediction exactly as
//   before, so the leaf's transform path is unchanged.
//
// Scope: §7.13.2.12 IBP is excluded.  It changes the DC candidate (the
// Ibp_Weights post-filter) and the video CLI already rejects --video-ibp with
// --video-cfl, so the caller only runs this search when IBP is off.
//
// Default off: the header is only instantiated from the AV2ENC_RD_UVFULL branch
// of `video_frame_decision.cpp`, so every historical stream keeps its bytes.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "codec/intra/modes/intra_modes.h"
#include "pipeline/still/still_cdf_tables.h"
#include "pipeline/video/video_eff_perc.h"
#include "pipeline/video/video_intra_search.h"

namespace av2enc::video::uv_search {

// AV2 YMode / UVMode values (§5.20.5.6).  The writer header that owns these
// names is private to the syntax TUs, so the values are repeated here against
// `intra::Mode`, which uses the same numbering.
constexpr int kDcPred = 0;
constexpr int kSmoothPred = 9;

inline bool y_mode_is_directional(int mode) {
  return mode >= 1 && mode <= 8;  // V..D67
}

// Default_Mode_List_Uv[UV_INTRA_MODES_CFL_NOT_ALLOWED] (§5.20.5.6):
// DC, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH, V, H, D45, D135, D67, D113, D157,
// D203.
inline int uv_mode_index(int uv_mode, int y_mode) {
  static const int kList[13] = {0, 9, 10, 11, 12, 1, 2, 3, 4, 8, 5, 6, 7};
  int index = 0;
  if (y_mode_is_directional(y_mode)) {
    if (uv_mode == y_mode) return 0;
    index = 1;
  }
  for (int i = 0; i < 13; ++i) {
    const int mode = kList[i];
    if (mode == y_mode && y_mode_is_directional(y_mode)) continue;
    if (mode == uv_mode) return index;
    ++index;
  }
  return 0;
}

// §8.2 information cost of one symbol under a forward-cumulative tile CDF
// (`cdf[0 .. n-2]` boundaries, the implicit last boundary is 32768).
inline double cdf_symbol_bits(const std::uint16_t* cdf, int n, int symbol) {
  if (cdf == nullptr || n < 2 || symbol < 0 || symbol >= n) return 0.0;
  const int lo = symbol == 0 ? 0 : static_cast<int>(cdf[symbol - 1]);
  const int hi = symbol == n - 1 ? 32768 : static_cast<int>(cdf[symbol]);
  const int freq = hi > lo ? hi - lo : 1;
  return -std::log2(static_cast<double>(freq) / 32768.0);
}

// uv_mode / is_cfl / cfl_index symbol cost.  The static §9.3 rows are the ones
// `video_syntax_writer.h` copies into the tile CDF image, so this is the
// information cost the writer charges before adaptation.
inline double mode_symbol_bits(int uv_mode, int y_mode, bool is_cfl,
                               int cfl_index) {
  // The writer only has the two-symbol cfl_index form without MHCCP
  // (`sym(&cdfs_.cfl_index[0], 2, node.cfl_index)`), which is the derived
  // alpha this search emits.
  if (is_cfl) {
    const double is_cfl_bits =
        cdf_symbol_bits(&::av2enc::still::kIsCflCdf[0][0], 2, 1);
    return is_cfl_bits +
           cdf_symbol_bits(&::av2enc::still::kCflIndexCdf[0], 2,
                           cfl_index == 0 ? 0 : 1);
  }
  const int ctx = y_mode_is_directional(y_mode) ? 1 : 0;
  const int index = uv_mode_index(uv_mode, y_mode);
  double bits = cdf_symbol_bits(&::av2enc::still::kUvModeCflNotAllowedCdf[ctx][0],
                                8, std::min(index, 7));
  if (index > 6) bits += 3.0;  // write_literal(3, index - 7)
  return bits;
}

// Clamped view of one source plane at the tile origin (tile-local chroma
// coordinates, same clamping as `chroma_search::source_sample`).
struct PlaneSrc {
  const std::uint8_t* data = nullptr;
  int stride = 0;
  int max_x = 0;
  int max_y = 0;

  int sample(int x, int y) const {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > max_x) x = max_x;
    if (y > max_y) y = max_y;
    return data[static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(x)];
  }
};

struct Params {
  // Coding-block origin and size in tile-local chroma samples.
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int y_mode = 0;
  int y_angle_delta = 0;
  // §5.20.9.1 availability of the chroma plane (counts are already in chroma
  // 4x4 units, derived from the chroma BlockDecoded map).
  intra_search::Availability avail;
  // §7.13.2.15/.16 filterTypeAbove/Left inputs for the chroma neighbours.
  bool above_smooth = false;
  bool left_smooth = false;
  // The sequence header always signals enable_intra_edge_filter = 1.
  bool enable_edge_filter = true;
  // Reconstruction planes (tile-local, coded size) and their strides.
  const std::uint8_t* recon[2] = {nullptr, nullptr};
  int recon_stride[2] = {0, 0};
  // Source planes.
  PlaneSrc src[2];
  // eff-perc distortion knobs: D = sum(w * (e^2 + g1 * mean3x3(e)^2 +
  // g2 * mean7x7(e)^2)) with the neutral Q8 weight map, exactly what
  // `sse_pipe` charges chroma on the eff-perc path.
  double csf_gain = 0.0;
  double csf2_gain = 0.0;
  double chroma_weight = 1.0;
  // Visible rows of the chroma block (--video-edge-mask); -1 = all rows.
  int mask_row = -1;
  // D units per bit for the mode symbols.
  double lambda = 0.0;
  // Stage-2 re-score count (0 disables the transform re-score).
  int top_n = 3;
  // Chroma coefficient-level RD (AV2ENC_RD_UVLEVEL): the number of trellis
  // proposals per plane on top of the scalar plan.  0 keeps the M4 behaviour.
  int level_variants = 0;
  // D units per bit for choosing between coefficient-level proposals.  The
  // level site is calibrated separately from the mode site (`lambda`): the
  // static coefficient model overprices chroma levels, so its measured weight
  // is lower.  A negative value keeps `lambda` (M4 behaviour).
  double level_lambda = -1.0;
  // Optional CfL candidate (caller supplies the derived-alpha prediction).
  bool have_cfl = false;
  std::vector<std::uint8_t> cfl_u;
  std::vector<std::uint8_t> cfl_v;
  int cfl_index = 1;
  int cfl_mh_dir = 0;
};

struct Result {
  int mode = kDcPred;
  int angle_delta = 0;
  bool is_cfl = false;
  int cfl_index = 1;
  int cfl_mh_dir = 0;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  // Winning chroma plans from stage 2 (including a winning level variant).
  // `have_plans` is false on the prediction-only path (top_n == 0), where the
  // caller must run its own transform.  When true the caller can emit these
  // plans directly, so the coded coefficients are exactly the scored ones.
  TxPlan u_plan;
  TxPlan v_plan;
  bool have_plans = false;
  double cost = 0.0;
  double distortion = 0.0;
  double bits = 0.0;
  int candidates = 0;
  int tx_rescored = 0;
  int level_plans = 0;
  int level_wins = 0;  // winning plans that replaced the scalar plan
};

namespace detail {

// Perceptual distortion of one w*h predictor/reconstruction, using the eff-perc
// error-domain CSF on a tightly packed source block (the caller's clamped copy,
// so out-of-picture reads never happen).
inline double perceptual_d(const Params& p,
                           const std::vector<std::uint8_t>& src,
                           const std::vector<std::uint8_t>& rec) {
  if (src.size() != static_cast<std::size_t>(p.w) *
                        static_cast<std::size_t>(p.h) ||
      rec.size() != src.size()) {
    return 0.0;
  }
  const int stride = p.w;
  return eff_perc::weighted_sse(
             /*weights=*/nullptr, p.w, p.h, src.data(), stride, 0, 0,
             rec.data(), stride, 0, 0, p.mask_row, p.csf_gain, p.csf2_gain) *
         p.chroma_weight;
}

}  // namespace detail

// Runs the search with an optional coefficient-level stage.
//
// `tx(plane, pred, plan)` produces the real chroma transform plan for a
// prediction (the caller's compute_tx).  `coef_bits(plane, plan)` returns the
// static §8.3 coefficient information bits of that plan.  `level(plane, pred,
// scalar, variant, out)` proposes an alternative coefficient set for the same
// prediction (the chroma trellis, variant 1..`p.level_variants`); returning
// false keeps the scalar plan.  A proposal is only a candidate: it is scored
// on the same D_perc + lambda * R as the scalar plan, so a proposal can never
// win on its own model.
template <typename TxFn, typename BitsFn, typename LevelFn>
bool search_with_levels(const Params& p, TxFn&& tx, BitsFn&& coef_bits,
                        LevelFn&& level, Result& out, std::string& error) {
  out = Result{};
  if (p.w <= 0 || p.h <= 0 || p.w > 32 || p.h > 32 ||
      p.recon[0] == nullptr || p.recon[1] == nullptr || p.src[0].data == nullptr ||
      p.src[1].data == nullptr) {
    error = "uv search: invalid block descriptor";
    return false;
  }
  const std::size_t count =
      static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h);

  // Clamped source blocks for the distortion term.
  std::vector<std::uint8_t> src_block[2];
  for (int plane = 0; plane < 2; ++plane) {
    src_block[plane].resize(count);
    for (int y = 0; y < p.h; ++y) {
      for (int x = 0; x < p.w; ++x) {
        src_block[plane][static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(p.w) +
                         static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(p.src[plane].sample(p.x + x, p.y + y));
      }
    }
  }

  // §7.13.2.1 edges: identical for every candidate of a plane, so prepare once.
  intra::IntraEdges edges[2];
  for (int plane = 0; plane < 2; ++plane) {
    intra::NeighborRequest request;
    request.recon = intra::PlaneView{p.recon[plane], p.recon_stride[plane]};
    request.x = p.x;
    request.y = p.y;
    request.w = p.w;
    request.h = p.h;
    request.max_x = p.avail.max_x;
    request.max_y = p.avail.max_y;
    request.have_left = p.avail.have_left != 0;
    request.have_above = p.avail.have_above != 0;
    request.num4_above_right = p.avail.num4_above_right;
    request.num4_below_left = p.avail.num4_below_left;
    request.color_plane = plane + 1;
    if (intra::prepare_edges(request, 8, edges[plane]) !=
        intra::IntraStatus::kOk) {
      error = "uv search: prepare_edges failed";
      return false;
    }
  }

  struct Candidate {
    int mode = kDcPred;
    int angle_delta = 0;
    bool is_cfl = false;
    int cfl_index = 1;
    int cfl_mh_dir = 0;
    std::vector<std::uint8_t> u;
    std::vector<std::uint8_t> v;
    TxPlan u_plan;
    TxPlan v_plan;
    double pred_cost = 0.0;
    double cost = 0.0;
    double distortion = 0.0;
    double bits = 0.0;
    double stage1 = 0.0;
    bool scored = false;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(16);

  std::vector<std::uint8_t> pred_u(count);
  std::vector<std::uint8_t> pred_v(count);
  for (int mode = 0; mode < intra::kNumModes; ++mode) {
    // §6.19.5.6: AngleDeltaUV is derived from AngleDeltaY only when
    // UVMode == YMode and YMode is directional; every other pair uses 0.
    const int delta = (y_mode_is_directional(p.y_mode) && mode == p.y_mode)
                          ? p.y_angle_delta
                          : 0;
    Candidate candidate;
    candidate.mode = mode;
    candidate.angle_delta = delta;
    candidate.bits =
        mode_symbol_bits(mode, p.y_mode, /*is_cfl=*/false, /*cfl_index=*/1);
    bool ok = true;
    for (int plane = 0; plane < 2 && ok; ++plane) {
      std::uint8_t* dst = plane == 0 ? pred_u.data() : pred_v.data();
      intra::PredictParams params;
      params.mode = static_cast<intra::Mode>(mode);
      params.angle_delta = delta;
      params.enable_edge_filter = p.enable_edge_filter;
      params.above_neighbor_is_smooth = p.above_smooth;
      params.left_neighbor_is_smooth = p.left_smooth;
      ok = intra::predict_intra(edges[plane], params, dst,
                                static_cast<std::size_t>(p.w)) ==
           intra::IntraStatus::kOk;
    }
    if (!ok) {
      error = "uv search: predict_intra failed";
      return false;
    }
    candidate.u = pred_u;
    candidate.v = pred_v;
    candidate.distortion =
        detail::perceptual_d(p, src_block[0], candidate.u) +
        detail::perceptual_d(p, src_block[1], candidate.v);
    candidate.stage1 = candidate.distortion + p.lambda * candidate.bits;
    ++out.candidates;
    candidates.push_back(std::move(candidate));
  }

  if (p.have_cfl && p.cfl_u.size() == count && p.cfl_v.size() == count) {
    Candidate candidate;
    candidate.mode = kDcPred;  // CfL keeps UVMode = DC_PRED in the syntax
    candidate.is_cfl = true;
    candidate.cfl_index = p.cfl_index;
    candidate.cfl_mh_dir = p.cfl_mh_dir;
    candidate.u = p.cfl_u;
    candidate.v = p.cfl_v;
    candidate.bits = mode_symbol_bits(kDcPred, p.y_mode, /*is_cfl=*/true,
                                      p.cfl_index);
    candidate.distortion =
        detail::perceptual_d(p, src_block[0], candidate.u) +
        detail::perceptual_d(p, src_block[1], candidate.v);
    candidate.stage1 = candidate.distortion + p.lambda * candidate.bits;
    ++out.candidates;
    candidates.push_back(std::move(candidate));
  }

  if (candidates.empty()) {
    error = "uv search: no candidates";
    return false;
  }

  for (Candidate& candidate : candidates) {
    candidate.cost = candidate.stage1;
    candidate.scored = false;
  }

  // Stage 2: re-score the cheapest `top_n` on the real transform.  Ties keep
  // the earlier candidate, matching the still encoder's strict `<` rule.
  const int top_n =
      std::min(std::max(p.top_n, 0), static_cast<int>(candidates.size()));
  if (top_n > 0) {
    std::vector<int> order(candidates.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
      order[i] = static_cast<int>(i);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) {
                       return candidates[static_cast<std::size_t>(a)].stage1 <
                              candidates[static_cast<std::size_t>(b)].stage1;
                     });
    for (int rank = 0; rank < top_n; ++rank) {
      Candidate& candidate =
          candidates[static_cast<std::size_t>(order[static_cast<std::size_t>(
              rank)])];
      double distortion = 0.0;
      double bits = candidate.bits;
      for (int plane = 0; plane < 2; ++plane) {
        const std::vector<std::uint8_t>& pred =
            plane == 0 ? candidate.u : candidate.v;
        TxPlan scalar;
        if (!tx(plane, pred, scalar)) {
          error = "uv search: chroma transform failed";
          return false;
        }
        TxPlan best = scalar;
        double best_distortion =
            detail::perceptual_d(p, src_block[plane], scalar.recon);
        double best_coef_bits = 0.0;
        if (!coef_bits(plane, scalar, best_coef_bits)) best_coef_bits = 0.0;
        const double level_lambda =
            p.level_lambda >= 0.0 ? p.level_lambda : p.lambda;
        double best_cost = best_distortion + level_lambda * best_coef_bits;
        bool level_won = false;
        for (int variant = 1; variant <= p.level_variants; ++variant) {
          TxPlan proposal;
          if (!level(plane, pred, scalar, variant, proposal)) continue;
          ++out.level_plans;
          const double d =
              detail::perceptual_d(p, src_block[plane], proposal.recon);
          double r = 0.0;
          if (!coef_bits(plane, proposal, r)) r = 0.0;
          const double cost = d + level_lambda * r;
          if (cost < best_cost) {
            best_cost = cost;
            best_distortion = d;
            best_coef_bits = r;
            best = std::move(proposal);
            level_won = true;
          }
        }
        if (level_won) ++out.level_wins;
        distortion += best_distortion;
        bits += best_coef_bits;
        if (plane == 0) {
          candidate.u_plan = std::move(best);
        } else {
          candidate.v_plan = std::move(best);
        }
      }
      candidate.distortion = distortion;
      candidate.bits = bits;
      candidate.cost = distortion + p.lambda * bits;
      candidate.scored = true;
      ++out.tx_rescored;
    }
    // Only the re-scored candidates are compared: a stage-1 prediction cost is
    // not on the same scale as a reconstruction cost (quantisation lowers D),
    // so the proxy may rank but never win.
    const Candidate* winner = nullptr;
    for (const Candidate& candidate : candidates) {
      if (!candidate.scored) continue;
      if (winner == nullptr || candidate.cost < winner->cost) {
        winner = &candidate;
      }
    }
    if (winner == nullptr) {
      error = "uv search: no re-scored candidate";
      return false;
    }
    out.mode = winner->mode;
    out.angle_delta = winner->angle_delta;
    out.is_cfl = winner->is_cfl;
    out.cfl_index = winner->cfl_index;
    out.cfl_mh_dir = winner->cfl_mh_dir;
    out.u = winner->u;
    out.v = winner->v;
    out.u_plan = winner->u_plan;
    out.v_plan = winner->v_plan;
    out.have_plans = true;
    out.cost = winner->cost;
    out.distortion = winner->distortion;
    out.bits = winner->bits;
    return true;
  }

  const Candidate* winner = &candidates.front();
  for (const Candidate& candidate : candidates) {
    if (candidate.cost < winner->cost) winner = &candidate;
  }
  out.mode = winner->mode;
  out.angle_delta = winner->angle_delta;
  out.is_cfl = winner->is_cfl;
  out.cfl_index = winner->cfl_index;
  out.cfl_mh_dir = winner->cfl_mh_dir;
  out.u = winner->u;
  out.v = winner->v;
  out.cost = winner->cost;
  out.distortion = winner->distortion;
  out.bits = winner->bits;
  return true;
}

// Prediction + transform search without a coefficient-level stage.  Keeps the
// M4 call surface (and the `p.level_variants == 0` behaviour) unchanged.
template <typename TxFn, typename BitsFn>
bool search(const Params& p, TxFn&& tx, BitsFn&& coef_bits, Result& out,
            std::string& error) {
  const auto no_levels = [](int, const std::vector<std::uint8_t>&,
                            const TxPlan&, int, TxPlan&) { return false; };
  return search_with_levels(p, tx, coef_bits, no_levels, out, error);
}

}  // namespace av2enc::video::uv_search
