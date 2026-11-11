#include "encoder/m2/advanced/advanced_decision.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace av2enc::m2::adv {
namespace {

std::int64_t max_sse() { return std::numeric_limits<std::int64_t>::max(); }

int64_t square(int64_t v) { return v * v; }

void copy_rect(const std::uint8_t* src, int src_stride, std::uint8_t* dst,
               int dst_stride, int x, int y, int w, int h) {
  for (int i = 0; i < h; ++i) {
    std::memcpy(dst + static_cast<std::ptrdiff_t>(y + i) * dst_stride + x,
                src + static_cast<std::ptrdiff_t>(y + i) * src_stride + x,
                static_cast<std::size_t>(w));
  }
}

// SSE of one warp candidate.  `warp_block_sse` uses the block-warp pass; when
// §7.13.3.14 forces extended warp we must run that pass instead.
int64_t warp_sse(const ia::WarpParams& params, const WarpSearchInput& in) {
  if (in.scratch == nullptr || in.source == nullptr || in.scratch_stride <= 0 ||
      in.source_stride <= 0) {
    return -1;
  }
  const bool extended = ia::warp_needs_extended(params, in.geom);
  const bool ok = extended
                      ? ia::warp_predict_ext(params, in.ref, in.geom, in.compound,
                                             in.scratch, in.scratch_stride)
                      : ia::warp_predict(params, in.ref, in.geom, in.compound,
                                         in.scratch, in.scratch_stride);
  if (!ok) return -1;
  int64_t sse = 0;
  for (int i = 0; i < in.geom.rect.h; ++i) {
    for (int j = 0; j < in.geom.rect.w; ++j) {
      const int raw = in.scratch[i * in.scratch_stride + j];
      const int a = in.compound ? ia::clip1(ia::round2(raw, 4), 8) : raw;
      const int b = in.source[i * in.source_stride + j];
      const int d = a - b;
      sse += d * d;
    }
  }
  return sse;
}

ia::WarpParams reduce_for_compare(const ia::WarpParams& target, bool six_param) {
  ia::WarpParams t = target;
  if (!six_param) {
    t.p[4] = -t.p[3];
    t.p[5] = t.p[2];
  }
  return ia::reduce_warp_model(t);
}

}  // namespace

// ---------------------------------------------------------------------------
// Warp delta
// ---------------------------------------------------------------------------

WmParams warpmv_local_model(const WmParams& candidate, int mid_x, int mid_y,
                            const int block_mv[2]) {
  WmParams p = ia::reduce_warp_model(candidate);
  p.p[0] = 0;
  p.p[1] = 0;
  if (block_mv != nullptr) {
    ia::get_warp_translation(p, mid_x, mid_y, block_mv);
  }
  return p;
}

int warp_delta_value(const WarpDeltaSymbols& symbols, int index) {
  if (index < 0 || index > 3) return 0;
  const int step_bits =
      kWarpDeltaStepBits + 1 - (symbols.high_precision ? 1 : 0);
  const int value = symbols.magnitude[index] << step_bits;
  return symbols.negative[index] ? -value : value;
}

ia::WarpParams apply_warp_delta(const ia::WarpParams& base,
                                const WarpDeltaSymbols& symbols,
                                bool six_param, int mid_x, int mid_y,
                                const int block_mv[2]) {
  ia::WarpParams p = base;
  p.p[0] = 0;
  p.p[1] = 0;
  p.p[2] += warp_delta_value(symbols, 0);
  p.p[3] += warp_delta_value(symbols, 1);
  if (six_param) {
    p.p[4] += warp_delta_value(symbols, 2);
    p.p[5] += warp_delta_value(symbols, 3);
  } else {
    p.p[4] = -p.p[3];
    p.p[5] = p.p[2];
  }
  p = ia::reduce_warp_model(p);
  if (block_mv != nullptr) {
    ia::get_warp_translation(p, mid_x, mid_y, block_mv);
  } else {
    p.p[0] = 0;
    p.p[1] = 0;
  }
  return p;
}

WarpDeltaQuantization quantize_warp_delta(const ia::WarpParams& base,
                                          const ia::WarpParams& target,
                                          bool high_precision, bool six_param,
                                          int mid_x, int mid_y,
                                          const int block_mv[2]) {
  WarpDeltaQuantization out{};
  out.symbols.high_precision = high_precision;
  const int step = high_precision ? kWarpDeltaStepHigh : kWarpDeltaStepStandard;
  const int max_value = high_precision ? 14 : 7;
  const int count = six_param ? 4 : 2;
  for (int i = 0; i < count; ++i) {
    const int diff = target.p[2 + i] - base.p[2 + i];
    const int magnitude =
        std::min(max_value, (std::abs(diff) + step / 2) / step);
    out.symbols.magnitude[i] = magnitude;
    out.symbols.negative[i] = diff < 0 && magnitude != 0;
  }
  out.model = apply_warp_delta(base, out.symbols, six_param, mid_x, mid_y,
                               block_mv);
  const ia::WarpParams compared = reduce_for_compare(target, six_param);
  for (int i = 2; i < 6; ++i) {
    out.param_error += square(out.model.p[i] - compared.p[i]);
  }
  return out;
}

WarpDeltaChoice choose_warp_delta_precision(const ia::WarpParams& base,
                                            const ia::WarpParams& target,
                                            bool six_param, int mid_x, int mid_y,
                                            const int block_mv[2]) {
  WarpDeltaChoice best{};
  best.high_precision = false;
  best.quantization = quantize_warp_delta(base, target, false, six_param, mid_x,
                                          mid_y, block_mv);
  WarpDeltaQuantization high = quantize_warp_delta(base, target, true, six_param,
                                                   mid_x, mid_y, block_mv);
  if (high.param_error < best.quantization.param_error) {
    best.high_precision = true;
    best.quantization = high;
  }
  return best;
}

bool warp_requires_extended(const ia::WarpParams& params,
                            const ia::WarpBlockGeometry& geom) {
  return ia::warp_needs_extended(params, geom);
}

WarpCodingDecision select_warp_coding(const WarpSearchInput& in) {
  WarpCodingDecision out{};
  if (in.candidates == nullptr || in.num_candidates <= 0 ||
      in.num_candidates > kMaxWarpRefCandidates) {
    return out;
  }

  int best_candidate = -1;
  int64_t best_direct = max_sse();
  WmParams best_direct_model{};
  for (int idx = 0; idx < in.num_candidates; ++idx) {
    // WARPMV uses the reduced candidate model with its translation re-derived
    // from the assigned MV (read_warp_delta()).
    const WmParams model =
        in.block_mv != nullptr
            ? warpmv_local_model(in.candidates[idx], in.mid_x, in.mid_y,
                                 in.block_mv)
            : in.candidates[idx];
    const int64_t sse = warp_sse(model, in);
    if (sse >= 0 && sse < best_direct) {
      best_direct = sse;
      best_candidate = idx;
      best_direct_model = model;
    }
  }
  if (best_candidate < 0) return out;

  out.use_warp = true;
  out.warp_idx = best_candidate;
  out.warpmv = true;
  out.model = best_direct_model;
  out.sse = best_direct;

  if (in.target == nullptr || in.block_mv == nullptr) {
    return out;
  }

  // Delta coding is only legal from an existing candidate.  Six-parameter
  // deltas are only legal from RefWarpIdx == 1.
  int64_t best_delta_sse = max_sse();
  int best_delta_idx = -1;
  bool best_six = false;
  bool best_high = false;
  WarpDeltaSymbols best_symbols{};
  ia::WarpParams best_model{};
  for (int idx = 0; idx < in.num_candidates; ++idx) {
    for (int six = 0; six <= (in.enable_six_param && idx == 1 ? 1 : 0); ++six) {
      const WarpDeltaChoice choice =
          choose_warp_delta_precision(in.candidates[idx], *in.target,
                                      six != 0, in.mid_x, in.mid_y, in.block_mv);
      const int64_t sse = warp_sse(choice.quantization.model, in);
      if (sse < 0 || sse >= best_delta_sse) continue;
      best_delta_sse = sse;
      best_delta_idx = idx;
      best_six = six != 0;
      best_high = choice.high_precision;
      best_symbols = choice.quantization.symbols;
      best_model = choice.quantization.model;
    }
  }

  if (best_delta_idx >= 0 && best_delta_sse < best_direct) {
    out.warpmv = false;
    out.warp_idx = best_delta_idx;
    out.six_param = best_six;
    out.high_precision = best_high;
    out.delta = best_symbols;
    out.model = best_model;
    out.sse = best_delta_sse;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Inter-intra
// ---------------------------------------------------------------------------

InterIntraCodingDecision decide_inter_intra(const InterIntraCodingInput& in) {
  InterIntraCodingDecision out{};
  if (in.is_warp && !in.warp_inter_intra_allowed) return out;
  if (in.search.bsize < 0 || in.search.bsize >= kBlockSizes) return out;
  if (in.is_warp) {
    const int w = partition::block_width(
        static_cast<partition::BlockSize>(in.search.bsize));
    const int h = partition::block_height(
        static_cast<partition::BlockSize>(in.search.bsize));
    if (w > 64 || h > 64) return out;  // warp_inter_intra is not codable
  }

  auto best = ia::search_inter_intra(in.search);
  if (!best.has_value()) return out;

  out.interintra_mode = static_cast<int>(best->mode);
  out.wedge_interintra = best->wedge_interintra;
  out.wedge_index = best->wedge_index;
  out.wedge_sign = 0;  // §5.20.7.15 read_interintra_mode()
  out.sse = best->sse;
  out.enabled = best->sse < in.inter_sse;
  return out;
}

void fill_inter_intra_symbols(const InterIntraCodingDecision& decision,
                              MotionModeSymbols* symbols) {
  if (symbols == nullptr) return;
  symbols->inter_intra = decision.enabled;
  symbols->interintra_mode = decision.interintra_mode;
  symbols->wedge_interintra = decision.wedge_interintra;
  symbols->wedge_index = decision.wedge_index;
}

void build_inter_intra_weights(int mode, int w, int h, std::uint8_t* out,
                               int stride) {
  if (out == nullptr || w <= 0 || h <= 0 || stride < w || mode < 0 || mode > 3) {
    return;
  }
  const auto enum_mode = static_cast<ia::InterIntraMode>(mode);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      out[static_cast<std::ptrdiff_t>(i) * stride + j] =
          static_cast<std::uint8_t>(
              ia::intra_mode_mask_value(enum_mode, w, h, i, j));
    }
  }
}

bool build_wedge_weights(int bsize, int wedge_sign, int wedge_index,
                         std::uint8_t* out, int stride) {
  if (out == nullptr || !ia::bsize_has_wedge(bsize) ||
      !ia::wedge_index_signalable(wedge_index)) {
    return false;
  }
  ia::build_wedge_mask(bsize, wedge_sign, wedge_index, out, stride);
  return true;
}

// ---------------------------------------------------------------------------
// BAWP
// ---------------------------------------------------------------------------

int explicit_bawp_alpha(int list_index, int first_ref_dist,
                        bool explicit_bawp_scale) {
  int scale = list_index + 1;
  if (first_ref_dist > 4) ++scale;
  if (!explicit_bawp_scale) scale = -scale;
  return 256 + 16 * scale;
}

int bawp_alpha_for_plane(int plane, int luma_alpha, int count) {
  if (plane <= 0) return luma_alpha;
  return count == 0 ? 256 : luma_alpha;
}

BawpDecision decide_bawp(const BawpDecisionInput& in) {
  BawpDecision out{};
  const ia::BawpJob& job = in.job;
  const int w = job.rect.w;
  const int h = job.rect.h;

  out.first_ref_dist =
      std::abs(ia::get_relative_dist(in.ref_order_hint, in.order_hint));
  const bool is_near = in.y_mode == static_cast<int>(YMode::kNearMv);
  const bool is_new = in.y_mode == static_cast<int>(YMode::kNewMv);
  out.list_index = ia::bawp_list_index(is_near, is_new, in.use_amvd);
  out.explicit_ctx =
      is_near ? 0 : ((is_new && in.use_amvd) ? 1 : 2);

  const bool conditions =
      in.allow_bawp && !in.ref_scaled && !in.frame_is_switch &&
      in.y_mode >= static_cast<int>(YMode::kNearMv) &&
      in.y_mode <= static_cast<int>(YMode::kNewMv) &&
      in.y_mode != static_cast<int>(YMode::kGlobalMv) && w >= 8 && h >= 8 &&
      in.source != nullptr && in.scratch != nullptr && in.source_stride > 0 &&
      in.scratch_stride >= job.rect.x + w && job.cur.data != nullptr;
  if (!conditions) {
    return out;
  }

  out.sse_off = ia::block_sse(in.source, in.source_stride, 0, 0, job.cur.data,
                              job.cur.stride, job.rect.x, job.rect.y, w, h);
  out.sse_best = out.sse_off;

  auto evaluate = [&](bool explicit_bawp, bool scale, int* alpha,
                      int* count) -> int64_t {
    copy_rect(job.cur.data, job.cur.stride, in.scratch, in.scratch_stride,
              job.rect.x, job.rect.y, w, h);
    ia::BawpJob candidate = job;
    candidate.dst = ia::MutablePlaneView{in.scratch, in.scratch_stride, 0, 0};
    candidate.explicit_bawp = explicit_bawp;
    candidate.explicit_bawp_scale = scale;
    candidate.list_index = out.list_index;
    candidate.first_ref_dist = out.first_ref_dist;
    ia::BawpFit fit{};
    if (!ia::bawp_apply_block(candidate, &fit)) return -1;
    *alpha = fit.alpha;
    *count = fit.count;
    return ia::block_sse(in.source, in.source_stride, 0, 0, in.scratch,
                         in.scratch_stride, job.rect.x, job.rect.y, w, h);
  };

  int best_alpha = 256;
  int best_count = 0;
  int alpha = 256;
  int count = 0;

  const int64_t fitted = evaluate(false, false, &alpha, &count);
  if (fitted >= 0 && fitted < out.sse_best) {
    out.sse_best = fitted;
    out.use_bawp = true;
    out.explicit_bawp = false;
    out.explicit_bawp_scale = false;
    best_alpha = alpha;
    best_count = count;
  }

  for (int scale = 0; scale <= 1; ++scale) {
    const int64_t sse = evaluate(true, scale != 0, &alpha, &count);
    if (sse >= 0 && sse < out.sse_best) {
      out.sse_best = sse;
      out.use_bawp = true;
      out.explicit_bawp = true;
      out.explicit_bawp_scale = scale != 0;
      best_alpha = alpha;
      best_count = count;
    }
  }

  out.luma_alpha = best_alpha;
  out.chroma_alpha =
      bawp_alpha_for_plane(1, best_alpha, best_count);

  if (!out.use_bawp || !in.has_chroma || in.chroma_job == nullptr ||
      in.chroma_source == nullptr || in.chroma_scratch == nullptr) {
    out.chroma_alpha = bawp_alpha_for_plane(1, best_alpha, best_count);
    return out;
  }

  const ia::BawpJob& cj = *in.chroma_job;
  const int cw = cj.rect.w;
  const int ch = cj.rect.h;
  if (cw <= 0 || ch <= 0 || cj.cur.data == nullptr ||
      in.chroma_scratch_stride < cj.rect.x + cw) {
    return out;
  }
  const int64_t chroma_off =
      ia::block_sse(in.chroma_source, in.chroma_source_stride, 0, 0, cj.cur.data,
                    cj.cur.stride, cj.rect.x, cj.rect.y, cw, ch);
  copy_rect(cj.cur.data, cj.cur.stride, in.chroma_scratch,
            in.chroma_scratch_stride, cj.rect.x, cj.rect.y, cw, ch);
  ia::BawpJob cjob = cj;
  cjob.dst = ia::MutablePlaneView{in.chroma_scratch, in.chroma_scratch_stride, 0,
                                  0};
  cjob.bawp_alpha = best_alpha;
  cjob.explicit_bawp = false;
  cjob.explicit_bawp_scale = false;
  ia::BawpFit chroma_fit{};
  if (ia::bawp_apply_block(cjob, &chroma_fit)) {
    const int64_t chroma_on = ia::block_sse(
        in.chroma_source, in.chroma_source_stride, 0, 0, in.chroma_scratch,
        in.chroma_scratch_stride, cj.rect.x, cj.rect.y, cw, ch);
    out.use_bawp_chroma = chroma_on < chroma_off;
    out.chroma_alpha = bawp_alpha_for_plane(1, best_alpha, chroma_fit.count);
  }
  return out;
}

BawpSymbols bawp_symbols_from_decision(const BawpDecision& decision) {
  BawpSymbols symbols{};
  symbols.use_bawp = decision.use_bawp;
  symbols.explicit_bawp = decision.explicit_bawp;
  symbols.explicit_bawp_scale = decision.explicit_bawp_scale;
  symbols.use_bawp_chroma = decision.use_bawp_chroma;
  symbols.explicit_ctx = decision.explicit_ctx;
  return symbols;
}

void apply_bawp_decision_to_job(const BawpDecision& decision, bool chroma,
                                ia::BawpJob* job) {
  if (job == nullptr) return;
  if (chroma) {
    job->bawp_alpha = decision.luma_alpha;
    job->explicit_bawp = false;
    job->explicit_bawp_scale = false;
    job->list_index = decision.list_index;
    job->first_ref_dist = decision.first_ref_dist;
  } else {
    job->explicit_bawp = decision.explicit_bawp;
    job->explicit_bawp_scale = decision.explicit_bawp_scale;
    job->list_index = decision.list_index;
    job->first_ref_dist = decision.first_ref_dist;
    job->bawp_alpha = 256;
  }
}

}  // namespace av2enc::m2::adv
