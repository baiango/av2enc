#include "pipeline/video/video_frame_internal.h"

#include <algorithm>
#include <numeric>

namespace av2enc::video {
namespace {

// AV2ENC_RD_KEY_TX (default off): one J for the key picture's intra luma
// transform plan.  The historical key-in-TX path ranks partitions by plain
// coded SSE plus an estimated rate; this gate ranks them by the eff-perc key
// distortion and the exact §8.3 plan rate, screens every candidate cheaply and
// re-scores the top-K exactly (docs/rd-lane3.md).
struct KeyTxRdConfig {
  bool on = false;
  int top_k = 4;
  bool level = true;
  // The measured matched-K point is 0.25 (docs/rd-lane3.md): the key TX
  // weight is a content-calibrated fraction of the inter/eff-perc weight.
  double tx_scale = 0.25;
  // < 0 selects 0.25 * rdoq_sse_scale_at(): the measured key/inter ratio.
  double level_scale = -1.0;
  // Score the exact-stage plan distortion with the eff-perc key CSF (1) or
  // plain coded SSE (0).  The base partition cost is plain SSE; the CSF
  // weights low-band error and therefore tolerates more high-frequency error,
  // which is the band inter prediction reads.
  double plan_csf = 1.0;
  // Report the plan histogram even on the historical (gate-off) path.
  bool stats = false;
};

inline const KeyTxRdConfig& key_tx_rd_config() {
  static const KeyTxRdConfig cfg = [] {
    KeyTxRdConfig out;
    if (const char* on = std::getenv("AV2ENC_RD_KEY_TX")) {
      out.on = on[0] != '\0' && std::strcmp(on, "0") != 0;
    }
    if (const char* v = std::getenv("AV2ENC_RD_KEY_TX_TOP")) {
      const int k = std::atoi(v);
      if (k >= 1) out.top_k = std::min(k, 8);
    }
    if (const char* v = std::getenv("AV2ENC_RD_KEY_LEVEL")) {
      out.level = v[0] != '\0' && std::strcmp(v, "0") != 0;
    }
    if (const char* v = std::getenv("AV2ENC_RD_KEY_TX_SCALE")) {
      const double s = std::atof(v);
      if (s > 0.0) out.tx_scale = s;
    }
    if (const char* v = std::getenv("AV2ENC_RD_KEY_LEVEL_SCALE")) {
      const double s = std::atof(v);
      if (s > 0.0) out.level_scale = s;
    }
    if (const char* v = std::getenv("AV2ENC_RD_KEY_TX_D")) {
      out.plan_csf = std::atof(v) > 0.0 ? 1.0 : 0.0;
    }
    if (const char* v = std::getenv("AV2ENC_RD_KEY_STATS")) {
      out.stats = v[0] != '\0' && std::strcmp(v, "0") != 0;
    }
    return out;
  }();
  return cfg;
}

// Plan/level counters for the required report.  Printed once at process exit
// when the gate is on (stderr; the key picture is one frame per stream).
struct KeyTxRdStats {
  long long leaves = 0;
  long long plans[8] = {};
  long long sizes[25] = {};
  long long types[16] = {};
  long long screen_flips = 0;
  long long screen_kept = 0;
  long long parts = 0;
  long long level_candidates = 0;
  double bits = 0.0;
  double distortion = 0.0;

  ~KeyTxRdStats() {
    const KeyTxRdConfig& cfg = key_tx_rd_config();
    if (!cfg.on && !cfg.stats) return;
    long long s4 = sizes[0];
    long long s8 = sizes[1];
    long long s16 = sizes[2];
    long long s32 = sizes[3];
    long long s64 = sizes[4];
    long long t_dct = types[0];
    long long t_adst = 0;
    for (int t = 1; t < 9; ++t) t_adst += types[t];
    long long t_idtx = types[9];
    std::fprintf(
        stderr,
        "key-tx-rd: leaves=%lld plans[none=%lld split=%lld horz=%lld "
        "vert=%lld] tx4=%lld tx8=%lld tx16=%lld tx32=%lld tx64=%lld "
        "dct=%lld adst=%lld idtx=%lld parts=%lld screen_flips=%lld "
        "screen_kept=%lld trellis_parts=%lld bits=%.1f d=%.1f\n",
        leaves, plans[0], plans[1], plans[2], plans[3], s4, s8, s16, s32,
        s64, t_dct, t_adst, t_idtx, parts, screen_flips, screen_kept,
        level_candidates, bits, distortion);
  }
};

inline KeyTxRdStats& key_tx_rd_stats() {
  static KeyTxRdStats stats;
  return stats;
}

// §5.20.6.3 partition signalling cost from the §9.3 initial CDF rows, walked
// by the real writer so the symbols (do_partition, tx_partition_type,
// tx_2or3_partition_type) and their contexts cannot drift from the bitstream.
class PartitionBitSink : public txpartition::SymbolSink {
 public:
  double bits = 0.0;

  bool symbol(std::uint16_t* cdf, int n, int value,
              std::string& /*error*/) override {
    if (cdf == nullptr || n < 2 || value < 0 || value >= n) return false;
    const int lo = value == 0 ? 0 : static_cast<int>(cdf[value - 1]);
    const int hi = value == n - 1 ? 32768 : static_cast<int>(cdf[value]);
    const int range = hi > lo ? hi - lo : 1;
    bits += -std::log2(static_cast<double>(range) / 32768.0);
    return true;
  }
};

double exact_partition_bits(int mi_size, txpartition::TxPartition part) {
  txpartition::TxPartitionCdfs cdfs;
  cdfs.reset(/*reduced_tx_part_set=*/false);
  txpartition::BlockTxContext ctx;
  ctx.mi_size = mi_size;
  ctx.is_inter = false;
  ctx.fsc_mode = false;
  ctx.reduced_tx_part_set = false;
  PartitionBitSink sink;
  txpartition::SplitGeometry geometry;
  std::string error;
  if (!txpartition::write_tx_partition(sink, cdfs, ctx, part, &geometry,
                                       error)) {
    return -1.0;
  }
  return sink.bits;
}

}  // namespace

void InterFrameEncoder::luma_neighbor_smooth(int r, int c, bool& above_smooth,
                          bool& left_smooth) const {
  above_smooth = false;
  left_smooth = false;
  if (r > 0 && c < mi_cols_) {
    above_smooth = intra_search::joint_is_smooth(
        intra_joint_[static_cast<std::size_t>(r - 1) *
                         static_cast<std::size_t>(mi_cols_) +
                     static_cast<std::size_t>(c)]);
  }
  if (c > 0 && r < mi_rows_) {
    left_smooth = intra_search::joint_is_smooth(
        intra_joint_[static_cast<std::size_t>(r) *
                         static_cast<std::size_t>(mi_cols_) +
                     static_cast<std::size_t>(c - 1)]);
  }
}

void InterFrameEncoder::set_intra_joint(int r, int c, int w4, int h4, int joint) {
  for (int yy = 0; yy < h4; ++yy) {
    const int rr = r + yy;
    if (rr >= mi_rows_) break;
    for (int xx = 0; xx < w4; ++xx) {
      const int cc = c + xx;
      if (cc >= mi_cols_) break;
      intra_joint_[static_cast<std::size_t>(rr) *
                       static_cast<std::size_t>(mi_cols_) +
                   static_cast<std::size_t>(cc)] =
          static_cast<std::uint8_t>(joint);
    }
  }
}

void InterFrameEncoder::chroma_neighbor_smooth(int r, int c,
                                               bool& above_smooth,
                                               bool& left_smooth) const {
  above_smooth = false;
  left_smooth = false;
  const int cols = std::max(1, mi_cols_ / 2);
  const int rows = std::max(1, mi_rows_ / 2);
  if (uv_smooth_.size() <
      static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows)) {
    return;
  }
  if (r > 0 && c < cols) {
    above_smooth = uv_smooth_[static_cast<std::size_t>(r - 1) *
                                  static_cast<std::size_t>(cols) +
                              static_cast<std::size_t>(c)] != 0;
  }
  if (c > 0 && r < rows) {
    left_smooth = uv_smooth_[static_cast<std::size_t>(r) *
                                 static_cast<std::size_t>(cols) +
                             static_cast<std::size_t>(c - 1)] != 0;
  }
}

void InterFrameEncoder::set_uv_smooth(int r, int c, int w4, int h4, int uv_mode,
                                      bool is_cfl) {
  // §7.13.2.15/.16: only SMOOTH / SMOOTH_V / SMOOTH_H neighbours set the flag,
  // and a CfL block is UVMode = DC_PRED.
  const bool smooth =
      !is_cfl && (uv_mode == 9 || uv_mode == 10 || uv_mode == 11);
  const int cols = std::max(1, mi_cols_ / 2);
  const int rows = std::max(1, mi_rows_ / 2);
  for (int yy = 0; yy < h4; ++yy) {
    const int rr = r + yy;
    if (rr < 0 || rr >= rows) continue;
    for (int xx = 0; xx < w4; ++xx) {
      const int cc = c + xx;
      if (cc < 0 || cc >= cols) continue;
      uv_smooth_[static_cast<std::size_t>(rr) * static_cast<std::size_t>(cols) +
                 static_cast<std::size_t>(cc)] =
          static_cast<std::uint8_t>(smooth ? 1 : 0);
    }
  }
}

ibc::FrameView InterFrameEncoder::ibc_frame_view() const {
  ibc::FrameView view;
  view.y = ibc::PlaneView{recon_[0].samples.data(), recon_[0].stride,
                          mi_cols_ * 4, mi_rows_ * 4};
  view.u = ibc::PlaneView{recon_[1].samples.data(), recon_[1].stride,
                          mi_cols_ * 2, mi_rows_ * 2};
  view.v = ibc::PlaneView{recon_[2].samples.data(), recon_[2].stride,
                          mi_cols_ * 2, mi_rows_ * 2};
  view.has_chroma = true;
  view.sub_x = 1;
  view.sub_y = 1;
  return view;
}

bool InterFrameEncoder::choose_ibc_leaf(int r, int c, int bsize, std::int64_t intra_sse,
                     IbcChoice& out, std::string& error) {
  out = IbcChoice{};
  if (!ibc_enabled_ || !ibcs::intrabc_available(true, bsize)) return true;
  const auto size = static_cast<partition::BlockSize>(bsize);
  const int w = partition::block_width(size);
  const int h = partition::block_height(size);
  const int x0 = c * 4;
  const int y0 = r * 4;
  const int visible_w = std::min(w, mi_cols_ * 4 - x0);
  const int visible_h = std::min(h, mi_rows_ * 4 - y0);
  const int cw = w / 2;
  const int ch = h / 2;
  // IBC is an 8-bit 4:2:0 tool here; a leaf whose chroma block is not the
  // plain half-size (4:2:0 shared-chroma partitions) is not coded.
  if (!ibc_available_for_leaf(bsize)) return true;
  const ibc::FrameView frame = ibc_frame_view();
  const ibcs::BvStack stack = ibcs::build_bv_stack(
      frame, ibc_config_, ibc_buffer_, r, c, bsize, ibc_neighbours_);

  morph::MorphContext mctx;
  mctx.mi_col = c;
  mctx.mi_row = r;
  mctx.mi_cols = mi_cols_;
  mctx.mi_rows = mi_rows_;
  mctx.avail_u = y0 > 0;
  mctx.avail_l = x0 > 0;
  mctx.bit_depth = 8;
  const morph::MorphRect rect{x0, y0, w, h};

  const auto consider = [&](ibc::BlockVector bv, int stack_index) {
    if ((bv.row & 7) != 0 || (bv.col & 7) != 0) return;
    // The predicate is what decides; the message is not used here.
    std::string ibc_why;
    if (!ibcs::is_bv_encodable(frame, ibc_config_, ibc_buffer_, r, c, bsize,
                               bv, ibc_why)) {
      return;
    }
    std::vector<std::uint8_t> luma(static_cast<std::size_t>(w) *
                                   static_cast<std::size_t>(h));
    std::vector<std::uint8_t> u(static_cast<std::size_t>(cw) *
                                static_cast<std::size_t>(ch));
    std::vector<std::uint8_t> v(u.size());
    if (!ibc::copy_block(frame, r, c, bsize, bv, luma.data(), w, u.data(),
                         cw, v.data(), cw, visible_w, visible_h)) {
      return;
    }
    const std::int64_t copy_sse = sse_against_source(
        source_, 0, x0 + tile_x_, y0 + tile_y_, w, h, luma);
    bool best_morph = false;
    std::int64_t best_sse = copy_sse;
    std::vector<std::uint8_t> adapted;
    if (plan_morph_) {
      std::string why;
      if (ibcs::is_bv_offset_valid(frame, ibc_config_, ibc_buffer_, r, c,
                                   bsize, bv, -1, -1, why)) {
        // The fit reads the reconstructed above/left template and the source
        // block at refX/refY, so evaluate it on a full-plane snapshot where
        // only this block has been written.
        out.morph_scratch.assign(recon_[0].samples.begin(),
                                 recon_[0].samples.end());
        morph::MorphPlane plane{out.morph_scratch.data(), recon_[0].stride};
        for (int yy = 0; yy < h; ++yy) {
          std::memcpy(out.morph_scratch.data() +
                          static_cast<std::size_t>(y0 + yy) *
                              static_cast<std::size_t>(recon_[0].stride) +
                          static_cast<std::size_t>(x0),
                      luma.data() + static_cast<std::size_t>(yy) *
                                        static_cast<std::size_t>(w),
                      static_cast<std::size_t>(w));
        }
        morph::MorphFit fit{};
        if (morph::morph_predict(mctx, plane, rect, bv.row, bv.col, &fit)) {
          adapted.resize(luma.size());
          for (int yy = 0; yy < h; ++yy) {
            std::memcpy(
                adapted.data() + static_cast<std::size_t>(yy) *
                                     static_cast<std::size_t>(w),
                out.morph_scratch.data() +
                    static_cast<std::size_t>(y0 + yy) *
                        static_cast<std::size_t>(recon_[0].stride) +
                    static_cast<std::size_t>(x0),
                static_cast<std::size_t>(w));
          }
          const std::int64_t morph_sse = sse_against_source(
              source_, 0, x0 + tile_x_, y0 + tile_y_, w, h, adapted);
          if (morph_sse < best_sse) {
            best_sse = morph_sse;
            best_morph = true;
          }
        }
      }
    }
    if (!out.found || best_sse < out.sse) {
      out.found = true;
      out.bv = bv;
      out.stack_index = stack_index;
      out.morph = best_morph;
      out.sse = best_sse;
      out.luma = best_morph ? adapted : luma;
      out.u = std::move(u);
      out.v = std::move(v);
    }
  };

  for (int i = 0; i < stack.count; ++i) {
    if (!stack.entries[static_cast<std::size_t>(i)].valid) continue;
    consider(stack.entries[static_cast<std::size_t>(i)].bv, i);
  }
  static constexpr int kPeriodic[] = {8, 16, 32, 64, 128};
  for (const int d : kPeriodic) {
    const ibc::BlockVector cands[2] = {ibc::BlockVector{-d * 8, 0},
                                       ibc::BlockVector{0, -d * 8}};
    for (const ibc::BlockVector& bv : cands) {
      consider(bv, -1);
    }
  }
  // A tie picks the IBC skip block: it codes no residual, so equal luma
  // distortion is a rate win over the intra leaf.
  if (!out.found || out.sse > intra_sse) {
    out.found = false;
  }
  (void)error;
  return true;
}

bool InterFrameEncoder::ibc_available_for_leaf(int bsize) const {
  const auto size = static_cast<partition::BlockSize>(bsize);
  const int w = partition::block_width(size);
  const int h = partition::block_height(size);
  if (w < 8 || h < 8) return false;
  // §7.13.3.18 copies exactly w x h luma and (w/2) x (h/2) chroma.  Reject
  // the shared-chroma ownership cases where the chroma residual block is not
  // the plain half-size.
  return w % 2 == 0 && h % 2 == 0 && w / 2 >= 4 && h / 2 >= 4;
}

bool InterFrameEncoder::palette_available_for_leaf(int bsize, int w, int h) const {
  if (!palette_enabled_) return false;
  pals::ScreenContentFlags flags;
  flags.single_picture_header = false;
  flags.allow_screen_content_tools = true;
  flags.allow_palette = true;
  flags.force_integer_mv = true;
  return pals::palette_allowed_for_block(flags, bsize, w, h);
}

bool InterFrameEncoder::choose_palette_leaf(int r, int c, int bsize, int w, int h,
                         std::int64_t intra_pred_sse, PaletteChoice& out,
                         std::string& error) {
  out = PaletteChoice{};
  if (!palette_available_for_leaf(bsize, w, h)) {
    return true;
  }
  const int x0 = c * 4;
  const int y0 = r * 4;
  const int visible_w = std::min(w, mi_cols_ * 4 - x0);
  const int visible_h = std::min(h, mi_rows_ * 4 - y0);
  if (visible_w <= 0 || visible_h <= 0) return true;
  std::uint8_t cache[16] = {};
  const int cache_n = palette::get_palette_cache(palette_state_, r, c,
                                                 c > 0, cache);
  std::vector<std::uint8_t> src(static_cast<std::size_t>(w) *
                                static_cast<std::size_t>(h));
  for (int yy = 0; yy < h; ++yy) {
    for (int xx = 0; xx < w; ++xx) {
      src[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) +
          static_cast<std::size_t>(xx)] =
          static_cast<std::uint8_t>(source_sample(
              source_, 0, tile_x_ + x0 + xx, tile_y_ + y0 + yy));
    }
  }
  palette::PaletteParams params;
  params.max_colors = 8;
  params.exact_when_possible = true;
  params.choose_direction = true;
  palette::PaletteResult result;
  std::string palette_error;
  if (!palette::analyze_palette(src.data(), w, w, h, visible_w, visible_h,
                                cache, cache_n, params, result,
                                palette_error)) {
    error = "analyze_palette failed: " + palette_error;
    return false;
  }
  // §5.20.8.1 delta-codes a strictly increasing palette.
  bool representable = result.has_palette;
  for (int i = 1; i < result.size && representable; ++i) {
    if (result.colors[static_cast<std::size_t>(i)] <=
        result.colors[static_cast<std::size_t>(i - 1)]) {
      representable = false;
    }
  }
  if (!representable) result.has_palette = false;
  if (!result.has_palette || result.sse >= intra_pred_sse) return true;
  out.luma.resize(static_cast<std::size_t>(w) *
                  static_cast<std::size_t>(h));
  palette::predict_palette_block(result, w, h, out.luma.data(), w);
  out.sse = result.sse;
  out.result = std::move(result);
  out.found = true;
  return true;
}

bool InterFrameEncoder::predict_intra_plane_mode(int plane, int x0, int y0, int w, int h,
                              intra::Mode mode, bool above_smooth,
                              bool left_smooth, const SbSink::State& st,
                              std::vector<std::uint8_t>& out) const {
  const int sb_mi = plane == 0 ? sb_size_ / 4 : sb_size_ / 8;
  const int mi_c = plane == 0 ? mi_cols_ : mi_cols_ / 2;
  const int mi_r = plane == 0 ? mi_rows_ : mi_rows_ / 2;
  const intra_search::Availability avail = intra_search::availability(
      st.decoded[static_cast<std::size_t>(plane)], x0 / 4, y0 / 4, w / 4,
      h / 4, mi_c, mi_r, sb_mi);
  intra::NeighborRequest request;
  request.recon =
      intra::PlaneView{recon_[plane].samples.data(), recon_[plane].stride};
  request.x = x0;
  request.y = y0;
  request.w = w;
  request.h = h;
  request.max_x = recon_[plane].width - 1;
  // §7.13.2 availability stops at the coded MI extent (MiRows/MiCols), not
  // at the SB-aligned reconstruction buffer used by --video-edge-inter.
  request.max_y =
      (plane == 0 ? mi_rows_ * 4 : mi_rows_ * 2) - 1;
  request.have_left = x0 > 0;
  request.have_above = y0 > 0;
  request.num4_above_right = avail.num4_above_right;
  request.num4_below_left = avail.num4_below_left;
  request.color_plane = plane;
  intra::PredictParams params;
  params.mode = mode;
  params.angle_delta = 0;
  params.enable_edge_filter = true;
  params.above_neighbor_is_smooth = above_smooth;
  params.left_neighbor_is_smooth = left_smooth;
  out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  return intra::predict_block(request, 8, params, out.data(),
                              static_cast<std::size_t>(w)) ==
         intra::IntraStatus::kOk;
}

bool InterFrameEncoder::encode_intra_luma_tx_partition(
    int r, int c, int w, int h, BlockSize size,
    const std::vector<std::uint8_t>& block_pred, int y_mode,
    int angle_delta, int mrl_index, bool mrl_sec_index,
    bool above_smooth, bool left_smooth, int mask_row, SbSink::State& st,
    TxPlan* plan, std::string& error) {
  const int x0 = c * 4;
  const int y0 = r * 4;
  const int ax0 = x0 + tile_x_;
  const int ay0 = y0 + tile_y_;
  const KeyTxRdConfig& kcfg = key_tx_rd_config();
  const double lambda = key_tx_rd_lambda(options_, st.current_qindex);
  const double exact_lambda = lambda * kcfg.tx_scale;
  const int qctx = ::av2enc::still::qctx::slice_for_base_q_idx(
      options_.tx_context_rate ? base_q_index() : st.current_qindex);
  const int mi_size = static_cast<int>(size);
  // AV2ENC_RD_KEY_TX: the key picture's distortion is the same eff-perc D the
  // leaf and its TX search use.  The source weight map is neutral at the
  // shipped shape (0), so the error-domain CSF carries the perceptual model.
  const double key_csf = eff_perc::enabled() ? eff_perc::key_csf_gain() : 0.0;
  const double key_csf2 = eff_perc::enabled() ? eff_perc::csf2_gain() : 0.0;
  const double level_scale =
      kcfg.level_scale > 0.0
          ? kcfg.level_scale
          : 0.25 * rdoq_sse_scale_at(options_, st.current_qindex);

  // §5.20.6.3 legal partitions.  HORZ4/VERT4 only appear in the 2or3 branch
  // (one legal direction); the still encoder's chain enumerates NONE/SPLIT/
  // HORZ/VERT, and this port keeps exactly that set.
  std::vector<txpart::TxPartition> options;
  options.push_back(txpart::TxPartition::kNone);
  const txpart::LegalSplits legal =
      txpart::legal_splits(mi_size, /*reduced_tx_part_set=*/false);
  for (int i = 0; i < legal.option_count; ++i) {
    const txpart::TxPartition option = legal.options[i];
    if (option != txpart::TxPartition::kSplit &&
        option != txpart::TxPartition::kHorz &&
        option != txpart::TxPartition::kVert) {
      continue;
    }
    txpart::SplitGeometry geometry;
    std::string geometry_error;
    if (!txpart::tx_partition_geometry(mi_size, 0, 0, option, &geometry,
                                       geometry_error)) {
      continue;
    }
    bool inside = true;
    for (int k = 0; k < geometry.count; ++k) {
      const txpart::SplitPart& part = geometry.parts[k];
      const int px = x0 + part.mi_col * 4;
      const int py = y0 + part.mi_row * 4;
      if (px < 0 || py < 0 ||
          px + txpart::tx_width(part.tx_size) > recon_[0].width ||
          py + txpart::tx_height(part.tx_size) > mi_rows_ * 4) {
        inside = false;
        break;
      }
    }
    if (inside) options.push_back(option);
  }

  // The decoder reconstructs the same parts in the same order, so a
  // candidate evaluation writes the real planes and the search restores the
  // block's recon samples and §6.19.2.3 cells before the next trial.
  struct Snapshot {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
    std::vector<std::uint8_t> recon;
    int c0 = 0;
    int r0 = 0;
    int cw4 = 0;
    int ch4 = 0;
    std::vector<std::uint8_t> decoded;
  };
  Snapshot snapshot;
  snapshot.x = x0;
  snapshot.y = y0;
  snapshot.w = w;
  snapshot.h = h;
  snapshot.recon.resize(static_cast<std::size_t>(w) *
                        static_cast<std::size_t>(h));
  for (int yy = 0; yy < h; ++yy) {
    for (int xx = 0; xx < w; ++xx) {
      const int px = x0 + xx;
      const int py = y0 + yy;
      snapshot.recon[static_cast<std::size_t>(yy) *
                         static_cast<std::size_t>(w) +
                     static_cast<std::size_t>(xx)] =
          (px < recon_[0].width && py < recon_[0].height)
              ? recon_[0].at(px, py)
              : 0;
    }
  }
  snapshot.c0 = x0 / 4;
  snapshot.r0 = y0 / 4;
  snapshot.cw4 = (w + 3) / 4;
  snapshot.ch4 = (h + 3) / 4;
  snapshot.decoded.resize(static_cast<std::size_t>(snapshot.cw4) *
                          static_cast<std::size_t>(snapshot.ch4));
  for (int yy = 0; yy < snapshot.ch4; ++yy) {
    for (int xx = 0; xx < snapshot.cw4; ++xx) {
      snapshot.decoded[static_cast<std::size_t>(yy) *
                           static_cast<std::size_t>(snapshot.cw4) +
                       static_cast<std::size_t>(xx)] =
          st.decoded[0].cell(snapshot.c0 + xx, snapshot.r0 + yy);
    }
  }
  const auto restore = [&]() {
    for (int yy = 0; yy < snapshot.h; ++yy) {
      for (int xx = 0; xx < snapshot.w; ++xx) {
        const int px = snapshot.x + xx;
        const int py = snapshot.y + yy;
        if (px >= recon_[0].width || py >= recon_[0].height) continue;
        recon_[0].at(px, py) =
            snapshot.recon[static_cast<std::size_t>(yy) *
                               static_cast<std::size_t>(snapshot.w) +
                           static_cast<std::size_t>(xx)];
      }
    }
    for (int yy = 0; yy < snapshot.ch4; ++yy) {
      for (int xx = 0; xx < snapshot.cw4; ++xx) {
        st.decoded[0].set_cell(
            snapshot.c0 + xx, snapshot.r0 + yy,
            snapshot.decoded[static_cast<std::size_t>(yy) *
                                 static_cast<std::size_t>(snapshot.cw4) +
                             static_cast<std::size_t>(xx)]);
      }
    }
  };

  struct Trial {
    txpart::TxPartition part = txpart::TxPartition::kNone;
    TxPlan plan;
    std::vector<std::uint8_t> recon;
    double cost = std::numeric_limits<double>::infinity();
    double distortion = 0.0;
    double rate = 0.0;
  };

  // One transform-leaf search.  The screen (`exact == false`) is the
  // historical call: heuristic rate, plain SSE, the configured TX lambda.  The
  // exact stage turns on the §8.3 coefficient model, the key TX lambda and the
  // coefficient trellis.
  const auto search_part = [&](int px, int py, int tw, int th, int part_mask,
                               const std::vector<std::uint8_t>& pred,
                               bool exact, double search_lambda,
                               TxPlan* out) {
    const int mapped_mode = std::max(
        0, mrl_stream::mapped_intra_mode_for_mrl(
               y_mode, angle_delta, mrl_index, tw, th));
    return search_luma_tx(
        source_, px, py, tw, th, pred, st.current_qindex,
        txtsearch::Mode::kIntra, search_lambda, options_.enable_idtx_intra,
        mapped_mode, options_.enable_tcq, qctx,
        /*enable_inter_ddt=*/false, /*search_partition=*/false,
        exact ? true : options_.tx_context_rate, /*trace=*/false,
        /*allow_rdoq=*/exact && kcfg.level, /*rdoq_qctx=*/qctx,
        level_scale, *out, error, part_mask,
        /*distortion_weight=*/nullptr, /*cdf_slice=*/nullptr, key_csf,
        key_csf2,
        /*intra_rdoq=*/exact && kcfg.level,
        /*model_rate_scale=*/1.0);
  };

  // Runs one candidate against the real planes and leaves its plan, its
  // block-sized reconstruction and its RD cost in `trial`.
  const auto evaluate = [&](txpart::TxPartition part, bool exact,
                            Trial* trial) -> bool {
    trial->part = part;
    trial->plan = TxPlan{};
    const double search_lambda = exact ? exact_lambda : lambda;
    double partition_bits = 3.0;
    if (exact) {
      partition_bits = exact_partition_bits(mi_size, part);
      if (partition_bits < 0.0) {
        error = "exact TX partition signalling cost failed";
        return false;
      }
    }
    if (part == txpart::TxPartition::kNone) {
      TxPlan whole;
      if (!search_part(ax0, ay0, w, h, mask_row, block_pred, exact,
                       search_lambda, &whole)) {
        return false;
      }
      trial->recon = whole.recon;
      trial->rate = whole.rate_bits + (exact ? partition_bits : 0.0);
      trial->plan = std::move(whole);
    } else {
      txpart::SplitGeometry geometry;
      std::string geometry_error;
      if (!txpart::tx_partition_geometry(mi_size, 0, 0, part, &geometry,
                                         geometry_error)) {
        error = "intra TX partition geometry failed: " + geometry_error;
        return false;
      }
      TxPlan merged;
      merged.searched = true;
      merged.partition = static_cast<int>(part);
      merged.rate_bits = 0.0;
      double rate = partition_bits;
      std::vector<std::uint8_t> block(
          static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0);
      for (int k = 0; k < geometry.count; ++k) {
        const txpart::SplitPart& split = geometry.parts[k];
        const int px = x0 + split.mi_col * 4;
        const int py = y0 + split.mi_row * 4;
        const int tw = txpart::tx_width(split.tx_size);
        const int th = txpart::tx_height(split.tx_size);
        const intra_search::Availability avail = intra_search::availability(
            st.decoded[0], px / 4, py / 4, tw / 4, th / 4, mi_cols_, mi_rows_,
            /*sb_mi=*/sb_size_ / 4);
        std::vector<std::uint8_t> pred;
        if (mrl_index > 0) {
          if (!predict_luma_mrl(recon_[0], avail, px, py, tw, th,
                                above_smooth, left_smooth, y_mode,
                                angle_delta, mrl_index, mrl_sec_index,
                                /*sb_height=*/64, pred, error)) {
            return false;
          }
        } else if (!intra_luma_part_predict(
                       recon_[0], avail, px, py, tw, th, y_mode,
                       angle_delta, options_.enable_ibp, above_smooth,
                       left_smooth, pred)) {
          error = "intra TX-part prediction failed";
          return false;
        }
        const int part_mask =
            mask_row < 0 ? -1
                         : std::clamp(mask_row - split.mi_row * 4, 0, th);
        TxPlan part_plan;
        if (!search_part(ax0 + split.mi_col * 4, ay0 + split.mi_row * 4, tw,
                         th, part_mask, pred, exact, search_lambda,
                         &part_plan)) {
          return false;
        }
        if (part_plan.parts.empty()) {
          error = "intra TX-part search returned no leaves";
          return false;
        }
        const TxPlan::Part& leaf = part_plan.parts[0];
        TxPlan::Part merged_part;
        merged_part.x = split.mi_col * 4;
        merged_part.y = split.mi_row * 4;
        merged_part.coded_w = tw;
        merged_part.coded_h = th;
        merged_part.tx_size = leaf.tx_size;
        merged_part.tx_type = leaf.tx_type;
        merged_part.mi_row = split.mi_row;
        merged_part.mi_col = split.mi_col;
        merged_part.eob = leaf.eob;
        merged_part.levels = leaf.levels;
        rate += part_plan.rate_bits;
        merged.parts.push_back(std::move(merged_part));
        write_plane_block(recon_[0], px, py, tw, th, part_plan.recon);
        st.decoded[0].fill(px / 4, py / 4, tw / 4, th / 4);
        for (int yy = 0; yy < th; ++yy) {
          for (int xx = 0; xx < tw; ++xx) {
            block[static_cast<std::size_t>(split.mi_row * 4 + yy) *
                      static_cast<std::size_t>(w) +
                  static_cast<std::size_t>(split.mi_col * 4 + xx)] =
                part_plan.recon[static_cast<std::size_t>(yy) *
                                    static_cast<std::size_t>(tw) +
                                static_cast<std::size_t>(xx)];
          }
        }
      }
      merged.tx_size = merged.parts[0].tx_size;
      merged.eob = merged.parts[0].eob;
      merged.tx_type = merged.parts[0].tx_type;
      merged.rate_bits = rate - partition_bits;
      trial->recon = std::move(block);
      trial->plan = std::move(merged);
      trial->rate = rate;
    }
    if (exact) {
      const double d_csf = kcfg.plan_csf > 0.0 ? key_csf : 0.0;
      const double d_csf2 = kcfg.plan_csf > 0.0 ? key_csf2 : 0.0;
      trial->distortion = eff_perc::weighted_sse(
          /*weights=*/nullptr, w, h, source_.y.data(),
          static_cast<int>(source_.width), ax0, ay0, trial->recon.data(), w,
          0, 0, mask_row, d_csf, d_csf2);
      trial->cost = trial->distortion + exact_lambda * trial->rate;
    } else {
      trial->distortion = static_cast<double>(
          sse_against_source(source_, 0, ax0, ay0, w, h, trial->recon,
                             mask_row));
      trial->cost = trial->distortion + lambda * trial->rate;
    }
    return true;
  };

  // Stage 1: screen every legal partition with the historical cost.
  std::vector<Trial> screen(options.size());
  for (std::size_t i = 0; i < options.size(); ++i) {
    restore();
    if (!evaluate(options[i], /*exact=*/false, &screen[i])) return false;
  }
  std::vector<std::size_t> order(options.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t a, std::size_t b) {
                     return screen[a].cost < screen[b].cost;
                   });

  Trial best;
  bool have_best = false;
  const std::size_t screen_winner = order[0];
  if (!kcfg.on) {
    best = std::move(screen[screen_winner]);
    have_best = true;
  } else {
    // Stage 2: exact plan rate + eff-perc D on the top-K screen candidates.
    const std::size_t top =
        std::min(order.size(), static_cast<std::size_t>(kcfg.top_k));
    for (std::size_t j = 0; j < top; ++j) {
      restore();
      Trial trial;
      if (!evaluate(options[order[j]], /*exact=*/true, &trial)) return false;
      if (!have_best || trial.cost < best.cost) {
        best = std::move(trial);
        have_best = true;
      }
    }
  }
  if ((kcfg.on || kcfg.stats) && !rd_driver::measuring()) {
    KeyTxRdStats& stats = key_tx_rd_stats();
    ++stats.leaves;
    if (kcfg.on) {
      if (best.part == screen[screen_winner].part) {
        ++stats.screen_kept;
      } else {
        ++stats.screen_flips;
      }
    }
    ++stats.plans[static_cast<int>(best.part)];
    for (const TxPlan::Part& part : best.plan.parts) {
      const int tw = txpartition::tx_width(part.tx_size);
      const int th = txpartition::tx_height(part.tx_size);
      if (tw == th && tw >= 4 && tw <= 64) {
        ++stats.sizes[static_cast<int>(part.tx_size)];
      }
      if (part.tx_type >= 0 && part.tx_type < 16) {
        ++stats.types[part.tx_type];
      }
      ++stats.parts;
    }
    stats.level_candidates +=
        kcfg.level ? static_cast<long long>(best.plan.parts.size()) : 0;
    stats.bits += best.plan.rate_bits;
    stats.distortion += best.distortion;
  }
  if (!have_best) {
    error = "intra TX-part search selected no candidate";
    return false;
  }
  restore();
  // Commit the winner: the whole block is written in one go (the parts tile
  // it), so the post-block state matches a whole-block reconstruction.
  write_plane_block(recon_[0], x0, y0, w, h, best.recon);
  st.decoded[0].fill(c, r, w / 4, h / 4);
  *plan = std::move(best.plan);
  plan->recon = std::move(best.recon);
  if (trace_) {
    std::cerr << "  intra tx part r=" << r << " c=" << c << " " << w << "x"
              << h << " partition=" << static_cast<int>(best.part)
              << " parts=" << plan->parts.size()
              << " rate=" << plan->rate_bits << " cost=" << best.cost
              << "\n";
  }
  return true;
}

}  // namespace av2enc::video
