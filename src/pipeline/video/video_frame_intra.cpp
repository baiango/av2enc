#include "pipeline/video/video_frame_internal.h"

namespace av2enc::video {

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
    if (!ibcs::is_bv_encodable(frame, ibc_config_, ibc_buffer_, r, c, bsize,
                               bv, nullptr)) {
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
                                   bsize, bv, -1, -1, &why)) {
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
                      luma.data() + static_cast<std::size_t>(yy) * w,
                      static_cast<std::size_t>(w));
        }
        morph::MorphFit fit{};
        if (morph::morph_predict(mctx, plane, rect, bv.row, bv.col, &fit)) {
          adapted.resize(luma.size());
          for (int yy = 0; yy < h; ++yy) {
            std::memcpy(
                adapted.data() + static_cast<std::size_t>(yy) * w,
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
      st.decoded[plane], x0 / 4, y0 / 4, w / 4, h / 4, mi_c, mi_r, sb_mi);
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
  const double lambda = key_tx_rd_lambda(options_, st.current_qindex);
  const int qctx = ::av2enc::still::qctx::slice_for_base_q_idx(
      options_.tx_context_rate ? base_q_index() : st.current_qindex);
  const int mi_size = static_cast<int>(size);

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
                                       &geometry_error)) {
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

  // Runs one candidate against the real planes and leaves its plan, its
  // block-sized reconstruction and its RD cost in `trial`.
  struct Trial {
    txpart::TxPartition part = txpart::TxPartition::kNone;
    TxPlan plan;
    std::vector<std::uint8_t> recon;
    double cost = std::numeric_limits<double>::infinity();
  };
  const auto evaluate = [&](txpart::TxPartition part,
                            Trial* trial) -> bool {
    trial->part = part;
    trial->plan = TxPlan{};
    if (part == txpart::TxPartition::kNone) {
      TxPlan whole;
      const int mapped_mode = std::max(
          0, mrl_stream::mapped_intra_mode_for_mrl(
                 y_mode, angle_delta, mrl_index, w, h));
      if (!search_luma_tx(source_, ax0, ay0, w, h, block_pred,
                          st.current_qindex, txtsearch::Mode::kIntra, lambda,
                          options_.enable_idtx_intra, mapped_mode,
                          options_.enable_tcq, qctx,
                          /*enable_inter_ddt=*/false,
                          /*search_partition=*/false,
                          options_.tx_context_rate, /*trace=*/false,
                          /*allow_rdoq=*/false, /*rdoq_qctx=*/0,
                          rdoq_sse_scale_at(options_, st.current_qindex),
                          whole, error, mask_row)) {
        return false;
      }
      trial->recon = whole.recon;
      trial->plan = std::move(whole);
      trial->cost = lambda * trial->plan.rate_bits;
    } else {
      txpart::SplitGeometry geometry;
      std::string geometry_error;
      if (!txpart::tx_partition_geometry(mi_size, 0, 0, part, &geometry,
                                         &geometry_error)) {
        error = "intra TX partition geometry failed: " + geometry_error;
        return false;
      }
      TxPlan merged;
      merged.searched = true;
      merged.partition = static_cast<int>(part);
      merged.rate_bits = 0.0;
      // §5.20.6.3: the partition symbols cost ~3 bits; the exact shared rate
      // model is the writer's, so the decision uses the still encoder's
      // constant estimate.
      double rate = 3.0;
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
        const int mapped_mode = std::max(
            0, mrl_stream::mapped_intra_mode_for_mrl(
                   y_mode, angle_delta, mrl_index, tw, th));
        const int part_mask =
            mask_row < 0 ? -1
                         : std::clamp(mask_row - split.mi_row * 4, 0, th);
        TxPlan part_plan;
        if (!search_luma_tx(
                source_, ax0 + split.mi_col * 4, ay0 + split.mi_row * 4,
                tw, th, pred, st.current_qindex, txtsearch::Mode::kIntra,
                lambda, options_.enable_idtx_intra, mapped_mode,
                options_.enable_tcq, qctx, /*enable_inter_ddt=*/false,
                /*search_partition=*/false, options_.tx_context_rate,
                /*trace=*/false, /*allow_rdoq=*/false, /*rdoq_qctx=*/0,
                rdoq_sse_scale_at(options_, st.current_qindex), part_plan,
                error, part_mask)) {
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
      merged.rate_bits = rate - 3.0;
      trial->recon = std::move(block);
      trial->plan = std::move(merged);
      trial->cost = lambda * rate;
    }
    const std::int64_t sse = sse_against_source(
        source_, 0, ax0, ay0, w, h, trial->recon, mask_row);
    trial->cost += static_cast<double>(sse);
    return true;
  };

  Trial best;
  bool have_best = false;
  for (const txpart::TxPartition option : options) {
    restore();
    Trial trial;
    if (!evaluate(option, &trial)) return false;
    if (!have_best || trial.cost < best.cost) {
      best = std::move(trial);
      have_best = true;
    }
  }
  if (!have_best) {
    error = "intra TX-part search selected no candidate";
    return false;
  }
  restore();
  // Commit the winner: the whole block is written in one go (the parts
  // tile it), so the post-block state matches a whole-block reconstruction.
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
