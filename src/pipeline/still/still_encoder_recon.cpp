// Still-picture encoder: source access, transform-part reconstruction
// and deblock metadata.  Split verbatim out of still_encoder.cpp; see
// still_encoder_internal.h for the shared declaration.

#include "pipeline/still/still_encoder_internal.h"

namespace av2enc::still {
namespace detail {

std::uint8_t Encoder::source_sample(int plane, int x, int y) const {
    const std::span<const std::uint8_t> data =
        plane == 0 ? picture_.y : (plane == 1 ? picture_.u : picture_.v);
    const int w = plane == 0 ? width_ : chroma_width_;
    const int h = plane == 0 ? height_ : chroma_height_;
    const int cx = clip3(0, w - 1, x);
    const int cy = clip3(0, h - 1, y);
    const int stride = plane == 0 ? width_ : chroma_width_;
    return data[static_cast<std::size_t>(cy) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(cx)];
  }

void Encoder::fill_source(int plane, int x0, int y0, int w, int h,
                 std::vector<std::uint8_t>& out) const {
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        out[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
            static_cast<std::size_t>(x)] =
            source_sample(plane, x0 + x, y0 + y);
      }
    }
  }

int Encoder::count_top_right(int plane, int abs_x4, int abs_y4, int w4) const {
    const int sub = plane == 0 ? 0 : 1;
    const int sb_c0 = sb_mi_col_ >> sub;
    const int sb_r0 = sb_mi_row_ >> sub;
    const int rel_x = abs_x4 - sb_c0;
    const int rel_y = abs_y4 - sb_r0;
    const int sb_w = sb_cols_plane_[plane];
    // §6.19.2.3 clear_block_decoded_flags: the top border is 1 for
    // x < (MiColEnd - c), and MiColEnd is the tile end, not the SB end. The
    // BlockDecoded array extends to 2 * sbSize4, so availability along the
    // top row covers up to two superblocks (or the tile remainder).
    const int plane_cols = plane == 0 ? mi_cols_ : mi_cols_ / 2;
    const int border_w = std::min(2 * (sb_mi_ >> sub), plane_cols - sb_c0);
    int count = 0;
    for (int i = 0; i < w4; ++i) {
      const int x = rel_x + w4 + i;
      if (x >= (rel_y == 0 ? border_w : sb_w)) break;
      if (rel_y - 1 >= 0 &&
          !decoded_[static_cast<std::size_t>(plane)].at(sb_c0 + x,
                                                        sb_r0 + rel_y - 1)) {
        break;
      }
      ++count;
    }
    return count;
  }

int Encoder::count_bottom_left(int plane, int abs_x4, int abs_y4, int h4) const {
    const int sub = plane == 0 ? 0 : 1;
    const int sb_c0 = sb_mi_col_ >> sub;
    const int sb_r0 = sb_mi_row_ >> sub;
    const int rel_x = abs_x4 - sb_c0;
    const int rel_y = abs_y4 - sb_r0;
    const int sb_h = sb_rows_plane_[plane];
    // clear_block_decoded_flags() presets the left border only within the
    // current SB (and clears the boundary row), so below-left availability can
    // never extend into the next SB row.
    int count = 0;
    for (int i = 0; i < h4; ++i) {
      const int y = rel_y + h4 + i;
      if (y >= sb_h) break;
      if (rel_x - 1 >= 0 &&
          !decoded_[static_cast<std::size_t>(plane)].at(sb_c0 + rel_x - 1,
                                                        sb_r0 + y)) {
        break;
      }
      ++count;
    }
    return count;
  }

Encoder::LumaSnapshot Encoder::take_luma_snapshot(int x0, int y0, int w, int h) const {
    LumaSnapshot s;
    s.x0 = x0;
    s.y0 = y0;
    s.w = w;
    s.h = h;
    s.recon.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int px = x0 + x;
        const int py = y0 + y;
        s.recon[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(x)] =
            (px < recon_[0].width && py < recon_[0].height)
                ? recon_[0].at(px, py)
                : 0;
      }
    }
    s.c0 = x0 / 4;
    s.r0 = y0 / 4;
    s.cw4 = (w + 3) / 4;
    s.ch4 = (h + 3) / 4;
    s.decoded.resize(static_cast<std::size_t>(s.cw4) *
                     static_cast<std::size_t>(s.ch4));
    for (int y = 0; y < s.ch4; ++y) {
      for (int x = 0; x < s.cw4; ++x) {
        s.decoded[static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(s.cw4) +
                  static_cast<std::size_t>(x)] =
            decoded_[0].at(s.c0 + x, s.r0 + y) ? 1 : 0;
      }
    }
    return s;
  }

void Encoder::restore_luma_snapshot(const LumaSnapshot& s) {
    for (int y = 0; y < s.h; ++y) {
      for (int x = 0; x < s.w; ++x) {
        const int px = s.x0 + x;
        const int py = s.y0 + y;
        if (px >= recon_[0].width || py >= recon_[0].height) continue;
        recon_[0].at(px, py) =
            s.recon[static_cast<std::size_t>(y) * static_cast<std::size_t>(s.w) +
                    static_cast<std::size_t>(x)];
      }
    }
    for (int y = 0; y < s.ch4; ++y) {
      for (int x = 0; x < s.cw4; ++x) {
        const int c = s.c0 + x;
        const int r = s.r0 + y;
        if (c >= decoded_[0].cols || r >= decoded_[0].rows) continue;
        decoded_[0].cells[static_cast<std::size_t>(r) *
                              static_cast<std::size_t>(decoded_[0].cols) +
                          static_cast<std::size_t>(c)] =
            s.decoded[static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(s.cw4) +
                      static_cast<std::size_t>(x)];
      }
    }
  }

bool Encoder::encode_luma_parts(int r, int c, int w, int h,
                       partition::BlockSize size, const ModeChoice& y,
                       int partition, std::vector<TxPlan>* plans,
                       std::vector<syntax::LumaTxPart>* parts,
                       double* block_sse) {
    tpart::SplitGeometry geometry;
    std::string geometry_error;
    if (!tpart::tx_partition_geometry(
            static_cast<int>(size), r, c,
            static_cast<tpart::TxPartition>(partition), &geometry,
            geometry_error)) {
      error_ = "tx partition geometry failed: " + geometry_error;
      return false;
    }
    plans->clear();
    parts->clear();
    for (int i = 0; i < geometry.count; ++i) {
      const tpart::SplitPart& part = geometry.parts[i];
      if (part.mi_col >= mi_cols_ || part.mi_row >= mi_rows_) {
        error_ = "tx partition part origin outside the coded plane";
        return false;
      }
      const int px = part.mi_col * 4;
      const int py = part.mi_row * 4;
      const int tw = tpart::tx_width(part.tx_size);
      const int th = tpart::tx_height(part.tx_size);
      std::vector<std::uint8_t> pred;
      if (!predict_block(0, px, py, tw, th, y.mode, y.angle_delta, pred,
                         /*uv_is_cfl=*/false, y.mrl_index, y.mrl_sec_index,
                         y.use_dip, y.dip_mode, y.dip_transpose,
                         /*block_mi_row=*/r, /*block_mi_col=*/c)) {
        return false;
      }
      TxPlan plan;
      if (!compute_tx(0, px, py, tw, th, pred, plan,
                      options_.tx_type || options_.idtx,
                      y.av2_mode, y.angle_delta, y.mrl_index)) {
        return false;
      }
      write_recon(0, px, py, tw, th, plan);
      decoded_[0].fill(px / 4, py / 4, tw / 4, th / 4);
      syntax::LumaTxPart syntax_part;
      syntax_part.tx_size = static_cast<std::uint8_t>(plan.tx_size);
      syntax_part.tx_type = static_cast<std::uint8_t>(plan.tx_type);
      syntax_part.sec_tx_type = static_cast<std::uint8_t>(plan.sec_tx_type);
      syntax_part.most_probable_stx_set =
          static_cast<std::uint8_t>(plan.most_probable_stx_set);
      syntax_part.eob = static_cast<std::uint32_t>(plan.eob);
      parts->push_back(syntax_part);
      plans->push_back(std::move(plan));
    }
    const int x0 = c * 4;
    const int y0 = r * 4;
    const int max_x = std::min(x0 + w, mi_cols_ * 4);
    const int max_y = std::min(y0 + h, mi_rows_ * 4);
    double sse = 0.0;
    for (int yy = y0; yy < max_y; ++yy) {
      for (int xx = x0; xx < max_x; ++xx) {
        const double d = static_cast<double>(source_sample(0, xx, yy)) -
                         static_cast<double>(recon_[0].at(xx, yy));
        sse += d * d;
      }
    }
    *block_sse = sse;
    return true;
  }

void Encoder::write_recon(int plane, int x0, int y0, int cw, int ch,
                 const TxPlan& plan) {
    // A coding block at the right/bottom frame edge may be larger than the
    // MiCols*4 / MiRows*4 coded plane: the decoder parses the full transform
    // block but only stores the part inside the plane. Clamp identically.
    const int max_x = recon_[plane].width;
    const int max_y = recon_[plane].height;
    for (int y = 0; y < ch; ++y) {
      const int py = y0 + y;
      if (py < 0 || py >= max_y) continue;
      for (int x = 0; x < cw; ++x) {
        const int px = x0 + x;
        if (px < 0 || px >= max_x) continue;
        recon_[plane].at(px, py) =
            plan.recon[static_cast<std::size_t>(y) *
                           static_cast<std::size_t>(cw) +
                       static_cast<std::size_t>(x)];
      }
    }
  }

void Encoder::fill_deblock_mi_parts(int mi_row, int mi_col, partition::BlockSize size,
                           int cw, int ch,
                           const std::vector<syntax::LumaTxPart>& parts,
                           int partition, int chroma_tx_size, bool has_chroma,
                           int chroma_mi_row, int chroma_mi_col,
                           partition::BlockSize chroma_bsize,
                           bool do_luma) {
    const int max_tx = tpart::max_tx_size_rect(static_cast<int>(size));
    fill_deblock_mi(mi_row, mi_col, size, cw, ch, max_tx, chroma_tx_size,
                    has_chroma, chroma_mi_row, chroma_mi_col, chroma_bsize,
                    do_luma);
    tpart::SplitGeometry geometry;
    std::string geometry_error;
    if (!tpart::tx_partition_geometry(static_cast<int>(size), mi_row, mi_col,
                                      static_cast<tpart::TxPartition>(partition),
                                      &geometry, geometry_error)) {
      return;
    }
    if (geometry.count != static_cast<int>(parts.size())) return;
    for (int i = 0; i < geometry.count; ++i) {
      const tpart::SplitPart& part = geometry.parts[i];
      const int w4 = tpart::tx_width(part.tx_size) / 4;
      const int h4 = tpart::tx_height(part.tx_size) / 4;
      for (int yy = 0; yy < h4; ++yy) {
        const int rr = part.mi_row + yy;
        if (rr >= mi_rows_) break;
        for (int xx = 0; xx < w4; ++xx) {
          const int cc = part.mi_col + xx;
          if (cc >= mi_cols_) break;
          deblock::DeblockMi& mi = deblock_mi_[0][static_cast<std::size_t>(
              rr * deblock_cols_ + cc)];
          mi.tx_size = static_cast<TxSize>(parts[static_cast<std::size_t>(i)]
                                               .tx_size);
          mi.tx_col_base = static_cast<std::uint16_t>(part.mi_col);
          mi.tx_row_base = static_cast<std::uint16_t>(part.mi_row);
        }
      }
    }
  }

void Encoder::fill_deblock_mi(int mi_row, int mi_col, partition::BlockSize size,
                     int cw, int ch, int luma_tx_size,
                     int chroma_tx_size, bool has_chroma,
                     int chroma_mi_row, int chroma_mi_col,
                     partition::BlockSize chroma_bsize, bool do_luma) {
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    const int w4 = w / 4;
    const int h4 = h / 4;
    const TxSize luma = static_cast<TxSize>(luma_tx_size);
    const TxSize chroma = static_cast<TxSize>(chroma_tx_size);
    (void)cw;
    (void)ch;
    const int clamped_w4 = std::min(w4, mi_cols_ - mi_col);
    const int clamped_h4 = std::min(h4, mi_rows_ - mi_row);
    if (do_luma) {
      for (int yy = 0; yy < h4; ++yy) {
        const int rr = mi_row + yy;
        if (rr >= mi_rows_) break;
        for (int xx = 0; xx < w4; ++xx) {
          const int cc = mi_col + xx;
          if (cc >= mi_cols_) break;
          const int idx = rr * deblock_cols_ + cc;
          deblock::DeblockMi mi;
          mi.block_col_base = static_cast<std::uint16_t>(mi_col);
          mi.block_row_base = static_cast<std::uint16_t>(mi_row);
          mi.block_w4 = static_cast<std::uint8_t>(clamped_w4);
          mi.block_h4 = static_cast<std::uint8_t>(clamped_h4);
          mi.sub_pu_size = luma;
          mi.tx_size = luma;
          mi.tx_col_base = static_cast<std::uint16_t>(mi_col);
          mi.tx_row_base = static_cast<std::uint16_t>(mi_row);
          mi.sub_pu_col_base = static_cast<std::uint16_t>(mi_col);
          mi.sub_pu_row_base = static_cast<std::uint16_t>(mi_row);
          mi.skip = 0;
          mi.region_intra = 1;
          mi.lossless = 0;
          // §7.17 deblock strength uses the block's effective qindex, which the
          // per-SB segmentation path changes.
          mi.qindex = static_cast<std::uint16_t>(current_qindex_);
          deblock_mi_[0][static_cast<std::size_t>(idx)] = mi;
        }
      }
    }
    if (!has_chroma) return;
    // Chroma metadata is owned by the ChromaMi block (not the luma leaf) so a
    // sub-8 leaf group deblocks as one 8x8 chroma block.
    const int c_bsize_w = partition::block_width(chroma_bsize) / 4;
    const int c_bsize_h = partition::block_height(chroma_bsize) / 4;
    const int c_w4 = std::min(c_bsize_w, mi_cols_ - chroma_mi_col);
    const int c_h4 = std::min(c_bsize_h, mi_rows_ - chroma_mi_row);
    for (int yy = 0; yy < c_bsize_h; ++yy) {
      const int rr = chroma_mi_row + yy;
      if (rr >= mi_rows_) break;
      for (int xx = 0; xx < c_bsize_w; ++xx) {
        const int cc = chroma_mi_col + xx;
        if (cc >= mi_cols_) break;
        const int idx = rr * deblock_cols_ + cc;
        deblock::DeblockMi mi = deblock_mi_[0][static_cast<std::size_t>(idx)];
        mi.block_col_base = static_cast<std::uint16_t>(chroma_mi_col);
        mi.block_row_base = static_cast<std::uint16_t>(chroma_mi_row);
        mi.block_w4 = static_cast<std::uint8_t>(c_w4);
        mi.block_h4 = static_cast<std::uint8_t>(c_h4);
        mi.tx_size = chroma;
        mi.tx_col_base = static_cast<std::uint16_t>(chroma_mi_col);
        mi.tx_row_base = static_cast<std::uint16_t>(chroma_mi_row);
        // §7.17 deblock strength uses the block's effective qindex, which the
        // per-SB segmentation path changes.
        mi.qindex = static_cast<std::uint16_t>(current_qindex_);
        deblock_mi_[1][static_cast<std::size_t>(idx)] = mi;
        deblock_mi_[2][static_cast<std::size_t>(idx)] = mi;
      }
    }
  }

}  // namespace detail
}  // namespace av2enc::still
