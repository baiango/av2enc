#include "codec/intra/ibc/core/ibc.h"

#include <algorithm>
#include <cstring>

namespace av2enc::ibc {
namespace {

int block_w(BlockSize bsize) { return partition::block_width(bsize); }
int block_h(BlockSize bsize) { return partition::block_height(bsize); }

bool stack_contains(const BvStack& stack, BlockVector bv) {
  for (int i = 0; i < stack.count; ++i) {
    if (stack.candidates[static_cast<std::size_t>(i)].bv == bv) return true;
  }
  return false;
}

void push_candidate(BvStack& stack, int capacity, BlockVector bv,
                    bool neighbour, bool dedupe) {
  if (stack.count >= capacity || stack.count >= 8) return;
  if (dedupe && stack_contains(stack, bv)) return;
  BvCandidate& c = stack.candidates[static_cast<std::size_t>(stack.count++)];
  c.bv = bv;
  c.neighbour = neighbour;
  c.valid = false;
}

}  // namespace

int stack_capacity(const IbcConfig& config) noexcept {
  return std::min(8, std::max(1, config.max_bvp_drl_bits_minus_1 + 2));
}

void IbcBufferState::reset(int cols, int rows) {
  mi_cols = std::max(0, cols);
  mi_rows = std::max(0, rows);
  cur_buf_row = 0;
  cur_buf_col = 0;
  entry_row.fill(-1);
  entry_col.fill(-1);
  entry_valid.fill(false);
  coded.assign(static_cast<std::size_t>(mi_cols) *
                   static_cast<std::size_t>(mi_rows),
               0);
}

void IbcBufferState::update_buffers(int mi_row, int mi_col) noexcept {
  if (mi_cols <= 0 || mi_rows <= 0) return;
  const int buf_row = mi_row >> (kIbcBufferSizeLog2 - kMiSizeLog2);
  const int buf_col = mi_col >> (kIbcBufferSizeLog2 - kMiSizeLog2);
  if (buf_row == cur_buf_row && buf_col == cur_buf_col) return;
  const int idx = ibc_buffer_index(cur_buf_row, cur_buf_col, 64);
  entry_row[static_cast<std::size_t>(idx)] = cur_buf_row;
  entry_col[static_cast<std::size_t>(idx)] = cur_buf_col;
  entry_valid[static_cast<std::size_t>(idx)] = true;
  cur_buf_row = buf_row;
  cur_buf_col = buf_col;
}

void IbcBufferState::mark_coded(int mi_row, int mi_col, int bw,
                                int bh) noexcept {
  if (mi_cols <= 0 || mi_rows <= 0) return;
  const int w4 = std::max(1, bw / kMiSize);
  const int h4 = std::max(1, bh / kMiSize);
  for (int y = 0; y < h4; ++y) {
    const int r = mi_row + y;
    if (r >= mi_rows) break;
    for (int x = 0; x < w4; ++x) {
      const int c = mi_col + x;
      if (c >= mi_cols) break;
      coded[static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols) +
            static_cast<std::size_t>(c)] = 1;
    }
  }
}

bool is_bv_valid(const FrameView& frame, const IbcConfig& config,
                 const IbcBufferState& state, int mi_row, int mi_col,
                 int bsize, BlockVector bv, std::string* reason) {
  const auto size = static_cast<BlockSize>(bsize);
  const int bw = block_w(size);
  const int bh = block_h(size);
  if (bw <= 0 || bh <= 0 ||
      (!state.coded.empty() &&
       (mi_row < 0 || mi_col < 0 || mi_row >= state.mi_rows ||
        mi_col >= state.mi_cols))) {
    if (reason != nullptr) *reason = "invalid IBC block position";
    return false;
  }
  const int mi_cols = state.mi_cols > 0 ? state.mi_cols : 1;
  const int mi_rows = state.mi_rows > 0 ? state.mi_rows : 1;
  const int total_sb64_per_row = ((mi_cols - 1) >> 4) + 1;
  const int delta_row = bv.row >> 3;
  const int delta_col = bv.col >> 3;
  bool local_ok = true;
  if (config.allow_local) {
    // check_valid_local_ibc() at the top-left and bottom-right corners.  The
    // shared partition uses the luma MI position for ChromaMiRow/Col.
    const int bottom_border = (bv.row & 7) != 0 ? 1 : 0;
    const int right_border = (bv.col & 7) != 0 ? 1 : 0;
    const int top = mi_row * kMiSize + delta_row;
    const int left = mi_col * kMiSize + delta_col;
    const int bottom = top + bh - 1 + bottom_border;
    const int right = left + bw - 1 + right_border;
    const auto check = [&](int x, int y) {
      if (x < 0 || y < 0 || (y >> kMiSizeLog2) >= state.mi_rows ||
          (x >> kMiSizeLog2) >= state.mi_cols) {
        return false;
      }
      return check_valid_local_ibc(
          x, y, mi_col, mi_row, state.coded.data(), state.mi_cols,
          state.cur_buf_row, state.cur_buf_col, state.entry_row.data(),
          state.entry_col.data(), state.entry_valid.data(), config.sb_size);
    };
    local_ok = check(left, top) && check(right, bottom);
  }
  if (config.allow_local && state.coded.empty()) {
    if (reason != nullptr) *reason = "local IBC state is not initialized";
    return false;
  }
  int code = 0;
  const bool ok = bv_geometry_valid(
      mi_row, mi_col, bw, bh, 0, 0, mi_rows, mi_cols, bv.row, bv.col,
      frame.has_chroma, mi_row, mi_col, config.allow_local,
      config.allow_global, config.allow_extra_ibc_range, local_ok,
      config.sb_size, total_sb64_per_row, mi_rows, mi_cols, &code);
  if (!ok && reason != nullptr) {
    switch (code) {
      case 1:
        *reason = "source outside the tile";
        break;
      case 2:
        *reason = "source not in a valid local IBC buffer";
        break;
      case 3:
        *reason = config.allow_global
                      ? "source inside the INTRABC delay window"
                      : "global IBC is disabled";
        break;
      case 4:
        *reason = "source violates the global wavefront constraint";
        break;
      default:
        *reason = "invalid block vector";
        break;
    }
  }
  return ok;
}

BvStack build_stack(const FrameView& frame, const IbcConfig& config,
                    const IbcBufferState& state, int mi_row, int mi_col,
                    int bsize, const NeighbourBvs& neighbours) {
  BvStack stack;
  const int capacity = stack_capacity(config);
  const auto size = static_cast<BlockSize>(bsize);
  const int bw = block_w(size);
  const int bh = block_h(size);

  // Neighbour BVs are scanned before the default IBC candidates (§7.12.2.20);
  // intra neighbours contribute nothing, so callers pass only IBC neighbours.
  if (neighbours.above != nullptr) {
    push_candidate(stack, capacity, *neighbours.above, true, true);
  }
  if (neighbours.left != nullptr) {
    push_candidate(stack, capacity, *neighbours.left, true, true);
  }
  if (neighbours.below_left != nullptr) {
    push_candidate(stack, capacity, *neighbours.below_left, true, true);
  }
  if (neighbours.above_right != nullptr) {
    push_candidate(stack, capacity, *neighbours.above_right, true, true);
  }

  // {row, col} in 1/8 luma samples.  add_to_ref_bv(dx, dy) stores row=dy.
  // The four IBC defaults are appended without pruning, exactly as
  // add_to_ref_bv() does (a 32x64 block repeats (0,-64)).
  push_candidate(stack, capacity, BlockVector{-config.sb_size * 8, 0}, false,
                 false);
  push_candidate(
      stack, capacity,
      BlockVector{0, -(config.sb_size + config.delay_pixels) * 8}, false,
      false);
  push_candidate(stack, capacity, BlockVector{-bh * 8, 0}, false, false);
  push_candidate(stack, capacity, BlockVector{0, -bw * 8}, false, false);

  for (int i = 0; i < stack.count; ++i) {
    stack.candidates[static_cast<std::size_t>(i)].valid =
        is_bv_valid(frame, config, state, mi_row, mi_col, bsize,
                    stack.candidates[static_cast<std::size_t>(i)].bv, nullptr);
  }
  return stack;
}

int find_stack_index(const BvStack& stack, BlockVector bv) noexcept {
  for (int i = 0; i < stack.count; ++i) {
    if (stack.candidates[static_cast<std::size_t>(i)].bv == bv) return i;
  }
  return -1;
}

bool copy_block(const FrameView& frame, int mi_row, int mi_col, int bsize,
                BlockVector bv, std::uint8_t* dst_y, int dst_stride_y,
                std::uint8_t* dst_u, int dst_stride_u, std::uint8_t* dst_v,
                int dst_stride_v, int visible_w, int visible_h) {
  const auto size = static_cast<BlockSize>(bsize);
  const int bw = block_w(size);
  const int bh = block_h(size);
  if (bw <= 0 || bh <= 0 || visible_w <= 0 || visible_h <= 0 ||
      visible_w > bw || visible_h > bh) {
    return false;
  }
  const int x = mi_col * kMiSize;
  const int y = mi_row * kMiSize;
  const int dx = plane_offset(bv.col, 0);
  const int dy = plane_offset(bv.row, 0);
  const int sx = x + dx;
  const int sy = y + dy;
  if (frame.y.data == nullptr || sx < 0 || sy < 0 ||
      sx + visible_w > frame.y.width || sy + visible_h > frame.y.height) {
    return false;
  }
  const std::uint8_t* src =
      frame.y.data + static_cast<std::size_t>(sy) *
                         static_cast<std::size_t>(frame.y.stride) +
      static_cast<std::size_t>(sx);
  copy_plane(src, frame.y.stride, dst_y, dst_stride_y, visible_w, visible_h);

  if (frame.has_chroma && dst_u != nullptr && dst_v != nullptr) {
    const int cdx = plane_offset(bv.col, frame.sub_x);
    const int cdy = plane_offset(bv.row, frame.sub_y);
    const int cx = (x >> frame.sub_x) + cdx;
    const int cy = (y >> frame.sub_y) + cdy;
    const int cw = (visible_w + (1 << frame.sub_x) - 1) >> frame.sub_x;
    const int ch = (visible_h + (1 << frame.sub_y) - 1) >> frame.sub_y;
    if (frame.u.data == nullptr || frame.v.data == nullptr || cx < 0 || cy < 0 ||
        cx + cw > frame.u.width || cy + ch > frame.u.height) {
      return false;
    }
    for (int plane = 0; plane < 2; ++plane) {
      const PlaneView& p = plane == 0 ? frame.u : frame.v;
      std::uint8_t* d = plane == 0 ? dst_u : dst_v;
      const int ds = plane == 0 ? dst_stride_u : dst_stride_v;
      const std::uint8_t* s =
          p.data + static_cast<std::size_t>(cy) *
                       static_cast<std::size_t>(p.stride) +
          static_cast<std::size_t>(cx);
      copy_plane(s, p.stride, d, ds, cw, ch);
    }
  }
  return true;
}

void copy_then_residual(const FrameView& frame, int mi_row, int mi_col,
                        int bsize, BlockVector bv, std::int16_t* residual_y,
                        int res_stride_y, std::int16_t* residual_u,
                        int res_stride_u, std::int16_t* residual_v,
                        int res_stride_v, int visible_w, int visible_h) {
  (void)bsize;
  const int x = mi_col * kMiSize;
  const int y = mi_row * kMiSize;
  const int dx = plane_offset(bv.col, 0);
  const int dy = plane_offset(bv.row, 0);
  const int sx = x + dx;
  const int sy = y + dy;
  for (int i = 0; i < visible_h; ++i) {
    for (int j = 0; j < visible_w; ++j) {
      const int src = frame.y.data[static_cast<std::size_t>(sy + i) *
                                       static_cast<std::size_t>(frame.y.stride) +
                                   static_cast<std::size_t>(sx + j)];
      const int cur = frame.y.data[static_cast<std::size_t>(y + i) *
                                       static_cast<std::size_t>(frame.y.stride) +
                                   static_cast<std::size_t>(x + j)];
      residual_y[static_cast<std::size_t>(i) *
                     static_cast<std::size_t>(res_stride_y) +
                 static_cast<std::size_t>(j)] =
          static_cast<std::int16_t>(cur - src);
    }
  }
  if (!frame.has_chroma || residual_u == nullptr || residual_v == nullptr) {
    return;
  }
  const int cdx = plane_offset(bv.col, frame.sub_x);
  const int cdy = plane_offset(bv.row, frame.sub_y);
  const int cx = (x >> frame.sub_x) + cdx;
  const int cy = (y >> frame.sub_y) + cdy;
  const int cw = (visible_w + (1 << frame.sub_x) - 1) >> frame.sub_x;
  const int ch = (visible_h + (1 << frame.sub_y) - 1) >> frame.sub_y;
  for (int plane = 0; plane < 2; ++plane) {
    const PlaneView& p = plane == 0 ? frame.u : frame.v;
    std::int16_t* r = plane == 0 ? residual_u : residual_v;
    const int rs = plane == 0 ? res_stride_u : res_stride_v;
    const int px = (x >> frame.sub_x);
    const int py = (y >> frame.sub_y);
    for (int i = 0; i < ch; ++i) {
      for (int j = 0; j < cw; ++j) {
        const int s = p.data[static_cast<std::size_t>(cy + i) *
                                 static_cast<std::size_t>(p.stride) +
                             static_cast<std::size_t>(cx + j)];
        const int c = p.data[static_cast<std::size_t>(py + i) *
                                 static_cast<std::size_t>(p.stride) +
                             static_cast<std::size_t>(px + j)];
        r[static_cast<std::size_t>(i) * static_cast<std::size_t>(rs) +
          static_cast<std::size_t>(j)] = static_cast<std::int16_t>(c - s);
      }
    }
  }
}

bool predict_copy(const FrameView& frame, int mi_row, int mi_col, int bsize,
                  BlockVector bv, int visible_w, int visible_h,
                  CopyResult& out, std::string* error) {
  const auto size = static_cast<BlockSize>(bsize);
  const int bw = block_w(size);
  const int bh = block_h(size);
  out.y.assign(static_cast<std::size_t>(bw) * static_cast<std::size_t>(bh), 0);
  std::uint8_t* u = nullptr;
  std::uint8_t* v = nullptr;
  int us = 0;
  int vs = 0;
  if (frame.has_chroma) {
    const int cw = (bw + (1 << frame.sub_x) - 1) >> frame.sub_x;
    const int ch = (bh + (1 << frame.sub_y) - 1) >> frame.sub_y;
    out.u.assign(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch), 0);
    out.v.assign(out.u.size(), 0);
    u = out.u.data();
    v = out.v.data();
    us = cw;
    vs = cw;
  }
  if (!copy_block(frame, mi_row, mi_col, bsize, bv, out.y.data(), bw, u, us, v,
                  vs, visible_w, visible_h)) {
    if (error != nullptr) *error = "IBC source block is outside the plane";
    return false;
  }
  return true;
}

Match search_block(const FrameView& frame, const IbcConfig& config,
                   const IbcBufferState& state, int mi_row, int mi_col,
                   int bsize, const SearchOptions& options,
                   const NeighbourBvs* neighbours) {
  Match best;
  const auto size = static_cast<BlockSize>(bsize);
  const int bw = block_w(size);
  const int bh = block_h(size);
  const int x = mi_col * kMiSize;
  const int y = mi_row * kMiSize;
  const BvStack stack =
      build_stack(frame, config, state, mi_row, mi_col, bsize,
                  neighbours != nullptr ? *neighbours : NeighbourBvs{});
  const auto evaluate = [&](BlockVector bv) {
    if (!is_bv_valid(frame, config, state, mi_row, mi_col, bsize, bv, nullptr)) {
      return;
    }
    const int dx = plane_offset(bv.col, 0);
    const int dy = plane_offset(bv.row, 0);
    const int sx = x + dx;
    const int sy = y + dy;
    if (sx < 0 || sy < 0 || sx + bw > frame.y.width ||
        sy + bh > frame.y.height) {
      return;
    }
    const std::uint8_t* src =
        frame.y.data + static_cast<std::size_t>(sy) *
                           static_cast<std::size_t>(frame.y.stride) +
        static_cast<std::size_t>(sx);
    const std::uint8_t* cur =
        frame.y.data + static_cast<std::size_t>(y) *
                           static_cast<std::size_t>(frame.y.stride) +
        static_cast<std::size_t>(x);
    std::int64_t sad = sad_block(src, frame.y.stride, cur, frame.y.stride, bw,
                                 bh);
    if (options.compare_chroma && frame.has_chroma) {
      const int cdx = plane_offset(bv.col, frame.sub_x);
      const int cdy = plane_offset(bv.row, frame.sub_y);
      const int cw = bw >> frame.sub_x;
      const int ch = bh >> frame.sub_y;
      const int cx = (x >> frame.sub_x) + cdx;
      const int cy = (y >> frame.sub_y) + cdy;
      if (cx >= 0 && cy >= 0 && cx + cw <= frame.u.width &&
          cy + ch <= frame.u.height) {
        for (int plane = 0; plane < 2; ++plane) {
          const PlaneView& p = plane == 0 ? frame.u : frame.v;
          sad += sad_block(p.data + static_cast<std::size_t>(cy) *
                                        static_cast<std::size_t>(p.stride) +
                            static_cast<std::size_t>(cx),
                          p.stride,
                          p.data + static_cast<std::size_t>(y >> frame.sub_y) *
                                        static_cast<std::size_t>(p.stride) +
                              static_cast<std::size_t>(x >> frame.sub_x),
                          p.stride, cw, ch);
        }
      }
    }
    if (!best.found || sad < best.sad) {
      best.found = true;
      best.bv = bv;
      best.sad = sad;
      best.stack_index = find_stack_index(stack, bv);
    }
  };

  if (options.stack_only) {
    for (int i = 0; i < stack.count; ++i) {
      if (stack.candidates[static_cast<std::size_t>(i)].valid) {
        evaluate(stack.candidates[static_cast<std::size_t>(i)].bv);
      }
    }
  } else {
    const int step = std::max(1, options.step);
    for (int sy = 0; sy + bh <= frame.y.height; sy += step) {
      for (int sx = 0; sx + bw <= frame.y.width; sx += step) {
        const BlockVector bv{(sy - y) * 8, (sx - x) * 8};
        evaluate(bv);
      }
    }
  }
  if (best.found && options.max_sad > 0 && best.sad > options.max_sad) {
    best.found = false;
    best.stack_index = -1;
  }
  return best;
}

std::int64_t block_sse(const std::uint8_t* a, int a_stride,
                       const std::uint8_t* b, int b_stride, int w, int h) {
  std::int64_t sse = 0;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int d = static_cast<int>(
                        a[static_cast<std::size_t>(i) *
                              static_cast<std::size_t>(a_stride) +
                          static_cast<std::size_t>(j)]) -
                    static_cast<int>(
                        b[static_cast<std::size_t>(i) *
                              static_cast<std::size_t>(b_stride) +
                          static_cast<std::size_t>(j)]);
      sse += static_cast<std::int64_t>(d) * d;
    }
  }
  return sse;
}

FrameView SyntheticScreen::view() const {
  FrameView out;
  out.y = PlaneView{y.data(), width, width, height};
  out.u = PlaneView{u.data(), width / 2, width / 2, height / 2};
  out.v = PlaneView{v.data(), width / 2, width / 2, height / 2};
  out.has_chroma = true;
  out.sub_x = 1;
  out.sub_y = 1;
  return out;
}

SyntheticScreen make_synthetic_screen(int width, int height) {
  SyntheticScreen s;
  s.width = width;
  s.height = height;
  const std::size_t ysize =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  s.y.assign(ysize, 40);
  s.u.assign(ysize / 4, 128);
  s.v.assign(ysize / 4, 128);
  // Repeated 64x32 tile row: two-colour stripes and a solid icon.  This is the
  // content IBC is designed for; a block copy is exact.
  const int stripe_end = std::min(64, height);
  for (int row = 32; row < stripe_end; ++row) {
    for (int col = 0; col < width; ++col) {
      std::uint8_t v = ((col / 8) & 1) ? 220 : 20;
      if (row >= 40 && row < 56 && (col % 64) >= 24 && (col % 64) < 40) {
        v = 160;
      }
      s.y[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(col)] = v;
    }
  }
  // A text-like band and flat UI panel.
  const int text_end = std::min(120, height);
  for (int row = 96; row < text_end; ++row) {
    for (int col = 8; col < width - 8; ++col) {
      s.y[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(col)] =
          ((col / 2) % 7 == 0) ? 250 : 235;
    }
  }
  return s;
}

}  // namespace av2enc::ibc
