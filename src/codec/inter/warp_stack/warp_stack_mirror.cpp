#include "codec/inter/warp_stack/warp_stack_mirror.h"

#include <algorithm>
#include <vector>

namespace av2enc::inter::warp::mirror {
namespace {

struct Event {
  inter::mc::Mv mv;
  int weight = 1;
  int row_offset = 0;
  int col_offset = 0;
};

struct Fold {
  std::vector<Event> events;
  int prune = 0;
  bool full() const {
    return static_cast<int>(events.size()) >= kMaxRefMvStackSize;
  }
  void add(Event e) {
    int hit = -1;
    if (prune < kMaxPrNum) {
      for (std::size_t i = 0; i < events.size(); ++i) {
        ++prune;
        if (events[i].mv == e.mv) {
          hit = static_cast<int>(i);
          break;
        }
      }
    }
    if (hit >= 0) {
      events[static_cast<std::size_t>(hit)].weight += e.weight;
      return;
    }
    if (!full()) events.push_back(e);
  }
};

const WarpMi* cell(const WarpGrid& g, int r, int c) {
  if (!g.inside(r, c)) return nullptr;
  return &g.at(r, c);
}

int block_w(const WarpMi& m) { return std::max<int>(1, m.w4) * kMiSize; }
int block_h(const WarpMi& m) { return std::max<int>(1, m.h4) * kMiSize; }

// Local re-derivation of §7.12.2.4 get_warp_motion_vector_xy_pos and
// §7.12.2.11 insert warp candidate.  Keeping both here (rather than calling the
// primary helpers) is what makes the randomized lockstep a parity test rather
// than a tautology.
inter::mc::Mv xy_pos_mv(const ia::WarpParams& mat, int pos_row, int pos_col,
                        int mi_row, int mi_col, int bw4, int bh4, int mi_rows,
                        int mi_cols) {
  const long long x = static_cast<long long>(pos_col) * kMiSize;
  const long long y = static_cast<long long>(pos_row) * kMiSize;
  const long long xc = static_cast<long long>(mat.p[2]) * x +
                       static_cast<long long>(mat.p[3]) * y + mat.p[0] -
                       (x << ia::kWarpedModelPrecBits);
  const long long yc = static_cast<long long>(mat.p[4]) * x +
                       static_cast<long long>(mat.p[5]) * y + mat.p[1] -
                       (y << ia::kWarpedModelPrecBits);
  int my = static_cast<int>(ia::round2_signed64(yc, ia::kWarpedModelPrecBits - 3));
  int mx = static_cast<int>(ia::round2_signed64(xc, ia::kWarpedModelPrecBits - 3));
  my = ia::clip3(ia::kMvLow + 1, ia::kMvUpp - 1, my);
  mx = ia::clip3(ia::kMvLow + 1, ia::kMvUpp - 1, mx);
  my = ia::clamp_mv_row(my, mi_row, bh4, mi_rows);
  mx = ia::clamp_mv_col(mx, mi_col, bw4, mi_cols);
  return inter::mc::Mv{my, mx};
}

void append_warp(const ia::WarpParams& params, WarpStack* stack) {
  if (stack->num_warp_found >= kMaxWarpRefCandidates) return;
  stack->warp_params[stack->num_warp_found] = params;
  ++stack->num_warp_found;
}

inter::mc::Mv candidate_mv(const WarpStackParams& p, const WarpMi& m, int list) {
  const bool gm = m.y_mode == kYModeGlobalMv || m.y_mode == kYModeGlobalGlobalMv;
  if (gm && p.global_mv_nonidentity && m.ref[list] != kRefTip &&
      std::min(block_w(m), block_h(m)) >= 8) {
    return p.global_mv[0];
  }
  return inter::mc::Mv{m.sub_mv[list][0], m.sub_mv[list][1]};
}

// §7.12.2.4, transcribed independently.
struct CornerBuild {
  int found = 0;
  bool all_same = true;
  bool negative = false;
  int pt[3][2] = {};
  int mv[3][2] = {};
  int refp[3][2] = {};
  ia::WarpParams model{};

  void corner(const WarpGrid& g, int sb_h4, int mi_row, int mi_col, int bw4,
              int bh4, std::int8_t ref0, int dr, int dc, int adjust) {
    const bool border = (mi_row % sb_h4) == 0;
    const int row = mi_row + dr;
    const int col = mi_col + dc;
    const int adjusted = dc + adjust;
    const int lookup_col =
        (dr < 0 && border)
            ? (mi_col - (mi_col & 1)) + (adjusted - (adjusted & 1))
            : mi_col + adjusted;
    if (border && adjusted == 0 && bw4 <= 2) return;
    const WarpMi* n = cell(g, row, lookup_col);
    if (n == nullptr) return;
    for (int list = 0; list < 2; ++list) {
      if (!n->written || !n->is_inter) continue;
      if (n->ref[list] != ref0) continue;
      pt[found][0] = (row + 1) * kMiSize;
      pt[found][1] = (col + 1) * kMiSize;
      if (n->motion_mode >= kMotionLocalWarp) {
        if (list > 0) return;
        const inter::mc::Mv m =
            xy_pos_mv(n->warp[list], row + 1, col + 1, mi_row, mi_col, bw4, bh4,
                      g.mi_rows, g.mi_cols);
        mv[found][0] = m.y;
        mv[found][1] = m.x;
      } else {
        mv[found][0] = n->sub_mv[list][0];
        mv[found][1] = n->sub_mv[list][1];
      }
      ++found;
      return;
    }
  }

  bool finish(int bw4, int bh4) {
    if (found != 3) {
      all_same = false;
      return false;
    }
    all_same = (mv[0][0] == mv[1][0] && mv[0][0] == mv[2][0] &&
                mv[0][1] == mv[1][1] && mv[0][1] == mv[2][1]);
    for (int n = 0; n < 3; ++n) {
      for (int c = 0; c < 2; ++c) {
        refp[n][c] = static_cast<int>(
            (static_cast<long long>(pt[n][c]) << ia::kWarpedModelPrecBits) +
            (static_cast<long long>(mv[n][c]) << kGmTransOnlyPrecDiff));
        if (refp[n][c] < 0) negative = true;
      }
    }
    if (all_same || negative) return false;

    int wl = 0;
    for (int v = bw4 * kMiSize; v > 1; v >>= 1) ++wl;
    int hl = 0;
    for (int v = bh4 * kMiSize; v > 1; v >>= 1) ++hl;

    ia::WarpParams m{};
    m.p[2] = (refp[1][1] - refp[0][1]) >> wl;
    m.p[4] = (refp[1][0] - refp[0][0]) >> wl;
    m.p[3] = (refp[2][1] - refp[0][1]) >> hl;
    m.p[5] = (refp[2][0] - refp[0][0]) >> hl;
    const long long t0 = static_cast<long long>(refp[0][1]) -
                         static_cast<long long>(m.p[2]) * pt[0][1] -
                         static_cast<long long>(m.p[3]) * pt[0][0];
    const long long t1 = static_cast<long long>(refp[0][0]) -
                         static_cast<long long>(m.p[4]) * pt[0][1] -
                         static_cast<long long>(m.p[5]) * pt[0][0];
    model = ia::reduce_warp_model(m);
    model.p[0] = ia::clip3(-ia::kWarpedModelTransClamp,
                           ia::kWarpedModelTransClamp -
                               (1 << ia::kWarpParamReduceBits),
                           static_cast<int>(std::clamp<long long>(
                               t0, -2147483648LL, 2147483647LL)));
    model.p[1] = ia::clip3(-ia::kWarpedModelTransClamp,
                           ia::kWarpedModelTransClamp -
                               (1 << ia::kWarpParamReduceBits),
                           static_cast<int>(std::clamp<long long>(
                               t1, -2147483648LL, 2147483647LL)));
    return true;
  }
};

}  // namespace

Status build_stack(const WarpStackParams& p, WarpStack* out) {
  if (out == nullptr || p.mi == nullptr || p.mi_cols <= 0 || p.mi_rows <= 0 ||
      p.mi_row < 0 || p.mi_col < 0 || p.mi_row >= p.mi_rows ||
      p.mi_col >= p.mi_cols) {
    return Status::kInvalidArgument;
  }
  if (p.bw4 <= 0 || p.bh4 <= 0 || p.bw4 > 16 || p.bh4 > 16) {
    return Status::kInvalidBlockSize;
  }
  if (p.sb_h4 <= 0 || (p.sb_h4 & (p.sb_h4 - 1)) != 0) {
    return Status::kInvalidArgument;
  }
  if (!is_inter_ref_value(p.ref_frame[0])) return Status::kInvalidArgument;
  if (p.is_compound) return Status::kUnsupportedCompound;
  if (p.use_intrabc) return Status::kUnsupportedIntrabc;
  if (p.use_ref_frame_mvs || p.use_temporal_first) {
    return Status::kUnsupportedTemporal;
  }
  if (p.max_drl_bits_minus_1 < 0 || p.max_drl_bits_minus_1 > 4) {
    return Status::kInvalidArgument;
  }

  *out = WarpStack{};
  for (int i = 0; i < kMaxWarpRefCandidates; ++i) {
    out->warp_params[i] = ia::WarpParams{};
  }
  const WarpGrid g = p.grid();
  const bool border = (p.mi_row % p.sb_h4) == 0;

  // ---- §7.12.2.3/.4 ----
  auto corner_pass = [&](int adjust) {
    CornerBuild c;
    c.corner(g, p.sb_h4, p.mi_row, p.mi_col, p.bw4, p.bh4, p.ref_frame[0], -1,
             -1, adjust);
    c.corner(g, p.sb_h4, p.mi_row, p.mi_col, p.bw4, p.bh4, p.ref_frame[0], -1,
             p.bw4 - 1, adjust);
    c.corner(g, p.sb_h4, p.mi_row, p.mi_col, p.bw4, p.bh4, p.ref_frame[0],
             p.bh4 - 1, -1, 0);
    if (c.finish(p.bw4, p.bh4)) append_warp(c.model, out);
  };
  if (p.derive_wrl) {
    corner_pass(0);
    if (out->num_warp_found == 0 && p.bw4 <= 16) corner_pass(1);
  }

  Fold fold;

  // ---- §7.12.2.6 scan point ----
  auto scan = [&](int dr, int dc) {
    int row = p.mi_row + dr;
    int col = p.mi_col + dc;
    if (dr < 0 && border) {
      col = (col >= 0 ? col / 2 : -((-col + 1) / 2)) * 2;
      dc = col - p.mi_col;
    }
    const WarpMi* n = cell(g, row, col);
    if (n == nullptr || !n->written) return;

    int weight = 1;
    if (dr == -1 && dc == -1) {
      weight = 0;
    } else if (dc < -1) {
      weight = 0;
    }

    // §7.12.2.9 before the stack-full check.
    if (p.derive_wrl && n->is_inter && n->motion_mode >= kMotionLocalWarp &&
        n->ref[0] == p.ref_frame[0]) {
      append_warp(n->warp[0], out);
    }
    if (fold.full()) return;
    if (!n->is_inter) return;
    for (int list = 0; list < 2; ++list) {
      if (n->ref[list] == p.ref_frame[0]) {
        fold.add(Event{candidate_mv(p, *n, list), weight, row - p.mi_row,
                       col - p.mi_col});
      }
    }
  };

  scan(p.bh4 - 1, -1);
  scan(-1, std::max(0, p.bw4 - 1 - (border ? 1 : 0)));
  if (p.bh4 >= 2) scan(0, -1);
  if (p.bw4 >= (border ? 4 : 2)) scan(-1, 0);
  if (p.bh4 <= 16) scan(p.bh4, -1);
  if (p.bw4 <= 16) scan(-1, border ? std::max(2, p.bw4) : p.bw4);
  scan(-1, border ? -2 : -1);

  std::vector<Event> nearest = fold.events;

  // ---- §7.12.2.5 scan col ----
  auto scan_col_at = [&](int dr, int dc) {
    const int row = p.mi_row + dr;
    const int col = p.mi_col + dc;
    const WarpMi* n = cell(g, row, col);
    if (n == nullptr || p.mi_col - 1 < 0) return;
    const WarpMi& other = g.at(row, p.mi_col - 1);
    if (n->col_base == other.col_base) return;
    scan(dr, dc);
  };
  {
    int dc = -3;
    if (p.bw4 == 1) dc += p.mi_col & 1;
    scan_col_at(p.bh4 - 1, dc);
    if (p.bh4 > 1) scan_col_at(0, dc);
  }

  // ---- §7.12.2.19 sorting: one move-to-front over the nearest prefix ----
  // `nearest` is the stack size at §7.12.2 step 14.  Weights may have grown
  // during step 15, so the max is taken after scan_col from the live records.
  const std::size_t nearest_end = nearest.size();
  const bool use_sort =
      p.reorder == DrlReorder::kAlways ||
      (p.reorder == DrlReorder::kConstraint && !p.use_temporal_first &&
       nearest_end >= 4);
  out->sorted = use_sort;
  if (use_sort && nearest_end >= 2) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < nearest_end; ++i) {
      if (fold.events[best].weight < fold.events[i].weight) best = i;
    }
    if (best != 0) std::swap(fold.events[0], fold.events[best]);
  }
  out->num_nearest = static_cast<int>(nearest_end);

  // ---- §7.12.2.20/.23 clamping ----
  auto clamp = [&](inter::mc::Mv mv) {
    mv.y = ia::clip3(-(p.mi_row + p.bh4) * kMiSize * 8 - mc::kMvBorder,
                     (p.mi_rows - p.mi_row) * kMiSize * 8 + mc::kMvBorder,
                     mv.y);
    mv.x = ia::clip3(-(p.mi_col + p.bw4) * kMiSize * 8 - mc::kMvBorder,
                     (p.mi_cols - p.mi_col) * kMiSize * 8 + mc::kMvBorder,
                     mv.x);
    return mv;
  };
  for (Event& e : fold.events) e.mv = clamp(e.mv);

  // Global candidate.
  if (!fold.full()) {
    bool found = false;
    if (fold.prune < kMaxPrNum) {
      for (const Event& e : fold.events) {
        ++fold.prune;
        if (e.mv == p.global_mv[0]) {
          found = true;
          break;
        }
      }
    }
    if (!found) fold.events.push_back(Event{p.global_mv[0], 1, 0, 0});
  }

  // MVP combinations.
  if (p.bw4 * kMiSize > 32 && p.bh4 * kMiSize > 32) {
    const int num = static_cast<int>(fold.events.size());
    auto combine = [&](int y, int x) {
      if (y >= static_cast<int>(fold.events.size()) ||
          x >= static_cast<int>(fold.events.size()) || fold.full()) {
        return;
      }
      inter::mc::Mv cand{fold.events[static_cast<std::size_t>(y)].mv.y,
                         fold.events[static_cast<std::size_t>(x)].mv.x};
      if (fold.prune < kMaxPrNum) {
        for (const Event& e : fold.events) {
          ++fold.prune;
          if (e.mv == cand) return;
        }
      }
      fold.events.push_back(Event{cand, 1, 0, 0});
    };
    if (num > 1) {
      combine(0, 1);
      combine(1, 0);
    }
    if (num > 2) {
      combine(0, 2);
      combine(2, 0);
      combine(1, 2);
      combine(2, 1);
    }
  }

  // Warp stack tail.
  if (p.derive_wrl) {
    if (p.warp_bank != nullptr) {
      for (int i = p.warp_bank->size() - 1; i >= 0; --i) {
        append_warp(p.warp_bank->at(i), out);
      }
    }
    append_warp(p.gm_params, out);
    append_warp(ia::WarpParams{}, out);
    append_warp(ia::WarpParams{}, out);
  }

  // Final clamp and publish.
  out->count = static_cast<int>(fold.events.size());
  for (int i = 0; i < out->count; ++i) {
    out->cand[i].mv[0] = clamp(fold.events[static_cast<std::size_t>(i)].mv);
    out->cand[i].mv[1] = out->cand[i].mv[0];
    out->cand[i].weight = fold.events[static_cast<std::size_t>(i)].weight;
    out->cand[i].row_offset =
        fold.events[static_cast<std::size_t>(i)].row_offset;
    out->cand[i].col_offset =
        fold.events[static_cast<std::size_t>(i)].col_offset;
    out->cand[i].cwp = kCwpEqual;
  }
  return Status::kOk;
}

bool warp_samples(const WarpSampleParams& p, WarpSampleSet* out) {
  if (out == nullptr || p.grid.mi == nullptr || p.w4 <= 0 || p.h4 <= 0 ||
      p.ref < 0 || p.ref > 1 || p.mi_row < 0 || p.mi_col < 0 ||
      p.mi_row >= p.mi_rows || p.mi_col >= p.mi_cols) {
    return false;
  }
  *out = WarpSampleSet{};

  auto stored_above = [&](int dc) {
    if (!p.grid.inside(p.mi_row - 1, p.mi_col + dc)) return false;
    if ((p.mi_row % p.sb_h4) != 0) return true;
    if (((p.mi_col + dc) % 2) == 0) return true;
    const int w = std::max<int>(1, p.grid.at(p.mi_row - 1, p.mi_col + dc).w4);
    if (w == 1) return false;
    return p.mi_col + dc + 1 < p.mi_cols;
  };
  auto append = [&](int dr, int dc) {
    if (out->num_samples >= kLeastSquaresSamplesMax) return;
    if (!p.grid.inside(p.mi_row + dr, p.mi_col + dc)) return;
    const WarpMi& n = p.grid.at(p.mi_row + dr, p.mi_col + dc);
    if (!n.written) return;
    for (int list = 0; list < 2; ++list) {
      if (n.ref[list] != p.ref_frame[p.ref]) continue;
      const int cw = std::max<int>(1, n.w4);
      const int ch = std::max<int>(1, n.h4);
      const int my = n.row_base * kMiSize + ch * 2 - 1;
      const int mx = n.col_base * kMiSize + cw * 2 - 1;
      out->cand[out->num_samples][0] = my * 8;
      out->cand[out->num_samples][1] = mx * 8;
      out->cand[out->num_samples][2] = my * 8 + n.mv[list][0];
      out->cand[out->num_samples][3] = mx * 8 + n.mv[list][1];
      ++out->num_samples;
      if (out->num_samples >= kLeastSquaresSamplesMax) return;
    }
  };

  bool top_left = true;
  bool top_right = true;
  if (p.avail_u && p.mi_row > 0) {
    const int off =
        p.grid.at(p.mi_row - 1, p.mi_col).col_base - p.mi_col;
    if (off < 0) top_left = false;
    int i = off;
    while (i < std::min(p.w4, p.mi_cols - p.mi_col)) {
      const int w = std::max<int>(1, p.grid.at(p.mi_row - 1, p.mi_col + i).w4);
      if (stored_above(i)) append(-1, i);
      i += w;
    }
    top_right = (i == p.w4) && (i < p.mi_cols - p.mi_col);
  }
  if (p.avail_l && p.mi_col > 0) {
    const int off = p.grid.at(p.mi_row, p.mi_col - 1).row_base - p.mi_row;
    if (off < 0) top_left = false;
    int i = off;
    while (i < std::min(p.h4, p.mi_rows - p.mi_row)) {
      const int h = std::max<int>(1, p.grid.at(p.mi_row + i, p.mi_col - 1).h4);
      append(i, -1);
      i += h;
    }
  }
  if (top_left && stored_above(-1)) append(-1, -1);
  if (top_right && p.w4 <= 16 && stored_above(p.w4)) append(-1, p.w4);
  return out->num_samples > 0;
}

}  // namespace av2enc::inter::warp::mirror
