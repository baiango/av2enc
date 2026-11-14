// Tests for the AV2 M2 skip / merge / DRL module.
//
// Build/run: ./build_and_test.sh
//
// Coverage:
//   * skip_mode_params() for 1..4 reference frames and restricted order hints
//   * skip_mode_frames() neighbour/TIP overrides and the frame-level fallback
//   * skip_mode block eligibility (is_comp_ref_allowed, segmentation, region)
//   * read_drl_idx() encode/decode for every start/m/index, truncation and
//     invalid indices
//   * single_mode decoding and skip vs merge vs NEWMV resolution
//   * §7.11.2 NewMvContext and the skip / skip_flag contexts
//   * spec-exact MV stack for every 4..64 luma block size: scan order, weights,
//     dedup, weight-sort tie-breaks, DRL_REORDER_DISABLED/CONSTRAINT/ALWAYS
//   * §7.12.2.20 MVP-combination gate: every block size, plus an audit of the
//     shared inter::mc::build_mv_stack component
//   * compound (skip_mode) stack, derived-SMVP projection for multi-reference
//     frames, and invalid DRL/mode combinations

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <cstdio>
#include <string>
#include <vector>

#include "codec/inter/merge/merge.h"

namespace merge = av2enc::video::merge;
namespace mc = av2enc::inter::mc;
using merge::BlockSize;
namespace partition = av2enc::partition;

namespace {

struct Test {
  int checks = 0;
  int failures = 0;
  void expect(bool cond, const std::string& what) {
    ++checks;
    if (!cond) {
      ++failures;
      if (failures <= 30) {
        std::printf("FAIL: %s\n", what.c_str());
      }
    }
  }
  void eq_int(long long got, long long want, const std::string& what) {
    ++checks;
    if (got != want) {
      ++failures;
      if (failures <= 30) {
        std::printf("FAIL: %s (got %lld, want %lld)\n", what.c_str(), got,
                    want);
      }
    }
  }
  void eq_mv(mc::Mv got, mc::Mv want, const std::string& what) {
    ++checks;
    if (!(got == want)) {
      ++failures;
      if (failures <= 30) {
        std::printf("FAIL: %s (got (%d,%d), want (%d,%d))\n", what.c_str(),
                    got.y, got.x, want.y, want.x);
      }
    }
  }
};

std::string label(const std::string& base, int a, int b) {
  return base + " [" + std::to_string(a) + "," + std::to_string(b) + "]";
}

struct Grid {
  int cols = 0;
  int rows = 0;
  std::vector<mc::MiInfo> mi;
  std::vector<std::uint8_t> y_mode;

  void init(int c, int r) {
    cols = c;
    rows = r;
    mi.assign(static_cast<std::size_t>(c) * static_cast<std::size_t>(r),
              mc::MiInfo{});
    y_mode.assign(mi.size(), 0);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        mc::MiInfo& m = at(row, col);
        m.ref[0] = merge::kRefNone;
        m.ref[1] = merge::kRefNone;
        m.mv[0][0] = 0;
        m.mv[0][1] = 0;
        m.mv[1][0] = 0;
        m.mv[1][1] = 0;
        m.is_inter = 0;
        m.w4 = 1;
        m.h4 = 1;
        m.col_base = static_cast<std::int16_t>(col);
        m.row_base = static_cast<std::int16_t>(row);
      }
    }
  }

  std::size_t index(int row, int col) const {
    return static_cast<std::size_t>(row) *
               static_cast<std::size_t>(cols) +
           static_cast<std::size_t>(col);
  }

  mc::MiInfo& at(int row, int col) { return mi[index(row, col)]; }
  const mc::MiInfo& at(int row, int col) const { return mi[index(row, col)]; }

  void set(int row, int col, std::int8_t ref0, std::int8_t ref1, mc::Mv mv0,
           mc::Mv mv1, bool inter = true, std::uint8_t mode = 0) {
    mc::MiInfo& m = at(row, col);
    m.ref[0] = ref0;
    m.ref[1] = ref1;
    m.mv[0][0] = mv0.y;
    m.mv[0][1] = mv0.x;
    m.mv[1][0] = mv1.y;
    m.mv[1][1] = mv1.x;
    m.is_inter = inter ? 1 : 0;
    m.w4 = 1;
    m.h4 = 1;
    m.col_base = static_cast<std::int16_t>(col);
    m.row_base = static_cast<std::int16_t>(row);
    y_mode[index(row, col)] = mode;
  }

  void fill(std::int8_t ref0, mc::Mv mv0, std::uint8_t mode = 0) {
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        set(row, col, ref0, merge::kRefNone, mv0, mv0, true, mode);
      }
    }
  }
};

merge::StackParams make_params(const Grid& grid, int mi_row, int mi_col,
                               int bw4, int bh4, std::int8_t ref0,
                               std::int8_t ref1 = merge::kRefNone) {
  merge::StackParams p;
  p.mi_row = mi_row;
  p.mi_col = mi_col;
  p.bw4 = bw4;
  p.bh4 = bh4;
  p.sb_h4 = 16;
  p.mi_cols = grid.cols;
  p.mi_rows = grid.rows;
  p.mi = grid.mi.data();
  p.ref_frame[0] = ref0;
  p.ref_frame[1] = ref1;
  p.is_compound = ref1 != merge::kRefNone;
  p.global_mv[0] = mc::Mv{0, 0};
  p.global_mv[1] = mc::Mv{0, 0};
  p.reorder = merge::DrlReorder::kDisabled;
  return p;
}

std::vector<BlockSize> all_block_sizes() {
  std::vector<BlockSize> out;
  for (int i = 0; i < partition::kBlockSizeCount; ++i) {
    const auto size = static_cast<BlockSize>(i);
    if (!partition::is_valid_block_size(size)) {
      continue;
    }
    if (partition::block_width(size) > 64 ||
        partition::block_height(size) > 64) {
      continue;
    }
    out.push_back(size);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Skip-mode frame params
// ---------------------------------------------------------------------------

void test_skip_mode_params(Test& t) {
  merge::SkipModeFrameParams p;
  merge::SkipModeFrameResult r;

  p = {};
  p.frame_is_intra = true;
  p.num_total_refs = 2;
  p.refs[0] = {8, false, true};
  p.refs[1] = {9, false, true};
  t.eq_int(static_cast<int>(merge::derive_skip_mode_params(p, &r)),
           static_cast<int>(merge::Status::kOk), "skip params intra call");
  t.expect(!r.allowed, "skip_mode not allowed on an intra frame");
  t.eq_int(r.frame[0], 0, "intra frame fallback ref0");
  t.eq_int(r.frame[1], 0, "intra frame fallback ref1");

  p = {};
  p.switch_frame = true;
  p.num_total_refs = 2;
  p.refs[0] = {8, false, true};
  p.refs[1] = {9, false, true};
  t.eq_int(static_cast<int>(merge::derive_skip_mode_params(p, &r)),
           static_cast<int>(merge::Status::kOk), "skip params switch call");
  t.expect(!r.allowed, "skip_mode not allowed on a switch frame");

  for (int n = 1; n <= 4; ++n) {
    p = {};
    p.num_total_refs = n;
    p.order_hint = 10;
    // |10-8| = 2 and |10-9| = 1, so the distance difference is 1.
    p.refs[0] = {8, false, true};
    p.refs[1] = {9, false, true};
    p.refs[2] = {6, false, true};
    p.refs[3] = {7, false, true};
    const merge::Status status = merge::derive_skip_mode_params(p, &r);
    t.eq_int(static_cast<int>(status), static_cast<int>(merge::Status::kOk),
             label("skip params refs", n, 0));
    t.expect(r.allowed, label("skip allowed refs", n, 0));
    t.eq_int(r.num_total_refs, n, label("skip num refs", n, 0));
    t.eq_int(r.frame[0], 0, label("skip frame0 refs", n, 0));
    t.eq_int(r.frame[1], n > 1 ? 1 : 0,
             label("skip frame1 refs", n, 0));
    t.eq_int(static_cast<int>(merge::validate_skip_mode_present(r, true)),
             static_cast<int>(merge::Status::kOk),
             label("skip present valid refs", n, 0));
  }

  p = {};
  p.num_total_refs = 2;
  p.order_hint = 10;
  p.refs[0] = {8, false, true};
  p.refs[1] = {4, false, true};  // distance 6, difference 4 > 1
  merge::derive_skip_mode_params(p, &r);
  t.eq_int(r.frame[1], 0, "skip second frame falls back when distances differ");

  p = {};
  p.num_total_refs = 2;
  p.order_hint = 10;
  p.refs[0] = {8, true, true};  // RESTRICTED_OH -> curToRef0 = 0
  p.refs[1] = {8, false, true};  // distance 2
  merge::derive_skip_mode_params(p, &r);
  t.eq_int(r.frame[1], 0, "restricted first ref forces second frame to 0");

  p = {};
  p.num_total_refs = 2;
  p.order_hint = 10;
  p.refs[0] = {8, true, true};
  p.refs[1] = {4, true, true};
  merge::derive_skip_mode_params(p, &r);
  t.eq_int(r.frame[1], 1, "two restricted refs have equal distance");

  p = {};
  p.num_total_refs = 0;
  t.eq_int(static_cast<int>(merge::derive_skip_mode_params(p, &r)),
           static_cast<int>(merge::Status::kInvalidRefCount),
           "zero refs rejected");
  p.num_total_refs = 5;
  t.eq_int(static_cast<int>(merge::derive_skip_mode_params(p, &r)),
           static_cast<int>(merge::Status::kInvalidRefCount),
           "five refs rejected");

  merge::SkipModeFrameResult not_allowed;
  t.eq_int(
      static_cast<int>(merge::validate_skip_mode_present(not_allowed, true)),
      static_cast<int>(merge::Status::kInvalidSkipMode),
      "skip_mode_present rejected when not allowed");
  t.eq_int(
      static_cast<int>(merge::validate_skip_mode_present(not_allowed, false)),
      static_cast<int>(merge::Status::kOk),
      "skip_mode_present = 0 always valid");
}

void test_skip_mode_frames(Test& t) {
  merge::SkipModeFrameParams params;
  params.num_total_refs = 4;
  params.order_hint = 10;
  params.refs[0] = {8, false, true};
  params.refs[1] = {9, false, true};
  merge::SkipModeFrameResult result;
  merge::derive_skip_mode_params(params, &result);

  merge::SkipModeFrameInputs in;
  std::int8_t out[2] = {9, 9};
  t.eq_int(static_cast<int>(merge::derive_skip_mode_frames(result, in, out)),
           static_cast<int>(merge::Status::kOk), "skip frames fallback call");
  t.eq_int(out[0], 0, "skip frames fallback ref0");
  t.eq_int(out[1], 1, "skip frames fallback ref1");

  in.neighbours.count = 1;
  in.neighbours.ref[0][0] = 2;
  in.neighbours.ref[0][1] = 3;
  merge::derive_skip_mode_frames(result, in, out);
  t.eq_int(out[0], 2, "neighbour compound override ref0");
  t.eq_int(out[1], 3, "neighbour compound override ref1");

  in.neighbours.ref[0][0] = 2;
  in.neighbours.ref[0][1] = merge::kRefNone;
  merge::derive_skip_mode_frames(result, in, out);
  t.eq_int(out[0], 0, "single neighbour does not override ref0");
  t.eq_int(out[1], 1, "single neighbour does not override ref1");

  in.neighbours.ref[0][0] = merge::kTipFrame;
  in.closest_past = 1;
  in.closest_future = 3;
  merge::derive_skip_mode_frames(result, in, out);
  t.eq_int(out[0], 1, "TIP neighbour uses ClosestPast");
  t.eq_int(out[1], 3, "TIP neighbour uses ClosestFuture");

  in.neighbours.ref[0][0] = 5;
  in.neighbours.ref[0][1] = 6;
  t.eq_int(static_cast<int>(merge::derive_skip_mode_frames(result, in, out)),
           static_cast<int>(merge::Status::kInvalidRefCount),
           "neighbour refs outside NumTotalRefs rejected");

  merge::SkipModeFrameResult disallowed;
  t.eq_int(
      static_cast<int>(merge::derive_skip_mode_frames(disallowed, in, out)),
      static_cast<int>(merge::Status::kInvalidSkipMode),
      "skip mode frames rejected when skip mode is not allowed");
}

void test_skip_mode_block_gate(Test& t) {
  for (BlockSize size : all_block_sizes()) {
    const bool comp = merge::is_comp_ref_allowed(size);
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    const bool thin = (w == 4 && h >= 16) || (h == 4 && w >= 16);
    t.eq_int(merge::is_thin_4xn_nx4_block(size) ? 1 : 0, thin ? 1 : 0,
             label("thin block", w, h));
    t.eq_int(comp ? 1 : 0, (std::min(w, h) >= 8 || thin) ? 1 : 0,
             label("is_comp_ref_allowed", w, h));

    merge::BlockSkipModeInputs in;
    in.size = size;
    bool allowed = false;
    t.eq_int(static_cast<int>(merge::skip_mode_syntax_allowed(in, &allowed)),
             static_cast<int>(merge::Status::kOk),
             label("skip block call", w, h));
    t.expect(!allowed, label("skip_block off when not present", w, h));

    in.skip_mode_present = true;
    merge::skip_mode_syntax_allowed(in, &allowed);
    t.eq_int(allowed ? 1 : 0, comp ? 1 : 0,
             label("skip_block follows is_comp_ref_allowed", w, h));

    in.seg_lvl_skip = true;
    merge::skip_mode_syntax_allowed(in, &allowed);
    t.expect(!allowed, label("SEG_LVL_SKIP disables skip_mode", w, h));

    in.seg_lvl_skip = false;
    in.seg_lvl_globalmv = true;
    merge::skip_mode_syntax_allowed(in, &allowed);
    t.expect(!allowed, label("SEG_LVL_GLOBALMV disables skip_mode", w, h));

    in.seg_lvl_globalmv = false;
    in.region_intra = true;
    merge::skip_mode_syntax_allowed(in, &allowed);
    t.expect(!allowed, label("intra region disables skip_mode", w, h));
  }

  merge::BlockSkipModeInputs bad;
  bad.size = BlockSize::kInvalid;
  bool allowed = false;
  t.eq_int(static_cast<int>(merge::skip_mode_syntax_allowed(bad, &allowed)),
           static_cast<int>(merge::Status::kInvalidBlockSize),
           "invalid block size rejected");
}

// ---------------------------------------------------------------------------
// DRL
// ---------------------------------------------------------------------------

void test_drl(Test& t) {
  for (int m = 1; m <= merge::kMaxDrlIndex; ++m) {
    for (int start = 0; start <= m + 1; ++start) {
      const int first = start <= m ? start : m;
      for (int index = first; index <= m; ++index) {
        const std::vector<std::uint8_t> bits =
            merge::encode_drl_index(index, start, m);
        const std::string tag =
            label("drl encode", index, start * 10 + m);
        if (index < m) {
          const std::size_t expected_ones =
              static_cast<std::size_t>(index - start);
          t.eq_int(static_cast<long long>(bits.size()),
                   static_cast<long long>(expected_ones + 1),
                   tag + " length");
          for (std::size_t i = 0; i < expected_ones; ++i) {
            t.eq_int(bits[i], 1, tag + " prefix one");
          }
          t.eq_int(bits[expected_ones], 0, tag + " terminator");
        } else {
          const std::size_t expected_ones =
              start <= m ? static_cast<std::size_t>(m - start) : 0u;
          t.eq_int(static_cast<long long>(bits.size()),
                   static_cast<long long>(expected_ones), tag + " exhaust len");
          for (std::uint8_t bit : bits) {
            t.eq_int(bit, 1, tag + " exhaust one");
          }
        }

        int decoded = -1;
        std::size_t consumed = 0;
        const merge::Status st = merge::decode_drl_index(
            std::span<const std::uint8_t>(bits), start, m, &decoded, &consumed);
        t.eq_int(static_cast<int>(st), static_cast<int>(merge::Status::kOk),
                 tag + " decode status");
        t.eq_int(decoded, index, tag + " decoded index");
        t.eq_int(static_cast<long long>(consumed),
                 static_cast<long long>(bits.size()), tag + " consumed");
      }
    }
  }

  t.expect(merge::encode_drl_index(1, 2, 3).empty(),
           "index below start rejected");
  t.expect(merge::encode_drl_index(6, 0, 5).empty(),
           "index above m rejected");
  t.expect(merge::encode_drl_index(0, 0, 0).empty(), "m = 0 rejected");
  t.expect(merge::encode_drl_index(0, 0, 6).empty(), "m > 5 rejected");
  t.expect(merge::encode_drl_index(0, -1, 3).empty(), "negative start rejected");
  t.expect(merge::encode_drl_index(3, 5, 3).empty(),
           "start beyond m + 1 rejected");

  const std::vector<std::uint8_t> truncated = {1, 1};
  int index = -1;
  std::size_t consumed = 0;
  t.eq_int(static_cast<int>(merge::decode_drl_index(truncated, 0, 3, &index,
                                                    &consumed)),
           static_cast<int>(merge::Status::kTruncatedDrlBits),
           "truncated DRL bits rejected");
  const std::vector<std::uint8_t> empty;
  t.eq_int(static_cast<int>(merge::decode_drl_index(empty, 0, 1, &index,
                                                    &consumed)),
           static_cast<int>(merge::Status::kTruncatedDrlBits),
           "empty DRL bits rejected");
  t.eq_int(static_cast<int>(
               merge::decode_drl_index(empty, 3, 2, &index, &consumed)),
           static_cast<int>(merge::Status::kOk),
           "start > m consumes no bits");
  t.eq_int(index, 2, "start > m returns m");

  t.eq_int(merge::drl_cdf_row(0), 0, "drl cdf row 0");
  t.eq_int(merge::drl_cdf_row(1), 1, "drl cdf row 1");
  t.eq_int(merge::drl_cdf_row(2), 2, "drl cdf row 2");
  t.eq_int(merge::drl_cdf_row(5), 2, "drl cdf row clamps at 2");
  t.eq_int(merge::drl_cdf_row(-1), 0, "drl cdf negative row");

  t.expect(merge::has_second_drl(merge::kYModeNearNearMv, false, false),
           "NEAR_NEARMV has a second DRL");
  t.expect(!merge::has_second_drl(merge::kYModeNearNearMv, true, false),
           "skip_mode has a single DRL");
  t.expect(!merge::has_second_drl(merge::kYModeNearNearMv, false, true),
           "use_optflow has a single DRL");
  t.expect(merge::has_second_drl(merge::kYModeNearNewMv, false, false),
           "NEAR_NEWMV has a second DRL");
  t.expect(!merge::has_second_drl(merge::kYModeNewNearMv, false, false),
           "NEW_NEARMV has a single DRL");

  int start = -1;
  t.eq_int(static_cast<int>(
               merge::derive_second_drl_start(merge::kYModeNearNearMv, true, 2,
                                              &start)),
           static_cast<int>(merge::Status::kOk), "second drl start call");
  t.eq_int(start, 3, "same-ref NEAR_NEARMV starts after RefMvIdx0");
  merge::derive_second_drl_start(merge::kYModeNearNearMv, false, 2, &start);
  t.eq_int(start, 0, "different refs start at 0");
  merge::derive_second_drl_start(merge::kYModeNearNewMv, true, 2, &start);
  t.eq_int(start, 0, "NEAR_NEWMV starts at 0");
  merge::derive_second_drl_start(merge::kYModeNearNearMv, true, 5, &start);
  t.eq_int(start, 6, "exhausted RefMvIdx0 yields start m + 1");
}

// ---------------------------------------------------------------------------
// single_mode and mode resolution
// ---------------------------------------------------------------------------

merge::MvStack make_stack(std::initializer_list<mc::Mv> mvs) {
  merge::MvStack stack;
  for (mc::Mv mv : mvs) {
    merge::StackCandidate& c =
        stack.cand[static_cast<std::size_t>(stack.count)];
    c.mv[0] = mv;
    c.mv[1] = mv;
    c.weight = 1;
    ++stack.count;
  }
  stack.num_nearest = stack.count;
  return stack;
}

void test_single_mode_and_resolution(Test& t) {
  merge::SingleMode mode = merge::SingleMode::kNearMv;
  for (int v = 0; v <= 2; ++v) {
    t.eq_int(static_cast<int>(merge::decode_single_mode(v, &mode)),
             static_cast<int>(merge::Status::kOk), label("single_mode", v, 0));
    t.eq_int(static_cast<int>(mode), v, label("single_mode value", v, 0));
    t.eq_int(merge::single_mode_y_mode(v), 14 + v,
             label("single_mode YMode", v, 0));
  }
  t.eq_int(static_cast<int>(merge::decode_single_mode(3, &mode)),
           static_cast<int>(merge::Status::kInvalidArgument),
           "single_mode 3 rejected");
  t.eq_int(static_cast<int>(merge::decode_single_mode(-1, &mode)),
           static_cast<int>(merge::Status::kInvalidArgument),
           "single_mode -1 rejected");
  t.expect(merge::single_mode_has_drl(merge::SingleMode::kNearMv),
           "NEARMV codes a DRL index");
  t.expect(!merge::single_mode_has_drl(merge::SingleMode::kGlobalMv),
           "GLOBALMV has no DRL index");
  t.expect(merge::single_mode_has_drl(merge::SingleMode::kNewMv),
           "NEWMV codes a DRL index");
  t.expect(merge::single_mode_has_mvd(merge::SingleMode::kNewMv),
           "NEWMV codes an MVD");
  t.expect(!merge::single_mode_has_mvd(merge::SingleMode::kNearMv),
           "NEARMV has no MVD");

  Grid grid;
  grid.init(32, 32);
  merge::StackParams params =
      make_params(grid, 8, 8, 2, 2, 1, merge::kRefNone);
  params.global_mv[0] = mc::Mv{4, -4};
  const merge::MvStack stack =
      make_stack({mc::Mv{10, 20}, mc::Mv{30, 40}, mc::Mv{-5, 6}});

  merge::ResolvedMode resolved;
  t.eq_int(static_cast<int>(merge::resolve_single_mode(
               stack, params, merge::SingleMode::kNearMv, 1, mc::Mv{},
               &resolved)),
           static_cast<int>(merge::Status::kOk), "resolve NEARMV");
  t.eq_mv(resolved.mv[0], (mc::Mv{30, 40}), "NEARMV uses stack[drl]");
  t.eq_mv(resolved.pred_mv[0], (mc::Mv{30, 40}),
          "NEARMV predictor is stack[drl]");
  t.eq_mv(resolved.mvd[0], (mc::Mv{0, 0}), "NEARMV has no MVD");
  t.eq_int(resolved.y_mode, merge::kYModeNearMv, "NEARMV YMode");

  t.eq_int(static_cast<int>(merge::resolve_single_mode(
               stack, params, merge::SingleMode::kGlobalMv, 0, mc::Mv{},
               &resolved)),
           static_cast<int>(merge::Status::kOk), "resolve GLOBALMV");
  t.eq_mv(resolved.mv[0], (mc::Mv{4, -4}), "GLOBALMV uses GlobalMvs[0]");
  t.eq_int(resolved.y_mode, merge::kYModeGlobalMv, "GLOBALMV YMode");

  t.eq_int(static_cast<int>(merge::resolve_single_mode(
               stack, params, merge::SingleMode::kNewMv, 2, mc::Mv{7, -3},
               &resolved)),
           static_cast<int>(merge::Status::kOk), "resolve NEWMV");
  t.eq_mv(resolved.mv[0], (mc::Mv{-5 + 7, 6 - 3}),
          "NEWMV adds the MVD to the predictor");
  t.eq_mv(resolved.mvd[0], (mc::Mv{7, -3}), "NEWMV returns the MVD");
  t.eq_int(resolved.y_mode, merge::kYModeNewMv, "NEWMV YMode");

  t.eq_int(static_cast<int>(merge::resolve_single_mode(
               stack, params, merge::SingleMode::kNearMv, 3, mc::Mv{},
               &resolved)),
           static_cast<int>(merge::Status::kInvalidDrlIndex),
           "DRL index equal to the stack count is invalid");
  t.eq_int(static_cast<int>(merge::resolve_single_mode(
               stack, params, merge::SingleMode::kNearMv, -1, mc::Mv{},
               &resolved)),
           static_cast<int>(merge::Status::kInvalidDrlIndex),
           "negative DRL index is invalid");
  t.eq_int(static_cast<int>(merge::resolve_single_mode(
               stack, params, merge::SingleMode::kNearMv, 0, mc::Mv{1, 1},
               &resolved)),
           static_cast<int>(merge::Status::kInvalidArgument),
           "NEARMV with an MVD is invalid");
  t.eq_int(static_cast<int>(merge::resolve_single_mode(
               stack, params, merge::SingleMode::kGlobalMv, 0, mc::Mv{1, 1},
               &resolved)),
           static_cast<int>(merge::Status::kInvalidArgument),
           "GLOBALMV with an MVD is invalid");

  // skip_flag is independent of the merge/new distinction.
  merge::ModeSyntax syntax;
  syntax.skip_flag = true;
  syntax.single_mode = merge::SingleMode::kNewMv;
  syntax.y_mode = merge::kYModeNewMv;
  syntax.drl_index = 0;
  syntax.mvd = mc::Mv{2, 2};
  syntax.ref_frame[0] = 1;
  syntax.ref_frame[1] = merge::kRefNone;
  t.eq_int(static_cast<int>(merge::resolve_mode(stack, params, syntax,
                                                &resolved)),
           static_cast<int>(merge::Status::kOk),
           "skip_flag = 1 with NEWMV is valid");
  t.expect(resolved.skip_flag, "skip_flag is preserved by resolution");

  syntax.single_mode = merge::SingleMode::kNearMv;
  syntax.y_mode = merge::kYModeNearMv;
  syntax.mvd = mc::Mv{};
  t.eq_int(static_cast<int>(merge::resolve_mode(stack, params, syntax,
                                                &resolved)),
           static_cast<int>(merge::Status::kOk),
           "skip_flag = 1 with NEARMV is valid");
  t.expect(resolved.skip_flag, "merge keeps skip_flag");

  syntax.y_mode = merge::kYModeGlobalMv;
  t.eq_int(static_cast<int>(merge::resolve_mode(stack, params, syntax,
                                                &resolved)),
           static_cast<int>(merge::Status::kInvalidArgument),
           "mismatched YMode/single_mode rejected");

  syntax.single_mode = merge::SingleMode::kNewMv;
  syntax.y_mode = merge::kYModeNewMv;
  syntax.ref_frame[0] = 2;
  t.eq_int(static_cast<int>(merge::resolve_mode(stack, params, syntax,
                                                &resolved)),
           static_cast<int>(merge::Status::kInvalidArgument),
           "reference mismatch rejected");

  // DRL selection: first exact match, start/m bound respected.
  int index = -1;
  bool found = false;
  const merge::MvStack dup = make_stack(
      {mc::Mv{1, 1}, mc::Mv{2, 2}, mc::Mv{1, 1}, mc::Mv{3, 3}});
  merge::choose_drl_index_for_mv(dup, mc::Mv{1, 1}, 0, 5, &index, &found);
  t.expect(found, "choose DRL found");
  t.eq_int(index, 0, "choose DRL picks the first match");
  merge::choose_drl_index_for_mv(dup, mc::Mv{1, 1}, 1, 5, &index, &found);
  t.eq_int(index, 2, "choose DRL honours start");
  merge::choose_drl_index_for_mv(dup, mc::Mv{1, 1}, 1, 1, &index, &found);
  t.expect(!found, "choose DRL honours m");
  merge::choose_drl_index_for_mv(dup, mc::Mv{9, 9}, 0, 5, &index, &found);
  t.expect(!found, "choose DRL reports a miss");
}

void test_skip_mode_resolution(Test& t) {
  Grid grid;
  grid.init(16, 16);
  const merge::StackParams params =
      make_params(grid, 8, 8, 2, 2, 2, 3);

  merge::MvStack stack;
  stack.cand[0].mv[0] = mc::Mv{10, 20};
  stack.cand[0].mv[1] = mc::Mv{30, 40};
  stack.cand[0].weight = 1;
  stack.cand[0].cwp = 8;
  stack.count = 1;

  merge::ModeSyntax syntax;
  syntax.skip_mode = true;
  syntax.y_mode = merge::kYModeNearNearMv;
  syntax.ref_frame[0] = 2;
  syntax.ref_frame[1] = 3;
  syntax.drl_index = 0;

  merge::ResolvedMode resolved;
  t.eq_int(static_cast<int>(merge::resolve_mode(stack, params, syntax,
                                                &resolved)),
           static_cast<int>(merge::Status::kOk), "skip_mode resolves");
  t.expect(resolved.compound, "skip_mode is compound");
  t.expect(resolved.skip_mode, "skip_mode is marked");
  t.expect(!resolved.skip_flag, "skip_mode alone does not set skip_flag");
  t.eq_mv(resolved.mv[0], (mc::Mv{10, 20}), "skip_mode list 0 MV");
  t.eq_mv(resolved.mv[1], (mc::Mv{30, 40}), "skip_mode list 1 MV");
  t.eq_mv(resolved.mvd[0], (mc::Mv{0, 0}), "skip_mode list 0 has no MVD");
  t.eq_mv(resolved.mvd[1], (mc::Mv{0, 0}), "skip_mode list 1 has no MVD");
  t.eq_int(resolved.cwp, 8, "skip_mode CWP from the stack");
  t.eq_int(resolved.y_mode, merge::kYModeNearNearMv,
           "skip_mode YMode is NEAR_NEARMV");

  syntax.skip_flag = true;
  merge::resolve_mode(stack, params, syntax, &resolved);
  t.expect(resolved.skip_flag, "skip_flag = 1 is preserved with skip_mode");
  syntax.skip_flag = false;
  merge::resolve_mode(stack, params, syntax, &resolved);
  t.expect(!resolved.skip_flag,
           "skip_mode with skip_flag = 0 is still valid syntax");

  syntax.mvd = mc::Mv{1, 0};
  t.eq_int(static_cast<int>(merge::resolve_mode(stack, params, syntax,
                                                &resolved)),
           static_cast<int>(merge::Status::kInvalidSkipMode),
           "skip_mode with an MVD rejected");
  syntax.mvd = mc::Mv{};

  syntax.y_mode = merge::kYModeGlobalMv;
  t.eq_int(static_cast<int>(merge::resolve_mode(stack, params, syntax,
                                                &resolved)),
           static_cast<int>(merge::Status::kInvalidSkipMode),
           "skip_mode must imply NEAR_NEARMV");
  syntax.y_mode = merge::kYModeNearNearMv;

  syntax.ref_frame[1] = 4;
  t.eq_int(static_cast<int>(merge::resolve_mode(stack, params, syntax,
                                                &resolved)),
           static_cast<int>(merge::Status::kInvalidSkipMode),
           "skip_mode reference pair mismatch rejected");
  syntax.ref_frame[1] = 3;

  syntax.drl_index = 1;
  t.eq_int(static_cast<int>(merge::resolve_mode(stack, params, syntax,
                                                &resolved)),
           static_cast<int>(merge::Status::kInvalidDrlIndex),
           "skip_mode DRL outside the stack rejected");
  syntax.drl_index = 6;
  t.eq_int(static_cast<int>(merge::resolve_mode(stack, params, syntax,
                                                &resolved)),
           static_cast<int>(merge::Status::kInvalidDrlIndex),
           "skip_mode DRL above m rejected");
  syntax.drl_index = 0;

  merge::ModeSyntax single_params_syntax = syntax;
  const merge::StackParams single_params =
      make_params(grid, 8, 8, 2, 2, 1, merge::kRefNone);
  t.eq_int(static_cast<int>(merge::resolve_mode(stack, single_params,
                                                single_params_syntax,
                                                &resolved)),
           static_cast<int>(merge::Status::kInvalidSkipMode),
           "skip_mode on a single-reference stack rejected");

  merge::ModeSyntax compound_syntax;
  compound_syntax.y_mode = merge::kYModeNearNearMv;
  compound_syntax.ref_frame[0] = 2;
  compound_syntax.ref_frame[1] = 3;
  t.eq_int(static_cast<int>(merge::resolve_mode(stack, params, compound_syntax,
                                                &resolved)),
           static_cast<int>(merge::Status::kUnsupportedFeature),
           "non-skip compound is rejected as out of scope");

  // Convenience entry point: derive the reference pair and resolve.
  merge::SkipModeFrameParams fp;
  fp.num_total_refs = 4;
  fp.order_hint = 10;
  fp.refs[0] = {8, false, true};
  fp.refs[1] = {9, false, true};
  merge::SkipModeFrameResult fr;
  merge::derive_skip_mode_params(fp, &fr);
  merge::SkipModeFrameInputs fi;
  fi.neighbours.count = 1;
  fi.neighbours.ref[0][0] = 2;
  fi.neighbours.ref[0][1] = 3;
  t.eq_int(static_cast<int>(merge::resolve_skip_mode(
               stack, params, fr, fi, 0, false, &resolved)),
           static_cast<int>(merge::Status::kOk), "resolve_skip_mode call");
  t.expect(resolved.compound, "resolve_skip_mode is compound");
  t.eq_mv(resolved.mv[1], (mc::Mv{30, 40}),
          "resolve_skip_mode uses the neighbour reference pair");

  fi.neighbours.ref[0][0] = 0;
  fi.neighbours.ref[0][1] = 1;
  t.eq_int(static_cast<int>(merge::resolve_skip_mode(
               stack, params, fr, fi, 0, false, &resolved)),
           static_cast<int>(merge::Status::kInvalidSkipMode),
           "resolve_skip_mode requires the stack reference pair");
}

// ---------------------------------------------------------------------------
// Mode contexts
// ---------------------------------------------------------------------------

void test_mode_contexts(Test& t) {
  merge::NeighborModes n;
  t.eq_int(merge::skip_mode_context(n), 0, "skip_mode ctx none");
  t.eq_int(merge::skip_flag_context(n, false), 0, "skip_flag ctx none");
  n.above.available = true;
  n.above.skip_mode = true;
  n.above.skip_flag = true;
  n.left.available = true;
  n.left.skip_mode = false;
  n.left.skip_flag = true;
  t.eq_int(merge::skip_mode_context(n), 1, "skip_mode ctx above only");
  t.eq_int(merge::skip_flag_context(n, false), 2, "skip_flag ctx both");
  t.eq_int(merge::skip_flag_context(n, true), 5,
           "skip_flag ctx adds the skip_mode half");
  n.left.skip_mode = true;
  t.eq_int(merge::skip_mode_context(n), 2, "skip_mode ctx both");

  Grid grid;
  grid.init(16, 16);
  // 8x8 block at (4,4).  Scan points: leftA (5,3), aboveA (3,5),
  // leftB (4,3), aboveB (3,4).
  merge::NewMvContextParams p;
  p.mi_row = 4;
  p.mi_col = 4;
  p.bw4 = 2;
  p.bh4 = 2;
  p.mi_cols = grid.cols;
  p.mi_rows = grid.rows;
  p.mi = grid.mi.data();
  p.y_mode = grid.y_mode.data();
  p.ref_frame[0] = 1;
  p.ref_frame[1] = merge::kRefNone;

  int ctx = -1;
  t.eq_int(static_cast<int>(merge::derive_new_mv_context(p, &ctx)),
           static_cast<int>(merge::Status::kOk), "new mv ctx call");
  t.eq_int(ctx, 0, "new mv ctx no neighbours");

  grid.set(3, 5, 1, merge::kRefNone, mc::Mv{1, 1}, mc::Mv{1, 1}, true,
           merge::kYModeNewMv);
  merge::derive_new_mv_context(p, &ctx);
  t.eq_int(ctx, 3, "above NEWMV neighbour adds nearest + new count");

  grid.set(5, 3, 1, merge::kRefNone, mc::Mv{2, 2}, mc::Mv{2, 2}, true,
           merge::kYModeNearNewMv);
  merge::derive_new_mv_context(p, &ctx);
  t.eq_int(ctx, 4,
           "NEW_NEARMV uses list 0 and both sides are nearest");

  grid.set(5, 3, 2, merge::kRefNone, mc::Mv{2, 2}, mc::Mv{2, 2}, true,
           merge::kYModeNewMv);
  grid.set(3, 5, 2, merge::kRefNone, mc::Mv{1, 1}, mc::Mv{1, 1}, true,
           merge::kYModeNewMv);
  merge::derive_new_mv_context(p, &ctx);
  t.eq_int(ctx, 0, "non-matching reference contributes no context");

  grid.set(3, 5, 1, merge::kRefNone, mc::Mv{1, 1}, mc::Mv{1, 1}, true,
           merge::kYModeNearMv);
  merge::derive_new_mv_context(p, &ctx);
  t.eq_int(ctx, 1, "NEARMV neighbour contributes nearest but no NewMvCount");

  p.is_compound = true;
  p.ref_frame[1] = 2;
  grid.set(3, 5, 1, 2, mc::Mv{1, 1}, mc::Mv{2, 2}, true,
           merge::kYModeNewNewMv);
  grid.set(5, 3, 1, merge::kRefNone, mc::Mv{2, 2}, mc::Mv{2, 2}, true,
           merge::kYModeNewMv);
  merge::derive_new_mv_context(p, &ctx);
  t.eq_int(ctx, 3, "compound pair match uses the compound branch");

  p.mi = nullptr;
  t.eq_int(static_cast<int>(merge::derive_new_mv_context(p, &ctx)),
           static_cast<int>(merge::Status::kInvalidArgument),
           "new mv ctx rejects a null grid");
}

// ---------------------------------------------------------------------------
// MV stack
// ---------------------------------------------------------------------------

void test_stack_scan_and_sort(Test& t) {
  Grid grid;
  grid.init(16, 16);
  // Block 8x8 at MI (4,4), not at a superblock border.
  grid.fill(1, mc::Mv{-100, -100});
  grid.set(5, 3, 1, merge::kRefNone, mc::Mv{10, 10}, mc::Mv{10, 10});
  grid.set(3, 5, 1, merge::kRefNone, mc::Mv{20, 20}, mc::Mv{20, 20});
  grid.set(4, 3, 1, merge::kRefNone, mc::Mv{30, 30}, mc::Mv{30, 30});
  grid.set(3, 3, 1, merge::kRefNone, mc::Mv{40, 40}, mc::Mv{40, 40});
  grid.set(6, 3, 1, merge::kRefNone, mc::Mv{50, 50}, mc::Mv{50, 50});
  grid.set(5, 1, 1, merge::kRefNone, mc::Mv{70, 70}, mc::Mv{70, 70});
  grid.set(4, 1, 1, merge::kRefNone, mc::Mv{80, 80}, mc::Mv{80, 80});

  merge::StackParams params =
      make_params(grid, 4, 4, 2, 2, 1, merge::kRefNone);
  merge::MvStack stack;
  t.eq_int(static_cast<int>(merge::build_mv_stack(params, &stack)),
           static_cast<int>(merge::Status::kOk), "stack build");
  t.eq_int(stack.count, 6, "stack fills to MAX_REF_MV_STACK_SIZE");
  t.eq_int(stack.num_nearest, 6, "numNearest snapshot after step 13");
  const mc::Mv want[6] = {{10, 10}, {20, 20}, {30, 30},
                          {-100, -100}, {50, 50}, {40, 40}};
  const int weight[6] = {1, 1, 1, 2, 1, 0};
  for (int i = 0; i < 6; ++i) {
    t.eq_mv(stack.cand[i].mv[0], want[i], label("scan order mv", i, 0));
    t.eq_int(stack.cand[i].weight, weight[i], label("scan weight", i, 0));
  }
  t.eq_int(stack.cand[0].row_offset, 1, "A row offset");
  t.eq_int(stack.cand[0].col_offset, -1, "A col offset");
  t.eq_int(stack.cand[3].row_offset, -1, "base row offset");
  t.eq_int(stack.cand[3].col_offset, 0, "base col offset");
  t.eq_int(stack.cand[5].row_offset, -1, "D row offset");
  t.eq_int(stack.cand[5].col_offset, -1, "D col offset");

  params.reorder = merge::DrlReorder::kConstraint;
  merge::build_mv_stack(params, &stack);
  // Weight sort is the §7.12.2.19 single move-to-front: moving the weight-2
  // base candidate into slot 0 displaces A to slot 3; B and C keep their
  // relative order.  (A full selection sort would also reorder B/C here; see
  // src/codec/inter/drl_reorder/README.md for the pinned divergence.)
  const mc::Mv sorted[6] = {{-100, -100}, {20, 20}, {30, 30},
                            {10, 10},    {50, 50}, {40, 40}};
  for (int i = 0; i < 6; ++i) {
    t.eq_mv(stack.cand[i].mv[0], sorted[i], label("sort constraint", i, 0));
  }

  params.reorder = merge::DrlReorder::kAlways;
  merge::build_mv_stack(params, &stack);
  for (int i = 0; i < 6; ++i) {
    t.eq_mv(stack.cand[i].mv[0], sorted[i], label("sort always", i, 0));
  }

  // Constraint sorting only runs with numNearest >= 4.  This fixture has two
  // candidates, so the disabled and constraint orders are identical while
  // ALWAYS reorders.
  Grid small;
  small.init(16, 16);
  small.fill(1, mc::Mv{-100, -100});
  small.set(3, 3, 1, merge::kRefNone, mc::Mv{1, 1}, mc::Mv{1, 1});
  small.set(1, 2, 1, merge::kRefNone, mc::Mv{2, 2}, mc::Mv{2, 2});
  // 4x4 block at (2,2): leftA (2,1), aboveA (1,2).
  small.set(2, 1, 1, merge::kRefNone, mc::Mv{9, 9}, mc::Mv{9, 9});
  small.set(1, 2, 1, merge::kRefNone, mc::Mv{2, 2}, mc::Mv{2, 2});
  merge::StackParams sp =
      make_params(small, 2, 2, 1, 1, 1, merge::kRefNone);
  sp.reorder = merge::DrlReorder::kConstraint;
  merge::build_mv_stack(sp, &stack);
  t.expect(stack.num_nearest < 4, "small fixture has < 4 nearest");
  const mc::Mv first_constraint = stack.cand[0].mv[0];
  sp.reorder = merge::DrlReorder::kDisabled;
  merge::build_mv_stack(sp, &stack);
  t.eq_mv(stack.cand[0].mv[0], first_constraint,
          "CONSTRAINT does not sort below four nearest");
  sp.reorder = merge::DrlReorder::kAlways;
  merge::build_mv_stack(sp, &stack);
  t.eq_mv(stack.cand[0].mv[0], (mc::Mv{-100, -100}),
          "ALWAYS sorts the highest-weight candidate first");

  // Dedup: identical neighbour MVs produce one candidate with summed weight.
  Grid dup;
  dup.init(16, 16);
  dup.set(5, 3, 1, merge::kRefNone, mc::Mv{7, 7}, mc::Mv{7, 7});
  dup.set(3, 5, 1, merge::kRefNone, mc::Mv{7, 7}, mc::Mv{7, 7});
  dup.set(4, 3, 1, merge::kRefNone, mc::Mv{7, 7}, mc::Mv{7, 7});
  dup.set(3, 3, 1, merge::kRefNone, mc::Mv{7, 7}, mc::Mv{7, 7});
  dup.set(6, 3, 1, merge::kRefNone, mc::Mv{7, 7}, mc::Mv{7, 7});
  dup.set(3, 6, 1, merge::kRefNone, mc::Mv{7, 7}, mc::Mv{7, 7});
  merge::StackParams dp =
      make_params(dup, 4, 4, 2, 2, 1, merge::kRefNone);
  dp.global_mv[0] = mc::Mv{7, 7};
  merge::build_mv_stack(dp, &stack);
  t.eq_int(stack.count, 1, "identical MVs dedup to one candidate");
  t.eq_int(stack.cand[0].weight, 5,
           "dedup sums weights (D contributes zero)");
  dp.global_mv[0] = mc::Mv{0, 0};
  merge::build_mv_stack(dp, &stack);
  t.eq_int(stack.count, 2, "a distinct global MV is appended");
  t.eq_mv(stack.cand[1].mv[0], (mc::Mv{0, 0}), "global MV candidate");

  // Other reference frames are ignored.
  Grid other;
  other.init(16, 16);
  other.set(5, 3, 2, merge::kRefNone, mc::Mv{11, 11}, mc::Mv{11, 11});
  merge::StackParams op =
      make_params(other, 4, 4, 2, 2, 1, merge::kRefNone);
  merge::build_mv_stack(op, &stack);
  t.eq_int(stack.count, 1, "other-reference neighbour ignored");
  t.eq_mv(stack.cand[0].mv[0], (mc::Mv{0, 0}),
          "other-reference stack keeps only the global MV");

  // Determinism.
  merge::MvStack again;
  merge::build_mv_stack(params, &again);
  merge::MvStack again2;
  merge::build_mv_stack(params, &again2);
  t.eq_int(again.count, again2.count, "stack build is deterministic (count)");
  for (int i = 0; i < again.count; ++i) {
    t.eq_mv(again.cand[i].mv[0], again2.cand[i].mv[0],
            label("deterministic mv", i, 0));
    t.eq_int(again.cand[i].weight, again2.cand[i].weight,
             label("deterministic weight", i, 0));
  }
}

// Fixture with three distinct pre-combination candidates plus the global MV.
// The MVP combinations are guaranteed to add new mixed candidates.
void fill_gate_fixture(Grid& grid, int mi_row, int mi_col, int bw4, int bh4) {
  grid.init(64, 64);
  // Step 6, step 7 and step 10 all visit distinct cells for every 4..64 block
  // size, so no pruning budget is spent on duplicates.
  grid.set(mi_row + bh4 - 1, mi_col - 1, 1, merge::kRefNone,
           mc::Mv{100, 200}, mc::Mv{100, 200});
  grid.set(mi_row - 1, mi_col + bw4 - 1, 1, merge::kRefNone,
           mc::Mv{300, 400}, mc::Mv{300, 400});
  grid.set(mi_row + bh4, mi_col - 1, 1, merge::kRefNone, mc::Mv{500, 600},
           mc::Mv{500, 600});
}

void test_mvp_gate_all_block_sizes(Test& t) {
  for (BlockSize size : all_block_sizes()) {
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    const int bw4 = w / 4;
    const int bh4 = h / 4;
    const bool gate = w > 32 && h > 32;
    t.eq_int(merge::mvp_combinations_allowed(size) ? 1 : 0, gate ? 1 : 0,
             label("mvp gate predicate", w, h));

    Grid grid;
    fill_gate_fixture(grid, 8, 8, bw4, bh4);
    const merge::StackParams params =
        make_params(grid, 8, 8, bw4, bh4, 1, merge::kRefNone);
    merge::MvStack spec;
    const merge::Status st = merge::build_mv_stack(params, &spec);
    t.eq_int(static_cast<int>(st), static_cast<int>(merge::Status::kOk),
             label("gate fixture build", w, h));

    merge::McStackAudit audit;
    t.eq_int(static_cast<int>(merge::audit_mc_stack(params, &audit)),
             static_cast<int>(merge::Status::kOk),
             label("gate fixture audit", w, h));
    t.expect(audit.mc_supported && audit.mc_call_ok,
             label("mc audit supported", w, h));
    t.eq_int(audit.expected_count, spec.count,
             label("audit expected count", w, h));
    t.expect(audit.mc_count >= 0, label("audit mc count present", w, h));

    if (gate) {
      t.expect(audit.matches_spec,
               label("64x64 shared stack matches spec", w, h));
      t.expect(!audit.mvp_combinations_not_gated,
               label("no combo gate defect above 32", w, h));
    } else {
      t.expect(!audit.matches_spec,
               label("shared stack diverges at or below 32", w, h));
      t.expect(audit.mvp_combinations_not_gated,
               label("combo gate defect detected", w, h));
      t.expect(audit.mc_count > spec.count,
               label("shared stack has extra combination candidates", w, h));
      // The spec fixture has at most the three scanned MVs plus the global MV.
      t.expect(spec.count <= 4,
               label("spec stack stops before combinations", w, h));
    }
  }
}

void test_full_stack_audit_64x64(Test& t) {
  Grid grid;
  grid.init(64, 64);
  // 64x64 block at MI (16,16): superblock border, even mi_col, so the step-13
  // scan point is the same for the spec and the shared component.
  grid.set(31, 15, 1, merge::kRefNone, mc::Mv{10, 10}, mc::Mv{10, 10});
  grid.set(15, 30, 1, merge::kRefNone, mc::Mv{20, 20}, mc::Mv{20, 20});
  grid.set(16, 15, 1, merge::kRefNone, mc::Mv{30, 30}, mc::Mv{30, 30});
  grid.set(15, 16, 1, merge::kRefNone, mc::Mv{40, 40}, mc::Mv{40, 40});
  grid.set(32, 15, 1, merge::kRefNone, mc::Mv{50, 50}, mc::Mv{50, 50});
  grid.set(15, 32, 1, merge::kRefNone, mc::Mv{60, 60}, mc::Mv{60, 60});
  grid.set(15, 14, 1, merge::kRefNone, mc::Mv{70, 70}, mc::Mv{70, 70});
  grid.set(31, 13, 1, merge::kRefNone, mc::Mv{80, 80}, mc::Mv{80, 80});
  grid.set(16, 13, 1, merge::kRefNone, mc::Mv{90, 90}, mc::Mv{90, 90});

  for (merge::DrlReorder reorder :
       {merge::DrlReorder::kDisabled, merge::DrlReorder::kConstraint,
        merge::DrlReorder::kAlways}) {
    merge::StackParams params =
        make_params(grid, 16, 16, 16, 16, 1, merge::kRefNone);
    params.reorder = reorder;
    merge::MvStack stack;
    t.eq_int(static_cast<int>(merge::build_mv_stack(params, &stack)),
             static_cast<int>(merge::Status::kOk), "64x64 full stack build");
    merge::McStackAudit audit;
    t.eq_int(static_cast<int>(merge::audit_mc_stack(params, &audit)),
             static_cast<int>(merge::Status::kOk), "64x64 full stack audit");
    t.expect(audit.mc_supported && audit.mc_call_ok,
             "64x64 full stack audit supported");
    t.expect(audit.matches_spec,
             "64x64 full stack matches the shared component");
    t.expect(!audit.mvp_combinations_not_gated,
             "64x64 full stack uses the spec combination gate");
  }
}

void test_sb_border_scan_audit(Test& t) {
  Grid grid;
  grid.init(64, 64);
  grid.fill(1, mc::Mv{500, 600});
  // 64x64 block at MI (16, 9): odd mi_col on a superblock border.
  grid.set(15, 6, 1, merge::kRefNone, mc::Mv{111, 111}, mc::Mv{111, 111});
  grid.set(15, 8, 1, merge::kRefNone, mc::Mv{500, 600}, mc::Mv{500, 600});
  const merge::StackParams params =
      make_params(grid, 16, 9, 16, 16, 1, merge::kRefNone);

  merge::McStackAudit audit;
  t.eq_int(static_cast<int>(merge::audit_mc_stack(params, &audit)),
           static_cast<int>(merge::Status::kOk), "sb border audit call");
  t.expect(audit.mc_supported, "sb border audit supported");
  t.expect(!audit.matches_spec, "sb border stack differs from spec");
  t.expect(audit.above_left_sb_border_scan_diff,
           "step 13 (-1,-1) vs (-1,-1-isSbBorder) detected");
}

void test_compound_stack(Test& t) {
  Grid grid;
  grid.init(32, 32);
  grid.fill(0, mc::Mv{900, 900});
  // Block 8x8 at (8,8), references (2,3).
  grid.set(9, 7, 2, 3, mc::Mv{10, 20}, mc::Mv{30, 40});
  grid.set(7, 9, 2, merge::kRefNone, mc::Mv{50, 60}, mc::Mv{50, 60});
  grid.set(8, 7, 2, merge::kRefNone, mc::Mv{70, 80}, mc::Mv{70, 80});
  grid.set(7, 7, 2, merge::kRefNone, mc::Mv{90, 100}, mc::Mv{90, 100});
  grid.set(10, 7, 2, merge::kRefNone, mc::Mv{110, 120}, mc::Mv{110, 120});
  grid.set(7, 10, 2, merge::kRefNone, mc::Mv{130, 140}, mc::Mv{130, 140});
  // Give one single-ref neighbour reference 3 so Compound add derived can
  // combine it with the list-0 single MV captured above.
  grid.set(10, 7, 3, merge::kRefNone, mc::Mv{210, 220}, mc::Mv{210, 220});

  const merge::StackParams params =
      make_params(grid, 8, 8, 2, 2, 2, 3);
  merge::MvStack stack;
  t.eq_int(static_cast<int>(merge::build_mv_stack(params, &stack)),
           static_cast<int>(merge::Status::kOk), "compound stack build");
  t.expect(stack.count >= 2, "compound stack has candidates");

  bool found_exact = false;
  bool found_derived = false;
  for (int i = 0; i < stack.count; ++i) {
    if (stack.cand[i].mv[0] == (mc::Mv{10, 20}) &&
        stack.cand[i].mv[1] == (mc::Mv{30, 40})) {
      found_exact = true;
    }
    if (stack.cand[i].mv[0] == (mc::Mv{50, 60}) &&
        stack.cand[i].mv[1] == (mc::Mv{210, 220})) {
      found_derived = true;
    }
  }
  t.expect(found_exact, "compound exact-pair neighbour is inserted");
  t.expect(found_derived,
           "compound add derived mixes list-0 and list-1 single MVs");

  merge::McStackAudit audit;
  merge::audit_mc_stack(params, &audit);
  t.expect(!audit.mc_supported, "compound stack is not auditable by mc");

  // Same-reference compound is legal (skip_mode with NumTotalRefs = 1).
  const merge::StackParams same =
      make_params(grid, 8, 8, 2, 2, 2, 2);
  t.eq_int(static_cast<int>(merge::build_mv_stack(same, &stack)),
           static_cast<int>(merge::Status::kOk),
           "same-reference compound stack builds");

  // Every supported reference count yields a usable compound stack.
  for (int n = 1; n <= 4; ++n) {
    const std::int8_t ref0 = 0;
    const std::int8_t ref1 = static_cast<std::int8_t>(n - 1);
    const merge::StackParams p =
        make_params(grid, 8, 8, 2, 2, ref0, ref1);
    t.eq_int(static_cast<int>(merge::build_mv_stack(p, &stack)),
             static_cast<int>(merge::Status::kOk),
             label("compound stack refs", n, 0));
    t.expect(stack.count > 0, label("compound stack non-empty refs", n, 0));
  }
}

void test_derived_projection(Test& t) {
  Grid grid;
  grid.init(16, 16);
  // 4x4 block at (4,4).  One neighbour uses reference 1 with a large MV.
  grid.set(4, 3, 2, merge::kRefNone, mc::Mv{8, 16}, mc::Mv{8, 16});

  int distances[5] = {0, 2, 4, 0, 0};
  merge::StackParams params =
      make_params(grid, 4, 4, 1, 1, 1, merge::kRefNone);
  params.frame_distance = distances;
  params.num_refs = 3;

  merge::MvStack stack;
  t.eq_int(static_cast<int>(merge::build_mv_stack(params, &stack)),
           static_cast<int>(merge::Status::kOk), "derived projection build");
  t.expect(stack.count >= 2, "derived projection stack has a candidate");
  t.eq_mv(stack.cand[0].mv[0], (mc::Mv{4, 8}),
          "derived MV is projected onto the current reference");

  merge::McStackAudit audit;
  merge::audit_mc_stack(params, &audit);
  t.expect(!audit.matches_spec,
           "shared stack omits the derived candidate");
  t.expect(audit.derived_candidates_missing,
           "derived-candidate omission is reported");

  // max_drl_bits_minus_1 caps the derived fill: with m = 1 the cap is 3.
  params.max_drl_bits_minus_1 = 0;  // cap = 2
  Grid two;
  two.init(16, 16);
  two.set(4, 3, 2, merge::kRefNone, mc::Mv{8, 16}, mc::Mv{8, 16});
  two.set(3, 4, 2, merge::kRefNone, mc::Mv{-8, 4}, mc::Mv{-8, 4});
  two.set(5, 3, 2, merge::kRefNone, mc::Mv{20, -10}, mc::Mv{20, -10});
  params.mi = two.mi.data();
  params.mi_cols = two.cols;
  params.mi_rows = two.rows;
  merge::build_mv_stack(params, &stack);
  t.eq_int(stack.count, 3,
           "derived fill cap plus the global MV limits the stack");
  bool third_projected = false;
  for (int i = 0; i < stack.count; ++i) {
    if (stack.cand[i].mv[0] == (mc::Mv{10, -5})) {
      third_projected = true;
    }
  }
  t.expect(!third_projected, "derived fill cap drops the third candidate");
}

void test_invalid_stack_params(Test& t) {
  Grid grid;
  grid.init(16, 16);
  merge::StackParams p = make_params(grid, 8, 8, 2, 2, 1, merge::kRefNone);
  merge::MvStack stack;

  p.mi = nullptr;
  t.eq_int(static_cast<int>(merge::build_mv_stack(p, &stack)),
           static_cast<int>(merge::Status::kInvalidArgument),
           "null grid rejected");
  p.mi = grid.mi.data();

  p.bw4 = 0;
  t.eq_int(static_cast<int>(merge::build_mv_stack(p, &stack)),
           static_cast<int>(merge::Status::kInvalidBlockSize),
           "zero width rejected");
  p.bw4 = 17;
  t.eq_int(static_cast<int>(merge::build_mv_stack(p, &stack)),
           static_cast<int>(merge::Status::kInvalidBlockSize),
           "width > 64 rejected");
  p.bw4 = 2;

  p.mi_row = -1;
  t.eq_int(static_cast<int>(merge::build_mv_stack(p, &stack)),
           static_cast<int>(merge::Status::kInvalidArgument),
           "negative block position rejected");
  p.mi_row = 8;

  p.use_intrabc = true;
  t.eq_int(static_cast<int>(merge::build_mv_stack(p, &stack)),
           static_cast<int>(merge::Status::kUnsupportedFeature),
           "intrabc rejected");
  p.use_intrabc = false;

  p.is_compound = true;
  p.ref_frame[1] = merge::kRefNone;
  t.eq_int(static_cast<int>(merge::build_mv_stack(p, &stack)),
           static_cast<int>(merge::Status::kInvalidArgument),
           "compound stack without a second reference rejected");
  p.is_compound = false;

  p.ref_frame[0] = merge::kRefNone;
  t.eq_int(static_cast<int>(merge::build_mv_stack(p, &stack)),
           static_cast<int>(merge::Status::kInvalidArgument),
           "none reference rejected");

  p.ref_frame[0] = 1;
  p.max_drl_bits_minus_1 = 5;
  t.eq_int(static_cast<int>(merge::build_mv_stack(p, &stack)),
           static_cast<int>(merge::Status::kInvalidArgument),
           "max_drl_bits_minus_1 > 4 rejected");
}

}  // namespace

void test_tip_ref_stack(Test& t) {
  Grid grid;
  grid.init(16, 16);
  // Every neighbour is a plain single-reference inter block on ref 0.
  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 8; ++col) {
      grid.set(row, col, 0, merge::kRefNone, mc::Mv{16, 16}, mc::Mv{16, 16});
    }
  }
  merge::StackParams p = make_params(grid, 4, 4, 2, 2, merge::kTipFrame, merge::kRefNone);
  merge::MvStack stack;
  // A TIP block's ordinary single-reference neighbours contribute no
  // candidate; §7.12.2 step 21 always adds the zero GlobalMvs[0].
  t.eq_int(static_cast<int>(merge::build_stack_checked(p, &stack)),
           static_cast<int>(merge::Status::kOk), "TIP-ref stack builds");
  t.eq_int(stack.count, 1, "TIP-ref stack holds only the GlobalMvs candidate");
  if (stack.count == 1) {
    t.eq_mv(stack.cand[0].mv[0], (mc::Mv{0, 0}), "TIP-ref GlobalMvs[0] is zero");
    t.eq_mv(stack.cand[0].mv[1], (mc::Mv{0, 0}), "TIP single list 1 matches");
  }

  // A compound neighbour with the frame's TIP pair needs §7.12.2.18, which
  // this builder does not synthesize: report it instead of a short stack.
  Grid compound_grid;
  compound_grid.init(16, 16);
  compound_grid.set(3, 3, 0, 1, mc::Mv{24, 8}, mc::Mv{-24, -8});
  merge::StackParams cp = make_params(compound_grid, 4, 4, 2, 2, merge::kTipFrame, merge::kRefNone);
  cp.closest_past = 0;
  cp.closest_future = 1;
  t.eq_int(static_cast<int>(merge::build_stack_checked(cp, &stack)),
           static_cast<int>(merge::Status::kUnsupportedFeature),
           "TIP-ref §7.12.2.18 candidate reported, not dropped");

  // A TIP neighbour of an ordinary block needs §7.12.2.17.
  Grid tip_neighbour;
  tip_neighbour.init(16, 16);
  tip_neighbour.set(3, 3, merge::kTipFrame, merge::kRefNone, mc::Mv{8, 8}, mc::Mv{8, 8});
  merge::StackParams np = make_params(tip_neighbour, 4, 4, 2, 2, 0, merge::kRefNone);
  np.closest_past = 0;
  np.closest_future = 1;
  t.eq_int(static_cast<int>(merge::build_stack_checked(np, &stack)),
           static_cast<int>(merge::Status::kUnsupportedFeature),
           "§7.12.2.17 TIP neighbour reported, not dropped");
}

int main() {
  Test t;
  std::printf("AV2 M2 merge / skip / DRL tests\n");
  test_skip_mode_params(t);
  test_skip_mode_frames(t);
  test_skip_mode_block_gate(t);
  test_drl(t);
  test_single_mode_and_resolution(t);
  test_skip_mode_resolution(t);
  test_mode_contexts(t);
  test_stack_scan_and_sort(t);
  test_mvp_gate_all_block_sizes(t);
  test_full_stack_audit_64x64(t);
  test_sb_border_scan_audit(t);
  test_compound_stack(t);
  test_derived_projection(t);
  test_invalid_stack_params(t);
  test_tip_ref_stack(t);

  std::printf("%d checks, %d failures\n", t.checks, t.failures);
  return t.failures == 0 ? 0 : 1;
}
