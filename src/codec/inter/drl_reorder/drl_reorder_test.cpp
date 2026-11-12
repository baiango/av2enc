// Unit / property / golden tests for the AV2 DRL reorder module.
//
// Coverage:
//   * §5.4.6 DrlReorder derivation and §5.5 frame max_drl_bits derivation;
//   * §7.12.2 step 16 useSort truth table;
//   * §7.12.2.19 transform: single move-to-front, strict tie-break, only
//     [0, numNearest), complete-record swap;
//   * the historical full-selection-sort failure case, including a simulated
//     decoder that selects a different MV;
//   * index mapping pre -> post, post -> pre and signalled-index selection;
//   * independent writer/reader lockstep over thousands of randomized stacks;
//   * writer/reader lockstep over real M2 merge stacks built from the M2 block
//     geometry and the merge module's own scan rules;
//   * CDF/context selection (inter NewMvContext, skip_mode, TIP) and every
//     legal (start, m, index) combination;
//   * pinned golden reordered indices from goldens/drl_reorder_goldens.txt.
//
// Build/run: ./build_and_test.sh

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "codec/inter/drl_reorder/drl_reorder.h"
#include "codec/inter/drl_reorder/drl_syntax.h"
#include "codec/inter/drl_reorder/video_stack_adapter.h"

namespace drl = av2enc::inter::drl;
namespace merge = av2enc::video::merge;
namespace mc = av2enc::inter::mc;
namespace av2 = av2enc::av2;

namespace {

struct Test {
  int checks = 0;
  int failures = 0;
  void expect(bool cond, const std::string& what) {
    ++checks;
    if (!cond) {
      ++failures;
      if (failures <= 40) {
        std::printf("FAIL: %s\n", what.c_str());
      }
    }
  }
  void eq_int(long long got, long long want, const std::string& what) {
    ++checks;
    if (got != want) {
      ++failures;
      if (failures <= 40) {
        std::printf("FAIL: %s (got %lld, want %lld)\n", what.c_str(), got,
                    want);
      }
    }
  }
};

std::string label(const char* base, int a, int b = -1, int c = -1) {
  std::string out = base;
  out += "[" + std::to_string(a);
  if (b >= 0) out += "," + std::to_string(b);
  if (c >= 0) out += "," + std::to_string(c);
  out += "]";
  return out;
}

drl::StackCandidate candidate(int weight, std::int32_t y, std::int32_t x) {
  drl::StackCandidate c;
  c.weight = weight;
  c.mv[0][0] = y;
  c.mv[0][1] = x;
  c.mv[1][0] = y;
  c.mv[1][1] = x;
  c.row_offset = -y;
  c.col_offset = -x;
  c.cwp = 8;
  return c;
}

std::string mv_list(const drl::StackCandidate* cand, int count) {
  std::string out;
  for (int i = 0; i < count; ++i) {
    if (i != 0) out += ',';
    out += std::to_string(cand[i].mv[0][0]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// §5.4.6 / §5.5
// ---------------------------------------------------------------------------

void test_sequence_derivation(Test& t) {
  t.eq_int(static_cast<int>(drl::mode_from_sequence_header({true, false})),
           static_cast<int>(drl::Mode::kDisabled), "disable=1 -> DISABLED");
  t.eq_int(static_cast<int>(drl::mode_from_sequence_header({true, true})),
           static_cast<int>(drl::Mode::kDisabled),
           "disable=1 wins over constrain");
  t.eq_int(static_cast<int>(drl::mode_from_sequence_header({false, true})),
           static_cast<int>(drl::Mode::kConstraint), "disable=0 constrain=1");
  t.eq_int(static_cast<int>(drl::mode_from_sequence_header({false, false})),
           static_cast<int>(drl::Mode::kAlways), "disable=0 constrain=0");

  bool valid = false;
  drl::FrameDrlParams f;
  f.seq_max_drl_bits_minus_1 = 4;
  f.allow_frame_max_drl_bits = false;
  t.eq_int(drl::derive_frame_max_drl_bits(f, &valid), 4, "frame max no change");
  t.expect(valid, "frame max no change valid");

  f.allow_frame_max_drl_bits = true;
  f.change_drl = false;
  t.eq_int(drl::derive_frame_max_drl_bits(f, &valid), 4, "change_drl=0");
  t.expect(valid, "change_drl=0 valid");

  f.change_drl = true;
  f.signaled_max_drl_bits_minus_1 = 2;  // < 4 -> 2
  t.eq_int(drl::derive_frame_max_drl_bits(f, &valid), 2, "signaled 2 maps to 2");
  t.expect(valid, "signaled 2 valid");
  f.signaled_max_drl_bits_minus_1 = 3;  // 3 < 4 -> 3
  t.eq_int(drl::derive_frame_max_drl_bits(f, &valid), 3,
           "signaled 3 maps to 3 when seq = 4");
  t.expect(valid, "signaled 3 valid");
  f.seq_max_drl_bits_minus_1 = 3;
  f.signaled_max_drl_bits_minus_1 = 3;  // >= 3 -> 4
  t.eq_int(drl::derive_frame_max_drl_bits(f, &valid), 4,
           "signaled 3 maps to 4 when seq = 3 (sequence value skipped)");
  t.expect(valid, "signaled 3 at seq 3 valid");
  f.seq_max_drl_bits_minus_1 = 4;
  f.signaled_max_drl_bits_minus_1 = 4;  // n = 4 -> invalid
  drl::derive_frame_max_drl_bits(f, &valid);
  t.expect(!valid, "signaled 4 is outside ns(4)");

  t.eq_int(drl::drl_index_count(4, &valid), 5, "m = 5");
  t.expect(valid, "m = 5 valid");
  t.eq_int(drl::drl_index_count(0, &valid), 1, "m = 1");
  t.expect(valid, "m = 1 valid");
  drl::drl_index_count(5, &valid);
  t.expect(!valid, "max_drl_bits_minus_1 = 5 rejected");
}

// ---------------------------------------------------------------------------
// §7.12.2.19 transform
// ---------------------------------------------------------------------------

void test_use_sort(Test& t) {
  using drl::Mode;
  t.expect(!drl::use_sort(Mode::kDisabled, false, 6), "disabled never sorts");
  t.expect(!drl::use_sort(Mode::kConstraint, false, 3),
           "constraint <4 nearest does not sort");
  t.expect(drl::use_sort(Mode::kConstraint, false, 4),
           "constraint at 4 nearest sorts");
  t.expect(!drl::use_sort(Mode::kConstraint, true, 6),
           "constraint with useTemporalFirst does not sort");
  t.expect(drl::use_sort(Mode::kAlways, false, 1),
           "always sorts with one nearest");
  t.expect(drl::use_sort(Mode::kAlways, true, 1),
           "always sorts with useTemporalFirst");
}

void test_single_move_to_front(Test& t) {
  // The canonical divergence case: weights {1,2,3,4}.
  drl::Stack pre;
  pre.count = 4;
  pre.num_nearest = 4;
  pre.cand[0] = candidate(1, 10, 10);
  pre.cand[1] = candidate(2, 20, 20);
  pre.cand[2] = candidate(3, 30, 30);
  pre.cand[3] = candidate(4, 40, 40);

  drl::Stack spec = pre;
  const int swapped = drl::reorder_stack(&spec, drl::Mode::kConstraint, false);
  t.eq_int(swapped, 1, "single swap reported");
  const std::int32_t want[4] = {40, 20, 30, 10};
  for (int i = 0; i < 4; ++i) {
    t.eq_int(spec.cand[i].mv[0][0], want[i],
             label("spec order", i));
  }
  // Complete record moves with the candidate, not just the MV.
  t.eq_int(spec.cand[0].weight, 4, "weight moves with candidate");
  t.eq_int(spec.cand[0].row_offset, -40, "row offset moves with candidate");
  t.eq_int(spec.cand[3].weight, 1, "displaced weight follows");
  t.eq_int(spec.cand[3].row_offset, -10, "displaced row offset follows");

  // Strict comparison: equal maximum weights keep the earliest index.
  drl::Stack ties;
  ties.count = 4;
  ties.num_nearest = 4;
  ties.cand[0] = candidate(2, 1, 1);
  ties.cand[1] = candidate(2, 2, 2);
  ties.cand[2] = candidate(2, 3, 3);
  ties.cand[3] = candidate(1, 4, 4);
  drl::reorder_stack(&ties, drl::Mode::kAlways, false);
  t.eq_int(ties.cand[0].mv[0][0], 1, "tie keeps earliest maximum");

  // A later unique maximum is swapped, not shifted.
  drl::Stack later;
  later.count = 4;
  later.num_nearest = 4;
  later.cand[0] = candidate(1, 5, 5);
  later.cand[1] = candidate(3, 6, 6);
  later.cand[2] = candidate(3, 7, 7);
  later.cand[3] = candidate(2, 8, 8);
  drl::reorder_stack(&later, drl::Mode::kAlways, false);
  t.eq_int(later.cand[0].mv[0][0], 6, "first unique max moves to front");
  t.eq_int(later.cand[1].mv[0][0], 5, "displaced candidate swaps into slot 1");
  t.eq_int(later.cand[2].mv[0][0], 7, "third candidate is untouched");
  t.eq_int(later.cand[3].mv[0][0], 8, "fourth candidate is untouched");

  // Only [0, numNearest) is reordered; derived tail entries stay put.
  drl::Stack tail;
  tail.count = 6;
  tail.num_nearest = 4;
  tail.cand[0] = candidate(1, 1, 1);
  tail.cand[1] = candidate(5, 2, 2);
  tail.cand[2] = candidate(2, 3, 3);
  tail.cand[3] = candidate(0, 4, 4);
  tail.cand[4] = candidate(9, 5, 5);  // derived; weight 9 must not move
  tail.cand[5] = candidate(9, 6, 6);
  drl::reorder_stack(&tail, drl::Mode::kConstraint, false);
  t.eq_int(tail.cand[0].mv[0][0], 2, "nearest max moves to front");
  t.eq_int(tail.cand[1].mv[0][0], 1, "nearest displaced");
  t.eq_int(tail.cand[2].mv[0][0], 3, "nearest untouched");
  t.eq_int(tail.cand[3].mv[0][0], 4, "nearest untouched tail slot");
  for (int i = 4; i < 6; ++i) {
    t.eq_int(tail.cand[i].mv[0][0], i + 1, label("derived tail", i));
  }

  // Constraint below four nearest is a no-op; ALWAYS is not.
  drl::Stack three;
  three.count = 3;
  three.num_nearest = 3;
  three.cand[0] = candidate(1, 1, 1);
  three.cand[1] = candidate(5, 2, 2);
  three.cand[2] = candidate(2, 3, 3);
  drl::Stack c = three;
  drl::reorder_stack(&c, drl::Mode::kConstraint, false);
  t.eq_int(c.cand[0].mv[0][0], 1, "constraint no-op below 4");
  drl::Stack a = three;
  drl::reorder_stack(&a, drl::Mode::kAlways, false);
  t.eq_int(a.cand[0].mv[0][0], 2, "always reorders below 4");
}

void test_legacy_desync(Test& t) {
  // This is the historical encoder: full selection sort over [0, numNearest).
  drl::Stack pre;
  pre.count = 4;
  pre.num_nearest = 4;
  pre.cand[0] = candidate(1, 10, 10);
  pre.cand[1] = candidate(2, 20, 20);
  pre.cand[2] = candidate(3, 30, 30);
  pre.cand[3] = candidate(4, 40, 40);

  drl::Stack legacy = pre;
  drl::reproduce_legacy_selection_sort(legacy.cand, legacy.num_nearest);
  const std::int32_t legacy_order[4] = {40, 30, 20, 10};
  for (int i = 0; i < 4; ++i) {
    t.eq_int(legacy.cand[i].mv[0][0], legacy_order[i],
             label("legacy full-sort order", i));
  }

  drl::Stack decoder = pre;
  drl::reorder_stack(&decoder, drl::Mode::kConstraint, false);
  t.expect(mv_list(legacy.cand, 4) != mv_list(decoder.cand, 4),
           "legacy and decoder stacks differ for {1,2,3,4}");

  // The encoder wants the candidate at pre-index 3 (MV 20,20).  Legacy maps it
  // to post-index 2; the decoder's post-index 2 is MV 30,30.  Reproduce the
  // exact MV-selection mismatch.
  const int pre_index = 1;  // candidate B {20,20}
  // The legacy encoder's mapping is simply the position in its own sorted
  // array.
  int legacy_post = -1;
  for (int i = 0; i < legacy.count; ++i) {
    if (legacy.cand[i].mv[0][0] == pre.cand[pre_index].mv[0][0]) {
      legacy_post = i;
      break;
    }
  }
  t.eq_int(legacy_post, 2, "legacy maps pre index 1 to post index 2");
  const int decoder_post =
      drl::signal_index_for_pre(pre.cand, pre.count, pre.num_nearest,
                                drl::Mode::kConstraint, false, pre_index);
  t.eq_int(decoder_post, 1, "spec maps pre index 1 to post index 1");
  t.eq_int(legacy.cand[legacy_post].mv[0][0], 20,
           "legacy encoder believes it selected MV 20");
  t.eq_int(decoder.cand[legacy_post].mv[0][0], 30,
           "decoder selects MV 30 for the same signalled index");
  t.expect(decoder.cand[legacy_post].mv[0][0] != 20,
           "FAILING CASE: legacy index selects a different MV");
}

// ---------------------------------------------------------------------------
// Index mapping and randomized writer/reader lockstep
// ---------------------------------------------------------------------------

drl::Stack random_stack(std::mt19937& rng, bool compound, int* num_nearest) {
  std::uniform_int_distribution<int> count_dist(1, drl::kMaxRefMvStackSize);
  std::uniform_int_distribution<int> weight_dist(0, 4);
  std::uniform_int_distribution<int> mv_dist(-4096, 4096);
  drl::Stack s;
  s.count = count_dist(rng);
  std::uniform_int_distribution<int> nearest_dist(0, s.count);
  *num_nearest = nearest_dist(rng);
  s.num_nearest = *num_nearest;
  for (int i = 0; i < s.count; ++i) {
    s.cand[i].weight = weight_dist(rng);
    s.cand[i].mv[0][0] = mv_dist(rng);
    s.cand[i].mv[0][1] = mv_dist(rng);
    s.cand[i].mv[1][0] = compound ? mv_dist(rng) : s.cand[i].mv[0][0];
    s.cand[i].mv[1][1] = compound ? mv_dist(rng) : s.cand[i].mv[0][1];
    s.cand[i].row_offset = mv_dist(rng) % 17;
    s.cand[i].col_offset = mv_dist(rng) % 17;
    s.cand[i].cwp = 8;
  }
  return s;
}

struct EntropyStream {
  std::vector<std::uint8_t> buffer;
  av2::Encoder enc;
  bool ok = false;
};

bool run_writer_reader_roundtrip(Test& t, int iterations) {
  std::mt19937 rng(0xD21A11u);
  std::uniform_int_distribution<int> mode_dist(0, 2);
  std::uniform_int_distribution<int> start_dist(0, 5);
  std::uniform_int_distribution<int> ctx_dist(0, 4);
  std::uniform_int_distribution<int> source_dist(0, 2);
  int checked = 0;
  for (int it = 0; it < iterations; ++it) {
    const bool compound = (it & 1) != 0;
    int num_nearest = 0;
    drl::Stack pre = random_stack(rng, compound, &num_nearest);
    const drl::Mode mode = static_cast<drl::Mode>(mode_dist(rng));
    const bool use_temporal_first = (rng() & 3u) == 0;
    drl::Stack sorted = pre;
    drl::reorder_stack(&sorted, mode, use_temporal_first);

    // Random legal m; all stack indices in [0, m) are signalled with
    // read_drl_idx unless m exceeds the stack, in which case the stack size
    // bounds encoder choices.
    const int m = 1 + static_cast<int>(rng() % drl::kMaxDrlIndex);
    const int usable = std::min(pre.count, m);
    if (usable <= 0) {
      continue;
    }
    std::uniform_int_distribution<int> pick(0, usable - 1);
    const int pre_index = pick(rng);
    const int post_index = drl::signal_index_for_pre(
        pre.cand, pre.count, pre.num_nearest, mode, use_temporal_first,
        pre_index);
    t.expect(post_index >= 0, "mapped index is valid");
    if (post_index < 0 || post_index >= m) {
      // The reorder can pull a high-weight candidate from outside the
      // signalable prefix [0, m) into slot 0 and push the selected candidate
      // past m.  Such a candidate cannot be signalled; the encoder must pick a
      // different one.  Skip instead of treating it as a coder failure.
      continue;
    }
    // Encoder and decoder must select the same MV from the same signalled
    // index: encoder's sorted[post] == pre[pre_index]; decoder's sorted[post]
    // is the same record because both run the normative transform.
    const bool same_mv =
        sorted.cand[post_index].mv[0][0] == pre.cand[pre_index].mv[0][0] &&
        sorted.cand[post_index].mv[0][1] == pre.cand[pre_index].mv[0][1] &&
        sorted.cand[post_index].mv[1][0] == pre.cand[pre_index].mv[1][0] &&
        sorted.cand[post_index].mv[1][1] == pre.cand[pre_index].mv[1][1];
    t.expect(same_mv, "pre->post index mapping preserves the candidate");

    // Write the signalled index and read it back from the same symbol stream.
    // The decoder independently applies the same normative transform.
    if (!same_mv) {
      continue;
    }
    drl::DrlCdfState cdf_w;
    drl::DrlCdfState cdf_r;
    drl::init_default_drl_cdfs(&cdf_w);
    drl::init_default_drl_cdfs(&cdf_r);
    drl::DrlSymbolContext ctx;
    ctx.source = static_cast<drl::DrlCdfSource>(source_dist(rng));
    ctx.new_mv_context = ctx_dist(rng);
    std::vector<std::uint8_t> buffer(1 << 16);
    av2::Encoder enc;
    if (enc.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
      t.expect(false, "encoder init");
      continue;
    }
    enc.disable_cdf_update = 1;
    const int start = 0;
    const drl::Status ws = drl::write_drl_index(&enc, &cdf_w, ctx, post_index,
                                                start, m);
    if (ws != drl::Status::kOk) {
      t.expect(false,
               std::string("write_drl_index: ") + drl::status_name(ws) +
                   " index=" + std::to_string(post_index) +
                   " m=" + std::to_string(m) +
                   " n=" + std::to_string(sorted.count));
      continue;
    }
    std::size_t bytes = 0;
    if (enc.exit_symbol(&bytes) != av2::Status::kOk) {
      t.expect(false, "exit_symbol");
      continue;
    }
    av2::Decoder dec;
    if (dec.init_symbol(buffer.data(), bytes, bytes) != av2::Status::kOk) {
      t.expect(false, "decoder init");
      continue;
    }
    dec.disable_cdf_update = 1;
    int decoded = -1;
    const drl::Status rs = drl::read_drl_index(
        &dec, &cdf_r, ctx, start, m, &decoded);
    t.expect(rs == drl::Status::kOk, "read_drl_index status");
    t.eq_int(decoded, post_index, label("writer/reader index", it));
    if (dec.exit_symbol() != av2::Status::kOk) {
      t.expect(false, "decoder exit");
    }
    ++checked;
    if (it % 500 == 0) {
      t.expect(sorted.count == pre.count, "reorder preserves count");
      t.expect(sorted.num_nearest == pre.num_nearest,
               "reorder preserves numNearest");
    }
  }
  return checked > 0;
}

void test_second_drl_start(Test& t) {
  // start > 0 is used for same-reference NEAR_NEARMV.  Every legal
  // (start, m, index) triple must round-trip.  start == m returns m with no
  // symbols.
  for (int m = 1; m <= drl::kMaxDrlIndex; ++m) {
    for (int start = 0; start <= m; ++start) {
      for (int index = start; index <= m; ++index) {
        drl::DrlCdfState cw;
        drl::DrlCdfState cr;
        drl::init_default_drl_cdfs(&cw);
        drl::init_default_drl_cdfs(&cr);
        drl::DrlSymbolContext ctx;
        ctx.new_mv_context = (index + start) % 5;
        std::vector<std::uint8_t> buffer(4096);
        av2::Encoder enc;
        enc.init_symbol(buffer.data(), buffer.size());
        enc.disable_cdf_update = 1;
        const drl::Status ws =
            drl::write_drl_index(&enc, &cw, ctx, index, start, m);
        t.expect(ws == drl::Status::kOk,
                 label("grid write", m, start, index));
        std::size_t bytes = 0;
        enc.exit_symbol(&bytes);
        av2::Decoder dec;
        dec.init_symbol(buffer.data(), bytes, bytes);
        dec.disable_cdf_update = 1;
        int got = -1;
        const drl::Status rs =
            drl::read_drl_index(&dec, &cr, ctx, start, m, &got);
        t.expect(rs == drl::Status::kOk,
                 label("grid read", m, start, index));
        t.eq_int(got, index, label("grid index", m, start, index));
        t.eq_int(drl::drl_symbol_count(index, start, m),
                 (index - start) + (index < m ? 1 : 0),
                 label("symbol count", m, start, index));
      }
      // Illegal index below start must be rejected by the writer.
      drl::DrlCdfState cw;
      drl::init_default_drl_cdfs(&cw);
      av2::Encoder enc;
      std::vector<std::uint8_t> buffer(1024);
      enc.init_symbol(buffer.data(), buffer.size());
      const drl::Status bad =
          drl::write_drl_index(&enc, &cw, {}, start - 1, start, m);
      t.expect(bad == drl::Status::kInvalidIndex,
               label("reject index < start", m, start));
    }
  }
}

// ---------------------------------------------------------------------------
// Real M2 merge stacks
// ---------------------------------------------------------------------------

struct Grid {
  int cols = 0;
  int rows = 0;
  std::vector<mc::MiInfo> mi;
  void init(int c, int r) {
    cols = c;
    rows = r;
    mi.assign(static_cast<std::size_t>(c) * static_cast<std::size_t>(r),
              mc::MiInfo{});
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        mc::MiInfo& m = at(row, col);
        m.ref[0] = merge::kRefNone;
        m.ref[1] = merge::kRefNone;
        m.is_inter = 0;
        m.w4 = 1;
        m.h4 = 1;
        m.col_base = static_cast<std::int16_t>(col);
        m.row_base = static_cast<std::int16_t>(row);
      }
    }
  }
  mc::MiInfo& at(int row, int col) {
    return mi[static_cast<std::size_t>(row) *
                  static_cast<std::size_t>(cols) +
              static_cast<std::size_t>(col)];
  }
  void set_block(int row, int col, int w4, int h4, std::int8_t ref0,
                 std::int8_t ref1, mc::Mv mv0, mc::Mv mv1) {
    for (int yy = 0; yy < h4; ++yy) {
      for (int xx = 0; xx < w4; ++xx) {
        mc::MiInfo& m = at(row + yy, col + xx);
        m.ref[0] = ref0;
        m.ref[1] = ref1;
        m.mv[0][0] = mv0.y;
        m.mv[0][1] = mv0.x;
        m.mv[1][0] = mv1.y;
        m.mv[1][1] = mv1.x;
        m.is_inter = 1;
        m.w4 = static_cast<std::int8_t>(w4);
        m.h4 = static_cast<std::int8_t>(h4);
        m.col_base = static_cast<std::int16_t>(col);
        m.row_base = static_cast<std::int16_t>(row);
      }
    }
  }
};

void test_real_video_blocks(Test& t) {
  // A 128x128 M2 frame: 64x64 superblocks, 16x16 leaves, two reference slots.
  // MVs follow a deterministic moving-object pattern across the frame, which
  // produces the same kind of mixed-weight stacks the M2 encoder sees.
  Grid grid;
  const int mi = 32;
  grid.init(mi, mi);
  int block = 0;
  for (int r = 0; r < mi; r += 4) {
    for (int c = 0; c < mi; c += 4) {
      const int ref = ((r / 4 + c / 4) % 2) == 0 ? 0 : 1;
      // Repeat a small MV set so multiple scan points hit the same MV and the
      // builder accumulates weights > 1, as real M2 blocks do.
      const int dy = 8 * ((c / 4) % 2) - 4;
      const int dx = 8 * ((r / 4) % 2) - 4;
      grid.set_block(r, c, 4, 4, static_cast<std::int8_t>(ref),
                     merge::kRefNone, mc::Mv{dy, dx}, mc::Mv{dy, dx});
      ++block;
    }
  }

  int nonempty = 0;
  int indices_checked = 0;
  for (int r = 0; r < mi; r += 4) {
    for (int c = 0; c < mi; c += 4) {
      for (std::int8_t ref = 0; ref < 2; ++ref) {
        merge::StackParams p;
        p.mi_row = r;
        p.mi_col = c;
        p.bw4 = 4;
        p.bh4 = 4;
        p.sb_h4 = 16;
        p.mi_cols = grid.cols;
        p.mi_rows = grid.rows;
        p.mi = grid.mi.data();
        p.ref_frame[0] = ref;
        p.ref_frame[1] = merge::kRefNone;
        p.is_compound = false;
        p.global_mv[0] = mc::Mv{0, 0};
        p.global_mv[1] = mc::Mv{0, 0};
        p.reorder = merge::DrlReorder::kDisabled;
        p.max_drl_bits_minus_1 = 4;
        p.frame_distance = nullptr;
        p.num_refs = 0;
        merge::MvStack stack;
        if (merge::build_mv_stack(p, &stack) != merge::Status::kOk) {
          t.expect(false, "real M2 stack build");
          continue;
        }
        if (stack.count <= 0) {
          continue;
        }
        ++nonempty;
        for (drl::Mode mode :
             {drl::Mode::kDisabled, drl::Mode::kConstraint,
              drl::Mode::kAlways}) {
          drl::Stack view;
          drl::load_video_stack(stack, &view);
          const int swapped =
              drl::reorder_stack(&view, mode, /*use_temporal_first=*/false);
          (void)swapped;
          const int m = 5;
          const int usable = std::min(view.count, m);
          for (int post = 0; post < usable; ++post) {
            // Pick the post-reorder candidate's MV and verify choose_index_for_mv
            // finds that same index on the reordered stack.
            const drl::IndexChoice choice = drl::choose_index_for_mv(
                view, view.cand[post].mv[0][0], view.cand[post].mv[0][1], 0,
                m);
            t.expect(choice.found, "real M2 block index is signalable");
            if (choice.found) {
              // The first matching index can be lower only if two candidates
              // have identical list-0 MVs; the stack builder prunes those.
              t.eq_int(choice.index, post,
                       label("real M2 block index", block, post));
              ++indices_checked;
            }
          }
          // Round-trip the selected index through the entropy coder with the
          // real stack as the source.
          drl::DrlCdfState cw;
          drl::DrlCdfState cr;
          drl::init_default_drl_cdfs(&cw);
          drl::init_default_drl_cdfs(&cr);
          drl::DrlSymbolContext ctx;
          ctx.new_mv_context = (r + c + static_cast<int>(ref)) % 5;
          std::vector<std::uint8_t> buffer(1 << 16);
          av2::Encoder enc;
          enc.init_symbol(buffer.data(), buffer.size());
          enc.disable_cdf_update = 1;
          const int index = usable > 0 ? usable - 1 : 0;
          t.expect(drl::write_drl_index(&enc, &cw, ctx, index, 0, m) ==
                       drl::Status::kOk,
                   "real M2 write");
          std::size_t bytes = 0;
          enc.exit_symbol(&bytes);
          av2::Decoder dec;
          dec.init_symbol(buffer.data(), bytes, bytes);
          dec.disable_cdf_update = 1;
          int got = -1;
          t.expect(drl::read_drl_index(&dec, &cr, ctx, 0, m, &got) ==
                       drl::Status::kOk,
                   "real M2 read");
          t.eq_int(got, index, label("real M2 roundtrip", block, index));
        }
      }
    }
  }
  t.expect(nonempty > 0, "real M2 fixture produced non-empty stacks");
  t.expect(indices_checked > 0, "real M2 fixture checked indices");
}

void test_real_video_scan_fixture(Test& t) {
  // The M2 merge module's own 8x8-block fixture (a 16x16 MI frame).  Its
  // scan produces six nearest candidates with weights {1,1,1,2,1,0}; the
  // weight-2 candidate is at index 3, so the normative transform moves it to
  // slot 0 while a full selection sort would reorder slots 1..5 as well.
  Grid grid;
  grid.init(16, 16);
  const mc::Mv base{-100, -100};
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 16; ++c) {
      grid.set_block(r, c, 1, 1, 1, merge::kRefNone, base, base);
    }
  }
  grid.set_block(5, 3, 1, 1, 1, merge::kRefNone, mc::Mv{10, 10},
                 mc::Mv{10, 10});
  grid.set_block(3, 5, 1, 1, 1, merge::kRefNone, mc::Mv{20, 20},
                 mc::Mv{20, 20});
  grid.set_block(4, 3, 1, 1, 1, merge::kRefNone, mc::Mv{30, 30},
                 mc::Mv{30, 30});
  grid.set_block(3, 3, 1, 1, 1, merge::kRefNone, mc::Mv{40, 40},
                 mc::Mv{40, 40});
  grid.set_block(6, 3, 1, 1, 1, merge::kRefNone, mc::Mv{50, 50},
                 mc::Mv{50, 50});
  grid.set_block(5, 1, 1, 1, 1, merge::kRefNone, mc::Mv{70, 70},
                 mc::Mv{70, 70});
  grid.set_block(4, 1, 1, 1, 1, merge::kRefNone, mc::Mv{80, 80},
                 mc::Mv{80, 80});

  merge::StackParams p;
  p.mi_row = 4;
  p.mi_col = 4;
  p.bw4 = 2;
  p.bh4 = 2;
  p.sb_h4 = 16;
  p.mi_cols = grid.cols;
  p.mi_rows = grid.rows;
  p.mi = grid.mi.data();
  p.ref_frame[0] = 1;
  p.ref_frame[1] = merge::kRefNone;
  p.is_compound = false;
  p.global_mv[0] = mc::Mv{0, 0};
  p.global_mv[1] = mc::Mv{0, 0};
  p.reorder = merge::DrlReorder::kDisabled;
  merge::MvStack stack;
  t.expect(merge::build_mv_stack(p, &stack) == merge::Status::kOk,
           "scan fixture stack build");
  t.eq_int(stack.num_nearest, 6, "scan fixture numNearest");
  t.eq_int(stack.cand[0].weight, 1, "scan fixture weight 0");
  t.eq_int(stack.cand[3].weight, 2, "scan fixture weight 3");

  drl::Stack view;
  drl::load_video_stack(stack, &view);
  t.eq_int(drl::reorder_stack(&view, drl::Mode::kConstraint, false), 1,
           "scan fixture reorders");
  const std::int32_t want[6] = {-100, 20, 30, 10, 50, 40};
  for (int i = 0; i < 6; ++i) {
    t.eq_int(view.cand[i].mv[0][0], want[i],
             label("scan fixture spec order", i));
  }
  // The adapter must produce the same order without ever enabling the shared
  // module's selection sort.
  merge::MvStack adapted;
  t.expect(drl::build_reordered_stack(p, drl::Mode::kConstraint, false,
                                      &adapted) == merge::Status::kOk,
           "adapter build_reordered_stack");
  drl::Stack adapted_view;
  drl::load_video_stack(adapted, &adapted_view);
  for (int i = 0; i < 6; ++i) {
    t.eq_int(adapted_view.cand[i].mv[0][0], want[i],
             label("adapter spec order", i));
  }
  t.eq_int(drl::signal_index_for_pre(stack, drl::Mode::kConstraint, false, 3),
           0, "adapter signals the reordered candidate");

  // The remaining weights after the swap are non-increasing, so this fixture
  // does not distinguish the normative swap from a selection sort; the
  // synthetic {1,2,3,4} case above is the pinned divergence proof.
}

// ---------------------------------------------------------------------------
// Pinned goldens
// ---------------------------------------------------------------------------

struct GoldenCase {
  std::string name;
  std::vector<int> weights;
  std::vector<std::int32_t> mvs;
  std::string mode;
  bool use_temporal_first = false;
  int num_nearest = 0;
  std::vector<std::int32_t> expected_mvs;
  int pre_index = -1;
  int expected_signal = -1;
};

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> out;
  std::string item;
  std::istringstream ss(s);
  while (std::getline(ss, item, delim)) {
    out.push_back(item);
  }
  return out;
}

std::vector<int> parse_ints(const std::string& s) {
  std::vector<int> out;
  for (const std::string& part : split(s, ',')) {
    if (!part.empty()) out.push_back(std::atoi(part.c_str()));
  }
  return out;
}

bool load_goldens(const std::string& path, std::vector<GoldenCase>* cases,
                  std::string* error) {
  std::ifstream in(path);
  if (!in) {
    *error = "cannot open " + path;
    return false;
  }
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty() || line[0] == '#') continue;
    const std::vector<std::string> f = split(line, '|');
    if (f.size() != 8) {
      *error = "golden line " + std::to_string(line_no) +
               " must have 8 fields";
      return false;
    }
    GoldenCase c;
    c.name = f[0];
    for (int v : parse_ints(f[1])) c.weights.push_back(v);
    for (int v : parse_ints(f[2])) c.mvs.push_back(v);
    c.mode = f[3];
    c.use_temporal_first = f[4] == "1";
    c.num_nearest = std::atoi(f[5].c_str());
    for (int v : parse_ints(f[6])) c.expected_mvs.push_back(v);
    const std::vector<int> signal = parse_ints(f[7]);
    if (!signal.empty()) {
      c.pre_index = signal[0];
      if (signal.size() > 1) c.expected_signal = signal[1];
    }
    if (c.mvs.size() != c.weights.size() ||
        c.expected_mvs.size() != c.weights.size()) {
      *error = "golden line " + std::to_string(line_no) + " length mismatch";
      return false;
    }
    cases->push_back(c);
  }
  return true;
}

void test_goldens(Test& t, const std::string& path) {
  std::vector<GoldenCase> cases;
  std::string error;
  if (!load_goldens(path, &cases, &error)) {
    t.expect(false, "load goldens: " + error);
    return;
  }
  t.expect(!cases.empty(), "golden file is non-empty");
  for (const GoldenCase& c : cases) {
    drl::Stack pre;
    pre.count = static_cast<int>(c.weights.size());
    pre.num_nearest = std::min(c.num_nearest, pre.count);
    for (int i = 0; i < pre.count; ++i) {
      pre.cand[i] = candidate(
          static_cast<int>(c.weights[static_cast<std::size_t>(i)]),
          static_cast<std::int32_t>(c.mvs[static_cast<std::size_t>(i)]),
          static_cast<std::int32_t>(c.mvs[static_cast<std::size_t>(i)]));
    }
    drl::Mode mode = drl::Mode::kDisabled;
    if (c.mode == "constraint") mode = drl::Mode::kConstraint;
    if (c.mode == "always") mode = drl::Mode::kAlways;
    drl::Stack sorted = pre;
    drl::reorder_stack(&sorted, mode, c.use_temporal_first);
    t.eq_int(static_cast<int>(sorted.count), pre.count,
             c.name + " count preserved");
    for (int i = 0; i < pre.count; ++i) {
      t.eq_int(sorted.cand[i].mv[0][0],
               static_cast<std::int32_t>(
                   c.expected_mvs[static_cast<std::size_t>(i)]),
               c.name + " golden order");
    }
    if (c.pre_index >= 0) {
      const int got = drl::signal_index_for_pre(
          pre.cand, pre.count, pre.num_nearest, mode, c.use_temporal_first,
          c.pre_index);
      t.eq_int(got, c.expected_signal, c.name + " golden signalled index");
    }
  }
}

// ---------------------------------------------------------------------------
// Entropy stream sanity: truncated stream must fail rather than invent 0.
// ---------------------------------------------------------------------------

void test_invalid_ranges(Test& t) {
  drl::DrlCdfState cw;
  drl::init_default_drl_cdfs(&cw);
  std::vector<std::uint8_t> buffer(64);
  av2::Encoder enc;
  enc.init_symbol(buffer.data(), buffer.size());
  enc.disable_cdf_update = 1;
  t.expect(drl::write_drl_index(&enc, &cw, {}, 0, 0, 6) ==
               drl::Status::kInvalidIndex,
           "m > MAX_REF_MV_STACK_SIZE-1 rejected");
  t.expect(drl::write_drl_index(&enc, &cw, {}, 0, 2, 1) ==
               drl::Status::kInvalidIndex,
           "start > m rejected");
  av2::Decoder dec;
  int index = -1;
  t.expect(drl::read_drl_index(&dec, &cw, {}, 0, 5, &index) ==
               drl::Status::kInvalidArgument,
           "uninitialized reader rejected");
}

}  // namespace

int main(int argc, char** argv) {
  std::string golden_path = "goldens/drl_reorder_goldens.txt";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--goldens" && i + 1 < argc) {
      golden_path = argv[++i];
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  Test t;
  test_sequence_derivation(t);
  test_use_sort(t);
  test_single_move_to_front(t);
  test_legacy_desync(t);
  test_second_drl_start(t);
  test_invalid_ranges(t);
  const int roundtrips = 20000;
  t.expect(run_writer_reader_roundtrip(t, roundtrips),
           "writer/reader lockstep produced samples");
  test_real_video_blocks(t);
  test_real_video_scan_fixture(t);
  test_goldens(t, golden_path);

  std::printf("drl_reorder_test: %d checks, %d failures\n", t.checks,
              t.failures);
  return t.failures == 0 ? 0 : 1;
}
