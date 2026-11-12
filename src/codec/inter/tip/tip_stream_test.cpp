// Unit tests + pinned goldens for inter/tip_stream.
//
//   tip_stream_test --goldens goldens/tip_goldens.txt
//   tip_stream_test --print-goldens
//
// Everything here is deterministic; there is no clock, allocation order or
// floating point in the pinned values.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "codec/inter/temporal/motion_field.h"
#include "codec/inter/temporal/temporal_math.h"
#include "codec/inter/temporal/temporal_predict.h"
#include "codec/inter/tip/tip_build.h"
#include "codec/inter/tip/tip_candidates.h"
#include "codec/inter/tip/tip_output.h"
#include "codec/inter/tip/tip_signalling.h"

using av2enc::BitWriter;

namespace tip = av2enc::inter::tip;
namespace temporal = av2enc::inter::temporal;

namespace {

int g_checks = 0;
int g_failures = 0;
std::map<std::string, std::string> g_goldens;

void check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

void check_eq(std::int64_t got, std::int64_t want, const std::string& what) {
  ++g_checks;
  if (got != want) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s (got %lld, want %lld)\n", what.c_str(),
                 static_cast<long long>(got), static_cast<long long>(want));
  }
}

std::string hex(std::span<const std::uint8_t> bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (std::uint8_t b : bytes) {
    out.push_back(kDigits[b >> 4]);
    out.push_back(kDigits[b & 0xf]);
  }
  return out;
}

std::string golden_bytes(std::size_t bits, std::span<const std::uint8_t> bytes) {
  return std::to_string(bits) + ":" + hex(bytes);
}

void golden(const std::string& name, const std::string& value) {
  g_goldens[name] = value;
  std::printf("%s = %s\n", name.c_str(), value.c_str());
}

std::uint64_t fnva(std::span<const std::uint8_t> data,
                   std::uint64_t state = 1469598103934665603ULL) {
  for (std::uint8_t b : data) {
    state ^= b;
    state *= 1099511628211ULL;
  }
  return state;
}

std::string hash_planes(const std::vector<std::uint8_t>& y,
                        const std::vector<std::uint8_t>& u,
                        const std::vector<std::uint8_t>& v) {
  std::uint64_t h = fnva(y);
  h = fnva(u, h);
  h = fnva(v, h);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(h));
  return buf;
}

// ---------------------------------------------------------------------------
// Owned motion field helpers.
// ---------------------------------------------------------------------------

struct MvFieldOwner {
  int w8 = 0;
  int h8 = 0;
  std::vector<std::int8_t> ref;
  std::vector<std::int16_t> mv;

  void init(int w, int h) {
    w8 = w;
    h8 = h;
    ref.assign(static_cast<std::size_t>(w) * h * 2, temporal::kRefNone);
    mv.assign(static_cast<std::size_t>(w) * h * 4, 0);
  }
  temporal::RefMvField view() {
    temporal::RefMvField out;
    out.w8 = w8;
    out.h8 = h8;
    out.ref = ref.data();
    out.mv = mv.data();
    return out;
  }
  temporal::SavedMvField saved() {
    temporal::SavedMvField out;
    out.w8 = w8;
    out.h8 = h8;
    out.ref = ref.data();
    out.mv = mv.data();
    return out;
  }
  // Constant field: every 8x8 uses list 0 and list 1 with `mv0`/`mv1`.
  void fill_constant(temporal::Mv mv0, temporal::Mv mv1) {
    for (int y = 0; y < h8; ++y) {
      for (int x = 0; x < w8; ++x) {
        const std::size_t pos =
            static_cast<std::size_t>(y) * w8 + static_cast<std::size_t>(x);
        ref[pos * 2 + 0] = 0;
        ref[pos * 2 + 1] = 0;
        mv[(pos * 2 + 0) * 2 + 0] = temporal::compression_mv(mv0.row);
        mv[(pos * 2 + 0) * 2 + 1] = temporal::compression_mv(mv0.col);
        mv[(pos * 2 + 1) * 2 + 0] = temporal::compression_mv(mv1.row);
        mv[(pos * 2 + 1) * 2 + 1] = temporal::compression_mv(mv1.col);
      }
    }
  }
};

struct FieldOwner {
  int w8 = 0;
  int h8 = 0;
  std::vector<std::int32_t> mv;
  std::vector<std::uint8_t> valid;
  std::vector<std::int32_t> offset;
  std::vector<std::int32_t> avg_mv;
  std::vector<std::uint8_t> avg_valid;

  void init(int mi_cols, int mi_rows) {
    w8 = mi_cols >> 1;
    h8 = mi_rows >> 1;
    const std::size_t cells = static_cast<std::size_t>(w8) * h8;
    mv.assign(cells * 2, 0);
    valid.assign(cells, 0);
    offset.assign(cells, 0);
    avg_mv.assign(cells * 2, 0);
    avg_valid.assign(cells, 0);
  }
  temporal::MotionField field() {
    temporal::MotionField out;
    out.w8 = w8;
    out.h8 = h8;
    out.mv = mv.data();
    out.valid = valid.data();
    out.offset = offset.data();
    return out;
  }
  temporal::MotionFieldScratch scratch() {
    temporal::MotionFieldScratch out;
    out.avg_mv = avg_mv.data();
    out.avg_valid = avg_valid.data();
    return out;
  }
};

struct PlanOwner {
  int mi_cols = 0;
  int mi_rows = 0;
  std::vector<temporal::TipBlock> blocks;
  std::vector<temporal::Mv> grid0;
  std::vector<temporal::Mv> grid1;
  int capacity = 0;

  void init(int cols, int rows, bool with_list1 = true) {
    mi_cols = cols;
    mi_rows = rows;
    capacity = ((cols + 3) / 4) * ((rows + 3) / 4);
    blocks.assign(static_cast<std::size_t>(capacity), temporal::TipBlock{});
    grid0.assign(static_cast<std::size_t>(cols) * rows, temporal::Mv{});
    if (with_list1) {
      grid1.assign(static_cast<std::size_t>(cols) * rows, temporal::Mv{});
    }
  }
  temporal::TipPlan plan() {
    temporal::TipPlan out;
    out.num_blocks = capacity;
    out.blocks = blocks.data();
    out.mv_grid = grid0.data();
    out.mv_grid1 = grid1.empty() ? nullptr : grid1.data();
    return out;
  }
};

struct PlaneOwner {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> data;
  void init(int w, int h, int seed) {
    width = w;
    height = h;
    data.resize(static_cast<std::size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int value = (x * 3 + y * 5 + seed * 17) & 0xff;
        data[static_cast<std::size_t>(y) * w +
             static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(value);
      }
    }
  }
  temporal::PlaneView view() const {
    temporal::PlaneView out;
    out.data = data.data();
    out.stride = width;
    out.width = width;
    out.height = height;
    return out;
  }
};

// A complete 3-plane picture.
struct FrameOwner {
  int width = 0;
  int height = 0;
  PlaneOwner plane[3];
  void init(int w, int h, int seed) {
    width = w;
    height = h;
    plane[0].init(w, h, seed);
    plane[1].init((w + 1) / 2, (h + 1) / 2, seed + 1);
    plane[2].init((w + 1) / 2, (h + 1) / 2, seed + 2);
  }
};

// ---------------------------------------------------------------------------
// Syntax tests.
// ---------------------------------------------------------------------------

tip::TipSequenceSyntax make_sequence() {
  tip::TipSequenceSyntax s;
  s.enable_ref_frame_mvs = true;
  s.reduced_ref_frame_mvs_mode = false;
  s.order_hint_bits_minus_1 = 3;
  s.enable_refmvbank = false;
  s.disable_drl_reorder = true;
  s.explicit_ref_frame_map = true;
  s.explicit_num_ref_frames = true;
  s.num_ref_frames_minus_1 = 7;
  s.seq_max_drl_bits_minus_1 = 4;
  s.num_same_ref_compound = 0;
  s.enable_tip = true;
  s.disable_tip_output = false;
  s.enable_tip_hole_fill = true;
  s.enable_df_sub_pu = false;
  s.enable_opfl_refine = temporal::RefineType::kNone;
  s.enable_refinemv = false;
  s.enable_tip_refinemv = false;
  return s;
}

void test_sequence_syntax() {
  const tip::TipSequenceSyntax base = make_sequence();
  {
    BitWriter writer;
    check(tip::write_tip_sequence_syntax(writer, base) ==
              temporal::TemporalStatus::kOk,
          "seq write");
    const std::size_t bits = writer.bit_size();
    temporal::BitReader reader(writer.bytes().data(), writer.bytes().size());
    tip::TipSequenceSyntax read;
    check(tip::read_tip_sequence_syntax(reader, &read) ==
              temporal::TemporalStatus::kOk,
          "seq read");
    check(read.enable_ref_frame_mvs && read.enable_tip &&
              read.enable_tip_hole_fill && !read.disable_tip_output &&
              read.enable_tip_output() && !read.enable_tip_refinemv,
          "seq round trip values");
    check_eq(static_cast<std::int64_t>(reader.position()),
             static_cast<std::int64_t>(bits), "seq consumed bits");
    golden("seq.video_tip", golden_bytes(bits, writer.bytes()));
  }
  {
    // TIP output disabled: no TIP OBU can exist, no explicit-QP bit.
    tip::TipSequenceSyntax s = base;
    s.disable_tip_output = true;
    BitWriter writer;
    check(tip::write_tip_sequence_syntax(writer, s) ==
              temporal::TemporalStatus::kOk,
          "seq write disabled output");
    check(!s.enable_tip_output(), "EnableTipOutput derived 0");
    golden("seq.tip_output_disabled", golden_bytes(writer.bit_size(),
                                                   writer.bytes()));
  }
  {
    // Refine-MV slice: enable_tip_refinemv only exists with opfl/refinemv.
    tip::TipSequenceSyntax s = base;
    s.enable_opfl_refine = temporal::RefineType::kAuto;
    s.enable_refinemv = true;
    s.enable_tip_refinemv = true;
    BitWriter writer;
    check(tip::write_tip_sequence_syntax(writer, s) ==
              temporal::TemporalStatus::kOk,
          "seq write refinemv");
    temporal::BitReader reader(writer.bytes().data(), writer.bytes().size());
    tip::TipSequenceSyntax read;
    check(tip::read_tip_sequence_syntax(reader, &read) ==
              temporal::TemporalStatus::kOk,
          "seq read refinemv");
    check(read.enable_tip_refinemv, "tip refinemv round trip");
    golden("seq.tip_refinemv", golden_bytes(writer.bit_size(),
                                            writer.bytes()));
  }
  {
    tip::TipSequenceSyntax bad = base;
    bad.enable_tip_refinemv = true;  // no opfl/refinemv -> bit absent
    BitWriter writer;
    check(tip::write_tip_sequence_syntax(writer, bad) ==
              temporal::TemporalStatus::kInvalidArgument,
          "seq rejects unreachable tip_refinemv");
    tip::TipSequenceSyntax bad2 = base;
    bad2.enable_tip = false;
    bad2.enable_tip_hole_fill = true;
    BitWriter writer2;
    check(tip::write_tip_sequence_syntax(writer2, bad2) ==
              temporal::TemporalStatus::kInvalidArgument,
          "seq rejects hole_fill without enable_tip");
  }
}

tip::TipFrameSyntax make_tip1_frame() {
  tip::TipFrameSyntax s;
  s.use_ref_frame_mvs = true;
  s.num_total_refs = 2;
  s.sb_64x64 = true;
  s.enable_tip = true;
  s.enable_tip_refinemv = false;
  s.enable_tip_output = true;
  s.enable_opfl_refine = temporal::RefineType::kNone;
  s.enable_refinemv = false;
  s.enable_tip_hole_fill = true;
  s.allow_tip_hole_fill = true;
  s.num_past_refs = 1;
  s.num_future_refs = 1;
  s.is_tip_frame = true;
  s.tip_global_wtd_index = 4;  // Tip_Weighting_Factor == 20
  s.tip_global_mv = temporal::Mv{8, 8};
  s.tip_sharp = true;
  return s;
}

void test_frame_syntax() {
  {
    const tip::TipFrameSyntax s = make_tip1_frame();
    const std::vector<std::uint8_t> payload =
        tip::make_tip_slice_bytes(s, nullptr);
    check(!payload.empty(), "tip1 payload");
    // Payload bits: parse it back.
    temporal::BitReader reader(payload.data(), payload.size());
    tip::TipFrameSyntax read = s;
    check(tip::read_tip_frame_syntax(reader, &read) ==
              temporal::TemporalStatus::kOk,
          "tip1 read");
    check(read.mode == temporal::TipFrameMode::kAsOutput, "tip1 AS_OUTPUT");
    check(read.tip_global_wtd_index == 4, "tip1 weight index");
    check(read.tip_global_mv == temporal::Mv(8, 8), "tip1 global mv");
    check(read.allow_tip_hole_fill, "tip1 hole fill bit");
    check(read.interp_filter == temporal::InterpFilter::kEighttapSharp,
          "tip1 sharp");
    // The last byte holds trailing_one_bit + zero padding.
    check(payload.back() != 0, "trailing bits present");
    golden("syntax.tip1_slice", hex(payload));
  }
  {
    // Weight 16 (index 2) => single-list store and no compound blend.
    tip::TipFrameSyntax s = make_tip1_frame();
    s.tip_global_wtd_index = 2;
    s.tip_global_mv = temporal::Mv{0, 0};
    s.tip_sharp = false;
    s.tip_regular = true;
    const std::vector<std::uint8_t> payload =
        tip::make_tip_slice_bytes(s, nullptr);
    temporal::BitReader reader(payload.data(), payload.size());
    tip::TipFrameSyntax read = s;
    check(tip::read_tip_frame_syntax(reader, &read) ==
              temporal::TemporalStatus::kOk,
          "tip1 read regular");
    check(read.tip_global_wtd_index == 2, "weight 16 index");
    check(read.interp_filter == temporal::InterpFilter::kEighttap,
          "EIGHTAP regular");
    golden("syntax.tip1_slice_regular", hex(payload));
  }
  {
    // TIP_FRAME_AS_REF: one bit for the mode, no global MV syntax.
    tip::TipFrameSyntax s = make_tip1_frame();
    s.is_tip_frame = false;
    s.requested_mode = temporal::TipFrameMode::kAsReference;
    BitWriter writer;
    check(tip::write_tip_frame_syntax(writer, s) ==
              temporal::TemporalStatus::kOk,
          "as-ref write");
    golden("frame.as_ref", golden_bytes(writer.bit_size(), writer.bytes()));
    temporal::BitReader reader(writer.bytes().data(), writer.bytes().size());
    tip::TipFrameSyntax read;
    read.use_ref_frame_mvs = true;
    read.num_total_refs = 2;
    read.sb_64x64 = true;
    read.enable_tip = true;
    read.enable_tip_output = true;
    read.enable_tip_hole_fill = true;
    read.num_past_refs = 1;
    read.num_future_refs = 1;
    read.tip_sharp = true;
    check(tip::read_tip_frame_syntax(reader, &read) ==
              temporal::TemporalStatus::kOk,
          "as-ref read");
    check(read.mode == temporal::TipFrameMode::kAsReference, "as-ref mode");
  }
  {
    // No TIP block (NumTotalRefs == 1): only frame_opfl_refine_type().
    tip::TipFrameSyntax s;
    s.use_ref_frame_mvs = true;
    s.num_total_refs = 1;
    s.enable_tip = true;
    s.enable_tip_output = true;
    s.enable_opfl_refine = temporal::RefineType::kNone;
    s.tip_sharp = true;
    BitWriter writer;
    check(tip::write_tip_frame_syntax(writer, s) ==
              temporal::TemporalStatus::kOk,
          "no tip block write");
    check_eq(static_cast<std::int64_t>(writer.bit_size()), 0,
             "no tip block writes no bits");
    golden("frame.no_tip_block", golden_bytes(writer.bit_size(),
                                              writer.bytes()));
  }
  {
    // A TIP OBU with EnableTipOutput == 0 is non-conforming.
    tip::TipFrameSyntax s = make_tip1_frame();
    s.enable_tip_output = false;
    const std::vector<std::uint8_t> payload =
        tip::make_tip_slice_bytes(s, nullptr);
    check(payload.empty(), "reject TIP OBU with output disabled");
    // Requesting AS_OUTPUT for a non-TIP OBU is rejected too.
    tip::TipFrameSyntax s2 = make_tip1_frame();
    s2.is_tip_frame = false;
    s2.requested_mode = temporal::TipFrameMode::kAsOutput;
    check(tip::make_tip_slice_bytes(s2, nullptr).empty(),
          "reject AS_OUTPUT on a non-TIP OBU");
  }
  {
    // §7.10.1 setup-TIP conformance.
    tip::TipFrameSyntax frame = make_tip1_frame();
    check(tip::resolve_tip_frame_syntax(&frame) ==
              temporal::TemporalStatus::kOk,
          "setup conditions resolve");
    tip::TipSetupConditions c = tip::make_tip_setup_conditions(
        false, frame, 0, 1, /*past_type=*/false, /*future_type=*/true,
        /*past_size=*/true, /*future_size=*/true);
    check(tip::check_tip_setup_conditions(c) ==
              temporal::TemporalStatus::kOk,
          "setup conditions: TIP output frame ok");
    c = tip::make_tip_setup_conditions(true, frame, 0, 1, false, true, true,
                                       true);
    check(tip::check_tip_setup_conditions(c) ==
              temporal::TemporalStatus::kInvalidArgument,
          "setup conditions: switch frame rejected");
    c = tip::make_tip_setup_conditions(false, frame, -1, 1, false, true, true,
                                       true);
    check(tip::check_tip_setup_conditions(c) ==
              temporal::TemporalStatus::kInvalidArgument,
          "setup conditions: missing ClosestPast rejected");
    c = tip::make_tip_setup_conditions(false, frame, 0, 1, false, false, true,
                                       true);
    check(tip::check_tip_setup_conditions(c) ==
              temporal::TemporalStatus::kInvalidArgument,
          "setup conditions: type eligibility required");
    c = tip::make_tip_setup_conditions(false, frame, 0, 1, false, true, true,
                                       false);
    check(tip::check_tip_setup_conditions(c) ==
              temporal::TemporalStatus::kInvalidArgument,
          "setup conditions: size eligibility required");
    const char* reason = nullptr;
    tip::check_tip_setup_conditions(tip::TipSetupConditions{}, &reason);
    check(reason != nullptr, "setup conditions reason");
  }
  {
    // OBU framing: leb128 + header byte (type 14) + payload.
    tip::TipFrameSyntax s = make_tip1_frame();
    BitWriter writer;
    check(tip::write_tip_frame_syntax(writer, s) ==
              temporal::TemporalStatus::kOk,
          "obu write");
    const std::vector<std::uint8_t> obu =
        tip::make_tip_frame_obu(tip::TipObuType::kRegularTip, &writer);
    check(obu.size() > 4, "obu size");
    // Annex B.2 leb128 prefix, then the OBU header byte: type 14 << 3 = 0x70.
    check(obu[1] == 0x38, "obu header type byte");
    check(tip::is_tip_frame_obu(static_cast<std::uint8_t>(14)),
          "type 14 is a TIP frame");
    check(tip::is_tip_frame_obu(static_cast<std::uint8_t>(13)),
          "type 13 is a TIP frame");
    check(!tip::is_tip_frame_obu(static_cast<std::uint8_t>(7)),
          "type 7 is not a TIP frame");
    golden("obu.regular_tip_obu", hex(obu));
    BitWriter leading_writer;
    check(tip::write_tip_frame_syntax(leading_writer, s) ==
              temporal::TemporalStatus::kOk,
          "leading obu write");
    const std::vector<std::uint8_t> leading = tip::make_tip_frame_obu(
        tip::TipObuType::kLeadingTip, &leading_writer);
    check(leading[1] == 0x34, "OBU_LEADING_TIP header byte");
    golden("obu.leading_tip_obu", hex(leading));
  }
}

// ---------------------------------------------------------------------------
// Candidate tests.
// ---------------------------------------------------------------------------

void test_candidates() {
  // A synthetic motion field used by get_tip_cand().
  FieldOwner field;
  field.init(16, 16);
  for (int y = 0; y < field.h8; ++y) {
    for (int x = 0; x < field.w8; ++x) {
      const std::size_t pos =
          static_cast<std::size_t>(y) * field.w8 + static_cast<std::size_t>(x);
      field.mv[pos * 2 + 0] = 48;  // 6 luma samples
      field.mv[pos * 2 + 1] = -16;
      field.valid[pos] = 1;
      field.offset[pos] = 4;
    }
  }
  temporal::MotionField mf = field.field();
  temporal::TipConfig cfg;
  cfg.mode = temporal::TipFrameMode::kAsOutput;
  cfg.tip_global_wtd_index = 2;
  cfg.num_past_refs = 1;
  cfg.num_future_refs = 1;
  cfg.order_hint = 1;
  cfg.closest_past_order_hint = 0;
  cfg.closest_future_order_hint = 4;
  cfg.interp_filter = temporal::InterpFilter::kEighttapSharp;

  tip::TipCandInput cand;
  cand.cand_row = 2;
  cand.cand_col = 2;
  cand.base_row = 2;
  cand.base_col = 2;
  cand.base_mv = temporal::Mv{1, 2};
  cand.field = &mf;

  // §7.22/§7.13.3.2 get_tip_cand: field projected to both lists + base MV.
  const temporal::MvPair pair = temporal::get_tip_cand(cand, cfg);
  check(pair.mv[0] == temporal::Mv(13, -2), "get_tip_cand list0");
  check(pair.mv[1] == temporal::Mv(-35, 14), "get_tip_cand list1");

  temporal::StackMv stack[temporal::kMaxRefMvStackSize]{};
  int num = 0;
  int prune = 0;
  check(tip::derive_ref_mv_candidate_from_tip_mode(cand, cfg, 2, stack, &num,
                                                   &prune) ==
            temporal::TemporalStatus::kOk,
        "derive ref mv candidate");
  check_eq(num, 1, "candidate count");
  check(stack[0].mv[0] == pair.mv[0] && stack[0].mv[1] == pair.mv[1],
        "candidate pair stored");
  check_eq(stack[0].weight, 2, "candidate weight");
  // A second identical candidate accumulates weight.
  check(tip::derive_ref_mv_candidate_from_tip_mode(cand, cfg, 3, stack, &num,
                                                   &prune) ==
            temporal::TemporalStatus::kOk,
        "derive ref mv candidate again");
  check_eq(num, 1, "candidate still one");
  check_eq(stack[0].weight, 5, "candidate weight accumulates");
  check_eq(prune, 1, "candidate prune count");
  golden("cand.ref_mv", std::to_string(num) + ":" + std::to_string(prune) +
                            ":" + std::to_string(stack[0].weight) + ":" +
                            std::to_string(stack[0].mv[0].row) + "," +
                            std::to_string(stack[0].mv[0].col) + "," +
                            std::to_string(stack[0].mv[1].row) + "," +
                            std::to_string(stack[0].mv[1].col));

  // §7.12.2.17 uses only the selected list and only RefStackMv[][0].
  temporal::StackMv single[temporal::kMaxRefMvStackSize]{};
  int single_num = 0;
  int single_prune = 0;
  check(tip::derive_single_ref_mv_candidate_from_tip_mode(
            cand, cfg, 1, 4, single, &single_num, &single_prune) ==
            temporal::TemporalStatus::kOk,
        "derive single ref candidate");
  check_eq(single_num, 1, "single candidate count");
  check(single[0].mv[0] == pair.mv[1], "single candidate list1 mv");
  check_eq(single[0].weight, 4, "single candidate weight");
  check(single_num == 1 &&
            tip::derive_single_ref_mv_candidate_from_tip_mode(
                cand, cfg, 1, 4, single, &single_num, &single_prune) ==
                temporal::TemporalStatus::kOk &&
            single[0].weight == 8,
        "single candidate accumulates weight");
  golden("cand.single_ref", std::to_string(single_num) + ":" +
                                std::to_string(single[0].weight));

  // §7.12.2.18: linearMv = subMvs[0] - subMvs[1], projected by pastOffset.
  tip::DerivedMvStack derived;
  const temporal::Mv sub[2] = {temporal::Mv(40, -16), temporal::Mv(-24, 8)};
  check(tip::tip_add_derived(sub, cfg, &derived) ==
            temporal::TemporalStatus::kOk,
        "tip add derived");
  check_eq(derived.count, 1, "derived count");
  const temporal::Mv want{
      temporal::clip3(temporal::kMvLow + 1, temporal::kMvUpp - 1,
                      40 - temporal::get_mv_projection(
                               temporal::Mv(64, -24), 1, 4)
                               .row),
      temporal::clip3(temporal::kMvLow + 1, temporal::kMvUpp - 1,
                      -16 - temporal::get_mv_projection(
                                temporal::Mv(64, -24), 1, 4)
                                .col)};
  check(derived.mv[0] == want, "derived mv value");
  check(tip::tip_add_derived(sub, cfg, &derived) ==
            temporal::TemporalStatus::kOk &&
            derived.count == 1 && derived.prune == 1,
        "derived prune duplicate");
  golden("cand.tip_add_derived",
         std::to_string(derived.count) + ":" + std::to_string(derived.prune) +
             ":" + std::to_string(derived.mv[0].row) + "," +
             std::to_string(derived.mv[0].col));
}

// ---------------------------------------------------------------------------
// Build tests.
// ---------------------------------------------------------------------------

struct BuildFixture {
  FrameOwner key;
  FrameOwner future;
  FrameOwner recon;
  FieldOwner field;
  PlanOwner plan;
  MvFieldOwner future_field;
  std::vector<std::int16_t> scratch;
  std::vector<std::uint8_t> block;

  void init(int width, int height) {
    key.init(width, height, 0);
    future.init(width, height, 7);
    recon.init(width, height, 0);
    const int mi_cols = 2 * ((width + 7) / 8);
    const int mi_rows = 2 * ((height + 7) / 8);
    field.init(mi_cols, mi_rows);
    plan.init(mi_cols, mi_rows);
    future_field.init(mi_cols >> 1, mi_rows >> 1);
    scratch.assign(2 * 16 * 16, 0);
    block.assign(16 * 16, 0);
  }

  temporal::MotionFieldConfig config(int order_hint) {
    temporal::MotionFieldConfig cfg;
    cfg.mi_cols = plan.mi_cols;
    cfg.mi_rows = plan.mi_rows;
    cfg.proj_step = 1;
    cfg.sb_64x64 = true;
    cfg.enable_tip = true;
    cfg.order_hint = order_hint;
    cfg.num_total_refs = 2;
    cfg.num_past_refs = 1;
    cfg.num_future_refs = 1;
    cfg.frame_width = key.width;
    cfg.frame_height = key.height;
    cfg.ref_order_hint[0] = 0;
    cfg.ref_order_hint[1] = 4;
    temporal::RefFrameInfo ref0;
    ref0.slot = 0;
    ref0.type = temporal::FrameType::kKey;
    ref0.width = key.width;
    ref0.height = key.height;
    ref0.order_hint = 0;
    ref0.num_saved_refs = 0;
    temporal::RefFrameInfo ref1;
    ref1.slot = 1;
    ref1.type = temporal::FrameType::kInter;
    ref1.width = future.width;
    ref1.height = future.height;
    ref1.order_hint = 4;
    ref1.num_saved_refs = 1;
    ref1.saved_order_hint[0] = 0;
    cfg.refs[0] = ref0;
    cfg.refs[1] = ref1;
    cfg.saved[0] = temporal::SavedMvField{};
    cfg.saved[1] = future_field.saved();
    return cfg;
  }

  temporal::MotionField field_view() { return field.field(); }
  temporal::MotionFieldScratch scratch_view() { return field.scratch(); }
  temporal::TipPlan plan_view() { return plan.plan(); }

  tip::TipBuildInput input(const temporal::MotionFieldConfig& cfg,
                           const temporal::TipConfig& tip) {
    tip::TipBuildInput in;
    in.config = cfg;
    in.tip = tip;
    first_ = field.field();
    scratch_ = field.scratch();
    plan_ = plan.plan();
    in.field = &first_;
    in.scratch = &scratch_;
    in.plan = &plan_;
    in.ref[0][0] = key.plane[0].view();
    in.ref[0][1] = future.plane[0].view();
    in.ref[1][0] = key.plane[1].view();
    in.ref[1][1] = future.plane[1].view();
    in.ref[2][0] = key.plane[2].view();
    in.ref[2][1] = future.plane[2].view();
    for (int p = 0; p < 3; ++p) {
      in.dst[p] = recon.plane[p].data.data();
      in.dst_stride[p] = recon.plane[p].width;
      in.dst_width[p] = recon.plane[p].width;
      in.dst_height[p] = recon.plane[p].height;
      in.sub_x[p] = p == 0 ? 0 : 1;
      in.sub_y[p] = p == 0 ? 0 : 1;
    }
    in.pred_scratch = scratch.data();
    in.block_scratch = block.data();
    return in;
  }

 private:
  temporal::MotionField first_;
  temporal::MotionFieldScratch scratch_;
  temporal::TipPlan plan_;
};

temporal::TipConfig make_tip_config(int weight_index, temporal::Mv global_mv) {
  temporal::TipConfig tip;
  tip.mode = temporal::TipFrameMode::kAsOutput;
  tip.tip_global_wtd_index = weight_index;
  tip.tip_global_mv = global_mv;
  tip.interp_filter = temporal::InterpFilter::kEighttapSharp;
  tip.enable_tip_refinemv = false;
  tip.enable_refinemv = false;
  tip.opfl_refine_type = temporal::RefineType::kNone;
  tip.allow_tip_hole_fill = true;
  tip.num_past_refs = 1;
  tip.num_future_refs = 1;
  tip.order_hint = 1;
  tip.closest_past_order_hint = 0;
  tip.closest_future_order_hint = 4;
  return tip;
}

void run_build(BuildFixture* fixture, int width, int height, int weight_index,
               temporal::Mv global_mv, const std::string& name) {
  fixture->init(width, height);
  fixture->future_field.fill_constant(temporal::Mv{0, 0},
                                      temporal::Mv{0, 0});
  auto cfg = fixture->config(1);
  const temporal::TipConfig tip = make_tip_config(weight_index, global_mv);
  tip::TipBuildInput in = fixture->input(cfg, tip);
  // Fix up the pointers to the fixture-owned objects.
  temporal::MotionField field_view = fixture->field.field();
  in.field = &field_view;
  temporal::MotionFieldScratch scratch_view = fixture->field.scratch();
  in.scratch = &scratch_view;
  temporal::TipPlan plan_view = fixture->plan.plan();
  in.plan = &plan_view;

  tip::TipBuildOutput out;
  const temporal::TemporalStatus status = tip::build_tip_frame(in, &out);
  check(status == temporal::TemporalStatus::kOk, name + " build status");
  check_eq(out.num_blocks, ((fixture->plan.mi_cols + 3) / 4) *
                               ((fixture->plan.mi_rows + 3) / 4),
           name + " block count");
  check_eq(out.field_valid_after_estimation,
           (fixture->plan.mi_cols / 2) * (fixture->plan.mi_rows / 2),
           name + " field valid");
  check_eq(out.field_invalid, 0, name + " no invalid entries");
  // Every 4x4 of every plan block carries the block MV.
  const temporal::Mv want0 = global_mv;
  const temporal::Mv want1 = global_mv;
  bool grid_ok = true;
  for (int r = 0; r < fixture->plan.mi_rows; ++r) {
    for (int c = 0; c < fixture->plan.mi_cols; ++c) {
      const std::size_t idx =
          static_cast<std::size_t>(r) * fixture->plan.mi_cols + c;
      if (fixture->plan.grid0[idx] != want0 ||
          fixture->plan.grid1[idx] != want1) {
        grid_ok = false;
      }
    }
  }
  check(grid_ok, name + " mv grid");
  check_eq(out.predicted_samples[0], width * height, name + " luma samples");
  golden(name, std::to_string(out.num_blocks) + ":" +
                   std::to_string(out.field_valid_after_estimation) + ":" +
                   std::to_string(out.hole_filled) + ":" +
                   std::to_string(out.predicted_blocks) + ":" +
                   hash_planes(fixture->recon.plane[0].data,
                               fixture->recon.plane[1].data,
                               fixture->recon.plane[2].data));
}

void test_build_uniform() {
  BuildFixture fixture;
  run_build(&fixture, 64, 64, 4, temporal::Mv{8, 8}, "build.tip1_64x64");
  run_build(&fixture, 640, 360, 4, temporal::Mv{8, 8},
            "build.tip1_640x360");
  run_build(&fixture, 64, 64, 2, temporal::Mv{0, 0},
            "build.tip_weight16_64x64");

  // A weight-16 build must equal a plain single-reference prediction with the
  // same MV and EIGHTAP_SHARP filter (the fromBuildTip CwpIdx==16 rule).
  BuildFixture w16;
  w16.init(64, 64);
  w16.future_field.fill_constant(temporal::Mv{0, 0}, temporal::Mv{0, 0});
  auto cfg = w16.config(1);
  const temporal::TipConfig tip = make_tip_config(2, temporal::Mv{5, -3});
  tip::TipBuildInput in = w16.input(cfg, tip);
  temporal::MotionField field_view = w16.field.field();
  in.field = &field_view;
  temporal::MotionFieldScratch scratch_view = w16.field.scratch();
  in.scratch = &scratch_view;
  temporal::TipPlan plan_view = w16.plan.plan();
  in.plan = &plan_view;
  tip::TipBuildOutput out;
  check(tip::build_tip_frame(in, &out) == temporal::TemporalStatus::kOk,
        "weight16 build");
  std::vector<std::uint8_t> reference(
      static_cast<std::size_t>(64) * 64, 0);
  check(temporal::predict_plane_u8(
            w16.key.plane[0].view(), 0, 0, 64, 64, temporal::Mv{5, -3},
            temporal::InterpFilter::kEighttapSharp, 0, 0, false, 64, 64, 64,
            64, temporal::RefArea{0, 0, 63, 63}, reference.data(), 64),
        "weight16 reference prediction");
  check(std::memcmp(reference.data(), w16.recon.plane[0].data.data(), 64 * 64) ==
            0,
        "weight16 equals single prediction");
}

void test_build_chained() {
  BuildFixture fixture;
  fixture.init(64, 64);
  fixture.future_field.fill_constant(temporal::Mv{0, 0}, temporal::Mv{0, 0});
  auto cfg1 = fixture.config(1);
  const temporal::TipConfig tip1 = make_tip_config(4, temporal::Mv{8, 8});
  tip::TipBuildInput in1 = fixture.input(cfg1, tip1);
  temporal::MotionField field_view = fixture.field.field();
  in1.field = &field_view;
  temporal::MotionFieldScratch scratch_view = fixture.field.scratch();
  in1.scratch = &scratch_view;
  temporal::TipPlan plan_view = fixture.plan.plan();
  in1.plan = &plan_view;
  tip::TipBuildOutput out1;
  check(tip::build_tip_frame(in1, &out1) == temporal::TemporalStatus::kOk,
        "chain tip1 build");

  // §7.22 store of the TIP1 frame.
  MvFieldOwner tip1_field;
  tip1_field.init(8, 8);
  temporal::RefMvField stored = tip1_field.view();
  tip::TipStoreInput store;
  store.mi_cols = 16;
  store.mi_rows = 16;
  store.order_hint = 1;
  store.closest_past = 0;
  store.closest_future = 1;
  store.closest_past_order_hint = 0;
  store.closest_future_order_hint = 4;
  store.weight16 = false;
  check(tip::store_tip_frame_motion_field(plan_view, store, &stored) ==
            temporal::TemporalStatus::kOk,
        "chain tip1 store");
  check_eq(tip1_field.ref[0], 0, "tip1 stored ref0");
  check_eq(tip1_field.ref[1], 1, "tip1 stored ref1");
  check_eq(tip1_field.mv[0], temporal::compression_mv(8),
           "tip1 stored mv list0 row");
  check_eq(tip1_field.mv[2], temporal::compression_mv(8),
           "tip1 stored mv list1 row");

  // TIP2: past = TIP1 (hint 1, saved hints {0,4}), future = P (hint 4).
  BuildFixture tip2;
  tip2.init(64, 64);
  tip2.future_field = fixture.future_field;  // zero field, ref 0
  auto cfg2 = tip2.config(2);
  cfg2.refs[0].slot = 2;
  cfg2.refs[0].type = temporal::FrameType::kInter;
  cfg2.refs[0].order_hint = 1;
  cfg2.refs[0].num_saved_refs = 2;
  cfg2.refs[0].saved_order_hint[0] = 0;
  cfg2.refs[0].saved_order_hint[1] = 4;
  cfg2.ref_order_hint[0] = 1;
  cfg2.ref_order_hint[1] = 4;
  cfg2.saved[0] = tip1_field.saved();
  // Reference pixels: TIP1's output is the past reference.
  tip2.key = fixture.recon;  // reuse the TIP1 reconstruction
  tip2.future = fixture.future;
  temporal::TipConfig tip2_cfg = make_tip_config(3, temporal::Mv{0, 0});
  tip2_cfg.order_hint = 2;
  tip2_cfg.closest_past_order_hint = 1;
  tip2_cfg.closest_future_order_hint = 4;
  tip::TipBuildInput in2 = tip2.input(cfg2, tip2_cfg);
  temporal::MotionField field2 = tip2.field.field();
  in2.field = &field2;
  temporal::MotionFieldScratch scratch2 = tip2.field.scratch();
  in2.scratch = &scratch2;
  temporal::TipPlan plan2 = tip2.plan.plan();
  in2.plan = &plan2;
  tip::TipBuildOutput out2;
  check(tip::build_tip_frame(in2, &out2) == temporal::TemporalStatus::kOk,
        "chain tip2 build");
  // Uniform field: the projected TIP1 MV is negated and rescaled.
  check_eq(out2.field_result.closest_past, 0, "tip2 closest past");
  check_eq(out2.field_result.closest_future, 1, "tip2 closest future");
  bool uniform = true;
  for (int i = 0; i < field2.w8 * field2.h8; ++i) {
    if (field2.mv[i * 2 + 0] != -8 || field2.mv[i * 2 + 1] != -8 ||
        field2.valid[i] == 0) {
      uniform = false;
    }
  }
  check(uniform, "tip2 field is uniform -8");
  check(plan2.blocks[0].mv[0] == temporal::Mv(-3, -3),
        "tip2 derived list0 mv");
  check(plan2.blocks[0].mv[1] == temporal::Mv(5, 5),
        "tip2 derived list1 mv");
  golden("build.tip2_chained",
         std::to_string(out2.num_blocks) + ":" +
             std::to_string(out2.field_valid_after_estimation) + ":" +
             std::to_string(out2.field_valid_after_setup) + ":" +
             std::to_string(plan2.blocks[0].mv[0].row) + "," +
             std::to_string(plan2.blocks[0].mv[0].col) + "," +
             std::to_string(plan2.blocks[0].mv[1].row) + "," +
             std::to_string(plan2.blocks[0].mv[1].col) + ":" +
             hash_planes(tip2.recon.plane[0].data, tip2.recon.plane[1].data,
                         tip2.recon.plane[2].data));
}

void test_build_hole_fill() {
  // 128x128, four 64x64 TMVP units.  A 12-pixel MV shifts every projected
  // 8x8 by one TMVP cell inside its unit, leaving one-row/column holes.
  BuildFixture fixture;
  fixture.init(128, 128);
  fixture.future_field.fill_constant(temporal::Mv{96, 96},
                                     temporal::Mv{96, 96});
  auto cfg = fixture.config(1);
  const temporal::TipConfig tip = make_tip_config(4, temporal::Mv{0, 0});
  tip::TipBuildInput in = fixture.input(cfg, tip);
  temporal::MotionField field_view = fixture.field.field();
  in.field = &field_view;
  temporal::MotionFieldScratch scratch_view = fixture.field.scratch();
  in.scratch = &scratch_view;
  temporal::TipPlan plan_view = fixture.plan.plan();
  in.plan = &plan_view;
  tip::TipBuildOutput out;
  check(tip::build_tip_frame(in, &out) == temporal::TemporalStatus::kOk,
        "hole fill build");
  check(out.hole_filled > 0, "hole fill actually filled entries");
  check_eq(out.field_invalid, 0, "hole fill leaves no invalid entries");
  golden("build.hole_fill",
         std::to_string(out.field_valid_after_estimation) + ":" +
             std::to_string(out.field_valid_after_setup) + ":" +
             std::to_string(out.hole_filled) + ":" +
             hash_planes(fixture.recon.plane[0].data,
                         fixture.recon.plane[1].data,
                         fixture.recon.plane[2].data));
}

void test_store_weight16() {
  BuildFixture fixture;
  fixture.init(64, 64);
  fixture.future_field.fill_constant(temporal::Mv{0, 0}, temporal::Mv{0, 0});
  auto cfg = fixture.config(1);
  const temporal::TipConfig tip = make_tip_config(2, temporal::Mv{3, 4});
  tip::TipBuildInput in = fixture.input(cfg, tip);
  temporal::MotionField field_view = fixture.field.field();
  in.field = &field_view;
  temporal::MotionFieldScratch scratch_view = fixture.field.scratch();
  in.scratch = &scratch_view;
  temporal::TipPlan plan_view = fixture.plan.plan();
  in.plan = &plan_view;
  tip::TipBuildOutput out;
  check(tip::build_tip_frame(in, &out) == temporal::TemporalStatus::kOk,
        "weight16 store build");
  MvFieldOwner stored_field;
  stored_field.init(8, 8);
  temporal::RefMvField stored = stored_field.view();
  tip::TipStoreInput store;
  store.mi_cols = 16;
  store.mi_rows = 16;
  store.order_hint = 1;
  store.closest_past = 0;
  store.closest_future = 1;
  store.closest_past_order_hint = 0;
  store.closest_future_order_hint = 4;
  store.weight16 = true;
  check(tip::store_tip_frame_motion_field(plan_view, store, &stored) ==
            temporal::TemporalStatus::kOk,
        "weight16 store");
  // §7.22 mirrors the single list into both, and compression is lossy.
  check_eq(stored_field.ref[0], 0, "weight16 ref0");
  check_eq(stored_field.ref[1], 0, "weight16 mirrored ref1");
  check_eq(stored_field.mv[0], stored_field.mv[2], "weight16 mirrored mv");
  check_eq(temporal::uncompression_mv(stored_field.mv[0]),
           temporal::uncompression_mv(temporal::compression_mv(3)),
           "weight16 compressed row");
  golden("store.weight16",
         std::to_string(stored_field.ref[0]) + ":" +
             std::to_string(stored_field.ref[1]) + ":" +
             std::to_string(stored_field.mv[0]) + ":" +
             std::to_string(stored_field.mv[1]));

  // 640x360: MiRows == 90 is not a multiple of 4, so the last 16x16 TIP block
  // is clamped to two MI rows.  ASan catches any out-of-bounds store.
  BuildFixture tall;
  tall.init(640, 360);
  tall.future_field.fill_constant(temporal::Mv{0, 0}, temporal::Mv{0, 0});
  auto tall_cfg = tall.config(1);
  const temporal::TipConfig tall_tip =
      make_tip_config(4, temporal::Mv{8, 8});
  tip::TipBuildInput tall_in = tall.input(tall_cfg, tall_tip);
  temporal::MotionField tall_field = tall.field.field();
  tall_in.field = &tall_field;
  temporal::MotionFieldScratch tall_scratch = tall.field.scratch();
  tall_in.scratch = &tall_scratch;
  temporal::TipPlan tall_plan = tall.plan.plan();
  tall_in.plan = &tall_plan;
  tip::TipBuildOutput tall_out;
  check(tip::build_tip_frame(tall_in, &tall_out) ==
            temporal::TemporalStatus::kOk,
        "640x360 build");
  MvFieldOwner tall_field_out;
  tall_field_out.init(80, 45);
  temporal::RefMvField tall_stored = tall_field_out.view();
  tip::TipStoreInput tall_store;
  tall_store.mi_cols = 160;
  tall_store.mi_rows = 90;
  tall_store.order_hint = 1;
  tall_store.closest_past = 0;
  tall_store.closest_future = 1;
  tall_store.closest_past_order_hint = 0;
  tall_store.closest_future_order_hint = 4;
  check(tip::store_tip_frame_motion_field(tall_plan, tall_store, &tall_stored) ==
            temporal::TemporalStatus::kOk,
        "640x360 store");
  // Row 44 (y8) is the last 8x8 row; block row 88 covers MI rows 88..91 but
  // only 88..89 exist, so y8 = 45 would be out of range.
  check_eq(tall_field_out.ref[static_cast<std::size_t>(44) * 80 * 2], 0,
           "640x360 last y8 written");
  bool edge_ok = true;
  for (int list = 0; list < 2; ++list) {
    const std::size_t idx =
        (static_cast<std::size_t>(44) * 80 + 79) * 2 + list;
    if (tall_field_out.ref[idx] != static_cast<std::int8_t>(list)) {
      edge_ok = false;
    }
  }
  check(edge_ok, "640x360 right/bottom edge stored");
}

// ---------------------------------------------------------------------------
// Output rules.
// ---------------------------------------------------------------------------

void test_output_rules() {
  tip::TipOutputConfig base;
  base.enable_tip = true;
  base.disable_tip_output = false;
  base.is_tip_frame = true;
  base.mode = temporal::TipFrameMode::kAsOutput;
  base.immediate_output = true;
  base.refreshes_slot = true;
  base.has_closest_past = true;
  base.has_closest_future = true;

  tip::TipOutputPlan plan;
  check(tip::plan_tip_output(base, &plan) == temporal::TemporalStatus::kOk &&
            plan.enable_tip_output && plan.tip_obu_allowed && plan.output_now &&
            !plan.non_output_tip,
        "output: TIP frame displayed");

  tip::TipOutputConfig t = base;
  t.immediate_output = false;
  check(tip::plan_tip_output(t, &plan) == temporal::TemporalStatus::kOk &&
            plan.tip_obu_allowed && plan.non_output_tip && !plan.output_now,
        "output: non-output TIP is a reference frame");

  t = base;
  t.immediate_output = false;
  t.refreshes_slot = false;
  check(tip::plan_tip_output(t, &plan) ==
            temporal::TemporalStatus::kInvalidArgument,
        "output: non-output TIP must refresh");

  t = base;
  t.disable_tip_output = true;
  check(tip::plan_tip_output(t, &plan) ==
            temporal::TemporalStatus::kInvalidArgument,
        "output: disable_tip_output forbids a TIP OBU");

  t = base;
  t.enable_tip = false;
  check(tip::plan_tip_output(t, &plan) ==
            temporal::TemporalStatus::kInvalidArgument &&
            !plan.enable_tip_output,
        "output: enable_tip=0 forbids a TIP OBU");

  t = base;
  t.has_closest_future = false;
  check(tip::plan_tip_output(t, &plan) ==
            temporal::TemporalStatus::kInvalidArgument,
        "output: TIP needs both closest refs");

  t = base;
  t.is_tip_frame = false;
  t.mode = temporal::TipFrameMode::kAsReference;
  t.immediate_output = false;
  check(tip::plan_tip_output(t, &plan) == temporal::TemporalStatus::kOk &&
            !plan.tip_obu_allowed && !plan.output_now,
        "output: regular non-output frame");

  t = base;
  t.is_tip_frame = false;
  t.mode = temporal::TipFrameMode::kAsOutput;
  check(tip::plan_tip_output(t, &plan) ==
            temporal::TemporalStatus::kInvalidArgument,
        "output: AS_OUTPUT needs a TIP OBU");

  check(tip::tip_implicit_output_inferred(true) == false,
        "monotonic makes implicit output inferred 0");
  check(tip::tip_implicit_output_inferred(false) == true,
        "non-monotonic can signal implicit output");

  golden("output.rules", "ok");
}

// ---------------------------------------------------------------------------
// Golden comparison.
// ---------------------------------------------------------------------------

bool load_goldens(const std::string& path, std::map<std::string, std::string>* out) {
  std::ifstream file(path);
  if (!file) return false;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    const std::size_t eq = line.find(" = ");
    if (eq == std::string::npos) continue;
    (*out)[line.substr(0, eq)] = line.substr(eq + 3);
  }
  return true;
}

void compare_goldens(const std::map<std::string, std::string>& expected) {
  for (const auto& [name, value] : g_goldens) {
    const auto it = expected.find(name);
    if (it == expected.end()) {
      ++g_failures;
      std::fprintf(stderr, "FAIL: golden missing: %s\n", name.c_str());
      continue;
    }
    ++g_checks;
    if (it->second != value) {
      ++g_failures;
      std::fprintf(stderr, "FAIL: golden %s\n  got  %s\n  want %s\n",
                   name.c_str(), value.c_str(), it->second.c_str());
    }
  }
  for (const auto& [name, value] : expected) {
    (void)value;
    ++g_checks;
    if (g_goldens.find(name) == g_goldens.end()) {
      ++g_failures;
      std::fprintf(stderr, "FAIL: unused golden: %s\n", name.c_str());
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string golden_path;
  bool print_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--goldens" && i + 1 < argc) {
      golden_path = argv[++i];
    } else if (arg == "--print-goldens") {
      print_only = true;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  test_sequence_syntax();
  test_frame_syntax();
  test_candidates();
  test_build_uniform();
  test_build_chained();
  test_build_hole_fill();
  test_store_weight16();
  test_output_rules();

  if (!print_only && !golden_path.empty()) {
    std::map<std::string, std::string> expected;
    if (!load_goldens(golden_path, &expected)) {
      std::fprintf(stderr, "cannot read goldens: %s\n", golden_path.c_str());
      return 2;
    }
    compare_goldens(expected);
  }

  std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
