// Unit tests for the long-term-reference / open-random-access layer.
//
//   * exact bit goldens for the CLK / OLK / P / leading frame headers;
//   * writer/reader round-trips for every role and edge value;
//   * byte equality with the untouched M2 header writer when
//     long_term_frame_id_bits == 0 (the two writers cannot drift);
//   * §5.18.2/§7.23 pool behaviour: refresh, the KEY_FRAME "first" rule, the
//     first-Regular-frame invalidation and long-term retention;
//   * §7.4.3/§7.4.4 conformance checks and the rejected cases;
//   * the two planners.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "codec/inter/longterm/longterm.h"
#include "codec/inter/longterm/longterm_stream.h"
#include "pipeline/video/video_headers.h"

namespace {

using namespace av2enc::inter::longterm;

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

std::string hex(std::span<const std::uint8_t> bytes) {
  std::string out;
  static const char* kDigits = "0123456789abcdef";
  for (const std::uint8_t byte : bytes) {
    out.push_back(kDigits[byte >> 4U]);
    out.push_back(kDigits[byte & 0x0FU]);
  }
  return out;
}

SequenceConfig golden_config() {
  SequenceConfig config;
  config.max_frame_width = 64;
  config.max_frame_height = 64;
  config.base_q_idx = 90;
  config.order_hint_bits = 4;
  config.num_ref_frames = 4;
  config.long_term_frame_id_bits = 2;
  return config;
}

SequenceConfig video_equivalent_config() {
  SequenceConfig config;
  config.max_frame_width = 64;
  config.max_frame_height = 64;
  config.base_q_idx = 90;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.long_term_frame_id_bits = 0;
  return config;
}

// Byte sequence of only the frame_header_info() bits (no leading
// is_first_tile_group, no alignment) as written by write_frame_header().
std::vector<std::uint8_t> header_bits_only(const SequenceConfig& config,
                                           const FrameHeader& header) {
  av2enc::BitWriter writer;
  write_frame_header(writer, config, header);
  return writer.take_bytes();
}


// A pool that has decoded one CLK (all slots refreshed, only the first valid).
PoolState pool_after_clk(const SequenceConfig& config, std::int32_t lt_id,
                         std::uint32_t order_hint = 0) {
  PoolState pool;
  reset_pool(config, &pool);
  FrameHeader clk{};
  std::string error;
  if (!plan_closed_loop_key(config, order_hint, lt_id, &clk, &error)) {
    std::printf("FAIL: pool_after_clk: %s\n", error.c_str());
    return pool;
  }
  PoolEvents events;
  if (!apply_frame(config, clk, true, &pool, &events, &error)) {
    std::printf("FAIL: pool_after_clk apply: %s\n", error.c_str());
  }
  return pool;
}

// ---------------------------------------------------------------------------
// Bit goldens
// ---------------------------------------------------------------------------
void test_goldens() {
  const SequenceConfig config = golden_config();

  // CLK, order_hint 0, long_term_id 0. 28 bits:
  //   uvlc(0) uvlc(0) = 11
  //   long_term_id_plus_1 f(2) = 01
  //   immediate_output_frame f(1) = 1
  //   frame_size_override_flag f(1) = 0
  //   order_hint f(4) = 0000
  //   allow_intrabc f(1) = 0
  //   disable_cdf_update f(1) = 1
  //   tile_info: uniform_tile_spacing_flag f(1) = 1
  //   base_q_idx f(8) = 01011010 (90)
  //   segmentation/using_qmatrix/delta_q/deblock[0]/deblock[1] = 00000
  //   tx_mode_select f(1) = 0
  //   reduced_tx_set f(2) = 00
  FrameHeader clk{};
  check(plan_closed_loop_key(config, /*order_hint=*/0, /*long_term_id=*/0, &clk,
                             nullptr),
        "golden CLK plan");
  check(hex(header_bits_only(config, clk)) == "d81ad000",
        "golden CLK header bits == d81ad000, got " +
            hex(header_bits_only(config, clk)));

  // The same header inside tile_group_obu(): is_first_tile_group f(1) = 1 and
  // byte_alignment() -> 29 bits padded to 4 bytes.
  {
    av2enc::BitWriter writer;
    write_tile_group_header(writer, config, clk);
    const std::vector<std::uint8_t> bytes = writer.take_bytes();
    check(hex(bytes) == "ec0d6800",
          "golden CLK tile-group header == ec0d6800, got " + hex(bytes));
  }

  // OLK, order_hint 1, long_term_id 1, ref_long_term_id = {0}, refresh slot 1.
  //   uvlc(0) uvlc(0) = 11
  //   long_term_id_plus_1 f(2) = 10
  //   num_key_ref_frames f(3) = 001
  //   ref_long_term_id[0] f(2) = 00
  //   (no immediate_output_frame bit: inferred 0 for OLK)
  //   frame_size_override_flag f(1) = 0
  //   order_hint f(4) = 0001
  //   refresh_frame_flags f(4) = 0010
  //   allow_intrabc f(1) = 0
  //   disable_cdf_update f(1) = 1
  //   tile_info f(1) = 1
  //   base_q_idx f(8) = 01011010
  //   00000 0 00
  const PoolState pool = pool_after_clk(config, /*lt_id=*/0);
  FrameHeader olk{};
  std::string olk_plan_error;
  check(plan_open_loop_key(config, pool, /*order_hint=*/1, /*olk_slot=*/1,
                           /*olk_long_term_id=*/1,
                           std::vector<std::int32_t>{0}, &olk,
                           &olk_plan_error),
        "golden OLK plan: " + olk_plan_error);
  const std::string olk_hex = hex(header_bits_only(config, olk));
  // 1+1+2+3+2+1+4+4+1+1+1+8+5+1+2 = 37 bits.
  check(olk_hex == "e2049ad000",
        "golden OLK header bits, got " + olk_hex);

  // P frame, order_hint 2, ref_frame_idx = {1}, refresh slot 2.
  FrameHeader p{};
  check(plan_inter_frame(config, FrameRole::kRegular, /*order_hint=*/2,
                         std::vector<std::uint8_t>{1}, /*refresh_slot=*/2,
                         /*immediate=*/true, &p, nullptr),
        "golden P plan");
  const std::string p_hex = hex(header_bits_only(config, p));
  // uvlc uvlc frame_is_inter=1 imm=1 fso=0 oh=0010 signal_primary=0
  // disable_cross=1 refresh=0100 map=1 nrefs=001 idx=01 intrabc=0 ...
  check(p_hex == "f129291ad000",
        "golden P header bits, got " + p_hex);

  // Leading frame, same layout but is_regular_role() is false.
  FrameHeader leading{};
  check(plan_inter_frame(config, FrameRole::kLeading, /*order_hint=*/3,
                         std::vector<std::uint8_t>{0}, /*refresh_slot=*/0,
                         /*immediate=*/false, &leading, nullptr),
        "golden leading plan");
  const std::string lead_hex = hex(header_bits_only(config, leading));
  check(lead_hex == "e1a3211ad000",
        "golden leading header bits, got " + lead_hex);
}

// ---------------------------------------------------------------------------
// M2 equivalence
// ---------------------------------------------------------------------------
void test_video_equivalence() {
  const SequenceConfig config = video_equivalent_config();

  // Sequence header.
  av2enc::video::VideoConfig m2;
  m2.max_frame_width = config.max_frame_width;
  m2.max_frame_height = config.max_frame_height;
  m2.seq_level_idx = config.seq_level_idx;
  m2.base_q_idx = config.base_q_idx;
  m2.order_hint_bits = config.order_hint_bits;
  m2.num_ref_frames = config.num_ref_frames;
  m2.num_total_refs = 1;
  m2.max_drl_bits_minus_1 = config.max_drl_bits_minus_1;
  m2.disable_cdf_update = config.disable_cdf_update;
  m2.enable_cfl_intra = config.enable_cfl_intra;
  const std::vector<std::uint8_t> ours_seq = make_sequence_header_obu(config);
  const std::vector<std::uint8_t> video_seq =
      av2enc::video::make_sequence_header_obu(m2);
  check(ours_seq == video_seq,
        "sequence header byte-identical to M2 (ours " + hex(ours_seq) +
            " vs " + hex(video_seq) + ")");

  // CLK frame with a dummy tile payload.
  const std::vector<std::uint8_t> tile = {0xDE, 0xAD, 0xBE, 0xEF};
  av2enc::video::VideoFrameHeader video_key;
  video_key.keyframe = true;
  video_key.order_hint = 0;
  video_key.ref_slot = 0;
  video_key.refresh_frame_flags = (1U << config.num_ref_frames) - 1U;
  FrameHeader clk{};
  check(plan_closed_loop_key(config, 0, -1, &clk, nullptr), "equiv CLK plan");
  check(make_frame_obu(config, clk, tile) ==
            av2enc::video::make_frame_obu(m2, video_key, tile),
        "CLK OBU byte-identical to M2");

  // P frame.
  av2enc::video::VideoFrameHeader video_p;
  video_p.keyframe = false;
  video_p.order_hint = 1;
  video_p.ref_slot = 0;
  video_p.refresh_frame_flags = 1;
  FrameHeader p{};
  check(plan_inter_frame(config, FrameRole::kRegular, 1,
                         std::vector<std::uint8_t>{0}, 0, true, &p, nullptr),
        "equiv P plan");
  check(make_frame_obu(config, p, tile) ==
            av2enc::video::make_frame_obu(m2, video_p, tile),
        "P OBU byte-identical to M2");

  // Temporal delimiter.
  check(make_temporal_delimiter_obu() ==
            av2enc::video::make_temporal_delimiter_obu(),
        "temporal delimiter byte-identical to M2");
}

// ---------------------------------------------------------------------------
// Writer / reader round-trip
// ---------------------------------------------------------------------------
bool round_trip(const SequenceConfig& config, const FrameHeader& header,
                const std::vector<std::uint8_t>& tile) {
  const std::vector<std::uint8_t> obu = make_frame_obu(config, header, tile);
  ObuView view;
  std::size_t consumed = 0;
  std::string error;
  if (!parse_obu(obu, &view, &consumed, &error)) {
    std::printf("FAIL: parse_obu: %s\n", error.c_str());
    return false;
  }
  if (consumed != obu.size() || view.type != obu_type_of(header.role)) {
    std::printf("FAIL: OBU framing\n");
    return false;
  }
  ParsedFrameHeader parsed;
  parsed.header.role = header.role;
  if (!read_tile_group_header(view.payload, config, &parsed, &error)) {
    std::printf("FAIL: read_tile_group_header: %s\n", error.c_str());
    return false;
  }
  const FrameHeader& got = parsed.header;
  bool ok = got.role == header.role &&
            got.frame_is_inter == header.frame_is_inter &&
            got.order_hint == header.order_hint &&
            got.long_term_id == header.long_term_id &&
            got.refresh_frame_flags == header.refresh_frame_flags &&
            got.num_total_refs == header.num_total_refs &&
            got.ref_frame_idx == header.ref_frame_idx &&
            got.ref_long_term_id == header.ref_long_term_id &&
            got.immediate_output_frame == header.immediate_output_frame;
  if (!ok) {
    std::printf("FAIL: header round-trip mismatch\n");
    return false;
  }
  const std::span<const std::uint8_t> tail = view.payload.subspan(parsed.tile_offset);
  if (tail.size() != tile.size() ||
      std::memcmp(tail.data(), tile.data(), tile.size()) != 0) {
    std::printf("FAIL: tile payload offset/size\n");
    return false;
  }
  return true;
}

void test_round_trip() {
  std::vector<std::uint8_t> tile(300);
  for (std::size_t i = 0; i < tile.size(); ++i) {
    tile[i] = static_cast<std::uint8_t>((i * 37U + 11U) & 0xFFU);
  }
  for (int bits = 0; bits <= 3; ++bits) {
    for (std::uint8_t refs : {std::uint8_t{2}, std::uint8_t{4},
                              std::uint8_t{8}}) {
      SequenceConfig config = golden_config();
      config.long_term_frame_id_bits = static_cast<std::uint8_t>(bits);
      config.num_ref_frames = refs;
      config.order_hint_bits = 5;

      const int max_id = max_long_term_id(config);
      const std::int32_t clk_id = max_id < 0 ? -1 : 0;

      FrameHeader clk{};
      check(plan_closed_loop_key(config, 7, clk_id, &clk, nullptr),
            "rt CLK plan");
      check(round_trip(config, clk, tile), "rt CLK");

      FrameHeader olk{};
      std::vector<std::int32_t> retain;
      if (max_id >= 0) retain.push_back(0);
      const std::int32_t olk_id = max_id < 0 ? -1 : (max_id >= 1 ? 1 : 0);
      const PoolState pool = pool_after_clk(config, clk_id);
      for (const LongTermListSyntax syntax :
           {LongTermListSyntax::kSpecV1,
            LongTermListSyntax::kOmitPinnedDav2d}) {
        SequenceConfig list_config = config;
        list_config.olk_long_term_list = syntax;
        std::vector<std::int32_t> list = retain;
        if (syntax == LongTermListSyntax::kOmitPinnedDav2d) list.clear();
        std::string olk_error;
        check(plan_open_loop_key(list_config, pool, 9,
                                 static_cast<std::uint8_t>(refs - 1), olk_id,
                                 list, &olk, &olk_error),
              "rt OLK plan: " + olk_error);
        check(round_trip(list_config, olk, tile), "rt OLK");
      }

      for (std::uint8_t role_u : {static_cast<std::uint8_t>(FrameRole::kRegular),
                                  static_cast<std::uint8_t>(FrameRole::kLeading)}) {
        const FrameRole role = static_cast<FrameRole>(role_u);
        FrameHeader p{};
        std::vector<std::uint8_t> refs_in = {0};
        if (refs > 2) refs_in.push_back(2);
        check(plan_inter_frame(config, role, 11, refs_in, 1,
                               role == FrameRole::kRegular, &p, nullptr),
              "rt P plan");
        check(round_trip(config, p, tile), "rt P");
      }
    }
  }

  // INTRA_ONLY_FRAME round-trip.
  {
    SequenceConfig config = golden_config();
    FrameHeader intra{};
    intra.role = FrameRole::kRegular;
    intra.frame_is_inter = false;
    intra.order_hint = 3;
    intra.refresh_frame_flags = 0x2;
    intra.immediate_output_frame = true;
    std::string error;
    check(validate_frame_header(config, intra, &error), "rt intra-only valid");
    check(round_trip(config, intra, tile), "rt intra-only");
  }
}

void test_sequence_round_trip() {
  for (std::uint8_t bits : {std::uint8_t{0}, std::uint8_t{1},
                             std::uint8_t{2}, std::uint8_t{3},
                             std::uint8_t{7}}) {
    for (std::uint8_t refs : {std::uint8_t{2}, std::uint8_t{4},
                               std::uint8_t{16}}) {
      SequenceConfig config = golden_config();
      config.long_term_frame_id_bits = bits;
      config.num_ref_frames = refs;
      config.order_hint_bits = 6;
      config.max_frame_width = 1280;
      config.max_frame_height = 720;
      const std::vector<std::uint8_t> obu = make_sequence_header_obu(config);
      ObuView view;
      std::size_t consumed = 0;
      std::string error;
      check(parse_obu(obu, &view, &consumed, &error), "seq parse_obu");
      check(view.type == 1, "seq OBU type");
      std::size_t bits_read = 0;
      check(read_sequence_header(view.payload, config, &bits_read, &error),
            "read_sequence_header: " + error);
    }
  }
}

void test_sequence_level_and_layer_gating() {
  // §5.4.1: seq_tier is present only for seq_level_idx > 3.
  for (const std::uint8_t level : {std::uint8_t{3}, std::uint8_t{5}}) {
    SequenceConfig config = golden_config();
    config.seq_level_idx = level;
    const std::vector<std::uint8_t> obu = make_sequence_header_obu(config);
    check(!obu.empty(), "level gating: seq OBU");
    ObuView view;
    std::size_t consumed = 0;
    std::string error;
    check(parse_obu(obu, &view, &consumed, &error),
          "level gating: parse_obu: " + error);
    std::size_t bits_read = 0;
    check(read_sequence_header(view.payload, config, &bits_read, &error),
          "level gating: read_sequence_header: " + error);
  }

  // B6: non-zero layer maxima require §5.4.1 dependency syntax this writer
  // does not implement. Reject instead of omitting mandatory fields.
  for (const bool mlayer : {false, true}) {
    SequenceConfig config = golden_config();
    if (mlayer) {
      config.max_mlayer_id = 2;
    } else {
      config.max_tlayer_id = 1;
    }
    std::string error;
    check(!validate_sequence_header_config(config, &error),
          "layer gating: validate rejects");
    check(error.find("not implemented") != std::string::npos,
          "layer gating: message names the missing syntax");
    check(make_sequence_header_obu(config).empty(),
          "layer gating: make_sequence_header_obu returns empty");
    av2enc::BitWriter writer;
    check(write_sequence_header(writer, config) == 0,
          "layer gating: write_sequence_header writes nothing");
    check(writer.bit_size() == 0, "layer gating: no partial header");
    check(!read_sequence_header(std::span<const std::uint8_t>{}, config,
                                nullptr, &error),
          "layer gating: reader rejects the same config");
  }
}

// ---------------------------------------------------------------------------
// Pool / random access
// ---------------------------------------------------------------------------
struct Scenario {
  SequenceConfig config;
  PoolState pool;
  std::string error;
};

void apply(Scenario* s, const FrameHeader& header, bool first_in_tu,
           PoolEvents* events, const std::string& what) {
  check(apply_frame(s->config, header, first_in_tu, &s->pool, events,
                    &s->error),
        what + ": " + s->error);
}

void apply_fails(const SequenceConfig& config, PoolState pool,
                 const FrameHeader& header, bool first_in_tu,
                 const std::string& what) {
  PoolEvents events;
  std::string error;
  const bool ok = apply_frame(config, header, first_in_tu, &pool, &events,
                             &error);
  check(!ok && !error.empty(), what + " (expected rejection)");
  if (ok) std::printf("  accepted unexpectedly: %s\n", what.c_str());
}

void test_pool_basic() {
  Scenario s;
  s.config = golden_config();
  reset_pool(s.config, &s.pool);

  FrameHeader clk{};
  check(plan_closed_loop_key(s.config, 0, 0, &clk, nullptr), "basic CLK plan");
  PoolEvents ev;
  apply(&s, clk, true, &ev, "basic CLK");
  // §7.23: KEY_FRAME updates only the first refreshed slot with RefValid = 1.
  check(ev.closed_random_access, "CLK is startCVS");
  check((ev.valid_after & 0x1U) != 0U, "CLK leaves slot 0 valid");
  check(ev.valid_after == 0x1U, "CLK leaves only slot 0 valid");
  check(ev.first_refreshed_slot_only, "CLK uses the 'first' rule");
  check(s.pool.slots[0].long_term_id == 0, "CLK stores its long-term id");
  check(s.pool.slots[1].long_term_id == 0,
        "refreshed sibling slots carry the id but are invalid");

  FrameHeader p1{};
  check(plan_inter_frame(s.config, FrameRole::kRegular, 1,
                         std::vector<std::uint8_t>{0}, 0, true, &p1, nullptr),
        "basic P1 plan");
  apply(&s, p1, true, &ev, "basic P1");
  check(ev.valid_after == 0x1U, "P1 keeps slot 0 valid");
  check(s.pool.slots[0].long_term_id == -1, "P1 clears the long-term id");

  // Referencing an invalid slot is rejected.
  FrameHeader bad{};
  check(plan_inter_frame(s.config, FrameRole::kRegular, 2,
                         std::vector<std::uint8_t>{1}, 0, true, &bad, nullptr),
        "basic bad plan");
  apply_fails(s.config, s.pool, bad, true, "P referencing invalid slot 1");
}

void test_long_term_retention() {
  // CLK A (id 0) -> OLK K (id 1, retains 0, refresh slot 1) with a same-TU P2
  // that references K -> P3 in the next TU references slot 0 (the far
  // long-term reference).
  Scenario s;
  s.config = golden_config();
  reset_pool(s.config, &s.pool);

  FrameHeader a{};
  check(plan_closed_loop_key(s.config, 0, 0, &a, nullptr), "ret CLK plan");
  PoolEvents ev;
  apply(&s, a, true, &ev, "ret CLK");

  FrameHeader k{};
  check(plan_open_loop_key(s.config, s.pool, 1, /*olk_slot=*/1,
                           /*olk_lt_id=*/1, std::vector<std::int32_t>{0}, &k,
                           &s.error),
        "ret OLK plan: " + s.error);
  apply(&s, k, true, &ev, "ret OLK");
  check(ev.open_random_access, "OLK is an open random access point");
  check(ev.refreshed_flags == 0x2U, "OLK refreshes only slot 1");
  check(ev.valid_after == 0x3U, "OLK keeps slot 0 valid (not refreshed by it)");
  check(s.pool.olk_encountered, "OlkEncountered = 1");
  check(s.pool.olk_refresh == 0x2U, "OlkRefresh = 1 << 1");

  FrameHeader p2{};
  check(plan_inter_frame(s.config, FrameRole::kRegular, 2,
                         std::vector<std::uint8_t>{1}, /*refresh=*/2, true, &p2,
                         nullptr),
        "ret P2 plan");
  apply(&s, p2, false, &ev, "ret P2");
  check(!ev.invalidation_ran, "no invalidation inside the OLK temporal unit");
  check(s.pool.slots[0].long_term_id == 0 && s.pool.slots[0].valid,
        "slot 0 still holds CLK A");
  check(s.pool.slots[1].long_term_id == 1, "slot 1 holds the OLK");

  FrameHeader p3{};
  check(plan_inter_frame(s.config, FrameRole::kRegular, 3,
                         std::vector<std::uint8_t>{0}, /*refresh=*/2, true, &p3,
                         nullptr),
        "ret P3 plan");
  apply(&s, p3, true, &ev, "ret P3");
  check(ev.invalidation_ran, "first Regular frame of the next TU invalidates");
  // Slot 0 is an LTR named by the OLK; slot 1 was refreshed by the OLK. Slot 2
  // holds the P frame from the OLK temporal unit, which is not kept.
  check(ev.invalidated_flags == 0x4U,
        "only the non-LTR, non-OlkRefresh slot is invalidated");
  check((ev.valid_after & 0x1U) != 0U,
        "the named long-term reference survives the invalidation");
  check(s.pool.slots[0].valid && s.pool.slots[0].long_term_id == 0,
        "slot 0 still carries long-term id 0");
  check(s.pool.slots[2].valid, "P3 refreshed slot 2");
  check(s.pool.slots[3].valid == false, "slot 3 stays invalid");
}

void test_invalidation_without_ltr() {
  // An OLK that leaves a live long-term reference in place must name it;
  // otherwise the §5.18.2 invalidation would keep a reference the §7.4.4 list
  // says does not survive.
  Scenario s;
  s.config = golden_config();
  reset_pool(s.config, &s.pool);
  FrameHeader a{};
  check(plan_closed_loop_key(s.config, 0, 0, &a, nullptr), "inv CLK plan");
  PoolEvents ev;
  apply(&s, a, true, &ev, "inv CLK");

  FrameHeader unnamed{};
  check(!plan_open_loop_key(s.config, s.pool, 1, 1, 1, {}, &unnamed,
                            &s.error),
        "an OLK must name every live long-term reference it does not refresh");

  // Refreshing the long-term slot is the other legal option: the reference is
  // gone at the OLK, so there is nothing to name.
  PoolState copy = s.pool;
  FrameHeader over{};
  check(plan_open_loop_key(s.config, s.pool, 1, /*olk_slot=*/0, 1, {}, &over,
                           &s.error),
        "OLK may refresh the long-term slot instead of naming it: " + s.error);
  std::string error;
  check(apply_frame(s.config, over, true, &copy, &ev, &error),
        "apply OLK over the long-term slot: " + error);
  check(copy.slots[0].long_term_id == 1,
        "slot 0 now holds the OLK's long-term id");

  // A rejected frame leaves the pool untouched.
  PoolState pristine = s.pool;
  check(!apply_frame(s.config, unnamed, true, &pristine, &ev, &error) &&
            pristine.slots[0].valid && pristine.slots[0].long_term_id == 0,
        "a rejected OLK leaves the pool unchanged");
}

void test_open_random_access_validation() {
  Scenario s;
  s.config = golden_config();
  reset_pool(s.config, &s.pool);
  FrameHeader a{};
  check(plan_closed_loop_key(s.config, 0, 0, &a, nullptr), "ora CLK plan");
  PoolEvents ev;
  apply(&s, a, true, &ev, "ora CLK");

  // An OLK that refreshes slot 0 (so it need not name id 0) but a later frame
  // still needs id 0: the required-id check fails.
  FrameHeader k{};
  std::string error;
  check(plan_open_loop_key(s.config, s.pool, 1, /*olk_slot=*/0, 1, {}, &k,
                           &error),
        "ora OLK plan: " + error);
  check(!validate_open_random_access(s.config, k, s.pool,
                                     std::vector<std::int32_t>{0}, &error),
        "OLK must name every required long-term id");

  // Naming an id that is not held.
  FrameHeader k2{};
  check(!plan_open_loop_key(s.config, s.pool, 1, 1, 1,
                            std::vector<std::int32_t>{1}, &k2, &error),
        "OLK cannot name an id that is not held");

  // The legal configuration.
  FrameHeader k3{};
  check(plan_open_loop_key(s.config, s.pool, 1, 1, 1,
                           std::vector<std::int32_t>{0}, &k3, &error),
        "OLK naming the live reference is legal: " + error);
  check(validate_open_random_access(s.config, k3, s.pool,
                                    std::vector<std::int32_t>{0}, &error),
        "validate_open_random_access accepts the legal OLK: " + error);

  // Refreshing a named long-term slot is rejected (the reference would not be
  // held at the first Regular frame).
  FrameHeader k4 = k3;
  k4.refresh_frame_flags = 0x3U;  // slots 0 and 1
  check(!validate_open_random_access(s.config, k4, s.pool,
                                     std::vector<std::int32_t>{0}, &error),
        "OLK cannot refresh a named long-term slot");
}

void test_rejected_headers() {
  const SequenceConfig config = golden_config();
  std::string error;

  const PoolState rej_pool = pool_after_clk(config, 0);
  FrameHeader olk{};
  check(plan_open_loop_key(config, rej_pool, 1, 1, 1,
                           std::vector<std::int32_t>{0}, &olk, &error),
        "rej OLK plan: " + error);
  {
    FrameHeader copy = olk;
    copy.immediate_output_frame = true;
    check(!validate_frame_header(config, copy, &error),
          "OLK with immediate_output_frame = 1 is rejected");
  }
  {
    FrameHeader copy = olk;
    copy.refresh_frame_flags = 0;
    check(!validate_frame_header(config, copy, &error),
          "OLK with refresh_frame_flags = 0 is rejected");
  }
  {
    FrameHeader copy = olk;
    copy.ref_long_term_id = {3};  // (1 << 2) - 1 is reserved
    check(!validate_frame_header(config, copy, &error),
          "ref_long_term_id == (1 << bits) - 1 is rejected");
  }
  {
    FrameHeader copy = olk;
    copy.ref_long_term_id.resize(8);
    check(!validate_frame_header(config, copy, &error),
          "num_key_ref_frames > 7 is rejected");
  }
  {
    FrameHeader copy = olk;
    copy.long_term_id = 2;  // max is 2 for two bits
    check(validate_frame_header(config, copy, &error),
          "long_term_id == max is legal");
    copy.long_term_id = 3;
    check(!validate_frame_header(config, copy, &error),
          "long_term_id == (1 << bits) - 1 is rejected");
  }

  SequenceConfig no_lt = config;
  no_lt.long_term_frame_id_bits = 0;
  {
    FrameHeader copy = olk;
    copy.ref_long_term_id = {0};
    check(!validate_frame_header(no_lt, copy, &error),
          "ref_long_term_id with long_term_frame_id_bits = 0 is rejected");
  }
  {
    FrameHeader copy = olk;
    copy.long_term_id = 0;
    check(!validate_frame_header(no_lt, copy, &error),
          "long_term_id with long_term_frame_id_bits = 0 is rejected");
  }
  {
    FrameHeader clk{};
    check(plan_closed_loop_key(no_lt, 0, -1, &clk, nullptr), "no-LT CLK");
    check(validate_frame_header(no_lt, clk, &error), "no-LT CLK is legal");
  }

  // A CLK cannot be planned with a non-allFrames mask.
  {
    FrameHeader clk{};
    check(plan_closed_loop_key(config, 0, -1, &clk, nullptr), "rej CLK plan");
    FrameHeader copy = clk;
    copy.refresh_frame_flags = 0x1U;
    check(!validate_frame_header(config, copy, &error),
          "CLK refreshes allFrames");
  }

  // Too many references.
  {
    FrameHeader p{};
    check(plan_inter_frame(config, FrameRole::kRegular, 1,
                           std::vector<std::uint8_t>{0, 1}, 0, true, &p,
                           nullptr),
          "two refs are legal");
    p.num_total_refs = 8;
    p.ref_frame_idx.resize(8, 0);
    check(!validate_frame_header(config, p, &error),
          "num_total_refs > ActiveNumRefFrames is rejected");
  }

  // intra-only cannot refresh all slots.
  {
    FrameHeader intra{};
    intra.role = FrameRole::kRegular;
    intra.frame_is_inter = false;
    intra.refresh_frame_flags = (1U << config.num_ref_frames) - 1U;
    check(!validate_frame_header(config, intra, &error),
          "INTRA_ONLY_FRAME cannot refresh all slots");
  }
  (void)0;
}

void test_temporal_unit_rules() {
  const SequenceConfig config = golden_config();
  std::string error;

  const PoolState tu_pool = pool_after_clk(config, 0);
  FrameHeader olk{};
  check(plan_open_loop_key(config, tu_pool, 1, 1, 1,
                           std::vector<std::int32_t>{0}, &olk, &error),
        "tu OLK plan: " + error);
  check(!validate_temporal_unit(config, std::vector<FrameHeader>{olk}, &error),
        "an OLK-only temporal unit has no output frame unit");

  FrameHeader p{};
  check(plan_inter_frame(config, FrameRole::kRegular, 2,
                         std::vector<std::uint8_t>{1}, 2, true, &p, nullptr),
        "tu P plan");
  check(validate_temporal_unit(config, std::vector<FrameHeader>{olk, p}, &error),
        "OLK + output Regular frame is a legal TU: " + error);

  FrameHeader clk{};
  check(plan_closed_loop_key(config, 0, 0, &clk, nullptr), "tu CLK plan");
  check(!validate_temporal_unit(config, std::vector<FrameHeader>{clk, olk},
                                &error),
        "a TU cannot mix CLK and OLK");

  FrameHeader p2{};
  check(plan_inter_frame(config, FrameRole::kRegular, 2,
                         std::vector<std::uint8_t>{1}, 2, true, &p2, nullptr),
        "tu P2 plan");
  check(!validate_temporal_unit(config, std::vector<FrameHeader>{p, clk},
                                &error),
        "a CLK must be the first coded frame unit");
  check(!validate_temporal_unit(config, std::vector<FrameHeader>{p, olk},
                                &error),
        "an OLK must be the first coded frame unit");

  FrameHeader leading{};
  check(plan_inter_frame(config, FrameRole::kLeading, 2,
                         std::vector<std::uint8_t>{0}, 2, true, &leading,
                         nullptr),
        "tu leading plan");
  check(!validate_temporal_unit(config, std::vector<FrameHeader>{leading, p2},
                                &error),
        "leading and regular frames cannot share a TU");
  check(validate_temporal_unit(config, std::vector<FrameHeader>{leading, leading},
                               &error),
        "a leading-only TU is legal");

  // Two output units must share one OrderHint.
  FrameHeader p3{};
  check(plan_inter_frame(config, FrameRole::kRegular, 3,
                         std::vector<std::uint8_t>{1}, 2, true, &p3, nullptr),
        "tu P3 plan");
  check(!validate_temporal_unit(config, std::vector<FrameHeader>{p2, p3},
                                &error),
        "two output frame units need the same OrderHint");
}

void test_leading_frames() {
  // After an OLK, a leading frame may reference a pre-OLK picture but it must
  // not refresh a long-term slot; the invalidation only runs for the first
  // Regular frame.
  Scenario s;
  s.config = golden_config();
  reset_pool(s.config, &s.pool);
  FrameHeader a{};
  check(plan_closed_loop_key(s.config, 0, 0, &a, nullptr), "lead CLK plan");
  PoolEvents ev;
  apply(&s, a, true, &ev, "lead CLK");

  FrameHeader k{};
  check(plan_open_loop_key(s.config, s.pool, 4, 1, 1,
                           std::vector<std::int32_t>{0}, &k, &s.error),
        "lead OLK plan: " + s.error);
  apply(&s, k, true, &ev, "lead OLK");
  check(is_leading_candidate(s.pool, 2), "order 2 < OLK order 4 is leading");
  check(!is_leading_candidate(s.pool, 5), "order 5 is not leading");

  // Leading frame in a later TU: refs the pre-OLK CLK (slot 0), refresh slot 2.
  FrameHeader lead{};
  check(plan_inter_frame(s.config, FrameRole::kLeading, 2,
                         std::vector<std::uint8_t>{0}, 2, true, &lead,
                         nullptr),
        "lead frame plan");
  apply(&s, lead, true, &ev, "lead frame");
  check(!ev.invalidation_ran, "a leading frame does not run the invalidation");
  check(s.pool.olk_encountered, "OlkEncountered survives the leading frame");

  // The first Regular frame then invalidates everything except OlkRefresh.
  FrameHeader reg{};
  check(plan_inter_frame(s.config, FrameRole::kRegular, 5,
                         std::vector<std::uint8_t>{1}, 2, true, &reg, nullptr),
        "lead regular plan");
  apply(&s, reg, true, &ev, "lead regular");
  check(ev.invalidation_ran, "the first Regular frame invalidates");
  check((ev.invalidated_flags & 0x4U) != 0U,
        "the leading frame's slot 2 is dropped");

  // A leading frame that refreshes the retained long-term slot is rejected.
  Scenario t;
  t.config = golden_config();
  reset_pool(t.config, &t.pool);
  FrameHeader a2{};
  check(plan_closed_loop_key(t.config, 0, 2, &a2, nullptr), "lead2 CLK plan");
  apply(&t, a2, true, &ev, "lead2 CLK");
  FrameHeader k2{};
  check(plan_open_loop_key(t.config, t.pool, 4, 1, 1,
                           std::vector<std::int32_t>{2}, &k2, &t.error),
        "lead2 OLK plan: " + t.error);
  apply(&t, k2, true, &ev, "lead2 OLK");
  FrameHeader bad_lead{};
  check(plan_inter_frame(t.config, FrameRole::kLeading, 2,
                         std::vector<std::uint8_t>{2}, /*refresh=*/2, true,
                         &bad_lead, nullptr),
        "lead2 bad plan");
  apply_fails(t.config, t.pool, bad_lead, true,
              "leading frame overwriting a retained long-term slot");
}

void test_closed_random_access() {
  // A second CLK mid-stream invalidates every slot even if long-term ids were
  // live, and only slot 0 survives its own update.
  Scenario s;
  s.config = golden_config();
  reset_pool(s.config, &s.pool);
  FrameHeader a{};
  check(plan_closed_loop_key(s.config, 0, 0, &a, nullptr), "cra CLK1 plan");
  PoolEvents ev;
  apply(&s, a, true, &ev, "cra CLK1");

  FrameHeader k{};
  check(plan_open_loop_key(s.config, s.pool, 1, 1, 1,
                           std::vector<std::int32_t>{0}, &k, &s.error),
        "cra OLK plan: " + s.error);
  apply(&s, k, true, &ev, "cra OLK");

  FrameHeader a2{};
  check(plan_closed_loop_key(s.config, 2, 2, &a2, nullptr), "cra CLK2 plan");
  std::string error;
  check(validate_closed_random_access(s.config, a2, &error),
        "CLK is a closed random access point: " + error);
  apply(&s, a2, true, &ev, "cra CLK2");
  check(ev.closed_random_access, "CLK2 is startCVS");
  check(ev.valid_after == 0x1U, "CLK2 leaves only slot 0 valid");
  check(s.pool.slots[0].long_term_id == 2, "CLK2 takes over the long-term id");
  check(s.pool.slots[1].valid == false, "the old OLK slot is gone");
}

void test_planners() {
  const SequenceConfig config = golden_config();
  PoolState pool;
  reset_pool(config, &pool);
  FrameHeader a{};
  check(plan_closed_loop_key(config, 0, 0, &a, nullptr), "plan CLK");
  PoolEvents ev;
  apply_frame(config, a, true, &pool, &ev, nullptr);

  FrameHeader bad{};
  std::string error;
  check(!plan_open_loop_key(config, pool, 1, 4, 1, {}, &bad, &error),
        "OLK slot >= NumRefFrames is rejected");
  check(!plan_inter_frame(config, FrameRole::kClosedLoopKey, 1, {}, 0, true,
                          &bad, &error),
        "plan_inter_frame rejects key roles");
  check(!plan_inter_frame(config, FrameRole::kRegular, 1,
                          std::vector<std::uint8_t>{0}, 4, true, &bad, &error),
        "refresh slot >= NumRefFrames is rejected");
}

// ---------------------------------------------------------------------------
// Pinned-dav2d-compatible OLK layout
// ---------------------------------------------------------------------------
void test_pinned_dav2d_layout() {
  SequenceConfig config = golden_config();
  config.olk_long_term_list = LongTermListSyntax::kOmitPinnedDav2d;
  const PoolState pool = pool_after_clk(config, /*lt_id=*/0);

  FrameHeader olk{};
  std::string error;
  check(plan_open_loop_key(config, pool, /*order_hint=*/1, /*olk_slot=*/1,
                           /*olk_long_term_id=*/1, {}, &olk, &error),
        "dav2d OLK plan: " + error);
  // is_first(1) uvlc uvlc lt_plus_1(2) fso(1) order(4) refresh(4)
  // intrabc(1) dcdf(1) uniform(1) base_q(8) 00000 tx(1) reduced(2)
  // = 33 bits, padded to 5 bytes.
  {
    av2enc::BitWriter writer;
    write_tile_group_header(writer, config, olk);
    const std::vector<std::uint8_t> bytes = writer.take_bytes();
    check(hex(bytes) == "f049ad0000",
          "golden pinned-dav2d OLK tile-group header, got " + hex(bytes));
  }
  check(round_trip(config, olk, std::vector<std::uint8_t>{0xAB, 0xCD}),
        "rt pinned-dav2d OLK");

  // The list is not expressible in this layout.
  FrameHeader with_list = olk;
  with_list.ref_long_term_id = {0};
  check(!validate_frame_header(config, with_list, &error),
        "the pinned-dav2d layout rejects ref_long_term_id");

  // Retention still works: §5.18.2 keeps slots whose RefLongTermId != -1.
  Scenario s;
  s.config = config;
  reset_pool(config, &s.pool);
  FrameHeader a{};
  check(plan_closed_loop_key(config, 0, 0, &a, nullptr), "dv CLK plan");
  PoolEvents ev;
  apply(&s, a, true, &ev, "dv CLK");
  apply(&s, olk, true, &ev, "dv OLK");
  FrameHeader p2{};
  check(plan_inter_frame(config, FrameRole::kRegular, 2,
                         std::vector<std::uint8_t>{1}, 2, true, &p2, nullptr),
        "dv P2 plan");
  apply(&s, p2, false, &ev, "dv P2");
  FrameHeader p3{};
  check(plan_inter_frame(config, FrameRole::kRegular, 3,
                         std::vector<std::uint8_t>{0}, 2, true, &p3, nullptr),
        "dv P3 plan");
  apply(&s, p3, true, &ev, "dv P3");
  check(ev.invalidation_ran, "dv: the first Regular frame invalidates");
  check((ev.valid_after & 0x1U) != 0U,
        "dv: the long-term reference survives without an explicit list");
  check(s.pool.slots[0].long_term_id == 0,
        "dv: slot 0 still carries id 0");

  // In the v1.0.0 layout the same OLK would be rejected: it silently keeps a
  // live long-term reference it does not name.
  SequenceConfig spec = config;
  spec.olk_long_term_list = LongTermListSyntax::kSpecV1;
  Scenario t;
  t.config = spec;
  reset_pool(spec, &t.pool);
  FrameHeader a2{};
  check(plan_closed_loop_key(spec, 0, 0, &a2, nullptr), "dv spec CLK plan");
  apply(&t, a2, true, &ev, "dv spec CLK");
  FrameHeader unnamed{};
  check(!plan_open_loop_key(spec, t.pool, 1, 1, 1, {}, &unnamed, &t.error),
        "dv: v1.0.0 OLK must name the live long-term reference");
}

// ---------------------------------------------------------------------------
// Stream assembler
// ---------------------------------------------------------------------------
void test_stream_assembly() {
  const SequenceConfig config = golden_config();
  const std::vector<std::uint8_t> tile = {0x11, 0x22, 0x33};

  std::vector<TemporalUnitPlan> units(3);
  FrameHeader a{};
  check(plan_closed_loop_key(config, 0, 0, &a, nullptr), "asm CLK");
  units[0].frames.push_back(PlannedFrame{a, tile});

  PoolState pool;
  reset_pool(config, &pool);
  PoolEvents ev;
  check(apply_frame(config, a, true, &pool, &ev, nullptr), "asm apply CLK");

  FrameHeader k{};
  std::string olk_error;
  check(plan_open_loop_key(config, pool, 1, 1, 1,
                           std::vector<std::int32_t>{0}, &k, &olk_error),
        "asm OLK: " + olk_error);
  units[1].frames.push_back(PlannedFrame{k, tile});
  FrameHeader p2{};
  check(plan_inter_frame(config, FrameRole::kRegular, 2,
                         std::vector<std::uint8_t>{1}, 2, true, &p2, nullptr),
        "asm P2");
  units[1].frames.push_back(PlannedFrame{p2, tile});
  FrameHeader p3{};
  check(plan_inter_frame(config, FrameRole::kRegular, 3,
                         std::vector<std::uint8_t>{0}, 2, true, &p3, nullptr),
        "asm P3");
  units[2].frames.push_back(PlannedFrame{p3, tile});

  AssembledSequence out;
  std::string error;
  check(assemble_sequence(config, units, &out, &error), "assemble: " + error);
  check(out.units.size() == 3, "three temporal units");
  check(out.frames.size() == 4, "four frames");
  check(out.frames[1].obu_type == 5, "second frame is an OLK OBU");
  check((out.valid_slots_at_end & 0x1U) != 0U,
        "the long-term slot is still valid at the end");
  check(out.frames[3].pool.invalidation_ran,
        "the first Regular frame after the OLK invalidated slots");

  // A sequence that does not start at a CLK is rejected.
  std::vector<TemporalUnitPlan> bad(1);
  bad[0].frames.push_back(PlannedFrame{p3, tile});
  AssembledSequence ignored;
  check(!assemble_sequence(config, bad, &ignored, &error),
        "a sequence must start at a CLK");
}

}  // namespace

int main() {
  test_goldens();
  test_video_equivalence();
  test_round_trip();
  test_sequence_round_trip();
  test_sequence_level_and_layer_gating();
  test_pool_basic();
  test_long_term_retention();
  test_invalidation_without_ltr();
  test_open_random_access_validation();
  test_rejected_headers();
  test_temporal_unit_rules();
  test_leading_frames();
  test_closed_random_access();
  test_planners();
  test_stream_assembly();
  test_pinned_dav2d_layout();

  std::printf("%d checks, %d failure(s)\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
