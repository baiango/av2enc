// Unit tests for the leading-picture framing, OBU/frame-header reader/writer,
// output suppression and §7.4.4 reference restrictions.
#include "codec/inter/leading/leading.h"

#include "codec/inter/leading/leading_reader.h"
#include "pipeline/video/video_headers.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using av2enc::inter::leading::FrameHeader;
using av2enc::inter::leading::FrameKind;
using av2enc::inter::leading::ObuType;
using av2enc::inter::leading::ParsedFrameHeader;
using av2enc::inter::leading::ParsedObu;
using av2enc::inter::leading::Plan;
using av2enc::inter::leading::SequenceConfig;

int failures = 0;
int checks = 0;

void expect(bool condition, const std::string& message) {
  ++checks;
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  }
}

std::string to_hex(const std::vector<std::uint8_t>& bytes) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2U);
  for (const std::uint8_t byte : bytes) {
    out.push_back(digits[byte >> 4U]);
    out.push_back(digits[byte & 0x0FU]);
  }
  return out;
}

SequenceConfig base_config() {
  SequenceConfig config;
  config.max_frame_width = 64;
  config.max_frame_height = 64;
  config.base_q_idx = 90;
  config.order_hint_bits = 4;
  config.num_ref_frames = 8;
  config.monotonic_output_order_flag = false;
  return config;
}

FrameHeader header(FrameKind kind, std::uint32_t order_hint,
                   std::uint32_t refresh, std::uint8_t ref_count,
                   std::uint8_t ref0, bool leading_output = false) {
  FrameHeader h;
  h.kind = kind;
  h.order_hint = order_hint;
  h.refresh_frame_flags = refresh;
  h.num_total_refs = ref_count;
  h.ref_frame_idx[0] = ref0;
  h.leading_output_frame = leading_output;
  return h;
}

FrameHeader from_parsed(const ParsedFrameHeader& parsed) {
  FrameHeader h;
  h.kind = parsed.kind;
  h.order_hint = parsed.order_hint;
  h.refresh_frame_flags = parsed.refresh_frame_flags;
  h.num_total_refs = parsed.num_total_refs;
  for (std::size_t i = 0; i < parsed.ref_frame_idx.size(); ++i) {
    h.ref_frame_idx[i] = parsed.ref_frame_idx[i];
  }
  h.leading_output_frame = parsed.immediate_output_frame;
  return h;
}

void test_sequence_header_matches_m2() {
  av2enc::video::VideoConfig video_config;
  video_config.max_frame_width = 64;
  video_config.max_frame_height = 64;
  video_config.num_ref_frames = 2;
  video_config.order_hint_bits = 4;
  SequenceConfig config = base_config();
  config.num_ref_frames = 2;
  config.monotonic_output_order_flag = true;  // M2 always writes 1.

  const std::vector<std::uint8_t> ours =
      av2enc::inter::leading::make_sequence_header_obu(config);
  const std::vector<std::uint8_t> m2 =
      av2enc::video::make_sequence_header_obu(video_config);
  expect(to_hex(ours) == to_hex(m2),
         "monotonic=1 sequence header must match the M2 writer");
}

void test_frame_headers_match_m2() {
  av2enc::video::VideoConfig video_config;
  video_config.max_frame_width = 64;
  video_config.max_frame_height = 64;
  video_config.num_ref_frames = 2;
  video_config.order_hint_bits = 4;

  SequenceConfig config = base_config();
  config.num_ref_frames = 2;
  config.monotonic_output_order_flag = true;
  const std::vector<std::uint8_t> tile{0xAA, 0x55};

  av2enc::video::VideoFrameHeader video_key;
  video_key.keyframe = true;
  video_key.order_hint = 0;
  video_key.refresh_frame_flags = 3;
  const std::vector<std::uint8_t> video_clk =
      av2enc::video::make_frame_obu(video_config, video_key, tile);
  const std::vector<std::uint8_t> ours_clk = av2enc::inter::leading::make_frame_obu(
      config, header(FrameKind::kClosedLoopKey, 0, 3, 0, 0), tile);
  expect(to_hex(ours_clk) == to_hex(video_clk),
         "CLK frame OBU must match the M2 writer");

  av2enc::video::VideoFrameHeader video_p;
  video_p.keyframe = false;
  video_p.order_hint = 1;
  video_p.ref_slot = 0;
  video_p.refresh_frame_flags = 1;
  const std::vector<std::uint8_t> video_regular =
      av2enc::video::make_frame_obu(video_config, video_p, tile);
  const std::vector<std::uint8_t> ours_regular =
      av2enc::inter::leading::make_frame_obu(
          config, header(FrameKind::kRegularTileGroup, 1, 1, 1, 0), tile);
  expect(to_hex(ours_regular) == to_hex(video_regular),
         "regular P frame OBU must match the M2 writer");
}

void test_bit_goldens() {
  const SequenceConfig config = base_config();
  const std::vector<std::uint8_t> tile{0xAA, 0x55};

  expect(to_hex(av2enc::inter::leading::make_sequence_header_obu(config)) ==
             "13048075000abffe0028006ee3800000101b8082",
         "sequence header golden");

  expect(to_hex(av2enc::inter::leading::make_frame_obu(
             config,
             header(FrameKind::kClosedLoopKey, 0, 255, 0, 0), tile)) ==
             "0710f035a000aa55",
         "CLK golden");

  expect(to_hex(av2enc::inter::leading::make_frame_obu(
             config,
             header(FrameKind::kOpenLoopKey, 4, 1, 0, 0), tile)) ==
             "0814f200b5a000aa55",
         "OLK golden");

  expect(to_hex(av2enc::inter::leading::make_frame_obu(
             config,
             header(FrameKind::kLeadingTileGroup, 1, 2, 1, 0), tile)) ==
             "0a18f028148235a000aa55",
         "leading non-output golden");

  expect(to_hex(av2enc::inter::leading::make_frame_obu(
             config,
             header(FrameKind::kRegularTileGroup, 5, 16, 1, 0), tile)) ==
             "0a1cf95109046b4000aa55",
         "regular golden");
}

void test_frame_round_trip() {
  const SequenceConfig config = base_config();
  const std::vector<std::uint8_t> tile{0xAA, 0x55};

  struct Case {
    const char* name;
    FrameHeader header;
    bool expect_immediate;
    bool expect_implicit;
  };
  const Case cases[] = {
      {"clk", header(FrameKind::kClosedLoopKey, 0, 255, 0, 0), true, false},
      {"olk", header(FrameKind::kOpenLoopKey, 4, 1, 0, 0), false, true},
      {"leading non-output",
       header(FrameKind::kLeadingTileGroup, 1, 2, 1, 0), false, false},
      {"leading output",
       header(FrameKind::kLeadingTileGroup, 2, 4, 1, 0, true), true, false},
      {"regular", header(FrameKind::kRegularTileGroup, 5, 16, 1, 0), true,
       false},
  };

  for (const Case& test : cases) {
    const std::vector<std::uint8_t> bytes =
        av2enc::inter::leading::make_frame_obu(config, test.header, tile);
    std::vector<ParsedObu> obus;
    std::string error;
    expect(av2enc::inter::leading::parse_obu_sequence(bytes, obus, error),
           std::string(test.name) + ": parse_obu_sequence (" + error + ")");
    if (obus.size() != 1U) {
      expect(false, std::string(test.name) + ": expected one OBU");
      continue;
    }
    ParsedFrameHeader parsed;
    expect(av2enc::inter::leading::parse_frame_header(
               obus[0].payload, obus[0].type, config, parsed, error),
           std::string(test.name) + ": parse_frame_header (" + error + ")");
    expect(parsed.kind == test.header.kind,
           std::string(test.name) + ": kind");
    expect(parsed.order_hint == test.header.order_hint,
           std::string(test.name) + ": order_hint");
    expect(parsed.refresh_frame_flags == test.header.refresh_frame_flags,
           std::string(test.name) + ": refresh_frame_flags");
    expect(parsed.immediate_output_frame == test.expect_immediate,
           std::string(test.name) + ": immediate_output_frame");
    expect(parsed.implicit_output_frame == test.expect_implicit,
           std::string(test.name) + ": implicit_output_frame");
    expect(parsed.num_total_refs == test.header.num_total_refs,
           std::string(test.name) + ": num_total_refs");
    expect(parsed.header_bytes > 0U, std::string(test.name) + ": header size");

    std::vector<std::uint8_t> parsed_tile(
        obus[0].payload.begin() +
            static_cast<std::ptrdiff_t>(parsed.header_bytes),
        obus[0].payload.end());
    expect(to_hex(parsed_tile) == to_hex(tile),
           std::string(test.name) + ": tile payload");
    const std::vector<std::uint8_t> rewritten =
        av2enc::inter::leading::make_frame_obu(config, from_parsed(parsed),
                                               parsed_tile);
    expect(to_hex(rewritten) == to_hex(bytes),
           std::string(test.name) + ": frame round trip");
  }
}

void test_sequence_round_trip() {
  SequenceConfig config = base_config();
  config.num_ref_frames = 8;
  config.order_hint_bits = 5;
  const std::vector<std::uint8_t> obu =
      av2enc::inter::leading::make_sequence_header_obu(config);
  std::vector<ParsedObu> obus;
  std::string error;
  expect(av2enc::inter::leading::parse_obu_sequence(obu, obus, error),
         "sequence: parse_obu_sequence (" + error + ")");
  if (obus.size() != 1U || obus[0].type != ObuType::kSequenceHeader) {
    expect(false, "sequence: one OBU_SEQUENCE_HEADER");
    return;
  }
  SequenceConfig parsed;
  expect(av2enc::inter::leading::parse_sequence_header(obus[0].payload, parsed,
                                                       error),
         "sequence: parse_sequence_header (" + error + ")");
  expect(parsed.max_frame_width == config.max_frame_width,
         "sequence: width");
  expect(parsed.max_frame_height == config.max_frame_height,
         "sequence: height");
  expect(parsed.order_hint_bits == config.order_hint_bits,
         "sequence: order_hint_bits");
  expect(parsed.num_ref_frames == config.num_ref_frames,
         "sequence: num_ref_frames");
  expect(parsed.monotonic_output_order_flag ==
             config.monotonic_output_order_flag,
         "sequence: monotonic_output_order_flag");
  expect(to_hex(av2enc::inter::leading::make_sequence_header_obu(parsed)) ==
             to_hex(obu),
         "sequence round trip");
}

void test_sequence_level_gating() {
  // §5.4.1: seq_tier is coded only for seq_level_idx > 3.  The level-3 header
  // must round-trip without the tier bit; a level-5 header carries it.
  for (const std::uint8_t level : {std::uint8_t{3}, std::uint8_t{5}}) {
    SequenceConfig config = base_config();
    config.seq_level_idx = level;
    const std::vector<std::uint8_t> obu =
        av2enc::inter::leading::make_sequence_header_obu(config);
    std::vector<ParsedObu> obus;
    std::string error;
    expect(av2enc::inter::leading::parse_obu_sequence(obu, obus, error),
           "level " + std::to_string(level) + ": parse_obu_sequence (" +
               error + ")");
    if (obus.size() != 1U || obus[0].type != ObuType::kSequenceHeader) {
      expect(false, "level " + std::to_string(level) +
                        ": one OBU_SEQUENCE_HEADER");
      continue;
    }
    SequenceConfig parsed;
    expect(av2enc::inter::leading::parse_sequence_header(obus[0].payload,
                                                         parsed, error),
           "level " + std::to_string(level) + ": parse_sequence_header (" +
               error + ")");
    expect(parsed.seq_level_idx == level,
           "level " + std::to_string(level) + ": seq_level_idx");
    expect(to_hex(av2enc::inter::leading::make_sequence_header_obu(parsed)) ==
               to_hex(obu),
           "level " + std::to_string(level) + ": round trip");
  }
}

void test_output_suppression() {
  SequenceConfig config = base_config();
  expect(av2enc::inter::leading::output_flags(
             config, header(FrameKind::kLeadingTileGroup, 1, 2, 1, 0))
                 .immediate == false,
         "leading non-output: immediate = 0");
  expect(av2enc::inter::leading::output_flags(
             config, header(FrameKind::kLeadingTileGroup, 1, 2, 1, 0))
                 .implicit == false,
         "leading non-output: implicit = 0");
  expect(av2enc::inter::leading::output_flags(
             config, header(FrameKind::kLeadingTileGroup, 1, 4, 1, 0, true))
                 .immediate == true,
         "leading output: immediate = 1");
  expect(av2enc::inter::leading::output_flags(
             config, header(FrameKind::kOpenLoopKey, 4, 1, 0, 0))
                 .implicit == true,
         "OLK: implicit = 1 when monotonic = 0");
  config.monotonic_output_order_flag = true;
  expect(av2enc::inter::leading::output_flags(
             config, header(FrameKind::kOpenLoopKey, 4, 1, 0, 0))
                 .implicit == false,
         "OLK: implicit = 0 when monotonic = 1");
}

Plan valid_plan() {
  Plan plan;
  plan.frames.push_back(header(FrameKind::kClosedLoopKey, 0, 255, 0, 0));
  plan.frames.push_back(header(FrameKind::kOpenLoopKey, 4, 1, 0, 0));
  plan.frames.push_back(header(FrameKind::kLeadingTileGroup, 1, 2, 1, 0));
  plan.frames.push_back(header(FrameKind::kLeadingTileGroup, 2, 4, 1, 0));
  plan.frames.push_back(header(FrameKind::kLeadingTileGroup, 3, 8, 1, 0));
  plan.frames.push_back(header(FrameKind::kRegularTileGroup, 5, 16, 1, 0));
  plan.frames.push_back(header(FrameKind::kRegularTileGroup, 6, 32, 1, 0));
  return plan;
}

void test_plan_validation() {
  const SequenceConfig config = base_config();
  std::string error;
  expect(av2enc::inter::leading::validate_plan(config, valid_plan(), error),
         "valid leading plan (" + error + ")");

  {
    Plan plan = valid_plan();
    plan.frames[2].refresh_frame_flags = 1;  // overlaps OlkRefresh
    expect(!av2enc::inter::leading::validate_plan(config, plan, error),
           "leading refresh overlap with OlkRefresh must fail");
  }
  {
    Plan plan = valid_plan();
    plan.frames[0].kind = FrameKind::kLeadingTileGroup;
    plan.frames[0].order_hint = 9;
    expect(!av2enc::inter::leading::validate_plan(config, plan, error),
           "leading frame before OLK must fail");
  }
  {
    Plan plan = valid_plan();
    plan.frames[5].ref_frame_idx[0] = 1;  // leading slot
    expect(!av2enc::inter::leading::validate_plan(config, plan, error),
           "regular frame referencing a leading frame must fail");
  }
  {
    Plan plan = valid_plan();
    plan.frames[5].ref_frame_idx[0] = 7;  // pre-OLK/invalid slot
    expect(!av2enc::inter::leading::validate_plan(config, plan, error),
           "regular frame referencing a pre-OLK frame must fail");
  }
  {
    SequenceConfig monotonic = config;
    monotonic.monotonic_output_order_flag = true;
    expect(!av2enc::inter::leading::validate_plan(monotonic, valid_plan(),
                                                  error),
           "OLK with monotonic_output_order_flag = 1 must fail");
  }
  {
    Plan plan = valid_plan();
    plan.frames[2].refresh_frame_flags = 0;  // non-output without refresh
    expect(!av2enc::inter::leading::validate_plan(config, plan, error),
           "non-output leading frame with refresh = 0 must fail");
  }
  {
    Plan plan = valid_plan();
    plan.frames[6].kind = FrameKind::kLeadingTileGroup;
    plan.frames[6].refresh_frame_flags = 64;
    plan.frames[6].leading_output_frame = true;
    expect(!av2enc::inter::leading::validate_plan(config, plan, error),
           "leading frame after the first regular frame must fail");
  }
}

void test_assembled_units() {
  const SequenceConfig config = base_config();
  const Plan plan = valid_plan();
  const std::vector<std::vector<std::uint8_t>> tiles(
      plan.frames.size(), std::vector<std::uint8_t>{0xAA, 0x55});
  const std::vector<std::vector<std::uint8_t>> units =
      av2enc::inter::leading::assemble_temporal_units(config, plan, tiles);
  expect(units.size() == plan.frames.size(), "one temporal unit per frame");
  if (units.size() != plan.frames.size()) return;
  for (const std::vector<std::uint8_t>& unit : units) {
    std::vector<ParsedObu> obus;
    std::string error;
    expect(av2enc::inter::leading::parse_obu_sequence(unit, obus, error),
           "unit parses (" + error + ")");
    expect(!obus.empty() && obus.front().type == ObuType::kTemporalDelimiter,
           "unit starts with TD");
  }
  std::vector<ParsedObu> obus;
  std::string error;
  expect(av2enc::inter::leading::parse_obu_sequence(units[0], obus, error),
         "CLK unit parses");
  expect(obus.size() == 3U, "CLK unit is TD + SEQ + CLK");
  expect(av2enc::inter::leading::parse_obu_sequence(units[1], obus, error),
         "OLK unit parses");
  expect(obus.size() == 3U, "OLK unit repeats SEQ for random access");
  expect(av2enc::inter::leading::parse_obu_sequence(units[2], obus, error),
         "leading unit parses");
  expect(obus.size() == 2U, "leading unit is TD + frame OBU only");
}

void test_parse_errors() {
  std::string error;
  ParsedObu obu;
  const std::vector<std::uint8_t> empty;
  expect(!av2enc::inter::leading::parse_obu(empty, obu, error),
         "empty OBU must fail");
  const std::vector<std::uint8_t> zero{0x00};
  expect(!av2enc::inter::leading::parse_obu(zero, obu, error),
         "zero-length OBU must fail");
  const std::vector<std::uint8_t> truncated{0x05, 0x10};
  expect(!av2enc::inter::leading::parse_obu(truncated, obu, error),
         "truncated OBU must fail");
}

}  // namespace

int main() {
  test_sequence_header_matches_m2();
  test_frame_headers_match_m2();
  test_bit_goldens();
  test_frame_round_trip();
  test_sequence_round_trip();
  test_sequence_level_gating();
  test_output_suppression();
  test_plan_validation();
  test_assembled_units();
  test_parse_errors();

  std::printf("%d checks, %d failure(s)\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
