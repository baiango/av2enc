// Unit tests for the AV2 random-access switch / RAS frame component.
//
// Covers §5.18.2 frame-type derivation, §6.17 allowed reference sets and
// conformance checks, §7.4.5 RAS reference restrictions, the sequence/frame
// header writer+reader round trip, bit-exact goldens, and the encoder-side
// "reuse the P payload or re-encode" strategy.

#include "codec/inter/random_access/random_access.h"

#include "core/bitstream/bitwriter.h"
#include "pipeline/video/video_headers.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ra = av2enc::inter::random_access;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* expression, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL line %d: %s\n", line, expression);
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

std::vector<std::uint8_t> bytes(std::initializer_list<unsigned> values) {
  std::vector<std::uint8_t> out;
  out.reserve(values.size());
  for (const unsigned value : values) {
    out.push_back(static_cast<std::uint8_t>(value));
  }
  return out;
}

bool contains(const ra::ValidationResult& result, ra::ViolationCode code) {
  for (const ra::Violation& violation : result.violations) {
    if (violation.code == code) return true;
  }
  return false;
}

ra::SequenceConfig make_config(std::uint16_t width, std::uint16_t height,
                               std::uint8_t long_term_bits) {
  ra::SequenceConfig config;
  config.max_frame_width = width;
  config.max_frame_height = height;
  config.long_term_frame_id_bits = long_term_bits;
  return config;
}

ra::RefState make_refs() {
  ra::RefState refs;
  refs.slots[0] = ra::RefSlot{true, 0, 64, 64, false};
  refs.slots[1] = ra::RefSlot{true, 1, 64, 64, false};
  refs.slots[2] = ra::RefSlot{true, -1, 64, 64, false};
  return refs;
}

ra::FrameHeader make_switch() {
  ra::FrameHeader header;
  header.obu_type = ra::ObuType::kSwitch;
  header.order_hint = 1;
  header.restricted_prediction_switch = true;
  header.num_total_refs = 1;
  header.ref_frame_idx[0] = 0;
  header.refresh_frame_flags = 2;
  header.immediate_output_frame = true;
  return header;
}

ra::FrameHeader make_ras() {
  ra::FrameHeader header;
  header.obu_type = ra::ObuType::kRas;
  header.order_hint = 2;
  header.num_key_ref_frames = 1;
  header.ref_long_term_id[0] = 0;
  header.num_total_refs = 1;
  header.ref_frame_idx[0] = 0;
  header.refresh_frame_flags = 2;
  header.immediate_output_frame = true;
  return header;
}

ra::FrameHeader make_key() {
  ra::FrameHeader header;
  header.obu_type = ra::ObuType::kClosedLoopKey;
  header.order_hint = 0;
  header.long_term_id_plus_1 = 1;
  header.immediate_output_frame = true;
  return header;
}

void test_frame_type_derivation() {
  const ra::FrameTypeInfo key =
      ra::derive_frame_type(ra::ObuType::kClosedLoopKey, false);
  CHECK(key.frame_type == ra::FrameType::kKey);
  CHECK(key.frame_is_intra);
  CHECK(!key.is_regular);
  CHECK(key.skips_primary_ref);
  CHECK(!key.codes_restricted_prediction_switch);

  const ra::FrameTypeInfo regular =
      ra::derive_frame_type(ra::ObuType::kRegularTileGroup, false);
  CHECK(regular.frame_type == ra::FrameType::kInter);
  CHECK(!regular.frame_is_intra);
  CHECK(regular.codes_frame_is_inter);
  CHECK(!regular.forces_explicit_ref_map);

  const ra::FrameTypeInfo sw =
      ra::derive_frame_type(ra::ObuType::kSwitch, true);
  CHECK(sw.frame_type == ra::FrameType::kSwitch);
  CHECK(sw.is_regular);
  CHECK(!sw.frame_is_intra);
  CHECK(sw.codes_restricted_prediction_switch);
  CHECK(sw.forces_explicit_ref_map);
  CHECK(sw.forces_frame_size_override);
  CHECK(!sw.derives_refresh_frame_flags);
  CHECK(!sw.codes_num_key_ref_frames);
  CHECK(sw.skips_primary_ref);
  CHECK(sw.resets_qm);

  const ra::FrameTypeInfo ras =
      ra::derive_frame_type(ra::ObuType::kRas, false);
  CHECK(ras.frame_type == ra::FrameType::kSwitch);
  CHECK(ras.derives_refresh_frame_flags);
  CHECK(ras.codes_num_key_ref_frames);
  CHECK(ras.resets_qm);  // RAS always resets QM
  CHECK(std::string(ra::frame_type_name(ras.frame_type)) == "SWITCH_FRAME");
  CHECK(std::string(ra::obu_type_name(ra::ObuType::kRas)) == "OBU_RAS_FRAME");
}

void test_reference_sets() {
  const ra::RefState refs = make_refs();
  const std::uint8_t ids[] = {0, 2};
  CHECK(ra::long_term_id_in_use(0, ids));
  CHECK(ra::long_term_id_in_use(2, ids));
  CHECK(!ra::long_term_id_in_use(1, ids));
  CHECK(!ra::long_term_id_in_use(-1, ids));

  CHECK(ra::allowed_reference_mask(ra::ObuType::kSwitch, refs, ids) == 0x7);
  CHECK(ra::allowed_reference_mask(ra::ObuType::kRas, refs, ids) == 0x1);
  CHECK(ra::allowed_reference_mask(ra::ObuType::kClosedLoopKey, refs, ids) ==
        0x0);

  const std::uint8_t only_zero[] = {0};
  CHECK(ra::derive_ras_refresh_flags(refs, only_zero, 3) == 0x6);
  CHECK(ra::derive_ras_refresh_flags(refs, only_zero, 2) == 0x2);
  CHECK(ra::derive_ras_refresh_flags(refs, ids, 3) == 0x6);

  ra::RefState updated = refs;
  const ra::SequenceConfig config = make_config(64, 64, 1);
  const ra::FrameHeader ras = make_ras();
  ra::update_reference_state(config, ras, 0x2, updated);
  CHECK(updated.slots[0].valid);
  CHECK(updated.slots[0].long_term_id == 0);  // protected long-term slot
  CHECK(updated.slots[1].long_term_id == -1);  // refreshed by the RAS
  CHECK(updated.slots[1].width == 64);

  ra::RefState switched = refs;
  const ra::FrameHeader sw = make_switch();
  ra::update_reference_state(config, sw, 0x2, switched);
  CHECK(switched.slots[0].restricted);   // marked before the update
  CHECK(!switched.slots[1].restricted);  // refreshed frame is not restricted
  CHECK(switched.slots[1].long_term_id == -1);

  ra::RefState keyed;
  const ra::FrameHeader key = make_key();
  ra::update_reference_state(config, key, 0, keyed);
  CHECK(keyed.slots[0].valid && keyed.slots[0].long_term_id == 0);
  CHECK(keyed.slots[1].valid && keyed.slots[1].long_term_id == 0);
}

void test_sequence_header() {
  // The long_term_frame_id_bits == 0 profile must be byte-identical to the M2
  // sequence header, proving the copied §5.4 syntax is unchanged.
  ra::SequenceConfig config = make_config(64, 64, 0);
  av2enc::video::VideoConfig video_config;
  video_config.max_frame_width = 64;
  video_config.max_frame_height = 64;
  video_config.seq_level_idx = config.seq_level_idx;
  video_config.base_q_idx = config.base_q_idx;
  video_config.order_hint_bits = config.order_hint_bits;
  video_config.num_ref_frames = config.num_ref_frames;
  video_config.max_drl_bits_minus_1 = config.max_drl_bits_minus_1;
  video_config.disable_cdf_update = config.disable_cdf_update;
  video_config.enable_cfl_intra = config.enable_cfl_intra;
  CHECK(ra::make_sequence_header_obu(config) ==
        av2enc::video::make_sequence_header_obu(video_config));

  const ra::SequenceConfig lt1 = make_config(64, 64, 1);
  const std::vector<std::uint8_t> expected64 = bytes(
      {0x13, 0x04, 0x80, 0x75, 0x00, 0x2a, 0xbf, 0xfe, 0x00, 0x28,
       0x00, 0x6e, 0x27, 0x80, 0x00, 0x00, 0x10, 0x1b, 0x80, 0x82});
  CHECK(ra::make_sequence_header_obu(lt1) == expected64);

  const ra::SequenceConfig lt1_128 = make_config(128, 128, 1);
  const std::vector<std::uint8_t> expected128 = bytes(
      {0x14, 0x04, 0x80, 0x75, 0x00, 0x2c, 0xdf, 0xff, 0x80, 0x0a,
       0x00, 0x1b, 0x89, 0xe0, 0x00, 0x00, 0x04, 0x06, 0xe0, 0x20, 0x80});
  CHECK(ra::make_sequence_header_obu(lt1_128) == expected128);

  ra::ObuView obu;
  std::size_t consumed = 0;
  std::string error;
  CHECK(ra::parse_obu(expected64, obu, consumed, error));
  CHECK(obu.obu_type == static_cast<std::uint8_t>(ra::ObuType::kSequenceHeader));
  CHECK(consumed == expected64.size());
  CHECK(error.empty());

  ra::SequenceHeaderView view;
  CHECK(ra::read_sequence_header(obu.payload, view, error));
  CHECK(view.max_frame_width == 64);
  CHECK(view.max_frame_height == 64);
  CHECK(view.order_hint_bits == 4);
  CHECK(view.num_ref_frames == 2);
  CHECK(view.long_term_frame_id_bits == 1);
  CHECK(view.max_drl_bits_minus_1 == 4);
  CHECK(view.enable_cfl_intra);
  CHECK(view.supported_profile);
}

void test_frame_header_goldens() {
  const ra::SequenceConfig config = make_config(64, 64, 1);

  const std::vector<std::uint8_t> key_obu = ra::make_frame_obu(
      config, make_key(), std::span<const std::uint8_t>());
  CHECK(key_obu == bytes({0x05, 0x10, 0xf8, 0x1a, 0xd0, 0x00}));

  const std::vector<std::uint8_t> switch_obu = ra::make_frame_obu(
      config, make_switch(), std::span<const std::uint8_t>());
  CHECK(switch_obu ==
        bytes({0x08, 0x28, 0xf8, 0xc5, 0xff, 0xe4, 0x6b, 0x40, 0x00}));

  const std::vector<std::uint8_t> ras_obu = ra::make_frame_obu(
      config, make_ras(), std::span<const std::uint8_t>());
  CHECK(ras_obu == bytes({0x08, 0x54, 0xe2, 0x91, 0x7f, 0xf9, 0x1a, 0xd0,
                          0x00}));

  ra::SequenceConfig compat = config;
  compat.ras_codes_refresh_frame_flags = true;
  const std::vector<std::uint8_t> ras_compat = ra::make_frame_obu(
      compat, make_ras(), std::span<const std::uint8_t>());
  CHECK(ras_compat == bytes({0x08, 0x54, 0xe2, 0x94, 0x5f, 0xfe, 0x46, 0xb4,
                             0x00}));

  ra::FrameHeader switch_open = make_switch();
  switch_open.restricted_prediction_switch = false;
  CHECK(ra::make_frame_obu(config, switch_open,
                           std::span<const std::uint8_t>()) ==
        bytes({0x08, 0x28, 0xe8, 0xc5, 0xff, 0xe4, 0x6b, 0x40, 0x00}));

  ra::ObuView obu;
  std::size_t consumed = 0;
  std::string error;
  CHECK(ra::parse_obu(ras_obu, obu, consumed, error));
  CHECK(obu.obu_type == static_cast<std::uint8_t>(ra::ObuType::kRas));

  ra::SequenceHeaderView view;
  const std::vector<std::uint8_t> seq = ra::make_sequence_header_obu(config);
  ra::ObuView seq_obu;
  CHECK(ra::parse_obu(seq, seq_obu, consumed, error));
  CHECK(ra::read_sequence_header(seq_obu.payload, view, error));

  struct Case {
    const char* name;
    std::vector<std::uint8_t> obu;
    ra::ObuType type;
    std::uint32_t order_hint;
    bool restricted;
  };
  const Case cases[] = {
      {"key", key_obu, ra::ObuType::kClosedLoopKey, 0, false},
      {"switch", switch_obu, ra::ObuType::kSwitch, 1, true},
      {"ras", ras_obu, ra::ObuType::kRas, 2, false},
  };
  for (const Case& item : cases) {
    ra::ObuView parsed;
    std::size_t used = 0;
    error.clear();
    CHECK(ra::parse_obu(item.obu, parsed, used, error));
    CHECK(parsed.obu_type == static_cast<std::uint8_t>(item.type));
    ra::FrameHeader header;
    std::size_t bits = 0;
    CHECK(ra::read_frame_header(parsed.payload, item.type, view, header, bits,
                                error));
    CHECK(error.empty());
    CHECK(header.order_hint == item.order_hint);
    CHECK(header.restricted_prediction_switch == item.restricted);
    CHECK((bits + 7) / 8 == parsed.payload.size());
    if (item.type == ra::ObuType::kClosedLoopKey) {
      CHECK(header.long_term_id_plus_1 == 1);
    } else if (item.type == ra::ObuType::kSwitch) {
      CHECK(header.num_total_refs == 1);
      CHECK(header.ref_frame_idx[0] == 0);
      CHECK(header.refresh_frame_flags == 2);
    } else {
      CHECK(header.num_key_ref_frames == 1);
      CHECK(header.ref_long_term_id[0] == 0);
      CHECK(header.num_total_refs == 1);
      CHECK(header.ref_frame_idx[0] == 0);
    }
  }

  ra::ObuView ras_view;
  CHECK(ra::parse_obu(ras_compat, ras_view, consumed, error));
  ra::FrameHeader compat_header;
  std::size_t compat_bits = 0;
  CHECK(ra::read_frame_header(ras_view.payload, ra::ObuType::kRas, view,
                              compat_header, compat_bits, error, true));
  CHECK(compat_header.refresh_frame_flags == 2);
}

void test_conformance() {
  const ra::SequenceConfig config = make_config(64, 64, 1);
  const ra::RefState refs = make_refs();

  const ra::ValidationResult valid_switch =
      ra::validate_header(config, make_switch(), refs);
  CHECK(valid_switch.ok);

  const ra::ValidationResult valid_ras = ra::validate_header(config, make_ras(),
                                                             refs);
  CHECK(valid_ras.ok);

  {
    ra::FrameHeader header = make_ras();
    header.ref_long_term_id[0] = 1;  // slot 0 holds long-term id 0
    const ra::ValidationResult result = ra::validate_header(config, header, refs);
    CHECK(!result.ok);
    CHECK(contains(result, ra::ViolationCode::kRasRefNotLongTerm));
  }
  {
    const ra::SequenceConfig no_lt = make_config(64, 64, 0);
    const ra::ValidationResult result =
        ra::validate_header(no_lt, make_ras(), refs);
    CHECK(!result.ok);
    CHECK(contains(result, ra::ViolationCode::kRasWithoutLongTermIds));
  }
  {
    ra::FrameHeader header = make_ras();
    header.order_hint = 16;  // 1 << OrderHintBits
    const ra::ValidationResult result = ra::validate_header(config, header, refs);
    CHECK(!result.ok);
    CHECK(contains(result, ra::ViolationCode::kRasOrderHintOutOfRange));
  }
  {
    const ra::ValidationResult result =
        ra::validate_header(config, make_switch(), refs, /*tlayer_id=*/1);
    CHECK(!result.ok);
    CHECK(contains(result, ra::ViolationCode::kTlayerNonZero));
  }
  {
    const ra::SequenceConfig bits2 = make_config(64, 64, 2);
    ra::RefState long_refs = refs;
    long_refs.slots[0].long_term_id = 3;  // (1 << 2) - 1 is reserved
    ra::FrameHeader header = make_ras();
    header.ref_long_term_id[0] = 3;
    const ra::ValidationResult result =
        ra::validate_header(bits2, header, long_refs);
    CHECK(!result.ok);
    CHECK(contains(result, ra::ViolationCode::kBadRefLongTermId));
  }
  {
    ra::FrameHeader header = make_switch();
    header.ref_frame_idx[0] = 5;  // NumRefFrames = 2
    const ra::ValidationResult result = ra::validate_header(config, header, refs);
    CHECK(!result.ok);
    CHECK(contains(result, ra::ViolationCode::kBadRefFrameIdx));
  }
  {
    ra::RefState big = refs;
    big.slots[0].width = 1024;
    big.slots[0].height = 1024;
    const ra::ValidationResult result =
        ra::validate_header(config, make_switch(), big);
    CHECK(!result.ok);
    CHECK(contains(result, ra::ViolationCode::kReferenceDimensions));
  }
  {
    ra::FrameHeader header = make_key();
    header.long_term_id_plus_1 = 2;  // field width is 1 bit
    const ra::ValidationResult result = ra::validate_header(config, header, refs);
    CHECK(!result.ok);
    CHECK(contains(result, ra::ViolationCode::kKeyLongTermIdReserved));
  }
  {
    ra::FrameHeader header = make_ras();
    header.refresh_frame_flags = 0x1;  // derived mask is 0x2
    const ra::ValidationResult result = ra::validate_header(config, header, refs);
    CHECK(!result.ok);
    CHECK(contains(result, ra::ViolationCode::kRasRefreshMismatch));
  }
}

void test_encoder_strategy() {
  const ra::SequenceConfig config = make_config(64, 64, 1);
  const ra::RefState refs = make_refs();
  const std::uint8_t only_zero[] = {0};

  ra::PFrameCandidate candidate;
  candidate.reference_slot = 0;
  candidate.refresh_frame_flags = 2;
  candidate.order_hint = 1;
  candidate.has_inter_blocks = true;

  const ra::PayloadPlan sw =
      ra::plan_switch_frame(config, refs, candidate, true);
  CHECK(sw.decision == ra::PayloadDecision::kReusePFrame);
  CHECK(sw.header.obu_type == ra::ObuType::kSwitch);
  CHECK(sw.header.restricted_prediction_switch);
  CHECK(sw.header.refresh_frame_flags == 2);

  ra::PFrameCandidate bad_switch = candidate;
  bad_switch.reference_slot = 6;
  CHECK(ra::plan_switch_frame(config, refs, bad_switch, true).decision ==
        ra::PayloadDecision::kReencodeRequired);

  candidate.order_hint = 2;
  const ra::PayloadPlan ras =
      ra::plan_ras_frame(config, refs, candidate, only_zero);
  CHECK(ras.decision == ra::PayloadDecision::kReusePFrame);
  CHECK(ras.header.obu_type == ra::ObuType::kRas);
  CHECK(ras.header.num_key_ref_frames == 1);
  CHECK(ras.header.ref_long_term_id[0] == 0);
  CHECK(ras.header.num_total_refs == 1);
  CHECK(ras.header.ref_frame_idx[0] == 0);
  CHECK(ras.header.refresh_frame_flags == 2);

  candidate.has_inter_blocks = false;
  const ra::PayloadPlan ras_intra =
      ra::plan_ras_frame(config, refs, candidate, only_zero);
  CHECK(ras_intra.decision == ra::PayloadDecision::kReusePFrame);
  CHECK(ras_intra.header.num_total_refs == 1);

  const std::uint8_t wrong_id[] = {1};
  CHECK(ra::plan_ras_frame(config, refs, candidate, wrong_id).decision ==
        ra::PayloadDecision::kReencodeRequired);

  const ra::SequenceConfig no_lt = make_config(64, 64, 0);
  CHECK(ra::plan_ras_frame(no_lt, refs, candidate, only_zero).decision ==
        ra::PayloadDecision::kReencodeRequired);
}

void test_reader_rejects_truncated() {
  const ra::SequenceConfig config = make_config(64, 64, 1);
  const std::vector<std::uint8_t> ras = ra::make_frame_obu(
      config, make_ras(), std::span<const std::uint8_t>());
  ra::ObuView obu;
  std::size_t consumed = 0;
  std::string error;
  CHECK(ra::parse_obu(ras, obu, consumed, error));
  ra::SequenceHeaderView view;
  const std::vector<std::uint8_t> seq = ra::make_sequence_header_obu(config);
  ra::ObuView seq_obu;
  CHECK(ra::parse_obu(seq, seq_obu, consumed, error));
  CHECK(ra::read_sequence_header(seq_obu.payload, view, error));

  ra::FrameHeader header;
  std::size_t bits = 0;
  error.clear();
  CHECK(!ra::read_frame_header(obu.payload.first(2), ra::ObuType::kRas, view,
                               header, bits, error));
  CHECK(!error.empty());
}

}  // namespace

int main() {
  test_frame_type_derivation();
  test_reference_sets();
  test_sequence_header();
  test_frame_header_goldens();
  test_conformance();
  test_encoder_strategy();
  test_reader_rejects_truncated();

  std::printf("%d checks, %d failure(s)\n", g_checks, g_failures);
  std::printf("%s\n", g_failures == 0 ? "RANDOM ACCESS TEST PASS"
                                      : "RANDOM ACCESS TEST FAIL");
  return g_failures == 0 ? 0 : 1;
}
