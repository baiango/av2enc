// Tests for the §5.4.7 / §5.18.3.3 / §5.18.3.4 screen-content signalling
// module.  Every syntax branch has a pinned bit-string golden and a pinned byte
// golden; derived rules and the still-picture assembler are covered separately.
#include "encoder/m1/scc_flags/scc_flags.h"

#include "encoder/m1/m1_headers.h"
#include "encoder/m1/scc_flags/scc_flags_bridge.h"
#include "encoder/partition/partition.h"

#include <cstdio>
#include <string>
#include <vector>

namespace scc = av2enc::m1::scc_flags;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

std::string hex(const std::vector<std::uint8_t>& bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  for (std::uint8_t b : bytes) {
    out.push_back(kDigits[b >> 4]);
    out.push_back(kDigits[b & 15]);
  }
  return out;
}

std::string bits_of(const av2enc::BitWriter& writer) {
  std::string out;
  const auto bytes = writer.bytes();
  for (std::size_t i = 0; i < writer.bit_size(); ++i) {
    const unsigned shift = 7U - static_cast<unsigned>(i & 7U);
    out.push_back(((bytes[i >> 3U] >> shift) & 1U) != 0U ? '1' : '0');
  }
  return out;
}

std::vector<std::uint8_t> bytes_of(const av2enc::BitWriter& writer) {
  const auto span = writer.bytes();
  return std::vector<std::uint8_t>(span.begin(), span.end());
}

// ---------------------------------------------------------------------------
// §5.4.7 sequence branches
// ---------------------------------------------------------------------------

struct SeqCase {
  const char* name;
  scc::SequenceSccConfig config;
  const char* bits;
  const char* bytes;
  int force_screen;
  int force_integer;
};

std::vector<SeqCase> seq_cases() {
  std::vector<SeqCase> cases;

  {
    SeqCase c;
    c.name = "single-picture (inferred SELECT/SELECT)";
    c.config.single_picture_header_flag = true;
    c.bits = "";
    c.bytes = "";
    c.force_screen = 2;
    c.force_integer = 2;
    cases.push_back(c);
  }
  {
    SeqCase c;
    c.name = "choose screen + choose integer (SELECT/SELECT)";
    c.config.single_picture_header_flag = false;
    c.config.seq_choose_screen_content_tools = true;
    c.config.seq_choose_integer_mv = true;
    c.bits = "11";
    c.bytes = "c0";
    c.force_screen = 2;
    c.force_integer = 2;
    cases.push_back(c);
  }
  {
    SeqCase c;
    c.name = "choose screen + force fractional (SELECT/0)";
    c.config.single_picture_header_flag = false;
    c.config.seq_choose_screen_content_tools = true;
    c.config.seq_choose_integer_mv = false;
    c.config.seq_force_integer_mv = false;
    c.bits = "100";
    c.bytes = "80";
    c.force_screen = 2;
    c.force_integer = 0;
    cases.push_back(c);
  }
  {
    SeqCase c;
    c.name = "choose screen + force integer (SELECT/1)";
    c.config.single_picture_header_flag = false;
    c.config.seq_choose_screen_content_tools = true;
    c.config.seq_choose_integer_mv = false;
    c.config.seq_force_integer_mv = true;
    c.bits = "101";
    c.bytes = "a0";
    c.force_screen = 2;
    c.force_integer = 1;
    cases.push_back(c);
  }
  {
    SeqCase c;
    c.name = "force screen tools off (0 / inferred SELECT)";
    c.config.single_picture_header_flag = false;
    c.config.seq_choose_screen_content_tools = false;
    c.config.seq_force_screen_content_tools = false;
    c.bits = "00";
    c.bytes = "00";
    c.force_screen = 0;
    c.force_integer = 2;
    cases.push_back(c);
  }
  {
    SeqCase c;
    c.name = "force screen on + choose integer (1/SELECT)";
    c.config.single_picture_header_flag = false;
    c.config.seq_choose_screen_content_tools = false;
    c.config.seq_force_screen_content_tools = true;
    c.config.seq_choose_integer_mv = true;
    c.bits = "011";
    c.bytes = "60";
    c.force_screen = 1;
    c.force_integer = 2;
    cases.push_back(c);
  }
  {
    SeqCase c;
    c.name = "force screen on + force fractional (1/0)";
    c.config.single_picture_header_flag = false;
    c.config.seq_choose_screen_content_tools = false;
    c.config.seq_force_screen_content_tools = true;
    c.config.seq_choose_integer_mv = false;
    c.config.seq_force_integer_mv = false;
    c.bits = "0100";
    c.bytes = "40";
    c.force_screen = 1;
    c.force_integer = 0;
    cases.push_back(c);
  }
  {
    SeqCase c;
    c.name = "force screen on + force integer (1/1)";
    c.config.single_picture_header_flag = false;
    c.config.seq_choose_screen_content_tools = false;
    c.config.seq_force_screen_content_tools = true;
    c.config.seq_choose_integer_mv = false;
    c.config.seq_force_integer_mv = true;
    c.bits = "0101";
    c.bytes = "50";
    c.force_screen = 1;
    c.force_integer = 1;
    cases.push_back(c);
  }
  return cases;
}

void test_sequence_branches() {
  for (const SeqCase& c : seq_cases()) {
    std::string error;
    av2enc::BitWriter writer;
    scc::SequenceSccState state;
    std::size_t bits = 0;
    const bool ok = scc::write_sequence_scc_config(writer, c.config, &state,
                                                   &bits, error);
    check(ok, std::string(c.name) + ": writes (" + error + ")");
    if (!ok) continue;
    check(bits_of(writer) == c.bits, std::string(c.name) + ": bit golden");
    check(hex(bytes_of(writer)) == c.bytes,
          std::string(c.name) + ": byte golden");
    check(bits == std::string(c.bits).size(),
          std::string(c.name) + ": bit count");
    check(state.seq_force_screen_content_tools == c.force_screen,
          std::string(c.name) + ": force_screen_content_tools");
    check(state.seq_force_integer_mv == c.force_integer,
          std::string(c.name) + ": force_integer_mv");

    const scc::SequenceSccState derived =
        scc::derive_sequence_scc_state(c.config);
    check(derived.seq_force_screen_content_tools == c.force_screen &&
              derived.seq_force_integer_mv == c.force_integer,
          std::string(c.name) + ": derive == write");

    // Independent reader round-trip.
    const std::vector<std::uint8_t> written = bytes_of(writer);
    scc::BitReader reader(written);
    scc::SequenceSccState read;
    check(scc::read_sequence_scc_config(
              reader, c.config.single_picture_header_flag, read, error),
          std::string(c.name) + ": reads (" + error + ")");
    check(read.seq_force_screen_content_tools == c.force_screen,
          std::string(c.name) + ": reader force_screen");
    check(read.seq_force_integer_mv == c.force_integer,
          std::string(c.name) + ": reader force_integer");
    check(read.coded_seq_choose_screen_content_tools ==
              derived.coded_seq_choose_screen_content_tools,
          std::string(c.name) + ": reader choose_screen presence");
    check(read.coded_seq_force_screen_content_tools ==
              derived.coded_seq_force_screen_content_tools,
          std::string(c.name) + ": reader force_screen presence");
    check(read.coded_seq_choose_integer_mv ==
              derived.coded_seq_choose_integer_mv,
          std::string(c.name) + ": reader choose_integer presence");
    check(read.coded_seq_force_integer_mv ==
              derived.coded_seq_force_integer_mv,
          std::string(c.name) + ": reader force_integer presence");
    check(reader.bit_pos() == bits,
          std::string(c.name) + ": reader consumed exactly the syntax");
  }
}

void test_sequence_validation() {
  std::string error;
  {
    scc::SequenceSccConfig c;
    c.single_picture_header_flag = true;
    c.seq_force_screen_content_tools = true;
    check(!scc::validate_sequence_scc_config(c, error),
          "single picture rejects an explicit force field");
  }
  {
    scc::SequenceSccConfig c;
    c.single_picture_header_flag = false;
    c.seq_choose_screen_content_tools = true;
    c.seq_force_screen_content_tools = true;
    check(!scc::validate_sequence_scc_config(c, error),
          "choose_screen and force_screen are exclusive");
  }
  {
    scc::SequenceSccConfig c;
    c.single_picture_header_flag = false;
    c.seq_force_integer_mv = true;
    check(!scc::validate_sequence_scc_config(c, error),
          "integer fields are absent when force_screen=0");
  }
  {
    scc::SequenceSccConfig c;
    c.single_picture_header_flag = false;
    c.seq_force_screen_content_tools = true;
    c.seq_choose_integer_mv = true;
    c.seq_force_integer_mv = true;
    check(!scc::validate_sequence_scc_config(c, error),
          "choose_integer and force_integer are exclusive");
  }
  for (const SeqCase& c : seq_cases()) {
    check(scc::validate_sequence_scc_config(c.config, error),
          std::string(c.name) + ": valid");
  }
}

// ---------------------------------------------------------------------------
// §5.18.3.3 / §5.18.3.4 frame branches
// ---------------------------------------------------------------------------

struct FrameCase {
  const char* name;
  bool single_picture;
  scc::SequenceSccConfig sequence;
  scc::FrameSccContext context;
  scc::FrameSccConfig frame;
  int seq_max_bvp = 0;
  bool allow_frame_max = false;
  const char* bits;
  const char* bytes;
  bool allow_screen = false;
  bool force_integer = false;
  bool allow_intrabc = false;
  bool allow_global = false;
  bool allow_local = false;
  int max_bvp = 0;
};

std::vector<FrameCase> frame_cases() {
  std::vector<FrameCase> cases;

  auto add = [&](const char* name, scc::FrameSccConfig frame,
                 const char* bits, const char* bytes) {
    FrameCase c;
    c.name = name;
    c.single_picture = true;
    c.frame = frame;
    c.bits = bits;
    c.bytes = bytes;
    c.allow_screen = frame.allow_screen_content_tools;
    c.force_integer = frame.force_integer_mv;
    c.allow_intrabc = frame.allow_intrabc;
    c.allow_global = frame.allow_global_intrabc;
    c.allow_local = frame.allow_local_intrabc;
    c.max_bvp = 0;
    cases.push_back(c);
  };

  {
    scc::FrameSccConfig f;
    add("all off", f, "00", "00");
  }
  {
    scc::FrameSccConfig f;
    f.allow_screen_content_tools = true;
    f.force_integer_mv = false;
    add("screen tools on, fractional MVs", f, "100", "80");
  }
  {
    scc::FrameSccConfig f;
    f.allow_screen_content_tools = true;
    f.force_integer_mv = true;
    add("screen tools on, integer MVs", f, "110", "c0");
  }
  {
    scc::FrameSccConfig f;
    f.allow_screen_content_tools = true;
    f.force_integer_mv = true;
    f.allow_intrabc = true;
    f.allow_global_intrabc = true;
    f.allow_local_intrabc = false;
    add("intrabc global-only", f, "11110", "f0");
  }
  {
    scc::FrameSccConfig f;
    f.allow_screen_content_tools = true;
    f.force_integer_mv = true;
    f.allow_intrabc = true;
    f.allow_global_intrabc = false;
    f.allow_local_intrabc = true;
    add("intrabc local inferred (global=0)", f, "1110", "e0");
  }
  {
    scc::FrameSccConfig f;
    f.allow_screen_content_tools = true;
    f.force_integer_mv = true;
    f.allow_intrabc = true;
    f.allow_global_intrabc = true;
    f.allow_local_intrabc = true;
    add("intrabc global+local", f, "11111", "f8");
  }

  // Inter frame that still parses the SCC region.
  {
    FrameCase c;
    c.name = "inter frame, SCC present, global/local inferred";
    c.single_picture = true;
    c.context.frame_is_intra = false;
    c.frame.allow_screen_content_tools = true;
    c.frame.force_integer_mv = true;
    c.frame.allow_intrabc = true;
    c.frame.allow_local_intrabc = true;
    c.bits = "111";
    c.bytes = "e0";
    c.allow_screen = true;
    c.force_integer = true;
    c.allow_intrabc = true;
    c.allow_global = false;
    c.allow_local = true;
    c.max_bvp = 0;
    cases.push_back(c);
  }
  // Inter frame where §5.18.2 omits the whole region (TIP as output).
  {
    FrameCase c;
    c.name = "inter TIP-as-output, SCC absent";
    c.single_picture = true;
    c.context.frame_is_intra = false;
    c.context.tip_frame_as_output = true;
    c.bits = "";
    c.bytes = "";
    c.max_bvp = 0;
    cases.push_back(c);
  }
  {
    FrameCase c;
    c.name = "inter BRU-inactive, SCC absent";
    c.single_picture = true;
    c.context.frame_is_intra = false;
    c.context.bru_inactive = true;
    c.bits = "";
    c.bytes = "";
    c.max_bvp = 0;
    cases.push_back(c);
  }

  // Sentence forced by seq_force_screen_content_tools == 1.
  {
    FrameCase c;
    c.name = "seq forces screen tools on and integer MVs";
    c.single_picture = false;
    c.sequence.single_picture_header_flag = false;
    c.sequence.seq_force_screen_content_tools = true;
    c.sequence.seq_choose_integer_mv = false;
    c.sequence.seq_force_integer_mv = true;
    c.frame.allow_screen_content_tools = true;
    c.frame.force_integer_mv = true;
    c.bits = "0";
    c.bytes = "00";
    c.allow_screen = true;
    c.force_integer = true;
    cases.push_back(c);
  }
  {
    FrameCase c;
    c.name = "seq forces screen tools on and fractional MVs";
    c.single_picture = false;
    c.sequence.single_picture_header_flag = false;
    c.sequence.seq_force_screen_content_tools = true;
    c.sequence.seq_choose_integer_mv = false;
    c.sequence.seq_force_integer_mv = false;
    c.frame.allow_screen_content_tools = true;
    c.frame.force_integer_mv = false;
    c.bits = "0";
    c.bytes = "00";
    c.allow_screen = true;
    c.force_integer = false;
    cases.push_back(c);
  }
  // seq_force_screen_content_tools == 0: both values inferred.
  {
    FrameCase c;
    c.name = "seq forces screen tools off; intrabc local inferred";
    c.single_picture = false;
    c.sequence.single_picture_header_flag = false;
    c.sequence.seq_force_screen_content_tools = false;
    c.frame.allow_intrabc = true;
    c.frame.allow_local_intrabc = true;
    c.bits = "10";
    c.bytes = "80";
    c.allow_screen = false;
    c.force_integer = false;
    c.allow_intrabc = true;
    c.allow_global = false;
    c.allow_local = true;
    cases.push_back(c);
  }
  // seq_force_integer_mv == 0 forces the frame bit away.
  {
    FrameCase c;
    c.name = "seq forces fractional MVs; frame bit absent";
    c.single_picture = false;
    c.sequence.single_picture_header_flag = false;
    c.sequence.seq_choose_screen_content_tools = true;
    c.sequence.seq_choose_integer_mv = false;
    c.sequence.seq_force_integer_mv = false;
    c.frame.allow_screen_content_tools = true;
    c.frame.force_integer_mv = false;
    c.bits = "10";
    c.bytes = "80";
    c.allow_screen = true;
    c.force_integer = false;
    cases.push_back(c);
  }
  {
    FrameCase c;
    c.name = "seq forces integer MVs; frame bit absent";
    c.single_picture = false;
    c.sequence.single_picture_header_flag = false;
    c.sequence.seq_choose_screen_content_tools = true;
    c.sequence.seq_choose_integer_mv = false;
    c.sequence.seq_force_integer_mv = true;
    c.frame.allow_screen_content_tools = true;
    c.frame.force_integer_mv = true;
    c.bits = "10";
    c.bytes = "80";
    c.allow_screen = true;
    c.force_integer = true;
    cases.push_back(c);
  }

  // change_bvp_drl branches.  seq_max=0: ns(2) 0 -> max 1, ns(2) 1 -> max 2.
  for (int target = 1; target <= 2; ++target) {
    FrameCase c;
    c.name = target == 1 ? "change_bvp_drl seq0 -> 1"
                         : "change_bvp_drl seq0 -> 2";
    c.single_picture = true;
    c.seq_max_bvp = 0;
    c.allow_frame_max = true;
    c.frame.allow_screen_content_tools = true;
    c.frame.force_integer_mv = true;
    c.frame.allow_intrabc = true;
    c.frame.allow_local_intrabc = true;
    c.frame.change_bvp_drl = true;
    c.frame.max_bvp_drl_bits_minus_1 = target;
    c.bits = target == 1 ? "111010" : "111011";
    c.bytes = target == 1 ? "e8" : "ec";
    c.allow_screen = true;
    c.force_integer = true;
    c.allow_intrabc = true;
    c.allow_local = true;
    c.max_bvp = target;
    cases.push_back(c);
  }
  // seq_max=1: ns 0 -> 0, ns 1 -> 2.
  for (int target : {0, 2}) {
    FrameCase c;
    c.name = target == 0 ? "change_bvp_drl seq1 -> 0"
                         : "change_bvp_drl seq1 -> 2";
    c.single_picture = true;
    c.seq_max_bvp = 1;
    c.allow_frame_max = true;
    c.frame.allow_screen_content_tools = true;
    c.frame.force_integer_mv = true;
    c.frame.allow_intrabc = true;
    c.frame.allow_local_intrabc = true;
    c.frame.change_bvp_drl = true;
    c.frame.max_bvp_drl_bits_minus_1 = target;
    c.bits = target == 0 ? "111010" : "111011";
    c.bytes = target == 0 ? "e8" : "ec";
    c.allow_screen = true;
    c.force_integer = true;
    c.allow_intrabc = true;
    c.allow_local = true;
    c.max_bvp = target;
    cases.push_back(c);
  }
  // allow_frame_max_bvp_drl_bits=1 with change=0.
  {
    FrameCase c;
    c.name = "allow_frame_max set, change_bvp_drl=0";
    c.single_picture = true;
    c.seq_max_bvp = 0;
    c.allow_frame_max = true;
    c.frame.allow_screen_content_tools = true;
    c.frame.force_integer_mv = true;
    c.frame.allow_intrabc = true;
    c.frame.allow_local_intrabc = true;
    c.bits = "11100";
    c.bytes = "e0";
    c.allow_screen = true;
    c.force_integer = true;
    c.allow_intrabc = true;
    c.allow_local = true;
    c.max_bvp = 0;
    cases.push_back(c);
  }
  return cases;
}

void test_frame_branches() {
  const std::vector<std::uint8_t> empty;
  for (const FrameCase& c : frame_cases()) {
    std::string error;
    scc::SequenceSccState sequence;
    if (c.sequence.single_picture_header_flag) {
      scc::BitReader reader(empty);
      if (!scc::read_sequence_scc_config(reader, true, sequence, error)) {
        check(false, std::string(c.name) + ": sequence setup");
        continue;
      }
    } else {
      av2enc::BitWriter seq_writer;
      std::size_t seq_bits = 0;
      check(scc::write_sequence_scc_config(seq_writer, c.sequence, &sequence,
                                          &seq_bits, error),
            std::string(c.name) + ": sequence writes");
    }

    av2enc::BitWriter writer;
    scc::FrameSccState state;
    std::size_t bits = 0;
    const bool ok = scc::write_frame_scc_params(
        writer, sequence, c.context, c.frame, c.seq_max_bvp,
        c.allow_frame_max, &state, &bits, error);
    check(ok, std::string(c.name) + ": writes (" + error + ")");
    if (!ok) continue;
    check(bits_of(writer) == c.bits, std::string(c.name) + ": bit golden");
    check(hex(bytes_of(writer)) == c.bytes,
          std::string(c.name) + ": byte golden");
    check(state.allow_screen_content_tools == c.allow_screen,
          std::string(c.name) + ": allow_screen_content_tools");
    check(state.force_integer_mv == c.force_integer,
          std::string(c.name) + ": force_integer_mv");
    check(state.allow_intrabc == c.allow_intrabc,
          std::string(c.name) + ": allow_intrabc");
    check(state.allow_global_intrabc == c.allow_global,
          std::string(c.name) + ": allow_global_intrabc");
    check(state.allow_local_intrabc == c.allow_local,
          std::string(c.name) + ": allow_local_intrabc");
    check(state.max_bvp_drl_bits_minus_1 == c.max_bvp,
          std::string(c.name) + ": max_bvp_drl_bits_minus_1");
    check(state.IsIntraBC == c.allow_intrabc,
          std::string(c.name) + ": IsIntraBC");

    const scc::FrameSccState derived = scc::derive_frame_scc_state(
        sequence, c.context, c.frame, c.seq_max_bvp, c.allow_frame_max);
    check(derived.allow_screen_content_tools == state.allow_screen_content_tools &&
              derived.force_integer_mv == state.force_integer_mv &&
              derived.allow_intrabc == state.allow_intrabc &&
              derived.allow_global_intrabc == state.allow_global_intrabc &&
              derived.allow_local_intrabc == state.allow_local_intrabc &&
              derived.max_bvp_drl_bits_minus_1 ==
                  state.max_bvp_drl_bits_minus_1,
          std::string(c.name) + ": derive == write");

    const std::vector<std::uint8_t> written = bytes_of(writer);
    scc::BitReader reader(written);
    scc::FrameSccState read;
    check(scc::read_frame_scc_params(reader, sequence, c.context,
                                     c.seq_max_bvp, c.allow_frame_max, read,
                                     error),
          std::string(c.name) + ": reads (" + error + ")");
    check(read.allow_screen_content_tools == c.allow_screen &&
              read.force_integer_mv == c.force_integer &&
              read.allow_intrabc == c.allow_intrabc &&
              read.allow_global_intrabc == c.allow_global &&
              read.allow_local_intrabc == c.allow_local &&
              read.max_bvp_drl_bits_minus_1 == c.max_bvp,
          std::string(c.name) + ": reader round-trip");
    check(reader.bit_pos() == bits,
          std::string(c.name) + ": reader consumed exactly the syntax");

    // Presence record must match the branch.
    const bool expect_allow_screen_bit =
        state.scc_present &&
        sequence.seq_force_screen_content_tools ==
            scc::kSelectScreenContentTools;
    check(read.coded_allow_screen_content_tools == expect_allow_screen_bit,
          std::string(c.name) + ": allow_screen presence");
    const bool expect_force_bit =
        expect_allow_screen_bit && state.allow_screen_content_tools &&
        sequence.seq_force_integer_mv == scc::kSelectIntegerMv;
    check(read.coded_force_integer_mv == expect_force_bit,
          std::string(c.name) + ": force_integer presence");
  }
}

void test_frame_validation() {
  std::string error;
  scc::SequenceSccState single;
  single.seq_force_screen_content_tools = scc::kSelectScreenContentTools;
  single.seq_force_integer_mv = scc::kSelectIntegerMv;

  {
    scc::FrameSccConfig f;
    f.allow_screen_content_tools = true;
    f.force_integer_mv = true;
    scc::FrameSccContext tip;
    tip.frame_is_intra = false;
    tip.tip_frame_as_output = true;
    check(!scc::validate_frame_scc_config(single, tip, f, 0, false, error),
          "absent SCC region rejects active flags");
  }
  {
    scc::FrameSccConfig f;
    f.force_integer_mv = true;
    scc::FrameSccContext intra;
    check(!scc::validate_frame_scc_config(single, intra, f, 0, false, error),
          "force_integer_mv requires allow_screen_content_tools");
  }
  {
    scc::FrameSccConfig f;
    f.allow_screen_content_tools = true;
    f.force_integer_mv = true;
    f.allow_intrabc = true;
    f.allow_global_intrabc = true;
    f.allow_local_intrabc = false;
    scc::FrameSccContext inter;
    inter.frame_is_intra = false;
    check(!scc::validate_frame_scc_config(single, inter, f, 0, false, error),
          "inter frame rejects allow_global_intrabc");
  }
  {
    scc::FrameSccConfig f;
    f.allow_screen_content_tools = true;
    f.force_integer_mv = true;
    f.allow_intrabc = true;
    f.allow_global_intrabc = false;
    f.allow_local_intrabc = false;
    scc::FrameSccContext intra;
    check(!scc::validate_frame_scc_config(single, intra, f, 0, false, error),
          "global=0 infers local=1");
  }
  {
    scc::FrameSccConfig f;
    f.allow_screen_content_tools = true;
    f.force_integer_mv = true;
    f.allow_intrabc = false;
    f.allow_global_intrabc = true;
    scc::FrameSccContext intra;
    check(!scc::validate_frame_scc_config(single, intra, f, 0, false, error),
          "IBC regions are absent when allow_intrabc=0");
  }
  {
    scc::FrameSccConfig f;
    f.allow_screen_content_tools = true;
    f.force_integer_mv = true;
    f.allow_intrabc = true;
    f.allow_local_intrabc = true;
    f.max_bvp_drl_bits_minus_1 = 1;
    scc::FrameSccContext intra;
    check(!scc::validate_frame_scc_config(single, intra, f, 0, false, error),
          "max_bvp cannot change when allow_frame_max is 0");
  }
  {
    scc::FrameSccConfig f;
    f.allow_screen_content_tools = true;
    f.force_integer_mv = true;
    f.allow_intrabc = true;
    f.allow_local_intrabc = true;
    f.change_bvp_drl = true;
    f.max_bvp_drl_bits_minus_1 = 0;  // unreachable from seq 0
    scc::FrameSccContext intra;
    check(!scc::validate_frame_scc_config(single, intra, f, 0, true, error),
          "change_bvp_drl rejects an unreachable target");
  }
  {
    // seq_force_screen_content_tools=0 forces allow off.
    scc::SequenceSccState off;
    off.seq_force_screen_content_tools = 0;
    off.seq_force_integer_mv = scc::kSelectIntegerMv;
    scc::FrameSccConfig f;
    f.allow_screen_content_tools = true;
    scc::FrameSccContext intra;
    check(!scc::validate_frame_scc_config(off, intra, f, 0, false, error),
          "seq force-off rejects allow=1");
  }
  {
    // seq_force_integer_mv=0 forbids force=1.
    scc::SequenceSccState frac;
    frac.seq_force_screen_content_tools = 1;
    frac.seq_force_integer_mv = 0;
    scc::FrameSccConfig f;
    f.allow_screen_content_tools = true;
    f.force_integer_mv = true;
    scc::FrameSccContext intra;
    check(!scc::validate_frame_scc_config(frac, intra, f, 0, false, error),
          "seq force-fractional rejects force_integer_mv=1");
  }
}

// ---------------------------------------------------------------------------
// Derived legality and MV precision
// ---------------------------------------------------------------------------

void test_legality() {
  scc::FrameSccState state;
  state.scc_present = true;
  state.allow_screen_content_tools = true;
  state.force_integer_mv = true;
  state.allow_intrabc = true;
  state.allow_local_intrabc = true;
  state.IsIntraBC = true;

  // Palette: luma only, DC only, MiSize >= BLOCK_8X8, <= 64 samples.
  check(scc::palette_allowed_for_block(state, 0, 0, 3, 8, 8),
        "palette 8x8 luma DC");
  check(!scc::palette_allowed_for_block(state, 1, 0, 3, 8, 8),
        "palette is luma-only");
  check(!scc::palette_allowed_for_block(state, 0, 1, 3, 8, 8),
        "palette needs YMode DC_PRED");
  check(!scc::palette_allowed_for_block(state, 0, 0, 2, 8, 4),
        "4x8 is below BLOCK_8X8");
  check(!scc::palette_allowed_for_block(state, 0, 0, 13, 64, 128),
        "palette height > 64");
  check(!scc::palette_allowed_for_block(state, 0, 0, 14, 128, 64),
        "palette width > 64");
  check(scc::palette_allowed_for_block(state, 0, 0, 19, 4, 16),
        "4x16 is MiSize >= BLOCK_8X8");
  check(!scc::palette_allowed_for_block(state, 0, 0, 255, 8, 8),
        "invalid block size is rejected");
  {
    scc::FrameSccState off = state;
    off.allow_screen_content_tools = false;
    check(!scc::palette_allowed_for_block(off, 0, 0, 3, 8, 8),
          "palette needs allow_screen_content_tools");
  }

  // IntraBC: not 64x64, not >64, not a chroma part.
  check(!scc::intrabc_allowed_for_block(state, 12, 64, 64, false),
        "IBC excludes BLOCK_64X64");
  check(scc::intrabc_allowed_for_block(state, 11, 64, 32, false),
        "IBC allows 64x32");
  check(scc::intrabc_allowed_for_block(state, 0, 4, 4, false),
        "IBC allows 4x4");
  check(!scc::intrabc_allowed_for_block(state, 11, 64, 32, true),
        "IBC excludes CHROMA_PART");
  check(!scc::intrabc_allowed_for_block(state, 13, 64, 128, false),
        "IBC excludes >64 height");
  check(!scc::intrabc_allowed_for_block(state, 255, 8, 8, false),
        "IBC rejects an invalid block size");
  check(!scc::intrabc_allowed_for_block(state, 11, 64, 32, false, false,
                                        false),
        "inter IBC needs MIXED_REGION");
  {
    scc::FrameSccState off = state;
    off.allow_intrabc = false;
    check(!scc::intrabc_allowed_for_block(off, 11, 64, 32, false),
          "IBC needs allow_intrabc");
  }

  // morph_pred is intra + screen tools + BAWP.
  check(scc::morph_pred_allowed(state, true, true),
        "morph_pred allowed for intra + screen + BAWP");
  check(!scc::morph_pred_allowed(state, true, false),
        "morph_pred needs enable_bawp");
  check(!scc::morph_pred_allowed(state, false, true),
        "morph_pred is intra-only");

  // MV precision derivation.
  check(scc::intrabc_mv_precision(state) == scc::kMvPrecisionOnePel,
        "force_integer_mv -> one-pel IBC precision");
  scc::FrameSccState frac = state;
  frac.force_integer_mv = false;
  check(scc::intrabc_mv_precision(frac) == scc::kMvPrecisionQuarterPel,
        "fractional MVs -> quarter-pel IBC precision");

  scc::SequenceSccState seq;
  scc::FrameSccContext intra;
  scc::FrameSccConfig cfg;
  cfg.allow_screen_content_tools = true;
  cfg.force_integer_mv = true;
  scc::FrameSccState forced =
      scc::derive_frame_scc_state(seq, intra, cfg, 0, false);
  check(!forced.codes_frame_mv_precision,
        "intra frame never codes FrameMvPrecision");
  check(forced.frame_mv_precision == scc::kMvPrecisionOnePel,
        "intra FrameMvPrecision is ONE_PEL");
  check(!forced.intrabc_precision_coded,
        "force_integer_mv removes intrabc_precision");

  scc::FrameSccContext inter;
  inter.frame_is_intra = false;
  scc::FrameSccConfig inter_cfg;
  inter_cfg.allow_screen_content_tools = true;
  inter_cfg.force_integer_mv = true;
  scc::FrameSccState inter_forced =
      scc::derive_frame_scc_state(seq, inter, inter_cfg, 0, false);
  check(!inter_forced.codes_frame_mv_precision,
        "force_integer_mv removes the inter frame-precision bits");
  check(inter_forced.frame_mv_precision == scc::kMvPrecisionOnePel,
        "inter forced frame precision is ONE_PEL");

  inter_cfg.force_integer_mv = false;
  inter_cfg.allow_intrabc = true;
  inter_cfg.allow_local_intrabc = true;
  scc::FrameSccState inter_frac =
      scc::derive_frame_scc_state(seq, inter, inter_cfg, 0, false);
  check(inter_frac.codes_frame_mv_precision,
        "fractional inter frame codes use_qtr/allow_high_precision_mv");
  check(inter_frac.intrabc_precision_coded,
        "fractional IBC codes intrabc_precision for NEWMV");
}

// ---------------------------------------------------------------------------
// Still-picture assembly and byte identity
// ---------------------------------------------------------------------------

std::vector<std::vector<std::uint8_t>> tile_payloads() {
  return {
      std::vector<std::uint8_t>{0x01},
      std::vector<std::uint8_t>{0xde, 0xad, 0xbe, 0xef},
      std::vector<std::uint8_t>(static_cast<std::size_t>(97), 0x5a),
  };
}

av2enc::m1::M1Config m1_config(int width, int height, bool deblock) {
  av2enc::m1::M1Config config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = 90;
  config.disable_cdf_update = true;
  config.enable_intra_edge_filter = false;
  config.enable_cfl_intra = false;
  config.enable_chroma_dctonly = true;
  config.apply_deblock_y_vert = deblock;
  config.apply_deblock_y_horz = deblock;
  config.apply_deblock_u = deblock;
  config.apply_deblock_v = deblock;
  return config;
}

void test_still_byte_identity() {
  for (int deblock = 0; deblock <= 1; ++deblock) {
    for (const auto& tile : tile_payloads()) {
      for (const auto& dims : {std::pair<int, int>{64, 64},
                               std::pair<int, int>{128, 64}}) {
        const av2enc::m1::M1Config config =
            m1_config(dims.first, dims.second, deblock != 0);
        scc::StillSccConfig scc_cfg;
        scc_cfg.sequence.single_picture_header_flag = true;
        scc::StillSccResult result;
        std::string error;
        const bool ok =
            scc::assemble_still_picture(config, scc_cfg, 0, false, tile,
                                        result, error);
        check(ok, "assemble still all-off (" + error + ")");
        if (!ok) continue;
        const std::vector<std::uint8_t> reference =
            av2enc::m1::assemble_m1_still_picture(config, tile);
        check(hex(result.frame) == hex(reference),
              "all-off still is byte-identical to M1 (deblock=" +
                  std::to_string(deblock) + ", " +
                  std::to_string(dims.first) + "x" +
                  std::to_string(dims.second) + ")");
        check(result.tile_bytes == tile.size(), "tile size preserved");
      }
    }
  }
}

void test_still_header_size() {
  const av2enc::m1::M1Config config = m1_config(64, 64, false);

  scc::StillSccConfig off;
  std::size_t off_bits = 0;
  std::size_t off_bytes = 0;
  std::string error;
  check(scc::still_frame_header_size(config, off, 0, false, &off_bits,
                                     &off_bytes, nullptr, error),
        "off header size (" + error + ")");
  // is_first(1) + uvlc(0)(1) + uvlc(0)(1) + 2 SCC bits + disable_cdf(1) +
  // tile_info(1) + base_q(8) + seg/qm/deltaq(3) + deblock luma(2, chroma
  // absent when both luma directions are off) + tx(1) + reduced_tx_set(2)
  // = 23 bits -> 3 bytes.
  check(off_bits == 23, "off header is 23 syntax bits");
  check(off_bytes == 3, "off header is 3 bytes");

  scc::StillSccConfig integer;
  integer.frame.allow_screen_content_tools = true;
  integer.frame.force_integer_mv = true;
  std::size_t int_bits = 0;
  std::size_t int_bytes = 0;
  scc::FrameSccState state;
  check(scc::still_frame_header_size(config, integer, 0, false, &int_bits,
                                     &int_bytes, &state, error),
        "integer header size (" + error + ")");
  check(int_bits == off_bits + 1,
        "integer header adds one screen bit (23 -> 24)");
  check(state.allow_screen_content_tools && state.force_integer_mv,
        "integer header state");
}

void test_still_reader() {
  const av2enc::m1::M1Config config = m1_config(64, 64, false);
  scc::StillSccConfig scc_cfg;
  scc_cfg.frame.allow_screen_content_tools = true;
  scc_cfg.frame.force_integer_mv = true;
  scc_cfg.frame.allow_intrabc = true;
  scc_cfg.frame.allow_global_intrabc = true;
  scc_cfg.frame.allow_local_intrabc = false;

  scc::StillSccResult result;
  std::string error;
  const std::vector<std::uint8_t> tile{0x11, 0x22};
  check(scc::assemble_still_picture(config, scc_cfg, 0, false, tile, result,
                                    error),
        "assemble intrabc still (" + error + ")");

  const scc::SequenceSccState sequence =
      scc::derive_sequence_scc_state(scc_cfg.sequence);
  scc::FrameSccContext context;
  context.frame_is_intra = true;
  scc::FrameSccState read;
  check(scc::read_still_frame_scc(result.frame, sequence, context, 0, false,
                                  read, error),
        "read SSCs out of the assembled still (" + error + ")");
  check(read.allow_screen_content_tools && read.force_integer_mv,
        "still reader: screen + integer");
  check(read.allow_intrabc && read.allow_global_intrabc &&
            !read.allow_local_intrabc,
        "still reader: global-only IBC");
  check(read.coded_allow_screen_content_tools &&
            read.coded_force_integer_mv &&
            read.coded_allow_global_intrabc,
        "still reader: presence record");
}

// ---------------------------------------------------------------------------
// Bridge to the existing palette/IBC modules
// ---------------------------------------------------------------------------

void test_bridge() {
  std::string error;
  {
    const av2enc::m1::palette_stream::ScreenContentFlags flags =
        av2enc::m1::palette_stream::palette_screen_content_flags();
    scc::BridgedSccConfig bridged;
    check(scc::from_palette_flags(flags, bridged, error),
          "palette flags bridge (" + error + ")");
    check(bridged.sequence.single_picture_header_flag,
          "palette bridge keeps single picture");
    check(bridged.frame.allow_screen_content_tools &&
              bridged.frame.force_integer_mv &&
              !bridged.frame.allow_intrabc,
          "palette bridge maps screen/force/IBC");
  }
  {
    av2enc::m1::palette_stream::ScreenContentFlags flags =
        av2enc::m1::palette_stream::palette_screen_content_flags();
    flags.allow_intrabc = true;
    flags.allow_global_intrabc = true;
    flags.allow_local_intrabc = false;
    scc::BridgedSccConfig bridged;
    check(scc::from_palette_flags(flags, bridged, error),
          "palette+global-IBC bridge (" + error + ")");
    check(bridged.frame.allow_intrabc && bridged.frame.allow_global_intrabc &&
              !bridged.frame.allow_local_intrabc,
          "palette bridge maps global-only IBC");
  }
  {
    av2enc::m1::palette_stream::ScreenContentFlags flags =
        av2enc::m1::palette_stream::palette_screen_content_flags();
    flags.single_picture_header = false;
    scc::BridgedSccConfig bridged;
    check(scc::from_palette_flags(flags, bridged, error),
          "non-single palette bridge (" + error + ")");
    check(!bridged.sequence.single_picture_header_flag &&
              bridged.sequence.seq_choose_screen_content_tools &&
              bridged.sequence.seq_choose_integer_mv,
          "non-single palette bridge selects both choose bits");
  }
  {
    av2enc::m1::palette_stream::ScreenContentFlags flags =
        av2enc::m1::palette_stream::palette_screen_content_flags();
    flags.allow_screen_content_tools = false;
    scc::BridgedSccConfig bridged;
    check(!scc::from_palette_flags(flags, bridged, error),
          "palette bridge rejects palette without screen tools");
  }
  {
    av2enc::m1::ibc_stream::StreamConfig config;
    scc::BridgedSccConfig bridged;
    check(scc::from_ibc_config(config, bridged, error),
          "ibc config bridge (" + error + ")");
    check(bridged.frame.allow_screen_content_tools &&
              bridged.frame.force_integer_mv && bridged.frame.allow_intrabc &&
              bridged.frame.allow_global_intrabc &&
              !bridged.frame.allow_local_intrabc,
          "ibc bridge maps the default global-IBC flags");
  }
  {
    av2enc::m1::ibc_stream::StreamConfig config;
    config.force_integer_mv = false;
    scc::BridgedSccConfig bridged;
    check(!scc::from_ibc_config(config, bridged, error),
          "ibc bridge rejects fractional MVs");
  }
}

}  // namespace

int main() {
  test_sequence_branches();
  test_sequence_validation();
  test_frame_branches();
  test_frame_validation();
  test_legality();
  test_still_byte_identity();
  test_still_header_size();
  test_still_reader();
  test_bridge();

  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
