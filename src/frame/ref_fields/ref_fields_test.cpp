// Frame-header reference machinery tests: §5.18.1 frame_header_copy,
// §5.18.2 primary_ref_frame / refresh / drl, §5.18.3.4 change_bvp_drl,
// §5.18.4.3 frame_size_with_refs.
//
// Covers pinned bit goldens, writer<->independent-reader round trips, every
// availability/consistency rule, and the M2 adapter's bit-exactness against
// the untouched M2 header writer.
//
// Build/run: ./build_and_test.sh in this directory (no CMake changes).
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "frame/ref_fields/ref_fields.h"
#include "frame/ref_fields/ref_fields_video_adapter.h"
#include "pipeline/video/video_headers.h"

namespace {

using namespace av2enc::frame::ref_fields;
using av2enc::BitWriter;

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

std::string hex(std::span<const std::uint8_t> bytes) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  for (std::uint8_t byte : bytes) {
    out.push_back(digits[byte >> 4U]);
    out.push_back(digits[byte & 0x0FU]);
  }
  return out;
}

void check_hex(BitWriter& writer, const char* expected,
               const std::string& what) {
  writer.align_to_byte();
  const std::string actual = hex(writer.bytes());
  check(actual == expected,
        what + ": got " + actual + ", want " + expected);
}

// A live state with two reference slots: slot 0 is a 128x64 INTER frame, slot 1
// is a 64x64 INTER frame.  Both are valid, non-restricted and q = 90.
FrameRefState make_state() {
  FrameRefState state;
  state.num_ref_frames = 2;
  state.max_frame_width = 128;
  state.max_frame_height = 64;
  state.frame_width_bits = 7;
  state.frame_height_bits = 7;
  state.order_hint_bits = 4;
  state.enable_order_hint = true;
  state.order_hint = 2;
  state.slot[0] = RefSlot{true, 1, 128, 64, 90, FrameType::kInter};
  state.slot[1] = RefSlot{true, 0, 64, 64, 90, FrameType::kInter};
  return state;
}

RefList make_refs(std::uint32_t count, std::uint8_t first,
                  std::uint8_t second = 0) {
  RefList refs;
  refs.num_total_refs = count;
  refs.index[0] = first;
  if (count > 1) refs.index[1] = second;
  return refs;
}

// ---------------------------------------------------------------------------
void test_reader_primitives() {
  // §4.11.3 uvlc: 0 -> "1", 1 -> "010", 2 -> "011", 3 -> "00100".
  {
    const std::uint8_t bytes[] = {0b10100110, 0b01000000};
    RefBitReader reader(bytes);
    std::uint32_t value = 99;
    check(reader.uvlc(&value) && value == 0, "uvlc(0)");
    check(reader.uvlc(&value) && value == 1, "uvlc(1)");
    check(reader.uvlc(&value) && value == 2, "uvlc(2)");
    check(reader.uvlc(&value) && value == 3, "uvlc(3)");
    check(reader.position() == 12, "uvlc consumed 12 bits");
  }
  // §4.11.7 ns(4): 0 -> "00", 1 -> "01", 2 -> "10", 3 -> "11".
  {
    const std::uint8_t bytes[] = {0b00011011};
    RefBitReader reader(bytes);
    for (std::uint32_t expected = 0; expected < 4; ++expected) {
      std::uint32_t value = 99;
      check(reader.ns(4, &value) && value == expected,
            "ns(4) value " + std::to_string(expected));
    }
  }
  // §4.11.6 su(4): 0b1000 -> -8, 0b0111 -> 7.
  {
    const std::uint8_t bytes[] = {0b10000111, 0b00000000};
    RefBitReader reader(bytes);
    std::int32_t value = 0;
    check(reader.su(4, &value) && value == -8, "su(4) -8");
    check(reader.su(4, &value) && value == 7, "su(4) 7");
  }
  // Truncation is an error, not silent zero-fill.
  {
    const std::uint8_t bytes[] = {0x00};
    RefBitReader reader(bytes);
    std::uint32_t value = 0;
    check(!reader.uvlc(&value), "uvlc rejects a truncated code");
    RefBitReader small(bytes);
    check(!small.bits(9, &value), "bits(n) rejects a short buffer");
  }
  check(ceil_log2(1) == 0 && ceil_log2(2) == 1 && ceil_log2(3) == 2 &&
            ceil_log2(4) == 2 && ceil_log2(8) == 3,
        "ceil_log2");
  check(bit_width(1) == 1 && bit_width(127) == 7 && bit_width(128) == 8,
        "bit_width");
}

// ---------------------------------------------------------------------------
void test_primary_ref_goldens() {
  std::string error;
  {
    BitWriter writer;
    PrimaryRefSignal signal;
    signal.signal_primary_ref_frame = true;
    signal.primary_ref_frame = 5;
    signal.disable_cross_frame_cdf_init = true;
    check(write_primary_ref_signal(writer, FrameType::kInter, false, signal,
                                   &error),
          "primary golden: write");
    // "1 1 101" = 11101 -> 0xe8.
    check_hex(writer, "e8", "primary signal p=5 disable=1");
  }
  {
    BitWriter writer;
    PrimaryRefSignal signal;
    signal.signal_primary_ref_frame = false;
    signal.disable_cross_frame_cdf_init = false;
    check(write_primary_ref_signal(writer, FrameType::kInter, false, signal,
                                   &error),
          "primary golden: write off");
    check_hex(writer, "00", "primary signal off disable=0");
  }
  {
    // TIP frames omit disable_cross_frame_cdf_init (§5.18.2).
    BitWriter writer;
    PrimaryRefSignal signal;
    signal.signal_primary_ref_frame = true;
    signal.primary_ref_frame = 0;
    check(write_primary_ref_signal(writer, FrameType::kInter, true, signal,
                                   &error),
          "primary golden: write TIP");
    check_hex(writer, "80", "primary TIP p=0 (no disable bit)");
  }
  {
    // Intra frames code nothing at all.
    BitWriter writer;
    PrimaryRefSignal signal;
    check(write_primary_ref_signal(writer, FrameType::kKey, false, signal,
                                   &error),
          "primary golden: key frame");
    check(writer.bit_size() == 0, "primary: key frame codes no bits");
  }
  {
    // A signalled primary on an intra frame is rejected, not silently dropped.
    BitWriter writer;
    PrimaryRefSignal signal;
    signal.signal_primary_ref_frame = true;
    check(!write_primary_ref_signal(writer, FrameType::kIntraOnly, false,
                                    signal, &error),
          "primary: intra-only cannot signal a primary");
    check(writer.bit_size() == 0, "primary: rejected write leaves no bits");
  }
  {
    // f(3) cannot represent PRIMARY_REF_NONE.
    BitWriter writer;
    PrimaryRefSignal signal;
    signal.signal_primary_ref_frame = true;
    signal.primary_ref_frame = kPrimaryRefNone;
    check(!write_primary_ref_signal(writer, FrameType::kInter, false, signal,
                                    &error),
          "primary: 7 does not fit f(3)");
  }
}

void test_primary_ref_round_trip() {
  std::string error;
  for (bool signalled : {false, true}) {
    for (bool disable : {false, true}) {
      for (bool tip : {false, true}) {
        BitWriter writer;
        PrimaryRefSignal signal;
        signal.signal_primary_ref_frame = signalled;
        signal.primary_ref_frame = signalled ? 3U : kPrimaryRefChoose;
        signal.disable_cross_frame_cdf_init = disable;
        check(write_primary_ref_signal(writer, FrameType::kInter, tip, signal,
                                       &error),
              "primary round trip: write");
        writer.align_to_byte();
        RefBitReader reader(writer.bytes());
        PrimaryRefSignal parsed;
        check(read_primary_ref_signal(reader, FrameType::kInter, tip, &parsed,
                                      &error),
              "primary round trip: read");
        check(parsed.signal_primary_ref_frame == signalled,
              "primary round trip: signal bit");
        check(parsed.primary_ref_frame ==
                  (signalled ? 3U : kPrimaryRefChoose),
              "primary round trip: value");
        if (!tip) {
          check(parsed.disable_cross_frame_cdf_init == disable,
                "primary round trip: disable bit");
        }
      }
    }
  }
  // A reserved f(3) value is rejected by the reader.
  {
    BitWriter writer;
    writer.put_bit(true);   // signal
    writer.put_bit(true);   // disable
    writer.put_bits(7U, 3U);  // reserved
    writer.align_to_byte();
    RefBitReader reader(writer.bytes());
    PrimaryRefSignal parsed;
    check(!read_primary_ref_signal(reader, FrameType::kInter, false, &parsed,
                                   &error),
          "primary reader rejects a reserved f(3) value");
  }
}

void test_primary_ref_resolution() {
  std::string error;
  const FrameRefState state = make_state();
  const RefList refs = make_refs(1, 0);

  // No signal: the derivation picks the ref with the closest q, then the
  // default is to reset the CDFs.
  {
    PrimaryRefSignal signal;
    signal.disable_cross_frame_cdf_init = true;
    PrimaryRefResolved out;
    check(resolve_primary_ref(FrameType::kInter, signal, refs, state, 90,
                              &out, &error),
          "resolve: default");
    check(out.derived_primary_ref_frame == 0, "resolve: derived ref 0");
    check(out.primary_ref_frame == 0, "resolve: primary ref 0");
    check(out.disable_cross_frame_cdf_init, "resolve: reset CDFs");
    check(!out.load_cdfs_from_reference, "resolve: no CDF load");
  }
  // Same, with disable_cross_frame_cdf_init = 0: the CDF state comes from the
  // reference slot and load_previous() runs.
  {
    PrimaryRefSignal signal;
    signal.disable_cross_frame_cdf_init = false;
    PrimaryRefResolved out;
    check(resolve_primary_ref(FrameType::kInter, signal, refs, state, 90,
                              &out, &error),
          "resolve: cross-frame CDF init");
    check(out.load_cdfs_from_reference, "resolve: CDFs load from a reference");
    check(out.cdf_reference_slot == 0, "resolve: CDF slot 0");
    check(out.load_previous_state, "resolve: load_previous() runs");
  }
  // Signalled primary replaces the derived choice.
  {
    const RefList two = make_refs(2, 0, 1);
    PrimaryRefSignal signal;
    signal.signal_primary_ref_frame = true;
    signal.primary_ref_frame = 1;
    signal.disable_cross_frame_cdf_init = false;
    PrimaryRefResolved out;
    check(resolve_primary_ref(FrameType::kInter, signal, two, state, 90, &out,
                              &error),
          "resolve: signalled primary");
    check(out.primary_ref_frame == 1, "resolve: signalled primary wins");
    check(out.cdf_reference_slot == 1, "resolve: CDF slot 1");
  }
  // primary_ref_frame >= NumTotalRefs is rejected (§6.17.2).
  {
    PrimaryRefSignal signal;
    signal.signal_primary_ref_frame = true;
    signal.primary_ref_frame = 1;
    PrimaryRefResolved out;
    check(!resolve_primary_ref(FrameType::kInter, signal, refs, state, 90,
                               &out, &error),
          "resolve: primary >= NumTotalRefs is rejected");
  }
  // RESTRICTED_OH references are rejected.
  {
    FrameRefState restricted = state;
    restricted.slot[0].order_hint = kRestrictedOrderHint;
    PrimaryRefSignal signal;
    signal.signal_primary_ref_frame = true;
    signal.primary_ref_frame = 0;
    PrimaryRefResolved out;
    check(!resolve_primary_ref(FrameType::kInter, signal, refs, restricted, 90,
                               &out, &error),
          "resolve: RESTRICTED_OH primary is rejected");
  }
  // An unavailable slot is rejected.
  {
    FrameRefState invalid = state;
    invalid.slot[0].valid = false;
    PrimaryRefSignal signal;
    PrimaryRefResolved out;
    check(!resolve_primary_ref(FrameType::kInter, signal, refs, invalid, 90,
                               &out, &error),
          "resolve: invalid reference slot is rejected");
  }
  // When the derivation finds nothing, the process forces PRIMARY_REF_NONE and
  // disable_cross_frame_cdf_init = 1 (§5.18.2 set_primary_ref_frame_and_ctx).
  {
    FrameRefState keys = state;
    keys.slot[0].frame_type = FrameType::kKey;
    keys.slot[1].frame_type = FrameType::kKey;
    PrimaryRefSignal signal;
    signal.disable_cross_frame_cdf_init = false;
    PrimaryRefResolved out;
    check(resolve_primary_ref(FrameType::kInter, signal, make_refs(2, 0, 1),
                              keys, 90, &out, &error),
          "resolve: no eligible ref");
    check(out.primary_ref_frame == kPrimaryRefNone,
          "resolve: forced to PRIMARY_REF_NONE");
    check(out.disable_cross_frame_cdf_init,
          "resolve: forced disable_cross_frame_cdf_init = 1");
    check(!out.load_cdfs_from_reference, "resolve: no CDF load after forcing");
  }
  // Intra frames: nothing is coded and the CDFs always reset.
  {
    PrimaryRefSignal signal;
    signal.disable_cross_frame_cdf_init = false;
    PrimaryRefResolved out;
    check(resolve_primary_ref(FrameType::kKey, signal, refs, state, 90, &out,
                              &error),
          "resolve: key frame");
    check(out.primary_ref_frame == kPrimaryRefNone, "resolve: key -> NONE");
    check(out.disable_cross_frame_cdf_init, "resolve: key -> reset CDFs");
  }
  // The q-based derivation: with equal q the later hint wins.
  {
    FrameRefState two = state;
    two.slot[1].order_hint = 3;
    two.slot[1].width = 128;
    two.slot[1].height = 64;
    std::uint32_t primary = 0;
    std::uint32_t secondary = 0;
    choose_primary_secondary_ref_frame(make_refs(2, 0, 1), two, 90, &primary,
                                       &secondary);
    check(primary == 1, "choose: closer/later order hint becomes primary");
    check(secondary == 0, "choose: the other ref becomes secondary");
  }
}

// ---------------------------------------------------------------------------
void test_drl_goldens() {
  std::string error;
  {
    // change_drl = 1, ns(4) value 3 -> "1" + "11" = 111 -> 0xe0.
    BitWriter writer;
    DrlConfig config;
    config.allow_frame_max_drl_bits = true;
    config.seq_max_drl_bits_minus_1 = 4;
    DrlSignal signal;
    signal.change_drl = true;
    signal.max_drl_bits_minus_1 = 3;
    check(write_drl(writer, config, signal, &error), "drl golden: write");
    check_hex(writer, "e0", "drl change seq=4 -> 3");
  }
  {
    // value > seq encodes value - 1; seq=2, value=3 -> ns(4, 2) = "10".
    BitWriter writer;
    DrlConfig config;
    config.allow_frame_max_drl_bits = true;
    config.seq_max_drl_bits_minus_1 = 2;
    DrlSignal signal;
    signal.change_drl = true;
    signal.max_drl_bits_minus_1 = 3;
    check(write_drl(writer, config, signal, &error), "drl golden: write +1");
    check_hex(writer, "c0", "drl change seq=2 -> 3");
  }
  {
    // Sequence value: a single 0 bit.
    BitWriter writer;
    DrlConfig config;
    config.allow_frame_max_drl_bits = true;
    config.seq_max_drl_bits_minus_1 = 4;
    DrlSignal signal;
    signal.change_drl = false;
    signal.max_drl_bits_minus_1 = 4;
    check(write_drl(writer, config, signal, &error), "drl golden: no change");
    check(writer.bit_size() == 1, "drl: no-change is one bit");
    check_hex(writer, "00", "drl no change");
  }
  {
    // Sequence flag off: no bits, and the value must be the sequence value.
    BitWriter writer;
    DrlConfig config;
    config.seq_max_drl_bits_minus_1 = 3;
    DrlSignal signal;
    signal.max_drl_bits_minus_1 = 3;
    check(write_drl(writer, config, signal, &error), "drl: flag off");
    check(writer.bit_size() == 0, "drl: flag off codes no bits");
    signal.max_drl_bits_minus_1 = 2;
    check(!write_drl(writer, config, signal, &error),
          "drl: cannot change without the sequence flag");
  }
  // change_bvp_drl: seq 0 -> 1 is "1" + ns(2, 0) = "10" -> 0x80.
  {
    BitWriter writer;
    BvDrlConfig config;
    config.allow_frame_max_bvp_drl_bits = true;
    config.seq_max_bvp_drl_bits_minus_1 = 0;
    BvDrlSignal signal;
    signal.change_bvp_drl = true;
    signal.max_bvp_drl_bits_minus_1 = 1;
    check(write_bvp_drl(writer, config, signal, &error), "bvp golden: write");
    check_hex(writer, "80", "bvp change seq=0 -> 1");
  }
  // The BVP field is syntax-conditional on allow_intrabc: a 2 is the largest
  // representable value (MAX_REF_BV_STACK_SIZE - 2).
  {
    BitWriter writer;
    BvDrlConfig config;
    config.allow_frame_max_bvp_drl_bits = true;
    config.seq_max_bvp_drl_bits_minus_1 = 0;
    BvDrlSignal signal;
    signal.change_bvp_drl = true;
    signal.max_bvp_drl_bits_minus_1 = 3;
    check(!write_bvp_drl(writer, config, signal, &error),
          "bvp: value above the stack bound is rejected");
  }
}

void test_drl_round_trip() {
  std::string error;
  for (std::uint32_t seq = 0; seq <= 4; ++seq) {
    for (std::uint32_t value = 0; value <= 4; ++value) {
      BitWriter writer;
      DrlConfig config;
      config.allow_frame_max_drl_bits = true;
      config.seq_max_drl_bits_minus_1 = seq;
      DrlSignal signal;
      signal.max_drl_bits_minus_1 = value;
      signal.change_drl = value != seq;
      const bool wrote = write_drl(writer, config, signal, &error);
      if (!wrote) {
        // A value below seq is representable; above is representable by one.
        check(value > seq + 1 || value > 4,
              "drl round trip: write failed for seq " + std::to_string(seq) +
                  " value " + std::to_string(value));
        continue;
      }
      writer.align_to_byte();
      RefBitReader reader(writer.bytes());
      DrlSignal parsed;
      check(read_drl(reader, config, &parsed, &error), "drl round trip: read");
      check(parsed.max_drl_bits_minus_1 == value,
            "drl round trip: value seq " + std::to_string(seq) + " = " +
                std::to_string(value));
      check(parsed.change_drl == signal.change_drl,
            "drl round trip: change flag");
    }
  }
  for (std::uint32_t seq = 0; seq <= 2; ++seq) {
    for (std::uint32_t value = 0; value <= 2; ++value) {
      BitWriter writer;
      BvDrlConfig config;
      config.allow_frame_max_bvp_drl_bits = true;
      config.seq_max_bvp_drl_bits_minus_1 = seq;
      BvDrlSignal signal;
      signal.max_bvp_drl_bits_minus_1 = value;
      signal.change_bvp_drl = value != seq;
      if (!write_bvp_drl(writer, config, signal, &error)) {
        check(false, "bvp round trip: write seq " + std::to_string(seq) +
                         " value " + std::to_string(value));
        continue;
      }
      writer.align_to_byte();
      RefBitReader reader(writer.bytes());
      BvDrlSignal parsed;
      check(read_bvp_drl(reader, config, &parsed, &error),
            "bvp round trip: read");
      check(parsed.max_bvp_drl_bits_minus_1 == value,
            "bvp round trip: value");
    }
  }
  // A sequence value outside the stack bound is rejected on both sides.
  {
    BitWriter writer;
    DrlConfig config;
    config.seq_max_drl_bits_minus_1 = 5;
    DrlSignal signal;
    signal.max_drl_bits_minus_1 = 5;
    check(!write_drl(writer, config, signal, &error),
          "drl: out-of-range sequence value is rejected");
    const std::uint8_t bytes[] = {0};
    RefBitReader reader(bytes);
    DrlSignal parsed;
    check(!read_drl(reader, config, &parsed, &error),
          "drl reader rejects an out-of-range sequence value");
  }
}

// ---------------------------------------------------------------------------
void test_refresh_goldens() {
  std::string error;
  {
    // Short refresh: has=1, frame_to_refresh (1 bit) = 0 -> "10".
    BitWriter writer;
    RefreshConfig config;
    config.enable_short_refresh_frame_flags = true;
    config.num_ref_frames = 2;
    RefreshSignal signal;
    signal.refresh_frame_flags = 1;
    check(write_refresh_frame_flags(writer, FrameType::kInter, config, signal,
                                    true, &error),
          "refresh golden: short slot 0");
    check_hex(writer, "80", "refresh short inter slot 0");
  }
  {
    // Short refresh with no refresh: has = 0.
    BitWriter writer;
    RefreshConfig config;
    config.enable_short_refresh_frame_flags = true;
    config.num_ref_frames = 2;
    RefreshSignal signal;
    signal.refresh_frame_flags = 0;
    check(write_refresh_frame_flags(writer, FrameType::kInter, config, signal,
                                    true, &error),
          "refresh golden: short none");
    check_hex(writer, "00", "refresh short none");
    signal.refresh_frame_flags = 1;
    check(write_refresh_frame_flags(writer, FrameType::kInter, config, signal,
                                    false, &error),
          "refresh: a non-output frame may refresh a slot");
    BitWriter none;
    RefreshSignal no_refresh;
    no_refresh.refresh_frame_flags = 0;
    check(!write_refresh_frame_flags(none, FrameType::kInter, config,
                                     no_refresh, false, &error),
          "refresh: a non-output frame must refresh a slot");
  }
  {
    // Generic form: f(NumRefFrames) = f(2).
    BitWriter writer;
    RefreshConfig config;
    config.num_ref_frames = 2;
    RefreshSignal signal;
    signal.refresh_frame_flags = 1;
    check(write_refresh_frame_flags(writer, FrameType::kInter, config, signal,
                                    true, &error),
          "refresh golden: generic");
    check_hex(writer, "40", "refresh generic slot 0 (2 bits)");
  }
  {
    // CLK with max_mlayer_id == 0 infers allFrames and codes nothing.
    BitWriter writer;
    RefreshConfig config;
    config.num_ref_frames = 2;
    config.obu_is_closed_loop_key = true;
    RefreshSignal signal;
    signal.refresh_frame_flags = 0;
    check(write_refresh_frame_flags(writer, FrameType::kKey, config, signal,
                                    true, &error),
          "refresh golden: CLK inference");
    check(writer.bit_size() == 0, "refresh: CLK codes no bits");
    check(signal.refresh_frame_flags == 0, "refresh: input is not modified");
  }
  {
    // Short-refresh KEY frame codes frame_to_refresh only.
    BitWriter writer;
    RefreshConfig config;
    config.enable_short_refresh_frame_flags = true;
    config.num_ref_frames = 2;
    config.obu_is_closed_loop_key = false;
    RefreshSignal signal;
    signal.refresh_frame_flags = 2;
    check(write_refresh_frame_flags(writer, FrameType::kKey, config, signal,
                                    true, &error),
          "refresh golden: short key");
    check_hex(writer, "80", "refresh short key slot 1 (1 bit)");
  }
  // Validation rules.
  {
    RefreshConfig config;
    config.num_ref_frames = 2;
    RefreshSignal signal;
    signal.refresh_frame_flags = 4;
    check(!validate_refresh_frame_flags(FrameType::kInter, config, signal,
                                        true, &error),
          "refresh: mask above NumRefFrames is rejected");
  }
  {
    RefreshConfig config;
    config.num_ref_frames = 4;
    RefreshSignal signal;
    signal.refresh_frame_flags = 0xF;
    check(!validate_refresh_frame_flags(FrameType::kIntraOnly, config, signal,
                                        true, &error),
          "refresh: INTRA_ONLY must not refresh all slots");
  }
  {
    RefreshConfig config;
    config.enable_short_refresh_frame_flags = true;
    config.num_ref_frames = 2;
    RefreshSignal signal;
    signal.has_refresh_frame_flags = true;
    signal.frame_to_refresh = 2;
    signal.refresh_frame_flags = 4;
    check(!validate_refresh_frame_flags(FrameType::kInter, config, signal,
                                        true, &error),
          "refresh: frame_to_refresh must be < NumRefFrames");
  }
  {
    RefreshConfig config;
    config.enable_short_refresh_frame_flags = true;
    config.num_ref_frames = 2;
    RefreshSignal signal;
    signal.has_refresh_frame_flags = false;
    signal.refresh_frame_flags = 1;
    check(!validate_refresh_frame_flags(FrameType::kInter, config, signal,
                                        true, &error),
          "refresh: has=0 infers a zero mask");
  }
  // The reader derives the same mask.
  {
    RefreshConfig config;
    config.enable_short_refresh_frame_flags = true;
    config.num_ref_frames = 2;
    const std::uint8_t bytes[] = {0x80};
    RefBitReader reader(bytes);
    RefreshSignal parsed;
    check(read_refresh_frame_flags(reader, FrameType::kInter, config, true,
                                   &parsed, &error),
          "refresh reader: short slot 0");
    check(parsed.has_refresh_frame_flags, "refresh reader: has = 1");
    check(parsed.frame_to_refresh == 0, "refresh reader: index 0");
    check(parsed.refresh_frame_flags == 1, "refresh reader: mask 1");
  }
}

// ---------------------------------------------------------------------------
void test_frame_size_with_refs() {
  std::string error;
  const FrameRefState state = make_state();
  const RefList refs = make_refs(1, 0);
  SizeConfig config;
  config.num_total_refs = 1;
  config.max_frame_width = 128;
  config.max_frame_height = 64;
  config.frame_width_bits = 7;
  config.frame_height_bits = 7;

  // found_ref = 1: one bit.
  {
    BitWriter writer;
    FrameSizeSignal signal;
    signal.width = 128;
    signal.height = 64;
    signal.prefer_found_ref = true;
    FrameSizeResolved out;
    check(write_frame_size_with_refs(writer, config, refs, state, signal, &out,
                                     &error),
          "size golden: found_ref");
    check_hex(writer, "80", "size found_ref");
    check(!out.explicit_size, "size: found_ref is not explicit");
    check(out.inherited_from == 0, "size: inherited from ref 0");
    check(out.width == 128 && out.height == 64, "size: inherited 128x64");
    check(out.mi_cols == 32 && out.mi_rows == 16,
          "size: compute_image_size for 128x64");
  }
  // Explicit frame_size(): found_ref = 0 then 7+7 bits.
  {
    BitWriter writer;
    FrameSizeSignal signal;
    signal.width = 128;
    signal.height = 64;
    signal.prefer_found_ref = false;
    FrameSizeResolved out;
    check(write_frame_size_with_refs(writer, config, refs, state, signal, &out,
                                     &error),
          "size golden: explicit");
    check_hex(writer, "7f7e", "size explicit 128x64");
    check(out.explicit_size, "size: explicit_size");
    check(out.inherited_from == kNoRef, "size: no inherited ref");
  }
  // found_ref on the second reference: "0 1" -> 0x40.
  {
    RefList two = make_refs(2, 1, 0);  // ref 1 is 64x64, ref 0 is 128x64
    SizeConfig c2 = config;
    c2.num_total_refs = 2;
    BitWriter writer;
    FrameSizeSignal signal;
    signal.width = 128;
    signal.height = 64;
    signal.prefer_found_ref = true;
    FrameSizeResolved out;
    check(write_frame_size_with_refs(writer, c2, two, state, signal, &out,
                                     &error),
          "size: found_ref second");
    check_hex(writer, "40", "size found_ref at index 1");
    check(out.inherited_from == 1, "size: inherited from ref index 1");
    check(out.found_ref[0] == false && out.found_ref[1] == true,
          "size: found_ref bits");
  }
  // Round trip.
  {
    BitWriter writer;
    FrameSizeSignal signal;
    signal.width = 128;
    signal.height = 64;
    signal.prefer_found_ref = true;
    FrameSizeResolved written;
    check(write_frame_size_with_refs(writer, config, refs, state, signal,
                                     &written, &error),
          "size round trip: write");
    writer.align_to_byte();
    RefBitReader reader(writer.bytes());
    FrameSizeResolved parsed;
    check(read_frame_size_with_refs(reader, config, refs, state, &parsed,
                                    &error),
          "size round trip: read");
    check(parsed.inherited_from == written.inherited_from,
          "size round trip: inherited index");
    check(parsed.width == 128 && parsed.height == 64,
          "size round trip: size");
  }
  // Explicit size above the sequence maximum is rejected.
  {
    BitWriter writer;
    FrameSizeSignal signal;
    signal.width = 256;
    signal.height = 64;
    signal.prefer_found_ref = false;
    FrameSizeResolved out;
    check(!write_frame_size_with_refs(writer, config, refs, state, signal,
                                      &out, &error),
          "size: above max_frame_width is rejected");
  }
  // 16x ratio bound (frame > 16 * ref in a dimension).
  {
    FrameRefState big = state;
    big.max_frame_width = 4096;
    big.max_frame_height = 4096;
    SizeConfig c = config;
    c.max_frame_width = 4096;
    c.max_frame_height = 4096;
    c.frame_width_bits = 12;
    c.frame_height_bits = 12;
    BitWriter writer;
    FrameSizeSignal signal;
    signal.width = 4096;  // 32x the 128-wide reference
    signal.height = 64;
    signal.prefer_found_ref = false;
    FrameSizeResolved out;
    check(!write_frame_size_with_refs(writer, c, refs, big, signal, &out,
                                      &error),
          "size: >16x reference width is rejected");
  }
  // 2x reduction bound: 32x64 against a 128x64 reference is more than 2x.
  {
    SizeConfig c = config;
    c.frame_width_bits = 7;
    c.frame_height_bits = 7;
    BitWriter writer;
    FrameSizeSignal signal;
    signal.width = 32;
    signal.height = 64;
    signal.prefer_found_ref = false;
    FrameSizeResolved out;
    check(!write_frame_size_with_refs(writer, c, refs, state, signal, &out,
                                      &error),
          "size: 4x reduction is rejected");
  }
  // RESTRICTED_OH references cannot supply a size.
  {
    FrameRefState restricted = state;
    restricted.slot[0].order_hint = kRestrictedOrderHint;
    BitWriter writer;
    FrameSizeSignal signal;
    signal.width = 128;
    signal.height = 64;
    signal.prefer_found_ref = true;
    FrameSizeResolved out;
    check(!write_frame_size_with_refs(writer, config, refs, restricted, signal,
                                      &out, &error),
          "size: RESTRICTED_OH reference is rejected");
    // The explicit form only needs RefValid, but validate_frame_size_against_refs
    // still requires the size to be sane.
    signal.prefer_found_ref = false;
    FrameSizeResolved explicit_out;
    check(write_frame_size_with_refs(writer, config, refs, restricted, signal,
                                     &explicit_out, &error),
          "size: explicit form accepts a restricted reference");
  }
  // NumTotalRefs == 0: the loop is empty, so frame_size() is coded.
  {
    RefList empty;
    SizeConfig c = config;
    c.num_total_refs = 0;
    BitWriter writer;
    FrameSizeSignal signal;
    signal.width = 128;
    signal.height = 64;
    FrameSizeResolved out;
    check(write_frame_size_with_refs(writer, c, empty, state, signal, &out,
                                     &error),
          "size: NumTotalRefs = 0 writes frame_size()");
    check(out.explicit_size, "size: NumTotalRefs = 0 is explicit");
    check_hex(writer, "fefc", "size NumTotalRefs = 0");
  }
}

// ---------------------------------------------------------------------------
void test_frame_header_copy() {
  std::string error;
  // Capture and replay a 13-bit header after 3 unrelated bits.
  {
    BitWriter writer;
    const std::size_t start = writer.bit_size();
    writer.put_bits(0b10101U, 5U);
    writer.put_bits(0b1100110U, 7U);
    writer.put_bit(true);
    const HeaderBits bits = capture_bits(writer.bytes(), start,
                                         writer.bit_size());
    check(bits.num_bits == 13, "copy: captured 13 bits");
    check(hex(bits.bytes) == "ae68", "copy: captured bits are packed");
    BitWriter replay;
    replay.put_bits(0b101U, 3U);
    check(write_frame_header_copy(replay, bits, &error) == 13,
          "copy: wrote 13 bits");
    replay.align_to_byte();
    // "101" + "1010111001101" = 1011 0101 1100 1101 -> b5 cd.
    check(hex(replay.bytes()) == "b5cd", "copy: bit-exact, unaligned replay");
    RefBitReader reader(replay.bytes(), 3);
    check(read_frame_header_copy(reader, bits, &error),
          "copy: reader verifies the replay");
    check(reader.position() == 16, "copy: reader consumed 13 bits");
  }
  // A single flipped bit is rejected.
  {
    BitWriter writer;
    const std::size_t start = writer.bit_size();
    writer.put_bits(0xA5U, 8U);
    const HeaderBits bits = capture_bits(writer.bytes(), start,
                                         writer.bit_size());
    const std::uint8_t matching[] = {0xA5};
    RefBitReader reader(matching);
    check(read_frame_header_copy(reader, bits, &error),
          "copy: matching header verifies");
    const std::uint8_t corrupted[] = {0x25};
    RefBitReader evil(corrupted);
    check(!read_frame_header_copy(evil, bits, &error),
          "copy: a one-bit corruption is rejected");
  }
  // Truncation and an empty store.
  {
    const std::uint8_t bytes[] = {0x80};
    RefBitReader reader(bytes);
    HeaderBits bits;
    bits.num_bits = 9;
    bits.bytes = {0xFF, 0x80};
    check(!read_frame_header_copy(reader, bits, &error),
          "copy: a truncated copy is rejected");
    BitWriter writer;
    check(write_frame_header_copy(writer, HeaderBits{}, &error) == 0,
          "copy: an empty header cannot be copied");
    check(!error.empty(), "copy: empty copy reports an error");
  }
  // FrameHeaderStore + §5.19 frame_header_present_flag.
  {
    BitWriter first;
    const std::size_t start = first.bit_size();
    first.put_bits(0x5AU, 8U);
    FrameHeaderStore store;
    store.record(first.bytes(), start, first.bit_size());
    check(store.present() && store.num_bits() == 8, "store: recorded 8 bits");

    BitWriter writer;
    check(write_nonfirst_tile_group_header(writer, store, true, &error),
          "store: write copy header");
    writer.align_to_byte();
    check(hex(writer.bytes()) == "ad00", "store: flag 1 + 0x5a");
    RefBitReader reader(writer.bytes());
    bool present = false;
    check(read_nonfirst_tile_group_header(reader, store, &present, &error),
          "store: read copy header");
    check(present, "store: frame_header_present_flag = 1");

    BitWriter no_header;
    check(write_nonfirst_tile_group_header(no_header, store, false, &error),
          "store: write omitted header");
    no_header.align_to_byte();
    check(hex(no_header.bytes()) == "00", "store: flag 0 only");
    RefBitReader reader2(no_header.bytes());
    check(read_nonfirst_tile_group_header(reader2, store, &present, &error),
          "store: read omitted header");
    check(!present, "store: frame_header_present_flag = 0");
  }
}

// ---------------------------------------------------------------------------
void test_video_adapter_bit_exact() {
  // With every new option off, the adapter must reproduce the untouched M2
  // sequence and frame headers byte for byte.
  av2enc::video::VideoConfig m2;
  m2.max_frame_width = 64;
  m2.max_frame_height = 64;
  m2.base_q_idx = 90;
  m2.order_hint_bits = 4;
  m2.num_ref_frames = 2;
  m2.num_total_refs = 1;
  m2.max_drl_bits_minus_1 = 4;
  m2.disable_cdf_update = true;

  VideoRefConfig config;
  config.max_frame_width = 64;
  config.max_frame_height = 64;
  config.base_q_idx = 90;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.num_total_refs = 1;
  config.seq_max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = true;

  std::string error;
  const std::vector<std::uint8_t> video_seq =
      av2enc::video::make_sequence_header_obu(m2);
  const std::vector<std::uint8_t> ref_seq =
      make_video_ref_sequence_header_obu(config, &error);
  check(!ref_seq.empty(), "adapter: sequence header written");
  check(hex(video_seq) == hex(ref_seq), "adapter: sequence header is bit-exact");

  FrameRefState state;
  state.num_ref_frames = 2;
  state.max_frame_width = 64;
  state.max_frame_height = 64;
  state.frame_width_bits = 6;
  state.frame_height_bits = 6;
  state.order_hint_bits = 4;
  state.slot[0] = RefSlot{true, 0, 64, 64, 90, FrameType::kKey};
  RefList refs;
  refs.num_total_refs = 1;
  refs.index[0] = 0;
  const VideoTileInfo tiles;

  for (bool key : {true, false}) {
    av2enc::video::VideoFrameHeader video_header;
    video_header.keyframe = key;
    video_header.order_hint = key ? 0U : 1U;
    video_header.ref_slot = 0;
    video_header.refresh_frame_flags = key ? 3U : 1U;
    BitWriter video_writer;
    av2enc::video::write_frame_header(video_writer, m2, video_header);
    video_writer.align_to_byte();

    VideoRefFrameHeader header;
    header.keyframe = key;
    header.frame_is_inter = !key;
    header.order_hint = video_header.order_hint;
    header.refresh_frame_flags = video_header.refresh_frame_flags;
    header.frame_width = 64;
    header.frame_height = 64;
    BitWriter writer;
    std::size_t bits = 0;
    check(write_video_ref_frame_header(writer, config, header, refs, state,
                                    tiles, &bits, &error) != 0,
          std::string("adapter: frame header ") + (key ? "key" : "inter"));
    writer.align_to_byte();
    check(hex(video_writer.bytes()) == hex(writer.bytes()),
          std::string("adapter: ") + (key ? "key" : "inter") +
              " frame header is bit-exact");
  }
}

void test_video_adapter_new_fields() {
  std::string error;
  VideoRefConfig config;
  config.max_frame_width = 128;
  config.max_frame_height = 64;
  config.base_q_idx = 90;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.num_total_refs = 1;
  config.seq_max_drl_bits_minus_1 = 4;
  config.seq_max_bvp_drl_bits_minus_1 = 0;
  config.allow_frame_max_drl_bits = true;
  config.allow_frame_max_bvp_drl_bits = true;
  config.enable_short_refresh_frame_flags = true;

  FrameRefState state;
  state.num_ref_frames = 2;
  state.max_frame_width = 128;
  state.max_frame_height = 64;
  state.frame_width_bits = 7;
  state.frame_height_bits = 7;
  state.order_hint_bits = 4;
  state.order_hint = 3;
  state.slot[0] = RefSlot{true, 2, 128, 64, 90, FrameType::kInter};
  RefList refs;
  refs.num_total_refs = 1;
  refs.index[0] = 0;
  const VideoTileInfo tiles;

  VideoRefFrameHeader header;
  header.keyframe = false;
  header.frame_is_inter = true;
  header.order_hint = 3;
  header.refresh_frame_flags = 1;
  header.signal_primary_ref_frame = true;
  header.primary_ref_frame = 0;
  header.disable_cross_frame_cdf_init = false;
  header.frame_size_override = true;
  header.prefer_found_ref = true;
  header.frame_width = 128;
  header.frame_height = 64;
  header.allow_intrabc = true;
  header.change_bvp_drl = true;
  header.max_bvp_drl_bits_minus_1 = 1;
  header.change_drl = true;
  header.max_drl_bits_minus_1 = 3;

  BitWriter writer;
  std::size_t bits = 0;
  check(write_video_ref_frame_header(writer, config, header, refs, state, tiles,
                                  &bits, &error) != 0,
        "adapter: new-field header written");
  check(bits > 0, "adapter: header bit count recorded");
  writer.align_to_byte();
  RefBitReader reader(writer.bytes());
  VideoRefParsedHeader parsed;
  check(read_video_ref_frame_header(reader, false, config, refs, state, tiles,
                                 &parsed, &error),
        "adapter: new-field header parsed");
  check(parsed.size.inherited_from == 0, "adapter: found_ref");
  check(parsed.size.width == 128 && parsed.size.height == 64,
        "adapter: inherited size");
  check(parsed.primary.primary_ref_frame == 0, "adapter: primary");
  check(!parsed.primary.disable_cross_frame_cdf_init,
        "adapter: cross-frame CDF init");
  check(parsed.refresh.has_refresh_frame_flags, "adapter: has refresh");
  check(parsed.refresh.frame_to_refresh == 0, "adapter: frame_to_refresh");
  check(parsed.drl.change_drl, "adapter: change_drl");
  check(parsed.drl.max_drl_bits_minus_1 == 3, "adapter: max_drl");
  check(parsed.header.allow_intrabc, "adapter: allow_intrabc");
  check(parsed.bvp_drl.change_bvp_drl, "adapter: change_bvp_drl");
  check(parsed.bvp_drl.max_bvp_drl_bits_minus_1 == 1, "adapter: max_bvp_drl");
  check(reader.remaining_bits() < 8, "adapter: header consumed to alignment");

  // A key frame with an explicit size.
  VideoRefFrameHeader key;
  key.keyframe = true;
  key.frame_size_override = true;
  key.frame_width = 128;
  key.frame_height = 64;
  key.refresh_frame_flags = 0;  // inferred allFrames for a CLK
  BitWriter key_writer;
  check(write_video_ref_frame_header(key_writer, config, key, refs, state, tiles,
                                  &bits, &error) != 0,
        "adapter: key frame with explicit size");
  key_writer.align_to_byte();
  RefBitReader key_reader(key_writer.bytes());
  VideoRefParsedHeader key_parsed;
  check(read_video_ref_frame_header(key_reader, true, config, refs, state, tiles,
                                 &key_parsed, &error),
        "adapter: key header parsed");
  check(key_parsed.size.explicit_size, "adapter: key explicit size");
  check(key_parsed.refresh.refresh_frame_flags == 3,
        "adapter: CLK refreshes all slots");
}

void test_video_adapter_tile_group_copy() {
  // Two tile groups in two OBUs: the second repeats the first header.
  std::string error;
  VideoRefConfig config;
  config.max_frame_width = 128;
  config.max_frame_height = 64;
  config.num_ref_frames = 2;
  config.num_total_refs = 1;
  config.seq_max_drl_bits_minus_1 = 4;
  config.enable_short_refresh_frame_flags = true;

  FrameRefState state;
  state.num_ref_frames = 2;
  state.max_frame_width = 128;
  state.max_frame_height = 64;
  state.frame_width_bits = 7;
  state.frame_height_bits = 7;
  state.order_hint_bits = 4;
  state.slot[0] = RefSlot{true, 0, 128, 64, 90, FrameType::kKey};
  RefList refs;
  refs.num_total_refs = 1;
  refs.index[0] = 0;

  VideoTileInfo tiles;
  tiles.cols_log2 = 1;
  tiles.rows_log2 = 0;
  tiles.min_cols_log2 = 0;
  tiles.max_cols_log2 = 6;
  tiles.min_tiles_log2 = 0;
  tiles.max_rows_log2 = 0;
  tiles.tile_count = 2;
  tiles.tile_size_bytes = 1;

  VideoRefFrameHeader header;
  header.keyframe = false;
  header.frame_is_inter = true;
  header.order_hint = 1;
  header.refresh_frame_flags = 1;
  header.frame_size_override = true;
  header.frame_width = 128;
  header.frame_height = 64;

  const std::vector<std::vector<std::uint8_t>> payloads = {
      std::vector<std::uint8_t>{0x12, 0x34, 0x56},
      std::vector<std::uint8_t>{0x65, 0x43, 0x21}};
  const std::vector<std::vector<std::uint8_t>> obus = make_video_ref_frame_obus(
      config, header, refs, state, tiles, payloads, /*split_tile_groups=*/true,
      &error);
  check(obus.size() == 2, "tile groups: two OBUs");
  if (obus.size() == 2) {
    // Parse the first group.
    const std::size_t leb0 = (obus[0][0] & 0x80U) != 0U ? 2U : 1U;
    const auto p0 = std::span<const std::uint8_t>(obus[0])
                        .subspan(1U + leb0, obus[0].size() - 1U - leb0);
    RefBitReader r0(p0);
    bool first = false;
    check(r0.bit(&first) && first, "tile groups: group 0 is first");
    const std::size_t start = r0.position();
    VideoRefParsedHeader parsed;
    check(read_video_ref_frame_header(r0, false, config, refs, state, tiles,
                                   &parsed, &error),
          "tile groups: group 0 header");
    FrameHeaderStore store;
    store.record(p0, start, r0.position());
    bool flag = false;
    std::uint32_t tg_start = 9;
    std::uint32_t tg_end = 9;
    check(r0.bit(&flag) && flag && r0.bits(1U, &tg_start) &&
              r0.bits(1U, &tg_end) && tg_start == 0 && tg_end == 0,
          "tile groups: group 0 range is tile 0");
    // The remaining bytes after byte_alignment are the tile payload.
    const std::size_t header_bytes = (r0.position() + 7U) / 8U;
    check(p0.size() == header_bytes + payloads[0].size(),
          "tile groups: group 0 has no tile size field");
    check(std::memcmp(p0.data() + header_bytes, payloads[0].data(),
                      payloads[0].size()) == 0,
          "tile groups: group 0 carries tile 0");

    // Parse the second group and verify the header copy.
    const std::size_t leb1 = (obus[1][0] & 0x80U) != 0U ? 2U : 1U;
    const auto p1 = std::span<const std::uint8_t>(obus[1])
                        .subspan(1U + leb1, obus[1].size() - 1U - leb1);
    RefBitReader r1(p1);
    bool first1 = true;
    check(r1.bit(&first1) && !first1, "tile groups: group 1 is not first");
    bool present = false;
    check(read_video_ref_nonfirst_tile_group_header(r1, store, &present,
                                                 &error),
          "tile groups: group 1 header");
    check(present, "tile groups: group 1 repeats the header");
    std::uint32_t tg1_start = 9;
    std::uint32_t tg1_end = 9;
    check(r1.bit(&flag) && flag && r1.bits(1U, &tg1_start) &&
              r1.bits(1U, &tg1_end) && tg1_start == 1 && tg1_end == 1,
          "tile groups: group 1 range is tile 1");
    const std::size_t header_bytes1 = (r1.position() + 7U) / 8U;
    check(p1.size() == header_bytes1 + payloads[1].size(),
          "tile groups: group 1 has no tile size field");
    check(std::memcmp(p1.data() + header_bytes1, payloads[1].data(),
                      payloads[1].size()) == 0,
          "tile groups: group 1 carries tile 1");
  }
  // A single-group frame keeps the size field for every tile but the last and
  // sets tile_start_and_end_present_flag = 0.
  {
    const auto whole = make_video_ref_frame_obus(
        config, header, refs, state, tiles, payloads,
        /*split_tile_groups=*/false, &error);
    check(whole.size() == 1, "single group: one OBU");
    if (whole.size() == 1) {
      const std::size_t leb = (whole[0][0] & 0x80U) != 0U ? 2U : 1U;
      const auto p = std::span<const std::uint8_t>(whole[0])
                         .subspan(1U + leb, whole[0].size() - 1U - leb);
      RefBitReader r(p);
      bool first = false;
      check(r.bit(&first) && first, "single group: is first");
      VideoRefParsedHeader parsed;
      check(read_video_ref_frame_header(r, false, config, refs, state, tiles,
                                     &parsed, &error),
            "single group: header");
      bool flag = true;
      check(r.bit(&flag) && !flag, "single group: range flag = 0");
      const std::size_t header_bytes = (r.position() + 7U) / 8U;
      check(p.size() == header_bytes + 1U + payloads[0].size() +
                             payloads[1].size(),
            "single group: one size field for the first tile");
      check(p[header_bytes] == payloads[0].size() - 1U,
            "single group: tile 0 size field");
    }
  }
}

}  // namespace

int main() {
  test_reader_primitives();
  test_primary_ref_goldens();
  test_primary_ref_round_trip();
  test_primary_ref_resolution();
  test_drl_goldens();
  test_drl_round_trip();
  test_refresh_goldens();
  test_frame_size_with_refs();
  test_frame_header_copy();
  test_video_adapter_bit_exact();
  test_video_adapter_new_fields();
  test_video_adapter_tile_group_copy();

  std::printf("ref_fields_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
