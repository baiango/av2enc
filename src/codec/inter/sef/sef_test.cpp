// Unit tests for the show-existing-frame syntax and bookkeeping.
//
// The byte goldens are cross-checked against a real dav2d decode in
// sef_gate.cpp. The syntax values here are derived by hand from §5.18.2:
//
//   cur_mfh_id uvlc(0) = '1'
//   seq_header_id uvlc(0) = '1'
//   frame_to_show_map_idx f(1) = idx
//   derive_sef_order_hint f(1)
//   [sef_order_hint f(4)]
//   trailing_one_bit = '1', then zero bits to the byte boundary
//
// so (2 refs, 4 hint bits, regular, idx 0, derive=1) is '11' '0' '1' '1' =
// 0xD8, and the OBU is 02 30 D8. With derive=0 and hint=1 it is
// '11' '0' '0' '0001' '1' = C1 80, and the OBU is 03 30 C1 80.

#include "codec/inter/sef/sef.h"

#include "core/bitstream/bitwriter.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace sef = av2enc::inter::sef;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

std::string hex(const std::vector<std::uint8_t>& bytes) {
  std::string out;
  char buffer[4];
  for (const std::uint8_t byte : bytes) {
    std::snprintf(buffer, sizeof(buffer), "%02x", static_cast<unsigned>(byte));
    if (!out.empty()) out.push_back(' ');
    out += buffer;
  }
  return out;
}

void expect_bytes(const std::vector<std::uint8_t>& got,
                  const std::vector<std::uint8_t>& want,
                  const std::string& what) {
  const bool ok = got == want;
  check(ok, what + " (got [" + hex(got) + "], want [" + hex(want) + "])");
}

std::vector<std::uint8_t> bytes(std::initializer_list<unsigned> values) {
  std::vector<std::uint8_t> out;
  for (const unsigned value : values) {
    out.push_back(static_cast<std::uint8_t>(value));
  }
  return out;
}

sef::SefParams video_params() {
  sef::SefParams params;
  params.num_ref_frames = 2;
  params.order_hint_bits = 4;
  return params;
}

// ---------------------------------------------------------------------------

void test_bit_goldens() {
  const sef::SefParams params = video_params();

  {
    sef::SefHeader header;
    header.kind = sef::SefKind::kRegular;
    header.frame_to_show_map_idx = 0;
    header.derive_sef_order_hint = true;
    check(sef::sef_frame_header_bits(params, header) == 4,
          "derive=1 syntax bits (2 refs)");
    expect_bytes(sef::make_sef_obu(params, header), bytes({0x02, 0x30, 0xD8}),
                 "regular idx0 derive=1 OBU");
  }
  {
    sef::SefHeader header;
    header.kind = sef::SefKind::kLeading;
    header.frame_to_show_map_idx = 0;
    header.derive_sef_order_hint = true;
    expect_bytes(sef::make_sef_obu(params, header), bytes({0x02, 0x2C, 0xD8}),
                 "leading idx0 derive=1 OBU");
  }
  {
    sef::SefHeader header;
    header.kind = sef::SefKind::kRegular;
    header.frame_to_show_map_idx = 0;
    header.derive_sef_order_hint = false;
    header.sef_order_hint = 1;
    check(sef::sef_frame_header_bits(params, header) == 8,
          "derive=0 syntax bits (2 refs)");
    expect_bytes(sef::make_sef_obu(params, header),
                 bytes({0x03, 0x30, 0xC1, 0x80}), "regular idx0 hint=1 OBU");
  }
  {
    sef::SefHeader header;
    header.kind = sef::SefKind::kLeading;
    header.frame_to_show_map_idx = 0;
    header.derive_sef_order_hint = false;
    header.sef_order_hint = 1;
    expect_bytes(sef::make_sef_obu(params, header),
                 bytes({0x03, 0x2C, 0xC1, 0x80}), "leading idx0 hint=1 OBU");
  }
  {
    sef::SefHeader header;
    header.frame_to_show_map_idx = 1;
    header.derive_sef_order_hint = true;
    expect_bytes(sef::make_sef_obu(params, header), bytes({0x02, 0x30, 0xF8}),
                 "regular idx1 derive=1 OBU");
  }
  {
    sef::SefHeader header;
    header.frame_to_show_map_idx = 1;
    header.derive_sef_order_hint = false;
    header.sef_order_hint = 0;
    expect_bytes(sef::make_sef_obu(params, header),
                 bytes({0x03, 0x30, 0xE0, 0x80}), "regular idx1 hint=0 OBU");
  }
  {
    // Two index bits: idx 2, derive=0, hint 5.
    sef::SefParams four = params;
    four.num_ref_frames = 4;
    sef::SefHeader header;
    header.frame_to_show_map_idx = 2;
    header.derive_sef_order_hint = false;
    header.sef_order_hint = 5;
    check(sef::sef_frame_header_bits(four, header) == 9,
          "derive=0 syntax bits (4 refs)");
    expect_bytes(sef::make_sef_obu(four, header),
                 bytes({0x03, 0x30, 0xE2, 0xC0}), "regular idx2 hint=5 OBU");
  }
  {
    // 16 refs -> 4 index bits, derive=1: 1+1+4+1 = 7 syntax bits -> E8.
    sef::SefParams sixteen = params;
    sixteen.num_ref_frames = 16;
    sef::SefHeader header;
    header.frame_to_show_map_idx = 15;
    header.derive_sef_order_hint = true;
    check(sef::sef_frame_header_bits(sixteen, header) == 7,
          "derive=1 syntax bits (16 refs)");
    expect_bytes(sef::make_sef_obu(sixteen, header),
                 bytes({0x02, 0x30, 0xFF}), "regular idx15 derive=1 OBU");
  }
  {
    // Temporal unit = TD (01 08) + SEF OBU.
    sef::SefHeader header;
    header.frame_to_show_map_idx = 0;
    header.derive_sef_order_hint = false;
    header.sef_order_hint = 1;
    expect_bytes(sef::make_sef_temporal_unit(params, header),
                 bytes({0x01, 0x08, 0x03, 0x30, 0xC1, 0x80}),
                 "TD + SEF temporal unit");
  }
}

void test_bit_lengths() {
  const sef::SefParams params = video_params();
  sef::SefHeader header;
  header.derive_sef_order_hint = true;
  check(sef::sef_frame_header_bits(params, header) == 4,
        "non-zero ids change the syntax length");
  header.cur_mfh_id = 1;  // uvlc(1) = 010 -> 3 bits
  check(sef::sef_frame_header_bits(params, header) == 6,
        "cur_mfh_id uvlc length");

  // Non-zero ids are representable; the reader preserves them and the
  // validator rejects them because this module has no MFH state.
  const std::vector<std::uint8_t> obu = sef::make_sef_obu(params, header);
  sef::SefHeader parsed;
  std::size_t consumed = 0;
  sef::SefError error = sef::SefError::kOk;
  check(sef::parse_sef_obu(obu, params, parsed, consumed, &error),
        "parse non-zero cur_mfh_id");
  check(parsed.cur_mfh_id == 1, "round-trip cur_mfh_id");
  sef::SlotMap map;
  map.num_ref_frames = 2;
  check(!sef::validate_sef_reference(map, params, parsed, &error),
        "non-zero cur_mfh_id rejected");
  check(error == sef::SefError::kUnsupportedMfh, "MFH error name");
}

void test_round_trip() {
  const unsigned ref_counts[] = {1, 2, 3, 4, 8, 16};
  const unsigned hint_bits[] = {1, 2, 4, 8, 16};
  for (const unsigned refs : ref_counts) {
    for (const unsigned bits : hint_bits) {
      sef::SefParams params;
      params.num_ref_frames = static_cast<std::uint8_t>(refs);
      params.order_hint_bits = static_cast<std::uint8_t>(bits);
      for (const sef::SefKind kind :
           {sef::SefKind::kLeading, sef::SefKind::kRegular}) {
        for (unsigned idx = 0; idx < refs; ++idx) {
          for (const bool derive : {false, true}) {
            sef::SefHeader header;
            header.kind = kind;
            header.frame_to_show_map_idx = static_cast<std::uint8_t>(idx);
            header.derive_sef_order_hint = derive;
            header.sef_order_hint =
                derive ? 0U : static_cast<std::uint32_t>((idx * 7U) %
                                                         (1U << bits));
            const std::vector<std::uint8_t> obu =
                sef::make_sef_obu(params, header);
            sef::SefHeader parsed;
            std::size_t consumed = 0;
            sef::SefError error = sef::SefError::kOk;
            const bool ok = sef::parse_sef_obu(obu, params, parsed, consumed,
                                               &error);
            check(ok, std::string("round-trip parse: ") +
                           sef::sef_error_name(error));
            check(consumed == obu.size(), "round-trip consumed all bytes");
            check(parsed.kind == header.kind, "round-trip kind");
            check(parsed.frame_to_show_map_idx ==
                      header.frame_to_show_map_idx,
                  "round-trip index");
            check(parsed.derive_sef_order_hint ==
                      header.derive_sef_order_hint,
                  "round-trip derive flag");
            if (!derive) {
              check(parsed.sef_order_hint == header.sef_order_hint,
                    "round-trip explicit hint");
            }
          }
        }
      }
    }
  }
}

void test_reader_errors() {
  const sef::SefParams params = video_params();
  sef::SefHeader header;
  const std::vector<std::uint8_t> obu = sef::make_sef_obu(params, header);

  {
    std::vector<std::uint8_t> truncated(obu.begin(), obu.end() - 1);
    sef::SefHeader parsed;
    std::size_t consumed = 0;
    sef::SefError error = sef::SefError::kOk;
    check(!sef::parse_sef_obu(truncated, params, parsed, consumed, &error),
          "truncated OBU rejected");
    check(error == sef::SefError::kBadObuLength, "truncated OBU error");
  }
  {
    // Set one of the zero padding bits after trailing_one_bit.
    std::vector<std::uint8_t> bad = obu;
    bad.back() |= 0x01;
    sef::SefHeader parsed;
    std::size_t consumed = 0;
    sef::SefError error = sef::SefError::kOk;
    check(!sef::parse_sef_obu(bad, params, parsed, consumed, &error),
          "bad trailing bits rejected");
    check(error == sef::SefError::kBadTrailingBits, "bad trailing bits error");
  }
  {
    // Change obu_type to a tile group.
    std::vector<std::uint8_t> bad = obu;
    bad[1] = 0x1C;  // OBU_REGULAR_TILE_GROUP
    sef::SefHeader parsed;
    std::size_t consumed = 0;
    sef::SefError error = sef::SefError::kOk;
    check(!sef::parse_sef_obu(bad, params, parsed, consumed, &error),
          "non-SEF OBU rejected");
    check(error == sef::SefError::kBadObuHeader, "non-SEF OBU error");
  }
  {
    const std::vector<std::uint8_t> empty;
    sef::SefHeader parsed;
    std::size_t consumed = 0;
    sef::SefError error = sef::SefError::kOk;
    check(!sef::parse_sef_obu(empty, params, parsed, consumed, &error),
          "empty OBU rejected");
  }
}

// ---------------------------------------------------------------------------
// Slot / retention / output rules
// ---------------------------------------------------------------------------

sef::SlotMap make_video_slot_map(unsigned frame_count) {
  sef::SlotMap map;
  map.num_ref_frames = 2;
  for (unsigned i = 0; i < frame_count; ++i) {
    sef::RefreshFrame frame;
    frame.key_frame = i == 0;
    frame.order_hint = i;
    frame.frame_store_id = i;
    frame.immediate_output_frame = true;
    frame.implicit_output_frame = false;
    frame.refresh_frame_flags =
        i == 0 ? ((1U << map.num_ref_frames) - 1U) : 1U;
    sef::apply_refresh(map, frame);
    // The immediately-output frame is shown as part of decoding this frame.
    sef::mark_shown(map, 0);
  }
  return map;
}

void test_key_frame_retention() {
  sef::SlotMap map;
  map.num_ref_frames = 2;
  sef::RefreshFrame key;
  key.key_frame = true;
  key.order_hint = 0;
  key.frame_store_id = 100;
  key.immediate_output_frame = true;
  key.refresh_frame_flags = 3;  // allFrames
  sef::apply_refresh(map, key);
  check(map.slots[0].valid, "key frame: first refreshed slot is valid");
  check(!map.slots[1].valid,
        "key frame: later refreshed slots are RefValid=0 (§7.23 first)");

  sef::RefreshFrame p;
  p.key_frame = false;
  p.order_hint = 1;
  p.frame_store_id = 101;
  p.immediate_output_frame = true;
  p.refresh_frame_flags = 1;
  sef::apply_refresh(map, p);
  check(map.slots[0].valid && map.slots[0].frame_store_id == 101,
        "P frame overwrites slot 0");
  check(!map.slots[1].valid, "slot 1 stays invalid after a 2-slot M2 P frame");
}

void test_retention_and_show_selection() {
  {
    const sef::SlotMap map = make_video_slot_map(2);
    check(map.slots[0].valid && map.slots[0].frame_store_id == 1,
          "after 2 frames slot 0 holds frame 1");
    check(map.slots[0].shown, "immediate output marks the frame shown");
    check(map.max_disp_order_hint() == 1, "max display hint");

    sef::SefHeader header;
    sef::SefError error = sef::SefError::kOk;
    check(sef::make_show_header(map, video_params(), 0, sef::SefKind::kRegular,
                                header, &error),
          "shown frame can be re-shown with derive=0");
    check(!header.derive_sef_order_hint, "shown frame uses derive=0");
    check(header.sef_order_hint == 1, "explicit hint is the reference hint");
    check(sef::validate_sef_reference(map, video_params(), header, &error),
          "derive=0 validation passes");
    check(sef::validate_dav2d_compat(map, video_params(), header, &error),
          "dav2d explicit-hint compatibility passes");
    check(sef::dav2d_needs_output_invisible(map.slots[0]),
          "already-shown SEF needs --outputinvisible in dav2d");

    sef::SefHeader derive1;
    derive1.frame_to_show_map_idx = 0;
    derive1.derive_sef_order_hint = true;
    check(!sef::validate_sef_reference(map, video_params(), derive1, &error),
          "derive=1 rejects an already-shown frame");
    check(error == sef::SefError::kReferenceNotHidden,
          "derive=1 shown-frame error");
  }
  {
    // Hidden frame: derive=1 is the conformant and preferred path.
    sef::SlotMap map = make_video_slot_map(1);
    sef::RefreshFrame hidden;
    hidden.key_frame = false;
    hidden.order_hint = 1;
    hidden.frame_store_id = 1;
    hidden.immediate_output_frame = false;
    hidden.implicit_output_frame = false;
    hidden.refresh_frame_flags = 1;
    sef::apply_refresh(map, hidden);
    check(map.slots[0].valid && !map.slots[0].shown, "hidden frame is unshown");

    sef::SefHeader header;
    sef::SefError error = sef::SefError::kOk;
    check(sef::make_show_header(map, video_params(), 0, sef::SefKind::kRegular,
                                header, &error),
          "hidden frame can be shown with derive=1");
    check(header.derive_sef_order_hint, "hidden frame prefers derive=1");
    check(header.sef_order_hint == 0, "derive=1 codes no hint bits");
    check(sef::validate_sef_reference(map, video_params(), header, &error),
          "derive=1 hidden validation passes");
    check(!sef::dav2d_needs_output_invisible(map.slots[0]),
          "hidden derive=1 output is visible without extra flags");
  }
}

void test_reference_validation_errors() {
  const sef::SlotMap map = make_video_slot_map(2);
  sef::SefError error = sef::SefError::kOk;
  {
    sef::SefHeader header;
    header.frame_to_show_map_idx = 2;
    check(!sef::validate_sef_reference(map, video_params(), header, &error),
          "index out of range rejected");
    check(error == sef::SefError::kBadMapIdx, "index out of range error");
  }
  {
    sef::SefHeader header;
    header.frame_to_show_map_idx = 1;  // invalid slot after a key frame
    check(!sef::validate_sef_reference(map, video_params(), header, &error),
          "invalid slot rejected");
    check(error == sef::SefError::kInvalidReference, "invalid slot error");
  }
  {
    sef::SefHeader header;
    header.frame_to_show_map_idx = 0;
    header.derive_sef_order_hint = false;
    header.sef_order_hint = 0;  // does not reconstruct to hint 1
    check(sef::validate_sef_reference(map, video_params(), header, &error),
          "spec allows an explicit hint different from the reference");
    check(!sef::validate_dav2d_compat(map, video_params(), header, &error),
          "dav2d rejects a mismatched explicit hint");
    check(error == sef::SefError::kOrderHintMismatch, "dav2d hint mismatch");
  }
  {
    // §5.18.10.1: an SEF with film_grain_params_present = 1 codes
    // apply_grain/fgm_id/grain_seed in its own header. Round-trip it.
    sef::SefParams params = video_params();
    params.film_grain_params_present = true;
    sef::SefHeader header;
    header.grain.apply_grain = true;
    header.grain.fgm_id = 5;
    header.grain.grain_seed = 0x1234;
    const std::vector<std::uint8_t> obu = sef::make_sef_obu(params, header);
    sef::SefHeader parsed;
    std::size_t consumed = 0;
    check(sef::parse_sef_obu(obu, params, parsed, consumed, &error),
          "film grain SEF accepted by the reader");
    check(parsed.grain.apply_grain && parsed.grain.fgm_id == 5 &&
              parsed.grain.grain_seed == 0x1234,
          "film grain SEF config round-trips");
    check(sef::sef_frame_header_bits(params, header) ==
              sef::sef_frame_header_bits(video_params(), sef::SefHeader{}) + 1 + 3 + 16,
          "film grain SEF header bits include apply_grain/fgm_id/seed");
    // A disabled grain config costs exactly the apply_grain bit.
    sef::SefHeader off;
    check(sef::sef_frame_header_bits(params, off) ==
              sef::sef_frame_header_bits(video_params(), off) + 1,
          "apply_grain = 0 costs one bit");
    // film_grain_params_present = 0 keeps the historical zero-bit tail.
    check(sef::sef_frame_header_bits(video_params(), header) ==
              sef::sef_frame_header_bits(video_params(), sef::SefHeader{}),
          "default sequence codes no film grain bits");
  }
}

void test_display_order_hint() {
  // No offset when the target is inside the high half of the window.
  check(sef::disp_order_hint(1, 1, 4) == 1, "disp hint 1");
  check(sef::disp_order_hint(5, 12, 4) == 5,
        "disp hint stays when maxDisp - 8 < hint");
  check(sef::disp_order_hint(3, 3, 4) == 3, "disp hint 3");
  // Low hint with a much larger maxDisp wraps into the next window.
  check(sef::disp_order_hint(2, 12, 4) == 18,
        "disp hint wraps by OrderHintBits");

  std::uint32_t lsbs = 0;
  check(sef::sef_order_hint_lsbs(1, 1, 4, lsbs) && lsbs == 1,
        "lsbs for hint 1");
  check(sef::sef_order_hint_lsbs(3, 3, 4, lsbs) && lsbs == 3,
        "lsbs for hint 3");
  check(sef::sef_order_hint_lsbs(18, 12, 4, lsbs) && lsbs == 2,
        "lsbs for wrapped hint 18");
  check(!sef::sef_order_hint_lsbs(2, 12, 4, lsbs),
        "low hint 2 is not signalable when maxDisp is 12");
}

void test_naming() {
  check(std::string(sef::sef_kind_name(sef::SefKind::kLeading)) ==
            "OBU_LEADING_SEF",
        "leading kind name");
  check(std::string(sef::sef_kind_name(sef::SefKind::kRegular)) ==
            "OBU_REGULAR_SEF",
        "regular kind name");
  check(sef::ceil_log2(1) == 0, "ceil_log2(1)");
  check(sef::ceil_log2(2) == 1, "ceil_log2(2)");
  check(sef::ceil_log2(3) == 2, "ceil_log2(3)");
  check(sef::ceil_log2(16) == 4, "ceil_log2(16)");
}

}  // namespace

int main() {
  test_bit_goldens();
  test_bit_lengths();
  test_round_trip();
  test_reader_errors();
  test_key_frame_retention();
  test_retention_and_show_selection();
  test_reference_validation_errors();
  test_display_order_hint();
  test_naming();

  std::printf("sef_test: %d checks, %d failure(s)\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
