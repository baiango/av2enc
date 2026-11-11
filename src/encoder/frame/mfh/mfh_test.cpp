// Multi-frame header tests: §5.7 syntax, §6.7 semantics, §5.18 inheritance,
// pinned bit goldens, round-trips and error paths.
//
// Build/run: ./build_and_test.sh in this directory (no CMake changes).
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "bitstream/bitwriter.h"
#include "bitstream/m0_headers.h"
#include "encoder/frame/mfh/mfh.h"
#include "encoder/frame/mfh/mfh_m2_adapter.h"
#include "encoder/m2/m2_headers.h"

namespace {

using namespace av2enc::frame::mfh;
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
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  for (std::uint8_t byte : bytes) {
    out.push_back(kDigits[byte >> 4U]);
    out.push_back(kDigits[byte & 0x0FU]);
  }
  return out;
}

void check_hex(std::span<const std::uint8_t> bytes, const char* expected,
               const std::string& what) {
  const std::string actual = hex(bytes);
  check(actual == expected, what + ": got " + actual + ", want " + expected);
}

// A §5.7 payload that exercises every field.
MultiFrameHeader full_header(bool ext_seg) {
  MultiFrameHeader header;
  header.seq_header_id = 3;
  header.mfh_id = 5;
  header.tlayer_id = 2;
  header.frame_size_present = true;
  header.frame_width = 1920;
  header.frame_height = 1080;
  header.deblocking_filter_update = true;
  header.apply_deblocking_filter = {true, false, true, false};
  header.seg_info_present = true;
  header.allow_seg_info_change = true;
  header.seg_info.ext_seg = ext_seg;
  header.seg_info.feature[0][kSegLvlAltQ] = SegmentFeature{true, -37};
  header.seg_info.feature[1][kSegLvlAltQ] = SegmentFeature{true, 351};
  header.seg_info.feature[2][kSegLvlSkip] = SegmentFeature{true, 0};
  header.seg_info.feature[3][kSegLvlGlobalMv] = SegmentFeature{true, 0};
  header.seg_info.feature[4][kSegLvlAltQ] = SegmentFeature{false, 0};
  return header;
}

void test_pinned_goldens() {
  // Minimal: seq 0, id 1, no size, no deblocking, no segments.
  // Bits 1 1 0 0 0 + trailing 1 + pad -> 11000010.
  {
    MultiFrameHeader header;
    std::string error;
    const auto obu = make_multi_frame_header_obu(header, &error);
    check(error.empty(), "minimal golden: no error");
    check_hex(obu, "020cc2", "minimal MFH OBU");
  }

  // 64x64, frame size present with 6-bit fields, deblocking update with all
  // four apply flags 0, 8 segment rows all disabled:
  //   uvlc(0)=1, uvlc(0)=1, present=1, 0101, 0101, 111111, 111111,
  //   update=1, 0000, seg_present=1, ext=0, allow=0, 24 feature bits = 0
  //   = 55 syntax bits; + extension flag 0 + trailing 1 + 7 pad = 8 bytes.
  {
    MultiFrameHeader header;
    header.frame_size_present = true;
    header.frame_width = 64;
    header.frame_height = 64;
    header.deblocking_filter_update = true;
    header.apply_deblocking_filter = {false, false, false, false};
    header.seg_info_present = true;
    std::string error;
    const auto obu = make_multi_frame_header_obu(header, &error);
    check(error.empty(), "64x64 golden: no error");
    check_hex(obu, "090ceabfff0800000080", "64x64 MFH OBU");
  }

  // 1920x1080 with 11-bit size fields and deblocking {1,0,1,0}.
  {
    MultiFrameHeader header;
    header.frame_size_present = true;
    header.frame_width = 1920;
    header.frame_height = 1080;
    header.deblocking_filter_update = true;
    header.apply_deblocking_filter = {true, false, true, false};
    std::string error;
    const auto obu = make_multi_frame_header_obu(header, &error);
    check(error.empty(), "1080p golden: no error");
    check_hex(obu, "070cf55dfe1be880", "1080p MFH OBU");
  }
}

bool payload_bit(std::span<const std::uint8_t> payload, std::size_t index) {
  return ((payload[index >> 3U] >> (7U - (index & 7U))) & 1U) != 0U;
}

void test_obu_extension_flag_position() {
  std::string error;

  // §5.2.1: after the §5.7 payload, an extensible OBU codes
  //   obu_extension_flag f(1)
  //   trailing_bits()
  // The minimal MFH has 5 syntax bits, so the payload byte is
  //   bits 0..4 = 1 1 0 0 0 (syntax)
  //   bit 5     = 0       (obu_extension_flag)
  //   bit 6     = 1       (trailing_one_bit)
  //   bit 7     = 0       (trailing_zero_bit)
  MultiFrameHeader minimal;
  const auto minimal_obu = make_multi_frame_header_obu(minimal, &error);
  check(error.empty(), "ext flag: minimal write");
  ObuView minimal_view;
  check(parse_obu(minimal_obu, 0, &minimal_view, &error),
        "ext flag: minimal parse");
  const auto minimal_payload = std::span<const std::uint8_t>(minimal_obu)
                                   .subspan(minimal_view.payload_offset,
                                            minimal_view.payload_size);
  check(minimal_payload.size() == 1, "ext flag: minimal payload is one byte");
  check(minimal_payload[0] == 0xC2, "ext flag: minimal payload 0xc2");
  check(!payload_bit(minimal_payload, 5),
        "ext flag: bit 5 (obu_extension_flag) is 0");
  check(payload_bit(minimal_payload, 6),
        "ext flag: bit 6 is the trailing one bit");
  check(!payload_bit(minimal_payload, 7),
        "ext flag: bit 7 is trailing zero padding");

  // The OBU *header* extension flag is a different bit in a different byte.
  const std::uint8_t header_byte =
      minimal_obu[minimal_view.payload_offset - 1];
  check((header_byte & 0x80U) == 0,
        "ext flag: obu_header_extension_flag is 0");
  check(((header_byte >> 2U) & 0x1FU) == kObuTypeMultiFrameHeader,
        "ext flag: OBU header type is 3");

  // For a field-rich header, the flag sits exactly at the end of the §5.7
  // syntax, followed by the trailing one bit.
  MultiFrameHeader full;
  full.frame_size_present = true;
  full.frame_width = 64;
  full.frame_height = 64;
  full.deblocking_filter_update = true;
  full.apply_deblocking_filter = {false, false, false, false};
  full.seg_info_present = true;
  BitWriter syntax;
  const std::size_t syntax_bits =
      write_multi_frame_header_syntax(syntax, full, &error);
  check(syntax_bits == 55, "ext flag: 64x64 syntax is 55 bits");
  const auto full_obu = make_multi_frame_header_obu(full, &error);
  ObuView full_view;
  check(parse_obu(full_obu, 0, &full_view, &error),
        "ext flag: full parse");
  const auto full_payload = std::span<const std::uint8_t>(full_obu)
                                .subspan(full_view.payload_offset,
                                         full_view.payload_size);
  check(!payload_bit(full_payload, syntax_bits),
        "ext flag: flag is the first bit after the syntax");
  check(payload_bit(full_payload, syntax_bits + 1U),
        "ext flag: trailing one follows the flag");

  // A reader must reject obu_extension_flag == 1.  This payload is a valid
  // 7-bit header (seq id 1, id 1, nothing else) with the flag in bit 7 and no
  // extension data or trailing bits after it.
  const std::vector<std::uint8_t> flag_set_without_data{0x51};
  MultiFrameHeader parsed;
  check(!parse_multi_frame_header_payload(flag_set_without_data, &parsed,
                                          &error),
        "ext flag: reader rejects flag=1 with no extension data");
  check(error.find("obu_extension_flag") != std::string::npos,
        "ext flag: reader error names obu_extension_flag");
}

void test_round_trip() {
  for (bool ext_seg : {false, true}) {
    MultiFrameHeader header = full_header(ext_seg);
    std::string error;
    const auto obu = make_multi_frame_header_obu(header, &error);
    check(error.empty(), "round-trip: write");
    check(!obu.empty(), "round-trip: non-empty OBU");

    MultiFrameHeader parsed;
    ObuView view;
    check(parse_multi_frame_header_obu(obu, 0, &parsed, &view, &error),
          "round-trip: parse OBU: " + error);
    check(view.obu_type == kObuTypeMultiFrameHeader, "round-trip: OBU type 3");
    check(view.tlayer_id == header.tlayer_id, "round-trip: tlayer");
    check(parsed.seq_header_id == header.seq_header_id,
          "round-trip: seq_header_id");
    check(parsed.mfh_id == header.mfh_id, "round-trip: mfh_id");
    check(parsed.tlayer_id == header.tlayer_id, "round-trip: tlayer copied");
    check(parsed.frame_size_present, "round-trip: size present");
    check(parsed.frame_width == header.frame_width, "round-trip: width");
    check(parsed.frame_height == header.frame_height, "round-trip: height");
    check(parsed.deblocking_filter_update, "round-trip: deblock update");
    check(parsed.apply_deblocking_filter == header.apply_deblocking_filter,
          "round-trip: deblock flags");
    check(parsed.seg_info_present, "round-trip: seg present");
    check(parsed.seg_info.ext_seg == ext_seg, "round-trip: ext_seg");
    check(parsed.allow_seg_info_change, "round-trip: allow seg change");
    check(parsed.seg_info.feature[0][kSegLvlAltQ].enabled &&
              parsed.seg_info.feature[0][kSegLvlAltQ].value == -37,
          "round-trip: ALT_Q -37");
    check(parsed.seg_info.feature[1][kSegLvlAltQ].value == 351,
          "round-trip: ALT_Q 351");
    check(parsed.seg_info.feature[4][kSegLvlAltQ].enabled == false,
          "round-trip: disabled feature");
  }

  // Explicit bit widths survive the round-trip even when wider than needed.
  {
    MultiFrameHeader header;
    header.frame_size_present = true;
    header.frame_width_bits = 12;
    header.frame_height_bits = 13;
    header.frame_width = 320;
    header.frame_height = 240;
    std::string error;
    const auto obu = make_multi_frame_header_obu(header, &error);
    check(error.empty(), "wide size bits: write");
    MultiFrameHeader parsed;
    check(parse_multi_frame_header_obu(obu, 0, &parsed, nullptr, &error),
          "wide size bits: parse");
    check(parsed.frame_width_bits == 12 && parsed.frame_height_bits == 13,
          "wide size bits preserved");
    check(parsed.frame_width == 320 && parsed.frame_height == 240,
          "wide size values preserved");
  }
}

void test_scan_and_store() {
  MultiFrameHeader first;
  first.mfh_id = 1;
  first.tlayer_id = 1;
  MultiFrameHeader second = full_header(false);
  second.mfh_id = 7;
  second.tlayer_id = 3;

  std::string error;
  const auto mfh1 = make_multi_frame_header_obu(first, &error);
  const auto mfh2 = make_multi_frame_header_obu(second, &error);
  std::vector<std::uint8_t> stream;
  const auto td = av2enc::m0::make_obu(
      av2enc::m0::ObuHeader{av2enc::m0::ObuType::kTemporalDelimiter}, {});
  stream.insert(stream.end(), td.begin(), td.end());
  stream.insert(stream.end(), mfh1.begin(), mfh1.end());
  stream.insert(stream.end(), mfh2.begin(), mfh2.end());

  MfhStore store;
  std::size_t count = 0;
  check(scan_multi_frame_headers(stream, &store, &count, &error),
        "scan: " + error);
  check(count == 2, "scan: two MFHs");
  check(store.size() == 2, "scan: store size");
  const MultiFrameHeader* found = store.find(1);
  check(found != nullptr && found->tlayer_id == 1, "scan: id 1 tlayer");
  found = store.find(7);
  check(found != nullptr && found->mfh_id == 7, "scan: id 7");
  check(store.find(0) == nullptr, "scan: id 0 absent");
  check(store.find(15) == nullptr, "scan: unused id absent");

  // Overwrite keeps the count stable.
  MultiFrameHeader replacement;
  replacement.mfh_id = 1;
  replacement.seq_header_id = 2;
  check(store.insert(replacement, &error), "store: overwrite");
  check(store.size() == 2, "store: overwrite does not grow");
  check(store.find(1)->seq_header_id == 2, "store: overwrite value");
  store.reset();
  check(store.size() == 0 && store.find(1) == nullptr, "store: reset");
}

void test_validation_errors() {
  auto expect_write_error = [](MultiFrameHeader header, const char* what) {
    std::string error;
    BitWriter writer;
    const std::size_t bits =
        write_multi_frame_header_syntax(writer, header, &error);
    check(bits == 0 && !error.empty(), std::string("reject: ") + what);
  };

  {
    MultiFrameHeader h;
    h.seq_header_id = 16;
    expect_write_error(h, "seq_header_id == MAX_SEQ_NUM");
  }
  {
    MultiFrameHeader h;
    h.mfh_id = 0;
    expect_write_error(h, "mfh_id 0");
  }
  {
    MultiFrameHeader h;
    h.mfh_id = 16;
    expect_write_error(h, "mfh_id == MAX_MFH_NUM");
  }
  {
    MultiFrameHeader h;
    h.tlayer_id = 4;
    expect_write_error(h, "tlayer_id 4");
  }
  {
    MultiFrameHeader h;
    h.frame_size_present = true;
    expect_write_error(h, "size present with zero dimensions");
  }
  {
    MultiFrameHeader h;
    h.frame_size_present = true;
    h.frame_width = 64;
    h.frame_height = 64;
    h.frame_width_bits = 5;  // 64 - 1 = 63 needs 6 bits
    expect_write_error(h, "width does not fit in the coded bits");
  }
  {
    MultiFrameHeader h;
    h.frame_size_present = true;
    h.frame_width = 64;
    h.frame_height = 64;
    h.frame_width_bits = 17;
    expect_write_error(h, "size bits > 16");
  }
  {
    MultiFrameHeader h;
    h.frame_width = 64;  // no present flag
    expect_write_error(h, "size fields without present flag");
  }
  {
    MultiFrameHeader h;
    h.apply_deblocking_filter[0] = true;
    expect_write_error(h, "deblocking flags without update");
  }
  {
    MultiFrameHeader h;
    h.allow_seg_info_change = true;
    expect_write_error(h, "allow_seg_info_change without seg info");
  }
  {
    MultiFrameHeader h;
    h.seg_info.feature[0][kSegLvlAltQ] = SegmentFeature{true, 1};
    expect_write_error(h, "segment features without seg info");
  }
  {
    MultiFrameHeader h;
    h.seg_info_present = true;
    h.seg_info.feature[0][kSegLvlAltQ] = SegmentFeature{true, 352};
    expect_write_error(h, "ALT_Q above MAXQ_BITS");
  }
  {
    MultiFrameHeader h;
    h.seg_info_present = true;
    h.seg_info.feature[0][kSegLvlAltQ] = SegmentFeature{true, -352};
    expect_write_error(h, "ALT_Q below -MAXQ_BITS");
  }
  {
    MultiFrameHeader h;
    h.seg_info_present = true;
    h.seg_info.feature[0][kSegLvlSkip] = SegmentFeature{true, 1};
    expect_write_error(h, "SEG_LVL_SKIP must be zero");
  }
  {
    MultiFrameHeader h;
    h.seg_info_present = true;
    h.seg_info.ext_seg = false;
    h.seg_info.feature[9][kSegLvlAltQ] = SegmentFeature{true, 1};
    expect_write_error(h, "feature row outside 8 segments");
  }

  // Reader-side malformed payloads.
  {
    MultiFrameHeader out;
    std::string error;
    check(!parse_multi_frame_header_payload(std::span<const std::uint8_t>{},
                                            &out, &error),
          "reject empty payload");
  }
  {
    // uvlc(16) for mfh_seq_header_id, then a minimal tail.
    auto uvlc = [](std::uint32_t value) {
      unsigned leading_zeros = 0;
      while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
        ++leading_zeros;
      }
      std::string bits(leading_zeros, '0');
      bits.push_back('1');
      const std::uint32_t payload =
          value - static_cast<std::uint32_t>((1ULL << leading_zeros) - 1ULL);
      for (int i = static_cast<int>(leading_zeros) - 1; i >= 0; --i) {
        bits.push_back(((payload >> i) & 1U) != 0U ? '1' : '0');
      }
      return bits;
    };
    auto pack = [](std::string bits) {
      bits.push_back('0');  // obu_extension_flag (§5.2.1)
      bits.push_back('1');  // trailing_bits
      while ((bits.size() & 7U) != 0U) bits.push_back('0');
      std::vector<std::uint8_t> bytes;
      for (std::size_t i = 0; i < bits.size(); i += 8) {
        std::uint8_t byte = 0;
        for (std::size_t j = 0; j < 8; ++j) {
          byte = static_cast<std::uint8_t>(
              (static_cast<unsigned>(byte) << 1U) |
              (bits[i + j] == '1' ? 1U : 0U));
        }
        bytes.push_back(byte);
      }
      return bytes;
    };

    const auto bad_seq = pack(uvlc(16) + uvlc(0) + "000");
    MultiFrameHeader out;
    std::string error;
    check(!parse_multi_frame_header_payload(bad_seq, &out, &error),
          "reject mfh_seq_header_id >= MAX_SEQ_NUM");

    const auto bad_id = pack(uvlc(0) + uvlc(15) + "000");
    check(!parse_multi_frame_header_payload(bad_id, &out, &error),
          "reject mfh_id >= MAX_MFH_NUM");

    const auto bad_trailing = pack(uvlc(0) + uvlc(0) + "000");
    std::vector<std::uint8_t> flipped = bad_trailing;
    flipped.back() |= 0x01U;  // set a nonzero padding bit
    check(!parse_multi_frame_header_payload(flipped, &out, &error),
          "reject nonzero trailing padding");
  }
  {
    // A valid OBU with the trailing 1 bit flipped to 0.
    MultiFrameHeader header;
    std::string error;
    auto obu = make_multi_frame_header_obu(header, &error);
    obu.back() = 0x00;  // 0xC2 -> 0x00 loses the trailing one bit
    MultiFrameHeader out;
    check(!parse_multi_frame_header_obu(obu, 0, &out, nullptr, &error),
          "reject corrupted trailing bits");
  }
  {
    // Truncated leb128/payload.
    std::string error;
    auto obu = make_multi_frame_header_obu(full_header(false), &error);
    obu.resize(obu.size() - 3);
    MultiFrameHeader out;
    check(!parse_multi_frame_header_obu(obu, 0, &out, nullptr, &error),
          "reject truncated MFH OBU");
  }
  {
    std::string error;
    ObuView view;
    MultiFrameHeader out;
    const std::vector<std::uint8_t> empty;
    check(!parse_obu(empty, 0, &view, &error), "reject OBU past end");
    const std::vector<std::uint8_t> truncated_leb{0x80};
    check(!parse_obu(truncated_leb, 0, &view, &error),
          "reject truncated leb128");
    const std::vector<std::uint8_t> zero_size{0x00};
    check(!parse_obu(zero_size, 0, &view, &error), "reject zero-size OBU");
    const std::vector<std::uint8_t> short_payload{0x05, 0x0c};
    check(!parse_obu(short_payload, 0, &view, &error),
          "reject truncated OBU payload");
    check(!parse_multi_frame_header_payload(short_payload, &out, &error),
          "reject short payload body");
  }
  {
    // Wrong OBU type.
    std::string error;
    const auto td = av2enc::m0::make_obu(
        av2enc::m0::ObuHeader{av2enc::m0::ObuType::kTemporalDelimiter}, {});
    MultiFrameHeader out;
    check(!parse_multi_frame_header_obu(td, 0, &out, nullptr, &error),
          "reject non-MFH OBU");
  }
  {
    // Extension flag set: §6.2.1 forbids it in conforming bitstreams.
    std::string error;
    MultiFrameHeader header;
    const auto obu = make_multi_frame_header_obu(header, &error);
    std::vector<std::uint8_t> extended = obu;
    extended[1] |= 0x80U;  // obu_extension_flag
    extended.insert(extended.begin() + 2, 0x00);  // extension byte
    extended[0] = static_cast<std::uint8_t>(extended[0] + 1);
    MultiFrameHeader out;
    check(!parse_multi_frame_header_obu(extended, 0, &out, nullptr, &error),
          "reject obu_extension_flag");
  }
}

SequenceFrameFields sequence_64x64(bool ext_seg = false) {
  SequenceFrameFields sequence;
  sequence.max_frame_width = 64;
  sequence.max_frame_height = 64;
  sequence.enable_ext_seg = ext_seg;
  sequence.sequence_headers[0] = true;
  sequence.sequence_headers[1] = true;
  return sequence;
}

MfhStore store_with(const MultiFrameHeader& header) {
  MfhStore store;
  std::string error;
  const bool ok = store.insert(header, &error);
  check(ok, "fixture store insert: " + error);
  return store;
}

void test_size_precedence() {
  std::string error;
  SequenceFrameFields sequence = sequence_64x64();

  // No MFH: sequence maximum applies.
  {
    FrameHeaderFields frame;
    ResolvedFrameFields out;
    check(resolve_frame_fields(MfhStore{}, sequence, frame, nullptr, &out,
                               &error),
          "size seq: " + error);
    check(out.frame_width == 64 && out.frame_height == 64,
          "size seq: maximum");
    check(out.frame_size_source == FieldSource::kSequence,
          "size seq: source");
    check(out.deblocking_source == FieldSource::kExplicit,
          "size seq: explicit deblocking");
    check(out.deblocking_bits_present, "size seq: deblocking bits present");
  }

  // MFH size present: used when the frame does not override.
  {
    MultiFrameHeader mfh;
    mfh.mfh_id = 1;
    mfh.frame_size_present = true;
    mfh.frame_width = 48;
    mfh.frame_height = 32;
    const MfhStore store = store_with(mfh);
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    ResolvedFrameFields out;
    check(resolve_frame_fields(store, sequence, frame, nullptr, &out, &error),
          "size mfh: " + error);
    check(out.frame_width == 48 && out.frame_height == 32,
          "size mfh: inherited");
    check(out.frame_size_source == FieldSource::kMfh, "size mfh: source");
    check(out.seq_header_id == 0, "size mfh: sequence via MFH");
  }

  // Explicit frame_size_override wins over the MFH.
  {
    MultiFrameHeader mfh;
    mfh.mfh_id = 1;
    mfh.frame_size_present = true;
    mfh.frame_width = 48;
    mfh.frame_height = 32;
    const MfhStore store = store_with(mfh);
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    frame.frame_size_override = true;
    frame.frame_width = 64;
    frame.frame_height = 64;
    ResolvedFrameFields out;
    check(resolve_frame_fields(store, sequence, frame, nullptr, &out, &error),
          "size override: " + error);
    check(out.frame_width == 64 && out.frame_height == 64,
          "size override: explicit wins");
    check(out.frame_size_source == FieldSource::kExplicit,
          "size override: source");
  }

  // MFH without the frame size flag falls back to the sequence maximum.
  {
    MultiFrameHeader mfh;
    mfh.mfh_id = 1;
    const MfhStore store = store_with(mfh);
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    ResolvedFrameFields out;
    check(resolve_frame_fields(store, sequence, frame, nullptr, &out, &error),
          "size absent: " + error);
    check(out.frame_size_source == FieldSource::kSequence,
          "size absent: sequence source");
  }

  // Errors: missing MFH, missing sequence header, oversized MFH.
  {
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    ResolvedFrameFields out;
    check(!resolve_frame_fields(MfhStore{}, sequence, frame, nullptr, &out,
                                &error),
          "error: missing MFH");
  }
  {
    MultiFrameHeader mfh;
    mfh.mfh_id = 1;
    mfh.seq_header_id = 5;
    const MfhStore store = store_with(mfh);
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    ResolvedFrameFields out;
    check(!resolve_frame_fields(store, sequence, frame, nullptr, &out, &error),
          "error: MFH sequence header unavailable");
  }
  {
    MultiFrameHeader mfh;
    mfh.mfh_id = 1;
    mfh.frame_size_present = true;
    mfh.frame_width = 128;
    mfh.frame_height = 64;
    const MfhStore store = store_with(mfh);
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    ResolvedFrameFields out;
    check(!resolve_frame_fields(store, sequence, frame, nullptr, &out, &error),
          "error: MFH size above sequence maximum");
  }
  {
    FrameHeaderFields frame;
    frame.frame_size_override = true;
    frame.frame_width = 128;
    frame.frame_height = 64;
    ResolvedFrameFields out;
    check(!resolve_frame_fields(MfhStore{}, sequence, frame, nullptr, &out,
                                &error),
          "error: explicit size above sequence maximum");
  }
  {
    FrameHeaderFields frame;
    frame.seq_header_id = 4;
    ResolvedFrameFields out;
    check(!resolve_frame_fields(MfhStore{}, sequence, frame, nullptr, &out,
                                &error),
          "error: frame sequence header unavailable");
  }
}

void test_deblocking_precedence() {
  std::string error;
  SequenceFrameFields sequence = sequence_64x64();
  MultiFrameHeader mfh;
  mfh.mfh_id = 1;
  mfh.deblocking_filter_update = true;
  mfh.apply_deblocking_filter = {true, false, true, false};
  const MfhStore store = store_with(mfh);

  {
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    // Explicit values are absent from the bitstream; the resolver must ignore
    // them and use the MFH.
    frame.apply_deblocking_filter = {false, true, false, true};
    ResolvedFrameFields out;
    check(resolve_frame_fields(store, sequence, frame, nullptr, &out, &error),
          "deblock mfh: " + error);
    check(out.deblocking_source == FieldSource::kMfh, "deblock mfh: source");
    check(!out.deblocking_bits_present, "deblock mfh: apply bits absent");
    check(out.apply_deblocking_filter ==
              std::array<bool, 4>{true, false, true, false},
          "deblock mfh: flags inherited");
    check(out.deblocking_delta_q_bits_present[0] &&
              !out.deblocking_delta_q_bits_present[1] &&
              out.deblocking_delta_q_bits_present[2] &&
              !out.deblocking_delta_q_bits_present[3],
          "deblock mfh: df_delta_q bits follow enabled planes");
  }

  // Luma disabled forces chroma off even when the MFH sets chroma flags.
  {
    MultiFrameHeader chroma = mfh;
    chroma.apply_deblocking_filter = {false, false, true, true};
    const MfhStore chroma_store = store_with(chroma);
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    ResolvedFrameFields out;
    check(resolve_frame_fields(chroma_store, sequence, frame, nullptr, &out,
                               &error),
          "deblock chroma: " + error);
    check(!out.apply_deblocking_filter[2] && !out.apply_deblocking_filter[3],
          "deblock chroma forced off when luma is off");
  }

  // Monochrome: no chroma planes at all.
  {
    SequenceFrameFields mono = sequence;
    mono.monochrome = true;
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    ResolvedFrameFields out;
    check(resolve_frame_fields(store, mono, frame, nullptr, &out, &error),
          "deblock mono: " + error);
    check(!out.apply_deblocking_filter[2] && !out.apply_deblocking_filter[3],
          "deblock mono: chroma off");
  }

  // CodedLossless wins over the MFH and codes no bits.
  {
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    frame.coded_lossless = true;
    ResolvedFrameFields out;
    check(resolve_frame_fields(store, sequence, frame, nullptr, &out, &error),
          "deblock lossless: " + error);
    check(out.deblocking_source == FieldSource::kForced,
          "deblock lossless: forced");
    check(out.apply_deblocking_filter ==
              std::array<bool, 4>{false, false, false, false},
          "deblock lossless: all off");
    check(!out.deblocking_bits_present, "deblock lossless: no bits");
  }

  // No update flag: frame-header bits are present and win.
  {
    MultiFrameHeader plain;
    plain.mfh_id = 1;
    const MfhStore plain_store = store_with(plain);
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    frame.apply_deblocking_filter = {true, true, false, false};
    ResolvedFrameFields out;
    check(resolve_frame_fields(plain_store, sequence, frame, nullptr, &out,
                               &error),
          "deblock explicit: " + error);
    check(out.deblocking_source == FieldSource::kExplicit,
          "deblock explicit: source");
    check(out.deblocking_bits_present, "deblock explicit: bits present");
    check(out.apply_deblocking_filter ==
              std::array<bool, 4>{true, true, false, false},
          "deblock explicit: values");
  }
}

void test_segmentation_precedence() {
  std::string error;
  SequenceFrameFields sequence = sequence_64x64();
  sequence.seq_seg_info_present = true;
  sequence.seq_allow_seg_info_change = false;
  sequence.seq_seg_info.feature[0][kSegLvlAltQ] = SegmentFeature{true, 10};

  MultiFrameHeader mfh;
  mfh.mfh_id = 1;
  mfh.seg_info_present = true;
  mfh.seg_info.feature[0][kSegLvlAltQ] = SegmentFeature{true, -20};
  const MfhStore store = store_with(mfh);

  // MFH has priority over sequence seg info when segmentation is enabled and
  // allow_change is 0: reuse without a bit.
  {
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    frame.segmentation_enabled = true;
    ResolvedFrameFields out;
    check(resolve_frame_fields(store, sequence, frame, nullptr, &out, &error),
          "seg mfh reuse: " + error);
    check(out.seg_info_source == FieldSource::kMfh, "seg mfh reuse: source");
    check(out.reuse_seg_info && !out.reuse_seg_info_bit_present,
          "seg mfh reuse: derived reuse, no bit");
    check(out.seg_info.feature[0][kSegLvlAltQ].value == -20,
          "seg mfh reuse: MFH value");
    check(out.max_segments == 8, "seg mfh reuse: 8 segments");
  }

  // allow_change == 1 codes reuse_seg_info; false selects a new explicit table.
  {
    MultiFrameHeader change = mfh;
    change.allow_seg_info_change = true;
    const MfhStore change_store = store_with(change);
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    frame.segmentation_enabled = true;
    frame.reuse_seg_info = false;
    frame.seg_info.feature[1][kSegLvlAltQ] = SegmentFeature{true, 7};
    ResolvedFrameFields out;
    check(resolve_frame_fields(change_store, sequence, frame, nullptr, &out,
                               &error),
          "seg change explicit: " + error);
    check(out.reuse_seg_info_bit_present, "seg change: bit present");
    check(!out.reuse_seg_info, "seg change: reuse false");
    check(out.seg_info_source == FieldSource::kExplicit,
          "seg change: explicit source");
    check(out.seg_info.feature[1][kSegLvlAltQ].value == 7,
          "seg change: explicit value");
  }
  {
    MultiFrameHeader change = mfh;
    change.allow_seg_info_change = true;
    const MfhStore change_store = store_with(change);
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    frame.segmentation_enabled = true;
    frame.reuse_seg_info = true;
    ResolvedFrameFields out;
    check(resolve_frame_fields(change_store, sequence, frame, nullptr, &out,
                               &error),
          "seg change reuse: " + error);
    check(out.reuse_seg_info, "seg change reuse: true");
    check(out.seg_info_source == FieldSource::kMfh,
          "seg change reuse: MFH source");
  }

  // ext_seg mismatch: the MFH branch is taken but haveSegParams is false, so
  // the sequence table is NOT used; a new table must be coded.
  {
    MultiFrameHeader ext = mfh;
    ext.seg_info.ext_seg = true;  // sequence enable_ext_seg == 0
    const MfhStore ext_store = store_with(ext);
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    frame.segmentation_enabled = true;
    frame.seg_info.feature[2][kSegLvlAltQ] = SegmentFeature{true, 3};
    ResolvedFrameFields out;
    check(resolve_frame_fields(ext_store, sequence, frame, nullptr, &out,
                               &error),
          "seg ext mismatch: " + error);
    check(out.seg_info_source == FieldSource::kExplicit,
          "seg ext mismatch: explicit new table");
    check(!out.reuse_seg_info, "seg ext mismatch: no reuse");
    check(out.seg_info.feature[2][kSegLvlAltQ].value == 3,
          "seg ext mismatch: explicit value");
  }

  // No MFH seg info: sequence table is reused when allowed.
  {
    MultiFrameHeader plain;
    plain.mfh_id = 1;
    const MfhStore plain_store = store_with(plain);
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    frame.segmentation_enabled = true;
    ResolvedFrameFields out;
    check(resolve_frame_fields(plain_store, sequence, frame, nullptr, &out,
                               &error),
          "seg sequence: " + error);
    check(out.seg_info_source == FieldSource::kSequence,
          "seg sequence: source");
    check(out.seg_info.feature[0][kSegLvlAltQ].value == 10,
          "seg sequence: value");
  }

  // Extended segments: MaxSegments follows the sequence enable_ext_seg.
  {
    SequenceFrameFields ext_sequence = sequence_64x64(true);
    ext_sequence.seq_seg_info_present = false;
    MultiFrameHeader ext;
    ext.mfh_id = 1;
    ext.seg_info_present = true;
    ext.seg_info.ext_seg = true;
    ext.seg_info.feature[15][kSegLvlAltQ] = SegmentFeature{true, 100};
    const MfhStore ext_store = store_with(ext);
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    frame.segmentation_enabled = true;
    ResolvedFrameFields out;
    check(resolve_frame_fields(ext_store, ext_sequence, frame, nullptr, &out,
                               &error),
          "seg ext: " + error);
    check(out.max_segments == 16, "seg ext: 16 segments");
    check(out.seg_info.feature[15][kSegLvlAltQ].value == 100,
          "seg ext: row 15 inherited");
  }

  // segmentation_enabled == 0: nothing is inherited.
  {
    FrameHeaderFields frame;
    frame.cur_mfh_id = 1;
    frame.segmentation_enabled = false;
    ResolvedFrameFields out;
    check(resolve_frame_fields(store, sequence, frame, nullptr, &out, &error),
          "seg disabled: " + error);
    check(out.seg_info_source == FieldSource::kAbsent,
          "seg disabled: absent");
  }
}

void test_layer_dependencies() {
  std::string error;
  SequenceFrameFields sequence = sequence_64x64();
  MultiFrameHeader mfh;
  mfh.mfh_id = 1;
  mfh.tlayer_id = 1;
  const MfhStore store = store_with(mfh);

  FrameHeaderFields frame;
  frame.cur_mfh_id = 1;
  frame.obu_mlayer_id = 0;
  frame.obu_tlayer_id = 1;
  ResolvedFrameFields out;

  LayerDependencyMap deps = LayerDependencyMap::identity();
  deps.tlayer[0][1][1] = true;
  check(resolve_frame_fields(store, sequence, frame, &deps, &out, &error),
        "layer deps pass: " + error);

  LayerDependencyMap bad_t = LayerDependencyMap::identity();
  bad_t.tlayer[0][1][1] = false;  // does not include MfhTLayerId 1
  bad_t.tlayer[0][1][0] = true;
  check(!resolve_frame_fields(store, sequence, frame, &bad_t, &out, &error),
        "layer deps: temporal dependency violation");

  LayerDependencyMap bad_m = LayerDependencyMap::identity();
  bad_m.mlayer[0][0] = false;
  check(!resolve_frame_fields(store, sequence, frame, &bad_m, &out, &error),
        "layer deps: layer dependency violation");

  // Without maps the embedded layer must be 0; a non-zero frame mlayer is
  // ambiguous and rejected.
  frame.obu_mlayer_id = 1;
  check(!resolve_frame_fields(store, sequence, frame, nullptr, &out, &error),
        "layer deps: missing map for multi-layer reference");
  frame.obu_mlayer_id = 0;

  // The default temporal rule allows t >= MfhTLayerId but not t < it.
  MultiFrameHeader forward;
  forward.mfh_id = 1;
  forward.tlayer_id = 2;
  const MfhStore forward_store = store_with(forward);
  frame.obu_tlayer_id = 1;
  check(!resolve_frame_fields(forward_store, sequence, frame, nullptr, &out,
                              &error),
        "layer deps: forward temporal reference without a map");
  frame.obu_tlayer_id = 2;
  check(resolve_frame_fields(forward_store, sequence, frame, nullptr, &out,
                             &error),
        "layer deps: equal temporal reference uses the default");
}

void test_m2_adapter_bit_exact() {
  for (const auto [width, height] :
       {std::pair<std::uint16_t, std::uint16_t>{64, 64},
        {256, 256}, {1280, 720}}) {
    av2enc::m2::M2Config m2_config;
    m2_config.max_frame_width = width;
    m2_config.max_frame_height = height;
    m2_config.base_q_idx = 90;
    m2_config.order_hint_bits = 4;
    m2_config.num_ref_frames = 2;
    m2_config.num_total_refs = 1;
    m2_config.disable_cdf_update = true;

    M2MfhConfig config;
    config.max_frame_width = width;
    config.max_frame_height = height;
    config.base_q_idx = 90;
    config.order_hint_bits = 4;
    config.num_ref_frames = 2;
    config.num_total_refs = 1;
    config.disable_cdf_update = true;

    std::string error;
    for (bool key : {true, false}) {
      av2enc::m2::M2FrameHeader m2_header;
      m2_header.keyframe = key;
      m2_header.order_hint = key ? 0U : 1U;
      m2_header.ref_slot = 0;
      m2_header.refresh_frame_flags = key ? 3U : 1U;

      M2MfhFrameHeader header;
      header.keyframe = key;
      header.frame_is_inter = !key;
      header.order_hint = m2_header.order_hint;
      header.ref_slot = 0;
      header.refresh_frame_flags = m2_header.refresh_frame_flags;

      BitWriter reference;
      av2enc::m2::write_tile_group_header(reference, m2_config, m2_header);
      BitWriter adapter;
      const std::size_t adapter_bits = write_m2_tile_group_header_with_mfh(
          adapter, config, header, nullptr, &error);
      check(error.empty(), "adapter baseline: no error");
      check(adapter_bits == reference.bit_size(), "adapter baseline: bit count");
      check(adapter.bytes().size() == reference.bytes().size(),
            "adapter baseline: byte count");
      check(std::memcmp(adapter.bytes().data(), reference.bytes().data(),
                        reference.bytes().size()) == 0,
            "adapter baseline: bytes exact at " + std::to_string(width) + "x" +
                std::to_string(height) + (key ? " key" : " inter"));
    }
  }

  M2MfhConfig config;
  config.max_frame_width = 64;
  config.max_frame_height = 64;
  config.base_q_idx = 90;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.num_total_refs = 1;
  config.disable_cdf_update = true;
  std::string error;

  // MFH reference saves the two apply_deblocking_filter bits and costs the
  // wider cur_mfh_id uvlc: net 1 bit for the default header.
  {
    MultiFrameHeader mfh;
    mfh.mfh_id = 1;
    mfh.deblocking_filter_update = true;
    mfh.apply_deblocking_filter = {false, false, false, false};
    M2MfhFrameHeader header;
    header.keyframe = true;
    header.frame_is_inter = false;
    BitWriter baseline;
    write_m2_frame_header_with_mfh(baseline, config, header, nullptr, &error);
    BitWriter with_mfh;
    write_m2_frame_header_with_mfh(with_mfh, config, header, &mfh, &error);
    check(baseline.bit_size() == with_mfh.bit_size() + 1,
          "adapter MFH: 1 bit saved in the frame header");
  }

  // Inter-frame frame_size_override is deliberately rejected.
  {
    M2MfhFrameHeader header;
    header.keyframe = false;
    header.frame_is_inter = true;
    header.frame_size_override = true;
    header.frame_width = 64;
    header.frame_height = 64;
    BitWriter writer;
    check(write_m2_tile_group_header_with_mfh(writer, config, header, nullptr,
                                              &error) == 0 &&
              !error.empty(),
          "adapter: inter frame_size_override rejected");
  }
}

void test_field_source_strings() {
  check(std::string(to_string(FieldSource::kAbsent)) == "absent", "source name");
  check(std::string(to_string(FieldSource::kSequence)) == "sequence",
        "source name");
  check(std::string(to_string(FieldSource::kMfh)) == "mfh", "source name");
  check(std::string(to_string(FieldSource::kExplicit)) == "explicit",
        "source name");
  check(std::string(to_string(FieldSource::kForced)) == "forced",
        "source name");
}

}  // namespace

int main() {
  test_pinned_goldens();
  test_obu_extension_flag_position();
  test_round_trip();
  test_scan_and_store();
  test_validation_errors();
  test_size_precedence();
  test_deblocking_precedence();
  test_segmentation_precedence();
  test_layer_dependencies();
  test_m2_adapter_bit_exact();
  test_field_source_strings();

  std::printf("mfh_test: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
