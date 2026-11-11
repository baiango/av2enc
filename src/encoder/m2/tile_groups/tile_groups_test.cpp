// Standalone tests for the M2 multi-tile-group framing component.
//
// No CUDA, no decoder, no external dependencies. Pinned goldens live in
// `goldens/` and are compared byte for byte; pass `--emit-goldens <dir>` to
// regenerate them after an intentional change.
#include "encoder/m2/tile_groups/tile_groups.h"

#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using av2enc::m2::tile_groups::FrameHeaderBits;
using av2enc::m2::tile_groups::FrameTileGroupPlan;
using av2enc::m2::tile_groups::ParsedTileGroup;
using av2enc::m2::tile_groups::TileGroupGeometry;
using av2enc::m2::tile_groups::TileGroupSequence;
using av2enc::m2::tile_groups::TileGroupSpan;
using av2enc::m2::tile_groups::TileGroupWriteParams;

int g_failures = 0;

void check(bool condition, const char* expression, const char* file, int line) {
  if (!condition) {
    ++g_failures;
    std::cerr << file << ':' << line << ": CHECK failed: " << expression << '\n';
  }
}

#define CHECK(expression) \
  ::check((expression), #expression, __FILE__, __LINE__)

TileGroupGeometry make_geometry(int cols_log2, int rows_log2,
                                int tile_size_bytes, bool use_bru = false,
                                bool bru_inactive = false) {
  TileGroupGeometry geometry;
  geometry.num_tiles = 1 << (cols_log2 + rows_log2);
  geometry.tile_cols_log2 = cols_log2;
  geometry.tile_rows_log2 = rows_log2;
  geometry.tile_size_bytes = tile_size_bytes;
  geometry.use_bru = use_bru;
  geometry.bru_inactive = bru_inactive;
  return geometry;
}

FrameHeaderBits make_header(std::size_t bit_count, std::uint32_t seed) {
  FrameHeaderBits bits;
  bits.bytes.assign((bit_count + 7U) / 8U, 0);
  bits.bit_count = bit_count;
  for (std::size_t i = 0; i < bit_count; ++i) {
    const std::uint32_t mixed =
        seed * 1103515245U + static_cast<std::uint32_t>(i) * 12345U + 0x9E3779B9U;
    if (((mixed >> 15U) & 1U) != 0U) {
      bits.bytes[i >> 3U] |=
          static_cast<std::uint8_t>(1U << (7U - (i & 7U)));
    }
  }
  return bits;
}

std::vector<std::vector<std::uint8_t>> make_payloads(int count,
                                                     int size_cycle) {
  std::vector<std::vector<std::uint8_t>> payloads;
  payloads.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    const std::size_t size =
        static_cast<std::size_t>(2 + ((i * 7 + 3) % size_cycle));
    std::vector<std::uint8_t> payload(size);
    for (std::size_t j = 0; j < size; ++j) {
      payload[j] = static_cast<std::uint8_t>((i * 41 + static_cast<int>(j) * 7) & 0xFF);
    }
    payloads.push_back(std::move(payload));
  }
  return payloads;
}

std::string to_hex(const std::vector<std::uint8_t>& bytes) {
  static const char* kDigits = "0123456789abcdef";
  std::string text;
  text.reserve(bytes.size() * 2U);
  for (std::uint8_t byte : bytes) {
    text.push_back(kDigits[byte >> 4U]);
    text.push_back(kDigits[byte & 0x0FU]);
  }
  return text;
}

bool from_hex(const std::string& text, std::vector<std::uint8_t>* bytes) {
  if ((text.size() & 1U) != 0U) return false;
  bytes->clear();
  bytes->reserve(text.size() / 2U);
  const auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  for (std::size_t i = 0; i < text.size(); i += 2U) {
    const int hi = nibble(text[i]);
    const int lo = nibble(text[i + 1U]);
    if (hi < 0 || lo < 0) return false;
    bytes->push_back(static_cast<std::uint8_t>((hi << 4) | lo));
  }
  return true;
}

bool write_hex_file(const std::string& path,
                    const std::vector<std::vector<std::uint8_t>>& groups) {
  std::ofstream out(path);
  if (!out) return false;
  out << "# Pinned M2 multi-tile-group framing golden. One line per group,\n";
  out << "# lowercase hex of the complete OBU payload (OBU header excluded).\n";
  for (const auto& group : groups) out << to_hex(group) << '\n';
  return static_cast<bool>(out);
}

bool read_hex_file(const std::string& path,
                   std::vector<std::vector<std::uint8_t>>* groups) {
  std::ifstream in(path);
  if (!in) return false;
  groups->clear();
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::uint8_t> bytes;
    if (!from_hex(line, &bytes)) return false;
    groups->push_back(std::move(bytes));
  }
  return !groups->empty();
}

void flip_bit(std::vector<std::uint8_t>* bytes, std::size_t bit_index) {
  (*bytes)[bit_index >> 3U] ^=
      static_cast<std::uint8_t>(1U << (7U - (bit_index & 7U)));
}

// Builds a payload per params without the whole-frame §6.18 validation. Used
// only to craft invalid sequences for the reader's rejection tests.
bool build_raw(const TileGroupGeometry& geometry,
               const FrameHeaderBits& header,
               const std::vector<TileGroupWriteParams>& params,
               const std::vector<std::vector<std::uint8_t>>& payloads,
               std::vector<std::vector<std::uint8_t>>* out,
               std::string* error) {
  out->clear();
  for (const auto& one : params) {
    std::vector<std::uint8_t> payload;
    if (!av2enc::m2::tile_groups::write_tile_group(geometry, header, one,
                                                   payloads, &payload, error)) {
      return false;
    }
    out->push_back(std::move(payload));
  }
  return true;
}

TileGroupWriteParams first_group(int start, int end) {
  TileGroupWriteParams params;
  params.is_first_tile_group = true;
  params.span.tg_start = start;
  params.span.tg_end = end;
  return params;
}

TileGroupWriteParams later_group(int start, int end, bool present) {
  TileGroupWriteParams params;
  params.is_first_tile_group = false;
  params.frame_header_present = present;
  params.span.tg_start = start;
  params.span.tg_end = end;
  return params;
}

struct GoldenCase {
  const char* name;
  TileGroupGeometry geometry;
  FrameHeaderBits header;
  bool copy_header;
  bool bru_default;
  std::vector<int> group_sizes;
  std::vector<std::vector<std::uint8_t>> payloads;
  std::vector<TileGroupSpan> groups;
};

GoldenCase make_golden_two_group() {
  GoldenCase c;
  c.name = "tile_groups_2group.hex";
  c.geometry = make_geometry(1, 1, 2, /*use_bru=*/true);
  c.header = make_header(20, 0xC0FFEEU);
  c.copy_header = false;
  c.bru_default = false;
  c.group_sizes = {2, 2};
  c.payloads = make_payloads(4, 13);
  // Give the two-byte size fields something to encode.
  c.payloads[1].resize(300, 0x5A);
  return c;
}

GoldenCase make_golden_three_group() {
  GoldenCase c;
  c.name = "tile_groups_3group.hex";
  c.geometry = make_geometry(1, 1, 1);
  c.header = make_header(13, 0x1234U);
  c.copy_header = true;
  c.bru_default = false;
  c.group_sizes = {1, 2, 1};
  c.payloads = make_payloads(4, 9);
  return c;
}

bool build_golden_case(GoldenCase* c, std::vector<std::vector<std::uint8_t>>* out,
                       FrameTileGroupPlan* plan_out, std::string* error) {
  std::vector<TileGroupSpan> groups;
  if (!av2enc::m2::tile_groups::plan_contiguous_tile_groups(
          c->geometry, c->group_sizes, c->copy_header, c->bru_default, &groups,
          error)) {
    return false;
  }
  // The 2-group golden exercises a mixed per-tile BRU activity map.
  if (c->geometry.use_bru && !groups.empty()) {
    groups[0].bru_tile_active = {1, 0};
    groups[1].bru_tile_active = {0, 1};
  }
  FrameTileGroupPlan plan;
  plan.geometry = c->geometry;
  plan.frame_header = c->header;
  plan.copy_frame_header = c->copy_header;
  plan.groups = groups;
  if (!av2enc::m2::tile_groups::write_frame_tile_groups(plan, c->payloads, out,
                                                        error)) {
    return false;
  }
  c->groups = std::move(groups);
  if (plan_out != nullptr) *plan_out = plan;
  return true;
}

void test_geometry_validation() {
  std::string error;
  {
    const auto geometry = make_geometry(1, 1, 1);
    CHECK(av2enc::m2::tile_groups::validate_tile_group_geometry(geometry,
                                                                &error));
  }
  {
    auto geometry = make_geometry(1, 1, 1);
    geometry.tile_rows_log2 = 0;  // 2 tiles, but tile_bits still 1
    geometry.num_tiles = 3;       // not a power of two
    CHECK(!av2enc::m2::tile_groups::validate_tile_group_geometry(geometry,
                                                                 &error));
  }
  {
    auto geometry = make_geometry(1, 1, 1);
    geometry.tile_size_bytes = 0;
    CHECK(!av2enc::m2::tile_groups::validate_tile_group_geometry(geometry,
                                                                 &error));
  }
  {
    auto geometry = make_geometry(1, 1, 5);
    CHECK(!av2enc::m2::tile_groups::validate_tile_group_geometry(geometry,
                                                                 &error));
  }
  {
    auto geometry = make_geometry(0, 0, 1);
    geometry.tile_size_bytes = 1;  // TileSizeBytes is not coded for one tile
    CHECK(!av2enc::m2::tile_groups::validate_tile_group_geometry(geometry,
                                                                 &error));
  }
  {
    auto geometry = make_geometry(1, 1, 1, /*use_bru=*/false,
                                  /*bru_inactive=*/true);
    CHECK(!av2enc::m2::tile_groups::validate_tile_group_geometry(geometry,
                                                                 &error));
  }
}

void test_single_group_round_trip() {
  const auto geometry = make_geometry(1, 1, 1);
  const auto header = make_header(17, 7);
  const auto payloads = make_payloads(4, 7);

  std::vector<TileGroupSpan> groups;
  std::string error;
  CHECK(av2enc::m2::tile_groups::plan_contiguous_tile_groups(
      geometry, std::vector<int>{4}, false, false, &groups, &error));

  FrameTileGroupPlan plan;
  plan.geometry = geometry;
  plan.frame_header = header;
  plan.groups = groups;

  std::vector<std::vector<std::uint8_t>> written;
  CHECK(av2enc::m2::tile_groups::write_frame_tile_groups(plan, payloads,
                                                         &written, &error));
  CHECK(written.size() == 1);

  TileGroupSequence parsed;
  CHECK(av2enc::m2::tile_groups::read_tile_group_sequence(
      written, geometry, header.bit_count, &parsed, &error));
  CHECK(parsed.groups.size() == 1);
  CHECK(parsed.groups[0].is_first_tile_group);
  CHECK(parsed.groups[0].frame_header_present);
  CHECK(!parsed.groups[0].tile_start_and_end_present);
  CHECK(parsed.groups[0].tg_start == 0);
  CHECK(parsed.groups[0].tg_end == 3);
  CHECK(parsed.groups[0].frame_header.bytes == header.bytes);
  CHECK(parsed.groups[0].tile_bytes.size() == 4);
  for (int i = 0; i < 4; ++i) {
    const std::size_t offset =
        parsed.groups[0].tile_offsets[static_cast<std::size_t>(i)];
    const std::size_t size =
        parsed.groups[0].tile_bytes[static_cast<std::size_t>(i)];
    const std::vector<std::uint8_t> slice(
        written[0].begin() + static_cast<std::ptrdiff_t>(offset),
        written[0].begin() + static_cast<std::ptrdiff_t>(offset + size));
    CHECK(slice == payloads[static_cast<std::size_t>(i)]);
  }
}

void test_two_group_shared_header() {
  const auto geometry = make_geometry(1, 1, 1);
  const auto header = make_header(21, 99);
  const auto payloads = make_payloads(4, 11);

  std::vector<TileGroupSpan> groups;
  std::string error;
  CHECK(av2enc::m2::tile_groups::plan_contiguous_tile_groups(
      geometry, std::vector<int>{2, 2}, false, false, &groups, &error));

  FrameTileGroupPlan plan;
  plan.geometry = geometry;
  plan.frame_header = header;
  plan.groups = groups;

  std::vector<std::vector<std::uint8_t>> written;
  CHECK(av2enc::m2::tile_groups::write_frame_tile_groups(plan, payloads,
                                                         &written, &error));
  CHECK(written.size() == 2);

  TileGroupSequence parsed;
  CHECK(av2enc::m2::tile_groups::read_tile_group_sequence(
      written, geometry, header.bit_count, &parsed, &error));
  CHECK(parsed.groups.size() == 2);
  CHECK(parsed.groups_with_shared_header == 1);
  CHECK(parsed.groups_with_header_copy == 0);
  CHECK(parsed.groups[1].tg_start == 2);
  CHECK(parsed.groups[1].tg_end == 3);
  CHECK(!parsed.groups[1].frame_header_present);
  CHECK(parsed.groups[1].frame_header.empty());
}

void test_three_group_header_copy() {
  const auto geometry = make_geometry(1, 1, 1);
  const auto header = make_header(13, 0xBEEFU);
  const auto payloads = make_payloads(4, 6);

  std::vector<TileGroupSpan> groups;
  std::string error;
  CHECK(av2enc::m2::tile_groups::plan_contiguous_tile_groups(
      geometry, std::vector<int>{1, 2, 1}, true, false, &groups, &error));

  FrameTileGroupPlan plan;
  plan.geometry = geometry;
  plan.frame_header = header;
  plan.copy_frame_header = true;
  plan.groups = groups;

  std::vector<std::vector<std::uint8_t>> written;
  CHECK(av2enc::m2::tile_groups::write_frame_tile_groups(plan, payloads,
                                                         &written, &error));
  CHECK(written.size() == 3);

  TileGroupSequence parsed;
  CHECK(av2enc::m2::tile_groups::read_tile_group_sequence(
      written, geometry, header.bit_count, &parsed, &error));
  CHECK(parsed.groups_with_shared_header == 0);
  CHECK(parsed.groups_with_header_copy == 2);
  for (std::size_t i = 1; i < parsed.groups.size(); ++i) {
    CHECK(parsed.groups[i].frame_header_present);
    CHECK(parsed.groups[i].frame_header_is_copy);
    CHECK(parsed.groups[i].frame_header.bytes == header.bytes);
  }
  // §6.17.1 NOTE: the copy is not byte-aligned like the original. The first
  // group's header starts after the is_first_tile_group bit (offset 1); a
  // later group's copy starts after is_first_tile_group +
  // frame_header_present_flag (offset 2). Every group still ends its header
  // on a byte boundary.
  CHECK(parsed.groups[0].header_bits % 8U == 0U);
  CHECK(parsed.groups[1].header_bits % 8U == 0U);
}

void test_bru_tile_active() {
  const auto geometry = make_geometry(1, 1, 1, /*use_bru=*/true);
  const auto header = make_header(15, 3);
  const auto payloads = make_payloads(4, 5);

  std::vector<TileGroupSpan> groups;
  std::string error;
  CHECK(av2enc::m2::tile_groups::plan_contiguous_tile_groups(
      geometry, std::vector<int>{2, 2}, false, false, &groups, &error));
  groups[0].bru_tile_active = {1, 0};
  groups[1].bru_tile_active = {0, 1};

  FrameTileGroupPlan plan;
  plan.geometry = geometry;
  plan.frame_header = header;
  plan.groups = groups;

  std::vector<std::vector<std::uint8_t>> written;
  CHECK(av2enc::m2::tile_groups::write_frame_tile_groups(plan, payloads,
                                                         &written, &error));

  TileGroupSequence parsed;
  CHECK(av2enc::m2::tile_groups::read_tile_group_sequence(
      written, geometry, header.bit_count, &parsed, &error));
  CHECK(parsed.groups[0].bru_tile_active == std::vector<std::uint8_t>({1, 0}));
  CHECK(parsed.groups[1].bru_tile_active == std::vector<std::uint8_t>({0, 1}));

  // NumTiles == 1 infers BruTileActives[0][0] = 1 and codes no bits.
  const auto single_geometry = make_geometry(0, 0, 0, /*use_bru=*/true);
  const auto single_payloads = make_payloads(1, 5);
  std::vector<TileGroupSpan> single_groups;
  CHECK(av2enc::m2::tile_groups::plan_contiguous_tile_groups(
      single_geometry, std::vector<int>{1}, false, false, &single_groups,
      &error));
  FrameTileGroupPlan single_plan;
  single_plan.geometry = single_geometry;
  single_plan.frame_header = make_header(9, 1);
  single_plan.groups = single_groups;
  std::vector<std::vector<std::uint8_t>> single_written;
  CHECK(av2enc::m2::tile_groups::write_frame_tile_groups(
      single_plan, single_payloads, &single_written, &error));
  TileGroupSequence single_parsed;
  CHECK(av2enc::m2::tile_groups::read_tile_group_sequence(
      single_written, single_geometry, single_plan.frame_header.bit_count,
      &single_parsed, &error));
  CHECK(single_parsed.groups[0].bru_tile_active.empty());
}

void test_bru_inactive() {
  const auto geometry =
      make_geometry(1, 1, 1, /*use_bru=*/true, /*bru_inactive=*/true);
  const auto header = make_header(19, 0xABCDU);
  const auto payloads = make_payloads(4, 5);

  std::vector<TileGroupSpan> groups;
  std::string error;
  CHECK(av2enc::m2::tile_groups::plan_contiguous_tile_groups(
      geometry, std::vector<int>{4}, false, false, &groups, &error));

  FrameTileGroupPlan plan;
  plan.geometry = geometry;
  plan.frame_header = header;
  plan.groups = groups;

  std::vector<std::vector<std::uint8_t>> written;
  CHECK(av2enc::m2::tile_groups::write_frame_tile_groups(plan, payloads,
                                                         &written, &error));
  CHECK(written.size() == 1);
  TileGroupSequence parsed;
  CHECK(av2enc::m2::tile_groups::read_tile_group_sequence(
      written, geometry, header.bit_count, &parsed, &error));
  CHECK(parsed.bru_inactive);
  CHECK(parsed.groups[0].bru_inactive);
  CHECK(parsed.groups[0].tile_bytes.empty());

  // A stray one in trailing_bits() is a framing error.
  std::vector<std::uint8_t> corrupt = written[0];
  flip_bit(&corrupt, header.bit_count + 1U + 1U);
  CHECK(!av2enc::m2::tile_groups::read_tile_group_sequence(
      std::vector<std::vector<std::uint8_t>>{corrupt}, geometry,
      header.bit_count, &parsed, &error));
}

void test_writer_rejections() {
  const auto geometry = make_geometry(1, 1, 1);
  const auto header = make_header(11, 5);
  const auto payloads = make_payloads(4, 5);
  std::string error;

  {
    FrameTileGroupPlan plan;
    plan.geometry = geometry;
    plan.frame_header = header;
    TileGroupSpan a;
    a.tg_start = 0;
    a.tg_end = 1;
    TileGroupSpan b;
    b.tg_start = 3;
    b.tg_end = 3;
    plan.groups = {a, b};
    CHECK(!av2enc::m2::tile_groups::validate_frame_tile_groups(plan, &error));
  }
  {
    FrameTileGroupPlan plan;
    plan.geometry = geometry;
    plan.frame_header = header;
    TileGroupSpan a;
    a.tg_start = 0;
    a.tg_end = 3;
    plan.groups = {a};
    plan.copy_frame_header = true;  // no later group, harmless
    CHECK(av2enc::m2::tile_groups::validate_frame_tile_groups(plan, &error));
  }
  {
    FrameTileGroupPlan plan;
    plan.geometry = make_geometry(1, 1, 1, /*use_bru=*/true);
    plan.frame_header = header;
    TileGroupSpan a;
    a.tg_start = 0;
    a.tg_end = 1;  // missing two bru_tile_active entries
    TileGroupSpan b;
    b.tg_start = 2;
    b.tg_end = 3;
    plan.groups = {a, b};
    CHECK(!av2enc::m2::tile_groups::validate_frame_tile_groups(plan, &error));
  }
  {
    FrameTileGroupPlan plan;
    plan.geometry = geometry;
    plan.frame_header = header;
    TileGroupSpan a;
    a.tg_start = 0;
    a.tg_end = 1;
    TileGroupSpan b;
    b.tg_start = 2;
    b.tg_end = 3;
    b.frame_header_present = true;  // plan says shared, group says copy
    plan.copy_frame_header = false;
    plan.groups = {a, b};
    CHECK(!av2enc::m2::tile_groups::validate_frame_tile_groups(plan, &error));
  }
}

// The reader must reject every non-conforming §6.18 layout. Each case starts
// from a valid 2-group stream and changes only the framing.
void test_reader_rejections() {
  const auto geometry = make_geometry(1, 1, 1);
  const auto header = make_header(19, 0x55AAU);
  const auto payloads = make_payloads(4, 7);
  std::string error;

  const auto expect_bad = [&](const std::vector<std::vector<std::uint8_t>>& seq,
                              const char* what) {
    TileGroupSequence parsed;
    std::string local;
    const bool ok = av2enc::m2::tile_groups::read_tile_group_sequence(
        seq, geometry, header.bit_count, &parsed, &local);
    if (ok) {
      ++g_failures;
      std::cerr << "CHECK failed: reader accepted " << what << '\n';
    }
  };

  {
    std::vector<std::vector<std::uint8_t>> seq;
    CHECK(build_raw(geometry, header,
                    {first_group(0, 1), later_group(3, 3, false)}, payloads,
                    &seq, &error));
    expect_bad(seq, "a tile gap");
  }
  {
    std::vector<std::vector<std::uint8_t>> seq;
    CHECK(build_raw(geometry, header,
                    {first_group(2, 3), later_group(0, 1, false)}, payloads,
                    &seq, &error));
    expect_bad(seq, "a reordered frame (first group does not start at 0)");
  }
  {
    std::vector<std::vector<std::uint8_t>> seq;
    CHECK(build_raw(geometry, header,
                    {first_group(0, 2), later_group(1, 3, false)}, payloads,
                    &seq, &error));
    expect_bad(seq, "overlapping tile groups");
  }
  {
    std::vector<std::vector<std::uint8_t>> seq;
    CHECK(build_raw(geometry, header, {first_group(0, 1)}, payloads, &seq,
                    &error));
    expect_bad(seq, "a missing final tile group");
  }
  {
    std::vector<std::vector<std::uint8_t>> seq;
    TileGroupWriteParams second = later_group(2, 3, true);
    second.is_first_tile_group = true;  // duplicate first flag
    CHECK(build_raw(geometry, header, {first_group(0, 1), second}, payloads,
                    &seq, &error));
    expect_bad(seq, "a duplicate is_first_tile_group bit");
  }
  {
    std::vector<std::vector<std::uint8_t>> seq;
    CHECK(build_raw(geometry, header,
                    {first_group(0, 1), later_group(2, 3, true)}, payloads,
                    &seq, &error));
    // Bit 0 is is_first_tile_group, bit 1 is frame_header_present_flag, so the
    // copied header starts at bit 2.
    flip_bit(&seq[1], 2U + 3U);
    expect_bad(seq, "a frame_header_copy that differs from the first header");
  }
  {
    std::vector<std::vector<std::uint8_t>> seq;
    CHECK(build_raw(geometry, header,
                    {first_group(0, 1), later_group(2, 3, false)}, payloads,
                    &seq, &error));
    // group 0: 1 flag + 19 header + 1 present flag + 4 tg bits = 25 bits, so
    // bits 25..31 are byte_alignment() padding and must be zero.
    flip_bit(&seq[0], 25U);
    expect_bad(seq, "non-zero byte_alignment() padding");
  }
  {
    std::vector<std::vector<std::uint8_t>> seq;
    CHECK(build_raw(geometry, header,
                    {first_group(0, 1), later_group(2, 3, false)}, payloads,
                    &seq, &error));
    ParsedTileGroup parsed;
    CHECK(av2enc::m2::tile_groups::read_tile_group(
        seq[0], geometry, header.bit_count, &parsed, &error));
    // Keep the header, the size field and the first tile, then leave one byte
    // for the last tile: below the 2-byte init_symbol() minimum.
    seq[0].resize(parsed.header_bytes + 1U + payloads[0].size() + 1U);
    expect_bad(seq, "a truncated last tile payload");
  }
  {
    std::vector<std::vector<std::uint8_t>> seq;
    CHECK(build_raw(geometry, header,
                    {first_group(0, 1), later_group(2, 3, false)}, payloads,
                    &seq, &error));
    ParsedTileGroup parsed;
    CHECK(av2enc::m2::tile_groups::read_tile_group(
        seq[0], geometry, header.bit_count, &parsed, &error));
    // First size field sits at header_bytes; claim a tile bigger than the
    // remaining OBU bytes.
    seq[0][parsed.header_bytes] = 0xFF;
    expect_bad(seq, "a tile_size_minus_1 past the OBU payload");
  }
  {
    std::vector<std::vector<std::uint8_t>> seq;
    CHECK(build_raw(geometry, header,
                    {first_group(0, 1), later_group(2, 3, false)}, payloads,
                    &seq, &error));
    // Drop the last tile down to one byte (below the 2-byte init_symbol
    // minimum). Group 1 is header + size field + two 5-byte payloads.
    seq[1].resize(seq[1].size() - 4U);
    expect_bad(seq, "a one-byte last tile payload");
  }
}

void test_goldens(const std::string& dir, bool emit) {
  const auto run = [&](GoldenCase c) {
    std::vector<std::vector<std::uint8_t>> written;
    std::string error;
    if (!build_golden_case(&c, &written, nullptr, &error)) {
      ++g_failures;
      std::cerr << "golden build failed for " << c.name << ": " << error << '\n';
      return;
    }
    const std::string path = dir + "/" + c.name;
    if (emit) {
      if (!write_hex_file(path, written)) {
        ++g_failures;
        std::cerr << "could not write golden " << path << '\n';
      } else {
        std::cout << "wrote " << path << '\n';
      }
      return;
    }
    std::vector<std::vector<std::uint8_t>> golden;
    if (!read_hex_file(path, &golden)) {
      ++g_failures;
      std::cerr << "could not read golden " << path << '\n';
      return;
    }
    if (golden.size() != written.size()) {
      ++g_failures;
      std::cerr << "golden " << c.name << ": group count differs\n";
      return;
    }
    for (std::size_t i = 0; i < written.size(); ++i) {
      if (golden[i] != written[i]) {
        ++g_failures;
        std::cerr << "golden " << c.name << " group " << i << " differs\n"
                  << "  expected " << to_hex(golden[i]) << "\n"
                  << "  actual   " << to_hex(written[i]) << "\n";
      }
    }
    // The pinned bytes must still parse into the expected §6.18 structure.
    TileGroupSequence parsed;
    if (!av2enc::m2::tile_groups::read_tile_group_sequence(
            golden, c.geometry, c.header.bit_count, &parsed, &error)) {
      ++g_failures;
      std::cerr << "golden " << c.name << " does not parse: " << error << '\n';
      return;
    }
    if (c.copy_header) {
      if (parsed.groups_with_header_copy !=
          static_cast<int>(c.group_sizes.size()) - 1) {
        ++g_failures;
        std::cerr << "golden " << c.name << ": expected header copies\n";
      }
    } else {
      if (parsed.groups_with_shared_header !=
          static_cast<int>(c.group_sizes.size()) - 1) {
        ++g_failures;
        std::cerr << "golden " << c.name << ": expected shared headers\n";
      }
    }
  };

  run(make_golden_two_group());
  run(make_golden_three_group());
}

std::string default_golden_dir() {
#ifdef AV2ENC_TILE_GROUP_GOLDEN_DIR
  return AV2ENC_TILE_GROUP_GOLDEN_DIR;
#else
  return "goldens";
#endif
}

}  // namespace

int main(int argc, char** argv) {
  std::string golden_dir = default_golden_dir();
  bool emit = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--emit-goldens" && i + 1 < argc) {
      golden_dir = argv[++i];
      emit = true;
    } else if (arg == "--help") {
      std::cout << "usage: tile_groups_test [--emit-goldens DIR]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << '\n';
      return 2;
    }
  }

  test_geometry_validation();
  test_single_group_round_trip();
  test_two_group_shared_header();
  test_three_group_header_copy();
  test_bru_tile_active();
  test_bru_inactive();
  test_writer_rejections();
  test_reader_rejections();
  test_goldens(golden_dir, emit);

  if (g_failures != 0) {
    std::cerr << "\nFAIL: " << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "PASS: tile group framing writer/reader, goldens, and "
               "invalid-sequence rejection\n";
  return 0;
}
