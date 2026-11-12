// BRU unit tests: pinned goldens, writer/reader round-trip, exhaustive
// bru_region_valid() (R10) verification against an independent oracle, and
// the deblock / IBC buffer gating helpers.
#include "codec/inter/bru/bru.h"

#include "pipeline/video/video_headers.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace {

using av2enc::inter::bru::BruMode;
using av2enc::inter::bru::FrameConfig;
using av2enc::inter::bru::ParsedStream;
using av2enc::inter::bru::SequenceConfig;
using av2enc::inter::bru::TilePlan;

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

std::uint64_t fnv1a64(std::span<const std::uint8_t> bytes) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const std::uint8_t byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string to_hex(std::span<const std::uint8_t> bytes) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (const std::uint8_t byte : bytes) {
    out.push_back(kDigits[byte >> 4U]);
    out.push_back(kDigits[byte & 0x0FU]);
  }
  return out;
}

// Independent, deliberately simple transcription of §7.2 bru_region_valid().
// It uses a dense ownership grid instead of the Rect list in bru_region.cpp.
bool reference_region_valid(int cols, int rows,
                            std::span<const BruMode> modes) {
  struct R {
    int l, r, t, b;
    bool live;
  };
  std::vector<R> rs;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      if (modes[static_cast<std::size_t>(y) * cols + x] ==
          BruMode::kActive) {
        rs.push_back(R{x - 1, x + 1, y - 1, y + 1, true});
      }
    }
  }
  bool again = true;
  while (again) {
    again = false;
    for (std::size_t i = 0; i < rs.size(); ++i) {
      if (!rs[i].live) continue;
      for (std::size_t j = i + 1; j < rs.size(); ++j) {
        if (!rs[j].live) continue;
        const bool apart = rs[i].r < rs[j].l || rs[j].r < rs[i].l ||
                           rs[i].b < rs[j].t || rs[j].b < rs[i].t;
        if (apart) continue;
        rs[i].l = std::min(rs[i].l, rs[j].l);
        rs[i].r = std::max(rs[i].r, rs[j].r);
        rs[i].t = std::min(rs[i].t, rs[j].t);
        rs[i].b = std::max(rs[i].b, rs[j].b);
        rs[j].live = false;
        again = true;
      }
    }
  }
  for (const R& q : rs) {
    if (!q.live) continue;
    for (int y = q.t; y <= q.b; ++y) {
      for (int x = q.l; x <= q.r; ++x) {
        if (x < 0 || x >= cols || y < 0 || y >= rows) continue;
        const BruMode m = modes[static_cast<std::size_t>(y) * cols + x];
        if (m == BruMode::kInactive) return false;
        const bool border = (x == q.l || x == q.r || y == q.t || y == q.b);
        if (border && m != BruMode::kSupport) return false;
      }
    }
  }
  return true;
}

SequenceConfig standard_sequence(bool enable_bru) {
  SequenceConfig c;
  c.max_frame_width = 256;
  c.max_frame_height = 128;
  c.enable_bru = enable_bru;
  c.order_hint_bits = 4;
  c.num_ref_frames = 2;
  c.base_q_idx = 90;
  return c;
}

FrameConfig key_frame() {
  FrameConfig f;
  f.keyframe = true;
  f.order_hint = 0;
  f.refresh_frame_flags = 3;
  f.tile_cols_log2 = 0;
  f.tile_rows_log2 = 0;
  return f;
}

FrameConfig inter_frame() {
  FrameConfig f = key_frame();
  f.keyframe = false;
  f.order_hint = 1;
  f.refresh_frame_flags = 2;
  f.num_total_refs = 1;
  f.ref_frame_idx[0] = 0;
  f.use_bru = false;
  return f;
}

FrameConfig bru_frame() {
  FrameConfig f = inter_frame();
  f.order_hint = 2;
  f.refresh_frame_flags = 1;  // bru_ref slot 0
  f.num_total_refs = 2;
  f.ref_frame_idx[0] = 0;
  f.ref_frame_idx[1] = 1;
  f.use_bru = true;
  f.bru_ref = 0;
  f.bru_inactive = false;
  f.tile_cols_log2 = 1;
  f.tile_rows_log2 = 0;
  f.tile_size_bytes_minus_1 = 0;
  return f;
}

TilePlan mixed_plan() {
  TilePlan plan;
  plan.sb_modes.assign(8, BruMode::kInactive);
  plan.sb_modes[0] = BruMode::kActive;
  plan.sb_modes[1] = BruMode::kSupport;
  plan.sb_modes[4] = BruMode::kSupport;
  plan.sb_modes[5] = BruMode::kSupport;
  plan.bru_tile_active = {1, 0};
  return plan;
}

struct Built {
  std::vector<std::uint8_t> sequence;
  std::vector<std::uint8_t> key;
  std::vector<std::uint8_t> inter;
  std::vector<std::uint8_t> bru;
  std::vector<std::uint8_t> tile_active;
  std::vector<std::uint8_t> tile_inactive;
  std::vector<std::uint8_t> all;
};

Built build_stream(bool enable_bru) {
  Built out;
  const SequenceConfig seq = standard_sequence(enable_bru);
  out.sequence = av2enc::inter::bru::make_sequence_header_obu(seq);

  TilePlan key_plan;
  key_plan.bru_tile_active = {1};
  key_plan.sb_modes.assign(8, BruMode::kActive);
  std::string error;
  std::vector<std::vector<std::uint8_t>> key_tiles{{0x00, 0x40}};
  out.key = av2enc::inter::bru::make_tile_group_obu(
      seq, key_frame(), key_plan, key_tiles, error);

  std::vector<std::vector<std::uint8_t>> inter_tiles{{0x00, 0x40}};
  out.inter = av2enc::inter::bru::make_tile_group_obu(
      seq, inter_frame(), key_plan, inter_tiles, error);

  const TilePlan plan = mixed_plan();
  const std::vector<BruMode> tile0 = {
      plan.sb_modes[0], plan.sb_modes[1], plan.sb_modes[4], plan.sb_modes[5]};
  if (!av2enc::inter::bru::encode_tile_payload(tile0, 2, 2, 0, true,
                                                out.tile_active, error)) {
    check(false, "encode active tile: " + error);
  }
  if (!av2enc::inter::bru::encode_tile_payload({}, 2, 2, 0, false,
                                                out.tile_inactive, error)) {
    check(false, "encode inactive tile: " + error);
  }
  std::vector<std::vector<std::uint8_t>> bru_tiles{out.tile_active,
                                                    out.tile_inactive};
  out.bru = av2enc::inter::bru::make_tile_group_obu(seq, bru_frame(), plan,
                                                    bru_tiles, error);
  check(!out.bru.empty(), "mixed BRU OBU: " + error);

  out.all = out.sequence;
  out.all.insert(out.all.end(), out.key.begin(), out.key.end());
  out.all.insert(out.all.end(), out.inter.begin(), out.inter.end());
  out.all.insert(out.all.end(), out.bru.begin(), out.bru.end());
  return out;
}

void test_reader_round_trip() {
  const Built built = build_stream(true);
  ParsedStream parsed;
  std::string error;
  check(av2enc::inter::bru::parse_stream(built.all, parsed, error),
        "reader parses writer stream: " + error);
  check(parsed.sequence.enable_bru, "sequence enable_bru = 1");
  check(parsed.frames.size() == 3, "three frame OBUs parsed");
  if (parsed.frames.size() != 3) return;
  check(parsed.frames[0].header.keyframe, "f0 keyframe");
  check(!parsed.frames[1].header.use_bru, "f1 use_bru = 0");
  check(parsed.frames[2].header.use_bru, "f2 use_bru = 1");
  check(parsed.frames[2].header.bru_ref == 0, "f2 bru_ref = 0");
  check(!parsed.frames[2].header.bru_inactive, "f2 bru_inactive = 0");
  check(parsed.frames[2].bru_tile_active ==
            std::vector<std::uint8_t>({1, 0}),
        "f2 bru_tile_active = {1,0}");
  check(parsed.frames[2].sb_modes == mixed_plan().sb_modes,
        "f2 BruModes map reconstructed exactly");
  check(parsed.frames[2].region_valid, "f2 passes R10");
}

void test_whole_frame_inactive() {
  const SequenceConfig seq = standard_sequence(true);
  FrameConfig f = bru_frame();
  f.bru_inactive = true;
  f.tile_cols_log2 = 0;
  f.tile_rows_log2 = 0;
  std::string error;
  TilePlan plan;
  const auto obu = av2enc::inter::bru::make_tile_group_obu(seq, f, plan, {},
                                                           error);
  check(!obu.empty(), "whole-inactive OBU: " + error);
  std::vector<std::uint8_t> bytes =
      av2enc::inter::bru::make_sequence_header_obu(seq);
  bytes.insert(bytes.end(), obu.begin(), obu.end());
  ParsedStream parsed;
  check(av2enc::inter::bru::parse_stream(bytes, parsed, error),
        "reader parses whole-inactive frame: " + error);
  check(parsed.frames.size() == 1, "one whole-inactive frame");
  if (parsed.frames.size() == 1) {
    check(parsed.frames[0].header.bru_inactive, "bru_inactive = 1");
    check(parsed.frames[0].whole_frame_inactive, "whole_frame_inactive flag");
    check(parsed.frames[0].region_valid, "whole-inactive R10 trivially true");
  }
}

void test_enable_bru_byte_identity() {
  SequenceConfig seq = standard_sequence(false);
  av2enc::video::VideoConfig m2;
  m2.max_frame_width = seq.max_frame_width;
  m2.max_frame_height = seq.max_frame_height;
  m2.order_hint_bits = seq.order_hint_bits;
  m2.num_ref_frames = seq.num_ref_frames;
  m2.base_q_idx = seq.base_q_idx;
  const auto ours = av2enc::inter::bru::make_sequence_header_obu(seq);
  const auto baseline = av2enc::video::make_sequence_header_obu(m2);
  check(ours == baseline, "enable_bru=0 sequence header byte identity");

  TilePlan plan;
  plan.bru_tile_active = {1};
  plan.sb_modes.assign(8, BruMode::kActive);
  const std::vector<std::uint8_t> tile{0x00, 0x40};
  std::vector<std::vector<std::uint8_t>> tiles{tile};
  std::string error;
  const auto key = av2enc::inter::bru::make_tile_group_obu(
      seq, key_frame(), plan, tiles, error);
  av2enc::video::VideoFrameHeader video_key;
  video_key.keyframe = true;
  video_key.order_hint = 0;
  video_key.refresh_frame_flags = 3;
  const auto baseline_key =
      av2enc::video::make_frame_obu(m2, video_key, tile);
  check(key == baseline_key, "enable_bru=0 key frame byte identity");

  const auto p = av2enc::inter::bru::make_tile_group_obu(
      seq, inter_frame(), plan, tiles, error);
  av2enc::video::VideoFrameHeader video_p;
  video_p.keyframe = false;
  video_p.order_hint = 1;
  video_p.refresh_frame_flags = 2;
  video_p.ref_slot = 0;
  const auto baseline_p = av2enc::video::make_frame_obu(m2, video_p, tile);
  check(p == baseline_p, "enable_bru=0 P frame byte identity");
}

void test_exhaustive_region_validation() {
  // All 3x3 ternary maps: 19,683 cases, including SUPPORT borders.
  std::vector<BruMode> modes(9);
  for (std::uint32_t code = 0; code < 19683U; ++code) {
    std::uint32_t v = code;
    for (int i = 0; i < 9; ++i) {
      modes[static_cast<std::size_t>(i)] =
          static_cast<BruMode>(v % 3U);
      v /= 3U;
    }
    const bool got = av2enc::inter::bru::bru_region_valid(3, 3, modes);
    const bool want = reference_region_valid(3, 3, modes);
    if (got != want) {
      check(false, "3x3 R10 mismatch at code " + std::to_string(code));
      return;
    }
  }
  check(true, "3x3 ternary R10 exhaustive (19683 maps)");

  // All 4x4 binary maps: 65,536 cases, ACTIVE/INACTIVE only.
  std::vector<BruMode> binary(16);
  for (std::uint32_t code = 0; code < 65536U; ++code) {
    for (int i = 0; i < 16; ++i) {
      binary[static_cast<std::size_t>(i)] =
          ((code >> i) & 1U) != 0U ? BruMode::kActive : BruMode::kInactive;
    }
    const bool got = av2enc::inter::bru::bru_region_valid(4, 4, binary);
    const bool want = reference_region_valid(4, 4, binary);
    if (got != want) {
      check(false, "4x4 R10 mismatch at code " + std::to_string(code));
      return;
    }
  }
  check(true, "4x4 binary R10 exhaustive (65536 maps)");
}

void test_gating_helpers() {
  const std::vector<BruMode> modes = {
      BruMode::kActive, BruMode::kSupport, BruMode::kInactive,
      BruMode::kActive, BruMode::kSupport, BruMode::kInactive};
  // sb_shift 4: mi (0,0)->active, (0,16)->support, (0,32)->inactive.
  check(av2enc::inter::bru::deblock_edge_allowed(
            false, 4, 0, 0, 3, modes),
        "deblock allowed when use_bru = 0");
  check(av2enc::inter::bru::deblock_edge_allowed(true, 4, 0, 0, 3, modes),
        "deblock allowed in BRU_ACTIVE superblock");
  check(!av2enc::inter::bru::deblock_edge_allowed(true, 4, 0, 16, 3, modes),
        "deblock skipped in BRU_SUPPORT superblock");
  check(!av2enc::inter::bru::deblock_edge_allowed(true, 4, 0, 32, 3, modes),
        "deblock skipped in BRU_INACTIVE superblock");

  check(av2enc::inter::bru::bru_ibc_buffer_valid(4, 0, 0, 3, modes),
        "IBC buffer valid in BRU_ACTIVE superblock");
  check(av2enc::inter::bru::bru_ibc_buffer_valid(4, 0, 16, 3, modes),
        "IBC buffer valid in BRU_SUPPORT superblock");
  check(!av2enc::inter::bru::bru_ibc_buffer_valid(4, 0, 32, 3, modes),
        "IBC buffer invalidated by BRU_INACTIVE superblock");
}

// Pinned goldens.  These are intentionally byte-level: any change to sequence
// header field order, BRU frame syntax or tile symbol order must update them.
constexpr std::uint64_t kGoldenSequenceNoBru = 0x93a5d37286d14c7dULL;
constexpr std::uint64_t kGoldenSequenceBru = 0x50d119d17f09be5dULL;
constexpr std::uint64_t kGoldenMixedObu = 0x97fdacb3697350ebULL;
constexpr std::uint64_t kGoldenTileActive = 0x68d947717219fb37ULL;
constexpr std::uint64_t kGoldenTileInactive = 0x98b61300c3d7247bULL;

void test_pinned_goldens(bool print) {
  const Built no_bru = build_stream(false);
  const Built bru = build_stream(true);
  const std::uint64_t seq_no = fnv1a64(no_bru.sequence);
  const std::uint64_t seq_yes = fnv1a64(bru.sequence);
  const std::uint64_t mixed = fnv1a64(bru.bru);
  const std::uint64_t active = fnv1a64(bru.tile_active);
  const std::uint64_t inactive = fnv1a64(bru.tile_inactive);
  if (print) {
    std::printf("kGoldenSequenceNoBru = 0x%016llxULL\n",
                static_cast<unsigned long long>(seq_no));
    std::printf("kGoldenSequenceBru   = 0x%016llxULL\n",
                static_cast<unsigned long long>(seq_yes));
    std::printf("kGoldenMixedObu      = 0x%016llxULL\n",
                static_cast<unsigned long long>(mixed));
    std::printf("kGoldenTileActive    = 0x%016llxULL\n",
                static_cast<unsigned long long>(active));
    std::printf("kGoldenTileInactive  = 0x%016llxULL\n",
                static_cast<unsigned long long>(inactive));
    std::printf("mixed OBU = %s\n", to_hex(bru.bru).c_str());
    std::printf("tile active = %s\n", to_hex(bru.tile_active).c_str());
    std::printf("tile inactive = %s\n", to_hex(bru.tile_inactive).c_str());
    return;
  }
  check(seq_no == kGoldenSequenceNoBru, "pinned enable_bru=0 sequence header");
  check(seq_yes == kGoldenSequenceBru, "pinned enable_bru=1 sequence header");
  check(mixed == kGoldenMixedObu, "pinned mixed BRU tile-group OBU");
  check(active == kGoldenTileActive, "pinned active tile payload");
  check(inactive == kGoldenTileInactive, "pinned inactive tile payload");
}

}  // namespace

int main(int argc, char** argv) {
  const bool print = argc > 1 && std::strcmp(argv[1], "--print-goldens") == 0;
  test_pinned_goldens(print);
  if (print) return 0;
  test_reader_round_trip();
  test_whole_frame_inactive();
  test_enable_bru_byte_identity();
  test_exhaustive_region_validation();
  test_gating_helpers();
  std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
