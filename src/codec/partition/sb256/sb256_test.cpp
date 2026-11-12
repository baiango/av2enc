// Tests for the 256x256-superblock reference and the use_256 stream.
//
// The build script also compiles partition.cpp and sb128.cpp, so every table,
// boundary, allowed-set, context and geometry result is cross-checked against
// the independent M1 partition module and the verified 128-SB module.

#include "test_common.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "pipeline/still/basic_cdf_tables.h"
#include "codec/partition/tree/partition.h"
#include "codec/partition/sb128/sb128.h"
#include "codec/partition/sb128/sb128_stream.h"
#include "codec/partition/sb256/sb256.h"
#include "codec/partition/sb256/sb256_stream.h"

namespace {

namespace s2 = av2enc::sb256;
namespace s1 = av2enc::sb128;
namespace p = av2enc::partition;

using av2enc::BitWriter;
using s2::BlockRect;
using s2::BlockSize;
using s2::Boundary;
using s2::Contexts;
using s2::Geometry;
using s2::PartitionType;

constexpr PartitionType kNone = PartitionType::kNone;
constexpr PartitionType kSplit = PartitionType::kSplit;
constexpr PartitionType kHorz = PartitionType::kHorz;
constexpr PartitionType kVert = PartitionType::kVert;

Geometry make_geometry(int width, int height, int sb_size = 256) {
  Geometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = sb_size;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;
  geometry.max_pb_aspect_ratio = 8;
  return geometry;
}

s1::Geometry to_sb128(const Geometry& geometry) {
  s1::Geometry out;
  out.width = geometry.width;
  out.height = geometry.height;
  out.sb_size = geometry.sb_size;
  out.has_chroma = geometry.has_chroma;
  out.chroma_subsampling_x = geometry.chroma_subsampling_x;
  out.chroma_subsampling_y = geometry.chroma_subsampling_y;
  out.max_pb_aspect_ratio = geometry.max_pb_aspect_ratio;
  return out;
}

p::FrameGeometry to_partition(const Geometry& geometry) {
  p::FrameGeometry out;
  out.width = geometry.width;
  out.height = geometry.height;
  out.sb_size = geometry.sb_size;
  out.has_chroma = geometry.has_chroma;
  out.chroma_subsampling_x = geometry.chroma_subsampling_x;
  out.chroma_subsampling_y = geometry.chroma_subsampling_y;
  out.max_pb_aspect_ratio = geometry.max_pb_aspect_ratio;
  return out;
}

std::uint64_t fnv1a(const std::string& text) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const char raw : text) {
    const unsigned char byte = static_cast<unsigned char>(raw);
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

// --- minimal SHA-256 (test-local; no dependency) ---------------------------
std::uint32_t rotr(std::uint32_t value, unsigned amount) {
  return (value >> amount) | (value << (32U - amount));
}

std::string sha256_hex(std::span<const std::uint8_t> data) {
  static constexpr std::uint32_t kK[64] = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
      0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
      0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
      0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
      0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
      0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
      0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
      0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
      0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

  std::vector<std::uint8_t> message(data.begin(), data.end());
  const std::uint64_t bit_length =
      static_cast<std::uint64_t>(message.size()) * 8ULL;
  message.push_back(0x80U);
  while (message.size() % 64 != 56) message.push_back(0U);
  for (int shift = 56; shift >= 0; shift -= 8) {
    message.push_back(
        static_cast<std::uint8_t>((bit_length >> static_cast<unsigned>(shift)) &
                                  0xffU));
  }

  std::uint32_t h[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  for (std::size_t chunk = 0; chunk < message.size(); chunk += 64) {
    std::uint32_t w[64] = {};
    for (int i = 0; i < 16; ++i) {
      const std::size_t at = chunk + static_cast<std::size_t>(i) * 4;
      w[i] = (static_cast<std::uint32_t>(message[at]) << 24U) |
             (static_cast<std::uint32_t>(message[at + 1]) << 16U) |
             (static_cast<std::uint32_t>(message[at + 2]) << 8U) |
             static_cast<std::uint32_t>(message[at + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 =
          rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3U);
      const std::uint32_t s1 =
          rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10U);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0];
    std::uint32_t b = h[1];
    std::uint32_t c = h[2];
    std::uint32_t d = h[3];
    std::uint32_t e = h[4];
    std::uint32_t f = h[5];
    std::uint32_t g = h[6];
    std::uint32_t hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t temp1 = hh + s1 + ch + kK[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }

  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (std::uint32_t word : h) {
    for (int shift = 28; shift >= 0; shift -= 4) {
      out.push_back(kHex[(word >> static_cast<unsigned>(shift)) & 0x0fU]);
    }
  }
  return out;
}

void test_block_tables() {
  CHECK(s2::is_valid_block_size(BlockSize::k256x256));
  CHECK(!s2::is_valid_block_size(BlockSize::kInvalid));
  CHECK(s2::block_width(BlockSize::k256x256) == 256);
  CHECK(s2::block_height(BlockSize::k256x256) == 256);
  CHECK(s2::num_4x4_blocks_wide(BlockSize::k256x256) == 64);
  CHECK(s2::num_4x4_blocks_high(BlockSize::k256x256) == 64);
  CHECK(s2::superblock_size(64) == BlockSize::k64x64);
  CHECK(s2::superblock_size(128) == BlockSize::k128x128);
  CHECK(s2::superblock_size(256) == BlockSize::k256x256);
  CHECK(s2::superblock_size(512) == BlockSize::kInvalid);
  CHECK(std::string(s2::block_size_name(BlockSize::k256x256)) == "256x256");

  CHECK(s2::partition_size_adjust(BlockSize::k256x256) == 9);
  CHECK(s2::partition_size_adjust(BlockSize::k256x128) == 8);
  CHECK(s2::partition_size_adjust(BlockSize::k128x256) == 7);
  CHECK(s2::partition_size_adjust_rect_type(BlockSize::k256x256) == 12);
  CHECK(s2::partition_size_adjust_rect_type(BlockSize::k256x128) == 11);
  CHECK(s2::partition_size_adjust_rect_type(BlockSize::k128x256) == 10);
  CHECK(s2::spec_partition_index(kSplit) == 9);
}

void test_sb_config() {
  std::string error;
  s2::SbConfig config;
  CHECK(s2::validate_sb_config(config, error));
  CHECK(s2::get_seq_sb_size(false, false) == BlockSize::k64x64);
  CHECK(s2::get_seq_sb_size(false, true) == BlockSize::k128x128);
  CHECK(s2::get_seq_sb_size(true, false) == BlockSize::k256x256);

  // Intra clamps the 256 flag back to 128 (§5.18.1).
  config.use_256x256_superblock = true;
  config.frame_is_intra = true;
  CHECK(s2::validate_sb_config(config, error));
  CHECK(s2::frame_sb_size(config) == BlockSize::k128x128);
  CHECK(s2::frame_sb_size_pixels(config) == 128);
  config.frame_is_intra = false;
  CHECK(s2::frame_sb_size(config) == BlockSize::k256x256);
  CHECK(s2::frame_sb_size_pixels(config) == 256);

  // Both flags at once is a configuration error: use_128 is not parsed.
  config.use_128x128_superblock = true;
  CHECK(!s2::validate_sb_config(config, error));

  config = s2::SbConfig{};
  config.use_128x128_superblock = true;
  CHECK(s2::frame_sb_size(config) == BlockSize::k128x128);
  config = s2::SbConfig{};
  CHECK(s2::frame_sb_size(config) == BlockSize::k64x64);
}

void test_partition_subsize() {
  using s2::partition_subsize;
  // The 256 SB split chain: 256 -> 128 -> 64.  Nothing smaller can signal
  // PARTITION_SPLIT.
  CHECK(partition_subsize(kSplit, BlockSize::k256x256) ==
        BlockSize::k128x128);
  CHECK(partition_subsize(kSplit, BlockSize::k128x128) == BlockSize::k64x64);
  CHECK(partition_subsize(kSplit, BlockSize::k64x64) == BlockSize::kInvalid);
  CHECK(partition_subsize(kHorz, BlockSize::k256x256) ==
        BlockSize::k256x128);
  CHECK(partition_subsize(kVert, BlockSize::k256x256) ==
        BlockSize::k128x256);
  // rect_type_implied_by_bsize() forbids the other direction.
  CHECK(partition_subsize(kHorz, BlockSize::k256x128) == BlockSize::kInvalid);
  CHECK(partition_subsize(kVert, BlockSize::k256x128) ==
        BlockSize::k128x128);
  CHECK(partition_subsize(kHorz, BlockSize::k128x256) ==
        BlockSize::k128x128);
  CHECK(partition_subsize(kVert, BlockSize::k128x256) == BlockSize::kInvalid);
  CHECK(partition_subsize(kNone, BlockSize::k256x256) ==
        BlockSize::k256x256);

  const auto h256 = s2::rect_type_implied_by_bsize(BlockSize::k128x256);
  CHECK(h256.implied && h256.partition == kHorz);
  const auto v256 = s2::rect_type_implied_by_bsize(BlockSize::k256x128);
  CHECK(v256.implied && v256.partition == kVert);
  CHECK(!s2::rect_type_implied_by_bsize(BlockSize::k256x256).implied);
}

void test_geometry() {
  struct Case {
    int width;
    int height;
    int sb_size;
    int mi_cols;
    int mi_rows;
    int sb_cols;
    int sb_rows;
  };
  const Case cases[] = {
      {256, 256, 256, 64, 64, 1, 1},
      {640, 360, 256, 160, 90, 3, 2},
      {1920, 1080, 256, 480, 270, 8, 5},
      {520, 300, 256, 130, 76, 3, 2},
      {640, 360, 128, 160, 90, 5, 3},
      {1920, 1080, 128, 480, 270, 15, 9},
      {520, 300, 128, 130, 76, 5, 3},
      {200, 136, 256, 50, 34, 1, 1},
      {8, 8, 256, 2, 2, 1, 1},
  };
  for (const Case& c : cases) {
    const Geometry geometry = make_geometry(c.width, c.height, c.sb_size);
    std::string error;
    CHECK(s2::validate_geometry(geometry, error));
    CHECK(s2::mi_cols(geometry) == c.mi_cols);
    CHECK(s2::mi_rows(geometry) == c.mi_rows);
    CHECK(s2::sb_size_mi(geometry) == c.sb_size / 4);
    CHECK(s2::sb_cols(geometry) == c.sb_cols);
    CHECK(s2::sb_rows(geometry) == c.sb_rows);
    CHECK(s2::entropy_left_context_rows(geometry) == c.sb_rows * (c.sb_size / 4));
    CHECK(s2::entropy_above_context_cols(geometry) ==
          c.sb_cols * (c.sb_size / 4));
  }

  std::string error;
  Geometry bad = make_geometry(256, 256, 256);
  bad.sb_size = 512;
  CHECK(!s2::validate_geometry(bad, error));
  bad = make_geometry(256, 256, 256);
  bad.max_pb_aspect_ratio = 3;
  CHECK(!s2::validate_geometry(bad, error));
  bad = make_geometry(256, 256, 256);
  bad.chroma_subsampling_y = false;
  CHECK(!s2::validate_geometry(bad, error));
}

void test_boundary_implied_256() {
  // 1080p with a 256 root: the last SB row starts at mi 256, MiRows = 270.
  // The half-block test (256 + 32 < 270) fails, and the square root implies
  // HORZ (no syntax).
  const Geometry g1080 = make_geometry(1920, 1080, 256);
  const Boundary bottom =
      s2::partition_implied(g1080, 256, 0, BlockSize::k256x256);
  CHECK(!bottom.has_rows);
  CHECK(bottom.has_cols);
  CHECK(bottom.implied && bottom.implied_partition == kHorz);

  const Boundary full =
      s2::partition_implied(g1080, 0, 0, BlockSize::k256x256);
  CHECK(full.has_rows && full.has_cols && !full.implied);

  // 520x300: MiCols = 130, so the right 256 column at mi 128 is only 2 mi
  // wide.  The square root implies VERT in the top row.
  const Geometry g520 = make_geometry(520, 300, 256);
  const Boundary right =
      s2::partition_implied(g520, 0, 128, BlockSize::k256x256);
  CHECK(right.has_rows);
  CHECK(!right.has_cols);
  CHECK(right.implied && right.implied_partition == kVert);

  // 256x128 is rect-implied VERT; 128x256 is rect-implied HORZ.
  const Boundary h =
      s2::partition_implied_at_boundary(g1080, 0, 0, BlockSize::k256x128);
  CHECK(h.has_rows && h.has_cols && !h.implied);
  CHECK(s2::rect_type_implied_by_bsize(BlockSize::k256x128).partition == kVert);
  CHECK(s2::rect_type_implied_by_bsize(BlockSize::k128x256).partition == kHorz);

  // Thin blocks always imply NONE.
  const Boundary thin =
      s2::partition_implied(g1080, 0, 0, BlockSize::k4x32);
  CHECK(thin.implied && thin.implied_partition == kNone);
}

void test_allowed_and_forced_256() {
  const Geometry g1080 = make_geometry(1920, 1080, 256);

  auto allowed = s2::init_allowed_partitions(
      g1080, 0, 0, BlockSize::k256x256, true, false);
  CHECK(allowed.count == 4);
  CHECK(allowed.is_allowed(kNone));
  CHECK(allowed.is_allowed(kSplit));
  CHECK(allowed.is_allowed(kHorz));
  CHECK(allowed.is_allowed(kVert));

  // Partial bottom 256 root: NONE fails the half-block test, SPLIT and HORZ
  // remain; read_partition returns the implied HORZ first.
  allowed = s2::init_allowed_partitions(
      g1080, 256, 0, BlockSize::k256x256, true, false);
  CHECK(!allowed.is_allowed(kNone));
  CHECK(allowed.is_allowed(kSplit));
  CHECK(allowed.is_allowed(kHorz));
  CHECK(allowed.is_allowed(kVert));

  // 256x128 forbids HORZ by rect_type_implied_by_bsize().
  allowed = s2::init_allowed_partitions(
      g1080, 0, 0, BlockSize::k256x128, true, false);
  CHECK(allowed.is_allowed(kNone));
  CHECK(allowed.is_allowed(kVert));
  CHECK(!allowed.is_allowed(kHorz));
  CHECK(!allowed.is_allowed(kSplit));

  // 128x256 forbids VERT.
  allowed = s2::init_allowed_partitions(
      g1080, 0, 0, BlockSize::k128x256, true, false);
  CHECK(allowed.is_allowed(kNone));
  CHECK(allowed.is_allowed(kHorz));
  CHECK(!allowed.is_allowed(kVert));

  // 128x128 can split, 64x64 cannot.
  CHECK(s2::init_allowed_partitions(g1080, 0, 0, BlockSize::k128x128, true,
                                    false)
            .is_allowed(kSplit));
  CHECK(!s2::init_allowed_partitions(g1080, 0, 0, BlockSize::k64x64, true,
                                     false)
             .is_allowed(kSplit));
}

void test_contexts_and_syntax_256() {
  const Geometry g = make_geometry(640, 360, 256);
  s2::MiSizeGrid grid(g);

  // 256 root, no decided neighbours: do_split row 36, do_square_split row 4,
  // rect_type row 48.
  Contexts ctx =
      s2::partition_contexts(g, 0, 0, BlockSize::k256x256, grid);
  CHECK(ctx.do_split == 36);
  CHECK(ctx.do_square_split == 4);
  CHECK(ctx.rect_type == 48);

  // A 64x64 leaf to the left/above moves the 256 root to 38/37/39 and the
  // do_square_split row to 6/5/7.
  // The left neighbour of a root at (0,64) is the last 64x64 block of the
  // previous superblock: rows 0..15, cols 48..63.
  grid.fill_block(0, 48, BlockSize::k64x64);
  ctx = s2::partition_contexts(g, 0, 64, BlockSize::k256x256, grid);
  CHECK(ctx.do_split == 38);
  CHECK(ctx.do_square_split == 6);

  s2::MiSizeGrid grid2(g);
  // Above neighbour of the root at (64,0): rows 48..63, cols 0..15.
  grid2.fill_block(48, 0, BlockSize::k64x64);
  ctx = s2::partition_contexts(g, 64, 0, BlockSize::k256x256, grid2);
  CHECK(ctx.do_split == 37);
  CHECK(ctx.do_square_split == 5);

  // Both neighbours of the root at (64,64): left (64,63) and above (63,64).
  grid2.fill_block(48, 64, BlockSize::k64x64);
  grid2.fill_block(64, 48, BlockSize::k64x64);
  ctx = s2::partition_contexts(g, 64, 64, BlockSize::k256x256, grid2);
  CHECK(ctx.do_split == 39);
  CHECK(ctx.do_square_split == 7);

  // 256x128 uses row 8 (32) and rect row 11 (44); 128x256 row 7 (28) / 10 (40).
  s2::MiSizeGrid empty(g);
  ctx = s2::partition_contexts(g, 0, 0, BlockSize::k256x128, empty);
  CHECK(ctx.do_split == 32);
  CHECK(ctx.do_square_split == 0);
  CHECK(ctx.rect_type == 44);
  ctx = s2::partition_contexts(g, 0, 0, BlockSize::k128x256, empty);
  CHECK(ctx.do_split == 28);
  CHECK(ctx.rect_type == 40);

  // §5.20.3.2 syntax at a full 256 root: do_split then do_square_split.
  s2::FrameTree frame;
  std::string error;
  CHECK(s2::build_reference_frame(g, frame, error));
  const s2::Node& root = frame.superblocks.front();
  CHECK(root.size == BlockSize::k256x256);
  CHECK(root.partition == kSplit);
  CHECK(!root.forced);
  ctx = s2::partition_contexts(g, root.mi_row, root.mi_col, root.size, empty);
  std::vector<s2::PartitionSymbol> plan;
  CHECK(s2::plan_partition_syntax(root, ctx, plan, error));
  CHECK(plan.size() == 2);
  CHECK(plan[0].kind == s2::PartitionSymbolKind::kDoSplit);
  CHECK(plan[0].value == 1);
  CHECK(plan[0].context == 36);
  CHECK(plan[1].kind == s2::PartitionSymbolKind::kDoSquareSplit);
  CHECK(plan[1].value == 1);
  CHECK(plan[1].context == 4);

  // A full 256x128 node is rect-implied VERT: do_split only, no rect_type.
  s2::Node rect;
  rect.mi_row = 0;
  rect.mi_col = 0;
  rect.size = BlockSize::k256x128;
  rect.partition = kVert;
  rect.allowed = s2::init_allowed_partitions(g, 0, 0, BlockSize::k256x128,
                                             true, false);
  CHECK(s2::plan_partition_syntax(rect, ctx, plan, error));
  CHECK(plan.size() == 1);
  CHECK(plan[0].kind == s2::PartitionSymbolKind::kDoSplit);
  CHECK(plan[0].value == 1);

  // Boundary-forced 256 root codes nothing.
  const Geometry g1080 = make_geometry(1920, 1080, 256);
  s2::FrameTree frame1080;
  CHECK(s2::build_reference_frame(g1080, frame1080, error));
  const s2::Node& forced = frame1080.superblocks[32];  // row 256, col 0
  CHECK(forced.mi_row == 256);
  CHECK(forced.forced);
  CHECK(forced.partition == kHorz);
  CHECK(s2::plan_partition_syntax(forced, ctx, plan, error));
  CHECK(plan.empty());

  // CDF rows: the [2] dimension is PlaneStart, not the SB flag.
  CHECK(s2::do_split_cdf_row(0, 36) == &av2enc::basic::kDoSplitCdf[0][36][0]);
  CHECK(s2::do_split_cdf_row(1, 63) == &av2enc::basic::kDoSplitCdf[1][63][0]);
  CHECK(s2::do_split_cdf_row(0, 64) == nullptr);
  CHECK(s2::do_square_split_cdf_row(0, 4) ==
        &av2enc::basic::kDoSquareSplitCdf[0][4][0]);
  CHECK(s2::do_square_split_cdf_row(0, 7) ==
        &av2enc::basic::kDoSquareSplitCdf[0][7][0]);
  CHECK(s2::do_square_split_cdf_row(0, 8) == nullptr);
  CHECK(s2::rect_type_cdf_row(0, 48) == &av2enc::basic::kRectTypeCdf[0][48][0]);
  CHECK(s2::rect_type_cdf_row(0, 64) == nullptr);
}

void test_chroma_edges_units() {
  const Geometry g = make_geometry(640, 360, 256);
  CHECK(s2::chroma_plane_residual_size(g, BlockSize::k256x256) ==
        BlockSize::k128x128);
  CHECK(s2::chroma_plane_residual_size(g, BlockSize::k256x128) ==
        BlockSize::k128x64);
  CHECK(s2::chroma_plane_residual_size(g, BlockSize::k128x256) ==
        BlockSize::k64x128);
  CHECK(s2::chroma_plane_residual_size(g, BlockSize::k128x128) ==
        BlockSize::k64x64);
  CHECK(s2::chroma_plane_residual_size(g, BlockSize::k64x64) ==
        BlockSize::k32x32);
  // Table 9.2 Subsampled_Size clamps 4x4 to 4x4 rather than BLOCK_INVALID;
  // chroma offset, not this table, suppresses chroma below 8x8 luma.
  CHECK(s2::chroma_plane_residual_size(g, BlockSize::k4x4) ==
        BlockSize::k4x4);

  // A partial 256x256 root keeps the coded size; only the visible rectangle
  // is clipped.
  const BlockRect bottom = s2::block_rect(g, 64, 0, BlockSize::k256x256);
  CHECK(bottom.x == 0 && bottom.y == 256);
  CHECK(bottom.width == 256 && bottom.height == 256);
  CHECK(bottom.visible_width == 256);
  CHECK(bottom.visible_height == 104);

  // CDEF: always 64x64 units, 16 per 256 SB, context shift 6.
  CHECK(s2::cdef_unit_size_pixels() == 64);
  CHECK(s2::cdef_units_per_sb(64) == 1);
  CHECK(s2::cdef_units_per_sb(128) == 2);
  CHECK(s2::cdef_units_per_sb(256) == 4);
  CHECK(s2::cdef_sb_width_log2(64) == 4);
  CHECK(s2::cdef_sb_width_log2(128) == 5);
  CHECK(s2::cdef_sb_width_log2(256) == 6);

  // CCSO with ccso_unit_matches_sb_size = 1.
  CHECK(s2::ccso_luma_size_log2_matching_sb(64) == 6);
  CHECK(s2::ccso_luma_size_log2_matching_sb(128) == 7);
  CHECK(s2::ccso_luma_size_log2_matching_sb(256) == 8);
  CHECK(s2::ccso_unit_mi_shift(64) == 4);
  CHECK(s2::ccso_unit_mi_shift(128) == 5);
  CHECK(s2::ccso_unit_mi_shift(256) == 6);
  // CCSO from tile alignment.
  CHECK(s2::ccso_luma_size_log2_from_tile_alignment(0) == 8);
  CHECK(s2::ccso_luma_size_log2_from_tile_alignment(32) == 7);
  CHECK(s2::ccso_luma_size_log2_from_tile_alignment(16) == 6);
  CHECK(s2::ccso_luma_size_log2(256, true, 0) == 8);
  CHECK(s2::ccso_luma_size_log2(256, false, 32) == 7);

  // 520x300 top-right 256 root is only 2 mi wide: chroma offset recursion
  // still maps each 8x8 leaf to a 4x4 chroma residual.
  const Geometry g520 = make_geometry(520, 300, 256);
  const s2::ChromaBlock inherited{0, 128, BlockSize::k256x256};
  const auto root = s2::resolve_chroma(
      g520, 0, 128, BlockSize::k256x256, kVert, true, false, inherited);
  CHECK(root.block.size == BlockSize::k256x256);
  CHECK(!root.chroma_offset);
}

std::string serialize_node(const s2::Node& node) {
  std::string out;
  out += std::to_string(node.mi_row) + "," + std::to_string(node.mi_col) + ",";
  out += std::to_string(static_cast<int>(node.size)) + "," +
         std::to_string(static_cast<int>(node.partition)) + "," +
         std::to_string(node.forced) + "," +
         std::to_string(node.has_chroma) + "," +
         std::to_string(node.chroma_offset) + "," +
         std::to_string(static_cast<int>(node.chroma.size)) + ";";
  for (const auto& child : node.children) out += serialize_node(child);
  return out;
}

std::string serialize_frame(const s2::FrameTree& frame) {
  std::string out;
  for (const auto& root : frame.superblocks) out += serialize_node(root);
  return out;
}

void test_reference_tree() {
  struct Case {
    int width;
    int height;
    int sb_size;
    int superblocks;
    int nodes;
    int leaves;
    std::uint64_t digest;
  };
  const Case cases[] = {
      {256, 256, 256, 1, 21, 16, 0xb1c19dcf73682ffcULL},
      {640, 360, 256, 6, 85, 60, 0x39966745e16f3603ULL},
      {1920, 1080, 256, 40, 712, 510, 0xcc96e8f191c1e408ULL},
      {520, 300, 256, 6, 178, 78, 0xffef91337db99a97ULL},
      {520, 300, 128, 15, 168, 78, 0x62676a57cd225e4dULL},
      {128, 128, 128, 1, 5, 4, 0xee86f38467a69566ULL},
  };
  for (const Case& c : cases) {
    const Geometry geometry = make_geometry(c.width, c.height, c.sb_size);
    s2::FrameTree frame;
    std::string error;
    CHECK(s2::build_reference_frame(geometry, frame, error));
    CHECK(frame.superblocks.size() ==
          static_cast<std::size_t>(c.superblocks));
    CHECK(frame.nodes == c.nodes);
    CHECK(frame.leaves == c.leaves);
    CHECK(fnv1a(serialize_frame(frame)) == c.digest);
    CHECK(s2::validate_frame(geometry, frame, error));

    std::vector<s2::LeafBlock> leaves;
    for (const auto& root : frame.superblocks) {
      s2::collect_leaves(root, leaves);
    }
    CHECK(leaves.size() == static_cast<std::size_t>(c.leaves));
    for (const auto& leaf : leaves) {
      CHECK(s2::is_square(leaf.size));
      CHECK(s2::block_width(leaf.size) >= 8);
      CHECK(s2::block_width(leaf.size) <= 64);
    }

    // Determinism: same geometry -> identical tree serialization.
    s2::FrameTree again;
    CHECK(s2::build_reference_frame(geometry, again, error));
    CHECK(serialize_frame(frame) == serialize_frame(again));
  }

  // The 256 root is the only place the +4 do_square_split row is used.
  const Geometry g = make_geometry(256, 256, 256);
  s2::FrameTree frame;
  std::string error;
  CHECK(s2::build_reference_frame(g, frame, error));
  CHECK(frame.superblocks.front().size == BlockSize::k256x256);
  CHECK(frame.superblocks.front().partition == kSplit);
  CHECK(frame.superblocks.front().children.size() == 4);
  for (const auto& child : frame.superblocks.front().children) {
    CHECK(child.size == BlockSize::k128x128);
    CHECK(child.partition == kSplit);
  }
}

void test_cross_check_tables() {
  for (int value = 0; value < s2::kBlockSizeCount; ++value) {
    const auto size = static_cast<BlockSize>(value);
    const auto s1size = static_cast<s1::BlockSize>(value);
    const auto psize = static_cast<p::BlockSize>(value);
    CHECK(s2::mi_width_log2(size) == s1::mi_width_log2(s1size));
    CHECK(s2::mi_height_log2(size) == s1::mi_height_log2(s1size));
    CHECK(s2::mi_width_log2(size) == p::mi_width_log2(psize));
    CHECK(s2::mi_height_log2(size) == p::mi_height_log2(psize));
    CHECK(s2::block_width(size) == s1::block_width(s1size));
    CHECK(s2::block_height(size) == s1::block_height(s1size));
    CHECK(s2::block_width(size) == p::block_width(psize));
    CHECK(s2::partition_size_adjust(size) ==
          s1::partition_size_adjust(s1size));
    CHECK(s2::partition_size_adjust(size) == p::partition_size_adjust(psize));
    CHECK(s2::partition_size_adjust_rect_type(size) ==
          s1::partition_size_adjust_rect_type(s1size));
    CHECK(s2::partition_size_adjust_rect_type(size) ==
          p::partition_size_adjust_rect_type(psize));
    const auto implied = s2::rect_type_implied_by_bsize(size);
    const auto s1implied = s1::rect_type_implied_by_bsize(s1size);
    const auto pimplied = p::rect_type_implied_by_bsize(psize);
    CHECK(implied.implied == s1implied.implied);
    CHECK(implied.implied == pimplied.implied);
    CHECK(static_cast<int>(implied.partition) ==
          static_cast<int>(s1implied.partition));
    CHECK(static_cast<int>(implied.partition) ==
          static_cast<int>(pimplied.partition));
    for (const PartitionType part : {kNone, kSplit, kHorz, kVert}) {
      const auto expected = static_cast<int>(s1::partition_subsize(
          static_cast<s1::PartitionType>(part), s1size));
      CHECK(static_cast<int>(s2::partition_subsize(part, size)) == expected);
      CHECK(static_cast<int>(s2::partition_subsize(part, size)) ==
            static_cast<int>(p::partition_subsize(
                static_cast<p::PartitionType>(part), psize)));
    }
  }
}

void test_cross_check_geometry_and_rules() {
  const int widths[] = {1, 7, 8, 9, 16, 63, 64, 65, 127, 128, 129,
                        200, 255, 256, 257, 520, 527, 640, 1080, 1920};
  const int heights[] = {1, 7, 8, 9, 16, 63, 64, 65, 127, 128, 129,
                         136, 255, 256, 300, 360, 361, 1080};
  for (const int width : widths) {
    for (const int height : heights) {
      const Geometry g256 = make_geometry(width, height, 256);
      const Geometry g128 = make_geometry(width, height, 128);
      const Geometry g64 = make_geometry(width, height, 64);
      const s1::Geometry s1g = to_sb128(g128);
      const p::FrameGeometry pg = to_partition(g128);

      // MiCols/MiRows do not depend on SbSize.
      CHECK(s2::mi_cols(g256) == s2::mi_cols(g64));
      CHECK(s2::mi_rows(g128) == s2::mi_rows(g256));
      CHECK(s2::mi_cols(g256) == s1::mi_cols(s1g));
      CHECK(s2::mi_rows(g256) == s1::mi_rows(s1g));
      CHECK(s2::mi_cols(g256) == p::mi_cols(pg));
      CHECK(s2::mi_rows(g256) == p::mi_rows(pg));

      const int mi_c = s2::mi_cols(g256);
      const int mi_r = s2::mi_rows(g256);
      CHECK(s2::sb_cols(g256) == (mi_c + 63) / 64);
      CHECK(s2::sb_rows(g256) == (mi_r + 63) / 64);
      CHECK(s2::sb_cols(g128) == (mi_c + 31) / 32);
      CHECK(s2::sb_rows(g128) == (mi_r + 31) / 32);
      CHECK(s2::sb_cols(g64) == (mi_c + 15) / 16);
      CHECK(s2::entropy_left_context_rows(g256) == s2::sb_rows(g256) * 64);
      CHECK(s2::entropy_above_context_cols(g256) == s2::sb_cols(g256) * 64);

      const int row_step = mi_r > 120 ? 7 : 1;
      const int col_step = mi_c > 120 ? 7 : 1;
      for (int row = -1; row < mi_r + 1; row += row_step) {
        for (int col = -1; col < mi_c + 1; col += col_step) {
          for (int value = 0; value < s2::kBlockSizeCount; ++value) {
            const auto size = static_cast<BlockSize>(value);
            const auto s1size = static_cast<s1::BlockSize>(value);
            const auto psize = static_cast<p::BlockSize>(value);
            const Boundary ours = s2::partition_implied(g128, row, col, size);
            const s1::Boundary theirs =
                s1::partition_implied(s1g, row, col, s1size);
            CHECK(ours.has_rows == theirs.has_rows);
            CHECK(ours.has_cols == theirs.has_cols);
            CHECK(ours.implied == theirs.implied);
            CHECK(static_cast<int>(ours.implied_partition) ==
                  static_cast<int>(theirs.implied_partition));
            if (row >= 0 && col >= 0) {
              for (const PartitionType part : {kNone, kSplit, kHorz, kVert}) {
                CHECK(s2::is_partition_allowed(g128, row, col, size, part, true,
                                               false) ==
                      s1::is_partition_allowed(
                          s1g, row, col, s1size,
                          static_cast<s1::PartitionType>(part), true, false));
              }
              const auto a =
                  s2::init_allowed_partitions(g128, row, col, size, true,
                                              false);
              const auto b = s1::init_allowed_partitions(
                  s1g, row, col, s1size, true, false);
              CHECK(a.count == b.count);
              for (int i = 0; i < 4; ++i) CHECK(a.allowed[i] == b.allowed[i]);
            }
            // The partition module does not model 256 SBs, but its pure
            // per-block rules do not care about the geometry's sb_size.
            if (row >= 0 && col >= 0 && (value % 3) == 0) {
              for (const PartitionType part : {kNone, kSplit, kHorz, kVert}) {
                CHECK(s2::is_partition_allowed(g128, row, col, size, part, true,
                                               false) ==
                      p::is_partition_allowed(
                          pg, row, col, psize,
                          static_cast<p::PartitionType>(part), true, false));
              }
            }
          }
        }
      }
    }
  }
}

void test_exhaustive_geometry_rules() {
  // Every mi position and every block size for a set of partial geometries.
  // This is the "no hidden boundary hole" check; the broad sweep above uses
  // sampling for the very large frames.
  const Geometry geometries[] = {
      make_geometry(520, 300, 128), make_geometry(300, 520, 128),
      make_geometry(256, 256, 128), make_geometry(256, 300, 128),
      make_geometry(300, 256, 128), make_geometry(200, 136, 128),
      make_geometry(136, 200, 128), make_geometry(128, 128, 128),
  };
  for (const Geometry& geometry : geometries) {
    const s1::Geometry s1g = to_sb128(geometry);
    const int cols = s2::mi_cols(geometry);
    const int rows = s2::mi_rows(geometry);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        for (int value = 0; value < s2::kBlockSizeCount; ++value) {
          const auto size = static_cast<BlockSize>(value);
          const auto s1size = static_cast<s1::BlockSize>(value);
          const Boundary ours = s2::partition_implied(geometry, row, col, size);
          const s1::Boundary theirs =
              s1::partition_implied(s1g, row, col, s1size);
          CHECK(ours.has_rows == theirs.has_rows);
          CHECK(ours.has_cols == theirs.has_cols);
          CHECK(ours.implied == theirs.implied);
          CHECK(static_cast<int>(ours.implied_partition) ==
                static_cast<int>(theirs.implied_partition));
          for (const PartitionType part : {kNone, kSplit, kHorz, kVert}) {
            CHECK(s2::is_partition_allowed(geometry, row, col, size, part, true,
                                           false) ==
                  s1::is_partition_allowed(
                      s1g, row, col, s1size,
                      static_cast<s1::PartitionType>(part), true, false));
          }
          const auto a =
              s2::init_allowed_partitions(geometry, row, col, size, true,
                                          false);
          const auto b =
              s1::init_allowed_partitions(s1g, row, col, s1size, true, false);
          CHECK(a.count == b.count);
          for (int i = 0; i < 4; ++i) CHECK(a.allowed[i] == b.allowed[i]);
        }
      }
    }
  }
}

void test_cross_check_contexts() {
  const Geometry geometries[] = {
      make_geometry(128, 128, 128), make_geometry(256, 256, 128),
      make_geometry(640, 360, 128), make_geometry(1920, 1080, 128),
      make_geometry(520, 300, 128),
  };
  const BlockSize sizes[] = {
      BlockSize::k256x256, BlockSize::k256x128, BlockSize::k128x256,
      BlockSize::k128x128, BlockSize::k128x64,   BlockSize::k64x128,
      BlockSize::k64x64,   BlockSize::k32x32,    BlockSize::k8x8,
  };
  for (const Geometry& geometry : geometries) {
    const s1::Geometry s1g = to_sb128(geometry);
    const p::FrameGeometry pg = to_partition(geometry);
    s2::MiSizeGrid ours(geometry);
    s1::MiSizeGrid theirs(s1g);
    p::MiSizeGrid part(pg);
    for (int row = 0; row < s2::mi_rows(geometry); ++row) {
      for (int col = 0; col < s2::mi_cols(geometry); ++col) {
        const BlockSize size =
            ((row / 4 + col / 4) & 1) != 0 ? BlockSize::k64x64
                                           : BlockSize::k32x32;
        ours.set(row, col, size);
        theirs.set(row, col, static_cast<s1::BlockSize>(size));
        part.set(row, col, static_cast<p::BlockSize>(size));
      }
    }
    for (int row = 0; row < s2::mi_rows(geometry); row += 5) {
      for (int col = 0; col < s2::mi_cols(geometry); col += 5) {
        for (const BlockSize size : sizes) {
          const Contexts a =
              s2::partition_contexts(geometry, row, col, size, ours);
          const s1::Contexts b = s1::partition_contexts(
              s1g, row, col, static_cast<s1::BlockSize>(size), theirs);
          const p::PartitionContexts c = p::partition_contexts(
              pg, row, col, static_cast<p::BlockSize>(size), part);
          CHECK(a.do_split == b.do_split);
          CHECK(a.do_split == c.do_split);
          CHECK(a.do_square_split == b.do_square_split);
          CHECK(a.do_square_split == c.do_square_split);
          CHECK(a.rect_type == b.rect_type);
          CHECK(a.rect_type == c.rect_type);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Header / stream tests
// ---------------------------------------------------------------------------

class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  bool read_bit() {
    if (pos_ >= bytes_.size() * 8U) {
      ok_ = false;
      return false;
    }
    const std::uint8_t byte = bytes_[pos_ / 8U];
    const int shift = 7 - static_cast<int>(pos_ % 8U);
    ++pos_;
    return ((byte >> shift) & 1U) != 0U;
  }

  std::uint32_t read_bits(unsigned count) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < count; ++i) {
      value = (value << 1U) | (read_bit() ? 1U : 0U);
    }
    return value;
  }

  std::uint32_t uvlc() {
    unsigned leading_zeros = 0;
    while (!read_bit()) {
      ++leading_zeros;
      if (leading_zeros > 31) {
        ok_ = false;
        return 0;
      }
    }
    const std::uint32_t base = (1U << leading_zeros) - 1U;
    return base + (leading_zeros == 0 ? 0 : read_bits(leading_zeros));
  }

  bool ok() const { return ok_; }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t pos_ = 0;
  bool ok_ = true;
};

struct ParsedSbFlags {
  bool use_256 = false;
  bool use_128 = false;
  bool read_use_128 = false;
};

ParsedSbFlags parse_sb_flags(const av2enc::sb256::StreamConfig& config,
                             const av2enc::sb256::SbFlags& flags) {
  BitWriter writer;
  s2::write_sequence_header(writer, config, flags);
  const std::vector<std::uint8_t> bytes = writer.take_bytes();
  BitReader reader(bytes);
  CHECK(reader.uvlc() == 0);        // seq_header_id
  CHECK(reader.read_bits(5) == 0);  // seq_profile_idc
  CHECK(reader.read_bit());         // single_picture_header_flag
  reader.read_bits(5);              // seq_level_idx
  CHECK(reader.uvlc() == 0);        // chroma_format_idc
  CHECK(reader.uvlc() == 1);        // bit_depth_idc
  const unsigned width_bits = reader.read_bits(4) + 1;
  const unsigned height_bits = reader.read_bits(4) + 1;
  reader.read_bits(width_bits);
  reader.read_bits(height_bits);
  CHECK(!reader.read_bit());  // seq_cropping_window_present_flag
  ParsedSbFlags out;
  out.use_256 = reader.read_bit();
  if (!out.use_256) {
    out.use_128 = reader.read_bit();
    out.read_use_128 = true;
  }
  CHECK(reader.ok());
  return out;
}

void test_headers() {
  av2enc::sb256::StreamConfig config;
  config.width = 256;
  config.height = 256;

  const ParsedSbFlags primary =
      parse_sb_flags(config, s2::SbFlags{true, false});
  CHECK(primary.use_256);
  CHECK(!primary.read_use_128);

  const ParsedSbFlags control =
      parse_sb_flags(config, s2::SbFlags{false, true});
  CHECK(!control.use_256);
  CHECK(control.read_use_128);
  CHECK(control.use_128);

  // The 256 header omits the use_128 bit, so it is exactly one bit shorter.
  BitWriter a;
  BitWriter b;
  const std::size_t bits256 = s2::write_sequence_header(a, config, {true, false});
  const std::size_t bits128 = s2::write_sequence_header(b, config, {false, true});
  CHECK(bits128 == bits256 + 1);

  struct Case {
    int width;
    int height;
    std::size_t bits;
  };
  // 128x128 with a 128 frame SB has one tile, so no tile increment bits: 22.
  const Case cases[] = {
      {128, 128, 22},
      {256, 256, 24},
      {640, 360, 24},
      {1920, 1080, 24},
      {520, 300, 24},
  };
  for (const Case& c : cases) {
    av2enc::sb256::StreamConfig cfg;
    cfg.width = static_cast<std::uint16_t>(c.width);
    cfg.height = static_cast<std::uint16_t>(c.height);
    BitWriter writer;
    CHECK(s2::write_frame_header(writer, cfg) == c.bits);
  }
}

void test_stream_builder() {
  struct Case {
    int width;
    int height;
    int leaves;
    int superblocks;
    std::size_t symbols;
    std::size_t frame_bytes;
    std::size_t tile_bytes;
    const char* frame_sha256;
  };
  const Case cases[] = {
      {256, 256, 16, 4, 24, 33, 12,
       "b058de2377804e5b643866bda5e3b16cfd4b435f9c3fafab1ccf19a8924886ee"},
      {640, 360, 60, 15, 90, 61, 39,
       "fd7da565aa788d56dd6406bea5e84c03586ba92ab2b333b18c5002a73ddc7f03"},
      {1920, 1080, 510, 135, 765, 335, 312,
       "71894e605cbb7fc54eca6cc0dee062112f4012046101b6830089962ffb153878"},
      {520, 300, 78, 15, 145, 89, 67,
       "6cd70e53f53124a9cd2a4734c5a204374e6ed00657b171b54da61243ffc5979d"},
  };
  for (const Case& c : cases) {
    av2enc::sb256::StreamConfig config;
    config.width = static_cast<std::uint16_t>(c.width);
    config.height = static_cast<std::uint16_t>(c.height);
    s2::StreamResult first;
    std::string error;
    CHECK(s2::make_sb256_stream(config, first, error));
    CHECK(first.leaves == c.leaves);
    CHECK(first.superblocks == c.superblocks);
    CHECK(first.symbols == c.symbols);
    CHECK(first.frame.size() == c.frame_bytes);
    CHECK(first.tile.size() == c.tile_bytes);
    CHECK(sha256_hex(first.frame) == c.frame_sha256);
    CHECK(!first.frame.empty() && !first.tile.empty());

    // Determinism in-process.
    s2::StreamResult second;
    CHECK(s2::make_sb256_stream(config, second, error));
    CHECK(first.frame == second.frame);
    CHECK(first.tile == second.tile);

    // The control header (use_256=0, use_128=1) carries byte-identical tile
    // syntax, because an intra frame is 128-SB either way.
    s2::StreamResult control;
    CHECK(s2::make_use128_control_stream(config, control, error));
    CHECK(control.tile == first.tile);
    CHECK(control.frame != first.frame);

    // The verified 128-SB module produces the same tile for the geometries it
    // supports (all except the 520x300 right partial column).
    if (c.width % 128 == 0) {
      s1::StreamConfig old;
      old.width = static_cast<std::uint16_t>(c.width);
      old.height = static_cast<std::uint16_t>(c.height);
      s1::StreamResult legacy;
      if (s1::make_sb128_stream(old, legacy, error)) {
        CHECK(legacy.tile == first.tile);
      }
    }
  }

  // Positive control stream at 128x128 with a pinned golden.
  av2enc::sb256::StreamConfig control;
  control.width = 128;
  control.height = 128;
  s2::StreamResult control_stream;
  std::string error;
  CHECK(s2::make_sb256_stream(control, control_stream, error));
  CHECK(control_stream.leaves == 4);
  CHECK(control_stream.superblocks == 1);
  CHECK(control_stream.symbols == 6);
  CHECK(control_stream.frame.size() == 25);
  CHECK(control_stream.tile.size() == 5);
  CHECK(sha256_hex(control_stream.frame) ==
        "6bc442569621e7db23911485a55904f7683b3c9cd238bf9acaaa95e803971380");

  s2::StreamResult use128;
  CHECK(s2::make_use128_control_stream(control, use128, error));
  CHECK(use128.tile == control_stream.tile);
  CHECK(use128.frame.size() == 25);
  CHECK(sha256_hex(use128.frame) ==
        "6f75573d9a5ee7050d07af82527ab98fe42081248677d14a109df20e6361c3ea");

  // Geometry support: all of the gate cases, plus partial and legacy sizes.
  struct Support {
    int width;
    int height;
    bool supported;
  };
  const Support support[] = {
      {256, 256, true},   {640, 360, true},   {1920, 1080, true},
      {520, 300, true},   {128, 128, true},   {200, 136, true},
      {136, 200, true},   {132, 132, true},   {1024, 576, true},
      {8, 8, true},       {1, 1, true},
  };
  for (const Support& s : support) {
    const Geometry geometry = make_geometry(s.width, s.height, 128);
    std::string why;
    CHECK(s2::stream_geometry_supported(geometry, why) == s.supported);
  }

  // 520x300 is rejected by the 128-SB minimal writer (its 64x64-only leaf
  // coder cannot represent the 8 px right edge) but is supported here.
  s1::StreamConfig odd;
  odd.width = 520;
  odd.height = 300;
  s1::StreamResult legacy;
  CHECK(!s1::make_sb128_stream(odd, legacy, error));

  // Adaptive CDFs are also supported (the decoder updates in lockstep).
  av2enc::sb256::StreamConfig adaptive;
  adaptive.width = 128;
  adaptive.height = 128;
  adaptive.disable_cdf_update = false;
  s2::StreamResult a;
  CHECK(s2::make_sb256_stream(adaptive, a, error));
  s2::StreamResult b;
  CHECK(s2::make_sb256_stream(adaptive, b, error));
  CHECK(a.frame == b.frame);
  adaptive.disable_cdf_update = true;
  s2::StreamResult frozen;
  CHECK(s2::make_sb256_stream(adaptive, frozen, error));
  CHECK(a.tile != frozen.tile);
}

}  // namespace

int main() {
  test_block_tables();
  test_sb_config();
  test_partition_subsize();
  test_geometry();
  test_boundary_implied_256();
  test_allowed_and_forced_256();
  test_contexts_and_syntax_256();
  test_chroma_edges_units();
  test_reference_tree();
  test_cross_check_tables();
  test_cross_check_geometry_and_rules();
  test_exhaustive_geometry_rules();
  test_cross_check_contexts();
  test_headers();
  test_stream_builder();
  return test::finish("sb256_test");
}
