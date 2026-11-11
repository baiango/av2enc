// Tests for the AV2 semi-decoupled partitioning (SDP) CPU reference.
//
// Coverage:
//   * implied/boundary rules for SHARED/LUMA/CHROMA trees, including the
//     CHROMA_PART 8x8 NONE and 64x64 ChromaPartitionKnown cases
//   * rect_type implied by bsize for CHROMA_PART
//   * allowed partitions per TreeType, ext/uneven gating, aspect ratio
//   * chroma offset / ChromaMiSize bookkeeping (incl. HORZ_3@8x32)
//   * SDP trigger at 64x64 in 64 and 128 superblocks, partial superblocks
//   * deterministic decisions and luma/chroma divergence
//   * symbol plans + independent decoder round trips and golden prefixes
//   * TreeType switch positions
//   * validation rejects malformed trees
//   * extended SDP (inter) region_type planning
//   * rect-only parity with encoder/partition when SDP is off

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "encoder/partition/partition.h"
#include "encoder/partition_ext/partition_ext.h"
#include "encoder/sdp/sdp.h"

namespace p = av2enc::partition;
namespace px = av2enc::partition_ext;
namespace s = av2enc::sdp;

static int g_failures = 0;

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      ++g_failures;                                                          \
      std::cerr << __FILE__ << ':' << __LINE__                               \
                << ": CHECK failed: " #condition << '\n';                    \
    }                                                                        \
  } while (false)

#define CHECK_EQ(actual, expected)                                           \
  do {                                                                       \
    const long long a_ = static_cast<long long>(actual);                     \
    const long long b_ = static_cast<long long>(expected);                   \
    if (a_ != b_) {                                                          \
      ++g_failures;                                                          \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK_EQ failed: "      \
                << #actual << " = " << a_ << ", expected " << b_ << '\n';    \
    }                                                                        \
  } while (false)

#define CHECK_STR(actual, expected)                                          \
  do {                                                                       \
    const std::string a_ = (actual);                                         \
    const std::string b_ = (expected);                                       \
    if (a_ != b_) {                                                          \
      ++g_failures;                                                          \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK_STR failed: "     \
                << #actual << " = \"" << a_ << "\", expected \"" << b_       \
                << "\"\n";                                                   \
    }                                                                        \
  } while (false)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static s::Tools intra_tools(bool ext = false, bool uneven = false) {
  s::Tools tools;
  tools.enable_sdp = true;
  tools.frame_is_intra = true;
  tools.enable_ext_partitions = ext;
  tools.enable_uneven_4way_partitions = uneven;
  return tools;
}

static s::Tools inter_tools(bool extended_sdp = true) {
  s::Tools tools;
  tools.enable_sdp = true;
  tools.enable_extended_sdp = extended_sdp;
  tools.frame_is_intra = false;
  tools.single_picture_header_flag = false;
  return tools;
}

static p::FrameGeometry geom(int w, int h, int sb) {
  p::FrameGeometry g;
  g.width = w;
  g.height = h;
  g.sb_size = sb;
  return g;
}

// Keep the exhaustive reference solver tractable on large frames.
static s::DecisionOptions capped_options(int width, int height) {
  s::DecisionOptions options;
  if (width * height > 200 * 200) {
    options.max_depth = 3;
  } else {
    options.max_depth = 6;
  }
  return options;
}

static std::uint32_t mix(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

static double pseudo(std::uint32_t seed, int r, int c, int size, int tree,
                     int tag) {
  std::uint32_t v = seed;
  v = mix(v + static_cast<std::uint32_t>(r) * 0x9e3779b1U);
  v = mix(v + static_cast<std::uint32_t>(c) * 0x85ebca6bU);
  v = mix(v + static_cast<std::uint32_t>(size) * 0xc2b2ae35U);
  v = mix(v + static_cast<std::uint32_t>(tree) * 0x27d4eb2fU);
  v = mix(v + static_cast<std::uint32_t>(tag) * 0x165667b1U);
  return static_cast<double>(v >> 8) / 16777216.0;
}

static s::NodeCosts generic_costs(const s::NodeInfo& info, std::uint32_t seed) {
  s::NodeCosts costs;
  const int r = info.mi_row;
  const int c = info.mi_col;
  const int size = static_cast<int>(info.size);
  const int tree = static_cast<int>(info.tree_type);
  costs.leaf = pseudo(seed, r, c, size, tree, 0) * 120.0;
  costs.none_signal = pseudo(seed, r, c, size, tree, 1) * 1.5;
  costs.split_signal = pseudo(seed, r, c, size, tree, 2) * 1.5;
  costs.horz_signal = pseudo(seed, r, c, size, tree, 3) * 1.5;
  costs.vert_signal = pseudo(seed, r, c, size, tree, 4) * 1.5;
  costs.horz3_signal = pseudo(seed, r, c, size, tree, 5) * 1.5;
  costs.vert3_signal = pseudo(seed, r, c, size, tree, 6) * 1.5;
  costs.horz4a_signal = pseudo(seed, r, c, size, tree, 7) * 1.5;
  costs.horz4b_signal = pseudo(seed, r, c, size, tree, 8) * 1.5;
  costs.vert4a_signal = pseudo(seed, r, c, size, tree, 9) * 1.5;
  costs.vert4b_signal = pseudo(seed, r, c, size, tree, 10) * 1.5;
  costs.region_type[0] = pseudo(seed, r, c, size, tree, 11) * 1.5;
  costs.region_type[1] = pseudo(seed, r, c, size, tree, 12) * 1.5;
  return costs;
}

// Force PARTITION_NONE everywhere.
static s::NodeCosts none_costs(const s::NodeInfo&) {
  s::NodeCosts costs;
  costs.leaf = 0.0;
  costs.none_signal = 0.0;
  costs.split_signal = s::kInfiniteCost;
  costs.horz_signal = s::kInfiniteCost;
  costs.vert_signal = s::kInfiniteCost;
  costs.horz3_signal = s::kInfiniteCost;
  costs.vert3_signal = s::kInfiniteCost;
  costs.horz4a_signal = s::kInfiniteCost;
  costs.horz4b_signal = s::kInfiniteCost;
  costs.vert4a_signal = s::kInfiniteCost;
  costs.vert4b_signal = s::kInfiniteCost;
  return costs;
}

// Force root HORZ, then VERT for 32x64 children, then NONE.
static s::NodeCosts horz_then_vert_costs(const s::NodeInfo& info) {
  s::NodeCosts costs;
  costs.leaf = 0.0;
  if (info.size == p::BlockSize::k64x64) {
    costs.leaf = s::kInfiniteCost;
    costs.horz_signal = 0.0;
    costs.horz3_signal = s::kInfiniteCost;
    costs.vert_signal = s::kInfiniteCost;
    costs.split_signal = s::kInfiniteCost;
    costs.horz4a_signal = s::kInfiniteCost;
    costs.horz4b_signal = s::kInfiniteCost;
    costs.vert4a_signal = s::kInfiniteCost;
    costs.vert4b_signal = s::kInfiniteCost;
  } else if (info.size == p::BlockSize::k64x32) {
    costs.leaf = s::kInfiniteCost;
    costs.vert_signal = 0.0;
    costs.horz_signal = s::kInfiniteCost;
    costs.split_signal = s::kInfiniteCost;
    costs.horz3_signal = s::kInfiniteCost;
    costs.vert3_signal = s::kInfiniteCost;
    costs.horz4a_signal = s::kInfiniteCost;
    costs.horz4b_signal = s::kInfiniteCost;
    costs.vert4a_signal = s::kInfiniteCost;
    costs.vert4b_signal = s::kInfiniteCost;
  } else {
    costs.horz_signal = s::kInfiniteCost;
    costs.vert_signal = s::kInfiniteCost;
    costs.split_signal = s::kInfiniteCost;
    costs.horz3_signal = s::kInfiniteCost;
    costs.vert3_signal = s::kInfiniteCost;
    costs.horz4a_signal = s::kInfiniteCost;
    costs.horz4b_signal = s::kInfiniteCost;
    costs.vert4a_signal = s::kInfiniteCost;
    costs.vert4b_signal = s::kInfiniteCost;
  }
  return costs;
}

// Force root HORZ and both 32x64 children HORZ (so ChromaFollowsLuma clears).
static s::NodeCosts horz_then_horz_costs(const s::NodeInfo& info) {
  s::NodeCosts costs;
  costs.leaf = 0.0;
  costs.none_signal = 0.0;
  if (info.size == p::BlockSize::k64x64) {
    costs.leaf = s::kInfiniteCost;
    costs.horz_signal = 0.0;
  } else if (info.size == p::BlockSize::k64x32) {
    costs.leaf = s::kInfiniteCost;
    costs.horz_signal = 0.0;
  } else {
    return none_costs(info);
  }
  costs.split_signal = s::kInfiniteCost;
  costs.vert_signal = s::kInfiniteCost;
  costs.horz3_signal = s::kInfiniteCost;
  costs.vert3_signal = s::kInfiniteCost;
  costs.horz4a_signal = s::kInfiniteCost;
  costs.horz4b_signal = s::kInfiniteCost;
  costs.vert4a_signal = s::kInfiniteCost;
  costs.vert4b_signal = s::kInfiniteCost;
  return costs;
}

static bool symbol_alphabet_ok(const s::Symbol& symbol);

static bool is_sdp_root(const s::Node& node) { return node.sdp_split; }

static const s::Node* find_sdp_child(const s::Node& root, s::TreeType tree) {
  if (root.sdp_split) {
    for (const s::Node& child : root.children) {
      if (child.tree_type == tree) return &child;
    }
  }
  for (const s::Node& child : root.children) {
    const s::Node* found = find_sdp_child(child, tree);
    if (found != nullptr) return found;
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Geometry / implied / allowed
// ---------------------------------------------------------------------------

static void test_implied_rules() {
  const p::FrameGeometry g = geom(640, 360, 64);
  // SHARED implied matches the rect-only reference everywhere.
  const p::BlockSize sizes[] = {
      p::BlockSize::k64x64, p::BlockSize::k32x64, p::BlockSize::k32x32,
      p::BlockSize::k16x32, p::BlockSize::k8x8,   p::BlockSize::k4x8,
      p::BlockSize::k4x32,  p::BlockSize::k8x64,
  };
  for (p::BlockSize size : sizes) {
    for (int r = 0; r < p::mi_rows(g); r += 8) {
      for (int c = 0; c < p::mi_cols(g); c += 8) {
        const p::PartitionBoundary base =
            p::partition_implied(g, r, c, size);
        const s::Boundary ours =
            s::sdp_implied(g, r, c, size, s::TreeType::kSharedPart);
        CHECK_EQ(ours.has_rows, base.has_rows);
        CHECK_EQ(ours.has_cols, base.has_cols);
        CHECK_EQ(ours.implied, base.implied);
      }
    }
  }

  // Thin blocks always imply NONE.
  for (p::BlockSize size :
       {p::BlockSize::k4x4, p::BlockSize::k4x32, p::BlockSize::k32x4,
        p::BlockSize::k8x64, p::BlockSize::k64x8}) {
    const s::Boundary b =
        s::sdp_implied(g, 0, 0, size, s::TreeType::kSharedPart);
    CHECK(b.implied);
    CHECK_EQ(static_cast<int>(b.implied_partition),
             static_cast<int>(s::PartitionType::kNone));
  }

  // CHROMA_PART 8x8 is terminal even when the half-block test fails.
  const p::FrameGeometry tiny = geom(8, 8, 64);
  const s::Boundary chroma8 =
      s::sdp_implied(tiny, 0, 0, p::BlockSize::k8x8,
                     s::TreeType::kChromaPart);
  CHECK(chroma8.implied);
  CHECK_EQ(static_cast<int>(chroma8.implied_partition),
           static_cast<int>(s::PartitionType::kNone));
  const s::Boundary shared8 =
      s::sdp_implied(tiny, 0, 0, p::BlockSize::k8x8,
                     s::TreeType::kSharedPart);
  // At the 8x8 frame origin the shared 8x8 block still has half-block overlap,
  // so it is not implied; only CHROMA_PART forces NONE here.
  CHECK(!shared8.implied);

  // CHROMA_PART 64x64 with ChromaPartitionKnown returns LumaPartitions.
  const s::Boundary known =
      s::sdp_implied(g, 0, 0, p::BlockSize::k64x64,
                     s::TreeType::kChromaPart, /*chroma_partition_known=*/true,
                     s::PartitionType::kHorz);
  CHECK(known.implied);
  CHECK_EQ(static_cast<int>(known.implied_partition),
           static_cast<int>(s::PartitionType::kHorz));
  const s::Boundary unknown =
      s::sdp_implied(g, 0, 0, p::BlockSize::k64x64,
                     s::TreeType::kChromaPart, /*chroma_partition_known=*/false);
  CHECK(!unknown.implied || unknown.implied_partition != s::PartitionType::kHorz);

  // Right-edge 128 SB: 520x128 -> MiCols 130, the last root is implied VERT.
  const p::FrameGeometry right = geom(520, 128, 128);
  const s::Boundary edge =
      s::sdp_implied(right, 0, 128, p::BlockSize::k128x128,
                     s::TreeType::kSharedPart);
  CHECK(edge.implied);
  CHECK_EQ(static_cast<int>(edge.implied_partition),
           static_cast<int>(s::PartitionType::kVert));
}

static void test_rect_implied() {
  CHECK(s::sdp_rect_implied(p::BlockSize::k4x8, s::TreeType::kSharedPart)
            .implied);
  // 8x16 is not in the base table; only CHROMA_PART adds it.
  CHECK(!s::sdp_rect_implied(p::BlockSize::k8x16,
                             s::TreeType::kSharedPart)
             .implied);
  // CHROMA adds 8x16/8x32 HORZ and 16x8/32x8 VERT.
  for (p::BlockSize size :
       {p::BlockSize::k8x16, p::BlockSize::k8x32}) {
    const p::RectTypeImplied r =
        s::sdp_rect_implied(size, s::TreeType::kChromaPart);
    CHECK(r.implied);
    CHECK_EQ(static_cast<int>(r.partition),
             static_cast<int>(p::PartitionType::kHorz));
  }
  for (p::BlockSize size :
       {p::BlockSize::k16x8, p::BlockSize::k32x8}) {
    const p::RectTypeImplied r =
        s::sdp_rect_implied(size, s::TreeType::kChromaPart);
    CHECK(r.implied);
    CHECK_EQ(static_cast<int>(r.partition),
             static_cast<int>(p::PartitionType::kVert));
  }
  // The same sizes are NOT implied in SHARED.
  CHECK(!s::sdp_rect_implied(p::BlockSize::k8x32,
                             s::TreeType::kSharedPart)
             .implied);
}

static void test_allowed_sets() {
  const p::FrameGeometry g = geom(128, 128, 64);
  const s::Tools plain = intra_tools(false, false);

  const s::PartitionSet shared = s::sdp_init_allowed(
      g, 0, 0, p::BlockSize::k64x64, s::TreeType::kSharedPart, true, false,
      s::RegionType::kMixedRegion, plain);
  CHECK(shared.is_allowed(s::PartitionType::kNone));
  CHECK(shared.is_allowed(s::PartitionType::kHorz));
  CHECK(shared.is_allowed(s::PartitionType::kVert));
  CHECK(!shared.is_allowed(s::PartitionType::kSplit));
  CHECK(!shared.is_allowed(s::PartitionType::kHorz3));

  const s::PartitionSet luma = s::sdp_init_allowed(
      g, 0, 0, p::BlockSize::k64x64, s::TreeType::kLumaPart, true, false,
      s::RegionType::kMixedRegion, plain);
  CHECK(luma.is_allowed(s::PartitionType::kNone));
  CHECK(luma.is_allowed(s::PartitionType::kHorz));
  CHECK(luma.is_allowed(s::PartitionType::kVert));
  CHECK(!luma.is_allowed(s::PartitionType::kSplit));

  const s::PartitionSet chroma = s::sdp_init_allowed(
      g, 0, 0, p::BlockSize::k64x64, s::TreeType::kChromaPart, true, false,
      s::RegionType::kMixedRegion, plain);
  CHECK(chroma.is_allowed(s::PartitionType::kNone));
  CHECK(chroma.is_allowed(s::PartitionType::kHorz));
  CHECK(chroma.is_allowed(s::PartitionType::kVert));

  // 8x8 CHROMA_PART: NONE is allowed even at the frame boundary.
  const p::FrameGeometry tiny = geom(8, 8, 64);
  const s::PartitionSet chroma8 = s::sdp_init_allowed(
      tiny, 0, 0, p::BlockSize::k8x8, s::TreeType::kChromaPart, true, false,
      s::RegionType::kMixedRegion, plain);
  CHECK(chroma8.is_allowed(s::PartitionType::kNone));
  const s::PartitionSet shared8 = s::sdp_init_allowed(
      tiny, 0, 0, p::BlockSize::k8x8, s::TreeType::kSharedPart, true, false,
      s::RegionType::kMixedRegion, plain);
  CHECK(shared8.is_allowed(s::PartitionType::kNone));

  // Extended rows: CHROMA restrictions.
  const s::Tools ext = intra_tools(true, true);
  const s::PartitionSet c64 = s::sdp_init_allowed(
      g, 0, 0, p::BlockSize::k64x64, s::TreeType::kChromaPart, true, false,
      s::RegionType::kMixedRegion, ext);
  CHECK(c64.is_allowed(s::PartitionType::kHorz3));
  CHECK(c64.is_allowed(s::PartitionType::kVert3));
  CHECK(c64.is_allowed(s::PartitionType::kHorz4A));
  CHECK(c64.is_allowed(s::PartitionType::kVert4A));
  const s::PartitionSet c16 = s::sdp_init_allowed(
      g, 0, 0, p::BlockSize::k16x16, s::TreeType::kChromaPart, true, false,
      s::RegionType::kMixedRegion, ext);
  CHECK(!c16.is_allowed(s::PartitionType::kHorz3));
  CHECK(!c16.is_allowed(s::PartitionType::kVert3));
  CHECK(!c16.is_allowed(s::PartitionType::kHorz4A));
  CHECK(!c16.is_allowed(s::PartitionType::kVert4A));
  // SHARED/LUMA allow the extended rows at 16x16.
  const s::PartitionSet s16 = s::sdp_init_allowed(
      g, 0, 0, p::BlockSize::k16x16, s::TreeType::kSharedPart, true, false,
      s::RegionType::kMixedRegion, ext);
  CHECK(s16.is_allowed(s::PartitionType::kHorz3));

  // reduce_pb_aspect_ratio: NONE rejected, a too-thin non-NONE rejected.
  p::FrameGeometry narrow = geom(128, 128, 64);
  narrow.max_pb_aspect_ratio = 2;
  CHECK(!s::sdp_partition_allowed(narrow, 0, 0, p::BlockSize::k64x16,
                                  s::PartitionType::kNone, s::TreeType::kSharedPart,
                                  true, false, s::RegionType::kMixedRegion,
                                  plain));
  p::FrameGeometry wide = geom(128, 128, 64);
  wide.max_pb_aspect_ratio = 8;
  CHECK(s::sdp_partition_allowed(wide, 0, 0, p::BlockSize::k64x16,
                                 s::PartitionType::kNone, s::TreeType::kSharedPart,
                                 true, false, s::RegionType::kMixedRegion,
                                 plain));

  // Mixed inter region rejects 4x4 subsizes; intra region allows them.
  s::Tools inter = inter_tools(true);
  // HORZ of 4x8 yields a 4x4 subsize: rejected for MIXED_REGION inter only.
  CHECK(!s::sdp_partition_allowed(g, 0, 0, p::BlockSize::k4x8,
                                  s::PartitionType::kHorz, s::TreeType::kSharedPart,
                                  true, false, s::RegionType::kMixedRegion,
                                  inter));
  CHECK(s::sdp_partition_allowed(g, 0, 0, p::BlockSize::k4x8,
                                 s::PartitionType::kHorz, s::TreeType::kSharedPart,
                                 true, false, s::RegionType::kIntraRegion,
                                 inter));
}

// ---------------------------------------------------------------------------
// Chroma bookkeeping
// ---------------------------------------------------------------------------

static void test_chroma_offset() {
  const p::FrameGeometry g = geom(128, 128, 64);
  CHECK(s::sdp_chroma_offset_for_subsize(g, p::BlockSize::k4x4));
  CHECK(s::sdp_chroma_offset_for_subsize(g, p::BlockSize::k4x8));
  CHECK(s::sdp_chroma_offset_for_subsize(g, p::BlockSize::k8x4));
  CHECK(!s::sdp_chroma_offset_for_subsize(g, p::BlockSize::k8x8));
  CHECK(!s::sdp_chroma_offset_for_subsize(g, p::BlockSize::k16x16));

  CHECK(s::sdp_chroma_offset_for_partition(g, s::PartitionType::kHorz,
                                           p::BlockSize::k8x8));
  CHECK(!s::sdp_chroma_offset_for_partition(g, s::PartitionType::kHorz,
                                            p::BlockSize::k16x16));
  // HORZ_3@8x32 has the explicit middle-chroma special case.
  CHECK(!s::sdp_chroma_offset_for_partition(g, s::PartitionType::kHorz3,
                                            p::BlockSize::k8x32));

  s::ChromaBlock inherited;
  inherited.mi_row = 0;
  inherited.mi_col = 0;
  inherited.size = p::BlockSize::k16x16;
  const s::ChromaState row =
      s::sdp_resolve_chroma(g, 8, 8, p::BlockSize::k16x16,
                            s::PartitionType::kNone, true, false, inherited);
  CHECK(!row.chroma_offset);
  CHECK_EQ(row.block.mi_row, 8);
  CHECK_EQ(row.block.mi_col, 8);
}

// ---------------------------------------------------------------------------
// Decisions: SDP structure and TreeType switches
// ---------------------------------------------------------------------------

static void test_chroma_offset_tree() {
  // SHARED tree 64x64 -> HORZ/VERT down to 32x8 -> HORZ(32x4).  The 32x4
  // subsize triggers chromaOffset, so only the last 32x4 child codes chroma,
  // and it owns the parent's 32x8 ChromaMi block.
  const p::FrameGeometry g = geom(64, 64, 64);
  s::Tools tools;
  tools.frame_is_intra = true;  // SDP off: plain SHARED tree
  s::DecisionOptions options;
  const auto costs = [](const s::NodeInfo& info) -> s::NodeCosts {
    s::NodeCosts c;
    c.leaf = 0.0;
    c.none_signal = 0.0;
    c.split_signal = s::kInfiniteCost;
    c.horz_signal = s::kInfiniteCost;
    c.vert_signal = s::kInfiniteCost;
    c.horz3_signal = s::kInfiniteCost;
    c.vert3_signal = s::kInfiniteCost;
    c.horz4a_signal = s::kInfiniteCost;
    c.horz4b_signal = s::kInfiniteCost;
    c.vert4a_signal = s::kInfiniteCost;
    c.vert4b_signal = s::kInfiniteCost;
    if (info.size == p::BlockSize::k64x64) {
      c.leaf = s::kInfiniteCost;
      c.horz_signal = 0.0;
    } else if (info.size == p::BlockSize::k64x32) {
      c.leaf = s::kInfiniteCost;
      c.vert_signal = 0.0;
    } else if (info.size == p::BlockSize::k32x32) {
      c.leaf = s::kInfiniteCost;
      c.horz_signal = 0.0;
    } else if (info.size == p::BlockSize::k32x16) {
      c.leaf = s::kInfiniteCost;
      c.horz_signal = 0.0;
    } else if (info.size == p::BlockSize::k32x8) {
      c.leaf = s::kInfiniteCost;
      c.horz_signal = 0.0;
    }
    return c;
  };
  const s::FrameDecision frame = s::decide_frame(g, tools, costs, options);
  if (frame.status != s::DecisionStatus::kOk) {
    std::cerr << "chroma offset tree: " << frame.error << '\n';
  }
  CHECK_EQ(static_cast<int>(frame.status),
           static_cast<int>(s::DecisionStatus::kOk));
  std::vector<s::LeafBlock> leaves;
  s::collect_leaves(frame.superblocks[0], leaves);
  int offset_owners = 0;
  int first_children = 0;
  for (const s::LeafBlock& leaf : leaves) {
    if (leaf.chroma_offset) {
      ++first_children;
      if (leaf.codes_chroma) {
        ++offset_owners;
        CHECK_EQ(static_cast<int>(leaf.chroma.size),
                 static_cast<int>(p::BlockSize::k32x8));
        // The ChromaMi block starts at the parent 32x8 origin, so the owning
        // 32x4 child is offset inside it.
        CHECK(leaf.mi_row >= leaf.chroma.mi_row);
        CHECK(leaf.mi_row < leaf.chroma.mi_row + 8);
        CHECK_EQ(leaf.mi_col, leaf.chroma.mi_col);
      }
    }
  }
  CHECK(offset_owners > 0);
  CHECK(first_children > offset_owners);  // the no-chroma sibling exists
  s::MiSizeGrid luma(g);
  s::MiSizeGrid chroma(g);
  s::build_plane_grids(g, frame.superblocks[0], luma, chroma);
  bool found = false;
  for (int r = 0; r < p::mi_rows(g); ++r) {
    for (int c = 0; c < p::mi_cols(g); ++c) {
      if (chroma.at(r, c) == p::BlockSize::k32x8) found = true;
    }
  }
  CHECK(found);

  // Plan/decode round trip for this chroma-offset tree.
  s::MiSizeGrid pl(g);
  s::MiSizeGrid pc(g);
  s::MiSizeGrid dl(g);
  s::MiSizeGrid dc(g);
  const s::SuperblockPlan plan =
      s::plan_superblock_symbols(g, tools, frame.superblocks[0], pl, pc);
  CHECK(plan.ok);
  s::Node decoded;
  std::string error;
  CHECK(s::decode_superblock_symbols(g, tools, 0, 0, plan.symbols, dl, dc,
                                     decoded, nullptr, error));
  CHECK_STR(s::tree_signature(decoded),
            s::tree_signature(frame.superblocks[0]));
}

static void test_sdp_all_none() {
  const p::FrameGeometry g = geom(64, 64, 64);
  const s::Tools tools = intra_tools();
  s::DecisionOptions options;
  const s::PartitionDecision sb = s::decide_superblock(
      g, tools, 0, 0, none_costs, options);
  CHECK_EQ(static_cast<int>(sb.status),
           static_cast<int>(s::DecisionStatus::kOk));
  CHECK(is_sdp_root(sb.root));
  CHECK_EQ(sb.root.children.size(), 2);
  if (sb.root.children.size() == 2) {
    const s::Node& luma = sb.root.children[0];
    const s::Node& chroma = sb.root.children[1];
    CHECK_EQ(static_cast<int>(luma.tree_type),
             static_cast<int>(s::TreeType::kLumaPart));
    CHECK_EQ(static_cast<int>(chroma.tree_type),
             static_cast<int>(s::TreeType::kChromaPart));
    CHECK(luma.chroma_follows_luma);
    CHECK(chroma.chroma_partition_known);
    CHECK_EQ(static_cast<int>(chroma.implied_luma_partition),
             static_cast<int>(s::PartitionType::kNone));
    CHECK(chroma.cfl_allowed_in_sdp);
    CHECK_EQ(static_cast<int>(luma.partition),
             static_cast<int>(s::PartitionType::kNone));
    CHECK_EQ(static_cast<int>(chroma.partition),
             static_cast<int>(s::PartitionType::kNone));
  }

  const s::FrameDecision frame = s::decide_frame(g, tools, none_costs, options);
  CHECK_EQ(static_cast<int>(frame.status),
           static_cast<int>(s::DecisionStatus::kOk));
  const std::vector<s::TreeSwitch> events = s::tree_type_events(frame);
  CHECK_EQ(events.size(), 3);
  if (events.size() == 3) {
    CHECK_EQ(static_cast<int>(events[0].kind),
             static_cast<int>(s::SwitchKind::kSdpIntraLumaStart));
    CHECK_EQ(static_cast<int>(events[1].kind),
             static_cast<int>(s::SwitchKind::kSdpIntraChromaStart));
    CHECK_EQ(static_cast<int>(events[2].kind),
             static_cast<int>(s::SwitchKind::kSdpIntraEnd));
    CHECK_EQ(events[0].mi_row, 0);
    CHECK_EQ(events[0].mi_col, 0);
  }
}

static void test_chroma_follows_luma() {
  // Root HORZ, children VERT -> ChromaFollowsLuma stays 1.
  {
    const p::FrameGeometry g = geom(64, 64, 64);
    s::DecisionOptions options;
    const s::PartitionDecision sb = s::decide_superblock(
        g, intra_tools(), 0, 0, horz_then_vert_costs, options);
    CHECK_EQ(static_cast<int>(sb.status),
             static_cast<int>(s::DecisionStatus::kOk));
    const s::Node* luma = find_sdp_child(sb.root, s::TreeType::kLumaPart);
    const s::Node* chroma = find_sdp_child(sb.root, s::TreeType::kChromaPart);
    CHECK(luma != nullptr);
    CHECK(chroma != nullptr);
    if (luma != nullptr) {
      CHECK_EQ(static_cast<int>(luma->partition),
               static_cast<int>(s::PartitionType::kHorz));
      CHECK(luma->chroma_follows_luma);
    }
    if (chroma != nullptr) {
      CHECK(chroma->chroma_partition_known);
      CHECK_EQ(static_cast<int>(chroma->partition),
               static_cast<int>(s::PartitionType::kHorz));
    }
  }
  // Root HORZ, children HORZ -> ChromaFollowsLuma clears to 0.
  {
    const p::FrameGeometry g = geom(64, 64, 64);
    s::DecisionOptions options;
    const s::PartitionDecision sb = s::decide_superblock(
        g, intra_tools(), 0, 0, horz_then_horz_costs, options);
    CHECK_EQ(static_cast<int>(sb.status),
             static_cast<int>(s::DecisionStatus::kOk));
    const s::Node* luma = find_sdp_child(sb.root, s::TreeType::kLumaPart);
    const s::Node* chroma = find_sdp_child(sb.root, s::TreeType::kChromaPart);
    CHECK(luma != nullptr);
    CHECK(chroma != nullptr);
    if (luma != nullptr) {
      CHECK_EQ(static_cast<int>(luma->partition),
               static_cast<int>(s::PartitionType::kHorz));
      CHECK(!luma->chroma_follows_luma);
    }
    if (chroma != nullptr) {
      CHECK(!chroma->chroma_partition_known);
    }
  }
  // Direct predicate spot checks.
  CHECK(s::chroma_follows_luma(
      s::PartitionType::kNone, {}));
  CHECK(s::chroma_follows_luma(
      s::PartitionType::kHorz,
      {s::PartitionType::kVert, s::PartitionType::kVert}));
  CHECK(!s::chroma_follows_luma(
      s::PartitionType::kHorz,
      {s::PartitionType::kVert, s::PartitionType::kNone}));
  CHECK(!s::chroma_follows_luma(
      s::PartitionType::kHorz,
      {s::PartitionType::kHorz, s::PartitionType::kVert}));
  CHECK(!s::chroma_follows_luma(
      s::PartitionType::kHorz4A,
      {s::PartitionType::kVert, s::PartitionType::kVert}));
  CHECK(s::cfl_allowed_in_sdp(s::PartitionType::kHorz, true,
                              s::PartitionType::kVert));
  CHECK(s::cfl_allowed_in_sdp(s::PartitionType::kHorz, false,
                              s::PartitionType::kHorz));
  CHECK(!s::cfl_allowed_in_sdp(s::PartitionType::kHorz, false,
                               s::PartitionType::kVert));
  CHECK(s::cfl_allowed_in_sdp(s::PartitionType::kHorz4A, false,
                              s::PartitionType::kHorz4B));
}

static void test_tree_divergence() {
  // Luma root HORZ with HORZ children clears ChromaFollowsLuma; the chroma
  // root is then free and this cost model picks VERT, so the two trees
  // genuinely diverge.
  const p::FrameGeometry g = geom(64, 64, 64);
  s::DecisionOptions options;
  const auto costs = [](const s::NodeInfo& info) -> s::NodeCosts {
    s::NodeCosts c;
    c.leaf = 0.0;
    c.none_signal = 0.0;
    c.split_signal = s::kInfiniteCost;
    c.horz_signal = s::kInfiniteCost;
    c.vert_signal = s::kInfiniteCost;
    c.horz3_signal = s::kInfiniteCost;
    c.vert3_signal = s::kInfiniteCost;
    c.horz4a_signal = s::kInfiniteCost;
    c.horz4b_signal = s::kInfiniteCost;
    c.vert4a_signal = s::kInfiniteCost;
    c.vert4b_signal = s::kInfiniteCost;
    if (info.tree_type == s::TreeType::kLumaPart &&
        (info.size == p::BlockSize::k64x64 ||
         info.size == p::BlockSize::k64x32)) {
      c.leaf = s::kInfiniteCost;
      c.horz_signal = 0.0;
    } else if (info.tree_type == s::TreeType::kChromaPart &&
               info.size == p::BlockSize::k64x64) {
      c.leaf = s::kInfiniteCost;
      c.vert_signal = 0.0;
    }
    return c;
  };
  const s::PartitionDecision sb =
      s::decide_superblock(g, intra_tools(), 0, 0, costs, options);
  CHECK_EQ(static_cast<int>(sb.status),
           static_cast<int>(s::DecisionStatus::kOk));
  const s::Node* luma = find_sdp_child(sb.root, s::TreeType::kLumaPart);
  const s::Node* chroma = find_sdp_child(sb.root, s::TreeType::kChromaPart);
  CHECK(luma != nullptr);
  CHECK(chroma != nullptr);
  if (luma != nullptr && chroma != nullptr) {
    CHECK_EQ(static_cast<int>(luma->partition),
             static_cast<int>(s::PartitionType::kHorz));
    CHECK(!luma->chroma_follows_luma);
    CHECK(!chroma->chroma_partition_known);
    CHECK_EQ(static_cast<int>(chroma->partition),
             static_cast<int>(s::PartitionType::kVert));
    CHECK(luma->partition != chroma->partition);
  }
}

static void test_128_superblock_events() {
  const p::FrameGeometry g = geom(128, 128, 128);
  s::DecisionOptions options;
  // SPLIT is only legal at 128x128; a generic cost model may or may not pick
  // it.  Force it with infinite rect costs.
  const auto split_costs = [](const s::NodeInfo& info) {
    s::NodeCosts costs = none_costs(info);
    if (info.size == p::BlockSize::k128x128) {
      costs.split_signal = 0.0;
      costs.leaf = s::kInfiniteCost;
    }
    return costs;
  };
  const s::PartitionDecision sb =
      s::decide_superblock(g, intra_tools(), 0, 0, split_costs, options);
  CHECK_EQ(static_cast<int>(sb.status),
           static_cast<int>(s::DecisionStatus::kOk));
  CHECK_EQ(static_cast<int>(sb.root.partition),
           static_cast<int>(s::PartitionType::kSplit));
  CHECK_EQ(sb.root.children.size(), 4);
  for (const s::Node& child : sb.root.children) {
    CHECK(is_sdp_root(child));
  }
  const s::FrameDecision frame =
      s::decide_frame(g, intra_tools(), split_costs, options);
  CHECK_EQ(static_cast<int>(frame.status),
           static_cast<int>(s::DecisionStatus::kOk));
  const std::vector<s::TreeSwitch> events = s::tree_type_events(frame);
  CHECK_EQ(events.size(), 12);  // 4 SDP roots x 3 switches
}

static void test_partial_superblocks() {
  struct Case {
    int w;
    int h;
    int sb;
  };
  const Case cases[] = {
      {64, 64, 64},    {640, 360, 64},  {640, 360, 128},
      {128, 128, 128}, {520, 128, 128}, {1080, 1920, 128},
      {136, 88, 64},   {518, 322, 128}, {8, 8, 64},
      {16, 8, 64},     {264, 200, 64},  {200, 264, 128},
  };
  for (const Case& item : cases) {
    const p::FrameGeometry g = geom(item.w, item.h, item.sb);
    s::DecisionOptions options = capped_options(item.w, item.h);
    s::Tools tools = intra_tools(true, true);
    const s::FrameDecision frame = s::decide_frame(
        g, tools,
        [](const s::NodeInfo& info) { return generic_costs(info, 4242); },
        options);
    if (frame.status != s::DecisionStatus::kOk) {
      std::cerr << "case " << item.w << 'x' << item.h << " sb " << item.sb
                << " failed: " << frame.error << '\n';
    }
    CHECK_EQ(static_cast<int>(frame.status),
             static_cast<int>(s::DecisionStatus::kOk));
    const int sb4 = p::sb_size_mi(g);
    int expected = 0;
    for (int r = 0; r < p::mi_rows(g); r += sb4) {
      for (int c = 0; c < p::mi_cols(g); c += sb4) {
        if (r < p::mi_rows(g) && c < p::mi_cols(g)) ++expected;
      }
    }
    CHECK_EQ(frame.superblocks.size(), expected);
    std::string error;
    CHECK(s::validate_frame(g, tools, frame, error));
    // Determinism.
    const s::FrameDecision again = s::decide_frame(
        g, tools,
        [](const s::NodeInfo& info) { return generic_costs(info, 4242); },
        options);
    CHECK_EQ(static_cast<int>(again.status),
             static_cast<int>(s::DecisionStatus::kOk));
    if (again.status == s::DecisionStatus::kOk) {
      CHECK_EQ(again.superblocks.size(), frame.superblocks.size());
      for (std::size_t i = 0; i < frame.superblocks.size(); ++i) {
        CHECK_STR(s::tree_signature(again.superblocks[i]),
                  s::tree_signature(frame.superblocks[i]));
      }
    }
  }
}

static void test_partial_64_superblock_counts() {
  const p::FrameGeometry g = geom(640, 360, 64);
  // 10x6 superblocks; the bottom row is partial but still in frame.
  s::DecisionOptions options;
  const s::FrameDecision frame =
      s::decide_frame(g, intra_tools(), none_costs, options);
  CHECK_EQ(static_cast<int>(frame.status),
           static_cast<int>(s::DecisionStatus::kOk));
  CHECK_EQ(frame.superblocks.size(), 60);
  const std::vector<s::TreeSwitch> events = s::tree_type_events(frame);
  CHECK_EQ(events.size(), 60 * 3);
  for (const s::Node& sb : frame.superblocks) CHECK(is_sdp_root(sb));
}

// ---------------------------------------------------------------------------
// Contexts
// ---------------------------------------------------------------------------

static void test_contexts() {
  const p::FrameGeometry g = geom(256, 256, 64);
  s::Node node;
  node.mi_row = 0;
  node.mi_col = 0;
  node.size = p::BlockSize::k64x64;
  node.tree_type = s::TreeType::kLumaPart;
  node.partition = s::PartitionType::kHorz;
  s::MiSizeGrid empty(g);
  const s::SymbolContexts c0 = s::node_contexts(g, node, empty);
  // 64x64: Partition_Size_Adjust = 3, Partition_Size_Adjust_Rect_Type = 6,
  // no neighbours -> do_split/do_ext ctx 12 and rect_type ctx 24.
  CHECK_EQ(c0.do_split, 12);
  CHECK_EQ(c0.rect_type, 24);
  CHECK_EQ(c0.do_ext, 12);
  // vary neighbours
  s::MiSizeGrid filled(g);
  filled.fill_block(0, 0, p::BlockSize::k64x64);
  filled.fill_block(0, 16, p::BlockSize::k32x32);
  filled.fill_block(16, 0, p::BlockSize::k32x32);
  // Left neighbour 32x32 (height log2 2 < 4) sets ctx1.
  node.mi_col = 16;
  node.mi_row = 16;
  s::MiSizeGrid left(g);
  left.fill_block(16, 8, p::BlockSize::k32x32);
  const s::SymbolContexts c1 = s::node_contexts(g, node, left);
  CHECK_EQ(c1.do_split, 12 + 2);
  CHECK_EQ(c1.rect_type, 24 + 2);

  // do_ext probes the midpoint row.
  s::Node wide;
  wide.size = p::BlockSize::k16x16;
  wide.tree_type = s::TreeType::kSharedPart;
  wide.partition = s::PartitionType::kHorz3;
  wide.mi_row = 0;
  wide.mi_col = 16;
  s::MiSizeGrid mid(g);
  mid.fill_block(0, 16, p::BlockSize::k16x16);
  mid.fill_block(8, 16, p::BlockSize::k8x8);  // r + h4/2
  const s::SymbolContexts cm = s::node_contexts(g, wide, mid);
  CHECK(cm.do_ext >= 0);

  CHECK_EQ(s::region_type_context(p::BlockSize::k8x8), 0);
  CHECK_EQ(s::region_type_context(p::BlockSize::k16x16), 1);
  CHECK_EQ(s::region_type_context(p::BlockSize::k32x32), 2);
  CHECK_EQ(s::region_type_context(p::BlockSize::k64x64), 3);
  CHECK_EQ(s::plane_start(s::TreeType::kSharedPart), 0);
  CHECK_EQ(s::plane_start(s::TreeType::kLumaPart), 0);
  CHECK_EQ(s::plane_start(s::TreeType::kChromaPart), 1);
}

// ---------------------------------------------------------------------------
// Symbol plan + independent decoder
// ---------------------------------------------------------------------------

static void round_trip(const p::FrameGeometry& g, const s::Tools& tools,
                       const s::CostFunction& costs, const char* label) {
  s::DecisionOptions options = capped_options(g.width, g.height);
  const s::FrameDecision frame = s::decide_frame(g, tools, costs, options);
  if (frame.status != s::DecisionStatus::kOk) {
    std::cerr << label << ": decide failed: " << frame.error << '\n';
    ++g_failures;
    return;
  }
  s::MiSizeGrid plan_luma(g);
  s::MiSizeGrid plan_chroma(g);
  s::MiSizeGrid dec_luma(g);
  s::MiSizeGrid dec_chroma(g);
  for (const s::Node& sb : frame.superblocks) {
    const s::SuperblockPlan plan = s::plan_superblock_symbols(
        g, tools, sb, plan_luma, plan_chroma);
    if (!plan.ok) {
      std::cerr << label << ": plan failed: " << plan.error << '\n';
      ++g_failures;
      return;
    }
    for (const s::Symbol& symbol : plan.symbols) {
      CHECK(symbol_alphabet_ok(symbol));
    }
    s::Node decoded;
    std::vector<s::TreeSwitch> decoded_switches;
    std::string error;
    if (!s::decode_superblock_symbols(g, tools, sb.mi_row, sb.mi_col,
                                      plan.symbols, dec_luma, dec_chroma,
                                      decoded, &decoded_switches, error)) {
      std::cerr << label << ": decode failed: " << error << '\n';
      ++g_failures;
      return;
    }
    CHECK_STR(s::tree_signature(decoded), s::tree_signature(sb));
    CHECK_EQ(decoded_switches.size(), plan.switches.size());
    for (std::size_t i = 0;
         i < std::min(decoded_switches.size(), plan.switches.size()); ++i) {
      CHECK_EQ(static_cast<int>(decoded_switches[i].kind),
               static_cast<int>(plan.switches[i].kind));
      CHECK_EQ(decoded_switches[i].mi_row, plan.switches[i].mi_row);
      CHECK_EQ(decoded_switches[i].mi_col, plan.switches[i].mi_col);
    }
  }
}

static void test_extended_symbol_golden() {
  // HORZ_4A at 64x64 (SHARED, SDP off): do_split, rect_type, do_ext,
  // do_uneven_4way, then the L(1) type literal.
  {
    const p::FrameGeometry g = geom(64, 64, 64);
    s::Tools tools;
    tools.enable_ext_partitions = true;
    tools.enable_uneven_4way_partitions = true;
    tools.frame_is_intra = true;
    s::DecisionOptions options;
    const auto costs = [](const s::NodeInfo& info) {
      s::NodeCosts c;
      c.leaf = info.size == p::BlockSize::k64x64 ? s::kInfiniteCost : 0.0;
      c.none_signal = 0.0;
      c.split_signal = s::kInfiniteCost;
      c.horz_signal = s::kInfiniteCost;
      c.vert_signal = s::kInfiniteCost;
      c.horz3_signal = s::kInfiniteCost;
      c.vert3_signal = s::kInfiniteCost;
      c.horz4a_signal = info.size == p::BlockSize::k64x64 ? 0.0
                                                          : s::kInfiniteCost;
      c.horz4b_signal = s::kInfiniteCost;
      c.vert4a_signal = s::kInfiniteCost;
      c.vert4b_signal = s::kInfiniteCost;
      return c;
    };
    const s::FrameDecision frame = s::decide_frame(g, tools, costs, options);
    CHECK_EQ(static_cast<int>(frame.status),
             static_cast<int>(s::DecisionStatus::kOk));
    s::MiSizeGrid luma(g);
    s::MiSizeGrid chroma(g);
    const s::SuperblockPlan plan = s::plan_superblock_symbols(
        g, tools, frame.superblocks[0], luma, chroma);
    CHECK(plan.ok);
    CHECK(plan.symbols.size() >= 5);
    if (plan.symbols.size() >= 5) {
      CHECK_EQ(static_cast<int>(plan.symbols[0].kind),
               static_cast<int>(s::SymbolKind::kDoSplit));
      CHECK_EQ(plan.symbols[0].value, 1);
      CHECK_EQ(static_cast<int>(plan.symbols[1].kind),
               static_cast<int>(s::SymbolKind::kRectType));
      CHECK_EQ(plan.symbols[1].value, 0);
      CHECK_EQ(static_cast<int>(plan.symbols[2].kind),
               static_cast<int>(s::SymbolKind::kDoExtPartition));
      CHECK_EQ(plan.symbols[2].value, 1);
      CHECK_EQ(static_cast<int>(plan.symbols[3].kind),
               static_cast<int>(s::SymbolKind::kDoUneven4wayPartition));
      CHECK_EQ(plan.symbols[3].value, 1);
      CHECK_EQ(static_cast<int>(plan.symbols[4].kind),
               static_cast<int>(s::SymbolKind::kUneven4wayType));
      CHECK_EQ(plan.symbols[4].value, 0);
      CHECK(plan.symbols[4].literal);
      CHECK_EQ(s::symbol_alphabet(plan.symbols[4].kind), 0);
    }
    s::MiSizeGrid dl(g);
    s::MiSizeGrid dc(g);
    s::Node decoded;
    std::string error;
    CHECK(s::decode_superblock_symbols(g, tools, 0, 0, plan.symbols, dl, dc,
                                       decoded, nullptr, error));
    CHECK_STR(s::tree_signature(decoded), s::tree_signature(frame.superblocks[0]));
  }
  // HORZ_3 with uneven 4-way disabled: do_ext is coded, do_uneven is implied 0.
  {
    const p::FrameGeometry g = geom(64, 64, 64);
    s::Tools tools;
    tools.enable_ext_partitions = true;
    tools.frame_is_intra = true;
    s::DecisionOptions options;
    const auto costs = [](const s::NodeInfo& info) {
      s::NodeCosts c;
      c.leaf = info.size == p::BlockSize::k64x64 ? s::kInfiniteCost : 0.0;
      c.none_signal = 0.0;
      c.split_signal = s::kInfiniteCost;
      c.horz_signal = s::kInfiniteCost;
      c.vert_signal = s::kInfiniteCost;
      c.horz3_signal = info.size == p::BlockSize::k64x64 ? 0.0
                                                         : s::kInfiniteCost;
      c.vert3_signal = s::kInfiniteCost;
      c.horz4a_signal = s::kInfiniteCost;
      c.horz4b_signal = s::kInfiniteCost;
      c.vert4a_signal = s::kInfiniteCost;
      c.vert4b_signal = s::kInfiniteCost;
      return c;
    };
    const s::FrameDecision frame = s::decide_frame(g, tools, costs, options);
    CHECK_EQ(static_cast<int>(frame.status),
             static_cast<int>(s::DecisionStatus::kOk));
    s::MiSizeGrid luma(g);
    s::MiSizeGrid chroma(g);
    const s::SuperblockPlan plan = s::plan_superblock_symbols(
        g, tools, frame.superblocks[0], luma, chroma);
    CHECK(plan.ok);
    CHECK(plan.symbols.size() >= 3);
    if (plan.symbols.size() >= 3) {
      CHECK_EQ(static_cast<int>(plan.symbols[0].kind),
               static_cast<int>(s::SymbolKind::kDoSplit));
      CHECK_EQ(static_cast<int>(plan.symbols[1].kind),
               static_cast<int>(s::SymbolKind::kRectType));
      CHECK_EQ(static_cast<int>(plan.symbols[2].kind),
               static_cast<int>(s::SymbolKind::kDoExtPartition));
      CHECK_EQ(plan.symbols[2].value, 1);
    }
    for (const s::Symbol& symbol : plan.symbols) {
      CHECK(symbol.kind != s::SymbolKind::kUneven4wayType);
    }
  }
  // SPLIT at 128x128: do_split=1 then do_square_split=1.
  {
    const p::FrameGeometry g = geom(128, 128, 128);
    s::Tools tools;
    tools.frame_is_intra = true;
    s::DecisionOptions options;
    const auto costs = [](const s::NodeInfo& info) {
      s::NodeCosts c;
      c.leaf = info.size == p::BlockSize::k128x128 ? s::kInfiniteCost : 0.0;
      c.none_signal = 0.0;
      c.split_signal =
          info.size == p::BlockSize::k128x128 ? 0.0 : s::kInfiniteCost;
      c.horz_signal = s::kInfiniteCost;
      c.vert_signal = s::kInfiniteCost;
      c.horz3_signal = s::kInfiniteCost;
      c.vert3_signal = s::kInfiniteCost;
      c.horz4a_signal = s::kInfiniteCost;
      c.horz4b_signal = s::kInfiniteCost;
      c.vert4a_signal = s::kInfiniteCost;
      c.vert4b_signal = s::kInfiniteCost;
      return c;
    };
    const s::FrameDecision frame = s::decide_frame(g, tools, costs, options);
    CHECK_EQ(static_cast<int>(frame.status),
             static_cast<int>(s::DecisionStatus::kOk));
    s::MiSizeGrid luma(g);
    s::MiSizeGrid chroma(g);
    const s::SuperblockPlan plan = s::plan_superblock_symbols(
        g, tools, frame.superblocks[0], luma, chroma);
    CHECK(plan.ok);
    CHECK(plan.symbols.size() >= 4);
    if (plan.symbols.size() >= 2) {
      CHECK_EQ(static_cast<int>(plan.symbols[0].kind),
               static_cast<int>(s::SymbolKind::kDoSplit));
      CHECK_EQ(plan.symbols[0].value, 1);
      CHECK_EQ(static_cast<int>(plan.symbols[1].kind),
               static_cast<int>(s::SymbolKind::kDoSquareSplit));
      CHECK_EQ(plan.symbols[1].value, 1);
    }
    // SPLIT children are 64x64, so each one is an SDP split only when SDP is
    // enabled.  Here it is off; the children keep coding SHARED.
    for (const s::Node& child : frame.superblocks[0].children) {
      CHECK(!child.sdp_split);
      CHECK_EQ(static_cast<int>(child.tree_type),
               static_cast<int>(s::TreeType::kSharedPart));
    }
  }
}

static void test_symbol_round_trip() {
  struct Case {
    int w;
    int h;
    int sb;
    bool ext;
    bool uneven;
  };
  const Case cases[] = {
      {64, 64, 64, false, false},   {128, 128, 128, false, false},
      {640, 360, 64, false, false}, {640, 360, 128, true, true},
      {136, 88, 64, true, false},   {520, 128, 128, true, true},
      {8, 8, 64, false, false},     {16, 16, 64, true, true},
      {264, 200, 64, true, true},   {200, 264, 128, false, false},
  };
  for (const Case& item : cases) {
    const p::FrameGeometry g = geom(item.w, item.h, item.sb);
    const s::Tools tools = intra_tools(item.ext, item.uneven);
    round_trip(g, tools, [](const s::NodeInfo& info) {
      return generic_costs(info, 12345);
    }, "intra-generic");
  }
  // All-NONE and forced rect trees too.
  round_trip(geom(640, 360, 64), intra_tools(), none_costs, "intra-none");
  round_trip(geom(64, 64, 64), intra_tools(), horz_then_vert_costs,
             "intra-horz-vert");
  round_trip(geom(64, 64, 64), intra_tools(), horz_then_horz_costs,
             "intra-horz-horz");
}

static bool symbol_alphabet_ok(const s::Symbol& symbol) {
  const int alphabet = s::symbol_alphabet(symbol.kind);
  if (symbol.literal) {
    return alphabet == 0 && symbol.value >= 0 && symbol.value <= 1;
  }
  return alphabet == 2 && symbol.value >= 0 && symbol.value <= 1;
}

static void test_golden_plan() {
  const p::FrameGeometry g = geom(64, 64, 64);
  s::DecisionOptions options;
  const s::FrameDecision frame =
      s::decide_frame(g, intra_tools(), none_costs, options);
  CHECK_EQ(static_cast<int>(frame.status),
           static_cast<int>(s::DecisionStatus::kOk));
  s::MiSizeGrid luma(g);
  s::MiSizeGrid chroma(g);
  const s::SuperblockPlan plan = s::plan_superblock_symbols(
      g, intra_tools(), frame.superblocks[0], luma, chroma);
  CHECK(plan.ok);
  // One do_split=0 for the luma root; the chroma root is forced by
  // ChromaPartitionKnown.  Context 12 = Partition_Size_Adjust[64x64] * 4.
  CHECK_EQ(plan.symbols.size(), 1);
  if (plan.symbols.size() == 1) {
    CHECK_EQ(static_cast<int>(plan.symbols[0].kind),
             static_cast<int>(s::SymbolKind::kDoSplit));
    CHECK_EQ(plan.symbols[0].value, 0);
    CHECK_EQ(plan.symbols[0].context, 12);
    CHECK_EQ(plan.symbols[0].plane, 0);
  }
  CHECK_EQ(plan.switches.size(), 3);

  // HORZ at the luma root codes do_split=1 then rect_type=0 (HORZ).
  s::MiSizeGrid luma2(g);
  s::MiSizeGrid chroma2(g);
  const s::FrameDecision frame2 =
      s::decide_frame(g, intra_tools(), horz_then_vert_costs, options);
  CHECK_EQ(static_cast<int>(frame2.status),
           static_cast<int>(s::DecisionStatus::kOk));
  const s::SuperblockPlan plan2 = s::plan_superblock_symbols(
      g, intra_tools(), frame2.superblocks[0], luma2, chroma2);
  CHECK(plan2.ok);
  CHECK(plan2.symbols.size() >= 2);
  if (plan2.symbols.size() >= 2) {
    CHECK_EQ(static_cast<int>(plan2.symbols[0].kind),
             static_cast<int>(s::SymbolKind::kDoSplit));
    CHECK_EQ(plan2.symbols[0].value, 1);
    CHECK_EQ(static_cast<int>(plan2.symbols[1].kind),
             static_cast<int>(s::SymbolKind::kRectType));
    CHECK_EQ(plan2.symbols[1].value, 0);
    CHECK_EQ(plan2.symbols[1].context, 24);
  }

  // Truncated plan must fail to decode.
  if (!plan2.symbols.empty()) {
    std::vector<s::Symbol> short_plan(plan2.symbols.begin(),
                                      plan2.symbols.end() - 1);
    s::MiSizeGrid dl(g);
    s::MiSizeGrid dc(g);
    s::Node decoded;
    std::string error;
    CHECK(!s::decode_superblock_symbols(g, intra_tools(), 0, 0, short_plan,
                                        dl, dc, decoded, nullptr, error));
  }
}

static void test_plane_grids() {
  const p::FrameGeometry g = geom(64, 64, 64);
  s::DecisionOptions options;
  const s::FrameDecision frame =
      s::decide_frame(g, intra_tools(), none_costs, options);
  CHECK_EQ(static_cast<int>(frame.status),
           static_cast<int>(s::DecisionStatus::kOk));
  s::MiSizeGrid luma(g);
  s::MiSizeGrid chroma(g);
  s::build_plane_grids(g, frame.superblocks[0], luma, chroma);
  CHECK_EQ(static_cast<int>(luma.at(0, 0)),
           static_cast<int>(p::BlockSize::k64x64));
  CHECK_EQ(static_cast<int>(chroma.at(0, 0)),
           static_cast<int>(p::BlockSize::k64x64));
  CHECK_EQ(static_cast<int>(luma.at(15, 15)),
           static_cast<int>(p::BlockSize::k64x64));
  CHECK_EQ(static_cast<int>(chroma.at(15, 15)),
           static_cast<int>(p::BlockSize::k64x64));
}

// ---------------------------------------------------------------------------
// Validation rejects malformed trees
// ---------------------------------------------------------------------------

static void test_validation_negatives() {
  const p::FrameGeometry g = geom(64, 64, 64);
  s::DecisionOptions options;
  const s::PartitionDecision good = s::decide_superblock(
      g, intra_tools(), 0, 0, none_costs, options);
  CHECK_EQ(static_cast<int>(good.status),
           static_cast<int>(s::DecisionStatus::kOk));
  std::string error;
  CHECK(s::validate_tree(g, intra_tools(), good.root, error));

  // Claim no SDP split.
  {
    s::Node bad = good.root;
    bad.sdp_split = false;
    bad.partition = s::PartitionType::kNone;
    CHECK(!s::validate_tree(g, intra_tools(), bad, error));
  }
  // Wrong TreeType on the chroma root.
  {
    s::Node bad = good.root;
    bad.children[1].tree_type = s::TreeType::kLumaPart;
    CHECK(!s::validate_tree(g, intra_tools(), bad, error));
  }
  // Stale ChromaFollowsLuma.
  {
    s::Node bad = good.root;
    bad.children[0].chroma_follows_luma = false;
    CHECK(!s::validate_tree(g, intra_tools(), bad, error));
  }
  // Stale ChromaPartitionKnown.
  {
    s::Node bad = good.root;
    bad.children[1].chroma_partition_known = false;
    CHECK(!s::validate_tree(g, intra_tools(), bad, error));
  }
  // Forced partition changed: the chroma root of the all-NONE split is forced
  // to NONE by ChromaPartitionKnown.
  {
    s::Node bad = good.root;
    bad.children[1].partition = s::PartitionType::kVert;
    CHECK(!s::validate_tree(g, intra_tools(), bad, error));
  }
  // Missing child.
  {
    s::Node bad = good.root;
    bad.children.pop_back();
    CHECK(!s::validate_tree(g, intra_tools(), bad, error));
  }
  // Stale allowed set.
  {
    s::Node bad = good.root;
    bad.children[0].allowed.allowed[static_cast<int>(
        s::PartitionType::kSplit)] = true;
    bad.children[0].allowed.count += 1;
    CHECK(!s::validate_tree(g, intra_tools(), bad, error));
  }
  // Invalid cost: negative.
  {
    const auto bad_cost = [](const s::NodeInfo& info) {
      s::NodeCosts costs = none_costs(info);
      costs.leaf = -1.0;
      return costs;
    };
    const s::PartitionDecision e =
        s::decide_superblock(g, intra_tools(), 0, 0, bad_cost, options);
    CHECK_EQ(static_cast<int>(e.status),
             static_cast<int>(s::DecisionStatus::kInvalidCost));
  }
  // Tools validation.
  {
    s::Tools bad = intra_tools();
    bad.enable_extended_sdp = true;  // requires enable_sdp (ok) but not single
    CHECK(!s::validate_tools(bad, error));
    s::Tools bad2;
    bad2.enable_uneven_4way_partitions = true;
    CHECK(!s::validate_tools(bad2, error));
    s::Tools bad3;
    bad3.enable_sdp = true;
    p::FrameGeometry mono = g;
    mono.has_chroma = false;
    const s::PartitionDecision e =
        s::decide_superblock(mono, bad3, 0, 0, none_costs, options);
    CHECK_EQ(static_cast<int>(e.status),
             static_cast<int>(s::DecisionStatus::kInvalidTools));
  }
}

// ---------------------------------------------------------------------------
// Extended SDP (inter frames)
// ---------------------------------------------------------------------------

static void test_inter_without_extended_sdp() {
  // FrameIsIntra == 0 disables the intra SDP trigger even when enable_sdp is
  // set, so a 64x64 root stays a single SHARED tree.
  const p::FrameGeometry g = geom(64, 64, 64);
  s::Tools tools;
  tools.enable_sdp = true;
  tools.frame_is_intra = false;
  tools.single_picture_header_flag = false;
  s::DecisionOptions options;
  const s::PartitionDecision sb =
      s::decide_superblock(g, tools, 0, 0, none_costs, options);
  CHECK_EQ(static_cast<int>(sb.status),
           static_cast<int>(s::DecisionStatus::kOk));
  CHECK(!sb.root.sdp_split);
  CHECK_EQ(static_cast<int>(sb.root.tree_type),
           static_cast<int>(s::TreeType::kSharedPart));
  CHECK_EQ(static_cast<int>(sb.root.partition),
           static_cast<int>(s::PartitionType::kNone));
  CHECK(!sb.root.region_type_coded);  // size == SbSize
  CHECK_EQ(sb.root.children.size(), 0);

  // With enable_extended_sdp on, a 64x64 superblock still has no eligible
  // node (bSize != SbSize never holds for a 64 SB).
  tools.enable_extended_sdp = true;
  const s::PartitionDecision sb2 =
      s::decide_superblock(g, tools, 0, 0, none_costs, options);
  CHECK_EQ(static_cast<int>(sb2.status),
           static_cast<int>(s::DecisionStatus::kOk));
  CHECK(!sb2.root.extended_sdp);
  CHECK(!sb2.root.region_type_coded);
}

static void test_extended_sdp() {
  CHECK(s::is_bsize_allowed_for_extended_sdp(p::BlockSize::k64x64,
                                             s::PartitionType::kHorz));
  CHECK(s::is_bsize_allowed_for_extended_sdp(p::BlockSize::k8x8,
                                             s::PartitionType::kVert));
  CHECK(!s::is_bsize_allowed_for_extended_sdp(p::BlockSize::k64x64,
                                              s::PartitionType::kNone));
  CHECK(!s::is_bsize_allowed_for_extended_sdp(p::BlockSize::k128x128,
                                              s::PartitionType::kHorz));
  CHECK(!s::is_bsize_allowed_for_extended_sdp(p::BlockSize::k4x8,
                                              s::PartitionType::kHorz));
  CHECK(!s::is_bsize_allowed_for_extended_sdp(p::BlockSize::k64x64,
                                              s::PartitionType::kHorz4A));

  const p::FrameGeometry g = geom(128, 128, 128);
  const s::Tools tools = inter_tools(true);
  s::DecisionOptions options;
  // Force SPLIT at 128x128 and HORZ at each 64x64 child, and prefer the
  // INTRA_REGION variant.
  const auto costs = [](const s::NodeInfo& info) -> s::NodeCosts {
    s::NodeCosts c;
    c.leaf = 0.0;
    c.none_signal = 0.0;
    c.split_signal = s::kInfiniteCost;
    c.horz_signal = s::kInfiniteCost;
    c.vert_signal = s::kInfiniteCost;
    c.horz3_signal = s::kInfiniteCost;
    c.vert3_signal = s::kInfiniteCost;
    c.horz4a_signal = s::kInfiniteCost;
    c.horz4b_signal = s::kInfiniteCost;
    c.vert4a_signal = s::kInfiniteCost;
    c.vert4b_signal = s::kInfiniteCost;
    c.region_type[0] = 0.0;
    c.region_type[1] = s::kInfiniteCost;
    if (info.size == p::BlockSize::k128x128) {
      c.split_signal = 0.0;
      c.leaf = s::kInfiniteCost;
    } else if (info.size == p::BlockSize::k64x64 &&
               info.tree_type == s::TreeType::kSharedPart) {
      c.horz_signal = 0.0;
      c.leaf = s::kInfiniteCost;
    } else if (info.size == p::BlockSize::k32x64) {
      c.leaf = 0.0;
    }
    return c;
  };
  const s::PartitionDecision sb =
      s::decide_superblock(g, tools, 0, 0, costs, options);
  if (sb.status != s::DecisionStatus::kOk) {
    std::cerr << "extended SDP decide failed: " << sb.error << '\n';
  }
  CHECK_EQ(static_cast<int>(sb.status),
           static_cast<int>(s::DecisionStatus::kOk));
  CHECK_EQ(static_cast<int>(sb.root.partition),
           static_cast<int>(s::PartitionType::kSplit));
  int extended = 0;
  for (const s::Node& child : sb.root.children) {
    if (child.extended_sdp) {
      ++extended;
      CHECK_EQ(static_cast<int>(child.region_type),
               static_cast<int>(s::RegionType::kIntraRegion));
      CHECK_EQ(static_cast<int>(child.partition),
               static_cast<int>(s::PartitionType::kHorz));
      CHECK(child.children.size() >= 2);
      if (!child.children.empty()) {
        const s::Node& tail = child.children.back();
        CHECK(tail.sdp_chroma_block);
        CHECK(tail.codes_chroma);
        CHECK_EQ(tail.chroma.mi_row, child.mi_row);
        CHECK_EQ(tail.chroma.mi_col, child.mi_col);
      }
    }
  }
  CHECK_EQ(extended, 4);  // 128x128 SPLIT -> four in-frame 64x64 children

  const s::FrameDecision frame = s::decide_frame(g, tools, costs, options);
  CHECK_EQ(static_cast<int>(frame.status),
           static_cast<int>(s::DecisionStatus::kOk));
  const std::vector<s::TreeSwitch> events = s::tree_type_events(frame);
  CHECK(events.size() >= 4 * 3);

  // MIXED_REGION plan: no extended tree, region_type=1.
  const auto mixed_costs = [](const s::NodeInfo& info) -> s::NodeCosts {
    s::NodeCosts c;
    c = s::NodeCosts{};
    c.leaf = 0.0;
    c.none_signal = 0.0;
    c.split_signal = s::kInfiniteCost;
    c.horz_signal = s::kInfiniteCost;
    c.vert_signal = s::kInfiniteCost;
    c.horz3_signal = s::kInfiniteCost;
    c.vert3_signal = s::kInfiniteCost;
    c.horz4a_signal = s::kInfiniteCost;
    c.horz4b_signal = s::kInfiniteCost;
    c.vert4a_signal = s::kInfiniteCost;
    c.vert4b_signal = s::kInfiniteCost;
    c.region_type[0] = s::kInfiniteCost;
    c.region_type[1] = 0.0;
    if (info.size == p::BlockSize::k128x128) {
      c.split_signal = 0.0;
      c.leaf = s::kInfiniteCost;
    } else if (info.size == p::BlockSize::k64x64 &&
               info.tree_type == s::TreeType::kSharedPart) {
      c.horz_signal = 0.0;
      c.leaf = s::kInfiniteCost;
    } else if (info.size == p::BlockSize::k32x64) {
      c.leaf = 0.0;
    }
    return c;
  };
  const s::FrameDecision mixed =
      s::decide_frame(g, tools, mixed_costs, options);
  CHECK_EQ(static_cast<int>(mixed.status),
           static_cast<int>(s::DecisionStatus::kOk));
  if (mixed.status == s::DecisionStatus::kOk) {
    const std::vector<s::TreeSwitch> mixed_events =
        s::tree_type_events(mixed);
    // No extended switches, only the intra SDP ones would appear; this is an
    // inter frame, so enable_sdp intra SDP does not fire.
    CHECK_EQ(mixed_events.size(), 0);
    for (const s::Node& sb2 : mixed.superblocks) {
      for (const s::Node& child : sb2.children) {
        CHECK(!child.extended_sdp);
        CHECK(child.region_type_coded);
        CHECK_EQ(static_cast<int>(child.region_type),
                 static_cast<int>(s::RegionType::kMixedRegion));
      }
    }
  }

  // Round trip the intra-region plan.
  s::MiSizeGrid luma(g);
  s::MiSizeGrid chroma(g);
  s::MiSizeGrid dl(g);
  s::MiSizeGrid dc(g);
  const s::SuperblockPlan plan =
      s::plan_superblock_symbols(g, tools, sb.root, luma, chroma);
  CHECK(plan.ok);
  if (plan.ok) {
    s::Node decoded;
    std::string error;
    CHECK(s::decode_superblock_symbols(g, tools, 0, 0, plan.symbols, dl, dc,
                                       decoded, nullptr, error));
    if (!error.empty()) std::cerr << "extended decode: " << error << '\n';
    CHECK_STR(s::tree_signature(decoded), s::tree_signature(sb.root));
  }
}

// ---------------------------------------------------------------------------
// Rect-only parity with encoder/partition (SDP disabled)
// ---------------------------------------------------------------------------

static void test_rect_only_parity() {
  struct Case {
    int w;
    int h;
    int sb;
  };
  const Case cases[] = {
      {64, 64, 64},  {640, 360, 64}, {128, 128, 128},
      {520, 128, 128}, {136, 88, 64}, {8, 8, 64},
  };
  for (const Case& item : cases) {
    const p::FrameGeometry g = geom(item.w, item.h, item.sb);
    s::Tools tools;
    tools.enable_sdp = false;
    tools.frame_is_intra = true;
    s::DecisionOptions options;
    p::DecisionOptions poptions;
    poptions.min_block_size_px = options.min_block_size_px;
    poptions.max_depth = options.max_depth;
    poptions.allow_rect = true;
    poptions.allow_split = true;

    const auto base_costs = [](const p::PartitionNodeInfo& info) {
      p::NodeCosts c;
      const int r = info.mi_row;
      const int ccol = info.mi_col;
      const int size = static_cast<int>(info.size);
      c.leaf = pseudo(999, r, ccol, size, 0, 0) * 100.0;
      c.none_signal = pseudo(999, r, ccol, size, 0, 1) * 1.0;
      c.split_signal = pseudo(999, r, ccol, size, 0, 2) * 1.0;
      c.horz_signal = pseudo(999, r, ccol, size, 0, 3) * 1.0;
      c.vert_signal = pseudo(999, r, ccol, size, 0, 4) * 1.0;
      return c;
    };
    const auto our_costs = [](const s::NodeInfo& info) {
      s::NodeCosts c;
      const int r = info.mi_row;
      const int ccol = info.mi_col;
      const int size = static_cast<int>(info.size);
      c.leaf = pseudo(999, r, ccol, size, 0, 0) * 100.0;
      c.none_signal = pseudo(999, r, ccol, size, 0, 1) * 1.0;
      c.split_signal = pseudo(999, r, ccol, size, 0, 2) * 1.0;
      c.horz_signal = pseudo(999, r, ccol, size, 0, 3) * 1.0;
      c.vert_signal = pseudo(999, r, ccol, size, 0, 4) * 1.0;
      c.horz3_signal = s::kInfiniteCost;
      c.vert3_signal = s::kInfiniteCost;
      c.horz4a_signal = s::kInfiniteCost;
      c.horz4b_signal = s::kInfiniteCost;
      c.vert4a_signal = s::kInfiniteCost;
      c.vert4b_signal = s::kInfiniteCost;
      return c;
    };

    const p::FrameDecision base =
        p::decide_frame(g, base_costs, poptions);
    const s::FrameDecision ours =
        s::decide_frame(g, tools, our_costs, options);
    CHECK_EQ(static_cast<int>(base.status),
             static_cast<int>(p::DecisionStatus::kOk));
    CHECK_EQ(static_cast<int>(ours.status),
             static_cast<int>(s::DecisionStatus::kOk));
    if (base.status == p::DecisionStatus::kOk &&
        ours.status == s::DecisionStatus::kOk) {
      CHECK_EQ(base.superblocks.size(), ours.superblocks.size());
      const std::size_t n =
          std::min(base.superblocks.size(), ours.superblocks.size());
      for (std::size_t i = 0; i < n; ++i) {
        // Structural leaf parity is the meaningful check: same geometry,
        // order and sizes.
        std::vector<p::LeafBlock> base_leaves;
        p::collect_leaves(base.superblocks[i], base_leaves);
        std::vector<s::LeafBlock> our_leaves;
        s::collect_leaves(ours.superblocks[i], our_leaves);
        CHECK_EQ(base_leaves.size(), our_leaves.size());
        const std::size_t m = std::min(base_leaves.size(), our_leaves.size());
        for (std::size_t j = 0; j < m; ++j) {
          CHECK_EQ(base_leaves[j].mi_row, our_leaves[j].mi_row);
          CHECK_EQ(base_leaves[j].mi_col, our_leaves[j].mi_col);
          CHECK_EQ(static_cast<int>(base_leaves[j].size),
                   static_cast<int>(our_leaves[j].size));
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  test_implied_rules();
  test_rect_implied();
  test_allowed_sets();
  test_chroma_offset();
  test_sdp_all_none();
  test_chroma_follows_luma();
  test_tree_divergence();
  test_128_superblock_events();
  test_partial_superblocks();
  test_partial_64_superblock_counts();
  test_contexts();
  test_extended_symbol_golden();
  test_symbol_round_trip();
  test_golden_plan();
  test_plane_grids();
  test_chroma_offset_tree();
  test_validation_negatives();
  test_inter_without_extended_sdp();
  test_extended_sdp();
  test_rect_only_parity();

  if (g_failures == 0) {
    std::cout << "sdp_test: all checks passed\n";
    return 0;
  }
  std::cerr << "sdp_test: " << g_failures << " failure(s)\n";
  return 1;
}
