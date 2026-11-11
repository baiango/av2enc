// Byte-level syntax tests for the extended-partition symbol model.
//
// 1. read_partition() symbol plans, including every implied case.
// 2. §8.3.2 do_ext_partition / do_uneven context derivation.
// 3. A pinned golden payload for a 64x64 HORZ_3 tree.
// 4. Encode -> normative §8.2 decoder round trips on 640x360 and 1080p.
// 5. The HORZ_3@8x32 shared-chroma case.

#include "test_util.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "encoder/m0_cdf_tables.h"
#include "encoder/partition_ext/ext_syntax.h"
#include "entropy/av2/av2_entropy.h"

int g_failures = 0;

namespace {

namespace av2 = av2enc::av2;

using pe::BlockSize;
using pe::PartitionType;
using pe::SymbolKind;

constexpr PartitionType kNone = PartitionType::kNone;
constexpr PartitionType kHorz = PartitionType::kHorz;
constexpr PartitionType kVert = PartitionType::kVert;
constexpr PartitionType kHorz3 = PartitionType::kHorz3;
constexpr PartitionType kVert3 = PartitionType::kVert3;
constexpr PartitionType kHorz4A = PartitionType::kHorz4A;
constexpr PartitionType kHorz4B = PartitionType::kHorz4B;
constexpr PartitionType kVert4A = PartitionType::kVert4A;
constexpr PartitionType kVert4B = PartitionType::kVert4B;
constexpr PartitionType kSplit = PartitionType::kSplit;

pe::PartitionSet make_set(std::initializer_list<PartitionType> parts) {
  pe::PartitionSet set;
  for (PartitionType p : parts) {
    set.allowed[static_cast<int>(p)] = true;
    ++set.count;
  }
  return set;
}

void expect_symbols(const char* label, const pe::SymbolPlan& plan,
                    std::initializer_list<pe::PartitionSymbol> expected) {
  if (!plan.ok) {
    ++g_failures;
    std::cerr << label << ": plan error: " << plan.error << '\n';
    return;
  }
  if (plan.symbols.size() != expected.size()) {
    ++g_failures;
    std::cerr << label << ": symbol count " << plan.symbols.size()
              << ", expected " << expected.size() << '\n';
    return;
  }
  std::size_t i = 0;
  for (const pe::PartitionSymbol& want : expected) {
    const pe::PartitionSymbol& got = plan.symbols[i];
    CHECK(got.kind == want.kind);
    CHECK_EQ(got.value, want.value);
    CHECK_EQ(got.context, want.context);
    CHECK_EQ(got.literal, want.literal);
    ++i;
  }
}

pe::PartitionSymbol sym(SymbolKind kind, int value, int context) {
  pe::PartitionSymbol s;
  s.kind = kind;
  s.value = value;
  s.context = context;
  s.literal = kind == SymbolKind::kUneven4wayType;
  return s;
}

void test_default_cdf_tables() {
  // The three partition CDFs that already exist in the M0/M1 writer must be
  // byte-identical to the §9.3 tables generated here; this pins the plane/context
  // flattening the extended rows rely on.
  CHECK(std::memcmp(pe::kDoSplitCdf, av2enc::m0::kDoSplitCdf,
                    sizeof(pe::kDoSplitCdf)) == 0);
  CHECK(std::memcmp(pe::kDoSquareSplitCdf, av2enc::m0::kDoSquareSplitCdf,
                    sizeof(pe::kDoSquareSplitCdf)) == 0);
  CHECK(std::memcmp(pe::kRectTypeCdf, av2enc::m0::kRectTypeCdf,
                    sizeof(pe::kRectTypeCdf)) == 0);
  // do_ext_partition / do_uneven_4way_partition have no M0 equivalent; their
  // first plane starts at the §9.3 values (uniform 16384 in the default table).
  CHECK_EQ(pe::kDoExtPartitionCdf[0][0][0], 16384);
  CHECK_EQ(pe::kDoUneven4wayPartitionCdf[0][0][0], 16384);
}

void test_symbol_plans() {
  pe::Boundary open;
  pe::SymbolContexts ctx;
  ctx.do_split = 12;
  ctx.do_square_split = 3;
  ctx.rect_type = 24;
  ctx.do_ext = 12;

  // Forced by the boundary: no symbols.
  pe::Boundary forced;
  forced.implied = true;
  forced.implied_partition = kHorz;
  pe::PartitionSet set = make_set({kNone, kHorz, kVert});
  pe::SymbolPlan plan = pe::plan_partition_symbols(forced, set,
                                                   BlockSize::k64x64, kHorz, ctx);
  CHECK(plan.ok);
  CHECK(plan.forced);
  CHECK(plan.symbols.empty());

  // Single allowed partition: no symbols.
  set = make_set({kVert});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x64, kVert, ctx);
  CHECK(plan.ok);
  CHECK(plan.forced);
  CHECK(plan.symbols.empty());

  // NONE.
  set = make_set({kNone, kHorz, kVert, kHorz3});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x64, kNone, ctx);
  expect_symbols("none", plan, {sym(SymbolKind::kDoSplit, 0, 12)});

  // HORZ.
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x64, kHorz, ctx);
  expect_symbols("horz", plan,
                 {sym(SymbolKind::kDoSplit, 1, 12),
                  sym(SymbolKind::kRectType, 0, 24),
                  sym(SymbolKind::kDoExtPartition, 0, 12)});

  // HORZ_3: do_split, rect_type, do_ext=1, do_uneven=0.
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x64, kHorz3, ctx);
  expect_symbols("horz3 implied uneven", plan,
                 {sym(SymbolKind::kDoSplit, 1, 12),
                  sym(SymbolKind::kRectType, 0, 24),
                  sym(SymbolKind::kDoExtPartition, 1, 12)});

  // 4-way needs uneven_4way_partition_type (L(1)).
  set = make_set({kNone, kHorz, kVert, kHorz3, kHorz4A, kHorz4B});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x64, kHorz4B, ctx);
  expect_symbols("horz4b", plan,
                 {sym(SymbolKind::kDoSplit, 1, 12),
                  sym(SymbolKind::kRectType, 0, 24),
                  sym(SymbolKind::kDoExtPartition, 1, 12),
                  sym(SymbolKind::kDoUneven4wayPartition, 1, 12),
                  sym(SymbolKind::kUneven4wayType, 1, 0)});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x64, kHorz4A, ctx);
  expect_symbols("horz4a", plan,
                 {sym(SymbolKind::kDoSplit, 1, 12),
                  sym(SymbolKind::kRectType, 0, 24),
                  sym(SymbolKind::kDoExtPartition, 1, 12),
                  sym(SymbolKind::kDoUneven4wayPartition, 1, 12),
                  sym(SymbolKind::kUneven4wayType, 0, 0)});

  // SPLIT (only legal at 128x128).
  set = make_set({kNone, kHorz, kVert, kSplit});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k128x128, kSplit, ctx);
  expect_symbols("split", plan,
                 {sym(SymbolKind::kDoSplit, 1, 12),
                  sym(SymbolKind::kDoSquareSplit, 1, 3)});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k128x128, kHorz, ctx);
  // No extended partition in the set -> do_ext_partition is implied 0 and is
  // not coded at all.
  expect_symbols("horz with split allowed", plan,
                 {sym(SymbolKind::kDoSplit, 1, 12),
                  sym(SymbolKind::kDoSquareSplit, 0, 3),
                  sym(SymbolKind::kRectType, 0, 24)});

  // Implied do_ext: HORZ is unavailable, HORZ_3 and 4A are available.
  set = make_set({kNone, kHorz3, kHorz4A});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x64, kHorz3, ctx);
  expect_symbols("implied do_ext 3way", plan,
                 {sym(SymbolKind::kDoSplit, 1, 12),
                  sym(SymbolKind::kDoUneven4wayPartition, 0, 12)});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x64, kHorz4A, ctx);
  expect_symbols("implied do_ext 4way", plan,
                 {sym(SymbolKind::kDoSplit, 1, 12),
                  sym(SymbolKind::kDoUneven4wayPartition, 1, 12),
                  sym(SymbolKind::kUneven4wayType, 0, 0)});

  // Implied do_uneven: only one 4-way flavour is available.
  set = make_set({kNone, kHorz, kHorz4A});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x64, kHorz4A, ctx);
  // Only horizontal partitions exist -> rect_type is also implied.
  expect_symbols("implied uneven", plan,
                 {sym(SymbolKind::kDoSplit, 1, 12),
                  sym(SymbolKind::kDoExtPartition, 1, 12),
                  sym(SymbolKind::kUneven4wayType, 0, 0)});

  // NONE not allowed, SPLIT not allowed, rect implied by the block size.
  set = make_set({kHorz, kHorz3});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x128, kHorz, ctx);
  expect_symbols("no none/split", plan,
                 {sym(SymbolKind::kDoExtPartition, 0, 12)});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x128, kHorz3, ctx);
  CHECK(plan.ok);

  // Vertical uneven 4-way.
  set = make_set({kNone, kVert, kVert4A, kVert4B});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x64, kVert4B, ctx);
  expect_symbols("vert4b", plan,
                 {sym(SymbolKind::kDoSplit, 1, 12),
                  sym(SymbolKind::kDoExtPartition, 1, 12),
                  sym(SymbolKind::kUneven4wayType, 1, 0)});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x64, kVert4A, ctx);
  expect_symbols("vert4a", plan,
                 {sym(SymbolKind::kDoSplit, 1, 12),
                  sym(SymbolKind::kDoExtPartition, 1, 12),
                  sym(SymbolKind::kUneven4wayType, 0, 0)});

  // Rejecting an unreachable request.
  set = make_set({kNone, kHorz, kVert});
  plan = pe::plan_partition_symbols(open, set, BlockSize::k64x64, kHorz3, ctx);
  CHECK(!plan.ok);
}

void test_node_signal_cost() {
  pe::Boundary open;
  const pe::PartitionSet set =
      make_set({kNone, kHorz, kVert, kHorz3, kHorz4A, kHorz4B});
  pe::SymbolCosts costs;
  costs.do_split[0] = 2.0;
  costs.do_split[1] = 1.0;
  costs.rect_type[0] = 4.0;
  costs.rect_type[1] = 8.0;
  costs.do_ext_partition[0] = 32.0;
  costs.do_ext_partition[1] = 16.0;
  costs.do_uneven_4way_partition[0] = 128.0;
  costs.do_uneven_4way_partition[1] = 64.0;
  costs.uneven_4way_type[0] = 256.0;
  costs.uneven_4way_type[1] = 512.0;

  CHECK(std::fabs(pe::node_signal_cost(open, set, BlockSize::k64x64, kNone,
                                       costs) -
                  2.0) < 1e-9);
  CHECK(std::fabs(pe::node_signal_cost(open, set, BlockSize::k64x64, kHorz,
                                       costs) -
                  (1.0 + 4.0 + 32.0)) < 1e-9);
  CHECK(std::fabs(pe::node_signal_cost(open, set, BlockSize::k64x64, kHorz3,
                                       costs) -
                  (1.0 + 4.0 + 16.0 + 128.0)) < 1e-9);
  CHECK(std::fabs(pe::node_signal_cost(open, set, BlockSize::k64x64, kHorz4A,
                                       costs) -
                  (1.0 + 4.0 + 16.0 + 64.0 + 256.0)) < 1e-9);
  CHECK(std::fabs(pe::node_signal_cost(open, set, BlockSize::k64x64, kHorz4B,
                                       costs) -
                  (1.0 + 4.0 + 16.0 + 64.0 + 512.0)) < 1e-9);
  // Not allowed -> infinite.
  CHECK(!std::isfinite(pe::node_signal_cost(
      open, set, BlockSize::k64x64, kVert3, costs)));
}

void test_ext_contexts() {
  const p::FrameGeometry frame = geometry_420(64, 64, 64);
  p::MiSizeGrid grid(frame);
  // bsh = max(Height_Log2(32x32) - 1, 1) = 2; h4 = 8.
  // Left column at r = 4 is covered by a 16x16 leaf (height_log2 2 -> not
  // smaller), and at r + 4 = 8 by an 8x8 leaf (height_log2 1 -> smaller).
  grid.fill_block(4, 3, BlockSize::k16x16);
  grid.fill_block(8, 3, BlockSize::k8x8);
  int ctx = pe::ext_partition_context(frame, 4, 4, BlockSize::k32x32,
                                      /*horizontal=*/true, grid);
  CHECK_EQ(ctx, pe::size_adjust(BlockSize::k32x32) * 4 + 0 * 2 + 1);

  grid.fill_block(4, 3, BlockSize::k8x8);
  ctx = pe::ext_partition_context(frame, 4, 4, BlockSize::k32x32, true, grid);
  CHECK_EQ(ctx, pe::size_adjust(BlockSize::k32x32) * 4 + 1 * 2 + 1);

  // Vertical direction: bsw = 2; above at c = 4 is 16x16 wide (log2 2), above
  // at c + 4 = 8 is 8x8 (log2 1 -> smaller).
  p::MiSizeGrid grid2(frame);
  grid2.fill_block(3, 4, BlockSize::k16x16);
  grid2.fill_block(3, 8, BlockSize::k8x8);
  ctx = pe::ext_partition_context(frame, 4, 4, BlockSize::k32x32,
                                  /*horizontal=*/false, grid2);
  CHECK_EQ(ctx, pe::size_adjust(BlockSize::k32x32) * 4 + 0 * 2 + 1);

  // Unset neighbours behave like BLOCK_256X256: no context bits.
  p::MiSizeGrid empty(frame);
  ctx = pe::ext_partition_context(frame, 0, 0, BlockSize::k64x64, true, empty);
  CHECK_EQ(ctx, pe::size_adjust(BlockSize::k64x64) * 4);

  // node_contexts carries the rect-only contexts through unchanged.
  pe::PartitionNode node;
  node.mi_row = 0;
  node.mi_col = 0;
  node.size = BlockSize::k64x64;
  node.partition = kHorz3;
  const pe::SymbolContexts contexts = pe::node_contexts(frame, node, empty);
  const p::PartitionContexts base =
      p::partition_contexts(frame, 0, 0, BlockSize::k64x64, empty);
  CHECK_EQ(contexts.do_split, base.do_split);
  CHECK_EQ(contexts.rect_type, base.rect_type);
  CHECK_EQ(contexts.do_ext, pe::size_adjust(BlockSize::k64x64) * 4);
}

void test_frame_roundtrip(const p::FrameGeometry& frame, pe::Tools tools,
                          const pe::CostFunction& costs,
                          const pe::DecisionOptions& options,
                          bool adaptive, const char* label) {
  const pe::FrameDecision decided =
      pe::decide_frame(frame, costs, tools, options);
  CHECK(decided.status == pe::DecisionStatus::kOk);
  std::string error;
  CHECK(pe::validate_frame(frame, tools, decided, error));
  if (!error.empty()) std::cerr << label << ": " << error << '\n';

  std::vector<std::uint8_t> buffer(1u << 20, 0);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  encoder.disable_cdf_update = adaptive ? 0u : 1u;
  pe::PartitionCdfs cdfs;
  pe::init_default_cdfs(cdfs);
  p::MiSizeGrid grid(frame);
  pe::SyntaxStats stats;

  for (const pe::PartitionNode& root : decided.superblocks) {
    CHECK(pe::encode_tree_symbols(frame, tools, root, cdfs, grid, encoder,
                                  /*disable_cdf_update=*/!adaptive, &stats,
                                  error));
    if (!error.empty()) std::cerr << label << ": " << error << '\n';
  }
  std::size_t payload = 0;
  CHECK(encoder.exit_symbol(&payload) == av2::Status::kOk);
  CHECK(payload > 0);

  av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload) ==
        av2::Status::kOk);
  decoder.disable_cdf_update = adaptive ? 0u : 1u;
  pe::PartitionCdfs decoded_cdfs;
  pe::init_default_cdfs(decoded_cdfs);
  p::MiSizeGrid decoded_grid(frame);
  pe::SyntaxStats decoded_stats;

  std::size_t index = 0;
  for (const pe::PartitionNode& root : decided.superblocks) {
    pe::PartitionNode decoded;
    decoded.mi_row = root.mi_row;
    decoded.mi_col = root.mi_col;
    decoded.size = root.size;
    CHECK(pe::decode_tree_symbols(frame, tools, decoded, decoded_cdfs,
                                  decoded_grid, decoder, &decoded_stats,
                                  error));
    if (!error.empty()) std::cerr << label << ": " << error << '\n';
    CHECK_STR(pe::tree_signature(decoded), pe::tree_signature(root));
    std::string tree_error;
    CHECK(pe::validate_tree(frame, tools, decoded, tree_error));
    if (!tree_error.empty()) std::cerr << label << ": " << tree_error << '\n';
    ++index;
  }
  CHECK_EQ(index, decided.superblocks.size());
  CHECK(decoder.exit_symbol() == av2::Status::kOk);
  CHECK_EQ(stats.symbols, decoded_stats.symbols);
  CHECK_EQ(stats.literal_bits, decoded_stats.literal_bits);
}

pe::CostFunction h3_cost() {
  return choose_partition([](const pe::NodeInfo& info) {
    if (info.size == BlockSize::k64x64 || info.size == BlockSize::k32x32 ||
        info.size == BlockSize::k16x16) {
      return kHorz3;
    }
    return kNone;
  });
}

pe::CostFunction mixed_cost() {
  return choose_partition([](const pe::NodeInfo& info) {
    if (info.size == BlockSize::k128x128) return kSplit;
    if (info.size == BlockSize::k64x64) return kHorz4A;
    if (info.size == BlockSize::k32x32) return kVert3;
    return kNone;
  });
}

void test_golden() {
  const p::FrameGeometry frame = geometry_420(64, 64, 64);
  const pe::Tools tools = ext_tools(true, true);
  pe::CostFunction costs = choose_partition([](const pe::NodeInfo& info) {
    return info.depth == 0 ? kHorz3 : kNone;
  });
  const pe::PartitionDecision decision =
      pe::decide_superblock(frame, 0, 0, costs, tools, pe::DecisionOptions());
  CHECK(decision.status == pe::DecisionStatus::kOk);
  CHECK(decision.root.partition == kHorz3);

  std::vector<std::uint8_t> buffer(4096, 0);
  av2::Encoder encoder;
  CHECK(encoder.init_symbol(buffer.data(), buffer.size()) == av2::Status::kOk);
  pe::PartitionCdfs cdfs;
  pe::init_default_cdfs(cdfs);
  p::MiSizeGrid grid(frame);
  std::vector<pe::PartitionSymbol> trace;
  std::string error;
  CHECK(pe::encode_tree_symbols(frame, tools, decision.root, cdfs, grid,
                                encoder, /*disable_cdf_update=*/true, nullptr,
                                error, &trace));
  CHECK(error.empty());
  std::size_t payload = 0;
  CHECK(encoder.exit_symbol(&payload) == av2::Status::kOk);

  // Contexts: root do_split 3*4 = 12, rect_type 6*4 = 24, do_ext 3*4 = 12;
  // leaves: 64x16 -> 15*4 = 60, 32x32 -> 2*4 = 8 (twice), and the bottom
  // 64x16 sees a 32x32 above -> 15*4 + 1 = 61.
  const std::vector<pe::PartitionSymbol> expected = {
      sym(SymbolKind::kDoSplit, 1, 12),
      sym(SymbolKind::kRectType, 0, 24),
      sym(SymbolKind::kDoExtPartition, 1, 12),
      sym(SymbolKind::kDoUneven4wayPartition, 0, 12),
      sym(SymbolKind::kDoSplit, 0, 60),
      sym(SymbolKind::kDoSplit, 0, 8),
      sym(SymbolKind::kDoSplit, 0, 8),
      sym(SymbolKind::kDoSplit, 0, 61),
  };
  CHECK_EQ(trace.size(), expected.size());
  if (trace.size() == expected.size()) {
    for (std::size_t i = 0; i < trace.size(); ++i) {
      CHECK(trace[i].kind == expected[i].kind);
      CHECK_EQ(trace[i].value, expected[i].value);
      CHECK_EQ(trace[i].context, expected[i].context);
      CHECK_EQ(trace[i].literal, expected[i].literal);
    }
  }

  std::string hex;
  char byte[4];
  for (std::size_t i = 0; i < payload; ++i) {
    std::snprintf(byte, sizeof(byte), "%02x", buffer[i]);
    hex += byte;
  }
  std::cout << "golden 64x64 HORZ_3 payload (" << payload << " bytes): " << hex
            << '\n';
  // Pinned below after the first verified run; the decoder check below is the
  // independent correctness gate.
  const std::string pinned = "d4600000";
  CHECK_STR(hex, pinned);

  // Decode the pinned bytes (independent §8.2 state machine).
  av2::Decoder decoder;
  CHECK(decoder.init_symbol(buffer.data(), buffer.size(), payload) ==
        av2::Status::kOk);
  decoder.disable_cdf_update = 1u;
  pe::PartitionCdfs decoded_cdfs;
  pe::init_default_cdfs(decoded_cdfs);
  p::MiSizeGrid decoded_grid(frame);
  pe::PartitionNode decoded;
  decoded.mi_row = 0;
  decoded.mi_col = 0;
  decoded.size = BlockSize::k64x64;
  CHECK(pe::decode_tree_symbols(frame, tools, decoded, decoded_cdfs,
                                decoded_grid, decoder, nullptr, error));
  CHECK(error.empty());
  CHECK_STR(pe::tree_signature(decoded), pe::tree_signature(decision.root));
  CHECK(decoder.exit_symbol() == av2::Status::kOk);
}

void test_shared_middle_chroma() {
  const p::FrameGeometry frame = geometry_420(64, 64, 64);
  const pe::Tools tools = ext_tools(true, true);
  // 64x64 -> VERT(32x64) -> HORZ(32x32) -> VERT(16x32) -> VERT(8x32) ->
  // HORZ_3 at 8x32: the shared middle chroma case.  (8x64 is a thin block that
  // is spec-implied PARTITION_NONE, so it cannot be the parent.)
  pe::CostFunction costs = choose_partition([](const pe::NodeInfo& info) {
    if (info.size == BlockSize::k64x64) return kVert;
    if (info.size == BlockSize::k32x64) return kHorz;
    if (info.size == BlockSize::k32x32) return kVert;
    if (info.size == BlockSize::k16x32) return kVert;
    if (info.size == BlockSize::k8x32) return kHorz3;
    return kNone;
  });
  pe::PartitionDecision decision =
      pe::decide_superblock(frame, 0, 0, costs, tools, pe::DecisionOptions());
  CHECK(decision.status == pe::DecisionStatus::kOk);
  std::string error;
  CHECK(pe::validate_tree(frame, tools, decision.root, error));
  if (!error.empty()) std::cerr << error << '\n';

  pe::PartitionNode* node = &decision.root;
  while (node->partition != kHorz3 && !node->children.empty()) {
    node = &node->children[0];
  }
  CHECK(node->partition == kHorz3);
  CHECK(node->size == BlockSize::k8x32);
  CHECK_EQ(node->mi_row, 0);
  CHECK_EQ(node->mi_col, 0);

  std::vector<pe::LeafBlock> leaves;
  pe::collect_leaves(*node, leaves);
  CHECK_EQ(leaves.size(), 4);
  // top 8x8, mid-left 4x16 (no chroma), mid-right 4x16 (shared 8x16), bottom
  // 8x8.
  CHECK(leaves[0].has_chroma);
  CHECK(leaves[0].chroma.size == BlockSize::k8x8);
  CHECK(!leaves[1].has_chroma);
  CHECK(leaves[2].has_chroma);
  CHECK(leaves[2].chroma.size == BlockSize::k8x16);
  CHECK_EQ(leaves[2].chroma.mi_row, 2);
  CHECK_EQ(leaves[2].chroma.mi_col, 0);
  CHECK(leaves[3].has_chroma);
  CHECK(leaves[3].chroma.size == BlockSize::k8x8);
  CHECK_EQ(leaves[3].chroma.mi_row, 6);

  // Chroma residual sizes for the two chroma-coded middle/bottom blocks.
  CHECK(p::chroma_plane_residual_size(frame, leaves[2].chroma.size) ==
        BlockSize::k4x8);
  CHECK(p::chroma_plane_residual_size(frame, leaves[3].chroma.size) ==
        BlockSize::k4x4);
}

void test_random_roundtrip() {
  const p::FrameGeometry frames[] = {
      geometry_420(64, 64, 64),
      geometry_420(128, 128, 128),
      geometry_420(640, 360, 64),
      geometry_420(520, 128, 128),
      geometry_420(8, 8, 64),
      geometry_420(136, 72, 64),
  };
  const pe::Tools tool_sets[] = {
      ext_tools(true, true),
      ext_tools(true, false),
      ext_tools(false, false),
  };
  for (const p::FrameGeometry& frame : frames) {
    for (const pe::Tools& tools : tool_sets) {
      for (unsigned seed = 0; seed < 3; ++seed) {
        char label[96];
        std::snprintf(label, sizeof(label), "random %dx%d sb%d seed%u ext%d",
                      frame.width, frame.height, frame.sb_size, seed,
                      tools.enable_ext_partitions ? 1 : 0);
        test_frame_roundtrip(frame, tools, random_cost(seed),
                             pe::DecisionOptions(), /*adaptive=*/false, label);
      }
      if (frame.width <= 192 && tools.enable_ext_partitions) {
        test_frame_roundtrip(frame, tools, random_cost(99),
                             pe::DecisionOptions(), /*adaptive=*/true,
                             "random adaptive");
      }
    }
  }
}

}  // namespace

int main() {
  test_default_cdf_tables();
  test_symbol_plans();
  test_node_signal_cost();
  test_ext_contexts();

  const pe::Tools tools = ext_tools(true, true);
  pe::DecisionOptions options;

  // 640x360, both fixed and adaptive CDFs.
  test_frame_roundtrip(geometry_420(640, 360, 64), tools, h3_cost(), options,
                       /*adaptive=*/false, "640x360 fixed");
  test_frame_roundtrip(geometry_420(640, 360, 64), tools, h3_cost(), options,
                       /*adaptive=*/true, "640x360 adaptive");

  // 1080p / 128x128 superblocks.
  test_frame_roundtrip(geometry_420(1920, 1080, 128), tools, mixed_cost(),
                       options, /*adaptive=*/false, "1080p fixed");

  // Partial superblock at the right/bottom edge.
  test_frame_roundtrip(geometry_420(520, 128, 128), tools, mixed_cost(), options,
                       /*adaptive=*/false, "520x128");

  test_golden();
  test_shared_middle_chroma();
  test_random_roundtrip();

  if (g_failures != 0) {
    std::cerr << g_failures << " failure(s)\n";
    return 1;
  }
  std::cout << "ext_syntax_test: all checks passed\n";
  return 0;
}
