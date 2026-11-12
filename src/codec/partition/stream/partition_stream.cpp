// M1 <-> partition_ext adapter implementation.  See partition_stream.h.

#include "partition_stream.h"

#include <algorithm>
#include <cstddef>

namespace av2enc::still::partition_stream {
namespace {

constexpr int kSentinelLog2 = 6;  // BLOCK_256X256: never "smaller"
constexpr std::size_t kTileCapacityBytes = 16U * 1024U * 1024U;

int log2_or_sentinel(BlockSize size, bool width) {
  if (!base::is_valid_block_size(size)) return kSentinelLog2;
  return width ? base::mi_width_log2(size) : base::mi_height_log2(size);
}

StillNode make_node(const pe::PartitionNode& node) {
  StillNode out;
  out.mi_row = static_cast<std::uint16_t>(node.mi_row);
  out.mi_col = static_cast<std::uint16_t>(node.mi_col);
  out.bsize = static_cast<std::uint8_t>(node.size);
  out.partition = static_cast<std::uint8_t>(node.partition);
  out.allowed_mask = to_allowed_mask(node.allowed);
  out.has_chroma = node.has_chroma;
  out.chroma_offset = node.chroma_offset;
  out.chroma = node.chroma;
  if (node.forced) out.flags |= syntax::kFlagForced;
  if (node.depth == 0) out.flags |= syntax::kFlagSbRoot;
  return out;
}

bool append_node(const pe::PartitionNode& node, FrameStream& out,
                 std::string& error) {
  const std::uint32_t index = static_cast<std::uint32_t>(out.nodes.size());
  out.nodes.push_back(make_node(node));
  if (node.partition == PartitionType::kNone) {
    LeafChroma leaf;
    leaf.node_index = index;
    leaf.has_chroma = node.has_chroma;
    leaf.chroma_offset = node.chroma_offset;
    leaf.chroma = node.chroma;
    out.leaves.push_back(leaf);
    return true;
  }
  for (const pe::PartitionNode& child : node.children) {
    if (!append_node(child, out, error)) return false;
  }
  const std::size_t subtree = out.nodes.size() - index - 1;
  if (subtree > 0xFFFFU) {
    error = "partition subtree too large for the M1 preorder stream";
    return false;
  }
  out.nodes[index].subtree = static_cast<std::uint16_t>(subtree);
  return true;
}

bool validate_node_geometry(const base::FrameGeometry& geometry,
                            const StillNode& node, std::string& error) {
  if (!base::is_valid_block_size(static_cast<BlockSize>(node.bsize))) {
    error = "node has an invalid block size";
    return false;
  }
  if (!pe::is_valid_partition(static_cast<PartitionType>(node.partition))) {
    error = "node has an invalid partition";
    return false;
  }
  if (node.mi_row < 0 || node.mi_col < 0 ||
      node.mi_row >= base::mi_rows(geometry) ||
      node.mi_col >= base::mi_cols(geometry)) {
    error = "node origin is outside the frame";
    return false;
  }
  return true;
}

bool check_frame_layout(const base::FrameGeometry& geometry,
                        const pe::FrameDecision& frame, std::string& error) {
  const int cols = base::sb_cols(geometry);
  const int rows = base::sb_rows(geometry);
  const int sb_mi = base::sb_size_mi(geometry);
  if (static_cast<int>(frame.superblocks.size()) != cols * rows) {
    error = "frame has " + std::to_string(frame.superblocks.size()) +
            " superblocks, expected " + std::to_string(cols * rows);
    return false;
  }
  for (int i = 0; i < static_cast<int>(frame.superblocks.size()); ++i) {
    const pe::PartitionNode& sb =
        frame.superblocks[static_cast<std::size_t>(i)];
    const int expect_row = (i / cols) * sb_mi;
    const int expect_col = (i % cols) * sb_mi;
    if (sb.mi_row != expect_row || sb.mi_col != expect_col || sb.depth != 0) {
      error = "superblock " + std::to_string(i) +
              " is not at the next raster position at depth 0";
      return false;
    }
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// CDFs
// ---------------------------------------------------------------------------

void init_tile_partition_cdfs(TilePartitionCdfs& cdfs) {
  pe::init_default_cdfs(cdfs.ext);
}

std::uint16_t* cdf_row(TilePartitionCdfs& cdfs, pe::SymbolKind kind,
                       int context, std::string& error) {
  if (context < 0 || context > pe::max_symbol_context(kind)) {
    error = "partition symbol context out of range";
    return nullptr;
  }
  switch (kind) {
    case pe::SymbolKind::kDoSplit:
      return &cdfs.ext.do_split[0][context][0];
    case pe::SymbolKind::kDoSquareSplit:
      return &cdfs.ext.do_square_split[0][context][0];
    case pe::SymbolKind::kRectType:
      return &cdfs.ext.rect_type[0][context][0];
    case pe::SymbolKind::kDoExtPartition:
      return &cdfs.ext.do_ext_partition[0][context][0];
    case pe::SymbolKind::kDoUneven4wayPartition:
      return &cdfs.ext.do_uneven_4way_partition[0][context][0];
    case pe::SymbolKind::kUneven4wayType:
      break;
  }
  error = "literal partition symbol has no CDF row";
  return nullptr;
}

// ---------------------------------------------------------------------------
// allowed_mask
// ---------------------------------------------------------------------------

std::uint16_t to_allowed_mask(const pe::PartitionSet& allowed) {
  std::uint16_t mask = 0;
  for (int p = 0; p < kPartitionCount; ++p) {
    if (allowed.allowed[p]) {
      mask = static_cast<std::uint16_t>(mask | (1U << p));
    }
  }
  return mask;
}

bool mask_allows(std::uint16_t mask, PartitionType partition) {
  const unsigned p = static_cast<unsigned>(partition);
  if (p >= static_cast<unsigned>(kPartitionCount)) return false;
  return (mask & (1U << p)) != 0;
}

int mask_count(std::uint16_t mask) {
  int count = 0;
  for (int p = 0; p < kPartitionCount; ++p) {
    if ((mask & (1U << p)) != 0) ++count;
  }
  return count;
}

std::uint8_t to_writer_mask8(std::uint16_t full_mask) {
  unsigned mask = 0;
  if (mask_allows(full_mask, PartitionType::kNone)) mask |= 1U << 0;
  if (mask_allows(full_mask, PartitionType::kSplit)) mask |= 1U << 1;
  const bool horz = mask_allows(full_mask, PartitionType::kHorz) ||
                    mask_allows(full_mask, PartitionType::kHorz3) ||
                    mask_allows(full_mask, PartitionType::kHorz4A) ||
                    mask_allows(full_mask, PartitionType::kHorz4B);
  const bool vert = mask_allows(full_mask, PartitionType::kVert) ||
                    mask_allows(full_mask, PartitionType::kVert3) ||
                    mask_allows(full_mask, PartitionType::kVert4A) ||
                    mask_allows(full_mask, PartitionType::kVert4B);
  if (horz) mask |= 1U << 2;
  if (vert) mask |= 1U << 3;
  return static_cast<std::uint8_t>(mask);
}

bool mask_fits_writer_u8(std::uint16_t full_mask) {
  return (full_mask & static_cast<std::uint16_t>(~kWriterMaskBits)) == 0;
}

// ---------------------------------------------------------------------------
// Contexts
// ---------------------------------------------------------------------------

int ext_partition_context(BlockSize size, bool horizontal,
                          const NeighbourContext& neighbours) {
  if (!base::is_valid_block_size(size)) return 0;
  const int adjust = pe::size_adjust(size);
  int ctx1 = 0;
  int ctx2 = 0;
  if (horizontal) {
    const int bsh = std::max(base::mi_height_log2(size) - 1, 1);
    ctx1 = neighbours.left_log2 < bsh ? 1 : 0;
    ctx2 = neighbours.left_mid_log2 < bsh ? 1 : 0;
  } else {
    const int bsw = std::max(base::mi_width_log2(size) - 1, 1);
    ctx1 = neighbours.above_log2 < bsw ? 1 : 0;
    ctx2 = neighbours.above_mid_log2 < bsw ? 1 : 0;
  }
  return adjust * 4 + ctx1 * 2 + ctx2;
}

NodeContexts contexts_from_neighbours(BlockSize size, PartitionType partition,
                                      const NeighbourContext& neighbours) {
  NodeContexts out;
  if (!base::is_valid_block_size(size)) return out;

  const int bsw = std::max(base::mi_width_log2(size), 1);
  const int bsh = std::max(base::mi_height_log2(size), 1);
  const int ctx1 = neighbours.left_log2 < bsh ? 1 : 0;
  const int ctx2 = neighbours.above_log2 < bsw ? 1 : 0;
  out.do_split = pe::size_adjust(size) * 4 + ctx1 * 2 + ctx2;
  out.rect_type = pe::size_adjust_rect_type(size) * 4 + ctx1 * 2 + ctx2;

  // do_square_split: raw log2 and explicit availability, exactly like
  // partition::partition_contexts().
  const int raw_bsw = base::mi_width_log2(size);
  const int raw_bsh = base::mi_height_log2(size);
  int square = 0;
  if (size == BlockSize::k256x256) square += 4;
  if (neighbours.left_available && neighbours.left_log2 < raw_bsh) square += 2;
  if (neighbours.above_available && neighbours.above_log2 < raw_bsw) square += 1;
  out.do_square_split = square;

  const PartitionType direction = pe::rect_direction(partition);
  if (direction != PartitionType::kNone) {
    out.do_ext =
        ext_partition_context(size, direction == PartitionType::kHorz,
                              neighbours);
  }
  return out;
}

NeighbourContext neighbours_from_grid(const base::FrameGeometry& geometry,
                                      int mi_row, int mi_col, BlockSize size,
                                      const base::MiSizeGrid& grid) {
  NeighbourContext out;
  if (!base::is_valid_block_size(size)) return out;
  const int h4 = base::num_4x4_blocks_high(size);
  const int w4 = base::num_4x4_blocks_wide(size);
  out.left_log2 =
      log2_or_sentinel(grid.at(mi_row, mi_col - 1), /*width=*/false);
  out.left_mid_log2 = log2_or_sentinel(
      grid.at(mi_row + (h4 >> 1), mi_col - 1), /*width=*/false);
  out.above_log2 =
      log2_or_sentinel(grid.at(mi_row - 1, mi_col), /*width=*/true);
  out.above_mid_log2 = log2_or_sentinel(
      grid.at(mi_row - 1, mi_col + (w4 >> 1)), /*width=*/true);
  out.left_available = base::is_inside(geometry, mi_row, mi_col - 1);
  out.above_available = base::is_inside(geometry, mi_row - 1, mi_col);
  return out;
}

NeighbourContext neighbours_from_arrays(std::span<const int> left_mi,
                                        std::span<const int> above_mi,
                                        int mi_row, int mi_col, BlockSize size,
                                        bool left_available,
                                        bool above_available) {
  NeighbourContext out;
  if (!base::is_valid_block_size(size)) return out;
  const auto read = [](std::span<const int> values, int index) {
    if (index < 0 || index >= static_cast<int>(values.size())) {
      return kSentinelLog2;
    }
    return values[static_cast<std::size_t>(index)];
  };
  const int h4 = base::num_4x4_blocks_high(size);
  const int w4 = base::num_4x4_blocks_wide(size);
  out.left_log2 = read(left_mi, mi_row);
  out.left_mid_log2 = read(left_mi, mi_row + (h4 >> 1));
  out.above_log2 = read(above_mi, mi_col);
  out.above_mid_log2 = read(above_mi, mi_col + (w4 >> 1));
  out.left_available = left_available;
  out.above_available = above_available;
  return out;
}

ChromaPlaneBlock chroma_plane_block(const base::FrameGeometry& geometry,
                                    const pe::ChromaBlock& block) {
  ChromaPlaneBlock out;
  if (!base::is_valid_block_size(block.size)) return out;
  const int sub_x = geometry.chroma_subsampling_x ? 1 : 0;
  const int sub_y = geometry.chroma_subsampling_y ? 1 : 0;
  out.mi_row = block.mi_row >> sub_y;
  out.mi_col = block.mi_col >> sub_x;
  out.residual_size = base::chroma_plane_residual_size(geometry, block.size);
  return out;
}

// ---------------------------------------------------------------------------
// M1 node shape
// ---------------------------------------------------------------------------

BlockSyntaxV2 to_block_syntax_v2(const StillNode& node) {
  BlockSyntaxV2 out;
  out.mi_row = node.mi_row;
  out.mi_col = node.mi_col;
  out.bsize = node.bsize;
  out.partition = node.partition;
  out.allowed_mask = node.allowed_mask;
  out.flags = node.flags;
  out.subtree = node.subtree;
  out.has_chroma = node.has_chroma ? 1 : 0;
  out.chroma_bsize = static_cast<std::uint8_t>(node.chroma.size);
  out.chroma_mi_row = static_cast<std::uint16_t>(node.chroma.mi_row);
  out.chroma_mi_col = static_cast<std::uint16_t>(node.chroma.mi_col);
  return out;
}

bool build_superblock_stream(const base::FrameGeometry& geometry,
                             const pe::Tools& tools,
                             const pe::PartitionNode& root, FrameStream& out,
                             std::string& error) {
  out.nodes.clear();
  out.leaves.clear();
  out.node_count = 0;
  out.leaf_count = 0;
  if (!pe::validate_tree(geometry, tools, root, error)) return false;
  if (root.depth != 0) {
    error = "superblock stream root is not at depth 0";
    return false;
  }
  if (!append_node(root, out, error)) {
    out.nodes.clear();
    out.leaves.clear();
    return false;
  }
  out.node_count = static_cast<int>(out.nodes.size());
  out.leaf_count = static_cast<int>(out.leaves.size());
  return true;
}

bool build_frame_stream(const base::FrameGeometry& geometry,
                        const pe::Tools& tools, const pe::FrameDecision& frame,
                        FrameStream& out, std::string& error) {
  out.nodes.clear();
  out.leaves.clear();
  out.node_count = 0;
  out.leaf_count = 0;
  if (!pe::validate_frame(geometry, tools, frame, error)) return false;
  if (!check_frame_layout(geometry, frame, error)) return false;
  for (const pe::PartitionNode& sb : frame.superblocks) {
    FrameStream one;
    if (!build_superblock_stream(geometry, tools, sb, one, error)) {
      out.nodes.clear();
      out.leaves.clear();
      return false;
    }
    const std::uint32_t base_index =
        static_cast<std::uint32_t>(out.nodes.size());
    out.nodes.insert(out.nodes.end(), one.nodes.begin(), one.nodes.end());
    for (const LeafChroma& leaf : one.leaves) {
      LeafChroma moved = leaf;
      moved.node_index += base_index;
      out.leaves.push_back(moved);
    }
  }
  out.node_count = static_cast<int>(out.nodes.size());
  out.leaf_count = static_cast<int>(out.leaves.size());
  return true;
}

// ---------------------------------------------------------------------------
// Symbol planning and emission
// ---------------------------------------------------------------------------

NodePlan plan_node(const base::FrameGeometry& geometry, const pe::Tools& tools,
                   const StillNode& node, const NodeContexts& contexts) {
  NodePlan plan;
  const BlockSize size = static_cast<BlockSize>(node.bsize);
  const PartitionType partition =
      static_cast<PartitionType>(node.partition);
  if (!validate_node_geometry(geometry, node, plan.error)) {
    plan.ok = false;
    return plan;
  }

  plan.boundary = pe::implied_partition(geometry, node.mi_row, node.mi_col,
                                        size);
  plan.allowed = pe::init_allowed_partitions(
      geometry, node.mi_row, node.mi_col, size, node.has_chroma,
      node.chroma_offset, tools);

  const std::uint16_t derived = to_allowed_mask(plan.allowed);
  if (derived != node.allowed_mask) {
    plan.ok = false;
    plan.error = "node has a stale allowed_mask";
    return plan;
  }
  if (!plan.allowed.is_allowed(partition)) {
    plan.ok = false;
    plan.error = "node uses a partition that is not allowed";
    return plan;
  }

  pe::SymbolContexts pe_contexts;
  pe_contexts.do_split = contexts.do_split;
  pe_contexts.do_square_split = contexts.do_square_split;
  pe_contexts.rect_type = contexts.rect_type;
  pe_contexts.do_ext = contexts.do_ext;
  plan.symbols =
      pe::plan_partition_symbols(plan.boundary, plan.allowed, size, partition,
                                 pe_contexts);
  plan.forced = plan.symbols.forced;
  if (!plan.symbols.ok) {
    plan.ok = false;
    plan.error = plan.symbols.error;
    return plan;
  }
  const bool node_forced = (node.flags & syntax::kFlagForced) != 0;
  if (node_forced != plan.forced) {
    plan.ok = false;
    plan.error = "node forced flag does not match the spec's implied rules";
    return plan;
  }
  return plan;
}

NodePlan plan_node(const base::FrameGeometry& geometry, const pe::Tools& tools,
                   const StillNode& node, const NeighbourContext& neighbours) {
  const NodeContexts contexts = contexts_from_neighbours(
      static_cast<BlockSize>(node.bsize),
      static_cast<PartitionType>(node.partition), neighbours);
  return plan_node(geometry, tools, node, contexts);
}

NodePlan plan_node(const base::FrameGeometry& geometry, const pe::Tools& tools,
                   const StillNode& node, const base::MiSizeGrid& grid) {
  const NeighbourContext neighbours =
      neighbours_from_grid(geometry, node.mi_row, node.mi_col,
                           static_cast<BlockSize>(node.bsize), grid);
  return plan_node(geometry, tools, node, neighbours);
}

bool emit_symbols(const pe::SymbolPlan& plan, TilePartitionCdfs& cdfs,
                  av2::Encoder& encoder, pe::SyntaxStats* stats,
                  std::string& error) {
  if (!plan.ok) {
    error = plan.error;
    return false;
  }
  for (const pe::PartitionSymbol& symbol : plan.symbols) {
    av2::Status status = av2::Status::kOk;
    if (symbol.literal) {
      status = encoder.encode_bool(symbol.value != 0);
      if (status == av2::Status::kOk && stats != nullptr) {
        ++stats->literal_bits;
      }
    } else {
      std::uint16_t* row =
          cdf_row(cdfs, symbol.kind, symbol.context, error);
      if (row == nullptr) return false;
      status = encoder.encode_symbol(row, 2, symbol.value);
      if (status == av2::Status::kOk && stats != nullptr) {
        ++stats->symbols;
      }
    }
    if (status != av2::Status::kOk) {
      error = std::string("entropy encode failed: ") +
              av2::status_name(status);
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Whole-frame reference codec
// ---------------------------------------------------------------------------

namespace {

bool encode_tree_walk(const base::FrameGeometry& geometry,
                      const pe::Tools& tools, const pe::PartitionNode& root,
                      TilePartitionCdfs& cdfs, av2::Encoder& encoder,
                      base::MiSizeGrid& grid, pe::SyntaxStats& stats,
                      std::string& error) {
  std::string walk_error;
  std::function<bool(const pe::PartitionNode&)> walk =
      [&](const pe::PartitionNode& node) -> bool {
    const StillNode m1 = make_node(node);
    const NodePlan plan = plan_node(geometry, tools, m1, grid);
    if (!plan.ok) {
      walk_error = plan.error;
      return false;
    }
    if (!emit_symbols(plan.symbols, cdfs, encoder, &stats, walk_error)) {
      return false;
    }
    if (node.partition == PartitionType::kNone) {
      grid.fill_block(node.mi_row, node.mi_col, node.size);
      ++stats.nodes;
      ++stats.leaves;
      return true;
    }
    ++stats.nodes;
    for (const pe::PartitionNode& child : node.children) {
      if (!walk(child)) return false;
    }
    return true;
  };
  if (!walk(root)) {
    error = walk_error;
    return false;
  }
  return true;
}

bool start_encoder(const base::FrameGeometry& geometry, bool disable_cdf_update,
                   av2::Encoder& encoder, std::vector<std::uint8_t>& buffer,
                   std::string& error) {
  (void)geometry;
  buffer.assign(kTileCapacityBytes, 0);
  const av2::Status status =
      encoder.init_symbol(buffer.data(), buffer.size());
  if (status != av2::Status::kOk) {
    error = std::string("entropy init failed: ") + av2::status_name(status);
    return false;
  }
  encoder.disable_cdf_update = disable_cdf_update ? 1U : 0U;
  return true;
}

bool finish_encoder(av2::Encoder& encoder, std::vector<std::uint8_t>& buffer,
                    std::vector<std::uint8_t>& out, std::string& error) {
  std::size_t payload_bytes = 0;
  const av2::Status status = encoder.exit_symbol(&payload_bytes);
  if (status != av2::Status::kOk) {
    error = std::string("entropy exit failed: ") + av2::status_name(status);
    return false;
  }
  out.assign(buffer.begin(),
             buffer.begin() + static_cast<std::ptrdiff_t>(payload_bytes));
  return true;
}

}  // namespace

bool encode_superblock_partitions(const base::FrameGeometry& geometry,
                                  const pe::Tools& tools,
                                  const pe::PartitionNode& root,
                                  TilePartitionCdfs& cdfs,
                                  bool disable_cdf_update,
                                  std::vector<std::uint8_t>& out,
                                  pe::SyntaxStats* stats, std::string& error) {
  out.clear();
  if (!pe::validate_tree(geometry, tools, root, error)) return false;
  av2::Encoder encoder;
  std::vector<std::uint8_t> buffer;
  if (!start_encoder(geometry, disable_cdf_update, encoder, buffer, error)) {
    return false;
  }
  base::MiSizeGrid grid(geometry);
  pe::SyntaxStats local{};
  if (!encode_tree_walk(geometry, tools, root, cdfs, encoder, grid, local,
                        error)) {
    return false;
  }
  if (!finish_encoder(encoder, buffer, out, error)) return false;
  if (stats != nullptr) *stats = local;
  return true;
}

bool encode_frame_partitions(const base::FrameGeometry& geometry,
                             const pe::Tools& tools,
                             const pe::FrameDecision& frame,
                             TilePartitionCdfs& cdfs, bool disable_cdf_update,
                             std::vector<std::uint8_t>& out,
                             pe::SyntaxStats* stats, std::string& error) {
  out.clear();
  if (!pe::validate_frame(geometry, tools, frame, error)) return false;
  if (!check_frame_layout(geometry, frame, error)) return false;

  av2::Encoder encoder;
  std::vector<std::uint8_t> buffer;
  if (!start_encoder(geometry, disable_cdf_update, encoder, buffer, error)) {
    return false;
  }
  base::MiSizeGrid grid(geometry);
  pe::SyntaxStats local{};
  for (const pe::PartitionNode& sb : frame.superblocks) {
    if (!encode_tree_walk(geometry, tools, sb, cdfs, encoder, grid, local,
                          error)) {
      return false;
    }
  }
  if (!finish_encoder(encoder, buffer, out, error)) return false;
  if (stats != nullptr) *stats = local;
  return true;
}

bool decode_frame_partitions(const base::FrameGeometry& geometry,
                             const pe::Tools& tools, TilePartitionCdfs& cdfs,
                             av2::Decoder& decoder, bool disable_cdf_update,
                             pe::FrameDecision& frame, pe::SyntaxStats* stats,
                             std::string& error) {
  frame = pe::FrameDecision{};
  decoder.disable_cdf_update = disable_cdf_update ? 1U : 0U;
  base::MiSizeGrid grid(geometry);
  pe::SyntaxStats local{};
  const int sb_mi = base::sb_size_mi(geometry);
  for (int row = 0; row < base::mi_rows(geometry); row += sb_mi) {
    for (int col = 0; col < base::mi_cols(geometry); col += sb_mi) {
      pe::PartitionNode root;
      root.mi_row = row;
      root.mi_col = col;
      root.size = base::superblock_size(geometry.sb_size);
      pe::SyntaxStats one{};
      if (!pe::decode_tree_symbols(geometry, tools, root, cdfs.ext, grid,
                                   decoder, &one, error)) {
        frame.superblocks.clear();
        frame.status = pe::DecisionStatus::kInvalidGeometry;
        return false;
      }
      local.symbols += one.symbols;
      local.literal_bits += one.literal_bits;
      local.nodes += one.nodes;
      local.leaves += one.leaves;
      frame.nodes += one.nodes;
      frame.leaves += one.leaves;
      frame.superblocks.push_back(std::move(root));
    }
  }
  frame.status = pe::DecisionStatus::kOk;
  if (stats != nullptr) *stats = local;
  return true;
}

// ---------------------------------------------------------------------------
// Cost model adapter
// ---------------------------------------------------------------------------

pe::CostFunction lift_rect_costs(const base::CostFunction& rect_costs,
                                 const ExtSignalCosts& ext) {
  return [rect_costs, ext](const pe::NodeInfo& info) {
    pe::NodeCosts out;
    if (rect_costs) {
      base::PartitionNodeInfo base_info;
      base_info.mi_row = info.mi_row;
      base_info.mi_col = info.mi_col;
      base_info.size = info.size;
      base_info.depth = info.depth;
      base_info.has_chroma = info.has_chroma;
      base_info.chroma_offset = info.chroma_offset;
      base_info.boundary.has_rows = info.boundary.has_rows;
      base_info.boundary.has_cols = info.boundary.has_cols;
      base_info.boundary.implied = info.boundary.implied;
      base_info.boundary.implied_partition =
          pe::to_base(info.boundary.implied_partition);
      // pe::PartitionSet is indexed by the 10 spec values; the rect-only
      // callback expects the compact 4-value base enum.  Map NONE and SPLIT
      // exactly and expose the whole horizontal/vertical family as the base
      // rect bit, matching to_writer_mask8().
      const bool base_allowed[base::kPartitionTypeCount] = {
          info.allowed.allowed[0],
          info.allowed.allowed[9],
          info.allowed.allowed[1] || info.allowed.allowed[3] ||
              info.allowed.allowed[5] || info.allowed.allowed[6],
          info.allowed.allowed[2] || info.allowed.allowed[4] ||
              info.allowed.allowed[7] || info.allowed.allowed[8]};
      for (int p = 0; p < base::kPartitionTypeCount; ++p) {
        base_info.allowed.allowed[p] = base_allowed[p];
        if (base_allowed[p]) ++base_info.allowed.count;
      }
      const base::NodeCosts costs = rect_costs(base_info);
      out.leaf = costs.leaf;
      out.none_signal = costs.none_signal;
      out.split_signal = costs.split_signal;
      out.horz_signal = costs.horz_signal;
      out.vert_signal = costs.vert_signal;
    }
    out.horz3_signal = ext.three_way;
    out.vert3_signal = ext.three_way;
    out.horz4a_signal = ext.uneven_4way;
    out.horz4b_signal = ext.uneven_4way;
    out.vert4a_signal = ext.uneven_4way;
    out.vert4b_signal = ext.uneven_4way;
    return out;
  };
}

}  // namespace av2enc::still::partition_stream
