#include "codec/partition/sub8/sub8.h"

#include "pipeline/still/still_tables.h"
#include "codec/transform/variants/tx_variants.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace av2enc::still::sub8 {
namespace {

int first_allowed(const partition::PartitionSet& set) {
  for (int i = 0; i < partition::kPartitionTypeCount; ++i) {
    if (set.allowed[i]) return i;
  }
  return -1;
}

}  // namespace

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

bool is_sub8(BlockSize size) noexcept {
  if (!partition::is_valid_block_size(size)) return false;
  return min_dim_px(size) == 4;
}

int min_dim_px(BlockSize size) noexcept {
  if (!partition::is_valid_block_size(size)) return 0;
  return std::min(partition::block_width(size), partition::block_height(size));
}

const char* block_name(BlockSize size) noexcept {
  return partition::block_size_name(size);
}

// ---------------------------------------------------------------------------
// Partition validity
// ---------------------------------------------------------------------------

bool partition_allowed(const FrameGeometry& geometry, int mi_row, int mi_col,
                       BlockSize size, PartitionType partition,
                       bool has_chroma, bool chroma_offset) {
  // init_allowed_partitions() includes the spec fallback that makes
  // PARTITION_NONE legal when the geometry forbids every candidate; asking
  // is_partition_allowed() alone would report that node as having no legal
  // partition at all.
  return partition::init_allowed_partitions(geometry, mi_row, mi_col, size,
                                            has_chroma, chroma_offset)
      .is_allowed(partition);
}

PartitionPlan plan_partition(const FrameGeometry& geometry, int mi_row,
                             int mi_col, BlockSize size, bool has_chroma,
                             bool chroma_offset) {
  PartitionPlan plan;
  plan.allowed = partition::init_allowed_partitions(
      geometry, mi_row, mi_col, size, has_chroma, chroma_offset);
  plan.implied = partition::partition_implied(geometry, mi_row, mi_col, size);
  plan.rect_type = partition::rect_type_implied_by_bsize(size);
  plan.none_allowed = plan.allowed.is_allowed(PartitionType::kNone);
  plan.allow_horz = plan.allowed.is_allowed(PartitionType::kHorz);
  plan.allow_vert = plan.allowed.is_allowed(PartitionType::kVert);

  // read_partition(): an implied partition wins when it is in the allowed set.
  if (plan.implied.implied &&
      plan.allowed.is_allowed(plan.implied.implied_partition)) {
    plan.forced = true;
    plan.forced_partition = plan.implied.implied_partition;
    return plan;
  }
  if (plan.allowed.count == 1) {
    plan.forced = true;
    const int index = first_allowed(plan.allowed);
    plan.forced_partition = static_cast<PartitionType>(index);
    return plan;
  }

  plan.codes_do_split = plan.none_allowed;
  plan.codes_do_square_split = plan.allowed.is_allowed(PartitionType::kSplit);
  // rect_type is coded only when it is not implied by the block size AND both
  // rect families are available.
  plan.codes_rect_type =
      !plan.rect_type.implied && plan.allow_horz && plan.allow_vert;
  return plan;
}

// ---------------------------------------------------------------------------
// Chroma ownership
// ---------------------------------------------------------------------------

namespace {

struct ChromaStep {
  ChromaInfo info;
  bool offset;
};

ChromaStep chroma_step(const FrameGeometry& geometry,
                       PartitionType partition, BlockSize size, int mi_row,
                       int mi_col, bool has_chroma, bool chroma_offset,
                       const ChromaInfo& inherited) {
  ChromaStep step;
  step.info = inherited;
  step.offset = chroma_offset;
  if (has_chroma && !step.offset) {
    // §5.20.3.1: ChromaMi* is captured at the last node whose chromaOffset was
    // still 0, even when this partition does not itself force an offset.
    step.offset = partition::is_chroma_offset_for_partition(
        geometry, partition, size);
    step.info.valid = true;
    step.info.mi_row = mi_row;
    step.info.mi_col = mi_col;
    step.info.size = size;
  }
  return step;
}

// Enumerates the in-frame children with their spec recursion arguments.
struct ChildSpec {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;
  bool has_chroma = false;
  bool chroma_offset = false;
};

int enumerate_children(const FrameGeometry& geometry, const Node& node,
                       bool next_offset, ChildSpec out[4]) {
  if (node.partition == PartitionType::kNone) return 0;
  const BlockSize subsize =
      partition::partition_subsize(node.partition, node.size);
  if (!partition::is_valid_block_size(subsize)) return 0;
  const int w4 = partition::num_4x4_blocks_wide(node.size);
  const int h4 = partition::num_4x4_blocks_high(node.size);
  const int half_w = w4 >> 1;
  const int half_h = h4 >> 1;
  const int mi_rows = partition::mi_rows(geometry);
  const int mi_cols = partition::mi_cols(geometry);
  int count = 0;
  const auto push = [&](int r, int c, bool has_chroma, bool offset) {
    if (r < mi_rows && c < mi_cols) {
      out[count].mi_row = r;
      out[count].mi_col = c;
      out[count].size = subsize;
      out[count].has_chroma = has_chroma;
      out[count].chroma_offset = offset;
      ++count;
    }
  };
  switch (node.partition) {
    case PartitionType::kSplit:
      // decode_partition() passes chromaOffset = 0 for all four children.
      push(node.mi_row, node.mi_col, node.has_chroma, false);
      push(node.mi_row, node.mi_col + half_w, node.has_chroma, false);
      push(node.mi_row + half_h, node.mi_col, node.has_chroma, false);
      push(node.mi_row + half_h, node.mi_col + half_w, node.has_chroma, false);
      break;
    case PartitionType::kHorz:
      push(node.mi_row, node.mi_col, node.has_chroma && !next_offset,
           next_offset);
      push(node.mi_row + half_h, node.mi_col, node.has_chroma, next_offset);
      break;
    case PartitionType::kVert:
      push(node.mi_row, node.mi_col, node.has_chroma && !next_offset,
           next_offset);
      push(node.mi_row, node.mi_col + half_w, node.has_chroma, next_offset);
      break;
    case PartitionType::kNone:
      break;
  }
  return count;
}

}  // namespace

void assign_chroma(const FrameGeometry& geometry, bool has_chroma,
                   bool chroma_offset, const ChromaInfo& inherited,
                   Node& node) {
  const ChromaStep step =
      chroma_step(geometry, node.partition, node.size, node.mi_row,
                  node.mi_col, has_chroma, chroma_offset, inherited);
  node.has_chroma = has_chroma;
  node.chroma_offset = chroma_offset;
  node.chroma = step.info;

  ChildSpec specs[4];
  const int count = enumerate_children(geometry, node, step.offset, specs);
  if (count != static_cast<int>(node.children.size())) return;
  for (int i = 0; i < count; ++i) {
    Node& child = node.children[static_cast<std::size_t>(i)];
    if (child.mi_row != specs[i].mi_row || child.mi_col != specs[i].mi_col) {
      continue;
    }
    assign_chroma(geometry, specs[i].has_chroma, specs[i].chroma_offset,
                  step.info, child);
  }
}

void collect_leaves(const Node& node, std::vector<const Node*>& leaves) {
  if (node.partition == PartitionType::kNone) {
    leaves.push_back(&node);
    return;
  }
  for (const Node& child : node.children) collect_leaves(child, leaves);
}

// ---------------------------------------------------------------------------
// Tree builder
// ---------------------------------------------------------------------------

namespace {

bool build_node(const FrameGeometry& geometry, const ChoosePartition& choose,
                Node& node, std::string& error) {
  const PartitionPlan plan =
      plan_partition(geometry, node.mi_row, node.mi_col, node.size,
                     node.has_chroma, node.chroma_offset);
  PartitionType partition = PartitionType::kNone;
  if (plan.forced) {
    partition = plan.forced_partition;
  } else {
    partition = choose(node, plan);
    if (!plan.allowed.is_allowed(partition)) {
      error = "chosen partition " +
              std::string(partition::partition_name(partition)) +
              " is not allowed at " + std::string(block_name(node.size)) +
              " (" + std::to_string(node.mi_row) + "," +
              std::to_string(node.mi_col) + ")";
      return false;
    }
  }
  // A boundary-implied node never codes syntax; make the caller's `forced`
  // flag match the derivation so the writer does not emit do_split.
  node.partition = partition;
  node.forced = plan.forced;

  // §5.20.3.1: the ChromaMi* capture happens at this node, before the
  // children are decoded.
  const ChromaStep step =
      chroma_step(geometry, partition, node.size, node.mi_row, node.mi_col,
                  node.has_chroma, node.chroma_offset, node.chroma);
  // Node::chroma_offset is the *incoming* decode_partition() argument, like
  // partition::PartitionNode::chroma_offset.  `step.offset` is the value the
  // children receive.
  node.chroma = step.info;
  if (partition == PartitionType::kNone) return true;

  ChildSpec specs[4];
  const int count = enumerate_children(geometry, node, step.offset, specs);
  if (count == 0) {
    error = "partition has no in-frame children";
    return false;
  }
  node.children.clear();
  node.children.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    Node child;
    child.mi_row = specs[i].mi_row;
    child.mi_col = specs[i].mi_col;
    child.size = specs[i].size;
    child.has_chroma = specs[i].has_chroma;
    child.chroma_offset = specs[i].chroma_offset;
    child.chroma = step.info;
    if (!build_node(geometry, choose, child, error)) return false;
    node.children.push_back(std::move(child));
  }
  return true;
}

}  // namespace

bool build_superblock(const FrameGeometry& geometry, int sb_mi_row,
                      int sb_mi_col, const ChoosePartition& choose, Node& out,
                      std::string& error) {
  if (geometry.sb_size != 64) {
    error = "sub8 tree builder only supports the M1 64x64 superblock";
    return false;
  }
  const int sb_mi = partition::sb_size_mi(geometry);
  if (sb_mi_row < 0 || sb_mi_col < 0 || sb_mi_row % sb_mi != 0 ||
      sb_mi_col % sb_mi != 0 || sb_mi_row >= partition::mi_rows(geometry) ||
      sb_mi_col >= partition::mi_cols(geometry)) {
    error = "superblock origin is not an in-frame aligned superblock";
    return false;
  }
  out = Node{};
  out.mi_row = sb_mi_row;
  out.mi_col = sb_mi_col;
  out.size = BlockSize::k64x64;
  out.has_chroma = true;
  out.chroma_offset = false;
  if (!build_node(geometry, choose, out, error)) return false;
  // Re-derive the whole chroma chain independently and require agreement.
  Node reference = out;
  assign_chroma(geometry, true, false, ChromaInfo{}, reference);
  std::vector<const Node*> a;
  std::vector<const Node*> b;
  collect_leaves(out, a);
  collect_leaves(reference, b);
  if (a.size() != b.size()) {
    error = "chroma derivation changed the leaf count";
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i]->has_chroma != b[i]->has_chroma ||
        a[i]->chroma_offset != b[i]->chroma_offset ||
        a[i]->chroma.valid != b[i]->chroma.valid ||
        a[i]->chroma.size != b[i]->chroma.size ||
        a[i]->chroma.mi_row != b[i]->chroma.mi_row ||
        a[i]->chroma.mi_col != b[i]->chroma.mi_col) {
      error = "chroma ownership disagrees with the decode recursion";
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Mode info
// ---------------------------------------------------------------------------

BlockSize chroma_residual_size(const FrameGeometry& geometry,
                               const ChromaInfo& chroma) {
  if (!chroma.valid) return BlockSize::kInvalid;
  return partition::chroma_plane_residual_size(geometry, chroma.size);
}

ModeInfoPlan mode_info_plan(const FrameGeometry& geometry, const Node& leaf,
                            bool enable_cfl_intra) {
  ModeInfoPlan plan;
  plan.codes_y_mode = true;
  plan.codes_uv_mode = leaf.has_chroma;
  plan.palette_allowed = false;
  if (!leaf.has_chroma) return plan;
  plan.uv_residual_size = chroma_residual_size(geometry, leaf.chroma);
  if (partition::is_valid_block_size(plan.uv_residual_size)) {
    plan.cfl_allowed =
        enable_cfl_intra &&
        partition::block_width(plan.uv_residual_size) <= 64 &&
        partition::block_height(plan.uv_residual_size) <= 64;
  }
  return plan;
}

// ---------------------------------------------------------------------------
// Coefficient contexts
// ---------------------------------------------------------------------------

int txb_skip_context(int plane, BlockSize plane_bsize, int tx_lw, int tx_lh,
                     int above, int left, bool u_has_cf) {
  if (!partition::is_valid_block_size(plane_bsize)) return 0;
  const int bw = partition::block_width(plane_bsize);
  const int bh = partition::block_height(plane_bsize);
  const int tw = 1 << tx_lw;
  const int th = 1 << tx_lh;
  if (plane == 0) {
    if (bw == tw && bh == th) return 0;
    const int top = std::min(above, 4);
    const int lhs = std::min(left, 4);
    return (top + lhs + 3) >> 1;
  }
  int ctx = (above != 0 ? 1 : 0) + (left != 0 ? 1 : 0);
  if (plane == 2) {
    if (bw * bh > tw * th) ctx += 3;
    if (u_has_cf) ctx += 6;
  } else {
    ctx += 6;
  }
  return ctx;
}

// ---------------------------------------------------------------------------
// Transform selection
// ---------------------------------------------------------------------------

int eob_multisize(int tx_size) {
  const auto params =
      txv::make_tx_params(static_cast<txv::TxSize>(tx_size),
                          txv::TxType::kDctDct);
  if (!params.has_value()) return -1;
  const int lw = params->log2_w;
  const int lh = params->log2_h;
  return std::min(lw, 5) + std::min(lh, 5) - 4;
}

bool tx_type_set1(int tx_size) {
  if (tx_size < 0 || tx_size >= 25) return false;
  const int sqr = av2enc::still::kTxSizeSqr[tx_size];
  const int sqr_up = av2enc::still::kTxSizeSqrUp[tx_size];
  if (sqr_up > 3) return false;                 // TX_SET_WIDE/HIGH_64
  if (sqr_up == 3 && sqr != 3) return false;    // TX_SET_WIDE/HIGH_32
  if (sqr_up == 3) return false;                // intra TX_SET_DCTONLY
  return true;                                  // TX_SET_INTRA_1
}

TxPlan tx_plan(const FrameGeometry& geometry, const Node& leaf, int eob_luma) {
  TxPlan plan;
  plan.luma_tx = av2enc::still::kMaxTxSizeRect[static_cast<int>(leaf.size)];
  plan.luma_eob_multisize = eob_multisize(plan.luma_tx);
  plan.luma_tx_type_coded = eob_luma > 1 && tx_type_set1(plan.luma_tx);
  plan.luma_tx_type_set1 = tx_type_set1(plan.luma_tx);
  if (leaf.has_chroma) {
    const BlockSize residual = chroma_residual_size(geometry, leaf.chroma);
    if (partition::is_valid_block_size(residual)) {
      plan.chroma_tx = av2enc::still::kMaxTxSizeRect[static_cast<int>(residual)];
      plan.chroma_eob_multisize = eob_multisize(plan.chroma_tx);
    }
  }
  // M1 uses TX_MODE_LARGEST; tx_mode_select would add tx_do_partition on
  // 4x8/8x4 (max dimension < 32).  Keep the flag false so no caller assumes
  // transform-tree syntax is present.
  plan.codes_tx_partition = false;
  return plan;
}

}  // namespace av2enc::still::sub8
