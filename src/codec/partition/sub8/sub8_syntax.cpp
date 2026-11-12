#include "codec/partition/sub8/sub8_syntax.h"

#include "pipeline/still/basic_cdf_tables.h"
#include "pipeline/still/still_cdf_tables.h"
#include "pipeline/still/still_tables.h"
#include "codec/entropy/qctx/qctx.h"
#include "codec/entropy/av2/av2_entropy.h"
#include "codec/transform/variants/tx_variants.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <vector>

namespace av2enc::still::sub8 {
namespace {

constexpr int kDcPred = 0;
constexpr int kSentinelLog2 = 6;
constexpr int kNonDirectionalModes = 5;
constexpr std::size_t kTileCapacityBytes = 1U << 20;

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

int tx_sz_ctx(int tx_size) {
  return (av2enc::still::kTxSizeSqr[tx_size] + av2enc::still::kTxSizeSqrUp[tx_size] + 1) >> 1;
}

// §5.20.7.23 / §8.3.2 neighbour state.  Shared by the writer and reader; the
// symbol I/O is kept separate on purpose so the reader is an independent
// implementation of the syntax order.
class Context {
 public:
  Context(const FrameGeometry& geometry, const SyntaxOptions& options)
      : mi_rows_(partition::mi_rows(geometry)),
        mi_cols_(partition::mi_cols(geometry)),
        cdfs_{} {
    init_cdfs(cdfs_, options);
    above_mi_.assign(static_cast<std::size_t>(mi_cols_) + 16, kSentinelLog2);
    left_mi_.assign(static_cast<std::size_t>(mi_rows_) + 16, kSentinelLog2);
    intra_joint_.assign(
        static_cast<std::size_t>(mi_rows_) * static_cast<std::size_t>(mi_cols_),
        0);
    uvcfl_.assign(
        static_cast<std::size_t>(mi_rows_ / 2) *
            static_cast<std::size_t>(mi_cols_ / 2),
        0);
    for (int plane = 0; plane < 3; ++plane) {
      const int cols = plane == 0 ? mi_cols_ : mi_cols_ / 2;
      const int rows = plane == 0 ? mi_rows_ : mi_rows_ / 2;
      const auto index = static_cast<std::size_t>(plane);
      const auto col_count = static_cast<std::size_t>(cols) + 16;
      const auto row_count = static_cast<std::size_t>(rows) + 16;
      above_level_[index].assign(col_count, 0);
      left_level_[index].assign(row_count, 0);
      above_dc_[index].assign(col_count, 0);
      left_dc_[index].assign(row_count, 0);
    }
  }

  CdfState& cdfs() { return cdfs_; }
  int mi_rows() const { return mi_rows_; }
  int mi_cols() const { return mi_cols_; }

  void clear_left() {
    std::fill(left_mi_.begin(), left_mi_.end(), kSentinelLog2);
    for (int plane = 0; plane < 3; ++plane) {
      const auto index = static_cast<std::size_t>(plane);
      std::fill(left_level_[index].begin(), left_level_[index].end(), 0);
      std::fill(left_dc_[index].begin(), left_dc_[index].end(), 0);
    }
  }

  // §8.3.2 do_split / rect_type / do_square_split context.
  int partition_context(BlockSize size, int mi_row, int mi_col,
                        bool rect_type) const {
    const int bsw = std::max(partition::mi_width_log2(size), 1);
    const int bsh = std::max(partition::mi_height_log2(size), 1);
    const int adjust =
        rect_type ? partition::partition_size_adjust_rect_type(size)
                  : partition::partition_size_adjust(size);
    const int left = left_mi_[static_cast<std::size_t>(mi_row)];
    const int above = above_mi_[static_cast<std::size_t>(mi_col)];
    return adjust * 4 + (left < bsh ? 2 : 0) + (above < bsw ? 1 : 0);
  }

  // y_mode_index / y_mode_offset context: the two neighbour joint modes.
  int y_mode_context(int mi_row, int mi_col, BlockSize size) const {
    const int w4 = partition::num_4x4_blocks_wide(size);
    const int h4 = partition::num_4x4_blocks_high(size);
    const int left_row = mi_row + h4 - 1;
    const int left_col = mi_col - 1;
    int ctx = 0;
    if (left_row >= 0 && left_row < mi_rows_ && left_col >= 0 &&
        left_col < mi_cols_) {
      ctx += joint_at(left_row, left_col) >= kNonDirectionalModes ? 1 : 0;
    }
    const int above_row = mi_row - 1;
    const int above_col = mi_col + w4 - 1;
    if (above_row >= 0 && above_row < mi_rows_ && above_col >= 0 &&
        above_col < mi_cols_) {
      ctx += joint_at(above_row, above_col) >= kNonDirectionalModes ? 1 : 0;
    }
    return ctx;
  }

  // is_cfl context at the chroma block origin (§8.3.2).
  int cfl_context(const ChromaInfo& chroma) const {
    if (!chroma.valid) return 0;
    const int row = chroma.mi_row / 2;
    const int col = chroma.mi_col / 2;
    const int cols = mi_cols_ / 2;
    const int rows = mi_rows_ / 2;
    int ctx = 0;
    if (row > 0 && col < cols && uvcfl_at(row - 1, col) != 0) ++ctx;
    if (col > 0 && row < rows && uvcfl_at(row, col - 1) != 0) ++ctx;
    return ctx;
  }

  // OR of the above/left level+dc contexts over a transform footprint.
  int above_context(int plane, int x4, int w4) const {
    const int limit = plane == 0 ? mi_cols_ : mi_cols_ / 2;
    int value = 0;
    for (int i = 0; i < w4; ++i) {
      const int x = x4 + i;
      if (x >= limit) break;
      value |= above_level_[static_cast<std::size_t>(plane)][static_cast<std::size_t>(x)];
      value |= above_dc_[static_cast<std::size_t>(plane)][static_cast<std::size_t>(x)];
    }
    return value;
  }

  int left_context(int plane, int y4, int h4) const {
    const int limit = plane == 0 ? mi_rows_ : mi_rows_ / 2;
    int value = 0;
    for (int i = 0; i < h4; ++i) {
      const int y = y4 + i;
      if (y >= limit) break;
      value |= left_level_[static_cast<std::size_t>(plane)][static_cast<std::size_t>(y)];
      value |= left_dc_[static_cast<std::size_t>(plane)][static_cast<std::size_t>(y)];
    }
    return value;
  }

  // Post-leaf updates.  All contexts are zero (EOB 0 / DC), but the footprint
  // must still be cleared exactly like coeffs()/decode_block().
  void update_leaf(const FrameGeometry& geometry, const Node& leaf) {
    const int w4 = partition::num_4x4_blocks_wide(leaf.size);
    const int h4 = partition::num_4x4_blocks_high(leaf.size);
    for (int y = 0; y < h4; ++y) {
      const int row = leaf.mi_row + y;
      if (row >= mi_rows_) break;
      left_mi_[static_cast<std::size_t>(row)] =
          partition::mi_height_log2(leaf.size);
      for (int x = 0; x < w4; ++x) {
        const int col = leaf.mi_col + x;
        if (col >= mi_cols_) break;
        intra_joint_[static_cast<std::size_t>(row) *
                         static_cast<std::size_t>(mi_cols_) +
                     static_cast<std::size_t>(col)] = kDcPred;
      }
    }
    for (int x = 0; x < w4; ++x) {
      const int col = leaf.mi_col + x;
      if (col >= mi_cols_) break;
      above_mi_[static_cast<std::size_t>(col)] =
          partition::mi_width_log2(leaf.size);
    }

    const TxPlan plan = tx_plan(geometry, leaf, 0);
    const auto luma =
        txv::make_tx_params(static_cast<txv::TxSize>(plan.luma_tx),
                            txv::TxType::kDctDct);
    if (luma.has_value()) {
      clear_tx_context(0, leaf.mi_col, leaf.mi_row,
                       luma->coded_w >> 2, luma->coded_h >> 2);
    }
    if (!leaf.has_chroma) return;
    const BlockSize residual = chroma_residual_size(geometry, leaf.chroma);
    if (!partition::is_valid_block_size(residual)) return;
    const int cx = leaf.chroma.mi_col / 2;
    const int cy = leaf.chroma.mi_row / 2;
    const int cw = partition::block_width(residual) >> 2;
    const int ch = partition::block_height(residual) >> 2;
    const auto chroma =
        txv::make_tx_params(static_cast<txv::TxSize>(plan.chroma_tx),
                            txv::TxType::kDctDct);
    if (chroma.has_value()) {
      const int tw = chroma->coded_w >> 2;
      const int th = chroma->coded_h >> 2;
      clear_tx_context(1, cx, cy, tw, th);
      clear_tx_context(2, cx, cy, tw, th);
    }
    for (int y = 0; y < ch; ++y) {
      const int row = cy + y;
      if (row >= mi_rows_ / 2) break;
      for (int x = 0; x < cw; ++x) {
        const int col = cx + x;
        if (col >= mi_cols_ / 2) break;
        uvcfl_[static_cast<std::size_t>(row) *
                   static_cast<std::size_t>(mi_cols_ / 2) +
               static_cast<std::size_t>(col)] = 0;
      }
    }
  }

 private:
  int joint_at(int row, int col) const {
    return intra_joint_[static_cast<std::size_t>(row) *
                            static_cast<std::size_t>(mi_cols_) +
                        static_cast<std::size_t>(col)];
  }
  int uvcfl_at(int row, int col) const {
    return uvcfl_[static_cast<std::size_t>(row) *
                      static_cast<std::size_t>(mi_cols_ / 2) +
                  static_cast<std::size_t>(col)];
  }
  void clear_tx_context(int plane, int x4, int y4, int w4, int h4) {
    const int col_limit = plane == 0 ? mi_cols_ : mi_cols_ / 2;
    const int row_limit = plane == 0 ? mi_rows_ : mi_rows_ / 2;
    for (int i = 0; i < w4; ++i) {
      const int x = x4 + i;
      if (x >= col_limit) break;
      above_level_[static_cast<std::size_t>(plane)][static_cast<std::size_t>(x)] = 0;
      above_dc_[static_cast<std::size_t>(plane)][static_cast<std::size_t>(x)] = 0;
    }
    for (int i = 0; i < h4; ++i) {
      const int y = y4 + i;
      if (y >= row_limit) break;
      left_level_[static_cast<std::size_t>(plane)][static_cast<std::size_t>(y)] = 0;
      left_dc_[static_cast<std::size_t>(plane)][static_cast<std::size_t>(y)] = 0;
    }
  }

  int mi_rows_ = 0;
  int mi_cols_ = 0;
  CdfState cdfs_{};
  std::vector<int> above_mi_;
  std::vector<int> left_mi_;
  std::vector<int> intra_joint_;
  std::vector<std::uint8_t> uvcfl_;
  std::array<std::vector<std::uint8_t>, 3> above_level_;
  std::array<std::vector<std::uint8_t>, 3> left_level_;
  std::array<std::vector<std::uint8_t>, 3> above_dc_;
  std::array<std::vector<std::uint8_t>, 3> left_dc_;
};

// The syntax elements that precede a leaf, derived once so the writer and
// reader agree on what is present.
struct LeafSyntax {
  bool has_chroma = false;
  int y_mode_ctx = 0;
  bool code_is_cfl = false;
  int cfl_ctx = 0;
  int uv_mode_ctx = 0;
  int luma_skip_ctx = 0;
  int u_skip_ctx = 0;
  int v_skip_ctx = 0;
  int luma_tsc = 0;
  int chroma_tsc = 0;
};

LeafSyntax leaf_syntax(const FrameGeometry& geometry, const Context& context,
                       const Node& leaf, const SyntaxOptions& options) {
  LeafSyntax syntax;
  syntax.has_chroma = leaf.has_chroma;
  syntax.y_mode_ctx =
      context.y_mode_context(leaf.mi_row, leaf.mi_col, leaf.size);
  const ModeInfoPlan mode = mode_info_plan(geometry, leaf, options.enable_cfl);
  syntax.code_is_cfl = mode.codes_uv_mode && mode.cfl_allowed;
  syntax.cfl_ctx = context.cfl_context(leaf.chroma);
  syntax.uv_mode_ctx = 0;  // YMode DC is non-directional
  const TxPlan tx = tx_plan(geometry, leaf, 0);
  syntax.luma_tsc = tx_sz_ctx(tx.luma_tx);
  const BlockSize luma_bsize = leaf.size;
  const auto luma_params =
      txv::make_tx_params(static_cast<txv::TxSize>(tx.luma_tx),
                          txv::TxType::kDctDct);
  int luma_lw = 0;
  int luma_lh = 0;
  int luma_w4 = 0;
  int luma_h4 = 0;
  if (luma_params.has_value()) {
    luma_lw = luma_params->log2_w;
    luma_lh = luma_params->log2_h;
    luma_w4 = luma_params->coded_w >> 2;
    luma_h4 = luma_params->coded_h >> 2;
  }
  syntax.luma_skip_ctx =
      txb_skip_context(0, luma_bsize, luma_lw, luma_lh,
                       context.above_context(0, leaf.mi_col, luma_w4),
                       context.left_context(0, leaf.mi_row, luma_h4), false);
  if (leaf.has_chroma) {
    const BlockSize residual = chroma_residual_size(geometry, leaf.chroma);
    syntax.chroma_tsc = tx_sz_ctx(tx.chroma_tx);
    const auto chroma_params =
        txv::make_tx_params(static_cast<txv::TxSize>(tx.chroma_tx),
                            txv::TxType::kDctDct);
    int cw = 0;
    int ch = 0;
    int cw4 = 0;
    int ch4 = 0;
    if (chroma_params.has_value()) {
      cw = chroma_params->log2_w;
      ch = chroma_params->log2_h;
      cw4 = chroma_params->coded_w >> 2;
      ch4 = chroma_params->coded_h >> 2;
    }
    const int cx = leaf.chroma.mi_col / 2;
    const int cy = leaf.chroma.mi_row / 2;
    syntax.u_skip_ctx =
        txb_skip_context(1, residual, cw, ch,
                         context.above_context(1, cx, cw4),
                         context.left_context(1, cy, ch4), false);
    syntax.v_skip_ctx =
        txb_skip_context(2, residual, cw, ch,
                         context.above_context(2, cx, cw4),
                         context.left_context(2, cy, ch4), false);
  }
  return syntax;
}

}  // namespace

void init_cdfs(CdfState& cdfs, const SyntaxOptions& options) {
  copy_array(cdfs.do_split, av2enc::basic::kDoSplitCdf);
  copy_array(cdfs.rect_type, av2enc::basic::kRectTypeCdf);
  copy_array(cdfs.y_mode_set, av2enc::still::kYModeSetCdf);
  copy_array(cdfs.y_mode_index, av2enc::still::kYModeIndexCdf);
  copy_array(cdfs.uv_mode, av2enc::still::kUvModeCflNotAllowedCdf);
  copy_array(cdfs.is_cfl, av2enc::still::kIsCflCdf);
  const auto& q = qctx::tables_for_base_q_idx(options.base_q_idx);
  copy_array(cdfs.txb_skip, *q.txb_skip);
  copy_array(cdfs.v_txb_skip, *q.v_txb_skip);
}

bool write_tile(const FrameGeometry& geometry,
                std::span<const Node> superblocks,
                const SyntaxOptions& options, std::vector<std::uint8_t>& tile,
                std::string& error) {
  if (geometry.sb_size != 64) {
    error = "sub8 writer only supports 64x64 superblocks";
    return false;
  }
  Context context(geometry, options);
  std::vector<std::uint8_t> buffer(kTileCapacityBytes);
  av2::Encoder encoder;
  av2::Status status = encoder.init_symbol(buffer.data(), buffer.size());
  if (status != av2::Status::kOk) {
    error = "entropy init failed";
    return false;
  }
  encoder.disable_cdf_update = options.disable_cdf_update ? 1 : 0;

  const auto sym = [&](std::uint16_t* cdf, int n, int value) {
    return encoder.encode_symbol(cdf, n, value) == av2::Status::kOk;
  };

  // Recursive writer.  The tree is already validated by build_superblock().
  std::function<bool(const Node&)> write_node = [&](const Node& node) -> bool {
    const PartitionPlan plan =
        plan_partition(geometry, node.mi_row, node.mi_col, node.size,
                       node.has_chroma, node.chroma_offset);
    if (plan.forced != node.forced) {
      error = "node forced flag is stale";
      return false;
    }
    if (!plan.forced) {
      if (plan.none_allowed) {
        const int ctx = context.partition_context(
            node.size, node.mi_row, node.mi_col, false);
        if (!sym(&context.cdfs().do_split[0][ctx][0], 2,
                 node.partition == PartitionType::kNone ? 0 : 1)) {
          error = "do_split failed";
          return false;
        }
        if (node.partition == PartitionType::kNone) {
          // fall through to the leaf
        }
      }
      if (node.partition != PartitionType::kNone) {
        if (plan.codes_do_square_split) {
          // PARTITION_SPLIT only exists at a 128x128/256x256 root, which this
          // 64-superblock writer deliberately does not model.
          error = "PARTITION_SPLIT is not supported by this writer";
          return false;
        }
        if (plan.codes_rect_type) {
          const int ctx = context.partition_context(
              node.size, node.mi_row, node.mi_col, true);
          if (!sym(&context.cdfs().rect_type[0][ctx][0], 2,
                   node.partition == PartitionType::kVert ? 1 : 0)) {
            error = "rect_type failed";
            return false;
          }
        }
      }
    }

    if (node.partition == PartitionType::kNone) {
      const LeafSyntax syntax = leaf_syntax(geometry, context, node, options);
      if (!sym(&context.cdfs().y_mode_set[0], 4, 0)) {
        error = "y_mode_set failed";
        return false;
      }
      if (!sym(&context.cdfs().y_mode_index[syntax.y_mode_ctx][0], 8, 0)) {
        error = "y_mode_index failed";
        return false;
      }
      if (syntax.has_chroma) {
        if (syntax.code_is_cfl &&
            !sym(&context.cdfs().is_cfl[syntax.cfl_ctx][0], 2, 0)) {
          error = "is_cfl failed";
          return false;
        }
        if (!sym(&context.cdfs().uv_mode[syntax.uv_mode_ctx][0], 8, 0)) {
          error = "uv_mode failed";
          return false;
        }
      }
      if (!sym(&context.cdfs().txb_skip[0][syntax.luma_tsc]
                            [syntax.luma_skip_ctx][0],
               2, 1)) {
        error = "luma all_zero failed";
        return false;
      }
      if (syntax.has_chroma) {
        if (!sym(&context.cdfs().txb_skip[0][syntax.chroma_tsc]
                              [syntax.u_skip_ctx][0],
                 2, 1)) {
          error = "U all_zero failed";
          return false;
        }
        if (!sym(&context.cdfs().v_txb_skip[syntax.v_skip_ctx][0], 2, 1)) {
          error = "V all_zero failed";
          return false;
        }
      }
      context.update_leaf(geometry, node);
      return true;
    }

    for (const Node& child : node.children) {
      if (!write_node(child)) return false;
    }
    return true;
  };

  if (superblocks.empty()) {
    error = "no superblocks to write";
    return false;
  }
  const int sb_mi = partition::sb_size_mi(geometry);
  const int sb_cols = (context.mi_cols() + sb_mi - 1) / sb_mi;
  const int sb_rows = (context.mi_rows() + sb_mi - 1) / sb_mi;
  if (static_cast<int>(superblocks.size()) != sb_cols * sb_rows) {
    error = "expected " + std::to_string(sb_cols * sb_rows) +
            " superblocks, got " + std::to_string(superblocks.size());
    return false;
  }
  int index = 0;
  for (const Node& sb : superblocks) {
    const int expect_row = (index / sb_cols) * sb_mi;
    const int expect_col = (index % sb_cols) * sb_mi;
    if (sb.mi_row != expect_row || sb.mi_col != expect_col) {
      error = "superblocks are not in raster order";
      return false;
    }
    if (sb.mi_col == 0) context.clear_left();
    if (!write_node(sb)) return false;
    ++index;
  }
  std::size_t size = 0;
  status = encoder.exit_symbol(&size);
  if (status != av2::Status::kOk) {
    error = std::string("entropy exit failed: ") + av2::status_name(status);
    return false;
  }
  tile.assign(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(size));
  return true;
}

bool read_tile(const FrameGeometry& geometry, std::span<const std::uint8_t> tile,
               const SyntaxOptions& options,
               std::vector<DecodedLeaf>& leaves, std::string& error) {
  if (geometry.sb_size != 64) {
    error = "sub8 reader only supports 64x64 superblocks";
    return false;
  }
  Context context(geometry, options);
  av2::Decoder decoder;
  av2::Status status =
      decoder.init_symbol(tile.data(), tile.size(), tile.size());
  if (status != av2::Status::kOk) {
    error = "entropy init failed";
    return false;
  }
  decoder.disable_cdf_update = options.disable_cdf_update ? 1 : 0;

  const auto sym = [&](std::uint16_t* cdf, int n, int* value) {
    return decoder.decode_symbol(cdf, n, value) == av2::Status::kOk;
  };

  std::function<bool(int, int, BlockSize, bool, bool, const ChromaInfo&)> read_node;
  read_node = [&](int mi_row, int mi_col, BlockSize size, bool has_chroma,
                  bool chroma_offset, const ChromaInfo& inherited) -> bool {
    const PartitionPlan plan =
        plan_partition(geometry, mi_row, mi_col, size, has_chroma,
                       chroma_offset);
    PartitionType partition = plan.forced_partition;
    if (!plan.forced) {
      bool do_split = true;  // implied 1 when PARTITION_NONE is not allowed
      if (plan.none_allowed) {
        int value = 0;
        const int ctx =
            context.partition_context(size, mi_row, mi_col, false);
        if (!sym(&context.cdfs().do_split[0][ctx][0], 2, &value)) {
          error = "do_split read failed";
          return false;
        }
        do_split = value != 0;
      }
      partition = PartitionType::kNone;
      if (do_split) {
        if (plan.codes_do_square_split) {
          // read_partition() would read do_square_split here; this subset
          // rejects it instead of guessing a CDF row.
          error = "PARTITION_SPLIT is not supported by this reader";
          return false;
        }
        if (plan.codes_rect_type) {
          const int ctx =
              context.partition_context(size, mi_row, mi_col, true);
          int rect = 0;
          if (!sym(&context.cdfs().rect_type[0][ctx][0], 2, &rect)) {
            error = "rect_type read failed";
            return false;
          }
          partition =
              rect != 0 ? PartitionType::kVert : PartitionType::kHorz;
        } else if (plan.rect_type.implied) {
          partition = plan.rect_type.partition;
        } else if (plan.allow_horz) {
          partition = PartitionType::kHorz;
        } else {
          partition = PartitionType::kVert;
        }
      }
    }

    // ChromaMi* capture at this node (§5.20.3.1).
    ChromaInfo chroma = inherited;
    bool offset = chroma_offset;
    if (has_chroma && !offset) {
      offset = partition::is_chroma_offset_for_partition(geometry, partition,
                                                         size);
      chroma.valid = true;
      chroma.mi_row = mi_row;
      chroma.mi_col = mi_col;
      chroma.size = size;
    }

    if (partition == PartitionType::kNone) {
      Node leaf;
      leaf.mi_row = mi_row;
      leaf.mi_col = mi_col;
      leaf.size = size;
      leaf.partition = PartitionType::kNone;
      leaf.has_chroma = has_chroma;
      leaf.chroma_offset = offset;
      leaf.chroma = chroma;
      const LeafSyntax syntax = leaf_syntax(geometry, context, leaf, options);
      int y_set = 0;
      if (!sym(&context.cdfs().y_mode_set[0], 4, &y_set)) {
        error = "y_mode_set read failed";
        return false;
      }
      if (y_set != 0) {
        error = "only y_mode_set 0 (DC) is supported";
        return false;
      }
      int y_index = 0;
      if (!sym(&context.cdfs().y_mode_index[syntax.y_mode_ctx][0], 8,
               &y_index)) {
        error = "y_mode_index read failed";
        return false;
      }
      if (y_index != 0) {
        error = "only DC_PRED is supported";
        return false;
      }
      DecodedLeaf decoded;
      decoded.mi_row = mi_row;
      decoded.mi_col = mi_col;
      decoded.size = size;
      decoded.has_chroma = has_chroma;
      decoded.chroma = chroma;
      decoded.y_mode = kDcPred;
      if (has_chroma) {
        int is_cfl = 0;
        if (syntax.code_is_cfl &&
            !sym(&context.cdfs().is_cfl[syntax.cfl_ctx][0], 2, &is_cfl)) {
          error = "is_cfl read failed";
          return false;
        }
        if (is_cfl != 0) {
          error = "CfL is not supported by this reader";
          return false;
        }
        int uv = 0;
        if (!sym(&context.cdfs().uv_mode[syntax.uv_mode_ctx][0], 8, &uv)) {
          error = "uv_mode read failed";
          return false;
        }
        if (uv != 0) {
          error = "only DC_PRED chroma is supported";
          return false;
        }
        decoded.is_cfl = false;
        decoded.uv_mode = 0;
      }
      int luma_skip = 0;
      if (!sym(&context.cdfs().txb_skip[0][syntax.luma_tsc]
                            [syntax.luma_skip_ctx][0],
               2, &luma_skip)) {
        error = "luma all_zero read failed";
        return false;
      }
      decoded.luma_all_zero = luma_skip != 0;
      if (has_chroma) {
        int u_skip = 0;
        int v_skip = 0;
        if (!sym(&context.cdfs().txb_skip[0][syntax.chroma_tsc]
                              [syntax.u_skip_ctx][0],
                 2, &u_skip)) {
          error = "U all_zero read failed";
          return false;
        }
        if (!sym(&context.cdfs().v_txb_skip[syntax.v_skip_ctx][0], 2,
                 &v_skip)) {
          error = "V all_zero read failed";
          return false;
        }
        decoded.u_all_zero = u_skip != 0;
        decoded.v_all_zero = v_skip != 0;
      }
      if (!decoded.luma_all_zero ||
          (has_chroma && (!decoded.u_all_zero || !decoded.v_all_zero))) {
        error = "non-zero residual is not supported by this reader";
        return false;
      }
      context.update_leaf(geometry, leaf);
      leaves.push_back(decoded);
      return true;
    }

    const BlockSize subsize = partition::partition_subsize(partition, size);
    if (!partition::is_valid_block_size(subsize)) {
      error = "partition has no subsize";
      return false;
    }
    const int w4 = partition::num_4x4_blocks_wide(size);
    const int h4 = partition::num_4x4_blocks_high(size);
    const int half_w = w4 >> 1;
    const int half_h = h4 >> 1;
    const int mi_rows = context.mi_rows();
    const int mi_cols = context.mi_cols();
    const auto recurse = [&](int r, int c, bool child_chroma, bool child_offset) {
      if (r >= mi_rows || c >= mi_cols) return true;
      return read_node(r, c, subsize, child_chroma, child_offset, chroma);
    };
    switch (partition) {
      case PartitionType::kSplit:
        if (!recurse(mi_row, mi_col, has_chroma, false)) return false;
        if (!recurse(mi_row, mi_col + half_w, has_chroma, false)) return false;
        if (!recurse(mi_row + half_h, mi_col, has_chroma, false)) return false;
        if (!recurse(mi_row + half_h, mi_col + half_w, has_chroma, false)) {
          return false;
        }
        break;
      case PartitionType::kHorz:
        if (!recurse(mi_row, mi_col, has_chroma && !offset, offset)) {
          return false;
        }
        if (!recurse(mi_row + half_h, mi_col, has_chroma, offset)) {
          return false;
        }
        break;
      case PartitionType::kVert:
        if (!recurse(mi_row, mi_col, has_chroma && !offset, offset)) {
          return false;
        }
        if (!recurse(mi_row, mi_col + half_w, has_chroma, offset)) {
          return false;
        }
        break;
      case PartitionType::kNone:
        break;
    }
    return true;
  };

  const int sb_mi = partition::sb_size_mi(geometry);
  const int sb_cols =
      (context.mi_cols() + sb_mi - 1) / sb_mi;
  const int sb_rows =
      (context.mi_rows() + sb_mi - 1) / sb_mi;
  for (int sb_row = 0; sb_row < sb_rows; ++sb_row) {
    for (int sb_col = 0; sb_col < sb_cols; ++sb_col) {
      if (sb_col == 0) context.clear_left();
      if (!read_node(sb_row * sb_mi, sb_col * sb_mi, BlockSize::k64x64,
                     true, false, ChromaInfo{})) {
        return false;
      }
    }
  }
  status = decoder.exit_symbol();
  if (status != av2::Status::kOk) {
    error = std::string("entropy exit failed: ") + av2::status_name(status);
    return false;
  }
  return true;
}

}  // namespace av2enc::still::sub8
