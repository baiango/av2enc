// Minimal M1 + MRL stream writer.  See mrl_stream.h.

#include "encoder/m1/mrl_stream/mrl_stream.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "bitstream/m0_headers.h"
#include "encoder/intra_modes/intra_modes.h"
#include "encoder/m1/m1_cdf_tables.h"
#include "encoder/m1/m1_tables.h"
#include "encoder/m1/mrl_stream/mrl_interaction.h"
#include "encoder/m1/mrl_stream/mrl_syntax.h"
#include "encoder/m1/partition_stream/partition_stream.h"
#include "encoder/m1/qctx/qctx.h"
#include "encoder/mrl/mrl.h"
#include "encoder/partition/partition.h"
#include "encoder/partition_ext/partition_ext.h"
#include "encoder/tx_variants/tx_variants.h"
#include "entropy/av2/av2_entropy.h"

namespace av2enc::m1::mrl_stream {
namespace {

namespace m0 = av2enc::m0;
namespace p = av2enc::partition;
namespace pe = av2enc::partition_ext;
namespace ps = av2enc::m1::partition_stream;
namespace txv = av2enc::txv;

constexpr int kDcPred = 0;
constexpr int kIntraModesCount = 13;

// Reordered_Y_Mode[INTRA_MODES] (§5.20.5.5).
constexpr int kReorderedY[kIntraModesCount] = {
    kDcPred, 9, 10, 11, 12, 3, 8, 1, 5, 4, 6, 2, 7};

// Default_Mode_List_Y[DIRECTIONAL_MODES_COUNT] (§5.20.5.5).
constexpr int kDefaultModeListY[56] = {
    17, 45, 3,  10, 24, 31, 38, 52, 15, 19, 43, 47, 1,  5,  8,  12,
    22, 26, 29, 33, 36, 40, 50, 54, 16, 18, 44, 46, 2,  4,  9,  11,
    23, 25, 30, 32, 37, 39, 51, 53, 14, 20, 42, 48, 0,  6,  7,  13,
    21, 27, 28, 34, 35, 41, 49, 55};

// Default_Mode_List_Uv[UV_INTRA_MODES_CFL_NOT_ALLOWED] (§5.20.5.6).
constexpr int kDefaultModeListUv[kIntraModesCount] = {
    kDcPred, 9, 10, 11, 12, 1, 2, 3, 4, 8, 5, 6, 7};

bool y_mode_is_directional(int mode) { return mode >= 1 && mode <= 8; }

int reordered_index(int mode) {
  for (int i = 0; i < kIntraModesCount; ++i) {
    if (kReorderedY[i] == mode) return i;
  }
  return 0;
}

struct YModeCode {
  int joint = 0;
  int mode_index = 0;
};

YModeCode y_mode_code(int mode, int angle_delta) {
  YModeCode out;
  const int r = reordered_index(mode);
  if (r < 5) {
    out.joint = r;
    out.mode_index = r;
    return out;
  }
  const int base = r - 5;
  const int dir_index = base * 7 + (angle_delta + 3);
  out.joint = 5 + dir_index;
  int pos = 0;
  for (int i = 0; i < 56; ++i) {
    if (kDefaultModeListY[i] == dir_index) {
      pos = i;
      break;
    }
  }
  out.mode_index = 5 + pos;
  return out;
}

int uv_mode_index(int uv_mode, int y_mode) {
  int index = 0;
  if (y_mode_is_directional(y_mode)) {
    if (uv_mode == y_mode) return 0;
    index = 1;
  }
  for (int i = 0; i < kIntraModesCount; ++i) {
    const int mode = kDefaultModeListUv[i];
    if (mode == y_mode && y_mode_is_directional(y_mode)) continue;
    if (mode == uv_mode) return index;
    ++index;
  }
  return 0;
}

int tx_sz_ctx(int tx_size) {
  return (kTxSizeSqr[tx_size] + kTxSizeSqrUp[tx_size] + 1) >> 1;
}

struct ModeCdfs {
  std::uint16_t y_mode_set[5];
  std::uint16_t y_mode_index[3][9];
  std::uint16_t y_mode_offset[3][7];
  std::uint16_t uv_mode[2][9];
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

void init_mode_cdfs(ModeCdfs& c, int base_q_idx) {
  const qctx::CoeffCdfSlice& q = qctx::tables_for_slice(
      qctx::slice_for_base_q_idx(base_q_idx));
  copy_array(c.y_mode_set, kYModeSetCdf);
  copy_array(c.y_mode_index, kYModeIndexCdf);
  copy_array(c.y_mode_offset, kYModeOffsetCdf);
  copy_array(c.uv_mode, kUvModeCflNotAllowedCdf);
  copy_array(c.txb_skip, *q.txb_skip);
  copy_array(c.v_txb_skip, *q.v_txb_skip);
}

int get_joint_mode(const std::vector<int>& joint, int mi_rows, int mi_cols,
                   int r, int c, int w4, int h4, int dir) {
  int mv_row = 0;
  int mv_col = 0;
  if (dir != 0) {
    mv_row = r - 1;
    mv_col = c + w4 - 1;
  } else {
    mv_col = c - 1;
    mv_row = r + h4 - 1;
  }
  if (mv_row >= 0 && mv_col >= 0 && mv_row < mi_rows && mv_col < mi_cols) {
    return joint[static_cast<std::size_t>(mv_row) *
                     static_cast<std::size_t>(mi_cols) +
                 static_cast<std::size_t>(mv_col)];
  }
  return 0;
}

int joint_context(const std::vector<int>& joint, int mi_rows, int mi_cols,
                  int r, int c, int w4, int h4) {
  return (get_joint_mode(joint, mi_rows, mi_cols, r, c, w4, h4, 0) >= 5 ? 1
                                                                        : 0) +
         (get_joint_mode(joint, mi_rows, mi_cols, r, c, w4, h4, 1) >= 5 ? 1
                                                                        : 0);
}

// §5.20.5.5 get_intra_y_mode_set inversion (identical to m1_syntax.cpp).
int y_mode_index_for(const std::vector<int>& joint, int mi_rows, int mi_cols,
                     int r, int c, int w4, int h4, int target) {
  int mode_index = 5;
  bool selected[56] = {};
  int dir_modes[2] = {0, 0};
  int count = 0;
  const int w = w4 * 4;
  const int h = h4 * 4;
  if (w >= 8 && h >= 8) {
    for (int dir = 0; dir < 2; ++dir) {
      int mode = get_joint_mode(joint, mi_rows, mi_cols, r, c, w4, h4, dir);
      if (mode >= 5) {
        mode -= 5;
        if (count == 0 || mode != dir_modes[0]) {
          if (mode == target) return mode_index;
          ++mode_index;
          selected[mode] = true;
          dir_modes[count++] = mode;
        }
      }
    }
    if (w * h > 64) {
      for (int i = 1; i <= 4; ++i) {
        for (int j = 0; j < count; ++j) {
          for (int sgn = -1; sgn <= 1; sgn += 2) {
            int mode = dir_modes[j] + i * sgn;
            if (mode < 0) {
              mode += 56;
            } else if (mode >= 56) {
              mode -= 56;
            }
            if (!selected[mode]) {
              if (mode == target) return mode_index;
              ++mode_index;
              selected[mode] = true;
            }
          }
        }
      }
    }
  }
  for (int i = 0; i < 56; ++i) {
    const int mode = kDefaultModeListY[i];
    if (!selected[mode]) {
      if (mode == target) return mode_index;
      ++mode_index;
    }
  }
  return mode_index;
}

class TileWriter {
 public:
  TileWriter(const p::FrameGeometry& geometry, const pe::Tools& tools,
             const MinimalMrlConfig& config, int mi_rows, int mi_cols,
             int sb_mi)
      : geometry_(geometry),
        tools_(tools),
        config_(config),
        mi_rows_(mi_rows),
        mi_cols_(mi_cols),
        mrl_ctx_(mi_rows, mi_cols, sb_mi),
        intra_joint_(static_cast<std::size_t>(mi_rows) *
                         static_cast<std::size_t>(mi_cols),
                     0) {
    ps::init_tile_partition_cdfs(partition_cdfs_);
    init_mode_cdfs(cdfs_, config.base_q_idx);
    init_mrl_cdfs(mrl_cdfs_);
    buffer_.resize(kTileCapacityBytes);
    encoder_.init_symbol(buffer_.data(), buffer_.size());
    encoder_.disable_cdf_update = config.disable_cdf_update ? 1 : 0;
  }

  bool run(pe::FrameDecision& frame, MinimalMrlFrame& out) {
    ps::build_frame_stream(geometry_, tools_, frame, stream_, out.error);
    if (!out.error.empty()) return false;

    p::MiSizeGrid grid(geometry_);
    pe::SyntaxStats stats{};
    std::size_t leaf_index = 0;
    int node_index = 0;
    for (const ps::M1Node& node : stream_.nodes) {
      const ps::NodePlan plan = ps::plan_node(geometry_, tools_, node, grid);
      if (!plan.ok) {
        out.error = "plan failed at node " + std::to_string(node_index) + ": " +
                    plan.error;
        return false;
      }
      if (!ps::emit_symbols(plan.symbols, partition_cdfs_, encoder_, &stats,
                            out.error)) {
        return false;
      }
      if (node.partition != static_cast<std::uint8_t>(pe::PartitionType::kNone)) {
        ++node_index;
        continue;
      }
      const auto bsize = static_cast<p::BlockSize>(node.bsize);
      const int x = node.mi_col * 4;
      const int y = node.mi_row * 4;
      const int w = p::block_width(bsize);
      const int h = p::block_height(bsize);
      MinimalMrlLeaf leaf;
      if (!config_.leaves.empty()) {
        if (leaf_index >= config_.leaves.size()) {
          out.error = "more leaves than explicit decisions";
          return false;
        }
        leaf = config_.leaves[leaf_index];
      } else {
        leaf = default_leaf(leaf_index, x, y, w, h);
      }
      if (!encode_leaf(node, leaf, out.error)) return false;

      MinimalMrlLeafRecord record;
      record.mi_row = node.mi_row;
      record.mi_col = node.mi_col;
      record.bsize = node.bsize;
      record.x = x;
      record.y = y;
      record.w = w;
      record.h = h;
      record.has_chroma = node.has_chroma;
      record.leaf = leaf;
      out.leaves.push_back(record);
      if (leaf.mrl_index > 0) ++out.mrl_blocks;
      if (leaf.mrl_sec_index) ++out.secondary_blocks;
      ++leaf_index;

      grid.fill_block(node.mi_row, node.mi_col, bsize);
      ++node_index;
    }
    if (!config_.leaves.empty() && leaf_index != config_.leaves.size()) {
      out.error = "explicit decisions do not match the leaf count";
      return false;
    }
    std::size_t payload_bytes = 0;
    const av2::Status status = encoder_.exit_symbol(&payload_bytes);
    if (status != av2::Status::kOk) {
      out.error = std::string("entropy exit_symbol failed: ") +
                  av2::status_name(status);
      return false;
    }
    out.tile_payload.assign(
        buffer_.begin(),
        buffer_.begin() + static_cast<std::ptrdiff_t>(payload_bytes));
    out.blocks = static_cast<int>(leaf_index);
    return true;
  }

 private:
  static constexpr std::size_t kTileCapacityBytes = 16U * 1024U * 1024U;

  static MinimalMrlLeaf default_leaf(std::size_t index, int x, int y, int w,
                                     int h) {
    MinimalMrlLeaf leaf;
    leaf.y_mode = 1;  // V_PRED
    leaf.angle_delta = 0;
    leaf.mrl_index = (x == 0 && y == 0) ? 0 : static_cast<int>(index % 4);
    leaf.mrl_sec_index = leaf.mrl_index > 0 && (index % 2) == 1 &&
                         !(w == 4 && h == 4);
    return leaf;
  }

  bool sym(std::uint16_t* cdf, int n, int symbol) {
    const av2::Status status = encoder_.encode_symbol(cdf, n, symbol);
    if (status != av2::Status::kOk) {
      encoder_error_ = std::string("entropy encode_symbol failed: ") +
                       av2::status_name(status);
      return false;
    }
    return true;
  }

  bool literal(int bits, std::uint32_t value) {
    for (int i = bits - 1; i >= 0; --i) {
      const av2::Status status =
          encoder_.encode_bool(((value >> static_cast<unsigned>(i)) & 1U) != 0);
      if (status != av2::Status::kOk) {
        encoder_error_ = std::string("entropy encode_bool failed: ") +
                         av2::status_name(status);
        return false;
      }
    }
    return true;
  }

  bool write_y_mode(int mode, int angle_delta) {
    const YModeCode code = y_mode_code(mode, angle_delta);
    const int mode_index =
        code.joint < 5
            ? code.mode_index
            : y_mode_index_for(intra_joint_, mi_rows_, mi_cols_, cur_r_,
                               cur_c_, cur_w4_, cur_h4_, code.joint - 5);
    const int ctx = joint_context(intra_joint_, mi_rows_, mi_cols_, cur_r_,
                                  cur_c_, cur_w4_, cur_h4_);
    if (mode_index < kIntraModesCount) {
      if (!sym(cdfs_.y_mode_set, 4, 0)) return false;
      if (mode_index <= 6) return sym(cdfs_.y_mode_index[ctx], 8, mode_index);
      if (!sym(cdfs_.y_mode_index[ctx], 8, 7)) return false;
      return sym(cdfs_.y_mode_offset[ctx], 6, mode_index - 7);
    }
    const int set = 1 + (mode_index - kIntraModesCount) / 16;
    const int second = (mode_index - kIntraModesCount) % 16;
    if (!sym(cdfs_.y_mode_set, 4, set)) return false;
    return literal(4, static_cast<std::uint32_t>(second));
  }

  bool write_skip_tx(int plane, int tx_size, int ctx) {
    const int tsc = tx_sz_ctx(tx_size);
    std::uint16_t* row =
        plane == 2 ? cdfs_.v_txb_skip[ctx]
                   : cdfs_.txb_skip[0][tsc][ctx];
    return sym(row, 2, 1);  // eob == 0
  }

  bool encode_leaf(const ps::M1Node& node, const MinimalMrlLeaf& leaf,
                   std::string& error) {
    const auto bsize = static_cast<p::BlockSize>(node.bsize);
    cur_r_ = node.mi_row;
    cur_c_ = node.mi_col;
    cur_w4_ = p::num_4x4_blocks_wide(bsize);
    cur_h4_ = p::num_4x4_blocks_high(bsize);
    const int w = p::block_width(bsize);
    const int h = p::block_height(bsize);
    int chroma_coded_w = 0;
    int chroma_coded_h = 0;

    if (!write_y_mode(leaf.y_mode, leaf.angle_delta)) {
      error = encoder_error_;
      return false;
    }

    const MrlBlockSyntax mrl_block{node.mi_row, node.mi_col, cur_w4_, cur_h4_};
    const MrlDecision decision{leaf.mrl_index, leaf.mrl_sec_index};
    const MrlSyntaxStatus mrl_status = write_mrl_syntax(
        encoder_, mrl_cdfs_, mrl_ctx_, mrl_block, leaf.y_mode, decision, &error);
    if (mrl_status != MrlSyntaxStatus::kOk) {
      if (error.empty()) error = mrl_syntax_status_name(mrl_status);
      return false;
    }

    if (node.has_chroma) {
      const ps::ChromaPlaneBlock cb =
          ps::chroma_plane_block(geometry_, node.chroma);
      if (!p::is_valid_block_size(cb.residual_size)) {
        error = "invalid chroma residual size";
        return false;
      }
      const int cw = p::block_width(cb.residual_size);
      const int ch = p::block_height(cb.residual_size);
      const int ctx = y_mode_is_directional(leaf.y_mode) ? 1 : 0;
      const int idx = uv_mode_index(kDcPred, leaf.y_mode);
      if (idx <= 6) {
        if (!sym(cdfs_.uv_mode[ctx], 8, idx)) {
          error = encoder_error_;
          return false;
        }
      } else {
        if (!sym(cdfs_.uv_mode[ctx], 8, 7)) {
          error = encoder_error_;
          return false;
        }
        if (!literal(3, static_cast<std::uint32_t>(idx - 7))) {
          error = encoder_error_;
          return false;
        }
      }
      chroma_coded_w = cw;
      chroma_coded_h = ch;
    }

    const auto luma = txv::make_tx_params(w, h);
    if (!luma.has_value()) {
      error = "unsupported luma transform size";
      return false;
    }
    // Plane order is Y, U, V.
    // TX_MODE_LARGEST: coded size == block size, so the luma skip context is 0.
    if (!write_skip_tx(0, luma->tx_size, 0)) {
      error = encoder_error_;
      return false;
    }
    if (node.has_chroma) {
      const auto u = txv::make_tx_params(chroma_coded_w, chroma_coded_h);
      const auto v = txv::make_tx_params(chroma_coded_w, chroma_coded_h);
      if (!u.has_value() || !v.has_value()) {
        error = "unsupported chroma transform size";
        return false;
      }
      // All neighbour levels are zero (every residual is skipped), so the
      // m1_syntax context derivation collapses to U = 6, V = 0 (eob_u = 0).
      if (!write_skip_tx(1, u->tx_size, 6)) {
        error = encoder_error_;
        return false;
      }
      if (!write_skip_tx(2, v->tx_size, 0)) {
        error = encoder_error_;
        return false;
      }
    }

    const YModeCode ycode = y_mode_code(leaf.y_mode, leaf.angle_delta);
    for (int yy = 0; yy < cur_h4_; ++yy) {
      const int rr = cur_r_ + yy;
      if (rr >= mi_rows_) break;
      for (int xx = 0; xx < cur_w4_; ++xx) {
        const int cc = cur_c_ + xx;
        if (cc >= mi_cols_) break;
        intra_joint_[static_cast<std::size_t>(rr) *
                         static_cast<std::size_t>(mi_cols_) +
                     static_cast<std::size_t>(cc)] = ycode.joint;
      }
    }
    return true;
  }

  const p::FrameGeometry& geometry_;
  const pe::Tools& tools_;
  const MinimalMrlConfig& config_;
  int mi_rows_ = 0;
  int mi_cols_ = 0;
  ps::FrameStream stream_{};
  ps::TilePartitionCdfs partition_cdfs_{};
  ModeCdfs cdfs_{};
  MrlCdfs mrl_cdfs_{};
  MrlContext mrl_ctx_;
  std::vector<int> intra_joint_;
  av2::Encoder encoder_{};
  std::vector<std::uint8_t> buffer_;
  std::string encoder_error_;

  int cur_r_ = 0;
  int cur_c_ = 0;
  int cur_w4_ = 0;
  int cur_h4_ = 0;
};

// ---------------------------------------------------------------------------
// Sequence header: exact copy of m1_headers.cpp's writer with enable_mrls.
// ---------------------------------------------------------------------------
void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
  }
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

void write_sequence_header(BitWriter& writer, const M1Config& c, bool enable_mrls) {
  put_uvlc(writer, 0);                        // seq_header_id
  writer.put_bits(0U, 5U);                    // seq_profile_idc
  writer.put_bit(true);                       // single_picture_header_flag
  writer.put_bits(c.seq_level_idx, 5U);       // seq_level_idx
  put_uvlc(writer, 0);                        // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                        // bit_depth_idc = 1 => 8-bit

  unsigned width_bits =
      bit_width(static_cast<std::uint32_t>(c.max_frame_width) - 1U);
  if (width_bits == 0) width_bits = 1;
  unsigned height_bits =
      bit_width(static_cast<std::uint32_t>(c.max_frame_height) - 1U);
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_width) - 1U, width_bits);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_height) - 1U,
                  height_bits);

  writer.put_bit(false);  // seq_cropping_window_present_flag

  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(false);  // use_128x128_superblock
  writer.put_bit(false);  // enable_sdp
  writer.put_bit(false);  // enable_ext_partitions
  writer.put_bit(false);  // reduce_pb_aspect_ratio

  writer.put_bit(false);  // enable_ext_seg
  writer.put_bit(false);  // seq_seg_info_present_flag

  writer.put_bit(false);                     // enable_dip
  writer.put_bit(c.enable_intra_edge_filter);
  writer.put_bit(enable_mrls);               // enable_mrls
  writer.put_bit(c.enable_cfl_intra);
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);    // enable_mhccp
  writer.put_bit(false);    // enable_ibp

  writer.put_bit(false);     // enable_refmvbank
  writer.put_bit(true);      // disable_drl_reorder
  put_ns(writer, 3U, 0U);    // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);     // allow_frame_max_bvp_drl_bits
  writer.put_bit(false);     // enable_bawp

  writer.put_bit(false);                  // enable_fsc
  writer.put_bit(c.enable_idtx_intra);    // enable_idtx_intra
  writer.put_bit(false);                  // enable_intra_ist
  writer.put_bit(false);                  // enable_inter_ist
  writer.put_bit(c.enable_chroma_dctonly);
  writer.put_bit(false);  // reduced_tx_part_set
  writer.put_bit(false);  // enable_cctx
  writer.put_bit(false);  // enable_tcq
  writer.put_bit(false);  // enable_parity_hiding
  writer.put_bit(false);  // separate_uv_delta_q
  writer.put_bit(true);   // equal_ac_dc_q
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q => BaseUVAcDeltaQ = 0
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  writer.put_bit(false);          // disable_loopfilters_across_tiles
  writer.put_bit(c.cdef.enable);  // enable_cdef
  writer.put_bit(false);          // enable_gdf
  writer.put_bit(false);          // enable_restoration
  writer.put_bit(c.ccso.enable);  // enable_ccso
  if (c.ccso.enable) {
    writer.put_bit(true);  // ccso_unit_matches_sb_size
  }
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2

  writer.put_bit(false);  // seq_tile_info_present_flag
  writer.put_bit(false);  // film_grain_params_present
}

// ---------------------------------------------------------------------------
// Partition choice: square leaves of `target` pixels, rect-only.
// ---------------------------------------------------------------------------
pe::CostFunction force_cost(int target) {
  return [target](const pe::NodeInfo& info) {
    pe::NodeCosts costs;
    costs.leaf = 0.0;
    costs.none_signal = pe::kInfiniteCost;
    costs.split_signal = pe::kInfiniteCost;
    costs.horz_signal = pe::kInfiniteCost;
    costs.vert_signal = pe::kInfiniteCost;
    costs.horz3_signal = pe::kInfiniteCost;
    costs.vert3_signal = pe::kInfiniteCost;
    costs.horz4a_signal = pe::kInfiniteCost;
    costs.horz4b_signal = pe::kInfiniteCost;
    costs.vert4a_signal = pe::kInfiniteCost;
    costs.vert4b_signal = pe::kInfiniteCost;

    pe::PartitionType want = pe::PartitionType::kNone;
    if (p::block_width(info.size) > target ||
        p::block_height(info.size) > target) {
      if (info.size == p::BlockSize::k128x128) {
        want = pe::PartitionType::kSplit;
      } else if (p::block_width(info.size) > p::block_height(info.size)) {
        want = pe::PartitionType::kVert;
      } else {
        want = pe::PartitionType::kHorz;
      }
    }
    if (!info.allowed.is_allowed(want)) {
      if (info.boundary.implied &&
          info.allowed.is_allowed(info.boundary.implied_partition)) {
        want = info.boundary.implied_partition;
      } else {
        for (int i = 0; i < pe::kPartitionCount; ++i) {
          const auto candidate = static_cast<pe::PartitionType>(i);
          if (info.allowed.is_allowed(candidate)) {
            want = candidate;
            break;
          }
        }
      }
    }
    switch (want) {
      case pe::PartitionType::kNone: costs.none_signal = 0.0; break;
      case pe::PartitionType::kSplit: costs.split_signal = 0.0; break;
      case pe::PartitionType::kHorz: costs.horz_signal = 0.0; break;
      case pe::PartitionType::kVert: costs.vert_signal = 0.0; break;
      case pe::PartitionType::kHorz3: costs.horz3_signal = 0.0; break;
      case pe::PartitionType::kVert3: costs.vert3_signal = 0.0; break;
      case pe::PartitionType::kHorz4A: costs.horz4a_signal = 0.0; break;
      case pe::PartitionType::kHorz4B: costs.horz4b_signal = 0.0; break;
      case pe::PartitionType::kVert4A: costs.vert4a_signal = 0.0; break;
      case pe::PartitionType::kVert4B: costs.vert4b_signal = 0.0; break;
    }
    return costs;
  };
}

// ---------------------------------------------------------------------------
// Expected reconstruction (loop filters are off in this minimal stream).
// ---------------------------------------------------------------------------
bool reconstruct_luma(const p::FrameGeometry& geometry,
                      const MinimalMrlConfig& config,
                      const std::vector<MinimalMrlLeafRecord>& leaves,
                      std::vector<std::uint8_t>& recon, std::string& error) {
  recon.assign(static_cast<std::size_t>(config.width) *
                   static_cast<std::size_t>(config.height),
               0);
  for (const MinimalMrlLeafRecord& rec : leaves) {
    const std::size_t offset =
        static_cast<std::size_t>(rec.y) *
            static_cast<std::size_t>(config.width) +
        static_cast<std::size_t>(rec.x);
    if (y_mode_is_directional(rec.leaf.y_mode)) {
      mrl::Block block;
      block.recon = intra::PlaneView{recon.data(), config.width};
      block.x = rec.x;
      block.y = rec.y;
      block.w = rec.w;
      block.h = rec.h;
      block.max_x = config.width - 1;
      block.max_y = config.height - 1;
      block.have_left = rec.x > 0;
      block.have_above = rec.y > 0;
      block.num4_above_right = 0;
      block.num4_below_left = 0;
      block.plane = 0;
      block.sb_height = config.sb_size;
      mrl::Params params;
      if (!make_predict_params(rec.leaf.y_mode, rec.leaf.angle_delta,
                               rec.leaf.mrl_index, rec.leaf.mrl_sec_index,
                               config.enable_intra_edge_filter, false, false,
                               &params, rec.w, rec.h)) {
        error = "invalid MRL prediction parameters";
        return false;
      }
      const mrl::MrlStatus status = mrl::predict_block(
          block, 8, params, recon.data() + offset, static_cast<std::size_t>(config.width));
      if (status != mrl::MrlStatus::kOk) {
        error = std::string("MRL reconstruction failed: ") +
                mrl::status_string(status);
        return false;
      }
    } else {
      intra::NeighborRequest request;
      request.recon = intra::PlaneView{recon.data(), config.width};
      request.x = rec.x;
      request.y = rec.y;
      request.w = rec.w;
      request.h = rec.h;
      request.max_x = config.width - 1;
      request.max_y = config.height - 1;
      request.have_left = rec.x > 0;
      request.have_above = rec.y > 0;
      request.num4_above_right = 0;
      request.num4_below_left = 0;
      request.color_plane = 0;
      intra::PredictParams params;
      params.mode = static_cast<intra::Mode>(rec.leaf.y_mode);
      params.angle_delta = rec.leaf.angle_delta;
      params.enable_edge_filter = config.enable_intra_edge_filter;
      const intra::IntraStatus status = intra::predict_block(
          request, 8, params, recon.data() + offset, static_cast<std::size_t>(config.width));
      if (status != intra::IntraStatus::kOk) {
        error = "intra reconstruction failed";
        return false;
      }
    }
  }
  (void)geometry;
  return true;
}

}  // namespace

std::vector<std::uint8_t> make_sequence_header_obu(const M1Config& config,
                                                    bool enable_mrls) {
  BitWriter writer;
  write_sequence_header(writer, config, enable_mrls);
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return m0::make_obu(m0::ObuHeader{m0::ObuType::kSequenceHeader},
                      writer.take_bytes());
}

bool write_minimal_mrl_frame(const MinimalMrlConfig& config,
                             MinimalMrlFrame& out) {
  out = MinimalMrlFrame{};
  if (config.width <= 0 || config.height <= 0 ||
      (config.width & 1) != 0 || (config.height & 1) != 0) {
    out.error = "M1 needs positive, even frame dimensions";
    return false;
  }
  if (config.sb_size != 64) {
    out.error = "the pinned M1 frame header only codes 64x64 superblocks";
    return false;
  }
  if (config.width > 65535 || config.height > 65535) {
    out.error = "frame dimensions must fit max_frame_width/height (16 bit)";
    return false;
  }
  if (config.base_q_idx < 1) {
    out.error = "base_q_idx 0 (lossless) is not implemented";
    return false;
  }
  if (config.enable_cfl_intra) {
    out.error = "the minimal writer codes chroma without CfL; disable it";
    return false;
  }
  if (config.target_block_size < 4 || config.target_block_size > 64 ||
      (config.target_block_size & (config.target_block_size - 1)) != 0) {
    out.error = "target_block_size must be a power of two in 4..64";
    return false;
  }
  if (config.width % config.target_block_size != 0 ||
      config.height % config.target_block_size != 0) {
    out.error = "frame dimensions must be a multiple of the target block size";
    return false;
  }

  p::FrameGeometry geometry;
  geometry.width = config.width;
  geometry.height = config.height;
  geometry.sb_size = config.sb_size;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;

  pe::Tools tools;
  tools.enable_ext_partitions = false;
  tools.enable_uneven_4way_partitions = false;

  pe::DecisionOptions options;
  options.min_block_size_px = config.target_block_size;
  options.max_depth = 12;
  options.allow_rect = true;
  options.allow_split = true;
  options.allow_3way = false;
  options.allow_uneven4 = false;

  pe::FrameDecision frame =
      pe::decide_frame(geometry, force_cost(config.target_block_size), tools,
                       options);
  if (frame.status != pe::DecisionStatus::kOk) {
    out.error = "partition decision failed";
    return false;
  }
  if (!pe::validate_frame(geometry, tools, frame, out.error)) {
    out.error = "partition frame invalid: " + out.error;
    return false;
  }

  M1Config header;
  header.max_frame_width = static_cast<std::uint16_t>(config.width);
  header.max_frame_height = static_cast<std::uint16_t>(config.height);
  header.base_q_idx = config.base_q_idx;
  header.disable_cdf_update = config.disable_cdf_update;
  header.enable_intra_edge_filter = config.enable_intra_edge_filter;
  header.enable_cfl_intra = config.enable_cfl_intra;
  header.enable_chroma_dctonly = true;
  header.enable_idtx_intra = false;
  header.tx_mode_select = false;
  header.reduced_tx_set = 0;
  header.apply_deblock_y_vert = false;
  header.apply_deblock_y_horz = false;
  header.apply_deblock_u = false;
  header.apply_deblock_v = false;
  header.cdef.enable = false;
  header.ccso.enable = false;

  TileWriter writer(geometry, tools, config, p::mi_rows(geometry),
                    p::mi_cols(geometry), p::sb_size_mi(geometry));
  if (!writer.run(frame, out)) return false;

  if (!reconstruct_luma(geometry, config, out.leaves, out.recon_y, out.error)) {
    return false;
  }

  BitWriter bit_writer;
  bit_writer.put_bit(true);  // is_first_tile_group
  av2enc::m1::write_frame_header(bit_writer, header);
  bit_writer.align_to_byte();
  std::vector<std::uint8_t> payload = bit_writer.take_bytes();
  payload.insert(payload.end(), out.tile_payload.begin(), out.tile_payload.end());

  const std::vector<std::uint8_t> clk = m0::make_obu(
      m0::ObuHeader{m0::ObuType::kClosedLoopKey}, payload);
  const std::vector<std::uint8_t> td = m0::make_obu(
      m0::ObuHeader{m0::ObuType::kTemporalDelimiter}, {});
  const std::vector<std::uint8_t> seq =
      make_sequence_header_obu(header, /*enable_mrls=*/true);

  out.temporal_unit.clear();
  out.temporal_unit.reserve(td.size() + seq.size() + clk.size());
  out.temporal_unit.insert(out.temporal_unit.end(), td.begin(), td.end());
  out.temporal_unit.insert(out.temporal_unit.end(), seq.begin(), seq.end());
  out.temporal_unit.insert(out.temporal_unit.end(), clk.begin(), clk.end());
  return true;
}

}  // namespace av2enc::m1::mrl_stream
