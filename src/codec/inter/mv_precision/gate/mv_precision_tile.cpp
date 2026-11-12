#include "codec/inter/mv_precision/gate/mv_precision_tile.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "pipeline/still/basic_cdf_tables.h"
#include "pipeline/still/still_tables.h"
#include "pipeline/video/video_cdf_tables.h"
#include "codec/inter/merge/merge.h"
#include "codec/partition/tree/partition.h"
#include "codec/entropy/av2/av2_entropy.h"

namespace av2enc::inter::mvprec::gate {
namespace {

using partition::BlockSize;
using partition::FrameGeometry;
namespace merge = av2enc::video::merge;

constexpr int kTileCapacityBytes = 16 * 1024 * 1024;

struct Cdfs {
  std::uint16_t do_split[2][64][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t is_inter[4][3];
  std::uint16_t skip[6][3];
  std::uint16_t single_mode[5][4];
  std::uint16_t drl_mode[3][5][3];
  std::uint16_t txb_skip[2][5][10][3];
  std::uint16_t v_txb_skip[12][3];
  MvPrecisionCdfState mv;
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

void init_cdfs(Cdfs* c) {
  copy_array(c->do_split, av2enc::basic::kDoSplitCdf);
  copy_array(c->rect_type, av2enc::basic::kRectTypeCdf);
  copy_array(c->is_inter, av2enc::video::kIsInterCdf);
  copy_array(c->skip, av2enc::video::kSkipCdf);
  std::memcpy(c->single_mode, av2enc::video::kSingleModeCdf,
              sizeof(c->single_mode));
  copy_array(c->drl_mode, av2enc::video::kDrlModeCdf);
  copy_array(c->txb_skip, av2enc::basic::kTxbSkipCdf);
  copy_array(c->v_txb_skip, av2enc::basic::kVTxbSkipCdf);
  init_default_mv_precision_cdfs(&c->mv);
}

BlockSize block_size_from_extent(int w4, int h4) {
  if (w4 == 1 && h4 == 2) return BlockSize::k4x8;
  if (w4 == 2 && h4 == 1) return BlockSize::k8x4;
  if (w4 == 1 && h4 == 1) return BlockSize::k4x4;
  if (w4 == 2 && h4 == 2) return BlockSize::k8x8;
  if (w4 == 4 && h4 == 4) return BlockSize::k16x16;
  if (w4 == 8 && h4 == 8) return BlockSize::k32x32;
  if (w4 == 16 && h4 == 16) return BlockSize::k64x64;
  if (w4 == 2 && h4 == 4) return BlockSize::k8x16;
  if (w4 == 4 && h4 == 2) return BlockSize::k16x8;
  if (w4 == 4 && h4 == 8) return BlockSize::k16x32;
  if (w4 == 8 && h4 == 4) return BlockSize::k32x16;
  if (w4 == 8 && h4 == 16) return BlockSize::k32x64;
  if (w4 == 16 && h4 == 8) return BlockSize::k64x32;
  return BlockSize::kInvalid;
}

int tx_sz_ctx(int tx_size) {
  return (av2enc::still::kTxSizeSqr[tx_size] +
          av2enc::still::kTxSizeSqrUp[tx_size] + 1) >>
         1;
}

class TileWriter {
 public:
  TileWriter(int width, int height, const TileConfig& config)
      : config_(config) {
    geometry_.width = width;
    geometry_.height = height;
    geometry_.sb_size = 64;
    geometry_.has_chroma = true;
    geometry_.chroma_subsampling_x = true;
    geometry_.chroma_subsampling_y = true;
    mi_cols_ = partition::mi_cols(geometry_);
    mi_rows_ = partition::mi_rows(geometry_);
    const std::size_t cells =
        static_cast<std::size_t>(mi_cols_) * static_cast<std::size_t>(mi_rows_);
    mi_.assign(cells, inter::mc::MiInfo{});
    neighbours_.assign(cells, NeighborMvInfo{});
    y_mode_.assign(cells, 0);
    is_inter_.assign(cells, 0);
    skip_.assign(cells, 0);
    for (std::size_t i = 0; i < cells; ++i) {
      mi_[i].ref[0] = merge::kRefNone;
      mi_[i].ref[1] = merge::kRefNone;
      mi_[i].w4 = 1;
      mi_[i].h4 = 1;
      neighbours_[i].available = false;
      neighbours_[i].ref[0] = merge::kRefNone;
      neighbours_[i].ref[1] = merge::kRefNone;
    }
    above_mi_.assign(static_cast<std::size_t>(mi_cols_) + 16, 6);
    left_mi_.assign(static_cast<std::size_t>(mi_rows_) + 16, 6);
    init_cdfs(&cdfs_);
    buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(buffer_.data(), buffer_.size());
    enc_.disable_cdf_update = 1;  // the gate's frame headers pin this
  }

  bool run(TileOutput* out, std::string* error) {
    const int sb_cols = (mi_cols_ + 15) / 16;
    const int sb_rows = (mi_rows_ + 15) / 16;
    std::size_t leaf_index = 0;
    for (int sb_r = 0; sb_r < sb_rows; ++sb_r) {
      for (int sb_c = 0; sb_c < sb_cols; ++sb_c) {
        if (!write_node(sb_r * 16, sb_c * 16, 16, 16, &leaf_index, out,
                        error)) {
          return false;
        }
      }
    }
    std::size_t bytes = 0;
    if (enc_.exit_symbol(&bytes) != av2::Status::kOk) {
      *error = "exit_symbol failed";
      return false;
    }
    out->payload.assign(buffer_.begin(),
                        buffer_.begin() + static_cast<std::ptrdiff_t>(bytes));
    return true;
  }

 private:
  int neighbor_positions(int pos[2][2], int r, int c, int w4, int h4) const {
    int n = 0;
    const auto add = [&](int nr, int nc) {
      if (n < 2 && inside(nr, nc)) {
        pos[n][0] = nr;
        pos[n][1] = nc;
        ++n;
      }
    };
    add(r + h4 - 1, c - 1);
    add(r - 1, c + w4 - 1);
    add(r, c - 1);
    add(r - 1, c);
    return n;
  }

  int is_inter_context(int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    const auto intra_at = [&](int i) {
      return is_inter_[cell(pos[i][0], pos[i][1])] == 0;
    };
    if (n == 2) {
      return (intra_at(0) && intra_at(1)) ? 3 : (intra_at(0) || intra_at(1));
    }
    if (n == 1) return 2 * (intra_at(0) ? 1 : 0);
    return 0;
  }

  int skip_context(int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    int ctx = 0;
    for (int i = 0; i < n; ++i) {
      ctx += skip_[cell(pos[i][0], pos[i][1])] != 0 ? 1 : 0;
    }
    return ctx;
  }

  bool sym(std::uint16_t* cdf, int n, int symbol) {
    if (enc_.encode_symbol(cdf, n, symbol) != av2::Status::kOk) {
      error_ = "encode_symbol failed";
      return false;
    }
    return true;
  }

  std::size_t cell(int r, int c) const {
    return static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols_) +
           static_cast<std::size_t>(c);
  }
  bool inside(int r, int c) const {
    return r >= 0 && r < mi_rows_ && c >= 0 && c < mi_cols_;
  }

  int& left_mi(int r) { return left_mi_[static_cast<std::size_t>(r)]; }
  int& above_mi(int c) { return above_mi_[static_cast<std::size_t>(c)]; }

  bool code_do_split(int value, int r, int c, BlockSize size) {
    const int bsw = std::max(partition::mi_width_log2(size), 1);
    const int bsh = std::max(partition::mi_height_log2(size), 1);
    const int ctx = partition::partition_size_adjust(size) * 4 +
                    (left_mi(r) < bsh ? 2 : 0) +
                    (above_mi(c) < bsw ? 1 : 0);
    return sym(&cdfs_.do_split[0][ctx][0], 2, value);
  }

  bool code_rect_type(int value, int r, int c, BlockSize size) {
    const int bsw = std::max(partition::mi_width_log2(size), 1);
    const int bsh = std::max(partition::mi_height_log2(size), 1);
    const int ctx = partition::partition_size_adjust_rect_type(size) * 4 +
                    (left_mi(r) < bsh ? 2 : 0) +
                    (above_mi(c) < bsw ? 1 : 0);
    return sym(&cdfs_.rect_type[0][ctx][0], 2, value);
  }

  bool write_node(int r, int c, int w4, int h4, std::size_t* leaf_index,
                  TileOutput* out, std::string* error) {
    const BlockSize size = block_size_from_extent(w4, h4);
    if (size == BlockSize::kInvalid) {
      *error = "unsupported partition extent";
      return false;
    }
    const int leaf_mi = config_.leaf_size / 4;
    if (w4 == leaf_mi && h4 == leaf_mi) {
      if (!code_do_split(0, r, c, size)) {
        *error = error_;
        return false;
      }
      if (!write_leaf(r, c, w4, h4, *leaf_index, out, error)) return false;
      ++*leaf_index;
      return true;
    }
    if (!code_do_split(1, r, c, size)) {
      *error = error_;
      return false;
    }
    const bool vertical = w4 >= h4;
    if (!code_rect_type(vertical ? 1 : 0, r, c, size)) {
      *error = error_;
      return false;
    }
    const int half_w = vertical ? w4 / 2 : w4;
    const int half_h = vertical ? h4 : h4 / 2;
    const int dr = vertical ? 0 : half_h;
    const int dc = vertical ? half_w : 0;
    for (int k = 0; k < 2; ++k) {
      if (!write_node(r + k * dr, c + k * dc, half_w, half_h, leaf_index, out,
                      error)) {
        return false;
      }
    }
    return true;
  }

  int new_mv_context(int r, int c, int w4, int h4, int ref) {
    merge::NewMvContextParams params;
    params.mi_row = r;
    params.mi_col = c;
    params.bw4 = w4;
    params.bh4 = h4;
    params.sb_h4 = 16;
    params.mi_cols = mi_cols_;
    params.mi_rows = mi_rows_;
    params.mi = mi_.data();
    params.y_mode = y_mode_.data();
    params.is_compound = false;
    params.ref_frame[0] = static_cast<std::int8_t>(ref);
    params.ref_frame[1] = merge::kRefNone;
    int ctx = 0;
    if (merge::derive_new_mv_context(params, &ctx) != merge::Status::kOk) {
      return 0;
    }
    return ctx;
  }

  mc::Mv choose_predictor(int r, int c, int w4, int h4, int ref) {
    merge::StackParams params;
    params.mi_row = r;
    params.mi_col = c;
    params.bw4 = w4;
    params.bh4 = h4;
    params.sb_h4 = 16;
    params.mi_cols = mi_cols_;
    params.mi_rows = mi_rows_;
    params.mi = mi_.data();
    params.ref_frame[0] = static_cast<std::int8_t>(ref);
    params.ref_frame[1] = merge::kRefNone;
    params.is_compound = false;
    params.global_mv[0] = mc::Mv{0, 0};
    params.global_mv[1] = mc::Mv{0, 0};
    params.reorder = merge::DrlReorder::kDisabled;
    params.max_drl_bits_minus_1 = 4;
    params.frame_distance = nullptr;
    params.num_refs = 0;
    merge::MvStack stack;
    if (merge::build_mv_stack(params, &stack) != merge::Status::kOk ||
        stack.count <= 0) {
      return mc::Mv{0, 0};
    }
    return stack.cand[0].mv[0];
  }

  // Choose the unsigned MVD for the gate leaf.  `precision` decides the shell
  // unit.  The returned vector is in 1/8-sample units.
  mc::Mv choose_mvd(int r, int c, std::size_t leaf_index, int precision,
                    bool adaptive, int* joint, int index[2]) {
    if (adaptive) {
      const int pattern = static_cast<int>((leaf_index / 3) % 3);
      const int magnitude =
          kAmvdIndexToMvd[static_cast<std::size_t>(pattern)];
      *joint = pattern == 0 ? kMvJointHnzvz
                            : (pattern == 1 ? kMvJointHzvnz
                                            : kMvJointHnzvnz);
      index[0] = pattern;
      index[1] = pattern;
      const bool row = *joint == kMvJointHzvnz || *joint == kMvJointHnzvnz;
      const bool col = *joint == kMvJointHnzvz || *joint == kMvJointHnzvnz;
      mc::Mv out;
      out.y = row ? magnitude : 0;
      out.x = col ? magnitude : 0;
      return out;
    }
    const int shift = me2::kMvPrecisionEighthPel - precision;
    const int radix = 1 << shift;
    int units_y = (r * 3 + c + static_cast<int>(leaf_index)) % 3 - 1;
    int units_x = (r + c * 2 + static_cast<int>(leaf_index)) % 3 - 1;
    if (units_y == 0 && units_x == 0) units_x = 1;
    mc::Mv out;
    out.y = units_y * radix;
    out.x = units_x * radix;
    *joint = kMvJointZero;
    index[0] = 0;
    index[1] = 0;
    return out;
  }

  bool write_leaf(int r, int c, int w4, int h4, std::size_t leaf_index,
                  TileOutput* out, std::string* error) {
    LeafStats stats;
    stats.mi_row = r;
    stats.mi_col = c;
    stats.w4 = w4;
    stats.h4 = h4;
    const int ref = 0;
    const bool adaptive =
        config_.enable_adaptive_mvd && (leaf_index % 3 == 1);
    const bool flex = config_.enable_flex_mvres && !adaptive &&
                      (leaf_index % 3 == 2);
    stats.use_amvd = adaptive;
    stats.mode = adaptive ? LeafMode::kAdaptiveMvd
                          : (flex ? LeafMode::kFlexPrecision
                                  : LeafMode::kFramePrecision);

    const NeighborSet neighbours =
        collect_neighbors(neighbours_.data(), mi_cols_, mi_rows_, r, c, w4, h4,
                          16);
    stats.new_mv_context = new_mv_context(r, c, w4, h4, ref);
    stats.amvd_context = use_amvd_ctx(neighbours, ref);
    stats.precision_context = use_most_probable_precision_ctx(neighbours);
    const int pb_ctx =
        pb_mv_precision_ctx(neighbours, config_.frame_precision);
    stats.interp_context =
        interp_filter_ctx(neighbours, ref, false);

    me2::FramePrecisionSyntax frame;
    if (!me2::me2_frame_precision_to_bits(config_.frame_precision,
                                          config_.enable_flex_mvres ? 1 : 0, 0,
                                          &frame)) {
      *error = "frame precision is not signallable";
      return false;
    }
    MvResConfig mv_config;
    mv_config.enable_adaptive_mvd = config_.enable_adaptive_mvd;
    mv_config.enable_mvd_sign_derive = config_.enable_mvd_sign_derive;
    mv_config.enable_flex_mvres = config_.enable_flex_mvres;
    BlockPrecisionSyntax precision_plan;
    const int target = flex ? config_.flex_target_precision
                            : config_.frame_precision;
    if (!plan_block_precision(mv_config, frame, kYModeNewMv, adaptive, target,
                              &precision_plan, error)) {
      return false;
    }
    stats.mv_precision = precision_plan.mv_precision;
    stats.use_most_probable_precision =
        precision_plan.use_most_probable_precision;
    stats.pb_mv_precision = precision_plan.pb_mv_precision;

    mc::Mv pred = choose_predictor(r, c, w4, h4, ref);
    // assign_mv(): lower the predictor when MvPrecision < HALF and the block
    // is not adaptive-MVD.
    if (precision_plan.mv_precision < me2::kMvPrecisionHalfPel &&
        !precision_plan.is_adaptive_mvd) {
      me2::me2_lower_mv_precision(&pred.y, &pred.x,
                                  precision_plan.mv_precision);
    }
    stats.pred = pred;

    int joint = kMvJointZero;
    int amvd_index[2] = {0, 0};
    mc::Mv diff = choose_mvd(r, c, leaf_index, precision_plan.mv_precision,
                             precision_plan.is_adaptive_mvd, &joint,
                             amvd_index);
    if (diff.y == 0 && diff.x == 0) diff.x = 1;

    // Keep the last nonzero sign consistent with enable_mvd_sign_derive.
    MvdSignConfig sign_config;
    sign_config.enable_mvd_sign_derive = config_.enable_mvd_sign_derive;
    sign_config.frame_precision = config_.frame_precision;
    sign_config.mv_precision = precision_plan.mv_precision;
    sign_config.y_mode = kYModeNewMv;
    sign_config.is_adaptive_mvd = precision_plan.is_adaptive_mvd;
    sign_config.is_compound = false;
    sign_config.ref_mv_idx = 0;
    stats.derived_sign = mvd_sign_derive_allowed(sign_config);
    if (stats.derived_sign) {
      const int shift =
          me2::kMvPrecisionEighthPel - precision_plan.mv_precision;
      int last_sign = 0;
      int last_comp = -1;
      const std::int32_t values[2] = {diff.y, diff.x};
      for (int comp = 0; comp < 2; ++comp) {
        if (values[comp] == 0) continue;
        last_sign += (values[comp] < 0 ? -values[comp] : values[comp]) >> shift;
        last_comp = comp;
      }
      if (last_comp >= 0) {
        const bool negative = (last_sign & 1) != 0;
        std::int32_t& component = last_comp == 0 ? diff.y : diff.x;
        const std::int32_t magnitude = component < 0 ? -component : component;
        component = negative ? -magnitude : magnitude;
      }
    }
    stats.diff = diff;
    stats.amvd_joint = joint;
    stats.amvd_index[0] = amvd_index[0];
    stats.amvd_index[1] = amvd_index[1];
    stats.mv.y = mv_clamp_to_integer(pred.y + diff.y);
    stats.mv.x = mv_clamp_to_integer(pred.x + diff.x);

    const int filter = (r + c + static_cast<int>(leaf_index)) % 3;
    stats.interp_filter = filter;

    // ---- symbols, in §5.20.7.6 order --------------------------------
    if (!sym(&cdfs_.is_inter[is_inter_context(r, c, w4, h4)][0], 2, 1)) {
      *error = error_;
      return false;
    }
    if (!sym(&cdfs_.skip[skip_context(r, c, w4, h4)][0], 2, 0)) {
      *error = error_;
      return false;
    }
    if (!sym(&cdfs_.single_mode[stats.new_mv_context][0], 3, 2)) {
      *error = error_;
      return false;
    }
    if (!write_use_amvd(enc_, cdfs_.mv,
                        use_amvd_cdf_index(kYModeNewMv, false),
                        stats.amvd_context, adaptive, error)) {
      return false;
    }
    // read_drl_idx(0, max_drl_bits_minus_1 + 1): index 0 is one zero symbol.
    if (!sym(&cdfs_.drl_mode[0][stats.new_mv_context][0], 2, 0)) {
      *error = error_;
      return false;
    }
    if (!write_block_precision(enc_, cdfs_.mv, stats.precision_context, pb_ctx,
                               precision_plan, error)) {
      return false;
    }
    if (precision_plan.is_adaptive_mvd) {
      AmvdMvSyntax syntax;
      syntax.joint = joint;
      syntax.index[0] = amvd_index[0];
      syntax.index[1] = amvd_index[1];
      syntax.magnitude[0] = joint == kMvJointHzvnz ||
                                    joint == kMvJointHnzvnz
                                ? kAmvdIndexToMvd[amvd_index[0]]
                                : 0;
      syntax.magnitude[1] = joint == kMvJointHnzvz ||
                                    joint == kMvJointHnzvnz
                                ? kAmvdIndexToMvd[amvd_index[1]]
                                : 0;
      if (!write_amvd_mv(enc_, cdfs_.mv, syntax, error)) return false;
    } else {
      if (!write_shell_mv(enc_, cdfs_.mv, diff.y, diff.x,
                          precision_plan.mv_precision, error)) {
        return false;
      }
    }
    mc::Mv mvd[1] = {diff};
    if (!write_mvd_signs(enc_, sign_config, mvd, 1, error)) return false;
    if (!write_block_interp_filter(enc_, cdfs_.mv, stats.interp_context,
                                   filter, error)) {
      return false;
    }
    // All-zero residual.  Every preceding txb in this gate is also
    // all-zero, so the level contexts are zero.  TX_MODE_LARGEST makes each
    // plane a single block-sized txb: Y 16x16, U/V 8x8.  The U skip context
    // carries +6; V is exactly 0 here (no area excess, EobU == 0).
    if (!sym(cdfs_.txb_skip[1][tx_sz_ctx(2)][0], 2, 1)) {
      *error = error_;
      return false;
    }
    if (!sym(cdfs_.txb_skip[1][tx_sz_ctx(1)][6], 2, 1)) {
      *error = error_;
      return false;
    }
    if (!sym(cdfs_.v_txb_skip[0], 2, 1)) {
      *error = error_;
      return false;
    }

    // ---- decoder-visible state -------------------------------------
    for (int yy = 0; yy < h4; ++yy) {
      for (int xx = 0; xx < w4; ++xx) {
        const std::size_t idx = cell(r + yy, c + xx);
        is_inter_[idx] = 1;
        skip_[idx] = 0;
        y_mode_[idx] = kYModeNewMv;
        inter::mc::MiInfo& m = mi_[idx];
        m.ref[0] = static_cast<std::int8_t>(ref);
        m.ref[1] = merge::kRefNone;
        m.mv[0][0] = stats.mv.y;
        m.mv[0][1] = stats.mv.x;
        m.mv[1][0] = stats.mv.y;
        m.mv[1][1] = stats.mv.x;
        m.is_inter = 1;
        m.w4 = static_cast<std::int8_t>(w4);
        m.h4 = static_cast<std::int8_t>(h4);
        m.col_base = static_cast<std::int16_t>(c);
        m.row_base = static_cast<std::int16_t>(r);
        NeighborMvInfo& n = neighbours_[idx];
        n.available = true;
        n.ref[0] = static_cast<std::int8_t>(ref);
        n.ref[1] = merge::kRefNone;
        n.uses_amvd = adaptive;
        n.use_most_probable_precision =
            precision_plan.use_most_probable_precision;
        n.mv_precision = precision_plan.mv_precision;
        n.y_mode = kYModeNewMv;
        n.interp_filter = filter;
      }
    }
    const BlockSize leaf_size = block_size_from_extent(w4, h4);
    const int leaf_h_log2 = partition::mi_height_log2(leaf_size);
    const int leaf_w_log2 = partition::mi_width_log2(leaf_size);
    for (int yy = 0; yy < h4; ++yy) left_mi(r + yy) = leaf_h_log2;
    for (int xx = 0; xx < w4; ++xx) above_mi(c + xx) = leaf_w_log2;

    out->leaves.push_back(stats);
    if (adaptive) ++out->amvd_blocks;
    if (flex) ++out->flex_blocks;
    if (!adaptive && !flex) ++out->frame_precision_blocks;
    if (stats.derived_sign) ++out->derived_sign_blocks;
    return true;
  }

  TileConfig config_;
  FrameGeometry geometry_{};
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  Cdfs cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> buffer_;
  std::string error_;
  std::vector<inter::mc::MiInfo> mi_;
  std::vector<NeighborMvInfo> neighbours_;
  std::vector<std::uint8_t> y_mode_;
  std::vector<std::uint8_t> is_inter_;
  std::vector<std::uint8_t> skip_;
  std::vector<int> above_mi_;
  std::vector<int> left_mi_;
};

}  // namespace

bool write_tile(int width, int height, const TileConfig& config,
                TileOutput* output, std::string* error) {
  if (output == nullptr || error == nullptr) return false;
  error->clear();
  *output = TileOutput{};
  if (width <= 0 || height <= 0 || width % 64 != 0 || height % 64 != 0) {
    *error = "tile requires 64-multiple dimensions";
    return false;
  }
  if (config.leaf_size != 16) {
    *error = "this gate writes 16x16 leaves only";
    return false;
  }
  TileWriter writer(width, height, config);
  return writer.run(output, error);
}

}  // namespace av2enc::inter::mvprec::gate
