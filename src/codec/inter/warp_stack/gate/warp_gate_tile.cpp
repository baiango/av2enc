#include "codec/inter/warp_stack/gate/warp_gate_tile.h"

#include <algorithm>
#include <cstring>

#include "pipeline/still/basic_cdf_tables.h"
#include "codec/inter/advanced/video/advanced_cdf_tables.h"
#include "pipeline/video/video_cdf_tables.h"
#include "codec/partition/tree/partition.h"
#include "codec/entropy/av2/av2_entropy.h"

namespace av2enc::inter::warp::gate {
namespace {

using partition::BlockSize;

constexpr int kNoneRef = -1;
constexpr int kIntraRef = 8;
constexpr int kTileCapacityBytes = 4U * 1024U * 1024U;

struct Cdfs {
  std::uint16_t do_split[2][64][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t is_inter[4][3];
  std::uint16_t skip[6][3];
  std::uint16_t single_mode[5][4];
  std::uint16_t drl_mode[3][5][3];
  std::uint16_t shell_set[3];
  std::uint16_t shell6_class0[9];
  std::uint16_t shell6_class1[9];
  std::uint16_t shell_last_two[3];
  std::uint16_t offset_low[2][3];
  std::uint16_t offset_class2[3];
  std::uint16_t offset_other[16][3];
  std::uint16_t col_mv_greater[2][3];
  std::uint16_t col_mv_index[4][3];
  std::uint16_t is_warp[5][3];
  std::uint16_t warp_mv[3];
  std::uint16_t use_local_warp[4][3];
  std::uint16_t use_extend_warp[3][3];
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

void init_cdfs(Cdfs& c) {
  copy_array(c.do_split, av2enc::basic::kDoSplitCdf);
  copy_array(c.rect_type, av2enc::basic::kRectTypeCdf);
  copy_array(c.is_inter, av2enc::video::kIsInterCdf);
  copy_array(c.skip, av2enc::video::kSkipCdf);
  copy_array(c.single_mode, av2enc::video::kSingleModeCdf);
  copy_array(c.drl_mode, av2enc::video::kDrlModeCdf);
  copy_array(c.shell_set, av2enc::video::kJointShellSetCdf);
  copy_array(c.shell6_class0, av2enc::video::kJointShell6Class0Cdf);
  copy_array(c.shell6_class1, av2enc::video::kJointShell6Class1Cdf);
  copy_array(c.shell_last_two, av2enc::video::kJointShellLastTwoClassesCdf);
  copy_array(c.offset_low, av2enc::video::kShellOffsetLowClassCdf);
  copy_array(c.offset_class2, av2enc::video::kShellOffsetClass2Cdf);
  copy_array(c.offset_other, av2enc::video::kShellOffsetOtherClassCdf);
  copy_array(c.col_mv_greater, av2enc::video::kColMvGreaterCdf);
  copy_array(c.col_mv_index, av2enc::video::kColMvIndexCdf);
  copy_array(c.is_warp, av2enc::video::adv::cdf::kIsWarpCdf);
  copy_array(c.warp_mv, av2enc::video::adv::cdf::kWarpMvCdf);
  copy_array(c.use_local_warp, av2enc::video::adv::cdf::kUseLocalWarpCdf);
  copy_array(c.use_extend_warp, av2enc::video::adv::cdf::kUseExtendWarpCdf);
}

int floor_log2(int v) {
  int n = 0;
  while (v > 1) {
    v >>= 1;
    ++n;
  }
  return n;
}

class TileWriter {
 public:
  TileWriter(const GateTilePlan& plan, const GateTileConfig& config)
      : plan_(plan), config_(config) {
    geometry_.width = plan.width;
    geometry_.height = plan.height;
    geometry_.sb_size = plan.sb_size;
    geometry_.has_chroma = true;
    geometry_.chroma_subsampling_x = true;
    geometry_.chroma_subsampling_y = true;
    mi_cols_ = partition::mi_cols(geometry_);
    mi_rows_ = partition::mi_rows(geometry_);
    sb_mi_ = partition::sb_size_mi(geometry_);
    sizes_.reset(geometry_);
    const std::size_t cells = static_cast<std::size_t>(mi_rows_) *
                              static_cast<std::size_t>(mi_cols_);
    is_inter_.assign(cells, 0);
    skip_.assign(cells, 0);
    ymode_.assign(cells, 0);
    ref0_.assign(cells, kIntraRef);
    ref1_.assign(cells, kNoneRef);
    grid_.assign(cells, WarpMi{});
    init_cdfs(cdfs_);
    buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(buffer_.data(), buffer_.size());
    enc_.disable_cdf_update = config_.disable_cdf_update ? 1 : 0;
  }

  bool run(std::string& error) {
    const int sb_cols = (mi_cols_ + sb_mi_ - 1) / sb_mi_;
    const int sb_rows = (mi_rows_ + sb_mi_ - 1) / sb_mi_;
    if (static_cast<int>(plan_.leaves.size()) != sb_cols * sb_rows) {
      error = "gate tile plan needs exactly one leaf per superblock";
      return false;
    }
    if (plan_.sb_size != 64 || mi_cols_ % sb_mi_ != 0 ||
        mi_rows_ % sb_mi_ != 0) {
      error = "gate tile writer requires 64x64 superblocks and frame "
              "dimensions divisible by 64";
      return false;
    }
    for (int sb = 0; sb < sb_cols * sb_rows; ++sb) {
      const int r = (sb / sb_cols) * sb_mi_;
      const int c = (sb % sb_cols) * sb_mi_;
      const partition::PartitionSet allowed = partition::init_allowed_partitions(
          geometry_, r, c, BlockSize::k64x64, true, false);
      if (!allowed.is_allowed(partition::PartitionType::kNone)) {
        error = "gate tile writer requires PARTITION_NONE at every superblock";
        return false;
      }
      const partition::PartitionContexts ctx =
          partition::partition_contexts(geometry_, r, c, BlockSize::k64x64,
                                        sizes_);
      if (!sym(&cdfs_.do_split[0][ctx.do_split][0], 2, 0)) {
        error = error_;
        return false;
      }
      if (!write_leaf(plan_.leaves[static_cast<std::size_t>(sb)], r, c,
                      error)) {
        return false;
      }
      sizes_.fill_block(r, c, BlockSize::k64x64);
    }
    std::size_t bytes = 0;
    const av2::Status status = enc_.exit_symbol(&bytes);
    if (status != av2::Status::kOk) {
      error = std::string("exit_symbol failed: ") + av2::status_name(status);
      return false;
    }
    payload_.assign(buffer_.begin(),
                    buffer_.begin() + static_cast<std::ptrdiff_t>(bytes));
    return true;
  }

  const std::vector<std::uint8_t>& payload() const { return payload_; }
  const std::vector<WarpMi>& grid() const { return grid_; }

 private:
  bool sym(std::uint16_t* cdf, int n, int symbol) {
    const av2::Status status = enc_.encode_symbol(cdf, n, symbol);
    if (status != av2::Status::kOk) {
      error_ = std::string("encode_symbol failed: ") +
               av2::status_name(status) + " n=" + std::to_string(n) +
               " sym=" + std::to_string(symbol);
      return false;
    }
    return true;
  }
  bool raw_bit(bool bit) {
    if (enc_.encode_bool(bit) != av2::Status::kOk) {
      error_ = "encode_bool failed";
      return false;
    }
    return true;
  }
  bool literal(int bits, int value) {
    for (int i = bits - 1; i >= 0; --i) {
      if (!raw_bit(((value >> i) & 1) != 0)) return false;
    }
    return true;
  }
  bool ns_bits(std::uint32_t n, std::uint32_t value) {
    unsigned w = 0;
    while ((1U << w) < n) ++w;
    const std::uint32_t m = (1U << w) - n;
    if (value < m) return literal(static_cast<int>(w - 1U),
                                  static_cast<int>(value));
    const std::uint32_t x = value + m;
    if (!literal(static_cast<int>(w - 1U), static_cast<int>(x >> 1U)))
      return false;
    return raw_bit((x & 1U) != 0U);
  }

  // §5.20.7.20 read_mv() for MvPrecision = 1/8 pel.
  bool write_mv(int dy, int dx) {
    const int ay = std::abs(dy);
    const int ax = std::abs(dx);
    const int shell = ay + ax;
    if (shell >= (1 << 17)) {
      error_ = "MVD shell out of range";
      return false;
    }
    const int cls = shell == 0 ? 0 : floor_log2(shell);
    const int base = cls == 0 ? 0 : (1 << cls);
    const int off = shell - base;
    const int setsym = cls >= 8 ? 1 : 0;
    const int class_sym = cls >= 8 ? cls - 8 : cls;
    if (!sym(&cdfs_.shell_set[0], 2, setsym)) return false;
    std::uint16_t* class_cdf =
        setsym ? &cdfs_.shell6_class1[0] : &cdfs_.shell6_class0[0];
    if (!sym(class_cdf, 8, class_sym)) return false;
    if (setsym && class_sym == 7) {
      if (!sym(&cdfs_.shell_last_two[0], 2, cls - 15)) return false;
    }
    if (cls < 2) {
      if (!sym(&cdfs_.offset_low[cls][0], 2, off)) return false;
    } else if (cls == 2) {
      if (off == 0) {
        if (!sym(&cdfs_.offset_class2[0], 2, 0)) return false;
      } else {
        if (!sym(&cdfs_.offset_class2[0], 2, 1)) return false;
        if (off == 1) {
          if (!raw_bit(false)) return false;
        } else {
          if (!raw_bit(true)) return false;
          if (!raw_bit(off == 3)) return false;
        }
      }
    } else {
      for (int i = 0; i < cls; ++i) {
        if (!sym(&cdfs_.offset_other[i][0], 2, (off >> i) & 1)) return false;
      }
    }
    if (shell > 0) {
      const int max_pair = shell >> 1;
      bool skip_col_bit = false;
      int col = 0;
      if (ax == max_pair && ((shell & 1) == 0)) {
        skip_col_bit = true;
        col = max_pair;
      } else if (ax <= max_pair) {
        col = ax;
      } else {
        col = ay;
      }
      if (max_pair > 0) {
        const int max_bits = std::min<int>(max_pair, 2);
        const int prefix_col = std::min<int>(col, 2);
        for (int i = 0; i < max_bits; ++i) {
          const int greater = prefix_col > i ? 1 : 0;
          if (!sym(&cdfs_.col_mv_greater[i][0], 2, greater)) return false;
          if (greater == 0) break;
        }
        if (max_pair > 2 && prefix_col == 2) {
          if (!ns_bits(static_cast<std::uint32_t>(max_pair - 1),
                       static_cast<std::uint32_t>(col - 2))) {
            return false;
          }
        }
      }
      if (!skip_col_bit) {
        const int index = ax <= max_pair ? 0 : 1;
        if (!sym(&cdfs_.col_mv_index[std::min(cls, 3)][0], 2, index))
          return false;
      }
    }
    if (dy != 0 && !raw_bit(dy < 0)) return false;
    if (dx != 0 && !raw_bit(dx < 0)) return false;
    return true;
  }

  std::size_t cell_index(int r, int c) const {
    return static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols_) +
           static_cast<std::size_t>(c);
  }
  bool inside(int r, int c) const {
    return r >= 0 && r < mi_rows_ && c >= 0 && c < mi_cols_;
  }
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
      return is_inter_[cell_index(pos[i][0], pos[i][1])] == 0;
    };
    if (n == 2) return (intra_at(0) && intra_at(1)) ? 3 : (intra_at(0) || intra_at(1));
    if (n == 1) return 2 * (intra_at(0) ? 1 : 0);
    return 0;
  }
  int skip_context(int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    int ctx = 0;
    for (int i = 0; i < n; ++i) {
      ctx += skip_[cell_index(pos[i][0], pos[i][1])] != 0 ? 1 : 0;
    }
    return ctx;
  }
  // §7.11.2/§7.11.3: WARP_NEWMV (18) must not increment NewMvCount, so only
  // YMode NEWMV is counted here (has_newmv_for_list).
  int new_mv_context(int r, int c, int w4, int h4) const {
    int new_mv_count = 0;
    const bool sb_border = (r & 15) == 0;
    const auto scan = [&](int dr, int dc) -> int {
      const int mr = r + dr;
      const int mc = c + dc;
      if (!inside(mr, mc)) return 0;
      const std::size_t idx = cell_index(mr, mc);
      if (is_inter_[idx] == 0) return 0;
      if (ymode_[idx] == kYModeNewMv) {
        new_mv_count = std::min(3, new_mv_count + 1);
      }
      return 1;
    };
    const int left_a = scan(h4 - 1, -1);
    const int above_a = scan(-1, w4 - 1);
    const int left_b = scan(0, -1);
    const int above_b = (w4 >= (sb_border ? 4 : 2)) ? scan(-1, 0) : 0;
    const int nearest = ((above_a || above_b) ? 1 : 0) +
                        ((left_a || left_b) ? 1 : 0);
    return nearest + (new_mv_count > 0 ? 2 : 0);
  }

  int local_warp_ctx(int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    int n = 0;
    const auto add = [&](int nr, int nc) {
      if (n >= 2 || !inside(nr, nc)) return;
      const bool above_sb = (r / sb_mi_) != (nr / sb_mi_);
      if (above_sb) return;
      pos[n][0] = nr;
      pos[n][1] = nc;
      ++n;
    };
    add(r + h4 - 1, c - 1);
    add(r - 1, c + w4 - 1);
    add(r, c - 1);
    add(r - 1, c);
    int has_warp = 0;
    int ctx = 0;
    for (int i = 0; i < n; ++i) {
      const int mode = grid_[cell_index(pos[i][0], pos[i][1])].motion_mode;
      if (mode >= kMotionLocalWarp) {
        has_warp = 1;
        ++ctx;
      }
      if (mode == kMotionLocalWarp) ++ctx;
    }
    ctx += has_warp;
    return std::min(ctx, 3);
  }

  WarpContext warp_context(int r, int c, int w4, int h4) const {
    WarpContextParams p;
    p.mi_row = r;
    p.mi_col = c;
    p.bw4 = w4;
    p.bh4 = h4;
    p.sb_h4 = sb_mi_;
    p.grid = WarpGrid{mi_cols_, mi_rows_, grid_.data()};
    p.ref_frame[0] = 0;
    p.ref_frame[1] = kNoneRef;
    return compute_warp_context(p);
  }

  bool write_leaf(const GateLeaf& leaf, int r, int c, std::string& error) {
    const int w4 = 16;
    const int h4 = 16;
    if (config_.force_integer_mv && leaf.is_warp) {
      error = "force_integer_mv is not supported by this gate";
      return false;
    }
    const WarpContext wctx = warp_context(r, c, w4, h4);
    if (leaf.is_warp && leaf.use_local_warp && !wctx.warp_sample_found[0]) {
      error = "LOCALWARP has no matching warp sample neighbour at " +
                std::to_string(r) + "," + std::to_string(c);
      return false;
    }

    if (!sym(&cdfs_.is_inter[is_inter_context(r, c, w4, h4)][0], 2, 1)) {
      error = error_;
      return false;
    }
    if (!sym(&cdfs_.skip[skip_context(r, c, w4, h4)][0], 2, 1)) {
      error = error_;
      return false;
    }
    // read_ref_frames(): NumTotalRefs == 1 -> RefFrame = {0, NONE}, no bits.

    const int is_warp_ctx =
        config_.force_warp_mv_ctx >= 0
            ? config_.force_warp_mv_ctx
            : std::min(wctx.warp_mv_count, 4);
    const int local_ctx = local_warp_ctx(r, c, w4, h4);
    if (config_.allow_warpmv_mode) {
      if (!sym(&cdfs_.is_warp[is_warp_ctx][0], 2,
               leaf.is_warp ? 1 : 0)) {
        error = error_;
        return false;
      }
    } else if (leaf.is_warp) {
      error = "is_warp signalled without allow_warpmv_mode";
      return false;
    }

    const int mode_ctx = new_mv_context(r, c, w4, h4);
    if (leaf.is_warp) {
      if (!sym(&cdfs_.warp_mv[0], 2, leaf.warp_mv ? 1 : 0)) {
        error = error_;
        return false;
      }
    } else {
      if (!sym(&cdfs_.single_mode[mode_ctx][0], 3, leaf.single_mode)) {
        error = error_;
        return false;
      }
    }

    // read_motion_mode().
    if (leaf.is_warp && !leaf.warp_mv) {
      // YMode == WARP_NEWMV.
      if (config_.frame_enable_extend_warp) {
        if (!sym(&cdfs_.use_extend_warp[0][0], 2,
                 leaf.use_extend_warp ? 1 : 0)) {
          error = error_;
          return false;
        }
      } else if (leaf.use_extend_warp) {
        error = "EXTENDWARP signalled without frame_enabled_motion_modes";
        return false;
      }
      if (!leaf.use_extend_warp && config_.frame_enable_local_warp) {
        if (!sym(&cdfs_.use_local_warp[local_ctx][0], 2,
                 leaf.use_local_warp ? 1 : 0)) {
          error = error_;
          return false;
        }
      } else if (leaf.use_local_warp) {
        error = "LOCALWARP signalled without frame_enabled_motion_modes";
        return false;
      }
    } else if (leaf.is_warp) {
      // YMode == WARPMV implies DELTAWARP with no motion-mode bits.
      if (!leaf.use_local_warp || leaf.use_extend_warp) {
        error = "WARPMV must map to DELTAWARP";
        return false;
      }
    } else {
      if (leaf.use_local_warp || leaf.use_extend_warp) {
        error = "motion mode requires WARP_NEWMV/WARPMV";
        return false;
      }
    }

    // Derived YMode / motion mode, validated against the plan.
    int y_mode;
    if (leaf.is_warp) {
      y_mode = leaf.warp_mv ? kYModeWarpMv : kYModeWarpNewMv;
    } else {
      y_mode = kYModeNearMv + leaf.single_mode;
    }
    int motion_mode = kMotionSimple;
    if (leaf.is_warp) {
      motion_mode = leaf.warp_mv ? kMotionDeltaWarp
                                 : (leaf.use_extend_warp
                                        ? kMotionExtendWarp
                                        : (leaf.use_local_warp ? kMotionLocalWarp
                                                               : kMotionDeltaWarp));
    }
    if (y_mode != leaf.y_mode || motion_mode != leaf.motion_mode) {
      error = "leaf y_mode/motion_mode does not match the signalled bits";
      return false;
    }

    // Resolve the prediction exactly as the decoder's assign_mv() does, using
    // the §7.12.2 stack this module builds from the already-coded neighbours.
    WarpStackParams sp;
    sp.mi_row = r;
    sp.mi_col = c;
    sp.bw4 = w4;
    sp.bh4 = h4;
    sp.sb_h4 = sb_mi_;
    sp.mi_cols = mi_cols_;
    sp.mi_rows = mi_rows_;
    sp.mi = grid_.data();
    sp.ref_frame[0] = 0;
    sp.ref_frame[1] = kNoneRef;
    sp.derive_wrl = w4 * 4 >= 8 && h4 * 4 >= 8;
    sp.global_mv[0] = inter::mc::Mv{0, 0};
    sp.max_drl_bits_minus_1 = config_.max_drl_bits_minus_1;
    WarpStack stack;
    if (build_warp_stack(sp, &stack) != Status::kOk) {
      error = "warp stack build failed";
      return false;
    }
    inter::mc::Mv pred{0, 0};
    if (y_mode == kYModeGlobalMv) {
      pred = inter::mc::Mv{0, 0};
    } else {
      if (leaf.drl_index < 0 || leaf.drl_index >= stack.count) {
        error = "DRL index outside the decoded stack";
        return false;
      }
      pred = stack.cand[leaf.drl_index].mv[0];
    }
    int mvd_y = leaf.mvd_y;
    int mvd_x = leaf.mvd_x;
    if (leaf.use_target_mv) {
      mvd_y = leaf.target_mv_y - pred.y;
      mvd_x = leaf.target_mv_x - pred.x;
    }

    const bool has_newmv =
        y_mode == kYModeNewMv || y_mode == kYModeWarpNewMv;
    if (has_newmv || y_mode == kYModeNearMv) {
      if (!code_drl_index(mode_ctx, leaf.drl_index)) {
        error = error_.empty() ? "drl index encode failed" : error_;
        return false;
      }
    }
    if (has_newmv) {
      if (!write_mv(mvd_y, mvd_x)) {
        error = error_.empty() ? "MVD encode failed" : error_;
        return false;
      }
    }
    // read_refinemv/read_compound_type emit nothing for a single reference with
    // enable_refinemv = enable_masked_compound = 0.
    // interp_filter is not switchable and LOCALWARP disables it.

    const inter::mc::Mv final_mv{pred.y + mvd_y, pred.x + mvd_x};

    // Update the context grids over the 64x64 leaf.
    for (int yy = 0; yy < h4; ++yy) {
      for (int xx = 0; xx < w4; ++xx) {
        const std::size_t idx = cell_index(r + yy, c + xx);
        is_inter_[idx] = 1;
        skip_[idx] = 1;
        ymode_[idx] = static_cast<std::uint8_t>(y_mode);
        ref0_[idx] = 0;
        ref1_[idx] = kNoneRef;
        WarpMi& m = grid_[idx];
        m = WarpMi{};
        m.ref[0] = 0;
        m.ref[1] = kNoneRef;
        m.mv[0][0] = final_mv.y;
        m.mv[0][1] = final_mv.x;
        m.sub_mv[0][0] = final_mv.y;
        m.sub_mv[0][1] = final_mv.x;
        m.is_inter = 1;
        m.written = 1;
        m.w4 = static_cast<std::int8_t>(w4);
        m.h4 = static_cast<std::int8_t>(h4);
        m.col_base = static_cast<std::int16_t>(c);
        m.row_base = static_cast<std::int16_t>(r);
        m.y_mode = static_cast<std::uint8_t>(y_mode);
        m.motion_mode = static_cast<std::uint8_t>(motion_mode);
      }
    }

    // §7.13.3.23: a LOCALWARP block stores the least-squares model of its
    // samples in WarpParams[][]; later corner scans read it.
    if (motion_mode == kMotionLocalWarp) {
      LocalWarpParams lp;
      lp.mi_row = r;
      lp.mi_col = c;
      lp.w4 = w4;
      lp.h4 = h4;
      lp.mi_cols = mi_cols_;
      lp.mi_rows = mi_rows_;
      lp.sb_h4 = sb_mi_;
      lp.grid = WarpGrid{mi_cols_, mi_rows_, grid_.data()};
      lp.ref_frame[0] = 0;
      lp.ref_frame[1] = kNoneRef;
      lp.avail_u = r > 0;
      lp.avail_l = c > 0;
      lp.block_mv = final_mv;
      const auto model = derive_local_warp(lp, 0);
      if (!model.has_value()) {
        error = "local warp derivation failed";
        return false;
      }
      for (int yy = 0; yy < h4; ++yy) {
        for (int xx = 0; xx < w4; ++xx) {
          grid_[cell_index(r + yy, c + xx)].warp[0] = model->params;
        }
      }
      // §7.13.3.19: a warped block stores one projected MV per 8x8 section in
      // SubMvs.  The MV stack and compound-add-derived read SubMvs at the
      // scanned cell, so this must not be the block MV.
      for (int i8 = 0; i8 < h4 / 2; ++i8) {
        for (int j8 = 0; j8 < w4 / 2; ++j8) {
          const inter::mc::Mv smv =
              warp_sub_block_mv(model->params, c * kMiSize, r * kMiSize, i8, j8);
          for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
              WarpMi& cell = grid_[cell_index(r + i8 * 2 + i, c + j8 * 2 + j)];
              cell.sub_mv[0][0] = smv.y;
              cell.sub_mv[0][1] = smv.x;
            }
          }
        }
      }
    }
    return true;
  }

  bool code_drl_index(int ctx, int index) {
    const int m = config_.max_drl_bits_minus_1 + 1;
    if (index < 0 || index > m) return false;
    for (int idx = 0; idx < m; ++idx) {
      const int mode = idx < index ? 1 : 0;
      if (!sym(&cdfs_.drl_mode[std::min(idx, 2)][ctx][0], 2, mode)) {
        return false;
      }
      if (mode == 0) break;
    }
    return true;
  }

  const GateTilePlan& plan_;
  const GateTileConfig& config_;
  partition::FrameGeometry geometry_{};
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  int sb_mi_ = 16;
  partition::MiSizeGrid sizes_;
  Cdfs cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> buffer_;
  std::vector<std::uint8_t> payload_;
  std::string error_;

  std::vector<std::uint8_t> is_inter_;
  std::vector<std::uint8_t> skip_;
  std::vector<std::uint8_t> ymode_;
  std::vector<std::int8_t> ref0_;
  std::vector<std::int8_t> ref1_;
  std::vector<WarpMi> grid_;
};

}  // namespace

bool write_gate_warp_tile(const GateTilePlan& plan, const GateTileConfig& config,
                          std::vector<std::uint8_t>& payload,
                          std::vector<WarpMi>* grid_out, int* mi_cols_out,
                          int* mi_rows_out, std::string& error) {
  error.clear();
  payload.clear();
  TileWriter writer(plan, config);
  if (!writer.run(error)) return false;
  payload = writer.payload();
  if (grid_out != nullptr) *grid_out = writer.grid();
  if (mi_cols_out != nullptr) *mi_cols_out = plan.width / 4;
  if (mi_rows_out != nullptr) *mi_rows_out = plan.height / 4;
  return true;
}

}  // namespace av2enc::inter::warp::gate
