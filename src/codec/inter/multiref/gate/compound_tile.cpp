#include "codec/inter/multiref/gate/compound_tile.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

#include "pipeline/still/basic_cdf_tables.h"
#include "pipeline/video/video_cdf_tables.h"
#include "codec/inter/multiref/gate/compound_cdf_tables.h"
#include "codec/partition/tree/partition.h"
#include "codec/entropy/av2/av2_entropy.h"

namespace av2enc::video::multiref::gate {
namespace {

using partition::BlockSize;

// §5.20.7.5 RefFrame sentinels.  Values 0..6 are reference indices.
constexpr int kNoneRef = -1;
constexpr int kIntraRef = 8;
constexpr int kNearNearMv = 19;
constexpr int kGlobalGlobalMv = 22;
constexpr int kTileCapacityBytes = 8U * 1024U * 1024U;

bool is_inter_ref_value(int ref) { return ref >= 0 && ref < 8; }

bool has_second_drl(int y_mode) {
  return y_mode == kNearNearMv || y_mode == 20 || y_mode == 21 || y_mode == 24;
}

struct Cdfs {
  std::uint16_t do_split[2][64][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t is_inter[4][3];
  std::uint16_t skip[6][3];
  std::uint16_t drl_mode[3][5][3];
  std::uint16_t comp_mode[5][3];
  std::uint16_t comp_ref0[3][6][3];
  std::uint16_t comp_ref1[3][2][6][3];
  std::uint16_t is_joint[2][3];
  std::uint16_t compound_mode_non_joint[5][6];
  std::uint16_t compound_mode_same_refs[5][5];
  std::uint16_t comp_group_idx[12][3];
  std::uint16_t compound_type[3];
  std::uint16_t wedge_quad[5];
  std::uint16_t wedge_angle[4][6];
  std::uint16_t wedge_dist1[5];
  std::uint16_t wedge_dist2[4];
  std::uint16_t cwp_idx[4][3];
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
  copy_array(c.drl_mode, av2enc::video::kDrlModeCdf);
  copy_array(c.comp_mode, kCompModeCdf);
  copy_array(c.comp_ref0, kCompRef0Cdf);
  copy_array(c.comp_ref1, kCompRef1Cdf);
  copy_array(c.is_joint, kIsJointCdf);
  copy_array(c.compound_mode_non_joint, kCompoundModeNonJointCdf);
  copy_array(c.compound_mode_same_refs, kCompoundModeSameRefsCdf);
  copy_array(c.comp_group_idx, kCompGroupIdxCdf);
  copy_array(c.compound_type, kCompoundTypeCdf);
  copy_array(c.wedge_quad, kWedgeQuadCdf);
  copy_array(c.wedge_angle, kWedgeAngleCdf);
  copy_array(c.wedge_dist1, kWedgeDist1Cdf);
  copy_array(c.wedge_dist2, kWedgeDist2Cdf);
  copy_array(c.cwp_idx, kCwpIdxCdf);
}

class TileWriter {
 public:
  TileWriter(const GateTilePlan& plan, const RefFrameMap& map,
             const GateTileConfig& config)
      : plan_(plan), map_(map), config_(config) {
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
    comp_group_.assign(cells, 0);
    init_cdfs(cdfs_);
    buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(buffer_.data(), buffer_.size());
    enc_.disable_cdf_update = config_.disable_cdf_update ? 1 : 0;
  }

  bool run(std::size_t* payload_bytes, std::string& error) {
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
    if (payload_bytes != nullptr) *payload_bytes = bytes;
    return true;
  }

  const std::vector<std::uint8_t>& payload() const { return payload_; }

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

  std::size_t cell(int r, int c) const {
    return static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols_) +
           static_cast<std::size_t>(c);
  }
  bool inside(int r, int c) const {
    return r >= 0 && r < mi_rows_ && c >= 0 && c < mi_cols_;
  }
  // Same neighbour order as the M2 writer / decode_block().
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
    if (n == 2) return (intra_at(0) && intra_at(1)) ? 3 : (intra_at(0) || intra_at(1));
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

  // §7.11.2 / M2 new_mv_context().
  int new_mv_context(int r, int c, int w4, int h4) const {
    int new_mv_count = 0;
    const bool sb_border = (r & 15) == 0;
    const auto scan = [&](int dr, int dc) -> int {
      const int mr = r + dr;
      const int mc = c + dc;
      if (!inside(mr, mc)) return 0;
      const std::size_t idx = cell(mr, mc);
      if (is_inter_[idx] == 0) return 0;
      if (ymode_[idx] == 16) new_mv_count = std::min(3, new_mv_count + 1);
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

  bool check_backward(int ref) const {
    if (!is_inter_ref_value(ref) || ref >= map_.num_total_refs) return false;
    return map_.frame_distance[ref] < 0;
  }

  int comp_mode_context(int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    const auto nsingle = [&](int i) {
      return !is_inter_ref_value(ref1_[cell(pos[i][0], pos[i][1])]);
    };
    const auto nref0 = [&](int i) { return ref0_[cell(pos[i][0], pos[i][1])]; };
    const auto nintra = [&](int i) {
      return is_inter_[cell(pos[i][0], pos[i][1])] == 0;
    };
    if (n == 2) {
      if (nsingle(0) && nsingle(1)) {
        return (check_backward(nref0(0)) ? 1 : 0) ^
               (check_backward(nref0(1)) ? 1 : 0);
      }
      if (nsingle(0)) {
        return 2 + ((check_backward(nref0(0)) || nintra(0)) ? 1 : 0);
      }
      if (nsingle(1)) {
        return 2 + ((check_backward(nref0(1)) || nintra(1)) ? 1 : 0);
      }
      return 4;
    }
    if (n == 1) {
      if (nsingle(0)) return check_backward(nref0(0)) ? 1 : 0;
      return 3;
    }
    return 1;
  }

  int is_joint_context(int ref0, int ref1) const {
    const int d0 = std::abs(map_.frame_distance[ref0]);
    const int d1 = std::abs(map_.frame_distance[ref1]);
    const bool same_side = map_.is_same_side(ref0, ref1);
    const bool restrict0 = static_cast<std::int32_t>(map_.order_hints[ref0]) == kRestrictedOrderHint;
    const bool restrict1 = static_cast<std::int32_t>(map_.order_hints[ref1]) == kRestrictedOrderHint;
    return (same_side || d0 != d1 || restrict0 != restrict1) ? 1 : 0;
  }

  int comp_group_context(int r, int c, int w4, int h4, int ref0,
                         int ref1) const {
    const int bck_order_hint = static_cast<int>(map_.order_hints[ref0]);
    const int fwd_order_hint = static_cast<int>(map_.order_hints[ref1]);
    const int bck = std::abs(get_relative_dist(
        static_cast<int>(map_.order_hint), fwd_order_hint));
    const int fwd = std::abs(get_relative_dist(
        bck_order_hint, static_cast<int>(map_.order_hint)));
    const int offset = (fwd == bck) ? 1 : 0;
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    int ctxs[2] = {0, 0};
    for (int i = 0; i < n && i < 2; ++i) {
      const std::size_t idx = cell(pos[i][0], pos[i][1]);
      if (is_inter_ref_value(ref1_[idx])) {
        ctxs[i] = comp_group_[idx];
      }
      // FurthestFuture (a future reference beyond the immediate pair) does not
      // exist in the two-reference M2 maps this gate writes.
    }
    return ctxs[1] + ctxs[0] + (std::min(ctxs[1], ctxs[0]) > 0 ? 1 : 0) +
           offset * 6;
  }

  bool code_do_split_context(int ctx, int value) {
    return sym(&cdfs_.do_split[0][ctx][0], 2, value);
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

  int count_refs(int ref, int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    int count = 0;
    for (int i = 0; i < n; ++i) {
      const std::size_t idx = cell(pos[i][0], pos[i][1]);
      if (ref0_[idx] == ref) ++count;
      if (ref1_[idx] == ref) ++count;
    }
    return count;
  }

  bool code_compound_ref(const CompoundRefPair& pair, int r, int c, int w4,
                         int h4, std::string& error) {
    const int n_total = map_.num_total_refs;
    if (n_total < 2) {
      error = "compound block with NumTotalRefs < 2";
      return false;
    }
    const int same = std::min(map_.num_same_ref_compound, n_total);
    int ref_frame[2] = {n_total - 1, n_total - 1};
    int n_found = 0;
    for (int ref = 0; ref < n_total - 1 && n_found < 2; ++ref) {
      int comp_ref = 0;
      bool implied = false;
      if (n_found == 0 && ref == 2) {
        comp_ref = 1;
        implied = true;
      } else if (n_found == 0 && ref + 1 >= same &&
                 ref + 1 == n_total - 1) {
        comp_ref = 1;
        implied = true;
      } else if (ref == pair.ref0 && n_found == 0) {
        comp_ref = 1;
      } else if (ref == pair.ref1 && n_found == 1) {
        comp_ref = 1;
      }
      if (!implied) {
        int ctx;
        const int this_count = count_refs(ref, r, c, w4, h4);
        int next_count = 0;
        for (int i = ref + 1; i < n_total; ++i) {
          next_count += count_refs(i, r, c, w4, h4);
        }
        if (this_count == next_count) {
          ctx = 1;
        } else if (this_count < next_count) {
          ctx = 0;
        } else {
          ctx = 2;
        }
        if (n_found == 0) {
          if (!sym(&cdfs_.comp_ref0[ctx][std::min(ref, 5)][0], 2, comp_ref)) {
            error = error_;
            return false;
          }
        } else {
          const int prev = ref_frame[0];
          const bool prev_forward =
              is_inter_ref_value(prev) && prev < n_total &&
              map_.frame_distance[prev] >= 0;
          const bool cur_forward =
              is_inter_ref_value(ref) && ref < n_total &&
              map_.frame_distance[ref] >= 0;
          const int bit_type = (prev_forward != cur_forward) ? 1 : 0;
          if (!sym(&cdfs_.comp_ref1[ctx][bit_type][std::min(ref, 5)][0], 2,
                   comp_ref)) {
            error = error_;
            return false;
          }
        }
      }
      if (comp_ref) {
        ref_frame[n_found] = ref;
        ++n_found;
        if (ref < same) --ref;
      }
    }
    if (ref_frame[0] != pair.ref0 || ref_frame[1] != pair.ref1) {
      error = "compound ref pair is not representable in this map";
      return false;
    }
    return true;
  }

  bool write_leaf(const GateLeaf& leaf, int r, int c, std::string& error) {
    const int w4 = 16;
    const int h4 = 16;
    const CompoundModeParams& mode = leaf.mode;
    if (!leaf.compound) {
      error = "gate tile writer only codes compound blocks";
      return false;
    }
    if (leaf.y_mode != kNearNearMv && leaf.y_mode != kGlobalGlobalMv) {
      error = "gate tile writer supports NEAR_NEARMV / GLOBAL_GLOBALMV only";
      return false;
    }
    if (mode.type == adv::CompoundType::kWedge &&
        !adv::bsize_has_wedge(static_cast<int>(BlockSize::k64x64))) {
      error = "wedge unsupported for 64x64";
      return false;
    }
    if (map_.num_total_refs < 2) {
      error = "compound leaf without two references";
      return false;
    }
    // read_is_inter / read_skip.
    if (!sym(&cdfs_.is_inter[is_inter_context(r, c, w4, h4)][0], 2, 1)) {
      error = error_;
      return false;
    }
    if (!sym(&cdfs_.skip[skip_context(r, c, w4, h4)][0], 2, 1)) {
      error = error_;
      return false;
    }
    // read_ref_frames: reference_select == 1.
    if (!sym(&cdfs_.comp_mode[comp_mode_context(r, c, w4, h4)][0], 2, 1)) {
      error = error_;
      return false;
    }
    CompoundRefPair pair;
    pair.ref0 = 0;
    pair.ref1 = 1;
    if (!code_compound_ref(pair, r, c, w4, h4, error)) return false;

    // YMode: non-joint compound.
    const int mode_ctx = new_mv_context(r, c, w4, h4);
    if (!sym(&cdfs_.is_joint[is_joint_context(pair.ref0, pair.ref1)][0], 2,
             0)) {
      error = error_;
      return false;
    }
    if (!sym(&cdfs_.compound_mode_non_joint[mode_ctx][0], 5,
             leaf.y_mode - kNearNearMv)) {
      error = error_;
      return false;
    }
    if (has_second_drl(leaf.y_mode)) {
      // The gate codes only distinct references, so the second read_drl_idx()
      // starts at 0.  Same-reference NEAR_NEARMV would start at RefMvIdx0 + 1
      // and is deliberately not produced here.
      if (pair.ref0 == pair.ref1) {
        error = "gate writer does not code same-reference compound";
        return false;
      }
      if (!code_drl_index(mode_ctx, leaf.drl_index[0])) {
        error = error_.empty() ? "drl0 encode failed" : error_;
        return false;
      }
      if (!code_drl_index(mode_ctx, leaf.drl_index[1])) {
        error = error_.empty() ? "drl1 encode failed" : error_;
        return false;
      }
    }

    // read_compound_type.
    const bool masked = mode.type != adv::CompoundType::kAverage;
    const int group_ctx =
        comp_group_context(r, c, w4, h4, pair.ref0, pair.ref1);
    if (!sym(&cdfs_.comp_group_idx[group_ctx][0], 2, masked ? 1 : 0)) {
      error = error_;
      return false;
    }
    if (masked) {
      const int type_symbol =
          mode.type == adv::CompoundType::kWedge ? 0 : 1;
      if (!sym(&cdfs_.compound_type[0], 2, type_symbol)) {
        error = error_;
        return false;
      }
      if (mode.type == adv::CompoundType::kWedge) {
        WedgeSyntax syntax;
        if (!wedge_index_to_syntax(mode.wedge_index, syntax)) {
          error = "wedge index is not signalable";
          return false;
        }
        if (!sym(&cdfs_.wedge_quad[0], 4, syntax.quad)) {
          error = error_;
          return false;
        }
        if (!sym(&cdfs_.wedge_angle[syntax.quad][0], 5, syntax.angle)) {
          error = error_;
          return false;
        }
        const int angle = syntax.quad * 5 + syntax.angle;
        const bool dist2 = angle >= 10 || angle == 5 || angle == 0;
        if (dist2) {
          if (!sym(&cdfs_.wedge_dist2[0], 3, syntax.dist - 1)) {
            error = error_;
            return false;
          }
        } else {
          if (!sym(&cdfs_.wedge_dist1[0], 4, syntax.dist)) {
            error = error_;
            return false;
          }
        }
        // The descriptor order is `read_wedge_mode(); L(1) wedge_sign;`.
        // read_wedge_mode() itself reads quad, angle and dist, then the raw
        // sign literal follows.
        if (enc_.encode_bool(mode.wedge_sign != 0) != av2::Status::kOk) {
          error = "wedge_sign encode failed";
          return false;
        }
      } else if (mode.type == adv::CompoundType::kDiffWtd) {
        if (enc_.encode_bool(mode.mask_type != 0) != av2::Status::kOk) {
          error = "mask_type encode failed";
          return false;
        }
      } else {
        error = "unsupported compound type";
        return false;
      }
    } else if (leaf.y_mode == kNearNearMv) {
      // enable_cwp: average + NEAR_NEARMV.
      int index = mode.cwp_index;
      if (index < 0 || index > kMaxCwpNum - 1) {
        error = "cwp index outside the signallable range";
        return false;
      }
      for (int idx = 0; idx < kMaxCwpNum - 1; ++idx) {
        const int bit = idx < index ? 1 : 0;
        if (!sym(&cdfs_.cwp_idx[idx][0], 2, bit)) {
          error = error_;
          return false;
        }
        if (bit == 0) break;
      }
    }

    // Update the context grids over the 64x64 leaf.
    for (int yy = 0; yy < h4; ++yy) {
      for (int xx = 0; xx < w4; ++xx) {
        const std::size_t idx = cell(r + yy, c + xx);
        is_inter_[idx] = 1;
        skip_[idx] = 1;
        ymode_[idx] = static_cast<std::uint8_t>(leaf.y_mode);
        ref0_[idx] = static_cast<std::int8_t>(pair.ref0);
        ref1_[idx] = static_cast<std::int8_t>(pair.ref1);
        comp_group_[idx] = masked ? 1 : 0;
      }
    }
    return true;
  }

  const GateTilePlan& plan_;
  const RefFrameMap& map_;
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
  std::vector<std::uint8_t> comp_group_;
};

}  // namespace

bool write_gate_compound_tile(const GateTilePlan& plan, const RefFrameMap& map,
                              const GateTileConfig& config,
                              std::vector<std::uint8_t>& payload,
                              std::string& error) {
  error.clear();
  payload.clear();
  std::string map_error;
  if (!validate_ref_frame_map(map, map_error)) {
    error = "invalid reference map: " + map_error;
    return false;
  }
  TileWriter writer(plan, map, config);
  if (!writer.run(nullptr, error)) return false;
  payload = writer.payload();
  return true;
}

}  // namespace av2enc::video::multiref::gate
