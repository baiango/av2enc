#include "codec/inter/drl_reorder/gate/drl_tile.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <set>

#include "codec/inter/drl_reorder/drl_syntax.h"
#include "codec/inter/drl_reorder/video_stack_adapter.h"
#include "pipeline/still/basic_cdf_tables.h"
#include "pipeline/still/still_tables.h"
#include "pipeline/video/video_cdf_tables.h"
#include "codec/partition/tree/partition.h"
#include "codec/entropy/av2/av2_entropy.h"

namespace av2enc::inter::drl::gate {
namespace {

using partition::BlockSize;
using partition::FrameGeometry;

constexpr int kYModeNearMv = 14;
constexpr int kTileCapacityBytes = 16 * 1024 * 1024;

struct Cdfs {
  std::uint16_t do_split[2][64][3];
  std::uint16_t do_square_split[2][8][3];
  std::uint16_t rect_type[2][64][3];
  std::uint16_t is_inter[4][3];
  std::uint16_t skip[6][3];
  std::uint16_t single_ref[3][6][3];
  std::uint16_t single_mode[5][4];
  std::uint16_t shell_set[3];
  std::uint16_t shell6_class0[9];
  std::uint16_t shell6_class1[9];
  std::uint16_t shell_last_two[3];
  std::uint16_t offset_low[2][3];
  std::uint16_t offset_class2[3];
  std::uint16_t offset_other[16][3];
  std::uint16_t col_mv_greater[2][3];
  std::uint16_t col_mv_index[4][3];
  DrlCdfState drl;
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

void init_cdfs(Cdfs* c) {
  copy_array(c->do_split, av2enc::basic::kDoSplitCdf);
  copy_array(c->do_square_split, av2enc::basic::kDoSquareSplitCdf);
  copy_array(c->rect_type, av2enc::basic::kRectTypeCdf);
  copy_array(c->is_inter, av2enc::video::kIsInterCdf);
  copy_array(c->skip, av2enc::video::kSkipCdf);
  copy_array(c->single_ref, av2enc::video::kSingleRefCdf);
  std::memcpy(c->single_mode, av2enc::video::kSingleModeCdf,
              sizeof(c->single_mode));
  copy_array(c->shell_set, av2enc::video::kJointShellSetCdf);
  copy_array(c->shell6_class0, av2enc::video::kJointShell6Class0Cdf);
  copy_array(c->shell6_class1, av2enc::video::kJointShell6Class1Cdf);
  copy_array(c->shell_last_two, av2enc::video::kJointShellLastTwoClassesCdf);
  copy_array(c->offset_low, av2enc::video::kShellOffsetLowClassCdf);
  copy_array(c->offset_class2, av2enc::video::kShellOffsetClass2Cdf);
  copy_array(c->offset_other, av2enc::video::kShellOffsetOtherClassCdf);
  copy_array(c->col_mv_greater, av2enc::video::kColMvGreaterCdf);
  copy_array(c->col_mv_index, av2enc::video::kColMvIndexCdf);
  init_default_drl_cdfs(&c->drl);
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

int floor_log2(int v) {
  int r = 0;
  while (v >= 2) {
    v >>= 1;
    ++r;
  }
  return r;
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
    y_mode_.assign(cells, 0);
    is_inter_.assign(cells, 0);
    skip_.assign(cells, 0);
    for (std::size_t i = 0; i < cells; ++i) {
      mi_[i].ref[0] = merge::kRefNone;
      mi_[i].ref[1] = merge::kRefNone;
      mi_[i].w4 = 1;
      mi_[i].h4 = 1;
    }
    above_mi_.assign(static_cast<std::size_t>(mi_cols_) + 16, 6);
    left_mi_.assign(static_cast<std::size_t>(mi_rows_) + 16, 6);
    init_cdfs(&cdfs_);
    buffer_.resize(kTileCapacityBytes);
    enc_.init_symbol(buffer_.data(), buffer_.size());
    enc_.disable_cdf_update = config_.disable_cdf_update ? 1 : 0;
  }

  bool run(const std::vector<LeafPlan>& leaves, TileOutput* out,
           std::string* error) {
    const int sb_cols = (mi_cols_ + 15) / 16;
    const int sb_rows = (mi_rows_ + 15) / 16;
    std::size_t expected = 0;
    if (config_.leaf_rect_16 != 0) {
      expected = static_cast<std::size_t>(mi_cols_ / 4) *
                 static_cast<std::size_t>(mi_rows_ / 4) * 2U;
    } else if (config_.leaf_size == 4) {
      // Two 4x8 / 8x4 leaves per 8x8 node.
      expected = static_cast<std::size_t>(mi_cols_ / 2) *
                 static_cast<std::size_t>(mi_rows_ / 2) * 2U;
    } else {
      const int leaf_mi = std::max(config_.leaf_size / 4, 1);
      expected = static_cast<std::size_t>(mi_cols_ / leaf_mi) *
                 static_cast<std::size_t>(mi_rows_ / leaf_mi);
    }
    if (leaves.size() != expected) {
      *error = "leaf plan size " + std::to_string(leaves.size()) +
               " does not match " + std::to_string(expected) + " 4x4 blocks";
      return false;
    }
    out->leaves.assign(leaves.size(), LeafStats{});
    std::size_t leaf_index = 0;
    for (int sb_r = 0; sb_r < sb_rows; ++sb_r) {
      for (int sb_c = 0; sb_c < sb_cols; ++sb_c) {
        if (!write_node(sb_r * 16, sb_c * 16, 16, 16, leaves, out,
                        &leaf_index, error)) {
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
    std::set<std::pair<std::int32_t, std::int32_t>> mvs;
    for (const LeafStats& s : out->leaves) {
      mvs.insert({s.chosen_mv.y, s.chosen_mv.x});
      if (s.divergent) ++out->divergent_blocks;
      if (s.spec_post != 0) ++out->reordered_blocks;
    }
    out->distinct_mvs = static_cast<int>(mvs.size());
    return true;
  }

 private:
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
    return ctx;  // skip_mode == 0
  }

  int count_refs(int ref, int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    int count = 0;
    for (int i = 0; i < n; ++i) {
      const inter::mc::MiInfo& m = mi_[cell(pos[i][0], pos[i][1])];
      if (m.ref[0] == ref) ++count;
      if (m.ref[1] == ref) ++count;
    }
    return count;
  }

  // §8.3 single_ref ctx, same formula as comp_ref: this vs all later refs.
  int single_ref_context(int ref, int r, int c, int w4, int h4) const {
    const int this_ref = count_refs(ref, r, c, w4, h4);
    int next_refs = 0;
    for (int i = ref + 1; i < 2; ++i) {
      next_refs += count_refs(i, r, c, w4, h4);
    }
    if (this_ref == next_refs) return 1;
    if (this_ref < next_refs) return 0;
    return 2;
  }

  int new_mv_context(int r, int c, int w4, int h4, int ref) const {
    merge::NewMvContextParams p;
    p.mi_row = r;
    p.mi_col = c;
    p.bw4 = w4;
    p.bh4 = h4;
    p.sb_h4 = 16;
    p.mi_cols = mi_cols_;
    p.mi_rows = mi_rows_;
    p.mi = mi_.data();
    p.y_mode = y_mode_.data();
    p.is_compound = false;
    p.ref_frame[0] = static_cast<std::int8_t>(ref);
    p.ref_frame[1] = merge::kRefNone;
    int ctx = 0;
    if (merge::derive_new_mv_context(p, &ctx) != merge::Status::kOk) {
      return 0;
    }
    return ctx;
  }

  bool code_do_split(int value, int r, int c, BlockSize size) {
    const int bsw = std::max(partition::mi_width_log2(size), 1);
    const int bsh = std::max(partition::mi_height_log2(size), 1);
    const int ctx = partition::partition_size_adjust(size) * 4 +
                    (left_mi(r) < bsh ? 2 : 0) +
                    (above_mi(c) < bsw ? 1 : 0);
    return sym(&cdfs_.do_split[0][ctx][0], 2, value);
  }

  bool code_do_square_split(int value, int r, int c, BlockSize size) {
    const int bsw = partition::mi_width_log2(size);
    const int bsh = partition::mi_height_log2(size);
    const bool avail_left = c > 0;
    const bool avail_above = r > 0;
    int ctx = 0;
    if (avail_left && left_mi(r) < bsh) ctx += 2;
    if (avail_above && above_mi(c) < bsw) ctx += 1;
    return sym(&cdfs_.do_square_split[0][ctx][0], 2, value);
  }

  bool code_rect_type(int value, int r, int c, BlockSize size) {
    const int bsw = std::max(partition::mi_width_log2(size), 1);
    const int bsh = std::max(partition::mi_height_log2(size), 1);
    const int ctx = partition::partition_size_adjust_rect_type(size) * 4 +
                    (left_mi(r) < bsh ? 2 : 0) +
                    (above_mi(c) < bsw ? 1 : 0);
    return sym(&cdfs_.rect_type[0][ctx][0], 2, value);
  }

  int& left_mi(int r) {
    return left_mi_[static_cast<std::size_t>(r)];
  }
  int& above_mi(int c) {
    return above_mi_[static_cast<std::size_t>(c)];
  }

  // Recursive PARTITION_HORZ / PARTITION_VERT ladder (PARTITION_SPLIT is only
  // valid at 128x128/256x256 in AV2 v1.0.0, so a 64x64 SB reaches 8x8 with
  // alternating rectangular partitions).
  bool write_node(int r, int c, int w4, int h4,
                  const std::vector<LeafPlan>& leaves, TileOutput* out,
                  std::size_t* leaf_index, std::string* error) {
    const BlockSize size = block_size_from_extent(w4, h4);
    if (size == BlockSize::kInvalid) {
      *error = "unsupported gate partition size";
      return false;
    }
    if (w4 == 4 && h4 == 4 && config_.leaf_rect_16 != 0) {
      if (!code_do_split(1, r, c, size)) {
        *error = error_;
        return false;
      }
      const bool vert = config_.leaf_rect_16 == 2;
      if (!code_rect_type(vert ? 1 : 0, r, c, size)) {
        *error = error_;
        return false;
      }
      const int w = vert ? 2 : 4;
      const int h = vert ? 4 : 2;
      for (int k = 0; k < 2; ++k) {
        if (*leaf_index >= leaves.size()) {
          *error = "leaf plan overrun";
          return false;
        }
        if (!write_leaf(r + (vert ? 0 : k * 2), c + (vert ? k * 2 : 0), w, h,
                        *leaf_index, leaves[*leaf_index], out, error)) {
          return false;
        }
        ++*leaf_index;
      }
      return true;
    }
    if (config_.leaf_size >= 8 && w4 * 4 == config_.leaf_size &&
        h4 * 4 == config_.leaf_size) {
      if (!code_do_split(0, r, c, size)) {
        *error = error_;
        return false;
      }
      if (*leaf_index >= leaves.size()) {
        *error = "leaf plan overrun";
        return false;
      }
      if (!write_leaf(r, c, w4, h4, *leaf_index, leaves[*leaf_index], out,
                      error)) {
        return false;
      }
      ++*leaf_index;
      return true;
    }
    if (w4 == 2 && h4 == 2) {  // 8x8: leaf or one rect split to 8x4/4x8
      if (config_.leaf_size == 8) {
        if (!code_do_split(0, r, c, size)) {
          *error = error_;
          return false;
        }
        if (*leaf_index >= leaves.size()) {
          *error = "leaf plan overrun";
          return false;
        }
        if (!write_leaf(r, c, 2, 2, *leaf_index, leaves[*leaf_index], out,
                        error)) {
          return false;
        }
        ++*leaf_index;
        return true;
      }
      const bool vertical = (((r >> 1) + (c >> 1)) & 1) == 0;
      if (!code_do_split(1, r, c, size)) {
        *error = error_;
        return false;
      }
      if (!code_rect_type(vertical ? 1 : 0, r, c, size)) {
        *error = error_;
        return false;
      }
      const int w = vertical ? 1 : 2;
      const int h = vertical ? 2 : 1;
      const int dr = vertical ? 0 : 1;
      const int dc = vertical ? 1 : 0;
      for (int k = 0; k < 2; ++k) {
        if (*leaf_index >= leaves.size()) {
          *error = "leaf plan overrun";
          return false;
        }
        if (!write_leaf(r + k * dr, c + k * dc, w, h, *leaf_index,
                        leaves[*leaf_index], out, error)) {
          return false;
        }
        ++*leaf_index;
      }
      return true;
    }
    if (!code_do_split(1, r, c, size)) {
      *error = error_;
      return false;
    }
    // PARTITION_SPLIT is invalid below 128x128, so no do_square_split symbol.
    const bool vertical = w4 >= h4;  // halve the longer dimension
    if (!code_rect_type(vertical ? 1 : 0, r, c, size)) {
      *error = error_;
      return false;
    }
    const int half_w = vertical ? w4 / 2 : w4;
    const int half_h = vertical ? h4 : h4 / 2;
    const int dr = vertical ? 0 : half_h;
    const int dc = vertical ? half_w : 0;
    for (int k = 0; k < 2; ++k) {
      if (!write_node(r + k * dr, c + k * dc, half_w, half_h, leaves, out,
                      leaf_index, error)) {
        return false;
      }
    }
    return true;
  }

  // Encoder-side decision for one 4x4 leaf.
  bool choose_candidate(int r, int c, int w4, int h4, const LeafPlan& plan,
                        LeafStats* stats, merge::MvStack* chosen,
                        Mode* active_mode, int* signalled_index,
                        std::string* error) {
    merge::StackParams p;
    p.mi_row = r;
    p.mi_col = c;
    p.bw4 = w4;
    p.bh4 = h4;
    p.sb_h4 = 16;
    p.mi_cols = mi_cols_;
    p.mi_rows = mi_rows_;
    p.mi = mi_.data();
    p.ref_frame[0] = plan.ref;
    p.ref_frame[1] = merge::kRefNone;
    p.is_compound = false;
    p.global_mv[0] = inter::mc::Mv{0, 0};
    p.global_mv[1] = inter::mc::Mv{0, 0};
    p.reorder = merge::DrlReorder::kDisabled;
    p.max_drl_bits_minus_1 = config_.max_drl_bits_minus_1;
    p.frame_distance = frame_distance_;
    p.num_refs = 2;
    merge::MvStack pre;
    if (merge::build_mv_stack(p, &pre) != merge::Status::kOk) {
      *error = "build_mv_stack failed";
      return false;
    }
    const int m = config_.max_drl_bits_minus_1 + 1;
    stats->stack_count = pre.count;
    stats->num_nearest = pre.num_nearest;
    stats->w4 = w4;
    stats->h4 = h4;
    drl::Stack view;
    drl::load_video_stack(pre, &view);
    stats->weights = drl::weight_string(view.cand, view.count);
    {
      std::string mvs;
      for (int i = 0; i < view.count; ++i) {
        if (i != 0) mvs += ';';
        mvs += std::to_string(view.cand[i].mv[0][0]) + "," +
               std::to_string(view.cand[i].mv[0][1]);
      }
      stats->stack_mvs = mvs;
    }
    if (pre.count <= 0) {
      // No spatial/global candidate: GLOBALMV would be the only legal mode.
      // The gate's grid always has the global candidate, so this is a
      // structural error rather than a fallback path.
      *error = "empty MV stack at a gate leaf";
      return false;
    }

    drl::Stack spec = view;
    drl::reorder_stack(&spec, config_.mode, config_.use_temporal_first);
    drl::Stack legacy = view;
    drl::reproduce_legacy_selection_sort(legacy.cand, legacy.num_nearest);

    if (plan.newmv) {
      // NEWMV seed block: use stack entry 0 as the predictor.  read_drl_idx()
      // is still present for NEWMV, so signal that entry's post-reorder index
      // and code the difference to the requested MV.
      drl::Stack spec_new = view;
      drl::reorder_stack(&spec_new, config_.mode, config_.use_temporal_first);
      drl::Stack legacy_new = view;
      drl::reproduce_legacy_selection_sort(legacy_new.cand,
                                          legacy_new.num_nearest);
      int spec_post = -1;
      int spec_perm[drl::kMaxRefMvStackSize];
      drl::reorder_permutation(view.cand, view.count, view.num_nearest,
                               config_.mode, config_.use_temporal_first,
                               spec_perm);
      drl::pre_to_post_index(spec_perm, view.count, 0, &spec_post);
      int legacy_post = 0;
      for (int k = 0; k < legacy_new.count; ++k) {
        if (legacy_new.cand[k].mv[0][0] == view.cand[0].mv[0][0] &&
            legacy_new.cand[k].mv[0][1] == view.cand[0].mv[0][1] &&
            legacy_new.cand[k].mv[1][0] == view.cand[0].mv[1][0] &&
            legacy_new.cand[k].mv[1][1] == view.cand[0].mv[1][1]) {
          legacy_post = k;
          break;
        }
      }
      const int signalled = config_.legacy_sort ? legacy_post : spec_post;
      if (spec_post < 0 || signalled < 0 || signalled >= m) {
        *error = "NEWMV predictor index is outside [0,m)";
        return false;
      }
      stats->pre_index = 0;
      stats->spec_post = spec_post;
      stats->legacy_post = legacy_post;
      stats->divergent = spec_post != legacy_post;
      stats->signalable = true;
      stats->signalled_index = signalled;
      stats->chosen_mv = plan.mv;
      stats->pred_mv = inter::mc::Mv{view.cand[0].mv[0][0],
                                     view.cand[0].mv[0][1]};
      stats->mvd = inter::mc::Mv{plan.mv.y - stats->pred_mv.y,
                                 plan.mv.x - stats->pred_mv.x};
      *chosen = pre;
      *active_mode = config_.mode;
      *signalled_index = signalled;
      return true;
    }

    const int usable = std::min(pre.count, m);
    int fallback = -1;
    int best = -1;
    bool best_divergent = false;
    for (int i = 0; i < usable; ++i) {
      int spec_post = -1;
      int legacy_post = -1;
      // Build the normative permutation; the legacy mapping is found by
      // scanning its own sorted array.
      int spec_perm[drl::kMaxRefMvStackSize];
      drl::reorder_permutation(view.cand, view.count, view.num_nearest,
                               config_.mode, config_.use_temporal_first,
                               spec_perm);
      if (!drl::pre_to_post_index(spec_perm, view.count, i, &spec_post)) {
        continue;
      }
      for (int k = 0; k < legacy.count; ++k) {
        if (legacy.cand[k].mv[0][0] == view.cand[i].mv[0][0] &&
            legacy.cand[k].mv[0][1] == view.cand[i].mv[0][1] &&
            legacy.cand[k].mv[1][0] == view.cand[i].mv[1][0] &&
            legacy.cand[k].mv[1][1] == view.cand[i].mv[1][1]) {
          legacy_post = k;
          break;
        }
      }
      if (fallback < 0 && spec_post >= 0 && spec_post < m) {
        fallback = i;
      }
      if (spec_post < 0 || spec_post >= m || legacy_post < 0 ||
          legacy_post >= m) {
        continue;
      }
      bool divergent = spec_post != legacy_post;
      if (divergent) {
        const auto& other = spec.cand[legacy_post].mv[0];
        divergent = other[0] != view.cand[i].mv[0][0] ||
                    other[1] != view.cand[i].mv[0][1];
      }
      if (divergent) {
        best = i;
        best_divergent = true;
        break;
      }
      if (best < 0) {
        best = i;
      }
    }
    if (best < 0) best = fallback;
    if (best < 0) {
      *error = "no signalable DRL candidate in [0,m)";
      return false;
    }
    // Recompute the two mappings for the chosen candidate.
    int spec_perm[drl::kMaxRefMvStackSize];
    drl::reorder_permutation(view.cand, view.count, view.num_nearest,
                             config_.mode, config_.use_temporal_first,
                             spec_perm);
    int spec_post = -1;
    drl::pre_to_post_index(spec_perm, view.count, best, &spec_post);
    int legacy_post = -1;
    for (int k = 0; k < legacy.count; ++k) {
      if (legacy.cand[k].mv[0][0] == view.cand[best].mv[0][0] &&
          legacy.cand[k].mv[0][1] == view.cand[best].mv[0][1] &&
          legacy.cand[k].mv[1][0] == view.cand[best].mv[1][0] &&
          legacy.cand[k].mv[1][1] == view.cand[best].mv[1][1]) {
        legacy_post = k;
        break;
      }
    }
    stats->pre_index = best;
    stats->spec_post = spec_post;
    stats->legacy_post = legacy_post;
    stats->divergent = best_divergent;
    stats->signalable = spec_post >= 0 && spec_post < m;
    stats->signalled_index = config_.legacy_sort ? legacy_post : spec_post;
    stats->chosen_mv = inter::mc::Mv{view.cand[best].mv[0][0],
                                     view.cand[best].mv[0][1]};
    *chosen = pre;
    *active_mode = config_.mode;
    *signalled_index = stats->signalled_index;
    if (*signalled_index < 0 || *signalled_index >= m) {
      *error = "chosen DRL index is outside [0,m)";
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

  bool write_literal(int bits, std::uint32_t value) {
    for (int i = bits - 1; i >= 0; --i) {
      if (!raw_bit(((value >> static_cast<unsigned>(i)) & 1U) != 0U)) {
        return false;
      }
    }
    return true;
  }

  bool write_ns_bits(std::uint32_t n, std::uint32_t value) {
    unsigned w = 0;
    while ((1U << w) < n) ++w;
    const std::uint32_t m = (1U << w) - n;
    if (value < m) return write_literal(static_cast<int>(w - 1U), value);
    const std::uint32_t x = value + m;
    if (!write_literal(static_cast<int>(w - 1U), x >> 1U)) return false;
    return raw_bit((x & 1U) != 0U);
  }

  // read_mv() §5.20.7.20 for MvPrecision = 1/8 pel (P = 6).
  bool write_mv(std::int32_t dy, std::int32_t dx) {
    const std::int32_t ay = std::abs(dy);
    const std::int32_t ax = std::abs(dx);
    const std::int32_t shell = ay + ax;
    if (shell >= (1 << 17)) return false;
    const int cls = shell == 0 ? 0 : floor_log2(static_cast<int>(shell));
    const std::int32_t base = cls == 0 ? 0 : (1 << cls);
    const std::int32_t off = shell - base;
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
      const std::int32_t max_pair = shell >> 1;
      bool skip_col_bit = false;
      std::int32_t col = 0;
      if (ax == max_pair && ((shell & 1) == 0)) {
        skip_col_bit = true;
        col = max_pair;
      } else if (ax <= max_pair) {
        col = ax;
      } else {
        col = ay;
      }
      if (max_pair > 0) {
        const int max_bits = std::min<std::int32_t>(max_pair, 2);
        const std::int32_t prefix_col = std::min<std::int32_t>(col, 2);
        for (int i = 0; i < max_bits; ++i) {
          const int greater = prefix_col > i ? 1 : 0;
          if (!sym(&cdfs_.col_mv_greater[i][0], 2, greater)) return false;
          if (greater == 0) break;
        }
        if (max_pair > 2 && prefix_col == 2) {
          if (!write_ns_bits(static_cast<std::uint32_t>(max_pair - 1),
                             static_cast<std::uint32_t>(col - 2))) {
            return false;
          }
        }
      }
      if (!skip_col_bit) {
        const int index = ax <= max_pair ? 0 : 1;
        if (!sym(&cdfs_.col_mv_index[std::min(cls, 3)][0], 2, index)) {
          return false;
        }
      }
    }
    if (dy != 0 && !raw_bit(dy < 0)) return false;
    if (dx != 0 && !raw_bit(dx < 0)) return false;
    return true;
  }

  bool write_leaf(int r, int c, int w4, int h4, std::size_t leaf_index,
                  const LeafPlan& plan, TileOutput* out,
                  std::string* error) {
    LeafStats stats;
    stats.mi_row = r;
    stats.mi_col = c;
    stats.ref = plan.ref;
    stats.w4 = w4;
    stats.h4 = h4;
    merge::MvStack chosen;
    Mode active = config_.mode;
    int signalled = 0;
    if (!choose_candidate(r, c, w4, h4, plan, &stats, &chosen, &active,
                          &signalled, error)) {
      return false;
    }
    const int new_mv_ctx = new_mv_context(r, c, w4, h4, plan.ref);
    stats.new_mv_context = new_mv_ctx;
    if (config_.force_global_mv) {
      stats.chosen_mv = mc::Mv{0, 0};
      stats.signalled_index = 0;
    }

    if (!sym(&cdfs_.is_inter[is_inter_context(r, c, w4, h4)][0], 2, 1)) {
      *error = error_;
      return false;
    }
    if (!sym(&cdfs_.skip[skip_context(r, c, w4, h4)][0], 2, 1)) {
      *error = error_;
      return false;
    }
    // reference_select == 0 -> no comp_mode bit.  With NumTotalRefs == 2 the
    // decoder reads one single_ref symbol at ref = 0; with one ref it reads
    // none and RefFrame[0] is implied 0.
    if (config_.num_total_refs > 1) {
      const int ref_ctx = single_ref_context(0, r, c, w4, h4);
      const int ref_symbol = plan.ref == 0 ? 1 : 0;
      if (!sym(&cdfs_.single_ref[ref_ctx][0][0], 2, ref_symbol)) {
        *error = error_;
        return false;
      }
    }
    if (config_.force_global_mv) {
      if (!sym(&cdfs_.single_mode[new_mv_ctx][0], 3, 1)) {
        *error = error_;
        return false;
      }
    } else {
    // single_mode: 0 = NEARMV, 2 = NEWMV; CDF row = NewMvContext.
    if (!sym(&cdfs_.single_mode[new_mv_ctx][0], 3, plan.newmv ? 2 : 0)) {
      *error = error_;
      return false;
    }
    if (config_.corrupt_index) {
      const int m = config_.max_drl_bits_minus_1 + 1;
      signalled = (signalled + 1) % m;
      stats.signalled_index = signalled;
    }
    DrlSymbolContext drl_ctx;
    drl_ctx.source = DrlCdfSource::kInter;
    drl_ctx.new_mv_context = new_mv_ctx;
    if (write_drl_index(&enc_, &cdfs_.drl, drl_ctx, signalled, 0,
                        config_.max_drl_bits_minus_1 + 1) != Status::kOk) {
      *error = "write_drl_index failed";
      return false;
    }
    if (plan.newmv && !write_mv(stats.mvd.y, stats.mvd.x)) {
      *error = "write_mv failed";
      return false;
    }
    }

    // Update the decoder-visible grids over this leaf.
    const inter::mc::Mv mv = stats.chosen_mv;
    for (int yy = 0; yy < h4; ++yy) {
      for (int xx = 0; xx < w4; ++xx) {
        const std::size_t idx = cell(r + yy, c + xx);
        is_inter_[idx] = 1;
        skip_[idx] = 1;
        y_mode_[idx] = config_.force_global_mv
                          ? 15
                          : (plan.newmv ? 16 : kYModeNearMv);
        inter::mc::MiInfo& m = mi_[idx];
        m.ref[0] = plan.ref;
        m.ref[1] = merge::kRefNone;
        m.mv[0][0] = mv.y;
        m.mv[0][1] = mv.x;
        m.mv[1][0] = mv.y;
        m.mv[1][1] = mv.x;
        m.is_inter = 1;
        m.w4 = static_cast<std::int8_t>(w4);
        m.h4 = static_cast<std::int8_t>(h4);
        m.col_base = static_cast<std::int16_t>(c);
        m.row_base = static_cast<std::int16_t>(r);
      }
    }
    const BlockSize bsize = block_size_from_extent(w4, h4);
    if (bsize == BlockSize::kInvalid) {
      *error = "unsupported gate leaf extent";
      return false;
    }
    for (int yy = 0; yy < h4; ++yy) {
      left_mi(r + yy) = partition::mi_height_log2(bsize);
    }
    for (int xx = 0; xx < w4; ++xx) {
      above_mi(c + xx) = partition::mi_width_log2(bsize);
    }
    out->leaves[leaf_index] = stats;
    return true;
  }

  TileConfig config_;
  FrameGeometry geometry_{};
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  // FrameDistance[0] = OrderHint(2) - 0 = 2, FrameDistance[1] = 2 - 1 = 1.
  const int frame_distance_[2] = {2, 1};

  Cdfs cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> buffer_;
  std::string error_;
  std::vector<inter::mc::MiInfo> mi_;
  std::vector<std::uint8_t> y_mode_;
  std::vector<std::uint8_t> is_inter_;
  std::vector<std::uint8_t> skip_;
  std::vector<int> above_mi_;
  std::vector<int> left_mi_;
};

}  // namespace

bool write_drl_tile(int width, int height, const std::vector<LeafPlan>& leaves,
                    const TileConfig& config, TileOutput* out,
                    std::string* error) {
  if (out == nullptr || error == nullptr) {
    return false;
  }
  error->clear();
  *out = TileOutput{};
  if (width <= 0 || height <= 0 || width % 64 != 0 || height % 64 != 0) {
    *error = "gate tile requires 64-multiple dimensions";
    return false;
  }
  TileWriter writer(width, height, config);
  return writer.run(leaves, out, error);
}

}  // namespace av2enc::inter::drl::gate
