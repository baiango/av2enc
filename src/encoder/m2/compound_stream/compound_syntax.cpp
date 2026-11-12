#include "encoder/m2/compound_stream/compound_syntax.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#include "encoder/m0_cdf_tables.h"
#include "encoder/m2/m2_cdf_tables.h"
#include "encoder/m2/multiref/gate/compound_cdf_tables.h"
#include "encoder/partition/partition.h"

namespace av2enc::m2::compound {
namespace {

using multiref::CompoundModeParams;
using multiref::WedgeSyntax;

// §9.3 defaults not (yet) generated into m2_cdf_tables.h.
inline constexpr std::uint16_t kSkipDrlModeCdf[3][3] = {
    21634, 0, 0, 17376, 0, 0, 18432, 75, 0,
};
inline constexpr std::uint16_t kUseOptflowCdf[2][3] = {
    11582, 31, 0, 24076, 26, 0,
};

bool is_inter_ref_value(int ref) { return ref >= 0 && ref < 8; }

}  // namespace

// ---------------------------------------------------------------------------
// Small §5.20.7.6 predicates.
// ---------------------------------------------------------------------------
bool has_newmv(int y_mode) {
  return y_mode == kNewMv || y_mode == kNewNewMv || y_mode == kNearNewMv ||
         y_mode == kNewNearMv || y_mode == kJointNewMv ||
         y_mode == kWarpNewMv;
}

bool has_nearmv(int y_mode) {
  return y_mode == kNearMv || y_mode == kNearNearMv ||
         y_mode == kNearNewMv || y_mode == kNewNearMv;
}

bool has_newmv_for_list(int y_mode, int list) {
  if (y_mode == kNewNewMv || y_mode == kNewMv) return true;
  if (list == 0) return y_mode == kNewNearMv || y_mode == kJointNewMv;
  return y_mode == kNearNewMv;
}

bool comp_ref_allowed(int block_w, int block_h) {
  const int m = block_w < block_h ? block_w : block_h;
  return m >= 8;
}

void derive_skip_mode_frames(const multiref::RefFrameMap& map,
                             int skip_mode_frame[2]) {
  skip_mode_frame[0] = 0;
  skip_mode_frame[1] = map.num_total_refs > 1 ? 1 : 0;
  if (map.num_total_refs > 1) {
    const int cur_to_ref0 = std::abs(map.frame_distance[0]);
    const int cur_to_ref1 = std::abs(map.frame_distance[1]);
    if (std::abs(cur_to_ref0 - cur_to_ref1) > 1) skip_mode_frame[1] = 0;
  }
}

bool CompoundFrameConfig::opfl_allowed(int ref0, int ref1) const {
  if (frame_is_switch || refs_scaled) return false;
  if (!is_inter_ref_value(ref0) || !is_inter_ref_value(ref1)) return false;
  const bool p0 = frame_distance(ref0) <= 0;
  const bool p1 = frame_distance(ref1) <= 0;
  return p0 != p1;
}

// ---------------------------------------------------------------------------
// Writer CDF state
// ---------------------------------------------------------------------------
struct ModeInfoWriter::Cdfs {
  std::uint16_t is_inter[4][3];
  std::uint16_t skip[6][3];
  std::uint16_t skip_mode[3][3];
  std::uint16_t single_ref[3][6][3];
  std::uint16_t comp_mode[5][3];
  std::uint16_t comp_ref0[3][6][3];
  std::uint16_t comp_ref1[3][2][6][3];
  std::uint16_t is_joint[2][3];
  std::uint16_t compound_mode_non_joint[5][6];
  std::uint16_t compound_mode_same_refs[5][5];
  std::uint16_t use_optflow[2][3];
  std::uint16_t single_mode[5][4];
  std::uint16_t drl_mode[3][5][3];
  std::uint16_t skip_drl_mode[3][3];
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

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------
ModeInfoWriter::ModeInfoWriter(const CompoundFrameConfig& config, int mi_cols,
                               int mi_rows)
    : config_(config),
      mi_cols_(mi_cols),
      mi_rows_(mi_rows),
      cells_(static_cast<std::size_t>(mi_cols) *
                 static_cast<std::size_t>(mi_rows),
             NeighbourCell{}),
      cdfs_(std::make_unique<Cdfs>()) {
  Cdfs& c = *cdfs_;
  std::memset(&c, 0, sizeof(c));
  copy_array(c.is_inter, m2::kIsInterCdf);
  copy_array(c.skip, m2::kSkipCdf);
  copy_array(c.skip_mode, m2::kSkipModeCdf);
  copy_array(c.single_ref, m2::kSingleRefCdf);
  copy_array(c.single_mode, m2::kSingleModeCdf);
  copy_array(c.drl_mode, m2::kDrlModeCdf);
  copy_array(c.skip_drl_mode, kSkipDrlModeCdf);
  copy_array(c.use_optflow, kUseOptflowCdf);
  copy_array(c.comp_mode, multiref::gate::kCompModeCdf);
  copy_array(c.comp_ref0, multiref::gate::kCompRef0Cdf);
  copy_array(c.comp_ref1, multiref::gate::kCompRef1Cdf);
  copy_array(c.is_joint, multiref::gate::kIsJointCdf);
  copy_array(c.compound_mode_non_joint,
             multiref::gate::kCompoundModeNonJointCdf);
  copy_array(c.compound_mode_same_refs,
             multiref::gate::kCompoundModeSameRefsCdf);
  copy_array(c.comp_group_idx, multiref::gate::kCompGroupIdxCdf);
  copy_array(c.compound_type, multiref::gate::kCompoundTypeCdf);
  copy_array(c.wedge_quad, multiref::gate::kWedgeQuadCdf);
  copy_array(c.wedge_angle, multiref::gate::kWedgeAngleCdf);
  copy_array(c.wedge_dist1, multiref::gate::kWedgeDist1Cdf);
  copy_array(c.wedge_dist2, multiref::gate::kWedgeDist2Cdf);
  copy_array(c.cwp_idx, multiref::gate::kCwpIdxCdf);
}

ModeInfoWriter::~ModeInfoWriter() = default;

bool ModeInfoWriter::sym(std::uint16_t* cdf, int n, int symbol,
                         std::string& error) {
  // CDF updates are driven by the caller's encoder setting, which must match
  // the frame header's disable_cdf_update.
  const av2::Status status = encoder_ref_->encode_symbol(cdf, n, symbol);
  if (status != av2::Status::kOk) {
    error = std::string("encode_symbol failed: ") + av2::status_name(status) +
            " n=" + std::to_string(n) + " sym=" + std::to_string(symbol);
    return false;
  }
  return true;
}

bool ModeInfoWriter::raw_bit(bool bit, std::string& error) {
  const av2::Status status = encoder_ref_->encode_bool(bit);
  if (status != av2::Status::kOk) {
    error = std::string("encode_bool failed: ") + av2::status_name(status);
    return false;
  }
  return true;
}

std::size_t ModeInfoWriter::cell_index(int r, int c) const {
  return static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols_) +
         static_cast<std::size_t>(c);
}

bool ModeInfoWriter::inside(int r, int c) const {
  return r >= 0 && r < mi_rows_ && c >= 0 && c < mi_cols_;
}

int ModeInfoWriter::neighbour_positions(int pos[2][2], int r, int c, int w4,
                                        int h4) const {
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

int ModeInfoWriter::count_refs(int r, int c, int w4, int h4, int ref) const {
  int pos[2][2] = {};
  const int n = neighbour_positions(pos, r, c, w4, h4);
  int count = 0;
  for (int i = 0; i < n; ++i) {
    const NeighbourCell& cell = cells_[cell_index(pos[i][0], pos[i][1])];
    if (cell.ref0 == ref) ++count;
    if (cell.ref1 == ref) ++count;
  }
  return count;
}

bool ModeInfoWriter::check_backward(int ref) const {
  if (ref == kTipRef) return true;
  if (!is_inter_ref_value(ref) || ref >= config_.num_total_refs()) return false;
  return config_.frame_distance(ref) < 0;
}

bool ModeInfoWriter::n_single(int r, int c) const {
  return !is_inter_ref_value(cells_[cell_index(r, c)].ref1);
}

int ModeInfoWriter::is_inter_context(int r, int c, int w4, int h4) const {
  int pos[2][2] = {};
  const int n = neighbour_positions(pos, r, c, w4, h4);
  const auto intra_at = [&](int i) {
    return cells_[cell_index(pos[i][0], pos[i][1])].is_inter == 0;
  };
  if (n == 2) {
    return (intra_at(0) && intra_at(1)) ? 3 : (intra_at(0) || intra_at(1));
  }
  if (n == 1) return 2 * (intra_at(0) ? 1 : 0);
  return 0;
}

int ModeInfoWriter::skip_context(int r, int c, int w4, int h4,
                                 bool skip_mode) const {
  int pos[2][2] = {};
  const int n = neighbour_positions(pos, r, c, w4, h4);
  int ctx = 0;
  for (int i = 0; i < n; ++i) {
    ctx += cells_[cell_index(pos[i][0], pos[i][1])].skip != 0 ? 1 : 0;
  }
  if (skip_mode) ctx += 3;  // SKIP_CONTEXTS >> 1
  return ctx;
}

int ModeInfoWriter::skip_mode_context(int r, int c, int w4, int h4) const {
  int pos[2][2] = {};
  const int n = neighbour_positions(pos, r, c, w4, h4);
  int ctx = 0;
  for (int i = 0; i < n; ++i) {
    ctx += cells_[cell_index(pos[i][0], pos[i][1])].skip_mode != 0 ? 1 : 0;
  }
  return ctx;
}

int ModeInfoWriter::new_mv_context(int r, int c, int w4, int h4,
                                   bool is_compound, int cur_ref0,
                                   int cur_ref1) const {
  int new_mv_count = 0;
  const bool sb_border = (r & 15) == 0;
  const auto scan = [&](int dr, int dc) -> int {
    const int mr = r + dr;
    const int mc = c + dc;
    if (!inside(mr, mc)) return 0;
    const NeighbourCell& cell = cells_[cell_index(mr, mc)];
    if (cell.is_inter == 0) return 0;
    bool found = false;
    if (is_compound) {
      if (cell.ref0 == cur_ref0 && cell.ref1 == cur_ref1) found = true;
      if (found && has_newmv(cell.ymode)) {
        new_mv_count = std::min(3, new_mv_count + 1);
      }
    } else {
      // §7.11.3 scans both reference lists even for single prediction, so a
      // compound neighbour whose list 1 matches RefFrame[0] also counts.
      for (int list = 0; list < 2; ++list) {
        const int candRef = list == 0 ? cell.ref0 : cell.ref1;
        if (candRef == cur_ref0) {
          if (has_newmv_for_list(cell.ymode, list)) {
            new_mv_count = std::min(3, new_mv_count + 1);
          }
          found = true;
          break;
        }
      }
    }
    return found ? 1 : 0;
  };
  const int left_a = scan(h4 - 1, -1);
  const int above_a = scan(-1, w4 - 1);
  const int left_b = scan(0, -1);
  const int above_b =
      (w4 >= (sb_border ? 4 : 2)) ? scan(-1, 0) : 0;
  const int nearest = ((above_a || above_b) ? 1 : 0) +
                      ((left_a || left_b) ? 1 : 0);
  return nearest + (new_mv_count > 0 ? 2 : 0);
}

int ModeInfoWriter::comp_mode_context(int r, int c, int w4, int h4) const {
  int pos[2][2] = {};
  const int n = neighbour_positions(pos, r, c, w4, h4);
  const auto single_at = [&](int i) { return n_single(pos[i][0], pos[i][1]); };
  const auto ref0_at = [&](int i) {
    return static_cast<int>(cells_[cell_index(pos[i][0], pos[i][1])].ref0);
  };
  const auto intra_at = [&](int i) {
    return cells_[cell_index(pos[i][0], pos[i][1])].is_inter == 0;
  };
  if (n == 2) {
    if (single_at(0) && single_at(1)) {
      return (check_backward(ref0_at(0)) ? 1 : 0) ^
             (check_backward(ref0_at(1)) ? 1 : 0);
    }
    if (single_at(0)) {
      return 2 + ((check_backward(ref0_at(0)) || intra_at(0)) ? 1 : 0);
    }
    if (single_at(1)) {
      return 2 + ((check_backward(ref0_at(1)) || intra_at(1)) ? 1 : 0);
    }
    return 4;
  }
  if (n == 1) {
    if (single_at(0)) return check_backward(ref0_at(0)) ? 1 : 0;
    return 3;
  }
  return 1;
}

int ModeInfoWriter::ref_context(int r, int c, int w4, int h4, int ref) const {
  const int this_count = count_refs(r, c, w4, h4, ref);
  int next_count = 0;
  for (int i = ref + 1; i < config_.num_total_refs(); ++i) {
    next_count += count_refs(r, c, w4, h4, i);
  }
  if (this_count == next_count) return 1;
  if (this_count < next_count) return 0;
  return 2;
}

int ModeInfoWriter::is_joint_context(int ref0, int ref1) const {
  const int d0 = std::abs(config_.frame_distance(ref0));
  const int d1 = std::abs(config_.frame_distance(ref1));
  const bool restrict0 = config_.order_hint_of(ref0) == -1;
  const bool restrict1 = config_.order_hint_of(ref1) == -1;
  return (config_.same_side(ref0, ref1) || d0 != d1 ||
          restrict0 != restrict1)
             ? 1
             : 0;
}

int ModeInfoWriter::comp_group_context(int r, int c, int w4, int h4, int ref0,
                                       int ref1) const {
  const int bck_order_hint = config_.order_hint_of(ref0);
  const int fwd_order_hint = config_.order_hint_of(ref1);
  const int bck =
      std::abs(multiref::get_relative_dist(static_cast<int>(config_.ref_map.order_hint),
                                           fwd_order_hint));
  const int fwd = std::abs(multiref::get_relative_dist(
      bck_order_hint, static_cast<int>(config_.ref_map.order_hint)));
  const int offset = (fwd == bck) ? 1 : 0;
  int pos[2][2] = {};
  const int n = neighbour_positions(pos, r, c, w4, h4);
  int ctxs[2] = {0, 0};
  const int furthest_future = config_.furthest_future();
  for (int i = 0; i < n && i < 2; ++i) {
    const NeighbourCell& cell = cells_[cell_index(pos[i][0], pos[i][1])];
    if (is_inter_ref_value(cell.ref1)) {
      ctxs[i] = cell.comp_group;
    } else if (furthest_future != kNoneRef && cell.ref0 == furthest_future) {
      // §9.3 comp_group_idx context: a single-reference neighbour that uses
      // FurthestFuture contributes 2.
      ctxs[i] = 2;
    }
  }
  return ctxs[1] + ctxs[0] + (std::min(ctxs[1], ctxs[0]) > 0 ? 1 : 0) +
         offset * 6;
}

void ModeInfoWriter::update_grid(const CompoundBlockResult& result) {
  for (int yy = 0; yy < result.h4; ++yy) {
    const int rr = result.mi_row + yy;
    if (rr >= mi_rows_) break;
    for (int xx = 0; xx < result.w4; ++xx) {
      const int cc = result.mi_col + xx;
      if (cc >= mi_cols_) break;
      NeighbourCell& cell = cells_[cell_index(rr, cc)];
      cell.is_inter = result.is_inter ? 1 : 0;
      cell.skip = result.skip_flag ? 1 : 0;
      cell.skip_mode = result.skip_mode ? 1 : 0;
      cell.ymode = static_cast<std::uint8_t>(result.y_mode);
      cell.comp_group = (result.compound.type != adv::CompoundType::kAverage &&
                         result.is_compound)
                            ? 1
                            : 0;
      cell.ref0 = static_cast<std::int8_t>(result.ref_frame[0]);
      cell.ref1 = static_cast<std::int8_t>(result.ref_frame[1]);
    }
  }
}

// ---------------------------------------------------------------------------
// Grammar
// ---------------------------------------------------------------------------
bool ModeInfoWriter::code_block(av2::Encoder& encoder,
                                const CompoundBlockPlan& plan,
                                CompoundBlockResult* result,
                                std::string& error) {
  error.clear();
  encoder_ref_ = &encoder;
  if (plan.w4 <= 0 || plan.h4 <= 0 || plan.mi_row < 0 || plan.mi_col < 0 ||
      plan.mi_row + plan.h4 > mi_rows_ || plan.mi_col + plan.w4 > mi_cols_) {
    error = "block geometry outside the tile";
    return false;
  }
  CompoundBlockResult res;
  res.mi_row = plan.mi_row;
  res.mi_col = plan.mi_col;
  res.w4 = plan.w4;
  res.h4 = plan.h4;
  res.skip_flag = plan.skip_flag;
  res.is_inter = true;

  // read_skip_mode() §5.20.5.9.
  const bool skip_eligible =
      config_.skip_mode_present &&
      comp_ref_allowed(plan.w4 * 4, plan.h4 * 4);
  if (skip_eligible) {
    if (!sym(&cdfs_->skip_mode[skip_mode_context(
                  plan.mi_row, plan.mi_col, plan.w4, plan.h4)][0],
             2, plan.skip_mode ? 1 : 0, error)) {
      return false;
    }
  } else if (plan.skip_mode) {
    error = "skip_mode requested but skip_mode_present/comp_ref_allowed is 0";
    return false;
  }
  res.skip_mode = skip_eligible && plan.skip_mode;
  if (res.skip_mode &&
      (config_.num_total_refs() < 2 ||
       config_.skip_mode_frame[0] >= config_.num_total_refs() ||
       config_.skip_mode_frame[1] >= config_.num_total_refs())) {
    error = "skip_mode needs two signallable skip_mode_frames";
    return false;
  }

  // read_is_inter() §5.20.7.3: skip_mode forces is_inter = 1.
  if (!res.skip_mode) {
    if (!sym(&cdfs_->is_inter[is_inter_context(plan.mi_row, plan.mi_col,
                                               plan.w4, plan.h4)][0],
             2, 1, error)) {
      return false;
    }
  }

  // read_skip() §5.20.5.10.
  if (!sym(&cdfs_->skip[skip_context(plan.mi_row, plan.mi_col, plan.w4,
                                     plan.h4, res.skip_mode)][0],
           2, plan.skip_flag ? 1 : 0, error)) {
    return false;
  }

  // inter_block_mode_info() §5.20.7.6.
  if (res.skip_mode) {
    res.ref_frame[0] = config_.skip_mode_frame[0];
    res.ref_frame[1] = config_.skip_mode_frame[1];
    if (!is_inter_ref_value(res.ref_frame[0]) ||
        !is_inter_ref_value(res.ref_frame[1])) {
      error = "skip_mode frames are not inter references";
      return false;
    }
  } else if (!code_ref_frames(plan, &res, error)) {
    return false;
  }
  res.is_compound = is_inter_ref_value(res.ref_frame[1]);
  if (res.is_compound) {
    if (res.ref_frame[0] == res.ref_frame[1]) {
      if (config_.num_same_ref_compound() < 1 ||
          res.ref_frame[0] >= config_.num_same_ref_compound()) {
        error = "same-reference compound is not signalled by this map";
        return false;
      }
    }
  }

  if (res.skip_mode) {
    if (plan.y_mode != kNearNearMv) {
      error = "skip_mode requires YMode NEAR_NEARMV";
      return false;
    }
    res.y_mode = kNearNearMv;
    res.use_optflow = false;
  } else if (res.is_compound) {
    if (!code_compound_mode(plan, &res, error)) return false;
  } else {
    // Single prediction (§5.20.7.6 else branch).
    if (res.ref_frame[0] == kTipRef) {
      error = "TIP blocks are not supported";
      return false;
    }
    if (config_.allow_warpmv_mode) {
      error = "warp-MV is not supported by this component";
      return false;
    }
    if (plan.y_mode < kNearMv || plan.y_mode > kNewMv) {
      error = "single-prediction YMode must be NEARMV/GLOBALMV/NEWMV";
      return false;
    }
    if (!sym(&cdfs_->single_mode[new_mv_context(plan.mi_row, plan.mi_col,
                                                plan.w4, plan.h4, false,
                                                res.ref_frame[0],
                                                res.ref_frame[1])][0],
             3, plan.y_mode - kNearMv, error)) {
      return false;
    }
    res.y_mode = plan.y_mode;
    res.use_optflow = false;
    if (res.y_mode == kNewMv) {
      error = "NEWMV MVD coding is out of scope for this component";
      return false;
    }
  }
  if (res.use_optflow && config_.opfl_refine_type == kRefineNone) {
    error = "use_optflow=1 with enable_opfl_refine=REFINE_NONE";
    return false;
  }

  // RefMvIdx / read_drl_idx() §5.20.7.8.
  if (!code_drl(plan, &res, error)) return false;

  // read_compound_type() §5.20.7.16.
  if (!code_compound_type(plan, &res, error)) return false;

  // cwp_idx §5.20.7.6.
  if (!code_cwp(plan, &res, error)) return false;

  // §5.20.7.6 p195: with REFINE_ALL, use_optflow is inferred after the
  // compound type and CWP weight are known.
  if (config_.opfl_refine_type == kRefineAll) {
    const bool derived =
        res.is_compound && !res.skip_mode && res.y_mode != kGlobalGlobalMv &&
        res.compound.type == adv::CompoundType::kAverage &&
        res.cwp_weight == adv::kCwpEqual &&
        config_.opfl_allowed(res.ref_frame[0], res.ref_frame[1]) &&
        std::min(plan.w4, plan.h4) * 4 >= 8;
    if (plan.use_optflow != derived) {
      error = "REFINE_ALL use_optflow inference does not match the decision";
      return false;
    }
    res.use_optflow = derived;
  }

  update_grid(res);
  if (result != nullptr) *result = res;
  return true;
}

bool ModeInfoWriter::code_ref_frames(const CompoundBlockPlan& plan,
                                     CompoundBlockResult* out,
                                     std::string& error) {
  const int n = config_.num_total_refs();
  if (n < 1) {
    error = "NumTotalRefs is zero";
    return false;
  }
  const bool compound =
      is_inter_ref_value(plan.ref_frame[1]);
  if (compound && plan.ref_frame[0] == plan.ref_frame[1]) {
    // Same-reference compound is legal only when NumSameRefCompound covers
    // the reference (§5.20.7.11).
    if (config_.num_same_ref_compound() < 1 ||
        plan.ref_frame[0] >= config_.num_same_ref_compound()) {
      error = "same-reference compound is not signalled by this map";
      return false;
    }
  }
  if (compound && !comp_ref_allowed(plan.w4 * 4, plan.h4 * 4)) {
    error = "compound prediction is not allowed for this block size";
    return false;
  }
  if (!compound && is_inter_ref_value(plan.ref_frame[1])) {
    error = "invalid RefFrame[1]";
    return false;
  }

  if (config_.reference_select() &&
      comp_ref_allowed(plan.w4 * 4, plan.h4 * 4)) {
    if (!sym(&cdfs_->comp_mode[comp_mode_context(plan.mi_row, plan.mi_col,
                                                 plan.w4, plan.h4)][0],
             2, compound ? 1 : 0, error)) {
      return false;
    }
  } else if (compound) {
    error = "compound block without reference_select/is_comp_ref_allowed";
    return false;
  }

  if (compound) {
    // Inverse of read_compound_ref() §5.20.7.11.
    const int same = config_.num_same_ref_compound();
    int ref_frame[2] = {n - 1, n - 1};
    int n_found = 0;
    for (int ref = 0; ref < n - 1 && n_found < 2; ++ref) {
      int comp_ref = 0;
      bool implied = false;
      if (n_found == 0 && ref == 2) {
        comp_ref = 1;
        implied = true;
      } else if (n_found == 0 && ref + 1 >= same && ref + 1 == n - 1) {
        comp_ref = 1;
        implied = true;
      } else if (ref == plan.ref_frame[0] && n_found == 0) {
        comp_ref = 1;
      } else if (ref == plan.ref_frame[1] && n_found == 1) {
        comp_ref = 1;
      }
      if (!implied) {
        const int ctx = ref_context(plan.mi_row, plan.mi_col, plan.w4,
                                    plan.h4, ref);
        if (n_found == 0) {
          if (!sym(&cdfs_->comp_ref0[ctx][std::min(ref, 5)][0], 2, comp_ref,
                   error)) {
            return false;
          }
        } else {
          const int prev = ref_frame[0];
          const bool prev_forward =
              is_inter_ref_value(prev) && prev < n &&
              config_.frame_distance(prev) >= 0;
          const bool cur_forward =
              is_inter_ref_value(ref) && ref < n &&
              config_.frame_distance(ref) >= 0;
          const int bit_type = (prev_forward != cur_forward) ? 1 : 0;
          if (!sym(&cdfs_->comp_ref1[ctx][bit_type][std::min(ref, 5)][0], 2,
                   comp_ref, error)) {
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
    if (ref_frame[0] != plan.ref_frame[0] ||
        ref_frame[1] != plan.ref_frame[1]) {
      error = "compound ref pair is not representable in this map";
      return false;
    }
    out->ref_frame[0] = ref_frame[0];
    out->ref_frame[1] = ref_frame[1];
    return true;
  }

  // Inverse of read_single_ref() §5.20.7.12.
  int chosen = -1;
  for (int ref = 0; ref < n - 1; ++ref) {
    const int bit = (plan.ref_frame[0] == ref) ? 1 : 0;
    if (!sym(&cdfs_->single_ref[ref_context(plan.mi_row, plan.mi_col, plan.w4,
                                            plan.h4, ref)][ref][0],
             2, bit, error)) {
      return false;
    }
    if (bit) {
      chosen = ref;
      break;
    }
  }
  if (chosen < 0) chosen = n - 1;
  if (chosen != plan.ref_frame[0]) {
    error = "single_ref selection is not representable in this map";
    return false;
  }
  out->ref_frame[0] = chosen;
  out->ref_frame[1] = kNoneRef;
  return true;
}

bool ModeInfoWriter::code_compound_mode(const CompoundBlockPlan& plan,
                                        CompoundBlockResult* out,
                                        std::string& error) {
  if (plan.y_mode == kNearNewMv || plan.y_mode == kNewNewMv ||
      plan.y_mode == kJointNewMv) {
    error = "compound NEWMV/JOINT modes are out of scope for this component";
    return false;
  }
  const int ctx = new_mv_context(plan.mi_row, plan.mi_col, plan.w4, plan.h4,
                                 true, out->ref_frame[0], out->ref_frame[1]);
  int y_mode = plan.y_mode;
  if (out->ref_frame[0] == out->ref_frame[1]) {
    // compound_mode_same_refs S(): values 0,1,2,3 map to
    // 19, 20, 22, 23 (NEAR_NEARMV, NEAR_NEWMV, GLOBAL_GLOBALMV, NEW_NEWMV).
    int symbol = -1;
    if (y_mode == kNearNearMv) symbol = 0;
    else if (y_mode == kNearNewMv) symbol = 1;
    else if (y_mode == kGlobalGlobalMv) symbol = 2;
    else if (y_mode == kNewNewMv) symbol = 3;
    if (symbol < 0) {
      error = "YMode is not signalled by compound_mode_same_refs";
      return false;
    }
    if (!sym(&cdfs_->compound_mode_same_refs[ctx][0], 4, symbol, error)) {
      return false;
    }
    y_mode = symbol < 2 ? kNearNearMv + symbol : kNearNearMv + symbol + 1;
  } else {
    const bool joint = y_mode == kJointNewMv;
    if (!sym(&cdfs_->is_joint[is_joint_context(out->ref_frame[0],
                                               out->ref_frame[1])][0],
             2, joint ? 1 : 0, error)) {
      return false;
    }
    if (joint) {
      error = "JOINT_NEWMV (joint MVD) is out of scope for this component";
      return false;
    }
    if (y_mode < kNearNearMv || y_mode > kNewNewMv) {
      error = "non-joint compound YMode outside 19..23";
      return false;
    }
    if (!sym(&cdfs_->compound_mode_non_joint[ctx][0], 5,
             y_mode - kNearNearMv, error)) {
      return false;
    }
  }
  if (y_mode == kNewNewMv || y_mode == kNearNewMv) {
    // These carry at least one NEWMV list and therefore an MVD.
    error = "compound NEWMV modes are out of scope for this component";
    return false;
  }
  out->y_mode = y_mode;

  // use_optflow §5.20.7.6.
  const bool optflow_eligible =
      y_mode != kGlobalGlobalMv &&
      config_.opfl_allowed(out->ref_frame[0], out->ref_frame[1]) &&
      std::min(plan.w4, plan.h4) * 4 >= 8;
  if (config_.opfl_refine_type == kRefineSwitchable && optflow_eligible) {
    if (!sym(&cdfs_->use_optflow[y_mode != kNearNearMv ? 1 : 0][0], 2,
             plan.use_optflow ? 1 : 0, error)) {
      return false;
    }
    out->use_optflow = plan.use_optflow;
  } else {
    // REFINE_ALL derives use_optflow after compound_type and cwp_idx
    // (§5.20.7.6 p195); it stays 0 for the DRL / compound-type steps.
    out->use_optflow = false;
    if (plan.use_optflow && config_.opfl_refine_type != kRefineAll) {
      error = "use_optflow decision does not match the frame's refine mode";
      return false;
    }
  }
  return true;
}

bool ModeInfoWriter::code_drl(const CompoundBlockPlan& plan,
                              CompoundBlockResult* out, std::string& error) {
  const int y_mode = out->y_mode;
  if (!has_nearmv(y_mode) && !has_newmv(y_mode)) return true;
  if (has_newmv(y_mode) && y_mode != kNewMv) {
    error = "NEWMV MVD coding is out of scope for this component";
    return false;
  }
  const int ctx = new_mv_context(plan.mi_row, plan.mi_col, plan.w4, plan.h4,
                                 out->is_compound, out->ref_frame[0],
                                 out->ref_frame[1]);
  const int m = config_.max_drl_bits_minus_1 + 1;
  const bool second_drl = (y_mode == kNearNearMv || y_mode == kNearNewMv) &&
                          !out->skip_mode && !out->use_optflow;

  const auto code_one = [&](int index, int cdf_ctx, int start) -> bool {
    if (index < start || index >= m) {
      error = "DRL index outside start..m-1";
      return false;
    }
    for (int idx = start; idx < m; ++idx) {
      const int mode = idx < index ? 1 : 0;
      std::uint16_t* cdf = nullptr;
      if (out->skip_mode) {
        cdf = &cdfs_->skip_drl_mode[std::min(idx, 2)][0];
      } else {
        cdf = &cdfs_->drl_mode[std::min(idx, 2)][cdf_ctx][0];
      }
      if (!sym(cdf, 2, mode, error)) return false;
      if (mode == 0) break;
    }
    return true;
  };

  if (second_drl) {
    const int start =
        (out->ref_frame[0] == out->ref_frame[1] && y_mode == kNearNearMv)
            ? plan.drl_index[0] + 1
            : 0;
    if (!code_one(plan.drl_index[0], ctx, 0)) return false;
    if (!code_one(plan.drl_index[1], ctx, start)) return false;
    out->drl_index[0] = plan.drl_index[0];
    out->drl_index[1] = plan.drl_index[1];
    out->ref_mv_idx = plan.drl_index[0];
  } else {
    if (!code_one(plan.drl_index[0], ctx, 0)) return false;
    out->drl_index[0] = plan.drl_index[0];
    out->drl_index[1] = 0;
    out->ref_mv_idx = plan.drl_index[0];
  }
  return true;
}

bool ModeInfoWriter::code_compound_type(const CompoundBlockPlan& plan,
                                        CompoundBlockResult* out,
                                        std::string& error) {
  const adv::CompoundType type = plan.compound.type;
  if (out->skip_mode || out->use_optflow || !out->is_compound) {
    if (type != adv::CompoundType::kAverage) {
      error = "masked compound cannot be signalled for skip_mode/optflow/"
              "single blocks";
      return false;
    }
    out->compound = plan.compound;
    out->compound.type = adv::CompoundType::kAverage;
    return true;
  }
  const bool masked = type != adv::CompoundType::kAverage;
  if (masked && !config_.enable_masked_compound) {
    error = "masked compound requires enable_masked_compound = 1";
    return false;
  }
  const bool wedge_ok =
      adv::bsize_has_wedge(static_cast<int>(partition::BlockSize::k64x64));
  if (config_.enable_masked_compound) {
    const int ctx = comp_group_context(plan.mi_row, plan.mi_col, plan.w4,
                                       plan.h4, out->ref_frame[0],
                                       out->ref_frame[1]);
    if (!sym(&cdfs_->comp_group_idx[ctx][0], 2, masked ? 1 : 0, error)) {
      return false;
    }
    if (masked) {
      if (!wedge_ok) {
        if (type != adv::CompoundType::kDiffWtd) {
          error = "wedge is not available for this block size";
          return false;
        }
      } else {
        const int type_symbol =
            type == adv::CompoundType::kWedge ? 0 : 1;
        if (!sym(&cdfs_->compound_type[0], 2, type_symbol, error)) {
          return false;
        }
      }
      if (type == adv::CompoundType::kWedge) {
        WedgeSyntax syntax;
        if (!multiref::wedge_index_to_syntax(plan.compound.wedge_index,
                                             syntax)) {
          error = "wedge index is not signalable";
          return false;
        }
        if (!sym(&cdfs_->wedge_quad[0], 4, syntax.quad, error)) return false;
        if (!sym(&cdfs_->wedge_angle[syntax.quad][0], 5, syntax.angle,
                 error)) {
          return false;
        }
        const int angle = syntax.quad * 5 + syntax.angle;
        const bool dist2 = angle >= 10 || angle == 5 || angle == 0;
        if (dist2) {
          if (!sym(&cdfs_->wedge_dist2[0], 3, syntax.dist - 1, error)) {
            return false;
          }
        } else {
          if (!sym(&cdfs_->wedge_dist1[0], 4, syntax.dist, error)) {
            return false;
          }
        }
        if (!raw_bit(plan.compound.wedge_sign != 0, error)) return false;
      } else if (type == adv::CompoundType::kDiffWtd) {
        if (!raw_bit(plan.compound.mask_type != 0, error)) return false;
      } else {
        error = "unsupported masked compound type";
        return false;
      }
    }
  }
  out->compound = plan.compound;
  if (!masked) out->compound.type = adv::CompoundType::kAverage;
  return true;
}

bool ModeInfoWriter::code_cwp(const CompoundBlockPlan& plan,
                              CompoundBlockResult* out, std::string& error) {
  out->cwp_weight = adv::kCwpEqual;
  if (!config_.enable_cwp) {
    if (plan.compound.cwp_index != 0) {
      error = "cwp_idx cannot be signalled when enable_cwp = 0";
      return false;
    }
    return true;
  }
  if (!out->is_compound) return true;
  if (out->skip_mode) {
    // Derived: CwpIdx = RefStackCwp[RefMvIdx].  The caller supplies the
    // stack-derived value; no bits are coded.
    out->cwp_weight = plan.compound.cwp_weight;
    return true;
  }
  if (out->use_optflow) return true;
  if (out->compound.type != adv::CompoundType::kAverage) return true;
  if (out->y_mode != kNearNearMv) return true;
  const int index = plan.compound.cwp_index;
  if (index < 0 || index >= adv::kMaxCwpNum) {
    error = "cwp index outside 0..MAX_CWP_NUM-1";
    return false;
  }
  for (int idx = 0; idx < adv::kMaxCwpNum - 1; ++idx) {
    const int bit = idx < index ? 1 : 0;
    if (!sym(&cdfs_->cwp_idx[idx][0], 2, bit, error)) return false;
    if (bit == 0) break;
  }
  out->cwp_weight =
      multiref::cwp_weight_from_index(index, config_.same_side(
                                                 out->ref_frame[0],
                                                 out->ref_frame[1]));
  return true;
}

// ---------------------------------------------------------------------------
// Tile payload writer: one 64x64 PARTITION_NONE leaf per superblock.
// ---------------------------------------------------------------------------
bool write_compound_tile(const std::vector<CompoundBlockPlan>& blocks, int width,
                         int height, const CompoundFrameConfig& config,
                         std::vector<std::uint8_t>& payload, std::string& error) {
  error.clear();
  payload.clear();
  if (width <= 0 || height <= 0 || width % 64 != 0 || height % 64 != 0) {
    error = "compound tile requires dimensions divisible by 64";
    return false;
  }
  partition::FrameGeometry geometry;
  geometry.width = width;
  geometry.height = height;
  geometry.sb_size = 64;
  geometry.has_chroma = true;
  geometry.chroma_subsampling_x = true;
  geometry.chroma_subsampling_y = true;
  const int mi_cols = partition::mi_cols(geometry);
  const int mi_rows = partition::mi_rows(geometry);
  const int sb_cols = mi_cols / 16;
  const int sb_rows = mi_rows / 16;
  if (static_cast<int>(blocks.size()) != sb_cols * sb_rows) {
    error = "one block plan per superblock is required";
    return false;
  }

  partition::MiSizeGrid sizes;
  sizes.reset(geometry);

  constexpr std::size_t kCapacity = 8U * 1024U * 1024U;
  std::vector<std::uint8_t> buffer(kCapacity);
  av2::Encoder encoder;
  encoder.init_symbol(buffer.data(), buffer.size());
  encoder.disable_cdf_update = config.disable_cdf_update ? 1 : 0;

  std::uint16_t do_split[2][64][3];
  std::memcpy(do_split, m0::kDoSplitCdf, sizeof(do_split));

  ModeInfoWriter writer(config, mi_cols, mi_rows);
  std::string local_error;
  for (int sb = 0; sb < sb_cols * sb_rows; ++sb) {
    const int r = (sb / sb_cols) * 16;
    const int c = (sb % sb_cols) * 16;
    const auto allowed = partition::init_allowed_partitions(
        geometry, r, c, partition::BlockSize::k64x64, true, false);
    if (!allowed.is_allowed(partition::PartitionType::kNone)) {
      error = "PARTITION_NONE is not allowed at a superblock";
      return false;
    }
    const partition::PartitionContexts ctx = partition::partition_contexts(
        geometry, r, c, partition::BlockSize::k64x64, sizes);
    if (encoder.encode_symbol(&do_split[0][ctx.do_split][0], 2, 0) !=
        av2::Status::kOk) {
      error = "do_split encode failed";
      return false;
    }
    const CompoundBlockPlan& plan = blocks[static_cast<std::size_t>(sb)];
    if (plan.mi_row != r || plan.mi_col != c || plan.w4 != 16 ||
        plan.h4 != 16) {
      error = "block plan geometry does not match its superblock";
      return false;
    }
    if (!writer.code_block(encoder, plan, nullptr, local_error)) {
      error = local_error;
      return false;
    }
    sizes.fill_block(r, c, partition::BlockSize::k64x64);
  }
  std::size_t bytes = 0;
  const av2::Status status = encoder.exit_symbol(&bytes);
  if (status != av2::Status::kOk) {
    error = std::string("exit_symbol failed: ") + av2::status_name(status);
    return false;
  }
  payload.assign(buffer.begin(),
                 buffer.begin() + static_cast<std::ptrdiff_t>(bytes));
  return true;
}

}  // namespace av2enc::m2::compound
