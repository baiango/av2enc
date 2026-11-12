// Independent reader for the compound / skip-mode inter mode-info grammar.
//
// This file is deliberately separate from compound_syntax.cpp: it re-derives
// every §8.3.2 context from its own neighbour grid and its own §9.3 CDF copy,
// then reads the symbols in the normative order.  It is the round-trip check
// for the writer (and the basis for the payload-level gate self-check).
#include "codec/inter/compound/compound_syntax.h"

#include <algorithm>
#include <cstring>
#include <memory>

#include "pipeline/still/basic_cdf_tables.h"
#include "pipeline/video/video_cdf_tables.h"
#include "codec/inter/multiref/gate/compound_cdf_tables.h"
#include "codec/partition/tree/partition.h"

namespace av2enc::video::compound {
namespace {

using multiref::CompoundModeParams;
using multiref::WedgeSyntax;

inline constexpr std::uint16_t kSkipDrlModeCdf[3][3] = {
    21634, 0, 0, 17376, 0, 0, 18432, 75, 0,
};
inline constexpr std::uint16_t kUseOptflowCdf[2][3] = {
    11582, 31, 0, 24076, 26, 0,
};

bool is_inter_ref_value(int ref) { return ref >= 0 && ref < 8; }

}  // namespace

struct ModeInfoReader::Cdfs {
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

ModeInfoReader::ModeInfoReader(const CompoundFrameConfig& config, int mi_cols,
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
  copy_array(c.is_inter, av2enc::video::kIsInterCdf);
  copy_array(c.skip, av2enc::video::kSkipCdf);
  copy_array(c.skip_mode, av2enc::video::kSkipModeCdf);
  copy_array(c.single_ref, av2enc::video::kSingleRefCdf);
  copy_array(c.single_mode, av2enc::video::kSingleModeCdf);
  copy_array(c.drl_mode, av2enc::video::kDrlModeCdf);
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

ModeInfoReader::~ModeInfoReader() = default;

bool ModeInfoReader::code_symbol(av2::Decoder& decoder, std::uint16_t* cdf,
                                 int n, int* symbol, std::string& error) {
  const av2::Status status = decoder.decode_symbol(cdf, n, symbol);
  if (status != av2::Status::kOk) {
    error = std::string("decode_symbol failed: ") + av2::status_name(status);
    return false;
  }
  return true;
}

bool ModeInfoReader::code_literal(av2::Decoder& decoder, int n,
                                  std::uint32_t* value, std::string& error) {
  const av2::Status status = decoder.read_literal(n, value);
  if (status != av2::Status::kOk) {
    error = std::string("read_literal failed: ") + av2::status_name(status);
    return false;
  }
  return true;
}

std::size_t ModeInfoReader::cell_index(int r, int c) const {
  return static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols_) +
         static_cast<std::size_t>(c);
}

bool ModeInfoReader::inside(int r, int c) const {
  return r >= 0 && r < mi_rows_ && c >= 0 && c < mi_cols_;
}

int ModeInfoReader::neighbour_positions(int pos[2][2], int r, int c, int w4,
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

int ModeInfoReader::count_refs(int r, int c, int w4, int h4, int ref) const {
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

bool ModeInfoReader::check_backward(int ref) const {
  if (ref == kTipRef) return true;
  if (!is_inter_ref_value(ref) || ref >= config_.num_total_refs()) return false;
  return config_.frame_distance(ref) < 0;
}

bool ModeInfoReader::n_single(int r, int c) const {
  return !is_inter_ref_value(cells_[cell_index(r, c)].ref1);
}

int ModeInfoReader::is_inter_context(int r, int c, int w4, int h4) const {
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

int ModeInfoReader::skip_context(int r, int c, int w4, int h4,
                                 bool skip_mode) const {
  int pos[2][2] = {};
  const int n = neighbour_positions(pos, r, c, w4, h4);
  int ctx = 0;
  for (int i = 0; i < n; ++i) {
    ctx += cells_[cell_index(pos[i][0], pos[i][1])].skip != 0 ? 1 : 0;
  }
  if (skip_mode) ctx += 3;
  return ctx;
}

int ModeInfoReader::skip_mode_context(int r, int c, int w4, int h4) const {
  int pos[2][2] = {};
  const int n = neighbour_positions(pos, r, c, w4, h4);
  int ctx = 0;
  for (int i = 0; i < n; ++i) {
    ctx += cells_[cell_index(pos[i][0], pos[i][1])].skip_mode != 0 ? 1 : 0;
  }
  return ctx;
}

int ModeInfoReader::new_mv_context(int r, int c, int w4, int h4,
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

int ModeInfoReader::comp_mode_context(int r, int c, int w4, int h4) const {
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

int ModeInfoReader::ref_context(int r, int c, int w4, int h4, int ref) const {
  const int this_count = count_refs(r, c, w4, h4, ref);
  int next_count = 0;
  for (int i = ref + 1; i < config_.num_total_refs(); ++i) {
    next_count += count_refs(r, c, w4, h4, i);
  }
  if (this_count == next_count) return 1;
  if (this_count < next_count) return 0;
  return 2;
}

int ModeInfoReader::is_joint_context(int ref0, int ref1) const {
  const int d0 = std::abs(config_.frame_distance(ref0));
  const int d1 = std::abs(config_.frame_distance(ref1));
  const bool restrict0 = config_.order_hint_of(ref0) == -1;
  const bool restrict1 = config_.order_hint_of(ref1) == -1;
  return (config_.same_side(ref0, ref1) || d0 != d1 ||
          restrict0 != restrict1)
             ? 1
             : 0;
}

int ModeInfoReader::comp_group_context(int r, int c, int w4, int h4, int ref0,
                                       int ref1) const {
  const int bck_order_hint = config_.order_hint_of(ref0);
  const int fwd_order_hint = config_.order_hint_of(ref1);
  const int bck = std::abs(multiref::get_relative_dist(
      static_cast<int>(config_.ref_map.order_hint), fwd_order_hint));
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

void ModeInfoReader::update_grid(const CompoundBlockResult& result) {
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

bool ModeInfoReader::read_block(av2::Decoder& decoder, int mi_row, int mi_col,
                                int w4, int h4, CompoundBlockResult* result,
                                std::string& error) {
  error.clear();
  return read_block_internal(decoder, mi_row, mi_col, w4, h4, result, error);
}

bool ModeInfoReader::read_block_internal(av2::Decoder& decoder, int mi_row,
                                         int mi_col, int w4, int h4,
                                         CompoundBlockResult* result,
                                         std::string& error) {
  if (result == nullptr || w4 <= 0 || h4 <= 0 || mi_row < 0 || mi_col < 0 ||
      mi_row + h4 > mi_rows_ || mi_col + w4 > mi_cols_) {
    error = "reader: block geometry outside the tile";
    return false;
  }
  CompoundBlockResult res;
  res.mi_row = mi_row;
  res.mi_col = mi_col;
  res.w4 = w4;
  res.h4 = h4;

  // read_skip_mode().
  const bool skip_eligible =
      config_.skip_mode_present && comp_ref_allowed(w4 * 4, h4 * 4);
  bool skip_mode = false;
  if (skip_eligible) {
    int symbol = 0;
    if (!code_symbol(decoder, &cdfs_->skip_mode[skip_mode_context(
                           mi_row, mi_col, w4, h4)][0],
                     2, &symbol, error)) {
      return false;
    }
    skip_mode = symbol != 0;
  }
  res.skip_mode = skip_mode;
  res.is_inter = true;

  // read_is_inter(): skip_mode forces is_inter = 1 without a symbol.
  if (!skip_mode) {
    int symbol = 0;
    if (!code_symbol(decoder, &cdfs_->is_inter[is_inter_context(
                           mi_row, mi_col, w4, h4)][0],
                     2, &symbol, error)) {
      return false;
    }
    if (symbol != 1) {
      error = "reader: intra blocks are outside this component";
      return false;
    }
  }

  // read_skip().
  {
    int symbol = 0;
    if (!code_symbol(decoder, &cdfs_->skip[skip_context(mi_row, mi_col, w4,
                                                         h4, skip_mode)][0],
                     2, &symbol, error)) {
      return false;
    }
    res.skip_flag = symbol != 0;
  }

  // read_ref_frames().
  if (skip_mode) {
    res.ref_frame[0] = config_.skip_mode_frame[0];
    res.ref_frame[1] = config_.skip_mode_frame[1];
  } else {
    const int n = config_.num_total_refs();
    bool compound = false;
    if (config_.reference_select() && comp_ref_allowed(w4 * 4, h4 * 4)) {
      int symbol = 0;
      if (!code_symbol(decoder, &cdfs_->comp_mode[comp_mode_context(
                             mi_row, mi_col, w4, h4)][0],
                       2, &symbol, error)) {
        return false;
      }
      compound = symbol != 0;
    }
    if (compound) {
      // read_compound_ref().
      int ref_frame[2] = {n - 1, n - 1};
      int n_found = 0;
      for (int ref = 0; ref < n - 1 && n_found < 2; ++ref) {
        int comp_ref = 0;
        bool implied = false;
        if (n_found == 0 && ref == 2) {
          comp_ref = 1;
          implied = true;
        } else if (n_found == 0 && ref + 1 >= config_.num_same_ref_compound() &&
                   ref + 1 == n - 1) {
          comp_ref = 1;
          implied = true;
        }
        (void)implied;
        if (!implied) {
          const int ctx = ref_context(mi_row, mi_col, w4, h4, ref);
          if (n_found == 0) {
            int symbol = 0;
            if (!code_symbol(decoder, &cdfs_->comp_ref0[ctx][std::min(ref, 5)][0],
                             2, &symbol, error)) {
              return false;
            }
            comp_ref = symbol;
          } else {
            const int prev = ref_frame[0];
            const bool prev_forward = is_inter_ref_value(prev) && prev < n &&
                                      config_.frame_distance(prev) >= 0;
            const bool cur_forward = is_inter_ref_value(ref) && ref < n &&
                                     config_.frame_distance(ref) >= 0;
            const int bit_type = (prev_forward != cur_forward) ? 1 : 0;
            int symbol = 0;
            if (!code_symbol(decoder,
                             &cdfs_->comp_ref1[ctx][bit_type][std::min(ref, 5)][0],
                             2, &symbol, error)) {
              return false;
            }
            comp_ref = symbol;
          }
        }
        if (comp_ref) {
          ref_frame[n_found] = ref;
          ++n_found;
          if (ref < config_.num_same_ref_compound()) --ref;
        }
      }
      res.ref_frame[0] = ref_frame[0];
      res.ref_frame[1] = ref_frame[1];
    } else {
      // read_single_ref().
      int chosen = n - 1;
      for (int ref = 0; ref < n - 1; ++ref) {
        int symbol = 0;
        if (!code_symbol(decoder,
                         &cdfs_->single_ref[ref_context(mi_row, mi_col, w4, h4,
                                                        ref)][ref][0],
                         2, &symbol, error)) {
          return false;
        }
        if (symbol) {
          chosen = ref;
          break;
        }
      }
      res.ref_frame[0] = chosen;
      res.ref_frame[1] = kNoneRef;
    }
  }
  res.is_compound = is_inter_ref_value(res.ref_frame[1]);

  // YMode.
  if (skip_mode) {
    res.y_mode = kNearNearMv;
    res.use_optflow = false;
  } else if (res.is_compound) {
    const int ctx = new_mv_context(mi_row, mi_col, w4, h4, true,
                                   res.ref_frame[0], res.ref_frame[1]);
    if (res.ref_frame[0] == res.ref_frame[1]) {
      int symbol = 0;
      if (!code_symbol(decoder, &cdfs_->compound_mode_same_refs[ctx][0], 4,
                       &symbol, error)) {
        return false;
      }
      res.y_mode =
          symbol < 2 ? kNearNearMv + symbol : kNearNearMv + symbol + 1;
    } else {
      int joint = 0;
      if (!code_symbol(decoder, &cdfs_->is_joint[is_joint_context(
                             res.ref_frame[0], res.ref_frame[1])][0],
                       2, &joint, error)) {
        return false;
      }
      if (joint) {
        error = "reader: JOINT_NEWMV is outside this component";
        return false;
      }
      int symbol = 0;
      if (!code_symbol(decoder, &cdfs_->compound_mode_non_joint[ctx][0], 5,
                       &symbol, error)) {
        return false;
      }
      res.y_mode = kNearNearMv + symbol;
    }
    if (res.y_mode == kNearNewMv || res.y_mode == kNewNearMv ||
        res.y_mode == kNewNewMv) {
      error = "reader: compound NEWMV MVD is outside this component";
      return false;
    }
    const bool optflow_eligible =
        res.y_mode != kGlobalGlobalMv &&
        config_.opfl_allowed(res.ref_frame[0], res.ref_frame[1]) &&
        std::min(w4, h4) * 4 >= 8;
    if (config_.opfl_refine_type == kRefineSwitchable && optflow_eligible) {
      int symbol = 0;
      if (!code_symbol(decoder,
                       &cdfs_->use_optflow[res.y_mode != kNearNearMv ? 1 : 0][0],
                       2, &symbol, error)) {
        return false;
      }
      res.use_optflow = symbol != 0;
    } else {
      res.use_optflow = false;  // REFINE_ALL is derived after cwp_idx
    }
  } else {
    int symbol = 0;
    if (!code_symbol(decoder, &cdfs_->single_mode[new_mv_context(
                           mi_row, mi_col, w4, h4, false, res.ref_frame[0],
                           res.ref_frame[1])][0],
                     3, &symbol, error)) {
      return false;
    }
    res.y_mode = kNearMv + symbol;
    res.use_optflow = false;
    if (res.y_mode == kNewMv) {
      error = "reader: NEWMV MVD is outside this component";
      return false;
    }
  }

  // read_drl_idx().
  const int m = config_.max_drl_bits_minus_1 + 1;
  const bool second_drl = (res.y_mode == kNearNearMv ||
                           res.y_mode == kNearNewMv) &&
                          !res.skip_mode && !res.use_optflow;
  const int ctx = new_mv_context(mi_row, mi_col, w4, h4, res.is_compound,
                                 res.ref_frame[0], res.ref_frame[1]);
  const auto read_one = [&](int start, int* index) -> bool {
    for (int idx = start; idx < m; ++idx) {
      int mode = 0;
      std::uint16_t* cdf =
          res.skip_mode ? &cdfs_->skip_drl_mode[std::min(idx, 2)][0]
                        : &cdfs_->drl_mode[std::min(idx, 2)][ctx][0];
      if (!code_symbol(decoder, cdf, 2, &mode, error)) return false;
      if (mode == 0) {
        *index = idx;
        return true;
      }
    }
    *index = m;
    return true;
  };
  if (has_nearmv(res.y_mode) || has_newmv(res.y_mode)) {
    int index0 = 0;
    if (!read_one(0, &index0)) return false;
    res.drl_index[0] = index0;
    res.ref_mv_idx = index0;
    if (second_drl) {
      const int start = (res.ref_frame[0] == res.ref_frame[1] &&
                         res.y_mode == kNearNearMv)
                            ? index0 + 1
                            : 0;
      int index1 = 0;
      if (!read_one(start, &index1)) return false;
      res.drl_index[1] = index1;
    }
  }

  // read_compound_type().
  res.compound = CompoundModeParams{};
  if (res.skip_mode || res.use_optflow || !res.is_compound) {
    res.compound.type = adv::CompoundType::kAverage;
  } else {
    bool masked = false;
    if (config_.enable_masked_compound) {
      int group = 0;
      if (!code_symbol(decoder, &cdfs_->comp_group_idx[comp_group_context(
                             mi_row, mi_col, w4, h4, res.ref_frame[0],
                             res.ref_frame[1])][0],
                       2, &group, error)) {
        return false;
      }
      masked = group != 0;
    }
    if (!masked) {
      res.compound.type = adv::CompoundType::kAverage;
    } else {
      const bool wedge_ok = adv::bsize_has_wedge(
          static_cast<int>(partition::BlockSize::k64x64));
      int type_symbol = 1;  // COMPOUND_DIFFWTD when wedge is unavailable
      if (wedge_ok) {
        if (!code_symbol(decoder, &cdfs_->compound_type[0], 2, &type_symbol,
                         error)) {
          return false;
        }
      }
      if (type_symbol == 0) {
        int quad = 0;
        int angle = 0;
        if (!code_symbol(decoder, &cdfs_->wedge_quad[0], 4, &quad, error)) {
          return false;
        }
        if (!code_symbol(decoder, &cdfs_->wedge_angle[quad][0], 5, &angle,
                         error)) {
          return false;
        }
        const int flat_angle = quad * 5 + angle;
        int dist = 0;
        if (flat_angle >= 10 || flat_angle == 5 || flat_angle == 0) {
          int d2 = 0;
          if (!code_symbol(decoder, &cdfs_->wedge_dist2[0], 3, &d2, error)) {
            return false;
          }
          dist = d2 + 1;
        } else {
          if (!code_symbol(decoder, &cdfs_->wedge_dist1[0], 4, &dist,
                           error)) {
            return false;
          }
        }
        const int index = multiref::wedge_syntax_to_index(quad, angle, dist);
        if (index < 0) {
          error = "reader: wedge syntax is not a legal wedge index";
          return false;
        }
        bool sign = false;
        const av2::Status status = decoder.read_bool(&sign);
        if (status != av2::Status::kOk) {
          error = "reader: wedge_sign read failed";
          return false;
        }
        res.compound.type = adv::CompoundType::kWedge;
        res.compound.wedge_index = index;
        res.compound.wedge_sign = sign ? 1 : 0;
      } else {
        bool mask = false;
        const av2::Status status = decoder.read_bool(&mask);
        if (status != av2::Status::kOk) {
          error = "reader: mask_type read failed";
          return false;
        }
        res.compound.type = adv::CompoundType::kDiffWtd;
        res.compound.mask_type = mask ? 1 : 0;
      }
    }
  }

  // cwp_idx.
  res.cwp_weight = adv::kCwpEqual;
  if (config_.enable_cwp && res.is_compound && !res.skip_mode &&
      !res.use_optflow && res.compound.type == adv::CompoundType::kAverage &&
      res.y_mode == kNearNearMv) {
    int index = 0;
    for (int idx = 0; idx < adv::kMaxCwpNum - 1; ++idx) {
      int bit = 0;
      if (!code_symbol(decoder, &cdfs_->cwp_idx[idx][0], 2, &bit, error)) {
        return false;
      }
      index = idx;
      if (bit == 0) break;
      index = idx + 1;
    }
    res.compound.cwp_index = index;
    res.cwp_weight = multiref::cwp_weight_from_index(
        index, config_.same_side(res.ref_frame[0], res.ref_frame[1]));
  }

  if (config_.opfl_refine_type == kRefineAll) {
    res.use_optflow =
        res.is_compound && !res.skip_mode &&
        res.y_mode != kGlobalGlobalMv &&
        res.compound.type == adv::CompoundType::kAverage &&
        res.cwp_weight == adv::kCwpEqual &&
        config_.opfl_allowed(res.ref_frame[0], res.ref_frame[1]) &&
        std::min(w4, h4) * 4 >= 8;
  }

  update_grid(res);
  *result = res;
  return true;
}

bool read_compound_tile(std::span<const std::uint8_t> payload, int width,
                        int height, const CompoundFrameConfig& config,
                        std::vector<CompoundBlockResult>& blocks,
                        std::string& error) {
  error.clear();
  blocks.clear();
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

  partition::MiSizeGrid sizes;
  sizes.reset(geometry);

  av2::Decoder decoder;
  decoder.disable_cdf_update = config.disable_cdf_update ? 1 : 0;
  const av2::Status status =
      decoder.init_symbol(payload.data(), payload.size(), payload.size());
  if (status != av2::Status::kOk) {
    error = std::string("init_symbol failed: ") + av2::status_name(status);
    return false;
  }
  std::uint16_t do_split[2][64][3];
  std::memcpy(do_split, av2enc::basic::kDoSplitCdf, sizeof(do_split));

  ModeInfoReader reader(config, mi_cols, mi_rows);
  blocks.reserve(static_cast<std::size_t>(sb_cols) *
                 static_cast<std::size_t>(sb_rows));
  for (int sb = 0; sb < sb_cols * sb_rows; ++sb) {
    const int r = (sb / sb_cols) * 16;
    const int c = (sb % sb_cols) * 16;
    const partition::PartitionContexts ctx = partition::partition_contexts(
        geometry, r, c, partition::BlockSize::k64x64, sizes);
    int split = 0;
    const av2::Status split_status =
        decoder.decode_symbol(&do_split[0][ctx.do_split][0], 2, &split);
    if (split_status != av2::Status::kOk || split != 0) {
      error = "read_compound_tile: do_split must be 0";
      return false;
    }
    CompoundBlockResult block;
    if (!reader.read_block(decoder, r, c, 16, 16, &block, error)) {
      return false;
    }
    blocks.push_back(block);
    sizes.fill_block(r, c, partition::BlockSize::k64x64);
  }
  const av2::Status exit_status = decoder.exit_symbol();
  if (exit_status != av2::Status::kOk) {
    error = std::string("exit_symbol failed: ") +
            av2::status_name(exit_status);
    return false;
  }
  return true;
}

}  // namespace av2enc::video::compound
