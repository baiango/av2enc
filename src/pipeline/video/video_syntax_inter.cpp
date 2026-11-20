// AV2 M2 P-frame tile syntax writer: inter block layer (§5.20.7).  is_inter/
// skip, reference selection, DRL, shell-coded MVD, compound/TIP motion modes
// and the §8.3 context grids they maintain.
#include "pipeline/video/video_syntax_writer.h"

#include "pipeline/video/video_interp_filter.h"

namespace av2enc::video::syntax {
namespace detail {

// add_neighbor() from decode_block(). NPosBuf keeps the first two valid
// positions in the order below-left, above-right, left, above.
int Writer::neighbor_positions(int pos[2][2]) const {
  int n = 0;
  const auto add = [&](int nr, int nc) {
    if (n < 2 && inside_mi(nr, nc)) {
      pos[n][0] = nr;
      pos[n][1] = nc;
      ++n;
    }
  };
  add(cur_r_ + cur_h4_ - 1, cur_c_ - 1);
  add(cur_r_ - 1, cur_c_ + cur_w4_ - 1);
  add(cur_r_, cur_c_ - 1);
  add(cur_r_ - 1, cur_c_);
  return n;
}

bool Writer::has_newmv_mode(int ymode) const { return ymode == 16; }

// §5.20.7.10 syntax gate for tip_mode: TipFrameMode != DISABLED, !skip_mode,
// Min(bw4, bh4) >= 2 and MiSize == ChromaMiSize.  This writer has no
// skip_mode and every inter leaf in a P frame owns its chroma block (SDP is
// keyframe-only), so the size predicate is the whole gate.
bool Writer::tip_mode_present() const { return plan_.tip_as_ref && cur_w4_ >= 2 && cur_h4_ >= 2; }

// §5.20.7.10 get_tip_ctx(): 0..2 = number of available neighbours (decoder
// line-buffer order: bottom-left, above-right, left, above) whose
// RefFrame[0] is TIP_FRAME.  Selects TileTipModeCdf[ctx].
int Writer::tip_context() const {
  int pos[2][2] = {};
  const int n = neighbor_positions(pos);
  int ctx = 0;
  for (int i = 0; i < n; ++i) {
    if (mv_grid_[cell_index(pos[i][0], pos[i][1])].ref[0] == merge::kTipFrame) {
      ++ctx;
    }
  }
  return ctx;
}

// find_mode_ctx() §7.11.2 for single prediction.
int Writer::new_mv_context() const {
  int new_mv_count = 0;
  const bool sb_border = (cur_r_ & 15) == 0;
  const auto scan = [&](int dr, int dc) -> int {
    const int mr = cur_r_ + dr;
    const int mc = cur_c_ + dc;
    if (!inside_mi(mr, mc)) return 0;
    const std::size_t idx = cell_index(mr, mc);
    if (is_inter_grid_[idx] == 0) return 0;
    if (has_newmv_mode(ymode_grid_[idx])) {
      new_mv_count = std::min(3, new_mv_count + 1);
    }
    return 1;
  };
  const int left_a = scan(cur_h4_ - 1, -1);
  const int above_a = scan(-1, cur_w4_ - 1);
  const int left_b = scan(0, -1);
  const int above_b =
      (cur_w4_ >= (sb_border ? 4 : 2)) ? scan(-1, 0) : 0;
  const int nearest = ((above_a || above_b) ? 1 : 0) +
                      ((left_a || left_b) ? 1 : 0);
  return nearest + (new_mv_count > 0 ? 2 : 0);
}

int Writer::is_inter_context() const {
  int pos[2][2] = {};
  const int n = neighbor_positions(pos);
  const auto intra_at = [&](int i) {
    return is_inter_grid_[cell_index(pos[i][0], pos[i][1])] == 0;
  };
  if (n == 2) {
    return (intra_at(0) && intra_at(1)) ? 3 : (intra_at(0) || intra_at(1));
  }
  if (n == 1) {
    return 2 * (intra_at(0) ? 1 : 0);
  }
  return 0;
}

int Writer::skip_context() const {
  int pos[2][2] = {};
  const int n = neighbor_positions(pos);
  int ctx = 0;
  for (int i = 0; i < n; ++i) {
    ctx += skip_grid_[cell_index(pos[i][0], pos[i][1])] != 0 ? 1 : 0;
  }
  return ctx;  // skip_mode == 0, so the +3 half-context is absent
}

bool Writer::code_is_inter(bool value) {
  return sym(&cdfs_.is_inter[is_inter_context()][0], 2, value ? 1 : 0);
}

bool Writer::code_skip_flag(bool value) {
  return sym(&cdfs_.skip[skip_context()][0], 2, value ? 1 : 0);
}

bool Writer::code_single_mode(int ctx, int single_mode) {
  return sym(&cdfs_.single_mode[ctx][0], 3, single_mode);
}

// §5.20.7.6 use_bawp + §7.13.3.25 sub-flags.  The encoder codes the fitted
// form (explicit_bawp = 0); `use_bawp_chroma` is always 0 for this path.
bool Writer::code_bawp(const BlockSyntax& node) {
  const int single_ctx =
      node.inter_mode == kInterNearMv
          ? 0
          : (node.inter_mode == kInterNewMv ? 2 : 0);
  if (!sym(&cdfs_.use_bawp[0], 2, node.use_bawp ? 1 : 0)) return false;
  if (!node.use_bawp) {
    if (node.explicit_bawp || node.explicit_bawp_scale ||
        node.use_bawp_chroma) {
      error_ = "BAWP sub-flags set while use_bawp is 0";
      return false;
    }
    return true;
  }
  if (!sym(&cdfs_.explicit_bawp[single_ctx][0], 2,
           node.explicit_bawp ? 1 : 0)) {
    return false;
  }
  if (node.explicit_bawp) {
    if (!sym(&cdfs_.explicit_bawp_scale[0], 2,
             node.explicit_bawp_scale ? 1 : 0)) {
      return false;
    }
  } else if (node.explicit_bawp_scale) {
    error_ = "explicit_bawp_scale set while explicit_bawp is 0";
    return false;
  }
  // HasChroma is always true in this 4:2:0 product path.
  if (!sym(&cdfs_.use_bawp_chroma[0], 2, node.use_bawp_chroma ? 1 : 0)) {
    return false;
  }
  return true;
}

int Writer::count_refs(int ref) const {
  int pos[2][2] = {};
  const int n = neighbor_positions(pos);
  int count = 0;
  for (int i = 0; i < n; ++i) {
    const inter::mc::MiInfo& mi =
        mv_grid_[cell_index(pos[i][0], pos[i][1])];
    if (mi.ref[0] == ref) ++count;
    if (mi.ref[1] == ref) ++count;
  }
  return count;
}

// §5.20.7.10 read_single_ref(): NumTotalRefs - 1 single_ref bits, decoded in
// reference order; the first set bit wins and the last reference is implied.
bool Writer::code_single_ref(int want) {
  const int num_refs = plan_.num_total_refs;
  if (num_refs <= 1) return true;
  if (want < 0 || want >= num_refs) return false;
  for (int ref = 0; ref < num_refs - 1; ++ref) {
    const int this_count = count_refs(ref);
    int next_count = 0;
    for (int i = ref + 1; i < num_refs; ++i) {
      next_count += count_refs(i);
    }
    const int ctx = this_count == next_count ? 1
                    : (this_count < next_count ? 0 : 2);
    const int bit = ref == want ? 1 : 0;
    if (trace_) {
      std::cerr << "  single_ref r=" << cur_r_ << " c=" << cur_c_
                << " " << cur_w4_ * 4 << "x" << cur_h4_ * 4
                << " want=" << want << " ref=" << ref << " bit=" << bit
                << " ctx=" << ctx << " this=" << this_count
                << " next=" << next_count << "\n";
    }
    if (!sym(&cdfs_.single_ref[ctx][ref][0], 2, bit)) return false;
    if (bit) return true;
  }
  return want == num_refs - 1;
}

// §5.20.7.11 read_compound_ref().  Writes exactly the comp_ref symbols the
// decoder reads for `node`'s (ref_index, ref_index1) pair.  NumSameRefCompound
// > 0 can repeat reference 0, which is the same-reference compound carrier.
bool Writer::code_compound_ref(const BlockSyntax& node) {
  const int n_total = plan_.num_total_refs;
  if (n_total < 2) return false;
  const int same = std::min<int>(config_.num_same_ref_compound, n_total);
  const int want0 = node.ref_index;
  const int want1 = node.ref_index1;
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
    } else if (n_found == 0 && ref == want0) {
      comp_ref = 1;
    } else if (n_found == 1 && ref == want1) {
      comp_ref = 1;
    }
    if (!implied) {
      // §9.3 comp_ref CDF context: compare this reference's neighbour
      // count with the sum of the later references.
      const int this_count = count_refs(ref);
      int next_count = 0;
      for (int i = ref + 1; i < n_total; ++i) {
        next_count += count_refs(i);
      }
      const int ctx = this_count == next_count
                          ? 1
                          : (this_count < next_count ? 0 : 2);
      if (n_found == 0) {
        if (!sym(&cdfs_.comp_ref0[ctx][std::min(ref, 5)][0], 2, comp_ref)) {
          return false;
        }
      } else {
        const int prev = ref_frame[0];
        const bool prev_forward =
            prev >= 0 && prev < n_total &&
            plan_.frame_distance[static_cast<std::size_t>(prev)] >= 0;
        const bool cur_forward =
            ref >= 0 && ref < n_total &&
            plan_.frame_distance[static_cast<std::size_t>(ref)] >= 0;
        const int bit_type = prev_forward != cur_forward ? 1 : 0;
        if (!sym(&cdfs_.comp_ref1[ctx][bit_type][std::min(ref, 5)][0], 2,
                 comp_ref)) {
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
  return ref_frame[0] == want0 && ref_frame[1] == want1;
}

bool Writer::code_drl_index(int ctx, int index, int start) {
  const int m = static_cast<int>(config_.max_drl_bits_minus_1) + 1;
  if (index < start || index > m) return false;
  for (int idx = start; idx < m; ++idx) {
    const int mode = idx < index ? 1 : 0;
    if (!sym(&cdfs_.drl_mode[std::min(idx, 2)][ctx][0], 2, mode)) {
      return false;
    }
    if (mode == 0) break;
  }
  return true;
}

// §5.20.7.8 read_drl_idx() with the §5.20.7.10 TIP CDF selection
// (TileTipDrlModeCdf[Min(idx, 2)]).  Used for a RefFrame[0] == TIP_FRAME
// leaf with either tip_pred_mode (NEARMV and NEWMV both code the index).
bool Writer::code_tip_drl_index(int index, int start) {
  const int m = static_cast<int>(config_.max_drl_bits_minus_1) + 1;
  if (index < start || index > m) return false;
  for (int idx = start; idx < m; ++idx) {
    const int mode = idx < index ? 1 : 0;
    if (!sym(&cdfs_.tip_drl_mode[std::min(idx, 2)][0], 2, mode)) {
      return false;
    }
    if (mode == 0) break;
  }
  return true;
}

bool Writer::write_ns_bits(std::uint32_t n, std::uint32_t value) {
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    return write_literal(static_cast<int>(w - 1U), value);
  }
  const std::uint32_t x = value + m;
  if (!write_literal(static_cast<int>(w - 1U), x >> 1U)) return false;
  return raw_bit((x & 1U) != 0U);
}

// read_mv() §5.20.7.20 for MvPrecision = 1/8 pel (P = 6).
bool Writer::write_mv(std::int32_t dy, std::int32_t dx) {
  const std::int32_t ay = std::abs(dy);
  const std::int32_t ax = std::abs(dx);
  const std::int32_t shell = ay + ax;
  // The shell index is bounded by 2^17 - 1 in the spec; MV_UPP/MV_LOW keep
  // each component in [-2^16, 2^16], so this is a defensive check.
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
      if (!sym(&cdfs_.offset_other[i][0], 2, (off >> i) & 1)) {
        return false;
      }
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

bool Writer::is_inter_ref_value(int ref) { return ref >= 0 && ref < 8; }

bool Writer::ref_backward(int ref) const {
  return is_inter_ref_value(ref) && ref < plan_.num_total_refs &&
         plan_.frame_distance[static_cast<std::size_t>(ref)] < 0;
}

// §5.20.7.10 comp_mode context (AV2 v1.0.0):
//   two neighbours with single prediction -> backward-ness mismatch
//   one single neighbour                  -> 2 + (backward or intra)
//   both compound                         -> 4
int Writer::comp_mode_context() const {
  int pos[2][2] = {};
  const int n = neighbor_positions(pos);
  const auto nsingle = [&](int i) {
    return !is_inter_ref_value(
        mv_grid_[cell_index(pos[i][0], pos[i][1])].ref[1]);
  };
  const auto nref0 = [&](int i) {
    return mv_grid_[cell_index(pos[i][0], pos[i][1])].ref[0];
  };
  const auto nintra = [&](int i) {
    return is_inter_grid_[cell_index(pos[i][0], pos[i][1])] == 0;
  };
  if (n == 2) {
    if (nsingle(0) && nsingle(1)) {
      return (ref_backward(nref0(0)) ? 1 : 0) ^
             (ref_backward(nref0(1)) ? 1 : 0);
    }
    if (nsingle(0)) {
      return 2 + ((ref_backward(nref0(0)) || nintra(0)) ? 1 : 0);
    }
    if (nsingle(1)) {
      return 2 + ((ref_backward(nref0(1)) || nintra(1)) ? 1 : 0);
    }
    return 4;
  }
  if (n == 1) {
    if (nsingle(0)) return ref_backward(nref0(0)) ? 1 : 0;
    return 3;
  }
  return 1;
}

int Writer::is_joint_context(int ref0, int ref1) const {
  const int d0 = std::abs(plan_.frame_distance[static_cast<std::size_t>(ref0)]);
  const int d1 = std::abs(plan_.frame_distance[static_cast<std::size_t>(ref1)]);
  const bool same_side =
      (plan_.frame_distance[static_cast<std::size_t>(ref0)] >= 0) ==
      (plan_.frame_distance[static_cast<std::size_t>(ref1)] >= 0);
  return (same_side || d0 != d1) ? 1 : 0;
}

// §7.8 FurthestFuture: the future reference (FrameDistance < 0) with the
// largest order hint.  With frame_distance = OrderHint - OrderHints[ref] the
// largest order hint is the largest (least negative) distance.
int Writer::furthest_future_ref() const {
  int best = -1;
  for (int ref = 0; ref < plan_.num_total_refs; ++ref) {
    const int dist = plan_.frame_distance[static_cast<std::size_t>(ref)];
    if (dist >= 0) continue;
    if (best < 0 ||
        dist > plan_.frame_distance[static_cast<std::size_t>(best)]) {
      best = ref;
    }
  }
  return best;
}

// §9.3 comp_group_idx context, matching the decoder's
// CompGroupIdxs/RefFrames neighbour scan.
int Writer::comp_group_context(int ref0, int ref1) const {
  const int bck = std::abs(plan_.frame_distance[static_cast<std::size_t>(ref1)]);
  const int fwd = std::abs(plan_.frame_distance[static_cast<std::size_t>(ref0)]);
  const int offset = (fwd == bck) ? 1 : 0;
  int pos[2][2] = {};
  const int n = neighbor_positions(pos);
  int ctxs[2] = {0, 0};
  const int furthest = furthest_future_ref();
  for (int i = 0; i < n && i < 2; ++i) {
    const std::size_t idx = cell_index(pos[i][0], pos[i][1]);
    const inter::mc::MiInfo& mi = mv_grid_[idx];
    if (mi.ref[1] >= 0 && mi.ref[1] < 8) {
      ctxs[i] = comp_group_grid_[idx] ? 1 : 0;
    } else if (furthest >= 0 && mi.ref[0] == furthest) {
      ctxs[i] = 2;
    }
  }
  return ctxs[1] + ctxs[0] +
         (std::min(ctxs[1], ctxs[0]) > 0 ? 1 : 0) + offset * 6;
}

// §5.20.7.10 read_ref_frames() + §5.20.7.6 compound mode for the gated
// subset: NumTotalRefs == 2 with num_same_ref_compound == 0, so the (0,1)
// pair is implied and no comp_ref bit is coded.  The only signalled compound
// mode is non-joint GLOBAL_GLOBALMV (compound_mode_non_joint = 3) with the
// AVERAGE blend: enable_masked_compound == 0 makes read_compound_type
// bitless and enable_cwp == 0 suppresses cwp_idx.
// Builds the §7.12.2 stack the decoder derives for one reference.  With
// `is_compound` the two-list stack (skip_mode / use_optflow form) is built;
// otherwise the single-reference stack for `ref`.  The global MVs come from
// the same §7.12.2.2 projection the writer stores in the MI grid.
bool Writer::build_mode_stack(const BlockSyntax& node, int ref, bool is_compound,
                      const BlockSyntax& mv_source,
                      merge::StackParams& params,
                      merge::MvStack& stack) const {
  params.mi_row = node.mi_row;
  params.mi_col = node.mi_col;
  params.bw4 = cur_w4_;
  params.bh4 = cur_h4_;
  params.sb_h4 = 16;
  params.mi_cols = mi_cols_;
  params.mi_rows = mi_rows_;
  params.mi = mv_grid_.data();
  params.ref_frame[0] = static_cast<std::int8_t>(ref);
  params.is_compound = is_compound;
  params.ref_frame[1] = is_compound
                            ? static_cast<std::int8_t>(mv_source.ref_index1)
                            : merge::kRefNone;
  params.global_mv[0] = block_global_mv_for_ref(mv_source, ref);
  params.global_mv[1] = is_compound
                            ? block_global_mv_for_ref(
                                  mv_source, mv_source.ref_index1)
                            : params.global_mv[0];
  params.reorder = merge::DrlReorder::kDisabled;
  params.max_drl_bits_minus_1 =
      static_cast<int>(config_.max_drl_bits_minus_1);
  if (plan_.num_total_refs > 1) {
    params.num_refs = plan_.num_total_refs;
    params.frame_distance = plan_.frame_distance.data();
  }
  return merge::build_stack_checked(params, &stack) == merge::Status::kOk;
}

// §7.13.3.8 opfl_allowed_for_refs() for this frame's reference pair.
bool Writer::opfl_allowed_for_pair(int ref0, int ref1) const {
  if (plan_.frame_is_switch) return false;
  const int d0 = plan_.frame_distance[static_cast<std::size_t>(ref0)];
  const int d1 = plan_.frame_distance[static_cast<std::size_t>(ref1)];
  return inter::temporal::opfl_allowed_for_refs(d0, d1, false, false, false);
}

// §5.20.7.17 non-GLOBALMV compound refine-MV decision (no bit: NEAR_NEARMV
// is not switchable).
bool Writer::refine_mv_derived(const BlockSyntax& node, bool* use) const {
  inter::temporal::RefineMvConfig ctx;
  ctx.enable_refinemv = plan_.enable_refinemv;
  ctx.opfl_refine_type =
      static_cast<inter::temporal::RefineType>(plan_.opfl_refine_type);
  ctx.frame_is_switch = plan_.frame_is_switch;
  ctx.is_compound = true;
  ctx.skip_mode = false;
  ctx.use_optflow = node.use_optflow != 0;
  ctx.is_global_global_mv = node.compound_mode == 3;
  ctx.motion_mode_simple = true;
  ctx.has_newmv = false;
  ctx.y_mode_near_near_mv = node.compound_mode == 0;
  ctx.y_mode_joint_new_mv = false;
  ctx.block_width = cur_w4_ * 4;
  ctx.block_height = cur_h4_ * 4;
  const int d0 = plan_.frame_distance[static_cast<std::size_t>(node.ref_index)];
  const int d1 =
      plan_.frame_distance[static_cast<std::size_t>(node.ref_index1)];
  // Synthetic origin: only the relative distances matter, and -1 is the
  // §5.18.3.1 restricted sentinel.
  ctx.order_hint = 64;
  ctx.ref0_order_hint = 64 - d0;
  ctx.ref1_order_hint = 64 - d1;
  inter::temporal::RefineMvSyntax syntax;
  if (inter::temporal::refine_mv_syntax(ctx, &syntax) !=
      inter::temporal::TemporalStatus::kOk) {
    return false;
  }
  *use = syntax.use_refinemv;
  return true;
}

bool Writer::write_compound_mode_info(const BlockSyntax& node, std::string& error) {
  if (!plan_.reference_select) {
    error = "compound block without reference_select";
    return false;
  }
  if (plan_.num_total_refs != 2) {
    error = "compound syntax is wired for NumTotalRefs == 2";
    return false;
  }
  const bool same_refs = node.ref_index == node.ref_index1;
  const bool same_ref_carrier =
      same_refs && config_.num_same_ref_compound > 0;
  if (same_refs && !same_ref_carrier) {
    error = "same-reference compound requires num_same_ref_compound > 0";
    return false;
  }
  if (same_ref_carrier) {
    if (node.ref_index != 0) {
      error = "same-reference compound is wired for reference pair (0, 0)";
      return false;
    }
    if (node.compound_mode != 3) {
      error = "same-reference compound is wired for GLOBAL_GLOBALMV only";
      return false;
    }
  } else if (node.ref_index != 0 || node.ref_index1 != 1) {
    error = "compound reference pair must be (0, 1)";
    return false;
  }
  if (!node.skip_flag) {
    error = "compound inter residual is not wired";
    return false;
  }
  // §5.20.7.6 non-joint values: 0 = NEAR_NEARMV (two per-list DRL indices),
  // 1 = NEAR_NEWMV (list-1 MVD), 2 = NEW_NEARMV (list-0 MVD), 3 =
  // GLOBAL_GLOBALMV.  is_joint = 1 selects JOINT_NEWMV and codes one joint
  // MVD instead of compound_mode_non_joint.
  const bool joint = node.compound_joint != 0;
  const bool near_near = !joint && node.compound_mode == 0;
  const bool near_new = !joint && node.compound_mode == 1;
  const bool new_near = !joint && node.compound_mode == 2;
  const bool global_global = !joint && node.compound_mode == 3;
  if (!joint && !near_near && !near_new && !new_near && !global_global) {
    error = "unsupported compound mode " +
            std::to_string(static_cast<int>(node.compound_mode));
    return false;
  }
  if ((joint || near_new || new_near) && same_refs) {
    error = "compound NEWMV modes require two distinct references";
    return false;
  }
  if ((joint || near_new || new_near) && plan_.enable_refinemv) {
    error = "compound NEWMV modes are not wired with enable_refinemv";
    return false;
  }
  if ((joint || near_new || new_near) && plan_.opfl_refine_type != 0) {
    error = "compound NEWMV modes are not wired with opfl_refine_type";
    return false;
  }
  if ((joint || near_new || new_near) &&
      (plan_.mv_precision != 0 || plan_.frame_force_integer_mv ||
       plan_.enable_adaptive_mvd || plan_.enable_flex_mvres ||
       plan_.enable_mvd_sign_derive)) {
    error = "compound NEWMV modes are wired for the plain 1/8-pel MVD path";
    return false;
  }
  if (!near_near && node.use_optflow) {
    error = "use_optflow is wired only on the NEAR_NEARMV carrier";
    return false;
  }
  if ((near_new || new_near || joint) && node.use_refinemv) {
    error = "compound refine-MV is wired only on the NEAR_NEARMV carrier";
    return false;
  }
  if (global_global && (node.inter_mode != kInterGlobalMv || node.mv_y != 0 ||
                        node.mv_x != 0 || node.mv1_y != 0 ||
                        node.mv1_x != 0)) {
    error = "GLOBAL_GLOBALMV compound must have zero MVs";
    return false;
  }
  // Validate the decision against the decoder's MV stack(s).  This is the
  // only place a compound MV can silently desynchronise.
  if (near_near) {
    merge::MvStack stacks[2];
    merge::StackParams params[2];
    inter::mc::Mv resolve[2];
    if (node.use_optflow) {
      merge::MvStack compound;
      merge::StackParams cparams;
      if (!build_mode_stack(node, 0, true, node, cparams, compound)) {
        error = "build_mv_stack (compound) failed";
        return false;
      }
      if (node.drl_index >= compound.count) {
        error = "compound DRL index outside the MV stack";
        return false;
      }
      resolve[0] = compound.cand[node.drl_index].mv[0];
      resolve[1] = compound.cand[node.drl_index].mv[1];
    } else {
      if (!build_mode_stack(node, 0, false, node, params[0], stacks[0]) ||
          !build_mode_stack(node, 1, false, node, params[1], stacks[1])) {
        error = "build_mv_stack (per-list) failed";
        return false;
      }
      if (merge::resolve_compound_near_near(stacks, node.drl_index,
                                            node.drl_index1, resolve) !=
          merge::Status::kOk) {
        error = "compound per-list DRL index outside the MV stack";
        return false;
      }
    }
    if (resolve[0].y != node.mv_y || resolve[0].x != node.mv_x ||
        resolve[1].y != node.mv1_y || resolve[1].x != node.mv1_x) {
      error = "compound MV does not match the stack candidates";
      return false;
    }
  } else if (near_new) {
    // §5.20.7.6 has_second_drl(): RefMvIdx0 selects the near list-0 entry
    // and RefMvIdx1 the list-1 predictor the new MV is coded against.
    merge::MvStack stacks[2];
    merge::StackParams params[2];
    if (!build_mode_stack(node, 0, false, node, params[0], stacks[0]) ||
        !build_mode_stack(node, 1, false, node, params[1], stacks[1])) {
      error = "build_mv_stack (NEAR_NEWMV) failed";
      return false;
    }
    if (node.drl_index < 0 || node.drl_index >= stacks[0].count ||
        node.drl_index1 < 0 || node.drl_index1 >= stacks[1].count) {
      error = "NEAR_NEWMV DRL index outside the per-list stacks";
      return false;
    }
    const inter::mc::Mv pred0 =
        merge::clamp_mv_to_integer(stacks[0].cand[node.drl_index].mv[0]);
    const inter::mc::Mv pred1 =
        merge::clamp_mv_to_integer(stacks[1].cand[node.drl_index1].mv[0]);
    const inter::mc::Mv expect0 = pred0;
    const inter::mc::Mv expect1 = merge::clamp_mv_to_integer(
        inter::mc::Mv{pred1.y + node.mvd_y, pred1.x + node.mvd_x});
    if (expect0.y != node.mv_y || expect0.x != node.mv_x ||
        expect1.y != node.mv1_y || expect1.x != node.mv1_x) {
      error = "NEAR_NEWMV MVs do not match the signalled MVD";
      return false;
    }
  } else if (new_near || joint) {
    // §5.20.7.6: NEW_NEARMV and JOINT_NEWMV have one RefMvIdx and read the
    // two-list stack (skip_mode form); both predictors come from that entry.
    merge::MvStack compound;
    merge::StackParams cparams;
    if (!build_mode_stack(node, 0, true, node, cparams, compound)) {
      error = "build_mv_stack (compound NEWMV) failed";
      return false;
    }
    if (node.drl_index < 0 || node.drl_index >= compound.count) {
      error = "compound NEWMV DRL index outside the two-list stack";
      return false;
    }
    inter::mc::Mv pred[2] = {
        merge::clamp_mv_to_integer(compound.cand[node.drl_index].mv[0]),
        merge::clamp_mv_to_integer(compound.cand[node.drl_index].mv[1])};
    inter::mc::Mv expect[2];
    if (new_near) {
      expect[0] = merge::clamp_mv_to_integer(
          inter::mc::Mv{pred[0].y + node.mvd_y, pred[0].x + node.mvd_x});
      expect[1] = pred[1];
    } else {
      const int d0 =
          plan_.frame_distance[static_cast<std::size_t>(node.ref_index)];
      const int d1 =
          plan_.frame_distance[static_cast<std::size_t>(node.ref_index1)];
      // §5.20.7.13 works in FrameDistance[] sign convention (negative = past);
      // this encoder stores OrderHint - OrderHints[ref], so negate it.
      const inter::mvprec::JointMvdPlan jmvd =
          inter::mvprec::joint_mvd_plan(-d0, -d1, false, false);
      inter::mvprec::assign_joint_mvd(
          pred, inter::mc::Mv{node.mvd_y, node.mvd_x}, jmvd,
          /*jmvd_scale_mode=*/0, /*use_amvd=*/false, expect);
    }
    if (expect[0].y != node.mv_y || expect[0].x != node.mv_x ||
        expect[1].y != node.mv1_y || expect[1].x != node.mv1_x) {
      error = "compound NEWMV MVs do not match the signalled MVD";
      return false;
    }
  }
  // §5.20.7.17: the derived refine-MV decision must match the prediction the
  // encoder applied.  NEAR_NEARMV is never switchable, so it codes no bit.
  bool derived_refinemv = false;
  if (near_near && !refine_mv_derived(node, &derived_refinemv)) {
    error = "refine_mv_syntax failed";
    return false;
  }
  if (derived_refinemv != (node.use_refinemv != 0)) {
    error = "compound refine-MV decision does not match §5.20.7.17";
    return false;
  }
  // §5.20.7.6 use_optflow is coded for a switchable frame over an allowed
  // reference pair, before the DRL indices and compound type.
  const bool optflow_eligible =
      near_near && !same_refs && plan_.opfl_refine_type == 1 &&
      opfl_allowed_for_pair(0, 1) && cur_w4_ * 4 >= 8 && cur_h4_ * 4 >= 8 &&
      !plan_.frame_is_switch;
  if (!optflow_eligible && node.use_optflow) {
    error = "use_optflow set outside the §5.20.7.6 gate";
    return false;
  }
  set_rate_tag(rate_hybrid::report::kTagCompMode);
  if (!sym(&cdfs_.comp_mode[comp_mode_context()][0], 2, 1)) {
    error = error_;
    return false;
  }
  // §5.20.7.11 read_compound_ref(): NumTotalRefs == 2 with
  // num_same_ref_compound == 0 implies (0, 1) with no bits; enabling the
  // same-reference carrier codes one comp_ref symbol.
  set_rate_tag(rate_hybrid::report::kTagSingleRef);
  if (!code_compound_ref(node)) {
    error = "write compound_ref failed";
    return false;
  }
  set_rate_tag(-1);
  merge::NewMvContextParams nparams;
  nparams.mi_row = node.mi_row;
  nparams.mi_col = node.mi_col;
  nparams.bw4 = cur_w4_;
  nparams.bh4 = cur_h4_;
  nparams.sb_h4 = 16;
  nparams.mi_cols = mi_cols_;
  nparams.mi_rows = mi_rows_;
  nparams.mi = mv_grid_.data();
  nparams.y_mode = ymode_grid_.data();
  nparams.is_compound = true;
  nparams.ref_frame[0] = static_cast<std::int8_t>(node.ref_index);
  nparams.ref_frame[1] = static_cast<std::int8_t>(node.ref_index1);
  int ctx = 0;
  if (merge::derive_new_mv_context(nparams, &ctx) != merge::Status::kOk) {
    error = "derive_new_mv_context failed";
    return false;
  }
  if (same_refs) {
    // §5.20.7.6: RefFrame[0] == RefFrame[1] selects
    // compound_mode_same_refs.  Value 2 is GLOBAL_GLOBALMV (0/1 map to
    // NEAR_NEARMV/NEAR_NEWMV and 3 to NEW_NEWMV).
    if (node.compound_mode != 3) {
      error = "same-reference compound expects GLOBAL_GLOBALMV";
      return false;
    }
    if (!sym(&cdfs_.compound_mode_same_refs[ctx][0], 4, 2)) {
      error = error_;
      return false;
    }
  } else {
    if (!sym(&cdfs_.is_joint[is_joint_context(0, 1)][0], 2,
             node.compound_joint)) {
      error = error_;
      return false;
    }
    if (!joint &&
        !sym(&cdfs_.compound_mode_non_joint[ctx][0], 5,
             node.compound_mode)) {
      error = error_;
      return false;
    }
  }
  if (optflow_eligible &&
      !sym(&cdfs_.use_optflow[node.compound_mode != 0 ? 1 : 0][0], 2,
           node.use_optflow ? 1 : 0)) {
    error = error_;
    return false;
  }
  // §5.20.7.6 jmvd_scale_mode is coded before the DRL index for
  // JOINT_NEWMV.  This writer only produces scale mode 0 (the raw
  // §7.9.4 projection).
  if (joint) {
    set_rate_tag(rate_hybrid::report::kTagMvd);
    std::string jmvd_error;
    if (!inter::mvprec::write_jmvd_scale_mode(
            enc_, cdfs_.mv_precision, /*use_amvd=*/false,
            /*mode=*/0, &jmvd_error)) {
      error = "write_jmvd_scale_mode failed: " + jmvd_error;
      return false;
    }
    set_rate_tag(-1);
  }
  if (near_near || near_new) {
    // §5.20.7.8 read_drl_idx(): has_second_drl() is true for a non-skip
    // NEAR_NEARMV/NEAR_NEWMV block without use_optflow, so both list indices
    // are coded.
    if (!code_drl_index(ctx, node.drl_index)) {
      error = "write drl_idx failed";
      return false;
    }
    if (!node.use_optflow) {
      int start = 0;
      const int drl_y_mode =
          near_near ? merge::kYModeNearNearMv : merge::kYModeNearNewMv;
      if (merge::derive_second_drl_start(
              drl_y_mode, node.ref_index == node.ref_index1, node.drl_index,
              &start) != merge::Status::kOk) {
        error = "derive_second_drl_start failed";
        return false;
      }
      if (!code_drl_index(ctx, node.drl_index1, start)) {
        error = "write second drl_idx failed";
        return false;
      }
    }
    // §5.20.7.17 read_refinemv(): only the switchable form codes a bit, and
    // NEAR_NEARMV is not switchable, so the decision above is implicit.
  } else if (new_near || joint) {
    // §5.20.7.6: NEW_NEARMV and JOINT_NEWMV code one RefMvIdx into the
    // two-list stack; there is no second DRL index.
    if (!code_drl_index(ctx, node.drl_index)) {
      error = "write drl_idx failed";
      return false;
    }
  }
  // §5.20.7.13 assign_mv(): NEAR_NEWMV codes the list-1 MVD, NEW_NEARMV the
  // list-0 MVD and JOINT_NEWMV one MVD for the derived base list.
  if (near_new || new_near || joint) {
    set_rate_tag(rate_hybrid::report::kTagMvd);
    if (!write_mv(node.mvd_y, node.mvd_x)) {
      error = error_;
      return false;
    }
    set_rate_tag(-1);
  }
  // §5.20.7.16 read_compound_type().  use_optflow (and the switchable
  // refine-MV form) select COMPOUND_AVERAGE and code no symbols.  Otherwise,
  // with enable_masked_compound the comp_group_idx symbol is present for
  // every compound block (0 = AVERAGE); a masked block then codes
  // compound_type and the mask symbols.
  if (node.use_optflow) {
    if (node.compound_masked) {
      error = "use_optflow compound must be COMPOUND_AVERAGE";
      return false;
    }
  } else if (plan_.enable_masked_compound) {
    if (node.compound_masked && plan_.num_total_refs != 2) {
      error = "masked compound is wired for two references";
      return false;
    }
    if (!sym(&cdfs_.comp_group_idx[comp_group_context(0, 1)][0], 2,
             node.compound_masked ? 1 : 0)) {
      error = error_;
      return false;
    }
    if (node.compound_masked) {
      const bool wedge_ok = inter::adv::bsize_has_wedge(
          static_cast<int>(node.bsize));
      if (node.compound_type == 0) {
        if (!wedge_ok) {
          error = "wedge compound is not available for this block size";
          return false;
        }
        if (!sym(&cdfs_.compound_type[0], 2, 0)) {
          error = error_;
          return false;
        }
        multiref::WedgeSyntax syntax;
        if (!multiref::wedge_index_to_syntax(node.wedge_index, syntax)) {
          error = "wedge index is not signalable";
          return false;
        }
        if (!sym(&cdfs_.wedge_quad[0], 4, syntax.quad) ||
            !sym(&cdfs_.wedge_angle[syntax.quad][0], 5, syntax.angle)) {
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
        } else if (!sym(&cdfs_.wedge_dist1[0], 4, syntax.dist)) {
          error = error_;
          return false;
        }
        if (!raw_bit(node.wedge_sign != 0)) {
          error = error_;
          return false;
        }
      } else if (node.compound_type == 1) {
        if (!sym(&cdfs_.compound_type[0], 2, 1)) {
          error = error_;
          return false;
        }
        if (!raw_bit(node.mask_type != 0)) {
          error = error_;
          return false;
        }
      } else {
        error = "invalid masked compound type";
        return false;
      }
    } else if (node.compound_type != 0) {
      error = "masked type without comp_group_idx";
      return false;
    }
  } else if (node.compound_masked) {
    error = "masked compound requires enable_masked_compound = 1";
    return false;
  }
  // §5.20.7.6 CwpIdx.  enable_cwp=1 codes cwp_idx for a compound
  // COMPOUND_AVERAGE SIMPLE-motion block that neither uses optflow nor
  // refine-MV, and only when YMode == NEAR_NEARMV (or JOINT_NEWMV with
  // jmvd_scale_mode == 0).  NEAR_NEWMV / NEW_NEARMV infer CWP_EQUAL and code
  // no bit.  With enable_cwp=0 the decoder infers CWP_EQUAL everywhere.
  if (!near_near && !joint && node.cwp_index != 0) {
    error = "cwp_idx is not signalled for this compound mode";
    return false;
  }
  const bool cwp_eligible =
      (near_near || joint) && !node.use_optflow && !node.use_refinemv &&
      !node.compound_masked;
  if (!plan_.enable_cwp) {
    if (node.cwp_index != 0) {
      error = "cwp_idx cannot be signalled when enable_cwp = 0";
      return false;
    }
  } else if (cwp_eligible) {
    if (node.cwp_index >= multiref::kMaxCwpNum) {
      error = "cwp_idx outside 0..MAX_CWP_NUM-1";
      return false;
    }
    for (int idx = 0; idx < multiref::kMaxCwpNum - 1; ++idx) {
      const int bit = idx < static_cast<int>(node.cwp_index) ? 1 : 0;
      if (!sym(&cdfs_.cwp_idx[idx][0], 2, bit)) {
        error = error_;
        return false;
      }
      if (bit == 0) break;
    }
  }
  // §5.20.7.6 switchable interp_filter, after the compound mode/type/CWP
  // symbols.  Compound leaves predict with EIGHTTAP here, so the decision
  // stores 0 in the carrier field; the symbol is still coded whenever the
  // §5.20.7.6 predicate requires it.
  if (interp_block_enabled()) {
    const bool large = cur_w4_ * 4 >= 8 && cur_h4_ * 4 >= 8;
    const int y_mode =
        joint ? inter::mvprec::kYModeJointNewMv
              : inter::mvprec::kYModeNearNearMv + node.compound_mode;
    if (inter::mvprec::needs_interp_filter(large, y_mode,
                                           /*motion_mode=*/0)) {
      const int filter_ctx = inter::mvprec::interp_filter_ctx(
          mv_res_neighbours(), node.ref_index, /*is_compound=*/true);
      std::string filter_error;
      if (!inter::mvprec::write_block_interp_filter(
              enc_, cdfs_.mv_precision, filter_ctx, node.mrl_index,
              &filter_error)) {
        error = "write_block_interp_filter failed: " + filter_error;
        return false;
      }
    } else if (node.mrl_index != 0) {
      error = "interp filter set outside the §5.20.7.6 gate";
      return false;
    }
  }
  return true;
}

// §5.20.7.6 / §5.20.7.8 for a leaf whose §5.20.7.10 tip_mode decoded as 1:
// Tip_Pred_Index_To_Mode -> {NEARMV, NEWMV}, then read_drl_idx() with the
// TileTipDrlModeCdf and (NEWMV only) the shell-coded MVD against the
// TIP-ref stack.  No comp_mode/single_ref/single_mode/use_bawp/motion-mode
// symbol follows a TIP reference.
bool Writer::write_tip_block_mode_info(const BlockSyntax& node, std::string& error) {
  if (node.compound) {
    error = "TIP block cannot be compound";
    return false;
  }
  if (node.tip_pred_mode > 1) {
    error = "invalid tip_pred_mode";
    return false;
  }
  if (plan_.mv_precision != 0 || plan_.frame_force_integer_mv || plan_.enable_adaptive_mvd ||
      plan_.enable_flex_mvres || plan_.enable_mvd_sign_derive) {
    // The MVD path is the historical 1/8-pel read_mv(); AMVD / flexible
    // resolution / derived sign change the symbol set for a NEWMV leaf.
    error = "TIP block syntax is wired for the plain 1/8-pel MVD path";
    return false;
  }
  // Rebuild the §7.12.2 single-list stack with RefFrame[0] == TIP_FRAME.
  // The builder reports kUnsupportedFeature when a TIP-derived candidate
  // (§7.12.2.15/.17/.18) would be needed; the producer only codes a TIP
  // leaf whose stack is reproducible.
  merge::StackParams params;
  params.mi_row = node.mi_row;
  params.mi_col = node.mi_col;
  params.bw4 = cur_w4_;
  params.bh4 = cur_h4_;
  params.sb_h4 = 16;
  params.mi_cols = mi_cols_;
  params.mi_rows = mi_rows_;
  params.mi = mv_grid_.data();
  params.ref_frame[0] = merge::kTipFrame;
  params.ref_frame[1] = merge::kRefNone;
  params.is_compound = false;
  // §7.12.2.2: GlobalMvs = {0, 0} when RefFrame[0] == TIP_FRAME.
  params.global_mv[0] = inter::mc::Mv{0, 0};
  params.global_mv[1] = params.global_mv[0];
  params.reorder = merge::DrlReorder::kDisabled;
  params.max_drl_bits_minus_1 = static_cast<int>(config_.max_drl_bits_minus_1);
  params.num_refs = plan_.num_total_refs;
  params.frame_distance = plan_.frame_distance.data();
  params.closest_past = plan_.temporal.closest_past;
  params.closest_future = plan_.temporal.closest_future;
  merge::MvStack stack;
  if (merge::build_stack_checked(params, &stack) != merge::Status::kOk) {
    error = "build_mv_stack (TIP) failed";
    return false;
  }
  if (stack.count <= 0) {
    error = "empty TIP MV stack";
    return false;
  }
  if (node.drl_index >= stack.count) {
    error = "TIP DRL index outside the MV stack";
    return false;
  }
  // §5.20.7.6 tip_pred_mode: 0 = NEARMV, 1 = NEWMV.
  if (!sym(&cdfs_.tip_pred_mode[0], 2, node.tip_pred_mode)) {
    error = error_;
    return false;
  }
  // §5.20.7.8 read_drl_idx() with TipDrlModeCdf.  Both TIP modes code it.
  if (!code_tip_drl_index(node.drl_index)) {
    error = error_;
    return false;
  }
  const merge::SingleMode mode =
      node.tip_pred_mode == 0 ? merge::SingleMode::kNearMv : merge::SingleMode::kNewMv;
  merge::ResolvedMode resolved;
  if (merge::resolve_single_mode(stack, params, mode, node.drl_index,
                                 inter::mc::Mv{node.mvd_y, node.mvd_x},
                                 &resolved) != merge::Status::kOk) {
    error = "TIP resolve_single_mode failed";
    return false;
  }
  if (resolved.mv[0].y != node.mv_y || resolved.mv[0].x != node.mv_x) {
    error = "TIP block MV does not match the signalled mode";
    return false;
  }
  if (node.tip_pred_mode == 1) {
    if (!write_mv(node.mvd_y, node.mvd_x)) {
      error = error_;
      return false;
    }
  } else if (node.mvd_y != 0 || node.mvd_x != 0) {
    error = "NEARMV TIP block carries an MVD";
    return false;
  }
  return true;
}

bool Writer::write_inter_mode_info(const BlockSyntax& node, std::string& error) {
  // §5.20.7.10 read_ref_frames() codes tip_mode before every other
  // reference/mode symbol, for every eligible inter leaf of a
  // TIP_FRAME_AS_REF frame.
  if (tip_mode_present()) {
    const int ctx = tip_context();
    if (ctx < 0 || ctx > 2) {
      error = "tip_mode context out of range";
      return false;
    }
    if (!sym(&cdfs_.tip_mode[ctx][0], 2, node.tip_mode ? 1 : 0)) {
      error = error_;
      return false;
    }
    if (node.tip_mode) {
      return write_tip_block_mode_info(node, error);
    }
  } else if (node.tip_mode) {
    error = "tip_mode set outside the §5.20.7.10 gate";
    return false;
  }
  if (node.compound) {
    return write_compound_mode_info(node, error);
  }
  if (node.inter_mode > kInterNewMv) {
    error = "invalid inter mode";
    return false;
  }
  if (node.ref_index >= plan_.num_total_refs) {
    error = "block reference index outside the frame reference map";
    return false;
  }

  // Validate the decision against the spec-exact merge/MV-stack layer.  This
  // catches producer bugs before they desynchronise the decoder.
  merge::StackParams params;
  params.mi_row = node.mi_row;
  params.mi_col = node.mi_col;
  params.bw4 = cur_w4_;
  params.bh4 = cur_h4_;
  params.sb_h4 = 16;
  params.mi_cols = mi_cols_;
  params.mi_rows = mi_rows_;
  params.mi = mv_grid_.data();
  params.ref_frame[0] = static_cast<std::int8_t>(node.ref_index);
  params.ref_frame[1] = merge::kRefNone;
  params.is_compound = false;
  // §7.12.2.2 GlobalMvs[0] for the MV-stack candidates.  With global motion
  // enabled the decoder projects the accepted model at the frame precision;
  // the encoder must feed the same value or a NEARMV/derived candidate can
  // resolve differently.
  const inter::mc::Mv global_mv = block_global_mv(node);
  params.global_mv[0] = global_mv;
  params.global_mv[1] = global_mv;
  // DRL reordering is not gated; the sequence header signals DISABLED.
  params.reorder = merge::DrlReorder::kDisabled;
  params.max_drl_bits_minus_1 =
      static_cast<int>(config_.max_drl_bits_minus_1);
  if (plan_.num_total_refs > 1) {
    params.num_refs = plan_.num_total_refs;
    params.frame_distance = plan_.frame_distance.data();
  }
  // §7.12.2 steps 5/12 temporal scan: the writer must rebuild the same
  // stack the decision path used, including the §7.9 motion-field
  // candidates.
  params.use_ref_frame_mvs = plan_.temporal.use_ref_frame_mvs;
  params.motion_field = plan_.temporal.field;
  params.temporal_scan = plan_.temporal.scan;
  params.order_hint = plan_.temporal.order_hint;
  params.order_hints = plan_.temporal.ref_order_hints.data();
  params.num_past_refs = plan_.temporal.num_past_refs;
  params.num_future_refs = plan_.temporal.num_future_refs;
  params.closest_past = plan_.temporal.closest_past;
  params.closest_future = plan_.temporal.closest_future;
  params.mv_traj = plan_.temporal.mv_traj;
  params.tile_mi_col_start = plan_.tile_mi_col;
  params.tile_mi_row_start = plan_.tile_mi_row;
  params.tile_mi_col_end = plan_.frame_mi_cols;
  params.tile_mi_row_end = plan_.frame_mi_rows;
  merge::MvStack stack;
  // The decoder's §7.12.2 stack includes the §7.12.2.21 bank candidates
  // when enable_refmvbank is set.  The encoder decision used them, so the
  // writer cannot rebuild the same stack without replaying the bank; the
  // decoder gate is the authority for this configuration.
  if (!config_.enable_refmvbank) {
  const drl::Mode reorder_mode =
      plan_.drl_reorder == 2
          ? drl::Mode::kAlways
          : (plan_.drl_reorder == 1 ? drl::Mode::kConstraint
                                    : drl::Mode::kDisabled);
  // §7.12.2 step 16 as the pinned decoder evaluates it: its
  // high-priority-TMVP term is not gated by use_ref_frame_mvs, so
  // DRL_REORDER_CONSTRAINT does not sort a single reference within two POCs.
  // See the matching gate in video_frame.cpp (load_stack).
  bool suppress = false;
  if (reorder_mode == drl::Mode::kConstraint) {
    bool past = false;
    bool future = false;
    for (int r = 0; r < plan_.num_total_refs && r < 7; ++r) {
      const int d = plan_.frame_distance[static_cast<std::size_t>(r)];
      if (d > 0) {
        past = true;
      } else if (d < 0) {
        future = true;
      }
    }
    if (!(past && future) && node.ref_index >= 0 && node.ref_index < 7) {
      const int d =
          plan_.frame_distance[static_cast<std::size_t>(node.ref_index)];
      suppress = (d < 0 ? -d : d) <= 2;
    }
  }
  const merge::Status stack_status =
      reorder_mode == drl::Mode::kDisabled
          ? merge::build_stack_checked(params, &stack)
          : drl::build_reordered_stack(params, reorder_mode, suppress,
                                       &stack);
  if (stack_status != merge::Status::kOk) {
    error = "build_mv_stack failed";
    return false;
  }

  if (node.inter_mode == kInterGlobalMv) {
    if (node.mv_y != 0 || node.mv_x != 0) {
      error = "GLOBALMV block has a non-zero MV";
      return false;
    }
  } else {
    if (stack.count <= 0) {
      error = "empty MV stack";
      return false;
    }
    const int drl = node.inter_mode == kInterNewMv ? 0 : node.drl_index;
    const merge::SingleMode mode = node.inter_mode == kInterNearMv
                                       ? merge::SingleMode::kNearMv
                                       : merge::SingleMode::kNewMv;
    if (plan_.mv_precision != 0 && node.inter_mode == kInterNewMv) {
      // The MVD is coded in precision units but the predictor is used
      // unchanged except for a frame coarser than 1/2 pel, where
      // §5.20.7.13 lowers it first.
      inter::mc::Mv pred = stack.cand[drl].mv[0];
      if (plan_.frame_force_integer_mv) {
        me2::me2_lower_mv_precision(&pred.y, &pred.x,
                                    me2::kMvPrecisionOnePel);
      }
      const inter::mc::Mv expected{pred.y + node.mvd_y, pred.x + node.mvd_x};
      if (expected.y != node.mv_y || expected.x != node.mv_x) {
        error = "block MV does not match the signalled inter mode";
        return false;
      }
    } else {
      merge::ResolvedMode resolved;
      if (merge::resolve_single_mode(
              stack, params, mode, drl,
              inter::mc::Mv{node.mvd_y, node.mvd_x}, &resolved) !=
          merge::Status::kOk) {
        error = "resolve_mode failed";
        return false;
      }
      if (resolved.mv[0].y != node.mv_y || resolved.mv[0].x != node.mv_x) {
        error = "block MV does not match the signalled inter mode";
        return false;
      }
    }
    if (node.inter_mode == kInterNearMv &&
        node.drl_index >= stack.count) {
      error = "DRL index outside the MV stack";
      return false;
    }
  }
  }

  if (plan_.num_total_refs > 1) {
    // §5.20.7.10 read_ref_frames(): when reference_select is on and the
    // block may be compound, comp_mode is coded first (0 = single ref).
    if (plan_.reference_select && cur_w4_ * 4 >= 8 && cur_h4_ * 4 >= 8) {
      set_rate_tag(rate_hybrid::report::kTagCompMode);
      if (!sym(&cdfs_.comp_mode[comp_mode_context()][0], 2, 0)) {
        error = error_;
        return false;
      }
      set_rate_tag(-1);
    }
    set_rate_tag(rate_hybrid::report::kTagSingleRef);
    if (!code_single_ref(node.ref_index)) {
      error = "write single_ref failed";
      return false;
    }
    set_rate_tag(-1);
  }

  int ctx = 0;
  if (plan_.num_total_refs > 1) {
    merge::NewMvContextParams nparams;
    nparams.mi_row = node.mi_row;
    nparams.mi_col = node.mi_col;
    nparams.bw4 = cur_w4_;
    nparams.bh4 = cur_h4_;
    nparams.sb_h4 = 16;
    nparams.mi_cols = mi_cols_;
    nparams.mi_rows = mi_rows_;
    nparams.mi = mv_grid_.data();
    nparams.y_mode = ymode_grid_.data();
    nparams.is_compound = false;
    nparams.ref_frame[0] = static_cast<std::int8_t>(node.ref_index);
    if (merge::derive_new_mv_context(nparams, &ctx) != merge::Status::kOk) {
      error = "derive_new_mv_context failed";
      return false;
    }
  } else {
    ctx = new_mv_context();
  }
  // YMode = NEARMV + single_mode. YMode 14 = NEARMV, 15 = GLOBALMV,
  // 16 = NEWMV, so single_mode is 0 / 1 / 2 in that order.
  const int single_mode =
      node.inter_mode == kInterGlobalMv
          ? 1
          : (node.inter_mode == kInterNearMv ? 0 : 2);
  set_rate_tag(rate_hybrid::report::kTagSingleMode);
  if (!code_single_mode(ctx, single_mode)) {
    error = error_;
    return false;
  }
  set_rate_tag(-1);
  // §5.20.7.6 use_amvd.  Present only for a NEWMV YMode (allow_amvd_mode()
  // covers NEWMV / NEW_NEWMV / NEAR_NEWMV / NEW_NEARMV / JOINT_NEWMV); this
  // writer produces single-reference NEWMV only.  The spec places it
  // immediately after single_mode and before use_bawp.
  const bool amvd_symbol_present =
      plan_.enable_adaptive_mvd && node.inter_mode == kInterNewMv;
  if (amvd_symbol_present) {
    const inter::mvprec::NeighborSet neigh = mv_res_neighbours();
    const int amvd_ctx = inter::mvprec::use_amvd_ctx(neigh, node.ref_index);
    std::string amvd_error;
    if (!inter::mvprec::write_use_amvd(
            enc_, cdfs_.mv_precision,
            inter::mvprec::use_amvd_cdf_index(inter::mvprec::kYModeNewMv,
                                              false),
            amvd_ctx, node.use_amvd != 0, &amvd_error)) {
      error = "write_use_amvd failed: " + amvd_error;
      return false;
    }
  } else if (node.use_amvd) {
    error = "use_amvd set outside the §5.20.7.6 gate";
    return false;
  }
  // §5.20.7.6: use_bawp follows single_mode for a single-prediction block
  // that is at least 8x8, not GLOBALMV, and not a switch frame.
  const bool bawp_available =
      plan_.allow_bawp && cur_w4_ * 4 >= 8 && cur_h4_ * 4 >= 8 &&
      node.inter_mode != kInterGlobalMv;
  if (bawp_available) {
    set_rate_tag(rate_hybrid::report::kTagBawp);
    if (!code_bawp(node)) {
      error = error_;
      return false;
    }
    set_rate_tag(-1);
  } else if (node.use_bawp || node.explicit_bawp ||
             node.use_bawp_chroma) {
    error = "BAWP symbols set outside the §5.20.7.6 gate";
    return false;
  }
  // §5.20.7.14 read_motion_mode(isCompound = 0).  It follows single_mode /
  // use_bawp and precedes read_drl_idx().  INTERINTRA is allowed only for a
  // non-compound block of 8x8..64x64 that does not use BAWP.
  if (!node.compound) {
    const int bw = cur_w4_ * 4;
    const int bh = cur_h4_ * 4;
    const bool interintra_allowed =
        plan_.enable_interintra && !node.use_bawp && bw >= 8 && bh >= 8 &&
        bw <= 64 && bh <= 64;
    if (!interintra_allowed && node.motion_mode != 0) {
      error = "INTERINTRA motion mode outside the §5.20.7.14 gate";
      return false;
    }
    if (plan_.enable_interintra) {
      adv::MotionModeSymbols symbols;
      symbols.is_warp_newmv = false;
      symbols.extend_allowed = false;
      symbols.local_allowed = false;
      symbols.inter_intra_allowed = interintra_allowed;
      symbols.inter_intra = node.motion_mode == 1;
      symbols.interintra_mode = node.interintra_mode;
      symbols.wedge_allowed = adv::kWedgeBits[node.bsize] != 0;
      symbols.wedge_interintra = node.wedge_interintra != 0;
      symbols.wedge_index = node.wedge_index;
      std::string mm_error;
      if (!adv::write_motion_mode(enc_, adv_cdfs_, node.bsize, symbols,
                                  &mm_error)) {
        error = "motion mode: " + mm_error;
        return false;
      }
    }
  }
  if (node.inter_mode == kInterNearMv || node.inter_mode == kInterNewMv) {
    // §5.20.7.6: read_drl_idx() is present for both NEARMV and NEWMV.
    // M2 NEWMV always uses DRL index 0 (the predictor resolve_mode used).
    const int drl = node.inter_mode == kInterNewMv ? 0 : node.drl_index;
    set_rate_tag(rate_hybrid::report::kTagDrl);
    if (!code_drl_index(ctx, drl)) {
      error = "write drl_idx failed";
      return false;
    }
    set_rate_tag(-1);
  }
  if (node.inter_mode == kInterNewMv) {
    set_rate_tag(rate_hybrid::report::kTagMvd);
    inter::mvprec::MvResConfig res_config;
    res_config.enable_adaptive_mvd = plan_.enable_adaptive_mvd;
    res_config.enable_mvd_sign_derive = plan_.enable_mvd_sign_derive;
    res_config.enable_flex_mvres = plan_.enable_flex_mvres;
    const me2::FramePrecisionSyntax frame_prec = frame_mv_precision();
    const int target = node.block_mv_precision >= 0
                           ? node.block_mv_precision
                           : frame_prec.frame_precision;
    inter::mvprec::BlockPrecisionSyntax precision_plan;
    std::string precision_error;
    if (!inter::mvprec::plan_block_precision(
            res_config, frame_prec, inter::mvprec::kYModeNewMv,
            node.use_amvd != 0, target, &precision_plan,
            &precision_error)) {
      error = "plan_block_precision failed: " + precision_error;
      return false;
    }
    if (node.use_amvd && !precision_plan.is_adaptive_mvd) {
      error = "use_amvd did not produce an IsAdaptiveMvd plan";
      return false;
    }
    const inter::mvprec::NeighborSet neigh = mv_res_neighbours();
    const int most_probable_ctx =
        inter::mvprec::use_most_probable_precision_ctx(neigh);
    const int pb_ctx =
        inter::mvprec::pb_mv_precision_ctx(neigh, frame_prec.frame_precision);
    if (!inter::mvprec::write_block_precision(
            enc_, cdfs_.mv_precision, most_probable_ctx, pb_ctx,
            precision_plan, &precision_error)) {        error = "write_block_precision failed: " + precision_error;
      return false;
    }
    inter::mvprec::MvdSignConfig sign_config;
    sign_config.enable_mvd_sign_derive = plan_.enable_mvd_sign_derive;
    sign_config.frame_precision = frame_prec.frame_precision;
    sign_config.mv_precision = precision_plan.mv_precision;
    sign_config.y_mode = inter::mvprec::kYModeNewMv;
    sign_config.is_adaptive_mvd = precision_plan.is_adaptive_mvd;
    sign_config.is_compound = false;
    sign_config.ref_mv_idx = 0;
    inter::mc::Mv mvd[1] = {inter::mc::Mv{node.mvd_y, node.mvd_x}};
    if (precision_plan.is_adaptive_mvd) {
      inter::mvprec::AmvdMvSyntax amvd;
      if (!inter::mvprec::amvd_mv_from_mv(mvd[0], &amvd,
                                          &precision_error)) {
        error = "amvd_mv_from_mv failed: " + precision_error;
        return false;
      }
      if (!inter::mvprec::write_amvd_mv(enc_, cdfs_.mv_precision, amvd,
                                        &precision_error)) {
        error = "write_amvd_mv failed: " + precision_error;
        return false;
      }
      if (!inter::mvprec::write_mvd_signs(enc_, sign_config, mvd, 1,
                                          &precision_error)) {
        error = "write_mvd_signs (AMVD) failed: " + precision_error;
        return false;
      }
    } else if (precision_plan.mv_precision ==
               me2::kMvPrecisionEighthPel) {
      // The historical 1/8-pel writer codes the sign bits itself.
      if (!write_mv(node.mvd_y, node.mvd_x)) {
        error = error_;
        return false;
      }
    } else {
      if (!inter::mvprec::write_shell_mv(
              enc_, cdfs_.mv_precision, node.mvd_y, node.mvd_x,
              precision_plan.mv_precision, &precision_error)) {
        error = "write_shell_mv failed: " + precision_error;
        return false;
      }
      if (!inter::mvprec::write_mvd_signs(enc_, sign_config, mvd, 1,
                                          &precision_error)) {
        error = "write_mvd_signs failed: " + precision_error;
        return false;
      }
    }
    set_rate_tag(-1);
  }
  // §5.18.5.1 / §5.20.7.6 switchable per-block interpolation filter.  The
  // frame header signed is_filter_switchable = 1; every leaf that passes
  // needs_interp_filter() codes one of the three TileInterpFilterCdf symbols.
  // On an inter leaf `mrl_index` carries the filter the decision chose (the
  // field is intra-only in every other consumer).  Gate off: no symbol and
  // the historical bitstream is untouched.
  if (interp_block_enabled()) {
    const bool large = cur_w4_ * 4 >= 8 && cur_h4_ * 4 >= 8;
    const int y_mode = node.inter_mode == kInterGlobalMv
                           ? inter::mvprec::kYModeGlobalMv
                           : (node.inter_mode == kInterNearMv
                                  ? inter::mvprec::kYModeNearMv
                                  : inter::mvprec::kYModeNewMv);
    if (inter::mvprec::needs_interp_filter(large, y_mode,
                                           node.motion_mode)) {
      const int filter_ctx = inter::mvprec::interp_filter_ctx(
          mv_res_neighbours(), node.ref_index, /*is_compound=*/false);
      std::string filter_error;
      if (!inter::mvprec::write_block_interp_filter(
              enc_, cdfs_.mv_precision, filter_ctx, node.mrl_index,
              &filter_error)) {
        error = "write_block_interp_filter failed: " + filter_error;
        return false;
      }
    } else if (node.mrl_index != 0) {
      error = "interp filter set outside the §5.20.7.6 gate";
      return false;
    }
  }
  return true;
}

// §7.12.2.2 GlobalMvs for this block at the frame precision.  The decoder
// stores the projected value in MiMv for GLOBALMV blocks and seeds the merge
// stack with it, so both the stack validation and the writer's own MI grid
// must use it instead of the (zero) coded mv.
inter::mc::Mv Writer::block_global_mv_for_ref(const BlockSyntax& node,
                                      int ref) const {
#if defined(AV2ENC_VIDEO_GME)
  if (ref >= 0 && plan_.gm_num_refs > ref) {
    inter::gme::MvPrecision precision = inter::gme::MvPrecision::kEighthPel;
    if (plan_.mv_precision == 1) {
      precision = inter::gme::MvPrecision::kQuarterPel;
    } else if (plan_.mv_precision == 2) {
      precision = inter::gme::MvPrecision::kHalfPel;
    }
    int projected[2] = {0, 0};
    inter::gme::get_warp_motion_vector(
        plan_.gm_params[static_cast<std::size_t>(ref)], node.mi_col,
        node.mi_row, cur_w4_, cur_h4_, mi_cols_, mi_rows_, precision,
        projected);
    return inter::mc::Mv{projected[0], projected[1]};
  }
#else
  (void)node;
  (void)ref;
#endif
  return inter::mc::Mv{0, 0};
}

inter::mc::Mv Writer::block_global_mv(const BlockSyntax& node) const {
  return block_global_mv_for_ref(node, node.ref_index);
}

// §5.18.2 FrameMvPrecision -> the two header bits.  With
// seq_force_integer_mv the sequence pins the frame to one pel, so the header
// codes no precision pair (and the tile MVDs use one-pel units).
me2::FramePrecisionSyntax Writer::frame_mv_precision() const {
  int target = me2::kMvPrecisionEighthPel;
  if (plan_.frame_force_integer_mv) {
    target = me2::kMvPrecisionOnePel;
  } else if (plan_.mv_precision == 1) {
    target = me2::kMvPrecisionQuarterPel;
  } else if (plan_.mv_precision == 2) {
    target = me2::kMvPrecisionHalfPel;
  }
  me2::FramePrecisionSyntax out;
  me2::me2_frame_precision_to_bits(
      target, plan_.enable_flex_mvres ? 1 : 0,
      plan_.frame_force_integer_mv ? 1 : 0, &out);
  return out;
}

int Writer::block_mv_precision(const BlockSyntax& node) const {
  if (node.block_mv_precision >= 0) {
    return node.block_mv_precision;
  }
  return frame_mv_precision().frame_precision;
}

// §8.3 MV-resolution neighbour set for the current block.
inter::mvprec::NeighborSet Writer::mv_res_neighbours() const {
  return inter::mvprec::collect_neighbors(
      prec_grid_.data(), mi_cols_, mi_rows_, cur_r_, cur_c_, cur_w4_,
      cur_h4_, 16);
}

void Writer::update_mode_grids(const BlockSyntax& node) {
  for (int yy = 0; yy < cur_h4_; ++yy) {
    const int rr = cur_r_ + yy;
    if (rr >= mi_rows_) break;
    for (int xx = 0; xx < cur_w4_; ++xx) {
      const int cc = cur_c_ + xx;
      if (cc >= mi_cols_) break;
      const std::size_t idx = cell_index(rr, cc);
      const bool tip = node.is_inter && node.tip_mode != 0;
      is_inter_grid_[idx] = node.is_inter ? 1 : 0;
      skip_grid_[idx] = node.skip_flag ? 1 : 0;
      comp_group_grid_[idx] =
          (node.compound && node.compound_masked) ? 1 : 0;
      ymode_grid_[idx] =
          node.is_inter
              ? static_cast<std::uint8_t>(
                    tip ? (node.tip_pred_mode ? 16 : 14)
                        : (node.compound
                               ? (node.compound_joint
                                      ? 24
                                      : 19 + node.compound_mode)
                               : (node.inter_mode == kInterGlobalMv
                                      ? 15
                                      : (node.inter_mode == kInterNearMv ? 14 : 16))))
              : node.y_mode;
      inter::mc::MiInfo& mi = mv_grid_[idx];
      mi = inter::mc::MiInfo{};
      // A TIP leaf stores RefFrame[0] == TIP_FRAME so later §7.12.2 scans
      // apply the §7.12.2.15/.17 arms (the builder reports
      // kUnsupportedFeature when one is needed).
      mi.ref[0] = node.is_inter
                      ? (tip ? merge::kTipFrame : static_cast<std::int8_t>(node.ref_index))
                      : kRefIntraAv2;
      mi.ref[1] =
          tip ? merge::kRefNone
              : (node.compound
                     ? static_cast<std::int8_t>(node.ref_index1)
                     : (node.is_inter && node.motion_mode == 1 ? kRefIntraAv2 : merge::kRefNone));
      // A GLOBALMV block stores GlobalMvs[0]; every other mode stores its
      // coded MV.  A compound block stores the per-list MVs.  This mirrors
      // decode_block() and keeps later merge-stack candidates identical to
      // the decoder's.
      inter::mc::Mv mv0 = inter::mc::Mv{node.mv_y, node.mv_x};
      inter::mc::Mv mv1 = inter::mc::Mv{node.mv1_y, node.mv1_x};
      if (!tip && node.is_inter && node.inter_mode == kInterGlobalMv) {
        mv0 = block_global_mv_for_ref(node, node.ref_index);
        mv1 = node.compound ? block_global_mv_for_ref(node, node.ref_index1)
                            : mv0;
      }
      mi.mv[0][0] = mv0.y;
      mi.mv[0][1] = mv0.x;
      mi.mv[1][0] = mv1.y;
      mi.mv[1][1] = mv1.x;
      mi.is_inter = node.is_inter ? 1 : 0;
      mi.w4 = static_cast<std::int8_t>(cur_w4_);
      mi.h4 = static_cast<std::int8_t>(cur_h4_);
      mi.col_base = static_cast<std::int16_t>(cur_c_);
      mi.row_base = static_cast<std::int16_t>(cur_r_);
      // §8.3 MV-resolution neighbour state.  Every decoded MI cell is
      // available; intra cells carry UseMostProbablePrecisions = 0 and the
      // frame precision, so they only affect the precision context through
      // the MvPrecision comparison (which uses the frame value).
      inter::mvprec::NeighborMvInfo& prec = prec_grid_[idx];
      prec = inter::mvprec::NeighborMvInfo{};
      prec.available = true;
      prec.ref[0] = mi.ref[0];
      prec.ref[1] = mi.ref[1];
      prec.y_mode = ymode_grid_[idx];
      const bool newmv_leaf = node.is_inter && !node.compound &&
                              (tip ? node.tip_pred_mode == 1 : node.inter_mode == kInterNewMv);
      prec.uses_amvd = newmv_leaf && node.use_amvd != 0;
      prec.mv_precision =
          node.is_inter ? block_mv_precision(node)
                        : frame_mv_precision().frame_precision;
      // §5.20.4.1 block state.  Intra leaves set
      // use_most_probable_precision = 0 and keep FrameMvPrecision; an inter
      // block that codes no precision symbol (NEARMV/GLOBALMV, or a NEWMV
      // block that kept the frame value) is 1.  Only a coarser
      // pb_mv_precision clears the flag.
      if (newmv_leaf) {
        prec.use_most_probable_precision =
            node.block_mv_precision < 0
                ? 1
                : (node.block_mv_precision ==
                           frame_mv_precision().frame_precision
                       ? 1
                       : 0);
      } else {
        prec.use_most_probable_precision = node.is_inter != 0;
      }
      // §8.3 interp_filter context state.  With the switchable frame filter an
      // inter leaf stores its coded block filter (0..2); intra cells and the
      // frame-global form keep the "unavailable" sentinel.  Ineligible inter
      // leaves (warp / large GLOBALMV / TIP) predict with EIGHTTAP and the
      // decision stores 0 in `mrl_index`, which is what the decoder leaves in
      // its MiMv entry for a block that codes no filter.
      prec.interp_filter =
          node.is_inter
              ? (interp_block_enabled() ? static_cast<int>(node.mrl_index) : 3)
              : 3;
    }
  }
}

}  // namespace detail
}  // namespace av2enc::video::syntax
