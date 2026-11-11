// AV2 v1.0.0 temporal inter tools — §7.9/§7.10/§7.22 implementation.
//
// This is a direct transcription of the spec's integer processes.  Every loop
// bound, rounding step and availability check is kept in spec order; the only
// concessions to C++ are index helpers and explicit casts.

#include "encoder/inter/temporal/motion_field.h"

#include <algorithm>
#include <array>
#include <cstddef>

#include "encoder/inter/temporal/temporal_math.h"

namespace av2enc::inter::temporal {
namespace {

constexpr int kMaxProjections = 16;

struct MfeState {
  const MotionFieldConfig* cfg = nullptr;
  MotionField* field = nullptr;
  TrajectoryField* traj = nullptr;
  MotionFieldResult* result = nullptr;
  int w8 = 0;
  int h8 = 0;
  int num_total_refs = 0;
  std::array<int, kRefsPerFrame> order_hint{};
  std::array<int, kRefsPerFrame> sort_ref{};
  int cur_idx = -1;
  std::array<int, kRefsPerFrame> stack{};
  std::array<int, kRefsPerFrame> depth{};
  std::array<std::uint8_t, kRefsPerFrame> visited{};
  std::array<std::array<std::uint8_t, 2>, kRefsPerFrame> checked{};
  int stack_count = 0;
  std::array<int, kMaxProjections> proc_ref{};
  std::array<std::uint8_t, kMaxProjections> proc_bwd{};
  std::array<int, kMaxProjections> proc_target{};
  int proc_count = 0;

  const RefFrameInfo& info(int rf) const { return cfg->refs[rf]; }

  bool eligible(int rf) const {
    if (rf < 0 || rf >= num_total_refs) {
      return false;
    }
    const RefFrameInfo& r = info(rf);
    return r.type != FrameType::kIntraOnly && r.type != FrameType::kKey &&
           r.width == cfg->frame_width && r.height == cfg->frame_height;
  }

  bool is_ref_overlay(int rf) const {
    if (rf < 0 || rf >= num_total_refs) {
      return false;
    }
    const RefFrameInfo& r = info(rf);
    for (int i = 0; i < r.num_saved_refs; ++i) {
      if (r.saved_order_hint[i] == r.order_hint) {
        return true;
      }
    }
    return false;
  }

  bool has_future_ref(int rf) const {
    if (order_hint[static_cast<std::size_t>(rf)] == kRestrictedOrderHint) {
      return false;
    }
    const RefFrameInfo& r = info(rf);
    for (int i = 0; i < r.num_saved_refs; ++i) {
      if (r.saved_order_hint[i] != kRestrictedOrderHint &&
          r.saved_order_hint[i] > r.order_hint) {
        return true;
      }
    }
    return false;
  }

  bool has_past_ref(int rf) const {
    if (order_hint[static_cast<std::size_t>(rf)] == kRestrictedOrderHint) {
      return false;
    }
    const RefFrameInfo& r = info(rf);
    for (int i = 0; i < r.num_saved_refs; ++i) {
      if (r.saved_order_hint[i] != kRestrictedOrderHint &&
          r.saved_order_hint[i] < r.order_hint) {
        return true;
      }
    }
    return false;
  }

  int dist_to_closest_interp_ref(int start_frame, bool find_forward) const {
    int closest = 0x7fffffff;
    if (!eligible(start_frame)) {
      return closest;
    }
    const RefFrameInfo& r = info(start_frame);
    for (int i = 0; i < r.num_saved_refs; ++i) {
      if (r.saved_order_hint[i] == kRestrictedOrderHint) {
        continue;
      }
      const int start_to_ref =
          get_relative_dist(order_hint[static_cast<std::size_t>(start_frame)],
                            r.saved_order_hint[i]);
      const int cur_to_ref =
          get_relative_dist(cfg->order_hint, r.saved_order_hint[i]);
      const bool two_sides =
          find_forward ? (start_to_ref > 0 && cur_to_ref > 0)
                       : (start_to_ref < 0 && cur_to_ref < 0);
      if (two_sides) {
        closest = std::min(closest, abs_int(start_to_ref));
      }
    }
    return closest;
  }

  void topo_sort_refs(int rf) {
    if (visited[static_cast<std::size_t>(rf)] != 0) {
      return;
    }
    visited[static_cast<std::size_t>(rf)] = 1;
    const RefFrameInfo& r = info(rf);
    if (r.type == FrameType::kInter) {
      for (int i = 0; i < r.num_saved_refs; ++i) {
        const int saved = r.saved_order_hint[i];
        if (saved == kRestrictedOrderHint) {
          continue;
        }
        for (int j = 0; j < num_total_refs; ++j) {
          if (order_hint[static_cast<std::size_t>(j)] == saved &&
              !is_ref_overlay(j)) {
            topo_sort_refs(j);
            break;
          }
        }
      }
    }
    depth[static_cast<std::size_t>(rf)] = stack_count;
    stack[static_cast<std::size_t>(stack_count)] = rf;
    ++stack_count;
  }

  int record_projection_with_type(bool doing_tip, int ref, int is_bwd,
                                  int target_frame, int max_check) {
    const int ref_idx = info(ref).slot;
    if (ref_idx < 0 || !eligible(ref)) {
      return proc_count;
    }
    const int ref_to_cur =
        get_relative_dist(order_hint[static_cast<std::size_t>(ref)],
                          cfg->order_hint);
    if (abs_int(ref_to_cur) > kMaxFrameDistance) {
      return proc_count;
    }
    if (cfg->use_bru &&
        (cfg->bru_ref == ref ||
         (doing_tip && cfg->bru_ref == target_frame))) {
      return proc_count;
    }
    if (doing_tip) {
      is_bwd = order_hint[static_cast<std::size_t>(ref)] <
                       order_hint[static_cast<std::size_t>(target_frame)]
                   ? 1
                   : 0;
    }
    if (proc_count >= max_check ||
        checked[static_cast<std::size_t>(ref)]
               [static_cast<std::size_t>(is_bwd)] != 0) {
      return proc_count;
    }
    checked[static_cast<std::size_t>(ref)]
           [static_cast<std::size_t>(is_bwd)] = 1;
    proc_ref[static_cast<std::size_t>(proc_count)] = ref;
    proc_bwd[static_cast<std::size_t>(proc_count)] =
        static_cast<std::uint8_t>(is_bwd);
    proc_target[static_cast<std::size_t>(proc_count)] = target_frame;
    ++proc_count;
    return proc_count;
  }

  void check_traj_intersect(int src_frame, int x8, int y8, int end_frame,
                            Mv mv);

  void projection(int src, int dst_sign, bool is_bwd, int target_frame);
};

void MfeState::check_traj_intersect(int src_frame, int x8, int y8,
                                    int end_frame, Mv mv) {
  if (traj == nullptr || end_frame == kRefNone) {
    return;
  }
  const int num_refs = traj->num_refs;
  for (int k = 0; k < 3; ++k) {
    const std::size_t pv_idx =
        (((static_cast<std::size_t>(k) * static_cast<std::size_t>(num_refs) +
           static_cast<std::size_t>(src_frame)) *
              static_cast<std::size_t>(h8) +
          static_cast<std::size_t>(y8)) *
             static_cast<std::size_t>(w8) +
         static_cast<std::size_t>(x8));
    if (traj->pos_valid[pv_idx] == 0) {
      continue;
    }
    const int traj_y8 = traj->pos[pv_idx * 2 + 0];
    const int traj_x8 = traj->pos[pv_idx * 2 + 1];
    const std::size_t end_idx =
        (static_cast<std::size_t>(end_frame) * static_cast<std::size_t>(h8) +
         static_cast<std::size_t>(traj_y8)) *
            static_cast<std::size_t>(w8) +
        static_cast<std::size_t>(traj_x8);
    if (traj->valid[end_idx] != 0) {
      continue;
    }
    for (int c = 0; c < 2; ++c) {
      const std::int32_t v =
          traj->mv[traj_mv_index(w8, h8, num_refs, src_frame, traj_y8, traj_x8,
                                 c)] +
          (c == 0 ? mv.row : mv.col);
      traj->mv[traj_mv_index(w8, h8, num_refs, end_frame, traj_y8, traj_x8,
                             c)] =
          static_cast<std::int32_t>(
              clip3(-kRefMvsLimit, kRefMvsLimit, static_cast<int>(v)));
    }
    traj->valid[end_idx] = 1;
    const SampledPosition pos = get_sampled_position(
        traj_x8, traj_y8, 1,
        Mv{traj->mv[traj_mv_index(w8, h8, num_refs, end_frame, traj_y8,
                                 traj_x8, 0)],
           traj->mv[traj_mv_index(w8, h8, num_refs, end_frame, traj_y8,
                                 traj_x8, 1)]},
        cfg->mi_cols, cfg->mi_rows, cfg->proj_step);
    if (check_block_position(pos.valid, traj_x8, traj_y8, pos.x8, pos.y8,
                             cfg->sb_64x64, cfg->proj_step)) {
      const std::size_t tp_idx = traj_pos_index(
          k, num_refs, w8, h8, end_frame, pos.y8, pos.x8, 0);
      traj->pos[tp_idx] = static_cast<std::int16_t>(traj_y8);
      traj->pos[tp_idx + 1] = static_cast<std::int16_t>(traj_x8);
      traj->pos_valid[tp_idx / 2] = 1;
    }
  }

  const SampledPosition end = get_sampled_position(
      x8, y8, 1, mv, cfg->mi_cols, cfg->mi_rows, cfg->proj_step);
  if (!end.valid) {
    return;
  }
  for (int k = 0; k < 3; ++k) {
    const std::size_t tp_idx =
        (((static_cast<std::size_t>(k) * static_cast<std::size_t>(num_refs) +
           static_cast<std::size_t>(end_frame)) *
              static_cast<std::size_t>(h8) +
          static_cast<std::size_t>(end.y8)) *
             static_cast<std::size_t>(w8) +
         static_cast<std::size_t>(end.x8)) *
        2;
    if (traj->pos_valid[tp_idx / 2] == 0) {
      continue;
    }
    const int traj_y8 = traj->pos[tp_idx + 0];
    const int traj_x8 = traj->pos[tp_idx + 1];
    if (!check_block_position(true, traj_x8, traj_y8, x8, y8, cfg->sb_64x64,
                              cfg->proj_step)) {
      continue;
    }
    const std::size_t src_idx =
        (static_cast<std::size_t>(src_frame) * static_cast<std::size_t>(h8) +
         static_cast<std::size_t>(traj_y8)) *
            static_cast<std::size_t>(w8) +
        static_cast<std::size_t>(traj_x8);
    if (traj->valid[src_idx] != 0) {
      continue;
    }
    for (int c = 0; c < 2; ++c) {
      const std::int32_t v =
          traj->mv[traj_mv_index(w8, h8, num_refs, end_frame, traj_y8,
                                 traj_x8, c)] -
          (c == 0 ? mv.row : mv.col);
      traj->mv[traj_mv_index(w8, h8, num_refs, src_frame, traj_y8, traj_x8,
                             c)] =
          static_cast<std::int32_t>(
              clip3(-kRefMvsLimit, kRefMvsLimit, static_cast<int>(v)));
    }
    traj->valid[src_idx] = 1;
    const SampledPosition pos = get_sampled_position(
        traj_x8, traj_y8, 1,
        Mv{traj->mv[traj_mv_index(w8, h8, num_refs, src_frame, traj_y8,
                                 traj_x8, 0)],
           traj->mv[traj_mv_index(w8, h8, num_refs, src_frame, traj_y8,
                                 traj_x8, 1)]},
        cfg->mi_cols, cfg->mi_rows, cfg->proj_step);
    if (check_block_position(pos.valid, traj_x8, traj_y8, pos.x8, pos.y8,
                             cfg->sb_64x64, cfg->proj_step)) {
      const std::size_t dst_idx = traj_pos_index(
          k, num_refs, w8, h8, src_frame, pos.y8, pos.x8, 0);
      traj->pos[dst_idx] = static_cast<std::int16_t>(traj_y8);
      traj->pos[dst_idx + 1] = static_cast<std::int16_t>(traj_x8);
      traj->pos_valid[dst_idx / 2] = 1;
    }
  }
}

void MfeState::projection(int src, int dst_sign, bool is_bwd,
                          int target_frame) {
  const RefFrameInfo& src_info = info(src);
  const SavedMvField& saved = cfg->saved[src];
  const int num_saved = src_info.num_saved_refs;
  if (saved.mv == nullptr || saved.ref == nullptr || saved.w8 != w8 ||
      saved.h8 != h8) {
    return;
  }
  const int ref_to_cur =
      get_relative_dist(order_hint[static_cast<std::size_t>(src)],
                        cfg->order_hint);

  std::array<int, kRefsPerFrame> start_ref_map{};
  for (int k = 0; k < kRefsPerFrame; ++k) {
    start_ref_map[static_cast<std::size_t>(k)] = kRefNone;
  }
  for (int k = 0; k < num_saved; ++k) {
    const int saved_oh = src_info.saved_order_hint[k];
    for (int rf = 0; rf < num_total_refs; ++rf) {
      if (saved_oh == order_hint[static_cast<std::size_t>(rf)] &&
          order_hint[static_cast<std::size_t>(rf)] !=
              order_hint[static_cast<std::size_t>(src)] &&
          !(saved_oh == kRestrictedOrderHint ||
            order_hint[static_cast<std::size_t>(rf)] == kRestrictedOrderHint)) {
        start_ref_map[static_cast<std::size_t>(k)] = rf;
        break;
      }
    }
  }

  const int list = is_bwd ? 1 : 0;
  for (int y8 = 0; y8 < h8; y8 += cfg->proj_step) {
    for (int x8 = 0; x8 < w8; x8 += cfg->proj_step) {
      const std::size_t pos = mf_index(w8, y8, x8);
      const std::int8_t src_ref = saved.ref[pos * 2 + static_cast<std::size_t>(list)];
      if (!is_inter_ref(src_ref) || src_ref >= kRefsPerFrame) {
        continue;
      }
      Mv mv2{uncompression_mv(saved.mv[(pos * 2 + static_cast<std::size_t>(list)) * 2 + 0]),
             uncompression_mv(saved.mv[(pos * 2 + static_cast<std::size_t>(list)) * 2 + 1])};
      const int mapped_ref = start_ref_map[static_cast<std::size_t>(src_ref)];
      if (cfg->enable_mv_traj) {
        check_traj_intersect(src, x8, y8, mapped_ref, mv2);
      }
      int ref_offset = get_relative_dist(
          order_hint[static_cast<std::size_t>(src)],
          src_info.saved_order_hint[src_ref]);
      if (src_info.saved_order_hint[src_ref] == kRestrictedOrderHint) {
        ref_offset = 0;
      }
      bool pos_valid = abs_int(ref_offset) <= kMaxFrameDistance;
      if (is_bwd) {
        ref_offset = -ref_offset;
      }
      if (!pos_valid || ref_offset < 0) {
        continue;
      }
      Mv mv = mv2;
      const Mv proj_mv =
          get_mv_projection(mv, ref_to_cur * dst_sign, ref_offset);
      const SampledPosition sampled =
          is_bwd ? get_sampled_position(x8, y8, 1, proj_mv, cfg->mi_cols,
                                        cfg->mi_rows, cfg->proj_step)
                 : get_sampled_position(x8, y8, dst_sign, proj_mv,
                                        cfg->mi_cols, cfg->mi_rows,
                                        cfg->proj_step);
      pos_valid = sampled.valid;
      pos_valid = check_block_position(pos_valid, x8, y8, sampled.x8,
                                       sampled.y8, cfg->sb_64x64,
                                       cfg->proj_step);
      if (!pos_valid) {
        continue;
      }
      const std::size_t dst = mf_index(w8, sampled.y8, sampled.x8);
      const bool replace =
          field->valid[dst] == 0 ||
          (target_frame != kRefNone &&
           target_frame == start_ref_map[static_cast<std::size_t>(src_ref)] &&
           field->offset[dst] != ref_offset);
      if (!replace) {
        continue;
      }
      if (cfg->enable_mv_traj && traj != nullptr) {
        const int phase =
            tmvp_phase(sampled.x8, cfg->sb_64x64, cfg->proj_step);
        const std::size_t tp_idx = traj_pos_index(
            phase, traj->num_refs, w8, h8, src, y8, x8, 0);
        traj->pos[tp_idx] = static_cast<std::int16_t>(sampled.y8);
        traj->pos[tp_idx + 1] = static_cast<std::int16_t>(sampled.x8);
        traj->pos_valid[tp_idx / 2] = 1;
        for (int c = 0; c < 2; ++c) {
          const std::int32_t v = c == 0 ? proj_mv.row : proj_mv.col;
          traj->mv[traj_mv_index(w8, h8, traj->num_refs, src, sampled.y8,
                                 sampled.x8, c)] =
              static_cast<std::int32_t>(clip3(-kRefMvsLimit, kRefMvsLimit,
                                              -static_cast<int>(v)));
        }
        traj->valid[(static_cast<std::size_t>(src) *
                         static_cast<std::size_t>(h8) +
                     static_cast<std::size_t>(sampled.y8)) *
                        static_cast<std::size_t>(w8) +
                    static_cast<std::size_t>(sampled.x8)] = 1;
        const int end_frame = start_ref_map[static_cast<std::size_t>(src_ref)];
        if (end_frame != kRefNone) {
          const Mv end_mv = get_mv_projection(
              mv, ref_offset - ref_to_cur * dst_sign, ref_offset);
          for (int c = 0; c < 2; ++c) {
            const std::int32_t v = c == 0 ? end_mv.row : end_mv.col;
            traj->mv[traj_mv_index(w8, h8, traj->num_refs, end_frame,
                                   sampled.y8, sampled.x8, c)] =
                static_cast<std::int32_t>(
                    clip3(-kRefMvsLimit, kRefMvsLimit, static_cast<int>(v)));
          }
          traj->valid[(static_cast<std::size_t>(end_frame) *
                           static_cast<std::size_t>(h8) +
                       static_cast<std::size_t>(sampled.y8)) *
                          static_cast<std::size_t>(w8) +
                      static_cast<std::size_t>(sampled.x8)] = 1;
          const SampledPosition target = get_sampled_position(
              x8, y8, 1, mv, cfg->mi_cols, cfg->mi_rows, cfg->proj_step);
          if (check_block_position(target.valid, sampled.x8, sampled.y8,
                                   target.x8, target.y8, cfg->sb_64x64,
                                   cfg->proj_step)) {
            const std::size_t tp_idx2 = traj_pos_index(
                phase, traj->num_refs, w8, h8, end_frame, target.y8,
                target.x8, 0);
            traj->pos[tp_idx2] = static_cast<std::int16_t>(sampled.y8);
            traj->pos[tp_idx2 + 1] = static_cast<std::int16_t>(sampled.x8);
            traj->pos_valid[tp_idx2 / 2] = 1;
          }
        }
      }
      if (is_bwd) {
        mv.row = -mv.row;
        mv.col = -mv.col;
      }
      field->valid[dst] = 1;
      field->mv[dst * 2 + 0] = mv.row;
      field->mv[dst * 2 + 1] = mv.col;
      field->offset[dst] = ref_offset;
    }
  }
}

}  // namespace

TemporalStatus motion_field_estimation(const MotionFieldConfig& config,
                                       MotionField* field,
                                       TrajectoryField* traj,
                                       MotionFieldResult* result) {
  if (field == nullptr || config.mi_cols <= 0 || config.mi_rows <= 0 ||
      config.num_total_refs < 0 || config.num_total_refs > kRefsPerFrame ||
      config.proj_step < 1 || config.proj_step > 2) {
    return TemporalStatus::kInvalidArgument;
  }
  const int w8 = config.mi_cols >> 1;
  const int h8 = config.mi_rows >> 1;
  if (w8 <= 0 || h8 <= 0 || field->w8 != w8 || field->h8 != h8 ||
      field->mv == nullptr || field->valid == nullptr ||
      field->offset == nullptr) {
    return TemporalStatus::kInvalidArgument;
  }
  if (config.enable_mv_traj) {
    if (traj == nullptr || traj->w8 != w8 || traj->h8 != h8 ||
        traj->num_refs < config.num_total_refs || traj->mv == nullptr ||
        traj->valid == nullptr || traj->pos == nullptr ||
        traj->pos_valid == nullptr) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  for (int rf = 0; rf < config.num_total_refs; ++rf) {
    const RefFrameInfo& r = config.refs[rf];
    if (r.slot < 0 || r.slot >= kMaxRefFrames) {
      return TemporalStatus::kInvalidArgument;
    }
  }

  for (int i = 0; i < w8 * h8; ++i) {
    field->valid[i] = 0;
    field->offset[i] = 0;
    field->mv[i * 2 + 0] = 0;
    field->mv[i * 2 + 1] = 0;
  }
  if (traj != nullptr) {
    const std::size_t count = static_cast<std::size_t>(traj->num_refs) *
                              static_cast<std::size_t>(w8) *
                              static_cast<std::size_t>(h8);
    for (std::size_t i = 0; i < count; ++i) {
      traj->valid[i] = 0;
      traj->pos_valid[i] = 0;
    }
    for (std::size_t i = 0; i < count * 2; ++i) {
      traj->mv[i] = 0;
      traj->pos[i] = 0;
    }
  }

  MfeState st;
  st.cfg = &config;
  st.field = field;
  st.traj = traj;
  st.result = result;
  st.w8 = w8;
  st.h8 = h8;
  st.num_total_refs = config.num_total_refs;
  for (int i = 0; i < config.num_total_refs; ++i) {
    st.order_hint[static_cast<std::size_t>(i)] = config.ref_order_hint[i];
    st.sort_ref[static_cast<std::size_t>(i)] = i;
  }
  // sortRef: sort by order hint using get_relative_dist().
  for (int i = 0; i < config.num_total_refs; ++i) {
    for (int j = i + 1; j < config.num_total_refs; ++j) {
      if (get_relative_dist(
              st.order_hint[static_cast<std::size_t>(st.sort_ref[static_cast<std::size_t>(j)])],
              st.order_hint[static_cast<std::size_t>(st.sort_ref[static_cast<std::size_t>(i)])]) < 0) {
        std::swap(st.sort_ref[static_cast<std::size_t>(i)],
                  st.sort_ref[static_cast<std::size_t>(j)]);
      }
    }
  }
  st.cur_idx = -1;
  for (int i = 0; i < config.num_total_refs; ++i) {
    if (get_relative_dist(
            st.order_hint[static_cast<std::size_t>(st.sort_ref[static_cast<std::size_t>(i)])],
            config.order_hint) < 0) {
      st.cur_idx = i;
    } else {
      break;
    }
  }

  for (int rf = 0; rf < config.num_total_refs; ++rf) {
    st.visited[static_cast<std::size_t>(rf)] = 0;
    st.depth[static_cast<std::size_t>(rf)] = -1;
    st.checked[static_cast<std::size_t>(rf)][0] = 0;
    st.checked[static_cast<std::size_t>(rf)][1] = 0;
  }
  st.stack_count = 0;
  for (int rf = 0; rf < config.num_total_refs; ++rf) {
    if (st.order_hint[static_cast<std::size_t>(rf)] != kRestrictedOrderHint) {
      st.topo_sort_refs(rf);
    }
  }
  if (st.stack_count < 2) {
    if (result != nullptr) {
      result->closest_past = kRefNone;
      result->closest_future = kRefNone;
      result->projection_count = 0;
    }
    return TemporalStatus::kOk;
  }

  int closest_past = kRefNone;
  int closest_future = kRefNone;
  if (config.enable_tip &&
      ((config.num_future_refs > 0 && config.num_past_refs > 0) ||
       config.num_past_refs >= 2)) {
    const int past = st.sort_ref[static_cast<std::size_t>(st.cur_idx)];
    int future;
    if (config.num_future_refs > 0 && config.num_past_refs > 0) {
      future = st.sort_ref[static_cast<std::size_t>(st.cur_idx + 1)];
    } else {
      future = st.sort_ref[static_cast<std::size_t>(st.cur_idx - 1)];
    }
    if (st.depth[static_cast<std::size_t>(past)] >
        st.depth[static_cast<std::size_t>(future)]) {
      st.record_projection_with_type(true, past, 1, future, kTipMfmvStackSize);
    } else {
      st.record_projection_with_type(true, future, 0, past, kTipMfmvStackSize);
    }
    closest_past = past;
    closest_future = future;
  }

  for (int group_idx = 0; group_idx < 2; ++group_idx) {
    int past_idx = st.cur_idx >= group_idx ? st.cur_idx - group_idx : -1;
    if (past_idx >= 0 &&
        !st.has_future_ref(st.sort_ref[static_cast<std::size_t>(past_idx)])) {
      past_idx = -1;
    }
    const int past_dist =
        past_idx >= 0
            ? st.dist_to_closest_interp_ref(
                  st.sort_ref[static_cast<std::size_t>(past_idx)], false)
            : -1;
    int future_idx =
        st.cur_idx < config.num_total_refs - group_idx - 1
            ? st.cur_idx + 1 + group_idx
            : -1;
    if (future_idx >= 0 &&
        !st.has_past_ref(st.sort_ref[static_cast<std::size_t>(future_idx)])) {
      future_idx = -1;
    }
    const int future_dist =
        future_idx >= 0
            ? st.dist_to_closest_interp_ref(
                  st.sort_ref[static_cast<std::size_t>(future_idx)], true)
            : -1;
    if (future_dist < past_dist) {
      if (future_idx >= 0) {
        st.record_projection_with_type(
            false, st.sort_ref[static_cast<std::size_t>(future_idx)], 0,
            kRefNone, kTipMfmvStackSize);
      }
      if (past_idx >= 0) {
        st.record_projection_with_type(
            false, st.sort_ref[static_cast<std::size_t>(past_idx)], 1,
            kRefNone, kTipMfmvStackSize);
      }
    } else {
      if (past_idx >= 0) {
        st.record_projection_with_type(
            false, st.sort_ref[static_cast<std::size_t>(past_idx)], 1,
            kRefNone, kTipMfmvStackSize);
      }
      if (future_idx >= 0) {
        st.record_projection_with_type(
            false, st.sort_ref[static_cast<std::size_t>(future_idx)], 0,
            kRefNone, kTipMfmvStackSize);
      }
    }
  }
  if (st.cur_idx >= 0) {
    st.record_projection_with_type(
        false, st.sort_ref[static_cast<std::size_t>(st.cur_idx)], 0, kRefNone,
        kTipMfmvStackSize);
  }
  if (st.cur_idx >= 1) {
    st.record_projection_with_type(
        false, st.sort_ref[static_cast<std::size_t>(st.cur_idx - 1)], 0,
        kRefNone, kTipMfmvStackSize);
  }
  for (int ri = st.stack_count - 1; ri > 0; --ri) {
    const int ref = st.stack[static_cast<std::size_t>(ri)];
    bool is_bwd =
        st.order_hint[static_cast<std::size_t>(ref)] < config.order_hint;
    for (int j = 0; j < 2; ++j) {
      if (st.checked[static_cast<std::size_t>(ref)]
                    [static_cast<std::size_t>(is_bwd ? 1 : 0)] == 0) {
        st.record_projection_with_type(false, ref, is_bwd ? 1 : 0, kRefNone,
                                       kMfmvStackSize);
      }
      is_bwd = !is_bwd;
    }
  }

  int process_count = st.proc_count;
  if (config.reduced_ref_frame_mvs_mode) {
    process_count = std::min(1, process_count);
  }
  for (int i = 0; i < process_count; ++i) {
    const int ref = st.proc_ref[static_cast<std::size_t>(i)];
    const bool is_bwd = st.proc_bwd[static_cast<std::size_t>(i)] != 0;
    const int target = st.proc_target[static_cast<std::size_t>(i)];
    st.projection(ref, is_bwd ? -1 : 1, is_bwd, target);
  }

  if (config.enable_mv_traj) {
    if (config.proj_step == 2) {
      for (int rf = 0; rf < config.num_total_refs; ++rf) {
        for (int y8 = 0; y8 < h8; y8 += 2) {
          for (int x8 = 0; x8 < w8; x8 += 2) {
            // fill_traj_mv(rf, y8, x8, dy, dx)
            for (int pass = 0; pass < 3; ++pass) {
              const int dy = pass == 0 ? 0 : 1;
              const int dx = pass == 0 ? 1 : (pass == 1 ? 0 : 1);
              if (y8 + dy == h8 || x8 + dx == w8) {
                continue;
              }
              const std::size_t cur =
                  (static_cast<std::size_t>(rf) * static_cast<std::size_t>(h8) +
                   static_cast<std::size_t>(y8)) *
                      static_cast<std::size_t>(w8) +
                  static_cast<std::size_t>(x8);
              if (traj->valid[cur] == 0) {
                continue;
              }
              int count = 1;
              int avg[2] = {traj->mv[traj_mv_index(w8, h8, traj->num_refs, rf,
                                                    y8, x8, 0)],
                            traj->mv[traj_mv_index(w8, h8, traj->num_refs, rf,
                                                    y8, x8, 1)]};
              const bool r_avail =
                  dx > 0 &&
                  tmvp_avail(x8, x8 + 2, w8, config.sb_64x64, config.proj_step) &&
                  traj->valid[(static_cast<std::size_t>(rf) *
                                   static_cast<std::size_t>(h8) +
                               static_cast<std::size_t>(y8)) *
                                  static_cast<std::size_t>(w8) +
                              static_cast<std::size_t>(x8 + 2)] != 0;
              const bool b_avail =
                  dy > 0 &&
                  tmvp_avail(y8, y8 + 2, h8, config.sb_64x64, config.proj_step) &&
                  traj->valid[(static_cast<std::size_t>(rf) *
                                   static_cast<std::size_t>(h8) +
                               static_cast<std::size_t>(y8 + 2)) *
                                  static_cast<std::size_t>(w8) +
                              static_cast<std::size_t>(x8)] != 0;
              const bool br_avail =
                  dy > 0 && dx > 0 &&
                  tmvp_avail(x8, x8 + 2, w8, config.sb_64x64, config.proj_step) &&
                  tmvp_avail(y8, y8 + 2, h8, config.sb_64x64, config.proj_step) &&
                  traj->valid[(static_cast<std::size_t>(rf) *
                                   static_cast<std::size_t>(h8) +
                               static_cast<std::size_t>(y8 + 2)) *
                                  static_cast<std::size_t>(w8) +
                              static_cast<std::size_t>(x8 + 2)] != 0;
              if (r_avail) {
                ++count;
                avg[0] += traj->mv[traj_mv_index(w8, h8, traj->num_refs, rf,
                                                 y8, x8 + 2, 0)];
                avg[1] += traj->mv[traj_mv_index(w8, h8, traj->num_refs, rf,
                                                 y8, x8 + 2, 1)];
              }
              if (b_avail) {
                ++count;
                avg[0] += traj->mv[traj_mv_index(w8, h8, traj->num_refs, rf,
                                                 y8 + 2, x8, 0)];
                avg[1] += traj->mv[traj_mv_index(w8, h8, traj->num_refs, rf,
                                                 y8 + 2, x8, 1)];
              }
              if (br_avail) {
                ++count;
                avg[0] += traj->mv[traj_mv_index(w8, h8, traj->num_refs, rf,
                                                 y8 + 2, x8 + 2, 0)];
                avg[1] += traj->mv[traj_mv_index(w8, h8, traj->num_refs, rf,
                                                 y8 + 2, x8 + 2, 1)];
              }
              for (int c = 0; c < 2; ++c) {
                traj->mv[traj_mv_index(w8, h8, traj->num_refs, rf, y8 + dy,
                                       x8 + dx, c)] =
                    static_cast<std::int32_t>(calc_avg(avg[c], count));
              }
              traj->valid[(static_cast<std::size_t>(rf) *
                               static_cast<std::size_t>(h8) +
                           static_cast<std::size_t>(y8 + dy)) *
                              static_cast<std::size_t>(w8) +
                          static_cast<std::size_t>(x8 + dx)] = 1;
            }
          }
        }
      }
    }
  }

  if (result != nullptr) {
    result->closest_past = closest_past;
    result->closest_future = closest_future;
    result->projection_count = process_count;
  }
  return TemporalStatus::kOk;
}

TemporalStatus fill_motion_field_sample_gap(const MotionFieldConfig& config,
                                            MotionField* field) {
  if (field == nullptr || config.mi_cols <= 0 || config.mi_rows <= 0) {
    return TemporalStatus::kInvalidArgument;
  }
  if (config.proj_step != 2) {
    return TemporalStatus::kOk;
  }
  const int w8 = config.mi_cols >> 1;
  const int h8 = config.mi_rows >> 1;
  if (field->w8 != w8 || field->h8 != h8) {
    return TemporalStatus::kInvalidArgument;
  }
  auto fill_tpl = [&](int y8, int x8, int dy, int dx) {
    if (field->valid[mf_index(w8, y8, x8)] == 0 || y8 + dy == h8 ||
        x8 + dx == w8) {
      return;
    }
    const int cur_offset = field->offset[mf_index(w8, y8, x8)];
    int count = 0;
    int avg[2] = {0, 0};
    for (int i = 0; i < 4; ++i) {
      const int cand_x = i & 1;
      const int cand_y = i >> 1;
      const bool available =
          dy >= cand_y && dx >= cand_x &&
          tmvp_avail(x8, x8 + 2 * cand_x, w8, config.sb_64x64,
                     config.proj_step) &&
          tmvp_avail(y8, y8 + 2 * cand_y, h8, config.sb_64x64,
                     config.proj_step) &&
          field->valid[mf_index(w8, y8 + 2 * cand_y, x8 + 2 * cand_x)] != 0;
      if (!available) {
        continue;
      }
      ++count;
      Mv proj;
      const std::size_t src =
          mf_index(w8, y8 + 2 * cand_y, x8 + 2 * cand_x);
      if (i == 0) {
        proj = Mv{field->mv[src * 2 + 0], field->mv[src * 2 + 1]};
      } else {
        proj = get_mv_projection_clamp(
            Mv{field->mv[src * 2 + 0], field->mv[src * 2 + 1]}, cur_offset,
            field->offset[src]);
      }
      avg[0] += proj.row;
      avg[1] += proj.col;
    }
    const std::size_t dst = mf_index(w8, y8 + dy, x8 + dx);
    field->offset[dst] = cur_offset;
    field->mv[dst * 2 + 0] = calc_avg(avg[0], count);
    field->mv[dst * 2 + 1] = calc_avg(avg[1], count);
    field->valid[dst] = 1;
  };
  for (int y8 = 0; y8 < h8; y8 += 2) {
    for (int x8 = 0; x8 < w8; x8 += 2) {
      fill_tpl(y8, x8, 0, 1);
      fill_tpl(y8, x8, 1, 0);
      fill_tpl(y8, x8, 1, 1);
    }
  }
  return TemporalStatus::kOk;
}

TemporalStatus setup_tip_motion_field(const MotionFieldConfig& config,
                                      const TipConfig& tip,
                                      MotionField* field,
                                      const TrajectoryField* traj,
                                      MotionFieldScratch* scratch,
                                      MotionFieldResult* result) {
  (void)traj;
  (void)result;
  if (field == nullptr || config.mi_cols <= 0 || config.mi_rows <= 0 ||
      field->w8 != (config.mi_cols >> 1) ||
      field->h8 != (config.mi_rows >> 1)) {
    return TemporalStatus::kInvalidArgument;
  }
  if (tip.mode == TipFrameMode::kDisabled) {
    return TemporalStatus::kOk;
  }
  const int w8 = field->w8;
  const int h8 = field->h8;
  const TipOffsets offsets = get_tip_offsets(tip);
  const int ref_offset = offsets.ref_offset;

  // §7.10.2 temporal scale.
  for (int y8 = 0; y8 < h8; y8 += config.proj_step) {
    for (int x8 = 0; x8 < w8; x8 += config.proj_step) {
      const std::size_t pos = mf_index(w8, y8, x8);
      const Mv mv{field->mv[pos * 2 + 0], field->mv[pos * 2 + 1]};
      if (field->valid[pos] != 0) {
        const int start_offset = field->offset[pos];
        const Mv scaled =
            get_mv_projection_clamp(mv, ref_offset, start_offset);
        field->mv[pos * 2 + 0] = scaled.row;
        field->mv[pos * 2 + 1] = scaled.col;
        field->valid[pos] = 1;
      }
      field->offset[pos] = ref_offset;
    }
  }

  if (tip.allow_tip_hole_fill) {
    const int step = config.proj_step;
    const int sb_size8 = 1 << tmvp_shift(config.sb_64x64, config.proj_step);
    static constexpr int kTipDirs[5][2] = {
        {-1, 0}, {0, -1}, {1, 0}, {0, 1}, {0, 0}};
    for (int y8 = 0; y8 < h8; y8 += sb_size8) {
      for (int x8 = 0; x8 < w8; x8 += sb_size8) {
        const int end_row = std::min(y8 + sb_size8, h8);
        const int end_col = std::min(x8 + sb_size8, w8);
        for (int row8 = y8; row8 < end_row; row8 += step) {
          for (int col8 = x8; col8 < end_col; col8 += step) {
            for (int dir = 0; dir < 4; ++dir) {
              const int dst_row = row8 + kTipDirs[dir][0] * step;
              const int dst_col = col8 + kTipDirs[dir][1] * step;
              if (dst_row < y8 || dst_row >= end_row || dst_col < x8 ||
                  dst_col >= end_col ||
                  field->valid[mf_index(w8, dst_row, dst_col)] != 0) {
                continue;
              }
              const std::size_t src = mf_index(w8, row8, col8);
              const std::size_t dst = mf_index(w8, dst_row, dst_col);
              field->valid[dst] = field->valid[src];
              field->mv[dst * 2 + 0] = field->mv[src * 2 + 0];
              field->mv[dst * 2 + 1] = field->mv[src * 2 + 1];
              field->offset[dst] = field->offset[src];
            }
          }
        }
      }
    }

    // §7.10.4 block average filter.  It needs a separate copy of the field.
    if (scratch == nullptr || scratch->avg_mv == nullptr ||
        scratch->avg_valid == nullptr) {
      return TemporalStatus::kInvalidArgument;
    }
    static constexpr int kWeightDivMult[6] = {0, 65536, 32768, 21845, 16384,
                                              13107};
    for (int y8 = 0; y8 < h8; y8 += sb_size8) {
      for (int x8 = 0; x8 < w8; x8 += sb_size8) {
        const int end_row = std::min(y8 + sb_size8, h8);
        const int end_col = std::min(x8 + sb_size8, w8);
        for (int row8 = y8; row8 < end_row; row8 += step) {
          for (int col8 = x8; col8 < end_col; col8 += step) {
            int mv[2] = {0, 0};
            int count = 0;
            for (int dir = 0; dir < 5; ++dir) {
              const int dst_row = row8 + kTipDirs[dir][0] * step;
              const int dst_col = col8 + kTipDirs[dir][1] * step;
              if (dst_row < y8 || dst_row >= end_row || dst_col < x8 ||
                  dst_col >= end_col ||
                  field->valid[mf_index(w8, dst_row, dst_col)] == 0) {
                continue;
              }
              const std::size_t src = mf_index(w8, dst_row, dst_col);
              mv[0] += field->mv[src * 2 + 0];
              mv[1] += field->mv[src * 2 + 1];
              ++count;
            }
            const std::size_t dst = mf_index(w8, row8, col8);
            if (count == 0) {
              scratch->avg_valid[dst] = 0;
              scratch->avg_mv[dst * 2 + 0] = -(1 << 15);
              scratch->avg_mv[dst * 2 + 1] = -(1 << 15);
            } else {
              scratch->avg_valid[dst] = 1;
              scratch->avg_mv[dst * 2 + 0] = round2_signed(
                  mv[0] * kWeightDivMult[count], 16);
              scratch->avg_mv[dst * 2 + 1] = round2_signed(
                  mv[1] * kWeightDivMult[count], 16);
            }
          }
        }
      }
    }
    for (int y8 = 0; y8 < h8; y8 += step) {
      for (int x8 = 0; x8 < w8; x8 += step) {
        const std::size_t pos = mf_index(w8, y8, x8);
        field->mv[pos * 2 + 0] = scratch->avg_mv[pos * 2 + 0];
        field->mv[pos * 2 + 1] = scratch->avg_mv[pos * 2 + 1];
        field->valid[pos] = scratch->avg_valid[pos];
      }
    }
  }

  const TemporalStatus status = fill_motion_field_sample_gap(config, field);
  if (status != TemporalStatus::kOk) {
    return status;
  }
  return TemporalStatus::kOk;
}

TemporalStatus build_tip_plan(const MotionFieldConfig& config,
                              const TipConfig& tip, const MotionField& field,
                              TipPlan* plan) {
  if (plan == nullptr || plan->blocks == nullptr || plan->mv_grid == nullptr ||
      plan->num_blocks < 0 || config.mi_cols <= 0 || config.mi_rows <= 0) {
    return TemporalStatus::kInvalidArgument;
  }
  int weight = kCwpEqual;
  if (!tip_weight(tip.tip_global_wtd_index, &weight)) {
    return TemporalStatus::kInvalidArgument;
  }
  const int bw4 = (tip.enable_tip_refinemv &&
                   tip.interp_filter == InterpFilter::kEighttapSharp)
                      ? 2
                      : 4;  // BLOCK_8X8 or BLOCK_16X16
  const int bh4 = bw4;
  const TipOffsets offsets = get_tip_offsets(tip);
  int count = 0;
  for (int row = 0; row < config.mi_rows; row += bh4) {
    for (int col = 0; col < config.mi_cols; col += bw4) {
      if (count >= plan->num_blocks) {
        return TemporalStatus::kInvalidArgument;
      }
      const int y8 = row >> 1;
      const int x8 = col >> 1;
      Mv local[2];
      const std::size_t pos = mf_index(field.w8, y8, x8);
      if (field.valid[pos] == 0) {
        local[0] = Mv{0, 0};
        local[1] = Mv{0, 0};
      } else {
        const Mv field_mv{field.mv[pos * 2 + 0], field.mv[pos * 2 + 1]};
        local[0] = get_mv_projection(field_mv, offsets.past_offset,
                                     offsets.ref_offset);
        local[1] = get_mv_projection(field_mv, offsets.future_offset,
                                     offsets.ref_offset);
      }
      TipBlock& block = plan->blocks[count];
      block.mi_row = row;
      block.mi_col = col;
      block.bw4 = bw4;
      block.bh4 = bh4;
      block.mv[0] = Mv{local[0].row + tip.tip_global_mv.row,
                       local[0].col + tip.tip_global_mv.col};
      block.mv[1] = Mv{local[1].row + tip.tip_global_mv.row,
                       local[1].col + tip.tip_global_mv.col};
      block.motion_field_valid = field.valid[pos] != 0;
      for (int i = 0; i < bh4; ++i) {
        for (int j = 0; j < bw4; ++j) {
          const int r = row + i;
          const int c = col + j;
          if (r < config.mi_rows && c < config.mi_cols) {
            const std::size_t index =
                static_cast<std::size_t>(r) *
                    static_cast<std::size_t>(config.mi_cols) +
                static_cast<std::size_t>(c);
            plan->mv_grid[index] = block.mv[0];
            if (plan->mv_grid1 != nullptr) {
              plan->mv_grid1[index] = block.mv[1];
            }
          }
        }
      }
      ++count;
    }
  }
  plan->offsets = offsets;
  plan->weight = weight;
  plan->num_blocks = count;
  return TemporalStatus::kOk;
}

Mv get_motion_field_mv(const MotionField& field, const TrajectoryField* traj,
                       int num_refs, int dst_ref, int order_hint,
                       int dst_order_hint, bool use_traj, int y8, int x8) {
  const std::size_t pos = mf_index(field.w8, y8, x8);
  if (use_traj && traj != nullptr && dst_ref >= 0 &&
      dst_ref < traj->num_refs) {
    const std::size_t idx =
        (static_cast<std::size_t>(dst_ref) * static_cast<std::size_t>(field.h8) +
         static_cast<std::size_t>(y8)) *
            static_cast<std::size_t>(field.w8) +
        static_cast<std::size_t>(x8);
    if (traj->valid[idx] != 0) {
      return Mv{traj->mv[traj_mv_index(field.w8, field.h8, traj->num_refs,
                                       dst_ref, y8, x8, 0)],
                traj->mv[traj_mv_index(field.w8, field.h8, traj->num_refs,
                                       dst_ref, y8, x8, 1)]};
    }
  }
  (void)num_refs;
  const Mv mv{field.mv[pos * 2 + 0], field.mv[pos * 2 + 1]};
  const int ref_offset = field.offset[pos];
  const int ref_to_dst = get_relative_dist(order_hint, dst_order_hint);
  return get_mv_projection(mv, ref_to_dst, ref_offset);
}

TemporalStatus temporal_sample(const TemporalScanInput& in, int delta_row,
                               int delta_col, StackMv* stack,
                               int* num_mv_found, int* prune_count) {
  if (in.field == nullptr || stack == nullptr || num_mv_found == nullptr ||
      prune_count == nullptr) {
    return TemporalStatus::kInvalidArgument;
  }
  if (*num_mv_found >= kMaxRefMvStackSize) {
    return TemporalStatus::kOk;
  }
  const int mv_row = in.mi_row + delta_row;
  const int mv_col = in.mi_col + delta_col;
  const int row_start =
      in.tile_mi_row_end > in.tile_mi_row_start ? in.tile_mi_row_start : 0;
  const int row_end =
      in.tile_mi_row_end > in.tile_mi_row_start ? in.tile_mi_row_end
                                                : in.mi_rows;
  const int col_start =
      in.tile_mi_col_end > in.tile_mi_col_start ? in.tile_mi_col_start : 0;
  const int col_end =
      in.tile_mi_col_end > in.tile_mi_col_start ? in.tile_mi_col_end
                                                : in.mi_cols;
  if (mv_row < row_start || mv_row >= row_end || mv_col < col_start ||
      mv_col >= col_end) {
    return TemporalStatus::kOk;
  }
  const int x8 = mv_col >> 1;
  const int y8 = mv_row >> 1;
  const std::size_t pos = mf_index(in.field->w8, y8, x8);
  if (in.field->valid[pos] == 0) {
    return TemporalStatus::kOk;
  }
  if (!in.is_compound) {
    const Mv cand = get_motion_field_mv(
        *in.field, in.traj, in.num_refs, in.ref0, in.order_hint,
        in.ref0_order_hint, in.use_traj, y8, x8);
    int idx = *num_mv_found;
    if (*prune_count >= kMaxPrNum) {
      idx = *num_mv_found;
    } else {
      for (int i = 0; i < *num_mv_found; ++i) {
        ++(*prune_count);
        if (cand == stack[i].mv[0]) {
          idx = i;
          break;
        }
      }
    }
    const int weight =
        abs_int(get_relative_dist(in.order_hint, in.ref0_order_hint)) <= 2 ? 2
                                                                           : 1;
    if (idx < *num_mv_found) {
      stack[idx].weight += weight;
    } else {
      stack[*num_mv_found].mv[0] = cand;
      stack[*num_mv_found].mv[1] = cand;
      stack[*num_mv_found].weight = weight;
      stack[*num_mv_found].cwp = kCwpEqual;
      ++(*num_mv_found);
    }
  } else {
    const Mv cand0 = get_motion_field_mv(
        *in.field, in.traj, in.num_refs, in.ref0, in.order_hint,
        in.ref0_order_hint, in.use_traj, y8, x8);
    const Mv cand1 = get_motion_field_mv(
        *in.field, in.traj, in.num_refs, in.ref1, in.order_hint,
        in.ref1_order_hint, in.use_traj, y8, x8);
    int idx = *num_mv_found;
    if (*prune_count >= kMaxPrNum) {
      idx = *num_mv_found;
    } else {
      for (int i = 0; i < *num_mv_found; ++i) {
        ++(*prune_count);
        if (cand0 == stack[i].mv[0] && cand1 == stack[i].mv[1]) {
          idx = i;
          break;
        }
      }
    }
    if (idx < *num_mv_found) {
      stack[idx].weight += 1;
    } else {
      stack[*num_mv_found].mv[0] = cand0;
      stack[*num_mv_found].mv[1] = cand1;
      stack[*num_mv_found].weight = 1;
      stack[*num_mv_found].cwp = kCwpEqual;
      ++(*num_mv_found);
    }
  }
  return TemporalStatus::kOk;
}

TemporalStatus temporal_scan(const TemporalScanInput& in, StackMv* stack,
                             int* num_mv_found, int* prune_count) {
  if (in.field == nullptr || stack == nullptr || num_mv_found == nullptr ||
      prune_count == nullptr || in.bw4 <= 0 || in.bh4 <= 0) {
    return TemporalStatus::kInvalidArgument;
  }
  const int step_w4 = in.bw4 >= 16 ? 4 : 2;
  const int step_h4 = in.bh4 >= 16 ? 4 : 2;
  const int row_end = std::min(in.bh4, 16);
  const int col_end = std::min(in.bw4, 16);
  const int delta_row = row_end - step_h4;
  const int delta_col = col_end - step_w4;
  const int start_found = *num_mv_found;
  if (delta_row >= 0 && delta_col >= 0) {
    const TemporalStatus s = temporal_sample(in, delta_row, delta_col, stack,
                                             num_mv_found, prune_count);
    if (s != TemporalStatus::kOk) {
      return s;
    }
  }
  if ((row_end >= 3 * step_h4 || col_end >= 3 * step_w4) &&
      start_found == *num_mv_found) {
    return temporal_sample(in, row_end >> 1, col_end >> 1, stack, num_mv_found,
                           prune_count);
  }
  return TemporalStatus::kOk;
}

MvPair get_tip_cand(const TipCandInput& in, const TipConfig& tip) {
  MvPair out;
  if (in.field == nullptr) {
    out.mv[0] = in.base_mv;
    out.mv[1] = in.base_mv;
    return out;
  }
  if (in.field->w8 <= 0 || in.field->h8 <= 0) {
    out.mv[0] = in.base_mv;
    out.mv[1] = in.base_mv;
    return out;
  }
  const int shift = in.is_16x16 ? 2 : 1;
  const int cand_row = in.base_row + (((in.cand_row - in.base_row) >> shift) << shift);
  const int cand_col = in.base_col + (((in.cand_col - in.base_col) >> shift) << shift);
  const int x8 = cand_col >> 1;
  const int y8 = cand_row >> 1;
  out.mv[0] = Mv{0, 0};
  out.mv[1] = Mv{0, 0};
  const int ref_x8 = clip3(0, in.field->w8 - 1, x8);
  const int ref_y8 = clip3(0, in.field->h8 - 1, y8);
  const std::size_t pos = mf_index(in.field->w8, ref_y8, ref_x8);
  if (in.field->w8 > 0 && in.field->h8 > 0 &&
      in.field->valid[pos] != 0) {
    const TipOffsets offsets = get_tip_offsets(tip);
    const Mv field_mv{in.field->mv[pos * 2 + 0], in.field->mv[pos * 2 + 1]};
    out.mv[0] =
        get_mv_projection(field_mv, offsets.past_offset, offsets.ref_offset);
    out.mv[1] =
        get_mv_projection(field_mv, offsets.future_offset, offsets.ref_offset);
  }
  for (int list = 0; list < 2; ++list) {
    out.mv[list].row = clip3(kMvLow + 1, kMvUpp - 1,
                             out.mv[list].row + in.base_mv.row);
    out.mv[list].col = clip3(kMvLow + 1, kMvUpp - 1,
                             out.mv[list].col + in.base_mv.col);
  }
  return out;
}

TemporalStatus store_motion_field_entry(const MotionFieldStoreEntry& in,
                                        RefMvField* out, int y8, int x8) {
  if (out == nullptr || out->ref == nullptr || out->mv == nullptr ||
      y8 < 0 || x8 < 0 || y8 >= out->h8 || x8 >= out->w8 ||
      in.i8 < 0 || in.i8 > 7 || in.j8 < 0 || in.j8 > 7) {
    return TemporalStatus::kInvalidArgument;
  }
  const std::size_t pos = mf_index(out->w8, y8, x8);
  int refs[2] = {in.refs[0], in.refs[1]};
  Mv mvs[2] = {in.mv[0], in.mv[1]};
  std::uint8_t allow[2] = {1, 1};
  if (in.is_wedge && in.mask != nullptr) {
    int count0 = 0;
    int count1 = 0;
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
        const int m =
            in.mask[static_cast<std::size_t>(in.i8 * 8 + i) *
                        static_cast<std::size_t>(in.mask_stride) +
                    static_cast<std::size_t>(in.j8 * 8 + j)];
        if (m > 60) {
          ++count0;
        }
        if (m < 4) {
          ++count1;
        }
      }
    }
    if (count0 >= 60) {
      allow[1] = 0;
    } else if (count1 >= 60) {
      allow[0] = 0;
    }
  }

  int final_refs[2] = {kRefNone, kRefNone};
  Mv final_mv[2];
  final_mv[0] = Mv{0, 0};
  final_mv[1] = Mv{0, 0};
  for (int list = 0; list < 2; ++list) {
    if (!is_inter_ref(refs[list])) {
      continue;
    }
    if (abs_int(mvs[list].row) <= kRefMvsLimit &&
        abs_int(mvs[list].col) <= kRefMvsLimit && allow[list] != 0) {
      final_refs[list] = refs[list];
      final_mv[list] = mvs[list];
    }
  }
  // §7.22 reordering: if one list is absent, copy the other.  If both are
  // present, order by past/future relative to the current frame.
  if (final_refs[0] != kRefNone && final_refs[1] == kRefNone) {
    final_refs[1] = final_refs[0];
    final_mv[1] = final_mv[0];
  } else if (final_refs[1] != kRefNone && final_refs[0] == kRefNone) {
    final_refs[0] = final_refs[1];
    final_mv[0] = final_mv[1];
  } else if (final_refs[0] != kRefNone && final_refs[1] != kRefNone) {
    const int rel0 = get_relative_dist(in.ref_order_hint[0], in.order_hint);
    const int rel1 = get_relative_dist(in.ref_order_hint[1], in.order_hint);
    bool to_switch;
    if (rel0 < 0 && rel1 < 0) {
      to_switch =
          get_relative_dist(in.ref_order_hint[0], in.ref_order_hint[1]) < 0;
    } else if (rel0 > 0 && rel1 > 0) {
      to_switch =
          get_relative_dist(in.ref_order_hint[0], in.ref_order_hint[1]) < 0;
    } else {
      to_switch = rel0 > 0 && rel1 < 0;
    }
    if (to_switch) {
      std::swap(final_refs[0], final_refs[1]);
      std::swap(final_mv[0], final_mv[1]);
    }
  }
  for (int list = 0; list < 2; ++list) {
    out->ref[pos * 2 + static_cast<std::size_t>(list)] =
        static_cast<std::int8_t>(final_refs[list]);
    out->mv[(pos * 2 + static_cast<std::size_t>(list)) * 2 + 0] =
        compression_mv(final_mv[list].row);
    out->mv[(pos * 2 + static_cast<std::size_t>(list)) * 2 + 1] =
        compression_mv(final_mv[list].col);
  }
  return TemporalStatus::kOk;
}

}  // namespace av2enc::inter::temporal
