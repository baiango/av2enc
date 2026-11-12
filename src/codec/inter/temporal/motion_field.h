// AV2 v1.0.0 temporal inter tools — motion field, projection, TIP setup.
//
// Public API for the §7.9 motion-field estimation (temporal MV projection),
// §7.10 TIP motion-field setup, §7.22 motion-field storage, the §7.12.2.7/.8
// temporal MV-stack scan, and TIP candidate derivation.
//
// The CPU driver in this component owns storage.  All core functions take POD
// views/pointers so the same code is usable from a CUDA kernel later.

#ifndef AV2ENC_ENCODER_INTER_TEMPORAL_MOTION_FIELD_H_
#define AV2ENC_ENCODER_INTER_TEMPORAL_MOTION_FIELD_H_

#include <cstddef>
#include <cstdint>

#include "codec/inter/temporal/temporal_types.h"

namespace av2enc::inter::temporal {

// ---------------------------------------------------------------------------
// Index helpers for the flat arrays in temporal_types.h.
// ---------------------------------------------------------------------------

inline std::size_t mf_index(int w8, int y8, int x8) {
  return static_cast<std::size_t>(y8) * static_cast<std::size_t>(w8) +
         static_cast<std::size_t>(x8);
}
inline std::size_t traj_mv_index(int w8, int h8, int num_refs, int rf, int y8,
                                 int x8, int comp) {
  (void)num_refs;
  return ((static_cast<std::size_t>(rf) * static_cast<std::size_t>(h8) +
           static_cast<std::size_t>(y8)) *
              static_cast<std::size_t>(w8) +
          static_cast<std::size_t>(x8)) *
             2 +
         static_cast<std::size_t>(comp);
}
inline std::size_t traj_pos_index(int k, int num_refs, int w8, int h8, int rf,
                                  int y8, int x8, int comp) {
  return (((static_cast<std::size_t>(k) * static_cast<std::size_t>(num_refs) +
            static_cast<std::size_t>(rf)) *
               static_cast<std::size_t>(h8) +
           static_cast<std::size_t>(y8)) *
              static_cast<std::size_t>(w8) +
          static_cast<std::size_t>(x8)) *
             2 +
         static_cast<std::size_t>(comp);
}

// ---------------------------------------------------------------------------
// §7.9 motion field estimation (temporal MV projection).
// ---------------------------------------------------------------------------

// Runs §7.9.1: sort/topologically order the reference frames, record the
// projections, and project the saved motion vectors.  `field` must already be
// allocated (w8 = mi_cols >> 1, h8 = mi_rows >> 1) and zero-initialised; the
// function initialises it.  `traj` may be null unless `enable_mv_traj` is set.
// When enabled, `scratch` must supply the TIP average buffers for
// tip_setup_motion_field(); it is otherwise unused here.
TemporalStatus motion_field_estimation(const MotionFieldConfig& config,
                                       MotionField* field,
                                       TrajectoryField* traj,
                                       MotionFieldResult* result);

// §7.10: temporally scale the motion field, optionally fill holes and smooth
// it, then fill the ProjStep==2 sample gaps.  `scratch` is required when
// `config.enable_tip` and `TipConfig::allow_tip_hole_fill` is set (pass the
// same scratch to build_tip_plan()).
TemporalStatus setup_tip_motion_field(const MotionFieldConfig& config,
                                      const TipConfig& tip,
                                      MotionField* field,
                                      const TrajectoryField* traj,
                                      MotionFieldScratch* scratch,
                                      MotionFieldResult* result);

// §7.10.5 fill_tpl_mvs_sample_gap().  Called by §7.10 for TIP frames and by
// §5.18.2 at the end of motion field estimation for non-TIP frames.
TemporalStatus fill_motion_field_sample_gap(const MotionFieldConfig& config,
                                            MotionField* field);

// §7.10.6 build_tip_process(): fills a per-block TIP plan and the per-4x4 MV
// grid.  `plan->num_blocks` is the capacity on input and the number of blocks
// on output.
TemporalStatus build_tip_plan(const MotionFieldConfig& config,
                              const TipConfig& tip, const MotionField& field,
                              TipPlan* plan);

// ---------------------------------------------------------------------------
// §7.12.2.7/.8 temporal scan.
// ---------------------------------------------------------------------------

struct TemporalScanInput {
  int mi_row = 0;  // current block origin, MI units
  int mi_col = 0;
  int mi_cols = 0;  // MiCols
  int mi_rows = 0;  // MiRows
  // Tile bounds for the spec's tile-relative is_inside().  Leave the ends at
  // zero for a single tile covering the frame.
  int tile_mi_col_start = 0;
  int tile_mi_col_end = 0;
  int tile_mi_row_start = 0;
  int tile_mi_row_end = 0;
  int bw4 = 1;
  int bh4 = 1;
  int ref0 = kRefNone;   // RefFrame[0]
  int ref1 = kRefNone;   // RefFrame[1]
  int ref0_order_hint = kRestrictedOrderHint;
  int ref1_order_hint = kRestrictedOrderHint;
  int order_hint = 0;
  bool is_compound = false;
  bool use_traj = false;
  const MotionField* field = nullptr;
  const TrajectoryField* traj = nullptr;  // optional (§7.12.2.8)
  int num_refs = 0;                       // for traj indexing
};

// Adds temporal candidates to the MV stack.  `*num_mv_found` and
// `*prune_count` are updated exactly like the spec's NumMvFound/PruneCount.
// The input stack must have room for kMaxRefMvStackSize entries.
TemporalStatus temporal_scan(const TemporalScanInput& in, StackMv* stack,
                             int* num_mv_found, int* prune_count);

// §7.12.2.8 helper: one candidate location (MI units).
TemporalStatus temporal_sample(const TemporalScanInput& in, int delta_row,
                               int delta_col, StackMv* stack,
                               int* num_mv_found, int* prune_count);

// §7.22/§7.13.3.2 get_tip_cand().
MvPair get_tip_cand(const TipCandInput& in, const TipConfig& tip);

// §7.12.2.8 get_motion_field_mv(): trajectory MV when available, otherwise a
// projection of the motion field onto the requested reference.
Mv get_motion_field_mv(const MotionField& field, const TrajectoryField* traj,
                       int num_refs, int dst_ref, int order_hint,
                       int dst_order_hint, bool use_traj, int y8, int x8);

// ---------------------------------------------------------------------------
// §7.22 motion field motion vector storage.
// ---------------------------------------------------------------------------

struct RefMvField {
  int w8 = 0;
  int h8 = 0;
  std::int8_t* ref = nullptr;   // [y8][x8][2]
  std::int16_t* mv = nullptr;   // [y8][x8][2][2], compressed
};

struct MotionFieldStoreEntry {
  int refs[2] = {kRefNone, kRefNone};
  Mv mv[2];
  bool is_wedge = false;
  const std::uint8_t* mask = nullptr;  // 64x64 wedge mask (luma)
  int mask_stride = 0;
  int i8 = 0;  // 8x8 position inside the block
  int j8 = 0;
  // §7.22 past/future ordering.
  int order_hint = 0;
  int ref_order_hint[2] = {kRestrictedOrderHint, kRestrictedOrderHint};
};

// Stores one 8x8 entry, applying the wedge allow-list and past/future ordering
// from §7.22.  `out` must be allocated by the caller.
TemporalStatus store_motion_field_entry(const MotionFieldStoreEntry& in,
                                        RefMvField* out, int y8, int x8);

}  // namespace av2enc::inter::temporal

#endif  // AV2ENC_ENCODER_INTER_TEMPORAL_MOTION_FIELD_H_
