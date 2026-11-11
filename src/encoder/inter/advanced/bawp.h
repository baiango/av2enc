#ifndef AV2ENC_INTER_ADVANCED_BAWP_H_
#define AV2ENC_INTER_ADVANCED_BAWP_H_

// CPU reference for AV2 v1.0.0 block adaptive weighted prediction
// (§7.13.3.25) and morphological prediction (§7.13.3.26).
//
// BAWP adjusts the inter prediction of the current block to match the
// reconstructed above/left neighbours. The encoder signals one mode flag
// (`use_bawp`) plus `explicit_bawp` / `explicit_bawp_scale`; the decoder mirrors
// the same least-squares or explicit fit.
//
// Integration sketch:
//
//   adv::BawpJob job;
//   job.ctx = {.plane = 0, .mi_col = mi_col, .mi_row = mi_row, ...};
//   job.rect = {x, y, w, h};
//   job.mv = {mv_row, mv_col};
//   job.cur = current_plane;      // reads above/left reconstruction
//   job.prev = reference_plane;   // or job.cur when morph_pred
//   job.dst = current_plane;      // in-place adjustment
//   adv::BawpFit fit;
//   adv::bawp_apply_block(job, &fit);   // fit.alpha is BawpAlpha for chroma
//
// The caller is responsible for the conformance requirements (§7.13.3.25): the
// reference block must be at least one sample inside the frame. The job returns
// false rather than reading out of bounds when they do not hold.
//
// `morph_pred` selects §7.13.3.26: the template reference is the current frame
// and `mv` is the IntraBC vector.

#include "bawp_common.h"

namespace av2enc::inter::adv {

struct BawpJob {
  BawpContext ctx{};
  BlockRect rect{};
  int mv[2] = {0, 0};  // 1/8 luma sample units

  // Luma alpha carried over to chroma (BawpAlpha).
  int bawp_alpha = 256;
  bool explicit_bawp = false;
  bool explicit_bawp_scale = false;
  int list_index = 2;
  int first_ref_dist = 0;
  bool morph_pred = false;

  PlaneView cur{};   // current frame plane; reads reconstructed neighbours
  PlaneView prev{};  // reference plane (== cur for morph_pred)
  MutablePlaneView dst{};
  // Optional: computed reference block origin, filled by the call for debugging.
  int ref_x = 0;
  int ref_y = 0;
};

// Derives and applies the adjustment. On success `fit_out` (optional) receives
// the fitted alpha/beta/count. For plane 0 the caller should keep
// `fit.alpha` as the frame's BawpAlpha for the chroma planes.
bool bawp_apply_block(const BawpJob& job, BawpFit* fit_out);

// Derives the fit only (no writes). Useful for unit tests and for deciding
// whether BAWP is worth signalling.
bool bawp_derive_block(const BawpJob& job, BawpFit* fit_out);

// Maps a YMode to the §7.13.3.25 `listIndex` used by explicit BAWP:
//   NEARMV -> 0, NEWMV && use_amvd -> 1, otherwise -> 2.
int bawp_list_index(bool is_near_mv, bool is_new_mv, bool use_amvd);

}  // namespace av2enc::inter::adv

#endif  // AV2ENC_INTER_ADVANCED_BAWP_H_
