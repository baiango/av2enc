// AV2 inter motion estimation — CPU reference API.
//
// Scope: 8-bit 4:2:0 luma ME with SAD/SATD, full / diamond / TZ-style search,
// integer-pel through 1/16-pel refinement, spatial MVP/merge candidates, and
// multiple references.  This is a correctness oracle for the CUDA kernel in
// src/cuda/inter/me/; it is deliberately scalar and single-threaded.
//
// No bitstream syntax is written here.  The output is a per-block MV (in
// 1/16 luma samples), the selected reference index, and the search cost for
// the MC / RDO / mode-decision components.
//
// Batch API, no per-block allocation: the caller supplies the job and result
// arrays; the implementation uses fixed-size stack state only.

#ifndef AV2ENC_ENCODER_INTER_ME_ME_H_
#define AV2ENC_ENCODER_INTER_ME_ME_H_

#include <cstdint>

#include "encoder/inter/me/me_core.h"

namespace av2enc::me {

enum class MeStatus {
  kOk = 0,
  kInvalidArgument,
  kUnsupportedFormat,
};

const char* me_status_string(MeStatus status);

// Validates the frame/ref/job/options and searches every job.  `field` may be
// null when the caller does not want spatial merge candidates.  `results` must
// have `num_jobs` entries.  Returns kInvalidArgument if any pointer, geometry,
// ref index, or option is out of range; in that case results are untouched.
MeStatus me_search_batch(const MeCoreFrame& cur, const MeCoreRef* refs,
                         int num_refs, const MeCoreJob* jobs, int num_jobs,
                         const MeCoreOptions& options,
                         const MeCoreMvField* field, MeCoreResult* results);

// Cost of one candidate with an explicit plane (0 = luma, 1/2 = chroma).
// Coordinates and the MV are plane-relative (chroma MVs are in 1/16 chroma
// samples).  The block must lie inside the current plane; subpel positions may
// reach 15/16 beyond the integer reference edge and are clipped during
// interpolation.  (The batch search entry point also accepts partial blocks.)
MeStatus me_cost_block(const MeCoreFrame& cur, const MeCoreRef& ref,
                       const MeCostJob& job, std::int32_t* out_cost);

// Normative subpel prediction for one block.  `out` receives w*h samples in
// raster order.  Integer MVs copy reference samples; fractional MVs use the
// requested Subpel_Filters row.  Used by tests and by the MC component as a
// reference oracle (the CUDA counterpart is me_interp_batch).
MeStatus me_interp_block(const MeCoreRef& ref, const MeInterpJob& job,
                         std::uint8_t* out);

// Spatial merge candidates for one block.  Writes up to `max_out` entries and
// returns the count (0 on invalid input).  A zero-MV candidate is appended
// when absent.  Safe to call with a null/empty field: returns the zero
// candidate.
int me_collect_mv_candidates(const MeCoreMvField& field, int x, int y, int w,
                             int h, int default_ref, MeMvCandidate* out,
                             int max_out);

}  // namespace av2enc::me

#endif  // AV2ENC_ENCODER_INTER_ME_ME_H_
