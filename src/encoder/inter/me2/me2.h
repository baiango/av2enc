// AV2 inter motion estimation, extended — CPU reference API.
//
// Additive extension of src/encoder/inter/me/ (unchanged).  Adds eighth- and
// sixteenth-pel search stages, deterministic exhaustive refinement, and
// adaptive per-block MV precision (enable_flex_mvres) with the frame-header
// and per-block signalling implications described in me2_precision.h.
//
// No bitstream syntax is written here.  Results are encoder-side decision
// data: the signalled MV (1/16 units, quantized to the selected precision),
// the selected precision, distortion at that MV, and a deterministic MV-rate
// proxy for RDO/mode decision.

#ifndef AV2ENC_ENCODER_INTER_ME2_ME2_H_
#define AV2ENC_ENCODER_INTER_ME2_ME2_H_

#include <cstdint>

#include "encoder/inter/me2/me2_core.h"

namespace av2enc::me2 {

enum class Me2Status {
  kOk = 0,
  kInvalidArgument,
  kUnsupportedFormat,
};

const char* me2_status_string(Me2Status status);

// Searches every job.  `field` may be null (no spatial merge candidates).
// `results` must have `num_jobs` entries.  Returns kInvalidArgument on any
// geometry/option/pointer failure; in that case results are untouched.
Me2Status me2_search_batch(const MeCoreFrame& cur, const MeCoreRef* refs,
                           int num_refs, const MeCoreJob* jobs, int num_jobs,
                           const Me2Options& options,
                           const MeCoreMvField* field, Me2Result* results);

// Cost of one explicit candidate at a signalled precision.  The MV must
// already lie on that precision's grid.  Plane coordinates/MV precision follow
// me::MeCostJob: plane 0 is luma, chroma MVs are 1/16 chroma samples.
struct Me2CostJob {
  me::MeCostJob base;
  int precision = kMvPrecisionEighthPel;
};

Me2Status me2_cost_block(const MeCoreFrame& cur, const MeCoreRef& ref,
                         const Me2CostJob& job, std::int32_t* out_cost);

// Normative subpel prediction, delegated to the ME reference oracle.
Me2Status me2_interp_block(const MeCoreRef& ref, const me::MeInterpJob& job,
                           std::uint8_t* out);

}  // namespace av2enc::me2

#endif  // AV2ENC_ENCODER_INTER_ME2_ME2_H_
