// Exact coefficient rate in the leaf decisions.
//
// The TX search prices its plans with the §8.3 coefficient model when
// --video-rdoq is on, but the skip/residual and intra/inter decisions charge
// the historical heuristic (`8 + 2.5*nz + 0.5*eob` per part).  That heuristic
// over-prices sparse residual plans, so a leaf whose exact coded residual is
// cheap can still be skipped.  AV2ENC_DECISION_EXACT_RATE makes the decision
// charge the plan's own measured rate instead.  Off by default.
#ifndef AV2ENC_PIPELINE_VIDEO_DECISION_RATE_H_
#define AV2ENC_PIPELINE_VIDEO_DECISION_RATE_H_

#include <cstdlib>

namespace av2enc::video::decision_rate {

inline bool exact() {
  const char* value = std::getenv("AV2ENC_DECISION_EXACT_RATE");
  if (value == nullptr) return false;
  if (value[0] == '\0' || value[0] == '0') return false;
  if (value[0] == 'n' || value[0] == 'N') return false;
  return true;
}

}  // namespace av2enc::video::decision_rate

#endif  // AV2ENC_PIPELINE_VIDEO_DECISION_RATE_H_
