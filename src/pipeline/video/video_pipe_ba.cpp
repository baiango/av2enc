// AV2ENC_PIPE_BA translation unit.
//
// The model and its mask builder are header-only (`video_eff_perc.h` is the
// same shape) because a lane worktree may not own the shared source list in
// `CMakeLists.txt`.  This unit carries the compile-time contract so a build
// that does compile it checks the mask range the pipeline relies on.
#include "pipeline/video/video_pipe_ba.h"

namespace av2enc::video::pipe_ba {

static_assert(kMaskNeutralQ8 == 256, "PIPE_BA mask must be neutral at Q8 256");
static_assert(kMaskFloorQ8 > 0, "PIPE_BA must never mask a sample to zero");
static_assert(kMaskFloorQ8 < kMaskNeutralQ8, "PIPE_BA floor must discount");
static_assert(kLowBandGain > 0.0, "PIPE_BA prices the low/mid error band");

}  // namespace av2enc::video::pipe_ba
