// Wave-3 frame types for the --video path: long-term references / OLK,
// leading pictures, show-existing-frame, SWITCH and RAS.
//
// These modes are compiled only when AV2ENC_VIDEO_FRAMES is defined (the
// product build and the CMake gate).  The standalone per-component gate
// scripts compile src/pipeline/video/video_frame.cpp alone, so keeping the
// module dependencies out of that TU keeps every existing gate linkable.
#pragma once

#include "pipeline/video/video_frame.h"

#include <span>
#include <string>

namespace av2enc::video::frames {

// Dispatches every default-off frame-type mode.  Returns false with `error`
// when no mode is selected or a combination is invalid.
bool encode_special_sequence(std::span<const VideoInputFrame> frames,
                             const VideoOptions& options,
                             VideoSequenceResult& result, std::string& error);

}  // namespace av2enc::video::frames
