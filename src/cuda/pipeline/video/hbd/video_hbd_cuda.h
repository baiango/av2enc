// CUDA entry point for the format video core (10-bit / 8-bit chroma formats).
//
// Decisions and bitstream assembly are shared with the CPU reference
// (`src/pipeline/video/hbd/video_hbd.cpp`); this module provides the pixel
// backend.  The caller owns the arena (or passes nullptr and the backend
// creates one sized from the device).
#pragma once

#include <span>
#include <string>

#include "pipeline/video/video_frame.h"

namespace av2enc::cuda {
class Arena;
}

namespace av2enc::cuda::video::hbd_gpu {

bool encode_video_sequence_format_gpu(
    std::span<const av2enc::video::VideoInputFrame> frames,
    const av2enc::video::VideoOptions& options,
    av2enc::video::VideoSequenceResult& result, std::string& error,
    av2enc::cuda::Arena* arena);

}  // namespace av2enc::cuda::video::hbd_gpu
