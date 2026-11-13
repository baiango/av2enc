// Internal orchestration entry points shared by the plain P-chain encoder and
// the frame-type wrappers (long-term/OLK, leading, SEF, SWITCH/RAS).
//
// These are not part of the product API in video_gpu.h.  The wrappers assemble
// their OBUs with the CPU reference modules and only replace the tile payload
// source with `encode_video_sequence_gpu()` on the plain sub-encodes, so the
// output is byte-identical whenever the plain path is.
#ifndef AV2ENC_CUDA_VIDEO_M2_SEQUENCE_INTERNAL_H_
#define AV2ENC_CUDA_VIDEO_M2_SEQUENCE_INTERNAL_H_

#include <span>
#include <string>

#include "pipeline/video/video_frame.h"
#include "video_gpu.h"

namespace av2enc::cuda::video::detail {

// True when the options select one of the wave-3 frame-type wrappers.
bool is_frame_type_mode(const av2enc::video::VideoOptions& options);

// Assembles a wrapped sequence.  The actual pixel work for every sub-encode is
// still performed by encode_video_sequence_gpu(); this function only mirrors
// src/pipeline/video/video_frames.cpp with the GPU tile payloads.
bool encode_special_sequence_gpu(
    std::span<const av2enc::video::VideoInputFrame> frames,
    const av2enc::video::VideoOptions& options,
    av2enc::video::VideoSequenceResult& result, std::string& error,
    VideoGpuStats* stats);

class Backend;

// B-frame (one closed IPBB GOP) and hierarchical-P temporal-layer plans.
// Both reuse the same device backend/frame encoder as the plain P chain.
bool encode_b_sequence_gpu(Backend& backend,
                           std::span<const av2enc::video::VideoInputFrame> frames,
                           const av2enc::video::VideoOptions& options,
                           av2enc::video::VideoSequenceResult& result,
                           std::string& error);
bool encode_temporal_sequence_gpu(
    Backend& backend,
    std::span<const av2enc::video::VideoInputFrame> frames,
    const av2enc::video::VideoOptions& options,
    av2enc::video::VideoSequenceResult& result, std::string& error);

}  // namespace av2enc::cuda::video::detail

#endif  // AV2ENC_CUDA_VIDEO_M2_SEQUENCE_INTERNAL_H_
