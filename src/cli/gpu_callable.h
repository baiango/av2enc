#pragma once

// Which GPU entry points this build links.
//
// These macros used to be defined inside src/main.cpp only.  When the CLI was
// split into cli/*.cpp, validate.cpp started testing AV2ENC_VIDEO_GPU_CALLABLE
// in a TU where it was undefined -- and an undefined identifier is 0 in #if, so
// the "not linked in this build" rejection was compiled in unconditionally.
// A CUDA build therefore rejected --video-gpu with a message claiming the
// backend was missing, even though both the header and the symbol were linked.
//
// Keep the availability rule in one place so every CLI TU agrees.

#if __has_include("cuda/pipeline/still/still_gpu.h")
#define AV2ENC_STILL_GPU_HEADER 1
#else
#define AV2ENC_STILL_GPU_HEADER 0
#endif

#if __has_include("cuda/pipeline/video/video_gpu.h")
#define AV2ENC_VIDEO_GPU_HEADER 1
#else
#define AV2ENC_VIDEO_GPU_HEADER 0
#endif

#if AV2ENC_STILL_GPU_HEADER && defined(AV2ENC_HAVE_STILL_GPU) && AV2ENC_HAVE_STILL_GPU
#define AV2ENC_STILL_GPU_CALLABLE 1
#else
#define AV2ENC_STILL_GPU_CALLABLE 0
#endif

#if AV2ENC_VIDEO_GPU_HEADER && defined(AV2ENC_HAVE_VIDEO_GPU) && AV2ENC_HAVE_VIDEO_GPU
#define AV2ENC_VIDEO_GPU_CALLABLE 1
#else
#define AV2ENC_VIDEO_GPU_CALLABLE 0
#endif
