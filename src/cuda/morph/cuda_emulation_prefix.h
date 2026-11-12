// Host-only CUDA keyword shim for the local emulation build.
//
// Included at the top of the emulation translation units (never by the real
// CUDA build). It lets morph_device.cuh compile as ordinary C++ and run one
// thread, exactly as the CPU emulation test expects.

#ifndef AV2ENC_CUDA_MORPH_CUDA_EMULATION_PREFIX_H_
#define AV2ENC_CUDA_MORPH_CUDA_EMULATION_PREFIX_H_

#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static

struct MorphEmuDim3 {
  unsigned x = 0;
  unsigned y = 0;
  unsigned z = 0;
};

static MorphEmuDim3 threadIdx;
static MorphEmuDim3 blockDim;

static inline void __syncthreads() {}

#endif  // AV2ENC_CUDA_MORPH_CUDA_EMULATION_PREFIX_H_
