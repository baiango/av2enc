// CPU emulation of the CUDA CfL entry points, for running the CUDA API test on
// a host without nvcc. This is NOT the GPU path and proves nothing about the
// kernel; it exists so the test logic itself can be exercised locally. The
// real bit-exactness gate is build_cuda_test.sh on the sm_86 box, which links
// src/cuda/cfl/cfl.cu instead of this file.
//
// Every function forwards to the shared CPU core in ../cfl.h, which is also
// what the kernel's arithmetic is transcribed from.

#include "../../../cuda/cfl/cfl.h"

#include "../cfl.h"

namespace av2enc::cuda {
namespace {

void predict_one(const cfl::Block& block, int alpha_q8) {
  cfl::Scratch scratch;
  cfl::compute_luma(block, scratch);
  cfl::apply_prediction(block, alpha_q8, scratch);
}

int predict_derived_one(const cfl::Block& block, int* alpha_out) {
  cfl::Scratch scratch;
  cfl::compute_luma(block, scratch);
  const int alpha = cfl::derive_implicit_alpha(block, scratch);
  cfl::apply_prediction(block, alpha, scratch);
  if (alpha_out != nullptr) {
    *alpha_out = alpha;
  }
  return alpha;
}

}  // namespace

cudaError_t cfl_predict(const cfl::Block& block, int alpha_q8,
                        cudaStream_t /*stream*/) {
  if (block.luma == nullptr || block.chroma == nullptr) {
    return cudaErrorInvalidValue;
  }
  predict_one(block, alpha_q8);
  return cudaSuccess;
}

cudaError_t cfl_predict_derived(const cfl::Block& block, int* d_alpha_out,
                                cudaStream_t /*stream*/) {
  if (block.luma == nullptr || block.chroma == nullptr) {
    return cudaErrorInvalidValue;
  }
  predict_derived_one(block, d_alpha_out);
  return cudaSuccess;
}

cudaError_t cfl_predict_batch(const CflJob* d_jobs, int job_count,
                              cudaStream_t /*stream*/) {
  if (job_count <= 0) {
    return cudaSuccess;
  }
  if (d_jobs == nullptr ||
      !av2enc_cuda_stub::is_device_pointer(d_jobs)) {
    // Matches the real contract: d_jobs must be device memory. This check
    // exists only in the emulation and catches a host array being passed.
    return cudaErrorInvalidValue;
  }
  for (int i = 0; i < job_count; ++i) {
    predict_one(d_jobs[i].block, d_jobs[i].alpha_q8);
  }
  return cudaSuccess;
}

cudaError_t cfl_predict_derived_batch(const cfl::Block* d_blocks,
                                      int* d_alpha_out, int job_count,
                                      cudaStream_t /*stream*/) {
  if (job_count <= 0) {
    return cudaSuccess;
  }
  if (d_blocks == nullptr ||
      !av2enc_cuda_stub::is_device_pointer(d_blocks)) {
    return cudaErrorInvalidValue;
  }
  for (int i = 0; i < job_count; ++i) {
    predict_derived_one(d_blocks[i],
                        d_alpha_out == nullptr ? nullptr : d_alpha_out + i);
  }
  return cudaSuccess;
}

}  // namespace av2enc::cuda
