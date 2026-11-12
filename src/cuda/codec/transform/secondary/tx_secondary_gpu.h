// CUDA AV2 secondary transform (IST) batch API.
//
// Pure device-pointer + stream interface: no allocation, no synchronization.
// The buffer stride cannot be implicit because AV2 transform blocks have
// different element counts, so every job carries explicit in/out offsets.
//
// The caller owns and uploads the job array.  Jobs may be processed in any
// order; each CUDA thread owns one complete transform block.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace av2enc::cuda {

struct TxSecondaryJob {
  int tx_size = 0;        // AV2 TX_SIZE 0..24
  int plane_tx_type = 0;  // TxType value; IST uses DCT_DCT (0) or ADST_ADST (3)
  int kernel = 0;         // resolved §7.15.3 kernel index
  int sec_tx_type = 0;    // 1..3
  int transpose = 0;      // §7.15.3 transpose flag
  int bit_depth = 8;      // 8..12
  int in_offset = 0;      // element offset into the input buffer
  int out_offset = 0;     // element offset into the output buffer
};

// dequant (int32, w*h per job) -> dequant (int32, in-place or out-of-place).
cudaError_t txsec_inverse_dequant(const std::int32_t* d_in,
                                  std::int32_t* d_out,
                                  const TxSecondaryJob* d_jobs, int njobs,
                                  cudaStream_t stream = nullptr);

// Encoder-side analysis projection (not normative); same layout as inverse.
cudaError_t txsec_forward_coeffs(const std::int32_t* d_in,
                                 std::int32_t* d_out,
                                 const TxSecondaryJob* d_jobs, int njobs,
                                 cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda
