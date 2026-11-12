// CUDA batch dequantization with AV2 quantizer matrices (§7.14.4).
//
// The CPU reference is `src/encoder/qm/`.  Both call the same integer helpers
// in `src/encoder/qm/qm_core.h`; no floating point is involved, so the two
// paths are bit-exact.
//
// The launcher allocates nothing, frees nothing, and does not synchronize.
// Tables are caller-uploaded device pointers (see kQuantizerMatrixBytes and
// kUserQmBytes).  Long-lived buffers belong in the arena.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::qm {

// Normative §9.4.3 table: [15][2][3600] bytes.
inline constexpr std::size_t kQuantizerMatrixBytes = 15u * 2u * 3600u;
// [level][t][plane][i][j] = 15 * 3 * 3 * 8 * 8 bytes.
inline constexpr std::size_t kUserQmBytes = 15u * 3u * 3u * 8u * 8u;

struct Job {
  std::int32_t level_offset = 0;    // into Batch::levels
  std::int32_t dequant_offset = 0;  // into Batch::dequant
  std::int32_t tx_size = 0;         // 0..24
  std::int32_t plane = 0;           // 0 = Y, 1 = U, 2 = V
  std::int32_t tx_type = 0;         // < 9 uses a QM
  // SegQMLevel[plane][segment] for <=8x8 blocks.
  std::int32_t seg_qm_level = 15;
  // qm_y[0] / qm_u[0] / qm_v[0] for >8x8 blocks.
  std::int32_t qm_level = 15;
  std::int32_t dc_quant = 0;
  std::int32_t ac_quant = 0;
  std::int32_t bit_depth = 8;       // BitDepth for the §7.14.4 Clip3 bound
  std::int32_t using_qmatrix = 0;
  std::int32_t qm_data_present = 0;
  std::int32_t allow_tcq = 0;
  std::int32_t lossless = 0;
  std::int32_t use_fsc = 0;
};

struct Batch {
  const std::int32_t* levels = nullptr;
  std::int32_t* dequant = nullptr;
  const Job* jobs = nullptr;
  std::int32_t num_jobs = 0;

  // Device copies of the normative tables.  qtable is required; user_qm may
  // be null when no job selects user data.
  const std::uint8_t* qtable = nullptr;
  const std::uint8_t* user_qm = nullptr;

  // Optional per-job result: 0 = ok, -1 = rejected job.  When null, invalid
  // jobs are skipped silently.
  std::int32_t* status = nullptr;
};

// One thread per job.  Returns the launch error; per-job validation is
// reported through Batch::status.
cudaError_t dequantize_batch(const Batch& batch, cudaStream_t stream = 0);

}  // namespace av2enc::cuda::qm
