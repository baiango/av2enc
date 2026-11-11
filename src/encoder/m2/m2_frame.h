// M2 two-frame P-frame integration on top of the M1 still encoder.
//
// Frame 0 is an intra keyframe. The M2-owned copy of the M1 intra syntax
// writer emits it (the M1 still integration TU was not linkable in this tree
// snapshot); the M1 tables, partition solver, intra prediction, transform and
// entropy coder are reused unchanged. Frame 1..N-1 are P frames that reference
// slot 0 and refresh slot 0.
//
// The P-frame decision path is deliberately small:
//   * partition tree from src/encoder/partition (same rules as M1)
//   * per-leaf inter candidates: GLOBALMV (zero MV), NEARMV (spatial MV stack)
//     and NEWMV (src/encoder/inter/me search quantised to 1/8 pel)
//   * per-leaf intra fallback: DC intra + DCT residual (M1 transform path)
//   * inter leaves use skip_flag = 1 (no residual in this milestone)
//
// This is a CPU correctness reference. The CUDA M2 tree is intentionally
// separate and not required for this gate.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::m2 {

struct M2InputFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::span<const std::uint8_t> y;
  std::span<const std::uint8_t> u;
  std::span<const std::uint8_t> v;
};

struct M2ReconFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

struct M2Options {
  std::uint8_t base_q_idx = 90;
  bool enable_intra_fallback = true;
  bool enable_newmv = true;
  bool enable_nearmv = true;
  bool force_global_mv = false;
  bool force_intra = false;
  bool force_single_block = false;
  int min_block_size = 16;
  int max_partition_depth = 4;
  // ME integer search radius in luma samples.
  int me_range = 16;
  // Debug trace to stderr.
  bool trace = false;
};

struct M2SequenceResult {
  // One temporal unit per input frame (TD + [SEQ] + frame OBU).
  std::vector<std::vector<std::uint8_t>> units;
  std::vector<M2ReconFrame> recon;
  std::vector<std::uint32_t> frame_bytes;
  std::vector<std::uint32_t> tile_bytes;
  std::vector<std::uint32_t> inter_blocks;
  std::vector<std::uint32_t> intra_blocks;
};

bool encode_m2_sequence(std::span<const M2InputFrame> frames,
                        const M2Options& options, M2SequenceResult& result,
                        std::string& error);

}  // namespace av2enc::m2
