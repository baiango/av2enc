#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::m1 {

struct StillPicture {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::span<const std::uint8_t> y;
  std::span<const std::uint8_t> u;
  std::span<const std::uint8_t> v;
};

struct M1Options {
  std::uint8_t base_q_idx = 90;
  bool enable_cfl = true;
  bool enable_deblock = true;
  bool enable_intra_edge_filter = true;
  bool adaptive_cdf = false;
  // Partition search floor. 8 is the M1 target; 4 exercises the 4x4 path.
  int min_block_size = 8;
  int max_partition_depth = 4;
  // Debug/per-tool gates: one PARTITION_NONE superblock and DC intra only.
  bool force_single_block = false;
  bool force_dc = false;
  // Full directional/smooth intra is bit-exact with the decoder. It stays
  // opt-in because the mode search is SSE-only (no rate term yet), which does
  // not reliably beat DC at fixed QP.
  bool luma_dc_only = true;
};

struct M1Result {
  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;
  std::vector<std::uint8_t> frame;
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  int leaves = 0;
};

bool encode_m1_still_picture(const StillPicture& picture, const M1Options& options,
                             M1Result& result, std::string& error);

}  // namespace av2enc::m1
