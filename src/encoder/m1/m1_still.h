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
  // Enable the full chroma intra search (directional/smooth/Paeth) instead of
  // DC/CfL only. Off by default until forced-UV decoder verification passes on
  // the M1 still corpus. CLI: --m1-uv-full (to be wired by the orchestrator).
  bool uv_full = false;
  // Rate-aware luma intra mode decision (`src/encoder/rdo/intra_decision.*`).
  // Off by default: the lambda scale is a documented calibration placeholder.
  bool rdo_intra = false;
  double rdo_sse_scale = 0.125;
  // Full-trellis coefficient RDOQ (`src/encoder/rdoq/`).  Same syntax, only the
  // quantised levels change.  Off by default; sse_scale is a calibration
  // placeholder (see the RDOQ README's lambda table).
  bool rdoq = false;
  double rdoq_sse_scale = 0.125;
  // Intra luma TX_TYPE search (`src/encoder/tx_type/`).  Chroma stays DCT_DCT
  // (enable_chroma_dctonly).  Off by default.
  bool tx_type = false;
  double tx_type_sse_scale = 0.125;
  // Opt-in CDEF (§7.18).  The strength search runs after deblock and its
  // filtered planes replace the reconstruction/output.  Default off so the
  // pinned `--m1` MD5s do not change.
  bool enable_cdef = false;
  // CdefStrengths: 1 forces one global set (no index symbols); 2..8 signal a
  // per-64x64 index.  4 is the search default ladder size.
  int cdef_num_strengths = 4;
  int cdef_damping = 3;
  bool cdef_on_skip_txfm = false;
  // Opt-in CCSO band-offset-only (§7.19, post-CDEF).  `ccso_bands_log2` sets
  // ccso_max_band_log2 (0..6; 3 = 8 bands).
  bool enable_ccso = false;
  int ccso_bands_log2 = 6;
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
