#pragma once

#include "core/bitstream/bitwriter.h"

#include <cstdint>
#include <span>
#include <vector>

// AV2 v1.0.0 OBU/header writer for the integrated M1 still picture.
//
// Optional tools are represented as plain-POD config blocks so this header
// stays independent of the component implementations (and of CUDA).  A tool
// that is disabled writes the same bits as before; enabling one changes the
// sequence/frame header and, for CDEF/restoration, the tile syntax.
//
// The M0 writer in bitstream/basic_headers.* is intentionally left untouched. M1
// needs a different sequence-header tool set (intra edge filter, CfL, chroma
// DCT-only transforms) and configurable deblocking flags, so the few extra
// bits live here rather than changing the pinned M0 bytes.
namespace av2enc::still {

// §5.18.7.10 cdef_params().  `strengths[i]` holds decoded-domain values:
// y_pri/uv_pri in 0..15, y_sec/uv_sec in {0,1,2,4}.  The writer derives the
// pri_zero shortcut and the 4 -> 3 secondary syntax mapping.
struct StillCdefStrength {
  int y_pri = 0;
  int y_sec = 0;
  int uv_pri = 0;
  int uv_sec = 0;
};

struct StillCdefConfig {
  bool enable = false;
  int damping = 3;         // CdefDamping, 3..6
  int num_strengths = 1;   // CdefStrengths, 1..8
  bool on_skip_txfm = false;
  StillCdefStrength strengths[8] = {};
};

// §5.18.7.12 ccso_params() for one plane.  M1 only signals band-offset-only
// CCSO (`bo_only`), which keeps the header and the offset table small.
struct StillCcsoPlane {
  bool enabled = false;
  bool bo_only = true;
  int scale_idx = 0;        // f(2)
  int max_band_log2 = 3;    // f(2 + bo_only)
  // ccso_offset_idx values in syntax order (band 0..(1<<max_band_log2)-1 for
  // bo_only).  The value is Ccso_Offset[idx] * (scale_idx + 1).
  std::uint8_t offset_idx[128] = {};
};

struct StillCcsoConfig {
  bool enable = false;
  // ccso_unit_matches_sb_size = 1 -> CcsoLumaSizeLog2 = 6 for 64x64 SBs.
  StillCcsoPlane plane[3];
};

struct StillConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // level 5.1
  std::uint8_t base_q_idx = 90;
  bool disable_cdf_update = true;
  bool enable_intra_edge_filter = true;
  bool enable_cfl_intra = true;
  // Keeps chroma on DCT_DCT, so the DCT-only coefficient scan/context path is
  // used for every chroma block.
  bool enable_chroma_dctonly = true;
  // §5.4.8 enable_idtx_intra.  Required for intra TX types that use an identity
  // 1D transform (IDTX/V_DCT/H_DCT), which the long-side sets always list.
  bool enable_idtx_intra = false;
  // M1 starts with TX_MODE_LARGEST (one transform per coding block). Set true
  // for TX_MODE_SELECT once TX partition syntax is wired.
  bool tx_mode_select = false;
  // §5.20.2 reduced_tx_set (2 bits).  0 keeps the full intra TX-type sets.
  int reduced_tx_set = 0;
  // apply_deblocking_filter[0..3].
  bool apply_deblock_y_vert = true;
  bool apply_deblock_y_horz = true;
  bool apply_deblock_u = true;
  bool apply_deblock_v = true;
  // Opt-in CDEF (§5.4.10 enable_cdef + §5.18.7.10 cdef_params + read_cdef).
  StillCdefConfig cdef;
  // Opt-in CCSO (§5.4.10 enable_ccso + §5.18.7.12 ccso_params + read_ccso).
  StillCcsoConfig ccso;
};

std::vector<std::uint8_t> make_sequence_header_obu(const StillConfig& config);

// Writes frame_header(isFirst = 1) -> frame_header_info() (§5.18.1-§5.18.2),
// without the leading is_first_tile_group bit and without byte_alignment().
// Returns the number of syntax bits written.
//
// Deblocking planes follow deblocking_filter_params() (§5.18.5.2): after
// apply_deblocking_filter[0..3], each enabled plane writes one
// df_delta_q_present f(1) before tx_mode_select. M1 always sends delta 0, so
// an enabled plane contributes exactly one zero bit. When deblocking is off no
// delta bit is present.
std::size_t write_frame_header(BitWriter& writer, const StillConfig& config);

std::vector<std::uint8_t> assemble_still_picture(
    const StillConfig& config, std::span<const std::uint8_t> tile_payload);

}  // namespace av2enc::still
