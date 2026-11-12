#pragma once

#include "core/bitstream/bitwriter.h"
#include "codec/transform/delta_q/dcdeltaq.h"
#include "codec/transform/qm/qm.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::rc {
struct SegmentationParams;
struct DeltaQParams;
}  // namespace av2enc::rc

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
  int quant_idx = 0;        // f(2), edge-classified form only
  int ext_filter = 0;       // f(3), edge-classified form only
  int edge_clf = 0;         // f(1), edge-classified form only
  int max_band_log2 = 3;    // f(2 + bo_only)
  // ccso_offset_idx values in syntax order: for bo_only the band-major table;
  // otherwise (d0, d1, band) with d0/d1 in [0, 3 - edge_clf).  The value is
  // Ccso_Offset[idx] * (scale_idx + 1).
  std::uint8_t offset_idx[128] = {};
};

struct StillCcsoConfig {
  bool enable = false;
  // ccso_unit_matches_sb_size = 1 -> CcsoLumaSizeLog2 = 6 for 64x64 SBs and 7
  // for a 128x128 frame superblock.
  StillCcsoPlane plane[3];
};

// Opt-in delta-Q / segmentation syntax (§5.18.7.1, §5.18.7.8).  The reference
// still path keeps one quantizer for the whole frame, so the signalled form is
// the neutral one: segment ids are all 0 (an all-disabled 8-segment table) and
// every superblock codes a zero reduced delta.  That exercises the syntax and
// the decoder state machine without changing the reconstruction.  Per-segment
// quantizer planning is a separate integration (it needs a per-SB decision and
// dequant path).
struct StillDeltaQConfig {
  bool segmentation_enabled = false;
  bool delta_q_present = false;
  int delta_q_res = 0;  // 0..3; must be 0 when delta_q_present is false
  // Optional non-neutral segmentation table (§5.4.9 seg_info): the per-segment
  // Q planner supplies SEG_LVL_ALT_Q values.  Null keeps the neutral table.
  // The pointed-to object must outlive the header/assembly call.
  const ::av2enc::rc::SegmentationParams* segmentation = nullptr;
};

// Builds the module parameter blocks that match `config`.  Returns false and
// sets `error` when the values cannot be represented.  Defined in
// still_headers.cpp so this header stays independent of the rc implementation.
bool make_delta_q_params(const StillDeltaQConfig& config,
                         ::av2enc::rc::SegmentationParams& segmentation,
                         ::av2enc::rc::DeltaQParams& delta_q,
                         std::string* error = nullptr);
// One plane of the §5.18.7.11 / §5.20.10.4-.6 Wiener-NS decision.  M1 has no
// inter frames, so temporal frame-filter prediction and bank reuse across
// frames are not represented.  The per-unit syntax that this stage writes is
// exactly what `wiener_write_unit()` (classes/wiener_syntax) emits: the
// decision vector is replayed through that writer in tile order.
struct StillLrPlane {
  bool enabled = false;
  // LoopRestorationSize shifts: size_luma = 512 >> shift,
  // size_chroma = (512 >> max(subX,subY)) >> shift.  M1 fixes both to 3.
  int shift = 3;
  int unit_rows = 0;
  int unit_cols = 0;
  // LrType[unit_rows][unit_cols]: 0 = RESTORE_NONE, 2 = WIENER_NONSEP.
  std::vector<std::uint8_t> unit_type;
  // LrWienerNs, stride 18 per unit.
  std::vector<std::int16_t> unit_coeffs;
  // Per-unit §5.20.10.6 syntax, one entry per unit.
  std::vector<std::uint8_t> unit_subset;
  std::vector<std::uint8_t> unit_uv_sym;
  std::vector<std::uint8_t> unit_merged;
  std::vector<std::int8_t> unit_bank_k;
};

// §5.4.10 enable_gdf / gdf_unit_matches_sb_size, §5.18.7.9 gdf_params().
// M1 signals no `use_gdf` symbols: gdf_per_block = 0, so every unit is
// filtered.  `cols`/`rows` describe the GdfBlks grid for diagnostics.
struct StillGdfConfig {
  bool enable = false;
  bool unit_matches_sb_size = true;
  bool per_block = false;
  int sb_size = 64;
  int pic_qc_idx = 0;
  int pic_scale_idx = 0;
  int blk_size = 64;
  int cols = 1;
  int rows = 1;
  int pix_scale = 1;
};

struct StillLrConfig {
  bool enable = false;
  // §5.4.10 lr_tools_uv_present.  Luma/chroma are controlled independently so
  // --still-chroma-lr can be tested without luma; the chroma planes share a
  // size and are enabled together.
  bool uv_present = false;
  // Luma uses RESTORE_PC_WIENER (§7.20.4) instead of non-separable Wiener NS.
  // plane[0] then carries LrType (RESTORE_PC_WIENER / RESTORE_NONE) and no
  // coefficients; chroma stays off.
  bool pc_wiener = false;
  StillLrPlane plane[3];
};

struct StillConfig {
  std::uint16_t max_frame_width = 64;
  std::uint16_t max_frame_height = 64;
  std::uint8_t seq_level_idx = 7;  // level 5.1
  std::uint8_t base_q_idx = 90;
  bool disable_cdf_update = true;
  bool enable_intra_edge_filter = true;
  bool enable_cfl_intra = true;
  // §5.4.5 sequence_intra_config() enable_ibp f(1).  IBP has no per-block
  // syntax; enabling it changes every DC and directional prediction of a
  // non-4x4 block in both the search and the reconstruction.
  bool enable_ibp = false;
  // §5.4.5 sequence_intra_config() enable_mrls f(1).
  bool enable_mrls = false;
  // §5.4.5 sequence_intra_config() enable_dip f(1).
  bool enable_dip = false;
  // §5.4.5 sequence_intra_config() enable_mhccp f(1).
  bool enable_mhccp = false;
  // §5.18.3.3 screen_content_params().  single_picture_header_flag infers
  // seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS, so the frame
  // codes allow_screen_content_tools and (when set) force_integer_mv.
  bool allow_screen_content_tools = false;
  bool force_integer_mv = true;
  // §5.20.8 palette_mode_info()/palette_tokens().  Requires screen tools.
  bool enable_palette = false;
  // §5.18.3.4 intrabc_params() + §5.20.5.3 use_intrabc.  `allow_intrabc`
  // requires screen tools; the M1 path signals the global reference region
  // (allow_global_intrabc = 1, allow_local_intrabc = 0), which needs no
  // four-buffer ring.  seq_max_bvp_drl_bits_minus_1 stays 0 and
  // allow_frame_max_bvp_drl_bits is 0, so no change_bvp_drl bits are coded.
  bool allow_intrabc = false;
  bool allow_global_intrabc = true;
  bool allow_local_intrabc = false;
  // §5.4.5 cfl_ds_filter_index f(2).  0 keeps the historical CfL filter.
  int cfl_ds_filter_index = 0;
  // Keeps chroma on DCT_DCT, so the DCT-only coefficient scan/context path is
  // used for every chroma block.
  bool enable_chroma_dctonly = true;
  // §5.4.8 enable_idtx_intra.  Required for intra TX types that use an identity
  // 1D transform (IDTX/V_DCT/H_DCT), which the long-side sets always list.
  bool enable_idtx_intra = false;
  // §5.4.8 enable_intra_ist / enable_inter_ist.  M1 only codes intra IST
  // (`sec_tx_type` on luma blocks); inter IST stays off.
  bool enable_intra_ist = false;
  bool enable_inter_ist = false;
  // §5.4.8 enable_cctx / enable_tcq / enable_parity_hiding.  For a
  // single-picture header choose_tcq_per_frame is inferred 0, so
  // enable_tcq=1 also forces enable_parity_hiding=0 and removes that bit from
  // the sequence header.  Both tools are off by default.
  bool enable_cctx = false;
  bool enable_tcq = false;
  // M1 starts with TX_MODE_LARGEST (one transform per coding block). Set true
  // for TX_MODE_SELECT once TX partition syntax is wired.
  bool tx_mode_select = false;
  // §5.20.2 reduced_tx_set (2 bits).  0 keeps the full intra TX-type sets.
  int reduced_tx_set = 0;
  // §5.4.8 reduced_tx_part_set f(1).  0 keeps the full TX-partition type set.
  bool reduced_tx_part_set = false;
  // §5.4.3 sequence_partition_config() superblock selection.  Opt-in and
  // mutually exclusive: use_256x256_superblock is present only when the frame
  // is 256-ruled, and then the use_128x128_superblock bit is absent.  An intra
  // frame is always coded with SbSize = 128x128 (§5.18.1), so both flags select
  // a 128x128 frame-superblock grid; use_256 only changes the sequence-level
  // size.  The M1 writer does not emit 256x256 coding blocks.
  bool use_128x128_superblock = false;
  bool use_256x256_superblock = false;
  // §5.4.3 extended-partition tools.  enable_uneven_4way_partitions is present
  // (and only read) when enable_ext_partitions is 1.  Default off keeps the
  // historical two zero bits.
  bool enable_ext_partitions = false;
  bool enable_uneven_4way_partitions = false;
  // §5.4.3 enable_sdp.  Present when !Monochrome; enable_extended_sdp is
  // inferred 0 because single_picture_header_flag = 1.  Opt-in; adds one bit.
  bool enable_sdp = false;
  // apply_deblocking_filter[0..3].
  bool apply_deblock_y_vert = true;
  bool apply_deblock_y_horz = true;
  bool apply_deblock_u = true;
  bool apply_deblock_v = true;
  // Opt-in CDEF (§5.4.10 enable_cdef + §5.18.7.10 cdef_params + read_cdef).
  StillCdefConfig cdef;
  // Opt-in CCSO (§5.4.10 enable_ccso + §5.18.7.12 ccso_params + read_ccso).
  StillCcsoConfig ccso;
  // Opt-in delta-Q / segmentation (neutral form; see StillDeltaQConfig).
  // Distinct from the DC/AC delta-Q fields below: this one signals the
  // per-superblock qindex / segmentation syntax.
  StillDeltaQConfig per_sb_delta_q;
  // Opt-in content interpretation OBU (§5.15) between the sequence header and
  // the frame.  The default parameters have every presence flag clear.
  bool content_interpretation = false;
  // Opt-in quantizer matrices (§5.18.6.2 setup_qm_params + §7.14.4).  The
  // default-off state writes the historical `using_qmatrix = 0` bit.  The
  // encoder only signals spec-defined default matrices (levels 0..14); no
  // quantizer_matrix_obu() is emitted, so no user QM data is present.
  qm::QmFrameState qm;
  // Complete OBU_QUANTIZATION_MATRIX carried before the coded frame when
  // user-defined QM data is present (§5.13, §7.3.8.9).  Empty keeps the
  // historical stream.
  std::vector<std::uint8_t> qm_obu;
  // Opt-in DC/AC delta-Q (§5.4.8 sequence fields + §5.18.6.1
  // quantization_params).  Defaults write exactly the historical bits:
  // equal_ac_dc_q=1, no separate UV deltas, all enables 0, all frame deltas 0.
  dcdeltaq::SequenceDeltaQ dc_ac_delta_q;
  dcdeltaq::FrameDeltaQ frame_delta_q;
  // Opt-in GDF (§7.20.5, applied in the loop-restoration stage after CCSO).
  StillGdfConfig gdf;
  // Opt-in film grain signalling.  The model OBU is prebuilt by the encoder
  // and inserted before the coded-frame OBU; the frame header codes
  // film_grain_config() (apply_grain is inferred 1 for a still picture).
  bool film_grain_present = false;
  int film_grain_id = 0;
  std::uint16_t grain_seed = 0;
  std::vector<std::uint8_t> film_grain_obu;
  // Opt-in non-separable Wiener loop restoration (§5.4.10 enable_restoration +
  // §5.18.7.11 lr_params + §5.20.10.4-.6 read_lr).  Applied after CCSO.
  StillLrConfig lr;
};

// Frame-level superblock size in pixels.  §5.18.1 clamps an intra frame to
// 128x128 even when the sequence header signals 256x256 superblocks.
inline int superblock_size_px(const StillConfig& config) {
  return (config.use_128x128_superblock || config.use_256x256_superblock) ? 128
                                                                          : 64;
}

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
