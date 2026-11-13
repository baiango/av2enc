// AV2 M2 P-frame tile syntax writer.
//
// Structure follows the M1 still-picture writer (partition tree, intra modes,
// transform-tree, EOB/coefficient coding) and adds the inter-frame block layer:
//
//   inter_frame_mode_info()   §5.20.7.1
//     read_is_inter()         §5.20.7.3
//     read_skip()             §5.20.5.10
//     inter_block_mode_info() §5.20.7.6
//       read_ref_frames()     §5.20.7.10
//       single_mode / drl_idx / read_mv (shell coding) §5.20.7.8, §5.20.7.20
//
// Intra blocks inside the P frame reuse the M1 intra mode and residual syntax.
// Inter blocks default to skip_flag = 1 (no residual).  With feature 1 enabled
// the writer emits the inter transform block: TX_MODE_LARGEST transform sizes,
// the mandatory inter luma TX_TYPE (DCT_DCT), and §8.3.2 inter coefficient
// contexts.
//
// The writer derives every entropy context from the block stream exactly as
// the decoder does. The producer supplies decisions and reconstruction data,
// not CDF state.
#pragma once

#include "pipeline/video/video_headers.h"
#include "pipeline/video/post/video_filter_plan.h"
#include "codec/entropy/avg_cdf/avg_cdf_bank.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <span>
#include <string>
#include <vector>

namespace av2enc::video::syntax {

inline constexpr int kPlaneCount = 3;

enum BlockFlags : std::uint8_t {
  kFlagForced = 1u << 0,
  kFlagIsCfl = 1u << 1,
  kFlagSbRoot = 1u << 2,
};

// Inter prediction mode as signalled through single_mode:
//   0 -> NEARMV (YMode 14), 1 -> NEWMV (16), 2 -> GLOBALMV (15)
enum InterMode : std::uint8_t {
  kInterGlobalMv = 0,
  kInterNearMv = 1,
  kInterNewMv = 2,
};

// §5.20.5.5 IntraJointMode value for a coded (YMode, AngleDeltaY): 0..4 for
// the reordered non-directional modes (DC, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH)
// and 5 + directional index otherwise.  The encoder needs it to derive the
// §7.13.2.15/.16 filterTypeAbove/Left of the following blocks; it is the same
// mapping the syntax writer tracks internally.
int intra_joint_mode(int y_mode, int angle_delta);

struct BlockSyntax {
  // --- tree ---
  std::uint16_t mi_row = 0;
  std::uint16_t mi_col = 0;
  std::uint8_t bsize = 0;
  std::uint8_t partition = 0;
  std::uint8_t allowed_mask = 0;
  std::uint8_t flags = 0;
  std::uint16_t subtree = 0;

  // --- leaf mode info ---
  std::uint8_t is_inter = 0;
  std::uint8_t skip_flag = 0;   // inter only; 1 means no residual
  std::uint8_t inter_mode = kInterGlobalMv;
  std::uint8_t drl_index = 0;
  // §5.18.2/§5.20.7.10 RefFrame[0] index into ref_frame_idx[] (0..6).
  std::uint8_t ref_index = 0;
  // Compound block (§5.20.7.10 comp_mode / §5.20.7.6).  `ref_index1` is
  // RefFrame[1]; `compound_joint` selects is_joint; `compound_mode` is the
  // compound_mode_non_joint symbol (YMode = NEAR_NEARMV + compound_mode).
  std::uint8_t compound = 0;
  std::uint8_t ref_index1 = 0;
  std::uint8_t compound_joint = 0;
  std::uint8_t compound_mode = 3;  // 3 = GLOBAL_GLOBALMV
  std::int32_t mv_y = 0;   // resolved MV, 1/8 luma samples
  std::int32_t mv_x = 0;
  std::int32_t mvd_y = 0;  // signalled MVD for NEWMV, 1/8 luma samples
  std::int32_t mvd_x = 0;

  std::uint8_t y_mode = 0;
  std::int8_t y_angle_delta = 0;
  std::uint8_t uv_mode = 0;

  // --- leaf residual ---
  std::uint8_t tx_size[kPlaneCount] = {0, 0, 0};
  std::uint32_t eob[kPlaneCount] = {0, 0, 0};
  std::uint32_t levels_offset = 0;

  // --- opt-in TX_MODE_SELECT (§5.20.6) luma plan -------------------------
  // `tx_partition` is a txpartition::TxPartition value for the coding block and
  // `tx_part_count` the number of transform leaves it produced (1 when there is
  // no partition syntax).  `tx_size[0]/eob[0]` still carry the first leaf so a
  // TX_MODE_LARGEST consumer keeps working; when tx_part_count != 0 the arrays
  // below describe every luma leaf in decode order and are authoritative.
  std::uint8_t tx_type[kPlaneCount] = {0, 0, 0};
  std::uint8_t tx_partition = 0;
  std::uint8_t tx_part_count = 0;
  std::uint8_t tx_part_tx_size[4] = {0, 0, 0, 0};
  std::uint8_t tx_part_tx_type[4] = {0, 0, 0, 0};
  std::int16_t tx_part_mi_row[4] = {0, 0, 0, 0};
  std::int16_t tx_part_mi_col[4] = {0, 0, 0, 0};
  std::uint32_t tx_part_eob[4] = {0, 0, 0, 0};
  std::uint32_t tx_part_levels_offset[4] = {0, 0, 0, 0};
};

struct FramePlan {
  int width = 0;
  int height = 0;
  int sb_size = 64;
  // false for the intra keyframe (intra_frame_mode_info), true for P frames
  // (inter_frame_mode_info, which codes is_inter/skip).
  bool inter_frame = true;
  // §5.18.2 per-frame reference map.  num_total_refs > 1 lists the explicit
  // map; reference_select enables per-block comp_mode (compound).
  int num_total_refs = 1;
  bool reference_select = false;
  // Frame MV precision: 0 = 1/8, 1 = 1/4, 2 = 1/2 pel.
  int mv_precision = 0;
  // FrameDistance[ref] for the MV stack's derived candidates.
  std::array<int, 7> frame_distance{0, 0, 0, 0, 0, 0, 0};
  // §5.18.9 global motion parameters for the MV-stack GlobalMvs.  Identity
  // models (the default) reproduce the historical zero GlobalMvs.  The writer
  // projects the centre MV per block with the frame precision.
  std::array<inter::gme::GmParams, 7> gm_params{};
  int gm_num_refs = 0;
  // 0 = DRL_REORDER_DISABLED, 1 = CONSTRAINT, 2 = ALWAYS.
  int drl_reorder = 0;
  // Opt-in §5.18.8.1 TX_MODE_SELECT (--video-tx-search).  Off keeps
  // tx_mode_select = 0 and TX_MODE_LARGEST, i.e. every historical stream is
  // byte-identical.  When on, every leaf's luma residual follows
  // write_tx_partition()/per-part transform blocks and the stored per-part
  // TX types.
  bool tx_mode_select = false;
  bool reduced_tx_part_set = false;
  // §5.18.2 frame-header reduced_tx_set (0 keeps the full TX-type sets).
  int reduced_tx_set = 0;

  int sb_cols() const;
  int sb_rows() const;
  int superblock_count() const;
};

// Emit the range-coded tile payload for one frame (key or inter). The caller
// wraps it in a CLK or regular tile-group OBU with av2enc::video::make_frame_obu().
bool assemble_video_frame(const FramePlan& plan,
                       std::span<const BlockSyntax> blocks,
                       std::span<const std::int32_t> levels,
                       const VideoConfig& config,
                       std::vector<std::uint8_t>& tile_payload,
                       std::string& error, std::size_t* tile_bytes = nullptr,
                       std::span<const std::uint16_t> cdf_seed = {},
                       std::vector<std::uint16_t>* cdf_final = nullptr,
                       const post::VideoFilterPlan* filters = nullptr);

// Opaque packed §8.2 CDF image (no neighbour/context state).  Used by the
// cross-frame continuity wiring in video_frame.cpp: the encoder seeds a frame
// from the primary reference's saved CDFs and exports the adapted tile CDFs so
// the average-CDF bank can run §8.2.4 / §7.5.
std::size_t cdf_image_words();
bool cdf_image_defaults_for_base_q_idx(int base_q_idx,
                                       std::vector<std::uint16_t>& out,
                                       std::string* error = nullptr);
bool cdf_image_tables(std::vector<std::uint16_t>& frame,
                      std::vector<std::uint16_t>& tile,
                      std::vector<std::uint16_t>& saved,
                      std::vector<std::uint16_t>& defaults,
                      std::vector<entropy::avg_cdf::CdfTable>& tables,
                      std::string* error = nullptr);

int y_mode_joint(int av2_mode, int angle_delta);
void build_scan(int w, int h, std::span<std::uint16_t> scan);

}  // namespace av2enc::video::syntax
