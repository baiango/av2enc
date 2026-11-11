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

#include "encoder/m2/m2_headers.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::m2::syntax {

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
};

struct FramePlan {
  int width = 0;
  int height = 0;
  int sb_size = 64;
  // false for the intra keyframe (intra_frame_mode_info), true for P frames
  // (inter_frame_mode_info, which codes is_inter/skip).
  bool inter_frame = true;

  int sb_cols() const;
  int sb_rows() const;
  int superblock_count() const;
};

// Emit the range-coded tile payload for one frame (key or inter). The caller
// wraps it in a CLK or regular tile-group OBU with m2::make_frame_obu().
bool assemble_m2_frame(const FramePlan& plan,
                       std::span<const BlockSyntax> blocks,
                       std::span<const std::int32_t> levels,
                       const M2Config& config,
                       std::vector<std::uint8_t>& tile_payload,
                       std::string& error, std::size_t* tile_bytes = nullptr);

int y_mode_joint(int av2_mode, int angle_delta);
void build_scan(int w, int h, std::span<std::uint16_t> scan);

}  // namespace av2enc::m2::syntax
