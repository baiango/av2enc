// AV2 v1.0.0 multiple reference line selection (MRL) — CPU reference.
//
// Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
//   §7.13.2.1   AboveRow/LeftCol preparation with MrlIndex > 0
//   §7.13.2.7   directional intra prediction, Mrl_Index_To_Delta, secondary line
//   §7.13.2.8   single directional prediction with mrlIndex
//
// Scope: luma directional prediction with mrl_index 0..3 and the optional
// secondary-line blend (mrl_sec_index). MRL is luma-only; chroma forces
// MrlIndex = 0 (§7.13.2.1). Non-directional modes cannot signal MRL.
//
// All arithmetic is integer and follows the spec exactly. The CUDA predictor
// in src/cuda/mrl/ executes the same device header against the same tests.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "encoder/intra_modes/intra_modes.h"

namespace av2enc::mrl {

using intra::Mode;
using intra::PlaneView;

inline constexpr int kMaxMrlIndex = 3;
inline constexpr int kMaxBlockDim = intra::kMaxBlockDim;
// MrlIndex > 0 needs indices down to -1-MrlIndex and up to
// w+h-1+2*MrlIndex+2 (after the §7.13.2.8 extension).
inline constexpr int kEdgeMinIndex = -2 - kMaxMrlIndex;
inline constexpr int kEdgeMaxIndex = 2 * kMaxBlockDim + 2 * kMaxMrlIndex + 2;
inline constexpr int kEdgeStorage = kEdgeMaxIndex - kEdgeMinIndex + 1;

// Mrl_Index_To_Delta[4] = { 0, 1, -1, 0 }.
constexpr int mrl_angle_delta(int mrl_index) {
  return mrl_index == 1 ? 1 : (mrl_index == 2 ? -1 : 0);
}

bool mode_allows_mrl(Mode mode) noexcept;

enum class MrlStatus : std::uint8_t {
  kOk = 0,
  kInvalidArgument = 1,
};

const char* status_string(MrlStatus status) noexcept;

// One prepared reference line. Logical indices may be negative.
class EdgeLine {
 public:
  std::int16_t& operator[](int index) noexcept {
    return samples_[static_cast<std::size_t>(index - kEdgeMinIndex)];
  }
  std::int16_t operator[](int index) const noexcept {
    return samples_[static_cast<std::size_t>(index - kEdgeMinIndex)];
  }
  void clear() noexcept { samples_.fill(0); }

 private:
  std::array<std::int16_t, kEdgeStorage> samples_{};
};

struct Block {
  PlaneView recon;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int max_x = 0;  // inclusive visible bound
  int max_y = 0;
  bool have_left = false;
  bool have_above = false;
  int num4_above_right = 0;
  int num4_below_left = 0;
  int plane = 0;       // 0 = luma (MRL applies), >0 = chroma (forced 0)
  int sb_height = 64;  // Block_Height[SbSize], luma samples
};

struct Edges {
  EdgeLine above;      // AboveRow
  EdgeLine left;       // LeftCol
  EdgeLine above_sec;  // AboveSecRow
  EdgeLine left_sec;   // LeftSecCol
  int w = 0;
  int h = 0;
  int bit_depth = 8;
  int mrl_index = 0;       // effective MrlIndex (0 for chroma)
  int above_mrl_index = 0; // sbBoundary ? 0 : mrl_index
  bool have_left = false;
  bool have_above = false;
  bool is_luma = true;
  int valid_above_px = 0;
  int valid_left_px = 0;
};

// §7.13.2.1 with MrlIndex = (plane == 0) ? mrl_index : 0.
MrlStatus prepare_edges(const Block& block, int bit_depth, int mrl_index,
                        Edges& edges) noexcept;

struct Params {
  Mode mode = Mode::kD45;
  int angle_delta = 0;  // -3..3
  int mrl_index = 0;    // 0..3; forced 0 for chroma inside prediction
  bool mrl_sec_index = false;
  bool enable_edge_filter = true;
  bool above_neighbor_is_smooth = false;
  bool left_neighbor_is_smooth = false;
};

// Validates mode/index availability. `mrl_sec_index` requires mrl_index > 0
// and a block that is not 4x4 (the decoder ignores it at 4x4).
MrlStatus validate_params(const Block& block, const Params& params) noexcept;

// Writes the h x w prediction. `edges` must have been produced by
// prepare_edges() with the same mrl_index (chroma forces 0).
MrlStatus predict(const Edges& edges, const Params& params,
                  std::uint8_t* dst, std::size_t dst_stride) noexcept;

// Convenience wrapper: prepare_edges() + predict().
MrlStatus predict_block(const Block& block, int bit_depth,
                        const Params& params, std::uint8_t* dst,
                        std::size_t dst_stride) noexcept;

}  // namespace av2enc::mrl
