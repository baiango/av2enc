// AV2 v1.0.0 §5.4.6 / §5.18.2 frame `allow_warpmv_mode` bit and constraints.
//
// Normative behaviour (spec p102/p214-215):
//
//   sequence_inter_config():
//     for (mode = INTERINTRA; mode < MOTION_MODES; mode++)
//         seq_enabled_motion_modes[mode] f(1)
//     if (any) seq_frame_motion_modes_present_flag f(1) else 0
//     if (seq_enabled_motion_modes[DELTAWARP]) enable_six_param_warp_delta f(1)
//
//   frame_header_info():
//     for (mode = INTERINTRA; mode < MOTION_MODES; mode++) {
//       if (!seq_frame_motion_modes_present_flag)
//           frame_enabled_motion_modes[mode] = seq_enabled_motion_modes[mode]
//       else if (seq_enabled_motion_modes[mode])
//           frame_enabled_motion_modes[mode] f(1)
//       else frame_enabled_motion_modes[mode] = 0
//     }
//     ...
//     if (!FrameIsIntra && frame_enabled_motion_modes[DELTAWARP])
//         allow_warpmv_mode f(1)
//     else allow_warpmv_mode = 0
//
//   inter_block_mode_info():
//     if (allow_warpmv_mode && Min(Block_Width, Block_Height) >= 8) is_warp S()
//     if (is_warp) { if (force_integer_mv) warp_mv = 1; else warp_mv S();
//                    YMode = warp_mv ? WARPMV : WARP_NEWMV; }
//
// The constraints this module enforces are exactly the ones the syntax makes
// reachable:
//   * the frame bit is absent (value 0) for an intra frame or when DELTAWARP is
//     not frame-enabled;
//   * `frame_enabled_motion_modes[mode] <= seq_enabled_motion_modes[mode]`;
//   * `is_warp` is signalled only when the frame bit is 1 and min(w,h) >= 8;
//   * `warp_mv` is signalled only when is_warp is 1 and force_integer_mv is 0,
//     and is inferred 1 when force_integer_mv is 1;
//   * YMode WARPMV / WARP_NEWMV require is_warp == 1 (and therefore
//     allow_warpmv_mode == 1); WARPMV implies motion_mode DELTAWARP.
#pragma once

#include <cstdint>
#include <string>

#include "bitstream/bitwriter.h"
#include "encoder/inter/mv_stack_full/bit_reader.h"
#include "encoder/inter/mv_stack_full/mv_stack_full.h"

namespace av2enc::inter::mv_stack_full {

enum class FrameStatus : std::int32_t {
  kOk = 0,
  kInvalidArgument,
  kConstraintViolated,
};

const char* frame_status_string(FrameStatus status);

// §5.4.6 sequence_inter_config() motion-mode block.
struct SequenceMotionModes {
  bool seq_enabled_motion_modes[kMotionModes] = {false, false, false, false,
                                                 false};
  bool seq_frame_motion_modes_present_flag = false;
  bool enable_six_param_warp_delta = false;
};

// §5.18.2 frame-level override.
struct FrameMotionModes {
  bool frame_is_intra = false;
  bool frame_enabled_motion_modes[kMotionModes] = {false, false, false, false,
                                                   false};
};

FrameStatus derive_frame_enabled_motion_modes(
    const SequenceMotionModes& seq, const bool frame_override[kMotionModes],
    FrameMotionModes* out, std::string* why = nullptr);

struct WarpmvMode {
  bool present = false;  // the f(1) syntax element is in the frame header
  bool allow_warpmv_mode = false;
};

FrameStatus derive_allow_warpmv_mode(const SequenceMotionModes& seq,
                                     const FrameMotionModes& frame,
                                     WarpmvMode* out,
                                     std::string* why = nullptr);

// Block-level gates implied by allow_warpmv_mode.
struct WarpBlockGates {
  bool is_warp_present = false;
  bool warp_mv_present = false;
  bool warp_mv_inferred = false;  // force_integer_mv && is_warp
};

FrameStatus derive_warp_block_gates(bool allow_warpmv_mode,
                                    bool force_integer_mv, int block_width,
                                    int block_height, bool is_warp,
                                    WarpBlockGates* out,
                                    std::string* why = nullptr);

// Checks the emitted/decoded block against the bit's constraints.
FrameStatus validate_warp_block(bool allow_warpmv_mode, bool force_integer_mv,
                                int block_width, int block_height, bool is_warp,
                                bool warp_mv, int y_mode, int motion_mode,
                                std::string* why = nullptr);

// Bit-level writer/reader.  `write_allow_warpmv_mode` writes the bit iff `out`
// says it is present and returns false on a self-inconsistent request.
bool write_allow_warpmv_mode(BitWriter* writer, const SequenceMotionModes& seq,
                             const FrameMotionModes& frame,
                             WarpmvMode* out, std::string* error);
bool read_allow_warpmv_mode(BitReader* reader, const SequenceMotionModes& seq,
                            const FrameMotionModes& frame, WarpmvMode* out,
                            std::string* error);

// §5.4.6 writer: the four motion-mode bits, the presence flag (only when any
// mode is enabled) and enable_six_param_warp_delta (only when DELTAWARP is
// enabled).
bool write_sequence_motion_modes(BitWriter* writer,
                                 const SequenceMotionModes& config,
                                 std::string* error);
bool read_sequence_motion_modes(BitReader* reader,
                                SequenceMotionModes* config,
                                std::string* error);

// §5.18.2 writer: one f(1) per sequence-enabled mode when the sequence says
// the frame may override them.
bool write_frame_motion_modes(BitWriter* writer,
                              const SequenceMotionModes& seq,
                              const FrameMotionModes& frame,
                              std::string* error);
bool read_frame_motion_modes(BitReader* reader,
                             const SequenceMotionModes& seq,
                             FrameMotionModes* frame, std::string* error);

}  // namespace av2enc::inter::mv_stack_full
