#include "encoder/inter/mv_stack_full/frame_flags.h"

#include <algorithm>
#include <string>

namespace av2enc::inter::mv_stack_full {

const char* frame_status_string(FrameStatus status) {
  switch (status) {
    case FrameStatus::kOk:
      return "ok";
    case FrameStatus::kInvalidArgument:
      return "invalid argument";
    case FrameStatus::kConstraintViolated:
      return "bitstream conformance constraint violated";
  }
  return "unknown";
}

namespace {

FrameStatus fail(FrameStatus status, std::string* why, const char* message) {
  if (why != nullptr) *why = message;
  return status;
}

}  // namespace

FrameStatus derive_frame_enabled_motion_modes(
    const SequenceMotionModes& seq, const bool frame_override[kMotionModes],
    FrameMotionModes* out, std::string* why) {
  if (out == nullptr) {
    return fail(FrameStatus::kInvalidArgument, why, "null output");
  }
  FrameMotionModes result;
  for (int mode = kMotionInterIntra; mode < kMotionModes; ++mode) {
    if (!seq.seq_enabled_motion_modes[mode]) {
      if (seq.seq_frame_motion_modes_present_flag &&
          frame_override != nullptr && frame_override[mode]) {
        return fail(FrameStatus::kConstraintViolated, why,
                    "frame enables a motion mode the sequence disabled");
      }
      result.frame_enabled_motion_modes[mode] = false;
    } else if (!seq.seq_frame_motion_modes_present_flag) {
      result.frame_enabled_motion_modes[mode] = true;
    } else {
      if (frame_override == nullptr) {
        return fail(FrameStatus::kInvalidArgument, why,
                    "sequence allows frame override but no bits supplied");
      }
      result.frame_enabled_motion_modes[mode] = frame_override[mode];
    }
  }
  *out = result;
  return FrameStatus::kOk;
}

FrameStatus derive_allow_warpmv_mode(const SequenceMotionModes& seq,
                                     const FrameMotionModes& frame,
                                     WarpmvMode* out, std::string* why) {
  if (out == nullptr) {
    return fail(FrameStatus::kInvalidArgument, why, "null output");
  }
  WarpmvMode result;
  const bool delta_enabled =
      frame.frame_enabled_motion_modes[kMotionDeltaWarp];
  result.present = !frame.frame_is_intra && delta_enabled;
  result.allow_warpmv_mode = false;
  if (result.present &&
      !seq.seq_enabled_motion_modes[kMotionDeltaWarp]) {
    return fail(FrameStatus::kConstraintViolated, why,
                "DELTAWARP frame-enabled without sequence enable");
  }
  *out = result;
  return FrameStatus::kOk;
}

FrameStatus derive_warp_block_gates(bool allow_warpmv_mode,
                                    bool force_integer_mv, int block_width,
                                    int block_height, bool is_warp,
                                    WarpBlockGates* out, std::string* why) {
  if (out == nullptr) {
    return fail(FrameStatus::kInvalidArgument, why, "null output");
  }
  WarpBlockGates result;
  result.is_warp_present =
      allow_warpmv_mode && std::min(block_width, block_height) >= 8;
  if (is_warp && !result.is_warp_present) {
    return fail(FrameStatus::kConstraintViolated, why,
                "is_warp set where the syntax cannot signal it");
  }
  result.warp_mv_present = is_warp && !force_integer_mv;
  result.warp_mv_inferred = is_warp && force_integer_mv;
  *out = result;
  return FrameStatus::kOk;
}

FrameStatus validate_warp_block(bool allow_warpmv_mode, bool force_integer_mv,
                                int block_width, int block_height, bool is_warp,
                                bool warp_mv, int y_mode, int motion_mode,
                                std::string* why) {
  WarpBlockGates gates;
  const FrameStatus status =
      derive_warp_block_gates(allow_warpmv_mode, force_integer_mv, block_width,
                              block_height, is_warp, &gates, why);
  if (status != FrameStatus::kOk) return status;

  if (force_integer_mv && is_warp && !warp_mv) {
    return fail(FrameStatus::kConstraintViolated, why,
                "force_integer_mv requires warp_mv == 1");
  }
  if (!is_warp) {
    if (y_mode == kYModeWarpMv || y_mode == kYModeWarpNewMv) {
      return fail(FrameStatus::kConstraintViolated, why,
                  "warp YMode without is_warp");
    }
    return FrameStatus::kOk;
  }
  const int expected = warp_mv ? kYModeWarpMv : kYModeWarpNewMv;
  if (y_mode != expected) {
    return fail(FrameStatus::kConstraintViolated, why,
                "YMode does not match warp_mv");
  }
  if (y_mode == kYModeWarpMv && motion_mode != kMotionDeltaWarp) {
    return fail(FrameStatus::kConstraintViolated, why,
                "WARPMV must resolve to DELTAWARP");
  }
  if (y_mode == kYModeWarpNewMv && motion_mode != kMotionExtendWarp &&
      motion_mode != kMotionLocalWarp && motion_mode != kMotionDeltaWarp) {
    return fail(FrameStatus::kConstraintViolated, why,
                "WARP_NEWMV must resolve to an extended/local/delta warp mode");
  }
  return FrameStatus::kOk;
}

bool write_sequence_motion_modes(BitWriter* writer,
                                 const SequenceMotionModes& config,
                                 std::string* error) {
  if (writer == nullptr) {
    if (error != nullptr) *error = "null BitWriter";
    return false;
  }
  bool any = false;
  for (int mode = kMotionInterIntra; mode < kMotionModes; ++mode) {
    writer->put_bit(config.seq_enabled_motion_modes[mode]);
    any = any || config.seq_enabled_motion_modes[mode];
  }
  if (any) {
    writer->put_bit(config.seq_frame_motion_modes_present_flag);
  } else if (config.seq_frame_motion_modes_present_flag) {
    if (error != nullptr) {
      *error = "seq_frame_motion_modes_present_flag without enabled modes";
    }
    return false;
  }
  if (config.seq_enabled_motion_modes[kMotionDeltaWarp]) {
    writer->put_bit(config.enable_six_param_warp_delta);
  } else if (config.enable_six_param_warp_delta) {
    if (error != nullptr) {
      *error = "enable_six_param_warp_delta without DELTAWARP";
    }
    return false;
  }
  return true;
}

bool read_sequence_motion_modes(BitReader* reader, SequenceMotionModes* config,
                                std::string* error) {
  if (reader == nullptr || config == nullptr) {
    if (error != nullptr) *error = "null argument";
    return false;
  }
  SequenceMotionModes out;
  bool any = false;
  for (int mode = kMotionInterIntra; mode < kMotionModes; ++mode) {
    bool bit = false;
    if (!reader->read_bit(&bit)) {
      if (error != nullptr) *error = "truncated motion-mode bits";
      return false;
    }
    out.seq_enabled_motion_modes[mode] = bit;
    any = any || bit;
  }
  if (any) {
    bool bit = false;
    if (!reader->read_bit(&bit)) {
      if (error != nullptr) *error = "truncated motion-mode presence flag";
      return false;
    }
    out.seq_frame_motion_modes_present_flag = bit;
  }
  if (out.seq_enabled_motion_modes[kMotionDeltaWarp]) {
    bool bit = false;
    if (!reader->read_bit(&bit)) {
      if (error != nullptr) *error = "truncated six-param flag";
      return false;
    }
    out.enable_six_param_warp_delta = bit;
  }
  *config = out;
  return true;
}

bool write_frame_motion_modes(BitWriter* writer,
                              const SequenceMotionModes& seq,
                              const FrameMotionModes& frame,
                              std::string* error) {
  if (writer == nullptr) {
    if (error != nullptr) *error = "null BitWriter";
    return false;
  }
  if (!seq.seq_frame_motion_modes_present_flag) {
    FrameMotionModes derived;
    if (derive_frame_enabled_motion_modes(seq, nullptr, &derived, error) !=
        FrameStatus::kOk) {
      return false;
    }
    for (int mode = kMotionInterIntra; mode < kMotionModes; ++mode) {
      if (derived.frame_enabled_motion_modes[mode] !=
          frame.frame_enabled_motion_modes[mode]) {
        if (error != nullptr) {
          *error = "frame motion modes disagree with the sequence";
        }
        return false;
      }
    }
    return true;
  }
  for (int mode = kMotionInterIntra; mode < kMotionModes; ++mode) {
    if (!seq.seq_enabled_motion_modes[mode]) {
      if (frame.frame_enabled_motion_modes[mode]) {
        if (error != nullptr) {
          *error = "frame enables a sequence-disabled motion mode";
        }
        return false;
      }
      continue;
    }
    writer->put_bit(frame.frame_enabled_motion_modes[mode]);
  }
  return true;
}

bool read_frame_motion_modes(BitReader* reader, const SequenceMotionModes& seq,
                             FrameMotionModes* frame, std::string* error) {
  if (reader == nullptr || frame == nullptr) {
    if (error != nullptr) *error = "null argument";
    return false;
  }
  bool override[kMotionModes] = {false, false, false, false, false};
  if (seq.seq_frame_motion_modes_present_flag) {
    for (int mode = kMotionInterIntra; mode < kMotionModes; ++mode) {
      if (!seq.seq_enabled_motion_modes[mode]) continue;
      bool bit = false;
      if (!reader->read_bit(&bit)) {
        if (error != nullptr) *error = "truncated frame motion-mode bit";
        return false;
      }
      override[mode] = bit;
    }
  }
  FrameMotionModes out;
  out.frame_is_intra = frame->frame_is_intra;
  if (derive_frame_enabled_motion_modes(seq, override, &out, error) !=
      FrameStatus::kOk) {
    return false;
  }
  *frame = out;
  return true;
}

bool write_allow_warpmv_mode(BitWriter* writer, const SequenceMotionModes& seq,
                             const FrameMotionModes& frame, WarpmvMode* out,
                             std::string* error) {
  if (writer == nullptr || out == nullptr) {
    if (error != nullptr) *error = "null argument";
    return false;
  }
  WarpmvMode required;
  if (derive_allow_warpmv_mode(seq, frame, &required, error) !=
      FrameStatus::kOk) {
    return false;
  }
  if (!required.present) {
    if (out->present) {
      if (error != nullptr) {
        *error = "allow_warpmv_mode requested where it is not present";
      }
      return false;
    }
    out->allow_warpmv_mode = false;
    return true;
  }
  if (!out->present) {
    // Absent is not an option when the condition holds: the bit is always
    // coded.  The encoder may only choose the value.
    out->present = true;
  }
  writer->put_bit(out->allow_warpmv_mode);
  return true;
}

bool read_allow_warpmv_mode(BitReader* reader, const SequenceMotionModes& seq,
                            const FrameMotionModes& frame, WarpmvMode* out,
                            std::string* error) {
  if (reader == nullptr || out == nullptr) {
    if (error != nullptr) *error = "null argument";
    return false;
  }
  WarpmvMode required;
  if (derive_allow_warpmv_mode(seq, frame, &required, error) !=
      FrameStatus::kOk) {
    return false;
  }
  out->present = required.present;
  if (!required.present) {
    out->allow_warpmv_mode = false;
    return true;
  }
  bool bit = false;
  if (!reader->read_bit(&bit)) {
    if (error != nullptr) *error = "truncated allow_warpmv_mode";
    return false;
  }
  out->allow_warpmv_mode = bit;
  return true;
}

}  // namespace av2enc::inter::mv_stack_full
