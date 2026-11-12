#include "codec/intra/screen_content/scc_flags_bridge.h"

namespace av2enc::still::scc_flags {

bool from_palette_flags(const palette_stream::ScreenContentFlags& flags,
                        BridgedSccConfig& out, std::string& error) {
  error.clear();
  out = BridgedSccConfig{};
  if (!palette_stream::validate_screen_content_flags(flags, error)) {
    return false;
  }

  out.sequence.single_picture_header_flag = flags.single_picture_header;
  if (!flags.single_picture_header) {
    // The palette adapter's own non-single writer uses
    // seq_choose_screen_content_tools = allow_screen_content_tools.  Express
    // the same semantics through SELECT so the frame header keeps the bit.
    out.sequence.seq_choose_screen_content_tools =
        flags.allow_screen_content_tools;
    out.sequence.seq_force_screen_content_tools = false;
    if (flags.allow_screen_content_tools) {
      out.sequence.seq_choose_integer_mv = flags.force_integer_mv;
      out.sequence.seq_force_integer_mv = false;
    }
  }

  FrameSccContext context;
  context.frame_is_intra = true;
  out.frame.allow_screen_content_tools = flags.allow_screen_content_tools;
  out.frame.force_integer_mv =
      flags.allow_screen_content_tools && flags.force_integer_mv;
  out.frame.allow_intrabc = flags.allow_intrabc;
  out.frame.allow_global_intrabc =
      flags.allow_intrabc && flags.allow_global_intrabc;
  out.frame.allow_local_intrabc =
      flags.allow_intrabc && flags.allow_local_intrabc;
  out.frame.change_bvp_drl = false;
  out.frame.max_bvp_drl_bits_minus_1 = 0;
  out.seq_max_bvp_drl_bits_minus_1 = 0;
  out.allow_frame_max_bvp_drl_bits = false;

  if (!validate_sequence_scc_config(out.sequence, error)) return false;
  const SequenceSccState sequence =
      derive_sequence_scc_state(out.sequence);
  return validate_frame_scc_config(sequence, context, out.frame,
                                   out.seq_max_bvp_drl_bits_minus_1,
                                   out.allow_frame_max_bvp_drl_bits, error);
}

bool from_ibc_config(const ibc_stream::StreamConfig& config,
                     BridgedSccConfig& out, std::string& error) {
  error.clear();
  out = BridgedSccConfig{};

  // Mirror the policy checks in ibc_stream::encode_ibc_stream().
  if (!config.force_integer_mv) {
    error = "ibc_stream only writes force_integer_mv=1";
    return false;
  }
  if (config.allow_intrabc &&
      !(config.allow_global_intrabc || config.allow_local_intrabc)) {
    error = "ibc_stream requires a usable IBC reference region";
    return false;
  }
  if (config.allow_intrabc && !config.allow_screen_content_tools) {
    error = "ibc_stream requires allow_screen_content_tools for IBC";
    return false;
  }

  out.sequence.single_picture_header_flag = true;
  FrameSccContext context;
  context.frame_is_intra = true;
  out.frame.allow_screen_content_tools = config.allow_screen_content_tools;
  out.frame.force_integer_mv =
      config.allow_screen_content_tools && config.force_integer_mv;
  out.frame.allow_intrabc = config.allow_intrabc;
  out.frame.allow_global_intrabc =
      config.allow_intrabc && config.allow_global_intrabc;
  out.frame.allow_local_intrabc =
      config.allow_intrabc && config.allow_local_intrabc;
  out.frame.change_bvp_drl = false;
  out.seq_max_bvp_drl_bits_minus_1 = config.seq_max_bvp_drl_bits_minus_1;
  out.allow_frame_max_bvp_drl_bits = false;
  out.frame.max_bvp_drl_bits_minus_1 =
      config.seq_max_bvp_drl_bits_minus_1;

  if (!validate_sequence_scc_config(out.sequence, error)) return false;
  const SequenceSccState sequence =
      derive_sequence_scc_state(out.sequence);
  return validate_frame_scc_config(sequence, context, out.frame,
                                   out.seq_max_bvp_drl_bits_minus_1,
                                   out.allow_frame_max_bvp_drl_bits, error);
}

}  // namespace av2enc::still::scc_flags
