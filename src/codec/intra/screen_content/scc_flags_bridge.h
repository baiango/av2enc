// Documented adapters from the existing palette/IBC stream modules to the
// §5.4.7/§5.18.3.3 SCC configuration owned by scc_flags.
//
// These are the only coupling points between this module and the tile coders;
// neither palette_stream nor ibc_stream is modified.  Each adapter:
//   1. runs the source module's own validation,
//   2. translates its flags into SequenceSccConfig/FrameSccConfig,
//   3. runs validate_sequence_scc_config()/validate_frame_scc_config().
//
// A caller can therefore keep using palette_stream/ibc_stream for tile coding
// while guaranteeing the header bits are the ones this module derives.

#pragma once

#include <string>

#include "codec/intra/ibc/stream/still_ibc_stream.h"
#include "codec/intra/palette/stream/palette_stream.h"
#include "codec/intra/screen_content/scc_flags.h"

namespace av2enc::still::scc_flags {

struct BridgedSccConfig {
  SequenceSccConfig sequence;
  FrameSccConfig frame;
  int seq_max_bvp_drl_bits_minus_1 = 0;
  bool allow_frame_max_bvp_drl_bits = false;
};

// palette_stream::ScreenContentFlags -> SCC config.  The palette adapter keeps
// single_picture_header in its flag block; when it is false the non-single
// §5.4.7 branch is selected.
bool from_palette_flags(const palette_stream::ScreenContentFlags& flags,
                        BridgedSccConfig& out, std::string& error);

// ibc_stream::StreamConfig -> SCC config.  The M1 IBC stream always uses the
// single-picture sequence header, so the sequence branch is inferred.
bool from_ibc_config(const ibc_stream::StreamConfig& config,
                     BridgedSccConfig& out, std::string& error);

}  // namespace av2enc::still::scc_flags
