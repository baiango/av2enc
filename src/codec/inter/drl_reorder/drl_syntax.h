// §5.20.7.8 read_drl_idx writer and independent reader.
//
// The two directions are intentionally separate functions.  The writer emits
// one `drl_mode` symbol per loop iteration; the reader loops until it observes
// drl_mode == 0 or exhausts [start, m).  Sharing a single "encode/decode
// index" helper would make a symmetric off-by-one invisible, so the test
// drives them against each other plus the pinned goldens.
//
// CDF selection (§8.3):
//   skip_mode == 0, RefFrame[0] != TIP_FRAME:
//       TileDrlModeCdf[Min(idx, 2)][NewMvContext]
//   skip_mode == 1:
//       TileSkipDrlModeCdf[Min(idx, 2)]
//   RefFrame[0] == TIP_FRAME:
//       TileTipDrlModeCdf[Min(idx, 2)]
//
// The M2 stream has skip_mode_present = 0, enable_tip = 0 and
// enable_ref_frame_mvs = 0, so it only uses the first row set, but the full
// selection is implemented so the context cannot silently diverge.

#pragma once

#include <cstdint>

#include "codec/entropy/av2/av2_entropy.h"

namespace av2enc::inter::drl {

inline constexpr int kDrlModeRows = 3;
inline constexpr int kDrlNewMvContexts = 5;

// One tile's independent CDF state.  Copy this per tile; never share it
// between tiles (the AV2 entropy coder mutates CDFs after each symbol).
struct DrlCdfState {
  // Default_Drl_Mode_Cdf [3, 5, 3].
  std::uint16_t drl_mode[kDrlModeRows][kDrlNewMvContexts][3];
  // Default_Skip_Drl_Mode_Cdf [3, 3].
  std::uint16_t skip_drl_mode[kDrlModeRows][3];
  // Default_Tip_Drl_Mode_Cdf [3, 3].
  std::uint16_t tip_drl_mode[kDrlModeRows][3];
};

void init_default_drl_cdfs(DrlCdfState* state);

enum class DrlCdfSource : std::uint8_t {
  kInter = 0,    // TileDrlModeCdf[Min(idx,2)][NewMvContext]
  kSkipMode = 1, // TileSkipDrlModeCdf[Min(idx,2)]
  kTip = 2,      // TileTipDrlModeCdf[Min(idx,2)]
};

// §8.3: Min(idx, 2).
int drl_cdf_row(int idx);

struct DrlSymbolContext {
  DrlCdfSource source = DrlCdfSource::kInter;
  // §7.11.2 NewMvContext, 0..4.  Ignored for kSkipMode / kTip.
  int new_mv_context = 0;
};

enum class Status : std::uint8_t {
  kOk = 0,
  kInvalidArgument,
  kInvalidIndex,
  kCoderError,
};

const char* status_name(Status status);

// Writes read_drl_idx(start, m) for `index`.  Valid indices are
// start <= index <= m.  index == m is the exhaustion path and therefore has no
// terminating zero symbol.
Status write_drl_index(av2::Encoder* enc, DrlCdfState* cdfs,
                       const DrlSymbolContext& ctx, int index, int start,
                       int m);

// Reads read_drl_idx(start, m).  The returned index is in [start, m].
Status read_drl_index(av2::Decoder* dec, DrlCdfState* cdfs,
                      const DrlSymbolContext& ctx, int start, int m,
                      int* index);

// Number of drl_mode symbols a conforming writer emits for this index.
int drl_symbol_count(int index, int start, int m);

}  // namespace av2enc::inter::drl
