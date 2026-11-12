// Independent reader ("stack mirror") for the §7.12.2 MV-stack record.
//
// This file deliberately shares no parsing helper with stack_io.cpp: it has its
// own cursor, its own field order and its own bounds checks.  Its job is to
// catch a divergence between the stack the encoder signalled against and the
// stack a second implementation believes was signalled.
//
// Two levels of checking:
//   * `parse()` reconstructs the record and rejects truncation, trailing bytes,
//     a bad checksum, and every structural invariant (counts, ranges, the
//     post-sort weight invariant, prune budgets);
//   * `lockstep()` re-resolves the prediction for every DRL index / mode pair
//     from the mirrored candidates and compares it with the encoder's
//     resolution.  A desync shows up as a differing MV at a specific index.
#pragma once

#include <cstdint>
#include <span>
#include <string>

#include "codec/inter/mv_stack/mv_stack_full.h"

namespace av2enc::inter::mv_stack_full {

struct MirrorCandidate {
  Mv mv[2];
  int weight = 0;
  int row_offset = 0;
  int col_offset = 0;
  int cwp = 0;
};

struct MirrorStack {
  int num_lists = 1;
  bool sorted = false;
  int count = 0;
  int num_nearest = 0;
  int num_warp_found = 0;
  int prune_count = 0;
  int single_prune_count = 0;
  int derived_prune_count = 0;
  int derived_count = 0;
  MirrorCandidate cand[kMaxRefMvStackSize];
  adv::WarpParams warp_params[kMaxWarpRefCandidates];
};

class StackMirror {
 public:
  bool parse(std::span<const std::uint8_t> record, std::string* error);
  const MirrorStack& stack() const { return stack_; }

  // Structural invariants that do not depend on the encoder.
  bool validate(std::string* error) const;

  // Independent §5.20.7.13 resolution: mode 0 = NEARMV (stack entry),
  // 1 = GLOBALMV (GlobalMvs[0]), 2 = NEWMV (stack entry + MVD).
  bool resolve_mv(int drl_index, int mode, Mv global, Mv mvd, Mv* out,
                  std::string* error) const;

  // Field-by-field comparison against the encoder's stack.
  bool equals(const Stack& encoder, std::string* why) const;

 private:
  MirrorStack stack_{};
};

// Free-function form of the independent resolution, usable on a MirrorStack
// that was copied out of a StackMirror.
bool mirror_resolve_mv(const MirrorStack& stack, int drl_index, int mode,
                       Mv global, Mv mvd, Mv* out, std::string* error);

struct LockstepReport {
  int drl_checked = 0;
  bool fields_equal = true;
  bool resolved_equal = true;
  int first_bad_drl = -1;
  int first_bad_mode = -1;
  std::string why;
};

// Resolves every DRL index in [0, min(count-1, kMaxDrlIndex)] for modes
// NEARMV and NEWMV (with a fixed non-zero MVD) and GLOBALMV, from both the
// encoder stack and the mirror, and reports the first mismatch.  `global` is
// GlobalMvs[0] for the block (the gate records it separately).
bool lockstep(const Stack& encoder, int num_lists, Mv global,
              const MirrorStack& mirror, LockstepReport* report);

}  // namespace av2enc::inter::mv_stack_full
