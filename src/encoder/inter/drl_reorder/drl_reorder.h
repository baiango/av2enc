// AV2 §5.4.6 / §7.12.2 DRL reordering, encoder transform and decoder mirror.
//
// Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
// (2026-05-28, commit 38f28e7):
//
//   §5.4.6   sequence_inter_config(): disable_drl_reorder,
//            constrain_drl_reorder, seq_max_drl_bits_minus_1,
//            allow_frame_max_drl_bits, and Table 6.4 (DrlReorder).
//   §5.5     frame_header_info(): change_drl / max_drl_bits_minus_1.
//   §5.20.7.6 inter block mode info: m = max_drl_bits_minus_1 + 1,
//            read_drl_idx(0, m), and the second read for has_second_drl().
//   §5.20.7.8 read_drl_idx(): loop over idx = start .. m-1, stop on
//            drl_mode == 0, return m on exhaustion.
//   §7.12.2  Find MV stack: step 14 snapshots numNearest, step 16 derives
//            useSort, step 17 invokes the sorting process.
//   §7.12.2.19 Sorting process: one selection step, i.e. move the single
//            highest-weight entry in [start, end) to `start` by swapping the
//            complete stack record (weight, cwp, row/col offset, both MVs).
//            It is *not* a full selection sort.  `end` is numNearest, so the
//            derived-SMVP and extra-search candidates appended later are never
//            reordered.
//   §8.3     drl_mode CDF selection: TileDrlModeCdf[Min(idx, 2)][NewMvContext],
//            TileSkipDrlModeCdf[Min(idx, 2)] when skip_mode == 1, and
//            TileTipDrlModeCdf[Min(idx, 2)] when RefFrame[0] == TIP_FRAME.
//
// Failure mode this module exists to prevent
// ------------------------------------------
// A selection sort over the whole [0, numNearest) prefix is observably different
// from the normative single swap whenever more than one non-maximum entry is
// out of weight order.  For weights {1, 2, 3, 4} the normative transform yields
// {4, 2, 3, 1}; a full selection sort yields {4, 3, 2, 1}.  The encoder then
// signals an index that selects a different MV in the decoder's stack.  The
// stream still parses, so the symptom is an MV/reconstruction mismatch, not a
// syntax error.  That is exactly the historical DRL_REORDER_CONSTRAINT desync.

#pragma once

#include <cstdint>
#include <span>
#include <string>

namespace av2enc::inter::drl {

// §9.2 constants.
inline constexpr int kMaxRefMvStackSize = 6;  // MAX_REF_MV_STACK_SIZE
inline constexpr int kMaxDrlIndex = kMaxRefMvStackSize - 1;
inline constexpr int kMaxDrlBitsMinus1 = kMaxRefMvStackSize - 2;  // 4
inline constexpr int kNewMvContexts = 5;

// Table 6.4.
enum class Mode : std::uint8_t {
  kDisabled = 0,    // DRL_REORDER_DISABLED
  kConstraint = 1,  // DRL_REORDER_CONSTRAINT
  kAlways = 2,      // DRL_REORDER_ALWAYS
};

const char* mode_name(Mode mode);

// §5.4.6 sequence-level derivation.
struct SequenceFlags {
  bool disable_drl_reorder = false;
  bool constrain_drl_reorder = false;
};

Mode mode_from_sequence_header(const SequenceFlags& flags);

// §5.5 frame header: max_drl_bits_minus_1 starts as the sequence value and may
// be overridden when allow_frame_max_drl_bits == 1 and change_drl == 1.
// The coded symbol has n = MAX_REF_MV_STACK_SIZE - 2 = 4 values; a decoded
// value >= seq_max_drl_bits_minus_1 is incremented, which spans the full
// 0..4 range.  `signaled` is ignored unless `change_drl` is set.
struct FrameDrlParams {
  int seq_max_drl_bits_minus_1 = kMaxDrlBitsMinus1;
  bool allow_frame_max_drl_bits = false;
  bool change_drl = false;
  int signaled_max_drl_bits_minus_1 = 0;
};

int derive_frame_max_drl_bits(const FrameDrlParams& params,
                              bool* valid = nullptr);

// m = max_drl_bits_minus_1 + 1: the number of drl_mode symbols the decoder may
// read, and the exhaustion value returned by read_drl_idx().
int drl_index_count(int max_drl_bits_minus_1, bool* valid = nullptr);

// One stack record, layout-neutral so that it can be mirrored from
// merge::StackCandidate / inter::mc stacks without depending on either.
struct StackCandidate {
  // mv[list][0] is the row (y) component, mv[list][1] the column (x) component.
  std::int32_t mv[2][2] = {{0, 0}, {0, 0}};
  int weight = 0;
  int row_offset = 0;
  int col_offset = 0;
  std::int32_t cwp = 8;  // CWP_EQUAL
};

struct Stack {
  StackCandidate cand[kMaxRefMvStackSize];
  int count = 0;
  int num_nearest = 0;
};

// §7.12.2 step 16.
bool use_sort(Mode mode, bool use_temporal_first, int num_nearest);

// §7.12.2.19, applied to [0, num_nearest).  On return `perm[post] = pre` for
// post in [0, num_nearest) and perm[i] == i elsewhere in [0, count).  The
// identity is returned when useSort() is false.  Returns the number of
// reordered positions (0 or 1 for the normative process).
int reorder_permutation(const StackCandidate* cand, int count, int num_nearest,
                        Mode mode, bool use_temporal_first, int* perm);

// Applies the normative transform in place.  Returns 1 when a swap happened,
// 0 otherwise.  `stack->count`/`num_nearest` are unchanged.
int reorder_stack(Stack* stack, Mode mode, bool use_temporal_first);

// Encoder-side index mapping.  `perm` is the array produced above.
// pre/post refer to the unsorted / sorted stack positions.
bool pre_to_post_index(const int* perm, int count, int pre_index, int* post);
bool post_to_pre_index(const int* perm, int count, int post_index, int* pre);

// Encoder-side choice: lowest post-reorder index in [start, Min(m, count-1)]
// whose list-0 MV equals `mv_row`/`mv_col`.  Deterministic; ties resolve to the
// lowest index (the decoder will read the same index).
struct IndexChoice {
  bool found = false;
  int index = 0;  // post-reorder (signalable) index
};
IndexChoice choose_index_for_mv(const Stack& stack, std::int32_t mv_row,
                                std::int32_t mv_col, int start, int m);

// Convenience: signal `pre_index` after the reorder.  Returns -1 when the
// pre-reorder index is not in [0, count).
int signal_index_for_pre(const StackCandidate* cand, int count,
                         int num_nearest, Mode mode, bool use_temporal_first,
                         int pre_index);

// Debug string "w0,w1,..." for tests and golden files.
std::string weight_string(const StackCandidate* cand, int count);

// Historical (incorrect) transform retained only to reproduce the desync.
// Full selection sort over [0, num_nearest).  New code must never call this.
void reproduce_legacy_selection_sort(StackCandidate* cand, int num_nearest);

}  // namespace av2enc::inter::drl
