// AV2 v1.0.0 temporal inter tools — reference-MV bank.
//
// Implements §7.12.2.21 `fill_mvp_from_ref_mv_bank()`, §5.20.2.2
// `reset_refmv_bank()`, and the `update_ref_mv_bank()` / `update_ref_mv_count()`
// state machine from §7.12.2.21.  The bank is a per-reference ring of four
// six-word parameter entries; the temporal component only supplies the data
// structure and the normative ordering, pruning and range checks.  Whether the
// bank is enabled is `enable_refmvbank` in the sequence header.

#ifndef AV2ENC_ENCODER_INTER_TEMPORAL_REF_MV_BANK_H_
#define AV2ENC_ENCODER_INTER_TEMPORAL_REF_MV_BANK_H_

#include "codec/inter/temporal/temporal_types.h"

namespace av2enc::inter::temporal {

// §7.12.2.21 get_rmb_list_index(): 0..8.
int get_rmb_list_index(const int ref_frames[2]);

// §7.12.2.21 rmb_params_equal(): p[1..5], the CWP index is not part of the key.
bool rmb_params_equal(const RefMvBankEntry& a, const RefMvBankEntry& b);

// Block context for update_ref_mv_count().  `sb_size4` is the superblock
// extent in 4x4 units (16 for 64x64, 32 for 128x128).
struct RefMvBankBlockContext {
  int mi_row = 0;
  int mi_col = 0;
  int bw4 = 1;
  int bh4 = 1;
  int sb_size4 = 16;
  bool chroma_part = false;
};

// §7.12.2.21 update_ref_mv_count().
void update_ref_mv_count(RefMvBank* bank, const RefMvBankBlockContext& ctx);

// §7.12.2.21 update_ref_mv_bank().  Returns true when an entry was written or
// refreshed.  `from_within_sb` corresponds to the spec's fromWithinSb: when
// false (seeding from the row above) the hit budget is not consulted.
bool update_ref_mv_bank(RefMvBank* bank, const RefMvBankBlockContext& ctx,
                        const int ref_frames[2], const Mv mvs[2],
                        int cwp_idx, bool from_within_sb);

// A row of already-coded mode info used by reset_ref_mv_bank().  Index
// `sources[mi_col]`; `bw4` is Num_4x4_Blocks_Wide of the block at that column.
struct RefMvBankRow {
  const RefMvBankRowSource* sources = nullptr;
  int mi_cols = 0;
};

// §5.20.2.2 reset_refmv_bank().  Called at the start of each superblock.  When
// `frame_is_intra` or `top_row` is set only the hit counters are cleared.
void reset_ref_mv_bank(RefMvBank* bank, int mi_row, int mi_col, int sb_size4,
                       bool frame_is_intra, bool top_row,
                       const RefMvBankRow& row);

// §7.12.2.21 fill_mvp_from_ref_mv_bank().  Appends bank candidates to the MV
// stack using the same NumMvFound/PruneCount conventions as the rest of
// §7.12.2.  `num_found` is both input and output.
void fill_mvp_from_ref_mv_bank(const RefMvBank& bank, const int ref_frames[2],
                               bool is_compound, bool use_intrabc,
                               int max_drl_bits_minus_1, int mi_row,
                               int mi_col, int mi_cols, int mi_rows, int bw,
                               int bh, int max_out, StackMv* stack,
                               int* num_found, int* prune_count);

}  // namespace av2enc::inter::temporal

#endif  // AV2ENC_ENCODER_INTER_TEMPORAL_REF_MV_BANK_H_
