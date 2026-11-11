// AV2 v1.0.0 temporal inter tools — reference-MV bank implementation.

#include "encoder/inter/temporal/ref_mv_bank.h"

#include <algorithm>

#include "encoder/inter/temporal/temporal_math.h"

namespace av2enc::inter::temporal {
namespace {

constexpr int kMaxRefMvCountDefault = 6;  // MAX_REF_MV_STACK_SIZE

// §7.12.2.21 check_rmb_cand().  The spec uses `/ 8` on the (signed) MV
// component; C++ integer division truncates toward zero, which is what the
// reference decoder's integer expression does.
void check_rmb_cand(const RefMvBankEntry& entry, const int cand_mvs[2][2],
                    bool is_compound, int mi_row, int mi_col, int mi_cols,
                    int mi_rows, int bw, int bh, int max_out, StackMv* stack,
                    int* num_found, int* prune_count) {
  if (*num_found >= max_out) {
    return;
  }
  if (*prune_count < kMaxPrNum) {
    for (int i = 0; i < *num_found; ++i) {
      ++(*prune_count);
      if (cand_mvs[0][0] == stack[i].mv[0].row &&
          cand_mvs[0][1] == stack[i].mv[0].col &&
          (!is_compound || (cand_mvs[1][0] == stack[i].mv[1].row &&
                            cand_mvs[1][1] == stack[i].mv[1].col))) {
        return;
      }
    }
  }
  for (int i = 0; i < 1 + (is_compound ? 1 : 0); ++i) {
    const int ref_y = mi_row * kMiSize + cand_mvs[i][0] / 8;
    const int ref_x = mi_col * kMiSize + cand_mvs[i][1] / 8;
    if (ref_x <= -bw || ref_y <= -bh || ref_x >= mi_cols * kMiSize ||
        ref_y >= mi_rows * kMiSize) {
      return;
    }
  }
  for (int i = 0; i < 1 + (is_compound ? 1 : 0); ++i) {
    stack[*num_found].mv[i].row = cand_mvs[i][0];
    stack[*num_found].mv[i].col = cand_mvs[i][1];
  }
  if (!is_compound) {
    stack[*num_found].mv[1] = stack[*num_found].mv[0];
  }
  stack[*num_found].cwp = static_cast<int>(entry.p[0]);
  ++(*num_found);
}

}  // namespace

int get_rmb_list_index(const int ref_frames[2]) {
  if (!is_inter_ref(ref_frames[0])) {
    return 8;
  }
  if (!is_inter_ref(ref_frames[1]) && ref_frames[0] <= 5) {
    return ref_frames[0];
  }
  if (ref_frames[0] == 0 && ref_frames[1] == 0) {
    return 6;
  }
  if (ref_frames[0] == 0 && ref_frames[1] == 1) {
    return 7;
  }
  return 8;
}

bool rmb_params_equal(const RefMvBankEntry& a, const RefMvBankEntry& b) {
  for (int i = 1; i < 6; ++i) {
    if (a.p[i] != b.p[i]) {
      return false;
    }
  }
  return true;
}

void update_ref_mv_count(RefMvBank* bank, const RefMvBankBlockContext& ctx) {
  if (bank == nullptr || ctx.chroma_part) {
    return;
  }
  const int sb_size4 = ctx.sb_size4;
  if (sb_size4 <= 0) {
    return;
  }
  int unit_size4 = sb_size4 >> 3;
  if (unit_size4 < 1) {
    unit_size4 = 1;
  }
  const int width_units = std::max(ctx.bw4 / unit_size4, 1);
  const int height_units = std::max(ctx.bh4 / unit_size4, 1);
  const int unit_count = width_units * height_units;
  if (ctx.mi_row % sb_size4 == 0 && ctx.mi_col % sb_size4 == 0) {
    bank->remain_hits = std::max(unit_count, 4);
    bank->unit_hits = 0;
  } else if (ctx.mi_row % unit_size4 == 0 && ctx.mi_col % unit_size4 == 0) {
    bank->remain_hits += unit_count;
    bank->unit_hits = 0;
  }
}

bool update_ref_mv_bank(RefMvBank* bank, const RefMvBankBlockContext& ctx,
                        const int ref_frames[2], const Mv mvs[2],
                        int cwp_idx, bool from_within_sb) {
  if (bank == nullptr) {
    return false;
  }
  if (from_within_sb) {
    update_ref_mv_count(bank, ctx);
    if (bank->remain_hits == 0 || bank->unit_hits >= 16) {
      return false;
    }
    --bank->remain_hits;
    ++bank->unit_hits;
  }
  ++bank->bank_hits;

  const int r0 = ref_frames[0];
  const int r1 = ref_frames[1];
  if (!is_inter_ref(r0)) {
    return false;
  }
  const bool is_compound = is_inter_ref(r1);
  const int ref = get_rmb_list_index(ref_frames);
  RefMvBankEntry p;
  p.p[0] = cwp_idx;
  p.p[1] = is_compound ? r0 + (r1 + 1) * kBankRefsPerFrame : r0;
  p.p[2] = mvs[0].row;
  p.p[3] = mvs[0].col;
  p.p[4] = is_compound ? mvs[1].row : 0;
  p.p[5] = is_compound ? mvs[1].col : 0;

  const int count = bank->size[ref];
  const int start = bank->start[ref];
  int found = -1;
  for (int i = 0; i < count; ++i) {
    const int idx = (start + i) % kRefMvBankSize;
    if (rmb_params_equal(bank->bank[ref][idx], p)) {
      found = i;
      break;
    }
  }
  if (found >= 0) {
    const int found_idx = (start + found) % kRefMvBankSize;
    RefMvBankEntry tmp = bank->bank[ref][found_idx];
    for (int i = found; i < count - 1; ++i) {
      const int idx0 = (start + i) % kRefMvBankSize;
      const int idx1 = (start + i + 1) % kRefMvBankSize;
      bank->bank[ref][idx0] = bank->bank[ref][idx1];
    }
    const int tail = (start + count - 1) % kRefMvBankSize;
    bank->bank[ref][tail] = tmp;
    return true;
  }

  const int idx = (start + count) % kRefMvBankSize;
  bank->bank[ref][idx] = p;
  if (count < kRefMvBankSize) {
    bank->size[ref] = count + 1;
  } else {
    bank->start[ref] = (start + 1) % kRefMvBankSize;
  }
  return true;
}

void reset_ref_mv_bank(RefMvBank* bank, int mi_row, int mi_col, int sb_size4,
                       bool frame_is_intra, bool top_row,
                       const RefMvBankRow& row) {
  if (bank == nullptr) {
    return;
  }
  bank->bank_hits = 0;
  bank->remain_hits = 0;
  bank->unit_hits = 0;
  if (frame_is_intra || top_row) {
    return;
  }
  if (row.sources == nullptr || row.mi_cols <= 0) {
    return;
  }
  RefMvBankBlockContext ctx;
  ctx.mi_row = mi_row;
  ctx.mi_col = mi_col;
  ctx.sb_size4 = sb_size4;
  ctx.bw4 = 1;
  ctx.bh4 = 1;
  int cand_col = mi_col;
  int row_hits = 0;
  while (cand_col < row.mi_cols && cand_col < mi_col + sb_size4 &&
         row_hits < 4) {
    const int cand_col2 = (cand_col >> 1) << 1;
    const RefMvBankRowSource& src =
        row.sources[static_cast<std::size_t>(cand_col2)];
    if (src.is_inter != 0) {
      ++row_hits;
      const int ref_frames[2] = {src.ref[0], src.ref[1]};
      const Mv mvs[2] = {Mv{src.mv[0][0], src.mv[0][1]},
                         Mv{src.mv[1][0], src.mv[1][1]}};
      update_ref_mv_bank(bank, ctx, ref_frames, mvs, src.cwp_idx, false);
    }
    const int step = src.bw4 > 0 ? src.bw4 : 1;
    cand_col += step;
  }
}

void fill_mvp_from_ref_mv_bank(const RefMvBank& bank, const int ref_frames[2],
                               bool is_compound, bool use_intrabc,
                               int max_drl_bits_minus_1, int mi_row,
                               int mi_col, int mi_cols, int mi_rows, int bw,
                               int bh, int max_out, StackMv* stack,
                               int* num_found, int* prune_count) {
  if (stack == nullptr || num_found == nullptr || prune_count == nullptr) {
    return;
  }
  const int ref = get_rmb_list_index(ref_frames);
  const int key =
      is_compound ? ref_frames[0] + (ref_frames[1] + 1) * kBankRefsPerFrame
                  : ref_frames[0];
  int max_ref_mv_count = max_drl_bits_minus_1 + 2;
  if (use_intrabc) {
    // The intrabc path uses max_bvp_drl_bits_minus_1; the caller can pass it
    // through the same argument.  Keep the common bound for safety.
    max_ref_mv_count = max_drl_bits_minus_1 + 2;
  }
  if (max_ref_mv_count > kMaxRefMvStackSize) {
    max_ref_mv_count = kMaxRefMvStackSize;
  }
  if (max_ref_mv_count > max_out) {
    max_ref_mv_count = max_out;
  }
  if (max_ref_mv_count > kMaxRefMvCountDefault) {
    max_ref_mv_count = kMaxRefMvCountDefault;
  }
  const int count = bank.size[ref];
  const int start = bank.start[ref];
  for (int i = count - 1; i >= 0 && *num_found < max_ref_mv_count; --i) {
    const int idx = (start + i) % kRefMvBankSize;
    const RefMvBankEntry& entry = bank.bank[ref][idx];
    if (entry.p[1] != key) {
      continue;
    }
    int cand_mvs[2][2] = {{0, 0}, {0, 0}};
    for (int list = 0; list < 2; ++list) {
      for (int comp = 0; comp < 2; ++comp) {
        cand_mvs[list][comp] =
            static_cast<int>(entry.p[2 + list * 2 + comp]);
      }
    }
    check_rmb_cand(entry, cand_mvs, is_compound, mi_row, mi_col, mi_cols,
                   mi_rows, bw, bh, max_out, stack, num_found, prune_count);
  }
  (void)kMaxRefMvCountDefault;
}

}  // namespace av2enc::inter::temporal
