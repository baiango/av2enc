#include "encoder/inter/drl_reorder/drl_reorder.h"

#include <algorithm>
#include <cstring>

namespace av2enc::inter::drl {
namespace {

int clip_int(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

void swap_candidate(StackCandidate* a, StackCandidate* b) {
  StackCandidate tmp;
  std::memcpy(&tmp, a, sizeof(tmp));
  std::memcpy(a, b, sizeof(tmp));
  std::memcpy(b, &tmp, sizeof(tmp));
}

}  // namespace

const char* mode_name(Mode mode) {
  switch (mode) {
    case Mode::kDisabled:
      return "DRL_REORDER_DISABLED";
    case Mode::kConstraint:
      return "DRL_REORDER_CONSTRAINT";
    case Mode::kAlways:
      return "DRL_REORDER_ALWAYS";
  }
  return "DRL_REORDER_INVALID";
}

Mode mode_from_sequence_header(const SequenceFlags& flags) {
  if (flags.disable_drl_reorder) {
    return Mode::kDisabled;
  }
  return flags.constrain_drl_reorder ? Mode::kConstraint : Mode::kAlways;
}

int derive_frame_max_drl_bits(const FrameDrlParams& params, bool* valid) {
  bool ok = params.seq_max_drl_bits_minus_1 >= 0 &&
            params.seq_max_drl_bits_minus_1 <= kMaxDrlBitsMinus1;
  int value = params.seq_max_drl_bits_minus_1;
  if (params.allow_frame_max_drl_bits && params.change_drl) {
    // §5.5: n = MAX_REF_MV_STACK_SIZE - 2 -> 4 values (0..3), then bump when
    // the decoded value reaches the sequence value.
    const int n = kMaxRefMvStackSize - 2;
    ok = ok && params.signaled_max_drl_bits_minus_1 >= 0 &&
         params.signaled_max_drl_bits_minus_1 < n;
    value = params.signaled_max_drl_bits_minus_1;
    if (value >= params.seq_max_drl_bits_minus_1) {
      value += 1;
    }
    ok = ok && value <= kMaxDrlBitsMinus1;
  }
  if (valid != nullptr) {
    *valid = ok;
  }
  return ok ? value : 0;
}

int drl_index_count(int max_drl_bits_minus_1, bool* valid) {
  const bool ok = max_drl_bits_minus_1 >= 0 &&
                  max_drl_bits_minus_1 <= kMaxDrlBitsMinus1;
  if (valid != nullptr) {
    *valid = ok;
  }
  return ok ? max_drl_bits_minus_1 + 1 : 0;
}

bool use_sort(Mode mode, bool use_temporal_first, int num_nearest) {
  switch (mode) {
    case Mode::kDisabled:
      return false;
    case Mode::kConstraint:
      return !use_temporal_first && num_nearest >= 4;
    case Mode::kAlways:
      return true;
  }
  return false;
}

int reorder_permutation(const StackCandidate* cand, int count, int num_nearest,
                        Mode mode, bool use_temporal_first, int* perm) {
  if (cand == nullptr || perm == nullptr || count < 0 ||
      count > kMaxRefMvStackSize) {
    return 0;
  }
  const int n = clip_int(num_nearest, 0, count);
  for (int i = 0; i < count; ++i) {
    perm[i] = i;
  }
  if (n <= 1 || !use_sort(mode, use_temporal_first, n)) {
    return 0;
  }

  // §7.12.2.19: exactly one selection step.  Ties keep the earliest index
  // because the comparison is strict (`if (maxWeight < WeightStack[idx])`).
  int max_weight = cand[0].weight;
  int max_idx = 0;
  for (int idx = 1; idx < n; ++idx) {
    if (max_weight < cand[idx].weight) {
      max_weight = cand[idx].weight;
      max_idx = idx;
    }
  }
  if (max_idx == 0) {
    return 0;
  }
  perm[0] = max_idx;
  perm[max_idx] = 0;
  return 1;
}

int reorder_stack(Stack* stack, Mode mode, bool use_temporal_first) {
  if (stack == nullptr) {
    return 0;
  }
  int perm[kMaxRefMvStackSize];
  const int swapped = reorder_permutation(
      stack->cand, stack->count, stack->num_nearest, mode, use_temporal_first,
      perm);
  if (swapped == 0) {
    return 0;
  }
  swap_candidate(&stack->cand[0], &stack->cand[perm[0]]);
  return 1;
}

bool pre_to_post_index(const int* perm, int count, int pre_index, int* post) {
  if (perm == nullptr || post == nullptr || count < 0 ||
      pre_index < 0 || pre_index >= count) {
    return false;
  }
  for (int i = 0; i < count; ++i) {
    if (perm[i] == pre_index) {
      *post = i;
      return true;
    }
  }
  return false;
}

bool post_to_pre_index(const int* perm, int count, int post_index, int* pre) {
  if (perm == nullptr || pre == nullptr || count < 0 ||
      post_index < 0 || post_index >= count) {
    return false;
  }
  *pre = perm[post_index];
  return true;
}

IndexChoice choose_index_for_mv(const Stack& stack, std::int32_t mv_row,
                                std::int32_t mv_col, int start, int m) {
  IndexChoice out;
  if (start < 0) {
    start = 0;
  }
  const int end = std::min(m, stack.count) - 1;
  for (int idx = start; idx <= end && idx < kMaxRefMvStackSize; ++idx) {
    if (stack.cand[idx].mv[0][0] == mv_row &&
        stack.cand[idx].mv[0][1] == mv_col) {
      out.found = true;
      out.index = idx;
      return out;
    }
  }
  return out;
}

int signal_index_for_pre(const StackCandidate* cand, int count,
                         int num_nearest, Mode mode, bool use_temporal_first,
                         int pre_index) {
  if (cand == nullptr || count < 0 || count > kMaxRefMvStackSize ||
      pre_index < 0 || pre_index >= count) {
    return -1;
  }
  int perm[kMaxRefMvStackSize];
  reorder_permutation(cand, count, num_nearest, mode, use_temporal_first, perm);
  int post = -1;
  if (!pre_to_post_index(perm, count, pre_index, &post)) {
    return -1;
  }
  return post;
}

std::string weight_string(const StackCandidate* cand, int count) {
  std::string out;
  for (int i = 0; i < count; ++i) {
    if (i != 0) {
      out += ',';
    }
    out += std::to_string(cand[i].weight);
  }
  return out;
}

void reproduce_legacy_selection_sort(StackCandidate* cand, int num_nearest) {
  if (cand == nullptr) {
    return;
  }
  for (int start = 0; start + 1 < num_nearest; ++start) {
    int max_weight = cand[start].weight;
    int max_idx = start;
    for (int idx = start + 1; idx < num_nearest; ++idx) {
      if (max_weight < cand[idx].weight) {
        max_weight = cand[idx].weight;
        max_idx = idx;
      }
    }
    if (max_idx != start) {
      swap_candidate(&cand[start], &cand[max_idx]);
    }
  }
}

}  // namespace av2enc::inter::drl
