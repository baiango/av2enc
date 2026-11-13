#include "codec/inter/mv_stack/ref_map.h"

#include <algorithm>
#include <array>
#include <string>

namespace av2enc::inter::mv_stack_full {
namespace {

constexpr int kDistScoreLookup[kDecayDistCap + 1] = {0, 64, 96, 112, 120, 124,
                                                     126};

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

int floor_log2(int value) {
  int bits = 0;
  while (value > 1) {
    value >>= 1;
    ++bits;
  }
  return bits;
}

int ceil_log2(std::uint32_t value) {
  if (value <= 1) return 0;
  int bits = 0;
  std::uint32_t v = value - 1;
  while (v != 0) {
    v >>= 1;
    ++bits;
  }
  return bits;
}

int relative_dist(int a, int b) { return clip3(-127, 127, a - b); }

bool dependency_ok(const DependencyMaps& deps, int mlayer, int tlayer) {
  if (mlayer < 0 || mlayer >= 64 || tlayer < 0 || tlayer >= 64) return false;
  return ((deps.mlayer_mask >> mlayer) & 1u) != 0 &&
         ((deps.tlayer_mask >> tlayer) & 1u) != 0;
}

struct ScoreEntry {
  int index = 0;
  int score = 0;
  int display_order = 0;
  int distance = 0;
  int base_q = 0;
  int layer = 0;
};

}  // namespace

bool RefSlotBank::refresh(int refresh_frame_flags, int order_hint,
                          int frame_width, int frame_height, int base_q_idx,
                          bool key_frame, int tlayer_id, int mlayer_id,
                          bool implicit_output, bool immediate_output,
                          int counter, std::string* error) {
  const auto fail = [&](const char* message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (num_ref_frames < 1 || num_ref_frames > kMaxNumRefSlots) {
    return fail("NumRefFrames outside 1..16");
  }
  if (counter < 0) {
    counter = frame_counter + 1;
  }
  frame_counter = counter;
  bool first = true;
  for (int i = 0; i < num_ref_frames; ++i) {
    if (((refresh_frame_flags >> i) & 1) == 0) continue;
    RefSlot& slot = slots[i];
    // §7.23: a KEY/SWITCH frame marks only the first refreshed slot valid.
    slot.valid = key_frame ? first : true;
    first = false;
    slot.restricted = false;
    slot.order_hint = order_hint;
    slot.frame_width = frame_width;
    slot.frame_height = frame_height;
    slot.counter = counter;
    slot.mlayer = mlayer_id;
    slot.tlayer = tlayer_id;
    slot.base_q_idx = base_q_idx;
    slot.implicit_output = implicit_output;
    slot.immediate_output = immediate_output;
    slot.is_bridge = false;
  }
  return true;
}

bool get_ref_frames(const RefMapParams& params, RefMap* out,
                    std::string* error) {
  const auto fail = [&](const char* message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (out == nullptr) return fail("null output");
  if (params.slots == nullptr) return fail("null slot bank");
  if (params.num_ref_frames < 1 ||
      params.num_ref_frames > kMaxNumRefSlots) {
    return fail("NumRefFrames outside 1..16");
  }
  if (params.active_num_ref_frames < 1 ||
      params.active_num_ref_frames > kRefsPerFrameConst) {
    return fail("ActiveNumRefFrames outside 1..7");
  }

  RefMap map;
  const int n = params.num_ref_frames;

  // §7.7 mapOrderHint / first_slot_with_ref / valid_ref_frame_size.
  int max_disp = 0;
  bool map_base_q[kMaxNumRefSlots] = {};
  int map_base_q_idx[kMaxNumRefSlots] = {};
  for (int i = 0; i < n; ++i) {
    map.map_order_hint[i] = -1;
    bool first = params.slots[i].valid;
    if (first) {
      for (int j = 0; j < i; ++j) {
        if (params.slots[j].valid &&
            params.slots[j].counter == params.slots[i].counter) {
          first = false;
          break;
        }
      }
    }
    const RefSlot& slot = params.slots[i];
    const bool bridge_ok = !params.is_bridge || i == params.bridge_frame_ref_idx;
    const bool allowed =
        ((params.allowed_frames >> i) & 1u) != 0;
    if (first && !slot.restricted && bridge_ok && allowed &&
        dependency_ok(params.deps, slot.mlayer, slot.tlayer)) {
      bool size_ok = true;
      if (params.check_res) {
        size_ok = 2 * params.frame_width >= slot.frame_width &&
                  2 * params.frame_height >= slot.frame_height &&
                  params.frame_width <= 16 * slot.frame_width &&
                  params.frame_height <= 16 * slot.frame_height;
      }
      if (size_ok) map.map_order_hint[i] = slot.order_hint;
      map_base_q[i] = true;
      map_base_q_idx[i] = slot.base_q_idx;
      max_disp = std::max(max_disp, slot.order_hint);
    }
  }

  // §7.7 scoring.
  std::array<ScoreEntry, kMaxNumRefSlots> entries{};
  int nranked = 0;
  int max_q = 0;
  int min_q = 0;
  for (int i = 0; i < n; ++i) {
    const int d = map.map_order_hint[i];
    if (d == -1) continue;
    const int q = map_base_q_idx[i];
    const int disp_diff = relative_dist(params.order_hint, d);
    const int t_dist =
        std::abs(disp_diff) + params.obu_mlayer_id - params.slots[i].mlayer;
    int score;
    if (max_disp > params.order_hint) {
      score = (t_dist << kDistWeightBits) + q;
    } else {
      score = kDistScoreLookup[std::min(t_dist, kDecayDistCap)] +
              std::max(t_dist - kDecayDistCap, 0) + q;
    }
    const int ref_ratio =
        floor_log2(params.slots[i].frame_width * params.slots[i].frame_height);
    score -= ref_ratio << 5;

    bool duplicate = false;
    for (int j = 0; j < nranked; ++j) {
      if (entries[static_cast<std::size_t>(j)].display_order == d &&
          entries[static_cast<std::size_t>(j)].score == score &&
          entries[static_cast<std::size_t>(j)].layer == params.slots[i].mlayer) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) continue;
    ScoreEntry& e = entries[static_cast<std::size_t>(nranked)];
    e.index = i;
    e.score = score;
    e.display_order = d;
    e.distance = disp_diff;
    e.base_q = q;
    e.layer = params.slots[i].mlayer;
    if (nranked == 0) {
      min_q = q;
      max_q = q;
    } else {
      min_q = std::min(q, min_q);
      max_q = std::max(q, max_q);
    }
    ++nranked;
  }

  // §7.7 get_unmapped_ref + drop.
  if (nranked > kRefsPerFrameConst) {
    const int q_thresh = (max_q + min_q + 1) / 2;
    int n_past = 0;
    int n_future = 0;
    int max_past_distance = 0;
    int max_future_distance = 0;
    int past_idx = 0;
    int future_idx = 0;
    for (int i = 0; i < nranked; ++i) {
      const ScoreEntry& e = entries[static_cast<std::size_t>(i)];
      if (e.base_q < q_thresh) continue;
      const int d = e.distance;
      if (d > 0) {
        if (d > max_past_distance) {
          max_past_distance = d;
          past_idx = i;
        }
        ++n_past;
      } else if (d < 0) {
        if (-d > max_future_distance) {
          max_future_distance = -d;
          future_idx = i;
        }
        ++n_future;
      }
    }
    int unmapped_idx = -1;
    if (n_past > n_future) {
      unmapped_idx = past_idx;
    } else if (n_past < n_future) {
      unmapped_idx = future_idx;
    } else if (n_past > 0) {
      unmapped_idx = max_past_distance >= max_future_distance ? past_idx
                                                              : future_idx;
    }
    if (unmapped_idx >= 0) {
      entries[static_cast<std::size_t>(unmapped_idx)].score = 0x7fffffff;
    }
  }

  // §7.7 bubble_sort_ref_scores (stable: swap only on strictly greater).
  for (int i = nranked - 1; i > 0; --i) {
    for (int j = 0; j < i; ++j) {
      if (entries[static_cast<std::size_t>(j)].score >
          entries[static_cast<std::size_t>(j + 1)].score) {
        std::swap(entries[static_cast<std::size_t>(j)],
                  entries[static_cast<std::size_t>(j + 1)]);
      }
    }
  }

  map.num_total_refs = std::min(nranked, params.active_num_ref_frames);
  for (int i = 0; i < map.num_total_refs; ++i) {
    map.ref_frame_idx[i] = entries[static_cast<std::size_t>(i)].index;
    map.scores_index[i] = entries[static_cast<std::size_t>(i)].index;
    map.scores_score[i] = entries[static_cast<std::size_t>(i)].score;
    map.scores_order_hint[i] = entries[static_cast<std::size_t>(i)].display_order;
    map.scores_distance[i] = entries[static_cast<std::size_t>(i)].distance;
  }

  // §7.7 restricted append.
  if (params.check_res && !params.is_bridge) {
    for (int i = 0; i < n; ++i) {
      const RefSlot& slot = params.slots[i];
      if (!slot.valid || !slot.restricted) continue;
      if (!dependency_ok(params.deps, slot.mlayer, slot.tlayer)) continue;
      if (((params.allowed_frames >> i) & 1u) == 0) continue;
      if (map.num_total_refs >= params.active_num_ref_frames) break;
      map.ref_frame_idx[map.num_total_refs] = i;
      map.scores_index[map.num_total_refs] = i;
      map.scores_score[map.num_total_refs] = 0;
      map.scores_order_hint[map.num_total_refs] = kRestrictedOrderHint;
      map.scores_distance[map.num_total_refs] = 0;
      ++map.num_total_refs;
    }
  }

  *out = map;
  return true;
}

bool get_past_future_cur_ref_lists(const RefMapParams& params, RefMap* map,
                                   std::string* error) {
  const auto fail = [&](const char* message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (map == nullptr || params.slots == nullptr) return fail("null argument");
  map->num_past_refs = 0;
  map->num_future_refs = 0;
  map->num_cur_refs = 0;
  map->closest_past = kRefNone;
  map->closest_future = kRefNone;
  map->furthest_future = kRefNone;
  for (int i = 0; i < map->num_total_refs; ++i) {
    const int slot = map->ref_frame_idx[i];
    if (slot < 0 || slot >= params.num_ref_frames) {
      return fail("reference index outside the slot bank");
    }
    if (params.slots[slot].restricted) continue;
    if (map->scores_distance[i] > 0) {
      ++map->num_past_refs;
      if (map->closest_past == kRefNone ||
          map->scores_distance[i] < map->scores_distance[map->closest_past]) {
        map->closest_past = i;
      }
    } else if (map->scores_distance[i] < 0) {
      ++map->num_future_refs;
      if (map->furthest_future == kRefNone ||
          params.slots[map->ref_frame_idx[map->furthest_future]].order_hint <
              params.slots[slot].order_hint) {
        map->furthest_future = i;
      }
      if (map->closest_future == kRefNone ||
          params.slots[slot].order_hint <
              params.slots[map->ref_frame_idx[map->closest_future]]
                  .order_hint) {
        map->closest_future = i;
      }
    } else {
      map->cur_refs[map->num_cur_refs] = i;
      ++map->num_cur_refs;
    }
  }
  map->skip_seg_frame =
      map->num_cur_refs > 0 ? map->cur_refs[0] : map->closest_past;
  // §7.8 tail: `if (SkipSegFrame == NONE) SkipSegFrame = 0`.
  if (map->skip_seg_frame == kRefNone) map->skip_seg_frame = 0;
  return true;
}

bool derive_explicit_ref_map(const RefMapParams& params, int num_total_refs,
                             const int* ref_frame_idx, RefMap* out,
                             std::string* error) {
  const auto fail = [&](const char* message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (out == nullptr) return fail("null output");
  if (params.slots == nullptr) return fail("null slot bank");
  if (params.num_ref_frames < 1 || params.num_ref_frames > kMaxNumRefSlots) {
    return fail("NumRefFrames outside 1..16");
  }
  if (num_total_refs < 0 || num_total_refs > kRefsPerFrameConst) {
    return fail("NumTotalRefs outside 0..7");
  }
  if (num_total_refs > 0 && ref_frame_idx == nullptr) {
    return fail("null ref_frame_idx");
  }
  RefMap map;
  map.num_total_refs = num_total_refs;
  for (int i = 0; i < num_total_refs; ++i) {
    const int slot_index = ref_frame_idx[i];
    if (slot_index < 0 || slot_index >= params.num_ref_frames) {
      return fail("ref_frame_idx outside NumRefFrames");
    }
    const RefSlot& slot = params.slots[slot_index];
    if (!slot.valid) return fail("ref_frame_idx names an invalid slot");
    if (params.check_res) {
      const bool size_ok =
          2 * params.frame_width >= slot.frame_width &&
          2 * params.frame_height >= slot.frame_height &&
          params.frame_width <= 16 * slot.frame_width &&
          params.frame_height <= 16 * slot.frame_height;
      if (!size_ok) return fail("reference frame size ratio is invalid");
    }
    map.ref_frame_idx[i] = slot_index;
    map.scores_index[i] = slot_index;
    map.scores_order_hint[i] = ref_order_hint(slot);
    // §5.18.2: ScoresDistance[i] = get_relative_dist(OrderHint,
    // RefOrderHint[ref_frame_idx[i]]) for the explicit form (the implicit form
    // produces it inside get_ref_frames()).
    map.scores_distance[i] =
        relative_dist(params.order_hint, ref_order_hint(slot));
    map.scores_score[i] = 0;
  }
  if (!get_past_future_cur_ref_lists(params, &map, error)) return false;
  *out = map;
  return true;
}

bool derive_implicit_ref_map(const RefMapParams& params, ImplicitRefMap* out,
                             std::string* error) {
  if (out == nullptr) return false;
  RefMapParams stage0 = params;
  stage0.check_res = false;
  if (!get_ref_frames(stage0, &out->stage0, error)) return false;
  if (!get_past_future_cur_ref_lists(stage0, &out->stage0, error)) {
    return false;
  }
  RefMapParams stage1 = params;
  stage1.check_res = true;
  if (!get_ref_frames(stage1, &out->stage1, error)) return false;
  if (!get_past_future_cur_ref_lists(stage1, &out->stage1, error)) {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Display order hint
// ---------------------------------------------------------------------------

bool get_max_disp_order_hint(const DispOrderHintParams& params,
                             bool only_showable, int* max_disp,
                             std::string* error) {
  if (max_disp == nullptr) {
    if (error != nullptr) *error = "null output";
    return false;
  }
  if (params.slots == nullptr) {
    if (error != nullptr) *error = "null slot bank";
    return false;
  }
  int max = 0;
  for (int i = 0; i < params.num_ref_frames; ++i) {
    const RefSlot& slot = params.slots[i];
    if (!slot.valid) continue;
    if (!dependency_ok(params.deps, slot.mlayer, slot.tlayer)) continue;
    if (only_showable && !slot.implicit_output && !slot.immediate_output) {
      continue;
    }
    max = std::max(max, ref_order_hint(slot));
  }
  *max_disp = max;
  return true;
}

bool get_disp_order_hint(const DispOrderHintParams& params,
                         int* disp_order_hint, std::string* error) {
  if (disp_order_hint == nullptr) {
    if (error != nullptr) *error = "null output";
    return false;
  }
  if (params.order_hint_bits < 1 || params.order_hint_bits > 16) {
    if (error != nullptr) *error = "OrderHintBits outside 1..16";
    return false;
  }
  if (params.obu_closed_loop_key ||
      (!params.is_sef && params.frame_type_switch &&
       params.restricted_prediction_switch)) {
    *disp_order_hint = params.order_hint_lsbs;
    return true;
  }
  int max_disp = 0;
  if (!get_max_disp_order_hint(params, /*only_showable=*/true, &max_disp,
                               error)) {
    return false;
  }
  int value = params.order_hint_lsbs;
  const int half = (1 << params.order_hint_bits) >> 1;
  const int offset = max_disp - half - params.order_hint_lsbs;
  if (offset >= 0) {
    value += ((offset >> params.order_hint_bits) + 1) << params.order_hint_bits;
  }
  if (value >= (1 << (kDisplayOrderHintBits - 1))) {
    if (error != nullptr) *error = "display order hint exceeds the bound";
    return false;
  }
  *disp_order_hint = value;
  return true;
}

// ---------------------------------------------------------------------------
// R77
// ---------------------------------------------------------------------------

bool validate_r77(std::span<const OperatingPointRefMap> maps, R77Result* out,
                  std::string* error) {
  const auto fail = [&](const char* message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (out == nullptr) return fail("null output");
  *out = R77Result{};
  if (maps.empty()) return fail("no operating points");

  for (const OperatingPointRefMap& m : maps) {
    if (m.num_total_refs < 0 || m.num_total_refs > kRefsPerFrameConst) {
      return fail("NumTotalRefs outside 0..7");
    }
  }

  bool disp_ok = true;
  bool mapping_ok = true;
  bool explicit_ok = true;
  for (std::size_t a = 0; a < maps.size(); ++a) {
    for (std::size_t b = a + 1; b < maps.size(); ++b) {
      const OperatingPointRefMap& x = maps[a];
      const OperatingPointRefMap& y = maps[b];
      if (x.disp_order_hint != y.disp_order_hint) disp_ok = false;

      const bool either_implicit = !x.explicit_map || !y.explicit_map;
      if (either_implicit) {
        if (x.num_total_refs != y.num_total_refs) {
          mapping_ok = false;
        } else {
          for (int i = 0; i < x.num_total_refs; ++i) {
            if (x.ref_identity[i] != y.ref_identity[i]) mapping_ok = false;
          }
        }
      }
      // Explicit: each buffer index must denote the same frame everywhere.
      if (x.explicit_map && y.explicit_map) {
        for (int i = 0; i < x.num_total_refs; ++i) {
          for (int j = 0; j < y.num_total_refs; ++j) {
            if (x.ref_frame_idx[i] == y.ref_frame_idx[j] &&
                x.ref_identity[i] != y.ref_identity[j]) {
              explicit_ok = false;
            }
          }
        }
      }
    }
  }
  out->disp_order_hint_consistent = disp_ok;
  out->mapping_consistent = mapping_ok;
  out->explicit_slot_consistent = explicit_ok;
  if (!disp_ok) {
    return fail("display order hint differs across operating points");
  }
  if (!mapping_ok) {
    return fail("implicit reference mapping differs across operating points");
  }
  if (!explicit_ok) {
    return fail("explicit buffer index names different frames across points");
  }
  return true;
}

// ---------------------------------------------------------------------------
// Frame-header reference-map syntax
// ---------------------------------------------------------------------------

namespace {

bool frame_uses_explicit_map(const FrameRefMapSyntax& gates) {
  if (gates.frame_is_switch || gates.is_bridge) return true;
  if (gates.seq_explicit_ref_frame_map) return gates.explicit_ref_frame_map;
  return false;
}

}  // namespace

bool write_frame_ref_map_syntax(av2enc::BitWriter* writer,
                                const FrameRefMapSyntax& syntax,
                                const RefMapParams& params,
                                const ImplicitRefMap& implicit,
                                std::string* error) {
  if (writer == nullptr) {
    if (error != nullptr) *error = "null BitWriter";
    return false;
  }
  const bool explicit_map = frame_uses_explicit_map(syntax);
  // §5.18.2 codes frame_explicit_ref_frame_map only when the sequence header
  // enables the choice; a SWITCH frame never codes it and always uses the
  // explicit form.
  if (syntax.seq_explicit_ref_frame_map && !syntax.frame_is_switch &&
      !syntax.is_bridge) {
    writer->put_bit(syntax.explicit_ref_frame_map);
  }
  if (!explicit_map) {
    // No map bits.  The mapping is the §7.7 result, which the encoder has
    // already derived.  Re-derive it and fail if the supplied stage-1 map does
    // not match, because the block syntax uses the encoder's copy.
    ImplicitRefMap check;
    if (!derive_implicit_ref_map(params, &check, error)) return false;
    if (check.stage1.num_total_refs != implicit.stage1.num_total_refs) {
      if (error != nullptr) *error = "implicit NumTotalRefs does not match §7.7";
      return false;
    }
    for (int i = 0; i < check.stage1.num_total_refs; ++i) {
      if (check.stage1.ref_frame_idx[i] != implicit.stage1.ref_frame_idx[i]) {
        if (error != nullptr) {
          *error = "implicit ref_frame_idx does not match §7.7";
        }
        return false;
      }
    }
    return true;
  }
  if (syntax.is_bridge) return true;  // NumTotalRefs = 1, index implied
  if (syntax.num_total_refs < 0 || syntax.num_total_refs > kRefsPerFrameConst) {
    if (error != nullptr) *error = "NumTotalRefs outside 0..7";
    return false;
  }
  writer->put_bits(static_cast<std::uint64_t>(syntax.num_total_refs), 3);
  const int index_bits = ceil_log2(static_cast<std::uint32_t>(params.num_ref_frames));
  for (int i = 0; i < syntax.num_total_refs; ++i) {
    if (syntax.ref_frame_idx[i] < 0 ||
        syntax.ref_frame_idx[i] >= params.num_ref_frames) {
      if (error != nullptr) *error = "ref_frame_idx outside NumRefFrames";
      return false;
    }
    writer->put_bits(static_cast<std::uint64_t>(syntax.ref_frame_idx[i]),
                     static_cast<unsigned>(index_bits));
  }
  return true;
}

bool read_frame_ref_map_syntax(BitReader* reader,
                               const FrameRefMapSyntax& gates,
                               const RefMapParams& params,
                               FrameRefMapResult* out, std::string* error) {
  if (reader == nullptr || out == nullptr) {
    if (error != nullptr) *error = "null argument";
    return false;
  }
  FrameRefMapResult result;
  bool explicit_map = false;
  if (gates.frame_is_switch || gates.is_bridge) {
    explicit_map = true;
  } else if (gates.seq_explicit_ref_frame_map) {
    std::uint32_t bit = 0;
    if (!reader->read_bits(1, &bit)) {
      if (error != nullptr) *error = "invalid frame_explicit_ref_frame_map";
      return false;
    }
    explicit_map = bit != 0;
  }
  result.explicit_map = explicit_map;
  if (explicit_map) {
    if (gates.is_bridge) {
      result.map.num_total_refs = 1;
      result.map.ref_frame_idx[0] = gates.bridge_frame_ref_idx;
      if (!derive_explicit_ref_map(params, 1, result.map.ref_frame_idx,
                                   &result.map, error)) {
        return false;
      }
      *out = result;
      return true;
    }
    std::uint32_t num_total_refs = 0;
    if (!reader->read_bits(3, &num_total_refs) ||
        num_total_refs > kRefsPerFrameConst) {
      if (error != nullptr) *error = "invalid num_total_refs";
      return false;
    }
    int indices[kRefsPerFrameConst] = {};
    const int index_bits = ceil_log2(static_cast<std::uint32_t>(params.num_ref_frames));
    for (int i = 0; i < static_cast<int>(num_total_refs); ++i) {
      std::uint32_t index = 0;
      if (!reader->read_bits(index_bits, &index) ||
          static_cast<int>(index) >= params.num_ref_frames) {
        if (error != nullptr) *error = "invalid ref_frame_idx";
        return false;
      }
      indices[i] = static_cast<int>(index);
    }
    // §5.18.2 computes ScoresDistance and then the §7.8 lists for the read
    // explicit map too.
    if (!derive_explicit_ref_map(params, static_cast<int>(num_total_refs),
                                 indices, &result.map, error)) {
      return false;
    }
    *out = result;
    return true;
  }
  ImplicitRefMap implicit;
  if (!derive_implicit_ref_map(params, &implicit, error)) return false;
  result.map = implicit.stage1;
  *out = result;
  return true;
}

}  // namespace av2enc::inter::mv_stack_full
