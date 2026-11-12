// AV2 v1.0.0 TIP frame-level signalling — writer + reader.

#include "codec/inter/tip/tip_signalling.h"

#include <cassert>
#include <cstdlib>

#include "codec/inter/temporal/temporal_math.h"

namespace av2enc::inter::tip {
namespace {

// §4.11.6 ns(n): non-symmetric unsigned coding.
void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
  assert(n >= 2U && value < n);
  unsigned w = 0;
  while ((1ULL << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

bool read_ns(BitReader& reader, std::uint32_t n, std::uint32_t* value) {
  if (n < 2U) {
    *value = 0;
    return true;
  }
  unsigned w = 0;
  while ((1ULL << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  std::uint32_t v = 0;
  if (!reader.get_bits(w - 1U, &v)) return false;
  if (v < m) {
    *value = v;
    return true;
  }
  bool bit = false;
  if (!reader.get_bit(&bit)) return false;
  const std::uint32_t x = (v << 1U) | (bit ? 1U : 0U);
  if (x - m >= n) return false;
  *value = x - m;
  return true;
}

bool refined_type(RefineType type) {
  return type != RefineType::kAuto;
}

// §5.18.3.2 for a TIP_FRAME_AS_OUTPUT frame.
RefineType derived_output_opfl(const TipFrameSyntax& s) {
  return (s.enable_tip_refinemv && s.enable_opfl_refine != RefineType::kNone)
             ? RefineType::kAll
             : RefineType::kNone;
}

RefineType resolved_opfl(const TipFrameSyntax& s) {
  if (s.mode == TipFrameMode::kAsOutput) return derived_output_opfl(s);
  if (s.enable_opfl_refine == RefineType::kAuto) return s.requested_opfl_refine;
  return s.enable_opfl_refine;
}

bool uses_equal_weight(const TipFrameSyntax& s) {
  return s.enable_tip_refinemv && s.num_future_refs > 0 &&
         s.num_past_refs > 0 &&
         (resolved_opfl(s) != RefineType::kNone || s.enable_refinemv);
}

InterpFilter filter_of(bool sharp, bool regular) {
  if (sharp) return InterpFilter::kEighttapSharp;
  return regular ? InterpFilter::kEighttap : InterpFilter::kEighttapSmooth;
}

}  // namespace

// ---------------------------------------------------------------------------
// §5.4.6 sequence slice.
// ---------------------------------------------------------------------------

TemporalStatus write_tip_sequence_syntax(BitWriter& writer,
                                         const TipSequenceSyntax& s) {
  if (s.order_hint_bits_minus_1 < 0 || s.order_hint_bits_minus_1 > 15 ||
      s.num_ref_frames_minus_1 < 0 || s.num_ref_frames_minus_1 > 15 ||
      s.long_term_frame_id_bits < 0 || s.long_term_frame_id_bits > 7 ||
      s.seq_max_drl_bits_minus_1 < 0 || s.seq_max_drl_bits_minus_1 > 5 ||
      s.seq_max_bvp_drl_bits_minus_1 < 0 ||
      s.seq_max_bvp_drl_bits_minus_1 > 3 ||
      s.num_same_ref_compound < 0 || s.num_same_ref_compound > 3) {
    return TemporalStatus::kInvalidArgument;
  }
  const bool tip_output = s.enable_tip_output();
  // Derived/rejected combinations must not be signalled.
  if (!s.enable_tip && (s.disable_tip_output || s.enable_tip_hole_fill))
    return TemporalStatus::kInvalidArgument;
  if (s.enable_tip_explicit_qp && !(tip_output && s.enable_df_sub_pu))
    return TemporalStatus::kInvalidArgument;
  const bool tip_refinemv_present =
      s.enable_tip &&
      (s.enable_opfl_refine != RefineType::kNone || s.enable_refinemv);
  if (s.enable_tip_refinemv && !tip_refinemv_present)
    return TemporalStatus::kInvalidArgument;

  writer.put_bit(s.enable_ref_frame_mvs);
  if (s.enable_ref_frame_mvs) writer.put_bit(s.reduced_ref_frame_mvs_mode);
  writer.put_bits(static_cast<std::uint64_t>(s.order_hint_bits_minus_1), 4);
  writer.put_bit(s.enable_refmvbank);
  writer.put_bit(s.disable_drl_reorder);
  if (!s.disable_drl_reorder) writer.put_bit(s.constrain_drl_reorder);
  writer.put_bit(s.explicit_ref_frame_map);
  writer.put_bit(s.explicit_num_ref_frames);
  if (s.explicit_num_ref_frames) {
    writer.put_bits(static_cast<std::uint64_t>(s.num_ref_frames_minus_1), 4);
  }
  writer.put_bits(static_cast<std::uint64_t>(s.long_term_frame_id_bits), 3);
  put_ns(writer, 5U, static_cast<std::uint32_t>(s.seq_max_drl_bits_minus_1));
  writer.put_bit(s.allow_frame_max_drl_bits);
  put_ns(writer, 3U,
         static_cast<std::uint32_t>(s.seq_max_bvp_drl_bits_minus_1));
  writer.put_bit(s.allow_frame_max_bvp_drl_bits);
  writer.put_bits(static_cast<std::uint64_t>(s.num_same_ref_compound), 2);
  writer.put_bit(s.enable_tip);
  if (s.enable_tip) {
    writer.put_bit(s.disable_tip_output);
    writer.put_bit(s.enable_tip_hole_fill);
  }
  writer.put_bit(s.enable_mv_traj);
  writer.put_bit(s.enable_bawp);
  writer.put_bit(s.enable_cwp);
  writer.put_bit(s.enable_imp_msk_bld);
  writer.put_bit(s.enable_df_sub_pu);
  if (tip_output && s.enable_df_sub_pu) writer.put_bit(s.enable_tip_explicit_qp);
  writer.put_bits(static_cast<std::uint64_t>(s.enable_opfl_refine), 2);
  writer.put_bit(s.enable_refinemv);
  if (tip_refinemv_present) writer.put_bit(s.enable_tip_refinemv);
  return TemporalStatus::kOk;
}

TemporalStatus read_tip_sequence_syntax(BitReader& reader,
                                        TipSequenceSyntax* s) {
  if (s == nullptr) return TemporalStatus::kInvalidArgument;
  TipSequenceSyntax out;
  if (!reader.get_bit(&out.enable_ref_frame_mvs)) {
    return TemporalStatus::kInvalidArgument;
  }
  if (out.enable_ref_frame_mvs) {
    if (!reader.get_bit(&out.reduced_ref_frame_mvs_mode)) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  std::uint32_t value = 0;
  if (!reader.get_bits(4, &value) || value > 15) {
    return TemporalStatus::kInvalidArgument;
  }
  out.order_hint_bits_minus_1 = static_cast<int>(value);
  if (!reader.get_bit(&out.enable_refmvbank) ||
      !reader.get_bit(&out.disable_drl_reorder)) {
    return TemporalStatus::kInvalidArgument;
  }
  out.constrain_drl_reorder = false;
  if (!out.disable_drl_reorder) {
    if (!reader.get_bit(&out.constrain_drl_reorder)) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  if (!reader.get_bit(&out.explicit_ref_frame_map) ||
      !reader.get_bit(&out.explicit_num_ref_frames)) {
    return TemporalStatus::kInvalidArgument;
  }
  out.num_ref_frames_minus_1 = 7;
  if (out.explicit_num_ref_frames) {
    if (!reader.get_bits(4, &value) || value > 15) {
      return TemporalStatus::kInvalidArgument;
    }
    out.num_ref_frames_minus_1 = static_cast<int>(value);
  }
  if (!reader.get_bits(3, &value) || value > 7) {
    return TemporalStatus::kInvalidArgument;
  }
  out.long_term_frame_id_bits = static_cast<int>(value);
  if (!read_ns(reader, 5U, &value) || value > 5) {
    return TemporalStatus::kInvalidArgument;
  }
  out.seq_max_drl_bits_minus_1 = static_cast<int>(value);
  if (!reader.get_bit(&out.allow_frame_max_drl_bits)) {
    return TemporalStatus::kInvalidArgument;
  }
  if (!read_ns(reader, 3U, &value) || value > 3) {
    return TemporalStatus::kInvalidArgument;
  }
  out.seq_max_bvp_drl_bits_minus_1 = static_cast<int>(value);
  if (!reader.get_bit(&out.allow_frame_max_bvp_drl_bits)) {
    return TemporalStatus::kInvalidArgument;
  }
  if (!reader.get_bits(2, &value) || value > 3) {
    return TemporalStatus::kInvalidArgument;
  }
  out.num_same_ref_compound = static_cast<int>(value);
  if (!reader.get_bit(&out.enable_tip)) {
    return TemporalStatus::kInvalidArgument;
  }
  if (out.enable_tip) {
    if (!reader.get_bit(&out.disable_tip_output) ||
        !reader.get_bit(&out.enable_tip_hole_fill)) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  if (!reader.get_bit(&out.enable_mv_traj) ||
      !reader.get_bit(&out.enable_bawp) ||
      !reader.get_bit(&out.enable_cwp) ||
      !reader.get_bit(&out.enable_imp_msk_bld) ||
      !reader.get_bit(&out.enable_df_sub_pu)) {
    return TemporalStatus::kInvalidArgument;
  }
  const bool tip_output = out.enable_tip_output();
  out.enable_tip_explicit_qp = false;
  if (tip_output && out.enable_df_sub_pu) {
    if (!reader.get_bit(&out.enable_tip_explicit_qp)) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  if (!reader.get_bits(2, &value) || value > 3) {
    return TemporalStatus::kInvalidArgument;
  }
  out.enable_opfl_refine = static_cast<RefineType>(value);
  if (!reader.get_bit(&out.enable_refinemv)) {
    return TemporalStatus::kInvalidArgument;
  }
  out.enable_tip_refinemv = false;
  if (out.enable_tip &&
      (out.enable_opfl_refine != RefineType::kNone || out.enable_refinemv)) {
    if (!reader.get_bit(&out.enable_tip_refinemv)) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  *s = out;
  return TemporalStatus::kOk;
}

// ---------------------------------------------------------------------------
// §5.18.3.2 frame_opfl_refine_type().
// ---------------------------------------------------------------------------

TemporalStatus write_tip_opfl_refine_type(BitWriter& writer,
                                          const TipFrameSyntax& s) {
  if (s.mode == TipFrameMode::kAsOutput) return TemporalStatus::kOk;
  if (s.enable_opfl_refine == RefineType::kAuto) {
    if (!refined_type(s.requested_opfl_refine)) {
      return TemporalStatus::kInvalidArgument;
    }
    const bool switchable =
        s.requested_opfl_refine == RefineType::kSwitchable;
    writer.put_bit(switchable);
    if (!switchable) writer.put_bit(s.requested_opfl_refine == RefineType::kAll);
    return TemporalStatus::kOk;
  }
  if (s.requested_opfl_refine != s.enable_opfl_refine) {
    return TemporalStatus::kInvalidArgument;
  }
  return TemporalStatus::kOk;
}

TemporalStatus read_tip_opfl_refine_type(BitReader& reader,
                                         const TipFrameSyntax& s,
                                         RefineType* out) {
  if (out == nullptr) return TemporalStatus::kInvalidArgument;
  if (s.mode == TipFrameMode::kAsOutput) {
    *out = derived_output_opfl(s);
    return TemporalStatus::kOk;
  }
  if (s.enable_opfl_refine == RefineType::kAuto) {
    bool switchable = false;
    if (!reader.get_bit(&switchable)) return TemporalStatus::kInvalidArgument;
    if (switchable) {
      *out = RefineType::kSwitchable;
      return TemporalStatus::kOk;
    }
    bool all = false;
    if (!reader.get_bit(&all)) return TemporalStatus::kInvalidArgument;
    *out = all ? RefineType::kAll : RefineType::kNone;
    return TemporalStatus::kOk;
  }
  *out = s.enable_opfl_refine;
  return TemporalStatus::kOk;
}

// ---------------------------------------------------------------------------
// §5.18.2 frame slice.
// ---------------------------------------------------------------------------

TemporalStatus resolve_tip_frame_syntax(TipFrameSyntax* s) {
  if (s == nullptr || s->num_total_refs < 0 ||
      s->num_total_refs > temporal::kRefsPerFrame) {
    return TemporalStatus::kInvalidArgument;
  }
  if (s->tmvp_sample_step_minus_1 != 0 &&
      s->tmvp_sample_step_minus_1 != 1) {
    return TemporalStatus::kInvalidArgument;
  }
  if (!s->has_tip_block()) {
    if (s->requested_mode != TipFrameMode::kDisabled) {
      return TemporalStatus::kInvalidArgument;
    }
    if (s->is_tip_frame) return TemporalStatus::kInvalidArgument;
    s->mode = TipFrameMode::kDisabled;
    s->allow_tip_hole_fill = false;
    s->opfl_refine_type = resolved_opfl(*s);
    s->interp_filter =
        filter_of(s->tip_sharp, s->tip_regular);
    return TemporalStatus::kOk;
  }
  if (s->is_tip_frame) {
    // §6.20 conformance: a TIP OBU must resolve to TIP_FRAME_AS_OUTPUT.
    if (!s->enable_tip_output) return TemporalStatus::kInvalidArgument;
    s->mode = TipFrameMode::kAsOutput;
  } else {
    if (s->requested_mode == TipFrameMode::kAsOutput) {
      return TemporalStatus::kInvalidArgument;
    }
    s->mode = s->requested_mode;
  }
  s->opfl_refine_type = resolved_opfl(*s);
  if (s->mode == TipFrameMode::kDisabled) {
    s->allow_tip_hole_fill = false;
    s->tip_global_wtd_index = 0;
    s->tip_global_mv = Mv{0, 0};
  } else if (s->enable_tip_hole_fill) {
    // keep the requested allow_tip_hole_fill value
  } else {
    s->allow_tip_hole_fill = false;
  }
  if (uses_equal_weight(*s)) s->tip_global_wtd_index = 0;
  if (s->mode == TipFrameMode::kAsOutput &&
      (s->tip_global_mv.row < -15 || s->tip_global_mv.row > 15 ||
       s->tip_global_mv.col < -15 || s->tip_global_mv.col > 15)) {
    return TemporalStatus::kInvalidArgument;
  }
  s->interp_filter = filter_of(s->tip_sharp, s->tip_regular);
  return TemporalStatus::kOk;
}

TemporalStatus write_tip_frame_syntax(BitWriter& writer,
                                      const TipFrameSyntax& syntax) {
  if (syntax.num_total_refs < 0 ||
      syntax.num_total_refs > temporal::kRefsPerFrame) {
    return TemporalStatus::kInvalidArgument;
  }
  if (syntax.use_ref_frame_mvs && syntax.num_total_refs > 1 &&
      !syntax.sb_64x64) {
    if (syntax.tmvp_sample_step_minus_1 != 0 &&
        syntax.tmvp_sample_step_minus_1 != 1) {
      return TemporalStatus::kInvalidArgument;
    }
    writer.put_bit(syntax.tmvp_sample_step_minus_1 != 0);
  }
  TipFrameSyntax s = syntax;
  const TemporalStatus status = resolve_tip_frame_syntax(&s);
  if (status != TemporalStatus::kOk) return status;
  if (!syntax.has_tip_block()) {
    return write_tip_opfl_refine_type(writer, s);
  }
  if (syntax.is_tip_frame) {
    // Mode is inferred from the OBU type; no tip_frame_mode bit.
  } else {
    writer.put_bit(s.mode == TipFrameMode::kAsReference);
  }
  const TemporalStatus opfl = write_tip_opfl_refine_type(writer, s);
  if (opfl != TemporalStatus::kOk) return opfl;
  if (s.mode != TipFrameMode::kDisabled && syntax.enable_tip_hole_fill) {
    writer.put_bit(syntax.allow_tip_hole_fill);
  }
  if (s.mode != TipFrameMode::kDisabled && !uses_equal_weight(s)) {
    if (syntax.tip_global_wtd_index < 0 ||
        syntax.tip_global_wtd_index > 7) {
      return TemporalStatus::kInvalidArgument;
    }
    writer.put_bits(static_cast<std::uint64_t>(syntax.tip_global_wtd_index), 3);
  }
  if (s.mode == TipFrameMode::kAsOutput) {
    const bool zero =
        syntax.tip_global_mv.row == 0 && syntax.tip_global_mv.col == 0;
    writer.put_bit(zero);
    if (!zero) {
      const int row = syntax.tip_global_mv.row;
      const int col = syntax.tip_global_mv.col;
      if (std::abs(row) > 15 || std::abs(col) > 15) {
        return TemporalStatus::kInvalidArgument;
      }
      writer.put_bits(static_cast<std::uint64_t>(std::abs(row)), 4);
      writer.put_bits(static_cast<std::uint64_t>(std::abs(col)), 4);
      if (row != 0) writer.put_bit(row < 0);
      if (col != 0) writer.put_bit(col < 0);
    }
  }
  writer.put_bit(syntax.tip_sharp);
  if (!syntax.tip_sharp) writer.put_bit(syntax.tip_regular);
  return TemporalStatus::kOk;
}

TemporalStatus read_tip_frame_syntax(BitReader& reader,
                                     TipFrameSyntax* syntax) {
  if (syntax == nullptr || syntax->num_total_refs < 0 ||
      syntax->num_total_refs > temporal::kRefsPerFrame) {
    return TemporalStatus::kInvalidArgument;
  }
  TipFrameSyntax s = *syntax;
  s.tmvp_sample_step_minus_1 = 0;
  if (s.use_ref_frame_mvs && s.num_total_refs > 1 && !s.sb_64x64) {
    bool bit = false;
    if (!reader.get_bit(&bit)) return TemporalStatus::kInvalidArgument;
    s.tmvp_sample_step_minus_1 = bit ? 1 : 0;
  }
  s.requested_mode = TipFrameMode::kDisabled;
  s.tip_global_wtd_index = 0;
  s.tip_global_mv = Mv{0, 0};
  s.allow_tip_hole_fill = false;
  if (!s.has_tip_block()) {
    if (s.is_tip_frame) return TemporalStatus::kInvalidArgument;
    s.mode = TipFrameMode::kDisabled;
    const TemporalStatus status =
        read_tip_opfl_refine_type(reader, s, &s.opfl_refine_type);
    if (status != TemporalStatus::kOk) return status;
    s.interp_filter = filter_of(s.tip_sharp, s.tip_regular);
    *syntax = s;
    return TemporalStatus::kOk;
  }
  if (s.is_tip_frame) {
    if (!s.enable_tip_output) return TemporalStatus::kInvalidArgument;
    s.mode = TipFrameMode::kAsOutput;
  } else {
    bool mode_bit = false;
    if (!reader.get_bit(&mode_bit)) return TemporalStatus::kInvalidArgument;
    s.mode = mode_bit ? TipFrameMode::kAsReference : TipFrameMode::kDisabled;
  }
  const TemporalStatus opfl =
      read_tip_opfl_refine_type(reader, s, &s.opfl_refine_type);
  if (opfl != TemporalStatus::kOk) return opfl;
  if (s.mode != TipFrameMode::kDisabled && s.enable_tip_hole_fill) {
    if (!reader.get_bit(&s.allow_tip_hole_fill)) {
      return TemporalStatus::kInvalidArgument;
    }
  } else {
    s.allow_tip_hole_fill = false;
  }
  if (s.mode != TipFrameMode::kDisabled && !uses_equal_weight(s)) {
    std::uint32_t index = 0;
    if (!reader.get_bits(3, &index) || index > 7) {
      return TemporalStatus::kInvalidArgument;
    }
    s.tip_global_wtd_index = static_cast<int>(index);
  } else {
    s.tip_global_wtd_index = 0;
  }
  s.tip_global_mv = Mv{0, 0};
  if (s.mode == TipFrameMode::kAsOutput) {
    bool zero = false;
    if (!reader.get_bit(&zero)) return TemporalStatus::kInvalidArgument;
    if (!zero) {
      std::uint32_t row = 0;
      std::uint32_t col = 0;
      if (!reader.get_bits(4, &row) || !reader.get_bits(4, &col)) {
        return TemporalStatus::kInvalidArgument;
      }
      int signed_row = static_cast<int>(row);
      int signed_col = static_cast<int>(col);
      if (row != 0) {
        bool sign = false;
        if (!reader.get_bit(&sign)) return TemporalStatus::kInvalidArgument;
        if (sign) signed_row = -signed_row;
      }
      if (col != 0) {
        bool sign = false;
        if (!reader.get_bit(&sign)) return TemporalStatus::kInvalidArgument;
        if (sign) signed_col = -signed_col;
      }
      s.tip_global_mv = Mv{signed_row, signed_col};
    }
  }
  if (!reader.get_bit(&s.tip_sharp)) return TemporalStatus::kInvalidArgument;
  s.tip_regular = false;
  if (!s.tip_sharp) {
    if (!reader.get_bit(&s.tip_regular)) return TemporalStatus::kInvalidArgument;
  }
  s.interp_filter = filter_of(s.tip_sharp, s.tip_regular);
  *syntax = s;
  return TemporalStatus::kOk;
}

// ---------------------------------------------------------------------------
// §7.10.1 setup-TIP conformance.
// ---------------------------------------------------------------------------

TipSetupConditions make_tip_setup_conditions(
    bool switch_frame, const TipFrameSyntax& frame, int closest_past,
    int closest_future, bool past_type_eligible, bool future_type_eligible,
    bool past_size_eligible, bool future_size_eligible) {
  TipSetupConditions out;
  out.switch_frame = switch_frame;
  out.closest_past_present = closest_past >= 0;
  out.closest_future_present = closest_future >= 0;
  out.mode = frame.mode;
  out.use_ref_frame_mvs = frame.use_ref_frame_mvs;
  out.has_both_refs = out.closest_past_present && out.closest_future_present;
  out.num_past_refs = frame.num_past_refs;
  out.past_type_eligible = past_type_eligible;
  out.future_type_eligible = future_type_eligible;
  out.past_size_eligible = past_size_eligible;
  out.future_size_eligible = future_size_eligible;
  return out;
}

TemporalStatus check_tip_setup_conditions(const TipSetupConditions& c,
                                          const char** reason) {
  const char* why = nullptr;
  if (c.switch_frame) {
    why = "switch frames cannot run setup TIP";
  } else if (!c.closest_past_present || !c.closest_future_present) {
    why = "ClosestPast and ClosestFuture are both required";
  } else if (c.mode == TipFrameMode::kDisabled) {
    why = "TipFrameMode is TIP_FRAME_DISABLED";
  } else if (!c.use_ref_frame_mvs) {
    why = "use_ref_frame_mvs is 0";
  } else if (!c.has_both_refs && c.num_past_refs < 2) {
    why = "need HasBothRefs or at least two past refs";
  } else if (!c.past_type_eligible && !c.future_type_eligible) {
    why = "neither closest ref is motion-field eligible by frame type";
  } else if (!c.past_size_eligible || !c.future_size_eligible) {
    why = "closest refs must match the frame size";
  }
  if (reason != nullptr) *reason = why;
  return why == nullptr ? TemporalStatus::kOk
                        : TemporalStatus::kInvalidArgument;
}

// ---------------------------------------------------------------------------
// OBU framing.
// ---------------------------------------------------------------------------

void finish_tip_obu_payload(BitWriter* writer) {
  assert(writer != nullptr);
  writer->put_bit(true);  // trailing_one_bit
  writer->align_to_byte();
}

std::vector<std::uint8_t> make_tip_frame_obu(TipObuType type,
                                             BitWriter* writer) {
  assert(writer != nullptr && is_tip_obu_type(type));
  finish_tip_obu_payload(writer);
  return av2enc::basic::make_obu(av2enc::basic::ObuHeader{tip_obu_type(type)}, writer->take_bytes());
}

std::vector<std::uint8_t> make_tip_slice_bytes(
    const TipFrameSyntax& syntax, TemporalStatus* status) {
  BitWriter writer;
  const TemporalStatus s = write_tip_frame_syntax(writer, syntax);
  if (status != nullptr) *status = s;
  if (s != TemporalStatus::kOk) return {};
  finish_tip_obu_payload(&writer);
  return writer.take_bytes();
}

}  // namespace av2enc::inter::tip
