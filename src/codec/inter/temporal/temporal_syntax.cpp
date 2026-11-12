// AV2 v1.0.0 temporal inter tools — syntax read/write.

#include "codec/inter/temporal/temporal_syntax.h"

#include <algorithm>

#include "codec/inter/temporal/refinemv.h"
#include "codec/inter/temporal/temporal_math.h"

namespace av2enc::inter::temporal {
namespace {

RefineType derived_opfl_type(const TemporalFrameConfig& cfg) {
  if (cfg.tip.mode == TipFrameMode::kAsOutput) {
    return (cfg.enable_tip_refinemv &&
            cfg.enable_opfl_refine != RefineType::kNone)
               ? RefineType::kAll
               : RefineType::kNone;
  }
  if (cfg.enable_opfl_refine == RefineType::kAuto) {
    return cfg.opfl_refine_type;
  }
  return cfg.enable_opfl_refine;
}

bool equal_weight(const TemporalFrameConfig& cfg) {
  return cfg.enable_tip_refinemv && cfg.num_future_refs > 0 &&
         cfg.num_past_refs > 0 &&
         (derived_opfl_type(cfg) != RefineType::kNone || cfg.enable_refinemv);
}

}  // namespace

bool BitReader::get_bit(bool* bit) {
  if (bit == nullptr || !ok_ || bit_pos_ >= size_ * 8) {
    ok_ = false;
    return false;
  }
  const std::size_t byte = bit_pos_ >> 3;
  const int shift = 7 - static_cast<int>(bit_pos_ & 7);
  *bit = ((data_[byte] >> shift) & 1) != 0;
  ++bit_pos_;
  return true;
}

bool BitReader::get_bits(unsigned num_bits, std::uint32_t* value) {
  if (value == nullptr || num_bits > 32) {
    ok_ = false;
    return false;
  }
  std::uint32_t result = 0;
  for (unsigned i = 0; i < num_bits; ++i) {
    bool bit = false;
    if (!get_bit(&bit)) {
      return false;
    }
    result = (result << 1) | (bit ? 1u : 0u);
  }
  *value = result;
  return true;
}

TemporalStatus write_sequence_temporal_syntax(BitWriter& writer,
                                              const TemporalSequenceConfig& cfg) {
  writer.put_bit(cfg.enable_ref_frame_mvs);
  if (cfg.enable_ref_frame_mvs) {
    writer.put_bit(cfg.reduced_ref_frame_mvs_mode);
  }
  writer.put_bit(cfg.enable_refmvbank);
  writer.put_bit(cfg.enable_bawp);
  writer.put_bit(cfg.enable_cwp);
  writer.put_bit(cfg.enable_imp_msk_bld);
  writer.put_bit(cfg.enable_df_sub_pu);
  writer.put_bit(cfg.enable_tip);
  if (cfg.enable_tip) {
    writer.put_bit(cfg.disable_tip_output);
    writer.put_bit(cfg.enable_tip_hole_fill);
  }
  writer.put_bit(cfg.enable_mv_traj);
  const bool enable_tip_output = cfg.enable_tip && !cfg.disable_tip_output;
  if (enable_tip_output && cfg.enable_df_sub_pu) {
    writer.put_bit(cfg.enable_tip_explicit_qp);
  }
  writer.put_bits(static_cast<std::uint64_t>(cfg.enable_opfl_refine), 2);
  writer.put_bit(cfg.enable_refinemv);
  if (cfg.enable_tip &&
      (cfg.enable_opfl_refine != RefineType::kNone || cfg.enable_refinemv)) {
    writer.put_bit(cfg.enable_tip_refinemv);
  }
  return TemporalStatus::kOk;
}

TemporalStatus read_sequence_temporal_syntax(BitReader& reader,
                                             TemporalSequenceConfig* cfg) {
  if (cfg == nullptr) {
    return TemporalStatus::kInvalidArgument;
  }
  bool bit = false;
  if (!reader.get_bit(&bit)) {
    return TemporalStatus::kInvalidArgument;
  }
  cfg->enable_ref_frame_mvs = bit;
  cfg->reduced_ref_frame_mvs_mode = false;
  if (cfg->enable_ref_frame_mvs) {
    if (!reader.get_bit(&cfg->reduced_ref_frame_mvs_mode)) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  if (!reader.get_bit(&cfg->enable_refmvbank) ||
      !reader.get_bit(&cfg->enable_bawp) ||
      !reader.get_bit(&cfg->enable_cwp) ||
      !reader.get_bit(&cfg->enable_imp_msk_bld) ||
      !reader.get_bit(&cfg->enable_df_sub_pu) ||
      !reader.get_bit(&cfg->enable_tip)) {
    return TemporalStatus::kInvalidArgument;
  }
  cfg->disable_tip_output = false;
  cfg->enable_tip_hole_fill = false;
  if (cfg->enable_tip) {
    if (!reader.get_bit(&cfg->disable_tip_output) ||
        !reader.get_bit(&cfg->enable_tip_hole_fill)) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  if (!reader.get_bit(&cfg->enable_mv_traj)) {
    return TemporalStatus::kInvalidArgument;
  }
  cfg->enable_tip_explicit_qp = false;
  const bool enable_tip_output = cfg->enable_tip && !cfg->disable_tip_output;
  if (enable_tip_output && cfg->enable_df_sub_pu) {
    if (!reader.get_bit(&cfg->enable_tip_explicit_qp)) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  std::uint32_t opfl = 0;
  if (!reader.get_bits(2, &opfl) || opfl > 3) {
    return TemporalStatus::kInvalidArgument;
  }
  cfg->enable_opfl_refine = static_cast<RefineType>(opfl);
  if (!reader.get_bit(&cfg->enable_refinemv)) {
    return TemporalStatus::kInvalidArgument;
  }
  cfg->enable_tip_refinemv = false;
  if (cfg->enable_tip &&
      (cfg->enable_opfl_refine != RefineType::kNone || cfg->enable_refinemv)) {
    if (!reader.get_bit(&cfg->enable_tip_refinemv)) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  return TemporalStatus::kOk;
}

TemporalStatus write_opfl_refine_type(BitWriter& writer,
                                      const TemporalFrameConfig& cfg) {
  if (cfg.tip.mode == TipFrameMode::kAsOutput) {
    return TemporalStatus::kOk;  // derived, no bits
  }
  if (cfg.enable_opfl_refine == RefineType::kAuto) {
    if (cfg.opfl_refine_type == RefineType::kAuto) {
      return TemporalStatus::kInvalidArgument;
    }
    const bool switchable = cfg.opfl_refine_type == RefineType::kSwitchable;
    writer.put_bit(switchable);
    if (!switchable) {
      writer.put_bit(cfg.opfl_refine_type == RefineType::kAll);
    }
    return TemporalStatus::kOk;
  }
  if (cfg.enable_opfl_refine != cfg.opfl_refine_type) {
    return TemporalStatus::kInvalidArgument;
  }
  return TemporalStatus::kOk;
}

TemporalStatus read_opfl_refine_type(BitReader& reader,
                                     const TemporalFrameConfig& cfg,
                                     RefineType* out) {
  if (out == nullptr) {
    return TemporalStatus::kInvalidArgument;
  }
  if (cfg.tip.mode == TipFrameMode::kAsOutput) {
    *out = (cfg.enable_tip_refinemv &&
            cfg.enable_opfl_refine != RefineType::kNone)
               ? RefineType::kAll
               : RefineType::kNone;
    return TemporalStatus::kOk;
  }
  if (cfg.enable_opfl_refine == RefineType::kAuto) {
    bool switchable = false;
    if (!reader.get_bit(&switchable)) {
      return TemporalStatus::kInvalidArgument;
    }
    if (switchable) {
      *out = RefineType::kSwitchable;
      return TemporalStatus::kOk;
    }
    bool all = false;
    if (!reader.get_bit(&all)) {
      return TemporalStatus::kInvalidArgument;
    }
    *out = all ? RefineType::kAll : RefineType::kNone;
    return TemporalStatus::kOk;
  }
  *out = cfg.enable_opfl_refine;
  return TemporalStatus::kOk;
}

TemporalStatus write_frame_temporal_syntax(BitWriter& writer,
                                           const TemporalFrameConfig& cfg) {
  if (cfg.num_total_refs < 0 || cfg.num_total_refs > kRefsPerFrame) {
    return TemporalStatus::kInvalidArgument;
  }
  if (cfg.use_ref_frame_mvs && cfg.num_total_refs > 1 && !cfg.sb_64x64) {
    writer.put_bit(cfg.tmvp_sample_step_minus_1 != 0);
  }
  TemporalFrameConfig effective = cfg;
  const bool tip_block =
      cfg.enable_tip && cfg.use_ref_frame_mvs && cfg.num_total_refs >= 2;
  if (!tip_block) {
    if (cfg.tip.mode != TipFrameMode::kDisabled) {
      return TemporalStatus::kInvalidArgument;
    }
    effective.tip.mode = TipFrameMode::kDisabled;
    return write_opfl_refine_type(writer, effective);
  }
  if (cfg.tip.mode == TipFrameMode::kAsOutput &&
      !(cfg.enable_tip_output && cfg.is_tip_frame)) {
    return TemporalStatus::kInvalidArgument;
  }
  if (cfg.enable_tip_output && cfg.is_tip_frame) {
    effective.tip.mode = TipFrameMode::kAsOutput;
  } else {
    writer.put_bit(cfg.tip.mode == TipFrameMode::kAsReference);
    if (cfg.tip.mode == TipFrameMode::kAsOutput) {
      effective.tip.mode = TipFrameMode::kAsOutput;
    }
  }
  const TemporalStatus opfl_status =
      write_opfl_refine_type(writer, effective);
  if (opfl_status != TemporalStatus::kOk) {
    return opfl_status;
  }
  effective.opfl_refine_type = derived_opfl_type(effective);
  if (effective.tip.mode != TipFrameMode::kDisabled &&
      cfg.enable_tip_hole_fill) {
    writer.put_bit(cfg.tip.allow_tip_hole_fill);
  }
  const bool uses_equal = equal_weight(effective);
  if (effective.tip.mode != TipFrameMode::kDisabled && !uses_equal) {
    if (cfg.tip.tip_global_wtd_index < 0 ||
        cfg.tip.tip_global_wtd_index > 7) {
      return TemporalStatus::kInvalidArgument;
    }
    writer.put_bits(static_cast<std::uint64_t>(cfg.tip.tip_global_wtd_index),
                    3);
  }
  if (effective.tip.mode == TipFrameMode::kAsOutput) {
    const bool zero =
        cfg.tip.tip_global_mv.row == 0 && cfg.tip.tip_global_mv.col == 0;
    writer.put_bit(zero);
    if (!zero) {
      const int row = cfg.tip.tip_global_mv.row;
      const int col = cfg.tip.tip_global_mv.col;
      if (abs_int(row) > 15 || abs_int(col) > 15) {
        return TemporalStatus::kInvalidArgument;
      }
      writer.put_bits(static_cast<std::uint64_t>(abs_int(row)), 4);
      writer.put_bits(static_cast<std::uint64_t>(abs_int(col)), 4);
      if (row != 0) {
        writer.put_bit(row < 0);
      }
      if (col != 0) {
        writer.put_bit(col < 0);
      }
    }
  }
  writer.put_bit(cfg.tip_sharp);
  if (!cfg.tip_sharp) {
    writer.put_bit(cfg.tip_regular);
  }
  return TemporalStatus::kOk;
}

TemporalStatus read_frame_temporal_syntax(BitReader& reader,
                                          TemporalFrameConfig* cfg) {
  if (cfg == nullptr || cfg->num_total_refs < 0 ||
      cfg->num_total_refs > kRefsPerFrame) {
    return TemporalStatus::kInvalidArgument;
  }
  cfg->tmvp_sample_step_minus_1 = 0;
  if (cfg->use_ref_frame_mvs && cfg->num_total_refs > 1 && !cfg->sb_64x64) {
    bool bit = false;
    if (!reader.get_bit(&bit)) {
      return TemporalStatus::kInvalidArgument;
    }
    cfg->tmvp_sample_step_minus_1 = bit ? 1 : 0;
  }
  cfg->tip = TipConfig{};
  cfg->tip.mode = TipFrameMode::kDisabled;
  const bool tip_block =
      cfg->enable_tip && cfg->use_ref_frame_mvs && cfg->num_total_refs >= 2;
  if (!tip_block) {
    return read_opfl_refine_type(reader, *cfg, &cfg->opfl_refine_type);
  }
  if (cfg->enable_tip_output && cfg->is_tip_frame) {
    cfg->tip.mode = TipFrameMode::kAsOutput;
  } else {
    bool mode_bit = false;
    if (!reader.get_bit(&mode_bit)) {
      return TemporalStatus::kInvalidArgument;
    }
    cfg->tip.mode =
        mode_bit ? TipFrameMode::kAsReference : TipFrameMode::kDisabled;
  }
  TemporalFrameConfig for_opfl = *cfg;
  for_opfl.tip.mode = cfg->tip.mode;
  const TemporalStatus opfl_status =
      read_opfl_refine_type(reader, for_opfl, &cfg->opfl_refine_type);
  if (opfl_status != TemporalStatus::kOk) {
    return opfl_status;
  }
  cfg->tip.allow_tip_hole_fill = false;
  if (cfg->tip.mode != TipFrameMode::kDisabled && cfg->enable_tip_hole_fill) {
    if (!reader.get_bit(&cfg->tip.allow_tip_hole_fill)) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  TemporalFrameConfig effective = *cfg;
  effective.opfl_refine_type = cfg->opfl_refine_type;
  const bool uses_equal = equal_weight(effective);
  cfg->tip.tip_global_wtd_index = 0;
  if (cfg->tip.mode != TipFrameMode::kDisabled && !uses_equal) {
    std::uint32_t index = 0;
    if (!reader.get_bits(3, &index) || index > 7) {
      return TemporalStatus::kInvalidArgument;
    }
    cfg->tip.tip_global_wtd_index = static_cast<int>(index);
  }
  cfg->tip.tip_global_mv = Mv{0, 0};
  if (cfg->tip.mode == TipFrameMode::kAsOutput) {
    bool zero = false;
    if (!reader.get_bit(&zero)) {
      return TemporalStatus::kInvalidArgument;
    }
    if (!zero) {
      std::uint32_t row = 0;
      std::uint32_t col = 0;
      if (!reader.get_bits(4, &row) || !reader.get_bits(4, &col)) {
        return TemporalStatus::kInvalidArgument;
      }
      int srow = static_cast<int>(row);
      int scol = static_cast<int>(col);
      if (row != 0) {
        bool sign = false;
        if (!reader.get_bit(&sign)) {
          return TemporalStatus::kInvalidArgument;
        }
        if (sign) {
          srow = -srow;
        }
      }
      if (col != 0) {
        bool sign = false;
        if (!reader.get_bit(&sign)) {
          return TemporalStatus::kInvalidArgument;
        }
        if (sign) {
          scol = -scol;
        }
      }
      cfg->tip.tip_global_mv = Mv{srow, scol};
    }
  }
  if (!reader.get_bit(&cfg->tip_sharp)) {
    return TemporalStatus::kInvalidArgument;
  }
  cfg->tip_regular = false;
  if (!cfg->tip_sharp) {
    if (!reader.get_bit(&cfg->tip_regular)) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  cfg->tip.interp_filter =
      cfg->tip_sharp
          ? InterpFilter::kEighttapSharp
          : (cfg->tip_regular ? InterpFilter::kEighttap
                              : InterpFilter::kEighttapSmooth);
  cfg->tip.enable_tip_refinemv = cfg->enable_tip_refinemv;
  cfg->tip.enable_refinemv = cfg->enable_refinemv;
  cfg->tip.opfl_refine_type = cfg->opfl_refine_type;
  cfg->tip.num_past_refs = cfg->num_past_refs;
  cfg->tip.num_future_refs = cfg->num_future_refs;
  return TemporalStatus::kOk;
}

TemporalStatus write_tip_mode_syntax(BitWriter& writer,
                                     const TipBlockSyntaxConfig& cfg) {
  const bool present =
      cfg.tip_frame_mode != TipFrameMode::kDisabled && !cfg.skip_mode &&
      std::min(cfg.bw4, cfg.bh4) >= 2 && cfg.mi_size_matches_chroma;
  if (present) {
    writer.put_bit(cfg.tip_mode);
  }
  return TemporalStatus::kOk;
}

TemporalStatus read_tip_mode_syntax(BitReader& reader,
                                    const TipBlockSyntaxConfig& cfg,
                                    TipBlockSyntaxConfig* out) {
  if (out == nullptr) {
    return TemporalStatus::kInvalidArgument;
  }
  *out = cfg;
  out->tip_mode = false;
  const bool present =
      cfg.tip_frame_mode != TipFrameMode::kDisabled && !cfg.skip_mode &&
      std::min(cfg.bw4, cfg.bh4) >= 2 && cfg.mi_size_matches_chroma;
  if (present) {
    if (!reader.get_bit(&out->tip_mode)) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  return TemporalStatus::kOk;
}

TemporalStatus write_tip_pred_mode_syntax(BitWriter& writer, int tip_pred_mode) {
  if (tip_pred_mode < 0 || tip_pred_mode > 1) {
    return TemporalStatus::kInvalidArgument;
  }
  writer.put_bit(tip_pred_mode != 0);
  return TemporalStatus::kOk;
}

TemporalStatus read_tip_pred_mode_syntax(BitReader& reader,
                                         int* tip_pred_mode) {
  if (tip_pred_mode == nullptr) {
    return TemporalStatus::kInvalidArgument;
  }
  bool bit = false;
  if (!reader.get_bit(&bit)) {
    return TemporalStatus::kInvalidArgument;
  }
  *tip_pred_mode = bit ? 1 : 0;
  return TemporalStatus::kOk;
}

TemporalStatus write_tip_block_syntax(BitWriter& writer,
                                      const TipBlockSyntaxConfig& cfg) {
  const TemporalStatus s = write_tip_mode_syntax(writer, cfg);
  if (s != TemporalStatus::kOk) {
    return s;
  }
  if (cfg.tip_mode) {
    return write_tip_pred_mode_syntax(writer, cfg.tip_pred_mode);
  }
  return TemporalStatus::kOk;
}

TemporalStatus read_tip_block_syntax(BitReader& reader,
                                     const TipBlockSyntaxConfig& cfg,
                                     TipBlockSyntaxConfig* out) {
  if (out == nullptr) {
    return TemporalStatus::kInvalidArgument;
  }
  TemporalStatus s = read_tip_mode_syntax(reader, cfg, out);
  if (s != TemporalStatus::kOk) {
    return s;
  }
  out->tip_pred_mode = 0;
  if (out->tip_mode) {
    s = read_tip_pred_mode_syntax(reader, &out->tip_pred_mode);
  }
  return s;
}

TemporalStatus write_refinemv_block_syntax(BitWriter& writer,
                                           const RefineMvBlockSyntaxConfig& cfg) {
  RefineMvSyntax syntax;
  const TemporalStatus s = refine_mv_syntax(cfg.ctx, &syntax);
  if (s != TemporalStatus::kOk) {
    return s;
  }
  if (syntax.switchable) {
    writer.put_bit(syntax.use_refinemv);
  }
  return TemporalStatus::kOk;
}

TemporalStatus read_refinemv_block_syntax(BitReader& reader,
                                          const RefineMvConfig& ctx,
                                          RefineMvBlockSyntaxConfig* out) {
  if (out == nullptr) {
    return TemporalStatus::kInvalidArgument;
  }
  out->ctx = ctx;
  RefineMvSyntax syntax;
  const TemporalStatus s = refine_mv_syntax(ctx, &syntax);
  if (s != TemporalStatus::kOk) {
    return s;
  }
  if (syntax.switchable) {
    if (!reader.get_bit(&syntax.use_refinemv)) {
      return TemporalStatus::kInvalidArgument;
    }
  }
  out->syntax = syntax;
  return TemporalStatus::kOk;
}

}  // namespace av2enc::inter::temporal
