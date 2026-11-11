// AV2 M2 loop-restoration adapters — implementation.

#include "encoder/m2/restoration/m2_restoration.h"

#include "entropy/av2/av2_entropy.h"

#include <algorithm>

namespace av2enc::m2::restoration {
namespace rest = ::av2enc::restoration;
namespace {

bool fail(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

bool valid_num_filter_classes(int n) {
  for (int i = 0; i < 8; ++i) {
    if (rest::kDecodeNumFilterClasses[i] == n) return true;
  }
  return false;
}

int ceil_div(int a, int b) { return b > 0 ? (a + b - 1) / b : 0; }

}  // namespace

// ---------------------------------------------------------------------------
// Tools / sequence row
// ---------------------------------------------------------------------------

bool validate_tools(const M2RestorationTools& tools, std::string* error) {
  if (!tools.ccso && !tools.wiener_ns) {
    return true;
  }
  // ccso_unit_matches_sb_size = 0 is legal; §5.18.7.12 then derives 6/7/8
  // from the tile start OR.  M2's 64x64 planner leaves it at 1.
  if (tools.wiener_ns) {
    if (!valid_num_filter_classes(tools.num_filter_classes)) {
      return fail(error,
                  "num_filter_classes is not in Decode_Num_Filter_Classes");
    }
    if (tools.num_filter_classes > 1 && !tools.frame_filters) {
      return fail(error,
                  "num_filter_classes > 1 requires frame-level filters");
    }
  }
  return true;
}

fs::SequenceFilterConfig make_sequence_filter_config(
    const M2RestorationTools& tools, std::string* error) {
  fs::SequenceFilterConfig config;
  config.disable_loopfilters_across_tiles = false;
  config.enable_cdef = false;
  config.enable_gdf = false;
  config.enable_restoration = tools.wiener_ns;
  // PC-Wiener is blocked until §7.20.4 lookup tables are available.
  config.lr_disable_luma_pc_wiener = true;
  config.lr_disable_luma_wiener_ns = false;
  config.lr_uv_present = tools.chroma_lr;
  config.lr_disable_chroma_wiener_ns = false;
  config.enable_ccso = tools.ccso;
  config.ccso_unit_matches_sb_size = tools.ccso_unit_matches_sb_size;
  config.cdef_on_skip_txfm_always_on = false;
  config.cdef_on_skip_txfm_disabled = true;
  config.df_par_bits_minus_2 = 0;
  if (!tools.wiener_ns && tools.chroma_lr) {
    fail(error, "chroma_lr requires wiener_ns");
  }
  return config;
}

std::size_t write_sequence_filter_config(BitWriter& writer,
                                         const M2RestorationTools& tools,
                                         std::string* error) {
  const fs::SequenceFilterConfig config =
      make_sequence_filter_config(tools, error);
  if (error != nullptr && !error->empty()) return 0;
  return fs::write_sequence_filter_config(
      writer, config, /*single_picture_header_flag=*/false, error);
}

bool read_sequence_filter_config(fs::BitReader& reader, M2RestorationTools* tools,
                                 std::string* error) {
  if (tools == nullptr) return fail(error, "null M2RestorationTools");
  fs::SequenceFilterConfig config;
  if (!fs::read_sequence_filter_config(
          reader, /*single_picture_header_flag=*/false,
          /*sb_64x64=*/true, &config, error)) {
    return false;
  }
  tools->ccso = config.enable_ccso;
  tools->ccso_unit_matches_sb_size = config.ccso_unit_matches_sb_size;
  tools->wiener_ns = config.enable_restoration;
  tools->frame_filters = false;  // frame-local decision, not a sequence flag
  tools->num_filter_classes = 1;
  tools->chroma_lr = config.lr_uv_present;
  return true;
}

// ---------------------------------------------------------------------------
// Frame header
// ---------------------------------------------------------------------------

fs::LrHeaderContext make_lr_header_context(const M2RestorationTools& tools,
                                            int sb_size, int base_q_idx,
                                            bool frame_is_intra) {
  fs::LrHeaderContext context;
  context.coded_lossless = false;
  context.enable_restoration = tools.wiener_ns;
  context.single_picture_header_flag = false;
  context.frame_is_intra = frame_is_intra;
  context.sb_size = sb_size;
  context.num_planes = 3;
  context.sub_x = 1;
  context.sub_y = 1;
  context.base_q_idx = base_q_idx;
  context.lr_tools_disable[0][rest::kRestorePcWiener] = true;
  context.lr_tools_disable[0][rest::kRestoreWienerNs] = false;
  context.lr_tools_disable[1][rest::kRestorePcWiener] = true;
  context.lr_tools_disable[1][rest::kRestoreWienerNs] = false;
  return context;
}

fs::CcsoHeaderContext make_ccso_header_context(const M2RestorationTools& tools,
                                               int sb_size, bool frame_is_intra,
                                               int tile_start_or) {
  fs::CcsoHeaderContext context;
  context.coded_lossless = false;
  context.enable_ccso = tools.ccso;
  context.single_picture_header_flag = false;
  context.frame_is_intra = frame_is_intra;
  context.num_planes = 3;
  context.sb_size = sb_size;
  context.ccso_unit_matches_sb_size = tools.ccso_unit_matches_sb_size;
  context.tile_start_or = tile_start_or;
  return context;
}

std::size_t write_lr_params(BitWriter& writer, fs::LrParams& params,
                            const fs::LrHeaderContext& context,
                            std::string* error) {
  return fs::write_lr_params(writer, params, context, error);
}

bool read_lr_params(fs::BitReader& reader, const fs::LrHeaderContext& context,
                    fs::LrParams* params, std::string* error) {
  return fs::read_lr_params(reader, context, params, error);
}

std::size_t write_ccso_params(BitWriter& writer, fs::CcsoParams& params,
                              const fs::CcsoHeaderContext& context,
                              std::string* error) {
  return fs::write_ccso_params(writer, params, context, error);
}

bool read_ccso_params(fs::BitReader& reader, const fs::CcsoHeaderContext& context,
                      fs::CcsoParams* params, std::string* error) {
  return fs::read_ccso_params(reader, context, params, error);
}

bool make_ccso_frame_params(const fs::CcsoParams& params,
                            fs::CcsoTileState* state, int mi_cols, int mi_rows,
                            int sub_x, int sub_y,
                            rest::CcsoFrameParams* out,
                            std::string* error) {
  if (state == nullptr || out == nullptr) {
    return fail(error, "null CCSO state/params");
  }
  if (mi_cols <= 0 || mi_rows <= 0) {
    return fail(error, "invalid CCSO geometry");
  }
  *out = rest::CcsoFrameParams{};
  out->ccso_luma_size_log2 = params.ccso_luma_size_log2;
  for (int plane = 0; plane < 3; ++plane) {
    if (!params.plane[plane].enabled) continue;
    if (state->blks[plane].empty() || state->blk_cols[plane] <= 0 ||
        state->blk_rows[plane] <= 0) {
      return fail(error,
                  "CCSO block grid is not initialised; call init_ccso_tile()");
    }
    if (!fs::ccso_fill_offsets(params.plane[plane], &out->plane[plane],
                               error)) {
      return false;
    }
    out->plane[plane].enabled = true;
    out->blks[plane] = state->blks[plane].data();
    out->blk_cols[plane] = state->blk_cols[plane];
    out->blk_rows[plane] = state->blk_rows[plane];
  }
  (void)mi_cols;
  (void)mi_rows;
  (void)sub_x;
  (void)sub_y;
  return true;
}

// ---------------------------------------------------------------------------
// CCSO tile symbols
// ---------------------------------------------------------------------------

bool init_ccso_tile(fs::CcsoTileState* state, const fs::CcsoParams& params,
                    int mi_cols, int mi_rows, int sub_x, int sub_y,
                    std::string* error) {
  if (!fs::ccso_tile_init(state, params, mi_cols, mi_rows, sub_x, sub_y,
                          error)) {
    return false;
  }
  // The M2 encoder's default decision is "filter this unit".  A caller may
  // replace the grid before writing symbols.
  for (int plane = 0; plane < 3; ++plane) {
    if (!state->blks[plane].empty()) {
      std::fill(state->blks[plane].begin(), state->blks[plane].end(), 1);
    }
  }
  return true;
}

bool write_read_ccso(av2::Encoder& encoder, fs::CcsoTileState& state,
                     int mi_row, int mi_col, const int blk[3],
                     std::string* error) {
  return fs::ccso_write_block(encoder, state, mi_row, mi_col, blk, error);
}

bool read_ccso(av2::Decoder& decoder, fs::CcsoTileState& state, int mi_row,
               int mi_col, int blk[3], std::string* error) {
  return fs::ccso_read_block(decoder, state, mi_row, mi_col, blk, error);
}

// ---------------------------------------------------------------------------
// LR header / tile symbols
// ---------------------------------------------------------------------------

bool init_lr_tile(fs::LrState* state, const fs::LrParams& params, int mi_cols,
                  int mi_rows, int sub_x, int sub_y, int sb_size,
                  std::string* error) {
  if (!fs::lr_state_init(state, params, mi_cols, mi_rows, sub_x, sub_y,
                         sb_size, error)) {
    return false;
  }
  // The encoder default is "WIENER_NS in every unit of every filtered plane".
  for (int plane = 0; plane < 3; ++plane) {
    if (state->params.frame_type[plane] == rest::kRestoreNone) continue;
    std::fill(state->unit_type[plane].begin(), state->unit_type[plane].end(),
              static_cast<std::uint8_t>(rest::kRestoreWienerNs));
  }
  return true;
}

int lr_block_unit_count(const fs::LrState& state, int mi_row, int mi_col,
                        int w4, int h4, int tile_mi_row_start,
                        int tile_mi_col_start) {
  return fs::lr_block_unit_count(state, mi_row, mi_col, w4, h4,
                                 tile_mi_row_start, tile_mi_col_start);
}

bool build_lr_decisions(const fs::LrState& state, int mi_row, int mi_col,
                        int w4, int h4, int tile_mi_row_start,
                        int tile_mi_col_start,
                        std::vector<fs::LrUnitDecision>* decisions,
                        std::string* error) {
  return fs::lr_build_decisions(state, mi_row, mi_col, w4, h4,
                                tile_mi_row_start, tile_mi_col_start,
                                decisions, error);
}

bool write_read_lr(av2::Encoder& encoder, fs::LrState& state, int mi_row,
                   int mi_col, int w4, int h4, int tile_mi_row_start,
                   int tile_mi_col_start,
                   const std::vector<fs::LrUnitDecision>& decisions,
                   std::string* error) {
  return fs::lr_write_block(encoder, state, mi_row, mi_col, w4, h4,
                            tile_mi_row_start, tile_mi_col_start, decisions,
                            error);
}

bool read_lr(av2::Decoder& decoder, fs::LrState& state, int mi_row, int mi_col,
             int w4, int h4, int tile_mi_row_start, int tile_mi_col_start,
             std::vector<fs::LrUnitDecision>* decisions, std::string* error) {
  return fs::lr_read_block(decoder, state, mi_row, mi_col, w4, h4,
                           tile_mi_row_start, tile_mi_col_start, decisions,
                           error);
}

bool write_read_lr_unit(av2::Encoder& encoder, fs::LrState& state, int plane,
                        int unit_row, int unit_col,
                        const fs::LrUnitDecision& decision, std::string* error) {
  return fs::lr_write_unit(encoder, state, plane, unit_row, unit_col, decision,
                           error);
}

bool read_lr_unit(av2::Decoder& decoder, fs::LrState& state, int plane,
                  int unit_row, int unit_col, fs::LrUnitDecision* decision,
                  std::string* error) {
  return fs::lr_read_unit(decoder, state, plane, unit_row, unit_col, decision,
                          error);
}

bool write_read_wienerns_filter(av2::Encoder& encoder, fs::LrState& state,
                                int plane, int unit_row, int unit_col,
                                const fs::LrUnitDecision& decision,
                                std::string* error) {
  return fs::lr_write_unit(encoder, state, plane, unit_row, unit_col, decision,
                           error);
}

bool read_wienerns_filter(av2::Decoder& decoder, fs::LrState& state, int plane,
                          int unit_row, int unit_col,
                          fs::LrUnitDecision* decision, std::string* error) {
  return fs::lr_read_unit(decoder, state, plane, unit_row, unit_col, decision,
                          error);
}

// ---------------------------------------------------------------------------
// Class map / PC-Wiener gates
// ---------------------------------------------------------------------------

bool validate_class_map(const fs::LrParams& params, int mi_cols, int mi_rows,
                        std::string* error) {
  if (!params.frame_filters_on[0]) return true;
  if (params.frame_type[0] != rest::kRestoreWienerNs &&
      params.frame_type[0] != rest::kRestoreSwitchable) {
    return fail(error, "frame filters require Wiener/switchable luma LR");
  }
  if (!valid_num_filter_classes(params.num_filter_classes)) {
    return fail(error,
                "num_filter_classes is not in Decode_Num_Filter_Classes");
  }
  if (params.num_filter_classes == 1) {
    // Gated single-class path: §7.20.4 class 0 everywhere, no lookup table.
    return true;
  }
  const fs::WienerNsClassMap& map = params.class_map;
  if (map.num_classes != params.num_filter_classes) {
    return fail(error, "class map num_classes does not match "
                       "num_filter_classes");
  }
  if (!map.subclass_grid.empty()) {
    if (static_cast<int>(map.subclass_grid.size()) != mi_cols * mi_rows) {
      return fail(error, "subclass_grid must be mi_cols x mi_rows");
    }
    for (std::uint8_t v : map.subclass_grid) {
      if (v >= params.num_filter_classes) {
        return fail(error, "subclass_grid value outside num_filter_classes");
      }
    }
    return true;
  }
  if (!map.has_subclass_lookup) {
    return fail(error,
                "multi-class frame filters: PC-Wiener class lookup tables "
                "(pc_wiener_sub_classify2.h, pc_wiener_lut_to_class.h) are "
                "absent from the spec text; supply an explicit class map");
  }
  if (static_cast<int>(map.filter_class_grid.size()) != mi_cols * mi_rows) {
    return fail(error, "filter_class_grid must be mi_cols x mi_rows");
  }
  for (std::uint8_t v : map.filter_class_grid) {
    if (v >= rest::kWienerNsClasses) {
      return fail(error, "filter_class_grid value outside 0..15");
    }
  }
  for (std::uint8_t v : map.subclass_lookup) {
    if (v >= params.num_filter_classes) {
      return fail(error, "subclass_lookup value outside num_filter_classes");
    }
  }
  return true;
}

bool make_single_class_map(fs::WienerNsClassMap* map, int mi_cols, int mi_rows,
                           std::string* error) {
  if (map == nullptr) return fail(error, "null class map");
  if (mi_cols <= 0 || mi_rows <= 0) return fail(error, "invalid MI grid");
  map->num_classes = 1;
  map->has_subclass_lookup = true;
  map->subclass_lookup.fill(0);
  map->filter_class_grid.assign(
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows), 0);
  map->subclass_grid.clear();
  return true;
}

bool validate_no_pc_wiener(const fs::LrParams& params, int mi_cols, int mi_rows,
                           std::string* error) {
  (void)mi_cols;
  (void)mi_rows;
  for (int plane = 0; plane < 3; ++plane) {
    if (params.frame_type[plane] == rest::kRestorePcWiener) {
      return fail(error,
                  "PC-Wiener is blocked: pc_wiener_sub_classify2.h, "
                  "pc_wiener_lut_to_class.h, pc_wiener_filters.h are absent "
                  "from the spec text");
    }
    if (params.frame_type[plane] < rest::kRestoreNone ||
        params.frame_type[plane] > rest::kRestoreSwitchable) {
      return fail(error, "FrameRestorationType outside 0..3");
    }
  }
  return true;
}

bool validate_frame_filter_layout(const fs::LrParams& params,
                                  std::string* error) {
  const int num_classes =
      params.frame_filters_on[0] ? params.num_filter_classes : 1;
  if (!valid_num_filter_classes(num_classes)) {
    return fail(error,
                "num_filter_classes is not in Decode_Num_Filter_Classes");
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (!params.frame_filters_on[plane]) continue;
    if (params.frame_type[plane] != rest::kRestoreWienerNs &&
        params.frame_type[plane] != rest::kRestoreSwitchable) {
      return fail(error, "frame_filters_on without Wiener/switchable type");
    }
    const int classes = (plane == 0) ? num_classes : 1;
    for (int c = 0; c < classes; ++c) {
      const fs::WienerNsFrameFilter& f = params.frame_filter[plane][c];
      if (f.match_index < 0 || f.match_index > c) {
        return fail(error, "frame filter match_index must be in 0..class");
      }
      if (c == 0 && f.match_index != 0) {
        return fail(error, "first frame-filter class must match the all-zero "
                           "filter");
      }
      if (!f.merged) {
        const int subsets = (plane == 0) ? 4 : 3;
        if (f.subset < 0 || f.subset >= subsets) {
          return fail(error, "frame filter subset out of range");
        }
      }
    }
  }
  // Adaptive frame-filter group search (§5.20.10.6) is not implemented.  The
  // deterministic group-0 layout is the only one the writer supports; the
  // layout above is exactly that (no use_alt_group symbols).
  return true;
}

// ---------------------------------------------------------------------------
// CPU application
// ---------------------------------------------------------------------------

rest::RestorationStatus apply_ccso_frame(
    const rest::RestorationFrame& frame, const fs::CcsoParams& params,
    const fs::CcsoTileState& state) {
  fs::CcsoTileState* mutable_state = const_cast<fs::CcsoTileState*>(&state);
  rest::CcsoFrameParams out;
  std::string error;
  if (!make_ccso_frame_params(params, mutable_state, frame.mi_cols,
                              frame.mi_rows, frame.subsampling_x,
                              frame.subsampling_y, &out, &error)) {
    return rest::RestorationStatus::kInvalidArgument;
  }
  return rest::ccso_frame(frame, out);
}

rest::RestorationStatus apply_lr_frame(
    const rest::RestorationFrame& frame, const fs::LrState& state) {
  for (int plane = 0; plane < 3; ++plane) {
    if (state.params.frame_type[plane] == rest::kRestorePcWiener) {
      return rest::RestorationStatus::kUnsupportedPcWiener;
    }
  }
  rest::LoopRestorationParams params;
  for (int plane = 0; plane < 3; ++plane) {
    params.frame_type[plane] =
        static_cast<rest::RestorationType>(state.params.frame_type[plane]);
    params.unit_size[plane] = state.unit_size[plane];
    params.unit_rows[plane] = state.unit_rows[plane];
    params.unit_cols[plane] = state.unit_cols[plane];
    params.unit_type[plane] = state.unit_type[plane].data();
    params.unit_coeffs[plane] = state.unit_coeffs[plane].data();
    params.frame_filters_on[plane] = state.params.frame_filters_on[plane];
    params.frame_coeffs[plane] = state.frame_coeffs[plane][0];
    if (params.frame_type[plane] == rest::kRestoreSwitchable) {
      for (std::size_t i = 0; i < state.unit_type[plane].size(); ++i) {
        if (state.unit_type[plane][i] == rest::kRestorePcWiener) {
          return rest::RestorationStatus::kUnsupportedPcWiener;
        }
      }
    }
  }
  params.num_filter_classes = state.params.num_filter_classes;
  params.filter_class_grid =
      state.params.class_map.filter_class_grid.empty()
          ? nullptr
          : state.params.class_map.filter_class_grid.data();
  params.subclass_lookup =
      state.params.class_map.has_subclass_lookup
          ? state.params.class_map.subclass_lookup.data()
          : nullptr;
  params.subclass_grid = state.params.class_map.subclass_grid.empty()
                             ? nullptr
                             : state.params.class_map.subclass_grid.data();
  params.gdf_frame_enable = 0;
  return rest::loop_restoration_frame(frame, params);
}

rest::RestorationStatus apply_restoration_frame(
    const rest::RestorationFrame& frame,
    const fs::CcsoTileState* ccso_state, const fs::LrState* lr_state) {
  if (ccso_state != nullptr && ccso_state->params.frame_flag) {
    const rest::RestorationStatus st =
        apply_ccso_frame(frame, ccso_state->params, *ccso_state);
    if (st != rest::RestorationStatus::kOk) return st;
  }
  if (lr_state != nullptr) {
    return apply_lr_frame(frame, *lr_state);
  }
  return rest::RestorationStatus::kOk;
}

// ---------------------------------------------------------------------------
// Exact syntax cost
// ---------------------------------------------------------------------------

namespace {

std::size_t sequence_row_bits(const M2RestorationTools& tools,
                              std::string* error) {
  BitWriter writer;
  const std::size_t bits = write_sequence_filter_config(writer, tools, error);
  if (error != nullptr && !error->empty()) return 0;
  return bits;
}

}  // namespace

bool measure_restoration_cost(const M2RestorationTools& tools, int mi_cols,
                              int mi_rows, int sb_size, int base_q_idx,
                              bool frame_is_intra, int tile_start_or,
                              const fs::CcsoParams* ccso,
                              const fs::CcsoTileState* ccso_state,
                              fs::LrState* lr_state, M2RestorationCost* cost,
                              std::string* error) {
  if (cost == nullptr) return fail(error, "null cost");
  if (mi_cols <= 0 || mi_rows <= 0) return fail(error, "invalid MI geometry");
  *cost = M2RestorationCost{};

  M2RestorationTools off;
  const std::size_t off_bits = sequence_row_bits(off, error);
  const std::size_t on_bits = sequence_row_bits(tools, error);
  if (error != nullptr && !error->empty()) return false;
  cost->sequence_bits = on_bits > off_bits ? on_bits - off_bits : 0;

  if (ccso != nullptr) {
    fs::CcsoParams copy = *ccso;
    BitWriter writer;
    const fs::CcsoHeaderContext context =
        make_ccso_header_context(tools, sb_size, frame_is_intra, tile_start_or);
    cost->ccso_header_bits = ::av2enc::m2::restoration::write_ccso_params(writer, copy, context, error);
    if (error != nullptr && !error->empty()) return false;
  }

  if (lr_state != nullptr) {
    fs::LrParams params = lr_state->params;
    BitWriter writer;
    const fs::LrHeaderContext context =
        make_lr_header_context(tools, sb_size, base_q_idx, frame_is_intra);
    cost->lr_header_bits = ::av2enc::m2::restoration::write_lr_params(writer, params, context, error);
    if (error != nullptr && !error->empty()) return false;
  }

  // Tile symbols share one arithmetic-coder stream per frame.  The M2 core
  // would interleave them with partition symbols; measuring a dedicated stream
  // gives the exact symbol cost without editing the core writer.
  std::vector<std::uint8_t> buffer(
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows) * 64U +
      4096U);
  av2::Encoder encoder;
  if (encoder.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    return fail(error, "entropy init_symbol failed");
  }
  encoder.disable_cdf_update = 1;

  bool wrote_any = false;
  if (ccso != nullptr && ccso_state != nullptr && ccso->frame_flag) {
    fs::CcsoTileState state = *ccso_state;
    const int unit = 1 << (ccso->ccso_luma_size_log2 - rest::kMiSizeLog2);
    for (int row = 0; row < mi_rows; row += unit) {
      for (int col = 0; col < mi_cols; col += unit) {
        const int blk[3] = {1, 1, 1};
        if (!write_read_ccso(encoder, state, row, col, blk, error)) {
          return false;
        }
        wrote_any = true;
      }
    }
  }

  if (lr_state != nullptr) {
    const int sb_mi = sb_size / rest::kMiSize;
    const int sb_cols = ceil_div(mi_cols, sb_mi);
    const int sb_rows = ceil_div(mi_rows, sb_mi);
    for (int sb_row = 0; sb_row < sb_rows; ++sb_row) {
      for (int sb_col = 0; sb_col < sb_cols; ++sb_col) {
        const int mi_row = sb_row * sb_mi;
        const int mi_col = sb_col * sb_mi;
        std::vector<fs::LrUnitDecision> decisions;
        if (!build_lr_decisions(*lr_state, mi_row, mi_col, sb_mi, sb_mi, 0, 0,
                                &decisions, error)) {
          return false;
        }
        if (!write_read_lr(encoder, *lr_state, mi_row, mi_col, sb_mi, sb_mi, 0,
                           0, decisions, error)) {
          return false;
        }
        wrote_any = true;
      }
    }
  }

  if (!wrote_any) return true;
  std::size_t payload = 0;
  if (encoder.exit_symbol(&payload) != av2::Status::kOk) {
    return fail(error, "entropy exit_symbol failed");
  }

  // Separate the two tool costs by measuring them in isolation.  A caller that
  // enables both gets the sum, which is what matters for a combined gate.
  if (ccso != nullptr && ccso_state != nullptr && ccso->frame_flag &&
      lr_state != nullptr) {
    M2RestorationCost ccso_only;
    M2RestorationTools ccso_tools = tools;
    ccso_tools.wiener_ns = false;
    ccso_tools.frame_filters = false;
    ccso_tools.chroma_lr = false;
    if (!measure_restoration_cost(ccso_tools, mi_cols, mi_rows, sb_size,
                                  base_q_idx, frame_is_intra, tile_start_or,
                                  ccso, ccso_state, nullptr, &ccso_only,
                                  error)) {
      return false;
    }
    M2RestorationCost lr_only;
    M2RestorationTools lr_tools = tools;
    lr_tools.ccso = false;
    if (!measure_restoration_cost(lr_tools, mi_cols, mi_rows, sb_size,
                                  base_q_idx, frame_is_intra, tile_start_or,
                                  nullptr, nullptr, lr_state, &lr_only,
                                  error)) {
      return false;
    }
    cost->ccso_symbol_bytes = ccso_only.ccso_symbol_bytes;
    cost->lr_symbol_bytes = lr_only.lr_symbol_bytes;
    return true;
  }

  if (ccso != nullptr && ccso_state != nullptr) {
    cost->ccso_symbol_bytes = payload;
  } else if (lr_state != nullptr) {
    cost->lr_symbol_bytes = payload;
  }
  return true;
}

}  // namespace av2enc::m2::restoration
