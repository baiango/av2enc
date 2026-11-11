#include "encoder/filters/ccso_reuse/ccso_reuse.h"

#include "encoder/filters/restoration/restoration_common.h"
#include "encoder/m1/filters_stream/filter_cdfs.h"

#include <algorithm>
#include <cstring>

namespace av2enc::ccso_reuse {
namespace {

namespace fs = av2enc::m1::filters_stream;
namespace rest = av2enc::restoration;

bool fail(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

int interval_for(const PlaneParams& p) {
  if (p.bo_only) return 1;
  if (p.edge_clf < 0 || p.edge_clf > 2) return 0;
  return 3 - p.edge_clf;
}

bool blk_grid(const HeaderContext& context, int plane, int* cols, int* rows) {
  if (context.mi_cols <= 0 || context.mi_rows <= 0) return false;
  const int width = (context.mi_cols * rest::kMiSize) >>
                    (plane == 0 ? 0 : context.sub_x);
  const int height = (context.mi_rows * rest::kMiSize) >>
                     (plane == 0 ? 0 : context.sub_y);
  ccso_blk_grid_size(width, height, derive_ccso_luma_size_log2(
                                       context.sb_size, context.tile_start_or,
                                       context.ccso_unit_matches_sb_size),
                     plane == 0 ? 0 : context.sub_x,
                     plane == 0 ? 0 : context.sub_y, cols, rows);
  return *cols > 0 && *rows > 0;
}

// Requirement set from §6.17.7.8:
//   ccso_ref_idx[plane] < NumTotalRefs
//   SavedCcsoPlanes[idx][plane] == 1
//   RefOrderHint[idx] != RESTRICTED_OH
// and for sb_reuse_ccso:
//   RefMiRows/MiCols[idx] == MiRows/MiCols
//   SavedCcsoLumaSizeLog2[idx][plane] == CcsoLumaSizeLog2
//   CcsoLumaSizeLog2 == CCSO_LUMA_SIZE_LOG2
bool resolve_reference(const HeaderContext& context, int plane, int ref_idx,
                       bool sb_reuse, const SlotStore& slots,
                       int* slot_out, const SavedPlane** saved_out,
                       std::string* error) {
  if (ref_idx < 0 || ref_idx >= context.num_total_refs) {
    return fail(error, "ccso_ref_idx outside NumTotalRefs");
  }
  if (context.ref_frame_idx == nullptr) {
    return fail(error, "ccso_ref_idx present without ref_frame_idx");
  }
  const int slot = context.ref_frame_idx[ref_idx];
  if (!slots.in_range(slot)) {
    return fail(error, "ccso reference slot outside NUM_REF_FRAMES");
  }
  const SavedSlot& saved_slot = slots.slot(slot);
  if (!saved_slot.valid) {
    return fail(error, "CCSO reference slot is not a valid reference frame");
  }
  if (saved_slot.order_hint == kRestrictedOrderHint) {
    return fail(error, "CCSO reference is a RESTRICTED_OH frame");
  }
  const SavedPlane& saved = saved_slot.plane[plane];
  if (!saved.saved) {
    return fail(error, "SavedCcsoPlanes[idx][plane] is 0");
  }
  if (sb_reuse) {
    if (saved_slot.mi_rows != context.mi_rows ||
        saved_slot.mi_cols != context.mi_cols) {
      return fail(error, "CCSO sb_reuse reference geometry differs");
    }
    if (saved.luma_size_log2 !=
        derive_ccso_luma_size_log2(context.sb_size, context.tile_start_or,
                                   context.ccso_unit_matches_sb_size)) {
      return fail(error, "SavedCcsoLumaSizeLog2 differs from CcsoLumaSizeLog2");
    }
    if (derive_ccso_luma_size_log2(context.sb_size, context.tile_start_or,
                                   context.ccso_unit_matches_sb_size) !=
        kCcsoLumaSizeLog2) {
      return fail(error, "sb_reuse_ccso requires CcsoLumaSizeLog2 == 8");
    }
    int cols = 0;
    int rows = 0;
    if (!blk_grid(context, plane, &cols, &rows)) {
      return fail(error, "invalid CCSO block grid for sb_reuse");
    }
    if (saved.blk_cols != cols || saved.blk_rows != rows ||
        saved.blks.size() != static_cast<std::size_t>(cols) *
                                  static_cast<std::size_t>(rows)) {
      return fail(error, "saved CcsoBlks grid does not match the frame");
    }
  }
  if (slot_out != nullptr) *slot_out = slot;
  if (saved_out != nullptr) *saved_out = &saved;
  return true;
}

}  // namespace

bool operator==(const PlaneParams& a, const PlaneParams& b) {
  return a.enabled == b.enabled && a.bo_only == b.bo_only &&
         a.scale_idx == b.scale_idx && a.quant_idx == b.quant_idx &&
         a.ext_filter == b.ext_filter && a.edge_clf == b.edge_clf &&
         a.max_band_log2 == b.max_band_log2 &&
         a.offset_idx == b.offset_idx;
}

int ceil_log2_int(int value) {
  if (value <= 1) return 0;
  int bits = 0;
  int v = value - 1;
  while (v != 0) {
    v >>= 1;
    ++bits;
  }
  return bits;
}

void ccso_blk_grid_size(int plane_width, int plane_height,
                        int ccso_luma_size_log2, int sub_x, int sub_y,
                        int* cols, int* rows) {
  const int shift_x = ccso_luma_size_log2 - sub_x;
  const int shift_y = ccso_luma_size_log2 - sub_y;
  const int blk_w = 1 << (shift_x < 0 ? 0 : shift_x);
  const int blk_h = 1 << (shift_y < 0 ? 0 : shift_y);
  if (cols != nullptr) *cols = (plane_width + blk_w - 1) / blk_w;
  if (rows != nullptr) *rows = (plane_height + blk_h - 1) / blk_h;
}

int derive_ccso_luma_size_log2(int sb_size, int tile_start_or,
                               bool ccso_unit_matches_sb_size) {
  if (ccso_unit_matches_sb_size) {
    int log2 = 0;
    int v = sb_size;
    while (v > 4) {
      v >>= 1;
      ++log2;
    }
    return log2 + rest::kMiSizeLog2;
  }
  if ((tile_start_or & 63) == 0) return 8;
  if ((tile_start_or & 31) == 0) return 7;
  return 6;
}

int ccso_offset_count(const PlaneParams& p) {
  if (p.max_band_log2 < 0 || p.max_band_log2 > 6) return 0;
  const int interval = interval_for(p);
  if (interval <= 0) return 0;
  return (1 << p.max_band_log2) * interval * interval;
}

bool validate_plane_params(const PlaneParams& p, std::string* error) {
  if (!p.enabled) return true;
  if (p.scale_idx < 0 || p.scale_idx > 3 || p.quant_idx < 0 ||
      p.quant_idx > 3 || p.ext_filter < 0 || p.ext_filter > 6 ||
      p.edge_clf < 0 || p.edge_clf > 2 || p.max_band_log2 < 0 ||
      p.max_band_log2 > 6) {
    return fail(error, "CCSO scalar field out of range");
  }
  const int max_band = p.bo_only ? rest::kCcsoMaxBandLog2BoOnly
                                 : rest::kCcsoMaxBandLog2Normal;
  if (p.max_band_log2 > max_band) {
    return fail(error, "CCSO max_band_log2 not representable");
  }
  if (p.bo_only) {
    if (p.quant_idx != 0 || p.ext_filter != 0 || p.edge_clf != 0) {
      return fail(error, "CCSO bo_only forces quant/ext/edge to 0");
    }
  } else {
    if (p.edge_clf > rest::kCcsoMaxEdgeClf) {
      return fail(error, "CCSO edge_clf not representable by f(1)");
    }
    const int quant = rest::kCcsoQuantSz[p.scale_idx & 3][p.quant_idx & 3];
    if (quant == 0 && p.edge_clf != 0) {
      return fail(error, "CCSO quantStep == 0 requires edge_clf == 0");
    }
  }
  const int need = ccso_offset_count(p);
  if (need <= 0 || static_cast<int>(p.offset_idx.size()) != need) {
    return fail(error, "CCSO offset_idx count mismatch");
  }
  for (std::uint8_t idx : p.offset_idx) {
    if (idx >= kCcsoOffsetRange) {
      return fail(error, "CCSO offset index outside 0..7");
    }
  }
  return true;
}

bool save_ccso_params(SlotStore& store, int i, const SavedFrame& frame,
                      std::string* error) {
  if (!store.in_range(i)) {
    return fail(error, "save_ccso_params: slot outside NUM_REF_FRAMES");
  }
  if (frame.params == nullptr || frame.blks == nullptr ||
      frame.blk_cols == nullptr || frame.blk_rows == nullptr) {
    return fail(error, "save_ccso_params: null frame state");
  }
  if (frame.mi_cols <= 0 || frame.mi_rows <= 0) {
    return fail(error, "save_ccso_params: invalid frame geometry");
  }
  if (frame.ccso_luma_size_log2 < 6 || frame.ccso_luma_size_log2 > 8) {
    return fail(error, "save_ccso_params: CcsoLumaSizeLog2 outside 6..8");
  }
  SavedSlot out;
  out.valid = true;
  out.order_hint = frame.order_hint;
  out.mi_cols = frame.mi_cols;
  out.mi_rows = frame.mi_rows;
  for (int plane = 0; plane < 3; ++plane) {
    SavedPlane& saved = out.plane[plane];
    saved.luma_size_log2 = frame.ccso_luma_size_log2;
    saved.params = frame.params->plane[plane];
    saved.saved = saved.params.enabled;
    saved.blk_cols = 0;
    saved.blk_rows = 0;
    saved.blks.clear();
    if (!saved.saved) continue;
    if (!validate_plane_params(saved.params, error)) return false;
    // The saved grid must be the grid implied by the current luma size.
    const int width = (frame.mi_cols * rest::kMiSize) >> (plane == 0 ? 0 : 1);
    const int height = (frame.mi_rows * rest::kMiSize) >> (plane == 0 ? 0 : 1);
    int cols = 0;
    int rows = 0;
    ccso_blk_grid_size(width, height, frame.ccso_luma_size_log2,
                       plane == 0 ? 0 : 1, plane == 0 ? 0 : 1, &cols, &rows);
    if (frame.blk_cols[plane] != cols || frame.blk_rows[plane] != rows) {
      return fail(error, "save_ccso_params: CcsoBlks grid size mismatch");
    }
    if (frame.blks[plane].size() !=
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows)) {
      return fail(error, "save_ccso_params: CcsoBlks buffer size mismatch");
    }
    saved.blk_cols = cols;
    saved.blk_rows = rows;
    saved.blks = frame.blks[plane];
  }
  store.slot(i) = out;
  return true;
}

std::size_t write_ccso_params(BitWriter& writer, FrameParams& params,
                              const HeaderContext& context,
                              std::string* error) {
  if (context.coded_lossless || !context.enable_ccso) {
    params.frame_flag = false;
    for (int plane = 0; plane < 3; ++plane) {
      params.plane[plane].enabled = false;
      params.reuse_ccso[plane] = false;
      params.sb_reuse_ccso[plane] = false;
      params.ccso_ref_idx[plane] = 0;
    }
    return 0;
  }
  params.ccso_luma_size_log2 =
      derive_ccso_luma_size_log2(context.sb_size, context.tile_start_or,
                                 context.ccso_unit_matches_sb_size);
  const std::size_t start = writer.bit_size();
  if (context.single_picture_header_flag) {
    params.frame_flag = true;
  } else {
    writer.put_bit(params.frame_flag);
  }
  if (!params.frame_flag) {
    for (int plane = 0; plane < 3; ++plane) {
      params.plane[plane].enabled = false;
      params.reuse_ccso[plane] = false;
      params.sb_reuse_ccso[plane] = false;
    }
    return writer.bit_size() - start;
  }

  for (int plane = 0; plane < context.num_planes; ++plane) {
    PlaneParams& p = params.plane[plane];
    if (!p.enabled) {
      params.reuse_ccso[plane] = false;
      params.sb_reuse_ccso[plane] = false;
    }
    writer.put_bit(p.enabled);
    if (!p.enabled) continue;

    if (context.frame_is_intra) {
      params.reuse_ccso[plane] = false;
      params.sb_reuse_ccso[plane] = false;
    } else {
      writer.put_bit(params.reuse_ccso[plane]);
      writer.put_bit(params.sb_reuse_ccso[plane]);
    }

    const bool reuse = params.reuse_ccso[plane] || params.sb_reuse_ccso[plane];
    if (reuse) {
      if (context.num_total_refs <= 0) {
        fail(error, "CCSO reuse requires NumTotalRefs >= 1");
        return 0;
      }
      if (params.ccso_ref_idx[plane] < 0 ||
          params.ccso_ref_idx[plane] >= context.num_total_refs) {
        fail(error, "ccso_ref_idx outside NumTotalRefs");
        return 0;
      }
      if (context.slots != nullptr) {
        const SavedPlane* saved = nullptr;
        if (!resolve_reference(context, plane, params.ccso_ref_idx[plane],
                               params.sb_reuse_ccso[plane], *context.slots,
                               nullptr, &saved, error)) {
          return 0;
        }
        if (params.reuse_ccso[plane] && saved->params != p) {
          fail(error, "reused CCSO parameters differ from the saved slot");
          return 0;
        }
        if (params.sb_reuse_ccso[plane] &&
            params.reuse_ccso[plane] == false && !saved->params.enabled) {
          fail(error, "sb_reuse_ccso needs an enabled saved plane");
          return 0;
        }
      }
      const int n = ceil_log2_int(context.num_total_refs);
      writer.put_bits(static_cast<std::uint64_t>(params.ccso_ref_idx[plane]),
                      static_cast<unsigned>(n));
    }

    if (params.reuse_ccso[plane]) continue;
    if (!validate_plane_params(p, error)) return 0;
    writer.put_bit(p.bo_only);
    writer.put_bits(static_cast<std::uint64_t>(p.scale_idx), 2U);
    if (p.bo_only) {
      p.quant_idx = 0;
      p.ext_filter = 0;
      p.edge_clf = 0;
    } else {
      writer.put_bits(static_cast<std::uint64_t>(p.quant_idx), 2U);
      writer.put_bits(static_cast<std::uint64_t>(p.ext_filter), 3U);
      const int quant = rest::kCcsoQuantSz[p.scale_idx & 3][p.quant_idx & 3];
      if (quant == 0) {
        p.edge_clf = 0;
      } else {
        writer.put_bit(p.edge_clf != 0);
      }
    }
    const int band_bits = 2 + (p.bo_only ? 1 : 0);
    writer.put_bits(static_cast<std::uint64_t>(p.max_band_log2),
                    static_cast<unsigned>(band_bits));
    const int interval = interval_for(p);
    const int bands = 1 << p.max_band_log2;
    std::size_t k = 0;
    for (int d0 = 0; d0 < interval; ++d0) {
      for (int d1 = 0; d1 < interval; ++d1) {
        for (int band = 0; band < bands; ++band) {
          fs::put_tu(writer, 7U, p.offset_idx[k++]);
        }
      }
    }
  }
  return writer.bit_size() - start;
}

bool read_ccso_params(BitReader& reader, const HeaderContext& context,
                      FrameParams* params, std::string* error) {
  if (params == nullptr) return fail(error, "null CCSO params");
  *params = FrameParams{};
  params->ccso_luma_size_log2 =
      derive_ccso_luma_size_log2(context.sb_size, context.tile_start_or,
                                 context.ccso_unit_matches_sb_size);
  if (context.coded_lossless || !context.enable_ccso) {
    params->frame_flag = false;
    return true;
  }
  if (context.single_picture_header_flag) {
    params->frame_flag = true;
  } else if (!reader.read_bit(&params->frame_flag)) {
    return fail(error, "truncated ccso_frame_flag");
  }
  if (!params->frame_flag) return true;

  for (int plane = 0; plane < context.num_planes; ++plane) {
    PlaneParams& p = params->plane[plane];
    if (!reader.read_bit(&p.enabled)) {
      return fail(error, "truncated ccso_planes");
    }
    if (!p.enabled) continue;

    if (!context.frame_is_intra) {
      if (!reader.read_bit(&params->reuse_ccso[plane]) ||
          !reader.read_bit(&params->sb_reuse_ccso[plane])) {
        return fail(error, "truncated CCSO reuse bits");
      }
    }

    const bool reuse =
        params->reuse_ccso[plane] || params->sb_reuse_ccso[plane];
    const SavedPlane* saved = nullptr;
    if (reuse) {
      const int n = ceil_log2_int(context.num_total_refs);
      std::uint32_t idx = 0;
      if (n > 0 && !reader.read_bits(static_cast<unsigned>(n), &idx)) {
        return fail(error, "truncated ccso_ref_idx");
      }
      params->ccso_ref_idx[plane] = static_cast<int>(idx);
      if (context.slots == nullptr) {
        return fail(error, "CCSO reuse requires a saved slot store");
      }
      if (!resolve_reference(context, plane, params->ccso_ref_idx[plane],
                             params->sb_reuse_ccso[plane], *context.slots,
                             nullptr, &saved, error)) {
        return false;
      }
      if (!params->reuse_ccso[plane] && !saved->params.enabled) {
        return fail(error, "sb_reuse_ccso needs an enabled saved plane");
      }
    }

    if (params->reuse_ccso[plane]) {
      // load_ccso_params(): the whole plane parameter set comes from the slot.
      p = saved->params;
      continue;
    }

    if (!reader.read_bit(&p.bo_only)) {
      return fail(error, "truncated ccso_bo_only");
    }
    std::uint32_t v = 0;
    if (!reader.read_bits(2U, &v)) return fail(error, "truncated ccso scale");
    p.scale_idx = static_cast<int>(v);
    if (p.bo_only) {
      p.quant_idx = 0;
      p.ext_filter = 0;
      p.edge_clf = 0;
    } else {
      if (!reader.read_bits(2U, &v)) return fail(error, "truncated ccso quant");
      p.quant_idx = static_cast<int>(v);
      if (!reader.read_bits(3U, &v)) {
        return fail(error, "truncated ccso ext_filter");
      }
      p.ext_filter = static_cast<int>(v);
      if (p.ext_filter > 6) {
        return fail(error, "CCSO ext_filter must be 0..6");
      }
      const int quant = rest::kCcsoQuantSz[p.scale_idx & 3][p.quant_idx & 3];
      if (quant == 0) {
        p.edge_clf = 0;
      } else {
        bool edge = false;
        if (!reader.read_bit(&edge)) {
          return fail(error, "truncated ccso edge_clf");
        }
        p.edge_clf = edge ? 1 : 0;
      }
    }
    const int band_bits = 2 + (p.bo_only ? 1 : 0);
    if (!reader.read_bits(static_cast<unsigned>(band_bits), &v)) {
      return fail(error, "truncated ccso max_band_log2");
    }
    p.max_band_log2 = static_cast<int>(v);
    const int max_band = p.bo_only ? rest::kCcsoMaxBandLog2BoOnly
                                   : rest::kCcsoMaxBandLog2Normal;
    if (p.max_band_log2 > max_band) {
      return fail(error, "CCSO max_band_log2 not representable");
    }
    const int interval = interval_for(p);
    const int bands = 1 << p.max_band_log2;
    p.offset_idx.clear();
    p.offset_idx.reserve(
        static_cast<std::size_t>(interval * interval * bands));
    for (int d0 = 0; d0 < interval; ++d0) {
      for (int d1 = 0; d1 < interval; ++d1) {
        for (int band = 0; band < bands; ++band) {
          std::uint32_t idx = 0;
          if (!reader.read_tu(7U, &idx)) {
            return fail(error, "truncated ccso_offset_idx");
          }
          p.offset_idx.push_back(static_cast<std::uint8_t>(idx));
        }
      }
    }
    if (!validate_plane_params(p, error)) return false;
  }
  return true;
}

bool resolve_ccso_params(const HeaderContext& context,
                         const FrameParams& syntax, const SlotStore& slots,
                         EffectiveFrame* out, std::string* error) {
  if (out == nullptr) return fail(error, "null effective CCSO frame");
  *out = EffectiveFrame{};
  out->frame_flag = syntax.frame_flag;
  out->ccso_luma_size_log2 =
      derive_ccso_luma_size_log2(context.sb_size, context.tile_start_or,
                                 context.ccso_unit_matches_sb_size);
  for (int plane = 0; plane < 3; ++plane) {
    const PlaneParams& p = syntax.plane[plane];
    EffectivePlane& e = out->plane[plane];
    if (!syntax.frame_flag || !p.enabled) {
      e.params.enabled = false;
      continue;
    }
    const bool reuse = syntax.reuse_ccso[plane] || syntax.sb_reuse_ccso[plane];
    const SavedPlane* saved = nullptr;
    if (reuse) {
      if (!resolve_reference(context, plane, syntax.ccso_ref_idx[plane],
                             syntax.sb_reuse_ccso[plane], slots,
                             &e.ref_slot, &saved, error)) {
        return false;
      }
    }
    if (syntax.reuse_ccso[plane]) {
      e.params = saved->params;
    } else {
      e.params = p;
      if (!validate_plane_params(e.params, error)) return false;
    }
    e.sb_reuse_ccso = syntax.sb_reuse_ccso[plane];
    int cols = 0;
    int rows = 0;
    if (!blk_grid(context, plane, &cols, &rows)) {
      return fail(error, "invalid CCSO block grid");
    }
    e.blk_cols = cols;
    e.blk_rows = rows;
    const std::size_t cells =
        static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows);
    if (syntax.sb_reuse_ccso[plane]) {
      e.blks = saved->blks;
    } else {
      e.blks.assign(cells, 0);
    }
  }
  return true;
}

bool use_ccso(const PlaneParams& desired,
              const std::vector<std::uint8_t>& desired_blks, int blk_cols,
              int blk_rows, int plane, const HeaderContext& context,
              const SlotStore& slots, ReuseChoice* out, std::string* error) {
  if (out == nullptr) return fail(error, "null CCSO reuse choice");
  *out = ReuseChoice{};
  if (plane < 0 || plane >= 3) {
    return fail(error, "CCSO plane outside 0..2");
  }
  if (context.coded_lossless || !context.enable_ccso) return true;
  if (!desired.enabled) return true;
  if (!validate_plane_params(desired, error)) return false;
  if (context.frame_is_intra) return true;
  if (context.num_total_refs <= 0) return true;

  for (int idx = 0; idx < context.num_total_refs; ++idx) {
    const SavedPlane* saved = nullptr;
    std::string trial_error;
    // The sb-reuse requirement set is a strict superset of the parameter
    // reuse set, so try it first and fall back to parameters only.
    bool sb_ok = resolve_reference(context, plane, idx, true, slots, nullptr,
                                   &saved, &trial_error);
    if (!sb_ok) {
      saved = nullptr;
      if (!resolve_reference(context, plane, idx, false, slots, nullptr,
                             &saved, &trial_error)) {
        continue;
      }
    }
    if (saved == nullptr) continue;
    const bool params_equal = saved->params == desired;
    bool blks_equal = false;
    if (sb_ok && blk_cols > 0 && blk_rows > 0 &&
        saved->blk_cols == blk_cols && saved->blk_rows == blk_rows &&
        saved->blks.size() == desired_blks.size() && !desired_blks.empty()) {
      blks_equal = saved->blks == desired_blks;
    }
    if (!params_equal && !blks_equal) continue;

    out->ccso_ref_idx = idx;
    out->reuse_ccso = params_equal;
    out->sb_reuse_ccso = blks_equal;
    out->ref_slot = context.ref_frame_idx != nullptr
                        ? context.ref_frame_idx[idx]
                        : -1;
    // Rough estimate: fresh parameters are 3 + ... bits; the exact count is
    // reported by write_ccso_params() at the call site, so this is only a
    // ranking hint.
    if (out->reuse_ccso) out->bits_saved += 1 + 2 + 4;
    if (out->sb_reuse_ccso) {
      out->bits_saved += static_cast<int>(desired_blks.size());
    }
    return true;
  }
  return true;
}

// ---------------------------------------------------------------------------
// §5.20.10.2 read_ccso()
// ---------------------------------------------------------------------------

bool TileState::init(const EffectiveFrame& frame, int mi_cols, int mi_rows,
                     int sub_x, int sub_y, std::string* error) {
  if (mi_cols <= 0 || mi_rows <= 0) {
    return fail(error, "invalid CCSO tile geometry");
  }
  if (frame.ccso_luma_size_log2 < 6 || frame.ccso_luma_size_log2 > 8) {
    return fail(error, "CcsoLumaSizeLog2 outside 6..8");
  }
  frame_ = frame;
  mi_cols_ = mi_cols;
  mi_rows_ = mi_rows;
  sub_x_ = sub_x;
  sub_y_ = sub_y;
  tile_mi_col_start_ = 0;
  tile_mi_row_start_ = 0;
  std::memcpy(cdf_, fs::kCcsoBlkCdf, sizeof(cdf_));
  for (int plane = 0; plane < 3; ++plane) {
    EffectivePlane& e = frame_.plane[plane];
    if (!e.params.enabled) {
      e.blk_cols = 0;
      e.blk_rows = 0;
      e.blks.clear();
      desired_[plane].clear();
      continue;
    }
    if (e.blk_cols <= 0 || e.blk_rows <= 0) {
      return fail(error, "missing CCSO block grid");
    }
    const std::size_t cells = static_cast<std::size_t>(e.blk_cols) *
                              static_cast<std::size_t>(e.blk_rows);
    if (e.blks.size() != cells) {
      e.blks.assign(cells, 0);
    }
    if (desired_[plane].size() != cells) desired_[plane].assign(cells, 0);
  }
  return true;
}

bool TileState::set_block_flags(int plane, const std::uint8_t* data, int cols,
                                int rows, std::string* error) {
  if (plane < 0 || plane >= 3) return fail(error, "CCSO plane out of range");
  EffectivePlane& e = frame_.plane[plane];
  if (!e.params.enabled) return true;
  if (data == nullptr || cols != e.blk_cols || rows != e.blk_rows) {
    return fail(error, "CcsoBlks decision grid mismatch");
  }
  const std::size_t cells =
      static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows);
  desired_[plane].assign(data, data + cells);
  return true;
}

bool TileState::codes_block_flags(int plane) const {
  if (plane < 0 || plane >= 3 || !frame_.frame_flag) return false;
  const EffectivePlane& e = frame_.plane[plane];
  return e.params.enabled && !e.sb_reuse_ccso;
}

int TileState::block_origin(int mi_row, int mi_col, int* shift_row,
                            int* shift_col, int* blk_h4, int* blk_w4) const {
  const int sr = frame_.ccso_luma_size_log2 - rest::kMiSizeLog2;
  const int sc = sr;
  if (sr < 0 || sc < 0) return 0;
  *shift_row = sr;
  *shift_col = sc;
  *blk_h4 = 1 << sr;
  *blk_w4 = 1 << sc;
  return (mi_row & (*blk_h4 - 1)) == 0 && (mi_col & (*blk_w4 - 1)) == 0;
}

int TileState::block_context(int plane, int mi_row, int mi_col, int shift_row,
                             int shift_col, int blk_w4) const {
  if (mi_col - blk_w4 < tile_mi_col_start_) return 0;
  const int row = mi_row >> shift_row;
  const int col = (mi_col - blk_w4) >> shift_col;
  const EffectivePlane& e = frame_.plane[plane];
  if (row < 0 || row >= e.blk_rows || col < 0 || col >= e.blk_cols) return 0;
  const int value =
      e.blks[static_cast<std::size_t>(row) *
                 static_cast<std::size_t>(e.blk_cols) +
             static_cast<std::size_t>(col)];
  return 2 * value;
}

bool TileState::write_block(av2::Encoder& encoder, int mi_row, int mi_col,
                            std::string* error) {
  if (!frame_.frame_flag) return true;
  int shift_row = 0;
  int shift_col = 0;
  int blk_h4 = 0;
  int blk_w4 = 0;
  if (!block_origin(mi_row, mi_col, &shift_row, &shift_col, &blk_h4,
                    &blk_w4)) {
    return true;
  }
  for (int plane = 0; plane < 3; ++plane) {
    EffectivePlane& e = frame_.plane[plane];
    if (!e.params.enabled) continue;
    // §5.20.10.2: sb_reuse_ccso suppresses the ccso_blk S() symbols.
    if (!codes_block_flags(plane)) continue;
    if (desired_[plane].empty()) {
      return fail(error, "missing encoder CcsoBlks decision grid");
    }
    const int ctx =
        block_context(plane, mi_row, mi_col, shift_row, shift_col, blk_w4);
    const int row = mi_row >> shift_row;
    const int col = mi_col >> shift_col;
    if (row < 0 || row >= e.blk_rows || col < 0 || col >= e.blk_cols) {
      continue;
    }
    const std::size_t cell =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(e.blk_cols) +
        static_cast<std::size_t>(col);
    const int symbol = desired_[plane][cell] != 0 ? 1 : 0;
    const av2::Status status =
        encoder.encode_symbol(&cdf_[plane][ctx][0], 2, symbol);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("ccso_blk encode failed: ") +
                             av2::status_name(status));
    }
    e.blks[cell] = static_cast<std::uint8_t>(symbol);
  }
  return true;
}

bool TileState::read_block(av2::Decoder& decoder, int mi_row, int mi_col,
                           int blk[3], std::string* error) {
  if (blk != nullptr) {
    blk[0] = 0;
    blk[1] = 0;
    blk[2] = 0;
  }
  if (!frame_.frame_flag) return true;
  int shift_row = 0;
  int shift_col = 0;
  int blk_h4 = 0;
  int blk_w4 = 0;
  if (!block_origin(mi_row, mi_col, &shift_row, &shift_col, &blk_h4,
                    &blk_w4)) {
    return true;
  }
  for (int plane = 0; plane < 3; ++plane) {
    EffectivePlane& e = frame_.plane[plane];
    if (!e.params.enabled) continue;
    if (!codes_block_flags(plane)) {
      if (blk != nullptr) {
        const int row = mi_row >> shift_row;
        const int col = mi_col >> shift_col;
        if (row >= 0 && row < e.blk_rows && col >= 0 && col < e.blk_cols) {
          blk[plane] = e.blks[static_cast<std::size_t>(row) *
                                  static_cast<std::size_t>(e.blk_cols) +
                              static_cast<std::size_t>(col)];
        }
      }
      continue;
    }
    const int ctx =
        block_context(plane, mi_row, mi_col, shift_row, shift_col, blk_w4);
    int symbol = 0;
    const av2::Status status =
        decoder.decode_symbol(&cdf_[plane][ctx][0], 2, &symbol);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("ccso_blk decode failed: ") +
                             av2::status_name(status));
    }
    const int row = mi_row >> shift_row;
    const int col = mi_col >> shift_col;
    if (row >= 0 && row < e.blk_rows && col >= 0 && col < e.blk_cols) {
      e.blks[static_cast<std::size_t>(row) *
                 static_cast<std::size_t>(e.blk_cols) +
             static_cast<std::size_t>(col)] =
          static_cast<std::uint8_t>(symbol);
    }
    if (blk != nullptr) blk[plane] = symbol;
  }
  return true;
}

bool make_restoration_params(const TileState& state,
                             restoration::CcsoFrameParams* out,
                             std::string* error) {
  if (out == nullptr) return fail(error, "null restoration params");
  *out = restoration::CcsoFrameParams{};
  const EffectiveFrame& frame = state.frame();
  out->ccso_luma_size_log2 = frame.ccso_luma_size_log2;
  for (int plane = 0; plane < 3; ++plane) {
    const EffectivePlane& e = frame.plane[plane];
    restoration::CcsoParams& rp = out->plane[plane];
    rp.enabled = e.params.enabled;
    rp.bo_only = e.params.bo_only;
    rp.scale_idx = e.params.scale_idx;
    rp.quant_idx = e.params.quant_idx;
    rp.ext_filter = e.params.ext_filter;
    rp.edge_clf = e.params.edge_clf;
    rp.max_band_log2 = e.params.max_band_log2;
    if (rp.enabled) {
      if (!restoration::ccso_set_offsets(rp, e.params.offset_idx.data(),
                                         e.params.offset_idx.size())) {
        return fail(error, "invalid CCSO offset table");
      }
      out->blks[plane] = e.blks.data();
      out->blk_cols[plane] = e.blk_cols;
      out->blk_rows[plane] = e.blk_rows;
    }
  }
  return true;
}

}  // namespace av2enc::ccso_reuse
