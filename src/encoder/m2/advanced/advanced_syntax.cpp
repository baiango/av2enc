#include "encoder/m2/advanced/advanced_syntax.h"

#include <algorithm>
#include <cstring>

namespace av2enc::m2::adv {
namespace {

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return false;
}

bool sym(av2::Encoder& enc, std::uint16_t* cdf, int n, int symbol,
         std::string* error) {
  const av2::Status status = enc.encode_symbol(cdf, n, symbol);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("encode_symbol failed: ") +
                           av2::status_name(status) + " n=" + std::to_string(n) +
                           " sym=" + std::to_string(symbol));
  }
  return true;
}

bool sym(av2::Decoder& dec, std::uint16_t* cdf, int n, int* symbol,
         std::string* error) {
  const av2::Status status = dec.decode_symbol(cdf, n, symbol);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("decode_symbol failed: ") +
                           av2::status_name(status) + " n=" + std::to_string(n));
  }
  return true;
}

bool raw_bit(av2::Encoder& enc, bool bit, std::string* error) {
  const av2::Status status = enc.encode_bool(bit);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("encode_bool failed: ") + av2::status_name(status));
  }
  return true;
}

bool raw_bit(av2::Decoder& dec, bool* bit, std::string* error) {
  const av2::Status status = dec.read_bool(bit);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("read_bool failed: ") + av2::status_name(status));
  }
  return true;
}

template <typename T, std::size_t N>
void copy_flat(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(T) * N);
}

int clamp_int(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

}  // namespace

// ---------------------------------------------------------------------------
// CDF state
// ---------------------------------------------------------------------------

void AdvancedCdfState::reset() {
  copy_flat(is_warp, cdf::kIsWarpCdf);
  copy_flat(warp_mv, cdf::kWarpMvCdf);
  copy_flat(warp_idx, cdf::kWarpIdxCdf);
  copy_flat(warp_with_mvd, cdf::kWarpWithMvdCdf);
  copy_flat(warp_precision, cdf::kWarpPrecisionCdf);
  copy_flat(warp_delta_low, cdf::kWarpDeltaParamLowCdf);
  copy_flat(warp_delta_high, cdf::kWarpDeltaParamHighCdf);
  copy_flat(warp_delta_sign, cdf::kWarpDeltaParamSignCdf);
  copy_flat(inter_intra, cdf::kInterIntraCdf);
  copy_flat(warp_inter_intra, cdf::kWarpInterIntraCdf);
  copy_flat(inter_intra_mode, cdf::kInterIntraModeCdf);
  copy_flat(wedge_inter_intra, cdf::kWedgeInterIntraCdf);
  copy_flat(wedge_quad, cdf::kWedgeQuadCdf);
  copy_flat(wedge_angle, cdf::kWedgeAngleCdf);
  copy_flat(wedge_dist1, cdf::kWedgeDist1Cdf);
  copy_flat(wedge_dist2, cdf::kWedgeDist2Cdf);
  copy_flat(use_bawp, cdf::kUseBawpCdf);
  copy_flat(use_bawp_chroma, cdf::kUseBawpChromaCdf);
  copy_flat(explicit_bawp, cdf::kExplicitBawpCdf);
  copy_flat(explicit_bawp_scale, cdf::kExplicitBawpScaleCdf);
  copy_flat(use_local_warp, cdf::kUseLocalWarpCdf);
  copy_flat(use_extend_warp, cdf::kUseExtendWarpCdf);
}

// ---------------------------------------------------------------------------
// §7.11.2 find_mode_ctx / §7.11.4 scan_point_warp_ctx
// ---------------------------------------------------------------------------

namespace {

struct WarpScan {
  const NeighborGrid& grid;
  int mi_row;
  int mi_col;
  int sb_mi_size;
  int ref0;
  int ref1;
  ModeContext* out;
};

void scan_point_warp_ctx(WarpScan& scan, int delta_row, int delta_col) {
  const int is_sb_border =
      (scan.mi_row & (scan.sb_mi_size - 1)) == 0 ? 1 : 0;
  if (delta_row < 0 && is_sb_border) {
    delta_col -= scan.mi_col & 1;
  }
  const int mv_row = scan.mi_row + delta_row;
  const int mv_col = scan.mi_col + delta_col;
  if (!scan.grid.inside(mv_row, mv_col)) return;
  const NeighborMi& n = scan.grid.at(mv_row, mv_col);
  if (!n.written || !n.is_inter) return;
  if (n.ref[0] == scan.ref0 || n.ref[1] == scan.ref0) {
    if (!scan.out->warp_sample_found[0]) {
      scan.out->extend_delta_row = delta_row;
      scan.out->extend_delta_col = delta_col;
    }
    scan.out->warp_sample_found[0] = true;
    if (n.motion_mode >= static_cast<int>(ia::MotionMode::kLocalWarp)) {
      ++scan.out->warp_mv_count;
    }
  }
  if (n.ref[0] == scan.ref1 || n.ref[1] == scan.ref1) {
    scan.out->warp_sample_found[1] = true;
  }
}

// §5.20.7.1 add_neighbor(): only the first two in-superblock-row positions
// feed NPos.
void add_npos(const NeighborGrid& grid, int mi_row, int sb_mi_size,
              int n_row, int n_col, ModeContext* out) {
  const bool above_sb_boundary =
      (mi_row / sb_mi_size) != (n_row / sb_mi_size);
  if (out->npos_count < 2 && grid.inside(n_row, n_col) && !above_sb_boundary) {
    out->npos[out->npos_count][0] = n_row;
    out->npos[out->npos_count][1] = n_col;
    ++out->npos_count;
  }
}

}  // namespace

ModeContext compute_mode_context(const NeighborGrid& grid, int mi_row,
                                 int mi_col, int bsize, int sb_mi_size,
                                 int ref0, int ref1) {
  ModeContext out{};
  if (grid.mi == nullptr || grid.mi_cols <= 0 || grid.mi_rows <= 0 ||
      sb_mi_size <= 0 || (sb_mi_size & (sb_mi_size - 1)) != 0 ||
      bsize < 0 || bsize >= kBlockSizes) {
    return out;
  }

  const int bw4 =
      partition::num_4x4_blocks_wide(static_cast<partition::BlockSize>(bsize));
  const int bh4 =
      partition::num_4x4_blocks_high(static_cast<partition::BlockSize>(bsize));
  if (bw4 <= 0 || bh4 <= 0) return out;

  const int is_sb_border = (mi_row & (sb_mi_size - 1)) == 0 ? 1 : 0;
  WarpScan scan{grid, mi_row, mi_col, sb_mi_size, ref0, ref1, &out};
  scan_point_warp_ctx(scan, bh4 - 1, -1);
  scan_point_warp_ctx(scan, -1, is_sb_border ? std::max(0, bw4 - 2) : bw4 - 1);
  scan_point_warp_ctx(scan, 0, -1);
  if (bw4 >= (is_sb_border ? 4 : 2)) {
    scan_point_warp_ctx(scan, -1, 0);
  }
  out.warp_mv_count = clamp_int(0, 4, out.warp_mv_count);

  add_npos(grid, mi_row, sb_mi_size, mi_row + bh4 - 1, mi_col - 1,
           &out);
  add_npos(grid, mi_row, sb_mi_size, mi_row - 1, mi_col + bw4 - 1,
           &out);
  add_npos(grid, mi_row, sb_mi_size, mi_row, mi_col - 1, &out);
  add_npos(grid, mi_row, sb_mi_size, mi_row - 1, mi_col, &out);

  int has_warp = 0;
  out.local_warp_ctx = 0;
  out.extend_warp_ctx = 0;
  for (int n = 0; n < out.npos_count; ++n) {
    const NeighborMi& mi = grid.at(out.npos[n][0], out.npos[n][1]);
    const int mode = mi.motion_mode;
    if (mode >= static_cast<int>(ia::MotionMode::kLocalWarp)) {
      has_warp = 1;
      ++out.extend_warp_ctx;
    }
    if (mode == static_cast<int>(ia::MotionMode::kLocalWarp)) {
      ++out.local_warp_ctx;
    }
  }
  out.local_warp_ctx += has_warp;
  out.local_warp_ctx_npos = out.local_warp_ctx;
  out.extend_warp_ctx_npos = out.extend_warp_ctx;
  return out;
}

int warp_mv_context(const NeighborGrid& grid, int mi_row, int mi_col, int bsize,
                    int sb_mi_size, int ref0, int ref1) {
  return compute_mode_context(grid, mi_row, mi_col, bsize, sb_mi_size, ref0,
                              ref1)
      .warp_mv_count;
}

// ---------------------------------------------------------------------------
// Wedge mode mapping
// ---------------------------------------------------------------------------

bool wedge_mode_symbols(int wedge_index, WedgeModeSymbols* out) {
  if (out == nullptr || wedge_index < 0 || wedge_index >= wt::kWedgeTypes) {
    return false;
  }
  for (int angle = 0; angle < wt::kWedgeAngles; ++angle) {
    for (int dist = 0; dist < wt::kNumWedgeDist; ++dist) {
      if (wt::kWedgeAngleDist2Index[angle][dist] != wedge_index) continue;
      WedgeModeSymbols s{};
      s.wedge_index = wedge_index;
      s.wedge_quad = angle / 5;
      s.wedge_angle = angle % 5;
      s.use_dist2 = (angle >= ia::kHwEdgeAngles) ||
                    (angle == 5) || (angle == 0);
      if (s.use_dist2) {
        s.wedge_dist2 = dist - 1;
      } else {
        s.wedge_dist1 = dist;
      }
      *out = s;
      return true;
    }
  }
  return false;
}

bool wedge_index_from_symbols(const WedgeModeSymbols& symbols, int* wedge_index) {
  if (wedge_index == nullptr) return false;
  if (symbols.wedge_quad < 0 || symbols.wedge_quad > 3 ||
      symbols.wedge_angle < 0 || symbols.wedge_angle > 4) {
    return false;
  }
  const int angle = symbols.wedge_quad * 5 + symbols.wedge_angle;
  if (angle < 0 || angle >= wt::kWedgeAngles) return false;
  int dist = 0;
  if (symbols.use_dist2) {
    if (symbols.wedge_dist2 < 0 ||
        symbols.wedge_dist2 >= wt::kNumWedgeDist - 1) {
      return false;
    }
    dist = symbols.wedge_dist2 + 1;
  } else {
    if (symbols.wedge_dist1 < 0 ||
        symbols.wedge_dist1 >= wt::kNumWedgeDist) {
      return false;
    }
    dist = symbols.wedge_dist1;
  }
  const int index = wt::kWedgeAngleDist2Index[angle][dist];
  if (index < 0 || index >= wt::kWedgeTypes) return false;
  *wedge_index = index;
  return true;
}

// ---------------------------------------------------------------------------
// Writers
// ---------------------------------------------------------------------------

bool write_is_warp(av2::Encoder& enc, AdvancedCdfState& cdfs, int warp_mv_ctx,
                   bool value, std::string* error) {
  if (warp_mv_ctx < 0 || warp_mv_ctx > 4) {
    return fail(error, "is_warp context outside 0..4");
  }
  return sym(enc, cdfs.is_warp[warp_mv_ctx], 2, value ? 1 : 0, error);
}

bool write_warp_mv(av2::Encoder& enc, AdvancedCdfState& cdfs, bool value,
                   std::string* error) {
  return sym(enc, cdfs.warp_mv, 2, value ? 1 : 0, error);
}

bool write_bawp_symbols(av2::Encoder& enc, AdvancedCdfState& cdfs,
                        const BawpSymbols& symbols, bool has_chroma,
                        std::string* error) {
  if (symbols.explicit_ctx < 0 || symbols.explicit_ctx > 2) {
    return fail(error, "explicit_bawp context outside 0..2");
  }
  if (!symbols.use_bawp) {
    if (symbols.explicit_bawp || symbols.use_bawp_chroma) {
      return fail(error, "BAWP sub-flags set while use_bawp is 0");
    }
    return sym(enc, cdfs.use_bawp, 2, 0, error);
  }
  if (!sym(enc, cdfs.use_bawp, 2, 1, error)) return false;
  if (!sym(enc, cdfs.explicit_bawp[symbols.explicit_ctx], 2,
           symbols.explicit_bawp ? 1 : 0, error)) {
    return false;
  }
  if (symbols.explicit_bawp) {
    if (!sym(enc, cdfs.explicit_bawp_scale, 2,
             symbols.explicit_bawp_scale ? 1 : 0, error)) {
      return false;
    }
  } else if (symbols.explicit_bawp_scale) {
    return fail(error, "explicit_bawp_scale set while explicit_bawp is 0");
  }
  if (has_chroma) {
    if (!sym(enc, cdfs.use_bawp_chroma, 2,
             symbols.use_bawp_chroma ? 1 : 0, error)) {
      return false;
    }
  } else if (symbols.use_bawp_chroma) {
    return fail(error, "use_bawp_chroma set for a monochrome frame");
  }
  return true;
}

bool write_wedge_mode(av2::Encoder& enc, AdvancedCdfState& cdfs,
                      const WedgeModeSymbols& symbols, std::string* error) {
  if (symbols.wedge_quad < 0 || symbols.wedge_quad > 3 ||
      symbols.wedge_angle < 0 || symbols.wedge_angle > 4) {
    return fail(error, "wedge quad/angle out of range");
  }
  int index = 0;
  if (!wedge_index_from_symbols(symbols, &index)) {
    return fail(error, "wedge symbols do not map to a signallable index");
  }
  if (index != symbols.wedge_index) {
    return fail(error, "wedge_index does not match its symbols");
  }
  if (!sym(enc, cdfs.wedge_quad, 4, symbols.wedge_quad, error)) return false;
  if (!sym(enc, cdfs.wedge_angle[symbols.wedge_quad], 5, symbols.wedge_angle,
           error)) {
    return false;
  }
  if (symbols.use_dist2) {
    if (symbols.wedge_dist2 < 0 || symbols.wedge_dist2 > 2) {
      return fail(error, "wedge_dist2 out of range");
    }
    return sym(enc, cdfs.wedge_dist2, 3, symbols.wedge_dist2, error);
  }
  if (symbols.wedge_dist1 < 0 || symbols.wedge_dist1 > 3) {
    return fail(error, "wedge_dist1 out of range");
  }
  return sym(enc, cdfs.wedge_dist1, 4, symbols.wedge_dist1, error);
}

bool write_wedge_sign(av2::Encoder& enc, bool sign, std::string* error) {
  return raw_bit(enc, sign, error);
}

bool write_inter_intra(av2::Encoder& enc, AdvancedCdfState& cdfs, bool is_warp,
                       int mi_size, const MotionModeSymbols& symbols,
                       std::string* error) {
  if (mi_size < 0 || mi_size >= kBlockSizes) {
    return fail(error, "MiSize outside BLOCK_SIZES");
  }
  const int size_group = kSizeGroup[mi_size];
  const int w = partition::block_width(static_cast<partition::BlockSize>(mi_size));
  const int h = partition::block_height(static_cast<partition::BlockSize>(mi_size));
  const bool wedge_allowed = kWedgeBits[mi_size] != 0;

  if (is_warp) {
    if (w > 64 || h > 64) {
      if (symbols.inter_intra) {
        return fail(error, "warp_inter_intra is not codable for >64x64");
      }
      return true;
    }
    if (!sym(enc, cdfs.warp_inter_intra[size_group], 2,
             symbols.inter_intra ? 1 : 0, error)) {
      return false;
    }
    if (!symbols.inter_intra) return true;
  } else {
    if (!sym(enc, cdfs.inter_intra[size_group], 2,
             symbols.inter_intra ? 1 : 0, error)) {
      return false;
    }
    if (!symbols.inter_intra) return true;
  }

  if (symbols.interintra_mode < 0 || symbols.interintra_mode > 3) {
    return fail(error, "interintra_mode outside 0..3");
  }
  if (!sym(enc, cdfs.inter_intra_mode[size_group], 4, symbols.interintra_mode,
           error)) {
    return false;
  }
  if (!wedge_allowed) {
    if (symbols.wedge_interintra) {
      return fail(error, "wedge_interintra is not codable for this MiSize");
    }
    return true;
  }
  if (!sym(enc, cdfs.wedge_inter_intra, 2, symbols.wedge_interintra ? 1 : 0,
           error)) {
    return false;
  }
  if (!symbols.wedge_interintra) return true;

  WedgeModeSymbols wedge{};
  if (!wedge_mode_symbols(symbols.wedge_index, &wedge)) {
    return fail(error, "wedge_interintra index is not signallable");
  }
  return write_wedge_mode(enc, cdfs, wedge, error);
}

bool write_motion_mode(av2::Encoder& enc, AdvancedCdfState& cdfs, int mi_size,
                       const MotionModeSymbols& symbols, std::string* error) {
  if (mi_size < 0 || mi_size >= kBlockSizes) {
    return fail(error, "MiSize outside BLOCK_SIZES");
  }
  if (symbols.is_warp_newmv) {
    if (symbols.inter_intra) {
      return fail(error, "inter_intra cannot accompany WARP_NEWMV");
    }
    if (symbols.extend_allowed) {
      if (!sym(enc, cdfs.use_extend_warp[std::min(symbols.extend_warp_ctx, 2)],
               2, symbols.use_extend_warp ? 1 : 0, error)) {
        return false;
      }
      if (symbols.use_extend_warp) return true;
    } else if (symbols.use_extend_warp) {
      return fail(error, "use_extend_warp set while not allowed");
    }
    if (symbols.local_allowed) {
      if (!sym(enc, cdfs.use_local_warp[std::min(symbols.local_warp_ctx, 3)], 2,
               symbols.use_local_warp ? 1 : 0, error)) {
        return false;
      }
      if (symbols.use_local_warp) return true;
    } else if (symbols.use_local_warp) {
      return fail(error, "use_local_warp set while not allowed");
    }
    return true;  // DELTAWARP
  }

  if (symbols.inter_intra_allowed) {
    if (!write_inter_intra(enc, cdfs, /*is_warp=*/false, mi_size, symbols,
                           error)) {
      return false;
    }
    if (symbols.inter_intra) return true;
  } else if (symbols.inter_intra) {
    return fail(error, "inter_intra set while not allowed");
  }
  if (symbols.local_allowed) {
    if (!sym(enc, cdfs.use_local_warp[std::min(symbols.local_warp_ctx, 3)], 2,
             symbols.use_local_warp ? 1 : 0, error)) {
      return false;
    }
  } else if (symbols.use_local_warp) {
    return fail(error, "use_local_warp set while not allowed");
  }
  return true;
}

bool write_warp_idx(av2::Encoder& enc, AdvancedCdfState& cdfs, int ref_warp_idx,
                    std::string* error) {
  if (ref_warp_idx < 0 || ref_warp_idx > 3) {
    return fail(error, "RefWarpIdx outside 0..3");
  }
  for (int idx = 0; idx < kMaxWarpRefCandidates - 1; ++idx) {
    const bool more = ref_warp_idx == 3 || idx < ref_warp_idx;
    if (!sym(enc, cdfs.warp_idx[idx], 2, more ? 1 : 0, error)) return false;
    if (!more) break;
  }
  return true;
}

bool write_warpmv_with_mvd(av2::Encoder& enc, AdvancedCdfState& cdfs, bool value,
                           std::string* error) {
  return sym(enc, cdfs.warp_with_mvd, 2, value ? 1 : 0, error);
}

bool write_warp_delta(av2::Encoder& enc, AdvancedCdfState& cdfs, int mi_size,
                      const WarpDeltaSymbols& symbols, bool six_param,
                      std::string* error) {
  if (mi_size < 0 || mi_size >= kBlockSizes) {
    return fail(error, "MiSize outside BLOCK_SIZES");
  }
  if (!six_param) {
    for (int i = 2; i < 4; ++i) {
      if (symbols.magnitude[i] != 0 || symbols.negative[i]) {
        return fail(error, "4-parameter warp delta carries unused parameters");
      }
    }
  }
  if (!sym(enc, cdfs.warp_precision[mi_size], 2,
           symbols.high_precision ? 1 : 0, error)) {
    return false;
  }
  const int count = six_param ? 4 : 2;
  const int max_value = symbols.high_precision ? 14 : 7;
  for (int i = 0; i < count; ++i) {
    const int value = symbols.magnitude[i];
    if (value < 0 || value > max_value) {
      return fail(error, "warp delta magnitude out of range for its precision");
    }
    const int group = (i == 1 || i == 2) ? 1 : 0;
    const int low = std::min(value, kWarpDeltaNumSymbolsLow - 1);
    if (!sym(enc, cdfs.warp_delta_low[group], 8, low, error)) return false;
    if (symbols.high_precision && low == kWarpDeltaNumSymbolsLow - 1) {
      if (!sym(enc, cdfs.warp_delta_high[group], 8,
               value - kWarpDeltaNumSymbolsLow + 1, error)) {
        return false;
      }
    }
    if (value != 0) {
      if (!raw_bit(enc, symbols.negative[i], error)) return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

bool read_is_warp(av2::Decoder& dec, AdvancedCdfState& cdfs, int warp_mv_ctx,
                  bool* value, std::string* error) {
  if (warp_mv_ctx < 0 || warp_mv_ctx > 4) {
    return fail(error, "is_warp context outside 0..4");
  }
  int symbol = 0;
  if (!sym(dec, cdfs.is_warp[warp_mv_ctx], 2, &symbol, error)) return false;
  *value = symbol != 0;
  return true;
}

bool read_warp_mv(av2::Decoder& dec, AdvancedCdfState& cdfs, bool* value,
                  std::string* error) {
  int symbol = 0;
  if (!sym(dec, cdfs.warp_mv, 2, &symbol, error)) return false;
  *value = symbol != 0;
  return true;
}

bool read_bawp_symbols(av2::Decoder& dec, AdvancedCdfState& cdfs,
                       int explicit_ctx, bool has_chroma, BawpSymbols* out,
                       std::string* error) {
  if (out == nullptr || explicit_ctx < 0 || explicit_ctx > 2) {
    return fail(error, "invalid BAWP read arguments");
  }
  BawpSymbols s{};
  s.explicit_ctx = explicit_ctx;
  int symbol = 0;
  if (!sym(dec, cdfs.use_bawp, 2, &symbol, error)) return false;
  s.use_bawp = symbol != 0;
  if (s.use_bawp) {
    if (!sym(dec, cdfs.explicit_bawp[explicit_ctx], 2, &symbol, error)) {
      return false;
    }
    s.explicit_bawp = symbol != 0;
    if (s.explicit_bawp) {
      if (!sym(dec, cdfs.explicit_bawp_scale, 2, &symbol, error)) return false;
      s.explicit_bawp_scale = symbol != 0;
    }
    if (has_chroma) {
      if (!sym(dec, cdfs.use_bawp_chroma, 2, &symbol, error)) return false;
      s.use_bawp_chroma = symbol != 0;
    }
  }
  *out = s;
  return true;
}

bool read_wedge_mode(av2::Decoder& dec, AdvancedCdfState& cdfs,
                     WedgeModeSymbols* out, std::string* error) {
  if (out == nullptr) return fail(error, "null wedge output");
  WedgeModeSymbols s{};
  if (!sym(dec, cdfs.wedge_quad, 4, &s.wedge_quad, error)) return false;
  if (s.wedge_quad < 0 || s.wedge_quad > 3) {
    return fail(error, "decoded wedge_quad outside 0..3");
  }
  if (!sym(dec, cdfs.wedge_angle[s.wedge_quad], 5, &s.wedge_angle, error)) {
    return false;
  }
  const int angle = s.wedge_quad * 5 + s.wedge_angle;
  s.use_dist2 = (angle >= ia::kHwEdgeAngles) || (angle == 5) || (angle == 0);
  if (s.use_dist2) {
    if (!sym(dec, cdfs.wedge_dist2, 3, &s.wedge_dist2, error)) return false;
  } else {
    if (!sym(dec, cdfs.wedge_dist1, 4, &s.wedge_dist1, error)) return false;
  }
  if (!wedge_index_from_symbols(s, &s.wedge_index)) {
    return fail(error, "decoded wedge symbols do not map to an index");
  }
  *out = s;
  return true;
}

bool read_wedge_sign(av2::Decoder& dec, int* sign, std::string* error) {
  bool bit = false;
  if (!raw_bit(dec, &bit, error)) return false;
  *sign = bit ? 1 : 0;
  return true;
}

bool read_inter_intra(av2::Decoder& dec, AdvancedCdfState& cdfs, bool is_warp,
                      int mi_size, bool wedge_allowed, bool* inter_intra,
                      int* interintra_mode, bool* wedge_interintra,
                      int* wedge_index, std::string* error) {
  if (mi_size < 0 || mi_size >= kBlockSizes) {
    return fail(error, "MiSize outside BLOCK_SIZES");
  }
  const int size_group = kSizeGroup[mi_size];
  const int w = partition::block_width(static_cast<partition::BlockSize>(mi_size));
  const int h = partition::block_height(static_cast<partition::BlockSize>(mi_size));
  const bool can_wedge = wedge_allowed && kWedgeBits[mi_size] != 0;
  int symbol = 0;
  if (is_warp) {
    if (w > 64 || h > 64) {
      *inter_intra = false;
      return true;
    }
    if (!sym(dec, cdfs.warp_inter_intra[size_group], 2, &symbol, error)) {
      return false;
    }
  } else {
    if (!sym(dec, cdfs.inter_intra[size_group], 2, &symbol, error)) {
      return false;
    }
  }
  *inter_intra = symbol != 0;
  if (!*inter_intra) return true;

  if (!sym(dec, cdfs.inter_intra_mode[size_group], 4, &symbol, error)) {
    return false;
  }
  *interintra_mode = symbol;
  *wedge_interintra = false;
  *wedge_index = 0;
  if (!can_wedge) return true;
  if (!sym(dec, cdfs.wedge_inter_intra, 2, &symbol, error)) return false;
  *wedge_interintra = symbol != 0;
  if (*wedge_interintra) {
    WedgeModeSymbols wedge{};
    if (!read_wedge_mode(dec, cdfs, &wedge, error)) return false;
    *wedge_index = wedge.wedge_index;
  }
  return true;
}

bool read_motion_mode(av2::Decoder& dec, AdvancedCdfState& cdfs, int size_group,
                      bool is_warp_newmv, bool extend_allowed, bool local_allowed,
                      int extend_warp_ctx, int local_warp_ctx,
                      bool inter_intra_allowed, bool wedge_allowed,
                      MotionModeSymbols* out, std::string* error) {
  if (out == nullptr || size_group < 0 || size_group > 3) {
    return fail(error, "invalid motion-mode read arguments");
  }
  MotionModeSymbols s{};
  s.is_warp_newmv = is_warp_newmv;
  s.extend_allowed = extend_allowed;
  s.local_allowed = local_allowed;
  s.extend_warp_ctx = extend_warp_ctx;
  s.local_warp_ctx = local_warp_ctx;
  s.inter_intra_allowed = inter_intra_allowed;
  int symbol = 0;

  if (is_warp_newmv) {
    if (extend_allowed) {
      if (!sym(dec, cdfs.use_extend_warp[std::min(extend_warp_ctx, 2)], 2,
               &symbol, error)) {
        return false;
      }
      s.use_extend_warp = symbol != 0;
      if (s.use_extend_warp) {
        *out = s;
        return true;
      }
    }
    if (local_allowed) {
      if (!sym(dec, cdfs.use_local_warp[std::min(local_warp_ctx, 3)], 2,
               &symbol, error)) {
        return false;
      }
      s.use_local_warp = symbol != 0;
    }
    *out = s;
    return true;
  }

  if (inter_intra_allowed) {
    // The main inter_intra bit decides whether the rest is present.
    if (!sym(dec, cdfs.inter_intra[size_group], 2, &symbol, error)) return false;
    s.inter_intra = symbol != 0;
    if (s.inter_intra) {
      if (!sym(dec, cdfs.inter_intra_mode[size_group], 4, &symbol, error)) {
        return false;
      }
      s.interintra_mode = symbol;
      if (wedge_allowed) {
        if (!sym(dec, cdfs.wedge_inter_intra, 2, &symbol, error)) return false;
        s.wedge_interintra = symbol != 0;
        if (s.wedge_interintra) {
          WedgeModeSymbols wedge{};
          if (!read_wedge_mode(dec, cdfs, &wedge, error)) return false;
          s.wedge_index = wedge.wedge_index;
        }
      }
      *out = s;
      return true;
    }
  }
  if (local_allowed) {
    if (!sym(dec, cdfs.use_local_warp[std::min(local_warp_ctx, 3)], 2, &symbol,
             error)) {
      return false;
    }
    s.use_local_warp = symbol != 0;
  }
  *out = s;
  return true;
}

bool read_warp_idx(av2::Decoder& dec, AdvancedCdfState& cdfs, int* ref_warp_idx,
                   std::string* error) {
  if (ref_warp_idx == nullptr) return fail(error, "null RefWarpIdx output");
  *ref_warp_idx = 3;
  for (int idx = 0; idx < kMaxWarpRefCandidates - 1; ++idx) {
    int symbol = 0;
    if (!sym(dec, cdfs.warp_idx[idx], 2, &symbol, error)) return false;
    if (symbol == 0) {
      *ref_warp_idx = idx;
      return true;
    }
  }
  return true;
}

bool read_warpmv_with_mvd(av2::Decoder& dec, AdvancedCdfState& cdfs, bool* value,
                          std::string* error) {
  int symbol = 0;
  if (!sym(dec, cdfs.warp_with_mvd, 2, &symbol, error)) return false;
  *value = symbol != 0;
  return true;
}

bool read_warp_delta(av2::Decoder& dec, AdvancedCdfState& cdfs, int mi_size,
                     bool six_param, WarpDeltaSymbols* out, std::string* error) {
  if (out == nullptr || mi_size < 0 || mi_size >= kBlockSizes) {
    return fail(error, "invalid warp delta read arguments");
  }
  WarpDeltaSymbols s{};
  int symbol = 0;
  if (!sym(dec, cdfs.warp_precision[mi_size], 2, &symbol, error)) return false;
  s.high_precision = symbol != 0;
  const int count = six_param ? 4 : 2;
  for (int i = 0; i < count; ++i) {
    const int group = (i == 1 || i == 2) ? 1 : 0;
    if (!sym(dec, cdfs.warp_delta_low[group], 8, &symbol, error)) return false;
    int value = symbol;
    if (s.high_precision && symbol == kWarpDeltaNumSymbolsLow - 1) {
      if (!sym(dec, cdfs.warp_delta_high[group], 8, &symbol, error)) return false;
      value += symbol;
    }
    s.magnitude[i] = value;
    if (value != 0) {
      bool negative = false;
      if (!raw_bit(dec, &negative, error)) return false;
      s.negative[i] = negative;
    }
  }
  *out = s;
  return true;
}

// ---------------------------------------------------------------------------
// Full ordered block writer/reader
// ---------------------------------------------------------------------------

namespace {

int derive_y_mode(const AdvancedBlockCoding& block) {
  if (block.is_warp) {
    return static_cast<int>(block.warp_mv ? YMode::kWarpMv : YMode::kWarpNewMv);
  }
  return static_cast<int>(YMode::kNearMv) + block.single_mode;
}

bool bawp_available(const AdvancedBlockCoding& block, int y_mode) {
  const int w = partition::block_width(
      static_cast<partition::BlockSize>(block.mi_size));
  const int h = partition::block_height(
      static_cast<partition::BlockSize>(block.mi_size));
  return block.allow_bawp && !block.ref_scaled && !block.frame_is_switch &&
         w >= 8 && h >= 8 && y_mode != static_cast<int>(YMode::kGlobalMv);
}

bool has_mvd(int y_mode, bool warpmv_with_mvd) {
  return y_mode == static_cast<int>(YMode::kNewMv) ||
         y_mode == static_cast<int>(YMode::kWarpNewMv) ||
         (y_mode == static_cast<int>(YMode::kWarpMv) && warpmv_with_mvd);
}

}  // namespace

bool write_advanced_inter_block(av2::Encoder& enc, AdvancedCdfState& cdfs,
                                const AdvancedBlockCoding& block,
                                const BaselineInterWriter& baseline,
                                std::string* error) {
  if (block.mi_size < 0 || block.mi_size >= kBlockSizes) {
    return fail(error, "MiSize outside BLOCK_SIZES");
  }
  const int w = partition::block_width(
      static_cast<partition::BlockSize>(block.mi_size));
  const int h = partition::block_height(
      static_cast<partition::BlockSize>(block.mi_size));
  const bool can_warp = block.allow_warpmv_mode && std::min(w, h) >= 8;

  int y_mode = 0;
  if (block.is_warp) {
    if (!can_warp) {
      return fail(error, "is_warp outside allow_warpmv_mode/min-size gate");
    }
    if (block.bawp.use_bawp || block.bawp.explicit_bawp ||
        block.bawp.use_bawp_chroma) {
      return fail(error, "BAWP symbols set on a warp block");
    }
    if (!write_is_warp(enc, cdfs, block.warp_mv_ctx, true, error)) return false;
    if (block.force_integer_mv) {
      if (!block.warp_mv) {
        return fail(error, "force_integer_mv requires warp_mv = 1");
      }
    } else if (!write_warp_mv(enc, cdfs, block.warp_mv, error)) {
      return false;
    }
    y_mode = derive_y_mode(block);
  } else {
    if (can_warp && !write_is_warp(enc, cdfs, block.warp_mv_ctx, false, error)) {
      return false;
    }
    if (block.single_mode < 0 || block.single_mode > 2) {
      return fail(error, "single_mode outside 0..2");
    }
    if (baseline.single_mode == nullptr) {
      return fail(error, "baseline single_mode callback is required");
    }
    if (!baseline.single_mode(baseline.ctx, block.new_mv_ctx,
                              block.single_mode)) {
      return fail(error, "baseline single_mode callback failed");
    }
    y_mode = derive_y_mode(block);
    if (bawp_available(block, y_mode)) {
      if (!write_bawp_symbols(enc, cdfs, block.bawp, block.has_chroma,
                              error)) {
        return false;
      }
    } else if (block.bawp.use_bawp || block.bawp.explicit_bawp ||
               block.bawp.use_bawp_chroma) {
      return fail(error, "BAWP symbols set outside the §5.20.7.6 gate");
    }
  }

  int motion_mode = 0;
  if (y_mode == static_cast<int>(YMode::kWarpMv)) {
    if (block.motion.is_warp_newmv) {
      return fail(error, "motion plan says WARP_NEWMV for a WARPMV block");
    }
    motion_mode = static_cast<int>(ia::MotionMode::kDeltaWarp);
  } else if (y_mode == static_cast<int>(YMode::kWarpNewMv)) {
    if (block.motion.inter_intra) {
      return fail(error, "inter_intra cannot accompany WARP_NEWMV");
    }
    if (block.motion.use_extend_warp && block.motion.use_local_warp) {
      return fail(error, "EXTENDWARP and LOCALWARP are mutually exclusive");
    }
    MotionModeSymbols motion = block.motion;
    motion.is_warp_newmv = true;
    motion.inter_intra = false;
    if (!write_motion_mode(enc, cdfs, block.mi_size, motion, error)) {
      return false;
    }
    motion_mode = motion.use_extend_warp
                      ? static_cast<int>(ia::MotionMode::kExtendWarp)
                      : (motion.use_local_warp
                             ? static_cast<int>(ia::MotionMode::kLocalWarp)
                             : static_cast<int>(ia::MotionMode::kDeltaWarp));
  } else {
    if (block.motion.is_warp_newmv) {
      return fail(error, "motion plan says WARP_NEWMV for a single-mode block");
    }
    if (block.motion.use_extend_warp) {
      return fail(error, "use_extend_warp requires WARP_NEWMV");
    }
    MotionModeSymbols motion = block.motion;
    motion.is_warp_newmv = false;
    if (!write_motion_mode(enc, cdfs, block.mi_size, motion, error)) {
      return false;
    }
    motion_mode = block.motion.inter_intra
                      ? static_cast<int>(ia::MotionMode::kInterIntra)
                      : (block.motion.use_local_warp
                             ? static_cast<int>(ia::MotionMode::kLocalWarp)
                             : static_cast<int>(ia::MotionMode::kSimple));
  }

  if (y_mode == static_cast<int>(YMode::kWarpMv) ||
      motion_mode == static_cast<int>(ia::MotionMode::kDeltaWarp)) {
    if (!write_warp_idx(enc, cdfs, block.warp_idx, error)) return false;
  }

  const bool with_mvd = y_mode == static_cast<int>(YMode::kWarpMv) &&
                        block.warp_idx < 2 && block.warpmv_with_mvd;
  if (y_mode == static_cast<int>(YMode::kWarpMv) && block.warp_idx < 2) {
    if (!write_warpmv_with_mvd(enc, cdfs, block.warpmv_with_mvd, error)) {
      return false;
    }
  } else if (block.warpmv_with_mvd) {
    return fail(error, "warpmv_with_mvd outside WARPMV/RefWarpIdx<2");
  }

  const bool drl_present = y_mode == static_cast<int>(YMode::kNearMv) ||
                           y_mode == static_cast<int>(YMode::kNewMv) ||
                           y_mode == static_cast<int>(YMode::kWarpNewMv);
  if (drl_present) {
    if (baseline.drl_index == nullptr) {
      return fail(error, "baseline drl_index callback is required");
    }
    if (!baseline.drl_index(baseline.ctx, block.new_mv_ctx, block.drl_index)) {
      return fail(error, "baseline drl_index callback failed");
    }
  } else if (block.drl_index != 0) {
    return fail(error, "drl_index set for a mode without DRL syntax");
  }

  if (has_mvd(y_mode, with_mvd)) {
    if (baseline.mvd == nullptr) {
      return fail(error, "baseline mvd callback is required");
    }
    if (!baseline.mvd(baseline.ctx, block.mvd_y, block.mvd_x)) {
      return fail(error, "baseline mvd callback failed");
    }
  } else if (block.mvd_y != 0 || block.mvd_x != 0) {
    return fail(error, "MVD set for a mode without MVD syntax");
  }

  if (motion_mode == static_cast<int>(ia::MotionMode::kDeltaWarp) &&
      y_mode == static_cast<int>(YMode::kWarpNewMv)) {
    if (block.six_param && block.warp_idx != 1) {
      return fail(error, "six-parameter delta requires RefWarpIdx == 1");
    }
    if (!write_warp_delta(enc, cdfs, block.mi_size, block.warp_delta,
                          block.six_param, error)) {
      return false;
    }
  } else {
    if (block.six_param) {
      return fail(error, "six_param set without a WARP_NEWMV delta");
    }
    for (int i = 0; i < 4; ++i) {
      if (block.warp_delta.magnitude[i] != 0) {
        return fail(error, "warp delta set without a WARP_NEWMV delta");
      }
    }
  }

  if (y_mode == static_cast<int>(YMode::kWarpMv)) {
    if (!write_inter_intra(enc, cdfs, /*is_warp=*/true, block.mi_size,
                           block.motion, error)) {
      return false;
    }
  }
  return true;
}

bool read_advanced_inter_block(av2::Decoder& dec, AdvancedCdfState& cdfs,
                               const AdvancedBlockCoding& gates,
                               const BaselineInterReader& baseline,
                               AdvancedBlockCoding* out, std::string* error) {
  if (out == nullptr || gates.mi_size < 0 || gates.mi_size >= kBlockSizes) {
    return fail(error, "invalid advanced block read arguments");
  }
  AdvancedBlockCoding block = gates;
  const int bawp_explicit_ctx = gates.bawp.explicit_ctx;
  block.bawp = BawpSymbols{};
  block.bawp.explicit_ctx = bawp_explicit_ctx;
  block.motion = gates.motion;
  block.motion.is_warp_newmv = false;
  block.motion.use_extend_warp = false;
  block.motion.use_local_warp = false;
  block.motion.inter_intra = false;
  block.motion.interintra_mode = 0;
  block.motion.wedge_interintra = false;
  block.motion.wedge_index = 0;
  block.warp_delta = WarpDeltaSymbols{};
  block.drl_index = 0;
  block.mvd_y = 0;
  block.mvd_x = 0;

  const int w = partition::block_width(
      static_cast<partition::BlockSize>(block.mi_size));
  const int h = partition::block_height(
      static_cast<partition::BlockSize>(block.mi_size));
  const bool can_warp = block.allow_warpmv_mode && std::min(w, h) >= 8;

  if (can_warp) {
    if (!read_is_warp(dec, cdfs, block.warp_mv_ctx, &block.is_warp, error)) {
      return false;
    }
  } else {
    block.is_warp = false;
  }

  int y_mode = 0;
  if (block.is_warp) {
    if (block.force_integer_mv) {
      block.warp_mv = true;
    } else if (!read_warp_mv(dec, cdfs, &block.warp_mv, error)) {
      return false;
    }
    y_mode = static_cast<int>(block.warp_mv ? YMode::kWarpMv
                                            : YMode::kWarpNewMv);
  } else {
    if (baseline.single_mode == nullptr) {
      return fail(error, "baseline single_mode callback is required");
    }
    if (!baseline.single_mode(baseline.ctx, block.new_mv_ctx,
                              &block.single_mode)) {
      return fail(error, "baseline single_mode callback failed");
    }
    if (block.single_mode < 0 || block.single_mode > 2) {
      return fail(error, "decoded single_mode outside 0..2");
    }
    y_mode = static_cast<int>(YMode::kNearMv) + block.single_mode;
    if (bawp_available(block, y_mode)) {
      if (!read_bawp_symbols(dec, cdfs, bawp_explicit_ctx, block.has_chroma,
                             &block.bawp, error)) {
        return false;
      }
    }
  }
  block.y_mode = y_mode;

  int motion_mode = 0;
  if (y_mode == static_cast<int>(YMode::kWarpMv)) {
    motion_mode = static_cast<int>(ia::MotionMode::kDeltaWarp);
  } else if (y_mode == static_cast<int>(YMode::kWarpNewMv)) {
    MotionModeSymbols motion{};
    if (!read_motion_mode(dec, cdfs, kSizeGroup[block.mi_size], true,
                          block.motion.extend_allowed,
                          block.motion.local_allowed, block.motion.extend_warp_ctx,
                          block.motion.local_warp_ctx, false,
                          block.motion.wedge_allowed, &motion, error)) {
      return false;
    }
    block.motion = motion;
    block.motion.is_warp_newmv = true;
    motion_mode = block.motion.use_extend_warp
                      ? static_cast<int>(ia::MotionMode::kExtendWarp)
                      : (block.motion.use_local_warp
                             ? static_cast<int>(ia::MotionMode::kLocalWarp)
                             : static_cast<int>(ia::MotionMode::kDeltaWarp));
  } else {
    MotionModeSymbols motion{};
    if (!read_motion_mode(dec, cdfs, kSizeGroup[block.mi_size], false, false,
                          block.motion.local_allowed, 0,
                          block.motion.local_warp_ctx,
                          block.motion.inter_intra_allowed,
                          block.motion.wedge_allowed, &motion, error)) {
      return false;
    }
    block.motion = motion;
    motion_mode = block.motion.inter_intra
                      ? static_cast<int>(ia::MotionMode::kInterIntra)
                      : (block.motion.use_local_warp
                             ? static_cast<int>(ia::MotionMode::kLocalWarp)
                             : static_cast<int>(ia::MotionMode::kSimple));
  }
  block.motion_mode = motion_mode;

  if (y_mode == static_cast<int>(YMode::kWarpMv) ||
      motion_mode == static_cast<int>(ia::MotionMode::kDeltaWarp)) {
    if (!read_warp_idx(dec, cdfs, &block.warp_idx, error)) return false;
  }

  if (y_mode == static_cast<int>(YMode::kWarpMv) && block.warp_idx < 2) {
    if (!read_warpmv_with_mvd(dec, cdfs, &block.warpmv_with_mvd, error)) {
      return false;
    }
  } else {
    block.warpmv_with_mvd = false;
  }

  const bool drl_present = y_mode == static_cast<int>(YMode::kNearMv) ||
                           y_mode == static_cast<int>(YMode::kNewMv) ||
                           y_mode == static_cast<int>(YMode::kWarpNewMv);
  if (drl_present) {
    if (baseline.drl_index == nullptr ||
        !baseline.drl_index(baseline.ctx, block.new_mv_ctx, &block.drl_index)) {
      return fail(error, "baseline drl_index callback failed");
    }
  }

  if (has_mvd(y_mode, block.warpmv_with_mvd)) {
    if (baseline.mvd == nullptr ||
        !baseline.mvd(baseline.ctx, &block.mvd_y, &block.mvd_x)) {
      return fail(error, "baseline mvd callback failed");
    }
  }

  if (motion_mode == static_cast<int>(ia::MotionMode::kDeltaWarp) &&
      y_mode == static_cast<int>(YMode::kWarpNewMv)) {
    if (!read_warp_delta(dec, cdfs, block.mi_size,
                         block.six_param && block.warp_idx == 1,
                         &block.warp_delta, error)) {
      return false;
    }
  }

  if (y_mode == static_cast<int>(YMode::kWarpMv)) {
    bool inter_intra = false;
    int mode = 0;
    bool wedge = false;
    int index = 0;
    if (!read_inter_intra(dec, cdfs, true, block.mi_size,
                          block.motion.wedge_allowed, &inter_intra, &mode,
                          &wedge, &index, error)) {
      return false;
    }
    block.motion.inter_intra = inter_intra;
    block.motion.interintra_mode = mode;
    block.motion.wedge_interintra = wedge;
    block.motion.wedge_index = index;
  }

  *out = block;
  return true;
}

// ---------------------------------------------------------------------------
// Sequence / frame header bits
// ---------------------------------------------------------------------------

bool write_sequence_motion_mode_config(av2enc::BitWriter* writer,
                                       const SequenceAdvancedConfig& config,
                                       std::string* error) {
  if (writer == nullptr) return fail(error, "null BitWriter");
  bool any_mode = false;
  for (int mode = 1; mode < kMotionModes; ++mode) {
    writer->put_bit(config.seq_enabled_motion_modes[mode]);
    any_mode = any_mode || config.seq_enabled_motion_modes[mode];
  }
  if (any_mode) {
    writer->put_bit(config.seq_frame_motion_modes_present_flag);
  } else if (config.seq_frame_motion_modes_present_flag) {
    return fail(error, "seq_frame_motion_modes_present_flag with no modes");
  }
  if (config.seq_enabled_motion_modes[static_cast<int>(ia::MotionMode::kDeltaWarp)]) {
    writer->put_bit(config.enable_six_param_warp_delta);
  } else if (config.enable_six_param_warp_delta) {
    return fail(error, "enable_six_param_warp_delta without DELTAWARP");
  }
  writer->put_bit(config.enable_masked_compound);
  return true;
}

bool write_sequence_bawp_cwp_bits(av2enc::BitWriter* writer,
                                  const SequenceAdvancedConfig& config,
                                  std::string* error) {
  if (writer == nullptr) return fail(error, "null BitWriter");
  writer->put_bit(config.enable_bawp);
  writer->put_bit(config.enable_cwp);
  writer->put_bit(config.enable_imp_msk_bld);
  return true;
}

bool write_sequence_global_motion_bit(av2enc::BitWriter* writer,
                                      const SequenceAdvancedConfig& config,
                                      std::string* error) {
  if (writer == nullptr) return fail(error, "null BitWriter");
  if (config.enable_global_motion) {
    return fail(error,
                "global motion estimation is outside this module; "
                "enable_global_motion must stay 0");
  }
  writer->put_bit(false);
  return true;
}

bool write_frame_motion_mode_bits(av2enc::BitWriter* writer,
                                  const FrameAdvancedConfig& config,
                                  std::string* error) {
  if (writer == nullptr) return fail(error, "null BitWriter");
  for (int mode = 1; mode < kMotionModes; ++mode) {
    if (!config.seq_enabled_motion_modes[mode]) {
      if (config.frame_enabled_motion_modes[mode]) {
        return fail(error, "frame motion mode enabled but not in sequence");
      }
      continue;
    }
    if (config.seq_frame_motion_modes_present_flag) {
      writer->put_bit(config.frame_enabled_motion_modes[mode]);
    } else if (config.frame_enabled_motion_modes[mode] !=
               config.seq_enabled_motion_modes[mode]) {
      return fail(error, "frame motion mode differs without a present flag");
    }
  }
  return true;
}

bool write_frame_bawp_warp_bits(av2enc::BitWriter* writer,
                                const FrameAdvancedConfig& config,
                                std::string* error) {
  if (writer == nullptr) return fail(error, "null BitWriter");
  const bool bawp_present = !config.frame_is_intra && config.enable_bawp;
  if (bawp_present) {
    writer->put_bit(config.allow_bawp);
  } else if (config.allow_bawp) {
    return fail(error, "allow_bawp set while BAWP is unavailable");
  }
  const bool warp_present =
      !config.frame_is_intra &&
      config.frame_enabled_motion_modes[static_cast<int>(ia::MotionMode::kDeltaWarp)];
  if (warp_present) {
    writer->put_bit(config.allow_warpmv_mode);
  } else if (config.allow_warpmv_mode) {
    return fail(error, "allow_warpmv_mode set while DELTAWARP is unavailable");
  }
  return true;
}

}  // namespace av2enc::m2::adv
