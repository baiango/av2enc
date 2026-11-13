// M1 filter/restoration syntax writers and mirrors.
//
// See filters_stream.h and README.md.  The header writers use the raw
// descriptors in bit_io.h; the tile functions use the normative §8.2 coder in
// src/entropy/av2.  The CDEF per-block context is the single implementation in
// src/cuda/m1/cdef/still_cdef_index_syntax.h.

#include "codec/filters/stream/filters_stream.h"

#include "cuda/pipeline/still/cdef/still_cdef_cdf.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace av2enc::still::filters_stream {
namespace {

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return false;
}

// Works for arrays of any rank: T is the complete element type, so a 2-D or
// 3-D CDF array binds with N = outer dimension.
template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

bool valid_num_filter_classes(int n) {
  for (int i = 0; i < 8; ++i) {
    if (restoration::kDecodeNumFilterClasses[i] == n) return true;
  }
  return false;
}

int num_filter_classes_index(int n) {
  for (int i = 0; i < 8; ++i) {
    if (restoration::kDecodeNumFilterClasses[i] == n) return i;
  }
  return -1;
}

// §5.18.7.11 Decode_Num_Filter_Classes is fit by f(3).
int decoder_num_filter_classes(int idx) {
  if (idx < 0 || idx > 7) return 0;
  return restoration::kDecodeNumFilterClasses[idx];
}

// §7.20.3 Wiener_Ns_Taps_Present[chroma][subset][coeff].
constexpr std::int8_t kTapsPresent[2][4][restoration::kWienerNsChromaCoeffs] = {
    {
        {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
    },
    {
        {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
};

int predict_group(const int counts[3]) {
  int pred = 0;
  for (int i = 1; i <= 2; ++i) {
    if (counts[i] > counts[pred]) pred = i;
  }
  return pred;
}

// §5.18.9.6 inverse_recenter().
int inverse_recenter(int r, int v) {
  if (v > 2 * r) return v;
  if ((v & 1) != 0) return r - ((v + 1) >> 1);
  return r + (v >> 1);
}

// §5.18.9.4 decode_unsigned_subexp_with_ref(), value form.
int unsigned_subexp_with_ref_value(int mx, int r, int /*k*/, int v) {
  if ((r << 1) <= mx) return inverse_recenter(r, v);
  return mx - 1 - inverse_recenter(mx - 1 - r, v);
}

// §5.18.9.5 decode_subexp() inverse.  Writes the raw bit sequence.
bool encode_subexp(BitWriter& writer, int num_syms, int k, int value,
                   std::string* error) {
  if (num_syms <= 0 || value < 0 || value >= num_syms) {
    return fail(error, "subexp value out of range");
  }
  int i = 0;
  int mk = 0;
  for (;;) {
    const int b2 = (i != 0) ? k + i - 1 : k;
    if (b2 < 0 || b2 > 30) {
      return fail(error, "subexp bit width out of range");
    }
    const int a = 1 << b2;
    if (num_syms <= mk + 3 * a) {
      const int n = num_syms - mk;
      if (value < mk || value >= num_syms) {
        return fail(error, "subexp final value out of range");
      }
      put_ns(writer, static_cast<unsigned>(n),
             static_cast<std::uint32_t>(value - mk));
      return true;
    }
    if (value >= mk + a) {
      writer.put_bit(true);  // subexp_more_bits
      ++i;
      mk += a;
    } else {
      writer.put_bit(false);
      writer.put_bits(static_cast<std::uint64_t>(value - mk),
                      static_cast<unsigned>(b2));
      return true;
    }
  }
}

// §5.18.9.5 decode_subexp().
bool decode_subexp(BitReader& reader, int num_syms, int k, int* value,
                   std::string* error) {
  if (value == nullptr || num_syms <= 0) {
    return fail(error, "invalid subexp argument");
  }
  int i = 0;
  int mk = 0;
  for (;;) {
    const int b2 = (i != 0) ? k + i - 1 : k;
    if (b2 < 0 || b2 > 30) {
      return fail(error, "subexp bit width out of range");
    }
    const int a = 1 << b2;
    if (num_syms <= mk + 3 * a) {
      const int n = num_syms - mk;
      std::uint32_t bits = 0;
      if (!reader.read_ns(static_cast<unsigned>(n), &bits)) {
        return fail(error, "truncated subexp final bits");
      }
      *value = mk + static_cast<int>(bits);
      return true;
    }
    bool more = false;
    if (!reader.read_bit(&more)) {
      return fail(error, "truncated subexp more bit");
    }
    if (more) {
      ++i;
      mk += a;
    } else {
      std::uint32_t bits = 0;
      if (!reader.read_bits(static_cast<unsigned>(b2), &bits)) {
        return fail(error, "truncated subexp bits");
      }
      *value = mk + static_cast<int>(bits);
      return true;
    }
  }
}

// §5.18.9.3 decode_signed_subexp_with_ref(), inverse.
bool encode_signed_subexp_with_ref(BitWriter& writer, int low, int high,
                                   int r, int k, int target,
                                   std::string* error) {
  if (high <= low || target < low || target >= high) {
    return fail(error, "signed subexp target out of range");
  }
  const int mx = high - low;
  const int r_off = r - low;
  const int x_off = target - low;
  for (int v = 0; v < mx; ++v) {
    if (unsigned_subexp_with_ref_value(mx, r_off, k, v) == x_off) {
      return encode_subexp(writer, mx, k, v, error);
    }
  }
  return fail(error, "signed subexp has no encoding (bad reference)");
}

bool decode_signed_subexp_with_ref(BitReader& reader, int low, int high,
                                   int r, int k, int* target,
                                   std::string* error) {
  if (target == nullptr || high <= low) {
    return fail(error, "invalid signed subexp argument");
  }
  int v = 0;
  if (!decode_subexp(reader, high - low, k, &v, error)) return false;
  *target = low + unsigned_subexp_with_ref_value(high - low, r - low, k, v);
  return true;
}

// §5.18.9.7 decode_4part() / decode_unsigned_4part() / decode_signed_4part().
struct FourPartCode {
  int base = 0;
  int rem = 0;
  int bits = 0;
  int value = 0;  // decode_4part() result
};

bool decode_4part_value(int num, int base, int rem, FourPartCode* out) {
  if (out == nullptr) return false;
  const int bits = 2 - num + std::max(1, base);
  if (bits < 1 || bits > 15) return false;
  const int offset = (base == 0) ? 0 : (1 << bits);
  if (rem < 0 || rem >= (1 << bits)) return false;
  out->base = base;
  out->rem = rem;
  out->bits = bits;
  out->value = offset + rem;
  return true;
}

int unsigned_4part_value(int k, int r, int v) {
  const int mx = 1 << k;
  if ((r << 1) <= mx) return inverse_recenter(r, v);
  return mx - 1 - inverse_recenter(mx - 1 - r, v);
}

bool encode_signed_4part(std::uint16_t* base_cdf, int low, int k, int r,
                         int target, av2::Encoder& encoder,
                         std::string* error) {
  const int r_off = r - low;
  const int x_off = target - low;
  FourPartCode found;
  bool have = false;
  for (int base = 0; base < 4 && !have; ++base) {
    const int bits = 2 - (6 - k) + std::max(1, base);
    for (int rem = 0; rem < (1 << bits); ++rem) {
      FourPartCode code;
      if (!decode_4part_value(6 - k, base, rem, &code)) continue;
      if (unsigned_4part_value(k, r_off, code.value) == x_off) {
        found = code;
        have = true;
        break;
      }
    }
  }
  if (!have) return fail(error, "signed 4part has no encoding");
  const av2::Status status =
      encoder.encode_symbol(base_cdf, 4, found.base);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("wiener_ns_base encode failed: ") +
                           av2::status_name(status));
  }
  for (int i = found.bits - 1; i >= 0; --i) {
    const bool bit = ((found.rem >> i) & 1) != 0;
    const av2::Status bit_status = encoder.encode_bool(bit);
    if (bit_status != av2::Status::kOk) {
      return fail(error, std::string("wiener_ns_rem encode failed: ") +
                             av2::status_name(bit_status));
    }
  }
  return true;
}

bool decode_signed_4part(std::uint16_t* base_cdf, int low, int k, int r,
                         av2::Decoder& decoder, int* target,
                         std::string* error) {
  int base = 0;
  av2::Status status = decoder.decode_symbol(base_cdf, 4, &base);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("wiener_ns_base decode failed: ") +
                           av2::status_name(status));
  }
  const int bits = 2 - (6 - k) + std::max(1, base);
  std::uint32_t rem = 0;
  status = decoder.read_literal(bits, &rem);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("wiener_ns_rem decode failed: ") +
                           av2::status_name(status));
  }
  FourPartCode code;
  if (!decode_4part_value(6 - k, base, static_cast<int>(rem), &code)) {
    return fail(error, "invalid wiener_ns_base/rem combination");
  }
  if (target != nullptr) {
    *target = low + unsigned_4part_value(k, r - low, code.value);
  }
  return true;
}

bool raw_bit(av2::Encoder& encoder, bool bit, std::string* error) {
  const av2::Status status = encoder.encode_bool(bit);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("raw bit encode failed: ") +
                           av2::status_name(status));
  }
  return true;
}

bool raw_bit(av2::Decoder& decoder, bool* bit, std::string* error) {
  const av2::Status status = decoder.read_bool(bit);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("raw bit decode failed: ") +
                           av2::status_name(status));
  }
  return true;
}

}  // namespace

// ===========================================================================
// §5.4.10 sequence_filter_config()
// ===========================================================================

namespace {
constexpr int kRestoreSwitchableTypes = kRestoreSwitchable;  // §3 constant
}  // namespace

std::size_t write_sequence_filter_config(BitWriter& writer,
                                         const SequenceFilterConfig& config,
                                         bool single_picture_header_flag,
                                         std::string* error) {
  if (!single_picture_header_flag && config.cdef_on_skip_txfm_always_on &&
      config.cdef_on_skip_txfm_disabled) {
    fail(error, "invalid cdef_on_skip_txfm flags");
    return 0;
  }
  const std::size_t start = writer.bit_size();
  writer.put_bit(config.disable_loopfilters_across_tiles);
  writer.put_bit(config.enable_cdef);
  writer.put_bit(config.enable_gdf);
  if (config.enable_gdf) {
    // sb_64x64 is the only superblock size the M2 video path signals.
    writer.put_bit(config.gdf_unit_matches_sb_size);
  }
  writer.put_bit(config.enable_restoration);
  if (config.enable_restoration) {
    writer.put_bit(config.lr_disable_luma_pc_wiener);
    writer.put_bit(config.lr_disable_luma_wiener_ns);
    // lr_tools_disable[1][RESTORE_PC_WIENER] is inferred to 1.
    writer.put_bit(config.lr_uv_present);
    if (config.lr_uv_present) {
      writer.put_bit(config.lr_disable_chroma_wiener_ns);
    }
  }
  writer.put_bit(config.enable_ccso);
  if (config.enable_ccso) {
    writer.put_bit(config.ccso_unit_matches_sb_size);
  }
  if (!single_picture_header_flag) {
    // CdefOnSkipTxfm: ALWAYS_ON, else DISABLED, else ADAPTIVE.
    writer.put_bit(config.cdef_on_skip_txfm_always_on);
    if (!config.cdef_on_skip_txfm_always_on) {
      writer.put_bit(config.cdef_on_skip_txfm_disabled);
    }
  }
  writer.put_bits(config.df_par_bits_minus_2, 2U);
  return writer.bit_size() - start;
}

bool read_sequence_filter_config(BitReader& reader,
                                 bool single_picture_header_flag,
                                 bool sb_64x64,
                                 SequenceFilterConfig* config,
                                 std::string* error) {
  if (config == nullptr) return fail(error, "null SequenceFilterConfig");
  bool gdf_unit_matches = false;
  bool bit = false;
  if (!reader.read_bit(&bit)) return fail(error, "truncated sequence filter");
  config->disable_loopfilters_across_tiles = bit;
  if (!reader.read_bit(&config->enable_cdef)) {
    return fail(error, "truncated enable_cdef");
  }
  if (!reader.read_bit(&config->enable_gdf)) {
    return fail(error, "truncated enable_gdf");
  }
  config->gdf_unit_matches_sb_size = true;
  if (config->enable_gdf && sb_64x64) {
    if (!reader.read_bit(&config->gdf_unit_matches_sb_size)) {
      return fail(error, "truncated gdf_unit_matches_sb_size");
    }
  }
  if (!reader.read_bit(&config->enable_restoration)) {
    return fail(error, "truncated enable_restoration");
  }
  if (config->enable_restoration) {
    if (!reader.read_bit(&config->lr_disable_luma_pc_wiener) ||
        !reader.read_bit(&config->lr_disable_luma_wiener_ns) ||
        !reader.read_bit(&config->lr_uv_present)) {
      return fail(error, "truncated lr_tools bits");
    }
    if (config->lr_uv_present) {
      if (!reader.read_bit(&config->lr_disable_chroma_wiener_ns)) {
        return fail(error, "truncated lr_tools_uv bits");
      }
    } else {
      config->lr_disable_chroma_wiener_ns =
          config->lr_disable_luma_wiener_ns;
    }
  }
  if (!reader.read_bit(&config->enable_ccso)) {
    return fail(error, "truncated enable_ccso");
  }
  if (config->enable_ccso) {
    if (!reader.read_bit(&config->ccso_unit_matches_sb_size)) {
      return fail(error, "truncated ccso_unit_matches_sb_size");
    }
  }
  if (!single_picture_header_flag) {
    bool always_on = false;
    bool disabled = false;
    if (!reader.read_bit(&always_on)) {
      return fail(error, "truncated cdef_on_skip_txfm_always_on");
    }
    if (!always_on) {
      if (!reader.read_bit(&disabled)) {
        return fail(error, "truncated cdef_on_skip_txfm_disabled");
      }
    }
    config->cdef_on_skip_txfm_always_on = always_on;
    config->cdef_on_skip_txfm_disabled = disabled;
  }
  std::uint32_t df_par = 0;
  if (!reader.read_bits(2, &df_par)) {
    return fail(error, "truncated df_par_bits_minus_2");
  }
  config->df_par_bits_minus_2 = static_cast<std::uint8_t>(df_par);
  return true;
}

// ===========================================================================
// §5.18.7.10 cdef_params()
// ===========================================================================

std::size_t write_cdef_params(BitWriter& writer, const CdefParams& params,
                              const CdefHeaderContext& context,
                              std::string* error) {
  if (context.coded_lossless || !context.enable_cdef) return 0;
  if (params.damping_minus_3 < 0 || params.damping_minus_3 > 3) {
    fail(error, "cdef damping outside 3..6");
    return 0;
  }
  if (params.strengths_minus_1 < 0 || params.strengths_minus_1 > 7) {
    fail(error, "cdef strengths outside 1..8");
    return 0;
  }
  const int strengths = params.strengths_minus_1 + 1;
  if (params.num_strengths != strengths) {
    fail(error, "cdef num_strengths disagrees with strengths_minus_1");
    return 0;
  }
  const std::size_t start = writer.bit_size();
  if (!context.single_picture_header_flag) {
    writer.put_bit(params.frame_enable);
    if (!params.frame_enable) return writer.bit_size() - start;
  }
  writer.put_bits(static_cast<std::uint32_t>(params.damping_minus_3), 2U);
  writer.put_bits(static_cast<std::uint32_t>(params.strengths_minus_1), 3U);
  const int skip_mode =
      context.single_picture_header_flag ? 2 : context.cdef_on_skip_txfm;
  if (skip_mode == 2) {
    writer.put_bit(params.on_skip_txfm_frame_enable);
  } else if (skip_mode == 1) {
    if (!params.on_skip_txfm_frame_enable) {
      fail(error, "CdefOnSkipTxfm ALWAYS_ON requires on_skip_txfm_enable");
      return 0;
    }
  } else if (params.on_skip_txfm_frame_enable) {
    fail(error, "CdefOnSkipTxfm DISABLED cannot enable on_skip_txfm");
    return 0;
  }
  for (int i = 0; i < strengths; ++i) {
    const CdefStrength& s = params.set[i];
    if (s.y_pri < 0 || s.y_pri > 15 || s.uv_pri < 0 || s.uv_pri > 15 ||
        (s.y_sec != 0 && s.y_sec != 1 && s.y_sec != 2 && s.y_sec != 4) ||
        (s.uv_sec != 0 && s.uv_sec != 1 && s.uv_sec != 2 && s.uv_sec != 4)) {
      fail(error, "cdef strength value out of range");
      return 0;
    }
    writer.put_bit(s.y_pri == 0);
    if (s.y_pri != 0) {
      writer.put_bits(static_cast<std::uint32_t>(s.y_pri), 4U);
    }
    writer.put_bits(static_cast<std::uint32_t>(s.y_sec == 4 ? 3 : s.y_sec),
                    2U);
    if (context.num_planes > 1) {
      writer.put_bit(s.uv_pri == 0);
      if (s.uv_pri != 0) {
        writer.put_bits(static_cast<std::uint32_t>(s.uv_pri), 4U);
      }
      writer.put_bits(static_cast<std::uint32_t>(s.uv_sec == 4 ? 3 : s.uv_sec),
                      2U);
    }
  }
  return writer.bit_size() - start;
}

bool read_cdef_params(BitReader& reader, const CdefHeaderContext& context,
                      CdefParams* params, std::string* error) {
  if (params == nullptr) return fail(error, "null CdefParams");
  *params = CdefParams{};
  if (context.coded_lossless || !context.enable_cdef) {
    params->frame_enable = false;
    return true;
  }
  if (!context.single_picture_header_flag) {
    if (!reader.read_bit(&params->frame_enable)) {
      return fail(error, "truncated cdef_frame_enable");
    }
    if (!params->frame_enable) return true;
  } else {
    params->frame_enable = true;
  }
  std::uint32_t v = 0;
  if (!reader.read_bits(2, &v)) return fail(error, "truncated cdef damping");
  params->damping_minus_3 = static_cast<int>(v);
  if (!reader.read_bits(3, &v)) return fail(error, "truncated cdef strengths");
  params->strengths_minus_1 = static_cast<int>(v);
  params->num_strengths = params->strengths_minus_1 + 1;
  const int skip_mode =
      context.single_picture_header_flag ? 2 : context.cdef_on_skip_txfm;
  if (skip_mode == 2) {
    if (!reader.read_bit(&params->on_skip_txfm_frame_enable)) {
      return fail(error, "truncated cdef_on_skip_txfm_frame_enable");
    }
  } else {
    params->on_skip_txfm_frame_enable = skip_mode == 1;
  }
  for (int i = 0; i < params->num_strengths; ++i) {
    CdefStrength& s = params->set[i];
    bool zero = false;
    if (!reader.read_bit(&zero)) return fail(error, "truncated cdef y_pri_zero");
    if (zero) {
      s.y_pri = 0;
    } else if (!reader.read_bits(4, &v)) {
      return fail(error, "truncated cdef y_pri_strength");
    } else {
      s.y_pri = static_cast<int>(v);
    }
    if (!reader.read_bits(2, &v)) return fail(error, "truncated cdef y_sec");
    s.y_sec = (v == 3U) ? 4 : static_cast<int>(v);
    if (context.num_planes > 1) {
      if (!reader.read_bit(&zero)) {
        return fail(error, "truncated cdef uv_pri_zero");
      }
      if (zero) {
        s.uv_pri = 0;
      } else if (!reader.read_bits(4, &v)) {
        return fail(error, "truncated cdef uv_pri_strength");
      } else {
        s.uv_pri = static_cast<int>(v);
      }
      if (!reader.read_bits(2, &v)) return fail(error, "truncated cdef uv_sec");
      s.uv_sec = (v == 3U) ? 4 : static_cast<int>(v);
    }
  }
  return true;
}

// ===========================================================================
// §5.20.10.1 read_cdef()
// ===========================================================================

bool cdef_tile_init(CdefTileState* state, int mi_cols, int mi_rows,
                    int strengths, int sb_width_log2, std::string* error) {
  if (state == nullptr || mi_cols <= 0 || mi_rows <= 0) {
    return fail(error, "invalid CDEF tile geometry");
  }
  if (strengths < 1 || strengths > 8) {
    return fail(error, "CdefStrengths outside 1..8");
  }
  state->grid.reset(mi_cols, mi_rows);
  copy_array(state->index0, av2enc::cuda::still::kCdefIndex0Cdf);
  std::memset(state->minus1, 0, sizeof(state->minus1));
  if (const std::uint16_t* p =
          av2enc::cuda::still::cdef_index_minus1_default(strengths)) {
    std::memcpy(state->minus1, p,
                sizeof(std::uint16_t) * static_cast<std::size_t>(strengths));
  }
  state->strengths = strengths;
  state->sb_width_log2 = sb_width_log2;
  return true;
}

void cdef_clear_superblock(CdefTileState* state, int mi_row, int mi_col) {
  if (state == nullptr) return;
  constexpr int kMask = ~(16 - 1);
  const int r = mi_row & kMask;
  const int c = mi_col & kMask;
  if (r >= 0 && c >= 0 && r < state->grid.mi_rows && c < state->grid.mi_cols) {
    state->grid.set(r, c, static_cast<std::int8_t>(-1));
  }
}

bool cdef_write_block(av2::Encoder& encoder, CdefTileState& state,
                      int tile_mi_col_start, int tile_mi_row_start,
                      int mi_row, int mi_col, int idx, std::string* error) {
  av2enc::cuda::still::CdefIndexCdfs cdfs;
  cdfs.index0 = &state.index0[0][0];
  cdfs.minus1 = state.minus1;
  const bool ok = av2enc::cuda::still::cdef_encode_block_index(
      &state.grid, state.strengths, state.sb_width_log2, tile_mi_col_start,
      tile_mi_row_start, mi_row, mi_col, idx, cdfs,
      [&encoder, error](std::uint16_t* cdf, int n, int symbol) {
        const av2::Status status = encoder.encode_symbol(cdf, n, symbol);
        if (status != av2::Status::kOk) {
          fail(error, std::string("cdef symbol encode failed: ") +
                         av2::status_name(status));
          return false;
        }
        return true;
      });
  if (!ok) {
    return fail(error, "cdef_idx is not signallable at mi (" +
                           std::to_string(mi_row) + "," +
                           std::to_string(mi_col) + ")");
  }
  return true;
}

bool cdef_read_block(av2::Decoder& decoder, CdefTileState& state,
                     int tile_mi_col_start, int tile_mi_row_start,
                     int mi_row, int mi_col, int* idx, std::string* error) {
  av2enc::cuda::still::CdefIndexCdfs cdfs;
  cdfs.index0 = &state.index0[0][0];
  cdfs.minus1 = state.minus1;
  const bool ok = av2enc::cuda::still::cdef_decode_block_index(
      &state.grid, state.strengths, state.sb_width_log2, tile_mi_col_start,
      tile_mi_row_start, mi_row, mi_col, cdfs,
      [&decoder, error](std::uint16_t* cdf, int n) {
        int symbol = 0;
        const av2::Status status = decoder.decode_symbol(cdf, n, &symbol);
        if (status != av2::Status::kOk) {
          fail(error, std::string("cdef symbol decode failed: ") +
                         av2::status_name(status));
          return -1;
        }
        return symbol;
      },
      idx);
  if (!ok) {
    return fail(error, "cdef_idx decode failed at mi (" +
                           std::to_string(mi_row) + "," +
                           std::to_string(mi_col) + ")");
  }
  return true;
}

// ===========================================================================
// §5.18.7.12 ccso_params()
// ===========================================================================

int ccso_offset_count(const CcsoPlaneParams& params) {
  if (params.max_band_log2 < 0 || params.max_band_log2 > 6) return 0;
  const int interval = params.bo_only ? 1 : (3 - params.edge_clf);
  if (interval <= 0) return 0;
  return (1 << params.max_band_log2) * interval * interval;
}

bool ccso_fill_offsets(const CcsoPlaneParams& params,
                       restoration::CcsoParams* out, std::string* error) {
  if (out == nullptr) return fail(error, "null CcsoParams");
  if (!params.enabled) {
    for (int i = 0; i < restoration::kCcsoOffsetEntries; ++i) {
      out->offsets[i] = 0;
    }
    out->enabled = false;
    return true;
  }
  const int need = ccso_offset_count(params);
  if (need <= 0) return fail(error, "invalid CCSO offset count");
  if (static_cast<int>(params.offset_idx.size()) != need) {
    return fail(error, "CCSO offset_idx count mismatch");
  }
  for (int i = 0; i < restoration::kCcsoOffsetEntries; ++i) {
    out->offsets[i] = 0;
  }
  const int bands = 1 << params.max_band_log2;
  const int interval = params.bo_only ? 1 : (3 - params.edge_clf);
  const int scale = params.scale_idx + 1;
  std::size_t k = 0;
  for (int d0 = 0; d0 < interval; ++d0) {
    for (int d1 = 0; d1 < interval; ++d1) {
      for (int band = 0; band < bands; ++band) {
        const std::uint8_t idx = params.offset_idx[k++];
        if (idx >= restoration::kCcsoOffsetRange) {
          return fail(error, "CCSO offset index outside 0..7");
        }
        out->offsets[restoration::ccso_offset_index(band, d0, d1)] =
            static_cast<std::int16_t>(restoration::kCcsoOffset[idx] * scale);
      }
    }
  }
  out->enabled = params.enabled;
  out->bo_only = params.bo_only;
  out->scale_idx = params.scale_idx;
  out->quant_idx = params.quant_idx;
  out->ext_filter = params.ext_filter;
  out->edge_clf = params.edge_clf;
  out->max_band_log2 = params.max_band_log2;
  return true;
}

namespace {

int derive_ccso_luma_size_log2(const CcsoHeaderContext& context) {
  int sb_log2 = 0;
  int v = context.sb_size;
  while (v > 4) {
    v >>= 1;
    ++sb_log2;
  }
  if (context.ccso_unit_matches_sb_size) {
    return sb_log2 + restoration::kMiSizeLog2;
  }
  if ((context.tile_start_or & 63) == 0) return 8;
  if ((context.tile_start_or & 31) == 0) return 7;
  return 6;
}

bool validate_ccso_plane(const CcsoPlaneParams& p, std::string* error) {
  if (!p.enabled) return true;
  if (p.scale_idx < 0 || p.scale_idx > 3 || p.quant_idx < 0 ||
      p.quant_idx > 3 || p.ext_filter < 0 || p.ext_filter > 6 ||
      p.edge_clf < 0 || p.edge_clf > 2 || p.max_band_log2 < 0 ||
      p.max_band_log2 > 6) {
    return fail(error, "CCSO scalar field out of range");
  }
  const int max_band =
      p.bo_only ? restoration::kCcsoMaxBandLog2BoOnly
                : restoration::kCcsoMaxBandLog2Normal;
  if (p.max_band_log2 > max_band) {
    return fail(error, "CCSO max_band_log2 not representable");
  }
  if (!p.bo_only) {
    if (p.edge_clf > restoration::kCcsoMaxEdgeClf) {
      return fail(error, "CCSO edge_clf not representable by f(1)");
    }
    const int quant = restoration::kCcsoQuantSz[p.scale_idx][p.quant_idx];
    if (quant == 0 && p.edge_clf != 0) {
      return fail(error, "CCSO quantStep == 0 requires edge_clf == 0");
    }
  }
  const int need = ccso_offset_count(p);
  if (need <= 0 ||
      static_cast<int>(p.offset_idx.size()) != need) {
    return fail(error, "CCSO offset_idx count mismatch");
  }
  for (std::uint8_t idx : p.offset_idx) {
    if (idx >= restoration::kCcsoOffsetRange) {
      return fail(error, "CCSO offset index outside 0..7");
    }
  }
  return true;
}

}  // namespace

std::size_t write_ccso_params(BitWriter& writer, CcsoParams& params,
                              const CcsoHeaderContext& context,
                              std::string* error) {
  if (context.coded_lossless || !context.enable_ccso) {
    params.frame_flag = false;
    for (int p = 0; p < 3; ++p) params.plane[p].enabled = false;
    return 0;
  }
  params.ccso_luma_size_log2 = derive_ccso_luma_size_log2(context);
  const std::size_t start = writer.bit_size();
  if (context.single_picture_header_flag) {
    params.frame_flag = true;
  } else {
    writer.put_bit(params.frame_flag);
  }
  if (!params.frame_flag) return writer.bit_size() - start;

  for (int plane = 0; plane < context.num_planes; ++plane) {
    CcsoPlaneParams& p = params.plane[plane];
    if (!validate_ccso_plane(p, error)) return 0;
    writer.put_bit(p.enabled);
    if (!p.enabled) continue;
    if (context.frame_is_intra) {
      params.reuse_ccso[plane] = false;
      params.sb_reuse_ccso[plane] = false;
    } else {
      if (params.reuse_ccso[plane] || params.sb_reuse_ccso[plane]) {
        fail(error, "CCSO reference reuse is not supported (intra only)");
        return 0;
      }
      writer.put_bit(params.reuse_ccso[plane]);
      writer.put_bit(params.sb_reuse_ccso[plane]);
    }
    writer.put_bit(p.bo_only);
    writer.put_bits(static_cast<std::uint32_t>(p.scale_idx), 2U);
    if (p.bo_only) {
      p.quant_idx = 0;
      p.ext_filter = 0;
      p.edge_clf = 0;
    } else {
      writer.put_bits(static_cast<std::uint32_t>(p.quant_idx), 2U);
      writer.put_bits(static_cast<std::uint32_t>(p.ext_filter), 3U);
      const int quant = restoration::kCcsoQuantSz[p.scale_idx][p.quant_idx];
      if (quant == 0) {
        p.edge_clf = 0;
      } else {
        writer.put_bit(p.edge_clf != 0);
      }
    }
    const int band_bits = 2 + (p.bo_only ? 1 : 0);
    writer.put_bits(static_cast<std::uint32_t>(p.max_band_log2),
                    static_cast<unsigned>(band_bits));
    const int interval = p.bo_only ? 1 : (3 - p.edge_clf);
    const int bands = 1 << p.max_band_log2;
    std::size_t k = 0;
    for (int d0 = 0; d0 < interval; ++d0) {
      for (int d1 = 0; d1 < interval; ++d1) {
        for (int band = 0; band < bands; ++band) {
          put_tu(writer, 7U, p.offset_idx[k++]);
        }
      }
    }
  }
  return writer.bit_size() - start;
}

bool read_ccso_params(BitReader& reader, const CcsoHeaderContext& context,
                      CcsoParams* params, std::string* error) {
  if (params == nullptr) return fail(error, "null CcsoParams");
  *params = CcsoParams{};
  params->ccso_luma_size_log2 = derive_ccso_luma_size_log2(context);
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
    CcsoPlaneParams& p = params->plane[plane];
    if (!reader.read_bit(&p.enabled)) {
      return fail(error, "truncated ccso_planes");
    }
    if (!p.enabled) continue;
    if (!context.frame_is_intra) {
      if (!reader.read_bit(&params->reuse_ccso[plane]) ||
          !reader.read_bit(&params->sb_reuse_ccso[plane])) {
        return fail(error, "truncated CCSO reuse bits");
      }
      if (params->reuse_ccso[plane] || params->sb_reuse_ccso[plane]) {
        return fail(error, "CCSO reference reuse is not supported");
      }
    }
    if (!reader.read_bit(&p.bo_only)) {
      return fail(error, "truncated ccso_bo_only");
    }
    std::uint32_t v = 0;
    if (!reader.read_bits(2, &v)) return fail(error, "truncated ccso scale");
    p.scale_idx = static_cast<int>(v);
    if (p.bo_only) {
      p.quant_idx = 0;
      p.ext_filter = 0;
      p.edge_clf = 0;
    } else {
      if (!reader.read_bits(2, &v)) {
        return fail(error, "truncated ccso quant");
      }
      p.quant_idx = static_cast<int>(v);
      if (!reader.read_bits(3, &v)) {
        return fail(error, "truncated ccso ext_filter");
      }
      p.ext_filter = static_cast<int>(v);
      if (p.ext_filter > 6) {
        return fail(error, "CCSO ext_filter must be 0..6");
      }
      const int quant = restoration::kCcsoQuantSz[p.scale_idx][p.quant_idx];
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
    const int max_band =
        p.bo_only ? restoration::kCcsoMaxBandLog2BoOnly
                  : restoration::kCcsoMaxBandLog2Normal;
    if (p.max_band_log2 > max_band) {
      return fail(error, "CCSO max_band_log2 not representable");
    }
    const int interval = p.bo_only ? 1 : (3 - p.edge_clf);
    const int bands = 1 << p.max_band_log2;
    p.offset_idx.clear();
    p.offset_idx.reserve(static_cast<std::size_t>(interval * interval * bands));
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
    if (!validate_ccso_plane(p, error)) return false;
  }
  return true;
}

// ===========================================================================
// §5.20.10.2 read_ccso()
// ===========================================================================

bool ccso_tile_init(CcsoTileState* state, const CcsoParams& params,
                    int mi_cols, int mi_rows, int sub_x, int sub_y,
                    std::string* error) {
  if (state == nullptr || mi_cols <= 0 || mi_rows <= 0) {
    return fail(error, "invalid CCSO tile geometry");
  }
  if (params.ccso_luma_size_log2 < 6 || params.ccso_luma_size_log2 > 8) {
    return fail(error, "CcsoLumaSizeLog2 outside 6..8");
  }
  state->params = params;
  state->mi_cols = mi_cols;
  state->mi_rows = mi_rows;
  state->tile_mi_col_start = 0;
  state->tile_mi_row_start = 0;
  copy_array(state->cdf, kCcsoBlkCdf);
  for (int plane = 0; plane < 3; ++plane) {
    const CcsoPlaneParams& p = params.plane[plane];
    state->blks[plane].clear();
    state->blk_cols[plane] = 0;
    state->blk_rows[plane] = 0;
    if (!p.enabled) continue;
    const int width = (mi_cols * restoration::kMiSize) >> (plane == 0 ? 0 : sub_x);
    const int height =
        (mi_rows * restoration::kMiSize) >> (plane == 0 ? 0 : sub_y);
    restoration::ccso_blk_grid_size(width, height, params.ccso_luma_size_log2,
                                    plane == 0 ? 0 : sub_x,
                                    plane == 0 ? 0 : sub_y,
                                    &state->blk_cols[plane],
                                    &state->blk_rows[plane]);
    state->blks[plane].assign(
        static_cast<std::size_t>(state->blk_cols[plane]) *
            static_cast<std::size_t>(state->blk_rows[plane]),
        0);
  }
  return true;
}

namespace {

bool ccso_block_origin(const CcsoTileState& state, int mi_row, int mi_col,
                       int* shift_row, int* shift_col, int* blk_h4,
                       int* blk_w4) {
  const int sr = state.ccso_luma_size_log2 - restoration::kMiSizeLog2;
  const int sc = sr;
  if (sr < 0 || sc < 0) return false;
  *shift_row = sr;
  *shift_col = sc;
  *blk_h4 = 1 << sr;
  *blk_w4 = 1 << sc;
  return (mi_row & (*blk_h4 - 1)) == 0 && (mi_col & (*blk_w4 - 1)) == 0;
}

int ccso_block_context(const CcsoTileState& state, int plane, int mi_row,
                       int mi_col, int shift_row, int shift_col, int blk_w4) {
  if (mi_col - blk_w4 < state.tile_mi_col_start) return 0;
  const int row = mi_row >> shift_row;
  const int col = (mi_col - blk_w4) >> shift_col;
  if (row < 0 || row >= state.blk_rows[plane] || col < 0 ||
      col >= state.blk_cols[plane]) {
    return 0;
  }
  const int value =
      state.blks[plane][static_cast<std::size_t>(row) *
                            static_cast<std::size_t>(state.blk_cols[plane]) +
                        static_cast<std::size_t>(col)];
  return 2 * value;
}

}  // namespace

bool ccso_write_block(av2::Encoder& encoder, CcsoTileState& state, int mi_row,
                      int mi_col, const int blk[3], std::string* error) {
  if (!state.params.frame_flag) return true;
  int shift_row = 0;
  int shift_col = 0;
  int blk_h4 = 0;
  int blk_w4 = 0;
  if (!ccso_block_origin(state, mi_row, mi_col, &shift_row, &shift_col,
                         &blk_h4, &blk_w4)) {
    return true;
  }
  for (int plane = 0; plane < 3; ++plane) {
    const CcsoPlaneParams& p = state.params.plane[plane];
    if (!p.enabled || state.params.sb_reuse_ccso[plane]) continue;
    const int ctx =
        ccso_block_context(state, plane, mi_row, mi_col, shift_row, shift_col,
                           blk_w4);
    const int symbol = blk != nullptr && blk[plane] != 0 ? 1 : 0;
    const av2::Status status =
        encoder.encode_symbol(&state.cdf[plane][ctx][0], 2, symbol);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("ccso_blk encode failed: ") +
                             av2::status_name(status));
    }
    const int row = mi_row >> shift_row;
    const int col = mi_col >> shift_col;
    if (row >= 0 && row < state.blk_rows[plane] && col >= 0 &&
        col < state.blk_cols[plane]) {
      state.blks[plane][static_cast<std::size_t>(row) *
                            static_cast<std::size_t>(state.blk_cols[plane]) +
                        static_cast<std::size_t>(col)] =
          static_cast<std::uint8_t>(symbol);
    }
  }
  return true;
}

bool ccso_read_block(av2::Decoder& decoder, CcsoTileState& state, int mi_row,
                     int mi_col, int blk[3], std::string* error) {
  for (int plane = 0; plane < 3; ++plane) {
    if (blk != nullptr) blk[plane] = 0;
  }
  if (!state.params.frame_flag) return true;
  int shift_row = 0;
  int shift_col = 0;
  int blk_h4 = 0;
  int blk_w4 = 0;
  if (!ccso_block_origin(state, mi_row, mi_col, &shift_row, &shift_col,
                         &blk_h4, &blk_w4)) {
    return true;
  }
  for (int plane = 0; plane < 3; ++plane) {
    const CcsoPlaneParams& p = state.params.plane[plane];
    if (!p.enabled || state.params.sb_reuse_ccso[plane]) continue;
    const int ctx =
        ccso_block_context(state, plane, mi_row, mi_col, shift_row, shift_col,
                           blk_w4);
    int symbol = 0;
    const av2::Status status =
        decoder.decode_symbol(&state.cdf[plane][ctx][0], 2, &symbol);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("ccso_blk decode failed: ") +
                             av2::status_name(status));
    }
    const int row = mi_row >> shift_row;
    const int col = mi_col >> shift_col;
    if (row >= 0 && row < state.blk_rows[plane] && col >= 0 &&
        col < state.blk_cols[plane]) {
      state.blks[plane][static_cast<std::size_t>(row) *
                            static_cast<std::size_t>(state.blk_cols[plane]) +
                        static_cast<std::size_t>(col)] =
          static_cast<std::uint8_t>(symbol);
    }
    if (blk != nullptr) blk[plane] = symbol;
  }
  return true;
}

// ===========================================================================
// §5.18.7.11 lr_params()
// ===========================================================================

int lr_count_units(int unit_size, int frame_size) {
  if (unit_size <= 0) return 1;
  const int n = (frame_size + (unit_size >> 1)) / unit_size;
  return n > 1 ? n : 1;
}

namespace {

int lr_unit_size_luma(const LrParams& params) {
  return restoration::kRestorationTileSizeMax >> params.luma_shift;
}

int lr_unit_size_chroma(const LrParams& params, int sub_x, int sub_y) {
  const int sub = std::max(sub_x, sub_y);
  return (restoration::kRestorationTileSizeMax >> sub) >> params.chroma_shift;
}

// Writes one lr_luma_use_* / lr_chroma_use_* flag group for a 64x64 SB.
bool write_lr_size_flags(BitWriter& writer, int shift, int sb_size,
                         bool luma, std::string* error) {
  if (shift < 0 || shift > 3) {
    return fail(error, "LoopRestorationSize shift outside 0..3");
  }
  if (sb_size == 256) {
    if (shift != 0) return fail(error, "256x256 SB only supports shift 0");
    return true;
  }
  writer.put_bit(shift == 1);
  if (shift == 1) return true;
  writer.put_bit(shift == 0);
  if (shift == 0) return true;
  if (sb_size == 128) {
    if (shift != 2) {
      return fail(error, "128x128 SB only supports shift 0,1,2");
    }
    return true;
  }
  // 64x64 SB: shift is 2 (quarter) or 3 (eighth).
  if (shift != 2 && shift != 3) {
    return fail(error, "64x64 SB supports shift 0,1,2,3");
  }
  writer.put_bit(shift == 2);
  (void)luma;
  return true;
}

bool read_lr_size_flags(BitReader& reader, int sb_size, int* shift,
                        std::string* error) {
  bool half = false;
  if (!reader.read_bit(&half)) return fail(error, "truncated lr size flag");
  if (half) {
    *shift = 1;
    return true;
  }
  if (sb_size == 256) {
    *shift = 0;
    return true;
  }
  bool max_size = false;
  if (!reader.read_bit(&max_size)) {
    return fail(error, "truncated lr max size flag");
  }
  if (max_size) {
    *shift = 0;
    return true;
  }
  if (sb_size == 128) {
    *shift = 2;
    return true;
  }
  bool quarter = false;
  if (!reader.read_bit(&quarter)) {
    return fail(error, "truncated lr quarter size flag");
  }
  *shift = quarter ? 2 : 3;
  return true;
}

// §5.20.10.6 read_wienerns_filter(..., readFrameFilters = 1).  Raw bits only:
// the class/group/match logic and coefficient subexp are f()/ns()/L(), not
// arithmetic-coded symbols.
bool write_frame_filters(BitWriter& writer, LrParams& params, int plane,
                         std::string* error) {
  const int num_classes =
      (plane == 0) ? params.num_filter_classes : 1;
  const int num_subsets = (plane == 0) ? 4 : 3;
  const int n_coeffs = (plane == 0) ? restoration::kWienerNsLumaCoeffs
                                    : restoration::kWienerNsChromaCoeffs;
  const bool chroma = plane > 0;

  int group_counts[3] = {num_classes, 0, 0};
  int group_hits[3] = {0, 0, 0};
  int group_base[3] = {0, 0, 0};
  for (int i = 1; i < 3; ++i) {
    group_base[i] = group_base[i - 1] + group_counts[i - 1];
  }

  for (int c = 0; c < num_classes; ++c) {
    group_counts[0] = c + 1;
    int pred_group = 0;
    if (c == 0) {
      pred_group = group_counts[1] > 2 ? 1 : predict_group(group_counts);
    } else {
      pred_group = predict_group(group_hits);
    }
    int num_zeros = 0;
    int alt_group = 0;
    for (int i = 0; i < 3; ++i) {
      if (i != pred_group) {
        if (group_counts[i] == 0) {
          ++num_zeros;
        } else {
          alt_group = i;
        }
      }
    }
    // M1 is intra: numRefFilters == 0 and group 2 is empty.  With
    // numClasses >= 1 the predicted group is always 0 and numZeros == 2, so
    // nothing is coded for use_alt_group.
    if (num_zeros != 2 || pred_group != 0 || alt_group != 0) {
      return fail(error,
                  "unsupported Wiener frame-filter group layout (intra only)");
    }
    const int group = 0;
    const int n = group_counts[group];
    const int ref = group_base[group] + (n >> 1);
    if (n == 1) {
      params.frame_filter[plane][c].match_index = group_base[group];
    } else {
      const int match = params.frame_filter[plane][c].match_index;
      if (match < group_base[group] || match >= group_base[group] + n) {
        return fail(error, "frame filter match_index outside its group");
      }
      if (!encode_signed_subexp_with_ref(writer, group_base[group],
                                         group_base[group] + n, ref, 4,
                                         match, error)) {
        return false;
      }
    }
    ++group_hits[group];
  }

  // Per-class merged flag, reference bank (always 0 for frame filters), then
  // coefficients relative to the matched filter.
  std::int16_t ref[16][restoration::kWienerNsCoeffStride] = {};
  for (int c = 0; c < num_classes; ++c) {
    WienerNsFrameFilter& f = params.frame_filter[plane][c];
    if (f.match_index < 0 || f.match_index >= num_classes) {
      return fail(error, "frame filter match_index out of range");
    }
    if (c == 0 && f.match_index != 0) {
      return fail(error, "first frame-filter class must match 0");
    }
    if (f.match_index > c) {
      return fail(error, "frame filter match_index refers to a later class");
    }
    writer.put_bit(f.merged);
    // fill_first_slot_of_bank_with_filter_match(): match 0 is all zero,
    // match m > 0 copies parameter class m - 1.
    for (int j = 0; j < n_coeffs; ++j) {
      ref[c][j] = (f.match_index == 0)
                      ? 0
                      : params.frame_filter[plane][f.match_index - 1].coeffs[j];
    }
    const int bank_ref = 0;
    if (f.merged) {
      for (int j = 0; j < n_coeffs; ++j) {
        f.coeffs[j] = ref[c][j];
      }
      continue;
    }
    if (f.subset < 0 || f.subset >= num_subsets) {
      return fail(error, "frame filter subset out of range");
    }
    if (plane > 0 && f.subset > 0) {
      // validate symmetry before writing anything
      for (int j = restoration::kWienerNsShortCoeffs; j + 1 < n_coeffs;
           j += 2) {
        if (f.coeffs[j] != f.coeffs[j + 1]) {
          return fail(error, "wiener_ns_uv_sym requires symmetric pairs");
        }
      }
    } else {
      f.uv_sym = false;
    }
    for (int s = 0; s < num_subsets - 1; ++s) {
      const bool len = s < f.subset;
      writer.put_bit(len);
      if (!len) break;
    }
    if (plane > 0 && f.subset > 0) {
      writer.put_bit(f.uv_sym);
    }
    for (int j = 0; j < n_coeffs; ++j) {
      const int min = restoration::kWienerNsTapsMin[chroma ? 1 : 0][j];
      const int k = restoration::kWienerNsTapsK[chroma ? 1 : 0][j];
      std::int16_t v = ref[c][j];
      if (kTapsPresent[chroma ? 1 : 0][f.subset][j] != 0) {
        const int target = f.coeffs[j];
        if (target < min || target >= min + (1 << k)) {
          return fail(error, "frame Wiener coefficient outside its tap range");
        }
        if (!encode_signed_subexp_with_ref(writer, min, min + (1 << k),
                                           ref[c][j], k - 3, target, error)) {
          return false;
        }
        v = static_cast<std::int16_t>(target);
      } else {
        v = 0;
      }
      f.coeffs[j] = v;
      if (plane > 0 && f.subset > 0 && f.uv_sym &&
          j >= restoration::kWienerNsShortCoeffs) {
        f.coeffs[j + 1] = v;
        ++j;
      }
    }
    // Luma carries 16 coefficients in an 18-entry array; the tail is not
    // signalled and must not leak into the §7.20 filter.
    for (int j = n_coeffs; j < restoration::kWienerNsCoeffStride; ++j) {
      f.coeffs[j] = 0;
    }
    (void)bank_ref;
  }
  return true;
}

bool read_frame_filters(BitReader& reader, LrParams* params, int plane,
                        std::string* error) {
  const int num_classes =
      (plane == 0) ? params->num_filter_classes : 1;
  const int num_subsets = (plane == 0) ? 4 : 3;
  const int n_coeffs = (plane == 0) ? restoration::kWienerNsLumaCoeffs
                                    : restoration::kWienerNsChromaCoeffs;
  const bool chroma = plane > 0;

  int group_counts[3] = {num_classes, 0, 0};
  int group_hits[3] = {0, 0, 0};
  int group_base[3] = {0, 0, 0};
  for (int i = 1; i < 3; ++i) {
    group_base[i] = group_base[i - 1] + group_counts[i - 1];
  }

  for (int c = 0; c < num_classes; ++c) {
    group_counts[0] = c + 1;
    int pred_group = 0;
    if (c == 0) {
      pred_group = group_counts[1] > 2 ? 1 : predict_group(group_counts);
    } else {
      pred_group = predict_group(group_hits);
    }
    int num_zeros = 0;
    int alt_group = 0;
    for (int i = 0; i < 3; ++i) {
      if (i != pred_group) {
        if (group_counts[i] == 0) {
          ++num_zeros;
        } else {
          alt_group = i;
        }
      }
    }
    bool use_alt_group = false;
    if (num_zeros == 2) {
      use_alt_group = false;
    } else if (!reader.read_bit(&use_alt_group)) {
      return fail(error, "truncated use_alt_group");
    }
    int group = pred_group;
    if (use_alt_group) {
      if (num_zeros == 1) {
        group = alt_group;
      } else {
        bool group_bit = false;
        if (!reader.read_bit(&group_bit)) {
          return fail(error, "truncated group_bit");
        }
        const int gb = group_bit ? 1 : 0;
        group = pred_group <= gb ? gb + 1 : gb;
      }
    }
    if (group < 0 || group > 2) {
      return fail(error, "frame-filter group outside 0..2");
    }
    const int n = group_counts[group];
    const int ref = group_base[group] + (n >> 1);
    if (n <= 0) return fail(error, "empty frame-filter group");
    if (n == 1) {
      params->frame_filter[plane][c].match_index = group_base[group];
    } else {
      int match = 0;
      if (!decode_signed_subexp_with_ref(reader, group_base[group],
                                         group_base[group] + n, ref, 4,
                                         &match, error)) {
        return false;
      }
      params->frame_filter[plane][c].match_index = match;
    }
    ++group_hits[group];
  }

  std::int16_t ref[16][restoration::kWienerNsCoeffStride] = {};
  for (int c = 0; c < num_classes; ++c) {
    WienerNsFrameFilter& f = params->frame_filter[plane][c];
    if (f.match_index < 0 || f.match_index > c) {
      return fail(error, "frame filter match_index refers to a later class");
    }
    if (!reader.read_bit(&f.merged)) {
      return fail(error, "truncated merged_param");
    }
    for (int j = 0; j < n_coeffs; ++j) {
      ref[c][j] = (f.match_index == 0)
                      ? 0
                      : params->frame_filter[plane][f.match_index - 1].coeffs[j];
    }
    if (f.merged) {
      for (int j = 0; j < n_coeffs; ++j) f.coeffs[j] = ref[c][j];
      continue;
    }
    f.subset = 0;
    for (int s = 0; s < num_subsets - 1; ++s) {
      bool len = false;
      if (!reader.read_bit(&len)) {
        return fail(error, "truncated wiener_ns_length");
      }
      if (!len) break;
      ++f.subset;
    }
    f.uv_sym = false;
    if (plane > 0 && f.subset > 0) {
      if (!reader.read_bit(&f.uv_sym)) {
        return fail(error, "truncated wiener_ns_uv_sym");
      }
    }
    for (int j = 0; j < n_coeffs; ++j) {
      const int min = restoration::kWienerNsTapsMin[chroma ? 1 : 0][j];
      const int k = restoration::kWienerNsTapsK[chroma ? 1 : 0][j];
      std::int16_t v = ref[c][j];
      if (kTapsPresent[chroma ? 1 : 0][f.subset][j] != 0) {
        int target = 0;
        if (!decode_signed_subexp_with_ref(reader, min, min + (1 << k),
                                           ref[c][j], k - 3, &target,
                                           error)) {
          return false;
        }
        v = static_cast<std::int16_t>(target);
      } else {
        v = 0;
      }
      f.coeffs[j] = v;
      if (plane > 0 && f.subset > 0 && f.uv_sym &&
          j >= restoration::kWienerNsShortCoeffs) {
        f.coeffs[j + 1] = v;
        ++j;
      }
    }
  }
  return true;
}

}  // namespace

std::size_t write_lr_params(BitWriter& writer, LrParams& params,
                            const LrHeaderContext& context,
                            std::string* error) {
  if (context.coded_lossless || !context.enable_restoration) {
    for (int p = 0; p < 3; ++p) {
      params.frame_type[p] = kRestoreNone;
      params.frame_filters_on[p] = false;
    }
    return 0;
  }
  const std::size_t start = writer.bit_size();
  if (params.num_filter_classes != 1 &&
      !valid_num_filter_classes(params.num_filter_classes)) {
    fail(error, "NumFilterClasses is not in Decode_Num_Filter_Classes");
    return 0;
  }
  params.num_filter_classes_idx =
      num_filter_classes_index(params.num_filter_classes);

  bool uses_luma_lr = false;
  bool uses_chroma_lr = false;
  for (int plane = 0; plane < context.num_planes; ++plane) {
    const bool chroma = plane > 0;
    int index_to_tool[4] = {kRestoreNone, 0, 0, 0};
    int tools_count = 1;
    for (int i = 1; i < kRestoreSwitchableTypes; ++i) {
      if (!context.lr_tools_disable[chroma ? 1 : 0][i]) {
        index_to_tool[tools_count++] = i;
      }
    }
    index_to_tool[tools_count] = kRestoreSwitchable;
    const bool allow_switchable = tools_count > 2;
    const int n = tools_count + (allow_switchable ? 1 : 0);
    const int target = params.frame_type[plane];
    int tool_index = -1;
    for (int i = 0; i < n; ++i) {
      if (index_to_tool[i] == target) {
        tool_index = i;
        break;
      }
    }
    if (tool_index < 0) {
      fail(error, "FrameRestorationType is not signallable for plane " +
                      std::to_string(plane));
      return 0;
    }
    put_ns(writer, static_cast<unsigned>(n),
           static_cast<std::uint32_t>(tool_index));
    if (target != kRestoreNone) {
      if (plane == 0) {
        uses_luma_lr = true;
      } else {
        uses_chroma_lr = true;
      }
    }
    int num_classes = 1;
    if (target == kRestoreWienerNs || target == kRestoreSwitchable) {
      writer.put_bit(params.frame_filters_on[plane]);
      if (params.frame_filters_on[plane]) {
        if (!context.frame_is_intra) {
          fail(error, "temporal frame-filter prediction is not supported");
          return 0;
        }
        if (plane == 0) {
          if (params.num_filter_classes_idx < 0 ||
              params.num_filter_classes_idx > 7) {
            fail(error, "num_filter_classes_idx outside f(3)");
            return 0;
          }
          writer.put_bits(
              static_cast<std::uint32_t>(params.num_filter_classes_idx), 3U);
          num_classes = params.num_filter_classes;
        }
      } else {
        num_classes = 1;
      }
    } else {
      params.frame_filters_on[plane] = false;
    }
    if (plane > 0 && params.frame_filters_on[plane]) num_classes = 1;
    (void)num_classes;
  }

  if (uses_luma_lr) {
    if (!write_lr_size_flags(writer, params.luma_shift, context.sb_size, true,
                             error)) {
      return 0;
    }
  }
  if (uses_chroma_lr) {
    if (!write_lr_size_flags(writer, params.chroma_shift, context.sb_size,
                             false, error)) {
      return 0;
    }
  }

  for (int plane = 0; plane < context.num_planes; ++plane) {
    if (!params.frame_filters_on[plane]) continue;
    if (!write_frame_filters(writer, params, plane, error)) return 0;
  }
  return writer.bit_size() - start;
}

bool read_lr_params(BitReader& reader, const LrHeaderContext& context,
                    LrParams* params, std::string* error) {
  if (params == nullptr) return fail(error, "null LrParams");
  *params = LrParams{};
  if (context.coded_lossless || !context.enable_restoration) {
    for (int p = 0; p < 3; ++p) {
      params->frame_type[p] = kRestoreNone;
      params->frame_filters_on[p] = false;
    }
    return true;
  }
  bool uses_luma_lr = false;
  bool uses_chroma_lr = false;
  params->num_filter_classes = 1;
  for (int plane = 0; plane < context.num_planes; ++plane) {
    const bool chroma = plane > 0;
    int index_to_tool[4] = {kRestoreNone, 0, 0, 0};
    int tools_count = 1;
    for (int i = 1; i < kRestoreSwitchableTypes; ++i) {
      if (!context.lr_tools_disable[chroma ? 1 : 0][i]) {
        index_to_tool[tools_count++] = i;
      }
    }
    index_to_tool[tools_count] = kRestoreSwitchable;
    const bool allow_switchable = tools_count > 2;
    const int n = tools_count + (allow_switchable ? 1 : 0);
    std::uint32_t tool_index = 0;
    if (!reader.read_ns(static_cast<unsigned>(n), &tool_index) ||
        static_cast<int>(tool_index) >= n) {
      return fail(error, "truncated FrameRestorationType tool_index");
    }
    params->frame_type[plane] = index_to_tool[tool_index];
    if (params->frame_type[plane] != kRestoreNone) {
      if (plane == 0) {
        uses_luma_lr = true;
      } else {
        uses_chroma_lr = true;
      }
    }
    const int type = params->frame_type[plane];
    if (type == kRestoreWienerNs || type == kRestoreSwitchable) {
      if (!reader.read_bit(&params->frame_filters_on[plane])) {
        return fail(error, "truncated frame_filters_on");
      }
      if (params->frame_filters_on[plane]) {
        if (!context.frame_is_intra) {
          return fail(error, "temporal frame-filter prediction is not supported");
        }
        if (plane == 0) {
          std::uint32_t idx = 0;
          if (!reader.read_bits(3, &idx)) {
            return fail(error, "truncated num_filter_classes_idx");
          }
          params->num_filter_classes_idx = static_cast<int>(idx);
          params->num_filter_classes = decoder_num_filter_classes(
              params->num_filter_classes_idx);
          if (params->num_filter_classes == 0) {
            return fail(error, "invalid num_filter_classes_idx");
          }
        }
      }
    } else {
      params->frame_filters_on[plane] = false;
    }
  }
  if (uses_luma_lr) {
    if (!read_lr_size_flags(reader, context.sb_size, &params->luma_shift,
                            error)) {
      return false;
    }
  }
  if (uses_chroma_lr) {
    if (!read_lr_size_flags(reader, context.sb_size, &params->chroma_shift,
                            error)) {
      return false;
    }
  }
  for (int plane = 0; plane < context.num_planes; ++plane) {
    if (!params->frame_filters_on[plane]) continue;
    if (!read_frame_filters(reader, params, plane, error)) return false;
  }
  return true;
}

// ===========================================================================
// Loop-restoration tile state
// ===========================================================================

bool lr_state_init(LrState* state, const LrParams& params, int mi_cols,
                   int mi_rows, int sub_x, int sub_y, int sb_size,
                   std::string* error) {
  if (state == nullptr || mi_cols <= 0 || mi_rows <= 0) {
    return fail(error, "invalid LR tile geometry");
  }
  for (int plane = 0; plane < 3; ++plane) {
    const int type = params.frame_type[plane];
    if (type != kRestoreNone && type != kRestoreWienerNs &&
        type != kRestoreSwitchable) {
      return fail(error, "unsupported FrameRestorationType (PC-Wiener/GDF)");
    }
    if (params.frame_filters_on[plane] && type != kRestoreWienerNs &&
        type != kRestoreSwitchable) {
      return fail(error, "frame_filters_on requires Wiener/switchable LR");
    }
  }
  state->params = params;
  state->mi_cols = mi_cols;
  state->mi_rows = mi_rows;
  state->sub_x = sub_x;
  state->sub_y = sub_y;
  state->sb_size = sb_size;

  copy_array(state->use_wiener_ns, kUseWienerNsCdf);
  copy_array(state->wiener_ns_length, kWienerNsLengthCdf);
  copy_array(state->wiener_ns_uv_sym, kWienerNsUvSymCdf);
  copy_array(state->wiener_ns_base, kWienerNsBaseCdf);
  std::memcpy(state->flex_restoration_type, kFlexRestorationTypeCdf,
              sizeof(state->flex_restoration_type));

  int luma_size = lr_unit_size_luma(params);
  int chroma_size = lr_unit_size_chroma(params, sub_x, sub_y);
  state->unit_size[0] = luma_size;
  state->unit_size[1] = chroma_size;
  state->unit_size[2] = chroma_size;
  const int min_luma = 64;
  const int min_chroma = 64 >> sub_y;
  if (luma_size < min_luma || chroma_size < min_chroma ||
      (luma_size & (luma_size - 1)) != 0 ||
      (chroma_size & (chroma_size - 1)) != 0) {
    return fail(error, "LoopRestorationSize violates §6.17.7.7");
  }

  for (int plane = 0; plane < 3; ++plane) {
    const int sx = (plane == 0) ? 0 : sub_x;
    const int sy = (plane == 0) ? 0 : sub_y;
    state->unit_cols[plane] =
        lr_count_units(state->unit_size[plane], (mi_cols * 4) >> sx);
    state->unit_rows[plane] =
        lr_count_units(state->unit_size[plane], (mi_rows * 4) >> sy);
    const std::size_t units =
        static_cast<std::size_t>(state->unit_cols[plane]) *
        static_cast<std::size_t>(state->unit_rows[plane]);
    state->unit_type[plane].assign(units, kRestoreNone);
    state->unit_coeffs[plane].assign(
        units * restoration::kWienerNsCoeffStride, 0);
  }

  // decode_tile(): RefLrWienerNs[plane][c][0][i] = min + ((1 << k) >> 1),
  // WienerNsPtr = 0, WienerNsBankSize = 0.
  for (int plane = 0; plane < 3; ++plane) {
    for (int c = 0; c < restoration::kWienerNsClasses; ++c) {
      state->bank_ptr[plane][c] = 0;
      state->bank_size[plane][c] = 0;
      for (int bank = 0; bank < 4; ++bank) {
        for (int j = 0; j < restoration::kWienerNsCoeffStride; ++j) {
          const int min =
              restoration::kWienerNsTapsMin[plane == 0 ? 0 : 1][j];
          const int k =
              restoration::kWienerNsTapsK[plane == 0 ? 0 : 1][j];
          state->ref_bank[plane][c][bank][j] =
              static_cast<std::int16_t>(min + ((1 << k) >> 1));
        }
      }
    }
  }

  for (int plane = 0; plane < 3; ++plane) {
    for (int c = 0; c < restoration::kWienerNsClasses; ++c) {
      for (int j = 0; j < restoration::kWienerNsCoeffStride; ++j) {
        state->frame_coeffs[plane][c][j] = 0;
      }
    }
    if (params.frame_filters_on[plane]) {
      const int n = (plane == 0) ? params.num_filter_classes : 1;
      for (int c = 0; c < n; ++c) {
        for (int j = 0; j < restoration::kWienerNsCoeffStride; ++j) {
          state->frame_coeffs[plane][c][j] =
              params.frame_filter[plane][c].coeffs[j];
        }
      }
    }
  }

  if (params.frame_filters_on[0] && params.num_filter_classes > 1) {
    const WienerNsClassMap& map = params.class_map;
    if (map.num_classes != params.num_filter_classes) {
      return fail(error,
                  "class map num_classes does not match num_filter_classes");
    }
    if (!map.has_subclass_lookup && map.subclass_grid.empty()) {
      return fail(error,
                  "frame filters with >1 class need a Pc_Wiener_Sub_Classify2 "
                  "subclass lookup (not in the spec text)");
    }
    if (map.has_subclass_lookup && !map.filter_class_grid.empty() &&
        static_cast<int>(map.filter_class_grid.size()) != mi_cols * mi_rows) {
      return fail(error, "filter_class_grid must be mi_cols x mi_rows");
    }
  }
  return true;
}

// ===========================================================================
// §5.20.10.4 read_lr() / §5.20.10.5 / §5.20.10.6 tile syntax
// ===========================================================================

bool lr_unit_range(const LrState& state, int plane, int mi_row, int mi_col,
                   int w4, int h4, int tile_mi_row_start,
                   int tile_mi_col_start, LrUnitRange* range,
                   std::string* error) {
  if (range == nullptr || plane < 0 || plane > 2) {
    return fail(error, "invalid read_lr argument");
  }
  const int type = state.params.frame_type[plane];
  if (type == kRestoreNone) {
    *range = LrUnitRange{};
    return true;
  }
  const int sub_x = (plane == 0) ? 0 : state.sub_x;
  const int sub_y = (plane == 0) ? 0 : state.sub_y;
  const int unit_size = state.unit_size[plane];
  if (unit_size <= 0) return fail(error, "invalid LoopRestorationSize");
  const int mi_cols = state.mi_cols - tile_mi_col_start;
  const int mi_rows = state.mi_rows - tile_mi_row_start;
  const int c = mi_col - tile_mi_col_start;
  const int r = mi_row - tile_mi_row_start;
  if (c < 0 || r < 0) return fail(error, "read_lr outside the tile");
  const int unit_rows =
      lr_count_units(unit_size, (mi_rows * 4) >> sub_y);
  const int unit_cols =
      lr_count_units(unit_size, (mi_cols * 4) >> sub_x);
  const int lr_row_offset = (tile_mi_row_start * 4 >> sub_y) / unit_size;
  const int lr_col_offset = (tile_mi_col_start * 4 >> sub_x) / unit_size;
  const int step_y = 4 >> sub_y;
  const int step_x = 4 >> sub_x;
  if (step_y <= 0 || step_x <= 0) {
    return fail(error, "invalid subsampling in read_lr");
  }
  const int unit_row_start =
      (r * step_y + unit_size - 1) / unit_size;
  const int unit_row_end =
      std::min(unit_rows,
               ((r + h4) * step_y + unit_size - 1) / unit_size);
  const int unit_col_start =
      (c * step_x + unit_size - 1) / unit_size;
  const int unit_col_end =
      std::min(unit_cols,
               ((c + w4) * step_x + unit_size - 1) / unit_size);
  range->row_begin = unit_row_start + lr_row_offset;
  range->row_end = unit_row_end + lr_row_offset;
  range->col_begin = unit_col_start + lr_col_offset;
  range->col_end = unit_col_end + lr_col_offset;
  return true;
}

int lr_block_unit_count(const LrState& state, int mi_row, int mi_col, int w4,
                        int h4, int tile_mi_row_start,
                        int tile_mi_col_start) {
  int count = 0;
  for (int plane = 0; plane < 3; ++plane) {
    LrUnitRange range;
    if (!lr_unit_range(state, plane, mi_row, mi_col, w4, h4,
                       tile_mi_row_start, tile_mi_col_start, &range, nullptr)) {
      continue;
    }
    count += (range.row_end - range.row_begin) *
             (range.col_end - range.col_begin);
  }
  return count;
}

bool lr_build_decisions(const LrState& state, int mi_row, int mi_col, int w4,
                        int h4, int tile_mi_row_start,
                        int tile_mi_col_start,
                        std::vector<LrUnitDecision>* decisions,
                        std::string* error) {
  if (decisions == nullptr) return fail(error, "null LR decision list");
  const int count = lr_block_unit_count(state, mi_row, mi_col, w4, h4,
                                        tile_mi_row_start, tile_mi_col_start);
  decisions->clear();
  decisions->reserve(static_cast<std::size_t>(count));
  for (int plane = 0; plane < 3; ++plane) {
    LrUnitRange range;
    if (!lr_unit_range(state, plane, mi_row, mi_col, w4, h4,
                       tile_mi_row_start, tile_mi_col_start, &range, error)) {
      return false;
    }
    for (int r = range.row_begin; r < range.row_end; ++r) {
      for (int c = range.col_begin; c < range.col_end; ++c) {
        const std::size_t index =
            static_cast<std::size_t>(r) *
                static_cast<std::size_t>(state.unit_cols[plane]) +
            static_cast<std::size_t>(c);
        LrUnitDecision decision;
        decision.type = state.unit_type[plane][index];
        if (decision.type != kRestoreWienerNs ||
            state.params.frame_filters_on[plane]) {
          decisions->push_back(decision);
          continue;
        }
        const bool chroma = plane > 0;
        const int n_coeffs = chroma ? restoration::kWienerNsChromaCoeffs
                                    : restoration::kWienerNsLumaCoeffs;
        const int num_subsets = chroma ? 3 : 4;
        const std::int16_t* coeffs =
            &state.unit_coeffs[plane]
             [index * restoration::kWienerNsCoeffStride];
        int subset = -1;
        for (int s = 0; s < num_subsets && subset < 0; ++s) {
          bool fits = true;
          for (int j = 0; j < n_coeffs; ++j) {
            if (coeffs[j] != 0 &&
                kTapsPresent[chroma ? 1 : 0][s][j] == 0) {
              fits = false;
              break;
            }
          }
          if (fits) subset = s;
        }
        if (subset < 0) {
          return fail(error,
                      "LR unit coefficients do not fit any Wiener subset");
        }
        decision.subset = subset;
        for (int j = 0; j < restoration::kWienerNsCoeffStride; ++j) {
          decision.coeffs[j] = coeffs[j];
        }
        if (chroma && subset > 0) {
          bool symmetric = true;
          for (int j = restoration::kWienerNsShortCoeffs; j + 1 < n_coeffs;
               j += 2) {
            if (decision.coeffs[j] != decision.coeffs[j + 1]) {
              symmetric = false;
              break;
            }
          }
          decision.uv_sym = symmetric;
        }
        decisions->push_back(decision);
      }
    }
  }
  return static_cast<int>(decisions->size()) == count;
}

namespace {

bool write_unit_filter(av2::Encoder& encoder, LrState& state, int plane,
                       int unit_row, int unit_col,
                       const LrUnitDecision& decision, std::string* error) {
  const bool chroma = plane > 0;
  const int n_coeffs = chroma ? restoration::kWienerNsChromaCoeffs
                              : restoration::kWienerNsLumaCoeffs;
  const int num_subsets = chroma ? 3 : 4;
  const int cls = 0;
  const int old_size = state.bank_size[plane][cls];
  const int old_ptr = state.bank_ptr[plane][cls];

  if (!raw_bit(encoder, decision.merged, error)) return false;

  int k_sel = decision.use_bank_k;
  if (k_sel < 0) {
    k_sel = (old_size <= 1) ? 0 : old_size - 1;
  }
  if (k_sel < 0 || k_sel >= std::max(1, old_size)) {
    return fail(error, "use_bank_k outside the current bank");
  }
  for (int k = 0; k < old_size - 1; ++k) {
    const bool bit = k == k_sel;
    if (!raw_bit(encoder, bit, error)) return false;
    if (bit) break;
  }
  const int ref_slot = (old_ptr - k_sel + 4) & 3;

  if (decision.merged) {
    if (state.bank_size[plane][cls] == 0) {
      state.bank_size[plane][cls] = 1;
    }
    const std::size_t base =
        (static_cast<std::size_t>(unit_row) *
             static_cast<std::size_t>(state.unit_cols[plane]) +
         static_cast<std::size_t>(unit_col)) *
        restoration::kWienerNsCoeffStride;
    for (int j = 0; j < n_coeffs; ++j) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] =
          state.ref_bank[plane][cls][ref_slot][j];
    }
    return true;
  }

  int new_ptr = old_ptr;
  int new_size = old_size;
  if (old_size < 4) {
    new_ptr = old_size;
    new_size = old_size + 1;
  } else {
    new_ptr = (old_ptr + 1) & 3;
  }

  if (decision.subset < 0 || decision.subset >= num_subsets) {
    return fail(error, "per-unit Wiener subset out of range");
  }
  for (int s = 0; s < num_subsets - 1; ++s) {
    const bool len = s < decision.subset;
    const av2::Status status = encoder.encode_symbol(
        &state.wiener_ns_length[std::min(plane, 1)][0], 2, len ? 1 : 0);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("wiener_ns_length encode failed: ") +
                             av2::status_name(status));
    }
    if (!len) break;
  }
  bool uv_sym = false;
  if (plane > 0 && decision.subset > 0) {
    uv_sym = decision.uv_sym;
    const av2::Status status = encoder.encode_symbol(
        &state.wiener_ns_uv_sym[0], 2, uv_sym ? 1 : 0);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("wiener_ns_uv_sym encode failed: ") +
                             av2::status_name(status));
    }
    for (int j = restoration::kWienerNsShortCoeffs; j + 1 < n_coeffs;
         j += 2) {
      if (decision.coeffs[j] != decision.coeffs[j + 1]) {
        return fail(error, "wiener_ns_uv_sym requires symmetric pairs");
      }
    }
  }

  state.bank_ptr[plane][cls] = new_ptr;
  state.bank_size[plane][cls] = new_size;

  const std::size_t base =
      (static_cast<std::size_t>(unit_row) *
           static_cast<std::size_t>(state.unit_cols[plane]) +
       static_cast<std::size_t>(unit_col)) *
      restoration::kWienerNsCoeffStride;
  for (int j = 0; j < n_coeffs; ++j) {
    const int min = restoration::kWienerNsTapsMin[chroma ? 1 : 0][j];
    const int k = restoration::kWienerNsTapsK[chroma ? 1 : 0][j];
    std::int16_t v = state.ref_bank[plane][cls][ref_slot][j];
    if (kTapsPresent[chroma ? 1 : 0][decision.subset][j] != 0) {
      const int target = decision.coeffs[j];
      if (target < min || target >= min + (1 << k)) {
        return fail(error, "per-unit Wiener coefficient outside its tap range");
      }
      if (!encode_signed_4part(state.wiener_ns_base, min, k,
                               state.ref_bank[plane][cls][ref_slot][j],
                               target, encoder, error)) {
        return false;
      }
      v = static_cast<std::int16_t>(target);
    } else {
      v = 0;
    }
    state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = v;
    state.ref_bank[plane][cls][new_ptr][j] = v;
    if (plane > 0 && decision.subset > 0 && uv_sym &&
        j >= restoration::kWienerNsShortCoeffs) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j) + 1] = v;
      state.ref_bank[plane][cls][new_ptr][j + 1] = v;
      ++j;
    }
  }
  return true;
}

bool read_unit_filter(av2::Decoder& decoder, LrState& state, int plane,
                      int unit_row, int unit_col, LrUnitDecision* decision,
                      std::string* error) {
  const bool chroma = plane > 0;
  const int n_coeffs = chroma ? restoration::kWienerNsChromaCoeffs
                              : restoration::kWienerNsLumaCoeffs;
  const int num_subsets = chroma ? 3 : 4;
  const int cls = 0;
  const int old_size = state.bank_size[plane][cls];
  const int old_ptr = state.bank_ptr[plane][cls];

  bool merged = false;
  if (!raw_bit(decoder, &merged, error)) return false;
  if (decision != nullptr) decision->merged = merged;

  int k_sel = 0;
  for (int k = 0; k < old_size - 1; ++k) {
    bool bit = false;
    if (!raw_bit(decoder, &bit, error)) return false;
    if (bit) {
      k_sel = k;
      break;
    }
    k_sel = k + 1;
  }
  const int ref_slot = (old_ptr - k_sel + 4) & 3;

  if (merged) {
    if (state.bank_size[plane][cls] == 0) {
      state.bank_size[plane][cls] = 1;
    }
    const std::size_t base =
        (static_cast<std::size_t>(unit_row) *
             static_cast<std::size_t>(state.unit_cols[plane]) +
         static_cast<std::size_t>(unit_col)) *
        restoration::kWienerNsCoeffStride;
    for (int j = 0; j < n_coeffs; ++j) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] =
          state.ref_bank[plane][cls][ref_slot][j];
      if (decision != nullptr) {
        decision->coeffs[j] = state.ref_bank[plane][cls][ref_slot][j];
      }
    }
    return true;
  }

  int new_ptr = old_ptr;
  int new_size = old_size;
  if (old_size < 4) {
    new_ptr = old_size;
    new_size = old_size + 1;
  } else {
    new_ptr = (old_ptr + 1) & 3;
  }

  int subset = 0;
  for (int s = 0; s < num_subsets - 1; ++s) {
    int len = 0;
    const av2::Status status = decoder.decode_symbol(
        &state.wiener_ns_length[std::min(plane, 1)][0], 2, &len);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("wiener_ns_length decode failed: ") +
                             av2::status_name(status));
    }
    if (len == 0) break;
    ++subset;
  }
  bool uv_sym = false;
  if (plane > 0 && subset > 0) {
    int sym = 0;
    const av2::Status status =
        decoder.decode_symbol(&state.wiener_ns_uv_sym[0], 2, &sym);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("wiener_ns_uv_sym decode failed: ") +
                             av2::status_name(status));
    }
    uv_sym = sym != 0;
  }

  state.bank_ptr[plane][cls] = new_ptr;
  state.bank_size[plane][cls] = new_size;

  const std::size_t base =
      (static_cast<std::size_t>(unit_row) *
           static_cast<std::size_t>(state.unit_cols[plane]) +
       static_cast<std::size_t>(unit_col)) *
      restoration::kWienerNsCoeffStride;
  if (decision != nullptr) {
    decision->subset = subset;
    decision->uv_sym = uv_sym;
  }
  for (int j = 0; j < n_coeffs; ++j) {
    const int min = restoration::kWienerNsTapsMin[chroma ? 1 : 0][j];
    const int k = restoration::kWienerNsTapsK[chroma ? 1 : 0][j];
    std::int16_t v = state.ref_bank[plane][cls][ref_slot][j];
    if (kTapsPresent[chroma ? 1 : 0][subset][j] != 0) {
      int target = 0;
      if (!decode_signed_4part(state.wiener_ns_base, min, k,
                               state.ref_bank[plane][cls][ref_slot][j],
                               decoder, &target, error)) {
        return false;
      }
      v = static_cast<std::int16_t>(target);
    } else {
      v = 0;
    }
    state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = v;
    state.ref_bank[plane][cls][new_ptr][j] = v;
    if (decision != nullptr) decision->coeffs[j] = v;
    if (plane > 0 && subset > 0 && uv_sym &&
        j >= restoration::kWienerNsShortCoeffs) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j) + 1] = v;
      state.ref_bank[plane][cls][new_ptr][j + 1] = v;
      if (decision != nullptr) decision->coeffs[j + 1] = v;
      ++j;
    }
  }
  return true;
}

bool write_unit_filter_frame_mode(LrState& state, int plane, int unit_row,
                                  int unit_col, std::string* error) {
  // frame_filters_on[plane]: read_wienerns_filter returns immediately, so the
  // per-unit coefficients are the frame filter table.
  const std::size_t base =
      (static_cast<std::size_t>(unit_row) *
           static_cast<std::size_t>(state.unit_cols[plane]) +
       static_cast<std::size_t>(unit_col)) *
      restoration::kWienerNsCoeffStride;
  const int n = (plane == 0) ? restoration::kWienerNsLumaCoeffs
                             : restoration::kWienerNsChromaCoeffs;
  for (int j = 0; j < n; ++j) {
    state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = state.frame_coeffs[plane][0][j];
  }
  (void)error;
  return true;
}

}  // namespace

bool lr_write_unit(av2::Encoder& encoder, LrState& state, int plane,
                   int unit_row, int unit_col,
                   const LrUnitDecision& decision, std::string* error) {
  if (plane < 0 || plane > 2) return fail(error, "LR plane out of range");
  if (unit_row < 0 || unit_col < 0 ||
      unit_row >= state.unit_rows[plane] ||
      unit_col >= state.unit_cols[plane]) {
    return fail(error, "LR unit index out of range");
  }
  const int frame_type = state.params.frame_type[plane];
  int type = kRestoreNone;
  if (frame_type == kRestoreWienerNs) {
    const bool on = decision.type == kRestoreWienerNs;
    if (!on && decision.type != kRestoreNone) {
      return fail(error, "WIENER_NONSEP unit must be on or off");
    }
    const av2::Status status =
        encoder.encode_symbol(&state.use_wiener_ns[0], 2, on ? 1 : 0);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("use_wiener_ns encode failed: ") +
                             av2::status_name(status));
    }
    type = on ? kRestoreWienerNs : kRestoreNone;
  } else if (frame_type == kRestoreSwitchable) {
    // read_lr_unit(): flex tool 0 is the restoration type value 0
    // (RESTORE_NONE), tool 1 is 1 (RESTORE_PC_WIENER), and no flex bit means
    // the default 2 (RESTORE_WIENER_NONSEP).
    if (decision.type != kRestoreNone &&
        decision.type != kRestorePcWiener &&
        decision.type != kRestoreWienerNs) {
      return fail(error, "invalid switchable LR unit type");
    }
    type = kRestoreWienerNs;
    for (int tool = 0; tool < kRestoreSwitchableTypes - 1; ++tool) {
      const bool on = decision.type == tool;
      const av2::Status status = encoder.encode_symbol(
          &state.flex_restoration_type[tool][plane][0], 2, on ? 1 : 0);
      if (status != av2::Status::kOk) {
        return fail(error, std::string("flex_restoration_type encode failed: ") +
                               av2::status_name(status));
      }
      if (on) {
        type = tool;
        break;
      }
    }
  } else if (frame_type == kRestoreNone) {
    if (decision.type != kRestoreNone) {
      return fail(error, "LR unit written for a RESTORE_NONE plane");
    }
    type = kRestoreNone;
  } else {
    return fail(error, "PC-Wiener frame type is not supported");
  }

  const std::size_t index =
      static_cast<std::size_t>(unit_row) *
          static_cast<std::size_t>(state.unit_cols[plane]) +
      static_cast<std::size_t>(unit_col);
  state.unit_type[plane][index] = static_cast<std::uint8_t>(type);
  if (type != kRestoreWienerNs) return true;
  if (state.params.frame_filters_on[plane]) {
    return write_unit_filter_frame_mode(state, plane, unit_row, unit_col,
                                        error);
  }
  return write_unit_filter(encoder, state, plane, unit_row, unit_col, decision,
                           error);
}

bool lr_read_unit(av2::Decoder& decoder, LrState& state, int plane,
                  int unit_row, int unit_col, LrUnitDecision* decision,
                  std::string* error) {
  if (plane < 0 || plane > 2) return fail(error, "LR plane out of range");
  if (unit_row < 0 || unit_col < 0 ||
      unit_row >= state.unit_rows[plane] ||
      unit_col >= state.unit_cols[plane]) {
    return fail(error, "LR unit index out of range");
  }
  LrUnitDecision local;
  if (decision == nullptr) decision = &local;
  *decision = LrUnitDecision{};
  const int frame_type = state.params.frame_type[plane];
  int type = kRestoreNone;
  if (frame_type == kRestoreWienerNs) {
    int on = 0;
    const av2::Status status =
        decoder.decode_symbol(&state.use_wiener_ns[0], 2, &on);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("use_wiener_ns decode failed: ") +
                             av2::status_name(status));
    }
    type = on != 0 ? kRestoreWienerNs : kRestoreNone;
  } else if (frame_type == kRestoreSwitchable) {
    type = kRestoreWienerNs;
    for (int tool = 0; tool < kRestoreSwitchableTypes - 1; ++tool) {
      int on = 0;
      const av2::Status status = decoder.decode_symbol(
          &state.flex_restoration_type[tool][plane][0], 2, &on);
      if (status != av2::Status::kOk) {
        return fail(error, std::string("flex_restoration_type decode failed: ") +
                               av2::status_name(status));
      }
      if (on != 0) {
        type = tool;
        break;
      }
    }
  } else if (frame_type == kRestoreNone) {
    type = kRestoreNone;
  } else {
    return fail(error, "PC-Wiener frame type is not supported");
  }
  decision->type = type;
  const std::size_t index =
      static_cast<std::size_t>(unit_row) *
          static_cast<std::size_t>(state.unit_cols[plane]) +
      static_cast<std::size_t>(unit_col);
  state.unit_type[plane][index] = static_cast<std::uint8_t>(type);
  if (type != kRestoreWienerNs) return true;
  if (state.params.frame_filters_on[plane]) {
    const std::size_t base = index * restoration::kWienerNsCoeffStride;
    const int n = (plane == 0) ? restoration::kWienerNsLumaCoeffs
                               : restoration::kWienerNsChromaCoeffs;
    for (int j = 0; j < n; ++j) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] =
          state.frame_coeffs[plane][0][j];
      decision->coeffs[j] = state.frame_coeffs[plane][0][j];
    }
    return true;
  }
  return read_unit_filter(decoder, state, plane, unit_row, unit_col, decision,
                          error);
}

bool lr_write_block(av2::Encoder& encoder, LrState& state, int mi_row,
                    int mi_col, int w4, int h4, int tile_mi_row_start,
                    int tile_mi_col_start,
                    const std::vector<LrUnitDecision>& decisions,
                    std::string* error) {
  const int count = lr_block_unit_count(state, mi_row, mi_col, w4, h4,
                                        tile_mi_row_start, tile_mi_col_start);
  if (static_cast<int>(decisions.size()) != count) {
    return fail(error, "read_lr decision count mismatch: expected " +
                           std::to_string(count) + ", got " +
                           std::to_string(decisions.size()));
  }
  int cursor = 0;
  for (int plane = 0; plane < 3; ++plane) {
    LrUnitRange range;
    if (!lr_unit_range(state, plane, mi_row, mi_col, w4, h4,
                       tile_mi_row_start, tile_mi_col_start, &range, error)) {
      return false;
    }
    for (int r = range.row_begin; r < range.row_end; ++r) {
      for (int c = range.col_begin; c < range.col_end; ++c) {
        if (!lr_write_unit(encoder, state, plane, r, c, decisions[static_cast<std::size_t>(cursor)],
                           error)) {
          return false;
        }
        ++cursor;
      }
    }
  }
  return true;
}

bool lr_read_block(av2::Decoder& decoder, LrState& state, int mi_row,
                   int mi_col, int w4, int h4, int tile_mi_row_start,
                   int tile_mi_col_start,
                   std::vector<LrUnitDecision>* decisions,
                   std::string* error) {
  const int count = lr_block_unit_count(state, mi_row, mi_col, w4, h4,
                                        tile_mi_row_start, tile_mi_col_start);
  if (decisions != nullptr) {
    decisions->clear();
    decisions->reserve(static_cast<std::size_t>(count));
  }
  for (int plane = 0; plane < 3; ++plane) {
    LrUnitRange range;
    if (!lr_unit_range(state, plane, mi_row, mi_col, w4, h4,
                       tile_mi_row_start, tile_mi_col_start, &range, error)) {
      return false;
    }
    for (int r = range.row_begin; r < range.row_end; ++r) {
      for (int c = range.col_begin; c < range.col_end; ++c) {
        LrUnitDecision decision;
        if (!lr_read_unit(decoder, state, plane, r, c, &decision, error)) {
          return false;
        }
        if (decisions != nullptr) decisions->push_back(decision);
      }
    }
  }
  return true;
}

}  // namespace av2enc::still::filters_stream
