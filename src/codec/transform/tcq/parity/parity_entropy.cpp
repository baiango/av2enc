#include "parity_entropy.h"

#include "codec/entropy/qctx/qctx.h"

#include <algorithm>
#include <cstddef>
#include <string>

namespace av2enc::tcq::parity {
namespace {

std::uint16_t* base_cdf_ptr(CdfStore& cdfs, const PositionSyntax& p,
                            const Geometry& geometry) {
  const std::int32_t tsc = tx_sz_ctx(geometry.tx_size);
  switch (p.base_kind) {
    case CdfKind::kCoeffBaseEob:
      return &cdfs.coeff_base_eob[tsc][p.base_ctx][0];
    case CdfKind::kCoeffBaseLfEob:
      return &cdfs.coeff_base_lf_eob[tsc][p.base_ctx][0];
    case CdfKind::kCoeffBaseEobUv:
      return &cdfs.coeff_base_eob_uv[p.base_ctx][0];
    case CdfKind::kCoeffBaseLfEobUv:
      return &cdfs.coeff_base_lf_eob_uv[p.base_ctx][0];
    case CdfKind::kCoeffBase:
      return &cdfs.coeff_base[tsc][p.base_ctx][p.base_ctx2][0];
    case CdfKind::kCoeffBaseLf:
      return &cdfs.coeff_base_lf[tsc][p.base_ctx][p.base_ctx2][0];
    case CdfKind::kCoeffBaseUv:
      return &cdfs.coeff_base_uv[p.base_ctx][0];
    case CdfKind::kCoeffBaseLfUv:
      return &cdfs.coeff_base_lf_uv[p.base_ctx][0];
    case CdfKind::kCoeffBasePh:
      return &cdfs.coeff_base_ph[p.base_ctx][0];
    default:
      return nullptr;
  }
}

std::uint16_t* br_cdf_ptr(CdfStore& cdfs, const PositionSyntax& p) {
  switch (p.br_kind) {
    case CdfKind::kCoeffBr:
      return &cdfs.coeff_br[p.br_ctx][0];
    case CdfKind::kCoeffBrLf:
      return &cdfs.coeff_br_lf[p.br_ctx][0];
    case CdfKind::kCoeffBrUv:
      return &cdfs.coeff_br_uv[p.br_ctx][0];
    default:
      return nullptr;
  }
}

std::uint16_t* eob_cdf_ptr(CdfStore& cdfs, CdfKind kind, std::int32_t ctx) {
  switch (kind) {
    case CdfKind::kEobPt16: return &cdfs.eob_pt_16[ctx][0];
    case CdfKind::kEobPt32: return &cdfs.eob_pt_32[ctx][0];
    case CdfKind::kEobPt64: return &cdfs.eob_pt_64[ctx][0];
    case CdfKind::kEobPt128: return &cdfs.eob_pt_128[ctx][0];
    case CdfKind::kEobPt256: return &cdfs.eob_pt_256[ctx][0];
    case CdfKind::kEobPt512: return &cdfs.eob_pt_512[ctx][0];
    case CdfKind::kEobPt1024: return &cdfs.eob_pt_1024[ctx][0];
    default: return nullptr;
  }
}

std::uint16_t* txb_skip_ptr(CdfStore& cdfs, const Geometry& geometry,
                            const EntropyConfig& config) {
  if (geometry.plane == 2) {
    return &cdfs.v_txb_skip[config.txb_skip_ctx][0];
  }
  return &cdfs.txb_skip[config.eligibility.is_inter ? 1 : 0]
                        [tx_sz_ctx(geometry.tx_size)][config.txb_skip_ctx][0];
}

bool write_literal(av2::Encoder& encoder, std::int32_t bits,
                   std::uint32_t value, std::string* error) {
  for (std::int32_t i = bits - 1; i >= 0; --i) {
    const bool bit = ((value >> static_cast<unsigned>(i)) & 1U) != 0U;
    const av2::Status status = encoder.encode_bool(bit);
    if (status != av2::Status::kOk) {
      if (error != nullptr) *error = av2::status_name(status);
      return false;
    }
  }
  return true;
}

bool read_literal(av2::Decoder& decoder, std::int32_t bits,
                  std::uint32_t* value, std::string* error) {
  std::uint32_t out = 0;
  for (std::int32_t i = 0; i < bits; ++i) {
    bool bit = false;
    const av2::Status status = decoder.read_bool(&bit);
    if (status != av2::Status::kOk) {
      if (error != nullptr) *error = av2::status_name(status);
      return false;
    }
    out = (out << 1U) | (bit ? 1U : 0U);
  }
  *value = out;
  return true;
}

}  // namespace

CdfStore::CdfStore(std::int32_t qctx) {
  const std::int32_t slice =
      qctx <= 0 ? 0
                : (qctx >= av2enc::still::qctx::kSliceCount ? av2enc::still::qctx::kSliceCount - 1
                                                 : qctx);
  const av2enc::still::qctx::CoeffCdfSlice& q = av2enc::still::qctx::tables_for_slice(slice);
  av2enc::still::qctx::copy_cdf(txb_skip, *q.txb_skip);
  av2enc::still::qctx::copy_cdf(v_txb_skip, *q.v_txb_skip);
  av2enc::still::qctx::copy_cdf(eob_pt_16, *q.eob_pt_16);
  av2enc::still::qctx::copy_cdf(eob_pt_32, *q.eob_pt_32);
  av2enc::still::qctx::copy_cdf(eob_pt_64, *q.eob_pt_64);
  av2enc::still::qctx::copy_cdf(eob_pt_128, *q.eob_pt_128);
  av2enc::still::qctx::copy_cdf(eob_pt_256, *q.eob_pt_256);
  av2enc::still::qctx::copy_cdf(eob_pt_512, *q.eob_pt_512);
  av2enc::still::qctx::copy_cdf(eob_pt_1024, *q.eob_pt_1024);
  av2enc::still::qctx::copy_cdf(eob_extra, *q.eob_extra);
  av2enc::still::qctx::copy_cdf(coeff_base_eob, *q.coeff_base_eob);
  av2enc::still::qctx::copy_cdf(coeff_base_lf_eob, *q.coeff_base_lf_eob);
  av2enc::still::qctx::copy_cdf(coeff_base_eob_uv, *q.coeff_base_eob_uv);
  av2enc::still::qctx::copy_cdf(coeff_base_lf_eob_uv, *q.coeff_base_lf_eob_uv);
  av2enc::still::qctx::copy_cdf(coeff_base, *q.coeff_base);
  av2enc::still::qctx::copy_cdf(coeff_base_lf, *q.coeff_base_lf);
  av2enc::still::qctx::copy_cdf(coeff_base_uv, *q.coeff_base_uv);
  av2enc::still::qctx::copy_cdf(coeff_base_lf_uv, *q.coeff_base_lf_uv);
  av2enc::still::qctx::copy_cdf(coeff_base_ph, *q.coeff_base_ph);
  av2enc::still::qctx::copy_cdf(coeff_br, *q.coeff_br);
  av2enc::still::qctx::copy_cdf(coeff_br_lf, *q.coeff_br_lf);
  av2enc::still::qctx::copy_cdf(coeff_br_uv, *q.coeff_br_uv);
  av2enc::still::qctx::copy_cdf(dc_sign, *q.dc_sign);
}

bool entropy_config_valid(const EntropyConfig& config) noexcept {
  if (config.qctx < 0 || config.qctx >= av2enc::still::qctx::kSliceCount) return false;
  if (config.txb_skip_ctx < 0 || config.txb_skip_ctx > 11) return false;
  if (config.dc_sign_ctx < 0 || config.dc_sign_ctx > 2) return false;
  return true;
}

// ---------------------------------------------------------------------------
// read_quant escape (§5.20.7.28)
// ---------------------------------------------------------------------------

bool write_escape_bits(av2::Encoder& encoder, std::int32_t x,
                       std::int32_t predictor, std::string* error) {
  if (x < 0) {
    if (error != nullptr) *error = "negative escape value";
    return false;
  }
  const std::int32_t m =
      clip3(1, 6, predictor <= 0 ? 0 : get_msb(
                                  static_cast<std::uint32_t>(predictor)));
  const std::int32_t k = m + 1;
  const std::int32_t c_max = std::min(m + 4, 6);
  if (x < (c_max << m)) {
    const std::int32_t q = x >> m;
    for (std::int32_t i = 0; i < q; ++i) {
      const av2::Status status = encoder.encode_bool(false);
      if (status != av2::Status::kOk) {
        if (error != nullptr) *error = av2::status_name(status);
        return false;
      }
    }
    const av2::Status status = encoder.encode_bool(true);
    if (status != av2::Status::kOk) {
      if (error != nullptr) *error = av2::status_name(status);
      return false;
    }
    return write_literal(encoder, m, static_cast<std::uint32_t>(
                                        x & ((1 << m) - 1)), error);
  }
  for (std::int32_t i = 0; i < c_max; ++i) {
    const av2::Status status = encoder.encode_bool(false);
    if (status != av2::Status::kOk) {
      if (error != nullptr) *error = av2::status_name(status);
      return false;
    }
  }
  std::int32_t g = 0;
  for (;;) {
    const std::int32_t length = k + g;
    const std::int32_t x_base = (c_max << m) + (1 << length) - (1 << k);
    if (x < x_base + (1 << length)) {
      for (std::int32_t i = 0; i < g; ++i) {
        const av2::Status status = encoder.encode_bool(false);
        if (status != av2::Status::kOk) {
          if (error != nullptr) *error = av2::status_name(status);
          return false;
        }
      }
      const av2::Status status = encoder.encode_bool(true);
      if (status != av2::Status::kOk) {
        if (error != nullptr) *error = av2::status_name(status);
        return false;
      }
      return write_literal(encoder, length,
                           static_cast<std::uint32_t>(x - x_base), error);
    }
    ++g;
    if (g > 30) {
      if (error != nullptr) *error = "escape value is not codable";
      return false;
    }
  }
}

bool read_escape_bits(av2::Decoder& decoder, std::int32_t predictor,
                      std::int32_t* x, std::string* error) {
  const std::int32_t m =
      clip3(1, 6, predictor <= 0 ? 0 : get_msb(
                                  static_cast<std::uint32_t>(predictor)));
  const std::int32_t k = m + 1;
  const std::int32_t c_max = std::min(m + 4, 6);
  std::int32_t q = 0;
  for (; q < c_max; ++q) {
    bool bit = false;
    const av2::Status status = decoder.read_bool(&bit);
    if (status != av2::Status::kOk) {
      if (error != nullptr) *error = av2::status_name(status);
      return false;
    }
    if (bit) break;
  }
  if (q == c_max) {
    std::int32_t length = -1;
    for (;;) {
      ++length;
      bool bit = false;
      const av2::Status status = decoder.read_bool(&bit);
      if (status != av2::Status::kOk) {
        if (error != nullptr) *error = av2::status_name(status);
        return false;
      }
      if (bit) break;
      if (length > 20) {
        if (error != nullptr) *error = "golomb length exceeds 20";
        return false;
      }
    }
    length += k;
    const std::int32_t x_base =
        (q << m) + (1 << length) - (1 << k);
    std::uint32_t rem = 0;
    if (!read_literal(decoder, length, &rem, error)) return false;
    *x = x_base + static_cast<std::int32_t>(rem);
    return true;
  }
  std::uint32_t rem = 0;
  if (!read_literal(decoder, m, &rem, error)) return false;
  *x = (q << m) + static_cast<std::int32_t>(rem);
  return true;
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

bool write_block(av2::Encoder& encoder, CdfStore& cdfs,
                 const BlockSyntax& syntax, const Geometry& geometry,
                 const EntropyConfig& config, std::string* error) {
  if (!entropy_config_valid(config)) {
    if (error != nullptr) *error = "invalid entropy config";
    return false;
  }
  if (config.eligibility.tx_type_is_idtx || config.eligibility.use_fsc) {
    if (error != nullptr) {
      *error = "IDTX/FSC coefficient syntax is outside this component";
    }
    return false;
  }
  if (geometry.plane != 2 && config.txb_skip_ctx > 9) {
    if (error != nullptr) *error = "luma/U txb_skip context must be 0..9";
    return false;
  }
  if (syntax.parity_hiding_eligible !=
      parity_hiding_eligible(geometry, config.eligibility)) {
    if (error != nullptr) {
      *error = "syntax eligibility does not match the entropy config";
    }
    return false;
  }

  std::uint16_t* skip_cdf = txb_skip_ptr(cdfs, geometry, config);
  av2::Status status =
      encoder.encode_symbol(skip_cdf, 2, syntax.eob == 0 ? 1 : 0);
  if (status != av2::Status::kOk) {
    if (error != nullptr) *error = av2::status_name(status);
    return false;
  }
  if (syntax.eob == 0) return true;

  std::uint16_t* eob_cdf =
      eob_cdf_ptr(cdfs, syntax.eob_kind, syntax.eob_ctx);
  if (eob_cdf == nullptr) {
    if (error != nullptr) *error = "unknown EOB CDF";
    return false;
  }
  status = encoder.encode_symbol(eob_cdf, syntax.eob_symbols,
                                 syntax.eob_symbol);
  if (status != av2::Status::kOk) {
    if (error != nullptr) *error = av2::status_name(status);
    return false;
  }
  if (syntax.eob_escape_bits > 0) {
    if (!write_literal(encoder, syntax.eob_escape_bits,
                       static_cast<std::uint32_t>(syntax.eob_escape_value),
                       error)) {
      return false;
    }
  }
  if (syntax.eob_pt >= 3) {
    status = encoder.encode_symbol(&cdfs.eob_extra[0], 2,
                                   syntax.eob_extra_symbol);
    if (status != av2::Status::kOk) {
      if (error != nullptr) *error = av2::status_name(status);
      return false;
    }
    if (!write_literal(encoder, syntax.eob_extra_lsb_bits,
                       static_cast<std::uint32_t>(
                           syntax.eob_extra_lsb_value),
                       error)) {
      return false;
    }
  }

  // First pass: base + br magnitudes, descending scan order.
  for (std::int32_t c = syntax.eob - 1; c >= 0; --c) {
    const PositionSyntax& p = syntax.positions[static_cast<std::size_t>(c)];
    std::uint16_t* base_cdf = base_cdf_ptr(cdfs, p, geometry);
    if (base_cdf == nullptr) {
      if (error != nullptr) *error = "unknown coeff_base CDF";
      return false;
    }
    status = encoder.encode_symbol(base_cdf, p.base_symbols, p.base_symbol);
    if (status != av2::Status::kOk) {
      if (error != nullptr) *error = av2::status_name(status);
      return false;
    }
    if (p.br_present) {
      std::uint16_t* br_cdf = br_cdf_ptr(cdfs, p);
      if (br_cdf == nullptr) {
        if (error != nullptr) *error = "unknown coeff_br CDF";
        return false;
      }
      status = encoder.encode_symbol(br_cdf, p.br_symbols, p.br_symbol);
      if (status != av2::Status::kOk) {
        if (error != nullptr) *error = av2::status_name(status);
        return false;
      }
    }
  }

  // Second pass: signs and read_quant escapes, descending scan order.
  std::int32_t hr_level_avg = 0;
  for (std::int32_t c = syntax.eob - 1; c >= 0; --c) {
    const PositionSyntax& p = syntax.positions[static_cast<std::size_t>(c)];
    if (p.sign_present) {
      if (p.dc_style_sign) {
        const std::int32_t sign_ctx =
            (p.pos == 0) ? config.dc_sign_ctx : 0;
        std::uint16_t* cdf =
            &cdfs.dc_sign[0][syntax.is_hidden ? 1 : 0][sign_ctx][0];
        status = encoder.encode_symbol(cdf, 2, p.sign);
        if (status != av2::Status::kOk) {
          if (error != nullptr) *error = av2::status_name(status);
          return false;
        }
      } else {
        status = encoder.encode_bool(p.sign != 0);
        if (status != av2::Status::kOk) {
          if (error != nullptr) *error = av2::status_name(status);
          return false;
        }
      }
    }
    if (p.escape_present) {
      const std::int32_t lvl_shift =
          (syntax.is_hidden && c == 0) ? 1 : 0;
      const std::int32_t predictor = hr_level_avg >> lvl_shift;
      if (predictor != p.hr_predictor) {
        if (error != nullptr) *error = "escape predictor mismatch";
        return false;
      }
      if (!write_escape_bits(encoder, p.escape_x, predictor, error)) {
        return false;
      }
      hr_level_avg =
          update_hr_level_avg(p.escape_x, hr_level_avg, lvl_shift);
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

bool read_block(av2::Decoder& decoder, CdfStore& cdfs,
                const Geometry& geometry, const EntropyConfig& config,
                BlockSyntax* syntax, DecodedBlock* decoded,
                std::string* error) {
  if (syntax == nullptr || decoded == nullptr) {
    if (error != nullptr) *error = "null output";
    return false;
  }
  if (!entropy_config_valid(config)) {
    if (error != nullptr) *error = "invalid entropy config";
    return false;
  }
  if (config.eligibility.tx_type_is_idtx || config.eligibility.use_fsc) {
    if (error != nullptr) {
      *error = "IDTX/FSC coefficient syntax is outside this component";
    }
    return false;
  }
  if (geometry.plane != 2 && config.txb_skip_ctx > 9) {
    if (error != nullptr) *error = "luma/U txb_skip context must be 0..9";
    return false;
  }
  *syntax = BlockSyntax{};
  syntax->parity_hiding_eligible =
      parity_hiding_eligible(geometry, config.eligibility);

  std::uint16_t* skip_cdf = txb_skip_ptr(cdfs, geometry, config);
  std::int32_t skip_symbol = 0;
  av2::Status status = decoder.decode_symbol(skip_cdf, 2, &skip_symbol);
  if (status != av2::Status::kOk) {
    if (error != nullptr) *error = av2::status_name(status);
    return false;
  }
  if (skip_symbol != 0) {
    syntax->eob = 0;
    return decode_syntax(*syntax, geometry, decoded, error);
  }

  const EobCdfDecision eob_decision =
      eob_cdf_decision(geometry, config.eligibility.is_inter);
  std::uint16_t* eob_cdf =
      eob_cdf_ptr(cdfs, eob_decision.kind, eob_decision.ctx);
  if (eob_cdf == nullptr) {
    if (error != nullptr) *error = "unknown EOB CDF";
    return false;
  }
  std::int32_t eob_symbol = 0;
  status = decoder.decode_symbol(eob_cdf, eob_decision.symbols, &eob_symbol);
  if (status != av2::Status::kOk) {
    if (error != nullptr) *error = av2::status_name(status);
    return false;
  }
  std::int32_t eob_pt = eob_symbol + 1;
  std::int32_t eob_escape = 0;
  if (eob_symbol == eob_decision.symbols - 1 &&
      eob_decision.escape_bits > 0) {
    std::uint32_t value = 0;
    if (!read_literal(decoder, eob_decision.escape_bits, &value, error)) {
      return false;
    }
    eob_escape = static_cast<std::int32_t>(value);
    eob_pt += eob_escape;
  }
  std::int32_t eob = eob_from_pt(eob_pt);
  std::int32_t eob_extra_symbol = 0;
  std::int32_t eob_lsb_bits = 0;
  std::int32_t eob_lsb_value = 0;
  if (eob_pt >= 3) {
    status = decoder.decode_symbol(&cdfs.eob_extra[0], 2, &eob_extra_symbol);
    if (status != av2::Status::kOk) {
      if (error != nullptr) *error = av2::status_name(status);
      return false;
    }
    eob_lsb_bits = eob_pt - 3;
    if (eob_lsb_bits > 0) {
      std::uint32_t value = 0;
      if (!read_literal(decoder, eob_lsb_bits, &value, error)) return false;
      eob_lsb_value = static_cast<std::int32_t>(value);
    }
    eob += (eob_extra_symbol << (eob_pt - 3)) + eob_lsb_value;
  }

  const std::size_t area = static_cast<std::size_t>(geometry.w) *
                           static_cast<std::size_t>(geometry.h);
  if (eob <= 0 || static_cast<std::size_t>(eob) > area) {
    if (error != nullptr) *error = "decoded EOB is outside the block";
    return false;
  }

  syntax->eob = eob;
  syntax->eob_kind = eob_decision.kind;
  syntax->eob_ctx = eob_decision.ctx;
  syntax->eob_symbols = eob_decision.symbols;
  syntax->eob_symbol = eob_symbol;
  syntax->eob_pt = eob_pt;
  syntax->eob_escape_bits =
      (eob_symbol == eob_decision.symbols - 1 &&
       eob_decision.escape_bits > 0)
          ? eob_decision.escape_bits
          : 0;
  syntax->eob_escape_value = eob_escape;
  syntax->eob_extra_symbol = eob_extra_symbol;
  syntax->eob_extra_lsb_bits = eob_lsb_bits;
  syntax->eob_extra_lsb_value = eob_lsb_value;

  std::vector<std::int32_t> scan(area, 0);
  build_scan(geometry.w, geometry.h, geometry.tx_class, scan);
  std::vector<std::int32_t> level_pre(area, 0);
  syntax->positions.assign(static_cast<std::size_t>(eob), PositionSyntax{});

  const bool hidden_possible =
      syntax->parity_hiding_eligible && eob > kPhThresh;
  bool is_hidden = false;
  std::int32_t sum_abs1 = 0;
  std::int32_t num_nz = 0;

  for (std::int32_t c = eob - 1; c >= 0; --c) {
    PositionSyntax& p = syntax->positions[static_cast<std::size_t>(c)];
    p.scan_index = c;
    p.pos = scan[static_cast<std::size_t>(c)];
    p.row = p.pos / geometry.w;
    p.col = p.pos % geometry.w;
    const LevelLimits limits =
        level_limits(p.row, p.col, geometry.tx_class, geometry.plane);
    p.is_lf = limits.is_lf;
    p.is_dc = p.pos == 0;

    const CdfDecision base = coeff_base_cdf_decision(
        c, eob, p.pos, p.row, p.col, level_pre, geometry, is_hidden, p.is_lf);
    p.base_kind = base.kind;
    p.base_ctx = base.ctx;
    p.base_ctx2 = base.ctx2;
    p.base_symbols = base.symbols;
    p.base_offset = (c == eob - 1) ? 1 : 0;

    std::uint16_t* base_cdf = base_cdf_ptr(cdfs, p, geometry);
    if (base_cdf == nullptr) {
      if (error != nullptr) *error = "unknown coeff_base CDF in reader";
      return false;
    }
    std::int32_t symbol = 0;
    status = decoder.decode_symbol(base_cdf, p.base_symbols, &symbol);
    if (status != av2::Status::kOk) {
      if (error != nullptr) *error = av2::status_name(status);
      return false;
    }
    p.base_symbol = symbol;
    std::int32_t level = symbol + p.base_offset;

    if (level > limits.base_levels &&
        !(limits.is_lf && limits.br_max == 0)) {
      const CdfDecision br = coeff_br_cdf_decision(
          p.pos, p.row, p.col, level_pre, geometry, p.is_lf);
      p.br_present = true;
      p.br_kind = br.kind;
      p.br_ctx = br.ctx;
      p.br_symbols = br.symbols;
      std::uint16_t* br_cdf = br_cdf_ptr(cdfs, p);
      if (br_cdf == nullptr) {
        if (error != nullptr) *error = "unknown coeff_br CDF in reader";
        return false;
      }
      std::int32_t br_symbol = 0;
      status = decoder.decode_symbol(br_cdf, p.br_symbols, &br_symbol);
      if (status != av2::Status::kOk) {
        if (error != nullptr) *error = av2::status_name(status);
        return false;
      }
      p.br_symbol = br_symbol;
      level += br_symbol;
    } else {
      p.br_present = false;
      p.br_kind = CdfKind::kNone;
      p.br_symbol = 0;
    }
    p.level = level;
    p.max_level = (is_hidden && c == 0) ? (kNumBaseLevels + 1)
                                       : limits.pre_max;
    level_pre[static_cast<std::size_t>(p.pos)] = level;

    if (hidden_possible && c > 0) {
      sum_abs1 ^= (std::min(level, kMaxBaseBrRange) & 1);
      if (level != 0) ++num_nz;
      is_hidden = num_nz >= kPhThresh;
    }
    if (is_hidden && c == 0) syntax->dc_ctx = p.base_ctx;
  }

  syntax->is_hidden = is_hidden;
  syntax->sum_abs1 = sum_abs1;
  syntax->num_nz = num_nz;

  std::int32_t hr_level_avg = 0;
  for (std::int32_t c = eob - 1; c >= 0; --c) {
    PositionSyntax& p = syntax->positions[static_cast<std::size_t>(c)];
    const std::int32_t level = level_pre[static_cast<std::size_t>(p.pos)];
    const bool hidden_dc = is_hidden && c == 0;
    const LevelLimits limits =
        level_limits(p.row, p.col, geometry.tx_class, geometry.plane);
    const std::int32_t max_level =
        hidden_dc ? (kNumBaseLevels + 1) : limits.pre_max;
    const bool sign_present =
        (level != 0) || (hidden_dc && sum_abs1 > 0);
    p.sign_present = sign_present;
    p.dc_style_sign =
        sign_is_dc_style(geometry.plane, geometry.tx_class, p.pos, p.row,
                         p.col);
    if (sign_present) {
      if (p.dc_style_sign) {
        const std::int32_t sign_ctx =
            (p.pos == 0) ? config.dc_sign_ctx : 0;
        std::uint16_t* cdf =
            &cdfs.dc_sign[0][is_hidden ? 1 : 0][sign_ctx][0];
        std::int32_t sign = 0;
        status = decoder.decode_symbol(cdf, 2, &sign);
        if (status != av2::Status::kOk) {
          if (error != nullptr) *error = av2::status_name(status);
          return false;
        }
        p.sign = sign;
      } else {
        bool bit = false;
        status = decoder.read_bool(&bit);
        if (status != av2::Status::kOk) {
          if (error != nullptr) *error = av2::status_name(status);
          return false;
        }
        p.sign = bit ? 1 : 0;
      }
    } else {
      p.sign = 0;
    }

    p.escape_present = level >= max_level;
    if (p.escape_present) {
      const std::int32_t lvl_shift = hidden_dc ? 1 : 0;
      const std::int32_t predictor = hr_level_avg >> lvl_shift;
      p.hr_predictor = predictor;
      std::int32_t x = 0;
      if (!read_escape_bits(decoder, predictor, &x, error)) return false;
      p.escape_x = x;
      p.escape_bits = escape_bit_count(x, predictor);
      hr_level_avg = update_hr_level_avg(x, hr_level_avg, lvl_shift);
    } else {
      p.escape_x = 0;
      p.escape_bits = 0;
      p.hr_predictor = 0;
    }
  }

  return decode_syntax(*syntax, geometry, decoded, error);
}

}  // namespace av2enc::tcq::parity
