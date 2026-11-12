// AV2 v1.0.0 §5.20.7.27 / §8.3 — `dc_sign_horz_vert`.

#include "codec/intra/supplemental/dc_sign_horz_vert.h"

namespace av2enc::still::missing_bits {

const char* tx_class_name(TxClass tx_class) noexcept {
  switch (tx_class) {
    case TxClass::k2D:
      return "TX_CLASS_2D";
    case TxClass::kHoriz:
      return "TX_CLASS_HORIZ";
    case TxClass::kVert:
      return "TX_CLASS_VERT";
  }
  return "TX_CLASS_INVALID";
}

const char* sign_element_name(SignElement element) noexcept {
  switch (element) {
    case SignElement::kDcSign:
      return "dc_sign";
    case SignElement::kDcSignHorzVert:
      return "dc_sign_horz_vert";
    case SignElement::kSignBit:
      return "sign_bit";
  }
  return "invalid";
}

SignElement select_sign_element(int plane, int row, int col,
                                TxClass tx_class) noexcept {
  if (plane == 0) {
    if (row == 0 && col == 0) {
      return SignElement::kDcSign;
    }
    if (tx_class == TxClass::kHoriz && col == 0) {
      return SignElement::kDcSignHorzVert;
    }
    if (tx_class == TxClass::kVert && row == 0) {
      return SignElement::kDcSignHorzVert;
    }
  }
  return SignElement::kSignBit;
}

int dc_sign_context(const std::int8_t* above_dc, const std::int8_t* left_dc,
                    int x4, int y4, int w4, int h4, int mi_cols,
                    int mi_rows) noexcept {
  int dc_sign = 0;
  if (above_dc != nullptr) {
    for (int k = 0; k < w4; ++k) {
      if (x4 + k < mi_cols) {
        const int sign = above_dc[x4 + k];
        if (sign == 1) {
          --dc_sign;
        } else if (sign == 2) {
          ++dc_sign;
        }
      }
    }
  }
  if (left_dc != nullptr) {
    for (int k = 0; k < h4; ++k) {
      if (y4 + k < mi_rows) {
        const int sign = left_dc[y4 + k];
        if (sign == 1) {
          --dc_sign;
        } else if (sign == 2) {
          ++dc_sign;
        }
      }
    }
  }
  if (dc_sign < 0) {
    return 1;
  }
  if (dc_sign > 0) {
    return 2;
  }
  return 0;
}

std::uint16_t* dc_sign_cdf_slot(DcSignCdfTable table, int ptype, int is_hidden,
                                int ctx) noexcept {
  if (table == nullptr || ptype < 0 || ptype > 1 || is_hidden < 0 ||
      is_hidden > 1 || ctx < 0 || ctx > 2) {
    return nullptr;
  }
  return &table[ptype][is_hidden][ctx][0];
}

std::uint16_t* dc_sign_horz_vert_cdf_slot(DcSignCdfTable table, int ptype,
                                          int is_hidden) noexcept {
  return dc_sign_cdf_slot(table, ptype, is_hidden, dc_sign_horz_vert_context());
}

namespace {

bool decode_symbol_checked(av2::Decoder& decoder, std::uint16_t* cdf, int* out) {
  if (cdf == nullptr || out == nullptr) {
    return false;
  }
  int symbol = 0;
  if (decoder.decode_symbol(cdf, 2, &symbol) != av2::Status::kOk) {
    return false;
  }
  if (symbol != 0 && symbol != 1) {
    return false;
  }
  *out = symbol;
  return true;
}

}  // namespace

bool write_dc_sign_horz_vert(av2::Encoder& encoder, std::uint16_t* cdf,
                             int sign) noexcept {
  if (cdf == nullptr || (sign != 0 && sign != 1)) {
    return false;
  }
  return encoder.encode_symbol(cdf, 2, sign) == av2::Status::kOk;
}

bool read_dc_sign_horz_vert(av2::Decoder& decoder, std::uint16_t* cdf,
                            int* sign) noexcept {
  return decode_symbol_checked(decoder, cdf, sign);
}

bool write_dc_sign(av2::Encoder& encoder, std::uint16_t* cdf, int ctx,
                   int sign) noexcept {
  if (cdf == nullptr || ctx < 0 || ctx > 2 || (sign != 0 && sign != 1)) {
    return false;
  }
  return encoder.encode_symbol(cdf, 2, sign) == av2::Status::kOk;
}

bool read_dc_sign(av2::Decoder& decoder, std::uint16_t* cdf, int ctx,
                  int* sign) noexcept {
  if (ctx < 0 || ctx > 2) {
    return false;
  }
  return decode_symbol_checked(decoder, cdf, sign);
}

bool write_sign_element(av2::Encoder& encoder, SignElement element,
                        std::uint16_t* cdf, int dc_sign_ctx,
                        int sign) noexcept {
  switch (element) {
    case SignElement::kDcSign:
      return write_dc_sign(encoder, cdf, dc_sign_ctx, sign);
    case SignElement::kDcSignHorzVert:
      return write_dc_sign_horz_vert(encoder, cdf, sign);
    case SignElement::kSignBit:
      if (sign != 0 && sign != 1) {
        return false;
      }
      return encoder.encode_bool(sign != 0) == av2::Status::kOk;
  }
  return false;
}

bool read_sign_element(av2::Decoder& decoder, SignElement element,
                       std::uint16_t* cdf, int dc_sign_ctx,
                       int* sign) noexcept {
  switch (element) {
    case SignElement::kDcSign:
      return read_dc_sign(decoder, cdf, dc_sign_ctx, sign);
    case SignElement::kDcSignHorzVert:
      return read_dc_sign_horz_vert(decoder, cdf, sign);
    case SignElement::kSignBit: {
      bool bit = false;
      if (sign == nullptr ||
          decoder.read_bool(&bit) != av2::Status::kOk) {
        return false;
      }
      *sign = bit ? 1 : 0;
      return true;
    }
  }
  return false;
}

}  // namespace av2enc::still::missing_bits
