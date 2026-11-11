// Pinned parity-hiding goldens for AV2 v1.0.0.
//
// Every value below is deterministic: the syntax fields come from
// `derive_syntax()`, the rate is the static §9.3 `-log2(p)` sum, and the
// payload bytes come from the normative §8.2 entropy coder with
// `disable_cdf_update = 1`, qctx 0, txb_skip_ctx 0, dc_sign_ctx 0 and
// is_inter = false.
//
// `parity_test --print-goldens` regenerates the same initialisers from the
// implementation, so a mismatch is either an intentional format change or a
// regression.  Do not update these values without reviewing the syntax change.
//
// Levels are in §5.20.7.30 scan order.  `base_kind` / `br_kind` / `eob_kind`
// are `CdfKind` values; `-1` means "not present".  `escape_x == -1` means no
// `read_quant` escape; `sign == -1` means no sign syntax.

#pragma once

#include "parity_hiding.h"

#include <array>
#include <cstdint>
#include <vector>

namespace av2enc::tcq::parity::golden {

struct GoldenPosition {
  std::int32_t scan_index;
  std::int32_t pos;
  std::int32_t level;
  std::int32_t base_kind;
  std::int32_t base_ctx;
  std::int32_t base_symbol;
  std::int32_t base_offset;
  std::int32_t br_present;
  std::int32_t br_kind;
  std::int32_t br_ctx;
  std::int32_t br_symbol;
  std::int32_t escape_x;
  std::int32_t escape_bits;
  std::int32_t sign;
  std::int32_t dc_style;
};

struct GoldenBlock {
  const char* name;
  std::int32_t tx_size;
  std::int32_t tx_class;  // 0 2D, 1 HORIZ, 2 VERT
  std::int32_t plane;
  std::int32_t area;
  std::int32_t eob;
  std::int32_t is_hidden;
  std::int32_t sum_abs1;
  std::int32_t num_nz;
  std::int32_t dc_ctx;
  std::int32_t eob_kind;
  std::int32_t eob_ctx;
  std::int32_t eob_symbol;
  std::int32_t eob_symbols;
  std::int32_t eob_pt;
  std::int32_t eob_extra_symbol;
  double rate_bits;
  std::array<std::int32_t, 64> levels;
  std::vector<GoldenPosition> positions;
  std::vector<std::uint8_t> payload;
};

inline const std::vector<GoldenBlock>& blocks() {
  static const std::vector<GoldenBlock> kBlocks = {
      {
          "hidden_dc6_eob5", 1, 0, 0, 64, 5, 1, 0, 4, 4,
          static_cast<std::int32_t>(CdfKind::kEobPt64), 0, 3, 7, 4, 0,
          32.491742431715238,
          {6, 1, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {
              {0, 0, 3, static_cast<std::int32_t>(CdfKind::kCoeffBasePh),
               4, 3, 0, 0, -1, 0, 0, 0, 2, 0, 1},
              {1, 8, 1, static_cast<std::int32_t>(CdfKind::kCoeffBaseLf),
               12, 1, 0, 0, -1, 0, 0, -1, 0, 0, 0},
              {2, 1, 2, static_cast<std::int32_t>(CdfKind::kCoeffBaseLf),
               10, 2, 0, 0, -1, 0, 0, -1, 0, 0, 0},
              {3, 16, 3, static_cast<std::int32_t>(CdfKind::kCoeffBaseLf),
               16, 3, 0, 0, -1, 0, 0, -1, 0, 0, 0},
              {4, 9, 2, static_cast<std::int32_t>(CdfKind::kCoeffBaseLfEob),
               1, 1, 1, 0, -1, 0, 0, -1, 0, 0, 0},
          },
          {0x1f, 0x1d, 0xe7, 0x2d, 0x80, 0x00},
      },
      {
          "hidden_dc8_horiz", 13, 1, 0, 64, 5, 1, 0, 4, 1,
          static_cast<std::int32_t>(CdfKind::kEobPt64), 0, 3, 7, 4, 0,
          30.920944985597174,
          {8, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {
              {0, 0, 3, static_cast<std::int32_t>(CdfKind::kCoeffBasePh),
               1, 3, 0, 0, -1, 0, 0, 1, 2, 0, 1},
              {1, 4, 1, static_cast<std::int32_t>(CdfKind::kCoeffBaseLf),
               22, 1, 0, 0, -1, 0, 0, -1, 0, 0, 1},
              {2, 8, 2, static_cast<std::int32_t>(CdfKind::kCoeffBaseLf),
               23, 2, 0, 0, -1, 0, 0, -1, 0, 0, 1},
              {3, 12, 3, static_cast<std::int32_t>(CdfKind::kCoeffBaseLf),
               23, 3, 0, 0, -1, 0, 0, -1, 0, 0, 1},
              {4, 16, 4, static_cast<std::int32_t>(CdfKind::kCoeffBaseLfEob),
               1, 3, 1, 0, -1, 0, 0, -1, 0, 0, 1},
          },
          {0x1f, 0x31, 0xd9, 0x75, 0x00, 0x00},
      },
      {
          "hidden_dc0", 0, 0, 0, 16, 5, 1, 0, 4, 4,
          static_cast<std::int32_t>(CdfKind::kEobPt16), 0, 3, 5, 4, 0,
          27.640566774555538,
          {0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {
              {0, 0, 0, static_cast<std::int32_t>(CdfKind::kCoeffBasePh),
               4, 0, 0, 0, -1, 0, 0, -1, 0, -1, 1},
              {1, 4, 1, static_cast<std::int32_t>(CdfKind::kCoeffBaseLf),
               13, 1, 0, 0, -1, 0, 0, -1, 0, 0, 0},
              {2, 1, 2, static_cast<std::int32_t>(CdfKind::kCoeffBaseLf),
               11, 2, 0, 0, -1, 0, 0, -1, 0, 0, 0},
              {3, 8, 3, static_cast<std::int32_t>(CdfKind::kCoeffBaseLf),
               16, 3, 0, 0, -1, 0, 0, -1, 0, 0, 0},
              {4, 5, 4, static_cast<std::int32_t>(CdfKind::kCoeffBaseLfEob),
               2, 3, 1, 0, -1, 0, 0, -1, 0, 0, 0},
          },
          {0x37, 0x01, 0x9c, 0x98, 0x00, 0x00},
      },
      {
          "chroma_eob1", 0, 0, 1, 16, 1, 0, 0, 0, 0,
          static_cast<std::int32_t>(CdfKind::kEobPt16), 2, 0, 5, 1, 0,
          13.38424726772767,
          {9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {
              {0, 0, 5,
               static_cast<std::int32_t>(CdfKind::kCoeffBaseLfEobUv),
               0, 4, 1, 0, -1, 0, 0, 4, 4, 0, 0},
          },
          {0x30, 0xd6, 0x00, 0x00},
      },
  };
  return kBlocks;
}

}  // namespace av2enc::tcq::parity::golden
