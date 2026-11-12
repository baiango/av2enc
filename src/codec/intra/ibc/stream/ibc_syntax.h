// Exact AV2 §5.20.5.3/.4 and §8.3 syntax writer/reader for intra block copy.
//
// This module owns the symbols that still_syntax.cpp does not have yet:
//   use_intrabc (S), skip_flag (S), intrabc_mode (S),
//   intrabc_drl_mode (L(1) x m), intrabc_precision (S),
//   assign_mv's shell MVD syntax (read_mv) and its sign bits.
//
// It uses the real §8.2 range coder, so a writer/reader round-trip exercises the
// same CDF adaptation a decoder performs.  CDFs are per tile and are mutated in
// place exactly like the spec.
#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "codec/intra/ibc/stream/ibc_cdfs.h"
#include "codec/intra/ibc/stream/ibc_stack.h"
#include "codec/entropy/av2/av2_entropy.h"

namespace av2enc::still::ibc_stream {

// §9.3 CDFs selected by §8.3.2 for the IBC syntax.  One copy per tile.
struct MvCdfs {
  std::uint16_t shell_set[2][3];
  std::uint16_t shell_last_two_classes[2][3];
  std::uint16_t shell_offset_class2[2][3];
  std::uint16_t shell_offset_low_class[2][2][3];
  std::uint16_t shell_offset_other_class[2][16][3];
  std::uint16_t col_mv_greater[2][2][3];
  std::uint16_t col_mv_index[2][4][3];
  // Indexed by MvPrecision P (0..6; P==2 is unreachable).  The first N
  // entries are used, where N includes the base rate and use count.
  std::uint16_t shell_class0[7][9];
  std::uint16_t shell_class1[7][9];
};

struct IbcCdfs {
  std::uint16_t intrabc[3][3];
  std::uint16_t intrabc_mode[3];
  std::uint16_t intrabc_precision[3];
  std::uint16_t skip[6][3];
  std::uint16_t palette_y_mode[3];
  MvCdfs mv;
};

void init_ibc_cdfs(IbcCdfs& cdfs) noexcept;

// MV_PRECISION_* values (§Table 6.19).
inline constexpr int kMvPrecisionOnePel = 3;
inline constexpr int kMvPrecisionHalfPel = 4;
inline constexpr int kMvPrecisionQuarterPel = 5;
inline constexpr int kMvPrecisionEighthPel = 6;

enum class IntrabcMode : int {
  kNewMv = 0,   // intrabc_mode = 0: MVD present
  kNearMv = 1,  // intrabc_mode = 1: RefStackMv only
};

struct ScreenContentConfig {
  bool force_integer_mv = true;
  bool enable_mvd_sign_derive = false;  // sequence value is 0 for M1
  int max_bvp_drl_bits_minus_1 = 0;
};

// §5.20.5.3 use_intrabc availability test.
bool intrabc_available(bool allow_intrabc, int bsize,
                       bool tree_is_chroma_part = false) noexcept;

// §5.20.5.4 payload.  `diff` is only coded for NEWMV; it is in 1/8 luma
// samples and must be an exact multiple of the selected MvPrecision quantum.
struct BvSyntax {
  IntrabcMode mode = IntrabcMode::kNearMv;
  int ref_mv_idx = 0;
  bool intrabc_precision_quarter = false;  // NEWMV && !force_integer_mv
  BlockVector diff{};
};

bool write_use_intrabc(av2::Encoder& enc, IbcCdfs& cdfs, int ctx, bool value,
                       std::string& error);
bool read_use_intrabc(av2::Decoder& dec, IbcCdfs& cdfs, int ctx, bool& value,
                      std::string& error);

bool write_skip_flag(av2::Encoder& enc, IbcCdfs& cdfs, int ctx, bool value,
                     std::string& error);
bool read_skip_flag(av2::Decoder& dec, IbcCdfs& cdfs, int ctx, bool& value,
                    std::string& error);

// read_mv() §5.20.7.20 in shell form, including the literal sign bits that
// assign_mv() writes for every nonzero component.  `diff8` is in 1/8 luma
// samples.  `mv_ctx` is MV_INTRABC_CONTEXT (1) for IBC.
bool write_mv(av2::Encoder& enc, MvCdfs& cdfs, int mv_ctx, int mv_precision,
              BlockVector diff8, bool derive_sign, std::string& error);
bool read_mv(av2::Decoder& dec, MvCdfs& cdfs, int mv_ctx, int mv_precision,
             bool derive_sign, BlockVector& diff8, std::string& error);

// Write/read intrabc_mode + DRL + precision + assign_mv for one block.  The
// writer needs the absolute target BV; the reader reconstructs it from the
// stack candidate and MVD.
bool write_intrabc_info(av2::Encoder& enc, IbcCdfs& cdfs,
                        const ScreenContentConfig& config,
                        const BvStack& stack, const BvSyntax& syntax,
                        BlockVector target_bv, std::string& error);
bool read_intrabc_info(av2::Decoder& dec, IbcCdfs& cdfs,
                       const ScreenContentConfig& config,
                       const BvStack& stack, BvSyntax& syntax,
                       BlockVector& bv, std::string& error);

// Exposed for tests and for the tile writer.
BlockVector stack_bv_at(const BvStack& stack, int idx) noexcept;
void lower_mv_precision(int mv_precision, BlockVector& bv) noexcept;
int mv_clamp_to_integer_component(int v) noexcept;

}  // namespace av2enc::still::ibc_stream
