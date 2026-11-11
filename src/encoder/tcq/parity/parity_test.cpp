// Deterministic tests for AV2 parity hiding.
//
// Covers:
//   * geometry, adjusted sizes, all three scan classes;
//   * §5.20.7.27 eligibility and every exclusion (plane, lossless, TCQ, IDTX,
//     FSC, frame flag);
//   * exact syntax derivation and the normative inverse over random blocks,
//     every parity, escapes, all EOB sizes, all TX sizes, planes and positions;
//   * the §8.2 entropy round-trip with static and adaptive CDFs;
//   * the encoder level choice, including exhaustive min-distortion oracle
//     checks on small blocks and the TCQ exclusion;
//   * the pinned goldens in parity_goldens.h.
//
// Run with `--print-goldens` to print the current syntax/payload of the
// golden blocks (the values that parity_goldens.h pins).

#include "../tcq_flags.h"
#include "parity_entropy.h"
#include "parity_goldens.h"
#include "parity_hiding.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using av2enc::tcq::parity::BlockSyntax;
using av2enc::tcq::parity::CdfKind;
using av2enc::tcq::parity::DecodedBlock;
using av2enc::tcq::parity::EntropyConfig;
using av2enc::tcq::parity::Eligibility;
using av2enc::tcq::parity::Geometry;
using av2enc::tcq::parity::QuantizeConfig;
using av2enc::tcq::parity::QuantizeMode;
using av2enc::tcq::parity::QuantizeResult;
using av2enc::tcq::parity::RateConfig;
using av2enc::tcq::parity::TxClass;

int checks = 0;
int failures = 0;

void check(bool condition, const char* what) {
  ++checks;
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

void check_near(double a, double b, double tolerance, const char* what) {
  ++checks;
  const double scale = 1.0 + std::max(std::fabs(a), std::fabs(b));
  if (!(std::fabs(a - b) <= tolerance * scale)) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s (%.17g vs %.17g)\n", what, a, b);
  }
}

std::uint32_t rng_state = 0x9e3779b9u;

std::uint32_t next_random() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

std::int32_t random_int(std::int32_t low, std::int32_t high) {
  return low + static_cast<std::int32_t>(
                   next_random() % static_cast<std::uint32_t>(high - low + 1));
}

Geometry geometry_for(std::int32_t tx, std::int32_t cls, std::int32_t plane) {
  Geometry geometry;
  std::string error;
  if (!av2enc::tcq::parity::make_geometry(
          tx, static_cast<TxClass>(cls), plane, &geometry, &error)) {
    std::fprintf(stderr, "geometry_for failed: %s\n", error.c_str());
    std::abort();
  }
  return geometry;
}

std::vector<std::int32_t> scan_of(const Geometry& geometry) {
  std::vector<std::int32_t> scan(
      static_cast<std::size_t>(geometry.w) *
          static_cast<std::size_t>(geometry.h),
      0);
  av2enc::tcq::parity::build_scan(geometry.w, geometry.h, geometry.tx_class,
                                  scan);
  return scan;
}

Eligibility standard_eligibility() {
  Eligibility eligibility;
  eligibility.allow_parity_hiding = true;
  return eligibility;
}

int count_nonzero(const std::vector<std::int32_t>& levels) {
  int count = 0;
  for (std::int32_t level : levels) {
    if (level != 0) ++count;
  }
  return count;
}

// AC non-zeros and their capped parity, computed exactly as the first pass.
void ac_stats(const Geometry& geometry,
              const std::vector<std::int32_t>& levels_scan,
              std::int32_t* num_nz, std::int32_t* sum_abs1) {
  const std::vector<std::int32_t> scan = scan_of(geometry);
  std::int32_t eob = 0;
  for (std::int32_t c = static_cast<std::int32_t>(levels_scan.size()) - 1;
       c >= 0; --c) {
    if (levels_scan[static_cast<std::size_t>(c)] != 0) {
      eob = c + 1;
      break;
    }
  }
  std::int32_t count = 0;
  std::int32_t sum = 0;
  for (std::int32_t c = eob - 1; c >= 1; --c) {
    const std::int32_t pos = scan[static_cast<std::size_t>(c)];
    const std::int32_t row = pos / geometry.w;
    const std::int32_t col = pos % geometry.w;
    const auto limits = av2enc::tcq::parity::level_limits(
        row, col, geometry.tx_class, geometry.plane);
    const std::int32_t abs_level =
        std::abs(levels_scan[static_cast<std::size_t>(c)]);
    const std::int32_t syntax =
        std::min(abs_level, limits.pre_max);
    sum ^= std::min(syntax, 6) & 1;
    if (abs_level != 0) ++count;
  }
  *num_nz = count;
  *sum_abs1 = sum;
}

// ---------------------------------------------------------------------------
// Constants / small helpers
// ---------------------------------------------------------------------------

void test_parity_hiding_constants() {
  check(av2enc::tcq::parity::kPhThresh == 4, "PHTHRESH");
  check(av2enc::tcq::parity::kNumBaseLevels == 2, "NUM_BASE_LEVELS");
  check(av2enc::tcq::parity::kCoeffBaseRange == 3, "COEFF_BASE_RANGE");
  check(av2enc::tcq::parity::kMaxBaseBrRange == 6, "MAX_BASE_BR_RANGE");
  check(av2enc::tcq::parity::kLfNumBaseLevels == 4, "LF_NUM_BASE_LEVELS");
  check(av2enc::tcq::parity::kLfBaseSymbols == 6, "LF_BASE_SYMBOLS");
  check(av2enc::tcq::parity::kPhContexts == 5, "COEFF_BASE_PH_CONTEXTS");
  check(av2enc::tcq::parity::clip3(1, 6, 0) == 1, "clip3 low");
  check(av2enc::tcq::parity::clip3(1, 6, 9) == 6, "clip3 high");
  check(av2enc::tcq::parity::get_msb(0) == 0, "msb 0");
  check(av2enc::tcq::parity::get_msb(1) == 0, "msb 1");
  check(av2enc::tcq::parity::get_msb(8) == 3, "msb 8");
  check(av2enc::tcq::parity::update_hr_level_avg(1, 0, 1) == 1,
        "hidden DC hr update shifts x");
  check(av2enc::tcq::parity::update_hr_level_avg(1, 4, 0) == 2,
        "non-hidden hr update");
  check(av2enc::tcq::parity::dequantize_level(3, 528, 0) == 198,
        "dequant level");
  check(av2enc::tcq::parity::dequantize_level(-3, 528, 1) == -99,
        "dequant level signed/shifted");
  check(av2enc::tcq::parity::escape_bit_count(0, 0) == 2,
        "escape bits x=0");
  check(av2enc::tcq::parity::escape_bit_count(5, 0) == 4,
        "escape bits x=5");
}

// ---------------------------------------------------------------------------
// Geometry / tables
// ---------------------------------------------------------------------------

void test_geometry() {
  for (std::int32_t tx = 0; tx < 25; ++tx) {
    for (std::int32_t cls = 0; cls < 3; ++cls) {
      for (std::int32_t plane = 0; plane < 3; ++plane) {
        const Geometry geometry = geometry_for(tx, cls, plane);
        check(geometry.coded_w == av2enc::tcq::parity::kTxWidth[tx],
              "coded width table");
        check(geometry.coded_h == av2enc::tcq::parity::kTxHeight[tx],
              "coded height table");
        check(geometry.w == std::min(geometry.coded_w, 32),
              "adjusted width");
        check(geometry.h == std::min(geometry.coded_h, 32),
              "adjusted height");
        const std::vector<std::int32_t> scan = scan_of(geometry);
        check(scan.size() ==
                  static_cast<std::size_t>(geometry.w * geometry.h),
              "scan size");
        std::vector<bool> seen(scan.size(), false);
        for (std::int32_t pos : scan) {
          check(pos >= 0 && pos < static_cast<std::int32_t>(scan.size()),
                "scan position range");
          if (pos >= 0 && pos < static_cast<std::int32_t>(scan.size())) {
            check(!seen[static_cast<std::size_t>(pos)],
                  "scan is a permutation");
            seen[static_cast<std::size_t>(pos)] = true;
          }
        }
      }
    }
  }

  // §5.20.7.30 order.
  const Geometry g8 = geometry_for(1, 0, 0);
  const std::vector<std::int32_t> scan8 = scan_of(g8);
  check(scan8[0] == 0 && scan8[1] == 8 && scan8[2] == 1 &&
            scan8[3] == 16 && scan8[4] == 9 && scan8[5] == 2,
        "8x8 diagonal scan head");
  const Geometry gh = geometry_for(13, 1, 0);  // TX_4X16
  const std::vector<std::int32_t> scanh = scan_of(gh);
  check(scanh[0] == 0 && scanh[1] == 4 && scanh[2] == 8 &&
            scanh[3] == 12 && scanh[4] == 16,
        "HORIZ column-major scan");
  const Geometry gv = geometry_for(14, 2, 0);  // TX_16X4
  const std::vector<std::int32_t> scanv = scan_of(gv);
  check(scanv[0] == 0 && scanv[1] == 1 && scanv[2] == 2 &&
            scanv[3] == 3 && scanv[4] == 4,
        "VERT row-major scan");

  check(av2enc::tcq::parity::eob_multisize(geometry_for(0, 0, 0)) == 0,
        "eobMultisize 4x4");
  check(av2enc::tcq::parity::eob_multisize(geometry_for(1, 0, 0)) == 2,
        "eobMultisize 8x8");
  check(av2enc::tcq::parity::eob_multisize(geometry_for(4, 0, 0)) == 6,
        "eobMultisize 64x64");
  check(av2enc::tcq::parity::eob_multisize(geometry_for(17, 0, 0)) == 5,
        "eobMultisize 16x64");
  check(av2enc::tcq::parity::eob_multisize(geometry_for(23, 0, 0)) == 3,
        "eobMultisize 4x64");
  check(av2enc::tcq::parity::eob_pt_for_eob(1) == 1, "eob_pt(1)");
  check(av2enc::tcq::parity::eob_pt_for_eob(2) == 2, "eob_pt(2)");
  check(av2enc::tcq::parity::eob_pt_for_eob(3) == 3, "eob_pt(3)");
  check(av2enc::tcq::parity::eob_pt_for_eob(5) == 4, "eob_pt(5)");
  check(av2enc::tcq::parity::eob_pt_for_eob(1024) == 11, "eob_pt(1024)");

  // §5.20.7.27 get_lf_limits.
  check(av2enc::tcq::parity::is_low_frequency(0, 0, TxClass::k2D, 0),
        "2D luma DC is LF");
  check(av2enc::tcq::parity::is_low_frequency(0, 3, TxClass::k2D, 0),
        "2D luma (0,3) is LF");
  check(!av2enc::tcq::parity::is_low_frequency(1, 3, TxClass::k2D, 0),
        "2D luma (1,3) is HF");
  check(av2enc::tcq::parity::is_low_frequency(0, 0, TxClass::k2D, 1),
        "2D chroma DC is LF");
  check(!av2enc::tcq::parity::is_low_frequency(0, 1, TxClass::k2D, 1),
        "2D chroma (0,1) is HF");
  check(av2enc::tcq::parity::is_low_frequency(9, 0, TxClass::kHoriz, 0),
        "HORIZ col<2 is LF");
  check(!av2enc::tcq::parity::is_low_frequency(0, 2, TxClass::kHoriz, 0),
        "HORIZ col 2 is HF");
  check(av2enc::tcq::parity::is_low_frequency(0, 9, TxClass::kVert, 0),
        "VERT row<2 is LF");
  check(!av2enc::tcq::parity::is_low_frequency(2, 0, TxClass::kVert, 0),
        "VERT row 2 is HF");

  const auto lf = av2enc::tcq::parity::level_limits(0, 0, TxClass::k2D, 0);
  check(lf.base_levels == 4 && lf.base_max == 5 && lf.br_max == 3 &&
            lf.pre_max == 8,
        "LF luma limits");
  const auto hf = av2enc::tcq::parity::level_limits(2, 2, TxClass::k2D, 0);
  check(hf.base_levels == 2 && hf.base_max == 3 && hf.br_max == 3 &&
            hf.pre_max == 6,
        "HF luma limits");
  const auto uvhf = av2enc::tcq::parity::level_limits(0, 1, TxClass::k2D, 1);
  check(uvhf.base_levels == 2 && uvhf.base_max == 3 && uvhf.br_max == 3 &&
            uvhf.pre_max == 6,
        "HF chroma limits");
  const auto uvl = av2enc::tcq::parity::level_limits(0, 0, TxClass::k2D, 1);
  check(uvl.base_levels == 4 && uvl.base_max == 5 && uvl.br_max == 0 &&
            uvl.pre_max == 5,
        "LF chroma limits");
}

void test_eligibility() {
  const Geometry luma = geometry_for(1, 0, 0);
  Eligibility e;
  e.allow_parity_hiding = true;
  check(av2enc::tcq::parity::parity_hiding_eligible(luma, e),
        "PH eligible");
  e.lossless = true;
  check(!av2enc::tcq::parity::parity_hiding_eligible(luma, e),
        "lossless excludes PH");
  e.lossless = false;
  e.allow_tcq = true;
  check(!av2enc::tcq::parity::parity_hiding_eligible(luma, e),
        "allow_tcq excludes PH");
  e.allow_tcq = false;
  e.tx_type_is_idtx = true;
  check(!av2enc::tcq::parity::parity_hiding_eligible(luma, e),
        "IDTX excludes PH");
  e.tx_type_is_idtx = false;
  e.use_fsc = true;
  check(!av2enc::tcq::parity::parity_hiding_eligible(luma, e),
        "FSC excludes PH");
  e.use_fsc = false;
  e.allow_parity_hiding = false;
  check(!av2enc::tcq::parity::parity_hiding_eligible(luma, e),
        "frame flag excludes PH");
  e.allow_parity_hiding = true;
  for (std::int32_t plane = 1; plane <= 2; ++plane) {
    check(!av2enc::tcq::parity::parity_hiding_eligible(
              geometry_for(1, 0, plane), e),
          "chroma excludes PH");
  }
}

// ---------------------------------------------------------------------------
// Context axes (2D vs 1D LF/HF buckets)
// ---------------------------------------------------------------------------

void test_context_axes() {
  Eligibility eligibility;
  eligibility.allow_parity_hiding = true;
  std::string error;

  // HORIZ 4x16: pos 2 is (row 0, col 2), a 1D HF coefficient.  Column-major
  // scan makes it index 32; keep index 33 non-zero so index 32 is not the EOB.
  {
    const Geometry geometry = geometry_for(13, 1, 0);
    std::vector<std::int32_t> levels(64, 0);
    levels[0] = 1;
    levels[32] = 2;
    levels[33] = 1;
    BlockSyntax syntax;
    check(av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                             &syntax, &error),
          "1D HF derive");
    const auto& p = syntax.positions[32];
    check(p.pos == 2 && !p.is_lf, "1D HF position");
    check(p.base_kind == CdfKind::kCoeffBase, "1D HF base CDF");
    check(p.base_ctx >= 15 && p.base_ctx <= 19,
          "1D HF luma base context is ctx2+15");
  }

  // HORIZ 4x16: pos 1 is (row 0, col 1), a 1D LF coefficient, index 16.
  // col != 0 so the LF bucket is 21+7+Min(ctx,4) (28..32).
  {
    const Geometry geometry = geometry_for(13, 1, 0);
    std::vector<std::int32_t> levels(64, 0);
    levels[0] = 1;
    levels[16] = 2;
    levels[17] = 1;
    BlockSyntax syntax;
    check(av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                             &syntax, &error),
          "1D LF derive");
    const auto& p = syntax.positions[16];
    check(p.pos == 1 && p.is_lf, "1D LF position");
    check(p.base_kind == CdfKind::kCoeffBaseLf, "1D LF base CDF");
    check(p.base_ctx >= 28 && p.base_ctx <= 32,
          "1D LF col!=0 luma base context");
  }

  // VERT 16x4: pos 32 is (row 2, col 0), a 1D HF coefficient, scan index 32.
  {
    const Geometry geometry = geometry_for(14, 2, 0);
    std::vector<std::int32_t> levels(64, 0);
    levels[0] = 1;
    levels[32] = 2;
    levels[33] = 1;
    BlockSyntax syntax;
    check(av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                             &syntax, &error),
          "VERT HF derive");
    const auto& p = syntax.positions[32];
    check(p.pos == 32 && !p.is_lf, "VERT HF position");
    check(p.base_ctx >= 15 && p.base_ctx <= 19,
          "VERT HF luma base context is ctx2+15");
  }

  // 1D chroma HF base contexts are ctx2+8 (8..11).
  {
    const Geometry geometry = geometry_for(13, 1, 1);
    std::vector<std::int32_t> levels(64, 0);
    levels[0] = 1;
    levels[32] = 2;
    levels[33] = 1;
    BlockSyntax syntax;
    check(av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                             &syntax, &error),
          "1D chroma HF derive");
    const auto& p = syntax.positions[32];
    check(p.base_kind == CdfKind::kCoeffBaseUv, "1D chroma HF base CDF");
    check(p.base_ctx >= 8 && p.base_ctx <= 11,
          "1D chroma HF base context is ctx2+8");
  }

  // 1D luma DC (`pos == 0`) coeff_br context uses mag + 7; 2D DC uses mag.
  {
    const Geometry geometry = geometry_for(13, 1, 0);
    std::vector<std::int32_t> levels(64, 0);
    levels[0] = 5;  // LF luma, > base_levels, br present with symbol 0
    BlockSyntax syntax;
    check(av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                             &syntax, &error),
          "1D DC br derive");
    const auto& p = syntax.positions[0];
    check(p.br_present && p.br_kind == CdfKind::kCoeffBrLf,
          "1D DC br present");
    check(p.br_ctx == 7, "1D DC br context is mag+7");
  }
  {
    const Geometry geometry = geometry_for(1, 0, 0);
    std::vector<std::int32_t> levels(64, 0);
    levels[0] = 5;
    BlockSyntax syntax;
    check(av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                             &syntax, &error),
          "2D DC br derive");
    const auto& p = syntax.positions[0];
    check(p.br_present && p.br_kind == CdfKind::kCoeffBrLf,
          "2D DC br present");
    check(p.br_ctx == 0, "2D DC br context is mag");
  }

  // 2D luma HF base contexts are 0..14 and never use the 1D 15..19 range.
  {
    const Geometry geometry = geometry_for(1, 0, 0);
    std::vector<std::int32_t> levels(64, 0);
    levels[0] = 1;
    levels[10] = 2;
    levels[11] = 1;
    BlockSyntax syntax;
    check(av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                             &syntax, &error),
          "2D HF derive");
    const auto& p = syntax.positions[10];
    if (!p.is_lf) {
      check(p.base_ctx >= 0 && p.base_ctx <= 14,
            "2D HF luma base context is 0..14");
    }
  }
}

// ---------------------------------------------------------------------------
// Interop with the existing TCQ header-flag derivation
// ---------------------------------------------------------------------------

void test_tcq_flag_interop() {
  using av2enc::tcq::HeaderFlags;
  using av2enc::tcq::HeaderInput;
  const Geometry geometry = geometry_for(1, 0, 0);

  // Single-picture still, TCQ off: the sequence flags allow PH.
  {
    HeaderInput input;
    input.single_picture_header_flag = true;
    input.enable_parity_hiding = true;
    input.frame_allow_parity_hiding = true;
    HeaderFlags flags;
    check(av2enc::tcq::derive_header_flags(input, &flags),
          "TCQ header flags derive (PH on)");
    check(flags.enable_parity_hiding && flags.allow_parity_hiding,
          "TCQ header flags allow PH with TCQ off");
    Eligibility eligibility;
    eligibility.allow_parity_hiding = flags.allow_parity_hiding;
    eligibility.allow_tcq = flags.allow_tcq;
    check(av2enc::tcq::parity::parity_hiding_eligible(geometry, eligibility),
          "derived flags make the block PH-eligible");
  }

  // Single-picture still, TCQ on: PH is forced off.
  {
    HeaderInput input;
    input.single_picture_header_flag = true;
    input.enable_tcq = true;
    input.enable_parity_hiding = true;
    input.frame_allow_parity_hiding = true;
    HeaderFlags flags;
    check(av2enc::tcq::derive_header_flags(input, &flags),
          "TCQ header flags derive (TCQ on)");
    check(!flags.enable_parity_hiding, "single-picture TCQ drops PH");
    check(!flags.allow_parity_hiding, "frame allow_parity_hiding is 0");
    Eligibility eligibility;
    eligibility.allow_parity_hiding = flags.allow_parity_hiding;
    eligibility.allow_tcq = flags.allow_tcq;
    check(!av2enc::tcq::parity::parity_hiding_eligible(geometry, eligibility),
          "TCQ-on flags make the block PH-ineligible");
  }

  // Per-frame TCQ with the frame choosing TCQ also forces PH off.
  {
    HeaderInput input;
    input.enable_tcq = true;
    input.choose_tcq_per_frame = true;
    input.enable_parity_hiding = true;
    input.frame_allow_tcq = true;
    input.frame_allow_parity_hiding = true;
    HeaderFlags flags;
    check(av2enc::tcq::derive_header_flags(input, &flags),
          "TCQ header flags derive (per-frame TCQ)");
    check(!flags.allow_parity_hiding, "per-frame TCQ forces PH off");
    Eligibility eligibility;
    eligibility.allow_parity_hiding = flags.allow_parity_hiding;
    eligibility.allow_tcq = flags.allow_tcq;
    check(!av2enc::tcq::parity::parity_hiding_eligible(geometry, eligibility),
          "per-frame TCQ flags make the block PH-ineligible");
  }
}

// ---------------------------------------------------------------------------
// DC-sign CDF axis: dc_sign uses the caller context; dc_sign_horz_vert uses 0
// ---------------------------------------------------------------------------

void test_sign_contexts() {
  RateConfig ctx0;
  ctx0.qctx = 0;
  ctx0.txb_skip_ctx = 0;
  ctx0.dc_sign_ctx = 0;
  RateConfig ctx2 = ctx0;
  ctx2.dc_sign_ctx = 2;

  // HORIZ 4x16: the coefficient at (row 0, col 0) is the true DC, but (row 1,
  // col 0) is the dc_sign_horz_vert class.  With no DC and one negative AC in
  // column 0, the only sign is horz-vert and must use context 0; the caller's
  // dc_sign_ctx must not matter.
  for (std::int32_t cls = 1; cls <= 2; ++cls) {
    const Geometry geometry = geometry_for(cls == 1 ? 13 : 14, cls, 0);
    const std::size_t area = 64;
    std::vector<std::int32_t> levels(area, 0);
    levels[1] = -1;  // scan index 1 is a col/row-zero coefficient
    Eligibility eligibility;
    eligibility.allow_parity_hiding = true;
    BlockSyntax syntax;
    std::string error;
    check(av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                             &syntax, &error),
          "sign context derive");
    check(syntax.positions[1].dc_style_sign,
          "1D col/row-zero coefficient uses dc_sign_horz_vert");
    check_near(av2enc::tcq::parity::block_rate_bits(syntax, geometry, ctx0),
               av2enc::tcq::parity::block_rate_bits(syntax, geometry, ctx2),
               0.0, "horz-vert sign ignores the caller dc_sign_ctx");
  }

  // The true DC does use the caller context.
  {
    const Geometry geometry = geometry_for(1, 0, 0);
    std::vector<std::int32_t> levels(64, 0);
    levels[0] = -1;
    Eligibility eligibility;
    eligibility.allow_parity_hiding = true;
    BlockSyntax syntax;
    std::string error;
    check(av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                             &syntax, &error),
          "DC sign derive");
    const double bits0 =
        av2enc::tcq::parity::block_rate_bits(syntax, geometry, ctx0);
    const double bits2 =
        av2enc::tcq::parity::block_rate_bits(syntax, geometry, ctx2);
    check(std::fabs(bits0 - bits2) > 1e-6,
          "true DC sign uses the caller dc_sign_ctx");
  }
}

// ---------------------------------------------------------------------------
// Syntax round-trip
// ---------------------------------------------------------------------------

void test_syntax_roundtrip() {
  int hidden_blocks = 0;
  int escape_blocks = 0;
  int lf_escape_blocks = 0;
  rng_state = 0x12345678u;
  for (std::int32_t tx = 0; tx < 25; ++tx) {
    for (std::int32_t cls = 0; cls < 3; ++cls) {
      for (std::int32_t plane = 0; plane < 3; ++plane) {
        const Geometry geometry = geometry_for(tx, cls, plane);
        const std::size_t area =
            static_cast<std::size_t>(geometry.w * geometry.h);
        const std::vector<std::int32_t> scan = scan_of(geometry);
        for (int trial = 0; trial < 8; ++trial) {
          std::vector<std::int32_t> levels(area, 0);
          const int density = 20 + trial * 9;
          for (std::size_t i = 0; i < area; ++i) {
            if (static_cast<int>(next_random() % 100) < density) {
              const std::int32_t mag = 1 + static_cast<std::int32_t>(
                                               next_random() % 14);
              levels[i] = (next_random() & 1U) != 0U ? -mag : mag;
            }
          }
          Eligibility eligibility;
          eligibility.allow_parity_hiding = true;
          // Make the hidden constraint legal when the block would hide.
          std::int32_t num_nz = 0;
          std::int32_t sum = 0;
          ac_stats(geometry, levels, &num_nz, &sum);
          if (av2enc::tcq::parity::parity_hiding_eligible(geometry,
                                                          eligibility)) {
            std::int32_t eob = 0;
            for (std::int32_t c = static_cast<std::int32_t>(area) - 1;
                 c >= 0; --c) {
              if (levels[static_cast<std::size_t>(c)] != 0) {
                eob = c + 1;
                break;
              }
            }
            if (eob > 4 && num_nz >= 4) {
              const std::int32_t dc_parity =
                  std::abs(levels[0]) & 1;
              if (dc_parity != sum) {
                // Flip the DC magnitude by one; both parities are legal.
                const std::int32_t old = levels[0];
                const std::int32_t mag = std::abs(old);
                const std::int32_t next =
                    (sum != 0) ? mag + 1 : std::max(0, mag - 1);
                levels[0] = old < 0 ? -next : next;
              }
            }
          }
          BlockSyntax syntax;
          std::string error;
          const bool derived = av2enc::tcq::parity::derive_syntax(
              levels, geometry, eligibility, &syntax, &error);
          check(derived, "derive_syntax random block");
          if (!derived) {
            std::fprintf(stderr, "  %s\n", error.c_str());
            continue;
          }
          if (syntax.is_hidden) ++hidden_blocks;
          for (const auto& p : syntax.positions) {
            if (p.escape_present) {
              ++escape_blocks;
              if (p.is_lf) ++lf_escape_blocks;
            }
          }
          if (syntax.is_hidden) {
            check((std::abs(levels[0]) & 1) == syntax.sum_abs1,
                  "hidden DC parity equals sumAbs1");
          }
          DecodedBlock decoded;
          const bool ok = av2enc::tcq::parity::decode_syntax(
              syntax, geometry, &decoded, &error);
          check(ok, "decode_syntax random block");
          if (!ok) {
            std::fprintf(stderr, "  %s\n", error.c_str());
            continue;
          }
          for (std::size_t c = 0; c < area; ++c) {
            check(decoded.quant_levels[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])] ==
                      levels[c],
                  "decoded level equals the input in scan order");
          }
          check(decoded.is_hidden == syntax.is_hidden,
                "decoded is_hidden");
          check(decoded.sum_abs1 == syntax.sum_abs1,
                "decoded sumAbs1");
          check(decoded.eob == syntax.eob, "decoded eob");
        }
      }
    }
  }
  check(hidden_blocks > 100, "random coverage hit hidden blocks");
  check(escape_blocks > 100, "random coverage hit escapes");
  check(lf_escape_blocks > 10, "random coverage hit LF escapes");
}

void test_parities_and_escapes() {
  const Geometry g = geometry_for(1, 0, 0);  // 8x8
  // 4 AC non-zeros (scan 1..4); DC parity 0/1 and AC parity 0/1.
  for (std::int32_t dc_parity = 0; dc_parity <= 1; ++dc_parity) {
    for (std::int32_t ac_parity = 0; ac_parity <= 1; ++ac_parity) {
      std::vector<std::int32_t> levels(64, 0);
      levels[0] = dc_parity ? 5 : 4;
      levels[1] = 1;                     // odd
      levels[2] = 2;                     // even
      levels[3] = 3;                     // odd
      levels[4] = ac_parity ? 3 : 4;     // odd : even
      const std::int32_t expected_sum = 1 ^ 0 ^ 1 ^ ac_parity;
      BlockSyntax syntax;
      std::string error;
      Eligibility eligibility;
      eligibility.allow_parity_hiding = true;
      const bool derived = av2enc::tcq::parity::derive_syntax(
          levels, g, eligibility, &syntax, &error);
      if (expected_sum != dc_parity) {
        check(!derived, "mismatched parity is rejected");
      } else {
        check(derived, "matched parity derives");
        if (derived) {
          check(syntax.is_hidden, "matched parity hides");
          check(syntax.sum_abs1 == expected_sum, "sumAbs1 value");
          DecodedBlock decoded;
          check(av2enc::tcq::parity::decode_syntax(syntax, g, &decoded, &error),
                "matched parity decodes");
          check(decoded.quant_levels[0] == levels[0],
                "matched parity DC round-trip");
        }
      }
    }
  }

  // Hidden DC escape values: q = |DC|/2.  q >= 3 sends an escape.
  for (std::int32_t dc = 0; dc <= 18; dc += 2) {
    std::vector<std::int32_t> levels(64, 0);
    levels[0] = dc;
    levels[1] = 1;
    levels[2] = 2;
    levels[3] = 3;
    levels[4] = 3;  // sum == 1, so keeps DC even? 1^0^1^1 = 1
    // Use an even AC sum instead: 1,2,3,4.
    levels[4] = 4;
    BlockSyntax syntax;
    std::string error;
    Eligibility eligibility;
    eligibility.allow_parity_hiding = true;
    check(av2enc::tcq::parity::derive_syntax(levels, g, eligibility, &syntax,
                                             &error),
          "hidden DC escape derive");
    const std::int32_t q = dc >> 1;
    check(syntax.positions[0].base_symbol == std::min(q, 3),
          "hidden DC base symbol is min(q,3)");
    check(syntax.positions[0].escape_present == (q >= 3),
          "hidden DC escape threshold");
    if (q >= 3) {
      check(syntax.positions[0].escape_x == q - 3,
            "hidden DC escape value");
      check(syntax.positions[0].hr_predictor == 0,
            "hidden DC escape predictor is shifted");
    } else {
      check(!syntax.positions[0].escape_present,
            "no hidden DC escape below q=3");
    }
    DecodedBlock decoded;
    check(av2enc::tcq::parity::decode_syntax(syntax, g, &decoded, &error),
          "hidden DC escape decode");
    check(decoded.quant_levels[0] == dc, "hidden DC escape round-trip");
  }

  // EOB-prefix escape boundaries: eob_pt == cdf_n still writes the zero
  // escape literal for the 256/512/1024 tables (escape_bits > 0).
  auto check_eob_boundary = [&](std::int32_t tx, std::int32_t eob) {
    const Geometry geometry = geometry_for(tx, 0, 0);
    const std::size_t area =
        static_cast<std::size_t>(geometry.w * geometry.h);
    std::vector<std::int32_t> levels(area, 0);
    if (eob > 0) levels[0] = 2;
    for (std::int32_t c = 1; c < eob; ++c) {
      levels[static_cast<std::size_t>(c)] = 2;
    }
    Eligibility eligibility;
    eligibility.allow_parity_hiding = true;
    BlockSyntax syntax;
    std::string error;
    check(av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                             &syntax, &error),
          "EOB boundary derive");
    check(syntax.eob == eob, "EOB boundary eob");
    const auto decision = av2enc::tcq::parity::eob_cdf_decision(
        geometry, false);
    if (syntax.eob_pt >= decision.symbols) {
      check(syntax.eob_symbol == decision.symbols - 1,
            "EOB boundary uses the escape symbol");
      check(syntax.eob_escape_bits == decision.escape_bits,
            "EOB boundary writes the literal");
    } else {
      check(syntax.eob_escape_bits == 0, "EOB boundary has no literal");
    }
    DecodedBlock decoded;
    check(av2enc::tcq::parity::decode_syntax(syntax, geometry, &decoded,
                                             &error),
          "EOB boundary decode");
    check(decoded.eob == eob, "EOB boundary round-trip");

    std::vector<std::uint8_t> buffer(1 << 17);
    av2enc::av2::Encoder encoder;
    encoder.init_symbol(buffer.data(), buffer.size());
    encoder.disable_cdf_update = 1;
    av2enc::tcq::parity::CdfStore cdfs(0);
    EntropyConfig config;
    config.eligibility = eligibility;
    config.disable_cdf_update = true;
    check(av2enc::tcq::parity::write_block(encoder, cdfs, syntax, geometry,
                                           config, &error),
          "EOB boundary write");
    std::size_t payload_bytes = 0;
    check(encoder.exit_symbol(&payload_bytes) == av2enc::av2::Status::kOk,
          "EOB boundary encoder exit");
    av2enc::av2::Decoder decoder;
    decoder.disable_cdf_update = 1;
    check(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
              av2enc::av2::Status::kOk,
          "EOB boundary decoder init");
    av2enc::tcq::parity::CdfStore cdfs2(0);
    BlockSyntax decoded_syntax;
    DecodedBlock decoded2;
    check(av2enc::tcq::parity::read_block(decoder, cdfs2, geometry, config,
                                          &decoded_syntax, &decoded2,
                                          &error),
          "EOB boundary read");
    check(decoded_syntax.eob == eob, "EOB boundary entropy eob");
    check(decoder.exit_symbol() == av2enc::av2::Status::kOk,
          "EOB boundary decoder exit");
    const std::vector<std::int32_t> scan = scan_of(geometry);
    for (std::int32_t c = 0; c < eob; ++c) {
      check(decoded2.quant_levels[static_cast<std::size_t>(
                scan[static_cast<std::size_t>(c)])] ==
                levels[static_cast<std::size_t>(c)],
            "EOB boundary entropy level");
    }
  };
  check_eob_boundary(2, 127);
  check_eob_boundary(2, 128);
  check_eob_boundary(3, 128);
  check_eob_boundary(3, 129);
  check_eob_boundary(4, 512);
  check_eob_boundary(4, 513);
  check_eob_boundary(17, 511);
  check_eob_boundary(17, 512);
  for (std::int32_t eob = 0; eob <= 256; ++eob) {
    check_eob_boundary(2, eob);  // eob_pt_256, every eob
  }
  for (std::int32_t eob = 0; eob <= 1024; eob += 16) {
    check_eob_boundary(3, eob);  // eob_pt_1024, sampled
  }

  // Non-hidden escapes at pre_max - 1, pre_max, pre_max + 1 for luma LF/HF
  // and chroma LF/HF.  No PH: numNz is deliberately below 4.
  struct EscapeCase {
    std::int32_t tx;
    std::int32_t cls;
    std::int32_t plane;
    std::int32_t pos;
    std::int32_t mag;
  };
  const EscapeCase cases[] = {
      {1, 0, 0, 0, 7},   // LF luma, pos 0
      {1, 0, 0, 0, 8},   // LF luma escape (x=0)
      {1, 0, 0, 0, 9},   // LF luma escape (x=1)
      {1, 0, 0, 24, 5},  // HF luma, pre_max-1
      {1, 0, 0, 24, 6},  // HF luma escape (x=0)
      {1, 0, 0, 24, 7},  // HF luma escape (x=1)
      {0, 0, 1, 0, 4},   // LF chroma, pre_max-1
      {0, 0, 1, 0, 5},   // LF chroma escape (x=0)
      {0, 0, 1, 0, 6},   // LF chroma escape (x=1)
      {0, 0, 1, 15, 5},  // HF chroma, pre_max-1
      {0, 0, 1, 15, 6},  // HF chroma escape (x=0)
  };
  for (const EscapeCase& c : cases) {
    const Geometry geometry = geometry_for(c.tx, c.cls, c.plane);
    std::vector<std::int32_t> levels(
        static_cast<std::size_t>(geometry.w * geometry.h), 0);
    const std::vector<std::int32_t> scan = scan_of(geometry);
    // Put the coefficient at the requested raster position, find its scan
    // index, and make it the EOB coefficient by leaving later scan entries 0.
    std::int32_t scan_index = -1;
    for (std::size_t i = 0; i < scan.size(); ++i) {
      if (scan[i] == c.pos) scan_index = static_cast<std::int32_t>(i);
    }
    check(scan_index >= 0, "escape case position is in the scan");
    if (scan_index < 0) continue;
    levels[static_cast<std::size_t>(scan_index)] = c.mag;
    BlockSyntax syntax;
    std::string error;
    Eligibility eligibility;
    eligibility.allow_parity_hiding = true;
    check(av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                             &syntax, &error),
          "escape case derives");
    const auto& p = syntax.positions.back();
    const auto limits = av2enc::tcq::parity::level_limits(
        p.row, p.col, geometry.tx_class, geometry.plane);
    check(p.escape_present == (c.mag >= limits.pre_max),
          "escape threshold equals pre_max");
    if (c.mag >= limits.pre_max) {
      check(p.escape_x == c.mag - limits.pre_max, "escape x value");
    }
    DecodedBlock decoded;
    check(av2enc::tcq::parity::decode_syntax(syntax, geometry, &decoded,
                                             &error),
          "escape case decodes");
    check(decoded.quant_levels[static_cast<std::size_t>(c.pos)] == c.mag,
          "escape case round-trip");
  }

  // EOB sizes 0..area and the EOB prefix.
  for (std::int32_t eob = 0; eob <= 64; ++eob) {
    std::vector<std::int32_t> levels(64, 0);
    if (eob > 0) levels[0] = 2;
    for (std::int32_t c = 1; c < eob; ++c) {
      levels[static_cast<std::size_t>(c)] = 2;
    }
    BlockSyntax syntax;
    std::string error;
    Eligibility eligibility;
    eligibility.allow_parity_hiding = true;
    check(av2enc::tcq::parity::derive_syntax(levels, g, eligibility, &syntax,
                                             &error),
          "EOB case derives");
    check(syntax.eob == eob, "EOB value");
    check(syntax.eob_pt ==
              (eob == 0 ? 0 : av2enc::tcq::parity::eob_pt_for_eob(eob)),
          "EOB prefix token");
    DecodedBlock decoded;
    check(av2enc::tcq::parity::decode_syntax(syntax, g, &decoded, &error),
          "EOB case decodes");
    check(decoded.eob == eob, "EOB round-trip");
  }
}

// ---------------------------------------------------------------------------
// Entropy
// ---------------------------------------------------------------------------

void test_entropy() {
  rng_state = 0xabcdef01u;
  for (int adaptive = 0; adaptive <= 1; ++adaptive) {
    for (std::int32_t tx = 0; tx < 25; ++tx) {
      for (std::int32_t cls = 0; cls < 3; ++cls) {
        for (std::int32_t plane = 0; plane < 3; ++plane) {
          const Geometry geometry = geometry_for(tx, cls, plane);
          const std::size_t area =
              static_cast<std::size_t>(geometry.w * geometry.h);
          const std::vector<std::int32_t> scan = scan_of(geometry);
          std::vector<std::int32_t> levels(area, 0);
          for (std::size_t i = 0; i < area; ++i) {
            if ((next_random() % 100) < 45) {
              const std::int32_t mag = 1 + static_cast<std::int32_t>(
                                               next_random() % 16);
              levels[i] = (next_random() & 1U) != 0U ? -mag : mag;
            }
          }
          Eligibility eligibility;
          eligibility.allow_parity_hiding = true;
          std::int32_t num_nz = 0;
          std::int32_t sum = 0;
          ac_stats(geometry, levels, &num_nz, &sum);
          std::int32_t eob = 0;
          for (std::int32_t c = static_cast<std::int32_t>(area) - 1;
               c >= 0; --c) {
            if (levels[static_cast<std::size_t>(c)] != 0) {
              eob = c + 1;
              break;
            }
          }
          if (av2enc::tcq::parity::parity_hiding_eligible(geometry,
                                                          eligibility) &&
              eob > 4 && num_nz >= 4) {
            const std::int32_t dc_parity = std::abs(levels[0]) & 1;
            if (dc_parity != sum) {
              const std::int32_t old = levels[0];
              const std::int32_t mag = std::abs(old);
              const std::int32_t next =
                  (sum != 0) ? mag + 1 : std::max(0, mag - 1);
              levels[0] = old < 0 ? -next : next;
            }
          }
          BlockSyntax syntax;
          std::string error;
          if (!av2enc::tcq::parity::derive_syntax(levels, geometry,
                                                  eligibility, &syntax,
                                                  &error)) {
            check(false, "entropy derive");
            continue;
          }
          std::vector<std::uint8_t> buffer(1 << 16);
          av2enc::av2::Encoder encoder;
          check(encoder.init_symbol(buffer.data(), buffer.size()) ==
                    av2enc::av2::Status::kOk,
                "encoder init");
          encoder.disable_cdf_update = adaptive ? 0 : 1;
          av2enc::tcq::parity::CdfStore cdfs(0);
          EntropyConfig config;
          config.eligibility = eligibility;
          config.qctx = 0;
          config.txb_skip_ctx = 0;
          config.dc_sign_ctx =
              plane == 0 ? static_cast<std::int32_t>(next_random() % 3) : 0;
          config.disable_cdf_update = encoder.disable_cdf_update != 0;
          check(av2enc::tcq::parity::write_block(encoder, cdfs, syntax,
                                                 geometry, config, &error),
                "entropy write");
          std::size_t payload_bytes = 0;
          check(encoder.exit_symbol(&payload_bytes) ==
                    av2enc::av2::Status::kOk,
                "encoder exit");
          av2enc::av2::Decoder decoder;
          decoder.disable_cdf_update = encoder.disable_cdf_update;
          check(decoder.init_symbol(buffer.data(), buffer.size(),
                                    payload_bytes) ==
                    av2enc::av2::Status::kOk,
                "decoder init");
          av2enc::tcq::parity::CdfStore cdfs2(0);
          BlockSyntax decoded_syntax;
          DecodedBlock decoded;
          check(av2enc::tcq::parity::read_block(decoder, cdfs2, geometry,
                                                config, &decoded_syntax,
                                                &decoded, &error),
                "entropy read");
          check(decoder.exit_symbol() == av2enc::av2::Status::kOk,
                "decoder exit");
          check(decoded_syntax.eob == syntax.eob, "entropy eob");
          check(decoded_syntax.is_hidden == syntax.is_hidden,
                "entropy is_hidden");
          check(decoded_syntax.sum_abs1 == syntax.sum_abs1,
                "entropy sumAbs1");
          for (std::size_t c = 0; c < area; ++c) {
            check(decoded.quant_levels[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])] ==
                      levels[c],
                  "entropy level round-trip");
          }
        }
      }
    }
  }

  // Multi-block session with one CDF store: adaptation must stay in sync.
  {
    const Geometry geometry = geometry_for(1, 0, 0);
    const std::size_t area = 64;
    const std::vector<std::int32_t> scan = scan_of(geometry);
    std::vector<std::vector<std::int32_t>> all_levels;
    std::vector<BlockSyntax> all_syntax;
    Eligibility eligibility;
    eligibility.allow_parity_hiding = true;
    for (int b = 0; b < 32; ++b) {
      std::vector<std::int32_t> levels(area, 1 + (b % 7));
      for (std::size_t i = 0; i < area; ++i) {
        if ((next_random() % 100) < 35) levels[i] = 0;
      }
      std::int32_t num_nz = 0;
      std::int32_t sum = 0;
      ac_stats(geometry, levels, &num_nz, &sum);
      std::int32_t eob = 0;
      for (std::int32_t c = static_cast<std::int32_t>(area) - 1; c >= 0;
           --c) {
        if (levels[static_cast<std::size_t>(c)] != 0) {
          eob = c + 1;
          break;
        }
      }
      if (eob > 4 && num_nz >= 4) {
        const std::int32_t dc_parity = std::abs(levels[0]) & 1;
        if (dc_parity != sum) {
          const std::int32_t old = levels[0];
          const std::int32_t mag = std::abs(old);
          const std::int32_t next =
              (sum != 0) ? mag + 1 : std::max(0, mag - 1);
          levels[0] = old < 0 ? -next : next;
        }
      }
      BlockSyntax syntax;
      std::string error;
      check(av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                               &syntax, &error),
            "multi-block derive");
      all_levels.push_back(levels);
      all_syntax.push_back(syntax);
    }
    std::vector<std::uint8_t> buffer(1 << 16);
    av2enc::av2::Encoder encoder;
    encoder.init_symbol(buffer.data(), buffer.size());
    av2enc::tcq::parity::CdfStore cdfs(0);
    EntropyConfig config;
    config.eligibility = eligibility;
    config.disable_cdf_update = false;
    for (const BlockSyntax& syntax : all_syntax) {
      std::string error;
      check(av2enc::tcq::parity::write_block(encoder, cdfs, syntax, geometry,
                                             config, &error),
            "multi-block write");
    }
    std::size_t payload_bytes = 0;
    check(encoder.exit_symbol(&payload_bytes) == av2enc::av2::Status::kOk,
          "multi-block encoder exit");
    av2enc::av2::Decoder decoder;
    decoder.disable_cdf_update = 0;
    check(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
              av2enc::av2::Status::kOk,
          "multi-block decoder init");
    av2enc::tcq::parity::CdfStore cdfs2(0);
    for (std::size_t b = 0; b < all_syntax.size(); ++b) {
      BlockSyntax syntax;
      DecodedBlock decoded;
      std::string error;
      check(av2enc::tcq::parity::read_block(decoder, cdfs2, geometry, config,
                                            &syntax, &decoded, &error),
            "multi-block read");
      check(syntax.eob == all_syntax[b].eob, "multi-block eob");
      for (std::size_t c = 0; c < area; ++c) {
        check(decoded.quant_levels[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])] ==
                  all_levels[b][c],
              "multi-block level round-trip");
      }
    }
    check(decoder.exit_symbol() == av2enc::av2::Status::kOk,
          "multi-block decoder exit");
  }

  // read_quant escape bit round-trip.
  for (std::int32_t predictor = 0; predictor <= 63; ++predictor) {
    for (std::int32_t x = 0; x <= 300; ++x) {
      std::vector<std::uint8_t> buffer(1024);
      av2enc::av2::Encoder encoder;
      encoder.init_symbol(buffer.data(), buffer.size());
      std::string error;
      check(av2enc::tcq::parity::write_escape_bits(encoder, x, predictor,
                                                   &error),
            "escape writer");
      std::size_t payload_bytes = 0;
      check(encoder.exit_symbol(&payload_bytes) ==
                av2enc::av2::Status::kOk,
            "escape encoder exit");
      av2enc::av2::Decoder decoder;
      check(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
                av2enc::av2::Status::kOk,
            "escape decoder init");
      std::int32_t decoded = -1;
      check(av2enc::tcq::parity::read_escape_bits(decoder, predictor, &decoded,
                                                  &error),
            "escape reader");
      check(decoded == x, "escape value round-trip");
      check(decoder.exit_symbol() == av2enc::av2::Status::kOk,
            "escape decoder exit");
    }
  }
}

// ---------------------------------------------------------------------------
// Quantizer
// ---------------------------------------------------------------------------

double distortion_of(const std::vector<std::int32_t>& levels,
                     const std::vector<double>& original, double step,
                     std::span<const double> steps = {}) {
  double sum = 0.0;
  for (std::size_t i = 0; i < levels.size(); ++i) {
    const double s = steps.empty() ? step : steps[i];
    const double err = original[i] - static_cast<double>(levels[i]) * s;
    sum += err * err;
  }
  return sum;
}

void test_quantizer() {
  rng_state = 0x5555aaaau;
  for (std::int32_t tx = 0; tx < 25; ++tx) {
    for (std::int32_t cls = 0; cls < 3; ++cls) {
      for (std::int32_t plane = 0; plane < 3; ++plane) {
        const Geometry geometry = geometry_for(tx, cls, plane);
        const std::size_t area =
            static_cast<std::size_t>(geometry.w * geometry.h);
        const std::vector<std::int32_t> scan = scan_of(geometry);
        for (int trial = 0; trial < 6; ++trial) {
          std::vector<int32_t> seed(area, 0);
          std::vector<double> original(area, 0.0);
          for (std::size_t i = 0; i < area; ++i) {
            if ((next_random() % 100) < 50) {
              seed[i] = random_int(-12, 12);
              if (seed[i] == 0) seed[i] = 1;
            }
            original[i] = static_cast<double>(seed[i]) * 3.0 +
                          (random_int(-20, 20) / 20.0);
          }
          for (int mode = 0; mode < 2; ++mode) {
            QuantizeConfig config;
            config.eligibility = standard_eligibility();
            config.mode = mode == 0 ? QuantizeMode::kMinDistortion
                                    : QuantizeMode::kRateDistortion;
            config.quant_step = 3.0;
            config.lambda = 4.0;
            config.rate.qctx = 0;
            config.rate.txb_skip_ctx = 0;
            config.rate.dc_sign_ctx = 0;
            config.exhaustive_max_area = 64;
            QuantizeResult result;
            std::string error;
            check(av2enc::tcq::parity::quantize(original, seed, geometry,
                                                config, &result, &error),
                  "quantize");
            if (error.size() != 0) {
              std::fprintf(stderr, "  %s\n", error.c_str());
            }
            // The output must decode to itself.
            DecodedBlock decoded;
            check(av2enc::tcq::parity::decode_syntax(result.syntax, geometry,
                                                     &decoded, &error),
                  "quantize output decodes");
            for (std::size_t c = 0; c < area; ++c) {
              check(decoded.quant_levels[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])] ==
                        result.levels[c],
                    "quantize levels round-trip");
            }
            const std::int32_t num_nz = count_nonzero(result.levels);
            if (result.hidden) {
              check((std::abs(result.levels[0]) & 1) ==
                        result.syntax.sum_abs1,
                    "quantize hidden parity");
              check(num_nz >= 4, "hidden block has at least 4 AC non-zeros");
            }
            if (!result.hidable) {
              check(!result.hidden, "ineligible block never hides");
            }
            // Determinism.
            QuantizeResult again;
            check(av2enc::tcq::parity::quantize(original, seed, geometry,
                                                config, &again, &error),
                  "quantize repeat");
            check(again.levels == result.levels, "quantize deterministic");
            check(again.eob == result.eob, "quantize eob deterministic");
            check_near(again.cost, result.cost, 0.0,
                       "quantize cost deterministic");
          }
        }
      }
    }
  }

  // Exhaustive min-distortion oracle on small blocks.
  rng_state = 0x0badf00du;
  for (int trial = 0; trial < 200; ++trial) {
    Geometry geometry = geometry_for(trial % 2 == 0 ? 0 : 1, 0, 0);
    const std::size_t area =
        static_cast<std::size_t>(geometry.w * geometry.h);
    const std::vector<std::int32_t> scan = scan_of(geometry);
    std::vector<std::int32_t> seed(area, 0);
    std::vector<double> original(area, 0.0);
    for (std::size_t i = 0; i < area; ++i) {
      if ((next_random() % 100) < 55) {
        const std::int32_t mag = random_int(1, 9);
        seed[i] = (next_random() & 1U) != 0U ? -mag : mag;
      }
      original[i] = static_cast<double>(seed[i]) * 2.0 +
                    (random_int(-15, 15) / 10.0);
    }
    QuantizeConfig config;
    config.eligibility = standard_eligibility();
    config.mode = QuantizeMode::kMinDistortion;
    config.quant_step = 2.0;
    config.lambda = 0.0;
    config.exhaustive_max_area = 64;
    std::vector<double> steps;
    if (trial % 3 == 0) {
      steps.resize(area);
      for (std::size_t i = 0; i < area; ++i) {
        steps[i] = 1.5 + static_cast<double>((i * 7) % 5);
      }
      config.dequant_step = steps;
    }
    QuantizeResult result;
    std::string error;
    check(av2enc::tcq::parity::quantize(original, seed, geometry, config,
                                        &result, &error),
          "oracle quantize");

    bool baseline_valid = false;
    {
      BlockSyntax syntax;
      std::string local;
      baseline_valid = av2enc::tcq::parity::derive_syntax(
          seed, geometry, config.eligibility, &syntax, &local);
    }
    double best = baseline_valid
                      ? distortion_of(seed, original, config.quant_step,
                                      config.dequant_step)
                      : std::numeric_limits<double>::infinity();
    if (!baseline_valid) {
      for (std::size_t c = 0; c < area; ++c) {
        for (std::int32_t delta : {1, -1}) {
          const std::int32_t old = seed[c];
          if (c != 0 && old == 0) continue;
          const std::int32_t mag = std::abs(old);
          std::int32_t next = mag + delta;
          if (next < 0) continue;
          std::vector<std::int32_t> candidate = seed;
          candidate[c] = old < 0 ? -next : next;
          if (c == 0 && old == 0) {
            // Both signs are legal for a zero DC; the library evaluates both.
            candidate[0] = 1;
            BlockSyntax syntax;
            std::string local;
            if (av2enc::tcq::parity::derive_syntax(
                    candidate, geometry, config.eligibility, &syntax, &local)) {
              best = std::min(best, distortion_of(candidate, original,
                                                  config.quant_step,
                                                  config.dequant_step));
            }
            candidate[0] = -1;
            if (av2enc::tcq::parity::derive_syntax(
                    candidate, geometry, config.eligibility, &syntax, &local)) {
              best = std::min(best, distortion_of(candidate, original,
                                                  config.quant_step,
                                                  config.dequant_step));
            }
            continue;
          }
          BlockSyntax syntax;
          std::string local;
          if (!av2enc::tcq::parity::derive_syntax(
                  candidate, geometry, config.eligibility, &syntax, &local)) {
            continue;
          }
          best = std::min(best, distortion_of(candidate, original,
                                              config.quant_step,
                                              config.dequant_step));
        }
      }
    }
    check_near(result.distortion, best, 1e-12,
               "min-distortion matches the exhaustive oracle");
  }

  // TCQ exclusion: allow_tcq means no hidden DC and no level change.
  {
    const Geometry geometry = geometry_for(1, 0, 0);
    std::vector<std::int32_t> seed(64, 0);
    std::vector<double> original(64, 0.0);
    seed[0] = 5;
    seed[1] = 1;
    seed[2] = 2;
    seed[3] = 3;
    seed[4] = 2;
    for (std::size_t i = 0; i < seed.size(); ++i) {
      original[i] = static_cast<double>(seed[i]) * 3.0;
    }
    QuantizeConfig config;
    config.eligibility = standard_eligibility();
    config.eligibility.allow_tcq = true;
    config.quant_step = 3.0;
    QuantizeResult result;
    std::string error;
    check(av2enc::tcq::parity::quantize(original, seed, geometry, config,
                                        &result, &error),
          "TCQ-excluded quantize");
    check(!result.hidden, "TCQ excludes parity hiding");
    check(result.levels == seed, "TCQ-excluded levels are unchanged");
  }

  // EOB shrink via an AC adjustment to zero.
  {
    const Geometry geometry = geometry_for(1, 0, 0);
    std::vector<std::int32_t> seed(64, 0);
    std::vector<double> original(64, 0.0);
    seed[0] = 5;
    seed[1] = 1;
    seed[2] = 2;
    seed[3] = 3;
    seed[4] = 2;
    for (std::size_t i = 0; i < seed.size(); ++i) {
      original[i] = static_cast<double>(seed[i]) * 3.0;
    }
    // Make the last AC coefficient unattractive so an AC->0 adjustment can
    // drop numNz below PHTHRESH and turn hiding off.
    original[4] = 0.0;
    QuantizeConfig config;
    config.eligibility = standard_eligibility();
    config.mode = QuantizeMode::kRateDistortion;
    config.quant_step = 3.0;
    config.lambda = 10.0;
    QuantizeResult result;
    std::string error;
    check(av2enc::tcq::parity::quantize(original, seed, geometry, config,
                                        &result, &error),
          "EOB-shrink quantize");
    check(result.eob <= 5, "EOB shrink stays within the block");
    check(result.syntax.eob == result.eob, "result eob matches syntax");
  }
}

// ---------------------------------------------------------------------------
// Goldens
// ---------------------------------------------------------------------------

void test_goldens() {
  for (const auto& golden : av2enc::tcq::parity::golden::blocks()) {
    const Geometry geometry = geometry_for(golden.tx_size, golden.tx_class,
                                           golden.plane);
    check(geometry.w * geometry.h == golden.area, "golden area");
    std::vector<std::int32_t> levels(golden.levels.begin(),
                                     golden.levels.begin() + golden.area);
    Eligibility eligibility;
    eligibility.allow_parity_hiding = true;
    BlockSyntax syntax;
    std::string error;
    check(av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                             &syntax, &error),
          "golden derive");
    check(syntax.eob == golden.eob, "golden eob");
    check(syntax.is_hidden == (golden.is_hidden != 0), "golden is_hidden");
    check(syntax.sum_abs1 == golden.sum_abs1, "golden sumAbs1");
    check(syntax.num_nz == golden.num_nz, "golden num_nz");
    check(syntax.dc_ctx == golden.dc_ctx, "golden dc context");
    check(static_cast<std::int32_t>(syntax.eob_kind) == golden.eob_kind,
          "golden eob CDF");
    check(syntax.eob_ctx == golden.eob_ctx, "golden eob ctx");
    check(syntax.eob_symbol == golden.eob_symbol, "golden eob symbol");
    check(syntax.eob_symbols == golden.eob_symbols, "golden eob symbols");
    check(syntax.eob_pt == golden.eob_pt, "golden eob pt");
    check(syntax.eob_extra_symbol == golden.eob_extra_symbol,
          "golden eob extra");
    check(syntax.positions.size() == golden.positions.size(),
          "golden position count");
    for (std::size_t i = 0;
         i < syntax.positions.size() && i < golden.positions.size(); ++i) {
      const auto& p = syntax.positions[i];
      const auto& g = golden.positions[i];
      check(p.scan_index == g.scan_index, "golden scan index");
      check(p.pos == g.pos, "golden pos");
      check(p.level == g.level, "golden level");
      check(static_cast<std::int32_t>(p.base_kind) == g.base_kind,
            "golden base CDF");
      check(p.base_ctx == g.base_ctx, "golden base ctx");
      check(p.base_symbol == g.base_symbol, "golden base symbol");
      check(p.base_offset == g.base_offset, "golden base offset");
      check((p.br_present ? 1 : 0) == g.br_present, "golden br present");
      check((p.br_present ? static_cast<std::int32_t>(p.br_kind) : -1) ==
                g.br_kind,
            "golden br CDF");
      check(p.br_ctx == g.br_ctx, "golden br ctx");
      check(p.br_symbol == g.br_symbol, "golden br symbol");
      check((p.escape_present ? p.escape_x : -1) == g.escape_x,
            "golden escape x");
      check(p.escape_bits == g.escape_bits, "golden escape bits");
      check((p.sign_present ? p.sign : -1) == g.sign, "golden sign");
      check((p.dc_style_sign ? 1 : 0) == g.dc_style, "golden sign style");
    }
    RateConfig rate;
    rate.qctx = 0;
    rate.txb_skip_ctx = 0;
    rate.dc_sign_ctx = 0;
    check_near(av2enc::tcq::parity::block_rate_bits(syntax, geometry, rate),
               golden.rate_bits, 1e-12, "golden rate bits");

    DecodedBlock decoded;
    check(av2enc::tcq::parity::decode_syntax(syntax, geometry, &decoded,
                                             &error),
          "golden decode");
    const std::vector<std::int32_t> scan = scan_of(geometry);
    for (std::int32_t c = 0; c < golden.area; ++c) {
      check(decoded.quant_levels[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])] ==
                levels[static_cast<std::size_t>(c)],
            "golden decoded level");
    }

    std::vector<std::uint8_t> buffer(1 << 16);
    av2enc::av2::Encoder encoder;
    encoder.init_symbol(buffer.data(), buffer.size());
    encoder.disable_cdf_update = 1;
    av2enc::tcq::parity::CdfStore cdfs(0);
    EntropyConfig config;
    config.eligibility = eligibility;
    config.disable_cdf_update = true;
    check(av2enc::tcq::parity::write_block(encoder, cdfs, syntax, geometry,
                                           config, &error),
          "golden entropy write");
    std::size_t payload_bytes = 0;
    check(encoder.exit_symbol(&payload_bytes) == av2enc::av2::Status::kOk,
          "golden entropy exit");
    check(payload_bytes == golden.payload.size(), "golden payload size");
    for (std::size_t i = 0; i < payload_bytes && i < golden.payload.size();
         ++i) {
      check(buffer[i] == golden.payload[i], "golden payload byte");
    }

    av2enc::av2::Decoder decoder;
    decoder.disable_cdf_update = 1;
    check(decoder.init_symbol(buffer.data(), buffer.size(), payload_bytes) ==
              av2enc::av2::Status::kOk,
          "golden decoder init");
    av2enc::tcq::parity::CdfStore cdfs2(0);
    BlockSyntax decoded_syntax;
    DecodedBlock decoded2;
    check(av2enc::tcq::parity::read_block(decoder, cdfs2, geometry, config,
                                          &decoded_syntax, &decoded2, &error),
          "golden entropy read");
    check(decoder.exit_symbol() == av2enc::av2::Status::kOk,
          "golden decoder exit");
    check(decoded_syntax.eob == syntax.eob, "golden entropy eob");
    check(decoded_syntax.is_hidden == syntax.is_hidden,
          "golden entropy hidden");
    for (std::int32_t c = 0; c < golden.area; ++c) {
      check(decoded2.quant_levels[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])] ==
                levels[static_cast<std::size_t>(c)],
            "golden entropy level");
    }
  }
}

// ---------------------------------------------------------------------------
// Coefficient-payload gate (separate writer and reader processes)
// ---------------------------------------------------------------------------

struct GateBlock {
  std::int32_t tx_size;
  std::int32_t tx_class;
  std::int32_t plane;
  Eligibility eligibility;
  std::int32_t txb_skip_ctx;
  std::int32_t dc_sign_ctx;
  std::vector<std::int32_t> levels;
  BlockSyntax syntax;
};

std::vector<GateBlock> gate_blocks() {
  std::vector<GateBlock> blocks;
  rng_state = 0xfeedfaceu;
  for (std::int32_t tx = 0; tx < 25; ++tx) {
    for (std::int32_t plane = 0; plane < 3; ++plane) {
      const Geometry geometry = geometry_for(tx, 0, plane);
      const std::size_t area =
          static_cast<std::size_t>(geometry.w * geometry.h);
      GateBlock block;
      block.tx_size = tx;
      block.tx_class = 0;
      block.plane = plane;
      block.eligibility.allow_parity_hiding = true;
      block.txb_skip_ctx = 0;
      block.dc_sign_ctx =
          plane == 0 ? static_cast<std::int32_t>((tx + plane) % 3) : 0;
      block.levels.assign(area, 0);
      for (std::size_t i = 0; i < area; ++i) {
        if ((next_random() % 100) < 40) {
          const std::int32_t mag = 1 + static_cast<std::int32_t>(
                                           next_random() % 12);
          block.levels[i] = (next_random() & 1U) != 0U ? -mag : mag;
        }
      }
      std::int32_t num_nz = 0;
      std::int32_t sum = 0;
      ac_stats(geometry, block.levels, &num_nz, &sum);
      std::int32_t eob = 0;
      for (std::int32_t c = static_cast<std::int32_t>(area) - 1; c >= 0;
           --c) {
        if (block.levels[static_cast<std::size_t>(c)] != 0) {
          eob = c + 1;
          break;
        }
      }
      if (av2enc::tcq::parity::parity_hiding_eligible(geometry,
                                                      block.eligibility) &&
          eob > 4 && num_nz >= 4) {
        const std::int32_t dc_parity = std::abs(block.levels[0]) & 1;
        if (dc_parity != sum) {
          const std::int32_t old = block.levels[0];
          const std::int32_t mag = std::abs(old);
          const std::int32_t next =
              (sum != 0) ? mag + 1 : std::max(0, mag - 1);
          block.levels[0] = old < 0 ? -next : next;
        }
      }
      std::string error;
      if (!av2enc::tcq::parity::derive_syntax(block.levels, geometry,
                                              block.eligibility,
                                              &block.syntax, &error)) {
        std::fprintf(stderr, "gate derive: %s\n", error.c_str());
        std::abort();
      }
      blocks.push_back(std::move(block));
    }
  }
  return blocks;
}

void write_u32(std::vector<std::uint8_t>* out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out->push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xFFU));
  }
}

void write_i32(std::vector<std::uint8_t>* out, std::int32_t value) {
  write_u32(out, static_cast<std::uint32_t>(value));
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& data, std::size_t* at) {
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    value |= static_cast<std::uint32_t>(data[(*at)++]) << (8 * i);
  }
  return value;
}

std::int32_t read_i32(const std::vector<std::uint8_t>& data, std::size_t* at) {
  return static_cast<std::int32_t>(read_u32(data, at));
}

bool write_gate(const std::string& path) {
  const std::vector<GateBlock> blocks = gate_blocks();
  std::vector<std::uint8_t> entropy_buffer(1 << 20);
  av2enc::av2::Encoder encoder;
  encoder.init_symbol(entropy_buffer.data(), entropy_buffer.size());
  encoder.disable_cdf_update = 0;
  av2enc::tcq::parity::CdfStore cdfs(0);
  for (const GateBlock& block : blocks) {
    const Geometry geometry =
        geometry_for(block.tx_size, block.tx_class, block.plane);
    EntropyConfig config;
    config.eligibility = block.eligibility;
    config.qctx = 0;
    config.txb_skip_ctx = block.txb_skip_ctx;
    config.dc_sign_ctx = block.dc_sign_ctx;
    config.disable_cdf_update = false;
    std::string error;
    if (!av2enc::tcq::parity::write_block(encoder, cdfs, block.syntax,
                                          geometry, config, &error)) {
      std::fprintf(stderr, "gate write: %s\n", error.c_str());
      return false;
    }
  }
  std::size_t payload_bytes = 0;
  if (encoder.exit_symbol(&payload_bytes) != av2enc::av2::Status::kOk) {
    std::fprintf(stderr, "gate encoder exit failed\n");
    return false;
  }

  std::vector<std::uint8_t> out;
  out.insert(out.end(), {'P', 'H', 'G', '1'});
  write_u32(&out, 0);  // qctx
  write_u32(&out, static_cast<std::uint32_t>(blocks.size()));
  write_u32(&out, static_cast<std::uint32_t>(payload_bytes));
  out.insert(out.end(), entropy_buffer.begin(),
             entropy_buffer.begin() +
                 static_cast<std::ptrdiff_t>(payload_bytes));
  for (const GateBlock& block : blocks) {
    write_i32(&out, block.tx_size);
    write_i32(&out, block.tx_class);
    write_i32(&out, block.plane);
    write_u32(&out, (block.eligibility.allow_parity_hiding ? 1U : 0U) |
                        (block.eligibility.lossless ? 2U : 0U) |
                        (block.eligibility.allow_tcq ? 4U : 0U) |
                        (block.eligibility.tx_type_is_idtx ? 8U : 0U) |
                        (block.eligibility.use_fsc ? 16U : 0U) |
                        (block.eligibility.is_inter ? 32U : 0U));
    write_i32(&out, block.txb_skip_ctx);
    write_i32(&out, block.dc_sign_ctx);
    write_u32(&out, static_cast<std::uint32_t>(block.levels.size()));
    for (std::int32_t level : block.levels) write_i32(&out, level);
  }
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return false;
  }
  const std::size_t written =
      std::fwrite(out.data(), 1, out.size(), file);
  std::fclose(file);
  if (written != out.size()) {
    std::fprintf(stderr, "short write to %s\n", path.c_str());
    return false;
  }
  std::printf("wrote %zu blocks, %zu payload bytes to %s\n", blocks.size(),
              payload_bytes, path.c_str());
  return true;
}

bool read_gate(const std::string& path) {
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return false;
  }
  std::vector<std::uint8_t> data;
  std::uint8_t chunk[4096];
  std::size_t got = 0;
  while ((got = std::fread(chunk, 1, sizeof(chunk), file)) != 0) {
    data.insert(data.end(), chunk, chunk + got);
  }
  std::fclose(file);
  if (data.size() < 16 || std::memcmp(data.data(), "PHG1", 4) != 0) {
    std::fprintf(stderr, "bad gate file\n");
    return false;
  }
  std::size_t at = 4;
  const std::uint32_t qctx = read_u32(data, &at);
  const std::uint32_t count = read_u32(data, &at);
  const std::uint32_t payload_bytes = read_u32(data, &at);
  if (at + payload_bytes > data.size()) {
    std::fprintf(stderr, "truncated gate payload\n");
    return false;
  }
  const std::uint8_t* payload = data.data() + at;
  at += payload_bytes;

  std::vector<GateBlock> blocks;
  for (std::uint32_t i = 0; i < count; ++i) {
    GateBlock block;
    block.tx_size = read_i32(data, &at);
    block.tx_class = read_i32(data, &at);
    block.plane = read_i32(data, &at);
    const std::uint32_t flags = read_u32(data, &at);
    block.eligibility.allow_parity_hiding = (flags & 1U) != 0U;
    block.eligibility.lossless = (flags & 2U) != 0U;
    block.eligibility.allow_tcq = (flags & 4U) != 0U;
    block.eligibility.tx_type_is_idtx = (flags & 8U) != 0U;
    block.eligibility.use_fsc = (flags & 16U) != 0U;
    block.eligibility.is_inter = (flags & 32U) != 0U;
    block.txb_skip_ctx = read_i32(data, &at);
    block.dc_sign_ctx = read_i32(data, &at);
    const std::uint32_t area = read_u32(data, &at);
    block.levels.resize(area);
    for (std::uint32_t j = 0; j < area; ++j) {
      block.levels[j] = read_i32(data, &at);
    }
    blocks.push_back(std::move(block));
  }

  av2enc::av2::Decoder decoder;
  decoder.disable_cdf_update = 0;
  if (decoder.init_symbol(payload, payload_bytes, payload_bytes) !=
      av2enc::av2::Status::kOk) {
    std::fprintf(stderr, "gate decoder init failed\n");
    return false;
  }
  av2enc::tcq::parity::CdfStore cdfs(static_cast<std::int32_t>(qctx));
  for (const GateBlock& block : blocks) {
    const Geometry geometry =
        geometry_for(block.tx_size, block.tx_class, block.plane);
    EntropyConfig config;
    config.eligibility = block.eligibility;
    config.qctx = static_cast<std::int32_t>(qctx);
    config.txb_skip_ctx = block.txb_skip_ctx;
    config.dc_sign_ctx = block.dc_sign_ctx;
    config.disable_cdf_update = false;
    BlockSyntax syntax;
    DecodedBlock decoded;
    std::string error;
    if (!av2enc::tcq::parity::read_block(decoder, cdfs, geometry, config,
                                         &syntax, &decoded, &error)) {
      std::fprintf(stderr, "gate read: %s\n", error.c_str());
      return false;
    }
    const std::vector<std::int32_t> scan = scan_of(geometry);
    for (std::size_t c = 0; c < block.levels.size(); ++c) {
      if (decoded.quant_levels[static_cast<std::size_t>(scan[c])] !=
          block.levels[c]) {
        std::fprintf(stderr, "gate mismatch: tx=%d plane=%d c=%zu\n",
                     block.tx_size, block.plane, c);
        return false;
      }
    }
  }
  if (decoder.exit_symbol() != av2enc::av2::Status::kOk) {
    std::fprintf(stderr, "gate decoder exit failed\n");
    return false;
  }
  std::printf("gate: %u blocks decoded and matched\n", count);
  return true;
}

// ---------------------------------------------------------------------------
// Optional golden printer
// ---------------------------------------------------------------------------

void print_goldens() {
  for (const auto& golden : av2enc::tcq::parity::golden::blocks()) {
    const Geometry geometry = geometry_for(golden.tx_size, golden.tx_class,
                                           golden.plane);
    std::vector<std::int32_t> levels(golden.levels.begin(),
                                     golden.levels.begin() + golden.area);
    Eligibility eligibility;
    eligibility.allow_parity_hiding = true;
    BlockSyntax syntax;
    std::string error;
    if (!av2enc::tcq::parity::derive_syntax(levels, geometry, eligibility,
                                            &syntax, &error)) {
      std::printf("// %s: derive failed: %s\n", golden.name, error.c_str());
      continue;
    }
    RateConfig rate;
    rate.qctx = 0;
    rate.txb_skip_ctx = 0;
    rate.dc_sign_ctx = 0;
    const double bits =
        av2enc::tcq::parity::block_rate_bits(syntax, geometry, rate);
    std::vector<std::uint8_t> buffer(1 << 16);
    av2enc::av2::Encoder encoder;
    encoder.init_symbol(buffer.data(), buffer.size());
    encoder.disable_cdf_update = 1;
    av2enc::tcq::parity::CdfStore cdfs(0);
    EntropyConfig config;
    config.eligibility = eligibility;
    config.disable_cdf_update = true;
    if (!av2enc::tcq::parity::write_block(encoder, cdfs, syntax, geometry,
                                          config, &error)) {
      std::printf("// %s: write failed: %s\n", golden.name, error.c_str());
      continue;
    }
    std::size_t payload_bytes = 0;
    encoder.exit_symbol(&payload_bytes);
    std::printf(
        "// %s: area=%d eob=%d hidden=%d sum=%d num=%d dc_ctx=%d "
        "eob_kind=%s eob_ctx=%d eob_symbol=%d/%d eob_pt=%d eob_extra=%d "
        "bits=%.17g bytes=%zu\n",
        golden.name, golden.area, syntax.eob, syntax.is_hidden ? 1 : 0,
        syntax.sum_abs1, syntax.num_nz, syntax.dc_ctx,
        av2enc::tcq::parity::cdf_kind_name(syntax.eob_kind), syntax.eob_ctx,
        syntax.eob_symbol, syntax.eob_symbols, syntax.eob_pt,
        syntax.eob_extra_symbol, bits, payload_bytes);
    std::printf("  levels={");
    for (std::int32_t i = 0; i < golden.area; ++i) {
      std::printf("%s%d", i ? "," : "", levels[static_cast<std::size_t>(i)]);
    }
    std::printf("};\n  positions:");
    for (const auto& p : syntax.positions) {
      std::printf(
          " {c%d,p%d,l%d,k%d,c%d,s%d,o%d,br%d,k%d,c%d,s%d,x%d,b%d,s%d,d%d}",
          p.scan_index, p.pos, p.level, static_cast<std::int32_t>(p.base_kind),
          p.base_ctx, p.base_symbol, p.base_offset, p.br_present ? 1 : 0,
          p.br_present ? static_cast<std::int32_t>(p.br_kind) : -1, p.br_ctx,
          p.br_symbol, p.escape_present ? p.escape_x : -1, p.escape_bits,
          p.sign_present ? p.sign : -1, p.dc_style_sign ? 1 : 0);
    }
    std::printf("\n  payload:");
    for (std::size_t i = 0; i < payload_bytes; ++i) {
      std::printf(" 0x%02x", buffer[i]);
    }
    std::printf("\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--print-goldens") == 0) {
      print_goldens();
      return 0;
    }
    if (std::strcmp(argv[i], "--write-gate") == 0 && i + 1 < argc) {
      return write_gate(argv[i + 1]) ? 0 : 1;
    }
    if (std::strcmp(argv[i], "--read-gate") == 0 && i + 1 < argc) {
      return read_gate(argv[i + 1]) ? 0 : 1;
    }
  }
  test_geometry();
  test_eligibility();
  test_parity_hiding_constants();
  test_context_axes();
  test_tcq_flag_interop();
  test_sign_contexts();
  test_syntax_roundtrip();
  test_parities_and_escapes();
  test_entropy();
  test_quantizer();
  test_goldens();
  std::printf("parity-hiding tests: %d checks, %d failures\n", checks,
              failures);
  return failures == 0 ? 0 : 1;
}
