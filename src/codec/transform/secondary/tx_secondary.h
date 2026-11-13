// AV2 intra-inter secondary transform (IST), CPU reference.
//
// Normative reference: AV2 v1.0.0 §5.20.8.2 transform_type(), §6.19.8.2
// sec_tx_type / most_probable_stx_set semantics, §7.15.3 Secondary transform
// process, §9.7 secondary transform tables, §5.4.8 sequence header bits.
//
// The secondary transform is a luma-only (plane 0), lossy-tool post-dequant
// step: the decoder dequantizes the coded levels, applies the IST matrix to the
// first n scan-order coefficients, and only then runs the primary 2D inverse
// transform.  This component owns:
//
//   * the §9.7 4x4 and 8x8 kernels and Stx_Scan_Map (generated header);
//   * the §7.15.3 scan-in / scan-out coefficient remapping;
//   * kernel selection from YMode + most_probable_stx_set + PlaneTxType;
//   * the §5.20.8.2 / §5.4.8 legality rules, including EOB limits;
//   * §9.3 Sec_Tx_Type / Most_Probable_Stx_Set CDF accessors;
//   * the two §5.4.8 header bits.
//
// The encoder-side `forward_secondary_coeffs_kernel()` is an analysis projection (the
// transpose of the normative inverse matrix with the same shift 7).  AV2 v1.0.0
// defines only the decoder inverse, so that function is explicitly not a
// normative bitstream process.
//
// This file is self-contained: it does not edit or depend on the M1 writer.
// See README.md for the exact integration call sites.
#pragma once

#include <cstddef>
#include <cstdint>

#include "core/bitstream/bitwriter.h"
#include "codec/transform/variants/tx_variants.h"
#include "codec/transform/secondary/tx_secondary_tables.h"

namespace av2enc::txsec {

using av2enc::txv::TxParams;
using av2enc::txv::TxType;

// §5.4.8 sequence_transform_quant_entropy_config() bits.  AV2 v1.0.0 has no
// single `enable_secondary_transform` syntax element: it codes two bits,
// enable_intra_ist and enable_inter_ist (in that order).
struct SecondaryTransformConfig {
  bool enable_intra_ist = false;
  bool enable_inter_ist = false;
};

// Convenience predicate for "IST can be used at all".
inline bool enable_secondary_transform(const SecondaryTransformConfig& c) noexcept {
  return c.enable_intra_ist || c.enable_inter_ist;
}

// Writes the two §5.4.8 bits in syntax order.  Insert exactly where the
// sequence header currently writes `enable_intra_ist` / `enable_inter_ist`.
void write_secondary_transform_header_bits(BitWriter& writer,
                                           const SecondaryTransformConfig& c);

// ---------------------------------------------------------------------------
// Sizes, limits and mode mapping
// ---------------------------------------------------------------------------

struct SecondarySizeInfo {
  int tx_size = 0;      // AV2 TX_SIZE 0..24
  int coded_w = 0;
  int coded_h = 0;
  int w = 0;            // Min(32, Tx_Width)
  int h = 0;            // Min(32, Tx_Height)
  int bwl = 0;          // Min(5, Tx_Width_Log2)
  bool large = false;   // w >= 8 && h >= 8
  int n = 0;            // kernel rows (8, 20 or 32)
  int n2 = 0;           // output coefficients (16 or 48)
  int scan_w = 0;       // 4 or 8
  int kernel_count = 0; // 14 (4x4) or 11 (8x8)
  int eob_limit = 0;    // §5.20.8.2 eobLim == n
};

// Derives the §7.15.3 block facts.  Returns false for an unsupported TX size.
bool secondary_size_info(const TxParams& params, TxType plane_tx_type,
                         SecondarySizeInfo* out);

// §5.20.8.2 eobLim (8, 20 or 32).
int secondary_eob_limit(const TxParams& params, TxType plane_tx_type);

// YMode 1..8 are directional (V..D67); DC/SMOOTH/PAETH are not.
bool is_directional_mode(int y_mode) noexcept;

// §5.20.8.2 mode derivation used by the IST kernel selection:
//   pAngle = Mode_To_Angle[mode] + AngleDeltaY*ANGLE_STEP
//            + Mrl_Index_To_Delta[MrlIndex]
//   mode = wide_angle_mapping(mode, Tx_Width, Tx_Height, pAngle)
// Returns -1 for an invalid mode or PAETH_PRED (IST is not allowed there).
// `coded_w`/`coded_h` are Tx_Width/Tx_Height (not the adjusted size).
int map_y_mode(int y_mode, int angle_delta, int mrl_index, int coded_w,
               int coded_h) noexcept;

// Number of most_probable_stx_set symbols: 4 for an ADST_ADST block with both
// coded dimensions >= 8, otherwise 7 (§7.15.3 / §9.3).
int most_probable_stx_set_count(TxType plane_tx_type, int coded_w,
                                int coded_h) noexcept;

// §7.15.3 kernel index.  `mapped_mode` is map_y_mode() output (0..11).
// For inter blocks kernel is always 0 and `most_probable_stx_set` is ignored
// (the syntax does not code it).  Returns false for an out-of-range set or a
// kernel index outside the selected 4x4/8x8 table.
bool kernel_for_syntax(int mapped_mode, TxType plane_tx_type,
                       int most_probable_stx_set, bool is_inter, int coded_w,
                       int coded_h, int* kernel) noexcept;

// §7.15.3 transpose = H_PRED / D157_PRED / D67_PRED / SMOOTH_H_PRED.
bool transpose_for_mode(int mapped_mode) noexcept;

// ---------------------------------------------------------------------------
// Syntax legality (§5.20.8.2 + §5.4.8)
// ---------------------------------------------------------------------------

struct SecondarySyntax {
  TxType plane_tx_type = TxType::kDctDct;  // compute_tx_type() for plane 0
  int coded_w = 0;                         // Tx_Width[txSz]
  int coded_h = 0;                         // Tx_Height[txSz]
  int y_mode = 0;                          // AV2 YMode 0..12
  int angle_delta = 0;                     // AngleDeltaY -3..3
  int mrl_index = 0;                       // MrlIndex 0..3
  int most_probable_stx_set = 0;           // 0..6, or 0..3 for large ADST
  int sec_tx_type = 0;                     // 0 = none, 1..3
  bool is_inter = false;
  bool lossless = false;
  int bit_depth = 8;
};

// True when §5.20.8.2 would code sec_tx_type for this block (regardless of the
// value, including 0).  `eob` is the coded scan-position EOB.
bool secondary_syntax_applicable(int plane, int eob,
                                 const SecondarySyntax& syntax,
                                 const SecondaryTransformConfig& config);

// Full validation for a nonzero sec_tx_type, including the kernel, set range,
// bit depth and EOB limit.
bool secondary_syntax_valid(int plane, int eob, const SecondarySyntax& syntax,
                            const SecondaryTransformConfig& config);

// One legal (sec_tx_type, most_probable_stx_set) pair and its resolved kernel.
struct SecondaryCandidate {
  int sec_tx_type = 0;
  int most_probable_stx_set = 0;
  int kernel = 0;
  bool transpose = false;
};

// Enumerates the legal nonzero sec_tx_type candidates for a block.  Intra
// returns 3 * most_probable_stx_set_count() entries (all kernel sets);
// inter has no most_probable_stx_set symbol and returns 3 entries (kernel 0).
// Returns 0 when the syntax rules disallow IST or the arguments are invalid.
int enumerate_secondary_candidates(const SecondarySyntax& syntax,
                                   const SecondaryTransformConfig& config,
                                   int eob, SecondaryCandidate* out,
                                   int cap) noexcept;

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

enum class SecondaryStatus : std::uint8_t {
  kNotApplicable = 0,  // no sec_tx_type, or the syntax rules disallow IST
  kApplied = 1,
  kInvalid = 2,        // malformed kernel / sec_tx_type / arguments
};

// Normative §7.15.3 inverse on a dequantized w*h raster coefficient block.
// `kernel` is the resolved §7.15.3 kernel index; this entry point does not
// re-derive it, so tests can exercise every legal set.  Returns false for an
// invalid kernel/sec_tx_type/bit depth or a too-small buffer.
bool inverse_secondary_dequant_kernel(const TxParams& params,
                                      TxType plane_tx_type, int kernel,
                                      int sec_tx_type, bool transpose,
                                      int bit_depth, std::int32_t* dequant,
                                      std::size_t dequant_len);

// Syntax-level inverse.  Applies the §5.20.8.2 legality rules, resolves the
// kernel from YMode / most_probable_stx_set, then runs the normative inverse.
// A !kApplied result leaves `dequant` unchanged.
SecondaryStatus apply_inverse_secondary(const TxParams& params, int plane,
                                        int eob,
                                        const SecondarySyntax& syntax,
                                        const SecondaryTransformConfig& config,
                                        std::int32_t* dequant,
                                        std::size_t dequant_len);

// Encoder-side analysis projection (not normative).  Reads the n2 output
// positions from a full w*h primary-coefficient block, writes the n secondary
// coefficients at the first n scan positions, and zeroes every other element.
// This is the transpose of the inverse matrix with the same Round2Signed(.,7),
// which is the least-squares projection when the kernel rows are orthonormal
// (they are within rounding; see README).
bool forward_secondary_coeffs_kernel(const TxParams& params,
                                     TxType plane_tx_type, int kernel,
                                     int sec_tx_type, bool transpose,
                                     int bit_depth, std::int32_t* coeff,
                                     std::size_t coeff_len);

// Encoder-side analysis projection for one block already quantised under a
// primary TX_TYPE.  Unlike forward_secondary_coeffs_kernel() this keeps every
// primary coefficient outside the n2 output positions: the normative inverse
// only overwrites those positions, so the coded block is the projected first n
// scan coefficients plus the untouched tail.  `levels` is the adjusted
// (Min(32,W) x Min(32,H)) raster level array; it is rewritten in place.
// This is the M1 still encoder's project_ist_levels(), moved here so the video
// writer can share it.
bool project_ist_levels(const TxParams& params, TxType plane_tx_type,
                        int kernel, int sec_tx_type, bool transpose,
                        std::int32_t* levels, std::size_t levels_len);

// ---------------------------------------------------------------------------
// CDFs and static symbol costs (§9.3)
// ---------------------------------------------------------------------------

// TileSecTxTypeCdf[is_inter][Tx_Size_Sqr[txSz]]; 5 entries (4 symbols).
const std::uint16_t* sec_tx_type_cdf(int tx_size, bool is_inter) noexcept;

// TileMostProbableStxSetCdf or TileMostProbableStxSetAdstCdf; 8 or 5 entries.
const std::uint16_t* most_probable_stx_set_cdf(TxType plane_tx_type,
                                                int coded_w,
                                                int coded_h) noexcept;

// most_probable_stx_set is coded only for intra blocks with sec_tx_type != 0.
inline bool signals_most_probable_stx_set(bool is_inter,
                                          int sec_tx_type) noexcept {
  return !is_inter && sec_tx_type != 0;
}

double sec_tx_type_bits(int symbol, int tx_size, bool is_inter) noexcept;
double most_probable_stx_set_bits(int symbol, TxType plane_tx_type,
                                  int coded_w, int coded_h) noexcept;

}  // namespace av2enc::txsec
