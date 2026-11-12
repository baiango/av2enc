// AV2 Cross Component Transform (CCTX), §5.4.8 and §7.14.6.
//
// The decoder signals `cctx_type` for an eligible chroma transform block and
// rotates the two dequantised chroma coefficient planes before the inverse
// transform:
//
//   plane 1 (U): outU = Round2Signed(u * M[a][0] + v * (-M[a][1]), 8)
//   plane 2 (V): outV = Round2Signed(u * M[a][1] + v * ( M[a][0]), 8)
//
// with CCTX_PREC_BITS = 8 and Cctx_Mtx = {{181,181},{222,128},{128,222},
// {181,-181},{222,-128},{128,-222}}.
//
// The encoder side here searches the seven types (NONE + six angles), applies
// the inverse rotation to the original U/V forward coefficients, scalar
// quantises the rotated pair, and scores the exact decoder reconstruction.
// Rate is intentionally not baked in: the caller adds the per-plane
// coefficient costs and `cctx_type_bits(type)` (the CDF symbol cost), which is
// the same separation the RDOQ component uses for TX-type symbols.
//
// CCTX is chroma-only.  It is independent of TCQ (`useTcq` is plane 0 only).

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::tcq {

inline constexpr int kCctxPrecBits = 8;
inline constexpr int kCctxTypeCount = 7;

enum class CctxType : std::int8_t {
  kNone = 0,
  k45 = 1,
  k30 = 2,
  k60 = 3,
  kMinus45 = 4,
  kMinus30 = 5,
  kMinus60 = 6,
};

// Cctx_Mtx[CCTX_TYPES - 1][2], cosine/sine shifted by CCTX_PREC_BITS.
inline constexpr std::int16_t kCctxMtx[kCctxTypeCount - 1][2] = {
    {181, 181},   {222, 128},   {128, 222},
    {181, -181},  {222, -128},  {128, -222},
};

const char* cctx_type_name(CctxType type) noexcept;

// §4.8 Round2Signed: symmetric rounding, not an arithmetic shift for negatives.
std::int32_t round2_signed(std::int64_t value, int n) noexcept;

// One §7.14.6 component.  `u` and `v` are the dequantised coefficients of
// planes 1 and 2.  `bitdepth` is the video bit depth (8 for M1/M2).
std::int32_t cctx_dequant_component(std::int32_t u, std::int32_t v, int plane,
                                    CctxType type, int bitdepth) noexcept;

struct CctxPair {
  std::int32_t u = 0;
  std::int32_t v = 0;
};

// Both §7.14.6 outputs for the pair.  `type == kNone` returns the inputs.
CctxPair cctx_dequant_pair(std::int32_t u, std::int32_t v, CctxType type,
                           int bitdepth) noexcept;

// §5.20.7.27 is_cctx_allowed().  `chroma_w`/`chroma_h` are the chroma residual
// block dimensions, used only for the non-4:2:0 branch.
bool cctx_allowed(bool enable_cctx, bool lossless, bool is420, int chroma_w,
                  int chroma_h) noexcept;

// Ideal code length of one `cctx_type` symbol under Default_Cctx_Type_Cdf.
double cctx_type_bits(CctxType type) noexcept;

struct CctxConfig {
  bool enabled = true;       // enable_cctx && frame is eligible
  double quant_step_u = 1.0; // scalar forward step for plane 1
  double quant_step_v = 1.0; // scalar forward step for plane 2
  int bitdepth = 8;
};

struct CctxResult {
  CctxType type = CctxType::kNone;
  // Signed scalar levels for the (possibly rotated) U and V coefficients, in
  // the same order as the input spans.  These are the values the plane-1 and
  // plane-2 entropy writers consume.
  std::vector<std::int32_t> u_levels;
  std::vector<std::int32_t> v_levels;
  std::int32_t eob_u = 0;
  std::int32_t eob_v = 0;
  double distortion = 0.0;  // exact decoder reconstruction SSE, both planes
};

// Enumerates every candidate (one per enabled type).  Rate is not part of
// this core; an RD-aware caller scores each candidate with its own coefficient
// model plus `cctx_type_bits(candidate.type)` when the frame signals the
// symbol.  The vector is ordered by increasing type value and always contains
// CCTX_NONE when `config.enabled` is false.
bool cctx_enumerate_scalar(std::span<const double> u_coeff,
                           std::span<const double> v_coeff, int w, int h,
                           const CctxConfig& config,
                           std::vector<CctxResult>* candidates,
                           std::string* error = nullptr);

// Convenience: minimum-distortion candidate (no rate).  Use
// cctx_enumerate_scalar() for an RD decision.
bool cctx_quantize_scalar(std::span<const double> u_coeff,
                          std::span<const double> v_coeff, int w, int h,
                          const CctxConfig& config, CctxResult* result,
                          std::string* error = nullptr);

// Forward rotation used by the encoder search (exact inverse of §7.14.6's
// matrix, including its 256/(c^2+s^2) scale).  Exposed for tests.
void cctx_forward_rotate(double u, double v, CctxType type, double* out_u,
                         double* out_v) noexcept;

}  // namespace av2enc::tcq
