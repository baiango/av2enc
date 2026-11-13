// Device/host core for the still picture coefficient decisions:
//
//   * the §7.16 / §8.3 static coefficient rate model that
//     `src/codec/rate/rdoq/rdoq_av2_model.cpp` and
//     `src/codec/transform/tcq/tcq_av2_model.cpp` implement on the host, and
//   * the two verified trellises (`src/codec/rate/rdoq/rdoq.cpp` full-trellis
//     RDOQ and `src/codec/transform/tcq/tcq.cpp` $9.3 trellis-coded
//     quantisation) ported loop-for-loop to `__host__ __device__` code.
//
// Why a port instead of the shipped CUDA batch kernels
// ---------------------------------------------------
// `cuda/codec/rate/rdoq/rdoq_batch.cu` and `cuda/codec/transform/tcq/tcq.cu`
// take caller-built rate tables.  A table entry is `-log2(p)` for one
// (position, state, level) triple, so the table length is
// `area * states * (max_level + 1)`; at 640x360 QP 90 a single 32x32 luma block
// reaches |level| 1311, i.e. a bound of 2687, which would need a ~350 MiB table
// for one block.  Both kernels instead read the rate on demand here, so the
// device computes the same numbers without a table.  The transform arithmetic
// is **not** reimplemented: it stays in `cuda/codec/transform/variants` and
// `cuda/pipeline/still/tx`.
//
// Exactness
// ---------
// The host models call `std::log2(p / 32768.0)`.  `p` is an integer in
// [1, 32768], so the device reads the same value from a host-built table
// (`DecTables::log2p[p]`), computed once with `std::log2`.  Everything else in
// the model is integer arithmetic and `+`/`*`/`/` on doubles, so the device
// rate equals the CPU rate bit for bit.  The trellis port keeps the CPU loop
// order, tie-break order and arithmetic order; `decide_core_test.cpp` compares
// both ports against `rdoq::quantize` / `tcq::quantize` driven by the shipped
// `av2` models on randomized blocks.
//
// This header is compiled by nvcc for the kernels (`__host__ __device__`) and
// by the host compiler for the parity test (`inline`).
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define DEC_HD __host__ __device__ __forceinline__
#else
#define DEC_HD inline
#endif

// The host pass of the same header uses the std:: spellings; device code uses
// the global CUDA math functions.  (g++ does not expose ::isfinite.)
#if defined(__CUDA_ARCH__)
#define DEC_ISFINITE(x) ::isfinite(x)
#define DEC_FABS(x) ::fabs(x)
#define DEC_FLOOR(x) ::floor(x)
#define DEC_ROUND(x) ::round(x)
#else
#define DEC_ISFINITE(x) std::isfinite(x)
#define DEC_FABS(x) std::fabs(x)
#define DEC_FLOOR(x) std::floor(x)
#define DEC_ROUND(x) std::round(x)
#endif

namespace av2enc::cuda::still::decide {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

inline constexpr int kDecCdfSlices = 4;
inline constexpr double kDecCdfScale = 32768.0;
inline constexpr double kDecImpossibleBits = 1.0e6;
inline constexpr int kDecMaxSymbols = 8;
inline constexpr int kDecStateCount = 8;
inline constexpr int kDecMaxStates = 64;  // rdoq::Config::prev_level_cap <= 63
inline constexpr int kDecMaxArea = 32 * 32;

// §5.20.7.27 Tcq_Next_State (mirrored from codec/transform/tcq/tcq.h).
#if defined(__CUDA_ARCH__)
#define DEC_TABLE static __device__ constexpr
#else
#define DEC_TABLE static constexpr
#endif

DEC_TABLE std::int8_t kDecNextState[kDecStateCount][2] = {
    {0, 4}, {4, 0}, {1, 5}, {5, 1}, {6, 2}, {2, 6}, {7, 3}, {3, 7},
};

// §9.2 Tx_Size_Sqr / Tx_Size_Sqr_Up (mirrored from
// pipeline/still/still_tables.h; a host static_assert in the orchestrator
// pins both copies to the shipped tables).
DEC_TABLE int kDecTxSizeSqr[25] = {
    0, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 0, 0,
};
DEC_TABLE int kDecTxSizeSqrUp[25] = {
    0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 4, 4,
};

// ---------------------------------------------------------------------------
// §5.20.7.27 TCQ mappings (mirrored from codec/transform/tcq/tcq.h)
// ---------------------------------------------------------------------------

DEC_HD int dec_next_state(int state, int parity) {
  return kDecNextState[state & 7][parity & 1];
}
DEC_HD int dec_state_q0(int state) { return (state >> 1) & 1; }
DEC_HD int dec_state_cdf_index(int state) { return (state >> 1) & 1; }

DEC_HD int dec_syntax_level_for(int full_level, int pre_max) {
  if (full_level < pre_max - 1) return full_level;
  return ((full_level ^ pre_max) & 1) != 0 ? pre_max - 1 : pre_max;
}

DEC_HD int dec_syntax_escape_for(int full_level, int pre_max) {
  const int level = dec_syntax_level_for(full_level, pre_max);
  if (level < pre_max - 1) return 0;
  return (full_level - level) / 2;
}

// ---------------------------------------------------------------------------
// Rate-table view
// ---------------------------------------------------------------------------
//
// The four §8.3.1 COEFF_CDF_Q_CTXS slices are uploaded as one contiguous
// device blob; `DecTables::base` points at it and every array is addressed by a
// byte offset, so the same POD works with host pointers in the parity test.

struct DecTableOffsets {
  std::uint32_t txb_skip = 0;
  std::uint32_t v_txb_skip = 0;
  std::uint32_t coeff_base = 0;
  std::uint32_t coeff_base_lf = 0;
  std::uint32_t coeff_base_uv = 0;
  std::uint32_t coeff_base_lf_uv = 0;
  std::uint32_t coeff_base_eob = 0;
  std::uint32_t coeff_base_lf_eob = 0;
  std::uint32_t coeff_base_eob_uv = 0;
  std::uint32_t coeff_base_lf_eob_uv = 0;
  std::uint32_t coeff_br = 0;
  std::uint32_t coeff_br_lf = 0;
  std::uint32_t coeff_br_uv = 0;
  std::uint32_t dc_sign = 0;
  std::uint32_t eob_extra = 0;
  std::uint32_t eob_pt_16 = 0;
  std::uint32_t eob_pt_32 = 0;
  std::uint32_t eob_pt_64 = 0;
  std::uint32_t eob_pt_128 = 0;
  std::uint32_t eob_pt_256 = 0;
  std::uint32_t eob_pt_512 = 0;
  std::uint32_t eob_pt_1024 = 0;
};

struct DecTables {
  const std::uint8_t* base = nullptr;
  // `log2p[p] = -std::log2(p / 32768.0)` for p in 1..32768; [0] is unused.
  const double* log2p = nullptr;
  DecTableOffsets off[kDecCdfSlices];
};

DEC_HD const std::uint16_t* dec_array(const DecTables& tables,
                                      std::uint32_t offset) {
  return reinterpret_cast<const std::uint16_t*>(tables.base + offset);
}

// ---------------------------------------------------------------------------
// Per-block model inputs
// ---------------------------------------------------------------------------

struct DecBlockInfo {
  int plane = 0;
  int coded_w = 0;
  int coded_h = 0;
  int w = 0;
  int h = 0;
  int tx_size = 0;
  int txb_skip_context = 0;
  int dc_sign_context = 0;
  int qctx = 0;
  int is_inter = 0;  // rdoq ignores it; tcq uses it for the EOB context
};

// One coded coefficient's frozen model contexts.  `pos` is the raster index of
// the scan position; the array is indexed by scan position.
struct DecPosition {
  std::int32_t pos = 0;
  std::int32_t row = 0;
  std::int32_t col = 0;
  std::int32_t is_lf = 0;
  std::int32_t base_levels = 2;
  std::int32_t base_max = 3;
  std::int32_t br_max = 3;
  std::int32_t pre_max = 6;
  std::int32_t max_level = 6;
  std::int32_t base_ctx = 0;
  std::int32_t eob_ctx = 0;
  std::int32_t br_ctx = 0;
  std::int32_t hr_pred = 0;
  std::int32_t is_dc = 0;
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

DEC_HD std::int32_t dec_abs_i32(std::int32_t value) {
  if (value == std::numeric_limits<std::int32_t>::min()) {
    return std::numeric_limits<std::int32_t>::max();
  }
  return value < 0 ? -value : value;
}

DEC_HD std::int32_t dec_floor_log2(std::uint32_t value) {
  std::int32_t result = 0;
  while (value > 1U) {
    value >>= 1U;
    ++result;
  }
  return result;
}

DEC_HD std::int32_t dec_clip3(std::int32_t low, std::int32_t high,
                              std::int32_t value) {
  return value < low ? low : (value > high ? high : value);
}

// Scan position `c` -> raster index for the AV2 §5.20.7.30 diagonal scan
// (`syntax::build_scan`, `rdoq::av2::Model::build_scan` and `still_tx`'s
// closed form all describe this order).
DEC_HD int dec_scan_pos(int c, int w, int h) {
  int diagonal = 0;
  int rem = c;
  for (;;) {
    const int lo = diagonal - (w - 1) > 0 ? diagonal - (w - 1) : 0;
    const int hi = diagonal < h - 1 ? diagonal : h - 1;
    const int len = hi - lo + 1;
    if (rem < len) {
      const int row = hi - rem;
      const int col = diagonal - row;
      return row * w + col;
    }
    rem -= len;
    ++diagonal;
  }
}

DEC_HD std::int32_t dec_pre_level_from_abs(std::int32_t abs_value,
                                           std::int32_t base_levels,
                                           std::int32_t base_max,
                                           std::int32_t pre_max) {
  if (abs_value <= base_levels) return abs_value;
  const std::int32_t base =
      base_levels + 1 < base_max ? base_levels + 1 : base_max;
  const std::int32_t pre = abs_value < pre_max ? abs_value : pre_max;
  return base + (pre - base);
}

// ---------------------------------------------------------------------------
// §7.16 ideal code length
// ---------------------------------------------------------------------------

DEC_HD double dec_symbol_bits(const DecTables& tables,
                              const std::uint16_t* cdf, std::int32_t n,
                              std::int32_t symbol) {
  if (n <= 1) return 0.0;
  if (cdf == nullptr || n > kDecMaxSymbols || symbol < 0 || symbol >= n) {
    return kDecImpossibleBits;
  }
  std::int32_t probability = 0;
  if (symbol == 0) {
    probability = cdf[0];
  } else if (symbol == n - 1) {
    probability = static_cast<std::int32_t>(kDecCdfScale) - cdf[n - 2];
  } else {
    probability = static_cast<std::int32_t>(cdf[symbol]) - cdf[symbol - 1];
  }
  if (probability <= 0 || probability > 32768) {
    return kDecImpossibleBits;
  }
  return tables.log2p[probability];
}

// §5.20.7.28 read_quant escape: raw q length + golomb + literal.
DEC_HD double dec_escape_bits(std::int32_t x, std::int32_t predictor) {
  if (x < 0) return kDecImpossibleBits;
  const std::int32_t m = dec_clip3(
      1, 6,
      predictor <= 0
          ? 0
          : dec_floor_log2(static_cast<std::uint32_t>(predictor)));
  const std::int32_t k = m + 1;
  const std::int32_t c_max = (m + 4 < 6) ? m + 4 : 6;
  double bits = 0.0;
  if (x < (c_max << m)) {
    const std::int32_t q = x >> m;
    bits += static_cast<double>(q + 1);
    bits += static_cast<double>(m);
    return bits;
  }
  bits += static_cast<double>(c_max);
  std::int32_t g = 0;
  for (;;) {
    const std::int32_t length = k + g;
    const std::int32_t x_base = (c_max << m) + (1 << length) - (1 << k);
    if (x < x_base + (1 << length)) {
      bits += static_cast<double>(g + 1 + length);
      return bits;
    }
    ++g;
    if (g > 30) return kDecImpossibleBits;
  }
}

// §7.14.4 dequantisation of one level with the frame qlookup step.  QP is
// pinned to 90 on the still GPU path, so the step is the normative 528; `denom`
// carries the coded-size shift and the TCQ +1.  Same integer arithmetic as
// `txv::dequantize` / `still_tx`'s device dequantiser.
DEC_HD std::int32_t dec_dequant_value(std::int32_t value, std::int32_t denom) {
  const std::int32_t aq = value < 0 ? -value : value;
  const std::int32_t high = (aq * 528) & 0xFFFFFF;
  const std::int32_t dq = ((high + 4) >> 3) / denom;
  const std::int32_t signed_value = value < 0 ? -dq : dq;
  return dec_clip3(-(1 << 15), (1 << 15) - 1, signed_value);
}

// ---------------------------------------------------------------------------
// Position table construction
// ---------------------------------------------------------------------------
//
// Mirrors `rdoq::av2::Model::init()` (`tcq == 0`) and
// `tcq::av2::Model::init()` (`tcq == 1`).  `reference_scan` is the frozen
// reference block in scan order; `level_pre_scratch` holds `area` ints and is
// raster-indexed (same as the host model's `level_pre` vector).
DEC_HD void dec_build_positions(const DecBlockInfo& info,
                                const std::int32_t* reference_scan, int tcq,
                                std::int32_t* level_pre_scratch,
                                DecPosition* out) {
  const int w = info.w;
  const int h = info.h;
  const int area = w * h;
  const bool luma = info.plane == 0;

  for (int c = 0; c < area; ++c) {
    const int pos = dec_scan_pos(c, w, h);
    const int row = pos / w;
    const int col = pos % w;
    const bool lf = luma ? (row + col) < 4 : (row + col) < 1;
    const int base_levels = lf ? 4 : 2;
    const int base_max = lf ? 5 : 3;
    const int pre_max = base_max + 3;
    const int abs_value = dec_abs_i32(reference_scan[c]);
    int pre_level = 0;
    if (tcq != 0) {
      const int syntax =
          abs_value == 0 ? 0 : dec_syntax_level_for(abs_value, pre_max);
      pre_level = dec_pre_level_from_abs(syntax, base_levels, base_max, pre_max);
    } else {
      pre_level =
          dec_pre_level_from_abs(abs_value, base_levels, base_max, pre_max);
    }
    level_pre_scratch[pos] = pre_level;
  }

  DEC_TABLE int kSigRef[5][2] = {
      {0, 1}, {1, 0}, {1, 1}, {0, 2}, {2, 0}};
  DEC_TABLE int kMagRef[3][2] = {{0, 1}, {1, 0}, {1, 1}};

  for (int c = 0; c < area; ++c) {
    DecPosition& p = out[c];
    p.pos = dec_scan_pos(c, w, h);
    p.row = p.pos / w;
    p.col = p.pos % w;
    p.is_lf = (((luma ? (p.row + p.col) < 4 : (p.row + p.col) < 1)) ? 1 : 0);
    p.base_levels = p.is_lf != 0 ? 4 : 2;
    p.base_max = p.is_lf != 0 ? 5 : 3;
    if (tcq != 0) {
      p.br_max = 3;
      p.max_level = p.is_lf != 0 ? 8 : 6;
    } else {
      p.br_max = (p.is_lf != 0 && !luma) ? 0 : 3;
      p.max_level = p.is_lf != 0 ? (luma ? 8 : 5) : 6;
    }
    p.pre_max = p.base_max + p.br_max;
    p.is_dc = (luma && p.pos == 0) ? 1 : 0;

    const int limit = p.is_lf != 0 ? 5 : 3;
    int mag = 0;
    const int refs = luma ? 5 : 3;
    for (int k = 0; k < refs; ++k) {
      const int rr = p.row + kSigRef[k][0];
      const int cc = p.col + kSigRef[k][1];
      if (rr < h && cc < w) {
        const int v = level_pre_scratch[rr * w + cc];
        mag += v < limit ? v : limit;
      }
    }
    const int mctx = (mag + 1) >> 1;
    if (luma) {
      if (p.is_lf != 0) {
        if (c == 0) {
          p.base_ctx = mctx < 8 ? mctx : 8;
        } else if (p.row + p.col < 2) {
          p.base_ctx = (mctx < 6 ? mctx : 6) + 9;
        } else {
          p.base_ctx = (mctx < 4 ? mctx : 4) + 16;
        }
      } else {
        const int ctx2 = mctx < 4 ? mctx : 4;
        p.base_ctx = ctx2;
        if (p.row + p.col >= 8) {
          p.base_ctx += 10;
        } else if (p.row + p.col >= 6) {
          p.base_ctx += 5;
        }
      }
    } else {
      const int ctx2 = mctx < 3 ? mctx : 3;
      p.base_ctx = info.plane == 1 ? ctx2 : ctx2 + 4;
    }

    if (c == 0) {
      p.eob_ctx = 0;
    } else if (c <= area / 8) {
      p.eob_ctx = 1;
    } else if (c <= area / 4) {
      p.eob_ctx = 2;
    } else {
      p.eob_ctx = 3;
    }

    int br_mag = 0;
    for (int k = 0; k < 3; ++k) {
      const int rr = p.row + kMagRef[k][0];
      const int cc = p.col + kMagRef[k][1];
      if (rr < h && cc < w) {
        const int v = level_pre_scratch[rr * w + cc];
        br_mag += v < 5 ? v : 5;
      }
    }
    br_mag = (br_mag + 1) >> 1;
    br_mag = br_mag < 6 ? br_mag : 6;
    if (luma) {
      if (p.is_lf != 0) {
        p.br_ctx = br_mag + (p.pos != 0 ? 7 : 0);
      } else {
        p.br_ctx = br_mag;
      }
    } else {
      p.br_ctx = br_mag < 3 ? br_mag : 3;
    }
  }

  // Escape predictor state from the reference reverse scan.
  std::int32_t hr_level_avg = 0;
  for (int c = area - 1; c >= 0; --c) {
    DecPosition& p = out[c];
    p.hr_pred = hr_level_avg;
    const std::int32_t abs_value = dec_abs_i32(reference_scan[c]);
    if (tcq != 0) {
      if (abs_value >= p.max_level - 1) {
        const std::int32_t x = dec_syntax_escape_for(abs_value, p.max_level);
        hr_level_avg = (x + hr_level_avg) >> 1;
      }
    } else {
      if (abs_value >= p.max_level) {
        const std::int32_t pre = dec_pre_level_from_abs(
            abs_value, p.base_levels, p.base_max, p.pre_max);
        const std::int32_t x = abs_value - pre;
        hr_level_avg = (x + hr_level_avg) >> 1;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// §7.16 coefficient costs (mirrors the two av2 models)
// ---------------------------------------------------------------------------

DEC_HD const std::uint16_t* dec_txb_skip_cdf(const DecTables& tables,
                                             const DecBlockInfo& info,
                                             int tcq) {
  const DecTableOffsets& o = tables.off[info.qctx];
  if (tcq != 0) {
    const int tsc =
        (kDecTxSizeSqr[info.tx_size] + kDecTxSizeSqrUp[info.tx_size] + 1) >> 1;
    const std::uint16_t* base = dec_array(tables, o.txb_skip);
    return base + ((info.is_inter ? 1 : 0) * 5 + tsc) * 10 * 3 +
           info.txb_skip_context * 3;
  }
  if (info.plane == 2) {
    const std::uint16_t* base = dec_array(tables, o.v_txb_skip);
    return base + info.txb_skip_context * 3;
  }
  const int tsc =
      (kDecTxSizeSqr[info.tx_size] + kDecTxSizeSqrUp[info.tx_size] + 1) >> 1;
  const std::uint16_t* base = dec_array(tables, o.txb_skip);
  return base + (tsc * 10 + info.txb_skip_context) * 3;
}

DEC_HD double dec_eob_bits(const DecTables& tables, const DecBlockInfo& info,
                           int eob, int tcq) {
  const DecTableOffsets& o = tables.off[info.qctx];
  const int area = info.w * info.h;
  if (eob < 0 || eob > area) return kDecImpossibleBits;
  double bits = dec_symbol_bits(tables, dec_txb_skip_cdf(tables, info, tcq), 2,
                                eob == 0 ? 1 : 0);
  if (eob == 0) return bits;

  std::int32_t eob_pt = 1;
  if (eob == 2) {
    eob_pt = 2;
  } else if (eob >= 3) {
    eob_pt = dec_floor_log2(static_cast<std::uint32_t>(eob - 1)) + 2;
  }

  const std::int32_t log2_w =
      dec_floor_log2(static_cast<std::uint32_t>(info.coded_w));
  const std::int32_t log2_h =
      dec_floor_log2(static_cast<std::uint32_t>(info.coded_h));
  const std::int32_t multisize = (log2_w < 5 ? log2_w : 5) +
                                 (log2_h < 5 ? log2_h : 5) - 4;

  const std::uint16_t* cdf = nullptr;
  std::int32_t n = 0;
  std::int32_t escape_raw_bits = 0;
  switch (multisize) {
    case 0:
      cdf = dec_array(tables, o.eob_pt_16);
      n = 5;
      break;
    case 1:
      cdf = dec_array(tables, o.eob_pt_32);
      n = 6;
      break;
    case 2:
      cdf = dec_array(tables, o.eob_pt_64);
      n = 7;
      break;
    case 3:
      cdf = dec_array(tables, o.eob_pt_128);
      n = 8;
      break;
    case 4:
      cdf = dec_array(tables, o.eob_pt_256);
      n = 8;
      escape_raw_bits = 1;
      break;
    case 5:
      cdf = dec_array(tables, o.eob_pt_512);
      n = 8;
      escape_raw_bits = 2;
      break;
    default:
      cdf = dec_array(tables, o.eob_pt_1024);
      n = 8;
      escape_raw_bits = 2;
      break;
  }
  const std::int32_t eob_ctx =
      tcq != 0 ? (info.is_inter ? 1 : 0) : (info.plane > 0 ? 2 : 0);
  cdf += static_cast<std::int64_t>(eob_ctx) * (n + 1);

  std::int32_t symbol = 0;
  if (eob_pt <= n) {
    symbol = eob_pt - 1;
  } else {
    symbol = n - 1;
  }
  bits += dec_symbol_bits(tables, cdf, n, symbol);
  if (symbol == n - 1 && escape_raw_bits > 0) {
    bits += static_cast<double>(escape_raw_bits);
  }

  if (eob_pt >= 3) {
    const std::int32_t base = (1 << (eob_pt - 2)) + 1;
    const std::int32_t extra = eob - base;
    const std::int32_t eob_extra = (extra >> (eob_pt - 3)) & 1;
    bits += dec_symbol_bits(tables, dec_array(tables, o.eob_extra), 2,
                            eob_extra);
    for (std::int32_t i = eob_pt - 4; i >= 0; --i) {
      bits += 1.0;
    }
  }
  return bits;
}

// RDOQ coefficient cost (`rdoq::av2::Model::coefficient_bits`).
DEC_HD double dec_rdoq_coefficient_bits(const DecTables& tables,
                                        const DecBlockInfo& info,
                                        const DecPosition* positions,
                                        std::int32_t scan_pos,
                                        std::int32_t signed_level,
                                        std::int32_t is_eob) {
  const DecTableOffsets& o = tables.off[info.qctx];
  const int area = info.w * info.h;
  if (scan_pos < 0 || scan_pos >= area) return kDecImpossibleBits;
  const int tsc =
      (kDecTxSizeSqr[info.tx_size] + kDecTxSizeSqrUp[info.tx_size] + 1) >> 1;
  const DecPosition& p = positions[scan_pos];
  const std::int32_t abs_value = dec_abs_i32(signed_level);
  const bool luma = info.plane == 0;

  if (abs_value == 0) {
    if (is_eob != 0) return kDecImpossibleBits;
    double bits = 0.0;
    if (p.is_lf != 0) {
      if (luma) {
        const std::uint16_t* base = dec_array(tables, o.coeff_base_lf);
        bits += dec_symbol_bits(
            tables, base + ((tsc * 33 + p.base_ctx) * 2) * 7, 6, 0);
      } else {
        const std::uint16_t* base = dec_array(tables, o.coeff_base_lf_uv);
        bits += dec_symbol_bits(tables, base + p.base_ctx * 7, 6, 0);
      }
    } else {
      if (luma) {
        const std::uint16_t* base = dec_array(tables, o.coeff_base);
        bits += dec_symbol_bits(
            tables, base + ((tsc * 20 + p.base_ctx) * 2) * 5, 4, 0);
      } else {
        const std::uint16_t* base = dec_array(tables, o.coeff_base_uv);
        bits += dec_symbol_bits(tables, base + p.base_ctx * 5, 4, 0);
      }
    }
    return bits;
  }

  const std::int32_t base_levels = p.base_levels;
  const std::int32_t base_max = p.base_max;
  const std::int32_t pre_max = p.pre_max;
  std::int32_t base_level = 0;
  std::int32_t br_level = 0;
  if (abs_value <= base_levels) {
    base_level = abs_value;
  } else {
    base_level = (base_levels + 1 < base_max) ? base_levels + 1 : base_max;
    const std::int32_t pre = abs_value < pre_max ? abs_value : pre_max;
    br_level = pre - base_level;
  }
  const std::int32_t pre_level = base_level + br_level;

  double bits = 0.0;
  if (is_eob != 0) {
    const std::int32_t symbol = base_level - 1;
    if (luma) {
      if (p.is_lf != 0) {
        const std::uint16_t* base = dec_array(tables, o.coeff_base_lf_eob);
        bits += dec_symbol_bits(tables, base + (tsc * 4 + p.eob_ctx) * 6, 5,
                                symbol);
      } else {
        const std::uint16_t* base = dec_array(tables, o.coeff_base_eob);
        bits += dec_symbol_bits(tables, base + (tsc * 4 + p.eob_ctx) * 4, 3,
                                symbol);
      }
    } else {
      if (p.is_lf != 0) {
        const std::uint16_t* base = dec_array(tables, o.coeff_base_lf_eob_uv);
        bits += dec_symbol_bits(tables, base + p.eob_ctx * 6, 5, symbol);
      } else {
        const std::uint16_t* base = dec_array(tables, o.coeff_base_eob_uv);
        bits += dec_symbol_bits(tables, base + p.eob_ctx * 4, 3, symbol);
      }
    }
  } else {
    if (luma) {
      if (p.is_lf != 0) {
        const std::uint16_t* base = dec_array(tables, o.coeff_base_lf);
        bits += dec_symbol_bits(
            tables, base + ((tsc * 33 + p.base_ctx) * 2) * 7, 6, base_level);
      } else {
        const std::uint16_t* base = dec_array(tables, o.coeff_base);
        bits += dec_symbol_bits(
            tables, base + ((tsc * 20 + p.base_ctx) * 2) * 5, 4, base_level);
      }
    } else {
      if (p.is_lf != 0) {
        const std::uint16_t* base = dec_array(tables, o.coeff_base_lf_uv);
        bits +=
            dec_symbol_bits(tables, base + p.base_ctx * 7, 6, base_level);
      } else {
        const std::uint16_t* base = dec_array(tables, o.coeff_base_uv);
        bits +=
            dec_symbol_bits(tables, base + p.base_ctx * 5, 4, base_level);
      }
    }
  }

  if (pre_level > base_levels && !(p.is_lf != 0 && !luma)) {
    if (luma) {
      if (p.is_lf != 0) {
        const std::uint16_t* base = dec_array(tables, o.coeff_br_lf);
        bits += dec_symbol_bits(tables, base + p.br_ctx * 5, 4, br_level);
      } else {
        const std::uint16_t* base = dec_array(tables, o.coeff_br);
        bits += dec_symbol_bits(tables, base + p.br_ctx * 5, 4, br_level);
      }
    } else {
      const std::uint16_t* base = dec_array(tables, o.coeff_br_uv);
      bits += dec_symbol_bits(tables, base + p.br_ctx * 5, 4, br_level);
    }
  }

  if (abs_value >= p.max_level) {
    bits += dec_escape_bits(abs_value - pre_level, p.hr_pred);
  }

  if (p.is_dc != 0) {
    const std::uint16_t* base = dec_array(tables, o.dc_sign);
    bits += dec_symbol_bits(tables, base + info.dc_sign_context * 3, 2,
                            signed_level < 0 ? 1 : 0);
  } else {
    bits += 1.0;
  }
  return bits;
}

// TCQ coefficient cost (`tcq::av2::Model::coefficient_bits`).
DEC_HD double dec_tcq_coefficient_bits(const DecTables& tables,
                                       const DecBlockInfo& info,
                                       const DecPosition* positions,
                                       std::int32_t scan_pos,
                                       std::int32_t base_level,
                                       std::int32_t state,
                                       std::int32_t is_eob) {
  const DecTableOffsets& o = tables.off[info.qctx];
  const int area = info.w * info.h;
  if (scan_pos < 0 || scan_pos >= area) return kDecImpossibleBits;
  const int tsc =
      (kDecTxSizeSqr[info.tx_size] + kDecTxSizeSqrUp[info.tx_size] + 1) >> 1;
  const DecPosition& p = positions[scan_pos];
  const std::int32_t abs_value = dec_abs_i32(base_level);
  const std::int32_t tcq = dec_state_cdf_index(state);

  if (abs_value == 0) {
    if (is_eob != 0) return kDecImpossibleBits;
    if (p.is_lf != 0) {
      const std::uint16_t* base = dec_array(tables, o.coeff_base_lf);
      return dec_symbol_bits(
          tables, base + ((tsc * 33 + p.base_ctx) * 2 + tcq) * 7, 6, 0);
    }
    const std::uint16_t* base = dec_array(tables, o.coeff_base);
    return dec_symbol_bits(
        tables, base + ((tsc * 20 + p.base_ctx) * 2 + tcq) * 5, 4, 0);
  }

  const std::int32_t pre_max = p.pre_max;
  const std::int32_t syntax = dec_syntax_level_for(abs_value, pre_max);
  std::int32_t base_symbol = 0;
  std::int32_t br_level = 0;
  if (syntax <= p.base_levels) {
    base_symbol = syntax;
  } else {
    base_symbol =
        (p.base_levels + 1 < p.base_max) ? p.base_levels + 1 : p.base_max;
    const std::int32_t pre = syntax < pre_max ? syntax : pre_max;
    br_level = pre - base_symbol;
  }
  const std::int32_t pre_level = base_symbol + br_level;

  double bits = 0.0;
  if (is_eob != 0) {
    const std::int32_t symbol = base_symbol - 1;
    if (p.is_lf != 0) {
      const std::uint16_t* base = dec_array(tables, o.coeff_base_lf_eob);
      bits += dec_symbol_bits(tables, base + (tsc * 4 + p.eob_ctx) * 6, 5,
                              symbol);
    } else {
      const std::uint16_t* base = dec_array(tables, o.coeff_base_eob);
      bits += dec_symbol_bits(tables, base + (tsc * 4 + p.eob_ctx) * 4, 3,
                              symbol);
    }
  } else {
    if (p.is_lf != 0) {
      const std::uint16_t* base = dec_array(tables, o.coeff_base_lf);
      bits += dec_symbol_bits(
          tables, base + ((tsc * 33 + p.base_ctx) * 2 + tcq) * 7, 6,
          base_symbol);
    } else {
      const std::uint16_t* base = dec_array(tables, o.coeff_base);
      bits += dec_symbol_bits(
          tables, base + ((tsc * 20 + p.base_ctx) * 2 + tcq) * 5, 4,
          base_symbol);
    }
  }

  if (pre_level > p.base_levels) {
    if (p.is_lf != 0) {
      const std::uint16_t* base = dec_array(tables, o.coeff_br_lf);
      bits += dec_symbol_bits(tables, base + p.br_ctx * 5, 4, br_level);
    } else {
      const std::uint16_t* base = dec_array(tables, o.coeff_br);
      bits += dec_symbol_bits(tables, base + p.br_ctx * 5, 4, br_level);
    }
  }

  if (pre_level >= p.max_level - 1) {
    bits += dec_escape_bits(dec_syntax_escape_for(abs_value, p.max_level),
                            p.hr_pred);
  }

  if (p.is_dc != 0) {
    const std::uint16_t* base = dec_array(tables, o.dc_sign);
    bits += dec_symbol_bits(tables, base + info.dc_sign_context * 3, 2,
                            base_level < 0 ? 1 : 0);
  } else {
    bits += 1.0;
  }
  return bits;
}

// ---------------------------------------------------------------------------
// Trellis ports
// ---------------------------------------------------------------------------

struct DecChoice {
  double cost;
  double distortion;
  double rate_bits;
  std::int32_t base_level;
  std::int32_t quant_level;
  std::int32_t next_state;
  std::int32_t abs_level;
  std::int32_t mode;  // rdoq only: 0 = all-zero end, 1 = continue, 2 = EOB
  std::int32_t terminal;
};

DEC_HD DecChoice dec_choice_infinite() {
  DecChoice c;
  c.cost = std::numeric_limits<double>::infinity();
  c.distortion = 0.0;
  c.rate_bits = 0.0;
  c.base_level = 0;
  c.quant_level = 0;
  c.next_state = 0;
  c.abs_level = 0;
  c.mode = 0;
  c.terminal = 0;
  return c;
}

DEC_HD DecChoice dec_choice_zero() {
  DecChoice c = dec_choice_infinite();
  c.cost = 0.0;
  return c;
}

// rdoq.cpp choice_better(): cost, distortion, rate, smaller magnitude,
// terminal before non-terminal, smaller signed level.
DEC_HD bool dec_rdoq_choice_better(const DecChoice& a, const DecChoice& b) {
  if (a.cost != b.cost) return a.cost < b.cost;
  if (a.distortion != b.distortion) return a.distortion < b.distortion;
  if (a.rate_bits != b.rate_bits) return a.rate_bits < b.rate_bits;
  if (a.abs_level != b.abs_level) return a.abs_level < b.abs_level;
  if (a.terminal != b.terminal) return a.terminal > b.terminal;
  return a.base_level < b.base_level;
}

// tcq.cpp choice_better(): cost, distortion, rate, smaller magnitude, then
// smaller Quant magnitude.
DEC_HD bool dec_tcq_choice_better(const DecChoice& a, const DecChoice& b) {
  if (a.cost != b.cost) return a.cost < b.cost;
  if (a.distortion != b.distortion) return a.distortion < b.distortion;
  if (a.rate_bits != b.rate_bits) return a.rate_bits < b.rate_bits;
  if (a.abs_level != b.abs_level) return a.abs_level < b.abs_level;
  return a.quant_level < b.quant_level;
}

DEC_HD bool dec_rate_valid(double rate) {
  return DEC_ISFINITE(rate) && rate >= 0.0;
}

// `rdoq.cpp` level_bound_for().
DEC_HD bool dec_rdoq_level_bound(double original, double quant_step,
                                 std::int32_t prev_level_cap,
                                 std::int32_t level_cap,
                                 std::int32_t allow_zero_to_one,
                                 std::int32_t* bound, bool* truncated) {
  const double ratio = DEC_FABS(original) / quant_step;
  if (!DEC_ISFINITE(ratio)) return false;
  double raw = DEC_FLOOR(2.0 * ratio) + 1.0;
  const double max_level = static_cast<double>(
      std::numeric_limits<std::int32_t>::max() - 1);
  if (raw > max_level) raw = max_level;
  std::int64_t value = static_cast<std::int64_t>(raw);
  if (original == 0.0 && !allow_zero_to_one) {
    value = 0;
  } else {
    value = value > prev_level_cap ? value : prev_level_cap;
    const std::int64_t with_margin = value + 64;
    value = with_margin < static_cast<std::int64_t>(
                              std::numeric_limits<std::int32_t>::max() - 1)
                ? with_margin
                : static_cast<std::int64_t>(
                      std::numeric_limits<std::int32_t>::max() - 1);
  }
  value = value > 0 ? value : 0;
  bool clipped = false;
  if (level_cap > 0 && value > static_cast<std::int64_t>(level_cap)) {
    value = level_cap;
    clipped = true;
  }
  *bound = static_cast<std::int32_t>(value);
  *truncated = clipped;
  return true;
}

// `tcq.cpp` bound_for().
DEC_HD bool dec_tcq_level_bound(double original, double quant_step,
                                std::int32_t level_cap,
                                std::int32_t allow_zero_to_one,
                                std::int32_t* bound, bool* truncated) {
  const double ratio = DEC_FABS(original) / quant_step;
  if (!DEC_ISFINITE(ratio)) return false;
  double raw = DEC_FLOOR(2.0 * ratio) + 2.0;
  if (original == 0.0 && !allow_zero_to_one) {
    raw = 0.0;
  }
  const double max_level = static_cast<double>(
      std::numeric_limits<std::int32_t>::max() - 1);
  if (raw > max_level) raw = max_level;
  std::int64_t value = static_cast<std::int64_t>(raw);
  if (!(original == 0.0 && !allow_zero_to_one)) {
    value = value > 1 ? value : 1;
    const std::int64_t with_margin = value + 64;
    value = with_margin < static_cast<std::int64_t>(
                              std::numeric_limits<std::int32_t>::max() - 1)
                ? with_margin
                : static_cast<std::int64_t>(
                      std::numeric_limits<std::int32_t>::max() - 1);
  }
  value = value > 0 ? value : 0;
  bool clipped = false;
  if (level_cap > 0 && value > static_cast<std::int64_t>(level_cap)) {
    value = level_cap;
    clipped = true;
  }
  *bound = static_cast<std::int32_t>(value);
  *truncated = clipped;
  return true;
}

struct DecRdoqConfig {
  double lambda = 0.0;
  double quant_step = 1.0;
  std::int32_t prev_level_cap = 2;
  std::int32_t level_cap = 0;
  std::int32_t allow_zero_to_one = 0;
};

struct DecTcqConfig {
  double lambda = 0.0;
  double quant_step = 1.0;
  std::int32_t level_cap = 0;
  std::int32_t allow_zero_to_one = 0;
};

// Per-block scratch for the trellis ports.  The DP state arrays are tiny but
// the EOB / suffix / bound tables are `kDecMaxArea` entries each; keeping them
// in the caller's device buffer instead of the kernel's local frame avoids a
// ~32 KiB/thread local allocation (which the driver reserves for the maximum
// block size and can fail with `cudaErrorMemoryAllocation` on a busy device).
struct DecTrellisScratch {
  double* eob_bits = nullptr;    // kDecMaxArea + 1
  double* suffix_zero = nullptr; // kDecMaxArea + 1
  std::int32_t* signs = nullptr;  // kDecMaxArea
  double* tail = nullptr;        // kDecMaxArea + 1
  std::int32_t* bounds = nullptr;  // kDecMaxArea
};

struct DecRdoqResult {
  std::int32_t eob = 0;
  double distortion = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  std::int32_t max_level_searched = 0;
  std::int32_t level_range_truncated = 0;
  std::int32_t status = 0;  // 0 ok, -1 rejected
};

struct DecTcqResult {
  std::int32_t eob = 0;
  double distortion = 0.0;
  double rate_bits = 0.0;
  double cost = 0.0;
  std::int32_t max_level_searched = 0;
  std::int32_t level_range_truncated = 0;
  std::int32_t status = 0;
};

// Full-trellis RDOQ (`rdoq::quantize`), one block.  `levels_out` receives the
// signed syntax levels in scan order; `chosen_level`/`chosen_mode` are
// `num_coefficients * (prev_level_cap + 1)` scratch cells.
DEC_HD DecRdoqResult dec_rdoq_quantize(const DecTables& tables,
                                       const DecBlockInfo& info,
                                       const DecPosition* positions,
                                       const double* original,
                                       const std::int32_t* scalar_level,
                                       std::int32_t num_coefficients,
                                       const DecRdoqConfig& config,
                                       const DecTrellisScratch& scratch,
                                       std::int32_t* levels_out,
                                       std::int32_t* chosen_level,
                                       std::uint8_t* chosen_mode) {
  DecRdoqResult result;
  const std::int32_t count = num_coefficients;
  const std::int32_t cap = config.prev_level_cap;
  const std::int32_t states = cap + 1;
  const double lambda = config.lambda;
  const double quant_step = config.quant_step;

  for (std::int32_t i = 0; i < count; ++i) levels_out[i] = 0;
  if (count == 0) {
    const double bits = dec_eob_bits(tables, info, 0, 0);
    if (!dec_rate_valid(bits)) {
      result.status = -1;
      return result;
    }
    result.rate_bits = bits;
    result.cost = lambda * bits;
    return result;
  }
  if (!(DEC_ISFINITE(lambda) && lambda >= 0.0 && DEC_ISFINITE(quant_step) &&
        quant_step > 0.0) ||
      states < 1 || states > kDecMaxStates || count > kDecMaxArea) {
    result.status = -1;
    return result;
  }

  // EOB costs, including the all-zero flag.
  double* eob_bits = scratch.eob_bits;
  for (std::int32_t i = 0; i <= count; ++i) {
    const double bits = dec_eob_bits(tables, info, i, 0);
    if (!dec_rate_valid(bits)) {
      result.status = -1;
      return result;
    }
    eob_bits[i] = bits;
  }

  // Distortion if every coefficient from i onward is forced to zero.
  double* suffix_zero = scratch.suffix_zero;
  std::int32_t* signs = scratch.signs;
  suffix_zero[count] = 0.0;
  for (std::int32_t i = count - 1; i >= 0; --i) {
    const double value = original[i];
    if (!DEC_ISFINITE(value)) {
      result.status = -1;
      return result;
    }
    suffix_zero[i] = suffix_zero[i + 1] + value * value;
    if (value < 0.0) {
      signs[i] = -1;
    } else if (value > 0.0) {
      signs[i] = 1;
    } else {
      signs[i] = scalar_level[i] < 0 ? -1 : 1;
    }
  }

  double dp_next[kDecMaxStates];
  double dd_next[kDecMaxStates];
  double dr_next[kDecMaxStates];
  double dp_cur[kDecMaxStates];
  double dd_cur[kDecMaxStates];
  double dr_cur[kDecMaxStates];
  for (std::int32_t s = 0; s < states; ++s) {
    dp_next[s] = std::numeric_limits<double>::infinity();
    dd_next[s] = 0.0;
    dr_next[s] = 0.0;
  }

  std::int32_t max_level_searched = 0;
  bool truncated = false;

  for (std::int32_t ii = count - 1; ii >= 0; --ii) {
    const double value = original[ii];
    const std::int32_t sign = signs[ii];

    std::int32_t bound = 0;
    bool clipped = false;
    if (!dec_rdoq_level_bound(value, quant_step, cap, config.level_cap,
                              config.allow_zero_to_one, &bound, &clipped)) {
      result.status = -1;
      return result;
    }
    truncated = truncated || clipped;

    const double ratio = DEC_FABS(value) / quant_step;
    std::int32_t closest = 0;
    if (DEC_ISFINITE(ratio)) {
      const double rounded = DEC_ROUND(ratio);
      if (rounded <= 0.0) {
        closest = 0;
      } else if (rounded >= static_cast<double>(bound)) {
        closest = bound;
      } else {
        closest = static_cast<std::int32_t>(rounded);
      }
    }
    std::int32_t max_here = 0;

    const double zero_after = suffix_zero[ii + 1];

    for (std::int32_t prev = 0; prev < states; ++prev) {
      DecChoice best = dec_choice_infinite();
      if (ii == 0) {
        best.distortion = suffix_zero[0];
        best.rate_bits = eob_bits[0];
        best.cost = best.distortion + lambda * best.rate_bits;
        best.base_level = 0;
        best.abs_level = 0;
        best.mode = 0;
        best.terminal = 1;
      }

      for (std::int32_t magnitude = 0; magnitude <= bound; ++magnitude) {
        const std::int32_t signed_level =
            magnitude == 0 ? 0 : sign * magnitude;
        const double reconstructed =
            static_cast<double>(signed_level) * quant_step;
        const double difference = value - reconstructed;
        const double distortion = difference * difference;
        const std::int32_t next_state = magnitude < cap ? magnitude : cap;

        const double normal_bits = dec_rdoq_coefficient_bits(
            tables, info, positions, ii, signed_level, 0);
        if (!dec_rate_valid(normal_bits)) {
          result.status = -1;
          return result;
        }
        if (DEC_ISFINITE(dp_next[next_state])) {
          DecChoice candidate;
          candidate.cost =
              distortion + lambda * normal_bits + dp_next[next_state];
          candidate.distortion = distortion + dd_next[next_state];
          candidate.rate_bits = normal_bits + dr_next[next_state];
          candidate.base_level = signed_level;
          candidate.abs_level = magnitude;
          candidate.mode = 1;
          candidate.terminal = 0;
          if (dec_rdoq_choice_better(candidate, best)) best = candidate;
        }

        if (magnitude != 0) {
          const double eob_coeff_bits = dec_rdoq_coefficient_bits(
              tables, info, positions, ii, signed_level, 1);
          if (!dec_rate_valid(eob_coeff_bits)) {
            result.status = -1;
            return result;
          }
          DecChoice candidate;
          candidate.distortion = distortion + zero_after;
          candidate.rate_bits = eob_coeff_bits + eob_bits[ii + 1];
          candidate.cost = candidate.distortion + lambda * candidate.rate_bits;
          candidate.base_level = signed_level;
          candidate.abs_level = magnitude;
          candidate.mode = 2;
          candidate.terminal = 1;
          if (dec_rdoq_choice_better(candidate, best)) best = candidate;
        }
        max_here = magnitude > max_here ? magnitude : max_here;
        if (magnitude >= closest && DEC_ISFINITE(best.cost) &&
            distortion >= best.cost) {
          break;
        }
      }
      dp_cur[prev] = best.cost;
      dd_cur[prev] = best.distortion;
      dr_cur[prev] = best.rate_bits;
      const std::int32_t slot = ii * states + prev;
      chosen_level[slot] = best.base_level;
      chosen_mode[slot] = static_cast<std::uint8_t>(best.mode);
    }
    max_level_searched =
        max_here > max_level_searched ? max_here : max_level_searched;

    for (std::int32_t s = 0; s < states; ++s) {
      dp_next[s] = dp_cur[s];
      dd_next[s] = dd_cur[s];
      dr_next[s] = dr_cur[s];
    }
  }

  if (!DEC_ISFINITE(dp_next[0])) {
    result.status = -1;
    return result;
  }

  std::int32_t state = 0;
  std::int32_t eob = count;
  for (std::int32_t i = 0; i < count; ++i) {
    const std::int32_t slot = i * states + state;
    const std::int32_t level = chosen_level[slot];
    const std::int32_t mode = chosen_mode[slot];
    if (mode == 0) {
      eob = i;
      break;
    }
    levels_out[i] = level;
    state = dec_abs_i32(level) < cap ? dec_abs_i32(level) : cap;
    if (mode == 2) {
      eob = i + 1;
      break;
    }
  }

  double distortion = 0.0;
  for (std::int32_t i = 0; i < count; ++i) {
    const double reconstructed =
        static_cast<double>(levels_out[i]) * quant_step;
    const double difference = original[i] - reconstructed;
    distortion += difference * difference;
  }
  double rate_bits = eob_bits[eob];
  std::int32_t previous = 0;
  for (std::int32_t i = 0; i < eob; ++i) {
    const std::int32_t level = levels_out[i];
    const std::int32_t is_eob = (i == eob - 1 && level != 0) ? 1 : 0;
    const double bits =
        dec_rdoq_coefficient_bits(tables, info, positions, i, level, is_eob);
    if (!dec_rate_valid(bits)) {
      result.status = -1;
      return result;
    }
    rate_bits += bits;
    previous = dec_abs_i32(level) < cap ? dec_abs_i32(level) : cap;
  }
  const double cost = distortion + lambda * rate_bits;
  if (!DEC_ISFINITE(distortion) || !DEC_ISFINITE(rate_bits) ||
      !dec_rate_valid(rate_bits) || !DEC_ISFINITE(cost)) {
    result.status = -1;
    return result;
  }
  result.eob = eob;
  result.distortion = distortion;
  result.rate_bits = rate_bits;
  result.cost = cost;
  result.max_level_searched = max_level_searched;
  result.level_range_truncated = truncated ? 1 : 0;
  result.status = 0;
  (void)previous;
  return result;
}

// §9.3 trellis-coded quantisation (`tcq::quantize`), one luma block.
// `base_out`/`quant_out`/`states_out` are scan-order outputs of
// `num_coefficients` entries; `dp_scratch` is `num_coefficients * 8` choices.
DEC_HD DecTcqResult dec_tcq_quantize(const DecTables& tables,
                                     const DecBlockInfo& info,
                                     const DecPosition* positions,
                                     const double* original,
                                     const std::int32_t* pre_max,
                                     const std::int32_t* level_context,
                                     std::int32_t num_coefficients,
                                     const DecTcqConfig& config,
                                     const DecTrellisScratch& scratch,
                                     std::int32_t* base_out,
                                     std::int32_t* quant_out,
                                     std::int32_t* states_out,
                                     DecChoice* dp_scratch) {
  DecTcqResult result;
  const std::int32_t n = num_coefficients;
  const double lambda = config.lambda;
  const double quant_step = config.quant_step;
  for (std::int32_t c = 0; c < n; ++c) {
    base_out[c] = 0;
    quant_out[c] = 0;
    states_out[c] = 0;
  }
  (void)level_context;
  if (!(DEC_ISFINITE(lambda) && lambda >= 0.0 && DEC_ISFINITE(quant_step) &&
        quant_step > 0.0) ||
      n < 0 || n > kDecMaxArea) {
    result.status = -1;
    return result;
  }
  if (n == 0) return result;

  std::int32_t* bounds = scratch.bounds;
  bool truncated = false;
  for (std::int32_t c = 0; c < n; ++c) {
    if (!DEC_ISFINITE(original[c])) {
      result.status = -1;
      return result;
    }
    bool clipped = false;
    if (!dec_tcq_level_bound(original[c], quant_step, config.level_cap,
                             config.allow_zero_to_one, &bounds[c], &clipped)) {
      result.status = -1;
      return result;
    }
    truncated = truncated || clipped;
  }

  const double half_step = quant_step * 0.5;
  DecChoice empty_suffix = dec_choice_zero();
  std::int32_t max_level_searched = 0;
  for (std::int32_t c = 0; c < n; ++c) {
    const double value = original[c];
    const int sign = value < 0.0 ? -1 : 1;
    const std::int32_t bound = bounds[c];
    const std::int32_t premax = pre_max[c] > 0 ? pre_max[c] : 6;
    max_level_searched = bound > max_level_searched ? bound : max_level_searched;
    for (std::int32_t s = 0; s < kDecStateCount; ++s) {
      DecChoice best = dec_choice_infinite();
      for (std::int32_t q = 0; q <= bound; ++q) {
        const bool nonzero = q > 0;
        const std::int32_t syntax_level =
            q == 0 ? 0 : dec_syntax_level_for(q, premax);
        const std::int32_t q0 = nonzero ? dec_state_q0(s) : 0;
        const std::int32_t magnitude = nonzero ? 2 * q - q0 : 0;
        const std::int32_t base = nonzero ? sign * q : 0;
        const std::int32_t quant = nonzero ? sign * magnitude : 0;
        const std::int32_t next = dec_next_state(s, syntax_level & 1);
        const double difference =
            value - static_cast<double>(quant) * half_step;
        const double local_dist = difference * difference;
        const double local_rate = dec_tcq_coefficient_bits(
            tables, info, positions, c, base, s, 0);
        if (!dec_rate_valid(local_rate)) {
          result.status = -1;
          return result;
        }
        const DecChoice& suffix = c > 0 ? dp_scratch[(c - 1) * kDecStateCount +
                                                     next]
                                        : empty_suffix;
        DecChoice candidate;
        candidate.cost = local_dist + lambda * local_rate + suffix.cost;
        candidate.distortion = local_dist + suffix.distortion;
        candidate.rate_bits = local_rate + suffix.rate_bits;
        candidate.base_level = base;
        candidate.quant_level = quant;
        candidate.next_state = next;
        candidate.abs_level = q;
        candidate.mode = 0;
        candidate.terminal = 0;
        if (dec_tcq_choice_better(candidate, best)) best = candidate;
      }
      dp_scratch[c * kDecStateCount + s] = best;
    }
  }

  const double all_zero_rate = dec_eob_bits(tables, info, 0, 1);
  if (!dec_rate_valid(all_zero_rate)) {
    result.status = -1;
    return result;
  }

  double* tail = scratch.tail;
  tail[n] = 0.0;
  for (std::int32_t c = n - 1; c >= 0; --c) {
    tail[c] = original[c] * original[c] + tail[c + 1];
  }
  const double all_zero_distortion = tail[0];
  DecChoice best = dec_choice_zero();
  best.cost = all_zero_distortion + all_zero_rate * lambda;
  best.distortion = all_zero_distortion;
  best.rate_bits = all_zero_rate;
  std::int32_t start_position = -1;

  for (std::int32_t c = 0; c < n; ++c) {
    const double value = original[c];
    const int sign = value < 0.0 ? -1 : 1;
    const std::int32_t bound = bounds[c];
    const std::int32_t premax = pre_max[c] > 0 ? pre_max[c] : 6;
    for (std::int32_t q = 1; q <= bound; ++q) {
      const std::int32_t syntax_level = dec_syntax_level_for(q, premax);
      const std::int32_t q0 = dec_state_q0(0);
      const std::int32_t magnitude = 2 * q - q0;
      const std::int32_t base = sign * q;
      const std::int32_t quant = sign * magnitude;
      const std::int32_t next = dec_next_state(0, syntax_level & 1);
      const double difference = value - static_cast<double>(quant) * half_step;
      const double local_dist = difference * difference;
      const double local_rate = dec_tcq_coefficient_bits(
          tables, info, positions, c, base, 0, 1);
      if (!dec_rate_valid(local_rate)) {
        result.status = -1;
        return result;
      }
      const double eob_rate = dec_eob_bits(tables, info, c + 1, 1);
      if (!dec_rate_valid(eob_rate)) {
        result.status = -1;
        return result;
      }
      const DecChoice& suffix =
          c > 0 ? dp_scratch[(c - 1) * kDecStateCount + next] : empty_suffix;
      const double tail_distortion = tail[c + 1];
      DecChoice candidate;
      candidate.cost = local_dist + tail_distortion +
                       lambda * (local_rate + eob_rate) + suffix.cost;
      candidate.distortion = local_dist + tail_distortion + suffix.distortion;
      candidate.rate_bits = local_rate + eob_rate + suffix.rate_bits;
      candidate.base_level = base;
      candidate.quant_level = quant;
      candidate.next_state = next;
      candidate.abs_level = q;
      candidate.mode = 0;
      candidate.terminal = 0;
      const bool better =
          dec_tcq_choice_better(candidate, best) ||
          (start_position >= 0 && !dec_tcq_choice_better(best, candidate) &&
           c < start_position);
      if (better) {
        best = candidate;
        start_position = c;
      }
    }
  }

  result.distortion = best.distortion;
  result.rate_bits = best.rate_bits;
  result.cost = best.cost;
  result.max_level_searched = max_level_searched;
  result.level_range_truncated = truncated ? 1 : 0;
  if (start_position < 0) {
    result.eob = 0;
    result.status = 0;
    return result;
  }
  result.eob = start_position + 1;
  base_out[start_position] = best.base_level;
  quant_out[start_position] = best.quant_level;
  states_out[start_position] = 0;
  std::int32_t state = best.next_state;
  for (std::int32_t c = start_position - 1; c >= 0; --c) {
    const DecChoice& choice = dp_scratch[c * kDecStateCount + state];
    base_out[c] = choice.base_level;
    quant_out[c] = choice.quant_level;
    states_out[c] = state;
    state = choice.next_state;
  }
  result.status = 0;
  return result;
}

#undef DEC_TABLE

}  // namespace av2enc::cuda::still::decide
