// Shared host/device implementation of the AV2 morphological prediction
// process (§7.13.3.26 -> §7.13.3.25 with morphPred = 1, plane = 0).
//
// This header is the single arithmetic transcription. It is compiled:
//   * into the CUDA kernels (morph.cu), as host+device inline functions;
//   * into the host emulation test, with the CUDA keywords stubbed, and
//     compared byte-for-byte against src/encoder/inter/morph/morph_common.h.
//
// Every step mirrors the CPU reference exactly: to_fullmv, the 12 -> 8
// template-extent rule, the plan's availability branches, the Div_Lut
// resolve_division approximation, `nor`/`der` as int32, the beta expression,
// and the final per-sample `clip1((orig*alpha + beta) >> 8, 8)` over the full
// w*h rectangle.
//
// Scope is the CPU reference's: luma, 8-bit. ctx.bit_depth != 8 is rejected by
// morph_derive_device() exactly like morph_reference_geometry().

#ifndef AV2ENC_CUDA_MORPH_MORPH_DEVICE_CUH_
#define AV2ENC_CUDA_MORPH_MORPH_DEVICE_CUH_

#include "cuda/codec/inter/morph/morph.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define AV2ENC_MORPH_HD __host__ __device__ __forceinline__
#else
#define AV2ENC_MORPH_HD inline
#endif

namespace av2enc::cuda::morph {
namespace detail {

inline constexpr int kMiSize = 4;
inline constexpr int kShift = 8;
inline constexpr int kDivLutBits = 7;
inline constexpr int kDivLutPrecBits = 9;

// §4.8 Clip3.
AV2ENC_MORPH_HD int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

// §4.8 Clip1 for the reference's 8-bit output.
AV2ENC_MORPH_HD int clip1_8(int value) {
  return clip3(0, 255, value);
}

// §4.8 FloorLog2.
AV2ENC_MORPH_HD int floor_log2(int value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

// §4.8 Round2.
AV2ENC_MORPH_HD int round2(int value, int n) {
  if (n <= 0) {
    return value;
  }
  return (value + (1 << (n - 1))) >> n;
}

// §7.13.3.25: 1/8-pel to integer luma samples, bias 3 plus 1 for non-negative
// values (round half up).
AV2ENC_MORPH_HD int to_fullmv(int mv) {
  return (mv + 3 + ((mv >= 0) ? 1 : 0)) >> 3;
}

// §7.13.3.22 Div_Lut entry.
AV2ENC_MORPH_HD int div_lut_entry(int f) {
  const int den = 128 + f;
  return (65536 + den / 2) / den;
}

// §7.13.3.25 resolve_division. D is always > 0 at the call site.
AV2ENC_MORPH_HD int resolve_division(int n_value, int d_value,
                                            int shift) {
  if (d_value <= 0) {
    return 1 << shift;
  }
  const bool sign_n = n_value < 0;
  int n = sign_n ? -n_value : n_value;
  const int shift_n = floor_log2(n);
  const int shift_d = floor_log2(d_value);
  const int e_d = d_value - (1 << shift_d);
  const int f_d = (shift_d > kDivLutBits)
                      ? round2(e_d, shift_d - kDivLutBits)
                      : (e_d << (kDivLutBits - shift_d));
  const int f_n = (shift_n > kDivLutBits)
                      ? round2(n, shift_n - kDivLutBits)
                      : (n << (kDivLutBits - shift_n));
  const int shift_add = shift_d - shift_n - shift;
  int ret = 0;
  if (shift_add <= 1) {
    const int shift0 = kDivLutPrecBits + kDivLutBits + shift_add;
    if (shift0 >= 0) {
      ret = (div_lut_entry(f_d) * f_n) >> shift0;
    } else {
      ret = (2 << shift) - 1;
    }
  }
  ret = ret < ((2 << shift) - 1) ? ret : ((2 << shift) - 1);
  if (sign_n) {
    ret = -ret;
  }
  return ret;
}

// §7.13.3.25 geometry. Mirrors morph_reference_geometry().
AV2ENC_MORPH_HD bool reference_geometry(
    const inter::morph::MorphContext& ctx,
    const inter::morph::MorphRect& rect, int mv_row, int mv_col, int* ref_x,
    int* ref_y, int* bw, int* bh) {
  if (ctx.bit_depth != 8 || ctx.mi_cols <= 0 || ctx.mi_rows <= 0 ||
      rect.w <= 0 || rect.h <= 0 || rect.x < 0 || rect.y < 0) {
    return false;
  }
  const int plane_w = ctx.mi_cols * kMiSize;
  const int plane_h = ctx.mi_rows * kMiSize;
  const int dy = to_fullmv(mv_row);
  const int dx = to_fullmv(mv_col);
  *ref_y = ctx.mi_row * kMiSize + dy;
  *ref_x = ctx.mi_col * kMiSize + dx;
  *bw = plane_w - rect.x;
  if (rect.w < *bw) {
    *bw = rect.w;
  }
  *bh = plane_h - rect.y;
  if (rect.h < *bh) {
    *bh = rect.h;
  }
  return *bw > 0 && *bh > 0;
}

// §7.13.3.25 conformance box.
AV2ENC_MORPH_HD bool reference_in_bounds(int ref_x, int ref_y, int bw,
                                                int bh, int plane_w,
                                                int plane_h) {
  return ref_x >= 1 && ref_y >= 1 && ref_x + bw <= plane_w &&
         ref_y + bh <= plane_h;
}

// §7.13.3.25 luma template plan. Mirrors morph_template_plan().
AV2ENC_MORPH_HD inter::morph::MorphTemplatePlan template_plan(
    int bw, int bh, bool avail_u, bool avail_l) {
  inter::morph::MorphTemplatePlan plan{};
  const int bw2 = bw < 16 ? bw : 16;
  const int bh2 = bh < 16 ? bh : 16;
  plan.width = bw2 == 12 ? 8 : bw2;
  plan.height = bh2 == 12 ? 8 : bh2;
  if (avail_u && avail_l) {
    if (plan.width == 16 && plan.height == 16) {
      plan.num_up = 16;
      plan.num_left = 16;
    } else if (plan.width > 4 && plan.height > 4) {
      plan.num_up = 8;
      plan.num_left = 8;
    } else if (plan.width < 16 && plan.height < 16) {
      plan.num_up = 4;
      plan.num_left = 4;
    } else if (plan.width == 16) {
      plan.num_up = 16;
    } else {
      plan.num_left = 16;
    }
  } else if (avail_u) {
    plan.num_up = plan.width;
  } else if (avail_l) {
    plan.num_left = plan.height;
  }
  if (plan.num_up > 0) {
    plan.up_step = plan.width / plan.num_up;
  }
  if (plan.num_left > 0) {
    plan.left_step = plan.height / plan.num_left;
  }
  return plan;
}

AV2ENC_MORPH_HD int read_sample(const std::uint8_t* frame, int stride,
                                       int x, int y) {
  return static_cast<int>(
      frame[static_cast<std::ptrdiff_t>(y) * static_cast<std::ptrdiff_t>(stride) +
            static_cast<std::ptrdiff_t>(x)]);
}

}  // namespace detail

// Derives the fit only. `frame` must already contain the IBC prediction, as in
// morph_derive(). Returns false for any §7.13.3.25 rejection.
AV2ENC_MORPH_HD bool morph_derive_device(
    const inter::morph::MorphContext& ctx, const std::uint8_t* frame,
    int stride, const inter::morph::MorphRect& rect, int mv_row, int mv_col,
    inter::morph::MorphFit* fit_out) {
  if (fit_out == nullptr || frame == nullptr || stride <= 0) {
    return false;
  }
  if (ctx.avail_u && rect.y < 1) {
    return false;
  }
  if (ctx.avail_l && rect.x < 1) {
    return false;
  }
  inter::morph::MorphFit fit{};
  if (!detail::reference_geometry(ctx, rect, mv_row, mv_col, &fit.ref_x,
                                  &fit.ref_y, &fit.bw, &fit.bh)) {
    return false;
  }
  const int plane_w = ctx.mi_cols * detail::kMiSize;
  const int plane_h = ctx.mi_rows * detail::kMiSize;
  if (!detail::reference_in_bounds(fit.ref_x, fit.ref_y, fit.bw, fit.bh,
                                   plane_w, plane_h)) {
    return false;
  }

  fit.alpha = 1 << detail::kShift;
  fit.beta = -(1 << (detail::kShift - 1));
  fit.plan = detail::template_plan(fit.bw, fit.bh, ctx.avail_u, ctx.avail_l);

  int sum_x = 0;
  int sum_y = 0;
  int sum_xx = 0;
  int sum_xy = 0;
  int count = 0;
  const int x = rect.x;
  const int y = rect.y;
  if (fit.plan.num_up > 0) {
    const int step = fit.plan.up_step;
    for (int i = step >> 1; i < fit.plan.width; i += step) {
      const int recon = detail::read_sample(frame, stride, x + i, y - 1);
      const int ref =
          detail::read_sample(frame, stride, fit.ref_x + i, fit.ref_y - 1);
      sum_x += ref;
      sum_y += recon;
      sum_xy += ref * recon;
      sum_xx += ref * ref;
    }
    count += fit.plan.num_up;
  }
  if (fit.plan.num_left > 0) {
    const int step = fit.plan.left_step;
    for (int i = step >> 1; i < fit.plan.height; i += step) {
      const int recon = detail::read_sample(frame, stride, x - 1, y + i);
      const int ref =
          detail::read_sample(frame, stride, fit.ref_x - 1, fit.ref_y + i);
      sum_x += ref;
      sum_y += recon;
      sum_xy += ref * recon;
      sum_xx += ref * ref;
    }
    count += fit.plan.num_left;
  }

  // morphPred == 1, plane == 0: no explicit-alpha or BawpAlpha branch.
  if (count > 0) {
    const int nor = sum_xy - sum_x * sum_y / count;
    const int der = sum_xx - sum_x * sum_x / count;
    if (der != 0 && nor != 0) {
      fit.alpha = detail::resolve_division(nor, der, detail::kShift);
      if (fit.alpha == 0) {
        fit.alpha = 1 << detail::kShift;
      }
    } else {
      fit.alpha = 1 << detail::kShift;
    }
    fit.beta = ((sum_y << detail::kShift) - sum_x * fit.alpha) / count;
  }
  fit.count = count;
  *fit_out = fit;
  return true;
}

// Runs one job cooperatively in a CUDA block. `fit_slot`/`status_slot` are
// the persistent-array overrides; when null the job's own pointers are used.
// All threads must call this with the same job.
//
// Thread 0 derives the fit because the template reduction is sequential; the
// barrier before the apply loop is required even when the reference block
// overlaps the predicted block (the zero-BV case), because the derive reads
// the frame and the apply writes it.
__device__ __forceinline__ void morph_run_job_device(
    const MorphJob& job, inter::morph::MorphFit* fit_slot, int* status_slot) {
  const int nthr = static_cast<int>(blockDim.x);
  if (nthr <= 0) {
    return;
  }
  __shared__ int shared_status;
  __shared__ int shared_alpha;
  __shared__ int shared_beta;
  __shared__ int shared_dx;
  __shared__ int shared_dy;
  const int tid = static_cast<int>(threadIdx.x);
  if (tid == 0) {
    inter::morph::MorphFit fit{};
    const bool ok = morph_derive_device(job.ctx, job.frame, job.stride,
                                        job.rect, job.mv_row, job.mv_col, &fit);
    shared_status = ok ? 1 : 0;
    shared_alpha = fit.alpha;
    shared_beta = fit.beta;
    shared_dx = detail::to_fullmv(job.mv_col);
    shared_dy = detail::to_fullmv(job.mv_row);
    int* status_out = (status_slot != nullptr) ? status_slot : job.status_out;
    if (status_out != nullptr) {
      *status_out = shared_status;
    }
    if (ok) {
      inter::morph::MorphFit* fit_out =
          (fit_slot != nullptr) ? fit_slot : job.fit_out;
      if (fit_out != nullptr) {
        *fit_out = fit;
      }
    }
  }
  __syncthreads();
  if (shared_status == 0 || job.frame == nullptr || job.rect.w <= 0 ||
      job.rect.h <= 0) {
    return;
  }
  const int w = job.rect.w;
  const int total = w * job.rect.h;
  for (int index = tid; index < total; index += nthr) {
    const int i = index / w;
    const int j = index - i * w;
    std::uint8_t* row = job.frame +
                        static_cast<std::ptrdiff_t>(job.rect.y + i) *
                            static_cast<std::ptrdiff_t>(job.stride);
    const int x = job.rect.x + j;
    const int orig = static_cast<int>(row[x]);
    row[x] = static_cast<std::uint8_t>(
        detail::clip1_8((orig * shared_alpha + shared_beta) >> detail::kShift));
    if (job.field_out != nullptr) {
      const std::size_t offset = static_cast<std::size_t>(index) * 2;
      job.field_out[offset] = static_cast<std::int16_t>(shared_dx);
      job.field_out[offset + 1] = static_cast<std::int16_t>(shared_dy);
    }
  }
}

}  // namespace av2enc::cuda::morph

#endif  // AV2ENC_CUDA_MORPH_MORPH_DEVICE_CUH_
