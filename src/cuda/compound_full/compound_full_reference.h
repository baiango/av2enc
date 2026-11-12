#ifndef AV2ENC_CUDA_COMPOUND_FULL_COMPOUND_FULL_REFERENCE_H_
#define AV2ENC_CUDA_COMPOUND_FULL_COMPOUND_FULL_REFERENCE_H_

// Independent host reference for the 10-bit paths, used only by the tests.
//
// The CPU advanced module (`inter::adv::compound_common.h`) hard-clips
// compound output to 8 bits, so it cannot validate a 10-bit implementation.
// The functions here restate the §7.13.3.14/.27-.30 and §7.13.3.1 formulas
// with `Clip1(..., BitDepth)` and read/write uint16_t samples. They share only
// the wedge mask and rounding helpers, which are bit-depth independent and are
// already pinned by the CPU advanced tests.
//
// This is test code. It is not part of the GPU API and is never compiled into
// the product path.

#include <cstddef>
#include <cstdint>

#include "compound_full.h"

#include "encoder/inter/mc/mc_core.h"

// Test-only host code: keep it out of nvcc's device pass entirely so host
// table references (adv::tables::*) can never leak into device code.
#if !defined(__CUDA_ARCH__)

namespace compound_full_ref {

namespace adv = av2enc::inter::adv;
namespace mc = av2enc::inter::mc;

inline int clip1_bd(int value, int bit_depth) {
  return adv::clip3(0, (1 << bit_depth) - 1, value);
}

inline int load(const void* base, std::size_t index, int is_16bit) {
  return is_16bit
             ? static_cast<int>(static_cast<const std::uint16_t*>(base)[index])
             : static_cast<int>(static_cast<const std::uint8_t*>(base)[index]);
}

inline void store(void* base, std::size_t index, int is_16bit, int value) {
  if (is_16bit) {
    static_cast<std::uint16_t*>(base)[index] =
        static_cast<std::uint16_t>(value);
  } else {
    static_cast<std::uint8_t*>(base)[index] = static_cast<std::uint8_t>(value);
  }
}

// §7.13.3.14 get_mask().
inline int implicit_mask(const av2enc::cuda::compound_full::BlendJob& job,
                         int i, int j) {
  const int last_x = (job.mi_cols * adv::kMiSize >> job.sub_x) - 1;
  const int last_y = (job.mi_rows * adv::kMiSize >> job.sub_y) - 1;
  const int x0 = job.ref_start_x[0] + j;
  const int x1 = job.ref_start_x[1] + j;
  const int y0 = job.ref_start_y[0] + i;
  const int y1 = job.ref_start_y[1] + i;
  const bool on0 = x0 >= 0 && x0 <= last_x && y0 >= 0 && y0 <= last_y;
  const bool on1 = x1 >= 0 && x1 <= last_x && y1 >= 0 && y1 <= last_y;
  if (on0 && !on1) return 2;
  if (on1 && !on0) return 0;
  return 1;
}

inline int diffwtd(int p0, int p1, int mask_type, int bit_depth) {
  int diff = adv::abs_int(p0 - p1);
  diff = adv::round2(diff, (bit_depth - 8) + adv::kCompoundPostRound);
  const int m = adv::clip3(0, 64, 38 + diff / 16);
  return mask_type ? (64 - m) : m;
}

// Host transcription of §7.13.3.27 (host tables are used directly; this
// header is excluded from the device pass).
inline int ref_wedge_mask(int bsize, int sign, int index, int i, int j) {
  using namespace av2enc::inter::adv;
  const int w = tables::kWedgeBlockWidth[bsize];
  const int h = tables::kWedgeBlockHeight[bsize];
  const int boundary = bsize <= tables::kBlock16x16 ? 0 : 1;
  const int dir = tables::kWedgeCodebook[index][0];
  const int xoff = 64 - ((tables::kWedgeCodebook[index][1] * w) >> 3);
  const int yoff = 64 - ((tables::kWedgeCodebook[index][2] * h) >> 3);
  const int y = (((yoff + i) << 1) - 128 + 1) * tables::kWedgeSinLut[dir];
  int d = (((xoff + j) << 1) - 128 + 1) * tables::kWedgeCosLut[dir] + y;
  if (boundary == 0) {
    d *= 2;
  }
  const int cd = adv::clip3(-31, 31, d);
  const int weight = cd >= 0 ? tables::kPosDist2BldWeight[cd]
                             : tables::kNegDist2BldWeight[-cd];
  const int master = weight << 2;
  return sign ? (64 - master) : master;
}

// Host transcription of §7.13.3.29.
inline int ref_intra_mask(av2enc::inter::adv::InterIntraMode mode, int w,
                          int h, int i, int j) {
  using namespace av2enc::inter::adv;
  const int scale = 128 / (h > w ? h : w);
  switch (mode) {
    case InterIntraMode::kVertical:
      return tables::kIiWeights1d[i * scale];
    case InterIntraMode::kHorizontal:
      return tables::kIiWeights1d[j * scale];
    case InterIntraMode::kSmooth:
      return tables::kIiWeights1d[(i < j ? i : j) * scale];
    default:
      return 32;
  }
}

inline int luma_mask(
    const av2enc::cuda::compound_full::BlendJob& job, int i, int j) {
  using namespace av2enc::cuda::compound_full;
  switch (job.type) {
    case kCompoundWedge:
      return ref_wedge_mask(job.bsize, job.wedge_sign, job.wedge_index, i,
                            j);
    case kCompoundDiffWtd:
      return diffwtd(job.p0[i * job.p0_stride + j],
                     job.p1[i * job.p1_stride + j], job.mask_type,
                     job.bit_depth);
    case kCompoundIntra:
      return ref_intra_mask(
          static_cast<adv::InterIntraMode>(job.interintra_mode), job.rect_w,
          job.rect_h, i, j);
    default:
      return 32;
  }
}

inline int mask_sample(
    const av2enc::cuda::compound_full::BlendJob& job, int i, int j) {
  using namespace av2enc::cuda::compound_full;
  const bool from_luma =
      job.type == kCompoundWedge || job.type == kCompoundDiffWtd;
  if (job.plane == 0 || !from_luma) {
    return luma_mask(job, i, j);
  }
  if (job.type == kCompoundDiffWtd && job.luma_p0 != nullptr &&
      job.luma_p1 != nullptr && job.luma_mask == nullptr) {
    const int m00 = diffwtd(job.luma_p0[(2 * i) * job.luma_p0_stride + 2 * j],
                            job.luma_p1[(2 * i) * job.luma_p1_stride + 2 * j],
                            job.mask_type, job.bit_depth);
    const int m01 =
        diffwtd(job.luma_p0[(2 * i) * job.luma_p0_stride + 2 * j + 1],
                job.luma_p1[(2 * i) * job.luma_p1_stride + 2 * j + 1],
                job.mask_type, job.bit_depth);
    if (!job.sub_y) {
      return adv::mask_subsample(m00, m01, 0, 0, job.sub_x, job.sub_y);
    }
    const int m10 =
        diffwtd(job.luma_p0[(2 * i + 1) * job.luma_p0_stride + 2 * j],
                job.luma_p1[(2 * i + 1) * job.luma_p1_stride + 2 * j],
                job.mask_type, job.bit_depth);
    const int m11 =
        diffwtd(job.luma_p0[(2 * i + 1) * job.luma_p0_stride + 2 * j + 1],
                job.luma_p1[(2 * i + 1) * job.luma_p1_stride + 2 * j + 1],
                job.mask_type, job.bit_depth);
    return adv::mask_subsample(m00, m01, m10, m11, job.sub_x, job.sub_y);
  }
  if (job.luma_mask == nullptr) {
    return 32;
  }
  const int stride = job.luma_mask_stride;
  if (!job.sub_x) {
    return job.luma_mask[i * stride + j];
  }
  if (!job.sub_y) {
    return adv::mask_subsample(job.luma_mask[i * stride + 2 * j],
                               job.luma_mask[i * stride + 2 * j + 1], 0, 0,
                               job.sub_x, job.sub_y);
  }
  return adv::mask_subsample(job.luma_mask[(2 * i) * stride + 2 * j],
                             job.luma_mask[(2 * i) * stride + 2 * j + 1],
                             job.luma_mask[(2 * i + 1) * stride + 2 * j],
                             job.luma_mask[(2 * i + 1) * stride + 2 * j + 1],
                             job.sub_x, job.sub_y);
}

// Runs one blend on host buffers. Writes the destination and, for plane 0,
// the mask (when `mask_out` is set).
inline void blend_job(const av2enc::cuda::compound_full::BlendJob& job) {
  using namespace av2enc::cuda::compound_full;
  for (int i = 0; i < job.rect_h; ++i) {
    for (int j = 0; j < job.rect_w; ++j) {
      const int p0 = job.p0 ? job.p0[i * job.p0_stride + j] : 0;
      const int p1 = job.p1 ? job.p1[i * job.p1_stride + j] : 0;
      const std::size_t dst_index =
          static_cast<std::size_t>(job.rect_y + i) * job.dst_stride +
          job.rect_x + j;
      const int intra =
          job.intra ? load(job.intra,
                           static_cast<std::size_t>(i) * job.intra_stride + j,
                           job.intra_16bit)
                    : 0;
      const int inter = load(job.dst, dst_index, job.dst_16bit);
      int value;
      if (job.is_inter_intra) {
        const int m =
            job.type == kCompoundWedge
                ? mask_sample(job, i, j)
                : ref_intra_mask(
                      static_cast<adv::InterIntraMode>(job.interintra_mode),
                      job.rect_w, job.rect_h, i, j);
        value = adv::round2(m * intra + (64 - m) * inter, 6);
      } else if (job.type == kCompoundAverage) {
        if (job.imp_mask_eligible) {
          const int m = implicit_mask(job, i, j);
          value = clip1_bd(
              adv::round2(m * p0 + (2 - m) * p1, 1 + adv::kCompoundPostRound),
              job.bit_depth);
        } else {
          const int cw = job.cwp_weight;
          value = clip1_bd(adv::round2(cw * p0 + (16 - cw) * p1,
                                       4 + adv::kCompoundPostRound),
                           job.bit_depth);
        }
      } else {
        const int m = mask_sample(job, i, j);
        value = clip1_bd(adv::round2(m * p0 + (64 - m) * p1,
                                     6 + adv::kCompoundPostRound),
                         job.bit_depth);
      }
      store(job.dst, dst_index, job.dst_16bit, value);
    }
    if (job.mask_out != nullptr && job.plane == 0) {
      for (int j = 0; j < job.rect_w; ++j) {
        job.mask_out[i * job.mask_out_stride + j] = static_cast<std::uint8_t>(
            job.type == kCompoundWedge
                ? ref_wedge_mask(job.bsize, job.wedge_sign, job.wedge_index,
                                 i, j)
                : luma_mask(job, i, j));
      }
    }
  }
}

// Independent 10-bit skip-mode prediction: two references, same MV, average.
inline void skip_job(const av2enc::cuda::compound_full::SkipJob& job) {
  using av2enc::cuda::compound_full::RefPlane;
  for (int plane = 0; plane < 3; ++plane) {
    const int sub_x = plane > 0 ? 1 : 0;
    const int sub_y = plane > 0 ? 1 : 0;
    const int px = job.x >> sub_x;
    const int py = job.y >> sub_y;
    const int pw = job.w >> sub_x;
    const int ph = job.h >> sub_y;
    const RefPlane& r0 = job.ref[0][plane];
    const RefPlane& r1 = job.ref[1][plane];
    const mc::McSampleLocation loc = mc::mc_derive_sample_location(
        px, py, mc::Mv{job.mv_row, job.mv_col}, sub_x, sub_y);
    const mc::InterpFilter fh =
        mc::mc_pass_filter(static_cast<mc::InterpFilter>(job.filter), pw <= 4);
    const mc::InterpFilter fv =
        mc::mc_pass_filter(static_cast<mc::InterpFilter>(job.filter), ph <= 4);
    auto sample = [&](const RefPlane& ref, int r, int c) {
      const int p_x = loc.start_x + loc.step_x * c;
      const int col = (p_x >> mc::kScaleSubpelBits) - 3;
      const int phase_h = (p_x >> 6) & mc::kSubpelMask;
      const int p_y = (loc.start_y & 1023) + loc.step_y * r;
      const int row_base = p_y >> mc::kScaleSubpelBits;
      const int phase_v = (p_y >> 6) & mc::kSubpelMask;
      const std::int16_t* fh_taps =
          &mc::kSubpelFilters[static_cast<int>(fh)][phase_h][0];
      const std::int16_t* fv_taps =
          &mc::kSubpelFilters[static_cast<int>(fv)][phase_v][0];
      int sum_v = 0;
      for (int t = 0; t < 8; ++t) {
        const int row = adv::clip3(
            0, ref.height - 1,
            (loc.start_y >> mc::kScaleSubpelBits) + row_base + t - 3);
        int hsum = 0;
        for (int u = 0; u < 8; ++u) {
          const int cx = adv::clip3(0, ref.width - 1, col + u);
          hsum += static_cast<int>(fh_taps[u]) *
                  load(ref.data,
                       static_cast<std::size_t>(row) * ref.stride + cx,
                       ref.is_16bit);
        }
        const int intermediate = mc::mc_round2(hsum, mc::kInterRound0);
        sum_v += static_cast<int>(fv_taps[t]) * intermediate;
      }
      return mc::mc_round2(sum_v, adv::kInterRound1Compound);
    };
    for (int r = 0; r < ph; ++r) {
      for (int c = 0; c < pw; ++c) {
        const int p0 = sample(r0, r, c);
        const int p1 = sample(r1, r, c);
        const int value =
            clip1_bd(adv::round2(job.cwp_weight * p0 +
                                     (16 - job.cwp_weight) * p1,
                                 4 + adv::kCompoundPostRound),
                     job.bit_depth);
        store(job.dst[plane],
              static_cast<std::size_t>(py + r) * job.dst_stride[plane] + px + c,
              job.dst_16bit[plane], value);
      }
    }
  }
}

}  // namespace compound_full_ref

#endif  // !defined(__CUDA_ARCH__)

#endif  // AV2ENC_CUDA_COMPOUND_FULL_COMPOUND_FULL_REFERENCE_H_
