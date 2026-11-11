// AV2 v1.0.0 temporal inter tools — OPFL implementation.

#include "encoder/inter/temporal/opfl.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "encoder/inter/temporal/temporal_math.h"

namespace av2enc::inter::temporal {
namespace {

std::size_t out_index(int stride4, int row4, int col4, int list, int comp) {
  return ((static_cast<std::size_t>(row4) * static_cast<std::size_t>(stride4) +
           static_cast<std::size_t>(col4)) *
              2 +
          static_cast<std::size_t>(list)) *
             2 +
         static_cast<std::size_t>(comp);
}

}  // namespace

bool opfl_allowed_for_refs(int d0, int d1, bool frame_is_switch,
                           bool ref0_scaled, bool ref1_scaled) {
  if (frame_is_switch || ref0_scaled || ref1_scaled) {
    return false;
  }
  return (d0 <= 0) != (d1 <= 0);
}

TemporalStatus opfl_compute(const OpflJob& job, const OpflOutput& out) {
  if (job.w <= 0 || job.h <= 0 || job.p0 == nullptr || job.p1 == nullptr ||
      job.p0_stride < job.w || job.p1_stride < job.w ||
      (job.n != 4 && job.n != 8) || job.w % job.n != 0 ||
      job.h % job.n != 0 || out.opfl_mvs == nullptr ||
      out.opfl_stride4 <= 0 || out.mv_deltas == nullptr ||
      out.delta_stride4 <= 0) {
    return TemporalStatus::kInvalidArgument;
  }
  int dist[2] = {0, 0};
  const int d0 = get_relative_dist(job.order_hint, job.ref0_order_hint);
  const int d1 = get_relative_dist(job.order_hint, job.ref1_order_hint);
  if (!opfl_scale_distances(d0, d1, dist)) {
    return TemporalStatus::kOk;  // §7.13.3.9 terminates immediately
  }

  const int w = job.w;
  const int h = job.h;
  std::vector<std::int32_t> tmp(static_cast<std::size_t>(w) *
                                static_cast<std::size_t>(h));
  std::vector<std::int32_t> pdiff(tmp.size());
  std::vector<std::int32_t> x_grad(tmp.size());
  std::vector<std::int32_t> y_grad(tmp.size());

  // §7.13.3.10 optflow_diff().
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const std::size_t idx =
          static_cast<std::size_t>(i) * static_cast<std::size_t>(w) +
          static_cast<std::size_t>(j);
      const int src0 =
          job.p0[static_cast<std::size_t>(i) *
                     static_cast<std::size_t>(job.p0_stride) +
                 static_cast<std::size_t>(j)];
      const int src1 =
          job.p1[static_cast<std::size_t>(i) *
                     static_cast<std::size_t>(job.p1_stride) +
                 static_cast<std::size_t>(j)];
      // Round2Signed(..., BitDepth - 8) is a no-op for 8-bit.
      tmp[idx] = dist[0] * src0 - dist[1] * src1;
      pdiff[idx] = src0 - src1;
    }
  }

  // §7.13.3.11 compute_grad().
  for (int i = 0; i < h; ++i) {
    const int i_start = (i >> kOpflGradUnitLog2) << kOpflGradUnitLog2;
    const int i_end = std::min(i_start + kOpflGradUnit, h) - 1;
    for (int j = 0; j < w; ++j) {
      const int j_start = (j >> kOpflGradUnitLog2) << kOpflGradUnitLog2;
      const int j_end = std::min(j_start + kOpflGradUnit, w) - 1;
      const int j_prev = std::max(j - 1, j_start);
      const int j_prev2 = std::max(j - 2, j_start);
      const int j_next = std::min(j + 1, j_end);
      const int j_next2 = std::min(j + 2, j_end);
      std::int32_t temp =
          42 * (tmp[static_cast<std::size_t>(i) * static_cast<std::size_t>(w) +
                     static_cast<std::size_t>(j_next)] -
                 tmp[static_cast<std::size_t>(i) * static_cast<std::size_t>(w) +
                     static_cast<std::size_t>(j_prev)]) -
          5 * (tmp[static_cast<std::size_t>(i) * static_cast<std::size_t>(w) +
                   static_cast<std::size_t>(j_next2)] -
                tmp[static_cast<std::size_t>(i) * static_cast<std::size_t>(w) +
                    static_cast<std::size_t>(j_prev2)]);
      if (j + 1 > j_end || j - 1 < j_start) {
        temp <<= 1;
      }
      x_grad[static_cast<std::size_t>(i) * static_cast<std::size_t>(w) +
             static_cast<std::size_t>(j)] = round2_signed(temp, 7);

      const int i_prev = std::max(i - 1, i_start);
      const int i_prev2 = std::max(i - 2, i_start);
      const int i_next = std::min(i + 1, i_end);
      const int i_next2 = std::min(i + 2, i_end);
      temp = 42 * (tmp[static_cast<std::size_t>(i_next) *
                           static_cast<std::size_t>(w) +
                       static_cast<std::size_t>(j)] -
                   tmp[static_cast<std::size_t>(i_prev) *
                           static_cast<std::size_t>(w) +
                       static_cast<std::size_t>(j)]) -
             5 * (tmp[static_cast<std::size_t>(i_next2) *
                          static_cast<std::size_t>(w) +
                      static_cast<std::size_t>(j)] -
                  tmp[static_cast<std::size_t>(i_prev2) *
                          static_cast<std::size_t>(w) +
                      static_cast<std::size_t>(j)]);
      if (i + 1 > i_end || i - 1 < i_start) {
        temp <<= 1;
      }
      y_grad[static_cast<std::size_t>(i) * static_cast<std::size_t>(w) +
             static_cast<std::size_t>(j)] = round2_signed(temp, 7);
    }
  }

  // §7.13.3.8 initialises OpflMvs to 2 * the starting MVs and clears
  // MvDeltas before the solve.  A det <= 0 block keeps these values.
  for (int r4 = 0; r4 < h / 4; ++r4) {
    for (int c4 = 0; c4 < w / 4; ++c4) {
      const int row4 = (job.opt_y >> 2) + r4;
      const int col4 = (job.opt_x >> 2) + c4;
      for (int list = 0; list < 2; ++list) {
        for (int comp = 0; comp < 2; ++comp) {
          const int base =
              comp == 0 ? job.mvs[list].row : job.mvs[list].col;
          out.opfl_mvs[out_index(out.opfl_stride4, row4, col4, list, comp)] =
              static_cast<std::int16_t>(clip3(-(1 << 17), (1 << 17) - 1,
                                              base * 2));
          out.mv_deltas[out_index(out.delta_stride4, row4, col4, list, comp)] =
              0;
        }
      }
    }
  }

  // §7.13.3.12 compute_opfl_mv().
  const int n = job.n;
  for (int i_base = 0; i_base < h; i_base += n) {
    for (int j_base = 0; j_base < w; j_base += n) {
      std::int64_t su2 = 0;
      std::int64_t sv2 = 0;
      std::int64_t suv = 0;
      std::int64_t suw = 0;
      std::int64_t svw = 0;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          const std::size_t idx =
              static_cast<std::size_t>(i_base + i) * static_cast<std::size_t>(w) +
              static_cast<std::size_t>(j_base + j);
          const std::int64_t u = x_grad[idx];
          const std::int64_t v = y_grad[idx];
          const std::int64_t e = pdiff[idx];
          su2 += u * u;
          suv += u * v;
          sv2 += v * v;
          suw += u * e;
          svw += v * e;
        }
      }
      su2 += n * n;
      sv2 += n * n;

      const int msb_su2 = 1 + get_msb(static_cast<std::uint32_t>(su2));
      const int msb_sv2 = 1 + get_msb(static_cast<std::uint32_t>(sv2));
      const int msb_suv = 1 + get_msb(static_cast<std::uint32_t>(abs_int(static_cast<int>(suv))));
      const int msb_suw = 1 + get_msb(static_cast<std::uint32_t>(abs_int(static_cast<int>(suw))));
      const int msb_svw = 1 + get_msb(static_cast<std::uint32_t>(abs_int(static_cast<int>(svw))));
      const int max_mult_msb =
          std::max(std::max(msb_su2 + msb_sv2,
                            std::max(msb_sv2 + msb_suw, msb_suv + msb_svw)),
                   std::max(msb_su2 + msb_svw, msb_suv + msb_suw));
      const int redbit = std::max(0, max_mult_msb - kMaxLsBits + 3) >> 1;
      su2 = round2_signed64(su2, redbit);
      sv2 = round2_signed64(sv2, redbit);
      suv = round2_signed64(suv, redbit);
      suw = round2_signed64(suw, redbit);
      svw = round2_signed64(svw, redbit);
      const std::int64_t det = su2 * sv2 - suv * suv;
      if (det <= 0) {
        continue;
      }
      const int bits = kMvRefinePrecBits - 1;
      const int sol[2] = {
          static_cast<int>(sv2 * suw - suv * svw),
          static_cast<int>(su2 * svw - suv * suw)};
      int solved[2] = {0, 0};
      divide_and_round_array(sol, static_cast<int>(det), bits, solved);
      int vx0 = -solved[0];
      int vy0 = -solved[1];
      int vx1 = vx0 * dist[1];
      int vy1 = vy0 * dist[1];
      vx0 = vx0 * dist[0];
      vy0 = vy0 * dist[0];
      const int mv_delta[2][2] = {
          {clip3(-kOpflMvDeltaLimit, kOpflMvDeltaLimit, vy0),
           clip3(-kOpflMvDeltaLimit, kOpflMvDeltaLimit, vx0)},
          {clip3(-kOpflMvDeltaLimit, kOpflMvDeltaLimit, vy1),
           clip3(-kOpflMvDeltaLimit, kOpflMvDeltaLimit, vx1)}};
      const int row4 = (job.opt_y + i_base) >> 2;
      const int col4 = (job.opt_x + j_base) >> 2;
      for (int list = 0; list < 2; ++list) {
        for (int comp = 0; comp < 2; ++comp) {
          const int delta = mv_delta[list][comp];
          const std::size_t idx =
              out_index(out.delta_stride4, row4, col4, list, comp);
          out.mv_deltas[idx] = static_cast<std::int16_t>(delta);
          const int base = comp == 0 ? job.mvs[list].row : job.mvs[list].col;
          const int new_comp = base * 2 + delta;
          const std::size_t oidx =
              out_index(out.opfl_stride4, row4, col4, list, comp);
          out.opfl_mvs[oidx] = static_cast<std::int16_t>(
              clip3(-(1 << 17), (1 << 17) - 1, new_comp));
        }
      }
    }
  }
  return TemporalStatus::kOk;
}

}  // namespace av2enc::inter::temporal
