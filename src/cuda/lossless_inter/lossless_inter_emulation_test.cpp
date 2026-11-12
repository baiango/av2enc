// Host-emulation parity test for the shared AV2 lossless inter device header.
//
// Compiles `lossless_inter_device.cuh` with the CUDA keywords stubbed and
// compares every integer result against the independent CPU reference in
// `src/encoder/lossless/inter/`.  Runs without nvcc or a GPU; the real
// CPU-vs-GPU test is `lossless_inter_test.cu`.
//
// Coverage:
//   * 4-sample WHT forward/inverse, every shift, odd lifting values;
//   * 2D lossless forward/inverse at 4x4, 8x8, 16x16, 32x32 and every
//     rectangular power-of-two combination, DCT_DCT and IDTX/FSC;
//   * DPCM first difference / cumulative sum, 4x4/8x8/rectangular, V and H;
//   * forward residual pipeline (difference before transform);
//   * reconstruction (inverse + cumsum + Clip1(pred + residual)) with
//     strided planes, block offsets and plane-edge writes;
//   * §5.20.6.1 transform-size selection and §5.18.2 lossless derivation.
//
// Build/run: ./build_and_test.sh (or ./build_gpu_test.sh on a CUDA host).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static

static inline void __syncthreads() {}

#include "lossless_inter_device.cuh"

#include "encoder/lossless/inter/lossless_inter.h"

namespace {

namespace li = av2enc::cuda::lossless_inter;
namespace cpu = av2enc::lossless::inter;

int g_checks = 0;
int g_failures = 0;
long long g_cases = 0;

void check(bool condition, const char* what, int a = 0, int b = 0, int c = 0,
           int d = 0) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 30) {
      std::printf("FAIL: %s (%d %d %d %d)\n", what, a, b, c, d);
    }
  }
}

std::uint32_t g_rng = 0x2778f00du;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
int random_range(int low, int high) {
  const std::uint32_t span =
      static_cast<std::uint32_t>(high - low + 1);
  return low + static_cast<int>(next_random() % span);
}

constexpr int kDims[4] = {4, 8, 16, 32};

std::vector<int> make_block(int n, int low, int high) {
  std::vector<int> v(static_cast<std::size_t>(n));
  for (int& value : v) value = random_range(low, high);
  return v;
}

void run_forward(const li::ForwardJob& job, std::vector<int>& scratch) {
  li::li_forward_block_device(job, 0, 1, scratch.data());
}

void run_inverse(const li::InverseJob& job, std::vector<int>& scratch) {
  li::li_inverse_residual_device(
      job.levels, job.residual, job.w, job.h, job.dct_dct, job.dpcm,
      job.vertical, 0, 1, scratch.data());
}

void run_reconstruct(const li::ReconJob& job, std::vector<int>& scratch,
                     std::vector<int>& residual) {
  li::li_reconstruct_block_device(job, 0, 1, scratch.data(), residual.data());
}

// ---------------------------------------------------------------------------
// 1. 4-sample WHT
// ---------------------------------------------------------------------------

void test_wht4() {
  std::printf("-- 4-sample WHT (forward/inverse, shifts, odd values)\n");
  int checks_before = g_failures;
  for (int iter = 0; iter < 4000; ++iter) {
    int in[4];
    // Half the cases are deliberately odd (a1 - d1) values.
    for (int i = 0; i < 4; ++i) {
      in[i] = (iter % 2 == 0) ? random_range(-3000, 3000)
                              : random_range(-3000, 3000) | 1;
    }
    int want[4];
    int got[4];
    cpu::forward_wht4(in, want);
    li::li_wht4_forward(in, got);
    ++g_cases;
    for (int i = 0; i < 4; ++i) {
      check(got[i] == want[i], "wht4 forward", iter, i, got[i], want[i]);
    }

    for (int shift = 0; shift <= 5; ++shift) {
      cpu::inverse_wht4(in, shift, want);
      li::li_wht4_inverse(in, shift, got);
      ++g_cases;
      for (int i = 0; i < 4; ++i) {
        check(got[i] == want[i], "wht4 inverse", shift, i, got[i], want[i]);
      }
    }

    // forward -> inverse(shift 0) must be the identity even for odd values.
    int levels[4];
    int back[4];
    li::li_wht4_forward(in, levels);
    li::li_wht4_inverse(levels, 0, back);
    ++g_cases;
    for (int i = 0; i < 4; ++i) {
      check(back[i] == in[i], "wht4 round trip", iter, i, back[i], in[i]);
    }
  }
  std::printf("   %d checks, %d failures\n", g_checks,
              g_failures - checks_before);
}

// ---------------------------------------------------------------------------
// 2. 2D forward / inverse, all sizes, DCT_DCT and IDTX
// ---------------------------------------------------------------------------

void test_block_transform() {
  std::printf("-- 2D lossless forward/inverse (4x4..32x32, DCT_DCT + IDTX)\n");
  const int checks_before = g_failures;
  std::vector<int> scratch(static_cast<std::size_t>(li::kMaxTransformSamples));
  for (int w : kDims) {
    for (int h : kDims) {
      for (int dct = 0; dct <= 1; ++dct) {
        for (int iter = 0; iter < 8; ++iter) {
          const int n = w * h;
          const int low = (iter < 4) ? -255 : -2047;
          const int high = (iter < 4) ? 255 : 2047;
          std::vector<int> residual = make_block(n, low, high);
          std::vector<int> got_levels(static_cast<std::size_t>(n));
          std::vector<int> want_levels(static_cast<std::size_t>(n));

          li::ForwardJob forward;
          forward.residual = residual.data();
          forward.levels = got_levels.data();
          forward.w = w;
          forward.h = h;
          forward.dct_dct = dct;
          run_forward(forward, scratch);

          cpu::forward_lossless_block(residual.data(), w, h, dct != 0,
                                      want_levels.data());
          ++g_cases;
          check(std::memcmp(got_levels.data(), want_levels.data(),
                            sizeof(int) * static_cast<std::size_t>(n)) == 0,
                "forward_lossless_block", w, h, dct, iter);

          // Device inverse of the CPU levels.
          std::vector<int> got_residual(static_cast<std::size_t>(n));
          std::vector<int> want_residual(static_cast<std::size_t>(n));
          li::InverseJob inverse;
          inverse.levels = want_levels.data();
          inverse.residual = got_residual.data();
          inverse.w = w;
          inverse.h = h;
          inverse.dct_dct = dct;
          run_inverse(inverse, scratch);
          cpu::inverse_lossless_block(want_levels.data(), w, h, dct != 0,
                                      want_residual.data());
          ++g_cases;
          check(std::memcmp(got_residual.data(), want_residual.data(),
                            sizeof(int) * static_cast<std::size_t>(n)) == 0,
                "inverse_lossless_block", w, h, dct, iter);

          // Device forward -> device inverse must reproduce the residual.
          std::vector<int> round_trip(static_cast<std::size_t>(n));
          li::InverseJob round_job;
          round_job.levels = got_levels.data();
          round_job.residual = round_trip.data();
          round_job.w = w;
          round_job.h = h;
          round_job.dct_dct = dct;
          run_inverse(round_job, scratch);
          ++g_cases;
          check(std::memcmp(round_trip.data(), residual.data(),
                            sizeof(int) * static_cast<std::size_t>(n)) == 0,
                "block transform round trip", w, h, dct, iter);
        }
      }
    }
  }
  std::printf("   %d checks, %d failures\n", g_checks,
              g_failures - checks_before);

  // Rectangular sizes (inter lossless transform sizes follow the coding block
  // extent): both DCT_DCT WHT and IDTX/FSC.
  const int rect[][2] = {{4, 8},  {8, 4},  {4, 16},  {16, 4},
                         {8, 16}, {16, 8}, {4, 32},  {32, 4},
                         {16, 32}, {32, 16}};
  for (const auto& dim : rect) {
    const int w = dim[0];
    const int h = dim[1];
    const int n = w * h;
    for (int dct = 0; dct <= 1; ++dct) {
      std::vector<int> residual = make_block(n, -255, 255);
      std::vector<int> got(static_cast<std::size_t>(n));
      std::vector<int> want(static_cast<std::size_t>(n));
      li::ForwardJob forward;
      forward.residual = residual.data();
      forward.levels = got.data();
      forward.w = w;
      forward.h = h;
      forward.dct_dct = dct;
      run_forward(forward, scratch);
      cpu::forward_lossless_block(residual.data(), w, h, dct != 0,
                                  want.data());
      ++g_cases;
      check(got == want, "rect forward", w, h, dct);
      std::vector<int> back(static_cast<std::size_t>(n));
      li::InverseJob inverse;
      inverse.levels = got.data();
      inverse.residual = back.data();
      inverse.w = w;
      inverse.h = h;
      inverse.dct_dct = dct;
      run_inverse(inverse, scratch);
      ++g_cases;
      check(back == residual, "rect round trip", w, h, dct);
    }
  }
}

// ---------------------------------------------------------------------------
// 3. DPCM
// ---------------------------------------------------------------------------

void test_dpcm() {
  std::printf("-- DPCM difference/cumsum (4x4/8x8/rectangular, V + H)\n");
  const int checks_before = g_failures;
  const int sizes[][2] = {{4, 4}, {8, 8}, {4, 8}, {8, 4},
                          {16, 16}, {32, 32}, {4, 32}, {32, 4}};
  for (const auto& dim : sizes) {
    const int w = dim[0];
    const int h = dim[1];
    const int n = w * h;
    for (int vertical = 0; vertical <= 1; ++vertical) {
      for (int iter = 0; iter < 8; ++iter) {
        std::vector<int> in = make_block(n, -255, 255);
        std::vector<int> got_diff(static_cast<std::size_t>(n));
        std::vector<int> want_diff(static_cast<std::size_t>(n));
        std::vector<int> got_cum(static_cast<std::size_t>(n));
        std::vector<int> want_cum(static_cast<std::size_t>(n));

        li::DpcmJob job;
        job.in = in.data();
        job.out = got_diff.data();
        job.w = w;
        job.h = h;
        job.vertical = vertical;
        li::li_dpcm_difference_device(job, 0, 1);
        cpu::dpcm_difference_block(in.data(), w, h, vertical != 0,
                                   want_diff.data());
        ++g_cases;
        check(got_diff == want_diff, "dpcm difference", w, h, vertical, iter);

        job.in = want_diff.data();
        job.out = got_cum.data();
        li::li_dpcm_cumsum_device(job, 0, 1);
        cpu::dpcm_cumsum_block(want_diff.data(), w, h, vertical != 0,
                               want_cum.data());
        ++g_cases;
        check(got_cum == want_cum, "dpcm cumsum", w, h, vertical, iter);

        ++g_cases;
        check(got_cum == in, "dpcm round trip", w, h, vertical, iter);
      }
    }
  }
  std::printf("   %d checks, %d failures\n", g_checks,
              g_failures - checks_before);
}

// ---------------------------------------------------------------------------
// 4. Forward residual pipeline (difference then transform)
// ---------------------------------------------------------------------------

void test_forward_pipeline() {
  std::printf("-- forward residual pipeline (DPCM then WHT/IDTX)\n");
  const int checks_before = g_failures;
  std::vector<int> scratch(static_cast<std::size_t>(li::kMaxTransformSamples));
  const int sizes[][2] = {{4, 4},   {8, 8},   {16, 16}, {32, 32},
                          {4, 8},   {8, 4},   {4, 16},  {16, 4},
                          {8, 16},  {16, 8},  {4, 32},  {32, 4},
                          {16, 32}, {32, 16}};
  for (const auto& dim : sizes) {
    const int w = dim[0];
    const int h = dim[1];
    for (int dct = 0; dct <= 1; ++dct) {
      for (int vertical = 0; vertical <= 1; ++vertical) {
        const int n = w * h;
        std::vector<int> residual = make_block(n, -255, 255);
        std::vector<int> diff(static_cast<std::size_t>(n));
        std::vector<int> want(static_cast<std::size_t>(n));
        std::vector<int> got(static_cast<std::size_t>(n));

        li::ForwardJob job;
        job.residual = residual.data();
        job.levels = got.data();
        job.w = w;
        job.h = h;
        job.dct_dct = dct;
        job.dpcm = 1;
        job.vertical = vertical;
        run_forward(job, scratch);

        cpu::dpcm_difference_block(residual.data(), w, h, vertical != 0,
                                   diff.data());
        cpu::forward_lossless_block(diff.data(), w, h, dct != 0, want.data());
        ++g_cases;
        check(got == want, "forward with dpcm", w, h, dct, vertical);
      }
    }
  }
  std::printf("   %d checks, %d failures\n", g_checks,
              g_failures - checks_before);
}

// ---------------------------------------------------------------------------
// 5. Reconstruction
// ---------------------------------------------------------------------------

void test_reconstruction() {
  std::printf("-- reconstruction (inverse + cumsum + Clip1(pred + residual))\n");
  const int checks_before = g_failures;
  std::vector<int> scratch(static_cast<std::size_t>(li::kMaxTransformSamples));
  std::vector<int> residual_scratch(
      static_cast<std::size_t>(li::kMaxTransformSamples));

  // Block offsets: interior and flush against the plane's last row/column.
  for (int w : kDims) {
    for (int h : kDims) {
      for (int dct = 0; dct <= 1; ++dct) {
        for (int dpcm = 0; dpcm <= 1; ++dpcm) {
          for (int vertical = 0; vertical <= 1; ++vertical) {
            if (dpcm == 0 && vertical == 1) continue;
            const int n = w * h;
            const int pred_stride = w + 3;
            const int recon_stride = w + 7;
            const int pred_rows = h + 5;
            const int recon_rows = h + 9;

            std::vector<int> source = make_block(n, -255, 255);
            std::vector<int> diff(static_cast<std::size_t>(n));
            std::vector<int> levels(static_cast<std::size_t>(n));
            if (dpcm != 0) {
              cpu::dpcm_difference_block(source.data(), w, h, vertical != 0,
                                         diff.data());
            } else {
              diff = source;
            }
            cpu::forward_lossless_block(diff.data(), w, h, dct != 0,
                                        levels.data());

            std::vector<std::uint8_t> pred(
                static_cast<std::size_t>(pred_stride) *
                static_cast<std::size_t>(pred_rows));
            for (std::uint8_t& value : pred) {
              value = static_cast<std::uint8_t>(random_range(0, 255));
            }
            std::vector<std::uint8_t> recon(
                static_cast<std::size_t>(recon_stride) *
                    static_cast<std::size_t>(recon_rows),
                0xA5);

            const int offsets[][2] = {
                {0, 0}, {2, 3},
                {pred_stride - w, pred_rows - h}};
            for (const auto& offset : offsets) {
              const int x = offset[0];
              const int y = offset[1];
              std::fill(recon.begin(), recon.end(), 0xA5);

              li::ReconJob job;
              job.levels = levels.data();
              job.pred =
                  pred.data() +
                  static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(pred_stride) +
                  static_cast<std::size_t>(x);
              job.recon =
                  recon.data() +
                  static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(recon_stride) +
                  static_cast<std::size_t>(x);
              job.pred_stride = pred_stride;
              job.recon_stride = recon_stride;
              job.w = w;
              job.h = h;
              job.dct_dct = dct;
              job.dpcm = dpcm;
              job.vertical = vertical;
              run_reconstruct(job, scratch, residual_scratch);

              // CPU reference assembly from the public lossless helpers.
              std::vector<int> inv(static_cast<std::size_t>(n));
              cpu::inverse_lossless_block(levels.data(), w, h, dct != 0,
                                          inv.data());
              std::vector<int> reference(static_cast<std::size_t>(n));
              if (dpcm != 0) {
                cpu::dpcm_cumsum_block(inv.data(), w, h, vertical != 0,
                                       reference.data());
              } else {
                reference = inv;
              }
              bool ok = true;
              for (int row = 0; row < h; ++row) {
                for (int col = 0; col < w; ++col) {
                  const int p =
                      pred[static_cast<std::size_t>(y + row) *
                               static_cast<std::size_t>(pred_stride) +
                           static_cast<std::size_t>(x + col)];
                  int value =
                      p + reference[static_cast<std::size_t>(row) *
                                        static_cast<std::size_t>(w) +
                                    static_cast<std::size_t>(col)];
                  value = value < 0 ? 0 : (value > 255 ? 255 : value);
                  const std::uint8_t want =
                      static_cast<std::uint8_t>(value);
                  const std::uint8_t got =
                      recon[static_cast<std::size_t>(y + row) *
                                static_cast<std::size_t>(recon_stride) +
                            static_cast<std::size_t>(x + col)];
                  if (got != want) {
                    ok = false;
                    check(false, "reconstruction sample", w, h, dct, dpcm);
                    break;
                  }
                }
                if (!ok) break;
              }
              ++g_cases;
              if (ok) check(true, "reconstruction block");

              // Nothing outside the block may be written.
              std::size_t outside_writes = 0;
              for (int row = 0; row < recon_rows; ++row) {
                for (int col = 0; col < recon_stride; ++col) {
                  const bool in_block =
                      row >= y && row < y + h && col >= x && col < x + w;
                  if (!in_block &&
                      recon[static_cast<std::size_t>(row) *
                                static_cast<std::size_t>(recon_stride) +
                            static_cast<std::size_t>(col)] != 0xA5) {
                    ++outside_writes;
                  }
                }
              }
              ++g_cases;
              check(outside_writes == 0, "reconstruction wrote outside block",
                    w, h, static_cast<int>(outside_writes));
            }
          }
        }
      }
    }
  }
  std::printf("   %d checks, %d failures\n", g_checks,
              g_failures - checks_before);
}

// ---------------------------------------------------------------------------
// 6. §5.20.6.1 transform-size selection
// ---------------------------------------------------------------------------

void test_tx_selection() {
  std::printf("-- §5.20.6.1 lossless transform-size selection\n");
  const int checks_before = g_failures;
  const int dims[] = {4, 8, 12, 16, 24, 32, 64};
  for (int bw : dims) {
    for (int bh : dims) {
      for (int is_inter = 0; is_inter <= 1; ++is_inter) {
        for (int fsc = 0; fsc <= 1; ++fsc) {
          for (int allow = 0; allow <= 1; ++allow) {
            for (int want = 0; want <= 1; ++want) {
              cpu::TxSizeDecision want_decision =
                  cpu::select_tx_size(bw, bh, is_inter != 0, fsc != 0,
                                      allow != 0, want != 0);
              li::TxSizeRequest request;
              request.block_w = bw;
              request.block_h = bh;
              request.is_inter = is_inter;
              request.fsc_mode = fsc;
              request.allow_select = allow;
              request.want_large = want;
              const li::TxSizeDecision got = li::li_select_tx_size(request);
              ++g_cases;
              check(got.coded_w == want_decision.coded_w &&
                        got.coded_h == want_decision.coded_h &&
                        got.lossless_tx_size ==
                            want_decision.lossless_tx_size &&
                        got.signal_lossless_tx_size ==
                            want_decision.signal_lossless_tx_size,
                    "select_tx_size", bw, bh, is_inter, fsc);
            }
          }
        }
      }
    }
  }

  for (int cw = 1; cw <= 64; ++cw) {
    for (int ch = 1; ch <= 64; ++ch) {
      int cpu_w = 0;
      int cpu_h = 0;
      int gpu_w = 0;
      int gpu_h = 0;
      const bool cpu_ok = cpu::find_tx_size(cw, ch, &cpu_w, &cpu_h);
      const bool gpu_ok = li::li_find_tx_size(cw, ch, &gpu_w, &gpu_h);
      ++g_cases;
      check(cpu_ok == gpu_ok &&
                (!cpu_ok || (cpu_w == gpu_w && cpu_h == gpu_h)),
            "find_tx_size", cw, ch);
    }
  }
  std::printf("   %d checks, %d failures\n", g_checks,
              g_failures - checks_before);
}

// ---------------------------------------------------------------------------
// 7. §5.18.2 lossless derivation
// ---------------------------------------------------------------------------

void test_lossless_derivation() {
  std::printf("-- §5.18.2 LosslessArray / per-segment qindex\n");
  const int checks_before = g_failures;
  for (int iter = 0; iter < 6000; ++iter) {
    li::FrameQuantInputs in;
    in.base_q_idx = random_range(0, 255);
    in.delta_q_present = random_range(0, 1);
    in.delta_q_y_dc = random_range(-64, 64);
    in.base_y_dc_delta_q = random_range(-32, 32);
    in.delta_q_u_dc = random_range(-64, 64);
    in.delta_q_v_dc = random_range(-64, 64);
    in.delta_q_u_ac = random_range(-64, 64);
    in.delta_q_v_ac = random_range(-64, 64);
    in.base_uv_dc_delta_q = random_range(-32, 32);
    in.base_uv_ac_delta_q = random_range(-32, 32);
    in.num_segments = random_range(-1, 20);
    for (int s = 0; s < li::kMaxSegments; ++s) {
      in.alt_q_enabled[s] = random_range(0, 1) != 0;
      in.alt_q[s] = random_range(-64, 64);
    }
    // Force exact-lossless cases regularly.
    if (iter % 4 == 0) {
      in.base_q_idx = 0;
      in.delta_q_present = 0;
      for (int s = 0; s < li::kMaxSegments; ++s) in.alt_q[s] = 0;
      in.delta_q_y_dc = 0;
      in.delta_q_u_dc = 0;
      in.delta_q_v_dc = 0;
      in.delta_q_u_ac = 0;
      in.delta_q_v_ac = 0;
      in.base_y_dc_delta_q = 0;
      in.base_uv_dc_delta_q = 0;
      in.base_uv_ac_delta_q = 0;
    }

    av2enc::lossless::inter::FrameQuantInputs cpu_in;
    cpu_in.base_q_idx = in.base_q_idx;
    cpu_in.delta_q_present = in.delta_q_present != 0;
    cpu_in.delta_q_y_dc = in.delta_q_y_dc;
    cpu_in.base_y_dc_delta_q = in.base_y_dc_delta_q;
    cpu_in.delta_q_u_dc = in.delta_q_u_dc;
    cpu_in.delta_q_v_dc = in.delta_q_v_dc;
    cpu_in.delta_q_u_ac = in.delta_q_u_ac;
    cpu_in.delta_q_v_ac = in.delta_q_v_ac;
    cpu_in.base_uv_dc_delta_q = in.base_uv_dc_delta_q;
    cpu_in.base_uv_ac_delta_q = in.base_uv_ac_delta_q;
    cpu_in.num_segments = in.num_segments;
    for (int s = 0; s < li::kMaxSegments; ++s) {
      cpu_in.alt_q_enabled[s] = in.alt_q_enabled[s];
      cpu_in.alt_q[s] = in.alt_q[s];
    }

    const cpu::FrameLossless want = cpu::derive_lossless(cpu_in, true);
    const li::FrameLossless got = li::li_derive_lossless(in);
    ++g_cases;
    bool ok = got.coded_lossless == want.coded_lossless &&
              got.has_lossless_segment == want.has_lossless_segment &&
              got.num_segments == want.num_segments;
    for (int s = 0; ok && s < li::kMaxSegments; ++s) {
      ok = got.lossless[s] == want.lossless[s] &&
           got.qindex[s] == want.qindex[s];
    }
    check(ok, "derive_lossless", iter, got.num_segments, want.num_segments);
  }
  std::printf("   %d checks, %d failures\n", g_checks,
              g_failures - checks_before);
}

}  // namespace

int main() {
  std::printf("lossless_inter emulation parity vs src/encoder/lossless/inter\n");
  test_wht4();
  test_block_transform();
  test_dpcm();
  test_forward_pipeline();
  test_reconstruction();
  test_tx_selection();
  test_lossless_derivation();

  std::printf(
      "lossless_inter_emulation_test: %lld cases, %d checks, %d failures\n",
      g_cases, g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
