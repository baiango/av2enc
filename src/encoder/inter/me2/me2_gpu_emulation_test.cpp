// Host emulation of the me2 warp-cooperative cost functor.
//
// The GPU kernel (me2_gpu.cu) computes each candidate cost as
//   warp_sum(me2_cost_lane(..., lane, 32))
// with integer shuffles.  Integer addition is associative, so the same total
// is obtained on the host by summing the 32 lane partials.  This TU does that
// and compares:
//   * candidate costs against me::me_cost_plane() (the CPU oracle),
//   * full me2_search_core() results under the simulated warp cost against the
//     scalar CPU instantiation.
//
// It needs no GPU or nvcc and runs as part of build_and_test.sh.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "encoder/inter/me/me_test_common.h"
#include "encoder/inter/me2/me2.h"
#include "encoder/inter/me2/me2_gpu_cost.cuh"

namespace {

using namespace av2enc::me;
using namespace av2enc::me::test;
using av2enc::me2::Me2Options;
using av2enc::me2::Me2Result;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, msg)                                                   \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      if (g_failures <= 20) {                                              \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
      }                                                                    \
    }                                                                      \
  } while (0)

struct SimWarpCost {
  std::int32_t operator()(const MeCorePlane& cur, const MeCorePlane& ref,
                          int x, int y, int w, int h, std::int32_t mv_x,
                          std::int32_t mv_y, int metric,
                          int filter) const {
    std::int32_t sum = 0;
    for (int lane = 0; lane < 32; ++lane) {
      sum += av2enc::me2::gpu::me2_cost_lane(cur, ref, x, y, w, h, mv_x, mv_y,
                                             metric, filter, lane, 32);
    }
    return sum;
  }
};

bool same_result(const Me2Result& a, const Me2Result& b) {
  return a.mv_x == b.mv_x && a.mv_y == b.mv_y &&
         a.search_mv_x == b.search_mv_x &&
         a.search_mv_y == b.search_mv_y && a.ref_index == b.ref_index &&
         a.cost == b.cost && a.search_cost == b.search_cost &&
         a.evals == b.evals && a.mv_rate_bins == b.mv_rate_bins &&
         a.precision_bins == b.precision_bins &&
         a.total_rate_bins == b.total_rate_bins &&
         a.precision == b.precision &&
         a.use_most_probable_precision == b.use_most_probable_precision &&
         a.pb_mv_precision == b.pb_mv_precision && a.stage == b.stage &&
         a.visible_w == b.visible_w && a.visible_h == b.visible_h;
}

void test_cost_parity() {
  TestFrame frame;
  frame.init(96, 80);
  fill_smooth_texture(&frame, 0x51u, 8);
  TestFrame ref = make_shifted_ref(frame, 3, -2);
  const MeCoreFrame cur = frame.view();
  const MeCoreRef refv = ref.ref_view();
  const SimWarpCost warp;
  const MeScalarCost scalar;
  const int sizes[][2] = {{16, 16}, {8, 8}, {4, 4}, {32, 16}, {12, 20}};
  const std::int32_t mvs[][2] = {{0, 0},  {16, 16},   {8, -8},
                                 {4, 4},  {2, -2},    {1, 1},
                                 {-14, 7}, {31, -45}, {-16, -16}};
  for (const auto& sz : sizes) {
    for (int metric = 0; metric < 2; ++metric) {
      for (int filter = 0; filter <= kMeMaxInterpFilter; ++filter) {
        for (const auto& mv : mvs) {
          const int x = 4;
          const int y = 4;
          const int w = sz[0];
          const int h = sz[1];
          const std::int32_t a =
              scalar(cur.plane[0], refv.plane[0], x, y, w, h, mv[0], mv[1],
                     metric, filter);
          const std::int32_t b =
              warp(cur.plane[0], refv.plane[0], x, y, w, h, mv[0], mv[1],
                   metric, filter);
          CHECK(a == b, "warp lane sum must equal scalar cost");
        }
      }
    }
  }
}

void test_search_parity() {
  TestFrame frame;
  frame.init(160, 128);
  fill_smooth_texture(&frame, 0x77u, 8);
  TestFrame ref = make_shifted_ref(frame, 2, 1);
  const MeCoreFrame cur = frame.view();
  const MeCoreRef refv = ref.ref_view();
  const MeScalarCost scalar;
  const SimWarpCost warp;

  const int methods[3] = {kMeFull, kMeDiamond, kMeTestZone};
  const int frame_precisions[3] = {av2enc::me2::kMvPrecisionHalfPel,
                                   av2enc::me2::kMvPrecisionQuarterPel,
                                   av2enc::me2::kMvPrecisionEighthPel};
  const int lambdas[2] = {0, 24};
  for (int m = 0; m < 3; ++m) {
    for (int fp = 0; fp < 3; ++fp) {
      for (int li = 0; li < 2; ++li) {
        Me2Options opt;
        opt.search.method = methods[m];
        opt.search.metric = kMeSad;
        opt.search.range = 4;
        opt.search.max_iterations = 3;
        opt.frame_precision = frame_precisions[fp];
        opt.enable_flex_mvres = 1;
        opt.mv_rate_lambda = lambdas[li];
        opt.mv_rate_shift = 0;

        const int bw = 16;
        const int bh = 16;
        const auto jobs = make_block_grid(160, 128, bw, bh);
        for (const MeCoreJob& job : jobs) {
          const Me2Result a =
              av2enc::me2::me2_search_core(cur, &refv, job, opt, nullptr,
                                           scalar);
          const Me2Result b =
              av2enc::me2::me2_search_core(cur, &refv, job, opt, nullptr,
                                           warp);
          CHECK(same_result(a, b),
                "scalar and warp-simulated search results must match");
        }
      }
    }
  }
}

void test_cost_api_parity() {
  TestFrame frame;
  frame.init(64, 64);
  fill_smooth_texture(&frame, 0x99u, 8);
  TestFrame ref = make_shifted_ref(frame, -1, 2);
  const MeCoreFrame cur = frame.view();
  const MeCoreRef refv = ref.ref_view();
  const SimWarpCost warp;
  const av2enc::me2::MvPrecision precisions[3] = {
      av2enc::me2::kMvPrecisionEighthPel,
      av2enc::me2::kMvPrecisionQuarterPel,
      av2enc::me2::kMvPrecisionHalfPel};
  for (int p = 0; p < 3; ++p) {
    av2enc::me2::Me2CostJob job;
    job.precision = precisions[p];
    job.base.x = 8;
    job.base.y = 8;
    job.base.w = 16;
    job.base.h = 16;
    // On-grid MV for the selected precision.
    job.base.mv_x = precisions[p] == av2enc::me2::kMvPrecisionHalfPel ? 8 : 4;
    job.base.mv_y = precisions[p] == av2enc::me2::kMvPrecisionHalfPel ? -8 : -4;
    job.base.metric = kMeSad;
    job.base.interp = kMeInterpRegular;
    std::int32_t cost = 0;
    const auto st = av2enc::me2::me2_cost_block(cur, refv, job, &cost);
    CHECK(st == av2enc::me2::Me2Status::kOk, "me2_cost_block ok");
    const std::int32_t wcost =
        warp(cur.plane[0], refv.plane[0], job.base.x, job.base.y,
             job.base.w, job.base.h, job.base.mv_x, job.base.mv_y,
             job.base.metric, job.base.interp);
    CHECK(cost == wcost, "me2_cost_block equals warp cost");
  }
}

void test_real_720p_parity(const std::string& root) {
  const char* clip_env = std::getenv("ME2_Y4M");
  const std::string clip =
      (clip_env != nullptr && clip_env[0] != '\0')
          ? std::string(clip_env)
          : root + "/testdata/shine_1280x720_60f.y4m";
  std::vector<TestFrame> frames;
  if (!load_y4m_frames(clip, 2, &frames) || frames.size() < 2) {
    std::printf("  (720p testdata absent; skipping real-frame parity)\n");
    return;
  }
  const MeCoreFrame cur = frames[0].view();
  const MeCoreRef refv = frames[1].ref_view();
  const MeScalarCost scalar;
  const SimWarpCost warp;
  Me2Options opt;
  opt.search.method = kMeDiamond;
  opt.search.metric = kMeSad;
  opt.search.range = 8;
  opt.frame_precision = av2enc::me2::kMvPrecisionEighthPel;
  opt.enable_flex_mvres = 1;
  opt.mv_rate_lambda = 16;

  const int bw = 64;
  const int bh = 64;
  const auto all = make_block_grid(1280, 720, bw, bh);
  // Deterministic subset across the frame.
  for (std::size_t i = 0; i < all.size(); i += 37) {
    const MeCoreJob& job = all[i];
    const Me2Result a =
        av2enc::me2::me2_search_core(cur, &refv, job, opt, nullptr, scalar);
    const Me2Result b =
        av2enc::me2::me2_search_core(cur, &refv, job, opt, nullptr, warp);
    CHECK(same_result(a, b), "720p scalar/warp search parity");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string root =
      argc > 1 ? argv[1] : std::string("../../../../..");
  std::printf("== me2 GPU host emulation ==\n");
  test_cost_parity();
  test_search_parity();
  test_cost_api_parity();
  const char* skip720 = std::getenv("ME2_SKIP_720P");
  if (skip720 == nullptr || skip720[0] == '\0' || skip720[0] == '0') {
    test_real_720p_parity(root);
  } else {
    std::printf("  (ME2_SKIP_720P set; skipping real-frame parity)\n");
  }
  std::printf("checks=%d failures=%d\n", g_checks, g_failures);
  if (g_failures != 0) {
    std::printf("FAIL\n");
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
