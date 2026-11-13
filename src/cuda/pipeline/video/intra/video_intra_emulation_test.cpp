// Host emulation of the M2 video intra candidate device body.
//
// Compiles `video_intra_device.cuh` with CUDA keywords stubbed and one emulated
// thread, then checks every predicted byte and SSE against the CPU reference
// (`src/codec/intra/modes/`).  The real sm_86 test is `video_intra_test.cu`.
//
// Build/run: ./build_emulation_test.sh
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static
#define __restrict__

struct EmuDim {
  unsigned x = 0;
  unsigned y = 0;
  unsigned z = 0;
};
static EmuDim threadIdx;
static EmuDim blockIdx;
static EmuDim blockDim;
static inline void __syncthreads() {}

#include "cuda/pipeline/video/intra/video_intra_device.cuh"
#include "cuda/pipeline/video/intra/video_intra_host.h"

#include "codec/intra/modes/intra_modes.h"

namespace {

namespace vi = av2enc::cuda::video::intra;
using av2enc::intra::IntraEdges;
using av2enc::intra::IntraStatus;
using av2enc::intra::NeighborRequest;
using av2enc::intra::PlaneView;
using av2enc::intra::PredictParams;

int g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, int line, const char* detail,
            int value) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 20) {
      std::printf("FAIL line %d: %s (%s=%d)\n", line, what, detail, value);
    }
  }
}

#define EXPECT(cond) expect((cond), #cond, __LINE__, "", 0)
#define EXPECT_EQ(a, b, what) \
  expect((a) == (b), #a " == " #b, __LINE__, what, static_cast<int>(a))

std::uint32_t g_rng = 0x2f6e2b1u;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
int random_range(int low, int high) {
  return low + static_cast<int>(next_random() %
                                static_cast<std::uint32_t>(high - low + 1));
}

// ---------------------------------------------------------------------------
// Candidate / availability contract
// ---------------------------------------------------------------------------

void test_candidates() {
  const vi::Candidate* list = vi::luma_candidates();
  EXPECT_EQ(vi::luma_candidate_count(), vi::kMaxCandidates, "count");
  // 13 base modes first.
  for (int i = 0; i < 13; ++i) {
    EXPECT_EQ(list[i].mode, i, "base mode");
    EXPECT_EQ(list[i].angle_delta, 0, "base delta");
  }
  // Then 8 directional modes x 6 deltas.
  int index = 13;
  const int deltas[6] = {-3, -2, -1, 1, 2, 3};
  for (int mode = 1; mode <= 8; ++mode) {
    for (int d : deltas) {
      EXPECT_EQ(list[index].mode, mode, "delta mode");
      EXPECT_EQ(list[index].angle_delta, d, "delta value");
      ++index;
    }
  }
  EXPECT_EQ(index, vi::kMaxCandidates, "table end");

  vi::Candidate out[vi::kMaxCandidates];
  EXPECT_EQ(vi::build_candidates(0, vi::kAllModesMask, 1, out,
                                 vi::kMaxCandidates),
            61, "full luma");
  EXPECT_EQ(vi::build_candidates(0, vi::kAllModesMask, 0, out,
                                 vi::kMaxCandidates),
            13, "base only");
  EXPECT_EQ(vi::build_candidates(0, vi::kDcOnlyMask, 1, out,
                                 vi::kMaxCandidates),
            1, "dc only");
  EXPECT_EQ(out[0].mode, 0, "dc mode");
  EXPECT_EQ(vi::build_candidates(0, vi::kNoDirectionalMask, 1, out,
                                 vi::kMaxCandidates),
            5, "no directional");
  EXPECT_EQ(vi::build_candidates(0, vi::kAllModesMask, 1, out, 60), -1,
            "capacity");
  EXPECT_EQ(vi::build_candidates(0, 0, 1, out, 61), -1, "empty mask");
  EXPECT_EQ(vi::build_candidates(0, 0x8000u, 1, out, 61), -1, "bad mask bit");
  EXPECT_EQ(vi::build_candidates(3, vi::kAllModesMask, 1, out, 61), -1,
            "bad plane");
  // Chroma: DC only, mask and deltas ignored.
  for (int plane = 1; plane <= 2; ++plane) {
    const int n = vi::build_candidates(plane, vi::kAllModesMask, 1, out, 1);
    EXPECT_EQ(n, 1, "chroma count");
    EXPECT_EQ(out[0].mode, 0, "chroma mode");
    EXPECT_EQ(out[0].angle_delta, 0, "chroma delta");
  }
}

void test_availability() {
  // 640x360 full-frame plane, 64x64 interior leaf.
  vi::AvailRequest req;
  req.x = 64;
  req.y = 64;
  req.w = 64;
  req.h = 64;
  req.plane_w = 640;
  req.plane_h = 360;
  req.tile_x1 = 640;
  req.tile_y1 = 360;
  vi::Availability a = vi::derive_availability(req);
  EXPECT_EQ(a.max_x, 639, "max_x");
  EXPECT_EQ(a.max_y, 359, "max_y");
  EXPECT_EQ(a.have_left, 1, "left");
  EXPECT_EQ(a.have_above, 1, "above");
  EXPECT_EQ(a.num4_above_right, 16, "above-right cap");
  EXPECT_EQ(a.num4_below_left, 0, "below-left raster");

  // Top-left leaf: no neighbours.
  req.x = 0;
  req.y = 0;
  a = vi::derive_availability(req);
  EXPECT_EQ(a.have_left, 0, "tl left");
  EXPECT_EQ(a.have_above, 0, "tl above");
  EXPECT_EQ(a.num4_above_right, 0, "tl above-right");
  EXPECT_EQ(a.num4_below_left, 0, "tl below-left");

  // Right/bottom partial leaf at 100x70: extends past the visible plane.
  req.x = 64;
  req.y = 64;
  req.w = 64;
  req.h = 64;
  req.plane_w = 100;
  req.plane_h = 70;
  req.tile_x1 = 0;
  req.tile_y1 = 0;
  a = vi::derive_availability(req);
  EXPECT_EQ(a.max_x, 99, "odd max_x");
  EXPECT_EQ(a.max_y, 69, "odd max_y");
  EXPECT_EQ(a.have_left, 1, "odd left");
  EXPECT_EQ(a.have_above, 1, "odd above");
  EXPECT_EQ(a.num4_above_right, 0, "odd above-right (clipped)");
  EXPECT_EQ(a.num4_below_left, 0, "odd below-left");

  // Tile-local plane with a tile origin: a leaf at the tile start has no
  // neighbours even though the frame continues.
  req.x = 32;
  req.y = 16;
  req.w = 32;
  req.h = 32;
  req.tile_x0 = 32;
  req.tile_y0 = 16;
  req.tile_x1 = 96;
  req.tile_y1 = 80;
  req.plane_w = 640;
  req.plane_h = 360;
  a = vi::derive_availability(req);
  EXPECT_EQ(a.max_x, 95, "tile max_x");
  EXPECT_EQ(a.max_y, 79, "tile max_y");
  EXPECT_EQ(a.have_left, 0, "tile left");
  EXPECT_EQ(a.have_above, 0, "tile above");
  EXPECT_EQ(a.num4_above_right, 0, "tile above-right");

  // Caller-supplied decoded count caps the geometric bound.
  req.tile_x0 = 0;
  req.tile_y0 = 0;
  req.x = 16;
  req.y = 16;
  req.decoded_above_right4 = 2;
  req.decoded_below_left4 = 1;
  a = vi::derive_availability(req);
  EXPECT_EQ(a.num4_above_right, 2, "decoded above-right");
  EXPECT_EQ(a.num4_below_left, 1, "decoded below-left");
}

// ---------------------------------------------------------------------------
// Device body vs CPU reference
// ---------------------------------------------------------------------------

void run_case(int w, int h, int x, int y, int plane, int mode, int angle_delta,
              int have_left, int have_above, int num4_above_right,
              int num4_below_left, int filter_above, int filter_left,
              int enable_edge_filter) {
  const int stride = w + x + 96;
  const int rows = h + y + 96;
  std::vector<std::uint8_t> recon(
      static_cast<std::size_t>(stride) * static_cast<std::size_t>(rows));
  for (std::uint8_t& v : recon) {
    v = static_cast<std::uint8_t>(next_random() & 0xFFu);
  }
  // Make the current block distinct so an accidental read is visible.
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      recon[static_cast<std::size_t>(y + i) * stride + x + j] =
          static_cast<std::uint8_t>((i * 11 + j * 3 + 29) & 0xFF);
    }
  }
  const int max_x = x + w + 32;
  const int max_y = y + h + 32;

  // CPU reference.
  NeighborRequest request;
  request.recon = PlaneView{recon.data(), stride};
  request.x = x;
  request.y = y;
  request.w = w;
  request.h = h;
  request.max_x = max_x;
  request.max_y = max_y;
  request.have_left = have_left != 0;
  request.have_above = have_above != 0;
  request.num4_above_right = num4_above_right;
  request.num4_below_left = num4_below_left;
  request.color_plane = plane;
  IntraEdges edges;
  if (av2enc::intra::prepare_edges(request, 8, edges) != IntraStatus::kOk) {
    EXPECT(false);
    return;
  }
  PredictParams params;
  params.mode = static_cast<av2enc::intra::Mode>(mode);
  params.angle_delta = angle_delta;
  params.enable_edge_filter = enable_edge_filter != 0;
  params.above_neighbor_is_smooth = filter_above != 0;
  params.left_neighbor_is_smooth = filter_left != 0;
  std::vector<std::uint8_t> want(static_cast<std::size_t>(w) * h);
  if (av2enc::intra::predict_intra(edges, params, want.data(),
                                   static_cast<std::size_t>(w)) !=
      IntraStatus::kOk) {
    EXPECT(false);
    return;
  }
  std::int64_t want_sse = 0;
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      const int diff = static_cast<int>(
                           recon[static_cast<std::size_t>(y + i) * stride + x +
                                 j]) -
                       static_cast<int>(want[static_cast<std::size_t>(i) * w +
                                            j]);
      want_sse += static_cast<std::int64_t>(diff) * diff;
    }
  }

  // Device body.
  vi::BlockJob job{};
  job.recon_plane = recon.data();
  job.recon_stride = stride;
  job.source_plane = recon.data();
  job.source_stride = stride;
  job.x = x;
  job.y = y;
  job.w = w;
  job.h = h;
  job.plane = plane;
  job.max_x = max_x;
  job.max_y = max_y;
  job.source_max_x = max_x;
  job.source_max_y = max_y;
  job.have_left = have_left;
  job.have_above = have_above;
  job.num4_above_right = num4_above_right;
  job.num4_below_left = num4_below_left;
  job.filter_type_above = filter_above;
  job.filter_type_left = filter_left;
  job.enable_edge_filter = enable_edge_filter;

  vi::detail::CandidateDesc desc{};
  desc.job_index = 0;
  desc.result_index = 0;
  desc.pred_offset = 0;
  desc.mode = mode;
  desc.angle_delta = angle_delta;

  std::vector<std::uint8_t> got(static_cast<std::size_t>(w) * h);
  std::vector<std::int16_t> residual(static_cast<std::size_t>(w) * h);
  vi::CandidateResult result{};
  blockDim.x = 1;
  threadIdx.x = 0;
  vi::detail::run_candidate(&job, &desc, 0, got.data(), residual.data(),
                            &result);
  blockDim.x = 0;

  if (std::memcmp(got.data(), want.data(), want.size()) != 0) {
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (got[i] != want[i]) {
        std::printf(
            "FAIL: mode=%d delta=%d %dx%d x=%d y=%d plane=%d filter=%d/%d "
            "edge=%d sample=%zu got=%u want=%u\n",
            mode, angle_delta, w, h, x, y, plane, filter_above, filter_left,
            enable_edge_filter, i, static_cast<unsigned>(got[i]),
            static_cast<unsigned>(want[i]));
        break;
      }
    }
    ++g_checks;
    ++g_failures;
    return;
  }
  EXPECT_EQ(result.sse, want_sse, "sse");
  // Residual = source - prediction (coded size).
  bool residual_ok = true;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const std::size_t row = i / static_cast<std::size_t>(w);
    const std::size_t col = i % static_cast<std::size_t>(w);
    const int sample = static_cast<int>(
        recon[(static_cast<std::size_t>(y) + row) * stride +
              static_cast<std::size_t>(x) + col]);
    if (residual[i] != static_cast<std::int16_t>(sample - static_cast<int>(got[i]))) {
      residual_ok = false;
      break;
    }
  }
  EXPECT(residual_ok);
  ++g_checks;
}

}  // namespace

int main() {
  test_candidates();
  test_availability();

  const int dims[5] = {4, 8, 16, 32, 64};
  for (int trial = 0; trial < 1600; ++trial) {
    const int w = dims[random_range(0, 4)];
    const int h = dims[random_range(0, 4)];
    const int x = random_range(0, 1) == 0 ? 0 : random_range(1, 40);
    const int y = random_range(0, 1) == 0 ? 0 : random_range(1, 40);
    const int have_left = x > 0 ? 1 : 0;
    const int have_above = y > 0 ? 1 : 0;
    const int plane = random_range(0, 2);
    const int mode = random_range(0, 12);
    const bool directional = mode >= 1 && mode <= 8;
    const int deltas[7] = {-3, -2, -1, 0, 1, 2, 3};
    const int delta = directional ? deltas[random_range(0, 6)] : 0;
    run_case(w, h, x, y, plane, mode, delta, have_left, have_above,
             random_range(0, w / 4), random_range(0, h / 4),
             random_range(0, 1), random_range(0, 1),
             random_range(0, 3) == 0 ? 0 : 1);
  }

  // Every mode and delta on a fixed edge state, both planes.
  const int deltas[7] = {-3, -2, -1, 0, 1, 2, 3};
  for (int mode = 0; mode < 13; ++mode) {
    for (int delta : deltas) {
      const int effective = (mode >= 1 && mode <= 8) ? delta : 0;
      run_case(16, 8, 8, 8, 0, mode, effective, 1, 1, 2, 1, 0, 1, 1);
      run_case(8, 16, 8, 8, 1, mode, effective, 1, 1, 1, 2, 1, 0, 1);
      run_case(4, 4, 0, 0, 2, mode, effective, 0, 0, 0, 0, 1, 1, 1);
    }
  }

  std::printf("video_intra_emulation_test: %s (%d checks)\n",
              g_failures == 0 ? "PASS" : "FAIL", g_checks);
  return g_failures == 0 ? 0 : 1;
}
