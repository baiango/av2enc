// Real-GPU test for the arena-backed intra workspace.
//
// Verifies the staging/dispatch contract and CPU parity for the tools wired so
// far.  The underlying kernels have their own exhaustive CPU-vs-GPU tests
// (`modes/intra_modes_test.cu`, `mrl/mrl_test.cu`, ...); this test proves the
// orchestrator-facing workspace drives them correctly from one Arena.
//
// Build/run: ./build_intra_workspace_test.sh   (sm_86 default)

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/codec/intra/modes/intra_modes.h"
#include "cuda/codec/intra/workspace/intra_workspace.h"
#include "cuda/error.h"
#include "codec/intra/modes/intra_modes.h"
#include "codec/intra/mrl/core/mrl.h"
#include "codec/intra/ibp/ibp.h"
#include "codec/intra/dip/dip.h"
#include "codec/intra/cfl/cfl.h"

namespace cuda = av2enc::cuda;
namespace cw = av2enc::cuda::intra_workspace;
namespace cmodes = av2enc::cuda::intra_modes;
namespace cfl = av2enc::cfl;
namespace cpu = av2enc::intra;

namespace {

long long g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what, long long got = 0,
           long long want = 0) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 20) {
      std::printf("FAIL: %s (got=%lld want=%lld)\n", what, got, want);
    }
  }
}

#define CHECK(cond) check((cond), #cond)
#define CHECK_EQ(a, b, what) \
  do {                       \
    const long long va_ = static_cast<long long>(a); \
    const long long vb_ = static_cast<long long>(b); \
    check(va_ == vb_, what, va_, vb_); \
  } while (false)

constexpr int kStride = 160;
constexpr int kPlaneW = 160;
constexpr int kPlaneH = 160;
constexpr int kPlaneBytes = kStride * kPlaneH;

std::uint32_t g_rng = 0x7f4a7c15u;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

bool cuda_ok(cudaError_t error) {
  return cuda::status_from_cuda(error) == cuda::Status::kOk;
}

// One CPU/GOU comparison case.
struct Case {
  int w;
  int h;
  int x;
  int y;
  int mode;
  int delta;
  int plane;
  int have_above;
  int have_left;
  int filter_above;
  int filter_left;
};

std::vector<Case> make_cases() {
  std::vector<Case> cases;
  const int modes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const int deltas[] = {0, -2, 1, 3};
  const struct {
    int w, h, x, y;
  } geom[] = {
      {8, 8, 0, 0},      // no neighbours
      {8, 8, 8, 8},      // interior, both neighbours
      {16, 16, 16, 0},   // above only
      {32, 32, 0, 32},   // left only
      {32, 16, 96, 96},  // interior rectangle
      {16, 32, 96, 96},  // interior tall rectangle
      {4, 4, 4, 4},      // smallest block
      {64, 64, 64, 64},  // largest block
      {32, 8, 32, 128},  // bottom edge (partial rows)
      {8, 32, 128, 32},  // right edge
  };
  for (const auto& g : geom) {
    for (int mode : modes) {
      for (int d : deltas) {
        if (d != 0 && !(mode >= 1 && mode <= 8)) {
          continue;
        }
        Case c;
        c.w = g.w;
        c.h = g.h;
        c.x = g.x;
        c.y = g.y;
        c.mode = mode;
        c.delta = d;
        c.plane = (g.x & 1) ? 1 : 0;
        c.have_above = g.y > 0 ? 1 : 0;
        c.have_left = g.x > 0 ? 1 : 0;
        c.filter_above = (mode == 9 || mode == 10) ? 1 : 0;
        c.filter_left = (mode == 9 || mode == 11) ? 1 : 0;
        cases.push_back(c);
      }
    }
  }
  return cases;
}

void fill_plane(std::vector<std::uint8_t>& plane) {
  for (std::uint8_t& v : plane) {
    v = static_cast<std::uint8_t>(next_random());
  }
}

// Test-only device buffers (the arena rule exempts tests; this buffer holds
// independently generated source planes and outputs, not module workspace).
struct TestBuffers {
  std::uint8_t* recon = nullptr;
  std::uint8_t* pred = nullptr;
  int pred_stride = 64;

  bool allocate(std::size_t pred_bytes) {
    return cudaMalloc(&recon, kPlaneBytes) == cudaSuccess &&
           cudaMalloc(&pred, pred_bytes) == cudaSuccess;
  }
  ~TestBuffers() {
    if (recon != nullptr) cudaFree(recon);
    if (pred != nullptr) cudaFree(pred);
  }
};

// --- §7.13.2 modes ---------------------------------------------------------

void test_modes(cw::Workspace& ws, cuda::Arena& arena) {
  const std::vector<Case> cases = make_cases();
  std::vector<Case> filter_cases;
  for (std::size_t i = 0; i < cases.size(); i += 17) {
    Case c = cases[i];
    c.have_above = 1;
    c.have_left = 1;
    c.x = 64;
    c.y = 64;
    filter_cases.push_back(c);
  }

  std::vector<std::uint8_t> h_plane(kPlaneBytes);
  std::vector<std::uint8_t> h_want(64 * 64);
  std::vector<std::uint8_t> h_got(64 * 64);
  const std::size_t pred_bytes =
      static_cast<std::size_t>(cases.size()) * 64 * 64;
  TestBuffers buffers;
  CHECK(buffers.allocate(pred_bytes));
  fill_plane(h_plane);
  CHECK(cuda_ok(cudaMemcpy(buffers.recon, h_plane.data(), kPlaneBytes,
                           cudaMemcpyHostToDevice)));

  // (a) CPU parity with the normative edge filter disabled.  The CPU module's
  // edge filter is documented as filtering in place; CUDA follows
  // §7.13.2.18's snapshot rule, so an enabled-filter CPU comparison is not a
  // valid oracle (see modes/README.md).
  std::vector<cmodes::IntraJob> jobs;
  jobs.reserve(cases.size());
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    cmodes::IntraBlock b;
    b.recon_plane = buffers.recon;
    b.recon_stride = kStride;
    b.x0 = c.x;
    b.y0 = c.y;
    b.width = c.w;
    b.height = c.h;
    b.max_x = kPlaneW - 1;
    b.max_y = kPlaneH - 1;
    b.have_left = c.have_left;
    b.have_above = c.have_above;
    b.num4_above_right = 0;
    b.num4_below_left = 0;
    b.plane = c.plane;
    b.mode = static_cast<cmodes::IntraMode>(c.mode);
    b.angle_delta = c.delta;
    b.filter_type_above = c.filter_above;
    b.filter_type_left = c.filter_left;
    b.enable_edge_filter = 0;

    cmodes::IntraJob job;
    job.block = b;
    job.pred = buffers.pred + static_cast<std::size_t>(i) * 64 * 64;
    job.residual = nullptr;
    jobs.push_back(job);
  }

  CHECK_EQ(static_cast<int>(ws.modes_predict(jobs.data(),
                                             static_cast<int>(jobs.size()))),
           static_cast<int>(cuda::Status::kOk), "workspace modes launch");
  CHECK_EQ(static_cast<int>(ws.synchronize()), 0, "workspace sync");

  long long samples = 0;
  long long mismatches = 0;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const Case& c = cases[i];
    CHECK(cuda_ok(cudaMemcpy(h_got.data(), jobs[i].pred,
                             static_cast<std::size_t>(c.w) * c.h,
                             cudaMemcpyDeviceToHost)));
    cpu::NeighborRequest req;
    req.recon.data = h_plane.data();
    req.recon.stride = kStride;
    req.x = c.x;
    req.y = c.y;
    req.w = c.w;
    req.h = c.h;
    req.max_x = kPlaneW - 1;
    req.max_y = kPlaneH - 1;
    req.have_left = c.have_left != 0;
    req.have_above = c.have_above != 0;
    req.num4_above_right = 0;
    req.num4_below_left = 0;
    req.color_plane = c.plane;
    cpu::PredictParams params;
    params.mode = static_cast<cpu::Mode>(c.mode);
    params.angle_delta = c.delta;
    params.enable_edge_filter = false;
    params.above_neighbor_is_smooth = c.filter_above != 0;
    params.left_neighbor_is_smooth = c.filter_left != 0;
    CHECK_EQ(static_cast<int>(cpu::predict_block(req, 8, params, h_want.data(),
                                                 static_cast<std::size_t>(c.w))),
             0, "cpu predict status");
    for (int k = 0; k < c.w * c.h; ++k) {
      ++samples;
      if (h_got[static_cast<std::size_t>(k)] !=
          h_want[static_cast<std::size_t>(k)]) {
        ++mismatches;
      }
    }
  }
  CHECK_EQ(mismatches, 0, "modes CPU parity");
  std::printf("  modes: cases=%zu samples=%lld mismatches=%lld\n",
              cases.size(), samples, mismatches);

  // (b) Staging with the edge filter enabled: workspace vs the module's
  // single-block entry point, byte for byte.
  std::vector<cmodes::IntraJob> filtered;
  for (const Case& c : filter_cases) {
    cmodes::IntraBlock b;
    b.recon_plane = buffers.recon;
    b.recon_stride = kStride;
    b.x0 = c.x;
    b.y0 = c.y;
    b.width = c.w;
    b.height = c.h;
    b.max_x = kPlaneW - 1;
    b.max_y = kPlaneH - 1;
    b.have_left = 1;
    b.have_above = 1;
    b.num4_above_right = 0;
    b.num4_below_left = 0;
    b.plane = c.plane;
    b.mode = static_cast<cmodes::IntraMode>(c.mode);
    b.angle_delta = c.delta;
    b.filter_type_above = 1;
    b.filter_type_left = 1;
    b.enable_edge_filter = 1;
    cmodes::IntraJob job;
    job.block = b;
    job.pred = nullptr;
    job.residual = nullptr;
    filtered.push_back(job);
  }
  std::vector<std::uint8_t*> direct_pred(filtered.size());
  for (std::size_t i = 0; i < filtered.size(); ++i) {
    CHECK(cuda_ok(cudaMalloc(&direct_pred[i], 64 * 64)));
    filtered[i].pred = buffers.pred + static_cast<std::size_t>(i) * 64 * 64;
  }
  CHECK_EQ(static_cast<int>(ws.modes_predict(
               filtered.data(), static_cast<int>(filtered.size()))),
           static_cast<int>(cuda::Status::kOk), "filtered workspace launch");
  CHECK_EQ(static_cast<int>(ws.synchronize()), 0, "filtered sync");
  long long filtered_mismatches = 0;
  for (std::size_t i = 0; i < filtered.size(); ++i) {
    CHECK(cuda_ok(cmodes::intra_predict(filtered[i].block, direct_pred[i])));
    CHECK(cuda_ok(cudaDeviceSynchronize()));
    CHECK(cuda_ok(cudaMemcpy(h_got.data(), filtered[i].pred,
                             static_cast<std::size_t>(filtered[i].block.width) *
                                 filtered[i].block.height,
                             cudaMemcpyDeviceToHost)));
    CHECK(cuda_ok(cudaMemcpy(h_want.data(), direct_pred[i],
                             static_cast<std::size_t>(filtered[i].block.width) *
                                 filtered[i].block.height,
                             cudaMemcpyDeviceToHost)));
    for (int k = 0; k < filtered[i].block.width * filtered[i].block.height;
         ++k) {
      if (h_got[static_cast<std::size_t>(k)] !=
          h_want[static_cast<std::size_t>(k)]) {
        ++filtered_mismatches;
      }
    }
    cudaFree(direct_pred[i]);
  }
  CHECK_EQ(filtered_mismatches, 0, "filtered staging parity");
  std::printf("  modes edge-filter staging: cases=%zu mismatches=%lld\n",
              filtered.size(), filtered_mismatches);
  (void)arena;
}

// --- MRL ------------------------------------------------------------------

void test_mrl(cw::Workspace& ws) {
  TestBuffers buffers;
  CHECK(buffers.allocate(525 * 64 * 64));
  std::uint8_t* recon = buffers.recon;
  std::uint8_t* pred = buffers.pred;
  struct MrlCase {
    int w, h, x, y, mode, delta, index, sec, have_above, have_left;
  };
  std::vector<MrlCase> cases;
  const struct {
    int w, h, x, y, have_above, have_left;
  } geom[] = {
      {8, 8, 0, 0, 0, 0},
      {16, 16, 96, 96, 1, 1},
      {32, 8, 32, 0, 0, 1},
      {8, 32, 0, 32, 1, 0},
      {64, 64, 64, 64, 1, 1},
  };
  const int modes[] = {1, 2, 3, 5, 7};
  const int deltas[] = {0, -2, 1};
  for (const auto& g : geom) {
    for (int mode : modes) {
      for (int delta : deltas) {
        for (int index = 0; index <= 3; ++index) {
          for (int sec = 0; sec <= 1; ++sec) {
            if (index == 0 && sec != 0) {
              continue;
            }
            cases.push_back({g.w, g.h, g.x, g.y, mode, delta, index, sec,
                             g.have_above, g.have_left});
          }
        }
      }
    }
  }

  std::vector<std::uint8_t> h_plane(kPlaneBytes);
  fill_plane(h_plane);
  CHECK(cuda_ok(cudaMemcpy(recon, h_plane.data(), kPlaneBytes,
                           cudaMemcpyHostToDevice)));

  std::vector<cuda::mrl::MrlJob> jobs;
  jobs.reserve(cases.size());
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const MrlCase& c = cases[i];
    cuda::mrl::MrlBlock b;
    b.recon = recon;
    b.recon_stride = kStride;
    b.x = c.x;
    b.y = c.y;
    b.w = c.w;
    b.h = c.h;
    b.max_x = kPlaneW - 1;
    b.max_y = kPlaneH - 1;
    b.have_left = c.have_left;
    b.have_above = c.have_above;
    b.num4_above_right = 0;
    b.num4_below_left = 0;
    b.plane = 0;
    b.sb_height = 64;

    cuda::mrl::MrlParams p;
    p.mode = c.mode;
    p.angle_delta = c.delta;
    p.mrl_index = c.index;
    p.mrl_sec_index = c.sec;
    p.enable_edge_filter = 1;
    p.filter_type_above = (c.mode == 9) ? 1 : 0;
    p.filter_type_left = (c.mode == 9) ? 1 : 0;

    cuda::mrl::MrlJob job;
    job.block = b;
    job.params = p;
    job.pred = pred + static_cast<std::size_t>(i) * 64 * 64;
    jobs.push_back(job);
  }

  CHECK_EQ(static_cast<int>(ws.mrl_predict(jobs.data(),
                                           static_cast<int>(jobs.size()))),
           static_cast<int>(cuda::Status::kOk), "workspace mrl launch");
  CHECK_EQ(static_cast<int>(ws.synchronize()), 0, "mrl sync");

  std::vector<std::uint8_t> h_got(64 * 64);
  std::vector<std::uint8_t> h_want(64 * 64);
  long long samples = 0;
  long long mismatches = 0;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const MrlCase& c = cases[i];
    av2enc::mrl::Block b;
    b.recon.data = h_plane.data();
    b.recon.stride = kStride;
    b.x = c.x;
    b.y = c.y;
    b.w = c.w;
    b.h = c.h;
    b.max_x = kPlaneW - 1;
    b.max_y = kPlaneH - 1;
    b.have_left = c.have_left != 0;
    b.have_above = c.have_above != 0;
    b.num4_above_right = 0;
    b.num4_below_left = 0;
    b.plane = 0;
    b.sb_height = 64;
    av2enc::mrl::Params p;
    p.mode = static_cast<av2enc::mrl::Mode>(c.mode);
    p.angle_delta = c.delta;
    p.mrl_index = c.index;
    p.mrl_sec_index = c.sec != 0;
    p.enable_edge_filter = true;
    p.above_neighbor_is_smooth = c.mode == 9;
    p.left_neighbor_is_smooth = c.mode == 9;
    CHECK_EQ(static_cast<int>(av2enc::mrl::predict_block(
                 b, 8, p, h_want.data(), static_cast<std::size_t>(c.w))),
             0, "cpu mrl predict");
    CHECK(cuda_ok(cudaMemcpy(h_got.data(), jobs[i].pred,
                             static_cast<std::size_t>(c.w) * c.h,
                             cudaMemcpyDeviceToHost)));
    for (int k = 0; k < c.w * c.h; ++k) {
      ++samples;
      if (h_got[static_cast<std::size_t>(k)] !=
          h_want[static_cast<std::size_t>(k)]) {
        ++mismatches;
      }
    }
  }
  CHECK_EQ(mismatches, 0, "MRL CPU parity");
  std::printf("  mrl: cases=%zu samples=%lld mismatches=%lld\n", cases.size(),
              samples, mismatches);
}

// --- IBP ------------------------------------------------------------------

void test_ibp(cw::Workspace& ws) {
  struct IbpCase {
    int w, h, x, y, mode, delta, plane, enable_ibp;
  };
  std::vector<IbpCase> cases;
  const struct {
    int w, h, x, y;
  } geom[] = {
      {8, 8, 0, 0},
      {8, 8, 32, 32},
      {16, 16, 64, 64},
      {32, 16, 96, 96},
      {16, 32, 96, 96},
      {64, 64, 64, 64},
  };
  const int modes[] = {0, 1, 2, 3, 5, 7};
  const int deltas[] = {0, -2, 2};
  for (const auto& g : geom) {
    for (int mode : modes) {
      for (int delta : deltas) {
        if (mode == 0 && delta != 0) {
          continue;
        }
        for (int enable = 0; enable <= 1; ++enable) {
          for (int plane = 0; plane <= 1; ++plane) {
            cases.push_back(
                {g.w, g.h, g.x, g.y, mode, delta, plane, enable});
          }
        }
      }
    }
  }

  TestBuffers buffers;
  CHECK(buffers.allocate(cases.size() * 64 * 64));
  std::vector<std::uint8_t> h_plane(kPlaneBytes);
  fill_plane(h_plane);
  CHECK(cuda_ok(cudaMemcpy(buffers.recon, h_plane.data(), kPlaneBytes,
                           cudaMemcpyHostToDevice)));

  std::vector<cuda::ibp::IbpJob> jobs;
  jobs.reserve(cases.size());
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const IbpCase& c = cases[i];
    cuda::ibp::IbpBlock b;
    b.intra.recon_plane = buffers.recon;
    b.intra.recon_stride = kStride;
    b.intra.x0 = c.x;
    b.intra.y0 = c.y;
    b.intra.width = c.w;
    b.intra.height = c.h;
    b.intra.max_x = kPlaneW - 1;
    b.intra.max_y = kPlaneH - 1;
    b.intra.have_left = c.x > 0 ? 1 : 0;
    b.intra.have_above = c.y > 0 ? 1 : 0;
    b.intra.num4_above_right = 0;
    b.intra.num4_below_left = 0;
    b.intra.plane = c.plane;
    b.intra.mode = static_cast<cmodes::IntraMode>(c.mode);
    b.intra.angle_delta = c.delta;
    b.intra.filter_type_above = 0;
    b.intra.filter_type_left = 0;
    b.intra.enable_edge_filter = 1;
    b.enable_ibp = c.enable_ibp;
    b.is_inter = 0;
    b.use_dip = 0;
    b.uv_is_cfl = 0;
    cuda::ibp::IbpJob job;
    job.block = b;
    job.pred = buffers.pred + static_cast<std::size_t>(i) * 64 * 64;
    jobs.push_back(job);
  }

  CHECK_EQ(static_cast<int>(ws.ibp_predict(jobs.data(),
                                           static_cast<int>(jobs.size()))),
           static_cast<int>(cuda::Status::kOk), "workspace ibp launch");
  CHECK_EQ(static_cast<int>(ws.synchronize()), 0, "ibp sync");

  std::vector<std::uint8_t> h_got(64 * 64);
  std::vector<std::uint8_t> h_want(64 * 64);
  long long samples = 0;
  long long mismatches = 0;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const IbpCase& c = cases[i];
    av2enc::ibp::NeighborRequest req;
    req.recon.data = h_plane.data();
    req.recon.stride = kStride;
    req.x = c.x;
    req.y = c.y;
    req.w = c.w;
    req.h = c.h;
    req.max_x = kPlaneW - 1;
    req.max_y = kPlaneH - 1;
    req.have_left = c.x > 0;
    req.have_above = c.y > 0;
    req.num4_above_right = 0;
    req.num4_below_left = 0;
    req.color_plane = c.plane;
    av2enc::ibp::PredictParams p;
    p.mode = static_cast<av2enc::ibp::Mode>(c.mode);
    p.angle_delta = c.delta;
    p.enable_ibp = c.enable_ibp != 0;
    p.enable_edge_filter = true;
    p.is_inter = false;
    p.above_neighbor_is_smooth = false;
    p.left_neighbor_is_smooth = false;
    p.use_dip = false;
    p.uv_is_cfl = false;
    CHECK_EQ(static_cast<int>(av2enc::ibp::predict_block(
                 req, 8, p, h_want.data(), static_cast<std::size_t>(c.w))),
             0, "cpu ibp predict");
    CHECK(cuda_ok(cudaMemcpy(h_got.data(), jobs[i].pred,
                             static_cast<std::size_t>(c.w) * c.h,
                             cudaMemcpyDeviceToHost)));
    for (int k = 0; k < c.w * c.h; ++k) {
      ++samples;
      if (h_got[static_cast<std::size_t>(k)] !=
          h_want[static_cast<std::size_t>(k)]) {
        ++mismatches;
      }
    }
  }
  CHECK_EQ(mismatches, 0, "IBP CPU parity");
  std::printf("  ibp: cases=%zu samples=%lld mismatches=%lld\n", cases.size(),
              samples, mismatches);
}

// --- DIP ------------------------------------------------------------------

void test_dip(cw::Workspace& ws) {
  struct DipCase {
    int w, h, x, y, mode, transpose, have_above, have_left;
  };
  std::vector<DipCase> cases;
  const struct {
    int w, h, x, y;
  } geom[] = {
      {8, 16, 0, 0},
      {16, 8, 32, 32},
      {16, 16, 64, 64},
      {32, 32, 96, 96},
      {8, 64, 96, 64},
      {64, 8, 64, 96},
      {64, 64, 64, 64},
  };
  for (const auto& g : geom) {
    for (int mode = 0; mode < 6; ++mode) {
      for (int transpose = 0; transpose <= 1; ++transpose) {
        for (int avail = 0; avail < 4; ++avail) {
          const int have_above = g.y > 0 ? (avail & 1) : 0;
          const int have_left = g.x > 0 ? ((avail >> 1) & 1) : 0;
          cases.push_back(
              {g.w, g.h, g.x, g.y, mode, transpose, have_above, have_left});
        }
      }
    }
  }

  TestBuffers buffers;
  CHECK(buffers.allocate(cases.size() * 64 * 64));
  std::vector<std::uint8_t> h_plane(kPlaneBytes);
  fill_plane(h_plane);
  CHECK(cuda_ok(cudaMemcpy(buffers.recon, h_plane.data(), kPlaneBytes,
                           cudaMemcpyHostToDevice)));

  std::vector<cuda::dip::DipJob> jobs;
  jobs.reserve(cases.size());
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const DipCase& c = cases[i];
    cuda::dip::DipBlock b;
    b.recon = buffers.recon;
    b.recon_stride = kStride;
    b.x = c.x;
    b.y = c.y;
    b.w = c.w;
    b.h = c.h;
    b.max_x = kPlaneW - 1;
    b.max_y = kPlaneH - 1;
    b.have_left = c.have_left;
    b.have_above = c.have_above;
    b.num4_above_right = 0;
    b.num4_below_left = 0;
    cuda::dip::DipJob job;
    job.block = b;
    job.params.mode = c.mode;
    job.params.transpose = c.transpose;
    job.pred = buffers.pred + static_cast<std::size_t>(i) * 64 * 64;
    jobs.push_back(job);
  }

  CHECK_EQ(static_cast<int>(ws.dip_predict(jobs.data(),
                                           static_cast<int>(jobs.size()))),
           static_cast<int>(cuda::Status::kOk), "workspace dip launch");
  CHECK_EQ(static_cast<int>(ws.synchronize()), 0, "dip sync");

  std::vector<std::uint8_t> h_got(64 * 64);
  std::vector<std::uint8_t> h_want(64 * 64);
  long long samples = 0;
  long long mismatches = 0;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const DipCase& c = cases[i];
    av2enc::dip::Block b;
    b.recon.data = h_plane.data();
    b.recon.stride = kStride;
    b.x = c.x;
    b.y = c.y;
    b.w = c.w;
    b.h = c.h;
    b.max_x = kPlaneW - 1;
    b.max_y = kPlaneH - 1;
    b.have_left = c.have_left != 0;
    b.have_above = c.have_above != 0;
    b.num4_above_right = 0;
    b.num4_below_left = 0;
    av2enc::dip::Params p;
    p.mode = c.mode;
    p.transpose = c.transpose != 0;
    CHECK_EQ(static_cast<int>(av2enc::dip::predict_block8(
                 b, 8, p, h_want.data(), static_cast<std::size_t>(c.w))),
             0, "cpu dip predict");
    CHECK(cuda_ok(cudaMemcpy(h_got.data(), jobs[i].pred,
                             static_cast<std::size_t>(c.w) * c.h,
                             cudaMemcpyDeviceToHost)));
    for (int k = 0; k < c.w * c.h; ++k) {
      ++samples;
      if (h_got[static_cast<std::size_t>(k)] !=
          h_want[static_cast<std::size_t>(k)]) {
        ++mismatches;
      }
    }
  }
  CHECK_EQ(mismatches, 0, "DIP CPU parity");
  std::printf("  dip: cases=%zu samples=%lld mismatches=%lld\n", cases.size(),
              samples, mismatches);
}

// --- CfL ------------------------------------------------------------------

constexpr int kChromaStride = 96;
constexpr int kChromaH = 96;
constexpr int kChromaBytes = kChromaStride * kChromaH;
constexpr int kCodedChroma = 80;

void test_cfl(cw::Workspace& ws) {
  struct CflCase {
    int w, h, x, y, filter, alpha_q8;
  };
  std::vector<CflCase> explicit_cases;
  const int sizes[] = {4, 8, 16, 32};
  const int positions[] = {0, 8, 16, 32, 48};
  const int alphas[] = {-256, -96, 0, 96, 256};
  for (int w : sizes) {
    for (int pos : positions) {
      if (pos + w > kCodedChroma) {
        continue;
      }
      for (int filter = 0; filter <= 3; ++filter) {
        for (int alpha : alphas) {
          explicit_cases.push_back({w, w, pos, pos, filter, alpha});
        }
      }
    }
  }

  std::vector<std::uint8_t> h_luma(kPlaneBytes);
  std::vector<std::uint8_t> h_chroma_base(kChromaBytes);
  fill_plane(h_luma);
  fill_plane(h_chroma_base);

  std::uint8_t* d_luma = nullptr;
  std::uint8_t* d_chroma = nullptr;
  int* d_alpha = nullptr;
  CHECK(cudaMalloc(&d_luma, kPlaneBytes) == cudaSuccess);
  CHECK(cudaMalloc(&d_chroma,
                   explicit_cases.size() * kChromaBytes) == cudaSuccess);
  CHECK(cudaMalloc(&d_alpha, explicit_cases.size() * sizeof(int)) ==
        cudaSuccess);
  CHECK(cuda_ok(cudaMemcpy(d_luma, h_luma.data(), kPlaneBytes,
                           cudaMemcpyHostToDevice)));

  std::vector<std::uint8_t> h_chroma_all(explicit_cases.size() * kChromaBytes);
  for (std::size_t i = 0; i < explicit_cases.size(); ++i) {
    std::memcpy(h_chroma_all.data() + i * kChromaBytes, h_chroma_base.data(),
                kChromaBytes);
  }
  CHECK(cuda_ok(cudaMemcpy(d_chroma, h_chroma_all.data(),
                           h_chroma_all.size(), cudaMemcpyHostToDevice)));

  std::vector<cuda::CflJob> jobs;
  jobs.reserve(explicit_cases.size());
  for (std::size_t i = 0; i < explicit_cases.size(); ++i) {
    const CflCase& c = explicit_cases[i];
    cuda::CflJob job;
    cfl::Block& b = job.block;
    b.luma = d_luma;
    b.luma_stride = kStride;
    b.luma_max_x = kPlaneW - 1;
    b.luma_max_y = kPlaneH - 1;
    b.sb_top_y = ((c.y * 2) / 64) * 64;
    b.chroma = d_chroma + i * kChromaBytes;
    b.chroma_stride = kChromaStride;
    b.x = c.x;
    b.y = c.y;
    b.w = c.w;
    b.h = c.h;
    b.coded_width = kCodedChroma;
    b.coded_height = kCodedChroma;
    b.have_above = c.y > 0 ? 1 : 0;
    b.have_left = c.x > 0 ? 1 : 0;
    b.bit_depth = 8;
    b.ds_filter = c.filter;
    b.sub_x = 1;
    b.sub_y = 1;
    job.alpha_q8 = c.alpha_q8;
    jobs.push_back(job);
  }

  CHECK_EQ(static_cast<int>(ws.cfl_predict(jobs.data(),
                                           static_cast<int>(jobs.size()))),
           static_cast<int>(cuda::Status::kOk), "workspace cfl launch");
  CHECK_EQ(static_cast<int>(ws.synchronize()), 0, "cfl sync");

  long long samples = 0;
  long long mismatches = 0;
  std::vector<std::uint8_t> h_case(kChromaBytes);
  std::vector<std::uint8_t> h_got(kChromaBytes);
  for (std::size_t i = 0; i < explicit_cases.size(); ++i) {
    const CflCase& c = explicit_cases[i];
    std::memcpy(h_case.data(), h_chroma_base.data(), kChromaBytes);
    cfl::Block b = jobs[i].block;
    b.luma = h_luma.data();
    b.chroma = h_case.data();
    cfl::Scratch scratch;
    cfl::compute_luma(b, scratch);
    cfl::apply_prediction(b, c.alpha_q8, scratch);
    CHECK(cuda_ok(cudaMemcpy(h_got.data(), jobs[i].block.chroma, kChromaBytes,
                             cudaMemcpyDeviceToHost)));
    for (int row = 0; row < c.h; ++row) {
      for (int col = 0; col < c.w; ++col) {
        const std::size_t index =
            static_cast<std::size_t>(c.y + row) * kChromaStride + c.x + col;
        ++samples;
        if (h_got[index] != h_case[index]) {
          ++mismatches;
        }
      }
    }
  }
  CHECK_EQ(mismatches, 0, "CfL explicit CPU parity");
  std::printf("  cfl explicit: cases=%zu samples=%lld mismatches=%lld\n",
              explicit_cases.size(), samples, mismatches);

  // Derived alpha: reset the chroma planes and compare both the applied Q8
  // alpha and every predicted sample.
  CHECK(cuda_ok(cudaMemcpy(d_chroma, h_chroma_all.data(),
                           h_chroma_all.size(), cudaMemcpyHostToDevice)));
  std::vector<cfl::Block> blocks;
  blocks.reserve(explicit_cases.size());
  for (std::size_t i = 0; i < explicit_cases.size(); ++i) {
    blocks.push_back(jobs[i].block);
  }
  CHECK_EQ(static_cast<int>(ws.cfl_predict_derived(
               blocks.data(), d_alpha, static_cast<int>(blocks.size()))),
           static_cast<int>(cuda::Status::kOk), "workspace cfl derived launch");
  CHECK_EQ(static_cast<int>(ws.synchronize()), 0, "cfl derived sync");
  std::vector<int> h_alpha(blocks.size());
  CHECK(cuda_ok(cudaMemcpy(h_alpha.data(), d_alpha,
                           h_alpha.size() * sizeof(int),
                           cudaMemcpyDeviceToHost)));
  long long derived_checks = 0;
  long long derived_mismatches = 0;
  for (std::size_t i = 0; i < explicit_cases.size(); ++i) {
    const CflCase& c = explicit_cases[i];
    std::memcpy(h_case.data(), h_chroma_base.data(), kChromaBytes);
    cfl::Block b = blocks[i];
    b.luma = h_luma.data();
    b.chroma = h_case.data();
    cfl::Scratch scratch;
    cfl::compute_luma(b, scratch);
    const int want_alpha = cfl::derive_implicit_alpha(b, scratch);
    cfl::apply_prediction(b, want_alpha, scratch);
    if (h_alpha[i] != want_alpha) {
      ++derived_mismatches;
    }
    CHECK(cuda_ok(cudaMemcpy(h_got.data(), blocks[i].chroma, kChromaBytes,
                             cudaMemcpyDeviceToHost)));
    for (int row = 0; row < c.h; ++row) {
      for (int col = 0; col < c.w; ++col) {
        const std::size_t index =
            static_cast<std::size_t>(c.y + row) * kChromaStride + c.x + col;
        ++derived_checks;
        if (h_got[index] != h_case[index]) {
          ++derived_mismatches;
        }
      }
    }
  }
  CHECK_EQ(derived_mismatches, 0, "CfL derived CPU parity");
  std::printf("  cfl derived: cases=%zu samples=%lld mismatches=%lld\n",
              explicit_cases.size(), derived_checks, derived_mismatches);

  cudaFree(d_luma);
  cudaFree(d_chroma);
  cudaFree(d_alpha);
}

// --- Palette --------------------------------------------------------------

void test_palette(cw::Workspace& ws) {
  struct PalCase {
    int w, h, visible_w, visible_h, size;
  };
  const PalCase cases[] = {
      {8, 8, 8, 8, 2},     {16, 16, 16, 16, 4}, {32, 8, 32, 8, 3},
      {8, 32, 8, 32, 5},   {64, 64, 40, 24, 6}, {16, 32, 16, 32, 8},
  };
  constexpr int kN = static_cast<int>(sizeof(cases) / sizeof(cases[0]));

  std::vector<std::uint8_t> h_src[kN];
  std::vector<std::uint8_t> h_colors[kN];
  std::vector<std::uint8_t> h_map[kN];
  std::vector<std::uint8_t> h_map_single[kN];
  std::vector<std::uint8_t> h_dst[kN];
  std::vector<std::uint8_t> h_dst_single[kN];
  std::vector<std::uint8_t> h_hash[kN];
  std::vector<std::uint8_t> h_hash_single[kN];
  std::vector<std::uint8_t> h_order[kN];
  std::vector<std::uint8_t> h_order_single[kN];
  cuda::PaletteDeviceBlock blocks[kN];
  cuda::PaletteContextDevice contexts[kN];
  std::uint8_t* d_src[kN] = {};
  std::uint8_t* d_colors[kN] = {};
  std::uint8_t* d_map[kN] = {};
  std::uint8_t* d_dst[kN] = {};
  std::uint8_t* d_hash[kN] = {};
  std::uint8_t* d_order[kN] = {};

  for (int i = 0; i < kN; ++i) {
    const PalCase& c = cases[i];
    const std::size_t pixels =
        static_cast<std::size_t>(c.w) * static_cast<std::size_t>(c.h);
    h_src[i].resize(pixels);
    fill_plane(h_src[i]);
    h_colors[i].resize(static_cast<std::size_t>(c.size));
    for (int k = 0; k < c.size; ++k) {
      h_colors[i][static_cast<std::size_t>(k)] =
          static_cast<std::uint8_t>(17 + k * 23);
    }
    h_map[i].assign(pixels, 0);
    h_map_single[i].assign(pixels, 0);
    h_dst[i].assign(pixels, 0);
    h_dst_single[i].assign(pixels, 0);
    h_hash[i].assign(pixels, 0);
    h_hash_single[i].assign(pixels, 0);
    h_order[i].assign(pixels * 8, 0);
    h_order_single[i].assign(pixels * 8, 0);

    CHECK(cudaMalloc(&d_src[i], pixels) == cudaSuccess);
    CHECK(cudaMalloc(&d_colors[i], static_cast<std::size_t>(c.size)) ==
          cudaSuccess);
    CHECK(cudaMalloc(&d_map[i], pixels) == cudaSuccess);
    CHECK(cudaMalloc(&d_dst[i], pixels) == cudaSuccess);
    CHECK(cudaMalloc(&d_hash[i], pixels) == cudaSuccess);
    CHECK(cudaMalloc(&d_order[i], pixels * 8) == cudaSuccess);
    CHECK(cuda_ok(cudaMemcpy(d_src[i], h_src[i].data(), pixels,
                             cudaMemcpyHostToDevice)));
    CHECK(cuda_ok(cudaMemcpy(d_colors[i], h_colors[i].data(),
                             static_cast<std::size_t>(c.size),
                             cudaMemcpyHostToDevice)));

    cuda::PaletteDeviceBlock b;
    b.src = d_src[i];
    b.src_stride = c.w;
    b.block_w = c.w;
    b.block_h = c.h;
    b.visible_w = c.visible_w;
    b.visible_h = c.visible_h;
    b.colors = d_colors[i];
    b.size = c.size;
    b.map = d_map[i];
    b.map_stride = c.w;
    b.dst = d_dst[i];
    b.dst_stride = c.w;
    b.dither = false;
    blocks[i] = b;

    cuda::PaletteContextDevice ctx;
    ctx.map = d_map[i];
    ctx.map_stride = c.w;
    ctx.width = c.w;
    ctx.height = c.h;
    ctx.size = c.size;
    ctx.hash = d_hash[i];
    ctx.order = d_order[i];
    ctx.order_stride = 8;
    contexts[i] = ctx;
  }

  CHECK_EQ(static_cast<int>(ws.palette_apply(blocks, contexts, kN)),
           static_cast<int>(cuda::Status::kOk), "workspace palette launch");
  CHECK_EQ(static_cast<int>(ws.synchronize()), 0, "palette sync");

  long long map_samples = 0;
  long long mismatches = 0;
  for (int i = 0; i < kN; ++i) {
    const PalCase& c = cases[i];
    const std::size_t pixels =
        static_cast<std::size_t>(c.w) * static_cast<std::size_t>(c.h);
    CHECK(cuda_ok(cudaMemcpy(h_map[i].data(), d_map[i], pixels,
                             cudaMemcpyDeviceToHost)));
    CHECK(cuda_ok(cudaMemcpy(h_dst[i].data(), d_dst[i], pixels,
                             cudaMemcpyDeviceToHost)));
    CHECK(cuda_ok(cudaMemcpy(h_hash[i].data(), d_hash[i], pixels,
                             cudaMemcpyDeviceToHost)));
    CHECK(cuda_ok(cudaMemcpy(h_order[i].data(), d_order[i], pixels * 8,
                             cudaMemcpyDeviceToHost)));

    CHECK(cudaMemset(d_map[i], 0, pixels) == cudaSuccess);
    CHECK(cudaMemset(d_dst[i], 0, pixels) == cudaSuccess);
    CHECK(cudaMemset(d_hash[i], 0, pixels) == cudaSuccess);
    CHECK(cudaMemset(d_order[i], 0, pixels * 8) == cudaSuccess);
    CHECK(cuda_ok(cuda::palette_build_map(blocks[i])));
    CHECK(cuda_ok(cuda::palette_predict(blocks[i])));
    CHECK(cuda_ok(cuda::palette_color_contexts(contexts[i])));
    CHECK(cuda_ok(cudaDeviceSynchronize()));
    CHECK(cuda_ok(cudaMemcpy(h_map_single[i].data(), d_map[i], pixels,
                             cudaMemcpyDeviceToHost)));
    CHECK(cuda_ok(cudaMemcpy(h_dst_single[i].data(), d_dst[i], pixels,
                             cudaMemcpyDeviceToHost)));
    CHECK(cuda_ok(cudaMemcpy(h_hash_single[i].data(), d_hash[i], pixels,
                             cudaMemcpyDeviceToHost)));
    CHECK(cuda_ok(cudaMemcpy(h_order_single[i].data(), d_order[i], pixels * 8,
                             cudaMemcpyDeviceToHost)));
    for (std::size_t k = 0; k < pixels; ++k) {
      ++map_samples;
      if (h_map[i][k] != h_map_single[i][k] ||
          h_dst[i][k] != h_dst_single[i][k] ||
          h_hash[i][k] != h_hash_single[i][k]) {
        ++mismatches;
      }
    }
    for (std::size_t k = 0; k < pixels * 8; ++k) {
      if (h_order[i][k] != h_order_single[i][k]) {
        ++mismatches;
      }
    }
  }
  CHECK_EQ(mismatches, 0, "palette batch vs single-block");
  std::printf("  palette: blocks=%d samples=%lld mismatches=%lld\n", kN,
              map_samples, mismatches);
  for (int i = 0; i < kN; ++i) {
    cudaFree(d_src[i]);
    cudaFree(d_colors[i]);
    cudaFree(d_map[i]);
    cudaFree(d_dst[i]);
    cudaFree(d_hash[i]);
    cudaFree(d_order[i]);
  }
}

// --- IntraBC --------------------------------------------------------------

void test_ibc(cw::Workspace& ws) {
  constexpr int kFrameW = 128;
  constexpr int kFrameH = 128;
  constexpr int kIbcChromaW = 64;
  constexpr int kIbcChromaH = 64;

  std::vector<std::uint8_t> h_y(kFrameW * kFrameH);
  std::vector<std::uint8_t> h_u(kIbcChromaW * kIbcChromaH);
  std::vector<std::uint8_t> h_v(kIbcChromaW * kIbcChromaH);
  fill_plane(h_y);
  fill_plane(h_u);
  fill_plane(h_v);

  std::uint8_t* d_y = nullptr;
  std::uint8_t* d_u = nullptr;
  std::uint8_t* d_v = nullptr;
  std::uint8_t* d_dst_y = nullptr;
  std::uint8_t* d_dst_u = nullptr;
  std::uint8_t* d_dst_v = nullptr;
  av2enc::ibc::BlockVector* d_bvs = nullptr;
  std::int64_t* d_sads = nullptr;
  std::uint8_t* d_valid = nullptr;
  CHECK(cudaMalloc(&d_y, h_y.size()) == cudaSuccess);
  CHECK(cudaMalloc(&d_u, h_u.size()) == cudaSuccess);
  CHECK(cudaMalloc(&d_v, h_v.size()) == cudaSuccess);
  CHECK(cudaMalloc(&d_dst_y, 64 * 64) == cudaSuccess);
  CHECK(cudaMalloc(&d_dst_u, 32 * 32) == cudaSuccess);
  CHECK(cudaMalloc(&d_dst_v, 32 * 32) == cudaSuccess);
  CHECK(cudaMalloc(&d_bvs, 32 * sizeof(av2enc::ibc::BlockVector)) ==
        cudaSuccess);
  CHECK(cudaMalloc(&d_sads, 32 * sizeof(std::int64_t)) == cudaSuccess);
  CHECK(cudaMalloc(&d_valid, 32) == cudaSuccess);
  CHECK(cuda_ok(cudaMemcpy(d_y, h_y.data(), h_y.size(),
                           cudaMemcpyHostToDevice)));
  CHECK(cuda_ok(cudaMemcpy(d_u, h_u.data(), h_u.size(),
                           cudaMemcpyHostToDevice)));
  CHECK(cuda_ok(cudaMemcpy(d_v, h_v.data(), h_v.size(),
                           cudaMemcpyHostToDevice)));

  cuda::IbcDeviceFrame frame;
  frame.y = d_y;
  frame.u = d_u;
  frame.v = d_v;
  frame.stride_y = kFrameW;
  frame.stride_u = kIbcChromaW;
  frame.stride_v = kIbcChromaW;
  frame.width = kFrameW;
  frame.height = kFrameH;
  frame.has_chroma = true;
  frame.sub_x = 1;
  frame.sub_y = 1;

  cuda::IbcDeviceGeometry geometry;
  geometry.mi_cols = kFrameW / 4;
  geometry.mi_rows = kFrameH / 4;
  geometry.sb_size = 64;
  geometry.allow_global = true;

  constexpr int kMiRow = 16;  // luma origin 64
  constexpr int kMiCol = 16;
  constexpr int kBsize = 6;  // 16x16

  std::vector<av2enc::ibc::BlockVector> bvs;
  for (int row = 0; row >= -64; row -= 32) {
    for (int col = 0; col >= -64; col -= 16) {
      av2enc::ibc::BlockVector bv;
      bv.row = row * 8;
      bv.col = col * 8;
      bvs.push_back(bv);
    }
  }
  CHECK(bvs.size() <= 32);
  CHECK(cuda_ok(cudaMemcpy(d_bvs, bvs.data(),
                           bvs.size() * sizeof(av2enc::ibc::BlockVector),
                           cudaMemcpyHostToDevice)));

  CHECK_EQ(static_cast<int>(ws.ibc_sad(frame, kMiRow, kMiCol, kBsize,
                                       bvs.data(), static_cast<int>(bvs.size()),
                                       d_sads)),
           static_cast<int>(cuda::Status::kOk), "workspace ibc sad");
  CHECK_EQ(static_cast<int>(ws.ibc_validate(
               geometry, kMiRow, kMiCol, kBsize, bvs.data(),
               static_cast<int>(bvs.size()), d_valid)),
           static_cast<int>(cuda::Status::kOk), "workspace ibc validate");
  CHECK_EQ(static_cast<int>(ws.synchronize()), 0, "ibc sync");

  std::int64_t* d_sads_direct = nullptr;
  std::uint8_t* d_valid_direct = nullptr;
  CHECK(cudaMalloc(&d_sads_direct, 32 * sizeof(std::int64_t)) == cudaSuccess);
  CHECK(cudaMalloc(&d_valid_direct, 32) == cudaSuccess);
  CHECK(cuda_ok(cuda::ibc_sad_batch(frame, kMiRow, kMiCol, kBsize, d_bvs,
                                    static_cast<int>(bvs.size()),
                                    d_sads_direct)));
  CHECK(cuda_ok(cuda::ibc_validate_global(
      geometry, kMiRow, kMiCol, kBsize, d_bvs, static_cast<int>(bvs.size()),
      d_valid_direct)));
  CHECK(cuda_ok(cudaDeviceSynchronize()));

  std::vector<std::int64_t> sads(bvs.size());
  std::vector<std::int64_t> sads_direct(bvs.size());
  std::vector<std::uint8_t> valid(bvs.size());
  std::vector<std::uint8_t> valid_direct(bvs.size());
  CHECK(cuda_ok(cudaMemcpy(sads.data(), d_sads,
                           sads.size() * sizeof(std::int64_t),
                           cudaMemcpyDeviceToHost)));
  CHECK(cuda_ok(cudaMemcpy(sads_direct.data(), d_sads_direct,
                           sads_direct.size() * sizeof(std::int64_t),
                           cudaMemcpyDeviceToHost)));
  CHECK(cuda_ok(cudaMemcpy(valid.data(), d_valid, valid.size(),
                           cudaMemcpyDeviceToHost)));
  CHECK(cuda_ok(cudaMemcpy(valid_direct.data(), d_valid_direct,
                           valid_direct.size(), cudaMemcpyDeviceToHost)));
  long long mismatches = 0;
  for (std::size_t i = 0; i < bvs.size(); ++i) {
    if (sads[i] != sads_direct[i] || valid[i] != valid_direct[i]) {
      ++mismatches;
    }
  }
  CHECK_EQ(mismatches, 0, "ibc sad/validate staging parity");

  // Copy block: workspace vs module entry point.
  const av2enc::ibc::BlockVector copy_bv = bvs.back();
  CHECK_EQ(static_cast<int>(ws.ibc_copy(frame, kMiRow, kMiCol, kBsize, copy_bv,
                                        d_dst_y, 16, d_dst_u, 8, d_dst_v, 8,
                                        16, 16)),
           static_cast<int>(cuda::Status::kOk), "workspace ibc copy");
  CHECK_EQ(static_cast<int>(ws.synchronize()), 0, "ibc copy sync");
  std::uint8_t* d_dst_y2 = nullptr;
  std::uint8_t* d_dst_u2 = nullptr;
  std::uint8_t* d_dst_v2 = nullptr;
  CHECK(cudaMalloc(&d_dst_y2, 64 * 64) == cudaSuccess);
  CHECK(cudaMalloc(&d_dst_u2, 32 * 32) == cudaSuccess);
  CHECK(cudaMalloc(&d_dst_v2, 32 * 32) == cudaSuccess);
  CHECK(cuda_ok(cuda::ibc_copy_block(frame, kMiRow, kMiCol, kBsize, copy_bv,
                                     d_dst_y2, 16, d_dst_u2, 8, d_dst_v2, 8,
                                     16, 16)));
  CHECK(cuda_ok(cudaDeviceSynchronize()));
  std::vector<std::uint8_t> copy_a(16 * 16);
  std::vector<std::uint8_t> copy_b(16 * 16);
  std::vector<std::uint8_t> copy_a_c(8 * 8);
  std::vector<std::uint8_t> copy_b_c(8 * 8);
  CHECK(cuda_ok(cudaMemcpy(copy_a.data(), d_dst_y, copy_a.size(),
                           cudaMemcpyDeviceToHost)));
  CHECK(cuda_ok(cudaMemcpy(copy_b.data(), d_dst_y2, copy_b.size(),
                           cudaMemcpyDeviceToHost)));
  CHECK(cuda_ok(cudaMemcpy(copy_a_c.data(), d_dst_u, copy_a_c.size(),
                           cudaMemcpyDeviceToHost)));
  CHECK(cuda_ok(cudaMemcpy(copy_b_c.data(), d_dst_u2, copy_b_c.size(),
                           cudaMemcpyDeviceToHost)));
  long long copy_mismatches = 0;
  for (std::size_t i = 0; i < copy_a.size(); ++i) {
    if (copy_a[i] != copy_b[i]) ++copy_mismatches;
  }
  for (std::size_t i = 0; i < copy_a_c.size(); ++i) {
    if (copy_a_c[i] != copy_b_c[i]) ++copy_mismatches;
  }
  CHECK_EQ(copy_mismatches, 0, "ibc copy parity");
  std::printf("  ibc: bvs=%zu sad/validate=%lld copy=%lld mismatches=%lld\n",
              bvs.size(), static_cast<long long>(bvs.size()), 320LL,
              mismatches + copy_mismatches);

  cudaFree(d_y);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_dst_y);
  cudaFree(d_dst_u);
  cudaFree(d_dst_v);
  cudaFree(d_dst_y2);
  cudaFree(d_dst_u2);
  cudaFree(d_dst_v2);
  cudaFree(d_bvs);
  cudaFree(d_sads);
  cudaFree(d_valid);
  cudaFree(d_sads_direct);
  cudaFree(d_valid_direct);
}

}  // namespace

int main() {
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    std::printf("no CUDA device\n");
    return 1;
  }
  std::printf("av2enc CUDA intra workspace test\n");
  std::printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);

  cw::Config config;
  config.max_jobs = 1024;
  config.supplemental.max_dc_jobs = 8;
  config.supplemental.max_corner_jobs = 64;
  config.supplemental.max_chroma_jobs = 64;
  config.supplemental.max_tx_blocks = 8;
  config.supplemental.max_sign_jobs = 64;

  const std::uint64_t required = cw::required_device_bytes(config);
  const std::uint64_t persistent = cw::persistent_device_bytes(config);
  const std::uint64_t scratch = cw::scratch_device_bytes(config);
  CHECK_EQ(required, persistent + scratch, "required = persistent + scratch");
  CHECK_EQ(required % cuda::Arena::kAlignment, 0, "required aligned");

  cuda::Arena arena;
  CHECK(cuda::is_ok(arena.init(required + 1024 * 1024)));
  cw::Workspace ws;
  std::string error;
  std::uint64_t need = 0;
  CHECK_EQ(static_cast<int>(ws.init(arena, config, &error, &need)),
           static_cast<int>(cuda::Status::kOk), "workspace init");
  CHECK_EQ(need, required, "init reports required bytes");
  CHECK_EQ(arena.used(), persistent, "init uses persistent bytes only");
  CHECK(ws.supplemental().initialized());
  CHECK(ws.staging() != nullptr);

  test_modes(ws, arena);
  test_mrl(ws);
  test_ibp(ws);
  test_dip(ws);
  test_cfl(ws);
  test_palette(ws);
  test_ibc(ws);

  CHECK_EQ(arena.used(), persistent, "arena used() flat after launches");
  CHECK_EQ(static_cast<int>(ws.synchronize()), 0, "final sync");

  // One-byte-short arena: distinct status, requirement reported, no residue.
  cuda::Arena small;
  CHECK(cuda::is_ok(small.init(required - 1)));
  cw::Workspace short_ws;
  std::uint64_t short_need = 0;
  std::string short_error;
  CHECK_EQ(static_cast<int>(short_ws.init(small, config, &short_error,
                                          &short_need)),
           static_cast<int>(cuda::Status::kArenaExhausted),
           "short arena status");
  CHECK_EQ(short_need, required, "short arena required bytes");
  CHECK_EQ(small.used(), 0, "short arena untouched");
  CHECK(!short_ws.initialized());

  // Invalid configuration.
  cw::Config bad;
  bad.max_jobs = 0;
  CHECK_EQ(cw::required_device_bytes(bad), 0, "invalid config bytes");
  cw::Workspace bad_ws;
  CHECK_EQ(static_cast<int>(bad_ws.init(arena, bad)),
           static_cast<int>(cuda::Status::kInvalidArgument),
           "invalid config status");

  std::printf(
      "  arena: required=%llu persistent=%llu scratch=%llu capacity=%llu "
      "peak=%llu\n",
      static_cast<unsigned long long>(required),
      static_cast<unsigned long long>(persistent),
      static_cast<unsigned long long>(scratch),
      static_cast<unsigned long long>(arena.capacity()),
      static_cast<unsigned long long>(arena.peak_used()));
  std::printf("intra_workspace_test: %s (%lld checks, %d failures)\n",
              g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
