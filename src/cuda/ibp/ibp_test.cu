// AV2 IBP CUDA test: CPU reference vs GPU, bit-exact.
//
// Build/run on the Linux/CUDA box: ./build_cuda_test.sh
//
// Coverage:
//   * every intra mode, angle delta -3..3, block shape 4..64, luma/chroma,
//     interior/top/left/right/bottom positions
//   * enable_ibp 0/1, edge-filter 0/1, filter-type combinations
//   * every IBP weight/mode combination that useIBP can produce
//   * batched API against the per-block API
//   * real 8-bit 4:2:0 luma blocks from tests/data/*.y4m
//   * invalid-argument rejection

#include "ibp.h"

#include "encoder/ibp/ibp.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace cibp = av2enc::cuda::ibp;
namespace ibp = av2enc::ibp;
namespace intra = av2enc::intra;

namespace {

[[noreturn]] void fatal(const char* expr, cudaError_t err, const char* file,
                        int line) {
  std::fprintf(stderr, "CUDA error: %s (%d) from %s at %s:%d\n",
               cudaGetErrorString(err), static_cast<int>(err), expr, file,
               line);
  std::exit(1);
}

#define CUDA_CHECK(expr)                                                  \
  do {                                                                    \
    const cudaError_t err_ = (expr);                                      \
    if (err_ != cudaSuccess) {                                            \
      fatal(#expr, err_, __FILE__, __LINE__);                             \
    }                                                                     \
  } while (false)

int g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, const char* detail, int a,
            int b, const char* file, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 20) {
      std::printf("FAIL %s:%d: %s [%s a=%d b=%d]\n", file, line, what, detail,
                  a, b);
    }
  }
}

#define CHECK(cond) expect((cond), #cond, "", 0, 0, __FILE__, __LINE__)
#define CHECK_EQ(a, b, detail) \
  expect((a) == (b), #a " == " #b, detail, static_cast<int>(a), \
         static_cast<int>(b), __FILE__, __LINE__)

std::uint32_t g_rng = 0x51ed270bU;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

struct Plane {
  int w = 0;
  int h = 0;
  int stride = 0;
  std::vector<std::uint8_t> data;
  std::uint8_t* d_data = nullptr;

  Plane(int width, int height, int pad)
      : w(width),
        h(height),
        stride(width + pad),
        data(static_cast<std::size_t>(stride) * static_cast<std::size_t>(height)) {
  }
  ~Plane() {
    if (d_data != nullptr) {
      cudaFree(d_data);
    }
  }
  Plane(const Plane&) = delete;
  Plane& operator=(const Plane&) = delete;

  // make_plane() returns by value.  CUDA/NVCC does not guarantee elision for
  // a host-side helper the way a modern host compiler does, so make the
  // harness type move-only instead of relying on it.
  Plane(Plane&& other) noexcept
      : w(other.w),
        h(other.h),
        stride(other.stride),
        data(std::move(other.data)),
        d_data(other.d_data) {
    other.d_data = nullptr;
  }
  Plane& operator=(Plane&& other) noexcept {
    if (this != &other) {
      if (d_data != nullptr) {
        cudaFree(d_data);
      }
      w = other.w;
      h = other.h;
      stride = other.stride;
      data = std::move(other.data);
      d_data = other.d_data;
      other.d_data = nullptr;
    }
    return *this;
  }

  std::uint8_t& at(int x, int y) {
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(x)];
  }

  void upload() {
    if (d_data == nullptr) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), data.size()));
    }
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data.size(),
                          cudaMemcpyHostToDevice));
  }
};

Plane make_plane(int w, int h, int pad, int kind) {
  Plane p(w, h, pad);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int v = 0;
      if (kind == 0) {
        v = (x * 5 + y * 3 + ((x * y) >> 3)) & 255;
      } else if (kind == 1) {
        v = static_cast<int>(next_random() & 255U);
      } else {
        v = 128 + ((x - w / 2) + 2 * (y - h / 2)) / 2;
        v = v < 0 ? 0 : (v > 255 ? 255 : v);
      }
      p.at(x, y) = static_cast<std::uint8_t>(v);
    }
  }
  p.upload();
  return p;
}

constexpr int kShapes[][2] = {
    {4, 4},   {4, 8},   {8, 4},   {8, 8},   {8, 16},  {16, 8},
    {16, 16}, {16, 32}, {32, 16}, {32, 32}, {32, 64}, {64, 32},
    {64, 64}, {4, 16},  {16, 4},  {8, 32},  {32, 8},  {4, 64},
    {64, 4},
};

int shape_count() {
  return static_cast<int>(sizeof(kShapes) / sizeof(kShapes[0]));
}

intra::NeighborRequest make_request(const Plane& p, int x, int y, int w, int h,
                                    int plane) {
  intra::NeighborRequest r;
  r.recon.data = p.data.data();
  r.recon.stride = p.stride;
  r.x = x;
  r.y = y;
  r.w = w;
  r.h = h;
  r.max_x = p.w - 1;
  r.max_y = p.h - 1;
  r.have_left = x > 0;
  r.have_above = y > 0;
  r.color_plane = plane;
  if (r.have_above) {
    r.num4_above_right = std::max(0, (r.max_x - (x + w - 1)) / 4);
  }
  if (r.have_left) {
    r.num4_below_left = std::max(0, (r.max_y - (y + h - 1)) / 4);
  }
  return r;
}

cibp::IbpBlock make_device_block(const Plane& p, int x, int y, int w, int h,
                                 int plane, ibp::Mode mode, int delta,
                                 bool enable_ibp, bool edge, int smooth_above,
                                 int smooth_left) {
  cibp::IbpBlock job;
  intra::NeighborRequest r = make_request(p, x, y, w, h, plane);
  cibp::IntraBlock& b = job.intra;
  b.recon_plane = p.d_data;
  b.recon_stride = r.recon.stride;
  b.x0 = r.x;
  b.y0 = r.y;
  b.width = r.w;
  b.height = r.h;
  b.max_x = r.max_x;
  b.max_y = r.max_y;
  b.have_left = r.have_left ? 1 : 0;
  b.have_above = r.have_above ? 1 : 0;
  b.num4_above_right = r.num4_above_right;
  b.num4_below_left = r.num4_below_left;
  b.plane = r.color_plane;
  b.mode = static_cast<cibp::IntraMode>(mode);
  b.angle_delta = delta;
  b.filter_type_above = smooth_above;
  b.filter_type_left = smooth_left;
  b.enable_edge_filter = edge ? 1 : 0;
  job.enable_ibp = enable_ibp ? 1 : 0;
  return job;
}

void compare_case(const Plane& p, int x, int y, int w, int h, int plane,
                  ibp::Mode mode, int delta, bool enable_ibp, bool edge,
                  int smooth_above, int smooth_left, const char* tag,
                  std::uint8_t* d_pred) {
  intra::NeighborRequest req = make_request(p, x, y, w, h, plane);
  intra::IntraEdges edges;
  if (intra::prepare_edges(req, 8, edges) != intra::IntraStatus::kOk) {
    CHECK(false);
    return;
  }
  ibp::PredictParams pp;
  pp.mode = mode;
  pp.angle_delta = delta;
  pp.enable_ibp = enable_ibp;
  pp.enable_edge_filter = edge;
  pp.above_neighbor_is_smooth = smooth_above != 0;
  pp.left_neighbor_is_smooth = smooth_left != 0;

  const std::size_t samples =
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  std::vector<std::uint8_t> cpu(samples);
  std::vector<std::uint8_t> gpu(samples);
  if (ibp::predict(edges, pp, cpu.data(), static_cast<std::size_t>(w)) !=
      intra::IntraStatus::kOk) {
    CHECK(false);
    return;
  }
  const cibp::IbpBlock job = make_device_block(
      p, x, y, w, h, plane, mode, delta, enable_ibp, edge, smooth_above,
      smooth_left);
  CUDA_CHECK(cibp::ibp_predict(job, d_pred, nullptr));
  CUDA_CHECK(cudaMemcpy(gpu.data(), d_pred, samples, cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < samples; ++i) {
    if (cpu[i] != gpu[i]) {
      CHECK_EQ(cpu[i], gpu[i], tag);
      std::printf("  plane=%d mode=%d delta=%d %dx%d at %d,%d sample=%zu\n",
                  plane, static_cast<int>(mode), delta, w, h, x, y, i);
      return;
    }
    ++g_checks;
  }
  // Host-side decision must agree with the CPU decision, including for a
  // candidate that useIBP forces on.
  const cibp::IbpDecision gd = cibp::ibp_decide(job);
  ibp::DecisionParams dp;
  dp.enable_ibp = enable_ibp;
  dp.mode = mode;
  dp.angle_delta = delta;
  dp.w = w;
  dp.h = h;
  dp.plane = plane;
  const ibp::Decision cd = ibp::decide(dp);
  CHECK_EQ(gd.valid, cd.valid, "valid");
  CHECK_EQ(gd.use_ibp, cd.use_ibp, "use");
  CHECK_EQ(gd.p_angle, cd.p_angle, "pangle");
}

void test_predictions() {
  const Plane luma = make_plane(160, 160, 7, 0);
  const Plane chroma = make_plane(80, 80, 5, 2);
  std::uint8_t* d_pred = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pred), 64 * 64));
  int cases = 0;
  for (int s = 0; s < shape_count(); ++s) {
    const int w = kShapes[s][0];
    const int h = kShapes[s][1];
    for (int plane = 0; plane <= 1; ++plane) {
      const Plane& p = plane == 0 ? luma : chroma;
      const int bw = plane == 0 ? w : (w > 32 ? 32 : w);
      const int bh = plane == 0 ? h : (h > 32 ? 32 : h);
      const int xs[] = {0, bw, (p.w - bw) / 2, p.w - bw};
      const int ys[] = {0, bh, (p.h - bh) / 2, p.h - bh};
      for (int xi = 0; xi < 4; ++xi) {
        for (int yi = 0; yi < 4; ++yi) {
          if (xs[xi] < 0 || ys[yi] < 0) continue;
          for (int m = 0; m < intra::kNumModes; ++m) {
            for (int delta = -3; delta <= 3; ++delta) {
              if (!intra::is_directional(static_cast<ibp::Mode>(m)) &&
                  delta != 0) {
                continue;
              }
              for (int enable = 0; enable <= 1; ++enable) {
                for (int edge = 0; edge <= 1; ++edge) {
                  for (int smooth = 0; smooth <= 1; ++smooth) {
                    compare_case(p, xs[xi], ys[yi], bw, bh, plane,
                                 static_cast<ibp::Mode>(m), delta,
                                 enable != 0, edge != 0, smooth, 1 - smooth,
                                 "predict", d_pred);
                    ++cases;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  CUDA_CHECK(cudaFree(d_pred));
  std::printf("GPU prediction cases: %d\n", cases);
  CHECK(cases > 1000);
}

void test_batch() {
  const Plane p = make_plane(128, 128, 4, 1);
  constexpr int kJobs = 96;
  std::vector<cibp::IbpJob> host_jobs(kJobs);
  std::vector<std::uint8_t*> preds(kJobs, nullptr);
  std::vector<std::vector<std::uint8_t>> refs(kJobs);
  std::vector<std::vector<std::uint8_t>> gots(kJobs);
  for (int k = 0; k < kJobs; ++k) {
    const std::size_t ki = static_cast<std::size_t>(k);
    const int s = k % shape_count();
    const int w = kShapes[s][0];
    const int h = kShapes[s][1];
    const int x = (k * 7) % (p.w - w + 1);
    const int y = (k * 11) % (p.h - h + 1);
    const int plane = 0;
    const int m = 1 + (k % 8);
    const int delta = -2 + 2 * (k % 3);
    const bool enable = (k & 1) != 0;
    const std::size_t samples =
        static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&preds[ki]), samples));
    host_jobs[ki].block = make_device_block(p, x, y, w, h, plane,
                                           static_cast<ibp::Mode>(m), delta,
                                           enable, true, 1, 0);
    host_jobs[ki].pred = preds[ki];
    refs[ki].resize(samples);
    gots[ki].resize(samples);

    intra::NeighborRequest req = make_request(p, x, y, w, h, plane);
    intra::IntraEdges edges;
    CHECK(intra::prepare_edges(req, 8, edges) == intra::IntraStatus::kOk);
    ibp::PredictParams pp;
    pp.mode = static_cast<ibp::Mode>(m);
    pp.angle_delta = delta;
    pp.enable_ibp = enable;
    pp.enable_edge_filter = true;
    pp.above_neighbor_is_smooth = true;
    pp.left_neighbor_is_smooth = false;
    CHECK(ibp::predict(edges, pp, refs[ki].data(),
                       static_cast<std::size_t>(w)) ==
          intra::IntraStatus::kOk);
  }

  cibp::IbpJob* d_jobs = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_jobs),
                        sizeof(cibp::IbpJob) * kJobs));
  CUDA_CHECK(cudaMemcpy(d_jobs, host_jobs.data(),
                        sizeof(cibp::IbpJob) * kJobs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cibp::ibp_predict_batch(d_jobs, kJobs, nullptr));
  for (int k = 0; k < kJobs; ++k) {
    const std::size_t ki = static_cast<std::size_t>(k);
    const int s = k % shape_count();
    const std::size_t samples = static_cast<std::size_t>(kShapes[s][0]) *
                                static_cast<std::size_t>(kShapes[s][1]);
    CUDA_CHECK(cudaMemcpy(gots[ki].data(), preds[ki], samples,
                          cudaMemcpyDeviceToHost));
    CHECK(std::memcmp(refs[ki].data(), gots[ki].data(), samples) == 0);
  }
  CUDA_CHECK(cudaFree(d_jobs));
  for (int k = 0; k < kJobs; ++k) {
    CUDA_CHECK(cudaFree(preds[static_cast<std::size_t>(k)]));
  }
  std::printf("GPU batch jobs: %d\n", kJobs);
}

struct Y4mFrame {
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> y;
};

bool read_y4m_luma(const std::string& path, Y4mFrame& frame) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::string header;
  std::getline(in, header);
  if (header.rfind("YUV4MPEG2", 0) != 0) return false;
  int w = 0;
  int h = 0;
  std::size_t pos = 0;
  while (pos < header.size()) {
    while (pos < header.size() && header[pos] == ' ') ++pos;
    if (pos >= header.size()) break;
    const char tag = header[pos++];
    const std::size_t start = pos;
    while (pos < header.size() && header[pos] != ' ') ++pos;
    const int value = std::atoi(header.substr(start, pos - start).c_str());
    if (tag == 'W') w = value;
    if (tag == 'H') h = value;
  }
  if (w <= 0 || h <= 0) return false;
  std::string marker;
  std::getline(in, marker);
  if (marker != "FRAME") return false;
  frame.w = w;
  frame.h = h;
  frame.y.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  in.read(reinterpret_cast<char*>(frame.y.data()),
          static_cast<std::streamsize>(frame.y.size()));
  return static_cast<std::size_t>(in.gcount()) == frame.y.size();
}

void test_real_blocks(const std::string& path, const char* label) {
  Y4mFrame frame;
  if (!read_y4m_luma(path, frame)) {
    std::printf("GPU real blocks: SKIP %s\n", path.c_str());
    return;
  }
  Plane p(frame.w, frame.h, 0);
  std::memcpy(p.data.data(), frame.y.data(), frame.y.size());
  p.upload();
  std::uint8_t* d_pred = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pred), 64 * 64));
  int cases = 0;
  const int block = 16;
  for (int y = 0; y + block <= frame.h; y += 64) {
    for (int x = 0; x + block <= frame.w; x += 64) {
      for (int m = static_cast<int>(ibp::Mode::kV);
           m <= static_cast<int>(ibp::Mode::kD67); ++m) {
        for (int delta = -2; delta <= 2; delta += 2) {
          for (int enable = 0; enable <= 1; ++enable) {
            compare_case(p, x, y, block, block, 0, static_cast<ibp::Mode>(m),
                         delta, enable != 0, true, 1, 0, "real", d_pred);
            ++cases;
          }
        }
      }
    }
  }
  CUDA_CHECK(cudaFree(d_pred));
  std::printf("GPU real blocks %s: %d cases\n", label, cases);
  CHECK(cases > 0);
}

void test_invalid() {
  cibp::IbpBlock job;
  job.intra.width = 4;
  job.intra.height = 4;
  job.intra.mode = static_cast<cibp::IntraMode>(ibp::Mode::kDc);
  std::uint8_t* d_pred = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pred), 64));
  CHECK_EQ(static_cast<int>(cibp::ibp_predict(job, nullptr, nullptr)),
           static_cast<int>(cudaErrorInvalidValue), "null pred");
  job.intra.width = 3;
  CHECK_EQ(static_cast<int>(cibp::ibp_predict(job, d_pred, nullptr)),
           static_cast<int>(cudaErrorInvalidValue), "bad width");
  job.intra.width = 8;
  job.intra.angle_delta = 4;
  CHECK_EQ(static_cast<int>(cibp::ibp_predict(job, d_pred, nullptr)),
           static_cast<int>(cudaErrorInvalidValue), "bad delta");
  CUDA_CHECK(cudaFree(d_pred));
}

}  // namespace

int main(int argc, char** argv) {
  std::printf("ibp_test_cuda: AV2 v1.0.0 IBP GPU parity\n");
  test_invalid();
  test_predictions();
  test_batch();
  const char* root = nullptr;
  if (argc > 1) {
    root = argv[1];
  } else {
#ifdef AV2ENC_TESTDATA_DIR
    root = AV2ENC_TESTDATA_DIR;
#endif
  }
  if (root != nullptr) {
    const std::string dir(root);
    test_real_blocks(dir + "/m0_64x64_1f.y4m", "64x64");
    test_real_blocks(dir + "/m0_640x360_1f.y4m", "640x360");
  }
  if (g_failures != 0) {
    std::printf("ibp_test_cuda: FAIL (%d checks, %d failures)\n", g_checks,
                g_failures);
    return 1;
  }
  std::printf("ibp_test_cuda: PASS (%d checks)\n", g_checks);
  return 0;
}
