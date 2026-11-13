// M2 video intra candidate generator: real-device CPU parity and contract test.
//
// Compares, on sm_86, every candidate's predictor plane and SSE against the CPU
// reference (`src/codec/intra/modes/`), for every luma mode + angle delta and
// every block size, plane and edge state; exercises odd geometry (100x70,
// 640x360 bottom/right partial leaves); checks the optional TX plan against
// `av2enc::txv`; and verifies the arena/status contract.
//
// Build/run: ./build_video_intra_test.sh [--quick]
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "codec/intra/modes/intra_modes.h"
#include "codec/transform/variants/tx_variants.h"
#include "cuda/arena.h"
#include "cuda/error.h"
#include "cuda/pipeline/video/intra/video_intra.h"
#include "cuda/pipeline/video/intra/video_intra_host.h"

namespace vi = av2enc::cuda::video::intra;
namespace intra = av2enc::intra;
namespace txv = av2enc::txv;

namespace {

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

long long g_checks = 0;
long long g_failures = 0;
long long g_candidates = 0;
long long g_samples = 0;
int g_printed = 0;
constexpr int kMaxPrinted = 15;
std::string g_label;

void fail(const char* what, const std::string& detail) {
  ++g_failures;
  if (g_printed < kMaxPrinted) {
    std::printf("FAIL [%s] %s: %s\n", g_label.c_str(), what, detail.c_str());
    ++g_printed;
  }
}

void check(bool condition, const char* what, const std::string& detail = {}) {
  ++g_checks;
  if (!condition) {
    fail(what, detail);
  }
}

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    const cudaError_t err_ = (expr);                                         \
    if (err_ != cudaSuccess) {                                               \
      std::printf("CUDA error %s (%d) at %s:%d\n", cudaGetErrorString(err_), \
                  static_cast<int>(err_), __FILE__, __LINE__);               \
      std::exit(2);                                                          \
    }                                                                        \
  } while (false)

std::uint32_t g_rng = 0x51f3c2a7u;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

// Closed form of `build_scan()` from still_encoder.cpp / still_tx.cu.
int scan_pos(int c, int w, int h) {
  int diagonal = 0;
  int rem = c;
  for (;;) {
    const int lo = diagonal - (w - 1) > 0 ? diagonal - (w - 1) : 0;
    const int hi = diagonal < h - 1 ? diagonal : h - 1;
    const int len = hi - lo + 1;
    if (rem < len) {
      const int row = hi - rem;
      return row * w + (diagonal - row);
    }
    rem -= len;
    ++diagonal;
  }
}

// ---------------------------------------------------------------------------
// Host / device planes
// ---------------------------------------------------------------------------

struct HostPlane {
  std::vector<std::uint8_t> data;
  int w = 0;
  int h = 0;
  int stride = 0;
  int rows = 0;
  std::uint8_t* dev = nullptr;
};

HostPlane make_plane(av2enc::cuda::Arena& arena, int w, int h,
                     std::uint32_t seed) {
  HostPlane plane;
  plane.w = w;
  plane.h = h;
  plane.stride = w + 64;
  plane.rows = h + 64;
  plane.data.resize(static_cast<std::size_t>(plane.stride) *
                    static_cast<std::size_t>(plane.rows));
  g_rng = seed | 1u;
  for (std::uint8_t& v : plane.data) {
    v = static_cast<std::uint8_t>(next_random() & 0xFFu);
  }
  // Smooth gradients make the directional modes genuinely different.
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int base = plane.data[static_cast<std::size_t>(y) *
                                      static_cast<std::size_t>(plane.stride) +
                                  static_cast<std::size_t>(x)];
      plane.data[static_cast<std::size_t>(y) *
                     static_cast<std::size_t>(plane.stride) +
                 static_cast<std::size_t>(x)] =
          static_cast<std::uint8_t>((base + 3 * x + 5 * y) & 0xFF);
    }
  }
  plane.dev = arena.allocate_array<std::uint8_t>(plane.data.size());
  if (plane.dev == nullptr) {
    std::printf("test plane allocation failed\n");
    std::exit(2);
  }
  CUDA_CHECK(cudaMemcpy(plane.dev, plane.data.data(), plane.data.size(),
                        cudaMemcpyHostToDevice));
  return plane;
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

bool cpu_predict(const HostPlane& recon, const vi::BlockJob& job,
                 const vi::Candidate& candidate,
                 std::vector<std::uint8_t>& out) {
  intra::NeighborRequest request;
  request.recon = intra::PlaneView{recon.data.data(), recon.stride};
  request.x = job.x;
  request.y = job.y;
  request.w = job.w;
  request.h = job.h;
  request.max_x = job.max_x;
  request.max_y = job.max_y;
  request.have_left = job.have_left != 0;
  request.have_above = job.have_above != 0;
  request.num4_above_right = job.num4_above_right;
  request.num4_below_left = job.num4_below_left;
  request.color_plane = job.plane;
  intra::IntraEdges edges;
  if (intra::prepare_edges(request, 8, edges) != intra::IntraStatus::kOk) {
    return false;
  }
  intra::PredictParams params;
  params.mode = static_cast<intra::Mode>(candidate.mode);
  params.angle_delta = candidate.angle_delta;
  params.enable_edge_filter = job.enable_edge_filter != 0;
  params.above_neighbor_is_smooth = job.filter_type_above != 0;
  params.left_neighbor_is_smooth = job.filter_type_left != 0;
  out.resize(static_cast<std::size_t>(job.w) * job.h);
  return intra::predict_intra(edges, params, out.data(),
                              static_cast<std::size_t>(job.w)) ==
         intra::IntraStatus::kOk;
}

std::int64_t cpu_sse(const HostPlane& source, const vi::BlockJob& job,
                     const std::uint8_t* pred) {
  std::int64_t sse = 0;
  for (int y = 0; y < job.h; ++y) {
    int sy = job.y + y;
    if (sy > job.source_max_y) sy = job.source_max_y;
    for (int x = 0; x < job.w; ++x) {
      int sx = job.x + x;
      if (sx > job.source_max_x) sx = job.source_max_x;
      const int sample = static_cast<int>(
          source.data[static_cast<std::size_t>(sy) *
                          static_cast<std::size_t>(source.stride) +
                      static_cast<std::size_t>(sx)]);
      const int diff =
          sample - static_cast<int>(pred[static_cast<std::size_t>(y) * job.w + x]);
      sse += static_cast<std::int64_t>(diff) * diff;
    }
  }
  return sse;
}

// ---------------------------------------------------------------------------
// One block: evaluate, then compare every candidate against the CPU reference
// ---------------------------------------------------------------------------

struct Rig {
  av2enc::cuda::Arena* arena = nullptr;
  vi::Workspace* workspace = nullptr;
};

bool compare_block(const Rig& rig, const HostPlane& recon,
                   const HostPlane& source, const vi::BlockJob& job,
                   bool run_tx) {
  vi::Candidate candidates[vi::kMaxCandidates];
  const int expected = vi::build_candidates(
      job.plane, job.mode_mask, job.include_angle_deltas, candidates,
      vi::kMaxCandidates);
  if (expected <= 0) {
    fail("build_candidates", "no candidates");
    return false;
  }

  vi::EvaluateOptions options;
  options.run_tx = run_tx;
  std::string error;
  const av2enc::cuda::Status status =
      rig.workspace->evaluate(&job, 1, options, nullptr, &error);
  if (status != av2enc::cuda::Status::kOk) {
    fail("evaluate", std::string(av2enc::cuda::status_name(status)) + " " +
                         error);
    return false;
  }
  if (rig.workspace->candidate_total() != expected) {
    fail("candidate_total", std::to_string(rig.workspace->candidate_total()) +
                                " vs " + std::to_string(expected));
    return false;
  }
  if (rig.workspace->synchronize() != av2enc::cuda::Status::kOk) {
    fail("synchronize", "failed");
    return false;
  }

  std::vector<vi::CandidateResult> results(
      static_cast<std::size_t>(expected));
  std::vector<std::uint8_t> pred(rig.workspace->prediction_elems());
  if (rig.workspace->download_results(results.data(), expected) !=
          av2enc::cuda::Status::kOk ||
      rig.workspace->download_prediction(pred.data(), pred.size()) !=
          av2enc::cuda::Status::kOk) {
    fail("download", "failed");
    return false;
  }

  std::vector<vi::TxCandidateResult> tx_results;
  std::vector<std::int16_t> residual;
  std::vector<std::int32_t> levels;
  if (run_tx) {
    tx_results.resize(static_cast<std::size_t>(expected));
    residual.resize(rig.workspace->prediction_elems());
    levels.resize(rig.workspace->tx_level_elems());
    if (rig.workspace->download_tx(tx_results.data(), expected) !=
            av2enc::cuda::Status::kOk ||
        rig.workspace->download_residual(residual.data(), residual.size()) !=
            av2enc::cuda::Status::kOk ||
        rig.workspace->download_prediction(pred.data(), pred.size()) !=
            av2enc::cuda::Status::kOk) {
      fail("download tx", "failed");
      return false;
    }
    // The level buffer is a device pointer; copy it through a small helper.
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(levels.size()) * sizeof(std::int32_t);
    if (bytes > 0) {
      CUDA_CHECK(cudaMemcpy(levels.data(), rig.workspace->levels_device(),
                            static_cast<std::size_t>(bytes),
                            cudaMemcpyDeviceToHost));
    }
  }

  std::vector<std::uint8_t> want;
  const auto tx_params = txv::make_tx_params(job.w, job.h, txv::TxType::kDctDct);
  for (int c = 0; c < expected; ++c) {
    const vi::Candidate& candidate = candidates[c];
    if (!cpu_predict(recon, job, candidate, want)) {
      fail("cpu predict", "prepare_edges/predict_intra failed");
      return false;
    }
    const vi::CandidateResult& result = results[static_cast<std::size_t>(c)];
    const std::uint8_t* got =
        pred.data() + static_cast<std::size_t>(result.pred_offset);
    if (result.block != 0 || result.mode != candidate.mode ||
        result.angle_delta != candidate.angle_delta ||
        result.plane != job.plane) {
      fail("result identity", "mismatch");
      return false;
    }
    const int memcmp_status = std::memcmp(got, want.data(), want.size());
    if (memcmp_status != 0) {
      std::size_t first = 0;
      while (first < want.size() && got[first] == want[first]) ++first;
      fail("prediction",
           "mode=" + std::to_string(candidate.mode) + " delta=" +
               std::to_string(candidate.angle_delta) + " size=" +
               std::to_string(job.w) + "x" + std::to_string(job.h) +
               " first=" + std::to_string(first) + " got=" +
               std::to_string(static_cast<int>(got[first])) + " want=" +
               std::to_string(static_cast<int>(want[first])));
      return false;
    }
    const std::int64_t want_sse = cpu_sse(source, job, want.data());
    if (result.sse != want_sse) {
      fail("sse", "mode=" + std::to_string(candidate.mode) + " delta=" +
                      std::to_string(candidate.angle_delta) + " got=" +
                      std::to_string(result.sse) + " want=" +
                      std::to_string(want_sse));
      return false;
    }
    ++g_candidates;
    g_samples += static_cast<long long>(job.w) * job.h;

    if (run_tx) {
      if (tx_params.has_value() == false) {
        fail("tx params", "unsupported size");
        return false;
      }
      const txv::TxParams& params = *tx_params;
      const int adj_w = params.w;
      const int adj_h = params.h;
      const int adj_elems = adj_w * adj_h;
      std::vector<std::int16_t> coded(
          static_cast<std::size_t>(job.w) * job.h);
      for (int y = 0; y < job.h; ++y) {
        int sy = job.y + y;
        if (sy > job.source_max_y) sy = job.source_max_y;
        for (int x = 0; x < job.w; ++x) {
          int sx = job.x + x;
          if (sx > job.source_max_x) sx = job.source_max_x;
          coded[static_cast<std::size_t>(y) * job.w + x] =
              static_cast<std::int16_t>(
                  static_cast<int>(source.data[
                      static_cast<std::size_t>(sy) *
                          static_cast<std::size_t>(source.stride) +
                      static_cast<std::size_t>(sx)]) -
                  static_cast<int>(want[static_cast<std::size_t>(y) * job.w +
                                        x]));
        }
      }
      std::vector<std::int16_t> adjusted(
          static_cast<std::size_t>(adj_elems));
      std::vector<double> coeff(static_cast<std::size_t>(adj_elems));
      std::vector<std::int32_t> cpu_levels(static_cast<std::size_t>(adj_elems));
      txv::downsample_residual(coded.data(), params, adjusted.data());
      txv::forward_transform(adjusted.data(), params, coeff.data(),
                             cpu_levels.data());
      int cpu_eob = 0;
      for (int i = 0; i < adj_elems; ++i) {
        if (cpu_levels[static_cast<std::size_t>(scan_pos(i, adj_w, adj_h))] !=
            0) {
          cpu_eob = i + 1;
        }
      }
      const vi::TxCandidateResult& tx = tx_results[static_cast<std::size_t>(c)];
      if (tx.eob != cpu_eob) {
        fail("tx eob", "mode=" + std::to_string(candidate.mode) +
                           " got=" + std::to_string(tx.eob) + " want=" +
                           std::to_string(cpu_eob));
        return false;
      }
      if (tx.levels_offset < 0 ||
          static_cast<std::size_t>(tx.levels_offset + adj_elems) >
              levels.size()) {
        fail("tx levels offset", std::to_string(tx.levels_offset));
        return false;
      }
      const std::int32_t* got_levels =
          levels.data() + static_cast<std::size_t>(tx.levels_offset);
      if (std::memcmp(got_levels, cpu_levels.data(),
                      static_cast<std::size_t>(adj_elems) *
                          sizeof(std::int32_t)) != 0) {
        std::size_t first = 0;
        while (first < static_cast<std::size_t>(adj_elems) &&
               got_levels[first] == cpu_levels[first]) {
          ++first;
        }
        fail("tx levels", "mode=" + std::to_string(candidate.mode) +
                              " first=" + std::to_string(first) + " got=" +
                              std::to_string(got_levels[first]) + " want=" +
                              std::to_string(cpu_levels[first]));
        return false;
      }
      // Residual = source - prediction at the coded size.
      const std::int16_t* got_residual =
          residual.data() + static_cast<std::size_t>(result.pred_offset);
      if (std::memcmp(got_residual, coded.data(),
                      coded.size() * sizeof(std::int16_t)) != 0) {
        fail("tx residual", "mismatch");
        return false;
      }
      ++g_checks;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Job construction
// ---------------------------------------------------------------------------

vi::BlockJob make_job(const HostPlane& recon, const HostPlane& source,
                      const vi::AvailRequest& avail, int plane,
                      std::uint32_t mask = vi::kAllModesMask,
                      int include_deltas = 1, int edge_filter = 1,
                      int filter_above = 0, int filter_left = 0) {
  const vi::Availability a = vi::derive_availability(avail);
  vi::BlockJob job{};
  job.recon_plane = recon.dev;
  job.recon_stride = recon.stride;
  job.source_plane = source.dev;
  job.source_stride = source.stride;
  job.x = avail.x;
  job.y = avail.y;
  job.w = avail.w;
  job.h = avail.h;
  job.plane = plane;
  job.max_x = a.max_x;
  job.max_y = a.max_y;
  job.source_max_x = source.w - 1;
  job.source_max_y = source.h - 1;
  job.have_left = a.have_left;
  job.have_above = a.have_above;
  job.num4_above_right = a.num4_above_right;
  job.num4_below_left = a.num4_below_left;
  job.filter_type_above = filter_above;
  job.filter_type_left = filter_left;
  job.enable_edge_filter = edge_filter;
  job.mode_mask = mask;
  job.include_angle_deltas = include_deltas;
  return job;
}

vi::AvailRequest avail_req(int x, int y, int w, int h, int plane_w, int plane_h,
                           int tile_x0 = 0, int tile_y0 = 0, int tile_x1 = 0,
                           int tile_y1 = 0) {
  vi::AvailRequest req;
  req.x = x;
  req.y = y;
  req.w = w;
  req.h = h;
  req.plane_w = plane_w;
  req.plane_h = plane_h;
  req.tile_x0 = tile_x0;
  req.tile_y0 = tile_y0;
  req.tile_x1 = tile_x1;
  req.tile_y1 = tile_y1;
  return req;
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------

struct Size {
  int w;
  int h;
};

const Size kSizes[] = {{4, 4},   {8, 8},   {16, 16}, {32, 32}, {64, 64},
                       {8, 16},  {16, 8},  {64, 8},  {8, 64},  {16, 64},
                       {64, 16}, {32, 8},  {8, 32}};

void test_matrix(const Rig& rig, const HostPlane& luma_recon,
                 const HostPlane& luma_source, const HostPlane& u_recon,
                 const HostPlane& u_source, const HostPlane& v_recon,
                 const HostPlane& v_source) {
  const struct {
    int plane;
    const HostPlane* recon;
    const HostPlane* source;
    const char* name;
  } planes[3] = {
      {0, &luma_recon, &luma_source, "Y"},
      {1, &u_recon, &u_source, "U"},
      {2, &v_recon, &v_source, "V"},
  };
  for (const auto& p : planes) {
    for (const Size& size : kSizes) {
      for (int edge = 0; edge < 8; ++edge) {
        int x = 24;
        int y = 24;
        int tile_x0 = 0;
        int tile_y0 = 0;
        int tile_x1 = 0;
        int tile_y1 = 0;
        switch (edge) {
          case 0:
            break;
          case 1:
            x = 0;
            y = 0;
            break;
          case 2:
            x = 0;
            break;
          case 3:
            y = 0;
            break;
          case 4:
            x = std::max(0, p.recon->w - size.w / 2);
            break;
          case 5:
            y = std::max(0, p.recon->h - size.h / 2);
            break;
          case 6:
            x = std::max(0, p.recon->w - size.w / 2);
            y = std::max(0, p.recon->h - size.h / 2);
            break;
          case 7:
            x = 32;
            y = 32;
            tile_x0 = 32;
            tile_y0 = 32;
            tile_x1 = std::min(p.recon->w, 96);
            tile_y1 = std::min(p.recon->h, 96);
            break;
          default:
            break;
        }
        if (x >= p.recon->w || y >= p.recon->h) {
          continue;
        }
        g_label = std::string(p.name) + " " + std::to_string(size.w) + "x" +
                  std::to_string(size.h) + " edge" + std::to_string(edge);
        const vi::AvailRequest req =
            avail_req(x, y, size.w, size.h, p.recon->w, p.recon->h, tile_x0,
                      tile_y0, tile_x1, tile_y1);
        const int filter_above = edge & 1;
        const int filter_left = (edge >> 1) & 1;
        const vi::BlockJob job =
            make_job(*p.recon, *p.source, req, p.plane, vi::kAllModesMask, 1,
                     edge == 7 ? 0 : 1, filter_above, filter_left);
        if (!compare_block(rig, *p.recon, *p.source, job, false)) {
          return;
        }
      }
    }
  }
}

void test_odd_geometry(const Rig& rig, const HostPlane& luma_recon,
                       const HostPlane& luma_source, const HostPlane& u_recon,
                       const HostPlane& u_source,
                       const HostPlane& v_recon,
                       const HostPlane& v_source) {
  // 100x70 plane: bottom/right partial 64x64 leaf, partial rows and columns.
  const struct {
    int x;
    int y;
    int w;
    int h;
  } blocks[] = {
      {0, 0, 64, 64},   {64, 64, 32, 8}, {68, 38, 32, 32},
      {0, 64, 64, 8},   {96, 0, 4, 64},  {32, 32, 16, 16},
      {0, 34, 64, 32},  {50, 50, 32, 16},
  };
  for (const auto& b : blocks) {
    g_label = "odd 100x70 " + std::to_string(b.w) + "x" + std::to_string(b.h);
    const vi::BlockJob job =
        make_job(luma_recon, luma_source,
                 avail_req(b.x, b.y, b.w, b.h, luma_recon.w, luma_recon.h), 0);
    if (!compare_block(rig, luma_recon, luma_source, job, false)) {
      return;
    }
  }

  // 640x360 bottom/right partial leaves.
  av2enc::cuda::Arena scratch;
  const av2enc::cuda::Status init_status = scratch.init_mb(128);
  if (init_status != av2enc::cuda::Status::kOk) {
    fail("arena", "640x360 scratch");
    return;
  }
  HostPlane big_recon = make_plane(scratch, 640, 360, 0x1234u);
  HostPlane big_source = make_plane(scratch, 640, 360, 0x5678u);
  const struct {
    int x;
    int y;
    int w;
    int h;
  } bottom[] = {
      {0, 320, 64, 64},   {576, 320, 64, 64}, {320, 320, 64, 64},
      {576, 0, 64, 64},   {576, 296, 64, 32}, {624, 344, 16, 16},
      {632, 100, 16, 16}, {100, 352, 32, 8},  {608, 352, 32, 8},
  };
  for (const auto& b : bottom) {
    g_label = "640x360 " + std::to_string(b.w) + "x" + std::to_string(b.h) +
              "@" + std::to_string(b.x) + "," + std::to_string(b.y);
    const vi::BlockJob job =
        make_job(big_recon, big_source,
                 avail_req(b.x, b.y, b.w, b.h, 640, 360), 0);
    if (!compare_block(rig, big_recon, big_source, job, false)) {
      return;
    }
  }

  // 50x35 chroma plane (4:2:0 of 100x70): partial 32x32 chroma leaves and the
  // §7.13.2.1 chroma > 32 rule with a 64-wide chroma job.
  const struct {
    int x;
    int y;
    int w;
    int h;
    int plane;
  } chroma[] = {
      {32, 32, 32, 32, 1}, {0, 0, 32, 32, 1}, {24, 20, 32, 32, 2},
      {0, 0, 64, 64, 1},   {0, 0, 64, 32, 2},
  };
  for (const auto& b : chroma) {
    g_label = "chroma odd " + std::to_string(b.w) + "x" +
              std::to_string(b.h) + " plane" + std::to_string(b.plane);
    const HostPlane& recon = b.plane == 1 ? u_recon : v_recon;
    const HostPlane& source = b.plane == 1 ? u_source : v_source;
    const vi::BlockJob job =
        make_job(recon, source,
                 avail_req(b.x, b.y, b.w, b.h, recon.w, recon.h), b.plane);
    if (!compare_block(rig, recon, source, job, false)) {
      return;
    }
  }
}

void test_masks(const Rig& rig, const HostPlane& recon,
                const HostPlane& source) {
  const struct {
    std::uint32_t mask;
    int deltas;
    int expected;
  } masks[] = {
      {vi::kDcOnlyMask, 1, 1},
      {vi::kNoDirectionalMask, 1, 5},
      {1u << 1, 0, 1},
      {1u << 1, 1, 7},
      {vi::kAllModesMask, 0, 13},
  };
  for (const auto& m : masks) {
    g_label = "mask " + std::to_string(m.mask) + " deltas" +
              std::to_string(m.deltas);
    const vi::BlockJob job =
        make_job(recon, source, avail_req(24, 24, 16, 16, recon.w, recon.h), 0,
                 m.mask, m.deltas);
    vi::Candidate candidates[vi::kMaxCandidates];
    check(vi::build_candidates(0, m.mask, m.deltas, candidates,
                               vi::kMaxCandidates) == m.expected,
          "candidate count");
    if (!compare_block(rig, recon, source, job, false)) {
      return;
    }
  }
}

void test_tx(const Rig& rig, const HostPlane& recon, const HostPlane& source) {
  for (const Size& size : kSizes) {
    g_label = "tx " + std::to_string(size.w) + "x" + std::to_string(size.h);
    const vi::BlockJob job =
        make_job(recon, source,
                 avail_req(48, 48, size.w, size.h, recon.w, recon.h), 0);
    if (!compare_block(rig, recon, source, job, true)) {
      return;
    }
  }
  // Chroma TX (one candidate) with a >32 chroma leaf.
  g_label = "tx chroma 64x64";
  const vi::BlockJob job =
      make_job(recon, source, avail_req(0, 0, 64, 64, recon.w, recon.h), 1);
  if (!compare_block(rig, recon, source, job, true)) {
    return;
  }
}

void test_batch(const Rig& rig, const HostPlane& luma_recon,
                const HostPlane& luma_source, const HostPlane& u_recon,
                const HostPlane& u_source) {
  // Mix sizes and planes in one evaluate() call, then compare each block with
  // its own single-block call.
  struct Entry {
    const HostPlane* recon;
    const HostPlane* source;
    int plane;
    int x;
    int y;
    int w;
    int h;
    std::uint32_t mask;
  };
  const Entry entries[] = {
      {&luma_recon, &luma_source, 0, 16, 16, 64, 64, vi::kAllModesMask},
      {&luma_recon, &luma_source, 0, 96, 16, 32, 32, vi::kNoDirectionalMask},
      {&luma_recon, &luma_source, 0, 128, 48, 16, 16, vi::kDcOnlyMask},
      {&luma_recon, &luma_source, 0, 0, 0, 4, 4, vi::kAllModesMask},
      {&u_recon, &u_source, 1, 16, 16, 32, 32, vi::kAllModesMask},
      {&u_recon, &u_source, 2, 0, 0, 8, 8, vi::kAllModesMask},
  };
  const int count = static_cast<int>(sizeof(entries) / sizeof(entries[0]));
  std::vector<vi::BlockJob> jobs;
  int expected_total = 0;
  for (const Entry& e : entries) {
    jobs.push_back(make_job(*e.recon, *e.source,
                            avail_req(e.x, e.y, e.w, e.h, e.recon->w,
                                      e.recon->h),
                            e.plane, e.mask));
    vi::Candidate candidates[vi::kMaxCandidates];
    expected_total += vi::build_candidates(e.plane, e.mask, 1, candidates,
                                           vi::kMaxCandidates);
  }
  g_label = "batch mixed";
  std::string error;
  const av2enc::cuda::Status status =
      rig.workspace->evaluate(jobs.data(), count, {}, nullptr, &error);
  if (status != av2enc::cuda::Status::kOk) {
    fail("batch evaluate", error);
    return;
  }
  check(rig.workspace->candidate_total() == expected_total, "batch total");
  check(rig.workspace->block_count() == count, "batch blocks");
  std::vector<int> offsets(static_cast<std::size_t>(count));
  std::vector<int> counts(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    offsets[static_cast<std::size_t>(i)] = rig.workspace->candidate_offset(i);
    counts[static_cast<std::size_t>(i)] = rig.workspace->candidate_count(i);
  }
  rig.workspace->synchronize();
  std::vector<vi::CandidateResult> batch_results(
      static_cast<std::size_t>(expected_total));
  std::vector<std::uint8_t> batch_pred(rig.workspace->prediction_elems());
  if (rig.workspace->download_results(batch_results.data(), expected_total) !=
          av2enc::cuda::Status::kOk ||
      rig.workspace->download_prediction(batch_pred.data(), batch_pred.size()) !=
          av2enc::cuda::Status::kOk) {
    fail("batch download", "failed");
    return;
  }

  for (int i = 0; i < count; ++i) {
    const Entry& e = entries[i];
    vi::Candidate candidates[vi::kMaxCandidates];
    const int n = vi::build_candidates(e.plane, e.mask, 1, candidates,
                                       vi::kMaxCandidates);
    check(counts[static_cast<std::size_t>(i)] == n, "batch per-block count");
    const int offset = offsets[static_cast<std::size_t>(i)];
    // Single-block reference run on a throwaway result buffer.
    std::string single_error;
    if (rig.workspace->evaluate(&jobs[static_cast<std::size_t>(i)], 1, {},
                                nullptr, &single_error) !=
        av2enc::cuda::Status::kOk) {
      fail("batch single evaluate", single_error);
      return;
    }
    rig.workspace->synchronize();
    std::vector<vi::CandidateResult> single(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> single_pred(rig.workspace->prediction_elems());
    rig.workspace->download_results(single.data(), n);
    rig.workspace->download_prediction(single_pred.data(), single_pred.size());
    for (int c = 0; c < n; ++c) {
      const vi::CandidateResult& a =
          batch_results[static_cast<std::size_t>(offset + c)];
      const vi::CandidateResult& b = single[static_cast<std::size_t>(c)];
      check(a.mode == b.mode && a.angle_delta == b.angle_delta &&
                a.plane == b.plane && a.sse == b.sse,
            "batch result identity",
            "block " + std::to_string(i) + " candidate " + std::to_string(c));
      const std::uint8_t* pa =
          batch_pred.data() + static_cast<std::size_t>(a.pred_offset);
      const std::uint8_t* pb =
          single_pred.data() + static_cast<std::size_t>(b.pred_offset);
      check(std::memcmp(pa, pb, static_cast<std::size_t>(e.w) * e.h) == 0,
            "batch prediction",
            "block " + std::to_string(i) + " candidate " + std::to_string(c));
    }
  }
}

void test_best(const Rig& rig) {
  av2enc::cuda::Arena scratch;
  if (scratch.init_mb(64) != av2enc::cuda::Status::kOk) {
    fail("arena", "best");
    return;
  }
  HostPlane flat = make_plane(scratch, 64, 64, 0x9a9au);
  std::memset(flat.data.data(), 128, flat.data.size());
  CUDA_CHECK(cudaMemcpy(flat.dev, flat.data.data(), flat.data.size(),
                        cudaMemcpyHostToDevice));
  g_label = "best tie";
  const vi::BlockJob job = make_job(flat, flat, avail_req(16, 16, 16, 16, 64, 64),
                                    0);
  if (rig.workspace->evaluate(&job, 1, {}, nullptr, nullptr) !=
      av2enc::cuda::Status::kOk) {
    fail("best evaluate", "failed");
    return;
  }
  vi::BestResult best{};
  if (rig.workspace->download_best(&best, 1) != av2enc::cuda::Status::kOk) {
    fail("best download", "failed");
    return;
  }
  check(best.mode == 0 && best.angle_delta == 0, "tie-break picks DC first");
  check(best.candidate_index == 0, "tie-break candidate index");
  check(best.sse == 0, "flat SSE");
}

void test_contract(const Rig& rig, const HostPlane& recon,
                   const HostPlane& source) {
  g_label = "contract";

  // Invalid job dimensions.
  {
    vi::BlockJob job =
        make_job(recon, source, avail_req(16, 16, 16, 16, recon.w, recon.h), 0);
    job.w = 5;
    std::string error;
    check(rig.workspace->evaluate(&job, 1, {}, nullptr, &error) ==
              av2enc::cuda::Status::kInvalidArgument,
          "bad width rejected");
  }
  // Contradictory availability.
  {
    vi::BlockJob job =
        make_job(recon, source, avail_req(0, 0, 16, 16, recon.w, recon.h), 0);
    job.have_left = 1;
    std::string error;
    check(rig.workspace->evaluate(&job, 1, {}, nullptr, &error) ==
              av2enc::cuda::Status::kInvalidArgument,
          "bad availability rejected");
  }
  // TX requested on a workspace without TX buffers.
  {
    av2enc::cuda::Arena small;
    if (small.init(16u << 20) != av2enc::cuda::Status::kOk) {
      fail("arena", "no-tx capacity");
      return;
    }
    vi::Config no_tx;
    no_tx.max_blocks = 1;
    no_tx.max_candidates = vi::kMaxCandidates;
    no_tx.max_block_area = 16 * 16;
    no_tx.enable_tx = false;
    vi::Workspace no_tx_workspace;
    if (no_tx_workspace.init(small, no_tx) != av2enc::cuda::Status::kOk) {
      fail("arena", "no-tx init");
      return;
    }
    const vi::BlockJob job =
        make_job(recon, source, avail_req(16, 16, 16, 16, recon.w, recon.h), 0);
    vi::EvaluateOptions options;
    options.run_tx = true;
    std::string error;
    check(no_tx_workspace.evaluate(&job, 1, options, nullptr, &error) ==
              av2enc::cuda::Status::kInvalidArgument,
          "TX without reservation rejected");
  }
  // Capacity overflow: a workspace sized for two blocks rejects three.
  {
    av2enc::cuda::Arena small;
    if (small.init(64u << 20) != av2enc::cuda::Status::kOk) {
      fail("arena", "small capacity");
      return;
    }
    vi::Config small_config;
    small_config.max_blocks = 2;
    small_config.max_candidates = vi::kMaxCandidates;
    small_config.max_block_area = 16 * 16;
    vi::Workspace small_workspace;
    if (small_workspace.init(small, small_config) !=
        av2enc::cuda::Status::kOk) {
      fail("arena", "small workspace init");
      return;
    }
    std::vector<vi::BlockJob> jobs;
    for (int i = 0; i < 3; ++i) {
      jobs.push_back(make_job(recon, source,
                              avail_req(16, 16, 16, 16, recon.w, recon.h), 0));
    }
    std::string error;
    check(small_workspace.evaluate(jobs.data(), 3, {}, nullptr, &error) ==
              av2enc::cuda::Status::kArenaExhausted,
          "capacity overflow reports arena exhausted");
  }
  // Download capacity checks need a non-empty plan.
  {
    const vi::BlockJob job =
        make_job(recon, source, avail_req(16, 16, 16, 16, recon.w, recon.h), 0);
    if (rig.workspace->evaluate(&job, 1, {}, nullptr, nullptr) !=
        av2enc::cuda::Status::kOk) {
      fail("contract", "single evaluate");
      return;
    }
    rig.workspace->synchronize();
    vi::CandidateResult one{};
    check(rig.workspace->download_results(&one, 0) ==
              av2enc::cuda::Status::kInvalidArgument,
          "download capacity");
  }
  // Empty batch is a no-op.
  check(rig.workspace->evaluate(nullptr, 0, {}, nullptr, nullptr) ==
            av2enc::cuda::Status::kOk,
        "empty batch ok");
  check(rig.workspace->candidate_total() == 0, "empty batch total");
  // Uninitialised workspace.
  {
    vi::Workspace uninitialised;
    std::string error;
    check(uninitialised.evaluate(nullptr, 0, {}, nullptr, &error) ==
              av2enc::cuda::Status::kNotInitialized,
          "uninitialised evaluate");
    check(uninitialised.synchronize() ==
              av2enc::cuda::Status::kNotInitialized,
          "uninitialised sync");
  }
}

void test_arena_contract() {
  g_label = "arena";
  vi::Config config;
  config.max_blocks = 2;
  config.max_candidates = vi::kMaxCandidates;
  config.max_block_area = 64 * 64;
  config.enable_tx = false;
  const std::uint64_t need = vi::required_device_bytes(config);
  check(need > 0, "required bytes");

  {
    av2enc::cuda::Arena arena;
    if (arena.init(need) != av2enc::cuda::Status::kOk) {
      fail("arena init exact", "failed");
      return;
    }
    vi::Workspace workspace;
    std::uint64_t reported = 0;
    const av2enc::cuda::Status status =
        workspace.init(arena, config, nullptr, &reported);
    check(status == av2enc::cuda::Status::kOk, "exact init");
    check(reported == need, "reported requirement");
    check(arena.used() == need, "exact consumption",
          std::to_string(arena.used()) + " vs " + std::to_string(need));
  }
  {
    av2enc::cuda::Arena arena;
    if (arena.init(need - 1) != av2enc::cuda::Status::kOk) {
      fail("arena init short", "failed");
      return;
    }
    vi::Workspace workspace;
    std::uint64_t reported = 0;
    const av2enc::cuda::Status status =
        workspace.init(arena, config, nullptr, &reported);
    check(status == av2enc::cuda::Status::kArenaExhausted, "short init status");
    check(reported == need, "short init requirement");
    check(arena.used() == 0, "short init restores the bump pointer");
    check(!workspace.initialized(), "short init leaves uninitialised");
  }
  {
    av2enc::cuda::Arena arena;
    if (arena.init(1u << 20) != av2enc::cuda::Status::kOk) {
      fail("arena init invalid", "failed");
      return;
    }
    vi::Workspace workspace;
    vi::Config bad;
    bad.max_blocks = 0;
    std::uint64_t reported = 123;
    check(workspace.init(arena, bad, nullptr, &reported) ==
              av2enc::cuda::Status::kInvalidArgument,
          "invalid config");
    check(reported == 0, "invalid config requirement");
  }
  {
    // TX reservation changes the requirement exactly.
    vi::Config tx_config = config;
    tx_config.enable_tx = true;
    const std::uint64_t tx_need = vi::required_device_bytes(tx_config);
    check(tx_need > need, "TX reservation grows the arena");
    av2enc::cuda::Arena arena;
    if (arena.init(tx_need) != av2enc::cuda::Status::kOk) {
      fail("arena init tx", "failed");
      return;
    }
    vi::Workspace workspace;
    check(workspace.init(arena, tx_config) == av2enc::cuda::Status::kOk,
          "tx init");
    check(arena.used() == tx_need, "tx exact consumption");
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--quick") == 0) {
      quick = true;
    } else {
      std::printf("usage: %s [--quick]\n", argv[0]);
      return 2;
    }
  }

  av2enc::cuda::Arena arena;
  if (arena.init_mb(512) != av2enc::cuda::Status::kOk) {
    std::printf("arena init failed\n");
    return 2;
  }

  // 256x192 luma with 128x96 chroma: room for interior and partial leaves.
  HostPlane luma_recon = make_plane(arena, 256, 192, 0xc0ffeeu);
  HostPlane luma_source = make_plane(arena, 256, 192, 0xbeefu);
  HostPlane u_recon = make_plane(arena, 128, 96, 0x12345u);
  HostPlane u_source = make_plane(arena, 128, 96, 0x6789au);
  HostPlane v_recon = make_plane(arena, 120, 88, 0x55aau);
  HostPlane v_source = make_plane(arena, 120, 88, 0xaa55u);

  vi::Config config;
  config.max_blocks = 8;
  config.max_candidates = vi::kMaxCandidates;
  config.max_block_area = 64 * 64;
  config.enable_tx = true;
  vi::Workspace workspace;
  if (workspace.init(arena, config) != av2enc::cuda::Status::kOk) {
    std::printf("workspace init failed\n");
    return 2;
  }
  const Rig rig{&arena, &workspace};

  test_arena_contract();
  test_contract(rig, luma_recon, luma_source);
  test_matrix(rig, luma_recon, luma_source, u_recon, u_source, v_recon,
              v_source);
  test_odd_geometry(rig, luma_recon, luma_source, u_recon, u_source, v_recon,
                    v_source);
  test_masks(rig, luma_recon, luma_source);
  if (!quick) {
    test_tx(rig, luma_recon, luma_source);
  }
  test_batch(rig, luma_recon, luma_source, u_recon, u_source);
  test_best(rig);

  std::printf(
      "video_intra_test: %s (%lld checks, %lld candidates, %lld samples, "
      "%lld failures)\n",
      g_failures == 0 ? "PASS" : "FAIL", g_checks, g_candidates, g_samples,
      g_failures);
  return g_failures == 0 ? 0 : 1;
}
