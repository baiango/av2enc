// CUDA M0 GPU job: runtime, transform and intra component integration.
//
// Everything arithmetic lives in the shared components owned by the other
// agents (docs/cuda-interface.md):
//   * runtime:   av2enc::cuda::Context / Arena / StagingBuffer
//   * transform: av2enc::cuda::make_tx_params / forward_transform /
//                inverse_transform / downsample_residual_64x64
//   * intra:     av2enc::cuda::dc_pred_value / dc_predict_residual /
//                add_residual_clip_to_plane
//
// This file only orchestrates them for one M0 coding block (raster order) and
// carries one small adapter for the 64x64 luma case: the transform component's
// reconstruct_64x64() needs a host-side predictor scalar, while the pipeline
// keeps predictions device-resident, so recon_64x64_adjusted_kernel() applies
// the component's 32x32 inverse residual with the normative 2x2 duplication.
// That adapter is 20 lines of duplication/clip, not a transform reimplementation.
//
// No transform or prediction arithmetic is duplicated here.  The 64x32/32x64
// coding blocks that only the diagnostic partition modes can emit are rejected
// explicitly because the current transform component covers adjusted sizes up
// to 32x32 per axis; the M0 target tree (k32x32) never emits them.
//
// The host pipeline synchronises once per frame, at the end of run_gpu_job().

#include "cuda/basic_pipeline.h"

#include "cuda/arena.h"
#include "cuda/context.h"
#include "cuda/intra.h"
#include "cuda/transform.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

namespace av2enc::cuda::detail {
namespace {

__device__ inline int clip3_dev(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

__device__ inline int clip1_dev(int value) { return clip3_dev(0, 255, value); }

// §4.8 Round2(x, n) = (x + (1 << (n - 1))) >> n, arithmetic shift.
bool check_cuda(cudaError_t status, const char* what, std::string& error) {
  if (status == cudaSuccess) {
    return true;
  }
  error = std::string(what) + " failed: " + cudaGetErrorString(status);
  return false;
}

bool check_status(Status status, const char* what, std::string& error) {
  if (is_ok(status)) {
    return true;
  }
  error = std::string(what) + " failed: " + status_name(status);
  return false;
}

// Integration glue for the transform component's 64x64 luma path: the
// component provides the adjusted 32x32 inverse residual, and this kernel
// applies the normative 2x2 duplication plus the clip.  It is deliberately a
// tiny adapter, not a reimplementation of the transform.
__global__ void recon_64x64_adjusted_kernel(
    const std::int16_t* __restrict__ residual,
    const std::uint8_t* __restrict__ pred_block,
    std::uint8_t* __restrict__ dst, int dst_stride, int x0, int y0, int max_x,
    int max_y) {
  const int pred = pred_block[0];
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= 32 * 32) {
    return;
  }
  const int j = idx / 32;
  const int i = idx - j * 32;
  const int value = clip1_dev(pred + residual[idx]);
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      const int x = x0 + 2 * i + dx;
      const int y = y0 + 2 * j + dy;
      if (x <= max_x && y <= max_y) {
        dst[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_stride) +
            static_cast<std::size_t>(x)] = static_cast<std::uint8_t>(value);
      }
    }
  }
}

struct ComponentScratch {
  std::int16_t* residual64 = nullptr;    // 64x64 full residual (64x64 luma only)
  std::int16_t* residual32 = nullptr;    // adjusted 32x32 / normal block
  double* coeff = nullptr;               // forward DCT scratch
  std::int32_t* dequant = nullptr;       // inverse DCT scratch
  std::int16_t* inv_residual = nullptr;  // inverse transform output
  std::uint8_t* pred_block = nullptr;    // 64x64 (first 1024 used otherwise)
  // Paired chroma scratch: U at [0, pels), V at [pels, 2*pels).  Both planes
  // have the same TxParams, so forward/inverse_transform run with nblocks=2.
  std::int16_t* residual_uv = nullptr;
  double* coeff_uv = nullptr;
  std::int32_t* dequant_uv = nullptr;
  std::int16_t* inv_uv = nullptr;
  std::uint8_t* pred_uv = nullptr;
};

// Runs one plane of one coding block through the shared transform/intra
// components.  All arithmetic is delegated; this function only derives the
// DcBlock/TxParams descriptors and the 64x64 adjustment glue.
bool component_plane(bool is_luma, const std::uint8_t* src, int src_stride,
                     int src_w, int src_h, std::uint8_t* recon,
                     int recon_stride, int x0, int y0, int width, int height,
                     int have_left, int have_above, std::int32_t* levels,
                     std::int32_t* pred_out, ComponentScratch& scratch,
                     cudaStream_t stream, int& launches, std::string& error) {
  const std::optional<TxParams> params =
      make_tx_params(width, height, is_luma);
  if (!params.has_value()) {
    error = "transform component does not support " +
            std::to_string(width) + "x" + std::to_string(height) +
            (is_luma ? " luma" : " chroma");
    return false;
  }

  DcBlock block{};
  block.recon_plane = recon;
  block.recon_stride = recon_stride;
  block.x0 = x0;
  block.y0 = y0;
  block.block_w = width;
  block.block_h = height;
  block.plane_max_x = src_w - 1;
  block.plane_max_y = src_h - 1;
  block.have_left = have_left;
  block.have_above = have_above;
  block.source_plane = src;
  block.source_stride = src_stride;
  block.source_max_x = src_w - 1;
  block.source_max_y = src_h - 1;

  if (!check_cuda(dc_pred_value(block, reinterpret_cast<int*>(pred_out), stream),
                  "intra::dc_pred_value", error)) {
    return false;
  }
  ++launches;

  const bool adjust_64x64 = is_luma && width == 64 && height == 64;
  if (adjust_64x64) {
    if (!check_cuda(dc_predict_residual(block, scratch.pred_block,
                                        scratch.residual64, stream),
                    "intra::dc_predict_residual(64x64)", error)) {
      return false;
    }
    ++launches;
    if (!check_cuda(downsample_residual_64x64(scratch.residual64,
                                              scratch.residual32, stream),
                    "transform::downsample_residual_64x64", error)) {
      return false;
    }
    ++launches;
    if (!check_cuda(forward_transform(scratch.residual32, scratch.coeff,
                                      levels, *params, 1, stream),
                    "transform::forward_transform(64x64)", error)) {
      return false;
    }
    ++launches;
    if (!check_cuda(inverse_transform(levels, scratch.dequant,
                                      scratch.inv_residual, *params, 1,
                                      stream),
                    "transform::inverse_transform(64x64)", error)) {
      return false;
    }
    ++launches;
    recon_64x64_adjusted_kernel<<<1, 1024, 0, stream>>>(
        scratch.inv_residual, scratch.pred_block, recon, recon_stride, x0, y0,
        src_w - 1, src_h - 1);
    ++launches;
    return check_cuda(cudaGetLastError(), "recon_64x64_adjusted_kernel", error);
  }

  if (!check_cuda(dc_predict_residual(block, scratch.pred_block,
                                      scratch.residual32, stream),
                  "intra::dc_predict_residual", error)) {
    return false;
  }
  ++launches;
  if (!check_cuda(forward_transform(scratch.residual32, scratch.coeff, levels,
                                    *params, 1, stream),
                  "transform::forward_transform", error)) {
    return false;
  }
  ++launches;
  if (!check_cuda(inverse_transform(levels, scratch.dequant,
                                    scratch.inv_residual, *params, 1, stream),
                  "transform::inverse_transform", error)) {
    return false;
  }
  ++launches;
  const cudaError_t recon_status =
      add_residual_clip_to_plane(scratch.pred_block, scratch.inv_residual,
                                 recon, recon_stride, x0, y0, width, height,
                                 src_w - 1, src_h - 1, stream);
  ++launches;
  return check_cuda(recon_status, "intra::add_residual_clip_to_plane", error);
}

// Chroma U/V pair for one coding block.  Prediction/residual stay per plane
// (the raster has neighbour dependencies), but the two planes share TxParams
// and their level buffers are adjacent, so the transform stages are batched
// through the component's nblocks interface.
bool component_chroma_pair(const GpuJob& job, const BlockDesc& b, int index,
                           int x0, int y0, int have_left, int have_above,
                           std::int32_t* u_levels, std::int32_t* v_levels,
                           std::int32_t* preds, ComponentScratch& scratch,
                           cudaStream_t stream, int& launches,
                           std::string& error) {
  (void)index;
  const std::optional<TxParams> params =
      make_tx_params(b.chroma_w, b.chroma_h, false);
  if (!params.has_value()) {
    error = "transform component does not support chroma " +
            std::to_string(b.chroma_w) + "x" + std::to_string(b.chroma_h);
    return false;
  }
  const int pels = b.chroma_pels;

  DcBlock u{};
  u.recon_plane = job.recon_u;
  u.recon_stride = job.chroma_padded_width;
  u.x0 = x0;
  u.y0 = y0;
  u.block_w = b.chroma_w;
  u.block_h = b.chroma_h;
  u.plane_max_x = job.chroma_width - 1;
  u.plane_max_y = job.chroma_height - 1;
  u.have_left = have_left;
  u.have_above = have_above;
  u.source_plane = job.u;
  u.source_stride = job.chroma_width;
  u.source_max_x = job.chroma_width - 1;
  u.source_max_y = job.chroma_height - 1;

  DcBlock v = u;
  v.recon_plane = job.recon_v;
  v.source_plane = job.v;

  if (!check_cuda(dc_pred_value(u, reinterpret_cast<int*>(preds + 1), stream),
                  "intra::dc_pred_value(U)", error) ||
      !check_cuda(dc_pred_value(v, reinterpret_cast<int*>(preds + 2), stream),
                  "intra::dc_pred_value(V)", error)) {
    return false;
  }
  launches += 2;
  if (!check_cuda(dc_predict_residual(u, scratch.pred_uv,
                                      scratch.residual_uv, stream),
                  "intra::dc_predict_residual(U)", error) ||
      !check_cuda(dc_predict_residual(v, scratch.pred_uv + pels,
                                      scratch.residual_uv + pels, stream),
                  "intra::dc_predict_residual(V)", error)) {
    return false;
  }
  launches += 2;
  if (!check_cuda(forward_transform(scratch.residual_uv, scratch.coeff_uv,
                                    u_levels, *params, 2, stream),
                  "transform::forward_transform(U/V pair)", error)) {
    return false;
  }
  ++launches;
  if (!check_cuda(inverse_transform(u_levels, scratch.dequant_uv,
                                    scratch.inv_uv, *params, 2, stream),
                  "transform::inverse_transform(U/V pair)", error)) {
    return false;
  }
  ++launches;
  if (!check_cuda(add_residual_clip_to_plane(
                      scratch.pred_uv, scratch.inv_uv, job.recon_u,
                      job.chroma_padded_width, x0, y0, b.chroma_w, b.chroma_h,
                      job.chroma_width - 1, job.chroma_height - 1, stream),
                  "intra::add_residual_clip_to_plane(U)", error) ||
      !check_cuda(add_residual_clip_to_plane(
                      scratch.pred_uv + pels, scratch.inv_uv + pels,
                      job.recon_v, job.chroma_padded_width, x0, y0, b.chroma_w,
                      b.chroma_h, job.chroma_width - 1, job.chroma_height - 1,
                      stream),
                  "intra::add_residual_clip_to_plane(V)", error)) {
    return false;
  }
  launches += 2;
  (void)v_levels;
  return true;
}

bool component_block(const GpuJob& job, const BlockDesc& b, int index,
                     ComponentScratch& scratch, cudaStream_t stream,
                     int& launches, std::string& error) {
  const int x0 = b.mi_col * 4;
  const int y0 = b.mi_row * 4;
  const int have_left = b.mi_col > 0 ? 1 : 0;
  const int have_above = b.mi_row > 0 ? 1 : 0;
  std::int32_t* preds = job.preds + static_cast<std::size_t>(index) * 3;
  std::int32_t* luma_levels = job.coeffs + b.coeff_offset;
  std::int32_t* u_levels = luma_levels + b.luma_pels;
  std::int32_t* v_levels = u_levels + b.chroma_pels;

  if (!component_plane(true, job.y, job.width, job.width, job.height,
                       job.recon_y, job.padded_width, x0, y0, b.luma_w, b.luma_h,
                       have_left, have_above, luma_levels, preds + 0, scratch,
                       stream, launches, error)) {
    return false;
  }
  return component_chroma_pair(job, b, index, x0 / 2, y0 / 2, have_left,
                               have_above, u_levels, v_levels, preds, scratch,
                               stream, launches, error);
}

bool component_block_supported(const BlockDesc& b) {
  return make_tx_params(b.luma_w, b.luma_h, true).has_value() &&
         make_tx_params(b.chroma_w, b.chroma_h, false).has_value();
}

// ---------------------------------------------------------------------------
// Persistent runtime state
// ---------------------------------------------------------------------------

struct PipelineRuntime {
  Context context;
  Arena arena;
  StagingBuffer upload_staging;
  StagingBuffer download_staging;
  int device = -1;
  bool context_ready = false;
};

PipelineRuntime& runtime() {
  // Intentionally leaked: CUDA teardown during static destruction is not worth
  // the ordering risk, and the process owns exactly one device context.
  static PipelineRuntime* instance = new PipelineRuntime();
  return *instance;
}

void set_vram_cap_bytes(std::uint64_t bytes) noexcept {
  Arena::set_capacity_cap_bytes(bytes);
}

std::uint64_t vram_cap_bytes() noexcept {
  return Arena::capacity_cap_bytes();
}

bool ensure_runtime(int device, std::string& error) {
  PipelineRuntime& rt = runtime();
  if (rt.context_ready) {
    if (rt.device != device) {
      error = "pipeline runtime already bound to device " +
              std::to_string(rt.device);
      return false;
    }
    return true;
  }
  std::string detail;
  const Status context_status = rt.context.init(device, &detail);
  if (!is_ok(context_status)) {
    error = std::string("Context::init failed: ") +
            status_name(context_status) +
            (detail.empty() ? "" : (": " + detail));
    return false;
  }
  rt.device = device;
  rt.context_ready = true;
  detail.clear();

  // No fixed VRAM preset. Take the largest reservation the device can serve,
  // keeping a small headroom for the context/driver, and honor an explicit
  // --vram-cap when one was set. The only real limit is cudaMalloc.
  std::uint64_t capacity =
      std::min(rt.context.total_vram_bytes(), rt.context.free_vram_bytes());
  capacity = capacity > Arena::kDeviceHeadroomBytes
                 ? capacity - Arena::kDeviceHeadroomBytes
                 : 0;
  const std::uint64_t user_cap = vram_cap_bytes();
  if (user_cap != 0 && user_cap < capacity) {
    capacity = user_cap;
  }
  if (capacity < Arena::kMiB) {
    capacity = Arena::kMiB;
  }

  Status arena_status = Status::kOk;
  for (int attempt = 0; attempt < 5; ++attempt) {
    arena_status = rt.arena.init(capacity, Arena::kAlignment, &detail);
    if (is_ok(arena_status) || capacity <= 256ull * Arena::kMiB) {
      break;
    }
    // Something else took VRAM between the query and the reservation: back off
    // and try again rather than failing the encode outright.
    capacity /= 2;
    detail.clear();
  }
  if (!is_ok(arena_status)) {
    error = std::string("Arena::init failed: ") + status_name(arena_status) +
            (detail.empty() ? "" : (": " + detail));
    return false;
  }
  return true;
}

bool ensure_staging(StagingBuffer& staging, std::size_t bytes, const char* what,
                    std::string& error) {
  if (bytes == 0) {
    return true;
  }
  if (staging.initialized() && staging.capacity() >= bytes) {
    return true;
  }
  staging.release();
  std::string detail;
  const Status status = staging.init(bytes, &detail);
  if (!is_ok(status)) {
    error = std::string(what) + " StagingBuffer::init failed: " +
            status_name(status) + (detail.empty() ? "" : (": " + detail));
    return false;
  }
  return true;
}

template <typename T>
bool arena_alloc(Arena& arena, std::size_t count, T** out, const char* what,
                 std::string& error) {
  const std::size_t bytes = count == 0 ? sizeof(T) : count * sizeof(T);
  void* pointer = nullptr;
  const Status status = arena.allocate_checked(bytes, &pointer);
  if (!is_ok(status)) {
    error = std::string("Arena::allocate_checked(") + what + ", " +
            std::to_string(bytes) + " bytes) failed: " + status_name(status);
    return false;
  }
  *out = static_cast<T*>(pointer);
  return true;
}

}  // namespace

bool run_gpu_job(const GpuJob& job, int device, std::string& error,
                 GpuTimings* timings) {
  error.clear();
  if (job.block_count < 0 || (job.block_count > 0 && job.blocks == nullptr)) {
    error = "invalid GPU job";
    return false;
  }
  if (!ensure_runtime(device, error)) {
    return false;
  }
  PipelineRuntime& rt = runtime();
  cudaStream_t stream = rt.context.stream();

  std::size_t total_coeffs = 0;
  for (int i = 0; i < job.block_count; ++i) {
    const BlockDesc& b = job.blocks[i];
    total_coeffs += static_cast<std::size_t>(b.luma_pels) +
                    2u * static_cast<std::size_t>(b.chroma_pels);
  }

  const std::size_t src_y = static_cast<std::size_t>(job.width) * job.height;
  const std::size_t src_c =
      static_cast<std::size_t>(job.chroma_width) * job.chroma_height;
  const std::size_t rec_y =
      static_cast<std::size_t>(job.padded_width) * job.padded_height;
  const std::size_t rec_c = static_cast<std::size_t>(job.chroma_padded_width) *
                            job.chroma_padded_height;
  const std::size_t block_bytes = static_cast<std::size_t>(job.block_count) *
                                  sizeof(BlockDesc);
  const std::size_t coeff_bytes = total_coeffs * sizeof(std::int32_t);
  const std::size_t pred_bytes = static_cast<std::size_t>(job.block_count) * 3 *
                                 sizeof(std::int32_t);

  // Per-frame bump allocation from the persistent arena.  Everything below is
  // dead after the single synchronisation at the end of this function.
  rt.arena.reset();
  std::uint8_t* d_y = nullptr;
  std::uint8_t* d_u = nullptr;
  std::uint8_t* d_v = nullptr;
  std::uint8_t* d_recon_y = nullptr;
  std::uint8_t* d_recon_u = nullptr;
  std::uint8_t* d_recon_v = nullptr;
  std::int32_t* d_coeffs = nullptr;
  std::int32_t* d_preds = nullptr;
  BlockDesc* d_blocks = nullptr;
  ComponentScratch scratch;
  if (!arena_alloc(rt.arena, src_y, &d_y, "source Y", error) ||
      !arena_alloc(rt.arena, src_c, &d_u, "source U", error) ||
      !arena_alloc(rt.arena, src_c, &d_v, "source V", error) ||
      !arena_alloc(rt.arena, rec_y, &d_recon_y, "recon Y", error) ||
      !arena_alloc(rt.arena, rec_c, &d_recon_u, "recon U", error) ||
      !arena_alloc(rt.arena, rec_c, &d_recon_v, "recon V", error) ||
      !arena_alloc(rt.arena, total_coeffs, &d_coeffs, "coefficients", error) ||
      !arena_alloc(rt.arena,
                   static_cast<std::size_t>(job.block_count) * 3, &d_preds,
                   "predictions", error) ||
      !arena_alloc(rt.arena, static_cast<std::size_t>(job.block_count),
                   &d_blocks, "block descriptors", error) ||
      !arena_alloc(rt.arena, 64 * 64, &scratch.residual64, "residual64",
                   error) ||
      !arena_alloc(rt.arena, 32 * 32, &scratch.residual32, "residual32",
                   error) ||
      !arena_alloc(rt.arena, 32 * 32, &scratch.coeff, "DCT scratch", error) ||
      !arena_alloc(rt.arena, 32 * 32, &scratch.dequant, "dequant scratch",
                   error) ||
      !arena_alloc(rt.arena, 32 * 32, &scratch.inv_residual,
                   "inverse residual", error) ||
      !arena_alloc(rt.arena, 64 * 64, &scratch.pred_block, "prediction block",
                   error) ||
      !arena_alloc(rt.arena, 2 * 32 * 32, &scratch.residual_uv,
                   "chroma residual pair", error) ||
      !arena_alloc(rt.arena, 2 * 32 * 32, &scratch.coeff_uv,
                   "chroma DCT pair", error) ||
      !arena_alloc(rt.arena, 2 * 32 * 32, &scratch.dequant_uv,
                   "chroma dequant pair", error) ||
      !arena_alloc(rt.arena, 2 * 32 * 32, &scratch.inv_uv,
                   "chroma inverse residual pair", error) ||
      !arena_alloc(rt.arena, 2 * 32 * 32, &scratch.pred_uv,
                   "chroma prediction pair", error)) {
    return false;
  }

  const std::size_t upload_bytes = src_y + 2 * src_c + block_bytes;
  const std::size_t download_bytes =
      rec_y + 2 * rec_c + coeff_bytes + pred_bytes;
  if (!ensure_staging(rt.upload_staging, upload_bytes, "upload", error) ||
      !ensure_staging(rt.download_staging, download_bytes, "download", error)) {
    return false;
  }

  // Pack the upload staging buffer: source planes and the block descriptors.
  std::uint8_t* upload =
      static_cast<std::uint8_t*>(rt.upload_staging.data());
  std::size_t upload_offset = 0;
  const auto stage = [&](const void* source, std::size_t bytes) {
    std::uint8_t* destination = upload + upload_offset;
    std::memcpy(destination, source, bytes);
    upload_offset += bytes;
    return destination;
  };
  const std::uint8_t* stage_y = stage(job.y, src_y);
  const std::uint8_t* stage_u = stage(job.u, src_c);
  const std::uint8_t* stage_v = stage(job.v, src_c);
  const BlockDesc* stage_blocks =
      reinterpret_cast<const BlockDesc*>(stage(job.blocks, block_bytes));

  cudaEvent_t events[4] = {nullptr, nullptr, nullptr, nullptr};
  bool ok = true;
  for (cudaEvent_t& event : events) {
    ok = ok && check_cuda(cudaEventCreate(&event), "cudaEventCreate", error);
  }
  if (!ok) {
    for (cudaEvent_t event : events) {
      if (event != nullptr) {
        cudaEventDestroy(event);
      }
    }
    return false;
  }

  GpuJob dev = job;
  dev.y = d_y;
  dev.u = d_u;
  dev.v = d_v;
  dev.recon_y = d_recon_y;
  dev.recon_u = d_recon_u;
  dev.recon_v = d_recon_v;
  dev.coeffs = d_coeffs;
  dev.preds = d_preds;
  dev.blocks = d_blocks;

  const auto sync_and_fail = [&]() {
    (void)rt.context.synchronize();
    for (cudaEvent_t event : events) {
      cudaEventDestroy(event);
    }
    return false;
  };

  cudaEventRecord(events[0], stream);
  if (!check_status(upload_async(stream, d_y, stage_y, src_y), "upload Y",
                   error) ||
      !check_status(upload_async(stream, d_u, stage_u, src_c), "upload U",
                   error) ||
      !check_status(upload_async(stream, d_v, stage_v, src_c), "upload V",
                   error) ||
      !check_status(upload_async(stream, d_blocks, stage_blocks, block_bytes),
                   "upload block descriptors", error) ||
      !check_cuda(cudaMemsetAsync(d_recon_y, 128, rec_y, stream),
                  "memset recon Y", error) ||
      !check_cuda(cudaMemsetAsync(d_recon_u, 128, rec_c, stream),
                  "memset recon U", error) ||
      !check_cuda(cudaMemsetAsync(d_recon_v, 128, rec_c, stream),
                  "memset recon V", error)) {
    return sync_and_fail();
  }
  cudaEventRecord(events[1], stream);

  int launches = 0;
  for (int i = 0; i < job.block_count; ++i) {
    const BlockDesc& b = job.blocks[i];
    if (b.real_coeffs == 0) {
      continue;
    }
    if (!component_block_supported(b)) {
      error = "shared transform component does not support coding block " +
              std::to_string(b.luma_w) + "x" + std::to_string(b.luma_h) +
              " (the M0 target k32x32 tree emits 32x32 and 64x64 blocks; "
              "64x32/32x64 appear only in diagnostic partition modes)";
      return sync_and_fail();
    }
    if (!component_block(dev, b, i, scratch, stream, launches, error)) {
      return sync_and_fail();
    }
  }
  cudaEventRecord(events[2], stream);

  // Device -> pinned staging downloads, one contiguous host block.
  std::uint8_t* download =
      static_cast<std::uint8_t*>(rt.download_staging.data());
  std::size_t download_offset = 0;
  const auto enqueue_download = [&](const void* device_source,
                                    std::size_t bytes) {
    std::uint8_t* destination = download + download_offset;
    download_offset += bytes;
    return download_async(stream, destination, device_source, bytes);
  };
  if (!check_status(enqueue_download(d_recon_y, rec_y), "download recon Y",
                   error) ||
      !check_status(enqueue_download(d_recon_u, rec_c), "download recon U",
                   error) ||
      !check_status(enqueue_download(d_recon_v, rec_c), "download recon V",
                   error) ||
      (coeff_bytes != 0 &&
       !check_status(enqueue_download(d_coeffs, coeff_bytes),
                    "download coefficients", error)) ||
      (pred_bytes != 0 &&
       !check_status(enqueue_download(d_preds, pred_bytes),
                    "download predictions", error))) {
    return sync_and_fail();
  }
  cudaEventRecord(events[3], stream);

  // Exactly one frame-boundary synchronisation.
  const Status sync_status = rt.context.synchronize();
  if (!is_ok(sync_status)) {
    error = std::string("Context::synchronize failed: ") +
            status_name(sync_status);
    for (cudaEvent_t event : events) {
      cudaEventDestroy(event);
    }
    return false;
  }

  float upload_ms = 0.0f;
  float kernel_ms = 0.0f;
  float download_ms = 0.0f;
  cudaEventElapsedTime(&upload_ms, events[0], events[1]);
  cudaEventElapsedTime(&kernel_ms, events[1], events[2]);
  cudaEventElapsedTime(&download_ms, events[2], events[3]);
  for (cudaEvent_t event : events) {
    cudaEventDestroy(event);
  }

  // Pinned staging -> caller buffers.
  std::size_t copy_offset = 0;
  const auto copy_out = [&](void* destination, std::size_t bytes) {
    std::memcpy(destination, download + copy_offset, bytes);
    copy_offset += bytes;
  };
  copy_out(job.recon_y, rec_y);
  copy_out(job.recon_u, rec_c);
  copy_out(job.recon_v, rec_c);
  if (coeff_bytes != 0) {
    copy_out(job.coeffs, coeff_bytes);
  }
  if (pred_bytes != 0) {
    copy_out(job.preds, pred_bytes);
  }

  if (timings != nullptr) {
    timings->upload_ms = static_cast<double>(upload_ms);
    timings->kernel_ms = static_cast<double>(kernel_ms);
    timings->download_ms = static_cast<double>(download_ms);
    timings->launches = launches;
  }
  return true;
}

}  // namespace av2enc::cuda::detail
