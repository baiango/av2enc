// GPU block stage for M10.  See high_bitdepth_gpu.h.
#include "cuda/formats/high_bitdepth_gpu.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/pipeline/still/hbd/hbd_tx.h"
#include "cuda/pipeline/still/hbd/hbd_cpu.h"
#include "pipeline/still/high_bitdepth/high_bitdepth_headers.h"
#include "codec/transform/variants/tx_variants.h"

namespace av2enc::cuda::high_bitdepth {
namespace {

__device__ __forceinline__ int clip3_dev(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}
__device__ __forceinline__ int round2_dev(int value, int n) {
  return n <= 0 ? value : ((value + (1 << (n - 1))) >> n);
}
__device__ __forceinline__ int floor_log2_dev(int value) {
  int r = 0;
  while (value > 1) {
    value >>= 1;
    ++r;
  }
  return r;
}

// §7.13.2.10 DC prediction.  One thread computes the sums in the same integer
// order as the CPU oracle; the rest of the block is a broadcast fill.
__global__ void dc_predict_kernel(const std::uint16_t* __restrict__ recon,
                                  int stride, int coded_w, int coded_h, int x0,
                                  int y0, int w, int h, int have_left,
                                  int have_above, std::uint16_t* __restrict__ pred) {
  __shared__ int s_value;
  if (threadIdx.x == 0) {
    const int max_x = coded_w - 1;
    const int max_y = coded_h - 1;
    int sum_left = 0;
    int sum_above = 0;
    if (have_left != 0) {
      for (int i = 0; i < h; ++i) {
        const int y = clip3_dev(0, max_y, y0 + i);
        sum_left += static_cast<int>(
            recon[static_cast<std::size_t>(y) * stride + (x0 - 1)]);
      }
    }
    if (have_above != 0) {
      for (int j = 0; j < w; ++j) {
        const int x = clip3_dev(0, max_x, x0 + j);
        sum_above += static_cast<int>(
            recon[static_cast<std::size_t>(y0 - 1) * stride + x]);
      }
    }
    int value = 1 << 9;  // DC bias at 10-bit
    if (have_left != 0 && have_above != 0) {
      const int denom = w + h;
      const int shift = 9 + floor_log2_dev(denom);
      value = round2_dev((sum_left + sum_above) * 512, shift);
    } else if (have_left != 0) {
      value = round2_dev(sum_left, floor_log2_dev(h));
    } else if (have_above != 0) {
      value = round2_dev(sum_above, floor_log2_dev(w));
    }
    s_value = clip3_dev(0, 1023, value);
  }
  __syncthreads();
  const int count = w * h;
  for (int i = static_cast<int>(threadIdx.x); i < count;
       i += static_cast<int>(blockDim.x)) {
    pred[i] = static_cast<std::uint16_t>(s_value);
  }
}

// recon = clip1(pred + residual, 10), stored only inside the coded plane.
__global__ void reconstruct_kernel(std::uint16_t* __restrict__ recon, int stride,
                                   int coded_w, int coded_h, int x0, int y0,
                                   int w, int h,
                                   const std::uint16_t* __restrict__ pred,
                                   const std::int16_t* __restrict__ residual) {
  const int count = w * h;
  for (int i = static_cast<int>(threadIdx.x); i < count;
       i += static_cast<int>(blockDim.x)) {
    const int y = i / w;
    const int x = i - y * w;
    const int px = x0 + x;
    const int py = y0 + y;
    if (px >= coded_w || py >= coded_h) continue;
    const int value = static_cast<int>(pred[i]) + static_cast<int>(residual[i]);
    recon[static_cast<std::size_t>(py) * stride + px] =
        static_cast<std::uint16_t>(clip3_dev(0, 1023, value));
  }
}

std::uint64_t align_up_bytes(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1u) & ~(alignment - 1u);
}

// Rewinds the arena on every return path: the M10 stage treats all of its
// device buffers as transient scratch for one picture.
class ArenaMark {
 public:
  explicit ArenaMark(Arena& arena) : arena_(arena), mark_(arena.mark()) {}
  ~ArenaMark() { arena_.rewind(mark_); }
  ArenaMark(const ArenaMark&) = delete;
  ArenaMark& operator=(const ArenaMark&) = delete;

 private:
  Arena& arena_;
  std::uint64_t mark_;
};

template <typename T>
bool arena_array(Arena& arena, std::size_t count, T** out, const char* what,
                 std::uint64_t arena_need, HighBitDepthGpuStatus* status,
                 std::string* error) {
  *out = nullptr;
  if (count == 0) return true;
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    if (status != nullptr) status->code = HighBitDepthGpuStatus::Code::kInvalidArgument;
    if (error != nullptr) *error = std::string("M10 GPU: oversized ") + what;
    return false;
  }
  void* raw = nullptr;
  const Status result = arena.allocate_checked(count * sizeof(T), &raw);
  if (result == Status::kArenaExhausted) {
    if (status != nullptr) {
      status->code = HighBitDepthGpuStatus::Code::kArenaTooSmall;
      status->required_bytes = arena_need;
    }
    if (error != nullptr) {
      *error = std::string("M10 GPU arena too small for ") + what + ": need " +
               std::to_string(arena_need) + " device bytes";
    }
    return false;
  }
  if (!is_ok(result)) {
    if (status != nullptr) status->code = HighBitDepthGpuStatus::Code::kInvalidArgument;
    if (error != nullptr) {
      *error = std::string("M10 GPU arena allocation failed for ") + what +
               ": " + status_name(result);
    }
    return false;
  }
  *out = static_cast<T*>(raw);
  return true;
}

bool check(cudaError_t status, const char* what, std::string* error) {
  if (status == cudaSuccess) return true;
  if (error != nullptr) {
    *error = std::string(what) + ": " + cudaGetErrorString(status);
  }
  return false;
}

int padded_extent(int coded) { return (coded + 63) & ~63; }

}  // namespace

std::uint64_t required_device_bytes(const HighBitDepthGpuConfig& config) {
  constexpr std::uint64_t kMaxCoded = 64ull * 64ull;
  if (config.mi_cols == 0 || config.mi_rows == 0) {
    // Enough for one aligned allocation even after base-alignment padding, so
    // a caller can still reach the module's geometry validation error.
    return 2u * Arena::kAlignment - 1u;
  }
  const int mi_cols = static_cast<int>(config.mi_cols);
  const int mi_rows = static_cast<int>(config.mi_rows);
  const int coded[3] = {mi_cols * 4, mi_cols * 2, mi_cols * 2};
  const int coded_h[3] = {mi_rows * 4, mi_rows * 2, mi_rows * 2};
  const int storage_w[3] = {padded_extent(coded[0]), padded_extent(coded[0]) / 2,
                            padded_extent(coded[0]) / 2};
  const int storage_h[3] = {padded_extent(coded_h[0]),
                            padded_extent(coded_h[0]) / 2,
                            padded_extent(coded_h[0]) / 2};

  std::uint64_t bytes = 0;
  const auto add = [&bytes](std::uint64_t count, std::uint64_t element) {
    bytes += align_up_bytes(count * element, Arena::kAlignment);
  };
  for (int plane = 0; plane < 3; ++plane) {
    add(static_cast<std::uint64_t>(coded[plane]) *
            static_cast<std::uint64_t>(coded_h[plane]),
        sizeof(std::uint16_t));  // d_src
    add(static_cast<std::uint64_t>(storage_w[plane]) *
            static_cast<std::uint64_t>(storage_h[plane]),
        sizeof(std::uint16_t));  // d_recon
  }
  add(kMaxCoded, sizeof(std::uint16_t));  // d_pred
  add(kMaxCoded, sizeof(std::int16_t));   // d_residual
  add(kMaxCoded, sizeof(std::int32_t));   // d_levels
  add(kMaxCoded, sizeof(std::int32_t));   // d_dequant
  add(kMaxCoded, sizeof(std::int16_t));   // d_adjusted
  add(kMaxCoded, sizeof(double));         // d_coeff
  add(1, sizeof(std::int32_t));           // d_eob
  add(1, sizeof(int));                    // d_errors
  add(1, sizeof(av2enc::cuda::hbd::HbdTxJob));  // d_job
  // Arena::init() aligns the usable base inside the reservation; budget one
  // alignment unit so the sum above is fully usable.
  bytes += Arena::kAlignment - 1u;
  return bytes;
}

bool high_bitdepth_gpu_available() { return true; }

namespace {

bool encode_high_bitdepth_gpu_impl(Arena& arena, const av2enc::fmt::Picture& picture,
                         const av2enc::high_bitdepth::HighBitDepthOptions& options,
                         av2enc::high_bitdepth::HighBitDepthResult& result, std::string* error,
                         HighBitDepthGpuStatus* status) {
  if (error != nullptr) error->clear();
  result = av2enc::high_bitdepth::HighBitDepthResult{};
  if (picture.format.depth != av2enc::fmt::BitDepth::k10 ||
      picture.format.chroma != av2enc::fmt::ChromaFormat::k420) {
    if (error != nullptr) *error = "M10 GPU requires a 10-bit 4:2:0 picture";
    return false;
  }
  if (!av2enc::high_bitdepth::high_bitdepth_geometry_supported(picture.format.width,
                                           picture.format.height, error)) {
    return false;
  }
  if (!av2enc::fmt::samples_in_range(picture)) {
    if (error != nullptr) *error = "M10 input samples must be in [0,1023]";
    return false;
  }
  av2enc::high_bitdepth::HighBitDepthConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(picture.format.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.format.height);
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = options.disable_cdf_update;
  std::string reason;
  if (!av2enc::high_bitdepth::validate_config(config, &reason)) {
    if (error != nullptr) *error = reason;
    return false;
  }

  const int mi_cols = static_cast<int>(picture.geometry.mi_cols);
  const int mi_rows = static_cast<int>(picture.geometry.mi_rows);
  const int coded[3] = {mi_cols * 4, mi_cols * 2, mi_cols * 2};
  const int coded_h[3] = {mi_rows * 4, mi_rows * 2, mi_rows * 2};
  const int storage_w[3] = {padded_extent(coded[0]), padded_extent(coded[0]) / 2,
                            padded_extent(coded[0]) / 2};
  const int storage_h[3] = {padded_extent(coded_h[0]),
                            padded_extent(coded_h[0]) / 2,
                            padded_extent(coded_h[0]) / 2};

  const HighBitDepthGpuConfig arena_config{
      picture.format.width, picture.format.height,
      static_cast<std::uint32_t>(mi_cols), static_cast<std::uint32_t>(mi_rows)};
  const std::uint64_t arena_need = required_device_bytes(arena_config);
  if (status != nullptr) status->required_bytes = arena_need;

  // All M10 device buffers are transient scratch for this picture.  Reserve
  // them from the caller's arena and rewind on every exit path.
  ArenaMark arena_mark(arena);

  av2enc::high_bitdepth::HighBitDepthPlane host_recon[3];
  for (int plane = 0; plane < 3; ++plane) {
    host_recon[plane].width = storage_w[plane];
    host_recon[plane].height = storage_h[plane];
    host_recon[plane].coded_width = coded[plane];
    host_recon[plane].coded_height = coded_h[plane];
    host_recon[plane].visible_width =
        static_cast<int>(picture.geometry.planes[plane].visible_width);
    host_recon[plane].visible_height =
        static_cast<int>(picture.geometry.planes[plane].visible_height);
    host_recon[plane].samples.assign(
        static_cast<std::size_t>(storage_w[plane]) * storage_h[plane], 0);
  }

  // Device buffers, all from the caller's arena.  See required_device_bytes().
  std::uint16_t* d_src[3] = {nullptr, nullptr, nullptr};
  std::uint16_t* d_recon[3] = {nullptr, nullptr, nullptr};
  std::uint16_t* d_pred = nullptr;
  std::int16_t* d_residual = nullptr;
  std::int32_t* d_levels = nullptr;
  std::int32_t* d_dequant = nullptr;
  std::int16_t* d_adjusted = nullptr;
  double* d_coeff = nullptr;
  std::int32_t* d_eob = nullptr;
  int* d_errors = nullptr;
  av2enc::cuda::hbd::HbdTxJob* d_job = nullptr;

  const std::size_t max_coded = 64U * 64U;
  bool ok = true;
  for (int plane = 0; plane < 3 && ok; ++plane) {
    const std::size_t plane_count =
        static_cast<std::size_t>(coded[plane]) * coded_h[plane];
    ok = arena_array(arena, plane_count, &d_src[plane], "source plane",
                     arena_need, status, error) &&
         arena_array(arena,
                     static_cast<std::size_t>(storage_w[plane]) *
                         storage_h[plane],
                     &d_recon[plane], "recon plane", arena_need, status,
                     error);
    if (ok) {
      ok = check(cudaMemcpy(d_src[plane],
                            picture.planes[plane].samples.data(),
                            plane_count * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "upload source plane", error);
    }
    if (ok) {
      ok = check(cudaMemset(d_recon[plane], 0,
                            static_cast<std::size_t>(storage_w[plane]) *
                                storage_h[plane] * sizeof(std::uint16_t)),
                 "zero recon plane", error);
    }
  }
  ok = ok &&
       arena_array(arena, max_coded, &d_pred, "pred", arena_need, status,
                   error) &&
       arena_array(arena, max_coded, &d_residual, "residual", arena_need,
                   status, error) &&
       arena_array(arena, max_coded, &d_levels, "levels", arena_need, status,
                   error) &&
       arena_array(arena, max_coded, &d_dequant, "dequant", arena_need,
                   status, error) &&
       arena_array(arena, max_coded, &d_adjusted, "adjusted", arena_need,
                   status, error) &&
       arena_array(arena, max_coded, &d_coeff, "coeff", arena_need, status,
                   error) &&
       arena_array(arena, 1, &d_eob, "eob", arena_need, status, error) &&
       arena_array(arena, 1, &d_errors, "errors", arena_need, status,
                   error) &&
       arena_array(arena, 1, &d_job, "tx job", arena_need, status, error);
  if (!ok) {
    if (error != nullptr && error->empty()) *error = "CUDA allocation failed";
    return false;
  }
  if (!check(cudaMemset(d_errors, 0, sizeof(int)), "zero error count", error)) {
    return false;
  }

  const std::vector<av2enc::high_bitdepth::HighBitDepthBlockDesc> plan =
      av2enc::high_bitdepth::make_block_plan(picture.format.width,
                                   picture.format.height);
  std::vector<av2enc::high_bitdepth::HighBitDepthBlock> blocks;
  blocks.reserve(plan.size());
  std::vector<std::int32_t> levels;
  std::vector<std::int32_t> host_levels(max_coded);
  std::vector<std::uint16_t> scan(max_coded);

  for (const auto& desc : plan) {
    const int x0 = static_cast<int>(desc.mi_col) * 4;
    const int y0 = static_cast<int>(desc.mi_row) * 4;
    const int luma = desc.luma_size;
    const int chroma = luma / 2;
    const int plane_coded_w[3] = {luma, chroma, chroma};
    const int plane_coded_h[3] = {luma, chroma, chroma};

    av2enc::high_bitdepth::HighBitDepthBlock block;
    block.mi_row = desc.mi_row;
    block.mi_col = desc.mi_col;
    block.luma_size = desc.luma_size;
    block.levels_offset = static_cast<std::uint32_t>(levels.size());

    for (int plane = 0; plane < 3; ++plane) {
      const auto params_opt = av2enc::txv::make_tx_params(
          plane_coded_w[plane], plane_coded_h[plane],
          av2enc::txv::TxType::kDctDct);
      if (!params_opt.has_value()) {
        if (error != nullptr) *error = "unsupported M10 transform size";
        return false;
      }
      const av2enc::txv::TxParams& params = *params_opt;
      block.tx_size[plane] = static_cast<std::uint8_t>(params.tx_size);

      av2enc::cuda::hbd::HbdTxJob job{};
      job.params = params;
      job.input =
          static_cast<std::uint8_t>(av2enc::cuda::hbd::HbdTxInput::kSourcePred);
      job.src_packed = 0;
      job.src_x = static_cast<std::int16_t>(plane == 0 ? x0 : x0 / 2);
      job.src_y = static_cast<std::int16_t>(plane == 0 ? y0 : y0 / 2);
      job.src_width = static_cast<std::int16_t>(
          picture.geometry.planes[plane].visible_width);
      job.src_height = static_cast<std::int16_t>(
          picture.geometry.planes[plane].visible_height);
      job.src_offset = 0;
      job.src_stride = coded[plane];
      job.pred_offset = 0;
      job.residual_adj_offset = 0;
      job.coeff_offset = 0;
      job.levels_offset = 0;
      job.dequant_offset = 0;
      job.residual_out_offset = 0;
      job.eob_offset = 0;

      const int px0 = job.src_x;
      const int py0 = job.src_y;
      const std::size_t elems =
          static_cast<std::size_t>(plane_coded_w[plane]) * plane_coded_h[plane];
      ok = check(cudaMemcpy(d_job, &job, sizeof(job),
                            cudaMemcpyHostToDevice),
                 "upload job", error);
      if (ok) {
        dc_predict_kernel<<<1, 256>>>(
            d_recon[plane], storage_w[plane], coded[plane],
            coded_h[plane], px0, py0, plane_coded_w[plane],
            plane_coded_h[plane], px0 > 0 ? 1 : 0, py0 > 0 ? 1 : 0,
            d_pred);
        ok = check(cudaGetLastError(), "dc_predict_kernel", error);
      }
      if (ok) {
        av2enc::cuda::hbd::HbdTxBuffers buffers;
        buffers.src = d_src[plane];
        buffers.pred = d_pred;
        buffers.residual_adj = d_adjusted;
        buffers.coeff = d_coeff;
        buffers.levels = d_levels;
        buffers.dequant = d_dequant;
        buffers.residual_out = d_residual;
        buffers.eob = d_eob;
        buffers.error_count = d_errors;
        ok = check(av2enc::cuda::hbd::hbd_tx_batch(d_job, 1, buffers, 10,
                                                   static_cast<int>(max_coded),
                                                  nullptr),
                   "hbd_tx_batch", error);
      }
      if (ok) {
        reconstruct_kernel<<<1, 256>>>(
            d_recon[plane], storage_w[plane], coded[plane],
            coded_h[plane], px0, py0, plane_coded_w[plane],
            plane_coded_h[plane], d_pred, d_residual);
        ok = check(cudaGetLastError(), "reconstruct_kernel", error);
      }
      std::int32_t eob = 0;
      if (ok) {
        ok = check(cudaMemcpy(&eob, d_eob, sizeof(eob),
                              cudaMemcpyDeviceToHost),
                   "download eob", error);
      }
      if (ok) {
        ok = check(cudaMemcpy(host_levels.data(), d_levels,
                              params.w * params.h * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost),
                   "download levels", error);
      }
      if (!ok) return false;
      if (eob < 0 || eob > params.w * params.h) {
        if (error != nullptr) *error = "GPU rejected the transform job";
        return false;
      }
      block.eob[plane] = static_cast<std::uint32_t>(eob);
      av2enc::hbd::build_scan(params.w, params.h, scan.data());
      for (int c = 0; c < eob; ++c) {
        levels.push_back(host_levels[scan[static_cast<std::size_t>(c)]]);
      }
      (void)elems;
    }
    blocks.push_back(block);
  }

  int errors = 0;
  if (!check(cudaMemcpy(&errors, d_errors, sizeof(errors),
                        cudaMemcpyDeviceToHost),
             "download error count", error)) {
    return false;
  }
  if (errors != 0) {
    if (error != nullptr) *error = "GPU transform job was rejected";
    return false;
  }

  for (int plane = 0; plane < 3; ++plane) {
    if (!check(cudaMemcpy(host_recon[plane].samples.data(),
                          d_recon[plane],
                          static_cast<std::size_t>(storage_w[plane]) *
                              storage_h[plane] * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "download recon plane", error)) {
      return false;
    }
  }

  std::vector<std::uint8_t> frame;
  std::uint32_t tile_bytes = 0;
  std::string stream_error;
  if (!av2enc::high_bitdepth::assemble_high_bitdepth_stream(
          picture, options, blocks.data(), blocks.size(), levels.data(),
          levels.size(), &frame, &tile_bytes, &stream_error)) {
    if (error != nullptr) *error = stream_error;
    return false;
  }

  result.tile_bytes = tile_bytes;
  result.frame_bytes = static_cast<std::uint32_t>(frame.size());
  result.frame = std::move(frame);
  result.blocks = std::move(blocks);
  result.levels = std::move(levels);
  for (int plane = 0; plane < 3; ++plane) {
    result.recon[plane] = std::move(host_recon[plane]);
  }
  return true;
}

}  // namespace

HighBitDepthGpuStatus encode_high_bitdepth_gpu(Arena& arena, const av2enc::fmt::Picture& picture,
                            const av2enc::high_bitdepth::HighBitDepthOptions& options,
                            av2enc::high_bitdepth::HighBitDepthResult& result,
                            std::string* error) {
  HighBitDepthGpuStatus status;
  if (!encode_high_bitdepth_gpu_impl(arena, picture, options, result, error, &status)) {
    if (status.code == HighBitDepthGpuStatus::Code::kOk) {
      status.code = HighBitDepthGpuStatus::Code::kInvalidArgument;
    }
    return status;
  }
  status.code = HighBitDepthGpuStatus::Code::kOk;
  return status;
}

bool encode_high_bitdepth_gpu(const av2enc::fmt::Picture& picture,
                    const av2enc::high_bitdepth::HighBitDepthOptions& options,
                    av2enc::high_bitdepth::HighBitDepthResult& result, std::string* error) {
  if (error != nullptr) error->clear();
  HighBitDepthGpuConfig config;
  config.width = picture.format.width;
  config.height = picture.format.height;
  config.mi_cols = picture.geometry.mi_cols;
  config.mi_rows = picture.geometry.mi_rows;
  const std::uint64_t need = required_device_bytes(config);
  Arena arena;
  std::string arena_error;
  const Status init = arena.init(need, Arena::kAlignment, &arena_error);
  if (!is_ok(init)) {
    if (error != nullptr) {
      *error = arena_error.empty()
                   ? std::string("M10 GPU arena init failed: ") +
                         status_name(init)
                   : arena_error;
    }
    return false;
  }
  const HighBitDepthGpuStatus status =
      encode_high_bitdepth_gpu(arena, picture, options, result, error);
  if (status.ok()) return true;
  if (error != nullptr && error->empty()) {
    if (status.code == HighBitDepthGpuStatus::Code::kArenaTooSmall) {
      *error = "M10 GPU arena too small: need " +
               std::to_string(status.required_bytes) + " device bytes";
    } else {
      *error = "M10 GPU encode failed";
    }
  }
  return false;
}

}  // namespace av2enc::cuda::high_bitdepth
