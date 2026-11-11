// GPU block stage for M10.  See m10_gpu.h.
#include "cuda/format/m10_gpu.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "cuda/m1/hbd/hbd_tx.h"
#include "cuda/m1/hbd/hbd_cpu.h"
#include "encoder/m10/m10_headers.h"
#include "encoder/tx_variants/tx_variants.h"

namespace av2enc::cuda::m10 {
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

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  bool allocate(std::size_t count) {
    reset();
    if (count == 0) return true;
    if (cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)) !=
        cudaSuccess) {
      ptr_ = nullptr;
      return false;
    }
    count_ = count;
    return true;
  }
  void reset() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      count_ = 0;
    }
  }
  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  std::size_t count() const { return count_; }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

bool check(cudaError_t status, const char* what, std::string* error) {
  if (status == cudaSuccess) return true;
  if (error != nullptr) {
    *error = std::string(what) + ": " + cudaGetErrorString(status);
  }
  return false;
}

int padded_extent(int coded) { return (coded + 63) & ~63; }

}  // namespace

bool m10_gpu_available() { return true; }

bool encode_m10_gpu(const av2enc::fmt::Picture& picture,
                    const av2enc::m10::M10Options& options,
                    av2enc::m10::M10Result& result, std::string* error) {
  if (error != nullptr) error->clear();
  result = av2enc::m10::M10Result{};
  if (picture.format.depth != av2enc::fmt::BitDepth::k10 ||
      picture.format.chroma != av2enc::fmt::ChromaFormat::k420) {
    if (error != nullptr) *error = "M10 GPU requires a 10-bit 4:2:0 picture";
    return false;
  }
  if (!av2enc::m10::m10_geometry_supported(picture.format.width,
                                           picture.format.height, error)) {
    return false;
  }
  if (!av2enc::fmt::samples_in_range(picture)) {
    if (error != nullptr) *error = "M10 input samples must be in [0,1023]";
    return false;
  }
  av2enc::m10::M10Config config;
  config.max_frame_width = static_cast<std::uint16_t>(picture.format.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.format.height);
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = options.disable_cdf_update;
  std::string reason;
  if (!av2enc::m10::validate_config(config, &reason)) {
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

  av2enc::m10::M10Plane host_recon[3];
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

  // Device buffers.  One allocation per encode; this is a standalone
  // correctness path, not the arena-backed product pipeline.
  DeviceBuffer<std::uint16_t> d_src[3];
  DeviceBuffer<std::uint16_t> d_recon[3];
  DeviceBuffer<std::uint16_t> d_pred;
  DeviceBuffer<std::int16_t> d_residual;
  DeviceBuffer<std::int32_t> d_levels;
  DeviceBuffer<std::int32_t> d_dequant;
  DeviceBuffer<std::int16_t> d_adjusted;
  DeviceBuffer<double> d_coeff;
  DeviceBuffer<std::int32_t> d_eob;
  DeviceBuffer<int> d_errors;
  DeviceBuffer<av2enc::cuda::hbd::HbdTxJob> d_job;

  const std::size_t max_coded = 64U * 64U;
  bool ok = true;
  for (int plane = 0; plane < 3 && ok; ++plane) {
    const std::size_t plane_count =
        static_cast<std::size_t>(coded[plane]) * coded_h[plane];
    ok = d_src[plane].allocate(plane_count) &&
         d_recon[plane].allocate(
             static_cast<std::size_t>(storage_w[plane]) * storage_h[plane]);
    if (ok) {
      ok = check(cudaMemcpy(d_src[plane].get(),
                            picture.planes[plane].samples.data(),
                            plane_count * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice),
                 "upload source plane", error);
    }
    if (ok) {
      ok = check(cudaMemset(d_recon[plane].get(), 0,
                            static_cast<std::size_t>(storage_w[plane]) *
                                storage_h[plane] * sizeof(std::uint16_t)),
                 "zero recon plane", error);
    }
  }
  ok = ok && d_pred.allocate(max_coded) && d_residual.allocate(max_coded) &&
       d_levels.allocate(max_coded) && d_dequant.allocate(max_coded) &&
       d_adjusted.allocate(max_coded) && d_coeff.allocate(max_coded) &&
       d_eob.allocate(1) && d_errors.allocate(1) && d_job.allocate(1);
  if (!ok) {
    if (error != nullptr && error->empty()) *error = "CUDA allocation failed";
    return false;
  }
  if (!check(cudaMemset(d_errors.get(), 0, sizeof(int)), "zero error count",
             error)) {
    return false;
  }

  const std::vector<av2enc::m10::M10BlockDesc> plan =
      av2enc::m10::make_block_plan(picture.format.width,
                                   picture.format.height);
  std::vector<av2enc::m10::M10Block> blocks;
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

    av2enc::m10::M10Block block;
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
      ok = check(cudaMemcpy(d_job.get(), &job, sizeof(job),
                            cudaMemcpyHostToDevice),
                 "upload job", error);
      if (ok) {
        dc_predict_kernel<<<1, 256>>>(
            d_recon[plane].get(), storage_w[plane], coded[plane],
            coded_h[plane], px0, py0, plane_coded_w[plane],
            plane_coded_h[plane], px0 > 0 ? 1 : 0, py0 > 0 ? 1 : 0,
            d_pred.get());
        ok = check(cudaGetLastError(), "dc_predict_kernel", error);
      }
      if (ok) {
        av2enc::cuda::hbd::HbdTxBuffers buffers;
        buffers.src = d_src[plane].get();
        buffers.pred = d_pred.get();
        buffers.residual_adj = d_adjusted.get();
        buffers.coeff = d_coeff.get();
        buffers.levels = d_levels.get();
        buffers.dequant = d_dequant.get();
        buffers.residual_out = d_residual.get();
        buffers.eob = d_eob.get();
        buffers.error_count = d_errors.get();
        ok = check(av2enc::cuda::hbd::hbd_tx_batch(d_job.get(), 1, buffers, 10,
                                                   static_cast<int>(max_coded),
                                                  nullptr),
                   "hbd_tx_batch", error);
      }
      if (ok) {
        reconstruct_kernel<<<1, 256>>>(
            d_recon[plane].get(), storage_w[plane], coded[plane],
            coded_h[plane], px0, py0, plane_coded_w[plane],
            plane_coded_h[plane], d_pred.get(), d_residual.get());
        ok = check(cudaGetLastError(), "reconstruct_kernel", error);
      }
      std::int32_t eob = 0;
      if (ok) {
        ok = check(cudaMemcpy(&eob, d_eob.get(), sizeof(eob),
                              cudaMemcpyDeviceToHost),
                   "download eob", error);
      }
      if (ok) {
        ok = check(cudaMemcpy(host_levels.data(), d_levels.get(),
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
  if (!check(cudaMemcpy(&errors, d_errors.get(), sizeof(errors),
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
                          d_recon[plane].get(),
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
  if (!av2enc::m10::assemble_m10_stream(
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

}  // namespace av2enc::cuda::m10
