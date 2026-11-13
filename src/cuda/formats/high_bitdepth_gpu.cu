// GPU block stage for M10.  See high_bitdepth_gpu.h.
#include "cuda/formats/high_bitdepth_gpu.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/formats/format_device.h"
#include "cuda/pipeline/still/hbd/hbd_tx.h"
#include "cuda/pipeline/still/hbd/hbd_cpu.h"
#include "formats/high_bitdepth/hbd_depth.h"
#include "formats/high_bitdepth/hbd_headers.h"
#include "pipeline/still/high_bitdepth/high_bitdepth_headers.h"
#include "pipeline/still/high_bitdepth/high_bitdepth_still.h"
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

// §7.13.3.22 Div_Lut.  The M10 DC sizes resolve to 512, but 4:2:2 chroma
// (16x32 -> w + h = 48) needs the full table, so use the normative form for
// every format.
__device__ const int kDivLutDev[129] = {
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465, 462,
    458, 455, 452, 449, 446, 443, 440, 437, 434, 431, 428, 426, 423, 420, 417,
    415, 412, 410, 407, 405, 402, 400, 397, 395, 392, 390, 388, 386, 383, 381,
    379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358, 356, 354, 352, 350,
    349, 347, 345, 343, 341, 340, 338, 336, 334, 333, 331, 329, 328, 326, 324,
    323, 321, 320, 318, 317, 315, 314, 312, 311, 309, 308, 306, 305, 303, 302,
    301, 299, 298, 297, 295, 294, 293, 291, 290, 289, 287, 286, 285, 284, 282,
    281, 280, 279, 278, 277, 275, 274, 273, 272, 271, 270, 269, 267, 266, 265,
    264, 263, 262, 261, 260, 259, 258, 257, 256};

// §7.13.2.11 approx_divide() with the normative Div_Lut.
__device__ __forceinline__ int approx_divide_dev(int numerator,
                                                 int denominator) {
  const int abs_denominator = denominator < 0 ? -denominator : denominator;
  if (abs_denominator == 0) return 0;
  const int n = floor_log2_dev(abs_denominator);
  const int e = abs_denominator - (1 << n);
  const int f = n > 7 ? round2_dev(e, n - 7) : e << (7 - n);
  const int scale = kDivLutDev[f < 128 ? f : 128];
  return round2_dev(numerator * (denominator < 0 ? -scale : scale), n + 9);
}

// §7.13.2.10 DC prediction.  One thread computes the sums in the same integer
// order as the CPU oracle; the rest of the block is a broadcast fill.
__global__ void dc_predict_kernel(const std::uint16_t* __restrict__ recon,
                                  int stride, int coded_w, int coded_h, int x0,
                                  int y0, int w, int h, int have_left,
                                  int have_above, int bit_depth,
                                  std::uint16_t* __restrict__ pred) {
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
    int value = 1 << (bit_depth - 1);  // DC neutral value, 512 at 10-bit
    if (have_left != 0 && have_above != 0) {
      value = approx_divide_dev(sum_left + sum_above, w + h);
    } else if (have_left != 0) {
      value = round2_dev(sum_left, floor_log2_dev(h));
    } else if (have_above != 0) {
      value = round2_dev(sum_above, floor_log2_dev(w));
    }
    s_value = clip3_dev(0, (1 << bit_depth) - 1, value);
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
                                   int w, int h, int bit_depth,
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
        static_cast<std::uint16_t>(
            clip3_dev(0, (1 << bit_depth) - 1, value));
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

}  // namespace

std::uint64_t required_device_bytes(const HighBitDepthGpuConfig& config) {
  constexpr std::uint64_t kMaxCoded = 64ull * 64ull;
  av2enc::fmt::Format format = config.format;
  if (format.width == 0 || format.height == 0) {
    format.width = config.width;
    format.height = config.height;
    format.chroma = av2enc::fmt::ChromaFormat::k420;
    format.depth = config.bit_depth == 8 ? av2enc::fmt::BitDepth::k8
                                         : av2enc::fmt::BitDepth::k10;
  }
  const fmtdev::FrameLayout layout = fmtdev::make_frame_layout(format);
  if (!layout.valid()) {
    // Enough for one aligned allocation even after base-alignment padding, so
    // a caller can still reach the module's geometry validation error.
    return 2u * Arena::kAlignment - 1u;
  }

  std::uint64_t bytes = 0;
  const auto add = [&bytes](std::uint64_t count, std::uint64_t element) {
    bytes += align_up_bytes(count * element, Arena::kAlignment);
  };
  // Each plane is its own arena allocation, so align them individually rather
  // than once for the combined frame.
  for (int plane = 0; plane < layout.num_planes; ++plane) {
    add(layout.planes[plane].coded_samples, sizeof(std::uint16_t));  // d_src
    add(layout.planes[plane].storage_samples,
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
  if (picture.format.depth != av2enc::fmt::BitDepth::k10) {
    if (error != nullptr) *error = "HBD GPU requires a 10-bit picture";
    return false;
  }
  if (!av2enc::high_bitdepth::high_bitdepth_geometry_supported(
          picture.format.width, picture.format.height, error)) {
    return false;
  }
  if (!av2enc::fmt::samples_in_range(picture)) {
    if (error != nullptr) *error = "HBD input samples must be in [0,1023]";
    return false;
  }
  const fmtdev::FrameLayout layout = fmtdev::make_frame_layout(picture.format);
  if (!layout.valid()) {
    if (error != nullptr) *error = "HBD GPU: invalid frame format";
    return false;
  }
  const int num_planes = layout.num_planes;
  const int bit_depth = 10;
  const int quant_step = av2enc::hbd::depth::qlookup(options.base_q_idx);
  if (quant_step <= 0) {
    if (error != nullptr) *error = "HBD GPU: unsupported base_q_idx";
    return false;
  }

  av2enc::hbd::HbdConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(picture.format.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.format.height);
  config.depth = av2enc::fmt::BitDepth::k10;
  config.chroma = picture.format.chroma;
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = options.disable_cdf_update;
  std::string reason;
  if (!av2enc::hbd::validate_config(config, &reason)) {
    if (error != nullptr) *error = reason;
    return false;
  }

  HighBitDepthGpuConfig arena_config;
  arena_config.width = picture.format.width;
  arena_config.height = picture.format.height;
  arena_config.mi_cols = picture.geometry.mi_cols;
  arena_config.mi_rows = picture.geometry.mi_rows;
  arena_config.format = picture.format;
  arena_config.bit_depth = bit_depth;
  const std::uint64_t arena_need = required_device_bytes(arena_config);
  if (status != nullptr) status->required_bytes = arena_need;

  // All device buffers are transient scratch for this picture.  Reserve them
  // from the caller's arena and rewind on every exit path.
  ArenaMark arena_mark(arena);

  av2enc::high_bitdepth::HighBitDepthPlane host_recon[3];
  for (int plane = 0; plane < num_planes; ++plane) {
    const fmtdev::PlaneLayout& p = layout.planes[plane];
    host_recon[plane].width = p.storage_w;
    host_recon[plane].height = p.storage_h;
    host_recon[plane].coded_width = p.coded_w;
    host_recon[plane].coded_height = p.coded_h;
    host_recon[plane].visible_width = p.visible_w;
    host_recon[plane].visible_height = p.visible_h;
    host_recon[plane].samples.assign(
        static_cast<std::size_t>(p.storage_w) * p.storage_h, 0);
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
  for (int plane = 0; plane < num_planes && ok; ++plane) {
    const fmtdev::PlaneLayout& p = layout.planes[plane];
    ok = arena_array(arena, p.coded_samples, &d_src[plane], "source plane",
                     arena_need, status, error) &&
         arena_array(arena, p.storage_samples, &d_recon[plane], "recon plane",
                     arena_need, status, error);
    if (ok) {
      ok = check(cudaMemcpy2D(d_src[plane],
                              static_cast<std::size_t>(p.coded_w) *
                                  sizeof(std::uint16_t),
                              picture.planes[plane].samples.data(),
                              static_cast<std::size_t>(
                                  picture.planes[plane].stride) *
                                  sizeof(std::uint16_t),
                              static_cast<std::size_t>(p.coded_w) *
                                  sizeof(std::uint16_t),
                              static_cast<std::size_t>(p.coded_h),
                              cudaMemcpyHostToDevice),
                 "upload source plane", error);
    }
    if (ok) {
      ok = check(cudaMemset(d_recon[plane], 0,
                            p.storage_samples * sizeof(std::uint16_t)),
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

    av2enc::high_bitdepth::HighBitDepthBlock block;
    block.mi_row = desc.mi_row;
    block.mi_col = desc.mi_col;
    block.luma_size = desc.luma_size;
    block.levels_offset = static_cast<std::uint32_t>(levels.size());
    block.tx_size[0] = block.tx_size[1] = block.tx_size[2] = 0;
    block.eob[0] = block.eob[1] = block.eob[2] = 0;

    for (int plane = 0; plane < num_planes; ++plane) {
      const fmtdev::PlaneLayout& p = layout.planes[plane];
      const int plane_coded_w = luma >> p.sub_x;
      const int plane_coded_h = luma >> p.sub_y;
      const auto params_opt = av2enc::txv::make_tx_params(
          plane_coded_w, plane_coded_h, av2enc::txv::TxType::kDctDct);
      if (!params_opt.has_value()) {
        if (error != nullptr) *error = "unsupported HBD transform size";
        return false;
      }
      av2enc::txv::TxParams params = *params_opt;
      block.tx_size[plane] = static_cast<std::uint8_t>(params.tx_size);

      // §7.14.2: the transform component bakes the QP-90 step (528) into
      // quant_factor.  Scale it so forward quantisation matches qlookup.
      if (quant_step != av2enc::txv::kQuantStep) {
        params.quant_factor *=
            static_cast<double>(av2enc::txv::kQuantStep) /
            static_cast<double>(quant_step);
      }

      const int px0 = x0 >> p.sub_x;
      const int py0 = y0 >> p.sub_y;
      dc_predict_kernel<<<1, 256>>>(
          d_recon[plane], p.stride, p.coded_w, p.coded_h, px0, py0,
          plane_coded_w, plane_coded_h, px0 > 0 ? 1 : 0, py0 > 0 ? 1 : 0,
          bit_depth, d_pred);
      ok = check(cudaGetLastError(), "dc_predict_kernel", error);
      if (!ok) return false;

      av2enc::cuda::hbd::HbdTxJob job{};
      job.params = params;
      job.input =
          static_cast<std::uint8_t>(av2enc::cuda::hbd::HbdTxInput::kSourcePred);
      job.src_packed = 0;
      job.src_x = static_cast<std::int16_t>(px0);
      job.src_y = static_cast<std::int16_t>(py0);
      job.src_width = static_cast<std::int16_t>(p.visible_w);
      job.src_height = static_cast<std::int16_t>(p.visible_h);
      job.src_offset = 0;
      job.src_stride = p.coded_w;
      job.pred_offset = 0;
      job.residual_adj_offset = 0;
      job.coeff_offset = 0;
      job.levels_offset = 0;
      job.dequant_offset = 0;
      job.residual_out_offset = 0;
      job.eob_offset = 0;
      job.quant_step = quant_step;

      ok = check(cudaMemcpy(d_job, &job, sizeof(job), cudaMemcpyHostToDevice),
                 "upload job", error);
      if (!ok) return false;

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
      ok = check(av2enc::cuda::hbd::hbd_tx_batch(
                     d_job, 1, buffers, bit_depth,
                     static_cast<int>(max_coded), nullptr),
                 "hbd_tx_batch", error);
      if (!ok) return false;

      reconstruct_kernel<<<1, 256>>>(
          d_recon[plane], p.stride, p.coded_w, p.coded_h, px0, py0,
          plane_coded_w, plane_coded_h, bit_depth, d_pred, d_residual);
      ok = check(cudaGetLastError(), "reconstruct_kernel", error);
      if (!ok) return false;

      std::int32_t eob = 0;
      ok = check(cudaMemcpy(&eob, d_eob, sizeof(eob), cudaMemcpyDeviceToHost),
                 "download eob", error);
      if (!ok) return false;
      ok = check(cudaMemcpy(host_levels.data(), d_levels,
                            static_cast<std::size_t>(params.w) * params.h *
                                sizeof(std::int32_t),
                            cudaMemcpyDeviceToHost),
                 "download levels", error);
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

  for (int plane = 0; plane < num_planes; ++plane) {
    const fmtdev::PlaneLayout& p = layout.planes[plane];
    if (!check(cudaMemcpy(host_recon[plane].samples.data(), d_recon[plane],
                          p.storage_samples * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "download recon plane", error)) {
      return false;
    }
  }

  // Reuse the verified 10-bit entropy/OBU writers.  The tile payload installs
  // the §8.3.1 coefficient-CDF slice for the real base_q_idx; the hbd header
  // carries the format and qindex.
  std::vector<std::uint8_t> payload;
  std::uint32_t tile_bytes = 0;
  av2enc::high_bitdepth::HighBitDepthOptions payload_options;
  payload_options.base_q_idx = 90;
  payload_options.disable_cdf_update = options.disable_cdf_update;
  payload_options.qctx = av2enc::hbd::depth::qctx_slice(options.base_q_idx);
  if (!av2enc::high_bitdepth::assemble_high_bitdepth_tile_payload(
          picture, payload_options, blocks.data(), blocks.size(),
          levels.data(), levels.size(), &payload, &tile_bytes, error)) {
    return false;
  }

  av2enc::hbd::HbdConfig hbd_config;
  hbd_config.max_frame_width = static_cast<std::uint16_t>(picture.format.width);
  hbd_config.max_frame_height =
      static_cast<std::uint16_t>(picture.format.height);
  hbd_config.depth = av2enc::fmt::BitDepth::k10;
  hbd_config.chroma = picture.format.chroma;
  hbd_config.base_q_idx = options.base_q_idx;
  hbd_config.disable_cdf_update = options.disable_cdf_update;
  std::string hbd_reason;
  if (!av2enc::hbd::validate_config(hbd_config, &hbd_reason)) {
    if (error != nullptr) *error = hbd_reason;
    return false;
  }

  result.tile_bytes = tile_bytes;
  result.frame = av2enc::hbd::assemble_still_picture(hbd_config, payload);
  result.frame_bytes = static_cast<std::uint32_t>(result.frame.size());
  result.blocks = std::move(blocks);
  result.levels = std::move(levels);
  for (int plane = 0; plane < num_planes; ++plane) {
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
  // Size the arena from the actual chroma geometry: 4:2:2/4:4:4/4:0:0 have
  // different plane storage than 4:2:0, and 4:4:4 is twice the 4:2:0 chroma.
  config.format = picture.format;
  config.bit_depth = 10;
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
