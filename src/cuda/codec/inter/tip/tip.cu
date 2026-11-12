// AV2 TIP — CUDA kernels and host launch wrappers.
//
// Every kernel body calls `tip_device.cuh`, the same code the host-emulation
// parity test exercises.  No entry point allocates, copies, or synchronises.

#include "tip_device.cuh"

#include <cstddef>
#include <cstdint>

namespace av2enc::cuda::tip {
namespace {

constexpr int kThreads = 128;

bool valid_field(const TipField& field, const TipMotionFieldConfig& config) {
  if (config.mi_cols <= 0 || config.mi_rows <= 0 ||
      (config.proj_step != 1 && config.proj_step != 2)) {
    return false;
  }
  if (field.mv == nullptr || field.valid == nullptr || field.offset == nullptr) {
    return false;
  }
  return field.w8 == (config.mi_cols >> 1) &&
         field.h8 == (config.mi_rows >> 1) && field.w8 > 0 && field.h8 > 0;
}

bool valid_scratch(const TipFieldScratch& scratch) {
  return scratch.avg_mv != nullptr && scratch.avg_valid != nullptr;
}

int lattice_width(int w8, int step) { return (w8 + step - 1) / step; }
int lattice_height(int h8, int step) { return (h8 + step - 1) / step; }

__global__ void scale_kernel(TipField field, TipMotionFieldConfig config,
                             int ref_offset) {
  const int width = (field.w8 + config.proj_step - 1) / config.proj_step;
  const int height = (field.h8 + config.proj_step - 1) / config.proj_step;
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= width * height) {
    return;
  }
  const int y8 = (index / width) * config.proj_step;
  const int x8 = (index % width) * config.proj_step;
  dev::scale_cell(field, ref_offset, y8, x8);
}

__global__ void hole_fill_kernel(TipField field, TipMotionFieldConfig config,
                                 int superblocks_w, int superblock8) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= superblocks_w * ((field.h8 + superblock8 - 1) / superblock8)) {
    return;
  }
  const int y8 = (index / superblocks_w) * superblock8;
  const int x8 = (index % superblocks_w) * superblock8;
  dev::hole_fill_superblock(field, config, y8, x8);
}

__global__ void block_average_kernel(TipField field,
                                     TipMotionFieldConfig config,
                                     TipFieldScratch scratch) {
  const int width = (field.w8 + config.proj_step - 1) / config.proj_step;
  const int height = (field.h8 + config.proj_step - 1) / config.proj_step;
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= width * height) {
    return;
  }
  const int y8 = (index / width) * config.proj_step;
  const int x8 = (index % width) * config.proj_step;
  dev::block_average_cell(field, config, scratch, y8, x8);
}

__global__ void copy_average_kernel(TipField field, TipMotionFieldConfig config,
                                    TipFieldScratch scratch) {
  const int width = (field.w8 + config.proj_step - 1) / config.proj_step;
  const int height = (field.h8 + config.proj_step - 1) / config.proj_step;
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= width * height) {
    return;
  }
  const int y8 = (index / width) * config.proj_step;
  const int x8 = (index % width) * config.proj_step;
  dev::copy_average_cell(field, scratch, y8, x8);
}

__global__ void sample_gap_kernel(TipField field,
                                  TipMotionFieldConfig config) {
  const int width = (field.w8 + 1) / 2;
  const int height = (field.h8 + 1) / 2;
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= width * height) {
    return;
  }
  const int y8 = (index / width) * 2;
  const int x8 = (index % width) * 2;
  dev::fill_sample_gap_cell(field, config, y8, x8, 0, 1);
  dev::fill_sample_gap_cell(field, config, y8, x8, 1, 0);
  dev::fill_sample_gap_cell(field, config, y8, x8, 1, 1);
}

__global__ void build_plan_kernel(TipField field, TipMotionFieldConfig config,
                                  TipOffsets offsets, int bw4, int bh4,
                                  Mv global_mv, int block_cols,
                                  TipPlanBlock* blocks, Mv* grid0, Mv* grid1) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= block_cols * ((config.mi_rows + bh4 - 1) / bh4)) {
    return;
  }
  const int row = (index / block_cols) * bh4;
  const int col = (index % block_cols) * bw4;
  dev::derive_tip_block(field, config, offsets, bw4, bh4, global_mv, row, col,
                        &blocks[index], grid0, grid1);
}

template <typename Sample>
__global__ void predict_jobs_kernel(const TipPredictJob* __restrict__ jobs) {
  const TipPredictJob job = jobs[blockIdx.x];
  dev::predict_job_thread<Sample>(job, static_cast<int>(threadIdx.x),
                                  static_cast<int>(blockDim.x));
}

template <typename Sample>
__global__ void predict_one_kernel(TipPredictJob job) {
  dev::predict_job_thread<Sample>(job, static_cast<int>(threadIdx.x),
                                  static_cast<int>(blockDim.x));
}

template <typename Sample>
__global__ void predict_plan_kernel(TipPlan plan,
                                    TipPlanPredictConfig config) {
  const TipPlanBlock block = plan.blocks[blockIdx.x];
  for (int plane = 0; plane < 3; ++plane) {
    TipPredictJob job;
    if (dev::plan_block_job(block, config, plane, &job)) {
      dev::predict_job_thread<Sample>(job, static_cast<int>(threadIdx.x),
                                      static_cast<int>(blockDim.x));
    }
  }
}

bool valid_job_config(const TipPredictJob& job) {
  if (job.dst == nullptr || job.w <= 0 || job.h <= 0 ||
      job.dst_stride < job.w || job.frame_w <= 0 || job.frame_h <= 0 ||
      job.num_refs < 1 || job.num_refs > 2 ||
      (job.bit_depth != 8 && job.bit_depth != 10) ||
      (job.sample_bytes != 1 && job.sample_bytes != 2)) {
    return false;
  }
  for (int list = 0; list < 2; ++list) {
    if (list < job.num_refs && job.cwp_weight != 16) {
      if (job.ref[list].data == nullptr || job.ref[list].stride <= 0 ||
          job.ref[list].width <= 0 || job.ref[list].height <= 0 ||
          job.ref[list].stride < job.ref[list].width) {
        return false;
      }
    }
  }
  if (job.num_refs == 1 || job.cwp_weight == 16) {
    if (job.ref[0].data == nullptr || job.ref[0].stride <= 0 ||
        job.ref[0].width <= 0 || job.ref[0].height <= 0 ||
        job.ref[0].stride < job.ref[0].width) {
      return false;
    }
  }
  for (int list = 0; list < (job.num_refs == 1 || job.cwp_weight == 16 ? 1 : 2);
       ++list) {
    if (job.use_ref_area != 0) {
      if (job.area_first_x[list] < 0 || job.area_first_y[list] < 0 ||
          job.area_last_x[list] >= job.ref[list].width ||
          job.area_last_y[list] >= job.ref[list].height ||
          job.area_first_x[list] > job.area_last_x[list] ||
          job.area_first_y[list] > job.area_last_y[list]) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

TipOffsets tip_get_offsets(const TipMotionFieldConfig& config) {
  return dev::tip_offsets(config);
}

cudaError_t tip_scale_motion_field(const TipField& field,
                                   const TipMotionFieldConfig& config,
                                   cudaStream_t stream) {
  if (!valid_field(field, config)) {
    return cudaErrorInvalidValue;
  }
  const TipOffsets offsets = dev::tip_offsets(config);
  const int count = lattice_width(field.w8, config.proj_step) *
                    lattice_height(field.h8, config.proj_step);
  scale_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      field, config, offsets.ref_offset);
  return cudaGetLastError();
}

cudaError_t tip_hole_fill_motion_field(const TipField& field,
                                       const TipMotionFieldConfig& config,
                                       cudaStream_t stream) {
  if (!valid_field(field, config)) {
    return cudaErrorInvalidValue;
  }
  const int superblock8 = 1 << dev::tmvp_shift(config.sb_64x64, config.proj_step);
  const int superblocks_w = (field.w8 + superblock8 - 1) / superblock8;
  const int superblocks_h = (field.h8 + superblock8 - 1) / superblock8;
  const int count = superblocks_w * superblocks_h;
  hole_fill_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      field, config, superblocks_w, superblock8);
  return cudaGetLastError();
}

cudaError_t tip_block_average_motion_field(const TipField& field,
                                           const TipMotionFieldConfig& config,
                                           const TipFieldScratch& scratch,
                                           cudaStream_t stream) {
  if (!valid_field(field, config) || !valid_scratch(scratch)) {
    return cudaErrorInvalidValue;
  }
  const int count = lattice_width(field.w8, config.proj_step) *
                    lattice_height(field.h8, config.proj_step);
  const int blocks = (count + kThreads - 1) / kThreads;
  block_average_kernel<<<blocks, kThreads, 0, stream>>>(field, config,
                                                        scratch);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  copy_average_kernel<<<blocks, kThreads, 0, stream>>>(field, config, scratch);
  return cudaGetLastError();
}

cudaError_t tip_fill_motion_field_sample_gap(
    const TipField& field, const TipMotionFieldConfig& config,
    cudaStream_t stream) {
  if (!valid_field(field, config)) {
    return cudaErrorInvalidValue;
  }
  if (config.proj_step != 2) {
    return cudaSuccess;
  }
  const int count = ((field.w8 + 1) / 2) * ((field.h8 + 1) / 2);
  sample_gap_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0,
                      stream>>>(field, config);
  return cudaGetLastError();
}

cudaError_t tip_setup_motion_field(const TipField& field,
                                   const TipMotionFieldConfig& config,
                                   const TipFieldScratch& scratch,
                                   int allow_tip_hole_fill,
                                   cudaStream_t stream) {
  if (!valid_field(field, config)) {
    return cudaErrorInvalidValue;
  }
  if (config.enable_tip == 0) {
    return cudaSuccess;
  }
  if (allow_tip_hole_fill != 0 && !valid_scratch(scratch)) {
    return cudaErrorInvalidValue;
  }
  const TipOffsets offsets = dev::tip_offsets(config);
  const int scale_count = lattice_width(field.w8, config.proj_step) *
                          lattice_height(field.h8, config.proj_step);
  const int scale_blocks = (scale_count + kThreads - 1) / kThreads;
  scale_kernel<<<scale_blocks, kThreads, 0, stream>>>(field, config,
                                                      offsets.ref_offset);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  if (allow_tip_hole_fill != 0) {
    const int superblock8 =
        1 << dev::tmvp_shift(config.sb_64x64, config.proj_step);
    const int superblocks_w = (field.w8 + superblock8 - 1) / superblock8;
    const int superblocks_h = (field.h8 + superblock8 - 1) / superblock8;
    const int sb_count = superblocks_w * superblocks_h;
    hole_fill_kernel<<<(sb_count + kThreads - 1) / kThreads, kThreads, 0,
                       stream>>>(field, config, superblocks_w, superblock8);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
    block_average_kernel<<<scale_blocks, kThreads, 0, stream>>>(field, config,
                                                                scratch);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
    copy_average_kernel<<<scale_blocks, kThreads, 0, stream>>>(field, config,
                                                              scratch);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
  }
  if (config.proj_step == 2) {
    const int gap_count = ((field.w8 + 1) / 2) * ((field.h8 + 1) / 2);
    sample_gap_kernel<<<(gap_count + kThreads - 1) / kThreads, kThreads, 0,
                        stream>>>(field, config);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
  }
  return cudaSuccess;
}

cudaError_t tip_build_plan(const TipField& field,
                           const TipMotionFieldConfig& config,
                           const TipPlanConfig& plan_config, TipPlan* plan,
                           cudaStream_t stream) {
  if (plan == nullptr || plan->blocks == nullptr || plan->mv_grid == nullptr ||
      plan->block_capacity < 0 || !valid_field(field, config) ||
      plan_config.bw4 <= 0 || plan_config.bh4 <= 0 ||
      plan_config.tip_global_wtd_index < 0 ||
      plan_config.tip_global_wtd_index > 7) {
    return cudaErrorInvalidValue;
  }
  const int block_cols = (config.mi_cols + plan_config.bw4 - 1) /
                         plan_config.bw4;
  const int block_rows = (config.mi_rows + plan_config.bh4 - 1) /
                         plan_config.bh4;
  const int count = block_cols * block_rows;
  if (count > plan->block_capacity) {
    return cudaErrorInvalidValue;
  }
  plan->num_blocks = count;
  plan->weight =
      inter::temporal::kTipWeightingFactor[plan_config.tip_global_wtd_index];
  plan->offsets = dev::tip_offsets(config);
  if (count == 0) {
    return cudaSuccess;
  }
  build_plan_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      field, config, plan->offsets, plan_config.bw4, plan_config.bh4,
      plan_config.tip_global_mv, block_cols, plan->blocks, plan->mv_grid,
      plan->mv_grid1);
  return cudaGetLastError();
}

cudaError_t tip_predict_block(const TipPredictJob& job,
                              cudaStream_t stream) {
  if (!valid_job_config(job)) {
    return cudaErrorInvalidValue;
  }
  if (job.sample_bytes == 1) {
    predict_one_kernel<std::uint8_t><<<1, kThreads, 0, stream>>>(job);
  } else {
    predict_one_kernel<std::uint16_t><<<1, kThreads, 0, stream>>>(job);
  }
  return cudaGetLastError();
}

cudaError_t tip_predict_batch(const TipPredictJob* device_jobs, int count,
                              int sample_bytes, cudaStream_t stream) {
  if (device_jobs == nullptr || count <= 0 ||
      (sample_bytes != 1 && sample_bytes != 2)) {
    return cudaErrorInvalidValue;
  }
  if (sample_bytes == 2) {
    predict_jobs_kernel<std::uint16_t><<<count, kThreads, 0, stream>>>(
        device_jobs);
  } else {
    predict_jobs_kernel<std::uint8_t><<<count, kThreads, 0, stream>>>(
        device_jobs);
  }
  return cudaGetLastError();
}

cudaError_t tip_predict_plan(const TipPlan& plan,
                             const TipPlanPredictConfig& config,
                             cudaStream_t stream) {
  if (plan.blocks == nullptr || plan.num_blocks < 0 ||
      (config.bit_depth != 8 && config.bit_depth != 10) ||
      (config.sample_bytes != 1 && config.sample_bytes != 2) ||
      config.frame_w <= 0 || config.frame_h <= 0) {
    return cudaErrorInvalidValue;
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (config.dst[plane] == nullptr || config.ref[plane][0].data == nullptr ||
        config.ref[plane][1].data == nullptr ||
        config.dst_stride[plane] < config.dst_width[plane] ||
        config.dst_width[plane] <= 0 || config.dst_height[plane] <= 0 ||
        config.sub_x[plane] < 0 || config.sub_x[plane] > 1 ||
        config.sub_y[plane] < 0 || config.sub_y[plane] > 1) {
      return cudaErrorInvalidValue;
    }
  }
  if (plan.num_blocks == 0) {
    return cudaSuccess;
  }
  if (config.sample_bytes == 1) {
    predict_plan_kernel<std::uint8_t><<<plan.num_blocks, kThreads, 0, stream>>>(
        plan, config);
  } else {
    predict_plan_kernel<std::uint16_t>
        <<<plan.num_blocks, kThreads, 0, stream>>>(plan, config);
  }
  return cudaGetLastError();
}

}  // namespace av2enc::cuda::tip
