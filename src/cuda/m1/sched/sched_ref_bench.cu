// Reference M1 stage executor (verification only).  See ref_exec.h.

#include "cuda/m1/sched/ref_exec.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/arena.h"
#include "cuda/cfl/cfl.h"
#include "cuda/intra_modes/intra_modes.h"
#include "cuda/tx_variants/tx_variants.h"

namespace av2enc::cuda::m1::sched {
namespace {

using av2enc::cuda::intra_modes::IntraBlock;
using av2enc::cuda::intra_modes::IntraJob;

struct ReconJob {
  const std::uint8_t* pred;
  const std::int16_t* residual;
  std::uint8_t* plane;
  int stride;
  int x;
  int y;
  int w;
  int h;
  int plane_w;
  int plane_h;
};

struct BlockCopyJob {
  const std::uint8_t* src;
  int src_stride;
  int src_x;
  int src_y;
  std::uint8_t* dst;
  int dst_stride;
  int dst_x;
  int dst_y;
  int w;
  int h;
  int dst_w;
  int dst_h;
};

struct ResidualJob {
  const std::uint8_t* source;
  int source_stride;
  int source_max_x;
  int source_max_y;
  int x;
  int y;
  int w;
  int h;
  const std::uint8_t* pred;
  int pred_stride;
  std::int16_t* residual;
  int residual_stride;
};

__global__ void recon_batch_kernel(const ReconJob* jobs, int count) {
  const int id = static_cast<int>(blockIdx.x);
  if (id >= count) {
    return;
  }
  const ReconJob& j = jobs[id];
  const int total = j.w * j.h;
  for (int i = static_cast<int>(threadIdx.x); i < total;
       i += static_cast<int>(blockDim.x)) {
    const int yy = i / j.w;
    const int xx = i - yy * j.w;
    const int py = j.y + yy;
    const int px = j.x + xx;
    if (py < 0 || py >= j.plane_h || px < 0 || px >= j.plane_w) {
      continue;
    }
    const int value = static_cast<int>(j.pred[i]) +
                      static_cast<int>(j.residual[i]);
    const int clipped = value < 0 ? 0 : (value > 255 ? 255 : value);
    j.plane[static_cast<std::size_t>(py) * static_cast<std::size_t>(j.stride) +
            static_cast<std::size_t>(px)] =
        static_cast<std::uint8_t>(clipped);
  }
}

__global__ void block_copy_batch_kernel(const BlockCopyJob* jobs, int count) {
  const int id = static_cast<int>(blockIdx.x);
  if (id >= count) {
    return;
  }
  const BlockCopyJob& j = jobs[id];
  const int total = j.w * j.h;
  for (int i = static_cast<int>(threadIdx.x); i < total;
       i += static_cast<int>(blockDim.x)) {
    const int yy = i / j.w;
    const int xx = i - yy * j.w;
    const int sy = j.src_y + yy;
    const int sx = j.src_x + xx;
    const int dy = j.dst_y + yy;
    const int dx = j.dst_x + xx;
    if (dy < 0 || dy >= j.dst_h || dx < 0 || dx >= j.dst_w) {
      continue;
    }
    j.dst[static_cast<std::size_t>(dy) * static_cast<std::size_t>(j.dst_stride) +
          static_cast<std::size_t>(dx)] =
        j.src[static_cast<std::size_t>(sy) *
                  static_cast<std::size_t>(j.src_stride) +
              static_cast<std::size_t>(sx)];
  }
}

__global__ void residual_batch_kernel(const ResidualJob* jobs, int count) {
  const int id = static_cast<int>(blockIdx.x);
  if (id >= count) {
    return;
  }
  const ResidualJob& j = jobs[id];
  const int total = j.w * j.h;
  for (int i = static_cast<int>(threadIdx.x); i < total;
       i += static_cast<int>(blockDim.x)) {
    const int yy = i / j.w;
    const int xx = i - yy * j.w;
    int sy = j.y + yy;
    int sx = j.x + xx;
    sy = sy < 0 ? 0 : (sy > j.source_max_y ? j.source_max_y : sy);
    sx = sx < 0 ? 0 : (sx > j.source_max_x ? j.source_max_x : sx);
    const int src = static_cast<int>(
        j.source[static_cast<std::size_t>(sy) *
                     static_cast<std::size_t>(j.source_stride) +
                 static_cast<std::size_t>(sx)]);
    const int pred =
        static_cast<int>(j.pred[static_cast<std::size_t>(yy) *
                                    static_cast<std::size_t>(j.pred_stride) +
                                static_cast<std::size_t>(xx)]);
    j.residual[static_cast<std::size_t>(yy) *
                   static_cast<std::size_t>(j.residual_stride) +
               static_cast<std::size_t>(xx)] =
        static_cast<std::int16_t>(src - pred);
  }
}

constexpr int kThreads = 128;

bool check_cuda(cudaError_t status, const char* what, std::string& error) {
  if (status == cudaSuccess) {
    return true;
  }
  error = std::string(what) + ": " + cudaGetErrorString(status);
  return false;
}

bool is_smooth_mode(int mode) {
  return mode == 9 || mode == 10 || mode == 11;  // SMOOTH / SMOOTH_V / SMOOTH_H
}

template <typename T>
bool alloc_array(av2enc::cuda::Arena& arena, std::size_t count, T** out,
                 const char* what, std::string& error) {
  if (count == 0) {
    *out = nullptr;
    return true;
  }
  void* raw = nullptr;
  const av2enc::cuda::Status status = arena.allocate_checked(
      count * sizeof(T), &raw, alignof(T) > av2enc::cuda::Arena::kAlignment
                                   ? alignof(T)
                                   : av2enc::cuda::Arena::kAlignment);
  if (!av2enc::cuda::is_ok(status) || raw == nullptr) {
    error = std::string("arena exhausted allocating ") + what;
    return false;
  }
  *out = static_cast<T*>(raw);
  return true;
}

}  // namespace

struct RefM1Executor::Impl {
  FramePlan frame{};
  RefExecOptions options{};
  av2enc::cuda::Arena arena;
  bool ready = false;
  int launches = 0;

  // Planes.
  std::uint8_t* recon[3] = {nullptr, nullptr, nullptr};
  std::uint8_t* source[3] = {nullptr, nullptr, nullptr};
  int recon_stride[3] = {0, 0, 0};
  int recon_height[3] = {0, 0, 0};
  int source_stride[3] = {0, 0, 0};

  // Per-job buffers (indexed by plan job order / absolute element offsets).
  std::uint8_t* pred = nullptr;
  std::int16_t* residual_coded = nullptr;
  std::int16_t* recon_residual = nullptr;
  std::int16_t* residual_adj = nullptr;
  std::int32_t* levels = nullptr;
  std::int32_t* dequant = nullptr;
  double* coeff = nullptr;
  TxBlockJob* tx_jobs = nullptr;
  ReconJob* recon_jobs = nullptr;

  // Prediction descriptors, grouped per level by kind.  Luma and non-CfL
  // chroma blocks share one heterogeneous batch (IntraJob carries its plane),
  // so a level normally needs one prediction launch plus the CfL sequence.
  IntraJob* predict_jobs = nullptr;
  IntraJob* cfl_jobs = nullptr;
  av2enc::cfl::Block* cfl_blocks = nullptr;
  BlockCopyJob* cfl_to_plane = nullptr;
  BlockCopyJob* cfl_from_plane = nullptr;
  ResidualJob* cfl_residual = nullptr;

  struct LevelDev {
    int predict_begin = 0;
    int predict_count = 0;
    int cfl_begin = 0;
    int cfl_count = 0;
    int tx_begin = 0;
    int tx_count = 0;
    int recon_begin = 0;
    int recon_count = 0;
    int span_begin = 0;
    int span_count = 0;
  };
  std::vector<LevelDev> level_dev;

  std::size_t workspace_bytes = 0;

  void reset_launches() { launches = 0; }
};

namespace {

// Builds one IntraBlock.  `plane` uses job.plane; the recon/source pointers
// are the persistent executor planes.
IntraBlock make_intra_block(const RefM1Executor::Impl& impl, const PlaneJob& j,
                            bool cfl_dc) {
  IntraBlock b;
  b.recon_plane = impl.recon[j.plane];
  b.recon_stride = impl.recon_stride[j.plane];
  b.x0 = j.x;
  b.y0 = j.y;
  b.width = j.w;
  b.height = j.h;
  b.max_x = j.max_x;
  b.max_y = j.max_y;
  b.have_left = (j.x > 0) ? 1 : 0;
  b.have_above = (j.y > 0) ? 1 : 0;
  b.num4_above_right = j.n4_above_right;
  b.num4_below_left = j.n4_below_left;
  b.plane = j.plane;
  b.mode = static_cast<av2enc::cuda::intra_modes::IntraMode>(cfl_dc ? 0 : j.mode);
  b.angle_delta = cfl_dc ? 0 : j.angle;
  b.enable_edge_filter = impl.options.enable_edge_filter ? 1 : 0;
  b.source_plane = impl.source[j.plane];
  b.source_stride = impl.source_stride[j.plane];
  b.source_max_x =
      (j.plane == 0 ? impl.frame.width : impl.frame.chroma_width) - 1;
  b.source_max_y =
      (j.plane == 0 ? impl.frame.height : impl.frame.chroma_height) - 1;
  return b;
}

int smooth_above(const std::vector<std::uint8_t>& map, int cols, int rows,
                 int cell_x, int cell_y) {
  if (cell_y <= 0 || cell_x >= cols || cell_x < 0 || cell_y > rows) {
    return 0;
  }
  return map[static_cast<std::size_t>(cell_y - 1) *
                 static_cast<std::size_t>(cols) +
             static_cast<std::size_t>(cell_x)] != 0
             ? 1
             : 0;
}

int smooth_left(const std::vector<std::uint8_t>& map, int cols, int rows,
                int cell_x, int cell_y) {
  if (cell_x <= 0 || cell_y >= rows || cell_y < 0 || cell_x > cols) {
    return 0;
  }
  return map[static_cast<std::size_t>(cell_y) *
                 static_cast<std::size_t>(cols) +
             static_cast<std::size_t>(cell_x - 1)] != 0
             ? 1
             : 0;
}

}  // namespace

RefM1Executor::RefM1Executor() : impl_(std::make_unique<Impl>()) {}
RefM1Executor::~RefM1Executor() { impl_->arena.release(); }

bool RefM1Executor::prepare(const Plan& plan, const FramePlan& frame,
                            const RefExecOptions& options, cudaStream_t stream,
                            std::string& error) {
  Impl& im = *impl_;
  im.frame = frame;
  im.options = options;
  im.ready = false;
  im.arena.release();

  const int n = static_cast<int>(plan.jobs().size());
  if (n == 0) {
    error = "empty plan";
    return false;
  }
  const int luma_pw = frame.padded_luma_width;
  const int luma_ph = frame.padded_luma_height;
  const int chroma_pw = frame.padded_chroma_width;
  const int chroma_ph = frame.padded_chroma_height;
  im.recon_stride[0] = luma_pw;
  im.recon_height[0] = luma_ph;
  im.recon_stride[1] = chroma_pw;
  im.recon_height[1] = chroma_ph;
  im.recon_stride[2] = chroma_pw;
  im.recon_height[2] = chroma_ph;
  im.source_stride[0] = frame.width;
  im.source_stride[1] = frame.chroma_width;
  im.source_stride[2] = frame.chroma_width;

  std::int64_t total_elems = 0;
  std::int64_t adj_elems = 0;
  int cfl_count = 0;
  for (const PlaneJob& j : plan.jobs()) {
    total_elems = std::max<std::int64_t>(
        total_elems,
        static_cast<std::int64_t>(j.buf_offset) +
            static_cast<std::int64_t>(j.w) * j.h);
    adj_elems = std::max<std::int64_t>(
        adj_elems, static_cast<std::int64_t>(j.adj_offset) +
                       static_cast<std::int64_t>(j.tx_w) * j.tx_h);
    if (j.is_cfl != 0) {
      ++cfl_count;
    }
  }
  const std::size_t luma_bytes =
      static_cast<std::size_t>(luma_pw) * static_cast<std::size_t>(luma_ph);
  const std::size_t chroma_bytes =
      static_cast<std::size_t>(chroma_pw) * static_cast<std::size_t>(chroma_ph);
  const std::size_t source_bytes =
      static_cast<std::size_t>(frame.width) * frame.height +
      2u * static_cast<std::size_t>(frame.chroma_width) * frame.chroma_height;
  const std::size_t buffer_bytes =
      static_cast<std::size_t>(total_elems) * 5u +
      static_cast<std::size_t>(adj_elems) * 18u;
  const std::size_t descriptor_bytes =
      static_cast<std::size_t>(n) *
          (sizeof(IntraJob) * 3u + sizeof(TxBlockJob) + sizeof(ReconJob)) +
      static_cast<std::size_t>(cfl_count) *
          (sizeof(av2enc::cfl::Block) + 2u * sizeof(BlockCopyJob) +
           sizeof(ResidualJob));
  const std::size_t plane_bytes = 3u * luma_bytes + 2u * chroma_bytes;
  const std::size_t capacity =
      plane_bytes + source_bytes + buffer_bytes + descriptor_bytes +
      8u * 1024u * 1024u;
  im.workspace_bytes = capacity;

  const av2enc::cuda::Status init_status =
      im.arena.init(static_cast<std::uint64_t>(capacity));
  if (!av2enc::cuda::is_ok(init_status)) {
    error = std::string("arena init: ") + av2enc::cuda::status_name(init_status);
    return false;
  }

  std::uint8_t* luma = nullptr;
  std::uint8_t* u_plane = nullptr;
  std::uint8_t* v_plane = nullptr;
  if (!alloc_array(im.arena, luma_bytes, &luma, "luma recon", error) ||
      !alloc_array(im.arena, chroma_bytes, &u_plane, "u recon", error) ||
      !alloc_array(im.arena, chroma_bytes, &v_plane, "v recon", error) ||
      !alloc_array(im.arena, source_bytes, &im.source[0], "source", error)) {
    return false;
  }
  im.recon[0] = luma;
  im.recon[1] = u_plane;
  im.recon[2] = v_plane;
  im.source[1] = im.source[0] + static_cast<std::size_t>(frame.width) * frame.height;
  im.source[2] = im.source[1] +
                 static_cast<std::size_t>(frame.chroma_width) * frame.chroma_height;

  if (!alloc_array(im.arena, static_cast<std::size_t>(total_elems), &im.pred,
                   "pred", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(total_elems),
                   &im.residual_coded, "residual_coded", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(total_elems),
                   &im.recon_residual, "recon_residual", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(adj_elems),
                   &im.residual_adj, "residual_adj", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(adj_elems), &im.levels,
                   "levels", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(adj_elems), &im.dequant,
                   "dequant", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(adj_elems), &im.coeff,
                   "coeff", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(n), &im.tx_jobs,
                   "tx_jobs", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(n), &im.recon_jobs,
                   "recon_jobs", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(n), &im.predict_jobs,
                   "predict_jobs", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(cfl_count),
                   &im.cfl_jobs, "cfl_jobs", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(cfl_count),
                   &im.cfl_blocks, "cfl_blocks", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(cfl_count),
                   &im.cfl_to_plane, "cfl_to_plane", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(cfl_count),
                   &im.cfl_from_plane, "cfl_from_plane", error) ||
      !alloc_array(im.arena, static_cast<std::size_t>(cfl_count),
                   &im.cfl_residual, "cfl_residual", error)) {
    return false;
  }

  // Smoothness maps for the edge filter type (§7.13.2.15/16).  The CPU builds
  // them from IntraJointMode / uv_smooth as the frame is coded; here the
  // decisions are already known for every leaf.
  const int mi_cols = frame.mi_cols;
  const int mi_rows = frame.mi_rows;
  const int chroma_cols = frame.mi_cols / 2;
  const int chroma_rows = frame.mi_rows / 2;
  std::vector<std::uint8_t> luma_smooth(
      static_cast<std::size_t>(mi_cols) * mi_rows, 0);
  std::vector<std::uint8_t> chroma_smooth(
      static_cast<std::size_t>(chroma_cols) * chroma_rows, 0);
  for (int bi = 0; bi < frame.block_count; ++bi) {
    const BlockPlan& b = frame.blocks[bi];
    if (is_smooth_mode(b.luma_mode)) {
      for (int y = 0; y < b.luma_h / 4; ++y) {
        for (int x = 0; x < b.luma_w / 4; ++x) {
          const int yy = b.mi_row + y;
          const int xx = b.mi_col + x;
          if (yy < mi_rows && xx < mi_cols) {
            luma_smooth[static_cast<std::size_t>(yy) * mi_cols + xx] = 1;
          }
        }
      }
    }
    if (b.has_chroma && b.is_cfl == 0 && is_smooth_mode(b.chroma_mode)) {
      for (int y = 0; y < b.chroma_h / 4; ++y) {
        for (int x = 0; x < b.chroma_w / 4; ++x) {
          const int yy = b.mi_row / 2 + y;
          const int xx = b.mi_col / 2 + x;
          if (yy < chroma_rows && xx < chroma_cols) {
            chroma_smooth[static_cast<std::size_t>(yy) * chroma_cols + xx] = 1;
          }
        }
      }
    }
  }

  std::vector<IntraJob> h_predict;
  std::vector<IntraJob> h_cfl;
  std::vector<av2enc::cfl::Block> h_cfl_blocks;
  std::vector<BlockCopyJob> h_to_plane;
  std::vector<BlockCopyJob> h_from_plane;
  std::vector<ResidualJob> h_cfl_residual;
  std::vector<TxBlockJob> h_tx(static_cast<std::size_t>(n));
  std::vector<ReconJob> h_recon(static_cast<std::size_t>(n));
  h_predict.reserve(static_cast<std::size_t>(n));
  h_cfl.reserve(static_cast<std::size_t>(cfl_count));
  h_cfl_blocks.reserve(static_cast<std::size_t>(cfl_count));
  h_to_plane.reserve(static_cast<std::size_t>(cfl_count));
  h_from_plane.reserve(static_cast<std::size_t>(cfl_count));
  h_cfl_residual.reserve(static_cast<std::size_t>(cfl_count));

  im.level_dev.assign(static_cast<std::size_t>(plan.levels()), Impl::LevelDev{});

  for (int level = 0; level < plan.levels(); ++level) {
    Impl::LevelDev& ld = im.level_dev[static_cast<std::size_t>(level)];
    const LevelBatch batch = plan.level_batch(level);
    ld.tx_begin = batch.job_begin;
    ld.tx_count = batch.job_end - batch.job_begin;
    ld.recon_begin = batch.job_begin;
    ld.recon_count = batch.job_end - batch.job_begin;
    ld.span_begin = batch.class_begin;
    ld.span_count = batch.class_end - batch.class_begin;
    ld.predict_begin = static_cast<int>(h_predict.size());
    ld.cfl_begin = static_cast<int>(h_cfl.size());
    for (int ji = batch.job_begin; ji < batch.job_end; ++ji) {
      const PlaneJob& j = plan.job(ji);
      const bool scaled = (j.w != j.tx_w) || (j.h != j.tx_h);
      const auto params = av2enc::txv::make_tx_params(
          j.w, j.h, av2enc::txv::TxType::kDctDct);
      if (!params.has_value()) {
        error = "unsupported M1 transform size " + std::to_string(j.w) + "x" +
                std::to_string(j.h);
        return false;
      }
      TxBlockJob tj;
      tj.params = *params;
      tj.in_offset = j.adj_offset;
      tj.out_offset = j.buf_offset;
      h_tx[static_cast<std::size_t>(ji)] = tj;

      ReconJob rj;
      rj.pred = im.pred + j.buf_offset;
      rj.residual = im.recon_residual + j.buf_offset;
      rj.plane = im.recon[j.plane];
      rj.stride = im.recon_stride[j.plane];
      rj.x = j.x;
      rj.y = j.y;
      rj.w = j.w;
      rj.h = j.h;
      rj.plane_w = im.recon_stride[j.plane];
      rj.plane_h = im.recon_height[j.plane];
      h_recon[static_cast<std::size_t>(ji)] = rj;

      if (j.plane == 0) {
        IntraJob job;
        job.block = make_intra_block(im, j, false);
        job.block.filter_type_above = smooth_above(
            luma_smooth, mi_cols, mi_rows, j.x / 4, j.y / 4);
        job.block.filter_type_left = smooth_left(
            luma_smooth, mi_cols, mi_rows, j.x / 4, j.y / 4);
        job.pred = im.pred + j.buf_offset;
        job.residual = (scaled ? im.residual_coded + j.buf_offset
                               : im.residual_adj + j.adj_offset);
        h_predict.push_back(job);
        ++ld.predict_count;
      } else if (j.is_cfl == 0) {
        IntraJob job;
        job.block = make_intra_block(im, j, false);
        job.block.filter_type_above = smooth_above(
            chroma_smooth, chroma_cols, chroma_rows, j.x / 4, j.y / 4);
        job.block.filter_type_left = smooth_left(
            chroma_smooth, chroma_cols, chroma_rows, j.x / 4, j.y / 4);
        job.pred = im.pred + j.buf_offset;
        job.residual = (scaled ? im.residual_coded + j.buf_offset
                               : im.residual_adj + j.adj_offset);
        h_predict.push_back(job);
        ++ld.predict_count;
      } else {
        IntraJob job;
        job.block = make_intra_block(im, j, true);
        job.block.filter_type_above = smooth_above(
            chroma_smooth, chroma_cols, chroma_rows, j.x / 4, j.y / 4);
        job.block.filter_type_left = smooth_left(
            chroma_smooth, chroma_cols, chroma_rows, j.x / 4, j.y / 4);
        job.pred = im.pred + j.buf_offset;
        job.residual = nullptr;
        h_cfl.push_back(job);
        ++ld.cfl_count;

        av2enc::cfl::Block cb;
        cb.luma = im.recon[0];
        cb.luma_stride = im.recon_stride[0];
        cb.luma_max_x = frame.mi_cols * 4 - 1;
        cb.luma_max_y = frame.mi_rows * 4 - 1;
        cb.sb_top_y = ((j.y * 2) / 64) * 64;
        cb.chroma = im.recon[j.plane];
        cb.chroma_stride = im.recon_stride[j.plane];
        cb.x = j.x;
        cb.y = j.y;
        cb.w = j.w;
        cb.h = j.h;
        cb.coded_width = frame.mi_cols * 2;
        cb.coded_height = frame.mi_rows * 2;
        cb.have_above = av2enc::cfl::cfl_avail_above(j.y, 1) ? 1 : 0;
        cb.have_left = av2enc::cfl::cfl_avail_left(j.x, 1) ? 1 : 0;
        cb.bit_depth = 8;
        cb.ds_filter = 0;
        cb.sub_x = 1;
        cb.sub_y = 1;
        h_cfl_blocks.push_back(cb);

        BlockCopyJob to;
        to.src = im.pred + j.buf_offset;
        to.src_stride = j.w;
        to.src_x = 0;
        to.src_y = 0;
        to.dst = im.recon[j.plane];
        to.dst_stride = im.recon_stride[j.plane];
        to.dst_x = j.x;
        to.dst_y = j.y;
        to.w = j.w;
        to.h = j.h;
        to.dst_w = im.recon_stride[j.plane];
        to.dst_h = im.recon_height[j.plane];
        h_to_plane.push_back(to);

        BlockCopyJob from = to;
        from.src = im.recon[j.plane];
        from.src_stride = im.recon_stride[j.plane];
        from.src_x = j.x;
        from.src_y = j.y;
        from.dst = im.pred + j.buf_offset;
        from.dst_stride = j.w;
        from.dst_x = 0;
        from.dst_y = 0;
        h_from_plane.push_back(from);

        ResidualJob rj2;
        rj2.source = im.source[j.plane];
        rj2.source_stride = im.source_stride[j.plane];
        rj2.source_max_x =
            (j.plane == 0 ? frame.width : frame.chroma_width) - 1;
        rj2.source_max_y =
            (j.plane == 0 ? frame.height : frame.chroma_height) - 1;
        rj2.x = j.x;
        rj2.y = j.y;
        rj2.w = j.w;
        rj2.h = j.h;
        rj2.pred = im.pred + j.buf_offset;
        rj2.pred_stride = j.w;
        rj2.residual = (scaled ? im.residual_coded + j.buf_offset
                               : im.residual_adj + j.adj_offset);
        rj2.residual_stride = j.w;
        h_cfl_residual.push_back(rj2);
      }
    }
  }

  // Upload descriptors once.
  const auto upload = [&](const void* host, void* device, std::size_t bytes,
                          const char* what) {
    if (bytes == 0) {
      return true;
    }
    return check_cuda(cudaMemcpyAsync(device, host, bytes,
                                      cudaMemcpyHostToDevice, stream),
                      what, error);
  };
  if (!upload(h_predict.data(), im.predict_jobs,
              h_predict.size() * sizeof(IntraJob), "upload predict jobs") ||
      !upload(h_cfl.data(), im.cfl_jobs, h_cfl.size() * sizeof(IntraJob),
              "upload cfl jobs") ||
      !upload(h_cfl_blocks.data(), im.cfl_blocks,
              h_cfl_blocks.size() * sizeof(av2enc::cfl::Block),
              "upload cfl blocks") ||
      !upload(h_to_plane.data(), im.cfl_to_plane,
              h_to_plane.size() * sizeof(BlockCopyJob), "upload cfl to plane") ||
      !upload(h_from_plane.data(), im.cfl_from_plane,
              h_from_plane.size() * sizeof(BlockCopyJob),
              "upload cfl from plane") ||
      !upload(h_cfl_residual.data(), im.cfl_residual,
              h_cfl_residual.size() * sizeof(ResidualJob),
              "upload cfl residual") ||
      !upload(h_tx.data(), im.tx_jobs, h_tx.size() * sizeof(TxBlockJob),
              "upload tx jobs") ||
      !upload(h_recon.data(), im.recon_jobs, h_recon.size() * sizeof(ReconJob),
              "upload recon jobs")) {
    return false;
  }
  if (!check_cuda(cudaStreamSynchronize(stream), "prepare sync", error)) {
    return false;
  }
  // Descriptors are device-resident now; the host-side block list is no longer
  // needed and may be owned by a temporary adapter.
  im.frame.blocks = nullptr;
  im.frame.block_count = 0;
  im.ready = true;
  return true;
}

int RefM1Executor::recon_stride(int plane) const {
  return impl_->recon_stride[plane];
}
int RefM1Executor::recon_height(int plane) const {
  return impl_->recon_height[plane];
}
std::size_t RefM1Executor::workspace_bytes() const {
  return impl_->workspace_bytes;
}
int RefM1Executor::launch_count() const { return impl_->launches; }
void RefM1Executor::reset_launch_count() { impl_->reset_launches(); }

bool RefM1Executor::upload_source(const std::uint8_t* y, const std::uint8_t* u,
                                  const std::uint8_t* v, cudaStream_t stream,
                                  std::string& error) {
  Impl& im = *impl_;
  const std::size_t luma_bytes =
      static_cast<std::size_t>(im.frame.width) * im.frame.height;
  const std::size_t chroma_bytes =
      static_cast<std::size_t>(im.frame.chroma_width) * im.frame.chroma_height;
  if (!check_cuda(cudaMemcpyAsync(im.source[0], y, luma_bytes,
                                  cudaMemcpyHostToDevice, stream),
                  "upload Y", error) ||
      !check_cuda(cudaMemcpyAsync(im.source[1], u, chroma_bytes,
                                  cudaMemcpyHostToDevice, stream),
                  "upload U", error) ||
      !check_cuda(cudaMemcpyAsync(im.source[2], v, chroma_bytes,
                                  cudaMemcpyHostToDevice, stream),
                  "upload V", error)) {
    return false;
  }
  return true;
}

bool RefM1Executor::zero_recon(cudaStream_t stream, std::string& error) {
  Impl& im = *impl_;
  for (int plane = 0; plane < 3; ++plane) {
    const std::size_t bytes = static_cast<std::size_t>(im.recon_stride[plane]) *
                              im.recon_height[plane];
    if (!check_cuda(cudaMemsetAsync(im.recon[plane], 0, bytes, stream),
                    "zero recon", error)) {
      return false;
    }
  }
  return true;
}

bool RefM1Executor::download_recon(std::uint8_t* y, std::uint8_t* u,
                                   std::uint8_t* v, cudaStream_t stream,
                                   std::string& error) {
  Impl& im = *impl_;
  std::uint8_t* dst[3] = {y, u, v};
  for (int plane = 0; plane < 3; ++plane) {
    const std::size_t bytes = static_cast<std::size_t>(im.recon_stride[plane]) *
                              im.recon_height[plane];
    if (!check_cuda(cudaMemcpyAsync(dst[plane], im.recon[plane], bytes,
                                    cudaMemcpyDeviceToHost, stream),
                    "download recon", error)) {
      return false;
    }
  }
  return true;
}

LevelOps RefM1Executor::ops() {
  Impl* im = impl_.get();
  LevelOps ops;
  ops.user = im;
  ops.predict = [](void* user, const Plan& plan, const LevelBatch& batch,
                   cudaStream_t stream, std::string& error) -> bool {
    (void)plan;
    Impl& s = *static_cast<Impl*>(user);
    if (s.level_dev.empty()) {
      error = "executor not prepared";
      return false;
    }
    const Impl::LevelDev& ld = s.level_dev[static_cast<std::size_t>(batch.level)];
    if (ld.predict_count > 0) {
      if (!check_cuda(av2enc::cuda::intra_modes::intra_predict_batch(
                          s.predict_jobs + ld.predict_begin, ld.predict_count,
                          stream),
                      "intra_predict_batch", error)) {
        return false;
      }
      ++s.launches;
    }
    if (ld.cfl_count > 0) {
      if (!check_cuda(av2enc::cuda::intra_modes::intra_predict_batch(
                          s.cfl_jobs + ld.cfl_begin, ld.cfl_count, stream),
                      "intra_predict_batch(cfl dc)", error)) {
        return false;
      }
      ++s.launches;
      block_copy_batch_kernel<<<ld.cfl_count, kThreads, 0, stream>>>(
          s.cfl_to_plane + ld.cfl_begin, ld.cfl_count);
      if (!check_cuda(cudaGetLastError(), "cfl stage-to-plane", error)) {
        return false;
      }
      ++s.launches;
      if (!check_cuda(av2enc::cuda::cfl_predict_derived_batch(
                          s.cfl_blocks + ld.cfl_begin, nullptr, ld.cfl_count,
                          stream),
                      "cfl_predict_derived_batch", error)) {
        return false;
      }
      ++s.launches;
      block_copy_batch_kernel<<<ld.cfl_count, kThreads, 0, stream>>>(
          s.cfl_from_plane + ld.cfl_begin, ld.cfl_count);
      if (!check_cuda(cudaGetLastError(), "cfl stage-back", error)) {
        return false;
      }
      ++s.launches;
      residual_batch_kernel<<<ld.cfl_count, kThreads, 0, stream>>>(
          s.cfl_residual + ld.cfl_begin, ld.cfl_count);
      if (!check_cuda(cudaGetLastError(), "cfl residual", error)) {
        return false;
      }
      ++s.launches;
    }
    return true;
  };

  ops.transform = [](void* user, const Plan& plan, const LevelBatch& batch,
                     cudaStream_t stream, std::string& error) -> bool {
    Impl& s = *static_cast<Impl*>(user);
    const Impl::LevelDev& ld = s.level_dev[static_cast<std::size_t>(batch.level)];
    for (int si = batch.class_begin; si < batch.class_end; ++si) {
      const ClassSpan& span = plan.class_spans()[static_cast<std::size_t>(si)];
      const PlaneJob& first = plan.job(span.job_begin);
      if (first.w == first.tx_w && first.h == first.tx_h) {
        continue;
      }
      const auto params = av2enc::txv::make_tx_params(
          first.w, first.h, av2enc::txv::TxType::kDctDct);
      if (!params.has_value()) {
        error = "downsample: unsupported transform size";
        return false;
      }
      const int count = span.job_end - span.job_begin;
      if (!check_cuda(av2enc::cuda::txv_downsample_residual(
                          s.residual_coded + first.buf_offset,
                          s.residual_adj + first.adj_offset, *params, count,
                          stream),
                      "txv_downsample_residual", error)) {
        return false;
      }
      ++s.launches;
    }
    if (ld.tx_count > 0) {
      if (!check_cuda(av2enc::cuda::txv_forward_transform_multi(
                          s.residual_adj, s.coeff, s.levels,
                          s.tx_jobs + ld.tx_begin, ld.tx_count, stream),
                      "txv_forward_transform_multi", error)) {
        return false;
      }
      ++s.launches;
      if (!check_cuda(av2enc::cuda::txv_inverse_transform_multi(
                          s.levels, s.recon_residual,
                          s.tx_jobs + ld.tx_begin, ld.tx_count, stream),
                      "txv_inverse_transform_multi", error)) {
        return false;
      }
      ++s.launches;
    }
    return true;
  };

  ops.reconstruct = [](void* user, const Plan& plan, const LevelBatch& batch,
                       cudaStream_t stream, std::string& error) -> bool {
    (void)plan;
    Impl& s = *static_cast<Impl*>(user);
    const Impl::LevelDev& ld = s.level_dev[static_cast<std::size_t>(batch.level)];
    if (ld.recon_count > 0) {
      recon_batch_kernel<<<ld.recon_count, kThreads, 0, stream>>>(
          s.recon_jobs + ld.recon_begin, ld.recon_count);
      if (!check_cuda(cudaGetLastError(), "recon_batch_kernel", error)) {
        return false;
      }
      ++s.launches;
    }
    return true;
  };

  ops.launch_count = [](void* user) -> int {
    return static_cast<Impl*>(user)->launches;
  };
  return ops;
}

}  // namespace av2enc::cuda::m1::sched
