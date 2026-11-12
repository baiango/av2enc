// Real CUDA parity test for AV2 TIP.
//
// Build/run on the Linux/CUDA target:
//   ./build_gpu_test.sh
//
// Coverage:
//   * §7.10.2–.5 randomised motion fields (projection, hole fill, block
//     average, temporal scaling, sample-gap fill, offsets) against the CPU
//     reference, through the kernels;
//   * §7.10.6 plan/4x4 grid parity;
//   * §7.13.3.2 8x8/16x16 prediction, 8-bit (CPU reference) and 10-bit
//     (bit-depth transcription), all weights/filters/edge blocks;
//   * a full-clip parity case: CPU motion-field estimation seeds the field,
//     then CPU and GPU independently run setup -> plan -> predict for two TIP
//     frames in 8-bit and 10-bit.
//
// Every device buffer comes from `av2enc::cuda::Arena`; there is no
// cudaMalloc/cudaFree/cudaMallocHost here.  The arena is sized from
// `required_device_bytes()`, the one-byte-short case is checked, and two
// repeated frames use mark()/rewind() with no net growth.

#define AV2ENC_TIP_TEST_NO_MAIN
#include "tip_emulation_test.cpp"

#include "cuda/arena.h"
#include "cuda/codec/inter/tip/tip.h"

namespace {

using av2enc::cuda::Arena;
using av2enc::cuda::Status;
using av2enc::cuda::is_ok;

namespace ctip = av2enc::cuda::tip;
namespace cdev = av2enc::cuda::tip::dev;
namespace temporal = av2enc::inter::temporal;

bool upload_bytes(void* device, const void* host, std::size_t bytes) {
  return cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}
bool download_bytes(void* host, const void* device, std::size_t bytes) {
  return cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool sync_ok() {
  return cudaStreamSynchronize(nullptr) == cudaSuccess;
}

void upload_field(const ctip::TipField& device, const HostField& host) {
  const std::size_t cells =
      static_cast<std::size_t>(host.w8) * static_cast<std::size_t>(host.h8);
  check(upload_bytes(device.mv, host.mv.data(), cells * 2 * sizeof(std::int32_t)),
        "upload field mv");
  check(upload_bytes(device.valid, host.valid.data(), cells * sizeof(std::uint8_t)),
        "upload field valid");
  check(upload_bytes(device.offset, host.offset.data(),
                     cells * sizeof(std::int32_t)),
        "upload field offset");
}

void download_field(HostField* host, const ctip::TipField& device) {
  const std::size_t cells =
      static_cast<std::size_t>(host->w8) * static_cast<std::size_t>(host->h8);
  check(download_bytes(host->mv.data(), device.mv,
                       cells * 2 * sizeof(std::int32_t)),
        "download field mv");
  check(download_bytes(host->valid.data(), device.valid,
                       cells * sizeof(std::uint8_t)),
        "download field valid");
  check(download_bytes(host->offset.data(), device.offset,
                       cells * sizeof(std::int32_t)),
        "download field offset");
}

// CPU build_tip_plan() + comparison against the downloaded GPU plan.
void compare_plan_with_cpu(const FieldCase& c, HostField& field,
                           const std::vector<ctip::TipPlanBlock>& gpu_blocks,
                           const std::vector<ctip::Mv>& gpu_grid0,
                           const std::vector<ctip::Mv>& gpu_grid1,
                           int gpu_weight, const ctip::TipOffsets& gpu_offsets,
                           int gpu_num_blocks) {
  const int block_cols = (c.mi_cols + c.bw4 - 1) / c.bw4;
  const int block_rows = (c.mi_rows + c.bw4 - 1) / c.bw4;
  const std::size_t count =
      static_cast<std::size_t>(block_cols) * block_rows;
  const std::size_t grid = static_cast<std::size_t>(c.mi_cols) * c.mi_rows;
  std::vector<temporal::TipBlock> cpu_blocks(count);
  std::vector<temporal::Mv> cpu_grid0(grid);
  std::vector<temporal::Mv> cpu_grid1(grid);
  temporal::TipPlan plan;
  plan.num_blocks = static_cast<int>(count);
  plan.blocks = cpu_blocks.data();
  plan.mv_grid = cpu_grid0.data();
  plan.mv_grid1 = cpu_grid1.data();
  const temporal::MotionFieldConfig cc = c.cpu_config();
  temporal::TipConfig tip = c.cpu_tip();
  tip.enable_tip_refinemv = c.bw4 == 2;
  const temporal::MotionField cf = field.cpu();
  const temporal::TemporalStatus status =
      temporal::build_tip_plan(cc, tip, cf, &plan);
  check(status == temporal::TemporalStatus::kOk, "GPU-case CPU plan status");
  check(gpu_num_blocks == plan.num_blocks, "GPU plan block count");
  check(gpu_weight == plan.weight, "GPU plan weight");
  check(gpu_offsets.ref_offset == plan.offsets.ref_offset &&
            gpu_offsets.past_offset == plan.offsets.past_offset &&
            gpu_offsets.future_offset == plan.offsets.future_offset,
        "GPU plan offsets");
  bool blocks_ok = true;
  for (int i = 0; i < plan.num_blocks; ++i) {
    const temporal::TipBlock& a = cpu_blocks[static_cast<std::size_t>(i)];
    const ctip::TipPlanBlock& b = gpu_blocks[static_cast<std::size_t>(i)];
    if (a.mi_row != b.mi_row || a.mi_col != b.mi_col || a.bw4 != b.bw4 ||
        a.bh4 != b.bh4 || a.mv[0].row != b.mv[0].row ||
        a.mv[0].col != b.mv[0].col || a.mv[1].row != b.mv[1].row ||
        a.mv[1].col != b.mv[1].col ||
        (a.motion_field_valid ? 1 : 0) != b.motion_field_valid) {
      blocks_ok = false;
      break;
    }
  }
  check(blocks_ok, "GPU plan blocks");
  bool grid_ok = true;
  for (std::size_t i = 0; i < grid; ++i) {
    if (cpu_grid0[i].row != gpu_grid0[i].row ||
        cpu_grid0[i].col != gpu_grid0[i].col ||
        cpu_grid1[i].row != gpu_grid1[i].row ||
        cpu_grid1[i].col != gpu_grid1[i].col) {
      grid_ok = false;
      break;
    }
  }
  check(grid_ok, "GPU plan mv grids");
}

// ---------------------------------------------------------------------------
// Device-backed field/plan/prediction case
// ---------------------------------------------------------------------------

void run_field_case(Arena& arena, int variant) {
  const FieldCase c = make_field_case(variant);
  const int w8 = c.mi_cols >> 1;
  const int h8 = c.mi_rows >> 1;
  HostField initial;
  initial.init(w8, h8);
  randomize_field(&initial,
                  variant % 3 == 0 ? 30 : (variant % 3 == 1 ? 65 : 100));

  HostField cpu_field = initial;
  HostScratch cpu_scratch;
  cpu_scratch.init(static_cast<std::size_t>(w8) * h8);
  const temporal::MotionFieldConfig cc = c.cpu_config();
  const temporal::TipConfig tip = c.cpu_tip();
  temporal::MotionField cf = cpu_field.cpu();
  temporal::MotionFieldScratch cs = cpu_scratch.cpu();
  check(temporal::setup_tip_motion_field(cc, tip, &cf, nullptr, &cs, nullptr) ==
            temporal::TemporalStatus::kOk,
        "CPU field setup status");

  const std::uint64_t mark = ctip::tip_frame_mark(arena);
  ctip::TipWorkspaceConfig wc;
  wc.mi_cols = c.mi_cols;
  wc.mi_rows = c.mi_rows;
  wc.bw4 = c.bw4;
  wc.bh4 = c.bw4;
  wc.include_plan = 1;
  ctip::TipWorkspace ws;
  std::size_t required = 0;
  const Status init_status = ctip::tip_workspace_init(arena, wc, &ws, &required);
  check(init_status == Status::kOk, "workspace init");
  check(required == ctip::required_device_bytes(wc), "workspace required size");
  check(ws.device_bytes == required, "workspace arena bytes");
  if (init_status != Status::kOk) {
    return;
  }

  upload_field(ws.field, initial);
  check(ctip::tip_setup_motion_field(ws.field, c.device_config(), ws.scratch,
                                     c.hole_fill, nullptr) == cudaSuccess,
        "GPU setup launch");
  check(sync_ok(), "GPU setup sync");
  HostField gpu_field;
  gpu_field.init(w8, h8);
  download_field(&gpu_field, ws.field);
  check(same_field(gpu_field, cpu_field), "GPU field setup parity");
  check(cdev::tip_offsets(c.device_config()).ref_offset ==
            temporal::get_tip_offsets(tip).ref_offset,
        "GPU ref offset parity");

  ctip::TipPlanConfig pc;
  pc.bw4 = c.bw4;
  pc.bh4 = c.bw4;
  pc.tip_global_wtd_index = c.weight_index;
  pc.tip_global_mv = c.global_mv;
  ctip::TipPlan plan = ws.plan;
  check(ctip::tip_build_plan(ws.field, c.device_config(), pc, &plan, nullptr) ==
            cudaSuccess,
        "GPU plan launch");
  check(sync_ok(), "GPU plan sync");
  const int block_cols = (c.mi_cols + c.bw4 - 1) / c.bw4;
  const int block_rows = (c.mi_rows + c.bw4 - 1) / c.bw4;
  const std::size_t block_count_size =
      static_cast<std::size_t>(block_cols) * block_rows;
  const std::size_t grid = static_cast<std::size_t>(c.mi_cols) * c.mi_rows;
  std::vector<ctip::TipPlanBlock> gpu_blocks(block_count_size);
  std::vector<ctip::Mv> gpu_grid0(grid);
  std::vector<ctip::Mv> gpu_grid1(grid);
  check(download_bytes(gpu_blocks.data(), plan.blocks,
                       block_count_size * sizeof(ctip::TipPlanBlock)),
        "download plan blocks");
  check(download_bytes(gpu_grid0.data(), plan.mv_grid,
                       grid * sizeof(ctip::Mv)),
        "download grid0");
  check(download_bytes(gpu_grid1.data(), plan.mv_grid1,
                       grid * sizeof(ctip::Mv)),
        "download grid1");
  compare_plan_with_cpu(c, cpu_field, gpu_blocks, gpu_grid0, gpu_grid1,
                        plan.weight, plan.offsets, plan.num_blocks);
  check(ctip::tip_frame_rewind(arena, mark) == Status::kOk, "frame rewind");
  check(arena.used() == mark, "no net arena growth");
}

// ---------------------------------------------------------------------------
// Prediction cases
// ---------------------------------------------------------------------------

template <typename Sample>
void upload_plane(Sample* device, const std::vector<Sample>& host) {
  check(upload_bytes(device, host.data(), host.size() * sizeof(Sample)),
        "upload ref plane");
}

template <typename Sample>
void run_gpu_predict_case(Arena& arena, int variant, int bit_depth) {
  const PredictCase c = make_predict_case(variant, bit_depth);
  RefPlane ref[2];
  if (bit_depth == 8) {
    fill_ref_u8(&ref[0], c, 17);
    fill_ref_u8(&ref[1], c, 91);
  } else {
    fill_ref_u16(&ref[0], c, 23);
    fill_ref_u16(&ref[1], c, 77);
  }

  // Expected output.
  std::vector<Sample> want(static_cast<std::size_t>(c.w) * c.h);
  ctip::TipPredictJob host_job{};
  host_job.ref[0].data = ref[0].data<Sample>();
  host_job.ref[0].stride = ref[0].stride;
  host_job.ref[0].width = ref[0].width;
  host_job.ref[0].height = ref[0].height;
  host_job.ref[1].data = ref[1].data<Sample>();
  host_job.ref[1].stride = ref[1].stride;
  host_job.ref[1].width = ref[1].width;
  host_job.ref[1].height = ref[1].height;
  host_job.dst = want.data();
  host_job.dst_stride = c.w;
  host_job.x = c.x;
  host_job.y = c.y;
  host_job.w = c.w;
  host_job.h = c.h;
  host_job.sub_x = c.sub_x;
  host_job.sub_y = c.sub_y;
  host_job.frame_w = c.frame_w;
  host_job.frame_h = c.frame_h;
  host_job.mv[0] = c.mv[0];
  host_job.mv[1] = c.mv[1];
  host_job.num_refs = c.num_refs;
  host_job.cwp_weight = c.weight;
  host_job.filter = c.filter;
  host_job.bit_depth = bit_depth;
  host_job.sample_bytes = bit_depth > 8 ? 2 : 1;
  host_job.use_ref_area = c.use_area;
  for (int list = 0; list < 2; ++list) {
    host_job.area_first_x[list] = c.area[list][0];
    host_job.area_first_y[list] = c.area[list][1];
    host_job.area_last_x[list] = c.area[list][2];
    host_job.area_last_y[list] = c.area[list][3];
  }
  if (bit_depth == 8) {
    std::vector<std::uint8_t> cpu_dst(static_cast<std::size_t>(c.w) * c.h);
    temporal::PlaneView pv[2];
    pv[0].data = ref[0].u8.data();
    pv[0].stride = ref[0].stride;
    pv[0].width = ref[0].width;
    pv[0].height = ref[0].height;
    pv[1].data = ref[1].u8.data();
    pv[1].stride = ref[1].stride;
    pv[1].width = ref[1].width;
    pv[1].height = ref[1].height;
    std::vector<std::int16_t> scratch(2 * 16 * 16, 0);
    temporal::TipPredictInput in;
    in.ref[0] = pv[0];
    in.ref[1] = pv[1];
    in.num_refs = c.num_refs;
    in.x = c.x;
    in.y = c.y;
    in.w = c.w;
    in.h = c.h;
    in.sub_x = c.sub_x;
    in.sub_y = c.sub_y;
    in.frame_w = c.frame_w;
    in.frame_h = c.frame_h;
    in.mv[0] = c.mv[0];
    in.mv[1] = c.mv[1];
    in.cwp_weight = c.weight;
    in.filter = static_cast<temporal::InterpFilter>(c.filter);
    in.use_ref_area = c.use_area != 0;
    for (int list = 0; list < 2; ++list) {
      in.area[list] = temporal::RefArea{c.area[list][0], c.area[list][1],
                                        c.area[list][2], c.area[list][3]};
    }
    in.dst = cpu_dst.data();
    in.dst_stride = c.w;
    in.scratch0 = scratch.data();
    in.scratch1 = scratch.data() + 16 * 16;
    in.scratch_stride = c.w;
    check(temporal::tip_predict_block(in) == temporal::TemporalStatus::kOk,
          "CPU predict status");
    want.assign(cpu_dst.begin(), cpu_dst.end());
  } else {
    ref_predict_job(host_job, want.data());
  }

  const std::uint64_t mark = ctip::tip_frame_mark(arena);
  Sample* d_ref0 = arena.allocate_array<Sample>(
      static_cast<std::size_t>(ref[0].stride) * ref[0].height);
  Sample* d_ref1 = arena.allocate_array<Sample>(
      static_cast<std::size_t>(ref[1].stride) * ref[1].height);
  Sample* d_dst = arena.allocate_array<Sample>(
      static_cast<std::size_t>(c.w) * c.h);
  ctip::TipPredictJob* d_job = arena.allocate_array<ctip::TipPredictJob>(1);
  check(d_ref0 != nullptr && d_ref1 != nullptr && d_dst != nullptr &&
            d_job != nullptr,
        "predict arena allocation");
  if (d_ref0 == nullptr || d_ref1 == nullptr || d_dst == nullptr ||
      d_job == nullptr) {
    (void)ctip::tip_frame_rewind(arena, mark);
    return;
  }

  ctip::TipPredictJob job = host_job;
  job.ref[0].data = d_ref0;
  job.ref[1].data = d_ref1;
  job.dst = d_dst;
  check(upload_bytes(d_ref0, ref[0].data<Sample>(),
                     static_cast<std::size_t>(ref[0].stride) * ref[0].height *
                         sizeof(Sample)),
        "upload ref0");
  check(upload_bytes(d_ref1, ref[1].data<Sample>(),
                     static_cast<std::size_t>(ref[1].stride) * ref[1].height *
                         sizeof(Sample)),
        "upload ref1");
  check(upload_bytes(d_job, &job, sizeof(job)), "upload job");
  check(ctip::tip_predict_batch(d_job, 1, job.sample_bytes, nullptr) ==
            cudaSuccess,
        "GPU predict batch launch");
  check(sync_ok(), "GPU predict sync");
  std::vector<Sample> got(static_cast<std::size_t>(c.w) * c.h);
  check(download_bytes(got.data(), d_dst, got.size() * sizeof(Sample)),
        "download prediction");
  check(got == want, "GPU prediction parity");

  // Single-job entry point uses a second destination.
  Sample* d_dst2 = arena.allocate_array<Sample>(
      static_cast<std::size_t>(c.w) * c.h);
  check(d_dst2 != nullptr, "predict single allocation");
  if (d_dst2 != nullptr) {
    job.dst = d_dst2;
    check(ctip::tip_predict_block(job, nullptr) == cudaSuccess,
          "GPU predict block launch");
    check(sync_ok(), "GPU predict block sync");
    std::vector<Sample> got2(static_cast<std::size_t>(c.w) * c.h);
    check(download_bytes(got2.data(), d_dst2, got2.size() * sizeof(Sample)),
          "download single prediction");
    check(got2 == want, "GPU single prediction parity");
  }
  check(ctip::tip_frame_rewind(arena, mark) == Status::kOk, "predict rewind");
  check(arena.used() == mark, "predict no net growth");
}

// ---------------------------------------------------------------------------
// Full clip
// ---------------------------------------------------------------------------

struct ClipRefs {
  std::vector<std::uint8_t> u8[3][2];
  std::vector<std::uint16_t> u16[3][2];
  temporal::PlaneView view8[3][2];
  int width[3] = {0, 0, 0};
  int height[3] = {0, 0, 0};
  int stride[3] = {0, 0, 0};
};

void make_clip(uint32_t seed, int mi_cols, int mi_rows, int bit_depth,
               ClipRefs* refs) {
  g_rng = seed;
  const int luma_w = mi_cols * 4;
  const int luma_h = mi_rows * 4;
  for (int plane = 0; plane < 3; ++plane) {
    const int sub = plane == 0 ? 0 : 1;
    const int w = luma_w >> sub;
    const int h = luma_h >> sub;
    refs->width[plane] = w;
    refs->height[plane] = h;
    refs->stride[plane] = w + (plane == 0 ? 0 : 4);
    for (int list = 0; list < 2; ++list) {
      const std::size_t count =
          static_cast<std::size_t>(refs->stride[plane]) * h;
      refs->u8[plane][list].resize(count);
      refs->u16[plane][list].resize(count);
      for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t v = next_random();
        refs->u8[plane][list][i] = static_cast<std::uint8_t>(v + list * 37u);
        refs->u16[plane][list][i] =
            static_cast<std::uint16_t>(((v >> 3) + list * 111u) % 1024u);
      }
      refs->view8[plane][list].data = refs->u8[plane][list].data();
      refs->view8[plane][list].stride = refs->stride[plane];
      refs->view8[plane][list].width = w;
      refs->view8[plane][list].height = h;
    }
  }
  (void)bit_depth;
}

void cpu_predict_plan_u8(const temporal::TipPlan& plan,
                         const temporal::TipConfig& tip,
                         const temporal::MotionFieldConfig& config,
                         const temporal::PlaneView ref[3][2],
                         std::vector<std::uint8_t> dst[3],
                         const int dst_stride[3], const int dst_width[3],
                         const int dst_height[3], const int sub[3]) {
  std::vector<std::int16_t> scratch(2 * 16 * 16, 0);
  const int frame_w = config.mi_cols * 4;
  const int frame_h = config.mi_rows * 4;
  for (int b = 0; b < plan.num_blocks; ++b) {
    const temporal::TipBlock& block = plan.blocks[b];
    for (int plane = 0; plane < 3; ++plane) {
      const int w = (block.bw4 * 4) >> sub[plane];
      const int h = (block.bh4 * 4) >> sub[plane];
      const int x = (block.mi_col * 4) >> sub[plane];
      const int y = (block.mi_row * 4) >> sub[plane];
      const int vis_w = std::min(w, dst_width[plane] - x);
      const int vis_h = std::min(h, dst_height[plane] - y);
      if (vis_w <= 0 || vis_h <= 0) continue;
      temporal::TipPredictInput in;
      in.ref[0] = ref[plane][0];
      in.ref[1] = ref[plane][1];
      in.num_refs = 2;
      in.x = x;
      in.y = y;
      in.w = vis_w;
      in.h = vis_h;
      in.sub_x = sub[plane];
      in.sub_y = sub[plane];
      in.frame_w = frame_w;
      in.frame_h = frame_h;
      in.mv[0] = block.mv[0];
      in.mv[1] = block.mv[1];
      in.cwp_weight = plan.weight;
      in.filter = tip.interp_filter;
      in.use_ref_area = false;
      in.dst = dst[plane].data() +
               static_cast<std::ptrdiff_t>(y) * dst_stride[plane] + x;
      in.dst_stride = dst_stride[plane];
      in.scratch0 = scratch.data();
      in.scratch1 = scratch.data() + 16 * 16;
      in.scratch_stride = w;
      (void)temporal::tip_predict_block(in);
    }
  }
}

void cpu_predict_plan_u16(const temporal::TipPlan& plan,
                          const temporal::TipConfig& tip,
                          const temporal::MotionFieldConfig& config,
                          const ClipRefs& refs,
                          std::vector<std::uint16_t> dst[3],
                          const int dst_stride[3], const int dst_width[3],
                          const int dst_height[3], const int sub[3]) {
  const int frame_w = config.mi_cols * 4;
  const int frame_h = config.mi_rows * 4;
  for (int b = 0; b < plan.num_blocks; ++b) {
    const temporal::TipBlock& block = plan.blocks[b];
    for (int plane = 0; plane < 3; ++plane) {
      const int w = (block.bw4 * 4) >> sub[plane];
      const int h = (block.bh4 * 4) >> sub[plane];
      const int x = (block.mi_col * 4) >> sub[plane];
      const int y = (block.mi_row * 4) >> sub[plane];
      const int vis_w = std::min(w, dst_width[plane] - x);
      const int vis_h = std::min(h, dst_height[plane] - y);
      if (vis_w <= 0 || vis_h <= 0) continue;
      ctip::TipPredictJob job{};
      job.ref[0].data = refs.u16[plane][0].data();
      job.ref[0].stride = refs.stride[plane];
      job.ref[0].width = refs.width[plane];
      job.ref[0].height = refs.height[plane];
      job.ref[1].data = refs.u16[plane][1].data();
      job.ref[1].stride = refs.stride[plane];
      job.ref[1].width = refs.width[plane];
      job.ref[1].height = refs.height[plane];
      job.dst = dst[plane].data() +
                static_cast<std::ptrdiff_t>(y) * dst_stride[plane] + x;
      job.dst_stride = dst_stride[plane];
      job.x = x;
      job.y = y;
      job.w = vis_w;
      job.h = vis_h;
      job.sub_x = sub[plane];
      job.sub_y = sub[plane];
      job.frame_w = frame_w;
      job.frame_h = frame_h;
      job.mv[0] = block.mv[0];
      job.mv[1] = block.mv[1];
      job.num_refs = 2;
      job.cwp_weight = plan.weight;
      job.filter = static_cast<int>(tip.interp_filter);
      job.bit_depth = 10;
      job.sample_bytes = 2;
      job.use_ref_area = 0;
      ref_predict_job(job, static_cast<std::uint16_t*>(job.dst));
    }
  }
}

// Builds a §7.9 field with the CPU reference so the GPU starts from the same
// post-estimation array a real frame would have.
HostField make_estimated_field(const temporal::MotionFieldConfig& config,
                               int seed) {
  g_rng = static_cast<std::uint32_t>(seed);
  const int w8 = config.mi_cols >> 1;
  const int h8 = config.mi_rows >> 1;
  const std::size_t cells = static_cast<std::size_t>(w8) * h8;
  temporal::RefMvField saved;
  saved.w8 = w8;
  saved.h8 = h8;
  std::vector<std::int8_t> saved_ref(cells * 2, 0);
  std::vector<std::int16_t> saved_mv(cells * 4, 0);
  saved.ref = saved_ref.data();
  saved.mv = saved_mv.data();
  for (std::size_t i = 0; i < cells; ++i) {
    saved_ref[i * 2 + 0] = 0;
    saved_ref[i * 2 + 1] = temporal::kRefNone;
    saved_mv[(i * 2 + 0) * 2 + 0] =
        static_cast<std::int16_t>(random_range(-900, 900));
    saved_mv[(i * 2 + 0) * 2 + 1] =
        static_cast<std::int16_t>(random_range(-900, 900));
  }
  temporal::MotionFieldConfig cfg = config;
  cfg.saved[1] = temporal::SavedMvField{};
  cfg.saved[1].w8 = w8;
  cfg.saved[1].h8 = h8;
  cfg.saved[1].mv = saved.mv;
  cfg.saved[1].ref = saved.ref;

  HostField field;
  field.init(w8, h8);
  temporal::MotionField mf = field.cpu();
  temporal::MotionFieldResult result;
  const temporal::TemporalStatus status =
      temporal::motion_field_estimation(cfg, &mf, nullptr, &result);
  check(status == temporal::TemporalStatus::kOk, "clip estimation status");
  return field;
}

void run_clip_case(Arena& arena, int variant, int bit_depth, int bw4,
                   int weight_index) {
  const int mi_cols = 40;
  const int mi_rows = 30;  // 160x120 luma, odd MI row count
  ClipRefs refs;
  make_clip(0x51a7u + static_cast<std::uint32_t>(variant), mi_cols, mi_rows,
            bit_depth, &refs);

  temporal::MotionFieldConfig config;
  config.mi_cols = mi_cols;
  config.mi_rows = mi_rows;
  config.proj_step = 1;
  config.sb_64x64 = true;
  config.enable_tip = true;
  config.order_hint = 1;
  config.num_total_refs = 2;
  config.num_past_refs = 1;
  config.num_future_refs = 1;
  config.frame_width = mi_cols * 4;
  config.frame_height = mi_rows * 4;
  config.ref_order_hint[0] = 0;
  config.ref_order_hint[1] = 4;
  config.refs[0] = temporal::RefFrameInfo{};
  config.refs[0].slot = 0;
  config.refs[0].type = temporal::FrameType::kKey;
  config.refs[0].width = config.frame_width;
  config.refs[0].height = config.frame_height;
  config.refs[0].order_hint = 0;
  config.refs[0].num_saved_refs = 0;
  config.refs[1] = temporal::RefFrameInfo{};
  config.refs[1].slot = 1;
  config.refs[1].type = temporal::FrameType::kInter;
  config.refs[1].width = config.frame_width;
  config.refs[1].height = config.frame_height;
  config.refs[1].order_hint = 4;
  config.refs[1].num_saved_refs = 1;
  config.refs[1].saved_order_hint[0] = 0;

  HostField estimated = make_estimated_field(config, 1000 + variant);
  // Sprinkle holes so hole fill and the average filter both run.
  const std::size_t cells =
      static_cast<std::size_t>(estimated.w8) * estimated.h8;
  for (std::size_t i = 0; i < cells; ++i) {
    if (chance(12)) estimated.valid[i] = 0;
  }

  temporal::TipConfig tip;
  tip.mode = temporal::TipFrameMode::kAsOutput;
  tip.tip_global_wtd_index = weight_index;
  tip.tip_global_mv = temporal::Mv{random_range(-32, 32),
                                    random_range(-32, 32)};
  tip.interp_filter = temporal::InterpFilter::kEighttapSharp;
  tip.enable_tip_refinemv = bw4 == 2;
  tip.allow_tip_hole_fill = true;
  tip.num_past_refs = 1;
  tip.num_future_refs = 1;
  tip.order_hint = 1;
  tip.closest_past_order_hint = 0;
  tip.closest_future_order_hint = 4;

  // CPU: setup -> plan -> predict.
  HostField cpu_field = estimated;
  HostScratch cpu_scratch;
  cpu_scratch.init(cells);
  temporal::MotionField cpu_mf = cpu_field.cpu();
  temporal::MotionFieldScratch cpu_ms = cpu_scratch.cpu();
  check(temporal::setup_tip_motion_field(config, tip, &cpu_mf, nullptr, &cpu_ms,
                                         nullptr) == temporal::TemporalStatus::kOk,
        "clip CPU setup");
  const int block_cols = (mi_cols + bw4 - 1) / bw4;
  const int block_rows = (mi_rows + bw4 - 1) / bw4;
  const std::size_t block_count =
      static_cast<std::size_t>(block_cols) * block_rows;
  const std::size_t grid_cells = static_cast<std::size_t>(mi_cols) * mi_rows;
  std::vector<temporal::TipBlock> cpu_blocks(block_count);
  std::vector<temporal::Mv> cpu_grid0(grid_cells);
  std::vector<temporal::Mv> cpu_grid1(grid_cells);
  temporal::TipPlan cpu_plan;
  cpu_plan.num_blocks = static_cast<int>(block_count);
  cpu_plan.blocks = cpu_blocks.data();
  cpu_plan.mv_grid = cpu_grid0.data();
  cpu_plan.mv_grid1 = cpu_grid1.data();
  check(temporal::build_tip_plan(config, tip, cpu_mf, &cpu_plan) ==
            temporal::TemporalStatus::kOk,
        "clip CPU plan");

  int sub[3] = {0, 1, 1};
  int dst_stride[3];
  int dst_width[3];
  int dst_height[3];
  for (int plane = 0; plane < 3; ++plane) {
    dst_width[plane] = refs.width[plane];
    dst_height[plane] = refs.height[plane];
    dst_stride[plane] = refs.width[plane] + 8;
  }
  std::vector<std::uint8_t> cpu_u8[3];
  std::vector<std::uint16_t> cpu_u16[3];
  for (int plane = 0; plane < 3; ++plane) {
    if (bit_depth == 8) {
      cpu_u8[plane].assign(
          static_cast<std::size_t>(dst_stride[plane]) * dst_height[plane], 0);
    } else {
      cpu_u16[plane].assign(
          static_cast<std::size_t>(dst_stride[plane]) * dst_height[plane], 0);
    }
  }
  if (bit_depth == 8) {
    cpu_predict_plan_u8(cpu_plan, tip, config, refs.view8, cpu_u8, dst_stride,
                        dst_width, dst_height, sub);
  } else {
    cpu_predict_plan_u16(cpu_plan, tip, config, refs, cpu_u16, dst_stride,
                         dst_width, dst_height, sub);
  }

  // GPU: upload, setup, plan, predict.
  const std::uint64_t mark = ctip::tip_frame_mark(arena);
  ctip::TipWorkspaceConfig wc;
  wc.mi_cols = mi_cols;
  wc.mi_rows = mi_rows;
  wc.bw4 = bw4;
  wc.bh4 = bw4;
  wc.include_plan = 1;
  ctip::TipWorkspace ws;
  std::size_t required = 0;
  check(ctip::tip_workspace_init(arena, wc, &ws, &required) == Status::kOk,
        "clip workspace init");
  upload_field(ws.field, estimated);
  ctip::TipMotionFieldConfig mc;
  mc.mi_cols = mi_cols;
  mc.mi_rows = mi_rows;
  mc.proj_step = 1;
  mc.sb_64x64 = 1;
  mc.enable_tip = 1;
  mc.order_hint = 1;
  mc.num_past_refs = 1;
  mc.num_future_refs = 1;
  mc.closest_past_order_hint = 0;
  mc.closest_future_order_hint = 4;
  check(ctip::tip_setup_motion_field(ws.field, mc, ws.scratch, 1, nullptr) ==
            cudaSuccess,
        "clip GPU setup launch");
  check(sync_ok(), "clip setup sync");
  HostField gpu_field;
  gpu_field.init(estimated.w8, estimated.h8);
  download_field(&gpu_field, ws.field);
  check(same_field(gpu_field, cpu_field), "clip field parity");

  ctip::TipPlanConfig pc;
  pc.bw4 = bw4;
  pc.bh4 = bw4;
  pc.tip_global_wtd_index = weight_index;
  pc.tip_global_mv = tip.tip_global_mv;
  ctip::TipPlan gpu_plan = ws.plan;
  check(ctip::tip_build_plan(ws.field, mc, pc, &gpu_plan, nullptr) == cudaSuccess,
        "clip GPU plan launch");
  check(sync_ok(), "clip plan sync");
  check(gpu_plan.num_blocks == cpu_plan.num_blocks, "clip block count");
  check(gpu_plan.weight == cpu_plan.weight, "clip plan weight");
  std::vector<ctip::TipPlanBlock> gpu_blocks(block_count);
  std::vector<ctip::Mv> gpu_grid0(grid_cells);
  std::vector<ctip::Mv> gpu_grid1(grid_cells);
  check(download_bytes(gpu_blocks.data(), gpu_plan.blocks,
                       block_count * sizeof(ctip::TipPlanBlock)),
        "clip download blocks");
  check(download_bytes(gpu_grid0.data(), gpu_plan.mv_grid,
                       grid_cells * sizeof(ctip::Mv)),
        "clip download grid0");
  check(download_bytes(gpu_grid1.data(), gpu_plan.mv_grid1,
                       grid_cells * sizeof(ctip::Mv)),
        "clip download grid1");
  bool blocks_ok = true;
  for (std::size_t i = 0; i < block_count; ++i) {
    const temporal::TipBlock& a = cpu_blocks[i];
    const ctip::TipPlanBlock& b = gpu_blocks[i];
    if (a.mi_row != b.mi_row || a.mi_col != b.mi_col || a.bw4 != b.bw4 ||
        a.bh4 != b.bh4 || a.mv[0].row != b.mv[0].row ||
        a.mv[0].col != b.mv[0].col || a.mv[1].row != b.mv[1].row ||
        a.mv[1].col != b.mv[1].col) {
      blocks_ok = false;
      break;
    }
  }
  check(blocks_ok, "clip plan blocks");
  bool grid_ok = true;
  for (std::size_t i = 0; i < grid_cells; ++i) {
    if (cpu_grid0[i].row != gpu_grid0[i].row ||
        cpu_grid0[i].col != gpu_grid0[i].col ||
        cpu_grid1[i].row != gpu_grid1[i].row ||
        cpu_grid1[i].col != gpu_grid1[i].col) {
      grid_ok = false;
      break;
    }
  }
  check(grid_ok, "clip mv grids");

  // Predict all blocks of the plan in one launch.
  ctip::TipPlanPredictConfig cfg{};
  void* d_plane[3][2] = {{nullptr, nullptr}, {nullptr, nullptr},
                         {nullptr, nullptr}};
  std::size_t plane_bytes[3] = {0, 0, 0};
  for (int plane = 0; plane < 3; ++plane) {
    plane_bytes[plane] = static_cast<std::size_t>(refs.stride[plane]) *
                         refs.height[plane] * (bit_depth == 8 ? 1 : 2);
    for (int list = 0; list < 2; ++list) {
      d_plane[plane][list] = arena.allocate(plane_bytes[plane]);
      check(d_plane[plane][list] != nullptr, "clip ref allocation");
      const void* src = bit_depth == 8
                            ? static_cast<const void*>(refs.u8[plane][list].data())
                            : static_cast<const void*>(refs.u16[plane][list].data());
      check(upload_bytes(d_plane[plane][list], src, plane_bytes[plane]),
            "clip upload ref");
      cfg.ref[plane][list].data = d_plane[plane][list];
      cfg.ref[plane][list].stride = refs.stride[plane];
      cfg.ref[plane][list].width = refs.width[plane];
      cfg.ref[plane][list].height = refs.height[plane];
    }
    cfg.dst[plane] = arena.allocate(
        static_cast<std::size_t>(dst_stride[plane]) * dst_height[plane] *
        (bit_depth == 8 ? 1 : 2));
    check(cfg.dst[plane] != nullptr, "clip dst allocation");
    cfg.dst_stride[plane] = dst_stride[plane];
    cfg.dst_width[plane] = dst_width[plane];
    cfg.dst_height[plane] = dst_height[plane];
    cfg.sub_x[plane] = sub[plane];
    cfg.sub_y[plane] = sub[plane];
  }
  cfg.frame_w = mi_cols * 4;
  cfg.frame_h = mi_rows * 4;
  cfg.bit_depth = bit_depth;
  cfg.sample_bytes = bit_depth == 8 ? 1 : 2;
  cfg.cwp_weight = cpu_plan.weight;
  cfg.filter = static_cast<int>(tip.interp_filter);
  check(ctip::tip_predict_plan(gpu_plan, cfg, nullptr) == cudaSuccess,
        "clip predict launch");
  check(sync_ok(), "clip predict sync");

  // Only the visible rectangle is part of the contract; the 8-sample row
  // padding stays untouched in both the CPU and GPU buffers.
  bool planes_ok = true;
  for (int plane = 0; plane < 3 && planes_ok; ++plane) {
    const std::size_t count =
        static_cast<std::size_t>(dst_stride[plane]) * dst_height[plane];
    if (bit_depth == 8) {
      std::vector<std::uint8_t> got(count, 0);
      check(download_bytes(got.data(), cfg.dst[plane], count),
            "clip download plane");
      for (int r = 0; r < dst_height[plane] && planes_ok; ++r) {
        for (int c = 0; c < dst_width[plane]; ++c) {
          const std::size_t i = static_cast<std::size_t>(r) * dst_stride[plane] + c;
          if (got[i] != cpu_u8[plane][i]) {
            std::printf("  plane %d (%d,%d) got %u want %u\n", plane, r, c,
                        static_cast<unsigned>(got[i]),
                        static_cast<unsigned>(cpu_u8[plane][i]));
            planes_ok = false;
            break;
          }
        }
      }
    } else {
      std::vector<std::uint16_t> got(count, 0);
      check(download_bytes(got.data(), cfg.dst[plane], count * 2),
            "clip download plane16");
      for (int r = 0; r < dst_height[plane] && planes_ok; ++r) {
        for (int c = 0; c < dst_width[plane]; ++c) {
          const std::size_t i = static_cast<std::size_t>(r) * dst_stride[plane] + c;
          if (got[i] != cpu_u16[plane][i]) {
            std::printf("  plane %d (%d,%d) got %u want %u\n", plane, r, c,
                        static_cast<unsigned>(got[i]),
                        static_cast<unsigned>(cpu_u16[plane][i]));
            planes_ok = false;
            break;
          }
        }
      }
    }
  }
  check(planes_ok, "full-clip plane parity");

  check(ctip::tip_frame_rewind(arena, mark) == Status::kOk, "clip rewind");
  check(arena.used() == mark, "clip no net growth");
}

}  // namespace

int main() {
  std::printf("tip_test: AV2 TIP CUDA parity\n");
  check(cudaSetDevice(0) == cudaSuccess, "CUDA device");

  ctip::TipWorkspaceConfig max_config;
  max_config.mi_cols = 160;
  max_config.mi_rows = 120;
  max_config.bw4 = 2;
  max_config.bh4 = 2;
  max_config.include_plan = 1;
  max_config.block_capacity = (120 / 2) * (160 / 2);
  const std::size_t required = ctip::required_device_bytes(max_config);
  const std::size_t slack = 32u * 1024u * 1024u;
  std::printf("required_device_bytes(%dx%d) = %zu bytes\n", max_config.mi_cols,
              max_config.mi_rows, required);

  Arena arena;
  std::string error_text;
  Status status = arena.init(required + slack, Arena::kAlignment, &error_text);
  check(is_ok(status), "arena init");
  if (!is_ok(status)) {
    std::printf("arena init failed: %s\n", error_text.c_str());
    return 1;
  }

  // One-byte-short negative test: distinct status + required size, no change.
  {
    Arena short_arena;
    Status short_status =
        short_arena.init(required - 1, Arena::kAlignment, &error_text);
    check(is_ok(short_status), "short arena init");
    ctip::TipWorkspace short_ws;
    std::size_t short_required = 0;
    const Status alloc = ctip::tip_workspace_init(short_arena, max_config,
                                                  &short_ws, &short_required);
    check(alloc == Status::kArenaExhausted, "short arena distinct status");
    check(short_required == required, "short arena required size");
    check(short_ws.field.mv == nullptr, "short arena no partial pointers");
    check(short_arena.used() == 0, "short arena bump unchanged");
    short_arena.release();
  }

  // Persistent workspace: two frames with mark/rewind, no net growth.
  {
    ctip::TipWorkspace ws;
    std::size_t ws_bytes = 0;
    status = ctip::tip_workspace_init(arena, max_config, &ws, &ws_bytes);
    check(is_ok(status), "persistent workspace init");
    check(ws_bytes == required, "persistent workspace bytes");
    const std::uint64_t used0 = arena.used();
    const std::uint64_t peak0 = arena.peak_used();
    for (int frame = 0; frame < 2; ++frame) {
      const std::uint64_t mark = ctip::tip_frame_mark(arena);
      HostField f;
      f.init(max_config.mi_cols >> 1, max_config.mi_rows >> 1);
      randomize_field(&f, 60 + frame * 15);
      upload_field(ws.field, f);
      ctip::TipMotionFieldConfig mc;
      mc.mi_cols = max_config.mi_cols;
      mc.mi_rows = max_config.mi_rows;
      mc.proj_step = frame == 0 ? 1 : 2;
      mc.sb_64x64 = 1;
      mc.enable_tip = 1;
      mc.order_hint = 10;
      mc.num_past_refs = 1;
      mc.num_future_refs = 1;
      mc.closest_past_order_hint = 8;
      mc.closest_future_order_hint = 12;
      check(ctip::tip_setup_motion_field(ws.field, mc, ws.scratch, 1,
                                         nullptr) == cudaSuccess,
            "persistent setup launch");
      check(sync_ok(), "persistent setup sync");
      check(ctip::tip_frame_rewind(arena, mark) == Status::kOk,
            "persistent rewind");
      check(arena.used() == used0, "persistent no net growth");
    }
    std::printf("persistent workspace: %zu bytes, peak %llu bytes\n", ws_bytes,
                static_cast<unsigned long long>(arena.peak_used()));
    (void)peak0;
  }

  // Randomised field + plan cases through the kernels.
  for (int variant = 0; variant < 48; ++variant) {
    run_field_case(arena, variant);
  }

  // 8-bit/10-bit prediction cases (8x8 and 16x16, all weights/filters).
  for (int variant = 0; variant < 120; ++variant) {
    run_gpu_predict_case<std::uint8_t>(arena, variant, 8);
    run_gpu_predict_case<std::uint16_t>(arena, variant, 10);
  }

  // Full clip: 8x8 and 16x16, 8-bit and 10-bit, several weights
  // (index 2 -> Tip_Weighting_Factor 16 single-reference, index 7 -> -4).
  const int weight_indices[4] = {0, 1, 2, 7};
  int clip_variant = 0;
  for (int bit_depth : {8, 10}) {
    for (int bw4 : {2, 4}) {
      for (int index : weight_indices) {
        run_clip_case(arena, clip_variant++, bit_depth, bw4, index);
      }
    }
  }

  std::printf("tip_test: %d checks, %d failures\n", g_checks, g_failures);
  std::printf("arena: used %llu, peak %llu, capacity %llu bytes\n",
              static_cast<unsigned long long>(arena.used()),
              static_cast<unsigned long long>(arena.peak_used()),
              static_cast<unsigned long long>(arena.capacity()));
  arena.release();
  return g_failures == 0 ? 0 : 1;
}
