// Host-emulation parity test for the shared AV2 TIP device arithmetic.
//
// Compiles `tip_device.cuh` with CUDA keywords stubbed and compares:
//   * §7.10.2–.5 motion-field construction against
//     `inter::temporal::setup_tip_motion_field()`,
//   * §7.10.6 plan construction against `build_tip_plan()`,
//   * §7.13.3.2 prediction for 8-bit against `tip_predict_block()`,
//   * 10-bit prediction against a bit-depth-parameterised CPU transcription
//     (the CPU reference is 8-bit only).
//
// Runs without nvcc or a GPU.  `tip_test.cu` runs the same cases through the
// kernels and the arena.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#else
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static
#include <cuda_runtime.h>
#endif

#include "tip_device.cuh"

#include "codec/inter/temporal/motion_field.h"
#include "codec/inter/temporal/tip.h"

namespace {

namespace inter = av2enc::inter;
namespace ctip = av2enc::cuda::tip;
namespace cdev = av2enc::cuda::tip::dev;
namespace temporal = av2enc::inter::temporal;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("FAIL: %s\n", what);
  }
}

std::uint32_t g_rng = 0x7a12c9e5u;
std::uint32_t next_random() {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}
int random_range(int low, int high) {
  return low + static_cast<int>(next_random() %
                                static_cast<std::uint32_t>(high - low + 1));
}
bool chance(int percent) {
  return static_cast<int>(next_random() % 100u) < percent;
}
int random_mv_component() {
  if (chance(20)) {
    return random_range(-(1 << 16) + 1, (1 << 16) - 1);
  }
  return random_range(-4096, 4096);
}

struct HostField {
  int w8 = 0;
  int h8 = 0;
  std::vector<std::int32_t> mv;
  std::vector<std::uint8_t> valid;
  std::vector<std::int32_t> offset;

  void init(int w, int h) {
    w8 = w;
    h8 = h;
    const std::size_t cells = static_cast<std::size_t>(w) * h;
    mv.assign(cells * 2, 0);
    valid.assign(cells, 0);
    offset.assign(cells, 0);
  }
  ctip::TipField dev() {
    ctip::TipField f;
    f.w8 = w8;
    f.h8 = h8;
    f.mv = mv.data();
    f.valid = valid.data();
    f.offset = offset.data();
    return f;
  }
  temporal::MotionField cpu() {
    temporal::MotionField f;
    f.w8 = w8;
    f.h8 = h8;
    f.mv = mv.data();
    f.valid = valid.data();
    f.offset = offset.data();
    return f;
  }
};

struct HostScratch {
  std::vector<std::int32_t> avg_mv;
  std::vector<std::uint8_t> avg_valid;
  void init(std::size_t cells) {
    avg_mv.assign(cells * 2, 0);
    avg_valid.assign(cells, 0);
  }
  ctip::TipFieldScratch dev() {
    ctip::TipFieldScratch s;
    s.avg_mv = avg_mv.data();
    s.avg_valid = avg_valid.data();
    return s;
  }
  temporal::MotionFieldScratch cpu() {
    temporal::MotionFieldScratch s;
    s.avg_mv = avg_mv.data();
    s.avg_valid = avg_valid.data();
    return s;
  }
};

// ---------------------------------------------------------------------------
// Motion-field case model
// ---------------------------------------------------------------------------

struct FieldCase {
  int mi_cols = 0;
  int mi_rows = 0;
  int proj_step = 1;
  int sb_64x64 = 1;
  int hole_fill = 1;
  int order_hint = 0;
  int num_past_refs = 0;
  int num_future_refs = 0;
  int closest_past = -1;
  int closest_future = -1;
  int weight_index = 0;
  ctip::Mv global_mv{};
  int bw4 = 4;

  ctip::TipMotionFieldConfig device_config() const {
    ctip::TipMotionFieldConfig c;
    c.mi_cols = mi_cols;
    c.mi_rows = mi_rows;
    c.proj_step = proj_step;
    c.sb_64x64 = sb_64x64;
    c.enable_tip = 1;
    c.order_hint = order_hint;
    c.num_past_refs = num_past_refs;
    c.num_future_refs = num_future_refs;
    c.closest_past_order_hint = closest_past;
    c.closest_future_order_hint = closest_future;
    return c;
  }

  temporal::MotionFieldConfig cpu_config() const {
    temporal::MotionFieldConfig c;
    c.mi_cols = mi_cols;
    c.mi_rows = mi_rows;
    c.proj_step = proj_step;
    c.sb_64x64 = sb_64x64 != 0;
    c.enable_tip = true;
    c.order_hint = order_hint;
    c.num_past_refs = num_past_refs;
    c.num_future_refs = num_future_refs;
    return c;
  }

  temporal::TipConfig cpu_tip() const {
    temporal::TipConfig tip;
    tip.mode = temporal::TipFrameMode::kAsOutput;
    tip.tip_global_wtd_index = weight_index;
    tip.tip_global_mv = global_mv;
    tip.interp_filter = temporal::InterpFilter::kEighttapSharp;
    tip.enable_tip_refinemv = false;
    tip.allow_tip_hole_fill = hole_fill != 0;
    tip.num_past_refs = num_past_refs;
    tip.num_future_refs = num_future_refs;
    tip.order_hint = order_hint;
    tip.closest_past_order_hint = closest_past;
    tip.closest_future_order_hint = closest_future;
    return tip;
  }
};

FieldCase make_field_case(int variant) {
  static const int kDims[][2] = {{8, 8},   {16, 8},  {8, 16}, {20, 18},
                                 {32, 24}, {64, 36}, {40, 24}, {96, 60},
                                 {16, 14}, {12, 22}};
  const int dim = variant % 10;
  FieldCase c;
  c.mi_cols = kDims[dim][0];
  c.mi_rows = kDims[dim][1];
  c.proj_step = chance(50) ? 1 : 2;
  c.sb_64x64 = chance(50) ? 1 : 0;
  c.hole_fill = chance(65) ? 1 : 0;
  c.order_hint = random_range(4, 25);
  const int mode = random_range(0, 2);
  if (mode == 0) {
    c.num_past_refs = 1;
    c.num_future_refs = 1;
    c.closest_past = c.order_hint - random_range(1, 8);
    c.closest_future = c.order_hint + random_range(1, 8);
  } else if (mode == 1) {
    c.num_past_refs = 2;
    c.num_future_refs = 0;
    c.closest_past = c.order_hint - random_range(1, 8);
    c.closest_future = -1;
  } else {
    c.num_past_refs = 0;
    c.num_future_refs = 1;
    c.closest_past = -1;
    c.closest_future = c.order_hint + random_range(1, 8);
  }
  c.weight_index = random_range(0, 7);
  c.global_mv = ctip::Mv{random_range(-600, 600), random_range(-600, 600)};
  c.bw4 = chance(50) ? 2 : 4;
  return c;
}

void randomize_field(HostField* field, int density) {
  const std::size_t cells =
      static_cast<std::size_t>(field->w8) * field->h8;
  for (std::size_t i = 0; i < cells; ++i) {
    field->valid[i] = chance(density) ? 1 : 0;
    field->mv[i * 2 + 0] = random_mv_component();
    field->mv[i * 2 + 1] = random_mv_component();
    field->offset[i] = random_range(-31, 31);
  }
}

// Runs the device stages on the host, one reference cell at a time.
void setup_device_field(HostField* field, const FieldCase& c,
                        const HostScratch& scratch) {
  const ctip::TipField f = field->dev();
  const ctip::TipMotionFieldConfig cfg = c.device_config();
  const ctip::TipOffsets offsets = cdev::tip_offsets(cfg);
  for (int y8 = 0; y8 < f.h8; y8 += cfg.proj_step) {
    for (int x8 = 0; x8 < f.w8; x8 += cfg.proj_step) {
      cdev::scale_cell(f, offsets.ref_offset, y8, x8);
    }
  }
  if (c.hole_fill != 0) {
    const int sb8 = 1 << cdev::tmvp_shift(cfg.sb_64x64, cfg.proj_step);
    for (int y8 = 0; y8 < f.h8; y8 += sb8) {
      for (int x8 = 0; x8 < f.w8; x8 += sb8) {
        cdev::hole_fill_superblock(f, cfg, y8, x8);
      }
    }
    const ctip::TipFieldScratch s = const_cast<HostScratch&>(scratch).dev();
    for (int y8 = 0; y8 < f.h8; y8 += cfg.proj_step) {
      for (int x8 = 0; x8 < f.w8; x8 += cfg.proj_step) {
        cdev::block_average_cell(f, cfg, s, y8, x8);
      }
    }
    for (int y8 = 0; y8 < f.h8; y8 += cfg.proj_step) {
      for (int x8 = 0; x8 < f.w8; x8 += cfg.proj_step) {
        cdev::copy_average_cell(f, s, y8, x8);
      }
    }
  }
  if (cfg.proj_step == 2) {
    for (int y8 = 0; y8 < f.h8; y8 += 2) {
      for (int x8 = 0; x8 < f.w8; x8 += 2) {
        cdev::fill_sample_gap_cell(f, cfg, y8, x8, 0, 1);
        cdev::fill_sample_gap_cell(f, cfg, y8, x8, 1, 0);
        cdev::fill_sample_gap_cell(f, cfg, y8, x8, 1, 1);
      }
    }
  }
}

bool same_field(const HostField& a, const HostField& b) {
  if (a.w8 != b.w8 || a.h8 != b.h8) {
    return false;
  }
  std::size_t i = 0;
  for (; i < a.mv.size(); ++i) {
    if (a.mv[i] != b.mv[i]) break;
  }
  if (i < a.mv.size()) {
    std::printf("  mv[%zu] %d vs %d\n", i, a.mv[i], b.mv[i]);
    return false;
  }
  for (i = 0; i < a.valid.size(); ++i) {
    if (a.valid[i] != b.valid[i]) break;
  }
  if (i < a.valid.size()) {
    std::printf("  valid[%zu] %u vs %u\n", i,
                static_cast<unsigned>(a.valid[i]),
                static_cast<unsigned>(b.valid[i]));
    return false;
  }
  for (i = 0; i < a.offset.size(); ++i) {
    if (a.offset[i] != b.offset[i]) break;
  }
  if (i < a.offset.size()) {
    std::printf("  offset[%zu] %d vs %d\n", i, a.offset[i], b.offset[i]);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Plan case
// ---------------------------------------------------------------------------

struct HostPlan {
  std::vector<ctip::TipPlanBlock> blocks;
  std::vector<std::int32_t> grid0;  // Mv as two int32 per cell
  std::vector<std::int32_t> grid1;
};

void run_plan_case(const FieldCase& c, const HostField& field) {
  const std::size_t cells =
      static_cast<std::size_t>(c.mi_cols) * c.mi_rows;
  HostPlan dev_plan;
  const int block_cols = (c.mi_cols + c.bw4 - 1) / c.bw4;
  const int block_rows = (c.mi_rows + c.bw4 - 1) / c.bw4;
  dev_plan.blocks.resize(static_cast<std::size_t>(block_cols) * block_rows);
  dev_plan.grid0.assign(cells * 2, 0);
  dev_plan.grid1.assign(cells * 2, 0);

  ctip::TipField f = const_cast<HostField&>(field).dev();
  const ctip::TipMotionFieldConfig cfg = c.device_config();
  const ctip::TipOffsets offsets = cdev::tip_offsets(cfg);
  const ctip::Mv global = c.global_mv;
  std::vector<ctip::Mv> grid0(cells);
  std::vector<ctip::Mv> grid1(cells);
  int index = 0;
  for (int row = 0; row < c.mi_rows; row += c.bw4) {
    for (int col = 0; col < c.mi_cols; col += c.bw4) {
      cdev::derive_tip_block(f, cfg, offsets, c.bw4, c.bw4, global, row, col,
                             &dev_plan.blocks[static_cast<std::size_t>(index)],
                             grid0.data(), grid1.data());
      ++index;
    }
  }

  std::vector<temporal::TipBlock> cpu_blocks(
      static_cast<std::size_t>(block_cols) * block_rows);
  std::vector<temporal::Mv> cpu_grid0(cells);
  std::vector<temporal::Mv> cpu_grid1(cells);
  temporal::TipPlan plan;
  plan.num_blocks = static_cast<int>(cpu_blocks.size());
  plan.blocks = cpu_blocks.data();
  plan.mv_grid = cpu_grid0.data();
  plan.mv_grid1 = cpu_grid1.data();
  const temporal::MotionFieldConfig cc = c.cpu_config();
  temporal::TipConfig tip = c.cpu_tip();
  tip.enable_tip_refinemv = c.bw4 == 2;
  tip.interp_filter = temporal::InterpFilter::kEighttapSharp;
  const temporal::MotionField cf = const_cast<HostField&>(field).cpu();
  const temporal::TemporalStatus status =
      temporal::build_tip_plan(cc, tip, cf, &plan);
  check(status == temporal::TemporalStatus::kOk, "CPU build_tip_plan status");
  check(plan.num_blocks == static_cast<int>(dev_plan.blocks.size()),
        "plan block count");
  check(plan.weight == inter::temporal::kTipWeightingFactor[c.weight_index],
        "plan weight");
  check(plan.offsets.ref_offset == offsets.ref_offset &&
            plan.offsets.past_offset == offsets.past_offset &&
            plan.offsets.future_offset == offsets.future_offset,
        "plan offsets");

  bool blocks_ok = true;
  for (int i = 0; i < plan.num_blocks; ++i) {
    const temporal::TipBlock& a = cpu_blocks[static_cast<std::size_t>(i)];
    const ctip::TipPlanBlock& b = dev_plan.blocks[static_cast<std::size_t>(i)];
    if (a.mi_row != b.mi_row || a.mi_col != b.mi_col || a.bw4 != b.bw4 ||
        a.bh4 != b.bh4 || a.mv[0].row != b.mv[0].row ||
        a.mv[0].col != b.mv[0].col || a.mv[1].row != b.mv[1].row ||
        a.mv[1].col != b.mv[1].col ||
        (a.motion_field_valid ? 1 : 0) != b.motion_field_valid) {
      blocks_ok = false;
      std::printf("  block %d mismatch\n", i);
      break;
    }
  }
  check(blocks_ok, "plan blocks");
  bool grid_ok = true;
  for (std::size_t i = 0; i < cells; ++i) {
    if (cpu_grid0[i].row != grid0[i].row ||
        cpu_grid0[i].col != grid0[i].col ||
        cpu_grid1[i].row != grid1[i].row ||
        cpu_grid1[i].col != grid1[i].col) {
      grid_ok = false;
      std::printf("  grid mismatch at %zu\n", i);
      break;
    }
  }
  check(grid_ok, "plan mv grids");
}

// ---------------------------------------------------------------------------
// Prediction reference (bit-depth parameterised transcription of
// temporal_predict.h; the CPU reference itself is 8-bit only)
// ---------------------------------------------------------------------------

struct RefPlane {
  std::vector<std::uint8_t> u8;
  std::vector<std::uint16_t> u16;
  int stride = 0;
  int width = 0;
  int height = 0;

  template <typename Sample>
  const Sample* data() const;
};
template <>
const std::uint8_t* RefPlane::data<std::uint8_t>() const {
  return u8.data();
}
template <>
const std::uint16_t* RefPlane::data<std::uint16_t>() const {
  return u16.data();
}

template <typename Sample>
std::int32_t ref_hsum(const Sample* ref, int stride, int row, int col,
                      int first_x, int last_x, const std::int16_t* filter) {
  std::int32_t sum = 0;
  for (int t = 0; t < 8; ++t) {
    const int cx = inter::adv::clip3(first_x, last_x, col + t);
    sum += static_cast<std::int32_t>(filter[t]) *
           static_cast<std::int32_t>(ref[static_cast<std::size_t>(row) *
                                            static_cast<std::size_t>(stride) +
                                        static_cast<std::size_t>(cx)]);
  }
  return sum;
}

template <typename Sample>
std::int32_t ref_sample(const Sample* ref, int stride, int first_x,
                        int first_y, int last_x, int last_y,
                        std::int32_t start_x, std::int32_t start_y,
                        std::int32_t step_x, std::int32_t step_y,
                        temporal::InterpFilter filter_h,
                        temporal::InterpFilter filter_v, int round1,
                        int bit_depth, bool clip_result, int r, int c) {
  const std::int32_t p_x = start_x + step_x * c;
  const int col = static_cast<int>(p_x >> 10) - 3;
  const int phase_h = static_cast<int>((p_x >> 6) & 15);
  const std::int16_t* fh = &inter::mc::kSubpelFilters[static_cast<int>(filter_h)]
                                                     [phase_h][0];
  const std::int32_t p_y = (start_y & 1023) + step_y * r;
  const int row_base = static_cast<int>(p_y >> 10);
  const int phase_v = static_cast<int>((p_y >> 6) & 15);
  const std::int16_t* fv = &inter::mc::kSubpelFilters[static_cast<int>(filter_v)]
                                                     [phase_v][0];
  std::int32_t sum_v = 0;
  for (int t = 0; t < 8; ++t) {
    const int row =
        inter::adv::clip3(first_y, last_y,
                          static_cast<int>(start_y >> 10) + row_base + t - 3);
    const std::int32_t intermediate =
        inter::adv::round2(ref_hsum(ref, stride, row, col, first_x, last_x, fh),
                           3);
    sum_v += static_cast<std::int32_t>(fv[t]) * intermediate;
  }
  const std::int32_t result = inter::adv::round2(sum_v, round1);
  return clip_result ? inter::adv::clip1(result, bit_depth) : result;
}

template <typename Sample>
void ref_predict_job(const ctip::TipPredictJob& job, Sample* dst) {
  const bool single = job.num_refs == 1 || job.cwp_weight == 16;
  const temporal::InterpFilter filter =
      static_cast<temporal::InterpFilter>(job.filter);
  const temporal::InterpFilter filter_h =
      inter::mc::mc_pass_filter(filter, job.w <= 4);
  const temporal::InterpFilter filter_v =
      inter::mc::mc_pass_filter(filter, job.h <= 4);
  int first_x[2] = {0, 0};
  int first_y[2] = {0, 0};
  int last_x[2] = {0, 0};
  int last_y[2] = {0, 0};
  const int lists = single ? 1 : 2;
  for (int list = 0; list < lists; ++list) {
    if (job.use_ref_area != 0) {
      first_x[list] = job.area_first_x[list];
      first_y[list] = job.area_first_y[list];
      last_x[list] = job.area_last_x[list];
      last_y[list] = job.area_last_y[list];
    } else {
      last_x[list] = job.ref[list].width - 1;
      last_y[list] = job.ref[list].height - 1;
    }
  }
  inter::adv::MvScaleOutput loc[2];
  for (int list = 0; list < lists; ++list) {
    int mv_pair[2] = {job.mv[list].row, job.mv[list].col};
    loc[list] = inter::adv::motion_vector_scaling(
        job.sub_x > 0 ? 1 : 0, job.sub_x, job.sub_y, job.x, job.y, mv_pair,
        false, job.ref[list].width << job.sub_x,
        job.ref[list].height << job.sub_y, job.frame_w, job.frame_h);
  }
  for (int r = 0; r < job.h; ++r) {
    for (int c = 0; c < job.w; ++c) {
      std::int32_t value = 0;
      if (single) {
        const Sample* ref = static_cast<const Sample*>(job.ref[0].data);
        value = ref_sample(ref, job.ref[0].stride, first_x[0], first_y[0],
                           last_x[0], last_y[0], loc[0].start_x, loc[0].start_y,
                           loc[0].step_x, loc[0].step_y, filter_h, filter_v, 11,
                           job.bit_depth, true, r, c);
      } else {
        const Sample* ref0 = static_cast<const Sample*>(job.ref[0].data);
        const Sample* ref1 = static_cast<const Sample*>(job.ref[1].data);
        const std::int32_t v0 =
            ref_sample(ref0, job.ref[0].stride, first_x[0], first_y[0],
                       last_x[0], last_y[0], loc[0].start_x, loc[0].start_y,
                       loc[0].step_x, loc[0].step_y, filter_h, filter_v, 7,
                       job.bit_depth, false, r, c);
        const std::int32_t v1 =
            ref_sample(ref1, job.ref[1].stride, first_x[1], first_y[1],
                       last_x[1], last_y[1], loc[1].start_x, loc[1].start_y,
                       loc[1].step_x, loc[1].step_y, filter_h, filter_v, 7,
                       job.bit_depth, false, r, c);
        value = inter::adv::clip1(
            inter::adv::round2(
                job.cwp_weight * v0 + (16 - job.cwp_weight) * v1, 8),
            job.bit_depth);
      }
      dst[static_cast<std::size_t>(r) *
              static_cast<std::size_t>(job.dst_stride) +
          static_cast<std::size_t>(c)] = static_cast<Sample>(value);
    }
  }
}

// ---------------------------------------------------------------------------
// Prediction case
// ---------------------------------------------------------------------------

struct PredictCase {
  int plane_w = 0;
  int plane_h = 0;
  int stride = 0;
  int frame_w = 0;
  int frame_h = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int sub_x = 0;
  int sub_y = 0;
  int weight = 8;
  int num_refs = 2;
  int filter = 2;
  int bit_depth = 8;
  int sample_bytes = 1;
  int use_area = 0;
  int area[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
  ctip::Mv mv[2];
};

PredictCase make_predict_case(int variant, int bit_depth) {
  PredictCase c;
  c.bit_depth = bit_depth;
  c.sample_bytes = bit_depth > 8 ? 2 : 1;
  c.plane_w = random_range(20, 48);
  c.plane_h = random_range(20, 48);
  c.stride = c.plane_w + random_range(0, 6);
  c.sub_x = chance(35) ? 1 : 0;
  c.sub_y = c.sub_x;
  if (c.sub_x == 0) {
    c.plane_w = (c.plane_w + 1) & ~1;
    c.plane_h = (c.plane_h + 1) & ~1;
  } else {
    c.plane_w = (c.plane_w + 3) & ~3;
    c.plane_h = (c.plane_h + 3) & ~3;
  }
  c.stride = c.plane_w + random_range(0, 6);
  c.frame_w = (c.plane_w << c.sub_x) + (chance(30) ? random_range(1, 8) : 0);
  c.frame_h = (c.plane_h << c.sub_y) + (chance(30) ? random_range(1, 8) : 0);
  const int bw4 = (variant & 1) ? 2 : 4;
  c.w = random_range(4, bw4 * 4);
  c.h = random_range(4, bw4 * 4);
  c.w = std::min(c.w, bw4 * 4);
  c.h = std::min(c.h, bw4 * 4);
  if (chance(25)) c.w = random_range(3, 5);  // four-tap downgrade
  if (chance(25)) c.h = random_range(3, 5);
  c.x = random_range(0, std::max(0, c.plane_w - 1));
  c.y = random_range(0, std::max(0, c.plane_h - 1));
  c.w = std::min(c.w, c.plane_w - c.x);
  c.h = std::min(c.h, c.plane_h - c.y);
  if (c.w <= 0 || c.h <= 0) {
    c.w = 4;
    c.h = 4;
  }
  c.num_refs = chance(25) ? 1 : 2;
  c.weight = inter::temporal::kTipWeightingFactor[random_range(0, 7)];
  if (c.num_refs == 1) c.weight = 8;
  c.filter = random_range(0, 3);
  c.mv[0] = ctip::Mv{random_range(-(1 << 16) + 1, (1 << 16) - 1),
                     random_range(-(1 << 16) + 1, (1 << 16) - 1)};
  c.mv[1] = ctip::Mv{random_range(-(1 << 16) + 1, (1 << 16) - 1),
                     random_range(-(1 << 16) + 1, (1 << 16) - 1)};
  c.use_area = chance(30) ? 1 : 0;
  const int lists = c.num_refs;
  for (int list = 0; list < lists; ++list) {
    const int ax = random_range(0, c.plane_w - 1);
    const int ay = random_range(0, c.plane_h - 1);
    c.area[list][0] = ax;
    c.area[list][1] = ay;
    c.area[list][2] = random_range(ax, c.plane_w - 1);
    c.area[list][3] = random_range(ay, c.plane_h - 1);
  }
  return c;
}

void fill_ref_u8(RefPlane* ref, const PredictCase& c, int seed_offset) {
  ref->stride = c.stride;
  ref->width = c.plane_w;
  ref->height = c.plane_h;
  ref->u8.resize(static_cast<std::size_t>(ref->stride) * c.plane_h);
  for (std::size_t i = 0; i < ref->u8.size(); ++i) {
    ref->u8[i] = static_cast<std::uint8_t>(
        next_random() + static_cast<std::uint32_t>(seed_offset));
  }
}

void fill_ref_u16(RefPlane* ref, const PredictCase& c, int seed_offset) {
  ref->stride = c.stride;
  ref->width = c.plane_w;
  ref->height = c.plane_h;
  ref->u16.resize(static_cast<std::size_t>(ref->stride) * c.plane_h);
  for (std::size_t i = 0; i < ref->u16.size(); ++i) {
    ref->u16[i] = static_cast<std::uint16_t>(
        (next_random() + static_cast<std::uint32_t>(seed_offset)) & 1023u);
  }
}

template <typename Sample>
void run_predict_device(const PredictCase& c, const RefPlane ref[2],
                        std::vector<Sample>* dst) {
  ctip::TipPredictJob job{};
  job.ref[0].data = ref[0].data<Sample>();
  job.ref[0].stride = ref[0].stride;
  job.ref[0].width = ref[0].width;
  job.ref[0].height = ref[0].height;
  job.ref[1].data = ref[1].data<Sample>();
  job.ref[1].stride = ref[1].stride;
  job.ref[1].width = ref[1].width;
  job.ref[1].height = ref[1].height;
  dst->assign(static_cast<std::size_t>(c.w) * c.h, 0);
  job.dst = dst->data();
  job.dst_stride = c.w;
  job.x = c.x;
  job.y = c.y;
  job.w = c.w;
  job.h = c.h;
  job.sub_x = c.sub_x;
  job.sub_y = c.sub_y;
  job.frame_w = c.frame_w;
  job.frame_h = c.frame_h;
  job.mv[0] = c.mv[0];
  job.mv[1] = c.mv[1];
  job.num_refs = c.num_refs;
  job.cwp_weight = c.weight;
  job.filter = c.filter;
  job.bit_depth = c.bit_depth;
  job.sample_bytes = c.sample_bytes;
  job.use_ref_area = c.use_area;
  for (int list = 0; list < 2; ++list) {
    job.area_first_x[list] = c.area[list][0];
    job.area_first_y[list] = c.area[list][1];
    job.area_last_x[list] = c.area[list][2];
    job.area_last_y[list] = c.area[list][3];
  }
  cdev::predict_job_thread<Sample>(job, 0, 1);
}

void run_predict_case_8(int variant) {
  const PredictCase c = make_predict_case(variant, 8);
  RefPlane ref[2];
  fill_ref_u8(&ref[0], c, 17);
  fill_ref_u8(&ref[1], c, 91);

  std::vector<std::uint8_t> dev_dst;
  run_predict_device<std::uint8_t>(c, ref, &dev_dst);

  std::vector<std::uint8_t> ref_dst(static_cast<std::size_t>(c.w) * c.h);
  {
    ctip::TipPredictJob job{};
    job.ref[0].data = ref[0].data<std::uint8_t>();
    job.ref[0].stride = ref[0].stride;
    job.ref[0].width = ref[0].width;
    job.ref[0].height = ref[0].height;
    job.ref[1].data = ref[1].data<std::uint8_t>();
    job.ref[1].stride = ref[1].stride;
    job.ref[1].width = ref[1].width;
    job.ref[1].height = ref[1].height;
    job.dst = ref_dst.data();
    job.dst_stride = c.w;
    job.x = c.x;
    job.y = c.y;
    job.w = c.w;
    job.h = c.h;
    job.sub_x = c.sub_x;
    job.sub_y = c.sub_y;
    job.frame_w = c.frame_w;
    job.frame_h = c.frame_h;
    job.mv[0] = c.mv[0];
    job.mv[1] = c.mv[1];
    job.num_refs = c.num_refs;
    job.cwp_weight = c.weight;
    job.filter = c.filter;
    job.bit_depth = 8;
    job.sample_bytes = 1;
    job.use_ref_area = c.use_area;
    for (int list = 0; list < 2; ++list) {
      job.area_first_x[list] = c.area[list][0];
      job.area_first_y[list] = c.area[list][1];
      job.area_last_x[list] = c.area[list][2];
      job.area_last_y[list] = c.area[list][3];
    }
    ref_predict_job(job, ref_dst.data());
  }

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
  const temporal::TemporalStatus status = temporal::tip_predict_block(in);
  check(status == temporal::TemporalStatus::kOk, "CPU tip_predict status");
  check(dev_dst == cpu_dst, "8-bit device == CPU tip_predict");
  check(ref_dst == cpu_dst, "8-bit local reference == CPU tip_predict");
}

void run_predict_case_10(int variant) {
  const PredictCase c = make_predict_case(variant, 10);
  RefPlane ref[2];
  fill_ref_u16(&ref[0], c, 23);
  fill_ref_u16(&ref[1], c, 77);

  std::vector<std::uint16_t> dev_dst;
  run_predict_device<std::uint16_t>(c, ref, &dev_dst);

  std::vector<std::uint16_t> want(static_cast<std::size_t>(c.w) * c.h);
  ctip::TipPredictJob job{};
  job.ref[0].data = ref[0].data<std::uint16_t>();
  job.ref[0].stride = ref[0].stride;
  job.ref[0].width = ref[0].width;
  job.ref[0].height = ref[0].height;
  job.ref[1].data = ref[1].data<std::uint16_t>();
  job.ref[1].stride = ref[1].stride;
  job.ref[1].width = ref[1].width;
  job.ref[1].height = ref[1].height;
  job.dst = want.data();
  job.dst_stride = c.w;
  job.x = c.x;
  job.y = c.y;
  job.w = c.w;
  job.h = c.h;
  job.sub_x = c.sub_x;
  job.sub_y = c.sub_y;
  job.frame_w = c.frame_w;
  job.frame_h = c.frame_h;
  job.mv[0] = c.mv[0];
  job.mv[1] = c.mv[1];
  job.num_refs = c.num_refs;
  job.cwp_weight = c.weight;
  job.filter = c.filter;
  job.bit_depth = 10;
  job.sample_bytes = 2;
  job.use_ref_area = c.use_area;
  for (int list = 0; list < 2; ++list) {
    job.area_first_x[list] = c.area[list][0];
    job.area_first_y[list] = c.area[list][1];
    job.area_last_x[list] = c.area[list][2];
    job.area_last_y[list] = c.area[list][3];
  }
  ref_predict_job(job, want.data());
  check(dev_dst == want, "10-bit device == bit-depth reference");
}

}  // namespace

#ifndef AV2ENC_TIP_TEST_NO_MAIN
int main() {
  std::printf("tip_emulation_test: shared device header vs CPU reference\n");

  // §7.9.4/.5 projection: every frame distance and extreme MVs.
  {
    const int kMvSamples[] = {0, 1, -1, 2047, -2047, 4096, -4096,
                              (1 << 16) - 1, -((1 << 16) - 1)};
    for (int denominator = 0; denominator <= temporal::kMaxFrameDistance; ++denominator) {
      for (int numerator = -temporal::kMaxFrameDistance;
           numerator <= temporal::kMaxFrameDistance;
           ++numerator) {
        for (int row : kMvSamples) {
          for (int col : kMvSamples) {
            const ctip::Mv mv{row, col};
            const temporal::Mv cpu = temporal::get_mv_projection(
                temporal::Mv{row, col}, numerator, denominator);
            const ctip::Mv got = cdev::mv_projection(mv, numerator, denominator);
            const ctip::Mv clamp = cdev::mv_projection_clamp(mv, numerator, denominator);
            const temporal::Mv cpu_clamp = temporal::get_mv_projection_clamp(
                temporal::Mv{row, col}, numerator, denominator);
            check(got.row == cpu.row && got.col == cpu.col,
                  "mv_projection parity");
            check(clamp.row == cpu_clamp.row && clamp.col == cpu_clamp.col,
                  "mv_projection_clamp parity");
          }
        }
      }
    }
  }

  // §7.10.2-.5, §7.10.6: 240 randomised motion fields.
  for (int variant = 0; variant < 240; ++variant) {
    const FieldCase c = make_field_case(variant);
    const int w8 = c.mi_cols >> 1;
    const int h8 = c.mi_rows >> 1;
    HostField initial;
    initial.init(w8, h8);
    randomize_field(&initial, variant % 3 == 0 ? 30 : (variant % 3 == 1 ? 65 : 100));

    HostField dev_field = initial;
    HostField cpu_field = initial;
    HostScratch dev_scratch;
    dev_scratch.init(static_cast<std::size_t>(w8) * h8);
    HostScratch cpu_scratch;
    cpu_scratch.init(static_cast<std::size_t>(w8) * h8);

    setup_device_field(&dev_field, c, dev_scratch);

    const temporal::MotionFieldConfig cc = c.cpu_config();
    const temporal::TipConfig tip = c.cpu_tip();
    temporal::MotionField cf = cpu_field.cpu();
    temporal::MotionFieldScratch cs = cpu_scratch.cpu();
    const temporal::TemporalStatus status =
        temporal::setup_tip_motion_field(cc, tip, &cf, nullptr, &cs, nullptr);
    check(status == temporal::TemporalStatus::kOk, "CPU setup status");
    check(same_field(dev_field, cpu_field), "field setup parity");
    check(cdev::tip_offsets(c.device_config()).ref_offset ==
              temporal::get_tip_offsets(tip).ref_offset,
          "ref offset parity");
    check(cdev::tip_offsets(c.device_config()).past_offset ==
              temporal::get_tip_offsets(tip).past_offset,
          "past offset parity");
    check(cdev::tip_offsets(c.device_config()).future_offset ==
              temporal::get_tip_offsets(tip).future_offset,
          "future offset parity");

    run_plan_case(c, dev_field);
  }

  // §7.10.5 on its own, including sparse fields.
  for (int variant = 0; variant < 60; ++variant) {
    FieldCase c;
    c.mi_cols = 8 * random_range(1, 12);
    c.mi_rows = 8 * random_range(1, 10);
    c.proj_step = 2;
    c.sb_64x64 = chance(50) ? 1 : 0;
    HostField initial;
    initial.init(c.mi_cols >> 1, c.mi_rows >> 1);
    randomize_field(&initial, variant % 2 == 0 ? 25 : 80);
    HostField dev_field = initial;
    HostField cpu_field = initial;
    const ctip::TipField f = dev_field.dev();
    const ctip::TipMotionFieldConfig cfg = c.device_config();
    for (int y8 = 0; y8 < f.h8; y8 += 2) {
      for (int x8 = 0; x8 < f.w8; x8 += 2) {
        cdev::fill_sample_gap_cell(f, cfg, y8, x8, 0, 1);
        cdev::fill_sample_gap_cell(f, cfg, y8, x8, 1, 0);
        cdev::fill_sample_gap_cell(f, cfg, y8, x8, 1, 1);
      }
    }
    temporal::MotionField cf = cpu_field.cpu();
    const temporal::TemporalStatus status =
        temporal::fill_motion_field_sample_gap(c.cpu_config(), &cf);
    check(status == temporal::TemporalStatus::kOk, "CPU gap fill status");
    check(same_field(dev_field, cpu_field), "sample-gap parity");
  }

  // §7.13.3.2: 8-bit and 10-bit prediction, all weights/filters/areas.
  for (int variant = 0; variant < 320; ++variant) {
    run_predict_case_8(variant);
    run_predict_case_10(variant);
  }

  std::printf("tip_emulation_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
#endif  // AV2ENC_TIP_TEST_NO_MAIN
