// Host-emulation parity test for src/cuda/compound_full.
//
// Compiled without nvcc: the shared device header is built as host code with
// the CUDA keywords stubbed, and the real `av2enc::cuda::Arena` is compiled
// against a malloc-based cuda_runtime stub. The test therefore exercises the
// exact arithmetic and the exact arena contract that the GPU path uses.
//
// Oracles:
//   * 8-bit blend  -> inter::adv::run_compound_job() (CPU reference)
//   * 10-bit blend -> compound_full_ref::blend_job() (independent literal)
//   * OPFL         -> inter::temporal::opfl_compute()
//   * refine-MV    -> inter::temporal::{search_refinemv, refine_predict_block}
//   * skip mode    -> av2enc::video::multiref::{build_compound_prediction,
//                                   apply_compound_prediction} (8-bit)
//                   + compound_full_ref::skip_job() (10-bit)
//
// Build/run: ./build_emulation_test.sh

#include "compound_full_device.cuh"
#include "compound_full_reference.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "codec/inter/advanced/compound.h"
#include "codec/inter/temporal/opfl.h"
#include "codec/inter/temporal/refinemv.h"
#include "codec/inter/multiref/compound_predict.h"

namespace cf = av2enc::cuda::compound_full;
namespace ref = compound_full_ref;
namespace adv = av2enc::inter::adv;
namespace mc = av2enc::inter::mc;
using av2enc::inter::adv::CompoundJob;
using av2enc::inter::adv::CompoundType;
using av2enc::inter::temporal::Mv;
using av2enc::inter::temporal::MvPair;
using av2enc::inter::temporal::PlaneView;
using av2enc::inter::temporal::RefArea;
using av2enc::inter::temporal::RefAreaInput;
using av2enc::inter::temporal::RefinePredictInput;
using av2enc::inter::temporal::RefineSearchInput;
using av2enc::inter::temporal::RefineSearchResult;

namespace {

int g_checks = 0;
int g_failures = 0;

void expect(bool condition, const char* what, int line) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    if (g_failures <= 40) {
      std::printf("FAIL line %d: %s\n", line, what);
    }
  }
}

#define EXPECT(cond) expect((cond), #cond, __LINE__)

std::uint32_t g_rng = 0x12345678u;
const std::uint8_t* g_wedge_cache = nullptr;
av2enc::cuda::Arena g_arena;
cf::CompoundFullWorkspace g_workspace;
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

int bsize_for(int w, int h) {
  for (int b = 0; b < cf::detail::kCfBlockSizes; ++b) {
    if (cf::detail::kCfWedgeBlockWidth[b] == w &&
        cf::detail::kCfWedgeBlockHeight[b] == h) {
      return b;
    }
  }
  return 0;
}

CompoundJob to_cpu_job(const cf::BlendJob& job, std::uint8_t* dst,
                       std::uint8_t* mask, const std::uint8_t* luma_mask) {
  CompoundJob cpu{};
  cpu.plane = job.plane;
  cpu.sub_x = job.sub_x;
  cpu.sub_y = job.sub_y;
  cpu.rect = adv::BlockRect{job.rect_x, job.rect_y, job.rect_w, job.rect_h};
  cpu.bsize = job.bsize;
  cpu.bit_depth = job.bit_depth;
  cpu.p0 = job.p0;
  cpu.p0_stride = job.p0_stride;
  cpu.p1 = job.p1;
  cpu.p1_stride = job.p1_stride;
  cpu.intra = job.intra ? static_cast<const std::uint8_t*>(job.intra) : nullptr;
  cpu.intra_stride = job.intra_stride;
  cpu.luma_mask = luma_mask;
  cpu.luma_mask_stride = job.luma_mask_stride;
  cpu.mask_out = mask;
  cpu.mask_out_stride = job.mask_out_stride;
  cpu.dst = dst;
  cpu.dst_stride = job.dst_stride;
  cpu.type = static_cast<CompoundType>(job.type);
  cpu.cwp_weight = job.cwp_weight;
  cpu.wedge_index = job.wedge_index;
  cpu.wedge_sign = job.wedge_sign;
  cpu.mask_type = job.mask_type;
  cpu.imp_mask_eligible = job.imp_mask_eligible;
  for (int l = 0; l < 2; ++l) {
    cpu.ref_start_x[l] = job.ref_start_x[l];
    cpu.ref_start_y[l] = job.ref_start_y[l];
  }
  cpu.mi_cols = job.mi_cols;
  cpu.mi_rows = job.mi_rows;
  cpu.is_inter_intra = job.is_inter_intra;
  cpu.interintra_mode =
      static_cast<av2enc::inter::adv::InterIntraMode>(job.interintra_mode);
  return cpu;
}

// ---------------------------------------------------------------------------
// Blend parity
// ---------------------------------------------------------------------------

struct BlendBuffers {
  int luma_w = 0;
  int luma_h = 0;
  std::vector<std::int16_t> p0;
  std::vector<std::int16_t> p1;
  std::vector<std::uint8_t> intra8;
  std::vector<std::uint16_t> intra16;
  std::vector<std::uint8_t> luma_mask;
};

void fill_preds(int count, std::vector<std::int16_t>* p0,
                std::vector<std::int16_t>* p1, int max_sample) {
  p0->resize(static_cast<std::size_t>(count));
  p1->resize(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    (*p0)[static_cast<std::size_t>(i)] = static_cast<std::int16_t>(
        random_range(0, max_sample) * 16 + random_range(-15, 15));
    (*p1)[static_cast<std::size_t>(i)] = static_cast<std::int16_t>(
        random_range(0, max_sample) * 16 + random_range(-15, 15));
  }
}

struct BlendMode {
  int type = cf::kCompoundAverage;
  int cwp = 8;
  int wedge_index = 0;
  int wedge_sign = 0;
  int mask_type = 0;
  int ii_mode = 0;
  int imp = 0;
  int ref_sel = 0;
};

// Runs one luma job through the emulated device body and the appropriate
// oracle, then returns the produced luma mask for chroma use.
void run_luma_blend(const BlendBuffers& buffers, const BlendMode& mode,
                    int bit_depth, bool use_cache, std::vector<std::uint8_t>* mask_out) {
  const int w = buffers.luma_w;
  const int h = buffers.luma_h;
  const int max_sample = bit_depth == 8 ? 255 : 1023;
  const std::size_t total = static_cast<std::size_t>(w) * h;

  cf::BlendJob job{};
  job.plane = 0;
  job.sub_x = 0;
  job.sub_y = 0;
  job.rect_w = w;
  job.rect_h = h;
  job.bsize = bsize_for(w, h);
  job.bit_depth = bit_depth;
  job.p0 = buffers.p0.data();
  job.p0_stride = w;
  job.p1 = buffers.p1.data();
  job.p1_stride = w;
  job.type = mode.type;
  job.cwp_weight = mode.cwp;
  job.wedge_index = mode.wedge_index;
  job.wedge_sign = mode.wedge_sign;
  job.mask_type = mode.mask_type;
  job.imp_mask_eligible = mode.imp;
  job.interintra_mode = mode.ii_mode;
  job.mi_cols = w;
  job.mi_rows = h;
  // RefStartX/Y patterns: both on, only 0 on, only 1 on, both off.
  job.ref_start_x[0] = mode.ref_sel == 1 ? -2 : 0;
  job.ref_start_x[1] = mode.ref_sel == 2 ? w + 2 : 0;
  job.ref_start_y[0] = mode.ref_sel == 3 ? h + 2 : 0;
  job.ref_start_y[1] = mode.ref_sel == 3 ? -2 : 0;
  mask_out->assign(total, 0);
  job.mask_out = mask_out->data();
  job.mask_out_stride = w;
  job.wedge_cache = use_cache ? g_wedge_cache : nullptr;

  std::vector<std::uint8_t> dst8_gpu(total, 0);
  std::vector<std::uint8_t> dst8_cpu(total, 0);
  std::vector<std::uint16_t> dst16_gpu(total, 0);
  std::vector<std::uint16_t> dst16_ref(total, 0);

  if (mode.type == cf::kCompoundIntra || mode.imp == 2) {
    // Inter-intra: dst must already hold the clipped inter prediction.
    for (std::size_t i = 0; i < total; ++i) {
      const int inter = random_range(0, max_sample);
      dst8_gpu[i] = static_cast<std::uint8_t>(inter);
      dst8_cpu[i] = static_cast<std::uint8_t>(inter);
      dst16_gpu[i] = static_cast<std::uint16_t>(inter);
      dst16_ref[i] = static_cast<std::uint16_t>(inter);
    }
  }

  if (bit_depth == 8) {
    job.dst = dst8_gpu.data();
    job.dst_stride = w;
    job.intra = buffers.intra8.data();
    job.intra_stride = w;
    job.intra_16bit = 0;
    if (mode.type == cf::kCompoundIntra || mode.imp == 2) {
      job.is_inter_intra = 1;
    }
    cf::detail::run_blend_job_device(job, 0, 1);
    const std::vector<std::uint8_t> gpu_mask = *mask_out;

    CompoundJob cpu = to_cpu_job(job, dst8_cpu.data(), nullptr,
                                 nullptr);
    cpu.mask_out = mask_out->data();
    cpu.mask_out_stride = w;
    adv::run_compound_job(cpu);
    EXPECT(std::memcmp(dst8_gpu.data(), dst8_cpu.data(), total) == 0);
    EXPECT(std::memcmp(gpu_mask.data(), mask_out->data(), total) == 0);
  } else {
    job.dst = dst16_gpu.data();
    job.dst_stride = w;
    job.dst_16bit = 1;
    job.intra = buffers.intra16.data();
    job.intra_stride = w;
    job.intra_16bit = 1;
    if (mode.type == cf::kCompoundIntra || mode.imp == 2) {
      job.is_inter_intra = 1;
    }
    cf::detail::run_blend_job_device(job, 0, 1);

    cf::BlendJob ref_job = job;
    ref_job.dst = dst16_ref.data();
    ref_job.dst_16bit = 1;
    ref_job.intra = buffers.intra16.data();
    ref_job.intra_16bit = 1;
    std::vector<std::uint8_t> ref_mask(total, 0);
    ref_job.mask_out = ref_mask.data();
    ref::blend_job(ref_job);
    EXPECT(std::memcmp(dst16_gpu.data(), dst16_ref.data(),
                       total * sizeof(std::uint16_t)) == 0);
    EXPECT(std::memcmp(mask_out->data(), ref_mask.data(), total) == 0);
  }
}

void test_blend() {
  const int sizes[][2] = {{4, 4},   {8, 8},   {8, 4},   {4, 8},   {16, 16},
                          {8, 16},  {16, 8},  {32, 32}, {32, 16}, {16, 32},
                          {64, 64}, {64, 32}, {32, 64}, {64, 16}, {16, 64},
                          {8, 32},  {32, 8},  {64, 8},  {8, 64}};
  int case_count = 0;
  for (const auto& size : sizes) {
    const int w = size[0];
    const int h = size[1];
    const int bsize = bsize_for(w, h);
    const int max_sample = 255;
    BlendBuffers buffers;
    buffers.luma_w = w;
    buffers.luma_h = h;
    fill_preds(w * h, &buffers.p0, &buffers.p1, max_sample);
    buffers.intra8.resize(static_cast<std::size_t>(w) * h);
    buffers.intra16.resize(static_cast<std::size_t>(w) * h);
    for (std::size_t i = 0; i < buffers.intra8.size(); ++i) {
      buffers.intra8[i] = static_cast<std::uint8_t>(random_range(0, 255));
      buffers.intra16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
    }

    std::vector<std::uint8_t> mask;
    std::vector<BlendMode> modes;
    for (int cwp : {8, 12, 4, 20, -4, 10, 6}) {
      BlendMode m;
      m.type = cf::kCompoundAverage;
      m.cwp = cwp;
      modes.push_back(m);
    }
    if (cf::detail::cf_bsize_has_wedge(bsize)) {
      const int indices[] = {0, 1, 17, 33, 34, 50, 67};
      for (int idx : indices) {
        for (int sign = 0; sign < 2; ++sign) {
          BlendMode m;
          m.type = cf::kCompoundWedge;
          m.wedge_index = idx;
          m.wedge_sign = sign;
          modes.push_back(m);
        }
      }
      if (w == 64 && h == 64) {
        for (int idx = 0; idx < cf::kWedgeTypes; ++idx) {
          for (int sign = 0; sign < 2; ++sign) {
            BlendMode m;
            m.type = cf::kCompoundWedge;
            m.wedge_index = idx;
            m.wedge_sign = sign;
            modes.push_back(m);
          }
        }
      }
      // Wedge inter-intra.
      for (int idx : {0, 33, 67}) {
        BlendMode m;
        m.type = cf::kCompoundWedge;
        m.wedge_index = idx;
        m.wedge_sign = 1;
        m.ii_mode = 3;
        m.imp = 2;  // marker: inter-intra path
        modes.push_back(m);
      }
    }
    for (int mt = 0; mt < 2; ++mt) {
      BlendMode m;
      m.type = cf::kCompoundDiffWtd;
      m.mask_type = mt;
      modes.push_back(m);
    }
    for (int ii = 0; ii < 4; ++ii) {
      BlendMode m;
      m.type = cf::kCompoundIntra;
      m.ii_mode = ii;
      m.imp = 2;
      modes.push_back(m);
    }

    for (const BlendMode& mode : modes) {
      for (int bit_depth : {8, 10}) {
        ++case_count;
        run_luma_blend(buffers, mode, bit_depth, (case_count % 2) != 0, &mask);
      }
    }
    // Implicit-mask patterns (§7.13.3.14 get_mask), including the four
    // reference on/off combinations.
    for (int sel = 0; sel < 4; ++sel) {
      BlendMode m;
      m.type = cf::kCompoundAverage;
      m.cwp = 8;
      m.imp = 1;
      m.ref_sel = sel;
      run_luma_blend(buffers, m, 8, false, &mask);
      run_luma_blend(buffers, m, 10, false, &mask);
    }

    // A wedge mask and diffwtd masks (per bit depth) for the chroma jobs below.
    std::vector<std::uint8_t> diff_mask[2][2];
    if (w >= 2 && h >= 2) {
      BlendMode mask_src;
      mask_src.type = cf::kCompoundWedge;
      mask_src.wedge_index = 33;
      mask_src.wedge_sign = 1;
      run_luma_blend(buffers, mask_src, 8, false, &mask);
      for (int bd = 0; bd < 2; ++bd) {
        for (int mt = 0; mt < 2; ++mt) {
          BlendMode diff_src;
          diff_src.type = cf::kCompoundDiffWtd;
          diff_src.mask_type = mt;
          run_luma_blend(buffers, diff_src, bd == 0 ? 8 : 10, false,
                         &diff_mask[bd][mt]);
        }
      }
    }

    // Chroma: average, wedge (luma mask subsample), diffwtd (both luma-mask
    // and luma-prediction derivations) and intra modes.
    if (w >= 2 && h >= 2) {
      const int cw = w / 2;
      const int ch = h / 2;
      BlendBuffers chroma;
      chroma.luma_w = cw;
      chroma.luma_h = ch;
      fill_preds(cw * ch, &chroma.p0, &chroma.p1, max_sample);
      chroma.intra8.resize(static_cast<std::size_t>(cw) * ch);
      chroma.intra16.resize(static_cast<std::size_t>(cw) * ch);
      for (std::size_t i = 0; i < chroma.intra8.size(); ++i) {
        chroma.intra8[i] = static_cast<std::uint8_t>(random_range(0, 255));
        chroma.intra16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
      }
      for (BlendMode mode : {BlendMode{cf::kCompoundAverage, 8, 0, 0, 0, 0, 0, 0},
                             BlendMode{cf::kCompoundWedge, 8, 33, 1, 0, 0, 0, 0},
                             BlendMode{cf::kCompoundDiffWtd, 8, 0, 0, 0, 0, 0, 0},
                             BlendMode{cf::kCompoundDiffWtd, 8, 0, 0, 1, 0, 0, 0},
                             BlendMode{cf::kCompoundIntra, 8, 0, 0, 0, 2, 2, 0}}) {
        for (int bit_depth : {8, 10}) {
          ++case_count;
          cf::BlendJob job{};
          job.plane = 1;
          job.sub_x = 1;
          job.sub_y = 1;
          job.rect_w = cw;
          job.rect_h = ch;
          job.bsize = bsize;
          job.bit_depth = bit_depth;
          job.p0 = chroma.p0.data();
          job.p0_stride = cw;
          job.p1 = chroma.p1.data();
          job.p1_stride = cw;
          job.type = mode.type;
          job.cwp_weight = mode.cwp;
          job.wedge_index = mode.wedge_index;
          job.wedge_sign = mode.wedge_sign;
          job.mask_type = mode.mask_type;
          job.interintra_mode = mode.ii_mode;
          const int bd_index = bit_depth == 8 ? 0 : 1;
          const std::vector<std::uint8_t>& chroma_mask =
              mode.type == cf::kCompoundDiffWtd
                  ? diff_mask[bd_index][mode.mask_type]
                  : mask;
          job.luma_mask = chroma_mask.data();
          job.luma_mask_stride = w;
          job.mi_cols = w;
          job.mi_rows = h;
          const std::size_t total = static_cast<std::size_t>(cw) * ch;
          if (mode.type == cf::kCompoundDiffWtd) {
            job.luma_p0 = buffers.p0.data();
            job.luma_p0_stride = w;
            job.luma_p1 = buffers.p1.data();
            job.luma_p1_stride = w;
          }
          std::vector<std::uint8_t> inter8(total, 0);
          std::vector<std::uint16_t> inter16(total, 0);
          if (mode.type == cf::kCompoundIntra) {
            for (std::size_t i = 0; i < total; ++i) {
              inter8[i] = static_cast<std::uint8_t>(random_range(0, 255));
              inter16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
            }
          }
          if (bit_depth == 8) {
            std::vector<std::uint8_t> dst_gpu = inter8;
            std::vector<std::uint8_t> dst_cpu = inter8;
            job.dst = dst_gpu.data();
            job.dst_stride = cw;
            job.intra = chroma.intra8.data();
            job.intra_stride = cw;
            if (mode.type == cf::kCompoundIntra) job.is_inter_intra = 1;
            cf::detail::run_blend_job_device(job, 0, 1);
            CompoundJob cpu = to_cpu_job(job, dst_cpu.data(), nullptr,
                                         chroma_mask.data());
            adv::run_compound_job(cpu);
            EXPECT(std::memcmp(dst_gpu.data(), dst_cpu.data(), total) == 0);
            if (mode.type == cf::kCompoundDiffWtd) {
              cf::BlendJob alt = job;
              alt.luma_mask = nullptr;
              std::vector<std::uint8_t> alt_dst = inter8;
              alt.dst = alt_dst.data();
              cf::detail::run_blend_job_device(alt, 0, 1);
              EXPECT(std::memcmp(alt_dst.data(), dst_gpu.data(), total) == 0);
            }
          } else {
            std::vector<std::uint16_t> dst_gpu = inter16;
            std::vector<std::uint16_t> dst_ref = inter16;
            job.dst = dst_gpu.data();
            job.dst_stride = cw;
            job.dst_16bit = 1;
            job.intra = chroma.intra16.data();
            job.intra_stride = cw;
            job.intra_16bit = 1;
            if (mode.type == cf::kCompoundIntra) job.is_inter_intra = 1;
            cf::detail::run_blend_job_device(job, 0, 1);
            cf::BlendJob ref_job = job;
            ref_job.dst = dst_ref.data();
            ref::blend_job(ref_job);
            EXPECT(std::memcmp(dst_gpu.data(), dst_ref.data(),
                               total * sizeof(std::uint16_t)) == 0);
            if (mode.type == cf::kCompoundDiffWtd) {
              cf::BlendJob alt = job;
              alt.luma_mask = nullptr;
              std::vector<std::uint16_t> alt_dst = inter16;
              alt.dst = alt_dst.data();
              cf::detail::run_blend_job_device(alt, 0, 1);
              EXPECT(std::memcmp(alt_dst.data(), dst_gpu.data(),
                                 total * sizeof(std::uint16_t)) == 0);
            }
          }
        }
      }
      // 4:2:0, 4:2:2, 4:4:4 and vertical-subsampled luma-mask paths.
      const int sub_pairs[][2] = {{1, 1}, {1, 0}, {0, 1}, {0, 0}};
      for (const auto& sub : sub_pairs) {
        const int sub_x = sub[0];
        const int sub_y = sub[1];
        const int pw = w >> sub_x;
        const int ph = h >> sub_y;
        cf::BlendJob job{};
        job.plane = 1;
        job.sub_x = sub_x;
        job.sub_y = sub_y;
        job.rect_w = pw;
        job.rect_h = ph;
        job.bsize = bsize;
        job.bit_depth = 8;
        std::vector<std::int16_t> p0(static_cast<std::size_t>(pw) * ph, 0);
        std::vector<std::int16_t> p1(p0.size(), 0);
        std::vector<std::uint8_t> dst_gpu(p0.size(), 0);
        std::vector<std::uint8_t> dst_cpu(p0.size(), 0);
        job.p0 = p0.data();
        job.p0_stride = pw;
        job.p1 = p1.data();
        job.p1_stride = pw;
        job.type = cf::kCompoundWedge;
        job.wedge_index = 33;
        job.wedge_sign = 1;
        job.luma_mask = mask.data();
        job.luma_mask_stride = w;
        job.dst = dst_gpu.data();
        job.dst_stride = pw;
        cf::detail::run_blend_job_device(job, 0, 1);
        CompoundJob cpu = to_cpu_job(job, dst_cpu.data(), nullptr, mask.data());
        adv::run_compound_job(cpu);
        EXPECT(std::memcmp(dst_gpu.data(), dst_cpu.data(), p0.size()) == 0);
      }
    }
  }
  std::printf("blend: %d luma/chroma fixture runs\n", case_count);
}

// ---------------------------------------------------------------------------
// Skip-mode parity
// ---------------------------------------------------------------------------

void test_skip() {
  const int luma = 64;
  const int chroma = luma / 2;
  std::vector<std::uint8_t> y0(static_cast<std::size_t>(luma) * luma);
  std::vector<std::uint8_t> y1(y0.size());
  std::vector<std::uint8_t> u0(static_cast<std::size_t>(chroma) * chroma);
  std::vector<std::uint8_t> u1(u0.size());
  std::vector<std::uint8_t> v0(u0.size());
  std::vector<std::uint8_t> v1(u0.size());
  std::vector<std::uint16_t> y0_16(y0.size()), y1_16(y1.size());
  std::vector<std::uint16_t> u0_16(u0.size()), u1_16(u1.size());
  std::vector<std::uint16_t> v0_16(v0.size()), v1_16(v0.size());
  for (std::size_t i = 0; i < y0.size(); ++i) {
    y0[i] = static_cast<std::uint8_t>(random_range(0, 255));
    y1[i] = static_cast<std::uint8_t>(random_range(0, 255));
    y0_16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
    y1_16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
  }
  for (std::size_t i = 0; i < u0.size(); ++i) {
    u0[i] = static_cast<std::uint8_t>(random_range(0, 255));
    u1[i] = static_cast<std::uint8_t>(random_range(0, 255));
    v0[i] = static_cast<std::uint8_t>(random_range(0, 255));
    v1[i] = static_cast<std::uint8_t>(random_range(0, 255));
    u0_16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
    u1_16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
    v0_16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
    v1_16[i] = static_cast<std::uint16_t>(random_range(0, 1023));
  }

  av2enc::inter::mc::RefFrame frames[2];
  const std::uint8_t* planes0[3] = {y0.data(), u0.data(), v0.data()};
  const std::uint8_t* planes1[3] = {y1.data(), u1.data(), v1.data()};
  for (int list = 0; list < 2; ++list) {
    for (int p = 0; p < 3; ++p) {
      frames[list].plane[p] = av2enc::inter::mc::PlaneView{
          list == 0 ? planes0[p] : planes1[p],
          p == 0 ? luma : chroma};
    }
    frames[list].mi_cols = luma / 4;
    frames[list].mi_rows = luma / 4;
  }

  const int rects[][4] = {{0, 0, 16, 16},   {16, 16, 32, 32}, {32, 32, 32, 32},
                          {48, 48, 16, 16}, {0, 32, 16, 16},  {48, 0, 16, 16},
                          {16, 0, 48, 16},  {0, 16, 16, 48},  {0, 0, 64, 64}};
  const int filters[] = {static_cast<int>(
                             av2enc::inter::mc::InterpFilter::kEighttap),
                         static_cast<int>(
                             av2enc::inter::mc::InterpFilter::kEighttapSharp)};
  for (const auto& rect : rects) {
    for (int filter : filters) {
      const int mv_row = random_range(-24, 24) * 8;
      const int mv_col = random_range(-24, 24) * 8;

      // 8-bit vs the CPU compound module.
      cf::SkipJob job{};
      job.bit_depth = 8;
      job.frame_w = luma;
      job.frame_h = luma;
      job.mi_cols = luma / 4;
      job.mi_rows = luma / 4;
      job.x = rect[0];
      job.y = rect[1];
      job.w = rect[2];
      job.h = rect[3];
      job.bsize = bsize_for(job.w, job.h);
      job.mv_row = mv_row;
      job.mv_col = mv_col;
      job.filter = filter;
      std::vector<std::uint8_t> dy0(static_cast<std::size_t>(luma) * luma, 0);
      std::vector<std::uint8_t> dy1(dy0.size(), 0);
      std::vector<std::uint8_t> du0(static_cast<std::size_t>(chroma) * chroma, 0);
      std::vector<std::uint8_t> du1(du0.size(), 0);
      std::vector<std::uint8_t> dv0(du0.size(), 0);
      std::vector<std::uint8_t> dv1(du0.size(), 0);
      std::uint8_t* dst0[3] = {dy0.data(), du0.data(), dv0.data()};
      std::uint8_t* dst1[3] = {dy1.data(), du1.data(), dv1.data()};
      int strides[3] = {luma, chroma, chroma};
      for (int p = 0; p < 3; ++p) {
        job.ref[0][p] = cf::RefPlane{planes0[p], strides[p],
                                     p == 0 ? luma : chroma,
                                     p == 0 ? luma : chroma, 0};
        job.ref[1][p] = cf::RefPlane{planes1[p], strides[p],
                                     p == 0 ? luma : chroma,
                                     p == 0 ? luma : chroma, 0};
        job.dst[p] = dst0[p];
        job.dst_stride[p] = strides[p];
      }
      std::vector<int> skip_mask(cf::kMaxMaskSamples, 0);
      cf::detail::run_skip_job_device(job, 0, 1, skip_mask.data());

      av2enc::video::multiref::CompoundPrediction prediction;
      std::string error;
      EXPECT(av2enc::video::multiref::build_compound_prediction(
          frames[0], frames[1],
          av2enc::inter::mc::BlockRect{job.x, job.y, job.w, job.h},
          job.bsize,
          av2enc::inter::mc::Mv{mv_row, mv_col},
          av2enc::inter::mc::Mv{mv_row, mv_col},
          static_cast<av2enc::inter::mc::InterpFilter>(filter),
          luma / 4, luma / 4, prediction, error));
      av2enc::video::multiref::CompoundModeParams params;
      params.type = CompoundType::kAverage;
      params.cwp_weight = 8;
      params.use_imp_mask = false;
      EXPECT(av2enc::video::multiref::apply_compound_prediction(
          prediction, params, dst1, strides, error));
      EXPECT(std::memcmp(dy0.data(), dy1.data(), dy0.size()) == 0);
      EXPECT(std::memcmp(du0.data(), du1.data(), du0.size()) == 0);
      EXPECT(std::memcmp(dv0.data(), dv1.data(), dv0.size()) == 0);

      // 10-bit vs the independent reference.
      cf::SkipJob job16 = job;
      job16.bit_depth = 10;
      const std::uint16_t* p0_16[3] = {y0_16.data(), u0_16.data(), v0_16.data()};
      const std::uint16_t* p1_16[3] = {y1_16.data(), u1_16.data(), v1_16.data()};
      std::vector<std::uint16_t> ky0(static_cast<std::size_t>(luma) * luma, 0);
      std::vector<std::uint16_t> ky1(ky0.size(), 0);
      std::vector<std::uint16_t> ku0(static_cast<std::size_t>(chroma) * chroma, 0);
      std::vector<std::uint16_t> ku1(ku0.size(), 0);
      std::vector<std::uint16_t> kv0(ku0.size(), 0);
      std::vector<std::uint16_t> kv1(ku0.size(), 0);
      std::uint16_t* dst16_gpu[3] = {ky0.data(), ku0.data(), kv0.data()};
      std::uint16_t* dst16_ref[3] = {ky1.data(), ku1.data(), kv1.data()};
      for (int p = 0; p < 3; ++p) {
        const int dim = p == 0 ? luma : chroma;
        job16.ref[0][p] = cf::RefPlane{p0_16[p], dim, dim, dim, 1};
        job16.ref[1][p] = cf::RefPlane{p1_16[p], dim, dim, dim, 1};
        job16.dst[p] = dst16_gpu[p];
        job16.dst_stride[p] = dim;
        job16.dst_16bit[p] = 1;
      }
      cf::detail::run_skip_job_device(job16, 0, 1, skip_mask.data());
      cf::SkipJob ref_job = job16;
      for (int p = 0; p < 3; ++p) {
        ref_job.dst[p] = dst16_ref[p];
      }
      ref::skip_job(ref_job);
      EXPECT(std::memcmp(ky0.data(), ky1.data(),
                         ky0.size() * sizeof(std::uint16_t)) == 0);
      EXPECT(std::memcmp(ku0.data(), ku1.data(),
                         ku0.size() * sizeof(std::uint16_t)) == 0);
      EXPECT(std::memcmp(kv0.data(), kv1.data(),
                         kv0.size() * sizeof(std::uint16_t)) == 0);
    }
  }
}

// ---------------------------------------------------------------------------
// OPFL parity
// ---------------------------------------------------------------------------

void test_opfl() {
  struct Case {
    int w;
    int h;
    int n;
    int oh;
    int oh0;
    int oh1;
  };
  const Case cases[] = {{8, 8, 8, 4, 2, 6},     {16, 8, 8, 4, 2, 6},
                        {16, 16, 8, 4, 6, 2},   {32, 16, 8, 8, 0, 8},
                        {32, 32, 8, 4, 0, 8},   {64, 64, 8, 4, 1, 7},
                        {8, 8, 4, 4, 2, 6},     {16, 16, 4, 4, 3, 6},
                        {32, 32, 4, 4, 0, 8},   {64, 32, 4, 4, 2, 6}};
  for (const Case& c : cases) {
    std::vector<std::uint8_t> p0(static_cast<std::size_t>(c.w) * c.h);
    std::vector<std::uint8_t> p1(p0.size());
    for (std::size_t i = 0; i < p0.size(); ++i) {
      p0[i] = static_cast<std::uint8_t>(random_range(0, 255));
      p1[i] = static_cast<std::uint8_t>(random_range(0, 255));
    }
    const int opt_x = (random_range(0, 1)) * c.n;
    const int opt_y = (random_range(0, 1)) * c.n;
    const int cols4 = c.w / 4;
    const int rows4 = opt_y / 4 + c.h / 4;
    const int stride4 = cols4 + 2;
    const std::size_t out_count =
        static_cast<std::size_t>(rows4) * stride4 * 4;
    std::vector<std::int16_t> mvs_gpu(out_count, 0);
    std::vector<std::int16_t> deltas_gpu(out_count, 0);
    std::vector<std::int16_t> mvs_cpu(out_count, 0);
    std::vector<std::int16_t> deltas_cpu(out_count, 0);
    std::vector<int> tmp(static_cast<std::size_t>(c.w) * c.h, 0);
    std::vector<int> pdiff(tmp.size(), 0);

    cf::OpflJob job{};
    job.w = c.w;
    job.h = c.h;
    job.bit_depth = 8;
    job.p0 = p0.data();
    job.p0_stride = c.w;
    job.p1 = p1.data();
    job.p1_stride = c.w;
    job.n = c.n;
    job.opt_x = opt_x;
    job.opt_y = opt_y;
    job.order_hint = c.oh;
    job.ref0_order_hint = c.oh0;
    job.ref1_order_hint = c.oh1;
    job.mv[0][0] = random_range(-16, 16) * 8;
    job.mv[0][1] = random_range(-16, 16) * 8;
    job.mv[1][0] = random_range(-16, 16) * 8;
    job.mv[1][1] = random_range(-16, 16) * 8;
    job.opfl_mvs = mvs_gpu.data();
    job.opfl_stride4 = stride4;
    job.mv_deltas = deltas_gpu.data();
    job.delta_stride4 = stride4;
    cf::detail::run_opfl_job_device(job, 0, 1, tmp.data(), pdiff.data());

    av2enc::inter::temporal::OpflJob cpu{};
    cpu.w = c.w;
    cpu.h = c.h;
    cpu.p0 = p0.data();
    cpu.p0_stride = c.w;
    cpu.p1 = p1.data();
    cpu.p1_stride = c.w;
    cpu.mvs[0] = Mv{job.mv[0][0], job.mv[0][1]};
    cpu.mvs[1] = Mv{job.mv[1][0], job.mv[1][1]};
    cpu.order_hint = c.oh;
    cpu.ref0_order_hint = c.oh0;
    cpu.ref1_order_hint = c.oh1;
    cpu.n = c.n;
    cpu.opt_x = opt_x;
    cpu.opt_y = opt_y;
    av2enc::inter::temporal::OpflOutput cpu_out{};
    cpu_out.opfl_mvs = mvs_cpu.data();
    cpu_out.opfl_stride4 = stride4;
    cpu_out.mv_deltas = deltas_cpu.data();
    cpu_out.delta_stride4 = stride4;
    EXPECT(av2enc::inter::temporal::opfl_compute(cpu, cpu_out) ==
           av2enc::inter::temporal::TemporalStatus::kOk);
    EXPECT(std::memcmp(mvs_gpu.data(), mvs_cpu.data(),
                       out_count * sizeof(std::int16_t)) == 0);
    EXPECT(std::memcmp(deltas_gpu.data(), deltas_cpu.data(),
                       out_count * sizeof(std::int16_t)) == 0);
  }
}

// ---------------------------------------------------------------------------
// Refine-MV parity
// ---------------------------------------------------------------------------

void test_refine() {
  const int plane = 80;
  std::vector<std::uint8_t> r0(static_cast<std::size_t>(plane) * plane);
  std::vector<std::uint8_t> r1(r0.size());
  for (std::size_t i = 0; i < r0.size(); ++i) {
    r0[i] = static_cast<std::uint8_t>(random_range(0, 255));
    r1[i] = static_cast<std::uint8_t>(random_range(0, 255));
  }
  const int blocks[][4] = {{8, 8, 8, 8},     {16, 16, 16, 16},
                           {0, 0, 32, 32},   {32, 32, 32, 32},
                           {48, 16, 16, 32}, {16, 48, 32, 16},
                           {0, 64, 16, 8},   {64, 0, 8, 16},
                           {24, 24, 40, 40}};
  for (const auto& b : blocks) {
    for (int use_area = 0; use_area < 2; ++use_area) {
      const int x = b[0];
      const int y = b[1];
      const int w = b[2];
      const int h = b[3];
      std::vector<std::uint8_t> scratch0(
          static_cast<std::size_t>(w + 4) * (h + 4));
      std::vector<std::uint8_t> scratch1(scratch0.size());
      RefineSearchInput in{};
      in.x = x;
      in.y = y;
      in.w = w;
      in.h = h;
      in.frame_w = plane;
      in.frame_h = plane;
      in.ref[0] = PlaneView{r0.data(), plane, plane, plane};
      in.ref[1] = PlaneView{r1.data(), plane, plane, plane};
      in.ref_mi_cols[0] = plane / 4;
      in.ref_mi_cols[1] = plane / 4;
      in.ref_mi_rows[0] = plane / 4;
      in.ref_mi_rows[1] = plane / 4;
      in.ref_luma_w[0] = plane;
      in.ref_luma_w[1] = plane;
      in.ref_luma_h[0] = plane;
      in.ref_luma_h[1] = plane;
      in.cand_mvs[0] = Mv{random_range(-16, 16) * 8, random_range(-16, 16) * 8};
      in.cand_mvs[1] = Mv{random_range(-16, 16) * 8, random_range(-16, 16) * 8};
      in.num_refs = 2;
      in.tip_pred = random_range(0, 1) != 0;
      in.switchable = random_range(0, 1) != 0;
      in.use_ref_area = use_area != 0;
      in.scratch0 = scratch0.data();
      in.scratch1 = scratch1.data();
      in.scratch_stride = w + 4;

      RefArea area[2]{};
      RefAreaInput area_in{};
      area_in.w = w;
      area_in.h = h;
      area_in.sub_x = 0;
      area_in.sub_y = 0;
      area_in.x = x;
      area_in.y = y;
      area_in.frame_w = plane;
      area_in.frame_h = plane;
      area_in.ref_mi_cols[0] = plane / 4;
      area_in.ref_mi_cols[1] = plane / 4;
      area_in.ref_mi_rows[0] = plane / 4;
      area_in.ref_mi_rows[1] = plane / 4;
      area_in.ref_luma_w[0] = plane;
      area_in.ref_luma_w[1] = plane;
      area_in.ref_luma_h[0] = plane;
      area_in.ref_luma_h[1] = plane;
      area_in.cand_mvs[0] = in.cand_mvs[0];
      area_in.cand_mvs[1] = in.cand_mvs[1];
      EXPECT(av2enc::inter::temporal::compute_ref_area(area_in, area) ==
             av2enc::inter::temporal::TemporalStatus::kOk);
      cf::RefAreaInput dev_area{};
      dev_area.w = w;
      dev_area.h = h;
      dev_area.x = x;
      dev_area.y = y;
      dev_area.frame_w = plane;
      dev_area.frame_h = plane;
      dev_area.ref_mi_cols[0] = plane / 4;
      dev_area.ref_mi_cols[1] = plane / 4;
      dev_area.ref_mi_rows[0] = plane / 4;
      dev_area.ref_mi_rows[1] = plane / 4;
      dev_area.ref_luma_w[0] = plane;
      dev_area.ref_luma_w[1] = plane;
      dev_area.ref_luma_h[0] = plane;
      dev_area.ref_luma_h[1] = plane;
      dev_area.cand_mv[0][0] = in.cand_mvs[0].row;
      dev_area.cand_mv[0][1] = in.cand_mvs[0].col;
      dev_area.cand_mv[1][0] = in.cand_mvs[1].row;
      dev_area.cand_mv[1][1] = in.cand_mvs[1].col;
      int area_dev[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
      cf::ref_area_compute(dev_area, area_dev);
      for (int list = 0; list < 2; ++list) {
        EXPECT(area_dev[list][0] == area[list].first_x);
        EXPECT(area_dev[list][1] == area[list].first_y);
        EXPECT(area_dev[list][2] == area[list].last_x);
        EXPECT(area_dev[list][3] == area[list].last_y);
        in.area[list] = area[list];
      }
      if (use_area == 0) {
        for (int list = 0; list < 2; ++list) {
          in.area[list] = RefArea{0, 0, plane - 1, plane - 1};
        }
      }

      RefineSearchResult cpu{};
      EXPECT(av2enc::inter::temporal::search_refinemv(in, &cpu) ==
             av2enc::inter::temporal::TemporalStatus::kOk);

      cf::RefineSearchJob job{};
      job.bit_depth = 8;
      job.x = x;
      job.y = y;
      job.w = w;
      job.h = h;
      job.frame_w = plane;
      job.frame_h = plane;
      job.ref[0] = cf::RefPlane{r0.data(), plane, plane, plane, 0};
      job.ref[1] = cf::RefPlane{r1.data(), plane, plane, plane, 0};
      job.cand_mv[0][0] = in.cand_mvs[0].row;
      job.cand_mv[0][1] = in.cand_mvs[0].col;
      job.cand_mv[1][0] = in.cand_mvs[1].row;
      job.cand_mv[1][1] = in.cand_mvs[1].col;
      job.num_refs = 2;
      job.tip_pred = in.tip_pred ? 1 : 0;
      job.switchable = in.switchable ? 1 : 0;
      job.use_ref_area = use_area;
      for (int list = 0; list < 2; ++list) {
        job.area[list][0] = in.area[list].first_x;
        job.area[list][1] = in.area[list].first_y;
        job.area[list][2] = in.area[list].last_x;
        job.area[list][3] = in.area[list].last_y;
      }
      cf::RefineResult gpu{};
      std::int64_t sad_shared[1] = {0};
      cf::detail::run_refine_search_device(job, &gpu, 0, 1, sad_shared);
      EXPECT(gpu.dx == cpu.dx);
      EXPECT(gpu.dy == cpu.dy);
      EXPECT(gpu.best_sad == cpu.best_sad);
      EXPECT(gpu.offset_mv[0][0] == cpu.offset_mvs.mv[0].row);
      EXPECT(gpu.offset_mv[0][1] == cpu.offset_mvs.mv[0].col);
      EXPECT(gpu.offset_mv[1][0] == cpu.offset_mvs.mv[1].row);
      EXPECT(gpu.offset_mv[1][1] == cpu.offset_mvs.mv[1].col);

      // Final prediction with the CPU-selected offset.
      std::vector<std::int16_t> p0scratch(static_cast<std::size_t>(w) * h);
      std::vector<std::int16_t> p1scratch(p0scratch.size());
      RefinePredictInput pin{};
      pin.ref[0] = in.ref[0];
      pin.ref[1] = in.ref[1];
      pin.num_refs = 2;
      pin.x = x;
      pin.y = y;
      pin.w = w;
      pin.h = h;
      pin.sub_x = 0;
      pin.sub_y = 0;
      pin.frame_w = plane;
      pin.frame_h = plane;
      pin.offset_mvs = cpu.offset_mvs;
      pin.cwp_weight = 8;
      pin.use_ref_area = in.use_ref_area;
      for (int list = 0; list < 2; ++list) {
        pin.area[list] = in.area[list];
      }
      std::vector<std::uint8_t> dst_cpu(static_cast<std::size_t>(w) * h);
      pin.dst = dst_cpu.data();
      pin.dst_stride = w;
      pin.scratch0 = p0scratch.data();
      pin.scratch1 = p1scratch.data();
      pin.scratch_stride = w;
      EXPECT(av2enc::inter::temporal::refine_predict_block(pin) ==
             av2enc::inter::temporal::TemporalStatus::kOk);

      cf::RefinePredictJob pjob{};
      pjob.bit_depth = 8;
      pjob.x = x;
      pjob.y = y;
      pjob.w = w;
      pjob.h = h;
      pjob.frame_w = plane;
      pjob.frame_h = plane;
      pjob.ref[0] = job.ref[0];
      pjob.ref[1] = job.ref[1];
      pjob.offset_mv[0][0] = cpu.offset_mvs.mv[0].row;
      pjob.offset_mv[0][1] = cpu.offset_mvs.mv[0].col;
      pjob.offset_mv[1][0] = cpu.offset_mvs.mv[1].row;
      pjob.offset_mv[1][1] = cpu.offset_mvs.mv[1].col;
      pjob.use_ref_area = use_area;
      for (int list = 0; list < 2; ++list) {
        pjob.area[list][0] = in.area[list].first_x;
        pjob.area[list][1] = in.area[list].first_y;
        pjob.area[list][2] = in.area[list].last_x;
        pjob.area[list][3] = in.area[list].last_y;
      }
      std::vector<std::uint8_t> dst_gpu(dst_cpu.size());
      pjob.dst = dst_gpu.data();
      pjob.dst_stride = w;
      cf::detail::run_refine_predict_device(pjob, 0, 1);
      EXPECT(std::memcmp(dst_gpu.data(), dst_cpu.data(), dst_cpu.size()) == 0);
    }
  }
}

// ---------------------------------------------------------------------------
// Arena + wedge cache
// ---------------------------------------------------------------------------

void test_arena() {
  cf::CompoundFullConfig config;
  config.max_jobs = 8;
  config.wedge_cache = 1;
  const std::uint64_t required = cf::required_device_bytes(config);
  EXPECT(required > 1900000u);
  EXPECT((required % 256) == 0);

  // Negative: one byte short must produce a distinct status and the exact
  // required size, with the arena rewound.
  av2enc::cuda::Arena small;
  std::string error;
  EXPECT(av2enc::cuda::is_ok(small.init(required - 1, 256, &error)));
  cf::CompoundFullWorkspace rejected;
  std::uint64_t reported = 0;
  EXPECT(cf::workspace_create(small, config, &rejected, &reported, &error) ==
         av2enc::cuda::Status::kArenaExhausted);
  EXPECT(reported == required);
  EXPECT(small.used() == 0);
  small.release();

  EXPECT(cf::workspace_create(g_arena, config, &g_workspace, &reported,
                              &error) == av2enc::cuda::Status::kOk);
  EXPECT(reported == required);
  EXPECT(g_workspace.required_bytes == required);
  EXPECT(g_arena.used() == required);
  EXPECT(g_arena.peak_used() == required);
  EXPECT(g_workspace.jobs != nullptr && g_workspace.skip_jobs != nullptr &&
         g_workspace.opfl_jobs != nullptr &&
         g_workspace.refine_search_jobs != nullptr &&
         g_workspace.refine_predict_jobs != nullptr &&
         g_workspace.refine_results != nullptr &&
         g_workspace.masks != nullptr && g_workspace.wedge_cache != nullptr);

  for (int slot = 0; slot < 16; ++slot) {
    cf::detail::wedge_cache_build_slot_device(g_workspace.wedge_cache, slot, 0,
                                              1);
  }
  g_wedge_cache = g_workspace.wedge_cache;
  for (int bsize : {3, 9, 12, 21, 28}) {
    const int w = cf::detail::kCfWedgeBlockWidth[bsize];
    const int h = cf::detail::kCfWedgeBlockHeight[bsize];
    for (int sign = 0; sign < 2; ++sign) {
      for (int index : {0, 17, 33, 67}) {
        for (int i : {0, h / 2, h - 1}) {
          for (int j : {0, w / 2, w - 1}) {
            const int expected =
                adv::wedge_mask_value(bsize, sign, index, i, j);
            EXPECT(cf::detail::wedge_cache_sample(g_workspace.wedge_cache,
                                                  bsize, sign, index, i, j) ==
                   expected);
          }
        }
      }
    }
  }

  // Two passes over the persistent workspace: no growth, and rewind returns to
  // the persistent mark.
  const std::uint64_t mark = g_arena.mark();
  EXPECT(mark == required);
  for (int pass = 0; pass < 2; ++pass) {
    EXPECT(g_arena.mark() == mark);
    EXPECT(g_arena.used() == required);
    EXPECT(g_arena.rewind(mark));
    EXPECT(g_arena.used() == required);
  }
  EXPECT(g_arena.peak_used() == required);
  std::printf("arena: required=%llu bytes, used=%llu, peak=%llu\n",
              static_cast<unsigned long long>(required),
              static_cast<unsigned long long>(g_arena.used()),
              static_cast<unsigned long long>(g_arena.peak_used()));
}

}  // namespace

int main() {
  cf::CompoundFullConfig config;
  config.max_jobs = 8;
  const std::uint64_t required = cf::required_device_bytes(config);
  std::string error;
  if (!av2enc::cuda::is_ok(
          g_arena.init(required + 65536, 256, &error))) {
    std::printf("compound_full_emulation: arena init failed: %s\n",
                error.c_str());
    return 2;
  }
  test_arena();
  int before = g_checks;
  test_blend();
  std::printf("blend checks: %d\n", g_checks - before);
  before = g_checks;
  test_skip();
  std::printf("skip checks: %d\n", g_checks - before);
  before = g_checks;
  test_opfl();
  std::printf("opfl checks: %d\n", g_checks - before);
  before = g_checks;
  test_refine();
  std::printf("refine checks: %d\n", g_checks - before);
  if (g_failures == 0) {
    std::printf("compound_full_emulation: %d checks passed\n", g_checks);
    return 0;
  }
  std::printf("compound_full_emulation: %d/%d checks FAILED\n", g_failures,
              g_checks);
  return 1;
}
