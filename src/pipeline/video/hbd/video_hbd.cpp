// Format video core: CPU reference + shared orchestration.
//
// The orchestration owns every decision (partition tree, MV stack, candidate
// comparison, syntax).  The backend only performs pixel/transform work, so the
// CUDA backend (`src/cuda/pipeline/video/hbd/`) is decision-identical by
// construction and the two paths can be compared byte for byte.
#include "pipeline/video/hbd/video_hbd.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

#include "codec/inter/merge/merge.h"
#include "codec/partition/tree/partition.h"
#include "codec/transform/variants/tx_variants.h"
#include "cuda/pipeline/still/hbd/hbd_cpu.h"
#include "formats/high_bitdepth/hbd_depth.h"
#include "pipeline/video/hbd/video_hbd_filters.h"
#include "pipeline/video/hbd/video_hbd_ops.h"
#include "pipeline/grain/grain_stream.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_syntax.h"

namespace av2enc::video::hbd {
namespace {

using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionNode;
namespace grain = ::av2enc::filters::grain;

constexpr std::int8_t kRefIntraAv2 = 8;
constexpr std::int8_t kRefNone = -1;

int plane_shift_x(const fmt::Format& format, int plane) {
  return plane == 0 ? 0 : fmt::subsampling_x(format.chroma);
}
int plane_shift_y(const fmt::Format& format, int plane) {
  return plane == 0 ? 0 : fmt::subsampling_y(format.chroma);
}

struct Plane {
  int width = 0;
  int height = 0;
  int stride = 0;
  std::vector<std::uint16_t> samples;

  void resize(int w, int h) {
    width = w;
    height = h;
    stride = w;
    samples.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h),
                   0);
  }
  std::uint16_t& at(int x, int y) {
    return samples[static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(stride) +
                   static_cast<std::size_t>(x)];
  }
  std::uint16_t at(int x, int y) const {
    return samples[static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(stride) +
                   static_cast<std::size_t>(x)];
  }
};

bool make_block_geometry(const fmt::Format& format, int luma_w, int luma_h,
                         int* cw, int* ch) {
  if (!fmt::has_chroma(format.chroma)) {
    *cw = 0;
    *ch = 0;
    return true;
  }
  const auto size = fmt::chroma_residual_size(luma_w, luma_h, format.chroma);
  if (!size.has_value()) return false;
  *cw = fmt::block_width(*size);
  *ch = fmt::block_height(*size);
  return true;
}

// §7.14.3 / §7.15.4.1 block transform for the format depth.  Mirrors the still
// path's `compute_tx` (forward quantiser from txv, depth-aware dequant and
// inverse from `hbd`).
bool transform_block(const std::uint16_t* src, int src_stride, int src_w,
                     int src_h, int x0, int y0, const std::uint16_t* pred,
                     int w, int h, int bit_depth, int quant_step,
                     PlaneTx& out) {
  auto params_opt = txv::make_tx_params(w, h, txv::TxType::kDctDct);
  if (!params_opt.has_value()) return false;
  txv::TxParams params = *params_opt;
  if (quant_step != txv::kQuantStep) {
    params.quant_factor *= static_cast<double>(txv::kQuantStep) /
                           static_cast<double>(quant_step);
  }
  out.tx_size = params.tx_size;
  out.adj_w = params.w;
  out.adj_h = params.h;

  const int cw = params.coded_w;
  const int ch = params.coded_h;
  const int coded_pels = cw * ch;
  std::vector<std::int16_t> residual(static_cast<std::size_t>(coded_pels));
  for (int y = 0; y < ch; ++y) {
    const int sy = std::clamp(y0 + y, 0, src_h - 1);
    for (int x = 0; x < cw; ++x) {
      const int sx = std::clamp(x0 + x, 0, src_w - 1);
      const int s = static_cast<int>(
          src[static_cast<std::size_t>(sy) * static_cast<std::size_t>(src_stride) +
              static_cast<std::size_t>(sx)]);
      const int p = static_cast<int>(
          pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
               static_cast<std::size_t>(x)]);
      residual[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
               static_cast<std::size_t>(x)] =
          static_cast<std::int16_t>(s - p);
    }
  }

  std::vector<std::int16_t> residual_adj(
      static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
  txv::downsample_residual(residual.data(), params, residual_adj.data());
  std::vector<double> coeff(
      static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
  out.levels.assign(coeff.size(), 0);
  txv::forward_transform(residual_adj.data(), params, coeff.data(),
                         out.levels.data());

  const int area = params.w * params.h;
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  av2enc::hbd::build_scan(params.w, params.h, scan.data());
  out.eob = 0;
  for (int c = area - 1; c >= 0; --c) {
    if (out.levels[scan[static_cast<std::size_t>(c)]] != 0) {
      out.eob = c + 1;
      break;
    }
  }

  std::vector<std::int32_t> dequant(static_cast<std::size_t>(area), 0);
  for (int i = 0; i < area; ++i) {
    const std::int32_t qc = out.levels[static_cast<std::size_t>(i)];
    const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
    const std::int64_t high = (aq * quant_step) & 0xFFFFFF;
    const std::int32_t dq =
        txv::round2(static_cast<std::int32_t>(high), txv::kQuantTableBits) /
        static_cast<std::int32_t>(params.dq_denom);
    const std::int32_t value = qc < 0 ? -dq : dq;
    dequant[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(
        std::clamp<std::int64_t>(value, fmt::bit_depth_limits(
                                             static_cast<fmt::BitDepth>(
                                                 bit_depth))
                                             .dequant_min,
                                 fmt::bit_depth_limits(
                                     static_cast<fmt::BitDepth>(bit_depth))
                                     .dequant_max));
  }
  std::vector<std::int16_t> recon_residual(
      static_cast<std::size_t>(coded_pels));
  av2enc::hbd::inverse_block(dequant.data(), params, recon_residual.data(),
                             bit_depth);
  out.recon.resize(static_cast<std::size_t>(coded_pels));
  for (int y = 0; y < ch; ++y) {
    for (int x = 0; x < cw; ++x) {
      const int p = static_cast<int>(
          pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
               static_cast<std::size_t>(x)]);
      const int r = static_cast<int>(
          recon_residual[static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(cw) +
                         static_cast<std::size_t>(x)]);
      out.recon[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(x)] =
          static_cast<std::uint16_t>(clip1_value(p + r, bit_depth));
    }
  }
  return true;
}

double estimate_tx_bits(const PlaneTx& plan, bool is_inter_luma) {
  if (plan.eob == 0) return 1.0;
  double bits = 8.0;
  if (is_inter_luma) bits += 4.0;
  int nonzero = 0;
  for (int c = 0; c < plan.eob; ++c) {
    if (plan.levels[static_cast<std::size_t>(c)] != 0) ++nonzero;
  }
  bits += static_cast<double>(nonzero) * 2.5 + plan.eob * 0.5;
  return bits;
}

// Fills the syntax node's luma plan (TX_MODE_LARGEST single DCT) and appends
// scan-order levels.
void write_luma_plan(syntax::BlockSyntax& entry, const PlaneTx& plan,
                     std::vector<std::int32_t>& levels) {
  entry.tx_type[0] = 0;
  entry.tx_size[0] = static_cast<std::uint8_t>(plan.tx_size);
  entry.eob[0] = static_cast<std::uint32_t>(plan.eob);
  entry.tx_partition = 0;
  entry.tx_part_count = 0;
  const int area = plan.adj_w * plan.adj_h;
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  av2enc::hbd::build_scan(plan.adj_w, plan.adj_h, scan.data());
  for (int c = 0; c < plan.eob; ++c) {
    levels.push_back(plan.levels[scan[static_cast<std::size_t>(c)]]);
  }
}

void write_chroma_plan(syntax::BlockSyntax& entry, int plane,
                       const PlaneTx& plan, std::vector<std::int32_t>& levels) {
  entry.tx_size[plane] = static_cast<std::uint8_t>(plan.tx_size);
  entry.eob[plane] = static_cast<std::uint32_t>(plan.eob);
  const int area = plan.adj_w * plan.adj_h;
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  av2enc::hbd::build_scan(plan.adj_w, plan.adj_h, scan.data());
  for (int c = 0; c < plan.eob; ++c) {
    levels.push_back(plan.levels[scan[static_cast<std::size_t>(c)]]);
  }
}

// ---------------------------------------------------------------------------
// CPU pixel backend
// ---------------------------------------------------------------------------

class CpuBackend final : public Backend {
 public:
  const char* name() const override { return "cpu"; }

  bool begin_frame(const BeginFrame& frame, std::string& error) override {
    (void)error;
    frame_ = frame;
    format_ = frame.format;
    depth_ = static_cast<int>(format_.depth);
    quant_step_ = av2enc::hbd::depth::qlookup(frame.base_q_idx);
    return true;
  }

  bool end_frame(std::string&) override { return true; }

  bool variance_costs(const std::vector<VarianceNode>& nodes,
                      std::vector<std::int64_t>& costs,
                      std::string&) override {
    costs.resize(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      costs[i] = hbd_variance_cost(frame_.source.plane[0].data,
                                   frame_.source.plane[0].stride, nodes[i].x,
                                   nodes[i].y, nodes[i].w, nodes[i].h);
    }
    return true;
  }

  bool me_batch(const std::vector<MeJob>& jobs, std::vector<Mv>& mvs,
                std::string& error) override {
    mvs.resize(jobs.size());
    const PlaneView& cur = frame_.source.plane[0];
    const int stages = 1 | 2 | (options_.me_subpel_eighth ? 4 : 0);
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      const MeJob& job = jobs[i];
      if (job.ref_index < 0 || job.ref_index >= frame_.num_refs) {
        error = "ME reference index out of range";
        return false;
      }
      const PlaneView& ref = frame_.refs[job.ref_index].plane[0];
      if (ref.data == nullptr) {
        error = "ME reference plane is missing";
        return false;
      }
      int mv_y = 0;
      int mv_x = 0;
      hbd_me_search(cur.data, cur.stride, ref.data, ref.stride, job.x, job.y,
                    job.w, job.h, ref.width, ref.height, options_.me_range,
                    stages, depth_, &mv_y, &mv_x);
      mvs[i] = Mv{mv_y, mv_x};
    }
    return true;
  }

  bool inter_sse_batch(const std::vector<InterJob>& jobs,
                       std::vector<std::int64_t>& sse,
                       std::string& error) override {
    sse.resize(jobs.size());
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      std::vector<std::uint16_t> y;
      if (!predict_luma(jobs[i], y, error)) return false;
      const PlaneView& src = frame_.source.plane[0];
      std::int64_t total = 0;
      for (int r = 0; r < jobs[i].h; ++r) {
        for (int c = 0; c < jobs[i].w; ++c) {
          const int s = static_cast<int>(
              src.data[static_cast<std::size_t>(jobs[i].y + r) *
                           static_cast<std::size_t>(src.stride) +
                       static_cast<std::size_t>(jobs[i].x + c)]);
          const int p = static_cast<int>(
              y[static_cast<std::size_t>(r) * static_cast<std::size_t>(jobs[i].w) +
                 static_cast<std::size_t>(c)]);
          const int d = s - p;
          total += static_cast<std::int64_t>(d) * d;
        }
      }
      sse[i] = total;
    }
    return true;
  }

  bool mc_420(const InterJob& job, std::vector<std::uint16_t> out[3],
              std::string& error) override {
    return predict_all(job, out, error);
  }

  bool intra_candidate(int x, int y, int w, int h,
                       std::array<PlaneTx, 3>& planes,
                       std::string& error) override {
    int cw = 0;
    int ch = 0;
    if (!make_block_geometry(format_, w, h, &cw, &ch)) {
      error = "invalid chroma residual size";
      return false;
    }
    const int sizes[3][2] = {{w, h}, {cw, ch}, {cw, ch}};
    const int plane_count = fmt::has_chroma(format_.chroma) ? 3 : 1;
    for (int plane = 0; plane < plane_count; ++plane) {
      const int pw = sizes[plane][0];
      const int ph = sizes[plane][1];
      const int px = x >> plane_shift_x(format_, plane);
      const int py = y >> plane_shift_y(format_, plane);
      const PlaneView& recon = frame_.recon.plane[plane];
      const PlaneView& src = frame_.source.plane[plane];

      std::vector<std::uint16_t> above(static_cast<std::size_t>(pw));
      std::vector<std::uint16_t> left(static_cast<std::size_t>(ph));
      const bool have_above = py > 0;
      const bool have_left = px > 0;
      // §7.13.2.1 clamps edge reads to the coded plane (maxX/maxY).  A partial
      // bottom/right coding block can extend past MiRows/MiCols, so the
      // neighbour rows/columns past the plane must not be read.
      const int max_x = recon.width - 1;
      const int max_y = recon.height - 1;
      for (int i = 0; i < pw; ++i) {
        const int sx = std::min(max_x, px + i);
        above[static_cast<std::size_t>(i)] =
            have_above ? recon.data[static_cast<std::size_t>(py - 1) *
                                         static_cast<std::size_t>(recon.stride) +
                                     static_cast<std::size_t>(sx)]
                       : 0;
      }
      for (int i = 0; i < ph; ++i) {
        const int sy = std::min(max_y, py + i);
        left[static_cast<std::size_t>(i)] =
            have_left ? recon.data[static_cast<std::size_t>(sy) *
                                        static_cast<std::size_t>(recon.stride) +
                                    static_cast<std::size_t>(px - 1)]
                      : 0;
      }
      std::vector<std::uint16_t> pred(static_cast<std::size_t>(pw) *
                                      static_cast<std::size_t>(ph));
      hbd_dc_predict(above.data(), left.data(), pw, ph, have_above, have_left,
                     depth_, pred.data(), pw);
      if (!transform_block(src.data, src.stride, src.width, src.height, px, py,
                           pred.data(), pw, ph, depth_, quant_step_,
                           planes[static_cast<std::size_t>(plane)])) {
        error = "intra transform failed";
        return false;
      }
    }
    return true;
  }

  bool inter_residual(const InterJob& job,
                      std::array<PlaneTx, 3>& planes,
                      std::string& error) override {
    std::vector<std::uint16_t> pred[3];
    if (!predict_all(job, pred, error)) return false;
    int cw = 0;
    int ch = 0;
    if (!make_block_geometry(format_, job.w, job.h, &cw, &ch)) {
      error = "invalid chroma residual size";
      return false;
    }
    const int sizes[3][2] = {{job.w, job.h}, {cw, ch}, {cw, ch}};
    const int plane_count = fmt::has_chroma(format_.chroma) ? 3 : 1;
    for (int plane = 0; plane < plane_count; ++plane) {
      const int px = job.x >> plane_shift_x(format_, plane);
      const int py = job.y >> plane_shift_y(format_, plane);
      const PlaneView& src = frame_.source.plane[plane];
      if (!transform_block(src.data, src.stride, src.width, src.height, px, py,
                           pred[plane].data(), sizes[plane][0],
                           sizes[plane][1], depth_, quant_step_,
                           planes[static_cast<std::size_t>(plane)])) {
        error = "inter residual transform failed";
        return false;
      }
    }
    return true;
  }

  bool write_recon(int x, int y, int w, int h,
                   const std::vector<std::uint16_t>& y_plane,
                   const std::vector<std::uint16_t>& u_plane,
                   const std::vector<std::uint16_t>& v_plane,
                   std::string& error) override {
    int cw = 0;
    int ch = 0;
    if (!make_block_geometry(format_, w, h, &cw, &ch)) {
      error = "invalid chroma residual size";
      return false;
    }
    const std::vector<std::uint16_t>* planes[3] = {&y_plane, &u_plane,
                                                   &v_plane};
    const int sizes[3][2] = {{w, h}, {cw, ch}, {cw, ch}};
    const int plane_count = fmt::has_chroma(format_.chroma) ? 3 : 1;
    for (int plane = 0; plane < plane_count; ++plane) {
      const int px = x >> plane_shift_x(format_, plane);
      const int py = y >> plane_shift_y(format_, plane);
      const PlaneView& recon = frame_.recon.plane[plane];
      const int pw = sizes[plane][0];
      const int ph = sizes[plane][1];
      const std::vector<std::uint16_t>& src = *planes[plane];
      for (int r = 0; r < ph; ++r) {
        const int ry = py + r;
        if (ry >= recon.height) break;
        for (int c = 0; c < pw; ++c) {
          const int rx = px + c;
          if (rx >= recon.width) break;
          const_cast<std::uint16_t*>(recon.data)[
              static_cast<std::size_t>(ry) *
                  static_cast<std::size_t>(recon.stride) +
              static_cast<std::size_t>(rx)] =
              src[static_cast<std::size_t>(r) * static_cast<std::size_t>(pw) +
                  static_cast<std::size_t>(c)];
        }
      }
    }
    return true;
  }

  void set_options(const Options& options) override { options_ = options; }

 private:
  bool predict_luma(const InterJob& job, std::vector<std::uint16_t>& out,
                    std::string& error) {
    if (job.ref_index < 0 || job.ref_index >= frame_.num_refs) {
      error = "inter reference index out of range";
      return false;
    }
    const PlaneView& ref = frame_.refs[job.ref_index].plane[0];
    if (ref.data == nullptr) {
      error = "inter reference plane is missing";
      return false;
    }
    out.resize(static_cast<std::size_t>(job.w) *
               static_cast<std::size_t>(job.h));
    hbd_mc_block(ref.data, ref.stride, 0, 0, ref.width - 1, ref.height - 1,
                 job.mv_y, job.mv_x, 0, 0, depth_, job.x, job.y, job.w, job.h,
                 out.data(), job.w);
    return true;
  }

  bool predict_all(const InterJob& job, std::vector<std::uint16_t> out[3],
                   std::string& error) {
    int cw = 0;
    int ch = 0;
    if (!make_block_geometry(format_, job.w, job.h, &cw, &ch)) {
      error = "invalid chroma residual size";
      return false;
    }
    if (job.ref_index < 0 || job.ref_index >= frame_.num_refs) {
      error = "inter reference index out of range";
      return false;
    }
    const int sizes[3][2] = {{job.w, job.h}, {cw, ch}, {cw, ch}};
    const int plane_count = fmt::has_chroma(format_.chroma) ? 3 : 1;
    for (int plane = 0; plane < plane_count; ++plane) {
      const PlaneView& ref = frame_.refs[job.ref_index].plane[plane];
      if (ref.data == nullptr) {
        error = "inter reference plane is missing";
        return false;
      }
      const int pw = sizes[plane][0];
      const int ph = sizes[plane][1];
      out[plane].resize(static_cast<std::size_t>(pw) *
                        static_cast<std::size_t>(ph));
      hbd_mc_block(ref.data, ref.stride, 0, 0, ref.width - 1, ref.height - 1,
                   job.mv_y, job.mv_x, plane_shift_x(format_, plane),
                   plane_shift_y(format_, plane), depth_,
                   job.x >> plane_shift_x(format_, plane),
                   job.y >> plane_shift_y(format_, plane), pw, ph,
                   out[plane].data(), pw);
    }
    return true;
  }

  BeginFrame frame_{};
  fmt::Format format_{};
  int depth_ = 8;
  int quant_step_ = txv::kQuantStep;
  Options options_{};
};

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

int relative_dist(std::uint32_t a, std::uint32_t b, int bits) {
  const int modulus = 1 << bits;
  int diff =
      static_cast<int>((a - b) & static_cast<std::uint32_t>(modulus - 1));
  if (diff >= (modulus >> 1)) diff -= modulus;
  return diff;
}

Mv quantize_mv(Mv mv, int precision) {
  const int quantum = precision == 1 ? 2 : (precision == 2 ? 4 : 1);
  if (quantum == 1) return mv;
  const auto round_component = [quantum](int value) {
    const int magnitude = value < 0 ? -value : value;
    const int rounded = ((magnitude + quantum / 2) / quantum) * quantum;
    return value < 0 ? -rounded : rounded;
  };
  return Mv{round_component(mv.y), round_component(mv.x)};
}

struct RefSlot {
  bool valid = false;
  std::uint32_t order_hint = 0;
  Plane plane[3];
};

struct InterCandidate {
  bool valid = false;
  int mode = syntax::kInterGlobalMv;
  int drl_index = 0;
  int ref_index = 0;
  Mv mv{};
  Mv mvd{};
  std::int64_t sse = 0;
};

class FrameEncoder {
 public:
  FrameEncoder(const VideoInputFrame& source, const std::vector<RefSlot>& refs,
               std::span<const int> frame_distance, int num_total_refs,
               const Options& options, Backend& backend, bool keyframe)
      : source_(source), refs_(refs), frame_distance_(frame_distance),
        num_total_refs_(num_total_refs), options_(options), backend_(backend),
        keyframe_(keyframe) {
    format_ = source.format;
    depth_ = static_cast<int>(format_.depth);
    width_ = static_cast<int>(source.width);
    height_ = static_cast<int>(source.height);
    geometry_.width = width_;
    geometry_.height = height_;
    geometry_.sb_size = 64;
    // The partition module derives its allowed set from the chroma geometry
    // (CHROMA_PART / Subsampled_Size).  Keep it in sync with the signalled
    // format; a wrong mask changes the partition symbol the writer emits.
    // 4:2:2 is modelled by partition::chroma_plane_residual_size(), so the
    // tree is not luma-only and 16-pixel 4:2:2 leaves decode exactly.
    geometry_.has_chroma = fmt::has_chroma(format_.chroma);
    geometry_.chroma_subsampling_x = fmt::subsampling_x(format_.chroma) != 0;
    geometry_.chroma_subsampling_y = fmt::subsampling_y(format_.chroma) != 0;
    mi_cols_ = partition::mi_cols(geometry_);
    mi_rows_ = partition::mi_rows(geometry_);
    quant_step_ = av2enc::hbd::depth::qlookup(options_.base_q_idx);
    mv_grid_.assign(static_cast<std::size_t>(mi_rows_) *
                        static_cast<std::size_t>(mi_cols_),
                    inter::mc::MiInfo{});
  }

  bool encode(std::vector<std::uint8_t>& tile_payload,
              std::vector<syntax::BlockSyntax>& blocks,
              std::vector<std::int32_t>& levels,
              post::VideoFilterPlan* filters,
              std::uint16_t* const recon_samples[3],
              const int recon_strides[3], std::string& error) {
    partition::DecisionOptions poptions;
    poptions.min_block_size_px = options_.min_block_size;
    poptions.max_depth = options_.max_partition_depth;
    poptions.allow_rect = true;
    poptions.allow_split = false;
    if (options_.force_single_block) {
      poptions.max_depth = 0;
      poptions.min_block_size_px = 64;
    }
    const double lambda =
        0.85 * std::pow(2.0, (static_cast<double>(options_.base_q_idx) - 12.0) /
                                 6.0);
    const double signal_cost = lambda * 2.0;
    const auto cost_fn = [this, signal_cost](
                             const partition::PartitionNodeInfo& info) {
      partition::NodeCosts costs;
      costs.leaf = static_cast<double>(block_variance_cost(info));
      costs.none_signal = 0.0;
      costs.split_signal = signal_cost;
      costs.horz_signal = signal_cost;
      costs.vert_signal = signal_cost;
      return costs;
    };
    const partition::FrameDecision frame =
        partition::decide_frame(geometry_, cost_fn, poptions);
    if (frame.status != partition::DecisionStatus::kOk) {
      error = "partition decision failed: " + frame.error;
      return false;
    }
    if (!partition::validate_frame(geometry_, frame, error)) {
      error = "partition tree invalid: " + error;
      return false;
    }
    blocks_.clear();
    levels_.clear();
    for (const PartitionNode& sb : frame.superblocks) {
      if (!collect_node(sb, error)) return false;
    }

    // §7.17-§7.20.5 loop filters run on the completed reconstruction, before
    // it becomes the output picture / reference.  Deblock is enabled for
    // every frame; the remaining tools are planned on the intra key frame.
    *filters = post::VideoFilterPlan{};
    if (options_.enable_deblock ||
        (keyframe_ && options_.loop_filters.any())) {
      int visible_w[3] = {width_, width_, width_};
      int visible_h[3] = {height_, height_, height_};
      const int sub_x = fmt::subsampling_x(format_.chroma);
      const int sub_y = fmt::subsampling_y(format_.chroma);
      visible_w[1] = visible_w[2] = static_cast<int>(width_) >> sub_x;
      visible_h[1] = visible_h[2] = static_cast<int>(height_) >> sub_y;
      std::uint16_t* const* recon_ptr = recon_samples;
      const int* recon_stride = recon_strides;
      FramePlanes source;
      source.plane[0] = PlaneView{source_.y16.data(), width_, width_, height_};
      if (fmt::has_chroma(format_.chroma)) {
        source.plane[1] =
            PlaneView{source_.u16.data(), visible_w[1], visible_w[1],
                      visible_h[1]};
        source.plane[2] =
            PlaneView{source_.v16.data(), visible_w[2], visible_w[2],
                      visible_h[2]};
      }
      if (!apply_frame_filters(
              format_, mi_cols_, mi_rows_, visible_w, visible_h, recon_ptr,
              recon_stride, source, blocks_, options_.base_q_idx, keyframe_,
              options_.enable_deblock, options_.loop_filters, filters,
              error)) {
        return false;
      }
    }

    syntax::FramePlan plan;
    plan.width = width_;
    plan.height = height_;
    plan.sb_size = 64;
    plan.inter_frame = !keyframe_;
    plan.tx_mode_select = false;
    plan.reduced_tx_part_set = false;
    plan.reduced_tx_set = 0;
    plan.num_total_refs = num_total_refs_;
    plan.reference_select = false;
    plan.mv_precision = options_.mv_precision;
    plan.drl_reorder = 0;
    for (int r = 0; r < 7 && r < static_cast<int>(frame_distance_.size()); ++r) {
      plan.frame_distance[static_cast<std::size_t>(r)] =
          frame_distance_[static_cast<std::size_t>(r)];
    }

    VideoConfig config;
    config.max_frame_width = static_cast<std::uint16_t>(width_);
    config.max_frame_height = static_cast<std::uint16_t>(height_);
    config.chroma = format_.chroma;
    config.depth = format_.depth;
    config.base_q_idx = options_.base_q_idx;
    config.order_hint_bits = 4;
    config.num_ref_frames = 2;
    config.num_total_refs = 1;
    config.max_drl_bits_minus_1 = 4;
    config.disable_cdf_update = true;
    config.enable_avg_cdf = false;
    config.enable_cfl_intra = true;
    config.cfl_ds_filter_index = options_.loop_filters.cfl_ds_filter_index;
    config.enable_deblock = options_.enable_deblock;
    config.tx_mode_select = false;

    std::size_t tile_bytes = 0;
    const post::VideoFilterPlan* filter_plan =
        filters->any() ? filters : nullptr;
    if (!syntax::assemble_video_frame(plan, blocks_, levels_, config,
                                      tile_payload, error, &tile_bytes, {},
                                      nullptr, filter_plan)) {
      return false;
    }
    blocks = std::move(blocks_);
    levels = std::move(levels_);
    return true;
  }

  const Plane& recon(int plane) const { return recon_[plane]; }
  std::uint32_t inter_blocks() const { return inter_blocks_; }
  std::uint32_t intra_blocks() const { return intra_blocks_; }
  int mi_cols() const { return mi_cols_; }
  int mi_rows() const { return mi_rows_; }

 private:
  std::int64_t block_variance_cost(
      const partition::PartitionNodeInfo& info) const {
    const partition::BlockRect rect =
        partition::block_rect(geometry_, info.mi_row, info.mi_col, info.size);
    return hbd_variance_cost(source_.y16.data(), width_, rect.x, rect.y,
                             rect.visible_width, rect.visible_height);
  }

  bool collect_node(const PartitionNode& node, std::string& error) {
    const std::size_t index = blocks_.size();
    blocks_.emplace_back();
    syntax::BlockSyntax& entry = blocks_.back();
    entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
    entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
    entry.bsize = static_cast<std::uint8_t>(node.size);
    entry.partition = static_cast<std::uint8_t>(node.partition);
    std::uint8_t mask = 0;
    for (int p = 0; p < partition::kPartitionTypeCount; ++p) {
      if (node.allowed.allowed[p]) mask = static_cast<std::uint8_t>(mask | (1U << p));
    }
    entry.allowed_mask = mask;
    if (node.forced) entry.flags |= syntax::kFlagForced;
    if (node.depth == 0) entry.flags |= syntax::kFlagSbRoot;
    if (node.partition == partition::PartitionType::kNone) {
      if (!encode_leaf(node, index, error)) return false;
    } else {
      for (const PartitionNode& child : node.children) {
        if (!collect_node(child, error)) return false;
      }
    }
    const std::size_t subtree = blocks_.size() - index - 1;
    if (subtree > 0xFFFFU) {
      error = "partition subtree too large";
      return false;
    }
    blocks_[index].subtree = static_cast<std::uint16_t>(subtree);
    return true;
  }

  merge::StackParams make_stack_params(int mi_row, int mi_col, int w4, int h4,
                                       int ref_index) const {
    merge::StackParams params;
    params.mi_row = mi_row;
    params.mi_col = mi_col;
    params.bw4 = w4;
    params.bh4 = h4;
    params.sb_h4 = 16;
    params.mi_cols = mi_cols_;
    params.mi_rows = mi_rows_;
    params.mi = mv_grid_.data();
    params.ref_frame[0] = static_cast<std::int8_t>(ref_index);
    params.ref_frame[1] = merge::kRefNone;
    params.is_compound = false;
    params.global_mv[0] = Mv{0, 0};
    params.global_mv[1] = Mv{0, 0};
    params.reorder = merge::DrlReorder::kDisabled;
    params.max_drl_bits_minus_1 = 4;
    if (num_total_refs_ > 1) {
      params.num_refs = num_total_refs_;
      params.frame_distance = frame_distance_.data();
    }
    return params;
  }

  bool load_stack(int mi_row, int mi_col, int w4, int h4, int ref_index,
                  merge::StackParams& params, merge::MvStack& stack) const {
    params = make_stack_params(mi_row, mi_col, w4, h4, ref_index);
    return merge::build_stack_checked(params, &stack) == merge::Status::kOk;
  }

  void update_mv_grid(int r, int c, int w4, int h4, int is_inter,
                      int ref_index, Mv mv) {
    for (int yy = 0; yy < h4; ++yy) {
      const int rr = r + yy;
      if (rr >= mi_rows_) break;
      for (int xx = 0; xx < w4; ++xx) {
        const int cc = c + xx;
        if (cc >= mi_cols_) break;
        inter::mc::MiInfo& mi =
            mv_grid_[static_cast<std::size_t>(rr) *
                         static_cast<std::size_t>(mi_cols_) +
                     static_cast<std::size_t>(cc)];
        mi = inter::mc::MiInfo{};
        mi.ref[0] = is_inter ? static_cast<std::int8_t>(ref_index)
                             : kRefIntraAv2;
        mi.ref[1] = kRefNone;
        mi.mv[0][0] = mv.y;
        mi.mv[0][1] = mv.x;
        mi.mv[1][0] = mv.y;
        mi.mv[1][1] = mv.x;
        mi.is_inter = static_cast<std::int8_t>(is_inter);
        mi.w4 = static_cast<std::int8_t>(w4);
        mi.h4 = static_cast<std::int8_t>(h4);
        mi.col_base = static_cast<std::int16_t>(c);
        mi.row_base = static_cast<std::int16_t>(r);
      }
    }
  }

  std::int64_t sse_against_source(int plane, int x0, int y0, int w, int h,
                                  const std::vector<std::uint16_t>& samples) {
    const PlaneView view = plane == 0
                               ? PlaneView{source_.y16.data(), width_, width_,
                                           height_}
                               : (plane == 1
                                      ? PlaneView{source_.u16.data(),
                                                  (width_ >> fmt::subsampling_x(
                                                                format_.chroma)),
                                                  (width_ >> fmt::subsampling_x(
                                                                format_.chroma)),
                                                  (height_ >> fmt::subsampling_y(
                                                                 format_.chroma))}
                                      : PlaneView{source_.v16.data(),
                                                  (width_ >> fmt::subsampling_x(
                                                                format_.chroma)),
                                                  (width_ >> fmt::subsampling_x(
                                                                format_.chroma)),
                                                  (height_ >> fmt::subsampling_y(
                                                                 format_.chroma))});
    std::int64_t sse = 0;
    for (int r = 0; r < h; ++r) {
      const int sy = std::clamp(y0 + r, 0, view.height - 1);
      for (int c = 0; c < w; ++c) {
        const int sx = std::clamp(x0 + c, 0, view.width - 1);
        const int s = static_cast<int>(
            view.data[static_cast<std::size_t>(sy) *
                          static_cast<std::size_t>(view.stride) +
                      static_cast<std::size_t>(sx)]);
        const int p = static_cast<int>(
            samples[static_cast<std::size_t>(r) * static_cast<std::size_t>(w) +
                    static_cast<std::size_t>(c)]);
        const int d = s - p;
        sse += static_cast<std::int64_t>(d) * d;
      }
    }
    return sse;
  }

  bool encode_leaf(const PartitionNode& node, std::size_t index,
                   std::string& error) {
    const int r = node.mi_row;
    const int c = node.mi_col;
    const int w = partition::block_width(node.size);
    const int h = partition::block_height(node.size);
    const int w4 = w >> 2;
    const int h4 = h >> 2;
    const int x0 = c * 4;
    const int y0 = r * 4;
    int cw = 0;
    int ch = 0;
    if (!make_block_geometry(format_, w, h, &cw, &ch)) {
      error = "invalid chroma residual size for a coded leaf";
      return false;
    }
    syntax::BlockSyntax& entry = blocks_[index];
    if (options_.trace) {
      std::cerr << "hbd leaf r=" << r << " c=" << c << " size=" << w << "x" << h
                << " key=" << (keyframe_ ? 1 : 0) << "\n";
    }

    const bool inter_fits = (x0 + w <= mi_cols_ * 4 && y0 + h <= mi_rows_ * 4);

    InterCandidate best;
    const auto consider = [&](InterCandidate cand) {
      if (!cand.valid) return;
      if (!best.valid || cand.sse < best.sse) best = cand;
    };

    if (!keyframe_ && inter_fits) {
      for (int ref_index = 0; ref_index < num_total_refs_; ++ref_index) {
        merge::StackParams stack_params;
        merge::MvStack stack;
        const bool have_stack =
            load_stack(r, c, w4, h4, ref_index, stack_params, stack);

        {
          InterJob job;
          job.x = x0;
          job.y = y0;
          job.w = w;
          job.h = h;
          job.mv_y = 0;
          job.mv_x = 0;
          job.ref_index = ref_index;
          std::vector<std::int64_t> sse;
          if (!backend_.inter_sse_batch({job}, sse, error)) return false;
          InterCandidate cand;
          cand.valid = true;
          cand.mode = syntax::kInterGlobalMv;
          cand.ref_index = ref_index;
          cand.mv = Mv{0, 0};
          cand.sse = sse[0];
          consider(cand);
        }

        if (have_stack && options_.enable_nearmv) {
          const int near_limit =
              (w > 32 && h > 32) ? stack.count : std::min(stack.count, 2);
          std::vector<InterJob> jobs;
          std::vector<InterCandidate> cands;
          for (int i = 0; i < near_limit; ++i) {
            merge::ResolvedMode resolved;
            if (merge::resolve_single_mode(stack, stack_params,
                                           merge::SingleMode::kNearMv, i,
                                           Mv{0, 0}, &resolved) !=
                merge::Status::kOk) {
              continue;
            }
            InterJob job;
            job.x = x0;
            job.y = y0;
            job.w = w;
            job.h = h;
            job.mv_y = resolved.mv[0].y;
            job.mv_x = resolved.mv[0].x;
            job.ref_index = ref_index;
            InterCandidate cand;
            cand.valid = true;
            cand.mode = syntax::kInterNearMv;
            cand.ref_index = ref_index;
            cand.drl_index = i;
            cand.mv = resolved.mv[0];
            jobs.push_back(job);
            cands.push_back(cand);
          }
          std::vector<std::int64_t> sse;
          if (!jobs.empty()) {
            if (!backend_.inter_sse_batch(jobs, sse, error)) return false;
            for (std::size_t i = 0; i < cands.size(); ++i) {
              cands[i].sse = sse[i];
              consider(cands[i]);
            }
          }
        }

        if (have_stack && options_.enable_newmv) {
          MeJob me_job;
          me_job.x = x0;
          me_job.y = y0;
          me_job.w = w;
          me_job.h = h;
          me_job.ref_index = ref_index;
          std::vector<Mv> mvs;
          if (!backend_.me_batch({me_job}, mvs, error)) return false;
          const Mv me_mv = quantize_mv(mvs[0], options_.mv_precision);
          const Mv pred = stack.cand[0].mv[0];
          const Mv delta{me_mv.y - pred.y, me_mv.x - pred.x};
          const Mv mvd = options_.mv_precision == 0
                             ? delta
                             : quantize_mv(delta, options_.mv_precision);
          merge::ResolvedMode resolved;
          const bool resolved_ok =
              options_.mv_precision == 0
                  ? merge::resolve_single_mode(
                        stack, stack_params, merge::SingleMode::kNewMv, 0, mvd,
                        &resolved) == merge::Status::kOk
                  : true;
          if (resolved_ok) {
            const Mv new_mv =
                options_.mv_precision == 0
                    ? resolved.mv[0]
                    : Mv{pred.y + mvd.y, pred.x + mvd.x};
            InterJob job;
            job.x = x0;
            job.y = y0;
            job.w = w;
            job.h = h;
            job.mv_y = new_mv.y;
            job.mv_x = new_mv.x;
            job.ref_index = ref_index;
            std::vector<std::int64_t> sse;
            if (!backend_.inter_sse_batch({job}, sse, error)) return false;
            InterCandidate cand;
            cand.valid = true;
            cand.mode = syntax::kInterNewMv;
            cand.ref_index = ref_index;
            cand.drl_index = 0;
            cand.mv = new_mv;
            cand.mvd = options_.mv_precision == 0 ? resolved.mvd[0] : mvd;
            cand.sse = sse[0];
            consider(cand);
          }
        }
      }
    }

    // Intra fallback.
    bool have_intra = false;
    std::int64_t intra_sse = 0;
    std::array<PlaneTx, 3> intra_planes{};
    if (keyframe_ || options_.enable_intra_fallback || options_.force_intra) {
      if (!backend_.intra_candidate(x0, y0, w, h, intra_planes, error)) {
        return false;
      }
      intra_sse = sse_against_source(0, x0, y0, w, h, intra_planes[0].recon);
      have_intra = true;
    }

    // Inter residual (rate-aware), mirroring the 8-bit core.
    bool inter_residual = false;
    std::array<PlaneTx, 3> res_planes{};
    std::vector<std::uint16_t> inter_y;
    std::vector<std::uint16_t> inter_u;
    std::vector<std::uint16_t> inter_v;
    std::int64_t inter_luma_sse = best.valid ? best.sse : 0;
    if (!keyframe_ && best.valid) {
      InterJob job;
      job.x = x0;
      job.y = y0;
      job.w = w;
      job.h = h;
      job.mv_y = best.mv.y;
      job.mv_x = best.mv.x;
      job.ref_index = best.ref_index;
      std::vector<std::uint16_t> out[3];
      if (!backend_.mc_420(job, out, error)) return false;
      inter_y = std::move(out[0]);
      inter_u = std::move(out[1]);
      inter_v = std::move(out[2]);
      inter_luma_sse = sse_against_source(0, x0, y0, w, h, inter_y);
      if (options_.enable_inter_residual) {
        const int cx = x0 >> fmt::subsampling_x(format_.chroma);
        const int cy = y0 >> fmt::subsampling_y(format_.chroma);
        const std::int64_t pred_total =
            inter_luma_sse + sse_against_source(1, cx, cy, cw, ch, inter_u) +
            sse_against_source(2, cx, cy, cw, ch, inter_v);
        if (!backend_.inter_residual(job, res_planes, error)) return false;
        const std::int64_t res_total =
            sse_against_source(0, x0, y0, w, h, res_planes[0].recon) +
            sse_against_source(1, cx, cy, cw, ch, res_planes[1].recon) +
            sse_against_source(2, cx, cy, cw, ch, res_planes[2].recon);
        const bool all_zero = res_planes[0].all_zero() &&
                              res_planes[1].all_zero() &&
                              res_planes[2].all_zero();
        const double lambda = options_.inter_rd_lambda;
        const double res_bits =
            3.0 + estimate_tx_bits(res_planes[0], true) +
            estimate_tx_bits(res_planes[1], false) +
            estimate_tx_bits(res_planes[2], false);
        const bool want =
            options_.force_inter_residual ||
            (!all_zero && static_cast<double>(res_total) + lambda * res_bits <
                              static_cast<double>(pred_total) + lambda * 2.0);
        if (want) {
          inter_residual = true;
          inter_y = res_planes[0].recon;
          inter_u = res_planes[1].recon;
          inter_v = res_planes[2].recon;
          inter_luma_sse = sse_against_source(0, x0, y0, w, h, inter_y);
        }
      }
    }

    bool use_intra = keyframe_ || options_.force_intra || !best.valid;
    if (!use_intra && have_intra) {
      if (options_.enable_inter_residual) {
        const double lambda = options_.inter_rd_lambda;
        const double intra_bits =
            3.0 + estimate_tx_bits(intra_planes[0], false) +
            estimate_tx_bits(intra_planes[1], false) +
            estimate_tx_bits(intra_planes[2], false);
        double inter_bits = 8.0;
        if (inter_residual) {
          inter_bits += 3.0 + estimate_tx_bits(res_planes[0], true) +
                        estimate_tx_bits(res_planes[1], false) +
                        estimate_tx_bits(res_planes[2], false);
        }
        use_intra = static_cast<double>(intra_sse) + lambda * intra_bits <
                    static_cast<double>(inter_luma_sse) + lambda * inter_bits;
      } else {
        use_intra = intra_sse < inter_luma_sse;
      }
    }

    const bool has_chroma = fmt::has_chroma(format_.chroma);
    const int plane_count = has_chroma ? 3 : 1;
    if (use_intra) {
      if (!have_intra) {
        error = "intra fallback unavailable";
        return false;
      }
      entry.is_inter = 0;
      entry.skip_flag = 0;
      entry.y_mode = 0;
      entry.y_angle_delta = 0;
      entry.uv_mode = 0;
      if (!backend_.write_recon(x0, y0, w, h, intra_planes[0].recon,
                                 intra_planes[1].recon, intra_planes[2].recon,
                                 error)) {
        return false;
      }
      copy_recon_planes(x0, y0, w, h, intra_planes);
      entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
      write_luma_plan(entry, intra_planes[0], levels_);
      for (int plane = 1; plane < plane_count; ++plane) {
        write_chroma_plan(entry, plane, intra_planes[plane], levels_);
      }
      update_mv_grid(r, c, w4, h4, 0, 0, Mv{0, 0});
      ++intra_blocks_;
      if (options_.trace) {
        std::cerr << "hbd intra r=" << r << " c=" << c << " " << w << "x" << h
                  << " sse=" << intra_sse << "\n";
      }
    } else {
      entry.is_inter = 1;
      entry.skip_flag = inter_residual ? 0 : 1;
      entry.inter_mode = static_cast<std::uint8_t>(best.mode);
      entry.drl_index = static_cast<std::uint8_t>(best.drl_index);
      entry.ref_index = static_cast<std::uint8_t>(best.ref_index);
      entry.mv_y = best.mode == syntax::kInterGlobalMv ? 0 : best.mv.y;
      entry.mv_x = best.mode == syntax::kInterGlobalMv ? 0 : best.mv.x;
      entry.mvd_y = best.mvd.y;
      entry.mvd_x = best.mvd.x;
      std::array<PlaneTx, 3> out_planes{};
      if (inter_residual) {
        out_planes = res_planes;
      } else {
        out_planes[0].recon = inter_y;
        out_planes[1].recon = inter_u;
        out_planes[2].recon = inter_v;
      }
      if (!backend_.write_recon(x0, y0, w, h, out_planes[0].recon,
                                 out_planes[1].recon, out_planes[2].recon,
                                 error)) {
        return false;
      }
      copy_recon_planes(x0, y0, w, h, out_planes);
      if (inter_residual) {
        entry.levels_offset = static_cast<std::uint32_t>(levels_.size());
        write_luma_plan(entry, res_planes[0], levels_);
        for (int plane = 1; plane < plane_count; ++plane) {
          write_chroma_plan(entry, plane, res_planes[plane], levels_);
        }
      }
      update_mv_grid(r, c, w4, h4, 1, best.ref_index, best.mv);
      ++inter_blocks_;
      if (options_.trace) {
        std::cerr << "hbd inter r=" << r << " c=" << c << " " << w << "x" << h
                  << " mode=" << static_cast<int>(best.mode)
                  << " drl=" << best.drl_index << " ref=" << best.ref_index
                  << " mv=(" << best.mv.y << "," << best.mv.x
                  << ") sse=" << inter_luma_sse
                  << " skip=" << static_cast<int>(entry.skip_flag) << "\n";
      }
    }
    return true;
  }

  void copy_recon_planes(int x, int y, int w, int h,
                         const std::array<PlaneTx, 3>& planes) {
    const int chroma_w = w >> fmt::subsampling_x(format_.chroma);
    const int chroma_h = h >> fmt::subsampling_y(format_.chroma);
    const int sizes[3][2] = {{w, h}, {chroma_w, chroma_h}, {chroma_w, chroma_h}};
    const int plane_count = fmt::has_chroma(format_.chroma) ? 3 : 1;
    for (int plane = 0; plane < plane_count; ++plane) {
      const PlaneTx& tx = planes[static_cast<std::size_t>(plane)];
      if (tx.recon.empty()) continue;
      const int px = x >> (plane == 0 ? 0 : fmt::subsampling_x(format_.chroma));
      const int py = y >> (plane == 0 ? 0 : fmt::subsampling_y(format_.chroma));
      const int pw = sizes[plane][0];
      const int ph = sizes[plane][1];
      for (int r = 0; r < ph; ++r) {
        const int ry = py + r;
        if (ry >= recon_[plane].height) break;
        for (int c = 0; c < pw; ++c) {
          const int rx = px + c;
          if (rx >= recon_[plane].width) break;
          recon_[plane].at(rx, ry) =
              tx.recon[static_cast<std::size_t>(r) *
                           static_cast<std::size_t>(pw) +
                       static_cast<std::size_t>(c)];
        }
      }
    }
  }

  const VideoInputFrame& source_;
  const std::vector<RefSlot>& refs_;
  std::span<const int> frame_distance_;
  int num_total_refs_ = 1;
  const Options& options_;
  Backend& backend_;
  bool keyframe_ = false;
  fmt::Format format_{};
  int depth_ = 8;
  int width_ = 0;
  int height_ = 0;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  int quant_step_ = txv::kQuantStep;
  FrameGeometry geometry_{};
  Plane recon_[3];
  std::vector<inter::mc::MiInfo> mv_grid_;
  std::vector<syntax::BlockSyntax> blocks_;
  std::vector<std::int32_t> levels_;
  std::uint32_t inter_blocks_ = 0;
  std::uint32_t intra_blocks_ = 0;
};

}  // namespace

int plane_count(const fmt::Format& format) {
  return fmt::num_planes(format.chroma);
}

std::unique_ptr<Backend> make_cpu_backend() {
  return std::unique_ptr<Backend>(new CpuBackend());
}

bool validate_options(const Options& options, std::size_t frame_count,
                      std::string& error) {
  const fmt::Format& format = options.format;
  if (!fmt::is_signalable(format)) {
    error = "format is not signalable in AV2 v1.0.0";
    return false;
  }
  if (options.base_q_idx == 0) {
    error = "lossless base_q_idx = 0 is not implemented";
    return false;
  }
  if (options.base_q_idx > fmt::max_quantizer(format.depth)) {
    error = "base_q_idx exceeds MaxQ for the signalled bit depth";
    return false;
  }
  if (!options.frame_base_q_idx.empty() &&
      options.frame_base_q_idx.size() != frame_count) {
    error = "frame_base_q_idx must be empty or have one entry per frame";
    return false;
  }
  if (options.num_total_refs < 1 || options.num_total_refs > kRefsPerFrame) {
    error = "num_total_refs must be in [1, 7]";
    return false;
  }
  if (options.num_ref_frames != 0 &&
      (options.num_ref_frames < 4 || options.num_ref_frames > kMaxRefSlots)) {
    error = "num_ref_frames must be 0 (auto) or in [4, 8]";
    return false;
  }
  if (options.num_ref_frames != 0 &&
      options.num_total_refs > options.num_ref_frames) {
    error = "num_total_refs exceeds NumRefFrames";
    return false;
  }
  if (options.mv_precision < 0 || options.mv_precision > 2) {
    error = "mv_precision must be 0, 1 or 2";
    return false;
  }
  if (options.min_block_size < 8 ||
      (options.min_block_size & (options.min_block_size - 1)) != 0) {
    error = "min_block_size must be a power of two >= 8";
    return false;
  }
  if (format.chroma == fmt::ChromaFormat::k422 && options.min_block_size < 16) {
    error = "4:2:2 video requires --video-min-block >= 16 (sub-16 leaves have "
            "no legal 4:2:2 chroma residual size)";
    return false;
  }
  if (options.max_partition_depth < 0 || options.max_partition_depth > 8) {
    error = "max_partition_depth must be in [0, 8]";
    return false;
  }
  if (options.loop_filters.pc_wiener && options.loop_filters.wiener) {
    error = "--video-pc-wiener replaces the luma Wiener NS decision; do not "
            "combine the two luma tools";
    return false;
  }
  // Same pinned-decoder constraint as the still writer: at 4:2:2/4:4:4 the
  // chroma lossless-mask read runs past its extent in the stripe at luma row
  // 248, so chroma LR + deblock cannot be decoder-exact.
  if (!fmt::has_chroma(format.chroma) &&
      format.depth == fmt::BitDepth::k8 && options.loop_filters.pc_wiener &&
      frame_count > 1) {
    // The pinned dav2d rejects the second frame of an 8-bit monochrome
    // PC-Wiener sequence (the key frame decodes bit-exactly).  The same plan
    // at 10-bit monochrome and at 8-bit 4:2:2/4:4:4 is gated exact, so this is
    // a depth/plane-count interaction in the decoder's inter-frame LR path.
    // Reproducer: av2enc Cmono.y4m -o x.ivf --video --chroma 400
    //   --video-pc-wiener --frames 2
    error =
        "--video-pc-wiener on 8-bit 4:0:0 does not decode past the key frame "
        "with pinned dav2d; use 10-bit monochrome or a chroma format";
    return false;
  }
  if (options.enable_deblock && options.loop_filters.chroma_lr &&
      fmt::has_chroma(format.chroma) &&
      fmt::subsampling_y(format.chroma) == 0) {
    error =
        "--video-chroma-lr with --video-deblock does not decode exactly at "
        "4:2:2/4:4:4 with pinned dav2d (chroma lossless-mask aliasing); omit "
        "one of the two";
    return false;
  }
  return true;
}

// §5.14 film grain helpers for the format video core.
grain::GrainChromaIdc grain_chroma_idc(fmt::ChromaFormat chroma) {
  switch (chroma) {
    case fmt::ChromaFormat::k400:
      return grain::GrainChromaIdc::k400;
    case fmt::ChromaFormat::k422:
      return grain::GrainChromaIdc::k422;
    case fmt::ChromaFormat::k444:
      return grain::GrainChromaIdc::k444;
    case fmt::ChromaFormat::k420:
    default:
      return grain::GrainChromaIdc::k420;
  }
}

grain::GrainFrameView grain_view_of(const VideoInputFrame& frame) {
  grain::GrainFrameView view;
  // The format core always carries packed uint16_t planes (LSB-aligned), even
  // for its 8-bit formats, so prefer them whenever they are present.
  const bool packed16 = !frame.y16.empty();
  const int sub_x = fmt::subsampling_x(frame.format.chroma);
  const bool has_chroma = fmt::has_chroma(frame.format.chroma);
  if (packed16) {
    view.y16 = frame.y16.data();
    if (has_chroma) {
      view.u16 = frame.u16.data();
      view.v16 = frame.v16.data();
    }
    view.bit_depth = frame.format.depth == fmt::BitDepth::k10 ? 10 : 8;
  } else {
    view.y = frame.y.data();
    if (has_chroma) {
      view.u = frame.u.data();
      view.v = frame.v.data();
    }
    view.bit_depth = 8;
  }
  view.width = static_cast<int>(frame.width);
  view.height = static_cast<int>(frame.height);
  view.y_stride = static_cast<int>(frame.width);
  view.c_stride = static_cast<int>(frame.width >> sub_x);
  return view;
}

bool insert_grain(const grain::GrainFramePlan& plan,
                  std::vector<std::uint8_t>& unit, std::string& error) {
  if (!plan.emit_model) return true;
  std::string grain_error;
  const std::vector<std::uint8_t> obu =
      grain::MakeFilmGrainObu(plan.obu, &grain_error);
  if (obu.empty()) {
    error = "film grain OBU: " + grain_error;
    return false;
  }
  std::vector<std::uint8_t> with_grain;
  if (!grain::InsertFilmGrainObu(unit, obu, &with_grain, &grain_error)) {
    error = "film grain OBU: " + grain_error;
    return false;
  }
  unit.swap(with_grain);
  return true;
}

bool encode_sequence(std::span<const VideoInputFrame> frames,
                     const Options& options, Backend& backend,
                     VideoSequenceResult& result, std::string& error) {
  error.clear();
  result = VideoSequenceResult{};
  if (frames.empty()) {
    error = "no input frames";
    return false;
  }
  if (!validate_options(options, frames.size(), error)) return false;
  if (!options.frame_base_q_idx.empty()) {
    error = "per-frame base_q_idx is not wired for the format video core";
    return false;
  }
  backend.set_options(options);
  const fmt::Format format = options.format;
  const std::uint32_t width = frames[0].width;
  const std::uint32_t height = frames[0].height;
  if (width == 0 || height == 0 || (width & 7U) != 0 || (height & 7U) != 0) {
    error = "the format video core requires non-zero width/height divisible by "
            "8";
    return false;
  }
  if (width > 0xFFFFU || height > 0xFFFFU) {
    error = "dimensions exceed the sequence header field width";
    return false;
  }
  const int mi_cols = static_cast<int>(2U * ((width + 7U) >> 3U));
  const int mi_rows = static_cast<int>(2U * ((height + 7U) >> 3U));
  const bool has_chroma = fmt::has_chroma(format.chroma);
  const int sub_x = fmt::subsampling_x(format.chroma);
  const int sub_y = fmt::subsampling_y(format.chroma);

  for (const VideoInputFrame& f : frames) {
    if (f.width != width || f.height != height) {
      error = "all frames must have the same dimensions";
      return false;
    }
    if (f.y16.size() < static_cast<std::size_t>(width) * height) {
      error = "input luma plane is too small";
      return false;
    }
    if (has_chroma) {
      const std::size_t chroma = static_cast<std::size_t>(width >> sub_x) *
                                 static_cast<std::size_t>(height >> sub_y);
      if (f.u16.size() < chroma || f.v16.size() < chroma) {
        error = "input chroma planes are too small";
        return false;
      }
    }
  }

  const int num_total_refs = options.num_total_refs;
  const int num_ref_frames =
      options.num_ref_frames != 0
          ? options.num_ref_frames
          : (num_total_refs > 1 ? 8 : 2);
  if (num_total_refs > num_ref_frames) {
    error = "num_total_refs exceeds NumRefFrames";
    return false;
  }
  const int order_hint_bits = 4;

  // §5.14 film grain: one planner per CVS, planned in coding order, one model
  // OBU per temporal unit at most.
  grain::GrainPlanOptions grain_options;
  grain_options.chroma_idc = grain_chroma_idc(format.chroma);
  grain::GrainStreamPlanner grain_planner(grain_options);

  Plane frame_recon[3];
  for (int plane = 0; plane < 3; ++plane) {
    if (plane > 0 && !has_chroma) continue;
    frame_recon[plane].resize(
        (mi_cols * 4) >> (plane == 0 ? 0 : sub_x),
        (mi_rows * 4) >> (plane == 0 ? 0 : sub_y));
  }

  VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.chroma = format.chroma;
  config.depth = format.depth;
  config.base_q_idx = options.base_q_idx;
  config.order_hint_bits = order_hint_bits;
  config.num_ref_frames = static_cast<std::uint8_t>(num_ref_frames);
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = true;
  config.enable_cfl_intra = true;
  config.cfl_ds_filter_index = options.loop_filters.cfl_ds_filter_index;
  config.enable_deblock = options.enable_deblock;
  config.film_grain_params_present = options.film_grain;
  apply_video_loop_filter_sequence_config(config, options.loop_filters);
  if (!fmt::has_chroma(format.chroma)) {
    // §5.4.10: monochrome codes no chroma loop-restoration tool list.
    config.lr_uv_present = false;
    config.lr_disable_chroma_wiener_ns = true;
  }
  if (options.loop_filters.pc_wiener) {
    // §5.4.10 tool list for RESTORE_PC_WIENER (the shared helper keeps the
    // historical luma-Wiener default).
    config.enable_restoration = true;
    config.lr_disable_luma_pc_wiener = false;
    // Chroma formats signal the PC-only list.  Monochrome keeps the NS tool
    // advertised as well: the pinned dav2d rejects an inter frame whose luma
    // loop-restoration tool list contains PC-Wiener alone, and an unused tool
    // in the list is harmless (§5.18.7.11 selects per frame).
    config.lr_disable_luma_wiener_ns = fmt::has_chroma(format.chroma);
    if (fmt::has_chroma(format.chroma)) {
      // A PC-only luma list leaves the inferred chroma list empty, and the
      // §5.18.7.11 frame row cannot code a chroma FrameRestorationType from an
      // empty list.  Signal a chroma NS tool that every frame leaves NONE.
      config.lr_uv_present = true;
      config.lr_disable_chroma_wiener_ns = false;
    }
  }

  const auto begin_frame = [&](const VideoInputFrame& source,
                               const std::vector<RefSlot>& refs,
                               std::string& err) {
    BeginFrame begin;
    begin.format = format;
    begin.mi_cols = mi_cols;
    begin.mi_rows = mi_rows;
    begin.width = static_cast<int>(width);
    begin.height = static_cast<int>(height);
    begin.base_q_idx = options.base_q_idx;
    begin.source.plane[0] =
        PlaneView{source.y16.data(), static_cast<int>(width),
                  static_cast<int>(width), static_cast<int>(height)};
    if (has_chroma) {
      begin.source.plane[1] =
          PlaneView{source.u16.data(), static_cast<int>(width >> sub_x),
                    static_cast<int>(width >> sub_x),
                    static_cast<int>(height >> sub_y)};
      begin.source.plane[2] =
          PlaneView{source.v16.data(), static_cast<int>(width >> sub_x),
                    static_cast<int>(width >> sub_x),
                    static_cast<int>(height >> sub_y)};
    }
    for (int plane = 0; plane < 3; ++plane) {
      if (plane > 0 && !has_chroma) continue;
      begin.recon.plane[plane] =
          PlaneView{frame_recon[plane].samples.data(),
                    frame_recon[plane].stride, frame_recon[plane].width,
                    frame_recon[plane].height};
    }
    begin.num_refs = static_cast<int>(refs.size());
    for (int r = 0; r < begin.num_refs && r < kMaxRefSlots; ++r) {
      for (int plane = 0; plane < 3; ++plane) {
        if (plane > 0 && !has_chroma) continue;
        begin.refs[r].plane[plane] =
            PlaneView{refs[static_cast<std::size_t>(r)]
                          .plane[plane]
                          .samples.data(),
                      refs[static_cast<std::size_t>(r)].plane[plane].stride,
                      refs[static_cast<std::size_t>(r)].plane[plane].width,
                      refs[static_cast<std::size_t>(r)].plane[plane].height};
      }
    }
    for (int plane = 0; plane < 3; ++plane) {
      if (plane > 0 && !has_chroma) continue;
      std::fill(frame_recon[plane].samples.begin(),
                frame_recon[plane].samples.end(), 0);
    }
    return backend.begin_frame(begin, err);
  };

  // Keyframe.
  {
    std::vector<RefSlot> no_refs;
    std::array<int, 7> distance{};
    if (!begin_frame(frames[0], no_refs, error)) return false;
    FrameEncoder encoder(frames[0], no_refs, distance, 1, options, backend,
                         /*keyframe=*/true);
    std::vector<std::uint8_t> payload;
    std::vector<syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    post::VideoFilterPlan filter_plan;
    std::uint16_t* recon_ptr[3] = {frame_recon[0].samples.data(),
                                   frame_recon[1].samples.data(),
                                   frame_recon[2].samples.data()};
    const int recon_stride[3] = {frame_recon[0].stride, frame_recon[1].stride,
                                 frame_recon[2].stride};
    if (!encoder.encode(payload, blocks, levels, &filter_plan, recon_ptr,
                        recon_stride, error)) {
      error = "keyframe encode failed: " + error;
      return false;
    }
    if (!backend.end_frame(error)) return false;
    // A PC-Wiener request that won no unit is a no-op for this stream.  The
    // pinned decoder rejects a sequence that advertises PC-Wiener when the
    // frame's FrameRestorationType is NONE, so drop the tool instead of
    // signalling an empty plan.
    if (options.loop_filters.pc_wiener && !filter_plan.lr.enabled) {
      config.enable_restoration = false;
      config.lr_disable_luma_pc_wiener = true;
      config.lr_disable_luma_wiener_ns = true;
      config.lr_uv_present = false;
      config.lr_disable_chroma_wiener_ns = true;
    }
    std::vector<std::uint8_t> unit;
    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);
    grain::GrainFramePlan key_grain;
    if (options.film_grain &&
        !grain_planner.PlanFrame(grain_view_of(frames[0]), 0, 0, 0, &key_grain,
                                 &error)) {
      error = "keyframe film grain: " + error;
      return false;
    }
    VideoFrameHeader header;
    header.keyframe = true;
    header.order_hint = 0;
    header.ref_slot = 0;
    header.refresh_frame_flags = (1U << config.num_ref_frames) - 1U;
    if (options.film_grain) {
      header.apply_grain = key_grain.apply_grain;
      header.fgm_id = static_cast<std::uint8_t>(key_grain.fgm_id);
      header.grain_seed = key_grain.grain_seed;
    }
    apply_video_loop_filter_frame_header(filter_plan, header);
    const std::vector<std::uint8_t> obu = make_frame_obu(config, header, payload);
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), seq.begin(), seq.end());
    unit.insert(unit.end(), obu.begin(), obu.end());
    if (options.film_grain && !insert_grain(key_grain, unit, error)) {
      error = "keyframe " + error;
      return false;
    }
    result.units.push_back(std::move(unit));

    VideoReconFrame recon;
    recon.width = width;
    recon.height = height;
    recon.format = format;
    for (int plane = 0; plane < 3; ++plane) {
      if (plane > 0 && !has_chroma) continue;
      const int pw = (mi_cols * 4) >> (plane == 0 ? 0 : sub_x);
      const int ph = (mi_rows * 4) >> (plane == 0 ? 0 : sub_y);
      std::vector<std::uint16_t> visible(
          static_cast<std::size_t>(width >> (plane == 0 ? 0 : sub_x)) *
          static_cast<std::size_t>(height >> (plane == 0 ? 0 : sub_y)));
      for (std::uint32_t yy = 0; yy < (height >> (plane == 0 ? 0 : sub_y));
           ++yy) {
        std::memcpy(visible.data() +
                        static_cast<std::size_t>(yy) *
                            static_cast<std::size_t>(width >>
                                                     (plane == 0 ? 0 : sub_x)),
                    frame_recon[plane].samples.data() +
                        static_cast<std::size_t>(yy) *
                            static_cast<std::size_t>(frame_recon[plane].stride),
                    static_cast<std::size_t>(width >>
                                             (plane == 0 ? 0 : sub_x)) *
                        sizeof(std::uint16_t));
      }
      (void)pw;
      (void)ph;
      if (plane == 0) recon.y16 = std::move(visible);
      if (plane == 1) recon.u16 = std::move(visible);
      if (plane == 2) recon.v16 = std::move(visible);
    }
    result.recon.push_back(std::move(recon));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(static_cast<std::uint32_t>(payload.size()));
    result.inter_blocks.push_back(0);
    result.intra_blocks.push_back(encoder.intra_blocks());
  }

  // Reference pool.
  std::array<RefSlot, kMaxRefSlots> slots;
  std::vector<int> recent_slots;
  slots[0].valid = true;
  slots[0].order_hint = 0;
  for (int plane = 0; plane < 3; ++plane) {
    if (plane > 0 && !has_chroma) continue;
    slots[0].plane[plane] = frame_recon[plane];
  }
  recent_slots.push_back(0);

  for (std::size_t i = 1; i < frames.size(); ++i) {
    const std::uint32_t order_hint =
        static_cast<std::uint32_t>(i) &
        static_cast<std::uint32_t>((1U << order_hint_bits) - 1U);
    const int ref_count = std::min<int>(
        num_total_refs, static_cast<int>(recent_slots.size()));
    std::vector<RefSlot> references;
    std::array<int, 7> frame_distance{};
    for (int r = 0; r < ref_count; ++r) {
      const int slot = recent_slots[static_cast<std::size_t>(r)];
      references.push_back(slots[static_cast<std::size_t>(slot)]);
      frame_distance[static_cast<std::size_t>(r)] =
          relative_dist(order_hint, slots[static_cast<std::size_t>(slot)]
                                        .order_hint,
                        order_hint_bits);
    }

    if (!begin_frame(frames[i], references, error)) return false;
    FrameEncoder encoder(frames[i], references, frame_distance, ref_count,
                         options, backend, /*keyframe=*/false);
    std::vector<std::uint8_t> payload;
    std::vector<syntax::BlockSyntax> blocks;
    std::vector<std::int32_t> levels;
    post::VideoFilterPlan filter_plan;
    std::uint16_t* recon_ptr[3] = {frame_recon[0].samples.data(),
                                   frame_recon[1].samples.data(),
                                   frame_recon[2].samples.data()};
    const int recon_stride[3] = {frame_recon[0].stride, frame_recon[1].stride,
                                 frame_recon[2].stride};
    if (!encoder.encode(payload, blocks, levels, &filter_plan, recon_ptr,
                        recon_stride, error)) {
      error = "P-frame " + std::to_string(i) + ": " + error;
      return false;
    }
    if (!backend.end_frame(error)) return false;

    const int write_slot = static_cast<int>(i % num_ref_frames);
    VideoFrameHeader header;
    header.keyframe = false;
    header.order_hint = order_hint;
    header.num_total_refs = static_cast<std::uint8_t>(ref_count);
    for (int r = 0; r < ref_count; ++r) {
      header.ref_frame_idx[static_cast<std::size_t>(r)] =
          static_cast<std::uint8_t>(recent_slots[static_cast<std::size_t>(r)]);
    }
    header.ref_slot = header.ref_frame_idx[0];
    header.refresh_frame_flags = 1U << write_slot;
    header.use_qtr_precision_mv = options.mv_precision == 1;
    header.allow_high_precision_mv = options.mv_precision != 2;
    apply_video_loop_filter_frame_header(filter_plan, header);

    grain::GrainFramePlan grain_plan;
    if (options.film_grain &&
        !grain_planner.PlanFrame(grain_view_of(frames[i]),
                                 static_cast<std::uint32_t>(i), 0, 0,
                                 &grain_plan, &error)) {
      error = "P-frame " + std::to_string(i) + " film grain: " + error;
      return false;
    }
    if (options.film_grain) {
      header.apply_grain = grain_plan.apply_grain;
      header.fgm_id = static_cast<std::uint8_t>(grain_plan.fgm_id);
      header.grain_seed = grain_plan.grain_seed;
    }

    const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> obu = make_frame_obu(config, header, payload);
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), obu.begin(), obu.end());
    if (options.film_grain && !insert_grain(grain_plan, unit, error)) {
      error = "P-frame " + std::to_string(i) + " " + error;
      return false;
    }
    result.units.push_back(std::move(unit));

    VideoReconFrame recon;
    recon.width = width;
    recon.height = height;
    recon.format = format;
    for (int plane = 0; plane < 3; ++plane) {
      if (plane > 0 && !has_chroma) continue;
      const int pw = static_cast<int>(width >> (plane == 0 ? 0 : sub_x));
      const int ph = static_cast<int>(height >> (plane == 0 ? 0 : sub_y));
      std::vector<std::uint16_t> visible(
          static_cast<std::size_t>(pw) * static_cast<std::size_t>(ph));
      for (int yy = 0; yy < ph; ++yy) {
        std::memcpy(visible.data() +
                        static_cast<std::size_t>(yy) *
                            static_cast<std::size_t>(pw),
                    frame_recon[plane].samples.data() +
                        static_cast<std::size_t>(yy) *
                            static_cast<std::size_t>(frame_recon[plane].stride),
                    static_cast<std::size_t>(pw) * sizeof(std::uint16_t));
      }
      if (plane == 0) recon.y16 = std::move(visible);
      if (plane == 1) recon.u16 = std::move(visible);
      if (plane == 2) recon.v16 = std::move(visible);
    }
    result.recon.push_back(std::move(recon));
    result.frame_bytes.push_back(
        static_cast<std::uint32_t>(result.units.back().size()));
    result.tile_bytes.push_back(static_cast<std::uint32_t>(payload.size()));
    result.inter_blocks.push_back(encoder.inter_blocks());
    result.intra_blocks.push_back(encoder.intra_blocks());

    slots[static_cast<std::size_t>(write_slot)].valid = true;
    slots[static_cast<std::size_t>(write_slot)].order_hint = order_hint;
    for (int plane = 0; plane < 3; ++plane) {
      if (plane > 0 && !has_chroma) continue;
      slots[static_cast<std::size_t>(write_slot)].plane[plane] =
          frame_recon[plane];
    }
    recent_slots.erase(
        std::remove(recent_slots.begin(), recent_slots.end(), write_slot),
        recent_slots.end());
    recent_slots.insert(recent_slots.begin(), write_slot);
    if (recent_slots.size() > static_cast<std::size_t>(kRefsPerFrame)) {
      recent_slots.resize(static_cast<std::size_t>(kRefsPerFrame));
    }
  }

  result.frame_output_index.resize(frames.size());
  for (std::size_t i = 0; i < frames.size(); ++i) {
    result.frame_output_index[i] = static_cast<std::int32_t>(i);
  }
  return true;
}

}  // namespace av2enc::video::hbd

namespace av2enc::video {

bool encode_video_sequence_format(std::span<const VideoInputFrame> frames,
                                  const VideoOptions& options,
                                  VideoSequenceResult& result,
                                  std::string& error) {
  hbd::Options format_options;
  format_options.format = options.format;
  format_options.base_q_idx = options.base_q_idx;
  format_options.frame_base_q_idx = options.frame_base_q_idx;
  format_options.num_total_refs = options.num_total_refs;
  format_options.num_ref_frames = options.num_ref_frames;
  format_options.mv_precision = options.mv_precision;
  format_options.me_subpel_eighth = options.me_subpel_eighth;
  format_options.min_block_size = options.min_block_size;
  format_options.max_partition_depth = options.max_partition_depth;
  format_options.me_range = options.me_range;
  format_options.force_single_block = options.force_single_block;
  format_options.force_intra = options.force_intra;
  format_options.enable_intra_fallback = options.enable_intra_fallback;
  format_options.enable_newmv = options.enable_newmv;
  format_options.enable_nearmv = options.enable_nearmv;
  format_options.enable_inter_residual = options.enable_inter_residual;
  format_options.force_inter_residual = options.force_inter_residual;
  format_options.inter_rd_lambda = options.inter_rd_lambda;
  format_options.trace = options.trace;
  format_options.enable_deblock = options.enable_deblock;
  format_options.loop_filters = options.loop_filters;
  format_options.film_grain = options.film_grain;

  // Only the flat key + P chain is wired.  Reject every other M2 feature by
  // name instead of silently dropping it.
  if (options.loop_filters.pc_wiener && options.loop_filters.wiener) {
    error = "--video-pc-wiener replaces the luma Wiener NS decision; do not "
            "combine the two luma tools";
    return false;
  }
  if (options.tile_cols > 1 || options.tile_rows > 1) {
    error = "--tiles is 8-bit 4:2:0 only";
    return false;
  }
  if (options.b_frames > 0 || options.temporal_layers > 0 ||
      options.long_term || options.olk || options.leading_frames > 0 ||
      options.sef || options.ra_switch || options.ras) {
    error = "B frames / temporal layers / frame-type wrappers are 8-bit 4:2:0 "
            "only";
    return false;
  }
  if (options.enable_intra_full || options.enable_tx_search ||
      options.enable_gme || options.enable_warp || options.compound ||
      options.drl_reorder || options.avg_cdf || options.adaptive_cdf ||
      options.cdf_continuity || options.mfh || options.video_timing) {
    error = "this M2 tool is not wired for the --bit-depth/--chroma video "
            "formats yet";
    return false;
  }

  auto backend = hbd::make_cpu_backend();
  return hbd::encode_sequence(frames, format_options, *backend, result, error);
}

}  // namespace av2enc::video
