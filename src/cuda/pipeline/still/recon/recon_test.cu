// CPU-vs-GPU parity test for the M1 reconstruction/deblock stage.
//
// The test builds a small CPU "shadow" of the default M1 decision path from
// the public components (partition, intra_modes, tx_variants, CfL, deblock),
// runs it with and without deblock, and checks that its reconstruction equals
// the official `av2enc::still::encode_still_picture` result.  The same per-block
// prediction + inverse-transform residual jobs are then sent to the GPU recon
// stage, which must reproduce the padded planes, the prediction availability
// state and the deblock metadata exactly.
//
// The shadow does not include encoder-private internals, so the test survives
// refactors of `still_encoder.cpp`; if the default decision path changes, the
// shadow-vs-StillResult gate fails loudly instead of silently testing the wrong
// blocks.
//
// Gates:
//   1. shadow CPU recon == `StillResult.recon_*` (visible crop), with and without
//      deblock, 64x64 and 640x360.
//   2. GPU recon planes == shadow CPU planes (full padded planes) ==
//      `StillResult.recon_*` (visible crop), with and without deblock.
//   3. GPU deblocked planes == CPU deblocked planes == `StillResult.recon_*`
//      with deblock enabled (normative recon -> deblock order; deblock is
//      never applied during prediction).
//   4. GPU prediction availability (have_left/above, above-right/below-left
//      counts) == the shadow's count_top_right/count_bottom_left at every leaf
//      and plane.
//   5. GPU decoded flags / IntraJointModes / UV smooth grids == the shadow's.
//   6. GPU `DeblockMi` grids == the shadow's CPU grids (all fields).
//   7. Synthetic padded-edge jobs: clip1(pred+residual) and plane clamping
//      match a scalar reference; batched == serial.
//
// Usage: ./recon_test [repo_root]

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "cuda/pipeline/still/recon/recon.h"
#include "codec/intra/cfl/cfl.h"
#include "codec/filters/deblock/deblock.h"
#include "codec/intra/modes/intra_modes.h"
#include "pipeline/still/still_encoder.h"
#include "codec/partition/tree/partition.h"
#include "codec/transform/variants/tx_variants.h"
#include "core/io/y4m.h"

namespace {

using av2enc::deblock::DeblockMi;
using av2enc::deblock::DeblockOptions;
using av2enc::intra::Mode;
using av2enc::still::StillOptions;
using av2enc::still::StillResult;
using av2enc::partition::BlockSize;
using av2enc::partition::FrameGeometry;
using av2enc::partition::PartitionNode;
using av2enc::cuda::still::ReconAvail;
using av2enc::cuda::still::ReconAvailRequest;
using av2enc::cuda::still::ReconBlockInfo;
using av2enc::cuda::still::ReconFrame;
using av2enc::cuda::still::ReconPredState;
using av2enc::cuda::still::ReconWriteJob;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_failures;                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
    }                                                                        \
  } while (0)

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    const cudaError_t err__ = (expr);                                        \
    ++g_checks;                                                              \
    if (err__ != cudaSuccess) {                                              \
      ++g_failures;                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__,      \
                   #expr, cudaGetErrorString(err__));                        \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// Shadow CPU encoder: the default M1 path built from public components.
// ---------------------------------------------------------------------------

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

struct ShadowPlane {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> samples;

  void resize(int w, int h, std::uint8_t value) {
    width = w;
    height = h;
    samples.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h),
                   value);
  }
  std::uint8_t& at(int x, int y) {
    return samples[static_cast<std::size_t>(y) * width + x];
  }
};

struct DecodedMap {
  int cols = 0;
  int rows = 0;
  std::vector<std::uint8_t> cells;

  void reset(int c, int r) {
    cols = c;
    rows = r;
    cells.assign(static_cast<std::size_t>(c) * static_cast<std::size_t>(r), 0);
  }
  bool at(int x, int y) const {
    if (x < 0 || y < 0 || x >= cols || y >= rows) return false;
    return cells[static_cast<std::size_t>(y) * cols + x] != 0;
  }
  void fill(int x, int y, int w, int h) {
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        if (x + i >= 0 && y + j >= 0 && x + i < cols && y + j < rows) {
          cells[static_cast<std::size_t>(y + j) * cols + x + i] = 1;
        }
      }
    }
  }
};

struct TxPlan {
  int tx_size = 0;
  std::vector<std::uint8_t> pred;
  std::vector<std::uint8_t> recon;
  std::vector<std::int16_t> residual;  // coded-size inverse-transform output
};

struct CapturedPlane {
  int plane = 0;
  int x0 = 0;
  int y0 = 0;
  int w = 0;
  int h = 0;
  std::vector<std::uint8_t> pred;
  std::vector<std::int16_t> residual;
};

struct AvailabilityMismatch {
  bool any = false;
  int row = -1;
  int col = -1;
  int plane = -1;
  int cpu_tr = 0, gpu_tr = 0;
  int cpu_bl = 0, gpu_bl = 0;
};

struct ShadowEncoder {
  ShadowEncoder(const av2enc::still::StillPicture& picture,
                const StillOptions& options)
      : picture_(picture), options_(options) {
    width_ = static_cast<int>(picture.width);
    height_ = static_cast<int>(picture.height);
    chroma_width_ = (width_ + 1) / 2;
    chroma_height_ = (height_ + 1) / 2;
    geometry_.width = width_;
    geometry_.height = height_;
    geometry_.sb_size = 64;
    geometry_.has_chroma = true;
    geometry_.chroma_subsampling_x = true;
    geometry_.chroma_subsampling_y = true;

    mi_cols_ = av2enc::partition::mi_cols(geometry_);
    mi_rows_ = av2enc::partition::mi_rows(geometry_);
    const int coded_luma_w = mi_cols_ * 4;
    const int coded_luma_h = mi_rows_ * 4;
    luma_stride_ = (coded_luma_w + 63) & ~63;
    luma_height_ = (coded_luma_h + 63) & ~63;
    chroma_stride_ = luma_stride_ / 2;
    chroma_padded_height_ = luma_height_ / 2;

    recon_[0].resize(luma_stride_, luma_height_, 0);
    recon_[1].resize(chroma_stride_, chroma_padded_height_, 0);
    recon_[2].resize(chroma_stride_, chroma_padded_height_, 0);

    intra_joint_.assign(static_cast<std::size_t>(mi_rows_) * mi_cols_, 0);
    uv_smooth_.assign(static_cast<std::size_t>(mi_rows_ / 2) * (mi_cols_ / 2),
                      0);
    for (int p = 0; p < 3; ++p) {
      decoded_[p].reset(p == 0 ? mi_cols_ : mi_cols_ / 2,
                        p == 0 ? mi_rows_ : mi_rows_ / 2);
    }
    deblock_cols_ = mi_cols_;
    deblock_rows_ = mi_rows_;
    DeblockMi neutral;
    neutral.qindex = options_.base_q_idx;
    for (int p = 0; p < 3; ++p) {
      deblock_mi_[p].assign(
          static_cast<std::size_t>(mi_cols_) * mi_rows_, neutral);
    }
  }

  std::uint8_t source_sample(int plane, int x, int y) const {
    const std::span<const std::uint8_t> data =
        plane == 0 ? picture_.y : (plane == 1 ? picture_.u : picture_.v);
    const int w = plane == 0 ? width_ : chroma_width_;
    const int h = plane == 0 ? height_ : chroma_height_;
    const int cx = clip3(0, w - 1, x);
    const int cy = clip3(0, h - 1, y);
    const int stride = plane == 0 ? width_ : chroma_width_;
    return data[static_cast<std::size_t>(cy) * stride + cx];
  }

  int count_top_right(int plane, int abs_x4, int abs_y4, int w4) const {
    const int sub = plane == 0 ? 0 : 1;
    const int sb_c0 = sb_mi_col_ >> sub;
    const int sb_r0 = sb_mi_row_ >> sub;
    const int rel_x = abs_x4 - sb_c0;
    const int rel_y = abs_y4 - sb_r0;
    const int sb_w = sb_cols_plane_[plane];
    const int plane_cols = plane == 0 ? mi_cols_ : mi_cols_ / 2;
    const int border_w = std::min(2 * (16 >> sub), plane_cols - sb_c0);
    int count = 0;
    for (int i = 0; i < w4; ++i) {
      const int x = rel_x + w4 + i;
      if (x >= (rel_y == 0 ? border_w : sb_w)) break;
      if (rel_y - 1 >= 0 &&
          !decoded_[plane].at(sb_c0 + x, sb_r0 + rel_y - 1)) {
        break;
      }
      ++count;
    }
    return count;
  }

  int count_bottom_left(int plane, int abs_x4, int abs_y4, int h4) const {
    const int sub = plane == 0 ? 0 : 1;
    const int sb_c0 = sb_mi_col_ >> sub;
    const int sb_r0 = sb_mi_row_ >> sub;
    const int rel_x = abs_x4 - sb_c0;
    const int rel_y = abs_y4 - sb_r0;
    const int sb_h = sb_rows_plane_[plane];
    int count = 0;
    for (int i = 0; i < h4; ++i) {
      const int y = rel_y + h4 + i;
      if (y >= sb_h) break;
      if (rel_x - 1 >= 0 &&
          !decoded_[plane].at(sb_c0 + rel_x - 1, sb_r0 + y)) {
        break;
      }
      ++count;
    }
    return count;
  }

  bool prepare_edges(int plane, int x0, int y0, int w, int h,
                     av2enc::intra::IntraEdges& edges) const {
    av2enc::intra::NeighborRequest request;
    request.recon = av2enc::intra::PlaneView{recon_[plane].samples.data(),
                                             recon_[plane].width};
    request.x = x0;
    request.y = y0;
    request.w = w;
    request.h = h;
    request.max_x = (plane == 0 ? mi_cols_ * 4 : mi_cols_ * 2) - 1;
    request.max_y = (plane == 0 ? mi_rows_ * 4 : mi_rows_ * 2) - 1;
    request.have_left = x0 > 0;
    request.have_above = y0 > 0;
    request.num4_above_right = count_top_right(plane, x0 >> 2, y0 >> 2, w >> 2);
    request.num4_below_left = count_bottom_left(plane, x0 >> 2, y0 >> 2, h >> 2);
    request.color_plane = plane;
    return av2enc::intra::prepare_edges(request, 8, edges) ==
           av2enc::intra::IntraStatus::kOk;
  }

  bool predict_block(int plane, int x0, int y0, int w, int h, Mode mode,
                     int angle_delta, std::vector<std::uint8_t>& out) {
    av2enc::intra::IntraEdges edges;
    if (!prepare_edges(plane, x0, y0, w, h, edges)) return false;
    // Default path: luma is DC and chroma is DC/CfL, so the smooth-neighbour
    // flags are always false.  (The intra_joint_/uv_smooth_ grids are still
    // maintained for state parity.)
    av2enc::intra::PredictParams params;
    params.mode = mode;
    params.angle_delta = angle_delta;
    params.enable_edge_filter =
        options_.enable_intra_edge_filter &&
        std::getenv("AV2ENC_STILL_NO_EDGE") == nullptr;
    params.above_neighbor_is_smooth = false;
    params.left_neighbor_is_smooth = false;
    out.resize(static_cast<std::size_t>(w) * h);
    return av2enc::intra::predict_intra(edges, params, out.data(),
                                        static_cast<std::size_t>(w)) ==
           av2enc::intra::IntraStatus::kOk;
  }

  std::int64_t sse_against_source(int plane, int x0, int y0, int w, int h,
                                  const std::vector<std::uint8_t>& pred) const {
    std::int64_t sse = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int src = source_sample(plane, x0 + x, y0 + y);
        const int p = pred[static_cast<std::size_t>(y) * w + x];
        const int d = src - p;
        sse += static_cast<std::int64_t>(d) * d;
      }
    }
    return sse;
  }

  std::int64_t block_variance_cost_at(int mi_row, int mi_col,
                                      BlockSize size) const {
    const av2enc::partition::BlockRect rect =
        av2enc::partition::block_rect(geometry_, mi_row, mi_col, size);
    const int w = rect.visible_width;
    const int h = rect.visible_height;
    if (w <= 0 || h <= 0) return 0;
    std::int64_t sum = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        sum += source_sample(0, rect.x + x, rect.y + y);
      }
    }
    const double mean = static_cast<double>(sum) / static_cast<double>(w * h);
    double sse = 0.0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const double d =
            static_cast<double>(source_sample(0, rect.x + x, rect.y + y)) -
            mean;
        sse += d * d;
      }
    }
    return static_cast<std::int64_t>(sse);
  }

  bool compute_tx(int plane, int x0, int y0, int cw, int ch,
                  const std::vector<std::uint8_t>& pred, TxPlan& plan) {
    const auto params_opt =
        av2enc::txv::make_tx_params(cw, ch, av2enc::txv::TxType::kDctDct);
    if (!params_opt.has_value()) return false;
    const av2enc::txv::TxParams& params = *params_opt;
    plan.tx_size = params.tx_size;
    plan.pred = pred;
    std::vector<std::int16_t> residual_coded(
        static_cast<std::size_t>(cw) * ch);
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const int src = source_sample(plane, x0 + x, y0 + y);
        const int p = pred[static_cast<std::size_t>(y) * cw + x];
        residual_coded[static_cast<std::size_t>(y) * cw + x] =
            static_cast<std::int16_t>(src - p);
      }
    }
    std::vector<std::int16_t> residual_adj(static_cast<std::size_t>(params.w) *
                                           params.h);
    av2enc::txv::downsample_residual(residual_coded.data(), params,
                                     residual_adj.data());
    std::vector<double> coeff(static_cast<std::size_t>(params.w) * params.h);
    std::vector<std::int32_t> levels(coeff.size(), 0);
    av2enc::txv::forward_transform(residual_adj.data(), params, coeff.data(),
                                   levels.data());
    if (plane == 0 && std::getenv("AV2ENC_STILL_ZERO_DC") != nullptr) {
      levels[0] = 0;
    }
    std::vector<std::int32_t> dequant(coeff.size());
    plan.residual.resize(static_cast<std::size_t>(cw) * ch);
    av2enc::txv::inverse_transform(levels.data(), params, dequant.data(),
                                   plan.residual.data());
    plan.recon.resize(static_cast<std::size_t>(cw) * ch);
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const int p = plan.pred[static_cast<std::size_t>(y) * cw + x];
        const int r = plan.residual[static_cast<std::size_t>(y) * cw + x];
        plan.recon[static_cast<std::size_t>(y) * cw + x] =
            static_cast<std::uint8_t>(clip3(0, 255, p + r));
      }
    }
    return true;
  }

  void write_recon(int plane, int x0, int y0, int cw, int ch,
                   const TxPlan& plan) {
    const int max_x = recon_[plane].width;
    const int max_y = recon_[plane].height;
    for (int y = 0; y < ch; ++y) {
      const int py = y0 + y;
      if (py < 0 || py >= max_y) continue;
      for (int x = 0; x < cw; ++x) {
        const int px = x0 + x;
        if (px < 0 || px >= max_x) continue;
        recon_[plane].at(px, py) =
            plan.recon[static_cast<std::size_t>(y) * cw + x];
      }
    }
  }

  void make_cfl_block(int plane, int cx0, int cy0, int cw, int ch,
                      av2enc::cfl::Block& block) {
    block.luma = recon_[0].samples.data();
    block.luma_stride = recon_[0].width;
    block.luma_max_x = mi_cols_ * 4 - 1;
    block.luma_max_y = mi_rows_ * 4 - 1;
    block.sb_top_y = (cy0 * 2 / 64) * 64;
    block.chroma = recon_[plane].samples.data();
    block.chroma_stride = recon_[plane].width;
    block.x = cx0;
    block.y = cy0;
    block.w = cw;
    block.h = ch;
    block.coded_width = mi_cols_ * 2;
    block.coded_height = mi_rows_ * 2;
    block.have_above = av2enc::cfl::cfl_avail_above(cy0, 1) ? 1 : 0;
    block.have_left = av2enc::cfl::cfl_avail_left(cx0, 1) ? 1 : 0;
    block.bit_depth = 8;
    block.ds_filter = 0;
    block.sub_x = 1;
    block.sub_y = 1;
  }

  // Default chroma decision: DC (CfL candidate scored with the U+V SSE).
  bool choose_uv_mode(int cx0, int cy0, int cw, int ch, bool& is_cfl) {
    is_cfl = false;
    if (std::getenv("AV2ENC_STILL_UV_DC") != nullptr || options_.force_dc ||
        !options_.enable_cfl) {
      return true;
    }
    std::vector<std::uint8_t> pred;
    if (!predict_block(1, cx0, cy0, cw, ch, Mode::kDc, 0, pred)) return false;
    std::int64_t best = sse_against_source(1, cx0, cy0, cw, ch, pred);
    std::vector<std::uint8_t> u_dc;
    std::vector<std::uint8_t> v_dc;
    if (!predict_block(1, cx0, cy0, cw, ch, Mode::kDc, 0, u_dc) ||
        !predict_block(2, cx0, cy0, cw, ch, Mode::kDc, 0, v_dc)) {
      return false;
    }
    for (int yy = 0; yy < ch; ++yy) {
      for (int xx = 0; xx < cw; ++xx) {
        recon_[1].at(cx0 + xx, cy0 + yy) =
            u_dc[static_cast<std::size_t>(yy) * cw + xx];
        recon_[2].at(cx0 + xx, cy0 + yy) =
            v_dc[static_cast<std::size_t>(yy) * cw + xx];
      }
    }
    av2enc::cfl::Scratch scratch;
    av2enc::cfl::Block ub;
    av2enc::cfl::Block vb;
    make_cfl_block(1, cx0, cy0, cw, ch, ub);
    make_cfl_block(2, cx0, cy0, cw, ch, vb);
    av2enc::cfl::compute_luma(ub, scratch);
    const int alpha_u = av2enc::cfl::derive_implicit_alpha(ub, scratch);
    av2enc::cfl::compute_luma(vb, scratch);
    const int alpha_v = av2enc::cfl::derive_implicit_alpha(vb, scratch);
    av2enc::cfl::apply_prediction(ub, alpha_u, scratch);
    av2enc::cfl::apply_prediction(vb, alpha_v, scratch);
    std::int64_t sse = 0;
    for (int yy = 0; yy < ch; ++yy) {
      for (int xx = 0; xx < cw; ++xx) {
        const int su = source_sample(1, cx0 + xx, cy0 + yy);
        const int pu = recon_[1].at(cx0 + xx, cy0 + yy);
        const int du = su - pu;
        sse += static_cast<std::int64_t>(du) * du;
        const int sv = source_sample(2, cx0 + xx, cy0 + yy);
        const int pv = recon_[2].at(cx0 + xx, cy0 + yy);
        const int dv = sv - pv;
        sse += static_cast<std::int64_t>(dv) * dv;
      }
    }
    if (sse < best) is_cfl = true;
    return true;
  }

  void fill_deblock_mi(const PartitionNode& node, int luma_tx_size,
                       int chroma_tx_size) {
    const int w4 = av2enc::partition::block_width(node.size) / 4;
    const int h4 = av2enc::partition::block_height(node.size) / 4;
    const av2enc::deblock::TxSize luma =
        static_cast<av2enc::deblock::TxSize>(luma_tx_size);
    const av2enc::deblock::TxSize chroma =
        static_cast<av2enc::deblock::TxSize>(chroma_tx_size);
    for (int yy = 0; yy < h4; ++yy) {
      const int rr = node.mi_row + yy;
      if (rr >= mi_rows_) break;
      for (int xx = 0; xx < w4; ++xx) {
        const int cc = node.mi_col + xx;
        if (cc >= mi_cols_) break;
        const int idx = rr * deblock_cols_ + cc;
        const int clamped_w4 = std::min(w4, mi_cols_ - node.mi_col);
        const int clamped_h4 = std::min(h4, mi_rows_ - node.mi_row);
        DeblockMi mi;
        mi.block_col_base = static_cast<std::uint16_t>(node.mi_col);
        mi.block_row_base = static_cast<std::uint16_t>(node.mi_row);
        mi.block_w4 = static_cast<std::uint8_t>(clamped_w4);
        mi.block_h4 = static_cast<std::uint8_t>(clamped_h4);
        mi.sub_pu_size = luma;
        mi.tx_size = luma;
        mi.tx_col_base = static_cast<std::uint16_t>(node.mi_col);
        mi.tx_row_base = static_cast<std::uint16_t>(node.mi_row);
        mi.sub_pu_col_base = static_cast<std::uint16_t>(node.mi_col);
        mi.sub_pu_row_base = static_cast<std::uint16_t>(node.mi_row);
        mi.skip = 0;
        mi.region_intra = 1;
        mi.lossless = 0;
        mi.qindex = options_.base_q_idx;
        deblock_mi_[0][idx] = mi;
        deblock_mi_[1][idx] = mi;
        deblock_mi_[2][idx] = mi;
        deblock_mi_[1][idx].tx_size = chroma;
        deblock_mi_[2][idx].tx_size = chroma;
      }
    }
  }

  // Members
  const av2enc::still::StillPicture& picture_;
  const StillOptions& options_;
  FrameGeometry geometry_;
  int width_ = 0;
  int height_ = 0;
  int chroma_width_ = 0;
  int chroma_height_ = 0;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  int luma_stride_ = 0;
  int luma_height_ = 0;
  int chroma_stride_ = 0;
  int chroma_padded_height_ = 0;
  ShadowPlane recon_[3];
  std::vector<int> intra_joint_;
  std::vector<std::uint8_t> uv_smooth_;
  DecodedMap decoded_[3];
  int sb_mi_row_ = 0;
  int sb_mi_col_ = 0;
  int sb_cols_plane_[3] = {0, 0, 0};
  int sb_rows_plane_[3] = {0, 0, 0};
  std::vector<DeblockMi> deblock_mi_[3];
  int deblock_cols_ = 0;
  int deblock_rows_ = 0;
};

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

ReconFrame g_frame{};
std::vector<DeblockMi> g_mi_expected[3];

struct DriverOutput {
  std::vector<std::uint8_t> recon[3];
  int stride[3] = {0, 0, 0};
  int height[3] = {0, 0, 0};
  std::vector<CapturedPlane> planes;
  std::vector<ReconBlockInfo> infos;
  std::vector<int> intra_joint;
  std::vector<std::uint8_t> uv_smooth;
  std::vector<std::uint8_t> decoded[3];
};

struct CudaPlanes {
  std::uint8_t* data[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};
  int height[3] = {0, 0, 0};

  ~CudaPlanes() { release(); }
  void release() {
    for (auto*& p : data) {
      if (p != nullptr) cudaFree(p);
      p = nullptr;
    }
  }
  bool alloc(int s0, int h0, int s1, int h1, int s2, int h2,
             std::string& error) {
    const int s[3] = {s0, s1, s2};
    const int h[3] = {h0, h1, h2};
    for (int p = 0; p < 3; ++p) {
      stride[p] = s[p];
      height[p] = h[p];
      const std::size_t bytes =
          static_cast<std::size_t>(s[p]) * static_cast<std::size_t>(h[p]);
      const cudaError_t err = cudaMalloc(&data[p], bytes);
      if (err != cudaSuccess) {
        error = cudaGetErrorString(err);
        return false;
      }
    }
    return true;
  }
};

struct CudaMi {
  DeblockMi* grid[3] = {nullptr, nullptr, nullptr};
  ~CudaMi() { release(); }
  void release() {
    for (auto*& p : grid) {
      if (p != nullptr) cudaFree(p);
      p = nullptr;
    }
  }
  bool alloc(int count, std::string& error) {
    for (int p = 0; p < 3; ++p) {
      const cudaError_t err = cudaMalloc(
          &grid[p], sizeof(DeblockMi) * static_cast<std::size_t>(count));
      if (err != cudaSuccess) {
        error = cudaGetErrorString(err);
        return false;
      }
    }
    return true;
  }
};

struct CudaState {
  ReconPredState state{};
  int* joint = nullptr;
  std::uint8_t* uv_smooth = nullptr;
  std::uint8_t* decoded[3] = {nullptr, nullptr, nullptr};

  ~CudaState() { release(); }
  void release() {
    if (joint) cudaFree(joint);
    if (uv_smooth) cudaFree(uv_smooth);
    for (auto*& p : decoded) {
      if (p) cudaFree(p);
      p = nullptr;
    }
    joint = nullptr;
    uv_smooth = nullptr;
  }
  bool alloc(int mi_cols, int mi_rows, std::string& error) {
    state = ReconPredState{};
    state.mi_cols = mi_cols;
    state.mi_rows = mi_rows;
    state.decoded_cols[0] = mi_cols;
    state.decoded_rows[0] = mi_rows;
    state.decoded_cols[1] = mi_cols / 2;
    state.decoded_rows[1] = mi_rows / 2;
    state.decoded_cols[2] = mi_cols / 2;
    state.decoded_rows[2] = mi_rows / 2;
    auto malloci = [&](auto** ptr, std::size_t n, const char* what) {
      const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(ptr), n);
      if (err != cudaSuccess) {
        error = std::string(what) + ": " + cudaGetErrorString(err);
        return false;
      }
      return true;
    };
    if (!malloci(&joint,
                 sizeof(int) * static_cast<std::size_t>(mi_cols) * mi_rows,
                 "intra_joint")) {
      return false;
    }
    const std::size_t uv_n = static_cast<std::size_t>(mi_cols / 2) *
                             static_cast<std::size_t>(mi_rows / 2);
    if (!malloci(&uv_smooth, uv_n, "uv_smooth")) return false;
    for (int p = 0; p < 3; ++p) {
      const std::size_t n = static_cast<std::size_t>(state.decoded_cols[p]) *
                            state.decoded_rows[p];
      if (!malloci(&decoded[p], n, "decoded")) return false;
    }
    state.intra_joint = joint;
    state.uv_smooth = uv_smooth;
    for (int p = 0; p < 3; ++p) state.decoded[p] = decoded[p];
    return true;
  }
};

// ---------------------------------------------------------------------------
// Drive one leaf: prediction, transform, capture, state updates, GPU commit.
// ---------------------------------------------------------------------------

bool drive_block(ShadowEncoder& enc, const PartitionNode& node,
                 const StillOptions& options, CudaState& gpu,
                 ReconAvailRequest* d_req, ReconAvail* d_avail,
                 DriverOutput& out, AvailabilityMismatch& mismatch,
                 std::string& error) {
  const int r = node.mi_row;
  const int c = node.mi_col;
  const int w = av2enc::partition::block_width(node.size);
  const int h = av2enc::partition::block_height(node.size);
  const int w4 = w >> 2;
  const int h4 = h >> 2;
  const int x0 = c * 4;
  const int y0 = r * 4;

  const BlockSize chroma_size =
      av2enc::partition::chroma_plane_residual_size(enc.geometry_, node.size);
  if (!av2enc::partition::is_valid_block_size(chroma_size)) {
    error = "invalid chroma residual size";
    return false;
  }
  const int cw = av2enc::partition::block_width(chroma_size);
  const int ch = av2enc::partition::block_height(chroma_size);
  const int cx0 = x0 / 2;
  const int cy0 = y0 / 2;

  // --- availability parity, before this block commits ---------------------
  {
    ReconAvailRequest req[3];
    req[0] = ReconAvailRequest{0, x0 >> 2, y0 >> 2, w >> 2, h >> 2};
    req[1] = ReconAvailRequest{1, cx0 >> 2, cy0 >> 2, cw >> 2, ch >> 2};
    req[2] = ReconAvailRequest{2, cx0 >> 2, cy0 >> 2, cw >> 2, ch >> 2};
    const int x0s[3] = {x0, cx0, cx0};
    const int y0s[3] = {y0, cy0, cy0};
    const int ws[3] = {w, cw, cw};
    const int hs[3] = {h, ch, ch};
    ReconAvail avail[3];
    CUDA_CHECK(cudaMemcpyAsync(d_req, req, sizeof(req),
                               cudaMemcpyHostToDevice, nullptr));
    CUDA_CHECK(av2enc::cuda::still::recon_block_availability(gpu.state, d_req,
                                                          d_avail, 3,
                                                          nullptr));
    CUDA_CHECK(cudaMemcpyAsync(avail, d_avail, sizeof(avail),
                               cudaMemcpyDeviceToHost, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int p = 0; p < 3; ++p) {
      const int cpu_tr =
          enc.count_top_right(p, x0s[p] >> 2, y0s[p] >> 2, ws[p] >> 2);
      const int cpu_bl =
          enc.count_bottom_left(p, x0s[p] >> 2, y0s[p] >> 2, hs[p] >> 2);
      const int cpu_left = x0s[p] > 0 ? 1 : 0;
      const int cpu_above = y0s[p] > 0 ? 1 : 0;
      if (cpu_tr != avail[p].num4_above_right ||
          cpu_bl != avail[p].num4_below_left ||
          cpu_left != avail[p].have_left ||
          cpu_above != avail[p].have_above) {
        if (!mismatch.any) {
          mismatch.any = true;
          mismatch.row = r;
          mismatch.col = c;
          mismatch.plane = p;
          mismatch.cpu_tr = cpu_tr;
          mismatch.gpu_tr = avail[p].num4_above_right;
          mismatch.cpu_bl = cpu_bl;
          mismatch.gpu_bl = avail[p].num4_below_left;
        }
      }
    }
  }

  // --- luma (default path is DC-only) ------------------------------------
  std::vector<std::uint8_t> luma_pred;
  if (!enc.predict_block(0, x0, y0, w, h, Mode::kDc, 0, luma_pred)) {
    error = "luma prediction failed";
    return false;
  }
  TxPlan luma_plan;
  if (!enc.compute_tx(0, x0, y0, w, h, luma_pred, luma_plan)) {
    error = "luma transform failed";
    return false;
  }
  enc.write_recon(0, x0, y0, w, h, luma_plan);

  // --- chroma (DC or CfL) -------------------------------------------------
  bool cfl = false;
  if (!enc.choose_uv_mode(cx0, cy0, cw, ch, cfl)) {
    error = "chroma mode selection failed";
    return false;
  }
  std::vector<std::uint8_t> u_pred;
  std::vector<std::uint8_t> v_pred;
  if (cfl) {
    // choose_uv_mode left the CfL prediction in the recon planes.
    u_pred.resize(static_cast<std::size_t>(cw) * ch);
    v_pred.resize(static_cast<std::size_t>(cw) * ch);
    for (int yy = 0; yy < ch; ++yy) {
      for (int xx = 0; xx < cw; ++xx) {
        u_pred[static_cast<std::size_t>(yy) * cw + xx] =
            enc.recon_[1].at(cx0 + xx, cy0 + yy);
        v_pred[static_cast<std::size_t>(yy) * cw + xx] =
            enc.recon_[2].at(cx0 + xx, cy0 + yy);
      }
    }
  } else {
    if (!enc.predict_block(1, cx0, cy0, cw, ch, Mode::kDc, 0, u_pred) ||
        !enc.predict_block(2, cx0, cy0, cw, ch, Mode::kDc, 0, v_pred)) {
      error = "chroma prediction failed";
      return false;
    }
  }
  TxPlan u_plan;
  TxPlan v_plan;
  if (!enc.compute_tx(1, cx0, cy0, cw, ch, u_pred, u_plan) ||
      !enc.compute_tx(2, cx0, cy0, cw, ch, v_pred, v_plan)) {
    error = "chroma transform failed";
    return false;
  }
  enc.write_recon(1, cx0, cy0, cw, ch, u_plan);
  enc.write_recon(2, cx0, cy0, cw, ch, v_plan);

  // --- capture the three GPU write jobs -----------------------------------
  CapturedPlane planes[3];
  planes[0] = CapturedPlane{0, x0, y0, w, h, luma_pred, luma_plan.residual};
  planes[1] = CapturedPlane{1, cx0, cy0, cw, ch, u_pred, u_plan.residual};
  planes[2] = CapturedPlane{2, cx0, cy0, cw, ch, v_pred, v_plan.residual};
  for (int p = 0; p < 3; ++p) {
    out.planes.push_back(std::move(planes[p]));
  }

  // --- state updates (default path: joint 0, no smooth chroma) ------------
  for (int yy = 0; yy < h4; ++yy) {
    const int rr = r + yy;
    if (rr >= enc.mi_rows_) break;
    for (int xx = 0; xx < w4; ++xx) {
      const int cc = c + xx;
      if (cc >= enc.mi_cols_) break;
      enc.intra_joint_[static_cast<std::size_t>(rr) * enc.mi_cols_ + cc] = 0;
    }
  }
  const int cr = r / 2;
  const int cc0 = c / 2;
  const int cw4 = cw / 4;
  const int ch4 = ch / 4;
  const int chroma_mi_rows = enc.mi_rows_ / 2;
  const int chroma_mi_cols = enc.mi_cols_ / 2;
  for (int yy = 0; yy < ch4; ++yy) {
    const int rr = cr + yy;
    if (rr >= chroma_mi_rows) break;
    for (int xx = 0; xx < cw4; ++xx) {
      const int cc = cc0 + xx;
      if (cc >= chroma_mi_cols) break;
      enc.uv_smooth_[static_cast<std::size_t>(rr) * chroma_mi_cols + cc] = 0;
    }
  }
  for (int p = 0; p < 3; ++p) {
    enc.decoded_[p].fill(p == 0 ? c : c / 2, p == 0 ? r : r / 2,
                         p == 0 ? w4 : cw4, p == 0 ? h4 : ch4);
  }
  enc.fill_deblock_mi(node, luma_plan.tx_size, u_plan.tx_size);

  // --- GPU commit ---------------------------------------------------------
  ReconBlockInfo info;
  info.mi_row = r;
  info.mi_col = c;
  info.w4 = w4;
  info.h4 = h4;
  info.chroma_w4 = cw4;
  info.chroma_h4 = ch4;
  info.joint = 0;
  info.uv_smooth = 0;
  info.qindex = options.base_q_idx;
  info.tx_size_luma = static_cast<std::uint8_t>(luma_plan.tx_size);
  info.tx_size_chroma = static_cast<std::uint8_t>(u_plan.tx_size);
  out.infos.push_back(info);
  CUDA_CHECK(
      av2enc::cuda::still::recon_commit_block(info, gpu.state, g_frame, nullptr));
  return true;
}

bool drive_frame(const av2enc::still::StillPicture& picture,
                 const StillOptions& options, CudaState& gpu, DriverOutput& out,
                 AvailabilityMismatch& mismatch, std::string& error) {
  if (!options.luma_dc_only) {
    error = "shadow supports the default DC-only luma path";
    return false;
  }
  ShadowEncoder enc(picture, options);

  av2enc::partition::DecisionOptions poptions;
  poptions.min_block_size_px = options.min_block_size;
  poptions.max_depth = options.max_partition_depth;
  poptions.allow_rect = true;
  poptions.allow_split = false;
  const double lambda =
      0.85 * std::pow(2.0, (options.base_q_idx - 12) / 6.0);
  const double signal_cost = lambda * 2.0;
  const auto cost_fn = [&](const av2enc::partition::PartitionNodeInfo& info) {
    av2enc::partition::NodeCosts costs;
    costs.leaf = static_cast<double>(
        enc.block_variance_cost_at(info.mi_row, info.mi_col, info.size));
    costs.none_signal = 0.0;
    costs.split_signal = signal_cost;
    costs.horz_signal = signal_cost;
    costs.vert_signal = signal_cost;
    return costs;
  };
  const av2enc::partition::FrameDecision frame =
      av2enc::partition::decide_frame(enc.geometry_, cost_fn, poptions);
  if (frame.status != av2enc::partition::DecisionStatus::kOk) {
    error = "partition decision failed";
    return false;
  }
  if (!av2enc::partition::validate_frame(enc.geometry_, frame, error)) {
    return false;
  }

  out.stride[0] = enc.recon_[0].width;
  out.stride[1] = enc.recon_[1].width;
  out.stride[2] = enc.recon_[2].width;
  out.height[0] = enc.recon_[0].height;
  out.height[1] = enc.recon_[1].height;
  out.height[2] = enc.recon_[2].height;

  ReconAvailRequest* d_req = nullptr;
  ReconAvail* d_avail = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_req),
                        sizeof(ReconAvailRequest) * 3));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_avail), sizeof(ReconAvail) * 3));

  for (const PartitionNode& sb : frame.superblocks) {
    for (int p = 0; p < 3; ++p) {
      enc.decoded_[p].reset(p == 0 ? enc.mi_cols_ : enc.mi_cols_ / 2,
                            p == 0 ? enc.mi_rows_ : enc.mi_rows_ / 2);
    }
    enc.sb_mi_row_ = sb.mi_row;
    enc.sb_mi_col_ = sb.mi_col;
    enc.sb_cols_plane_[0] = std::min(16, enc.mi_cols_ - sb.mi_col);
    enc.sb_rows_plane_[0] = std::min(16, enc.mi_rows_ - sb.mi_row);
    enc.sb_cols_plane_[1] = std::min(8, enc.mi_cols_ / 2 - sb.mi_col / 2);
    enc.sb_rows_plane_[1] = std::min(8, enc.mi_rows_ / 2 - sb.mi_row / 2);
    enc.sb_cols_plane_[2] = enc.sb_cols_plane_[1];
    enc.sb_rows_plane_[2] = enc.sb_rows_plane_[1];
    CUDA_CHECK(av2enc::cuda::still::recon_begin_sb(gpu.state, sb.mi_row,
                                                sb.mi_col));

    std::vector<av2enc::partition::LeafBlock> leaves;
    av2enc::partition::collect_leaves(sb, leaves);
    for (const auto& leaf : leaves) {
      PartitionNode node;
      node.mi_row = leaf.mi_row;
      node.mi_col = leaf.mi_col;
      node.size = leaf.size;
      node.depth = leaf.depth;
      node.has_chroma = leaf.has_chroma;
      node.chroma_offset = leaf.chroma_offset;
      node.partition = av2enc::partition::PartitionType::kNone;
      if (!drive_block(enc, node, options, gpu, d_req, d_avail, out, mismatch,
                       error)) {
        cudaFree(d_req);
        cudaFree(d_avail);
        return false;
      }
    }
  }
  cudaFree(d_req);
  cudaFree(d_avail);

  for (int p = 0; p < 3; ++p) {
    out.recon[p] = enc.recon_[p].samples;
    g_mi_expected[p] = enc.deblock_mi_[p];
  }
  out.intra_joint = enc.intra_joint_;
  out.uv_smooth = enc.uv_smooth_;
  for (int p = 0; p < 3; ++p) out.decoded[p] = enc.decoded_[p].cells;
  return true;
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

bool compare_u8(const std::uint8_t* got, const std::uint8_t* want,
                std::size_t n, const char* label) {
  for (std::size_t i = 0; i < n; ++i) {
    if (got[i] != want[i]) {
      std::fprintf(stderr, "  %s mismatch at %zu: got %u want %u\n", label, i,
                   static_cast<unsigned>(got[i]),
                   static_cast<unsigned>(want[i]));
      return false;
    }
  }
  return true;
}

bool compare_int(const int* got, const int* want, std::size_t n,
                 const char* label) {
  for (std::size_t i = 0; i < n; ++i) {
    if (got[i] != want[i]) {
      std::fprintf(stderr, "  %s mismatch at %zu: got %d want %d\n", label, i,
                   got[i], want[i]);
      return false;
    }
  }
  return true;
}

bool compare_visible(const std::vector<std::uint8_t>& padded, int stride,
                     const std::vector<std::uint8_t>& expected, int w, int h,
                     const char* label) {
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int got = padded[static_cast<std::size_t>(y) * stride + x];
      const int want = expected[static_cast<std::size_t>(y) * w + x];
      if (got != want) {
        std::fprintf(stderr, "  %s mismatch at (%d,%d): got %d want %d\n",
                     label, x, y, got, want);
        return false;
      }
    }
  }
  return true;
}

bool load_y4m(const std::filesystem::path& path, av2enc::Y4mFrame& frame,
              av2enc::still::StillPicture& picture, std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  if (reader.read_frame(frame, error) != av2enc::Y4mReadResult::kFrame) {
    if (error.empty()) error = "no frame in " + path.string();
    return false;
  }
  picture.width = frame.width();
  picture.height = frame.height();
  picture.y = frame.y();
  picture.u = frame.u();
  picture.v = frame.v();
  return true;
}

// ---------------------------------------------------------------------------
// Frame case
// ---------------------------------------------------------------------------

struct CaseResult {
  int leaves = 0;
  double write_ms = 0.0;
  double deblock_ms = 0.0;
};

CaseResult run_case(const std::filesystem::path& path, bool enable_deblock,
                    bool time_it) {
  CaseResult result;
  std::string error;
  av2enc::Y4mFrame y4m;
  av2enc::still::StillPicture picture;
  if (!load_y4m(path, y4m, picture, error)) {
    std::fprintf(stderr, "FAIL load %s: %s\n", path.c_str(), error.c_str());
    ++g_failures;
    ++g_checks;
    return result;
  }
  const int w = static_cast<int>(picture.width);
  const int h = static_cast<int>(picture.height);
  FrameGeometry geometry;
  geometry.width = w;
  geometry.height = h;
  const int mi_cols = av2enc::partition::mi_cols(geometry);
  const int mi_rows = av2enc::partition::mi_rows(geometry);

  StillOptions options;
  options.base_q_idx = 90;
  options.enable_deblock = enable_deblock;
  options.luma_dc_only = true;

  // --- official CPU reference --------------------------------------------
  StillResult ref_db;
  StillResult ref_plain;
  StillOptions plain_options = options;
  plain_options.enable_deblock = false;
  if (!av2enc::still::encode_still_picture(picture, options, ref_db, error) ||
      !av2enc::still::encode_still_picture(picture, plain_options, ref_plain,
                                           error)) {
    std::fprintf(stderr, "FAIL reference encode: %s\n", error.c_str());
    ++g_failures;
    ++g_checks;
    return result;
  }
  result.leaves = ref_db.leaves;

  // --- device frame -------------------------------------------------------
  const int coded_w = mi_cols * 4;
  const int coded_h = mi_rows * 4;
  const int luma_stride = (coded_w + 63) & ~63;
  const int luma_height = (coded_h + 63) & ~63;
  CudaPlanes planes;
  if (!planes.alloc(luma_stride, luma_height, luma_stride / 2,
                    luma_height / 2, luma_stride / 2, luma_height / 2, error)) {
    std::fprintf(stderr, "FAIL cudaMalloc planes: %s\n", error.c_str());
    ++g_failures;
    ++g_checks;
    return result;
  }
  CudaMi mi;
  if (!mi.alloc(mi_cols * mi_rows, error)) {
    std::fprintf(stderr, "FAIL cudaMalloc mi: %s\n", error.c_str());
    ++g_failures;
    ++g_checks;
    return result;
  }
  CudaState gpu;
  if (!gpu.alloc(mi_cols, mi_rows, error)) {
    std::fprintf(stderr, "FAIL cudaMalloc state: %s\n", error.c_str());
    ++g_failures;
    ++g_checks;
    return result;
  }
  g_frame = ReconFrame{};
  for (int p = 0; p < 3; ++p) {
    g_frame.plane[p].data = planes.data[p];
    g_frame.plane[p].stride = planes.stride[p];
    g_frame.plane[p].height = planes.height[p];
    g_frame.deblock_mi[p] = mi.grid[p];
  }
  g_frame.mi_cols = mi_cols;
  g_frame.mi_rows = mi_rows;
  g_frame.base_q_idx = options.base_q_idx;
  CUDA_CHECK(av2enc::cuda::still::recon_init_frame(g_frame));
  CUDA_CHECK(av2enc::cuda::still::recon_init_state(gpu.state));

  // --- drive the shadow loop, committing GPU state per block -------------
  DriverOutput out;
  AvailabilityMismatch mismatch;
  if (!drive_frame(picture, options, gpu, out, mismatch, error)) {
    std::fprintf(stderr, "FAIL drive: %s\n", error.c_str());
    ++g_failures;
    ++g_checks;
    return result;
  }
  CHECK(!mismatch.any);
  if (mismatch.any) {
    std::fprintf(stderr,
                 "  availability mismatch r=%d c=%d plane=%d cpu=%d/%d "
                 "gpu=%d/%d\n",
                 mismatch.row, mismatch.col, mismatch.plane, mismatch.cpu_tr,
                 mismatch.cpu_bl, mismatch.gpu_tr, mismatch.gpu_bl);
  }
  CHECK(out.planes.size() == static_cast<std::size_t>(out.infos.size()) * 3u);

  // Shadow CPU recon must equal the official CPU reference (unfiltered).
  for (int p = 0; p < 3; ++p) {
    const int pw = p == 0 ? w : (w + 1) / 2;
    const int ph = p == 0 ? h : (h + 1) / 2;
    const std::vector<std::uint8_t>& want =
        p == 0 ? ref_plain.recon_y
               : (p == 1 ? ref_plain.recon_u : ref_plain.recon_v);
    CHECK(compare_visible(out.recon[p], out.stride[p], want, pw, ph,
                          "shadow-vs-StillResult-unfiltered"));
  }

  // --- upload all captured block jobs and batch one recon launch ----------
  std::size_t total_pels = 0;
  for (const CapturedPlane& cp : out.planes) {
    total_pels += static_cast<std::size_t>(cp.w) * cp.h;
  }
  std::vector<std::uint8_t> pred_all(total_pels);
  std::vector<std::int16_t> res_all(total_pels);
  std::vector<std::size_t> pred_off(out.planes.size(), 0);
  std::size_t off = 0;
  for (std::size_t i = 0; i < out.planes.size(); ++i) {
    pred_off[i] = off;
    const CapturedPlane& cp = out.planes[i];
    std::copy(cp.pred.begin(), cp.pred.end(), pred_all.begin() + off);
    std::copy(cp.residual.begin(), cp.residual.end(), res_all.begin() + off);
    off += static_cast<std::size_t>(cp.w) * cp.h;
  }
  std::uint8_t* d_pred = nullptr;
  std::int16_t* d_res = nullptr;
  ReconWriteJob* d_jobs = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pred), total_pels));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_res),
                        total_pels * sizeof(std::int16_t)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_jobs),
                        sizeof(ReconWriteJob) * out.planes.size()));
  CUDA_CHECK(
      cudaMemcpy(d_pred, pred_all.data(), total_pels, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_res, res_all.data(),
                        total_pels * sizeof(std::int16_t),
                        cudaMemcpyHostToDevice));
  std::vector<ReconWriteJob> host_jobs(out.planes.size());
  for (std::size_t i = 0; i < out.planes.size(); ++i) {
    const CapturedPlane& cp = out.planes[i];
    ReconWriteJob j;
    j.pred = d_pred + pred_off[i];
    j.residual = d_res + pred_off[i];
    j.plane = planes.data[cp.plane];
    j.stride = planes.stride[cp.plane];
    j.height = planes.height[cp.plane];
    j.x0 = cp.x0;
    j.y0 = cp.y0;
    j.w = cp.w;
    j.h = cp.h;
    host_jobs[i] = j;
  }
  CUDA_CHECK(cudaMemcpy(d_jobs, host_jobs.data(),
                        sizeof(ReconWriteJob) * host_jobs.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(av2enc::cuda::still::recon_write_blocks(
      d_jobs, static_cast<int>(host_jobs.size())));
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- compare GPU padded planes against the shadow ----------------------
  std::vector<std::uint8_t> gpu_plane[3];
  for (int p = 0; p < 3; ++p) {
    const std::size_t bytes = static_cast<std::size_t>(planes.stride[p]) *
                              static_cast<std::size_t>(planes.height[p]);
    gpu_plane[p].resize(bytes);
    CUDA_CHECK(cudaMemcpy(gpu_plane[p].data(), planes.data[p], bytes,
                          cudaMemcpyDeviceToHost));
    CHECK(compare_u8(gpu_plane[p].data(), out.recon[p].data(), bytes,
                     "gpu-vs-cpu-padded-recon"));
    const int pw = p == 0 ? w : (w + 1) / 2;
    const int ph = p == 0 ? h : (h + 1) / 2;
    const std::vector<std::uint8_t>& want =
        p == 0 ? ref_plain.recon_y
               : (p == 1 ? ref_plain.recon_u : ref_plain.recon_v);
    CHECK(compare_visible(gpu_plane[p], planes.stride[p], want, pw, ph,
                          "gpu-vs-StillResult-unfiltered"));
  }

  // --- compare prediction state ------------------------------------------
  {
    const int n_joint = mi_cols * mi_rows;
    const std::size_t n_uv = static_cast<std::size_t>(mi_cols / 2) *
                             static_cast<std::size_t>(mi_rows / 2);
    std::vector<int> joint(n_joint);
    std::vector<std::uint8_t> uv_smooth(n_uv);
    CUDA_CHECK(cudaMemcpy(joint.data(), gpu.joint, sizeof(int) * n_joint,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uv_smooth.data(), gpu.uv_smooth, n_uv,
                          cudaMemcpyDeviceToHost));
    CHECK(compare_int(joint.data(), out.intra_joint.data(), n_joint,
                      "intra_joint"));
    CHECK(compare_u8(uv_smooth.data(), out.uv_smooth.data(), n_uv,
                     "uv_smooth"));
    for (int p = 0; p < 3; ++p) {
      const std::size_t n = static_cast<std::size_t>(gpu.state.decoded_cols[p]) *
                            gpu.state.decoded_rows[p];
      std::vector<std::uint8_t> dec(n);
      CUDA_CHECK(
          cudaMemcpy(dec.data(), gpu.decoded[p], n, cudaMemcpyDeviceToHost));
      CHECK(compare_u8(dec.data(), out.decoded[p].data(), n, "decoded"));
    }
  }

  // --- compare deblock metadata ------------------------------------------
  for (int p = 0; p < 3; ++p) {
    const std::size_t n = static_cast<std::size_t>(mi_cols) * mi_rows;
    std::vector<DeblockMi> grid(n);
    CUDA_CHECK(cudaMemcpy(grid.data(), mi.grid[p], sizeof(DeblockMi) * n,
                          cudaMemcpyDeviceToHost));
    ++g_checks;
    int bad = -1;
    for (std::size_t i = 0; i < n; ++i) {
      const DeblockMi& a = grid[i];
      const DeblockMi& b = g_mi_expected[p][i];
      if (a.block_col_base != b.block_col_base ||
          a.block_row_base != b.block_row_base || a.block_w4 != b.block_w4 ||
          a.block_h4 != b.block_h4 || a.sub_pu_size != b.sub_pu_size ||
          a.tx_size != b.tx_size || a.tx_col_base != b.tx_col_base ||
          a.tx_row_base != b.tx_row_base ||
          a.sub_pu_col_base != b.sub_pu_col_base ||
          a.sub_pu_row_base != b.sub_pu_row_base || a.skip != b.skip ||
          a.region_intra != b.region_intra || a.lossless != b.lossless ||
          a.qindex != b.qindex) {
        bad = static_cast<int>(i);
        break;
      }
    }
    if (bad >= 0) {
      ++g_failures;
      std::fprintf(stderr, "FAIL deblock mi grid plane %d differs at %d\n", p,
                   bad);
    }
  }

  // --- deblock ------------------------------------------------------------
  DeblockOptions dopts;
  for (int i = 0; i < 4; ++i) dopts.apply_deblocking_filter[i] = true;
  dopts.frame_is_intra = true;
  dopts.enable_sdp = false;

  if (enable_deblock) {
    std::vector<std::uint8_t> cpu_filtered[3];
    av2enc::deblock::DeblockFrame cpu_frame;
    for (int p = 0; p < 3; ++p) {
      cpu_filtered[p] = out.recon[p];
      cpu_frame.planes[p].data = cpu_filtered[p].data();
      cpu_frame.planes[p].stride = out.stride[p];
      cpu_frame.mi[p] = g_mi_expected[p].data();
    }
    cpu_frame.mi_cols = mi_cols;
    cpu_frame.mi_rows = mi_rows;
    cpu_frame.subsampling_x = 1;
    cpu_frame.subsampling_y = 1;
    const av2enc::deblock::DeblockStatus st =
        av2enc::deblock::deblock_frame(cpu_frame, dopts);
    CHECK(st == av2enc::deblock::DeblockStatus::kOk);
    for (int p = 0; p < 3; ++p) {
      const int pw = p == 0 ? w : (w + 1) / 2;
      const int ph = p == 0 ? h : (h + 1) / 2;
      const std::vector<std::uint8_t>& want =
          p == 0 ? ref_db.recon_y
                 : (p == 1 ? ref_db.recon_u : ref_db.recon_v);
      CHECK(compare_visible(cpu_filtered[p], out.stride[p], want, pw, ph,
                            "cpu-deblocked-vs-StillResult"));
    }

    CUDA_CHECK(av2enc::cuda::still::recon_deblock_frame(g_frame, dopts));
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int p = 0; p < 3; ++p) {
      const std::size_t bytes = static_cast<std::size_t>(planes.stride[p]) *
                                planes.height[p];
      std::vector<std::uint8_t> filtered(bytes);
      CUDA_CHECK(cudaMemcpy(filtered.data(), planes.data[p], bytes,
                            cudaMemcpyDeviceToHost));
      CHECK(compare_u8(filtered.data(), cpu_filtered[p].data(), bytes,
                       "gpu-deblocked-vs-cpu-deblocked-padded"));
      const int pw = p == 0 ? w : (w + 1) / 2;
      const int ph = p == 0 ? h : (h + 1) / 2;
      const std::vector<std::uint8_t>& want =
          p == 0 ? ref_db.recon_y
                 : (p == 1 ? ref_db.recon_u : ref_db.recon_v);
      CHECK(compare_visible(filtered, planes.stride[p], want, pw, ph,
                            "gpu-deblocked-vs-StillResult"));
    }
  } else {
    for (int p = 0; p < 3; ++p) {
      const int pw = p == 0 ? w : (w + 1) / 2;
      const int ph = p == 0 ? h : (h + 1) / 2;
      const std::vector<std::uint8_t>& want =
          p == 0 ? ref_plain.recon_y
                 : (p == 1 ? ref_plain.recon_u : ref_plain.recon_v);
      CHECK(compare_visible(gpu_plane[p], planes.stride[p], want, pw, ph,
                            "cpu-driver-vs-StillResult-no-deblock"));
    }
  }

  // --- timing -------------------------------------------------------------
  if (time_it) {
    cudaEvent_t e0;
    cudaEvent_t e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    constexpr int kIters = 50;
    for (int i = 0; i < 5; ++i) {
      av2enc::cuda::still::recon_write_blocks(
          d_jobs, static_cast<int>(host_jobs.size()));
    }
    cudaEventRecord(e0);
    for (int i = 0; i < kIters; ++i) {
      av2enc::cuda::still::recon_write_blocks(
          d_jobs, static_cast<int>(host_jobs.size()));
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    result.write_ms = ms / kIters;
    for (int i = 0; i < 3; ++i) {
      av2enc::cuda::still::recon_deblock_frame(g_frame, dopts);
    }
    cudaEventRecord(e0);
    for (int i = 0; i < kIters; ++i) {
      av2enc::cuda::still::recon_deblock_frame(g_frame, dopts);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    result.deblock_ms = ms / kIters;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
  }

  cudaFree(d_pred);
  cudaFree(d_res);
  cudaFree(d_jobs);
  return result;
}

// ---------------------------------------------------------------------------
// Synthetic padded-edge test
// ---------------------------------------------------------------------------

bool synthetic_test() {
  const int stride = 80;
  const int height = 80;
  const int n = stride * height;
  std::vector<std::uint8_t> ref(n, 7);
  std::uint8_t* dev = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev), n));
  CUDA_CHECK(cudaMemcpy(dev, ref.data(), n, cudaMemcpyHostToDevice));

  struct JobSpec {
    int x0, y0, w, h;
    int pred_base;
    int res_base;
  };
  const JobSpec js[3] = {
      {0, 0, 32, 32, 16, 0},
      {32, 0, 32, 32, 240, 40},  // saturates high
      {64, 64, 32, 32, 4, -40},  // clamps at the padded right/bottom edge
  };
  const int w = 32;
  const int h = 32;
  std::vector<ReconWriteJob> jobs(3);
  std::vector<std::uint8_t*> d_pred(3, nullptr);
  std::vector<std::int16_t*> d_res(3, nullptr);
  for (int i = 0; i < 3; ++i) {
    std::vector<std::uint8_t> pred(static_cast<std::size_t>(w) * h);
    std::vector<std::int16_t> res(static_cast<std::size_t>(w) * h);
    for (int k = 0; k < w * h; ++k) {
      pred[k] = static_cast<std::uint8_t>(js[i].pred_base + (k % 9));
      res[k] = static_cast<std::int16_t>(js[i].res_base + ((k % 13) - 6));
      const int v = static_cast<int>(pred[k]) + static_cast<int>(res[k]);
      const int c = v < 0 ? 0 : (v > 255 ? 255 : v);
      const int x = js[i].x0 + (k % w);
      const int y = js[i].y0 + (k / w);
      if (x >= 0 && x < stride && y >= 0 && y < height) {
        ref[static_cast<std::size_t>(y) * stride + x] =
            static_cast<std::uint8_t>(c);
      }
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pred[i]), pred.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_res[i]),
                          res.size() * sizeof(std::int16_t)));
    CUDA_CHECK(cudaMemcpy(d_pred[i], pred.data(), pred.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res[i], res.data(),
                          res.size() * sizeof(std::int16_t),
                          cudaMemcpyHostToDevice));
    jobs[i].pred = d_pred[i];
    jobs[i].residual = d_res[i];
    jobs[i].plane = dev;
    jobs[i].stride = stride;
    jobs[i].height = height;
    jobs[i].x0 = js[i].x0;
    jobs[i].y0 = js[i].y0;
    jobs[i].w = w;
    jobs[i].h = h;
  }

  ReconWriteJob* d_jobs = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_jobs),
                        sizeof(ReconWriteJob) * 3));
  CUDA_CHECK(cudaMemcpy(d_jobs, jobs.data(), sizeof(ReconWriteJob) * 3,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(av2enc::cuda::still::recon_write_blocks(d_jobs, 3));
  std::vector<std::uint8_t> got(n, 0);
  CUDA_CHECK(cudaMemcpy(got.data(), dev, n, cudaMemcpyDeviceToHost));
  CHECK(compare_u8(got.data(), ref.data(), ref.size(), "synthetic-batch"));

  std::uint8_t* dev2 = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev2), n));
  std::vector<std::uint8_t> init(n, 7);
  CUDA_CHECK(cudaMemcpy(dev2, init.data(), n, cudaMemcpyHostToDevice));
  for (int i = 0; i < 3; ++i) {
    jobs[i].plane = dev2;
    CUDA_CHECK(av2enc::cuda::still::recon_write_block(jobs[i]));
  }
  std::vector<std::uint8_t> got_serial(n, 0);
  CUDA_CHECK(cudaMemcpy(got_serial.data(), dev2, n, cudaMemcpyDeviceToHost));
  CHECK(compare_u8(got_serial.data(), ref.data(), ref.size(),
                   "synthetic-serial"));

  for (int i = 0; i < 3; ++i) {
    cudaFree(d_pred[i]);
    cudaFree(d_res[i]);
  }
  cudaFree(d_jobs);
  cudaFree(dev);
  cudaFree(dev2);
  return true;
}

// ---------------------------------------------------------------------------
// Tile-pass-through test: the wrapper must forward MiColStarts/MiRowStarts so
// §7.17 tile edges behave exactly like the CPU reference.
// ---------------------------------------------------------------------------

bool tile_deblock_test() {
  const int mi_cols = 16;
  const int mi_rows = 16;
  const int luma_w = mi_cols * 4;
  const int luma_h = mi_rows * 4;
  const int chroma_w = mi_cols * 2;
  const int chroma_h = mi_rows * 2;
  const std::uint16_t tile_cols[2] = {0, 8};

  std::vector<std::uint8_t> host[3];
  const int pw[3] = {luma_w, chroma_w, chroma_w};
  const int ph[3] = {luma_h, chroma_h, chroma_h};
  std::uint32_t seed = 12345;
  for (int p = 0; p < 3; ++p) {
    host[p].resize(static_cast<std::size_t>(pw[p]) * ph[p]);
    for (int y = 0; y < ph[p]; ++y) {
      for (int x = 0; x < pw[p]; ++x) {
        seed = seed * 1664525u + 1013904223u;
        host[p][static_cast<std::size_t>(y) * pw[p] + x] =
            static_cast<std::uint8_t>(((x * 3 + y * 5) & 0x3F) +
                                      ((seed >> 28) & 0x0F));
      }
    }
  }

  std::vector<DeblockMi> mi(static_cast<std::size_t>(mi_cols) * mi_rows);
  for (int r = 0; r < mi_rows; r += 2) {
    for (int c = 0; c < mi_cols; c += 2) {
      for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
          DeblockMi& m = mi[static_cast<std::size_t>(r + y) * mi_cols + c + x];
          m.block_col_base = static_cast<std::uint16_t>(c);
          m.block_row_base = static_cast<std::uint16_t>(r);
          m.block_w4 = 2;
          m.block_h4 = 2;
          m.sub_pu_size = av2enc::deblock::TxSize::k8x8;
          m.tx_size = av2enc::deblock::TxSize::k8x8;
          m.tx_col_base = static_cast<std::uint16_t>(c);
          m.tx_row_base = static_cast<std::uint16_t>(r);
          m.sub_pu_col_base = static_cast<std::uint16_t>(c);
          m.sub_pu_row_base = static_cast<std::uint16_t>(r);
          m.skip = 0;
          m.region_intra = 1;
          m.lossless = 0;
          m.qindex = 90;
        }
      }
    }
  }

  DeblockOptions dopts;
  for (int i = 0; i < 4; ++i) dopts.apply_deblocking_filter[i] = true;
  dopts.frame_is_intra = true;
  dopts.enable_sdp = false;

  // CPU reference with tile starts.
  std::vector<std::uint8_t> cpu[3];
  av2enc::deblock::DeblockFrame cpu_frame;
  for (int p = 0; p < 3; ++p) {
    cpu[p] = host[p];
    cpu_frame.planes[p].data = cpu[p].data();
    cpu_frame.planes[p].stride = pw[p];
    cpu_frame.mi[p] = mi.data();
  }
  cpu_frame.mi_cols = mi_cols;
  cpu_frame.mi_rows = mi_rows;
  cpu_frame.subsampling_x = 1;
  cpu_frame.subsampling_y = 1;
  cpu_frame.tile_col_starts = tile_cols;
  cpu_frame.num_tile_cols = 2;
  CHECK(av2enc::deblock::deblock_frame(cpu_frame, dopts) ==
        av2enc::deblock::DeblockStatus::kOk);

  // GPU through the recon wrapper, same tile starts.
  ReconFrame rf;
  std::uint8_t* dev[3] = {nullptr, nullptr, nullptr};
  DeblockMi* d_mi = nullptr;
  std::uint16_t* d_tile_cols = nullptr;
  bool ok = true;
  for (int p = 0; p < 3; ++p) {
    if (cudaMalloc(reinterpret_cast<void**>(&dev[p]),
                   host[p].size()) != cudaSuccess) {
      ok = false;
    } else if (cudaMemcpy(dev[p], host[p].data(), host[p].size(),
                          cudaMemcpyHostToDevice) != cudaSuccess) {
      ok = false;
    }
    rf.plane[p].data = dev[p];
    rf.plane[p].stride = pw[p];
    rf.plane[p].height = ph[p];
    rf.deblock_mi[p] = nullptr;
  }
  if (ok && cudaMalloc(reinterpret_cast<void**>(&d_mi),
                       sizeof(DeblockMi) * mi.size()) == cudaSuccess) {
    if (cudaMemcpy(d_mi, mi.data(), sizeof(DeblockMi) * mi.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      ok = false;
    }
  } else {
    ok = false;
  }
  for (int p = 0; p < 3; ++p) rf.deblock_mi[p] = d_mi;
  rf.mi_cols = mi_cols;
  rf.mi_rows = mi_rows;
  rf.base_q_idx = 90;
  if (ok &&
      cudaMalloc(reinterpret_cast<void**>(&d_tile_cols),
                 sizeof(tile_cols)) == cudaSuccess) {
    if (cudaMemcpy(d_tile_cols, tile_cols, sizeof(tile_cols),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      ok = false;
    }
  } else {
    ok = false;
  }
  rf.tile_col_starts = d_tile_cols;
  rf.num_tile_cols = 2;
  CHECK(ok);
  if (ok) {
    CHECK(av2enc::cuda::still::recon_deblock_frame(rf, dopts) == cudaSuccess);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    for (int p = 0; p < 3; ++p) {
      std::vector<std::uint8_t> got(host[p].size());
      CUDA_CHECK(cudaMemcpy(got.data(), dev[p], got.size(),
                            cudaMemcpyDeviceToHost));
      CHECK(compare_u8(got.data(), cpu[p].data(), cpu[p].size(),
                       "tile-deblock-gpu-vs-cpu"));
    }
  }
  for (int p = 0; p < 3; ++p) {
    if (dev[p]) cudaFree(dev[p]);
  }
  if (d_mi) cudaFree(d_mi);
  if (d_tile_cols) cudaFree(d_tile_cols);
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  const std::string root = argc > 1 ? argv[1] : ".";
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    std::fprintf(stderr, "no CUDA device\n");
    return 2;
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  std::printf("device: %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);

  synthetic_test();
  tile_deblock_test();

  const std::filesystem::path data =
      std::filesystem::path(root) / "tests" / "data";
  struct Case {
    const char* file;
    bool deblock;
    bool time;
  };
  const Case cases[4] = {
      {"basic_64x64_1f.y4m", true, false},
      {"basic_64x64_1f.y4m", false, false},
      {"basic_640x360_1f.y4m", true, true},
      {"basic_640x360_1f.y4m", false, false},
  };
  for (const Case& c : cases) {
    const int before = g_failures;
    const CaseResult r = run_case(data / c.file, c.deblock, c.time);
    std::printf("case %-20s deblock=%d leaves=%d checks=%d failures=%d%s\n",
                c.file, c.deblock ? 1 : 0, r.leaves, g_checks,
                g_failures - before, (g_failures == before) ? " OK" : " FAIL");
    if (c.time) {
      std::printf("  recon write batch: %.4f ms   deblock: %.4f ms\n",
                  r.write_ms, r.deblock_ms);
    }
  }

  std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
