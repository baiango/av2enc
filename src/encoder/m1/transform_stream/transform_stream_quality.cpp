// Real-residual quality probe for the M1 transform-stream adapter.
//
// It replays the M1 partition tree and TX geometry on the pinned test frames
// with the M1 default DC intra mode, maintaining a reconstruction plane so the
// prediction of every block uses its own decoded neighbours (the same
// trajectory a decoder sees).  Three variants are replayed independently:
//
//   0. scalar quantisation (DCT_DCT)
//   1. adapter RDOQ, DCT_DCT only
//   2. adapter RDOQ + full intra TX-type selection (enable_idtx_intra = 0)
//
// For each variant it builds a real §8.2 syntax stream with
// `m1::syntax::assemble_m1_frame` (BlockSyntax::tx_type carries the chosen
// type), reports the actual tile bytes, and computes the visible-sample PSNR
// against the source.  The streams are decoder-valid; use --write-ivf and
// tools/validate_dav2d.sh to gate them.
//
// Usage:
//   ./transform_stream_quality [--input FILE] [--tx-types]
//                              [--sse-scale X] [--qp N] [--depth N]
//                              [--min-block N]
//                              [--enable-idtx] [--write-ivf PREFIX]
//
// This is a quantisation / TX-type measurement on a fixed partition tree and
// DC mode; it is not a full-encoder BD-rate claim.
#include "encoder/m1/transform_stream/transform_stream.h"

#include "cuda/m1/plan/plan.h"
#include "encoder/intra_modes/intra_modes.h"
#include "encoder/m1/m1_headers.h"
#include "encoder/m1/m1_syntax.h"
#include "encoder/rdoq/rdoq_av2_model.h"
#include "encoder/tx_type/tx_type.h"
#include "encoder/tx_variants/tx_variants.h"
#include "io/ivf.h"
#include "io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace ts = av2enc::m1::ts;
using av2enc::txtype::TxType;
using av2enc::txv::TxParams;

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

struct Options {
  std::string input;
  bool tx_types = false;
  double sse_scale = 0.125;
  int qp = 90;
  int depth = 4;
  int min_block = 8;
  bool enable_idtx = false;  // M1 signals enable_idtx_intra = 0
  std::string write_ivf;     // prefix
};

struct Plane {
  int width = 0;   // visible width
  int height = 0;  // visible height
  int stride = 0;  // allocation stride
  int rows = 0;    // allocation rows
  std::vector<std::uint8_t> samples;

  void reset() { std::fill(samples.begin(), samples.end(), 0); }
  std::uint8_t at(int x, int y) const {
    return samples[static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(stride) +
                   static_cast<std::size_t>(x)];
  }
  void set(int x, int y, std::uint8_t value) {
    samples[static_cast<std::size_t>(y) * static_cast<std::size_t>(stride) +
            static_cast<std::size_t>(x)] = value;
  }
};

std::uint8_t source_sample(const Plane& plane, int x, int y) {
  const int cx = clip3(0, plane.width - 1, x);
  const int cy = clip3(0, plane.height - 1, y);
  return plane.samples[static_cast<std::size_t>(cy) *
                           static_cast<std::size_t>(plane.stride) +
                       static_cast<std::size_t>(cx)];
}

bool predict_dc(const Plane& recon, int x, int y, int w, int h, int max_x,
                int max_y, int color_plane, std::vector<std::uint8_t>* pred) {
  av2enc::intra::NeighborRequest request;
  request.recon.data = recon.samples.data();
  request.recon.stride = recon.stride;
  request.x = x;
  request.y = y;
  request.w = w;
  request.h = h;
  request.max_x = max_x;
  request.max_y = max_y;
  request.have_above = y > 0;
  request.have_left = x > 0;
  request.color_plane = color_plane;
  av2enc::intra::PredictParams params;
  params.mode = av2enc::intra::Mode::kDc;
  pred->resize(static_cast<std::size_t>(w * h));
  return av2enc::intra::predict_block(request, 8, params, pred->data(),
                                      static_cast<std::size_t>(w)) ==
         av2enc::intra::IntraStatus::kOk;
}

struct PlaneVariant {
  int eob = 0;
  std::vector<std::int32_t> levels_scan;
  std::vector<std::int16_t> recon_coded;
  double sse = 0.0;
  double bits = 0.0;
  int nonzero = 0;
};

PlaneVariant scalar_plane(const std::vector<std::int16_t>& residual, int w,
                          int h, int plane_index) {
  PlaneVariant out;
  const TxParams params = *av2enc::txv::make_tx_params(w, h);
  const int area = params.w * params.h;
  std::vector<std::int16_t> adj(static_cast<std::size_t>(area));
  std::vector<double> coeff(static_cast<std::size_t>(area));
  std::vector<std::int32_t> raster(static_cast<std::size_t>(area));
  av2enc::txv::downsample_residual(residual.data(), params, adj.data());
  av2enc::txv::forward_transform(adj.data(), params, coeff.data(),
                                 raster.data());
  std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
  av2enc::txtype::diagonal_scan(params.w, params.h, scan.data());
  out.levels_scan.resize(static_cast<std::size_t>(area), 0);
  for (int c = 0; c < area; ++c) {
    const std::int32_t value =
        raster[static_cast<std::size_t>(scan[static_cast<std::size_t>(c)])];
    out.levels_scan[static_cast<std::size_t>(c)] = value;
    if (value != 0) {
      out.eob = c + 1;
      ++out.nonzero;
    }
  }
  std::vector<std::int32_t> dequant(static_cast<std::size_t>(area));
  out.recon_coded.resize(
      static_cast<std::size_t>(params.coded_w * params.coded_h));
  av2enc::txv::inverse_transform(raster.data(), params, dequant.data(),
                                 out.recon_coded.data());
  std::int64_t sse = 0;
  for (std::size_t i = 0; i < out.recon_coded.size(); ++i) {
    const std::int64_t d =
        static_cast<std::int64_t>(out.recon_coded[i]) - residual[i];
    sse += d * d;
  }
  out.sse = static_cast<double>(sse);

  av2enc::rdoq::av2::BlockInfo info;
  info.plane = plane_index;
  info.params = params;
  av2enc::rdoq::av2::Model model;
  std::string error;
  if (model.init(info, out.levels_scan, &error)) {
    out.bits = model.scalar_bits();
  }
  return out;
}

PlaneVariant adapter_plane(const std::vector<std::int16_t>& residual, int w,
                           int h, double lambda, bool full_types,
                           bool enable_idtx, int plane_index,
                           ts::Decision* decision) {
  ts::Request request;
  request.residual_coded = residual.data();
  request.coded_w = w;
  request.coded_h = h;
  request.intra_mode = 0;
  request.angle_delta = 0;
  request.reduced_tx_set = 0;
  request.plane = plane_index;
  request.txb_skip_context = 0;
  request.dc_sign_context = 0;
  request.lambda = lambda;
  request.enable_idtx_intra = enable_idtx;
  if (!full_types) {
    const TxType dct = TxType::kDctDct;
    request.candidate_types = std::span<const TxType>(&dct, 1);
  }
  std::string error;
  if (!ts::quantize_block(request, decision, &error)) {
    std::fprintf(stderr, "adapter failed: %s\n", error.c_str());
    std::exit(1);
  }
  PlaneVariant out;
  out.eob = decision->eob;
  out.levels_scan = decision->levels;
  out.recon_coded = decision->recon_coded;
  out.sse = decision->sse;
  out.bits = decision->rate_bits;
  out.nonzero = decision->nonzero;
  return out;
}

struct LeafData {
  // Replayed separately per variant (the reconstruction trajectory differs).
  std::vector<std::int16_t> luma_res[3];
  std::vector<std::uint8_t> luma_pred[3];
  ts::Decision luma_decision[3];
  PlaneVariant luma[3];
  PlaneVariant u[3];
  PlaneVariant v[3];
};

struct Stats {
  long long blocks = 0;
  long long samples = 0;
  long long nonzero = 0;
  long long eob_sum = 0;
  double residual_sse = 0.0;
  double visible_sse = 0.0;
  double bits = 0.0;
  double cost = 0.0;
  long long types[16] = {};
  std::uint32_t tile_bytes = 0;
  std::uint32_t frame_bytes = 0;
};

bool build_stream(const av2enc::m1::plan::FramePlan& plan,
                  const std::vector<LeafData>& leaves, const Options& options,
                  int which, std::vector<std::uint8_t>* frame,
                  std::uint32_t* tile_bytes, std::string* error) {
  std::vector<av2enc::m1::syntax::BlockSyntax> blocks;
  blocks.reserve(plan.tree.size());
  std::vector<std::int32_t> levels;

  for (const av2enc::m1::plan::TreeNode& node : plan.tree) {
    av2enc::m1::syntax::BlockSyntax entry;
    entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
    entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
    entry.bsize = node.block_size;
    entry.partition = node.partition;
    entry.allowed_mask = node.allowed_mask;
    entry.flags = 0;
    if (node.sb_root) entry.flags |= av2enc::m1::syntax::kFlagSbRoot;
    if (node.forced) entry.flags |= av2enc::m1::syntax::kFlagForced;
    entry.subtree = static_cast<std::uint16_t>(node.subtree);
    if (node.child_count == 0) {
      if (node.leaf_index >= leaves.size() ||
          node.leaf_index >= plan.blocks.size()) {
        *error = "plan leaf index outside the replay data";
        return false;
      }
      const LeafData& leaf = leaves[node.leaf_index];
      const av2enc::m1::plan::BlockPlan& block = plan.blocks[node.leaf_index];
      entry.y_mode = 0;
      entry.y_angle_delta = 0;
      entry.uv_mode = 0;
      entry.levels_offset = static_cast<std::uint32_t>(levels.size());
      entry.tx_size[0] = static_cast<std::uint8_t>(block.luma.tx_size);
      entry.tx_type[0] = static_cast<std::uint8_t>(
          which == 2 ? static_cast<int>(leaf.luma_decision[which].type) : 0);
      entry.eob[0] = static_cast<std::uint32_t>(leaf.luma[which].eob);
      levels.insert(levels.end(), leaf.luma[which].levels_scan.begin(),
                    leaf.luma[which].levels_scan.begin() +
                        leaf.luma[which].eob);
      entry.tx_size[1] = static_cast<std::uint8_t>(block.u.tx_size);
      entry.tx_type[1] = 0;
      entry.eob[1] = static_cast<std::uint32_t>(leaf.u[which].eob);
      levels.insert(levels.end(), leaf.u[which].levels_scan.begin(),
                    leaf.u[which].levels_scan.begin() + leaf.u[which].eob);
      entry.tx_size[2] = static_cast<std::uint8_t>(block.v.tx_size);
      entry.tx_type[2] = 0;
      entry.eob[2] = static_cast<std::uint32_t>(leaf.v[which].eob);
      levels.insert(levels.end(), leaf.v[which].levels_scan.begin(),
                    leaf.v[which].levels_scan.begin() + leaf.v[which].eob);
    }
    blocks.push_back(entry);
  }

  av2enc::m1::M1Config config;
  config.max_frame_width = static_cast<std::uint16_t>(plan.width);
  config.max_frame_height = static_cast<std::uint16_t>(plan.height);
  config.base_q_idx = static_cast<std::uint8_t>(options.qp);
  config.disable_cdf_update = true;
  config.enable_intra_edge_filter = true;
  config.enable_cfl_intra = false;
  config.enable_chroma_dctonly = true;
  config.tx_mode_select = false;
  config.apply_deblock_y_vert = false;
  config.apply_deblock_y_horz = false;
  config.apply_deblock_u = false;
  config.apply_deblock_v = false;

  av2enc::m1::syntax::FramePlan frame_plan;
  frame_plan.width = plan.width;
  frame_plan.height = plan.height;
  frame_plan.sb_size = plan.sb_size;
  std::size_t encoded_tile_bytes = 0;
  if (!av2enc::m1::syntax::assemble_m1_frame(frame_plan, blocks, levels,
                                             config, *frame, *error,
                                             &encoded_tile_bytes)) {
    return false;
  }
  *tile_bytes = static_cast<std::uint32_t>(encoded_tile_bytes);
  return true;
}

void write_plane_block(Plane* recon, int x0, int y0, int w, int h,
                       const std::vector<std::uint8_t>& pred,
                       const std::vector<std::int16_t>& recon_res) {
  for (int y = 0; y < h; ++y) {
    const int py = y0 + y;
    if (py < 0 || py >= recon->rows) continue;
    for (int x = 0; x < w; ++x) {
      const int px = x0 + x;
      if (px < 0 || px >= recon->stride) continue;
      const std::size_t local =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
          static_cast<std::size_t>(x);
      const int p = pred[local];
      const int r = recon_res[local];
      recon->set(px, py, static_cast<std::uint8_t>(clip3(0, 255, p + r)));
    }
  }
}

bool replay_variant(const av2enc::m1::plan::FramePlan& plan,
                    const Plane& luma, const Plane& u, const Plane& v,
                    const Options& options, int which,
                    std::vector<LeafData>* leaves) {
  Plane recon_luma;
  recon_luma.stride = plan.luma_stride;
  recon_luma.rows = plan.luma_padded_height;
  recon_luma.samples.assign(
      static_cast<std::size_t>(recon_luma.stride * recon_luma.rows), 0);
  Plane recon_u;
  recon_u.stride = plan.chroma_stride;
  recon_u.rows = plan.chroma_padded_height;
  recon_u.samples.assign(
      static_cast<std::size_t>(recon_u.stride * recon_u.rows), 0);
  Plane recon_v = recon_u;

  std::vector<std::uint8_t> pred;
  for (std::size_t index = 0; index < plan.blocks.size(); ++index) {
    const av2enc::m1::plan::BlockPlan& block = plan.blocks[index];
    LeafData& leaf = (*leaves)[index];

    // --- luma ---
    if (!predict_dc(recon_luma, block.x, block.y, block.luma_w, block.luma_h,
                    plan.coded_luma_w - 1, plan.coded_luma_h - 1, 0, &pred)) {
      std::fprintf(stderr, "luma prediction failed\n");
      return false;
    }
    leaf.luma_pred[which] = pred;
    leaf.luma_res[which].resize(
        static_cast<std::size_t>(block.luma_w * block.luma_h));
    for (int y = 0; y < block.luma_h; ++y) {
      for (int x = 0; x < block.luma_w; ++x) {
        const int src = source_sample(luma, block.x + x, block.y + y);
        const int p = pred[static_cast<std::size_t>(y * block.luma_w + x)];
        leaf.luma_res[which][static_cast<std::size_t>(y * block.luma_w + x)] =
            static_cast<std::int16_t>(src - p);
      }
    }
    const TxParams params = *av2enc::txv::make_tx_params(
        block.luma.coded_w, block.luma.coded_h);
    const double lambda =
        ts::default_lambda(params, options.sse_scale);
    if (which == 0) {
      leaf.luma[which] = scalar_plane(leaf.luma_res[which],
                                      block.luma.coded_w, block.luma.coded_h, 0);
    } else {
      leaf.luma[which] = adapter_plane(
          leaf.luma_res[which], block.luma.coded_w, block.luma.coded_h,
          lambda, /*full_types=*/which == 2, options.enable_idtx, 0,
          &leaf.luma_decision[which]);
    }
    write_plane_block(&recon_luma, block.x, block.y, block.luma_w,
                      block.luma_h, leaf.luma_pred[which],
                      leaf.luma[which].recon_coded);

    // --- chroma (scalar DCT_DCT, DC prediction from decoded chroma) ---
    for (int plane_index = 1; plane_index <= 2; ++plane_index) {
      const Plane& source = plane_index == 1 ? u : v;
      Plane& recon = plane_index == 1 ? recon_u : recon_v;
      PlaneVariant& out = plane_index == 1 ? leaf.u[which] : leaf.v[which];
      const int cw = block.chroma_w;
      const int ch = block.chroma_h;
      if (!predict_dc(recon, block.chroma_x, block.chroma_y, cw, ch,
                      plan.coded_chroma_w - 1, plan.coded_chroma_h - 1,
                      plane_index, &pred)) {
        std::fprintf(stderr, "chroma prediction failed\n");
        return false;
      }
      std::vector<std::int16_t> residual(
          static_cast<std::size_t>(cw * ch));
      for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
          const int src =
              source_sample(source, block.chroma_x + x, block.chroma_y + y);
          const int p = pred[static_cast<std::size_t>(y * cw + x)];
          residual[static_cast<std::size_t>(y * cw + x)] =
              static_cast<std::int16_t>(src - p);
        }
      }
      out = scalar_plane(residual, cw, ch, plane_index);
      write_plane_block(&recon, block.chroma_x, block.chroma_y, cw, ch, pred,
                        out.recon_coded);
    }
  }
  return true;
}

Stats collect_stats(const av2enc::m1::plan::FramePlan& plan,
                    const std::vector<LeafData>& leaves, const Plane& luma,
                    const Options& options, int which) {
  Stats stats;
  stats.blocks = static_cast<long long>(leaves.size());
  for (std::size_t index = 0; index < leaves.size(); ++index) {
    const LeafData& leaf = leaves[index];
    const av2enc::m1::plan::BlockPlan& block = plan.blocks[index];
    const PlaneVariant& chosen = leaf.luma[which];
    stats.nonzero += chosen.nonzero;
    stats.eob_sum += chosen.eob;
    stats.residual_sse += chosen.sse;
    stats.bits += chosen.bits;
    TxParams params = *av2enc::txv::make_tx_params(
        block.luma.coded_w, block.luma.coded_h);
    if (which != 0) params = leaf.luma_decision[which].params;
    stats.cost += chosen.sse +
                  ts::default_lambda(params, options.sse_scale) * chosen.bits;
    if (which == 2) {
      const int type = static_cast<int>(leaf.luma_decision[which].type);
      if (type >= 0 && type < 16) ++stats.types[type];
    }

    for (int y = 0; y < block.luma_h; ++y) {
      const int py = block.y + y;
      if (py < 0 || py >= luma.height) continue;
      for (int x = 0; x < block.luma_w; ++x) {
        const int px = block.x + x;
        if (px < 0 || px >= luma.width) continue;
        const std::size_t local =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(block.luma_w) +
            static_cast<std::size_t>(x);
        const int p = leaf.luma_pred[which][local];
        const int r = chosen.recon_coded[local];
        const int value = clip3(0, 255, p + r);
        const int src = luma.samples[static_cast<std::size_t>(py) *
                                         static_cast<std::size_t>(luma.stride) +
                                     static_cast<std::size_t>(px)];
        const double difference = static_cast<double>(value - src);
        stats.visible_sse += difference * difference;
        ++stats.samples;
      }
    }
  }
  (void)options;
  return stats;
}

double psnr(double sse, long long samples) {
  if (sse <= 0.0 || samples <= 0) return 0.0;
  return 10.0 * std::log10(255.0 * 255.0 *
                           static_cast<double>(samples) / sse);
}

void print_stats(const char* label, const Stats& stats) {
  std::printf(
      "  %-18s bits %10.1f  cost %12.1f  PSNR %6.3f dB  nz %7lld"
      "  eob %8lld  tile %7u B\n",
      label, stats.bits, stats.cost, psnr(stats.visible_sse, stats.samples),
      stats.nonzero, stats.eob_sum, stats.tile_bytes);
}

bool write_ivf(const std::string& path, std::span<const std::uint8_t> frame,
               std::uint32_t width, std::uint32_t height, std::string* error) {
  av2enc::IvfHeader header;
  header.width = static_cast<std::uint16_t>(width);
  header.height = static_cast<std::uint16_t>(height);
  header.fps_num = 30;
  header.fps_den = 1;
  av2enc::IvfWriter writer;
  if (!writer.open(path, header, *error)) return false;
  if (!writer.write_frame(frame, 0, *error)) return false;
  return writer.close(*error);
}

bool run_file(const std::filesystem::path& path, const Options& options) {
  av2enc::Y4mReader reader(path);
  std::string error;
  if (!reader.open(error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return false;
  }
  av2enc::Y4mFrame y4m;
  if (reader.read_frame(y4m, error) != av2enc::Y4mReadResult::kFrame) {
    std::fprintf(stderr, "no frame: %s\n", error.c_str());
    return false;
  }
  const int width = static_cast<int>(y4m.width());
  const int height = static_cast<int>(y4m.height());
  if ((width & 1) != 0 || (height & 1) != 0) {
    std::fprintf(stderr, "4:2:0 needs even dimensions\n");
    return false;
  }

  Plane luma;
  luma.width = width;
  luma.height = height;
  luma.stride = width;
  luma.rows = height;
  const std::span<const std::uint8_t> y = y4m.y();
  luma.samples.assign(y.begin(), y.end());
  Plane u;
  u.width = (width + 1) / 2;
  u.height = (height + 1) / 2;
  u.stride = u.width;
  u.rows = u.height;
  const std::span<const std::uint8_t> uu = y4m.u();
  u.samples.assign(uu.begin(), uu.end());
  Plane v = u;
  const std::span<const std::uint8_t> vv = y4m.v();
  v.samples.assign(vv.begin(), vv.end());

  av2enc::m1::plan::PlanOptions plan_options;
  plan_options.base_q_idx = static_cast<std::uint8_t>(options.qp);
  plan_options.min_block_size = options.min_block;
  plan_options.max_partition_depth = options.depth;
  plan_options.allow_rect = true;
  plan_options.allow_split = false;
  av2enc::m1::plan::FramePlan plan;
  if (!av2enc::m1::plan::build_frame_plan(width, height, luma.samples,
                                          plan_options, plan, error)) {
    std::fprintf(stderr, "plan: %s\n", error.c_str());
    return false;
  }

  std::vector<LeafData> leaves(plan.blocks.size());
  const int variants = options.tx_types ? 3 : 2;
  for (int which = 0; which < variants; ++which) {
    if (!replay_variant(plan, luma, u, v, options, which, &leaves)) {
      return false;
    }
  }
  if (!options.tx_types) {
    // Variant 2 is unused; keep the statistics consistent.
    for (LeafData& leaf : leaves) {
      leaf.luma[2] = leaf.luma[1];
      leaf.u[2] = leaf.u[1];
      leaf.v[2] = leaf.v[1];
      leaf.luma_decision[2] = leaf.luma_decision[1];
    }
  }

  Stats stats[3];
  for (int which = 0; which < 3; ++which) {
    stats[which] = collect_stats(plan, leaves, luma, options, which);
  }
  std::vector<std::uint8_t> frames[3];
  for (int which = 0; which < 3; ++which) {
    if (which == 2 && !options.tx_types) {
      stats[2].tile_bytes = 0;
      stats[2].frame_bytes = 0;
      continue;
    }
    if (!build_stream(plan, leaves, options, which, &frames[which],
                      &stats[which].tile_bytes, &error)) {
      std::fprintf(stderr, "stream %d: %s\n", which, error.c_str());
      return false;
    }
    stats[which].frame_bytes =
        static_cast<std::uint32_t>(frames[which].size());
  }

  if (!options.write_ivf.empty()) {
    const char* suffix[3] = {".scalar.ivf", ".rdoq.ivf", ".txtype.ivf"};
    for (int which = 0; which < variants; ++which) {
      const std::string out = options.write_ivf + suffix[which];
      if (!write_ivf(out, frames[which], static_cast<std::uint32_t>(width),
                     static_cast<std::uint32_t>(height), &error)) {
        std::fprintf(stderr, "ivf: %s\n", error.c_str());
        return false;
      }
      std::printf("  wrote %s\n", out.c_str());
    }
  }

  std::printf("%s  %dx%d  qp=%d leaves=%d lambda_scale=%.3f\n",
              path.c_str(), width, height, options.qp, plan.leaves,
              options.sse_scale);
  print_stats("scalar", stats[0]);
  print_stats("RDOQ DCT-only", stats[1]);
  if (options.tx_types) print_stats("RDOQ + TX types", stats[2]);
  const auto saving = [](const Stats& base, const Stats& other) {
    return base.tile_bytes > 0
               ? 100.0 * (static_cast<double>(base.tile_bytes) -
                          static_cast<double>(other.tile_bytes)) /
                     static_cast<double>(base.tile_bytes)
               : 0.0;
  };
  std::printf("  tile bytes: scalar %u -> RDOQ-DCT %u (%.2f%%), frame %u -> %u\n",
              stats[0].tile_bytes, stats[1].tile_bytes,
              saving(stats[0], stats[1]), stats[0].frame_bytes,
              stats[1].frame_bytes);
  if (options.tx_types) {
    std::printf(
        "  tile bytes: scalar %u -> TX-type+RDOQ %u (%.2f%%), frame %u -> %u\n",
        stats[0].tile_bytes, stats[2].tile_bytes, saving(stats[0], stats[2]),
        stats[0].frame_bytes, stats[2].frame_bytes);
    long long non_dct = 0;
    for (int t = 0; t < 16; ++t) {
      if (t != static_cast<int>(TxType::kDctDct)) non_dct += stats[2].types[t];
    }
    std::printf("  TX-type wins: non-DCT %lld / %lld", non_dct,
                static_cast<long long>(plan.blocks.size()));
    for (int t = 0; t < 16; ++t) {
      if (stats[2].types[t] != 0) std::printf("  %d:%lld", t, stats[2].types[t]);
    }
    std::printf("\n");
  }
  return true;
}

void usage() {
  std::printf(
      "usage: transform_stream_quality [--input FILE] [--tx-types]\n"
      "         [--sse-scale X] [--qp N] [--depth N] [--min-block N]\n"
      "         [--enable-idtx] [--write-ivf PREFIX]\n");
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--input") {
      options.input = next("--input");
    } else if (arg == "--tx-types") {
      options.tx_types = true;
    } else if (arg == "--sse-scale") {
      options.sse_scale = std::stod(next("--sse-scale"));
    } else if (arg == "--qp") {
      options.qp = std::stoi(next("--qp"));
    } else if (arg == "--depth") {
      options.depth = std::stoi(next("--depth"));
    } else if (arg == "--min-block") {
      options.min_block = std::stoi(next("--min-block"));
    } else if (arg == "--enable-idtx") {
      options.enable_idtx = true;
    } else if (arg == "--write-ivf") {
      options.write_ivf = next("--write-ivf");
    } else if (arg == "-h" || arg == "--help") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }
  if (options.qp < 0 || options.qp > 90) {
    std::fprintf(stderr, "the §9.3 model covers base_q_idx <= 90\n");
    return 2;
  }

  std::vector<std::string> inputs;
  if (options.input.empty()) {
    inputs = {"tests/data/m0_64x64_1f.y4m",
              "tests/data/m0_640x360_1f.y4m"};
  } else {
    inputs = {options.input};
  }
  bool ok = true;
  for (const std::string& input : inputs) {
    if (!std::filesystem::exists(input)) {
      std::fprintf(stderr, "missing input %s\n", input.c_str());
      ok = false;
      continue;
    }
    if (!run_file(input, options)) ok = false;
  }
  return ok ? 0 : 1;
}
