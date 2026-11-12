// M2 inter-residual decoder gate.
//
// Encodes one intra keyframe with the stock M2 encoder, then encodes every P
// frame with this component's decision path and the *patched* M2 syntax writer
// (see patch_video_syntax.py), writes an IVF, decodes it with the pinned dav2d and
// compares the decoded planes against the encoder reconstruction sample by
// sample.
//
// The gate is deliberately GLOBALMV (zero MV) and inter-only: it isolates the
// residual path under test (TX sizes, inter TX types, coefficient contexts)
// from the motion-search, merge and intra-fallback paths owned by the M2 core.
//
// Usage:
//   video_residual_gate [--input clip.y4m] [--frames N] [--depth N]
//                    [--tx-types dct|all] [--out out.ivf] [--compare-core]
//                    [--force-skip] [--no-decode] [--trace]
#include "codec/inter/mc/mc.h"
#include "pipeline/video/video_frame.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_syntax.h"
#include "codec/transform/inter_residual/inter_residual.h"
#include "codec/partition/tree/partition.h"
#include "core/io/ivf.h"
#include "core/io/y4m.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// GATE_USE_PATCHED selects the build against the patched copy of the M2 syntax
// writer (see patch_video_syntax.py).  The unpatched build is byte-compared with it
// by gate/build_gate.sh for the DCT-only case.
#ifndef GATE_USE_PATCHED
#define GATE_USE_PATCHED 0
#endif

#if GATE_USE_PATCHED
namespace av2enc::video::syntax {
void video_residual_gate_set_inter_tx_types(const std::uint8_t* types, int count);
}  // namespace av2enc::video::syntax
#endif

namespace {

namespace msyntax = av2enc::video::syntax;
namespace residual = av2enc::video::residual;
namespace partition = av2enc::partition;
namespace inter = av2enc::inter;
using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionNode;
using partition::PartitionType;

struct OwnedFrame {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> y;
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
};

std::uint8_t square_sample(int x, int y, int frame, int plane) {
  const int sx = 8 + 4 * frame;
  const int sy = 8 + 2 * frame;
  const int in_square = x >= sx && x < sx + 32 && y >= sy && y < sy + 32;
  if (plane == 0) {
    if (in_square) {
      const int lx = x - sx;
      const int ly = y - sy;
      return static_cast<std::uint8_t>(40 + ((lx * 37 + ly * 17) & 0x7F));
    }
    return static_cast<std::uint8_t>(96 + ((x + y) >> 3));
  }
  if (plane == 1) {
    return static_cast<std::uint8_t>(in_square ? 100 : 128);
  }
  return static_cast<std::uint8_t>(in_square ? 150 : 128);
}

std::vector<OwnedFrame> make_synthetic_clip(int frames, int width, int height) {
  std::vector<OwnedFrame> out;
  const int cw = width / 2;
  const int ch = height / 2;
  for (int f = 0; f < frames; ++f) {
    OwnedFrame frame;
    frame.width = static_cast<std::uint32_t>(width);
    frame.height = static_cast<std::uint32_t>(height);
    frame.y.resize(static_cast<std::size_t>(width * height));
    frame.u.resize(static_cast<std::size_t>(cw * ch));
    frame.v.resize(static_cast<std::size_t>(cw * ch));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        frame.y[static_cast<std::size_t>(y * width + x)] =
            square_sample(x, y, f, 0);
      }
    }
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        frame.u[static_cast<std::size_t>(y * cw + x)] =
            square_sample(x * 2, y * 2, f, 1);
        frame.v[static_cast<std::size_t>(y * cw + x)] =
            square_sample(x * 2, y * 2, f, 2);
      }
    }
    out.push_back(std::move(frame));
  }
  return out;
}

bool load_y4m(const std::string& path, int max_frames,
              std::vector<OwnedFrame>& out, std::string& error) {
  av2enc::Y4mReader reader(path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  while (static_cast<int>(out.size()) < max_frames) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    OwnedFrame owned;
    owned.width = frame.width();
    owned.height = frame.height();
    owned.y.assign(frame.y().begin(), frame.y().end());
    owned.u.assign(frame.u().begin(), frame.u().end());
    owned.v.assign(frame.v().begin(), frame.v().end());
    out.push_back(std::move(owned));
  }
  if (out.empty()) {
    error = "Y4M input contained no frames";
    return false;
  }
  return true;
}

std::string find_dav2d() {
  if (const char* env = std::getenv("DAV2D")) {
    if (std::filesystem::exists(env)) return env;
  }
  for (const char* candidate :
       {"tools/bin/dav2d", "tools/bin/dav2d-macos", "dav2d", "dav2d-macos"}) {
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}

struct Plane {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> samples;
  std::uint8_t at(int x, int y) const {
    return samples[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)];
  }
  void set(int x, int y, std::uint8_t value) {
    samples[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x)] = value;
  }
};

struct LeafInfo {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int tx_size = 0;
  int tx_type = 0;
  int eob = 0;
  int skip = 0;
};

struct Segment {
  std::vector<LeafInfo> leaves;
  std::vector<msyntax::BlockSyntax> blocks;
  std::vector<std::int32_t> levels;
  std::vector<std::uint8_t> tx_types;  // coding order, non-skip inter leaves
  bool all_dct = true;
  std::uint32_t inter_leaves = 0;
  std::uint32_t residual_leaves = 0;
  std::uint32_t non_dct_leaves = 0;
};

class PFrameEncoder {
 public:
  PFrameEncoder(const OwnedFrame& source, const av2enc::video::VideoReconFrame& reference,
                const residual::DecisionOptions& options, bool trace)
      : source_(source), options_(options), trace_(trace) {
    width_ = static_cast<int>(source.width);
    height_ = static_cast<int>(source.height);
    chroma_width_ = width_ / 2;
    chroma_height_ = height_ / 2;
    geometry_.width = width_;
    geometry_.height = height_;
    geometry_.sb_size = 64;
    geometry_.has_chroma = true;
    geometry_.chroma_subsampling_x = true;
    geometry_.chroma_subsampling_y = true;
    mi_cols_ = partition::mi_cols(geometry_);
    mi_rows_ = partition::mi_rows(geometry_);

    recon_[0].width = width_;
    recon_[0].height = height_;
    recon_[0].samples.assign(static_cast<std::size_t>(width_ * height_), 0);
    for (int plane = 1; plane < 3; ++plane) {
      recon_[plane].width = chroma_width_;
      recon_[plane].height = chroma_height_;
      recon_[plane].samples.assign(
          static_cast<std::size_t>(chroma_width_ * chroma_height_), 0);
    }

    ref_.plane[0] = inter::mc::PlaneView{reference.y.data(), width_};
    ref_.plane[1] = inter::mc::PlaneView{reference.u.data(), chroma_width_};
    ref_.plane[2] = inter::mc::PlaneView{reference.v.data(), chroma_width_};
    ref_.mi_cols = mi_cols_;
    ref_.mi_rows = mi_rows_;
  }

  bool encode(int max_depth, bool force_single_block, Segment& segment,
              std::string& error) {
    partition::DecisionOptions poptions;
    poptions.min_block_size_px = force_single_block ? 64 : 8;
    poptions.max_depth = force_single_block ? 0 : max_depth;
    poptions.allow_rect = true;
    poptions.allow_split = false;
    const auto cost_fn = [this](const partition::PartitionNodeInfo& info) {
      partition::NodeCosts costs;
      costs.leaf = static_cast<double>(block_variance_cost_at(info));
      costs.none_signal = 0.0;
      costs.split_signal = 0.0;
      costs.horz_signal = 0.0;
      costs.vert_signal = 0.0;
      return costs;
    };
    const partition::FrameDecision tree =
        partition::decide_frame(geometry_, cost_fn, poptions);
    if (tree.status != partition::DecisionStatus::kOk) {
      error = "partition decision failed: " + tree.error;
      return false;
    }
    if (!partition::validate_frame(geometry_, tree, error)) {
      error = "partition tree invalid: " + error;
      return false;
    }
    segment = Segment{};
    for (const PartitionNode& sb : tree.superblocks) {
      if (!collect_node(sb, segment, error)) return false;
    }
    return true;
  }

  const Plane& recon(int plane) const { return recon_[plane]; }

 private:
  std::int64_t block_variance_cost_at(
      const partition::PartitionNodeInfo& info) const {
    const partition::BlockRect rect =
        partition::block_rect(geometry_, info.mi_row, info.mi_col, info.size);
    const int w = rect.visible_width;
    const int h = rect.visible_height;
    if (w <= 0 || h <= 0) return 0;
    std::int64_t sum = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        sum += source_sample(0, rect.x + x, rect.y + y);
      }
    }
    const double mean =
        static_cast<double>(sum) / static_cast<double>(w * h);
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

  int source_sample(int plane, int x, int y) const {
    const int w = plane == 0 ? width_ : chroma_width_;
    const int h = plane == 0 ? height_ : chroma_height_;
    const int cx = clip3(0, w - 1, x);
    const int cy = clip3(0, h - 1, y);
    const std::vector<std::uint8_t>& data =
        plane == 0 ? source_.y : (plane == 1 ? source_.u : source_.v);
    return data[static_cast<std::size_t>(cy) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(cx)];
  }

  bool collect_node(const PartitionNode& node, Segment& segment,
                    std::string& error) {
    const std::size_t index = segment.blocks.size();
    segment.blocks.emplace_back();
    msyntax::BlockSyntax& entry = segment.blocks.back();
    entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
    entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
    entry.bsize = static_cast<std::uint8_t>(node.size);
    entry.partition = static_cast<std::uint8_t>(node.partition);
    std::uint8_t mask = 0;
    for (int p = 0; p < partition::kPartitionTypeCount; ++p) {
      if (node.allowed.allowed[p]) mask |= static_cast<std::uint8_t>(1u << p);
    }
    entry.allowed_mask = mask;
    if (node.forced) entry.flags |= msyntax::kFlagForced;
    if (node.depth == 0) entry.flags |= msyntax::kFlagSbRoot;
    if (node.partition == PartitionType::kNone) {
      if (!encode_leaf(node, index, segment, error)) return false;
    } else {
      for (const PartitionNode& child : node.children) {
        if (!collect_node(child, segment, error)) return false;
      }
    }
    const std::size_t subtree = segment.blocks.size() - index - 1;
    if (subtree > 0xFFFFU) {
      error = "partition subtree too large";
      return false;
    }
    segment.blocks[index].subtree = static_cast<std::uint16_t>(subtree);
    return true;
  }

  bool encode_leaf(const PartitionNode& node, std::size_t index,
                   Segment& segment, std::string& error) {
    const int r = node.mi_row;
    const int c = node.mi_col;
    const int w = partition::block_width(node.size);
    const int h = partition::block_height(node.size);
    const int cw = w / 2;
    const int ch = h / 2;
    const int x0 = c * 4;
    const int y0 = r * 4;
    const int cx0 = x0 / 2;
    const int cy0 = y0 / 2;
    msyntax::BlockSyntax& entry = segment.blocks[index];

    // GLOBALMV: zero MV inter prediction for every leaf.
    std::vector<std::uint8_t> luma(static_cast<std::size_t>(w * h));
    std::vector<std::uint8_t> u(static_cast<std::size_t>(cw * ch));
    std::vector<std::uint8_t> v(static_cast<std::size_t>(cw * ch));
    inter::mc::MutablePlane dst[3];
    dst[0] = inter::mc::MutablePlane{luma.data(), w};
    dst[1] = inter::mc::MutablePlane{u.data(), cw};
    dst[2] = inter::mc::MutablePlane{v.data(), cw};
    const inter::mc::BlockRect rect{x0, y0, w, h};
    if (inter::mc::predict_block_420(ref_, rect, inter::mc::Mv{0, 0},
                                     inter::mc::InterpFilter::kEighttap,
                                     dst) != inter::mc::McStatus::kOk) {
      error = "inter prediction failed";
      return false;
    }

    // Coded-size source block with edge clamping (M1 source_sample rule).
    struct BlockBuffers {
      std::vector<std::uint8_t> source[3];
      std::vector<std::uint8_t> prediction[3];
    } buffers;
    const int pw[3] = {w, cw, cw};
    const int ph[3] = {h, ch, ch};
    const int px[3] = {x0, cx0, cx0};
    const int py[3] = {y0, cy0, cy0};
    for (int plane = 0; plane < 3; ++plane) {
      buffers.source[plane].resize(
          static_cast<std::size_t>(pw[plane] * ph[plane]));
      for (int y = 0; y < ph[plane]; ++y) {
        for (int x = 0; x < pw[plane]; ++x) {
          buffers.source[plane][static_cast<std::size_t>(y * pw[plane] + x)] =
              static_cast<std::uint8_t>(
                  source_sample(plane, px[plane] + x, py[plane] + y));
        }
      }
      buffers.prediction[plane] =
          plane == 0 ? luma : (plane == 1 ? u : v);
    }

    residual::BlockGeometry bg;
    bg.mi_row = r;
    bg.mi_col = c;
    bg.luma_w = w;
    bg.luma_h = h;
    bg.chroma_w = cw;
    bg.chroma_h = ch;

    residual::BlockInput input;
    input.geometry = bg;
    for (int plane = 0; plane < 3; ++plane) {
      input.source[plane].data = buffers.source[plane];
      input.source[plane].stride = pw[plane];
      input.prediction[plane].data = buffers.prediction[plane];
      input.prediction[plane].stride = pw[plane];
    }
    residual::BlockDecision decision;
    if (!residual::decide_inter_block(input, options_, decision, error)) {
      return false;
    }

    entry.is_inter = 1;
    entry.inter_mode = msyntax::kInterGlobalMv;
    entry.drl_index = 0;
    entry.mv_y = 0;
    entry.mv_x = 0;
    entry.mvd_y = 0;
    entry.mvd_x = 0;
    ++segment.inter_leaves;

    if (decision.skip) {
      entry.skip_flag = 1;
      write_plane(0, x0, y0, w, h, buffers.prediction[0]);
      write_plane(1, cx0, cy0, cw, ch, buffers.prediction[1]);
      write_plane(2, cx0, cy0, cw, ch, buffers.prediction[2]);
    } else {
      entry.skip_flag = 0;
      residual::InterResidualBlock syntax;
      if (!residual::make_syntax_block(bg, decision, &syntax, &segment.levels,
                                       error)) {
        return false;
      }
      for (int plane = 0; plane < 3; ++plane) {
        entry.tx_size[plane] = syntax.plane[plane].tx_size;
        entry.eob[plane] = syntax.plane[plane].eob;
      }
      entry.levels_offset = syntax.plane[0].levels_offset;
      if (syntax.plane[0].eob > 0) {
        segment.tx_types.push_back(syntax.plane[0].tx_type);
        if (syntax.plane[0].tx_type != 0) {
          segment.all_dct = false;
          ++segment.non_dct_leaves;
        }
      }
      ++segment.residual_leaves;
      for (int plane = 0; plane < 3; ++plane) {
        write_plane(plane, plane == 0 ? x0 : cx0, plane == 0 ? y0 : cy0,
                    pw[plane], ph[plane], decision.plane[plane].recon);
      }
    }

    LeafInfo leaf;
    leaf.x = x0;
    leaf.y = y0;
    leaf.w = w;
    leaf.h = h;
    leaf.tx_size = decision.plane[0].tx_size;
    leaf.tx_type = decision.plane[0].tx_type;
    leaf.eob = static_cast<int>(entry.eob[0]);
    leaf.skip = entry.skip_flag;
    segment.leaves.push_back(leaf);

    if (trace_) {
      std::cout << "  leaf " << w << "x" << h << " @" << r << "," << c
                << " skip=" << static_cast<int>(entry.skip_flag);
      if (!decision.skip) {
        std::cout << " eob=" << entry.eob[0] << "/" << entry.eob[1] << "/"
                  << entry.eob[2] << " txty=" << decision.plane[0].tx_type
                  << " cost=" << decision.cost << " skip_cost="
                  << decision.skip_cost;
      }
      std::cout << "\n";
    }
    return true;
  }

  void write_plane(int plane, int x0, int y0, int w, int h,
                   const std::vector<std::uint8_t>& samples) {
    for (int y = 0; y < h; ++y) {
      const int py = y0 + y;
      if (py < 0 || py >= recon_[plane].height) continue;
      for (int x = 0; x < w; ++x) {
        const int px = x0 + x;
        if (px < 0 || px >= recon_[plane].width) continue;
        recon_[plane].set(px, py,
                          samples[static_cast<std::size_t>(y * w + x)]);
      }
    }
  }

  const OwnedFrame& source_;
  residual::DecisionOptions options_;
  bool trace_ = false;
  int width_ = 0;
  int height_ = 0;
  int chroma_width_ = 0;
  int chroma_height_ = 0;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  FrameGeometry geometry_;
  Plane recon_[3];
  inter::mc::RefFrame ref_;
};

struct Stats {
  std::uint64_t sse = 0;
  std::uint64_t samples = 0;
  std::uint32_t bytes = 0;
  std::uint32_t tile_bytes = 0;
  std::uint32_t inter_leaves = 0;
  std::uint32_t residual_leaves = 0;
  std::uint32_t non_dct_leaves = 0;
  double psnr = 0.0;
};

double psnr_from_sse(std::uint64_t sse, std::uint64_t samples) {
  if (sse == 0 || samples == 0) return 99.0;
  const double mse =
      static_cast<double>(sse) / static_cast<double>(samples);
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

msyntax::FramePlan make_plan(int width, int height) {
  msyntax::FramePlan plan;
  plan.width = width;
  plan.height = height;
  plan.sb_size = 64;
  plan.inter_frame = true;
  return plan;
}

av2enc::video::VideoConfig make_config(int width, int height) {
  av2enc::video::VideoConfig config;
  config.max_frame_width = static_cast<std::uint16_t>(width);
  config.max_frame_height = static_cast<std::uint16_t>(height);
  config.base_q_idx = 90;
  config.order_hint_bits = 4;
  config.num_ref_frames = 2;
  config.num_total_refs = 1;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = true;
  config.enable_cfl_intra = true;
  return config;
}

bool assemble(const Segment& segment, int width, int height,
              std::vector<std::uint8_t>& tile, std::size_t* tile_bytes,
              std::string& error) {
  const msyntax::FramePlan plan = make_plan(width, height);
  const av2enc::video::VideoConfig config = make_config(width, height);
#if GATE_USE_PATCHED
  msyntax::video_residual_gate_set_inter_tx_types(
      segment.tx_types.data(), static_cast<int>(segment.tx_types.size()));
#endif
  return msyntax::assemble_video_frame(plan, segment.blocks, segment.levels, config,
                                    tile, error, tile_bytes);
}

bool compare_y4m(const std::string& decoded_path,
                 const std::vector<av2enc::video::VideoReconFrame>& recon,
                 const std::vector<std::vector<LeafInfo>>& diagnostics,
                 std::string& error) {
  av2enc::Y4mReader reader(decoded_path);
  if (!reader.open(error)) return false;
  av2enc::Y4mFrame frame;
  std::size_t index = 0;
  while (true) {
    const av2enc::Y4mReadResult status = reader.read_frame(frame, error);
    if (status == av2enc::Y4mReadResult::kEof) break;
    if (status == av2enc::Y4mReadResult::kError) return false;
    if (index >= recon.size()) {
      error = "decoder produced more frames than the encoder";
      return false;
    }
    const av2enc::video::VideoReconFrame& expected = recon[index];
    const auto compare = [&](std::span<const std::uint8_t> decoded,
                             const std::vector<std::uint8_t>& reference,
                             int plane) {
      if (decoded.size() != reference.size()) {
        error = "plane size mismatch";
        return false;
      }
      for (std::size_t i = 0; i < decoded.size(); ++i) {
        if (decoded[i] != reference[i]) {
          error = "recon mismatch on frame " + std::to_string(index) +
                  " plane " + std::to_string(plane) + " at " +
                  std::to_string(i) + ": decoded " +
                  std::to_string(decoded[i]) + " expected " +
                  std::to_string(reference[i]);
          if (index < diagnostics.size()) {
            const int width = plane == 0
                                  ? static_cast<int>(expected.width)
                                  : static_cast<int>((expected.width + 1) / 2);
            const int px = static_cast<int>(i % static_cast<std::size_t>(width));
            const int py = static_cast<int>(i / static_cast<std::size_t>(width));
            for (const LeafInfo& leaf : diagnostics[index]) {
              const int lx = plane == 0 ? leaf.x : leaf.x / 2;
              const int ly = plane == 0 ? leaf.y : leaf.y / 2;
              const int lw = plane == 0 ? leaf.w : leaf.w / 2;
              const int lh = plane == 0 ? leaf.h : leaf.h / 2;
              if (px >= lx && px < lx + lw && py >= ly && py < ly + lh) {
                error += " [leaf " + std::to_string(leaf.w) + "x" +
                         std::to_string(leaf.h) + " @" +
                         std::to_string(leaf.x) + "," +
                         std::to_string(leaf.y) + " tx=" +
                         std::to_string(leaf.tx_size) + " type=" +
                         std::to_string(leaf.tx_type) + " eob=" +
                         std::to_string(leaf.eob) +
                         " skip=" + std::to_string(leaf.skip) + "]";
                break;
              }
            }
          }
          return false;
        }
      }
      return true;
    };
    if (!compare(frame.y(), expected.y, 0)) return false;
    if (!compare(frame.u(), expected.u, 1)) return false;
    if (!compare(frame.v(), expected.v, 2)) return false;
    ++index;
  }
  if (index != recon.size()) {
    error = "decoder produced " + std::to_string(index) + " of " +
            std::to_string(recon.size()) + " frames";
    return false;
  }
  return true;
}

av2enc::video::VideoReconFrame to_recon(const Plane& y, const Plane& u,
                                  const Plane& v, int width, int height) {
  av2enc::video::VideoReconFrame out;
  out.width = static_cast<std::uint32_t>(width);
  out.height = static_cast<std::uint32_t>(height);
  out.y = y.samples;
  out.u = u.samples;
  out.v = v.samples;
  return out;
}

bool run(const std::vector<OwnedFrame>& frames, int max_depth,
         bool force_single_block, bool all_tx_types, bool force_skip,
         bool decode, const std::string& out_path, bool trace) {
  if (frames.size() < 2) {
    std::cerr << "need at least two frames\n";
    return false;
  }
  const int width = static_cast<int>(frames[0].width);
  const int height = static_cast<int>(frames[0].height);
  std::string error;

  // Keyframe from the stock M2 encoder (one frame).
  av2enc::video::VideoOptions key_options;
  key_options.force_single_block = force_single_block;
  key_options.min_block_size = force_single_block ? 64 : 8;
  key_options.max_partition_depth = force_single_block ? 0 : max_depth;
  std::vector<av2enc::video::VideoInputFrame> key_input(1);
  key_input[0].width = frames[0].width;
  key_input[0].height = frames[0].height;
  key_input[0].y = frames[0].y;
  key_input[0].u = frames[0].u;
  key_input[0].v = frames[0].v;
  av2enc::video::VideoSequenceResult key_sequence;
  if (!av2enc::video::encode_video_sequence(key_input, key_options, key_sequence,
                                      error)) {
    std::cerr << "keyframe encode failed: " << error << "\n";
    return false;
  }
  if (key_sequence.units.size() != 1) {
    std::cerr << "keyframe encoder produced " << key_sequence.units.size()
              << " units\n";
    return false;
  }

  std::vector<std::vector<std::uint8_t>> units;
  units.push_back(key_sequence.units[0]);
  std::vector<av2enc::video::VideoReconFrame> recon;
  recon.push_back(key_sequence.recon[0]);
  std::vector<std::vector<LeafInfo>> diagnostics;
  diagnostics.emplace_back();

  residual::DecisionOptions options;
  options.allow_residual = !force_skip;
  options.try_inter_tx_types = all_tx_types;

  const av2enc::video::VideoConfig config = make_config(width, height);

  Stats total;
  std::cout << "== inter-residual gate: " << frames.size() << " frame(s) "
            << width << "x" << height << " depth=" << max_depth
            << " tx-types=" << (all_tx_types ? "all" : "dct")
            << (force_skip ? " force-skip" : "") << " ==\n";
  std::cout << "  frame 0: keyframe bytes=" << units[0].size() << "\n";

  for (std::size_t i = 1; i < frames.size(); ++i) {
    PFrameEncoder encoder(frames[i], recon.back(), options, trace);
    Segment segment;
    if (!encoder.encode(max_depth, force_single_block, segment, error)) {
      std::cerr << "P-frame " << i << " decision failed: " << error << "\n";
      return false;
    }
    std::vector<std::uint8_t> tile;
    std::size_t tile_bytes = 0;
    if (!assemble(segment, width, height, tile, &tile_bytes, error)) {
      std::cerr << "P-frame " << i << " assemble failed: " << error << "\n";
      return false;
    }
    av2enc::video::VideoFrameHeader header;
    header.keyframe = false;
    header.order_hint = static_cast<std::uint32_t>(i);
    header.ref_slot = 0;
    header.refresh_frame_flags = 1;
    const std::vector<std::uint8_t> td =
        av2enc::video::make_temporal_delimiter_obu();
    const std::vector<std::uint8_t> obu =
        av2enc::video::make_frame_obu(config, header, tile);
    std::vector<std::uint8_t> unit;
    unit.reserve(td.size() + obu.size());
    unit.insert(unit.end(), td.begin(), td.end());
    unit.insert(unit.end(), obu.begin(), obu.end());
    units.push_back(std::move(unit));

    av2enc::video::VideoReconFrame frame_recon =
        to_recon(encoder.recon(0), encoder.recon(1), encoder.recon(2), width,
                 height);
    recon.push_back(std::move(frame_recon));
    diagnostics.push_back(segment.leaves);

    // Distortion against the source.
    const av2enc::video::VideoReconFrame& mine = recon.back();
    std::uint64_t sse = 0;
    const auto accumulate = [&sse](std::span<const std::uint8_t> a,
                                   const std::vector<std::uint8_t>& b) {
      for (std::size_t k = 0; k < a.size(); ++k) {
        const int d = static_cast<int>(a[k]) - static_cast<int>(b[k]);
        sse += static_cast<std::uint64_t>(d * d);
      }
    };
    accumulate(frames[i].y, mine.y);
    accumulate(frames[i].u, mine.u);
    accumulate(frames[i].v, mine.v);
    const std::uint64_t samples =
        static_cast<std::uint64_t>(width * height) +
        2 * static_cast<std::uint64_t>((width / 2) * (height / 2));
    total.sse += sse;
    total.samples += samples;
    total.bytes += static_cast<std::uint32_t>(units.back().size());
    total.tile_bytes += static_cast<std::uint32_t>(tile_bytes);
    total.inter_leaves += segment.inter_leaves;
    total.residual_leaves += segment.residual_leaves;
    total.non_dct_leaves += segment.non_dct_leaves;
    std::cout << "  frame " << i << ": bytes=" << units.back().size()
              << " tile=" << tile_bytes
              << " inter=" << segment.inter_leaves
              << " residual=" << segment.residual_leaves
              << " non_dct=" << segment.non_dct_leaves
              << " psnr=" << psnr_from_sse(sse, samples) << "\n";
  }
  total.psnr = psnr_from_sse(total.sse, total.samples);
  std::cout << "  total: bytes=" << total.bytes
            << " (keyframe " << units[0].size() << ", P total "
            << (total.bytes) << ")"
            << " tile_bytes=" << total.tile_bytes
            << " residual_leaves=" << total.residual_leaves << "/"
            << total.inter_leaves
            << " non_dct_leaves=" << total.non_dct_leaves
            << " psnr=" << total.psnr << "\n";

  // IVF write.
  av2enc::IvfWriter writer;
  av2enc::IvfHeader ivf_header;
  ivf_header.width = static_cast<std::uint16_t>(width);
  ivf_header.height = static_cast<std::uint16_t>(height);
  ivf_header.fps_num = 30;
  ivf_header.fps_den = 1;
  ivf_header.frame_count = static_cast<std::uint32_t>(units.size());
  if (!writer.open(out_path, ivf_header, error)) {
    std::cerr << "IVF open failed: " << error << "\n";
    return false;
  }
  for (std::size_t i = 0; i < units.size(); ++i) {
    if (!writer.write_frame(units[i], i, error)) {
      std::cerr << "IVF write failed: " << error << "\n";
      return false;
    }
  }
  if (!writer.close(error)) {
    std::cerr << "IVF close failed: " << error << "\n";
    return false;
  }

  if (!decode) {
    std::cout << "  (decoder comparison skipped)\n";
    return true;
  }
  const std::string dav2d = find_dav2d();
  if (dav2d.empty()) {
    std::cerr << "DECODE SKIP: no dav2d binary found (set DAV2D)\n";
    return false;
  }
  const std::string decoded_path = out_path + ".decoded.y4m";
  const std::string command = dav2d + " -q -i " + out_path + " -o " +
                              decoded_path + " --threads 0";
  std::cout << "  decode: " << command << "\n";
  if (std::system(command.c_str()) != 0) {
    std::cerr << "DECODE FAIL\n";
    return false;
  }
  if (!compare_y4m(decoded_path, recon, diagnostics, error)) {
    std::cerr << "RECON MISMATCH: " << error << "\n";
    return false;
  }
  std::cout << "  PASS: " << recon.size()
            << " frame(s) decoded, recon exact\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string input;
  std::string out_path = "video_residual_gate.ivf";
  int frames = 4;
  int depth = 0;
  bool all_tx_types = true;
  bool force_skip = false;
  bool decode = true;
  bool trace = false;
  bool force_single_block = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input = argv[++i];
    } else if (arg == "--frames" && i + 1 < argc) {
      frames = std::atoi(argv[++i]);
    } else if (arg == "--depth" && i + 1 < argc) {
      depth = std::atoi(argv[++i]);
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--tx-types" && i + 1 < argc) {
      const std::string value = argv[++i];
      all_tx_types = value == "all";
    } else if (arg == "--dct-only") {
      all_tx_types = false;
    } else if (arg == "--force-skip") {
      force_skip = true;
    } else if (arg == "--single-block") {
      force_single_block = true;
    } else if (arg == "--no-decode") {
      decode = false;
    } else if (arg == "--trace") {
      trace = true;
    } else if (arg == "--help") {
      std::cout
          << "usage: video_residual_gate [--input clip.y4m] [--frames N] "
             "[--depth N] [--tx-types dct|all] [--dct-only] [--single-block] "
             "[--out out.ivf] [--force-skip] [--no-decode] [--trace]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }
  if (frames < 2) frames = 2;

  std::vector<OwnedFrame> owned;
  if (!input.empty()) {
    std::string error;
    if (!load_y4m(input, frames, owned, error)) {
      std::cerr << "Y4M LOAD FAIL: " << error << "\n";
      return 2;
    }
  } else {
    owned = make_synthetic_clip(frames, 64, 64);
  }
  const bool ok = run(owned, depth, force_single_block, all_tx_types,
                      force_skip, decode, out_path, trace);
  std::cout << (ok ? "M2 RESIDUAL GATE PASS" : "M2 RESIDUAL GATE FAIL") << "\n";
  return ok ? 0 : 1;
}
