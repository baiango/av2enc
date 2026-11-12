// Integrated M1 still-picture encoder.
//
// This file is the integration point for the wave-1 components:
//   partition (src/encoder/partition)
//   full intra prediction (src/encoder/intra_modes)
//   transform variants (src/encoder/tx_variants)
//   CfL (src/encoder/cfl)
//   deblocking (src/encoder/filters/deblock)
//
// It keeps the M0 path in src/encoder/basic_encoder.cpp untouched and uses the
// normative §8.2 entropy coder from src/entropy/av2 unchanged.
#include "pipeline/still/still_encoder.h"

#include "codec/intra/cfl/cfl.h"
#include "codec/filters/cdef/cdef.h"
#include "codec/filters/deblock/deblock.h"
#include "codec/filters/loop_restoration/restoration.h"
#include "codec/intra/modes/intra_modes.h"
#include "pipeline/still/still_headers.h"
#include "pipeline/still/still_syntax.h"
#include "pipeline/still/still_tables.h"
#include "codec/entropy/qctx/qctx.h"
#include "codec/partition/tree/partition.h"
#include "codec/rate/rc/rc_tables.h"
#include "codec/rate/rdo/intra_decision.h"
#include "codec/rate/rdo/rdo.h"
#include "codec/rate/rdoq/rdoq.h"
#include "codec/rate/rdoq/rdoq_av2_model.h"
#include "codec/transform/tx_type/tx_type.h"
#include "codec/transform/variants/tx_variants.h"

#include "cuda/pipeline/still/cdef/still_cdef.h"

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

namespace av2enc::still {
namespace {

using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionType;

// YMode numeric values (§6.19.5.5).
constexpr int kDcPred = 0;
constexpr int kV = 1;
constexpr int kH = 2;
constexpr int kD45 = 3;
constexpr int kD135 = 4;
constexpr int kD113 = 5;
constexpr int kD157 = 6;
constexpr int kD203 = 7;
constexpr int kD67 = 8;
constexpr int kSmooth = 9;
constexpr int kSmoothV = 10;
constexpr int kSmoothH = 11;
constexpr int kPaeth = 12;

int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}
int clip1(int value) { return clip3(0, 255, value); }

int av2_y_mode(intra::Mode mode) {
  switch (mode) {
    case intra::Mode::kDc: return kDcPred;
    case intra::Mode::kV: return kV;
    case intra::Mode::kH: return kH;
    case intra::Mode::kD45: return kD45;
    case intra::Mode::kD135: return kD135;
    case intra::Mode::kD113: return kD113;
    case intra::Mode::kD157: return kD157;
    case intra::Mode::kD203: return kD203;
    case intra::Mode::kD67: return kD67;
    case intra::Mode::kSmooth: return kSmooth;
    case intra::Mode::kSmoothV: return kSmoothV;
    case intra::Mode::kSmoothH: return kSmoothH;
    case intra::Mode::kPaeth: return kPaeth;
  }
  return kDcPred;
}

bool y_mode_is_directional(int mode) { return mode >= kV && mode <= kD67; }
bool y_mode_is_smooth(int mode) {
  return mode == kSmooth || mode == kSmoothV || mode == kSmoothH;
}

struct Plane {
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
    return samples[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)];
  }
  std::uint8_t at(int x, int y) const {
    return samples[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)];
  }
};

struct TxPlan {
  int plane = 0;
  int tx_size = 0;
  int tx_type = 0;  // txv::TxType; luma may signal it, chroma is DCT_DCT
  int coded_w = 0;
  int coded_h = 0;
  int adj_w = 0;
  int adj_h = 0;
  int eob = 0;
  std::vector<std::int32_t> levels;
  std::vector<std::uint8_t> pred;
  std::vector<std::uint8_t> recon;
};

// Decision/reconstruction output: the preorder node stream plus the flat
// scan-order coefficient levels.  syntax::assemble_still_frame() turns this into
// the tile payload and the OBU-framed temporal unit.
struct SyntaxStream {
  std::vector<syntax::BlockSyntax> blocks;
  std::vector<std::int32_t> levels;
};

struct ModeChoice {
  intra::Mode mode = intra::Mode::kDc;
  int angle_delta = 0;
  int av2_mode = kDcPred;
  std::int64_t sse = 0;
};

struct UvChoice {
  bool is_cfl = false;
  intra::Mode mode = intra::Mode::kDc;
  int angle_delta = 0;
  int av2_mode = kDcPred;
  std::int64_t sse = 0;
};

// Per-4x4 decoded flags for count_top_right_avail / count_bottom_left_avail.
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
    return cells[static_cast<std::size_t>(y) * static_cast<std::size_t>(cols) +
                 static_cast<std::size_t>(x)] != 0;
  }
  void fill(int x, int y, int w, int h) {
    for (int j = 0; j < h; ++j) {
      for (int i = 0; i < w; ++i) {
        if (x + i >= 0 && y + j >= 0 && x + i < cols && y + j < rows) {
          cells[static_cast<std::size_t>(y + j) *
                    static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(x + i)] = 1;
        }
      }
    }
  }
};

class Encoder {
 public:
  Encoder(const StillPicture& picture, const StillOptions& options)
      : picture_(picture), options_(options) {
    width_ = static_cast<int>(picture.width);
    height_ = static_cast<int>(picture.height);
    chroma_width_ = (width_ + 1) / 2;
    chroma_height_ = (height_ + 1) / 2;
    // §7.14.2 qlookup(base_q_idx), 8-bit.  At the historical M1 operating point
    // (90) this is exactly txv::kQuantStep, so the pinned path is unchanged.
    quant_step_ = rc::kQlookupQ0To255[options.base_q_idx];
    // §8.3.1 coefficient-CDF qctx slice for the RDOQ rate model.  The syntax
    // writer selects the same slice from the same raw base_q_idx.
    qctx_slice_ = qctx::slice_for_base_q_idx(options.base_q_idx);

    geometry_.width = width_;
    geometry_.height = height_;
    geometry_.sb_size = 64;
    geometry_.has_chroma = true;
    geometry_.chroma_subsampling_x = true;
    geometry_.chroma_subsampling_y = true;

    mi_cols_ = partition::mi_cols(geometry_);
    mi_rows_ = partition::mi_rows(geometry_);
    const int coded_luma_w = mi_cols_ * 4;
    const int coded_luma_h = mi_rows_ * 4;
    // The decoder reconstructs a whole coding/transform block even when it
    // extends past the MiCols*4 x MiRows*4 coded plane. Allocate the usual
    // 64-sample superblock padding so edge blocks stay in bounds; prediction
    // still clamps to the coded plane (max_x/max_y below).
    luma_stride_ = (coded_luma_w + 63) & ~63;
    luma_height_ = (coded_luma_h + 63) & ~63;
    chroma_stride_ = luma_stride_ / 2;
    chroma_padded_height_ = luma_height_ / 2;

    // dav2d initializes CurrFrame to 0. CfL's L/lumaAvg read the reconstructed
    // luma below/right of the current block (clamped only at block edges), so
    // zero padding is part of the encoder/decoder contract; 128 shifts CfL
    // predictions by 1 LSB and then cascades.
    recon_[0].resize(luma_stride_, luma_height_, 0);
    recon_[1].resize(chroma_stride_, chroma_padded_height_, 0);
    recon_[2].resize(chroma_stride_, chroma_padded_height_, 0);

    intra_joint_.assign(
        static_cast<std::size_t>(mi_rows_) * static_cast<std::size_t>(mi_cols_),
        0);
    uv_smooth_.assign(
        static_cast<std::size_t>(mi_rows_ / 2) *
            static_cast<std::size_t>(mi_cols_ / 2),
        0);

    for (int plane = 0; plane < 3; ++plane) {
      decoded_[plane].reset(plane == 0 ? mi_cols_ : mi_cols_ / 2,
                            plane == 0 ? mi_rows_ : mi_rows_ / 2);
    }

    if (options_.enable_deblock) {
      deblock_cols_ = mi_cols_;
      deblock_rows_ = mi_rows_;
      deblock::DeblockMi neutral;
      neutral.qindex = options_.base_q_idx;
      for (int plane = 0; plane < 3; ++plane) {
        deblock_mi_[plane].assign(
            static_cast<std::size_t>(mi_cols_) *
                static_cast<std::size_t>(mi_rows_),
            neutral);
      }
    }
  }

  // Walks the decided partition tree, performs prediction/transform/
  // reconstruction, and emits the equivalent BlockSyntax stream.  The syntax
  // itself is written later by syntax::assemble_still_frame(), so this function
  // owns decisions and reconstruction only.
  bool encode(SyntaxStream* stream, std::string& error) {
    FrameGeometry geometry = geometry_;
    partition::DecisionOptions poptions;
    poptions.min_block_size_px = options_.min_block_size;
    poptions.max_depth = options_.max_partition_depth;
    poptions.allow_rect = true;
    poptions.allow_split = false;
    if (options_.force_single_block) {
      poptions.max_depth = 0;
      poptions.min_block_size_px = 64;
    }

    const double lambda = 0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
    const double signal_cost = lambda * 2.0;

    const auto cost_fn = [&](const partition::PartitionNodeInfo& info) {
      partition::NodeCosts costs;
      costs.leaf = static_cast<double>(
          block_variance_cost_at(info.mi_row, info.mi_col, info.size));
      if (std::getenv("AV2ENC_STILL_SQUARE_ONLY") != nullptr &&
          partition::block_width(info.size) !=
              partition::block_height(info.size)) {
        costs.leaf += 1.0e18;
      }
      if (std::getenv("AV2ENC_STILL_LEAF32") != nullptr &&
          (partition::block_width(info.size) > 32 ||
           partition::block_height(info.size) > 32)) {
        costs.leaf += 1.0e18;
      }
      costs.none_signal = 0.0;
      costs.split_signal = signal_cost;
      costs.horz_signal = signal_cost;
      costs.vert_signal = signal_cost;
      return costs;
    };

    const partition::FrameDecision frame =
        partition::decide_frame(geometry, cost_fn, poptions);
    if (frame.status != partition::DecisionStatus::kOk) {
      error = "partition decision failed: " + frame.error;
      return false;
    }
    if (!partition::validate_frame(geometry, frame, error)) {
      error = "partition tree invalid: " + error;
      return false;
    }
    leaves_ = frame.leaves;

    blocks_.clear();
    levels_.clear();
    for (int sbi = 0; sbi < static_cast<int>(frame.superblocks.size()); ++sbi) {
      const partition::PartitionNode& sb = frame.superblocks[sbi];
      // Reset SB-local decoded flags exactly like clear_block_decoded_flags().
      for (int plane = 0; plane < 3; ++plane) {
        decoded_[plane].reset(plane == 0 ? mi_cols_ : mi_cols_ / 2,
                              plane == 0 ? mi_rows_ : mi_rows_ / 2);
      }
      sb_mi_row_ = sb.mi_row;
      sb_mi_col_ = sb.mi_col;
      sb_cols_plane_[0] = std::min(16, mi_cols_ - sb.mi_col);
      sb_rows_plane_[0] = std::min(16, mi_rows_ - sb.mi_row);
      sb_cols_plane_[1] = std::min(8, mi_cols_ / 2 - sb.mi_col / 2);
      sb_rows_plane_[1] = std::min(8, mi_rows_ / 2 - sb.mi_row / 2);
      sb_cols_plane_[2] = sb_cols_plane_[1];
      sb_rows_plane_[2] = sb_rows_plane_[1];

      if (!collect_node(sb)) {
        error = error_;
        return false;
      }
    }

    stream->blocks = std::move(blocks_);
    stream->levels = std::move(levels_);
    return true;
  }

  const Plane& recon(int plane) const { return recon_[plane]; }
  Plane& recon_mut(int plane) { return recon_[plane]; }
  int leaves() const { return leaves_; }
  int mi_cols() const { return mi_cols_; }
  int mi_rows() const { return mi_rows_; }
  int deblock_cols() const { return deblock_cols_; }
  int deblock_rows() const { return deblock_rows_; }
  const deblock::DeblockMi* deblock_mi(int plane) const {
    return deblock_mi_[plane].data();
  }

 private:
  // ---------------------------------------------------------------------
  // Decision-stream collection
  // ---------------------------------------------------------------------
  int& intra_joint(int r, int c) {
    return intra_joint_[static_cast<std::size_t>(r) *
                            static_cast<std::size_t>(mi_cols_) +
                        static_cast<std::size_t>(c)];
  }
  std::uint8_t& uv_smooth(int r, int c) {
    return uv_smooth_[static_cast<std::size_t>(r) *
                          static_cast<std::size_t>(mi_cols_ / 2) +
                      static_cast<std::size_t>(c)];
  }

  // Append this node (and its whole subtree) to the preorder syntax stream.
  // No syntax is written here: the stream is handed to
  // syntax::assemble_still_frame() once the frame is fully decided.
  bool collect_node(const partition::PartitionNode& node) {
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

    if (trace_) {
      std::cerr << "m1 node r=" << node.mi_row << " c=" << node.mi_col
                << " size=" << partition::block_size_name(node.size)
                << " part=" << partition::partition_name(node.partition)
                << " forced=" << (node.forced ? 1 : 0) << " allowed="
                << node.allowed.allowed[0] << node.allowed.allowed[1]
                << node.allowed.allowed[2] << node.allowed.allowed[3] << "\n";
    }

    if (node.partition == PartitionType::kNone) {
      if (!encode_block(node, index)) return false;
    } else {
      for (const partition::PartitionNode& child : node.children) {
        if (!collect_node(child)) return false;
      }
    }
    const std::size_t subtree = blocks_.size() - index - 1;
    if (subtree > 0xFFFFU) {
      error_ = "partition subtree too large for the syntax stream";
      return false;
    }
    blocks_[index].subtree = static_cast<std::uint16_t>(subtree);
    return true;
  }

  // ---------------------------------------------------------------------
  // Source / reconstruction helpers
  // ---------------------------------------------------------------------
  std::uint8_t source_sample(int plane, int x, int y) const {
    const std::span<const std::uint8_t> data =
        plane == 0 ? picture_.y : (plane == 1 ? picture_.u : picture_.v);
    const int w = plane == 0 ? width_ : chroma_width_;
    const int h = plane == 0 ? height_ : chroma_height_;
    const int cx = clip3(0, w - 1, x);
    const int cy = clip3(0, h - 1, y);
    const int stride = plane == 0 ? width_ : chroma_width_;
    return data[static_cast<std::size_t>(cy) * static_cast<std::size_t>(stride) +
                static_cast<std::size_t>(cx)];
  }

  void fill_source(int plane, int x0, int y0, int w, int h,
                   std::vector<std::uint8_t>& out) const {
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        out[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
            static_cast<std::size_t>(x)] =
            source_sample(plane, x0 + x, y0 + y);
      }
    }
  }

  int count_top_right(int plane, int abs_x4, int abs_y4, int w4) const {
    const int sub = plane == 0 ? 0 : 1;
    const int sb_c0 = sb_mi_col_ >> sub;
    const int sb_r0 = sb_mi_row_ >> sub;
    const int rel_x = abs_x4 - sb_c0;
    const int rel_y = abs_y4 - sb_r0;
    const int sb_w = sb_cols_plane_[plane];
    // §6.19.2.3 clear_block_decoded_flags: the top border is 1 for
    // x < (MiColEnd - c), and MiColEnd is the tile end, not the SB end. The
    // BlockDecoded array extends to 2 * sbSize4, so availability along the
    // top row covers up to two superblocks (or the tile remainder).
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
    // clear_block_decoded_flags() presets the left border only within the
    // current SB (and clears the boundary row), so below-left availability can
    // never extend into the next SB row.
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

  // ---------------------------------------------------------------------
  // Prediction
  // ---------------------------------------------------------------------
  bool edge_filter_enabled() const {
    return options_.enable_intra_edge_filter &&
           std::getenv("AV2ENC_STILL_NO_EDGE") == nullptr;
  }

  bool prepare_edges(int plane, int x0, int y0, int w, int h,
                     intra::IntraEdges& edges) const {
    intra::NeighborRequest request;
    request.recon = intra::PlaneView{recon_[plane].samples.data(),
                                     recon_[plane].width};
    request.x = x0;
    request.y = y0;
    request.w = w;
    request.h = h;
    request.max_x = (plane == 0 ? mi_cols_ * 4 : mi_cols_ * 2) - 1;
    request.max_y = (plane == 0 ? mi_rows_ * 4 : mi_rows_ * 2) - 1;
    request.have_left = x0 > 0;
    request.have_above = y0 > 0;
    const int sub = plane == 0 ? 0 : 1;
    request.num4_above_right =
        count_top_right(plane, x0 >> 2, y0 >> 2, w >> 2);
    request.num4_below_left =
        count_bottom_left(plane, x0 >> 2, y0 >> 2, h >> 2);
    (void)sub;
    request.color_plane = plane;
    return intra::prepare_edges(request, 8, edges) == intra::IntraStatus::kOk;
  }

  bool neighborhood_smooth(int plane, int r, int c, int w4, int h4,
                           bool& above_smooth, bool& left_smooth) const {
    above_smooth = false;
    left_smooth = false;
    if (plane == 0) {
      // IntraJointMode stores the reordered non-directional index (DC=0,
      // SMOOTH=1, SMOOTH_V=2, SMOOTH_H=3, PAETH=4) or 5+directional index.
      const auto joint_is_smooth = [](int joint) {
        return joint >= 1 && joint <= 3;
      };
      if (r > 0 && c < mi_cols_) {
        above_smooth = joint_is_smooth(
            intra_joint_[static_cast<std::size_t>(r - 1) *
                             static_cast<std::size_t>(mi_cols_) +
                         static_cast<std::size_t>(c)]);
      }
      if (c > 0 && r < mi_rows_) {
        left_smooth = joint_is_smooth(
            intra_joint_[static_cast<std::size_t>(r) *
                             static_cast<std::size_t>(mi_cols_) +
                         static_cast<std::size_t>(c - 1)]);
      }
    } else {
      // Callers pass the chroma 4x4-grid index (luma MI / 2), which is already
      // the index used to write uv_smooth_. Do not halve it again.
      const int cr = r;
      const int cc = c;
      const int cmi_cols = mi_cols_ / 2;
      if (cr > 0 && cc < cmi_cols) {
        above_smooth =
            uv_smooth_[static_cast<std::size_t>(cr - 1) *
                           static_cast<std::size_t>(cmi_cols) +
                       static_cast<std::size_t>(cc)] != 0;
      }
      if (cc > 0 && cr < mi_rows_ / 2) {
        left_smooth =
            uv_smooth_[static_cast<std::size_t>(cr) *
                           static_cast<std::size_t>(cmi_cols) +
                       static_cast<std::size_t>(cc - 1)] != 0;
      }
    }
    (void)w4;
    (void)h4;
    return true;
  }

  bool predict_block(int plane, int x0, int y0, int w, int h,
                     intra::Mode mode, int angle_delta,
                     std::vector<std::uint8_t>& out) {
    intra::IntraEdges edges;
    if (!prepare_edges(plane, x0, y0, w, h, edges)) {
      error_ = "prepare_edges failed";
      return false;
    }
    bool above_smooth = false;
    bool left_smooth = false;
    neighborhood_smooth(plane, y0 >> 2, x0 >> 2, w >> 2, h >> 2,
                        above_smooth, left_smooth);
    intra::PredictParams params;
    params.mode = mode;
    params.angle_delta = angle_delta;
    params.enable_edge_filter = edge_filter_enabled();
    params.above_neighbor_is_smooth = above_smooth;
    params.left_neighbor_is_smooth = left_smooth;
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    if (intra::predict_intra(edges, params, out.data(),
                             static_cast<std::size_t>(w)) !=
        intra::IntraStatus::kOk) {
      error_ = "predict_intra failed";
      return false;
    }
    return true;
  }

  std::int64_t sse_against_source(int plane, int x0, int y0, int w, int h,
                                  const std::vector<std::uint8_t>& pred) const {
    std::int64_t sse = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int src = source_sample(plane, x0 + x, y0 + y);
        const int p =
            pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                 static_cast<std::size_t>(x)];
        const int d = src - p;
        sse += static_cast<std::int64_t>(d) * d;
      }
    }
    return sse;
  }

  bool choose_luma_mode(int x0, int y0, int w, int h, ModeChoice& best) {
    const char* forced_mode = std::getenv("AV2ENC_STILL_FORCE_Y");
    if (x0 > 0) {
      if (const char* second = std::getenv("AV2ENC_STILL_FORCE_Y2")) {
        forced_mode = second;
      }
    }
    if (forced_mode != nullptr) {
      int mode = 0;
      int delta = 0;
      if (std::sscanf(forced_mode, "%d:%d", &mode, &delta) == 2) {
        best.mode = static_cast<intra::Mode>(mode);
        best.angle_delta = delta;
        best.av2_mode = mode;
        best.sse = 0;
        return true;
      }
    }
    if (options_.luma_dc_only || options_.force_dc ||
        std::getenv("AV2ENC_STILL_LUMA_DC") != nullptr) {
      best.mode = intra::Mode::kDc;
      best.angle_delta = 0;
      best.av2_mode = kDcPred;
      best.sse = 0;
      return true;
    }
    intra::IntraEdges edges;
    if (!prepare_edges(0, x0, y0, w, h, edges)) {
      error_ = "luma prepare_edges failed";
      return false;
    }
    bool above_smooth = false;
    bool left_smooth = false;
    neighborhood_smooth(0, y0 >> 2, x0 >> 2, w >> 2, h >> 2, above_smooth,
                        left_smooth);
    std::array<intra::ModeCandidate, 61> candidates;
    const int count = intra::build_mode_candidates(candidates, true);
    if (count <= 0) {
      error_ = "build_mode_candidates failed";
      return false;
    }
    std::vector<std::uint8_t> pred;
    pred.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    const bool base_only = std::getenv("AV2ENC_STILL_BASE_ONLY") != nullptr;
    const bool no_directional = std::getenv("AV2ENC_STILL_NO_DIR") != nullptr;
    std::vector<av2enc::rdo::IntraCandidateCost> rdo_candidates;
    const av2enc::rdo::IntraRdoConfig rdo_config =
        av2enc::rdo::make_intra_rdo_config(
            static_cast<double>(quant_step_), options_.rdo_sse_scale, true);
    if (options_.rdo_intra) {
      rdo_candidates.reserve(static_cast<std::size_t>(count));
    }
    best.sse = std::numeric_limits<std::int64_t>::max();
    for (int i = 0; i < count; ++i) {
      intra::PredictParams params;
      params.mode = candidates[static_cast<std::size_t>(i)].mode;
      params.angle_delta = candidates[static_cast<std::size_t>(i)].angle_delta;
      if (base_only && params.angle_delta != 0) continue;
      if (no_directional && intra::is_directional(params.mode)) continue;
      params.enable_edge_filter = edge_filter_enabled();
      params.above_neighbor_is_smooth = above_smooth;
      params.left_neighbor_is_smooth = left_smooth;
      if (intra::predict_intra(edges, params, pred.data(),
                               static_cast<std::size_t>(w)) !=
          intra::IntraStatus::kOk) {
        error_ = "luma predict_intra failed";
        return false;
      }
      const std::int64_t sse = sse_against_source(0, x0, y0, w, h, pred);
      if (sse < best.sse) {
        best.sse = sse;
        best.mode = params.mode;
        best.angle_delta = params.angle_delta;
        best.av2_mode = av2_y_mode(params.mode);
      }
      if (options_.rdo_intra) {
        // Probe the real quantised levels for this candidate.  compute_tx()
        // reads only the source and the predictor, so it does not perturb the
        // reconstruction state.
        TxPlan probe;
        if (!compute_tx(0, x0, y0, w, h, pred, probe)) {
          error_ = "rate-aware intra: compute_tx probe failed";
          return false;
        }
        std::vector<std::uint16_t> probe_scan(
            static_cast<std::size_t>(probe.adj_w * probe.adj_h));
        syntax::build_scan(probe.adj_w, probe.adj_h, probe_scan);
        std::vector<std::int32_t> probe_levels;
        probe_levels.reserve(static_cast<std::size_t>(probe.eob));
        for (int scan_pos = 0; scan_pos < probe.eob; ++scan_pos) {
          probe_levels.push_back(
              probe.levels[probe_scan[static_cast<std::size_t>(scan_pos)]]);
        }
        av2enc::rdo::IntraCandidateCost candidate;
        candidate.mode = av2_y_mode(params.mode);
        candidate.angle_delta = params.angle_delta;
        candidate.sse = static_cast<double>(sse);
        candidate.syntax_rate_bits =
            av2enc::rdo::default_intra_syntax_rate_bits(candidate.mode,
                                                        candidate.angle_delta);
        candidate.coefficient_rate_bits =
            av2enc::rdo::intra_coefficient_rate_bits_from_levels(
                probe_levels, probe.eob, probe.adj_w * probe.adj_h);
        rdo_candidates.push_back(candidate);
      }
    }
    if (options_.rdo_intra && !rdo_candidates.empty()) {
      const av2enc::rdo::IntraModeChoice choice =
          av2enc::rdo::select_intra_mode(rdo_candidates, rdo_config);
      if (choice.found) {
        best.mode = static_cast<intra::Mode>(choice.mode);
        best.angle_delta = choice.angle_delta;
        best.av2_mode = choice.mode;
        best.sse = static_cast<std::int64_t>(choice.sse);
      }
    }
    return true;
  }

  std::int64_t block_variance_cost_at(int mi_row, int mi_col,
                                      BlockSize size) const {
    const partition::BlockRect rect =
        partition::block_rect(geometry_, mi_row, mi_col, size);
    const int w = rect.visible_width;
    const int h = rect.visible_height;
    if (w <= 0 || h <= 0) return 0.0;
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
        const double d = static_cast<double>(source_sample(
                            0, rect.x + x, rect.y + y)) -
                         mean;
        sse += d * d;
      }
    }
    return static_cast<std::int64_t>(sse);
  }

  // ---------------------------------------------------------------------
  // Transform / reconstruction
  // ---------------------------------------------------------------------

  // The transform component bakes the QP-90 forward quantiser entry (528)
  // into TxParams.  For any other base_q_idx, scale quant_factor by the ratio
  // of the normative qlookup step, so forward quantisation and the inverse
  // dequant below agree with the step the decoder derives from base_q_idx.
  void apply_qindex_step(txv::TxParams& params) const {
    if (quant_step_ == txv::kQuantStep) return;
    params.quant_factor *=
        static_cast<double>(txv::kQuantStep) / static_cast<double>(quant_step_);
  }

  // §7.14.4 dequant + inverse transform with the frame's qlookup step.
  void dequant_inverse_qindex(const std::int32_t* levels,
                              const txv::TxParams& params,
                              std::int32_t* dequant,
                              std::int16_t* residual) const {
    if (quant_step_ == txv::kQuantStep) {
      txv::inverse_transform(levels, params, dequant, residual);
      return;
    }
    const int n = params.w * params.h;
    for (int i = 0; i < n; ++i) {
      const std::int32_t qc = levels[i];
      const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
      const std::int64_t high = (aq * quant_step_) & 0xFFFFFF;
      const std::int32_t dq = txv::round2(static_cast<std::int32_t>(high),
                                          txv::kQuantTableBits) /
                              params.dq_denom;
      const std::int32_t value = qc < 0 ? -dq : dq;
      dequant[i] = std::max(-(1 << 15), std::min((1 << 15) - 1, value));
    }
    txv::inverse_block(dequant, params, residual);
  }

  bool compute_tx(int plane, int x0, int y0, int cw, int ch,
                  const std::vector<std::uint8_t>& pred, TxPlan& plan,
                  bool choose_tx_type = false, int y_mode = 0,
                  int y_angle_delta = 0) {
    std::optional<txv::TxParams> params_opt =
        txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
    if (!params_opt.has_value()) {
      error_ = "unsupported transform size";
      return false;
    }
    apply_qindex_step(*params_opt);
    txv::TxType tx_type = txv::TxType::kDctDct;
    const int coded_pels = cw * ch;
    std::vector<std::int16_t> residual_coded(
        static_cast<std::size_t>(coded_pels));
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const int src = source_sample(plane, x0 + x, y0 + y);
        const int p =
            pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                 static_cast<std::size_t>(x)];
        residual_coded[static_cast<std::size_t>(y) *
                           static_cast<std::size_t>(cw) +
                       static_cast<std::size_t>(x)] =
            static_cast<std::int16_t>(src - p);
      }
    }

    // Intra luma TX_TYPE selection.  The chooser runs the full candidate set
    // (SSE + lambda * rate); we then re-run the chosen type through the common
    // quantise/RDOQ/inverse path below so both features compose.
    if (choose_tx_type) {
      int mapped_mode = txtype::map_intra_mode(y_mode, y_angle_delta, cw, ch);
      if (mapped_mode < 0) mapped_mode = 0;
      txtype::TxTypeQuery query;
      query.coded_w = cw;
      query.coded_h = ch;
      query.mapped_mode = mapped_mode;
      query.reduced_tx_set = 0;
      query.dct_only = false;
      txtype::TxTypeCostConfig cost;
      cost.lambda = av2enc::rdo::lambda_sse_from_quant_step(
          1.0 / params_opt->quant_factor, options_.tx_type_sse_scale);
      txtype::TxTypeChoice choice;
      // §5.4.8: M1 signals enable_idtx_intra = 0, so identity-based intra
      // transforms (IDTX / V_DCT / H_DCT) are not allowed.  The long-side sets
      // list them, so filter those candidates before the decision.
      txtype::TxTypeSet candidates;
      if (!txtype::intra_tx_candidates(query, &candidates)) {
        error_ = "tx_type candidate enumeration failed";
        return false;
      }
      txtype::TxTypeCandidate usable[txtype::kMaxTxTypeCandidates];
      int usable_count = 0;
      for (int i = 0; i < candidates.count; ++i) {
        const txtype::TxTypeCandidate& candidate = candidates.candidates[i];
        // enable_idtx_intra = 0 forbids any 1D identity transform, which
        // covers IDTX, V/H_DCT, V/H_ADST and V/H_FLIPADST.
        txv::Tx1dType row_type = txv::Tx1dType::kDct;
        txv::Tx1dType col_type = txv::Tx1dType::kDct;
        txv::tx_1d_types(candidate.type, &row_type, &col_type);
        if (row_type == txv::Tx1dType::kIdt ||
            col_type == txv::Tx1dType::kIdt) {
          continue;
        }
        usable[usable_count++] = candidate;
      }
      if (usable_count == 0) {
        error_ = "no legal non-identity intra tx_type for this size/mode";
        return false;
      }
      if (!txtype::choose_tx_type_from_candidates(
              residual_coded.data(), query, usable, usable_count, cost,
              &choice, /*levels_out=*/nullptr, /*recon_out=*/nullptr)) {
        error_ = "tx_type selection failed";
        return false;
      }
      tx_type = choice.candidate.type;
      const auto chosen =
          txv::make_tx_params(cw, ch, tx_type);
      if (!chosen.has_value()) {
        error_ = "unsupported transform size for the chosen tx_type";
        return false;
      }
      params_opt = chosen;
      apply_qindex_step(*params_opt);
    }
    plan.plane = plane;
    plan.coded_w = cw;
    plan.coded_h = ch;
    plan.pred = pred;

    // Quantise/inverse parameterised by the effective transform.  Runs once
    // for the chosen TX type, and a second time as DCT_DCT when TX_TYPE
    // selection plus RDOQ leaves eob <= 1 (the decoder infers DCT there).
    const auto run_quant = [&](const txv::TxParams& p) -> bool {
      std::vector<std::int16_t> residual_adj(
          static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h));
      txv::downsample_residual(residual_coded.data(), p, residual_adj.data());
      std::vector<double> coeff(
          static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h));
      plan.levels.assign(coeff.size(), 0);
      txv::forward_transform(residual_adj.data(), p, coeff.data(),
                             plan.levels.data());

      const int area = p.w * p.h;
      std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
      syntax::build_scan(p.w, p.h, scan);

      // Full-trellis RDOQ: same syntax, better coefficient levels.  The
      // trellis rate model is the static qctx-0 AV2 model; the syntax writer
      // and the reconstruction both consume the returned levels, so the
      // stream stays decoder-exact.
      if (options_.rdoq) {
        std::vector<std::int32_t> scalar_scan(
            static_cast<std::size_t>(area));
        std::vector<av2enc::rdoq::Coefficient> input(
            static_cast<std::size_t>(area));
        for (int c = 0; c < area; ++c) {
          const int pos = scan[static_cast<std::size_t>(c)];
          scalar_scan[static_cast<std::size_t>(c)] = plan.levels[pos];
          input[static_cast<std::size_t>(c)].original = coeff[pos];
          input[static_cast<std::size_t>(c)].scalar_level = plan.levels[pos];
          input[static_cast<std::size_t>(c)].level_context = 0;
        }
        av2enc::rdoq::av2::BlockInfo info;
        info.plane = plane;
        info.params = p;
        info.txb_skip_context = 0;
        info.dc_sign_context = 0;
        info.qctx = qctx_slice_;
        av2enc::rdoq::av2::Model model;
        std::string rdoq_error;
        if (!model.init(info, scalar_scan, &rdoq_error)) {
          error_ = "rdoq model init failed: " + rdoq_error;
          return false;
        }
        av2enc::rdoq::Config rdoq_config;
        rdoq_config.quant_step = 1.0 / p.quant_factor;
        rdoq_config.lambda = av2enc::rdo::lambda_sse_from_quant_step(
            rdoq_config.quant_step, options_.rdoq_sse_scale);
        rdoq_config.max_coefficients = area;
        av2enc::rdoq::RdoqResult rdoq_result;
        if (!av2enc::rdoq::rdoq_quantize(input, rdoq_config,
                                         model.rate_model(), rdoq_result,
                                         &rdoq_error)) {
          error_ = "rdoq quantize failed: " + rdoq_error;
          return false;
        }
        std::fill(plan.levels.begin(), plan.levels.end(), 0);
        for (int c = 0; c < rdoq_result.eob; ++c) {
          plan.levels[scan[static_cast<std::size_t>(c)]] =
              rdoq_result.levels[static_cast<std::size_t>(c)];
        }
      }

      if (plane == 0 && std::getenv("AV2ENC_STILL_ZERO_DC") != nullptr) {
        plan.levels[0] = 0;
      }
      plan.eob = 0;
      for (int c = area - 1; c >= 0; --c) {
        if (plan.levels[scan[static_cast<std::size_t>(c)]] != 0) {
          plan.eob = c + 1;
          break;
        }
      }
      return true;
    };

    if (!run_quant(*params_opt)) return false;
    if (plane == 0 && tx_type != txv::TxType::kDctDct && plan.eob <= 1) {
      const auto dct = txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
      if (!dct.has_value()) {
        error_ = "unsupported transform size for the DCT fallback";
        return false;
      }
      tx_type = txv::TxType::kDctDct;
      params_opt = dct;
      apply_qindex_step(*params_opt);
      if (!run_quant(*params_opt)) return false;
    }

    const txv::TxParams& params = *params_opt;
    plan.tx_size = params.tx_size;
    plan.tx_type = static_cast<int>(tx_type);
    plan.adj_w = params.w;
    plan.adj_h = params.h;
    std::vector<std::int32_t> dequant(
        static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
    std::vector<std::int16_t> recon_residual(
        static_cast<std::size_t>(coded_pels));
    dequant_inverse_qindex(plan.levels.data(), params, dequant.data(),
                           recon_residual.data());
    plan.recon.resize(static_cast<std::size_t>(coded_pels));
    for (int y = 0; y < ch; ++y) {
      for (int x = 0; x < cw; ++x) {
        const int p =
            pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                 static_cast<std::size_t>(x)];
        const int r =
            recon_residual[static_cast<std::size_t>(y) *
                               static_cast<std::size_t>(cw) +
                           static_cast<std::size_t>(x)];
        plan.recon[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                   static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(clip1(p + r));
      }
    }
    return true;
  }

  // Append one plane's levels to the flat array in *scan order*: the syntax
  // writer consumes levels_[offset + c] as scan position c.
  void append_levels(const TxPlan& plan) {
    const int area = plan.adj_w * plan.adj_h;
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    syntax::build_scan(plan.adj_w, plan.adj_h, scan);
    for (int c = 0; c < plan.eob; ++c) {
      levels_.push_back(plan.levels[scan[static_cast<std::size_t>(c)]]);
    }
  }

  void write_recon(int plane, int x0, int y0, int cw, int ch,
                   const TxPlan& plan) {
    // A coding block at the right/bottom frame edge may be larger than the
    // MiCols*4 / MiRows*4 coded plane: the decoder parses the full transform
    // block but only stores the part inside the plane. Clamp identically.
    const int max_x = recon_[plane].width;
    const int max_y = recon_[plane].height;
    for (int y = 0; y < ch; ++y) {
      const int py = y0 + y;
      if (py < 0 || py >= max_y) continue;
      for (int x = 0; x < cw; ++x) {
        const int px = x0 + x;
        if (px < 0 || px >= max_x) continue;
        recon_[plane].at(px, py) =
            plan.recon[static_cast<std::size_t>(y) *
                           static_cast<std::size_t>(cw) +
                       static_cast<std::size_t>(x)];
      }
    }
  }

  // ---------------------------------------------------------------------
  // Coding a block
  // ---------------------------------------------------------------------
  bool encode_block(const partition::PartitionNode& node, std::size_t index) {
    const int r = node.mi_row;
    const int c = node.mi_col;
    const int w = partition::block_width(node.size);
    const int h = partition::block_height(node.size);
    const int w4 = w >> 2;
    const int h4 = h >> 2;
    const int x0 = c * 4;
    const int y0 = r * 4;

    ModeChoice y;
    if (!choose_luma_mode(x0, y0, w, h, y)) return false;

    std::vector<std::uint8_t> luma_pred;
    if (!predict_block(0, x0, y0, w, h, y.mode, y.angle_delta, luma_pred)) {
      return false;
    }
    TxPlan luma_plan;
    if (!compute_tx(0, x0, y0, w, h, luma_pred, luma_plan,
                    options_.tx_type, y.av2_mode, y.angle_delta)) {
      return false;
    }
    write_recon(0, x0, y0, w, h, luma_plan);

    const BlockSize chroma_size =
        partition::chroma_plane_residual_size(geometry_, node.size);
    if (!partition::is_valid_block_size(chroma_size)) {
      error_ = "invalid chroma residual size";
      return false;
    }
    const int cw = partition::block_width(chroma_size);
    const int ch = partition::block_height(chroma_size);
    const int cx0 = x0 / 2;
    const int cy0 = y0 / 2;

    UvChoice uv;
    if (!choose_uv_mode(cx0, cy0, cw, ch, y.av2_mode, y.angle_delta, uv)) {
      return false;
    }

    std::vector<std::uint8_t> u_pred;
    std::vector<std::uint8_t> v_pred;
    bool cfl = uv.is_cfl;
    if (cfl) {
      // The CfL component expects the DC prediction in the chroma plane and
      // reconstructed above/left neighbours. Derive the implicit alpha and
      // write the CfL prediction into the planes.
      if (!predict_block(1, cx0, cy0, cw, ch, intra::Mode::kDc, 0, u_pred)) {
        return false;
      }
      if (!predict_block(2, cx0, cy0, cw, ch, intra::Mode::kDc, 0, v_pred)) {
        return false;
      }
      for (int yy = 0; yy < ch; ++yy) {
        for (int xx = 0; xx < cw; ++xx) {
          recon_[1].at(cx0 + xx, cy0 + yy) =
              u_pred[static_cast<std::size_t>(yy) * cw + xx];
          recon_[2].at(cx0 + xx, cy0 + yy) =
              v_pred[static_cast<std::size_t>(yy) * cw + xx];
        }
      }
      cfl::Scratch scratch;
      cfl::Block ub;
      cfl::Block vb;
      make_cfl_block(1, cx0, cy0, cw, ch, ub);
      make_cfl_block(2, cx0, cy0, cw, ch, vb);
      cfl::compute_luma(ub, scratch);
      const int alpha_u = cfl::derive_implicit_alpha(ub, scratch);
      cfl::compute_luma(vb, scratch);
      const int alpha_v = cfl::derive_implicit_alpha(vb, scratch);
      cfl::apply_prediction(ub, alpha_u, scratch);
      cfl::apply_prediction(vb, alpha_v, scratch);
      for (int yy = 0; yy < ch; ++yy) {
        for (int xx = 0; xx < cw; ++xx) {
          u_pred[static_cast<std::size_t>(yy) * cw + xx] =
              recon_[1].at(cx0 + xx, cy0 + yy);
          v_pred[static_cast<std::size_t>(yy) * cw + xx] =
              recon_[2].at(cx0 + xx, cy0 + yy);
        }
      }
    } else {
      if (!predict_block(1, cx0, cy0, cw, ch, uv.mode, uv.angle_delta,
                         u_pred)) {
        return false;
      }
      if (!predict_block(2, cx0, cy0, cw, ch, uv.mode, uv.angle_delta,
                         v_pred)) {
        return false;
      }
    }

    TxPlan u_plan;
    TxPlan v_plan;
    if (!compute_tx(1, cx0, cy0, cw, ch, u_pred, u_plan)) return false;
    if (!compute_tx(2, cx0, cy0, cw, ch, v_pred, v_plan)) return false;
    write_recon(1, cx0, cy0, cw, ch, u_plan);
    write_recon(2, cx0, cy0, cw, ch, v_plan);

    if (trace_) {
      std::cerr << "m1 block r=" << r << " c=" << c
                << " size=" << partition::block_size_name(node.size)
                << " y=" << y.av2_mode << "/" << y.angle_delta
                << " uv=" << (cfl ? -1 : uv.av2_mode) << "/"
                << uv.angle_delta << " cfl=" << (cfl ? 1 : 0)
                << " tx=" << luma_plan.coded_w << "x" << luma_plan.coded_h
                << " eob=" << luma_plan.eob << "\n";
    }

    // Fill this leaf's syntax entry: modes, transform sizes, EOBs, and its
    // slice of the flat scan-order coefficient array.  No bits are written
    // here; syntax::assemble_still_frame() emits the frame after the tree is done.
    syntax::BlockSyntax& leaf = blocks_[index];
    leaf.y_mode = static_cast<std::uint8_t>(y.av2_mode);
    leaf.y_angle_delta = static_cast<std::int8_t>(y.angle_delta);
    leaf.uv_mode = static_cast<std::uint8_t>(uv.av2_mode);
    if (cfl) leaf.flags |= syntax::kFlagIsCfl;
    const TxPlan* plans[3] = {&luma_plan, &u_plan, &v_plan};
    leaf.levels_offset = static_cast<std::uint32_t>(levels_.size());
    for (int plane = 0; plane < 3; ++plane) {
      const TxPlan& plan = *plans[plane];
      leaf.tx_size[plane] = static_cast<std::uint8_t>(plan.tx_size);
      leaf.tx_type[plane] = static_cast<std::uint8_t>(plan.tx_type);
      leaf.eob[plane] = static_cast<std::uint32_t>(plan.eob);
      append_levels(plan);
    }

    // Update the mode state that intra prediction reads for later blocks.
    const int joint = syntax::y_mode_joint(y.av2_mode, y.angle_delta);
    for (int yy = 0; yy < h4; ++yy) {
      const int rr = r + yy;
      if (rr >= mi_rows_) break;
      for (int xx = 0; xx < w4; ++xx) {
        const int cc = c + xx;
        if (cc >= mi_cols_) break;
        intra_joint(rr, cc) = joint;
      }
    }
    const int cr = r / 2;
    const int cc0 = c / 2;
    const int cw4 = cw / 4;
    const int ch4 = ch / 4;
    const int chroma_mi_rows = mi_rows_ / 2;
    const int chroma_mi_cols = mi_cols_ / 2;
    for (int yy = 0; yy < ch4; ++yy) {
      const int rr = cr + yy;
      if (rr >= chroma_mi_rows) break;
      for (int xx = 0; xx < cw4; ++xx) {
        const int cc = cc0 + xx;
        if (cc >= chroma_mi_cols) break;
        uv_smooth(rr, cc) =
            (!cfl && y_mode_is_smooth(uv.av2_mode)) ? 1 : 0;
      }
    }

    for (int plane = 0; plane < 3; ++plane) {
      decoded_[plane].fill(
          plane == 0 ? c : c / 2, plane == 0 ? r : r / 2,
          plane == 0 ? w4 : cw4, plane == 0 ? h4 : ch4);
    }

    if (options_.enable_deblock) {
      fill_deblock_mi(node, cw, ch, luma_plan.tx_size, u_plan.tx_size);
    }
    return true;
  }

  void make_cfl_block(int plane, int cx0, int cy0, int cw, int ch,
                      cfl::Block& block) {
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
    block.have_above = cfl::cfl_avail_above(cy0, 1) ? 1 : 0;
    block.have_left = cfl::cfl_avail_left(cx0, 1) ? 1 : 0;
    block.bit_depth = 8;
    block.ds_filter = 0;
    block.sub_x = 1;
    block.sub_y = 1;
  }

  bool choose_uv_mode(int cx0, int cy0, int cw, int ch, int y_mode,
                      int y_angle_delta, UvChoice& best) {
    best.is_cfl = false;
    best.sse = std::numeric_limits<std::int64_t>::max();
    const bool uv_full =
        options_.uv_full || std::getenv("AV2ENC_STILL_UV_FULL") != nullptr;
    if (std::getenv("AV2ENC_STILL_UV_DC") != nullptr) {
      best.mode = intra::Mode::kDc;
      best.angle_delta = 0;
      best.av2_mode = kDcPred;
      best.sse = 0;
      return true;
    }
    // Debug: force one chroma mode for decoder verification.
    if (const char* forced = std::getenv("AV2ENC_STILL_FORCE_UV")) {
      int mode = 0;
      int delta = 0;
      if (std::sscanf(forced, "%d:%d", &mode, &delta) == 2 &&
          mode >= 0 && mode <= 12 && delta >= -3 && delta <= 3) {
        // Only UVMode == YMode can carry a non-zero AngleDeltaUV, and the
        // decoder derives it from AngleDeltaY; the explicit delta here is
        // accepted only for that case.
        const int derived =
            (y_mode_is_directional(y_mode) && mode == y_mode)
                ? y_angle_delta
                : 0;
        if (delta != 0 && delta != derived) {
          error_ = "AV2ENC_STILL_FORCE_UV non-zero delta requires UVMode==YMode";
          return false;
        }
        best.mode = static_cast<intra::Mode>(mode);
        best.angle_delta = derived;
        best.av2_mode = av2_y_mode(best.mode);
        best.sse = 0;
        return true;
      }
      error_ = "invalid AV2ENC_STILL_FORCE_UV (expected mode:delta)";
      return false;
    }
    if (options_.force_dc || (!options_.enable_cfl && !uv_full)) {
      best.mode = intra::Mode::kDc;
      best.angle_delta = 0;
      best.av2_mode = kDcPred;
      best.sse = 0;
      return true;
    }
    std::vector<std::uint8_t> pred;

    // Try the 13 base modes. AngleDeltaUV is not signalled: §6.19.5.6
    // derives it as AngleDeltaY only when UVMode == YMode and YMode is
    // directional; every other combination uses 0. There is no separate UV
    // angle-delta syntax to search over.
    static const intra::Mode kBase[13] = {
        intra::Mode::kDc,      intra::Mode::kV,       intra::Mode::kH,
        intra::Mode::kD45,     intra::Mode::kD135,    intra::Mode::kD113,
        intra::Mode::kD157,    intra::Mode::kD203,    intra::Mode::kD67,
        intra::Mode::kSmooth,  intra::Mode::kSmoothV, intra::Mode::kSmoothH,
        intra::Mode::kPaeth};

    for (int i = 0; i < 13; ++i) {
      const intra::Mode mode = kBase[i];
      // §6.19.5.6: if UVMode == YMode and YMode is directional, the decoder
      // derives AngleDeltaUV = AngleDeltaY. AngleDeltaUV is not signalled, so
      // evaluate and store that derived delta for this candidate. All other
      // UV modes use AngleDeltaUV = 0.
      const int delta =
          (y_mode_is_directional(y_mode) &&
           static_cast<int>(mode) == y_mode)
              ? y_angle_delta
              : 0;
      // M1 default: chroma is DC or CfL. Directional chroma prediction is
      // enabled only behind uv_full (--still-uv-full) until forced-UV decoder
      // verification passes.
      if (!uv_full && mode != intra::Mode::kDc) continue;
      if (!predict_block(1, cx0, cy0, cw, ch, mode, delta, pred)) return false;
      std::int64_t sse = sse_against_source(1, cx0, cy0, cw, ch, pred);
      if (uv_full) {
        // Score both chroma planes for the full search so a U-only win cannot
        // pick a mode that is worse for V. The default DC/CfL path keeps the
        // historical U-only DC score for decision parity.
        if (!predict_block(2, cx0, cy0, cw, ch, mode, delta, pred)) return false;
        sse += sse_against_source(2, cx0, cy0, cw, ch, pred);
      }
      if (sse < best.sse) {
        best.sse = sse;
        best.mode = mode;
        best.angle_delta = delta;
        best.av2_mode = av2_y_mode(mode);
      }
    }

    if (options_.enable_cfl) {
      std::vector<std::uint8_t> u_dc;
      std::vector<std::uint8_t> v_dc;
      if (!predict_block(1, cx0, cy0, cw, ch, intra::Mode::kDc, 0, u_dc)) {
        return false;
      }
      if (!predict_block(2, cx0, cy0, cw, ch, intra::Mode::kDc, 0, v_dc)) {
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
      cfl::Scratch scratch;
      cfl::Block ub;
      cfl::Block vb;
      make_cfl_block(1, cx0, cy0, cw, ch, ub);
      make_cfl_block(2, cx0, cy0, cw, ch, vb);
      cfl::compute_luma(ub, scratch);
      const int alpha_u = cfl::derive_implicit_alpha(ub, scratch);
      cfl::compute_luma(vb, scratch);
      const int alpha_v = cfl::derive_implicit_alpha(vb, scratch);
      cfl::apply_prediction(ub, alpha_u, scratch);
      cfl::apply_prediction(vb, alpha_v, scratch);
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
      if (sse < best.sse) {
        best.sse = sse;
        best.is_cfl = true;
        best.mode = intra::Mode::kDc;
        best.angle_delta = 0;
        best.av2_mode = kDcPred;
      }
    }
    return true;
  }

  void fill_deblock_mi(const partition::PartitionNode& node, int cw, int ch,
                       int luma_tx_size, int chroma_tx_size) {
    const int w = partition::block_width(node.size);
    const int h = partition::block_height(node.size);
    const int w4 = w / 4;
    const int h4 = h / 4;
    const TxSize luma = static_cast<TxSize>(luma_tx_size);
    const TxSize chroma = static_cast<TxSize>(chroma_tx_size);
    (void)cw;
    (void)ch;
    for (int yy = 0; yy < h4; ++yy) {
      const int rr = node.mi_row + yy;
      if (rr >= mi_rows_) break;
      for (int xx = 0; xx < w4; ++xx) {
        const int cc = node.mi_col + xx;
        if (cc >= mi_cols_) break;
        const int idx = rr * deblock_cols_ + cc;
        const int clamped_w4 = std::min(w4, mi_cols_ - node.mi_col);
        const int clamped_h4 = std::min(h4, mi_rows_ - node.mi_row);
        deblock::DeblockMi mi;
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
        deblock_mi_[0][static_cast<std::size_t>(idx)] = mi;
        deblock_mi_[1][static_cast<std::size_t>(idx)] = mi;
        deblock_mi_[2][static_cast<std::size_t>(idx)] = mi;
        deblock_mi_[1][static_cast<std::size_t>(idx)].tx_size = chroma;
        deblock_mi_[2][static_cast<std::size_t>(idx)].tx_size = chroma;
      }
    }
  }

  // Members
  const StillPicture& picture_;
  const StillOptions& options_;
  FrameGeometry geometry_;
  int width_ = 0;
  int height_ = 0;
  int chroma_width_ = 0;
  int chroma_height_ = 0;
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  // §7.14.2 qlookup(base_q_idx).  528 at the historical M1 QP 90.
  int quant_step_ = txv::kQuantStep;
  // §8.3.1 COEFF_CDF_Q_CTXS slice for the RDOQ rate model.
  int qctx_slice_ = 0;
  int luma_stride_ = 0;
  int luma_height_ = 0;
  int chroma_stride_ = 0;
  int chroma_padded_height_ = 0;

  std::string error_;
  Plane recon_[3];

  // Preorder partition nodes and the flat scan-order coefficient levels.  The
  // syntax writer rebuilds every CDF context from these two arrays.
  std::vector<syntax::BlockSyntax> blocks_;
  std::vector<std::int32_t> levels_;

  std::vector<int> intra_joint_;
  std::vector<std::uint8_t> uv_smooth_;
  std::array<DecodedMap, 3> decoded_;

  int sb_mi_row_ = 0;
  int sb_mi_col_ = 0;
  int sb_cols_plane_[3] = {0, 0, 0};
  int sb_rows_plane_[3] = {0, 0, 0};

  bool trace_ = std::getenv("AV2ENC_STILL_TRACE") != nullptr;

  std::array<std::vector<deblock::DeblockMi>, 3> deblock_mi_;
  int leaves_ = 0;
  int deblock_cols_ = 0;
  int deblock_rows_ = 0;

  using TxSize = deblock::TxSize;
};

}  // namespace

bool encode_still_picture(const StillPicture& picture,
                             const StillOptions& options, StillResult& result,
                             std::string& error) {
  error.clear();
  if (picture.width == 0 || picture.height == 0) {
    error = "empty picture";
    return false;
  }
  if ((picture.width & 1U) != 0U || (picture.height & 1U) != 0U) {
    error = "M1 4:2:0 requires even dimensions";
    return false;
  }
  // base_q_idx 1..255 is supported: the coefficient CDFs come from the §8.3.1
  // qctx slice for the frame QP and the quantiser uses qlookup(base_q_idx).
  // 0 is lossless, which this path does not implement.
  if (options.base_q_idx == 0) {
    error = "M1 does not support lossless base_q_idx = 0";
    return false;
  }
  const std::size_t luma_size = static_cast<std::size_t>(picture.width) *
                                static_cast<std::size_t>(picture.height);
  const std::size_t chroma_size =
      static_cast<std::size_t>((picture.width + 1U) / 2U) *
      static_cast<std::size_t>((picture.height + 1U) / 2U);
  if (picture.y.size() < luma_size || picture.u.size() < chroma_size ||
      picture.v.size() < chroma_size) {
    error = "picture planes are too small";
    return false;
  }

  Encoder encoder(picture, options);
  SyntaxStream stream;
  if (!encoder.encode(&stream, error)) {
    return false;
  }

  // Deblock the reconstruction before CDEF or output.  Intra prediction already
  // used the unfiltered reconstruction, matching normative filter order.
  if (options.enable_deblock) {
    deblock::DeblockFrame frame;
    for (int plane = 0; plane < 3; ++plane) {
      frame.planes[plane].data = encoder.recon_mut(plane).samples.data();
      frame.planes[plane].stride = encoder.recon_mut(plane).width;
      frame.mi[plane] = encoder.deblock_mi(plane);
    }
    frame.mi_cols = encoder.deblock_cols();
    frame.mi_rows = encoder.deblock_rows();
    frame.subsampling_x = 1;
    frame.subsampling_y = 1;
    deblock::DeblockOptions dopts;
    for (int i = 0; i < 4; ++i) dopts.apply_deblocking_filter[i] = true;
    dopts.frame_is_intra = true;
    dopts.enable_sdp = false;
    const deblock::DeblockStatus status =
        deblock::deblock_frame(frame, dopts);
    if (status != deblock::DeblockStatus::kOk) {
      error = std::string("deblock failed: ") +
              deblock::deblock_status_string(status);
      return false;
    }
  }

  // CDEF (§7.18) runs after deblock and before output.  The search needs the
  // original picture (only for the SSE metric) and the deblocked reconstruction.
  cuda::still::CdefStagePlan cdef_plan;
  cuda::still::CdefMiGrids cdef_grids;
  Plane cdef_planes[3];
  bool cdef_ready = false;
  if (options.enable_cdef) {
    if (options.cdef_num_strengths < 1 || options.cdef_num_strengths > 8) {
      error = "cdef_num_strengths outside 1..8";
      return false;
    }
    cuda::still::CdefMiInputs mi_inputs;
    mi_inputs.mi_cols = encoder.mi_cols();
    mi_inputs.mi_rows = encoder.mi_rows();
    if (!cuda::still::still_cdef_build_mi_grids(mi_inputs, &cdef_grids, &error)) {
      return false;
    }

    // The search/filter want MI-aligned reference planes; the visible picture
    // may be narrower.  Only the visible area is scored, so zero padding is
    // sufficient.
    const int mi_luma_w = encoder.mi_cols() * 4;
    const int mi_luma_h = encoder.mi_rows() * 4;
    const int mi_chroma_w = mi_luma_w / 2;
    const int mi_chroma_h = mi_luma_h / 2;
    const int ref_w[3] = {mi_luma_w, mi_chroma_w, mi_chroma_w};
    const int ref_h[3] = {mi_luma_h, mi_chroma_h, mi_chroma_h};
    const int vis_w[3] = {static_cast<int>(picture.width),
                          static_cast<int>((picture.width + 1U) / 2U),
                          static_cast<int>((picture.width + 1U) / 2U)};
    const int vis_h[3] = {static_cast<int>(picture.height),
                          static_cast<int>((picture.height + 1U) / 2U),
                          static_cast<int>((picture.height + 1U) / 2U)};
    std::vector<std::uint8_t> ref_planes[3];
    const std::uint8_t* ref_ptr[3] = {nullptr, nullptr, nullptr};
    for (int plane = 0; plane < 3; ++plane) {
      const std::span<const std::uint8_t> src =
          plane == 0 ? picture.y : (plane == 1 ? picture.u : picture.v);
      ref_planes[plane].assign(
          static_cast<std::size_t>(ref_w[plane]) *
              static_cast<std::size_t>(ref_h[plane]),
          0);
      for (int y = 0; y < vis_h[plane]; ++y) {
        std::copy_n(src.begin() +
                        static_cast<std::ptrdiff_t>(y) * vis_w[plane],
                    vis_w[plane],
                    ref_planes[plane].begin() +
                        static_cast<std::ptrdiff_t>(y) * ref_w[plane]);
      }
      ref_ptr[plane] = ref_planes[plane].data();
    }

    cuda::still::CdefStageFrame cdef_frame{};
    for (int plane = 0; plane < 3; ++plane) {
      const Plane& recon = encoder.recon(plane);
      cdef_planes[plane].resize(recon.width, recon.height, 0);
      cdef_frame.plane[plane].src = recon.samples.data();
      cdef_frame.plane[plane].dst = cdef_planes[plane].samples.data();
      cdef_frame.plane[plane].stride = recon.width;
      cdef_frame.plane[plane].height = recon.height;
      cdef_frame.ref[plane] = ref_ptr[plane];
      cdef_frame.ref_stride[plane] = ref_w[plane];
    }
    cdef_frame.width = vis_w[0];
    cdef_frame.height = vis_h[0];
    cdef_frame.mi_cols = encoder.mi_cols();
    cdef_frame.mi_rows = encoder.mi_rows();
    cdef_frame.skip = cdef_grids.skip.data();
    cdef_frame.lossless = cdef_grids.lossless.data();
    cdef_frame.lossless_chroma = cdef_grids.lossless_chroma.data();

    cuda::still::CdefStageOptions cdef_options;
    cdef_options.damping = options.cdef_damping;
    cdef_options.num_strengths = options.cdef_num_strengths;
    cdef_options.on_skip_txfm = options.cdef_on_skip_txfm;
    cdef_options.include_off = true;
    if (!cuda::still::still_cdef_plan(cdef_frame, cdef_options, &cdef_plan,
                                &error)) {
      error = "cdef plan failed: " + error;
      return false;
    }

    // Every coding block whose 64x64 origin is already coded writes nothing;
    // fill every leaf so the syntax writer validates the value everywhere.
    const int n64_cols = cdef_plan.n64_cols;
    for (syntax::BlockSyntax& leaf : stream.blocks) {
      if (leaf.partition != static_cast<std::uint8_t>(PartitionType::kNone)) {
        continue;
      }
      const int br = leaf.mi_row >> 4;
      const int bc = leaf.mi_col >> 4;
      if (br < 0 || bc < 0 || br >= cdef_plan.n64_rows ||
          bc >= n64_cols) {
        error = "cdef index grid does not cover a coding block";
        return false;
      }
      leaf.cdef_idx = cdef_plan.cdef_idx[static_cast<std::size_t>(br) *
                                             static_cast<std::size_t>(n64_cols) +
                                         static_cast<std::size_t>(bc)];
    }
    cdef_ready = true;
  }

  // CCSO reads the *pre-CDEF* luma for its band classifier, so keep a copy of
  // the deblocked planes before CDEF overwrites the reconstruction.
  Plane pre_cdef_planes[3];
  if (options.enable_ccso) {
    for (int plane = 0; plane < 3; ++plane) {
      pre_cdef_planes[plane] = encoder.recon(plane);
    }
  }

  // Apply CDEF now: CCSO consumes the post-CDEF frame, and the syntax stage
  // only needs the decisions (already in `stream`).
  if (cdef_ready) {
    cdef::CdefFrame filter_frame{};
    for (int plane = 0; plane < 3; ++plane) {
      const Plane& recon = encoder.recon(plane);
      filter_frame.planes[plane].src = recon.samples.data();
      filter_frame.planes[plane].dst = cdef_planes[plane].samples.data();
      filter_frame.planes[plane].stride = recon.width;
    }
    filter_frame.mi_cols = encoder.mi_cols();
    filter_frame.mi_rows = encoder.mi_rows();
    filter_frame.subsampling_x = 1;
    filter_frame.subsampling_y = 1;
    filter_frame.skip = cdef_grids.skip.data();
    filter_frame.lossless = cdef_grids.lossless.data();
    filter_frame.lossless_chroma = cdef_grids.lossless_chroma.data();
    filter_frame.cdef_idx = cdef_plan.cdef_idx.data();
    filter_frame.n64_cols = cdef_plan.n64_cols;
    filter_frame.n64_rows = cdef_plan.n64_rows;
    const cdef::CdefStatus status =
        cdef::cdef_frame(filter_frame, cdef_plan.search.options);
    if (status != cdef::CdefStatus::kOk) {
      error = std::string("cdef filter failed: ") +
              cdef::cdef_status_string(status);
      return false;
    }
    for (int plane = 0; plane < 3; ++plane) {
      encoder.recon_mut(plane).samples.swap(cdef_planes[plane].samples);
    }
  }

  StillConfig config;

  // CCSO (§7.19).  Band-offset-only, one offset per band and plane; the fit
  // minimises SSE against the original picture on visible samples.
  restoration::CcsoFrameParams ccso_params;
  std::vector<std::uint8_t> ccso_blk_grid;
  int ccso_blk_cols = 0;
  int ccso_blk_rows = 0;
  bool ccso_ready = false;
  if (options.enable_ccso) {
    const int bands_log2 = options.ccso_bands_log2;
    if (bands_log2 < 0 || bands_log2 > 6) {
      error = "ccso_bands_log2 outside 0..6";
      return false;
    }
    const int bands = 1 << bands_log2;
    const int vis_w[3] = {static_cast<int>(picture.width),
                          static_cast<int>((picture.width + 1U) / 2U),
                          static_cast<int>((picture.width + 1U) / 2U)};
    const int vis_h[3] = {static_cast<int>(picture.height),
                          static_cast<int>((picture.height + 1U) / 2U),
                          static_cast<int>((picture.height + 1U) / 2U)};
    const std::span<const std::uint8_t> src_planes[3] = {
        picture.y, picture.u, picture.v};
    const Plane& curr_luma = pre_cdef_planes[0];
    const int band_shift = 8 - bands_log2;
    for (int plane = 0; plane < 3; ++plane) {
      const int sub = plane == 0 ? 0 : 1;
      const Plane& rec = encoder.recon(plane);
      std::vector<double> sum_e(static_cast<std::size_t>(bands), 0.0);
      std::vector<double> sum_e2(static_cast<std::size_t>(bands), 0.0);
      std::vector<double> count(static_cast<std::size_t>(bands), 0.0);
      for (int y = 0; y < vis_h[plane]; ++y) {
        for (int x = 0; x < vis_w[plane]; ++x) {
          const int lx = x << sub;
          const int ly = y << sub;
          const int band = curr_luma.at(lx, ly) >> band_shift;
          const int src = src_planes[plane][static_cast<std::size_t>(y) *
                                                static_cast<std::size_t>(
                                                    vis_w[plane]) +
                                            static_cast<std::size_t>(x)];
          const int d = src - rec.at(x, y);
          sum_e[static_cast<std::size_t>(band)] += d;
          sum_e2[static_cast<std::size_t>(band)] +=
              static_cast<double>(d) * d;
          count[static_cast<std::size_t>(band)] += 1.0;
        }
      }
      StillCcsoPlane& plane_config = config.ccso.plane[plane];
      plane_config.enabled = true;
      plane_config.bo_only = true;
      plane_config.scale_idx = 0;
      plane_config.max_band_log2 = bands_log2;
      int nonzero = 0;
      for (int band = 0; band < bands; ++band) {
        int best_idx = 0;
        double best_sse = sum_e2[static_cast<std::size_t>(band)];
        for (int i = 0; i < 8; ++i) {
          const double off = static_cast<double>(restoration::kCcsoOffset[i]);
          const double sse = sum_e2[static_cast<std::size_t>(band)] -
                             2.0 * off * sum_e[static_cast<std::size_t>(band)] +
                             off * off * count[static_cast<std::size_t>(band)];
          if (sse < best_sse - 1e-9) {
            best_sse = sse;
            best_idx = i;
          }
        }
        plane_config.offset_idx[band] =
            static_cast<std::uint8_t>(best_idx);        if (best_idx != 0) ++nonzero;
      }
      if (nonzero == 0 && plane == 0) {
        plane_config.enabled = false;
      }
      restoration::CcsoParams& params = ccso_params.plane[plane];
      params.enabled = plane_config.enabled;
      params.bo_only = true;
      params.scale_idx = plane_config.scale_idx;
      params.quant_idx = 0;
      params.ext_filter = 0;
      params.edge_clf = 0;
      params.max_band_log2 = bands_log2;
      if (params.enabled &&
          !restoration::ccso_set_offsets(params, plane_config.offset_idx,
                                         static_cast<std::size_t>(bands))) {
        error = "ccso_set_offsets failed";
        return false;
      }
    }
    ccso_params.ccso_luma_size_log2 = 6;
    for (int plane = 0; plane < 3; ++plane) {
      const int sub = plane == 0 ? 0 : 1;
      const int plane_w = (encoder.mi_cols() * 4) >> sub;
      const int plane_h = (encoder.mi_rows() * 4) >> sub;
      int cols = 0;
      int rows = 0;
      restoration::ccso_blk_grid_size(plane_w, plane_h, 6, sub, sub, &cols,
                                      &rows);
      ccso_blk_cols = cols;
      ccso_blk_rows = rows;
      break;
    }
    ccso_blk_grid.assign(static_cast<std::size_t>(ccso_blk_cols) *
                             static_cast<std::size_t>(ccso_blk_rows),
                         1);
    for (int plane = 0; plane < 3; ++plane) {
      ccso_params.blks[plane] = ccso_blk_grid.data();
      ccso_params.blk_cols[plane] = ccso_blk_cols;
      ccso_params.blk_rows[plane] = ccso_blk_rows;
    }
    restoration::RestorationFrame filter_frame{};
    for (int plane = 0; plane < 3; ++plane) {
      const Plane& curr = pre_cdef_planes[plane];
      Plane& rec = encoder.recon_mut(plane);
      filter_frame.curr[plane].data =
          const_cast<std::uint8_t*>(curr.samples.data());
      filter_frame.curr[plane].stride = curr.width;
      filter_frame.cdef[plane].data = rec.samples.data();
      filter_frame.cdef[plane].stride = rec.width;
      filter_frame.lr[plane].data = rec.samples.data();
      filter_frame.lr[plane].stride = rec.width;
    }
    filter_frame.mi_cols = encoder.mi_cols();
    filter_frame.mi_rows = encoder.mi_rows();
    filter_frame.subsampling_x = 1;
    filter_frame.subsampling_y = 1;
    const restoration::RestorationStatus status =
        restoration::ccso_frame(filter_frame, ccso_params);
    if (status != restoration::RestorationStatus::kOk) {
      error = std::string("ccso failed: ") +
              restoration::restoration_status_string(status);
      return false;
    }
    bool any_plane = false;
    for (int plane = 0; plane < 3; ++plane) {
      any_plane = any_plane || config.ccso.plane[plane].enabled;
    }
    if (!any_plane) {
      config.ccso.enable = false;
      ccso_ready = false;
    } else {
      config.ccso.enable = true;
      ccso_ready = true;
    }
  }

  config.max_frame_width = static_cast<std::uint16_t>(picture.width);
  config.max_frame_height = static_cast<std::uint16_t>(picture.height);
  config.base_q_idx = options.base_q_idx;
  config.disable_cdf_update = !options.adaptive_cdf;
  config.enable_intra_edge_filter =
      options.enable_intra_edge_filter &&
      std::getenv("AV2ENC_STILL_NO_EDGE") == nullptr;
  config.enable_cfl_intra = options.enable_cfl;
  config.enable_chroma_dctonly = true;
  config.enable_idtx_intra = false;
  config.tx_mode_select = false;
  config.reduced_tx_set = 0;
  config.apply_deblock_y_vert = options.enable_deblock;
  config.apply_deblock_y_horz = options.enable_deblock;
  config.apply_deblock_u = options.enable_deblock;
  config.apply_deblock_v = options.enable_deblock;
  if (cdef_ready) {
    const cuda::still::CdefHeaderParams& header = cdef_plan.header;
    config.cdef.enable = header.frame_enable;
    config.cdef.damping = header.damping_minus_3 + 3;
    config.cdef.num_strengths = header.num_strengths;
    config.cdef.on_skip_txfm = header.on_skip_txfm_frame_enable;
    for (int i = 0; i < header.num_strengths; ++i) {
      config.cdef.strengths[i].y_pri = header.set[i].y_pri_strength;
      config.cdef.strengths[i].y_sec = header.set[i].y_sec_value;
      config.cdef.strengths[i].uv_pri = header.set[i].uv_pri_strength;
      config.cdef.strengths[i].uv_sec = header.set[i].uv_sec_value;
    }
  }

  // Syntax writing is a separate stage: it re-derives every CDF context from
  // the block stream, exactly like the GPU path will.
  syntax::FramePlan plan;
  plan.width = static_cast<int>(picture.width);
  plan.height = static_cast<int>(picture.height);
  plan.sb_size = 64;
  if (ccso_ready) {
    for (int plane = 0; plane < 3; ++plane) {
      plan.ccso_blks[plane] = ccso_blk_grid.data();
      plan.ccso_blk_cols[plane] = ccso_blk_cols;
      plan.ccso_blk_rows[plane] = ccso_blk_rows;
    }
  }
  std::size_t tile_bytes = 0;
  if (!syntax::assemble_still_frame(plan, stream.blocks, stream.levels, config,
                                 result.frame, error, &tile_bytes)) {
    return false;
  }
  result.tile_bytes = static_cast<std::uint32_t>(tile_bytes);
  result.frame_bytes = static_cast<std::uint32_t>(result.frame.size());
  result.leaves = encoder.leaves();

  const auto crop = [](const Plane& plane, int width, int height) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(width) *
                                  static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
      std::copy_n(plane.samples.begin() +
                      static_cast<std::ptrdiff_t>(y) * plane.width,
                  width,
                  out.begin() + static_cast<std::ptrdiff_t>(y) * width);
    }
    return out;
  };
  result.recon_y = crop(encoder.recon(0), static_cast<int>(picture.width),
                        static_cast<int>(picture.height));
  result.recon_u =
      crop(encoder.recon(1), static_cast<int>((picture.width + 1U) / 2U),
           static_cast<int>((picture.height + 1U) / 2U));
  result.recon_v =
      crop(encoder.recon(2), static_cast<int>((picture.width + 1U) / 2U),
           static_cast<int>((picture.height + 1U) / 2U));
  return true;
}

}  // namespace av2enc::still
