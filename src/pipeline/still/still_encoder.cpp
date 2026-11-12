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
#include "codec/intra/cfl/mhccp/mhccp.h"
#include "codec/filters/cdef/cdef.h"
#include "codec/filters/deblock/deblock.h"
#include "codec/filters/loop_restoration/classes/wiener_decision.h"
#include "codec/filters/loop_restoration/restoration.h"
#include "codec/intra/dip/dip.h"
#include "codec/intra/ibp/ibp.h"
#include "codec/filters/gdf/gdf_decision.h"
#include "codec/filters/pc_wiener/pc_wiener_stream.h"
#include "pipeline/grain/film_grain_syntax.h"
#include "pipeline/grain/grain_stream.h"
#include "codec/intra/modes/intra_modes.h"
#include "codec/intra/mrl/core/mrl.h"
#include "codec/intra/mrl/stream/mrl_interaction.h"
#include "codec/intra/palette/core/palette.h"
#include "codec/intra/ibc/core/ibc.h"
#include "codec/intra/ibc/stream/ibc_stack.h"
#include "codec/intra/ibc/stream/ibc_syntax.h"
#include "codec/intra/screen_content/scc_flags.h"
#include "pipeline/still/still_headers.h"
#include "pipeline/still/still_syntax.h"
#include "pipeline/still/still_tables.h"
#include "codec/entropy/qctx/qctx.h"
#include "codec/partition/tree/partition.h"
#include "codec/partition/ext/partition_ext.h"
#include "codec/partition/sdp/sdp.h"
#include "codec/rate/rc/rc.h"
#include "codec/rate/rc/rc_tables.h"
#include "codec/rate/rdo/intra_decision.h"
#include "codec/rate/rdo/rdo.h"
#include "codec/rate/rdoq/rdoq.h"
#include "codec/rate/rdoq/rdoq_av2_model.h"
#include "codec/transform/delta_q/dcdeltaq.h"
#include "codec/transform/qm/qm.h"
#include "codec/transform/secondary/tx_secondary.h"
#include "codec/transform/secondary/tx_secondary_tables.h"
#include "codec/transform/tcq/cctx.h"
#include "codec/transform/tcq/tcq.h"
#include "codec/transform/tcq/tcq_av2_model.h"
#include "codec/transform/tcq/tcq_cdf_tables.h"
#include "codec/transform/tx_partition/tx_partition.h"
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
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace av2enc::still {
namespace {

using partition::BlockSize;
using partition::FrameGeometry;
using partition::PartitionType;
namespace pe = partition_ext;
namespace sdpp = av2enc::sdp;
namespace tpart = av2enc::video::txpartition;

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

// Encoder-side analysis projection for the secondary transform.  The
// component's forward_secondary_coeffs_kernel() only covers the 4x4/8x8 case
// (it zeroes the whole block); for larger transforms the normative inverse
// touches only the top-left sub-block that the §9.7 scan map addresses, so
// project there and leave every other primary coefficient untouched.
bool project_ist_levels(const txv::TxParams& params,
                        txv::TxType plane_tx_type, int kernel,
                        int sec_tx_type, bool transpose,
                        std::vector<std::int32_t>& levels) {
  txsec::SecondarySizeInfo info;
  if (!txsec::secondary_size_info(params, plane_tx_type, &info)) return false;
  if (sec_tx_type < 1 || sec_tx_type >= txsec::kStxTypes) return false;
  if (kernel < 0 || kernel >= info.kernel_count) return false;
  const int w = info.w;
  const int h = info.h;
  const std::size_t elems =
      static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  if (levels.size() < elems) return false;
  const int kernel_width =
      info.large ? txsec::kIst8x8Width : txsec::kIst4x4Width;
  const std::int8_t* matrix =
      info.large ? &txsec::kIst8x8Kernel[kernel][sec_tx_type - 1][0][0]
                 : &txsec::kIst4x4Kernel[kernel][sec_tx_type - 1][0][0];
  const int scan_bwl = info.large ? 3 : 2;
  const int scan_mask = (1 << scan_bwl) - 1;

  std::vector<std::uint16_t> scan(elems);
  syntax::build_scan(w, h, scan);

  std::vector<int> out_pos(static_cast<std::size_t>(info.n2));
  std::vector<std::int32_t> x(static_cast<std::size_t>(info.n2));
  for (int i = 0; i < info.n2; ++i) {
    const int pos = info.large
                        ? txsec::kStxScanOrder8x8[txsec::kStxScanMap
                              [kernel][sec_tx_type - 1][i]]
                        : txsec::kStxScanOrder4x4[i];
    const int sx = pos & scan_mask;
    const int sy = pos >> scan_bwl;
    const int index = transpose ? sx * w + sy : sy * w + sx;
    if (index < 0 || static_cast<std::size_t>(index) >= elems) return false;
    out_pos[static_cast<std::size_t>(i)] = index;
    x[static_cast<std::size_t>(i)] = levels[static_cast<std::size_t>(index)];
  }

  std::vector<std::int32_t> sec(static_cast<std::size_t>(info.n));
  for (int j = 0; j < info.n; ++j) {
    std::int64_t sum = 0;
    for (int i = 0; i < info.n2; ++i) {
      sum += static_cast<std::int64_t>(x[static_cast<std::size_t>(i)]) *
             static_cast<std::int64_t>(matrix[j * kernel_width + i]);
    }
    const std::int64_t magnitude = sum < 0 ? -sum : sum;
    const std::int64_t rounded =
        (magnitude + (std::int64_t{1} << 6)) >> 7;
    sec[static_cast<std::size_t>(j)] =
        static_cast<std::int32_t>(sum < 0 ? -rounded : rounded);
  }

  // The decoder reads the first n scan positions, zeroes them, and writes the
  // n2 output positions.  Positions it overwrites must not be coded again.
  auto in_first_n = [&](int index) {
    for (int j = 0; j < info.n; ++j) {
      if (static_cast<int>(scan[static_cast<std::size_t>(j)]) == index) {
        return true;
      }
    }
    return false;
  };
  for (int i = 0; i < info.n2; ++i) {
    const int index = out_pos[static_cast<std::size_t>(i)];
    if (!in_first_n(index)) levels[static_cast<std::size_t>(index)] = 0;
  }
  for (int j = 0; j < info.n; ++j) {
    levels[scan[static_cast<std::size_t>(j)]] = sec[static_cast<std::size_t>(j)];
  }
  return true;
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
  // §5.20.8.2 secondary transform (IST), luma only; 0 = off.
  int sec_tx_type = 0;
  int most_probable_stx_set = 0;
  // Syntax levels (raster order): the §5.20.7.27 full levels the writer
  // consumes.  Under TCQ this is `tcq::Result::base_levels`; otherwise it is
  // the scalar/RDOQ level.
  std::vector<std::int32_t> levels;
  // §7.14.4 dequant values when they differ from `levels` (TCQ maps the
  // syntax level to `2*q - q0`); empty means `levels` is used directly.
  std::vector<std::int32_t> dequant_levels;
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
  // §5.20.5.5 MRL syntax carried with the chosen luma mode.
  int mrl_index = 0;
  bool mrl_sec_index = false;
  // §5.20.7.9 DIP alternative to ordinary DC prediction (luma only).
  bool use_dip = false;
  int dip_mode = 0;
  bool dip_transpose = false;
};

struct UvChoice {
  bool is_cfl = false;
  intra::Mode mode = intra::Mode::kDc;
  int angle_delta = 0;
  int av2_mode = kDcPred;
  std::int64_t sse = 0;
  // §5.20.7.32 CfL signalling and the Q8 alphas the reconstruction applies.
  int cfl_index = 1;         // 0 explicit, 1 derived
  int cfl_alpha_signs = 0;
  int cfl_alpha_u = 0;       // signalled symbols (magnitude - 1)
  int cfl_alpha_v = 0;
  int cfl_alpha_q8_u = 0;
  int cfl_alpha_q8_v = 0;
  // MHCCP (cfl_index == 2): cfl_mh_dir and the §7.13.7 derived Q16 params.
  int cfl_mh_dir = 0;
  av2enc::cfl::mhccp::MultiParams mh_params_u{};
  av2enc::cfl::mhccp::MultiParams mh_params_v{};
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

struct ChromaInfo {
  bool valid = false;
  int mi_row = 0;   // luma 4x4 units
  int mi_col = 0;
  BlockSize size = BlockSize::kInvalid;  // ChromaMiSize
};

class Encoder {
 public:
  Encoder(const StillPicture& picture, const StillOptions& options,
          const dcdeltaq::QuantFrameState* delta_quant = nullptr)
      : picture_(picture), options_(options), delta_quant_(delta_quant) {
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

    // §5.18.6.2: one picture-level QM set (pic_qm_num_minus_1 = 0) with the
    // same spec-defined default matrix level for all planes and the single
    // segment.  >8x8 blocks read qm_y/u/v[0]; <=8x8 blocks read
    // SegQMLevel[plane][0].  No quantizer_matrix_obu() is emitted, so
    // QmDataPresent[] stays 0 and only the normative default tables apply.
    if (options_.qm) {
      qm_state_.using_qmatrix = true;
      qm_state_.pic_qm_num_minus_1 = 0;
      const auto level = static_cast<std::uint8_t>(options_.qm_level);
      qm_state_.qm_y[0] = level;
      qm_state_.qm_u[0] = level;
      qm_state_.qm_v[0] = level;
      for (int plane = 0; plane < 3; ++plane) {
        qm_state_.seg_qm_level[plane][0] = level;
      }
      if (options_.user_qm) {
        // §5.4.11: a real user-defined matrix for the selected level.  All-32
        // is the identity (Round2(q * 32, 5) == q), so the §7.14.4 dequant is
        // unchanged while the stream carries genuine user QM data.
        qm_state_.qm_data_present[level] = true;
        qm::fill_user_qm_constant(&qm_state_.user_qm[level], 32);
      }
    }

    geometry_.width = width_;
    geometry_.height = height_;
    geometry_.sb_size = (options.sb128 || options.sb256) ? 128 : 64;
    sb_mi_ = partition::sb_size_mi(geometry_);
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
    // §5.20.5.5: a CHROMA_PART leaf reads YMode/AngleDeltaY from the co-located
    // luma tree at the ChromaMi origin.  Keep the decided luma modes so the
    // chroma tree (coded after luma in SDP) can reuse them for the UV CDF
    // context, uv_mode_index and the derived AngleDeltaUV.
    luma_mode_.assign(static_cast<std::size_t>(mi_rows_) *
                          static_cast<std::size_t>(mi_cols_),
                      kDcPred);
    luma_angle_.assign(static_cast<std::size_t>(mi_rows_) *
                           static_cast<std::size_t>(mi_cols_),
                       0);
    uv_smooth_.assign(
        static_cast<std::size_t>(mi_rows_ / 2) *
            static_cast<std::size_t>(mi_cols_ / 2),
        0);

    for (int plane = 0; plane < 3; ++plane) {
      decoded_[plane].reset(plane == 0 ? mi_cols_ : mi_cols_ / 2,
                            plane == 0 ? mi_rows_ : mi_rows_ / 2);
    }

    if (options_.palette) {
      palette_state_.reset(mi_cols_, mi_rows_);
    }
    if (options_.ibc) {
      ibc_buffer_.reset(mi_cols_, mi_rows_);
      ibc_neighbours_.reset(mi_cols_, mi_rows_);
      ibc_config_ = ibc::IbcConfig{};
      ibc_config_.sb_size = geometry_.sb_size;
      ibc_config_.allow_global = true;
      // The M1 frame header signals the global region only; the four-buffer
      // local ring is not modelled by the writer.
      ibc_config_.allow_local = false;
      ibc_config_.max_bvp_drl_bits_minus_1 = 0;
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
    const double lambda = 0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
    const double signal_cost = lambda * 2.0;
    // Sub-8 opt-in: the 4-sample floor needs the full rect recursion depth.
    const int min_block_px = options_.sub8 ? 4 : options_.min_block_size;
    const int max_depth =
        options_.sub8
            ? std::max(options_.max_partition_depth, 8)
            : options_.max_partition_depth;

    // Shared leaf-cost policy: SSE block variance plus the writer's 64x64 leaf
    // limit (one TX_MODE_LARGEST transform block per plane).  Both solvers use
    // it so rect-only choices do not drift.
    const auto leaf_cost = [&](int mi_row, int mi_col, BlockSize size) {
      double cost = static_cast<double>(
          block_variance_cost_at(mi_row, mi_col, size));
      if (partition::block_width(size) > 64 ||
          partition::block_height(size) > 64) {
        cost += 1.0e18;
      }
      if (std::getenv("AV2ENC_STILL_SQUARE_ONLY") != nullptr &&
          partition::block_width(size) != partition::block_height(size)) {
        cost += 1.0e18;
      }
      if (std::getenv("AV2ENC_STILL_LEAF32") != nullptr &&
          (partition::block_width(size) > 32 ||
           partition::block_height(size) > 32)) {
        cost += 1.0e18;
      }
      return cost;
    };

    if (options_.sdp) {
      sdpp::Tools tools;
      tools.enable_sdp = true;
      tools.frame_is_intra = true;
      tools.single_picture_header_flag = true;
      sdpp::DecisionOptions sopt;
      // First decoder-gated configuration: every intra SDP 64x64 split carries
      // a forced PARTITION_NONE luma tree and a forced PARTITION_NONE chroma
      // tree.  The two-tree walk, PlaneStart-selected CDFs and chroma ownership
      // are exercised; searching the sub-trees is the next step.
      sopt.min_block_size_px = 64;
      sopt.max_depth = 0;
      sopt.allow_rect = false;
      sopt.allow_split = false;
      sopt.allow_3way = false;
      sopt.allow_uneven4 = false;
      sopt.allow_sdp = true;
      sopt.allow_extended_sdp = false;
      const auto sdp_cost = [&](const sdpp::NodeInfo& info) {
        sdpp::NodeCosts costs;
        if (info.tree_type == sdpp::TreeType::kChromaPart) {
          costs.leaf = 0.0;
        } else {
          costs.leaf = leaf_cost(info.mi_row, info.mi_col, info.size);
        }
        costs.none_signal = 0.0;
        costs.horz_signal = sdpp::kInfiniteCost;
        costs.vert_signal = sdpp::kInfiniteCost;
        costs.split_signal = sdpp::kInfiniteCost;
        return costs;
      };
      const sdpp::FrameDecision frame =
          sdpp::decide_frame(geometry, tools, sdp_cost, sopt);
      if (frame.status != sdpp::DecisionStatus::kOk) {
        error = "SDP partition decision failed: " + frame.error;
        return false;
      }
      if (!sdpp::validate_frame(geometry, tools, frame, error)) {
        error = "SDP partition tree invalid: " + error;
        return false;
      }
      leaves_ = frame.leaves;
      blocks_.clear();
      levels_.clear();
      for (const sdpp::Node& sb : frame.superblocks) {
        begin_superblock(sb);
        if (!collect_sdp_node(sb)) {
          error = error_;
          return false;
        }
      }
      stream->blocks = std::move(blocks_);
      stream->levels = std::move(levels_);
      return true;
    }

    if (options_.ext_partitions) {
      pe::Tools tools;
      tools.enable_ext_partitions = true;
      tools.enable_uneven_4way_partitions = true;
      pe::DecisionOptions eopt;
      // The extended families include 1/8 subsizes: HORZ_4A@32x32 is 32x4.
      // Chroma ownership for those shared-chroma cases is wired, so the ext
      // search uses the 4-sample floor.
      eopt.min_block_size_px = 4;
      eopt.max_depth = max_depth;
      eopt.allow_rect = true;
      eopt.allow_split = geometry_.sb_size == 128;
      eopt.allow_3way = true;
      eopt.allow_uneven4 = true;
      // With the extended families unavailable this order reproduces
      // partition::decide_frame's tie-breaks exactly.
      eopt.tie_break = pe::TieBreakOrder::kRectCompatOrder;
      if (options_.force_single_block) {
        eopt.max_depth = 0;
        eopt.min_block_size_px = 64;
      }
      const auto cost_fn = [&](const pe::NodeInfo& info) {
        pe::NodeCosts costs;
        costs.leaf = leaf_cost(info.mi_row, info.mi_col, info.size);
        costs.none_signal = 0.0;
        costs.split_signal = signal_cost;
        costs.horz_signal = signal_cost;
        costs.vert_signal = signal_cost;
        const int w = partition::block_width(info.size);
        const int h = partition::block_height(info.size);
        // Chroma ownership is now carried by the leaf (ChromaMi), so the
        // shared 8x16/16x8 middle block of a 3-way split and the first
        // children of a 4-way split are codable.  The spec's allowed-set
        // validation rejects every geometrically impossible candidate; these
        // thresholds only keep the unexpectedly small sizes out of the search.
        costs.horz3_signal = (h >= 32 && w >= 8) ? signal_cost
                                                  : pe::kInfiniteCost;
        costs.vert3_signal = (w >= 32 && h >= 8) ? signal_cost
                                                  : pe::kInfiniteCost;
        const double h4 = (h >= 32 && w >= 8) ? signal_cost : pe::kInfiniteCost;
        // Vertical uneven 4-way with a 4-wide 1/8 child still produces a
        // tile desync at the bottom edge (reproducer: 640x40 vertical
        // stripes); keep it at 8-wide subsizes until that is fixed.
        const double v4 = (w >= 64 && h >= 8) ? signal_cost
                                               : pe::kInfiniteCost;
        costs.horz4a_signal = h4;
        costs.horz4b_signal = h4;
        costs.vert4a_signal = v4;
        costs.vert4b_signal = v4;
        return costs;
      };
      const pe::FrameDecision frame =
          pe::decide_frame(geometry, cost_fn, tools, eopt);
      if (frame.status != pe::DecisionStatus::kOk) {
        error = "extended partition decision failed: " + frame.error;
        return false;
      }
      if (!pe::validate_frame(geometry, tools, frame, error)) {
        error = "extended partition tree invalid: " + error;
        return false;
      }
      leaves_ = frame.leaves;
      blocks_.clear();
      levels_.clear();
      for (const pe::PartitionNode& sb : frame.superblocks) {
        begin_superblock(sb);
        if (!collect_node_ext(sb)) {
          error = error_;
          return false;
        }
      }
      stream->blocks = std::move(blocks_);
      stream->levels = std::move(levels_);
      return true;
    }

    partition::DecisionOptions poptions;
    poptions.min_block_size_px = min_block_px;
    poptions.max_depth = max_depth;
    poptions.allow_rect = true;
    // PARTITION_SPLIT exists only at a 128x128 (or 256x256) node.  The 64x64
    // superblock keeps the historical rect-only search.
    poptions.allow_split = geometry_.sb_size == 128;
    if (options_.force_single_block) {
      poptions.max_depth = 0;
      poptions.min_block_size_px = 64;
    }

    const auto cost_fn = [&](const partition::PartitionNodeInfo& info) {
      partition::NodeCosts costs;
      costs.leaf = leaf_cost(info.mi_row, info.mi_col, info.size);
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
    for (const partition::PartitionNode& sb : frame.superblocks) {
      begin_superblock(sb);
      if (!collect_node(sb, ChromaInfo{})) {
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
  const qm::QmFrameState& qm_state() const { return qm_state_; }
  int leaves() const { return leaves_; }
  int ibc_blocks() const { return ibc_blocks_; }
  const std::vector<palette::PaletteResult>& palette_results() const {
    return palette_results_;
  }
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
  // §5.20.3.1 chroma bookkeeping: a node with HasChroma and chromaOffset == 0
  // captures the inherited chroma block before its children are decoded.
  // `node.has_chroma` / `node.chroma_offset` are the decoded recursion args, so
  // the ChromaMi block is simply the last such node on the path.
  static ChromaInfo chroma_step(const partition::PartitionNode& node,
                                const ChromaInfo& inherited) {
    if (node.has_chroma && !node.chroma_offset) {
      return ChromaInfo{true, node.mi_row, node.mi_col, node.size};
    }
    return inherited;
  }

  int& intra_joint(int r, int c) {
    return intra_joint_[static_cast<std::size_t>(r) *
                            static_cast<std::size_t>(mi_cols_) +
                        static_cast<std::size_t>(c)];
  }

  // Co-located luma YMode / AngleDeltaY at a 4x4 position (§5.20.5.5).  A
  // CHROMA_PART leaf has no luma mode of its own; the decoder reads the mode of
  // the luma tree block at the ChromaMi origin.
  int luma_mode_at(int r, int c) const {
    const int rr = std::clamp(r, 0, mi_rows_ - 1);
    const int cc = std::clamp(c, 0, mi_cols_ - 1);
    return luma_mode_[static_cast<std::size_t>(rr) *
                          static_cast<std::size_t>(mi_cols_) +
                      static_cast<std::size_t>(cc)];
  }
  int luma_angle_at(int r, int c) const {
    const int rr = std::clamp(r, 0, mi_rows_ - 1);
    const int cc = std::clamp(c, 0, mi_cols_ - 1);
    return luma_angle_[static_cast<std::size_t>(rr) *
                           static_cast<std::size_t>(mi_cols_) +
                       static_cast<std::size_t>(cc)];
  }
  std::uint8_t& uv_smooth(int r, int c) {
    return uv_smooth_[static_cast<std::size_t>(r) *
                          static_cast<std::size_t>(mi_cols_ / 2) +
                      static_cast<std::size_t>(c)];
  }

  // Reset the per-superblock availability state exactly like
  // clear_block_decoded_flags() and record the SB extents used by the
  // top-right/below-left counters.
  template <typename Node>
  void begin_superblock(const Node& sb) {
    for (int plane = 0; plane < 3; ++plane) {
      decoded_[plane].reset(plane == 0 ? mi_cols_ : mi_cols_ / 2,
                            plane == 0 ? mi_rows_ : mi_rows_ / 2);
    }
    sb_mi_row_ = sb.mi_row;
    sb_mi_col_ = sb.mi_col;
    sb_cols_plane_[0] = std::min(sb_mi_, mi_cols_ - sb.mi_col);
    sb_rows_plane_[0] = std::min(sb_mi_, mi_rows_ - sb.mi_row);
    sb_cols_plane_[1] = std::min(sb_mi_ / 2, mi_cols_ / 2 - sb.mi_col / 2);
    sb_rows_plane_[1] = std::min(sb_mi_ / 2, mi_rows_ / 2 - sb.mi_row / 2);
    sb_cols_plane_[2] = sb_cols_plane_[1];
    sb_rows_plane_[2] = sb_rows_plane_[1];

    // §7.14.2 get_qindex(segmentId) for the real per-SB segmentation path.
    // current_qindex_ feeds forward quantisation, dequantisation and the
    // deblock qindex; the syntax writer signals the same segment id.
    current_qindex_ = options_.base_q_idx;
    if (!options_.sb_segment.empty()) {
      const int sb_cols = (mi_cols_ + sb_mi_ - 1) / sb_mi_;
      const int sb_index =
          (sb.mi_row / sb_mi_) * sb_cols + (sb.mi_col / sb_mi_);
      if (sb_index >= 0 &&
          static_cast<std::size_t>(sb_index) < options_.sb_segment.size() &&
          options_.sb_segment[static_cast<std::size_t>(sb_index)] == 1) {
        current_qindex_ = std::clamp(
            static_cast<int>(options_.base_q_idx) + options_.per_segment_q_delta,
            1, 255);
      }
    }
    quant_step_ = rc::kQlookupQ0To255[current_qindex_];
  }

  // §5.20.5.7 per-SB segment id for a leaf at (mi_row, mi_col).  All three
  // partition collectors (rect, ext, SDP) must signal the same value that
  // begin_superblock() used for forward quant/dequant/deblock.
  std::uint8_t sb_segment_id(int mi_row, int mi_col) const {
    if (options_.sb_segment.empty()) return 0;
    const int sb_cols = (mi_cols_ + sb_mi_ - 1) / sb_mi_;
    const int sb_index = (mi_row / sb_mi_) * sb_cols + (mi_col / sb_mi_);
    if (sb_index < 0 ||
        static_cast<std::size_t>(sb_index) >= options_.sb_segment.size()) {
      return 0;
    }
    return options_.sb_segment[static_cast<std::size_t>(sb_index)];
  }

  // Append this node (and its whole subtree) to the preorder syntax stream.
  // No syntax is written here: the stream is handed to
  // syntax::assemble_still_frame() once the frame is fully decided.
  bool collect_node(const partition::PartitionNode& node,
                    const ChromaInfo& inherited_chroma) {
    const ChromaInfo chroma = chroma_step(node, inherited_chroma);
    const std::size_t index = blocks_.size();
    blocks_.emplace_back();
    syntax::BlockSyntax& entry = blocks_.back();
    entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
    entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
    entry.bsize = static_cast<std::uint8_t>(node.size);
    entry.partition = static_cast<std::uint8_t>(node.partition);
    std::uint16_t mask = 0;
    for (int p = 0; p < partition::kPartitionTypeCount; ++p) {
      if (node.allowed.allowed[p]) mask = static_cast<std::uint16_t>(mask | (1U << p));
    }
    entry.allowed_mask = mask;
    if (node.forced) entry.flags |= syntax::kFlagForced;
    if (node.depth == 0) entry.flags |= syntax::kFlagSbRoot;
    entry.flags |= syntax::kFlagHasChromaInfo;
    entry.has_chroma = node.has_chroma ? 1 : 0;
    entry.chroma_offset = node.chroma_offset ? 1 : 0;
    entry.chroma_bsize = static_cast<std::uint8_t>(chroma.size);
    entry.chroma_mi_row = static_cast<std::uint16_t>(chroma.mi_row);
    entry.chroma_mi_col = static_cast<std::uint16_t>(chroma.mi_col);
    entry.segment_id = sb_segment_id(node.mi_row, node.mi_col);

    if (trace_) {
      std::cerr << "m1 node r=" << node.mi_row << " c=" << node.mi_col
                << " size=" << partition::block_size_name(node.size)
                << " part=" << static_cast<int>(node.partition)
                << " forced=" << (node.forced ? 1 : 0) << " allowed="
                << node.allowed.allowed[0] << node.allowed.allowed[1]
                << node.allowed.allowed[2] << node.allowed.allowed[3] << "\n";
    }

    if (node.partition == PartitionType::kNone) {
      if (!encode_block(node.mi_row, node.mi_col, node.size, index,
                        node.has_chroma, chroma)) {
        return false;
      }
    } else {
      for (const partition::PartitionNode& child : node.children) {
        if (!collect_node(child, chroma)) return false;
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

  // Extended-partition variant of collect_node().  Chroma ownership comes from
  // pe::PartitionNode::has_chroma/chroma_offset/chroma, so a shared 8x16/16x8
  // middle block (HORZ_3@8x32 / VERT_3@32x8) and the first children of a 4-way
  // split are handled exactly like sub-8 leaves.
  bool collect_node_ext(const pe::PartitionNode& node) {
    const std::size_t index = blocks_.size();
    blocks_.emplace_back();
    syntax::BlockSyntax& entry = blocks_.back();
    entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
    entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
    entry.bsize = static_cast<std::uint8_t>(node.size);
    entry.partition = static_cast<std::uint8_t>(node.partition);
    std::uint16_t mask = 0;
    for (int p = 0; p < pe::kPartitionCount; ++p) {
      if (node.allowed.allowed[p]) mask = static_cast<std::uint16_t>(mask | (1U << p));
    }
    entry.allowed_mask = mask;
    if (node.forced) entry.flags |= syntax::kFlagForced;
    if (node.depth == 0) entry.flags |= syntax::kFlagSbRoot;
    entry.flags |= syntax::kFlagHasChromaInfo;
    entry.has_chroma = node.has_chroma ? 1 : 0;
    entry.chroma_offset = node.chroma_offset ? 1 : 0;
    entry.chroma_bsize = static_cast<std::uint8_t>(node.chroma.size);
    entry.chroma_mi_row = static_cast<std::uint16_t>(node.chroma.mi_row);
    entry.chroma_mi_col = static_cast<std::uint16_t>(node.chroma.mi_col);
    // §5.20.5.7 intra_segment_id(): the ext tree must code the same per-SB
    // segment as collect_node(); otherwise quant/dequant/deblock use segment 1
    // locally while the bitstream says segment 0.
    entry.segment_id = sb_segment_id(node.mi_row, node.mi_col);

    if (trace_) {
      std::cerr << "m1 ext node r=" << node.mi_row << " c=" << node.mi_col
                << " size=" << partition::block_size_name(node.size)
                << " part=" << static_cast<int>(node.partition)
                << " forced=" << (node.forced ? 1 : 0) << "\n";
    }

    if (node.partition == pe::PartitionType::kNone) {
      const ChromaInfo chroma{
          node.has_chroma, node.chroma.mi_row, node.chroma.mi_col,
          node.chroma.size};
      if (!encode_block(node.mi_row, node.mi_col, node.size, index,
                        node.has_chroma, chroma)) {
        return false;
      }
    } else {
      for (const pe::PartitionNode& child : node.children) {
        if (!collect_node_ext(child)) return false;
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

  // Append one SDP luma/chroma tree (preorder).  The sdp split node codes no
  // partition syntax and carries the two 64x64 tree roots as children.
  bool collect_sdp_node(const sdpp::Node& node) {
    const std::size_t index = blocks_.size();
    blocks_.emplace_back();
    syntax::BlockSyntax& entry = blocks_.back();
    entry.mi_row = static_cast<std::uint16_t>(node.mi_row);
    entry.mi_col = static_cast<std::uint16_t>(node.mi_col);
    entry.bsize = static_cast<std::uint8_t>(node.size);
    entry.partition = static_cast<std::uint8_t>(node.partition);
    entry.tree_type = static_cast<std::uint8_t>(node.tree_type);
    std::uint16_t mask = 0;
    for (int p = 0; p < sdpp::kPartitionCount; ++p) {
      if (node.allowed.allowed[p]) {
        mask = static_cast<std::uint16_t>(mask | (1U << p));
      }
    }
    entry.allowed_mask = mask;
    if (node.forced) entry.flags |= syntax::kFlagForced;
    if (node.depth == 0) entry.flags |= syntax::kFlagSbRoot;
    if (node.sdp_split) entry.flags |= syntax::kFlagSdpSplit;
    // §5.20.5.7: carry the per-SB segment id into the two-tree walk too.
    entry.segment_id = sb_segment_id(node.mi_row, node.mi_col);

    if (trace_) {
      std::cerr << "m1 sdp node r=" << node.mi_row << " c=" << node.mi_col
                << " size=" << partition::block_size_name(node.size)
                << " tree=" << sdpp::tree_type_name(node.tree_type)
                << " part=" << static_cast<int>(node.partition)
                << " sdp_split=" << (node.sdp_split ? 1 : 0)
                << " forced=" << (node.forced ? 1 : 0) << "\n";
    }

    if (node.sdp_split) {
      if (node.children.size() != 2) {
        error_ = "SDP split node does not have exactly two tree children";
        return false;
      }
      for (const sdpp::Node& child : node.children) {
        if (!collect_sdp_node(child)) return false;
      }
    } else if (node.partition == sdpp::PartitionType::kNone) {
      const bool has_chroma = node.codes_chroma;
      const ChromaInfo chroma{node.has_chroma, node.chroma.mi_row,
                              node.chroma.mi_col, node.chroma.size};
      if (!encode_block(node.mi_row, node.mi_col, node.size, index,
                        has_chroma, chroma,
                        static_cast<int>(node.tree_type))) {
        return false;
      }
    } else {
      for (const sdpp::Node& child : node.children) {
        if (!collect_sdp_node(child)) return false;
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
    const int border_w = std::min(2 * (sb_mi_ >> sub), plane_cols - sb_c0);
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

  // `block_mi_row`/`block_mi_col` are the coding-block origin in 4x4 units.
  // Transform-part callers pass it so the §7.13.2.15/.16 smooth-neighbour
  // flags come from the adjacent coding block, as the decoder does; the
  // default (-1) derives it from the prediction origin.
  bool predict_block(int plane, int x0, int y0, int w, int h,
                     intra::Mode mode, int angle_delta,
                     std::vector<std::uint8_t>& out,
                     bool uv_is_cfl = false, int mrl_index = 0,
                     bool mrl_sec_index = false, bool use_dip = false,
                     int dip_mode = 0, bool dip_transpose = false,
                     int block_mi_row = -1, int block_mi_col = -1) {
    if (mrl_index > 0) {
      // MRL is luma-only (§7.13.2.1); chroma always predicts line 0.
      if (plane != 0) {
        error_ = "MRL is not signalable for chroma";
        return false;
      }
      return predict_luma_mrl(x0, y0, w, h, mode, angle_delta, mrl_index,
                              mrl_sec_index, out, block_mi_row,
                              block_mi_col);
    }
    if (use_dip) {
      if (plane != 0 || mode != intra::Mode::kDc) {
        error_ = "DIP is luma DC only";
        return false;
      }
      return predict_luma_dip(x0, y0, w, h, dip_mode, dip_transpose, out);
    }
    intra::IntraEdges edges;
    if (!prepare_edges(plane, x0, y0, w, h, edges)) {
      error_ = "prepare_edges failed";
      return false;
    }
    bool above_smooth = false;
    bool left_smooth = false;
    // §7.13.2.15/.16: the smooth flags are read at the coding-block origin
    // (MiRow/MiCol), not at the transform-part origin.
    const int smooth_row = block_mi_row >= 0 ? block_mi_row : (y0 >> 2);
    const int smooth_col = block_mi_col >= 0 ? block_mi_col : (x0 >> 2);
    neighborhood_smooth(plane, smooth_row, smooth_col, w >> 2, h >> 2,
                        above_smooth, left_smooth);
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    if (options_.enable_ibp) {
      // §7.13.2.7-.12: IBP changes DC (post-filter), directional edge
      // filtering and the two-angle blend.  ibp::predict is the drop-in
      // replacement for intra::predict_intra and never mutates the edges.
      ibp::PredictParams params;
      params.mode = mode;
      params.angle_delta = angle_delta;
      params.enable_ibp = true;
      params.enable_edge_filter = edge_filter_enabled();
      params.above_neighbor_is_smooth = above_smooth;
      params.left_neighbor_is_smooth = left_smooth;
      params.use_dip = false;
      params.uv_is_cfl = uv_is_cfl;
      if (ibp::predict(edges, params, out.data(),
                       static_cast<std::size_t>(w)) !=
          intra::IntraStatus::kOk) {
        error_ = "ibp::predict failed";
        return false;
      }
      return true;
    }
    intra::PredictParams params;
    params.mode = mode;
    params.angle_delta = angle_delta;
    params.enable_edge_filter = edge_filter_enabled();
    params.above_neighbor_is_smooth = above_smooth;
    params.left_neighbor_is_smooth = left_smooth;
    if (intra::predict_intra(edges, params, out.data(),
                             static_cast<std::size_t>(w)) !=
        intra::IntraStatus::kOk) {
      error_ = "predict_intra failed";
      return false;
    }
    return true;
  }

  // §5.20.7.9: DIP is an alternative to ordinary DC at the same YMode, not a
  // new mode.  Evaluates the normative candidate set and keeps it only when it
  // beats the incumbent prediction by SSE.
  bool evaluate_luma_dip(int x0, int y0, int w, int h, ModeChoice& best) {
    if (!options_.enable_dip || !dip::dip_block_available(w, h)) return true;
    const dip::Block block = make_dip_block(x0, y0, w, h);
    dip::Edges dip_edges;
    if (dip::prepare_edges(block, 8, dip_edges) != dip::Status::kOk) {
      error_ = "dip::prepare_edges failed";
      return false;
    }
    std::vector<std::uint8_t> dip_src(
        static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (int yy = 0; yy < h; ++yy) {
      for (int xx = 0; xx < w; ++xx) {
        dip_src[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(xx)] =
            source_sample(0, x0 + xx, y0 + yy);
      }
    }
    dip::DecisionParams dip_params;
    dip_params.enable_dip = true;
    dip_params.y_mode = kDcPred;
    dip_params.palette_size_y = 0;
    dip_params.tree_is_chroma_part = false;
    dip_params.metric = dip::CostMetric::kSse;
    dip::Decision dip_decision;
    if (dip::decide(dip_edges, dip_src.data(), static_cast<std::size_t>(w),
                    dip_params, dip_decision) != dip::Status::kOk ||
        !dip_decision.available) {
      error_ = "dip::decide failed";
      return false;
    }
    if (static_cast<std::int64_t>(dip_decision.cost.sse) < best.sse) {
      best.sse = static_cast<std::int64_t>(dip_decision.cost.sse);
      best.mode = intra::Mode::kDc;
      best.angle_delta = 0;
      best.av2_mode = kDcPred;
      best.mrl_index = 0;
      best.mrl_sec_index = false;
      best.use_dip = true;
      best.dip_mode = dip_decision.mode;
      best.dip_transpose = dip_decision.transpose;
    }
    return true;
  }

  // §5.20.7.9 DIP prediction for a luma DC block.
  dip::Block make_dip_block(int x0, int y0, int w, int h) {
    dip::Block block;
    block.recon =
        dip::PlaneView::from_u8(recon_[0].samples.data(), recon_[0].width);
    block.x = x0;
    block.y = y0;
    block.w = w;
    block.h = h;
    block.max_x = mi_cols_ * 4 - 1;
    block.max_y = mi_rows_ * 4 - 1;
    block.have_left = x0 > 0;
    block.have_above = y0 > 0;
    block.num4_above_right = count_top_right(0, x0 >> 2, y0 >> 2, w >> 2);
    block.num4_below_left = count_bottom_left(0, x0 >> 2, y0 >> 2, h >> 2);
    return block;
  }

  bool predict_luma_dip(int x0, int y0, int w, int h, int dip_mode,
                        bool dip_transpose, std::vector<std::uint8_t>& out) {
    const dip::Block block = make_dip_block(x0, y0, w, h);
    dip::Params params;
    params.mode = dip_mode;
    params.transpose = dip_transpose;
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    if (dip::predict_block8(block, 8, params, out.data(),
                            static_cast<std::size_t>(w)) !=
        dip::Status::kOk) {
      error_ = "dip::predict_block8 failed";
      return false;
    }
    return true;
  }

  // §7.13.2.1/.7/.8 MRL prediction for a directional luma candidate.
  // mrl::prepare_edges needs the extra reference lines, so it cannot reuse the
  // w+h-sample IntraEdges prepared for MrlIndex = 0.
  bool predict_luma_mrl(int x0, int y0, int w, int h, intra::Mode mode,
                        int angle_delta, int mrl_index, bool mrl_sec_index,
                        std::vector<std::uint8_t>& out, int block_mi_row = -1,
                        int block_mi_col = -1) {
    bool above_smooth = false;
    bool left_smooth = false;
    const int smooth_row = block_mi_row >= 0 ? block_mi_row : (y0 >> 2);
    const int smooth_col = block_mi_col >= 0 ? block_mi_col : (x0 >> 2);
    neighborhood_smooth(0, smooth_row, smooth_col, w >> 2, h >> 2,
                        above_smooth, left_smooth);
    mrl::Params params;
    if (!mrl_stream::make_predict_params(
            av2_y_mode(mode), angle_delta, mrl_index, mrl_sec_index,
            edge_filter_enabled(), above_smooth, left_smooth, &params, w, h)) {
      error_ = "MRL prediction parameters rejected";
      return false;
    }
    mrl::Block block;
    block.recon =
        intra::PlaneView{recon_[0].samples.data(), recon_[0].width};
    block.x = x0;
    block.y = y0;
    block.w = w;
    block.h = h;
    block.max_x = mi_cols_ * 4 - 1;
    block.max_y = mi_rows_ * 4 - 1;
    block.have_left = x0 > 0;
    block.have_above = y0 > 0;
    block.num4_above_right = count_top_right(0, x0 >> 2, y0 >> 2, w >> 2);
    block.num4_below_left = count_bottom_left(0, x0 >> 2, y0 >> 2, h >> 2);
    block.plane = 0;
    // §7.13.2.1: sbHeight = Block_Height[SbSize].  An intra frame uses the
    // frame SbSize (128x128 under --still-sb128/sb256, 64x64 otherwise), so
    // the MRL above-row reset must follow geometry_, not a fixed 64.
    block.sb_height = geometry_.sb_size;
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    if (mrl::predict_block(block, 8, params, out.data(),
                           static_cast<std::size_t>(w)) !=
        mrl::MrlStatus::kOk) {
      error_ = "mrl::predict_block failed";
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
      if (options_.enable_dip || options_.palette) {
        // Both the DIP and the palette candidates are scored against the
        // ordinary DC prediction the encoder would otherwise code, so its SSE
        // must be real here (the value is unused otherwise).
        std::vector<std::uint8_t> dc_pred;
        if (!predict_block(0, x0, y0, w, h, intra::Mode::kDc, 0, dc_pred)) {
          return false;
        }
        best.sse = sse_against_source(0, x0, y0, w, h, dc_pred);
        if (!evaluate_luma_dip(x0, y0, w, h, best)) return false;
      }
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
    const bool mrl_search = options_.enable_mrl;
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
      if (options_.enable_ibp) {
        ibp::PredictParams ibp_params;
        ibp_params.mode = params.mode;
        ibp_params.angle_delta = params.angle_delta;
        ibp_params.enable_ibp = true;
        ibp_params.enable_edge_filter = params.enable_edge_filter;
        ibp_params.above_neighbor_is_smooth = above_smooth;
        ibp_params.left_neighbor_is_smooth = left_smooth;
        if (ibp::predict(edges, ibp_params, pred.data(),
                         static_cast<std::size_t>(w)) !=
            intra::IntraStatus::kOk) {
          error_ = "luma ibp::predict failed";
          return false;
        }
      } else if (intra::predict_intra(edges, params, pred.data(),
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
        best.mrl_index = 0;
        best.mrl_sec_index = false;
      }
      // §5.20.5.5 MRL: only a directional luma mode can signal mrl_index, so
      // the extra search runs on those candidates only.  mrl_index == 0 is the
      // candidate just evaluated above.
      if (mrl_search && intra::is_directional(params.mode)) {
        const bool sec_allowed = !(w == 4 && h == 4);
        for (int mrl_index = 1; mrl_index <= mrl::kMaxMrlIndex; ++mrl_index) {
          for (int sec = 0; sec <= (sec_allowed ? 1 : 0); ++sec) {
            std::vector<std::uint8_t> mrl_pred;
            if (!predict_luma_mrl(x0, y0, w, h, params.mode,
                                  params.angle_delta, mrl_index, sec != 0,
                                  mrl_pred)) {
              return false;
            }
            const std::int64_t mrl_sse =
                sse_against_source(0, x0, y0, w, h, mrl_pred);
            if (mrl_sse < best.sse) {
              best.sse = mrl_sse;
              best.mode = params.mode;
              best.angle_delta = params.angle_delta;
              best.av2_mode = av2_y_mode(params.mode);
              best.mrl_index = mrl_index;
              best.mrl_sec_index = sec != 0;
            }
          }
        }
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
    // §5.20.7.9: DIP alternative to DC, compared against the best ordinary
    // prediction by SSE.
    if (!base_only && !evaluate_luma_dip(x0, y0, w, h, best)) return false;
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

  // §7.14.2/§7.14.4 quantizer for one adjusted-raster coefficient: the DC
  // quantizer at (0,0) and the AC quantizer elsewhere.  Without delta-Q both
  // are the frame qlookup step.
  int coefficient_q(int plane, int index) const {
    if (delta_quant_ == nullptr) return quant_step_;
    const dcdeltaq::PlaneQuant& q = delta_quant_->plane[plane];
    return index == 0 ? q.dc : q.ac;
  }

  // §7.14.4 dequant + optional IST inverse + inverse transform with the
  // frame's qlookup step.  With QM enabled the per-coefficient
  // q2 = Round2(q * m, 5) replaces the scalar 528 so the encoder
  // reconstruction matches what the decoder derives.
  bool dequant_inverse_qindex(const std::int32_t* levels,
                              const txv::TxParams& params, int plane, int eob,
                              int y_mode, int y_angle_delta, int sec_tx_type,
                              int stx_set, int mrl_index,
                              std::int32_t* dequant,
                              std::int16_t* residual) {
    const int n = params.w * params.h;
    // §7.14.4: TCQ doubles dqDenom for luma 2-D transform classes.  IDTX is
    // TX_CLASS_2D too (qm_core.h's `< IDTX` predicate is only for useQm).
    const bool tcq_shift = options_.tcq && plane == 0 && params.tx_type <= 9;
    const int dq_denom =
        tcq_shift ? (params.dq_denom << 1) : params.dq_denom;
    if (options_.delta_q) {
      for (int i = 0; i < n; ++i) {
        const std::int32_t qc = levels[i];
        const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
        const std::int64_t high =
            (aq * coefficient_q(plane, i)) & 0xFFFFFF;
        const std::int32_t dq = txv::round2(static_cast<std::int32_t>(high),
                                            txv::kQuantTableBits) /
                                dq_denom;
        const std::int32_t value = qc < 0 ? -dq : dq;
        dequant[i] = std::max(-(1 << 15), std::min((1 << 15) - 1, value));
      }
    } else if (options_.qm) {
      qm::QmDequantParams qp;
      qp.state = &qm_state_;
      qp.tx_size = params.tx_size;
      qp.plane = plane;
      qp.tx_type = params.tx_type;
      qp.dc_quant = quant_step_;
      qp.ac_quant = quant_step_;
      qp.bit_depth = 8;
      if (!qm::dequantize_block(levels, qp, dequant)) {
        error_ = "qm dequantize failed";
        return false;
      }
    } else {
      for (int i = 0; i < n; ++i) {
        const std::int32_t qc = levels[i];
        const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
        const std::int64_t high = (aq * quant_step_) & 0xFFFFFF;
        const std::int32_t dq = txv::round2(static_cast<std::int32_t>(high),
                                            txv::kQuantTableBits) /
                                dq_denom;
        const std::int32_t value = qc < 0 ? -dq : dq;
        dequant[i] = std::max(-(1 << 15), std::min((1 << 15) - 1, value));
      }
    }

    // §7.15.3: the decoder applies the secondary inverse to the dequantized
    // luma coefficients before the primary 2-D inverse.
    if (plane == 0 && sec_tx_type != 0) {
      txsec::SecondaryTransformConfig cfg;
      cfg.enable_intra_ist = true;
      txsec::SecondarySyntax stx;
      stx.plane_tx_type = static_cast<txv::TxType>(params.tx_type);
      stx.coded_w = params.coded_w;
      stx.coded_h = params.coded_h;
      stx.y_mode = y_mode;
      stx.angle_delta = y_angle_delta;
      stx.mrl_index = mrl_index;
      stx.most_probable_stx_set = stx_set;
      stx.sec_tx_type = sec_tx_type;
      stx.is_inter = false;
      stx.lossless = false;
      stx.bit_depth = 8;
      const txsec::SecondaryStatus status = txsec::apply_inverse_secondary(
          params, plane, eob, stx, cfg, dequant,
          static_cast<std::size_t>(n));
      if (status != txsec::SecondaryStatus::kApplied) {
        error_ = "secondary transform inverse failed";
        return false;
      }
    }

    txv::inverse_block(dequant, params, residual);
    return true;
  }

  bool compute_tx(int plane, int x0, int y0, int cw, int ch,
                  const std::vector<std::uint8_t>& pred, TxPlan& plan,
                  bool choose_tx_type = false, int y_mode = 0,
                  int y_angle_delta = 0, int mrl_index = 0) {
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
      int mapped_mode = mrl_stream::mapped_intra_mode_for_mrl(
          y_mode, y_angle_delta, mrl_index, cw, ch);
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
        // enable_idtx_intra = 0 forbids every identity 1D transform.  With the
        // flag on, only the pure 2-D identity type (IDTX) is enabled; the
        // one-axis types (V/H_DCT, V/H_ADST, V/H_FLIPADST) still need the
        // class-specific scans and contexts and remain out of scope.
        txv::Tx1dType row_type = txv::Tx1dType::kDct;
        txv::Tx1dType col_type = txv::Tx1dType::kDct;
        txv::tx_1d_types(candidate.type, &row_type, &col_type);
        const bool is_idtx =
            row_type == txv::Tx1dType::kIdt && col_type == txv::Tx1dType::kIdt;
        const bool mixed_identity =
            (row_type == txv::Tx1dType::kIdt) !=
            (col_type == txv::Tx1dType::kIdt);
        if (mixed_identity) continue;
        if (is_idtx && !options_.idtx) continue;
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
      plan.dequant_levels.clear();
      if (options_.delta_q) {
        // DC/AC delta-Q: scale the scalar qf by qindex_step / q_coefficient.
        txv::forward_block(residual_adj.data(), p, coeff.data());
        for (std::size_t i = 0; i < coeff.size(); ++i) {
          const double factor =
              static_cast<double>(p.quant_factor) *
              static_cast<double>(quant_step_) /
              static_cast<double>(coefficient_q(
                  plane, static_cast<int>(i)));
          plan.levels[i] = static_cast<std::int32_t>(
              std::lround(coeff[i] * factor));
        }
      } else if (options_.qm) {
        // QM changes the per-coefficient quantizer to q2 = Round2(q * m, 5).
        // The forward target scales the scalar qf by q / q2 so the returned
        // levels reconstruct to the same residual under QM dequant.
        txv::forward_block(residual_adj.data(), p, coeff.data());
        qm::QmDequantParams qp;
        qp.state = &qm_state_;
        qp.tx_size = p.tx_size;
        qp.plane = plane;
        qp.tx_type = p.tx_type;
        qp.dc_quant = quant_step_;
        qp.ac_quant = quant_step_;
        qp.bit_depth = 8;
        std::vector<std::int32_t> q2(coeff.size());
        if (!qm::build_q2_matrix(qp, q2.data())) {
          error_ = "qm q2 matrix build failed";
          return false;
        }
        for (std::size_t i = 0; i < coeff.size(); ++i) {
          const double factor =
              static_cast<double>(p.quant_factor) *
              static_cast<double>(quant_step_) /
              static_cast<double>(q2[i]);
          plan.levels[i] = static_cast<std::int32_t>(
              std::lround(coeff[i] * factor));
        }
      } else {
        txv::forward_transform(residual_adj.data(), p, coeff.data(),
                               plan.levels.data());
      }

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

      // §5.20.7.27 trellis-coded quantisation.  Luma TX_CLASS_2D only.  The
      // DP returns the syntax base levels (what coeff_base/br and the
      // read_quant escape code) and the mapped Quant magnitudes that §7.14.4
      // dequantises after the extra dqDenom shift.
      if (options_.tcq && plane == 0 && p.tx_type <= 9) {
        std::vector<av2enc::tcq::Coefficient> input(
            static_cast<std::size_t>(area));
        std::vector<std::int32_t> reference(static_cast<std::size_t>(area));
        for (int c = 0; c < area; ++c) {
          const int pos = scan[static_cast<std::size_t>(c)];
          const int row = pos / p.w;
          const int col = pos % p.w;
          input[static_cast<std::size_t>(c)].original = coeff[pos];
          input[static_cast<std::size_t>(c)].level_context = 0;
          input[static_cast<std::size_t>(c)].pre_max =
              (row + col) < 4 ? 8 : 6;
          // Frozen neighbour/escape reference: the scalar level at half the
          // syntax step, exactly like the component's RD harness.
          const std::int32_t scalar = plan.levels[pos];
          const std::int32_t magnitude = std::abs(scalar);
          const std::int32_t base = (magnitude + 1) / 2;
          reference[static_cast<std::size_t>(c)] =
              scalar < 0 ? -base : base;
        }
        av2enc::tcq::av2::BlockInfo info;
        info.plane = 0;
        info.params = p;
        info.txb_skip_context = 0;
        info.dc_sign_context = 0;
        info.qctx = qctx_slice_;
        info.is_inter = false;
        info.tx_class_2d = true;
        av2enc::tcq::av2::Model model;
        std::string tcq_error;
        if (!model.init(info, reference, &tcq_error)) {
          error_ = "tcq model init failed: " + tcq_error;
          return false;
        }
        av2enc::tcq::Config tcq_config;
        tcq_config.quant_step = 1.0 / p.quant_factor;
        tcq_config.lambda = av2enc::rdo::lambda_sse_from_quant_step(
            tcq_config.quant_step, options_.tcq_sse_scale);
        tcq_config.max_coefficients = area;
        av2enc::tcq::Result tcq_result;
        if (!av2enc::tcq::quantize(input, tcq_config, model.rate_model(),
                                   tcq_result, &tcq_error)) {
          error_ = "tcq quantize failed: " + tcq_error;
          return false;
        }
        std::fill(plan.levels.begin(), plan.levels.end(), 0);
        plan.dequant_levels.assign(static_cast<std::size_t>(area), 0);
        for (int c = 0; c < tcq_result.eob; ++c) {
          const int pos = scan[static_cast<std::size_t>(c)];
          plan.levels[pos] = tcq_result.base_levels[static_cast<std::size_t>(c)];
          plan.dequant_levels[pos] =
              tcq_result.quant_levels[static_cast<std::size_t>(c)];
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

    // §5.20.8.2 / §7.15.3 intra IST.  The spec defines only the decoder
    // inverse; the encoder uses the component's analysis projection on the
    // quantised primary coefficients (documented as non-normative) and picks
    // the candidate that minimises reconstructed SSE + lambda * symbol bits.
    if (options_.ist && plane == 0) {
      txsec::SecondaryTransformConfig ist_cfg;
      ist_cfg.enable_intra_ist = true;
      txsec::SecondarySyntax stx;
      stx.plane_tx_type = tx_type;
      stx.coded_w = params.coded_w;
      stx.coded_h = params.coded_h;
      stx.y_mode = y_mode;
      stx.angle_delta = y_angle_delta;
      stx.mrl_index = mrl_index;
      stx.is_inter = false;
      stx.lossless = false;
      stx.bit_depth = 8;
      if (plan.eob >= 2 &&
          txsec::secondary_syntax_applicable(0, plan.eob, stx, ist_cfg)) {
        txsec::SecondaryCandidate candidates[32];
        const int count = txsec::enumerate_secondary_candidates(
            stx, ist_cfg, plan.eob, candidates, 32);
        if (count > 0) {
          const int area = params.w * params.h;
          std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
          syntax::build_scan(params.w, params.h, scan);
          const auto eob_of = [&](const std::vector<std::int32_t>& lv) {
            for (int c = area - 1; c >= 0; --c) {
              if (lv[scan[static_cast<std::size_t>(c)]] != 0) return c + 1;
            }
            return 0;
          };
          const auto distortion =
              [&](const std::vector<std::int32_t>& lv, int ev, int sec,
                  int set, bool* ok) -> double {
            std::vector<std::int32_t> dq(static_cast<std::size_t>(area));
            std::vector<std::int16_t> rr(static_cast<std::size_t>(coded_pels));
            if (!dequant_inverse_qindex(lv.data(), params, 0, ev, y_mode,
                                        y_angle_delta, sec, set, mrl_index,
                                        dq.data(), rr.data())) {
              *ok = false;
              return 0.0;
            }
            double sse = 0.0;
            for (int y = 0; y < ch; ++y) {
              for (int x = 0; x < cw; ++x) {
                const int p = pred[static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(cw) +
                                   static_cast<std::size_t>(x)];
                const int r = rr[static_cast<std::size_t>(y) *
                                     static_cast<std::size_t>(cw) +
                                 static_cast<std::size_t>(x)];
                const int v = clip1(p + r);
                const double d =
                    static_cast<double>(source_sample(plane, x0 + x, y0 + y)) -
                    static_cast<double>(v);
                sse += d * d;
              }
            }
            return sse;
          };
          bool ok = true;
          double best_cost = distortion(plan.levels, plan.eob, 0, 0, &ok);
          if (!ok) return false;
          double lambda = av2enc::rdo::lambda_sse_from_quant_step(
              1.0 / params.quant_factor, options_.ist_sse_scale);
          std::vector<std::int32_t> best_levels;
          int best_eob = plan.eob;
          int best_sec = 0;
          int best_set = 0;
          for (int i = 0; i < count; ++i) {
            const txsec::SecondaryCandidate& candidate = candidates[i];
            std::vector<std::int32_t> lv = plan.levels;
            if (!project_ist_levels(params, tx_type, candidate.kernel,
                                    candidate.sec_tx_type,
                                    candidate.transpose, lv)) {
              continue;
            }
            const int ev = eob_of(lv);
            if (ev < 2) continue;
            txsec::SecondarySyntax cs = stx;
            cs.sec_tx_type = candidate.sec_tx_type;
            cs.most_probable_stx_set = candidate.most_probable_stx_set;
            if (!txsec::secondary_syntax_valid(0, ev, cs, ist_cfg)) continue;
            ok = true;
            const double sse =
                distortion(lv, ev, candidate.sec_tx_type,
                           candidate.most_probable_stx_set, &ok);
            if (!ok) continue;
            const double bits =
                txsec::sec_tx_type_bits(candidate.sec_tx_type, params.tx_size,
                                        false) +
                txsec::most_probable_stx_set_bits(
                    candidate.most_probable_stx_set, tx_type, cw, ch);
            const double cost = sse + lambda * bits;
            if (cost < best_cost) {
              best_cost = cost;
              best_levels = std::move(lv);
              best_eob = ev;
              best_sec = candidate.sec_tx_type;
              best_set = candidate.most_probable_stx_set;
            }
          }
          if (best_sec != 0) {
            plan.levels = std::move(best_levels);
            plan.eob = best_eob;
            plan.sec_tx_type = best_sec;
            plan.most_probable_stx_set = best_set;
          }
        }
      }
    }

    std::vector<std::int32_t> dequant(
        static_cast<std::size_t>(params.w) * static_cast<std::size_t>(params.h));
    std::vector<std::int16_t> recon_residual(
        static_cast<std::size_t>(coded_pels));
    const std::int32_t* recon_levels =
        plan.dequant_levels.empty() ? plan.levels.data()
                                    : plan.dequant_levels.data();
    if (!dequant_inverse_qindex(recon_levels, params, plane, plan.eob,
                                y_mode, y_angle_delta, plan.sec_tx_type,
                                plan.most_probable_stx_set, mrl_index,
                                dequant.data(), recon_residual.data())) {
      return false;
    }
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

  // §5.4.8 enable_cctx / §7.14.6.  Chroma U/V are quantised jointly: each of
  // the seven rotation types is enumerated, and the exact decoder
  // reconstruction (integer §7.14.4 dequant, §7.14.6 rotation, inverse
  // transform) picks the winner by SSE + lambda * cctx_type bits.
  // ---------------------------------------------------------------------
  // §5.20.6 TX_MODE_SELECT luma partition
  // ---------------------------------------------------------------------
  // Snapshot of the prediction/reconstruction state that a partition trial
  // mutates: the luma recon region and the plane-0 BlockDecoded map.
  struct LumaSnapshot {
    int x0 = 0;
    int y0 = 0;
    int w = 0;
    int h = 0;
    std::vector<std::uint8_t> recon;
    int c0 = 0;
    int r0 = 0;
    int cw4 = 0;
    int ch4 = 0;
    std::vector<std::uint8_t> decoded;
  };

  LumaSnapshot take_luma_snapshot(int x0, int y0, int w, int h) const {
    LumaSnapshot s;
    s.x0 = x0;
    s.y0 = y0;
    s.w = w;
    s.h = h;
    s.recon.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int px = x0 + x;
        const int py = y0 + y;
        s.recon[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) +
                static_cast<std::size_t>(x)] =
            (px < recon_[0].width && py < recon_[0].height)
                ? recon_[0].at(px, py)
                : 0;
      }
    }
    s.c0 = x0 / 4;
    s.r0 = y0 / 4;
    s.cw4 = (w + 3) / 4;
    s.ch4 = (h + 3) / 4;
    s.decoded.resize(static_cast<std::size_t>(s.cw4) *
                     static_cast<std::size_t>(s.ch4));
    for (int y = 0; y < s.ch4; ++y) {
      for (int x = 0; x < s.cw4; ++x) {
        s.decoded[static_cast<std::size_t>(y) *
                      static_cast<std::size_t>(s.cw4) +
                  static_cast<std::size_t>(x)] =
            decoded_[0].at(s.c0 + x, s.r0 + y) ? 1 : 0;
      }
    }
    return s;
  }

  void restore_luma_snapshot(const LumaSnapshot& s) {
    for (int y = 0; y < s.h; ++y) {
      for (int x = 0; x < s.w; ++x) {
        const int px = s.x0 + x;
        const int py = s.y0 + y;
        if (px >= recon_[0].width || py >= recon_[0].height) continue;
        recon_[0].at(px, py) =
            s.recon[static_cast<std::size_t>(y) * static_cast<std::size_t>(s.w) +
                    static_cast<std::size_t>(x)];
      }
    }
    for (int y = 0; y < s.ch4; ++y) {
      for (int x = 0; x < s.cw4; ++x) {
        const int c = s.c0 + x;
        const int r = s.r0 + y;
        if (c >= decoded_[0].cols || r >= decoded_[0].rows) continue;
        decoded_[0].cells[static_cast<std::size_t>(r) *
                              static_cast<std::size_t>(decoded_[0].cols) +
                          static_cast<std::size_t>(c)] =
            s.decoded[static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(s.cw4) +
                      static_cast<std::size_t>(x)];
      }
    }
  }

  // Encodes every transform part of one partition, feeding each part's
  // reconstruction into the next part's prediction (the decoder does the same
  // through make_intra_prediction/store_tx_info per TX block).
  bool encode_luma_parts(int r, int c, int w, int h,
                         partition::BlockSize size, const ModeChoice& y,
                         int partition, std::vector<TxPlan>* plans,
                         std::vector<syntax::LumaTxPart>* parts,
                         double* block_sse) {
    tpart::SplitGeometry geometry;
    std::string geometry_error;
    if (!tpart::tx_partition_geometry(
            static_cast<int>(size), r, c,
            static_cast<tpart::TxPartition>(partition), &geometry,
            &geometry_error)) {
      error_ = "tx partition geometry failed: " + geometry_error;
      return false;
    }
    plans->clear();
    parts->clear();
    for (int i = 0; i < geometry.count; ++i) {
      const tpart::SplitPart& part = geometry.parts[i];
      if (part.mi_col >= mi_cols_ || part.mi_row >= mi_rows_) {
        error_ = "tx partition part origin outside the coded plane";
        return false;
      }
      const int px = part.mi_col * 4;
      const int py = part.mi_row * 4;
      const int tw = tpart::tx_width(part.tx_size);
      const int th = tpart::tx_height(part.tx_size);
      std::vector<std::uint8_t> pred;
      if (!predict_block(0, px, py, tw, th, y.mode, y.angle_delta, pred,
                         /*uv_is_cfl=*/false, y.mrl_index, y.mrl_sec_index,
                         y.use_dip, y.dip_mode, y.dip_transpose,
                         /*block_mi_row=*/r, /*block_mi_col=*/c)) {
        return false;
      }
      TxPlan plan;
      if (!compute_tx(0, px, py, tw, th, pred, plan,
                      options_.tx_type || options_.idtx,
                      y.av2_mode, y.angle_delta, y.mrl_index)) {
        return false;
      }
      write_recon(0, px, py, tw, th, plan);
      decoded_[0].fill(px / 4, py / 4, tw / 4, th / 4);
      syntax::LumaTxPart syntax_part;
      syntax_part.tx_size = static_cast<std::uint8_t>(plan.tx_size);
      syntax_part.tx_type = static_cast<std::uint8_t>(plan.tx_type);
      syntax_part.sec_tx_type = static_cast<std::uint8_t>(plan.sec_tx_type);
      syntax_part.most_probable_stx_set =
          static_cast<std::uint8_t>(plan.most_probable_stx_set);
      syntax_part.eob = static_cast<std::uint32_t>(plan.eob);
      parts->push_back(syntax_part);
      plans->push_back(std::move(plan));
    }
    const int x0 = c * 4;
    const int y0 = r * 4;
    const int max_x = std::min(x0 + w, mi_cols_ * 4);
    const int max_y = std::min(y0 + h, mi_rows_ * 4);
    double sse = 0.0;
    for (int yy = y0; yy < max_y; ++yy) {
      for (int xx = x0; xx < max_x; ++xx) {
        const double d = static_cast<double>(source_sample(0, xx, yy)) -
                         static_cast<double>(recon_[0].at(xx, yy));
        sse += d * d;
      }
    }
    *block_sse = sse;
    return true;
  }

  // Tries NONE/SPLIT/HORZ/VERT by exact reconstruction SSE + a small syntax
  // cost, then re-runs the winner so the recon/decoded state is committed.
  bool choose_luma_tx_partition(int r, int c, int w, int h,
                                partition::BlockSize size,
                                const ModeChoice& y,
                                std::vector<TxPlan>* plans,
                                std::vector<syntax::LumaTxPart>* parts,
                                int* partition_out) {
    std::vector<tpart::TxPartition> options;
    options.push_back(tpart::TxPartition::kNone);
    const tpart::LegalSplits legal =
        tpart::legal_splits(static_cast<int>(size),
                            /*reduced_tx_part_set=*/false);
    for (int i = 0; i < legal.option_count; ++i) {
      const tpart::TxPartition option = legal.options[i];
      if (option == tpart::TxPartition::kNone) continue;
      if (option != tpart::TxPartition::kSplit &&
          option != tpart::TxPartition::kHorz &&
          option != tpart::TxPartition::kVert) {
        continue;
      }
      tpart::SplitGeometry geometry;
      std::string geometry_error;
      if (!tpart::tx_partition_geometry(static_cast<int>(size), r, c, option,
                                        &geometry, &geometry_error)) {
        continue;
      }
      bool inside = true;
      for (int k = 0; k < geometry.count; ++k) {
        if (geometry.parts[k].mi_col >= mi_cols_ ||
            geometry.parts[k].mi_row >= mi_rows_) {
          inside = false;
          break;
        }
      }
      if (inside) options.push_back(option);
    }

    // Debug gate hook: force one legal partition by name.
    bool have_forced = false;
    tpart::TxPartition forced = tpart::TxPartition::kNone;
    if (const char* value = std::getenv("AV2ENC_STILL_FORCE_TX_PARTITION")) {
      if (std::strcmp(value, "none") == 0) {
        have_forced = true;
        forced = tpart::TxPartition::kNone;
      } else if (std::strcmp(value, "split") == 0) {
        have_forced = true;
        forced = tpart::TxPartition::kSplit;
      } else if (std::strcmp(value, "horz") == 0) {
        have_forced = true;
        forced = tpart::TxPartition::kHorz;
      } else if (std::strcmp(value, "vert") == 0) {
        have_forced = true;
        forced = tpart::TxPartition::kVert;
      }
    }

    const double lambda =
        0.85 * std::pow(2.0, (options_.base_q_idx - 12) / 6.0);
    const LumaSnapshot snapshot = take_luma_snapshot(c*4, r*4, w, h);
    double best_cost = std::numeric_limits<double>::infinity();
    int best_partition = static_cast<int>(tpart::TxPartition::kNone);
    for (const tpart::TxPartition option : options) {
      if (have_forced && option != forced) continue;
      restore_luma_snapshot(snapshot);
      std::vector<TxPlan> trial_plans;
      std::vector<syntax::LumaTxPart> trial_parts;
      double sse = 0.0;
      if (!encode_luma_parts(r, c, w, h, size, y, static_cast<int>(option),
                             &trial_plans, &trial_parts, &sse)) {
        return false;
      }
      const double syntax_bits =
          option == tpart::TxPartition::kNone ? 0.0 : 3.0;
      const double cost = sse + lambda * syntax_bits;
      if (cost < best_cost) {
        best_cost = cost;
        best_partition = static_cast<int>(option);
      }
    }
    restore_luma_snapshot(snapshot);
    double sse = 0.0;
    if (!encode_luma_parts(r, c, w, h, size, y, best_partition, plans, parts,
                           &sse)) {
      return false;
    }
    *partition_out = best_partition;
    return true;
  }

  bool compute_chroma_cctx(int x0, int y0, int cw, int ch,
                           const std::vector<std::uint8_t>& u_pred,
                           const std::vector<std::uint8_t>& v_pred,
                           TxPlan& u_plan, TxPlan& v_plan) {
    const std::optional<txv::TxParams> params_opt =
        txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
    if (!params_opt.has_value()) {
      error_ = "unsupported chroma transform size for CCTX";
      return false;
    }
    txv::TxParams params = *params_opt;
    apply_qindex_step(params);
    const int area = params.w * params.h;
    const int coded_pels = cw * ch;

    const auto residual_adj = [&](int plane,
                                  const std::vector<std::uint8_t>& pred) {
      std::vector<std::int16_t> coded(static_cast<std::size_t>(coded_pels));
      for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
          const int src = source_sample(plane, x0 + x, y0 + y);
          const int p =
              pred[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                   static_cast<std::size_t>(x)];
          coded[static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(x)] =
              static_cast<std::int16_t>(src - p);
        }
      }
      std::vector<std::int16_t> adj(static_cast<std::size_t>(area));
      txv::downsample_residual(coded.data(), params, adj.data());
      return adj;
    };
    const std::vector<std::int16_t> u_adj = residual_adj(1, u_pred);
    const std::vector<std::int16_t> v_adj = residual_adj(2, v_pred);
    std::vector<double> u_coeff(static_cast<std::size_t>(area));
    std::vector<double> v_coeff(static_cast<std::size_t>(area));
    txv::forward_block(u_adj.data(), params, u_coeff.data());
    txv::forward_block(v_adj.data(), params, v_coeff.data());

    av2enc::tcq::CctxConfig config;
    config.enabled = true;
    config.quant_step_u = 1.0 / params.quant_factor;
    config.quant_step_v = config.quant_step_u;
    config.bitdepth = 8;
    std::vector<av2enc::tcq::CctxResult> candidates;
    std::string cctx_error;
    if (!av2enc::tcq::cctx_enumerate_scalar(u_coeff, v_coeff, params.w,
                                            params.h, config, &candidates,
                                            &cctx_error)) {
      error_ = "cctx enumerate failed: " + cctx_error;
      return false;
    }

    // §7.14.4 dequant for one chroma plane (no QM, no delta-Q, no TCQ shift).
    const auto dequant_plane = [&](const std::vector<std::int32_t>& levels,
                                   std::vector<std::int32_t>& dequant) {
      dequant.assign(static_cast<std::size_t>(area), 0);
      for (int i = 0; i < area; ++i) {
        const std::int32_t qc = levels[static_cast<std::size_t>(i)];
        const std::int64_t aq = qc < 0 ? -static_cast<std::int64_t>(qc) : qc;
        const std::int64_t high = (aq * quant_step_) & 0xFFFFFF;
        const std::int32_t dq = txv::round2(static_cast<std::int32_t>(high),
                                            txv::kQuantTableBits) /
                                params.dq_denom;
        dequant[static_cast<std::size_t>(i)] =
            std::max(-(1 << 15), std::min((1 << 15) - 1,
                                          qc < 0 ? -dq : dq));
      }
    };
    const auto reconstruct = [&](const av2enc::tcq::CctxResult& candidate,
                                 std::vector<std::int16_t>& u_res,
                                 std::vector<std::int16_t>& v_res) {
      std::vector<std::int32_t> du;
      std::vector<std::int32_t> dv;
      dequant_plane(candidate.u_levels, du);
      dequant_plane(candidate.v_levels, dv);
      std::vector<std::int32_t> pu(static_cast<std::size_t>(area));
      std::vector<std::int32_t> pv(static_cast<std::size_t>(area));
      for (int i = 0; i < area; ++i) {
        const av2enc::tcq::CctxPair pair = av2enc::tcq::cctx_dequant_pair(
            du[static_cast<std::size_t>(i)], dv[static_cast<std::size_t>(i)],
            candidate.type, 8);
        pu[static_cast<std::size_t>(i)] = pair.u;
        pv[static_cast<std::size_t>(i)] = pair.v;
      }
      u_res.assign(static_cast<std::size_t>(coded_pels), 0);
      v_res.assign(static_cast<std::size_t>(coded_pels), 0);
      txv::inverse_block(pu.data(), params, u_res.data());
      txv::inverse_block(pv.data(), params, v_res.data());
    };
    const auto distortion = [&](const av2enc::tcq::CctxResult& candidate,
                                const std::vector<std::int16_t>& u_res,
                                const std::vector<std::int16_t>& v_res) {
      double sse = 0.0;
      for (int plane = 1; plane <= 2; ++plane) {
        const std::vector<std::uint8_t>& pred = plane == 1 ? u_pred : v_pred;
        const std::vector<std::int16_t>& res = plane == 1 ? u_res : v_res;
        for (int y = 0; y < ch; ++y) {
          for (int x = 0; x < cw; ++x) {
            const std::size_t index =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
                static_cast<std::size_t>(x);
            const int v = clip1(static_cast<int>(pred[index]) +
                                static_cast<int>(res[index]));
            const double d =
                static_cast<double>(source_sample(plane, x0 + x, y0 + y)) -
                static_cast<double>(v);
            sse += d * d;
          }
        }
      }
      (void)candidate;
      return sse;
    };

    const double lambda = av2enc::rdo::lambda_sse_from_quant_step(
        1.0 / params.quant_factor, 0.125);
    // §5.20.7.30 scan order: eob is the last non-zero *scan* position, not the
    // raster index.  The component returns raster-order eobs; the syntax needs
    // scan order.
    std::vector<std::uint16_t> scan(static_cast<std::size_t>(area));
    syntax::build_scan(params.w, params.h, scan);
    const auto scan_eob = [&](const std::vector<std::int32_t>& levels) {
      for (int c = area - 1; c >= 0; --c) {
        if (levels[static_cast<std::size_t>(
                scan[static_cast<std::size_t>(c)])] != 0) {
          return c + 1;
        }
      }
      return 0;
    };
    // RDOQ is a per-plane trellis over the quantised coefficients.  CCTX
    // quantises the *rotated* U/V pair with the scalar step, so run the same
    // trellis on the rotated coefficients of each candidate type; the decoder
    // dequantises the levels and applies the §7.14.6 inverse rotation, so the
    // resulting stream is exact.  Without this the --still-rdoq flag was
    // silently ignored on CCTX chroma.
    const auto rdoq_tune = [&](av2enc::tcq::CctxResult& candidate) -> bool {
      std::vector<double> rot_u(static_cast<std::size_t>(area));
      std::vector<double> rot_v(static_cast<std::size_t>(area));
      for (int i = 0; i < area; ++i) {
        av2enc::tcq::cctx_forward_rotate(
            u_coeff[static_cast<std::size_t>(i)],
            v_coeff[static_cast<std::size_t>(i)], candidate.type,
            &rot_u[static_cast<std::size_t>(i)],
            &rot_v[static_cast<std::size_t>(i)]);
      }
      for (int plane = 1; plane <= 2; ++plane) {
        const std::vector<double>& rotated = plane == 1 ? rot_u : rot_v;
        std::vector<std::int32_t>& levels =
            plane == 1 ? candidate.u_levels : candidate.v_levels;
        std::vector<std::int32_t> scalar_scan(
            static_cast<std::size_t>(area));
        std::vector<av2enc::rdoq::Coefficient> input(
            static_cast<std::size_t>(area));
        for (int c = 0; c < area; ++c) {
          const int pos = scan[static_cast<std::size_t>(c)];
          scalar_scan[static_cast<std::size_t>(c)] =
              levels[static_cast<std::size_t>(pos)];
          input[static_cast<std::size_t>(c)].original =
              rotated[static_cast<std::size_t>(pos)];
          input[static_cast<std::size_t>(c)].scalar_level =
              levels[static_cast<std::size_t>(pos)];
          input[static_cast<std::size_t>(c)].level_context = 0;
        }
        av2enc::rdoq::av2::BlockInfo info;
        info.plane = plane;
        info.params = params;
        info.txb_skip_context = 0;
        info.dc_sign_context = 0;
        info.qctx = qctx_slice_;
        av2enc::rdoq::av2::Model model;
        std::string rdoq_error;
        if (!model.init(info, scalar_scan, &rdoq_error)) {
          error_ = "cctx rdoq model init failed: " + rdoq_error;
          return false;
        }
        av2enc::rdoq::Config rdoq_config;
        rdoq_config.quant_step = 1.0 / params.quant_factor;
        rdoq_config.lambda = av2enc::rdo::lambda_sse_from_quant_step(
            rdoq_config.quant_step, options_.rdoq_sse_scale);
        rdoq_config.max_coefficients = area;
        av2enc::rdoq::RdoqResult rdoq_result;
        if (!av2enc::rdoq::rdoq_quantize(input, rdoq_config,
                                         model.rate_model(), rdoq_result,
                                         &rdoq_error)) {
          error_ = "cctx rdoq quantize failed: " + rdoq_error;
          return false;
        }
        std::fill(levels.begin(), levels.end(), 0);
        for (int c = 0; c < rdoq_result.eob; ++c) {
          levels[static_cast<std::size_t>(
              scan[static_cast<std::size_t>(c)])] =
              rdoq_result.levels[static_cast<std::size_t>(c)];
        }
      }
      return true;
    };
    if (options_.rdoq) {
      for (av2enc::tcq::CctxResult& candidate : candidates) {
        if (!rdoq_tune(candidate)) return false;
      }
    }
    int best = -1;
    double best_cost = std::numeric_limits<double>::infinity();
    int best_eob_u = 0;
    int best_eob_v = 0;
    std::vector<std::int16_t> best_u_res;
    std::vector<std::int16_t> best_v_res;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      const av2enc::tcq::CctxResult& candidate = candidates[i];
      const int eob_u = scan_eob(candidate.u_levels);
      // The decoder infers cctx_type = 0 and reads no symbol when the U EOB
      // is 0 (all_zero) or 1.  A rotation with eob_u <= 1 is therefore
      // unsignalable.
      if (candidate.type != av2enc::tcq::CctxType::kNone && eob_u <= 1) {
        continue;
      }
      std::vector<std::int16_t> u_res;
      std::vector<std::int16_t> v_res;
      reconstruct(candidate, u_res, v_res);
      const double sse = distortion(candidate, u_res, v_res);
      const double bits =
          eob_u > 1 ? av2enc::tcq::cctx_type_bits(candidate.type) : 0.0;
      const double cost = sse + lambda * bits;
      if (cost < best_cost) {
        best_cost = cost;
        best = static_cast<int>(i);
        best_eob_u = eob_u;
        best_eob_v = scan_eob(candidate.v_levels);
        best_u_res = std::move(u_res);
        best_v_res = std::move(v_res);
      }
    }
    if (best < 0) {
      error_ = "no signallable CCTX candidate";
      return false;
    }
    const av2enc::tcq::CctxResult& chosen =
        candidates[static_cast<std::size_t>(best)];

    const auto fill_plan = [&](TxPlan& plan, int plane,
                               const std::vector<std::int32_t>& levels,
                               std::int32_t eob,
                               const std::vector<std::uint8_t>& pred,
                               const std::vector<std::int16_t>& res) {
      plan.plane = plane;
      plan.tx_size = params.tx_size;
      plan.tx_type = static_cast<int>(txv::TxType::kDctDct);
      plan.coded_w = cw;
      plan.coded_h = ch;
      plan.adj_w = params.w;
      plan.adj_h = params.h;
      plan.eob = eob;
      plan.levels = levels;
      plan.dequant_levels.clear();
      plan.pred = pred;
      plan.recon.resize(static_cast<std::size_t>(coded_pels));
      for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
          const std::size_t index =
              static_cast<std::size_t>(y) * static_cast<std::size_t>(cw) +
              static_cast<std::size_t>(x);
          plan.recon[index] = static_cast<std::uint8_t>(
              clip1(static_cast<int>(pred[index]) +
                    static_cast<int>(res[index])));
        }
      }
    };
    fill_plan(u_plan, 1, chosen.u_levels, best_eob_u, u_pred, best_u_res);
    fill_plan(v_plan, 2, chosen.v_levels, best_eob_v, v_pred, best_v_res);
    cctx_type_ = static_cast<int>(chosen.type);
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
  // Intra block copy (§5.20.5.3/.4, §7.13.3.18)
  // ---------------------------------------------------------------------
  ibc::FrameView ibc_frame_view() const {
    // The intra/palette paths read only the reconstructed planes.  Bounds are
    // the coded MiCols*4 / MiRows*4 plane; is_bv_valid owns the tile rules.
    ibc::FrameView view;
    view.y = ibc::PlaneView{recon_[0].samples.data(), recon_[0].width,
                            mi_cols_ * 4, mi_rows_ * 4};
    view.u = ibc::PlaneView{recon_[1].samples.data(), recon_[1].width,
                            mi_cols_ * 2, mi_rows_ * 2};
    view.v = ibc::PlaneView{recon_[2].samples.data(), recon_[2].width,
                            mi_cols_ * 2, mi_rows_ * 2};
    view.has_chroma = true;
    view.sub_x = 1;
    view.sub_y = 1;
    return view;
  }

  // Does copying `bv` reproduce the source exactly on all three planes?
  // `sse` receives the luma error even when chroma differs.
  bool ibc_copy_sse(const ibc::FrameView& frame, int r, int c,
                    partition::BlockSize size, ibc::BlockVector bv,
                    int visible_w, int visible_h, std::int64_t* sse) {
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    const int x0 = c * 4;
    const int y0 = r * 4;
    std::vector<std::uint8_t> copy_y(static_cast<std::size_t>(w) *
                                     static_cast<std::size_t>(h));
    const int cw = (w + 1) / 2;
    const int ch = (h + 1) / 2;
    std::vector<std::uint8_t> copy_u(static_cast<std::size_t>(cw) *
                                     static_cast<std::size_t>(ch));
    std::vector<std::uint8_t> copy_v(copy_u.size());
    if (!ibc::copy_block(frame, r, c, static_cast<int>(size), bv,
                         copy_y.data(), w,
                         copy_u.data(), cw, copy_v.data(), cw, visible_w,
                         visible_h)) {
      return false;
    }
    std::int64_t total = 0;
    for (int yy = 0; yy < visible_h; ++yy) {
      for (int xx = 0; xx < visible_w; ++xx) {
        const int d = static_cast<int>(
                          copy_y[static_cast<std::size_t>(yy) *
                                     static_cast<std::size_t>(w) +
                                 static_cast<std::size_t>(xx)]) -
                      static_cast<int>(source_sample(0, x0 + xx, y0 + yy));
        total += static_cast<std::int64_t>(d) * d;
      }
    }
    *sse = total;
    if (total != 0) return true;
    const int cvw = (visible_w + 1) / 2;
    const int cvh = (visible_h + 1) / 2;
    for (int plane = 1; plane < 3; ++plane) {
      const std::vector<std::uint8_t>& copy = plane == 1 ? copy_u : copy_v;
      const int px0 = x0 / 2;
      const int py0 = y0 / 2;
      for (int yy = 0; yy < cvh; ++yy) {
        for (int xx = 0; xx < cvw; ++xx) {
          if (copy[static_cast<std::size_t>(yy) *
                       static_cast<std::size_t>(cw) +
                   static_cast<std::size_t>(xx)] !=
              source_sample(plane, px0 + xx, py0 + yy)) {
            *sse = 1;  // non-zero marks "not an exact copy"
            return true;
          }
        }
      }
    }
    return true;
  }

  // Screen content usually has few distinct luma values; only then is the
  // bounded window search worth its cost on natural photographs.
  bool ibc_low_color_rank(int x0, int y0, int w, int h) const {
    const int visible_w = std::min(w, mi_cols_ * 4 - x0);
    const int visible_h = std::min(h, mi_rows_ * 4 - y0);
    bool seen[256] = {};
    int distinct = 0;
    for (int yy = 0; yy < visible_h; ++yy) {
      for (int xx = 0; xx < visible_w; ++xx) {
        const int v = source_sample(0, x0 + xx, y0 + yy);
        if (!seen[v]) {
          seen[v] = true;
          if (++distinct > 8) return false;
        }
      }
    }
    return true;
  }

  struct IbcChoice {
    bool found = false;
    ibc::BlockVector bv{};
    int stack_index = -1;  // >= 0 => NEARMV with that RefMvIdx
  };

  // Choose an exact-copy block vector (skip_flag = 1 codes no residual, so a
  // non-exact copy could only reduce quality).  Stack candidates come first
  // (NEARMV is cheaper); then periodic offsets and a bounded local window
  // (NEWMV when the vector is not a stack entry).
  bool choose_ibc(int r, int c, partition::BlockSize size, IbcChoice& out) {
    out = IbcChoice{};
    if (!options_.ibc ||
        !ibc_stream::intrabc_available(true, static_cast<int>(size))) {
      return true;
    }
    const ibc::FrameView frame = ibc_frame_view();
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    const int x0 = c * 4;
    const int y0 = r * 4;
    const int visible_w = std::min(w, mi_cols_ * 4 - x0);
    const int visible_h = std::min(h, mi_rows_ * 4 - y0);
    const ibc_stream::BvStack stack = ibc_stream::build_bv_stack(
        frame, ibc_config_, ibc_buffer_, r, c, static_cast<int>(size),
        ibc_neighbours_);
    const auto try_bv = [&](ibc::BlockVector bv, int stack_index) {
      // Integer block vectors only (force_integer_mv = 1).
      if ((bv.row & 7) != 0 || (bv.col & 7) != 0) return false;
      if (!ibc_stream::is_bv_encodable(frame, ibc_config_, ibc_buffer_, r, c,
                                       static_cast<int>(size), bv, nullptr)) {
        return false;
      }
      std::int64_t sse = 0;
      if (!ibc_copy_sse(frame, r, c, size, bv, visible_w, visible_h, &sse) ||
          sse != 0) {
        return false;
      }
      out.found = true;
      out.bv = bv;
      out.stack_index = stack_index;
      return true;
    };
    for (int i = 0; i < stack.count; ++i) {
      if (!stack.entries[static_cast<std::size_t>(i)].valid) continue;
      if (try_bv(stack.entries[static_cast<std::size_t>(i)].bv, i)) {
        return true;
      }
    }
    static constexpr int kPeriodic[] = {8, 16, 32, 64, 128};
    for (const int d : kPeriodic) {
      const ibc::BlockVector cands[2] = {ibc::BlockVector{-d * 8, 0},
                                         ibc::BlockVector{0, -d * 8}};
      for (const ibc::BlockVector& bv : cands) {
        if (try_bv(bv, -1)) return true;
      }
    }
    if (!ibc_low_color_rank(x0, y0, w, h)) return true;
    const int step = std::max(1, options_.ibc_search_step);
    for (int sy = std::max(0, y0 - 64); sy <= y0; sy += step) {
      for (int sx = std::max(0, x0 - 64); sx <= x0; sx += step) {
        const ibc::BlockVector bv{(sy - y0) * 8, (sx - x0) * 8};
        if (try_bv(bv, -1)) return true;
      }
    }
    return true;
  }

  // ---------------------------------------------------------------------
  // Coding a block
  // ---------------------------------------------------------------------
  bool encode_block(int r, int c, partition::BlockSize size,
                    std::size_t index, bool has_chroma,
                    const ChromaInfo& chroma, int tree_type = 0) {
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    const int w4 = w >> 2;
    const int h4 = h >> 2;
    const int x0 = c * 4;
    const int y0 = r * 4;
    // §6.19.3.1: a CHROMA_PART leaf codes no luma mode, TX size or luma
    // residual; a LUMA_PART/SHARED leaf codes luma (and chroma only when it
    // owns a chroma block).
    const bool chroma_part = tree_type == 2;
    const bool do_luma = !chroma_part;
    if (options_.ibc) {
      ibc_stream::begin_ibc_block(ibc_buffer_, r, c);
    }

    ModeChoice y;
    palette::PaletteResult palette_result;
    bool use_palette = false;
    IbcChoice ibc_choice;
    bool use_ibc = false;
    std::vector<TxPlan> luma_plans;
    std::vector<syntax::LumaTxPart> luma_parts;
    int luma_partition = 0;
    if (do_luma) {
      if (!choose_luma_mode(x0, y0, w, h, y)) return false;

      // §5.20.5.5: the decoder sets IntraJointModes for the whole coding block
      // before residual(), so a later transform part inside the block reads the
      // current block's joint mode.  Update it before per-part prediction.  (For
      // the single-transform path this is equivalent to updating it afterwards.)
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

      // §5.20.8 palette candidate: luma-only and only when the luma mode is DC
      // (palette_mode_info is conditioned on YMode == DC_PRED).  A palette-only
      // block codes eob = 0 on all planes, so compare its prediction SSE with
      // the chosen intra prediction and keep it only when it is better.
      if (options_.palette && w >= 8 && h >= 8 && w <= 64 && h <= 64) {
        std::uint8_t cache[16] = {};
        const int cache_n = palette::get_palette_cache(palette_state_, r, c,
                                                       c > 0, cache);
        std::vector<std::uint8_t> pal_src(
            static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
        for (int yy = 0; yy < h; ++yy) {
          for (int xx = 0; xx < w; ++xx) {
            pal_src[static_cast<std::size_t>(yy) * static_cast<std::size_t>(w) +
                    static_cast<std::size_t>(xx)] =
                source_sample(0, x0 + xx, y0 + yy);
          }
        }
        const int visible_w = std::min(w, mi_cols_ * 4 - x0);
        const int visible_h = std::min(h, mi_rows_ * 4 - y0);
        palette::PaletteParams palette_params;
        palette_params.max_colors = 8;
        palette_params.exact_when_possible = true;
        palette_params.choose_direction = true;
        std::string palette_error;
        if (!palette::analyze_palette(
                pal_src.data(), w, w, h, visible_w, visible_h, cache, cache_n,
                palette_params, palette_result, palette_error)) {
          error_ = "analyze_palette failed: " + palette_error;
          return false;
        }
        // Defensive: §5.20.8.1 codes a strictly increasing palette; a duplicate
        // would make the delta syntax illegal.  Fall back to intra prediction
        // instead of emitting a stream the decoder rejects.
        bool representable = palette_result.has_palette;
        for (int i = 1; i < palette_result.size && representable; ++i) {
          if (palette_result.colors[static_cast<std::size_t>(i)] <=
              palette_result.colors[static_cast<std::size_t>(i - 1)]) {
            representable = false;
          }
        }
        if (palette_result.has_palette && !representable) {
          palette_result.has_palette = false;
        }
        if (palette_result.has_palette && palette_result.sse < y.sse) {
          use_palette = true;
          y.mode = intra::Mode::kDc;
          y.angle_delta = 0;
          y.av2_mode = kDcPred;
          y.mrl_index = 0;
          y.mrl_sec_index = false;
          y.use_dip = false;
          y.dip_mode = 0;
          y.dip_transpose = false;
          y.sse = palette_result.sse;
        }
      }

      // §5.20.5.3/.4 intra block copy.  skip_flag = 1 codes no residual, so the
      // encoder only selects an exact copy (checked on all three planes with the
      // same `copy_block` arithmetic the decoder uses).  Blocks whose chroma
      // residual block is not the plain 4:2:0 half-size (shared-chroma
      // partitions) are skipped: §7.13.3.18's copy would not cover that block.
      if (options_.ibc) {
        const BlockSize ibc_chroma =
            partition::chroma_plane_residual_size(geometry_, size);
        if (partition::is_valid_block_size(ibc_chroma) &&
            partition::block_width(ibc_chroma) == (w / 2) &&
            partition::block_height(ibc_chroma) == (h / 2)) {
          if (!choose_ibc(r, c, size, ibc_choice)) return false;
          use_ibc = ibc_choice.found;
        }
      }
      if (use_ibc) {
        // read_intrabc_info() forces DC luma/chroma and clears palette/DIP
        // state.
        use_palette = false;
        y.mode = intra::Mode::kDc;
        y.angle_delta = 0;
        y.av2_mode = kDcPred;
        y.mrl_index = 0;
        y.mrl_sec_index = false;
        y.use_dip = false;
        y.dip_mode = 0;
        y.dip_transpose = false;
        y.sse = 0;
      }

      if (use_ibc) {
        std::vector<std::uint8_t> luma_pred;
        luma_pred.resize(static_cast<std::size_t>(w) *
                         static_cast<std::size_t>(h));
        const ibc::FrameView frame = ibc_frame_view();
        const int visible_w = std::min(w, mi_cols_ * 4 - x0);
        const int visible_h = std::min(h, mi_rows_ * 4 - y0);
        if (!ibc::copy_block(frame, r, c, static_cast<int>(size),
                             ibc_choice.bv, luma_pred.data(), w, nullptr, 0,
                             nullptr, 0, visible_w, visible_h)) {
          error_ = "IBC luma copy failed";
          return false;
        }
        const std::optional<txv::TxParams> params_opt =
            txv::make_tx_params(w, h, txv::TxType::kDctDct);
        if (!params_opt.has_value()) {
          error_ = "unsupported IBC block transform size";
          return false;
        }
        TxPlan luma_plan;
        luma_plan.plane = 0;
        luma_plan.tx_size = params_opt->tx_size;
        luma_plan.tx_type = static_cast<int>(txv::TxType::kDctDct);
        luma_plan.coded_w = w;
        luma_plan.coded_h = h;
        luma_plan.adj_w = params_opt->w;
        luma_plan.adj_h = params_opt->h;
        luma_plan.eob = 0;
        luma_plan.pred = luma_pred;
        luma_plan.recon = luma_pred;
        write_recon(0, x0, y0, w, h, luma_plan);
        luma_plans.push_back(std::move(luma_plan));
      } else if (use_palette) {
        std::vector<std::uint8_t> luma_pred;
        luma_pred.resize(
            static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
        palette::predict_palette_block(palette_result, w, h, luma_pred.data(),
                                       w);
        // Transform-skip: a palette-only block codes eob = 0 on every plane.
        const std::optional<txv::TxParams> params_opt =
            txv::make_tx_params(w, h, txv::TxType::kDctDct);
        if (!params_opt.has_value()) {
          error_ = "unsupported palette block transform size";
          return false;
        }
        TxPlan luma_plan;
        luma_plan.plane = 0;
        luma_plan.tx_size = params_opt->tx_size;
        luma_plan.tx_type = static_cast<int>(txv::TxType::kDctDct);
        luma_plan.coded_w = w;
        luma_plan.coded_h = h;
        luma_plan.adj_w = params_opt->w;
        luma_plan.adj_h = params_opt->h;
        luma_plan.eob = 0;
        luma_plan.pred = luma_pred;
        luma_plan.recon = luma_pred;
        write_recon(0, x0, y0, w, h, luma_plan);
        luma_plans.push_back(std::move(luma_plan));
      } else if (options_.tx_partition) {
        if (!choose_luma_tx_partition(r, c, w, h, size, y, &luma_plans,
                                      &luma_parts, &luma_partition)) {
          return false;
        }
      } else {
        std::vector<std::uint8_t> luma_pred;
        if (!predict_block(0, x0, y0, w, h, y.mode, y.angle_delta, luma_pred,
                           /*uv_is_cfl=*/false, y.mrl_index, y.mrl_sec_index,
                           y.use_dip, y.dip_mode, y.dip_transpose)) {
          return false;
        }
        TxPlan luma_plan;
        if (!compute_tx(0, x0, y0, w, h, luma_pred, luma_plan,
                        options_.tx_type || options_.idtx, y.av2_mode,
                        y.angle_delta, y.mrl_index)) {
          return false;
        }
        write_recon(0, x0, y0, w, h, luma_plan);
        luma_plans.push_back(std::move(luma_plan));
      }
    }

    if (do_luma) {
      // Record the final signalled luma mode over the whole coding block for
      // the CHROMA_PART trees (SDP codes luma before chroma).
      for (int yy = 0; yy < h4; ++yy) {
        const int rr = r + yy;
        if (rr >= mi_rows_) break;
        for (int xx = 0; xx < w4; ++xx) {
          const int cc = c + xx;
          if (cc >= mi_cols_) break;
          const std::size_t index =
              static_cast<std::size_t>(rr) *
                  static_cast<std::size_t>(mi_cols_) +
              static_cast<std::size_t>(cc);
          luma_mode_[index] = static_cast<std::uint8_t>(y.av2_mode);
          luma_angle_[index] = static_cast<std::int8_t>(y.angle_delta);
        }
      }
    }

    // §5.20.3.1: a sub-8 leaf that is not the chroma owner codes luma only;
    // the owner codes the whole inherited ChromaMi block.  For every leaf at
    // 8x8 or above ChromaMi is the leaf itself, so this reduces to the
    // historical behaviour.
    const BlockSize chroma_bsize = chroma.valid ? chroma.size : size;
    const BlockSize chroma_size =
        partition::chroma_plane_residual_size(geometry_, chroma_bsize);
    if (has_chroma && !partition::is_valid_block_size(chroma_size)) {
      error_ = "invalid chroma residual size";
      return false;
    }
    const int cw = partition::block_width(chroma_size);
    const int ch = partition::block_height(chroma_size);
    const int cx0 = chroma.valid ? chroma.mi_col * 2 : x0 / 2;
    const int cy0 = chroma.valid ? chroma.mi_row * 2 : y0 / 2;

    // §5.20.5.5/.6: YMode for a CHROMA_PART leaf is the co-located luma-tree
    // mode; it drives the UV CDF context, uv_mode_index and AngleDeltaUV.  For
    // SHARED/LUMA_PART leaves it is the leaf's own coded luma mode.
    const int uv_y_mode =
        do_luma ? y.av2_mode
                : luma_mode_at(chroma.valid ? chroma.mi_row : r,
                               chroma.valid ? chroma.mi_col : c);
    const int uv_y_angle_delta =
        do_luma ? y.angle_delta
                : luma_angle_at(chroma.valid ? chroma.mi_row : r,
                                chroma.valid ? chroma.mi_col : c);
    UvChoice uv;
    TxPlan u_plan;
    TxPlan v_plan;
    std::vector<std::uint8_t> u_pred;
    std::vector<std::uint8_t> v_pred;
    bool cfl = false;
    if (has_chroma) {
      if (use_ibc) {
        cctx_type_ = 0;
        // §7.13.3.17: the chroma block vector offset is the luma offset scaled
        // by the 4:2:0 subsampling.  The source region was already validated by
        // is_bv_encodable(), so this read is inside the coded plane.
        u_pred.resize(static_cast<std::size_t>(cw) *
                      static_cast<std::size_t>(ch));
        v_pred.resize(u_pred.size());
        const int cdx = ibc::plane_offset(ibc_choice.bv.col, 1);
        const int cdy = ibc::plane_offset(ibc_choice.bv.row, 1);
        const int sx = cx0 + cdx;
        const int sy = cy0 + cdy;
        for (int yy = 0; yy < ch; ++yy) {
          for (int xx = 0; xx < cw; ++xx) {
            u_pred[static_cast<std::size_t>(yy) * static_cast<std::size_t>(cw) +
                    static_cast<std::size_t>(xx)] =
                recon_[1].at(sx + xx, sy + yy);
            v_pred[static_cast<std::size_t>(yy) * static_cast<std::size_t>(cw) +
                    static_cast<std::size_t>(xx)] =
                recon_[2].at(sx + xx, sy + yy);
          }
        }
        for (int plane = 1; plane < 3; ++plane) {
          TxPlan& plan = plane == 1 ? u_plan : v_plan;
          const std::optional<txv::TxParams> params_opt =
              txv::make_tx_params(cw, ch, txv::TxType::kDctDct);
          if (!params_opt.has_value()) {
            error_ = "unsupported IBC chroma transform size";
            return false;
          }
          plan.plane = plane;
          plan.tx_size = params_opt->tx_size;
          plan.tx_type = static_cast<int>(txv::TxType::kDctDct);
          plan.coded_w = cw;
          plan.coded_h = ch;
          plan.adj_w = params_opt->w;
          plan.adj_h = params_opt->h;
          plan.eob = 0;
          plan.pred = plane == 1 ? u_pred : v_pred;
          plan.recon = plan.pred;
        }
      } else {
        if (!choose_uv_mode(cx0, cy0, cw, ch, uv_y_mode, uv_y_angle_delta,
                            uv)) {
          return false;
        }
        cfl = uv.is_cfl;
        if (cfl && (cw > 32 || ch > 32)) {
          // §7.13.2 dispatch: a CfL chroma block with w > 32 or h > 32 must use
          // the DC intra prediction subsampled process (§7.13.2.11), which this
          // encoder does not implement.  M1's largest chroma block is 32x32, so
          // this is a guard for a future format/SB configuration, not a path
          // the current search can select.
          error_ =
              "CfL chroma block exceeds 32 samples: §7.13.2.11 DC-subsampled "
              "prediction is not implemented";
          return false;
        }
        if (cfl) {
          // The CfL component expects the DC prediction in the chroma plane and
          // reconstructed above/left neighbours. Derive the implicit alpha and
          // write the CfL prediction into the planes.
          if (!predict_block(1, cx0, cy0, cw, ch, intra::Mode::kDc, 0, u_pred,
                             /*uv_is_cfl=*/true)) {
            return false;
          }
          if (!predict_block(2, cx0, cy0, cw, ch, intra::Mode::kDc, 0, v_pred,
                             /*uv_is_cfl=*/true)) {
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
          // The mode decision already chose between derived, explicit and MHCCP
          // hypotheses by SSE; apply exactly the chosen prediction so the
          // reconstruction matches the signalled syntax.
          cfl::compute_luma(ub, scratch);
          if (uv.cfl_index == 2) {
            const int num4_above_right =
                count_top_right(1, cx0 >> 2, cy0 >> 2, cw >> 2);
            const int num4_below_left =
                count_bottom_left(1, cx0 >> 2, cy0 >> 2, ch >> 2);
            av2enc::cfl::mhccp::Block mh_u = av2enc::cfl::mhccp::make_block(
                ub, num4_above_right, num4_below_left, mi_cols_ * 4,
                mi_rows_ * 4);
            av2enc::cfl::mhccp::Block mh_v = av2enc::cfl::mhccp::make_block(
                vb, num4_above_right, num4_below_left, mi_cols_ * 4,
                mi_rows_ * 4);
            av2enc::cfl::mhccp::apply_mhccp_pair(mh_u, mh_v, uv.cfl_mh_dir,
                                                 uv.mh_params_u, uv.mh_params_v);
          } else {
            cfl::apply_prediction(ub, uv.cfl_alpha_q8_u, scratch);
            cfl::apply_prediction(vb, uv.cfl_alpha_q8_v, scratch);
          }
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

        cctx_type_ = 0;
        if (options_.cctx) {
          if (!compute_chroma_cctx(cx0, cy0, cw, ch, u_pred, v_pred, u_plan,
                                   v_plan)) {
            return false;
          }
        } else {
          if (!compute_tx(1, cx0, cy0, cw, ch, u_pred, u_plan)) return false;
          if (!compute_tx(2, cx0, cy0, cw, ch, v_pred, v_plan)) return false;
        }
      }
      write_recon(1, cx0, cy0, cw, ch, u_plan);
      write_recon(2, cx0, cy0, cw, ch, v_plan);
    }

    if (trace_ && do_luma) {
      const TxPlan& luma_plan = luma_plans.front();
      std::cerr << "m1 block r=" << r << " c=" << c
                << " size=" << partition::block_size_name(size)
                << " y=" << y.av2_mode << "/" << y.angle_delta
                << " uv=" << (cfl ? -1 : uv.av2_mode) << "/"
                << uv.angle_delta << " cfl=" << (cfl ? 1 : 0)
                << " tx=" << luma_plan.coded_w << "x" << luma_plan.coded_h
                << " type=" << luma_plan.tx_type
                << " parts=" << luma_plans.size()
                << " eob=" << luma_plan.eob
                << " ist=" << luma_plan.sec_tx_type << "/"
                << luma_plan.most_probable_stx_set << "\n";
    }

    // Fill this leaf's syntax entry: modes, transform sizes, EOBs, and its
    // slice of the flat scan-order coefficient array.  No bits are written
    // here; syntax::assemble_still_frame() emits the frame after the tree is done.
    syntax::BlockSyntax& leaf = blocks_[index];
    leaf.flags |= syntax::kFlagHasChromaInfo;
    leaf.has_chroma = has_chroma ? 1 : 0;
    leaf.tree_type = static_cast<std::uint8_t>(tree_type);
    leaf.chroma_bsize = static_cast<std::uint8_t>(chroma_bsize);
    leaf.chroma_mi_row =
        static_cast<std::uint16_t>(chroma.valid ? chroma.mi_row : r);
    leaf.chroma_mi_col =
        static_cast<std::uint16_t>(chroma.valid ? chroma.mi_col : c);
    leaf.y_mode = static_cast<std::uint8_t>(uv_y_mode);
    leaf.y_angle_delta = static_cast<std::int8_t>(uv_y_angle_delta);
    leaf.mrl_index = static_cast<std::int8_t>(y.mrl_index);
    leaf.mrl_sec_index = static_cast<std::uint8_t>(y.mrl_sec_index ? 1 : 0);
    leaf.uv_mode = static_cast<std::uint8_t>(uv.av2_mode);
    if (cfl) leaf.flags |= syntax::kFlagIsCfl;
    leaf.cfl_index = static_cast<std::uint8_t>(uv.cfl_index);
    leaf.cfl_alpha_signs = static_cast<std::uint8_t>(uv.cfl_alpha_signs);
    leaf.cfl_alpha_u = static_cast<std::uint8_t>(uv.cfl_alpha_u);
    leaf.cfl_alpha_v = static_cast<std::uint8_t>(uv.cfl_alpha_v);
    leaf.cfl_mh_dir = static_cast<std::uint8_t>(uv.cfl_mh_dir);
    leaf.use_dip = static_cast<std::uint8_t>(y.use_dip ? 1 : 0);
    leaf.dip_transpose = static_cast<std::uint8_t>(y.dip_transpose ? 1 : 0);
    leaf.dip_mode = static_cast<std::uint8_t>(y.dip_mode);
    if (use_palette) {
      leaf.flags |= syntax::kFlagIsPalette;
      leaf.palette_index = static_cast<std::uint32_t>(palette_results_.size());
      palette::store_palette_state(palette_state_, r, c, w, h, palette_result);
      palette_results_.push_back(std::move(palette_result));
    }
    if (use_ibc) {
      leaf.flags |= syntax::kFlagIsIbc;
      leaf.ibc_skip = 1;
      leaf.ibc_mode = ibc_choice.stack_index >= 0 ? 1 : 0;
      leaf.ibc_ref_mv_idx =
          ibc_choice.stack_index >= 0
              ? static_cast<std::uint8_t>(ibc_choice.stack_index)
              : 0;
      leaf.ibc_bv_row = ibc_choice.bv.row;
      leaf.ibc_bv_col = ibc_choice.bv.col;
    }
    const TxPlan* plans[3] = {luma_plans.empty() ? nullptr : luma_plans.data(),
                              &u_plan, &v_plan};
    leaf.levels_offset = static_cast<std::uint32_t>(levels_.size());
    leaf.sec_tx_type =
        do_luma ? static_cast<std::uint8_t>(luma_plans.front().sec_tx_type) : 0;
    leaf.most_probable_stx_set =
        do_luma
            ? static_cast<std::uint8_t>(luma_plans.front().most_probable_stx_set)
            : 0;
    leaf.cctx_type = static_cast<std::uint8_t>(cctx_type_);
    if (options_.tx_partition && do_luma) {
      leaf.tx_part_count = static_cast<std::uint8_t>(luma_parts.size());
      leaf.tx_partition = static_cast<std::uint8_t>(luma_partition);
      for (std::size_t i = 0; i < luma_parts.size(); ++i) {
        leaf.tx_parts[i] = luma_parts[i];
      }
      for (const TxPlan& plan : luma_plans) append_levels(plan);
      leaf.tx_size[0] = static_cast<std::uint8_t>(luma_plans.front().tx_size);
      leaf.tx_type[0] = static_cast<std::uint8_t>(luma_plans.front().tx_type);
      leaf.eob[0] = static_cast<std::uint32_t>(luma_plans.front().eob);
    } else if (do_luma) {
      leaf.tx_size[0] = static_cast<std::uint8_t>(luma_plans.front().tx_size);
      leaf.tx_type[0] = static_cast<std::uint8_t>(luma_plans.front().tx_type);
      leaf.eob[0] = static_cast<std::uint32_t>(luma_plans.front().eob);
      append_levels(luma_plans.front());
    } else {
      // A CHROMA_PART leaf codes no luma transform.
      leaf.tx_size[0] = 0;
      leaf.tx_type[0] = 0;
      leaf.eob[0] = 0;
    }
    for (int plane = 1; plane < 3; ++plane) {
      const TxPlan& plan = *plans[plane];
      leaf.tx_size[plane] = static_cast<std::uint8_t>(plan.tx_size);
      leaf.tx_type[plane] = static_cast<std::uint8_t>(plan.tx_type);
      leaf.eob[plane] = static_cast<std::uint32_t>(plan.eob);
      append_levels(plan);
    }

    // intra_joint was updated before the residual so per-part prediction sees
    // this coding block's mode; only uv_smooth remains to record here.
    const int cr = chroma.valid ? chroma.mi_row / 2 : r / 2;
    const int cc0 = chroma.valid ? chroma.mi_col / 2 : c / 2;
    const int cw4 = cw / 4;
    const int ch4 = ch / 4;
    const int chroma_mi_rows = mi_rows_ / 2;
    const int chroma_mi_cols = mi_cols_ / 2;
    if (has_chroma) {
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
    }

    if (do_luma) {
      decoded_[0].fill(c, r, w4, h4);
    }
    if (has_chroma) {
      decoded_[1].fill(cc0, cr, cw4, ch4);
      decoded_[2].fill(cc0, cr, cw4, ch4);
    }

    if (options_.enable_deblock) {
      const int chroma_mi_row = chroma.valid ? chroma.mi_row : r;
      const int chroma_mi_col = chroma.valid ? chroma.mi_col : c;
      if (options_.tx_partition && do_luma) {
        fill_deblock_mi_parts(r, c, size, cw, ch, luma_parts, luma_partition,
                              u_plan.tx_size, has_chroma, chroma_mi_row,
                              chroma_mi_col, chroma_bsize, do_luma);
      } else {
        fill_deblock_mi(
            r, c, size, cw, ch,
            do_luma ? luma_plans.front().tx_size
                    : static_cast<int>(u_plan.tx_size),
            u_plan.tx_size, has_chroma, chroma_mi_row, chroma_mi_col,
            chroma_bsize, do_luma);
      }
    }
    if (use_ibc) ++ibc_blocks_;
    if (options_.ibc) {
      ibc_stream::finish_ibc_block(
          ibc_buffer_, ibc_neighbours_, r, c, static_cast<int>(size),
          use_ibc, use_ibc ? ibc_choice.bv : ibc::BlockVector{});
    }
    return true;
  }

  // §5.20.6.3: with TX partitioning the deblocking TX grid is per transform
  // part; SubPuSize stays Max_Tx_Size_Rect[MiSize] like the decoder.
  void fill_deblock_mi_parts(int mi_row, int mi_col, partition::BlockSize size,
                             int cw, int ch,
                             const std::vector<syntax::LumaTxPart>& parts,
                             int partition, int chroma_tx_size, bool has_chroma,
                             int chroma_mi_row, int chroma_mi_col,
                             partition::BlockSize chroma_bsize,
                             bool do_luma = true) {
    const int max_tx = tpart::max_tx_size_rect(static_cast<int>(size));
    fill_deblock_mi(mi_row, mi_col, size, cw, ch, max_tx, chroma_tx_size,
                    has_chroma, chroma_mi_row, chroma_mi_col, chroma_bsize,
                    do_luma);
    tpart::SplitGeometry geometry;
    std::string geometry_error;
    if (!tpart::tx_partition_geometry(static_cast<int>(size), mi_row, mi_col,
                                      static_cast<tpart::TxPartition>(partition),
                                      &geometry, &geometry_error)) {
      return;
    }
    if (geometry.count != static_cast<int>(parts.size())) return;
    for (int i = 0; i < geometry.count; ++i) {
      const tpart::SplitPart& part = geometry.parts[i];
      const int w4 = tpart::tx_width(part.tx_size) / 4;
      const int h4 = tpart::tx_height(part.tx_size) / 4;
      for (int yy = 0; yy < h4; ++yy) {
        const int rr = part.mi_row + yy;
        if (rr >= mi_rows_) break;
        for (int xx = 0; xx < w4; ++xx) {
          const int cc = part.mi_col + xx;
          if (cc >= mi_cols_) break;
          deblock::DeblockMi& mi = deblock_mi_[0][static_cast<std::size_t>(
              rr * deblock_cols_ + cc)];
          mi.tx_size = static_cast<TxSize>(parts[static_cast<std::size_t>(i)]
                                               .tx_size);
          mi.tx_col_base = static_cast<std::uint16_t>(part.mi_col);
          mi.tx_row_base = static_cast<std::uint16_t>(part.mi_row);
        }
      }
    }
  }

  void make_cfl_block(int plane, int cx0, int cy0, int cw, int ch,
                      cfl::Block& block) {
    block.luma = recon_[0].samples.data();
    block.luma_stride = recon_[0].width;
    block.luma_max_x = mi_cols_ * 4 - 1;
    block.luma_max_y = mi_rows_ * 4 - 1;
    block.sb_top_y = (cy0 * 2 / geometry_.sb_size) * geometry_.sb_size;
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
    block.ds_filter = options_.cfl_ds_filter;
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
      // The CfL base prediction must not receive the IBP DC post-filter
      // (§7.13.2.12 excludes uv_is_cfl blocks).
      if (!predict_block(1, cx0, cy0, cw, ch, intra::Mode::kDc, 0, u_dc,
                         /*uv_is_cfl=*/true)) {
        return false;
      }
      if (!predict_block(2, cx0, cy0, cw, ch, intra::Mode::kDc, 0, v_dc,
                         /*uv_is_cfl=*/true)) {
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
      int alpha_u = cfl::derive_implicit_alpha(ub, scratch);
      cfl::compute_luma(vb, scratch);
      int alpha_v = cfl::derive_implicit_alpha(vb, scratch);
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
      int cfl_index = 1;
      av2enc::cfl::Signalling explicit_syntax{};
      if (options_.cfl_explicit) {
        // §5.20.7.32 CFL_EXPLICIT: exhaustive sign/magnitude search on the
        // source planes.  The derived prediction above is the incumbent, so a
        // tie keeps cfl_index = 1 and the historical stream.
        cfl::SourcePlane su;
        su.plane = picture_.u.data();
        su.stride = chroma_width_;
        su.max_x = chroma_width_ - 1;
        su.max_y = chroma_height_ - 1;
        cfl::SourcePlane sv;
        sv.plane = picture_.v.data();
        sv.stride = chroma_width_;
        sv.max_x = chroma_width_ - 1;
        sv.max_y = chroma_height_ - 1;
        const std::optional<cfl::SearchResult> explicit_choice =
            cfl::search_alpha(ub, vb, su, sv,
                              cfl::SearchMode::kExhaustiveSse, scratch);
        if (explicit_choice.has_value()) {
          cfl::apply_prediction_pair(ub, vb, explicit_choice->alpha, scratch);
          std::int64_t explicit_sse = 0;
          for (int yy = 0; yy < ch; ++yy) {
            for (int xx = 0; xx < cw; ++xx) {
              const int du = source_sample(1, cx0 + xx, cy0 + yy) -
                             recon_[1].at(cx0 + xx, cy0 + yy);
              explicit_sse += static_cast<std::int64_t>(du) * du;
              const int dv = source_sample(2, cx0 + xx, cy0 + yy) -
                             recon_[2].at(cx0 + xx, cy0 + yy);
              explicit_sse += static_cast<std::int64_t>(dv) * dv;
            }
          }
          if (explicit_sse < sse) {
            sse = explicit_sse;
            cfl_index = 0;
            explicit_syntax = explicit_choice->syntax;
            alpha_u = cfl::alpha_q8(explicit_choice->alpha.u);
            alpha_v = cfl::alpha_q8(explicit_choice->alpha.v);
          } else {
            // Restore the derived prediction that won the comparison.
            cfl::compute_luma(ub, scratch);
            cfl::apply_prediction(ub, cfl::derive_implicit_alpha(ub, scratch),
                                  scratch);
            cfl::apply_prediction(vb, cfl::derive_implicit_alpha(vb, scratch),
                                  scratch);
          }
        }
      }
      int cfl_mh_dir = 0;
      av2enc::cfl::mhccp::MultiParams mh_params_u{};
      av2enc::cfl::mhccp::MultiParams mh_params_v{};
      if (options_.mhccp &&
          av2enc::cfl::mhccp::allowed(cw, ch, /*enable_mhccp=*/true,
                                      /*is_chroma_part=*/false,
                                      /*frame_is_intra=*/true,
                                      /*cfl_allowed_in_sdp=*/false,
                                      /*lossless=*/false)) {
        // §7.13.6-.8: a third hypothesis with derived Q16 weights.  Only
        // cfl_mh_dir is signalled.  Compared against the incumbent by SSE.
        const int num4_above_right =
            count_top_right(1, cx0 >> 2, cy0 >> 2, cw >> 2);
        const int num4_below_left =
            count_bottom_left(1, cx0 >> 2, cy0 >> 2, ch >> 2);
        av2enc::cfl::mhccp::Block mh_u = av2enc::cfl::mhccp::make_block(
            ub, num4_above_right, num4_below_left, mi_cols_ * 4,
            mi_rows_ * 4);
        av2enc::cfl::mhccp::Block mh_v = av2enc::cfl::mhccp::make_block(
            vb, num4_above_right, num4_below_left, mi_cols_ * 4,
            mi_rows_ * 4);
        av2enc::cfl::mhccp::SourcePlane mh_su;
        mh_su.plane = picture_.u.data();
        mh_su.stride = chroma_width_;
        mh_su.max_x = chroma_width_ - 1;
        mh_su.max_y = chroma_height_ - 1;
        av2enc::cfl::mhccp::SourcePlane mh_sv;
        mh_sv.plane = picture_.v.data();
        mh_sv.stride = chroma_width_;
        mh_sv.max_x = chroma_width_ - 1;
        mh_sv.max_y = chroma_height_ - 1;
        const std::optional<av2enc::cfl::mhccp::Candidate> multi =
            av2enc::cfl::mhccp::search_mhccp(mh_u, mh_v, mh_su, mh_sv);
        if (multi.has_value()) {
          av2enc::cfl::mhccp::apply_mhccp_pair(mh_u, mh_v, multi->mh_dir,
                                               multi->params_u,
                                               multi->params_v);
          if (multi->total_sse() < sse) {
            sse = multi->total_sse();
            cfl_index = 2;
            cfl_mh_dir = multi->mh_dir;
            mh_params_u = multi->params_u;
            mh_params_v = multi->params_v;
            alpha_u = 0;
            alpha_v = 0;
            explicit_syntax = av2enc::cfl::Signalling{};
          } else {
            // Restore the incumbent prediction.
            cfl::compute_luma(ub, scratch);
            if (cfl_index == 0) {
              cfl::apply_prediction(ub, alpha_u, scratch);
              cfl::apply_prediction(vb, alpha_v, scratch);
            } else {
              cfl::apply_prediction(ub, cfl::derive_implicit_alpha(ub, scratch),
                                    scratch);
              cfl::apply_prediction(vb, cfl::derive_implicit_alpha(vb, scratch),
                                    scratch);
            }
          }
        }
      }
      if (sse < best.sse) {
        best.sse = sse;
        best.is_cfl = true;
        best.mode = intra::Mode::kDc;
        best.angle_delta = 0;
        best.av2_mode = kDcPred;
        best.cfl_index = cfl_index;
        best.cfl_alpha_signs = explicit_syntax.alpha_signs;
        best.cfl_alpha_u = explicit_syntax.alpha_u;
        best.cfl_alpha_v = explicit_syntax.alpha_v;
        best.cfl_alpha_q8_u = alpha_u;
        best.cfl_alpha_q8_v = alpha_v;
        best.cfl_mh_dir = cfl_mh_dir;
        best.mh_params_u = mh_params_u;
        best.mh_params_v = mh_params_v;
      }
    }
    return true;
  }

  void fill_deblock_mi(int mi_row, int mi_col, partition::BlockSize size,
                       int cw, int ch, int luma_tx_size,
                       int chroma_tx_size, bool has_chroma,
                       int chroma_mi_row, int chroma_mi_col,
                       partition::BlockSize chroma_bsize, bool do_luma = true) {
    const int w = partition::block_width(size);
    const int h = partition::block_height(size);
    const int w4 = w / 4;
    const int h4 = h / 4;
    const TxSize luma = static_cast<TxSize>(luma_tx_size);
    const TxSize chroma = static_cast<TxSize>(chroma_tx_size);
    (void)cw;
    (void)ch;
    const int clamped_w4 = std::min(w4, mi_cols_ - mi_col);
    const int clamped_h4 = std::min(h4, mi_rows_ - mi_row);
    if (do_luma) {
      for (int yy = 0; yy < h4; ++yy) {
        const int rr = mi_row + yy;
        if (rr >= mi_rows_) break;
        for (int xx = 0; xx < w4; ++xx) {
          const int cc = mi_col + xx;
          if (cc >= mi_cols_) break;
          const int idx = rr * deblock_cols_ + cc;
          deblock::DeblockMi mi;
          mi.block_col_base = static_cast<std::uint16_t>(mi_col);
          mi.block_row_base = static_cast<std::uint16_t>(mi_row);
          mi.block_w4 = static_cast<std::uint8_t>(clamped_w4);
          mi.block_h4 = static_cast<std::uint8_t>(clamped_h4);
          mi.sub_pu_size = luma;
          mi.tx_size = luma;
          mi.tx_col_base = static_cast<std::uint16_t>(mi_col);
          mi.tx_row_base = static_cast<std::uint16_t>(mi_row);
          mi.sub_pu_col_base = static_cast<std::uint16_t>(mi_col);
          mi.sub_pu_row_base = static_cast<std::uint16_t>(mi_row);
          mi.skip = 0;
          mi.region_intra = 1;
          mi.lossless = 0;
          // §7.17 deblock strength uses the block's effective qindex, which the
          // per-SB segmentation path changes.
          mi.qindex = static_cast<std::uint8_t>(current_qindex_);
          deblock_mi_[0][static_cast<std::size_t>(idx)] = mi;
        }
      }
    }
    if (!has_chroma) return;
    // Chroma metadata is owned by the ChromaMi block (not the luma leaf) so a
    // sub-8 leaf group deblocks as one 8x8 chroma block.
    const int c_bsize_w = partition::block_width(chroma_bsize) / 4;
    const int c_bsize_h = partition::block_height(chroma_bsize) / 4;
    const int c_w4 = std::min(c_bsize_w, mi_cols_ - chroma_mi_col);
    const int c_h4 = std::min(c_bsize_h, mi_rows_ - chroma_mi_row);
    for (int yy = 0; yy < c_bsize_h; ++yy) {
      const int rr = chroma_mi_row + yy;
      if (rr >= mi_rows_) break;
      for (int xx = 0; xx < c_bsize_w; ++xx) {
        const int cc = chroma_mi_col + xx;
        if (cc >= mi_cols_) break;
        const int idx = rr * deblock_cols_ + cc;
        deblock::DeblockMi mi = deblock_mi_[0][static_cast<std::size_t>(idx)];
        mi.block_col_base = static_cast<std::uint16_t>(chroma_mi_col);
        mi.block_row_base = static_cast<std::uint16_t>(chroma_mi_row);
        mi.block_w4 = static_cast<std::uint8_t>(c_w4);
        mi.block_h4 = static_cast<std::uint8_t>(c_h4);
        mi.tx_size = chroma;
        mi.tx_col_base = static_cast<std::uint16_t>(chroma_mi_col);
        mi.tx_row_base = static_cast<std::uint16_t>(chroma_mi_row);
        // §7.17 deblock strength uses the block's effective qindex, which the
        // per-SB segmentation path changes.
        mi.qindex = static_cast<std::uint8_t>(current_qindex_);
        deblock_mi_[1][static_cast<std::size_t>(idx)] = mi;
        deblock_mi_[2][static_cast<std::size_t>(idx)] = mi;
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
  // §7.14.2 qlookup(base_q_idx).  528 at the historical M1 QP 90.  For the
  // per-SB segmentation path this is qlookup(current_qindex_).
  int quant_step_ = txv::kQuantStep;
  int current_qindex_ = 90;
  // §8.3.1 COEFF_CDF_Q_CTXS slice for the RDOQ rate model.
  int qctx_slice_ = 0;
  // Frame-level QM state shared by the header writer and §7.14.4 dequant.
  qm::QmFrameState qm_state_;
  // Optional per-plane DC/AC quantizers from the delta-Q search.
  const dcdeltaq::QuantFrameState* delta_quant_ = nullptr;
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
  // §5.20.7.27 cctx_type chosen for the current leaf's chroma pair (0 = none).
  int cctx_type_ = 0;

  std::vector<int> intra_joint_;
  // §5.20.5.5 co-located luma YMode/AngleDeltaY per 4x4, for CHROMA_PART
  // leaves whose mode context comes from the luma tree.
  std::vector<std::uint8_t> luma_mode_;
  std::vector<std::int8_t> luma_angle_;
  std::vector<std::uint8_t> uv_smooth_;
  std::array<DecodedMap, 3> decoded_;
  // §5.20.8 per-MI palette sizes/colours (get_palette_cache source) and the
  // per-leaf palette decisions handed to the syntax writer.
  palette::PaletteState palette_state_;
  std::vector<palette::PaletteResult> palette_results_;
  // §5.20.5.1/.4 intra block copy: buffer bookkeeping, the exact BV neighbour
  // map, its config, and the per-leaf decisions (for the syntax writer, which
  // rebuilds the same stack from the block syntax).
  ibc::IbcBufferState ibc_buffer_;
  ibc_stream::BvNeighbourMap ibc_neighbours_;
  ibc::IbcConfig ibc_config_{};
  int ibc_blocks_ = 0;

  int sb_mi_ = 16;
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
  // CDEF's block-index context and CCSO's unit size are still derived from a
  // 64x64 superblock.  Reject them by name instead of emitting a desynchronised
  // stream.
  if (options.sb128 && options.sb256) {
    error = "sb128 and sb256 are mutually exclusive";
    return false;
  }
  if (options.ext_partitions && options.force_single_block) {
    error = "--still-ext-partitions cannot be combined with --still-single";
    return false;
  }
  if (options.sdp) {
    if (options.sb128 || options.sb256) {
      error = "--still-sdp is only wired for 64x64 superblocks";
      return false;
    }
    if (options.ext_partitions || options.sub8) {
      error = "--still-sdp is not wired with extended partitions or sub-8";
      return false;
    }
    if (options.enable_cdef || options.enable_ccso) {
      error = "--still-sdp with CDEF/CCSO is not wired yet";
      return false;
    }
    if (options.enable_segmentation || options.enable_delta_q) {
      error = "--still-sdp with segmentation/delta-Q is not wired yet";
      return false;
    }
  }
  if (options.delta_q_res < 0 || options.delta_q_res > 3) {
    error = "delta_q_res must be in [0, 3]";
    return false;
  }
  if (options.qm) {
    if (options.qm_level < 0 || options.qm_level > 14) {
      error = "--still-qm-level must be in 0..14 (15 is the no-QM sentinel)";
      return false;
    }
    if (options.rdoq) {
      error = "--still-qm is not supported with --still-rdoq yet";
      return false;
    }
  }
  if (options.cctx) {
    if (options.qm) {
      error = "--still-cctx is not supported with --still-qm yet";
      return false;
    }
    if (options.delta_q) {
      error = "--still-cctx is not supported with --still-dc-ac-delta-q yet";
      return false;
    }
  }
  if (options.tx_partition) {
    if (options.qm) {
      error = "--still-tx-partition is not supported with --still-qm yet";
      return false;
    }
    if (options.delta_q) {
      error =
          "--still-tx-partition is not supported with --still-dc-ac-delta-q "
          "yet";
      return false;
    }
    // These change the intra prediction of the coding block (reference-line
    // selection, DIP tap geometry, IBP post-filter), which is applied per
    // transform part.  Reject instead of guessing the per-part semantics.
    if (options.enable_dip) {
      error = "--still-tx-partition is not supported with --still-dip yet";
      return false;
    }
    if (options.enable_mrl) {
      error = "--still-tx-partition is not supported with --still-mrl yet";
      return false;
    }
    if (options.enable_ibp) {
      error = "--still-tx-partition is not supported with --still-ibp yet";
      return false;
    }
  }
  if (options.ddt) {
    error =
        "--still-ddt is not signallable in a still picture: §5.4.8 omits "
        "enable_inter_ddt when single_picture_header_flag = 1 and §7.15.4.1 "
        "applies DDT only to inter blocks; use the M2 inter residual path "
        "(src/codec/transform/ddt/gate)";
    return false;
  }
  if (options.tcq) {
    if (options.qm) {
      error = "--still-tcq is not supported with --still-qm yet";
      return false;
    }
    if (options.rdoq) {
      error = "--still-tcq is not supported with --still-rdoq yet";
      return false;
    }
    if (options.delta_q) {
      error = "--still-tcq is not supported with --still-dc-ac-delta-q yet";
      return false;
    }
    if (options.ist) {
      // The IST analysis projects the Quant domain; under TCQ the writer
      // carries the §5.20.7.27 base levels, so the projected block cannot be
      // mapped back to syntax levels without a second trellis.  Reject instead
      // of producing a stream whose reconstruction disagrees with the decoder.
      error = "--still-tcq is not supported with --still-ist yet";
      return false;
    }
  }
  if (options.delta_q) {
    if (options.qm) {
      error = "--still-dc-ac-delta-q is not supported with --still-qm yet";
      return false;
    }
    if (options.rdoq) {
      error = "--still-dc-ac-delta-q is not supported with --still-rdoq yet";
      return false;
    }
    if ((picture.width % 8U) != 0U || (picture.height % 8U) != 0U) {
      error = "--still-dc-ac-delta-q requires width and height multiples of 8";
      return false;
    }
  }
  // IBP's MrlIndex scope is 0 (src/codec/intra/ibp) and MRL changes the edge
  // filter rules, so the two cannot be combined in one sequence.
  if (options.enable_mrl && options.enable_ibp) {
    error = "--still-mrl and --still-ibp cannot be combined";
    return false;
  }
  if (options.enable_mrl && options.rdo_intra) {
    error = "--still-mrl and --still-rdo-intra cannot be combined";
    return false;
  }
  if (options.cfl_ds_filter < 0 || options.cfl_ds_filter > 3) {
    error = "cfl_ds_filter_index must be in 0..3";
    return false;
  }
  if (options.mhccp && !options.enable_cfl) {
    error = "--still-mhccp requires CfL (do not combine with --no-cfl)";
    return false;
  }
  // Palette intra is opt-in (`--still-palette`).  §5.20.8 is luma-only and is
  // only signalled for DC luma blocks of at least 8x8; the encoder decision in
  // encode_block() and the palette_stream adapter enforce both.
  // IntraBC is opt-in (`--still-ibc`).  The §5.20.5.1 ring/delay model is
  // defined for 64x64 superblocks, so the 128/256-SB modes are rejected rather
  // than signalled with a mismatched delay window.
  if (options.ibc && (options.sb128 || options.sb256)) {
    error = "--still-ibc requires 64x64 superblocks (drop --still-sb128/256)";
    return false;
  }
  if (options.ibc_search_step < 1 || options.ibc_search_step > 64) {
    error = "ibc_search_step must be in 1..64";
    return false;
  }
  // §5.4.7/§5.18.3.3/.4: let the screen-content module validate the flag
  // combination the frame header will carry.  Single-picture M1 infers
  // seq_force_screen_content_tools = seq_force_integer_mv = SELECT_*, so the
  // frame codes allow_screen_content_tools/force_integer_mv and, for IBC, the
  // global-local region bits.
  {
    scc_flags::SequenceSccConfig seq_cfg;
    const scc_flags::SequenceSccState seq_state =
        scc_flags::derive_sequence_scc_state(seq_cfg);
    scc_flags::FrameSccContext ctx;
    ctx.frame_is_intra = true;
    scc_flags::FrameSccConfig scc;
    scc.allow_screen_content_tools = options.palette || options.ibc;
    scc.force_integer_mv = scc.allow_screen_content_tools;
    scc.allow_intrabc = options.ibc;
    scc.allow_global_intrabc = options.ibc;
    scc.allow_local_intrabc = false;
    scc.max_bvp_drl_bits_minus_1 = 0;
    std::string scc_error;
    if (!scc_flags::validate_frame_scc_config(
            seq_state, ctx, scc, /*seq_max_bvp_drl_bits_minus_1=*/0,
            /*allow_frame_max_bvp_drl_bits=*/false, scc_error)) {
      error = "screen-content flags: " + scc_error;
      return false;
    }
    const scc_flags::FrameSccState state = scc_flags::derive_frame_scc_state(
        seq_state, ctx, scc, 0, false);
    // make_intrabc_bv() only emits integer sample offsets; the syntax would
    // otherwise need intrabc_precision + quarter-pel MVDs.
    if (state.IsIntraBC && !state.force_integer_mv) {
      error = "intra block copy requires force_integer_mv = 1";
      return false;
    }
  }
  if (options.per_segment_q &&
      (options.enable_segmentation || options.enable_delta_q || options.qm ||
       options.ist)) {
    error = "per-segment Q cannot be combined with the neutral segmentation, "
            "delta-Q, QM or IST paths in this integration";
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

  // Optional frame-level DC/AC delta-Q decision.  The component's search fixes
  // the sequence base deltas at 0 and returns the legal frame deltas plus the
  // derived per-plane quantizers used by forward quantisation and dequant.
  dcdeltaq::SequenceDeltaQ seq_delta_q;
  dcdeltaq::FrameDeltaQ frame_delta_q;
  dcdeltaq::QuantFrameState delta_quant;
  const dcdeltaq::QuantFrameState* delta_quant_ptr = nullptr;
  if (options.delta_q) {
    dcdeltaq::FramePlanesView view;
    view.width = static_cast<int>(picture.width);
    view.height = static_cast<int>(picture.height);
    view.y = picture.y;
    view.u = picture.u;
    view.v = picture.v;
    dcdeltaq::DeltaQSearchConfig search_config;
    search_config.base_q_idx = options.base_q_idx;
    search_config.equal_ac_dc_q = false;
    search_config.separate_uv_delta_q = false;
    search_config.max_delta_magnitude = 24;
    search_config.max_passes = 3;
    search_config.max_luma_blocks = 2048;
    dcdeltaq::DeltaQSearchResult search_result;
    std::string search_error;
    if (!dcdeltaq::search_frame_delta_q(view, search_config, &search_result,
                                        &search_error)) {
      error = "delta-q search failed: " + search_error;
      return false;
    }
    seq_delta_q = search_result.seq;
    frame_delta_q = search_result.frame;
    delta_quant = search_result.quant;
    delta_quant_ptr = &delta_quant;
    if (std::getenv("AV2ENC_STILL_TRACE") != nullptr) {
      std::cerr << "delta-q: frame ydc=" << frame_delta_q.delta_q_y_dc
                << " udc=" << frame_delta_q.delta_q_u_dc
                << " vdc=" << frame_delta_q.delta_q_v_dc
                << " uac=" << frame_delta_q.delta_q_u_ac
                << " vac=" << frame_delta_q.delta_q_v_ac
                << " yq=" << delta_quant.plane[0].dc << "/"
                << delta_quant.plane[0].ac
                << " uq=" << delta_quant.plane[1].dc << "/"
                << delta_quant.plane[1].ac
                << " vq=" << delta_quant.plane[2].dc << "/"
                << delta_quant.plane[2].ac << "\n";
    }
  }

  // §5.4.9 / §5.18.7.1 real per-SB segmentation: assign each superblock to
  // segment 0 (base qindex) or segment 1 (base + per_segment_q_delta) using a
  // deterministic luma-variance split.  The encoder uses the derived qindex for
  // forward quant, dequant and the deblock grid; the syntax writer signals the
  // same segment id.
  rc::SegmentationParams segment_params;
  StillOptions encode_options = options;
  if (options.per_segment_q) {
    const int sb_px = (options.sb128 || options.sb256) ? 128 : 64;
    const int width = static_cast<int>(picture.width);
    const int height = static_cast<int>(picture.height);
    const int sb_cols = (width + sb_px - 1) / sb_px;
    const int sb_rows = (height + sb_px - 1) / sb_px;
    const std::size_t sb_count =
        static_cast<std::size_t>(sb_cols) * static_cast<std::size_t>(sb_rows);
    encode_options.sb_segment.assign(sb_count, 0);
    std::vector<std::uint64_t> variance(sb_count, 0);
    std::uint64_t total_variance = 0;
    for (int sby = 0; sby < sb_rows; ++sby) {
      for (int sbx = 0; sbx < sb_cols; ++sbx) {
        const int x0 = sbx * sb_px;
        const int y0 = sby * sb_px;
        const int w = std::min(sb_px, width - x0);
        const int h = std::min(sb_px, height - y0);
        std::uint64_t sum = 0;
        std::uint64_t sumsq = 0;
        for (int y = 0; y < h; ++y) {
          const std::uint8_t* row =
              picture.y.data() +
              static_cast<std::size_t>(y0 + y) * static_cast<std::size_t>(width) +
              static_cast<std::size_t>(x0);
          for (int x = 0; x < w; ++x) {
            const std::uint64_t v = row[x];
            sum += v;
            sumsq += v * v;
          }
        }
        const std::uint64_t n = static_cast<std::uint64_t>(w) * h;
        const std::uint64_t var =
            n == 0 ? 0 : (sumsq * n - sum * sum) / (n * n);
        variance[static_cast<std::size_t>(sby * sb_cols + sbx)] = var;
        total_variance += var;
      }
    }
    const std::uint64_t mean_variance =
        sb_count == 0 ? 0 : total_variance / sb_count;
    for (std::size_t i = 0; i < sb_count; ++i) {
      encode_options.sb_segment[i] =
          variance[i] <= mean_variance ? 1 : 0;
    }
    segment_params.enabled = true;
    segment_params.update_map = true;
    std::string seg_error;
    if (!rc::set_segment_alt_q(segment_params, /*segment_id=*/1,
                               options.per_segment_q_delta, &seg_error)) {
      error = "per-segment Q: " + seg_error;
      return false;
    }
  }

  Encoder encoder(picture, encode_options, delta_quant_ptr);
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
    dopts.enable_sdp = options.sdp;
    // §7.17.6: chroma filter strength uses the effective UV AC delta.  Without
    // this the decoder and encoder deblock with different thresholds whenever
    // a UV AC delta is signalled.
    dopts.delta_q_u_ac = frame_delta_q.delta_q_u_ac;
    dopts.delta_q_v_ac = frame_delta_q.delta_q_v_ac;
    dopts.base_uv_ac_delta_q = seq_delta_q.base_uv_ac_delta_q;
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
    // §7.18.1 Skips[]: IBC is the only still-intra path that codes
    // skip_flag = 1, and §5.20.10.1 read_cdef() leaves such blocks unfiltered
    // unless cdef_on_skip_txfm_frame_enable is set.  Materialise the skip grid
    // from the decided IBC leaves instead of assuming every block is intra.
    std::vector<std::uint8_t> cdef_skip_grid(
        static_cast<std::size_t>(mi_inputs.mi_cols) *
            static_cast<std::size_t>(mi_inputs.mi_rows),
        0);
    bool any_skip = false;
    for (const syntax::BlockSyntax& leaf : stream.blocks) {
      if ((leaf.flags & syntax::kFlagIsIbc) == 0 || leaf.ibc_skip == 0) {
        continue;
      }
      const BlockSize leaf_size = static_cast<BlockSize>(leaf.bsize);
      const int leaf_w4 = partition::num_4x4_blocks_wide(leaf_size);
      const int leaf_h4 = partition::num_4x4_blocks_high(leaf_size);
      for (int y = 0; y < leaf_h4; ++y) {
        const int rr = leaf.mi_row + y;
        if (rr >= mi_inputs.mi_rows) break;
        for (int x = 0; x < leaf_w4; ++x) {
          const int cc = leaf.mi_col + x;
          if (cc >= mi_inputs.mi_cols) break;
          cdef_skip_grid[static_cast<std::size_t>(rr) *
                             static_cast<std::size_t>(mi_inputs.mi_cols) +
                         static_cast<std::size_t>(cc)] = 1;
          any_skip = true;
        }
      }
    }
    if (any_skip) mi_inputs.skip = cdef_skip_grid.data();
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

  // CCSO reads the *pre-CDEF* luma for its band classifier and loop
  // restoration uses it as CurrFrame, so keep a copy of the deblocked planes
  // before CDEF overwrites the reconstruction.
  //
  // Luma can only have one FrameRestorationType: PC-Wiener and Wiener NS are
  // alternatives for plane 0.  Chroma LR is a different plane and composes.
  if (options.enable_pc_wiener && options.enable_wiener) {
    error = "--still-pc-wiener and --still-wiener select two luma "
            "FrameRestorationTypes";
    return false;
  }
  // The combined luma PC-Wiener + chroma Wiener header row composes: plane 0
  // is RESTORE_PC_WIENER and planes 1/2 are RESTORE_WIENER_NONSEP.  Only two
  // luma FrameRestorationTypes conflict (checked above).
  const bool use_loop_restoration = options.enable_wiener ||
                                    options.enable_chroma_lr ||
                                    options.enable_pc_wiener;
  // §7.20: LrFrame := CdefFrame, then Wiener/PC-Wiener read CdefFrame and
  // write LrFrame, then GDF reads CdefFrame and rewrites LrFrame luma.  Keep
  // the pre-CDEF CurrFrame and the post-CCSO CdefFrame alive as separate
  // buffers so GDF no longer consumes the restoration output.
  Plane pre_cdef_planes[3];
  if (options.enable_ccso || use_loop_restoration || options.enable_gdf) {
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

  // CCSO (§7.19).  The default is band-offset-only with one offset per band
  // and plane; the fit minimises SSE against the original picture.  The
  // opt-in --still-ccso-edge form additionally searches
  // (scale, quant, ext_filter, edge_clf), fits a (band, cls0, cls1) offset
  // table and prunes units whose filter does not reduce SSE (ccso_blk = 0).
  restoration::CcsoFrameParams ccso_params;
  std::vector<std::uint8_t> ccso_blk_grid;
  std::vector<std::uint8_t> ccso_edge_grids[3];
  const std::uint8_t* ccso_grid_ptr[3] = {nullptr, nullptr, nullptr};
  int ccso_blk_cols[3] = {0, 0, 0};
  int ccso_blk_rows[3] = {0, 0, 0};
  bool ccso_ready = false;
  if (options.enable_ccso) {
    const int bands_log2 = options.ccso_bands_log2;
    if (bands_log2 < 0 || bands_log2 > 6) {
      error = "ccso_bands_log2 outside 0..6";
      return false;
    }
    const int vis_w[3] = {static_cast<int>(picture.width),
                          static_cast<int>((picture.width + 1U) / 2U),
                          static_cast<int>((picture.width + 1U) / 2U)};
    const int vis_h[3] = {static_cast<int>(picture.height),
                          static_cast<int>((picture.height + 1U) / 2U),
                          static_cast<int>((picture.height + 1U) / 2U)};
    const std::span<const std::uint8_t> src_planes[3] = {
        picture.y, picture.u, picture.v};
    const Plane& curr_luma = pre_cdef_planes[0];
    // §5.18.7.12: CcsoLumaSizeLog2 = Mi_Width_Log2[SbSize] + 2 (6 for 64,
    // 7 for a 128x128 frame superblock).
    const int ccso_luma_log2 =
        (options.sb128 || options.sb256) ? 7 : 6;
    ccso_params.ccso_luma_size_log2 = ccso_luma_log2;

    if (!options.ccso_edge) {
      const int bands = 1 << bands_log2;
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
            const double off =
                static_cast<double>(restoration::kCcsoOffset[i]);
            const double sse =
                sum_e2[static_cast<std::size_t>(band)] -
                2.0 * off * sum_e[static_cast<std::size_t>(band)] +
                off * off * count[static_cast<std::size_t>(band)];
            if (sse < best_sse - 1e-9) {
              best_sse = sse;
              best_idx = i;
            }
          }
          plane_config.offset_idx[band] =
              static_cast<std::uint8_t>(best_idx);
          if (best_idx != 0) ++nonzero;
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
      // One shared all-ones grid keeps the published --still-ccso bytes.
      int cols = 0;
      int rows = 0;
      restoration::ccso_blk_grid_size(encoder.mi_cols() * 4,
                                      encoder.mi_rows() * 4, ccso_luma_log2,
                                      0, 0, &cols, &rows);
      ccso_blk_grid.assign(static_cast<std::size_t>(cols) *
                               static_cast<std::size_t>(rows),
                           1);
      for (int plane = 0; plane < 3; ++plane) {
        ccso_grid_ptr[plane] = ccso_blk_grid.data();
        ccso_blk_cols[plane] = cols;
        ccso_blk_rows[plane] = rows;
        ccso_params.blks[plane] = ccso_grid_ptr[plane];
        ccso_params.blk_cols[plane] = cols;
        ccso_params.blk_rows[plane] = rows;
      }
    } else {
      // Edge-classified form.  §5.18.7.12 limits max_band_log2 to 3 here.
      const int edge_bands_log2 =
          std::min(bands_log2, restoration::kCcsoMaxBandLog2Normal);
      const int bands = 1 << edge_bands_log2;
      const int band_shift = 8 - edge_bands_log2;
      const int mi_luma_w = encoder.mi_cols() * 4;
      const int mi_luma_h = encoder.mi_rows() * 4;
      const int luma_stride = curr_luma.width;
      for (int plane = 0; plane < 3; ++plane) {
        const int sub = plane == 0 ? 0 : 1;
        const Plane& rec = encoder.recon(plane);
        const int shift = ccso_luma_log2 - sub;  // CCSO unit size
        int cols = 0;
        int rows = 0;
        restoration::ccso_blk_grid_size((mi_luma_w) >> sub,
                                        (mi_luma_h) >> sub, ccso_luma_log2,
                                        sub, sub, &cols, &rows);

        struct EdgeFit {
          double sse = std::numeric_limits<double>::infinity();
          int scale_idx = 0;
          int quant_idx = 0;
          int ext_filter = 0;
          int edge_clf = 0;
          std::vector<std::uint8_t> idx;
        } best_fit;
        for (int scale_idx = 0; scale_idx < 4; ++scale_idx) {
          for (int quant_idx = 0; quant_idx < 4; ++quant_idx) {
            const int quant =
                restoration::kCcsoQuantSz[scale_idx][quant_idx];
            if (quant == 0) continue;
            for (int ext_filter = 0; ext_filter < 7; ++ext_filter) {
              for (int edge_clf = 0; edge_clf < 2; ++edge_clf) {
                const int interval = 3 - edge_clf;
                const int cells = bands * interval * interval;
                std::vector<double> sum_e(
                    static_cast<std::size_t>(cells), 0.0);
                std::vector<double> sum_e2(
                    static_cast<std::size_t>(cells), 0.0);
                std::vector<double> count(
                    static_cast<std::size_t>(cells), 0.0);
                const int dy = restoration::kCcsoPos[ext_filter][0];
                const int dx = restoration::kCcsoPos[ext_filter][1];
                for (int y = 0; y < vis_h[plane]; ++y) {
                  for (int x = 0; x < vis_w[plane]; ++x) {
                    const int lx = x << sub;
                    const int ly = y << sub;
                    const int c = curr_luma.at(lx, ly);
                    const int px = std::clamp(lx + dx, 0, mi_luma_w - 1);
                    const int py = std::clamp(ly + dy, 0, mi_luma_h - 1);
                    const int mx = std::clamp(lx - dx, 0, mi_luma_w - 1);
                    const int my = std::clamp(ly - dy, 0, mi_luma_h - 1);
                    const int plus =
                        curr_luma.samples[static_cast<std::size_t>(py) *
                                              static_cast<std::size_t>(
                                                  luma_stride) +
                                          static_cast<std::size_t>(px)];
                    const int minus =
                        curr_luma.samples[static_cast<std::size_t>(my) *
                                              static_cast<std::size_t>(
                                                  luma_stride) +
                                          static_cast<std::size_t>(mx)];
                    const int cls0 =
                        restoration::ccso_score(plus - c, quant, edge_clf);
                    const int cls1 =
                        restoration::ccso_score(minus - c, quant, edge_clf);
                    const int band = c >> band_shift;
                    const std::size_t cell =
                        static_cast<std::size_t>(
                            (cls0 * interval + cls1) * bands + band);
                    const int src =
                        src_planes[plane][static_cast<std::size_t>(y) *
                                              static_cast<std::size_t>(
                                                  vis_w[plane]) +
                                          static_cast<std::size_t>(x)];
                    const int d = src - rec.at(x, y);
                    sum_e[cell] += d;
                    sum_e2[cell] += static_cast<double>(d) * d;
                    count[cell] += 1.0;
                  }
                }
                const int scale = scale_idx + 1;
                EdgeFit fit;
                fit.scale_idx = scale_idx;
                fit.quant_idx = quant_idx;
                fit.ext_filter = ext_filter;
                fit.edge_clf = edge_clf;
                fit.idx.assign(static_cast<std::size_t>(cells), 0);
                double total = 0.0;
                for (int cell = 0; cell < cells; ++cell) {
                  int best_idx = 0;
                  double best_sse = sum_e2[static_cast<std::size_t>(cell)];
                  for (int i = 0; i < 8; ++i) {
                    const double off =
                        static_cast<double>(restoration::kCcsoOffset[i]) *
                        scale;
                    const double sse =
                        sum_e2[static_cast<std::size_t>(cell)] -
                        2.0 * off * sum_e[static_cast<std::size_t>(cell)] +
                        off * off * count[static_cast<std::size_t>(cell)];
                    if (sse < best_sse - 1e-9) {
                      best_sse = sse;
                      best_idx = i;
                    }
                  }
                  fit.idx[static_cast<std::size_t>(cell)] =
                      static_cast<std::uint8_t>(best_idx);
                  total += best_sse;
                }
                fit.sse = total;
                if (fit.sse < best_fit.sse) best_fit = std::move(fit);
              }
            }
          }
        }

        StillCcsoPlane& plane_config = config.ccso.plane[plane];
        plane_config.enabled = true;
        plane_config.bo_only = false;
        plane_config.scale_idx = best_fit.scale_idx;
        plane_config.quant_idx = best_fit.quant_idx;
        plane_config.ext_filter = best_fit.ext_filter;
        plane_config.edge_clf = best_fit.edge_clf;
        plane_config.max_band_log2 = edge_bands_log2;
        const int cells = static_cast<int>(best_fit.idx.size());
        for (int k = 0; k < cells && k < 128; ++k) {
          plane_config.offset_idx[k] = best_fit.idx[static_cast<std::size_t>(k)];
        }
        restoration::CcsoParams& params = ccso_params.plane[plane];
        params.enabled = true;
        params.bo_only = false;
        params.scale_idx = plane_config.scale_idx;
        params.quant_idx = plane_config.quant_idx;
        params.ext_filter = plane_config.ext_filter;
        params.edge_clf = plane_config.edge_clf;
        params.max_band_log2 = edge_bands_log2;
        if (!restoration::ccso_set_offsets(
                params, plane_config.offset_idx,
                static_cast<std::size_t>(cells))) {
          error = "ccso_set_offsets failed";
          return false;
        }

        // Per-unit ccso_blk decision: keep a unit only when the fitted filter
        // lowers SSE on the visible samples of that unit.
        std::vector<double> unit_on(
            static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows),
            0.0);
        std::vector<double> unit_off(unit_on.size(), 0.0);
        const int interval = 3 - plane_config.edge_clf;
        const int quant = restoration::kCcsoQuantSz[plane_config.scale_idx]
                                                   [plane_config.quant_idx];
        const int dy = restoration::kCcsoPos[plane_config.ext_filter][0];
        const int dx = restoration::kCcsoPos[plane_config.ext_filter][1];
        for (int y = 0; y < vis_h[plane]; ++y) {
          for (int x = 0; x < vis_w[plane]; ++x) {
            const int lx = x << sub;
            const int ly = y << sub;
            const int c = curr_luma.at(lx, ly);
            const int px = std::clamp(lx + dx, 0, mi_luma_w - 1);
            const int py = std::clamp(ly + dy, 0, mi_luma_h - 1);
            const int mx = std::clamp(lx - dx, 0, mi_luma_w - 1);
            const int my = std::clamp(ly - dy, 0, mi_luma_h - 1);
            const int plus =
                curr_luma.samples[static_cast<std::size_t>(py) *
                                      static_cast<std::size_t>(luma_stride) +
                                  static_cast<std::size_t>(px)];
            const int minus =
                curr_luma.samples[static_cast<std::size_t>(my) *
                                      static_cast<std::size_t>(luma_stride) +
                                  static_cast<std::size_t>(mx)];
            const int cls0 =
                restoration::ccso_score(plus - c, quant, plane_config.edge_clf);
            const int cls1 =
                restoration::ccso_score(minus - c, quant, plane_config.edge_clf);
            if (cls0 >= interval || cls1 >= interval) continue;
            const int band = c >> band_shift;
            const int off =
                params.offsets[restoration::ccso_offset_index(band, cls0,
                                                              cls1)];
            const int src =
                src_planes[plane][static_cast<std::size_t>(y) *
                                      static_cast<std::size_t>(vis_w[plane]) +
                                  static_cast<std::size_t>(x)];
            const int base = rec.at(x, y);
            const int filtered =
                std::clamp(base + off, 0, 255);
            const int unit_col = x >> shift;
            const int unit_row = y >> shift;
            const std::size_t unit =
                static_cast<std::size_t>(unit_row) *
                    static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(unit_col);
            if (unit >= unit_on.size()) continue;
            const double d_off = src - base;
            const double d_on = src - filtered;
            unit_off[unit] += d_off * d_off;
            unit_on[unit] += d_on * d_on;
          }
        }
        ccso_edge_grids[plane].assign(unit_on.size(), 1);
        int enabled_units = 0;
        for (std::size_t u = 0; u < unit_on.size(); ++u) {
          if (unit_on[u] + 1e-9 < unit_off[u]) {
            ccso_edge_grids[plane][u] = 1;
            ++enabled_units;
          } else {
            ccso_edge_grids[plane][u] = 0;
          }
        }
        plane_config.enabled = enabled_units > 0;
        params.enabled = plane_config.enabled;
        ccso_grid_ptr[plane] = ccso_edge_grids[plane].data();
        ccso_blk_cols[plane] = cols;
        ccso_blk_rows[plane] = rows;
        ccso_params.blks[plane] = ccso_grid_ptr[plane];
        ccso_params.blk_cols[plane] = cols;
        ccso_params.blk_rows[plane] = rows;
      }
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

  // §7.20.2 CdefFrame: the post-CDEF/CCSO picture before loop restoration.
  // Wiener, PC-Wiener and GDF all sample this buffer; only LrFrame receives
  // their output.  Keep it separate from the reconstruction so the stages can
  // compose in normative order.
  Plane cdef_frame_planes[3];
  if (use_loop_restoration || options.enable_gdf) {
    for (int plane = 0; plane < 3; ++plane) {
      cdef_frame_planes[plane] = encoder.recon(plane);
    }
  }

  // §7.20 loop restoration (non-separable Wiener NS), after CDEF and CCSO.
  // The search minimises SSE against the source; per-unit filters at
  // LoopRestorationSize 64 (luma) / 32 (chroma) keep the tile syntax simple.
  restoration::classes::WienerDecision lr_decision;
  if (use_loop_restoration) {
    const int mi_luma_w = encoder.mi_cols() * 4;
    const int mi_luma_h = encoder.mi_rows() * 4;
    const int mi_chroma_w = mi_luma_w / 2;
    const int mi_chroma_h = mi_luma_h / 2;
    const int target_w[3] = {mi_luma_w, mi_chroma_w, mi_chroma_w};
    const int target_h[3] = {mi_luma_h, mi_chroma_h, mi_chroma_h};
    const int vis_w[3] = {static_cast<int>(picture.width),
                          static_cast<int>((picture.width + 1U) / 2U),
                          static_cast<int>((picture.width + 1U) / 2U)};
    const int vis_h[3] = {static_cast<int>(picture.height),
                          static_cast<int>((picture.height + 1U) / 2U),
                          static_cast<int>((picture.height + 1U) / 2U)};
    std::vector<std::uint8_t> target_storage[3];
    const std::uint8_t* target[3] = {nullptr, nullptr, nullptr};
    for (int plane = 0; plane < 3; ++plane) {
      const std::span<const std::uint8_t> src =
          plane == 0 ? picture.y : (plane == 1 ? picture.u : picture.v);
      target_storage[plane].assign(
          static_cast<std::size_t>(target_w[plane]) *
              static_cast<std::size_t>(target_h[plane]),
          0);
      for (int y = 0; y < vis_h[plane]; ++y) {
        std::copy_n(src.begin() +
                        static_cast<std::ptrdiff_t>(y) * vis_w[plane],
                    vis_w[plane],
                    target_storage[plane].begin() +
                        static_cast<std::ptrdiff_t>(y) * target_w[plane]);
      }
      target[plane] = target_storage[plane].data();
    }

    restoration::classes::WienerDecisionInput lr_input;
    for (int plane = 0; plane < 3; ++plane) {
      lr_input.core.plane[plane].curr =
          pre_cdef_planes[plane].samples.data();
      lr_input.core.plane[plane].curr_stride = pre_cdef_planes[plane].width;
      lr_input.core.plane[plane].cdef =
          cdef_frame_planes[plane].samples.data();
      lr_input.core.plane[plane].cdef_stride = cdef_frame_planes[plane].width;
      lr_input.target[plane] = target[plane];
      lr_input.target_stride[plane] = target_w[plane];
    }
    lr_input.core.mi_cols = encoder.mi_cols();
    lr_input.core.mi_rows = encoder.mi_rows();
    lr_input.core.sub_x = 1;
    lr_input.core.sub_y = 1;
    lr_input.core.bit_depth = 8;
    // §7.20.1: sbShift = Mi_Width_Log2[SbSize] and unitRow uses that SbSize.
    // An intra frame uses SbSize 128 under --still-sb128/sb256 (§5.18.1).
    const int lr_sb_size = (options.sb128 || options.sb256) ? 128 : 64;
    const int lr_sb_shift = lr_sb_size == 128 ? 5 : 4;
    lr_input.core.sb_shift = lr_sb_shift;
    lr_input.base_q_idx = options.base_q_idx;

    restoration::classes::WienerDecisionOptions lr_options;
    lr_options.allow_luma = options.enable_wiener;
    lr_options.allow_chroma = options.enable_chroma_lr;
    lr_options.allow_per_unit = true;
    lr_options.allow_frame_filters = false;
    lr_options.max_num_filter_classes = 1;
    lr_options.lambda_bits = 0.0;  // distortion-only integration gate
    // §5.18.7.11: a 128x128 SbSize caps the luma shift at 2 (unit 128) and
    // the chroma shift at 2 (unit 64 at 4:2:0); 64x64 keeps unit 64/32.
    // One unit per frame superblock keeps read_lr's per-SB visit shape.
    lr_options.forced_luma_shift = lr_sb_size == 128 ? 2 : 3;
    lr_options.forced_chroma_shift = lr_sb_size == 128 ? 2 : 3;
    const restoration::classes::WienerDecisionStatus lr_status =
        restoration::classes::wiener_decide(lr_input, lr_options, &lr_decision);
    if (lr_status != restoration::classes::WienerDecisionStatus::kOk) {
      error = std::string("wiener decision failed: ") +
              restoration::classes::wiener_decision_status_string(lr_status);
      return false;
    }

    bool lr_any = false;
    for (int plane = 0; plane < 3; ++plane) {
      lr_any = lr_any ||
               lr_decision.params.frame_type[plane] ==
                   restoration::kRestoreWienerNs;
    }
    if (lr_any) {
      restoration::LoopRestorationParams lr_params;
      restoration::classes::wiener_decision_to_loop_params(lr_decision,
                                                           &lr_params);
      Plane lr_planes[3];
      restoration::RestorationFrame lr_frame{};
      for (int plane = 0; plane < 3; ++plane) {
        const Plane& rec = encoder.recon(plane);
        lr_planes[plane].resize(rec.width, rec.height, 0);
        lr_frame.curr[plane].data =
            const_cast<std::uint8_t*>(pre_cdef_planes[plane].samples.data());
        lr_frame.curr[plane].stride = pre_cdef_planes[plane].width;
        lr_frame.cdef[plane].data = cdef_frame_planes[plane].samples.data();
        lr_frame.cdef[plane].stride = cdef_frame_planes[plane].width;
        lr_frame.lr[plane].data = lr_planes[plane].samples.data();
        lr_frame.lr[plane].stride = lr_planes[plane].width;
      }
      lr_frame.mi_cols = encoder.mi_cols();
      lr_frame.mi_rows = encoder.mi_rows();
      lr_frame.subsampling_x = 1;
      lr_frame.subsampling_y = 1;
      lr_frame.bit_depth = 8;
      lr_frame.sb_width = lr_sb_size;
      lr_frame.sb_height = lr_sb_size;
      lr_frame.sb_shift = lr_sb_shift;
      const restoration::RestorationStatus lr_filter_status =
          restoration::loop_restoration_frame(lr_frame, lr_params);
      if (lr_filter_status != restoration::RestorationStatus::kOk) {
        error = std::string("loop restoration failed: ") +
                restoration::restoration_status_string(lr_filter_status);
        return false;
      }
      for (int plane = 0; plane < 3; ++plane) {
        encoder.recon_mut(plane).samples.swap(lr_planes[plane].samples);
      }

      config.lr.enable = true;
      config.lr.uv_present = true;
      for (int plane = 0; plane < 3; ++plane) {
        StillLrPlane& dst = config.lr.plane[plane];
        dst.enabled = lr_decision.params.frame_type[plane] ==
                      restoration::kRestoreWienerNs;
        dst.shift = plane == 0 ? lr_decision.params.luma_shift
                               : lr_decision.params.chroma_shift;
        dst.unit_rows = lr_decision.unit_rows[plane];
        dst.unit_cols = lr_decision.unit_cols[plane];
        dst.unit_type = lr_decision.unit_type[plane];
        dst.unit_coeffs = lr_decision.unit_coeffs[plane];
        const std::size_t units = dst.unit_type.size();
        dst.unit_subset.assign(units, 0);
        dst.unit_uv_sym.assign(units, 0);
        dst.unit_merged.assign(units, 0);
        dst.unit_bank_k.assign(units, -1);
        if (lr_decision.plane[plane].units.size() != units) {
          error = "loop restoration unit plan size mismatch";
          return false;
        }
        for (std::size_t i = 0; i < units; ++i) {
          const restoration::classes::WienerUnitPlan& plan =
              lr_decision.plane[plane].units[i];
          dst.unit_subset[i] = static_cast<std::uint8_t>(plan.subset);
          dst.unit_uv_sym[i] = plan.uv_sym ? 1 : 0;
          dst.unit_merged[i] = plan.merged ? 1 : 0;
          dst.unit_bank_k[i] = static_cast<std::int8_t>(plan.use_bank_k);
        }
      }
    }
  }

  // §7.20.4 PC-Wiener (luma only), also after CDEF/CCSO.  The decision uses
  // the recovered classifier/filter bank and emits per-unit on/off symbols.
  if (options.enable_pc_wiener) {
    namespace pcw = av2enc::filters::pc_wiener_stream;
    const int mi_luma_w = encoder.mi_cols() * 4;
    const int mi_luma_h = encoder.mi_rows() * 4;
    const int vis_w = static_cast<int>(picture.width);
    const int vis_h = static_cast<int>(picture.height);
    std::vector<std::uint8_t> target(
        static_cast<std::size_t>(mi_luma_w) *
            static_cast<std::size_t>(mi_luma_h),
        0);
    for (int y = 0; y < vis_h; ++y) {
      const std::uint8_t* row =
          picture.y.data() + static_cast<std::ptrdiff_t>(y) * vis_w;
      std::copy_n(row, vis_w,
                  target.begin() + static_cast<std::ptrdiff_t>(y) * mi_luma_w);
      // Edge-replicate the MI padding so the per-unit SSE decision is not
      // biased by zeros outside the visible picture.
      std::fill(target.begin() + static_cast<std::ptrdiff_t>(y) * mi_luma_w +
                    vis_w,
                target.begin() + static_cast<std::ptrdiff_t>(y + 1) *
                                    mi_luma_w,
                vis_w > 0 ? row[vis_w - 1] : 0);
    }
    for (int y = vis_h; y < mi_luma_h; ++y) {
      const std::uint8_t* prev =
          target.data() + static_cast<std::ptrdiff_t>(y - 1) * mi_luma_w;
      std::copy_n(prev, mi_luma_w,
                  target.begin() + static_cast<std::ptrdiff_t>(y) * mi_luma_w);
    }

    pcw::DecisionPlanes planes;
    planes.width = mi_luma_w;
    planes.height = mi_luma_h;
    planes.bit_depth = 8;
    planes.base_q_idx = options.base_q_idx;
    planes.sub_x = 1;
    planes.sub_y = 1;
    // §5.18.7.11: SbSize 128 limits the luma unit shift to 2 (see the Wiener
    // decision above); 64x64 keeps the M1 64/32 units.
    const int pcw_sb_size = (options.sb128 || options.sb256) ? 128 : 64;
    planes.luma_shift = pcw_sb_size == 128 ? 2 : 3;
    planes.chroma_shift = pcw_sb_size == 128 ? 2 : 3;
    for (int plane = 0; plane < 3; ++plane) {
      planes.curr[plane] = pre_cdef_planes[plane].samples.data();
      planes.curr_stride[plane] = pre_cdef_planes[plane].width;
      planes.cdef[plane] = cdef_frame_planes[plane].samples.data();
      planes.cdef_stride[plane] = cdef_frame_planes[plane].width;
    }
    planes.target[0] = target.data();
    planes.target_stride[0] = mi_luma_w;

    // §7.20.1 LrTxSkip[row][col] = skip_flag || (eob == 0) for luma.  M1 has
    // no block-skip syntax, so it is exactly the per-4x4 eob==0 map.
    std::vector<std::uint8_t> lr_tx_skip(
        static_cast<std::size_t>(encoder.mi_cols()) *
            static_cast<std::size_t>(encoder.mi_rows()),
        0);
    for (const syntax::BlockSyntax& leaf : stream.blocks) {
      if (leaf.partition != static_cast<std::uint8_t>(PartitionType::kNone)) {
        continue;
      }
      const int w4 =
          partition::block_width(static_cast<BlockSize>(leaf.bsize)) / 4;
      const int h4 =
          partition::block_height(static_cast<BlockSize>(leaf.bsize)) / 4;
      const std::uint8_t value = leaf.eob[0] == 0 ? 1 : 0;
      for (int r = 0; r < h4; ++r) {
        const int row = leaf.mi_row + r;
        if (row >= encoder.mi_rows()) break;
        for (int c = 0; c < w4; ++c) {
          const int col = leaf.mi_col + c;
          if (col >= encoder.mi_cols()) break;
          lr_tx_skip[static_cast<std::size_t>(row) *
                         static_cast<std::size_t>(encoder.mi_cols()) +
                     static_cast<std::size_t>(col)] = value;
        }
      }
    }
    planes.lr_tx_skip = lr_tx_skip.data();
    planes.lr_tx_skip_stride = encoder.mi_cols();

    pcw::DecisionResult pc_result;
    if (!pcw::DecidePerUnitPcWiener(planes, 0.0, &pc_result, &error)) {
      error = "PC-Wiener decision failed: " + error;
      return false;
    }
    if (pc_result.active_units > 0) {
      const int unit_size =
          pcw::kRestorationTileSizeMax >> pc_result.params.luma_shift;
      Plane lr_luma;
      const Plane& rec = encoder.recon(0);
      lr_luma.resize(rec.width, rec.height, 0);
      const int filtered = pcw::ApplyPerUnitPcWiener(
          planes, pc_result.unit_type[0], unit_size,
          lr_luma.samples.data(), lr_luma.width);
      if (filtered <= 0) {
        error = "PC-Wiener application failed";
        return false;
      }
      encoder.recon_mut(0).samples.swap(lr_luma.samples);

      config.lr.enable = true;
      config.lr.uv_present = true;
      config.lr.pc_wiener = true;
      StillLrPlane& dst = config.lr.plane[0];
      dst.enabled = true;
      dst.shift = pc_result.params.luma_shift;
      dst.unit_rows = pcw::CountUnits(unit_size, mi_luma_h);
      dst.unit_cols = pcw::CountUnits(unit_size, mi_luma_w);
      dst.unit_type = pc_result.unit_type[0];
      // Plane 0 is PC-Wiener.  Chroma LR, when enabled above, keeps its
      // RESTORE_WIENER_NONSEP plan; the pair has no syntax conflict.
    }
  }

  // §7.20.5 GDF: last stage of loop restoration.  §7.20.1 runs the Wiener or
  // PC-Wiener filter for the block first and then conditionally applies GDF to
  // LrFrame; both sample CdefFrame (inside the stripe) and CurrFrame (outside),
  // never each other's output.  So GDF reads the post-CCSO CdefFrame snapshot
  // and rewrites the luma LrFrame produced by the restoration stage above.
  if (options.enable_gdf) {
    namespace gdf = av2enc::filters::gdf;
    const int gdf_w = encoder.mi_cols() * 4;
    const int gdf_h = encoder.mi_rows() * 4;
    const int vis_w = static_cast<int>(picture.width);
    const int vis_h = static_cast<int>(picture.height);
    std::vector<std::uint16_t> curr16(
        static_cast<std::size_t>(gdf_w) * static_cast<std::size_t>(gdf_h));
    std::vector<std::uint16_t> cdef16(curr16.size());
    std::vector<std::uint16_t> lr16(curr16.size());
    std::vector<std::uint16_t> original16(curr16.size());
    for (int y = 0; y < gdf_h; ++y) {
      const std::uint8_t* curr_row =
          pre_cdef_planes[0].samples.data() +
          static_cast<std::ptrdiff_t>(y) * pre_cdef_planes[0].width;
      const std::uint8_t* cdef_row =
          cdef_frame_planes[0].samples.data() +
          static_cast<std::ptrdiff_t>(y) * cdef_frame_planes[0].width;
      const std::uint8_t* lr_row =
          encoder.recon(0).samples.data() +
          static_cast<std::ptrdiff_t>(y) * encoder.recon(0).width;
      const std::uint8_t* src_row = nullptr;
      std::uint8_t edge = 0;
      if (y < vis_h) {
        src_row = picture.y.data() + static_cast<std::ptrdiff_t>(y) * vis_w;
        edge = vis_w > 0 ? src_row[vis_w - 1] : 0;
      } else {
        const int last = vis_h > 0 ? vis_h - 1 : 0;
        src_row = picture.y.data() + static_cast<std::ptrdiff_t>(last) * vis_w;
        edge = vis_w > 0 ? src_row[vis_w - 1] : 0;
      }
      for (int x = 0; x < gdf_w; ++x) {
        const std::size_t i =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(gdf_w) +
            static_cast<std::size_t>(x);
        curr16[i] = curr_row[x];
        cdef16[i] = cdef_row[x];
        lr16[i] = lr_row[x];
        original16[i] = x < vis_w ? src_row[x] : edge;
      }
    }

    gdf::GdfSourceFrame source;
    source.curr = curr16.data();
    source.cdef = cdef16.data();
    source.stride = gdf_w;
    gdf::GdfPlane original;
    original.data = original16.data();
    original.width = gdf_w;
    original.height = gdf_h;
    original.stride = gdf_w;
    // §5.18.7.9: SbSize is the frame superblock size (128x128 for an intra
    // frame under --still-sb128/sb256, 64x64 otherwise).  The sequence-level
    // gdf_unit_matches_sb_size is only coded when get_seq_sb_size() is 64 and
    // is inferred 0 here, so GdfBlkSize = Max(Block_Width[SbSize], 128).
    const int gdf_sb_size = (options.sb128 || options.sb256) ? 128 : 64;
    gdf::FrameGdfContext gdf_context;
    gdf_context.enable_gdf = true;
    gdf_context.gdf_unit_matches_sb_size = gdf_sb_size == 64;
    gdf_context.coded_lossless = false;
    gdf_context.single_picture_header_flag = true;
    gdf_context.sb_size = gdf_sb_size;
    gdf_context.width = gdf_w;
    gdf_context.height = gdf_h;
    gdf_context.mi_cols = encoder.mi_cols();
    gdf_context.mi_rows = encoder.mi_rows();
    gdf_context.disable_loopfilters_across_tiles = false;
    gdf_context.tile_cols = 1;
    gdf_context.tile_rows = 1;

    gdf::GdfDecisionOptions gdf_options;
    gdf_options.filter.base_q_idx = options.base_q_idx;
    gdf_options.filter.frame_is_intra = true;
    gdf_options.filter.bit_depth = 8;
    gdf_options.disable_cdf_update = !options.adaptive_cdf;
    gdf_options.lambda_bits = 0.0;
    gdf::GdfDecision gdf_decision;
    if (!gdf::DecideGdf(source, original, gdf_context, gdf_options,
                        &gdf_decision, &error)) {
      error = "GDF decision failed: " + error;
      return false;
    }
    if (gdf_decision.params.frame_enable) {
      gdf::FrameGdfParams params = gdf_decision.params;
      params.per_block = false;  // M1 codes no use_gdf symbols
      std::vector<std::uint16_t> out16(lr16);
      if (!gdf::ApplyGdfToFrame(source, gdf_options.filter, gdf_context, params,
                                {}, out16.data(), gdf_w, &error)) {
        error = "GDF apply failed: " + error;
        return false;
      }
      Plane& rec = encoder.recon_mut(0);
      for (int y = 0; y < gdf_h; ++y) {
        for (int x = 0; x < gdf_w; ++x) {
          rec.samples[static_cast<std::size_t>(y) *
                          static_cast<std::size_t>(rec.width) +
                      static_cast<std::size_t>(x)] =
              static_cast<std::uint8_t>(
                  out16[static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(gdf_w) +
                        static_cast<std::size_t>(x)]);
        }
      }
      config.gdf.enable = true;
      // The sequence row only carries gdf_unit_matches_sb_size for a 64x64
      // sequence SB; a 128x128 frame SbSize makes it inferred 0.  The header
      // writer keys the bit on this field, and gdf_params() re-derives the
      // same 128-sample GdfBlkSize from SbSize.
      config.gdf.unit_matches_sb_size = gdf_sb_size == 64;
      config.gdf.per_block = false;
      config.gdf.sb_size = gdf_sb_size;
      config.gdf.pic_qc_idx = params.pic_qc_idx;
      config.gdf.pic_scale_idx = params.pic_scale_idx;
      config.gdf.blk_size = params.blk_size;
      config.gdf.cols = params.cols;
      config.gdf.rows = params.rows;
      config.gdf.pix_scale = params.pix_scale;
    }
  }

  // Film grain (§5.14/§5.18.10).  The default signals a fixed two-point 4:2:0
  // model; --still-grain-adaptive estimates the model from the reconstruction
  // with the GrainStreamPlanner.  Synthesis is decoder-side, so the encoder
  // recon stays unfiltered either way.
  if (options.enable_grain) {
    namespace grain = av2enc::filters::grain;
    if (options.grain_adaptive) {
      // §5.18.10.1 infers apply_grain = 1 for a single-picture header, so the
      // estimator may not cancel.  min_activity = 0 keeps the planner's
      // validity checks but never selects "no grain" for content with a
      // non-zero scaling curve; a genuinely flat frame falls through to the
      // no-film-grain case below (no OBU, film_grain_params_present = 0).
      grain::GrainPlanOptions plan_options;
      plan_options.chroma_idc = grain::GrainChromaIdc::k420;
      plan_options.estimate.min_activity = 0.0;
      grain::GrainStreamPlanner planner(plan_options);
      grain::GrainFrameView view;
      view.y = encoder.recon(0).samples.data();
      view.u = encoder.recon(1).samples.data();
      view.v = encoder.recon(2).samples.data();
      view.width = static_cast<int>(picture.width);
      view.height = static_cast<int>(picture.height);
      view.y_stride = encoder.recon(0).width;
      view.c_stride = encoder.recon(1).width;
      grain::GrainFramePlan plan;
      if (!planner.PlanFrame(view, /*frame_index=*/0, /*tlayer_id=*/0,
                             /*mlayer_id=*/0, &plan, &error)) {
        error = "film grain planner failed: " + error;
        return false;
      }
      if (plan.apply_grain) {
        if (!plan.emit_model) {
          error = "film grain planner reused a slot without a model OBU";
          return false;
        }
        std::string grain_error;
        config.film_grain_obu =
            grain::MakeFilmGrainObu(plan.obu, &grain_error);
        if (config.film_grain_obu.empty()) {
          error = "film grain OBU failed: " + grain_error;
          return false;
        }
        config.film_grain_present = true;
        config.film_grain_id = plan.fgm_id;
        config.grain_seed = plan.grain_seed;
      }
    } else {
      grain::GrainModel model;
      model.chroma_scaling_from_luma = true;
      model.num_y_points = 2;
      model.point_y_value[0] = 0;
      model.point_y_value[1] = 128;
      model.point_y_scaling[0] = 64;
      model.point_y_scaling[1] = 88;
      model.point_y_increment_bits_minus_1 = 7;
      model.point_y_scaling_bits_minus_5 = 3;
      model.grain_scaling_minus_8 = 0;
      model.ar_coeff_lag = 2;
      model.bits_per_ar_coeff_y_minus_5 = 3;
      model.bits_per_ar_coeff_cb_minus_5 = 3;
      model.bits_per_ar_coeff_cr_minus_5 = 3;
      const int positions = 2 * model.ar_coeff_lag * (model.ar_coeff_lag + 1);
      for (int i = 0; i < positions; ++i) {
        const int tap = ((i % 3) - 1) * 8;
        model.ar_coeffs_y[i] = tap;
        model.ar_coeffs_cb[i] = tap / 2;
        model.ar_coeffs_cr[i] = tap / 2;
      }
      model.ar_coeffs_cb[positions] = 0;
      model.ar_coeffs_cr[positions] = 0;
      model.ar_coeff_shift_minus_6 = 1;
      model.overlap_flag = true;
      model.clip_to_restricted_range = false;
      model = grain::NormalizeModel(model, grain::GrainChromaIdc::k420);
      std::string grain_why;
      if (!grain::ValidateModel(model, grain::GrainChromaIdc::k420,
                                &grain_why)) {
        error = "film grain model is invalid: " + grain_why;
        return false;
      }
      grain::FilmGrainObu obu;
      obu.update_flags = 1;  // slot 0
      obu.chroma_idc = grain::GrainChromaIdc::k420;
      obu.models[0] = model;
      std::string grain_error;
      config.film_grain_obu = grain::MakeFilmGrainObu(obu, &grain_error);
      if (config.film_grain_obu.empty()) {
        error = "film grain OBU failed: " + grain_error;
        return false;
      }
      config.film_grain_present = true;
      config.film_grain_id = 0;
      config.grain_seed = 0;
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
  config.enable_idtx_intra = options.idtx;
  config.enable_intra_ist = options.ist;
  config.enable_inter_ist = false;
  config.enable_tcq = options.tcq;
  config.enable_cctx = options.cctx;
  config.cfl_ds_filter_index = options.cfl_ds_filter;
  config.enable_dip = options.enable_dip;
  config.enable_palette = options.palette;
  config.enable_mhccp = options.mhccp;
  config.allow_screen_content_tools = options.palette || options.ibc;
  config.force_integer_mv = true;
  config.allow_intrabc = options.ibc;
  config.allow_global_intrabc = options.ibc;
  config.allow_local_intrabc = false;
  config.enable_ibp = options.enable_ibp;
  config.enable_mrls = options.enable_mrl;
  config.tx_mode_select = options.tx_partition;
  config.reduced_tx_set = 0;
  config.reduced_tx_part_set = false;
  config.use_128x128_superblock = options.sb128;
  config.use_256x256_superblock = options.sb256;
  config.enable_ext_partitions = options.ext_partitions;
  config.enable_uneven_4way_partitions = options.ext_partitions;
  config.enable_sdp = options.sdp;
  config.qm = encoder.qm_state();
  if (options.user_qm) {
    const auto level = static_cast<int>(options.qm_level);
    qm::QmObuConfig obu;
    obu.bit_map = static_cast<std::uint16_t>(1U << level);
    obu.chroma_info_present = true;
    obu.levels[level].is_default = false;
    obu.levels[level].user = config.qm.user_qm[level];
    std::string qm_error;
    config.qm_obu = qm::make_quantizer_matrix_obu(obu, &qm_error);
    if (config.qm_obu.empty()) {
      error = "quantizer_matrix_obu failed: " + qm_error;
      return false;
    }
  }
  config.dc_ac_delta_q = seq_delta_q;
  config.frame_delta_q = frame_delta_q;
  config.apply_deblock_y_vert = options.enable_deblock;
  config.apply_deblock_y_horz = options.enable_deblock;
  config.apply_deblock_u = options.enable_deblock;
  config.apply_deblock_v = options.enable_deblock;
  config.per_sb_delta_q.segmentation_enabled =
      options.enable_segmentation || options.per_segment_q;
  config.per_sb_delta_q.delta_q_present = options.enable_delta_q;
  config.per_sb_delta_q.delta_q_res = options.enable_delta_q ? options.delta_q_res : 0;
  config.per_sb_delta_q.segmentation =
      options.per_segment_q ? &segment_params : nullptr;
  config.content_interpretation = options.content_interpretation;
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
  plan.sb_size = config.use_128x128_superblock ||
                         config.use_256x256_superblock
                     ? 128
                     : 64;
  if (ccso_ready) {
    for (int plane = 0; plane < 3; ++plane) {
      plan.ccso_blks[plane] = ccso_grid_ptr[plane];
      plan.ccso_blk_cols[plane] = ccso_blk_cols[plane];
      plan.ccso_blk_rows[plane] = ccso_blk_rows[plane];
    }
  }
  std::size_t tile_bytes = 0;
  if (!syntax::assemble_still_frame(plan, stream.blocks, stream.levels, config,
                                 result.frame, error, &tile_bytes,
                                 encoder.palette_results(),
                                 std::span<const std::uint16_t>(options.cdf_seed),
                                 &result.cdf_final)) {
    return false;
  }
  result.tile_bytes = static_cast<std::uint32_t>(tile_bytes);
  result.frame_bytes = static_cast<std::uint32_t>(result.frame.size());
  result.leaves = encoder.leaves();
  result.ibc_blocks = encoder.ibc_blocks();

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
  // Hand the exact header/filter configuration and the CDEF index grid to the
  // GPU orchestrator (it reuses them instead of re-deriving the parameters).
  result.config = config;
  if (cdef_ready) {
    result.cdef_idx = cdef_plan.cdef_idx;
    result.cdef_n64_cols = cdef_plan.n64_cols;
    result.cdef_n64_rows = cdef_plan.n64_rows;
  }
  if (ccso_ready) {
    for (int plane = 0; plane < 3; ++plane) {
      const std::uint8_t* grid = ccso_grid_ptr[plane];
      const int cols = ccso_blk_cols[plane];
      const int rows = ccso_blk_rows[plane];
      if (grid == nullptr || cols <= 0 || rows <= 0) {
        continue;
      }
      result.ccso_blks[plane].assign(
          grid, grid + static_cast<std::size_t>(cols) *
                         static_cast<std::size_t>(rows));
      result.ccso_blk_cols[plane] = cols;
      result.ccso_blk_rows[plane] = rows;
    }
  }
  return true;
}

}  // namespace av2enc::still
