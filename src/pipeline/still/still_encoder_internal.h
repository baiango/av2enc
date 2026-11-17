// Private declarations shared by the still-picture encoder translation units.
//
// Split verbatim out of still_encoder.cpp.  The class declaration and its doc
// comments stay here; the member definitions live in
//   still_encoder.cpp          constructor, partition-tree walk, frame assembly
//   still_encoder_decision.cpp mode/partition/IBC/TX-partition decisions
//   still_encoder_quant.cpp    quantisation, dequant, RDOQ, coefficient levels
//   still_encoder_recon.cpp    source access, reconstruction, deblock metadata
// The enclosing namespace gained a `detail` level so the definitions can be
// shared by several translation units; nothing else changed.
#pragma once

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
namespace detail {


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

inline int clip3(int low, int high, int value) {
  return value < low ? low : (value > high ? high : value);
}
inline int clip1(int value) { return clip3(0, 255, value); }

inline int av2_y_mode(intra::Mode mode) {
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

inline bool y_mode_is_directional(int mode) { return mode >= kV && mode <= kD67; }
inline bool y_mode_is_smooth(int mode) {
  return mode == kSmooth || mode == kSmoothV || mode == kSmoothH;
}

// Encoder-side analysis projection for the secondary transform.  The
// component's forward_secondary_coeffs_kernel() only covers the 4x4/8x8 case
// (it zeroes the whole block); for larger transforms the normative inverse
// touches only the top-left sub-block that the §9.7 scan map addresses, so
// project there and leave every other primary coefficient untouched.
inline bool project_ist_levels(const txv::TxParams& params,
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
          const dcdeltaq::QuantFrameState* delta_quant = nullptr);
  // Walks the decided partition tree, performs prediction/transform/
  // reconstruction, and emits the equivalent BlockSyntax stream.  The syntax
  // itself is written later by syntax::assemble_still_frame(), so this function
  // owns decisions and reconstruction only.
  bool encode(SyntaxStream* stream, std::string& error);
  const Plane& recon(int plane) const { return recon_[plane]; }
  Plane& recon_mut(int plane) { return recon_[plane]; }
  const qm::QmFrameState& qm_state() const { return qm_state_; }
  int leaves() const { return leaves_; }
  int ibc_blocks() const { return ibc_blocks_; }
  const std::vector<palette::PaletteResult>& palette_results() const {
    return palette_results_;
  }
  const std::vector<std::uint32_t>& palette_index_by_leaf() const {
    return palette_index_by_leaf_;
  }
  const std::vector<StillResult::IbcPlan>& ibc_by_leaf() const {
    return ibc_by_leaf_;
  }
  int mi_cols() const { return mi_cols_; }
  int mi_rows() const { return mi_rows_; }
  int deblock_cols() const { return deblock_cols_; }
  int deblock_rows() const { return deblock_rows_; }
  const deblock::DeblockMi* deblock_mi(int plane) const {
    return deblock_mi_[static_cast<std::size_t>(plane)].data();
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
      decoded_[static_cast<std::size_t>(plane)].reset(
          plane == 0 ? mi_cols_ : mi_cols_ / 2,
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
                    const ChromaInfo& inherited_chroma);
  // Extended-partition variant of collect_node().  Chroma ownership comes from
  // pe::PartitionNode::has_chroma/chroma_offset/chroma, so a shared 8x16/16x8
  // middle block (HORZ_3@8x32 / VERT_3@32x8) and the first children of a 4-way
  // split are handled exactly like sub-8 leaves.
  bool collect_node_ext(const pe::PartitionNode& node);
  // Append one SDP luma/chroma tree (preorder).  The sdp split node codes no
  // partition syntax and carries the two 64x64 tree roots as children.
  bool collect_sdp_node(const sdpp::Node& node);
  // ---------------------------------------------------------------------
  // Source / reconstruction helpers
  // ---------------------------------------------------------------------
  std::uint8_t source_sample(int plane, int x, int y) const;
  void fill_source(int plane, int x0, int y0, int w, int h,
                   std::vector<std::uint8_t>& out) const;
  int count_top_right(int plane, int abs_x4, int abs_y4, int w4) const;
  int count_bottom_left(int plane, int abs_x4, int abs_y4, int h4) const;
  // ---------------------------------------------------------------------
  // Prediction
  // ---------------------------------------------------------------------
  bool edge_filter_enabled() const {
    return options_.enable_intra_edge_filter &&
           std::getenv("AV2ENC_STILL_NO_EDGE") == nullptr;
  }

  bool prepare_edges(int plane, int x0, int y0, int w, int h,
                     intra::IntraEdges& edges) const;
  bool neighborhood_smooth(int plane, int r, int c, int w4, int h4,
                           bool& above_smooth, bool& left_smooth) const;
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
                     int block_mi_row = -1, int block_mi_col = -1);
  // §5.20.7.9: DIP is an alternative to ordinary DC at the same YMode, not a
  // new mode.  Evaluates the normative candidate set and keeps it only when it
  // beats the incumbent prediction by SSE.
  bool evaluate_luma_dip(int x0, int y0, int w, int h, ModeChoice& best);
  // §5.20.7.9 DIP prediction for a luma DC block.
  dip::Block make_dip_block(int x0, int y0, int w, int h);
  bool predict_luma_dip(int x0, int y0, int w, int h, int dip_mode,
                        bool dip_transpose, std::vector<std::uint8_t>& out);
  // §7.13.2.1/.7/.8 MRL prediction for a directional luma candidate.
  // mrl::prepare_edges needs the extra reference lines, so it cannot reuse the
  // w+h-sample IntraEdges prepared for MrlIndex = 0.
  bool predict_luma_mrl(int x0, int y0, int w, int h, intra::Mode mode,
                        int angle_delta, int mrl_index, bool mrl_sec_index,
                        std::vector<std::uint8_t>& out, int block_mi_row = -1,
                        int block_mi_col = -1);
  std::int64_t sse_against_source(int plane, int x0, int y0, int w, int h,
                                  const std::vector<std::uint8_t>& pred) const;
  bool choose_luma_mode(int x0, int y0, int w, int h, ModeChoice& best);
  std::int64_t block_variance_cost_at(int mi_row, int mi_col,
                                      BlockSize size) const;
  // ---------------------------------------------------------------------
  // Transform / reconstruction
  // ---------------------------------------------------------------------

  // The transform component bakes the QP-90 forward quantiser entry (528)
  // into TxParams.  For any other base_q_idx, scale quant_factor by the ratio
  // of the normative qlookup step, so forward quantisation and the inverse
  // dequant below agree with the step the decoder derives from base_q_idx.
  void apply_qindex_step(txv::TxParams& params) const;
  // §7.14.2/§7.14.4 quantizer for one adjusted-raster coefficient: the DC
  // quantizer at (0,0) and the AC quantizer elsewhere.  Without delta-Q both
  // are the frame qlookup step.
  int coefficient_q(int plane, int index) const;
  // §7.14.4 dequant + optional IST inverse + inverse transform with the
  // frame's qlookup step.  With QM enabled the per-coefficient
  // q2 = Round2(q * m, 5) replaces the scalar 528 so the encoder
  // reconstruction matches what the decoder derives.
  bool dequant_inverse_qindex(const std::int32_t* levels,
                              const txv::TxParams& params, int plane, int eob,
                              int y_mode, int y_angle_delta, int sec_tx_type,
                              int stx_set, int mrl_index,
                              std::int32_t* dequant,
                              std::int16_t* residual);
  bool compute_tx(int plane, int x0, int y0, int cw, int ch,
                  const std::vector<std::uint8_t>& pred, TxPlan& plan,
                  bool choose_tx_type = false, int y_mode = 0,
                  int y_angle_delta = 0, int mrl_index = 0);
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

  LumaSnapshot take_luma_snapshot(int x0, int y0, int w, int h) const;
  void restore_luma_snapshot(const LumaSnapshot& s);
  // Encodes every transform part of one partition, feeding each part's
  // reconstruction into the next part's prediction (the decoder does the same
  // through make_intra_prediction/store_tx_info per TX block).
  bool encode_luma_parts(int r, int c, int w, int h,
                         partition::BlockSize size, const ModeChoice& y,
                         int partition, std::vector<TxPlan>* plans,
                         std::vector<syntax::LumaTxPart>* parts,
                         double* block_sse);
  // Tries NONE/SPLIT/HORZ/VERT by exact reconstruction SSE + a small syntax
  // cost, then re-runs the winner so the recon/decoded state is committed.
  bool choose_luma_tx_partition(int r, int c, int w, int h,
                                partition::BlockSize size,
                                const ModeChoice& y,
                                std::vector<TxPlan>* plans,
                                std::vector<syntax::LumaTxPart>* parts,
                                int* partition_out);
  bool compute_chroma_cctx(int x0, int y0, int cw, int ch,
                           const std::vector<std::uint8_t>& u_pred,
                           const std::vector<std::uint8_t>& v_pred,
                           TxPlan& u_plan, TxPlan& v_plan);
  // Append one plane's levels to the flat array in *scan order*: the syntax
  // writer consumes levels_[offset + c] as scan position c.
  void append_levels(const TxPlan& plan);
  void write_recon(int plane, int x0, int y0, int cw, int ch,
                   const TxPlan& plan);
  // ---------------------------------------------------------------------
  // Intra block copy (§5.20.5.3/.4, §7.13.3.18)
  // ---------------------------------------------------------------------
  ibc::FrameView ibc_frame_view() const;
  // Does copying `bv` reproduce the source exactly on all three planes?
  // `sse` receives the luma error even when chroma differs.
  bool ibc_copy_sse(const ibc::FrameView& frame, int r, int c,
                    partition::BlockSize size, ibc::BlockVector bv,
                    int visible_w, int visible_h, std::int64_t* sse);
  // Screen content usually has few distinct luma values; only then is the
  // bounded window search worth its cost on natural photographs.
  bool ibc_low_color_rank(int x0, int y0, int w, int h) const;
  struct IbcChoice {
    bool found = false;
    ibc::BlockVector bv{};
    int stack_index = -1;  // >= 0 => NEARMV with that RefMvIdx
  };

  // Choose an exact-copy block vector (skip_flag = 1 codes no residual, so a
  // non-exact copy could only reduce quality).  Stack candidates come first
  // (NEARMV is cheaper); then periodic offsets and a bounded local window
  // (NEWMV when the vector is not a stack entry).
  bool choose_ibc(int r, int c, partition::BlockSize size, IbcChoice& out);
  // ---------------------------------------------------------------------
  // Coding a block
  // ---------------------------------------------------------------------
  bool encode_block(int r, int c, partition::BlockSize size,
                    std::size_t index, bool has_chroma,
                    const ChromaInfo& chroma, int tree_type = 0);
  // §5.20.6.3: with TX partitioning the deblocking TX grid is per transform
  // part; SubPuSize stays Max_Tx_Size_Rect[MiSize] like the decoder.
  void fill_deblock_mi_parts(int mi_row, int mi_col, partition::BlockSize size,
                             int cw, int ch,
                             const std::vector<syntax::LumaTxPart>& parts,
                             int partition, int chroma_tx_size, bool has_chroma,
                             int chroma_mi_row, int chroma_mi_col,
                             partition::BlockSize chroma_bsize,
                             bool do_luma = true);
  void make_cfl_block(int plane, int cx0, int cy0, int cw, int ch,
                      cfl::Block& block);
  bool choose_uv_mode(int cx0, int cy0, int cw, int ch, int y_mode,
                      int y_angle_delta, UvChoice& best);
  void fill_deblock_mi(int mi_row, int mi_col, partition::BlockSize size,
                       int cw, int ch, int luma_tx_size,
                       int chroma_tx_size, bool has_chroma,
                       int chroma_mi_row, int chroma_mi_col,
                       partition::BlockSize chroma_bsize, bool do_luma = true);
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
  // Leaf-order decision plan exposed through StillResult (preorder, one entry
  // per leaf, matching the GPU FramePlan block list).
  std::vector<std::uint32_t> palette_index_by_leaf_;
  std::vector<StillResult::IbcPlan> ibc_by_leaf_;

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

}  // namespace detail

// The unit was a single anonymous namespace inside av2enc::still, so these names
// used to resolve from av2enc::still itself (the intra/chroma component suites
// spell them that way).  Keep exactly that spelling.
using detail::BlockSize;
using detail::ChromaInfo;
using detail::DecodedMap;
using detail::Encoder;
using detail::FrameGeometry;
using detail::ModeChoice;
using detail::PartitionType;
using detail::Plane;
using detail::SyntaxStream;
using detail::TxPlan;
using detail::UvChoice;
using detail::av2_y_mode;
using detail::clip1;
using detail::clip3;
using detail::project_ist_levels;
using detail::y_mode_is_directional;
using detail::y_mode_is_smooth;
}  // namespace av2enc::still
