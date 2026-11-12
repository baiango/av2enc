#include "codec/inter/mv_stack/gate/gate_stream.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

#include "core/bitstream/bitwriter.h"
#include "core/bitstream/basic_headers.h"
#include "codec/inter/mv_stack/frame_flags.h"
#include "codec/inter/mv_stack/mv_stack_full.h"
#include "codec/inter/mc/mc.h"
#include "pipeline/still/basic_cdf_tables.h"
#include "codec/inter/advanced/video/advanced_cdf_tables.h"
#include "codec/inter/advanced/video/advanced_syntax.h"
#include "pipeline/video/video_cdf_tables.h"
#include "codec/partition/tree/partition.h"
#include "codec/entropy/av2/av2_entropy.h"

namespace av2enc::inter::mv_stack_full::gate {
namespace {

namespace video_adv = av2enc::video::adv;
using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;
using partition::BlockSize;

constexpr int kOrderHintBits = 4;
constexpr int kNumRefFrames = 4;
constexpr int kMaxDrlBitsMinus1 = 4;
constexpr int kSbSize = 64;

// ---------------------------------------------------------------------------
// Bit helpers (the same primitives the M2 header writer uses)
// ---------------------------------------------------------------------------

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) ++leading_zeros;
  writer.put_bits(0U, leading_zeros);
  writer.put_bit(true);
  const std::uint64_t payload =
      static_cast<std::uint64_t>(value) - ((1ULL << leading_zeros) - 1ULL);
  writer.put_bits(payload, leading_zeros);
}

void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
  assert(n >= 2U && value < n);
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

unsigned bit_width(std::uint32_t value) {
  unsigned bits = 0;
  while (value != 0) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

std::uint32_t ceil_log2(std::uint32_t value) {
  if (value <= 1U) return 0U;
  std::uint32_t bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

std::uint32_t mi_cols_of(int width) { return 2U * ((static_cast<std::uint32_t>(width) + 7U) >> 3U); }
std::uint32_t mi_rows_of(int height) { return 2U * ((static_cast<std::uint32_t>(height) + 7U) >> 3U); }

void write_tile_info(BitWriter& writer, int width, int height) {
  const std::uint32_t sb_cols = (mi_cols_of(width) + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows_of(height) + 15U) >> 4U;
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) writer.put_bit(false);
  if (sb_rows > 1U) writer.put_bit(false);
}

// §5.4.6, with seq_enabled_motion_modes[DELTAWARP] = 1.
std::vector<std::uint8_t> make_sequence_header(int width, int height) {
  BitWriter writer;
  put_uvlc(writer, 0);      // seq_header_id
  writer.put_bits(0U, 5U);  // seq_profile_idc
  writer.put_bit(false);    // single_picture_header_flag
  writer.put_bits(7U, 5U);  // seq_level_idx
  writer.put_bit(false);    // seq_tier
  put_uvlc(writer, 0);      // chroma_format_idc
  put_uvlc(writer, 1);      // bit_depth_idc
  writer.put_bits(0U, 3U);  // seq_lcr_id
  writer.put_bit(false);    // still_picture
  writer.put_bits(0U, 2U);  // max_tlayer_id
  writer.put_bits(0U, 3U);  // max_mlayer_id
  writer.put_bit(true);     // monotonic_output_order_flag
  unsigned width_bits = bit_width(static_cast<std::uint32_t>(width) - 1U);
  unsigned height_bits = bit_width(static_cast<std::uint32_t>(height) - 1U);
  if (width_bits == 0) width_bits = 1;
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(width) - 1U, width_bits);
  writer.put_bits(static_cast<std::uint64_t>(height) - 1U, height_bits);
  writer.put_bit(false);  // seq_cropping_window_present_flag
  writer.put_bit(false);  // seq_initial_display_delay_present_flag
  writer.put_bit(false);  // decoder_model_info_present_flag
  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(false);  // use_128x128_superblock
  writer.put_bit(false);  // enable_sdp
  writer.put_bit(false);  // enable_ext_partitions
  writer.put_bit(false);  // reduce_pb_aspect_ratio
  writer.put_bit(false);  // enable_ext_seg
  writer.put_bit(false);  // seq_seg_info_present_flag
  writer.put_bit(false);  // enable_dip
  writer.put_bit(true);   // enable_intra_edge_filter
  writer.put_bit(false);  // enable_mrls
  writer.put_bit(true);   // enable_cfl_intra
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);  // enable_mhccp
  writer.put_bit(false);  // enable_ibp

  SequenceMotionModes motion;
  motion.seq_enabled_motion_modes[kMotionLocalWarp] = true;
  motion.seq_enabled_motion_modes[kMotionDeltaWarp] = true;
  motion.seq_frame_motion_modes_present_flag = true;
  motion.enable_six_param_warp_delta = false;
  std::string error;
  if (!write_sequence_motion_modes(&writer, motion, &error)) return {};

  writer.put_bit(false);  // enable_masked_compound
  writer.put_bit(false);  // enable_ref_frame_mvs
  writer.put_bits(static_cast<std::uint64_t>(kOrderHintBits) - 1U, 4U);
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder
  writer.put_bit(true);   // explicit_ref_frame_map
  writer.put_bit(true);   // explicit_num_ref_frames
  writer.put_bits(static_cast<std::uint64_t>(kNumRefFrames) - 1U, 4U);
  writer.put_bits(0U, 3U);  // long_term_frame_id_bits
  put_ns(writer, 5U, kMaxDrlBitsMinus1);
  writer.put_bit(false);  // allow_frame_max_drl_bits
  put_ns(writer, 3U, 0U);
  writer.put_bit(false);  // allow_frame_max_bvp_drl_bits
  writer.put_bits(0U, 2U);  // num_same_ref_compound
  writer.put_bit(false);  // enable_tip
  writer.put_bit(false);  // enable_mv_traj
  writer.put_bit(false);  // enable_bawp
  writer.put_bit(false);  // enable_cwp
  writer.put_bit(false);  // enable_imp_msk_bld
  writer.put_bit(false);  // enable_df_sub_pu
  writer.put_bits(0U, 2U);  // enable_opfl_refine
  writer.put_bit(false);  // enable_refinemv
  writer.put_bit(false);  // enable_bru
  writer.put_bit(false);  // enable_adaptive_mvd
  writer.put_bit(false);  // enable_mvd_sign_derive
  writer.put_bit(false);  // enable_flex_mvres
  writer.put_bit(false);  // enable_global_motion
  writer.put_bit(false);  // enable_short_refresh_frame_flags
  writer.put_bit(false);  // seq_choose_screen_content_tools
  writer.put_bit(false);  // seq_force_screen_content_tools
  writer.put_bit(false);  // enable_fsc
  writer.put_bit(false);  // enable_idtx_intra
  writer.put_bit(false);  // enable_intra_ist
  writer.put_bit(false);  // enable_inter_ist
  writer.put_bit(true);   // enable_chroma_dctonly
  writer.put_bit(false);  // enable_inter_ddt
  writer.put_bit(false);  // reduced_tx_part_set
  writer.put_bit(false);  // enable_cctx
  writer.put_bit(false);  // enable_tcq
  writer.put_bit(false);  // enable_parity_hiding
  writer.put_bit(false);  // enable_avg_cdf
  writer.put_bit(false);  // separate_uv_delta_q
  writer.put_bit(true);   // equal_ac_dc_q
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q
  writer.put_bit(false);  // uv_ac_delta_q_enabled
  writer.put_bit(false);  // disable_loopfilters_across_tiles
  writer.put_bit(false);  // enable_cdef
  writer.put_bit(false);  // enable_gdf
  writer.put_bit(false);  // enable_restoration
  writer.put_bit(false);  // enable_ccso
  writer.put_bit(false);  // cdef_on_skip_txfm_always_on
  writer.put_bit(true);   // cdef_on_skip_txfm_disabled
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2
  writer.put_bit(false);  // seq_tile_info_present_flag
  writer.put_bit(false);  // film_grain_params_present
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

struct GateFrameHeader {
  bool keyframe = false;
  bool frame_enabled_motion_modes[kMotionModes] = {false, false, false, false,
                                                   false};
  std::uint32_t order_hint = 0;
  std::uint32_t refresh_frame_flags = 1;
  int num_total_refs = 0;
  std::uint8_t ref_frame_idx[7] = {};
  bool reference_select = false;
  bool allow_warpmv_mode = false;
};

std::size_t write_frame_header_syntax(BitWriter& writer,
                                      const GateFrameHeader& h, int width,
                                      int height) {
  const std::size_t start = writer.bit_size();
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  if (!h.keyframe) writer.put_bit(true);  // frame_is_inter
  writer.put_bit(true);  // immediate_output_frame
  writer.put_bit(false);  // frame_size_override_flag
  writer.put_bits(h.order_hint, kOrderHintBits);
  if (h.keyframe) {
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
    writer.put_bits(h.refresh_frame_flags, kNumRefFrames);
    writer.put_bit(true);  // frame_explicit_ref_frame_map
    writer.put_bits(static_cast<std::uint64_t>(h.num_total_refs), 3U);
    for (int i = 0; i < h.num_total_refs; ++i) {
      writer.put_bits(h.ref_frame_idx[i],
                      ceil_log2(static_cast<std::uint32_t>(kNumRefFrames)));
    }
    writer.put_bit(false);  // allow_intrabc
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTAP
  }
  if (!h.keyframe) {
    // §5.18.2 after read_interpolation_filter(): one bit per sequence-enabled
    // motion mode, in INTERINTRA..EXTENDWARP order.
    writer.put_bit(h.frame_enabled_motion_modes[kMotionLocalWarp]);
    writer.put_bit(h.frame_enabled_motion_modes[kMotionDeltaWarp]);
  }
  writer.put_bit(true);  // disable_cdf_update
  write_tile_info(writer, width, height);
  writer.put_bits(90U, 8U);  // base_q_idx
  writer.put_bit(false);     // segmentation_enabled
  writer.put_bit(false);     // using_qmatrix
  writer.put_bit(false);     // delta_q_present
  writer.put_bit(false);     // apply_deblocking_filter[0]
  writer.put_bit(false);     // apply_deblocking_filter[1]
  writer.put_bit(false);     // tx_mode_select = 0
  if (!h.keyframe) {
    writer.put_bit(h.reference_select);
    writer.put_bit(false);  // skip_mode_present
    // §5.18.2: present iff !FrameIsIntra && frame_enabled_motion_modes[DELTAWARP]
    // (DELTAWARP is sequence-enabled for this stream).
    writer.put_bit(h.allow_warpmv_mode);
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_frame_obu(const GateFrameHeader& header,
                                         int width, int height,
                                         std::span<const std::uint8_t> tile) {
  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_syntax(writer, header, width, height);
  writer.align_to_byte();
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile.begin(), tile.end());
  const ObuType type =
      header.keyframe ? ObuType::kClosedLoopKey : ObuType::kRegularTileGroup;
  return av2enc::basic::make_obu(ObuHeader{type}, payload);
}

std::vector<std::uint8_t> make_td() {
  return av2enc::basic::make_obu(ObuHeader{ObuType::kTemporalDelimiter},
                              std::span<const std::uint8_t>{});
}

// ---------------------------------------------------------------------------
// Tile writer
// ---------------------------------------------------------------------------

struct Cdfs {
  std::uint16_t do_split[2][64][3];
  std::uint16_t is_inter[4][3];
  std::uint16_t skip[6][3];
  std::uint16_t single_mode[5][4];
  std::uint16_t drl_mode[3][5][3];
  std::uint16_t single_ref[3][6][3];
  std::uint16_t shell_set[3];
  std::uint16_t shell6_class0[9];
  std::uint16_t shell6_class1[9];
  std::uint16_t shell_last_two[3];
  std::uint16_t offset_low[2][3];
  std::uint16_t offset_class2[3];
  std::uint16_t offset_other[16][3];
  std::uint16_t col_mv_greater[2][3];
  std::uint16_t col_mv_index[4][3];
  video_adv::AdvancedCdfState advanced;
};

template <typename T, std::size_t N>
void copy_array(T (&dst)[N], const T (&src)[N]) {
  std::memcpy(dst, src, sizeof(dst));
}

void init_cdfs(Cdfs& c) {
  copy_array(c.do_split, av2enc::basic::kDoSplitCdf);
  copy_array(c.is_inter, av2enc::video::kIsInterCdf);
  copy_array(c.skip, av2enc::video::kSkipCdf);
  copy_array(c.single_mode, av2enc::video::kSingleModeCdf);
  copy_array(c.drl_mode, av2enc::video::kDrlModeCdf);
  copy_array(c.single_ref, av2enc::video::kSingleRefCdf);
  copy_array(c.shell_set, av2enc::video::kJointShellSetCdf);
  copy_array(c.shell6_class0, av2enc::video::kJointShell6Class0Cdf);
  copy_array(c.shell6_class1, av2enc::video::kJointShell6Class1Cdf);
  copy_array(c.shell_last_two, av2enc::video::kJointShellLastTwoClassesCdf);
  copy_array(c.offset_low, av2enc::video::kShellOffsetLowClassCdf);
  copy_array(c.offset_class2, av2enc::video::kShellOffsetClass2Cdf);
  copy_array(c.offset_other, av2enc::video::kShellOffsetOtherClassCdf);
  copy_array(c.col_mv_greater, av2enc::video::kColMvGreaterCdf);
  copy_array(c.col_mv_index, av2enc::video::kColMvIndexCdf);
  c.advanced.reset();
}

struct LeafPlan {
  bool is_warp = false;
  int single_mode = 0;  // 0 NEARMV, 1 GLOBALMV, 2 NEWMV
  int drl_index = 0;
  Mv mvd;
  int warp_idx = 3;
  int y_mode = kYModeNearMv;
  int motion_mode = kMotionSimple;
  Mv mv;
};

struct TileResult {
  std::vector<std::uint8_t> payload;
  std::vector<std::uint8_t> recon_y;
  std::vector<std::uint8_t> recon_u;
  std::vector<std::uint8_t> recon_v;
  StreamCoverage coverage;
};

class TileWriter {
 public:
  TileWriter(int width, int height, const PlaneRef& ref, bool allow_warpmv,
             int num_total_refs, bool simple_plan = false)
      : width_(width),
        allow_warpmv_(allow_warpmv),
        num_total_refs_(num_total_refs),
        simple_plan_(simple_plan),
        ref_(ref) {
    geometry_.width = width;
    geometry_.height = height;
    geometry_.sb_size = kSbSize;
    geometry_.has_chroma = true;
    geometry_.chroma_subsampling_x = true;
    geometry_.chroma_subsampling_y = true;
    mi_cols_ = partition::mi_cols(geometry_);
    mi_rows_ = partition::mi_rows(geometry_);
    sb_mi_ = partition::sb_size_mi(geometry_);
    sizes_.reset(geometry_);
    const std::size_t cells =
        static_cast<std::size_t>(mi_rows_) * static_cast<std::size_t>(mi_cols_);
    mi_.assign(cells, MiCell{});
    nbr_.assign(cells, video_adv::NeighborMi{});
    is_inter_.assign(cells, 0);
    skip_.assign(cells, 0);
    ymode_.assign(cells, 0);
    init_cdfs(cdfs_);
    buffer_.resize(8U * 1024U * 1024U);
    enc_.init_symbol(buffer_.data(), buffer_.size());
    enc_.disable_cdf_update = 1;
    const int cw = width / 2;
    const int ch = height / 2;
    result_.recon_y.assign(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0);
    result_.recon_u.assign(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch), 0);
    result_.recon_v.assign(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch), 0);
  }

  bool run(std::string& error) {
    const int sb_cols = mi_cols_ / sb_mi_;
    const int sb_rows = mi_rows_ / sb_mi_;
    if (mi_cols_ % sb_mi_ != 0 || mi_rows_ % sb_mi_ != 0) {
      error = "gate tile writer requires dimensions divisible by 64";
      return false;
    }
    int leaf_index = 0;
    for (int sbr = 0; sbr < sb_rows; ++sbr) {
      for (int sbc = 0; sbc < sb_cols; ++sbc, ++leaf_index) {
        const int r = sbr * sb_mi_;
        const int c = sbc * sb_mi_;
        const partition::PartitionSet allowed =
            partition::init_allowed_partitions(geometry_, r, c,
                                               BlockSize::k64x64, true, false);
        if (!allowed.is_allowed(partition::PartitionType::kNone)) {
          error = "gate tile writer requires PARTITION_NONE at every superblock";
          return false;
        }
        const partition::PartitionContexts ctx =
            partition::partition_contexts(geometry_, r, c, BlockSize::k64x64,
                                          sizes_);
        if (!sym(&cdfs_.do_split[0][ctx.do_split][0], 2, 0)) {
          error = error_;
          return false;
        }
        if (!write_leaf(leaf_index, sbr, sbc, error)) return false;
        sizes_.fill_block(r, c, BlockSize::k64x64);
      }
    }
    std::size_t bytes = 0;
    const av2::Status status = enc_.exit_symbol(&bytes);
    if (status != av2::Status::kOk) {
      error = std::string("exit_symbol failed: ") + av2::status_name(status);
      return false;
    }
    result_.payload.assign(
        buffer_.begin(),
        buffer_.begin() + static_cast<std::ptrdiff_t>(bytes));
    return true;
  }

  TileResult take() { return std::move(result_); }

 private:
  bool sym(std::uint16_t* cdf, int n, int symbol) {
    const av2::Status status = enc_.encode_symbol(cdf, n, symbol);
    if (status != av2::Status::kOk) {
      error_ = std::string("encode_symbol failed: ") +
               av2::status_name(status);
      return false;
    }
    return true;
  }

  bool raw_bit(bool bit) {
    if (enc_.encode_bool(bit) != av2::Status::kOk) {
      error_ = "encode_bool failed";
      return false;
    }
    return true;
  }
  bool literal(int bits, int value) {
    for (int i = bits - 1; i >= 0; --i) {
      if (!raw_bit(((value >> i) & 1) != 0)) return false;
    }
    return true;
  }
  bool ns_bits(std::uint32_t n, std::uint32_t value) {
    unsigned w = 0;
    while ((1U << w) < n) ++w;
    const std::uint32_t m = (1U << w) - n;
    if (value < m) {
      return literal(static_cast<int>(w - 1U), static_cast<int>(value));
    }
    const std::uint32_t x = value + m;
    if (!literal(static_cast<int>(w - 1U), static_cast<int>(x >> 1U))) {
      return false;
    }
    return raw_bit((x & 1U) != 0U);
  }
  static int floor_log2_int(int v) {
    int n = 0;
    while (v > 1) {
      v >>= 1;
      ++n;
    }
    return n;
  }
  // §5.20.7.20 read_mv() for MvPrecision = 1/8 pel.
  bool write_mv(int dy, int dx) {
    const int ay = std::abs(dy);
    const int ax = std::abs(dx);
    const int shell = ay + ax;
    if (shell >= (1 << 17)) {
      error_ = "MVD shell out of range";
      return false;
    }
    const int cls = shell == 0 ? 0 : floor_log2_int(shell);
    const int base = cls == 0 ? 0 : (1 << cls);
    const int off = shell - base;
    const int setsym = cls >= 8 ? 1 : 0;
    const int class_sym = cls >= 8 ? cls - 8 : cls;
    if (!sym(&cdfs_.shell_set[0], 2, setsym)) return false;
    std::uint16_t* class_cdf =
        setsym ? &cdfs_.shell6_class1[0] : &cdfs_.shell6_class0[0];
    if (!sym(class_cdf, 8, class_sym)) return false;
    if (setsym && class_sym == 7) {
      if (!sym(&cdfs_.shell_last_two[0], 2, cls - 15)) return false;
    }
    if (cls < 2) {
      if (!sym(&cdfs_.offset_low[cls][0], 2, off)) return false;
    } else if (cls == 2) {
      if (off == 0) {
        if (!sym(&cdfs_.offset_class2[0], 2, 0)) return false;
      } else {
        if (!sym(&cdfs_.offset_class2[0], 2, 1)) return false;
        if (off == 1) {
          if (!raw_bit(false)) return false;
        } else {
          if (!raw_bit(true)) return false;
          if (!raw_bit(off == 3)) return false;
        }
      }
    } else {
      for (int i = 0; i < cls; ++i) {
        if (!sym(&cdfs_.offset_other[i][0], 2, (off >> i) & 1)) return false;
      }
    }
    if (shell > 0) {
      const int max_pair = shell >> 1;
      bool skip_col_bit = false;
      int col = 0;
      if (ax == max_pair && ((shell & 1) == 0)) {
        skip_col_bit = true;
        col = max_pair;
      } else if (ax <= max_pair) {
        col = ax;
      } else {
        col = ay;
      }
      if (max_pair > 0) {
        const int max_bits = std::min<int>(max_pair, 2);
        const int prefix_col = std::min<int>(col, 2);
        for (int i = 0; i < max_bits; ++i) {
          const int greater = prefix_col > i ? 1 : 0;
          if (!sym(&cdfs_.col_mv_greater[i][0], 2, greater)) return false;
          if (greater == 0) break;
        }
        if (max_pair > 2 && prefix_col == 2) {
          if (!ns_bits(static_cast<std::uint32_t>(max_pair - 1),
                       static_cast<std::uint32_t>(col - 2))) {
            return false;
          }
        }
      }
      if (!skip_col_bit) {
        const int index = ax <= max_pair ? 0 : 1;
        if (!sym(&cdfs_.col_mv_index[std::min(cls, 3)][0], 2, index)) {
          return false;
        }
      }
    }
    if (dy != 0 && !raw_bit(dy < 0)) return false;
    if (dx != 0 && !raw_bit(dx < 0)) return false;
    return true;
  }

  std::size_t cell(int r, int c) const {
    return static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols_) +
           static_cast<std::size_t>(c);
  }
  bool inside(int r, int c) const {
    return r >= 0 && r < mi_rows_ && c >= 0 && c < mi_cols_;
  }

  int neighbor_positions(int pos[2][2], int r, int c, int w4, int h4) const {
    int n = 0;
    const auto add = [&](int nr, int nc) {
      if (n < 2 && inside(nr, nc)) {
        pos[n][0] = nr;
        pos[n][1] = nc;
        ++n;
      }
    };
    add(r + h4 - 1, c - 1);
    add(r - 1, c + w4 - 1);
    add(r, c - 1);
    add(r - 1, c);
    return n;
  }

  int is_inter_context(int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    const auto intra_at = [&](int i) {
      return is_inter_[cell(pos[i][0], pos[i][1])] == 0;
    };
    if (n == 2) return (intra_at(0) && intra_at(1)) ? 3 : (intra_at(0) || intra_at(1));
    if (n == 1) return 2 * (intra_at(0) ? 1 : 0);
    return 0;
  }

  int skip_context(int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    int ctx = 0;
    for (int i = 0; i < n; ++i) {
      ctx += skip_[cell(pos[i][0], pos[i][1])] != 0 ? 1 : 0;
    }
    return ctx;
  }

  // §7.11.2 single-prediction new-MV context, including the neighbour-ref
  // comparison that §7.11.3 requires.
  int new_mv_context(int r, int c, int w4, int h4) const {
    int new_mv_count = 0;
    const bool sb_border = (r & (sb_mi_ - 1)) == 0;
    const auto scan = [&](int dr, int dc) -> int {
      const int mr = r + dr;
      const int mc = c + dc;
      if (!inside(mr, mc)) return 0;
      const std::size_t idx = cell(mr, mc);
      if (is_inter_[idx] == 0) return 0;
      if (mi_[idx].ref[0] != 0) return 0;
      if (has_newmv_for_list(ymode_[idx], 0)) {
        new_mv_count = std::min(3, new_mv_count + 1);
      }
      return 1;
    };
    const int left_a = scan(h4 - 1, -1);
    const int above_a = scan(-1, w4 - 1);
    const int left_b = scan(0, -1);
    const int above_b = (w4 >= (sb_border ? 4 : 2)) ? scan(-1, 0) : 0;
    const int nearest = ((above_a || above_b) ? 1 : 0) +
                        ((left_a || left_b) ? 1 : 0);
    return nearest + (new_mv_count > 0 ? 2 : 0);
  }

  int count_refs(int ref, int r, int c, int w4, int h4) const {
    int pos[2][2] = {};
    const int n = neighbor_positions(pos, r, c, w4, h4);
    int count = 0;
    for (int i = 0; i < n; ++i) {
      const std::size_t idx = cell(pos[i][0], pos[i][1]);
      for (int list = 0; list < 2; ++list) {
        if (mi_[idx].ref[list] == ref) ++count;
      }
    }
    return count;
  }

  // §5.20.7.10 read_single_ref(): NumTotalRefs - 1 single_ref bits.
  bool code_single_ref(int want, int r, int c, int w4, int h4) {
    if (num_total_refs_ <= 1) return true;
    for (int ref = 0; ref < num_total_refs_ - 1; ++ref) {
      const int this_count = count_refs(ref, r, c, w4, h4);
      int next_count = 0;
      for (int i = ref + 1; i < num_total_refs_; ++i) {
        next_count += count_refs(i, r, c, w4, h4);
      }
      int ctx;
      if (this_count == next_count) {
        ctx = 1;
      } else if (this_count < next_count) {
        ctx = 0;
      } else {
        ctx = 2;
      }
      const int bit = ref == want ? 1 : 0;
      if (!sym(&cdfs_.single_ref[ctx][ref][0], 2, bit)) return false;
      if (bit) return true;
    }
    return want == num_total_refs_ - 1;
  }

  bool code_drl_index(int ctx, int index) {
    const int m = kMaxDrlBitsMinus1 + 1;
    if (index < 0 || index > m) return false;
    for (int idx = 0; idx < m; ++idx) {
      const int mode = idx < index ? 1 : 0;
      if (!sym(&cdfs_.drl_mode[std::min(idx, 2)][ctx][0], 2, mode)) return false;
      if (mode == 0) break;
    }
    return true;
  }

  bool write_leaf(int leaf_index, int sbr, int sbc, std::string& error) {
    const int w4 = sb_mi_;
    const int h4 = sb_mi_;
    const int r = sbr * sb_mi_;
    const int c = sbc * sb_mi_;

    // Build the §7.12.2 stack from the already-coded neighbours.
    StackParams sp;
    sp.grid = MiGrid{mi_cols_, mi_rows_, mi_.data()};
    sp.mi_row = r;
    sp.mi_col = c;
    sp.bw4 = w4;
    sp.bh4 = h4;
    sp.sb_h4 = sb_mi_;
    sp.ref_frame[0] = 0;
    sp.ref_frame[1] = kRefNone;
    sp.derive_wrl = w4 * 4 >= 8 && h4 * 4 >= 8;
    sp.max_drl_bits_minus_1 = kMaxDrlBitsMinus1;
    sp.gm_params_present = true;
    Stack stack;
    BuildState build_state;
    const Status stack_status = build(sp, &stack, &build_state);
    if (stack_status != Status::kOk) {
      error = "stack build failed: " + std::string(status_string(stack_status));
      return false;
    }
    WarpContextResult wctx;
    if (compute_warp_context(sp, &wctx) != Status::kOk) {
      error = "warp context failed";
      return false;
    }

    // Plan the leaf.  The pattern is deterministic and documented in the
    // header; the choice of DRL index is made so that at least one leaf names a
    // §7.12.2.20 candidate.
    LeafPlan plan;
    // Deterministic gate plan.  `allow_warpmv_mode` is 1 in frame 3's header,
    // so every block codes `is_warp` (which also validates the §7.11.4
    // WarpMvCount CDF row on the decoder).  NEARMV blocks deliberately select
    // the last stack entry whenever the §7.12.2.20 extra search appended one,
    // so the signalled index names a global-MV / MVP-combination candidate and
    // the decoder must reproduce the same ordering.  Actual WARPMV/DELTAWARP
    // blocks are opt-in (GATE_WARP_BLOCKS); see README.md.
    const bool warp_blocks = std::getenv("GATE_WARP_BLOCKS") != nullptr;
    const bool want_warp =
        !simple_plan_ && allow_warpmv_ && warp_blocks && (leaf_index % 7) == 3;
    const bool want_global = simple_plan_ || (leaf_index % 7) == 5;
    const bool want_new_mv =
        !simple_plan_ && ((leaf_index % 7) == 0 || (leaf_index % 7) == 1);
    if (want_warp) {
      if (stack.num_warp_found == 0) {
        error = "WARPMV leaf without a warp candidate";
        return false;
      }
      plan.is_warp = true;
      plan.warp_idx = 0;
      plan.y_mode = kYModeWarpNewMv;
      plan.motion_mode = kMotionDeltaWarp;
      plan.single_mode = 0;
      plan.drl_index = 0;
      plan.mvd = Mv{0, 0};
      plan.mv = Mv{0, 0};
      ++result_.coverage.warpmv;
    } else if (want_global) {
      plan.single_mode = 1;
      plan.y_mode = kYModeGlobalMv;
      plan.motion_mode = kMotionSimple;
      plan.mv = Mv{0, 0};
      ++result_.coverage.global_mv;
    } else if (want_new_mv) {
      const int ty = 8 * ((leaf_index % 3) - 1);
      const int tx = 8 * (((leaf_index / 3) % 3) - 1);
      plan.single_mode = 2;
      plan.y_mode = kYModeNewMv;
      plan.motion_mode = kMotionSimple;
      plan.drl_index = 0;
      const Mv pred = stack.count > 0 ? stack.cand[0].mv[0] : Mv{};
      plan.mv = Mv{ty, tx};
      plan.mvd = Mv{ty - pred.y, tx - pred.x};
    } else {
      plan.single_mode = 0;
      plan.y_mode = kYModeNearMv;
      plan.motion_mode = kMotionSimple;
      // Prefer a §7.12.2.20 candidate (index >= scanned_count) over the
      // nearest scanned one.  GATE_DRL_INDEX=<k> pins one index for the
      // byte-level ordering probe (see README.md).
      int drl = stack.count > stack.scanned_count ? stack.count - 1 : 0;
      if (const char* d = std::getenv("GATE_DRL_INDEX")) {
        const int requested = std::atoi(d);
        drl = requested < stack.count ? requested : 0;
      }
      plan.drl_index = drl;
      plan.mv = stack.cand[drl].mv[0];
      if (drl >= stack.scanned_count) {
        ++result_.coverage.near_extra;
      } else {
        ++result_.coverage.near_neighbour;
      }
    }
    if (std::getenv("GATE_TRACE_STACK") != nullptr) {
      std::fprintf(stderr,
                   "stack leaf=%d rc=(%d,%d) count=%d scanned=%d nearest=%d "
                   "derived=%d warpf=%d warpins=%d prune=%d drl=%d "
                   "mode=%d mv=(%d,%d)\n",
                   leaf_index, r, c, stack.count, stack.scanned_count,
                   stack.num_nearest, stack.derived_count,
                   stack.num_warp_found, build_state.trace.warp_inserted,
                   stack.prune_count, plan.drl_index, plan.y_mode, plan.mv.y,
                   plan.mv.x);
      for (int i = 0; i < stack.count; ++i) {
        std::fprintf(stderr,
                     "  cand[%d] mv=(%d,%d) w=%d off=(%d,%d)\n", i,
                     stack.cand[i].mv[0].y, stack.cand[i].mv[0].x,
                     stack.cand[i].weight, stack.cand[i].row_offset,
                     stack.cand[i].col_offset);
      }
    }
    result_.coverage.max_stack = std::max(result_.coverage.max_stack, stack.count);
    result_.coverage.max_warp_found =
        std::max(result_.coverage.max_warp_found, stack.num_warp_found);
    if (stack.count > stack.scanned_count) ++result_.coverage.mvp_combinations;
    if (allow_warpmv_) ++result_.coverage.is_warp_bits;

    // is_inter / skip.
    if (!sym(&cdfs_.is_inter[is_inter_context(r, c, w4, h4)][0], 2, 1)) {
      error = error_;
      return false;
    }
    if (!sym(&cdfs_.skip[skip_context(r, c, w4, h4)][0], 2, 1)) {
      error = error_;
      return false;
    }

    // read_ref_frames(): reference_select == 0, so there is no comp_mode bit,
    // but NumTotalRefs > 1 still codes read_single_ref().  RefFrame[0] is
    // always index 0 (the map's first entry).
    if (!code_single_ref(0, r, c, w4, h4)) {
      error = error_;
      return false;
    }

    const int mode_ctx = new_mv_context(r, c, w4, h4);
    video_adv::NeighborGrid nbr_grid{mi_cols_, mi_rows_, nbr_.data()};
    const video_adv::ModeContext mode_context = video_adv::compute_mode_context(
        nbr_grid, r, c, static_cast<int>(BlockSize::k64x64), sb_mi_, 0, -1);
    video_adv::AdvancedBlockCoding block;
    block.mi_size = static_cast<int>(BlockSize::k64x64);
    block.new_mv_ctx = mode_ctx;
    block.warp_mv_ctx = std::min(wctx.warp_mv_count, 4);
    block.allow_warpmv_mode = allow_warpmv_;
    block.force_integer_mv = false;
    block.has_chroma = true;
    block.is_warp = plan.is_warp;
    block.warp_mv = false;  // WARP_NEWMV: allow_warpmv_mode gates is_warp
    block.single_mode = plan.single_mode;
    block.drl_index = plan.drl_index;
    block.mvd_y = plan.mvd.y;
    block.mvd_x = plan.mvd.x;
    block.warp_idx = plan.warp_idx;
    block.warpmv_with_mvd = false;
    block.motion.local_allowed = plan.is_warp && wctx.warp_sample_found[0];
    block.motion.local_warp_ctx = mode_context.local_warp_ctx;
    block.motion.extend_allowed = false;
    block.motion.inter_intra_allowed = false;
    block.motion.wedge_allowed = false;
    block.motion.use_local_warp = false;
    block.motion.use_extend_warp = false;
    block.motion.inter_intra = false;
    block.y_mode = plan.y_mode;
    block.motion_mode = plan.motion_mode;

    video_adv::BaselineInterWriter baseline;
    baseline.ctx = this;
    baseline.single_mode = [](void* ctx, int new_mv_ctx, int symbol) {
      TileWriter* self = static_cast<TileWriter*>(ctx);
      return self->sym(&self->cdfs_.single_mode[new_mv_ctx][0], 3, symbol);
    };
    baseline.drl_index = [](void* ctx, int new_mv_ctx, int index) {
      TileWriter* self = static_cast<TileWriter*>(ctx);
      return self->code_drl_index(new_mv_ctx, index);
    };
    baseline.mvd = [](void* ctx, int dy, int dx) {
      TileWriter* self = static_cast<TileWriter*>(ctx);
      return self->write_mv(dy, dx);
    };

    if (!video_adv::write_advanced_inter_block(enc_, cdfs_.advanced, block, baseline,
                                           &error)) {
      return false;
    }

    // Reconstruct: skip_flag = 1, so the prediction is the reconstruction.
    mc::RefFrame ref;
    ref.plane[0] = mc::PlaneView{ref_.y, ref_.stride_y};
    ref.plane[1] = mc::PlaneView{ref_.u, ref_.stride_uv};
    ref.plane[2] = mc::PlaneView{ref_.v, ref_.stride_uv};
    ref.mi_cols = mi_cols_;
    ref.mi_rows = mi_rows_;
    mc::MutablePlane dst[3];
    const std::size_t luma_stride = static_cast<std::size_t>(width_);
    const int cw = width_ / 2;
    const std::size_t chroma_stride = static_cast<std::size_t>(cw);
    dst[0] = mc::MutablePlane{result_.recon_y.data() +
                                  static_cast<std::size_t>(r) * 4 * luma_stride +
                                  static_cast<std::size_t>(c) * 4,
                              width_};
    dst[1] = mc::MutablePlane{result_.recon_u.data() +
                                  static_cast<std::size_t>(r) * 2 * chroma_stride +
                                  static_cast<std::size_t>(c) * 2,
                              cw};
    dst[2] = mc::MutablePlane{result_.recon_v.data() +
                                  static_cast<std::size_t>(r) * 2 * chroma_stride +
                                  static_cast<std::size_t>(c) * 2,
                              cw};
    const mc::BlockRect rect{c * 4, r * 4, 64, 64};
    const mc::McStatus mc_status = mc::predict_block_420(
        ref, rect, plan.mv, mc::InterpFilter::kEighttap, dst);
    if (mc_status != mc::McStatus::kOk) {
      error = std::string("predict_block_420 failed: ") +
              mc::mc_status_string(mc_status);
      return false;
    }

    // Publish the leaf in the mode-info grid.
    for (int yy = 0; yy < h4; ++yy) {
      for (int xx = 0; xx < w4; ++xx) {
        const std::size_t idx = cell(r + yy, c + xx);
        MiCell& m = mi_[idx];
        m.ref[0] = 0;
        m.ref[1] = kRefNone;
        m.mv[0][0] = plan.mv.y;
        m.mv[0][1] = plan.mv.x;
        m.mv[1][0] = plan.mv.y;
        m.mv[1][1] = plan.mv.x;
        m.sub_mv[0][0] = plan.mv.y;
        m.sub_mv[0][1] = plan.mv.x;
        m.sub_mv[1][0] = plan.mv.y;
        m.sub_mv[1][1] = plan.mv.x;
        m.is_inter = 1;
        m.written = 1;
        m.w4 = 16;
        m.h4 = 16;
        m.col_base = static_cast<std::int16_t>(c);
        m.row_base = static_cast<std::int16_t>(r);
        m.y_mode = static_cast<std::uint8_t>(plan.y_mode);
        m.motion_mode = static_cast<std::uint8_t>(plan.motion_mode);
        m.cwp = kCwpEqual;
        m.warp[0] = adv::WarpParams{};
        m.warp[1] = adv::WarpParams{};
        is_inter_[idx] = 1;
        skip_[idx] = 1;
        ymode_[idx] = static_cast<std::uint8_t>(plan.y_mode);
        video_adv::NeighborMi& n = nbr_[idx];
        n.written = true;
        n.is_inter = true;
        n.ref[0] = 0;
        n.ref[1] = -1;
        n.y_mode = plan.y_mode;
        n.motion_mode = plan.motion_mode;
      }
    }
    return true;
  }

  int width_ = 0;
  bool allow_warpmv_ = false;
  int num_total_refs_ = 1;
  bool simple_plan_ = false;
  PlaneRef ref_{};
  partition::FrameGeometry geometry_{};
  int mi_cols_ = 0;
  int mi_rows_ = 0;
  int sb_mi_ = 16;
  partition::MiSizeGrid sizes_;
  Cdfs cdfs_{};
  av2::Encoder enc_{};
  std::vector<std::uint8_t> buffer_;
  std::string error_;
  std::vector<MiCell> mi_;
  std::vector<video_adv::NeighborMi> nbr_;
  std::vector<std::uint8_t> is_inter_;
  std::vector<std::uint8_t> skip_;
  std::vector<std::uint8_t> ymode_;
  TileResult result_;
};

bool all_identity(const adv::WarpParams& p) { return p.identity(); }

}  // namespace

bool build_stream(const StreamInput& input, StreamOutput& output,
                  std::string& error) {
  error.clear();
  output = StreamOutput{};
  if (input.width <= 0 || input.height <= 0 || input.width % 64 != 0 ||
      input.height % 64 != 0) {
    error = "gate requires dimensions divisible by 64";
    return false;
  }
  if (input.key_tile.empty() || input.p1_tile.empty() || input.key_recon.y == nullptr ||
      input.p1_recon.y == nullptr) {
    error = "missing M2 tile payloads or reference reconstructions";
    return false;
  }

  // Frame 2: a single-reference frame written by this module.  It exists so the
  // multi-reference frame has a second, distinct reference (P2) instead of the
  // key frame; it uses the same stack machinery with allow_warpmv_mode = 0.
  TileWriter writer2(input.width, input.height, input.key_recon,
                     /*allow_warpmv=*/false, /*num_total_refs=*/1,
                     /*simple_plan=*/true);
  if (!writer2.run(error)) return false;
  TileResult tile2 = writer2.take();

  // Frame 3: the multi-reference frame.  RefFrame[0] is slot 2 (the frame just
  // coded), so the reconstruction uses tile2's output.
  PlaneRef frame2_ref;
  frame2_ref.y = tile2.recon_y.data();
  frame2_ref.u = tile2.recon_u.data();
  frame2_ref.v = tile2.recon_v.data();
  frame2_ref.stride_y = input.width;
  frame2_ref.stride_uv = input.width / 2;
  TileWriter writer3(input.width, input.height, frame2_ref,
                     /*allow_warpmv=*/true, /*num_total_refs=*/3);
  if (!writer3.run(error)) return false;
  TileResult tile3 = writer3.take();
  (void)all_identity;

  const std::vector<std::uint8_t> seq =
      make_sequence_header(input.width, input.height);
  if (seq.empty()) {
    error = "sequence header write failed";
    return false;
  }
  const std::vector<std::uint8_t> td = make_td();

  GateFrameHeader key_header;
  key_header.keyframe = true;
  key_header.order_hint = 0;
  key_header.refresh_frame_flags = (1U << kNumRefFrames) - 1U;
  GateFrameHeader p1_header;
  p1_header.order_hint = 1;
  p1_header.refresh_frame_flags = 1U << 1;  // refresh slot 1
  p1_header.num_total_refs = 1;
  p1_header.ref_frame_idx[0] = 0;
  p1_header.reference_select = false;
  p1_header.frame_enabled_motion_modes[kMotionLocalWarp] = true;
  p1_header.frame_enabled_motion_modes[kMotionDeltaWarp] = true;
  p1_header.allow_warpmv_mode = false;
  GateFrameHeader p2_header;
  p2_header.order_hint = 2;
  p2_header.refresh_frame_flags = 1U << 2;  // refresh slot 2
  p2_header.num_total_refs = 1;
  p2_header.ref_frame_idx[0] = 0;
  p2_header.reference_select = false;
  p2_header.frame_enabled_motion_modes[kMotionLocalWarp] = true;
  p2_header.frame_enabled_motion_modes[kMotionDeltaWarp] = true;
  p2_header.allow_warpmv_mode = false;
  GateFrameHeader p3_header;
  p3_header.order_hint = input.order_hint;
  p3_header.refresh_frame_flags = 1U << 3;
  p3_header.num_total_refs = 3;
  p3_header.ref_frame_idx[0] = 2;  // RefFrame[0] -> P2
  p3_header.ref_frame_idx[1] = 1;  // RefFrame[1] -> P1
  p3_header.ref_frame_idx[2] = 0;  // RefFrame[2] -> key
  p3_header.reference_select = false;
  p3_header.frame_enabled_motion_modes[kMotionLocalWarp] = true;
  p3_header.frame_enabled_motion_modes[kMotionDeltaWarp] = true;
  p3_header.allow_warpmv_mode = true;


  const std::vector<std::uint8_t> key_obu =
      make_frame_obu(key_header, input.width, input.height, input.key_tile);
  const std::vector<std::uint8_t> p1_obu =
      make_frame_obu(p1_header, input.width, input.height, input.p1_tile);
  const std::vector<std::uint8_t> p2_obu =
      make_frame_obu(p2_header, input.width, input.height, tile2.payload);
  const std::vector<std::uint8_t> p3_obu =
      make_frame_obu(p3_header, input.width, input.height, tile3.payload);

  output.units.clear();
  const auto make_unit = [&](const std::vector<std::uint8_t>& frame,
                             bool with_seq) {
    std::vector<std::uint8_t> unit;
    unit.insert(unit.end(), td.begin(), td.end());
    if (with_seq) unit.insert(unit.end(), seq.begin(), seq.end());
    unit.insert(unit.end(), frame.begin(), frame.end());
    return unit;
  };
  output.units.push_back(make_unit(key_obu, true));
  output.units.push_back(make_unit(p1_obu, false));
  output.units.push_back(make_unit(p2_obu, false));
  output.units.push_back(make_unit(p3_obu, false));
  output.recon2_y = std::move(tile2.recon_y);
  output.recon2_u = std::move(tile2.recon_u);
  output.recon2_v = std::move(tile2.recon_v);
  output.p2_tile_bytes = static_cast<std::uint32_t>(tile2.payload.size());
  output.p3_tile_bytes = static_cast<std::uint32_t>(tile3.payload.size());
  output.recon_y = std::move(tile3.recon_y);
  output.recon_u = std::move(tile3.recon_u);
  output.recon_v = std::move(tile3.recon_v);
  output.coverage = tile3.coverage;
  return true;
}

}  // namespace av2enc::inter::mv_stack_full::gate
