#include "codec/inter/compound/compound_stream.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>

#include "core/bitstream/bitwriter.h"
#include "core/bitstream/basic_headers.h"
#include "codec/inter/temporal/opfl.h"
#include "pipeline/video/video_headers.h"
#include "pipeline/video/video_syntax.h"
#include "codec/inter/multiref/compound_predict.h"
#include "codec/partition/tree/partition.h"

namespace av2enc::video::compound {
namespace {

using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;
using multiref::CompoundModeParams;
using multiref::CompoundPrediction;
namespace mc = av2enc::inter::mc;
namespace temporal = av2enc::inter::temporal;

bool is_inter_ref_value(int ref) { return ref >= 0 && ref < 8; }

constexpr int kOrderHintBits = 4;
constexpr int kNumRefFrames = 2;
constexpr int kFutureOrderHint = 4;

void put_uvlc(BitWriter& writer, std::uint32_t value) {
  unsigned leading_zeros = 0;
  while (value >= ((1ULL << (leading_zeros + 1U)) - 1ULL)) {
    ++leading_zeros;
    assert(leading_zeros < 32U);
  }
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

std::uint32_t mi_cols_of(int width) {
  return 2U * ((static_cast<std::uint32_t>(width) + 7U) >> 3U);
}
std::uint32_t mi_rows_of(int height) {
  return 2U * ((static_cast<std::uint32_t>(height) + 7U) >> 3U);
}

void write_tile_info(BitWriter& writer, int width, int height) {
  const std::uint32_t sb_cols = (mi_cols_of(width) + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows_of(height) + 15U) >> 4U;
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) writer.put_bit(false);
  if (sb_rows > 1U) writer.put_bit(false);
}

// §5.4.6 sequence_inter_config(), parameterised just enough for this gate.
std::vector<std::uint8_t> make_sequence_header(const CompoundStreamInput& in) {
  const CompoundFrameConfig& cfg = in.config;
  BitWriter writer;
  put_uvlc(writer, 0);              // seq_header_id
  writer.put_bits(0U, 5U);          // seq_profile_idc
  writer.put_bit(false);            // single_picture_header_flag
  writer.put_bits(7U, 5U);          // seq_level_idx = 5.1
  writer.put_bit(false);            // seq_tier
  put_uvlc(writer, 0);              // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);              // bit_depth_idc = 8-bit
  writer.put_bits(0U, 3U);          // seq_lcr_id
  writer.put_bit(false);            // still_picture
  writer.put_bits(0U, 2U);          // max_tlayer_id
  writer.put_bits(0U, 3U);          // max_mlayer_id
  writer.put_bit(true);             // monotonic_output_order_flag
  unsigned width_bits = bit_width(static_cast<std::uint32_t>(in.width) - 1U);
  unsigned height_bits = bit_width(static_cast<std::uint32_t>(in.height) - 1U);
  if (width_bits == 0) width_bits = 1;
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(in.width) - 1U, width_bits);
  writer.put_bits(static_cast<std::uint64_t>(in.height) - 1U, height_bits);
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
  for (int mode = 1; mode < 5; ++mode) {
    writer.put_bit(false);  // seq_enabled_motion_modes[mode]
  }
  writer.put_bit(cfg.enable_masked_compound);
  writer.put_bit(false);  // enable_ref_frame_mvs
  writer.put_bits(static_cast<std::uint64_t>(kOrderHintBits) - 1U, 4U);
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder
  writer.put_bit(true);   // explicit_ref_frame_map
  writer.put_bit(true);   // explicit_num_ref_frames
  writer.put_bits(static_cast<std::uint64_t>(kNumRefFrames) - 1U, 4U);
  writer.put_bits(0U, 3U);  // long_term_frame_id_bits
  put_ns(writer, 5U,
         static_cast<std::uint32_t>(cfg.max_drl_bits_minus_1));
  writer.put_bit(false);  // allow_frame_max_drl_bits
  put_ns(writer, 3U, 0U);  // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);  // allow_frame_max_bvp_drl_bits
  writer.put_bits(static_cast<std::uint64_t>(cfg.num_same_ref_compound()), 2U);
  writer.put_bit(false);  // enable_tip
  writer.put_bit(false);  // enable_mv_traj
  writer.put_bit(false);  // enable_bawp
  writer.put_bit(cfg.enable_cwp);
  writer.put_bit(false);  // enable_imp_msk_bld (the gate keeps this off)
  writer.put_bit(false);  // enable_df_sub_pu
  writer.put_bits(static_cast<std::uint64_t>(cfg.opfl_refine_type), 2U);
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
  writer.put_bit(false);     // uv_ac_delta_q_enabled
  writer.put_bit(false);     // disable_loopfilters_across_tiles
  writer.put_bit(false);     // enable_cdef
  writer.put_bit(false);     // enable_gdf
  writer.put_bit(false);     // enable_restoration
  writer.put_bit(false);     // enable_ccso
  writer.put_bit(false);     // cdef_on_skip_txfm_always_on
  writer.put_bit(true);      // cdef_on_skip_txfm_disabled
  writer.put_bits(0U, 2U);   // df_par_bits_minus_2
  writer.put_bit(false);     // seq_tile_info_present_flag
  writer.put_bit(false);     // film_grain_params_present
  writer.put_bit(false);     // obu_extension_flag
  writer.put_bit(true);      // trailing one bit
  writer.align_to_byte();
  return av2enc::basic::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

struct GateFrameHeader {
  bool keyframe = false;
  std::uint32_t order_hint = 0;
  std::uint32_t refresh_frame_flags = 1;
  int num_total_refs = 0;
  std::uint8_t ref_frame_idx[7] = {};
  bool reference_select = false;
  bool skip_mode_present = false;
};

std::size_t write_frame_header_syntax(BitWriter& writer,
                                      const GateFrameHeader& h, int width,
                                      int height, std::uint8_t base_q_idx) {
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
      writer.put_bits(h.ref_frame_idx[i], 1U);
    }
    writer.put_bit(false);  // allow_intrabc
    writer.put_bit(false);  // use_qtr_precision_mv
    writer.put_bit(true);   // allow_high_precision_mv
    writer.put_bit(false);  // is_filter_switchable
    writer.put_bits(0U, 2U);  // interpolation_filter = EIGHTAP
  }
  writer.put_bit(true);  // disable_cdf_update
  write_tile_info(writer, width, height);
  writer.put_bits(base_q_idx, 8U);
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present
  writer.put_bit(false);  // apply_deblocking_filter[0]
  writer.put_bit(false);  // apply_deblocking_filter[1]
  writer.put_bit(false);  // tx_mode_select = 0
  if (!h.keyframe) {
    writer.put_bit(h.reference_select);
    writer.put_bit(h.skip_mode_present);
  }
  writer.put_bits(0U, 2U);  // reduced_tx_set
  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_frame_obu(const GateFrameHeader& header,
                                         int width, int height,
                                         std::uint8_t base_q_idx,
                                         std::span<const std::uint8_t> tile) {
  BitWriter writer;
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header_syntax(writer, header, width, height, base_q_idx);
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

mc::RefFrame make_ref(const RefPlaneView& view, int width, int height) {
  mc::RefFrame ref;
  ref.plane[0] = mc::PlaneView{view.y, view.stride_y};
  ref.plane[1] = mc::PlaneView{view.u, view.stride_uv};
  ref.plane[2] = mc::PlaneView{view.v, view.stride_uv};
  ref.mi_cols = width / 4;
  ref.mi_rows = height / 4;
  return ref;
}

void copy_rect(const RefPlaneView& ref, int plane, int x, int y, int w, int h,
               std::uint8_t* dst, int dst_stride) {
  const std::uint8_t* src = plane == 0 ? ref.y : (plane == 1 ? ref.u : ref.v);
  const int src_stride = plane == 0 ? ref.stride_y : ref.stride_uv;
  for (int r = 0; r < h; ++r) {
    std::memcpy(dst + static_cast<std::ptrdiff_t>(r) * dst_stride,
                src + static_cast<std::ptrdiff_t>(y + r) * src_stride + x,
                static_cast<std::size_t>(w));
  }
}

// Verifies §7.13.3.8-.12 OPFL for a compound block.  The gate stream is built
// so both predictions are equal at the starting MVs; that makes every OPFL
// delta exactly zero (the gradient is multiplied by p0 - p1 == 0).  A non-zero
// refinement is reported instead of silently diverging from the decoder.
bool check_opfl_zero(const CompoundBlockPlan& leaf,
                     const CompoundFrameConfig& cfg,
                     const RefPlaneView& ref0, const RefPlaneView& ref1,
                     int x, int y, std::string& error) {
  constexpr int kN = 8;
  const int w = leaf.w4 * 4;
  const int h = leaf.h4 * 4;
  if (w % kN != 0 || h % kN != 0) {
    error = "OPFL block must be a multiple of 8";
    return false;
  }
  const int cells = (w / 4) * (h / 4);
  std::vector<std::int16_t> opfl(static_cast<std::size_t>(cells) * 4, 0);
  std::vector<std::int16_t> delta(static_cast<std::size_t>(cells) * 4, 0);
  temporal::OpflJob job;
  job.w = w;
  job.h = h;
  job.p0 = ref0.y + static_cast<std::ptrdiff_t>(y) * ref0.stride_y + x;
  job.p0_stride = ref0.stride_y;
  job.p1 = ref1.y + static_cast<std::ptrdiff_t>(y) * ref1.stride_y + x;
  job.p1_stride = ref1.stride_y;
  job.mvs[0] = temporal::Mv{0, 0};
  job.mvs[1] = temporal::Mv{0, 0};
  job.order_hint = static_cast<int>(cfg.ref_map.order_hint);
  job.ref0_order_hint = cfg.order_hint_of(leaf.ref_frame[0]);
  job.ref1_order_hint = cfg.order_hint_of(leaf.ref_frame[1]);
  job.n = kN;
  temporal::OpflOutput out;
  out.opfl_mvs = opfl.data();
  out.opfl_stride4 = w / 4;
  out.mv_deltas = delta.data();
  out.delta_stride4 = w / 4;
  if (temporal::opfl_compute(job, out) !=
      temporal::TemporalStatus::kOk) {
    error = "opfl_compute failed";
    return false;
  }
  for (std::int16_t value : delta) {
    if (value != 0) {
      error = "OPFL produced a non-zero refinement in this gate stream";
      return false;
    }
  }
  return true;
}

bool predict_leaf(const CompoundBlockPlan& leaf, const CompoundFrameConfig& cfg,
                  const mc::RefFrame& ref_a, const mc::RefFrame& ref_b,
                  const RefPlaneView& view_a, const RefPlaneView& view_b,
                  int width, int height, std::uint8_t* dst[3],
                  int strides[3], std::string& error) {
  const int x = leaf.mi_col * 4;
  const int y = leaf.mi_row * 4;
  const int w = leaf.w4 * 4;
  const int h = leaf.h4 * 4;
  const int cw = w / 2;
  const int ch = h / 2;
  const int cx = x / 2;
  const int cy = y / 2;

  if (!is_inter_ref_value(leaf.ref_frame[1])) {
    // Zero-MV single reference: an integer-position copy.
    copy_rect(view_a, 0, x, y, w, h, dst[0] + static_cast<std::ptrdiff_t>(y) * strides[0] + x, strides[0]);
    copy_rect(view_a, 1, cx, cy, cw, ch, dst[1] + static_cast<std::ptrdiff_t>(cy) * strides[1] + cx, strides[1]);
    copy_rect(view_a, 2, cx, cy, cw, ch, dst[2] + static_cast<std::ptrdiff_t>(cy) * strides[2] + cx, strides[2]);
    return true;
  }

  CompoundPrediction prediction;
  if (!multiref::build_compound_prediction(
          ref_a, ref_b, mc::BlockRect{x, y, w, h},
          static_cast<int>(partition::BlockSize::k64x64),
          mc::Mv{0, 0}, mc::Mv{0, 0}, mc::InterpFilter::kEighttap,
          width / 4, height / 4, prediction, error)) {
    return false;
  }
  CompoundModeParams params = leaf.compound;
  params.use_imp_mask = false;
  if (params.type == adv::CompoundType::kAverage) {
    if (leaf.skip_mode) {
      params.cwp_weight = leaf.compound.cwp_weight;
    } else {
      params.cwp_weight = multiref::cwp_weight_from_index(
          params.cwp_index,
          cfg.same_side(leaf.ref_frame[0], leaf.ref_frame[1]));
    }
  }
  if (!multiref::apply_compound_prediction(prediction, params, dst, strides,
                                           error)) {
    return false;
  }
  if (leaf.use_optflow &&
      !check_opfl_zero(leaf, cfg, view_a, view_b, x, y, error)) {
    return false;
  }
  return true;
}

}  // namespace

bool build_compound_stream(const CompoundStreamInput& input,
                           CompoundStreamOutput& output, std::string& error) {
  error.clear();
  output = CompoundStreamOutput{};
  if (input.width <= 0 || input.height <= 0 || input.width % 64 != 0 ||
      input.height % 64 != 0) {
    error = "compound stream requires dimensions divisible by 64";
    return false;
  }
  if (input.key_tile.empty() || input.future_tile.empty()) {
    error = "M2 tile payloads missing";
    return false;
  }
  if (input.key_recon.y == nullptr || input.future_recon.y == nullptr) {
    error = "reference reconstructions missing";
    return false;
  }
  const int sb_cols = input.width / 64;
  const int sb_rows = input.height / 64;
  if (static_cast<int>(input.leaves.size()) != sb_cols * sb_rows) {
    error = "leaf count must equal the superblock count";
    return false;
  }
  const CompoundFrameConfig& cfg = input.config;
  if (cfg.num_total_refs() != 2) {
    error = "frame 2 requires NumTotalRefs = 2";
    return false;
  }

  std::vector<std::uint8_t> tile;
  if (!write_compound_tile(input.leaves, input.width, input.height, cfg, tile,
                           error)) {
    return false;
  }
  output.tile_bytes = static_cast<std::uint32_t>(tile.size());

  // Encoder reconstruction with exactly the decoder's prediction.
  output.recon_y.assign(static_cast<std::size_t>(input.width) *
                            static_cast<std::size_t>(input.height),
                        0);
  output.recon_u.assign(static_cast<std::size_t>(input.width / 2) *
                            static_cast<std::size_t>(input.height / 2),
                        0);
  output.recon_v = output.recon_u;
  mc::RefFrame refs[2] = {
      make_ref(input.key_recon, input.width, input.height),
      make_ref(input.future_recon, input.width, input.height)};
  std::uint8_t* dst[3] = {output.recon_y.data(), output.recon_u.data(),
                          output.recon_v.data()};
  int strides[3] = {input.width, input.width / 2, input.width / 2};
  for (const CompoundBlockPlan& leaf : input.leaves) {
    const int a = leaf.ref_frame[0];
    const int b = is_inter_ref_value(leaf.ref_frame[1]) ? leaf.ref_frame[1] : a;
    if ((a != 0 && a != 1) || (b != 0 && b != 1)) {
      error = "leaf references must be 0 or 1";
      return false;
    }
    const RefPlaneView& view_a = a == 0 ? input.key_recon : input.future_recon;
    const RefPlaneView& view_b = b == 0 ? input.key_recon : input.future_recon;
    if (!predict_leaf(leaf, cfg, refs[a], refs[b], view_a, view_b, input.width,
                      input.height, dst, strides, error)) {
      return false;
    }
  }

  // Assemble the three temporal units.
  const std::vector<std::uint8_t> seq = make_sequence_header(input);
  std::vector<std::uint8_t> td = make_td();

  GateFrameHeader key_header;
  key_header.keyframe = true;
  key_header.order_hint = 0;
  key_header.refresh_frame_flags = (1U << kNumRefFrames) - 1U;
  GateFrameHeader future_header;
  future_header.order_hint = static_cast<std::uint32_t>(kFutureOrderHint);
  future_header.refresh_frame_flags = 2U;  // refresh slot 1
  future_header.num_total_refs = 1;
  future_header.ref_frame_idx[0] = 0;
  future_header.reference_select = false;
  future_header.skip_mode_present = false;
  GateFrameHeader compound_header;
  compound_header.order_hint = static_cast<std::uint32_t>(cfg.ref_map.order_hint);
  compound_header.refresh_frame_flags = 1U;  // refresh slot 0
  compound_header.num_total_refs = 2;
  compound_header.ref_frame_idx[0] = 0;
  compound_header.ref_frame_idx[1] = 1;
  compound_header.reference_select = cfg.reference_select();
  compound_header.skip_mode_present = cfg.skip_mode_present;

  const std::vector<std::uint8_t> key_obu = make_frame_obu(
      key_header, input.width, input.height, input.base_q_idx, input.key_tile);
  const std::vector<std::uint8_t> future_obu =
      make_frame_obu(future_header, input.width, input.height,
                     input.base_q_idx, input.future_tile);
  const std::vector<std::uint8_t> compound_obu =
      make_frame_obu(compound_header, input.width, input.height,
                     input.base_q_idx, tile);

  output.units.clear();
  output.units.reserve(3);
  std::vector<std::uint8_t> unit0;
  unit0.insert(unit0.end(), td.begin(), td.end());
  unit0.insert(unit0.end(), seq.begin(), seq.end());
  unit0.insert(unit0.end(), key_obu.begin(), key_obu.end());
  std::vector<std::uint8_t> unit1;
  unit1.insert(unit1.end(), td.begin(), td.end());
  unit1.insert(unit1.end(), future_obu.begin(), future_obu.end());
  std::vector<std::uint8_t> unit2;
  unit2.insert(unit2.end(), td.begin(), td.end());
  unit2.insert(unit2.end(), compound_obu.begin(), compound_obu.end());
  output.units.push_back(std::move(unit0));
  output.units.push_back(std::move(unit1));
  output.units.push_back(std::move(unit2));
  output.order_hint = static_cast<int>(cfg.ref_map.order_hint);
  output.future_order_hint = kFutureOrderHint;
  return true;
}

bool check_disabled_byte_identity(int width, int height,
                                  std::uint8_t base_q_idx, bool* identical,
                                  std::size_t* bytes, std::string& error) {
  error.clear();
  if (identical != nullptr) *identical = false;
  if (bytes != nullptr) *bytes = 0;
  if (width <= 0 || height <= 0 || width % 64 != 0 || height % 64 != 0) {
    error = "byte-identity control requires dimensions divisible by 64";
    return false;
  }

  CompoundFrameConfig cfg;
  cfg.ref_map.num_total_refs = 1;
  cfg.ref_map.ref_frame_idx[0] = 0;
  cfg.ref_map.order_hint = 2;
  cfg.ref_map.order_hints[0] = 0;
  cfg.ref_map.frame_distance[0] = 2;
  cfg.ref_map.reference_select = false;
  cfg.ref_map.num_same_ref_compound = 0;
  cfg.skip_mode_present = false;
  cfg.opfl_refine_type = kRefineNone;
  cfg.enable_masked_compound = false;
  cfg.enable_cwp = false;

  const int sb_cols = width / 64;
  const int sb_rows = height / 64;
  std::vector<CompoundBlockPlan> plans;
  plans.reserve(static_cast<std::size_t>(sb_cols) * static_cast<std::size_t>(sb_rows));
  for (int sb_r = 0; sb_r < sb_rows; ++sb_r) {
    for (int sb_c = 0; sb_c < sb_cols; ++sb_c) {
      CompoundBlockPlan plan;
      plan.mi_row = sb_r * 16;
      plan.mi_col = sb_c * 16;
      plan.w4 = 16;
      plan.h4 = 16;
      plan.skip_flag = true;
      plan.y_mode = kGlobalMv;
      plan.ref_frame[0] = 0;
      plan.ref_frame[1] = kNoneRef;
      plans.push_back(plan);
    }
  }
  std::vector<std::uint8_t> mine;
  if (!write_compound_tile(plans, width, height, cfg, mine, error)) {
    return false;
  }

  // The M2 core writer for exactly the same tree and decisions.
  syntax::FramePlan core_plan;
  core_plan.width = width;
  core_plan.height = height;
  core_plan.sb_size = 64;
  core_plan.inter_frame = true;
  std::vector<syntax::BlockSyntax> blocks;
  blocks.reserve(plans.size());
  for (const CompoundBlockPlan& plan : plans) {
    syntax::BlockSyntax b;
    b.mi_row = static_cast<std::uint16_t>(plan.mi_row);
    b.mi_col = static_cast<std::uint16_t>(plan.mi_col);
    b.bsize = static_cast<std::uint8_t>(partition::BlockSize::k64x64);
    b.partition = static_cast<std::uint8_t>(partition::PartitionType::kNone);
    b.flags = syntax::kFlagSbRoot;
    b.allowed_mask =
        1U << static_cast<unsigned>(partition::PartitionType::kNone);
    b.is_inter = 1;
    b.skip_flag = 1;
    b.inter_mode = syntax::kInterGlobalMv;
    blocks.push_back(b);
  }
  VideoConfig core_config;
  core_config.max_frame_width = static_cast<std::uint16_t>(width);
  core_config.max_frame_height = static_cast<std::uint16_t>(height);
  core_config.base_q_idx = base_q_idx;
  core_config.num_total_refs = 1;
  core_config.disable_cdf_update = true;
  std::vector<std::uint8_t> core_tile;
  if (!syntax::assemble_video_frame(core_plan, blocks, {}, core_config, core_tile,
                                 error, nullptr)) {
    return false;
  }
  if (bytes != nullptr) *bytes = mine.size();
  if (identical != nullptr) *identical = mine == core_tile;
  if (identical != nullptr && !*identical) {
    error = "disabled compound tile is not byte-identical to the M2 core "
            "(mine " +
            std::to_string(mine.size()) + " bytes, core " +
            std::to_string(core_tile.size()) + " bytes)";
    return false;
  }
  return true;
}

}  // namespace av2enc::video::compound
