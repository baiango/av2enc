#include "codec/inter/multiref/gate/compound_stream.h"

#include <cassert>
#include <cstddef>

#include "core/bitstream/bitwriter.h"
#include "core/bitstream/basic_headers.h"
#include "codec/inter/multiref/compound_predict.h"
#include "codec/partition/tree/partition.h"

namespace av2enc::video::multiref::gate {
namespace {

using av2enc::basic::ObuHeader;
using av2enc::basic::ObuType;

constexpr int kOrderHintBits = 4;
constexpr int kNumRefFrames = 2;
constexpr int kMaxDrlBitsMinus1 = 4;

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

// Sequence header for the gate: the M2 layout (§5.4) with masked compound,
// CWP and implicit-mask blending enabled.
std::vector<std::uint8_t> make_sequence_header(int width, int height) {
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
  for (int mode = 1; mode < 5; ++mode) {
    writer.put_bit(false);  // seq_enabled_motion_modes[mode]
  }
  writer.put_bit(true);   // enable_masked_compound  (gate)
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
  writer.put_bit(true);   // enable_cwp            (gate)
  writer.put_bit(true);   // enable_imp_msk_bld    (gate)
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
  put_trailing_bits(writer);
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
};

std::size_t write_frame_header_syntax(BitWriter& writer,
                                      const GateFrameHeader& h,
                                      int width, int height) {
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
    writer.put_bit(h.reference_select);  // reference_select
    writer.put_bit(false);               // skip_mode_present
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

}  // namespace

std::vector<GateLeaf> make_default_gate_leaves(int superblock_count) {
  std::vector<GateLeaf> leaves(static_cast<std::size_t>(superblock_count));
  for (int i = 0; i < superblock_count; ++i) {
    GateLeaf& leaf = leaves[static_cast<std::size_t>(i)];
    switch (i % 4) {
      case 0:
        leaf.y_mode = 22;  // GLOBAL_GLOBALMV
        leaf.mode.type = adv::CompoundType::kAverage;
        leaf.mode.cwp_weight = kCwpEqual;
        break;
      case 1:
        leaf.y_mode = 19;  // NEAR_NEARMV: CWP is signalled
        leaf.drl_index[0] = 0;
        leaf.drl_index[1] = 0;
        leaf.mode.type = adv::CompoundType::kAverage;
        leaf.mode.cwp_index = 1;
        leaf.mode.cwp_weight = cwp_weight_from_index(1, /*same_side=*/false);
        break;
      case 2:
        leaf.y_mode = 22;
        leaf.mode.type = adv::CompoundType::kWedge;
        leaf.mode.wedge_index = 22;
        leaf.mode.wedge_sign = 0;
        break;
      default:
        leaf.y_mode = 22;
        leaf.mode.type = adv::CompoundType::kDiffWtd;
        leaf.mode.mask_type = 0;
        break;
    }
  }
  return leaves;
}

bool build_compound_stream(const CompoundStreamInput& input,
                           CompoundStreamOutput& output, std::string& error) {
  error.clear();
  output = CompoundStreamOutput{};
  if (input.width <= 0 || input.height <= 0 || (input.width % 64) != 0 ||
      (input.height % 64) != 0) {
    error = "gate stream requires dimensions divisible by 64";
    return false;
  }
  if (input.key_tile.empty() || input.p1_tile.empty()) {
    error = "M2 tile payloads missing";
    return false;
  }
  if (input.key_recon.y == nullptr || input.p1_recon.y == nullptr) {
    error = "reference reconstructions missing";
    return false;
  }
  const int sb_cols = input.width / 64;
  const int sb_rows = input.height / 64;
  if (static_cast<int>(input.leaves.size()) != sb_cols * sb_rows) {
    error = "leaf count must equal the superblock count";
    return false;
  }
  if ((input.width % 8) != 0 || (input.height % 8) != 0) {
    error = "gate stream requires dimensions divisible by 8";
    return false;
  }

  RefSlotBank slots{};
  slots[0] = RefSlot{true, true, 0};   // key frame
  slots[1] = RefSlot{true, false, 1};  // first P frame
  const std::uint8_t ref_idx[2] = {0, 1};
  if (!build_ref_frame_map(slots, ref_idx, input.order_hint,
                           /*reference_select=*/true,
                           /*num_same_ref_compound=*/0, output.ref_map,
                           error)) {
    return false;
  }

  GateTilePlan plan;
  plan.width = input.width;
  plan.height = input.height;
  plan.sb_size = 64;
  plan.leaves = input.leaves;
  GateTileConfig tile_config;
  tile_config.enable_masked_compound = input.enable_masked_compound;
  tile_config.enable_cwp = input.enable_cwp;
  std::vector<std::uint8_t> p2_tile;
  if (!write_gate_compound_tile(plan, output.ref_map, tile_config, p2_tile,
                                error)) {
    return false;
  }
  output.p2_tile_bytes = static_cast<std::uint32_t>(p2_tile.size());

  // Reconstruct frame 2 with exactly the prediction the decoder derives:
  // zero MVs (GLOBAL_GLOBALMV / NEAR_NEARMV with an all-zero stack).
  const int cw = input.width / 2;
  const int ch = input.height / 2;
  output.recon_y.assign(
      static_cast<std::size_t>(input.width) * static_cast<std::size_t>(input.height),
      0);
  output.recon_u.assign(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch), 0);
  output.recon_v.assign(static_cast<std::size_t>(cw) * static_cast<std::size_t>(ch), 0);

  mc::RefFrame refs[2];
  const mc::BlockRect full{0, 0, input.width, input.height};
  refs[0].plane[0] = mc::PlaneView{input.key_recon.y, input.key_recon.stride_y};
  refs[0].plane[1] = mc::PlaneView{input.key_recon.u, input.key_recon.stride_uv};
  refs[0].plane[2] = mc::PlaneView{input.key_recon.v, input.key_recon.stride_uv};
  refs[0].mi_cols = input.width / 4;
  refs[0].mi_rows = input.height / 4;
  refs[1].plane[0] = mc::PlaneView{input.p1_recon.y, input.p1_recon.stride_y};
  refs[1].plane[1] = mc::PlaneView{input.p1_recon.u, input.p1_recon.stride_uv};
  refs[1].plane[2] = mc::PlaneView{input.p1_recon.v, input.p1_recon.stride_uv};
  refs[1].mi_cols = input.width / 4;
  refs[1].mi_rows = input.height / 4;

  int leaf_index = 0;
  for (int sb_r = 0; sb_r < sb_rows; ++sb_r) {
    for (int sb_c = 0; sb_c < sb_cols; ++sb_c, ++leaf_index) {
      const GateLeaf& leaf = input.leaves[static_cast<std::size_t>(leaf_index)];
      const mc::BlockRect rect{sb_c * 64, sb_r * 64, 64, 64};
      CompoundPrediction prediction;
      if (!build_compound_prediction(refs[0], refs[1], rect,
                                     static_cast<int>(partition::BlockSize::k64x64),
                                     mc::Mv{0, 0}, mc::Mv{0, 0},
                                     mc::InterpFilter::kEighttap,
                                     input.width / 4, input.height / 4,
                                     prediction, error)) {
        return false;
      }
      CompoundModeParams params = leaf.mode;
      if (params.type == adv::CompoundType::kAverage) {
        // The decoder derives the CWP weight from is_same_side() and the
        // signalled index; mirror that here so the reconstruction matches.
        params.cwp_weight = cwp_weight_from_index(
            params.cwp_index, output.ref_map.is_same_side(0, 1));
      }
      // The gate never uses the implicit-mask blend: its average blocks are
      // GLOBAL_GLOBALMV and the CWP block has a non-equal weight, so
      // §7.13.3.14's eligibility condition is false in both cases.
      params.use_imp_mask = false;
      std::uint8_t* dst[3] = {output.recon_y.data(), output.recon_u.data(),
                              output.recon_v.data()};
      int strides[3] = {input.width, cw, cw};
      if (!apply_compound_prediction(prediction, params, dst, strides, error)) {
        return false;
      }
    }
  }
  (void)full;

  // Assemble the three temporal units with the gate's headers.
  const std::vector<std::uint8_t> seq = make_sequence_header(input.width, input.height);
  const std::vector<std::uint8_t> td = make_td();

  GateFrameHeader key_header;
  key_header.keyframe = true;
  key_header.order_hint = 0;
  key_header.refresh_frame_flags = (1U << kNumRefFrames) - 1U;
  GateFrameHeader p1_header;
  p1_header.order_hint = 1;
  p1_header.refresh_frame_flags = 2U;  // refresh slot 1
  p1_header.num_total_refs = 1;
  p1_header.ref_frame_idx[0] = 0;
  p1_header.reference_select = false;
  GateFrameHeader p2_header;
  p2_header.order_hint = input.order_hint;
  p2_header.refresh_frame_flags = 1U;  // refresh slot 0
  p2_header.num_total_refs = 2;
  p2_header.ref_frame_idx[0] = 0;
  p2_header.ref_frame_idx[1] = 1;
  p2_header.reference_select = true;

  const std::vector<std::uint8_t> key_obu =
      make_frame_obu(key_header, input.width, input.height, input.key_tile);
  const std::vector<std::uint8_t> p1_obu =
      make_frame_obu(p1_header, input.width, input.height, input.p1_tile);
  const std::vector<std::uint8_t> p2_obu =
      make_frame_obu(p2_header, input.width, input.height, p2_tile);

  output.units.clear();
  output.units.reserve(3);
  std::vector<std::uint8_t> unit0;
  unit0.insert(unit0.end(), td.begin(), td.end());
  unit0.insert(unit0.end(), seq.begin(), seq.end());
  unit0.insert(unit0.end(), key_obu.begin(), key_obu.end());
  std::vector<std::uint8_t> unit1;
  unit1.insert(unit1.end(), td.begin(), td.end());
  unit1.insert(unit1.end(), p1_obu.begin(), p1_obu.end());
  std::vector<std::uint8_t> unit2;
  unit2.insert(unit2.end(), td.begin(), td.end());
  unit2.insert(unit2.end(), p2_obu.begin(), p2_obu.end());
  output.units.push_back(std::move(unit0));
  output.units.push_back(std::move(unit1));
  output.units.push_back(std::move(unit2));
  return true;
}

}  // namespace av2enc::video::multiref::gate
