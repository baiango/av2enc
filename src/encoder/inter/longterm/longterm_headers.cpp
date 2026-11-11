// Exact syntax bits for the long-term-reference fields of the AV2 sequence and
// frame headers, plus the matching reader.
//
// The writer is a field-for-field restatement of the M2 header writer
// (src/encoder/m2/m2_headers.cpp) with the §5.4.6/§5.18.2 long-term fields
// added. `longterm_test.cpp` asserts byte equality against the M2 writer when
// long_term_frame_id_bits == 0, so the two cannot drift apart silently.
#include "encoder/inter/longterm/longterm.h"

#include <cassert>
#include <cstring>

#include "bitstream/m0_headers.h"

namespace av2enc::inter::longterm {
namespace {

using av2enc::m0::ObuHeader;
using av2enc::m0::ObuType;

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
  assert(n >= 2U);
  assert(value < n);
  unsigned w = 0;
  while ((1U << w) < n) {
    ++w;
  }
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

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

std::uint32_t mi_cols(const SequenceConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
}
std::uint32_t mi_rows(const SequenceConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
}

void write_tile_info(BitWriter& writer, const SequenceConfig& c) {
  const std::uint32_t sb_cols = (mi_cols(c) + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows(c) + 15U) >> 4U;
  assert(sb_cols >= 1U && sb_rows >= 1U);
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) {
    writer.put_bit(false);  // increment_tile_cols_log2 = 0
  }
  if (sb_rows > 1U) {
    writer.put_bit(false);  // increment_tile_rows_log2 = 0
  }
}

bool set_error(std::string* error, const char* message) {
  if (error != nullptr) *error = message;
  return false;
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------
class BitReader {
 public:
  explicit BitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  bool bit(bool* out) {
    if (bit_pos_ >= bytes_.size() * 8U) return false;
    const std::size_t byte = bit_pos_ >> 3U;
    const unsigned shift = 7U - static_cast<unsigned>(bit_pos_ & 7U);
    *out = ((bytes_[byte] >> shift) & 1U) != 0U;
    ++bit_pos_;
    return true;
  }

  bool bits(unsigned count, std::uint64_t* out) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < count; ++i) {
      bool b = false;
      if (!bit(&b)) return false;
      value = (value << 1U) | (b ? 1U : 0U);
    }
    *out = value;
    return true;
  }

  bool uvlc(std::uint32_t* out) {
    unsigned leading_zeros = 0;
    while (true) {
      bool b = false;
      if (!bit(&b)) return false;
      if (b) break;
      ++leading_zeros;
      if (leading_zeros >= 32U) return false;
    }
    std::uint64_t payload = 0;
    if (!bits(leading_zeros, &payload)) return false;
    *out = static_cast<std::uint32_t>(((1ULL << leading_zeros) - 1ULL) +
                                      payload);
    return true;
  }

  bool ns(std::uint32_t n, std::uint32_t* out) {
    if (n < 2U) return false;
    unsigned w = 0;
    while ((1U << w) < n) ++w;
    const std::uint32_t m = (1U << w) - n;
    std::uint64_t v = 0;
    if (!bits(w - 1U, &v)) return false;
    if (v < m) {
      *out = static_cast<std::uint32_t>(v);
      return true;
    }
    bool extra = false;
    if (!bit(&extra)) return false;
    *out = static_cast<std::uint32_t>((v << 1U) - m + (extra ? 1U : 0U));
    return true;
  }

  void align_to_byte() { bit_pos_ = (bit_pos_ + 7U) & ~std::size_t{7U}; }

  std::size_t bit_pos() const { return bit_pos_; }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_pos_ = 0;
};

bool read_flag(BitReader& reader, bool* out) { return reader.bit(out); }

bool read_bits(BitReader& reader, unsigned count, std::uint64_t* out) {
  return reader.bits(count, out);
}

}  // namespace

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
unsigned ceil_log2(std::uint32_t value) {
  if (value <= 1U) return 0U;
  unsigned bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0U) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

int max_long_term_id(const SequenceConfig& config) {
  if (config.long_term_frame_id_bits == 0U) return -1;
  return (1 << config.long_term_frame_id_bits) - 2;
}

int max_long_term_id_plus_1(const SequenceConfig& config) {
  if (config.long_term_frame_id_bits == 0U) return 0;
  return (1 << config.long_term_frame_id_bits) - 1;
}

bool validate_sequence_header_config(const SequenceConfig& c,
                                     std::string* error) {
  if (c.max_mlayer_id != 0U) {
    return set_error(
        error,
        "max_mlayer_id > 0 needs seq_max_mlayer_cnt_minus_1 and "
        "mlayer_dependency_present_flag/map (§5.4.1), not implemented");
  }
  if (c.max_tlayer_id != 0U) {
    return set_error(
        error,
        "max_tlayer_id > 0 needs tlayer_dependency_present_flag/map "
        "(§5.4.1), not implemented");
  }
  if (c.seq_level_idx > 31U) {
    return set_error(error, "seq_level_idx is outside f(5)");
  }
  if (c.order_hint_bits < 1U || c.order_hint_bits > 16U) {
    return set_error(error, "order_hint_bits is outside 1..16");
  }
  if (c.num_ref_frames < 1U || c.num_ref_frames > 16U) {
    return set_error(error, "num_ref_frames is outside 1..16");
  }
  if (c.long_term_frame_id_bits >= 8U) {
    return set_error(error, "long_term_frame_id_bits is outside f(3)");
  }
  if (c.max_drl_bits_minus_1 >= 5U) {
    return set_error(error, "max_drl_bits_minus_1 is outside f(3)");
  }
  if (error != nullptr) error->clear();
  return true;
}

// ---------------------------------------------------------------------------
// sequence_header_obu() (§5.4.1..§5.4.10) for single_picture_header_flag = 0
// ---------------------------------------------------------------------------
std::size_t write_sequence_header(BitWriter& writer,
                                  const SequenceConfig& c) {
  if (!validate_sequence_header_config(c, nullptr)) {
    return 0;
  }
  const std::size_t start = writer.bit_size();
  assert(c.order_hint_bits >= 1U && c.order_hint_bits <= 16U);
  assert(c.num_ref_frames >= 1U && c.num_ref_frames <= 16U);
  assert(c.long_term_frame_id_bits < 8U);
  assert(c.max_drl_bits_minus_1 < 5U);  // MAX_REF_MV_STACK_SIZE - 1 == 5

  put_uvlc(writer, 0);                   // seq_header_id
  writer.put_bits(0U, 5U);               // seq_profile_idc
  writer.put_bit(false);                 // single_picture_header_flag = 0
  writer.put_bits(c.seq_level_idx, 5U);  // seq_level_idx
  // §5.4.1: seq_tier is coded only when seq_level_idx > 3 and the header is
  // not a single-picture header (this writer always uses single_picture = 0).
  if (c.seq_level_idx > 3U) {
    writer.put_bit(false);               // seq_tier = 0 (Main tier)
  }
  put_uvlc(writer, 0);                   // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                   // bit_depth_idc = 1 => 8-bit

  // §5.4.1 non-single-picture header.
  writer.put_bits(0U, 3U);  // seq_lcr_id
  writer.put_bit(false);    // still_picture
  writer.put_bits(c.max_tlayer_id, 2U);
  writer.put_bits(c.max_mlayer_id, 3U);
  writer.put_bit(c.monotonic_output_order_flag);

  unsigned width_bits =
      bit_width(static_cast<std::uint32_t>(c.max_frame_width) - 1U);
  if (width_bits == 0U) width_bits = 1;
  unsigned height_bits =
      bit_width(static_cast<std::uint32_t>(c.max_frame_height) - 1U);
  if (height_bits == 0U) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_width) - 1U,
                  width_bits);
  writer.put_bits(static_cast<std::uint64_t>(c.max_frame_height) - 1U,
                  height_bits);

  writer.put_bit(false);  // seq_cropping_window_present_flag
  writer.put_bit(false);  // seq_initial_display_delay_present_flag
  writer.put_bit(false);  // decoder_model_info_present_flag

  // §5.4.3 sequence_partition_config(): 64x64 superblocks, no SDP.
  writer.put_bit(false);  // use_256x256_superblock
  writer.put_bit(false);  // use_128x128_superblock
  writer.put_bit(false);  // enable_sdp
  writer.put_bit(false);  // enable_ext_partitions
  writer.put_bit(false);  // reduce_pb_aspect_ratio

  // §5.4.4 sequence_segment_config().
  writer.put_bit(false);  // enable_ext_seg
  writer.put_bit(false);  // seq_seg_info_present_flag

  // §5.4.5 sequence_intra_config().
  writer.put_bit(false);  // enable_dip
  writer.put_bit(true);   // enable_intra_edge_filter
  writer.put_bit(false);  // enable_mrls
  writer.put_bit(c.enable_cfl_intra);
  writer.put_bits(0U, 2U);  // cfl_ds_filter_index
  writer.put_bit(false);    // enable_mhccp
  writer.put_bit(false);    // enable_ibp

  // §5.4.6 sequence_inter_config() for single_picture_header_flag = 0.
  for (int mode = 1; mode < 5; ++mode) {  // INTERINTRA..EXTENDWARP
    writer.put_bit(false);                // seq_enabled_motion_modes[mode]
  }
  writer.put_bit(false);  // enable_masked_compound
  writer.put_bit(false);  // enable_ref_frame_mvs
  writer.put_bits(static_cast<std::uint64_t>(c.order_hint_bits) - 1U, 4U);
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder -> DRL_REORDER_DISABLED
  writer.put_bit(c.explicit_ref_frame_map);
  writer.put_bit(c.explicit_num_ref_frames);
  if (c.explicit_num_ref_frames) {
    writer.put_bits(static_cast<std::uint64_t>(c.num_ref_frames) - 1U, 4U);
  }
  writer.put_bits(c.long_term_frame_id_bits, 3U);
  put_ns(writer, 5U, c.max_drl_bits_minus_1);  // MAX_REF_MV_STACK_SIZE - 1
  writer.put_bit(false);  // allow_frame_max_drl_bits
  put_ns(writer, 3U, 0U);  // seq_max_bvp_drl_bits_minus_1
  writer.put_bit(false);   // allow_frame_max_bvp_drl_bits
  writer.put_bits(0U, 2U); // num_same_ref_compound
  writer.put_bit(false);   // enable_tip
  writer.put_bit(false);   // enable_mv_traj
  writer.put_bit(false);   // enable_bawp
  writer.put_bit(false);   // enable_cwp
  writer.put_bit(false);   // enable_imp_msk_bld
  writer.put_bit(false);   // enable_df_sub_pu
  writer.put_bits(0U, 2U); // enable_opfl_refine = REFINE_NONE
  writer.put_bit(false);   // enable_refinemv
  writer.put_bit(false);   // enable_bru
  writer.put_bit(false);   // enable_adaptive_mvd
  writer.put_bit(false);   // enable_mvd_sign_derive
  writer.put_bit(false);   // enable_flex_mvres
  writer.put_bit(false);   // enable_global_motion
  writer.put_bit(c.enable_short_refresh_frame_flags);

  // §5.4.7 sequence_scc_config().
  writer.put_bit(false);  // seq_choose_screen_content_tools
  writer.put_bit(false);  // seq_force_screen_content_tools

  // §5.4.8 sequence_transform_quant_entropy_config().
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
  writer.put_bits(23U, 5U);  // base_uv_ac_delta_q => BaseUVAcDeltaQ = 0
  writer.put_bit(false);     // uv_ac_delta_q_enabled

  // §5.4.10 sequence_filter_config().
  writer.put_bit(false);  // disable_loopfilters_across_tiles
  writer.put_bit(false);  // enable_cdef
  writer.put_bit(false);  // enable_gdf
  writer.put_bit(false);  // enable_restoration
  writer.put_bit(false);  // enable_ccso
  writer.put_bit(false);  // cdef_on_skip_txfm_always_on
  writer.put_bit(true);   // cdef_on_skip_txfm_disabled
  writer.put_bits(0U, 2U);  // df_par_bits_minus_2

  // §5.4.2 sequence_tile_config().
  writer.put_bit(false);  // seq_tile_info_present_flag

  writer.put_bit(false);  // film_grain_params_present

  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_sequence_header_obu(
    const SequenceConfig& config) {
  if (!validate_sequence_header_config(config, nullptr)) return {};
  BitWriter writer;
  write_sequence_header(writer, config);
  writer.put_bit(false);  // obu_extension_flag
  put_trailing_bits(writer);
  return av2enc::m0::make_obu(ObuHeader{ObuType::kSequenceHeader},
                              writer.take_bytes());
}

std::vector<std::uint8_t> make_temporal_delimiter_obu() {
  return av2enc::m0::make_obu(ObuHeader{ObuType::kTemporalDelimiter},
                              std::span<const std::uint8_t>{});
}

// ---------------------------------------------------------------------------
// frame_header_info() (§5.18.2)
// ---------------------------------------------------------------------------
std::size_t write_frame_header(BitWriter& writer, const SequenceConfig& c,
                               const FrameHeader& h) {
  const std::size_t start = writer.bit_size();
  const bool key = is_key_role(h.role);
  const bool intra = key || !h.frame_is_inter;

  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header

  if (!key) {
    writer.put_bit(h.frame_is_inter);  // frame_is_inter -> INTER/INTRA_ONLY
  }
  // LongTermId = -1 unless this is a key frame with a coded id.
  if (key) {
    const std::uint32_t plus_one =
        h.long_term_id < 0 ? 0U
                           : static_cast<std::uint32_t>(h.long_term_id) + 1U;
    writer.put_bits(plus_one, c.long_term_frame_id_bits);
  }
  // num_key_ref_frames: OLK (and RAS, not modelled) only.
  if (h.role == FrameRole::kOpenLoopKey && c.long_term_frame_id_bits != 0U &&
      c.olk_long_term_list == LongTermListSyntax::kSpecV1) {
    writer.put_bits(h.ref_long_term_id.size(), 3U);
    for (const std::uint8_t id : h.ref_long_term_id) {
      writer.put_bits(id, c.long_term_frame_id_bits);
    }
  }
  // immediate_output_frame: inferred 0 for OLK.
  if (h.role == FrameRole::kOpenLoopKey) {
    assert(!h.immediate_output_frame);
  } else {
    writer.put_bit(h.immediate_output_frame);
  }
  // implicit_output_frame is only coded when it can be non-zero.
  if (!c.monotonic_output_order_flag && !h.immediate_output_frame) {
    writer.put_bit(false);  // implicit_output_frame
  }

  writer.put_bit(false);  // frame_size_override_flag = 0
  writer.put_bits(h.order_hint, c.order_hint_bits);

  if (!intra) {
    writer.put_bit(false);  // signal_primary_ref_frame
    writer.put_bit(true);   // disable_cross_frame_cdf_init
  }

  if (key) {
    if (h.role == FrameRole::kClosedLoopKey && c.max_mlayer_id == 0U) {
      // refresh_frame_flags = allFrames, no bits.
    } else {
      // enable_short_refresh_frame_flags == 0 is validated.
      assert(!c.enable_short_refresh_frame_flags);
      writer.put_bits(h.refresh_frame_flags, c.num_ref_frames);
    }
  } else {
    writer.put_bits(h.refresh_frame_flags, c.num_ref_frames);
  }

  if (intra) {
    // frame_size() codes no bits with frame_size_override_flag == 0.
    // screen_content_params(): seq_force_screen_content_tools == 0.
    writer.put_bit(false);  // allow_intrabc
  } else {
    writer.put_bit(true);                       // frame_explicit_ref_frame_map
    writer.put_bits(h.num_total_refs, 3U);      // num_total_refs
    const unsigned idx_bits = ceil_log2(c.num_ref_frames);
    for (std::size_t i = 0; i < h.ref_frame_idx.size(); ++i) {
      writer.put_bits(h.ref_frame_idx[i], idx_bits);
    }
    // frame_size() codes no bits; get_ref_frames(1) is skipped.
    // screen_content_params(): no bits. intrabc_params():
    writer.put_bit(false);  // allow_intrabc
    // max_drl_bits_minus_1 = seq value; allow_frame_max_drl_bits == 0.
    writer.put_bit(false);   // use_qtr_precision_mv
    writer.put_bit(true);    // allow_high_precision_mv -> 1/8 pel
    writer.put_bit(false);   // is_filter_switchable
    writer.put_bits(0U, 2U); // interpolation_filter = EIGHTTAP
  }

  writer.put_bit(c.disable_cdf_update);

  write_tile_info(writer, c);

  writer.put_bits(c.base_q_idx, 8U);  // base_q_idx
  writer.put_bit(false);              // segmentation_enabled
  writer.put_bit(false);              // using_qmatrix
  writer.put_bit(false);              // delta_q_present
  writer.put_bit(false);              // apply_deblocking_filter[0]
  writer.put_bit(false);              // apply_deblocking_filter[1]
  writer.put_bit(false);              // tx_mode_select = 0
  if (!intra) {
    writer.put_bit(false);  // reference_select
    writer.put_bit(false);  // skip_mode_present
  }
  // allow_bawp / allow_warpmv_mode disabled at sequence level.
  writer.put_bits(0U, 2U);  // reduced_tx_set
  // global_motion_params() / film_grain_config() code no bits.

  return writer.bit_size() - start;
}

std::size_t write_tile_group_header(BitWriter& writer,
                                    const SequenceConfig& config,
                                    const FrameHeader& header) {
  const std::size_t start = writer.bit_size();
  writer.put_bit(true);  // is_first_tile_group
  write_frame_header(writer, config, header);
  // tile_start_and_end_present_flag is absent (NumTiles == 1).
  writer.align_to_byte();
  return writer.bit_size() - start;
}

std::vector<std::uint8_t> make_frame_obu(
    const SequenceConfig& config, const FrameHeader& header,
    std::span<const std::uint8_t> tile_payload) {
  BitWriter writer;
  write_tile_group_header(writer, config, header);
  std::vector<std::uint8_t> payload = writer.take_bytes();
  payload.insert(payload.end(), tile_payload.begin(), tile_payload.end());
  ObuHeader obu;
  obu.type = static_cast<ObuType>(obu_type_of(header.role));
  return av2enc::m0::make_obu(obu, payload);
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------
namespace {

bool read_uvlc(BitReader& reader, std::uint32_t* out, std::string* error) {
  if (!reader.uvlc(out)) {
    return set_error(error, "uvlc() ran past the end of the OBU");
  }
  return true;
}

bool expect_flag(BitReader& reader, bool expected, const char* name,
                 std::string* error) {
  bool value = false;
  if (!read_flag(reader, &value)) {
    if (error != nullptr) *error = std::string("truncated bit: ") + name;
    return false;
  }
  if (value != expected) {
    if (error != nullptr) {
      *error = std::string(name) + " = " + (value ? "1" : "0") +
               ", expected " + (expected ? "1" : "0");
    }
    return false;
  }
  return true;
}

}  // namespace

bool parse_obu(std::span<const std::uint8_t> bytes, ObuView* view,
               std::size_t* consumed, std::string* error) {
  if (view == nullptr || consumed == nullptr) {
    return set_error(error, "parse_obu: null output");
  }
  std::size_t pos = 0;
  std::uint64_t size = 0;
  unsigned shift = 0;
  bool terminated = false;
  while (pos < bytes.size()) {
    const std::uint8_t byte = bytes[pos++];
    size |= static_cast<std::uint64_t>(byte & 0x7FU) << shift;
    shift += 7U;
    if ((byte & 0x80U) == 0U) {
      terminated = true;
      break;
    }
    if (shift >= 35U) return set_error(error, "leb128 length too large");
  }
  if (!terminated) return set_error(error, "truncated leb128 length");
  const std::size_t prefix_size = pos;
  if (pos >= bytes.size()) return set_error(error, "missing OBU header");
  const std::uint8_t first = bytes[pos++];
  const bool extension = (first & 0x80U) != 0U;
  view->type = static_cast<std::uint8_t>((first >> 2U) & 0x1FU);
  view->tlayer_id = static_cast<std::uint8_t>(first & 0x03U);
  if (extension) {
    if (pos >= bytes.size()) return set_error(error, "truncated OBU header");
    ++pos;  // obu_mlayer_id + obu_xlayer_id
  }
  const std::size_t header_size = pos - prefix_size;
  if (size < header_size) return set_error(error, "OBU size smaller than its header");
  const std::size_t payload_size = static_cast<std::size_t>(size) - header_size;
  if (payload_size > bytes.size() - pos) {
    return set_error(error, "OBU payload runs past the input");
  }
  view->payload = bytes.subspan(pos, payload_size);
  *consumed = prefix_size + static_cast<std::size_t>(size);
  return true;
}

bool read_tile_group_header(std::span<const std::uint8_t> payload,
                            const SequenceConfig& c, ParsedFrameHeader* out,
                            std::string* error) {
  if (out == nullptr) return set_error(error, "read_tile_group_header: null");
  BitReader reader(payload);
  // The OBU type is not part of the tile-group payload, so the caller seeds
  // the role. Preserve it across the result reset.
  const FrameRole role = out->header.role;
  *out = ParsedFrameHeader{};
  out->header.role = role;
  if (!expect_flag(reader, true, "is_first_tile_group", error)) return false;

  FrameHeader& h = out->header;
  std::uint32_t cur_mfh_id = 0;
  std::uint32_t seq_header_id = 0;
  if (!read_uvlc(reader, &cur_mfh_id, error)) return false;
  if (!read_uvlc(reader, &seq_header_id, error)) return false;
  if (cur_mfh_id != 0U || seq_header_id != 0U) {
    return set_error(error, "unexpected cur_mfh_id / seq_header_id");
  }

  // The OBU type is not in this payload, so the role is resolved from the
  // coded fields: key frames code neither frame_is_inter nor num_total_refs.
  // The caller supplies the role out of band for key frames; for inter frames
  // the coded frame_is_inter decides INTER vs INTRA_ONLY. We therefore parse
  // using out->header.role as the initial guess and require it to be set.

  bool frame_is_inter = true;
  if (!is_key_role(h.role)) {
    if (!read_flag(reader, &frame_is_inter)) {
      return set_error(error, "truncated frame_is_inter");
    }
    h.frame_is_inter = frame_is_inter;
  } else {
    // FrameIsIntra == 1 for CLK/OLK; no bits are coded.
    h.frame_is_inter = false;
  }
  const bool key = is_key_role(h.role);
  const bool intra = key || !h.frame_is_inter;

  if (key) {
    std::uint64_t plus_one = 0;
    if (!read_bits(reader, c.long_term_frame_id_bits, &plus_one)) {
      return set_error(error, "truncated long_term_id_plus_1");
    }
    h.long_term_id = static_cast<std::int32_t>(plus_one) - 1;
  }

  if (h.role == FrameRole::kOpenLoopKey && c.long_term_frame_id_bits != 0U &&
      c.olk_long_term_list == LongTermListSyntax::kSpecV1) {
    std::uint64_t count = 0;
    if (!read_bits(reader, 3U, &count)) {
      return set_error(error, "truncated num_key_ref_frames");
    }
    h.ref_long_term_id.resize(static_cast<std::size_t>(count));
    for (std::size_t i = 0; i < h.ref_long_term_id.size(); ++i) {
      std::uint64_t id = 0;
      if (!read_bits(reader, c.long_term_frame_id_bits, &id)) {
        return set_error(error, "truncated ref_long_term_id");
      }
      h.ref_long_term_id[i] = static_cast<std::uint8_t>(id);
    }
  }

  if (h.role == FrameRole::kOpenLoopKey) {
    h.immediate_output_frame = false;
  } else {
    if (!read_flag(reader, &h.immediate_output_frame)) {
      return set_error(error, "truncated immediate_output_frame");
    }
  }
  if (!c.monotonic_output_order_flag && !h.immediate_output_frame) {
    bool implicit = false;
    if (!read_flag(reader, &implicit)) {
      return set_error(error, "truncated implicit_output_frame");
    }
    if (implicit) {
      return set_error(error, "implicit_output_frame = 1 is not supported");
    }
  }

  bool size_override = false;
  if (!read_flag(reader, &size_override)) {
    return set_error(error, "truncated frame_size_override_flag");
  }
  if (size_override) {
    return set_error(error, "frame_size_override_flag = 1 is not supported");
  }

  std::uint64_t order_hint = 0;
  if (!read_bits(reader, c.order_hint_bits, &order_hint)) {
    return set_error(error, "truncated order_hint");
  }
  h.order_hint = static_cast<std::uint32_t>(order_hint);

  if (!intra) {
    if (!expect_flag(reader, false, "signal_primary_ref_frame", error)) {
      return false;
    }
    if (!expect_flag(reader, true, "disable_cross_frame_cdf_init", error)) {
      return false;
    }
  }

  if (key && h.role == FrameRole::kClosedLoopKey && c.max_mlayer_id == 0U) {
    h.refresh_frame_flags = (1U << c.num_ref_frames) - 1U;
  } else {
    std::uint64_t flags = 0;
    if (!read_bits(reader, c.num_ref_frames, &flags)) {
      return set_error(error, "truncated refresh_frame_flags");
    }
    h.refresh_frame_flags = static_cast<std::uint32_t>(flags);
  }

  if (intra) {
    if (!expect_flag(reader, false, "allow_intrabc", error)) return false;
    h.num_total_refs = 0;
    h.ref_frame_idx.clear();
  } else {
    if (!expect_flag(reader, true, "frame_explicit_ref_frame_map", error)) {
      return false;
    }
    std::uint64_t total = 0;
    if (!read_bits(reader, 3U, &total)) {
      return set_error(error, "truncated num_total_refs");
    }
    h.num_total_refs = static_cast<std::uint8_t>(total);
    h.ref_frame_idx.resize(h.num_total_refs);
    const unsigned idx_bits = ceil_log2(c.num_ref_frames);
    for (std::size_t i = 0; i < h.ref_frame_idx.size(); ++i) {
      std::uint64_t idx = 0;
      if (!read_bits(reader, idx_bits, &idx)) {
        return set_error(error, "truncated ref_frame_idx");
      }
      h.ref_frame_idx[i] = static_cast<std::uint8_t>(idx);
    }
    if (!expect_flag(reader, false, "allow_intrabc", error)) return false;
    if (!expect_flag(reader, false, "use_qtr_precision_mv", error)) {
      return false;
    }
    if (!expect_flag(reader, true, "allow_high_precision_mv", error)) {
      return false;
    }
    if (!expect_flag(reader, false, "is_filter_switchable", error)) {
      return false;
    }
    std::uint64_t filter = 0;
    if (!read_bits(reader, 2U, &filter) || filter != 0U) {
      return set_error(error, "unexpected interpolation_filter");
    }
  }

  bool disable_cdf_update = false;
  if (!read_flag(reader, &disable_cdf_update)) {
    return set_error(error, "truncated disable_cdf_update");
  }
  if (disable_cdf_update != c.disable_cdf_update) {
    return set_error(error, "disable_cdf_update differs from the sequence");
  }

  // tile_info(): uniform spacing, one tile.
  if (!expect_flag(reader, true, "uniform_tile_spacing_flag", error)) {
    return false;
  }
  const std::uint32_t mi_c = mi_cols(c);
  const std::uint32_t mi_r = mi_rows(c);
  if (((mi_c + 15U) >> 4U) > 1U &&
      !expect_flag(reader, false, "increment_tile_cols_log2", error)) {
    return false;
  }
  if (((mi_r + 15U) >> 4U) > 1U &&
      !expect_flag(reader, false, "increment_tile_rows_log2", error)) {
    return false;
  }

  std::uint64_t base_q_idx = 0;
  if (!read_bits(reader, 8U, &base_q_idx)) {
    return set_error(error, "truncated base_q_idx");
  }
  if (base_q_idx != c.base_q_idx) {
    return set_error(error, "base_q_idx differs from the sequence");
  }
  if (!expect_flag(reader, false, "segmentation_enabled", error)) return false;
  if (!expect_flag(reader, false, "using_qmatrix", error)) return false;
  if (!expect_flag(reader, false, "delta_q_present", error)) return false;
  if (!expect_flag(reader, false, "apply_deblocking_filter[0]", error)) {
    return false;
  }
  if (!expect_flag(reader, false, "apply_deblocking_filter[1]", error)) {
    return false;
  }
  if (!expect_flag(reader, false, "tx_mode_select", error)) return false;
  if (!intra) {
    if (!expect_flag(reader, false, "reference_select", error)) return false;
    if (!expect_flag(reader, false, "skip_mode_present", error)) return false;
  }
  std::uint64_t reduced_tx_set = 0;
  if (!read_bits(reader, 2U, &reduced_tx_set) || reduced_tx_set != 0U) {
    return set_error(error, "unexpected reduced_tx_set");
  }

  out->header_bits = reader.bit_pos();
  reader.align_to_byte();
  out->tile_offset = reader.bit_pos() / 8U;
  if (out->tile_offset > payload.size()) {
    return set_error(error, "tile offset past the end of the OBU");
  }
  out->header_bits = reader.bit_pos();
  return true;
}

bool read_sequence_header(std::span<const std::uint8_t> payload,
                          const SequenceConfig& c, std::size_t* bits,
                          std::string* error) {
  if (!validate_sequence_header_config(c, error)) return false;
  BitReader reader(payload);
  std::uint32_t uvlc = 0;
  if (!read_uvlc(reader, &uvlc, error) || uvlc != 0U) {
    return set_error(error, "unexpected seq_header_id");
  }
  std::uint64_t v = 0;
  if (!read_bits(reader, 5U, &v) || v != 0U) {
    return set_error(error, "unexpected seq_profile_idc");
  }
  if (!expect_flag(reader, false, "single_picture_header_flag", error)) {
    return false;
  }
  if (!read_bits(reader, 5U, &v) || v != c.seq_level_idx) {
    return set_error(error, "unexpected seq_level_idx");
  }
  if (c.seq_level_idx > 3U &&
      !expect_flag(reader, false, "seq_tier", error)) {
    return false;
  }
  if (!read_uvlc(reader, &uvlc, error) || uvlc != 0U) {
    return set_error(error, "unexpected chroma_format_idc");
  }
  if (!read_uvlc(reader, &uvlc, error) || uvlc != 1U) {
    return set_error(error, "unexpected bit_depth_idc");
  }
  if (!read_bits(reader, 3U, &v) || v != 0U) {
    return set_error(error, "unexpected seq_lcr_id");
  }
  if (!expect_flag(reader, false, "still_picture", error)) return false;
  if (!read_bits(reader, 2U, &v) || v != c.max_tlayer_id) {
    return set_error(error, "unexpected max_tlayer_id");
  }
  if (!read_bits(reader, 3U, &v) || v != c.max_mlayer_id) {
    return set_error(error, "unexpected max_mlayer_id");
  }
  if (!expect_flag(reader, c.monotonic_output_order_flag,
                   "monotonic_output_order_flag", error)) {
    return false;
  }
  unsigned width_bits =
      bit_width(static_cast<std::uint32_t>(c.max_frame_width) - 1U);
  if (width_bits == 0U) width_bits = 1;
  unsigned height_bits =
      bit_width(static_cast<std::uint32_t>(c.max_frame_height) - 1U);
  if (height_bits == 0U) height_bits = 1;
  if (!read_bits(reader, 4U, &v) || v + 1U != width_bits) {
    return set_error(error, "unexpected frame_width_bits_minus_1");
  }
  if (!read_bits(reader, 4U, &v) || v + 1U != height_bits) {
    return set_error(error, "unexpected frame_height_bits_minus_1");
  }
  if (!read_bits(reader, width_bits, &v) ||
      v != static_cast<std::uint64_t>(c.max_frame_width) - 1U) {
    return set_error(error, "unexpected max_frame_width_minus_1");
  }
  if (!read_bits(reader, height_bits, &v) ||
      v != static_cast<std::uint64_t>(c.max_frame_height) - 1U) {
    return set_error(error, "unexpected max_frame_height_minus_1");
  }
  if (!expect_flag(reader, false, "seq_cropping_window_present_flag", error)) {
    return false;
  }
  if (!expect_flag(reader, false, "seq_initial_display_delay_present_flag",
                   error)) {
    return false;
  }
  if (!expect_flag(reader, false, "decoder_model_info_present_flag", error)) {
    return false;
  }
  for (int i = 0; i < 5; ++i) {  // partition config
    if (!expect_flag(reader, false, "sequence_partition_config", error)) {
      return false;
    }
  }
  for (int i = 0; i < 2; ++i) {  // segment config
    if (!expect_flag(reader, false, "sequence_segment_config", error)) {
      return false;
    }
  }
  if (!expect_flag(reader, false, "enable_dip", error)) return false;
  if (!expect_flag(reader, true, "enable_intra_edge_filter", error)) {
    return false;
  }
  if (!expect_flag(reader, false, "enable_mrls", error)) return false;
  if (!expect_flag(reader, c.enable_cfl_intra, "enable_cfl_intra", error)) {
    return false;
  }
  if (!read_bits(reader, 2U, &v) || v != 0U) {
    return set_error(error, "unexpected cfl_ds_filter_index");
  }
  if (!expect_flag(reader, false, "enable_mhccp", error)) return false;
  if (!expect_flag(reader, false, "enable_ibp", error)) return false;
  for (int mode = 1; mode < 5; ++mode) {
    if (!expect_flag(reader, false, "seq_enabled_motion_modes", error)) {
      return false;
    }
  }
  if (!expect_flag(reader, false, "enable_masked_compound", error)) {
    return false;
  }
  if (!expect_flag(reader, false, "enable_ref_frame_mvs", error)) return false;
  if (!read_bits(reader, 4U, &v) ||
      v + 1U != static_cast<std::uint64_t>(c.order_hint_bits)) {
    return set_error(error, "unexpected order_hint_bits_minus_1");
  }
  if (!expect_flag(reader, false, "enable_refmvbank", error)) return false;
  if (!expect_flag(reader, true, "disable_drl_reorder", error)) return false;
  if (!expect_flag(reader, c.explicit_ref_frame_map,
                   "explicit_ref_frame_map", error)) {
    return false;
  }
  if (!expect_flag(reader, c.explicit_num_ref_frames,
                   "explicit_num_ref_frames", error)) {
    return false;
  }
  if (c.explicit_num_ref_frames) {
    if (!read_bits(reader, 4U, &v) ||
        v + 1U != static_cast<std::uint64_t>(c.num_ref_frames)) {
      return set_error(error, "unexpected num_ref_frames_minus_1");
    }
  }
  if (!read_bits(reader, 3U, &v) || v != c.long_term_frame_id_bits) {
    return set_error(error, "unexpected long_term_frame_id_bits");
  }
  std::uint32_t ns_value = 0;
  if (!reader.ns(5U, &ns_value) || ns_value != c.max_drl_bits_minus_1) {
    return set_error(error, "unexpected seq_max_drl_bits_minus_1");
  }
  if (!expect_flag(reader, false, "allow_frame_max_drl_bits", error)) {
    return false;
  }
  if (!reader.ns(3U, &ns_value) || ns_value != 0U) {
    return set_error(error, "unexpected seq_max_bvp_drl_bits_minus_1");
  }
  if (!expect_flag(reader, false, "allow_frame_max_bvp_drl_bits", error)) {
    return false;
  }
  if (!read_bits(reader, 2U, &v) || v != 0U) {
    return set_error(error, "unexpected num_same_ref_compound");
  }
  const char* const inter_bits[] = {
      "enable_tip",       "enable_mv_traj",    "enable_bawp",
      "enable_cwp",       "enable_imp_msk_bld", "enable_df_sub_pu"};
  for (const char* name : inter_bits) {
    if (!expect_flag(reader, false, name, error)) return false;
  }
  if (!read_bits(reader, 2U, &v) || v != 0U) {
    return set_error(error, "unexpected enable_opfl_refine");
  }
  const char* const rest[] = {"enable_refinemv",      "enable_bru",
                              "enable_adaptive_mvd",  "enable_mvd_sign_derive",
                              "enable_flex_mvres",    "enable_global_motion"};
  for (const char* name : rest) {
    if (!expect_flag(reader, false, name, error)) return false;
  }
  if (!expect_flag(reader, c.enable_short_refresh_frame_flags,
                   "enable_short_refresh_frame_flags", error)) {
    return false;
  }
  if (!expect_flag(reader, false, "seq_choose_screen_content_tools", error)) {
    return false;
  }
  if (!expect_flag(reader, false, "seq_force_screen_content_tools", error)) {
    return false;
  }
  // §5.4.8, written in the same order as the writer.
  if (!expect_flag(reader, false, "enable_fsc", error)) return false;
  if (!expect_flag(reader, false, "enable_idtx_intra", error)) return false;
  if (!expect_flag(reader, false, "enable_intra_ist", error)) return false;
  if (!expect_flag(reader, false, "enable_inter_ist", error)) return false;
  if (!expect_flag(reader, true, "enable_chroma_dctonly", error)) {
    return false;
  }
  if (!expect_flag(reader, false, "enable_inter_ddt", error)) return false;
  if (!expect_flag(reader, false, "reduced_tx_part_set", error)) return false;
  if (!expect_flag(reader, false, "enable_cctx", error)) return false;
  if (!expect_flag(reader, false, "enable_tcq", error)) return false;
  if (!expect_flag(reader, false, "enable_parity_hiding", error)) return false;
  if (!expect_flag(reader, false, "enable_avg_cdf", error)) return false;
  if (!expect_flag(reader, false, "separate_uv_delta_q", error)) return false;
  if (!expect_flag(reader, true, "equal_ac_dc_q", error)) return false;
  if (!read_bits(reader, 5U, &v) || v != 23U) {
    return set_error(error, "unexpected base_uv_ac_delta_q");
  }
  if (!expect_flag(reader, false, "uv_ac_delta_q_enabled", error)) {
    return false;
  }
  if (!expect_flag(reader, false, "disable_loopfilters_across_tiles", error)) {
    return false;
  }
  if (!expect_flag(reader, false, "enable_cdef", error)) return false;
  if (!expect_flag(reader, false, "enable_gdf", error)) return false;
  if (!expect_flag(reader, false, "enable_restoration", error)) return false;
  if (!expect_flag(reader, false, "enable_ccso", error)) return false;
  if (!expect_flag(reader, false, "cdef_on_skip_txfm_always_on", error)) {
    return false;
  }
  if (!expect_flag(reader, true, "cdef_on_skip_txfm_disabled", error)) {
    return false;
  }
  if (!read_bits(reader, 2U, &v) || v != 0U) {
    return set_error(error, "unexpected df_par_bits_minus_2");
  }
  if (!expect_flag(reader, false, "seq_tile_info_present_flag", error)) {
    return false;
  }
  if (!expect_flag(reader, false, "film_grain_params_present", error)) {
    return false;
  }
  if (bits != nullptr) *bits = reader.bit_pos();
  return true;
}

}  // namespace av2enc::inter::longterm
