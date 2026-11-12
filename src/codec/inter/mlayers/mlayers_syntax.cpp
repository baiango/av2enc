#include "codec/inter/mlayers/mlayers_syntax.h"

#include "core/bitstream/bitwriter.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace av2enc::inter::mlayers {
namespace {

constexpr int kObuSequenceHeader = 1;
constexpr int kObuTemporalDelimiter = 2;
constexpr int kObuMsdo = 20;

bool fail(std::string& error, std::string message) {
  error = std::move(message);
  return false;
}

// §4.11.3 uvlc().
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

// §4.11.8 ns(n).
void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
  assert(n >= 2U);
  assert(value < n);
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
  while (value != 0U) {
    value >>= 1U;
    ++bits;
  }
  return bits;
}

// CeilLog2 from §4.11.1 for value >= 1.
std::uint32_t ceil_log2(std::uint32_t value) {
  if (value <= 1U) return 0U;
  std::uint32_t bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0U) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

void put_trailing_bits(BitWriter& writer) {
  writer.put_bit(true);
  writer.align_to_byte();
}

std::uint32_t mi_cols(const SequenceHeaderConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_width) + 7U) >> 3U);
}
std::uint32_t mi_rows(const SequenceHeaderConfig& c) {
  return 2U * ((static_cast<std::uint32_t>(c.max_frame_height) + 7U) >> 3U);
}

void write_tile_info(BitWriter& writer, const SequenceHeaderConfig& c) {
  const std::uint32_t sb_cols = (mi_cols(c) + 15U) >> 4U;
  const std::uint32_t sb_rows = (mi_rows(c) + 15U) >> 4U;
  assert(sb_cols >= 1U && sb_rows >= 1U);
  writer.put_bit(true);  // uniform_tile_spacing_flag
  if (sb_cols > 1U) writer.put_bit(false);  // increment_tile_cols_log2
  if (sb_rows > 1U) writer.put_bit(false);  // increment_tile_rows_log2
}

bool validate_header_config(const SequenceHeaderConfig& c, std::string& error) {
  if (c.max_frame_width == 0 || c.max_frame_height == 0) {
    return fail(error, "frame dimensions must be non-zero");
  }
  if (c.seq_level_idx > 31U) {
    return fail(error, "seq_level_idx is outside f(5)");
  }
  if (c.order_hint_bits < 1U || c.order_hint_bits > 16U) {
    return fail(error, "order_hint_bits is outside 1..16");
  }
  if (c.num_ref_frames < 1U || c.num_ref_frames > 16U) {
    return fail(error, "num_ref_frames is outside 1..16");
  }
  if (c.max_drl_bits_minus_1 >= 5U) {
    return fail(error, "max_drl_bits_minus_1 is outside 0..4");
  }
  return validate_sequence_layer_config(c.layers, error);
}

void write_spec_layer_blocks(BitWriter& writer,
                             const SequenceLayerConfig& layers) {
  if (layers.max_mlayer_id > 0) {
    writer.put_bit(layers.mlayer_dependency_present_flag);
    if (layers.mlayer_dependency_present_flag) {
      for (int curr = 1; curr <= layers.max_mlayer_id; ++curr) {
        for (int ref = curr; ref >= 0; --ref) {
          writer.put_bit(
              layers.dependencies.mlayer[static_cast<std::size_t>(curr)]
                                      [static_cast<std::size_t>(ref)]);
        }
      }
    }
  }
  if (layers.max_tlayer_id > 0) {
    writer.put_bit(layers.tlayer_dependency_present_flag);
    if (layers.tlayer_dependency_present_flag) {
      if (layers.max_mlayer_id > 0) {
        writer.put_bit(layers.multi_tlayer_dependency_map_present_flag);
      }
      for (int m = 0; m <= layers.max_mlayer_id; ++m) {
        for (int curr = 1; curr <= layers.max_tlayer_id; ++curr) {
          for (int ref = curr; ref >= 0; --ref) {
            if (layers.multi_tlayer_dependency_map_present_flag || m == 0) {
              writer.put_bit(
                  layers.dependencies.tlayer[static_cast<std::size_t>(m)]
                                           [static_cast<std::size_t>(curr)]
                                           [static_cast<std::size_t>(ref)]);
            }
          }
        }
      }
    }
  }
}

// dav2d f9dafbc parses the dependency blocks in this order, with
// `n = 1..max-1` and an n-bit mask per row. It is not the v1.0.0 syntax.
template <std::size_t N>
void write_compat_map_rows(BitWriter& writer,
                           const std::array<std::array<bool, N>, N>& map,
                           int maximum) {
  for (int n = 1; n < maximum; ++n) {
    std::uint32_t mask = 0;
    for (int r = 0; r < n; ++r) {
      if (map[static_cast<std::size_t>(n)][static_cast<std::size_t>(r)]) {
        mask |= 1U << static_cast<unsigned>(r);
      }
    }
    writer.put_bits(mask, static_cast<unsigned>(n));
  }
}

void write_compat_layer_blocks(BitWriter& writer,
                               const SequenceLayerConfig& layers) {
  if (layers.max_tlayer_id > 0) {
    writer.put_bit(layers.tlayer_dependency_present_flag);
    if (layers.tlayer_dependency_present_flag) {
      // The decoder uses one map for all embedded layers.
      write_compat_map_rows(writer, layers.dependencies.tlayer[0],
                            layers.max_tlayer_id);
    }
  }
  if (layers.max_mlayer_id > 0) {
    writer.put_bit(layers.mlayer_dependency_present_flag);
    if (layers.mlayer_dependency_present_flag) {
      write_compat_map_rows(writer, layers.dependencies.mlayer,
                            layers.max_mlayer_id);
    }
  }
}

void write_baseline_tail(BitWriter& writer, const SequenceHeaderConfig& c) {
  // §5.4.3 sequence_partition_config().
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

  // §5.4.6 sequence_inter_config().
  for (int mode = 1; mode < 5; ++mode) {
    writer.put_bit(false);  // seq_enabled_motion_modes[mode]
  }
  writer.put_bit(false);  // enable_masked_compound
  writer.put_bit(false);  // enable_ref_frame_mvs
  writer.put_bits(static_cast<std::uint64_t>(c.order_hint_bits) - 1U, 4U);
  writer.put_bit(false);  // enable_refmvbank
  writer.put_bit(true);   // disable_drl_reorder
  writer.put_bit(true);   // explicit_ref_frame_map
  writer.put_bit(true);   // explicit_num_ref_frames
  writer.put_bits(static_cast<std::uint64_t>(c.num_ref_frames) - 1U, 4U);
  writer.put_bits(0U, 3U);  // long_term_frame_id_bits
  put_ns(writer, 5U, c.max_drl_bits_minus_1);
  writer.put_bit(false);   // allow_frame_max_drl_bits
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
  writer.put_bit(false);   // enable_short_refresh_frame_flags

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
  writer.put_bits(0U, 2U); // df_par_bits_minus_2

  // §5.4.2 sequence_tile_config().
  writer.put_bit(false);  // seq_tile_info_present_flag

  writer.put_bit(false);  // film_grain_params_present
}

void append_leb128(std::vector<std::uint8_t>& out, std::uint32_t value) {
  do {
    std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7FU);
    value >>= 7U;
    if (value != 0U) byte = static_cast<std::uint8_t>(byte | 0x80U);
    out.push_back(byte);
  } while (value != 0U);
}

bool read_leb128(std::span<const std::uint8_t> bytes, std::size_t& offset,
                 std::uint32_t& value) {
  value = 0;
  unsigned shift = 0;
  while (offset < bytes.size()) {
    const std::uint8_t byte = bytes[offset++];
    value |= static_cast<std::uint32_t>(byte & 0x7FU) << shift;
    if ((byte & 0x80U) == 0U) return true;
    shift += 7U;
    if (shift >= 32U) return false;
  }
  return false;
}

class BitReader {
 public:
  BitReader(std::span<const std::uint8_t> bytes, std::size_t bit_offset)
      : bytes_(bytes), bit_pos_(bit_offset) {}

  bool get_bit(bool& out) {
    if (bit_pos_ >= bytes_.size() * 8U) return false;
    const std::size_t byte = bit_pos_ >> 3U;
    const unsigned shift = 7U - static_cast<unsigned>(bit_pos_ & 7U);
    out = ((bytes_[byte] >> shift) & 1U) != 0U;
    ++bit_pos_;
    return true;
  }

  bool get_bits(unsigned count, std::uint32_t& out) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < count; ++i) {
      bool bit = false;
      if (!get_bit(bit)) return false;
      value = (value << 1U) | (bit ? 1U : 0U);
    }
    out = value;
    return true;
  }

  bool get_uvlc(std::uint32_t& out) {
    unsigned leading_zeros = 0;
    bool bit = false;
    while (true) {
      if (!get_bit(bit)) return false;
      if (bit) break;
      ++leading_zeros;
      if (leading_zeros > 31U) return false;
    }
    std::uint32_t payload = 0;
    if (leading_zeros > 0 && !get_bits(leading_zeros, payload)) return false;
    out = payload + ((1U << leading_zeros) - 1U);
    return true;
  }

  bool get_ns(std::uint32_t n, std::uint32_t& out) {
    if (n < 2U) return false;
    unsigned w = 0;
    while ((1U << w) < n) ++w;
    const std::uint32_t m = (1U << w) - n;
    std::uint32_t value = 0;
    if (!get_bits(w - 1U, value)) return false;
    if (value < m) {
      out = value;
      return true;
    }
    bool extra = false;
    if (!get_bit(extra)) return false;
    out = ((value << 1U) | (extra ? 1U : 0U)) - m;
    return true;
  }

  std::size_t bit_position() const { return bit_pos_; }
  std::size_t bits_left() const {
    const std::size_t total = bytes_.size() * 8U;
    return bit_pos_ >= total ? 0 : total - bit_pos_;
  }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_pos_ = 0;
};

bool expect_zero_bits(BitReader& reader, unsigned count, const char* name,
                      std::string& error) {
  std::uint32_t value = 0;
  if (!reader.get_bits(count, value)) {
    return fail(error, std::string("truncated ") + name);
  }
  if (value != 0U) {
    return fail(error, std::string(name) + " is not the baseline value");
  }
  return true;
}

bool read_spec_layer_blocks(BitReader& reader, ParsedSequenceHeader& out,
                            std::string& error) {
  const int max_m = out.max_mlayer_id;
  const int max_t = out.max_tlayer_id;
  out.dependencies = default_dependency_maps(max_m, max_t);

  if (max_m > 0) {
    bool bit = false;
    if (!reader.get_bit(bit)) return fail(error, "mlayer_dependency_present_flag");
    out.mlayer_dependency_present_flag = bit;
    if (bit) {
      for (int curr = 1; curr <= max_m; ++curr) {
        for (int ref = curr; ref >= 0; --ref) {
          bool dep = false;
          if (!reader.get_bit(dep)) return fail(error, "mlayer_dependency_map");
          out.dependencies.mlayer[static_cast<std::size_t>(curr)]
                               [static_cast<std::size_t>(ref)] = dep;
          out.mlayer_dependency_coded[static_cast<std::size_t>(curr)]
                                    [static_cast<std::size_t>(ref)] = true;
        }
      }
    }
  }
  if (max_t > 0) {
    bool bit = false;
    if (!reader.get_bit(bit)) return fail(error, "tlayer_dependency_present_flag");
    out.tlayer_dependency_present_flag = bit;
    if (bit) {
      if (max_m > 0) {
        if (!reader.get_bit(bit)) {
          return fail(error, "multi_tlayer_dependency_map_present_flag");
        }
        out.multi_tlayer_dependency_map_present_flag = bit;
      }
      for (int m = 0; m <= max_m; ++m) {
        for (int curr = 1; curr <= max_t; ++curr) {
          for (int ref = curr; ref >= 0; --ref) {
            if (out.multi_tlayer_dependency_map_present_flag || m == 0) {
              bool dep = false;
              if (!reader.get_bit(dep)) return fail(error, "tlayer_dependency_map");
              out.dependencies.tlayer[static_cast<std::size_t>(m)]
                                    [static_cast<std::size_t>(curr)]
                                    [static_cast<std::size_t>(ref)] = dep;
              out.tlayer_dependency_coded[static_cast<std::size_t>(m)]
                                         [static_cast<std::size_t>(curr)]
                                         [static_cast<std::size_t>(ref)] = true;
            }
          }
        }
      }
      // multi_tlayer_dependency_map_present_flag == 0: the decoder copies the
      // coded layer-0 map into every remaining embedded layer.
      if (!out.multi_tlayer_dependency_map_present_flag) {
        for (int m = 1; m <= max_m; ++m) {
          out.dependencies.tlayer[static_cast<std::size_t>(m)] =
              out.dependencies.tlayer[0];
        }
      }
    } else {
      // Inferred default map.
      out.dependencies.tlayer = default_dependency_maps(max_m, max_t).tlayer;
    }
  }
  return true;
}

bool read_compat_layer_blocks(BitReader& reader, ParsedSequenceHeader& out,
                              std::string& error) {
  const int max_m = out.max_mlayer_id;
  const int max_t = out.max_tlayer_id;
  out.dependencies = default_dependency_maps(max_m, max_t);

  if (max_t > 0) {
    bool bit = false;
    if (!reader.get_bit(bit)) return fail(error, "tlayer_dependency_present_flag");
    out.tlayer_dependency_present_flag = bit;
    if (bit) {
      for (int n = 1; n < max_t; ++n) {
        std::uint32_t mask = 0;
        if (!reader.get_bits(static_cast<unsigned>(n), mask)) {
          return fail(error, "tlayer_dependency_map");
        }
        for (int r = 0; r < n; ++r) {
          if ((mask & (1U << static_cast<unsigned>(r))) != 0U) {
            out.dependencies.tlayer[static_cast<std::size_t>(n)]
                                  [static_cast<std::size_t>(n)]
                                  [static_cast<std::size_t>(r)] = true;
            out.tlayer_dependency_coded[static_cast<std::size_t>(n)]
                                       [static_cast<std::size_t>(n)]
                                       [static_cast<std::size_t>(r)] = true;
          }
        }
      }
    }
  }
  if (max_m > 0) {
    bool bit = false;
    if (!reader.get_bit(bit)) return fail(error, "mlayer_dependency_present_flag");
    out.mlayer_dependency_present_flag = bit;
    if (bit) {
      for (int n = 1; n < max_m; ++n) {
        std::uint32_t mask = 0;
        if (!reader.get_bits(static_cast<unsigned>(n), mask)) {
          return fail(error, "mlayer_dependency_map");
        }
        for (int r = 0; r < n; ++r) {
          if ((mask & (1U << static_cast<unsigned>(r))) != 0U) {
            out.dependencies.mlayer[static_cast<std::size_t>(n)]
                                  [static_cast<std::size_t>(r)] = true;
            out.mlayer_dependency_coded[static_cast<std::size_t>(n)]
                                      [static_cast<std::size_t>(r)] = true;
          }
        }
      }
    }
  }
  return true;
}

bool read_baseline_tail(BitReader& reader, ParsedSequenceHeader& out,
                        std::string& error) {
  // §5.4.3 partition config. This writer emits the all-zero baseline.
  if (!expect_zero_bits(reader, 5U, "sequence_partition_config", error)) {
    return false;
  }
  // §5.4.4 segment config.
  if (!expect_zero_bits(reader, 2U, "sequence_segment_config", error)) {
    return false;
  }
  // §5.4.5 intra config: dip, edge, mrls, cfl, cfl_ds(2), mhccp, ibp.
  bool dip = false, edge = false, mrls = false, cfl = false, mhccp = false,
       ibp = false;
  std::uint32_t cfl_ds = 0;
  if (!reader.get_bit(dip) || !reader.get_bit(edge) || !reader.get_bit(mrls) ||
      !reader.get_bit(cfl) || !reader.get_bits(2U, cfl_ds) ||
      !reader.get_bit(mhccp) || !reader.get_bit(ibp)) {
    return fail(error, "sequence_intra_config");
  }
  (void)dip; (void)edge; (void)mrls; (void)cfl; (void)cfl_ds;
  (void)mhccp; (void)ibp;

  // §5.4.6 sequence_inter_config(). This writer disables every tool that
  // introduces conditional syntax.
  if (!expect_zero_bits(reader, 4U, "seq_enabled_motion_modes", error)) {
    return false;
  }
  bool masked = false, ref_frame_mvs = false;
  if (!reader.get_bit(masked) || !reader.get_bit(ref_frame_mvs)) {
    return fail(error, "sequence_inter_config");
  }
  if (masked || ref_frame_mvs) {
    return fail(error,
                "reader only supports the baseline sequence_inter_config");
  }
  std::uint32_t order_hint_bits_minus_1 = 0;
  if (!reader.get_bits(4U, order_hint_bits_minus_1)) {
    return fail(error, "order_hint_bits_minus_1");
  }
  out.order_hint_bits = static_cast<int>(order_hint_bits_minus_1) + 1;
  bool refmvbank = false, disable_drl = false;
  if (!reader.get_bit(refmvbank) || !reader.get_bit(disable_drl)) {
    return fail(error, "sequence_inter_config");
  }
  if (refmvbank || !disable_drl) {
    return fail(error,
                "reader only supports DRL_REORDER_DISABLED without ref-mv bank");
  }
  bool explicit_ref_frame_map = false, explicit_num_ref_frames = false;
  if (!reader.get_bit(explicit_ref_frame_map) ||
      !reader.get_bit(explicit_num_ref_frames)) {
    return fail(error, "explicit reference config");
  }
  if (!explicit_ref_frame_map || !explicit_num_ref_frames) {
    return fail(error,
                "reader only supports the baseline explicit reference config");
  }
  std::uint32_t num_ref_frames_minus_1 = 0;
  if (!reader.get_bits(4U, num_ref_frames_minus_1)) {
    return fail(error, "num_ref_frames_minus_1");
  }
  out.num_ref_frames = static_cast<int>(num_ref_frames_minus_1) + 1;
  if (!expect_zero_bits(reader, 3U, "long_term_frame_id_bits", error)) {
    return false;
  }
  std::uint32_t drl = 0, bvp = 0;
  if (!reader.get_ns(5U, drl) || !reader.get_ns(3U, bvp)) {
    return fail(error, "seq_max_drl_bits_minus_1");
  }
  if (!expect_zero_bits(reader, 1U, "allow_frame_max_drl_bits", error) ||
      !expect_zero_bits(reader, 1U, "allow_frame_max_bvp_drl_bits", error) ||
      !expect_zero_bits(reader, 2U, "num_same_ref_compound", error) ||
      !expect_zero_bits(reader, 1U, "enable_tip", error) ||
      !expect_zero_bits(reader, 1U, "enable_mv_traj", error) ||
      !expect_zero_bits(reader, 1U, "enable_bawp", error) ||
      !expect_zero_bits(reader, 1U, "enable_cwp", error) ||
      !expect_zero_bits(reader, 1U, "enable_imp_msk_bld", error) ||
      !expect_zero_bits(reader, 1U, "enable_df_sub_pu", error) ||
      !expect_zero_bits(reader, 2U, "enable_opfl_refine", error) ||
      !expect_zero_bits(reader, 1U, "enable_refinemv", error) ||
      !expect_zero_bits(reader, 1U, "enable_bru", error) ||
      !expect_zero_bits(reader, 1U, "enable_adaptive_mvd", error) ||
      !expect_zero_bits(reader, 1U, "enable_mvd_sign_derive", error) ||
      !expect_zero_bits(reader, 1U, "enable_flex_mvres", error) ||
      !expect_zero_bits(reader, 1U, "enable_global_motion", error) ||
      !expect_zero_bits(reader, 1U, "enable_short_refresh_frame_flags",
                        error)) {
    return false;
  }

  // §5.4.7 sequence_scc_config().
  if (!expect_zero_bits(reader, 2U, "sequence_scc_config", error)) {
    return false;
  }

  // §5.4.8 sequence_transform_quant_entropy_config().
  if (!expect_zero_bits(reader, 1U, "enable_fsc", error) ||
      !expect_zero_bits(reader, 1U, "enable_idtx_intra", error) ||
      !expect_zero_bits(reader, 1U, "enable_intra_ist", error) ||
      !expect_zero_bits(reader, 1U, "enable_inter_ist", error)) {
    return false;
  }
  bool chroma_dctonly = false;
  if (!reader.get_bit(chroma_dctonly)) {
    return fail(error, "enable_chroma_dctonly");
  }
  if (!chroma_dctonly) {
    return fail(error, "enable_chroma_dctonly must be 1 in the baseline");
  }
  if (!expect_zero_bits(reader, 1U, "enable_inter_ddt", error) ||
      !expect_zero_bits(reader, 1U, "reduced_tx_part_set", error) ||
      !expect_zero_bits(reader, 1U, "enable_cctx", error) ||
      !expect_zero_bits(reader, 1U, "enable_tcq", error) ||
      !expect_zero_bits(reader, 1U, "enable_parity_hiding", error) ||
      !expect_zero_bits(reader, 1U, "enable_avg_cdf", error) ||
      !expect_zero_bits(reader, 1U, "separate_uv_delta_q", error)) {
    return false;
  }
  bool equal_ac_dc_q = false;
  if (!reader.get_bit(equal_ac_dc_q)) return fail(error, "equal_ac_dc_q");
  if (!equal_ac_dc_q) {
    return fail(error, "equal_ac_dc_q must be 1 in the baseline");
  }
  std::uint32_t base_uv_ac_delta_q = 0;
  if (!reader.get_bits(5U, base_uv_ac_delta_q)) {
    return fail(error, "base_uv_ac_delta_q");
  }
  if (base_uv_ac_delta_q != 23U) {
    return fail(error, "base_uv_ac_delta_q must be 23 in the baseline");
  }
  if (!expect_zero_bits(reader, 1U, "uv_ac_delta_q_enabled", error) ||
      !expect_zero_bits(reader, 1U, "disable_loopfilters_across_tiles", error) ||
      !expect_zero_bits(reader, 1U, "enable_cdef", error) ||
      !expect_zero_bits(reader, 1U, "enable_gdf", error) ||
      !expect_zero_bits(reader, 1U, "enable_restoration", error) ||
      !expect_zero_bits(reader, 1U, "enable_ccso", error) ||
      !expect_zero_bits(reader, 1U, "cdef_on_skip_txfm_always_on", error)) {
    return false;
  }
  bool cdef_disabled = false;
  if (!reader.get_bit(cdef_disabled)) {
    return fail(error, "cdef_on_skip_txfm_disabled");
  }
  if (!cdef_disabled) {
    return fail(error, "cdef_on_skip_txfm_disabled must be 1 in the baseline");
  }
  if (!expect_zero_bits(reader, 2U, "df_par_bits_minus_2", error) ||
      !expect_zero_bits(reader, 1U, "seq_tile_info_present_flag", error) ||
      !expect_zero_bits(reader, 1U, "film_grain_params_present", error)) {
    return false;
  }
  return true;
}

}  // namespace

ObuHeaderFields inferred_obu_header_fields(std::uint8_t obu_type,
                                           std::uint8_t tlayer_id) {
  ObuHeaderFields fields;
  fields.extension_flag = false;
  fields.obu_type = obu_type;
  fields.tlayer_id = tlayer_id;
  fields.mlayer_id = 0;
  fields.xlayer_id =
      (obu_type == kObuTemporalDelimiter || obu_type == kObuMsdo)
          ? static_cast<std::uint8_t>(kGlobalXLayerId)
          : static_cast<std::uint8_t>(0);
  return fields;
}

bool validate_obu_header_fields(const ObuHeaderFields& f, std::string& error) {
  if (f.obu_type > 31U) {
    return fail(error, "obu_type is outside f(5)");
  }
  if (f.tlayer_id > kMaxTlayerId) {
    return fail(error, "obu_tlayer_id is outside f(2)");
  }
  if (f.mlayer_id > kMaxMlayerId) {
    return fail(error, "obu_mlayer_id is outside f(3)");
  }
  if (f.xlayer_id > kGlobalXLayerId) {
    return fail(error, "obu_xlayer_id is outside f(5)");
  }
  if (!f.extension_flag) {
    const ObuHeaderFields inferred =
        inferred_obu_header_fields(f.obu_type, f.tlayer_id);
    if (f.mlayer_id != inferred.mlayer_id ||
        f.xlayer_id != inferred.xlayer_id) {
      return fail(error,
                  "OBU header extension flag is 0 but the layer ids differ "
                  "from the §5.2.2 inference");
    }
  }
  const bool global = f.xlayer_id == kGlobalXLayerId;
  if (global && (f.mlayer_id != 0U || f.tlayer_id != 0U)) {
    return fail(error,
                "GLOBAL_XLAYER_ID requires obu_mlayer_id = obu_tlayer_id = 0");
  }
  if (global) {
    switch (f.obu_type) {
      case 2:   // OBU_TEMPORAL_DELIMITER
      case 8:   // OBU_METADATA_SHORT
      case 9:   // OBU_METADATA_GROUP
      case 15:  // OBU_BUFFER_REMOVAL_TIMING
      case 16:  // OBU_LAYER_CONFIGURATION_RECORD
      case 17:  // OBU_ATLAS_SEGMENT
      case 18:  // OBU_OPERATING_POINT_SET
      case 20:  // OBU_MSDO
      case 25:  // OBU_PADDING
        break;
      default:
        return fail(error,
                    "OBU type is not allowed with obu_xlayer_id = "
                    "GLOBAL_XLAYER_ID");
    }
  }
  if ((f.obu_type == kObuTemporalDelimiter || f.obu_type == kObuMsdo) &&
      !global) {
    return fail(error,
                "OBU_TEMPORAL_DELIMITER/OBU_MSDO require GLOBAL_XLAYER_ID");
  }
  switch (f.obu_type) {
    case 1:   // OBU_SEQUENCE_HEADER
    case 2:   // OBU_TEMPORAL_DELIMITER
    case 16:  // OBU_LAYER_CONFIGURATION_RECORD
    case 17:  // OBU_ATLAS_SEGMENT
    case 18:  // OBU_OPERATING_POINT_SET
      if (f.mlayer_id != 0U || f.tlayer_id != 0U) {
        return fail(error,
                    "layer-non-specific OBU requires obu_mlayer_id = "
                    "obu_tlayer_id = 0");
      }
      break;
    case 4:   // OBU_CLOSED_LOOP_KEY
    case 5:   // OBU_OPEN_LOOP_KEY
    case 10:  // OBU_SWITCH
    case 21:  // OBU_RAS_FRAME
      if (f.tlayer_id != 0U) {
        return fail(error,
                    "CLK/OLK/SWITCH/RAS require obu_tlayer_id = 0");
      }
      break;
    default:
      break;
  }
  return true;
}

std::vector<std::uint8_t> make_obu_header_bytes(
    const ObuHeaderFields& fields) {
  std::vector<std::uint8_t> out;
  unsigned value = (fields.extension_flag ? 0x80U : 0x00U) |
                   ((static_cast<unsigned>(fields.obu_type) & 0x1FU) << 2U) |
                   (static_cast<unsigned>(fields.tlayer_id) & 0x03U);
  out.push_back(static_cast<std::uint8_t>(value));
  if (fields.extension_flag) {
    value = ((static_cast<unsigned>(fields.mlayer_id) & 0x07U) << 5U) |
            (static_cast<unsigned>(fields.xlayer_id) & 0x1FU);
    out.push_back(static_cast<std::uint8_t>(value));
  }
  return out;
}

bool make_obu_header(std::uint8_t obu_type, std::uint8_t tlayer_id,
                     std::uint8_t mlayer_id, std::uint8_t xlayer_id,
                     std::vector<std::uint8_t>& out, std::string& error) {
  const ObuHeaderFields inferred =
      inferred_obu_header_fields(obu_type, tlayer_id);
  ObuHeaderFields fields;
  fields.obu_type = obu_type;
  fields.tlayer_id = tlayer_id;
  fields.mlayer_id = mlayer_id;
  fields.xlayer_id = xlayer_id;
  fields.extension_flag =
      (mlayer_id != inferred.mlayer_id || xlayer_id != inferred.xlayer_id);
  if (!validate_obu_header_fields(fields, error)) return false;
  out = make_obu_header_bytes(fields);
  return true;
}

bool parse_obu_header(std::span<const std::uint8_t> bytes, std::size_t offset,
                      ObuHeaderFields& fields, std::size_t& header_bytes,
                      std::string& error) {
  if (offset >= bytes.size()) {
    return fail(error, "truncated OBU header");
  }
  const std::uint8_t first = bytes[offset];
  fields.extension_flag = (first & 0x80U) != 0U;
  fields.obu_type = static_cast<std::uint8_t>((first >> 2U) & 0x1FU);
  fields.tlayer_id = static_cast<std::uint8_t>(first & 0x03U);
  header_bytes = 1;
  if (fields.extension_flag) {
    if (offset + 1 >= bytes.size()) {
      return fail(error, "truncated OBU extension header");
    }
    const std::uint8_t ext = bytes[offset + 1];
    fields.mlayer_id = static_cast<std::uint8_t>((ext >> 5U) & 0x07U);
    fields.xlayer_id = static_cast<std::uint8_t>(ext & 0x1FU);
    header_bytes = 2;
  } else {
    const ObuHeaderFields inferred =
        inferred_obu_header_fields(fields.obu_type, fields.tlayer_id);
    fields.mlayer_id = inferred.mlayer_id;
    fields.xlayer_id = inferred.xlayer_id;
  }
  return true;
}

SequenceHeaderConfig baseline_sequence_header_config(
    std::uint16_t width, std::uint16_t height, std::uint8_t base_q_idx,
    std::uint8_t num_ref_frames) {
  SequenceHeaderConfig config;
  config.max_frame_width = width;
  config.max_frame_height = height;
  config.base_q_idx = base_q_idx;
  config.num_ref_frames = num_ref_frames;
  config.order_hint_bits = 4;
  config.max_drl_bits_minus_1 = 4;
  config.disable_cdf_update = true;
  config.enable_cfl_intra = true;
  config.monotonic_output_order_flag = true;
  config.layers = default_sequence_layer_config(0, 0);
  return config;
}

bool write_sequence_header_payload(const SequenceHeaderConfig& config,
                                   SequenceSyntaxDialect dialect,
                                   std::vector<std::uint8_t>& payload,
                                   std::size_t& bit_count,
                                   std::string& error) {
  if (!validate_header_config(config, error)) return false;
  const SequenceLayerConfig& layers = config.layers;
  BitWriter writer;

  put_uvlc(writer, 0);                    // seq_header_id
  writer.put_bits(0U, 5U);                // seq_profile_idc = 0
  writer.put_bit(false);                  // single_picture_header_flag = 0
  writer.put_bits(config.seq_level_idx, 5U);  // seq_level_idx
  if (config.seq_level_idx > 3U) {
    writer.put_bit(false);                // seq_tier = 0 (Main)
  }
  put_uvlc(writer, 0);                    // chroma_format_idc = 4:2:0
  put_uvlc(writer, 1);                    // bit_depth_idc = 1 => 8-bit

  // §5.4.1 non-single-picture header.
  writer.put_bits(0U, 3U);                // seq_lcr_id
  writer.put_bit(false);                  // still_picture
  writer.put_bits(static_cast<std::uint64_t>(layers.max_tlayer_id), 2U);
  writer.put_bits(static_cast<std::uint64_t>(layers.max_mlayer_id), 3U);
  if (layers.max_mlayer_id > 0 && dialect == SequenceSyntaxDialect::kSpecV1_0_0) {
    const std::uint32_t n = ceil_log2(
        static_cast<std::uint32_t>(layers.max_mlayer_id) + 1U);
    writer.put_bits(
        static_cast<std::uint64_t>(layers.seq_max_mlayer_cnt - 1), n);
  }
  writer.put_bit(config.monotonic_output_order_flag);

  unsigned width_bits =
      bit_width(static_cast<std::uint32_t>(config.max_frame_width) - 1U);
  if (width_bits == 0) width_bits = 1;
  unsigned height_bits =
      bit_width(static_cast<std::uint32_t>(config.max_frame_height) - 1U);
  if (height_bits == 0) height_bits = 1;
  writer.put_bits(width_bits - 1U, 4U);
  writer.put_bits(height_bits - 1U, 4U);
  writer.put_bits(static_cast<std::uint64_t>(config.max_frame_width) - 1U,
                  width_bits);
  writer.put_bits(static_cast<std::uint64_t>(config.max_frame_height) - 1U,
                  height_bits);

  writer.put_bit(false);  // seq_cropping_window_present_flag
  writer.put_bit(false);  // seq_initial_display_delay_present_flag
  writer.put_bit(false);  // decoder_model_info_present_flag

  if (dialect == SequenceSyntaxDialect::kSpecV1_0_0) {
    write_spec_layer_blocks(writer, layers);
  } else {
    write_compat_layer_blocks(writer, layers);
  }

  write_baseline_tail(writer, config);
  bit_count = writer.bit_size();
  payload = writer.take_bytes();
  return true;
}

bool make_sequence_header_obu(const SequenceHeaderConfig& config,
                              SequenceSyntaxDialect dialect,
                              std::vector<std::uint8_t>& out,
                              std::string& error) {
  std::vector<std::uint8_t> payload;
  std::size_t bit_count = 0;
  if (!write_sequence_header_payload(config, dialect, payload, bit_count,
                                     error)) {
    return false;
  }
  BitWriter writer;
  for (std::size_t i = 0; i < bit_count; ++i) {
    const std::uint8_t byte = payload[i >> 3U];
    const unsigned shift = 7U - static_cast<unsigned>(i & 7U);
    writer.put_bit(((byte >> shift) & 1U) != 0U);
  }
  writer.put_bit(false);  // obu_extension_flag (§5.2.1 extensible OBU tail)
  put_trailing_bits(writer);
  out = make_obu(static_cast<std::uint8_t>(ObuType::kSequenceHeader), 0, 0, 0,
                 writer.take_bytes());
  return true;
}

bool parse_obu(std::span<const std::uint8_t> bytes, std::size_t offset,
               ParsedObu& out, std::string& error) {
  std::uint32_t size = 0;
  std::size_t pos = offset;
  if (!read_leb128(bytes, pos, size)) {
    return fail(error, "invalid OBU leb128 length");
  }
  if (size == 0 || pos >= bytes.size()) {
    return fail(error, "empty OBU");
  }
  std::size_t header_bytes = 0;
  if (!parse_obu_header(bytes, pos, out.header, header_bytes, error)) {
    return false;
  }
  if (!validate_obu_header_fields(out.header, error)) return false;
  const std::size_t total = static_cast<std::size_t>(size);
  if (pos + total > bytes.size() || total < header_bytes) {
    return fail(error, "truncated OBU payload");
  }
  out.total_bytes = (pos - offset) + total;
  out.payload = bytes.subspan(pos + header_bytes, total - header_bytes);
  return true;
}

std::vector<std::uint8_t> make_obu(std::uint8_t obu_type,
                                   std::uint8_t tlayer_id,
                                   std::uint8_t mlayer_id,
                                   std::uint8_t xlayer_id,
                                   std::span<const std::uint8_t> payload,
                                   bool* extension_flag) {
  const ObuHeaderFields inferred =
      inferred_obu_header_fields(obu_type, tlayer_id);
  ObuHeaderFields fields;
  fields.obu_type = obu_type;
  fields.tlayer_id = tlayer_id;
  fields.mlayer_id = mlayer_id;
  fields.xlayer_id = xlayer_id;
  fields.extension_flag =
      (mlayer_id != inferred.mlayer_id || xlayer_id != inferred.xlayer_id);
  const std::vector<std::uint8_t> header = make_obu_header_bytes(fields);
  if (extension_flag != nullptr) *extension_flag = fields.extension_flag;
  std::vector<std::uint8_t> out;
  append_leb128(out, static_cast<std::uint32_t>(header.size() + payload.size()));
  out.insert(out.end(), header.begin(), header.end());
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

bool parse_sequence_header_obu(std::span<const std::uint8_t> obu,
                               SequenceSyntaxDialect dialect,
                               ParsedSequenceHeader& out, std::string& error) {
  ParsedObu envelope;
  if (!parse_obu(obu, 0, envelope, error)) return false;
  if (static_cast<int>(envelope.header.obu_type) != kObuSequenceHeader) {
    return fail(error, "not a sequence header OBU");
  }
  if (envelope.header.extension_flag ||
      envelope.header.mlayer_id != 0 ||
      envelope.header.xlayer_id != 0) {
    return fail(error,
                "OBU_SEQUENCE_HEADER requires mlayer 0 / xlayer 0 (§6.2.2)");
  }
  if (!validate_obu_header_fields(envelope.header, error)) return false;

  BitReader reader(envelope.payload, 0);
  std::uint32_t v = 0;
  if (!reader.get_uvlc(v)) return fail(error, "seq_header_id");
  out.seq_header_id = static_cast<int>(v);
  if (!reader.get_bits(5U, v)) return fail(error, "seq_profile_idc");
  out.seq_profile_idc = static_cast<int>(v);
  bool bit = false;
  if (!reader.get_bit(bit)) return fail(error, "single_picture_header_flag");
  out.single_picture_header_flag = bit;
  if (!reader.get_bits(5U, v)) return fail(error, "seq_level_idx");
  out.seq_level_idx = static_cast<int>(v);
  if (out.seq_level_idx > 3 && !out.single_picture_header_flag) {
    if (!reader.get_bit(bit)) return fail(error, "seq_tier");
    if (bit) return fail(error, "high tier is not supported by this reader");
  }
  if (!reader.get_uvlc(v)) return fail(error, "chroma_format_idc");
  out.chroma_format_idc = static_cast<int>(v);
  if (!reader.get_uvlc(v)) return fail(error, "bit_depth_idc");
  out.bit_depth_idc = static_cast<int>(v);

  if (out.single_picture_header_flag) {
    out.seq_lcr_id = 0;
    out.still_picture = true;
    out.max_tlayer_id = 0;
    out.max_mlayer_id = 0;
    out.seq_max_mlayer_cnt = 1;
    out.monotonic_output_order_flag = true;
  } else {
    if (!reader.get_bits(3U, v)) return fail(error, "seq_lcr_id");
    out.seq_lcr_id = static_cast<int>(v);
    if (!reader.get_bit(bit)) return fail(error, "still_picture");
    out.still_picture = bit;
    if (!reader.get_bits(2U, v)) return fail(error, "max_tlayer_id");
    out.max_tlayer_id = static_cast<int>(v);
    if (!reader.get_bits(3U, v)) return fail(error, "max_mlayer_id");
    out.max_mlayer_id = static_cast<int>(v);
    if (out.max_mlayer_id > 0 &&
        dialect == SequenceSyntaxDialect::kSpecV1_0_0) {
      const std::uint32_t n = ceil_log2(
          static_cast<std::uint32_t>(out.max_mlayer_id) + 1U);
      if (!reader.get_bits(n, v)) {
        return fail(error, "seq_max_mlayer_cnt_minus_1");
      }
      out.seq_max_mlayer_cnt = static_cast<int>(v) + 1;
      out.seq_max_mlayer_cnt_coded = true;
    } else {
      out.seq_max_mlayer_cnt = 1;
      out.seq_max_mlayer_cnt_coded = false;
    }
    if (!reader.get_bit(bit)) {
      return fail(error, "monotonic_output_order_flag");
    }
    out.monotonic_output_order_flag = bit;
  }
  out.dialect = dialect;

  std::uint32_t width_bits_minus_1 = 0, height_bits_minus_1 = 0;
  if (!reader.get_bits(4U, width_bits_minus_1) ||
      !reader.get_bits(4U, height_bits_minus_1)) {
    return fail(error, "frame size bits");
  }
  std::uint32_t width_minus_1 = 0, height_minus_1 = 0;
  if (!reader.get_bits(width_bits_minus_1 + 1U, width_minus_1) ||
      !reader.get_bits(height_bits_minus_1 + 1U, height_minus_1)) {
    return fail(error, "frame size");
  }
  out.max_frame_width = static_cast<int>(width_minus_1) + 1;
  out.max_frame_height = static_cast<int>(height_minus_1) + 1;

  if (!reader.get_bit(bit)) {
    return fail(error, "seq_cropping_window_present_flag");
  }
  if (bit) {
    for (int i = 0; i < 4; ++i) {
      if (!reader.get_uvlc(v)) return fail(error, "seq_cropping_window");
    }
  }
  if (!out.single_picture_header_flag) {
    if (!reader.get_bit(bit)) {
      return fail(error, "seq_initial_display_delay_present_flag");
    }
    if (bit) {
      if (!reader.get_bits(4U, v)) {
        return fail(error, "seq_initial_display_delay_minus_1");
      }
    }
    if (!reader.get_bit(bit)) return fail(error, "decoder_model_info_present_flag");
    if (bit) {
      return fail(error,
                  "decoder model info is not supported by this reader");
    }
  }

  if (dialect == SequenceSyntaxDialect::kSpecV1_0_0) {
    if (!read_spec_layer_blocks(reader, out, error)) return false;
  } else {
    if (!read_compat_layer_blocks(reader, out, error)) return false;
  }
  out.presence = derive_mlayer_presence_map(out.dependencies.mlayer,
                                            out.max_mlayer_id);

  // Validate the effective map. The compat dialect cannot express every
  // dependency (dav2d's rows stop before max_mlayer_id), so only the spec
  // dialect is held to the full §5.4.1 invariants.
  if (dialect == SequenceSyntaxDialect::kSpecV1_0_0) {
    std::string map_error;
    if (!validate_dependency_maps(out.dependencies, out.max_mlayer_id,
                                  out.max_tlayer_id, map_error)) {
      return fail(error, "parsed dependency maps: " + map_error);
    }
    if (out.max_mlayer_id > 0 &&
        (out.seq_max_mlayer_cnt < 1 ||
         out.seq_max_mlayer_cnt > out.max_mlayer_id + 1)) {
      return fail(error, "parsed SeqMaxMlayerCnt is out of range");
    }
  }

  if (!read_baseline_tail(reader, out, error)) return false;

  // §5.2.1 extensible-OBU tail: obu_extension_flag(0) + trailing_bits().
  bool extension = false;
  if (!reader.get_bit(extension)) {
    return fail(error, "missing obu_extension_flag");
  }
  if (extension) {
    return fail(error,
                "obu_extension_flag must be 0 (§6.2.1 conformance)");
  }
  bool trailing_one = false;
  if (!reader.get_bit(trailing_one)) return fail(error, "missing trailing_one_bit");
  if (!trailing_one) return fail(error, "trailing_one_bit must be 1");
  while (reader.bits_left() > 0) {
    bool zero = false;
    if (!reader.get_bit(zero)) return fail(error, "trailing_zero_bit");
    if (zero) return fail(error, "trailing_zero_bit must be 0");
  }
  return true;
}

bool write_key_frame_header(const SequenceHeaderConfig& config,
                            const KeyFrameHeader& header,
                            std::vector<std::uint8_t>& header_bits,
                            std::size_t& bit_count, std::string& error) {
  if (!validate_header_config(config, error)) return false;
  if (!header.immediate_output_frame &&
      !config.monotonic_output_order_flag) {
    return fail(error,
                "non-immediate output needs implicit_output_frame, which this "
                "gate helper does not write");
  }
  BitWriter writer;
  put_uvlc(writer, 0);  // cur_mfh_id
  put_uvlc(writer, 0);  // seq_header_id_in_frame_header
  // OBU_CLOSED_LOOP_KEY: keyFrame, no frame_is_inter bit.
  // long_term_id_plus_1 absent (long_term_frame_id_bits == 0).
  writer.put_bit(header.immediate_output_frame);
  // implicit_output_frame absent when immediate_output_frame == 1 or
  // monotonic_output_order_flag == 1.
  writer.put_bit(false);  // frame_size_override_flag
  writer.put_bits(header.order_hint, config.order_hint_bits);
  // FrameIsIntra => primary_ref_frame = PRIMARY_REF_NONE, no bits.
  // §5.18.2 p99: CLK + max_mlayer_id > 0 codes refresh_frame_flags.
  if (config.layers.max_mlayer_id > 0) {
    const std::uint32_t mask = (1U << config.num_ref_frames) - 1U;
    if ((header.refresh_frame_flags & ~mask) != 0U) {
      return fail(error, "refresh_frame_flags exceeds NumRefFrames");
    }
    writer.put_bits(header.refresh_frame_flags, config.num_ref_frames);
  } else if (header.refresh_frame_flags !=
             ((1U << config.num_ref_frames) - 1U)) {
    return fail(error,
                "CLK with max_mlayer_id == 0 infers refresh_frame_flags = "
                "allFrames");
  }
  writer.put_bit(false);  // allow_intrabc
  writer.put_bit(config.disable_cdf_update);
  write_tile_info(writer, config);
  writer.put_bits(config.base_q_idx, 8U);
  writer.put_bit(false);  // segmentation_enabled
  writer.put_bit(false);  // using_qmatrix
  writer.put_bit(false);  // delta_q_present
  writer.put_bit(false);  // apply_deblocking_filter[0]
  writer.put_bit(false);  // apply_deblocking_filter[1]
  writer.put_bit(false);  // tx_mode_select = 0
  writer.put_bits(0U, 2U);  // reduced_tx_set
  bit_count = writer.bit_size();
  header_bits = writer.take_bytes();
  return true;
}

}  // namespace av2enc::inter::mlayers
