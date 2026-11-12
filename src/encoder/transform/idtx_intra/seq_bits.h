// §5.4.8 sequence_transform_quant_entropy_config() subset writer/reader.
//
// This exists to pin the `enable_idtx_intra` conditional:
//
//   enable_fsc f(1)
//   if ( enable_fsc ) { enable_idtx_intra = 1 }
//   else              { enable_idtx_intra f(1) }
//
// The rest of the function (IST, chroma-DCT-only, DDT, reduced_tx_part_set,
// CCTX, TCQ, parity hiding, average CDFs, delta-Q) is written too so the bit
// offsets and the fsc-dependent length are testable end to end.  `seg_info()`
// that follows the function is not part of it and is not written here.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace av2enc::idtx {

struct TransformSeqConfig {
  bool enable_fsc = false;
  bool enable_idtx_intra = false;  // raw bit; ignored when enable_fsc
  bool enable_intra_ist = false;
  bool enable_inter_ist = false;
  bool enable_chroma_dctonly = true;
  bool monochrome = false;
  bool single_picture_header = true;
  bool enable_inter_ddt = false;
  bool reduced_tx_part_set = false;
  bool enable_cctx = false;
  bool enable_tcq = false;
  bool choose_tcq_per_frame = false;
  bool enable_parity_hiding = false;
  bool enable_avg_cdf = false;
  bool avg_cdf_type = false;
  bool separate_uv_delta_q = false;
  bool equal_ac_dc_q = true;
  int base_y_dc_delta_q = 0;      // f(5)
  bool y_dc_delta_q_enabled = false;
  int base_uv_dc_delta_q = 0;     // f(5)
  bool uv_dc_delta_q_enabled = false;
  int base_uv_ac_delta_q = 23;    // f(5), M1 writes 23 -> BaseUVAcDeltaQ = 0
  bool uv_ac_delta_q_enabled = false;
};

struct TransformSeqBits {
  bool enable_idtx_intra = false;  // effective value
  bool idtx_bit_coded = true;      // false when enable_fsc == 1
  std::size_t bit_size = 0;        // bits produced / consumed
};

// MSB-first, no trailing-bit padding (the caller owns byte alignment).
std::vector<std::uint8_t> write_transform_quant_entropy_config(
    const TransformSeqConfig& config, TransformSeqBits* info = nullptr);

// Reads from `bit_offset` in `data`.  Returns false on truncation or an
// out-of-range field.  `bits_consumed` receives the field bits.
bool read_transform_quant_entropy_config(const std::uint8_t* data,
                                         std::size_t byte_count,
                                         std::size_t bit_offset,
                                         TransformSeqConfig* config,
                                         std::size_t* bits_consumed,
                                         TransformSeqBits* info = nullptr);

// Effective `enable_idtx_intra` and whether the bit is present.
TransformSeqBits transform_seq_flags(const TransformSeqConfig& config);

}  // namespace av2enc::idtx
