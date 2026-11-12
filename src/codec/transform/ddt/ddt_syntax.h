// DDT bitstream syntax.
//
// §5.4.8 sequence_transform_quant_entropy_config() (AV2 v1.0.0 PDF p. 63):
//
//     if ( Monochrome ) {
//         enable_chroma_dctonly = 0
//     } else {
//         enable_chroma_dctonly f(1)
//     }
//     if ( !single_picture_header_flag ) {
//         enable_inter_ddt f(1)
//     }
//     reduced_tx_part_set f(1)
//
// §6.4.8 (p. 269): "enable_inter_ddt equal to 1 specifies that DDT is allowed
// for inter blocks in the coded video sequence. enable_inter_ddt equal to 0
// specifies that DDT is not allowed for inter blocks."
//
// This is the *entire* DDT syntax.  There is no frame-header flag, no per-block
// flag and no CDF.  The per-block transform type is still signalled by the
// ordinary `inter_tx_type` / `inter_tx_type_offset` symbols (§5.20.8.2); DDTX
// and FDDT are derived from that type by §7.15.4.1.
//
// Single-picture streams (M1, `single_picture_header_flag = 1`) omit the bit.
// They cannot contain inter blocks, so DDT is unreachable there; the value is
// effectively 0.  (The spec does not spell out an inferred value for the
// omitted bit, but a still picture has no inter blocks by construction.)
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "core/bitstream/bitwriter.h"

namespace av2enc::video::ddt {

// Number of bits the DDT syntax element contributes to the sequence header.
// Exactly one f(1), or zero for a single-picture header.
inline constexpr std::size_t kEnableInterDdtBits = 1;
inline constexpr std::size_t kEnableInterDdtBitsStillPicture = 0;

// Pinned goldens for the one-bit element (MSB-first, the BitWriter order).
inline constexpr char kGoldenEnableInterDdtOff[] = "0";
inline constexpr char kGoldenEnableInterDdtOn[] = "1";

// Writes `enable_inter_ddt f(1)` iff the sequence header is not a
// single-picture header.  Returns the number of bits written (0 or 1).
std::size_t write_enable_inter_ddt(BitWriter& writer,
                                   bool single_picture_header_flag,
                                   bool enable_inter_ddt);

// Reads the same element from `data` at `bit_offset` (MSB-first).  On success
// `*enable_inter_ddt` is set (false when the bit is absent) and `*next_bit`
// advances by 0 or 1.  Returns false only when the bit is present and out of
// range.
bool read_enable_inter_ddt(std::span<const std::uint8_t> data,
                           std::size_t bit_offset,
                           bool single_picture_header_flag,
                           bool* enable_inter_ddt, std::size_t* next_bit);

// §5.4.8 byte cost of the element. `sequence_syntax_bits_without_ddt` is the
// total sequence_header_obu() syntax bit count with the element omitted
// (excluding the §5.2.1 extension flag and trailing bits).  Returns the
// number of complete payload bytes added by inserting the bit, i.e. 0 or 1:
// the syntax cost is always one bit, but the byte-aligned OBU payload grows
// only when the total crosses a byte boundary.
std::size_t payload_bytes_added(std::size_t sequence_syntax_bits_without_ddt,
                                bool single_picture_header_flag);

}  // namespace av2enc::video::ddt
