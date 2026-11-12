// Pinned golden bytes for the §5.13 / §5.18.6.2 quantizer-matrix syntax.
//
// These arrays are deliberately hand-checkable and are asserted byte-for-byte
// by qm_obu_test.cpp.  They were derived from the syntax rules (not captured
// from the encoder) and cross-checked against the bit strings below:
//
//   reset, chroma present:  bit_map f(15)=0 | chroma=1 | trailing 1
//                           -> 00 01 | 1....... = 00 01 80
//   level 0, default:       bit_map=1 (14 zeros,1) | chroma=1 | default=1
//                           | trailing 1 -> 00 03 | 11...... = 00 03 c0
//   level 5, default mono:  bit_map=1<<5 | chroma=0 | default=1 | trailing 1
//                           -> 00 40 | 11...... = 00 40 c0
//
// All full-OBU arrays include the Annex B.2 leb128 size and the 0x58
// obu_header byte (obu_header_extension_flag=0, obu_type=22, obu_tlayer_id=0).
#pragma once

#include <cstdint>

namespace av2enc::qm::obu::goldens {

// quantizer_matrix_obu: qm_bit_map = 0, qm_chroma_info_present_flag = 1.
inline constexpr std::uint8_t kResetChroma[] = {
    0x04, 0x58, 0x00, 0x01, 0x80,
};

// quantizer_matrix_obu: qm_bit_map = 0, qm_chroma_info_present_flag = 0.
inline constexpr std::uint8_t kResetMono[] = {
    0x04, 0x58, 0x00, 0x00, 0x80,
};

// level 0, qm_is_default_flag = 1, chroma present.
inline constexpr std::uint8_t kLevel0DefaultChroma[] = {
    0x04, 0x58, 0x00, 0x03, 0xc0,
};

// level 5, qm_is_default_flag = 1, mono.
inline constexpr std::uint8_t kLevel5DefaultMono[] = {
    0x04, 0x58, 0x00, 0x40, 0xc0,
};

// level 5, user-defined constant 32 (identity), chroma present.
inline constexpr std::uint8_t kLevel5User32Chroma[] = {
    0x0d, 0x58, 0x00, 0x41, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xfc,
};

// levels 0 and 14, both default, chroma present.
inline constexpr std::uint8_t kLevels0And14Default[] = {
    0x04, 0x58, 0x80, 0x03, 0xe0,
};

// level 3, user-defined constant 32, mono.
inline constexpr std::uint8_t kLevel3User32Mono[] = {
    0x0c, 0x58, 0x00, 0x10, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff,
};

// level 2, default, no chroma info.
inline constexpr std::uint8_t kLevel2DefaultMonoNoChroma[] = {
    0x04, 0x58, 0x00, 0x08, 0xc0,
};

// level 7, default, obu_mlayer_id = 2, obu_tlayer_id = 1.
inline constexpr std::uint8_t kLevel7DefaultMlayer2Tlayer1[] = {
    0x05, 0xd9, 0x40, 0x01, 0x01, 0xc0,
};

// §5.18.6.2 frame-header bit strings (MSB first), pinned after the leading
// using_qmatrix bit.  "frame_off" is a single zero bit.
inline constexpr char kFrameOffBits[] = "0";
inline constexpr char kFrameY5UvSameBits[] = "101011";
inline constexpr char kFrameY3U7Bits[] = "1001100111";
inline constexpr char kFrameSeg2SeparateBits[] =
    "101001111001001000100";
inline constexpr char kFrameMonoY14Bits[] = "11110";

}  // namespace av2enc::qm::obu::goldens
