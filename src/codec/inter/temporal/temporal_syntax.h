// AV2 v1.0.0 temporal inter tools — bit-level syntax.
//
// These helpers cover only the temporal slice of the sequence header, the TIP
// frame-header fields, the TIP block selector, and `read_refinemv()`.  The M2
// header writer owns the surrounding fields; README.md lists the exact call
// sites and bit order.  The round-trip tests in temporal_test.cpp pin every
// conditional path.
//
// Syntax references:
//   §5.4.6   sequence_inter_config()
//   §5.18.2  frame_header_info() (use_ref_frame_mvs / ProjStep / TIP block)
//   §5.18.3.2 frame_opfl_refine_type()
//   §5.20.7.10 read_ref_frames() (tip_mode)
//   §5.20.7.6 inter_block_mode_info() (tip_pred_mode)
//   §5.20.7.17 read_refinemv()

#ifndef AV2ENC_ENCODER_INTER_TEMPORAL_TEMPORAL_SYNTAX_H_
#define AV2ENC_ENCODER_INTER_TEMPORAL_TEMPORAL_SYNTAX_H_

#include <cstddef>
#include <cstdint>
#include <span>

#include "core/bitstream/bitwriter.h"
#include "codec/inter/temporal/temporal_types.h"

namespace av2enc::inter::temporal {

// Minimal MSB-first bit reader.  No range coder: these are raw f(n) bits.
class BitReader {
 public:
  BitReader(const std::uint8_t* data, std::size_t size)
      : data_(data), size_(size) {}

  bool get_bit(bool* bit);
  bool get_bits(unsigned num_bits, std::uint32_t* value);
  std::size_t position() const { return bit_pos_; }
  bool ok() const { return ok_; }

 private:
  const std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t bit_pos_ = 0;
  bool ok_ = true;
};

// §5.4.6 temporal slice, starting at enable_ref_frame_mvs and ending after
// enable_tip_refinemv.  The fields between them are written so the bit
// positions match the full sequence_inter_config().
TemporalStatus write_sequence_temporal_syntax(BitWriter& writer,
                                              const TemporalSequenceConfig& cfg);
TemporalStatus read_sequence_temporal_syntax(BitReader& reader,
                                             TemporalSequenceConfig* cfg);

// §5.18.2 fields: tmvp_sample_step_minus_1 (conditional), the TIP frame block,
// and frame_opfl_refine_type().  Other frame-header fields are outside this
// helper; call it at the positions documented in README.md.
TemporalStatus write_frame_temporal_syntax(BitWriter& writer,
                                           const TemporalFrameConfig& cfg);
TemporalStatus read_frame_temporal_syntax(BitReader& reader,
                                          TemporalFrameConfig* cfg);

// §5.18.3.2 alone (also used by the TIP-disabled frame path).
TemporalStatus write_opfl_refine_type(BitWriter& writer,
                                      const TemporalFrameConfig& cfg);
TemporalStatus read_opfl_refine_type(BitReader& reader,
                                     const TemporalFrameConfig& cfg,
                                     RefineType* out);

// §5.20.7.10 tip_mode and §5.20.7.6 tip_pred_mode.  The combined helper
// writes the two bits contiguously and is intended for round-trip tests; in a
// real tile they are read in read_ref_frames() and inter_block_mode_info()
// respectively.  Use the split helpers at integration sites.
TemporalStatus write_tip_mode_syntax(BitWriter& writer,
                                     const TipBlockSyntaxConfig& cfg);
TemporalStatus read_tip_mode_syntax(BitReader& reader,
                                    const TipBlockSyntaxConfig& cfg,
                                    TipBlockSyntaxConfig* out);
TemporalStatus write_tip_pred_mode_syntax(BitWriter& writer, int tip_pred_mode);
TemporalStatus read_tip_pred_mode_syntax(BitReader& reader, int* tip_pred_mode);
TemporalStatus write_tip_block_syntax(BitWriter& writer,
                                      const TipBlockSyntaxConfig& cfg);
TemporalStatus read_tip_block_syntax(BitReader& reader,
                                     const TipBlockSyntaxConfig& cfg,
                                     TipBlockSyntaxConfig* out);

// §5.20.7.17 read_refinemv().  The reader path consumes one bit only when the
// decision is switchable.
TemporalStatus write_refinemv_block_syntax(BitWriter& writer,
                                           const RefineMvBlockSyntaxConfig& cfg);
TemporalStatus read_refinemv_block_syntax(BitReader& reader,
                                          const RefineMvConfig& ctx,
                                          RefineMvBlockSyntaxConfig* out);

}  // namespace av2enc::inter::temporal

#endif  // AV2ENC_ENCODER_INTER_TEMPORAL_TEMPORAL_SYNTAX_H_
