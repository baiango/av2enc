#ifndef AV2ENC_ENCODER_INTER_GME_GME_SYNTAX_H_
#define AV2ENC_ENCODER_INTER_GME_GME_SYNTAX_H_

// AV2 v1.0.0 §5.18.9 global motion syntax writer/reader.
//
// `global_motion_params()` is signalled at the end of `frame_header_info()`:
//
//   ... frame_reference_mode(); skip_mode_params();
//   if (!FrameIsIntra && enable_bawp) allow_bawp f(1)
//   if (!FrameIsIntra && frame_enabled_motion_modes[DELTAWARP])
//       allow_warpmv_mode f(1)
//   reduced_tx_set f(2)
//   global_motion_params()
//   film_grain_config()
//
// The sequence bit `enable_global_motion` sits in
// `sequence_inter_config()` after `enable_flex_mvres` and before
// `enable_short_refresh_frame_flags` (only when
// `single_picture_header_flag == 0`).
//
// The writer is self-contained: it does not depend on the M2 tile writer and
// does not change any existing header. Callers insert the returned bits at the
// exact positions above.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "core/bitstream/bitwriter.h"
#include "gme.h"

namespace av2enc::inter::gme {

// ---------------------------------------------------------------------------
// §5.4.6 sequence_inter_config() tail bit. Exact insertion point:
//   enable_flex_mvres f(1)
//   enable_global_motion f(1)      <-- this helper
//   enable_short_refresh_frame_flags f(1)
// ---------------------------------------------------------------------------
void write_sequence_enable_global_motion(BitWriter& writer, bool enable);

// Tiny MSB-first reader used by the round-trip tests. The production decoder
// owns its own bit reader; this one exists so the syntax writer can be tested
// in isolation.
class GmBitReader {
 public:
  GmBitReader() = default;
  explicit GmBitReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  void reset(std::span<const std::uint8_t> bytes) {
    bytes_ = bytes;
    bit_pos_ = 0;
  }

  bool read_bit(bool* bit);
  bool read_bits(unsigned num_bits, std::uint32_t* value);
  bool read_ns(std::uint32_t n, std::uint32_t* value);

  std::size_t bit_pos() const { return bit_pos_; }
  std::size_t bits_remaining() const { return bytes_.size() * 8U - bit_pos_; }

 private:
  std::span<const std::uint8_t> bytes_{};
  std::size_t bit_pos_ = 0;
};

// Per-reference saved global motion (§5.18.9.1 SavedGmParams /
// SavedOrderHints / RefNumTotalRefs). Indexed by reference *slot*.
struct GmRefHistory {
  std::array<int, kNumRefFrames> ref_num_total_refs{};
  std::array<std::array<int, kMaxRefsPerFrame>, kNumRefFrames> saved_order_hints{};
  std::array<std::array<std::array<std::int32_t, 6>, kMaxRefsPerFrame>, kNumRefFrames>
      saved_gm_params{};
};

// Inputs shared by the writer and reader. For the writer, `params` holds the
// target model for each reference list entry. For the reader, `params` is
// ignored and results are written to the output array.
struct GmSyntaxContext {
  bool frame_is_intra = false;
  bool enable_global_motion = false;
  bool switch_frame = false;
  int num_total_refs = 0;
  // Current frame order hint and OrderHints[ref] for each ref list entry.
  int order_hint = 0;
  std::array<int, kMaxRefsPerFrame> ref_order_hint{};
  // §5.18.2 ref_frame_idx[] (slots).
  std::array<std::uint8_t, kMaxRefsPerFrame> ref_frame_idx{};
  // Base model selection (§5.18.9.1). `our_ref < 0` means NumTotalRefs (no
  // base), which is the common encoder choice.
  int our_ref = -1;
  int their_ref = 0;
  GmRefHistory history{};
  // Writer target. Must be identity for refs with dist == 0 or RESTRICTED_OH.
  std::array<GmParams, kMaxRefsPerFrame> params{};
  // Writer: whether to emit use_global_motion = 1. The reader returns the
  // decoded value through `use_global_motion_out`.
  bool use_global_motion = false;
  bool use_global_motion_out = false;
};

// §5.18.9.1 global_motion_params(). Returns the number of bits written. On
// failure `error` is set and the writer may contain partial bits; callers
// validate before committing a header.
std::size_t write_global_motion_params(BitWriter& writer, const GmSyntaxContext& context,
                                       std::string* error);

// Reads global_motion_params() and fills `params_out[0..num_total_refs-1]`.
bool read_global_motion_params(GmBitReader& reader, const GmSyntaxContext& context,
                               std::array<GmParams, kMaxRefsPerFrame>* params_out,
                               std::string* error, bool* use_global_motion_out = nullptr);

// §5.18.9.1 scale_warp_model. Writes six Q16 parameters to `out`.
void scale_warp_model(const std::int32_t base[6], int base_distance, int distance,
                      std::int32_t out[6]);

// Exposed for tests: one read/write_global_param() pair.
bool write_global_param(BitWriter& writer, int ref, int index, const std::int32_t prev[6],
                        const GmParams& params, std::string* error);
bool read_global_param(GmBitReader& reader, int ref, int index, const std::int32_t prev[6],
                       std::int32_t out[6], std::string* error);

}  // namespace av2enc::inter::gme

#endif  // AV2ENC_ENCODER_INTER_GME_GME_SYNTAX_H_
