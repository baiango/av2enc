// AV2 v1.0.0 screen-content signalling: §5.4.7, §5.18.3.3, §5.18.3.4.
//
// This module is the single source of truth for the *header* side of the
// screen-content tools.  It owns:
//
//   §5.4.7    sequence_scc_config()            (sequence header)
//   §5.18.3.3 screen_content_params()          (frame header)
//   §5.18.3.4 intrabc_params()                 (frame header)
//   §6.4.7 / §6.17.3.2 / §6.17.3.3            (derived semantics)
//
// plus the block legality rules those flags gate:
//
//   §5.20.5.3  use_intrabc availability (allow_intrabc, size, chroma part,
//              region type)
//   §5.20.5.5  palette_mode_info availability (allow_screen_content_tools,
//              MiSize >= BLOCK_8X8, Block_Width/Height <= 64, luma only)
//   §5.20.5.4  intrabc_precision (NEWMV && !force_integer_mv)
//   §5.18.2    FrameMvPrecision = MV_PRECISION_ONE_PEL when force_integer_mv
//
// It deliberately does *not* own palette tokens, IBC MVD coding, or the rest
// of the frame header.  The existing palette/IBC stream modules keep their own
// tile coders; this module supplies/validates the bits they depend on and is
// exercised through documented calls (see README.md and scc_flags_bridge.*).
//
// Spec source: spec/AV2_Spec_v1.0.0.pdf (commit 38f28e7).
//   §5.4.7    printed p63-64
//   §6.4.7    p270
//   §5.18.3.3 p113-114
//   §5.18.3.4 p114
//   §6.17.3.2 / §6.17.3.3 p335
//   §5.20.5.3 p171
//   §5.20.5.4 p172

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"

namespace av2enc::m1 {

struct M1Config;  // encoder/m1/m1_headers.h

namespace scc_flags {

// §9.2 Table values used by the derived rules.
inline constexpr int kSelectScreenContentTools = 2;  // SELECT_SCREEN_CONTENT_TOOLS
inline constexpr int kSelectIntegerMv = 2;           // SELECT_INTEGER_MV
inline constexpr int kMvPrecisionOnePel = 3;         // MV_PRECISION_ONE_PEL
inline constexpr int kMvPrecisionQuarterPel = 5;     // MV_PRECISION_QUARTER_PEL

// BLOCK_8X8 in partition::BlockSize numeric order.  §5.20.5.5 compares MiSize
// (the enum value), not the block area.
inline constexpr int kBlock8x8 = 3;
inline constexpr int kBlock64x64 = 12;
inline constexpr int kBlockSizeCount = 29;  // enum_block_size range (0..28)
inline constexpr int kMaxScreenContentBlockDim = 64;

// ---------------------------------------------------------------------------
// §5.4.7 sequence_scc_config()
// ---------------------------------------------------------------------------

// Exact syntax selector.  Only the fields on the branch selected by the spec
// are consulted; `validate_sequence_scc_config()` rejects a config that asks
// for a field the branch does not code.
struct SequenceSccConfig {
  // §5.4.1 single_picture_header_flag.  When true §5.4.7 infers both
  // seq_force_* values and codes zero bits.
  bool single_picture_header_flag = true;

  // Non-single-picture branch only.
  bool seq_choose_screen_content_tools = false;
  // Present when seq_choose_screen_content_tools == 0.
  bool seq_force_screen_content_tools = false;

  // Present only when the resulting seq_force_screen_content_tools > 0.
  bool seq_choose_integer_mv = false;
  // Present when seq_choose_integer_mv == 0.
  bool seq_force_integer_mv = false;
};

// Decoded §5.4.7 state, including which bits were present.
struct SequenceSccState {
  bool single_picture_header_flag = true;
  int seq_force_screen_content_tools = kSelectScreenContentTools;  // 0/1/2
  int seq_force_integer_mv = kSelectIntegerMv;                     // 0/1/2

  bool coded_seq_choose_screen_content_tools = false;
  bool coded_seq_force_screen_content_tools = false;
  bool coded_seq_choose_integer_mv = false;
  bool coded_seq_force_integer_mv = false;
};

// Rejects configs whose fields are not present on the selected branch.
bool validate_sequence_scc_config(const SequenceSccConfig& config,
                                  std::string& error);

// Pure function: config -> inferred/coded state.  Callers may use this without
// touching a bitstream (e.g. to assert what a header must decode to).
SequenceSccState derive_sequence_scc_state(const SequenceSccConfig& config);

// Writes §5.4.7.  Returns false (with `error`) on an invalid config.  On
// success `*bits` is the number of bits written (0 for single picture) and
// `*state` the decoded state.
bool write_sequence_scc_config(BitWriter& writer,
                               const SequenceSccConfig& config,
                               SequenceSccState* state, std::size_t* bits,
                               std::string& error);

// Reads §5.4.7.  The caller must already have parsed
// single_picture_header_flag (§5.4.1) and positioned the reader at the first
// §5.4.7 bit (which is absent for a single-picture header).
bool read_sequence_scc_config(class BitReader& reader,
                              bool single_picture_header_flag,
                              SequenceSccState& state, std::string& error);

// ---------------------------------------------------------------------------
// §5.18.3.3 screen_content_params() + §5.18.3.4 intrabc_params()
// ---------------------------------------------------------------------------

// Whether screen_content_params()/intrabc_params() are parsed for this frame.
//
// §5.18.2 parses both unconditionally inside the FrameIsIntra branch.  For an
// inter frame they are parsed only when
//   TipFrameMode != TIP_FRAME_AS_OUTPUT && !bru_inactive && !IsBridge.
struct FrameSccContext {
  bool frame_is_intra = true;
  bool tip_frame_as_output = false;
  bool bru_inactive = false;
  bool is_bridge = false;
};

inline bool frame_scc_present(const FrameSccContext& context) noexcept {
  return context.frame_is_intra ||
         (!context.tip_frame_as_output && !context.bru_inactive &&
          !context.is_bridge);
}

// Exact frame syntax selector.  Fields not present on the branch are rejected
// by validate_frame_scc_config().
struct FrameSccConfig {
  // screen_content_params()
  bool allow_screen_content_tools = false;
  bool force_integer_mv = false;

  // intrabc_params()
  bool allow_intrabc = false;
  bool allow_global_intrabc = false;
  bool allow_local_intrabc = false;
  // Only present when allow_intrabc && allow_frame_max_bvp_drl_bits.
  bool change_bvp_drl = false;
  // Derived value to code when change_bvp_drl is set.  When change is not set
  // this must equal seq_max_bvp_drl_bits_minus_1.
  int max_bvp_drl_bits_minus_1 = 0;
};

// Decoded §5.18.3.3/.4 state plus the §5.18.2/§5.20.5 derived rules.
struct FrameSccState {
  bool scc_present = false;

  bool allow_screen_content_tools = false;
  bool force_integer_mv = false;
  bool allow_intrabc = false;
  bool allow_global_intrabc = false;
  bool allow_local_intrabc = false;
  int max_bvp_drl_bits_minus_1 = 0;

  bool coded_allow_screen_content_tools = false;
  bool coded_force_integer_mv = false;
  bool coded_allow_intrabc = false;
  bool coded_allow_global_intrabc = false;
  bool coded_allow_local_intrabc = false;
  bool coded_change_bvp_drl = false;

  // Derived (§6.17.3.3, §5.20.5.3): intra block copy is usable in this frame.
  bool IsIntraBC = false;

  // Derived (§5.18.2 p104-105): for an inter frame that is not
  // force_integer_mv, use_qtr_precision_mv/allow_high_precision_mv are coded
  // after intrabc_params().  force_integer_mv == 1 removes those bits and sets
  // FrameMvPrecision = MV_PRECISION_ONE_PEL.
  bool codes_frame_mv_precision = false;
  // Meaningful only when !codes_frame_mv_precision.  A non-forced inter frame
  // has the real value in the following frame-header syntax, so this stays 0.
  int frame_mv_precision = kMvPrecisionOnePel;

  // Derived (§5.20.5.4): base MvPrecision used by read_intrabc_info().
  int intrabc_base_mv_precision = kMvPrecisionOnePel;
  // intrabc_precision is coded for a NEWMV IBC block only when
  // !force_integer_mv.
  bool intrabc_precision_coded = false;
};

// `seq_max_bvp_drl_bits_minus_1` and `allow_frame_max_bvp_drl_bits` come from
// §5.4.6 sequence_inter_config().  M1 writes 0 / false (one DRL bit).
bool validate_frame_scc_config(const SequenceSccState& sequence,
                               const FrameSccContext& context,
                               const FrameSccConfig& config,
                               int seq_max_bvp_drl_bits_minus_1,
                               bool allow_frame_max_bvp_drl_bits,
                               std::string& error);

FrameSccState derive_frame_scc_state(const SequenceSccState& sequence,
                                     const FrameSccContext& context,
                                     const FrameSccConfig& config,
                                     int seq_max_bvp_drl_bits_minus_1,
                                     bool allow_frame_max_bvp_drl_bits);

// Writes screen_content_params() followed by intrabc_params().
bool write_frame_scc_params(BitWriter& writer,
                            const SequenceSccState& sequence,
                            const FrameSccContext& context,
                            const FrameSccConfig& config,
                            int seq_max_bvp_drl_bits_minus_1,
                            bool allow_frame_max_bvp_drl_bits,
                            FrameSccState* state, std::size_t* bits,
                            std::string& error);

bool read_frame_scc_params(class BitReader& reader,
                           const SequenceSccState& sequence,
                           const FrameSccContext& context,
                           int seq_max_bvp_drl_bits_minus_1,
                           bool allow_frame_max_bvp_drl_bits,
                           FrameSccState& state, std::string& error);

// ---------------------------------------------------------------------------
// Block legality (§5.20.5.3 / §5.20.5.4 / §5.20.5.5)
// ---------------------------------------------------------------------------

// §5.20.5.5: palette_mode_info() is parsed only for
//   PlaneStart == 0 (luma), YMode == DC_PRED,
//   MiSize >= BLOCK_8X8, Block_Width <= 64, Block_Height <= 64,
//   allow_screen_content_tools == 1.
bool palette_allowed_for_block(const FrameSccState& state, int plane,
                               int y_mode, int bsize, int block_w,
                               int block_h) noexcept;

// §5.20.5.3: intra frames code use_intrabc only when
//   allow_intrabc && TreeType != CHROMA_PART &&
//   Block_Width <= 64 && Block_Height <= 64 && MiSize != BLOCK_64X64.
// Inter frames additionally require RegionType == MIXED_REGION
// (§5.20.7.3).
bool intrabc_allowed_for_block(const FrameSccState& state, int bsize,
                               int block_w, int block_h,
                               bool tree_is_chroma_part,
                               bool region_is_mixed = true,
                               bool frame_is_intra = true) noexcept;

// §5.20.5.4: morph_pred is signalled only for
//   FrameIsIntra && allow_screen_content_tools && enable_bawp.
bool morph_pred_allowed(const FrameSccState& state, bool frame_is_intra,
                        bool enable_bawp) noexcept;

// IBC motion-vector precision for a block (§5.20.5.4): integer when
// force_integer_mv, otherwise quarter-pel until intrabc_precision says
// otherwise.
int intrabc_mv_precision(const FrameSccState& state) noexcept;

// ---------------------------------------------------------------------------
// Bit reader (independent of the writer; MSB-first, same convention as
// av2enc::BitWriter)
// ---------------------------------------------------------------------------

class BitReader {
 public:
  BitReader(std::span<const std::uint8_t> bytes) noexcept
      : bytes_(bytes) {}
  BitReader(const std::uint8_t* data, std::size_t byte_count) noexcept
      : bytes_(data, byte_count) {}
  BitReader(const std::vector<std::uint8_t>& bytes) noexcept
      : bytes_(bytes) {}

  bool get_bit(bool& out) noexcept;
  bool get_bits(unsigned count, std::uint32_t& out) noexcept;
  // uvlc() (§4.11.3).
  bool get_uvlc(std::uint32_t& out) noexcept;
  // ns(n) (§4.11.8).
  bool get_ns(std::uint32_t n, std::uint32_t& out) noexcept;

  std::size_t bit_pos() const noexcept { return bit_pos_; }
  std::size_t byte_size() const noexcept { return bytes_.size(); }
  bool byte_aligned() const noexcept { return (bit_pos_ & 7U) == 0U; }
  bool skip_bits(std::size_t count) noexcept;
  bool ok() const noexcept { return ok_; }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_pos_ = 0;
  bool ok_ = true;
};

// ---------------------------------------------------------------------------
// Complete still-picture assembly (M1 subset)
// ---------------------------------------------------------------------------
//
// Writes TD + sequence header + CLK with the §5.4.7/§5.18.3.3/.4 bits owned by
// this module.  The part of the M1 frame header after intrabc_params() is
// spliced verbatim from m1::write_frame_header(), so the default (all-off)
// header is byte-identical to the production M1 writer.
//
// Requires single_picture_header_flag == 1: the M1 sequence-header writer
// hardcodes it.  A non-single sequence header belongs to the M2 writer; use
// write_sequence_scc_config() there.

struct StillSccConfig {
  SequenceSccConfig sequence;
  FrameSccConfig frame;
};

struct StillSccResult {
  std::vector<std::uint8_t> frame;  // TD + SEQ + CLK
  SequenceSccState sequence_state;
  FrameSccState frame_state;
  std::size_t frame_header_bits = 0;   // is_first_tile_group + frame header
  std::size_t frame_header_bytes = 0;  // after byte_alignment()
  std::size_t tile_bytes = 0;
};

bool assemble_still_picture(const M1Config& config, const StillSccConfig& scc,
                            int seq_max_bvp_drl_bits_minus_1,
                            bool allow_frame_max_bvp_drl_bits,
                            std::span<const std::uint8_t> tile_payload,
                            StillSccResult& out, std::string& error);

// Bit size of the still frame header (is_first_tile_group + frame_header()),
// before and after byte_alignment().  Useful for callers that need to locate
// the tile payload inside an existing still.
bool still_frame_header_size(const M1Config& config,
                             const StillSccConfig& scc,
                             int seq_max_bvp_drl_bits_minus_1,
                             bool allow_frame_max_bvp_drl_bits,
                             std::size_t* bits, std::size_t* bytes,
                             FrameSccState* state, std::string& error);

// Parses §5.18.3.3/.4 out of a TD + SEQ + CLK byte stream produced by any
// M1-compatible still writer (this module, palette_stream, ibc_stream).  The
// caller supplies the §5.4.7 sequence state (single-picture M1 infers both
// SELECT values; use read_sequence_scc_config() for a non-single header).  The
// CLK tile payload is not touched.
bool read_still_frame_scc(std::span<const std::uint8_t> frame,
                          const SequenceSccState& sequence,
                          const FrameSccContext& context,
                          int seq_max_bvp_drl_bits_minus_1,
                          bool allow_frame_max_bvp_drl_bits,
                          FrameSccState& state, std::string& error);

}  // namespace scc_flags
}  // namespace av2enc::m1
