// AV2 v1.0.0 frame-header reference machinery.
//
// Normative reference: spec/AV2_Spec_v1.0.0.pdf (commit 38f28e7).
//
//   §5.18.1  frame_header() / frame_header_copy() / header_bit[i]
//   §5.18.2  frame_header_info(): primary_ref_frame, disable_cross_frame_cdf_init,
//            refresh_frame_flags / has_refresh_frame_flags / frame_to_refresh,
//            max_drl_bits_minus_1 / change_drl
//   §5.18.3.4 intrabc_params(): change_bvp_drl / max_bvp_drl_bits_minus_1
//   §5.18.4.1 frame_size()
//   §5.18.4.3 frame_size_with_refs() / found_ref
//   §5.18.4.4 compute_image_size()
//   §6.17.1  frame_header_copy semantics (bit-for-bit, not byte aligned)
//   §6.17.2  Frame header info semantics
//   §6.17.3.3 Intra block copy params semantics
//   §6.17.4.3 Frame size with refs semantics
//   §6.17.4.4 Compute image size function semantics
//
// The module is deliberately independent of the M1/M2 core:
//   * every syntax element has a writer and a *separate* reader.  The reader
//     parses bits; it never consults writer-side state.  Any validation the
//     spec attaches to a value is enforced on both sides.
//   * availability/consistency checks are expressed against a live reference
//     state (RefValid / RefOrderHint / RefFrameWidth+Height / RefBaseQIdx) so
//     an encoder can prove a header is conforming before it emits it.
//
// This file mirrors §5.18 verbatim for the four mechanisms that the audit
// (docs/spec-audit/03-frame-header.md) lists as ABSENT:
//   * primary_ref_frame (§5.18.2 + §6.17.2 CDF-init consequence)
//   * max_drl_bits_minus_1 / change_drl and change_bvp_drl /
//     max_bvp_drl_bits_minus_1
//   * has_refresh_frame_flags / frame_to_refresh
//   * frame_size_with_refs() / found_ref
//   * frame_header_copy() / header_bit[i]
#pragma once

#include "bitstream/bitwriter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace av2enc::frame::ref_fields {

// §3 constants used by these syntax elements.
inline constexpr std::uint32_t kNumRefFrames = 16;  // NUM_REF_FRAMES
inline constexpr std::uint32_t kRefsPerFrame = 7;   // REFS_PER_FRAME
inline constexpr std::uint32_t kPrimaryRefNone = 7;    // PRIMARY_REF_NONE
inline constexpr std::uint32_t kPrimaryRefChoose = 8;  // PRIMARY_REF_CHOOSE
inline constexpr std::int32_t kRestrictedOrderHint = -1;  // RESTRICTED_OH
inline constexpr std::uint32_t kMaxRefMvStackSize = 6;    // MAX_REF_MV_STACK_SIZE
inline constexpr std::uint32_t kMaxRefBvStackSize = 4;    // MAX_REF_BV_STACK_SIZE
inline constexpr std::uint32_t kNoRef = 0xFFFFFFFFu;

// FrameType, §6.17.2 Table.
enum class FrameType : std::uint8_t {
  kKey = 0,        // KEY_FRAME
  kInter = 1,      // INTER_FRAME
  kIntraOnly = 2,  // INTRA_ONLY_FRAME
  kSwitch = 3,     // SWITCH_FRAME
};

const char* to_string(FrameType type);

unsigned ceil_log2(std::uint32_t value);
unsigned bit_width(std::uint32_t value);

// ---------------------------------------------------------------------------
// Independent bit reader (§4.11).  Every parse function in this module takes
// this reader; nothing on the read path calls a writer helper.
// ---------------------------------------------------------------------------
class RefBitReader {
 public:
  explicit RefBitReader(std::span<const std::uint8_t> bytes,
                        std::size_t bit_position = 0);

  bool bit(bool* out);
  bool bits(unsigned count, std::uint32_t* out);
  // §4.11.3 uvlc().
  bool uvlc(std::uint32_t* out);
  // §4.11.7 ns( n ).
  bool ns(std::uint32_t n, std::uint32_t* out);
  // §4.11.6 su( n ).
  bool su(unsigned count, std::int32_t* out);

  std::size_t position() const { return bit_position_; }
  std::size_t total_bits() const { return bytes_.size() * 8U; }
  std::size_t remaining_bits() const;
  bool seek(std::size_t bit_position);
  std::span<const std::uint8_t> bytes() const { return bytes_; }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t bit_position_ = 0;
};

// ---------------------------------------------------------------------------
// Live reference state (§7.23) that these syntax elements are validated
// against.  Only the fields the header machinery needs are modelled.
// ---------------------------------------------------------------------------
struct RefSlot {
  bool valid = false;           // RefValid[i]
  std::int32_t order_hint = 0;  // RefOrderHint[i]; RESTRICTED_OH == -1
  std::uint32_t width = 0;      // RefFrameWidth[i]
  std::uint32_t height = 0;     // RefFrameHeight[i]
  std::uint8_t base_q_idx = 0;  // RefBaseQIdx[i]
  FrameType frame_type = FrameType::kInter;  // RefFrameType[i]
};

// The frame's reference list (ref_frame_idx[]) and NumTotalRefs.
struct RefList {
  std::array<std::uint8_t, kRefsPerFrame> index{};
  std::uint32_t num_total_refs = 0;

  void clear() {
    index.fill(0);
    num_total_refs = 0;
  }
  bool push(std::uint8_t slot) {
    if (num_total_refs >= kRefsPerFrame) return false;
    index[num_total_refs++] = slot;
    return true;
  }
};

struct FrameRefState {
  std::array<RefSlot, kNumRefFrames> slot{};
  std::uint32_t num_ref_frames = 1;   // §5.4.6 NumRefFrames
  std::uint32_t max_frame_width = 0;  // max_frame_width_minus_1 + 1
  std::uint32_t max_frame_height = 0;
  std::uint32_t frame_width_bits = 1;   // frame_width_bits_minus_1 + 1
  std::uint32_t frame_height_bits = 1;  // frame_height_bits_minus_1 + 1
  bool single_picture_header_flag = false;
  bool monotonic_output_order_flag = true;
  // Needed by get_relative_dist() in the primary-reference ranking.
  std::uint32_t order_hint_bits = 1;  // OrderHintBits
  bool enable_order_hint = true;
  std::int32_t order_hint = 0;  // OrderHint of the frame being signalled
};

bool slot_constraints_ok(const FrameRefState& state, std::uint32_t slot,
                         std::string* error);

// ---------------------------------------------------------------------------
// primary_ref_frame / disable_cross_frame_cdf_init (§5.18.2, §6.17.2)
//
//   if ( FrameIsIntra || FrameType == SWITCH_FRAME ) {
//     primary_ref_frame = PRIMARY_REF_NONE
//   } else {
//     signal_primary_ref_frame f(1)
//     if ( !is_tip_frame() ) disable_cross_frame_cdf_init f(1)
//     if ( signal_primary_ref_frame ) primary_ref_frame f(3)
//     else primary_ref_frame = PRIMARY_REF_CHOOSE
//   }
//
// CDF-init consequence (§5.18.2 set_primary_ref_frame_and_ctx):
//   primary_ref_frame == PRIMARY_REF_NONE || disable_cross_frame_cdf_init
//     -> init_non_coeff_cdfs() + init_coeff_cdfs()   (defaults)
//   otherwise
//     -> load_cdfs( ref_frame_idx[ primary_ref_frame ] ) + load_previous()
//        (the whole frame's CDF state and previous-frame state come from the
//         reference slot)
//   The process *forces* disable_cross_frame_cdf_init = 1 and
//   primary_ref_frame = PRIMARY_REF_NONE when the derived primary reference is
//   PRIMARY_REF_NONE, so the effective value differs from the coded bit.
// ---------------------------------------------------------------------------
struct PrimaryRefSignal {
  bool signal_primary_ref_frame = false;
  std::uint32_t primary_ref_frame = kPrimaryRefChoose;  // f(3) when signalled
  bool disable_cross_frame_cdf_init = true;
};

struct PrimaryRefResolved {
  // Coded/effective values after the §5.18.2 forcing.
  std::uint32_t primary_ref_frame = kPrimaryRefNone;
  std::uint32_t derived_primary_ref_frame = kPrimaryRefNone;
  std::uint32_t secondary_ref_frame = kPrimaryRefNone;
  bool coded_disable_cross_frame_cdf_init = true;
  bool disable_cross_frame_cdf_init = true;  // effective

  // Consequence: where the frame's starting CDF state comes from.
  bool load_cdfs_from_reference = false;  // false => init_*_cdfs()
  std::uint8_t cdf_reference_slot = 0;    // ref_frame_idx[primary] when true
  bool load_previous_state = false;       // DerivedPrimaryRefFrame != NONE
};

// Writes/reads the primary-ref block.  `frame_type` decides whether any bit is
// present (intra/switch frames code nothing).  `is_tip` removes the
// disable_cross_frame_cdf_init bit (§5.18.2 skips it for TIP frames).
bool write_primary_ref_signal(BitWriter& writer, FrameType frame_type,
                              bool is_tip, const PrimaryRefSignal& signal,
                              std::string* error);
bool read_primary_ref_signal(RefBitReader& reader, FrameType frame_type,
                             bool is_tip, PrimaryRefSignal* signal,
                             std::string* error);

// Validates the signalled values and resolves the whole block, including the
// §5.18.2 derived primary reference (choose_primary_secondary_ref_frame) and
// the CDF-init consequence.
bool resolve_primary_ref(FrameType frame_type, const PrimaryRefSignal& signal,
                         const RefList& refs, const FrameRefState& state,
                         std::uint8_t base_q_idx, PrimaryRefResolved* out,
                         std::string* error);

// The derivation of §5.18.2 choose_primary_secondary_ref_frame(), reduced to
// the parts that decide the CDF source.  `base_q_idx` is the current frame's
// base_q_idx.  Exposed for tests.
void choose_primary_secondary_ref_frame(const RefList& refs,
                                        const FrameRefState& state,
                                        std::uint8_t base_q_idx,
                                        std::uint32_t* primary,
                                        std::uint32_t* secondary);

// ---------------------------------------------------------------------------
// max_drl_bits_minus_1 / change_drl (§5.18.2) and
// change_bvp_drl / max_bvp_drl_bits_minus_1 (§5.18.3.4)
//
//   max_drl_bits_minus_1 = seq_max_drl_bits_minus_1
//   if ( allow_frame_max_drl_bits ) {
//     change_drl f(1)
//     if ( change_drl ) {
//       max_drl_bits_minus_1 ns( MAX_REF_MV_STACK_SIZE - 2 )
//       if ( max_drl_bits_minus_1 >= seq_max_drl_bits_minus_1 )
//         max_drl_bits_minus_1 += 1
//     }
//   }
//
// and, inside intrabc_params(), only when allow_intrabc == 1:
//   max_bvp_drl_bits_minus_1 = seq_max_bvp_drl_bits_minus_1
//   if ( allow_frame_max_bvp_drl_bits ) {
//     change_bvp_drl f(1)
//     if ( change_bvp_drl ) {
//       max_bvp_drl_bits_minus_1 ns( 2 )
//       if ( >= seq_max_bvp_drl_bits_minus_1 ) += 1
//     }
//   }
// ---------------------------------------------------------------------------
struct DrlConfig {
  bool allow_frame_max_drl_bits = false;    // §5.4.6
  std::uint32_t seq_max_drl_bits_minus_1 = 0;
};

struct DrlSignal {
  bool change_drl = false;
  std::uint32_t max_drl_bits_minus_1 = 0;  // resolved value
};

struct BvDrlConfig {
  bool allow_frame_max_bvp_drl_bits = false;  // §5.4.6
  std::uint32_t seq_max_bvp_drl_bits_minus_1 = 0;
};

struct BvDrlSignal {
  bool change_bvp_drl = false;
  std::uint32_t max_bvp_drl_bits_minus_1 = 0;
};

bool write_drl(BitWriter& writer, const DrlConfig& config,
               const DrlSignal& signal, std::string* error);
bool read_drl(RefBitReader& reader, const DrlConfig& config, DrlSignal* signal,
              std::string* error);

bool write_bvp_drl(BitWriter& writer, const BvDrlConfig& config,
                   const BvDrlSignal& signal, std::string* error);
bool read_bvp_drl(RefBitReader& reader, const BvDrlConfig& config,
                  BvDrlSignal* signal, std::string* error);

// ---------------------------------------------------------------------------
// refresh_frame_flags / has_refresh_frame_flags / frame_to_refresh
// (§5.18.2, §6.17.2)
//
//   if ( FrameType == KEY_FRAME ) {
//     if ( obu == CLK && max_mlayer_id == 0 ) refresh_frame_flags = allFrames
//     else if ( enable_short_refresh_frame_flags ) {
//       frame_to_refresh f( CeilLog2( NumRefFrames ) )
//       refresh_frame_flags = 1 << frame_to_refresh
//     } else refresh_frame_flags f( NumRefFrames )
//   } else if ( FrameType == SWITCH_FRAME ) {
//     refresh_frame_flags f( NumRefFrames )
//   } else if ( enable_short_refresh_frame_flags ) {
//     has_refresh_frame_flags f(1)
//     if ( has_refresh_frame_flags ) {
//       frame_to_refresh f( CeilLog2( NumRefFrames ) )
//       refresh_frame_flags = 1 << frame_to_refresh
//     } else refresh_frame_flags = 0
//   } else refresh_frame_flags f( NumRefFrames )
// ---------------------------------------------------------------------------
struct RefreshConfig {
  bool enable_short_refresh_frame_flags = false;  // §5.4.6
  std::uint32_t num_ref_frames = 1;
  bool obu_is_closed_loop_key = true;  // CLK inference branch
  bool max_mlayer_id_zero = true;      // max_mlayer_id == 0
};

struct RefreshSignal {
  bool has_refresh_frame_flags = false;
  std::uint32_t frame_to_refresh = 0;
  std::uint32_t refresh_frame_flags = 0;  // resolved mask
};

// `immediate_output_frame` participates because §6.17.2 requires a non-zero
// refresh mask for every frame that is not output immediately.
bool write_refresh_frame_flags(BitWriter& writer, FrameType frame_type,
                               const RefreshConfig& config,
                               const RefreshSignal& signal,
                               bool immediate_output_frame,
                               std::string* error);
bool read_refresh_frame_flags(RefBitReader& reader, FrameType frame_type,
                              const RefreshConfig& config,
                              bool immediate_output_frame,
                              RefreshSignal* signal, std::string* error);
bool validate_refresh_frame_flags(FrameType frame_type,
                                  const RefreshConfig& config,
                                  const RefreshSignal& signal,
                                  bool immediate_output_frame,
                                  std::string* error);

// ---------------------------------------------------------------------------
// frame_size_with_refs() / found_ref (§5.18.4.3)
//
//   for ( i = 0; i < NumTotalRefs; i++ ) {
//     found_ref f(1)
//     if ( found_ref == 1 ) {
//       FrameWidth  = RefFrameWidth[  ref_frame_idx[ i ] ]
//       FrameHeight = RefFrameHeight[ ref_frame_idx[ i ] ]
//       break
//     }
//   }
//   if ( NumTotalRefs == 0 || found_ref == 0 ) frame_size()
//   else compute_image_size()
//
// frame_size() is the explicit form (§5.18.4.1):
//   frame_width_minus_1  f( frame_width_bits_minus_1  + 1 )
//   frame_height_minus_1 f( frame_height_bits_minus_1 + 1 )
// ---------------------------------------------------------------------------
struct SizeConfig {
  std::uint32_t num_total_refs = 0;
  std::uint32_t max_frame_width = 0;
  std::uint32_t max_frame_height = 0;
  std::uint32_t frame_width_bits = 1;
  std::uint32_t frame_height_bits = 1;
};

struct FrameSizeSignal {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  // Writer preference.  When true the writer inherits from the first reference
  // whose live size equals (width, height); when false it codes the explicit
  // frame_size() fields.
  bool prefer_found_ref = true;
};

struct FrameSizeResolved {
  std::array<bool, kRefsPerFrame> found_ref{};
  std::uint32_t inherited_from = kNoRef;  // ref-list index i, or kNoRef
  bool explicit_size = false;             // frame_size() was coded
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  // §5.18.4.4 compute_image_size() luma-derived unit counts.
  std::uint32_t mi_cols = 0;
  std::uint32_t mi_rows = 0;
};

bool write_frame_size_with_refs(BitWriter& writer, const SizeConfig& config,
                                const RefList& refs,
                                const FrameRefState& state,
                                const FrameSizeSignal& signal,
                                FrameSizeResolved* out, std::string* error);
bool read_frame_size_with_refs(RefBitReader& reader, const SizeConfig& config,
                               const RefList& refs,
                               const FrameRefState& state,
                               FrameSizeResolved* out, std::string* error);

// §6.17.4.3 ratio bounds: for every reference in the list,
//   2 * FrameWidth  >= RefFrameWidth,  FrameWidth  <= 16 * RefFrameWidth
//   2 * FrameHeight >= RefFrameHeight, FrameHeight <= 16 * RefFrameHeight
bool validate_frame_size_against_refs(const FrameSizeResolved& size,
                                      const RefList& refs,
                                      const FrameRefState& state,
                                      std::string* error);

// ---------------------------------------------------------------------------
// frame_header_copy() / header_bit[i] (§5.18.1, §6.17.1)
//
//   frame_header( isFirst ) {
//     if ( isFirst ) { ... frame_header_info(); NumFrameHeaderBits = ... }
//     else           { frame_header_copy() }
//   }
//   frame_header_copy() { for ( i = 0; i < NumFrameHeaderBits; i++ ) header_bit[ i ] f(1) }
//
// The copy is bit-for-bit and *not* byte-aligned: it starts wherever the
// previous syntax left the bit position.  A reader therefore has to replay the
// same number of bits, not the same number of bytes.
// ---------------------------------------------------------------------------
struct HeaderBits {
  std::size_t num_bits = 0;
  // Packed MSB-first, ceil(num_bits / 8) bytes, zero padding in the last byte.
  std::vector<std::uint8_t> bytes;

  bool empty() const { return num_bits == 0; }
  bool bit(std::size_t index) const;
};

// Extracts the bits [start_bit, end_bit) of a writer byte stream.  This is how
// NumFrameHeaderBits is captured without re-running the header writer.
HeaderBits capture_bits(std::span<const std::uint8_t> writer_bytes,
                        std::size_t start_bit, std::size_t end_bit);

std::size_t write_frame_header_copy(BitWriter& writer, const HeaderBits& bits,
                                    std::string* error);
bool read_frame_header_copy(RefBitReader& reader, const HeaderBits& expected,
                            std::string* error);

// Shared state for the multi-tile-group path: the first tile group of a frame
// records its frame_header_info() bits; every later tile group may either
// repeat them with frame_header_present_flag = 1 or omit the header.
class FrameHeaderStore {
 public:
  void reset();
  // Records `writer`'s bits in [start_bit, writer.bit_size()).
  void record(std::span<const std::uint8_t> writer_bytes, std::size_t start_bit,
              std::size_t end_bit);
  bool present() const { return bits_.num_bits != 0; }
  std::size_t num_bits() const { return bits_.num_bits; }
  const HeaderBits& bits() const { return bits_; }

  std::size_t write_copy(BitWriter& writer, std::string* error) const;
  bool verify_copy(RefBitReader& reader, std::string* error) const;

 private:
  HeaderBits bits_;
};

// §5.19 tile_group_obu() header for a non-first tile group:
//   frame_header_present_flag f(1)
//   if ( frame_header_present_flag ) frame_header_copy()
bool write_nonfirst_tile_group_header(BitWriter& writer,
                                      const FrameHeaderStore& store,
                                      bool frame_header_present,
                                      std::string* error);
bool read_nonfirst_tile_group_header(RefBitReader& reader,
                                     const FrameHeaderStore& store,
                                     bool* frame_header_present,
                                     std::string* error);

}  // namespace av2enc::frame::ref_fields
