// CUDA port of the normative AV2 v1.0.0 §8.2 symbol entropy coder.
//
// Mapping (docs/architecture.md, src/entropy/av2/README.md "CUDA port plan"):
//   * one CUDA workgroup (block) per tile;
//   * exactly one active coding lane per tile (lane 0 runs the serial coder);
//   * per-tile POD state, caller-owned per-tile output buffer;
//   * no mutable globals, no hidden allocations, no exceptions in the hot path;
//   * the §9.2 tables and the §8.2.6 arithmetic helpers come from
//     src/entropy/av2/av2_entropy.h unchanged (read-only reuse).
//
// Byte exactness
// --------------
// The device encoder is the exact inverse of the same §8.2 state machine the
// CPU reference implements.  It reuses av2::compute_boundaries(),
// av2::adapt_cdf(), av2::cdf_is_valid() and av2::floor_log2(), and mirrors
// av2::Encoder's low/range/carry/renormalisation and §8.2.4 termination
// bit-for-bit.  The only difference is *where* the bits live while a tile is
// being coded: a shared-memory staging buffer (kEntropyStageBytes) is flushed
// to the caller's output buffer when full and once at exit_symbol().  A carry
// that propagates into an already-flushed byte does a read-modify-write on that
// byte, so the payload is identical to the CPU coder's bit-at-a-time writer.
//
// A tile is serial.  Do not add intra-tile parallelism to the range coder.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

#include "../codec/entropy/av2/av2_entropy.h"

// `AV2_CUDA_UNROLL_1` is a device directive; the host pass of nvcc (and a
// host-only compiler) would warn about it.  Keep it device-only.
#if defined(__CUDA_ARCH__)
#define AV2_CUDA_UNROLL_1 _Pragma("unroll 1")
#else
#define AV2_CUDA_UNROLL_1
#endif

namespace av2enc::cuda {

using av2enc::av2::Status;

// One warp per tile.  Lane 0 codes; lanes 1..31 only help with the block-level
// CDF staging copy (before/after the serial loop) and are idle otherwise.
inline constexpr uint32_t kEntropyBlockThreads = 32;

// Shared-memory output staging per tile.  A tile's bits are written here and
// flushed to global memory as whole bytes; never one global store per bit.
inline constexpr uint32_t kEntropyStageBytes = 4096;  // 32 Kbit per tile

// Shared-memory CDF staging per tile, in uint16_t entries.  A tile whose CDF
// set fits is coded against shared memory and copied back afterwards; a larger
// set is coded directly in global memory (still correct, just slower).  The
// full §9.3 default set is ~46.7 KiB and does not fit the 48 KiB baseline.
inline constexpr uint32_t kEntropyCdfStageEntries = 2048;  // 4 KiB

// ---------------------------------------------------------------------------
// Caller-owned symbol tape
// ---------------------------------------------------------------------------
//
// The syntax layer emits a compact per-tile tape.  The entropy block walks it
// serially; the tape is not re-run through syntax derivation.
//
//   kSymbol   n = alphabet size 1..8, cdf_id selects the tile's CDF copy,
//             symbol is the coded value.
//   kBool     symbol = bit (0/1).
//   kLiteral  n = width 0..32, literal = value (MSB first; n read_bools).
//   kEnd      stop; remaining entries are ignored.
//
// Encode: kLiteral emits n raw bits via encode_bool (matching the CPU
// reference, which has no encode_literal and increments bool_count only).
// Decode: kLiteral calls read_literal(n), which adds n to symbol_count exactly
// like the CPU decoder.
enum class EntropyOp : uint8_t {
  kSymbol = 0,
  kBool = 1,
  kLiteral = 2,
  kEnd = 3,
};

struct EntropyTapeEntry {
  uint8_t op;        // EntropyOp
  uint8_t n;         // alphabet size (kSymbol) or width (kLiteral)
  uint16_t cdf_id;   // kSymbol: index into the tile's CDF set
  uint16_t symbol;   // kSymbol: coded value; kBool: bit
  uint16_t reserved;
  uint32_t literal;  // kLiteral: value
};

static_assert(sizeof(EntropyTapeEntry) == 12, "tape entry ABI");
static_assert(std::is_trivially_copyable_v<EntropyTapeEntry>);

inline EntropyTapeEntry make_symbol_entry(uint16_t cdf_id, uint8_t n,
                                          uint16_t symbol) noexcept {
  EntropyTapeEntry e{};
  e.op = static_cast<uint8_t>(EntropyOp::kSymbol);
  e.n = n;
  e.cdf_id = cdf_id;
  e.symbol = symbol;
  return e;
}

inline EntropyTapeEntry make_bool_entry(bool bit) noexcept {
  EntropyTapeEntry e{};
  e.op = static_cast<uint8_t>(EntropyOp::kBool);
  e.symbol = bit ? 1u : 0u;
  return e;
}

inline EntropyTapeEntry make_literal_entry(uint8_t width,
                                           uint32_t value) noexcept {
  EntropyTapeEntry e{};
  e.op = static_cast<uint8_t>(EntropyOp::kLiteral);
  e.n = width;
  e.literal = value;
  return e;
}

inline EntropyTapeEntry make_end_entry() noexcept {
  EntropyTapeEntry e{};
  e.op = static_cast<uint8_t>(EntropyOp::kEnd);
  return e;
}

// ---------------------------------------------------------------------------
// Per-tile descriptors and results (caller-owned device arrays)
// ---------------------------------------------------------------------------
//
// `cdfs` is one independent mutable copy of the tile's CDF set:
//   cdf_count arrays of cdf_entries uint16_t words.
// A tile must never share this copy with another tile.  The coder updates it in
// place (unless disable_cdf_update).  The caller can inspect the final CDFs
// after the launch, which is what the tests compare against the CPU coder.
//
// `flags` bits:
//   kEntropyEnableCdfUpdate  opt in to adaptive CDFs.  The default
//                            (`flags == 0`) is frozen, matching the M0
//                            bitstream policy; adaptive CDFs change pinned
//                            MD5s and are opt-in only.
//   kEntropyDisableStaging   write bits straight to `out` instead of the
//                            shared-memory window (correct but slower; a
//                            test/diagnostic mode)
//   kEntropyDisableCdfUpdate explicit freeze; redundant with the default but
//                            kept for callers that want to be unambiguous.
enum EntropyFlags : uint32_t {
  kEntropyEnableCdfUpdate = 1u << 0,
  kEntropyDisableStaging = 1u << 1,
  kEntropyDisableCdfUpdate = 1u << 2,
};

// Encode descriptors:
//   tape/tape_entries   input symbol tape
//   cdfs/cdf_count/cdf_entries  per-tile mutable CDF copies
//   out/out_capacity    caller-owned tile payload buffer (bytes)
//
// Decode descriptors:
//   in/payload_bytes    tile payload and its normative `sz`
//   backing_bytes       size of the `in` allocation; sz > backing is a
//                       kTruncatedInput error (same as the CPU decoder)
//   tape/tape_entries   op tape; decoded values are written back in place
//                       (kSymbol -> symbol, kBool -> symbol, kLiteral -> literal)
struct EntropyEncodeDesc {
  const EntropyTapeEntry* tape;
  uint32_t tape_entries;
  uint32_t flags;
  uint16_t* cdfs;
  uint32_t cdf_count;
  uint32_t cdf_entries;
  uint8_t* out;
  uint32_t out_capacity;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
};

struct EntropyDecodeDesc {
  const uint8_t* in;
  uint32_t payload_bytes;
  uint32_t backing_bytes;
  EntropyTapeEntry* tape;
  uint32_t tape_entries;
  uint32_t flags;
  uint16_t* cdfs;
  uint32_t cdf_count;
  uint32_t cdf_entries;
  uint32_t reserved0;
  uint32_t reserved1;
};

struct EntropyTileResult {
  uint32_t status;             // av2::Status (kOk on a clean finish)
  uint32_t payload_bytes;      // encode: exit_symbol() size; decode: declared sz
  uint64_t symbol_count;       // coder diagnostics, same meaning as CPU
  uint64_t bool_count;
  uint64_t tape_entries_used;  // entries consumed (stops at the first error)
  uint64_t cycles;             // clock64() delta of the serial lane
};

static_assert(sizeof(EntropyEncodeDesc) == 56, "encode descriptor ABI");
static_assert(sizeof(EntropyDecodeDesc) == 56, "decode descriptor ABI");
static_assert(sizeof(EntropyTileResult) == 40, "tile result ABI");
static_assert(std::is_trivially_copyable_v<EntropyEncodeDesc>);
static_assert(std::is_trivially_copyable_v<EntropyDecodeDesc>);
static_assert(std::is_trivially_copyable_v<EntropyTileResult>);

// ---------------------------------------------------------------------------
// Device coder — big-endian bit writer with a staging window
// ---------------------------------------------------------------------------

namespace detail {

struct StageWriter {
  uint8_t* out = nullptr;
  uint8_t* stage = nullptr;   // shared-memory staging, may be null
  uint32_t stage_bytes = 0;
  uint32_t capacity_bits = 0;
  uint32_t prefix_bits = 0;   // total bits emitted (same as CPU Encoder)
  uint32_t stage_base_bit = 0;
};

// Read a bit: staged bits from shared memory, flushed bits from `out`.
AV2_ENTROPY_HD inline uint32_t stage_bit_at(const StageWriter& w,
                                             uint32_t bit) noexcept {
  if (w.stage != nullptr && bit >= w.stage_base_bit) {
    const uint32_t off = bit - w.stage_base_bit;
    return (static_cast<uint32_t>(w.stage[off >> 3]) >> (7u - (off & 7u))) &
           1u;
  }
  return (static_cast<uint32_t>(w.out[bit >> 3]) >> (7u - (bit & 7u))) & 1u;
}

AV2_ENTROPY_HD inline void stage_set_bit(StageWriter& w, uint32_t bit,
                                          uint32_t value) noexcept {
  if (w.stage != nullptr && bit >= w.stage_base_bit) {
    const uint32_t off = bit - w.stage_base_bit;
    const uint32_t mask = 1u << (7u - (off & 7u));
    uint8_t* p = &w.stage[off >> 3];
    *p = (value != 0u) ? static_cast<uint8_t>(*p | mask)
                       : static_cast<uint8_t>(*p &
                                              static_cast<uint8_t>(~mask));
    return;
  }
  const uint32_t mask = 1u << (7u - (bit & 7u));
  uint8_t* p = &w.out[bit >> 3];
  *p = (value != 0u)
           ? static_cast<uint8_t>(*p | mask)
           : static_cast<uint8_t>(*p & static_cast<uint8_t>(~mask));
}

// Flush staged bits [stage_base_bit, prefix_bits) into `out`.  The bit count is
// a multiple of 8 by construction.  Never called with stage == nullptr.
AV2_ENTROPY_HD inline void stage_flush(StageWriter& w) noexcept {
  const uint32_t bits = w.prefix_bits - w.stage_base_bit;
  if (bits == 0u) {
    return;
  }
  const uint32_t bytes = bits >> 3;
  uint8_t* dst = w.out + (w.stage_base_bit >> 3);
AV2_CUDA_UNROLL_1
  for (uint32_t i = 0; i < bytes; ++i) {
    dst[i] = w.stage[i];
  }
  // The next bits start at staging offset 0; clear the window so that an OR of
  // a 1 bit cannot resurrect an earlier flush's bit.
AV2_CUDA_UNROLL_1
  for (uint32_t i = 0; i < bytes; ++i) {
    w.stage[i] = 0;
  }
  w.stage_base_bit += bits;
}

AV2_ENTROPY_HD inline void stage_append(StageWriter& w, uint32_t value,
                                         uint32_t n) noexcept {
  for (uint32_t k = 0; k < n; ++k) {
    if (w.stage != nullptr &&
        (w.prefix_bits - w.stage_base_bit) == (w.stage_bytes << 3)) {
      stage_flush(w);
    }
    stage_set_bit(w, w.prefix_bits, (value >> (n - 1u - k)) & 1u);
    ++w.prefix_bits;
  }
}

// Final carry propagation, identical to av2::detail::add_prefix_carry().  A
// carry only travels through a run of 1 bits and may cross the staging
// boundary; those bytes get a read-modify-write in `out`.
AV2_ENTROPY_HD inline Status add_prefix_carry(StageWriter& w,
                                               uint32_t carry) noexcept {
  uint32_t i = w.prefix_bits;
  while (carry != 0u && i > 0u) {
    --i;
    const uint32_t sum = stage_bit_at(w, i) + carry;
    stage_set_bit(w, i, sum & 1u);
    carry = sum >> 1u;
  }
  return (carry == 0u) ? Status::kOk : Status::kInvalidState;
}

AV2_ENTROPY_HD inline Status commit_interval(StageWriter& w, uint32_t& low,
                                              uint32_t& range, uint32_t l,
                                              uint32_t width,
                                              uint32_t renorm_bits) noexcept {
  if (renorm_bits > w.capacity_bits ||
      w.prefix_bits > w.capacity_bits - renorm_bits) {
    return Status::kBufferTooSmall;
  }

  const uint32_t new_low = low + l;
  const uint32_t carry = new_low >> 15u;
  low = new_low & av2enc::av2::kSymbolValueMask;
  if (carry != 0u) {
    const Status status = add_prefix_carry(w, carry);
    if (status != Status::kOk) {
      return status;
    }
  }

  if (renorm_bits != 0u) {
    const uint32_t chunk = low >> (15u - renorm_bits);
    stage_append(w, chunk, renorm_bits);
    low = (low << renorm_bits) & av2enc::av2::kSymbolValueMask;
  }
  range = width << renorm_bits;
  if (range < av2enc::av2::kSymbolRangeInit || range > 0xFFFFu) {
    return Status::kInvalidState;
  }
  return Status::kOk;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Per-tile encoder — exact inverse of av2::Encoder
// ---------------------------------------------------------------------------

struct TileEncoder {
  detail::StageWriter bw;
  uint32_t low = 0;
  uint32_t range = av2enc::av2::kSymbolRangeInit;
  uint64_t symbol_count = 0;
  uint64_t bool_count = 0;
  uint8_t finished = 0;
  uint8_t failed = 0;
  uint8_t disable_cdf_update = 0;
  uint8_t reserved = 0;

  // Same contract as av2::Encoder::init_symbol(): caller-owned output span.
  // `stage`/`stage_bytes` select the shared-memory staging window; passing
  // nullptr/0 writes bits directly to `buffer` (correct but slower).
  AV2_ENTROPY_HD inline Status init(uint8_t* buffer, size_t capacity_bytes,
                                    uint8_t* stage, uint32_t stage_bytes,
                                    bool stage_zeroed = false) noexcept {
    if (buffer == nullptr && capacity_bytes != 0) {
      return Status::kInvalidArgument;
    }
    if (capacity_bytes > static_cast<size_t>(0xFFFFFFFFu / 8u)) {
      return Status::kInvalidArgument;
    }
    bw.out = buffer;
    bw.stage = stage;
    bw.stage_bytes = (stage != nullptr) ? stage_bytes : 0u;
    bw.capacity_bits = static_cast<uint32_t>(capacity_bytes) * 8u;
    bw.prefix_bits = 0;
    bw.stage_base_bit = 0;
    low = 0;
    range = av2enc::av2::kSymbolRangeInit;
    symbol_count = 0;
    bool_count = 0;
    finished = 0;
    failed = 0;
    reserved = 0;
    if (bw.stage != nullptr && !stage_zeroed) {
AV2_CUDA_UNROLL_1
      for (uint32_t i = 0; i < bw.stage_bytes; ++i) {
        bw.stage[i] = 0;
      }
    }
    return Status::kOk;
  }

  AV2_ENTROPY_HD inline Status encode_symbol(uint16_t* cdf, int n,
                                             int symbol) noexcept {
    if (failed != 0u) {
      return Status::kEncoderFailed;
    }
    if (finished != 0u) {
      return Status::kAlreadyFinished;
    }
    if (n < 1 || n > av2enc::av2::kMaxSymbols || symbol < 0 || symbol >= n) {
      return Status::kInvalidSymbol;
    }
    if (n == 1) {
      if (symbol != 0) {
        return Status::kInvalidSymbol;
      }
      ++symbol_count;
      return Status::kOk;
    }
    if (!av2enc::av2::cdf_is_valid(cdf, n)) {
      return Status::kInvalidCdf;
    }

    av2enc::av2::Boundaries boundaries{};
    const Status boundary_status =
        av2enc::av2::compute_boundaries(range, cdf, n, boundaries);
    if (boundary_status != Status::kOk) {
      return boundary_status;
    }

    const uint32_t cur = boundaries.cur[symbol];
    const uint32_t prev = (symbol == 0) ? range : boundaries.cur[symbol - 1];
    const uint32_t width = prev - cur;
    if (width == 0u) {
      // A zero-probability symbol cannot be represented by the range coder.
      // Non-selected zero-width entries are fine and the decoder skips them;
      // requesting one here leaves the coder state untouched, exactly like
      // av2::Encoder::encode_symbol().
      return Status::kInvalidSymbol;
    }
    const uint32_t l =
        (symbol == 0) ? 0u : (range - boundaries.cur[symbol - 1]);
    const uint32_t renorm_bits = 15u - av2enc::av2::floor_log2(width);

    const Status commit_status =
        detail::commit_interval(bw, low, range, l, width, renorm_bits);
    if (commit_status != Status::kOk) {
      failed = 1;
      return commit_status;
    }

    if (disable_cdf_update == 0u) {
      av2enc::av2::adapt_cdf(cdf, n, symbol);
    }
    ++symbol_count;
    return Status::kOk;
  }

  AV2_ENTROPY_HD inline Status encode_bool(bool bit) noexcept {
    if (failed != 0u) {
      return Status::kEncoderFailed;
    }
    if (finished != 0u) {
      return Status::kAlreadyFinished;
    }
    if ((range & 1u) != 0u) {
      return Status::kInvalidState;
    }

    const uint32_t cur = range >> 1u;
    const uint32_t l = bit ? cur : 0u;
    const uint32_t width = cur;
    const uint32_t renorm_bits = 15u - av2enc::av2::floor_log2(width);
    const uint32_t old_range = range;

    const Status commit_status =
        detail::commit_interval(bw, low, range, l, width, renorm_bits);
    if (commit_status != Status::kOk) {
      failed = 1;
      return commit_status;
    }
    if (range != old_range) {
      failed = 1;
      return Status::kInvalidState;
    }
    ++bool_count;
    return Status::kOk;
  }

  AV2_ENTROPY_HD inline Status exit_symbol(size_t* payload_bytes) noexcept {
    if (failed != 0u) {
      return Status::kEncoderFailed;
    }
    if (finished != 0u) {
      return Status::kAlreadyFinished;
    }

    const uint32_t delta =
        (av2enc::av2::kTrailingFirstBit - low) & av2enc::av2::kSymbolValueMask;
    const uint32_t sum = low + delta;
    const uint32_t carry = sum >> 15u;
    if ((sum & av2enc::av2::kSymbolValueMask) !=
        av2enc::av2::kTrailingFirstBit) {
      failed = 1;
      return Status::kInvalidState;
    }

    constexpr uint32_t kMarkerBits = 15;
    if (bw.prefix_bits > 0xFFFFFFFFu - kMarkerBits - 7u) {
      failed = 1;
      return Status::kInvalidState;
    }
    const uint32_t pad_bits =
        (8u - ((bw.prefix_bits + kMarkerBits) & 7u)) & 7u;
    const uint32_t total_bits = bw.prefix_bits + kMarkerBits + pad_bits;
    if (total_bits > bw.capacity_bits) {
      failed = 1;
      return Status::kBufferTooSmall;
    }

    if (carry != 0u) {
      const Status carry_status = detail::add_prefix_carry(bw, carry);
      if (carry_status != Status::kOk) {
        failed = 1;
        return carry_status;
      }
    }

    detail::stage_append(bw, av2enc::av2::kTrailingFirstBit, kMarkerBits);
    if (pad_bits != 0u) {
      detail::stage_append(bw, 0u, pad_bits);
    }
    if (bw.stage != nullptr) {
      detail::stage_flush(bw);
    }

    finished = 1;
    if (payload_bytes != nullptr) {
      *payload_bytes = static_cast<size_t>(total_bits / 8u);
    }
    return Status::kOk;
  }
};

static_assert(std::is_trivially_copyable_v<TileEncoder>);
static_assert(std::is_standard_layout_v<TileEncoder>);

// ---------------------------------------------------------------------------
// Per-tile decoder — mirrors av2::Decoder
// ---------------------------------------------------------------------------

struct TileDecoder {
  const uint8_t* in = nullptr;
  uint32_t payload_bytes = 0;
  uint32_t payload_bits = 0;
  uint32_t bit_pos = 0;
  int64_t max_bits = 0;
  uint32_t value = 0;
  uint32_t range = av2enc::av2::kSymbolRangeInit;
  uint64_t symbol_count = 0;
  uint64_t bool_count = 0;
  uint8_t finished = 0;
  uint8_t failed = 0;
  uint8_t disable_cdf_update = 0;
  uint8_t reserved = 0;
  uint32_t cache_byte_index = 0xFFFFFFFFu;
  uint32_t cache_byte = 0;

  AV2_ENTROPY_HD inline uint32_t cached_bit(uint32_t pos) noexcept {
    const uint32_t index = pos >> 3;
    if (index != cache_byte_index) {
      cache_byte_index = index;
      cache_byte = static_cast<uint32_t>(in[index]);
    }
    return (cache_byte >> (7u - (pos & 7u))) & 1u;
  }

  AV2_ENTROPY_HD inline bool read_bits(uint32_t pos, uint32_t n,
                                        uint32_t* out) noexcept {
    if (n > payload_bits || pos > payload_bits - n) {
      return false;
    }
    uint32_t result = 0;
    for (uint32_t i = 0; i < n; ++i) {
      result = (result << 1u) | cached_bit(pos + i);
    }
    *out = result;
    return true;
  }

  AV2_ENTROPY_HD inline Status init(const uint8_t* data, size_t size_bytes,
                                    size_t sz) noexcept {
    if (sz > size_bytes) {
      return Status::kTruncatedInput;
    }
    if (sz != 0u && data == nullptr) {
      return Status::kInvalidArgument;
    }
    if (sz > static_cast<size_t>(0xFFFFFFFFu / 8u)) {
      return Status::kInvalidArgument;
    }

    in = data;
    payload_bytes = static_cast<uint32_t>(sz);
    payload_bits = payload_bytes * 8u;
    bit_pos = 0;
    max_bits = static_cast<int64_t>(payload_bits) - 15;
    value = 0;
    range = av2enc::av2::kSymbolRangeInit;
    symbol_count = 0;
    bool_count = 0;
    finished = 0;
    failed = 0;
    reserved = 0;
    cache_byte_index = 0xFFFFFFFFu;
    cache_byte = 0;

    const uint32_t num_bits = av2enc::av2::min_u32(payload_bits, 15u);
    uint32_t buf = 0;
    if (!read_bits(0, num_bits, &buf)) {
      return Status::kTruncatedInput;
    }
    value = av2enc::av2::kSymbolValueMask ^ (buf << (15u - num_bits));
    bit_pos = num_bits;
    return Status::kOk;
  }

  AV2_ENTROPY_HD inline Status decode_symbol(uint16_t* cdf, int n,
                                             int* symbol) noexcept {
    if (failed != 0u) {
      return Status::kDecoderFailed;
    }
    if (finished != 0u) {
      return Status::kAlreadyFinished;
    }
    if (symbol == nullptr || n < 1 || n > av2enc::av2::kMaxSymbols) {
      return Status::kInvalidArgument;
    }
    if (n == 1) {
      *symbol = 0;
      ++symbol_count;
      return Status::kOk;
    }
    if (!av2enc::av2::cdf_is_valid(cdf, n)) {
      return Status::kInvalidCdf;
    }

    av2enc::av2::Boundaries boundaries{};
    const Status boundary_status =
        av2enc::av2::compute_boundaries(range, cdf, n, boundaries);
    if (boundary_status != Status::kOk) {
      return boundary_status;
    }

    uint32_t prev = range;
    uint32_t cur = boundaries.cur[0];
    int selected = 0;
    while (selected < n - 1 && value < cur) {
      prev = cur;
      ++selected;
      cur = boundaries.cur[selected];
    }
    if (value < cur) {
      return Status::kInvalidCdf;
    }

    const uint32_t new_range = prev - cur;
    if (new_range == 0u) {
      // Defensive: the do/while above cannot stop on a zero-width interval,
      // but a repeated boundary must never feed floor_log2(0).
      return Status::kInvalidCdf;
    }
    const uint32_t new_value = value - cur;
    const uint32_t bits = 15u - av2enc::av2::floor_log2(new_range);

    uint32_t num_bits = 0;
    if (max_bits > 0) {
      const int64_t clipped =
          (max_bits < static_cast<int64_t>(bits))
              ? max_bits
              : static_cast<int64_t>(bits);
      num_bits = static_cast<uint32_t>(clipped);
    }

    uint32_t new_data = 0;
    if (!read_bits(bit_pos, num_bits, &new_data)) {
      failed = 1;
      return Status::kTruncatedInput;
    }
    bit_pos += num_bits;

    const uint32_t mask = (bits == 0u) ? 0u : ((1u << bits) - 1u);
    const uint32_t padded_data = new_data << (bits - num_bits);
    value = static_cast<uint32_t>(new_value << bits) | (padded_data ^ mask);
    range = new_range << bits;
    max_bits -= static_cast<int64_t>(bits);

    if (disable_cdf_update == 0u) {
      av2enc::av2::adapt_cdf(cdf, n, selected);
    }
    ++symbol_count;
    *symbol = selected;
    return Status::kOk;
  }

  AV2_ENTROPY_HD inline Status read_bool(bool* bit) noexcept {
    if (failed != 0u) {
      return Status::kDecoderFailed;
    }
    if (finished != 0u) {
      return Status::kAlreadyFinished;
    }
    if (bit == nullptr) {
      return Status::kInvalidArgument;
    }
    if ((range & 1u) != 0u) {
      return Status::kInvalidState;
    }

    const uint32_t cur = range >> 1u;
    const bool symbol = value < cur;
    uint32_t adjusted = value;
    if (!symbol) {
      adjusted -= cur;
    }

    const uint32_t num_bits = (max_bits > 0) ? 1u : 0u;
    uint32_t new_data = 0;
    if (!read_bits(bit_pos, num_bits, &new_data)) {
      failed = 1;
      return Status::kTruncatedInput;
    }
    bit_pos += num_bits;

    value = (adjusted << 1u) | (new_data ^ 1u);
    max_bits -= 1;
    if (value >= range) {
      failed = 1;
      return Status::kInvalidState;
    }
    ++bool_count;
    *bit = symbol;
    return Status::kOk;
  }

  AV2_ENTROPY_HD inline Status read_literal(int n,
                                            uint32_t* value_out) noexcept {
    if (failed != 0u) {
      return Status::kDecoderFailed;
    }
    if (finished != 0u) {
      return Status::kAlreadyFinished;
    }
    if (value_out == nullptr || n < 0 || n > 32) {
      return Status::kInvalidArgument;
    }

    uint32_t accum = 0;
    symbol_count += static_cast<uint64_t>(n);
AV2_CUDA_UNROLL_1
    for (int i = 0; i < n; ++i) {
      bool bit = false;
      const Status status = read_bool(&bit);
      if (status != Status::kOk) {
        return status;
      }
      accum = (accum << 1u) | (bit ? 1u : 0u);
    }
    *value_out = accum;
    return Status::kOk;
  }

  AV2_ENTROPY_HD inline Status exit_symbol() noexcept {
    if (failed != 0u) {
      return Status::kDecoderFailed;
    }
    if (finished != 0u) {
      return Status::kAlreadyFinished;
    }
    if (max_bits < av2enc::av2::kMinSymbolMaxBits) {
      failed = 1;
      return Status::kSymbolMaxBitsUnderflow;
    }

    const int64_t tail = max_bits + 15;
    const int64_t lookback = (tail < 15) ? tail : 15;
    const int64_t trailing = static_cast<int64_t>(bit_pos) - lookback;
    const int64_t padding_end =
        static_cast<int64_t>(bit_pos) + ((max_bits > 0) ? max_bits : 0);

    if (trailing < 0 || padding_end != static_cast<int64_t>(payload_bits) ||
        (padding_end & 7) != 0) {
      failed = 1;
      return Status::kInvalidTrailingBits;
    }
    if (cached_bit(static_cast<uint32_t>(trailing)) == 0u) {
      failed = 1;
      return Status::kInvalidTrailingBits;
    }
    for (int64_t pos = trailing + 1; pos < padding_end; ++pos) {
      if (cached_bit(static_cast<uint32_t>(pos)) != 0u) {
        failed = 1;
        return Status::kInvalidTrailingBits;
      }
    }

    finished = 1;
    return Status::kOk;
  }
};

static_assert(std::is_trivially_copyable_v<TileDecoder>);
static_assert(std::is_standard_layout_v<TileDecoder>);

// ---------------------------------------------------------------------------
// Tape execution (one lane)
// ---------------------------------------------------------------------------

namespace detail {

AV2_ENTROPY_HD inline Status tape_cdf(const EntropyTapeEntry& entry,
                                      uint16_t* cdfs, uint32_t cdf_count,
                                      uint32_t cdf_entries,
                                      uint16_t** out) noexcept {
  if (static_cast<uint32_t>(entry.cdf_id) >= cdf_count || cdfs == nullptr) {
    return Status::kInvalidArgument;
  }
  if (static_cast<uint32_t>(entry.n) + 1u > cdf_entries) {
    return Status::kInvalidArgument;
  }
  *out = cdfs + static_cast<size_t>(entry.cdf_id) * cdf_entries;
  return Status::kOk;
}

// Encode a tape.  Stops at the first error or kEnd.  Deterministic given the
// same CDF copies as the CPU reference.
AV2_ENTROPY_HD inline Status encode_tape(TileEncoder& enc,
                                         const EntropyTapeEntry* tape,
                                         uint32_t count, uint16_t* cdfs,
                                         uint32_t cdf_count,
                                         uint32_t cdf_entries,
                                         uint32_t* used) noexcept {
AV2_CUDA_UNROLL_1
  for (uint32_t i = 0; i < count; ++i) {
    *used = i + 1u;
    const EntropyTapeEntry entry = tape[i];
    const uint8_t op = entry.op;
    if (op == static_cast<uint8_t>(EntropyOp::kEnd)) {
      return Status::kOk;
    }
    if (op == static_cast<uint8_t>(EntropyOp::kSymbol)) {
      uint16_t* cdf = nullptr;
      const Status status =
          tape_cdf(entry, cdfs, cdf_count, cdf_entries, &cdf);
      if (status != Status::kOk) {
        return status;
      }
      const Status coded = enc.encode_symbol(
          cdf, static_cast<int>(entry.n), static_cast<int>(entry.symbol));
      if (coded != Status::kOk) {
        return coded;
      }
    } else if (op == static_cast<uint8_t>(EntropyOp::kBool)) {
      const Status status = enc.encode_bool(entry.symbol != 0u);
      if (status != Status::kOk) {
        return status;
      }
    } else if (op == static_cast<uint8_t>(EntropyOp::kLiteral)) {
      if (entry.n > 32u) {
        return Status::kInvalidArgument;
      }
AV2_CUDA_UNROLL_1
      for (uint32_t k = 0; k < entry.n; ++k) {
        const uint32_t bit = (entry.literal >> (entry.n - 1u - k)) & 1u;
        const Status status = enc.encode_bool(bit != 0u);
        if (status != Status::kOk) {
          return status;
        }
      }
    } else {
      return Status::kInvalidArgument;
    }
  }
  return Status::kOk;
}

// Decode a tape in place: kSymbol writes `symbol`, kBool writes `symbol`,
// kLiteral writes `literal`.  Counts match av2::Decoder exactly (read_literal
// adds n to symbol_count).
AV2_ENTROPY_HD inline Status decode_tape(TileDecoder& dec,
                                         EntropyTapeEntry* tape,
                                         uint32_t count, uint16_t* cdfs,
                                         uint32_t cdf_count,
                                         uint32_t cdf_entries,
                                         uint32_t* used) noexcept {
AV2_CUDA_UNROLL_1
  for (uint32_t i = 0; i < count; ++i) {
    *used = i + 1u;
    EntropyTapeEntry& entry = tape[i];
    const uint8_t op = entry.op;
    if (op == static_cast<uint8_t>(EntropyOp::kEnd)) {
      return Status::kOk;
    }
    if (op == static_cast<uint8_t>(EntropyOp::kSymbol)) {
      uint16_t* cdf = nullptr;
      const Status status =
          tape_cdf(entry, cdfs, cdf_count, cdf_entries, &cdf);
      if (status != Status::kOk) {
        return status;
      }
      int symbol = 0;
      const Status decoded = dec.decode_symbol(
          cdf, static_cast<int>(entry.n), &symbol);
      if (decoded != Status::kOk) {
        return decoded;
      }
      entry.symbol = static_cast<uint16_t>(symbol);
    } else if (op == static_cast<uint8_t>(EntropyOp::kBool)) {
      bool bit = false;
      const Status status = dec.read_bool(&bit);
      if (status != Status::kOk) {
        return status;
      }
      entry.symbol = bit ? 1u : 0u;
    } else if (op == static_cast<uint8_t>(EntropyOp::kLiteral)) {
      uint32_t literal = 0;
      const Status status =
          dec.read_literal(static_cast<int>(entry.n), &literal);
      if (status != Status::kOk) {
        return status;
      }
      entry.literal = literal;
    } else {
      return Status::kInvalidArgument;
    }
  }
  return Status::kOk;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Kernels and host launch helpers (definitions in entropy.cu)
// ---------------------------------------------------------------------------
//
// Grid = one block per tile; block = kEntropyBlockThreads.  Descriptors and
// results are caller-owned device arrays.  The launch helper does not allocate,
// does not synchronize, and returns the launch status; results must still be
// copied out by the caller after the launch completes.
__global__ void entropy_encode_kernel(const EntropyEncodeDesc* __restrict__ tiles,
                                      EntropyTileResult* __restrict__ results,
                                      uint32_t tile_count);

__global__ void entropy_decode_kernel(const EntropyDecodeDesc* __restrict__ tiles,
                                      EntropyTileResult* __restrict__ results,
                                      uint32_t tile_count);

cudaError_t launch_entropy_encode(const EntropyEncodeDesc* d_tiles,
                                  EntropyTileResult* d_results,
                                  uint32_t tile_count,
                                  cudaStream_t stream = nullptr);

cudaError_t launch_entropy_decode(const EntropyDecodeDesc* d_tiles,
                                  EntropyTileResult* d_results,
                                  uint32_t tile_count,
                                  cudaStream_t stream = nullptr);

}  // namespace av2enc::cuda

#undef AV2_CUDA_UNROLL_1
