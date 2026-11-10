// Normative AV2 v1.0.0 symbol entropy coder (§8.2).
//
// This is a from-scratch implementation of the *decoder* state machine in
// AV2_Spec_v1.0.0.pdf §8.2.1–§8.2.6, plus the exact inverse encoder.  The
// decoder is the normative object: the encoder is built so that a normative
// decoder reproduces the same symbol sequence and CDF state.
//
// Implemented clauses:
//   §8.2.2  init_symbol(sz): 15-bit inverted initial value, SymbolMaxBits
//   §8.2.3  read_bool(): half-range split, one-bit value renormalisation
//   §8.2.4  exit_symbol(): trailing 1-bit, zero padding, byte alignment
//   §8.2.5  read_literal(n)
//   §8.2.6  read_symbol(cdf): inverse CDF boundaries, Prob_Inc, EC_PROB_SHIFT,
//           bit-granular renormalisation with padded zero bits, CDF update
//   §9.2    Prob_Inc[7][8], Para_Adjustment_List[125][3] (generated header)
//
// CUDA portability contract
// -------------------------
// The coder is a POD state machine plus caller-owned byte buffers.  All hot
// paths are `AV2_ENTROPY_HD` inline functions: no mutable globals, no heap, no
// exceptions, no virtual calls, no STL containers.  The only host-only pieces
// are the std::span overloads and the §8.2.4 CDF copy/average helpers in
// av2_entropy.cpp.  For a CUDA build the same header can be included from a
// .cu translation unit: av2_entropy_tables.h switches to
// `static __device__ __constant__` in nvcc's device pass and stays
// `inline constexpr` in the host pass.  Verified with nvcc 12.8, sm_86.
//
// Deliberately no:
//   * global mutable state, heap allocation, exceptions, or virtual calls;
//   * std::vector in the coder (the test binary may use it);
//   * AV2 syntax/context derivation (§8.3) — callers own the Tile*Cdf arrays.
//
// Byte-buffer contract
// --------------------
// Encoder: caller supplies one output span per tile.  The encoder writes only
// forward, never reads the output except for carry propagation across already
// written prefix bits.  State is POD and can live in registers/shared memory.
// Decoder: caller supplies one read-only span per tile and the tile's `sz`.
//
// A tile is serial.  Tiles are independent; never share an Encoder/Decoder or
// a CDF array between two tiles without external synchronization.

#pragma once

#include <cstddef>
#include <cstdint>

#include "av2_entropy_tables.h"

#if !defined(__CUDACC__)
#include <span>
#endif

#if defined(__CUDACC__)
#define AV2_ENTROPY_HD __host__ __device__
#else
#define AV2_ENTROPY_HD
#endif

namespace av2enc::av2 {

// §8.2: the coder's probability scale and initial range/value width.
inline constexpr uint32_t kCdfScale = 1u << 15;          // 32768
inline constexpr uint32_t kSymbolRangeInit = 1u << 15;   // 1 << 15
inline constexpr uint32_t kSymbolValueMask = (1u << 15) - 1u;
inline constexpr uint32_t kTrailingFirstBit = 1u << 14;  // 15-bit 100...0
inline constexpr int kProbShift = 7;                     // EC_PROB_SHIFT
inline constexpr int kMaxSymbols = 8;                    // Prob_Inc rows: N-2 <= 6
inline constexpr uint32_t kCdfCountCap = 32;             // cdf[N] cap
inline constexpr int64_t kMinSymbolMaxBits = -14;        // §8.2.4 requirement

// The normative default CDF tables live in spec/tables/default_cdf.json.  This
// module only needs the two constant tables above; callers copy the Tile*Cdf
// arrays they need (one independent copy per tile) and pass them in.

enum class Status : uint8_t {
  kOk = 0,
  kInvalidArgument,        // null pointer / size out of range
  kInvalidCdf,             // malformed inverse CDF or degenerate interval
  kInvalidSymbol,          // symbol not in [0, N)
  kInvalidState,           // coder invariant violated
  kBufferTooSmall,         // encoder output span exhausted
  kTruncatedInput,         // decoder payload shorter than declared sz
  kInvalidTrailingBits,    // §8.2.4 marker/padding check failed
  kSymbolMaxBitsUnderflow, // §8.2.4 SymbolMaxBits < -14
  kAlreadyFinished,        // encode/decode after exit_symbol()
  kEncoderFailed,          // encoder previously hit a fatal error
  kDecoderFailed,          // decoder previously hit a fatal error
};

const char* status_name(Status status) noexcept;

// ---------------------------------------------------------------------------
// Small §8.2 helpers (exported for tests and for the future CUDA core).
// ---------------------------------------------------------------------------

AV2_ENTROPY_HD inline uint32_t floor_log2(uint32_t v) noexcept {
#if defined(__CUDA_ARCH__)
  return 31u - static_cast<uint32_t>(__clz(v));
#elif defined(__GNUC__) || defined(__clang__)
  return 31u - static_cast<uint32_t>(__builtin_clz(v));
#else
  uint32_t r = 0;
  while (v > 1u) {
    v >>= 1u;
    ++r;
  }
  return r;
#endif
}

AV2_ENTROPY_HD inline uint32_t min_u32(uint32_t a, uint32_t b) noexcept {
  return a < b ? a : b;
}

// A normative CDF array has N+1 entries:
//   cdf[0..N-2]  inverse CDF values, strictly increasing in [0, 32767]
//   cdf[N-1]     constant base adaptation rate (index into Para list, < 125)
//   cdf[N]       use count, 0..32
AV2_ENTROPY_HD inline bool cdf_is_valid(const uint16_t* cdf, int n) noexcept {
  if (cdf == nullptr || n < 1 || n > kMaxSymbols) {
    return false;
  }
  if (n == 1) {
    return true;  // non-normative degenerate alphabet; no table is consulted
  }
  if (cdf[n - 1] >= kNumParaCombinations || cdf[n] > kCdfCountCap) {
    return false;
  }
  for (int i = 0; i < n - 1; ++i) {
    if (cdf[i] >= kCdfScale) {
      return false;
    }
    if (i > 0 && cdf[i] <= cdf[i - 1]) {
      return false;
    }
  }
  return true;
}

// §8.2.6 inverse-CDF boundaries.  `cur[s]` is the cumulative boundary for
// symbol s; the boundaries are strictly descending and cur[N-1] == 0.
struct Boundaries {
  uint32_t cur[kMaxSymbols];
};

AV2_ENTROPY_HD inline Status compute_boundaries(
    uint32_t symbol_range, const uint16_t* cdf, int n,
    Boundaries& out) noexcept {
  if (cdf == nullptr || n < 2 || n > kMaxSymbols ||
      symbol_range < kSymbolRangeInit || symbol_range > 0xFFFFu) {
    return Status::kInvalidCdf;
  }
  uint32_t prev = symbol_range;
  for (int s = 0; s < n; ++s) {
    if (s != n - 1 && cdf[s] >= kCdfScale) {
      return Status::kInvalidCdf;
    }
    const uint32_t f =
        (s == n - 1) ? 0u : (kCdfScale - static_cast<uint32_t>(cdf[s]));
    const uint32_t pp = ((f >> kProbShift) << 4) +
                        static_cast<uint32_t>(kProbInc[n - 2][s]);
    const uint32_t cur = (((symbol_range >> 8) * pp) >> 7) << 3;
    if (cur >= prev) {
      return Status::kInvalidCdf;  // non-positive interval
    }
    out.cur[s] = cur;
    prev = cur;
  }
  return Status::kOk;
}

// §8.2.6 CDF adaptation.  Must be called with the same symbol on encoder and
// decoder after a symbol has been coded, unless disable_cdf_update is set.
AV2_ENTROPY_HD inline void adapt_cdf(uint16_t* cdf, int n, int symbol) noexcept {
  const uint32_t time_interval =
      (cdf[n] > 31u) ? 2u : ((cdf[n] > 15u) ? 1u : 0u);
  const int rate =
      3 + static_cast<int>(time_interval) +
      static_cast<int>(min_u32(floor_log2(static_cast<uint32_t>(n)), 2u)) +
      static_cast<int>(kParaAdjustmentList[cdf[n - 1]][time_interval]);
  for (int i = 0; i < n - 1; ++i) {
    uint32_t value = cdf[i];
    if (i < symbol) {
      value -= value >> rate;
    } else {
      value += (kCdfScale - value) >> rate;
    }
    cdf[i] = static_cast<uint16_t>(value);
  }
  if (cdf[n] < kCdfCountCap) {
    ++cdf[n];
  }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

// Encoder invariant after each successful symbol:
//   32768 <= range < 65536, and range is a multiple of 8
//   0 <= low < 32768                 (the un-emitted 15-bit suffix)
//   output bits [0, prefix_bits)     (fixed prefix, modulo a final carry)
//   code value = prefix * 2^15 + low, interval width = range
// exit_symbol() replaces `low` with the §8.2.4 marker 0x4000 and pads to a
// byte boundary.  The last 15 information bits are then exactly 1 + 14 zeros,
// which is what exit_symbol() requires.
struct Encoder {
  uint8_t* out = nullptr;
  uint32_t capacity_bits = 0;
  uint32_t prefix_bits = 0;    // total bits written (after exit: incl. marker)
  uint32_t marker_bit_pos = 0; // §8.2.4 marker position; valid after exit
  uint32_t low = 0;
  uint32_t range = kSymbolRangeInit;
  uint64_t symbol_count = 0;
  uint64_t bool_count = 0;
  uint8_t finished = 0;
  uint8_t failed = 0;
  uint8_t disable_cdf_update = 0;
  uint8_t reserved = 0;

  // Encoder counterpart of §8.2.2.  `sz` is not known until exit_symbol(),
  // because the arithmetic coder decides the payload length; the output span
  // is the resource limit.  This allocates nothing.
  AV2_ENTROPY_HD Status init_symbol(uint8_t* buffer,
                                    size_t capacity_bytes) noexcept;

  // §8.2.6 inverse.  `cdf` has n+1 entries and is updated in place unless
  // disable_cdf_update is set.  Atomic on invalid symbol/CDF: no state change.
  AV2_ENTROPY_HD Status encode_symbol(uint16_t* cdf, int n,
                                      int symbol) noexcept;

  // §8.2.3 inverse.  Range is unchanged by a boolean; one raw bit is emitted.
  AV2_ENTROPY_HD Status encode_bool(bool bit) noexcept;

  // §8.2.4 termination.  Writes the trailing 1-bit marker and zero padding,
  // then returns the final tile payload size in bytes.
  AV2_ENTROPY_HD Status exit_symbol(size_t* payload_bytes) noexcept;

#if !defined(__CUDACC__)
  inline Status init_symbol(std::span<std::byte> buffer) noexcept {
    return init_symbol(reinterpret_cast<uint8_t*>(buffer.data()),
                       buffer.size());
  }
#endif
};

namespace detail {

AV2_ENTROPY_HD inline uint32_t bit_at(const uint8_t* data,
                                      uint32_t bit_pos) noexcept {
  return (static_cast<uint32_t>(data[bit_pos >> 3]) >>
          (7u - (bit_pos & 7u))) &
         1u;
}

AV2_ENTROPY_HD inline void set_bit_at(uint8_t* data, uint32_t bit_pos,
                                      uint32_t bit) noexcept {
  const uint32_t mask = 1u << (7u - (bit_pos & 7u));
  if (bit != 0u) {
    data[bit_pos >> 3] = static_cast<uint8_t>(data[bit_pos >> 3] | mask);
  } else {
    data[bit_pos >> 3] =
        static_cast<uint8_t>(data[bit_pos >> 3] & static_cast<uint8_t>(~mask));
  }
}

AV2_ENTROPY_HD inline bool read_bits(const uint8_t* data, uint32_t payload_bits,
                                     uint32_t pos, uint32_t n,
                                     uint32_t* out) noexcept {
  if (n > payload_bits || pos > payload_bits - n) {
    return false;
  }
  uint32_t value = 0;
  for (uint32_t i = 0; i < n; ++i) {
    value = (value << 1u) | bit_at(data, pos + i);
  }
  *out = value;
  return true;
}

// Append the low `n` bits of `value`, most-significant bit first.  Capacity is
// checked by the caller.
AV2_ENTROPY_HD inline void append_bits_unchecked(Encoder& e, uint32_t value,
                                                uint32_t n) noexcept {
  for (uint32_t k = 0; k < n; ++k) {
    const uint32_t bit = (value >> (n - 1u - k)) & 1u;
    set_bit_at(e.out, e.prefix_bits, bit);
    ++e.prefix_bits;
  }
}

// Add `carry` to the big-endian bit array in [0, prefix_bits).  This is the
// final carry propagation from the low 15-bit register into the emitted
// prefix.  A carry can only travel through a run of 1 bits; it never changes
// bits beyond the prefix.
AV2_ENTROPY_HD inline Status add_prefix_carry(Encoder& e,
                                              uint32_t carry) noexcept {
  uint32_t i = e.prefix_bits;
  while (carry != 0u && i > 0u) {
    --i;
    const uint32_t sum = bit_at(e.out, i) + carry;
    set_bit_at(e.out, i, sum & 1u);
    carry = sum >> 1u;
  }
  return carry == 0u ? Status::kOk : Status::kInvalidState;
}

// Commit an already-computed symbol interval.  Called only after all
// validation, so a failure here is a poisoned coder, not a rejected argument.
AV2_ENTROPY_HD inline Status commit_interval(Encoder& e, uint32_t l,
                                             uint32_t width,
                                             uint32_t renorm_bits) noexcept {
  if (renorm_bits > e.capacity_bits ||
      e.prefix_bits > e.capacity_bits - renorm_bits) {
    e.failed = 1;
    return Status::kBufferTooSmall;
  }

  const uint32_t new_low = e.low + l;
  const uint32_t carry = new_low >> 15u;
  e.low = new_low & kSymbolValueMask;
  if (carry != 0u) {
    const Status status = add_prefix_carry(e, carry);
    if (status != Status::kOk) {
      e.failed = 1;
      return status;
    }
  }

  if (renorm_bits != 0u) {
    const uint32_t chunk = e.low >> (15u - renorm_bits);
    append_bits_unchecked(e, chunk, renorm_bits);
    e.low = (e.low << renorm_bits) & kSymbolValueMask;
  }
  e.range = width << renorm_bits;
  if (e.range < kSymbolRangeInit || e.range > 0xFFFFu) {
    e.failed = 1;
    return Status::kInvalidState;
  }
  return Status::kOk;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

// Decoder state is exactly the §8.2 state:
//   value       SymbolValue  (15/16-bit, always < range)
//   range       SymbolRange  (multiple of 8, 32768 <= range < 65536)
//   max_bits    SymbolMaxBits (may go negative; bounded by exit_symbol)
//   bit_pos     get_position()
// init_symbol() reads Min(8*sz, 15) bits; every later read is clipped to
// SymbolMaxBits, so no padding byte is ever read as data.
struct Decoder {
  const uint8_t* in = nullptr;
  uint32_t payload_bytes = 0;
  uint32_t payload_bits = 0;
  uint32_t bit_pos = 0;
  int64_t max_bits = 0;
  uint32_t value = 0;
  uint32_t range = kSymbolRangeInit;
  uint64_t symbol_count = 0;
  uint64_t bool_count = 0;
  uint8_t finished = 0;
  uint8_t failed = 0;
  uint8_t disable_cdf_update = 0;
  uint8_t reserved = 0;

  // §8.2.2.  `sz` is the tile payload byte count from the bitstream syntax.
  // `size_bytes` is the size of the backing buffer; sz > size_bytes is a
  // truncation error.  The decoder never accesses beyond sz bytes.
  AV2_ENTROPY_HD Status init_symbol(const uint8_t* data, size_t size_bytes,
                                    size_t sz) noexcept;

  // §8.2.6.  `cdf` has n+1 entries and is updated in place unless
  // disable_cdf_update is set.
  AV2_ENTROPY_HD Status decode_symbol(uint16_t* cdf, int n,
                                      int* symbol) noexcept;

  // §8.2.3.  The decoder is the normative direction for read_bool().
  AV2_ENTROPY_HD Status read_bool(bool* bit) noexcept;

  // §8.2.5.
  AV2_ENTROPY_HD Status read_literal(int n, uint32_t* value_out) noexcept;

  // §8.2.4.  Verifies the trailing 1 bit, the zero padding, and byte
  // alignment.  On failure the decoder is poisoned.
  AV2_ENTROPY_HD Status exit_symbol() noexcept;

#if !defined(__CUDACC__)
  inline Status init_symbol(std::span<const std::byte> data,
                            size_t sz) noexcept {
    return init_symbol(reinterpret_cast<const uint8_t*>(data.data()),
                       data.size(), sz);
  }
#endif
};

// ---------------------------------------------------------------------------
// Encoder implementations (§8.2.2/§8.2.3/§8.2.4/§8.2.6 inverse)
// ---------------------------------------------------------------------------

AV2_ENTROPY_HD inline Status Encoder::init_symbol(uint8_t* buffer,
                                                  size_t capacity_bytes) noexcept {
  if (buffer == nullptr && capacity_bytes != 0) {
    return Status::kInvalidArgument;
  }
  if (capacity_bytes > static_cast<size_t>(0xFFFFFFFFu / 8u)) {
    return Status::kInvalidArgument;
  }
  out = buffer;
  capacity_bits = static_cast<uint32_t>(capacity_bytes) * 8u;
  prefix_bits = 0;
  marker_bit_pos = 0;
  low = 0;
  range = kSymbolRangeInit;
  symbol_count = 0;
  bool_count = 0;
  finished = 0;
  failed = 0;
  reserved = 0;
  return Status::kOk;
}

AV2_ENTROPY_HD inline Status Encoder::encode_symbol(uint16_t* cdf, int n,
                                                    int symbol) noexcept {
  if (failed != 0u) {
    return Status::kEncoderFailed;
  }
  if (finished != 0u) {
    return Status::kAlreadyFinished;
  }
  if (n < 1 || n > kMaxSymbols || symbol < 0 || symbol >= n) {
    return Status::kInvalidSymbol;
  }
  if (n == 1) {
    if (symbol != 0) {
      return Status::kInvalidSymbol;
    }
    ++symbol_count;
    return Status::kOk;
  }
  if (!cdf_is_valid(cdf, n)) {
    return Status::kInvalidCdf;
  }

  Boundaries boundaries{};
  const Status boundary_status = compute_boundaries(range, cdf, n, boundaries);
  if (boundary_status != Status::kOk) {
    return boundary_status;
  }

  const uint32_t cur = boundaries.cur[symbol];
  const uint32_t prev = (symbol == 0) ? range : boundaries.cur[symbol - 1];
  const uint32_t width = prev - cur;
  const uint32_t l = (symbol == 0) ? 0u : (range - boundaries.cur[symbol - 1]);
  const uint32_t renorm_bits = 15u - floor_log2(width);

  const Status commit_status = detail::commit_interval(*this, l, width,
                                                       renorm_bits);
  if (commit_status != Status::kOk) {
    return commit_status;
  }

  if (disable_cdf_update == 0u) {
    adapt_cdf(cdf, n, symbol);
  }
  ++symbol_count;
  return Status::kOk;
}

AV2_ENTROPY_HD inline Status Encoder::encode_bool(bool bit) noexcept {
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
  const uint32_t renorm_bits = 15u - floor_log2(width);
  const uint32_t old_range = range;

  const Status commit_status = detail::commit_interval(*this, l, width,
                                                       renorm_bits);
  if (commit_status != Status::kOk) {
    return commit_status;
  }
  if (range != old_range) {
    failed = 1;
    return Status::kInvalidState;
  }
  ++bool_count;
  return Status::kOk;
}

AV2_ENTROPY_HD inline Status Encoder::exit_symbol(
    size_t* payload_bytes) noexcept {
  if (failed != 0u) {
    return Status::kEncoderFailed;
  }
  if (finished != 0u) {
    return Status::kAlreadyFinished;
  }

  // Choose the final code value C = prefix * 2^15 + low + delta so that
  // C mod 2^15 == 0x4000.  Because range >= 2^15 after every symbol, the
  // interval [prefix*2^15 + low, ... + range) always contains such a value.
  const uint32_t delta = (kTrailingFirstBit - low) & kSymbolValueMask;
  const uint32_t sum = low + delta;
  const uint32_t carry = sum >> 15u;
  if ((sum & kSymbolValueMask) != kTrailingFirstBit) {
    failed = 1;
    return Status::kInvalidState;
  }

  constexpr uint32_t kMarkerBits = 15;
  if (prefix_bits > 0xFFFFFFFFu - kMarkerBits - 7u) {
    failed = 1;
    return Status::kInvalidState;
  }
  const uint32_t pad_bits = (8u - ((prefix_bits + kMarkerBits) & 7u)) & 7u;
  const uint32_t total_bits = prefix_bits + kMarkerBits + pad_bits;
  if (total_bits > capacity_bits) {
    failed = 1;
    return Status::kBufferTooSmall;
  }

  if (carry != 0u) {
    const Status carry_status = detail::add_prefix_carry(*this, carry);
    if (carry_status != Status::kOk) {
      failed = 1;
      return carry_status;
    }
  }

  marker_bit_pos = prefix_bits;
  detail::append_bits_unchecked(*this, kTrailingFirstBit, kMarkerBits);
  if (pad_bits != 0u) {
    detail::append_bits_unchecked(*this, 0u, pad_bits);
  }

  finished = 1;
  if (payload_bytes != nullptr) {
    *payload_bytes = static_cast<size_t>(total_bits / 8u);
  }
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// Decoder implementations (§8.2.2–§8.2.6)
// ---------------------------------------------------------------------------

AV2_ENTROPY_HD inline Status Decoder::init_symbol(const uint8_t* data,
                                                  size_t size_bytes,
                                                  size_t sz) noexcept {
  if (sz > size_bytes) {
    return Status::kTruncatedInput;
  }
  if (sz != 0 && data == nullptr) {
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
  range = kSymbolRangeInit;
  symbol_count = 0;
  bool_count = 0;
  finished = 0;
  failed = 0;
  reserved = 0;

  const uint32_t num_bits = min_u32(payload_bits, 15u);
  uint32_t buf = 0;
  if (!detail::read_bits(data, payload_bits, 0, num_bits, &buf)) {
    return Status::kTruncatedInput;
  }
  value = kSymbolValueMask ^ (buf << (15u - num_bits));
  bit_pos = num_bits;
  return Status::kOk;
}

AV2_ENTROPY_HD inline Status Decoder::decode_symbol(uint16_t* cdf, int n,
                                                    int* symbol) noexcept {
  if (failed != 0u) {
    return Status::kDecoderFailed;
  }
  if (finished != 0u) {
    return Status::kAlreadyFinished;
  }
  if (symbol == nullptr || n < 1 || n > kMaxSymbols) {
    return Status::kInvalidArgument;
  }
  if (n == 1) {
    *symbol = 0;
    ++symbol_count;
    return Status::kOk;
  }
  if (!cdf_is_valid(cdf, n)) {
    return Status::kInvalidCdf;
  }

  Boundaries boundaries{};
  const Status boundary_status =
      compute_boundaries(range, cdf, n, boundaries);
  if (boundary_status != Status::kOk) {
    return boundary_status;
  }

  // §8.2.6 do/while: symbol 0 is the interval [boundary[0], range).
  uint32_t prev = range;
  uint32_t cur = boundaries.cur[0];
  int selected = 0;
  while (selected < n - 1 && value < cur) {
    prev = cur;
    ++selected;
    cur = boundaries.cur[selected];
  }
  if (value < cur) {
    return Status::kInvalidCdf;  // only possible for a degenerate last interval
  }

  const uint32_t new_range = prev - cur;
  const uint32_t new_value = value - cur;
  const uint32_t bits = 15u - floor_log2(new_range);

  uint32_t num_bits = 0;
  if (max_bits > 0) {
    const int64_t clipped =
        (max_bits < static_cast<int64_t>(bits))
            ? max_bits
            : static_cast<int64_t>(bits);
    num_bits = static_cast<uint32_t>(clipped);
  }

  uint32_t new_data = 0;
  if (!detail::read_bits(in, payload_bits, bit_pos, num_bits, &new_data)) {
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
    adapt_cdf(cdf, n, selected);
  }
  ++symbol_count;
  *symbol = selected;
  return Status::kOk;
}

AV2_ENTROPY_HD inline Status Decoder::read_bool(bool* bit) noexcept {
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
  if (!detail::read_bits(in, payload_bits, bit_pos, num_bits, &new_data)) {
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

AV2_ENTROPY_HD inline Status Decoder::read_literal(int n,
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

  uint32_t value_accum = 0;
  symbol_count += static_cast<uint64_t>(n);
  for (int i = 0; i < n; ++i) {
    bool bit = false;
    const Status status = read_bool(&bit);
    if (status != Status::kOk) {
      return status;
    }
    value_accum = (value_accum << 1u) | (bit ? 1u : 0u);
  }
  *value_out = value_accum;
  return Status::kOk;
}

AV2_ENTROPY_HD inline Status Decoder::exit_symbol() noexcept {
  if (failed != 0u) {
    return Status::kDecoderFailed;
  }
  if (finished != 0u) {
    return Status::kAlreadyFinished;
  }
  if (max_bits < kMinSymbolMaxBits) {
    failed = 1;
    return Status::kSymbolMaxBitsUnderflow;
  }

  // §8.2.4:
  //   trailingBitPosition = get_position() - Min(15, SymbolMaxBits + 15)
  //   advance by Max(0, SymbolMaxBits)
  //   paddingEndPosition = get_position()
  const int64_t tail = max_bits + 15;
  const int64_t lookback = (tail < 15) ? tail : 15;
  const int64_t trailing = static_cast<int64_t>(bit_pos) - lookback;
  const int64_t padding_end =
      static_cast<int64_t>(bit_pos) + ((max_bits > 0) ? max_bits : 0);

  if (trailing < 0 ||
      padding_end != static_cast<int64_t>(payload_bits) ||
      (padding_end & 7) != 0) {
    failed = 1;
    return Status::kInvalidTrailingBits;
  }
  if (detail::bit_at(in, static_cast<uint32_t>(trailing)) == 0u) {
    failed = 1;
    return Status::kInvalidTrailingBits;
  }
  for (int64_t pos = trailing + 1; pos < padding_end; ++pos) {
    if (detail::bit_at(in, static_cast<uint32_t>(pos)) != 0u) {
      failed = 1;
      return Status::kInvalidTrailingBits;
    }
  }

  finished = 1;
  return Status::kOk;
}

// ---------------------------------------------------------------------------
// §8.2.4 CDF copy/average helpers for multi-tile tile groups.
// `n` is the symbol count (CDF array length is n+1).  `num_log2` is
// Min(3, FloorLog2(TileCols * TileRows)); `tile_num` is TileNum.
// ---------------------------------------------------------------------------

Status copy_cdf(uint16_t* dst, const uint16_t* src, int n) noexcept;
Status average_cdf(uint16_t* cdf, const uint16_t* tile_cdf, int n,
                   uint32_t num_log2, uint32_t tile_num) noexcept;

}  // namespace av2enc::av2
