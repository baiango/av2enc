# entropy/ — adaptive multi-symbol range coder

Standalone arithmetic-coding foundation for the AV2 encoder.  It implements a
32-bit range coder with byte renormalization and an adaptive multi-symbol CDF
(1–16 symbols).  It contains **no AV2 syntax** and is **not** the normative AV2
symbol decoder; AV2 v1.0.0 §8.2 specifies a bit-oriented 15-bit
`SymbolRange`/`SymbolValue` coder with inverse CDFs.  This module is a generic,
well-tested foundation on which an AV2-shaped model layer and a normative coder
can be developed.  The exact delta is in "What must change for AV2" below.

All code here was written from the standard range-coding/arithmetic-coding
algorithm; no encoder source was consulted or copied.

## Files

| File | Purpose |
|---|---|
| `range_coder.h` | Public API: `RangeCoderStatus`, `AdaptiveCdf`, `Encoder`, `Decoder` |
| `range_coder.cpp` | Coder, carry handling, CDF validation and adaptation |
| `entropy_test.cpp` | Standalone test binary |
| `build_and_test.sh` | `g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic` build + run |

```sh
./src/entropy/build_and_test.sh
# On macOS /usr/bin/g++ is clang; a real GCC can be selected explicitly:
CXX=g++-16 ./src/entropy/build_and_test.sh
```

## Design choice

* **32-bit range, 8-bit renormalization.** The range register is `uint32_t`;
  whenever it falls below `2^24`, one byte is emitted and the range is shifted
  left by 8.  Output is therefore a byte string, which is what AV2 tile data
  needs.
* **15-bit probability precision** (`kProbTotal = 32768`).  AV2's normative
  CDFs are also 15-bit (§8.2.6 uses `1 << 15`), so the scale matches, although
  AV2 stores inverse CDFs with per-CDF rate/count metadata (see the AV2 delta
  section).  The 15-bit total keeps the arithmetic comfortable: after
  renormalization the per-unit range is at least `2^(24-15) = 512`, so a symbol
  with frequency 1 still receives a non-empty interval.
* **Adaptation is deliberately separate from the coder.** The coder only
  consumes a CDF.  `AdaptiveCdf` owns the distribution and its update rule, so
  a normative AV2 model can replace it without touching the interval
  arithmetic.

No global state: every encoder, decoder and model is an object.

## Algorithm

### Encoding one symbol

Given an ascending CDF `cdf[0..n]`, `cdf[0] = 0`, `cdf[n] = 32768`, and symbol
`s`:

```
unit       = range / 32768
low       += unit * cdf[s]
range      = unit * (cdf[s+1] - cdf[s])
while range < 2^24:
    emit one byte with carry propagation
    range <<= 8
```

`low` is held in 64 bits.  Its bit 32 is the only possible carry out of the
32-bit value; `low < 2^33` is an invariant.  Bytes are emitted through a
one-byte cache plus a count of pending `0xFF` bytes:

* A pending byte can only be changed by a future carry if it is `0xFF`.
* Non-`0xFF` bytes are committed immediately.
* When a carry arrives, it is added to every pending byte.  The cache byte
  (which is at most `0xFE`) absorbs it; pending `0xFF` bytes wrap to `0x00`,
  reproducing binary carry propagation.
* The first cache byte is a zero guard.  A carry can turn it into 1; it gives
  the final low room to overflow without touching an already-emitted byte.

### Finishing

`Encoder::finish()` calls the byte-shift routine exactly five times.  Those
five shifts flush the 32-bit final `low` plus the guard/carry byte, and commit
any `0xFF` bytes that were held back for carry propagation.  Consequently the
number of bytes appended by `finish()` is `4 + cache_size` (where `cache_size`
is 1 plus the number of pending `0xFF` bytes; usually 5), and the total stream
length is always

```
L = 5 + R
```

where `R` is the number of renormalization shifts performed while coding.  An
empty symbol sequence therefore produces exactly five zero bytes.  `finish()`
must be called once; a second call returns `kEncoderFinished`.

### Decoding

`Decoder::init()` requires at least five bytes and reads them into a 32-bit
code register, discarding the first (guard) byte:

```
range = 0xFFFFFFFF
code  = B1<<24 | B2<<16 | B3<<8 | B4
```

For each symbol:

```
unit  = range / 32768
value = code / unit
if value >= 32768: value = 32767          // top-of-range clamp
find s with cdf[s] <= value < cdf[s+1]
code  -= unit * cdf[s]
range  = unit * (cdf[s+1] - cdf[s])
while range < 2^24:
    if no input byte left: return kTruncatedInput
    code = (code << 8) | next_byte
    range <<= 8
```

The decoder performs no unchecked pointer arithmetic.  Any input, including
random bytes, can only produce a valid symbol or an error; it cannot read out
of bounds or loop forever.

### Single-symbol alphabet

`n == 1` carries zero information.  `encode()` consumes no range and emits
nothing for the symbol; `decode()` returns 0 without reading.  The only bytes
in such a stream are the five flush bytes (`00 00 00 00 00`).

## Adaptive model

`AdaptiveCdf` stores **ascending cumulative frequencies in [0, 32768]**:

```
cdf[0] = 0
cdf[n] = 32768
cdf[i] < cdf[i+1]          (every symbol has frequency >= 1)
freq[i] = cdf[i+1] - cdf[i]
```

### Initialization

* `uniform(n, rate)` — reserve one unit per symbol, then distribute
  `32768 - n` units as evenly as possible with cumulative rounding.
* `from_frequencies(span, rate)` — same apportionment, proportional to the
  supplied positive frequencies.  Exact sum, no drift.
* `from_cumulative(span, rate)` — validates and copies an explicit table; this
  is the entry point for AV2's generated CDF tables.

Apportionment reserves one unit per symbol and then spreads the rest using a
running remainder (`acc += available * weight; share = acc / total; acc %=
total`).  This makes the shares sum to `available` exactly, so the final CDF is
always exactly 32768 and every symbol stays codable.

### Update

After a symbol `s` is coded (on **both** encoder and decoder side):

```
increment = max(1, 32768 >> rate)
freq'[s]  = freq[s] + increment      (other frequencies unchanged)
CDF       = apportion(freq')
```

`rate` is a shift in `[0, 15]`:

| rate | increment | rough adaptation speed |
|---:|---:|---|
| 0 | 0 | frozen (static CDF) |
| 2 | 8192 | very fast |
| 4 | 2048 | fast |
| 5 | 1024 | default |
| 8 | 128 | slow |
| 15 | 1 | very slow |

The update is applied *after* the symbol is coded, so the first symbol always
uses the initial table.  `reset()` restores the construction-time CDF (the
adaptation rate is kept).  `set_adaptation_rate(0)` freezes a model.

This is a clean, exact integer model, **not** the AV2 normative update rule.
See "What must change for AV2" below.

## Bitstream format

```
byte 0        guard/carry byte (0 or 1 for this encoder; ignored by decoder)
bytes 1..4    initial 32-bit code (big-endian byte order)
bytes 5..L-1  further bytes, consumed one per renormalization
```

* The stream has **no header, no length, no symbol count, and no checksum**.
  It is a raw entropy-coded byte sequence; the caller must know how many
  symbols to decode (or code an end-of-sequence symbol).
* The decoder consumes the stream exactly: after decoding the encoded number
  of symbols, `bytes_remaining() == 0`.  Consequently any truncation is
  detected as `kTruncatedInput` while decoding that expected count.
* `finish()` performs five byte shifts; therefore
  `L = 5 + (number of renormalization shifts during coding)`.
  An empty symbol sequence produces `L = 5`, all zero.  In the common case
  `finish()` appends five bytes; if a run of `0xFF` bytes is pending, it
  appends those plus four more, and the total length still follows `L = 5 + R`.
* The final flush completes the representation of the final `low`.  The first
  five bytes are the guard and the initial code; the two groups overlap only
  for short streams.
* Byte order is most-significant-first with respect to the arithmetic code
  value.  Symbol values never appear literally in the output.

### Golden vector

Model: `AdaptiveCdf::uniform(4, rate=5)`.

Symbols: `0, 1, 2, 3, 3, 2, 1, 0, 0, 2`

Stream (7 bytes):

```
00 1D 38 FD 67 7C 34
```

Final model after the ten updates:

```
cdf = { 0, 8667, 16370, 25064, 32768 }
```

`entropy_test.cpp::test_golden_vector` pins both values.  If the format or the
model changes intentionally, update the test and this section together.

A second pinned case exercises the pending-`0xFF` flush path: alphabet 2,
`rate=0`, symbols `1, 0, 0, ...` (one 1 then fifteen 0s) produce

```
00 7F FF C0 00 00 00
```

`finish()` appends six bytes for this stream because one `0xFF` is pending, yet
the total is still `R + 5` (here `R = 2`).  See `test_pending_ff_flush`.

## Invariants

1. `1 <= n <= 16`; `cdf` has `n+1` entries.
2. `cdf[0] == 0`, `cdf[n] == 32768`, strictly increasing.
3. `sum(freq[i]) == 32768` after every model construction and every update.
4. Before coding each symbol, `range >= 2^24`; hence `unit >= 512`.
5. During encoding `low < 2^33`; bit 32 is the only carry bit.
6. `cache_size >= 1`; the pending bytes are `cache` followed by
   `cache_size - 1` copies of `0xFF`.
7. Encoder and decoder models must be identical before each symbol, and both
   apply the same update after it.  `encode()`/`decode()` own the update so the
   application cannot accidentally update them differently.
8. `finish()` performs five byte shifts and is valid exactly once; the final
   stream has `R + 5` bytes, where `R` is the renormalization count.
9. A failed `encode()` (invalid symbol) is atomic: no coder or model state
   changes.
10. A fatal decoder error poisons the decoder; subsequent `decode()` calls
    return `kDecoderFailed` until `reset()`.
11. After decoding the encoded number of symbols, the decoder has consumed all
    `L` bytes (`bytes_remaining() == 0`).  For `n > 1`, decoding more symbols
    than were encoded eventually returns `kTruncatedInput`.

## Error handling

`RangeCoderStatus` values:

| status | meaning |
|---|---|
| `kOk` | success |
| `kInvalidArgument` | null output pointer |
| `kInvalidSymbol` | symbol outside the model's alphabet |
| `kEncoderFinished` | encode/finish after finish |
| `kDecoderNotInitialized` | decode before init |
| `kDecoderAlreadyInitialized` | init twice |
| `kDecoderFailed` | decoder already hit a fatal error |
| `kTruncatedInput` | init or renormalization ran out of bytes |

Model construction reports malformed alphabets/CDFs/frequencies by returning
`std::nullopt`; `set_adaptation_rate()` returns `false` for a rate above 15.

**Truncation is detectable.** A valid stream is consumed exactly: after
decoding the encoded number of symbols, `bytes_remaining() == 0`.  Therefore
removing any byte causes `kTruncatedInput` while decoding that expected count
and the successfully decoded prefix still matches.  This is tested
exhaustively for every cut of a 600-symbol stream.

**Corruption is not inherently detectable.** An arithmetic-coded payload has no
redundancy; flipping bytes can produce a different symbol sequence that is
still a valid stream.  The leading guard byte is ignored by the decoder, so
corrupting it has no effect on the decoded symbols.  A real container (AV2
OBU/tile length, or a test harness) must supply the byte count and, if desired,
a checksum.  The corruption tests therefore treat a bit flip as "detected"
when the coder errors **or** the decoded symbols no longer match.  The coder
itself never crashes or reads out of bounds on arbitrary bytes.

The coder does not throw.  The only possible exception source is `std::vector`
allocation inside the encoder (`std::bad_alloc`), consistent with the project's
use of the standard library.

## Tests

`entropy_test.cpp` covers:

* round trips for alphabets 2–16, lengths 0–4096, rates 0/2/4/5/8/12;
* initialization from frequency and cumulative tables;
* skewed sources (`p = 0.5, 0.9, 0.99, 0.999`) and constant sources;
* one-symbol alphabet;
* a 1,000,000-symbol sequence (about 0.25 MB output, ~0.2 s locally);
* model invariants over 200,000 random updates per configuration;
* truncation at every possible cut of a 600-symbol stream (and sampled cuts of
  a 5,000-symbol stream), including decoder poisoning;
* bit flips, random garbage, invalid models, invalid symbols, API misuse;
* encoder/decoder CDF equality after every round trip;
* a pending-`0xFF` flush path with a pinned byte vector;
* a golden byte vector.

Sanitizer build (manual, not part of the script):

```sh
g++ -std=c++20 -O1 -g -fsanitize=address,undefined \
    -fno-omit-frame-pointer -I src/entropy \
    src/entropy/range_coder.cpp src/entropy/entropy_test.cpp -o /tmp/entropy_asan
/tmp/entropy_asan
```

## What must change for AV2

The AV2 v1.0.0 specification is in `spec/AV2_Spec_v1.0.0.pdf`.  The relevant
clauses are §8.1 (bit parsing, MSB-first), §8.2.1–§8.2.6 (Symbol decoder),
§8.3.1–§8.3.2 (CDF selection and context derivation), plus the
`init_coeff_cdfs` / `init_non_coeff_cdfs` tables and the `Prob_Inc` /
`Para_Adjustment_List` constant tables.  This module is a **generic** range
coder, not the normative AV2 symbol decoder.  The differences are real, not
cosmetic:

| aspect | this foundation | AV2 v1.0.0 §8.2 |
|---|---|---|
| state | 32-bit `range`, 64-bit `low` | 15-bit `SymbolRange`, 15-bit `SymbolValue` |
| init | read 5 bytes, drop a guard byte | `init_symbol(sz)`: read `Min(8*sz,15)` bits, `SymbolValue = ((1<<15)-1) ^ paddedBuf`, `SymbolRange = 1<<15`, track `SymbolMaxBits = 8*sz - 15` |
| renormalization | one byte when `range < 2^24` | `bits = 15 - FloorLog2(newRange)`; read `bits` bits, `SymbolRange = newRange << bits`, invert the padded data (`paddedData ^ mask`) |
| CDF | ascending cumulative, total 32768 | inverse CDF; array length `N+1`; entry `N-1` is the constant base rate, entry `N` is a use count capped at 32 |
| adaptation | `freq[s] += max(1, 32768>>rate)`, exact renormalize | `timeInterval = cdf[N]>31 ? 2 : cdf[N]>15 ? 1 : 0`; `rate = 3 + timeInterval + Min(FloorLog2(N),2) + Para_Adjustment_List[cdf[N-1]][timeInterval]`; shift CDF boundaries; `cdf[N] += (cdf[N] < 32)` |
| binary symbols | a 2-symbol `AdaptiveCdf` | dedicated `read_bool()` half-range path (§8.2.3) |
| trailing bits | none | `exit_symbol()` verifies a `1` bit followed by zeros, consumes OBU trailing bits, requires `SymbolMaxBits >= -14` (§8.2.4) |
| alphabet | 1..16 | `read_symbol` requires `N > 1`; the `ns` descriptor describes "at most 8 entries" |

Consequences:

1. **Replace the coder core for a conformant encoder.** The normative byte
   format is defined by the §8.2 decoder, which is bit-oriented and keeps a
   15-bit code value.  A 32-bit `low`/`range` coder with byte flush cannot
   produce an AV2-decodable tile payload even if the CDF math were perfect.
   The class boundaries (`Encoder`, `Decoder`, `AdaptiveCdf`) and the test
   methodology can be reused; the state machine cannot.
2. **Replace the CDF model.** Implement the `N+1`-entry AV2 layout: inverse
   cumulative values in entries `0..N-2`, the constant base rate in `N-1`, and
   the use count in `N`.  Generate `Default_*` CDF tables, `EC_PROB_SHIFT`
   (`= 7`), `Prob_Inc[7][8]`, and `Para_Adjustment_List` with a script; never
   hand-type them.  The update timing, shift direction and exact integer
   rounding are bitstream-defining.
3. **Coefficient contexts.** AV2 has a large set of generated coefficient CDF
   arrays (`TxbSkipCdf`, `EobPt*Cdf`, `CoeffBase*Cdf`, `CoeffBr*Cdf`, sign and
   IDTX tables) and picks them from the `coeffs()` context.  `init_coeff_cdfs`
   selects one of four quantization contexts from `base_q_idx`; the coefficient
   CDF selection itself is in §8.3.2 and the `coeffs()` syntax.  Context
   derivation belongs in the syntax layer, with per-tile copies (`Tile*Cdf`),
   save/average on tile-group exit (`avg_cdf`), and load/blend across frames
   (`load_cdfs` / `blend_cdfs`).  None of that is present here.
4. **Bit I/O and framing.** §8.1 reads bits MSB-first.  `init_symbol` reads a
   bit window, `SymbolMaxBits` may go negative while zero padding is consumed,
   and `exit_symbol` enforces the trailing-bit convention and byte alignment.
   The generic encoder here has no bit writer, no `sz` parameter, no
   `SymbolMaxBits`, and no OBU/tile framing.
5. **What is reusable.** The model API shape, exact-CDF validation, test
   harness (round trips, skew, long sequences, truncation/corruption safety,
   model synchronization) and the documentation of invariants carry over.  The
   `AdaptiveCdf` class can be reimplemented internally in AV2's inverse layout
   while exposing a normalized probability view to a new coder; the current
   ascending representation is not that view.

The specification defines the **decoder**.  An AV2 encoder must invert §8.2 and
choose a final value inside the last interval that satisfies `exit_symbol`'s
trailing `1`-bit and zero-padding requirements.  That termination choice is not
provided by this module.

### Top risk

**The top risk is assuming AV2 uses the same arithmetic state machine as this
generic range coder.**  AV2 v1.0.0 §8.2 is a 15-bit, bit-oriented
`SymbolRange`/`SymbolValue` coder with an inverted initial value, variable-bit
renormalization, `SymbolMaxBits` padding, a dedicated `read_bool`, and a
trailing `1`-bit terminator.  Round-trip tests against ourselves prove none of
those conventions.  A stream can be perfectly self-consistent and still be
rejected by an AV2 decoder at the first `init_symbol` or a few symbols later.

The second risk is the exact CDF metadata and integer adaptation: the inverse
CDF ordering, the per-CDF base rate and use count, `Prob_Inc`,
`Para_Adjustment_List`, the `Min(FloorLog2(N),2)` term, and the update rounding
all change every subsequent interval.  Both risks must be retired with
spec-derived conformance vectors and a black-box AV2 decoder — not with this
suite alone.
