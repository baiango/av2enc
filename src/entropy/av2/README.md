# entropy/av2 — normative AV2 §8.2 symbol coder

From-scratch implementation of the AV2 v1.0.0 entropy coder.  The **decoder is
the normative object**; the encoder is its exact inverse.  No encoder source
from AVM, SVT, clave, av2-turbo, or any other codec was consulted or copied.

- Spec: `spec/AV2_Spec_v1.0.0.pdf`, v1.0.0, 2026-05-28, commit `38f28e7`
  (sha256 `e9916f091e4e83446aad6b4601641c5b292e569c144c4163b26a4497573b533f`).
- Constant tables: generated from the PDF by `generate_tables.py`; the output
  is checked in as `av2_entropy_tables.h`.
- CPU code is the **correctness reference only**.  The target is one CUDA
  workgroup per tile on the RTX 3060; see "CUDA port" below.

## Files

| File | Purpose |
|---|---|
| `av2_entropy.h` | POD encoder/decoder state, `AV2_ENTROPY_HD` hot-path state machine |
| `av2_entropy.cpp` | `status_name`, §8.2.4 `copy_cdf`/`average_cdf` host helpers |
| `av2_entropy_tables.h` | generated `Prob_Inc[7][8]`, `Para_Adjustment_List[125][3]` |
| `generate_tables.py` | PDF → table generator (checks the pinned PDF sha256) |
| `av2_entropy_test.cpp` | correctness suite |
| `build_and_test.sh` | g++ C++20 build, warnings, optional sanitizer |

```sh
./src/entropy/av2/build_and_test.sh
SANITIZE=1 WERROR=1 ./src/entropy/av2/build_and_test.sh
CXX=g++-13 ./src/entropy/av2/build_and_test.sh
```

There is no platform-specific code.  `build_and_test.sh` is a plain g++ build;
the sanitizer option uses `-fsanitize=address,undefined`.

## API

The coder has no mutable globals and never allocates.  The caller owns the
byte span and one independent CDF copy per tile.

```cpp
#include "av2_entropy.h"
using namespace av2enc::av2;

// N+1 entries: N-1 inverse-CDF values, base rate, use count.
uint16_t cdf_enc[9] = { /* ... */ };
uint16_t cdf_dec[9] = { /* identical initial copy */ };
uint8_t payload[4096];

Encoder enc;
enc.init_symbol(payload, sizeof payload);   // caller-owned tile buffer
enc.encode_symbol(cdf_enc, /*n=*/8, symbol);  // updates cdf_enc in place
size_t sz = 0;
enc.exit_symbol(&sz);                       // writes marker + padding

Decoder dec;
dec.init_symbol(payload, sizeof payload, sz);  // sz from the bitstream syntax
dec.decode_symbol(cdf_dec, 8, &symbol);        // independent CDF copy
dec.exit_symbol();                             // verifies trailing bits
```

`Encoder`/`Decoder` are trivially copyable, standard-layout PODs.  All hot
methods are `AV2_ENTROPY_HD` (`__host__ __device__` under nvcc) and
`noexcept`; `std::span` overloads exist only for host code.  A CUDA translation
unit can include the same header and call the same methods: in nvcc's device
pass the generated tables become `static __device__ __constant__`, while the
host pass keeps `inline constexpr`.  A dummy `nvcc -arch=sm_86 -c` kernel that
calls init/encode/decode/read_bool/read_literal/exit compiles cleanly; no
kernel is implemented in this module.

`init_symbol` on the encoder takes an output span rather than `sz`: the payload
length is an *output* of arithmetic coding (`exit_symbol` returns it).  On the
decoder it is the normative `init_symbol(sz)`.

`set disable_cdf_update = 1` on both sides to freeze CDFs.  `n == 1` is a
non-normative convenience alphabet: it consumes no bits and performs no CDF
update.  The specification invokes `read_symbol` only for `N > 1`.

## Normative state machine

All state is per tile.

### §8.2.2 `init_symbol(sz)`

```
numBits     = Min(8*sz, 15)
buf         = f(numBits)                  // MSB-first
SymbolValue = ((1 << 15) - 1) ^ (buf << (15 - numBits))
SymbolRange = 1 << 15
SymbolMaxBits = 8*sz - 15
```

`SymbolMaxBits` counts bits still available.  When it is negative, the decoder
has consumed `-SymbolMaxBits` implicit zero padding bits (which appear inverted
in `SymbolValue`).  It may go negative during `read_symbol`/`read_bool`.

### §8.2.6 `read_symbol(cdf)` / `encode_symbol`

`cdf` is the inverse CDF:

```
cdf[0..N-2]  strictly increasing values in [0, 32767]
cdf[N-1]     constant base adaptation rate, index into Para_Adjustment_List
cdf[N]       use count, 0..32
```

Boundaries (`EC_PROB_SHIFT = 7`):

```
prev = SymbolRange
for s = 0..N-1:
    f   = (s == N-1) ? 0 : (1 << 15) - cdf[s]
    pp  = ((f >> 7) << 4) + Prob_Inc[N-2][s]
    cur = (((SymbolRange >> 8) * pp) >> 7) << 3
    if cur >= prev: invalid CDF
    boundary[s] = cur
    prev = cur
```

Decode selects the first `s` with `SymbolValue >= boundary[s]`, then:

```
newRange = prev - cur          // prev is boundary[s-1], or SymbolRange for s=0
newValue = SymbolValue - cur
bits     = 15 - FloorLog2(newRange)          // 0..15
numBits  = Clip3(0, bits, SymbolMaxBits)
newData  = f(numBits)
SymbolRange = newRange << bits
SymbolValue = (newValue << bits) |
              ((newData << (bits - numBits)) ^ ((1 << bits) - 1))
SymbolMaxBits -= bits
```

The encoder uses the same boundaries.  It adds the interval's lower bound to
its low register, sets its range to `newRange`, and emits the `bits` bits that
the decoder will read.  The implementation keeps a 15-bit low register plus a
fixed output prefix and handles a final carry into the prefix; it does not use
a 64-bit `low` and has no output allocation.

### CDF adaptation (§8.2.6)

Applied after a symbol on **both** sides unless `disable_cdf_update`:

```
timeInterval = cdf[N] > 31 ? 2 : cdf[N] > 15 ? 1 : 0
rate = 3 + timeInterval + Min(FloorLog2(N), 2)
       + Para_Adjustment_List[cdf[N-1]][timeInterval]

for i = 0 .. N-2:
    cdf[i] = (i < symbol) ? cdf[i] - (cdf[i] >> rate)
                          : cdf[i] + (((1 << 15) - cdf[i]) >> rate)

cdf[N] += (cdf[N] < 32)
```

The shift is integer floor, not rounding.  The encoder and decoder must start
from identical CDF copies and update in lockstep.

### §8.2.3 `read_bool` (and encoder inverse)

```
cur    = SymbolRange >> 1
symbol = SymbolValue < cur
if (!symbol) SymbolValue -= cur
numBits = Clip3(0, 1, SymbolMaxBits)
newData = f(numBits)
SymbolValue = (SymbolValue << 1) | (newData ^ 1)
SymbolMaxBits -= 1
// SymbolRange is unchanged
```

**Spec ambiguity resolved here.**  The published §8.2.3 step list does not
assign `SymbolRange`, unlike §8.2.6.  This implementation leaves it unchanged.
That is consistent because `SymbolRange` is always a multiple of 8 (initial
32768; §8.2.6 boundaries are multiples of 8 and `newRange << bits` preserves
the factor), hence even.  With even `SymbolRange`, `cur = SymbolRange/2` and
both branches have width `cur`; shifting the value by one and appending the next
bit maps either branch back to a full `SymbolRange` interval.  `read_bool` is
therefore a raw equiprobable bit that consumes one stream bit and does not
change the range.  The encoder emits bits with bit 1 = lower `SymbolValue` half
(x upper half), bit 0 = upper `SymbolValue` half, matching the decoder's
`symbol = SymbolValue < cur`.

`read_literal(n)` calls `read_bool` n times and adds n to `FrameSymbolCount`.
`bool_count` in the state struct is diagnostic and is not `FrameSymbolCount`.

### §8.2.4 `exit_symbol()` and encoder termination

Decoder checks, in order:

```
require SymbolMaxBits >= -14
trailing = get_position() - Min(15, SymbolMaxBits + 15)
padding_end = get_position() + Max(0, SymbolMaxBits)
require padding_end is byte aligned
require bit[trailing] == 1
require bit[x] == 0 for trailing < x < padding_end
```

`padding_end` is the end of the declared `sz` bytes (an implementation also
checks this explicitly).  `trailing` is not necessarily 15 bits before the end
of the payload: when the decoder consumed padding, it moves.

The encoder termination is the inverse choice.  After all symbols the interval
is `[A, A+range)` at the current precision, with `range >= 2^15`.  `exit_symbol`
chooses the unique `C` in that interval with `C mod 2^15 == 0x4000` (a `1`
followed by fourteen zeros), emits its bits, and pads the final byte with
zeros.  This always exists because the interval width is at least `2^15`.
`marker_bit_pos` records where the `1` was written.  Consequences:

- A stream with no symbols is `80 00` (16 bits).  A one-byte `80` is also a
  conformant empty stream: `SymbolMaxBits` becomes `-7`, and `trailing` is bit
  0.
- The last 15 *information* bits are `1` + fourteen zeros.  If the encoder
  needs `p` pad bits (`0..7`), the decoder's `trailing` is `P-15`, not
  `end-15`; the tests pin both cases.
- `SymbolMaxBits == -14` is accepted; `-15` is rejected.

## Invariants

1. `SymbolRange` is a multiple of 8 and lies in `[32768, 65535]` after every
   symbol and after `read_bool`; `SymbolValue < SymbolRange`.
2. `SymbolMaxBits = 8*sz - 15 - (number of renormalisation bits requested)`
   before clipping.  When non-negative it equals `payload_bits - bit_pos`.
3. `bit_pos` counts bits actually read from the payload and never exceeds
   `8*sz`.  Reads are clipped to `SymbolMaxBits`; missing bits are implicit
   zeros and appear as ones after the `^ mask` inversion.
4. The encoder `range` is in `[32768, 65535]`, `low < 32768`, and the final
   code lies in `[prefix*2^15 + low, prefix*2^15 + low + range)`.
5. Encoder and decoder CDF arrays are identical before each symbol and after
   every update.  CDF values stay strictly increasing; `cdf[N-1]` is untouched;
   `cdf[N]` is capped at 32.
6. No tile reads or writes another tile's state or CDF array.  There are no
   globals and no allocations in the hot path.
7. `exit_symbol` is valid once.  A failed decoder/encoder is poisoned
   (`kEncoderFailed`/`kDecoderFailed`) until re-initialised.
8. `FrameSymbolCount` is per tile; the coder exposes `symbol_count` for
   diagnostics.  `avg_cdf`/`copy_cdf` are tile-group operations and are not
   called from the per-symbol loop.

## CDF validity and table provenance

`cdf_is_valid` accepts `N` in `[1, 8]`.  `N > 8` would index past
`Prob_Inc[7][8]`; the specification's §9.3 default arrays have at most 8
symbols, and the `ns` descriptor says "at most 8 entries".  It requires:

- `cdf[0..N-2]` strictly increasing and `< 32768`;
- `cdf[N-1] < 125` (the Para table has 125 rows);
- `cdf[N] <= 32`;
- all computed intervals strictly positive.

All 4,715 default CDF rows in `spec/tables/default_cdf.json` satisfy these
conditions, and were checked over random adaptation sequences while developing
this module.  `Prob_Inc` and `Para_Adjustment_List` were extracted from §9.2 of
the PDF and byte-compared against the official v1.0.0 attachment headers.

## What is verified

`av2_entropy_test.cpp` currently covers:

- round trips for every symbol count N = 1..8, with adaptation on and off,
  uniform and spec-derived CDFs, every symbol value, and multiple rates/counts;
- every CDF `timeInterval` threshold (0/15/16/31/32), the `Min(FloorLog2(N),2)`
  term, exact `Para_Adjustment_List` rates, and the count cap;
- interleaved `read_symbol`/`read_bool`/`read_literal` streams, including all
  literal widths 0..32, with an independent A/P golden vector;
- `exit_symbol` marker and zero-padding checks, `SymbolMaxBits == -14`, the
  `-15` rejection, no-symbol streams, extra zero bytes, and both padding cases;
- truncated backing buffers (`kTruncatedInput`), every shorter declared `sz`
  under ASan/UBSan, 200 random bit flips, and random garbage buffers;
- encoder buffer exhaustion and the poisoned-state behaviour;
- §8.2.4 `copy_cdf`/`average_cdf` arithmetic;
- eight independent 4×2-style tiles encoded/decoded with separate state and
  CDF copies;
- a 200,000-symbol stream to catch interval/CDF drift;
- POD/standard-layout/trivially-copyable static assertions for CUDA.

The pinned byte vectors are **regression vectors generated by this state
machine**, not official conformance vectors.  The specification publishes no
entropy golden vectors.

## Not verified / TODO

- **No external conformance yet.**  This module has not been wrapped in an AV2
  tile payload and decoded by `dav2d` or `avmdec`.  The next gate is a minimal
  M0 tile with one symbol set, then the AVM/`dav2d` black-box decode.  A stream
  can round-trip against this implementation and still disagree with the
  specification on an unexercised convention.
- The byte/OBU/tile framing, `sz` derivation, tile-group `avg_cdf` selection
  (`enable_avg_cdf`, `avg_cdf_type`, `context_update_tile_id`), and §8.3 CDF
  selection are outside this module.  The caller owns `Tile*Cdf` copies and
  the `copyCdf`/`avgCdf` decision from §8.2.4.
- Only a subset of the §9.3 default CDF tables is embedded in tests.  The
  generated `spec/tables/default_cdf.json` remains the source for the syntax
  layer.
- A dummy `nvcc -arch=sm_86 -c` translation unit including this header and
  calling the full API compiles.  No CUDA kernel has been written or benchmarked
  here; the kernel mapping above is a design only.
- The encoder's termination choice is not specified by AV2 (the spec defines
  only the decoder).  Any final code in the last interval that satisfies the
  §8.2.4 marker/zero rules is valid; this implementation pins one choice.
- `read_bool`'s missing `SymbolRange` assignment is the main editorial
  ambiguity.  The resolution above is internally consistent and matches the
  even-range invariant, but should be rechecked against the next spec revision.

## CUDA port plan

Target: Linux, CUDA 12.8, `sm_86` (RTX 3060, 28 SMs), one CUDA **workgroup
(block)** per tile.  A tile is serial; tiles are independent.  Target layouts
are 4×2 (8 tiles) and 8×2 (16 tiles).  Do not add intra-tile parallelism to the
range coder.

### Mapping

- `gridDim.x = TileCols * TileRows`; `blockIdx.x` is the tile index.
- One block per tile.  Block size 32 (one warp) is enough: lane 0 runs the
  serial `Encoder`/`Decoder`; other lanes can preload the tile's CDFs and token
  tape.  No `__syncthreads()` inside the per-symbol loop.  Use barriers only
  before/after the tile.
- Coder state lives in registers of the coding lane: `Encoder`/`Decoder` are
  56/64 bytes.  If a future design shares state across lanes, copy the POD into
  shared memory; do not make it a global.
- Per-tile output lives in **global memory** at a caller-provided offset.
  `Encoder::init_symbol(uint8_t*, capacity)` writes directly there; no
  `cudaMalloc`, no device-side allocation, no host round trip.
- `Prob_Inc` and `Para_Adjustment_List` are emitted as
  `static __device__ __constant__` arrays in nvcc's device pass
  (`inline constexpr` on the host), so the serial lane gets constant-memory
  broadcast loads without a separate table file.
- Tile CDFs: one independent copy per tile.  The full §9.3 default set is
  23,917 `uint16_t` values = 47,834 bytes (~46.7 KiB) per tile.  That is tight
  against the 48 KiB default shared-memory budget once alignment and other
  shared state are included; M0 should keep only the CDF arrays its syntax
  actually uses in shared memory, with the rest in per-tile global memory, and
  use the opt-in carveout (`cudaFuncSetAttribute`) if the full set is needed.
  Never share a CDF array between tiles.
- §8.2.4 averaging is a tile-group operation.  Run it on the CPU or in a
  separate small kernel after all tile blocks complete; it is not in the
  per-symbol path.

### Symbol feed

The architecture puts tile symbol generation before the entropy stage.  For
CUDA, the symbol generator should write a compact per-tile tape in global
memory: `(cdf_id, symbol)` entries for `read_symbol`, plus literal/bool entries.
The entropy block walks the tape serially.  A CDF-id table maps to the tile's
shared/global CDF copy.  The alternative — re-running syntax inside the entropy
block — makes the serial lane do mode-decision work and is not recommended.

### Output capacity

Arithmetic coding cannot know the final byte count before encoding.  Preallocate
a conservative per-tile capacity (level/tile-size bound, or a measured worst
case plus margin) and expose `kBufferTooSmall` as a host-visible error flag.
Do not fall back to `cudaMalloc` in the kernel and do not silently truncate.
For the first M0 streams, a two-pass encode (measure, then emit) is acceptable
on the CPU reference; the GPU path should use a preallocated arena.

### Top CUDA risk

**Serialization/occupancy, not arithmetic.**  A tile has exactly one serial
dependency chain, so a block has one active coding lane.  4×2 gives 8 blocks,
8×2 gives 16; that is at most 16 active warps for a 28-SM GPU (<1 warp/SM).
The entropy stage will be latency-bound on global bit writes, and adding SMs
does not help.  Mitigations: batch multiple frames before the entropy launch,
use a persistent tile queue, or accept the entropy stage as a small serial tail
while other stages saturate the GPU.  The second risk is global-memory bit
writing: the current writer does bit-at-a-time writes and a final carry
propagation pass over the emitted prefix; the CUDA port should buffer bits in a
32/64-bit accumulator and flush bytes/words, with an explicit carry/cache
scheme, rather than doing one store per bit to global memory.

The 8×2 tile layout was measured at roughly +5% IVF bytes versus 1×1 on one
720p frame (see `docs/architecture.md`); tile boundaries are a real rate cost,
so do not add tiles beyond what the occupancy argument requires.
