# CUDA AV2 §8.2 entropy port

CUDA implementation of the normative AV2 v1.0.0 §8.2 symbol entropy coder.
One workgroup per tile, one active coding lane per tile. The device coder is
byte-identical to the CPU reference in `src/entropy/av2/`, which remains the
correctness oracle; that module is reused read-only and was not modified.

Status: **byte-exact and passing** on the primary target (RTX 3060, `sm_86`,
CUDA 12.8). `989,905` checks, `0` failures on the benchmark run below.

| File | Purpose |
|---|---|
| `entropy.h` | POD tape/descriptor/result types, device `TileEncoder`/`TileDecoder`, kernel declarations |
| `entropy.cu` | Encode/decode kernels and host launch helpers |
| `entropy_test.cu` | CPU-vs-GPU correctness suite, corruption tests, benchmarks |
| `build_entropy_test.sh` | `nvcc` build (default `-arch=sm_86 -lineinfo`) and run |
| `README-entropy.md` | this file |

```sh
./src/cuda/build_entropy_test.sh
./src/cuda/build_entropy_test.sh --no-bench
ARCH=sm_120 ./src/cuda/build_entropy_test.sh    # secondary build/smoke target
WERROR=1 ./src/cuda/build_entropy_test.sh
```

No Blackwell-only features are used. The build needs only `nvcc` and the
existing CPU reference sources.

## API

```cpp
using namespace av2enc::cuda;

// One entry per coded operation.  12 bytes, trivially copyable.
struct EntropyTapeEntry {
  uint8_t op;        // EntropyOp::kSymbol / kBool / kLiteral / kEnd
  uint8_t n;         // alphabet size 1..8, or literal width 0..32
  uint16_t cdf_id;   // kSymbol: index into the tile's CDF set
  uint16_t symbol;   // kSymbol: value; kBool: bit
  uint16_t reserved;
  uint32_t literal;  // kLiteral: value
};

// Caller-owned device arrays, one per tile.
struct EntropyEncodeDesc {
  const EntropyTapeEntry* tape;  uint32_t tape_entries; uint32_t flags;
  uint16_t* cdfs;                uint32_t cdf_count;     uint32_t cdf_entries;
  uint8_t* out;                  uint32_t out_capacity;
  uint32_t reserved0, reserved1, reserved2;
};

struct EntropyDecodeDesc {
  const uint8_t* in;             uint32_t payload_bytes;  // normative sz
  uint32_t backing_bytes;        // allocation size; sz > backing = truncated
  EntropyTapeEntry* tape;        // ops in, decoded values written in place
  uint32_t tape_entries;         uint32_t flags;
  uint16_t* cdfs;                uint32_t cdf_count;      uint32_t cdf_entries;
  uint32_t reserved0, reserved1;
};

struct EntropyTileResult {
  uint32_t status;             // av2enc::av2::Status; kOk == 0
  uint32_t payload_bytes;      // encode: exit_symbol() size; decode: declared sz
  uint64_t symbol_count;
  uint64_t bool_count;
  uint64_t tape_entries_used;
  uint64_t cycles;             // clock64() delta of the serial lane
};

cudaError_t launch_entropy_encode(const EntropyEncodeDesc* d_tiles,
                                  EntropyTileResult* d_results,
                                  uint32_t tile_count,
                                  cudaStream_t stream = nullptr);
cudaError_t launch_entropy_decode(const EntropyDecodeDesc* d_tiles,
                                  EntropyTileResult* d_results,
                                  uint32_t tile_count,
                                  cudaStream_t stream = nullptr);
```

Launch geometry is fixed: grid = `tile_count`, block = `kEntropyBlockThreads`
(32). Lane 0 runs the serial coder; lanes 1..31 only help with the block-level
CDF staging copy before and after the serial loop. The launch helpers do not
allocate, do not synchronize, and return the launch status; copy descriptors and
results in/out yourself.

### Flags

| Flag | Effect |
|---|---|
| *(none)* | **Frozen CDFs** — the M0 default; pinned MD5s are unchanged |
| `kEntropyEnableCdfUpdate` | Opt-in §8.2.6 adaptive CDFs |
| `kEntropyDisableCdfUpdate` | Explicit freeze (redundant with the default) |
| `kEntropyDisableStaging` | Write bits directly to `out`; test/diagnostic mode |

Adaptive CDFs are **not** the default. The CPU reference's `Encoder` field
defaults to adaptation on; the CUDA API deliberately inverts that policy to
match the project rule ("enable adaptive CDFs by default" is a confirmed
do-not-repeat). The tests cover both modes.

### Tape semantics

- `kSymbol`: `encode_symbol(cdf, n, symbol)` / `decode_symbol(cdf, n, &symbol)`.
- `kBool`: `encode_bool` / `read_bool`; `symbol` is the bit.
- `kLiteral`: encode as `n` × `encode_bool` (the CPU encoder has no
  `encode_literal`, so `bool_count` increases, not `symbol_count`); decode as
  `read_literal(n)` (which adds `n` to `symbol_count`, matching the CPU
  decoder). Widths 0..32.
- `kEnd`: stop; remaining entries are ignored.
- Decode writes `kSymbol`/`kBool` results to `symbol` and `kLiteral` results to
  `literal` **in place**. Use a mutable copy of the op tape.

`cdfs` is one independent mutable copy per tile (`cdf_count` arrays of
`cdf_entries` words). The coder updates it in place when adaptation is enabled;
the caller can inspect the final CDFs after the launch. A CDF must not be shared
between tiles.

## Byte-exactness

The device encoder reuses `av2::cdf_is_valid`, `av2::compute_boundaries`,
`av2::adapt_cdf`, `av2::floor_log2`, and the generated §9.2 tables unchanged.
It mirrors `av2::Encoder`'s low/range/carry/renormalisation, zero-width guard,
and §8.2.4 termination bit-for-bit. The only representation difference is where
bits live while a tile is coded:

- a 4 KiB shared-memory staging window per tile (`kEntropyStageBytes`);
- flushes at window-full boundaries and once at `exit_symbol()`, writing whole
  bytes to the caller's output buffer;
- a carry that reaches an already-flushed byte does a read-modify-write on that
  byte, preserving the exact prefix-carry semantics;
- nothing is written past the returned `sz` byte.

The decoder mirrors `av2::Decoder` including `SymbolMaxBits` clipping, the
poisoned-state statuses, `read_bool`'s unchanged range, the repeated-boundary
skip, and the §8.2.4 trailing/padding checks. A one-byte cached payload reader
avoids a global load per bit without changing the bits read.

### Zero-probability CDF entries

The CPU coder accepts equal adjacent inverse-CDF entries (a zero-probability
symbol in the model) and the §8.2.6 decoder skips a repeated boundary. The CUDA
port supports the same case:

- the exact table `{27718, 32641, 32641, 90, 32}` (cdf[1] == cdf[2]) is tested
  frozen and adaptive, with every symbol including the zero-probability symbol
  2 present in the stream;
- `Prob_Inc` still gives that symbol a positive range interval, so it is
  encodable; the encoder's `width == 0` rejection is mirrored as a defensive
  guard with no state change;
- the CPU regression case that drives `cdf[0] == cdf[1]` after 500 symbol-0
  updates is reproduced through the GPU tape path, and the decoder reaches the
  same final CDF;
- the final CDFs (including the equality) match the CPU side byte-for-byte.

## Test coverage

`entropy_test.cu` compares every GPU case against `av2::Encoder`/`av2::Decoder`:

- pinned CPU golden vectors: `mv_joint`/`delta_q`/`binary`/`n3`, adaptive and
  static, plus the bool-only `9d31a80000` and interleaved `3a77858ac88000`;
- random symbol streams for alphabets 2..8, rates `{0,5,56,124}`, counts
  `{0,5,16,31,32}`, adaptation on/off, uniform and random CDFs (560 cases);
- mixed symbol/bool/literal tapes, all literal widths 0..32, `n == 1`;
- empty streams (`80 00`, one-byte `80`), single-symbol streams, zero-length
  decode, exit-tail / padding variants;
- 4×2 (8 tiles) and 8×2 (16 tiles), independent CDF copies and tapes, with and
  without staging, plus a batch-vs-solo tile-independence check;
- declared-`sz` truncation sweep, `sz > backing`, 200 random single-bit flips,
  64 random garbage buffers;
- output-buffer exhaustion at capacities `0, 1, 2, sz/2, sz-1, sz, sz+1`;
- invalid CDFs, alphabet size 9, out-of-range symbols, unknown tape ops,
  out-of-range `cdf_id`, literal width 33, null tape;
- CDF staging limit (250 CDFs = 2500 entries → global fallback; 204 CDFs =
  2040 entries → staged);
- the 2048-symbol skewed benchmark stream as a regression.

Every successful encode also checks that the GPU did not write past `sz` (the
tail stays at the 0xCD fill), and every case compares final CDFs and coder
counters.

## Measured performance

RTX 3060, 28 SMs, CUDA 12.8, `sm_86`, `-O2 -lineinfo`, 2048-symbol tile with
`kEntropyEnableCdfUpdate` (fresh CDF per launch, one CDF per tile, payload
~650 B). Numbers are the minimum of 6–50 individually timed launches.

| Tiles | Launch (ms) | Per-tile cycles (median) | Aggregate tiles/s |
|---:|---:|---:|---:|
| 1 | 4.57 | 8.86 M | 219 |
| 4 | 4.57 | 8.85 M | 876 |
| 8 (4×2) | 4.55 | 8.81 M | 1,758 |
| 16 (8×2) | 4.57 | 8.85 M | 3,500 |
| 32 | 4.56 | 8.83 M | 7,011 |
| 64 | 4.56 | 8.83 M | 14,022 |
| 256 | 5.12 | 9.44 M | 50,000 |

- Per-tile latency: **~4.57 ms per 2048-symbol tile** (~8.86 M cycles,
  ~4.3 k cycles/symbol); decode is ~4.21 ms for the same 16-tile batch.
- Occupancy: encode **11 blocks/SM** (11 active warps, **22.9 %** of 48 warps);
  decode 16 blocks/SM (33.3 %). Each block uses 8 KiB static shared memory.
- 1..16 tiles all take one wave on 28 SMs, so the 8×2 frame wall time is ~the
  single-tile latency (~4.57 ms). 256 tiles fit in one occupancy wave
  (28 × 11 = 308 blocks) and reach ~50 k tiles/s aggregate.
- Staging vs direct global bit writes: 8.86 M vs 9.36 M cycles (1.06×). The
  benefit is small for a 650-byte payload because per-symbol work dominates; it
  grows with payload size and still removes per-bit global stores.
- Adaptive vs frozen CDFs: 8.85 M vs 7.94 M cycles (1.11×). The frozen default
  is faster and is what the M0 bitstream uses.

### Serial-tail risk

A tile is strictly serial: one active coding lane, one warp per block. 4×2 and
8×2 give 8 and 16 blocks on 28 SMs — fewer than one warp per SM. The entropy
stage therefore **cannot be shorter than one tile's serial latency** (~4.57 ms
for 2048 symbols), no matter how many SMs are available. Tiles beyond the SM
count only queue in later waves.

For context, the CPU reference encodes the same 2048-symbol tile in
**~100–135 µs** (≈50–66 ns/symbol) on this shared box — roughly **34–46×
faster per tile** than one GPU tile. For a 16-tile 8×2 frame the CPU needs
~1.6–2.2 ms serially vs the GPU's ~4.57 ms in one wave. The GPU port is
byte-exact and keeps the work on the device, but it is **not a throughput win
over the CPU entropy coder at M0 tile counts**. Treat the entropy stage as a
small serial tail to overlap with the next frame's compute, or batch
independent frames (the 256-tile aggregate is ~50 k tiles/s).

This is the dominant risk for the CUDA entropy plan. Mitigations implemented:

- shared-memory output staging and whole-byte flushes, never per-bit global
  stores;
- cooperative (32-lane) CDF staging and 4 KiB output zeroing, so the serial lane
  does not pay for them;
- the shared CDF array is passed directly into the tape runner; a
  runtime-selected generic pointer made every CDF access a generic load and was
  **2× slower** — keep the staged/global paths separate;
- default frozen CDFs; the adaptive path is opt-in.

Further work would need a register-cached CDF model and alphabet-size
specialization (templated `n`) to cut the per-symbol instruction/latency chain.
The current priority is exactness and integration, not micro-optimisation.

## Limits and integration notes

- **CDF staging cap:** `kEntropyCdfStageEntries` = 2048 `uint16_t` entries
  (4 KiB). A tile whose CDF set fits is coded from shared memory and copied
  back; a larger set is coded directly from global memory (correct, slower).
  The full §9.3 default set (~46.7 KiB) does not fit the 48 KiB baseline; use
  per-context subsets or global CDFs, or raise the carveout deliberately.
- **Output capacity:** arithmetic coding cannot know `sz` in advance. Provide a
  conservative per-tile capacity; exhaustion returns `kBufferTooSmall` and
  poisons the encoder, exactly like the CPU reference. No device allocation and
  no silent truncation.
- **Decoder input:** `payload_bytes` is the normative `sz` from the syntax
  layer; `backing_bytes` is the allocation size. `sz > backing` returns
  `kTruncatedInput`.
- **CDF lifetime:** the kernel mutates the caller's per-tile CDF copies. Reset
  them before each independent encode/decode (the benchmark does this with
  `reset_cdfs()`); reusing an adapted copy as the next frame's initial model is
  a syntax-layer decision.
- **No globals / no allocations:** the only device globals are the read-only
  `__constant__` §9.2 tables emitted by `av2_entropy_tables.h` in nvcc's device
  pass. The launch path contains no `cudaMalloc`/`cudaFree`, no hidden
  synchronization, and no exceptions.
- **One lane per tile:** do not add intra-tile parallelism to the range coder;
  tiles are independent, symbols within a tile are not.
- This module does not write the OBU/tile framing or choose tile layouts. It
  encodes/decodes exactly the tape and `sz` the caller supplies.

## Validation environment

- RTX 3060 12 GB, cc 8.6, 28 SMs, driver/CUDA 12.8 (V12.8.93).
- `nvcc -std=c++20 -O2 -lineinfo -arch=sm_86`, warnings
  `-Wall -Wextra -Wconversion -Wshadow`, `WERROR=1` clean.
  `-Wpedantic` is omitted for the CUDA build because nvcc's host stub uses
  GNU-style line markers; the CPU reference build keeps it.
- `ncu` counters are blocked in the container; timings use `clock64()` and
  CUDA events. Nsight Systems 2025.1.3 is available for overlap analysis.
