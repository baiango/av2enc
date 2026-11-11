# CDEF — CPU reference (AV2 §7.18)

Wave-2 workstream 9 (see `docs/agent-waves.md`). Spec-only implementation of
the AV2 Constrained Directional Enhancement Filter for **8-bit 4:2:0**
reconstructed frames.

**No bitstream integration.** This directory filters planes and reports
decisions. Root `CMakeLists.txt`, `src/main.cpp`, `src/encoder/m0_still.cpp`,
the entropy coder and the bitstream paths are untouched; a later integration
pass owns wiring.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`):

| Spec | Process |
|---|---|
| §5.18.7.10 | CDEF params syntax (damping, strength count, per-strength values, skip flag) |
| §6.17.7.6 | CDEF params semantics |
| §5.20.9.3 | `is_inside_filter_region` |
| §7.18 | CDEF process: 8×8 block traversal, one parameter set per 64×64 luma block |
| §7.18.1 | CDEF block process: copy, skip/lossless, direction, strength derivation |
| §7.18.2 | CDEF direction process (8 directions, variance) |
| §7.18.3 | CDEF filter process (primary/secondary taps, `constrain`, clip) |

## Files

```text
src/encoder/filters/cdef/
  cdef_core.h         shared host/device arithmetic + normative tables, the
                      §7.18.1 derivation helpers (cdef_luma_params,
                      cdef_chroma_params, cdef_block_skip_flags,
                      cdef_filter_region) and cdef_filter_sample
  cdef.h              CPU API (planes, metadata, options, inspection)
  cdef.cpp            validation, traversal, per-block decision/filter
  cdef_search.h       encoder-side strength search (header-only, host+device):
                      legal candidate ladder, per-64x64 SSE evaluation with the
                      real filter, greedy header-list selection
  cdef_test_common.h  synthetic frames, metadata builders, Y4M loader
  cdef_test.cpp       CPU tests (242 checks) + timing report
  cdef_search_test.cpp search tests (140 checks): objective == filter SSE,
                      determinism, skip/lossless branches, real 640x360
  build_and_test.sh   standalone build/run (no root CMake changes)
  README.md           this file
```

The CUDA kernel is in `src/cuda/cdef/` and calls the same `cdef_core.h`
functions, so CPU and GPU output are bit-exact.

## Algorithm

### Traversal (§7.18)

```text
for r = 0; r < MiRows; r += 2          # 8×8 luma blocks
  for c = 0; c < MiCols; c += 2
    idx = cdef_idx[r & ~15][c & ~15]   # one parameter set per 64×64
    block(r, c, idx)
```

`MiCols = 2 * ((width + 7) >> 3)` and `MiRows = 2 * ((height + 7) >> 3)` are
even, so the 8×8 luma and 4×4 chroma grids tile the MI-aligned planes exactly.

### Block process (§7.18.1)

1. Copy the 8×8 luma block and 4×4 chroma blocks from `CurrFrame` to
   `CdefFrame`.
2. If `idx == -1`, stop (copy only).
3. Derive `skip`:
   - if `cdef_on_skip_txfm_frame_enable == 0`, skip is the AND of the four MI
     `Skips` cells;
   - `skip |= LosslessArray[SegmentIds]` over the four cells.
4. If `skip`, stop. Derive `skipChroma` from `ChromaSegmentIds`; chroma is
   copied but not filtered when set.
5. Run the direction process on the **unfiltered luma** 8×8.
6. Derive luma strengths/direction/damping and filter plane 0.
7. Derive chroma strengths/direction/damping and filter planes 1 and 2.

### Strength derivation

```text
coeffShift = BitDepth - 8                         # 0 for our scope
priStr     = cdef_y_pri_strength[idx] << coeffShift
secStr     = cdef_y_sec_strength[idx] << coeffShift
dir        = (priStr == 0) ? 0 : yDir
varStr     = (var >> 6) ? min(FloorLog2(var >> 6), 12) : 0
priStr     = var ? (priStr * (4 + varStr) + 8) >> 4 : 0
damping    = CdefDamping + coeffShift
```

Chroma uses `cdef_uv_pri/sec_strength` **without** variance scaling,
`dir = Cdef_Uv_Dir[SubsamplingX][SubsamplingY][yDir]`, and
`damping = CdefDamping + coeffShift - 1`.

### Filter (§7.18.3)

For every sample `x`:

```text
sum = 0, min = max = x
for k in {0, 1}:                       # distance-1 / distance-2 taps
  for sign in {-1, +1}:
    p = get_at(dir, k, sign)
    if available:
      sum += Cdef_Pri_Taps[priStr & 1][k] * constrain(p - x, priStr, damping)
      min = Min(p, min); max = Max(p, max)
    for dirOff in {-2, +2}:
      s = get_at((dir + dirOff) & 7, k, sign)
      if available:
        sum += Cdef_Sec_Taps[priStr & 1][k] * constrain(s - x, secStr, damping)
        min = Min(s, min); max = Max(s, max)
out = Clip3(min, max, x + ((8 + sum - (sum < 0)) >> 4))
```

`Cdef_Pri_Taps = {{4,2},{3,3}}`, `Cdef_Sec_Taps = {{2,1},{2,1}}`. The tap row
is selected by the **final** `priStr` parity, as the spec specifies.
`constrain(diff, threshold, damping)` returns 0 for a zero threshold, and
otherwise `sign * Clip3(0, |diff|, threshold - (|diff| >> max(0,
damping - FloorLog2(threshold))))`.

`get_at` maps the plane coordinate back to a luma MI cell and returns
"unavailable" when that cell is outside the filter region (§5.20.9.3): the tile
bounds when `disable_loopfilters_across_tiles` is set, the whole frame
otherwise. Unavailable taps contribute neither to `sum` nor to `min`/`max`.

### Direction process (§7.18.2)

The 8×8 luma block is reduced to eight directional projections
(`partial[8][15]`), each cost is the normalized sum of squared projections
using `Div_Table = {0,840,420,280,210,168,140,120,105}`, and the direction with
the largest cost wins (ties keep the lower index). Variance is
`(bestCost - cost[(yDir + 4) & 7]) >> 10`. All costs are 64-bit integers in
both the CPU and CUDA paths; no floating point is used.

## Strength search (encoder side)

`cdef_search.h` answers "which strength set should this 64x64 block use?":

```cpp
CdefSearchFrame frame;      // src = unfiltered recon, ref = original picture
CdefSearchOptions options;  // damping, num_strengths 1..8, candidate ladder
CdefSearchResult plan;      // strengths, cdef_idx grid, per-block SSE
cdef_search_plan(frame, options, &plan, &error);
```

It evaluates every candidate set with the same `cdef_filter_sample` the decoder
path uses, so `plan.sse_total` is exactly the SSE of the samples `cdef_frame()`
produces (checked by `cdef_search_test.cpp`). Selection is greedy over the
candidate ladder with deterministic tie-breaks, and it never returns an
unsignalable index: `cdef_idx` is always in `[0, num_strengths)`.

Two rules that matter for bitstream integration:

* with `CdefStrengths == 1` the decoder codes no index symbol, so every block is
  forced to set 0 (`plan.forced_single_strength`);
* with `CdefStrengths >= 2` there is no `-1` index, so "do not filter" needs an
  all-zero strength set in the header (`CdefSearchOptions::include_off`).

The M1 adapter lives in `src/cuda/m1/cdef/`.

## API

```cpp
namespace av2enc::cdef {

struct CdefStrength { int y_pri, y_sec, uv_pri, uv_sec; };
struct CdefOptions {
  int damping;                        // CdefDamping, 3..6
  int num_strengths;                  // 1..8
  CdefStrength strengths[8];
  bool on_skip_txfm;                  // cdef_on_skip_txfm_frame_enable
  bool disable_loopfilters_across_tiles;
  int bit_depth;                      // 8 only
};

struct CdefPlane { const uint8_t* src; uint8_t* dst; int stride; };

struct CdefFrame {
  CdefPlane planes[3];
  int mi_cols, mi_rows;               // even
  int subsampling_x, subsampling_y;   // must be 1,1
  const uint8_t* skip;                // MI grid
  const uint8_t* lossless;            // luma MI grid
  const uint8_t* lossless_chroma;     // optional
  const int8_t* cdef_idx;             // per 64×64, -1 or strength index
  int n64_cols, n64_rows;             // 0 derives ceil(mi/16)
  const uint16_t* tile_col_starts; int num_tile_cols;
  const uint16_t* tile_row_starts; int num_tile_rows;
};

CdefStatus cdef_frame(const CdefFrame&, const CdefOptions&);
CdefStatus cdef_inspect_block(const CdefFrame&, const CdefOptions&,
                              int r, int c, CdefBlockInfo*);
}
```

`CdefBlockInfo` reports the block index, skip/skip-chroma flags, detected
direction and variance, derived `varStr`, final luma/chroma strengths,
directions, damping values and the filter region, without writing pixels.

`cdef_status_string()` names validation failures. `cdef_supported()` returns
false for 10-bit, non-4:2:0, odd MI grids, short strides, null pointers,
in-place (`src == dst`) planes, missing grids, out-of-range options, an
out-of-range `cdef_idx`, and invalid tile starts.

## Metadata contract

- One entry per 4×4 luma cell for `skip`, `lossless` and (optionally)
  `lossless_chroma`.
- `cdef_idx` is one entry per 64×64 luma block, row-major with `n64_cols`;
  `-1` means "copy without filtering".
- Tile starts are in luma MI units; `num_* == 0` means one implicit tile.
- Luma plane is `mi_cols*4` × `mi_rows*4`; chroma is `mi_cols*2` ×
  `mi_rows*2`; `stride` may be larger (padded) but not smaller.
- The caller owns all memory; the CPU path allocates nothing and does not
  synchronise.

## Source/destination

`src` and `dst` must be **different buffers**. CDEF is a pure function of
`CurrFrame`: every tap reads the unfiltered source, and the output `CdefFrame`
is written only. In-place filtering would read already-filtered neighbours and
is rejected.

## Tests

`./build_and_test.sh [--no-run]` builds with
`-Wall -Wextra -Wpedantic -Wconversion -Wshadow` and runs two binaries:
`cdef_test` (242 checks) and `cdef_search_test` (140 checks). The 242:

1. `constrain` / `FloorLog2` boundaries.
2. Direction search on constant, horizontal-line, vertical-line and
   ±45°-line blocks: expected `yDir` and non-zero variance; flat gives
   `yDir = 0`, `var = 0`.
3. Filter golden: an isolated +4 bump on a flat 100 background with four
   primary taps produces `101`; flat neighbourhoods are unchanged for every
   primary strength; a 1-code checker ripple moves toward the mean and stays
   in range.
4. Flat/pattern frames: guards intact and flat content byte-identical.
5. `idx = -1` copy, fully-skipped blocks, `on_skip_txfm` on/off, per-cell
   lossless (whole 8×8 copied), chroma lossless (luma filtered, chroma copied).
6. Zero strengths leave the frame unchanged; stronger primary changes at least
   as many samples; damping changes the result.
7. Tile filter regions: enabling `disable_loopfilters_across_tiles` changes
   output near the boundary, leaves samples far from it untouched, and has no
   effect without tiles.
8. Partial blocks (10×6 MI) at frame edges: deterministic, guards clean.
9. `cdef_inspect_block` agrees with `cdef_frame` for every block.
10. 60 randomized frames (sizes 16×16 … 328×100 luma-MI) with random strengths,
    damping, tiles, skip/lossless cells and `cdef_idx`: deterministic,
    guard-clean, 206,518 changed samples.
11. Real reconstructed `decoded.y4m` 640×360: deterministic, 181,145 samples
    changed, mean |delta| 1.0830, max 4.
12. Validation failures for every rejected input class.

`SANITIZE=1 ./build_and_test.sh` is clean under ASan + UBSan.

This is not a `dav2d` gate: the component produces no stream.

## Measured CPU time (correctness oracle — not a performance target)

`-O2`, 3–5 iterations, `decoded.y4m` and a synthetic 1080p frame, one strength
set, all planes:

| Resolution | Apple M-series | Vast.ai Xeon E5 (shared) |
|---|---:|---:|
| 640×360 | ~9–12 ms/frame | ~35 ms/frame |
| 1920×1080 | ~79–94 ms/frame | ~345 ms/frame |

The Xeon number varies with host load. Per `docs/architecture.md`, the CPU path
is deliberately scalar and single-threaded; the product timing is the GPU
kernel in `src/cuda/cdef/`.

## Integration notes

1. **Placement.** Run after reconstruction and deblocking, before CCSO and loop
   restoration. Normative order: reconstruct → deblock → CDEF → CCSO → LR.
   The filtered frame is the reference for later pictures.
2. **Inputs.** Build `skip`, `lossless` and `lossless_chroma` from the
   partition/segment decisions and `cdef_idx` from the per-64×64 strength
   search. For intra M1 content `skip` is normally 0; lossless comes from the
   segment.
3. **Frame level.** `CdefDamping`, `CdefStrengths` and the strength arrays come
   from §5.18.7.10; `cdef_on_skip_txfm_frame_enable` comes from the sequence
   `CdefOnSkipTxfm` mode or the frame bit.
4. **Tiles.** Pass `MiColStarts`/`MiRowStarts` and set
   `disable_loopfilters_across_tiles` from §5.4.10. Filtering is otherwise
   allowed across tile boundaries.
5. **Geometry.** Pad reconstruction planes to the MI-aligned size before
   calling. The encoder's 64-aligned buffers satisfy this.
6. **Strength search.** This component evaluates a supplied parameter set; it
   does not choose strengths. An encoder-side search can call `cdef_frame` for
   candidate sets or use `cdef_inspect_block` plus its own cost model.
7. **Not wired in.** No root build, `main`, M0 still path, entropy or bitstream
   edits.

## Limitations / deliberately out of scope

- 8-bit 4:2:0 only; 10-bit, 4:2:2 and 4:4:4 are separate workstreams.
- Even MI grid (frame padded to a multiple of 8).
- No strength/damping search or RDO; the caller supplies the parameters.
- No bitstream syntax or OBU writing.
- No SIMD, threading or CPU optimisation: this is the correctness oracle.
