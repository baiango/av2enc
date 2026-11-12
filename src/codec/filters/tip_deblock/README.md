# TIP deblocking filter — CPU reference (AV2 §7.16)

Reference implementation of the **Deblocking filter for TIP process**,
AV2 Bitstream & Decoding Process Specification v1.0.0 (2026-05-28, commit
`38f28e7`), pages 528–529, plus the shared §7.17.3 / §7.17.7 kernels it invokes.

This component is **standalone**: it does not write bitstream syntax, touch the
root CMake build, `src/main.cpp`, M1/M2, or CUDA. The audit entry is
`docs/spec-audit/10-filters.md` §7.16 (three ABSENT rows).

## Where it runs relative to §7.17 (read this first)

§7.16 and §7.17 are **alternatives, never a pipeline**. Per §7.2
`decode_frame_wrapup` (spec pages 376–377):

```text
if (TipFrameMode == TIP_FRAME_AS_OUTPUT) {
  if (apply_deblocking_filter_tip)   §7.16          // this component
  LrFrame = CurrFrame
  reference_frame_update()                          // no §7.17/§7.18/§7.19/§7.20
} else if (bru_inactive) {
  LrFrame = copy(FrameStore[ref_frame_idx[bru_ref]])
} else if (!ShowExistingFrame) {
  if (apply_deblocking_filter[0] || apply_deblocking_filter[1]) §7.17
  §7.18 CDEF
  §7.19 CCSO
  §7.20 LR
}
```

Consequences:

* A **TIP_FRAME_AS_OUTPUT** frame runs §7.16 exactly once (if the flag is set)
  and then goes straight to reference/output. Do **not** run §7.17 on it and do
  **not** run CDEF/CCSO/LR: `frame_filters_on[i]` is forced to 0 for this
  branch, and the §7.16 output is what later frames predict from.
* A **TIP_FRAME_AS_REF** frame (TIP blocks inside an ordinary inter frame) is a
  normal inter frame. §7.16 does not run; §7.17 → §7.18 → §7.19 → §7.20 run as
  usual.
* SEF / bridge / BRU-inactive frames run neither.
* M1 is intra-only and can never reach this process.

Call-site rule: **branch on `TipFrameMode` before the §7.17 call**. The API
encodes this: `tip_deblock_frame()` returns `kNotApplicable` (no writes) for a
frame that is not `TIP_FRAME_AS_OUTPUT` with the flag on.

## §5.18.2 flag interaction (`enable_df_sub_pu`)

`apply_deblocking_filter_tip` is signalled only for `TIP_FRAME_AS_OUTPUT` and
only after `allow_df_sub_pu`, which itself exists only when the sequence flag
`enable_df_sub_pu` is 1:

```text
if (TipFrameMode == TIP_FRAME_AS_OUTPUT) {
  if (enable_df_sub_pu) allow_df_sub_pu              f(1)
  if (allow_df_sub_pu)  apply_deblocking_filter_tip  f(1)
  else                  apply_deblocking_filter_tip = 0
}
```

So enabling §7.16 in a stream requires **all three**:

1. sequence `enable_df_sub_pu = 1`,
2. TIP frame `allow_df_sub_pu = 1`,
3. TIP frame `apply_deblocking_filter_tip = 1`.

§7.16 does **not** use sub-PU edges: `allow_df_sub_pu` is a syntax
precondition, not a filter input. The component rejects
`apply_deblocking_filter_tip == 1` without (1) and (2) with
`kUnsupportedFlagCombination`, because that bit cannot be coded.

`tip_deblock_flag_signallable()` answers “is the bit codable?” and
`tip_deblock_is_active()` answers “must the filter run?”.

### TIP frame output flags

Per §5.4.6/§5.18.2:

```text
EnableTipOutput = enable_tip && !disable_tip_output
if (enable_tip && use_ref_frame_mvs && NumTotalRefs >= 2 && !bru_inactive) {
  if (EnableTipOutput && is_tip_frame())
    TipFrameMode = TIP_FRAME_AS_OUTPUT      // no tip_frame_mode bit is coded
  else
    TipFrameMode = tip_frame_mode           // 0 disabled, 1 as-reference
}
```

Only `TIP_FRAME_AS_OUTPUT` can run §7.16. `TIP_FRAME_AS_REF` is an ordinary
inter frame (TIP blocks, then §7.17), and `TIP_FRAME_DISABLED` never reaches
this process. `immediate_output_frame`, `ShowExistingFrame` and
`disable_tip_output` do not change the filter itself; they only decide whether
a TIP frame is built/output, and SEF frames do not run §7.16.

The caller resolves these bits and passes `TipFrameMode`; the component does
not parse syntax.

## Files

```text
src/encoder/filters/tip_deblock/
  tip_deblock_core.h        normative arithmetic + traversal (shared kernels)
  tip_deblock.h             public CPU API, ordering / flag contract
  tip_deblock.cpp           validation + entry points
  tip_deblock_test_common.h synthetic 4:2:0 frames, write-envelope helper
  tip_deblock_test.cpp      5,636 checks
  gate/tip_deblock_gate.cpp synthetic grid gate + dav2d gate seam
  build_and_test.sh         standalone build/run (--asan supported)
  README.md                 this file
```

The pixel kernels are the same inline functions used by §7.17
(`filter_max_width`, `filter_choice`, `sample_filtering`, `get_q`, `round2`,
`clip3`, `kSideThresholds`, `kWMult`, `kQThreshMults`, `kQFirst`) from
`encoder/filters/deblock/deblock_core.h`.

## API

```cpp
namespace av2enc::tip_deblock {

enum class TipSize : int32_t { kBlock8x8 = 8, kBlock16x16 = 16 };
TipSize select_tip_size(bool enable_tip_refinemv, InterpFilter filter);
TipSize tip_deblock_tip_size(const TipDeblockOptions&);

struct TipDeblockFrame {
  TipDeblockPlane planes[3];          // 8-bit mutable planes + strides
  int mi_cols, mi_rows;               // must both be even
  int subsampling_x, subsampling_y;   // must be 1,1
  const uint16_t* tile_col_starts; int num_tile_cols;
  const uint16_t* tile_row_starts; int num_tile_rows;
};

struct TipDeblockOptions {
  TipFrameMode tip_frame_mode;        // §5.18.2
  bool apply_deblocking_filter_tip;
  bool enable_df_sub_pu, allow_df_sub_pu;
  bool disable_loopfilters_across_tiles;
  int base_q_idx, delta_q_u_ac, delta_q_v_ac, base_uv_ac_delta_q;
  int bit_depth;                      // 8 only
  bool enable_tip_refinemv;           // tipSize input
  InterpFilter tip_interp_filter;     // tipSize input
};

struct TipDeblockPlaneInfo { int base_filter_level, q_thr, side; };
struct TipDeblockEdgeInfo { /* lattice flags, widths, width, apply_filter */ };

TipDeblockPlaneInfo tip_deblock_plane_info(const TipDeblockOptions&, int plane);
TipDeblockStatus tip_deblock_frame(TipDeblockFrame&, const TipDeblockOptions&);
TipDeblockStatus tip_deblock_plane_pass(TipDeblockFrame&, const TipDeblockOptions&,
                                        int plane, int pass);
TipDeblockEdgeInfo tip_deblock_inspect_edge(const TipDeblockFrame&,
                                            const TipDeblockOptions&,
                                            int plane, int pass, int x, int y);
bool tip_deblock_flag_signallable(const TipDeblockOptions&);
bool tip_deblock_is_active(const TipDeblockOptions&);
}
```

`tip_deblock_inspect_edge()` runs the full decision path without writing
pixels. `x`/`y` are **plane** coordinates and must be a position the §7.16
loops visit (vertical: `x` a positive multiple of `sw`, `y` a multiple of 4;
horizontal: `y` a positive multiple of `sh`, `x` a multiple of 4). Otherwise it
returns a zeroed record with `on_screen == 0`.

### Quantizer inputs

For a TIP output frame, `base_q_idx`, `DeltaQUAc` and `DeltaQVAc` come from the
TIP frame header when `enable_tip_explicit_qp == 1`; otherwise §5.18.2 sets
them to `Round2(RefX[slot0] + RefX[slot1], 1)` over the closest past/future
references. `BaseUVAcDeltaQ` is the sequence-level value and is not averaged.

## Exact rules

### tipSize

```text
tipSize = (enable_tip_refinemv && TipInterpFilter == EIGHTTAP_SHARP)
              ? BLOCK_8X8 : BLOCK_16X16
```

`InterpFilter` is `av2enc::inter::mc::InterpFilter` (aliased by the temporal
component), so the eventual TIP stream agent can pass its `TipConfig` fields
directly.

### Per-plane strength (once per plane, not per edge)

```text
baseFilterLevel = base_q_idx
plane 1 (U): += DeltaQUAc + BaseUVAcDeltaQ
plane 2 (V): += DeltaQVAc + BaseUVAcDeltaQ
qThr = Round2(get_q(baseFilterLevel, 0), QUANT_TABLE_BITS) >> 6
qInd = Clip3(0, MAX_SIDE_TABLE - 1, baseFilterLevel - 24 * (BitDepth - 8))
side = Max(Side_Thresholds[qInd] + (1 << (12 - BitDepth)), 0) >> (13 - BitDepth)
```

Differences from §7.17 that are easy to get wrong:

* no `q_clamped()` special case, no segment id, no per-cell `qindex`;
* no `DfDeltaQ` term (`lvlSeg = qc + DF_DELTA_SCALE * DfDeltaQ[i]` does **not**
  apply here);
* `QUANT_TABLE_BITS` is 3 (the same shift as `deblock::kDfShift` semantics, but
  a distinct constant).

### Edge lattice and traversal

```text
for plane in {Y,U,V}:
  subX = plane ? SubsamplingX : 0;  subY = plane ? SubsamplingY : 0
  sw = Block_Width[tipSize]  >> subX       // 8/16 luma, 4/8 chroma for 4:2:0
  sh = Block_Height[tipSize] >> subY
  h  = (MiRows * MI_SIZE) >> subY
  w  = (MiCols * MI_SIZE) >> subX

  for y = 0; y < h; y += 4:               // vertical boundaries
    for x = 0; x < w; x += sw:
      if (x > 0):
        vertTileEdge = is_vert_tile_edge(x, subX)
        (pos, neg) = filter_maximum_width(plane, filterSize = sw, vertTileEdge)
        if (!disable_loopfilters_across_tiles || !vertTileEdge):
          width = filter_choice(x, y, plane, qThr, side, dx=1, dy=0,
                                neg, pos, MI_SIZE)
          if (width > 0):
            for i = 0..3: sample_filtering(x, y+i, qThr, dx=1, dy=0,
                                           Min(width,neg), Min(width,pos), 0, 0)

  for x = 0; x < w; x += 4:               // horizontal boundaries
    for y = 0; y < h; y += sh:
      if (y > 0):
        horzTileEdge = is_horz_tile_edge(y, subY)
        if (!disable_loopfilters_across_tiles || !horzTileEdge):
          horz64Edge = ((y << subY) % 64) == 0
          (pos, neg) = filter_maximum_width(plane, filterSize = sh, horz64Edge)
          width = filter_choice(x, y, plane, qThr, side, dx=0, dy=1,
                                neg, pos, MI_SIZE)
          if (width > 0):
            for i = 0..3: sample_filtering(x+i, y, qThr, dx=0, dy=1,
                                           Min(width,neg), Min(width,pos), 0, 0)
```

Exactness notes:

* `y += 4` in the vertical pass and `x += 4` in the horizontal pass are
  **literal**; they are not scaled by `SubsamplingY/X`. For 4:2:0 chroma this
  means 4 chroma rows/columns per group.
* `x == 0` / `y == 0` are the only on-screen exclusions. There is no
  `onScreen`, `sbEdge` block-edge, skip, lossless, TX-size or sub-PU logic.
* `filterSize` is `sw`/`sh` **unclipped**. §7.16 has no screen-edge
  `Min(filterSize, 16/8)` step (that belongs to §7.17.2).
* `is_vert_tile_edge(x, subX)` computes
  `col = (x << subX) >> MI_SIZE_LOG2` and tests membership in `MiColStarts`;
  `is_horz_tile_edge(y, subY)` does the same with `MiRowStarts`.
* `disable_loopfilters_across_tiles` suppresses **both** vertical and
  horizontal tile edges. Only the vertical tile edge feeds the `sbEdge` width
  clamp; the horizontal pass's `sbEdge` is the 64×64-grid edge only, so a
  horizontal tile edge does not widen the clamp.
* `filter_maximum_width` (§7.17.3): `filterSize <= 4 → 1`, `== 8 → 3`,
  `== 16 → luma 6 / chroma 4`, `> 16 → luma 8 / chroma 4`; `sbEdge` clamps
  `maxWidthNeg = Min(maxWidthPos, luma 6 / chroma 2)`.

For 4:2:0 the reachable width classes are:

| plane | tipSize 8x8 | tipSize 16x16 |
|---|---|---|
| luma | filterSize 8 → pos/neg 3 | filterSize 16 → pos/neg 6 (neg 6 when sbEdge) |
| chroma | filterSize 4 → pos/neg 1 | filterSize 8 → pos/neg 3 (neg 2 when sbEdge) |

So the `sbEdge` negative clamp is only observable on 16x16-tip chroma.

### Filter choice / sample filtering

`filter_choice` is §7.17.7.2 with `count = MI_SIZE` and the plane-coordinate
`x, y`; `sample_filtering` is §7.17.7.1 with `prevLossless = currLossless = 0`
and `Min(width, maxWidthNeg/Pos)`. Both are the shared kernels from
`deblock_core.h`.

## Tests

`./build_and_test.sh` builds with
`-Wall -Wextra -Wpedantic -Wconversion -Wshadow` and runs **5,636 checks**,
then the gate:

1. `tipSize` selection for every `enable_tip_refinemv × filter` combination.
2. Strength derivation over **all 256 `base_q_idx` values** × 3 planes × 2
   chroma-delta configurations against an independent transcription of the
   §7.16 formula, plus 13 hand-checked literals, negative levels and the
   `qThr == side == 0` cases.
3. `filter_maximum_width` for every tipSize/plane/sbEdge/tile combination,
   including the 64-grid chroma clamp and the unvisited-lattice checks.
4. **Exact filtered samples** for 8x8 and 16x16 tips on luma and chroma,
   hand-derived from §7.17.7 (e.g. 100/120 step → `103 106 109 111 114 117`
   for a 3-tap class, `102 103 105 106 108 109 111 112 114 115 117 118` for a
   6-tap class).
5. Unvisited boundaries: a step inside a TIP block is untouched.
6. Per-plane chroma deltas: U can be disabled while Y/V filter.
7. Vertical/horizontal tile edges, luma and chroma, with
   `disable_loopfilters_across_tiles` on and off.
8. The 64×64-grid `sbEdge` clamp, plus proof that a horizontal tile edge is
   not clamped.
9. The complete `(TipFrameMode, apply, enable_df_sub_pu, allow_df_sub_pu)`
   matrix: active / not-applicable / non-signallable, with pixel-unchanged
   assertions for the inactive and rejected cases.
10. Flat / alternating texture / `base_q_idx = 0` produce no changes.
11. 120 randomized frames: guard columns, determinism, and a conservative
    write-envelope invariant (no changed pixel outside
    `[edge - maxWidthNeg, edge + maxWidthPos)` on a visited edge).
12. A real reconstructed frame (`decoded.y4m`): deterministic, 734 (16x16) /
    1245 (8x8) samples changed, max |delta| 4.

Local result: **5,636 checks, 0 failures**; ASan + UBSan clean.

```sh
./src/encoder/filters/tip_deblock/build_and_test.sh
./src/encoder/filters/tip_deblock/build_and_test.sh --asan
```

## Gate and decoder seam

`build_and_test.sh` also runs `gate/tip_deblock_gate.cpp`.

**Available today — synthetic TIP-sized grid.** A 320×192 (MI 80×48)
block-structured reconstruction with discontinuities on both the 8×8 and 16×16
grids is filtered with both `tipSize` values. The gate asserts that Y/U/V all
change, that only visited TIP edges are modified, that guards stay clean, that
flat content is unchanged, and that the result is deterministic. Result:
`PASS: synthetic §7.16 TIP grid` (13,500/6,332/6,332 samples for 16×16 and
26,108/11,260/11,260 for 8×8).

**Seam — real `dav2d --strict 1 --threads 1` gate.** There is no complete TIP
stream in this tree: `src/encoder/m2/video_headers.cpp:169` writes
`enable_tip = 0`, no `OBU_REGULAR_TIP` / `TipFrameMode == TIP_FRAME_AS_OUTPUT`
writer exists, and `video_headers.cpp:174` writes `enable_df_sub_pu = 0`, so
`apply_deblocking_filter_tip` cannot be coded. A concurrent TIP stream
component (`src/encoder/inter/tip_stream/`) now provides signalling, the
§7.10.6 build (`build_tip_frame`) and output planning; its own `dav2d` gate
was still being assembled when this component landed, so no complete TIP
stream artifact was available to compare against.

A **decoder test vector with TIP OBUs is present**:
`testdata/sample_480x270_130f.ivf` (70 IVF units → 130 output frames, OBU type
14, decodes cleanly with `dav2d --strict 1 --threads 1`). It cannot close this
gate by itself: `dav2d` exposes only whole-chain `--inloopfilters` toggles, so
`none` versus `deblock` also changes every later reference frame; isolating
§7.16 needs the encoder's pre-filter TIP frame (or a decoder instrumentation
hook), which is exactly the stream agent's deliverable. The gate here is
implemented and ready for it:

```sh
tip_deblock_gate --tip-ivf stream.ivf --recon-y4m encoder_recon.y4m \
                 [--dav2d tools/bin/dav2d-macos] [--repo-root /path/to/av2enc]
```

It requires at least one TIP OBU (type 13/14), runs
`dav2d -q -i <ivf> -o <y4m> --strict 1 --threads 1`, and requires zero Y/U/V
sample differences against the encoder reconstruction. Exit codes: 0 pass,
1 fail/mismatch, 77 dav2d unavailable.

Prerequisites owned by the TIP stream agent:

1. sequence header: `enable_tip = 1`, `disable_tip_output = 0`,
   `enable_df_sub_pu = 1`;
2. TIP frame header: `TipFrameMode == TIP_FRAME_AS_OUTPUT`,
   `allow_df_sub_pu = 1`, `apply_deblocking_filter_tip = 1`;
3. build the §7.10.6 TIP frame, call `tip_deblock_frame()` exactly once, dump
   that frame as the Y4M recon, and do not run §7.17/CDEF/CCSO/LR.

## Call sites / integration notes

No files outside this directory were changed. The places an integration pass
must touch:

* `src/encoder/m2/video_headers.cpp:169` — `enable_tip` (currently hard `false`);
  §7.16 cannot be reached until a TIP output frame exists.
* `src/encoder/m2/video_headers.cpp:174` — `enable_df_sub_pu` (currently hard
  `false`); must be `true` for `apply_deblocking_filter_tip` to be codable.
* TIP frame header writer — add `allow_df_sub_pu` and
  `apply_deblocking_filter_tip` inside `TipFrameMode == TIP_FRAME_AS_OUTPUT`.
  `src/encoder/inter/tip_stream/tip_signalling.*` owns the surrounding TIP
  frame bits but stops before the quantizer/deblock block where these two
  flags live.
* `src/encoder/m2/post/loop_filters.cpp:270` (`run_video_post_filters`) — the
  natural post-filter hook. For a TIP output frame, replace step 1 (§7.17
  deblock) with `tip_deblock_frame()` and skip CDEF/CCSO/LR, because
  `frame_filters_on[]` is 0 for that frame. Do not chain both.
* `src/encoder/m2/video_frame.cpp:1027` — the next frame uses
  `result.recon.back()`. The §7.16 output must be what is pushed there (the
  current path pushes raw reconstruction and no filters).
* M1 (`src/encoder/m1/`) is intra-only; this process is never applicable.

The caller must provide the plane buffers, tile starts, and the quantizer /
flag state above. There is **no MI grid**: §7.16 ignores partition, transform,
skip, lossless, segmentation and sub-PU metadata.

## Limitations

* 8-bit only; 4:2:0 only; even MI grid required (frame dimensions padded to a
  multiple of 8). 10-bit / 4:2:2 / 4:4:4 are later workstreams.
* BRU does not matter here: a BRU-inactive frame follows the §7.2
  `bru_inactive` branch and never runs §7.16.
* No SIMD, no threading, no CUDA: this is the CPU correctness oracle. The
  spec arithmetic is integer-only and shared with the §7.17 kernels, so a
  future device port can reuse `tip_deblock_core.h` if the §7.17 core is made
  device-visible.
* The synthetic gate is not a decoder conformance claim. It proves traversal,
  strength and sample math; the dav2d gate above is the conformance seam.
