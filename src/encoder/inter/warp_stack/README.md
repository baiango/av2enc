# Warp stack — §7.11.4, §7.12.2.3/.4/.9/.11/.20, §7.12.3, §7.13.3.23

Normative warp machinery for AV2 v1.0.0, plus an independent mirror and a
`dav2d` gate that reconstructs LOCALWARP blocks exactly.

The audit (`docs/spec-audit/06-motion.md`) lists these processes as **ABSENT**:
there is no warp context, no corner-derived warp model, no warp candidate
stack, and no §7.12.3 sample set anywhere in `src/`.  This component adds them.
It owns only `src/encoder/inter/warp_stack/`; nothing in `m2/`, `inter/advanced/`
or the build system is edited.

The pixel side is **not** reimplemented: prediction uses
`adv::warp_predict()` / `adv::warp_predict_ext()` from
`src/encoder/inter/advanced/`.  What this component adds is all the *state* that
decides which model is used and which `warp_idx`/`is_warp` symbols are coded.

## Files

```text
src/encoder/inter/warp_stack/
  warp_stack.h/.cpp          normative machinery + §7.12.2 stack builder
  warp_stack_mirror.h/.cpp   independent vector-based re-derivation (tests only)
  warp_stack_io.h            canonical stack snapshot format
  warp_stack_io.cpp          writer
  warp_stack_reader.cpp      hand-written reader (independent of the writer)
  warp_stack_test.cpp        goldens, corner cases, randomized lockstep
  gate/warp_gate_tile.h/.cpp minimal single-reference tile writer (gate only)
  gate/warp_gate.cpp         stream assembly, reconstruction, dav2d comparison
  build_and_test.sh          unit tests + gate
  README.md                  this file
```

## API

### §7.11.4 scan point warp context

```cpp
WarpContext compute_warp_context(const WarpContextParams&);
void scan_point_warp_context(params, delta_row, delta_col, WarpContext*);
```

`WarpContext` is exactly `WarpMvCount`, `WarpSampleFound[2]` and
`ExtraWarpSample` (`ExtendDeltaRow`/`ExtendDeltaCol`).  The four §7.11.2 scan
points are run in order, including the top-superblock-row even-column
alignment (`deltaCol -= MiCol & 1`).  `WarpMvCount` is the `TileIsWarpCdf` row;
`ExtendDelta*` is the §7.13.3.24 extended-warp seed.

### §7.12.2.3/.4 generate points from corners

```cpp
CornerResult generate_points_from_corners(grid, sb_h4, mi_row, mi_col,
                                          bw4, bh4, ref_frame0, iter, stack);
bool warp_corner_process(..., delta_row, delta_col, adjust_col, CornerResult*);
inter::mc::Mv get_warp_motion_vector_xy_pos(...);
```

Implements `CornersFound`, `allMvsSame`, the negative-`refPts` bail-out, the
`(refPts[1][1]-refPts[0][1]) >> widthLog2` model, `reduce_warp_model` from the
advanced component, the translation clamp, and the two-call pattern
(`iter = 0`, then `iter = 1` when `NumWarpFound == 0 && bw4 <= 16`).  The
corner probe uses the spec's even-column rule
`(MiCol - (MiCol&1)) + (deltaCol - (deltaCol&1))`, which is *not* the §7.11.4
rule; both are transcribed and tested separately.  `reduce_warp_model` is the
normative 1/64-grid reduction; there is no separate forced-ROTZOOM step in
§7.12.2.3.  (The brief's "reduce-to-rotzoom" phrasing is the global-motion
degradation of §5.18.9, which is a different process; the corner model is a
general affine fit reduced to the signalled grid.)

### §7.12.2.9/.11/.20 warp stack

```cpp
Status build_warp_stack(const WarpStackParams&, WarpStack*);
void insert_warp_candidate(const adv::WarpParams&, WarpStack*);
void add_warp_motion_vector(grid, mi_row, mi_col, ref0, derive_wrl,
                            mv_row, mv_col, WarpStack*);
```

`WarpStack` contains `RefStackMv`/weights/offsets/`RefStackCwp`, `num_nearest`,
`WarpParamStack[4]` and `NumWarpFound`.  Step order:

1. identities (init);
2. §7.12.2.3 corner model (`iter 0`, then `iter 1`);
3. scan points 6–13 and the §7.12.2.5 column scan;
4. `numNearest`, optional §7.12.2.19 one move-to-front;
5. step 21: pre-clamp, global candidate, MVP combinations
   (`>32 && >32` luma), then the warp tail:
   `WarpBankParams[ref]` newest-first → `gm_params[ref]` → two
   `Default_Warp_Params`;
6. step 22 clamp.

`add_warp_motion_vector` is called inside the scan point **before** the
`NumMvFound >= MAX_REF_MV_STACK_SIZE` bail-out, as §7.12.2.6 requires, so a full
MV stack cannot suppress the warp stack.

`build_warp_stack` is single-reference by design (§5.20.7.6 defines
`DeriveWrl = !skip_mode && !isCompound && ...`, so warp candidates never exist
for a compound block).  `is_compound`, `use_intrabc`, and
`use_ref_frame_mvs`/`use_temporal_first` are rejected with
`kUnsupportedCompound`/`kUnsupportedIntrabc`/`kUnsupportedTemporal`.

### §7.12.3 find warp samples + §7.13.3.23 model

```cpp
bool find_warp_samples(const WarpSampleParams&, WarpSampleSet*);
void add_warp_sample(const WarpSampleParams&, delta_row, delta_col, ...);
bool above_sample_stored(const WarpSampleParams&, delta_col);
std::optional<LocalWarpResult> derive_local_warp(const LocalWarpParams&, int ref);
adv::WarpParams estimate_warp_from_samples(const WarpSampleSet&, mid_x, mid_y,
                                           const int block_mv[2]);
```

`WarpSampleSet::cand[i]` is the spec's `CandList[ref][i][0..3]`
(`src_y, src_x, dst_y, dst_x`, 1/8 luma samples), capped at
`LEAST_SQUARES_SAMPLES_MAX = 8`.  Availability, `doTopLeft`/`doTopRight`,
`above_sample_stored`'s SB-border even-column rule and the two-list add order
are transcribed.  Samples read `Mvs` (see the SubMvs note below).

`estimate_warp_from_samples` is a direct transcription of the §7.13.3.23
normal equations with a **64-bit** determinant and a 64-bit
`resolve_divisor`.  It deliberately does not call `adv::warp_estimate()`: that
helper narrows the determinant to `int`, and for a 64x64 block the determinant
exceeds `2^31`.  For the gate's four-sample set the component's helper returns
`p2 = 32832` where the spec/decoder value is `65920`; that difference breaks
decoder parity.  The pixel prediction still uses the advanced component.

### §7.13.3.19 SubMvs storage

```cpp
inter::mc::Mv warp_sub_block_mv(const adv::WarpParams&, int x, int y,
                                int i8, int j8);
```

For a LOCALWARP block the decoder stores one projected MV per **8x8 section**
in `SubMvs` (all four 4x4 cells of the section), while `Mvs` at the block origin
keeps the block MV.  The MV stack, the non-warp corner path and
compound-add-derived read `SubMvs`; §7.12.3 reads `Mvs`.  `WarpMi` therefore has
both fields, and the gate fills `sub_mv` from the derived model.

## Desync hazards

These are the audit's two hazards plus one found while making the gate exact.
All three still parse; the reconstruction or the context differs.

### 1. `WarpMvCount` selects the `TileIsWarpCdf` row

`is_warp` is coded with `TileIsWarpCdf[WarpMvCount]`.  `WarpMvCount` counts
matching scan *points* (not distinct blocks) whose `MotionModes` is
`>= LOCALWARP`, after the top-SB-row `deltaCol -= MiCol & 1` alignment.  If the
encoder omits the alignment, an odd-column scan point lands on a different
neighbour and the row changes; the same `is_warp` bit is then decoded with a
different probability.  The gate proves the context is right because it uses
`compute_warp_context()` for the row; the unit test
`test_context_sb_even_column` pins the aligned and un-aligned lookups.

A wrong row does not necessarily change the first `is_warp` value (the
arithmetic coder may still land in the encoded interval), which is why this is
a *latent* desync rather than a parse error.  In the gate stream the third
LOCALWARP block is the first block with `WarpMvCount != 0` (it sees two warp
neighbours, count 4); every other row-0 block matches while that one diverges
if the context is wrong.

### 2. `WarpParamStack` order

`warp_idx` is an index into a four-entry array initialised to
`Default_Warp_Params`.  §7.12.2.20 appends in this order:

```text
WarpBankParams[ref]   newest → oldest   (§7.12.2.20 warp bank, ring buffer)
gm_params[ref]                         (identity when global motion is off)
Default_Warp_Params
Default_Warp_Params
```

`insert_warp_candidate` caps at `MAX_WARP_REF_CANDIDATES = 4` and does **not**
deduplicate, so reordering the four insertions (for example defaults first)
changes what `warp_idx = k` means while the bits still parse.  `WarpParamBank`
is a per-reference ring: `record()` overwrites the oldest entry and advances
`start`.  `test_warp_stack_tail_order` pins the exact order and asserts a
defaults-first stack produces a different snapshot.

### 3. `SubMvs` vs `Mvs` for a warped neighbour

`warp_sub_block_mv` is the third hazard.  A LOCALWARP neighbour stores the
warp-projected MV **per 8x8 section** in `SubMvs`; the block MV lives in `Mvs`
at the block origin.  The MV stack and the non-warp corner path read `SubMvs`
at the scanned cell, while §7.12.3 reads `Mvs`.  If the encoder replicates the
block MV into every cell (the natural thing to do for a translational block),
the stack candidate for a block that touches a warped neighbour is wrong.  In
the gate this moved the final MV of the last block from `(12,-1)` to `(8,0)` and
produced 3,735 sample mismatches while the stream still parsed.
`test_sub_mvs_feed_the_stack` pins both sides, including the negative control.

## Tests

`./src/encoder/inter/warp_stack/build_and_test.sh` builds and runs:

| Test | What it covers |
|---|---|
| `test_context_basic` | `WarpMvCount` counts scan points (leftA+leftB = 2), `WarpSampleFound`, `ExtendDelta*`, intra/unwritten neighbours, `ref[1] == NONE` |
| `test_context_sb_even_column` | top-SB-row even-column alignment; aligned vs un-aligned lookups differ |
| `test_corner_process` | corner geometry, warp-model corner MV, SB-border guard, 3-corner requirement |
| `test_generate_corners` | valid model (`p2/p3/p4/p5/p0/p1` pinned), `allMvsSame`, negative `refPts`, `iter = 1` |
| `test_warp_bank_and_insert` | ring order/overwrite, four-candidate cap, no dedup |
| `test_warp_stack_tail_order` | §7.12.2.20 order + defaults-first negative control |
| `test_add_warp_motion_vector` | `IsInters`/`DeriveWrl`/`MotionModes`/ref guards |
| `test_find_warp_samples` | above/left sample geometry, no-availability top-left probe, left-only |
| `test_above_sample_stored` | SB-border even-column and 1-MI-wide rejection |
| `test_find_warp_samples_cap` | 8-sample cap, both lists, distinct references |
| `test_local_warp_models` | non-degenerate fit and `det == 0` translation fallback |
| `test_build_rejects_unsupported` | compound / intrabc / temporal / size / ref validation, primary and mirror |
| `test_sub_block_mv` | §7.13.3.19 projection, translation, scale, clip |
| `test_sub_mvs_feed_the_stack` | stack uses `SubMvs`, §7.12.3 uses `Mvs`, negative control |
| `test_warp_estimate_golden` | pinned 64-bit determinant models (including `det > 2^31`) |
| `test_sort_spec_exact` | §7.12.2.19 is one move-to-front (`{4,2,3,1}`, not `{4,3,2,1}`) |
| `test_wired_stack_parity` | randomized grids: candidates, weights, offsets, `numNearest` match `m2::merge::build_stack_checked` (single-ref, `derive_wrl = 0`, reorder disabled, no global-MV substitution — the subset the wired stack implements).  Excludes one query shape where the wired stack deviates from §7.12.2.5 (next row). |
| `test_wired_scan_col_deviation` | pins the spec's conditional `scan_col` adjustment (`bw4 == 1` only) and shows `merge.cpp` scans an extra column for odd `MiCol` with `bw4 != 1` |
| `test_global_mv_substitution` | §7.12.2.13 `get_mv()` global branch |
| `test_corner_then_scan_order` | corner model precedes §7.12.2.9 scan models |
| `test_lockstep_randomized` | 600 randomized grids/stacks + sample sets: primary vs independent mirror snapshot equality |
| `test_snapshot_roundtrip` | writer → independent reader → writer; malformed input rejected |
| `test_goldens` | pinned full stack snapshots |

Goldens are literal expected values in `warp_stack_test.cpp` (stack snapshots,
`wmmat`, `SubMvs`, sample lists and models), not generated at test time.
`--emit-goldens` prints the two snapshot goldens for maintenance.

The wired-stack comparison also found one deviation: `merge.cpp`'s `scan_col()`
adds `MiCol & 1` unconditionally, while §7.12.2.5 does so only when
`Num_4x4_Blocks_Wide[MiSize] == 1`.  For an odd `MiCol` and a wider block the
wired stack scans one column to the right.  `test_wired_scan_col_deviation`
pins both behaviours; the parity loop excludes that query shape so the rest of
the spatial subset can still be compared candidate-for-candidate.

Randomized lockstep uses a fixed `std::mt19937` seed, so a failure is
reproducible.  The mirror is a separate translation unit with a
collect-then-fold implementation; it shares only the public types.

## Gate

`gate/warp_gate.cpp` writes a 192x128 two-frame stream:

```text
frame 0  OBU_CLOSED_LOOP_KEY  produced by the unmodified M2 core
frame 1  OBU_REGULAR_TILE_GROUP, NumTotalRefs = 1, six 64x64 leaves:
  A (0,0) NEWMV (0,0)       D (1,1) WARP_NEWMV + LOCALWARP (3 samples)
  B (0,1) NEWMV (0,8)       E (0,2) WARP_NEWMV + LOCALWARP (1 sample, det=0)
  C (1,0) NEWMV (8,0)       F (1,2) WARP_NEWMV + LOCALWARP (D+E, above-right
                            corner is itself a LOCALWARP block)
```

The sequence header enables `LOCALWARP` and `DELTAWARP` (the latter is required
to code `allow_warpmv_mode`); the frame header enables both and sets
`allow_warpmv_mode = 1`.  `is_warp` uses `compute_warp_context()`, and
`use_local_warp` uses the §8.3.2 NPos context.  The tile writer resolves each
block's MV from `build_warp_stack` and the decoder's `assign_mv` rule, then
stores the §7.13.3.19 `SubMvs`.

Reconstruction calls `derive_local_warp()` (encoder side), asserts it equals the
model the tile writer stored, and predicts with `adv::warp_predict`.  The gate
then runs

```sh
dav2d --strict 1 --threads 1 -q -i warp_gate.ivf -o warp_gate_decoded.y4m
```

and requires every Y/U/V sample of frame 1 to match.  Current result:

```text
warp_gate: 192x128, tile 11 bytes, 3 LOCALWARP blocks
warp_gate: LOCALWARP reconstruction exact (36864 luma + chroma samples)
```

`build_and_test.sh` then runs a **negative control**: `--negative-context`
forces the `TileIsWarpCdf` row to 0 for every block (the same stream otherwise).
The decoder no longer produces a P frame:

```text
warp_gate: negative control dropped the P frame
```

That is the §7.11.4 hazard made observable: the third LOCALWARP block is the
only one with a non-zero `WarpMvCount` (4), so forcing row 0 desynchronises the
arithmetic decoder even though every earlier block is unchanged.

The gate exits 77 when no `dav2d` is available; the unit tests still pass.

## Integration sites

Nothing below is edited by this component; these are the splice points an M2
integration needs.

1. **Mode context** — `m2_syntax.cpp:1461` (`new_mv_context`) should call
   `compute_warp_context()` and expose `WarpMvCount`, `WarpSampleFound[2]` and
   `ExtendDeltaRow/Col`.  `is_warp` then uses
   `adv::cdf::kIsWarpCdf[WarpMvCount]`.  The current M2 writer omits the warp
   scan entirely; that is only safe while `DeriveWrl` is 0.
2. **Stack builder** — `merge.cpp:130` (`StackBuilder::run`) is the place for
   steps 3/4 (after the global MV), 9 (inside `scan_point`, before the
   full-stack check) and the §7.12.2.20 warp tail (after MVP combinations).
   `merge::StackParams` needs the `WarpParamStack`/`gm_params[ref]`/warp bank
   inputs; `build_warp_stack()` can be used as the reference or ported into
   `StackBuilder`.
3. **SubMvs** — `m2_frame.cpp:852` and `m2_syntax.cpp:1744` (`update_mv_grid`)
   replicate the block MV into every cell.  For a LOCALWARP block they must
   instead store `warp_sub_block_mv(model, x, y, i8, j8)` per 8x8 section.
4. **Local warp model** — at the predict site, `derive_local_warp()` supplies
   `LocalWarpParams[0]`; `adv::warp_predict`/`warp_predict_ext` already consume
   `WarpParams`.  The model must be derived before `SubMvs` storage and stored in
   `WarpParams[row][col][0]` for later corner scans.
5. **Headers** — flip `seq_enabled_motion_modes[LOCALWARP]`, write
   `seq_frame_motion_modes_present_flag`, keep `enable_six_param_warp_delta`
   consistent with `DELTAWARP`, add the per-mode frame bits and
   `allow_warpmv_mode` (`m2_headers.cpp:150`, `:275`, `:294`).  The gate's
   `write_sequence_header_warp`/`write_frame_header_warp` are exact copies with
   those bits and can be used as the reference.
6. **Prediction** — the M2 predictor must call
   `warp_predict`/`warp_predict_ext` for `motion_mode >= LOCALWARP` instead of
   `predict_plane`; `warp_needs_extended()` already implements the
   shear/scale/size fallback.

## Verification

```sh
./src/encoder/inter/warp_stack/build_and_test.sh            # 7,488 checks + gate
WERROR=1 ./src/encoder/inter/warp_stack/build_and_test.sh
SANITIZE=1 ./src/encoder/inter/warp_stack/build_and_test.sh --no-gate
DAV2D=tools/bin/dav2d-macos ./src/encoder/inter/warp_stack/build_and_test.sh
```

The gate result above was produced against the pinned
`tools/bin/dav2d-macos` (dav2d `f9dafbc`) on macOS.  Linux uses
`tools/bin/dav2d` with no source change.
