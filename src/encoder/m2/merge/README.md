# M2 skip / merge / DRL

Correctness layer for AV2 inter block-mode decisions in the M2 P-frame path.
It is deliberately self-contained under `src/encoder/m2/merge/`; it does not
modify the M2 core, `src/encoder/inter/mc/`, `src/main.cpp`, or `CMakeLists.txt`.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`), §5.18.8.2, §5.20.5.9, §5.20.7.1-.13,
§6.19.7.5-.10, §7.9.4, §7.11.2-.3, §7.12.2 (spatial subset), §7.12.2.19,
§7.12.2.20, §7.12.2.23.

## Files

```text
src/encoder/m2/merge/
  merge.h             public API and spec types
  merge.cpp           skip_mode, DRL, single_mode, contexts, MV stack, audit
  merge_test.cpp      1014 checks; every block size 4..64, refs 1..4, audit
  build_and_test.sh   standalone build/run (no CMake changes)
  README.md           this file
```

## Build and test

```sh
./src/encoder/m2/merge/build_and_test.sh
SANITIZE=1 ./src/encoder/m2/merge/build_and_test.sh
CXX=clang++ ./src/encoder/m2/merge/build_and_test.sh
```

Current result (CPU, no CUDA):

```text
AV2 M2 merge / skip / DRL tests
1014 checks, 0 failures
```

ASan + UBSan clean. The test compiles `partition.cpp` and `mc.cpp` because the
audit wrapper calls the shared `inter::mc::build_mv_stack` for comparison; no
shared file is edited.

## Block-mode semantics

The three independent notions that are easy to conflate:

| Concept | Syntax | Meaning |
|---|---|---|
| `skip_flag` | §5.20.5.10, always read for an inter block | No residual. It says nothing about the MV mode. A `NEWMV` block may have `skip_flag = 1`. |
| merge / NEARMV | `single_mode = 0`, `YMode = NEARMV` | MV = `RefStackMv[RefMvIdx][0]`, no MVD. A non-zero `RefMvIdx` is signalled with `drl_mode`. |
| `skip_mode` | §5.20.5.9 / §5.20.7.6 | A *compound* merge shortcut. It is **not** `skip_flag`. `read_skip_mode()` is followed by `read_is_inter()` and `read_skip()`, so a `skip_mode` block still codes `skip_flag` and may carry residual. |
| NEWMV | `single_mode = 2`, `YMode = NEWMV` | MV = `RefStackMv[RefMvIdx][0] + MVD`; both DRL index and MVD are coded. |
| GLOBALMV | `single_mode = 1`, `YMode = GLOBALMV` | MV = `GlobalMvs[0]`; no DRL index and no MVD. |

`single_mode` decoding is the whole single-reference mode decision:

```text
single_mode S()                  // 0, 1, 2
YMode = NEARMV + single_mode     // 14, 15, 16
```

`decode_single_mode()` rejects values outside 0..2; `single_mode_has_drl()` is
true for NEARMV/NEWMV and false for GLOBALMV; `single_mode_has_mvd()` is true
only for NEWMV.

### skip_mode

`skip_mode` is enabled per frame by `skip_mode_present`, then gated per block by
`read_skip_mode()`:

```text
skip_mode = 0 if SEG_LVL_SKIP or SEG_LVL_GLOBALMV is active
skip_mode = 0 if !skip_mode_present
skip_mode = 0 if !is_comp_ref_allowed()   // Min(w,h) >= 8 or 4xN/Nx4 thin
skip_mode = 0 if RegionType == INTRA_REGION
otherwise skip_mode S()
```

When `skip_mode = 1`:

* `RefFrame[0..1] = skip_mode_frames()` (neighbour pair override, then
  `SkipModeFrame[]` fallback);
* `is_inter = 1`, `YMode = NEAR_NEARMV`, no `single_mode`, no compound-mode
  syntax, no MVD, `use_optflow = 0`;
* `find_mv_stack(isCompound = 1)`;
* one DRL index is still read (`has_second_drl()` is forced to 0 by
  `skip_mode`), and `TileSkipDrlModeCdf[Min(idx,2)]` is used;
* the two list MVs are `RefStackMv[RefMvIdx][0]` and `[1]`;
* `CwpIdx = RefStackCwp[RefMvIdx]`;
* the interpolation filter is `EIGHTTAP_SHARP`.

`derive_skip_mode_params()` implements §5.18.8.2, including the
`|curToRef0 - curToRef1| > 1` fallback and `RESTRICTED_OH` handling.
`derive_skip_mode_frames()` implements §5.20.7.10 exactly, including the
neighbour-compound override and the TIP `Min/Max(ClosestPast, ClosestFuture)`
case.

## DRL

`read_drl_idx(start, m)` is:

```text
for (idx = start; idx < m; idx++) {
  drl_mode S();
  if (drl_mode == 0) return idx;
}
return m;
```

Consequences implemented here:

* the exhaustion value `m` is **not** followed by a zero bit;
* `start > m` reads no bits and returns `m` (this can happen for
  `RefMvIdx0 + 1` on same-reference `NEAR_NEARMV`);
* `m = max_drl_bits_minus_1 + 1`, at most 5 for `MAX_REF_MV_STACK_SIZE = 6`;
* the CDF row is `Min(idx, 2)`;
* `has_second_drl()` is true only for `NEAR_NEARMV` / `NEAR_NEWMV` with
  `!skip_mode && !use_optflow`.

`encode_drl_index()` and `decode_drl_index()` round-trip every start/m/index
combination. `choose_drl_index_for_mv()` returns the first exact match in
`[start, Min(m, count-1)]`; lowest index wins deterministically.

## MV stack

`build_mv_stack()` is a spec-exact spatial implementation of §7.12.2,
including compound (two-list) blocks:

* setup global MV, steps 6-13 scan, `numNearest` snapshot, col scan at -3;
* `useSort = DrlReorder == ALWAYS || (DrlReorder == CONSTRAINT &&
  !useTemporalFirst && numNearest >= 4)`; selection sort over
  `[0, numNearest)` with strict `<` (the first maximum wins a tie);
* `DRL_REORDER_DISABLED` leaves insertion order intact;
* derived-SMVP fill (§7.12.2.16/.22) for multi-reference single blocks,
  including the `max_drl_bits_minus_1 + 2` cap and §7.9.4 projection;
* compound-add-derived (§7.12.2.14) non-temporal subset;
* global-MV insertion with `PruneCount`-limited dedup;
* §7.12.2.20 MVP-combination gate and final §7.12.2.23 clamping.

Insertion order, dedup, weights, offsets and sort tie-breaks are covered for
every valid block size with luma dimensions 4..64 (29 sizes, including
rectangular and `4xN`/`Nx4` thin sizes).

## Shared-component bug: §7.12.2.20 MVP gate

AV2 §7.12.2.20 says, before the `insert_mvp_candidate()` calls:

> If `Block_Width[ MiSize ]` is greater than 32 and `Block_Height[ MiSize ]`
> is greater than 32, extra candidates are added as follows:

`src/encoder/inter/mc/mc.cpp` inserts the six combinations unconditionally
(`// Step 21 (continued): MVP combination candidates.`). This is wrong for every
block with `width <= 32` or `height <= 32`. The M2 core currently hides it by
limiting NEARMV candidates to indices 0..1 when either dimension is `<= 32`
(`m2_frame.cpp`, `near_limit`), which is conservative and loses valid
pre-combination candidates 2..5.

The merge module does not edit `mc`. Instead:

* `merge::build_mv_stack()` applies the spec gate;
* `merge::audit_mc_stack()` calls `inter::mc::build_mv_stack()` for a
  single-reference block and compares it with the spec stack and with
  instrumented variants, reporting:
  * `mvp_combinations_not_gated`,
  * `above_left_sb_border_scan_diff`,
  * `derived_candidates_missing`,
  * `extra_mv_low_clamp`;
* `merge::build_stack_checked()` always returns the spec-exact stack and
  optionally fills an audit.

Other deviations found while auditing `mc.cpp` (report only):

1. **Step 13 scan point.** Spec: `scan_point(-1, -1 - isSbBorder)`.
   `mc.cpp` uses `scan_point(-1, -1)`. On a superblock top border with an odd
   `mi_col` the two produce different columns. `audit_mc_stack()` detects it
   through `above_left_sb_border_scan_diff`.
2. **Derived SMVP.** `mc.cpp` does not implement §7.12.2.16/.22. With one
   reference frame this is benign; with two or more references it drops
   projected candidates.
3. **Extra MV_LOW/MV_UPP clamp.** `mc.cpp` clamps stack candidates to
   `[MV_LOW+1, MV_UPP-1]` before the §5.20.9.4/.5 picture clamp. The spec's
   extra-search/clamping processes only apply the picture clamp; `SubMvs` are
   already integer-clamped by `assign_mv()`. Harmless for in-range encoder
   input, but reported by `extra_mv_low_clamp`.
4. **`useTemporalFirst` term.** `mc.cpp`'s sort predicate omits
   `!useTemporalFirst`. M2 has `enable_ref_frame_mvs = 0`, so this cannot
   differ today; it will matter when temporal MV projection is enabled.

`mc.cpp` also has no compound-stack API, so `audit_mc_stack()` marks compound
parameters `mc_supported = false`.

## Integration notes for the M2 core (not applied)

1. **Use the checked stack.** In `m2_frame.cpp`, replace `load_stack()`'s
   `inter::mc::build_mv_stack()` call (and the duplicate validation in
   `m2_syntax.cpp::write_inter_mode_info()`) with
   `merge::build_stack_checked(params, &stack, &audit)`. Map parameters:

   ```cpp
   merge::StackParams p;
   p.mi_row = mi_row;
   p.mi_col = mi_col;
   p.bw4 = w4;
   p.bh4 = h4;
   p.sb_h4 = 16;
   p.mi_cols = mi_cols_;
   p.mi_rows = mi_rows_;
   p.mi = mv_grid_.data();              // MiInfo grid
   p.ref_frame[0] = inter::mc::kRefLast; // opaque label; see step 3
   p.ref_frame[1] = merge::kRefNone;
   p.global_mv[0] = inter::mc::Mv{0, 0};
   p.reorder = merge::DrlReorder::kDisabled;  // disable_drl_reorder = 1
   p.max_drl_bits_minus_1 = 4;
   p.num_refs = 1;
   p.frame_distance = nullptr;          // needed only for multiple refs
   ```

   The resolved mode then comes from `merge::resolve_single_mode()` or
   `merge::resolve_mode()`. The syntax writer keeps the same
   `single_mode`/`drl_idx`/`read_mv` symbol order.

2. **Delete the small-block workaround.** Once the checked stack is used, the
   `(w > 32 && h > 32) ? stack.count : Min(stack.count, 2)` limit in
   `m2_frame.cpp` is no longer needed and should be removed. For blocks at or
   below 32 in either dimension the spec stack now ends before the
   MVP-combination candidates, and every returned index is genuinely
   signalable.

3. **Reference labels.** The merge module follows the spec's RefFrame values:
   `0..6` are inter reference list indices, `7 = TIP_FRAME`,
   `8 = INTRA_FRAME`, `-1 = NONE`. `inter::mc` uses `kRefIntra = 0`, so its
   grid is not directly spec-compatible when reference list index 0 is in use.
   The current M2 single-reference stream labels its only slot `kRefLast = 1`
   (and intra cells `0`), which is accepted because labels are opaque to the
   stack comparisons. For multi-reference M2, either:
   * keep mc labels (`1..7`, intra `0`) and pass them consistently to
     `merge::StackParams` / `MiInfo`, or
   * switch the grid to spec labels (`0..6`, intra `8`) and remap at the MC
     boundary. Do not mix the two in one grid: a spec inter ref 0 would
     otherwise match an mc intra cell.

4. **Multi-reference context.** For `NumTotalRefs > 1`, set
   `p.num_refs` and `p.frame_distance[ref] = FrameDistance[ref]` (signed
   `get_relative_dist(OrderHint, RefOrderHint[ref_frame_idx[ref]])`, with the
   restricted-order-hint sign rule). This enables the §7.12.2.16 projection.
   With one reference it can stay `nullptr`.

5. **DRL reordering.** When `disable_drl_reorder = 1` (current M2 header), pass
   `DrlReorder::kDisabled`. To enable reordering, write
   `disable_drl_reorder = 0` and `constrain_drl_reorder` in the sequence
   header, and pass `kConstraint` / `kAlways` to both the encoder and the
   decoder side. The `useTemporalFirst` term is false for M2.

6. **Enabling `skip_mode`.** This is not a syntax-only change:
   * write `skip_mode_present = 1` only after
     `merge::derive_skip_mode_params().allowed`; `reference_select` must be 1
     for compound references to be signalable;
   * in `m2_headers.cpp`, stop forcing `skip_mode_present = 0`;
   * in the block writer, emit `read_skip_mode()` before `read_is_inter()` and
     before `read_skip()`; `skip_mode` blocks still code `skip_flag`;
   * for a `skip_mode` block emit no `single_mode`, no compound-mode bits and no
     MVD; emit exactly one `drl_idx` using `TileSkipDrlModeCdf`, and use
     `TileSkipModeCdf` for the `skip_mode` symbol itself;
   * derive the two references with `skip_mode_frames()` (the neighbour
     override matters), not just `(0,1)`;
   * `skip_mode` requires compound prediction: two `predict_plane()` results
     blended with `CwpIdx` (`CWP_EQUAL = 8/8` while `enable_cwp = 0`) and the
     `EIGHTTAP_SHARP` filter. The current M2 code has only single-reference
     prediction, so this must be implemented before skip_mode can be gated by
     `dav2d`. With one reference both list MVs may be identical, but the
     bitstream still has compound semantics.

7. **Contexts.** `m2_syntax.cpp::new_mv_context()` currently increments
   `NewMvCount` for any inter neighbour with a NEWMV-family YMode, without
   checking that the neighbour's reference matches the current block, and it
   does not handle compound blocks. Replace it with
   `merge::derive_new_mv_context()` (needs a YMode grid parallel to `MiInfo`).
   Also use `merge::skip_mode_context()` and
   `merge::skip_flag_context()`; the latter adds 3 to the `skip_flag` context
   when the current block has `skip_mode = 1`. The M2 writer already has the
   `TileSkipModeCdf` table; it is unused while `skip_mode_present = 0`.

8. **No CMake changes required.** `merge.cpp` only depends on
   `partition::block_width/height` and three `inter::mc` types/one function.
   When the M2 core adopts it, add `merge.cpp` to the M2 target in
   `CMakeLists.txt` (owned by the integration pass) and keep the standalone
   test script as the component gate.

## Tests

`merge_test.cpp` covers:

* `skip_mode_params()` for `NumTotalRefs = 1..4`, distance fallback,
  restricted order hints, intra/switch frames, invalid ref counts;
* `skip_mode_frames()` neighbour compound, TIP, single-neighbour fallback,
  invalid refs;
* block skip eligibility for all 29 block sizes (`is_comp_ref_allowed`, thin
  4xN/Nx4, segmentation features, intra region);
* DRL encode/decode: every `m = 1..5`, `start = 0..m+1`, every valid index,
  bit patterns, exhaustion, truncation, invalid indices, CDF rows, second-DRL
  start;
* `single_mode` values, YMode mapping, DRL/MVD presence, GLOBALMV/NEARMV/NEWMV
  resolution, skip_flag independence, reference mismatch, invalid DRL;
* skip_mode resolution (compound two-list MVs, no MVD, CWP, skip_flag
  independence) and rejection of malformed skip_mode / non-skip compound;
* `resolve_skip_mode()` deriving the reference pair through
  `skip_mode_frames()` and rejecting a stack built for a different pair;
* `NewMvContext` single and compound branches, reference matching,
  NEW/NEW_NEAR/NEAR distinction;
* MV stack scan order/weights/offsets, dedup (including `PruneCount`-limited
  behaviour), weight-sort tie-breaks, disabled/constraint/always ordering,
  determinism, other-reference filtering;
* §7.12.2.20 gate and `inter::mc` audit for every block size 4..64;
* the step-13 superblock-border scan difference;
* compound stacks, same-reference compound, reference counts 1..4,
  compound-add-derived mixing;
* derived-SMVP projection and the `max_drl_bits_minus_1 + 2` fill cap;
* invalid stack parameters (null grid, dimension limits, intrabc, missing
  second reference, bad `max_drl_bits_minus_1`).

## Spec ambiguities / surprises

1. **`skip_mode` with one reference.** `skip_mode_params()` allows
   `skip_mode_present` on any non-intra, non-switch frame. With
   `NumTotalRefs = 1`, `SkipModeFrame = {0, 0}` and `is_comp_ref_allowed()`
   checks only the block size, so a same-reference compound block is
   syntactically possible. The spec does not forbid it. This module supports
   it, but an encoder should normally require two references.

2. **`skip_flag` is still coded in `skip_mode`.** "Skip mode" is a mode-info
   shortcut, not a residual shortcut. Any implementation that treats
   `skip_mode` as "no residual" will mismatch on streams where a
   `skip_mode` block has coefficients.

3. **`skip_mode_frames()` can ignore `SkipModeFrame[]`.** A compound neighbour
   pair overrides the frame-level pair, and a TIP neighbour replaces it with
   `Min/Max(ClosestPast, ClosestFuture)`. Deriving the refs as simply
   `(0, 1)` is wrong.

4. **`read_drl_idx()` with `start > m`.** For same-reference `NEAR_NEARMV`,
   `start = RefMvIdx0 + 1` can be `m + 1`. The loop then reads nothing and
   returns `m`, i.e. the returned index can be smaller than `start`. There is
   no conformance requirement that the two DRL indices differ. Implemented
   literally.

5. **DRL index vs `NumMvFound`.** `RefStackMv` is initialized to zero for all
   `MAX_REF_MV_STACK_SIZE` entries, so a `RefMvIdx` at or beyond `NumMvFound`
   decodes to a zero MV rather than being rejected by the syntax. No
   conformance requirement ties `RefMvIdx` to `NumMvFound`. This module is
   stricter on the encoder side (`drl_index < stack.count`) to catch producer
   bugs; do not infer that the decoder enforces it.

6. **MVP-combination gate is strict `> 32`.** 32x64 and 64x32 do **not** get
   the combination candidates. Only 64x64 (in the 64x64-superblock M2
   configuration) qualifies.

7. **`TileSkipDrlModeCdf`.** The DRL CDF selection for `skip_mode` is
   `TileSkipDrlModeCdf[Min(idx,2)]`, not `TileDrlModeCdf`. The M2 writer
   currently always uses `TileDrlModeCdf`; that is only valid while
   `skip_mode_present = 0`.

8. **RESTRICTED_OH.** `FrameDistance[]` flips sign for restricted references,
   and `skip_mode_params()` forces the first two `curToRef` values to zero
   when the corresponding order hint is `RESTRICTED_OH`. The module models the
   sentinel as a boolean (`RefOrderHint::restricted`) rather than the spec's
   numeric sentinel; callers must set the boolean when the saved order hint is
   restricted.
