# mv_stack_full — complete §7.12.2 MV stack, `allow_warpmv_mode`, §7.7/R77

Spec-only completion of the AV2 v1.0.0 motion-vector prediction stack, the
`allow_warpmv_mode` frame bit, and the implicit reference map / display-order
hint rules. It is a standalone component: nothing in `src/encoder/m2/`,
`CMakeLists.txt`, `src/main.cpp` or any other agent's directory is modified.

Normative reference: **AV2 Bitstream & Decoding Process Specification v1.0.0**
(2026-05-28, commit `38f28e7`), §5.4.6, §5.18.2, §5.20.7.6–.20, §6.4.1,
§7.7, §7.8, §7.11.4, §7.12.2.1–.23. Findings were cross-checked against the
wired components `av2enc::video::merge`, `av2enc::video::adv` and `adv::build_ref_frame_list`.

## Files

| File | Purpose |
|---|---|
| `mv_stack_full.h/.cpp` | §7.12.2 steps 1–22: global MV, corner warp (§7.12.2.3/.4), scan point/col (§7.12.2.5/.6), add-warp (§7.12.2.9/.11), the full add-ref-MV dispatch (§7.12.2.10) including every TIP branch, derived stacks (§7.12.2.14–.18), sorting (§7.12.2.19), the exact extra search (§7.12.2.20), fill-derived (§7.12.2.22), clamping (§7.12.2.23), plus §7.11.4 `WarpMvCount`. |
| `frame_flags.h/.cpp` | §5.4.6/§5.18.2 `seq_enabled_motion_modes`, `seq_frame_motion_modes_present_flag`, `frame_enabled_motion_modes`, the `allow_warpmv_mode` bit and its constraints, `is_warp`/`warp_mv` gates. |
| `ref_map.h/.cpp` | §7.7 `get_ref_frames`, §7.8 `get_past_future_cur_ref_lists`, §5.18.2 `get_disp_order_hint`/`get_max_disp_order_hint`, the §6.4.1 **R77** operating-point checks, and the frame-header ref-map syntax (implicit vs explicit). |
| `stack_io.h/.cpp` | Deterministic stack record (magic, version, checksum). |
| `stack_mirror.h/.cpp` | Independent reader + structural invariants + per-DRL lockstep resolution. |
| `mv_stack_full_test.cpp` | 12 747 checks, pinned goldens. |
| `gate/` | Multi-reference M2 stream writer + `dav2d` gate. |

## API

`build(const StackParams&, Stack*, BuildState*)` runs the normative order:

1. setup global MV (lists used by the block);
2. §7.12.2.3 corners (iter 0, then iter 1 when `NumWarpFound == 0 && bw4 <= 16`);
3. optional temporal-first scan (injected hook);
4. scan points 6–11, temporal scan, scan point 13, `numNearest`, scan col;
5. sort (one move-to-front, §7.12.2.19);
6. compound fill-derived (step 18);
7. single fill-derived (step 20);
8. **extra search** (step 21);
9. clamping (step 22).

`StackParams` carries the full MI grid (`MiCell`: `Mvs`, `SubMvs`, `YMode`,
`MotionModes`, `WarpParams`, `TipSizes16x16`, `MiColBase`/`MiRowBase`),
`GmType[]`/`GlobalMvs[]`/`gm_params[]`, `FrameDistance[]`/`OrderHints[]`,
closest past/future, `DeriveWrl`, `use_ref_frame_mvs`/`enable_mv_traj`, the
warp-parameter bank and the §7.10 TIP motion field.

Every sub-process is public (`scan_point`, `add_reference_mv`,
`add_warp_motion_vector`, `single_add_derived`, `compound_add_derived`,
`derive_*_tip_*`, `fill_derived`, `extra_search`, `clamp_stack`,
`sort_stack_range`, `compute_warp_context`, `generate_points_from_corners`), so
the tests exercise the branch conditions directly.

`Stack::scanned_count` is the stack size after step 20 and before the extra
search; an index `>= scanned_count` names a candidate that only exists because
of §7.12.2.20 (global / MVP combination / derived insertion).

### §7.12.2.20 details that the audit called out

* **Clamp order**: the existing candidates are clamped with
  `clamp_mv_row`/`clamp_mv_col` *before* the global-MV candidate is compared and
  inserted. A candidate whose MV is outside the picture therefore compares
  equal to (the clamped) `GlobalMvs[]` and is not duplicated.
* **Global candidate**: added only when `NumMvFound < MAX_REF_MV_STACK_SIZE &&
  !use_intrabc`, with the shared `PruneCount < MAX_PR_NUM` budget.
* **MVP combinations**: only when `Block_Width > 32 && Block_Height > 32`;
  `num` is captured after the global candidate; the six `(yCand, xCand)` pairs
  are inserted in the normative order with the mixed-component construction.
* **Warp candidates**: when `DeriveWrl`, the bank is read newest-first, then
  `gm_params[RefFrame[0]]`, then two `Default_Warp_Params`, capped by
  `MAX_WARP_REF_CANDIDATES` with **no deduplication** (§7.12.2.11).

### `allow_warpmv_mode`

`derive_allow_warpmv_mode()` implements:

```
present = !FrameIsIntra && frame_enabled_motion_modes[DELTAWARP]
```

with `frame_enabled_motion_modes[mode] <= seq_enabled_motion_modes[mode]`.
`derive_warp_block_gates()` and `validate_warp_block()` enforce:

* `is_warp` only when `allow_warpmv_mode && Min(Block_Width, Block_Height) >= 8`;
* `warp_mv` only when `is_warp && !force_integer_mv`; `force_integer_mv`
  infers `warp_mv = 1`;
* `YMode == WARPMV` ⇒ `motion_mode == DELTAWARP`, and warp YModes require
  `is_warp`.

The sequence/frame bit writers are byte-compared with
`av2enc::video::adv::write_sequence_motion_mode_config` / `write_frame_bawp_warp_bits`.

### R77 / implicit reference mapping

`derive_implicit_ref_map()` runs both `get_ref_frames(0)` (before frame size)
and `get_ref_frames(1)` (after it). `RefMap` reproduces `mapOrderHint`,
`first_slot_with_ref`, `valid_ref_frame_size`, the distance-decay / base-Q /
resolution score, dedup, the `NRanked > REFS_PER_FRAME` drop rule, the stable
bubble sort, and the restricted append. §7.8 returns `ClosestPast`,
`ClosestFuture`, `FurthestFuture`, the past/future/current lists and
`SkipSegFrame`; `SkipSegFrame` is left as `NONE` when there is no past or
current reference (the spec leaves `ClosestPast == NONE`; `adv` substitutes 0 —
see “Deviations” below).

`get_disp_order_hint()` implements the CLK / restricted-SWITCH short-circuit,
`get_max_disp_order_hint(onlyShowable = 1)`, the wrap adjustment and the
`< 1 << (DISPLAY_ORDER_HINT_BITS - 1)` conformance bound.

`validate_r77()` checks all three §6.4.1 requirements across a set of operating
points: equal display order hints, an identical implicit mapping per reference
index, and an explicit buffer index that denotes the same frame everywhere.

`write_frame_ref_map_syntax` / `read_frame_ref_map_syntax` implement the
§5.18.2 presence rules: `SWITCH`/bridge forces an explicit map; otherwise
`explicit_ref_frame_map == 0` writes **no** bits and the reader derives the
mapping with §7.7.

## Tests

```sh
./src/encoder/inter/mv_stack_full/build_and_test.sh
```

* **merge parity** — 900 randomized grids (all block sizes 4…64, single and
  compound, with and without `FrameDistance`) compared candidate-for-candidate
  with `av2enc::video::merge::build_mv_stack`. `Reorder::kDisabled` is the parity
  configuration because the wired merge sort is the audit's known
  full-selection-sort deviation; a negative control pins the normative
  move-to-front result `{4,1,2,3}`.
* **§7.12.2.20** — clamp-before-global, global candidate presence, the >32×32
  gate at 16/32/64, warp-bank newest-first order and the four-entry cap,
  the no-dedup rule, and the `PruneCount` exhaustion path.
* **§7.12.2.10** — every branch: single search, single-add-derived (same-side
  projection), TIP single-ref derived, TIP add-derived, compound search,
  compound add-derived, TIP compound.
* **corners** — three corners inserted, all-equal MVs rejected.
* **`allow_warpmv_mode`** — writer/reader round-trip, every constraint, and bit
  equality with `av2enc::video::adv`.
* **§7.7/§7.8** — 700 randomized DPBs compared with
  `adv::build_ref_frame_list` and `adv::get_past_future_cur_ref_lists`;
  restricted append; display-order-hint wrap, short-circuits and bound.
* **R77** — accept and each rejection case.
* **stack mirror** — 600 randomized stacks: record parse, invariants, full
  field equality, and lockstep resolution for every DRL index ×
  {NEARMV, GLOBALMV, NEWMV}. Corruption, truncation and trailing bytes are
  rejected.
* **goldens** — eight pinned FNV-1a hashes over the serialized records.

## Gate

```
dav2d --strict 1 --threads 1
```

The gate builds a four-frame AV2 stream:

| frame | writer | references |
|---|---|---|
| 0 | M2 core (`OBU_CLOSED_LOOP_KEY`) | — |
| 1 | M2 core (regular tile group) | slot 0 |
| 2 | this module | slot 0 |
| 3 | this module, **NumTotalRefs = 3**, `ref_frame_idx = {2,1,0}`, `reference_select = 0`, **`allow_warpmv_mode = 1`**, `seq_enabled_motion_modes = {LOCALWARP, DELTAWARP}` | slots 2,1,0 |

Frame 3 uses 64×64 `PARTITION_NONE` leaves (the only size where the
`>32×32` MVP-combination gate fires) with NEARMV, GLOBALMV and NEWMV leaves.
`is_warp` is coded for every block, which also validates the §7.11.4
`WarpMvCount` CDF selection. Result:

```
== mv_stack_full gate: 192x192 ==
  p2 tile=12  p3 tile=14 (multi-ref)
  coverage: is_warp_bits=9 warp-blocks=0 global=1 near-extra=4 near-neighbour=0
            stack-grew=8 max-stack=6 max-warp=4
  PASS: 4 frame(s) decoded with --strict 1 --threads 1, recon exact
```

`near-extra=4` means four NEARMV blocks actually *select* a §7.12.2.20
candidate (index >= `scanned_count`); `stack-grew=8` means the extra search grew
the stack on eight of the nine leaves (up to the six-entry maximum);
`max-warp=4` is the full warp-candidate stack.

Probe flags (deterministic): `GATE_DRL_INDEX=<k>` pins the NEARMV index,
`GATE_SKIP_COVERAGE=1` bypasses the coverage assertion, and
`GATE_TRACE_STACK=1` prints each leaf’s post-search stack.

### Finding: the earlier “extra-search selection mismatch” was a gate bug

The first version of the gate set `plan.drl_index` but never copied it into
`video_adv::AdvancedBlockCoding::drl_index`, so every block signalled DRL index 0
while the encoder reconstruction used the planned (higher) index. That was
fixed in `gate/gate_stream.cpp` (`block.drl_index = plan.drl_index`), and the
gate now selects a §7.12.2.20 candidate wherever one exists:

```
stack leaf=2 rc=(0,32) count=2 scanned=1 nearest=1 warpf=3 warpins=0 prune=5
                   drl=1 mode=14 mv=(0,0)
  cand[0] mv=(0,-8) w=2 off=(15,-1)   <- scanned (left neighbour)
  cand[1] mv=(0,0)  w=1 off=(0,0)     <- §7.12.2.20 global-MV candidate
stack leaf=4 rc=(16,16) count=6 scanned=4 nearest=4 warpf=4 warpins=0 prune=16
                   drl=5 mode=14 mv=(-8,0)
  cand[0] mv=(-8,0) w=2 off=(15,-1)
  cand[1] mv=(0,-8) w=2 off=(-1,14)
  cand[2] mv=(0,0)  w=1 off=(-1,16)
  cand[3] mv=(-8,-8) w=0 off=(-1,-2)
  cand[4] mv=(0,0)  w=1 off=(0,0)     <- MVP combination (prune budget spent)
  cand[5] mv=(-8,0) w=1 off=(0,0)     <- MVP combination (prune budget spent)
```

Warp input is inert in this frame: no neighbour has `MotionModes >= LOCALWARP`
(`warpins=0`, no §7.12.2.9 insertion), `is_warp = 0` on every block, and the
`warpf=3/4` entries are only the §7.12.2.20 warp *parameter* candidates, which
never enter `RefStackMv` and are never selected.  The divergence was entirely in
the MV list.

Leaf 2 proves the global candidate is at index `scanned_count`.  Leaf 4 has
`prune=16`, i.e. `PruneCount == MAX_PR_NUM` exactly when the six MVP
combinations are inserted, so §7.12.2.20's `if (PruneCount < MAX_PR_NUM)` guard
suppresses pruning and the combination candidates are appended even when they
duplicate an existing entry ([4] = [2] = `(0,0)`, [5] = [0] = `(-8,0)`).  The
decoder reproduces that state — index 5 resolves to `(-8,0)` — which validates
both the insertion order and the shared prune-budget accounting.

Byte-level probe — every stack index, including all §7.12.2.20 entries, is
recon-exact:

```sh
for k in 0 1 2 3 4 5; do
  GATE_DRL_INDEX=$k GATE_SKIP_COVERAGE=1 \
    $OUT_DIR/mv_stack_gate --out /tmp/drl$k.ivf
done
```

```
DRL index 0: PASS: 4 frame(s) decoded with --strict 1 --threads 1, recon exact
DRL index 1: PASS ...
DRL index 2: PASS ...
DRL index 3: PASS ...
DRL index 4: PASS ...
DRL index 5: PASS ...
```

### Open finding: `is_warp` blocks are rejected by the pinned decoder

Any frame with an `is_warp = 1` block (WARPMV *or* WARP_NEWMV, identity or
LOCALWARP) makes the pinned `dav2d` (`f9dafbc`) fail with
`Error decoding frame: Invalid argument`. The same frame with
`allow_warpmv_mode = 1` and `is_warp = 0` on every block decodes exactly, so
the frame bit, its position and the §7.11.4 context are correct. The
`GATE_WARP_BLOCKS=1` run reproduces it; the build script asserts the failure is
still present and prints “expected FAIL”. The §7.12.2.9/.11 warp-stack
construction remains covered by the unit tests and the `av2enc::video::adv`
writer/reader round-trip, but there is **no decoder-side gate for warp blocks**
in this tree.

## Deviations noted while implementing

* `PruneCount` can exceed `MAX_PR_NUM` within one pruning sweep (the guard is
  outside the loop); the mirror’s invariant allows
  `MAX_PR_NUM - 1 + NumMvFound`.
* §7.8 leaves `SkipSegFrame == ClosestPast == NONE` when nothing is past or
  current; `adv::get_past_future_cur_ref_lists` substitutes 0. The parity test
  normalises that one field and documents it.
* A single-reference stack’s list-1 entries are unused by the spec; they are
  canonicalised to list 0 so the record is deterministic.
* `use_intrabc` is accepted but the §7.12.2.20 `add_to_ref_bv` candidates
  (intra block copy) are out of scope; the global-candidate `!use_intrabc`
  guard is implemented and tested.
* The §7.12.2.21 ref-MV bank is a separate component; the caller splices it
  between steps 18 and 20 (`StackParams` has no `enable_refmvbank`).

## Ownership

Only `src/encoder/inter/mv_stack_full/` is owned here. The gate compiles the
M2 core, `av2enc::video::merge`, `av2enc::video::adv` and `advanced` sources read-only.
