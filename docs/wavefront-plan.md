# Block-wavefront parallelism in the video encoder — design and dependency audit

Status: **stages 0–3 landed** (serial restructure, block dump, skew-2 wavefront;
see §9).  Originally an audit + plan with no code, written against `f179999` on
2026-11-16.  Pins in scope: 25,009 `a1d0dba1`, 25,141 `3984677b`, 23,690
`1b99127a`, 24,113 `7f03eb6e` (the last is the explicit-λ control; see
`docs/simd-plan.md:199`).

Read this next to the code. Line numbers are from the audit tree and will rot.

---

## 0. Verdict first

1. **The separation already exists.** `decide()` fills `blocks_` /
   `levels_` / `palettes_`; `write()` serialises them with its own private CDF and
   context state. Entropy is **not** interleaved with decisions. The bitwriter
   does **not** need restructuring — that is the good news, and it is the whole
   reason a wavefront is even discussable.
2. **The binding constraint is the decision state, and specifically the
   neighbour-availability predicate.** AV2's merge stack treats "this neighbour
   cell is not coded yet" as an *input* (`merge.cpp:592`), and the
   `BlockDecoded` map for intra above-right is superblock-local
   (`video_intra_search.h:130-166`). A schedule may therefore not run a block
   *early* any more than it may run it *late*: both change the candidate set.
   This, not the writer, is what forces coarse granularity and a skewed
   schedule.
3. **Hard ceiling.** At superblock granularity the maximum independent blocks per
   anti-diagonal is **4** (480×240) and **12** (1440×720); the *average*
   concurrency over a skew-2 schedule is **2.3** and **6.1**. Because the
   measured trees hold only ~1.5–1.7 leaves per superblock, sub-superblock
   granularity buys nothing on these clips. With the measured phase shares
   (`§4.5`) a perfect byte-identical wavefront over the leaf walk is worth
   **≈1.35× end-to-end** on both sizes — on top of the existing 1.15×/1.50×
   thread scaling.
4. **Determinism is reachable**, but only with a *level-synchronous
   superblock-diagonal schedule with skew 2 and strict within-superblock
   serialisation*. Every cheaper variant I could construct either reorders the
   availability set or races shared scratch.

If the goal is speed, the honest ranking is: (a) parallelise
`run_video_loop_filters` and the partition probe better, (b) then the leaf walk.
If the goal is a stream-exact wavefront, §6 is the smallest shape that works.

---

## 1. Separation: does `decide()` buffer, or stream into the bitwriter?

**It buffers. Completely.** `decide()` never touches the entropy coder; the CDF
state lives inside `syntax::Writer`, constructed in `write()`. Even the SDP path
— the one place that *plans* symbols during `decide()` — only fills
`BlockSyntax` fields through a pure planner (`video_frame.cpp:3615-3660`);
nothing is encoded. `deblock()` and `apply_loop_filters()` run *between*
`decide()` and `write()` and mutate pixels plus `filter_plan_`, never the
buffers.

Call sites, in order:

- `InterFrameEncoder::encode()` is the single-frame entry:
  `video_frame.cpp:3563` `bool encode(tile_payload, blocks, levels, error)`, and
  `:3566` `if (!decide(error)) return false;` → `:3567` `apply_tip_block_ref` →
  `:3568` `deblock` → `:3569` `apply_loop_filters` → `:3570` `write(...)`.
- `decide()` = `video_frame.cpp:2920`. It (i) solves the whole-frame partition
  tree (`:3133` `partition::decide_frame(geometry_, cost_fn, poptions)`, or
  `:3113` for the extended tree), (ii) optionally pre-computes pure ME
  (`:3154`), then (iii) walks the tree per superblock and fills the buffers
  (`:3167` `collect_node(sb, error)` inside `collect_superblocks`, `:3157`).
- The accumulators are `blocks_` (`:8121`), `levels_` (`:8130`), `palettes_`
  (`:8119`). Every leaf appends its coefficients at decision time, e.g.
  `video_frame.cpp:7512` `entry.levels_offset = levels_.size();` followed by
  `write_luma_plan(entry, res_y, levels_)`; the palette list at `:7288-7289`.
- `write()` = `video_frame.cpp:3405`. It builds a `syntax::FramePlan` from
  options + the decided buffers, then calls `syntax::assemble_video_frame`
  at `:3553`, and finally *moves the buffers out*: `:3558-3559`
  `blocks = std::move(blocks_); levels = std::move(levels_);`
  (`palettes_` is moved into the plan at `:3454`). `write()` is destructive:
  the encoder cannot write twice.
- `syntax::assemble_video_frame` = `video_syntax.cpp:5478`; it constructs
  `Writer` (`:5516`) and calls `writer.run()`. `Writer` owns *everything* it
  needs: CDFs (`video_syntax.cpp:866` `init_cdfs`), its own above/left context
  (`:893-912`), its own MI/MV/skip/mode grids (`:903-908`), its own palette
  writer and IBC state (`:915-951`), its own delta-Q plan (`:973`).
- The writer walks superblocks in raster order and *validates* that order:
  `video_syntax.cpp:1180-1197` rejects a superblock that is not at
  `(sb_index / sb_cols, sb_index % sb_cols)`. So `blocks_` must stay in raster
  superblock order, with each superblock's tree contiguous. It already is
  (`collect_node` appends in pre-order, `video_frame.cpp:3799`, `:3838`).

**Consequence.** Decisions may be *computed* in any order that preserves their
own dependency semantics, as long as the results are stored in the original
vector order. The entropy writer stays strictly serial and untouched. This is
the architecture the plan in §6 keeps.

The only `decide()`-time use of CDF-ish data I could find is a *static* model:
`video_tx_search.cpp:98-108` builds `txpart::TxPartitionCdfs` with
`cdfs.reset(...)` — the §9.3 defaults — and `src/codec/rate/rdoq/` uses generated
default tables (`rdoq_cdf_tables.h`, sourced from `spec/tables/default_cdf.json`).
No live, adapting CDF is read at decision time. See §5, condition D4.

---

## 2. Decision inputs: what each decision reads from its neighbours

Coordinate convention: `r`/`c` are luma 4×4 MI units; `(r-1, c-1)` is above-left;
`(r+h4, c-1)` is below-left. "reads" means the value enters the *decision*.

### 2.1 Intra luma mode search (`--video-intra-full`, in `quality`/`nightly`)

Entry: `video_frame.cpp:6538` `intra_search::availability(decoded_[0], ...)` →
`:6544` `luma_neighbor_smooth(r,c,...)` → `:6548` `intra_search::search_luma`.

| Input | Offset / extent | Source | Available in a wavefront? |
|---|---|---|---|
| edge samples above | row `r-1`, cols `[c, c+w4-1]` (pixels `y0-1`, `x0..x0+w-1`) | `recon_[0]` | yes, if row above is complete |
| edge samples left | col `c-1`, rows `[r, r+h4-1]` | `recon_[0]` | yes, if left column complete |
| above-right **count** | cells `(r-1, c+w4+i)`, `i < w4`, gated by `decoded_` | `video_intra_search.h:143-153` | see below — this is the awkward one |
| above-right samples | up to `4*count` pixels at `(y0-1, x0+w ..)` | `IntraEdges` | yes if the count is reproduced |
| below-left **count** | cells `(r+h4+i, c-1)`, `i < h4`, capped at the SB height | `video_intra_search.h:157-166` | SB-local |
| above-left sample | `(r-1, c-1)` only | `IntraEdges` | yes |
| smooth flags | `intra_joint_[r-1][c]`, `intra_joint_[r][c-1]` | `video_frame.cpp:4374-4390` | causal |

Is above-right **required**? Not formally — `count = 0` is legal and this
implementation produces `count = 0` whenever the block sits at the picture edge
or `have_above == 0`. But its *value* changes the prediction, so it must be
reproduced bit-exactly, not merely "be valid".

The reach, exactly (`video_intra_search.h:143-153`):

- `rel_y > 0` (block not on the SB's top row): `x < sb_w`, i.e. **the count never
  leaves the current superblock**. Availability is read from the SB-local
  `decoded_` map.
- `rel_y == 0` (block on the SB's top row): `x < border_w = min(2*sb_mi, ...)`
  and the map lookup is skipped (`rel_y - 1 >= 0` is false), so the count is
  **assumed** up to 2 superblock widths from the SB origin. Samples come from
  row `y0-1`, i.e. the superblock row above, at columns up to `sb_c0 + 2*sb_mi -
  1`. Since `2*sb_mi` is exactly 2 SB columns, this reaches at most **one
  superblock column to the right** of the current SB: `(r-1, c+1)` in SB units.

This is corroborated by `docs/negative-results.md:62` ("the top/left border is 1
out to two superblocks") and `:66` (below-left must not cross into the next SB
row).

Chroma DC/CfL: same function with `cx0/cy0`, `chroma_max_*`, and
`decoded_[1]`/`decoded_[2]` (`video_frame.cpp:6793`); `intra_dc_predict`
(`:584`) passes `num4_above_right = 0` and reads only above/left.

### 2.2 Inter MV predictors (merge stack)

Per reference: `video_frame.cpp:5934` `load_stack(r, c, w4, h4, ref_index, ...)`
→ `merge::build_stack_checked` (`merge.cpp:1414`, decl `merge.h:429`) → `StackBuilder::run`
(`merge.cpp:138`). Scan order and offsets, with `sb_border = (mi_row & 15) == 0`
(`merge.cpp:140`):

| Step | Call | Cell(s) read | Line |
|---|---|---|---|
| 6 | `scan_point(bh4-1, -1)` | `(r+h4-1, c-1)` left-bottom | `merge.cpp:150` |
| 7 | `scan_point(-1, bw4-1-(sb?1:0))` | `(r-1, c+bw4-1[-(1)])` above-right | `:151-152` |
| 8 | `scan_point(0, -1)` | `(r, c-1)` left-top | `:154` |
| 9 | `scan_point(-1, 0)` | `(r-1, c)` above | `:157` |
| 10 | `scan_point(bh4, -1)` | `(r+h4, c-1)` below-left | `:160` |
| 11 | `scan_point(-1, bw4 or 2)` | `(r-1, c+bw4)` above-right-2 | `:163` |
| 12 | `scan_point(-1, -1[-(1)])` | `(r-1, c-1[or c-2])` above-left | `:172/:174` |
| 15 | `scan_col(-3)` | `(r+h4-1, c-3+(c&1))`, `(r, c-3+(c&1))` | `:181`, `:620-626` |

Two properties matter more than the list:

1. **`ref[0] == kRefNone` means "not coded yet" and the candidate is *skipped*,
   not defaulted** (`merge.cpp:592-594`). Ordering is therefore part of the
   input. Running a block earlier adds a candidate; running it later removes
   one. Either way the DRL index changes and the stream changes.
2. When `sb_border` is true, every `delta_row < 0` column is rounded down to even
   (`merge.cpp:586-589`). So above/above-right reads from the SB's top row land
   on even cell columns of the *previous* SB row.

Cross-superblock reach of the above-right reads: `(r-1, c+bw4)` with `c+bw4`
at most the SB's right edge ⇒ at most the **first column of SB `(r-1, c+1)`**
(one SB column). `mvp_combination` candidates and `fill_derived` add no reads.
This mirrors `me_core.h:600-627` (A1/B1/B0/A0/B2), which is the same envelope in
sample coordinates and is *not* on the video decision path.

`--video-refmvbank` (off by default, `video_frame.h:204`) is the one option that
widens this: `reset_refmvbank_for_sb` (`video_frame.cpp:4089`) seeds from the
coded row above over `[mi_col, mi_col + sb_size4)` (`ref_mv_bank.cpp:196-211`)
and, more importantly, **the bank size/start rings are only cleared at a
superblock row start** (`video_frame.cpp:4096-4102`), so the bank is a *left-to-
right chain along the whole SB row*. Any wavefront holds for the pins, but
`--video-refmvbank` would additionally require column order within a row.
`--video-ref-frame-mvs` off ⇒ no temporal scan (`merge.cpp:272-273`).

### 2.3 Partition, skip, delta-Q

| Decision | Neighbour input | Evidence |
|---|---|---|
| Partition tree (rect) | **none.** `decide_frame` loops superblocks (`partition.cpp:1237`) and each `solve_node` prices leaves from source variance + the measured probe | `partition.cpp:1215-1250`, `video_frame.cpp:3005-3046` |
| Partition tree (ext) | **none**, same shape | `video_frame.cpp:3113` |
| Skip / residual flag | **none** (RD against SSE + a closed-form bit estimate) | `video_frame.cpp:7199-7242` |
| Gate: `skip_seg` / `globalmv_seg` | segment id only (see below) | `video_frame.cpp:5880-5891` |
| Delta-Q / segmentation qindex | per-SB table lookup `sb_qindex_[sb]`, where `sb = (mi_row/qp_sb_mi())*qp_sb_cols() + mi_col/qp_sb_mi()` — **pure index, no neighbour** | `video_frame.cpp:3778-3787` |
| Segment id prediction | `deltaq::segment_prediction(enc_seg_grid_, r, c)` = above/left | `video_frame.cpp:7579` |
| SDP partition context (keyframe, `--video-sdp` only) | `sdp_luma_sizes_` / `sdp_chroma_sizes_`, filled in decode order and read for the current node's context | `video_frame.cpp:3727-3730`, `:8126-8127` |
| IBC BV stack / palette cache (keyframe, opt-in) | above/left caches; validity via the frame-global `coded` map | `video_frame.cpp:4457`, `:4612`, `ibc.h:84-99` |

The partition tree being neighbour-independent is important: it means the tree
(and hence every block's geometry, and the node counts used for buffering) can be
computed once, up front, in parallel, with no ordering constraints at all. The
existing probe already does exactly that (`video_frame.cpp:2369-2392`).

### 2.4 Entropy / CDF state at decision time

None. Audited:

- `video_frame.cpp:1154` `estimate_tx_bits`, `:1188` `luma_plan_bits` (wrapper),
  `:1170` `search_plan_estimate_bits`, `:1322` `inter_syntax_bits`,
  `:1235` `leaf_rd_lambda`, `:1297` `intra_luma_syntax_bits` — all closed-form
  functions of the quantiser and the local plan.
- TX-partition / TX-type rates: static §9.3 default CDFs, rebuilt per call
  (`video_tx_search.cpp:98-108`).
- RDOQ coefficient rate: generated default tables (`src/codec/rate/rdoq/`).
- `compute_tx` (`video_frame.cpp:848`) takes no above/left level context and no
  coder state.
- The only CDF-ish members on the encoder are `cdf_seed_` / `cdf_final_`
  (`video_frame.cpp:8052-8053`), both **write-only** from `write()`'s point of
  view (`:3555`).

So decisions are already entropy-independent. That is what makes byte-identity
possible at all, and it is the assumption to protect in review.

---

## 3. Mutable state during `decide()`

Everything below is a member of `InterFrameEncoder` (`video_frame.cpp:8040-8160`).

| State | Written by | Read by a *different* block's decision? | Causality |
|---|---|---|---|
| `mv_grid_` (`:8106`) | `update_mv_grid` `:7995`, TIP `:7923` | **yes** — merge stack, refmvbank seed | causal: left (`c-1..c-3`), above (`r-1`), above-left, above-right (≤ +1 SB), below-left (**SB-local**), all gated by "already coded" |
| `intra_joint_` (`:8107`) | `set_intra_joint` `:4392` | **yes** — smooth flags | `(r-1,c)`, `(r,c-1)` only |
| `recon_[3]` (`:8095`) | `write_plane_block` (`:1724`) per leaf | **yes** — intra edges, BAWP template, IBC copy, morph | above/left; IBC additionally reads arbitrary *already-coded* samples (BV validity) |
| `decoded_[3]` (`:8120`) | reset per SB `:3765`, fill per leaf `:7621` | **yes** — intra above-right/below-left counts | **SB-local only** (§6.19.2.3) |
| `ibc_buffer_` / `ibc_neighbours_` (`:8111-8112`) | `begin/finish_ibc_block` `:5845`,`:7627` | **yes** | causal; `IbcBufferState.coded` is monotone |
| `palette_state_` (`:8118`) | `store_palette_state` `:7292` | **yes** | above/left cache |
| `refmv_bank_` (`:8139`) | reset `:4089`, update `:4137` | **yes** (opt-in) | row-above seed (own SB columns) + left chain along the SB row |
| `enc_seg_grid_` (`:8070`) | `commit_segment_block` `:7610` | **yes** (opt-in) | above/left |
| `sdp_luma_sizes_`/`sdp_chroma_sizes_` (`:8126-8127`) | `:3727-3730` | **yes** (SDP only) | above/left |
| `current_qindex_`, `current_segment_id_`, `quant_step_`, `plane_quant_` (`:8061`,`:8062`,`:8055`,`:8058`) | `:3778-3787` at each SB root | same SB only | **SB-local** — must become per-worker |
| `compound_y_/u_/v_` (`:8098-8100`) | `:5221`, `:5382`, `:5436` | no, but **shared scratch** | race hazard, not causality |
| `blocks_`, `levels_`, `palettes_`, `sub_pus_` (`:8121`,`:8130`,`:8119`,`:8104`) | appends | no (consumed by `write`) | append order must be the coding order |
| `inter_blocks_`, `intra_blocks_`, `temporal_selected_`, `amvd_selected_`, … (`:8131-8162`) | `++` | no | global counters, order-free |
| `me_cache_` (`:8044`) | `precompute_leaf_me` `:4296` (before the walk) | read-only during the walk | global, immutable in this phase |
| `sb_qindex_`, `sb_segment_`, `sb_delta_steps_` (`:8069-8075`) | `prepare_qp_plan` / fixpoint, before the walk | read-only | global, immutable |
| `filter_plan_`, `sdp_frame_`, `tip_*` | after/around the walk | no | frame-level |
| Rate model | — | **no state at all** | pure functions (§2.4) |

Classification for scheduling:

- **Causal, cross-block, must be sequenced:** `mv_grid_`, `intra_joint_`,
  `recon_`, `palette_state_`, `ibc_*`, `enc_seg_grid_`, `sdp_*`.
- **Causal but superblock-scoped:** `decoded_`, the delta-Q qindex block.
- **Row-chain (must become left-to-right along a row):** `refmv_bank_`.
- **Output-only, must be re-ordered deterministically:** `blocks_`, `levels_`,
  `palettes_`, `sub_pus_`.
- **Scratch that is secretly shared:** `compound_y_/u_/v_` — the sharpest
  parallelism bug in the file.
- **Global but read-only in this phase:** `me_cache_`, the QP plan,
  reference planes, the temporal motion field, `gm_plan_`.

No member was found whose *value* depends on the total order of decisions in a
way that a causal schedule could not reproduce, with one exception: the
`frame.cost` double accumulation in `partition.cpp:1250`. It is accumulated in
raster order and is not read by any decision (grep: no reader in `src/`), so it
may drift in the last bits if the tree solve is parallelised. Keep it ordered,
or accept a log-only difference.

---

## 4. Granularity and width

### 4.1 Geometry

| | 480×240 | 1440×720 |
|---|---|---|
| MI grid (`mi_cols × mi_rows`) | 120 × 60 | 360 × 180 |
| 64×64 superblocks | 8 × 4 = **32** | 23 × 12 = **276** |
| 32×32 | 15 × 8 = 120 | 45 × 23 = 1035 |
| 16×16 | 30 × 15 = 450 | 90 × 45 = 4050 |
| 8×8 | 60 × 30 = 1800 | 180 × 90 = 16200 |

Neither size is a multiple of 64 (480 = 7.5 SB, 240 = 3.75 SB; 1440 = 22.5 SB,
720 = 11.25 SB), so the last superblock row and column are **partial but still
coded**: they take part in the schedule with the same weight. Visible-edge
leaves are clipped (`partition::block_rect`) and the inter search is disabled for
blocks crossing the MI edge unless `--video-edge-inter`
(`video_frame.cpp:5860-5866`), which makes the last SB row cheaper — a load
imbalance, not a correctness issue. At 16×16 and 8×8 the grids divide exactly.

### 4.2 Naive anti-diagonal width (`r + c` = const)

`max units on one anti-diagonal = min(rows, cols)`:

| Unit | 480×240 | 1440×720 |
|---|---:|---:|
| 64×64 SB | **4** | **12** |
| 32×32 | 8 | 23 |
| 16×16 (min-block-16 leaves) | 15 | 45 |
| 8×8 | 30 | 90 |

These numbers are **not reachable**, for two independent reasons: the dependency
reach (§4.3) and the actual leaf density (§4.4).

### 4.3 Dependency-corrected width

A block's above-right reads reach up to one block *width* to the right in the row
above, and a leaf is capped at 64 px (`video_frame.cpp:3021` rejects leaves
wider than 64; `partition_probe` likewise). So in a schedule unit of `u` MI
cells the cross-unit reach is `16/u` unit columns, and the ordering
`key = s*r + c` is safe iff `s ≥ 1 + 16/u`:

| Unit (MI) | reach (units) | skew `s` | diagonals | max width 480p | max width 720p | avg concurrency 480p | avg concurrency 720p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64 SB (16) | 1 | **2** | 14 / 45 | **4** | **12** | **2.3** | **6.1** |
| 32 (8) | 2 | 3 | 36 / 111 | 5 | 15 | 3.3 | 9.3 |
| 16 (4) | 4 | 5 | 100 / 310 | 6 | 18 | 4.5 | 13.1 |
| 8 (2) | 8 | 9 | 321 / 981 | 7 | 20 | 5.6 | 16.5 |

`diagonals = s*(rows-1) + cols`; `avg concurrency = units / diagonals`. The
"max width" is `min(rows, ceil(cols/s))`.

Notice the inversion: **finer units are worse per unit of skew** because the
reach is measured in absolute pixels, not in units. A superblock absorbs its own
above-right reach; a 16×16 unit does not.

### 4.4 What the real trees actually contain

Measured on the two clips (block counts printed by the encoder):

| Stream | frames | inter leaves | intra leaves | leaves/frame | SBs/frame | leaves per SB |
|---|---:|---:|---:|---:|---:|---:|
| 480×240 `--qp 110 --fast-decode` | 60 | 2893 | 259 | ~52 | 32 | **~1.6** |
| 1440×720 `--qp 217 --fast-decode --video-ext-partitions` | 24 | 7761 | 3581 | ~473 | 276 | **~1.7** |

The trees are *coarse*: on average one to two leaves per superblock, mostly one
64×64 or a pair of 32×32. A 64×64 leaf occupies a 4×4 block of the 16×16 cell
grid, so a 15-cell 16×16 anti-diagonal in practice holds 1–2 **decisions**.
**The block-granular columns of §4.2 are fiction on these clips.** The attainable
unit set is the superblock grid: 32 and 276 independent units.

### 4.5 The hard ceiling, stated plainly

- Peak independent work: **4 blocks (480×240), 12 blocks (1440×720)**.
- Average concurrency over a skew-2 schedule: **2.3 / 6.1**.
- `--cpu-threads 8` therefore cannot be saturated by a byte-identical wavefront
  at any granularity on these sizes. 8 threads are useful, but they will be
  idle most of the time unless the schedule is allowed to *change the stream*.
- Frame-time budget (sampled, 1 thread, `sample(1)`; see Appendix A):

| Phase | 480×240 nightly | 1440×720 nightly+ext |
|---|---:|---:|
| `decide()` total | 79 % | 56 % |
| — of which the leaf walk (`collect_node[_ext]`) | **48 %** | **31 %** |
| — of which the partition probe | ~16 % | ~25 % (already parallel) |
| `run_video_loop_filters` (CDEF/CCSO/LR/GDF) | ~15 % | ~3 % |
| `write()` (entropy) | **0.5 %** | **0.1 %** |
| unattributed | ~6 % | ~40 % (profile gaps) |

Amdahl over the leaf walk alone, at the average concurrency:

- 480×240: `1 / (1 - 0.48 + 0.48/2.3) = 1.37×`
- 1440×720: `1 / (1 - 0.31 + 0.31/6.1) = 1.36×`

Measured today (1 vs 8 threads, identical md5): **1.15×** (480p nightly) and
**1.50×** (720p nightly+ext), from the already-parallel probe and leaf ME. So
the wavefront's realistic prize is a further ~1.2–1.35×, not 4–8×. **The
bitwriter is not the problem; the serial tail is the loop filters and the
sub-linear schedule.**

---

## 5. Determinism conditions

A wavefront is byte-identical **iff** all five hold. Each is stated with the
concrete failure mode.

**D1. Within a superblock, leaves execute in the exact tree pre-order.**
`collect_node` walks parent-then-children (`video_frame.cpp:3799-3841`), which is
the decoder's coding order. *Counterexample:* run the four quadrants of a
`PARTITION_SPLIT` SB concurrently. A leaf in the top-right quadrant whose
`bh4 ≤ 16` scans below-left at `(r+h4, c-1)` (`merge.cpp:160`) and, in raster
order, finds the bottom-left quadrant already coded; under a parallel quadrant
walk it finds `ref[0] == kRefNone` and skips the candidate (`merge.cpp:592`).
The stack loses one entry, every `drl_index` after it shifts, and the block codes
a different MV. Same for `decoded_` (`video_frame.cpp:3765` resets it per SB, so
above-right/below-left availability inside the SB depends on the walk order).

**D2. A superblock may not be scheduled concurrent with its above-right
neighbour.** *Counterexample:* 480×240, SB `(0,2)` and SB `(1,1)` share
`r + c = 2`. Order them `(1,1)` first. The first leaf of `(1,1)` is a 64×64 block
at mi `(16,16)`; `sb_border` is true, so the step-11 above-right scan is
`scan_point(-1, 16)` → column 32 → the cell `(15, 32)`, inside SB `(0,2)`
(`merge.cpp:163`, `:586-589`). SB `(0,2)` has not run, so `ref[0] == kRefNone`
and the candidate is dropped. And for intra, the same block's
`num4_above_right` is `min(w4, 32 - (rel_x+w4)) = 16`
(`video_intra_search.h:143-153`), so the predictor reads 64 samples of row 15 at
MI columns 32..47 — **zeros** if SB `(0,2)` has not reconstructed them
(`recon_` is zero-filled at `video_frame.cpp:3187-3189`). Fix: skew 2 (§4.3).

**D3. The above and left pixels a block reads must be fully written.** The recon
planes are zero-initialised per pass (`video_frame.cpp:3186-3191`), so a
half-written row above produces a *legal-looking but wrong* DC/SMOOTH/PAETH
prediction — it will not error, it will silently choose a different mode.
*Counterexample:* any diagonal schedule that starts row `r` before row `r-1` has
finished its column span, e.g. a "wavefront" that advances by column.

**D4. The live adaptive CDF must never enter a decision.** All decision-time
rates are closed-form or built from the §9.3 defaults (§2.4); if that ever
changed, block N's cost would depend on N−1's coding order and no schedule but
the serial one would be stable. *Counterexample (hypothetical, currently not
reachable):* `--video-tx-context-rate` reading `Writer`'s `cdfs_` instead of
`txpart::TxPartitionCdfs::reset()` (`video_tx_search.cpp:98-108`) would make the
TX-partition choice depend on the CDF carried from earlier blocks; two blocks on
the same diagonal would then price the same split differently. Keep
`video_tx_search.cpp:99` as an invariant, and keep `cdf_seed_` write-only.

**D5. Shared mutable scratch is per-worker.** *Counterexample:* two leaves in
flight both `compound_y_.assign(w*h, 0)` (`video_frame.cpp:5221`, `:5382`) and
then read `compound_y_` for their SSE. One leaf compares against the other's
prediction, picks a different mode, and the stream changes — silently,
nondeterministically, and only under contention. Same class: `sub_pus_`
(`:5113`). This is the most likely *silent* failure in a naive implementation.

Two more things that do **not** break the stream today but must stay out of the
decision path:

- `frame.cost` accumulation order (`partition.cpp:1250`) — no reader.
- Counter totals and `std::cerr` trace interleaving — order-free.

And one that does: the delta-Q fixpoint (`video_frame.cpp:3204-3238`) re-runs the
*whole* `collect_frame()` up to 8 times and feeds the previous pass's decisions
into `plan_delta_q_steps`. Any nondeterminism inside a pass changes the qindex
plan and therefore the stream. The wavefront must be a pure function of the pass
input, which D1–D5 guarantee.

---

## 6. Implementation shape

### 6.1 The smallest workable design

**Schedule.** Replace the per-superblock loop at `video_frame.cpp:3157-3170` with
a level-synchronous diagonal schedule over superblocks:

```
sb_grid: rows R, cols C          // partition::sb_rows/sb_cols
key(sbr, sbc) = 2*sbr + sbc      // skew 2: absorbs the +1 SB above-right reach
for d in 0 .. 2*(R-1)+C-1:
    parallel_for(members(d), workers, run_one_sb)   // independent by D2
```

`members(d)` is built once per frame, in increasing `sbc` (a stable order, so the
`parallel_for` contiguous chunking from `parallel_for.h:52-78` is reproducible).
Inside a superblock, `collect_node` stays **exactly** as it is: same recursion,
same leaf order, same neighbour predicates. That single decision — "an SB is the
atomic unit" — is what buys D1, D2 and the `decoded_` reset for free.

**Buffering.** One `SbSink` per superblock:

```
struct SbSink {
  std::vector<syntax::BlockSyntax> blocks;    // pre-order, SB-local indices
  std::vector<std::int32_t>        levels;
  std::vector<palette::PaletteResult> palettes;
  std::vector<post::VideoSubPuRect>   sub_pus;
};
```

`encode_leaf` currently reaches for `levels_`/`blocks_`/`palettes_` directly
(20 references between `video_frame.cpp:5772-7635`); those become a `SbSink&`
parameter. `entry.levels_offset = levels_.size()` (`:7326`, `:7373`, `:7512`)
and `entry.palette_index = palettes_.size()` (`:7288`) are then SB-local; a
single serial merge pass adds the base offsets. `entry.subtree`
(`:3838`) is already correct SB-locally, because an SB's nodes are contiguous.

**Merge (serial, cheap).** In superblock raster order:

```
base_blocks = 0; base_levels = 0; base_palettes = 0
for sb in raster order:
    for each block: blocks_.push_back(block)
    for each block with levels_offset >= 0: entry.levels_offset += base_levels
    for each palette ref: entry.palette_index += base_palettes
    base_levels += sink.levels.size(); ...
```

This restores the exact `blocks_`/`levels_`/`palettes_` layout the writer
validates (`video_syntax.cpp:1180-1197`). `write()` is untouched.

**Per-worker state.** Bundle everything in §3 that is SB-local:
`decoded_[3]`, `current_qindex_`, `current_segment_id_`, `quant_step_`,
`plane_quant_[3]`, `compound_y_/u_/v_`, and the `sub_pus_` sink. Share, read-mostly:
`recon_`, `mv_grid_`, `intra_joint_`, `palette_state_`, `ibc_*`,
`enc_seg_grid_`, `sdp_*`, `me_cache_`, the QP plan, the references. `refmv_bank_`
and `enc_seg_grid_`'s left context need a per-SB copy chained along the row
(only when those opt-in tools are on).

**Entropy writer.** Unchanged and serial. Its cost is 0.1–0.5 % of frame time
(§4.5), so there is nothing to gain by splitting it, and splitting it would
break D4.

**Load imbalance.** Two independent mechanisms:

1. *Within a diagonal:* units are independent (D2), so a work-stealing pull over
   `members(d)` is byte-safe. The schedule does not have to be reproducible —
   only the unit results do — which is the one place where non-determinism is
   allowed.
2. *Across diagonals:* bound the wavefront by a fixed lag `L` (e.g. 2 diagonals)
   and let rows run ahead, so a cheap diagonal does not stall the pool. This is
   still safe with skew 2 as long as the lag respects `s`. Start with the plain
   level-synchronous version; only add lag if the profile demands it.

Expected gain per §4.5 is ~1.35× on the leaf walk; do not build a scheduler more
elaborate than the gain.

### 6.2 Staged plan, each stage gated

**Stage 0 — instrumentation (no behaviour change).**
Add a debug dump of the decided frame: for every `BlockSyntax` in order, its
`(mi_row, mi_col, bsize, partition, is_inter, mode, ref, drl_index, mv,
skip_flag, eob[*], levels digest)` plus the palette list, written to
`$AV2ENC_BLOCKDUMP`. Gate: the dump is byte-identical across 10 runs and across
1/8 threads on all four pins, and its md5 is recorded in the doc. This is the
tool that turns "the stream changed" into "leaf 431's `drl_index` moved".

**Stage 1 — single-threaded restructure with identical output.**
Introduce `SbSink`, thread it through `encode_leaf`, split `collect_frame` into
`plan → run_one_sb → merge`, but keep the loop serial and in raster order. No
threads. Gate: all four pins byte-identical (`a1d0dba1`/`3984677b`/`1b99127a`/
`7f03eb6e`), the Stage-0 dump unchanged, `dav2d --strict 1` on each pin, and the
`--fast-decode` gate suite (`tools/run_gates.sh`). This stage is *guaranteed*
safe and is 80 % of the work.

**Stage 2 — 2-way wavefront.**
Two SBs from the same diagonal, hard-wired (no pool dependency, so this can land
before the persistent pool exists). Gate: Stage-1 gate, plus 20 repeated runs at
2 threads with identical md5, plus ThreadSanitizer on the 4-frame 640×360 clip.

**Stage 3 — N-way, `workers = min(cpu_threads, members(d).size())`.**
Gate: Stage-2 gate, plus a thread sweep `--cpu-threads 1..8` producing one md5
per pin, plus a wall-clock A/B against the Stage-1 binary (report the ratio, and
revert if it is below ~1.15×, because the added state will not be worth it).

**Explicit non-goals.** No entropy-side parallelism, no change to `write()`, no
change to the partition solver's tie-breaks, no per-leaf (sub-SB) scheduling in
the first landing. Each would break a condition in §5 or buy nothing (§4.4).

---

## 7. Risks, ranked by probability of a *silent* stream change

1. **Neighbour availability mismatched (above-right / below-left / merge
   "not yet coded").** Highest probability because both failure directions
   produce a *valid* stream. Catch: Stage-0 block dump diffed 1 vs N threads
   (per-leaf `drl_index`, `mv`, `y_mode`), the four pin md5s, and
   `dav2d --strict 1` plus recon equality — plus a targeted test that forces
   skew-1 ordering and must fail (a positive control: if skew 1 also passes, the
   dump is not sensitive enough).
2. **Shared scratch aliasing (`compound_y_/u_/v_`, `sub_pus_`, and the
   `levels_`/`blocks_` appends).** Catch: ThreadSanitizer over a 4-frame 640×360
   run, plus 20×repeat md5 at 8 threads (a race usually shows as run-to-run
   md5 drift), plus a single-thread vs 8-thread `cmp` of the IVF.
3. **Output re-ordering / offset fix-up (`levels_offset`, `palette_index`,
   `subtree`, SB order).** Lower probability of silence — the writer's
   validation (`video_syntax.cpp:1180-1197`) rejects a misplaced superblock and
   a stale `levels_offset` normally trips the entropy coder or `dav2d`. Catch:
   Stage-0 dump + `dav2d --strict 1` + the decoded-payload byte comparison the
   nightly gate already does.
4. **`refmv_bank_` / segmentation row chains.** Only if those opt-in tools are
   enabled; the pins do not use them, so this is a *future* risk. Catch:
   `src/pipeline/video/video_mv_traj_gate.sh` (bank/mv-traj) at
   `--cpu-threads 1,8`. Note there is **no** segmentation/delta-Q gate registered
   in `tools/run_gates.sh` (grep: no hit) — add one before enabling those tools
   under a wavefront.
5. **Delta-Q fixpoint picking up nondeterminism.** Catch: `--video-delta-q`
   gate stream + repeated runs; the existing neutral-fallback path
   (`video_frame.cpp:3219-3238`) already errors out loudly when the plan does not
   reproduce.
6. **CDF carried across a boundary by accident (if anyone ever parallelises
   `write()`).** Catch: the `--avg-cdf --adaptive-cdf --cdf-continuity` pin
   `1b99127a` — it exists precisely to pin the CDF chain.

---

## 8. Unverified, and the experiment that settles it

| Claim | Status | Experiment |
|---|---|---|
| Leaves per superblock on the *pinned* streams (I measured the same commands that reproduce the pins, but 1.6/1.7 is clip-specific) | measured, generalisation unverified | Stage-0 dump; histogram `(mi_row+mi_col)` per frame |
| The 40 % unattributed slice in the 1440×720 profile | **unverified** — `sample` gaps, and `AV2ENC_PROFILE` produced no summary in this build | rebuild with the built-in profiler and `jq '.stages[]'`, or add temporary `Scope` timers around `decide`/`write`/`deblock` |
| Whether `run_video_loop_filters` can be parallelised by plane/unit | **unverified** | read `video_loop_filters.cpp`; it is 15 % at 240p and therefore worth more than the writer |
| That no decision reads `frame.cost` | verified by grep at this commit | re-grep when the RDO code (being edited concurrently) lands |
| Whether a `pe::`/ext-tree node-count helper exists for pre-sizing sinks | **unverified** | `grep -n "nodes" src/codec/partition/ext/*.h`; fall back to a local recursion, or to per-SB `std::vector` growth |
| Skew 2 is sufficient for *all* tools (I verified the reach analytically and against `video_intra_search.h` + `merge.cpp`) | analytic + code-read, **not** tested | the positive control in risk 1: build with skew 1 and confirm the block dump differs |

---

## Appendix A — measurements used above

Host: macOS arm64, `build/av2enc` (2026-11-16 19:28). Single runs, not medians;
treat as indicative. All md5s reproduced the pins.

```
# 480x240, nightly, 60 frames
t1 6.45 s   md5 1b99127a4cd1d386fa5600fb63c57a83  (23,690 B)
t8 5.63 s   md5 1b99127a4cd1d386fa5600fb63c57a83   -> 1.15x

# 1440x720, nightly + ext partitions, 24 frames
t1 41.43 s  md5 b8afd8509cd3870d887f25dcbbdc55fb
t8 27.62 s  md5 b8afd8509cd3870d887f25dcbbdc55fb  -> 1.50x
```

Commands (both reproduce the pinned decisions):

```
build/av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/a.ivf \
  --video --frames 60 --qp 110 --cpu-threads {1,8} --fast-decode

build/av2enc testdata/space8k_1440x720_60f.y4m -o /tmp/b.ivf \
  --video --frames 24 --qp 217 --cpu-threads {1,8} --fast-decode \
  --video-key-qp-offset -57 --video-leaf-rd-lambda 250 --video-ext-partitions
```

Profile (1 thread, `/usr/bin/sample <pid> 3`, 1 ms interval, samples in the
per-frame loop):

- 480×240, 60 frames: 1995 samples — `decide` 1571 (79 %), `collect_node` 953
  (48 %), `search_luma_tx` 490, `partition_probe_leaf_cost` ≈327 (16 %),
  `run_video_loop_filters` ≈290 (15 %), `write` 9 (0.5 %).
- 1440×720, 12 frames: 4433 samples — `decide` 2486 (56 %),
  `collect_node_ext` 1386 (31 %), `search_luma_tx` 658,
  `run_video_loop_filters` ≈133 (3 %), `write` 5 (0.1 %).

Leaf counts are the encoder's own summary line (`inter blocks`/`intra blocks`).

---

## 9. Stage log

### 9.1 Stage 0 — per-leaf decision dump (landed)

`AV2ENC_BLOCKDUMP=<path>` writes one line per decided leaf to `<path>`, in
raster leaf order:

```
mi_row mi_col bsize partition is_inter mode ref drl_index mv_x mv_y skip eob0 eob1 eob2 levels_digest[ pal=size,direction,colors]
```

- `mode` is `inter_mode` for an inter leaf and `y_mode` for an intra leaf.
- `eob0..2` are `BlockSyntax::eob[0..2]` (luma, U, V).
- `levels_digest` is a 64-bit FNV-1a (`%016llx`) over the quantised `int32`
  levels of this leaf in writer order, i.e. exactly the run the leaf appended to
  `levels_` between its `encode_leaf()` entry and return.
- `pal=…` is appended only for a palette leaf: palette size, direction and the
  size colour bytes as lowercase hex, comma-separated.

Zero cost when unset: one predictable `block_dump_enabled()` branch per leaf;
no allocation, no formatting and no file. Records are buffered during the walk
and sorted by `(mi_row, mi_col)` before printing, which is a no-op for the
serial raster walk but makes the text independent of the completion order a
wavefront will use. Pre-order leaves of a raster tree are in raster order, so
the sort cannot reorder anything the serial encoder produces.

Dump md5s, 480×240 pins (60 frames):

| Pin | Encode args | Dump md5 |
|---|---|---|
| default, 25,009 `a1d0dba1` | `--video --frames 60 --qp 189` | `ab38bfb606c46f109f2102303fafa9e0` |
| quality, 25,141 `3984677b` | `… --qp 190 --video-rd-preset quality` | `2558da60a0918b0205bf32f9da086e5a` |
| nightly, 23,690 `1b99127a` | `… --qp 110 --fast-decode` | `d96b3a5368411d8a960d821271a49b52` |
| explicit-λ, 24,113 `7f03eb6e` | `… --qp 110 --fast-decode --video-qp-lambda --video-tx-lambda 136 --video-key-tx-lambda 136 --video-leaf-rd-lambda 125 --video-rdoq-scale 0.1 --video-partition-rd-lambda 2.75e-6` | `3deaf2520bf2b137e8336437e073ea31` |

Each md5 is identical across 10 repeated runs and across `--cpu-threads 1` and
`--cpu-threads 8` (11 runs per pin, one distinct dump md5 per pin; the IVF is
`cmp`-identical across thread counts as before). The 1440×720 recipe
(36,572 B / `c4ba2813…`) also reproduces and dumps 22,732 leaves.

### 9.2 Stage 1 — serial plan → run_one_sb → merge (landed)

The frame walk is split into three pieces with the same serial, raster-order
drive.  No thread is created; `--cpu-threads` still only feeds the existing
partition probe and leaf-ME precompute, and needs no new plumbing because a
serially filled `SbSink` has no thread affinity.

```
struct SbSink {                      // one per superblock, raster order
  std::vector<syntax::BlockSyntax>        blocks;      // tree pre-order
  std::vector<std::int32_t>               levels;
  std::vector<palette::PaletteResult>     palettes;
  std::vector<post::VideoSubPuRect>       sub_pus;
  std::vector<LeafDumpRecord>             leaf_dump;   // stage 0 records
  std::vector<std::uint8_t>               compound_y, compound_u, compound_v;
  void reset();
};
```

- `plan_pass()` — the frame-level reset that used to open `collect_frame()`
  (buffer clears, palette/state/grid resets, segment contexts).  It is called
  once per pass, so the delta-Q fixpoint still sees the identical starting
  state.  `plan_sdp_pass()` is the same idea for the SDP keyframe walk.
- `run_one_sb(sink, sb, error)` — clears one sink and runs that superblock's
  whole tree (`collect_node` / `collect_node_ext` / `collect_sdp_node`).  The
  depth-0 branch keeps the superblock-local resets (`decoded_`, refmvbank, QP)
  exactly where they were, and the recursion order inside the superblock is
  untouched.  `SbSink&` is threaded through `encode_leaf`, `encode_tip_leaf`,
  `collect_sub_pus` and the two compound predictors, so the per-worker scratch
  from §3 now belongs to the unit, not the encoder.
- `merge_sinks()` — appends in raster superblock order and rebases the only
  three SB-local references: `levels_offset` (with `kNoLevels` = `0xFFFFFFFF`
  as the "no residual" marker, mapped back to the writer's 0),
  `tx_part_levels_offset[0..tx_part_count)`, and `palette_index`.  Block order
  and `subtree` are already correct inside a sink.

Deliberately unchanged: `write()`, the partition solver and its tie-breaks, the
within-superblock pre-order, and the fact that `sub_pus_` is never cleared
between passes (the historical walk never cleared it; clearing it now would
change what the deblock grid sees).

**Results.** The four 480×240 pins and the 1440×720 recipe reproduce their bytes
and md5s exactly (`a1d0dba1` 25,009; `3984677b` 25,141; `1b99127a` 23,690;
`7f03eb6e` 24,113; `c4ba2813` 36,572).  The four dump md5s are *unchanged* from
§9.1 and `cmp`-identical line for line to the stage-0 dumps:

| Pin | Dump md5 (stage 0 == stage 1) |
|---|---|
| default | `ab38bfb606c46f109f2102303fafa9e0` |
| quality | `2558da60a0918b0205bf32f9da086e5a` |
| nightly | `d96b3a5368411d8a960d821271a49b52` |
| explicit-λ | `3deaf2520bf2b137e8336437e073ea31` |

Gates: `tools/refactor/check_thread_determinism.sh` OK (1 vs 8 bit-identical at
both operating points), `gate_primitives`, `gate_build_seam`,
`tools/test.sh` 10/10, `gate_video_nightly`, `gate_video_qp_lambda`.

**Timing.** The restructure is a bookkeeping move; it adds one sink clear per
superblock, a vector `insert` per buffer per superblock, and one offset pass.
There is no measured difference outside run-to-run noise (see §9.2.1).  No
speedup is claimed; stage 1 exists to make stage 2 reviewable.

#### 9.2.1 Timing (no claim, honest A/B)

Same host, `--cpu-threads 8`, HEAD binary alternated with the stage-1 binary,
IVFs `cmp`-identical to the pins in every run:

| Workload | HEAD | stage 1 | spread |
|---|---:|---:|---|
| 480p default pin, 3 runs | 0.575 s (0.567–0.580) | 0.534 s (0.533–0.580) | run-to-run ±5 % |
| 720p recipe, 2 runs | 38.66 s (35.93–41.39) | 38.47 s (35.84–41.10) | run-to-run ±8 % |

The ranges overlap completely; the restructure costs nothing measurable at
superblock granularity (one sink clear and a few vector inserts per
superblock).  Stage 2 is where the wall-clock change has to show up.

### 9.3 Stage 2 — 2-way skew-2 wavefront (landed)

The middle loop of `collect_frame()` (`run_superblocks()`) now dispatches the
superblocks on the §6.1 diagonal with **skew 2**, and runs two units from the
same diagonal at a time with one `std::thread`.  Nothing else changed: the unit
is still a whole superblock, `collect_node*()` keeps its serial pre-order,
`merge_sinks()` still rebases in raster order, and `write()` is untouched.

```
key(sbr, sbc) = 2*sbr + sbc      // sb -> diagonal
for each diagonal, in increasing key:
    for pairs of members (increasing sbc):
        member[i]   on the calling thread
        member[i+1] on one std::thread
        join
```

Skew 2 versus `r + c` is the §4.3/§5 D2 argument, and it is now the schedule's
load-bearing invariant.  The only cross-superblock reads are left (`key-1`),
above (`key-2`), above-left (`key-3`) and above-right one superblock column
into the row above (`key-1`, `merge.cpp:586-589`/`:163`,
`video_intra_search.h:143-153`).  Two units on the same diagonal are therefore
independent.  With `r + c`, SB `(0,2)` and SB `(1,1)` would share a diagonal.

**Per-unit state.** The walk's superblock-scoped state moved from
`InterFrameEncoder` into `SbSink::State` (one per unit):

| field | reason |
|---|---|
| `decoded[3]` | §6.19.2.3 map; reset at the unit root, read by the unit's own intra availability |
| `current_qindex`, `current_segment_id`, `quant_step`, `plane_quant[3]` | written at the unit root from the QP plan; the encoder keeps a frame-level copy only for the partition probe |
| `inter_blocks`, `intra_blocks`, `temporal_added`, `temporal_selected`, `same_ref_blocks`, `refmvbank_added`, `refmvbank_selected`, `mv_traj_derived`, `amvd_selected`, `flex_precision_selected`, `derived_sign_selected`, `tip_blocks` | accumulators; summed in raster order by `merge_sinks()` |

`make_stack_params`/`load_stack` now take the unit state so the merge stack's
`temporal_added`/`mv_traj_derived`/`refmvbank_added` counters land in the unit
rather than on the encoder.  `compound_y/u/v` were already sink-local.

**Shared, deliberately:** `recon_`, `mv_grid_`, `intra_joint_` (cell writes are
confined to the unit's own rectangle, cross-unit reads are causal),
`me_cache_`, the QP plan, the reference planes, `qm_state_`, `options_`,
`source_`.  **Not shared:** everything in the table above, `blocks_`, `levels_`,
`palettes_`, `sub_pus_`, `leaf_dump_`.

**Kept serial** (`wavefront_allowed()`): intra keyframes (palette cache, IBC
ring/neighbor map, morph state, SDP plane grids are row-ordered), and
`--video-refmvbank` (each unit seeds from the *whole* coded row above, which a
diagonal does not have), and segmentation/delta-Q (above/left context map with
a superblock-row clear; §7.4 notes there is no gate for it).
`--cpu-threads 1` takes the historical raster loop and never constructs a thread.

**Results.** Four 480×240 pins and the 1440×720 recipe reproduce their bytes and
md5s at 1 and 8 threads, and the Stage-0 dumps are `cmp`-identical between the
two thread counts and equal to §9.1/§9.2:

| Pin | dump md5 |
|---|---|
| default | `ab38bfb606c46f109f2102303fafa9e0` |
| quality | `2558da60a0918b0205bf32f9da086e5a` |
| nightly | `d96b3a5368411d8a960d821271a49b52` |
| explicit-λ | `3deaf2520bf2b137e8336437e073ea31` |

20 repeated 8-thread nightly runs give one md5, and the sweep
`--cpu-threads 1..8` gives one md5.  Timing and the stage-3 decision rule are in
§9.4.

### 9.4 Stage 3 — N-way pool wavefront + the schedule rule (landed)

**Schedule.**  `run_wavefront()` keeps the §6.1 level-synchronous shape and
runs one fork-join per diagonal through the persistent pool
(`parallel::parallel_for`, so `PoolScope` from the composition root is used and
no thread is created per diagonal):

```
for each diagonal d (increasing key):
    workers_d = min(cpu_threads, members(d).size())
    parallel_for(members(d), workers_d, run_one_sb)   // fork-join
```

The pool cannot express a wait in the middle of a unit, which is why the
schedule stays phased per diagonal rather than lagged across diagonals (§6.1
item 2 stays unimplemented; it measured no need for it).  A one-member diagonal
runs inline with no parallel region, so that single unit keeps the candidate
loops; chunk assignment inside a diagonal is timing-dependent by design, and
only the per-unit sinks have to be reproducible.

**The candidate-level interaction.**  Phase B parallelises TX-type and
intra-mode candidates *inside* `run_one_sb` (`video_tx_search.cpp:813`,
`video_intra_search.h:308`), and a nested `parallel_for` falls back to serial
(`parallel_for.h:267`).  So an outer wavefront **suppresses** the inner
parallelism.  Measured on this host with the same binary, interleaved, medians:
`AV2ENC_WAVEFRONT=0` is inner-only / serial superblocks; `=1` is the N-way
wavefront.

| Point | inner-only | N-way | speedup |
|---|---:|---:|---:|
| 480x240, 60 f, t=2 | 5.353 s | 4.392 s | 1.22x |
| 480x240, 60 f, t=4 | 5.410 s | 3.974 s | 1.36x |
| 480x240, 60 f, t=8 | 5.092 s | 3.794 s | **1.34x** |
| 640x360, 4 f, t=8 | 0.561 s | 0.445 s | 1.26x |
| 1440x720, 24 f, t=8 | 22.188 s | 15.810 s | **1.40x** |

The outer schedule wins at every point, including 480x240 whose average
diagonal width is only 2.29.  The predicted "outer may be slower than today"
did not happen: the candidate loops on these clips are short (one to two
leaves per superblock, a handful of candidates each), so two concurrent
superblocks buy more than two concurrent candidates inside one.

**Rule.**  There is still a gate, because the wavefront only pays when a
diagonal carries at least two independent units:

```
use_wavefront =
    cpu_threads > 1
 && a persistent pool is active
 && wavefront_allowed()                     // not keyframe, no refmvbank,
                                            // no segmentation/delta-Q
 && average_diagonal_width >= 2.0
average_diagonal_width = sb_rows*sb_cols / (2*(sb_rows-1) + sb_cols)
```

2.29 (480x240) and 6.1 (1440x720) pass; a two-superblock-wide frame (width 1.0,
one unit per diagonal) keeps the serial walk so its candidate loops still get
the pool.  `--cpu-threads 1` takes the serial raster loop unconditionally.
`AV2ENC_WAVEFRONT=0|1` overrides the rule for A/B measurement only; both
schedules code identical bytes.

**End-to-end A/B vs main HEAD (stage 1)**, same host, interleaved, medians of
3-5, IVFs identical in every run:

| Workload | stage 1 | stage 3 | ratio |
|---|---:|---:|---:|
| 480x240 `--qp 106 --fast-decode`, t=1 | 6.498 s | 6.440 s | 0.99 (no change) |
| 480x240 `--qp 106 --fast-decode`, t=8 | 4.816 s | 3.655 s | **1.32x** |
| 1440x720 recipe, 24 f, t=1 | 33.250 s | 33.226 s | 1.00 (no change) |
| 1440x720 recipe, 24 f, t=8 | 21.578 s | 14.959 s | **1.44x** |

**Gates.**  Four 480x240 pins and the 1440x720 recipe byte-identical at
`--cpu-threads 1..8`; the Stage-0 dumps `cmp`-identical across thread counts and
equal to §9.1; 20 repeated 8-thread nightly runs give one md5;
`tools/refactor/check_wavefront_pins.sh` (pins + dumps),
`tools/refactor/check_thread_determinism.sh`, `gate_primitives`,
`gate_build_seam`, `gate_video_nightly`, `gate_video_qp_lambda`, and
`tools/test.sh` 10/10 all pass.  A skew-1 control (key `sbr + sbc`) built from
the same tree fails at 8 threads on the nightly pin ("P-frame 5: resolve_mode
failed") and its Stage-0 dump differs from the 1-thread dump at line 453, so the
schedule is load-bearing and the dump is sensitive to it.

## 10. Integrated result (merged base, 2026-11-17)

Merged on top of phase B (persistent pool + candidate parallelism) and phase A
(parallel gate runner). Measured on a 60-frame workload unless stated; the
interleaved A/B in §9.4 is the defensible causal number, because absolute times
on this host drift ~20% between sessions.

| Workload | Before NEON / threading | Now | Cumulative |
|---|---:|---:|---:|
| 480×240 QP106, 1 thread | 16.55 s | 7.40 s (7th: 6.95/7.40/7.43) | **2.2×** |
| 480×240 QP106, 8 threads | 7.97 s | 4.05 s (3.82/4.49/4.05) | **2.0×** (14.8 fps) |
| 1440×720 QP217, 8 threads | 148.0 s | 26.7 s (25.80/27.55) | **5.6×** (2.25 fps) |

Thread scaling is now **1.8×** at 480×240 and **2.2×** at 1440×720 (24 f A/B:
33.25 → 14.96 s), against 1.16×/1.36× before phases B and C. Inverting Amdahl at
n=8, the parallel fraction rose from **~16%/31%** to **~51%/63%**.

Gate suite: 50 gates, 351 s wall at `--gate-jobs 6`; `gate_wavefront_pins`
(5 pins × 2 thread counts + dump oracle) is 132 s of that.

Deliberately still serial under the wavefront: intra keyframes, `--video-refmvbank`,
segmentation/delta-Q — each for a stated state-ordering reason; enabling any of
them needs its own gate first (there is no segmentation gate today).
