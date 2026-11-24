# RD joint plan — full exact tree DP on the chained-B carrier

Status: **Phase 1 KILL (measured).** The prune already captures the shape pool
on this carrier.  The full tree DP is implemented as an opt-in sub-gate
(`AV2ENC_RD_JOINT_DP`, default off) and instrumented; on the full 60-frame
clip it finds a J improvement on 5.6 % of superblocks for an aggregate
**0.155 %** J gain, and on the fully instrumented 16-frame head 1.1 % of
superblocks for **0.061 %**.  The gate was 0.5 % aggregate J **and** 10 % of
superblocks changed.  Phase 2 (the 16-frame proof and the 60-frame
rate-matched point) was not run.

Branch `exp/joint-dp`, worktree `../joint-dp`, from main `36bf35c6` (the joint
plan + the two-pass CPU batch).  No push, no merge.

## 0. Answer first

1. The current joint solve is a **prune of the reference tree**: it expands
   each reference leaf to the uniform 8x8 tree and then, at every node of that
   one tree, keeps the split children or collapses the node.  Inside a
   reference *leaf* that expansion can refine or collapse freely, but at a
   reference **internal** node the orientation is fixed and no shape can cross
   it: the solve cannot choose VERT where the expansion built HORZ, nor a
   different rect combination across a reference partition.
2. The full DP replaces that with

   ```text
   J(node) = min(leaf_J, sum J(children) + lambda * partition_bits)
   ```

   over every legal §5.20.3 shape — NONE, HORZ, VERT, and SPLIT where §9.2
   defines it — down to the 8x8 inter coding-block floor.
3. **It does not pay.** On the full 60-frame clip (1,647 priced superblocks)
   the DP arm's exact J is **11,156,979** against the joint prune's
   **11,174,334** — a **0.155 %** gain on 92 / 1,647 superblocks (5.6 %).  On
   the fully instrumented 16-frame head it is **1,833,476** against
   **1,834,587** — **0.061 %** on 5 / 450 superblocks (1.1 %).  Both Phase-1
   walls fail.
4. The mechanism is the reason, not the search: the current prune already
   collapses the head to one 64x64 leaf on most superblocks and the tail to
   ~1.1 leaves per superblock at this QP.  The set of shapes that a different
   rect orientation can still win is small and worth ~0.15 % J, against the
   ~2.5 % J the joint solve's own history says is needed for +0.15 K
   (4.85 % J bought +0.286 K on this carrier).

## 1. What the DP adds over the prune

The prune walks the *expanded* reference tree.  At a 64x64 reference leaf that
tree's first split is always HORZ (64x32), then VERT (32x32), and so on; the
prune may collapse any node but cannot choose VERT at 64x64, HORZ at a 64x32
node, or an asymmetric combination across a reference *internal* partition.
The DP is a strict superset of those shapes:

| node | prune (uniform expansion) | DP |
|---|---|---|
| 64x64 | NONE, HORZ→64x32 | NONE, HORZ→64x32, VERT→32x64 |
| 64x32 | NONE, VERT→32x32 | NONE, HORZ→64x16, VERT→32x32 |
| 32x64 | NONE, HORZ→32x32 | NONE, HORZ→32x32, VERT→16x64 |
| 32x32 | NONE, HORZ→32x16 | NONE, HORZ→32x16, VERT→16x32 |
| … | … | … down to 8x8, aspect ≤ 8 |

Each shape is a real §5.20.3 partition for the geometry: the candidate set is
`partition::init_allowed_partitions()` (which already carries the boundary,
aspect-ratio and chroma rules), spec-forced boundary partitions are honoured,
and a split whose child would cross the 8x8 floor is not a candidate.  SPLIT
is included only where `partition_subsize()` defines it (128/256 roots).

## 2. The solve

The leaf J is the existing joint exact leaf cost (`joint_p_measure_leaf`):
the eff-perc inter D + `lambda * R_exact` through the interval model and the
live inter-syntax probe, with the compound/TIP pricing and the B compound
exact-rate unchanged.  The partition term is the driver's constant
`lambda * 2` bits per internal symbol, the same term the prune uses.

The shape search is a **memoised DP**: each distinct §5.20.3 node
`(mi_row, mi_col, size)` is measured once, from the superblock's entry state,
so a 64x64 superblock costs **~196 leaf trials** (225 dyadic nodes minus the
boundary-forced prunes) instead of the exponential shape enumeration
(~2,200).  Every measurement restores the entry snapshot, so the DP value is
a well-defined function of the node; a leaf is the same trial the prune
performs.

The DP value is only a proposal.  The chosen tree is re-priced by the exact
raster `joint_p_measure_tree()` walk from the same entry state (context
flowing through the committed layout), and the existing reference guard
decides the commit: `min(reference, joint prune, DP)` under the exact J.
`AV2ENC_RD_JOINT_DP_COMMIT=0` is the Phase-1 instrument (the joint prune still
commits, so the stream is byte-identical); `=1` adds the DP tree to the
candidate set.

Tie-break is deterministic: search order NONE, HORZ, VERT, SPLIT; a later
shape wins only on a **strict** cost improvement, so ties keep the coarser
first candidate.  The DP itself is a pure function of the entry state, so the
same stream always picks the same shapes.

## 3. Gates

| Variable | Default | Meaning |
|---|---|---|
| `AV2ENC_RD_JOINT_DP` | off | DP sub-gate (inert unless `AV2ENC_RD_JOINT_P` is armed) |
| `AV2ENC_RD_JOINT_DP_COMMIT` | 0 | 1 = the DP tree can be committed; 0 = measure only (Phase 1) |
| `AV2ENC_RD_JOINT_DP_SAMPLE` | 0 | 0 = every superblock; N > 0 = every Nth, rotating per picture |

The `rd-joint-dp` line shares the `AV2ENC_RD_JOINT_P_STATS` gate.  No CLI,
preset or pin change; the flat and gate-off paths never enter the DP.
`AV2ENC_RD_JOINT_DP=1 --fast-decode` reproduces the flat pins byte for byte
(§6).

## 4. Phase 1 — instrumentation

Two measurements, all with `--nightly-b`, `--cpu-threads 8`, QP116/119 and
the exact compound/TIP recipe.  `AV2ENC_RD_JOINT_DP_COMMIT=0`, so the output
stream is the current joint arm's (the 16-frame pin and the 60-frame QP119
pin both reproduce, §6); only the DP's exact J is measured on top.

`J` is summed over the sampled superblocks with the reference/joint/DP trees.
`base = min(reference, joint)`; `arm = min(reference, DP)`; both arms share
the same reference guard and the same entry state, so `gain` is the J the DP
would add to the current solve.

| run | pictures | sampled SBs | DP improved | layout changed | J base | J arm | gain |
|---|---|---:|---:|---:|---:|---:|---:|
| head, full DP | f1-15 (16 f) | 450 | 5 (**1.11 %**) | 8 (1.78 %) | 1,834,587 | 1,833,476 | **1,111 (0.061 %)** |
| **clip, full DP** | f1-59 (60 f) | **1,647** | **92 (5.59 %)** | **131 (7.95 %)** | **11,174,334** | **11,156,979** | **17,354 (0.155 %)** |

(A preliminary every-4th-superblock sample over the same 60-frame clip gave
0.244 % on 413 superblocks, so the full run is not a sampling artefact.)

The measurement excludes superblocks whose reference tree the exact leaf cost
cannot price (30 / 480 in the head run; 209 / 1,856 in the 60-frame run — a
superblock is excluded when *any* reference leaf is one of the known
compound-residual refusals).  Those keep the reference layout in both arms.
The 60-frame run arms 1,856 superblocks (58 coded P pictures; one P picture
is a TIP output with no tile).

Per-superblock detail (head run): 5 SBs improve, 8 change layout, 0 DP
failures; the DP's own tree is 1,054 J below the joint tree, and the
reference guard removes the rest (`arm` is 57 J below `dp`).  On the full
clip the DP's standalone tree is 11,252,190 — **0.70 % worse** than the joint
tree — because it wins on 92 SBs and loses on the other 1,555; the per-SB
reference guard converts that into the 0.155 % net.  That is the shape of the
result: a small winner set filtered by the guard, not a uniformly better
tree.

The memoised shape value is **not** what caps the gain: on the head run the
DP's internal `sum J(children)` (1,834,429) is *above* the exact raster
re-price of the same tree (1,833,533), and on the full clip it is 11,271,702
vs 11,252,190.  Measuring each node from the superblock entry state is
therefore mildly conservative here, not optimistic; a flowing-context DP
would choose from the same shape pool and could not recover more than the
exact walk already found.

The Phase-1 runs use the serial walk because the batch's pass-1 worker
counters are deliberately dropped (they describe trials the serial walk never
performs).  The DP itself runs inside batch pass 1 unchanged, and the commit
smoke in §6 confirms the two-round batch reproduces the serial commit
byte-for-byte.

## 5. Winning shapes

Shapes used by the DP on the improved superblocks that the current joint tree
does **not** contain at that superblock, summed over the full 60-frame run.
`all` counts every DP node of that shape on an improved superblock;
`not in joint` is the subset absent from the joint tree, `not in reference`
the subset absent from the gate-off tree as well:

| shape | all | not in joint | not in reference |
|---|---:|---:|---:|
| 64x64 HORZ | 67 | 21 | 23 |
| 64x64 VERT | 25 | 17 | 17 |
| 64x32 NONE | 57 | 23 | 50 |
| 64x32 HORZ | 56 | 46 | 29 |
| 64x32 VERT | 10 | 5 | 5 |
| 64x16 NONE | 54 | 48 | 53 |
| 64x16 HORZ | 3 | 3 | 3 |
| 64x16 VERT | 4 | 1 | 1 |
| 64x8 NONE | 3 | 3 | 3 |
| 32x64 NONE | 18 | 17 | 17 |
| 32x64 HORZ | 12 | 10 | 8 |
| 32x64 VERT | 6 | 5 | 4 |
| 32x32 NONE | 20 | 17 | 17 |
| 32x32 HORZ | 9 | 3 | 3 |
| 32x32 VERT | 9 | 8 | 8 |
| 32x16 NONE | 11 | 3 | 10 |
| 32x16 HORZ | 1 | 1 | 1 |
| 32x8 NONE | 1 | 1 | 1 |
| 16x32 NONE | 10 | 8 | 8 |
| 16x32 HORZ | 1 | 1 | 1 |
| 16x32 VERT | 1 | 1 | 1 |
| 16x64 NONE | 6 | 5 | 4 |
| 16x16 HORZ | 1 | 0 | 1 |
| 16x8 NONE | 1 | 0 | 1 |
| 8x32 NONE | 1 | 1 | 1 |

Two families carry almost all of it: **the wrong orientation at the top of
the superblock** (64x64 VERT where the expansion built HORZ; 64x32 HORZ where
it built VERT) and **thin horizontal rects** (64x16, 64x8, 32x16, 16x32),
which the uniform expansion only reaches as intermediate nodes of a square,
never as a leaf choice with the other side kept coarse.  The head run's five
wins are the same family (16x32, 16x64, 64x16, 64x32, 32x32 VERT, 32x64 VERT,
64x32 HORZ, 64x64 HORZ).

The leaf histogram of the winning trees shows the direction: the DP moves
16x16 leaves to 32x32 (`dp 0/12/113/116/0` vs `joint 0/38/230/92/10` over
buckets 8/16/32/64).  It also **drops the 64+ bucket to zero** on those
superblocks: the DP is not a uniform coarsening, it is a re-shaping of the
1-9 leaves the joint tree already keeps there.

## 6. Conformance

* `AV2ENC_RD_JOINT_DP=1 --fast-decode` QP106 60 f: **25,008 B /
  `0639b43630abb454f7be5ac494d2efb4`**; QP110: **23,690 B /
  `1b99127a4cd1d386fa5600fb63c57a83`** — byte-identical, the gate is inert
  without the joint plan.
* `AV2ENC_RD_JOINT_DP=1` (commit 0) 16 f joint QP116: **17,222 B /
  `2a7fbe710542a89c3a9d2c93fab94101`** — the Phase-1 overlay does not change
  the committed stream.  (Re-recorded 2026-11-25 after `6ec51ab4` re-keyed the
  chain, `BUG-GATES-002`; the 11-21 reading was 19,723 B / `505430de…`.)
* `AV2ENC_RD_JOINT_DP=1` (commit 0) 60 f joint QP119, both the stride sample
  and the full run: **22,576 B /
  `4b622b15a0715820114765e17259868f`** — the published joint arm's stream
  (re-recorded 2026-11-25, `BUG-GATES-002`; the 11-21 reading was 24,845 B /
  `40de8b68…`).
* Commit-path smoke (not the Phase-2 proof): `AV2ENC_RD_JOINT_DP=1
  AV2ENC_RD_JOINT_DP_COMMIT=1` 16 f QP116 gives **19,716 B /
  `ba8f880361de49b9b718df799f993b5f`**, pinned `dav2d --strict 1` decodes it
  sample-exactly (`--video-recon` equals the decoded payload) and `avmdec`
  reports 16/16 recon ok.  It exercises the thin 64x16/64x8 leaves the DP
  proposes; the J gain on that picture is 0.030 % (see §4), so the stream is
  only 7 B smaller.
* The same 16 f commit through the two-round batch
  (`AV2ENC_RD_JOINT_BATCH=8 AV2ENC_RD_JOINT_BATCH_CHAIN=1
  AV2ENC_RD_JOINT_BATCH_ROUNDS=2`) is **byte-identical** to the serial commit
  (`ba8f8803…`, 19,716 B) and passes the same pinned-dav2d and `avmdec`
  checks.  The DP runs inside batch pass 1 unchanged; the K-safe speed path
  composes.
* `AV2ENC_RD_JOINT_DP=0` and unset are the existing paths; no byte moved.
* `ctest --test-dir build-release` 12/12.
* The TIP field guard 3, the compound exact-rate and the NEWMV-off state are
  untouched; the DP changes only which partition shapes are proposed.

## 7. Files

| file | role |
|---|---|
| `src/pipeline/video/video_joint_plan.h` | DP gate/config, shape key and the `rd-joint-dp` counters |
| `src/pipeline/video/video_frame_decision.cpp` | `joint_p_decide_sb`'s memoised shape DP, exact re-price and the Phase-1 overlay |
| `docs/rd-joint-dp.md` | this report |

## 8. Verdict

**Phase 1 KILL.**  Aggregate J gain **0.155 % (full 60 f)** / **0.061 %
(16 f head)** against the 0.5 % wall, and **5.59 % / 1.11 %** of superblocks
improved against the 10 % wall.  The prune's collapse pool is essentially the
shape pool on this carrier: once the reference tree is expanded to 8x8 and
pruned, the remaining shape freedom is worth 17,354 J on the full clip
(0.155 %), not the ~2.5 % J the +0.15 K gate needs.

Projecting that with the joint solve's own measured slope (4.85 % J bought
+0.286 K from 87.284 to 87.570) gives **~+0.01 K**, i.e. ~87.58 against the
87.720 gate.  This is a projection, not a measurement: no 60-frame
rate-matched point was encoded, because the Phase-1 gate closed first.

The DP is not promoted; the opt-in code stays as the measured negative
result.  No 16-frame Phase-2 proof and no 60-frame rate-matched K were run.
The commit-path 16-frame smoke in §6 exists only to prove the `COMMIT=1` code
path produces a decodable stream.

What would change the verdict: a carrier where the joint tree still emits
several leaves per superblock *after* the prune (a lower QP, a finer texture
clip, or a min-block-4 intra search) — there the DP's extra orientations have
something to bite on.  On this 240p/100 kbps point the current solve is
already at the shape floor.
