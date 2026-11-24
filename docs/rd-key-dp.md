# RD key DP — full §5.20.3 tree DP on the key driver

Status: **Phase 1 KILL (measured).** The memoised tree DP does not beat the key
driver's reference-tree prune: on the 60-frame QP119 chained-B key it finds
**0 of 32** superblocks where the exact DP J is below the prune J, changes
24 of 32 layouts, and its own tree is **10.71 % worse** under the exact
objective.  The commit rule is `min(prune, DP)`, so the arm's aggregate key J
gain is **0.0000 %** against the 0.5 % wall.  Phase 2 was not run.

Branch `exp/key-dp`, worktree `../key-dp`, from main `d138b7a8`.  No push, no
merge.

## 0. Answer first

1. The P-frame full DP was killed at 0.155 % aggregate J on 5.59 % of
   superblocks (`docs/rd-joint-dp.md`).  The key driver
   (`seq_driver_decide_sb`) prunes the *same* expanded reference tree, and the
   key is ~65 % of the stream at 100 kbps, so this experiment asked whether the
   shape-reorientation freedom the prune lacks is worth more on an intra-only
   picture, where prediction is orientation-sensitive.
2. `AV2ENC_RD_KEY_DP` (default off, inert without the sequential key driver)
   replaces the key prune with

   ```text
   J(node) = min(leaf_J, sum J(children) + lambda * 2)
   ```

   over every legal §5.20.3 shape down to `AV2ENC_RD_DRIVER_MIN` (default 4),
   honouring spec-forced boundary splits and the `AV2ENC_RD_DRIVER_ACT`
   activity gate.  Each distinct node is priced once with the key driver's own
   leaf measure (same D, same calibrated R, same lambda), memoised from the
   superblock entry state; the chosen tree is then re-priced by the exact raster
   walk and committed only on a strict win over the prune.
3. **It does not pay.**  On the 60-frame QP119 chained-B key (32 priceable
   superblocks):

   | | J |
   |---|---:|
   | prune tree (the arm that commits today) | **385,154** |
   | DP proposal, exact raster re-price | 426,396 (**+10.71 %**) |
   | arm = `min(prune, DP)` | **385,154** (**0.0000 %** gain) |

   0 / 32 superblocks improve, 24 / 32 change layout, 0 fails.  The same
   measurement on the flat QP112 `--video-rd-preset best` key gives
   **387,940** vs **430,839** (+11.06 %), also 0 / 32 improved and 24 / 32
   changed.
4. The mechanism is the **measurement state, not the shape set**.  The prune
   measures a node's leaf in the reconstruction its already-decided preceding
   siblings built; the memoised DP prices every node from the *superblock entry
   state*.  For an intra leaf that state is missing the reconstructed
   above/left neighbours inside the superblock, so the non-root leaves are
   overpriced, and the bias grows with tree depth.  The DP therefore prefers
   coarser trees: on the head run its own tree's cold value is 475,790 against
   an exact re-price of 424,450 (12.1 % cold inflation), while the exact
   re-price of the *prune* tree reproduces the prune's own cost to the last
   bit (`AV2ENC_RD_KEY_DP_CHECK=1`, delta 0 on all 32 superblocks).  The exact
   guard rejects all 32 proposals.
5. **Phase-1 KILL.**  Aggregate key J gain 0.0000 % < 0.5 %; the layout-change
   wall (24/32 = 75 % ≥ 10 %) passes, so the J wall alone closes the line.  No
   Phase-2 proof (16-frame pinned `dav2d` / `avmdec`) and no rate-matched
   60-frame K were run.

## 1. What the DP adds over the key prune

The prune walks the expanded reference tree.  `seq_build_uniform()` splits each
node along its long axis, so from 64x64 the expansion is
64x64 `HORZ` → (64x32) → 64x32 `VERT` → (32x32) → … and reaches 16x16, 8x8,
8x4, 4x4 only through that one orientation chain.  The prune may keep or
collapse any node of that tree, but an internal reference node has a fixed
orientation and no shape can cross it.  The DP is a strict superset of those
shapes:

| node | prune (uniform expansion) | DP |
|---|---|---|
| 64x64 | NONE, HORZ→64x32 | NONE, HORZ→64x32, VERT→32x64 |
| 64x32 | NONE, VERT→32x32 | NONE, HORZ→64x16, VERT→32x32 |
| 32x64 | not in the expansion | NONE, HORZ→32x32, VERT→16x64 |
| 32x32 | NONE, HORZ→32x16 | NONE, HORZ→32x16, VERT→16x32 |
| 32x16 | NONE, VERT→16x16 | NONE, HORZ→32x8, VERT→16x16 |
| 8x8 | NONE, HORZ→8x4 | NONE, HORZ→8x4, VERT→4x8 |
| 8x4 | NONE, VERT→4x4 | NONE, VERT→4x4 |
| … | … | … down to `AV2ENC_RD_DRIVER_MIN` |

Each shape is a real §5.20.3 partition for the geometry: the candidate set is
`partition::init_allowed_partitions()` (boundary, aspect-ratio and chroma
rules), spec-forced boundary partitions are honoured, `can_split()` applies the
`AV2ENC_RD_DRIVER_ACT` activity screen to optional sub-16 splits, and a split
whose child would cross the driver floor is not a candidate.

## 2. The solve

The leaf J is the existing key driver measurement
(`seq_driver_measure_leaf()`): the BA-shaped key D (eff-perc key low-band CSF +
optional HF/edge terms + weighted chroma SSE) plus `lambda * R` with the
calibrated §8.3 plan rate and the driver's
`key_tx_rd_lambda * AV2ENC_RD_KEY_TX_SCALE * AV2ENC_RD_DRIVER_LAMBDA` lambda.
The partition term is the driver's constant `lambda * 2` bits per internal
symbol.

The shape search is a memoised DP: each distinct node
`(mi_row, mi_col, size, depth, hasChroma, chromaOffset, inherited ChromaMi)`
is measured once, from the superblock entry state, so a 480x240 key (32
superblocks) costs 52,120 leaf trials / 52,190 nodes at the default floor 4
instead of an exponential shape enumeration — about 2.7x the prune's own
~19.7k measured leaves.  Every measurement restores the entry snapshot, so the
DP value is a well-defined function of the node.

The DP value is only a proposal.  The chosen tree is re-priced by the exact
raster walk from the same entry state (each leaf in decode order, with the
partition term), and the prune's tree — whose `chosen.cost` is exactly that
same walk for the tree that commits today — is replaced only on a **strict**
cost win.  `AV2ENC_RD_KEY_DP_COMMIT=0` (default) is the Phase-1 instrument:
the prune still commits, so the stream is byte-identical.  `=1` lets a strict
DP win commit through the normal `collect_node()` path.

Tie-break is deterministic: search order NONE, HORZ, VERT, SPLIT; a later
shape replaces an earlier one only on a strict improvement, so ties keep the
coarser first candidate.  The DP is a pure function of the entry state, so the
same stream always proposes the same trees.

## 3. Gates

| Variable | Default | Meaning |
|---|---|---|
| `AV2ENC_RD_KEY_DP` | off | full key tree DP; inert unless the sequential key driver is armed (`AV2ENC_RD_DRIVER`, `--video-rd-driver` or `--nightly-b`) |
| `AV2ENC_RD_KEY_DP_COMMIT` | 0 | 1 = a strict DP win commits; 0 = Phase-1 instrument (bytes unchanged) |
| `AV2ENC_RD_KEY_DP_CHECK` | 0 | 1 = re-price the prune tree with the same exact walk every superblock and print the delta (accounting check; inert for the stream) |

The `rd-key-dp` report shares the `rd-driver` gate's counters and is printed at
process exit.  No CLI, preset, CMake or pin edit.  `AV2ENC_RD_KEY_DP=1` on the
flat `--fast-decode` path (no driver) is inert and reproduces the pins (§5).

## 4. Phase 1 — instrumentation

All runs use the release build, `--cpu-threads 8`, `AV2ENC_RD_KEY_DP=1` and
`AV2ENC_RD_KEY_DP_COMMIT=0`; only the key picture enters the key driver, so the
DP sees 32 superblocks (480x240).  `J` is the exact raster re-price of the
prune tree and of the DP tree from the same superblock entry state.  `arm` is
`min(prune, DP)`, the tree the commit rule would choose.

```sh
# Phase 1, chained-B key (60 f, QP119)
env AV2ENC_RD_KEY_DP=1 AV2ENC_RD_KEY_DP_COMMIT=0 \
  av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/keydp_b119.ivf \
    --video --frames 60 --qp 119 --cpu-threads 8 --video-gops 7 --nightly-b

# Phase 1, flat key (60 f, QP112, --video-rd-preset best)
env AV2ENC_RD_KEY_DP=1 AV2ENC_RD_KEY_DP_COMMIT=0 \
  av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/keydp_flat112.ivf \
    --video --frames 60 --qp 112 --cpu-threads 8 --video-rd-preset best

# accounting check: re-price the prune tree with the DP's exact walk
env AV2ENC_RD_KEY_DP=1 AV2ENC_RD_KEY_DP_CHECK=1 \
  av2enc testdata/nasa_480x240_30fps_60f.y4m -o /tmp/keydp_check.ivf \
    --frames 1 --qp 112 --cpu-threads 8 --video-rd-preset best
```

| run | priceable SBs | DP J < prune J | layout changed | J prune | J DP proposal | arm gain |
|---|---:|---:|---:|---:|---:|---:|
| **B QP119, 60 f** | 32 | **0 (0.00 %)** | **24 (75.00 %)** | **385,154** | 426,396 | **0 (0.0000 %)** |
| flat QP112 `best`, 60 f | 32 | **0 (0.00 %)** | **24 (75.00 %)** | **387,940** | 430,839 | **0 (0.0000 %)** |
| B QP116, 16 f head | 32 | 0 (0.00 %) | 24 (75.00 %) | 383,935 | 424,450 | 0 (0.0000 %) |

The DP proposal is **+10.71 %** (B QP119), **+11.06 %** (flat QP112) and
**+10.55 %** (B 16 f head) above the prune under the exact objective.  The
`min` guard therefore contributes exactly nothing: the arm is the existing
prune on every superblock.  Leaf histograms (short-side buckets
4 / 8 / 16 / 32 / 64+) and the per-node counters are in the `rd-key-dp` line.
On the B QP119 run `leaves=72643` is the driver total: the prune's ~19.8k
measured leaves, the DP's `leaf_trials=52120` and the exact re-price of its
proposal.  The DP's chosen trees there hold **756** leaves against the prune's
**1,599** (`dp=454/249/36/9/8` vs `prune=1184/374/24/9/8`), i.e. the DP drops
730 of the 4-px-class leaves (the 16-f head is the same shape: 784 vs 1,581,
`491/230/45/10/8` vs `1145/393/26/9/8`).

`AV2ENC_RD_KEY_DP_CHECK=1` on the flat QP112 key prints
`prune=<J> reprice=<J> delta=0.0` for all 32 superblocks: the DP's exact walk
is the same objective the prune sums, so the comparison and the commit rule
are aligned (the DP is not being rejected by a different price).

### Winning shapes

**Empty.**  No superblock improves, so the "winning shapes absent from the
prune tree" table has no rows.  The DP does return a different tree on 24/32
superblocks — mostly coarser (756 leaves against the prune's 1,599 at QP119) —
and it does reach shapes the expansion never contains (the wrong rect
orientation at the superblock top, 64x64 `VERT`, 64x32 `HORZ`, thin
64x16/32x16 leaves are all in its candidate set), but every proposal prices
*above* the prune tree under the exact walk.  The gate is on the objective, not
on shape availability.

### Why the proposal loses

The key leaf cost is measured on the reconstruction the leaf builds, and an
intra leaf's prediction reads reconstructed above/left neighbours.  The prune
is sequential: at a split node it measures the first child, keeps its recon,
then measures the second child in that recon, and re-encodes the winner.  The
memoised DP measures every node from the superblock entry state, where the
preceding siblings inside the same superblock have not been reconstructed —
the same approximation the P DP made, but inter prediction does not depend on
the current reconstruction while intra prediction does.

The bias is visible in the DP's own numbers: its cold value for its chosen tree
is **475,790** on the head run while the exact raster re-price of the *same*
tree is **424,450** (12.1 % inflated); the full-clip proposals show the same
inflation (474,528 vs 426,396 on B QP119, 476,904 vs 430,839 on flat QP112).
The cold price grows with leaf count, so the DP systematically under-splits;
24/32 layouts change mostly toward coarser trees and every one loses exactly
the J the prune's sequential measurement already captured.  This is why the P
result (0.155 %, prune already at ~1.1 leaves/superblock) does not transfer:
the key's prune keeps ~49 leaves per superblock (1,581 leaves over 32
superblocks on the head run) and their context is the thing the cold DP cannot
see.

What would change the verdict is a **flowing-context DP**: price each node in
the reconstruction of the tree that leads to it (a context-dependent
recurrence, no simple memo), or restrict the DP to the superblock top where the
entry state is the true context.  Both are a redesign of the recurrence, not a
shape-set change, and were outside this time box.

## 5. Conformance

* `env AV2ENC_RD_KEY_DP=1` flat `--fast-decode` QP106 60 f: **25,008 B /
  `0639b43630abb454f7be5ac494d2efb4`**; QP110: **23,690 B /
  `1b99127a4cd1d386fa5600fb63c57a83`** — byte-identical, the gate is inert
  without the key driver.
* `env AV2ENC_RD_KEY_DP=1` (commit 0) 16 f chained B QP116: **19,723 B /
  `505430de620af361a18b42503543dbda`** — the instrument does not change the
  committed stream.
* `env AV2ENC_RD_KEY_DP=1` (commit 0) 60 f chained B QP119: **24,845 B /
  `40de8b687d929652a312407764227398`** — the published joint arm's stream.

> **Historical (2026-11-21), not re-runnable here.** These two readings are
> from the `exp/key-dp` branch (§0, "No push, no merge"), which was never
> merged: `AV2ENC_RD_KEY_DP` has zero hits in `src/`, `tests/` or `tools/` and
> never appeared in this repository's `src/` history, and no `exp/key-dp`
> branch or worktree exists to build it. The values above are what that branch
> measured against the then-current chain, and they are internally consistent
> (instrument off == instrument on at 19,723). Do not "update" them: today's
> chain yields 17,222 B (`6ec51ab4` re-keyed every `--nightly-b` stream,
> `BUG-GATES-002`) but there is no instrument left to compare it against, and
> `gate_video_joint_plan.sh` — now 11/11 — verifies main's stream, not this
> branch's byte-neutrality claim.
* `env AV2ENC_RD_KEY_DP=1` (commit 0) 60 f flat QP112 `--video-rd-preset best`:
  **25,007 B / `9a36aeb7c02733342e4ec5db300e4d6d`** — the promoted `best` pin.
* `AV2ENC_RD_KEY_DP=0` and unset are the existing paths; no byte moved.  The
  TIP field guard 3, the compound exact-rate, the NEWMV-off state, the CLI,
  the presets and the pins are untouched; the DP changes only which partition
  shapes are proposed, and `min(prune, DP)` never loses.
* `ctest --test-dir build-release` 12/12.
* No `COMMIT=1` stream was produced: no superblock improves, so the commit
  branch is unreachable on both Phase-1 carriers.  That is the Phase-1 kill,
  not an untested path in the report.

## 6. Files

| file | role |
|---|---|
| `src/pipeline/video/video_frame_internal.h` | `AV2ENC_RD_KEY_DP` gate/config, `rd-key-dp` counters and shape report |
| `src/pipeline/video/video_frame_decision.cpp` | `seq_driver_decide_sb`'s memoised shape DP, exact re-price, accounting check |
| `src/pipeline/video/video_joint_plan.h` | shared §5.20.3 shape key (`shape_slot` / `shape_words`) reused by the report |
| `docs/rd-key-dp.md` | this report |

## 7. Verdict

**Phase 1 KILL.**  Aggregate key J gain **0.0000 %** against the 0.5 % wall on
the required 60-frame QP119 chained-B key, and **0.0000 %** on the flat QP112
key.  0 / 32 superblocks improve on both, although 24 / 32 change layout, so
the failure is not a missing shape: it is the cold measurement state of the
memoised recurrence.  The DP is not promoted; the opt-in code stays as the
measured negative result.  No Phase-2 proof and no rate-matched 60-frame K
were run, and the standing point stays **87.570**.
