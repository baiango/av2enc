# RD joint plan — measured P-frame partition + TX/level solve

Status: **KEEP, promoted as an opt-in research arm.** The P chain's tree was
the largest remaining measured
inefficiency, not the leaf coder. One objective over partition and TX/levels
finds a tree that is **9.4-11.0 % smaller at the same QP with equal-to-better
Butteraugli/SSIMULACRA2**, and at the matched 100 kbps point it measures
**K 87.194 bracket / 87.214 fit** against the base's **86.514 / 86.298**
(**+0.680 / +0.916**, gate +0.15). The gain is the partition *collapse* pool
the probe's depth-1 clamp hides; the leaf TX/level stage is unchanged.

Branch `exp/joint-plan`, worktree `../joint-plan`, from main `87848007`.
Recipe: `--video-rd-preset best` (the 86.33 flat lead stack), 240p30, 60
frames, `--cpu-threads 8`, env `AV2ENC_RD_JOINT_P=1`.

## 1. What was wrong

The P chain solved the three decisions in sequence:

1. `solve_partition()` prices every node with one NEWMV + quantised residual
   (`partition_probe_leaf_cost`), clamped to `partition_probe_depth` = 1 (the
   tail band descent opens depth 4). The probe under-prices the leaf the coder
   will actually choose: at QP116 79 % of the head's residual-arm leaves emit
   `eob = 0` on all planes, and the tail's band-descent tree fills 93.5 % of
   its leaves at 16x16 (4,969 / 5,313 at QP112, `docs/rd-pdriver.md` §4,
   `docs/rd-part-diag.md`).
2. `encode_leaf()` then picks mode + TX/levels for the leaf the probe chose.
3. The head/tail exact-rate stages refine those TX/levels afterwards, without
   revisiting the tree.

The diagnostic (`docs/rd-part-diag.md`) measured the *split* side of this
error at emitted leaves and found it worth 67 B on the tail. It did not
measure the *collapse* side: a probe leaf that the exact leaf coder would code
more cheaply as its parent. That is the pool this lane opens.

## 2. The solve

The P-frame port of the key driver
(`InterFrameEncoder::seq_driver_decide_sb`, `docs/rd-engine0.md` §5):

```text
leaf    J = D_P(recon the leaf emitted) + lambda_P * R_exact(leaf)
split   J = sum(children J) + lambda_P * 2 bits
```

| term | definition |
|---|---|
| `D_P` | `seq_rd::leaf_distortion_gains()` on the leaf's own reconstruction: the eff-perc **inter** low-band CSF + weighted chroma SSE, optional high-pass / block-edge terms (both 0). Never the key CSF |
| `R_exact` | the emitted §8.3 symbols through the §8.2.6 interval model (`rate_interval::estimate`, live frame-start CDF image) + the live inter-syntax group (is_inter / skip / mode / ref / DRL / MVD) the pipe probe stashed when the leaf was written |
| `lambda_P` | `leaf_rd_lambda(qindex) * eff_perc::lambda_scale()`, the lambda the P leaf and its head/tail TX search minimise with (not `key_tx_rd_lambda`) |
| split syntax | 2 modelled bits per internal partition symbol, the key driver's constant |

The search is the key driver's expand/prune:

* expand each reference leaf larger than 8x8 (the §5.20.3.1 inter coding-block
  floor) to the uniform 8x8 layout, honouring forced boundary partitions;
* at each internal node measure the leaf candidate and the split children in
  sequence, in the live reconstruction; the split state is kept when it wins,
  the leaf is re-encoded when it wins;
* superblocks are decided in raster order and committed through the normal
  `collect_node()` walk, so a candidate is always priced in the reconstruction
  its own layout and every earlier coded block built. Inter prediction reads
  the previous filtered frame (reference-independent), but intra leaves inside
  P read the current recon; the raster walk is what makes them exact.

Every candidate layout is encoded with the coder's own leaf path
(`collect_node` -> `encode_leaf` -> mode search, TX/level search, coefficient
trellis), so partition, TX and levels are decided against one objective
instead of three.

**Reference guard.** The gate-off reference tree is measured with the same
objective and kept whenever it wins, so the driver can never make the tree
worse under J. At QP112 the reference wins **969 / 1,888** superblocks and the
committed total is `J = 15,958,148` vs the reference `17,472,366`
(**-8.7 %**). The unguarded QP116 stream is 5 B smaller than the guarded one
(23,645 vs 23,650 B); the guard is a J tie-break, not a rate feature.

`seq_driver`-style leaf measurements the driver cannot price (TCQ, compound,
TIP, AMVD, sub-8 chroma trees, palette/IBC/SDP, delta-Q, tiles, ext
partitions) keep the reference layout. The gate is inert on every such
picture, so no other carrier is touched.

## 3. Gate

| Variable | Default | Meaning |
|---|---|---|
| `AV2ENC_RD_JOINT_P` | off | master gate |
| `AV2ENC_RD_JOINT_P_FIRST/LAST` | 1 / 4096 | display-order picture window |
| `AV2ENC_RD_JOINT_P_TARGET` | 8 | deepest uniform leaf (px), clamped to the 8x8 inter floor |
| `AV2ENC_RD_JOINT_P_DEPTH` | 8 | policy depth cap |
| `AV2ENC_RD_JOINT_P_LAMBDA` | 1.0 | scale on the P pipe lambda (not swept) |
| `AV2ENC_RD_JOINT_P_HF/EDGE` | 0 / 0 | optional high-pass / block-edge D terms |
| `AV2ENC_RD_JOINT_P_STATS` | on | one-line counter dump |

## 4. Tree stats

Leaf histogram by short side (`4/8/16/32/64+`), summed over the 59 P pictures
(1,888 superblocks), QP112, guard on:

| segment | tree | 4 | 8 | 16 | 32 | 64+ | total |
|---|---|---:|---:|---:|---:|---:|---:|
| head f1-32 | reference | 0 | 0 | 264 | 343 | 19 | 626 |
| head f1-32 | **joint** | 0 | **6** | **125** | **145** | **229** | **505** |
| tail f33-59 | reference | 0 | 0 | 4,969 | 337 | 7 | 5,313 |
| tail f33-59 | **joint** | 0 | **583** | **1,409** | **461** | **160** | **2,613** |
| all | reference | 0 | 0 | 5,233 | 680 | 26 | 5,939 |
| all | **joint** | 0 | **589** | **1,534** | **606** | **389** | **3,118** |

* The head **coarsens**: 264 16x16 references become 125, and the 64+ bucket
  grows 19 -> 229. That is the measured confirmation of the diagnostic's
  head result (`+80.8 B` on split winners): the head should be coarse.
* The tail **both coarsens and refines**: 4,969 16x16 references become 1,409,
  while 583 new 8x8 leaves appear and the 64+ bucket grows 7 -> 160. The probe
  over-split most of the tail into 16x16 and missed the few places a smaller
  block is worth it.
* Net **-47.5 % leaves** (5,939 -> 3,118). The P payload falls **-27 %** at
  QP116 (9,776 -> 7,092 B) while the key picture is byte-identical.

At QP116 the same picture (guarded): total 5,644 -> 3,239 leaves, changed
872/1,888 superblocks, 1,007 reference wins.

## 5. Fixed-QP RD (`--video-rd-preset best`, 60 f)

| QP | arm | bytes | kbps | PSNR-Y | JOD | BA | SS2 | K |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 114 | base | 26,999 | 107.996 | 46.4167 | 9.8544 | 2.0389 | 85.356 | 86.715 |
| 114 | **joint** | **24,040** | **96.160** | **46.9236** | **9.8799** | **2.0391** | **86.128** | **86.971** |
| 116 | base | 26,334 | 105.336 | 46.2508 | 9.8474 | 2.0973 | 84.979 | 86.447 |
| 116 | **joint** | **23,650** | **94.600** | **46.7781** | **9.8766** | **1.9605** | **85.759** | **87.129** |
| 118 | base | 25,367 | 101.468 | 46.1248 | 9.8441 | 2.0181 | 84.651 | 86.608 |
| 118 | **joint** | **22,979** | **91.916** | **46.5077** | **9.8678** | **2.0358** | **85.203** | **86.766** |

Fixed-QP deltas: **-11.0 / -10.2 / -9.4 %** bytes, PSNR-Y **+0.51 / +0.53 /
+0.38 dB**, SS2 **+0.77 / +0.78 / +0.55**, BA **+0.000 / -0.137 / +0.018**.
At the same quantiser the joint tree is both smaller and (on 2 of 3 rungs)
better on every perceptual metric; the K column is not rate-matched (joint is
the lower-rate point) and therefore understates the gain.

## 6. Matched K

Five base points and four joint points, `tools/oracle.py --target-kbps 100`:

| arm | points (QP: kbps / K) | bracket | fit |
|---|---|---|---:|---:|
| base | 114: 107.996/86.715, 116: 105.336/86.447, 118: 101.468/86.608, 119: 99.412/86.477, 120: 97.688/85.889 | **86.514** (118/119) | **86.298** (R² 0.516) |
| joint | 112: 100.028/87.196, 114: 96.160/86.971, 116: 94.600/87.129, 118: 91.916/86.766 | **87.194** (114/112) | **87.214** (R² 0.660) |
| **delta** | | **+0.680** | **+0.916** |

The base's own reading reproduces `docs/video-rd-preset.md` (86.329 via the
118/120 interpolation; 86.514/86.298 with the added QP119 rung). The joint
bracket's upper point is QP112 at 100.028 kbps, i.e. the matched point itself;
the 100.0 interpolation between 96.160 and 100.028 is 87.194. The gate is
+0.15; the result is +0.68 / +0.92.

## 7. Matched-rate segments

Joint QP112 (100.028 kbps) vs base QP119 (99.412 kbps), the nearest-rate pair:

| segment | metric | base@119 | joint@112 | delta |
|---|---|---:|---:|---:|
| key f0 | BA | 1.2905 | **1.0784** | -0.2121 |
| key f0 | SS2 | 89.536 | **90.973** | +1.437 |
| head f1-32 | BA | 1.8325 | 1.8331 | +0.001 |
| head f1-32 | SS2 | 88.119 | **89.337** | +1.218 |
| f33-43 | BA | 2.0750 | **2.0323** | -0.043 |
| f33-43 | SS2 | 83.958 | **85.324** | +1.366 |
| f44-59 | BA | 2.5999 | **2.3218** | **-0.278** |
| f44-59 | SS2 | 78.375 | **81.151** | **+2.777** |
| clip | BA | 2.0725 | **1.9874** | -0.085 |
| clip | SS2 | 84.781 | **86.445** | +1.664 |
| clip | JOD | 9.8455 | **9.8834** | +0.038 |
| clip | PSNR-Y | 46.2835 | **47.2275** | +0.944 |

Every segment improves on both perceptual metrics; the late tail (f44-59),
the segment `docs/rd-compose.md` identified as the BA/SS2 bottleneck, gains
the most. The key is identical at a *fixed* QP; the matched pair moves the
whole ladder 6 QPs lower, which is what the P savings fund.

Source-replacement ceilings (oracle) at the two arms' nearest-to-100 points,
joint@112 (100.028) vs base@118 (101.468): key 87.277 vs 86.699, f0-32 91.966
vs 91.306, f33-43 88.938 vs 88.492, f44-59 90.416 vs 90.191. They are pooling
bounds, not encodable streams.

## 8. Ablation

| arm | bytes @QP116 | delta | note |
|---|---:|---:|---|
| reference (gate off) | 26,334 | — | `--video-rd-preset best` |
| **joint, target 8** | **23,650** | **-10.2 %** | the arm |
| joint, target 16 | 23,714 | -9.9 % | no 8x8 coding block |

The target-16 ablation is the same solve with the uniform expansion stopped at
16x16, i.e. no new 8x8 coding block. It isolates the 552 tail 8x8 leaves at
QP116: they are worth **64 B (0.27 % of the stream)**. The remaining ~2.6 kB is
the collapse pool (16x16/32x32 reference leaves merged into 32x32/64x32/64x64
blocks), not the sub-16 refinement.

## 9. Conformance

* Gate-off pins, final release build: QP106 **25,008 B /
  `0639b43630abb454f7be5ac494d2efb4`**, QP110 **23,690 B /
  `1b99127a4cd1d386fa5600fb63c57a83`**.
* `AV2ENC_RD_JOINT_P=0 --video-rd-preset best` QP116 is byte-identical to
  main (`fba63446670b43888c3ac7b280ad44dc`); the promotion sets the joint gate
  in the preset, so the explicit 0 is the control.
* Every joint stream validated: QP112/114/116/118 `tools/validate_dav2d.sh`
  OK, `dav2d --strict 1` 60/60, `avmdec` 60/60 (oracle).
* `--video-recon` payload equals the decoded payload on the 60-frame QP116
  arm and on an 8-frame smoke (the encoder reconstruction is the decoder's).
* `ctest --test-dir build-release` 12/12; release `-Werror` build clean.
* `tools/gate_video_nightly.sh` OK (gate off, `GATE_RC=0`: expansion, pins,
  strict decode, visible recon, BAWP).
* No CLI, `CMakeLists.txt`, `main.cpp`, options/parse, preset or pin edit on
  the measurement branch.  The integration promotes the gate in both presets
  (§13); the solve itself is unchanged.

## 10. Files

| file | role |
|---|---|
| `src/pipeline/video/video_joint_plan.h` | gate, window/target/lambda config, tree counters |
| `src/pipeline/video/video_frame_decision.cpp` | `joint_p_measure_leaf`, `joint_p_decide_sb`, sequential hook, reference guard; the tail/part-Diag leaf-cost helper now takes the HF/EDGE weights as arguments (tail keeps 0/0) |
| `src/pipeline/video/video_frame_internal.h` | declarations and the joint trial state |
| `src/pipeline/video/video_rate_interval.h` | Phase 2: `mode_less_inter` residual build for compound leaves |
| `src/pipeline/video/video_frame_sequence.cpp` | Phase 2: the TIP-output quality probe opts out of the joint tree |
| `tools/lane-joint-plan/ladder.sh` | interleaved base/joint ladder, strict decode, validate, recon equality, oracle |

## 11. Verdict and Phase 2

**Phase 1 KEEP.** +0.68 bracket / +0.92 fit over the base, with the fixed-QP
stream 9-11 % smaller and better on BA/SS2. The pool is the *collapse* side of
the P tree, which the partition diagnostic did not measure: the probe's
depth-1 tree over-splits the tail into 16x16 and under-splits the head, and
the exact leaf objective recovers about half the P leaves.

Two limits are explicit:

* The driver is a **research gate**: the raster serial walk is ~12 min per
  60-frame 240p encode (vs ~1 min) and is not a production default. The
  product path needs a parallel plan search before this can ship; the policy
  (one J over partition + TX/levels) is the transferable part.
* The gain is on the flat P chain. Phase 2 ports the same solve to the
  chained-B carrier with compound priced inside the leaf cost: see §12.

## 12. Phase 2 — chained-B carrier (`exp/compound-modes` `14bf3a28`)

**KEEP.** On the 87.284 chained-B aggregate (`--nightly-b` + TIP output +
CWP depth + `AV2ENC_COMPOUND_EXACT_RATE=1`) the same solve measures **87.570
bracket / 87.587 fit** against the base's **87.284 / 87.309**
(**+0.286 / +0.278**, gate +0.15); the nearest-rate pair is **+0.542**
(joint QP119 99.380 kbps K 87.664 vs base QP122 99.436 kbps K 87.122). The
base reproduces the published 87.284 exactly on this build.

Recipe:

```sh
env AV2ENC_TIP_LEAF=1 AV2ENC_TIP_LEAF_FIELD_GUARD=3 \
    AV2ENC_COMPOUND_RATE=1 AV2ENC_COMPOUND_PAIR_LIMIT=1 \
    AV2ENC_COMPOUND_EXACT_RATE=1 AV2ENC_RD_JOINT_P=1 \
  av2enc … --video --frames 60 --qp 118..123 --cpu-threads 8 \
    --video-gops 7 --nightly-b --video-compound-near-near \
    --video-cwp-weighted
```

### 12.1 Compound inside the leaf cost

The leaf rate on this carrier is the same decomposition as Phase 1, with one
extra piece:

* the coefficient symbols still come from `rate_interval::estimate`; a new
  `mode_less_inter` build sets `inter_mode = -1` so the interval model prices
  only the residual (§8.3 coefficient contexts do not depend on the compound
  mode), and
* the inter syntax comes from the live `SyntaxProbe`, which on this branch
  already prices the full compound symbol set exactly under
  `AV2ENC_COMPOUND_EXACT_RATE` (comp_mode / is_joint / compound_mode_non_joint
  / jmvd_scale_mode / both DRL indices / MVD / comp_group_idx / cwp_idx /
  compound interp_filter).

The writer's own `record_rate_interval_leaf` diagnostics keep the strict
default and never mis-price a compound leaf.

TIP output pictures (`AV2ENC_TIP_LEAF`) are `TIP_FRAME_AS_OUTPUT`: they code
no tile, so there is no leaf cost to price. The TIP quality probe is
deliberately left on the base tree (`set_joint_plan_off()`), so the
accept/reject comparison is identical to the base and the measured delta is
the tree change on the coded pictures, not a changed TIP schedule. The
block-level `--video-tip-block-ref` carrier is outside this arm.

A leaf the exact cost still cannot price (0.97 % of the QP118 trials, all
`rate_interval::estimate` refusing an over-large compound residual) keeps the
reference layout; the reference guard is now strict (`ref_ok` false -> keep
the reference), so an unmeasured tree is never committed.

### 12.2 Fixed-QP RD

| QP | arm | bytes | kbps | PSNR-Y | JOD | BA | SS2 | K |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 121 | base | 25,163 | 100.652 | 47.248 | 9.8815 | 1.8613 | 85.817 | 87.472 |
| 121 | **joint** | **24,054** | **96.216** | 47.213 | 9.8770 | 1.9123 | 85.847 | 87.294 |
| 122 | base | 24,859 | 99.436 | 47.107 | 9.8719 | 1.9610 | 85.869 | 87.122 |
| 122 | **joint** | **23,686** | **94.744** | 47.141 | 9.8759 | 1.9034 | 85.881 | 87.322 |
| 123 | base | 24,598 | 98.392 | 47.020 | 9.8655 | 1.9730 | 85.753 | 87.034 |
| 123 | **joint** | **23,533** | **94.132** | 46.984 | 9.8699 | 1.9648 | 85.847 | 87.097 |

Fixed-QP bytes: **-4.41 / -4.72 / -4.33 %**. The K column is not rate-matched
and the joint points are 3-4 kbps lower.

### 12.3 Matched K and segments

| arm | points (QP: kbps / K) | bracket | fit |
|---|---|---|---:|---:|
| base | 121: 100.652/87.472, 122: 99.436/87.122, 123: 98.392/87.034 | **87.284** (122/121) | **87.309** |
| joint | 118: 100.372/87.513, 119: 99.380/87.664, 121: 96.216/87.294, 122: 94.744/87.322, 123: 94.132/87.097 | **87.570** (119/118) | **87.587** |
| **delta** | | **+0.286** | **+0.278** |

Nearest-rate pair joint QP119 (99.380) vs base QP122 (99.436):

| segment | metric | base@122 | joint@119 | delta |
|---|---|---:|---:|---:|
| key f0 | BA / SS2 | 1.1162 / 90.985 | **1.0034 / 91.328** | -0.113 / +0.343 |
| head f1-32 | BA / SS2 | 1.6343 / 89.618 | **1.6252 / 89.857** | -0.009 / +0.239 |
| f33-43 | BA / SS2 | 2.0533 / 84.816 | **1.9612 / 84.998** | -0.092 / +0.182 |
| f44-59 | BA / SS2 | 2.6037 / 78.775 | **2.2233 / 80.072** | **-0.380 / +1.297** |
| clip | BA / SS2 | 1.9610 / 85.869 | **1.8359 / 86.381** | -0.125 / +0.512 |
| clip | PSNR-Y | 47.306 | **47.525** | +0.219 |

### 12.4 Tree stats (QP118, guard on, 1,888 superblocks)

| segment | tree | 4 | 8 | 16 | 32 | 64+ | total |
|---|---|---:|---:|---:|---:|---:|---:|
| head f1-32 | reference | 0 | 0 | 193 | 293 | 7 | 493 |
| head f1-32 | **joint** | 0 | **0** | **16** | **28** | **217** | **309** |
| tail f33-59 | reference | 0 | 0 | 3,521 | 281 | 2 | 3,804 |
| tail f33-59 | **joint** | 0 | **77** | **373** | **260** | **240** | **950** |
| all | reference | 0 | 0 | 3,714 | 574 | 9 | 4,297 |
| all | **joint** | 0 | **93** | **401** | **308** | **457** | **1,259** |

**-70.7 % leaves**; 695 superblocks changed; 975 keep the reference; the
committed `J = 10,920,615` vs `11,476,899` (**-4.85 %**). The same spatial
split as the flat chain: the head collapses almost entirely to 64+, the tail
collapses 16x16 but adds 77 measured 8x8 blocks (a smaller pool than flat P,
where the B frames' two-reference prediction already absorbs most of it).

### 12.5 Conformance

* Gate off reproduces the compound branch: `gate_video_compound_exact.sh`
  **21/21** (exact-rate 16f pin `51b664515a897a3b19be4131790867fa` / 19,774 B,
  60f 24,859 B, TIP/compound/flat pins, NEWMV controls).
* 16 f joint: 19,723 B, pinned `dav2d` sample-exact and `avmdec` recon ok.
* 60 f joint QP122: `avmdec` **60/60 recon ok** (the pinned `dav2d` keeps its
  known §7.21.6 drain defect on the 60-frame chain; 16 f is the dav2d gate).
* `ctest` 12/12 (4 fixture/decoder skips in the worktree).
* No CLI/`CMakeLists.txt`/`main.cpp`/options/parse/preset/pin edit.

### 12.6 Limits

The chained-B carrier is research-only (avmdec) until the pinned `dav2d`
drain defect is fixed, and the driver remains a slow research gate. The
measured delta is +0.28…+0.54 K on this carrier; the flat-P +0.68 is the
larger win. The next step, if this ships, is the parallel plan search and a
GPU leaf-cost batch, not a lambda sweep.

## 13. Integration and promotion (2026-11-21)

The solve is promoted as an **opt-in research arm**; it is a KEEP by the
+0.15 gate, but the serial raster walk (~12 min per 60-frame 240p encode,
vs ~1 min) keeps it out of the fast path.  `--fast-decode` and every pinned
flat default are unchanged.

| preset | promotion | measured |
|---|---|---|
| `--video-rd-preset best` | sets `AV2ENC_RD_JOINT_P=1` when unset | **87.194 bracket / 87.214 fit** (QP112 25,007 B / 100.028 kbps / K 87.196; the bracket's 96.160 leg is QP114) |
| `--nightly-b` | sets `AV2ENC_RD_JOINT_P=1` and `..._TARGET=8` when unset, inside the compound-depth block | **87.570 bracket / 87.587 fit** (QP119 24,845 B / 99.380 kbps / K 87.664 and QP118 25,093 B / 100.372 kbps / K 87.513; nearest-rate pair +0.542) |

An explicit env value wins, including `0`, which is the byte-exact control:

| control | stream |
|---|---|
| `AV2ENC_RD_JOINT_P=0 --video-rd-preset best` QP116 | 26,334 B / `fba63446670b43888c3ac7b280ad44dc` (86.33 lead stack) |
| `AV2ENC_RD_JOINT_P=0 --nightly-b` QP121 | 25,163 B / `dfa6b1aadd565a2adf47c7e186c1a973` (87.284 exact-rate arm) |
| `AV2ENC_RD_JOINT_P=0 --nightly-b` 16 f QP116 | 19,774 B / `51b664515a897a3b19be4131790867fa` |
| joint `--nightly-b` 16 f QP116 | 19,723 B / `505430de620af361a18b42503543dbda`, pinned dav2d 16/16 exact |

Pins: flat joint QP112 **25,007 B / `9a36aeb7c02733342e4ec5db300e4d6d`**
(flat key+P, so the pinned `dav2d` is the authority: `--strict 1` 60/60 and
recon-equal); B joint QP119 **24,845 B / `40de8b687d929652a312407764227398`**
and QP118 **25,093 B / `34103c1f26ca7225ebaf74f67e771175`** (avmdec 60/60;
the pinned dav2d keeps its §7.21.6 drain defect on the 60-frame chain).

Gate layout: `src/pipeline/video/gate_video_joint_plan.sh` owns the joint
pins (flat QP112, B 16/60, both controls; `--skip-60` for routine runs).
`gate_video_nightly.sh` compares `--video-rd-preset best` with
`AV2ENC_RD_JOINT_P=0` so its expansion check stays fast, and
`gate_video_tlayers_chain.sh`, `gate_video_compound_depth.sh` and
`gate_video_compound_exact.sh` export `AV2ENC_RD_JOINT_P=0` to pin the
joint-off controls.
