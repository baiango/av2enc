# Exact leaf/partition RD (`AV2ENC_EXACT_RD`)

Status: **measured, env-gated, default off; no K gain, not shipped in any
preset.** Branch `exp/rd-exact`. The gate stays in the tree for reproduction;
unset is byte-identical to the pinned streams.

## Defect

Two decisions used proxies and only discovered the coded cost afterwards:

1. `encode_leaf()` ranked inter candidates by prediction-domain luma SSE plus
   the static `inter_syntax_bits()` estimate, then ran the real predict -> TX
   search -> quantise -> reconstruct -> rate path only for that winner. A
   candidate with a slightly worse prediction can code a much cheaper
   residual, so the proxy ranking is not a total order of leaf cost.
2. `partition_probe_leaf_cost()` priced every partition node with one NEWMV
   prediction + one quantised luma residual. The depth-1 tree was therefore
   compared at a cost the leaf would never actually use.

## What was built

`AV2ENC_EXACT_RD=1` switches both decisions to measured costs. There is no CLI
flag; the switches are environment-only:

| Variable | Default | Meaning |
|---|---|---|
| `AV2ENC_EXACT_RD` | off | master gate; unset/`0`/`false`/`off` keeps every historical path |
| `AV2ENC_EXACT_RD_TOP` | `2` | number of cheap candidates re-scored (1..8) |
| `AV2ENC_EXACT_RD_PARTITION` | on | set `0` to keep the NEWMV probe only (ablation) |

### 1. Top-N leaf rescoring

`consider()` keeps the full candidate list (inter modes from every reference,
NEWMV, compound) with the existing cheap key: `sse + lambda *
inter_syntax_bits()`, or raw SSE when `--video-leaf-rd` is off. The proxy
winner is evaluated first, then the next cheapest candidates in stable order
(`cost`, then search order). Each candidate runs the shared real leaf path:
`predict_all` / warp / compound, BAWP and inter-intra, `search_luma_tx` (the
`vf_detail::compute_tx` path), chroma `compute_tx` / CCTX, the existing
skip-vs-residual test, and reconstruction. The candidate cost is

```
J = D + lambda * R
```

with `D` the reconstruction SSE of every plane the leaf owns, `R` the same
syntax + coefficient estimate the leaf decision already charges (mode/MV
syntax, BAWP/inter-intra flags, luma TX plan + chroma coefficients), and
`lambda = leaf_rd_lambda(qindex)`. The intra fallback is already fully coded
and enters the same final comparison with its total reconstruction D and the
same lambda. Ties keep the earlier candidate / inter over intra, so `N=all`
visits a deterministic total order.

### 2. Exact partition leaf cost

A full exact cost per partition node would require encoding every candidate
layout (root, HORZ, VERT, SPLIT), i.e. several extra frames per frame. The
chosen compromise is one refinement pass:

1. Solve the depth-1 tree with the historical NEWMV probe and encode the frame.
2. `encode_leaf()` records the measured `(D, R)` of the mode it actually
   emitted in `SbSink::leaf_costs`.
3. Re-solve the tree with `D + partition_lambda * R` for every node the first
   pass encoded; nodes the first pass did not visit keep the NEWMV probe as
   the early bound.
4. If the leaf set changed, encode once more with the refined tree and emit
   that stream. The refinement is not iterated (one pass, <=2x encode cost).

The refinement is restricted to the rect tree (`--video-ext-partitions` keeps
the historical solver) and to frames without segmentation/delta-Q/AQ, where
the superblock quantizer is not a per-node quantity.

### 3. Instrumentation

Cumulative counters are printed once at process exit, only under the gate:

```
exact-rd: top_n=2 partition=1 leaves=... rescored=... inter_winner_changed=...
          intra_flipped=... proxy_residual=... exact_residual=...
          partition_overrides=... partition_tree_changed=...
          partition_refine_runs=...
```

`leaves`/`rescored`/`inter_winner_changed` count every leaf evaluation, so a
stream that needed the second encode pass appears twice. `proxy_residual` is
the residual decision of the proxy winner; `exact_residual` is the chosen
candidate's. `partition_tree_changed` counts frames whose leaf set changed.

## Method

Worktree `exp/rd-exact` from `main`, `cmake --preset dev`,
`cmake --build build-dev -j2`. Host Apple M4, `--cpu-threads 8`, single tile,
8-bit 4:2:0, 60-frame `testdata/nasa_480x240_30fps_60f.y4m`. Encode ->
`dav2d-macos --strict 1` -> `tools/quality.py` (CVVDP 0.5.7, `standard_fhd`,
MPS) -> `tools/extra_metrics.py` -> `tools/kpi.py`. One metric job at a time,
`nice -n 10`. Every measured stream decoded 60/60 pictures. K is interpolated
linearly in rate; the local `--nightly` baseline is 82.048 at 100.0 kbps
(QP105/106/107: 81.81/82.15/80.51), the tuned drift baseline is 83.795
(QP110/111: 83.94/83.75).

## Results

### `--nightly` (bar: QP106 pin K 82.15 at 100.03 kbps; local K@100 = 82.05)

| Arm | QP | Bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 105 | 25,342 | 101.368 | 9.7216 | 3.5413 | 80.396 | 81.81 |
| baseline | 106 | 25,008 | 100.032 | 9.7334 | 3.4202 | 80.837 | **82.15** |
| baseline | 107 | 24,885 | 99.540 | 9.7113 | 4.3229 | 80.256 | 80.51 |
| **baseline K@100** | — | — | 100.0 | — | — | — | **82.05** |
| N=2, leaf+partition | 105 | 25,366 | 101.464 | 9.7009 | 3.6538 | 79.822 | 81.41 |
| N=2, leaf+partition | 106 | 25,108 | 100.432 | 9.7147 | 3.5687 | 80.424 | 81.73 |
| N=2, leaf+partition | 107 | 24,868 | 99.472 | 9.7027 | 3.9011 | 80.118 | 81.07 |
| **N=2 K@100** | — | — | 100.0 | — | — | — | **81.43** |
| N=3, leaf+partition | 105 | 25,570 | 102.280 | 9.6996 | 3.6574 | 80.096 | 81.45 |
| N=3, leaf+partition | 106 | 25,329 | 101.316 | 9.7114 | 3.6902 | 80.403 | 81.50 |
| N=3, leaf+partition | 107 | 24,637 | 98.548 | 9.7019 | 3.5442 | 80.278 | 81.68 |
| **N=3 K@100** | — | — | 100.0 | — | — | — | **81.59** |

Both lose to the pin. The K loss is almost entirely Butteraugli: JOD moves by
-0.02..-0.03 and SS2 by -0.3..-0.6, while BA is +0.13..+0.34 against the
matched baseline.

Ablations at QP106 (each stream decoded 60/60):

| Arm | Bytes | kbps | K | Note |
|---|---:|---:|---:|---|
| baseline | 25,008 | 100.032 | 82.15 | |
| N=1, partition exact | 25,183 | 100.732 | 81.76 | proxy leader only; isolates partition refinement |
| N=2, leaf only | 25,082 | 100.328 | 79.80 | `AV2ENC_EXACT_RD_PARTITION=0` |
| N=3, leaf only | 25,172 | 100.688 | 82.52 | `AV2ENC_EXACT_RD_PARTITION=0` |
| N=3, leaf only, bracketed (105/106/107) | — | 100.0 | **82.12** | 82.33 / 82.52 / 81.22 at 100.50 / 100.69 / 97.90 kbps |

The leaf-only and leaf+partition ablations disagree by more than 1 K in both
directions at the same QP (N=2: +1.93 for partition refinement; N=3: -1.02).
That is the same order as the metric's QP ripple, so neither partition
component is a resolvable lever on its own. The bracketed N=3 leaf-only number
(82.12) is the best `--nightly` variant and it is flat against the 82.05
local baseline.

### Tuned drift recipe (bar: K 83.80 at 100.0 kbps)

`--nightly --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget
3250`, QP109..112 (QP112 added to bracket 100 kbps):

| Arm | QP | Bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 110 | 25,146 | 100.584 | 9.7605 | 2.6562 | 81.250 | 83.94 |
| baseline | 111 | 24,950 | 99.800 | 9.7513 | 2.7310 | 81.422 | 83.75 |
| **baseline K@100** | — | — | 100.0 | — | — | — | **83.80** |
| N=2, leaf+partition | 109 | 25,941 | 103.764 | 9.7288 | 2.9296 | 81.303 | 83.18 |
| N=2, leaf+partition | 110 | 25,300 | 101.200 | 9.7282 | 2.9477 | 80.480 | 83.00 |
| N=2, leaf+partition | 111 | 25,264 | 101.056 | 9.7250 | 2.8638 | 81.152 | 83.27 |
| N=2, leaf+partition | 112 | 24,778 | 99.112 | 9.7338 | 2.9901 | 80.859 | 83.00 |
| **N=2 K@100** | — | — | 100.0 | — | — | — | **83.13** |
| N=3, leaf+partition | 109 | 26,075 | 104.300 | 9.7389 | 2.8713 | 81.340 | 83.36 |
| N=3, leaf+partition | 110 | 25,484 | 101.936 | 9.7421 | 2.7453 | 81.307 | 83.65 |
| N=3, leaf+partition | 111 | 25,303 | 101.212 | 9.7294 | 2.8343 | 81.092 | 83.35 |
| N=3, leaf+partition | 112 | 24,798 | 99.192 | 9.7395 | 2.9125 | 80.908 | 83.20 |
| **N=3 K@100** | — | — | 100.0 | — | — | — | **83.26** |
| N=3, leaf only | 109 | 25,972 | 103.888 | 9.7477 | 3.2198 | 81.395 | 82.70 |
| N=3, leaf only | 110 | 25,327 | 101.312 | 9.7540 | 2.8354 | 81.384 | 83.52 |
| N=3, leaf only | 111 | 25,151 | 100.604 | 9.7439 | 3.1208 | 81.121 | 82.83 |
| N=3, leaf only | 112 | 24,600 | 98.400 | 9.7409 | 3.0912 | 81.174 | 82.88 |
| **N=3 leaf-only K@100** | — | — | 100.0 | — | — | — | **82.84** |

No arm reaches 83.80. The closest single point (N=3 leaf+partition QP110,
83.65 at 101.94 kbps) is above the bar's rate and its neighbouring QPs fall
back to 83.35/83.20. The loss is again Butteraugli: JOD is flat
(-0.01..-0.03), SS2 within noise, BA +0.1..+0.3 against the matched baseline.

## Counters

Leaf-only runs (one encode pass, clean per-leaf counts):

| Arm | inter leaves | rescored | winner changed | intra won | proxy residual | exact residual |
|---|---:|---:|---:|---:|---:|---:|
| N=2, QP106 | 2,899 | 5,798 | 389 (13.4%) | 0 | 635 | 632 |
| N=3, QP106 | 2,892 | 8,676 | 465 (16.1%) | 5 | 633 | 633 |
| N=3 + drift, QP110 | 2,853 | 8,559 | 485 (17.0%) | 4 | 1,696 | 1,689 |

Full runs contain two encodes when the partition tree changed; their counters
are exactly the sum of both passes (e.g. N=3 QP106: 5,906 leaves, 17,718
rescored, 854 winner changes). `partition_tree_changed` is 59/59 P frames for
every full run, i.e. the refinement re-encoded every frame.

The important number is `proxy_residual` vs `exact_residual`: the exact
selection does **not** select more residual leaves (633 vs 633 at QP106,
1,696 vs 1,689 with drift). The damage is not a skip/residual threshold
change; it is which mode/MV/residual wins once the real reconstruction is
priced.

## Encode cost

Dev build (`-O1`), QP106 `--nightly`, 8 threads, median of 3 wall-clock runs:

| Arm | Median | vs baseline |
|---|---:|---:|
| baseline | 6.63 s | 1.00x |
| N=2, leaf only | 9.10 s | 1.37x |
| N=3, leaf only | 9.46 s | 1.43x |
| N=2, leaf+partition | 17.32 s | 2.61x |
| N=3, leaf+partition | 22.37 s | 3.37x |

The partition refinement re-encodes every frame (the leaf set changed on all
59 P frames), so it costs almost exactly 2x; the top-N rescoring adds the
rest.

## Conformance

- Gate off: all five pins byte-identical — 25,009 `a1d0dba1`, 25,141
  `3984677b`, 23,690 `1b99127a`, 24,113 `7f03eb6e`, 720p 36,572 `c4ba2813`.
- Gate on: every measured stream decodes 60/60 under `dav2d --strict 1`;
  `--video-recon` equals the decoded payload byte for byte (N=2 QP106 and N=3
  QP110+drift both checked with `cmp`).
- `tools/test.sh --build build-dev` 10/10; `gate_video_nightly.sh`,
  `gate_video_qp_lambda.sh` and `gate_video_partitions.sh` pass (gate off,
  where their pins apply).

## Verdict

**No K gain, do not enable.** At `--nightly` the best measured iso-rate point
is 82.12 (N=3 leaf-only) against the 82.05 local baseline / 82.15 pin; at the
tuned drift operating point every arm is 83.13-83.26 against 83.80. The
differences are inside the documented +/-1.6 K QP ripple, but every bracket is
on the same side: JOD is flat, SS2 is flat, and Butteraugli gets worse. Exact
SSE + estimated-rate RD is not aligned with the K objective here; the proxy's
prediction-SSE preference was doing useful work by keeping quantization noise
out of the detail band, and the exact path spends that away.

If this is revisited, the two missing pieces are (a) an entropy-accurate leaf
rate (`luma_plan_bits` is still a nonzero-count heuristic, so the exact
objective is only as exact as its R term) and (b) a partition refinement that
measures both sides of each depth-1 comparison instead of mixing exact leaf
costs with a differently calibrated NEWMV probe.
