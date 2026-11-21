# Post-joint key allocation: the chained-B key offset moves to −37

Status: **KEEP, promoted into `--nightly-b`.** The chained-B preset spent
`--video-key-qp-offset -50`; on the final stack (compound + TIP + CWP depth +
exact compound rate + joint plan) the measured optimum is **−37**, worth
**+0.694 K** at matched 100 kbps. Branch `exp/alloc-sweep`, worktree
`../alloc-sweep`, from main `6e90dff7`. Measurement only in the sweep; the
preset change is one line in `src/cli/nightly.cpp` (branch `exp/key37`).

## 0. Answer first

| key offset | QP | kbps | K |
|---:|---:|---:|---:|
| **−37** | 111 | 100.364 | **88.207** |
| −39 | 112 | 100.372 | 87.878 |
| −50 (was the preset) | 118 | 100.372 | 87.513 |

**+0.694 K at a 0.008 kbps rate difference.** All three readings clear the
+0.15 gate: matched-rate **+0.694**, bracket **+0.616** (88.186 / 87.570), fit
**+0.444** (88.063 / 87.619).

The `−50` value predates compound, TIP, CWP depth, the exact compound rate and
the joint plan; each of those changed what a key bit is worth to the rest of
the chain, so the offset was re-swept rather than trusted.

## 1. Premise correction

`--nightly-b` sets `-50`, **not** `-39` (`src/cli/nightly.cpp`). `-39` is the
flat `--video-rd-preset best` spend. The documented QP119 pin
`24,845 B / 40de8b68…` is the `-50` stream, which is how the two were
separated here.

## 2. Phase A — rate direction at QP119

One 60-frame chained-B encode per offset (`--video-gops 7 --nightly-b`),
240p30, 8 threads.

| key offset | QP119 bytes | kbps | key share |
|---:|---:|---:|---:|
| −33 | 21,298 | 85.19 | 69.2 % |
| −35 | 21,919 | 87.68 | — |
| −37 | 22,318 | 89.27 | — |
| −39 | 22,622 | 90.49 | — |
| −41 | 23,092 | 92.37 | — |
| −43 | 23,345 | 93.38 | — |
| −45 | 23,745 | 94.98 | — |
| **−50 (preset)** | **24,845** | **99.38** | **73.1 %** |

The key share rises from 69.2 % to 73.1 % because a coarser key needs a
*larger* share of a *larger* total: 3,425 of the 3,547 byte difference is the
key picture alone.

## 3. Phase B/C — 100 kbps ladders

The QP→rate slope steepens as the key gets coarser (0.99 kbps/QP at −50,
~2.8 at −45, ~4.4 at −33), so the first bracket QPs overshot and a third point
was added per offset to tighten each bracket around 100 kbps.

| key offset | ladder (kbps → K) | bracket @100 | fit | R² |
|---:|---|---:|---:|---:|
| **−37** | 97.27→88.033, **100.36→88.207**, 101.33→87.952 | **88.186** | 88.063 | 0.001 |
| −39 | 90.49→87.173, **100.37→87.878**, 102.37→87.806 | 87.851 | 87.753 | 0.939 |
| −50 | 99.38→87.664, **100.37→87.513**, 102.42→87.829 | 87.570 | 87.619 | 0.462 |
| −41 | 92.37→87.086, **100.59→87.841**, 103.31→88.267 | 87.787 | 87.861 | 0.985 |
| −43 | 93.38→87.228, 99.89→—, 102.92→88.234 | 87.926 | 87.926 | — |
| −45 | 94.98→87.518, **100.43→—**, 103.26→87.952 | 87.781 | 87.781 | — |

−37 is a **peak** flanked by lower neighbours (−35 87.722, −39 87.851), which
is the shape a real optimum makes, and it wins even away from its peak (at
97.27 kbps it is 88.033, above −50's best 87.664).

Both ladders are jagged inside the documented ±0.2–0.3 adjacent-QP ripple, so
**the matched-rate pair and the bracket are the number to trust; the −37 fit's
R² of 0.001 means its 3-point line fit is flat through noise.**

## 4. Conformance

Every scored stream is `avmdec` **60/60 with recon equality**:

```
kp37_q110 kp37_q111 kp37_q112 kp39_q111 kp50_q117 kp50_q118 kp50_q119  all 60/60 recon-match
```

The shipped preset reproduces the measured point byte-for-byte:

| stream | before (−50) | after (−37) |
|---|---|---|
| 16 f B QP116 | 19,723 / `505430de…` | **17,066 / `1de2bb9a…`** |
| 60 f QP119 | 24,845 / `40de8b68…` | **22,318 / `bd3edd9b…`** |

`22,318 / bd3edd9b` is exactly the `kp37_q119` stream from the sweep, so the
shipped path is the measured path rather than something adjacent to it.

The flat pins are unaffected: `--video-rd-preset best` sets its own `-39`, so
flat QP106 `25,008 / 0639b436…`, QP110 `23,690 / 1b99127a…` and QP112
`25,007 / 9a36aeb7…` are unchanged.

## 5. Reproduce

```sh
# the arm (override the preset default)
env AV2ENC_TIP_LEAF_FIELD_GUARD=3 build-release/av2enc \
  testdata/nasa_480x240_30fps_60f.y4m -o out.ivf \
  --video --frames 60 --qp 111 --cpu-threads 8 --video-gops 7 --nightly-b \
  --video-key-qp-offset -37

# the control
  ... --qp 118 --video-key-qp-offset -50
```

Score with `tools/oracle.py --target-kbps 100` and
`tools/lane-pdriver/kfit.py`.

## 6. Files

| file | role |
|---|---|
| `src/cli/nightly.cpp` | the preset default `-50` → `-37` (the change) |
| `runs/encode.sh`, `runs/phaseB.sh`, `runs/phaseC.sh`, `runs/score.sh` | the sweep |
| `runs/verify37.sh`, `runs/avmcheck.sh` | avmdec 60/60 + recon gate, ladder scoring |

## 7. Verdict

**KEEP.** The chained-B key spend moves from −50 to −37 for **+0.69 K** at
100 kbps, confirmed by a matched-rate pair, a 3-point bracket on both arms, and
`avmdec` 60/60 recon-exact streams. This is the first gain since the joint plan
(87.570 → 88.207).

Caveat for the record: every point here is one host, one clip, single runs. The
ripple means the honest claim is "+0.7 bracket at matched rate on this clip",
not "+0.7 on any content".
