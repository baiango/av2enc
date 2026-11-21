# TIP leaf probe against the joint-plan coded tree — measured kill

Status: **KILL (Phase 1).** Re-probing the coded TIP alternative with
`AV2ENC_RD_JOINT_P=1` left on changes **no committed accept/reject decision**
on the chained-B carrier at 16 and 60 pictures, QP116/119. The only probe-value
flip (display 36, 60 f QP119) is **0.005 dB** from the 0.30 dB boundary and is
already masked by `AV2ENC_TIP_LEAF_FIELD_GUARD=3`. Phase 2 was not run: no K
was measured and no 60-frame stream was rate-matched.

Branch `exp/tip-decide`, parent `d138b7a8`, worktree `../tip-decide`. Recipe:
`--nightly-b` (joint plan + TIP + CWP depth + exact compound rate), 240p30,
`--cpu-threads 8`, `--video-gops 2` for 16 pictures and `7` for 60.

## 1. What was tested

The TIP output path replaces a dead-end chained-B picture with a §7.10
`TIP_FRAME_AS_OUTPUT` picture. A local quality probe encodes the coded B
alternative and keeps TIP only when

```text
sse_tip <= sse_coded * 10^(margin_db / 10)     margin_db = 0.30
```

The probe was deliberately forced to the base tree by
`probe.set_joint_plan_off()`, while the committed encoder now runs the joint
plan. If the joint plan changes the coded picture enough, the old decision can
be wrong. This experiment re-runs the probe with the joint plan left on and
compares the decisions.

### Gate

`AV2ENC_TIP_PROBE_JOINT` (default **0**), read in
`src/pipeline/video/video_frame_sequence.cpp`:

- **0** — exactly one probe, joint plan off, the historical path. Behaviour
  and bytes unchanged.
- **1** — the base probe runs as before, and a second `InterFrameEncoder` with
  the joint plan **on** consumes the same source, references, temporal context
  and `frame_options`. Both trees are scored with the same `sse_tip`, margin
  and field guard. Phase 1 keeps the committed decision on the base probe, so
  gate 1 is diagnostic and byte-neutral (verified below).

Per candidate the trace prints the base and joint lines:

```text
tip-leaf decide display=4 sse_tip=79582 sse_coded=78147 margin_db=0.3 keep=0 probe=base
tip-leaf decide display=4 sse_tip=79582 sse_coded=77603 margin_db=0.3 keep=0 probe=joint
```

and the end-of-sequence summary:

```text
tip-probe-joint summary candidates=22 base_keep=1 joint_keep=1 \
    flips_to_tip=0 flips_to_coded=0 flips=[none]
```

`base_keep`/`joint_keep` are the committed decisions (field guard included).
Raw probe-value flips are read from the `SSE keep b/j` column of the tables
below.

### Fidelity check

On every rejected candidate the probe's joint recon is compared with the
committed coded encoder's recon after both sides' `pad_edge_recon()`, over the
whole allocated luma plane (480 × 256 = **122,880 samples**). The joint plan is
deterministic, so the probe must be sample-exact or the counterfactual is
invalid.

| run | rejected candidates checked | samples compared | mismatches |
|---|---:|---:|---:|
| 16 f QP116 | 6 / 6 | 737,280 | **0** |
| 16 f QP119 | 5 / 6 (display 2 kept TIP) | 614,400 | **0** |
| 60 f QP119 | 21 / 22 (display 2 kept TIP) | 2,580,480 | **0** |

All 32 comparisons are sample-exact: the joint probe is a valid
counterfactual for the committed coded picture.

### Scheduling constraint

An 8-picture run is **not expressible** under `--nightly-b`: the preset fixes
`--video-tlayers 8` and the chained-B form requires `--video-gops > 1`, so the
smallest plan is 16 pictures (2 × 8; a 4-picture tail would make 20).

```text
$ ... --frames 8 --qp 119 --video-gops 2 --nightly-b
av2enc: error: --video-tlayers --video-gops 2 codes 16 frames
        (plus an optional 4-frame tail block); got 8
```

`--video-gops 1` is a single closed 8-frame GOP, does not set `chained_b`, and
therefore arms no TIP leaves (zero candidates). The minimum meaningful sample
is 16 pictures.

## 2. Results

Legend: `base coded` / `joint coded` are the probe's luma SSE for the coded B;
`dBase`/`dJoint = 10·log10(sse_tip / sse_coded)`, positive means the TIP
synthesis is worse; the SSE rule keeps at ≤ +0.30 dB. `guard` marks a candidate
whose SSE rule keeps but field guard 3 forces coded (`B`/`J` = base/joint);
`commit` is the final decision after the guard.

### 16 f QP116 — all six candidates are guard-rejected on both trees

| disp | sse_tip | base coded | joint coded | dBase dB | dJoint dB | SSE keep b/j | guard | commit b/j |
|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 2 | 57741 | 58039 | 57897 | -0.022 | -0.012 | 1/1 | BJ | 0/0 |
| 4 | 72895 | 71390 | 71134 | +0.091 | +0.106 | 1/1 | BJ | 0/0 |
| 6 | 83936 | 80675 | 80577 | +0.172 | +0.177 | 1/1 | BJ | 0/0 |
| 10 | 99927 | 99831 | 99657 | +0.004 | +0.012 | 1/1 | BJ | 0/0 |
| 12 | 99856 | 100103 | 100179 | -0.011 | -0.014 | 1/1 | BJ | 0/0 |
| 14 | 100897 | 101721 | 101710 | -0.035 | -0.035 | 1/1 | BJ | 0/0 |

### 16 f QP119 — one TIP leaf; no flip

| disp | sse_tip | base coded | joint coded | dBase dB | dJoint dB | SSE keep b/j | guard | commit b/j |
|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 2 | 63790 | 63815 | 63815 | -0.002 | -0.002 | 1/1 | - | 1/1 |
| 4 | 79582 | 78147 | 77603 | +0.079 | +0.109 | 1/1 | BJ | 0/0 |
| 6 | 91515 | 88081 | 87273 | +0.166 | +0.206 | 1/1 | BJ | 0/0 |
| 10 | 105824 | 105891 | 105891 | -0.003 | -0.003 | 1/1 | BJ | 0/0 |
| 12 | 105982 | 106322 | 106413 | -0.014 | -0.018 | 1/1 | BJ | 0/0 |
| 14 | 108003 | 107706 | 107714 | +0.012 | +0.012 | 1/1 | BJ | 0/0 |

### 60 f QP119 — the product point; no committed flip

| disp | sse_tip | base coded | joint coded | dBase dB | dJoint dB | SSE keep b/j | guard | commit b/j |
|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 2 | 63790 | 63815 | 63815 | -0.002 | -0.002 | 1/1 | - | 1/1 |
| 4 | 79582 | 78147 | 77603 | +0.079 | +0.109 | 1/1 | BJ | 0/0 |
| 6 | 91515 | 88081 | 87273 | +0.166 | +0.206 | 1/1 | BJ | 0/0 |
| 10 | 105824 | 105891 | 105891 | -0.003 | -0.003 | 1/1 | BJ | 0/0 |
| 12 | 105982 | 106322 | 106413 | -0.014 | -0.018 | 1/1 | BJ | 0/0 |
| 14 | 108003 | 107706 | 107714 | +0.012 | +0.012 | 1/1 | BJ | 0/0 |
| 18 | 115091 | 112965 | 112937 | +0.081 | +0.082 | 1/1 | BJ | 0/0 |
| 20 | 119818 | 118242 | 118366 | +0.058 | +0.053 | 1/1 | BJ | 0/0 |
| 22 | 121838 | 121106 | 121043 | +0.026 | +0.028 | 1/1 | BJ | 0/0 |
| 26 | 126875 | 126213 | 126385 | +0.023 | +0.017 | 1/1 | BJ | 0/0 |
| 28 | 127206 | 126906 | 126779 | +0.010 | +0.015 | 1/1 | BJ | 0/0 |
| 30 | 144522 | 142427 | 141495 | +0.063 | +0.092 | 1/1 | BJ | 0/0 |
| 34 | 220421 | 216435 | 218774 | +0.079 | +0.033 | 1/1 | BJ | 0/0 |
| 36 | 229243 | 214168 | 213747 | **+0.295** | **+0.304** | **1/0** | B | 0/0 |
| 38 | 223621 | 218756 | 221028 | +0.096 | +0.051 | 1/1 | BJ | 0/0 |
| 42 | 183774 | 180416 | 182337 | +0.080 | +0.034 | 1/1 | BJ | 0/0 |
| 44 | 166611 | 162201 | 163782 | +0.117 | +0.074 | 1/1 | BJ | 0/0 |
| 46 | 178892 | 163915 | 165004 | +0.380 | +0.351 | 0/0 | - | 0/0 |
| 50 | 186723 | 148501 | 147685 | +0.995 | +1.019 | 0/0 | - | 0/0 |
| 52 | 142961 | 136188 | 138324 | +0.211 | +0.143 | 1/1 | BJ | 0/0 |
| 54 | 145250 | 141050 | 138983 | +0.127 | +0.192 | 1/1 | BJ | 0/0 |
| 58 | 112358 | 109852 | 111986 | +0.098 | +0.014 | 1/1 | BJ | 0/0 |

### Summary

| run | candidates | base keep | joint keep | committed flips | raw SSE flips | guard hits base/joint |
|---|---:|---:|---:|---:|---:|---:|
| 16 f QP116 | 6 | 0 | 0 | 0 | 0 | 6 / 6 |
| 16 f QP119 | 6 | 1 | 1 | 0 | 0 | 5 / 5 |
| 60 f QP119 | 22 | 1 | 1 | 0 | 1 (display 36) | 19 / 18 |

The only raw probe-value flip is **display 36** at 60 f QP119:

- base tree: +0.295 dB → inside the 0.30 dB margin, SSE says keep;
- joint tree: +0.304 dB → outside the margin, SSE says reject;
- direction: **tip → coded**;
- distance to the margin: **0.005 dB** (well inside the 0.05 dB kill band);
- the base decision is already overridden by field guard 3, and the joint
  SSE decision is to reject, so the **committed decision is coded on both
  trees**. There is no stream flip.

The largest base→joint change of the coded picture is **0.084 dB**
(display 58; 0.068 dB at display 52, 0.064 dB at display 54). The 0.30 dB
margin is roughly four times larger, which is why no decision can move except
at the boundary.

### A side observation

At 60 f QP119 the field guard is the dominant TIP filter: 19 of 22 dead-end
leaves are guard-rejected on the base tree, two more are clear SSE rejects
(displays 46 and 50, +0.35 to +1.02 dB), and exactly **one** picture (display
2) remains a TIP output. In this carrier, the probe margin is almost never the
operative decision; the §7.22 compound/CWP field guard is. That is consistent
with the preset's guard being required for decoder exactness, but it means a
change to the probe's SSE source has very little surface to act on.

## 3. Wall cost

All runs are 8-thread CPU encodes on a shared M-series host (load average
8–11 on 10 cores while these ran), so wall times are noisy. CPU time
(`user + sys`) is the more stable number. Gate-on Phase 1 runs **two** probes
per candidate (base + joint); a production Phase 2 would need only the joint
probe and would be cheaper than these diagnostic numbers.

| run | gate | wall s | user+sys s | note |
|---|---|---:|---:|---|
| 16 f QP116 | 0 | 122.09 | 186.36 | pin |
| 16 f QP116 | 1 | 217.60 | 290.06 | no trace |
| 16 f QP119 | 0 | 129.84 | 182.29 | |
| 16 f QP119 | 1 | 171.07 | 249.70 | no trace |
| 60 f QP119 | 0 | 527.71 | 772.43 | pin |
| 60 f QP119 | 1 | 866.11 | 1198.02 | no trace |

The diagnostic gate adds ~35–55 % CPU time, dominated by the extra joint-tree
probe. Trace output is large but does not change bytes.

## 4. Pins and gate-off evidence

All gate-off streams were produced with an explicit
`env AV2ENC_TIP_PROBE_JOINT=0`.

| stream | bytes | MD5 | pin |
|---|---:|---|---|
| flat QP106 `--fast-decode` | 25,008 | `0639b43630abb454f7be5ac494d2efb4` | exact |
| flat QP110 `--fast-decode` | 23,690 | `1b99127a4cd1d386fa5600fb63c57a83` | exact |
| B 16 f QP116 | 19,723 | `505430de620af361a18b42503543dbda` | exact |
| B 60 f QP119 | 24,845 | `40de8b687d929652a312407764227398` | exact |
| B 16 f QP119 (control) | 18,995 | `bee07dc2c9b22230677e5dd612eaf40c` | — |

The gate-on Phase-1 outputs are **byte-identical** to the gate-off controls:

- B 16 f QP116: 19,723 / `505430de…`
- B 16 f QP119: 18,995 / `bee07dc2…`
- B 60 f QP119: 24,845 / `40de8b68…`

so the extra joint probe and the trace are byte-neutral. The 16 f stream
decodes under pinned `dav2d` (`f9dafbc`, 16 frames; `--strict 1 --threads 1`
decodes 16/16 and `tools/validate_dav2d.sh` reports OK); the 60 f stream is
`avmdec` 60/60 (`recon=n/a`), matching the documented status of the existing
pin.

## 5. Verdict

**KILL.** The stated Phase-1 kill conditions are met twice over:

1. there are **zero committed flips** in all three runs;
2. the only raw probe-value flip (display 36) is **within 0.05 dB of the
   margin** and is already neutralised by the field guard.

Phase 2 was therefore not started. No gate-on stream was committed, no
rate-match was run, and no K was measured. The TIP decision on this carrier is
not being made on a coded tree that materially differs from the committed one:
the joint plan moves the coded recon by at most ~0.08 dB, while the decision
margin is 0.30 dB.

What would change the answer: a carrier/operating point where the joint plan
moves the coded recon by a substantial fraction of the margin, or a probe margin
much tighter than 0.05 dB. Neither is the current preset.

## 6. Exact commands

Build:

```sh
cmake -S . -B build-tip -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON
cmake --build build-tip -j8
```

16 f QP116, gate off / gate on:

```sh
env AV2ENC_TIP_PROBE_JOINT=0 build-tip/av2enc \
  testdata/nasa_480x240_30fps_60f.y4m -o /tmp/tipdec/off16q116.ivf \
  --video --frames 16 --qp 116 --cpu-threads 8 --video-gops 2 --nightly-b

env AV2ENC_TIP_PROBE_JOINT=1 build-tip/av2enc \
  testdata/nasa_480x240_30fps_60f.y4m -o /tmp/tipdec/on16q116.ivf \
  --video --frames 16 --qp 116 --cpu-threads 8 --video-gops 2 --nightly-b \
  --video-trace
```

16 f QP119: same two commands with `--qp 119` and file names
`off16q119.ivf` / `on16q119.ivf`.

60 f QP119:

```sh
env AV2ENC_TIP_PROBE_JOINT=0 build-tip/av2enc \
  testdata/nasa_480x240_30fps_60f.y4m -o /tmp/tipdec/off60q119.ivf \
  --video --frames 60 --qp 119 --cpu-threads 8 --video-gops 7 --nightly-b

env AV2ENC_TIP_PROBE_JOINT=1 build-tip/av2enc \
  testdata/nasa_480x240_30fps_60f.y4m -o /tmp/tipdec/on60q119.ivf \
  --video --frames 60 --qp 119 --cpu-threads 8 --video-gops 7 --nightly-b \
  --video-trace
```

Flat controls:

```sh
env AV2ENC_TIP_PROBE_JOINT=0 build-tip/av2enc \
  testdata/nasa_480x240_30fps_60f.y4m -o /tmp/tipdec/flat106.ivf \
  --video --frames 60 --qp 106 --cpu-threads 8 --fast-decode
env AV2ENC_TIP_PROBE_JOINT=0 build-tip/av2enc \
  testdata/nasa_480x240_30fps_60f.y4m -o /tmp/tipdec/flat110.ivf \
  --video --frames 60 --qp 110 --cpu-threads 8 --fast-decode
```

8-picture attempt (rejected by the plan, recorded to document the scheduling
constraint):

```sh
env AV2ENC_TIP_PROBE_JOINT=1 build-tip/av2enc \
  testdata/nasa_480x240_30fps_60f.y4m -o /tmp/tipdec/x.ivf \
  --video --frames 8 --qp 119 --cpu-threads 8 --video-gops 2 --nightly-b
```
