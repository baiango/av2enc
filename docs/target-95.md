# Target 95 at 240p / 100 kbps: feasibility

Status: **read-only feasibility analysis.** Branch `exp/target-95`, base
`82d75489`. No encoder code was changed; this document is the deliverable.
No third-party encoder or decoder source, model, weight or asset is vendored
or copied. AVM and SVT are black-box reference binaries only, as in the rest
of the repo.

The question: **is K 95 at 480x240 / 30 fps / 100 kbps reachable by any
first-party mechanism, and what class would it require?**

## 0. Verdict

**No first-party mechanism in this repository reaches K 95 at 240p/100 kbps.
No measured first-party mechanism even bounds above the AVM reference point of
90.2. On the measured evidence, K 95 requires a new codec class, not an AV2
tuning result.**

| quantity | value | status |
|---|---:|---|
| target K | **95.0** | target |
| best first-party matched-rate K | **85.10** | measured (`eff-perc`) |
| strongest first-party rate-free/segment ceiling | **90.12** | measured (`eff-perc`, source replacement) |
| best realized conventional AV2 point | **90.2** | black-box AVM reference |
| best non-decodable structural bound | **89.8** | perfect spacing-2 neighbour, not a stream |
| SVT wall | **88.8** | black-box reference |
| K 95 gap from best first-party | **9.90 K** | arithmetic |
| K 95 gap from AVM | **4.81 K** | arithmetic |

The gap is not a JOD problem. At the stated target (JOD 9.90, BA 0.44,
SS2 92), **76.8% of the 9.90 K gap from eff-perc is Butteraugli**, 16.5% is
SSIMULACRA2 and 6.7% is CVVDP. From AVM, **90.3% of the remaining 4.81 K is
Butteraugli**. Even with perfect CVVDP and SSIMULACRA2, a BA of 1.29 (AVM's
value) caps K at **92.5**; a BA of 2.27 (eff-perc's value) caps K at **89.2**.
A 95 point is impossible with BA above **0.75**, whatever the other two
metrics do.

A measured conventional-class rate ladder on the same fixture does not cross
K 95 until roughly **0.6-0.8 Mbps** (SVT), 6-8x the target rate. The best
realized AV2 encoder (AVM) is still at **92.76** at **188.9 kbps** (1.9x the
target), 2.24 K short. The conventional class needs hundreds of kbps, not
100 kbps, to reach 95 on this content.

**Recommendation: close the implementation gap to ~90 first. Do not start a
95 research program now.** A 95 claim would require a new codec class with a
new bitstream, normative decoder and conformance suite, plus first-party
training data/compute; it is not a branch-sized encoder task. The concrete
reopen gate is in section 6.

## 1. Target math

K is `tools/kpi.py`:

```text
q_cvvdp = clamp(JOD / 10, 0, 1)          # JOD 10 is the top of the scale
q_ba    = 1 / (1 + BA / 3)               # BA 3 is "clearly visible"
q_ss2   = clamp((SS2 + 50) / 150, 0, 1)  # SS2 100 is perfect, -50 is the floor

K = 100 * (0.50*q_cvvdp + 0.25*q_ba + 0.25*q_ss2)
```

The target triple in the task (JOD 9.90, BA 0.44, SS2 92) evaluates to
**K 94.97**, quoted as 95. The 0.03 is rounding. K 95 is a surface, not one
triple; the iso-K frontier below shows how much Butteraugli can be traded for
JOD/SS2.

### 1.1 Component targets and measured points

| point | JOD | BA | SS2 | q_cvvdp | q_ba | q_ss2 | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| **target (quoted)** | **9.90** | **0.44** | **92.0** | **0.990** | **0.872** | **0.947** | **94.97** |
| eff-perc @100.0 kbps | 9.768 | 2.273 | 82.23 | 0.977 | 0.569 | 0.882 | **85.10** |
| AVM QP124 @99.3 kbps | 9.862 | 1.289 | 90.38 | 0.986 | 0.699 | 0.936 | **90.19** |
| SVT CRF36.5 @99.4 kbps | 9.826 | 1.583 | 88.87 | 0.983 | 0.655 | 0.926 | **88.64** |

The eff-perc row is linearly interpolated to exactly 100.000 kbps from the
shipped QP119/QP120 points (`docs/eff-perc.md`). The shipped QP119 point is
85.18 at 100.920 kbps, JOD 9.7713 / BA 2.2630 / SS2 82.441; the interpolation
to 100.0 gives the row above. AVM is the upstream `e6f99d3` black-box point
from `bench/avm_baseline/README.md`; SVT is 4.3.0 preset 0, reproduced in
this analysis.

### 1.2 Gap decomposition

| from → target | ΔJOD | ΔBA | ΔSS2 | ΔK from JOD | ΔK from BA | ΔK from SS2 | total | BA share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eff-perc 85.10 → 95 | +0.132 | −1.833 | +9.77 | +0.66 | **+7.58** | +1.63 | **+9.87** | **76.8%** |
| AVM 90.19 → 95 | +0.038 | −0.849 | +1.62 | +0.19 | **+4.32** | +0.27 | **+4.78** | **90.3%** |
| SVT 88.64 → 95 | +0.074 | −1.143 | +3.13 | +0.37 | **+5.44** | +0.52 | **+6.33** | **85.9%** |

Read the table as a weighted-credit decomposition, not as independent knobs:
`ΔK_JOD = 0.5*Δq_cvvdp*100`, and so on. The BA term dominates every route.
The totals differ from the quoted K differences by up to 0.03 because the
component rows are rounded; the exact K values are in §1.1.

A second way to see the same fact:

| scenario | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|
| eff-perc today | 9.768 | 2.273 | 82.23 | **85.10** |
| hit the JOD and SS2 targets, keep today's BA | 9.90 | 2.273 | 92.0 | **87.39** |
| hit the BA target, keep today's JOD and SS2 | 9.768 | 0.44 | 82.23 | **92.68** |
| hit all three targets | 9.90 | 0.44 | 92.0 | **94.97** |

The BA-only move buys 7.58 K; the other two targets together buy only 2.29 K
without BA. This is why an encoder that only improves "detail" or only
improves CVVDP cannot approach 95.

### 1.3 Iso-K frontier and the hard BA bound

Setting K = 95 and solving for the required Butteraugli:

```text
q_ba = 3.8 - 2*q_cvvdp - q_ss2
BA_required = 3 * (1/q_ba - 1)
```

| JOD | SS2 | q_cvvdp | q_ss2 | required q_ba | maximum BA |
|---:|---:|---:|---:|---:|---:|
| 9.90 | 92 | 0.990 | 0.947 | 0.873 | **0.435** |
| 9.95 | 92 | 0.995 | 0.947 | 0.863 | 0.475 |
| 9.95 | 94 | 0.995 | 0.960 | 0.850 | 0.529 |
| 9.90 | 100 | 0.990 | 1.000 | 0.820 | 0.659 |
| 10.00 | 92 | 1.000 | 0.947 | 0.853 | 0.516 |
| 10.00 | 100 | 1.000 | 1.000 | 0.800 | **0.750** |

The last row is the absolute hard bound: **with perfect CVVDP (JOD 10) and
perfect SSIMULACRA2 (100), K 95 still requires BA ≤ 0.75.** There is no
combination of the other two metrics that permits BA 1.0 or more.

The corresponding "max K at fixed BA" values make the class gap explicit:

| BA held at | q_ba | max K with JOD 10 / SS2 100 |
|---:|---:|---:|
| 0.44 (target) | 0.872 | 96.80 |
| 0.75 | 0.800 | 95.00 |
| 1.289 (AVM) | 0.699 | **92.49** |
| 1.583 (SVT) | 0.655 | **91.37** |
| 2.273 (eff-perc) | 0.569 | **89.23** |

So even a perfect CVVDP and perfect SS2 cannot rescue AVM's Butteraugli to 95.
The target is a **Butteraugli target**, and the conventional encoders are
1.3-2.3 BA at 100 kbps, not 0.44.

## 2. Measured bounds at 240p/100 kbps

All internal numbers below are from `docs/k-oracle.md`,
`docs/oracle-search.md`, `docs/eff-perc.md` and the per-lane combo documents.
"Matched" means a K interpolated to 100.000 kbps from a QP ladder that brackets
the target. "Ceiling" means the rate-free source-replacement bound from
`tools/oracle.py`; it is a pooling bound, not an encodable stream.

### 2.1 Internal oracles and ceilings

| mechanism | best matched K | ceiling | ceiling kind | ceiling gap to 95 | source |
|---|---:|---:|---|---:|---|
| nightly flat-P | 82.05 (pin 82.16) | **87.96** | source replacement, f44-59 | −7.04 | `docs/oracle-search.md` §10 |
| prediction/ME combinations | 82.33 | **88.56** | source replacement, best segment | −6.44 | `docs/combo-pred.md` |
| transform/entropy combinations | 82.49 | **88.46** | source replacement, best segment | −6.54 | `docs/combo-tx.md` |
| filter/allocation combinations | 82.96 | **88.96** | source replacement, f44-59 | −6.04 | `docs/combo-filter.md` |
| eff-perc perceptual RD | **85.10** | **90.12** | source replacement, f0-32 | −4.88 | `docs/eff-perc.md`, `docs/oracle-search.md` |
| deep inter anchor | 84.02 | 87.69 | all-inter allocation at 2.4x rate | −7.31 | `docs/k-oracle.md` |
| perfect anchor, spacing 2 | — | **89.8** | perfect neighbour, not decodable | −5.2 | `docs/hidden-anchor-oracle.md` |
| perfect anchor, spacing 4 | — | 87.2 | perfect reference, not decodable | −7.8 | `docs/hidden-anchor-oracle.md` |

No first-party matched-rate point is above 85.10. No first-party ceiling is
above 90.12. The only first-party bounds above SVT are the filter/allocation
ceiling 88.96, the eff-perc ceiling 90.12, and the non-decodable perfect-anchor
bound 89.8; all three are below AVM's realized 90.2. The perfect spacing-2
neighbour bound is the strongest structural bound in the repo and is still
5.2 K below 95; it is also not decodable because the perfect reference is not
in the DPB.

### 2.2 Conventional reference points

| reference | rate | JOD | BA | SS2 | K | note |
|---|---:|---:|---:|---:|---:|---|
| SVT-AV1 4.3.0 p0, CRF 36.5 | 99.4 | 9.826 | 1.583 | 88.87 | 88.64 | the Stage-1 wall, 88.8 at 100.0 |
| AVM e6f99d3, QP 124 | 99.3 | 9.862 | 1.289 | 90.38 | 90.19 | clean upstream; 90.2 at 100.0 |
| AVM e6f99d3, QP 110 | 121.2 | 9.872 | 1.185 | 91.37 | 90.84 | strict decode |
| AVM e6f99d3, QP 95 | 155.0 | 9.897 | 0.975 | 92.37 | 92.08 | strict decode |
| AVM e6f99d3, QP 90 | 175.1 | 9.933 | 0.950 | 92.63 | 92.43 | strict decode |
| AVM e6f99d3, QP 85 | 188.9 | 9.932 | 0.888 | 92.86 | 92.76 | strict decode |

AVM is the best realized conventional AV2 point. At **1.9x the target rate**
it is still **2.24 K short** of 95. The AVM QP80 and below points did not pass
`dav2d --strict 1` on this host; they are not used.

### 2.3 Conventional-class rate ladder (diagnostic)

Because the AVM ladder stops at 188.9 kbps on this host, the conventional-class
rate curve is extended with SVT-AV1 4.3.0 preset 0 on the same fixture. This is
a black-box reference ladder, not a first-party candidate and not an AV2
conformance claim. It answers one question: *how much rate does a
state-of-the-art conventional encoder need to reach K 95 on this content?*

| CRF | kbps | JOD | BA | SS2 | K |
|---:|---:|---:|---:|---:|---:|
| 36.5 | 99.4 | 9.826 | 1.583 | 88.87 | 88.64 |
| 34 | 110.1 | 9.842 | 1.412 | 89.33 | 89.43 |
| 31 | 125.7 | 9.858 | 1.265 | 90.10 | 90.23 |
| 28 | 138.1 | 9.865 | 1.192 | 90.78 | 90.68 |
| 25 | 155.8 | 9.880 | 1.065 | 91.50 | 91.43 |
| 22 | 178.1 | 9.890 | 0.991 | 91.80 | 91.87 |
| 18 | 202.2 | 9.898 | 0.933 | 92.38 | 92.29 |
| 15 | 229.7 | 9.911 | 0.859 | 92.79 | 92.79 |
| 12 | 263.1 | 9.920 | 0.803 | 93.11 | 93.17 |
| 9 | 350.0 | 9.932 | 0.679 | 94.06 | 94.06 |
| 6 | 446.1 | 9.955 | 0.621 | 94.60 | 94.58 |
| **3** | **777.6** | **9.969** | **0.519** | **95.61** | **95.43** |
| 1 | 1427.2 | 9.984 | 0.279 | 97.53 | 97.38 |

The first measured point above 95 is **777.6 kbps**, 7.8x the target.
Interpolating between 446.1 (94.58) and 777.6 (95.43) crosses 95 at roughly
**0.59-0.61 Mbps**, about 6x the target. The curve is shallow: from 100 kbps
to 200 kbps it gains about 3.5 K; from 200 to 400 kbps about 2.0 K; and even at
1.4 Mbps it is only 97.4. The K metric saturates near the top, and the last
few K are the expensive ones.

**What this means.** A 95 claim at 100 kbps is not asking for a 5% tool
improvement. It is asking for the quality that the best conventional encoders
deliver at **0.6-0.8 Mbps**, at **100 kbps**: a **6-8x rate-efficiency jump**.
AVM is better than SVT at the same rate, but even an optimistic *extrapolated*
AVM curve from its measured 188.9 kbps point lands around **0.3 Mbps** (3x),
not 100 kbps. No combination of AV2 tools measured in this repo is within a
factor of that.

### 2.4 Caveats

* **One fixture.** All numbers are the 480x240 / 60-frame NASA clip. The
  absolute K values are content-specific; the component arithmetic and the
  class ordering are not.
* **QP ripple.** The same recipe moves 0.3-1.9 K between adjacent QPs. A single
  point is not an oracle. Only a bracketing ladder at 100.000 kbps counts.
* **Ceilings are pooling bounds.** A source-replacement row replaces frames
  with the source and re-pools the sequence. It is not an encodable stream and
  cannot be quoted as a matched-rate result.
* **Perfect reference is not decodable.** The spacing-2 perfect-neighbour bound
  is a diagnostic; the perfect reference is not in the DPB.
* **Reference encoders are black boxes.** AVM and SVT source was not read,
  copied or transcribed. They are used exactly as the repo's existing wall
  references.

## 3. Where the gap lives

The metric decomposition says Butteraugli. The segment and structural work
says *where* that Butteraugli error is.

### 3.1 Segment attribution (vs SVT, from `docs/luma-audit.md`)

| replacement segment | nightly ΔK | eff-perc ΔK | read |
|---|---:|---:|---|
| key f0 | +0.00 | +0.01 | the key picture is not the K gap |
| f0-32 (key + static head) | +0.90 | +1.24 | predictor/static-chain value |
| f33-43 (fade onset) | +1.10 | +0.72 | allocation value |
| f44-59 (tail) | **+3.76** | **+1.46** | the largest single segment |

The decoded-domain hybrid attributes **3.32 K of the 3.93 K SVT gap to luma**
and **0.61 K to chroma** (`docs/eff-perc.md`). The keyframe is only 1/60 of the
metric: replacing it with SVT or AVM moves K by +0.00/+0.04. Its value is as a
predictor and as a bit budget, not as a picture.

The source-replacement ceilings point the same way but do not reach 95:
nightly's best is the **tail** (+5.911 → 87.96); eff-perc's best is the
**static head** (+5.017 → 90.12). Even infinite bits on one segment leave the
rest of the chain in place. No segment replacement is a 95 claim.

### 3.2 Structural attribution

From `docs/luma-audit.md`:

* **77-91% of P-frame error is in skip leaves.** The static-section drift is
  residual-free skip with nonzero-MV subpel MC; loop filters off and BAWP off
  do not fix it. It is a decision/reference problem, not a filter bug.
* **The fade tail is a low-frequency error floor.** In f44-59 the nightly
  stream is −5.06 dB full-res and **−8.66 dB at lp8** vs SVT: the tail gap is
  low-frequency drift, not fine detail.
* **The keyframe is a coefficient field.** 79% of the nightly stream is the
  key tile; 97% of its metered symbol bits are coefficient syntax; intra has no
  trellis. But the keyframe picture is not the K gap, and a better key alone
  buys +0.00/+0.04 K.
* **The coefficient entropy is within ≤0.3 K of its own bound.** Pooled
  contexts cost 39% more; the coder is 1.015x its own per-context bound. There
  is no hidden 5 K in entropy coding.

This is why a post-filter, a better keyframe, or a coefficient-entropy rewrite
cannot be the 95 path. The error is distributed over the P chain and is
low-frequency; it has to be prevented by prediction/allocation or by a
different coding paradigm, not cleaned up afterwards.

### 3.3 Two gaps, not one

It is important not to merge the two gaps into one number:

| gap | from → to | size | what it is | evidence |
|---|---|---:|---|---|
| implementation / tool composition | 85.10 → 90.2 | **5.1 K** | the first-party encoder is 5.1 K behind a full conventional AV2 encoder | eff-perc 85.10; AVM 90.2; ceilings 88.5-90.12 |
| class | 90.2 → 95 | **4.8 K** | the conventional class itself is 4.8 K short at 100 kbps | SVT needs 0.6-0.8 Mbps for 95; AVM 92.76 at 188.9 kbps |
| total | 85.10 → 95 | **9.9 K** | | |

The first gap is worth closing: it is conventional work with a measured
ceiling near 90 and a conformance story. The second gap is not an enlarged
version of the first; it is the distance from the best conventional encoder to
a quality class that the conventional curve only reaches at 6-8x the rate.
A program that closes the first gap and then claims the second is confusing
an implementation milestone with a new codec class.

## 4. Mechanism classes

The classes below are ordered by how much of the 95 gap they can plausibly
close. "Measured" means a matched-rate or ceiling measurement in this repo;
"estimate" means an engineering estimate from the measured conventional curve
and known class behavior, not a claim.

| # | class | expected K ceiling @100 kbps | conformance | first-party feasibility | data/compute | risk | oracle now |
|---:|---|---|---|---|---|---|---|
| 1 | within-AV2 conventional (prediction/transform/entropy/allocation) | **85-90.2 measured**; first-party ceiling 90.12 | AV2 yes | high; existing code | low | schedule | yes |
| 2 | encoder-side learned decision layer (existing syntax) | 86-91 (bounded by AV2 syntax) | AV2 yes | medium | medium | overfit / metric proxy | yes |
| 3 | signaled neural prediction/residual | 90-92 (estimate) | new AV2 tool or new codec; not v1.0.0 | low | high | high | no |
| 4 | learned transform/entropy (signaled) | 90-93 (estimate) | new tool/codec | low | high | high | no |
| 5 | generative detail synthesis | 90-95+ (unproven; high variance) | new codec/decoder | very low | very high | very high | no |
| 6 | neural post-filter (decoder-side, non-signaled) | +0.5-2 over its base; not 95 | bitstream yes, output non-normative | medium | medium | conformance/claim | yes, as post-processed Y4M |
| 7 | new codec class outside AV2 | unknown; only class with a conceivable 95 path | own bitstream/decoder | very low | very high | very high | no, needs new harness |

### 4.1 Within-AV2 conventional

**What it is:** prediction/ME, transform/quantisation, entropy coding, loop
filters, frame/reference structure and bit allocation, all using existing AV2
v1.0.0 syntax.

**Measured:** the best first-party matched point is **85.10** (eff-perc).
The best ceiling from a decodable-carrier lane is **88.96**
(filter/allocation); the strongest first-party ceiling is **90.12**
(eff-perc source replacement). Every composition lane measured at 240p is
below 88.8 except those two ceilings. The perfect spacing-2 neighbour bound is
89.8 and is not decodable. AVM, a full conventional AV2 encoder, is 90.2.

**Can it exceed AVM 90.2?** Not by any measured first-party mechanism. AVM is
a realized conventional point, so the class as a whole can reach 90.2; but the
first-party search is 5.1 K below it, and the strongest first-party bound is
0.08 K below AVM. To reach 95 the class would have to beat AVM by **4.8 K**,
which is outside every measured bound in the repo. "Conventional AV2 could in
principle be better than AVM" is true but is not a plan; the burden of proof is
a matched-rate ladder above 90.2, and no such result exists.

**Expected ceiling:** ~90.2 for the class, 88-90 for the current encoder.
The realistic conventional program is an **implementation gap** from 85.10 to
~90, not a 95 program.

**Conformance:** yes, if it stays inside AV2 v1.0.0. **Feasibility:** high; the
code and gates exist. **Data/compute:** low; CPU/GPU encode, no training.
**Risk:** schedule and opportunity cost, not feasibility. **Oracle:** yes,
`tools/oracle.py` with a bracketing ladder.

Sub-lane evidence:

* **Prediction/ME:** 22 legal combinations, best 82.331, ceiling 88.555.
  Multi-reference P loses 0.67/0.86 K; B frames measured negative;
  hierarchical B reaches 84.40 but stays 0.45 below the P chain; the
  hidden-anchor real schedule is 8-9 K below its perfect-reference oracle;
  TIP is 50.50 at G=4. The perfect spacing-2 neighbour bound is 89.8, below
  AVM.
* **Transform/entropy:** 27 legal sets, best 82.490, ceiling 88.459. RDOQ/TX
  search are inside the ripple; CfL/IBP/IST/DIP/IDTX/TCQ/MHCCP are negative or
  inert. Coefficient entropy is ≤0.3 K.
* **Filters/allocation:** 22 legal sets, best 82.957, ceiling 88.964. CDEF/
  CCSO/PC-Wiener/GDF stacks are negative or inert; the best arm is key-offset
  −26 + drift 2.0/2500, its realized K is **6.01 below its ceiling** and
  **5.84 below SVT**.
* **Structure:** deep inter anchor 84.02, allocation ceiling 87.69 at 2.4x
  rate; B/TIP/alt-ref are measured dead at 240p.

The conventional lane has been searched broadly and closed at the measured
wall. The remaining work is implementation quality, not an undiscovered 95
tool.

### 4.2 Learned/neural coding (first-party)

**The constraint is part of the mechanism.** No third-party code, model,
weight or asset. Any neural tool must be implemented and trained first-party.
The repo currently has no neural training or inference stack, and one 60-frame
240p fixture is not a training corpus. A learned path therefore starts with a
data-acquisition and framework effort, not with a model.

The current oracle only decodes AV2. A signaled neural tool cannot be scored
until a decoder implements it; a decoder-side post-filter can be scored as a
post-processed Y4M, but that is not the same claim.

#### 4.2.1 Neural prediction / residual

Two very different things share this name:

* **Encoder-side learned predictor/mode selector.** It chooses among existing
  AV2 modes or prices them with a learned distortion; the bitstream syntax is
  unchanged. This is conformance-safe and measurable now. Its ceiling is the
  existing syntax/quantiser, not a new class: the measured eff-perc decision
  layer buys +1.3 K over the drift wall (83.80 → 85.10) and +0.25 K over
  pipe-perc, and is capped at 85.10. A learned predictor might add another
  ~0.5-1.5 K at best; it cannot repeal the BA floor of the residual coder.
  **Estimate: 86-91, not 95.**
* **Signaled neural residual/prediction.** The decoder must run a new model or
  a new predictor mode. This is a new AV2 tool (spec amendment) or a new codec.
  A 20-30% BD-rate gain at this operating point is only about +1.5-2.5 K on the
  measured conventional curve, starting from AVM 90.2 → ~92. **Estimate:
  90-92, not 95.** The BA target needs a 3x reduction over AVM, not a 20-30%
  rate saving.

**Data/compute:** high; a learned predictor needs a large, diverse corpus and
GPU training. **Risk:** overfitting to the fixture, non-conformance, and a
model that improves PSNR/SS2 but not the BA term. **Oracle:** encoder-side
variant yes; signaled variant no.

#### 4.2.2 Learned transform / entropy

A learned transform or entropy model is a new normative tool: the decoder must
invert the transform or parse the new symbols. It is not an AV2 v1.0.0 stream.
The measured entropy headroom is tiny (≤0.3 K), and the transform lane's best
ceiling is 88.46, so the upside is in prediction/residual, not in a new
transform. A learned transform with a 10-20% rate gain is worth ~+0.7-1.5 K on
the measured curve. **Estimate: 90-93, not 95.** Cost and risk are the same as
4.2.1; conformance requires a spec change or a new codec.

#### 4.2.3 Generative detail synthesis

A generative decoder can synthesize plausible texture at low rate. That is the
one mechanism class with a theoretical path to a much higher perceptual score
at 100 kbps. But the score is **full-reference**: BA and SS2 compare the
decoded picture to the source, not to a distribution of plausible pictures.
Hallucinated detail that is not the source's detail is an error, and at
BA 0.44 the allowance is near-transparency. A generative model would have to
synthesize the *actual* missing detail, not merely plausible detail. No
first-party evidence shows that at 100 kbps on this fixture; the measured
conventional curve is still at BA 0.62 at 446 kbps.

**Expected K:** unknown; 90-95+ in the optimistic case, but unproven and
high-variance. It is the only class that could in principle clear 95, and it
would be a new codec/decoder. **Data/compute:** very high (generative training
and inference). **Risk:** very high: metric divergence, non-reproducible
sampling, conformance, deployment cost. **Oracle:** no, not as an AV2 stream;
a new harness and decoder are required.

#### 4.2.4 Neural post-filter

A decoder-side non-signaled filter is the cheapest neural experiment: decode
the existing stream, run a first-party model, score the filtered Y4M. It can
remove ringing/blocking and perhaps improve BA by a few tenths. It cannot
recover the low-frequency tail error or the missing texture: at 100 kbps the
error is structural, not a filter artifact. The existing loop filters (CDEF,
CCSO, GDF, Wiener/PC-Wiener) are already in the measured chain, and the
marginal filter-combination search is negative or inert at 240p. A learned
post-filter might add **+0.5-2 K** over its base: 85.1 → 85.6-87.1 on the
current encoder, or 90.2 → 90.7-92.2 on AVM. **It is not a 95 path.**

**Conformance:** the bitstream remains AV2-conformant, but the normative
decoder output is not the claimed output. The claim is "AV2 + post-filter",
which is a different experiment from "AV2". If the product ships the filter,
the filter is part of the product; for a fair comparison it must be applied to
both our stream and the reference stream. **Measurable now:** yes, as a
post-processed Y4M; not as a stream-only K.

### 4.3 Hybrid signaled tools vs decoder-side non-signaled filters

| property | signaled tool | decoder-side non-signaled filter |
|---|---|---|
| bitstream | new syntax; AV2-conformant only if standardized | existing AV2 bitstream |
| decoder | must implement the tool; pinned `dav2d`/`avmdec` will reject unknown syntax | any post-processor; decoder unchanged |
| rate | adds syntax/header bits and model/side-information cost | no rate change |
| quality claim | codec output | post-processed output, not normative decode |
| oracle | no, until a decoder exists | yes, on decoded Y4M |
| conformance risk | high; spec/decoder change | bitstream conforms, claim is non-normative |
| first-party feasibility | low; new syntax + decoder + training | medium; model + inference |
| 95 potential | only as part of a new codec/tool | no; bounded by decoded information |

The distinction matters for any future claim:

* A **signaled** learned tool is a new codec/tool. It needs a bitstream
  specification, a normative decoder, a conformance suite and a matched-rate
  comparison. It cannot be called an AV2 v1.0.0 result.
* A **non-signaled** filter preserves the bitstream but changes the output.
  It is legitimate as a deployment feature, but it must be reported as
  post-processing and compared with the same post-filter applied to the
  reference. It cannot be mixed into the AV2 K column.

### 4.4 New codec class outside AV2 conformance

This is the only class with a conceivable path to 95 at 100 kbps. It means a
new bitstream and a new normative decoder, most plausibly a neural/generative
codec, or a substantially different conventional design. It is not an AV2
profile and cannot be validated with `dav2d`/`avmdec` as the claim; those
become external references only.

**Why it is the only 95 candidate:** the measured SVT curve needs
0.6-0.8 Mbps for K 95; AVM's measured points stop at 188.9 kbps, still 2.24 K
short, and an optimistic extrapolation puts its 95 crossing near 0.3 Mbps.
A 3-8x rate-efficiency jump is a different coding paradigm, not a tool. The
metric math also says the new class must deliver **BA ≤0.44 at JOD 9.9 /
SS2 92** (or ≤0.75 with perfect JOD/SS2), which is near-transparency at
100 kbps.

**Conformance:** own bitstream, own normative decoder, own conformance suite;
independent decoder verification. **First-party feasibility:** very low in the
current repo; no neural stack, no corpus. **Data/compute:** very high;
multi-quarter/year, team, GPU training and inference. **Risk:** very high; no
guarantee the class reaches BA 0.44; deployment on RTX 3060 12 GB / 1 fps
1080p may be impossible for a heavy model. **Oracle:** no as an AV2 stream;
needs a new harness that measures rate and decoded quality, with a conformance
suite.

### 4.5 What is not a mechanism

* **Segment ceiling.** Source replacement is infinite bits on a frame range.
  The best first-party ceiling is 90.12; it is not an encodable 95 stream.
* **Perfect reference.** The spacing-2 perfect neighbour is 89.8 and is not
  decodable. It is a bound, not a candidate.
* **Metric gaming.** Optimizing only CVVDP cannot work: JOD is already 9.77 and
  the q_cvvdp term is 48.84 of 50. The hard bound says BA ≤0.75 even at perfect
  JOD/SS2, so there is no "cheap" 95 via the other two metrics.
* **A third-party encoder output.** AVM/SVT are references, not first-party
  mechanisms. Their output cannot be the deliverable.

## 5. What a 95 claim would require

A valid 95 claim at 240p/100 kbps needs all of the following. Missing any one
turns it into a ceiling, a post-processed result or a non-conformant stream.

1. **Matched-rate K 95.0 at exactly 100.000 kbps.** A QP ladder that brackets
   the target (three points better than two), K via `tools/kpi.py`, no
   single-point or nearest-rate result. The QP ripple is up to 1.9 K; a
   knife-edge point is not a claim.
2. **Component proof.** Report JOD, BA, SS2 and the q values. The point must
   satisfy the iso-K equation. At JOD 9.90 / SS2 92, BA must be ≤0.435; the
   absolute hard bound is BA ≤0.75 even with JOD 10 / SS2 100. A **95 claim**
   with BA 1.0 or more is arithmetically impossible.
3. **Bitstream/decoder conformance.** For AV2: every stream passes
   `tools/validate_dav2d.sh` and `dav2d --strict 1` 60/60, with the encoder
   `--video-recon` equal to the decoded payload where applicable. No new
   syntax outside the spec. For a new codec: its own normative decoder and
   conformance suite, plus an independent decoder check.
4. **No third-party code/assets.** Original implementation from the AV2 spec
   or a new design. No AVM/SVT/clave/av2-turbo source, no third-party model
   code, no pretrained weights, no third-party training assets. Any corpus and
   weights need provenance.
5. **Reproducible streams.** Encoder commit/binary, exact command, fixture
   provenance, stream MD5, decoder version, CVVDP version/display model and
   Butteraugli/SSIMULACRA2 versions. Another person must be able to rerun it
   and get the same K within the documented ripple.
6. **Independent verification.** A second decoder for the new bitstream or a
   conformance test; a second team reruns the encode and the metric. A
   self-reported K with no independent decode is not a claim.
7. **Headroom.** A rate-free/segment ceiling above 95 for the same mechanism,
   or a second matched point above 95. A single point exactly at 95 with no
   headroom is a measurement, not a program.
8. **Deployment truth.** If the mechanism is neural, it must fit the stated
   target (RTX 3060 12 GB baseline, 1 fps 1080p or the 240p point), with
   deterministic inference. A cloud-only or non-reproducible model is not a
   product result.

## 6. Ranked paths and recommendation

### 6.1 Ranking

| rank | path | expected K @100 | conformance | cost | risk | verdict |
|---:|---|---|---|---|---|---|
| **1** | **close the implementation gap to ~90** (conventional AV2: prediction, transform, allocation, filters, structure) | 85 → 88.8 → ~90.2 | AV2 yes | months, existing code | schedule; no new class | **GO** |
| 2 | conventional AV2 research beyond AVM (new tools inside the spec) | unproven; would need >90.2 first | AV2 if spec-compliant | high | no measured evidence | **NO-GO for 95**; revisit only with a >92 oracle |
| 3 | neural post-filter / hybrid on top of conventional | +0.5-2 over base; not 95 | bitstream yes, output non-normative | medium | claim/conformance; limited upside | **bounded probe at most** |
| 4 | signaled learned prediction/transform/entropy | 90-93 (estimate) | new tool/codec; not v1.0.0 | very high | data/compute, conformance | **NO-GO for 95** unless a new-codec program |
| 5 | new codec class outside AV2 (neural/generative) | unknown; only conceivable 95 path | own bitstream/decoder | very high (multi-quarter/year) | very high | **NO-GO now**; separate program only |

### 6.2 Recommendation

**GO: path 1, close the implementation gap to ~90 first.** The measured
first-party envelope is 85.10, the strongest first-party ceiling is 90.12, and
AVM is 90.2. The next milestone is SVT parity 88.8, then AVM parity ~90. This
is the only path with a measured mechanism above the current point and a
credible conformance story. It is an implementation/tool-composition program,
not a 95 program.

**NO-GO: start a 95 research program now.** There is no first-party mechanism
above 90.2, no first-party ceiling above 95, and the conventional class needs
0.6-0.8 Mbps for K 95. A 95 program would be a new-codec program with a new
bitstream, decoder, conformance suite, first-party corpus and GPU training
budget. That is not a branch-sized task, and starting it before the
conventional gap is closed would spend the remaining budget on the least
evidenced path.

**Reopen gate for 95.** Reopen only when a first-party mechanism satisfies
both:

1. a matched-rate K **>90.2** at 100.000 kbps from a bracketing ladder
   (ideally >92, so there is headroom), and
2. a rate-free/segment ceiling **>95** for the same mechanism.

If both hold, the mechanism is worth a 95 investigation. If either fails, it
is not a 95 path; report the envelope and stop. This is the same oracle-first
rule as `docs/oracle-search.md`, with 95 instead of 88.8 as the wall.

### 6.3 What would have to be true for 95

* A first-party mechanism produces a **matched-rate K ≥95 at 100 kbps**, not a
  ceiling, on the pinned fixture.
* It reaches **BA ≤0.44 at JOD 9.9 / SS2 92**, or an equivalent iso-K point
  with **BA ≤0.75** and better JOD/SS2. Given the measured conventional BA of
  1.29 (AVM) and 2.27 (eff-perc), this is a **3-5x Butteraugli improvement at
  the same rate**, not a 5% tool gain.
* The mechanism is either an AV2-conformant tool (spec amendment + decoder) or
  a new codec with a normative decoder and conformance suite. A non-signaled
  post-filter can be a product feature, but it is not a codec-class 95 claim.
* It is built without third-party code/assets, with a first-party training
  corpus and a compute plan that fits the deployment target.
* It reproduces: exact streams, MD5s, decoder, metric versions, and an
  independent decode.
* It survives the measured rate ladder: a second point above 95, or a ceiling
  above 95, so the result is not a single QP-ripple artifact.

Until then, the honest statement is: **95 at 240p/100 kbps is not reachable by
any measured first-party mechanism; it is a new-codec-class research question,
not an AV2 encoder target.**

## 7. Reproduction

Fixture: `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, 8-bit 4:2:0, 30 fps.
K: `tools/kpi.py`. CVVDP 0.5.7, `standard_fhd`, MPS. Butteraugli and
SSIMULACRA2: `tools/extra_metrics.py`. Decoder: `tools/bin/dav2d-macos`
(pinned `f9dafbc`). AVM: upstream `e6f99d3`. SVT: 4.3.0 preset 0.

### 7.1 AVM rate probe

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
AVMENC=${AVMENC:?set to the upstream e6f99d3 avmenc build}
for qp in 124 110 95 90 85; do
  "$AVMENC" --end-usage=q --cpu-used=6 --threads=8 \
      --limit=60 --qp=$qp -o /tmp/avm_qp${qp}.ivf "$REF"
  tools/bin/dav2d-macos --strict 1 -i /tmp/avm_qp${qp}.ivf \
      -o /tmp/avm_qp${qp}.y4m
  python3 tools/quality.py --ref "$REF" --test /tmp/avm_qp${qp}.y4m \
      --json /tmp/avm_qp${qp}.q.json
  python3 tools/extra_metrics.py --ref "$REF" --test /tmp/avm_qp${qp}.y4m \
      --json /tmp/avm_qp${qp}.e.json
  python3 tools/kpi.py --quality-json /tmp/avm_qp${qp}.q.json \
      --extra-json /tmp/avm_qp${qp}.e.json
done
```

The binary used here is the clean upstream build documented in
`bench/avm_baseline/README.md` (SHA-256
`20503ec8d48b368747bf6f6dfae9bcb3ce5fe3d22411dd50255ec2b5b3d14b1c`).
QP 80 and below did not pass `dav2d --strict 1` on this host and are excluded.
They are not needed for the conclusion: the valid 188.9 kbps point is already
2.24 K short.

### 7.2 Conventional-class SVT ladder

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m
for crf in 36.5 34 31 28 25 22 18 15 12 9 6 3 1; do
  SvtAv1EncApp -i "$REF" -b /tmp/svt_crf${crf}.ivf --preset 0 --crf $crf \
      --fps-num 30 --fps-denom 1 --progress 0
  ffmpeg -v error -y -i /tmp/svt_crf${crf}.ivf -pix_fmt yuv420p \
      -f yuv4mpegpipe /tmp/svt_crf${crf}.y4m
  python3 tools/quality.py --ref "$REF" --test /tmp/svt_crf${crf}.y4m \
      --json /tmp/svt_crf${crf}.q.json
  python3 tools/extra_metrics.py --ref "$REF" --test /tmp/svt_crf${crf}.y4m \
      --json /tmp/svt_crf${crf}.e.json
  python3 tools/kpi.py --quality-json /tmp/svt_crf${crf}.q.json \
      --extra-json /tmp/svt_crf${crf}.e.json
done
```

This is a black-box reference ladder, not a first-party candidate and not an
AV2 conformance claim. The CRF 36.5 point reproduces the documented SVT wall
(24,844 B / 99.4 kbps / JOD 9.826 / BA 1.583 / SS2 88.87 / K 88.64).

### 7.3 Internal evidence

* `docs/k-oracle.md` — the 83.80/85.10 internal walls and the 90/95 stage
  definitions.
* `docs/oracle-search.md` — the 88.8/90.2 walls, the candidate queue, the
  oracle rules and the generated results.
* `docs/eff-perc.md` — the 85.10 envelope and its sweep.
* `docs/luma-audit.md` — the segment/band attribution and the 3.32 luma +
  0.61 chroma split.
* `docs/combo-pred.md`, `docs/combo-tx.md`, `docs/combo-filter.md` — the
  measured composition ceilings 88.56 / 88.46 / 88.96.
* `docs/hidden-anchor-oracle.md` — the perfect spacing-2/4/8 bounds.
* `docs/negative-results.md` — the do-not-repeat list, including AQ, delta-Q,
  dead zones, multi-reference and B/TIP.
* `docs/oracle-search.md` — the learned/tool-class closure at 0 K and the
  "95 is a new-codec program" note in the standing queue.
* `docs/tool-matrix.md` — the legal carriers and the measured dead tools.
* `tools/kpi.py` — the exact K formula; `tools/oracle.py` — the matched-rate
  and segment harness.

## Appendix A: measured AVM rate probe

| QP | bytes | kbps | JOD | BA | SS2 | K | strict decode |
|---:|---:|---:|---:|---:|---:|---:|---|
| 124 | 24,836 | 99.3 | 9.862 | 1.289 | 90.38 | 90.19 | 60/60 |
| 110 | 30,291 | 121.2 | 9.872 | 1.185 | 91.37 | 90.84 | 60/60 |
| 95 | 38,754 | 155.0 | 9.897 | 0.975 | 92.37 | 92.08 | 60/60 |
| 90 | 43,771 | 175.1 | 9.933 | 0.950 | 92.63 | 92.43 | 60/60 |
| 85 | 47,224 | 188.9 | 9.932 | 0.888 | 92.86 | 92.76 | 60/60 |

At 100.0 kbps the QP124/QP122 interpolation is **90.2** (JOD 9.860, BA 1.29,
SS2 90.39), the value used throughout this document.

## Appendix B: measured SVT conventional ladder

| CRF | bytes | kbps | JOD | BA | SS2 | K |
|---:|---:|---:|---:|---:|---:|---:|
| 36.5 | 24,844 | 99.4 | 9.826 | 1.583 | 88.87 | 88.64 |
| 34 | 27,515 | 110.1 | 9.842 | 1.412 | 89.33 | 89.43 |
| 31 | 31,433 | 125.7 | 9.858 | 1.265 | 90.10 | 90.23 |
| 28 | 34,517 | 138.1 | 9.865 | 1.192 | 90.78 | 90.68 |
| 25 | 38,960 | 155.8 | 9.880 | 1.065 | 91.50 | 91.43 |
| 22 | 44,519 | 178.1 | 9.890 | 0.991 | 91.80 | 91.87 |
| 18 | 50,546 | 202.2 | 9.898 | 0.933 | 92.38 | 92.29 |
| 15 | 57,413 | 229.7 | 9.911 | 0.859 | 92.79 | 92.79 |
| 12 | 65,770 | 263.1 | 9.920 | 0.803 | 93.11 | 93.17 |
| 9 | 87,489 | 350.0 | 9.932 | 0.679 | 94.06 | 94.06 |
| 6 | 111,537 | 446.1 | 9.955 | 0.621 | 94.60 | 94.58 |
| 3 | 194,411 | 777.6 | 9.969 | 0.519 | 95.61 | 95.43 |
| 1 | 356,797 | 1427.2 | 9.984 | 0.279 | 97.53 | 97.38 |

The first point above 95 is 777.6 kbps; the interpolation between 446.1 and
777.6 crosses 95 at about 0.59 Mbps. Even at 1.43 Mbps the metric is only
97.38: the last few K of K are the expensive ones.
