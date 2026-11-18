# 4:2:0 oracle: lossless wall, 100 kbps bracket, and the 95 verdict

Status: **measured oracle definition, read-only on encoder code.** Branch
`exp/luma-420`, base `cbfc0061`. This commit adds this document only. Every
stream and hybrid here is 8-bit **4:2:0**; no 4:2:2 or 4:4:4 row is mixed in.
External encoders (SVT-AV1, AVM) are black-box binaries. No third-party
encoder/decoder source, model or weight is read, copied or vendored.

The two questions this document separates:

1. **The lossless wall.** What is exact 4:2:0 at this content, and why is
   K=100 there not a 100 kbps oracle?
2. **The 100 kbps limit.** What is the best measured 4:2:0 K at 100 kbps, what
   do allocation and source-replacement ceilings bound, and does any measured
   bound support K=95?

## 0. TL;DR

| quantity | value | kind |
|---|---:|---|
| exact lossless 4:2:0, av2enc `--video-lossless` | **3,631.4 kbps** (907,849 B / 2 s) | decodable, exact |
| exact lossless 4:2:0, FFV1 cross-check | 7,462.7 kbps (1,865,674 B / 2 s) | black-box, exact |
| K at exact reconstruction | **100.000** | by definition |
| best decodable stream at ~100 kbps | **90.192** at 99.344 kbps (90.188 at 100.000) | matched ladder |
| best per-frame allocation ceiling | **90.965** (AVM ladder, 99.268 kbps) | decoded-domain mix |
| best allocation, SVT-AV1 p0 ladder | **90.438** (99.688 kbps) | decoded-domain mix |
| best allocation, av2enc eff-perc ladder | **86.417** (99.956 kbps) | decoded-domain mix |
| strongest single-segment source ceiling | **94.072** (AVM, f0-32) | infinite bits on 33 frames |
| perfect chroma at a measured luma | **+0.46 to +0.93 K** | free-plane bound |
| perfect luma at a measured chroma | **+5.95 to +7.79 K** | free-plane bound |
| AVM + source luma (free perfect luma) | **96.144** | not a stream |
| SVT + source luma (free perfect luma) | **95.798** | not a stream |

**No rate-feasible measured bound reaches 95 at 100 kbps.** The best
decodable point is 90.192, the strongest allocation ceiling is 90.965, and the
strongest single-segment source ceiling is 94.072. The only measured values
above 95 at a ~100 kbps base rate are free-luma hybrids: they replace the
entire luma plane with the source and charge **zero luma rate**. They prove
that luma is the binding channel, not that 95 is encodable at 100 kbps.

The lossless point is K=100 at **3.63 Mbps**, 36x the target rate. It is the
top of the K axis, not a 100 kbps oracle. At 100 kbps the measured limit is
bracketed below 95 for every encodable mechanism; a 95 claim would require a
new luma-coding class, not another AV2 tool.

## 1. Definitions

`K` is `tools/kpi.py`:

```text
q_cvvdp = clamp(JOD / 10, 0, 1)
q_ba    = 1 / (1 + BA / 3)
q_ss2   = clamp((SS2 + 50) / 150, 0, 1)
K       = 100 * (0.50*q_cvvdp + 0.25*q_ba + 0.25*q_ss2)
```

Every row is 480x240, 30 fps, 60 frames, 8-bit 4:2:0
(`testdata/nasa_480x240_30fps_60f.y4m`). The duration is exactly 2.000 s, so
`kbps = IVF_bytes * 8 / 2 / 1000`; a 25,000 B file is 100.000 kbps.

Five bound types are used, and they are never interchangeable:

| bound | what it is | encodable? |
|---|---|---|
| **exact lossless** | a real stream whose decoded planes equal the source | yes, at its own rate |
| **matched-rate fixed-QP** | a real stream, K interpolated to 100.000 kbps over a QP ladder | yes |
| **per-frame allocation** | best mix of decoded frames from one encoder's ladder under a 25,000 B budget, assembled and re-scored | no, decoded-domain mix |
| **source-segment** | replace one frame range with source frames, charge no rate for it, re-score | no, infinite bits on the segment |
| **source-plane** | replace Y or U/V with the source plane, charge no rate for it, re-score | no, free perfect plane |

The allocation bound is an **upper bound on per-frame allocation from that
ladder if frames are treated independently** (`docs/k-oracle.md`). The
source-segment and source-plane rows are **pooling bounds on what that part
could buy**, not streams and not global upper bounds. Only a matched-rate row
is an oracle for an encoder.

## 2. The lossless wall

### 2.1 av2enc exact lossless is legal on this path

`--video-lossless` is the CPU video writer's Q=0 mode. It accepts 8-bit 4:2:0
and was run on the fixture:

```sh
av2enc testdata/nasa_480x240_30fps_60f.y4m \
  -o /tmp/nasa_ll.ivf --video-lossless --frames 60 --cpu-threads 8
```

| item | value |
|---|---:|
| IVF bytes | **907,849** |
| duration | 2.000 s |
| bitrate | **3,631.396 kbps** (3.631 Mbps) |
| bits per pixel | 1.0508 |
| ratio to 100 kbps | **36.31x** |
| avmdec reconstruction | 60 frames, 10,368,000 B, payload-identical |
| dav2d `--strict 1` reconstruction | 60 frames, payload-identical |
| source payload MD5 | `20d6540feffa21d3f9eae6b8e77265bb` |

The decoder output and the source payload are byte-identical, not merely
PSNR-close. `--video-lossless` is therefore legal and exact on this 4:2:0
path; the black-box FFV1 run is only a cross-check.

### 2.2 Black-box cross-check

```sh
ffmpeg -i testdata/nasa_480x240_30fps_60f.y4m -c:v ffv1 -level 3 \
  -threads 8 -y /tmp/nasa_ffv1.mkv
```

| item | value |
|---|---:|
| Matroska bytes | **1,865,674** |
| bitrate | **7,462.696 kbps** (7.463 Mbps) |
| bits per pixel | 2.1593 |
| ratio to 100 kbps | **74.63x** |
| decode to Y4M | byte-identical to the source (`cmp` clean) |

The first-party lossless writer is 2.05x smaller than FFV1 on this clip, but
both are far above the deployment rate.

### 2.3 K=100 is definitional at lossless

At exact reconstruction, `JOD=10`, `BA=0`, `SS2=100`, so
`q_cvvdp=q_ba=q_ss2=1` and `K=100.000`. Measured on the source against
itself: `cvvdp=10.0000`, `butteraugli=0.0000`, `ssimulacra2=100.000`,
`K=100.00`. The exact lossless stream is K=100 **at 3,631.4 kbps**.

This is **not a 100 kbps oracle**:

* It is a point at 36x the target rate. It says the metric can reach its
  ceiling when both planes are exact; it says nothing about what is encodable
  at 100 kbps.
* A lossless stream codes luma and chroma exactly. At 100 kbps neither plane
  can be exact; the question is how the rate is split and how the error is
  shaped.
* The K axis saturates at 100. Quoting "K=100 at lossless" as a bound on the
  100 kbps limit is the same error as quoting a 0-byte source as a rate bound.

The useful lossless fact is the distance: the top of the K axis costs
3.63 Mbps on this clip, while the best measured 100 kbps stream is 90.2 K.
The 100 kbps problem lives entirely in the 90-100 band.

## 3. Fixed-QP points at 100 kbps

All rows are real decodable streams. K is measured by `tools/quality.py`
(AVM decoder, 4:2:0 only), `tools/extra_metrics.py` and `tools/kpi.py`.
Matched K is linear interpolation of K across the encoder's own QP ladder to
100.000 kbps; a single point is labelled as such.

| point | bytes | kbps | JOD | BA | SS2 | K | matched K @100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| av2enc nightly QP106 | 25,008 | 100.032 | 9.7334 | 3.4202 | 80.837 | **82.155** | 82.155 (single point) |
| av2enc eff-perc QP119 | 25,230 | 100.920 | 9.7713 | 2.2630 | 82.441 | **85.180** | **85.099** (official QP119/QP120) |
| SVT-AV1 4.3.0 p0 CRF 36.5 | 24,844 | 99.376 | 9.8262 | 1.5827 | 88.866 | **88.641** | **88.690** (local 7-point ladder) |
| AVM e6f99d3 QP124 | 24,836 | 99.344 | 9.8616 | 1.2889 | 90.382 | **90.192** | **90.188** (local 7-point ladder) |

The two pins re-score exactly: `tools/oracle.py` reports nightly 25,008 B /
100.032 kbps / K 82.155 and eff-perc 25,230 B / 100.920 kbps / K 85.180.
SVT and AVM re-score to 88.641 and 90.192. The eff-perc pin requires a clean
build at this base; a stale `build-dev` produced 22,446 B and was discarded.
The clean binary is functionally `cbfc0061` (the only commits ahead in the
source tree are docs; the binary was built at `8ccf72d8`, whose diff from
`cbfc0061` is two docs files).

The local 7-point ladders used for the allocation section are:

| encoder | points | rate span |
|---|---|---:|
| av2enc eff-perc | QP 110, 113, 116, 119, 122, 125, 128 | 126.4 → 85.0 kbps |
| SVT-AV1 p0 | CRF 30, 33, 35, 36.5, 38, 40, 42 | 129.8 → 83.4 kbps |
| AVM e6f99d3 | QP 112, 116, 120, 124, 128, 132, 136 | 117.4 → 82.4 kbps |

## 4. Per-frame allocation ceiling

The fixed-QP point spends the same QP on every frame. The allocation ceiling
asks the next question: **if every frame could choose its own QP from that
encoder's ladder, under the same 25,000 B budget, what K would the assembled
sequence reach?**

### 4.1 Method

1. Encode the 7-point ladder above. Decode every stream to 4:2:0 Y4M.
2. Read per-frame coded sizes from the IVF temporal units. For av2enc and SVT
   each TU is one output frame (60 TUs / 60 frames). AVM has 32 TUs and 60
   output frames; the output-frame group of every TU was measured by decoding
   each IVF prefix and differencing the output count. The GOP groups are
   QP-invariant on this fixture, so a group is the allocation unit and its
   cost is the whole TU.
3. Score every decoded frame against the source: per-frame JOD from CVVDP
   `Q_tc` (the same `standard_fhd`/MPS model as `tools/quality.py`),
   per-frame Butteraugli and SSIMULACRA2 from the pinned tools.
4. Solve a multiple-choice knapsack over frame groups: one variant per group,
   cost = the TU bytes, utility = `0.5*JOD/10 + 0.25*q_ba + 0.25*q_ss2`,
   budget = 25,000 B minus the IVF header and 12 B/TU. The DP is exact for
   the additive proxy; three objective variants (K proxy, BA+SS2, SS2-only)
   were run and the best measured assembly is reported.
5. Assemble the chosen frames into one 4:2:0 Y4M and re-score it with the
   normal oracle. The rate is the sum of the chosen TU bytes plus the real
   IVF overhead, not the Y4M size.

The assembled file is **not a bitstream**. It ignores temporal consistency,
per-frame QP signalling and the fact that changing one frame's QP changes the
predictions of later frames. It is a decoded-domain upper bound on what this
ladder's per-frame allocation can buy, exactly the "allocation bound" in
`docs/k-oracle.md`.

### 4.2 Result

| encoder | fixed matched K | allocation K | allocation kbps | Δ vs fixed | JOD | BA | SS2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AVM e6f99d3 | 90.188 | **90.965** | 99.268 | **+0.777** | 9.8627 | 1.1354 | 91.095 |
| SVT-AV1 p0 | 88.690 | **90.438** | 99.688 | **+1.748** | 9.8537 | 1.2156 | 90.272 |
| av2enc eff-perc | 85.069 | **86.417** | 99.956 | **+1.348** | 9.8274 | 1.9830 | 83.373 |

The AVM allocation is the strongest measured allocation ceiling:
**90.965 at 99.268 kbps** (a K-proxy assembly at 99.356 kbps measures 90.961).
It is still **4.035 K below 95**. The mix uses 24,817 B of the 25,000 B budget;
the remaining 183 B cannot be spent without a multi-group swap, so the ceiling
is quoted at its measured rate rather than padded to exactly 100.000 kbps.
The SVT and av2enc mixes use 24,922 B and 24,989 B respectively.

The allocation gain is real but modest: 0.8 K for AVM, 1.7 K for SVT, 1.3 K
for av2enc. It comes from spending the high-rate variants on the hard frames
and the low-rate variants on the easy ones. It does not change the class of
the result. The DP maximizes an additive per-frame proxy and the assembled
stream is then measured; 90.965 is the best measured allocation point, not a
proof that no other mix could be slightly higher.

## 5. Source-segment ceilings

Replace one frame range of the base decoded stream with the source and
re-score the 60-frame sequence. The replacement is free (infinite bits on
that range), so this is a ceiling on the value of that range, not a stream.
Rows are from `tools/oracle.py` for av2enc and AVM; the SVT rows use the same
segment definition on the decoded SVT Y4M because `avmdec` does not decode
AV1.

| base | key f0 | f0-32 | f33-43 | f44-59 |
|---|---:|---:|---:|---:|
| nightly 82.155 | +0.084 (82.239) | +3.700 (85.855) | +2.212 (84.367) | **+5.911 (88.066)** |
| eff-perc 85.180 | +0.125 (85.305) | **+5.017 (90.197)** | +2.119 (87.300) | +3.815 (88.995) |
| SVT p0 88.641 | +0.144 (88.785) | **+4.459 (93.100)** | +1.743 (90.384) | +2.700 (91.341) |
| AVM 90.192 | +0.091 (90.283) | **+3.881 (94.072)** | +1.839 (92.031) | +2.263 (92.455) |

The strongest single-segment row is **AVM f0-32 = 94.072**, still 0.928 K
below 95. Infinite bits on the key alone are worth +0.09 K: the keyframe is
1/60 of the metric, exactly as the luma audit found. The segment that owns
the largest nightly gain is the fade tail (+5.911), but even that leaves the
nightly base at 88.066.

## 6. Luma and chroma attribution

Replace exactly one plane with the source, keep the other plane decoded, and
re-score. This charges no rate for the replaced plane, so it is the maximum a
perfect coder of that plane could buy **at the other plane's measured
quality**. It is a free-plane bound, not a stream.

| base | base K | source Y (keep base UV) | ΔK | source UV (keep base Y) | ΔK |
|---|---:|---:|---:|---:|---:|
| eff-perc QP119 | 85.180 | **92.965** | **+7.785** | **86.109** | **+0.929** |
| SVT p0 CRF36.5 | 88.641 | **95.798** | **+7.157** | **89.103** | **+0.462** |
| AVM QP124 | 90.192 | **96.144** | **+5.952** | **90.670** | **+0.478** |

Read the two columns as different experiments:

* **Perfect chroma buys less than 1 K.** The largest measured chroma gain is
  +0.929 K (eff-perc); AVM and SVT are +0.478/+0.462. This is consistent with
  the audit's 0.61 K chroma share and with `docs/eff-chroma.md`'s +0.44 K
  SVT-chroma ceiling. A perfect chroma coder cannot reach 95 from any measured
  luma.
* **Perfect luma buys roughly 6-8 K.** The largest is +7.785 K at eff-perc
  luma; AVM and SVT are +5.952/+7.157. This is the absolute cost of the luma
  error, not the gap to SVT. The audit's 3.32 K is the gap to SVT's luma at
  our chroma, a smaller and different quantity.

The direct cross-candidate split at the current pins reproduces the audit's
shape. Swapping SVT's decoded chroma into eff-perc's luma gives
**85.750** (+0.570 K); swapping eff-perc's chroma into SVT's luma gives
**87.960** (−0.681 K from SVT's 88.641). The 3.461 K eff→SVT gap therefore
decomposes as **2.780 K luma + 0.681 K chroma**, versus the audit's
3.32 K luma + 0.61 K chroma on its slightly different base. Chroma is a
sub-1 K term; luma owns the rest.

The free-luma rows are the important 95 result:

* AVM + source luma = **96.144** (JOD 9.9903, BA 0.3725, SS2 93.725).
* SVT + source luma = **95.798** (JOD 9.9822, BA 0.4084, SS2 93.296).
* eff-perc + source luma = 92.965, still below 95 because its chroma and
  other errors are larger.

A perfect luma coder on AVM's chroma crosses 95. That is the only measured
**~100 kbps** bound in this document above 95 (exact lossless is above 95 only
at 3.63 Mbps). It charges zero luma rate, so it is a statement about **where
the quality is**, not a statement that an encoder can deliver it at 100 kbps.

## 7. The 100 kbps bracket

### 7.1 Bounds, strongest first

| bound | K | rate | encodable? | reaches 95? |
|---|---:|---:|---|---|
| exact lossless | 100.000 | 3,631.4 kbps | yes | yes, at 36x rate |
| AVM + source luma | **96.144** | 99.344 (free luma) | no | yes, free-plane only |
| SVT + source luma | 95.798 | 99.376 (free luma) | no | yes, free-plane only |
| AVM source f0-32 | **94.072** | 99.344 base + free segment | no | no |
| AVM allocation | **90.965** | 99.268 | no (decoded mix) | no |
| SVT allocation | 90.438 | 99.688 | no (decoded mix) | no |
| AVM fixed QP124 | **90.192** | 99.344 | yes | no |
| SVT fixed CRF36.5 | 88.641 | 99.376 | yes | no |
| eff-perc allocation | 86.417 | 99.956 | no (decoded mix) | no |
| eff-perc fixed QP119 | 85.180 | 100.920 | yes | no |
| nightly fixed QP106 | 82.155 | 100.032 | yes | no |

### 7.2 The bracket

* **Best known lower bound (achievable): 90.192.** A real AV2 stream at
  99.344 kbps decodes to K 90.192 (90.2 at 100.0). This is the best measured
  rate-feasible point.
* **Strongest measured allocation upper bound: 90.965.** Treating frames
  independently and mixing the AVM ladder gains 0.777 K; the assembled stream
  is not a bitstream.
* **Strongest single-segment upper bound: 94.072.** Infinite bits on AVM's
  f0-32 still leaves 0.928 K to 95.
* **Strongest free-plane bound: 96.144.** Perfect luma crosses 95, but it
  charges no luma rate.

For **rate-feasible, decodable** mechanisms, every measured bound is below 95:
90.192 achieved, 90.965 allocation, 94.072 single-segment. That is the sense in
which the 4:2:0 100 kbps limit is **bracketed below 95** here. The bracket is
empirical, not a proof: a new mechanism could exceed it, but it would have to
beat every measured bound by at least 0.93 K (over the single-segment ceiling)
and 4.04 K (over the allocation ceiling).

If "strongest measured upper bound" means the strongest rate-feasible pooling
bound, it is **94.072** (AVM source f0-32). If it means the strongest
per-frame allocation bound, it is **90.965** (AVM mixed ladder). Both are below
95. The only ~100 kbps bound that clears 95 is the free-luma bound **96.144**.

**Does any bound support 95 at 100 kbps?** Yes, exactly one at that rate: the
free perfect luma plane (96.144 on AVM, 95.798 on SVT). No rate-feasible bound
does.
Therefore the 4:2:0 100 kbps limit **for encodable streams is bracketed below
95**; the free-luma bound identifies luma as the binding channel but is not a
95 claim. The only values above 95 at ~100 kbps are free-plane hybrids. They
are useful diagnostics, not a bracket on encodable streams. A 95 claim cannot
cite them: they set the luma rate to zero and keep the base's chroma bits, so
they are not rate-feasible.

### 7.3 What "95 at 100 kbps" would require

K=95 is a surface, not one triple. With the task's triple (JOD 9.90, BA 0.44,
SS2 92) K is 94.97. The binding component is Butteraugli:

| scenario | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|
| target triple | 9.90 | 0.44 | 92 | 94.97 |
| AVM QP124 today | 9.862 | 1.289 | 90.38 | 90.19 |
| AVM allocation today | 9.863 | 1.135 | 91.10 | 90.97 |
| AVM + source luma | 9.990 | 0.373 | 93.73 | 96.14 |
| perfect JOD/SS2, AVM today's BA | 10.00 | 1.289 | 100 | **92.49 cap** |
| perfect JOD/SS2, allocation BA | 10.00 | 1.135 | 100 | **93.14 cap** |
| perfect JOD/SS2, target BA | 10.00 | 0.44 | 100 | **96.80** |

Even with perfect CVVDP and SSIMULACRA2, AVM's Butteraugli 1.289 caps K at
92.49; the allocation's 1.135 caps it at 93.14. To reach 95 with perfect
JOD/SS2, BA must be at most 0.75. At the target triple, BA must be about 0.44.
AVM must therefore cut Butteraugli by 66% (1.289 → 0.44) at the same rate.

What would have to be true:

1. A **decodable** stream at ≤100.000 kbps with K≥95, or a rate-feasible bound
   at 100 kbps that reaches 95 from a decodable base. A source segment, a
   source plane, or a decoded-frame mix does not qualify.
2. The luma error must be nearly eliminated at that rate. The free-luma rows
   show that luma perfection alone is sufficient to cross 95 on AVM/SVT
   chroma; perfect chroma alone is not (90.67 at AVM).
3. The rate needed to buy that luma must fit inside the 100 kbps budget.
   Conventional AV2 at 188.9 kbps is 92.76; SVT needs 0.6-0.8 Mbps for 95.
   A 95 claim at 100 kbps is a 6-8x rate-efficiency jump, not a tool tuning
   result.
4. The claim needs a bracketing ladder at 100.000 kbps and the decoded stream,
   not an extrapolation from a 188.9 kbps point or a free-plane hybrid.

No measured rate-feasible mechanism satisfies this. The 95 gap is a luma
coding-class gap, not a chroma, entropy or loop-filter gap.

## 8. Reproduce

All commands use `testdata/nasa_480x240_30fps_60f.y4m`, 60 frames, 8-bit
4:2:0, `--cpu-threads 8`. The clean av2enc binary is functionally `cbfc0061`
(only docs commits ahead). SVT and AVM are black-box binaries.

```sh
REF=testdata/nasa_480x240_30fps_60f.y4m

# exact lossless: av2enc (legal, exact) and FFV1 cross-check
av2enc "$REF" -o /tmp/nasa_ll.ivf --video-lossless --frames 60 --cpu-threads 8
ffmpeg -i "$REF" -c:v ffv1 -level 3 -threads 8 -y /tmp/nasa_ffv1.mkv
tools/bin/dav2d-macos -i /tmp/nasa_ll.ivf -o /tmp/nasa_ll.y4m \
  --muxer yuv4mpeg2 -q --strict 1
python3 - <<'PY'
import sys
sys.path.insert(0, "tools")
import quality
a = quality.read_frames("testdata/nasa_480x240_30fps_60f.y4m")[1]
b = quality.read_frames("/tmp/nasa_ll.y4m")[1]
print("exact lossless:", a == b, len(a), "frames")
PY

# pins and 100 kbps ladders
av2enc "$REF" -o /tmp/nightly.ivf --video --frames 60 --cpu-threads 8 \
  --qp 106 --nightly
AV2ENC_EFF_PERC=1 av2enc "$REF" -o /tmp/eff119.ivf --video --frames 60 \
  --cpu-threads 8 --qp 119 --nightly \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
SvtAv1EncApp -i "$REF" -b /tmp/svt.ivf --preset 0 --crf 36.5 \
  --fps-num 30 --fps-denom 1 --progress 0
/path/to/avm_latest/build/avmenc --end-usage=q --cpu-used=6 --threads=8 \
  --limit=60 --qp=124 -o /tmp/avm.ivf "$REF"

# matched-rate oracle (real streams)
python3 tools/oracle.py --ref "$REF" \
  --ivf nightly@106=/tmp/nightly.ivf --target-kbps 100 \
  --json /tmp/nightly.json
python3 tools/oracle.py --ref "$REF" \
  --ivf eff@110=/tmp/e110.ivf --ivf eff@113=/tmp/e113.ivf \
  --ivf eff@116=/tmp/e116.ivf --ivf eff@119=/tmp/e119.ivf \
  --ivf eff@122=/tmp/e122.ivf --ivf eff@125=/tmp/e125.ivf \
  --ivf eff@128=/tmp/e128.ivf --target-kbps 100 --json /tmp/eff.json

# allocation ceiling: encode the 7-point ladder, decode, score per frame,
# solve the multiple-choice knapsack at 25,000 B, assemble, re-score.
# AVM groups: decode every IVF prefix and difference the output frame count.
# The DP recurrence is:
#   dp[g+1][b+c(g,v)] = max(dp[g+1][b+c(g,v)], dp[g][b] + u(g,v))
# with u = 0.5*JOD/10 + 0.25/(1+BA/3) + 0.25*(SS2+50)/150.
# The allocation is a decoded-domain bound; the assembled Y4M is not a stream.

# plane attribution: replace exactly one plane of the decoded base with the
# source, keep the other plane decoded, and re-score with tools/quality.py's
# CVVDP plus tools/extra_metrics.py.
```

The per-frame DP and the AVM prefix mapping are a scratch harness (not
committed), the same status as the audit's per-frame PSNR harness. The
measurement inputs and the assembled Y4M are reproducible from the commands
above; `tools/oracle.py` is the committed oracle for every real stream.

## 9. Verdict

1. **Lossless is not the 100 kbps oracle.** Exact 4:2:0 is 3,631.4 kbps with
   the first-party lossless writer (7,462.7 kbps with FFV1). K=100 there is
   definitional; the rate is 36x the target.
2. **The best decodable point near 100 kbps is AVM 90.192** (99.344 kbps;
   90.188 at 100.000). SVT is 88.641, eff-perc is 85.180, nightly is 82.155.
   The measured pins reproduce exactly.
3. **The best measured allocation ceiling is AVM 90.965** (99.268 kbps),
   then SVT 90.438 and eff-perc 86.417. Allocation buys 0.8-1.7 K, not a class.
4. **The strongest single-segment source ceiling is AVM f0-32 94.072.**
   Infinite bits on 33 frames still leave 0.928 K to 95.
5. **Chroma is not the lever.** A perfect chroma coder buys +0.46 to +0.93 K;
   the audit's 0.61 K share is reproduced (0.681 K in the current eff→SVT
   split). A perfect luma coder buys +5.95 to +7.79 K and crosses 95 on AVM's
   chroma (96.144). That free-plane hybrid is the only measured ~100 kbps
   value above 95.
6. **No rate-feasible measured bound supports 95 at 100 kbps.** For encodable
   streams the limit is bracketed below 95: 90.192 achieved, 90.965 allocation,
   94.072 single segment. To claim 95 one would need a decodable ≤100 kbps
   stream with K≥95, which means closing the luma gap at 100 kbps; no measured
   mechanism does, and the conventional class needs 0.6-0.8 Mbps for 95.

The honest statement is therefore: **95 at 100 kbps is a luma-coding-class
claim, not an AV2 tuning result. The measured 4:2:0 100 kbps limit is below
95 for every rate-feasible mechanism; the only above-95 bound is a free perfect
luma plane, which is not a stream.**
