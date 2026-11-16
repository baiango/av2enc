# QP-scaled RD lambdas

Status: **measured positive at both operating points.** The five rate/distortion
weights that used one constant calibrated at a single QP now follow per-site
power laws fitted at the 240p/100 kbps and 720p/300 kbps points. The aggregate
is **+0.012 JOD / +0.69 dB at 240p** and **+0.046 JOD / −0.065 dB at 720p**,
both at iso-rate, with 60/60 pictures decoded and `--video-recon` equal to the
decoded payload.

The mode is opt-in (`--video-qp-lambda`) and CPU-only. With the flag off every
default, `quality` and `--nightly` stream is byte-identical
(25,009 `a1d0dba1`, 25,141 `3984677b`, 24,113 `7f03eb6e`).

## The defect

| site | legacy constant | QP dependence |
|---|---|---|
| `--video-leaf-rd-lambda` | 125 anchored at QP104 | `sqrt(step/98)` only (exponent 0.5) |
| `--video-tx-lambda` | 136 (nightly) | none |
| `--video-key-tx-lambda` | 0 = inherit 136 | none |
| `--video-rdoq-scale` | 0.1 at QP110 | `step^2` on the coefficient lambda |
| partition fallback price | 700 SSE/bit at QP190 | none |
| partition probe scale | 2.75e-6 at QP110 | `step^2` (lambda) |

The defect is structural, not a bad constant: the same QP-invariant number
prices a decision at QP 110 and at QP 217, where the reconstruction step grows
21.7x. `lambda = c * step^2` is the SSE-domain relation, but it is not what the
*measured optimum* follows on any of these sites — only the probe scale is close
to Q², and even there the fitted exponent is 2.22. The measurements below fix
the exponent per site instead of assuming one.

## Operating points and steps

`step` is the luma reconstruction step `qlookup(q)/8`; the partition probe uses
the full `qlookup(q)` value (both are stated where used).

| point | clip | QP | luma step | key base q | pin |
|---|---|---:|---:|---:|---|
| 240p | `nasa_480x240_30fps_60f.y4m`, 60 f | 110 | 118 | 83 (`--video-key-qp-offset -27`) | 24,113 B / 96.452 kbps / 9.7130 / 43.8901 |
| 720p | `space8k_1440x720_60f.y4m`, 60 f | 217 | 2560 | 160 (`--video-key-qp-offset -57`) | 37,003 B / 296.024 kbps / 7.2726 / 31.3342 |

Build: `cmake -S . -B build-t -DAV2ENC_ENABLE_CUDA=OFF
-DAV2ENC_ALLOW_CPU_ONLY=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-t
-j1`. Encodes: CPU `--video`, `--cpu-threads 8`, single tile, 8-bit 4:2:0.
Quality: `tools/quality.py` (CVVDP 0.5.7, `standard_fhd`, MPS), serial. Decode:
`dav2d-macos --strict 1`, picture count, payload equality against
`--video-recon`.

Iso-rate is the CVVDP interpolated on the arm's own adjacent-QP curve to
96.452 kbps (240p) or 296.024 kbps (720p). QP 109/108 and 215/214 brackets are
quoted in the tables; no arm is compared at a rate more than ~2% away without
interpolation.

## Exponent scan: 0.5, 1.0, 2.0

For each site the table gives the lambda implied by the three test exponents at
the two points (normalised at the legacy anchor), the measured optimum, and the
measured RD outcome. `lambda(QP)` is the effective SSE-per-bit weight (for the
RDOQ and probe rows, the *coefficient/probe lambda* after multiplying the scale
by `step^2`).

### leaf (`--video-leaf-rd-lambda`), anchor 125 at step 98

| exponent | lambda @110 | lambda @217 | measured |
|---:|---:|---:|---|
| 0.5 (legacy) | 137 | 639 | pin; leaves **+0.043 JOD** at 720p |
| 1.0 | 151 | 3,266 | 266.1 kbps / 7.1188 at fixed QP; iso-rate ≈ **+0.00** (c=600 ladder +0.002) |
| 2.0 | 181 | 85,287 | 248.4 kbps / **4.8423 JOD** at fixed QP — collapse |
| **0.72 (fit)** | **143** | **1,310** | 240p +0.002, 720p **+0.043** |

Measured optimum: 137–150 at 240p (flat), ~1,000–1,500 at 720p. No test
exponent is within 5% at both (0.5 is 2.0x low at 720p, 1.0 is 2.5x high,
2.0 is 65x high); 0.72 is within 5% at both.

### TX search (`--video-tx-lambda`), anchor 136 at step 2560

| exponent | lambda @110 | lambda @217 | measured |
|---:|---:|---:|---|
| 0.5 | 29.2 | 136 | 240p 96.492 / 9.7089 (−0.004); 720p = pin |
| 1.0 | 6.27 | 136 | 240p 97.352 / 9.7170 (≈0 at iso-rate); 720p = pin |
| 2.0 | 0.29 | 136 | not run: the weight is below the rate-free limit; 720p = pin |
| **0.3385 (fit)** | **48.0** | **136.0** | 240p +0.000, 720p +0.000 |

Measured optimum: 48 at 240p, 136 at 217. Because the anchor sits at the 720p
point, every test exponent is the pin at 720p; the 240p differences are all
inside ±0.005 JOD. The fitted form is neutral at both and is the only one that
also removes the 2.8x relative drift.

### key TX (`--video-key-tx-lambda`), keys at QP 93 / 160

Single-keyframe sweeps (`--frames 1`, key points QP 93 and 160):

| lambda | 240p key JOD / PSNR-Y | 720p key JOD / PSNR-Y |
|---:|---|---|
| 4 | 9.9156 / 45.70 | 8.4453 / 34.91 |
| **8** | **9.9148 / 45.62** | **8.4473 / 34.92** |
| 16 | 9.9117 / 45.50 | 8.4414 / 34.92 |
| 32 | 9.9113 / 45.33 | 8.4319 / 34.89 |
| 136 | 9.8837 / 44.90 | 8.3987 / 34.81 |

Exponent 0 covers both (the optimum is 4–8 at both QPs). Exponent 0.5 would put
63 at the 720p key (between the measured λ32 and λ136 rows, both negative), 1.0
would put 498, 2.0 would put ~31,000. The fitted constant is 8.

### RDOQ (`--video-rdoq-scale`), scale s in `lambda = s * quant_step^2`

| coefficient-lambda exponent | s @110 | s @217 | measured |
|---:|---:|---:|---|
| 0.5 | 0.1 | 0.0010 | near trellis-off: −0.007 at 720p |
| 1.0 | 0.1 | 0.0046 | measured s=0.005: −0.007 at 720p |
| 2.0 (legacy) | 0.1 | 0.100 | pin (0/0) |
| **1.5 (fit)** | **0.1** | **0.0215** | 240p 0.000, 720p **+0.006** |

Exponents below 1.5 make the trellis inert at high QP; 2.0 leaves the leaf
guard and the trellis 475x out of step. The fitted scale is exactly the pin at
the 240p anchor and the measured 0.02 at 720p.

### Partition

Probe lambda = `scale * step^2` (full `qlookup` step 944 / 20480):

| probe-lambda exponent | lambda @110 | lambda @217 | measured |
|---:|---:|---:|---|
| 0.5 | 2.45 | 0.011 | far below the rate-free limit |
| 1.0 | 2.45 | 0.053 | 1.375e-6 at 240p (λ 1.22) is −0.012 |
| 2.0 (legacy) | 2.45 | 1.15 | part-only +0.007 at 720p |
| **2.22 (fit)** | **2.45** | **2.27** | 240p 0.000, 720p +0.007 (probe alone +0.021) |

The probe lambda is nearly QP-flat (2.45 → 2.27), so the *scale* must rise with
QP: 2.75e-6 → 5.41e-6. The 1.375e-6 arm at 240p measures −0.012 at iso-rate,
which is the empirical cost of the low exponents.

The fallback price (variance + syntax bits) has no independent flag. Measured
with the composed set:

| fallback exponent | price @110 | price @217 | composed 240p | composed 720p |
|---:|---:|---:|---:|---:|
| 2.0 (step²) | 7.34 | 3,457 | +0.007 | +0.032 |
| 1.10 | 7.34 | 217 | +0.012 | +0.043 |
| **1.44 (fit)** | **7.34** | **617** | **+0.012** | **+0.046** |

The 240p optimum wants a far cheaper fallback price (fine keyframe leaves) and
the 720p optimum wants ~600, not step² (~3,500); the two-point fit is 1.44. The
site stays off (700 at every QP) when `--video-partition-rd-lambda` is given
explicitly, which is the control.

## Shipped fits

`src/pipeline/video/qp_lambda.h`:

| site | form | @240p | @720p |
|---|---|---:|---:|
| leaf | `125 * (step/98)^0.72` | 143 | 1,310 |
| TX | `136 * (step/2560)^0.3385` | 48.0 | 136.0 |
| key TX | `8` | 8 | 8 |
| RDOQ scale | `0.1 * (step/118)^-0.5` | 0.100 | 0.0215 |
| probe scale | `2.75e-6 * (step/944)^0.22` | 2.75e-6 | 5.41e-6 |
| partition fallback | `7.344 * (step/944)^1.44` | 7.34 | 617 |

An explicitly supplied numeric flag keeps the legacy form (the leaf anchor, or
the exact constant for the other sites) and is the A/B control. `--nightly`
sets `--video-tx-lambda 136` and the probe scale internally only when the user
did not; under `--video-qp-lambda` those are treated as anchors and the fits
above apply.

## Matrix

ΔJOD (and ΔPSNR-Y) at iso-rate, one clip per point. Every arm is CPU 8-bit
4:2:0, one tile, 60 frames.

| site | scaling | 240p ΔJOD | 240p ΔPSNR-Y | 720p ΔJOD | 720p ΔPSNR-Y |
|---|---|---:|---:|---:|---:|
| leaf | `step^0.72` | +0.002 | +0.06 | **+0.043** | +0.06 |
| TX | `step^0.3385` | +0.000 | +0.00 | +0.000 | +0.00 |
| key TX | constant 8 | **+0.011** | +0.40 | −0.006 | −0.03 |
| RDOQ | coefficient `step^1.5` | +0.000 | +0.00 | +0.006 | +0.01 |
| partition | probe `step^2.22`, fallback `step^1.44` | +0.012 | +0.55 | +0.007 | −0.06 |
| **composed** | all five | **+0.012** | **+0.69** | **+0.046** | **−0.065** |

The key row is the only negative one at 720p (−0.006, inside the documented
±0.01–0.02 QP ripple); λ16 was measured at 720p (−0.039) and λ4 loses the
keyframe single-picture test, so 8 is the two-point optimum. The composed set
is positive at both, so no point is traded for the other.

## Composed result

```sh
# 240p, --qp 110
build-t/av2enc testdata/nasa_480x240_30fps_60f.y4m -o n.ivf --video \
  --frames 60 --cpu-threads 8 --qp 110 --nightly --video-qp-lambda
# 240p QP ladder: 109 -> 95.988 kbps / 9.7270 / 44.5714
#                  108 -> 97.164 kbps / 9.7216 / 44.5985
# at 96.452 kbps:                9.7249 / 44.5821   (+0.0119 / +0.692)
# pin:                           9.7130 / 43.8901

# 720p, --qp 217, key offset -57
build-t/av2enc testdata/space8k_1440x720_60f.y4m -o h.ivf --video \
  --frames 60 --cpu-threads 8 --qp 217 --nightly \
  --video-key-qp-offset -57 --video-qp-lambda
# 720p QP ladder: 215 -> 292.944 kbps / 7.3120 / 31.2555
#                  214 -> 302.488 kbps / 7.3313 / 31.2996
# at 296.024 kbps:                7.3182 / 31.2697   (+0.0456 / -0.0645)
# pin:                            7.2726 / 31.3342
```

Pinned scaled streams at the fixed-QP points:

| stream | bytes | MD5 | decoded | recon |
|---|---:|---|---|---|
| 240p QP110 composed | 23,690 | `1b99127a4cd1d386fa5600fb63c57a83` | 60/60 | exact |
| 720p QP217 composed | 35,646 | `41bc1182499fcc7b308bb528f8eb2d40` | 60/60 | exact |

At the fixed QP the composed set is 3.7% (720p) / 1.8% (240p) smaller; spending
that rate on QP is where the iso-rate gain comes from. The 240p PSNR gain is
large (+0.69 dB) while its CVVDP move is small: the saving comes mostly from the
keyframe/fallback tree, which buys average fidelity without moving the pooled
worst-region metric. At 720p CVVDP moves +0.046 while PSNR-Y gives back 0.065
dB — a worst-region redistribution, which is the direction CVVDP rewards (the
same shape as the refresh-budget and leaf-lambda results).

## Verification

* `src/pipeline/video/gate_video_qp_lambda.sh --bin build-t/av2enc` — OK:
  three pins byte-identical, the all-explicit control reproduces the nightly
  pin, each site is live (and the RDOQ fit is exactly 0.1 at 240p), both scaled
  streams hit their pins, decode 60/60 under `dav2d --strict 1`, and the decoded
  payload equals `--video-recon`.
* `tools/gate_video_nightly.sh --bin build-t/av2enc` — OK (pins, expansion,
  CDEF A/B, strict decode, visible recon, BAWP).
* Rejections: `--video-qp-lambda` with `--video-gpu`, `--rc crf`/`vbr`, and on
  still/m1 carriers is a named usage error.
* The flag-off pins are re-checked after the final build: 25,009
  `a1d0dba1...`, 25,141 `3984677b...`, 24,113 `7f03eb6e...`.

## Limits

* One clip per point and one host. The 240p CVVDP gain (+0.012) is under one QP
  step of ripple; the +0.69 dB PSNR and the 2.9% rate saving are the stronger
  signals there. The 720p gain (+0.046) is above the ripple.
* The 720p content has a pulldown cadence and sits at the quantiser ceiling;
  the key site's −0.006 there is within the local QP scatter.
* The fits are measured on the `--nightly` aggregate. They are gated to the
  same surface (CPU `--video`, cqp, 8-bit 4:2:0, single tile); other tools can
  change which site binds, and the flag stays opt-in until it is re-measured on
  them.
* The partition fallback exponent (1.44) is a two-point fit: 240p measures the
  fallback directly, 720p constrains it through the composed set and the
  probe-only arm, so the 617 value is the least-constrained number here.
