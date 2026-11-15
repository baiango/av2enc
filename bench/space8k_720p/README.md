# 720p60 @ 300 kbps: native 8K content vs a smooth upscale

Cross-encoder calibration on real detail, because the 240p clip's conclusion does
not transfer: the *smooth* 1440x720 test clip flatters av2enc by ~1.4 JOD.

## Source and clip

`First-8K-Video-from-Space_orig.mp4`: 8192x4096 HEVC, 23.976 fps, 189 s,
138 Mbps, full-range `yuvj420p`. One second extracted from t=20 s:

```sh
ffmpeg -ss 20 -i First-8K-Video-from-Space_orig.mp4 \
  -vf "scale=1440:720:flags=lanczos,fps=60,format=yuv420p" -frames:v 60 \
  -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -f yuv4mpegpipe testdata/space8k_1440x720_60f.y4m
```

The 23.976 -> 60 conversion duplicates frames, so the clip carries a pulldown
cadence: both encoders code ~24 distinct pictures per second. That is the same
for both arms, so the comparison is fair, but it is not native 60 fps content.

## av2enc

```sh
build/av2enc testdata/space8k_1440x720_60f.y4m -o /tmp/x.ivf \
  --video --frames 60 --cpu-threads 8 --qp <Q> --nightly
```

## SVT-AV1 preset 0

```sh
SvtAv1EncApp -i testdata/space8k_1440x720_60f.y4m -b /tmp/s.ivf \
  --preset 0 --crf <C> --fps-num 60 --fps-denom 1 --progress 0
dav1d -q -i /tmp/s.ivf -o /tmp/s.y4m
```

## Measured

Both decoded with the same tools and scored with `tools/quality.py`
(CVVDP 0.5.7, `standard_fhd`, MPS).

| Encoder | Setting | kbps | CVVDP | PSNR-Y |
|---|---|---:|---:|---:|
| av2enc | QP180 | 467.5 | — | — |
| av2enc | QP190 | 385.7 | — | — |
| av2enc | QP200 | 320.3 | 7.1242 | 31.42 |
| **av2enc** | **QP202** | **308.9** | **7.0490** | **31.26** |
| SVT p0 | CRF60 | 518.7 | — | — |
| SVT p0 | CRF62 | 377.3 | 8.5520 | 35.76 |
| **SVT p0** | **CRF63** | **288.9** | **8.2301** | **34.62** |

At 300 kbps (linear in rate): **av2enc 7.01, SVT 8.27 -> -1.26 JOD, -3.3 dB.**

## Why this matters

The same measurement on the smooth `nasa_1440x720_60f` upscale gives the
opposite answer (+0.16 JOD for av2enc). The difference is content detail, and it
is visible in the bit split:

| clip | key share | P share | our PSNR |
|---|---:|---:|---:|
| smooth upscale, QP114 | 93.9% | 5.9% | 47.95 dB |
| native detail, QP202 | 29.6% | 69.3% | 31.26 dB |

On the upscale the budget is a keyframe; on real detail it is a P chain, and the
P chain is where we lose ~3.3 dB at the same rate. The 240p result (-0.11 JOD)
understates this because that clip is also low-detail.

Retunes measured and rejected at QP202 (iso-rate 300 kbps in brackets):
`--video-leaf-rd-lambda 60` (7.03), `40` (7.01), `--video-key-qp-offset -10`
(6.51 at 282.8 kbps). Neither the leaf RD price nor the key allocation explains
the gap.

## Practical reading

- Quote 240p and 720p numbers with the content described. A smooth test clip can
  flip the sign of the cross-encoder result.
- SVT at CRF63 is at the edge of its CRF range on this content; av2enc at QP202
  is likewise near its quantiser ceiling. Both encoders are in their extreme
  low-rate regime, so this is a stress point, not a typical one.
- The remaining deficit is inter coding on detailed content, not allocation,
  entropy, framing or the leaf lambda.

## Key offset at 720p: -57, not -27

The 240p reference-intro offset (-27) is badly under-funded at 720p. Measured
at QP217 vs the -27/QP202 baseline, both ~300 kbps:

| key offset | QP | kbps | CVVDP | PSNR-Y |
|---|---:|---:|---:|---:|
| -27 (nightly default) | 202 | 308.9 | 7.0490 | 31.26 |
| **-57** | **217** | **296.0** | **7.2726** | 31.33 |
| -51 | 217 | 276.5 | 7.1242 | 31.14 |

**+0.22 JOD at 3% lower rate.** The mechanism is measured: an oracle test that
makes one decoded frame perfect is worth **+0.356 JOD at 720p but only +0.004 at
240p**. The P chain is skip-dominated (10 QP of P-frame QP moves a frame by
76 B), so it cannot absorb budget, and the keyframe's error is carried forward by
skips into the sustained component CVVDP weights. Fund the key directly.

At 240p the opposite holds: the keyframe is saturated flat in JOD above ~32 kB,
and offsets -21..-45 are flat at iso-rate, so -27 stays correct there.

After this the gap to SVT preset 0 at 300 kbps is ~1.05 JOD (7.27 vs 8.27).
