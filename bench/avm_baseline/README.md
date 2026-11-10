# AVM baseline

Reference encoder baseline for the 720p test clip.

- Source: `testdata/shine_1280x720_60f.y4m` (1280x720, 60 frames, 60 fps, 1 s)
- Encoder: AVM macOS arm64 nightly 2026-11-10, commit `cd7d8bc3db`
- Settings: `--end-usage=q --cpu-used=6 --threads=8`, fixed QP
- Decoder: dav2d `f9dafbc` (every stream decodes)
- Metrics: `tools/quality.py`, CVVDP 0.5.7, `standard_fhd`, MPS
- Host: **Apple M4, 10 cores, 16 GiB, macOS 26.6.2** (Mac-mini.local)
- Vast.ai box: **2× Intel Xeon E5-2690 v4 (28C/56T), 125 GiB, RTX 3060 12 GB**. Do not compare CSS across the two machines; rerun AVM on the box for a like-for-like speed ratio. See `docs/hardware.md`.

| QP | IVF bytes | kbps | PSNR-Y | PSNR avg | SSIM All | CVVDP (JOD) | Encode time | realtime× |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 110 | 152,787 | 1222.3 | 51.116 | 51.976 | 0.99759 | 9.9160 | 65.4 s | 0.0153 |
| 135 | 126,782 | 1014.3 | 49.990 | 50.912 | 0.99716 | 9.8641 | 54.5 s | 0.0183 |
| 160 | 93,050 | 744.4 | 48.616 | 49.619 | 0.99636 | 9.8266 | 49.3 s | 0.0203 |
| 185 | 62,315 | 498.5 | 46.545 | 47.654 | 0.99460 | 9.7221 | 42.3 s | 0.0236 |
| 200 | 46,080 | 368.6 | 44.198 | 45.404 | 0.99119 | 9.3623 | 34.4 s | 0.0291 |
| 210 | 33,543 | 268.3 | 41.649 | 42.897 | 0.98484 | 8.6225 | 37.4 s | 0.0268 |
| 235 | 3,273 | 26.2 | 31.674 | 33.071 | 0.93018 | 5.8911 | 27.1 s | 0.0369 |

Notes:

- AVM's internal `--psnr` summary for QP 160 is 49.619 dB, matching
  `tools/quality.py` after frame-rate normalization.
- The source is a 1-second clip from the user's 4K VP9 file; it is a relatively
  easy, smooth TV-demo shot. Do not generalize these numbers to hard content.
- CVVDP is computed over all 60 frames with `standard_fhd`. The 720p frame is
  treated as the central portion of an FHD display; keep this setting fixed for
  all future comparisons.

## Speed vs CVVDP

At iso-CVVDP, interpolated from the six operating points:

| CVVDP (JOD) | AVM speed | AVM bitrate |
|---:|---:|---:|
| 8.0 | 0.0291× realtime | 213 kbps |
| 8.5 | 0.0272× realtime | 258 kbps |
| 9.0 | 0.0279× realtime | 320 kbps |
| 9.5 | 0.0270× realtime | 418 kbps |

**CSS-AUC over CVVDP [8.0, 9.5] = 0.0279× realtime** for AVM.

The curve extends down to 5.89 JOD (QP 235), but the 8.0–9.5 range is the useful
comparison window for a first-generation encoder. If our encoder cannot reach
9.5 JOD, the common range shrinks and the score is reported only there.

```sh
python3 tools/score_speed.py --reference AVM \
  --targets 8.0,8.5,9.0,9.5 --range 8.0 9.5 \
  bench/avm_baseline/qp*.json \
  --json bench/avm_baseline/speed_score.json \
  --plot bench/avm_baseline/speed_score.png
```

`CSS(C)` is the encode speed at a target CVVDP; `CSS-AUC` is the mean speed over
the common CVVDP range. Higher is better, but read it with the iso-quality
bitrate.

## QBS

Using `CVVDP >= 9.0 JOD`, `speed >= 0.02×`:

| Bitrate ceiling | Feasible | QBS | Point |
|---|---:|---:|---|
| 500 kbps | yes | **9.7221 JOD** | QP 185, 498.5 kbps, 0.0236× |
| 400 kbps | yes | **9.3623 JOD** | QP 200, 368.6 kbps, 0.0291× |

At the 500 kbps budget, the minimum-rate feasible point is QP 200
(368.6 kbps, 9.3623 JOD), and the fastest feasible point is also QP 200
(0.0291×). At 400 kbps, QP 200 is the QBS point.

This is the bar for our encoder: **at 720p60, reach at least 9.36 JOD at
≤400 kbps, with speed measured against AVM on the same machine.** Tighten from
there.

```sh
python3 tools/score_tradeoff.py --reference AVM \
  --q-min 9.0 --r-max 400 --s-min 0.02 \
  bench/avm_baseline/qp*.json \
  --json bench/avm_baseline/qbs_400kbps.json
```

## Reproduce

```sh
tools/bin/avm-macos/avmenc testdata/shine_1280x720_60f.y4m \
  -o bench/avm_baseline/qp160_720p_60f.ivf \
  --ivf --end-usage=q --qp=160 --cpu-used=6 --threads=8 --psnr=1

python3 tools/quality.py \
  --ref testdata/shine_1280x720_60f.y4m \
  --ivf bench/avm_baseline/qp160_720p_60f.ivf \
  --json bench/avm_baseline/qp160_720p_60f.json
```

## Comparing our encoder

Run the same `tools/quality.py` command on our IVF at a similar bitrate. The
comparison is meaningful only if:

1. `dav2d` decodes our stream.
2. Bitrates are within ~5–10%.
3. The same source clip, frame count, display model, and CVVDP version are used.
