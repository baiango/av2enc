# SVT-AV1 reference point

Cross-encoder calibration on the NASA 480x240/30 60-frame clip, not an AV2
parity claim. Encoder: **SVT-AV1 v4.1.0 (release)**, macOS arm64.

## Command

```sh
SvtAv1EncApp \
  -i testdata/nasa_480x240_30fps_60f.y4m \
  -b /tmp/svt_crf42.ivf \
  --preset 4 --crf 42 \
  --fps-num 30 --fps-denom 1 --progress 0

dav1d -q -i /tmp/svt_crf42.ivf -o /tmp/svt_crf42.y4m
DAV2D=tools/bin/dav2d-macos tools/quality.py \
  --ref testdata/nasa_480x240_30fps_60f.y4m \
  --test /tmp/svt_crf42.y4m --json /tmp/svt_crf42.json
```

## Result

| Stream | IVF bytes | kbps | CVVDP | PSNR-Y | Encode |
|---|---:|---:|---:|---:|---:|
| SVT-AV1 CRF 42, preset 4 | 25,039 | 100.16 | 9.7560 JOD | 45.9248 dB | 306 fps (0.196 s/60 f) |
| **SVT-AV1 CRF 36.5, preset 0** | **25,064** | **100.26** | **9.8304 JOD** | **48.8001 dB** | **16.6 fps (3.61 s/60 f)** |

Preset-4 CRF sweep: 40 = 106.2 kbps, 41 = 101.7, **42 = 100.16**, 43 = 95.7.
Preset-0 sweep: 35 = 106.0, 36 = 101.7, **36.5 = 100.26**, 37 = 97.7.

Preset-0 low-rate points (same clip, decoded and scored identically):

| Preset-0 CRF | kbps | CVVDP | PSNR-Y |
|---:|---:|---:|---:|
| 42 | 83.10 | 9.8023 | 47.5412 |
| 45 | 76.46 | 9.7757 | 46.7260 |
| 50 | 65.19 | 9.7171 | 45.4659 |
| 55 | 54.22 | 9.5849 | 43.7657 |
`dav1d` decoded all streams; the same `tools/quality.py` CVVDP 0.5.7 /
`standard_fhd` / MPS model used for AV2 scored the decoded Y4M.

At the same 100 kbps point, the current `--nightly` AV2 stream is 9.5868 JOD /
41.6953 dB. SVT-AV1 preset 4 is +0.169 JOD / +4.22 dB; preset 0 is **+0.244 JOD
/ +7.10 dB** over nightly. Speed is not comparable: preset 4 is 306 fps,
preset 0 is 16.6 fps, nightly AV2 is 7.6 fps on the M4.

At **iso-CVVDP 9.585**, preset 0 reaches the same perceptual quality at
**54.22 kbps** versus nightly's 99.756 kbps. The AV2 stream therefore uses
**~1.84x the rate** (84% more) at equal CVVDP; the 100 kbps JOD gap is small
only because CVVDP is near saturation.
