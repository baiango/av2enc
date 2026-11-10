# AVM 360p baseline

Exact **640x360**, no padding. Source: 2 seconds (120 frames, 60 fps) from the
user's 4K VP9 clip at 10 s.

Encoder: AVM macOS arm64 nightly `cd7d8bc3db`, `--end-usage=q --qp=90
--cpu-used=6 --threads=8`.
Decoder: `dav2d` f9dafbc. Metrics: `tools/quality.py`, CVVDP 0.5.7
(`standard_fhd`, MPS on Apple M4).

| Case | IVF | Bitrate | PSNR-Y | PSNR avg | SSIM All | CVVDP | Encode |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 frame | 10,129 B | 4,861.9 kbps* | 48.98 dB | 49.99 dB | 0.99719 | 9.9556 | 2.38 s |
| 2 s / 120 frames | 114,419 B | 457.7 kbps | 52.05 dB | 52.86 dB | 0.99776 | 9.9306 | 43.36 s |

\*One-frame bitrate is normalized over 1/60 s, so it is not comparable to the
2-second bitrate.

Decode speed for the 2-second stream on the M4:

```text
decode_seconds:     0.0834 s
decode_fps:         1439.3 fps
decode_realtime_x: 24.0x
```

Notes:

- AVM's 2-second inter encode is the video baseline. Our M0 encoder is
  still-picture only.
- Our current 640x384 padded 360p still was 19,772 bytes / 47.48 dB; it is not
  an apples-to-apples comparison because of the pad and the lack of inter
  prediction.
- Exact 640x360 support in `av2enc` is in progress; the current stream is
  rejected by `dav2d` due to the bottom partial superblock partition.
