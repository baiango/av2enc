# M0 encoder results

Pinned decoder: `dav2d` f9dafbc. Gate: `tools/validate_dav2d.sh`.

## 64x64 regression frame

| | |
|---|---|
| Input | `tests/data/m0_64x64_1f.y4m` |
| IVF | 1,138 bytes, FourCC AV02 |
| Decoded | 64x64 yuv420p, 1 frame |
| MD5 | `7efc3162bc0a4785685eba291a8096fd` |
| PSNR | Y 41.72 dB, average 42.13 dB |

Locked by `tests/m0_encode_smoke.sh`.

## 360p source frame

The M0 encoder currently requires dimensions that are multiples of 64. Exact
640x360 is rejected:

```text
av2enc: error: M0 requires dimensions that are multiples of 64, got 640x360
```

Nearest valid size is 640x384 (640x360 padded with 24 black rows).

| Frame | IVF | PSNR Y | PSNR avg | Notes |
|---|---:|---:|---:|---|
| First frame at 0 s | 245 bytes | 70.87 dB | 72.63 dB | black/title frame; trivial |
| Content at 10 s | 19,772 bytes | 46.90 dB | 47.48 dB | real content, padded to 384 |

Both decoded with `dav2d` and passed the gate.

Exact 360p needs partial-superblock handling (or a render-size crop) in the M0
block layer. Until then, 640x384 is the supported proxy.
