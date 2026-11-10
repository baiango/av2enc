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

## Exact 640x360

| | |
|---|---|
| Input | `tests/data/m0_640x360_1f.y4m` (no padding) |
| IVF | 19,751 bytes, FourCC AV02 |
| Decoded | 640x360 yuv420p, 1 frame |
| MD5 | `56f11d4a63ff07f2356ef5064b84fe0e` |
| PSNR | Y 46.62 dB, average 47.20 dB |

The bottom partial superblock (360 = 5×64 + 40) uses `PARTITION_NONE` at the
64×64 SB level. The 32×16 / 32×8 implied-partition paths and rectangular edge
transforms are implemented for narrower edge cases. Both 64x64 and 640x360 are
locked by `tests/m0_encode_smoke.sh`.

## Adaptive CDFs (opt-in)

`--adaptive-cdf` enables §8.2.6 adaptation. Default remains frozen CDFs so the
pinned bitstreams do not change.

| Frame | Mode | IVF | MD5 | PSNR avg |
|---|---|---:|---|---:|
| 64x64 | frozen | 1,138 B | `7efc…` | 42.13 dB |
| 64x64 | adaptive | 1,094 B | `7efc…` | 42.13 dB |
| 640x360 | frozen | 19,751 B | `56f1…` | 47.20 dB |
| 640x360 | adaptive | 18,807 B | `56f1…` | 47.20 dB |

Adaptive CDFs save **3.9%** at 64x64 and **4.8%** at 640x360 with identical
reconstruction and `dav2d` decoding.

## CUDA M0 pipeline

The CUDA path is bit-identical to the CPU reference:

| Case | CPU/GPU frame MD5 | `dav2d` MD5 | Recon | CUDA time |
|---|---|---|---|---|
| 64x64 | `08c906ab…` | `7efc3162…` | bit-exact | 1.34 ms |
| 640x360 | `1bbe4c63…` | `56f11d4a…` | bit-exact (PSNR ∞) | 34.1 ms |

640x360 timing breakdown from the shared 3060:

```text
upload    0.049 ms
kernel   25.773 ms
download  0.138 ms
entropy   7.633 ms
total    34.100 ms   (2,740 launches)
```

The CPU reference is 42.0 ms for the same frame, so this is 1.23×. The CUDA path
is **launch-bound**, not compute-bound: 2,740 launches is the problem. Entropy
is still on the CPU; `src/cuda/entropy.*` is ready but not wired in yet.



## AVM comparison, exact 640x360, QP 90

| | AVM | av2enc M0 |
|---|---:|---:|
| One frame, IVF | 10,129 B | 19,751 B |
| PSNR-Y | 48.98 dB | 46.62 dB |
| PSNR average | 49.99 dB | 47.20 dB |
| CVVDP | 9.9556 JOD | not measured |
| Encode time | 2.38 s | < 0.1 s (CPU reference) |

AVM is about **1.95× smaller** and **2.8 dB better** on the same frame. That is
expected: AVM uses full intra prediction, partitions, and RDO; M0 is DC-intra
only with a fixed 32×32 tree.

AVM 2-second inter encode (120 frames, QP 90): 114,419 bytes, 457.7 kbps,
PSNR average 52.86 dB, CVVDP 9.9306, encode 43.36 s (2.77 fps, 0.046× realtime
on M4), decode 0.083 s (1,439 fps, 24× realtime). Our encoder is still-picture
only, so this is the video target, not a direct comparison.

The earlier 640x384 padded proxy is obsolete; exact 640x360 is supported.
