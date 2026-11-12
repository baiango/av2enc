# M1 intra RDO experiment

This directory is an **experiment harness**, not encoder code. It measures the
difference between the current SSE-only M1 luma intra decision and the
rate-aware decision in `src/encoder/rdo/intra_decision.*`.

It never edits `src/encoder/m1/still_encoder.cpp`. `patch_still_intra_rdo.py` reads that
file and writes a patched copy under `build-rdo-exp/`; the patch is active only
when `AV2ENC_RDO_INTRA` is set. With the variable unset, the patched binary must
be byte-identical to the unpatched CLI (the runner checks this).

## Build and run

```sh
cd src/encoder/rdo/experiment
./run_still_intra_rdo.sh
./run_still_intra_rdo.sh --scales "0.02,0.06,0.125"
./run_still_intra_rdo.sh --input /tmp/testsrc2_64x64.y4m
```

The defaults use the repository fixtures:

- `testdata/basic_64x64_1f.y4m`
- `tests/data/basic_640x360_1f.y4m`

The runner builds `build-rdo-exp/av2enc-rdo` and `build-rdo-exp/still_dump_rdo`,
encodes three cases per input, and calls `tools/quality.py`:

| Case | Command | Meaning |
|---|---|---|
| `dc` | `--still` | Current default: DC luma |
| `sse` | `--still --still-full-intra` | Current full search, SSE-only |
| `rdo` | `--still --still-full-intra` + `AV2ENC_RDO_INTRA=1` | Rate-aware selection |

For each `rdo` stream it also runs `tools/still_verify/verify.py` against an
encoder-side dump produced by the same patched code. Every run below was
bit-exact.

## Results (1 frame, QP 90, default scale 0.125)

Repository fixtures:

| Input | Case | Bytes | PSNR-Y | PSNR-avg | CVVDP |
|---|---|---:|---:|---:|---:|
| 64x64 | DC | 936 | 42.5016 | 43.2809 | 9.9009 |
| 64x64 | SSE-only full | 979 | 42.5048 | 43.2627 | 9.9035 |
| 64x64 | RDO full | 965 | 42.5158 | 43.2684 | 9.9144 |
| 640x360 | DC | 13,355 | 45.0892 | 46.3683 | 9.8867 |
| 640x360 | SSE-only full | 13,199 | 45.0227 | 46.3016 | 9.8843 |
| 640x360 | RDO full | 13,152 | 44.9578 | 46.2460 | 9.8830 |

At a lower scale (`0.02`), the 640x360 fixture reaches 13,149 bytes and
CVVDP 9.8874, slightly above DC's 9.8867, but PSNR-Y is 0.0466 dB lower.
At `0.005` it reaches 13,107 bytes, CVVDP 9.8883 and PSNR-Y 45.0729
(-0.0163 dB vs DC). At `0.06`, the balanced point on these two fixtures,
64x64 is 967 bytes / CVVDP 9.9103 and 640x360 is 13,163 bytes /
CVVDP 9.8864 (DC: 936 B / 9.9009 and 13,355 B / 9.8867). The discrete mode
switches make the curve non-monotonic; this is why the scale must be calibrated
on the target corpus rather than treated as a constant.

`testsrc2` (the input used by `tools/still_verify/run_matrix.sh`):

| Input | Case | Bytes | PSNR-Y | PSNR-avg | CVVDP |
|---|---|---:|---:|---:|---:|
| 64x64 | DC | 2,083 | 42.1929 | 41.8239 | 9.9744 |
| 64x64 | SSE-only full | 2,041 | 42.3789 | 41.9451 | 9.9801 |
| 64x64 | RDO full @0.125 | 2,024 | 42.2802 | 41.8874 | 9.9796 |
| 640x360 | DC | 17,437 | 43.1470 | 44.2421 | 9.9018 |
| 640x360 | SSE-only full | 17,397 | 42.8073 | 43.9273 | 9.8906 |
| 640x360 | RDO full @0.02 | 16,998 | 42.7085 | 43.8624 | 9.8951 |

`testsrc2` rows were generated exactly as in `tools/still_verify/run_matrix.sh`:

```sh
ffmpeg -y -v error -f lavfi -i "testsrc2=size=64x64:rate=1:duration=1" \
  -frames:v 1 -pix_fmt yuv420p /tmp/src64_testsrc2.y4m
ffmpeg -y -v error -f lavfi -i "testsrc2=size=640x360:rate=1:duration=1" \
  -frames:v 1 -pix_fmt yuv420p /tmp/src360_testsrc2.y4m
```

## Interpretation

- Rate-aware selection consistently improves over **SSE-only full intra**:
  fewer bytes at 64x64 and 640x360 on both content sets.
- It does **not** make full intra universally beat DC. The result is
  content-dependent:
  - `testsrc2` 64x64: RDO full beats DC on bytes, PSNR-Y and CVVDP
    (-59 B, +0.0873 dB, +0.0052 JOD at scale 0.125).
  - repository 64x64 fixture: DC remains smaller (936 vs 965 B); RDO buys
    +0.0135 JOD for +29 B.
  - 640x360: RDO full saves 200+ bytes versus DC, with a small CVVDP/PSNR
    tradeoff. At scale 0.02, CVVDP is marginally above DC but PSNR-Y is lower.
- The decision is sensitive to the lambda scale. There is no single scale that
  wins every corpus; calibrate on the target content before changing the
  default.

## Why this is not a final quality claim

The coefficient-rate term uses the documented RDOQ bring-up model over the
candidate's real quantised levels (`intra_coefficient_rate_bits_from_levels`).
The mode/angle syntax term is also a placeholder. Both are pluggable through
`IntraRateEstimator`; the entropy owner must supply exact AV2 symbol costs
before these numbers become a BD-rate claim. The streams decode and reconstruct
bit-exactly, but the rate model is provisional.

This is a CPU-only, one-frame, correctness/experiment result. It is not a
performance measurement and no production bitstream or default was changed.
