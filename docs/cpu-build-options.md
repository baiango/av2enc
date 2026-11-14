# CPU build options (`AV2ENC_NATIVE`, `AV2ENC_LTO`, `AV2ENC_CPU_FAST`)

The CPU encoder is the **correctness oracle**, not the product target, so the
default build stays portable: `-O3`, no host tuning, byte-identical streams on
every compiler/host. These three CMake cache options only change **code
generation** for the host C++ targets and are all **OFF by default**.

| Option | Default | Effect |
|---|---|---|
| `AV2ENC_NATIVE` | `OFF` | `-mcpu=native` on Apple clang/arm64, `-march=native` otherwise (Clang/GCC). The binary is then host-specific. |
| `AV2ENC_LTO` | `OFF` | Full `-flto` on `av2enc_core`, the CLI and the dump/gate harnesses, plus the matching C++ link flag. Ignored with a warning on CUDA builds. |
| `AV2ENC_CPU_FAST` | `OFF` | Aggregate: implies `AV2ENC_NATIVE` + `AV2ENC_LTO` and adds `-funroll-loops -fno-math-errno -fno-trapping-math`. |

```sh
cmake -S . -B build-int -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON \
      -DAV2ENC_CPU_FAST=ON
cmake --build build-int -j8
```

`cmake` prints the effective settings at configure time:

```text
-- av2enc: host-CPU options: native=ON lto=ON cpu_fast=ON
```

## Bitstream identity

None of the flags can change integer results:

- No fast-math is ever added (no `-ffast-math`, `-funsafe-math-optimizations`,
  `-fassociative-math`, `-freciprocal-math`).
- `-ffp-contract=off` stays on `av2enc_core` and is appended after the
  `-mcpu=native` options, so native tuning cannot re-enable FMA contraction in
  the binary64 transforms.
- `-fno-math-errno` only stops libm calls from writing `errno`;
  `-fno-trapping-math` only assumes FP exceptions are not observed. Neither
  permits reassociation or different rounding.
- `-funroll-loops`, LTO and `-mcpu/-march=native` are result-neutral; integer
  decisions, tie-breaks and entropy state are unchanged.

Verified with the tuned build at the CPU benchmark points and the pinned
baseline:

| Stream | Bytes | MD5 | Default vs tuned |
|---|---:|---|---|
| 480x240 QP189, 60 f | 25,009 | `a1d0dba1259b20a4b628d4cd1e14c33c` | identical |
| 1440x720 QP90, 60 f | 958,023 | `f9b6e086aa7d9da78792cf74b9746741` | identical |

The 19 pinned gate streams are also unchanged with the tuned binary
(`feature_gate.sh --bin build-int/av2enc`: 30/30 checks pass).

`tools/refactor/verify_baseline.sh --check` also passes on the default build
(19 streams, ctest 23/23, 30 feature checks), so the portable default is
unchanged.

## Measured gain (Apple M4, macOS, Apple clang 21)

`nasa_480x240_30fps_60f.y4m --video --frames 60 --qp 189` and
`nasa_1440x720_60f.y4m --video --frames 60 --qp 90`, same source, default
`build-base` vs `-DAV2ENC_CPU_FAST=ON` `build-int`, interleaved runs on the same
host. The host is a fanless M4 MacBook Air that was also running other work, so
the table reports the **best of N** wall times; medians were 5–15% higher for
both binaries.

| Case | Default | `AV2ENC_CPU_FAST=ON` | Gain | N |
|---|---:|---:|---:|---:|
| 480x240 QP189 | 2.49 s (24.1 fps) | 2.00 s (30.0 fps) | +24% fps | 16 |
| 1440x720 QP90 | 26.2 s (2.29 fps) | 19.6 s (3.05 fps) | +33% fps | 6 |

Bitrate is unchanged by construction (same bytes): 100.0 kbps at 480x240,
7664 kbps at 1440x720.

### Where the gain comes from

Almost all of it is **full LTO**. On this tree, plain `-flto` cuts the encode
time by ~20–25% because the hot search/transform chain spans many small TUs and
LTO inlines the intra/inter search and filter helpers across them.

Two negative/neutral results worth keeping:

- CMake's `INTERPROCEDURAL_OPTIMIZATION` property selects **`-flto=thin`** with
  AppleClang, which measured **~18% slower** than the default build on the same
  workload (480x240: 2.27 s default vs 2.67 s `-flto=thin`, CPU time). The
  option therefore passes plain `-flto` explicitly instead of using the CMake
  IPO property.
- `-mcpu=native` and `-funroll-loops` were **within noise** on the M4 at both
  benchmark points (LTO-only and LTO+native+unroll differ by <2%). They stay in
  the aggregate because they are free there and `-march=native` can matter on
  x86-64 hosts with a wider autovectorization ISA.
- `-O2` instead of the Release `-O3` was slower with LTO (480x240: 1.99 s vs
  1.82 s CPU time), so the release optimization level is unchanged.
- `-fomit-frame-pointer` was not added: arm64 already omits the frame pointer
  at `-O2`, and on x86-64 the win is noise compared to losing unwindable stacks.

## Portability caveats

- `AV2ENC_NATIVE` / `AV2ENC_CPU_FAST` binaries require the build CPU (and, for
  arm64 Apple silicon, the same core generation). Do not distribute them; use
  the default build for release artifacts and for the CUDA product build.
- `AV2ENC_LTO` is ignored with a warning when `AV2ENC_ENABLE_CUDA=ON`: the
  device link is driven by `nvcc`, which does not accept `-flto`. The CUDA
  product build is unaffected; LTO is a host-CPU benchmark knob.
- LTO link flags are guarded with `$<LINK_LANGUAGE:CXX>` so CUDA device links
  are never touched.

## Interaction with `--cpu-threads`

`AV2ENC_CPU_FAST` and `--cpu-threads N` are orthogonal: the first tunes the
generated host code, the second parallelises independent tile encoders and the
batched NEWMV search. Combined on the M4 (best-of-2, interleaved):

| Case | Default, 1 thread | `CPU_FAST`, 8 threads | Gain |
|---|---:|---:|---:|
| 480x240 QP189, 60 f | 20.8 fps | 63.7 fps | 3.06x |
| 1440x720 QP90, 60 f | 2.58 fps | 6.41 fps | 2.48x |

The streams are byte-identical in every configuration (`a1d0dba1...` and
`f9b6e086...`).
