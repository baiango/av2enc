# CfL explicit alpha path (AV2 §5.20.7.32 / §6.19.7.25)

This directory adds the **explicit** chroma-from-luma alpha path next to the
existing derived-only CfL core. It owns no files outside
`src/encoder/cfl/alpha/`; `../cfl.*`, `src/cuda/cfl/`, M1/M2 and the build files
are untouched.

What was missing before: `../cfl_common.h` already defined `Alpha`,
`Signalling`, `signalable()` and `explicit_signalling()`, and
`../cfl.cpp::search_alpha()` could find an explicit pair, but nothing selected
between derived and explicit, nothing signalled `CFL_EXPLICIT`, and M1 always
wrote `cfl_index = CFL_DERIVED_ALPHA` (`still_syntax.cpp:795`). This component
supplies that path end to end, including the exact symbol writer/reader.

Prediction is not reimplemented. `apply_candidate()` calls the shared
`cfl::apply_prediction()`, which is the same arithmetic the CUDA kernel uses, so
CPU/GPU parity is inherited from the existing CfL core.

## Files

| Path | Purpose |
|---|---|
| `alpha.h` / `alpha.cpp` | Quantisation, joint-sign explicit search, derived-vs-explicit RD/SSE decision, candidate evaluation/apply. |
| `syntax.h` / `syntax.cpp` | Pinned §9.3 CDFs, §8.3 contexts, model-rate calculation, exact `read_cfl_alphas()` writer/reader. |
| `alpha_test.cpp` | 43,149 checks: quantisation, all signs/magnitudes/U-V pairs, CDF pinning, writer/reader round-trips and pinned goldens, search recovery, RD/tie-break, clipping, edges, partial blocks, randomised selection. |
| `alpha_gpu_test.cu` | Host-only CUDA API test against the shared `src/cuda/cfl` kernel; 110 cases / 3.7M samples. |
| `cuda_emulation.cpp`, `cuda_stub/` | Local no-nvcc emulation of the CUDA entry points (test logic only). |
| `alpha_quality.cpp` | Prediction-level derived-vs-explicit quality delta tool. |
| `build_and_test.sh` | CPU build + test. |
| `build_cuda_test.sh` | Real `nvcc` CPU-vs-GPU gate (run on the sm_86 box). |
| `build_emulated_gpu_test.sh` | Runs the CUDA test logic against the CPU core. |
| `build_quality.sh` | Builds (and optionally runs) the quality tool. |

## API

```cpp
#include "encoder/cfl/alpha/alpha.h"

using namespace av2enc::cfl::alpha;

// 1. Quantise a fitted slope (Q8) to the signalled 1..8 eighths.
Alpha a = quantise_q8(q8);                 // 0 -> CFL_SIGN_ZERO, ties away
AlphaPair p = quantise_pair(q8_u, q8_v);   // (zero,zero) is repaired
// ...or snap the two §7.13.5 derived slopes without reading the source:
AlphaPair q = quantise_derived_pair(u, v, scratch);

// 2. Exhaustive explicit search: 8 joint sign symbols x 1..8 magnitudes.
Config config;                             // use_rate=false => SSE
CflCdfs cdfs = default_cfl_cdfs();
Scratch scratch;
auto best = search_explicit(u, v, src_u, src_v, config, cdfs, scratch);

// 3. Choose CFL_DERIVED_ALPHA vs CFL_EXPLICIT (RD or SSE).
config.cost.use_rate = true;
config.cost.lambda_q8 = 558;               // lambda=2.178 * 256
auto decision = decide_cfl_alpha(u, v, src_u, src_v, config, cdfs, scratch);
if (decision) {
  // decision->chosen.syntax is the exact Signalling to code
  apply_candidate(u, v, decision->chosen, scratch);
}

// 4. Write/read the exact syntax (is_cfl is the preceding uv-mode symbol).
av2::Encoder enc;
enc.init_symbol(buffer, sizeof buffer);
write_cfl_mode_and_alphas(enc, cdfs, is_cfl_context(above, left),
                          /*is_cfl=*/true, decision->chosen.syntax);
size_t sz = 0;
enc.exit_symbol(&sz);
```

`Candidate` carries `syntax`, `explicit_pair`, the exact applied `q8_u`/`q8_v`,
per-plane SSE, and the model `rate_q8`. `Candidate::cost_q16()` is
`65536 * (SSE + lambda * bits)` in exact integer arithmetic.

### Deterministic tie-break

`decide_cfl_alpha()` chooses:

1. lower RD cost (or lower SSE when `use_rate == false`);
2. lower total SSE;
3. `Config::tie_break` — derived by default, so an equal-cost block keeps the
   existing derived stream; `kPreferExplicit` flips it;
4. explicit candidates prefer lower `cfl_alpha_signs`, then lower
   `cfl_alpha_u`, then lower `cfl_alpha_v` (iteration order + strict `<`).

No floating point is used in the decision. `rate_q8` is computed from the §8.2
boundaries with an integer `log2` routine, so the same inputs give the same
decision on every host.

## Exact syntax

`§5.20.7.32` with CfL enabled and MHCCP not allowed:

```
cfl_index          S()   // 2 symbols: CFL_EXPLICIT=0, CFL_DERIVED_ALPHA=1
if (cfl_index != CFL_EXPLICIT) return
cfl_alpha_signs    S()   // 8 symbols, §9.3 Default_Cfl_Sign_Cdf
signU = (signs + 1) / 3
signV = (signs + 1) % 3
if (signU != CFL_SIGN_ZERO) cfl_alpha_u S()   // 8 symbols
if (signV != CFL_SIGN_ZERO) cfl_alpha_v S()   // 8 symbols
```

CDF contexts (§8.3 Tables 8.4/8.5):

| `cfl_alpha_signs` | signU | signV | `cfl_alpha_u` ctx | `cfl_alpha_v` ctx |
|---:|---:|---:|---:|---:|
| 0 | zero | neg | n/a | 0 |
| 1 | zero | pos | n/a | 3 |
| 2 | neg | zero | 0 | n/a |
| 3 | neg | neg | 1 | 1 |
| 4 | neg | pos | 2 | 4 |
| 5 | pos | zero | 3 | n/a |
| 6 | pos | neg | 4 | 2 |
| 7 | pos | pos | 5 | 5 |

`cfl_alpha_u` symbol is `|CflAlphaU| - 1` in 0..7; the prediction uses
`CflAlphaU * 32` (Q8). `(zero, zero)` has no sign symbol and is rejected by
`signalable()` and every entry point here.

Pinned default CDFs (§9.3, pages 627/647-648) live in `syntax.h` and are checked
byte-for-byte against the literal spec text **and** the repo-generated
`src/encoder/m1/still_cdf_tables.h`.

### Pinned goldens

`is_cfl=1` at context 1 followed by the CfL alpha syntax:

| Mode | Payload (hex) |
|---|---|
| explicit, signs=3 `(neg,neg)`, `cfl_alpha_u=7` (8/8), `cfl_alpha_v=2` (3/8) | `732c4000` |
| `CFL_DERIVED_ALPHA` | `a00000` |
| alpha syntax only (no `is_cfl`): all 8 sign symbols, magnitudes 1..8/8..1, then derived | `06e7550bf9bc5652780402b66f0000` |

These are regression pins from the normative §8.2 coder, not published
conformance vectors. The derived payload is byte-identical to the sequence M1
emits today (tested by re-running its raw `is_cfl` + `cfl_index` calls).

## Minimal stream / exact call sites

A full `dav2d`-decodable IVF requires the M1 tile syntax, transform, residual and
frame headers, and this directory may not edit `still_syntax.*` / `still_encoder.*`.
What is provided instead:

* a **minimal CfL tile payload** (the pinned payloads above) that exercises
  the real §8.2 coder end to end and round-trips the exact fields;
* the **exact call sites** for wiring it into M1:

1. `src/encoder/m1/still_syntax.cpp:790` `code_uv_mode()` — replace
   `sym(&cdfs_.cfl_index[0], 2, 1)` (line 795) with
   `alpha::write_cfl_alphas(enc, cdfs_, node.cfl_syntax)`. `BlockSyntax` needs
   the four `Signalling` fields (or one `Signalling` member). Derived-only
   streams are unchanged (`write_cfl_alphas` emits the same `cfl_index=1`).
2. `src/encoder/m1/still_encoder.cpp:1094` and `:1330` — after
   `make_cfl_block()`, call `alpha::decide_cfl_alpha()` instead of the two
   `derive_implicit_alpha()` + `apply_prediction()` pairs, and store
   `decision->chosen.syntax` in the leaf. `apply_candidate()` replaces the two
   `apply_prediction()` calls.
3. `src/encoder/m1/rdo_decide/chroma_rdo.cpp:246`
   `default_chroma_syntax_bits()` — the hard-coded 1.5 bits for CfL is the
   derived path only. Use `alpha::alpha_syntax_rate_q8(syntax, cdfs) / 256.0`
   so explicit candidates pay for their sign/magnitude symbols.
4. `src/cuda/m1/cfl/still_cfl.cu:105` — the GPU M1 path calls
   `cfl_predict_derived_batch()`. For explicit alphas call
   `cfl_predict_batch()` with `alpha_q8 = CflAlpha * 32` per plane; the shared
   kernel already accepts arbitrary Q8. Reconstruct/apply then matches the CPU
   path byte for byte (see the parity test).

## Gate and parity

CPU reference:

```sh
./src/encoder/cfl/alpha/build_and_test.sh          # 43149 checks passed
SANITIZE=1 WERROR=1 ./src/encoder/cfl/alpha/build_and_test.sh
```

Local CUDA-API emulation (no GPU, test logic only):

```sh
./src/encoder/cfl/alpha/build_emulated_gpu_test.sh
# emulation contract: host descriptor arrays rejected
# cfl_alpha_cuda_test: 110 cases, 3702784 samples, 2 alphas, all exact
```

Real CPU-vs-GPU gate (must run on the Linux + CUDA box; no `nvcc` or GPU is
available on the macOS host used for this change):

```sh
./src/encoder/cfl/alpha/build_cuda_test.sh
# expected: cfl_alpha_cuda_test: <N> cases, <M> samples, <K> alphas, all exact
# localize a fault to one case:
ALPHA_CUDA_VERBOSE=1 ./src/encoder/cfl/alpha/build_cuda_test.sh
# optional memory gate:
compute-sanitizer --tool memcheck \
  ./src/encoder/cfl/alpha/build-alpha-cuda/cfl_alpha_cuda_test
```

Every case is followed by `cudaDeviceSynchronize()` and reports
`case <n> [<name>] stage <stage>` on failure. The batch entry point
(`cfl_predict_batch`) takes a **device** `CflJob*`; the test uploads its host
descriptor array with `cudaMemcpy` before launching. The emulation's
`cudaMalloc` registry rejects a host `CflJob*` so that contract mistake fails
locally instead of only as an illegal access on the device.

The GPU test links `src/cuda/cfl/cfl.cu` and compares full chroma planes for:
all 17 per-plane alphas, all 8 joint signs at sampled magnitudes, frame edges /
superblock top / partial coded grids, clipping at 0/255, derived alpha
(including the reported implicitAlpha), the CPU-selected candidate, and the
batch entry point.

## Quality delta (prediction level)

`alpha_quality.cpp` measures the alpha decision alone: ideal DC (source block
mean), original luma, source chroma neighbours for the derived regression.
Absolute PSNR is optimistic; the derived-vs-explicit difference is the result.
M1's QP-90 calibration is lambda ≈ 2.18 SSE/bit, so the sweep is 0..8.

`testdata/shine_256x256_1f.y4m`, 8x8 chroma blocks (256 blocks, 32,768 chroma
samples):

| Mode | SSE | PSNR |
|---|---:|---:|
| derived only | 501,806 | 36.28 dB |
| explicit only | 281,618 | 38.79 dB |
| RD selection, lambda=0 | 245,574 | 39.38 dB |
| RD selection, lambda=2.18 | 245,604 | 39.38 dB |

At lambda=2.18 the selector picks explicit for 63/256 blocks. The derived
`cfl_index` symbol costs 0.69 bits/block under the default CDF; the selector's
average alpha-syntax cost is 2.5 bits/block because the explicit blocks pay the
sign and magnitude symbols. Explicit-only is **+2.51 dB** over derived-only,
and selecting per block is **+3.10 dB** over derived-only (because derived is
occasionally much better and is kept).

`testdata/basic_64x64_1f.y4m`, 8x8 blocks (16 blocks):

| Mode | SSE | PSNR |
|---|---:|---:|
| derived only | 198,402 | 28.27 dB |
| explicit only | 82,102 | 32.10 dB |
| RD selection, lambda=2.18 | 69,535 | 32.82 dB |

Explicit-only is **+3.83 dB**; per-block selection is **+4.55 dB**. These are
prediction-SSE numbers, not end-to-end bitrate/quality: the coefficient rate
cost of switching modes is not modelled here (see Limitations).

## Limitations

* MHCCP (`CFL_MULTI`, §7.13.6) is out of scope; the writer rejects it rather
  than inventing the `cfl_mhccp`/`cfl_mh_dir` symbols.
* The RD model covers the alpha syntax only. It has no coefficient/transform
  rate term, so the quality numbers above are not a bitrate-matched result.
* The derived-vs-explicit decision is host-side. GPU parity covers prediction,
  which is the only part of this path that touches pixels.
* The minimal payloads are CfL syntax fragments, not complete AV2 frames; they
  cannot be fed to `dav2d` without the M1 integration listed above.
* 4:2:2/4:4:4 are inherited from the core's arithmetic but are not encoder
  formats and are not exercised here.
