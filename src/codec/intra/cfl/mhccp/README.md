# MHCCP — multi-hypothesis cross-component prediction (AV2 §7.13.6)

Correctness-first CPU reference, shared device path and CUDA kernel for
`cfl_index == CFL_MULTI`, plus the `read_cfl_alphas()` syntax that selects it.

This component owns no M0/M1 core file. Everything lives in
`src/encoder/cfl/mhccp/`.

## Files

| Path | Purpose |
|---|---|
| `mhccp_common.h` | POD `Block`/`Refs`/`MultiParams`, CflRef capture, §7.13.8 division tables, `mul_fixed32_adapt`, Gaussian elimination, prediction. Deliberately standard-library-free so the CPU reference, host-emulation test and CUDA kernel compile the same integer code. |
| `mhccp.h` / `mhccp.cpp` | CPU API: validation, `is_mhccp_allowed`, source-SSE evaluation, direction search, explicit/derived/multi hypothesis decision. |
| `mhccp_device.cuh` | Device preparation/sample functions shared by the kernel and the host-emulation test. |
| `mhccp_gpu.h` / `mhccp_gpu.cu` | CUDA single-block and batch entry points. One CUDA block per chroma block; thread 0 captures CflRef/derives Q16 weights, all threads predict. |
| `mhccp_syntax.h` / `mhccp_syntax.cpp` | §5.20.7.32 writer/reader: `cfl_mhccp`, `cfl_index`, `cfl_mh_dir`, explicit alpha symbols. |
| `mhccp_cdf_tables.h` | The six §9.3 default CfL CDFs and `Size_Group[29]`. |
| `mhccp_test.cpp` | CPU correctness tests. |
| `mhccp_emulation_test.cpp` | Host-emulation parity test; no GPU/nvcc. |
| `mhccp_gpu_test.cu` | CUDA vs CPU byte-exactness test (run on the Linux box). |
| `build_and_test.sh`, `build_emulation_test.sh`, `build_gpu_test.sh` | Standalone builds. |

## API

```cpp
#include "encoder/cfl/mhccp/mhccp.h"
#include "encoder/cfl/mhccp/mhccp_syntax.h"

namespace mh = av2enc::cfl::mhccp;

mh::Block u = mh::make_block(cfl_u, num4_above_right, num4_below_left,
                             tile_col_end, tile_row_end);
mh::Block v = mh::make_block(cfl_v, num4_above_right, num4_below_left,
                             tile_col_end, tile_row_end);
// u.chroma / v.chroma hold the DC prediction on input.
// Above/left reconstructed chroma must already be present.

mh::SourcePlane su{src_u, stride_u, max_x_u, max_y_u};
mh::SourcePlane sv{src_v, stride_v, max_x_v, max_y_v};

auto best = mh::search_mhccp(u, v, su, sv);       // tries cfl_mh_dir 0..2
if (best) {
  mh::apply_mhccp_pair(u, v, best->mh_dir,
                       best->params_u, best->params_v);
  // best->params_u/v are Q16 weights; best->mh_dir is cfl_mh_dir;
  // best->count_u/v are the least-squares sample counts.
  mh::CflSyntax syntax = mh::make_multi_syntax(best->mh_dir);
  int size_group = mh::size_group_for_chroma_block(u.w, u.h);
  // write_cfl_alphas(...) emits cfl_mhccp=1, cfl_mh_dir, and no alpha.
}
```

For a full CfL decision (explicit quantised alpha, derived alpha, or MHCCP):

```cpp
cfl::Scratch scratch;
mh::DecisionOptions options;          // allow_explicit/derived/multi, tool flags
auto decision = mh::decide_cfl(cu, cv, su, sv, num4_above_right,
                               num4_below_left, tile_col_end, tile_row_end,
                               options, scratch);
if (decision) {
  mh::apply_decision(cu, cv, *decision, scratch);
  // decision->cfl_index, cfl_mh_dir, alpha, syntax are the exact symbols.
}
```

`mh::allowed(w, h, enable_mhccp, is_chroma_part, frame_is_intra,
cfl_allowed_in_sdp, lossless)` mirrors §5.20.5.6 `is_mhccp_allowed()`.

### Geometry contract

`Block` extends the existing `cfl::Block` with the fields §7.13.2.1 needs:
`chroma_max_x/y`, `num4_above_right`, `num4_below_left`, `mi_col_end`,
`mi_row_end` (in **luma samples**, `MiColEnd * MI_SIZE`), and `sb_luma_y`.
`make_block()` derives the single-tile values from `coded_width`/`coded_height`
and `sb_top_y`; tile streams must pass the real tile ends and may overwrite
`chroma_max_x/y` with padded reconstruction bounds.

Only 8-bit 4:2:0 with `4 <= w,h <= 32` is accepted. That is exactly the
non-lossless `is_mhccp_allowed()` range, and it keeps CflRefWidth/Height below
the 128-sample cap. Lossless 4x4 is also accepted when the caller enables it.

## Spec mapping

| Spec | Implementation |
|---|---|
| §7.13.2.1 CflRef capture | `capture_refs()` |
| §7.13.2.1 `get_cfl_luma_sample` | `cfl_luma_sample()` |
| §7.13.6 MHCCP prediction | `predict_sample()`, `apply_mhccp()` |
| §7.13.7 derive multi params | `derive_multi_params()` |
| §7.13.8 Gaussian elimination | `derive_multi_params()` (`get_division_scale_shift`, `mul_fixed32_adapt`) |
| §9.2 constants/tables | `mhccp_common.h`, `mhccp_cdf_tables.h` |
| §5.20.7.32 / §6.19.7.25 syntax | `mhccp_syntax.*` |

### Weights and quantisation

* MHCCP has **no signalled weights**. `multiParams[3]` is derived from the
  reconstructed above/left chroma and the filtered luma by §7.13.7; only
  `cfl_mh_dir` (0..2) is coded. The fixed-point quantisation is the Q16
  (`MHCCP_BITS = 16`) normalisation/elimination in §7.13.7-§7.13.8.
* `CFL_EXPLICIT` alpha is signalled as sign + `1..8` eighths; the prediction
  weight is Q8 (`magnitude << 5`). `CFL_DERIVED_ALPHA` is `resolve_division(..., 8)`
  and is not signalled. `snap_alpha_q8()` is the encoder-side quantiser.
* `cfl_index`: 0 explicit, 1 derived, 2 multi. `cfl_mhccp` is coded only when
  `enable_cfl_intra && is_mhccp_allowed()`. When `enable_cfl_intra == 0`,
  `cfl_mhccp = 1` is implicit and `cfl_index` is not coded.

## Tests and parity

CPU (synthetic + real 4:2:0):

```sh
./src/encoder/cfl/mhccp/build_and_test.sh
SANITIZE=1 WERROR=1 ./src/encoder/cfl/mhccp/build_and_test.sh
```

* `660691` checks, `0` failures (ASan/UBSan clean).
* Independent literal transcription of capture, least squares, Gaussian
  elimination and prediction agrees on every synthetic case (4/8/16/32,
  top-left/interior/SB-top/partial, all `cfl_mh_dir`, filters 0..3,
  above-right/below-left extents).
* Real 4:2:0: `237` blocks per plane from `tests/data/basic_64x64_1f.y4m` and
  `18636` blocks per plane from `tests/data/basic_640x360_1f.y4m` agree
  byte-for-byte on both U and V.
* Conditioning: `count == 0` returns `{0, 0, 1<<16}`; constant luma/chroma
  (singular normal equations) stays finite and within 1 LSB of the constant;
  `get_division_scale_shift` approximates `denom * scale >> shift = 65536`
  for all tested denominators; 20,000 random `mul_fixed32_adapt` cases match a
  64-bit reference.
* Syntax: every explicit sign/magnitude, derived, and all three MHCCP
  directions round-trip through the normative §8.2 coder for all
  `enable_cfl_intra` / `is_mhccp_allowed` / `Size_Group` combinations, with
  both fixed and adaptive CDFs.

Host emulation of the shared device path:

```sh
./src/encoder/cfl/mhccp/build_emulation_test.sh
# mhccp_emulation_test: 1201 cases, 16803 checks, 0 failures
```

GPU (Linux + CUDA box; not run on the macOS host because there is no `nvcc`):

```sh
./src/encoder/cfl/mhccp/build_gpu_test.sh
ARCH=sm_120 ./src/encoder/cfl/mhccp/build_gpu_test.sh
```

`mhccp_gpu_test.cu` compares every predicted sample and every derived Q16
parameter against `mhccp.cpp`, for explicit and derived modes, directions 0..2,
4x4..32x32, edge/partial blocks, and a 24-job batch with per-job parameter
writeback.

## Decoder gate / core integration

**No minimal decodable stream can be emitted without core edits.** The current
M1 writer has no way to signal `enable_mhccp`, `cfl_mhccp`, `cfl_index ==
CFL_MULTI`, or `cfl_mh_dir`, and the decision path has no MHCCP candidate.
The local syntax round-trip above is not a frame gate. The exact required call
sites are:

1. `src/encoder/m1/still_headers.h` — add `bool enable_mhccp = false;` to
   `StillConfig` (next to `enable_cfl_intra`, line 65).
2. `src/encoder/m1/still_headers.cpp:119` — replace the hardcoded
   `writer.put_bit(false); // enable_mhccp` with `writer.put_bit(c.enable_mhccp);`
   (the bit position is already `f(1)` after `cfl_ds_filter_index`).
3. `src/encoder/m1/still_syntax.h` — add `kFlagIsMhccp = 1u << 3` to
   `BlockFlags` (line 52), and a `std::uint8_t cfl_mh_dir` to `BlockSyntax`
   (leaf mode info, near line 80). Keep the existing `kFlagIsCfl` for
   `CFL_DERIVED_ALPHA`.
4. `src/encoder/m1/still_syntax.cpp` — add the two CDFs to `CdfState` (near line
   177), copy them in `init_cdfs()` (near line 225), and replace
   `code_uv_mode()` (line 790) with the §5.20.7.32 order:
   `S() cfl_mhccp` when `config_.enable_cfl_intra && is_mhccp_allowed()`,
   forced multi when `!enable_cfl_intra`, `S() cfl_index` only when
   `!cfl_mhccp`, `S() cfl_mh_dir` when multi, and the explicit alpha branch.
   The outer `is_cfl` symbol must be coded when
   `cflAllowed || is_mhccp_allowed()`, not only when `enable_cfl_intra`;
   MHCCP is an `is_cfl == 1` UV mode. The call at line 1432 must pass the
   MHCCP flag/direction.
5. `src/encoder/m1/still_encoder.cpp` — around `choose_uv_mode()` (line 1222) and
   the current CfL candidate (line 1313), build `mh::Block` descriptors with
   `make_block()`, call `mh::decide_cfl()`/`search_mhccp()`, and set
   `kFlagIsCfl` or `kFlagIsMhccp` plus `cfl_mh_dir` on the leaf (line 1148).
   Add `config.enable_mhccp` beside `config.enable_cfl_intra` at line 1814.
6. `src/main.cpp` — expose an opt-in `--mhccp` flag only after 1-5 pass
   `tools/validate_dav2d.sh`; do not enable it by default before the gate.

With those edits the gate is: encode one intra frame with `enable_mhccp=1`,
force a single MHCCP chroma block, decode with `dav2d --strict 1`, and require
an exact recon match before any quality measurement.

### Build integration (not done here)

`CMakeLists.txt` is outside this component's ownership. To link the module,
add `src/encoder/cfl/mhccp/mhccp.cpp` and
`src/encoder/cfl/mhccp/mhccp_syntax.cpp` to `av2enc_core`, and
`src/encoder/cfl/mhccp/mhccp_gpu.cu` to the CUDA object target. The standalone
scripts above already verify both paths without touching the root build.

## Missing normative tables

Nothing in §7.13.6-§7.13.8 is missing from the cached spec. The two CDF arrays
needed by `read_cfl_alphas()` exist in `spec/tables/default_cdf.json` as
`Default_Cfl_Mhccp_Cdf` and `Default_Cfl_Mh_Dir_Cdf`, but there is no generated
C++ header for them in `src/encoder/m1/still_cdf_tables.h`. They are copied
verbatim into `mhccp_cdf_tables.h` together with `Size_Group[29]` from
`spec/tables/block_tables.json`. All division tables (`Division_Pow2_W/O/B`)
and constants (`MHCCP_BITS`, `DIV_PREC_BITS`, `DIV_PREC_BITS_POW2`,
`DIV_SLOT_BITS`) are in the spec text/`constants.json`. The three tables are
reproduced as `__host__ __device__ constexpr` accessor functions rather than
namespace-scope arrays, so the device translation unit compiles without
`--expt-relaxed-constexpr`.

## Limitations

* 4:2:0 only (the encoder format); 4:2:2/4:4:4 are rejected.
* Non-lossless `w,h <= 32` (spec `is_mhccp_allowed`); lossless 4x4 allowed.
* No rate term in `decide_cfl()`; the caller owns lambda and the entropy-rate
  estimate.
* Not wired into M1/RDO/bitstream; see the gate call sites above.
* GPU parity is host-emulation-verified here; the nvcc test must be run on the
  Linux + CUDA box.
