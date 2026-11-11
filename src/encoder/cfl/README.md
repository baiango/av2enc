# CfL — chroma from luma (AV2 §7.13.5)

CPU reference + CUDA kernel for AV2 v1.0.0 chroma-from-luma prediction, the
M1 intra tool from `docs/tool-roadmap.md`. This component is **prediction
only**: it produces chroma samples and the syntax values a future entropy pass
must code. It contains no bitstream/entropy edits and no integration into
`src/encoder/m0_still.cpp`.

## Files

| Path | Purpose |
|---|---|
| `cfl_common.h` | POD `Block` descriptor, §4.8 math, Div_Lut, §7.13.5 filters, alpha/signalling helpers. Shared verbatim with the CUDA kernel. |
| `cfl.h` / `cfl.cpp` | CPU reference: luma average, `L`, explicit/derived prediction, alpha search, `cflAllowed`. |
| `cfl_test.cpp` | 13,423 checks: published tables, signalling ranges, edges/partial blocks, alpha search, and a random cross-check against a literal sequential transcription of the spec. |
| `build_and_test.sh` | Standalone `g++` C++20 build (`SANITIZE=1`, `WERROR=1`, `OUT=...`). |
| `../cuda/cfl/` | CUDA kernel and CPU-vs-GPU bit-exactness test. |

Build and run the CPU reference:

```sh
./src/encoder/cfl/build_and_test.sh
SANITIZE=1 WERROR=1 ./src/encoder/cfl/build_and_test.sh
```

Expected output: `cfl_test: 13423 checks passed`.

## Spec mapping

| Spec | Implementation |
|---|---|
| §5.20.7.32 `read_cfl_alphas()` | `Signalling`, `explicit_signalling()`, `derived_signalling()`, `pair_from_signalling()` |
| §6.19.7.25 Table 6.26/6.27 | `kIndex*`, `kSign*`, `joint_signs()`, `signalable()` |
| §7.13.5 luma average | `compute_luma()` (`luma_avg`, `luma_above[]`, `luma_left[]`) |
| §7.13.5 array `L` | `compute_luma()` (`scratch.luma[i*w+j]`, 3 fractional bits) |
| §7.13.5 derived alpha | `derive_implicit_alpha()` |
| §7.13.5 prediction | `apply_prediction()` / `predict_*()` |
| §4.8 `Round2Signed`, §7.13.2.11 `approx_divide`, §7.13.3.22 `resolve_division` | `cfl_common.h` |
| `Cfl_Filters_420/422` | `ds_weight_420/422()` (checked entry-by-entry against the literal tables) |

## Alpha ranges and signalling

* `cfl_index` ∈ {`CFL_EXPLICIT`=0, `CFL_DERIVED_ALPHA`=1, `CFL_MULTI`=2}.
  `CFL_MULTI` is MHCCP and is **not implemented** here.
* Explicit per-plane alpha: sign ∈ {zero, neg, pos}, magnitude 1..8 eighths.
  `cfl_alpha_u` / `cfl_alpha_v` are the 0..7 symbols `magnitude - 1`
  (`CFL_ALPHABET_SIZE = 8`). Q8 alpha used by the prediction is
  `sign * magnitude * 32`.
* `cfl_alpha_signs = 3 * signU + signV - 1` (`CFL_JOINT_SIGNS = 8`);
  `(zero, zero)` is prohibited.
* Derived alpha is `resolve_division(nor, der, 8)`, Q8, range 0..511, and is
  not signalled.
* Entropy contexts for the future writer: `CFL_CONTEXTS = 3` (`is_cfl`) and
  `CFL_ALPHA_CONTEXTS = 6` (`cfl_alpha_u/v`). This component exposes the
  constants but does not write symbols.

## Block descriptor

`Block` is a trivially copyable POD shared with CUDA. Chroma coordinates;
luma origin is `(x << sub_x, y << sub_y)`; `chroma` holds the DC prediction on
input and receives CfL prediction on output.

* `luma_max_x/y` — inclusive bounds of the **reconstructed** luma plane. Pass
  the padded-64 reconstruction plane bounds, not the visible picture size. A
  partial bottom/right block reconstructs into the padding and the decoder
  reads those samples back, so clipping to the visible size would desync CfL.
* `coded_width/height` — `MiCols*MI_SIZE>>sub_x`, `MiRows*MI_SIZE>>sub_y`.
  These drive §7.13.5's right/bottom `prevSample` replication.
* `have_above` / `have_left` — `AvailUChroma` / `AvailLChroma`, derived at the
  **chroma mi** granularity (§5.20.4.1), not `y > 0`. For 4:2:0 with 64x64
  superblocks, the first chroma mi row/column covers chroma rows/cols 0..3, so
  these are false for `y < 4` / `x < 4` even when a sample row exists.
* `sb_top_y` — top luma row of the enclosing superblock (`(luma_y/64)*64`).
  §7.13.5 clamps the above-luma window to `max(sb_top_y - 1, ...)`.
* `cfl_sb_top_y(chroma_y, sub_y)`, `cfl_avail_above(chroma_y, sub_y)` and
  `cfl_avail_left(chroma_x, sub_x)` implement the two derivations above so the
  integration pass does not have to re-derive the mi arithmetic.

Only 8-bit 4:2:0 is an encoder format. The 4:2:2/4:4:4 branches are transcribed
and unit-tested at the weight level, but are not part of M1.

## Integration hooks (intra / RDO)

```cpp
av2enc::cfl::Scratch scratch;             // reuse per block; no allocation
av2enc::cfl::compute_luma(u_block, scratch);   // luma is plane-independent

// (a) derived alpha, no search:
int alpha = av2enc::cfl::derive_implicit_alpha(u_block, scratch);
av2enc::cfl::apply_prediction(u_block, alpha, scratch);
av2enc::cfl::apply_prediction(v_block, alpha, scratch);

// (b) RDO alpha decision (encoder-only; bitstream unchanged):
auto best = av2enc::cfl::search_alpha(
    u_block, v_block, src_u, src_v,
    av2enc::cfl::SearchMode::kExhaustiveSse, scratch);
if (best) {
  av2enc::cfl::apply_prediction_pair(u_block, v_block, best->alpha, scratch);
  // best->syntax.{cfl_index, alpha_signs, alpha_u, alpha_v} are the exact S()
  // symbols the entropy pass must code; best->total_sse() is the distortion.
}

// (c) cheap alpha without source planes:
search_alpha(..., av2enc::cfl::SearchMode::kNeighbourLeastSquares, ...);
```

`cfl_allowed(chroma_w, chroma_h, enable_cfl_intra, is_chroma_part,
frame_is_intra, cfl_allowed_in_sdp, lossless)` mirrors the `cflAllowed`
derivation in §5.20.5.6 so the mode search can skip CfL when it is illegal.

## Coverage / what the tests prove

* Div_Lut (all 129 entries) against the published table and the closed form.
* Every `Cfl_Filters_420` / `Cfl_Filters_422` entry; each filter integrates to
  8, so constant luma maps to itself.
* All 8 `cfl_alpha_signs` combinations and the full magnitude 1..8 range;
  `(zero, zero)` rejected; snap/round-trip.
* Constant-luma invariance (CfL is a no-op), clipping at 0/255.
* Right/bottom `prevSample` replication, superblock-top luma clamp, partial
  blocks reconstructed into padding.
* Derived alpha recovers `magnitude * 32` Q8 for synthetic linear
  luma/chroma pairs; zero variance yields 0.
* Exhaustive search recovers a known alpha with zero SSE; both-zero avoidance
  always returns a signalable pair.
* 400 randomised blocks compared to an independent literal transcription of
  the spec pseudocode (luma average, L, prediction, derived alpha).
* ASan/UBSan clean.

## Limitations

* MHCCP (`CFL_MULTI`, §7.13.6) is out of scope.
* No transform/reconstruction: prediction operates on planes the caller has
  already populated with DC prediction.
* No RD cost with a rate term; `search_alpha` returns distortion (SSE) only.
  The RDO pass owns lambda and the entropy rate estimate.
* CUDA kernel is component-tested, not wired into `m0_pipeline`.

## M1 verification finding (m1_verify, default CfL)

The default M1 chroma path was reported as desyncing while `AV2ENC_M1_UV_DC=1`
was exact. Bisecting on the box showed the **CfL module was correct**; the
fault was in M1's chroma mode selection/integration. It is resolved by M1
commit `ff254d0` (default chroma restricted to DC/CfL until directional UV
prediction is bit-exact, SB border availability fixed, frame initialization
aligned with the decoder).

Measured on `99b9394` (before) and `ff254d0` (after), `--m1`, default CfL:

| Resolution | Before U/V diffs | After |
|---|---|---|
| 64x64 | U 125/1024, V 163/1024, max 7 | 0/0 |
| 640x360 | U 28800/57600, V 28664/57600, max 114 | 0/0 |

`tools/m1_verify/run_matrix.sh --sizes "64x64 640x360" --modes dc` now reports
`2/2 cases bit-exact`, and 1920x1080 is also exact.

Evidence that the CfL path itself is sound (measured on `99b9394` with a
scratch M1 that forces the UV mode per run):

| Forced UV mode | Result at 640x360 |
|---|---|
| CfL only (`is_cfl=1`, derived alpha) | bit-exact |
| DC only, `is_cfl=0` still coded | bit-exact |
| DC / V / H / D135 / D113 / D157 / SMOOTH_H / PAETH | bit-exact |
| D45 / D203 / D67 / SMOOTH / SMOOTH_V | mismatch |

Pure-CfL runs (every chroma block CfL) are `0/0/0` at 64x64, 640x360 and
1920x1080, so the §7.13.5 math, derived alpha, edge handling and CfL syntax
are correct. The first differing sample in the failing default stream was in a
**cfl=0** block (640x360 luma 320,0 used UV mode 10; 64x64 luma 32,32 used UV
mode 8), which is why the fix belongs to the chroma mode-selection path.

### Non-CfL notes (for their owners)

These do not affect the current default (`ff254d0` selects only DC/CfL), but
they will resurface if directional chroma prediction is re-enabled:

1. `src/encoder/m1/m1_still.cpp`, `neighborhood_smooth()` chroma branch: it
   used to halve the coordinates a second time (`cr = r / 2; cc = c / 2;`).
   **Fixed**: the caller already passes the chroma 4x4-grid index
   (`y0 >> 2, x0 >> 2`), so the function now uses `cr = r; cc = c;`.
2. `src/encoder/intra_modes/` directional/smooth prediction for kBase indices
   3, 7, 8, 9, 10 (D45, D203, D67, SMOOTH, SMOOTH_V) diverges when forced; the
   first difference is on the no-above top row. The intra agent has WIP tests
   for the above-right / SB-border paths.
3. `choose_uv_mode()` scores base candidates with U-only SSE
   (`sse_against_source(1, ...)`) but scores CfL with U+V SSE, biasing the
   choice against CfL. Score both planes for every candidate.

### Luma-plane contract

CfL reads reconstructed luma from the current block plus the above/left
neighbour lines; the caller must make the whole plane consistent with the
decoder's `CurrFrame`, including samples outside the visible picture. `luma_max`
may be the coded (`MiCols*4-1`) or the padded-64 bound — both were tested
bit-exact once M1 clamps chroma residual sizes at frame edges. Do not replace
the plane's initialization/padding values with encoder-only defaults: the
decoder's values are part of the prediction input wherever availability allows
a read.
