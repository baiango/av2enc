# Global motion estimation (AV2 v1.0.0)

CPU reference for frame-level global motion: model fitting from
correspondences / per-block warps / block MVs, deterministic degradation and
quantisation, `global_motion_params()` syntax, `GLOBALMV` projection, and a
minimal `dav2d` decoder gate.

**Component only.** Nothing outside `src/encoder/inter/gme/` is edited. The M2
core still hard-codes `enable_global_motion = 0`; the exact integration call
sites are listed below. There is **no BD-rate or integration claim**.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(commit `38f28e7`):

- §5.4.6 `sequence_inter_config()` — `enable_global_motion`
- §5.18.9.1–.6 — `global_motion_params()`, `read_global_param()`,
  `scale_warp_model()`, subexp/recenter coding
- §7.12.2.1–.2 — setup global MV / get warp motion vector
- §7.13.3.19–.21 — block warp, extended warp fallback, setup shear
- §9.2 — `GM_*`, `WARPEDMODEL_*`, `Default_Warp_Params`

## Status

| Item | State |
|---|---|
| GM types (`IDENTITY`, `ROTZOOM`, `AFFINE`), Q16 models | complete |
| Fit from correspondences (weighted integer least squares) | complete |
| Fit from per-block warps (centre + 4 offsets) | complete |
| Fit from block MVs | complete |
| Deterministic outlier rejection | complete |
| AFFINE → ROTZOOM / IDENTITY degradation | complete |
| Quantisation to §5.18.9 grids and forced-integer-MV mode | complete |
| `global_motion_params()` writer + reader (round-trip) | complete |
| §7.12.2.2 projection, MV clamp, precision lowering | complete |
| `GLOBALMV` block decision helper | complete |
| CPU unit/synthetic/real-720p tests | 21750 checks, pass |
| Minimal `dav2d` gate (ROTZOOM / AFFINE / shear / base-diff / 720p) | 6 streams, 0 mismatches |
| CUDA kernel | not implemented (CPU reference only) |
| Runtime M2 integration / RDO | not wired |

## Files

```text
src/encoder/inter/gme/
  gme_core.h       §4.8/§9.2 integer helpers, constants, shear validity
  gme.h            public API: types, fitting, validation, projection
  gme.cpp          integer least squares, quantisation, degradation, proxy
  gme_syntax.h     §5.18.9 writer/reader, history, scale_warp_model
  gme_syntax.cpp   global_motion_params() + read/write_global_param
  gme_test.cpp     21750 checks: syntax, math, fits, synthetic, real 720p
  gme_gate.cpp     two-frame dav2d gate (standalone; see below)
  build_and_test.sh
  README.md
```

## Data model

`GmParams` is six Q16 values plus `GmType`:

```text
ref_x = p[2]*x + p[3]*y + p[0]      p[0] = x/column translation
ref_y = p[4]*x + p[5]*y + p[1]      p[1] = y/row translation
```

Correspondences and block MVs use **1/8 luma samples** (`q8`); one luma sample
is 8 q8 units. This matches §7.13.3 motion-vector precision. A `GmPoint` maps
`(src_x, src_y)` in the current frame to `(dst_x, dst_y)` in the reference.
`GmBlockMv` is `dst = centre + mv`; `GmWarpSample` carries a per-block model and
is sampled at the centre and four radius offsets so rotation/zoom is retained.

`IDENTITY` has `p = {0,0,65536,0,0,65536}`. `ROTZOOM` requires
`p[4] = -p[3]` and `p[5] = p[2]`; `AFFINE` has no symmetry constraint.
`GmType` values match §5.18.9 (`IDENTITY=0`, `ROTZOOM=1`, `AFFINE=2`).

### API sketch

```cpp
#include "encoder/inter/gme/gme.h"
#include "encoder/inter/gme/gme_syntax.h"

av2enc::inter::gme::GmOptions opt;
opt.frame_width = width;
opt.frame_height = height;
auto estimate = av2enc::inter::gme::estimate_gm_block_mvs(blocks, opt);
if (estimate.diag.accepted && !estimate.diag.identity) {
  auto use = av2enc::inter::gme::global_mv_for_block(
      estimate.params, /*ref=*/0, mi_col, mi_row, bw4, bh4, mi_cols, mi_rows);
  // use.use_warp selects the warp predictor; use.mv is the 1/8-pel centre MV.
}

av2enc::inter::gme::GmSyntaxContext ctx;
ctx.frame_is_intra = false;
ctx.enable_global_motion = true;
ctx.use_global_motion = true;
ctx.num_total_refs = 1;
ctx.order_hint = 1;
ctx.ref_order_hint[0] = 0;
ctx.ref_frame_idx[0] = 0;
ctx.params[0] = estimate.params;
std::string error;
av2enc::inter::gme::write_global_motion_params(writer, ctx, &error);
```

`make_gm_frame_plan(estimates, true)` (a `std::span<const GmEstimate>`) is the
convenience form for the sequence/frame enable bits.

## Estimation

All accumulation is integer (`__int128`) in a fixed order; the only
non-integer values are reporting fields in `GmDiagnostics`. Same input and
order produce the same model and the same bytes. Tests reverse the point order
and require identical output.

1. Weighted means `x̄, ȳ, ū, v̄`; centred coordinates are shifted by a power of
   two so the normal equations stay inside 128-bit arithmetic.
2. `kTranslation`: `t = (ū - x̄, v̄ - ȳ)`.
3. `kRotZoom`: weighted similarity least squares
   `a = Σw(xu+yv) / Σw(x²+y²)`, `b = Σw(xv-yu) / Σw(x²+y²)`,
   then `p2=p5=a`, `p3=-b`, `p4=b`.
4. `kAffine`: 2×2 normal equations per output axis, solved with the exact
   `det = Sxx*Syy - Sxy²` and a 128-bit long-division `ratio_q16` (round to
   nearest, no host FP environment dependence).
5. Translation is recovered from the means: `t = ū - a x̄ - b ȳ` (and the row
   equivalent) and converted to Q16.
6. Optional outlier rejection: median residual × 4 (floor
   `outlier_threshold_q8`) for up to `outlier_passes`, then refit. The median is
   order-independent; ties keep all points.
7. `finalize_gm_params()` quantises, degrades and validates. A non-identity
   model is discarded to `IDENTITY` when its weighted correspondence SSE does
   not improve by `min_gain_percent` (default 1%) or when the scale,
   determinant, corner projection or optional shear guard fails.

`fit_*` returns the raw model; `estimate_*` returns the final model plus
diagnostics (`points_in/used`, identity/model SSE, gain dB, codability, shear
validity, degradation reason).

### Quantisation and degradation (§5.18.9)

- Translation `p[0..1]`: grid `2^(16-3) = 8192` (1/8 pel), range
  `[-16383·8192, +16383·8192]` ≈ ±2047.875 luma samples.
- Alpha `p[2..5]`: grid `2^(16-10) = 64` (1/1024), offsets from the identity
  centre are clamped to `±511·64 = ±32704` (≈ ±0.499).
- `force_integer_mv` quantises translations to whole pixels (1<<16) and keeps
  the syntax fractional bits zero; this mirrors the note in §5.18.9.2 where the
  decoder discards fractional translation bits in setup global MV.
- AFFINE whose `|p4+p3|` and `|p5-p2|` are both ≤ `rotzoom_mismatch_q16`
  (default 1/16) is degraded to ROTZOOM; otherwise an AFFINE that fails the
  scale/shear/corner guard is retried as its similarity projection and finally
  as IDENTITY.
- `ROTZOOM` with `p2=p5=65536, p3=p4=0` is the translation-only model. AV2 has
  no separate `TRANSLATION` GmType; this form is legal `ROTZOOM`.

### Validation

`gm_params_codable()` checks type/range/grid and ROTZOOM symmetry.
`validate_gm_params()` additionally checks `det > 0`, scale limits,
`setup_shear_valid()` (optional, `require_shear_valid`) and, when frame
geometry is supplied, that all four frame corners stay within
`max_corner_displacement_q8`. Note that a shear-invalid model is still
**codable**: the decoder uses the §7.13.3.20 extended-warp fallback. The gate
verifies that path.

## Projection and GLOBALMV

`get_warp_motion_vector()` transcribes §7.12.2.2:

```text
x = MiCol*4 + bw/2 - 1,  y = MiRow*4 + bh/2 - 1
xc = (p2-1)*x + p3*y + p0
yc = p4*x + (p5-1)*y + p1
mv = (Round2Signed(yc,16-bits), Round2Signed(xc,16-bits))
```

then `MV_LOW/MV_UPP`, §5.20.9.4/.5 `clamp_mv_row/col`, and
`lower_mv_precision()` for precision < 1/2 pel. `global_mv_for_block()` reports:

- `usable = GmType > IDENTITY`;
- `use_warp = min(block_w, block_h) >= 8` (the §7.13.3.14 `useWarp == 2`
  condition);
- the projected 1/8-pel centre MV (used for stack contexts and for blocks
  smaller than 8×8).

`GLOBALMV` syntax itself is M2-owned: `single_mode = 1` selects YMode 15, no
DRL index and no MVD. `GLOBAL_GLOBALMV` is 22 in compound streams. This
component supplies the model and the encoder-side prediction mode, not the
tile symbols.

## Syntax writer / reader

`write_global_motion_params()` emits exactly §5.18.9.1:

```text
use_global_motion f(1)
[ if use ] our_ref ns(NumTotalRefs+1)          (absent on SWITCH_FRAME)
[ if base ref has refs ] their_ref ns(RefNumTotalRefs)
for each ref:
  if dist == 0 or OrderHints[ref] == RESTRICTED_OH: IDENTITY, no bits
  else:
    is_global f(1)
    if is_global: is_rot_zoom f(1)              (ROTZOOM vs AFFINE)
    read/write_global_param(2,3[,4,5],0,1)
```

`scale_warp_model()` is the exact §5.18.9.1 base/their-ref predictor;
`write_global_param` is the inverse of `read_global_param` (signed subexp with
reference + inverse recenter). `GmRefHistory` holds `RefNumTotalRefs`,
`SavedOrderHints`, `SavedGmParams`. `our_ref = -1` means `NumTotalRefs` (no base
model), the common encoder choice. The reader round-trips every type and both
the no-base and saved-base paths in `gme_test`.

The sequence bit is one bit:

```text
enable_flex_mvres f(1)
enable_global_motion f(1)      <-- write_sequence_enable_global_motion()
enable_short_refresh_frame_flags f(1)
```

## Tests

```sh
./src/encoder/inter/gme/build_and_test.sh           # 21750 checks
./src/encoder/inter/gme/build_and_test.sh --gate    # + dav2d gate
SANITIZE=1 WERROR=1 ./src/encoder/inter/gme/build_and_test.sh
```

Coverage:

- sequence bit position;
- `read/write_global_param` at grid/range extremes and every index;
- full `global_motion_params` round-trip for ROTZOOM and AFFINE, use=0,
  intra/disabled early return, and the saved-base path;
- `scale_warp_model` equal/negative distance and `baseDistance=0`;
- projection, precision lowering, MV clamps;
- codability, range/grid rejection, ROTZOOM symmetry, shear-required profile;
- AFFINE→ROTZOOM and reflection→IDENTITY degradation;
- exact pan / zoom / rotation / affine fits, noisy 10% outliers, fits from
  warps and block MVs;
- input-order determinism;
- synthetic pan / zoom / rotation / combined frames through a real block
  matcher;
- real 720p consecutive pair and a real-720p frame warped by a known model.

Measured on the development machine (Apple clang 21, `-O2`):

| Case | points used | type | corr. gain | bilinear image SSE identity → model |
|---|---:|---|---:|---:|
| synthetic pan (9.5, −5.25 px) | 356 | ROTZOOM | 15.36 dB | 196,887,735 → 35,740 |
| synthetic zoom (1.06×) | 243 | ROTZOOM | 8.52 dB | 797,848,125 → 159,840 |
| synthetic rotation (2°) | 319 | ROTZOOM | 11.33 dB | 321,108,416 → 109,199 |
| synthetic pan+zoom+rotate | 320 | ROTZOOM | 13.11 dB | 561,095,106 → 270,723 |
| real 720p pair 0/1 | 727 | IDENTITY | candidate −6.25 dB | 282,051 → 282,051 |
| real 720p pair 0/10 | 738 | IDENTITY | candidate −8.93 dB | 14,969,651 → 14,969,651 |
| real 720p + known 1.02×/0.5°/(3,−2) px | 682 | ROTZOOM | 10.37 dB | 372,973,132 → 4,644,720 |

The real consecutive `shine` pair 0/1 is almost static (identity MSE ≈ 0.31 ⇒
≈53 dB); pair 0/10 has local motion but no coherent global model. In both cases
the estimator correctly rejects a noisy non-identity candidate. The known-warp
variant recovers `p0=180224` (2.75 px) and `p1=-139264` (−2.125 px) against the
injected (3, −2) px, with 10.37 dB image SSE gain. The bilinear SSE is a
**quality proxy**, not the normative predictor.

## Decoder gate

`gme_gate` is a self-contained two-frame stream builder. It does **not** edit
the M2 core:

1. Calls the existing `m2::encode_m2_sequence()` with
   `force_global_mv=true, force_single_block=true, enable_intra_fallback=false`
   to get a real intra keyframe and an all-`GLOBALMV` skip P tile.
2. Re-wraps the key OBU with a sequence header whose `enable_global_motion` bit
   is 1. The copy of the sequence writer is checked byte-identical against
   `m2::make_sequence_header_obu()` when the bit is 0.
3. Writes the P frame with `m2::write_frame_header()` followed by
   `write_global_motion_params()` at the §5.18.9 insertion point.
4. Reads the GM syntax back and reconstructs the P frame with
   `adv::warp_predict()` / `warp_predict_ext()` using the *decoded* parameters.
5. Runs the pinned `dav2d` (`f9dafbc`) and compares every Y/U/V sample.

Results (all with `dav2d -q --threads 1`):

| Gate case | GM type | gm bits | decoded mismatches |
|---|---|---:|---:|
| 256×256 synthetic ROTZOOM | ROTZOOM | 42 | 0 |
| 256×256 translation-only | ROTZOOM (alpha identity) | 32 | 0 |
| 256×256 AFFINE | AFFINE | 68 | 0 |
| 256×256 shear-invalid (extended warp) | AFFINE | 73 | 0 |
| 3-frame base-diff (P1 → P2) | ROTZOOM ×2 | 42 + 40 | 0 |
| 1280×704 real 720p crop | ROTZOOM | 42 | 0 |

All six streams also pass `tools/validate_dav2d.sh` (`OK: dav2d decoded ...`).
The base-diff stream exercises `our_ref`/`their_ref`, `SavedGmParams`,
`SavedOrderHints`, and `scale_warp_model` with `baseDistance=1, dist=1`; the
P2 header codes its model differentially against the P1 model.

The gate is a syntax/prediction gate, not an encoder integration: the P frame
is skip-only, prediction is generated from the key frame, and no residual or
RDO is involved. It proves the bit ordering, parameter coding, `GLOBALMV` path,
block warp and extended-warp fallback agree with the decoder.

## Integration call sites (not edited)

The following are the exact places the M2 core must change to enable this
component. Line numbers are from the current tree; use the surrounding symbol
names if they drift.

1. **`src/encoder/m2/m2_headers.h` (`M2Config`)**
   Add `bool enable_global_motion = false;` and per-reference
   `std::array<gme::GmParams, 7> gm_params;` plus the
   `gme::GmSyntaxContext` (or the history) needed for differential coding.

2. **`src/encoder/m2/m2_headers.cpp:181`**
   Replace `writer.put_bit(false); // enable_global_motion` with
   `gme::write_sequence_enable_global_motion(writer, config.enable_global_motion);`.

3. **`src/encoder/m2/m2_headers.cpp:295`**
   After `writer.put_bits(0U, 2U); // reduced_tx_set`, insert
   `gme::write_global_motion_params(writer, config.gm_context, &error)` for
   non-intra frames. This is the §5.18.9 position, before
   `film_grain_config()`.

4. **`src/encoder/m2/m2_frame.h` (`M2Options`) / `m2_frame.cpp`**
   Add GM enable/estimation options. Before the per-frame P encode, call
   `gme::estimate_gm_block_mvs()` (or `estimate_gm_warps`) per reference and
   `gme::make_gm_frame_plan()`. Store the accepted `GmParams` in the frame's
   `GmSyntaxContext`.

5. **`src/encoder/m2/m2_frame.cpp:591-605`**
   The `GLOBALMV` candidate currently uses `predict_luma({0,0}, ...)` and sets
   `mv={0,0}`. Replace it with:
   - if `GmType > IDENTITY` and the block is ≥8×8: predict with
     `adv::warp_predict()` / `warp_predict_ext()` using the accepted model;
   - otherwise: use `gme::global_mv_for_block()` and normal MC;
   - `entry.mv_y/mv_x` remain 0 for `GLOBALMV` (the decoder derives the MV), but
     the reconstruction must be the warped prediction, not the zero-MV copy.

6. **`src/encoder/m2/m2_syntax.cpp:1653-1654`**
   `params.global_mv[0/1]` is passed to `merge::build_stack_checked()`. For
   mixed `NEARMV`/`GLOBALMV` streams it must be the projected global MV for the
   reference, otherwise the encoder's merge stack diverges from the decoder's.
   `GLOBALMV` itself still codes no MVD; this is context/stack parity only.

7. **`src/encoder/m2/advanced/advanced_syntax.cpp:1171-1180`**
   `write_sequence_global_motion_bit()` currently rejects
   `enable_global_motion=1`. Replace its body with
   `gme::write_sequence_enable_global_motion(writer, config.enable_global_motion);`
   (or delete the helper once call site 2 owns the bit).

8. **`src/encoder/m2/post/tiled_frame.cpp:187` and
   `src/encoder/m2/multiref/gate/compound_stream.cpp:158`**
   These independent gate headers hard-code the GM bit to 0. They are correct
   as long as those gates do not use GM; if they gain a GM mode, they need the
   same one-line replacement. They are not on the M2 production path.

9. **`src/encoder/inter/mc/README.md:278`**
   The note that `get_mv()` returns zero `GlobalMvs` for `GLOBALMV` neighbours
   only holds while GM is disabled. After integration, the MC/merge layer must
   take the frame's `GmType`/projected `GlobalMvs`.

10. **`src/encoder/m2/advanced/README.md:343`**
    Replace the "Global motion estimation is not implemented" status line once
    the above is wired.

No changes are made here to `src/encoder/inter/advanced/`, M1/M2 core,
`src/main.cpp`, or `CMakeLists.txt`.

## Limitations

- Frame-level estimation only; no homography/projective model (AV2 syntax does
  not signal one).
- The fit operates on q8 luma correspondences. Scaled references are not
  resampled by this component; the caller must supply correspondences in the
  current-frame coordinate system and the model is applied by the existing
  scaled-reference MC path.
- Acceptance uses a correspondence-SSE gain heuristic, not a full RDO rate
  term. The caller should compare GLOBALMV vs NEARMV/NEWMV with the real
  entropy cost.
- No CUDA kernel. A future kernel can include `gme_core.h`; the fitting
  arithmetic is integer-only.
- The gate uses the existing M2 keyframe and `GLOBALMV` skip tile. It does not
  establish a bitrate or quality result for a real inter encoder.
- The syntax writer/reader supports all 7 reference-list entries, and the unit
  tests round-trip multiple per-reference models. The decoder gate is
  single-reference because the M2 core currently pins `NumTotalRefs = 1`; a
  multi-reference decoder gate needs an M2 tile writer that codes
  `reference_select`/compound refs.
