# Morphological prediction — `morph_pred` (§5.20.5.4, §7.13.3.26)

Complete component for the AV2 v1.0.0 morphological-prediction tool that the
tile/block audit (`docs/spec-audit/04-tile-block.md`) listed as **ABSENT**:

* the `morph_pred S()` block symbol and its `TileMorphPredCdf[ctx]` context;
* the §7.13.3.26 prediction process (BAWP with `morphPred = 1`);
* the §6.19.5.4 `is_offset_mv_valid(-1, -1)` allowed-mode check;
* a real `dav2d --strict 1 --threads 1` gate stream.

Nothing outside this directory was edited. The production call sites are
listed at the end; the shipped M1/M2 paths still cannot signal the tool.

Status: **46,823 checks pass with `--out-prefix` (46,819 tests-only).
Two gate streams decode bit-exactly with
`dav2d` f9dafbc at `--strict 1 --threads 1`, and the morph-on/off decoded
frames differ (so the decoder really runs the predictor).**

```sh
./src/encoder/inter/morph/build_and_test.sh            # tests + gate
SANITIZE=1 WERROR=1 ./src/encoder/inter/morph/build_and_test.sh
SKIP_GATE=1 ./src/encoder/inter/morph/build_and_test.sh
```

## Every normative occurrence of `morph` in the PDF

Pages are PDF page numbers (the audit uses the same numbering for these
sections).

| § | Page | Normative text | Implemented in |
|---|---|---|---|
| §5.4.6 | 267 | `enable_bawp f(1)` in `sequence_inter_config()`. Single-picture branch writes it; imports `enable_imp_msk_bld = 0`, `NumRefFrames = 2`, `long_term_frame_id_bits = 0`. | `morph_stream.cpp` sequence header; `MorphPredGate::enable_bawp` |
| §6.4.6 | 267–268 | `enable_bawp = 1` means `morph_pred` can be used for intra frames when `allow_screen_content_tools`; otherwise `morph_pred` is not used and inferred 0. NOTE: `allow_bawp` is inter-only; `morph_pred` is signalled only when `FrameIsIntra == 1 && allow_screen_content_tools == 1`. | `morph_pred_present()` / `morph_pred_allowed()` |
| §5.20.4.1 | 164 | `MorphPreds[r+y][c+x] = use_intrabc && morph_pred` for every 4x4 of a decoded block. | `MorphStreamResult::morph_ctx_value`, stream `morph_preds_` grid; production call site 2/3 |
| §5.20.5.4 | 172–173 | In `read_intrabc_info()`, after `assign_mv(0)`: `if (FrameIsIntra && allow_screen_content_tools && enable_bawp) morph_pred S(); else morph_pred = 0`. | `morph_syntax.h/.cpp`, `write_morph_pred`/`read_morph_pred` |
| §6.19.5.4 | 354–355 | `morph_pred = 1` selects brightness adaptation; conformance requires `is_offset_mv_valid(-1,-1) == 1`. | `morph_pred_allowed()` + stream `is_bv_offset_valid(..., -1, -1)` |
| §7.13.3.1 | 467–468 | In the general inter-block process, after prediction/BAWP: `if (plane == 0 && use_intrabc == 1 && morph_pred == 1)` invoke §7.13.3.26 with `x, y, w, h, Mvs[candRow][candCol][0]`. | Stream reconstruction order: IBC copy → morph adapt |
| §7.13.3.25 | 499–502 | BAWP process. The `morphPred` branches are: `prevFrame = CurrFrame`; `explicit_bawp && !morphPred`; `plane == 0 && !morphPred` for `BawpAlpha`; fit and full-block adaptation. | `morph_common.h` (`morph_derive`, `morph_apply_fit`) |
| §7.13.3.26 | 502–503 | Morphological prediction has no separate body: invoke §7.13.3.25 with `plane = 0`, `morphPred = 1`. | `morph_predict()` |
| §8.3.2 | 588–589 | `morph_pred` CDF is `TileMorphPredCdf[ctx]`; `ctx` is the sum of `MorphPreds[NPos[n]]` over the §5.20.4.1 `NPos` list (0..2). | `morph_symbol_context()` / `morph_neighbours()` |
| §9.3 + `init_non_coeff_cdfs` | 328–329, 738–739 | `Default_Morph_Pred_Cdf[3][2+1] = {{31715,55,0},{19667,1,0},{10555,91,0}}`; tile CDF init copies it. | `init_morph_cdfs()`, pinned in tests |

There is **no other normative occurrence**. In particular, the search found
none in §5.18.2 and none coupling morph to implicit mask blending.

## `enable_imp_msk_bld` interaction (correction)

The task brief mentions "§5.18.2 `enable_imp_msk_bld` interaction". In the
v1.0.0 PDF:

* §5.18.2 is *frame header info syntax* and contains no
  `enable_imp_msk_bld` bit.
* `enable_imp_msk_bld` is written in **§5.4.6** `sequence_inter_config()` and
  defined in **§6.4.6**. It controls only the implicit compound mask blend in
  §7.13.3.14 (`get_mask`), which is an **inter compound** path.
* In the `single_picture_header_flag == 1` branch used by every still stream
  in this tree, `enable_imp_msk_bld` is **inferred 0** and is not coded.
* It has **no effect** on `morph_pred` legality or prediction. `morph_pred` is
  an intra/IBC tool; it cannot be combined with compound prediction.

`MorphPredGate` carries the flag only so the tests can pin that independence.
`morph_pred_allowed()` never reads it.

## API

| File | Purpose |
|---|---|
| `morph_common.h` | §7.13.3.25/26 reference: `MorphContext`, `MorphRect`, `MorphPlane`, `MorphTemplatePlan`, `MorphFit`, `morph_reference_geometry()`, `morph_reference_in_bounds()`, `morph_template_plan()`, `morph_derive()`, `morph_apply_fit()`, `morph_predict()`. Also `to_fullmv`, `resolve_division`, `Div_Lut`. |
| `morph_syntax.h/.cpp` | `MorphCdfs`, `init_morph_cdfs()`, `morph_neighbours()`, `morph_symbol_context()`, `write_morph_pred()`, `read_morph_pred()`, `MorphPredGate`, `morph_pred_present()`, `morph_pred_allowed()`. |
| `morph_stream.h/.cpp` | Minimal still-picture writer (`encode_morph_stream`, `write_morph_ivf`, `write_morph_y4m`) used only for the decoder gate. |
| `morph_test.cpp` | 46,819 tests-only / 46,823 with `--out-prefix`; `--emit-goldens`; mode-selection benchmark. |
| `morph_test_goldens.h` | Pinned alpha/beta/count/FNV hash for all 22 IBC-legal block sizes × 4 `AvailU`/`AvailL` combinations (88 rows). |
| `build_and_test.sh` | Standalone C++20 build, tests, pinned MD5s and `dav2d --strict 1 --threads 1` recon compare. |

The predictor is luma/8-bit only, matching the wired profile. It adapts
`CurrFrame[0]` in place; the caller must have written the IBC prediction first.

## Syntax details

`morph_pred` is read only inside `read_intrabc_info()`:

```text
if (FrameIsIntra && allow_screen_content_tools && enable_bawp) morph_pred S()
else                                                           morph_pred = 0
```

CDF context (frozen CDFs are used by the gate streams):

```text
ctx = 0
for (n = 0; n < NNum; n++) ctx += MorphPreds[NPos[n][0]][NPos[n][1]]
```

`NPos` is the §5.20.4.1 non-buffered neighbour list: at most two entries, both
in the current superblock row. The writer keeps a per-4x4 `MorphPreds` grid and
updates it with `use_intrabc && morph_pred` after every block.

Allowed-mode validation is exact:

* syntax bits (`FrameIsIntra`, `allow_screen_content_tools`, `enable_bawp`);
* `use_intrabc == 1`;
* `is_offset_mv_valid(-1, -1) == 1` (§6.19.5.4), always checked even when the
  stream writer's optional `strict_border` flag is off;
* §7.13.3.25 reference box: `refX >= 1`, `refY >= 1`,
  `refX + bw <= planeWidth`, `refY + bh <= planeHeight`.

## Prediction

`morph_predict()` is a literal transcription of §7.13.3.25 with
`morphPred = 1`:

* `prevFrame = CurrFrame`; `refX/refY` from `MiCol/MiRow` and
  `to_fullmv(mv)`; `bw/bh` clipped to the luma plane.
* Template plan: `bw2 = Min(16,bw)`, `bh2 = Min(16,bh)`,
  `width = bw2 == 12 ? 8 : bw2`, same for height; the four
  `AvailU`/`AvailL` branches choose `numUp/numLeft`; steps are
  `width/numUp` and `height/numLeft`.
* Least-squares fit over `CurrFrame` above/left vs the current-frame source
  block at `(refX-1, refY-1)`, then `resolve_division(nor, der, 8)`.
* `beta = ((sumY << 8) - sumX * alpha) / count`; `count == 0` leaves
  `alpha = 256`, `beta = -128`.
* The final loop adapts all `w*h` samples, not only the in-frame part:
  `Clip1((orig * alpha + beta) >> 8)`.

The differential test suite also compares every case against an independent
64-bit literal transcription of the same process.

## Gate

The gate writer emits an 8-bit 4:2:0, one-frame, one-tile still:

* sequence header: `enable_bawp = 1` (the only sequence bit that makes
  `morph_pred` signallable); `enable_imp_msk_bld` is inferred 0 by the
  single-picture branch;
* frame header: `allow_screen_content_tools = 1`, `force_integer_mv = 1`,
  `allow_intrabc = 1`, global only;
* 512×128, 64×64 SBs, HORZ to 64×32 leaves;
* row 0 is `SMOOTH_PRED` (no residual) so the reconstructed source region is
  non-flat; row 1 from column 2 uses IBC with a fixed legal BV and a mix of
  `morph_pred = 0/1`;
* the reconstruction order is IBC copy → morphological prediction.

Pinned streams (both `dav2d --strict 1 --threads 1` exact):

| stream | bytes | MD5 |
|---|---:|---|
| `morph_gate.ivf` | 99 | `b9246821ff00a2948011af36d0737b4e` |
| `morph_gate.off.ivf` | 95 | `6a633e910baa7fa070a6eb23c1fc1b3a` |

The gate's `morph_pred=1` blocks all use a **fitted** (not identity) alpha:
8 blocks, alpha range −146..256. The on/off decoded luma differs at 9,792 of
65,536 samples (max absolute difference 1, mean 0.149), which proves the
decoder is not treating `morph_pred` as 0. The build script fails if the two
decoded Y4Ms are identical.

## Tests

`morph_test` covers:

* §7.13.3.26 prediction for **all 22 IBC-legal block sizes** (4×4 through
  64×32/32×64/16×64/64×16/8×64/64×8; 64×64 is excluded by the IBC syntax)
  × **all four `AvailU`/`AvailL` combinations** (88 cases), with pinned
  alpha/beta/count and FNV-1a hash per case;
* an independent literal fit/applier, all sample comparisons;
* zero-template defaults (`alpha=256`, `beta=-128`, “subtract 1”);
* reference-box rejection (`refX<1`, `refY<1`, overflow, `avail_u` at `y=0`);
* `to_fullmv`/`resolve_division`/`Div_Lut` pins;
* CDF defaults, writer/reader round-trips for every ctx and value with frozen
  and adaptive CDFs, invalid ctx rejection;
* exhaustive `NPos` derivation for every legal size at representative
  positions against a literal `add_neighbor()` implementation;
* `morph_pred_present`/`morph_pred_allowed` truth tables, including
  `enable_imp_msk_bld` independence;
* stream construction, all three contexts, mixed 0/1 symbols, invalid
  `-1,-1` border rejection with `strict_border = 0`.

## Mode-selection impact (quality / rate)

**Prediction-only** on the 256×256 `testdata/shine_256x256_1f.y4m` Y plane:
32 candidate 64×32 blocks in raster order; each block copies the block one
64-wide column to its left (first column wraps to the previous row); 20
candidates have a valid `refX/refY>=1` source. The source frame is used as the
reconstructed context, so this isolates the brightness fit.

| metric | copy (morph_pred=0) | RDO pick (SSE) |
|---|---:|---:|
| selected blocks | 0 | 9 / 20 (45.0%) |
| PSNR vs source | 15.534 dB | 18.683 dB |
| gain | — | **+3.150 dB** |
| ideal flag cost | — | **+80.7 bits / 20 blocks** (≈4.0 bits/block) |

Flag cost at the default §9.3 CDFs for the gate's 12 IBC symbols: **77.2
bits** on, **10.8 bits** all-zero, **+66.4 bits incremental** (≈5.5 bits per
symbol at these contexts). The actual IVF delta is 4 bytes (99 vs 95) because
the arithmetic coder's final-code alignment dominates a 12-symbol tile; the
ideal entropy is the meaningful rate figure. A production RDO decision must
therefore be `SSE_morph + λ·R_flag < SSE_copy`, not SSE-only; the fit is
cheap (≤32 template samples) but the ctx-2 flag is expensive under the default
CDF.

This is a predictor-level measurement, not a coded-stream BD-rate: there is no
residual coding, in-loop filtering, or production IBC mode decision in the
gate writer.

## Production call sites (not edited)

Exact snippets: `INTEGRATION.md`.

1. **`src/encoder/m1/m1_headers.h/.cpp`**
   * add `bool enable_bawp` (and screen-content/intrabc fields) to `M1Config`;
   * `write_sequence_header()` currently hard-codes `writer.put_bit(false);
     // enable_bawp` (`m1_headers.cpp:127`); write the new field instead;
   * `write_frame_header_syntax()` hard-codes `allow_screen_content_tools = 0`
     and `allow_intrabc = 0` (`m1_headers.cpp:170-171`); replace with the same
     screen/intrabc bits the gate writer emits.
2. **`src/encoder/m1/ibc_stream/m1_ibc_stream.cpp`** (the existing IBC adapter;
   its README explicitly lists `morph_pred` as out of scope):
   * after `write_intrabc_info()` and before the residual, write
     `morph_pred` with `morph_symbol_context()` over a per-4x4
     `MorphPreds` grid;
   * after `ibc::copy_block()`, call `morph_predict()` when the symbol is 1;
   * update `MorphPreds` for the whole block;
   * validate with `morph_pred_allowed()` (including
     `is_bv_offset_valid(..., -1, -1)`).
3. **`src/encoder/m1/m1_syntax.cpp` / `m1_still.cpp`** (production path):
   * the stock writer has no IBC mode-info branch, so morph cannot be reached
     there until the IBC adapter is merged;
   * when it is, mirror the same two edits and apply the predictor after the
     IBC copy; do not apply it to chroma.
4. **`src/main.cpp`** — expose an opt-in flag (e.g. `--m1-ibc-morph`) only
   after 1–3; no CLI change is needed for this component's tests.
5. **`src/encoder/ibc/README.md:70,198`** and
   `docs/spec-audit/04-tile-block.md:111` are now stale: `morph_pred` exists
   as a tested component with a decoder gate; it is still not wired into the
   shipped M1 path.

## Limits

* 8-bit 4:2:0, luma only (the spec's morph process is plane 0).
* The gate writer is a minimal still writer, not the production M1 encoder.
* `morph_pred` is never signalled by the shipped CLI or M2 core; the two edits
  above and an IBC mode-info path are required.
* The gate's SMOOTH row-0 blocks are a test scaffold for a non-flat
  reconstruction; a production encoder gets non-flat contexts from real
  residual coding.
* No CUDA twin is added here. The arithmetic is integer and the same header is
  portable to a future kernel.
