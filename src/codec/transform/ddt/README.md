# M2 inter DDT (Data Dependent Transform)

Status: **component + still/inter gates; wired into the CPU `--video` writer as
`--video-ddt`; still rejected.** DDT/DDTX/FDDT is inter-only: §5.4.8 omits
`enable_inter_ddt` for a single-picture header and §7.15.4.1 substitutes the
kernel only when `is_inter = 1`. There is no frame or block DDT syntax. The
completion-wave still path therefore rejects `--still-ddt` by name
(`src/codec/transform/ddt/still_ddt_gate.sh` checks the diagnostic and the
pinned still streams). The real proof is the video product gate
(`src/pipeline/video/gate_video_sdp_ddt.sh`) plus the component gate
`gate/build_gate.sh`: a 3-frame M2 inter stream with `enable_inter_ddt = 1`,
decoded by the pinned `dav2d`, with DDTX and FDDT reconstruction
sample-for-sample exact. The completion wave also updated the gate's source
list for the wave-2/3 inter modules (`drl_reorder`, `mv_precision`) so it still
links on the current tree.

Standalone, spec-only implementation of AV2 `enable_inter_ddt` (DDTX / FDDT)
for the M2 inter path.  Nothing outside this directory is modified, wired into
CMake, or committed.

Specification: `spec/AV2_Spec_v1.0.0.pdf` (pinned SHA-256
`e9916f09…b533f`).  Section numbers below are AV2 v1.0.0 section numbers; the
PDF page is the physical page of the pinned PDF.

## 1. What DDT is (normative findings)

Every normative occurrence of "DDT" in the pinned PDF:

| Location | Content |
|---|---|
| §2, PDF p. 19 | Glossary: "DDT — Data Dependent Transform." |
| §5.4.8, PDF p. 63 (printed p. 64) | `enable_inter_ddt f(1)`, present iff `single_picture_header_flag == 0`, between `enable_chroma_dctonly` and `reduced_tx_part_set`. |
| §6.4.8, PDF p. 269 | "`enable_inter_ddt` equal to 1 specifies that DDT is allowed for inter blocks in the coded video sequence. `enable_inter_ddt` equal to 0 specifies that DDT is not allowed for inter blocks." |
| §7.15.2.1, PDF pp. 519–520 | 1D inverse transform: `Ddtx_Kernel8/16` branches and the `FDDT` reverse-sample branch. |
| §7.15.4.1, PDF pp. 525–526 | `get_transform_1d_type()`: `useDdt = enable_inter_ddt && !use_intrabc && is_inter`; ADST→DDTX, FDST→FDDT when `sz != 4`. |
| §9.6, PDF p. 848 | `Ddtx_Kernel8[8][8]`, `Ddtx_Kernel16[16][16]` numeric tables. |
| Table 7.1, PDF p. 526 | 1D type values: 0 DCT, 1 IDT, 2 ADST, 3 FDST, **4 DDTX, 5 FDDT**. |

The transform substitution is exactly (§7.15.4.1):

```c
useDdt = enable_inter_ddt && !use_intrabc && is_inter
t = Transform_1d_Type[PlaneTxType][dir]      // dir 0 = row/width, 1 = col/height
if (useDdt && (t == ADST || t == FDST) && sz != 4)
    return (t == ADST) ? DDTX : FDDT
return t
```

Consequences, all verified against the pinned text:

* **No per-block or per-frame syntax.**  DDT is derived from the ordinary
  `inter_tx_type` / `inter_tx_type_offset` symbols (§5.20.8.2).  A block whose
  signalled 2D type has an ADST/FDST component and whose adjusted 1D length is
  8 or 16 is reconstructed with the DDT kernel instead.
* **`sz != 4`** excludes 4-length transforms.  `sz == 32` passes the predicate
  and produces DDTX/FDDT, but §7.15.2.1's `sz == 32` branch uses
  `Dct_Kernel32` for *every* type, so DDTX/FDDT alias DCT there and do not
  change the reconstruction.
* **Only lengths 8 and 16 matter**, because that is where `Ddtx_Kernel8/16`
  exist and where the non-DDT ADST/FDST kernels differ.
* **FDDT is DDTX with the sample axis reversed**: the inverse branch is
  `Ddtx_KernelN[j][N-1-i]`.  In matrix terms the forward uses `K[r][N-1-i]`
  and the inverse uses `K[j][N-1-i]`.
* **Planes:** luma and chroma are both subject.  For chroma,
  `compute_tx_type()` (§5.20.7.29) first applies `enable_chroma_dctonly` and
  the chroma TX set; if the result is ADST/FDST at an 8/16 length, DDT applies
  to chroma too.
* **IntraBC:** `use_intrabc = 1` blocks have `is_inter = 1` (§5.20.5.2), so the
  explicit `!use_intrabc` guard matters.  DDT is off for IBC.
* **Only the inverse is normative.**  AV2 is a decoder specification; there is
  no encoder forward process.  This component's forward is an integer-exact
  transpose of the normative kernel pair (see §4).

## 2. Legal block sizes / types

DDT changes a block iff the signalled luma/chroma `TxType` is in
`Tx_Type_In_Set_Inter` (§5.20.7.29) for the size, `enable_inter_ddt = 1`,
`reduced_tx_set = 0` for luma (all reduced inter sets are DCT/IDT only),
`enable_chroma_dctonly = 0` for chroma, and at least one ADST/FDST component is
at adjusted length 8 or 16.

The 100 legal `(tx_size, tx_type)` pairs (`d` = that direction gets DDT,
`D` = adjusted length 32, so the predicate derives DDTX/FDDT but the kernel is
DCT and nothing changes):

| TX size | set | legal types | DDT-active types |
|---|---|---|---|
| 4x4 | INTER_1 | 0..15 | — |
| 8x8 | INTER_1 | 0..15 | 1..8, 12..15 |
| 16x16 | INTER_2 | 0..11 | 1..8 |
| 32x32 | DCT_IDTX | 0, 9 | — |
| 64x64 | DCTONLY | 0 | — |
| 4x8 | INTER_1 | 0..15 | 1, 3, 4, 6, 7, 8, 12, 14 (col 8) |
| 8x4 | INTER_1 | 0..15 | 2, 3, 5, 6, 7, 8, 13, 15 (row 8) |
| 8x16 | INTER_1 | 0..15 | 1..8, 12..15 |
| 16x8 | INTER_1 | 0..15 | 1..8, 12..15 |
| 4x16 | INTER_1 | 0..15 | 1, 3, 4, 6, 7, 8, 12, 14 (col 16) |
| 16x4 | INTER_1 | 0..15 | 2, 3, 5, 6, 7, 8, 13, 15 (row 16) |
| 16x32 | HIGH_32 | 0, 2, 5, 9, 10, 11, 13, 15 | 2, 5, 13, 15 (row 16) |
| 32x16 | WIDE_32 | 0, 1, 4, 9, 10, 11, 12, 14 | 1, 4, 12, 14 (col 16) |
| 8x32 | HIGH_32 | as 16x32 | 2, 5, 13, 15 (row 8) |
| 32x8 | WIDE_32 | as 32x16 | 1, 4, 12, 14 (col 8) |
| 16x64 | HIGH_64 | 0, 2, 5, 10 | 2, 5 (row 16) |
| 64x16 | WIDE_64 | 0, 1, 4, 11 | 1, 4 (col 16) |
| 8x64 | HIGH_64 | 0, 2, 5, 10 | 2, 5 (row 8) |
| 64x8 | WIDE_64 | 0, 1, 4, 11 | 1, 4 (col 8) |
| 4x32, 32x4, 4x64, 64x4 | HIGH/WIDE_* | — | — (only length-4/32 components) |
| 32x64, 64x32, 64x64 | DCTONLY | 0 | — |

Notes:
* The WIDE_64/HIGH_64 inversion table is only reachable with
  `is_long_side_dct = 1` (the symbol is not coded for 64 sets), which is why
  the legal set is `{0,1,4,11}` / `{0,2,5,10}` rather than the full
  `Tx_Type_In_Set_Inter` row union.
* `16x16` uses `TX_SET_INTER_2`, not `TX_SET_INTER_1`.
* `d`/`D` at length 32 is included in `Plan::any_ddt()` (it is what
  §7.15.4.1 derives) but excluded from `plan_has_ddt()` and from the
  "DDT-active" column because §7.15.2.1 aliases the kernel to DCT32.

`ddt_constraints.cpp` implements this table from the spec rather than calling
`m2/residual/inter_tx_type.cpp`, because that sibling has an under-permissive
`INTER_2` row (see §7).

## 3. API

```cpp
#include "encoder/m2/ddt/ddt.h"

av2enc::video::ddt::Context ctx;
ctx.enable_inter_ddt = true;
ctx.is_inter = true;
ctx.use_intrabc = false;
ctx.bit_depth = 8;           // 8 or 10

// Segmented to the row/column directions for this TxParams.
av2enc::video::ddt::Plan plan = make_plan(params, ctx);
bool changes = plan_has_ddt(params, ctx);

// Normative decoder process (§7.15.2.1 / §7.15.4.1 / §7.15.4).
ddt::inverse_block(dequant_adjusted, params, ctx, residual_coded);
ddt::inverse_1d(src, n, shift, col_tx, type, bit_depth, out);

// Encoder forward (not in the spec; see §4) and its dequant-scale level map.
ddt::forward_block(residual_adjusted, params, ctx, coeff);
int32_t level = ddt::level_from_coeff(coeff, params);  // round(coeff*dq/66)
double qf = ddt::quant_factor(params);                 // dq/66

// Constraint validation over all tools.
ddt::BlockSignal signal{...};
ddt::Validation v = ddt::validate(signal);
// v.status, v.ddt_active, v.plan, v.tcq_eligible, v.qm_eligible, v.ist_eligible
```

`inverse_block()` is a drop-in for `txv::inverse_block()` for all 25 sizes and
16 TX types; it is byte-identical to it when DDT is off (tested).

### Tool interactions

* **TCQ** (§7.14.4): eligibility is `allow_tcq && plane==0 && !Lossless &&
  get_tx_class(PlaneTxType) == TX_CLASS_2D`.  DDT does not change
  `PlaneTxType`, so TCQ + DDT coexist exactly as TCQ + ADST/FDST did.  The
  dequant `shift += 1` is outside this component.
* **Parity hiding**: `plane==0 && PlaneTxType != IDTX`; DDT types are never
  IDTX, so they can carry parity hiding.  No transform interaction.
* **IST** (§7.15.3): inter IST is only signalled when `TxType == DCT_DCT`, so
  a DDT block can never have `sec_tx_type != 0`.  `validate()` returns
  `kIstConflict` for that (illegal) combination.  Intra IST is irrelevant
  (`is_inter = 0`).
* **QM** (§7.14.4): `useQm` requires `PlaneTxType < IDTX` and keys the matrix
  by `txSz`, not by 1D type.  DDT leaves QM indexing untouched.  Note that
  1D-class types (`V_ADST`, `H_ADST`, 12..15) have `TxType >= IDTX` and are
  therefore never QM-eligible even though they can carry DDT.
* **Lossless / FSC**: `compute_tx_type()` returns IDTX/DCT_DCT, never
  ADST/FDST, so DDT never applies; `validate()` rejects the pairing.
* **DPCM**: `use_dpcm_y`/`use_dpcm_uv` cumulative sums are applied to
  `Residual` *after* §7.15.4, so they are orthogonal to the 1D kernel choice.
* **`enable_idtx_intra` / `enable_fsc`**: neither gates DDT.  Identity inter
  types remain legal when `enable_inter_ddt = 0`; DDT only rewrites ADST/FDST.

## 4. Forward transform (encoder side)

The specification defines only the decoder inverse.  `forward_*` is an
integer-exact transpose of the normative kernel pair:

* kernel `K` with `K K^T = (64^2 n) I` (the DDT tables share the DCT/ADST
  normalisation; row norms are ≈ `64√n`);
* forward shift `f = 12 + log2(n) - Transform_Shift[txSz][dir]`;
* FDDT uses `K[r][n-1-i]` (mirrored sample axis), DDTX uses `K[r][i]`;
* the odd-log2 rectangular prescale is inverted with the spec's own
  `2896/4096` rational;
* the row pass carries 8 fractional bits into the column pass so 64-coded
  sizes (small `Transform_Shift`, large column gain) keep their precision.

Output coefficients are in the decoder's dequant scale: with the shared
`QUANT_TABLE_BITS = 3` dequant step, `level = round(coeff * dq_denom / 66)`.
That matches the existing `528/8` quantiser; `quant_factor()` returns
`dq_denom/66`.

No floating point is used by either direction.  All accumulation is `int64`,
all rounding is §4.8 `Round2` (floor for negatives), and every entry point is
deterministic.  The inverse's own integer rounding is not losslessly
invertible; the tests assert 1D `|error| <= 20` for every `Transform_Shift`
and 2D `max |error| <= 96`, `mean <= 24` over all 100 legal pairs on residuals
in `[-120, 120]`.

## 5. Syntax and bit cost

`ddt_syntax.h` writes/reads the one element:

```cpp
write_enable_inter_ddt(writer, single_picture_header_flag, value);
read_enable_inter_ddt(bytes, bit_offset, single_picture_header_flag,
                      &value, &next_bit);
```

* **Syntax cost:** exactly **1 bit** per sequence header when
  `single_picture_header_flag = 0`; **0 bits** for a single-picture header
  (M1).  There is no frame or block cost and no CDF.
* **Byte cost:** the OBU payload is byte-aligned, so the extra bit can add 0
  or 1 payload byte depending on the total header length.
  `payload_bytes_added(total_syntax_bits, ...)` computes this.  In the gate's
  256x256 M2 sequence header the bit is at syntax bit 120; the syntax grows
  from 144 to 145 bits and the payload stays at 19 bytes (0 extra bytes).
* **Pinned goldens:** `0` / `1` (MSB-first); round-trip tested at every bit
  offset and for the absent-bit case.

`enable_inter_ddt` is read once per sequence header.  It is not an
`allow_ddt` per frame; the frame header has no DDT field.

## 6. M1 / M2 call sites

### M2 (the only path that can use DDT)

| Site | Current state | Required change |
|---|---|---|
| `src/encoder/m2/video_headers.cpp:195` | `writer.put_bit(false); // enable_inter_ddt` | write the configured value; add a field to `VideoConfig`. |
| `src/encoder/m2/residual/inter_residual.h:76` | `DecisionOptions.enable_inter_ddt` exists but no reader | thread it into `evaluate_plane()`. |
| `src/encoder/m2/residual/inter_residual.cpp:280` | `txv::forward_transform(...)` | `ddt::forward_block(...)` + `level_from_coeff()` (or `quant_factor()`), otherwise the analysis uses ADST/FDST while the decoder uses DDT. |
| `src/encoder/m2/residual/inter_residual.cpp:338` | `txv::inverse_transform(...)` | `ddt::inverse_block(...)` with the block `Context`; this is the one call the gate patches. |
| `src/encoder/m2/residual/inter_residual.cpp:473` | RDOQ `quant_step = 1/params.quant_factor` | with the DDT forward, use `1/ddt::quant_factor(params)` (= `66/dq_denom`). |
| `src/encoder/m2/residual/inter_residual.cpp:531` | passes `options.enable_inter_ddt` to `inter_tx_candidates()` | the candidate filter's meaning is wrong (see §7); DDT does not remove identity types.  Either fix `inter_tx_type.cpp` or stop filtering on this flag. |
| `src/encoder/m2/residual/inter_coeff_syntax.cpp:266` | passes `enable_inter_ddt=false` to `inter_tx_symbol()` | the symbol mapping does not depend on DDT; the flag only drives the same incorrect filter. |
| `src/encoder/m2/rdo/residual_decision.cpp:19` | forces `enable_inter_ddt=false` | set it from the sequence/frame config. |
| `src/encoder/m2/post/tiled_frame.cpp:200`, `src/encoder/m2/multiref/gate/compound_stream.cpp:167`, `src/encoder/inter/gme/gme_gate.cpp:258` | hard-coded `put_bit(false)` in other header writers | if those streams ever enable DDT, write the same bit. |
| `src/encoder/m2/video_headers.cpp:295` | `reduced_tx_set = 0` | DDT requires 0 for luma; already satisfied. |

The **decoder** needs no change: the normative process is §7.15.2.1 and the
pinned dav2d binary already implements it (it decodes the gate streams
exactly).  The repository itself contains no decoder.

### M1 (still pictures)

`single_picture_header_flag = 1` removes the bit (§5.4.8), and a still picture
has no inter blocks, so DDT is unreachable in M1.  There is no M1 call site and
no M1 stream can be affected.

## 7. Upstream findings (not changed; reported for the integrator)

1. **`inter_tx_type_is_ddt()` is misnamed / wrong.**  It returns true for
   `IDTX`, `V_DCT`, `H_DCT`, `V_ADST`, `H_ADST`, `V_FLIPADST`, `H_FLIPADST`
   (`inter_tx_type.cpp:228`).  DDT is the ADST/FDST → DDTX/FDDT substitution,
   *not* an identity transform.  `enable_inter_ddt = 0` does **not** make
   identity inter types illegal; it only means the decoder keeps ADST/FDST.
   The current filter therefore withholds legal `IDTX`-family candidates when
   DDT is off (a compression loss, not an invalid stream).
2. **`kInSetInter[6]` (INTER_2) is under-permissive.**  The spec
   `Tx_Type_In_Set_Inter` row for `TX_SET_INTER_2` is
   `{1×12, 0×4}` (types 0..11); `inter_tx_type.cpp` has only the first 8 set.
   `DDT` legality in this component uses the spec table, so 16x16 types 8..11
   (including `ADST_ADST`) are DDT-active here even though the current M2
   candidate enumerator cannot signal them.
3. **`txv::forward_block()` has no DDT hook.**  The shared transform component
   only knows `Tx1dType` 0..3.  This component deliberately does not modify
   it; the M2 integration should call `ddt::forward_block()` for blocks where
   `plan_has_ddt()` is true and keep `txv` otherwise (or extend `txv` with the
   two new 1D types).
4. **`txv` identity forward is not scale-correct for mixed identity/DDT
   blocks** (it copies the residual while the decoder multiplies by
   `get_identity_scale`).  `ddt::forward_block()` compensates the identity
   direction, which is why it is not a drop-in for `txv::forward_block()` on
   IDTX blocks (the inverse is a drop-in).

## 8. Tests

`build_and_test.sh` builds a standalone CPU test (no CUDA, no CMake):

* pinned `Ddtx_Kernel8/16` values and absolute-sum, row-norm bounds;
* `get_transform_1d_type()` for all flag/length combinations, including the
  `sz == 32` DCT alias and the `sz == 4` exclusion;
* 1D DDTX/FDDT goldens, reverse-symmetry, 8/10-bit clipping, `Round2` floor,
  and an independent naive int64 reference;
* forward/inverse 1D and 2D round-trips over all shifts and all 100 legal
  `(tx_size, tx_type)` pairs, with real-image residuals (see below);
* `inverse_block()` **byte-identical** to `txv::inverse_block()` for every
  non-DDT size/type combination (25 × 16);
* the full constraint matrix (sizes, types, planes, `use_intrabc`, lossless,
  FSC, `enable_chroma_dctonly`, `reduced_tx_set`, IST conflict, TCQ/QM
  eligibility);
* a real 16x16 luma block from `testdata/shine_256x256_1f.y4m` (pixel-128
  residual, ADST_ADST, 528-step quantiser): pinned levels + pinned source
  residual, max/mean/SSE bounds, determinism, and a check that the DDT
  reconstruction differs from ADST_ADST;
* syntax goldens (`0x00` / `0x80`), round-trips at all bit offsets,
  absent-bit inference, and `payload_bytes_added()`;
* a forward-scale compatibility test against
  `txv::forward_transform()`/`quantize()`/`dequantize()` for every non-DDT,
  non-IDTX block (the DDT forward is intentionally a different transform in
  DDT directions).

Sanitizers: `SANITIZE=1 ./build_and_test.sh` (ASan + UBSan, clean).

## 9. Decoder gate

`gate/build_gate.sh` compiles the existing M2 inter-residual gate driver
unchanged and links **build-directory copies** of:

* `video_syntax.cpp` (patched so non-DCT inter TX types are signalled),
* `video_headers.cpp` (the §5.4.8 bit reads `DDT_GATE_ENABLE`),
* `inter_residual.cpp` (the reconstruction inverse calls
  `ddt::inverse_block()`; one call site).

The core files are never edited.  The gate:

1. builds a deterministic 3-frame 256x256 motion clip from
   `testdata/shine_256x256_1f.y4m`;
2. encodes it with the flag 0 and flag 1 (DDTX pass: forced TX type 1
   `ADST_DCT`/DCT_ADST; FDDT pass: forced type 4 `FLIPADST_DCT`/DCT_FLIPADST);
3. finds the single differing bit (bit 504 in the gate stream; sequence
   syntax bit 120), asserts it is inside the sequence-header OBU, and splices
   it into the flag-0 stream;
4. asserts the spliced stream is **byte-identical** to the flag-1 stream;
5. decodes with the pinned `tools/bin/dav2d-macos` and requires the decoder
   reconstruction to match the DDT encoder reconstruction sample-for-sample;
6. asserts DDT actually ran (per pass: 62 column-DDTX, 45 row-DDTX, 62
   column-FDDT, 45 row-FDDT directions);
7. negative control: the unspliced flag-0 stream must fail the same comparison
   (the decoder then uses ADST/FDST while the encoder used DDT).

Result on this tree:

```
DDTX      (type 1, column): 62 blocks / 62 directions, decode exact
DDTX_ROW  (type 2, row):    45 blocks / 45 directions, decode exact
FDDT      (type 4, column): 62 blocks / 62 directions, decode exact
FDDT_ROW  (type 5, row):    45 blocks / 45 directions, decode exact
DDT GATE PASS
```

Run:

```sh
./src/encoder/m2/ddt/build_and_test.sh            # unit tests
./src/encoder/m2/ddt/build_and_test.sh --gate     # unit + decoder gate
SANITIZE=1 ./src/encoder/m2/ddt/build_and_test.sh
```

## 10. Product integration: `--video-ddt`

The CPU `--video` writer now signals `enable_inter_ddt` and applies the
§7.15.4.1 kernels:

* `VideoConfig::enable_inter_ddt` writes the §5.4.8 bit (non-single-picture
  header).  Off keeps it 0 and every historical stream byte-identical.
* `VideoOptions::enable_inter_ddt` (`--video-ddt`) drives a per-block inter
  transform-set selection.  Inter luma residual leaves run the existing luma
  TX search (`video_tx_search.cpp`) even when `--video-tx-search` is off:
  `Options::search_partition` stays false (TX_MODE_LARGEST codes no partition
  syntax) while the TX-type candidates run with the DDT-aware forward /
  inverse pair.  `--video-tx-search` composes and searches the partition too.
* `video_tx_search.cpp::evaluate_candidate()` uses
  `ddt::forward_block()`/`ddt::inverse_block()` exactly when
  `ddt::plan_has_ddt()` is true; otherwise it keeps the historical `txv` path,
  so non-DDT candidates and every `enable_inter_ddt = 0` stream are unchanged.
  `ddt::quant_factor()` replaces the scalar factor for those leaves.
* The encoder no longer runs the old `inter_tx_type_is_ddt()` candidate
  filter (README §7.1): DDT changes kernels, not the legal TX-type set.
* Coefficient coding is unchanged: the signalled type is still the ordinary
  §5.20.8.2 `inter_tx_type` symbol, and `inter_tx_symbol()` accepts every legal
  type.  Chroma is DCT-only (`enable_chroma_dctonly = 1`), so no chroma DDT.

Evidence: `src/pipeline/video/gate_video_sdp_ddt.sh` encodes 64x64 and 640x360
two-frame streams, decodes with `dav2d --strict 1`, requires the exact picture
count and sample-exact `--video-recon`, and (via `--video-trace`) requires at
least one *coded* DDT leaf for the `--video-inter-residual` /
`--video-tx-search` cases.  It also composes `--video-sdp` with `--video-ddt`.
Local result: 63 coded DDT leaves at 640x360/QP105 and 115 with the TX search,
all sample-exact.

Constraint list: CPU-only 8-bit 4:2:0, not with `--video-lossless`, `--video-tip`
or `--svc-layers` (those header writers do not signal the bit).

## 11. Specification gaps

Nothing required for the decoder was missing.  Two points are worth recording:

1. **No forward transform.**  AV2 defines only the inverse (§7.15); the
   forward here is an encoder-side integer transpose (documented in §4), not a
   normative process.
2. **No explicit inference for the omitted bit on single-picture headers.**
   §5.4.8 omits `enable_inter_ddt` when `single_picture_header_flag = 1` and
   §6.4.8 does not state an inferred value.  A still picture has no inter
   blocks, so DDT is unreachable; this component treats the absent bit as 0
   (and writes nothing).
