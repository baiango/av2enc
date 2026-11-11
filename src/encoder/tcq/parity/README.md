# AV2 parity hiding (`src/encoder/tcq/parity/`)

Status: **component only.** New directory; no existing file is modified, nothing
is wired into `CMakeLists.txt`, `src/main.cpp`, the M1/M2 core, the entropy
writer or the bitstream writer, and nothing is committed.

This component implements AV2 v1.0.0 parity hiding end to end at the coefficient
level:

* the §5.4.8 / §5.18.2 flag interactions and every exclusion (the actual
  sequence/frame derivation is reused read-only from `../tcq_flags.h`; the
  tests check that hand-off);
* the exact §5.20.7.27 two-pass coefficient syntax, including
  `coeff_base_ph`, `sumAbs1`, `isHidden`, the `maxLevel = NUM_BASE_LEVELS + 1`
  DC override, `Quant = 2 * Quant + sumAbs1`, and the
  `TileDcSignCdf[ptype][isHidden][ctx]` axis;
* the §5.20.7.28 `read_quant` escape with `lvlShift = 1` for the hidden DC;
* an encoder quantiser that chooses levels so the hidden parity is signalled
  correctly (mandatory when the frame allows it and `numNz >= PHTHRESH`);
* a normative decoder inverse and a byte-level §8.2 entropy round-trip;
* pinned syntax/entropy goldens, deterministic random tests over all 25 TX
  sizes, all three scan classes and all three planes, and a real-coefficient
  RD comparison against scalar and RDOQ.

The rest of `src/encoder/tcq/` (TCQ, CCTX, header flags) is reused read-only
where it overlaps. Parity hiding and TCQ are mutually exclusive by the spec:
`allow_parity_hiding` is 0 whenever `allow_tcq` is 1.

## Normative vs encoder design

The spec defines the decoder. Everything in `derive_syntax()`,
`decode_syntax()`, `parity_entropy.*` and the static rate model is derived from
the PDF. The **only non-normative part is the level choice**: the spec does not
say which coefficient an encoder should perturb to satisfy
`|DC| & 1 == sumAbs1`.

`quantize()` uses a deliberately small, deterministic candidate set:

* if the seed levels already satisfy the constraint, use them unchanged;
* otherwise try a ±1 magnitude step on the DC (both signs when the DC is zero)
  and a ±1 step on each AC coefficient that flips the capped parity
  `min(level, 6) & 1` (plus the special case where reducing the fourth AC
  non-zero to zero legally drops `numNz` below `PHTHRESH` and turns hiding
  off);
* `kMinDistortion` selects the smallest coefficient SSE;
* `kRateDistortion` selects the smallest `SSE + lambda * bits` with the static
  §9.3 model.

Blocks up to `exhaustive_max_area` (default 64) search every AC position.
Larger blocks keep the best eight AC positions by local distortion delta.
This is an encoder design, not a claimed normative algorithm.

There is **no per-block "hide / do not hide" bit**. When the frame signals
`allow_parity_hiding = 1` and a block has at least four AC non-zeros, the
decoder derives `isHidden = 1` and the hidden-DC syntax is mandatory. A block
can only avoid it by having fewer than `PHTHRESH` AC non-zeros (which the
quantiser may choose as an RD option) or by the frame not allowing the tool.

## Files

| File | Purpose |
|---|---|
| `parity_hiding.h` / `.cpp` | geometry, scans, contexts, exact syntax, decoder inverse, static rate, level choice |
| `parity_entropy.h` / `.cpp` | §8.2 entropy writer/reader for the coefficient payload, CDF store, escape bits |
| `parity_goldens.h` | pinned syntax and payload goldens |
| `parity_test.cpp` | deterministic tests, exhaustive min-distortion oracle, payload gate |
| `parity_rd_compare.cpp` | real-coefficient scalar / RDOQ / parity-hiding RD harness |
| `build_and_test.sh` | standalone build, tests, gate and RD measurement |

Dependencies are read-only: `encoder/m1/qctx/qctx.h` (the committed §9.3 CDF
slices), `entropy/av2/av2_entropy.h` (§8.2 coder), and, for the RD harness,
`tx_variants`, `rdoq`, `rdo` and `y4m`.

## Build and test

```sh
cd src/encoder/tcq/parity
./build_and_test.sh
PARITY_SANITIZE=1 PARITY_SKIP_RD=1 ./build_and_test.sh   # ASan + UBSan
PARITY_SKIP_RD=1 ./build_and_test.sh                     # tests + gate
PARITY_QPS="60 90 120" ./build_and_test.sh               # RD sweep
```

No CMake; the script writes only to `${TMPDIR:-/tmp}`. Current result:

```
parity-hiding tests: 2054823 checks, 0 failures
coefficient-payload gate: 75 blocks, 9775 payload bytes, decoded and matched
```

Both core and gate are clean under `-fsanitize=address,undefined`.

`parity_test --print-goldens` regenerates the syntax and payload of the pinned
golden blocks. `parity_test --write-gate FILE` / `--read-gate FILE` runs the
coefficient-payload gate in two processes.

## API

```cpp
namespace av2enc::tcq::parity {

// Geometry / tables
bool make_geometry(int tx_size, TxClass, int plane, Geometry*, std::string*);
void build_scan(int w, int h, TxClass, std::span<int>);
LevelLimits level_limits(int row, int col, TxClass, int plane);
bool is_low_frequency(int row, int col, TxClass, int plane);
int eob_multisize(const Geometry&);
int tx_sz_ctx(int tx_size);

struct Eligibility {
  bool allow_parity_hiding;  // frame flag
  bool lossless;             // CodedLossless / segment lossless
  bool allow_tcq;            // frame allow_tcq
  bool tx_type_is_idtx;
  bool use_fsc;
  bool is_inter;             // EOB / txb_skip CDF context
};
bool parity_hiding_eligible(const Geometry&, const Eligibility&);

// Exact syntax (all arrays in §5.20.7.30 scan order unless stated)
struct PositionSyntax { /* base/br symbols, CDF kind+ctx, escape x+bits, sign */ };
struct BlockSyntax { /* eob, isHidden, sumAbs1, dc_ctx, EOB syntax, positions */ };
bool derive_syntax(std::span<const int32_t> levels_scan,
                   const Geometry&, const Eligibility&,
                   BlockSyntax*, std::string*);
struct DecodedBlock { std::vector<int32_t> quant_levels; /* raster */ ... };
bool decode_syntax(const BlockSyntax&, const Geometry&,
                   DecodedBlock*, std::string*);

// Static §9.3 rate model
struct RateConfig { int qctx, txb_skip_ctx, dc_sign_ctx; bool is_inter; };
double block_rate_bits(const BlockSyntax&, const Geometry&, const RateConfig&);
int escape_bit_count(int x, int predictor);

// Encoder level choice
struct QuantizeConfig {
  Eligibility eligibility;
  QuantizeMode mode;        // kMinDistortion or kRateDistortion
  double quant_step, lambda;
  std::span<const double> dequant_step;  // optional per-position step (QM)
  RateConfig rate;
  int exhaustive_max_area;
  int pruned_ac_candidates;
};
struct QuantizeResult {
  std::vector<int32_t> levels;  // full Quant[], scan order
  BlockSyntax syntax;
  int eob;
  bool hidden, hidable, hiding_turned_off;
  int adjusted_scan_index, adjusted_delta;
  double distortion, rate_bits, cost;
};
bool quantize(std::span<const double> original,
              std::span<const int32_t> levels_seed,
              const Geometry&, const QuantizeConfig&,
              QuantizeResult*, std::string*);
bool quantize_scalar(...);
bool apply_parity_hiding(std::span<const double> original,
                         std::span<int32_t> levels, ...);

// §8.2 entropy
struct CdfStore { explicit CdfStore(int qctx); /* per-tile mutable CDFs */ };
struct EntropyConfig { Eligibility eligibility; int qctx, txb_skip_ctx,
                       dc_sign_ctx; bool disable_cdf_update; };
bool write_block(av2::Encoder&, CdfStore&, const BlockSyntax&,
                 const Geometry&, const EntropyConfig&, std::string*);
bool read_block(av2::Decoder&, CdfStore&, const Geometry&,
                const EntropyConfig&, BlockSyntax*, DecodedBlock*,
                std::string*);
}
```

`quantize()`/`decode_syntax()` use different but documented conventions:
encoder inputs and `QuantizeResult::levels` are in **scan order**;
`DecodedBlock::quant_levels` is the normative **raster** `Quant[]`;
`decoded_levels()` converts back to scan order.

## Exact syntax

The implementation follows §5.20.7.27 literally:

1. `eob` is the last non-zero scan position + 1. `txb_skip`, `eob_pt` and the
   EOB extra bits use the existing `m1::qctx` tables. `eobMultiSize` is
   `Min(log2(coded_w),5)+Min(log2(coded_h),5)-4`.
2. First pass, `c = eob-1 .. 0`:
   * `c == eob-1` uses `coeff_base_eob`/`coeff_base_lf_eob` (or the `_uv`
     variants) with `level = symbol + 1`;
   * otherwise `coeff_base`/`coeff_base_lf` (or `_uv`) with `level = symbol`;
   * `c == 0` and `isHidden` uses `TileCoeffBasePhCdf[Min(ctx,4)]`;
   * `coeff_br` is read when `level > baseLevels` and not LF chroma;
   * `sumAbs1 ^= Min(level, 6) & 1` and `numNz` count only `c > 0`;
   * `isHidden = numNz >= PHTHRESH` (4).
3. The hidden DC base symbol is `Min(|dc| >> 1, 3)`. `maxLevel` is
   `NUM_BASE_LEVELS + 1 = 3`; the escape is present from `q >= 3` and
   `x = q - 3`.
4. Second pass, `c = eob-1 .. 0`: sign, then `read_quant`.
   * sign presence is `level != 0 || (isHidden && c == 0 && sumAbs1 > 0)`;
   * luma DC-style signs: the true DC (`pos==0`) uses
     `TileDcSignCdf[0][isHidden][caller ctx]`; `dc_sign_horz_vert`
     (`HORIZ && col==0`, `VERT && row==0`, not the true DC) uses context 0;
     all other signs are bypass bits;
   * the hidden DC escape predictor is `hrLevelAvg >> 1`; every other escape
     uses `hrLevelAvg`;
   * `Quant = 2 * Quant + sumAbs1` only at `c == 0 && isHidden`.
5. §7.14.4 dequant is unchanged: parity hiding adds **no** shift. The skipped
   bit is inferred, so the decoder dequantises the full `Quant[]`.

Context/CDF axes that are easy to get wrong and are covered by the tests:

* `coeff_base`/`coeff_base_lf` include the `(tcqState >> 1) & 1` axis. Parity
  hiding and TCQ are mutually exclusive, so it is always index 0 here.
* LF luma 1D contexts are `LF_SIG_COEF_CONTEXTS_2D + Min(ctx,6)` /
  `+7 + Min(ctx,4)` (21..32), not the 2D 9/16 buckets; HF luma 1D base
  contexts are `Min(ctx,4) + 15` (15..19).
* LF 1D chroma contexts start at `LF_SIG_COEF_CONTEXTS_2D_UV = 8`.
* `coeff_br` for a 1D luma DC uses `mag + 7`; 2D DC uses `mag`.
* `coeff_base_ph` uses five 3-clipped reference neighbours and no tx-size or
  TCQ axis.
* `dc_sign` has the `isHidden` group; `dc_sign_horz_vert` uses context 0 of
  the same table.
* `eob_pt` escape literals are written whenever the last CDF symbol is used,
  including `eob_pt == cdf_n` with a zero literal.

Cross-block contexts are inputs/outputs, not re-derived here: the caller passes
`RateConfig::txb_skip_ctx` and `RateConfig::dc_sign_ctx` (the Above/Left
`txb_skip` and DC-sign contexts) and receives `DecodedBlock::cul_level` and
`decoded.dc_category` to write into `AboveLevelContext`/`LeftLevelContext` and
`AboveDcContext`/`LeftDcContext` for the next block.

## Pinned goldens

`parity_goldens.h` pins syntax, rate bits and entropy bytes for:

| Name | Block | What it pins |
|---|---|---|
| `hidden_dc6_eob5` | 8x8 2D luma | hidden DC, `q=3`, escape `x=0`, minimum EOB for hiding, PH context 4 |
| `hidden_dc8_horiz` | 4x16 HORIZ luma | 1D LF base/br contexts (22/23), `q=4`, escape `x=1`, `dc_sign` on every `col==0` coefficient |
| `hidden_dc0` | 4x4 2D luma | hidden DC = 0, no DC sign, `coeff_base_ph` context 4 |
| `chroma_eob1` | 4x4 plane 1 | `eob == 1`, LF chroma `coeff_base_lf_eob_uv`, no `coeff_br`, escape `x=4` |

Payloads (static CDFs, qctx 0):

```
hidden_dc6_eob5   1f 1d e7 2d 80 00
hidden_dc8_horiz  1f 31 d9 75 00 00
hidden_dc0        37 01 9c 98 00 00
chroma_eob1       30 d6 00 00
```

The tests compare every symbol, context, escape, sign, rate bit and payload
byte, then decode the payload with the §8.2 reader and compare the
reconstructed `Quant[]`.

## Tests

`parity_test.cpp` covers:

* all 25 TX sizes × 3 scan classes × 3 planes: geometry, adjusted size, scan
  permutation and order, `txSzCtx`, `eobMultiSize`, LF limits;
* every eligibility exclusion (frame flag, plane, lossless, TCQ, IDTX, FSC),
  including interop with `tcq::derive_header_flags()` from `../tcq_flags.h`;
* random syntax/decode round-trips over all sizes and planes, with all
  DC/AC parity combinations;
* all parities of `|DC| & 1` vs `sumAbs1`, including the rejection of illegal
  mismatches;
* hidden DC escape values `q = 0..9`, non-hidden escapes at
  `pre_max-1/pre_max/pre_max+1` for luma LF/HF and chroma LF/HF;
* every EOB from 0 to 256 on 16x16, sampled EOBs on 32x32, and the
  `eob_pt == cdf_n` escape-literal boundaries for 16x16/32x32/64x64;
* entropy round-trips with static and adaptive CDFs, a 32-block adaptive
  session, and 64×301 `read_quant` escape values;
* the quantiser over all sizes/planes, determinism, hidden-parity invariants,
  TCQ exclusion, EOB shrink, and an exhaustive min-distortion oracle for
  4x4/8x8 blocks;
* the four goldens.

The coefficient-payload gate writes 75 blocks (all TX sizes × planes) with one
adaptive entropy stream and decodes them in a second process.

## Real-coefficient RD

`parity_rd_compare` uses the real Y4M fixtures, DC-predicted 8x8 residual
blocks, the existing `tx_variants` forward transform, and one lambda for every
candidate (`rdo::lambda_sse_from_quant_step(1/quant_factor, 0.125)`). All
candidates are scored with the same static §9.3 model and coefficient SSE.

640x360, `tests/data/m0_640x360_1f.y4m`, 3600 luma blocks:

| QP | scalar cost | RDOQ vs scalar | RDOQ+PH(dist) vs RDOQ | RDOQ+PH(RD) vs RDOQ | RD bits vs RDOQ | hidden blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 60  | 288,400 | −1.60% | −0.69% | **−0.99%** | −1.45% | 770 / 3600 |
| 90  | 1,123,049 | −4.50% | −1.16% | **−1.61%** | −2.71% | 702 / 3600 |
| 120 | 4,027,380 | −8.12% | −0.15% | **−0.73%** | −2.33% | 589 / 3600 |

64x64 QP 90: RDOQ+PH(RD) is **−1.86%** vs RDOQ (bits −3.13%), RDOQ+PH(dist)
is −1.35%.

The `scalar+PH(dist)` row is included for completeness but is not a realistic
seed: at QP 60 it is +0.63% vs RDOQ and at QP 120 +7.08%, because scalar levels
need many AC parity repairs. Parity hiding pays when the seed is RDOQ.

These are static-model bits on real coefficients, not a tile re-encode; the RD
harness models quantisation only.

## Decoder gate

### What runs here

`build_and_test.sh` runs the coefficient-payload gate in two processes:

```
parity_test --write-gate  $TMP/parity_gate.ph
parity_test --read-gate   $TMP/parity_gate.ph
```

The writer emits one adaptive §8.2 entropy stream for 75 blocks (all 25 TX
sizes × 3 planes, with PH-legal levels). The reader uses the normative §8.2
decoder plus the §5.20.7.27 inverse and matches every reconstructed `Quant[]`.
This is a real byte-level decoder round-trip, but **not** a complete AV2 tile.

### Why a full stream is blocked

No existing core writer can signal parity hiding:

| File | Site | Hardcoded |
|---|---|---|
| `src/encoder/m1/m1_headers.cpp` | 136–138 | `enable_cctx = enable_tcq = enable_parity_hiding = 0` |
| `src/encoder/m2/m2_headers.cpp` | 197–199 | same |
| `src/bitstream/m0_headers.cpp` | 243–245 | same |
| `src/encoder/m1/tiles_stream/tiles_stream.cpp` | 160–161 | same |
| `src/encoder/sb128/sb128_stream.cpp` | 490 | same |
| `src/encoder/lossless/lossless.cpp` | 2846 | same |
| `src/encoder/pipeline/tiled_still_gate.cpp` | 186 | same |
| `src/encoder/m1/mrl_stream/mrl_stream.cpp` | 592 | same |

In addition, the M1/M2 coefficient writers have no hidden-DC path:
`m1_syntax.cpp:1127–1300`, `m2_syntax.cpp:1239–1414`,
`write_quant_escape` at `m1_syntax.cpp:889` / `m2_syntax.cpp:980`, and
`CdfState`/`init_cdfs` at `m1_syntax.cpp:169–253` /
`m2_syntax.cpp:215–321`.

### Exact construction once the core edits are allowed

The simplest legal stream is **TCQ off, PH on**, so the single-picture path
needs no `choose_tcq_per_frame` bit:

1. Sequence header (`m1_headers.cpp:137–138`, `m2_headers.cpp:198–199`):
   `enable_tcq = 0`, `enable_parity_hiding = 1`. The `enable_parity_hiding`
   bit is present because `enable_tcq == 0`. `M1Config`/`M2Config` have no PH
   field today; add one (or hardcode 1 for a gate build).
2. Frame header: after `delta_q_present`
   (`m1_headers.cpp:178`, `m2_headers.cpp:285`) and before
   `deblocking_filter_params()`, write `allow_parity_hiding f(1)`.
   `allow_tcq` is inferred 0 (`enable_tcq = 0`, `choose_tcq_per_frame = 0`).
   For M2 add a per-frame `allow_parity_hiding` field; a still can hardcode 1.
3. Coefficient writer: add `coeff_base_ph[5][5]` to `CdfState` and copy it in
   `init_cdfs`; compute `sumAbs1`/`isHidden` in the first pass; use
   `coeff_base_ph` for `c == 0 && isHidden`; use `maxLevel = 3`; change the
   sign presence and `dc_sign` group; pass `lvlShift = 1` to the escape for
   the hidden DC.
4. Decision (`m1_still.cpp:882–950`, `m2_frame.cpp:154–230`,
   `m2/residual/inter_residual.cpp:280–330`): after scalar/RDOQ, call
   `parity::quantize()` with the scan-order coefficients and seed levels, and
   write the returned full levels back. Reconstruction consumes the returned
   levels unchanged (`m1_still.cpp:978`, `m2_frame.cpp:210–230`); parity hiding
   adds no dequant shift.
5. Then the gate is
   `tools/validate_dav2d.sh parity.ivf tests/data/m0_64x64_1f.y4m` with
   `enable_tcq = 0`, `enable_parity_hiding = 1`, `allow_tcq = 0`,
   `allow_parity_hiding = 1`.

Exact call sites:

| Path | Function / site | Change |
|---|---|---|
| M1 | `m1_still.cpp:882–950` `run_quant` lambda | build scan-order `original`/`seed`; call `parity::quantize`; scatter `result.levels` back to `plan.levels` |
| M1 | `m1_still.cpp:765–784` `dequant_inverse_qindex` | no change for PH; keep the standard `dq_denom` |
| M1 | `m1_syntax.cpp:169–253` | add `coeff_base_ph`; `dc_sign` already has the `[2][2][3][3]` shape |
| M1 | `m1_syntax.cpp:1127–1300` | first-pass hidden state and second-pass sign/escape |
| M2 | `m2_frame.cpp:154–230` `compute_tx` | same quantiser call after `forward_transform` |
| M2 | `m2/residual/inter_residual.cpp:280–330` | same for the inter residual path |
| M2 | `m2_syntax.cpp:215–321`, `1239–1414` | same syntax changes |
| M2 | `m2_headers.cpp:285` | insert the `allow_parity_hiding` bit |

## Limitations

* The RD harness is quantisation-only: fixed DC prediction, 8x8 blocks, no
  partition/mode re-decision and no real tile re-encode. Bits are static
  `-log2(p)`; CDF adaptation is not modelled in the RD numbers.
* The entropy codec covers the `coeff_base`/`coeff_br`/sign path only. IDTX/FSC
  use a different coefficient syntax (`coeff_base_idtx`, `coeff_base_bob`,
  `idtx_sign`); the codec rejects those flags rather than pretending to support
  them. Parity hiding is ineligible for IDTX/FSC anyway.
* The quantiser's RD search explores parity-repair candidates when the seed
  parity is wrong. It does not re-run a full RDOQ trellis with the hidden-DC
  constraint inside the DP; it post-processes RDOQ levels, which is what the
  AVM "trellis off" path does.
* `kRateDistortion` uses the static §9.3 model, not the live adaptive CDF
  state.
* No full dav2d stream gate was run: see the blocked sites above.

## Spec tables / missing data

Everything needed is committed:

* §5.20.7.27 offsets (`Sig_Ref_Diff_Offset`, `Mag_Ref_Offset_With_Tx_Class`),
  `Tx_Width`/`Tx_Height`, `eobMultisize`, `Tcq_Next_State` — `spec/tables/`;
* `Default_Coeff_Base_Ph_Cdf` and every coefficient CDF —
  `src/encoder/m1/qctx/qctx_cdf_tables.h` (generated from
  `spec/tables/default_cdf.json`);
* `PHTHRESH`, `NUM_BASE_LEVELS`, `COEFF_BASE_RANGE`, `LF_NUM_BASE_LEVELS`,
  `LF_BASE_SYMBOLS`, `COEFF_BASE_PH_CONTEXTS`, `LF_SIG_COEF_CONTEXTS_2D*` —
  `spec/tables/constants.csv`.

The PDF is not redistributed. No numeric table is hand-typed in this
component; the few small offsets are copied from §5.20.7.27 and cited inline.
