# AV2 intra transform-type selection (`src/encoder/tx_type/`)

M1 component: choose the **intra TX_TYPE** for a transform block.

`src/encoder/tx_variants/` owns *how* one block is transformed (kernels,
quantisation, adjusted sizes). This component owns *which* `TX_TYPE` an intra
luma block signals: the legal set per size (§5.20.8.3), the codable candidates
per set and intra mode (§5.20.8.2/§5.20.7.29), a full per-candidate
transform/quantise/inverse evaluation, and an `SSE + λ·rate` decision with a
deterministic tie-break.

It is a CPU reference: no bitstream writer, no partition search, no frame
reconstruction, and it is not wired into `src/encoder/m1/`. The GPU batch
counterpart is `src/cuda/tx_type/`.

## Scope

- 8-bit lossy intra luma, all 25 coded TX sizes (4x4 … 64x64, square and rect)
- all 16 `TX_TYPE` values for evaluation; the *legal* subset per size and mode
- `reduced_tx_set` 0–3 and the chroma-style `enable_chroma_dctonly` override
- §7.15.2.1 length-32 ADST/FDST → DCT aliasing (including adjusted 64-point
  sizes)

Out of scope: lossless/FSC (`fsc_mode` → IDTX), inter (no
`intra_tx_type` symbol; inter sets are named only), the inter-only DDT/FDDT 1D
transforms, secondary transforms (IST), and DPCM.

## Spec facts this component pins

| Fact | Where |
|---|---|
| `get_tx_set()` set per size, for an intra block | §5.20.8.3 |
| `reduced_tx_set == 2` forces `DCT_DCT` for intra | §5.20.8.2 |
| `eob == 1` forces `DCT_DCT`; `eob == 0` never signals a type and the decoder resets luma `TxTypes` to `DCT_DCT` | §5.20.8.2, §5.20.7.28 |
| `INTRA_1` = 7 mode-dependent symbols (`Md_Idx_To_Type`), `INTRA_2` = 2 (`DCT_DCT`, `ADST_ADST`) | §5.20.8.2, §9.2, §9.3 |
| `WIDE_32`/`HIGH_32` signal `is_long_side_dct` + one of 4 types; `WIDE_64`/`HIGH_64` imply the flag = 1 and signal one of 4 | §5.20.8.2 |
| Length-32 ADST/FDST use `Dct_Kernel32`; 64-point sizes are adjusted 32-point transforms | §7.15.2.1, §7.15.4 |
| `Transform_1d_Type` is indexed `[txType][dir]`, dir 0 = row (horizontal) | §7.15.4.1 |

## Intra TX set per size (`reduced_tx_set = 0`)

`TX_SIZE` order is §6.19.6.1. `n` is the number of evaluated candidates; the
last column is what the bitstream actually codes.

| Coded size | `get_tx_set` | Candidates (n) | Syntax |
|---|---|---:|---|
| 4x4, 8x8, 16x16, 4x8, 8x4, 8x16, 16x8, 4x16, 16x4 | `TX_SET_INTRA_1` | 7 mode-dependent | `intra_tx_type` (7 symbols) |
| 16x32, 32x16, 8x32, 32x8, 4x32, 32x4 | `TX_SET_HIGH_32` / `WIDE_32` | 8 (2 × 4) | `is_long_side_dct` + `intra_tx_type` (4 symbols) |
| 16x64, 64x16, 8x64, 64x8, 4x64, 64x4 | `TX_SET_HIGH_64` / `WIDE_64` | 4 | `intra_tx_type` (4 symbols), flag implied 1 |
| 32x32, 32x64, 64x32, 64x64 | `TX_SET_DCTONLY` | 1 | no type syntax (`DCT_DCT`) |

`reduced_tx_set`:

| Value | Effect (intra luma) |
|---:|---|
| 0 | full sets above |
| 1 | sizes below 32x32 → `TX_SET_INTRA_2` (`DCT_DCT`, `ADST_ADST`); long/64 sets unchanged |
| 2 | `DCT_DCT` only, every size |
| 3 | like 1 |

M1 currently writes `reduced_tx_set = 0` (`src/encoder/m1/m1_headers.cpp`) and
always codes symbol 0, i.e. `DCT_DCT`, so enabling this component changes the
bitstream and needs its own `dav2d` gate.

### Candidate sets in detail

- **INTRA_1 / INTRA_2**: exactly `Md_Idx_To_Type[Size_Class[txSz]][mode][symbol]`
  for symbol `0..6` / `0..1` (generated table, §9.2).
- **WIDE_64 / HIGH_64**: `Tx_Type_Inv_Long[1][orientation][symbol]`,
  orientation 0 = wide, 1 = high.
- **WIDE_32 / HIGH_32**: four candidates for `is_long_side_dct = 0` then four
  for `is_long_side_dct = 1`.
- **DCTONLY / reduced_tx_set == 2 / `dct_only`**: `{DCT_DCT}`.

`Tx_Type_In_Set_Intra` (§5.20.7.29) is generated too and every candidate is
asserted to be a member of its set.

### Where the length-32 alias matters

§7.15.2.1 has no length-32 ADST/FDST kernel, so those dimensions use DCT. The
intra sets above never combine a 32-point dimension with an ADST/FDST on that
same dimension, so the alias does not collapse any *legal intra* candidate:

- the 64-wide/tall sets are `IDTX`/`DCT` on the long side;
- the flag = 1 long-32 branches use DCT on the long side;
- 32x32/32x64/64x32/64x64 are `DCTONLY`.

It still matters for the API: `evaluate_intra_tx_type()` accepts any `TX_TYPE`
(exhaustive tests, inter-style comparisons, future sets), and the transform
layer must alias. This component therefore verifies the alias for every size ×
type pair that has an ADST/FDST on a 32-sample dimension (144 pairs), including
64x64 vs its adjusted 32x32 transform.

## Files

| File | Contents |
|---|---|
| `tx_type.h` | public API: sets, candidates, symbols, rate model, selection |
| `tx_type.cpp` | implementation (delegates the transform maths to `txv::*`) |
| `tx_type_tables.h` | generated §5.20.7.29 / §9.2 / §9.3 numbers + scans |
| `generate_tx_type_tables.py` | regenerates the header from `spec/tables/` |
| `tx_type_test.cpp` | self-test (382,552 checks) |
| `build_and_test.sh` | standalone build + run (links the real CPU components) |

Regenerate the tables:

```sh
python3 generate_tx_type_tables.py          # writes tx_type_tables.h
python3 generate_tx_type_tables.py --check  # fails if the header is stale
```

The generator needs `spec/tables/_cache/spec.html` for
`Tx_Type_In_Set_Intra` (that table exists only in the cached HTML; the other
inputs are committed JSON). The generated header is committed, so normal builds
do not need the cache.

## API

```cpp
namespace av2enc::txtype {

struct TxTypeQuery {
  int coded_w = 0, coded_h = 0;  // 4..64
  int mapped_mode = 0;           // wide_angle_mapping() output, 0..12
  int reduced_tx_set = 0;        // §5.20.2, 0..3
  bool dct_only = false;         // enable_chroma_dctonly / force DCT
};

int  map_intra_mode(int mode, int angle_delta, int w, int h);
bool intra_tx_set(const TxTypeQuery&, TxSet*);
bool intra_tx_candidates(const TxTypeQuery&, TxTypeSet*);   // symbol order
int  intra_tx_types_for_size(int w, int h, int reduced, bool dct_only,
                             TxType* out, int cap);
bool tx_type_symbol(const TxTypeQuery&, TxType, int* symbol,
                    bool* is_long_side_dct);
bool candidate_codable(TxType, int eob);

struct TxTypeCostConfig {
  double lambda = 0.0;            // cost = sse + lambda * rate_bits
  TxTypeRateFn rate_fn = nullptr; // nullptr -> spec-CDF type bits + nonzero
  void* rate_user = nullptr;
  double coefficient_bits = 1.0;
  bool use_static_type_cost = true;
};

bool evaluate_intra_tx_type(const int16_t* residual_coded, ...);
int  evaluate_intra_tx_candidates(const int16_t* residual_coded, ...);
bool choose_intra_tx_type(const int16_t* residual_coded, const TxTypeQuery&,
                          const TxTypeCostConfig&, TxTypeChoice*,
                          int32_t* levels_out, int16_t* recon_out);
bool choose_tx_type_from_candidates(...);   // explicit candidate subset
}
```

`TxTypeCandidate` carries the `TxParams` that `tx_variants` needs, so a caller
can go straight from a decision to `txv::forward_transform()`. `TxTypeChoice`
returns the type, symbol, `is_long_side_dct`, cost, SSE, rate bits, EOB, nonzero
count, and (optionally) the winning `levels` and coded-size inverse residual.

Buffers are tightly packed and caller-owned; the implementation uses fixed-size
stack scratch (≤ 32 KiB), so there are no heap allocations in the hot path.

## Cost model

```
rate_bits = type_bits + coefficient_bits * nonzero        (default model)
cost      = sse + lambda * rate_bits
```

- `sse` is the reconstruction error **in the coded domain**:
  `sum((inverse_transform(levels) - residual_coded)^2)`, the same metric as
  `txv::tx_select`.
- `type_bits` is the static cost of the signalled type: `-log2(P(symbol))` from
  the §9.3 default CDFs (`Default_Intra_Tx_Type_Set1/Set2/Long`, plus
  `Default_Is_Long_Side_Dct` for WIDE_32/HIGH_32). It is 0 when the set codes no
  type.
- `rate_fn` replaces the default model; it receives `(query, candidate, eob,
  nonzero, sum_abs, sse, type_bits)` and returns bits. This is the hook for real
  §8.2 entropy costs — the default coefficient term is a **proxy**, not an
  entropy model (see `docs/negative-results.md`, "RDO / rate control").

`lambda` is in SSE units per bit and is deliberately a caller input. The tests
use `rdo::lambda_sse_from_quant_step(528/8)` (the actual QP 90 quantiser step)
and the `rdo::lambda_sse_from_qp(90)` placeholder for comparison.

### Deterministic tie-break

In order:

1. lowest `cost`,
2. lowest `sse`,
3. lowest numeric `TxType` (`DCT_DCT` first).

Candidates that are not codable (`eob ≤ 1` and type ≠ `DCT_DCT`) are skipped.
`DCT_DCT` is a member of every legal set, so a winner always exists.
`candidate_index` is the position in the symbol-ordered set (for reporting); the
decision never depends on it.

## Integration API for the TX stage

`src/encoder/m1/m1_still.cpp::compute_tx()` currently hardcodes
`make_tx_params(cw, ch, kDctDct)`. The minimal wiring is:

```cpp
// q.mapped_mode = txtype::map_intra_mode(y_mode, angle_delta, cw, ch)  (luma)
txtype::TxTypeQuery q{cw, ch, mapped_mode, /*reduced_tx_set=*/0, false};
txtype::TxTypeCostConfig cfg;
cfg.lambda = /* SSE-domain lambda from the caller */;

txtype::TxTypeChoice choice;
std::int32_t levels[w * h];
std::int16_t recon_residual[cw * ch];
if (!txtype::choose_intra_tx_type(residual_coded.data(), q, cfg, &choice,
                                  levels, recon_residual)) { ... }

const txv::TxParams& params = choice.candidate.params;  // has tx_type
plan.eob = choice.eval.eob;                             // or recompute
plan.levels = levels;                                   // w*h, raster order
// pred + recon_residual -> plan.recon (as compute_tx does today)
```

Then `TxPlan`/`BlockSyntax` need a `tx_type` field, and
`src/encoder/m1/m1_syntax.cpp` must encode it instead of the hardcoded
`intra_tx_type` symbol 0:

- `set = intra_tx_set(q)`;
- `symbol`/`is_long_side_dct` from `tx_type_symbol(q, choice.candidate.type, ...)`;
- `sym(TileIntraTxTypeSet1Cdf[Tx_Size_Sqr[txSz]], 7, symbol)` for INTRA_1,
  `sym(TileIntraTxTypeSet2Cdf[Tx_Size_Sqr[txSz]], 2, symbol)` for INTRA_2,
  `sym(TileIntraTxTypeLongCdf[Tx_Size_Sqr[txSz]], 4, symbol)` for the long sets,
  and `sym(TileIsLongSideDctCdf[0], 2, is_long_side_dct)` for WIDE_32/HIGH_32;
- skip the type when `eob <= 1` or `reduced_tx_set == 2` (the `TxTypeSyntax`
  enum in `m1_syntax.cpp` already classifies this).

Chroma never signals a type (`compute_tx_type()` derives it); keep chroma on
`DCT_DCT` or use `q.dct_only = true` and the `UVMode`-derived set separately.

The GPU stage consumes the same `TxTypeSet`/`TxTypeCandidate` values; see
`src/cuda/tx_type/tx_type.h` for the batch call and the buffer layout.

## Tests

```sh
cd src/encoder/tx_type
./build_and_test.sh                 # -O2
SANITIZE=1 ./build_and_test.sh      # ASan + UBSan
WERROR=1 ./build_and_test.sh
```

Result on macOS/clang and Linux/g++:

```
382552 checks, 0 failures
RESULT: PASS
```

and the same count under `SANITIZE=1`. Suite breakdown:

| Suite | Checks | What it pins |
|---|---:|---|
| `tables` | 519 | generated §9.2/§9.3 values, set enum values |
| `get_tx_set` | 155 | all 25 sizes × `reduced_tx_set` 0–3 |
| `candidate_sets` | 66,674 | 25 sizes × 13 modes × 4 reduced sets: counts, uniqueness, `Tx_Type_In_Set_Intra` membership, symbol round-trip, table identity |
| `mode_mapping` | 4,557 | `wide_angle_mapping` wiring for every mode/delta/size |
| `diagonal_scan` | 18,317 | §5.20.7.30 scans vs generated 4/8/16/32 tables, permutation for all 25 adjusted sizes, EOB |
| `exhaustive_transforms` | 12,800 | 25 sizes × 16 types × 2 residual kinds against a direct `txv` pipeline, bit-exact levels/recon, metrics, determinism |
| `aliasing_32` | 161,601 | every ADST/FDST at a 32-sample dimension ≡ DCT (144 size/type pairs), 64x64 ≡ adjusted 32x32 |
| `rate_model` | 40 | CDF symbol costs, flag cost, default/custom rate plumbing |
| `tie_break` | 2,080 | cost→SSE→type ordering, symbol-order independence |
| `selection_bruteforce` | 39,675 | 3,600 cases vs brute force over all 16 types; winner levels/recon bit-exact |
| `real_frames` | 76,134 | 64x64 and 640x360 luma from `tests/data` |

## Results (QP 90)

The `tx_variants` quantiser is fixed at `q(90) = 528` (dequant step 66), so
"QP 90" is inherent. Lambdas used: `544.5` (`0.125 · 66²`) and the
`rdo::lambda_sse_from_qp(90)` placeholder (`25,267.6`). Residuals come from DC
prediction of the **original** frame (`intra::predict_block`), which isolates the
transform-type decision from reconstruction drift.

Selections on `tests/data/m0_64x64_1f.y4m` (luma, λ = 544.5):

| Block grid / mode | DCT_DCT | ADST_ADST | DCT_ADST | ADST_DCT | flip variants |
|---|---:|---:|---:|---:|---:|
| 64x64, 1 block, DC (`DCTONLY`) | 1 | 0 | 0 | 0 | 0 |
| 16x16, DC | 11 | 4 | 1 | 0 | 0 |
| 8x8, DC | 37 | 14 | 5 | 2 | 6 |
| 4x4, D45 | 114 | 73 | 26 | 12 | 31 |

Selections on `tests/data/m0_640x360_1f.y4m` (full 16x16 and 32x32 blocks):

| Grid / mode | DCT_DCT | ADST_ADST | DCT_ADST | ADST_DCT | flip variants | DCT lower cost |
|---|---:|---:|---:|---:|---:|---:|
| 16x16, DC | 752 | 44 | 21 | 20 | 43 | 128 non-DCT wins |
| 16x16, D45 | 752 | 44 | 21 | 20 | 43 | 128 |
| 16x16, V | 753 | 45 | 20 | 18 | 44 | 127 |
| 16x16, DC, `reduced_tx_set=1` | 790 | 90 | 0 | 0 | 0 | 90 |
| 32x32, DC (`DCTONLY`) | 220 | 0 | 0 | 0 | 0 | 0 |
| 16x16, DC, `rdo` lambda | 752 | 44 | 21 | 20 | 43 | 128 |

Long-32 shapes (the only intra sets where `IDTX` / `H_DCT` / `V_DCT` are
codable), λ = 544.5:

| Grid / mode | DCT_DCT | IDTX | H_DCT | V_DCT | ADST/FDST variants | DCT lower cost |
|---|---:|---:|---:|---:|---:|---:|
| 16x32, DC | 384 | 3 | 21 | 0 | 32 | 56 non-DCT wins |
| 32x16, DC | 387 | 2 | 8 | 11 | 32 | 53 |

**Does ADST/IDTX beat DCT on SSE + λ·rate?** Yes for ADST variants, on a
minority of blocks: 128/880 (14.5%) at 16x16/DC/λ=544.5, and the aggregate cost
drops from 13,485,919 (always DCT) to 13,056,571 (−3.2%). Only 60 of those 128
blocks also have lower SSE — the rest win on the rate term (fewer nonzero
coefficients at slightly higher SSE). Aggregate SSE is 0.37% *worse* than
always-DCT, which is the expected distortion/rate trade at λ>0.

IDTX is only codable in the `is_long_side_dct = 0` branch of a 32-wide/tall set
(16x32, 32x16, 8x32, …, 4x32, 32x4). On the 16x32/32x16 grids it wins a small
but non-zero share (3/440 and 2/440). The "identity on one axis" types matter
more: `H_DCT` wins 21/440 on 16x32 and 8/440 on 32x16, `V_DCT` wins 11/440 on
32x16. The long-32 sets are worth having: on 16x32 the aggregate cost is 12.32M
vs 15.26M for always-DCT (−19.2%), almost all of it from choosing the long side
correctly. FLIPADST variants also appear
(8x8 and 4x4 grids), so "flip" is not dead code.

The decision is insensitive between λ = 544.5 and the much larger `rdo`
placeholder λ (25,267.6): the same types win on 640x360/16x16. That is a
property of this content and the coarse rate proxy, not a calibration claim.

These numbers measure the selector only. They are **not** an encoded bitstream
and must not be quoted as coding gains until the M1 integration is wired and
`dav2d`-gated.

## Notes / limitations

- The coefficient-rate term is a proxy. Wire `rate_fn` to the real §8.2 costs
  before trusting absolute costs; `docs/negative-results.md` warns about the
  default RDO model.
- `TxTypeQuery::mapped_mode` must be the wide-angle-mapped mode. Use
  `map_intra_mode()` (it calls `intra::map_angle`); do not pass the signalled
  `YMode` directly for a directional block.
- The selector is greedy per transform block. Joint optimisation with the
  partition/mode decision is out of scope.
- Lossless/FSC (`IDTX` forced), inter blocks, IST and DPCM are not modelled.
- No bitstream, no MD5 claim, no speed claim.
