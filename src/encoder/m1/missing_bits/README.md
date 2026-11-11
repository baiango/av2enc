# `src/encoder/m1/missing_bits/` — four absent AV2 v1.0.0 rules

Spec-only reference implementations, writer/reader syntax, pinned goldens,
exhaustive tests, and a `dav2d --strict 1 --threads 1` gate for the
syntax-bearing element.

Normative source: **AV2 Bitstream & Decoding Process Specification v1.0.0**
(2026-05-28, commit `38f28e7`).  Inputs: `docs/spec-audit/07-prediction.md`
and `docs/spec-audit/04-tile-block.md`.

This directory **adds files only**.  It does not edit `src/encoder/m1/m1_*`,
`src/main.cpp`, the CMake build, or any existing core/component.

```
missing_bits/
  missing_bits.h                 umbrella include
  dc_intra_subsampled.h/.cpp     §7.13.2.11
  intra_filter_corner.h/.cpp     §7.13.2.14
  dc_sign_horz_vert.h/.cpp       §5.20.7.27 writer + reader + context
  save_subpu.h/.cpp              §7.13.3.15
  missing_bits_test.cpp          exhaustive unit tests
  missing_bits_goldens.h         pinned process + entropy + gate goldens
  gate_dc_sign_horz_vert.cpp     minimal 32x16 stream writer
  gate_dav2d.py                  pinned strict/threads-1 decoder gate
  build_and_test.sh              standalone build + tests + gate
```

## 1. What was absent and what is here

| Rule | Audit row | Production status | Added here | Call site |
|---|---|---|---|---|
| §7.13.2.11 DC intra **subsampled** (`stepW/stepH = 2` when extent > 32) | I19 **ABSENT** | `intra_modes.cpp:544-580` implements only §7.13.2.10 (`predict_dc`) | `dc_intra_subsampled()` + `DcSubsampledTrace` + Div_Lut reference | `predict_dc()` would be replaced by a branch on `(w > 32 || h > 32)` before `intra_modes.cpp:544`; every CfL/DC caller (e.g. `m1_still.cpp:1094`) reaches it through `predict_intra` |
| §7.13.2.14 filter corner (5/6/5 subset, `Round2(...,4)`, guard `w+h >= 24`) | I22 (row says WIRED) | `intra_modes.cpp:130-136` (`filter_corner`) + `:719-723` guard | independent `filter_corner_value()` / `filter_corner_applies()` / `filter_corner_if_applicable()` | same guard in `predict_intra`; the module is the reference the production copy is checked against |
| §5.20.7.27 `dc_sign_horz_vert S()` + ctx | §5.20.7.27 **ABSENT** | `m1_syntax.cpp:1276-1288` only codes `dc_sign` and falls back to `raw_bit` | `select_sign_element()`, `dc_sign_context()`, `write/read_dc_sign_horz_vert()`, `write/read_sign_element()` | `m1_syntax.cpp:1278-1283`: replace the two `raw_bit(negative)` branches with `write_sign_element()` |
| §7.13.3.15 `SubPuSize` / `SubPuColBase` / `SubPuRowBase` | N29 **ABSENT** | `deblock_core.h:118-127` stores `Max_Tx_Size_Rect[MiSize]`, used only when `allow_df_sub_pu == 1` | `find_tx_size()`, `SubPuGrid`, `save_subpu()` | §7.13.3.14 `predict_subblock` after the per-list prediction loop; in this tree the analogue is `m2_frame.cpp:488` (`inter::mc::predict_plane`), fed by the reference-MV/sub-PU stacks in `inter/temporal/refinemv.cpp` and `opfl.cpp` |

The audit's §7.13.2.12/.13 numbering for the corner process is off by two:
the 5/6/5 rule is **§7.13.2.14**.  §7.13.2.12 is IBP DC and §7.13.2.13 is
smooth prediction; both were out of scope here.

## 2. API

### `dc_intra_subsampled.h`

```cpp
struct DcSubsampledInput {
  const std::int16_t* above; const std::int16_t* left;
  int w, h, bit_depth; bool have_above, have_left;
};
struct DcSubsampledTrace { int step_w, step_h, sum, count, avg; };

constexpr int dc_subsampled_step(int extent);   // 2 if extent > 32, else 1
int approx_divide(int numerator, int denominator);
bool dc_intra_subsampled(const DcSubsampledInput&, std::uint16_t* dst,
                         std::size_t stride, DcSubsampledTrace* = nullptr);
```

`step_w`/`step_h` are 0 in the trace when the corresponding edge is
unavailable.  Output is 16-bit so 10/12-bit predictions are not truncated; the
8-bit pipeline narrows.

### `intra_filter_corner.h`

```cpp
constexpr int filter_corner_value(int left0, int above_m1, int above0);
constexpr bool filter_corner_applies(int w, int h, int p_angle);
void filter_corner(av2enc::intra::EdgeRow& above, av2enc::intra::EdgeRow& left);
bool filter_corner_if_applicable(int w, int h, int p_angle,
                                 av2enc::intra::EdgeRow& above,
                                 av2enc::intra::EdgeRow& left);
```

`filter_corner_applies` is the non-IBP guard
`pAngle > 90 && pAngle < 180 && w + h >= 24`.  The value is
`(LeftCol[0]*5 + AboveRow[-1]*6 + AboveRow[0]*5 + 8) >> 4`.

### `dc_sign_horz_vert.h`

```cpp
enum class TxClass { k2D, kHoriz, kVert };
enum class SignElement { kDcSign, kDcSignHorzVert, kSignBit };
SignElement select_sign_element(int plane, int row, int col, TxClass);
constexpr int dc_sign_horz_vert_context();          // 0
int dc_sign_context(const int8_t* above_dc, const int8_t* left_dc,
                    int x4, int y4, int w4, int h4, int mi_cols, int mi_rows);
std::uint16_t* dc_sign_cdf_slot(DcSignCdfTable, int ptype, int hidden, int ctx);
std::uint16_t* dc_sign_horz_vert_cdf_slot(DcSignCdfTable, int ptype, int hidden);

bool write_dc_sign_horz_vert(av2::Encoder&, std::uint16_t* cdf, int sign);
bool read_dc_sign_horz_vert(av2::Decoder&, std::uint16_t* cdf, int* sign);
bool write_dc_sign(av2::Encoder&, std::uint16_t* cdf, int ctx, int sign);
bool read_dc_sign(av2::Decoder&, std::uint16_t* cdf, int ctx, int* sign);
bool write_sign_element(av2::Encoder&, SignElement, std::uint16_t* cdf,
                        int dc_sign_ctx, int sign);
bool read_sign_element(av2::Decoder&, SignElement, std::uint16_t* cdf,
                       int dc_sign_ctx, int* sign);
```

`dc_sign_horz_vert` uses `TileDcSignCdf[ptype][isHidden][0]` (§8.3 p603):
its context is the constant 0, unlike `dc_sign`, which uses the three-way
neighbour context above.  Both elements adapt the same `[ptype][isHidden]`
row in place unless `disable_cdf_update` is set.

### `save_subpu.h`

```cpp
enum class TxSize { /* TX_SIZES_ALL order, 25 values */ };
int find_tx_size(int w, int h);                 // index or kTxInvalid

struct SubPuGrid {
  int mi_rows, mi_cols;
  std::vector<std::uint8_t>  size[2];
  std::vector<std::int32_t>  col_base[2], row_base[2];
  // size_at / col_base_at / row_base_at
};

bool save_subpu(int plane, int x, int y, int w, int h,
                int subsampling_x, int subsampling_y,
                int pu_width, int pu_height, SubPuGrid&, std::string*);
```

The process early-terminates when `w == PuWidth && h == PuHeight`.
`SubPuSize = find_tx_size(w, h)` of the **predicted subblock**, not
`Max_Tx_Size_Rect[MiSize]`.  Both planes are indexed on the luma MI grid;
chroma uses `x >> (2 - subX)`, `y >> (2 - subY)` and writes
`h >> (2 - subY)` by `w >> (2 - subX)` cells.

## 3. Tests

`./src/encoder/m1/missing_bits/build_and_test.sh` builds with
`-std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow` and runs
**70,693,371 checks**:

* **DC subsampled** — all `w,h` in `1..64`, all four availability
  combinations, 8/10/12-bit; step/count/sum/avg and every output sample.
  Odd extents (`7x5`, `33x7`, `63x37`, ...) are included.
* **Production parity, inactive DC** — power-of-two `4..32` extents, all four
  availabilities, random edges: `dc_intra_subsampled` is byte-identical to
  `av2enc::intra::predict_intra(..., kDc)`.  This is the §7.13.2.10 vs
  §7.13.2.11 equality proof when no step is 2.
* **Corner** — guard over all `w,h in 1..64` and `pAngle in 1..269`; the
  5/6/5 `Round2(...,4)` value over all 8-bit `(left0, above_m1, above0)`
  triples (16.7 M cases); both corner slots updated.
* **Production parity, inactive corner** — all directional modes × angle
  deltas × `4,8,16,32,64`: pre-applying the reference to an `IntraEdges` copy
  leaves `predict_intra` output byte-identical whenever the guard is false.
  The active direction is also counted and must change at least one
  prediction, proving the production call site is live.
* **`dc_sign_horz_vert`** — element selection for every `plane/row/col/class`;
  `dc_sign_context` against a literal transcription on 4,000 random neighbour
  grids; writer/reader round-trip (frozen and adaptive) with CDF-state
  equality; the full `write_sign_element`/`read_sign_element` dispatch; pinned
  entropy bytes for a 10-symbol sequence.
* **`save_subPU`** — all 25 TX sizes vs the deblock component's
  `Tx_Width/Tx_Height` tables; early termination leaves the grid untouched;
  every valid `4..64` shape × plane 0/1/2 × subsampling 0/1 × origins; odd
  sizes rejected; out-of-bounds rejected; pinned 32x16 luma and 16x8 chroma
  sub-PU examples.

Regenerate process/entropy goldens with:

```
./src/encoder/m1/missing_bits/build_and_test.sh --dump-goldens
```

## 4. `dav2d` gate

```
./src/encoder/m1/missing_bits/build_and_test.sh            # tests + gate
SKIP_GATE=1 ./src/encoder/m1/missing_bits/build_and_test.sh
DAV2D=/path/to/dav2d ./src/encoder/m1/missing_bits/build_and_test.sh
```

The gate writer emits a **32x16 single-leaf frame**:

* forced partition chain `64x64 → HORZ → 64x32 → VERT → 32x32 → HORZ → 32x16`
  (only one leaf is in frame);
* `DC_PRED`, one 32x16 luma transform (`TX_MODE_LARGEST`), WIDE_32 set,
  `is_long_side_dct = 1`, `intra_tx_type = H_DCT` (symbol 3);
* `eob = 2`, levels 1 at scan position 0 (`dc_sign`) and scan position 1
  (row 1, col 0 → `dc_sign_horz_vert`);
* chroma all-zero (16x8 residual).

Streams:

| file | what | gate expectation |
|---|---|---|
| `hv_pos.ivf` | horz-vert sign 0, normative ctx 0 | `--strict 1 --threads 1` decodes; IVF + Y4M MD5 pinned |
| `hv_zero.ivf` | eob = 1, DCT_DCT, no horz-vert symbol | `--strict 1 --threads 1` decodes; IVF + Y4M MD5 pinned |
| `hv_ctx1.ivf` | sign 1 coded with ctx 1 | **must be rejected** — the decoder derives ctx 0 |
| `hv_neg.ivf` | sign 1, normative ctx 0 | `--strict 0` decodes to a different pinned image |

Pins (2026-11-12, `tools/bin/dav2d-macos`, `f9dafbc`):

```
hv_pos   ivf ada97e3f12b8582c65e5c2ac0826b0ec  y4m 7a1349d9de949b949a05682fc606af86
hv_zero  ivf a398d9cfb44d6ed8ceed044121e6a65b  y4m 54d3959bbd3c80c05af8f900e417a069
hv_ctx1  ivf 75d363fcdc1ce67f53eae2445675340b  rejected
hv_neg   ivf 9275c76e44a3492229750c3737c53d6d  y4m ebfbfaa4b8ecc8eb96547ca22a23c986 (strict 0)
```

### dav2d strict limitation for negative 1D coefficients

This pinned dav2d build rejects **any negative H_DCT/V_DCT coefficient** with
`--strict 1` (`Error decoding frame: Invalid argument`, no data decoded), and
decodes the same stream fine with `--strict 0`.  The behaviour depends only on
the sign: flipping the `dc_sign_horz_vert` symbol to 1 makes strict fail, while
flipping the DC sign does not.  It reproduces on both 32x16 WIDE_32/H_DCT and
16x32 HIGH_32/V_DCT streams.  It is a decoder-side check, not a syntax or
prediction mismatch: the reference writer/reader round-trip and the `--strict 0`
pixels agree with the transform.  The gate therefore uses sign 0 for the
strict-1 positive path, pins the negative path at strict 0, and uses the
wrong-context control to prove the element and its context are consumed.

## 5. What is deliberately not here

* No production integration.  Wiring the DC branch, the corner (already
  wired), the sign dispatch, or `save_subPU` requires edits in
  `src/encoder/m1/m1_syntax.cpp` / `src/encoder/m1/m1_still.cpp` /
  `src/encoder/m2/m2_frame.cpp`, which this change is not allowed to make.
* No IBP corner form (`applyIbp == 1`) and no `enable_ibp` stream.
* No CE/CFL interaction: §7.13.2.11 is only reachable for a >32-sample chroma
  block, which current M1 streams cannot signal (audit I3/I19).
