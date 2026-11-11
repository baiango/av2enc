# AV2 quantizer matrices (QM) — CPU reference

Normative source: `spec/AV2_Spec_v1.0.0.pdf`.

| Spec | Item |
|---|---|
| §9.4.3 | `Qm_Offset[25]`, `Quantizer_Matrix[15][2][3600]` |
| §9.4.2 | informative 32x32 / 32x16 / 16x32 subsampling (used as a validation check) |
| §5.4.11 | `user_defined_qm()` |
| §5.13 | `quantizer_matrix_obu()` |
| §5.18.6.2 | `setup_qm_params()` |
| §7.14.4 | dequantization, `q2 = Round2(q * m, 5)` |

The HTML draft only says the arrays live in `qm_offset.h` / `quantizer_matrix.h`.
The PDF renders the numbers, so `generate_qm_tables.py` extracts and validates
them from the pinned PDF (`sha256 e9916f09…b533f`). It refuses to emit if the
shape, block layout, shared offsets, or the §9.4.2 derivation check fail.

## Files

| File | Purpose |
|---|---|
| `generate_qm_tables.py` | strict PDF extractor / validator |
| `qm_constants.h` | generated level constants and `kQmOffset` |
| `qm_tables.h` | generated `kQuantizerMatrix` (108,000 entries) |
| `qm_tables.json` | audit copy of the same numbers |
| `qm_core.h` | shared integer math (CPU **and** CUDA) |
| `qm.h` / `qm.cpp` | public API, dequant, QM OBU / frame-header writer |
| `qm_test.cpp` | 258k CPU checks |
| `build_test.sh` | builds/runs `qm_test` (optional ASan/UBSan) |
| `qm_gate.cpp` | bit-splices a QM OBU + `using_qmatrix=1` header into an M1 stream |
| `qm_gate.sh` | decoder gate (dav2d + avmdec) |

## Levels

* `0..14` are the 15 normative matrix levels (`NUM_CUSTOM_QMS = 15`).
* `15` is the §7.14.4 sentinel: `segLvl < NUM_CUSTOM_QMS` fails, so no QM is
  applied. Lossless segments set `SegQMLevel = 15`.
* `qm_y`, `qm_u`, `qm_v` are 4-bit (`f(4)`), so all 0..15 values are valid.

## Dequantization rules implemented

For a coded `txSz`:

```
tw = min(32, Tx_Width[txSz]);  th = min(32, Tx_Height[txSz])
pels = Tx_Width[txSz] * Tx_Height[txSz]
shift = (pels > 256) + (pels > 1024)
shift += allow_tcq && plane == 0 && !lossless && tx_type < IDTX && !use_fsc
dqDenom = 1 << shift

segLvl = (tw > 8 || th > 8) ? qm_y/qm_u/qm_v[0] : SegQMLevel[plane][segment]
useQm  = using_qmatrix && tx_type < IDTX && segLvl < 15
useUserQm = useQm && tw <= 8 && th <= 8 && QmDataPresent[segLvl]

q = (i == 0 && j == 0) ? get_dc_quant(plane) : get_ac_quant(plane)
m = useUserQm ? UserQm[segLvl][t][plane][...] 
              : Quantizer_Matrix[segLvl][plane > 0][Qm_Offset[txSz] + i*tw + j]
q2 = useQm ? Round2(q * m, 5) : q
dqHigh = abs(level) * q2
dq = Round2(dqHigh & 0xFFFFFF, 3)
dequant = sign(level) * (dq / dqDenom)
```

User-QM selection inside `tw <= 8 && th <= 8`:

| shape | matrix |
|---|---|
| `tw < th` | `UserQm[level][2]` (4x8), direct `[i][j]` |
| `tw > th` | `UserQm[level][1]` (8x4), direct `[i][j]` |
| `tw == th` | `UserQm[level][0]` (8x8), `[i*8/th][j*8/tw]` (4x4 subsamples 2x) |

This is the exact §7.14.4 `useUserQm` branch. Larger transforms always use the
default table, even when a user QM level is signalled.

`dequantize_block()` returns the §7.14.4 `dq2` values before the int16 clip
that `tx_variants::dequantize()` applies. Integration code that keeps that clip
should apply it after the QM-aware dequant (it is the same clip already used by
the M1/M2 recon paths).

## API

```cpp
#include "encoder/qm/qm.h"

av2enc::qm::QmFrameState state;      // seg_qm_level defaults to 15
state.using_qmatrix = true;
state.qm_y[0] = state.qm_u[0] = state.qm_v[0] = 3;
state.seg_qm_level[0][0] = 3;        // luma segment 0
state.qm_data_present[3] = true;     // a QM OBU supplied level 3
av2enc::qm::fill_user_qm_constant(&state.user_qm[3], 32);  // identity

av2enc::qm::QmDequantParams p;
p.state = &state;
p.tx_size = 1;                        // 8x8
p.plane = 0;
p.tx_type = 0;                        // DCT_DCT (< IDTX)
p.dc_quant = 64;
p.ac_quant = 132;

std::int32_t q2[1024];
std::int32_t dequant[1024];
av2enc::qm::build_q2_matrix(p, q2);
av2enc::qm::dequantize_block(levels, p, dequant);
```

Writing syntax:

```cpp
BitWriter writer;
std::string error;
av2enc::qm::write_setup_qm_params(writer, state, /*NumPlanes=*/3,
                                  /*segmentation_enabled=*/false, &error);

av2enc::qm::QmObuConfig obu;
obu.bit_map = 1U << 3;
obu.chroma_info_present = true;
obu.levels[3].is_default = false;
av2enc::qm::fill_user_qm_constant(&obu.levels[3].user, 32);
auto bytes = av2enc::qm::make_quantizer_matrix_obu(obu, &error);
```

`make_quantizer_matrix_obu()` returns a complete Annex B.2 OBU
(`leb128 length | obu_header | payload`) with type 22 and trailing bits.

## Tests

`./build_test.sh` (also `SANITIZE=1 ./build_test.sh`):

* generated-table shape, every `Qm_Offset`, golden luma/chroma entries;
* all 25 TX sizes × 15 levels × luma/chroma, q2 and dequant;
* `dqDenom` for 4x4…64x64, rectangular sizes, TCQ luma extra shift;
* `using_qmatrix` off, level 15, IDTX/1-D TX types, lossless segment;
* user-QM selection for 4x4/8x8/8x4/4x8 and fallback for 8x16/32x32;
* `user_defined_qm()` round-trip (symmetric 8x8, transposed 4x8, copies,
  random matrices) and `quantizer_matrix_obu()` round-trip;
* `setup_qm_params()` bit layouts (`uv_same`, separate U/V, mono, off).

Result on this host: **258,446 checks, 0 failures** (release and
ASan+UBSan).

`./qm_gate.sh`:

```
dav2d: using_qmatrix default matrices OK (b5ca… != b28f…)
dav2d: 64x64 default matrices OK (ee4c… != 41b3…)
avmdec: level 0 user identity == base; default QM differs
avmdec: level 5 user identity == base; default QM differs
avmdec: level 14 user identity == base; default QM differs
qm gate: OK
```

The gate uses an 8x8 4:2:0 picture, so every transform is 8x8 or 4x4 (the only
range where user QM data is used). An all-32 user matrix is the identity
(`Round2(q*32,5) == q`), so the avmdec MD5 must equal the base stream. The
`qm_is_default_flag = 1` variants decode and change pixels, proving QM scaling
is applied. Full details and the pinned-decoder limitation are in the gate
script header.

## Integration notes (no core files were edited)

Encoder header writer:

* `src/encoder/m1/m1_headers.h`: add `qm::QmFrameState qm;` to `M1Config`.
* `src/encoder/m1/m1_headers.cpp` — `write_frame_header_syntax()` currently
  writes `writer.put_bit(false); // using_qmatrix` (around line 177). Replace
  with `qm::write_setup_qm_params(writer, config.qm, 3, false, &error)`.
* `src/encoder/m1/m1_headers.cpp` — `assemble_m1_still_picture()` should insert
  `qm::make_quantizer_matrix_obu()` after the sequence header when a custom
  level is used. §7.3.8.9 requires the level to be available before the frame
  header that references it.
* Same change in `src/bitstream/m0_headers.cpp` (line ~310),
  `src/encoder/m2/m2_headers.cpp` (line ~284),
  `src/encoder/m2/post/tiled_frame.cpp` (line ~445),
  `src/encoder/m2/multiref/gate/compound_stream.cpp` (line ~234), and
  `src/encoder/pipeline/tiled_still_gate.cpp` (line ~263) if M0/M2/gates need
  QM. `src/encoder/m2/deltaq/m2_deltaq.cpp` hardcodes
  `using_qmatrix = false` and is the natural M2 state owner.
* `src/encoder/m1/m1_still.cpp`: populate `config.qm` (frame-level
  `qm_y/u/v[0]`, per-segment `SegQMLevel`, `QmDataPresent` for OBU levels) and
  make reconstruction use the QM-aware dequant below. Lossless segments must
  set `SegQMLevel = 15` (§7.14.4 and §5.18.2).

Dequantization call sites (all currently hardcode q2 = 528):

* `src/encoder/tx_variants/tx_variants.cpp` — `dequantize()` line ~258
  (`kQuantStep`). Add an overload taking `qm::QmDequantParams` (or a q2
  matrix) and call `core::dequant_coeff()`.
* `src/encoder/m1/m1_still.cpp` — `dequant_inverse_qindex()` line ~767 is the
  per-frame-step dequant path; replace its inner loop with a q2 matrix. The
  normal path at line ~928 and
  `src/encoder/m1/transform_stream/transform_stream.cpp` line ~257 call
  `txv::inverse_transform()` and inherit the QM-aware dequant once the
  overload is selected. `transform_stream_quality.cpp` line ~153 is the same.
* `src/encoder/tx_type/tx_type.cpp` line ~376 builds candidate recon the same
  way.
* `src/encoder/m2/residual/inter_residual.cpp` line ~338 and
  `src/encoder/rdoq/m1_rdoq_quality.cpp` lines ~461/549 also call
  `txv::inverse_transform()` for recon/cost; they need the QM-aware overload
  when QM is enabled.
* M2: `src/encoder/m2/m2_frame.cpp` lines ~206–227 computes its own §7.14.4
  dequant with a per-frame `quant_step`; insert `qm::build_q2_matrix()` there
  and use the per-coefficient q2 in the loop.
* CUDA M1: `src/cuda/m1/tx/m1_tx.cu` `dequant_dev()` line ~60 and
  `src/cuda/m1/hbd/hbd_tx.cu` `dequant_dev()` line ~81. Both are the same
  `abs(q)*528` shape; `src/cuda/qm/` has a batch kernel that can replace them,
  or pass a q2 matrix into the existing kernels.
* CUDA transform variants: `src/cuda/tx_variants/tx_variants.cu`
  `dequant_dev()` line ~56 and `txv_dequantize()` line ~432.
* `src/cuda/transform.cu` / `src/encoder/m0_still.cpp`: M0 reference only.

Encoder-side (forward) quantization:

* QM is a dequant scaling. To keep encoder recon equal to decoder recon, the
  forward quantizer should use the same per-coefficient q2 (or an equivalent
  weighting by `m`) and reconstruction must call the QM-aware dequant. The
  existing `txv::TxParams::quant_factor` is a single scalar; it is not
  QM-aware.
* RDO/RDOQ lambda and rate models are QM-agnostic today; pass the real
  per-coefficient q2/quant step once the dequant overload is wired.

Decoder-side (not implemented here):

* There is no QM OBU parser in `src/bitstream/`. A decoder/validator path needs
  `quantizer_matrix_obu()`, `QmProtected/QmSeen/QmDataPresent` reset rules
  (§6.12, §7.3.8.9), and `setup_qm_params()` state. This module only writes
  the syntax; `QmFrameState` is the state a parser would populate.
* `dav2d` (pinned) rejects **all** QM OBUs ("QM OBUs with user-defined QMs are
  currently unsupported"), but accepts `using_qmatrix = 1` with the default
  matrices and no QM OBU. `avmdec` accepts both default and user-defined QM
  OBUs; use it for the user-QM gate.

## Limitations

* 8-bit lossy path only. No high-bit-depth clip, no lossless DPCM/IDTX, no FSC,
  no secondary transform (`sec_tx_type`) interaction.
* No QM OBU listener/parser and no QM OBU availability state machine.
* No forward-quantizer QM weighting (documented above); this module is the
  normative dequant + syntax writer half.
* User QM data is meaningful only for transforms with `tw <= 8 && th <= 8`,
  exactly as §7.14.4 specifies.
