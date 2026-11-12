# M2 inter residual (`src/encoder/m2/residual/`)

Integration-ready inter residual path for the M2 P-frame encoder: transform /
quantise / dequantise / inverse for inter residuals, the inter TX size and TX
type rules, the coefficient entropy contexts, and a decision API that turns
"inter prediction + source" into `levels` / `eob` / reconstruction with an
explicit skip-vs-residual cost.  CPU reference plus a GPU-ready batch
interface and device-emulation parity test.

This directory is self-contained.  It does not edit the M2 core, the M1 path,
`src/cuda/m1/`, `src/main.cpp` or `CMakeLists.txt`; the decoder gate applies its
three-edit patch to a **copy** of `video_syntax.cpp` in the build directory.

## Status

| Piece | State |
|---|---|
| Decision API (pred + source -> levels/eob/recon + skip cost) | done, unit-tested |
| Transform / quant / dequant / inverse | reuses `src/encoder/tx_variants/`, no math duplicated |
| RDOQ | reuses `src/encoder/rdoq/` (`rdoq::av2::Model` + trellis) |
| Inter TX size derivation | done; M2 signals `TX_MODE_LARGEST`, so no bits |
| Inter TX type sets / symbols / contexts | done (`inter_tx_type.*`) |
| Coefficient entropy contexts (base, BR, EOB, DC sign, CBF) | done (`inter_coeff_syntax.*`) |
| Decoder gate | done: non-skip inter streams decode sample-exactly with `dav2d` |
| CPU vs GPU | batch interface + device emulation parity (`inter_residual_device.cuh`) |
| Real CUDA kernel | not built here (no CUDA toolchain on this host); launch plan documented |

## What the entropy layer does differently from the M1/M2 intra writer

The M2 writer already codes intra residuals (and, since the parallel M2 update,
DCT-only inter residuals).  The inter path differs from intra in exactly four
places, all from the spec:

| Syntax | intra | inter | spec |
|---|---|---|---|
| `all_zero` / TXB skip CDF | `TileTxbSkipCdf[0][txSzCtx][ctx]` | `TileTxbSkipCdf[1][txSzCtx][ctx]` | §8.3.2 `all_zero` |
| luma EOB CDF context | `eobCtx = 0` | `eobCtx = 1` (chroma stays 2) | §5.20.7.27 |
| luma `transform_type()` | `intra_tx_type` (+ `eob == 1 => DCT_DCT`) | `inter_tx_type` (+ `_offset` for INTER_1/2), coded even at `eob == 1` | §5.20.8.2 |
| chroma TX type | derived | DCT_DCT (`enable_chroma_dctonly = 1`) | §5.20.7.29 |

Everything else is the M1 derivation and is shared verbatim: `coeff_base` /
`coeff_base_lf` / `coeff_base_eob*` contexts from neighbour levels, `coeff_br`,
`read_quant` escapes, `culLevel`/`dcCategory` context updates, and the DC sign
rule (the `dc_sign` CDF is used for the **luma DC only**; every other DC and all
AC signs are raw `sign_bit` literals — §5.20.7.27, same as the M1 writer).

This is verified two ways:

* **byte parity**: with DCT_DCT only, a frame assembled through
  `residual::write_inter_tx_type` + the M2 writer is byte-identical to the same
  frame assembled by the stock M2 writer (`gate/build_gate.sh`, 64x64 and
  1280x720);
* **decoder parity**: streams that use non-DCT inter TX types decode
  sample-exactly with the pinned `dav2d`.

## TX sizes (`inter_tx_sizes`)

M2 writes `tx_mode_select = 0` (TX_MODE_LARGEST).  In `read_block_tx_size()`
the `TX_MODE_SELECT` branch is entered only when `UIntraTxPartSet` allows it, so
a non-skip inter block codes **no** TX size syntax at all:
`read_tx_size(allowSelect = 1)` returns 0 and `LumaTxSizes` is filled with
`Max_Tx_Size_Rect[MiSize]`.  Therefore every plane uses exactly one transform
block:

```
luma   TX size = tx_size_index(block_w, block_h)
chroma TX size = tx_size_index(chroma_w, chroma_h)   // 4:2:0 residual size
```

`TX_MODE_SELECT` (luma TX partitions, `LumaTxMiddle`/`LumaTxScanOrder`) is not
modelled; `inter_coeff_syntax` validates that the declared TX size equals the
TX_MODE_LARGEST geometry and rejects anything else.

## TX types (`inter_tx_type.*`)

`get_tx_set(txSz, 0)` for `is_inter = 1`, `reduced_tx_set = 0`:

| TX size | set | codable types after filtering |
|---|---|---|
| 4x4, 4x8, 8x4, 8x8, 8x16, 16x8, 4x16, 16x4 | `TX_SET_INTER_1` | 7 |
| 16x16 | `TX_SET_INTER_2` | 7 |
| 32x32 | `TX_SET_DCT_IDTX` | 1 (DCT_DCT) |
| 16x32, 8x32, 4x32, 32x16, 32x8, 32x4 | `TX_SET_WIDE_32` / `HIGH_32` | 3 (+ `is_long_side_dct`) |
| 16x64, 8x64, 4x64, 64x16, 64x8, 64x4 | `TX_SET_WIDE_64` / `HIGH_64` | 3 |
| 32x64, 64x32, 64x64 | `TX_SET_DCTONLY` | 1, no symbol |

Filters applied to the spec tables:

* `enable_inter_ddt = 0` (M2's sequence header): every candidate with a 1D
  identity component (IDTX, V/H_DCT, V/H_ADST, V/H_FLIPADST) is removed.  The
  decoder *would* accept them, but §6.4.8 says DDT is not allowed.
* `Tx_Type_In_Set_Inter` aliasing: a type outside the set is reconstructed as
  DCT_DCT by `compute_tx_type()`, so `TX_SET_INTER_2`'s `ADST_FLIPADST` is not
  codable even though `Tx_Type_Inter_Inv_Set2` lists it.
* `FLIPADST_ADST` and `ADST_FLIPADST` are withheld because
  `src/encoder/tx_variants/tx_variants.cpp` maps their 1D transforms to each
  other (`kTransform1dType[7]`/`[8]`), which breaks decoder bit-exactness.
  Verified per type with `dav2d`: types 0..6 decode exactly, type 7 does not.
  Fixing that two-row swap in the shared component re-enables both (the unit
  test asserts they are currently absent).

Contexts are spec-exact and deliberately **not** merged: `inter_tx_type` uses
`min(w, 32) + min(h, 32) - 4` while `inter_tx_type_offset` uses `w + h - 4`
(§8.3.2).

## API

```cpp
#include "encoder/m2/residual/inter_residual.h"

residual::BlockGeometry geom{mi_row, mi_col, 64, 64, 32, 32};
residual::BlockInput input;
input.geometry = geom;
input.source[0] = {/* 64*64 luma samples */};
input.prediction[0] = {/* inter prediction */};
// ... planes 1, 2 at 32x32

residual::DecisionOptions options;   // base_q_idx, lambda, tx-type search, rdoq
residual::BlockDecision decision;
std::string error;
if (!residual::decide_inter_block(input, options, decision, error)) { ... }

if (!decision.skip) {
  residual::InterResidualBlock syntax;      // tx_size/tx_type/eob per plane
  std::vector<std::int32_t> levels;         // scan order, Y then U then V
  residual::make_syntax_block(geom, decision, &syntax, &levels, error);
  // -> av2enc::video::syntax::BlockSyntax{tx_size[], eob[], levels_offset} + the levels
  // -> residual::write_inter_residual(...) or the M2 writer
}
```

`BlockDecision` reports `distortion`, `rate_bits` and `cost` for the residual,
and `skip_distortion` / `skip_rate_bits` / `skip_cost`, so the caller can
compare against other inter candidates (different MVs, intra fallback).
`decision.zero_residual` marks a residual that quantised to all-zero (forced to
skip).  `decision.plane[p].recon` is the decoder-exact reconstruction of the
block (`clip(pred + inverse_transform(levels))`), so the caller writes it
straight into the reference frame; for a skip decision it is the prediction
(this holds for `allow_residual = false` too).

Rate accounting: per-plane type selection uses a CDF-based estimate; the final
skip-vs-residual comparison measures the block's **own symbol cost** by running
`write_inter_residual` with a `-log2 P` counting sink over a neutral context
(`measure_block_bits`).  This matters: with the estimate alone the decision
coded ~2x more residuals than the same decision with measured bits.

Batch form (GPU-ready POD, element offsets only):

```cpp
residual::BatchInput batch;    // spans: source[3], prediction[3], blocks
residual::BatchOutput out;     // levels, recon, eob, tx_size, tx_type, skip, cost
residual::decide_inter_batch(batch, out, &error);
```

## GPU path

`inter_residual_device.cuh` holds the device-side integer work and is compiled
twice: by nvcc, and — with the CUDA keywords stubbed — by
`inter_residual_emulation_test.cpp` on the host.  The emulation test compares
residual gather, EOB, the full §8.3.2 coefficient plan (base/BR context, escape
flags, DC-sign context, `codes_br`, `pre_level`) and the skip rule against the
CPU reference over every 4..32 transform size, both 2D classes, all three
planes and randomized sparse/large-magnitude level blocks.

The transforms are not reimplemented: a real kernel calls
`src/cuda/tx_variants/` (`txv_forward_transform`, `txv_quantize`,
`txv_inverse_transform`), whose CPU/GPU parity is gated by that component.  The
launch sequence is in `docs/INTEGRATION.md`.

## Decoder gate

```sh
./gate/build_gate.sh --quick                     # 64x64 synthetic, depth 0
./gate/build_gate.sh --depth 3 --frames 3        # 64x64, mixed block sizes
./gate/build_gate.sh --input testdata/shine_1280x720_60f.y4m --frames 3 --depth 4
```

The gate encodes the intra keyframe with the stock `encode_video_sequence()` and
every P frame with this component (GLOBALMV zero MV, inter only — it isolates
the residual path from ME/merge/intra fallback), assembles the tile with a
patched *copy* of the M2 writer, writes an IVF, decodes it with `dav2d` and
compares Y/U/V sample by sample against the encoder reconstruction.  It also
builds an unpatched binary and byte-compares the DCT-only streams.

Measured (macOS host, `dav2d-macos`, `base_q_idx = 90`):

```
64x64 synthetic, depth 3, 3 frames
  DCT-only patched vs stock M2 writer tile: identical bytes
  DCT:     P bytes 1379 + 1456   PSNR 42.5753
  all TX:  P bytes 1364 + 1452   PSNR 42.6295   (5 non-DCT leaves)  decode exact

1280x720 shine, depth 0, 3 frames (240 inter leaves/frame, incl. 64x16 edges)
  skip only:            P bytes 502    PSNR 36.4047
  inter residual (DCT): P bytes 14424  PSNR 36.4783
  all TX types:         same as DCT at depth 0 (64x64 => TX_SET_DCTONLY)
  DCT-only patched vs stock writer: identical bytes; decode exact

1280x720 shine, depth 4, 2 frames
  DCT:     P bytes 6633  PSNR 47.8278
  all TX:  P bytes 6745  PSNR 47.8783  (76 non-DCT leaves of 111 residual)
  decode exact

1280x720 shine, depth 4, 3 frames (1211-1211 inter leaves/frame, 4x4..64x64)
  all TX:  P bytes 11499 PSNR 47.6043  (145 non-DCT leaves of 203 residual)
  decode exact
```

The zero-MV P frames are a stress harness, not a rate comparison against the M2
core (which uses ME, NEWMV and intra fallback: 2230 bytes / PSNR 36.57 on the
same clip).

## Limitations

* `TX_MODE_SELECT` (luma TX partitions) is not modelled; M2 pins
  `TX_MODE_LARGEST`.
* `enable_chroma_dctonly = 1` is assumed.  With chroma types enabled the chroma
  TX type derivation and its 1D coefficient classes (`TX_CLASS_HORIZ/VERT`:
  different scan, `coeff_base` context set, `coeff_base_idtx`, DC-sign variants)
  would be needed.
* No parity hiding, TCQ, CC TX, FSC, IST: M2 disables all of them, so the
  contexts are 2D-only.
* The component keeps its own copy of the §9.3 coefficient CDFs (the M2 writer's
  `CdfState` is private).  This is exact because `disable_cdf_update = 1`;
  adaptive CDFs need the two states merged (see `docs/INTEGRATION.md`).
* `src/encoder/tx_variants/` swaps FLIPADST_ADST/ADST_FLIPADST 1D types (see
  above).  Not this directory's file; reported, not patched.
* 8-bit 4:2:0 only.
* `AV2ENC_VIDEO_RESIDUAL_TX_TYPE=<n>` forces one luma TX type (diagnostic knob used
  to isolate a single transform against `dav2d`; not a decision input).

## Files

```text
inter_residual.h/.cpp        decision API, TX sizes, batch interface, rate
inter_tx_type.h/.cpp         inter TX sets, candidates, symbols, contexts
inter_coeff_syntax.h/.cpp    entropy writer + coefficient plan (§5.20.7.27)
inter_coeff_plan_types.h     POD shared with the device code
video_residual_cdf.h            generated §9.3 inter-TX-type + EOB16/32 CDFs
inter_residual_device.cuh    device-side integer helpers (shared with nvcc)
inter_residual_test.cpp      unit tests (475 checks)
inter_residual_emulation_test.cpp  CPU vs device-emulation parity (469 checks)
build_and_test.sh            standalone build + tests (+ --gate)
gate/                        patcher, gate driver, gate build script
docs/INTEGRATION.md          exact call sites and GPU launch plan
docs/video_syntax_inter.patch   generated patch against the current M2 writer
tools/                       spec section extractor, CDF generator
```
