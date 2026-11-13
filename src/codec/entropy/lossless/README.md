# encoder/lossless — AV2 Q=0 lossless stills, DPCM, FSC

From-scratch CPU reference for AV2 v1.0.0 lossless still-picture coding.  The
normative source is `spec/AV2_Spec_v1.0.0.pdf` (commit `38f28e7`); no encoder
code from AVM, SVT-AV1/AV2, clave, av2-turbo or another codec was consulted.
`dav2d` and `avmdec` are used only as black-box validators.

This module is self-contained and deliberately **not** wired into
`CMakeLists.txt`, `src/main.cpp`, or the M1/M2 writers.  `build_and_test.sh`
compiles the same sources a core target would compile.

## Why a separate module

A lossless frame changes the syntax and reconstruction path in ways a lossy
writer cannot express with a config bit:

| Area | Lossless behaviour |
|---|---|
| Frame header | `CodedLossless = 1`; deblock/GDF/CDEF/LR/CCSO params return before writing any bits; `read_tx_mode()` returns `ONLY_4X4` with no bit |
| Quantization | `base_q_idx = 0`; `get_q(0, delta <= 0) = 64`; dequant scale is exactly 8 |
| Transform | `DCT_DCT` lossless uses the reversible 4x4 WHT (§7.15.2.2); `IDTX` lossless uses `Dequant >> 3` (§7.15.4) |
| Mode info | `use_dpcm_y/uv`, `dpcm_mode_y/uv`, `fsc_mode`; `lossless_tx_size` |
| Residual | DPCM first-difference residual; FSC `bob`/`coeff_base_bob`/`coeff_base_idtx`/`coeff_br_idtx`/`idtx_sign` path |
| Transform type | lossless intra codes no intra TX type; FSC forces IDTX |

## API (`lossless.h`)

```cpp
namespace av2enc::lossless {

struct LosslessInputs { int base_q_idx; bool delta_q_present;
                        int delta_q_y_dc, base_y_dc_delta_q, ...;
                        bool alt_q_enabled[16]; int alt_q[16];
                        int num_segments; };
struct LosslessFlags { bool coded_lossless, has_lossless_segment;
                       bool lossless[16]; };

LosslessFlags compute_lossless_flags(const LosslessInputs&);
int qindex_for_segment(const LosslessInputs&, int segment_id);

// Reversible helpers (no bitstream state):
void inverse_wht4(const int in[4], int shift, int out[4]);
void forward_wht4(const int in[4], int out[4]);
void forward_lossless_4x4(const int residual[16], int levels[16]);
void inverse_lossless_4x4(const int levels[16], int residual[16]);
void dpcm_difference_4x4/8x8(...);
void dpcm_cumsum_4x4/8x8(...);

enum class FscMode { kOff, kAuto, kForce };
struct Options { int sb_size = 64; int seq_level_idx = 31;
                 bool adaptive_cdf = false; bool enable_dpcm = true;
                 FscMode fsc = FscMode::kOff;
                 bool enable_intra_edge_filter = false;
                 bool verify_exact = true; };
struct Stats { int luma_blocks, fsc_blocks, dpcm_y_blocks, dpcm_uv_blocks;
               uint32_t tile_bytes, frame_bytes;
               uint64_t max_abs_recon_error; bool exact; };

bool encode_still(const StillPicture&, const Options&,
                  EncodeResult&, std::string& error);
bool decode_temporal_unit(std::span<const uint8_t>,
                          DecodeResult&, std::string& error);
}
```

`encode_still()` returns the Annex-B temporal unit (TD + sequence header + CLK)
and the decoder-path reconstruction.  `decode_temporal_unit()` is a true syntax
reader: it parses the OBUs, frame header, partition tree, mode info,
coefficients, and reconstructs the frame with the normative §8.2 decoder state
machine.  It does not reuse the encoder's decisions.

`Stats::exact` is true only when every reconstructed sample matches the source.
`Stats::max_abs_recon_error` is the maximum absolute sample error (0 for a
correct lossless encode).

## Scope

Supported and gated:

* 8-bit 4:2:0, one tile, one still frame, 64x64 superblocks.
* Width/height positive multiples of 8.
* 8x8 luma coding blocks.  Non-FSC blocks use four 4x4 `DCT_DCT` transforms
  per luma block and one 4x4 transform per chroma plane.  FSC blocks use one
  8x8 `IDTX` luma transform (`lossless_tx_size = 1`) and one 4x4 `IDTX`
  chroma transform per plane.
* Partition trees with `PARTITION_NONE` at 8x8 and `HORZ`/`VERT` above it,
  including the §5.20.3.2 implied-partition rule at frame boundaries.
* DC, V-DPCM, and H-DPCM intra prediction.  The mode search is a simple
  sum-of-absolute-residual choice; correctness, not RD optimality, is the goal.
* FSC luma coefficient coding with the `bob`/IDTX path and all §8.3.2
  contexts.  `FscMode::kForce` is used by the gate; `kAuto` picks FSC only when
  the residual has a non-zero-prefix scan pattern that makes the FSC suffix
  cheaper.
* `disable_cdf_update = 1` by default; `adaptive_cdf = true` enables the
  normative §8.2.6 adaptation and is gated with both decoders.
* §5.18.2 `CodedLossless`/`LosslessArray` derivation from base Q, delta-Q,
  per-plane deltas, and per-segment `SEG_LVL_ALT_Q`.

Not implemented (blocked by scope, not by missing tables):

* Segmentation/delta-Q **syntax** writing.  The encoder emits
  `segmentation_enabled = 0`, `delta_q_present = 0`, one segment.  The flag
  derivation API accepts per-segment `ALT_Q` for callers and tests.
* 10-bit/12-bit, 4:2:2, 4:4:4, monochrome.
* 4x4 coding blocks, 128x128/256x256 superblocks, multiple tiles.
* Inter lossless (`lossless_inter_tx_type`, the inter branch of
  `lossless_tx_size`, lossless inter reconstruction).  This is a still-picture
  module; inter lossless needs the M2 inter syntax and is not attempted here.
* Palette, IntraBC, CfL, MHCCP, IBP, DIP, MRL, IST/DDT, TCQ, parity hiding,
  QM, multiple frames / frame-end CDF update.

## Normative details that cost debugging time

1. **The WHT is not self-inverse when the `>>1` lifting step sees an odd
   value.**  `forward_wht4()` is the algebraic inverse of the decoder's lifting
   steps, and `forward_lossless_4x4()` applies the inverse in the opposite
   order (columns first, then rows) because the decoder does rows then columns.
   Using `inverse_wht4(..., 0)` as the forward transform produces off-by-one
   coefficients and a non-exact reconstruction.
2. **The partition context grids store Mi log2 values, not block-size enum
   values.**  The sentinel is `BLOCK_256X256` (enum 18), whose
   `Mi_Width_Log2`/`Mi_Height_Log2` is 6.  Storing the sentinel as a raw `6`
   and then indexing `Mi_*_Log2[6]` silently reads `16x16` and makes the
   partition CDFs diverge from the decoder after the first row/column.
3. **FSC has two passes.**  The first pass codes all `coeff_base_bob` /
   `coeff_base_idtx` / `coeff_br_idtx` levels; the second pass codes
   `idtx_sign` and `read_quant` in scan order from `c = 0`.  Interleaving the
   two makes streams decode only when the first non-zero coefficient is the
   last coefficient in the scan.
4. **Lossless 4x4 DCT_DCT dequant is exactly `level * 8`.**  The inverse WHT
   row pass uses `>>3`, so the encoder-side forward transform must produce the
   level array, not a scaled residual.
5. **DPCM stores the first difference**, because the decoder applies a
   cumulative sum after the inverse transform (§7.15.4), not before it.

## Tables

`lossless_cdf_tables.h` is generated from `spec/tables/default_cdf.json` by
`generate_cdf_tables.py` and contains only the lossless-unique arrays:

* `Default_Fsc_Mode_Cdf`
* `Default_Use_Dpcm_Y_Cdf`, `Default_Dpcm_Mode_Y_Cdf`
* `Default_Use_Dpcm_UV_Cdf`, `Default_Dpcm_Mode_UV_Cdf`
* `Default_Lossless_Tx_Size_Cdf`

All coefficient CDFs (including `Default_Coeff_Base_Bob_Cdf`,
`Default_Coeff_Base_Idtx_Cdf`, `Default_Coeff_Br_Idtx_Cdf`,
`Default_Idtx_Sign_Cdf`, `Default_Lossless_*`) come from the existing
`src/encoder/m1/qctx/qctx_cdf_tables.h` slice 0 for `base_q_idx = 0`.  The FSC
and DPCM tables were present in `default_cdf.json`, so no tool is blocked by a
missing normative table.

## Tests

```sh
./src/encoder/lossless/build_and_test.sh
SANITIZE=1 WERROR=1 ./src/encoder/lossless/build_and_test.sh
```

`lossless_test.cpp` covers:

* `compute_lossless_flags()`: base Q 0/positive, `delta_q_present`, positive and
  negative per-plane deltas, per-segment `ALT_Q`, mixed lossless/lossy segments.
* exact inverse of `inverse_wht4`/`forward_wht4` and the 4x4 lossless WHT pair
  over thousands of random vectors;
* DPCM first-difference/cumulative-sum round trips for V and H;
* full encoder → reader round trips over sizes 8x8, 16x16, 32x32, 64x64,
  136x88; flat/gradient/random/screen patterns; FSC off/force; DPCM on/off;
* adaptive-CDF round trip;
* bitstream determinism, block statistics, and malformed/truncated input
  rejection.

Each round trip checks the encoder reconstruction, the reader reconstruction,
and every sample of Y/U/V against the source (0-LSB requirement).

`lossless_gate.cpp` is the external gate.  It writes an IVF for each case and
requires the pinned decoders to decode the stream and reproduce the source YUV
exactly.  It runs both `dav2d-macos` and `avmdec` when available.  Cases:

* 64x64 screen content, FSC off/force;
* 136x88 gradient, FSC off/force;
* 136x88 random, FSC force;
* 64x64 screen content, FSC auto;
* 640x360 gradient FSC off and 640x360 random FSC force (multi-superblock);
* 64x64 random, adaptive CDF;
* the real `tests/data/basic_64x64_1f.y4m` frame, FSC off/force.

Observed on the dev host (dav2d `f9dafbc`, avmdec from `tools/bin/avm-macos`):

```
lossless_test: 72113 checks passed
lossless_gate: 22 decoder checks passed
PASS: lossless CPU tests + external decoder gate
```

`dav2d --strict 1` and `avmdec` both decode every gated stream, including FSC
and adaptive CDF, and the decoded I420 bytes equal the source exactly.

## Integration notes

The module is not in `CMakeLists.txt` and no call sites were added, because
this workstream was told not to edit the core build or `src/main.cpp`.  To
integrate later:

* add `src/encoder/lossless/lossless.cpp` (and `lossless_test.cpp` as a test)
  to `av2enc_core`; its dependencies are already in the archive:
  `bitstream/bitwriter.cpp`, `entropy/av2/av2_entropy.cpp`,
  `encoder/intra_modes/intra_modes.cpp`, `encoder/partition/partition.cpp`;
* call `encode_still()` from a CLI flag (for example `--lossless`) and write
  the returned temporal unit through the existing `IvfWriter`;
* call `decode_temporal_unit()` in the CPU oracle/gate path, or compare its
  reconstruction against the source.

The external gate does not need a core edit: `build_and_test.sh` compiles the
module and the gate standalone.

### CUDA block stage (`--lossless --still-gpu`)

`Options::precomputed` accepts an array of `BlockCoding` (one per 8×8 luma
block, raster order) produced by
`cuda::still_lossless::compute_block_coding()`.  The writer then uses those
modes and syntax levels instead of running `choose_plan()` and the forward
transform; prediction, the §8.2 entropy write and the decoder-side inverse
(and therefore the exactness check) are unchanged.  Note the reference's
chroma search quirk that the device stage reproduces: its V/H candidates copy
`best` and change only the DPCM flags, so the cost is evaluated with the
incumbent `uv_mode` and the candidate mode is assigned after the comparison.

## Quality exactness

For every gated lossless case the reconstruction is **0 LSB**:

* `encode_still()` runs the decoder path (`inverse WHT` or IDTX, optional DPCM
  cumulative sum, add prediction, `Clip1`) and reports
  `max_abs_recon_error = 0`, `exact = true`;
* `decode_temporal_unit()` independently parses the stream and reproduces the
  same samples;
* `dav2d` and `avmdec` produce byte-identical I420 output.

The only required table source is the committed
`spec/tables/default_cdf.json`.  No lossless/DPCM/FSC table is blocked or
missing.
