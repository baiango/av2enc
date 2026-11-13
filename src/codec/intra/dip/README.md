# AV2 DIP (data-driven intra prediction) — CPU reference

Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`). Implemented clauses:

| Clause | Contents |
|---|---|
| §5.4.5 | `sequence_intra_config()` `enable_dip f(1)` |
| §5.20.7.9 | `dip_mode_info()`: `use_dip S()`, `dip_transpose L(1)`, `dip_mode S()` |
| §6.19.7.8 | DIP mode info semantics |
| §7.13.2.1 | `AboveRow` / `LeftCol` with `useDip` (extra `w/4`, `h/4` samples) |
| §7.13.2.3 | data driven intra prediction process |
| §7.13.2.4 | 11-feature extraction and `dip_avg` |
| §7.13.2.5 | `Dip_Weights[dip_mode][i*8+j][k]` Q10 transform |
| §7.13.2.6 | 8x8 -> `w x h` resampling |
| §8.3.1/.2 | `TileUseDipCdf[ctx]`, `TileDipModeCdf` |
| §9.3 | `Default_Use_Dip_Cdf`, `Default_Dip_Mode_Cdf`, `Dip_Weights` |

## Normative tables

All DIP tables are present in the pinned PDF. Nothing was guessed.

| Table | Location | Status |
|---|---|---|
| `Dip_Weights[6][64][11]` | PDF pages 1095–1100, immediately after the Wiener tables | present, extracted verbatim |
| `Default_Use_Dip_Cdf[3][3]` | §9.3 `Default_Use_Dip_Cdf[DIP_CTXS][2+1]` | present |
| `Default_Dip_Mode_Cdf[6+1]` | §9.3 | present |

`Dip_Weights` is the only table this module generates. Its SHA-256 over the
comma-separated 4224 values (mode major, 64 positions, 11 features) is
`e5deedf1ed42fb944e233733c735593b04a3ae7116b51045bc692a11f53d04f7`;
the sum is `393140`. `generate_dip_weights.py` re-extracts the table from
`spec/AV2_Spec_v1.0.0.pdf` and refuses to write anything unless the value count,
checksum, sum, and `int16` range checks pass. The generated initializer body
`dip_weights_data.inc` is included by both the CPU table and the CUDA device
table, so the two cannot drift.

The HTML specification omits §9.3/§9.6 numeric tables (it says they live in
header files); the PDF is the source, as for the other generated tables in this
repository.

## Availability and edge samples

`dip_mode_info()` is present exactly when

```
enable_dip && YMode == DC_PRED && PaletteSizeY == 0 &&
Block_Width > 4 && Block_Height > 4 &&
Block_Width * Block_Height >= 128 &&
TreeType != CHROMA_PART
```

The decode process then sets `useDip = plane == 0 && use_dip`. DIP is therefore
luma-only even if `use_dip` was signalled for a block that also has chroma.

With the AV2 transform extents 4/8/16/32/64 this leaves 15 valid sizes:
`8x16, 8x32, 8x64, 16x8, 16x16, 16x32, 16x64, 32x8, 32x16, 32x32, 32x64,
64x8, 64x16, 64x32, 64x64`. `8x8` fails the area check; any 4-sample axis fails
the width/height check.

Edge preparation follows §7.13.2.1 with `MrlIndex = 0` (DIP requires
`DC_PRED`, and MRL requires a directional mode) and the DIP counts

```
numAboveNeeded = w + (w >> 2)
numLeftNeeded  = h + (h >> 2)
```

The extra quarter is required by the 5-tap `dip_avg` downsampler: it reads
`5 * (n >> 2)` samples. Existing non-DIP `intra::prepare_edges` fills
`w + h` samples, which is insufficient for wide/short or tall/narrow DIP blocks
(e.g. 64x8 needs 80 above samples but `w+h = 72`). Integration must use
`dip::prepare_edges`.

## API (`dip.h`)

```cpp
using av2enc::dip::Block;      // recon plane + x/y/w/h/max_x/max_y/availability
using av2enc::dip::Edges;      // AboveRow/LeftCol, bit depth, w/h
using av2enc::dip::Params;     // mode 0..5, transpose

Status prepare_edges(const Block&, int bit_depth, Edges&);
Status extract_features(const Edges&, bool transpose,
                        std::array<int, 11>& f);
void   transform(const std::array<int, 11>& f, int mode, bool transpose,
                 int bit_depth, std::array<uint16_t, 64>& dip_pred);
void   resample(const Edges&, const std::array<uint16_t, 64>& dip_pred,
                uint16_t* dst, std::size_t stride);

Status predict(const Edges&, const Params&, uint16_t* dst, std::size_t stride);
Status predict8(const Edges&, const Params&, uint8_t* dst, std::size_t stride);
Status predict_block(const Block&, int bit_depth, const Params&, uint16_t*, std::size_t);
Status predict_block8(const Block&, int bit_depth, const Params&, uint8_t*, std::size_t);
```

The CPU reference supports 8..12-bit samples (`PlaneView::from_u8` /
`from_u16`). `predict8` is the 8-bit M1 integration wrapper.

### Decision API

```cpp
DecisionParams { enable_dip, y_mode, palette_size_y, tree_is_chroma_part,
                 metric = kSad | kSse };
Decision { available, mode, transpose, cost };
Status decide(const Edges&, const uint8_t* src, std::size_t src_stride,
              const DecisionParams&, Decision&);
```

`decide()` checks the exact syntax predicate, then evaluates the normative
candidates `(mode 0..5) x (transpose 0/1)` and returns the lowest SAD or SSE.
Ties are broken deterministically by lower mode, then `transpose == false`.
It returns `kUnavailable` without evaluating candidates when DIP cannot be
signalled. This is the hook for luma DC RDO: DIP is an alternative to ordinary
DC prediction at the same `YMode == DC_PRED`, not a new YMode.

### Syntax codec

`dip_syntax.h` writes/reads the exact syntax structure using the repository's
§8.2 range coder:

```cpp
DipCdfState cdfs;                 // per-tile copies of the two Tile*Cdf arrays
write_dip_mode_info(context, info, encoder, cdfs, &error);
read_dip_mode_info (context, decoder, cdfs, info, &error);
```

Order and descriptors are `use_dip S()`, then if set `dip_transpose L(1)`,
`dip_mode S()`. Use-dip context is the number of above/left neighbours whose
`UseDip` flag is 1 (`DIP_CTXS = 3`). Unsignalled fields are inferred to
`use_dip = 0`, `dip_transpose = 0`, `dip_mode = 0`.

## Tests

```
src/encoder/dip/build_and_test.sh           # CPU reference + real Y4M blocks
SANITIZE=1 src/encoder/dip/build_and_test.sh
```

The test contains an independent, literal transcription of the pipeline and
never uses the production implementation to generate expectations. Coverage:

- the table checksum, spot values, and range guards;
- all 25 combinations of 4/8/16/32/64 for availability, and all 15 DIP sizes;
- every `have_left`/`have_above` combination, including both-unavailable filler
  values 127/129/128;
- partial right/bottom blocks via reduced `max_x`/`max_y`;
- all 6 modes x 2 transpose values;
- 8-, 10-, and 12-bit samples;
- real luma blocks from `testdata/shine_256x256_1f.y4m` (2880 predictions);
- decision selection and syntax predicate/CDF/codec roundtrips;
- rejection of 8x8, short stride, null plane, and illegal availability.

Result on this host (Apple clang 21, arm64): `3082063 checks, 0 failures`; ASan+UBSan also pass.

## Integration call sites (component boundary)

Status: the M1 still writer and the M2 video keyframe are integrated and
decoder-gated (see "Decoder gate").  The other sequence-header writers listed
below still hardcode `enable_dip = 0`; the list is kept as the checklist for
the remaining carriers.

The `dip` component itself only exports the reference pipeline, the syntax
codec and the `decide()` hook; the carrier edits live in the still/video
writers. To wire DIP into another stream:

1. **Sequence header bit.** Replace the hardcoded `enable_dip = 0` at:
   - `src/bitstream/basic_headers.cpp:218`
   - `src/encoder/m1/still_headers.cpp:114`
   - `src/encoder/m1/tiles_stream/tiles_stream.cpp:139`
   - `src/encoder/m2/video_headers.cpp:141`
   - `src/encoder/m2/post/tiled_frame.cpp:151`
   - `src/encoder/m2/multiref/gate/compound_stream.cpp:122`
   - `src/encoder/m10/high_bitdepth_headers.cpp:130`
   - `src/encoder/pipeline/tiled_still_gate.cpp:164`

   Add `bool enable_dip` to `StillConfig` (`src/encoder/m1/still_headers.h:58-66`) or
   the corresponding config struct, and write it at `sequence_intra_config()`
   bit position 0 (before `enable_intra_edge_filter`).

2. **Block syntax fields.** Add `use_dip`, `dip_transpose`, `dip_mode` to
   `av2enc::still::syntax::BlockSyntax` (`src/encoder/m1/still_syntax.h:73-86`). Add the two
   CDF copies to `CdfState` and `init_cdfs()` (`src/encoder/m1/still_syntax.cpp:172-224`).
   Call `write_dip_mode_info()` after `code_uv_mode()` and before `code_tx()`
   (`src/encoder/m1/still_syntax.cpp:1437`, after `code_uv_mode`), passing
   `context.w/h` = current block dimensions and the neighbour UseDip context.

3. **Encoder decision.** In `TileEncoder::choose_luma_mode()`
   (`src/encoder/m1/still_encoder.cpp:620-700`), when `YMode == DC_PRED` and the
   syntax predicate holds, call `dip::decide()` on the DIP edge set and compare
   its cost with ordinary DC. Carry the selected `{use_dip, dip_mode,
   dip_transpose}` in `ModeChoice` and `BlockSyntax`.

4. **Prediction.** In `TileEncoder::predict_block()`
   (`src/encoder/m1/still_encoder.cpp:562-588`) and the final encode path
   (`src/encoder/m1/still_encoder.cpp:1041-1047`), branch on `use_dip && plane == 0`
   and call `dip::prepare_edges()` + `dip::predict8()` with the same
   `max_x/max_y`, `num4_above_right`, `num4_below_left`. Do not reuse
   `intra::prepare_edges` for DIP (sample counts differ).

5. **GPU path.** The GPU encoder's chosen-mode predictor is
   `src/cuda/m1/still_gpu.cu:1615` and the search path is
   `src/cuda/m1/search/luma_search.cu:429` (both call
   `intra_modes::intra_predict_batch`), plus the M2 backend at
   `src/cuda/m2/video_backend_cuda.cu:538`. Add a DIP dispatch to
   `src/cuda/dip/` for `use_dip && plane == 0`; see `src/cuda/dip/README.md`.

## Decoder gate

The still and video carriers are now integrated, so the pinned decoder gate
exists and passes:

| Carrier | Command | Result |
|---|---|---|
| Still | `--still-dip` (`intra_gate.sh` cases `dip_64`, `dip_360`, `dip_cdef_64`, `dip_ibp_360`, `palette_scc128_dip`) | `dav2d f9dafbc --strict 1` decodes; recon exact |
| Video keyframe | `--video-dip` (`src/pipeline/video/video_dip_gate.sh`) | `dav2d f9dafbc --strict 1` decodes 64x64 and 640x360, 1 and 4 pictures; recon exact with `--video-intra-full` on/off and with `--video-tx-search` |

The video path writes §5.4.5 `enable_dip = 1`, codes §5.20.7.9
`dip_mode_info()` per eligible luma DC block, keeps the §8.3.1 `UseDip`
neighbour grid, and reuses this module's `prepare_edges`/`predict_block8`.
See `src/pipeline/video/README.md` for the decision and the by-name
rejections (`--tiles`, `--video-sdp`, `--video-lossless`, alternate frame
schedules, `--video-gpu`, non-8-bit/4:2:0).

## Known limits

- GPU kernel is 8-bit only (matches M1); CPU reference supports 8..12-bit.
- CPU decision API is 8-bit (`decide()` rejects other depths); prediction is
  depth-generic.
- DIP is luma/DC-only; chroma keeps ordinary DC even when `use_dip == 1`.
- No core files, `src/main.cpp`, or `CMakeLists.txt` were edited.
