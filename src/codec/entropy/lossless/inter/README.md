# encoder/lossless/inter — AV2 lossless P and B frames

From-scratch CPU reference for the lossless inter paths that
`docs/spec-audit/08-transform.md` and `04-tile-block.md` list as missing:

| Audit item | Status here |
|---|---|
| `lossless_inter_tx_type S()` (§5.20.8.2, p237–238) | **WIRED** in a real P-frame tile, including the §9.3 `Default_Lossless_Inter_Tx_Type_Cdf` symbol |
| inter `lossless_tx_size` selection (§5.20.6.1, p182) | **WIRED** (`lossless_tx_size = 1` → `find_tx_size(Min(32,w),Min(32,h))`; `0` → `TX_4X4`) |
| `CodedLossless` / `LosslessArray` for inter frames (§5.18.2, p108) | **COMPONENT API + tests** (`derive_lossless()` with per-segment `ALT_Q`) |
| inter residual WHT / IDTX / DPCM reconstruction (§7.15.4, p525–528) | **WIRED** |
| FSC coefficient syntax beyond 4×4/8×8 (§5.20.7.27 `coeff_base_bob` / `coeff_base_idtx` / `coeff_br_idtx` / `idtx_sign`) | **WIRED** for 4×4…32×32 rectangular luma transforms; unit-tested at 8×16/16×8/16×32/32×16/32×32 |
| lossless 4:4:4 / 4:2:2 | **WIRED** and decoder-gated (spec §6.4.2 Table 6.2 allows both; no lossless restriction) |
| lossless 4:0:0 (Monochrome) | **WIRED** and decoder-gated (Table 6.2 `chroma_format_idc = 1`; no chroma mode or residual syntax, `HasChroma = 0`) |
| 10-bit lossless | **WIRED** and decoder-gated (`bit_depth_idc = 0`, f(9) `base_q_idx`, Clip1(10)) |
| hierarchical-P schedule | **WIRED** and decoder-gated (`--video-tlayers 4/8`; see "CLI wiring") |
| hierarchical-B schedule (two-sided prediction) | **WIRED** and decoder-gated (`--video-lossless-b --video-tlayers 4/8`): two references, `reference_select = 1`, `COMPOUND_AVERAGE` interior blocks |

Normative source: `spec/AV2_Spec_v1.0.0.pdf` (v1.0.0, commit `38f28e7`).
`dav2d` is a black-box validator only.

## CLI wiring

`av2enc --video-lossless` (CPU) routes the sequence to `encode_video()`; the
module's sequence/frame headers, `CodedLossless` quantization, DPCM/WHT/IDTX
reconstruction and FSC syntax are used unchanged.  Supported surface:

* 8-bit and 10-bit, 4:0:0/4:2:0/4:2:2/4:4:4 (`--bit-depth 8|10 --chroma 400|420|422|444`).
* the flat key+P chain and the hierarchical-P/B temporal-layer schedules
  (`--video-tlayers 4|8|16` for P, `--video-lossless-b --video-tlayers 4|8`
  for B).  The schedules come from `src/codec/inter/tlayers/`; the sequence
  header raises `max_tlayer_id`/NumRefFrames, clears
  `monotonic_output_order_flag`, and the frame OBU carries `obu_tlayer_id`.
  A hierarchical-B interior frame carries two references (one past, one
  future) and codes two-sided `COMPOUND_AVERAGE` prediction; the layer-0 B
  anchor (display index 7 of an 8-frame GOP) keeps one reference.
* `--video-lossless-intra-period N` sets `Options::intra_period` (0 = all
  inter), `--video-min-block 8|16|32` selects the coding block size, and
  `--lossless-fsc` forces `force_intra_fsc`.

Samples are stored as `uint16_t` (`Picture::bit_depth`) at every depth; the
reconstruction clips to `[0, (1 << bit_depth) - 1]` and the DC predictor uses
the §7.13.3.22 `Div_Lut` form (the `w + h` denominator is not a power of two
for rectangular leaves).  The module's sequence/frame headers carry
`bit_depth_idc`, the depth-scaled `base_q_idx` (f(8)/f(9)) and, for
`max_tlayer_id > 0`, the `tlayer_dependency_present_flag`.

The decoder gates are `src/pipeline/video/video_lossless_gate.sh` (720p plain
4/8-frame streams, `--video-tlayers 4/8`, `--video-lossless-b --video-tlayers
4/8`, and every supported format including 4:0:0 at 640x360 plus 64x64, all
`dav2d --strict 1` with 0-diff recon) and this module's own
`build_and_test.sh`, whose gate now covers 10-bit 4:2:0/4:4:4, 4:0:0 and the
hierarchical-P/B schedules.

## What is encoded

* 8/10-bit, one tile, 64×64 superblocks.
* One key frame (`OBU_CLOSED_LOOP_KEY`) followed by P frames
  (`OBU_REGULAR_TILE_GROUP`).  The flat chain keeps `NumRefFrames = 2` and a
  single total reference; the hierarchical plans use one slot per picture and
  up to two total references.  Every non-key block is inter or intra according
  to `Options::intra_period`.
* A hierarchical-P inter block uses `single_mode = GLOBALMV` with identity
  global motion (zero MV).  In a hierarchical-B frame every inter leaf codes
  `comp_mode = COMPOUND_REFERENCE`, the implied `(0, 1)` reference pair, and
  non-joint `compound_mode_non_joint = 3` (`GLOBAL_GLOBALMV`); the prediction is
  §7.13.3.14 `COMPOUND_AVERAGE` at `CWP_EQUAL`, i.e.
  `Clip1(Round2(ref0 + ref1, 1))` at zero MV.  There is no motion search — the
  purpose is the lossless transform / coefficient syntax, not RD.
* Lossless `read_tx_size`:
  * intra non-FSC → `TX_4X4`;
  * intra FSC or inter → one `lossless_tx_size` symbol; `1` selects
    `find_tx_size(Min(32,bw),Min(32,bh))`, `0` selects `TX_4X4`.
* Lossless `transform_type` for inter luma:
  * `TX_4X4` → `lossless_inter_tx_type S()`, `1 = IDTX`, `0 = DCT_DCT`;
  * larger transforms → IDTX with no symbol.
* Lossless reconstruction: IDTX copies levels (`Dequant >> (3-shift) == level`
  at q=0), DCT_DCT 4×4 uses the reversible §7.15.2.2 WHT, then the DPCM
  cumulative sum on intra blocks, then `Clip1`.
* FSC: `useFsc = enable_fsc && PlaneTxType==IDTX && plane==0 &&
  (fsc_mode || is_inter)`.  The writer/reader handle arbitrary 4×4…32×32
  transforms; `segEob = min(32,w)*min(32,h)` and the §5.20.7.30 2-D scan are
  generated, not specialised to 4×4/8×8.

Scope limits: 12-bit, SDP/CHROMA_PART, multiple tiles, real motion search,
compound modes other than `COMPOUND_AVERAGE`/`GLOBAL_GLOBALMV`, and lossy
quantisation are not encoded.  The independent reader (`decode_video()`) walks
the reference slots (including a key frame's all-slots refresh) and returns
pictures in order-hint (display) order, so the flat chain and both hierarchical
schedules are checked against the source, not only against the encoder's
recon.  The FSC coefficient syntax is lossless-level only here; the same syntax
with lossy levels is exercised by `fsc_coeff_roundtrip()`.

## API (`lossless_inter.h`)

```cpp
namespace av2enc::lossless::inter {

struct FrameQuantInputs { ... };       // base Q, delta-Q, per-segment ALT_Q
struct FrameLossless { bool coded_lossless, has_lossless_segment;
                       bool lossless[16]; int qindex[16]; };
FrameLossless derive_lossless(const FrameQuantInputs&, bool frame_is_inter);

bool make_geometry(int w, int h, ChromaFormat, Geometry*, std::string*);

// reversible helpers
void forward_lossless_block(const int*, int w, int h, bool dct_dct, int*);
void inverse_lossless_block(const int*, int w, int h, bool dct_dct, int*);
void dpcm_difference_block / dpcm_cumsum_block(...);

// §5.20.6.1 / §5.20.8.2
bool find_tx_size(int coded_w, int coded_h, int*, int*);
TxSizeDecision select_tx_size(int bw, int bh, bool is_inter, bool fsc_mode,
                              bool allow_select, bool want_large);

// writer + independent reader
bool encode_video(std::span<const Picture>, const Options&, EncodeResult*,
                  std::string*);
bool decode_video(std::span<const std::vector<uint8_t>> units, DecodeResult*,
                  std::string*);

// FSC coefficient syntax at 4x4..32x32, independent writer/reader
bool fsc_coeff_roundtrip(const int* levels, const CoeffContext&, int* decoded,
                         std::string*);

std::span<const GoldenCase> golden_cases();
}
```

`EncodeResult::units` holds one Annex-B temporal unit per frame.  Unit 0 is
`TD + sequence header + CLK`; later units are `TD + P` or `TD + B` (the OBU is
always `OBU_REGULAR_TILE_GROUP`).  `EncodeResult::recon` is the encoder's
decoder-path reconstruction in display order and `stats[i].exact` is true only
when every sample matches the source.

`decode_video()` parses the OBUs, sequence header, every frame header, the
partition tree, inter/intra mode info, transform sizes and coefficients,
reconstructs every frame through the reference slots, and returns pictures in
order-hint order.  It does not reuse the encoder's decisions.

`Options::block_size` is the target coding-block size (8/16/32; boundary blocks
split down to 8).  `Options::intra_period` forces one intra block per N in a P
or B frame; `0` means all inter, `1` means all intra.  `Options::temporal_layers`
selects a closed hierarchical GOP and `Options::b_schedule` switches it from
the single-reference P plan to the two-reference B plan.  `inter_4x4_only` +
`inter_4x4_idtx` pin the §5.20.8.2 branch for tests, and `force_intra_fsc`
pins `fsc_mode = 1`.

## Normative details that cost debugging time

1. **`lossless_inter_tx_type` is `S()`, not a raw bit.**  It uses
   `Default_Lossless_Inter_Tx_Type_Cdf = {16384, 0, 0}`.  Writing a raw bit
   makes the tile decode differently once a DCT_DCT 4×4 inter block appears.
2. **`fsc_mode` context is `3` in an inter frame.**  §8.3.2 (p590) says:
   if `FrameIsIntra || RegionType == INTRA_REGION`, sum `FscModes` of the
   `NPos` neighbours; otherwise the context is the constant `3`.  Using the
   neighbour sum in a P frame desynchronises every FSC block.
3. **`NumRefFrames` must be ≥ 2 for this encoder.**  With `NumRefFrames = 1`
   `dav2d --strict 1` silently drops the P frame even though `--strict 0`
   decodes it.  The sequence header therefore signals
   `explicit_num_ref_frames = 1` (`NumRefFrames = 2`) and the P header writes a
   1-bit `ref_frame_idx`.
4. **Chroma type follows the luma 4×4 transform.**  For an inter block with
   `TX_4X4` luma, `compute_tx_type()` returns `TxTypes[y4][x4]` for chroma.
   The chroma 4×4 cell maps to luma 4×4 index
   `(cx << SubsamplingX) >> 2`, not `cx >> SubsamplingX`; the latter is only
   right for 4:2:0/4:2:2.
5. **Lossless IDTX copies levels.**  At q=0, `Dequant = level*8 >> shift` and
   §7.15.4 says `Residual = Dequant >> (3-shift)`, i.e. exactly `level` for
   every lossless size.  The WHT path is still required for DCT_DCT 4×4.
6. **DPCM stores the first difference.**  The decoder applies the cumulative
   sum after the inverse transform, so the encoder must difference first.
7. **`frame_size_override_flag` and `order_hint` precede the primary-ref
   fields** in a P frame header (§5.18.2, p98), and `refresh_frame_flags`
   follows them.
8. **An all-zero luma block infers `DCT_DCT`.**  `lossless_inter_tx_type` is
   only present when `eob > 0`; §5.20.7.27 sets `TxTypes = DCT_DCT` for
   `all_zero`.  The encoder and the reader must both store that effective type
   for the chroma `compute_tx_type()` derivation, otherwise a chroma block
   mapped to an all-zero luma cell uses IDTX on one side and DCT_DCT on the
   other (the residual is identical in levels, the reconstruction is not).
9. **`read_compound_ref()` is bitless for the (0, 1) pair.**  With
   `NumTotalRefs == 2` and `NumSameRefCompound == 0`, the
   `ref + 1 == NumTotalRefs - 1` branch infers `comp_ref = 1`; coding a
   `comp_ref` symbol shifts the whole tile.  Only `comp_mode`, `is_joint` and
   `compound_mode_non_joint` are coded for a two-reference B leaf.

## Tests

```sh
./src/codec/entropy/lossless/inter/build_and_test.sh
SANITIZE=1 WERROR=1 ./src/codec/entropy/lossless/inter/build_and_test.sh
GATE_REQUIRED=1 ./src/codec/entropy/lossless/inter/build_and_test.sh
```

`lossless_inter_test.cpp` covers:

* `derive_lossless()`: base Q 0/positive, `delta_q_present`, per-segment
  `ALT_Q`, mixed lossless/lossy segments, with `frame_is_inter` true/false;
* `find_tx_size()` / `select_tx_size()`: 4×4, 8×8, 16×16, 32×32 and clamped
  >32 sizes;
* exact WHT and DPCM round trips over thousands of random vectors;
* `fsc_coeff_roundtrip()` at 4×4, 8×8, 8×16, 16×8, 16×16, 16×32, 32×16,
  32×32, the all-zero block, and a ±2047 (lossy-magnitude) 16×16 block;
* full key + P round trips at 64×64, 136×88 and 640×360, block sizes 8/16/32,
  all-inter / mixed / all-intra P frames, FSC on/off, 4:2:0/4:2:2/4:4:4 and
  4:0:0, 8/10-bit, hierarchical-P and hierarchical-B (gop 4/8);
* every round trip checks the encoder reconstruction and the independent
  reader against the source (0-LSB), not just the two reconstructions against
  each other;
* a pinned MD5 for each stream in `golden_cases()`, checked by the test.

`lossless_inter_gate.cpp` writes one IVF per case (2 frames, or 4/8 for the
temporal-layer schedules) and runs `dav2d --strict 1 --threads 1`.  Passed
cases (dev host, `dav2d f9dafbc`):

```
ok   64x64_420_b8_p0
ok   64x64_420_b16_p4
ok   64x64_420_b32_p4
ok   136x88_420_b8_p0
ok   136x88_420_b16_p4
ok   640x360_420_b16_p4
ok   64x64_444_b16_p4
ok   64x64_422_b16_p4
ok   64x64_420p10_b16_p4
ok   64x64_444p10_b16_p4
ok   640x360_420p10_b16_p4
ok   64x64_420_tl4
ok   64x64_420p10_tl4
ok   64x64_400_b16_p4
ok   640x360_400_b16_p4
ok   64x64_400p10_b16_p4
ok   64x64_420_b8_gop4
ok   64x64_420_b8_gop8
ok   640x360_420_b8_gop8
ok   64x64_420p10_b8_gop8
ok   64x64_400_b8_gop8
lossless_inter_gate: 42 checks passed
```

The dev-host test run reports `lossless_inter_test: 6391 checks, 0 failures`.

Each case proves two things: the module's reader reproduces every frame in
display order, and `dav2d` outputs exactly the source samples for the same
frames (all Y/U/V samples; luma only for 4:0:0).  A truncated frame is a
failure even when the decoder exits 0.

## Integration / call sites

The module is part of `av2enc_core` (`CMakeLists.txt`) and is the writer behind
`av2enc --video-lossless` / `--video-lossless-b` (`src/main.cpp`):
`run_video_lossless_encode()` supplies the 8-bit 4:2:0 pictures, and
`run_video_lossless_format_encode()` supplies 10-bit and the 4:0:0/4:2:2/4:4:4
Y4M cores.  `EncodeResult::units` is written one temporal unit per IVF frame,
and `EncodeResult::recon` goes to `--video-recon`.
