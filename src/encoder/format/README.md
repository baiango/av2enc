# Format / high-bit-depth component (`src/encoder/format/`)

Status: **standalone component, not integrated.** This directory implements the
format layer needed for 10-bit, 4:2:2 and 4:4:4 support, plus the CPU reference
for the GPU sample path. It does **not** write a sequence header, frame header,
tile payload, or bitstream, and nothing here is wired into
`src/encoder/m0_still.cpp`. No 10-bit stream has been encoded or decoded.

The CUDA sibling (`src/cuda/format/`) contains the three per-sample kernels a
high-bit-depth path needs (expand 8→10/12, reduce 10/12→8, replicate edge
padding) and a test that compares every sample against the CPU component. The
test was not run on this macOS host (no `nvcc`); run it on the `sm_86` box.

## Spec findings that constrain the work

Normative reference: AV2 v1.0.0, commit `38f28e7`.

1. **AV2 v1.0.0 has no 12-bit signalling.** `bit_depth_idc` is one bit:
   `0 = 10-bit`, `1 = 8-bit`; values greater than 1 are reserved for AOMedia
   (§6.4.1 Table 6.3). A conforming stream cannot request 12-bit. This component
   supports 12-bit *internal samples* (buffers, clipping, conversions, tests),
   but `is_signalable()` is false and `bit_depth_idc()` returns `-1`. Do not emit
   a 12-bit sequence header.
2. **`chroma_format_idc` is not monotonic** (§6.4.1 Table 6.2):
   `4:2:0 = 0`, `4:0:0 = 1`, `4:4:4 = 2`, `4:2:2 = 3`.
3. **Profiles constrain the format** (Annex A.2):
   4:0:0/4:2:0 → profile 0 (also 1/2, different interoperability point),
   4:2:2 → profile 3, 4:4:4 → profile 4. No profile allows 12-bit. Profile 31
   (Configurable) exists but this component deliberately does not use it.
4. **`MaxQ` grows with bit depth** (§6.4.1 Table 6.3):
   `255` at 8-bit, `303` at 10-bit (hypothetical `351` at 12-bit).
5. **`base_q_idx` is `f(8)` at 8-bit and `f(9)` otherwise** (§5.18.6.1). The M0
   header writer currently hard-codes 8 bits.
6. **Coded and visible plane sizes differ** (§5.18.4.4, §7.21.2). Coded luma is
   `MiCols * 4 = 8 * ceil(width/8)` by `8 * ceil(height/8)`; chroma coded size is
   that extent shifted right by the subsampling factors. Visible chroma is
   `(width + subX) >> subX` by `(height + subY) >> subY`, i.e. ceil division.
   Decoder frame stores are coded-size; picture output is visible-size.
7. **Many normative processes are bit-depth-dependent** even though the
   coefficient entropy alphabet is not. The checklist in `format.h` /
   `format_checklist()` captures them; the important ones are in the integration
   plan below.

## Format matrix

| Format | Internal samples | Signalable | Profile | `chroma_format_idc` | `bit_depth_idc` | MaxQ | `base_q_idx` |
|---|---|:---:|---:|---:|---:|---:|---:|
| 8-bit 4:2:0 | yes | yes | 0 | 0 | 1 | 255 | 8 bits |
| 10-bit 4:2:0 | yes | yes | 0 | 0 | 0 | 303 | 9 bits |
| 8-bit 4:2:2 | yes | yes | 3 | 3 | 1 | 255 | 8 bits |
| 10-bit 4:2:2 | yes | yes | 3 | 3 | 0 | 303 | 9 bits |
| 8-bit 4:4:4 | yes | yes | 4 | 2 | 1 | 255 | 8 bits |
| 10-bit 4:4:4 | yes | yes | 4 | 2 | 0 | 303 | 9 bits |
| 12-bit any | yes | **no** | — | — | reserved | 351 | 9 bits |
| 8/10-bit 4:0:0 | yes | yes | 0 | 1 | 1/0 | 255/303 | 8/9 bits |

## Sample model

* Planar, row-major, one `std::uint16_t` per sample, LSB-aligned: valid range is
  `[0, (1 << BitDepth) - 1]`, upper bits are zero.
* 8-bit frames use the same `uint16_t` container so there is one code path for
  every depth. This doubles 8-bit reference memory, which is acceptable for a
  correctness/utility layer; the GPU hot path should use the native sample type
  it already uses.
* `Picture` owns coded-size planes. The visible rectangle starts at `(0,0)` and
  has `visible_width x visible_height`; the remainder is padding.
* `replicate_borders()` copies the last visible column and row into the padding
  (corners stay corners). Integration must do this, or clamp every read, so edge
  prediction and vector loads never see uninitialized samples.

## Geometry

`make_frame_geometry(Format)` returns `FrameGeometry`:

* `mi_cols = 2 * ((width + 7) >> 3)`, `mi_rows` likewise (§5.18.4.4).
* `coded_width = mi_cols * 4`, `coded_height = mi_rows * 4`.
* Plane 0: coded `coded_width x coded_height`, visible `width x height`.
* Planes 1/2: coded `coded_width >> subX` by `coded_height >> subY`, visible
  `(width + (1<<subX) - 1) >> subX` by `(height + (1<<subY) - 1) >> subY`.
* `stride` is in samples and tight; `offset` is the sample offset into a single
  contiguous frame; `frame_bytes = total_samples * 2`.

Examples (10-bit; geometry is depth-independent):

| Visible | Chroma | Coded luma | Coded chroma | Visible chroma |
|---|---|---|---|---|
| 640x360 | 4:2:0 | 640x360 | 320x180 | 320x180 |
| 641x361 | 4:2:0 | 648x368 | 324x184 | 321x181 |
| 640x360 | 4:2:2 | 640x360 | 320x360 | 320x360 |
| 641x361 | 4:2:2 | 648x368 | 324x368 | 321x361 |
| 640x360 | 4:4:4 | 640x360 | 640x360 | 640x360 |
| 641x361 | 4:4:4 | 648x368 | 648x368 | 641x361 |

Block edge rules are exposed by `block_rect()`:

* Luma uses the requested `BlockSize`; chroma uses the normative
  `Subsampled_Size` mapping (§5.20.7.26), not a hand-rolled shift.
* Chroma origin is `luma_x >> subX`, `luma_y >> subY`.
* `visible_width/height` clip the nominal block to the visible plane;
  `inside` says whether any sample is visible; `partial` says the block is
  clipped.
* The table has real invalid cases: for 4:2:2, luma blocks 4x8, 4x16, 4x32, 8x64, 64x128 and 128x256 have no
  legal chroma residual size
  (`BLOCK_INVALID` in `Subsampled_Size`). A 4:2:0 4x4 luma block maps to a 4x4
  chroma residual even though only 2x2 chroma samples are visible at the edge;
  `block_rect()` reports that as `partial`.

The `Subsampled_Size` table and block dimensions are generated from
`spec/tables/block_tables.json` by `generate_format_tables.py`. Do not hand-edit
`format_tables.h`; the build script runs the generator in `--check` mode.

## Conversion from the M0 8-bit 4:2:0 input

`convert_420_8_to_picture(source, chroma, depth, picture)`:

* luma: `sample << (depth - 8)`.
* 4:2:0 → 4:2:0: chroma copy.
* 4:2:0 → 4:2:2: horizontal sample replication.
* 4:2:0 → 4:4:4: horizontal and vertical sample replication.
* 4:0:0: chroma dropped; the reverse conversion is rejected.
* edges: source chroma coordinates are clamped, so odd dimensions replicate the
  last chroma column/row; coded padding is then filled by `replicate_borders()`.

`convert_to_420_8(picture, out)` is the reverse. It uses **nearest** (top-left)
chroma sampling, not averaging, specifically so that
`4:2:0 → N-bit target → 4:2:0` is byte-exact. Tests pin that round-trip for
even and odd dimensions, 8/10/12-bit, and 4:0:0/4:2:0/4:2:2/4:4:4. An averaging
downsample would be a different (lossy) tool and is not provided.

Depth conversions: `8→10→8` and `10→12→10` are exact for every value;
`12→10→12` is lossy as expected. `reduce_to_8()` rounds to nearest and saturates
at 255.

## API map

| Area | Functions |
|---|---|
| Depth | `sample_max`, `bit_depth_idc`, `bit_depth_from_idc`, `max_quantizer`, `base_q_idx_bits`, `clip_sample` |
| Chroma | `chroma_format_idc`, `chroma_format_from_idc`, `subsampling_x/y`, `num_planes`, `has_chroma` |
| Blocks | `block_width/height`, `find_block_size`, `chroma_residual_size` |
| Geometry | `mi_cols/rows`, `make_frame_geometry`, `PlaneGeometry`, `FrameGeometry` |
| Buffers | `allocate_picture`, `sample_clamped`, `replicate_borders` |
| Range | `check_sample_range`, `clamp_samples`, `samples_in_range`, `expand_8_to_depth`, `reduce_to_8`, `convert_sample_depth` |
| Conversion | `convert_420_8_to_picture`, `convert_to_420_8` |
| Edges | `block_rect`, `visible_extent` |
| Requirements | `format_requirements`, `cdf_q_context`, `format_checklist` |

`format_checklist()` returns the header/transform/entropy/filter items as data;
it is the machine-readable form of the integration notes below.

## Test coverage

`./build_and_test.sh` builds and runs `format_test`:
**143,361 checks, PASS** (macOS CPU, no CUDA).

Groups:

| Group | What it pins |
|---|---|
| depth helpers | idc inversion, MaxQ, `f(8)/f(9)`, clip bounds, 8↔10↔12 round-trips |
| chroma mapping | non-monotonic idc values, subsampling, plane counts |
| block tables | 29 block sizes, 4:2:2 invalid widths, 4:4:4 identity, 4:2:0 mapping |
| geometry | MI alignment, odd dimensions, coded vs visible, plane offsets/bytes |
| validation | named profiles, 12-bit reserved, invalid enums/dimensions |
| range/buffers | clamped edge reads, out-of-range counting, clamping, border fill |
| conversion round-trips | 4 formats × 3 depths × even/odd dimensions, exact 8-bit result |
| conversion errors | null pointers, short strides, zero size, monochrome reverse |
| block rects | right/bottom edge clipping, 4:2:2 full-height chroma, 4:4:4 identity |
| requirements | per-format values, §8.3.1 CDF thresholds, checklist contents |

`src/cuda/format/build_high_bitdepth_test.sh` runs the CPU-vs-GPU test on the
`sm_86` box. It compares every sample of expand/reduce/border replication with
the real CPU component and exercises non-tight strides. Not run on this host.

## Integration plan (required changes outside this directory)

Do this in order: 10-bit 4:2:0 first, then 4:2:2, then 4:4:4. Do not claim any
format works until a minimal stream passes `tools/validate_dav2d.sh`.

### 1. Header writer — `src/bitstream/m0_headers.cpp` (owner: scaffold/header agent)

* `seq_profile_idc`: 0 for 4:2:0, 3 for 4:2:2, 4 for 4:4:4 (`profile_idc()`).
* `chroma_format_idc`: use `chroma_format_idc()`, **not** an enum cast; 4:4:4 is
  2 and 4:2:2 is 3.
* `bit_depth_idc`: use `bit_depth_idc()`; 10-bit is 0, 8-bit is 1. Reject 12-bit.
* `base_q_idx`: write `f(base_q_idx_bits())` bits (8 or 9), and validate against
  `max_quantizer()`.
* Keep `single_picture_header_flag` and the M0 tool disables unchanged. AV2
  colour description is not in the sequence header (it lives in the LCR), so no
  new colour syntax is needed for a single-layer stream.
* Add a header unit test for `bit_depth_idc = 0` + `chroma_format_idc = 3/2`; the
  existing M0 test only covers 8-bit 4:2:0.

### 2. Input / I/O — `src/io/y4m.*` (owner: scaffold agent)

* Short term: keep the 8-bit 4:2:0 reader and call
  `convert_420_8_to_picture()` for high-depth/other-chroma experiments.
* Later: accept `C420p10`, `C422p10`, `C444p10` (and 8-bit 4:2:2/4:4:4) with
  `uint16_t` samples; validate the Y4M colorspace against `Format`.
* Do not silently reinterpret packed 10-bit formats (P010/v210) as planar; this
  component expects planar LSB-aligned samples.

### 3. Reconstruction / transform — `src/encoder/m0_still.cpp`, `src/cpu/transform.*`, `src/encoder/tx_variants/`

* Sample type: `uint16_t` instead of `uint8_t`; residual stays `int16_t` for
  10-bit (range ±1023) and 12-bit (±4095).
* `clip1()` must become `[0, (1 << BitDepth) - 1]`, not `[0,255]`
  (`m0_still.cpp` currently hard-codes `clip3(0,255,…)`).
* DC predictor neutral value must be `1 << (BitDepth - 1)` (512 for 10-bit,
  2048 for 12-bit), not `128` (`pred_value = 128` in `m0_still.cpp`).
* Inverse transform row clip is `±(1 << (BitDepth+7)) - 1`; column clip is
  `±(1 << BitDepth) - 1`. M0 currently hard-codes `-(1<<15)..(1<<15)-1` and
  `-256..255`. Intermediates must be `int32_t` (10-bit row pass exceeds int16).
* Dequant clip is `±(1 << (7+BitDepth)) - 1` (§7.14.3).
* `Transform_Shift` is bit-depth independent; do not change it.
* Chroma TX geometry uses `chroma_residual_size()` / `Subsampled_Size`
  (§5.20.7.26). 4:2:2 halves width only; 4:4:4 behaves like luma.
* Forward transform/quant scaling needs revalidation at the new residual
  magnitude; the dequant table itself does not scale by bit depth, the qindex
  range does (MaxQ 303 at 10-bit).

### 4. Entropy — `src/entropy/av2/` (owner: entropy agent)

* No CDF table changes are required: coefficient symbol alphabets and default
  CDFs are bit-depth independent.
* `init_coeff_cdfs()` selects the CDF set from the raw `base_q_idx`:
  `<=90 → 0`, `<=140 → 1`, `<=190 → 2`, else `3` (§8.3.1). Do not normalize by
  bit depth; 10-bit streams can legitimately use index 3.
* 10-bit coefficient magnitudes are up to ~4× larger; the existing escape /
  Golomb path must be exercised with such values in a unit test even though the
  syntax is unchanged.
* `base_q_idx` is now 9 bits in the header; the entropy coder only sees the
  parsed value, so this is a header/bitstream change, not a CDF change.
* CDF reset/adaptation remains per tile as today.

### 5. Prediction / intra — `src/encoder/intra_modes/`, `src/cuda/intra.*`

* All edge reads must clamp to the *visible* plane (`sample_clamped` semantics).
* `maxX/maxY` are coded-size based in the spec, but reading into uninitialized
  coded padding is wrong; either replicate borders or clamp to visible.
* DC/intra sums and averages use `BitDepth`-scaled samples; the neutral value and
  rounding must match §7.13.2.10 for non-8-bit.
* 4:2:2 chroma has full-height, half-width planes; 4:4:4 chroma is full
  resolution. Do not reuse 4:2:0 block/scan assumptions.

### 6. Filters (when enabled) — `src/encoder/filters/*`

* Deblock q index: subtract `24 * (BitDepth - 8)` before `Side_Thresholds`;
  `side` uses `(1 << (12-BitDepth))` and `>> (13-BitDepth)` (§7.17).
* CDEF: `coeffShift = BitDepth - 8`; direction search uses
  `(sample >> (BitDepth-8)) - 128` (§7.18).
* CCSO/loop restoration: band offsets and rounding are bit-depth scaled;
  `LoopRestorationSize[plane]` is per plane and constrained by
  `64 >> SubsamplingY` (§7.19, §7.20).
* Filter chroma planes at their subsampled origins.

### 7. CUDA frame path — `src/cuda/*`, `src/cuda/arena.*`

* Frame/reference buffers are `uint16_t` at 10-bit; sizes from
  `FrameGeometry::frame_bytes` (2 bytes/sample), including coded padding.
* A 1080p 4:4:4 10-bit frame is ~12.4 MB (3 × 1920×1080 × 2) vs ~3.1 MB for
  8-bit 4:2:0; a 4K 4:4:4 10-bit frame is ~49.8 MB. Recompute the 7,500 MB
  `compat8g` pool sizes and fail cleanly if the requested config does not fit.
* Feed strides in samples (`PlaneGeometry::stride`) to kernels; do not assume
  visible width == stride.
* `src/cuda/format/high_bitdepth.*` provides expand/reduce/border replication;
  wire these into upload and recon download first, then extend transform/intra.
* Keep `cudaMalloc` out of the frame loop; the test's `cudaMalloc` is test-only.

### 8. Validation gates (mandatory)

1. Build a 64x64 10-bit 4:2:0 key frame with the M0 tool subset (DC intra,
   32x32 DCT, no filters). Run `tools/validate_dav2d.sh out.ivf`; require exit 0
   and the expected frame count/resolution.
2. Compare decoder Y4M with encoder reconstruction: 0 LSB mismatch. The recon
   dump must be 10-bit with the same chroma format.
3. Pin MD5 and record the `dav2d` commit in the gate log
   (`docs/tool-validation.md` / roadmap table).
4. Repeat for 4:2:2 (profile 3) and 4:4:4 (profile 4). Use a separate stream per
   profile; do not put 4:2:2 and 4:4:4 in one sequence.
5. Try 8-bit 4:2:2/4:4:4 first if the 10-bit transform path is not ready; it
   isolates chroma geometry from bit-depth arithmetic.
6. Only after decode + recon pass, measure PSNR/BD-rate. No quality number from
   this component.

### 9. What the component intentionally does not do

* No OBU/frame-header serialization (owned by `src/bitstream/`).
* No entropy symbol generation or CDF changes (owned by `src/entropy/av2/`).
* No transform implementation (owned by `src/cpu/transform.*`,
  `src/encoder/tx_variants/`, `src/cuda/transform.*`).
* No film grain, palette, CCSO, LR, CDEF, deblock.
* No packed 10-bit input formats, no chroma interpolation/resampling filters, no
  colour-space conversion.
* No 12-bit bitstream path; that would require a future spec revision.

## Build and run

```sh
# CPU reference + CUDA test if nvcc is present
./src/encoder/format/build_and_test.sh

# CUDA-only test on the sm_86 box
./src/cuda/format/build_high_bitdepth_test.sh
"${TMPDIR:-/tmp}/av2enc_format_build/high_bitdepth_cuda_test"

# Regenerate / verify the normative geometry table
python3 src/encoder/format/generate_format_tables.py --check
```

Build artifacts go to `${TMPDIR:-/tmp}/av2enc_format_build`; nothing is
committed. The component does not touch the root `CMakeLists.txt` or the M0
build.

## Files

| File | Purpose |
|---|---|
| `format.h` / `format.cpp` | types, geometry, range/clipping, conversion, requirements |
| `format_tables.h` | generated block dimensions and `Subsampled_Size` |
| `generate_format_tables.py` | generator from `spec/tables/block_tables.json` |
| `format_test.cpp` | 143k-check CPU test |
| `build_and_test.sh` | standalone CPU test; runs CUDA test if `nvcc` exists |
| `src/cuda/format/high_bitdepth.{h,cu}` | expand/reduce/border kernels |
| `src/cuda/format/high_bitdepth_test.cu` | CPU-vs-GPU bit-exact test |
| `src/cuda/format/build_high_bitdepth_test.sh` | CUDA test build |
