# GDF stream component — AV2 §5.4.10 / §5.18.7.9 / §5.20.10.3 / §7.20.5

Encoder-side syntax and decision for the **Guided Detail Filter** (GDF) of the
AV2 Bitstream & Decoding Process Specification v1.0.0 (2026-05-28, commit
`38f28e7`).

The pixel filter is the recovered normative reference already in
`src/encoder/filters/spec_tables/gdf.cpp` (+ `gdf_tables.h`); this component
does not modify it and does not reimplement it.  The audit found GDF entirely
unreachable: `enable_gdf = 0`, `gdf_unit_matches_sb_size`, `gdf_frame_enable`,
`gdf_per_block`, `gdf_pic_qc_idx`, `gdf_pic_scale_idx` and `use_gdf S()` had no
writer.  This directory supplies all of them, plus an encoder decision and a
real `dav2d` gate.

**Integration status (2026-11-12).**  GDF is wired into the opt-in CPU still
path as `--still-gdf`: sequence `enable_gdf`/`gdf_unit_matches_sb_size`, the
`gdf_params()` row between deblock and CDEF, and the recovered filter applied
in the loop-restoration stage after CDEF/CCSO.  M1 signals `gdf_per_block = 0`,
so no `use_gdf` tile symbols are written (the unit tests still pin that syntax).
Gated by `tools/still_verify/loop_restoration_gate.sh` with `dav2d --strict 1`
and an exact recon compare at 64x64 and 640x360.  GDF overwrites `LrFrame` for
every filtered luma block, so combining it with luma/chroma loop restoration is
rejected until the §7.20.1 pre-GDF `CdefFrame` split is modelled.

## Files

```text
src/encoder/filters/gdf_stream/
  gdf_syntax.h / .cpp       §5.4.10, §5.18.7.9, §5.20.10.3; layout derivation
  gdf_decision.h / .cpp     §7.20.1 region mapping, §7.20.5 apply + decision
  gdf_stream_test.cpp       9,148 checks, pinned goldens, ASan/UBSan clean
  gdf_stream_gate.cpp       M1 still + dav2d recon-exact gate
  gdf_quality.cpp           PSNR / signalling-bit measurement
  run_quality.sh            640x360 and 720p quality report
  build_and_test.sh         standalone build + tests + gate
  README.md                 this file
```

## API

All types live in `av2enc::filters::gdf`.  `BitWriter` is
`av2enc::BitWriter`; `BitReader` is
`av2enc::still::filters_stream::BitReader` (the existing MSB-first reader).

### Sequence (§5.4.10)

```cpp
struct SequenceGdfFlags {
  bool enable_gdf = false;
  bool gdf_unit_matches_sb_size = false;  // inferred 0 when not coded
};
std::size_t WriteSequenceGdfFlags(BitWriter&, const SequenceGdfFlags&,
                                  int sb_size, std::string* error);
bool ReadSequenceGdfFlags(BitReader&, int sb_size, SequenceGdfFlags*,
                          std::string* error);
```

Exact rows:

```text
f(1) enable_gdf;
if ( enable_gdf && get_seq_sb_size() == BLOCK_64X64 )
  f(1) gdf_unit_matches_sb_size;
else
  gdf_unit_matches_sb_size = 0
```

`sb_size` is `Block_Width[SbSize]` (64 or 128).  With a 128×128 superblock the
size bit is never coded.

### Frame (§5.18.7.9)

```cpp
struct FrameGdfContext {
  bool enable_gdf, gdf_unit_matches_sb_size, coded_lossless;
  bool single_picture_header_flag;
  int sb_size;                 // 64 or 128
  int width, height;           // visible luma
  int mi_cols, mi_rows;        // padded frame size / 4
  bool disable_loopfilters_across_tiles;
  int tile_cols, tile_rows;
  std::vector<int> mi_col_starts, mi_row_starts;
};

struct FrameGdfParams {
  bool frame_enable, per_block;
  int pic_qc_idx;              // f(2), 0..3
  int pic_scale_idx;           // f(2), 0..3
  int blk_size;                // GdfBlkSize (64/128), derived
  int cols, rows;              // GdfBlks dimensions
  int pix_scale;               // GdfPixScale = 1 + pic_scale_idx
  bool per_block_present;      // was gdf_per_block coded?
};

GdfUnitLayout DeriveGdfUnitLayout(const FrameGdfContext&);
std::size_t WriteFrameGdfParams(BitWriter&, const FrameGdfContext&,
                                FrameGdfParams*, std::string* error);
bool ReadFrameGdfParams(BitReader&, const FrameGdfContext&,
                        FrameGdfParams*, std::string* error);
```

`WriteFrameGdfParams` writes exactly the `gdf_params()` row:

```text
if ( CodedLossless || !enable_gdf ) gdf_frame_enable = 0
else {
  if ( single_picture_header_flag ) gdf_frame_enable = 1
  else f(1) gdf_frame_enable;
  if ( !gdf_frame_enable ) return
  gdfBlkSize = Max(Block_Width[SbSize], GDF_MIN_SIZE);   // 64/128 -> 128
  if ( gdf_unit_matches_sb_size )
    gdfBlkSize = Block_Width[SbSize];
  else if ( SbSize == BLOCK_64X64 ) {
    a = OR(MiColStarts) | OR(MiRowStarts);
    if ( a & 16 ) gdfBlkSize = 64;      // 64-sample aligned tile start
  }
  GdfBlkSize = gdfBlkSize;
  if ( MiCols*4 > gdfBlkSize || MiRows*4 > gdfBlkSize ||
       (disable_loopfilters_across_tiles && (TileRows > 1 || TileCols > 1)) )
    f(1) gdf_per_block;
  else
    gdf_per_block = 0;
  f(2) gdf_pic_qc_idx;
  f(2) gdf_pic_scale_idx;
  GdfPixScale = 1 + gdf_pic_scale_idx;
}
```

`Gdf_unit_matches_sb_size = 1` gives 64×64 units on a 64×64 superblock;
`= 0` gives 128×128 units unless a tile starts on a 64-sample boundary
(`a & 16`).  A 128×128 superblock always gives 128×128 units.

### Per-block `use_gdf` (§5.20.10.3)

```cpp
bool GdfSymbolAt(const GdfUnitLayout&, const FrameGdfParams&, int sb_size,
                 int mi_row, int mi_col, GdfSymbolPosition*);

class GdfBlockCoder {           // one per tile; owns TileUseGdfCdf
  void Reset();
  bool WriteAt(av2::Encoder&, const GdfUnitLayout&, const FrameGdfParams&,
               int sb_size, int mi_row, int mi_col, bool use_gdf,
               bool* coded);
  bool ReadAt(av2::Decoder&, const GdfUnitLayout&, const FrameGdfParams&,
              int sb_size, int mi_row, int mi_col, bool* use_gdf,
              bool* coded);
};
```

`use_gdf` is an `S()` symbol with `TileUseGdfCdf` = `Default_Use_Gdf_Cdf` =
`{14593, 32, 0}`.  There is **no neighbour context**: the CDF has a single
entry; its state is the per-tile CDF copy and it adapts unless
`disable_cdf_update` is set.

The predicate is exactly `read_gdf()`:

```text
if ( !gdf_frame_enable || !gdf_per_block ) return
sbSize4 = Num_4x4_Blocks_Wide[SbSize]                 // 16 / 32
if ( MiRow % sbSize4 != 0 || MiCol % sbSize4 != 0 ) return
sbRow = MiRow / sbSize4; sbCol = MiCol / sbSize4
sbPerGdf = GdfBlkSize / Block_Width[SbSize]           // 1 / 2
if ( sbCol % sbPerGdf != 0 || sbRow % sbPerGdf != 0 ) return
S() use_gdf
GdfBlks[ sbRow / sbPerGdf ][ sbCol / sbPerGdf ] = use_gdf
```

Call it once per coding block, at the §5.20.10.3 point (after the segment id,
before `read_cdef()`/`read_ccso()`).  The coder returns `coded = false` for
every block that is not a signalled GDF-unit origin.

### Filter application and regions (§7.20.1 / §7.20.5)

```cpp
struct GdfPlane       { const uint16_t* data; int width, height, stride; };
struct GdfSourceFrame { const uint16_t* curr, *cdef; int stride; };
struct GdfFilterInputs{ bool frame_is_intra; int base_q_idx, bit_depth;
                        int num_total_refs, order_hints[2],
                            frame_distance[2], restricted_oh; };
struct GdfStripeRegion{ int x, y, w, h, stripe_start_y, stripe_end_y; };

int  GdfUnitRegions(const GdfUnitLayout&, int row, int col,
                    GdfStripeRegion out[8]);
bool ApplyGdfToFrame(const GdfSourceFrame&, const GdfFilterInputs&,
                     const FrameGdfContext&, const FrameGdfParams&,
                     const std::vector<uint8_t>& unit_flags,
                     uint16_t* out, int out_stride, std::string* error);
```

`out` must already contain a copy of `CdefFrame`.  `unit_flags` is ignored when
`params.per_block == 0` (every unit is filtered) and otherwise must be a
row-major `cols × rows` grid.

**Why regions are stripe-aligned.**  §7.20.1 applies the filter per 4×4 block
and looks up `GdfBlks[ stripeRow*MI_SIZE/GdfBlkSize ][ x/GdfBlkSize ]` with
`stripeRow = Min(MiRowEnd-1, ((row+2)>>4)<<4)`.  Stripes are offset upwards by
8 luma samples (`stripeNum = (lumaY+8)/64`), so a "64×64 GDF unit" is the
stripe-aligned range

```text
row r:  rows [16r-2 .. 16r+13]  (samples 64r-8 .. 64r+55)
128 unit r: rows [32r-2 .. 32r+29] (samples 128r-8 .. 128r+119)
```

clamped at the top and at `MiRows-1` (the spec's `Min(MiRowEnd-1, ...)` pulls
the tail of the frame into the last grid row).  Horizontally the index is plain
`x/GdfBlkSize`, i.e. sample-aligned.  `GdfUnitRegions()` returns the exact
per-stripe sample rectangles; `ApplyGdfToFrame()` calls the reference once per
(stripe × GDF unit) rectangle with that stripe's `LumaStripeStartY/EndY`.

The reference's split-equivalence property makes this exact: applying a whole
stripe region equals applying every 4×4 block inside it, and a 128-unit split
into two 64-units produces bit-identical output (tested).

### Encoder decision

```cpp
struct GdfDecisionOptions {
  GdfFilterInputs filter;
  bool search_pic_indices;     // search qc × scale, or use the two fixed values
  int pic_qc_idx, pic_scale_idx;
  bool disable_cdf_update;     // for the exact use_gdf bit-cost measurement
  double lambda_bits;          // SSE per signalled bit; 0 = quality only
};

struct GdfDecision {
  FrameGdfParams params;
  std::vector<uint8_t> unit_flags;    // GdfBlks grid (empty for per_block = 0)
  std::vector<uint8_t> block64_flags; // unit-local 64x64 report grid
  int block64_cols, block64_rows;
  int64_t sse_before, sse_after;
  int enabled_units, total_units;
  int64_t header_bits, symbol_bytes, symbol_bits;
  double psnr_before, psnr_after;
  int chosen_qc_idx, chosen_scale_idx;
};

bool DecideGdf(const GdfSourceFrame& recon, const GdfPlane& original,
               const FrameGdfContext&, const GdfDecisionOptions&,
               GdfDecision*, std::string* error);
bool MeasureUseGdfSymbols(const GdfUnitLayout&, const FrameGdfParams&,
                          int sb_size, const std::vector<uint8_t>& flags,
                          bool disable_cdf_update, int64_t* bytes,
                          int64_t* bits, std::string* error);
```

The decision:

1. applies the reference to the reconstruction for each
   `gdf_pic_qc_idx × gdf_pic_scale_idx` candidate;
2. measures `SSE(original, recon+GDF)` against `SSE(original, recon)` per GDF
   unit, and in 64×64 pieces inside each unit (`block64_flags`);
3. chooses `gdf_per_block = 0` (filter every unit) or `1` (per-unit flags),
   whichever has the lower `SSE + lambda_bits × signalled bits`, when the
   syntax codes `gdf_per_block`;
4. on a regular (non-single-picture) frame, also considers
   `gdf_frame_enable = 0` and its one header bit;
5. costs the per-block symbols with the real §8.2 coder (adaptation or frozen
   CDF, mirroring `disable_cdf_update`) and reports the exact payload bytes.

`lambda_bits = 0` is quality-only: every unit whose filtered SSE is lower is
enabled.  A positive lambda trades SSE for bits.

## Tests

```sh
./src/encoder/filters/gdf_stream/build_and_test.sh            # tests + gate
./src/encoder/filters/gdf_stream/build_and_test.sh --test-only
./src/encoder/filters/gdf_stream/build_and_test.sh --asan
```

`gdf_stream_test.cpp`: **9,148 checks, 0 failures**, ASan + UBSan clean.

Pinned goldens:

| item | bits | bytes |
|---|---:|---|
| `enable_gdf=0`, SB 64 | 1 | `00` |
| `enable_gdf=1`, matches=0, SB 64 | 2 | `80` |
| `enable_gdf=1`, matches=1, SB 64 | 2 | `c0` |
| `enable_gdf=1`, SB 128 (matches inferred 0) | 1 | `80` |
| single picture, 64×64, qc=2, scale=1, no `per_block` bit | 4 | `90` (blk 64/128) |
| single picture, 256×192, qc=2, scale=1, `per_block=0` | 5 | `48` |
| regular frame, 256×192, enable=1, `per_block=0`, qc=2/scale=1 | 6 | `a4` |
| use_gdf all 0, 8×4 grid, frozen CDF | 56 | `00000000020000` |
| use_gdf all 1, 8×4 grid | 48 | `fffffff80000` |
| use_gdf alternating, 8×4 grid | 48 | `441c39e9c000` |
| use_gdf one on at (1,2), 8×4 grid | 56 | `0009a200040000` |

Also pinned: the region tables for 640×360 / 64×64 / 256×192, exact coverage
of every pixel by exactly one region, the 128-unit vs 64-unit split equality,
disabled units leaving pixels untouched, the decision's determinism and
non-regression, and an FNV-1a hash of the filtered 64×64 output
(`0x5299de7ef082b393`).

The symbol reader is an independent `av2::Decoder` pass with a fresh CDF; the
frame/sequence readers are independent `BitReader` implementations.

## Gate

`gdf_stream_gate` builds a real M1 still picture with the shipped CPU M1
encoder, rewrites only the sequence/frame headers, applies the reference to
the encoder reconstruction, and decodes with `dav2d --strict 1 --threads 1`.
It requires **0 Y/U/V sample differences**.

Phases:

* **A. `enable_gdf=0` control** — the rewritten stream must be byte-identical
  to the M1 encoder's own output.  It also decodes to the M1 reconstruction.
* **B. decision-selected GDF** — searched `gdf_pic_qc_idx` /
  `gdf_pic_scale_idx`, 0 diffs.
* **C. forced parameters** — `gdf_pic_qc_idx=2`, `gdf_pic_scale_idx=1`,
  0 diffs.
* **D. opposite `GdfBlkSize`** (frames ≥ 128×128) — `gdf_unit_matches_sb_size`
  0 vs 1, so both 64×64 and 128×128 units, 0 diffs.

Results (this machine, `dav2d-macos` f9dafbc):

| clip | QP | base | GDF | GDF bits | PSNR-Y | result |
|---|---:|---:|---:|---:|---:|---|
| 64×64 | 90 | 892 B | 893 B | 6 | 42.502 → 42.536 | PASS |
| 640×360 | 90 | 7,228 B | 7,229 B | 7 | 46.310 → 46.433 | PASS |
| 640×360 | 140 | 1,715 B | 1,716 B | 7 | 39.372 → 39.645 | PASS |
| 1280×720 | 140 | 10,167 B | 10,168 B | 7 | 34.296 → 34.485 | PASS |

At 640×360/QP90 both `GdfBlkSize` 64 and 128 were gated (0 diffs); the
`enable_gdf=0` control was byte-identical in every run.

## Quality and bit cost

```sh
./src/encoder/filters/gdf_stream/run_quality.sh
QP=140 ./src/encoder/filters/gdf_stream/run_quality.sh
```

`gdf_quality` encodes one frame with M1, runs the decision on the real
reconstruction, and writes `base.y4m`, `gdf_frame.y4m` (per_block = 0) and
`gdf_block.y4m` (per_block = 1).  `run_quality.sh` measures PSNR-Y with ffmpeg
and CVVDP-JOD with `cvvdp` (`standard_fhd`, device `mps`).

`shine_1280x720_60f.y4m`, first frame; 640×360 is the top-left crop.  CVVDP is
ColorVideoVDP v0.5.7 (37.84 pix/deg, `standard_fhd`, MPS) on the same Y4M files
the gate compares.

**QP 90**

| clip | mode | PSNR-Y | CVVDP | signalling |
|---|---|---:|---:|---|
| 640×360 | base | 46.310 | 9.8826 | 0 |
| 640×360 | gdf_frame | 46.433 | 9.8746 | 7 bits |
| 640×360 | gdf_block | 46.433 | 9.8746 | 7 + 72 bits (60/60 units) |
| 720p | base | 41.689 | 9.8458 | 0 |
| 720p | gdf_frame | 41.756 | 9.8435 | 7 bits |
| 720p | gdf_block | 41.758 | 9.8436 | 7 + 272 bits (72/240 units) |

**QP 140** (`qc=3`, `scale=3`; the tables are tuned for the higher range)

| clip | mode | PSNR-Y | CVVDP | signalling |
|---|---|---:|---:|---|
| 640×360 | base | 39.372 | 9.2593 | 0 |
| 640×360 | gdf_frame | 39.645 | 9.2711 | 7 bits |
| 640×360 | gdf_block | 39.645 | 9.2711 | 7 + 72 bits (60/60 units) |
| 720p | base | 34.296 | 9.1988 | 0 |
| 720p | gdf_frame | 34.485 | 9.2175 | 7 bits |
| 720p | gdf_block | 34.485 | 9.2175 | 7 + 264 bits (98/240 units) |

Bit cost is per frame: 6–7 bits of sequence+frame syntax (2 sequence, 4 frame,
plus the `gdf_per_block` bit when the syntax codes it) and the exact
arithmetic-coded `use_gdf` payload for per-block mode.  Frame-wide GDF costs
1 byte per frame; per-block mode costs 10 B at 640×360 and 34 B at 720p in the
QP 140 rows above (0.6 % and 0.33 % of the frame).

Honest reading: at QP 90 the PSNR-Y gain is small (+0.07…+0.12 dB) and CVVDP
slightly decreases (−0.002…−0.008 JOD); at QP 140 the gain is +0.19…+0.27 dB
PSNR-Y and +0.012…+0.019 JOD for 1 byte of signalling.  Per-block selection is not a quality win on this
content when every unit improves; it becomes useful only when some units
regress, where the decision then skips them.

## Call sites

No files outside this directory were changed.  An integration pass needs:

**Sequence header**
* `src/encoder/m1/still_headers.cpp:147` (`put_bit(false); // enable_gdf`) and
  `src/encoder/m2/video_headers.cpp:209` — call `WriteSequenceGdfFlags()`.
* `src/bitstream/basic_headers.cpp:257` and
  `src/encoder/sb128/sb128_stream.cpp:499` — same (M0 and 128-SB writer).
* `src/encoder/m1/filters_stream/filters_stream.cpp:317-319` currently fails
  on `enable_gdf`; replace the single bit with `WriteSequenceGdfFlags()` and
  the reader mirror with `ReadSequenceGdfFlags()`.  The function already
  carries `SequenceFilterConfig::enable_gdf`; add
  `gdf_unit_matches_sb_size`.

**Frame header** (after `deblocking_filter_params()`, before `cdef_params()`)
* `src/encoder/m1/still_headers.cpp:197-199` — insert `WriteFrameGdfParams()`
  between the `df_delta_q_present` bits and `cdef_params()`.
* `src/encoder/m2/video_headers.cpp:288-289` — insert between the deblock bits and
  `tx_mode_select` (M2 disables CDEF/LR/CCSO, so GDF is next in §5.18.7 order).
* `src/encoder/m2/post/loop_filters.cpp:459-470` (`write_video_deblock_params`
  caller) and `src/encoder/m2/post/tiled_frame.cpp:465-482` — the tiled M2
  frame-header writer uses the same insertion point.

**Tile syntax** (once per coding block, before `read_cdef()`/`read_ccso()`)
* `src/encoder/m1/still_syntax.cpp:1407-1415` — insert a
  `GdfBlockCoder::WriteAt()` call before
  `cdef_encode_block_index(...)`; `r, c` are the leaf's `MiRow/MiCol`.
* `src/encoder/m2/video_syntax.cpp` — the M2 leaf writer currently codes no
  CDEF/CCSO.  Insert the `GdfBlockCoder` at the corresponding point in
  `inter_frame_mode_info()` / `intra_frame_mode_info()` (after the segment id,
  before the mode info that follows `read_gdf()`).
* `src/encoder/m2/post/tiled_frame.cpp` — the tiled M2 block writer, if used
  for product streams.

**Filter application**
* `src/encoder/m1/still_encoder.cpp` after the deblock/CDEF/CCSO stages and before
  `result.recon_y/u/v` are copied out.  `curr` and `cdef` are the same buffer
  when CDEF/CCSO are off; with CDEF on, `CurrFrame` and `CdefFrame` are
  different buffers (§7.20.2).
* `src/encoder/m2/post/loop_filters.cpp:426-431` — immediately after the CCSO
  stage and before `frame.output_[plane] = frame.cdef_out_[plane]`; this is the
  `CdefFrame -> LrFrame` point.
* `src/encoder/m2/video_frame.cpp:1027,1052` — push the GDF output (not raw recon)
  as the reference for the next frame.

**Encoder decision**
* `src/encoder/m1/still_encoder.cpp` after reconstruction, using the source plane
  and the post-deblock/CDEF/CCSO reconstruction.
* `src/encoder/m2/video_frame.cpp` per frame, with `frame_is_intra` and the
  `OrderHints`/`FrameDistance`/`RESTRICTED_OH` inputs from §5.18.2.

## Limitations / deliberate scope

* **`gdf_per_block = 1` is not in the dav2d gate.**  The gate reuses the M1
  encoder's existing range-coded tile payload; inserting `use_gdf` symbols
  would require re-encoding the tile (core edit).  The gate therefore codes
  `gdf_per_block = 0`, which is legal whenever the bit is present and filters
  every unit.  The symbol writer/reader, its position predicate and its exact
  payload bytes are pinned by `gdf_stream_test.cpp`; when a tile writer gains a
  `read_gdf()` hook, the existing decision already produces the grid.
  When the decision prefers per-block mode (e.g. 720p/QP140), the gate prints
  `decision preferred per-block: N/M units` and still emits the frame-wide
  stream.
* **Multi-tile grid indexing is not modelled.**  `read_gdf()` indexes
  `GdfBlks` with `MiRow/MiCol`, while the filter uses tile-relative `x,y`; the
  specification's `GdfBlks` is frame-global, so the two disagree when
  `MiRowStart/MiColStart != 0`.  This component assumes one tile (the M1/M2
  product configuration).  `disable_loopfilters_across_tiles` is still honoured
  for the `gdf_per_block` presence rule and the `a & 16` GdfBlkSize case.
* 8-bit is gated (the M1 still path).  The reference and API support 8/10/12
  bit; 10-bit is the only HBD signalable in v1.0.0.
* Luma only, as §7.20.5 defines it.  No CUDA, no threading, no SIMD; this is
  the CPU correctness oracle.
* The gate's sequence/frame header writers are local copies of the M1 writers
  (plus the GDF rows), because the core writers do not have the hook yet.  They
  are proven byte-identical to the M1 output with `enable_gdf=0`.
