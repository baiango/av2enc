# M2 CDEF adapter — post-deblock planning, header bits, §5.20.10.1 indices

Status: **integration-ready adapter + a real two-frame CDEF stream gate.**
No M2 core edits, no `main.cpp`, no `CMakeLists.txt`. The CPU path is the
correctness reference; the GPU kernel is the M1 stage in
`src/cuda/m1/cdef/` (the adapter produces exactly the options/grid it consumes).

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`).

```text
src/encoder/m2/cdef/
  video_cdef.h            adapter API: FrameView/Options/Plan, header + tile syntax
  video_cdef.cpp          planning, filtering, §5.18.7.10 bits, §5.20.10.1 symbols
  video_cdef_test.cpp     host tests (107 checks)
  video_cdef_gate.cpp     two-frame M2 stream with CDEF, dav2d sample-exact gate
  build_and_test.sh    standalone build/run (no CMake changes)
  README.md            this file
```

## Placement — the CDEF slot of the M2 post chain

```text
reconstruct -> deblock -> CDEF -> CCSO -> (loop restoration) -> reference
```

`run_video_post_filters()` in `src/encoder/m2/post/loop_filters.cpp` already runs
the three stages in that order. This adapter owns the CDEF slot itself:

1. **Decide** one strength set per frame (`CdefStrengths == 1`) or one strength
   per 64×64 luma block (`CdefStrengths` 2..8) with the §7.18 filter as the
   objective (SSE against the original picture).
2. **Filter** the post-deblock reconstruction into a separate destination. The
   result is the reference frame for later pictures and the output frame.
3. **Signal** the §5.18.7.10 `cdef_params()` fields/bits and, when
   `CdefStrengths >= 2`, the §5.20.10.1 per-block `cdef_idx` symbols.

The adapter is CPU-only, allocation-free in `filter()`, and never touches
device memory. The same `Plan::options` and `Plan::cdef_idx` are valid inputs
to `cuda::still::still_cdef_filter()` (GPU product path).

## API

```cpp
namespace av2enc::video::cdef_adapter {

struct PlaneView { const uint8_t* data; int stride; };

struct FrameView {
  int width, height;                 // visible, both divisible by 8
  int mi_cols, mi_rows;              // 2*((dim+7)>>3)
  PlaneView recon[3];                // post-deblock, MI-aligned stride
  PlaneView source[3];               // original picture (search only)
  const uint8_t* skip;               // MI grids, required (FrameGrids::zero)
  const uint8_t* lossless;
  const uint8_t* lossless_chroma;    // optional
  const uint16_t* tile_col_starts; int num_tile_cols;
  const uint16_t* tile_row_starts; int num_tile_rows;
};

struct Options {
  int damping = 3;                   // CdefDamping 3..6
  int num_strengths = 1;             // CdefStrengths 1..8
  bool on_skip_txfm = false;         // frame bit (see cdef_on_skip_txfm)
  int cdef_on_skip_txfm = 0;         // sequence mode: 0 DISABLED, 1 ALWAYS_ON, 2 ADAPTIVE
  bool disable_loopfilters_across_tiles = false;
  bool include_off = true;           // keep the all-zero set in the ladder
  int64_t chroma_weight = 1;
  const cdef::CdefStrength* candidates; int candidate_count;  // optional
};

struct Plan {
  cdef::CdefOptions options;         // for cdef_frame() / the GPU kernel
  int num_strengths;
  int mi_cols, mi_rows, n64_cols, n64_rows;
  std::vector<int8_t> cdef_idx;      // per 64x64, raster, [0, num_strengths)
  int64_t sse_before, sse_after; int blocks_off;
  bool forced_single_strength;
  bool header_only_emittable;        // == (num_strengths == 1)
  filters_stream::CdefParams params; // decoded-domain §5.18.7.10 fields
  int header_bits;                   // bits write_frame_header_cdef() writes
  int index_bits;                    // measured standalone index cost (bits)
};

bool plan(const FrameView&, const Options&, Plan*, std::string*);
bool filter(const FrameView&, const Plan&, uint8_t* const dst[3],
            const int dst_stride[3], std::string*);

std::size_t write_frame_header_cdef(BitWriter&, const Plan&,
                                    bool single_picture_header,
                                    int cdef_on_skip_txfm, std::string*);
std::size_t write_sequence_filter_config(BitWriter&, bool enable_cdef,
                                         bool on_skip_always_on,
                                         bool on_skip_disabled, std::string*);

// §5.20.9.6 + §5.20.10.1, one instance per tile.
struct TileIndexWriter {
  bool init(int mi_cols, int mi_rows, int strengths, int sb_width_log2,
            std::string*);
  void clear_superblock(int mi_row, int mi_col);
  bool write(av2::Encoder&, int mi_row, int mi_col, int idx, std::string*);
};

// Standalone exact index stream (bit cost + the tile-hook reference).
bool write_index_symbols(const Plan&, std::vector<uint8_t>* payload,
                         std::string*);
bool read_index_symbols(std::span<const uint8_t>, const Plan&,
                        std::vector<int8_t>* grid, std::string*);

struct FrameGrids { ... void zero(int mi_cols, int mi_rows); void bind(FrameView*); };
bool build_mi_grids(...);  // Skips[] + LosslessArray[SegmentIds[]]
}
```

`Plan::params` is directly assignable to
`av2enc::video::post::VideoTiledFrameHeader::cdef`; set
`cdef_frame_enable = true`.

## Exact call order

### 1. Sequence header — §5.4.10

Written once, before any frame. In this tree that is
`av2enc::video::post::make_video_sequence_header_obu()` with
`VideoPostSequenceOptions::filters.enable_cdef = true`. The default M2 sequence
mode is `CDEF_ON_SKIP_TXFM_DISABLED` (`cdef_on_skip_txfm_disabled = true`),
which matches `Options::cdef_on_skip_txfm = 0` and `on_skip_txfm = false`.
`write_sequence_filter_config()` in this adapter writes the same row for an
integrator that owns its own sequence header.

### 2. Frame header — §5.18.7.10

`cdef_params()` is read in this position and nowhere else:

```text
frame_header_info():
  ...
  base_q_idx
  segmentation_enabled / using_qmatrix / delta_q_present
  deblocking_filter_params()          // includes df_delta_q_present bits
  gdf_params()                        // no bits (GDF off)
  cdef_params()                       // <-- here
  lr_params()                         // excluded (restoration off)
  ccso_params()                       // if enabled
  tx_mode_select
  reference_select / skip_mode_present (inter only)
  reduced_tx_set
```

With `single_picture_header_flag = 0` (M2) the row starts with the
`cdef_frame_enable` f(1) bit — unlike M1, where it is inferred from the single
picture header. Then damping f(2), strengths−1 f(3), the skip-mode bit only for
`CDEF_ON_SKIP_TXFM_ADAPTIVE`, and one strength row per set
(`y_pri_zero`, `y_pri` f(4), `y_sec` f(2), `uv_pri_zero`, `uv_pri` f(4),
`uv_sec` f(2)); syntax `y_sec = 3` decodes as the value 4.

`VideoPostSequenceOptions`/`VideoTiledFrameHeader` do this for you:

```cpp
sequence.filters.enable_cdef = true;
sequence.filters.cdef_on_skip_txfm_disabled = true;
VideoTiledFrameHeader fh;
fh.cdef_frame_enable = true;
fh.cdef = plan.params;          // frame_enable is set by the adapter
```

`Plan::header_bits` is the exact bit count in that context (measured by the
shared writer, not hand-counted): 20 bits for a single strength with non-zero
luma and chroma primary (19 bits for the all-zero set), plus 4 bits per extra
non-zero primary.

### 3. Tile start — §5.20.9.6

At every superblock start, **before** `read_partition()`:

```cpp
cdef_clear_superblock(&state, mi_row, mi_col);
```

M2 uses 64×64 superblocks, and a CDEF block is also 64×64, so this clears the
one origin the superblock's blocks will hit.

### 4. Per coding block — §5.20.10.1

`read_cdef()` runs in `intra_frame_mode_info()` / `inter_frame_mode_info()`,
after `intra_segment_id()` / `read_skip()` and before `read_delta_qindex()` and
the mode syntax. Issue it for **every** coding block; it is a no-op once the
64×64 origin has been coded:

```cpp
cdef_adapter::TileIndexWriter cdef_tile;
cdef_tile.init(mi_cols, mi_rows, plan.num_strengths, /*sb_width_log2=*/4, &err);
cdef_tile.tile_mi_col_start = tile_mi_col_start;
cdef_tile.tile_mi_row_start = tile_mi_row_start;
...
cdef_tile.write(encoder, mi_row, mi_col, node->cdef_idx, &err);
```

With `CdefStrengths == 1` nothing is coded (the decoder forces index 0). With
`CdefStrengths == 2` one `cdef_index0` symbol is coded; with 3..8 a
`cdef_index_minus_1` follows. The §8.3.1 `cdef_index0` context (left block in
the current tile and the block above inside the current superblock row) is
implemented once in `src/cuda/m1/cdef/still_cdef_index_syntax.h` and reused here.
The default CDFs are `src/cuda/m1/cdef/still_cdef_cdf.h`.

### 5. Pixels — after the whole frame is reconstructed and deblocked

```cpp
Plan plan;
cdef_adapter::plan(view, options, &plan, &err);   // uses the original picture
cdef_adapter::filter(view, plan, dst, stride, &err);
// dst (CdefFrame) becomes CurrFrame for CCSO and the reference for later frames
```

`filter()` requires the same stride on source and destination (the CPU
`cdef::CdefPlane` has one stride) and rejects aliasing planes.

## What currently emits a real stream

`CdefStrengths == 1` codes **zero tile symbols**, so CDEF can be enabled with
header-only changes even though the M2 tile writer has no `read_cdef()` hook.
`Plan::header_only_emittable` reports this. The gate in this directory does
exactly that and decodes with `dav2d`.

`CdefStrengths >= 2` is a **component-only decision** until the M2 syntax writer
calls `TileIndexWriter::write()` at step 4. `Plan::index_bits` is measured by
encoding the grid through the normative §8.2 coder, so the cost is known
(288 and 280 bits for the two 1280×720 frames; 112 bits at 640×360) — the hook,
not the rate model, is the missing piece.

## Gate

```sh
./build_and_test.sh                       # unit tests + 2-frame 720p gate
INPUT=tests/data/basic_640x360_1f.y4m ./build_and_test.sh   # 640x360 (needs 2 frames)
SANITIZE=1 ./build_and_test.sh --no-gate
WERROR=1   ./build_and_test.sh --no-gate
```

The gate:

1. encodes two frames with the untouched M2 encoder at QP 90
   (`force_intra = true`, because the core predicts from the unfiltered
   reference — see below);
2. plans and filters each reconstruction (single-strength and a component-only
   4-strength per-64×64 plan);
3. rebuilds the stream with `enable_cdef = 1` and a `cdef_params()` row per
   frame, reusing the **same tile payloads**;
4. decodes both the baseline and the CDEF stream with the pinned `dav2d` and
   requires sample-exact Y/U/V equality:
   `decoded(baseline) == M2 recon` and `decoded(cdef) == adapter filter`;
5. writes `video_cdef_src.y4m`, `video_cdef_base.decoded.y4m`,
   `video_cdef.decoded.y4m` and `video_cdef_component4.y4m` for
   `tools/quality.py`.

Current result (Mac dev host, `dav2d` f9dafbc, QP 90, one tile):

```text
== M2 CDEF gate: 2 frame(s) 1280x720 ==
  frame 0: inter 0 intra 618 blocks
  frame 1: inter 0 intra 618 blocks
  post-chain slot    deblock -> CDEF matches
  stream bytes       baseline 87602, cdef 87608 (+6)
  baseline vs recon  frame(s) 2/2, 0 sample diffs, max 0
  cdef vs filter     frame(s) 2/2, 0 sample diffs, max 0
  CDEF changed 245909 samples, header 40 bits
```

The P frame is intra-coded (`force_intra`), which is what makes it independent
of the reference-frame caveat above. The `post-chain slot` line runs
`run_video_post_filters()` twice (deblock-only, then deblock+CDEF with the same
plan) and requires its CDEF output to equal `cdef_adapter::filter()` on the
deblocked planes.

The +6 bytes are the two 20-bit `cdef_params()` rows (plus byte alignment);
tile bytes are unchanged.

## Quality at QP 90

`tools/quality.py` (ffmpeg PSNR/SSIM + ColorVideoVDP 0.5.7, `standard_fhd`,
MPS) on the decoded streams. `component4` is the 4-strength per-64×64 plan
applied by the CPU filter; it is not signallable yet.

1280×720, two frames, full variance-driven partition tree:

| stream | PSNR-Y | PSNR-avg | SSIM-Y | SSIM-all | CVVDP (JOD) |
|---|---:|---:|---:|---:|---:|
| baseline | 45.2834 | 46.3587 | 0.993632 | 0.993802 | 9.8046 |
| CDEF (1 set) | **45.4257** | **46.4983** | **0.993998** | **0.994107** | 9.7957 |
| component4 | 45.4513 | 46.5231 | 0.994052 | 0.994147 | 9.7924 |

640×360, two frames (the M0 test still duplicated into a 2-frame clip):

| stream | PSNR-Y | PSNR-avg | SSIM-Y | SSIM-all | CVVDP (JOD) |
|---|---:|---:|---:|---:|---:|
| baseline | 45.1689 | 46.0928 | 0.992461 | 0.992687 | 9.8188 |
| CDEF (1 set) | **45.3499** | **46.2665** | **0.993099** | **0.993205** | 9.8019 |
| component4 | 45.3973 | 46.3076 | 0.993271 | 0.993323 | 9.8088 |

Bit cost:

| resolution | baseline | CDEF | Δ | index (component4) |
|---|---:|---:|---:|---:|
| 1280×720 (2f) | 87,602 B | 87,608 B | +6 B | 288 + 280 bits |
| 640×360 (2f) | 32,167 B | 32,173 B | +6 B | 112 + 112 bits |

**Verdict.** CDEF is a consistent **PSNR/SSIM win at near-zero rate**
(+0.14 dB luma at 720p, +0.18 dB at 640×360, SSIM-Y +0.0004/+0.0006) for
+6 bytes per two frames. Per-64×64 selection adds +0.03 dB over the single set
for ~36 bytes/frame at 720p. It is **not a perceptual win** on this content:
CVVDP decreases by 0.0089 JOD at 720p and 0.0169 JOD at 640×360 for the
header-only plan; the 4-set plan recovers part of it at 720p (9.7924) but is
still below baseline. Same conclusion as the M1 still-picture gate: wire it as
an opt-in tool for PSNR/SSIM, do not claim a perceptual gain.

## The one core integration step left

The untouched M2 core predicts every P frame from the **unfiltered**
reconstruction (`VideoSequenceResult::recon`). A decoder, however, runs the loop
filters before storing the reference. Enabling CDEF on an inter frame therefore
desynchronises the reference unless the encoder also filters it. The gate uses
`force_intra = true` to stay exact.

The core change (owner: M2 core, **not applied here**) is:

1. after encoding frame *i*, run `cdef_adapter::plan()` + `filter()` on its
   reconstruction (post-deblock; with deblock disabled, on the reconstruction);
2. store the **filtered** planes in `result.recon[i]` and use them as the
   reference for frame *i+1*;
3. keep the tile payload unchanged for `CdefStrengths == 1`, or add the
   `TileIndexWriter::write()` hook (step 4 above) for `>= 2`.

That is a two-line change in `video_frame.cpp` around
`const VideoReconFrame& reference = result.recon.back();` plus the tile hook; it
is deliberately left to the core owner.

## Build integration (integrator)

Add `src/encoder/m2/cdef/video_cdef.cpp` to `av2enc_core`. It needs the already
listed `src/cuda/m1/cdef/still_cdef.cpp` (host CDEF planner), the §7.18 pixel core
`src/encoder/filters/cdef/cdef.cpp`, `src/encoder/m1/filters_stream/filters_stream.cpp`
and `src/entropy/av2/av2_entropy.cpp`. No CUDA symbols and no new include path
(`src` already covers `encoder/m2/cdef/video_cdef.h`). The gate binary is
standalone and deliberately not part of CMake.

## Limitations

- 8-bit 4:2:0 only; dimensions divisible by 8; MI grids even.
- One stride for source and destination.
- `CdefStrengths >= 2` needs the M2 tile-symbol hook; the header writer, the
  context and the index stream are implemented and tested.
- Inter frames need the core reference fix above; the gate uses intra P frames.
- Strength search is SSE-only (no rate term), over the fixed 10-set ladder.
- `src/cuda/m2/` is not involved; the GPU CDEF entry point is
  `cuda::still::still_cdef_filter()` with `Plan::options` / `Plan::cdef_idx`.
- No commit; no changes outside `src/encoder/m2/cdef/`,
  `src/cuda/m1/cdef/` and `src/encoder/filters/cdef/`.
