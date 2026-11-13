# M1 GPU path — CDEF stage (AV2 §7.18 / §5.18.7.10)

Integration-ready CDEF stage for the M1 GPU still-picture path. It takes the
unfiltered (deblocked) reconstruction planes plus MI metadata and returns

* the filtered planes (`CdefFrame`),
* the per-64×64 `cdef_idx` grid, and
* the `cdef_params()` header buffer.

Normative placement: **reconstruct → deblock → CDEF → CCSO → LR**. M1 has no
CCSO/LR, so CDEF consumes the deblocked reconstruction and its output is the
frame that is written out and referenced later.

```text
src/cuda/m1/cdef/
  still_cdef.h                stage contract (CUDA-free): header params, MI grids,
                           stage input/options/plan, device POD
  still_cdef.cpp              header packing, MI-grid materialisation, planning;
                           part of the orchestrator's host OBJECT library
  still_cdef_cuda.h           CUDA entry points
  still_cdef.cu               §7.18 filter kernel + encoder-side SSE kernel
  still_cdef_index_syntax.h   §5.20.10.1 read_cdef(): encoder + decoder mirror
  still_cdef_cdf.h            default CDEF CDFs (§9.3)
  still_cdef_test.cpp         host stage tests (519 checks)
  still_cdef_gpu_test.cu      CPU-vs-GPU parity + timing on a real recon dump
  run_stream_gate.cpp      dav2d stream gate (baseline vs CDEF)
  build_and_test.sh        standalone build/run
  README.md                this file
```

The sample arithmetic is shared with the CPU reference through
`encoder/filters/cdef/cdef_core.h` and `cdef_search.h`, so the CPU reference,
the CUDA filter and the CUDA search are all the same expressions.

## API

```cpp
namespace av2enc::cuda::still {

// ---- §5.18.7.10 header buffer -------------------------------------------
struct CdefHeaderStrength {
  bool y_pri_zero; int y_pri_strength;      // 0..15, written when !zero
  int y_sec_strength;                       // syntax 0..3, 3 decodes as 4
  bool uv_pri_zero; int uv_pri_strength;
  int uv_sec_strength;
  int y_sec_value, uv_sec_value;             // decoded value (0/1/2/4)
};
struct CdefHeaderParams {
  bool frame_enable;
  int damping_minus_3;       // f(2)
  int strengths_minus_1;     // f(3)
  bool on_skip_txfm_frame_enable;  // f(1) when CdefOnSkipTxfm == ADAPTIVE
  int num_strengths;
  CdefHeaderStrength set[8];
  int bits;                  // exact bit cost of cdef_params()
};
bool still_cdef_build_header(const cdef::CdefOptions&, bool single_picture_header,
                          CdefHeaderParams*, std::string* error);

// ---- MI metadata -> §7.18.1 grids --------------------------------------
bool still_cdef_build_mi_grids(const CdefMiInputs&, CdefMiGrids*, std::string*);

// ---- planning -----------------------------------------------------------
bool still_cdef_plan(const CdefStageFrame&, const CdefStageOptions&,
                  CdefStagePlan*, std::string*);

// ---- device (still_cdef_cuda.h) -------------------------------------------
cudaError_t still_cdef_filter(const CdefDeviceFrame&, const cdef::CdefOptions&,
                           cudaStream_t = nullptr);
cudaError_t still_cdef_search_sse(const CdefDeviceFrame&,
                               const std::uint8_t* const ref[3],
                               const int ref_stride[3],
                               const cdef::CdefStrength* candidates,
                               int candidate_count, int damping,
                               bool on_skip_txfm, bool across_tiles,
                               std::int64_t* d_sse, cudaStream_t = nullptr);
}
```

`CdefStageFrame` is the integration point: three `CdefPlaneBinding`s
(`src`/`dst`/`stride`/`height`), the original picture planes for the SSE
metric, visible + MI geometry, the MI grids, and tile starts.
`CdefStageOptions` carries `damping`, `num_strengths`, `on_skip_txfm`,
`disable_loopfilters_across_tiles`, the candidate ladder and the chroma SSE
weight.

`still_cdef_plan()` runs the search (CPU reference) and fills
`CdefStagePlan{search, header, cdef_idx, header_bits, index_bits_estimate,
sse_total, sse_off_total, forced_single_strength}`. On the GPU path the
orchestrator either uploads `plan.cdef_idx` and calls `still_cdef_filter()`, or
runs `still_cdef_search_sse()` and feeds the downloaded table to
`cdef::search_detail::select()` — the same decision rule, so both paths pick the
same strengths.

### Orchestrator wiring (one tile, one frame)

```cpp
// after recon_deblock_frame(); planes are device-resident in the arena
CdefDeviceFrame df;                 // src = deblocked recon, dst = cdef buffer
df.skip = d_skip; df.lossless = d_lossless; df.lossless_chroma = d_lc;
df.mi_cols = mi_cols; df.mi_rows = mi_rows;
df.n64_cols = (mi_cols + 15) / 16; df.n64_rows = (mi_rows + 15) / 16;

std::vector<CdefStrength> pool;
cdef::CdefSearchOptions so;         // damping / num_strengths / include_off ...
cdef::search_detail::build_pool(so, &pool, &error);   // host
still_cdef_search_sse(df, d_ref, ref_stride, d_pool, pool.size(), damping,
                   /*on_skip_txfm=*/false, /*across_tiles=*/false, d_sse);
// one download (the orchestrator's single per-frame sync):
std::vector<std::int64_t> sse(pool.size() * blocks);
cudaMemcpy(sse.data(), d_sse, ..., cudaMemcpyDeviceToHost);
std::vector<std::int64_t> off_sse = /* the identity-set row (put it in pool) */;
cdef::CdefSearchResult result;
cdef::search_detail::select(pool, n64_cols, n64_rows, sse, off_sse, so,
                            &result, &error);
cudaMemcpy(d_idx, result.cdef_idx.data(), blocks, cudaMemcpyHostToDevice);
still_cdef_filter(df, result.options);
```

The CPU-only path is `still_cdef_plan()` + `cdef_frame()`: no device memory, one
call each.

## Decision rules implemented here

* **Per-64×64 selection.** For every 64×64 luma block the search evaluates each
  candidate strength set with the *exact* §7.18 block filter and picks the one
  with the smallest SSE against the original picture (luma + chroma, visible
  area only). Greedy header list construction, deterministic tie-break (lowest
  candidate/header index).
* **No "off" syntax.** §5.20.10.1 always codes an index in `[0, CdefStrengths)`
  once CDEF is enabled and the block is not skipped. A block can only be left
  unfiltered if the header carries an all-zero strength set; the default ladder
  starts with one and `include_off` keeps it in the pool. With
  `CdefStrengths == 1` the decoder reads no index symbol at all and every block
  is forced to set 0.
* **Pruning.** After the per-block assignment, strength sets that no block
  selected are removed and the index grid is remapped. This is what turns "off
  + one set" into `CdefStrengths == 1` (zero index symbols) on content where
  filtering always wins — the 64×64 gate case does exactly that.
* **Damping.** `CdefDamping` from the options (3..6); luma uses
  `CdefDamping + coeffShift`, chroma `- 1`. `coeffShift` is 0 at 8-bit.
* **Skip / lossless.** `skip` is the AND of the four MI `Skips` cells unless
  `cdef_on_skip_txfm_frame_enable`; it is OR-ed with
  `LosslessArray[SegmentIds]`. Chroma is copied unfiltered when
  `LosslessArray[ChromaSegmentIds]` is set. For M1 (intra, QP 90) both grids are
  all-zero, and `still_cdef_build_mi_grids()` produces them from optional
  per-MI inputs.
* **Tile / frame edges.** `cdef_filter_region()` implements §5.20.9.3: the whole
  frame, or the tile bounds when `disable_loopfilters_across_tiles` is set. Taps
  outside the region contribute neither to the sum nor to the min/max clamp.
  M1 is one tile.
* **Edges.** The 8×8 traversal covers the full MI grid
  (`MiCols = 2*((W+7)>>3)`, `MiRows` likewise), so partial 64×64 blocks at the
  right/bottom edge filter their in-frame 8×8 blocks normally.

## Building and running

```sh
cd src/cuda/m1/cdef

# host stage tests (519 checks; no CUDA needed)
./build_and_test.sh

# stream gate: emit baseline + CDEF streams and decode both with dav2d
DAV2D=tools/bin/dav2d-macos ./build_and_test.sh \
    --gate tests/data/basic_640x360_1f.y4m

# CPU-vs-GPU parity + timings (on the 3060 box)
./build_and_test.sh --cuda --arch=sm_86
```

The gate writes `baseline.ivf`, `cdef.ivf`, both decoded Y4Ms and
`recon.bin` (the plane dump consumed by the CUDA parity test) under
`build-m1-cdef/gate/`.

## Gate results

Measured on the Mac dev box with the pinned decoder **dav2d f9dafbc**,
M1 defaults (QP 90, CfL + deblock on, depth 4), one frame, one tile.
`CdefStrengths = 1` for the emitted streams (zero tile bits), so the only
bitstream change is the header.

| Still | IVF base | IVF+CDEF | ΔB | decoded==recon | decoded==CDEF | PSNR-Y Δ | PSNR-all* Δ | SSIM-Y base→cdef | CVVDP base→cdef | 4-set PSNR-all* Δ | 4-set CVVDP |
|---|---:|---:|---:|---|---|---:|---:|---|---|---:|---|
| 64×64 | 892 | 894 | +2 | 0/0/0 | 0/0/0 | +0.0755 | +0.0710 | 0.98929→0.98952 | 9.9009→9.8992 | +0.0710 | 9.8992 |
| 256×256 | 5,292 | 5,295 | +3 | 0/0/0 | 0/0/0 | +0.4137 | +0.3977 | 0.99466→0.99584 | 9.9495→9.9494 | +0.4481 | 9.9500 |
| 640×360 | 13,311 | 13,314 | +3 | 0/0/0 | 0/0/0 | +0.1212 | +0.1179 | 0.99273→0.99306 | 9.8867→9.8819 | +0.1452 | 9.8846 |
| 1920×1080 | 61,530 | 61,532 | +2 | 0/0/0 | 0/0/0 | +0.1429 | +0.1396 | 0.99500→0.99522 | 9.8976→9.8962 | +0.1714 | 9.8970 |

PSNR/CVVDP come from `tools/quality.py` (ffmpeg PSNR + ColorVideoVDP 0.5.7,
`standard_fhd`, MPS) on the decoded streams; "4-set" rows are the component
output written as Y4M and scored the same way.
`*` sample-weighted over 4:2:0 (Y counts 4× the chroma samples). "4-set plan" is
the component-only measurement: the same stage with `num_strengths = 4`
(off + 3 sets, 1–3 bits per 64×64 block: ~180 bits at 640×360, ~1,530 at 1080p)
and the per-block index grid applied by the CPU filter. It cannot be signalled
until the M1 writer codes `read_cdef` (see below).

What the search picks at QP 90: a weak luma set plus a strong luma/chroma set;
the strong-only candidates are rejected. In the 4-set plan most blocks pick
"copy" (35/60 at 640×360, 354/510 at 1080p, 6/16 at 256×256), i.e. CDEF helps a
quarter to a half of the 64×64 blocks — which is why a single global set is
close to the per-block result but the per-block grid is what prevents the
regressions (the 640×360 U plane above). At 64×64 every block is filtered, so
the planner prunes to one set and emits zero index bits.

Selected sets / CPU time (single-threaded reference, 640×360): plan 81 ms,
filter 8.6 ms. The GPU kernel is the product path; the search runs one kernel
launch for the whole table.

**Verdict at QP 90.** CDEF is a small, consistent **PSNR/SSIM** win:
+0.07 dB (64×64) to +0.41 dB (256×256) luma and +0.12/+0.14 dB at 640×360/1080p,
with SSIM-Y improving everywhere, for +2–3 header bytes and unchanged tile
bytes (per-block index selection adds another +0.02–0.05 dB).

It is **not** a perceptual win on this corpus: CVVDP moves by −0.0048 JOD at
640×360, −0.0014 at 1080p, −0.0001 at 256×256 and −0.0017 at 64×64. The 4-set
plan recovers part of that (−0.0021 at 640×360) but stays inside ±0.05 % of the
metric. So: worth wiring as an **opt-in** tool for its PSNR/SSIM gain at
near-zero rate, not worth claiming as a quality improvement until more content
and a rate-aware strength search are measured. Strong strengths (≥12) lose; the
search rejects them per frame. Chroma must be allowed to stay unfiltered (the
640×360 global set drops U by 0.04 dB while luma +0.12 dB; the per-block plan
recovers it, and a uv=0 set is in the ladder).

## What the M1 syntax owner must add (exact)

Everything below is a change to `src/encoder/m1/` — **not** part of this
workstream, and not applied here.

### 1. Sequence header (§5.4.10) — `still_headers.cpp`

`sequence_filter_config()` currently writes `enable_cdef = 0`. Set it to 1.
That is the only sequence-header change.

### 2. Frame header (§5.18.5.2 + §5.18.7.10) — `still_headers.cpp`

In `write_frame_header()`, immediately **after the four
`apply_deblocking_filter[]` bits** and **before `tx_mode_select`**, write:

```cpp
// §5.18.5.2: one f(1) per enabled plane, then
// §5.18.7.10 cdef_params().
if (c.apply_deblock_y_vert) {
  for (int i = 0; i < 4; ++i) writer.put_bit(false);  // df_delta_q_present[i] = 0
}
if (c.cdef_frame_enable) {
  // single_picture_header_flag => cdef_frame_enable = 1, no bit
  writer.put_bits(c.cdef_damping_minus_3, 2);      // 0 => CdefDamping = 3
  writer.put_bits(c.cdef_strengths_minus_1, 3);    // 0 => CdefStrengths = 1
  writer.put_bit(c.cdef_on_skip_txfm_frame_enable); // ADAPTIVE (single picture)
  for (int i = 0; i < c.cdef_strengths; ++i) {
    writer.put_bit(s.y_pri_zero[i]);
    if (!s.y_pri_zero[i]) writer.put_bits(s.y_pri[i], 4);
    writer.put_bits(s.y_sec_syntax[i], 2);   // 3 means the value 4
    writer.put_bit(s.uv_pri_zero[i]);
    if (!s.uv_pri_zero[i]) writer.put_bits(s.uv_pri[i], 4);
    writer.put_bits(s.uv_sec_syntax[i], 2);
  }
}
```

Use `CdefHeaderParams` from `still_cdef.h`: `damping_minus_3`,
`strengths_minus_1`, `on_skip_txfm_frame_enable`, `set[i].*` and `bits` are
already validated and mapped (including `4 -> 3`).

> **The four `df_delta_q_present` bits are mandatory.** `dav2d` f9dafbc
> consumes one f(1) per enabled deblocking plane (§5.18.5.2 in the HTML
> cross-reference). The current M1 writer omits them; for 64×64/640×360 the
> omission is invisible because the four bits fall in zero padding before the
> byte-aligned tile, so the stream is byte-identical either way. As soon as
> `cdef_params()` is inserted between them and `read_tx_mode()`, the bit
> position matters: without the four bits both the 64×64 and 640×360 CDEF
> streams are rejected (`Error decoding frame: Invalid argument`), and with
> them both decode and match the §7.18 reference byte-for-byte. The stream gate
> prints this check on every run (`df_delta_q_present check`). When deblocking
> is disabled (all four apply flags 0) no `df_delta_q_present` bit is written.

### 3. Mode info (§5.20.10.1) — `still_syntax.cpp`

`read_cdef()` runs in `intra_frame_mode_info`, **after `intra_segment_id()` and
before `read_intra_y_mode()`**, for every coding block whose `TreeType !=
CHROMA_PART`. In `still_syntax.cpp` that is exactly the start of the leaf emitter,
immediately before `if (!code_y_mode(node))`. M1 emits no segment/skip syntax
there, so the CDEF symbols are the first symbols of the leaf.

`BlockSyntax` has no CDEF field yet; add one (e.g.
`std::int8_t cdef_idx = 0;`), fill it from the plan in the CPU encoder and the
GPU bridge, and pass it to the helper below. Use the helper:

```cpp
#include "cuda/m1/cdef/still_cdef_index_syntax.h"
#include "cuda/m1/cdef/still_cdef_cdf.h"

// per frame, once (init_non_coeff_cdfs):
std::uint16_t cdef_index0[4][3];
std::memcpy(cdef_index0, kCdefIndex0Cdf, sizeof(cdef_index0));
std::uint16_t cdef_minus1[8] = {};
if (const auto* p = cdef_index_minus1_default(cdef_strengths))
  std::memcpy(cdef_minus1, p, sizeof(std::uint16_t) * cdef_strengths);
CdefIndexCdfs cdfs{&cdef_index0[0][0], cdef_minus1};
CdefIndexGrid grid;
grid.reset(mi_cols, mi_rows);            // clear_cdef(): -1 everywhere

// per coding block, in syntax order:
if (!cdef_encode_block_index(&grid, cdef_strengths, /*sb_size4=*/4,
                             tile_mi_col, tile_mi_row,       // MiColStart/RowStart
                             mi_row, mi_col, cdef_idx_value, cdfs,
                             [&](std::uint16_t* cdf, int n, int symbol) {
                               return sym(cdf, n, symbol);   // existing helper
                             })) {
  error = "cdef_idx not signallable";
  return false;
}
```

The helper codes nothing when `CdefStrengths == 1` (matching the spec), codes
one `cdef_index0` symbol with the §8.3.1 neighbour context (left and
above-within-SB `cdef_idx == 0`), then `cdef_index_minus_1` when
`CdefStrengths > 2`. Repeat calls for the same 64×64 are no-ops, which mirrors
`cdef_idx[r][c] != -1`.

`clear_cdef()` (§5.20.9.6) must run at each superblock start; with a 64×64 SB
and a single 64×64 CDEF block it is exactly `grid.set(sb_r, sb_c, -1)`
(`grid.reset()` at frame start covers it for one tile).

**CDFs.** `kCdefIndex0Cdf[4][3]` and the
`kCdefIndexMinus1With{3..8}Cdf` tables are in `still_cdef_cdf.h`, copied from
`spec/tables/default_cdf.json`. They are not in `still_cdf_tables.h` yet; add them
to `generate_still_tables.py` (the JSON keys are `Default_Cdef_Index0_Cdf` and
`Default_Cdef_Index_Minus1_WithN_Cdf`) or include the header directly.

### 4. Frame-level control

* `CdefStrengths == 1` is the zero-tile-bit mode used by the gate. It is the
  safe first step: only headers change.
* `CdefStrengths >= 2` requires step 3 and an all-zero strength set if the
  encoder should be able to leave a block unfiltered.
* `cdef_on_skip_txfm_frame_enable`: for M1 (intra, `skip_flag == 0`) either
  value decodes identically; `false` is the default in the ladder plan.

## CUDA parity

`still_cdef_gpu_test.cu` consumes `recon.bin` from the gate and requires:

* every candidate row of the GPU SSE table to equal the host table;
* the GPU table to select the same header strengths and `cdef_idx` grid;
* every filtered byte on all three planes to equal the CPU reference;
* the §5.18.7.10 header parity: `still_cdef_build_header()` /
  `CdefHeaderParams::bits` must equal the bits and the round-trip of the
  verified `filters_stream::write_cdef_params()` for both the CPU-selected and
  the GPU-selected plan, in the M1 single-picture context;
* the §5.20.10.1 index-grid parity: the selected grid must encode and decode
  through the normative §8.2 coder to the same `cdef_idx` for both plans;
* the copy path: an all-`-1` grid must copy every sample on all three planes
  byte-exactly (the `cdef_idx == -1` branch of §7.18.1);
* timings (search kernel, filter kernel) and the PSNR delta.

The header and index checks are the same ones `still_cdef_test.cpp` now runs on
the real dump on the host (`test_real_recon_dump`), so they are executed even
without a GPU; only the kernel byte comparison needs CUDA.

It needs `nvcc` on the 3060 box:

```sh
cd src/cuda/pipeline/still/cdef
./build_and_test.sh --gate tests/data/basic_640x360_1f.y4m --cuda --arch=sm_86
# optional fixture QP for the gate/dump:
./build-m1-cdef/run_stream_gate <src.y4m> <out_dir> --qp 1 --dav2d tools/bin/dav2d \
    --dump-planes /tmp/recon_qp1.bin
./build-m1-cdef/still_cdef_gpu_test /tmp/recon_qp1.bin
```

**Measured (RTX 3060, `sm_86`, CUDA 13.0, 2026-11-13).** The kernel byte
comparison and GPU timings ran:

```
QP90 640x360: still_cdef_test 517 checks, 0 failures; stream gate PASS
              search table 10x60, 0 mismatches; filter output Y/U/V 0 mismatches
              copy path 0 mismatches; still_cdef_gpu_test 129 checks, 0 failures
              timing: search 13.762 ms, filter 0.228 ms (20 iters)
QP1  640x360: gate PASS (1 all-zero strength set), filter/copy parity 0 mismatches
              still_cdef_gpu_test 129 checks, 0 failures
```

The kernel uses the same inline arithmetic as the CPU path
(`cdef_filter_sample`, `cdef_direction`, `cdef_luma_params`,
`cdef_chroma_params`, `cdef_eval_block`), and the test now proves it on both a
normal-QP and a near-lossless fixture.  The 4-set index stream round-trips
through the normative §8.2 coder.


## Indirect verification performed here (no M1 edits)

* The §7.18 filter arithmetic was cross-checked against `dav2d` end-to-end:
  enabling CDEF in a rewritten copy of real M1 streams and decoding produces
  **zero** byte differences on all three planes versus this stage's filter
  (64×64, 256×256, 640×360, 1920×1080).
* The rewritten headers are proven faithful by a self-check that must reproduce
  the M1 stream byte-for-byte before CDEF is enabled.
* `cdef_idx` symbol coding round-trips through the normative §8.2 coder for
  `CdefStrengths` 1..8 (including the neighbour context), but the index symbols
  cannot be injected into an M1 tile without step 3 above.

## Limitations / notes

* 8-bit 4:2:0 only, MI-aligned planes, separate source/destination.
* The stream gate needs both the baseline and the CDEF stream to decode, so it
  covers sizes whose baseline M1 stream decodes. 1280×720 currently fails at
  the **baseline** (`dav2d: Invalid argument`), i.e. before CDEF is involved;
  that is a pre-existing M1 item. 64×64, 256×256, 640×360 and 1920×1080 pass.
* The gate compares visible planes. Sizes that are not a multiple of 8 have
  MI-aligned padding rows/columns that `StillResult.recon_*` does not expose; the
  stage itself receives the padded planes and filters the full MI grid, but the
  gate cannot check those samples.
* Strength search is SSE-only (no rate term), over a fixed legal ladder of 10
  sets. A larger ladder or RDO would trade a few more bits per 64×64 block.
* No commit; no changes outside `src/encoder/filters/cdef/`,
  `src/cuda/cdef/` and `src/cuda/m1/cdef/`.
