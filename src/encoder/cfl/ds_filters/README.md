# CfL luma down-sampling filters (`cfl_ds_filter_index`)

AV2 v1.0.0 §5.4.5 `sequence_intra_config()` / §6.4.5 semantics / §7.13.5
"Predict chroma from luma". The component owns the normative down-sampling
filters, their edge behaviour, and the encoder-side index search. It does not
edit `src/encoder/cfl/cfl.cpp`, `src/cuda/cfl/`, M1/M2 or any build file.

```
src/encoder/cfl/ds_filters/
├── ds_filters.h / .cpp        filters, prediction, SSE/RD search
├── ds_filters_test.cpp        22,950 CPU checks + pinned goldens
├── goldens.h                  pinned CRCs for every filter
├── build_and_test.sh          standalone CPU build (no CMake)
├── build_cuda_test.sh         CUDA seam test (skipped without nvcc)
├── ds_filters_cuda_test.cu    this component vs cfl.cpp vs cfl.cu
├── gate/
│   ├── build_gate.sh          builds an instrumented M1 copy (see below)
│   ├── check_gate.sh          non-zero index -> dav2d -> 0 recon diffs
│   └── parse_seq_index.py     independent sequence-header reader
└── quality/
    ├── quality_per_filter.sh  640x360 and 720p quality/rate deltas
    └── y4m_psnr.py            plane PSNR
```

## API

```cpp
#include "encoder/cfl/ds_filters/ds_filters.h"

using namespace av2enc::cfl;

// --- normative filters -----------------------------------------------------
int f = ds::normalise_index(3);              // 3 -> 0 (§6.4.5 note)
bool ok = ds::valid_index(3);                // every f(2) value is legal
int w = ds::weight_420(filter, dy, dx);      // dy,dx in {-1,0,1}
int w = ds::weight_422(filter, dx);
int w = ds::weight(filter, sub_x, sub_y, dy, dx);

// --- prediction ------------------------------------------------------------
ds::LumaScratch luma;
ds::compute_luma(block, filter, luma);       // uses the explicit index
// or ds::compute_luma(block, luma);         // uses block.ds_filter
int alpha_q8 = ds::derive_implicit_alpha(block, luma);
ds::apply_prediction(block, alpha_q8, luma);
std::int64_t sse = ds::prediction_sse(block, alpha_q8, luma, source);

// --- per-block index search ------------------------------------------------
ds::SearchOptions options;                   // SSE mode by default
auto choice = ds::search_filter(u, v, src_u, src_v, options,
                                SearchMode::kExhaustiveSse, scratch);
choice->index;                               // 0, 1 or 2
choice->evaluation.alpha;                    // existing cfl::search_alpha pair
choice->costs[0..2].total_sse();

// --- per-sequence index search ---------------------------------------------
ds::SequenceFilterSearch seq;
for (each CfL block)
  seq.add_block(u, v, src_u, src_v, SearchMode::kExhaustiveSse, scratch);
int sequence_index = seq.choose(options);    // signal this in §5.4.5
```

`ds::SearchOptions::rate_distortion` adds `lambda * rate_bits[index]`. The
syntax element is sequence-level, so its rate is constant per sequence and SSE
and RD select the same index unless a caller supplies a non-constant rate
model. Ties are deterministic: only a *strictly* lower cost replaces the
current best, so equal costs select the lowest index.

Alpha selection is delegated to the existing `cfl::search_alpha()`; the
down-sampling component deliberately does not fork the bitstream's alpha
decision.

## Normative filters

| Index | 4:2:0 | 4:2:2 |
|---|---|---|
| 0 | `{{0,0,0},{0,2,2},{0,2,2}}` | `{0,4,4}` |
| 1 | `{{0,0,0},{1,2,1},{1,2,1}}` | `{2,4,2}` |
| 2 | `{{0,1,0},{1,4,1},{0,1,0}}` | `{0,8,0}` |
| 3 | aliases 0 | aliases 0 |

4:4:4 (and the spec's 4:4:0 fallback) is `8 * v`. Every filter sums to 8, so
constant luma maps to itself. All arithmetic is integer (`Round2Signed` /
`approx_divide` from `cfl_common.h`).

Edge handling is the §7.13.5 one, and is identical to `cfl.cpp`:

* above luma clamps to `max(sbTop - 1, y - 1 - subY + dy)`;
* plane reads clamp to `luma_max_x/y`;
* above/left average loops replicate the last sample when the block crosses
  `MiCols*MI_SIZE >> subX` / `MiRows*MI_SIZE >> subY`;
* array `L` clamps negative taps on the first block row/column and on a
  64-sample luma boundary;
* partial and odd blocks are exercised by the synthetic fixtures, including a
  coded 30x30 grid with a block that extends past it.

## Tests

```sh
./src/encoder/cfl/ds_filters/build_and_test.sh
SANITIZE=1 WERROR=1 ./src/encoder/cfl/ds_filters/build_and_test.sh
GATE=1 QUALITY=1 ./src/encoder/cfl/ds_filters/build_and_test.sh
CUDA=1 ./src/encoder/cfl/ds_filters/build_cuda_test.sh   # on the GPU box
```

Measured on the macOS reference host:

```
ds_filters_test: 22950 checks, 0 failures
ASan/UBSan:     22950 checks, 0 failures
```

Coverage:

* every `Cfl_Filters_420` / `Cfl_Filters_422` entry against literal arrays,
  index-3 aliasing, 4:4:4 fallback, filter sums;
* byte equality with `cfl::compute_luma` / `cfl::apply_prediction` /
  `cfl::derive_implicit_alpha` for all 4 indices on 5 synthetic edge cases and
  5 real 720p blocks;
* constant-luma invariance (all filters identical) and a non-uniform fixture
  where filters 0/1/2 differ;
* deterministic search, manual-minimum cross-check, RD penalty, sequence
  accumulator, and a tie that must choose index 0;
* 40 pinned golden entries (5 synthetic + 5 real blocks x 4 filters), each
  pinning array `L`, `implicitAlpha`, prediction with `alpha = +2` eighths and
  prediction with derived alpha. Regenerate deliberately with
  `build/ds_filters_test --print-goldens`.

## GPU seam

No kernel edit is needed. `cfl::Block::ds_filter` is the ABI field:

```
ds::search_filter / SequenceFilterSearch::choose
        -> int cfl_ds_filter_index
        -> Block.ds_filter  (0..3, 3 aliases 0)
        -> src/cuda/cfl/cfl.cu: cfl_predict[_derived]()
```

`src/cuda/cfl/cfl.cu` already normalises 3->0 and calls the shared
`ds_weight()`; its existing test already sweeps indices 0..3.
`ds_filters_cuda_test.cu` adds an independent three-way check (this component
vs `cfl.cpp` vs `cfl.cu`) and runs the host search, then applies the chosen
index on the GPU. The M1 GPU path has the same seam:
`src/cuda/m1/cfl/m1_cfl.h:70` (`frame.ds_filter`) and
`src/cuda/m1/chroma/m1_chroma.h:61` (`M1ChromaOptions::ds_filter`).

This host has no `nvcc`, so the CUDA test is delivered but not run here; the
CPU path and the M1 gate below are the executed evidence.

## Decoder gate

`gate/check_gate.sh` emits real M1 streams with `cfl_ds_filter_index` 0, 1 and
2, confirms the value with an independent sequence-header parser, dumps the
encoder reconstruction, decodes with the pinned
`dav2d --strict 1 --threads 1`, and compares every plane:

```
index  parsed  Y/U/V diffs            result
0      0         plane Y: exact ...   BIT-EXACT
1      1         plane Y: exact ...   BIT-EXACT
2      2         plane Y: exact ...   BIT-EXACT
check_gate: all three indices bit-exact
```

The production M1 writer hardcodes index 0 and M1 is owned by another agent, so
`build_gate.sh` does **not** modify the repository sources. It copies
`m1_still.cpp` and `m1_headers.cpp` into `gate/build/`, applies a two-line
instrumentation patch to those copies (both read `AV2ENC_CFL_DS_FILTER`), and
links them in front of the existing `libav2enc_core.a`. The patch is exactly
the integration change listed below.

## Quality / rate per filter

`quality/quality_per_filter.sh`, real `shine` content, `--m1`, decoded with the
pinned dav2d (deltas relative to index 0):

| Size | Filter | Bytes | Y PSNR | U PSNR | V PSNR | Weighted PSNR | Δbytes | ΔW PSNR |
|---|---|---|---|---|---|---|---|---|
| 640x360 | 0 | 14046 | 44.7999 | 51.3830 | 50.1759 | 46.0400 | 0 | 0 |
| 640x360 | 1 | 14471 | 44.7999 | 49.6793 | 50.1851 | 45.9402 | +425 | -0.0998 |
| 640x360 | 2 | 15023 | 44.7999 | 48.8186 | 49.7492 | 45.8454 | +977 | -0.1946 |
| 720p | 0 | 35142 | 41.6886 | 53.6473 | 52.3825 | 43.2908 | 0 | 0 |
| 720p | 1 | 36604 | 41.6886 | 51.3667 | 51.9398 | 43.2355 | +1462 | -0.0553 |
| 720p | 2 | 37876 | 41.6886 | 50.3093 | 51.4672 | 43.1938 | +2734 | -0.0970 |

On this content index 0 is best; the search must therefore be allowed to keep
it (it does, and ties go to 0). The data also shows the filters are not
aliases: non-zero indices change the prediction and the coded rate.

## Exact M1/M2 call sites

| Site | Current code | Required integration |
|---|---|---|
| `src/encoder/m1/m1_still.cpp:1217` | `block.ds_filter = 0;` in `make_cfl_block()` | set the chosen sequence index |
| `src/encoder/m1/m1_still.cpp:1094-1104` | mode-decision `compute_luma` / `derive_implicit_alpha` / `apply_prediction` | unchanged; inherits `block.ds_filter` |
| `src/encoder/m1/m1_still.cpp:1330-1340` | final reconstruction CfL path | unchanged; inherits `block.ds_filter` |
| `src/encoder/m1/m1_headers.cpp:118` | `writer.put_bits(0U, 2U); // cfl_ds_filter_index` | write `ds::clamp_sequence_index(chosen)` |
| `src/encoder/m1/m1_still.cpp:1814` | `config.enable_cfl_intra = options.enable_cfl;` | add the sequence index beside it |
| `src/encoder/m2/m2_headers.cpp:145` | hardcoded 0 | write the chosen index |
| `src/encoder/m2/post/tiled_frame.cpp:155` | hardcoded 0 | write the chosen index |
| `src/encoder/m2/multiref/gate/compound_stream.cpp:126` | hardcoded 0 | gate-only; keep or parameterise |
| `src/encoder/m2/m2_syntax.cpp:222-225, 881-891, 1970` | CfL syntax flags only | no prediction path yet; when M2 CfL prediction lands, set `Block.ds_filter` as in M1 |
| `src/cuda/m1/cfl/m1_cfl.h:70`, `m1_cfl.cu:64` | `frame.ds_filter = block.ds_filter` | already generic |
| `src/cuda/m1/chroma/m1_chroma.h:61`, `m1_chroma.cu:56,140` | validates `options.ds_filter` | already generic |
| `src/cuda/m1/sched/sched_ref_bench.cu:568` | `cb.ds_filter = 0` | pass the sequence index in a real run |
| `src/cuda/cfl/cfl.h` / `cfl.cu` | consumes `Block.ds_filter` via shared `ds_weight` | no change |

## Limitations

* Only 8-bit 4:2:0 is an encoder format; 4:2:2/4:4:4 are table-tested only.
* The search uses SSE (or a caller-supplied rate model), not a full RDO loop;
  alpha selection remains the existing exhaustive SSE search.
* The gate builds instrumented *copies* of two M1 translation units because
  this agent does not own M1. The production integration is the two-line patch
  shown in `gate/build_gate.sh`.
* CUDA execution is not available on this host; the seam test must be run on
  the Ampere box (`build_cuda_test.sh`).
