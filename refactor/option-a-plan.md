# Option A move — execution plan

Branch: `refactor/option-a`. Feature freeze is in effect: no behaviour changes,
no new features, no stream changes.

## Rules

1. `git mv` only. No content edits in a move commit except include paths, source
   lists, and build-script root detection.
2. One domain per commit. After each commit run, on the CPU host:
   ```sh
   cmake --build build -j8 && ctest --test-dir build
   tools/refactor/verify_baseline.sh --check
   ```
   A byte change in any pinned stream means the move is wrong: revert, do not
   re-pin.
3. After each domain also configure the CUDA tree on the 3060 at least once
   before moving on:
   ```sh
   cmake --build build-cuda -j16 && ctest --test-dir build-cuda
   ```
4. Update `refactor/mapping.csv` as rows move.
5. Build scripts: replace hard-coded `cd ../../..` root detection with a
   find-up loop that walks to the directory containing `CMakeLists.txt`.
   Do this **before** moving that tool, so depth changes cannot break it.
6. Includes are rewritten with `git grep -l` + exact replacement, then the
   compiler is the check.
7. Do not move the top-level `spec/` directory; generated tables keep their
   existing include paths (`spec/tables/...`) to avoid touching generators.

## Target mapping

### Core

| Old | New |
|---|---|
| `src/io` | `src/core/io` |
| `src/bitstream` | `src/core/bitstream` |
| `src/profile` | `src/core/telemetry` |

### Partition

| Old | New |
|---|---|
| `src/encoder/partition` | `src/codec/partition/tree` |
| `src/encoder/partition_ext` | `src/codec/partition/ext` |
| `src/encoder/sb128` | `src/codec/partition/sb128` |
| `src/encoder/sb256` | `src/codec/partition/sb256` |
| `src/encoder/sdp` | `src/codec/partition/sdp` |
| `src/encoder/m1/sub8` | `src/codec/partition/sub8` |

### Intra

| Old | New |
|---|---|
| `src/encoder/intra_modes` | `src/codec/intra/modes` |
| `src/encoder/mrl` | `src/codec/intra/mrl/core` |
| `src/encoder/m1/mrl_stream` | `src/codec/intra/mrl/stream` |
| `src/encoder/palette` | `src/codec/intra/palette/core` |
| `src/encoder/m1/palette_stream` | `src/codec/intra/palette/stream` |
| `src/encoder/ibc` | `src/codec/intra/ibc/core` |
| `src/encoder/m1/ibc_stream` | `src/codec/intra/ibc/stream` |
| `src/encoder/cfl` | `src/codec/intra/cfl` |
| `src/encoder/dip` | `src/codec/intra/dip` |
| `src/encoder/ibp` | `src/codec/intra/ibp` |
| `src/encoder/m1/missing_bits` | `src/codec/intra/supplemental` |
| `src/encoder/m1/scc_flags` | `src/codec/intra/screen_content` |

### Inter

| Old | New |
|---|---|
| `src/encoder/inter/*` | `src/codec/inter/*` (same subdirectory names) |
| `src/encoder/m2/compound_stream` | `src/codec/inter/compound` |
| `src/encoder/inter/advanced` | `src/codec/inter/advanced` (unchanged name) |

### Transform

| Old | New |
|---|---|
| `src/encoder/tx_variants` | `src/codec/transform/tx` |
| `src/encoder/tx_type` | `src/codec/transform/tx_type` |
| `src/encoder/tx_secondary` | `src/codec/transform/secondary` |
| `src/encoder/transform/idtx_intra` | `src/codec/transform/idtx` |
| `src/encoder/tcq` | `src/codec/transform/tcq` |
| `src/encoder/qm` | `src/codec/transform/qm` |
| `src/encoder/quant/dcdeltaq` | `src/codec/transform/delta_q` |
| `src/encoder/m2/ddt` | `src/codec/transform/ddt` |
| `src/encoder/m2/tx_partition` | `src/codec/transform/tx_partition` |

### Entropy

| Old | New |
|---|---|
| `src/entropy/av2` | `src/codec/entropy/av2` |
| `src/encoder/entropy/avg_cdf` | `src/codec/entropy/avg_cdf` |
| `src/encoder/entropy/continuity` | `src/codec/entropy/cdf_continuity` |
| `src/encoder/m1/deltaq` | `src/codec/entropy/deltaq/still` |
| `src/encoder/m2/deltaq` | `src/codec/entropy/deltaq/video` |
| `src/encoder/lossless` | `src/codec/entropy/lossless` |

### Filters

| Old | New |
|---|---|
| `src/encoder/filters/deblock` | `src/codec/filters/deblock` |
| `src/encoder/filters/cdef` | `src/codec/filters/cdef` |
| `src/encoder/filters/restoration` | `src/codec/filters/loop_restoration` |
| `src/encoder/filters/gdf_stream` | `src/codec/filters/gdf` |
| `src/encoder/filters/pc_wiener_stream` | `src/codec/filters/pc_wiener` |
| `src/encoder/filters/lr_chroma` | `src/codec/filters/loop_restoration_chroma` |
| `src/encoder/filters/tip_deblock` | `src/codec/filters/tip_deblock` |
| `src/encoder/filters/ccso_reuse` | `src/codec/filters/ccso` |
| `src/encoder/filters/spec_tables` | `src/codec/filters/tables` |

### Rate

| Old | New |
|---|---|
| `src/encoder/rdo` | `src/codec/rate/rdo` |
| `src/encoder/rdoq` | `src/codec/rate/rdoq` |
| `src/encoder/m1/rdo_decide` | `src/codec/rate/decide` |
| `src/encoder/m1/rdo_loop` | `src/codec/rate/joint` |
| `src/encoder/rc` | `src/codec/rate/rc` |

### Frame, formats, pipeline

| Old | New |
|---|---|
| `src/encoder/hdr` | `src/frame/metadata/hdr` |
| `src/encoder/metadata/types` | `src/frame/metadata/types` |
| `src/encoder/metadata/gaps` | `src/frame/metadata/gaps` |
| `src/encoder/metadata/content_interpretation` | `src/frame/metadata/content_interpretation` |
| `src/encoder/frame/mfh` | `src/frame/mfh` |
| `src/encoder/frame/ref_fields` | `src/frame/ref_fields` |
| `src/encoder/syntax/small_header` | `src/frame/small_header` |
| `src/encoder/format/chroma` | `src/formats/chroma` |
| `src/encoder/hbd` | `src/formats/high_bitdepth` |
| `src/encoder/tiles` | `src/pipeline/tiles` |
| `src/encoder/m1/tiles_stream` | `src/pipeline/tiles/stream` |
| `src/encoder/m2/tile_groups` | `src/pipeline/tiles/groups` |
| `src/encoder/pipeline` | `src/pipeline/planner` |
| `src/encoder/svc` | `src/pipeline/svc` |
| `src/multi_gpu` | `src/pipeline/dispatch` |
| `src/encoder/conformance` | `src/core/conformance` |
| `src/encoder/tables/missing` | `src/spec_tables/missing` |

### Profiles (last, riskiest)

| Old | New |
|---|---|
| `src/encoder/m1` (remaining core files) | `src/pipeline/still` |
| `src/encoder/m1/filters_stream` | `src/codec/filters/stream` |
| `src/encoder/m1/transform_stream` | `src/codec/transform/stream` |
| `src/encoder/m1/partition_stream` | `src/codec/partition/stream` |
| `src/encoder/m1/qctx` | `src/codec/entropy/qctx` |
| `src/encoder/m2` (remaining core files) | `src/pipeline/video` |
| `src/encoder/m2/post` | `src/pipeline/video/post` |
| `src/encoder/m2/restoration` | `src/codec/filters/loop_restoration/video` |
| `src/encoder/m2/multiref` | `src/codec/inter/multiref` |
| `src/encoder/m2/advanced` | `src/codec/inter/advanced/video` |
| `src/encoder/basic_encoder.cpp` | `src/pipeline/still/basic_encoder.cpp` |
| `src/encoder/m10` | `src/pipeline/still/high_bitdepth` |
| `src/encoder/*.cpp` left | `src/app/` |

### CUDA mirror

`src/cuda/<tool>` → `src/cuda/codec/<domain>/<tool>` matching the CPU path,
`src/cuda/m1` → `src/cuda/pipeline/still`, `src/cuda/m2` →
`src/cuda/pipeline/video`, `src/cuda/format` → `src/cuda/formats`.

## Exit criteria

- No `src/encoder` directory (or only a temporary `app` shim).
- `src/{core,codec,frame,formats,pipeline,app,cuda}` only.
- CPU `ctest` 10/10, baseline `--check` OK, feature gate 30/30.
- CUDA `ctest` 10/10 on the 3060.
- `refactor/mapping.csv` complete; no `UNMAPPED`.
