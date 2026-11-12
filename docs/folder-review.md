# Folder structure review

Date: 2026-11-12. Scope: `src/` after the feature waves (26 feature tasks and
the CUDA parity wave). This is an assessment, not a move.

## Verdict

**Yes, the structure needs revision.** Implementation is now feature-complete;
the tree is the bottleneck. The problems are not aesthetic:

| Problem | Evidence |
|---|---|
| Tool code is grouped by *origin*, not domain | `inter/gme`, `m2/ddt`, `m2/tx_partition`, `m2/compound_stream`, `transform/idtx_intra`, `quant/dcdeltaq`, `syntax/small_header` all live in different parents |
| CPU/GPU split lives in two trees | 809 files under `src/encoder`, 266 under `src/cuda`, 1:1 per tool but no enforced naming |
| Adapter/core split | `m1/{mrl_stream,palette_stream,ibc_stream,scc_flags,missing_bits,tiles_stream,transform_stream}` next to cores in `encoder/{mrl,palette,ibc,...}` |
| Profile code mixed with tools | `encoder/m1` (15 subdirs) and `encoder/m2` (13) contain tools, not just composition |
| Build scripts | **212** `build*.sh` files |
| CMake reach | only **74** TUs in `CMakeLists.txt`; most of the tree is gate-only |
| Duplicate definitions | `qlookup` ×3, `coeff_cdf_slice` ×3 |
| Junk | empty `src/cpu`, `src/build-dcdeltaq`, `src/build-hdr`, `src/profile/build` |
| Orphans / untested | 81 orphan files, 401 without tests, 59 without a decoder gate |

The tree currently reads as “one directory per agent”. That was efficient during
the wave and is now the main navigation cost.

## Target layout (revised for the full feature set)

Two levels of ambition. **Recommendation: do A now, defer B.**

### Option A — domain grouping, GPU stays a mirror (lower risk)

```
src/
  core/           io, bitstream primitives, OBU, IVF, Y4M
  spec/           generated tables + spec_math (single source)
  codec/
    partition/    partition, partition_ext, sb128, sb256, sdp, sub8
    intra/        intra_modes, mrl, palette, ibc, cfl, dip, ibp,
                  missing_bits (DC-subsampled/corner), scc
    inter/        me, mc, merge, mv_stack, drl, warp, gme, temporal,
                  tip, compound, morph, mv_precision, bru, bframes
    transform/    tx_variants, tx_type, tx_partition, secondary, ddt,
                  idtx, tcq, qm, dcdeltaq
    entropy/      av2 coder, cdf, avg_cdf, continuity, deltaq, lossless
    filters/      deblock, cdef, ccso, lr (wiener, pc_wiener, gdf,
                  lr_chroma, tip_deblock), film_grain
    rate/         rdo, rdo_loop, rdo_decide, rdoq, rc, twopass, hrd
  frame/          headers, ref_fields, mfh, metadata (hdr, types,
                  gaps, content interpretation)
  formats/        chroma formats, hbd (10-bit)
  pipeline/       still, video, tiled (tiles, tile_groups), svc,
                  dispatch (multi_gpu)
  app/            one temporary composition root (profiles redesigned later)
  cuda/           mirror of codec/ + formats/ + pipeline/, same tool names
```

Rules:

- one tool, one directory name across CPU and GPU (`codec/intra/mrl` ↔
  `cuda/intra/mrl`);
- adapters (`mrl_stream`, `palette_stream`, `ibc_stream`, `scc_flags`) move
  **into** their tool as `stream/`, not under `m1/`;
- `m1`/`m2`/`m10` disappear as tool containers; their composition code moves to
  `pipeline/` and `app/`;
- `src/cuda/m1` and `src/cuda/m2` become `cuda/pipeline/{m1,m2}`.

### Option B — co-locate GPU inside each tool (cleaner, higher risk)

`codec/intra/mrl/{core,stream,gpu}` with one directory per tool.

This removes the twin-tree problem entirely but touches nearly every include and
every build script. Defer until after A, if at all.

## Sequencing

1. **Delete the four empty directories** — zero risk, do immediately.
2. **Freeze features** when the last CUDA gaps close (TCQ, chroma LR, GME).
3. **Move only after the baseline is recorded** — `refactor/verify_baseline.sh`
   already pins 19 streams, ctest 10/10, 30 feature-gate cases.
4. Move in this order: `core`/`spec` → `codec/partition` → `codec/intra` →
   `codec/transform` → `codec/filters` → `codec/inter` → `codec/entropy` →
   `codec/rate` → `frame` → `formats` → `pipeline` → `app` → `cuda` mirror.
5. One tool per commit; `git mv` only; update includes and source lists; run
   `verify_baseline.sh --check` and the tool’s own gate.

## What not to change

- Bitstream behaviour. A move must not alter a byte; the baseline check catches
  it and a failure means revert, never re-pin.
- Entropy code layout before the shared-CDF work; it is serial and stable.
- The generated tables’ contents; only relocate and de-duplicate the copies.

## Main risks

- 212 build scripts hard-code relative paths (`ROOT=$(cd ../../..)`); several
  will need path fixes, and the ones with pinned SHA paths need review.
- `AV2ENC_STILL_GPU_SOURCES` uses `GLOB_RECURSE ... CONFIGURE_DEPENDS`, so every
  new directory triggers a configure; make source lists explicit first
  (refactor Phase 1).
- Three duplicate helper definitions and hand-copied tables must be consolidated
  after the move, not during it.

## Bottom line

The structure is not broken enough to block correctness, but it is now the
largest source of friction and duplicated maintenance. Option A is a bounded,
mechanical change with a working baseline; Option B is a second, optional pass.
