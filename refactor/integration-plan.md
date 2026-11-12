# Integration plan — one branch per workstream

Status: active from 2026-11-12. Base commit is tagged `integration-base`.
Integration follows `docs/refactor-plan.md` §8 (dependency order, one tool per
commit, opt-in only). This file is the contract agents work under.

## 1. Why branches

The folder refactor is finished (`src/{core,codec,frame,formats,pipeline,spec_tables,cuda}`).
Integration must **not** move, rename, split or delete directories, and must not
reintroduce profile ids (`m0/m1/m2/m10`) or `src/encoder/`. Every workstream gets
its own `integration/<slug>` branch checked out in its own worktree, so two
agents can never edit the same file at the same time.

| Branch | Worktree (`/Users/arin/work/av2enc-wt/…`) | Scope (refactor-plan §8) | Wave |
|---|---|---|---|
| `integration/primitives` | `primitives` | shared `spec_math`, CDF registry, qlookup; no stream change | 1 |
| `integration/intra` | `intra` | MRL, palette, IntraBC, DIP, IBP, CfL variants, MHCCP, DC-subsampled | 1 |
| `integration/transform` | `transform` | QM, DC/AC delta-Q, IST, TCQ, parity hiding | 1 |
| `integration/entropy` | `entropy` | segmentation, average CDF, MFH, content interpretation, timing/decoder model/BRT, metadata | 1 |
| `integration/partition` | `partition` | 128/256 SB, SDP, extended partitions, sub-8 blocks | 1 |
| `integration/filters` | `filters` | CDEF, CCSO, Wiener NS, PC-Wiener, GDF, chroma LR, film grain | 1 |
| `integration/formats` | `formats` | 10-bit, 4:2:2 / 4:4:4 / 4:0:0 | 2 |
| `integration/inter` | `inter` | multi-ref, compound, B-frames, DRL, subpel, warp, GME, inter-intra, BAWP, TIP, OPFL, refine-MV, temporal MV/layers | 2 |
| `integration/frames` | `frames` | long-term refs, OLK, RAS/switch, SEF, leading frames | 3 |
| `integration/pipelines` | `pipelines` | tiles, SVC, two-pass/RC, multi-GPU | 3 |
| `integration/rebaseline` | (assistant) | CLI consolidation, final pinned-stream re-baseline | 4 |

Wave 2 starts only after wave 1 merges cleanly; wave 3 after wave 2. The
assistant owns merge, CUDA verification and the final re-baseline.

## 2. Hard rules for integration agents

1. **Stay in your worktree.** All commands run with `workdir=<your worktree>`.
   Never edit `/Users/arin/work/av2enc` (the main checkout) or another worktree.
2. **Do not restructure.** No directory moves/renames/deletes, no new top-level
   trees, no `src/encoder/`. Add files only inside the existing leaf directory
   for the tool.
3. **Opt-in only.** Every integrated tool gets a new CLI flag, default **off**.
   Default `--still` / `--still-gpu` streams must stay byte-identical.
   Test-only knobs go in the `Debug options` help section, not a new section.
4. **Never re-pin.** `tools/refactor/verify_baseline.sh --check` must pass before
   every commit. If a pinned MD5 moves, revert the change — do not update the
   baseline.
5. **One tool per commit.** Prefix `integration(<slug>): <tool>`. Keep local
   refactors separate from wiring. Small diffs merge; big diffs rot.
6. **CPU is the reference.** Verify locally with the CPU-only build. Do **not**
   ssh to the 3060 box and do not touch `/workspace/av2enc-a`; the assistant
   runs `build_gpu_test.sh` / CUDA `ctest` per branch after you report.
7. **No merging, rebasing or pushing.** Commit on your branch only. The
   assistant merges in order, resolves shared-file conflicts and re-runs CUDA
   gates.
8. **Stay in scope.** Shared files (`src/main.cpp`, `CMakeLists.txt`,
   `src/pipeline/still/*`, `src/pipeline/video/*`) may only receive the minimal
   wiring hunks for your tools. If you need a change in another group's module,
   report it as a dependency instead of editing it.

## 3. What "integrated" means

A tool is integrated when, following the existing `--still-cdef` pattern:

- the leaf implementation in `src/codec/<area>/<tool>/` (CPU) and the mirrored
  `src/cuda/...` (GPU, when one exists) is called from a composition root;
- a new CLI flag selects it (off by default), with parse + help wiring
  (product flags in `Options`, test knobs in the `Debug options` section);
- headers/syntax for the tool are emitted only when the flag is on;
- the reconstruction/report path applies the tool in normative order;
- a focused test/gate encodes a minimal stream and decodes it with
  `dav2d --strict 1` through `tools/validate_dav2d.sh`;
- module tests (CPU, and CUDA test source where applicable) are listed in the
  final report for the assistant to run on sm_86.

Do not claim end-to-end integration for a tool without a passing stream.

## 4. Local gate commands

```sh
cd /Users/arin/work/av2enc-wt/<slug>
cmake -S . -B build-int -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON
cmake --build build-int -j8
ctest --test-dir build-int            # must stay 10/10
tools/refactor/verify_baseline.sh --check
```

`tools/bin/` and `spec/AV2_Spec_v1.0.0.pdf` are symlinked into each worktree
(they are gitignored); use `tools/validate_dav2d.sh` for decoder gates.

## 5. Shared-file hot spots (expect merge conflicts; keep hunks tiny)

- `src/main.cpp` — Options struct, parse branches, help text.
- `CMakeLists.txt` — source lists; add lines only.
- `src/pipeline/still/still_encoder.{h,cpp}` / `still_headers.*` / `still_syntax.*`
  — options, header config, per-block syntax.
- `src/pipeline/video/video_*.{h,cpp}` — frame/block syntax.
- `docs/*.md` — status tables; prefer appending a short row.

## 5. Merge log

Merged to `main` with `--no-ff` after local CPU gates and a CUDA build+ctest on
the 3060. Each merge is re-verified on `main`:

| Merge commit | Branch | Notes |
|---|---|---|
| `c4e06fd` | `integration/primitives` | shared `spec_math`, qctx cross-check fix, generator paths; no stream change |
| `c44de03` | `integration/partition` | `--still-sb128`, `--still-sb256`, `--still-ext-partitions`; sub-8/SDP deferred |
| `f17745f` | `integration/entropy` | `--still-segmentation`, `--still-delta-q`, `--lossless`, `--still-content-interpretation` |
| `783c814` | `integration/transform` | `--still-qm`, `--still-ist`, `--still-dc-ac-delta-q` (renamed to avoid the entropy flag) |
| `f52d6bb` | `integration/intra` | `--still-ibp`, `--still-mrl`, `--still-cfl-ds`, `--still-cfl-explicit`, `--still-dip`, `--still-mhccp`; palette disabled |
| `549cb1d` | `integration/filters` | `--still-wiener`, `--still-chroma-lr`, `--still-pc-wiener`, `--still-gdf`, `--still-ccso-edge`, `--still-grain` |

Post-merge fixes on `main`:

| Commit | Fix |
|---|---|
| `f0f453f` | restore `--still-gpu` after the refactor moved `still_gpu.h` (stale `__has_include`) |
| `ebf09a8` | reject CPU-only intra flags in the `--still-gpu` path instead of ignoring them |

### Environment incident

The `integration/transform` branch accidentally committed the untracked
`tools/bin` and `testdata` symlinks (a `git add -A` after worktrees were
linked). Merging it replaced the real directories. Both were removed from the
index, the real directories were restored (testdata from the Vast.ai tree,
`dav2d-macos` re-downloaded from the pinned `f9dafbc` release), and
`.gitignore` now matches the names with or without a trailing slash so a
symlink cannot be committed again. All integration branches were checked; only
`integration/transform` was affected.

## 6. Agent report format

## 6. Agent report format

Final message must state: branch, commits (hash + title), tools wired and their
default state, files touched, CPU gates (build/ctest/baseline) with results,
new pinned test streams (flag + MD5, if any), CUDA tests the assistant must run,
remaining work, and merge-order/conflict notes.
