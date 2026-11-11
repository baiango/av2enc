# Refactor plan — preparing to restructure the tree

Status: **preparation only**. No file moves are proposed until the current
feature wave is wired and every component has a stream gate. The goal of this
document is to make the eventual move mechanical, reviewable, and provably
bitstream-neutral.

## 1. Why this is now urgent

The repository grew by ~20 independent components in one wave. Measured state:

| Metric | Value |
|---|---:|
| Source files (`*.cpp`, `*.cu`, `*.h`, `*.cuh`) | 689 |
| Compiled translation units | 375 |
| Lines under `src/` | ~223,000 |
| Standalone `build*.sh` scripts | 124 |
| CMake libraries for all CPU code | 1 (`av2enc_core`) |
| CUDA targets | 1 (`av2enc_m1_gpu` OBJECT) |
| Root-level build directories | 25 |
| Files defining `Round2` | 56 |
| Files defining/using `qlookup` | 19 |
| Files defining `Clip3` | 19 |

The build is not the problem yet; the *navigation and duplication* are. There
are five separate RDO trees (`encoder/rdo`, `encoder/rdoq`,
`encoder/m1/rdo_decide`, `encoder/m1/rdo_loop`, `encoder/m2/rdo`), every new
tool lives as `encoder/<tool>` **plus** `encoder/m1/<tool>_stream` **plus**
`src/cuda/<tool>`, and shared spec math is copied per module.

## 2. Non-negotiable constraints

1. **Bitstream stability.** A refactor commit must not change a single byte of
   any pinned stream. If a pinned MD5 changes, the commit is wrong — do not
   re-pin it.
2. **No behavior change mixed with movement.** Move-only commits; logic edits
   are separate and individually gated.
3. **Both build modes stay green.** CPU-only macOS and CUDA `sm_86` Linux
   build and pass every gate after every commit.
4. **`-ffp-contract=off` must follow every target that contains transform
   math.** Splitting `av2enc_core` is the single most likely way to silently
   change 720p/rectangular bitstreams.
5. **One writer.** No background agents during the move.
6. **History is preserved in the working repo.** Use `git mv`; the publish
   clone is re-exported *after* the refactor, never rebased mid-move.

## 3. Phase 0 — freeze, inventory, baseline

Do this before touching a path.

### 3.1 Drain and freeze

- Let every running agent report; commit or park its directory.
- Tag the tree: `git tag pre-refactor-<date>`.
- Record the exact commit, OpenCode session counts, and tool versions in the
  baseline file.
- Clean untracked build noise so `find`/`grep` inventories are honest:
  remove the 25 `build-*` directories, `src/build-hdr`, `__pycache__`,
  `spec/tables/_cache`. All are gitignored; none should survive into moves.

### 3.2 Baseline manifest (the contract)

Generate `refactor/baseline.json` plus a human-readable
`docs/refactor-baseline.md` containing:

- **Pinned streams**: M0 64/640, M1 64/640, native 1080p, QP 90/91/141/191/255
  qctx set, M2 2f/8f 720p, tiled 2/4/8/16/32-tile cases, palette/MRL/IBC
  gate streams. Store *encoder MD5* and *decoded MD5* separately.
- **Gate inventory**: `ctest` list (10), `feature_gate.sh` (30 cases),
  `m3_streams`, `m3_gpu` matrix, M1 GPU matrix (112 cases), per-module check
  counts (e.g. `qm_test: 258446`).
- **Timings**: clean CPU build, clean CUDA build, full `ctest`, so later
  regressions in build time are visible.
- **Toolchain**: clang/gcc/nvcc versions, `dav2d` commit `f9dafbc`, CUDA arch.

### 3.3 Verification script

Add `tools/refactor/verify_baseline.sh` with two modes:

```sh
tools/refactor/verify_baseline.sh --record   # writes refactor/baseline.json
tools/refactor/verify_baseline.sh --check    # exits non-zero on any drift
```

`--check` must run, in order:

1. configure + build CPU and CUDA;
2. `ctest`;
3. `tools/m1_verify/feature_gate.sh`;
4. the encoder MD5 table (rebuild every pinned stream, compare bytes);
5. the decoder MD5 table (`dav2d --strict 1 --threads 1`);
6. module gates whose paths moved in the current phase.

Until `--check` is trustworthy, no refactor begins. It is the entire safety
net; a move that cannot be verified this way is not ready to move.

### 3.4 Inventory and mapping table

Add `tools/refactor/inventory.py` (read-only, no build) that emits:

- every source file with LOC, owning CMake target, namespace, and `build*.sh`;
- an include graph (`#include "..."` edges) and its strongly connected parts;
- duplicate symbol definitions for the shared helpers (`Round2`,
  `Round2Signed`, `Clip1/3`, `qlookup`, `get_qindex`, `coeff_cdf_slice`);
- files with no test, and tests with no gate;
- all generated files and the script that regenerates each.

Then write `refactor/mapping.csv` — **one row per file**, old path → new path,
with a reason column. This is reviewed as data, not as a diff. The move
scripts consume it, so the review happens before anything moves.

## 4. Target layout

Design rules:

- **One tool, one directory.** A tool owns its CPU core, syntax/stream adapter,
  GPU implementation, tests, and gate. No more `encoder/x` + `encoder/m1/x_stream`
  + `cuda/x` triples.
- **Depth is capped at four** below `src/`.
- **Front-ends are out of scope for now.** `src/encoder/m0_still.cpp`,
  `src/encoder/m1/`, `src/encoder/m2/`, and `src/encoder/m10/` keep their
  current paths; they will be re-pointed at the consolidated tools but not
  reorganized into a separate layer in this pass.
- **GPU code is co-located** (`<tool>/gpu/`), because it is always parallel to
  the CPU path and drifts when separated. Targets stay split so CPU-only builds
  remain possible.

```
CMakeLists.txt              # options + subdirs only
cmake/                      # warnings, fp-contract, cuda helpers
src/
  core/                     # bitwriter/reader, range coder, OBU, IVF, Y4M, geometry
  spec/                     # generated tables + spec_math (single source of truth)
  codec/
    intra/
      edges/  modes/
      cfl/{derived,alpha,mhccp}/
      mrl/{core,syntax,stream}/
      palette/{core,syntax,stream}/
      ibc/{core,syntax,stream}/
      dip/  ibp/
    inter/
      me/  mc/  merge/  subpel/  warp/  gme/  temporal/  compound/
    transform/
      tx/  tx_type/  secondary/  tcq/  qm/
    entropy/
      cdf/  deltaq/  segmentation/
    filters/
      deblock/  cdef/  ccso/  lr/  wiener/  pc_wiener/  gdf/
    rate/
      rdo/  rdoq/  rc/  twopass/
    lossless/
  frame/                    # sequence header, frame header, syntax primitives
  pipeline/
    still/  video/  tiled/  svc/  grain/  dispatch/
  encoder/                  # unchanged front-ends (m0/m1/m2/m10) for now
    m0_still.cpp  m1/  m2/  m10/
  gpu/
    common/                 # arena, streams, events, staging
tests/
tools/
docs/
```

Mapping of today's directories:

| Today | Target |
|---|---|
| `src/encoder/m1/mrl_stream` + `src/encoder/mrl` + `src/cuda/mrl` | `codec/intra/mrl/{core,syntax,stream,gpu}` |
| `src/encoder/m1/palette_stream` + `palette` + `cuda/palette` | `codec/intra/palette/…` |
| `src/encoder/m1/ibc_stream` + `ibc` + `cuda/ibc` | `codec/intra/ibc/…` |
| `src/encoder/cfl` + `cfl/alpha` + `cfl/mhccp` | `codec/intra/cfl/{derived,alpha,mhccp}` |
| `src/encoder/dip` + `cuda/dip` | `codec/intra/dip/{cpu,gpu}` |
| `src/encoder/ibp` + `cuda/ibp` | `codec/intra/ibp/{cpu,gpu}` |
| `src/encoder/{rdo,rdoq}` + `m1/rdo_decide` + `m1/rdo_loop` + `m2/rdo` | `codec/rate/{rdo,rdoq}/` + `codec/rate/rdo/loop/` |
| `src/encoder/{partition,partition_ext,sb128,sdp}` | `codec/partition/` (one geometry module) |
| `src/encoder/tiles` + `m1/tiles_stream` + `m2/post/tiled_frame` | `pipeline/tiled/` |
| `src/encoder/filters/restoration/{classes,spec_tables}` | `codec/filters/{lr,pc_wiener,gdf}` |
| `src/encoder/{qm,tx_secondary,tcq}` | `codec/transform/…` |
| `src/encoder/{m0_still,m1,m2,m10}` | unchanged (deferred) |
| `src/encoder/rc/twopass` | `codec/rate/rc/twopass` |
| `src/encoder/svc` | `pipeline/svc` |
| `src/encoder/hdr` | `frame/metadata/hdr` |
| `src/multi_gpu/runtime` | `pipeline/dispatch/multi_gpu` |

## 5. Execution order

Each phase is a batch of commits with the same exit criteria: CPU+CUDA build,
`ctest`, baseline `--check`, no MD5 change.

### Phase 1 — make the build graph explicit (no moves)

- Replace the `GLOB_RECURSE ... CONFIGURE_DEPENDS` for GPU sources with an
  explicit list per target.
- Split `av2enc_core` into object libraries per current top-level module
  (`av2enc_core_base`, `…_codec`, `…_filters`), all still under
  one interface target so link order is unchanged.
- Attach `-ffp-contract=off` to **each** new target that contains transform
  code, and add a configure-time assertion that the flag is present.
- Convert 10–15 of the 124 `build*.sh` scripts into CMake tests; leave the
  rest as wrappers calling one shared runner.

Exit: same binaries bit-for-bit; `ctest` grows, nothing else changes.

### Phase 2 — move leaves first

Move modules with no dependents of their own: `io`, `bitstream`, `entropy`,
`spec/tables`, `format`, then the small tool cores (`mrl`, `dip`, `ibp`,
`tcq`, `qm`). One commit per tool. Update includes with a script that rewrites
only quote-includes; fix CMake source lists manually.

### Phase 3 — consolidate one tool at a time

For each tool: bring `core`, `syntax`, `stream`, `gpu`, `tests` together, keep
the exact same entry points, and make the tool export a single header
(`<tool>/<tool>.h`) that the M1/M2 encoder includes. Extract the stream adapter out of
`m1_syntax.cpp` only when its call sites are already behind one function.

Exit per tool: its gate passes and the pinned streams are unchanged.

### Phase 4 — de-duplicate shared math

Introduce `codec/common/` (or extend `spec/`) with a single implementation of
`Round2`, `Round2Signed`, `Clip1/3`, `qlookup`, `get_qindex`, coefficient-CDF
selection. Replace the 56/19/19 copies. **This is a behavior-bearing phase**:
gate each replacement individually, because a "harmless" rounding difference
changes the bitstream.

### Phase 5 — split the mega-files

Current hotspots and a target decomposition:

| File | LOC | Split into |
|---|---:|---|
| `encoder/lossless/lossless.cpp` | 3,291 | dpcm / fsc / lossless-q |
| `encoder/m1/filters_stream/filters_stream.cpp` | 2,340 | cdef / ccso / lr / deblock syntax |
| `cuda/m1/m1_gpu.cu` | 2,245 | stage launchers per tool |
| `encoder/m0_still.cpp` | 2,161 | deferred with the M0 front-end |
| `encoder/m2/m2_syntax.cpp` | 2,102 | frame syntax / block syntax / coefficient coding |
| `encoder/m1/m1_still.cpp` | 1,881 | decision / quant / recon / entropy |
| `encoder/rc/twopass/twopass.cpp` | 1,758 | allocation / VBV / planning |

Splitting these is the highest-risk part of the refactor. Do one file per
commit, moving functions verbatim, with `nm` diffing of the object files before
and after to prove the symbol set only reorganizes.

### Phase 6 — namespaces and docs

- Make namespace mirror path: `av2enc::codec::intra::mrl`, etc. A namespace
  rename is a separate commit per top-level namespace, no logic edits.
- Reorganize `docs/` into `architecture/`, `formats/`, `gates/`, `results/`,
  `negative/`, and add a single `docs/README.md` index.
- Move the 24 result docs' gate commands into `tools/run_gates.sh`.

## 6. Review rules for refactor commits

- Title prefix `refactor:`; body states the phase and the mapping rows touched.
- A move commit may only contain `git mv`, include-path updates, and CMake
  source-list updates. No function edits.
- `git diff -M --stat` must show renames (>90% similarity) for moved files.
- Never re-pin a hash. Never "fix" a mismatch by regenerating expectations.
- Never combine a move with a feature wire-up.
- Keep a tag per phase so any phase can be bisected or reverted wholesale.

## 7. Rollback

- `git tag pre-refactor-<date>` plus per-phase tags.
- The publish clone is rebuilt only after the full sequence lands; the
  rewritten-history export is treated as a release step, not part of the
  refactor.
- If a phase cannot pass `verify_baseline.sh --check`, revert the phase tag
  rather than debugging forward.

## 8. Effort estimate

| Phase | Effort | Risk |
|---|---|---|
| 0 baseline + inventory | 1 session | low |
| 1 explicit build graph | 1–2 sessions | medium (FP flags) |
| 2 leaf moves | 1 session | low |
| 3 tool consolidation | 4–6 sessions | medium |
| 4 shared math | 2–3 sessions | **high** (bitstream) |
| 5 mega-file splits | 3–5 sessions | **high** |
| 6 namespaces + docs | 2 sessions | low |

Total roughly two working weeks of focused sessions, dominated by phases 4
and 5. Phase 0 is worth doing now even if the rest waits: the baseline manifest
and inventory are useful immediately for wiring the 19 pending components.
