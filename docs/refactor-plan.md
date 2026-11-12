# Refactor plan — preparing to restructure the tree

Status: **preparation only**. The goal is to make the eventual move
mechanical, reviewable, and provably bitstream-neutral.

**Sequencing decision (binding):** new AV2 features land as standalone,
gate-tested components. **No component is wired into the M1/M2 core until this
refactor is finished.** Integration is a separate pass that starts only after
Phase 7, so it is done once, against the final directory layout, instead of
twice.

## 0. Feature freeze policy

Adding a feature after a freeze is more expensive than removing one, so the
order is deliberate:

1. **The freeze starts only when no spec-audit row is unclassified.** Every
   `ABSENT`/`PARTIAL` row must be IMPLEMENTED, INTEGRATION, OUT-OF-SCOPE or
   IMPOSSIBLE with evidence (`docs/spec-audit/README.md`).
2. Until then, newly found gaps land **immediately** as leaf modules in their
   target domain (`src/encoder/<domain>/<tool>/`), never in core, and never
   with `CMakeLists.txt` edits.
3. Late modules cost one row in `refactor/mapping.csv`; the mapping is data, so
   they do not force a rebase of the move.
4. **No `src/encoder/**` path is moved while a feature agent owns it.** The
   refactor runs in lanes that cannot collide: build graph (`CMakeLists.txt`),
   `tools/refactor/`, docs, tests, and removal of ignored build artifacts.
5. Removal after the freeze is cheap and expected; duplication and dead
   components are deleted then, not now.

## 1. Why this is now urgent

The repository grew by ~20 independent components in one wave. Measured state:

| Metric | Value |
|---|---:|
| Source files (`*.cpp`, `*.cu`, `*.h`, `*.cuh`) | 1,064 |
| Lines under `src/` | ~504,800 |
| Translation units named in `CMakeLists.txt` | 74 |
| `build*.sh` scripts | 184 |
| Untracked / in-flight files | 144 |
| Files with no test | 401 |
| Files with no decoder gate | 59 |
| Orphan files (no includer, not in CMake) | 81 |
| Duplicate helper definitions | `qlookup` ×3, `coeff_cdf_slice` ×3 |
| Ignored build directories | 39 |

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
- **Gate inventory**: `ctest` list (10), `feature_gate.sh` (30 cases), every
  component `build_and_test.sh` added by the feature waves (including the 20
  in-flight feature modules),
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
3. `tools/still_verify/feature_gate.sh`;
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
- **No profile layer in this pass.** The existing `m0`/`m1`/`m2`/`m10`
  directories are dissolved: stream-writing logic moves to `pipeline/`,
  tool selection moves to one temporary entry point, and the profile concept
  is designed fresh afterwards. Carrying the current profile shape through the
  refactor would bake the old layering into the new tree.
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
  app/                      # temporary composition root (see note below)
    encode.cpp
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
| `src/encoder/basic_encoder.cpp` | `pipeline/still/` writer + `app/` defaults |
| `src/encoder/m1/{still_encoder,still_syntax,still_headers}` | `pipeline/still/` + `codec/*` hooks |
| `src/encoder/m1/<tool>_stream` | `codec/<domain>/<tool>/stream` |
| `src/encoder/m2/*` | `pipeline/video/` + `frame/` |
| `src/encoder/m10/*` | `pipeline/still/` (high-bit-depth path) |
| `src/encoder/pipeline`, `tiles`, `svc` | `pipeline/…` |
| `src/encoder/rc/twopass` | `codec/rate/rc/twopass` |
| `src/encoder/svc` | `pipeline/svc` |
| `src/encoder/hdr`, `metadata/{content_interpretation,types}` | `frame/metadata/…` |
| `src/encoder/hbd`, `format/chroma` | `codec/format/{hbd,chroma}` |
| `src/encoder/inter/{longterm,random_access,sef,leading,tlayers,drl_reorder,gme,temporal,advanced,me2,mc}` | `codec/inter/…` |
| `src/encoder/frame/mfh`, `syntax/small_header` | `frame/{mfh,small_header}` |
| `src/encoder/rc/hrd` | `codec/rate/hrd` |
| `src/encoder/sb256` | `codec/partition/sb256` |
| `src/encoder/m2/ddt` | `codec/transform/ddt` |
| `src/encoder/entropy/avg_cdf` | `codec/entropy/avg_cdf` |
| `src/encoder/tcq/parity` | `codec/transform/tcq/parity` |
| `src/encoder/quant/dcdeltaq` | `codec/quant/dcdeltaq` |
| `src/encoder/filters/lr_chroma` | `codec/filters/loop_restoration_chroma` |
| `src/encoder/cfl/ds_filters` | `codec/intra/cfl/ds_filters` |
| `src/multi_gpu/runtime` | `pipeline/dispatch/multi_gpu` |
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
`still_syntax.cpp` only when its call sites are already behind one function.

Exit per tool: its gate passes and the pinned streams are unchanged.

### Phase 4 — dissolve the profile directories

There is no target `profiles/` in this plan, and no preservation of the current
`m0/m1/m2/m10` shape. Work through the front-ends once, in this order:

1. Move stream assembly (sequence header + frame header + tile payload + OBU
   framing) out of `basic_encoder.cpp`, `still_encoder.cpp`, `video_frame.cpp`, and `m10`
   into `pipeline/still/` and `pipeline/video/`. This code is format logic, not
   profile logic, and is what most profiles will share.
2. Move per-profile defaults (tool set, QP policy, search effort, enabled
   features) into one flat `app/encode.cpp` with an explicit options struct.
   It must be boring and temporary: one translation unit, no `app/m0/`,
   `app/m1/` subdirectories.
3. Delete `src/encoder/basic_encoder.cpp`, `src/encoder/m1/`, `src/encoder/m2/`,
   `src/encoder/m10/` once nothing includes them.

Exit: the same streams from the same CLI flags, with the profile shape gone.
A new profile layer can then be designed against `pipeline/` + `codec/` without
inheriting the current directory layout.

### Phase 5 — de-duplicate shared math

Introduce `codec/common/` (or extend `spec/`) with a single implementation of
`Round2`, `Round2Signed`, `Clip1/3`, `qlookup`, `get_qindex`, coefficient-CDF
selection. Replace the 56/19/19 copies. **This is a behavior-bearing phase**:
gate each replacement individually, because a "harmless" rounding difference
changes the bitstream.

### Phase 6 — split the mega-files

Current hotspots and a target decomposition:

| File | LOC | Split into |
|---|---:|---|
| `encoder/lossless/lossless.cpp` | 3,291 | dpcm / fsc / lossless-q |
| `encoder/m1/filters_stream/filters_stream.cpp` | 2,340 | cdef / ccso / lr / deblock syntax |
| `cuda/m1/still_gpu.cu` | 2,245 | stage launchers per tool |
| `encoder/basic_encoder.cpp` | 2,161 | still writer moved to `pipeline/still` in phase 4 |
| `encoder/m2/video_syntax.cpp` | 2,102 | frame syntax / block syntax / coefficient coding |
| `encoder/m1/still_encoder.cpp` | 1,881 | decision / quant / recon / entropy |
| `encoder/rc/twopass/twopass.cpp` | 1,758 | allocation / VBV / planning |

Splitting these is the highest-risk part of the refactor. Do one file per
commit, moving functions verbatim, with `nm` diffing of the object files before
and after to prove the symbol set only reorganizes.

### Phase 7 — namespaces and docs

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

## 8. Integration pass — starts only after Phase 7

The 28 existing components plus the 20 features currently in flight are *not*
wired during the refactor. Once the tree is final, integration runs in
dependency order, one tool per commit, each with the same gate discipline:

1. shared primitives (shared `spec_math`, CDF registry, qlookup) — no stream change;
2. intra tools: MRL, palette, IntraBC, DIP, IBP, CfL variants, MHCCP;
3. transform/quant: QM, DC/AC delta-Q, IST, TCQ, parity hiding;
4. entropy/header: delta-Q/segmentation, average CDF, MFH, content
   interpretation, timing/decoder model/BRT, metadata types;
5. partitions/SB: 128 and 256 superblocks, SDP, extended partitions, sub-8;
6. filters: CDEF, CCSO, Wiener NS, PC-Wiener, GDF, chroma LR, film grain;
7. inter: multi-ref, compound, B-frames, DRL reorder, subpel, warp, GME,
   inter-intra, BAWP, TIP, OPFL, refine-MV, temporal MV, temporal layers;
8. frame types: long-term refs, OLK, RAS/switch, SEF, leading frames;
9. formats: 10-bit, 4:2:2/4:4:4/4:0:0;
10. pipelines: tiles, SVC, two-pass/RC, multi-GPU;
11. CLI and final pinned-stream re-baseline.

Rule: integration commits may change bitstreams (that is their purpose), but
they must be opt-in, default-off, and must leave every pinned default stream
byte-identical. A gate failure reverts the integration commit, not the pin.

## 9. Effort estimate

| Phase | Effort | Risk |
|---|---|---|
| 0 baseline + inventory | 1 session | low |
| 1 explicit build graph | 1–2 sessions | medium (FP flags) |
| 2 leaf moves | 1 session | low |
| 3 tool consolidation | 4–6 sessions | medium |
| 4 dissolve profiles | 3–4 sessions | **high** (front-end wiring) |
| 5 shared math | 2–3 sessions | **high** (bitstream) |
| 6 mega-file splits | 3–5 sessions | **high** |
| 7 namespaces + docs | 2 sessions | low |

Total roughly two and a half working weeks of focused sessions, dominated by
phases 4-6. Phase 0 is worth doing now even if the rest waits: the baseline manifest
and inventory are useful immediately for wiring the 19 pending components.
