# Folder-refactor Phase 0

This directory is the safety net for the tree move described in
[`../docs/refactor-plan.md`](../docs/refactor-plan.md). It is intentionally
**read-only with respect to `src/`**: nothing here moves, renames, edits, or
deletes source, and nothing here edits `CMakeLists.txt`.

The feature wave is still landing. The inventory therefore walks the
filesystem, not just `git ls-files`, so untracked in-flight feature
directories appear in the report and in `mapping.csv` with an explicit
`in-flight` note. Re-run both tools at the actual freeze/tag; a report is only
a snapshot of the moment it was generated.

Prerequisites: Python 3.9+, bash, CMake (the repo already requires 3.24+), and
`tools/bin/dav2d` or `DAV2D` for `verify_baseline.sh`.

## 1. `tools/refactor/inventory.py`

Read-only; no build and no source edits.

```sh
python3 tools/refactor/inventory.py
```

Writes:

- `refactor/inventory.json` — machine-readable report.
- `refactor/inventory.md` — the same data as a reviewable summary.

Options:

```sh
python3 tools/refactor/inventory.py \
    --root . --src src \
    --json refactor/inventory.json \
    --markdown refactor/inventory.md
```

What is in the report, per source file:

| Field | Meaning |
|---|---|
| `loc` | physical lines |
| `in_cmake`, `cmake_targets`, `cmake_glob` | named in a `CMakeLists.txt`, inferred target, or covered by the M1 GPU `GLOB_RECURSE` |
| `build_scripts` | `build*.sh` scripts in the file's own directory |
| `namespaces` | namespace declarations in the file (comments stripped) |
| `test_present`, `test_evidence` | file is a test, has a same-stem test, or is included by a test |
| `gate_present`, `gate_scripts` | a directory/ancestor gate script or gate source exists |
| `generated`, `generated_marker`, `generator` | generated-file marker plus the likely generator script |
| `git_tracked` | tracked at snapshot time; `false` means in-flight/untracked |

Top-level sections:

- **Plan vs observed** — the counts in `docs/refactor-plan.md` beside the
  observed snapshot. The plan is dated and its counts do not match the current
  tree. For the shared helpers the report separates **definitions** from
  **files mentioning (code)** and **files mentioning (raw)**, so a stale
  "definers" number cannot be mistaken for a duplicate-definition count.
- **Shared-helper definitions** — `Round2`, `Round2Signed`, `Clip1`, `Clip3`,
  `qlookup`, `get_qindex`, `coeff_cdf_slice`. A definition is a function body
  (`){`), not a call. `duplicate: true` means more than one file defines it.
- **Include graph** — quoted `#include "..."` edges, unresolved includes, and
  non-trivial strongly connected components (cycles).
- **Generated files** — marker plus the generator when one can be identified.
- **Orphans** — source neither in CMake nor covered by a directory
  `build*.sh`; candidates for cleanup, not for a move.
- **Files** — the full table.

Build directories (`build`, `build-*`, `build_*`), `__pycache__`,
`spec/tables/_cache`, and similar artefacts are skipped and listed under
`ignored_directories`.

## 2. `tools/refactor/verify_baseline.sh`

The baseline contract. It configures and builds from clean directories,
runs the gates, encodes every pinned fixture, and decodes with
`tools/validate_dav2d.sh`.

```sh
tools/refactor/verify_baseline.sh --record   # writes refactor/baseline.json
tools/refactor/verify_baseline.sh --check    # exit non-zero on any drift
```

Default output: `refactor/baseline.json`. Default builds:
`refactor/build/cpu` and `refactor/build/cuda` (gitignored generated
artefacts; safe to delete).

### What `--record` captures

- clean CPU configure/build times;
- CUDA configure/build times when `nvcc` is available (otherwise an explicit
  `cuda.skipped=true` with a reason — never a silent pass);
- `ctest` names, statuses, counts and wall time;
- `tools/still_verify/feature_gate.sh` counts and every case;
- encoder IVF MD5s for the M0 and M1 pinned fixtures, including the
  `--adaptive-cdf` M0 variants, the QP 91/141/191/255 qctx set, and the
  generated 1920×1080 M0 still;
- decoded MD5s produced through `tools/validate_dav2d.sh`;
- git revision/dirty state, host, compiler/CMake/nvcc versions, and the
  decoder version plus SHA-256.

`--record` **refuses to re-pin a changed stream**. The known constants already
published in `tools/still_verify/feature_gate.sh` and `tests/basic_encode_smoke.sh`
are embedded in the script as `known_md5`; if the current build disagrees,
recording stops. A baseline is a contract, not a snapshot of whatever the
encoder happened to output.

### What `--check` compares

Hard failures (exit 1):

- any ctest name/status/count change, including a new or missing test;
- any feature-gate case/count/MD5/byte change;
- any pinned stream's encoder MD5, decoded MD5, IVF byte count, args, or
  `validate_dav2d.sh` result;
- the same stream changing against the published pin, even if
  `baseline.json` was edited;
- baseline decoder version/SHA change;
- CPU build failure.

Warnings only (environment, not bitstream): compiler/CMake/nvcc version
changes and CUDA availability changes. Build times are recorded but not
compared.

Exit codes:

| Code | Meaning |
|---:|---|
| 0 | recorded, or check passed |
| 1 | drift, failed build/gate, or baseline refused |
| 2 | usage error or missing baseline |
| 77 | no decoder: decoded MD5s cannot be verified (explicit skip) |

The decoder check is deliberately early: with no `dav2d`, both modes exit 77
rather than recording or passing without decoded MD5s. `DAV2D=/path/to/dav2d`
overrides discovery.

Useful options:

```sh
# Smoke-check an existing binary without rebuilding; ctest is reported skipped,
# so this is not a substitute for the full --check.
tools/refactor/verify_baseline.sh --check --no-build --bin refactor/build/cpu/av2enc

# Fail rather than skip when CUDA cannot be attempted.
tools/refactor/verify_baseline.sh --check --require-cuda

# Keep logs.
tools/refactor/verify_baseline.sh --check --verbose
```

### Re-recording policy

Only re-record after a commit that is *supposed* to change a stream
(integration, not movement), and only with the corresponding pin update
reviewed. A refactor commit that changes a pinned MD5 is wrong: revert the
commit, do not update `baseline.json`.

## 3. `refactor/mapping.csv`

Columns: `old_path,new_path,note`. Seeded from the mapping table in
`docs/refactor-plan.md`, then extended so every current directory under
`src/` has a row, including:

- the ten immediate directories under `src/`;
- every current encoder, CUDA, filter, inter, profile, and pipeline module;
- `src/build-dcdeltaq`, `src/build-hdr`, `src/profile/build`, and the empty
  `src/cpu` as explicit delete/no-move rows;
- in-flight untracked feature directories with `in-flight` in the note.

A row with an empty `new_path` is a delete, never a move. Rows with a target
containing spaces/`+` (for example `src/pipeline/still + src/codec/*`) are
split destinations; child rows carry the concrete target. The file is data:
review it before writing any move script.

## 4. What is intentionally not automated

- **No moves, renames, deletes, include rewriting, or CMake edits.** The move
  script does not exist yet; mapping review comes first.
- **No automatic mapping decision.** Unmatched directories are marked
  `UNMAPPED` for review rather than guessed.
- **No full gate matrix.** Phase 0 runs `ctest` (10 tests) and
  `tools/still_verify/feature_gate.sh` (30 cases). It does not run every
  component `build_and_test.sh`, the 112-case M1 GPU matrix, `m3_streams`,
  `m3_gpu`, or hundreds of thousands of per-module checks (for example
  `qm_test: 258446`). Those still run per component owner.
- **No new pinned-stream suite.** The plan lists more future pins (tiled,
  palette/MRL/IBC, M2 2f/8f, QoS set). This script pins the currently
  published M0/M1 fixtures and the feature gate; adding new pins is a separate
  decision with its own reference output.
- **No CUDA build on a machine without nvcc.** The baseline records the
  absence explicitly and `--require-cuda` turns it into a failure.
- **No commits or tags.** `git tag pre-refactor-<date>` is a human step at the
  freeze.
- **No build-noise cleanup.** The inventory ignores it; deleting the 25
  `build-*` directories and other ignored artefacts remains a manual Phase 0
  step before the tag.
- **No code edits for Phase 5 duplication.** The inventory reports duplicate
  definitions; consolidation is behavior-bearing and gated separately.

## 5. Phase 0 exit criteria

The move may start only when all of these hold:

1. `inventory.json`/`inventory.md` were regenerated after the freeze commit;
   the plan-vs-observed table and all duplicate helpers have been reviewed.
2. `mapping.csv` has no `UNMAPPED` rows, every immediate `src/` directory and
   every in-flight directory has a row, and the delete rows are agreed.
3. `baseline.json` was recorded from a clean build: CPU build green,
   `ctest` 10/10, `feature_gate.sh` 30/30, all pinned encoder/decoder MD5s
   present, decoder version pinned. CUDA is either built green or explicitly
   recorded as unavailable.
4. `verify_baseline.sh --check` passes unchanged.
5. A deliberate one-byte change to a pinned stream makes `--check` exit
   non-zero (prove the net catches drift, not just that it prints OK).
6. With no decoder available, both modes exit 77 and do not write/pass a
   baseline.
7. The freeze tag exists (`git tag pre-refactor-<date>`) and build noise has
   been removed. Neither is done by these tools.
