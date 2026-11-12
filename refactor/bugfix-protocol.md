# Bug-fix protocol for subagents

This is the contract for parallel bug fixing. It complements
`refactor/integration-plan.md` (integration) and `AGENTS.md` (code rules).
Read it before claiming a bug.

## 1. Roles and concurrency caps

| Role | Count | May edit code | May merge |
|---|---:|---|---|
| Scout | up to 16 | **no** (read-only repro) | no |
| Fixer | 6–8 | own worktree only | no |
| Integrator | 2 (still root, video root) | shared composition roots | yes |
| GPU verifier | 1 | own worktree | no |
| Coordinator | 1 | bug board/claims only | no |

The caps exist because of the shared files listed in §5, not because of agent
tooling. Twenty fixers would collide; twenty scouts are fine.

## 2. Bug IDs, board and claims

Bug IDs: `BUG-<area>-<number>`, e.g. `BUG-INTRA-014`, `BUG-GPU-003`.
Areas match the lanes in §5 (`INTRA`, `TRANSFORM`, `PARTITION`, `ENTROPY`,
`FILTERS`, `INTER`, `FORMATS`, `PIPELINES`, `GPU`, `CLI`, `GATES`, `DOCS`).

The board is `refactor/bugs.csv` with columns:

```text
id,lane,title,severity,status,owner,branch,reproducer,gate,files,notes
```

Statuses: `open`, `claimed`, `reproduced`, `fixing`, `gated`, `merged`,
`blocked`, `duplicate`, `wontfix`.

Claim before writing code:

```sh
tools/agents/claim.sh claim BUG-INTRA-014 --owner <name> \
    --lane INTRA --branch fix/INTRA/BUG-INTRA-014-palette-row
tools/agents/claim.sh release BUG-INTRA-014 --owner <name>
tools/agents/claim.sh list
```

Claims live in the primary worktree under `.agents/claims/<id>/` (gitignored)
and are atomic. If a claim exists, do **not** start a second fix. Read the
holder, then either join as reviewer or mark the board entry `duplicate`.

## 3. Before starting: deduplicate

Run, in this order:

```sh
tools/agents/status.sh
grep -n '<BUG-ID>\|<symptom keywords>' refactor/bugs.csv
git branch -a --list '*<BUG-ID>*' '*<slug>*'
git log --all --oneline --grep='<BUG-ID>\|<symptom>'
```

Also inspect existing worktrees (`av2enc-wt/*`, `av2enc-fix/*`, `merge-*`) with
read-only git commands. If the fix already exists, add a reviewer note instead
of duplicating it. Cross-inspection is encouraged; cross-editing is forbidden.

## 4. Worktrees and branches

```sh
tools/agents/new_fix.sh BUG-INTRA-014 palette-row --lane INTRA
```

This creates:

- worktree `../av2enc-fix/BUG-INTRA-014`
- branch `fix/INTRA/BUG-INTRA-014-palette-row`
- symlinks for `tools/bin`, `testdata`, and the spec PDF

One bug per branch. Never commit `tools/bin`, `testdata`, build dirs or
`.agents/`. Rebase on `main` before requesting a merge.

## 5. Lane ownership and shared roots

Leaf lanes own their directories and gates:

| Lane | Owns |
|---|---|
| INTRA | `src/codec/intra/*` |
| TRANSFORM | `src/codec/transform/*`, `src/codec/rate/rdoq` |
| PARTITION | `src/codec/partition/*`, `src/pipeline/planner` |
| ENTROPY | `src/codec/entropy/*`, `src/frame/{mfh,metadata}` |
| FILTERS | `src/codec/filters/*`, `src/pipeline/grain` |
| INTER | `src/codec/inter/*` |
| FORMATS | `src/formats/*`, `src/pipeline/still/high_bitdepth` |
| PIPELINES | `src/pipeline/{tiles,svc,dispatch}`, `src/pipeline/video/post` |
| GPU | `src/cuda/**` |

**Shared roots are integrator-only:**

- `src/main.cpp`
- `CMakeLists.txt`, `tests/CMakeLists.txt`, `tests/gates/CMakeLists.txt`
- `src/pipeline/still/still_encoder.{h,cpp}`
- `src/pipeline/still/still_syntax.{h,cpp}`
- `src/pipeline/still/still_headers.{h,cpp}`
- `src/pipeline/video/video_frame.{h,cpp}`
- `src/pipeline/video/video_syntax.{h,cpp}`
- `src/pipeline/video/video_headers.{h,cpp}`
- `refactor/baseline.json`
- `docs/negative-results.md` (append-only; integrator merges)

If a leaf fix needs a shared-root hunk, put the exact patch or description in
the bug report and assign it to the integrator. Do not edit the shared file.

## 6. Workflow for a fixer

1. Claim the bug; create the worktree/branch.
2. Reproduce with the smallest command; record it in the board.
3. Add a failing check first (unit test, gate case, or pinned-stream check).
4. Fix the leaf cause. One logical change per commit; title
   `fix(<lane>): <BUG-ID> <what>`.
5. Run the minimum gates in §7.
6. Update the board row to `gated` with commit hash and gate output.
7. Request merge from the integrator; attach the report in §9.
8. After merge, mark `merged` and release the claim.

If the bug is not reproducible, mark `duplicate`, `wontfix` or `blocked` with
the evidence. Do not guess a fix.

## 7. Gates per branch (minimum)

```sh
cmake -S . -B build-fix -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DAV2ENC_ENABLE_CUDA=OFF -DAV2ENC_ALLOW_CPU_ONLY=ON
cmake --build build-fix -j3
ctest --test-dir build-fix                 # 10/10 default; 17/17 with -DAV2ENC_PRODUCT_GATES=ON
tools/refactor/verify_baseline.sh --check  # 19 streams, 30 features
```

Then the lane gate(s) from §5/`docs/validation.md` (`tools/test.sh --gates`). For pixel-affecting fixes,
also run `tools/validate_dav2d.sh` plus sample-exact recon (`still_recon_dump`,
`still_dump`, or `--video-recon`). Default streams must stay byte-identical
unless the bug *is* a pinned-stream bug; in that case the coordinator and
integrator approve the new pin explicitly.

CUDA fixes are verified only by the GPU verifier, in the shared queue. A CPU
build is a pre-test, not coverage.

## 8. Integrator rules

- Merge leaf branches in bug-ID order, rebase each on current `main`, run the
  branch's gate plus `ctest` + baseline after every merge.
- Own shared-root hunks; keep CLI/help/validation consistent.
- Never merge a branch that moves a default pin without an approved board row.
- One merge commit per branch: `merge: fix/<lane>/<bug-id>`.
- Keep `refactor/bugs.csv` current; release claims for merged rows.

## 9. Fixer report format

```text
BUG: <id> — <title>
Branch: fix/<lane>/<id>-<slug>  Head: <hash>
Root cause: <one paragraph>
Fix: <files and behaviour>
Reproducer: <exact command>
Gate: <commands and results; pins>
Cross-inspection: <branches/bugs checked to avoid duplication>
Risks/remaining: <known interactions>
Claim: released?  Board: updated to <status>
```

## 10. Anti-duplication checklist

- Board row exists and is claimed by me.
- No branch or open worktree matches the bug ID or symptom.
- No merged commit already references the bug ID.
- If a related fix exists, I am reviewing/continuing it, not rewriting it.
- Shared-root changes are assigned to one integrator, not copied into my tree.
- The GPU queue has one owner per verification run.
