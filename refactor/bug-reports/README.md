# Bug-scout instructions

You are a read-only bug scout for av2enc. Find **real, reproducible bugs** on the
current `main`; do not refactor or implement fixes.

## Hard limits

- Do **not** edit any file under `/Users/arin/work/av2enc` (or any worktree).
- Do **not** create branches/worktrees, commit, push, or run `git` write
  commands.
- Do **not** run `cmake --build`, `ctest`, or long gate suites. Use the
  existing binaries in `/Users/arin/work/av2enc/build`.
- Do **not** use the 3060, ssh, or the network.
- Keep commands short (< ~5 s each) and avoid large encodes/1080p loops.
- Write only to `/Users/arin/work/bug-reports/` and `/tmp`.

## What counts

Report only issues with concrete evidence:

- wrong/desynced bitstream, recon mismatch, or decoder rejection;
- accepted-and-ignored flag, missing validation, wrong rejection;
- crash/UB/race, stale path/probe, gate hole;
- a documented constraint that the code no longer enforces.

Classify `confirmed` (reproduced), `suspected` (strong static evidence, repro
blocked), or `doc/gap` (documentation/validation gap). Do not report style or
speculative cleanups.

## Deduplication

Before finalising, check:

```sh
grep -n '<keyword>' /Users/arin/work/av2enc/refactor/bugs.csv
git -C /Users/arin/work/av2enc log --all --oneline --grep='<keyword>'
ls /Users/arin/work/bug-reports/
```

If another report/board row already covers it, mark it `duplicate` and cite the
ID/file instead of re-reporting.

## Output

Write exactly one file: `/Users/arin/work/bug-reports/<lane>.md` (replace
`<lane>` with your assigned lane). Use:

```markdown
# <lane> bug scout

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| ... | high/med/low | confirmed/suspected/doc | ... | command | output/short | ... | new/duplicate |

## Details
### <id> — <title>
Repro:
```sh
...
```
Observed:
...
Expected:
...
Likely cause:
...
```

Use IDs `SCOUT-<LANE>-<nn>`. Keep the file under ~150 lines. In your final
message report the count of confirmed/suspected/doc bugs and the file path.
