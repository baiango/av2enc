# Tool-improvement loop

Status: standing agent loop for **implementation quality**, not new tools or
reference-topology changes. It improves existing AV2 tools one contained patch
at a time and keeps only measured wins.

## What runs

```
tool-quality probes   rank the weak implementations
        ↓
improvement loop      patch one tool -> probe + 240p K -> keep/revert
        ↓
oracle                matched-rate K, ceilings, 88.8 / 90.2 gates
```

The loop is an **agent**, not a deterministic script: it has to write and revert
code. The oracle and probes are deterministic and can be run by hand.

## Prerequisites

- A 4:2:0 reference: `testdata/nasa_480x240_30fps_60f.y4m`.
- `build-release/av2enc` (or another build).
- The AVM decoder, gitignored: `tools/bin/avmdec`, or `AVMDEC=/path/to/avmdec`.
- `butteraugli_main` and `ssimulacra2` on `PATH` for the secondary metrics.
- 4:2:0 only. Do not mix 4:2:0 and 4:4:4 in one ladder; the tools reject it.

## Run the deterministic oracle

```sh
python3 tools/oracle.py \
  --ref testdata/nasa_480x240_30fps_60f.y4m \
  --ivf qp106=/tmp/arm-q106.ivf \
  --ivf qp108=/tmp/arm-q108.ivf \
  --target-kbps 100 --json /tmp/arm-oracle.json
```

It reports matched-rate K, source/segment ceilings, and the 88.8/90.2 gaps.

## Run the architecture queue

```sh
python3 tools/oracle_search.py \
  --registry docs/oracle-candidates.json \
  --bin build-release/av2enc \
  --ref testdata/nasa_480x240_30fps_60f.y4m \
  --target-kbps 100
```

Use `--only NAME` to run one candidate, `--dry-run` to inspect the plan, and
`--status` to select registry states. The registry is the queue; a candidate is
a code source plus flags, not a bare flag combination.

## Run the tool-improvement loop

Start a new OpenCode agent with the prompt below. It runs 10 attempts without
asking, creates one branch/worktree per iteration, and keeps only patches that
pass the probe/K gate.

```text
Work in <worktree> (branch exp/tool-loop, from main). Do not modify the main
checkout or other worktrees.

READ: docs/luma-audit.md, docs/negative-results.md, docs/tool-matrix.md,
docs/oracle-420.md, docs/target-95.md, docs/tool-quality.md if present.

Run 10 iterations; do not stop after the first or after a win.
For each iteration:
1. Pick the weakest unimplemented tool family (ME/subpel/skip, then
   residual/TX/entropy, then filters/allocation, then eff-perc realization).
2. Create a worktree/branch from main. Baseline: a tool-isolating probe plus
   240p QP104/106/108 K.
3. Implement ONE contained, default-off env-gated improvement. No new syntax,
   no preset/pin change.
4. Measure the probe and matched-rate K with the AVM oracle; run pins and
   tools/test.sh.
5. Keep only if the gain is sign-consistent and >= +0.15 K, or a clear probe
   gap closes; otherwise commit the diagnostic as negative and revert.
6. Update docs/tool-improvement-queue.md: target, patch, probe/K, verdict.
Stop only after 10 attempts or when the queue is exhausted/blocked.

Constraints: no third-party code or source copying (AVM/SVT are black-box
binaries only); 4:2:0 only; gate-off pins byte-identical; warning-free build;
tests 10/10; no push, no merge; no history rewrite.
```

## Inspect progress

```sh
cd <tool-loop-worktree>
git log --oneline
cat docs/tool-improvement-queue.md
```

A winning patch stays on its branch for review. It never becomes a default or a
pin change automatically. A negative result is committed as evidence and the
code is reverted.

## First run (2026-11-18)

The first 10-attempt run finished with **no kept patch**. Best arm CDEF strength
4 was +0.065 K but sign-inconsistent; the rest were negative. Ledger:
`docs/tool-improvement-queue.md`.

The run targeted shallow flag/patch changes. The per-tool probes
(`docs/tool-quality.md`) then identified the actual gaps:

1. **Key/intra coefficient coding** — similar key bytes to AVM (22,995 vs
   23,500 B) but 8.4 dB behind on a static detail key. Coefficient
   quantisation/entropy, not TX size. Do not repeat the intra-trellis patch
   that lost -0.76 K.
2. **Leaf-RD skip pricing** — nightly skips 76-91% of inter leaves on the
   fractional-MV probes; disabling `--video-leaf-rd` gains +4.1 K there because
   it codes residual instead of skipping. Test on natural 240p, then make the
   skip decision rate-aware rather than disabling the tool.
3. **BAWP rate-aware acceptance** — BAWP selection is SSE-only; disabling it
   raises K on a fade. Price its rate, not just its SSE.

The next loop must start from these targets; a tool-toggle sweep is not the
work.

## What this loop is not

- It is not a new-tool or new-codec program.
- It does not tune below the wall (SVT 88.8, AVM 90.2).
- It does not copy AVM/SVT source; black-box binaries are the only external
  reference.
- It cannot invent an implementation without an agent writing the patch.
