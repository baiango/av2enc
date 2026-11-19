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

## Scripted runner

The script owns the iteration count. It calls the OpenCode agent once per
iteration and never stops after the first one:

```sh
python3 tools/agent_loop.py \
  --iterations 20 \
  --worktree ../agent-loop \
  --branch exp/agent-loop
```

It finds the OpenCode CLI automatically (`OPENCODE_CLI` overrides it), creates
the worktree from `main` if missing, symlinks `testdata`/`tools/bin`, and logs
each run under `build/agent-loop/`. Each call performs exactly one iteration;
the ledger is the memory between calls.

Useful flags: `--model`, `--agent`, `--prompt-file`, `--ledger`,
`--max-hours`, `--stop-file`, `--sleep`, `--fail-fast`, `--dry-run`.

For a continuous run, drop the caps:

```sh
python3 tools/agent_loop.py \
  --iterations 0 \
  --max-hours 0 \
  --max-stale 5 \
  --iteration-timeout 3600 \
  --worktree ../agent-loop \
  --branch exp/agent-loop \
  --sleep 2
```

`--iterations 0` means until stopped; `--max-hours 0` means no wall-clock cap;
`--max-stale 5` stops only after five consecutive agent calls leave the ledger
unchanged; `--iteration-timeout` kills a stuck call and continues. The five-lane
default is 10,800 s (3 h), not 1 h: under five concurrent lanes a single
build/measure iteration can take close to an hour. Stop it with
the stop file (`--stop-file`) or Ctrl-C.

## Five parallel lanes

`tools/agent_loop_multi.sh` starts five isolated lanes. Each lane has its own
worktree, branch, ledger and fixed target, so code edits cannot collide:

```sh
tools/agent_loop_multi.sh
# optional: AV2ENC_LANES=5 AV2ENC_ITERATIONS=10 AV2ENC_MAX_STALE=5 AV2ENC_TIMEOUT=10800
```

Targets: P-head residual, P-tail residual/allocation, inter transform/quant,
inter entropy/context, and head prediction/ME. Logs are
`build/agent-loop/lane-N.log`; each lane writes
`docs/agent-loop-ledger-N.md`. Stop all lanes with the shared stop file
(`../agent-loop/STOP`). A kept patch stays on its lane branch for review.

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

## Second run (2026-11-18)

Ten attempts from the per-tool benchmark, branch `exp/tool-loop2`, ledger
`docs/tool-improvement-queue-2.md`. Two default-off gates were kept:

* `AV2ENC_SKIP_DRIFT` (+`AV2ENC_SKIP_DRIFT_FADE`) — future-error surcharge on
  skips, non-zero-MV always and zero-MV when the leaf shows a DC fade bias.
  `me_half` 76.105 (was 71.615), `fade` 84.128 (82.701), natural 240p 83.473
  (82.149); `static_repeat` and `near_static` byte-identical.
* `AV2ENC_DECISION_EXACT_RATE` — the leaf decisions charge the TX plan's own
  §8.3 rate instead of the historical heuristic. Natural 240p 83.112, `me_half`
  73.012; composes with the skip gate to 83.790 on natural.

The intra/key trellis (even with an SSE guard), rate-aware BAWP acceptance and
the eff-perc stack all measured negative and were reverted. The key coefficient
gap remains open: the trellis moves the key RD curve by tenths of a dB, not the
8 dB the probe reports.



## Third run: key/intra closed (2026-11-19)

The key/intra lane ran 20 attempts and kept nothing. Diagnosis: the 8.4 dB key
gap is not prediction (<=2-3 dB headroom, mostly outside syntax), not band
shape (error is 6.7-10.3x AVM in every DCT ring, i.e. a ~2.6x coarser quantiser
step), and not the entropy coder (the key field at matched quality is larger
than AVM's entire key). The largest measured defect, a 5.2x coefficient-rate
underprice in the intra TX search, is worth <=0.35 dB and fails on natural
content. A perfect key is worth +0.00 K, so key work is not a K lever. Do not
target key/intra unless it is a dependent-quantiser research project. See
`docs/tool-improvement-queue-3.md`.


## Fourth run: eff-perc/filter decisions closed (2026-11-19)

21 attempts, 0 kept. The eff-perc 90.12 ceiling is the f0-32 source row
(90.197), not the key: key +0.125, f0-32 +5.017, f33-43 +2.119, f44-59
+3.815. The head P frames are 4.9 dB behind at ~45 B/frame; the tail is 69%
coded residual and rate-bound. Filter/allocation's 88.96 ceiling is f44-59,
but 91% of nightly's tail error is uncoded skip error; filters and refresh
placement cannot remove prediction error. Decision/refresh/rate queues are
exhausted for both mechanisms. Remaining: per-frame/head P prediction and
residual coding efficiency. See `docs/realization-gaps.md`.


## Oracle check rule (2026-11-19)

The kept P-tail stack measures **84.508 matched K**, with segment ceilings
**89.603** (f0-32) and **88.654** (f44-59); the old nightly ceilings were
85.855 and 88.066, the old eff-perc ceilings 90.197 and 88.995. The source
replacement bound moved, but the matched-rate point did not; `tools/oracle.py`
returns NO-GO (<88.8). This is a **realization gain**, not a higher oracle.
See `docs/agent-loop-oracle-check.md`.

No patch is kept on matched K alone. It must report a matched-rate envelope and
its segment ceilings; a segment ceiling above the wall is not a stream.


## Mechanical envelope gate (2026-11-19)

The loop no longer accepts a keep/revert judgement written in prose. After every
agent call the script runs `tools/envelope.py` and the **script** decides:

```
agent edits src/ + writes build/agent-loop/spec.json
        ↓
tools/envelope.py   auto-brackets each arm around 100 kbps
        ↓
tools/oracle.py     matched K per arm (AVM decoder)
        ↓
envelope = max K over arms; baseline = the "base" arm
        ↓
delta >= 0.15 K ? keep src/ : revert src/
```

The spec is one JSON object:

```json
{"ref": "testdata/nasa_480x240_30fps_60f.y4m",
 "bin": "build-release/av2enc", "frames": 60,
 "base_args": ["--video","--frames","60","--cpu-threads","8","--nightly"],
 "qp": 106, "target_kbps": 100.0,
 "arms": [{"name": "base", "env": {}},
          {"name": "cand", "env": {"AV2ENC_TOOL": "1"}}]}
```

Rules the script enforces, not the prompt:

- `base` must exist and have an empty env, so the baseline is measured on the
  same build and a default-on patch cannot hide in a second tree.
- The runner supplies the lane's **established recipe** with `--base-env`
  (lane 2: the kept skip/exact-rate stack). That recipe is the baseline and is
  applied to every arm. It must never be an arm of its own: re-enabling the
  committed stack is not a new tool, and without this rule the first batch
  reported the stack as a +2.614 K "envelope gain".
- Every knob value of the candidate is an arm. The envelope is the best arm;
  one winning arm under an unchanged envelope is not progress.
- QP is bracketed automatically (`qp` is only a start) and a candidate that
  cannot bracket is reported, not scored.
- On FAIL the runner reverts `src/` (`git checkout -- src` + `git clean -fd
  src`) and keeps the ledger. On PASS it leaves the patch in place.
- A missing spec is a FAIL. An iteration that does not produce a spec cannot be
  kept.
- Every result is appended to `build/agent-loop/envelope.jsonl`
  (`base_k`, `envelope_k`, `delta`, `best_arm`, per-arm brackets).
- **The ceiling is part of the gate.** For the best arm the runner also runs
  `tools/oracle.py` with segment replacement and requires the target segment's
  source-replacement ceiling to beat the **base arm's ceiling on the same
  bracket point** by +0.15 K (published reference: f0-32 90.197, f44-59
  88.995 at QP119; comparing a QP120 candidate to those would add ~0.17 of
  measurement bias). A higher matched-K envelope under an unchanged ceiling is
  a realization gain and FAILs, however large the envelope delta.

Run it directly:

```sh
tools/envelope.py --spec build/agent-loop/spec.json --threshold 0.15
```

The five-lane launcher passes `--envelope-gate --envelope-revert` and
`AV2ENC_ENVELOPE_DELTA`, so every lane is judged by the same mechanical gate.
This was smoke-tested end to end: a stub agent wrote a base-only spec and touched
`src/`; the gate measured base 82.149 / envelope 82.149 / delta +0.000, returned
FAIL, and `src/` was reverted while the ledger and `envelope.jsonl` were kept.

## Objective: the oracle ceiling

The loop improves the **oracle**, not a single K point. A tool is progress only
if the best matched-rate K over its knobs (its envelope) rises. A single arm
that gains 0.2 K under the same ceiling is a **realization gain**, not an oracle
gain, and is not kept on that basis. Segment/source ceilings are reported but
are pooling bounds, not streams. See `docs/agent-loop-oracle-check.md`.

## What this loop is not

- It is not a new-tool or new-codec program.
- It does not tune below the wall (SVT 88.8, AVM 90.2).
- It does not copy AVM/SVT source; black-box binaries are the only external
  reference.
- It cannot invent an implementation without an agent writing the patch.
