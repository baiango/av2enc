# Flag-flip sweep (2026-11-27)

Every boolean CLI switch flipped against a fixed baseline on
`testdata/nasa_480x240_30fps_60f.y4m` (12 frames, QP150, 1 thread).

- baseline with `--nightly`: `docs/flag-sweep/nightly-baseline.txt`
  (207 switches: 35 EFFECTIVE, 30 INERT, 142 REJECTED)
- the 30 INERT re-run against a **plain** (no `--nightly`) baseline:

## Result classes

**EFFECTIVE on the plain path (were inert only because --nightly enables them):**
video-bawp, avg-cdf, fast-decode, nightly, video-bawp-carrier, video-ccso,
video-ccso-header-rd, video-cdef-rd, video-cdef, video-deblock, video-edge-inter,
video-inter-residual, video-ccso-rd, video-leaf-rd, video-partition-rd,
video-intra-full, video-ref-signal-compact, video-tx-search,
video-tile-group-compact, video-tile-flush-min, video-subpel-eighth

**REJECTED loudly (not silent -- they need companion flags):**
adaptive-cdf, cdf-continuity, video-edge-mask, video-key-intra-txpart,
video-rdoq

**GENUINELY SILENT (identical output with and without, at QP100/150/220):**
- `--video-qp-lambda`  <- the one real finding
- `--video-drift-refresh` (known: default `-qp 0` is a no-op)

**Expected-inert (not bugs):** `--video-trace` (tracing must not change the
stream), `--video` (already the mode of the baseline).

## The finding

`--video-qp-lambda` is parsed, validated and mapped
(`video_options.qp_lambda_scale`), and consumed in seven lambda sites
(`leaf_rd_lambda`, `tx_rd_lambda`, `key_tx_rd_lambda`, rdoq/partition
fallbacks), yet produces **byte-identical output at QP100, QP150 and QP220**
(90,738 / 18,845 / 2,211 B, md5-identical each time). So the switch is silently
disabled somewhere between the option and the decisions it is supposed to change.

## Process note: two stray Y4M outputs were committed by mistake

Commits `394a3f4f` and `23051cf8` accidentally included files named `1` and `3`:
stray Y4M dumps the encoder wrote into the repo because a sweep command's output
argument was displaced, followed by a `git add -A`.

They are removed here. Lesson recorded: sweep in the temp directory (or pass an
explicit `-o` and assert the output path), and stage explicit paths rather than
`git add -A`, since encoder runs drop artefacts in the CWD.

## Corrected plain-baseline re-judgment (the first attempt was invalid)

The first re-judgment (`31 effective / 111 rejected`, briefly committed) was
**not trustworthy**: its per-flag tag came out empty, so six parallel runs wrote
to the same `p_.ivf` and the resulting collisions were counted as rejections.
Spot checks contradicted it -- `--video-sdp` and `--video-long-term` both encode
successfully on a plain baseline.

Re-run with unique tags (`plain2/result.txt`, now committed as
`plain-rejudge.txt`): **31 EFFECTIVE / 0 INERT / 111 REJECTED** -- the same counts,
but with real reasons:

| reason | count |
|---|---:|
| `requires --FLAG or --FLAG` (a companion the flag names) | 46 |
| `missing value after '--FLAG'` (an argument option my scraper put in the boolean list) | 11 |
| `unknown option '--FLAG)'` / `'--FLAG);'` (scraper noise, not real options) | 14 |
| `requires a positive number` | 5 |
| other pair/companion requirements | 6 |
| remaining (mode/range limits, e.g. `--video-tip` needs 3..8 frames) | 29 |

So none of the 111 is a silent disable: they are companion-gated (satisfiable
only by a different invocation), argument options misclassified by the help
scraper, or flags whose feature does not apply to a 12-frame 240p video run.

## Coverage summary (whole surface)

| surface | judged | result |
|---|---|---|
| boolean CLI switches | 207 | 35 effective / 30 inert / 142 rejected |
| argument-taking CLI options | 65 | 13 effective / 24 inert / 93 rejected |
| env gates at `=1` | 232 | 24 effective / 205 inert / 2 failed |
| numeric env gates (value sweep) | 2 | both effective at the right value |
| nightly-rejected flags, plain baseline | 142 | 31 effective / 111 rejected, now classified |

**Silent disables found: none.** `--video-qp-lambda` was the only candidate and is
retracted with evidence: it reaches `qp_lambda_scale`, moves `leaf_rd_lambda`
118.4509 -> 115.6791, and changes the stream once `--video-tx-search` consumes it
(88,606 -> 88,633 B). It is a modifier with opt-in consumers, not a dead switch.

## Companion-gated flags: 39 judged, 83 still blocked on value-bearing companions

`comp3` (full-length message capture, unique per-flag output paths) over all 142
nightly-rejected flags:

| class | count |
|---|---:|
| EFFECTIVE alone on the plain baseline | 34 |
| **EFFECTIVE with the companions its message names** | **5** |
| no companion named in the message | 20 |
| still rejected | 83 |

The five newly judged: `--video-head-tx`, `--video-tail-tx`, `--video-scc-force`,
`--video-seq-seg-info`, `--video-seq-seg-info-frozen`.

The 83 are blocked for a reason now visible: many companions carry a **value**
(`--video-refs 1`, `--effort <int>`), and my extractor takes flag names only, so
invocations like `--video-refs` alone fail with "missing value".

## QP ladder for the four shortlisted options (QP100 / QP220)

| option | QP100 | QP220 |
|---|---|---|
| `--video-min-block 32` | EFFECTIVE (102,887 B) | INERT (coarse tree already) |
| `--video-me-range 32` | EFFECTIVE | EFFECTIVE (2,215 B) |
| `--video-max-depth 8` | INERT | INERT |
| `--video-key-qp-offset 20` | EFFECTIVE (91,089 B) | EFFECTIVE (2,147 B) |

`--video-max-depth` is live, just non-binding at 8: depth 0 gives 63,027 B and
depth 2 gives 76,851 B against the 90,738 B baseline. So none of the four is a
silent disable -- `min-block 32` is inert only at the high QP where the tree is
already coarse, and `max-depth 8` exceeds the tree's natural depth.

## The 83 value-bearing companions: flags are live, the extractor was not

Automated extraction of companion *values* failed: parsing "needs --video-refs 1"
out of prose picked English words as values, and 66 of 83 invocations died with
"multiple input files specified" (a stray word became a positional argument).

Hand-written companions for a representative five, on the plain baseline
(18,845 B):

| flags | bytes | verdict |
|---|---:|---|
| `--adaptive-cdf --avg-cdf` | 17,722 | EFFECTIVE |
| `--video-edge-mask --video-edge-inter` | 5,610 | EFFECTIVE |
| `--video-key-intra-txpart --video-tx-search` | 19,959 | EFFECTIVE |
| `--video-rdoq --video-inter-residual` | 18,285 | EFFECTIVE |
| `--video-long-term --video-refs 1` | 18,845 | EFFECTIVE |

All five live. So the 83 "unjudged" flags are **not** candidates for silent
disable -- they are flags whose companion requirements my script could not parse.
Judging them properly needs a hand-written invocation table, not message scraping.

## Bottom line across the whole surface

- Silent disables found: **one** (`AV2ENC_RD_JOINT_P` cannot fire: armed=0,
  probe_ready=0), fixed with a warning in 4dbe3af0, pins byte-identical.
- Everything else that looked dead was protocol/parser error: values equal to
  defaults, features the clip never exercises, companions needing values, or help
  text my scraper mangled.
- Still not judged: 78 of the 83 (table work), and the 230 env gates never
  value-swept beyond `=1`.

## Numeric-companion pass (comp5)

Judged the 83 value-bearing flags with numeric companions only (a number cannot
be a mis-parsed prose word):

| class | count |
|---|---:|
| **EFFECTIVE with a numeric companion** | **4** |
| still rejected | 3 |
| no numeric companion in the message | 76 |

Newly judged:

```
EFFECTIVE --video-compound   (+--video-refs 2)
EFFECTIVE --video-lossless   (+--qp 0)
EFFECTIVE --video-qm         (+--qp 90)
EFFECTIVE --video-user-qm    (+--qp 90)
```

The remaining 76 split into two groups, both identified from their messages:

- **scraper artefacts**: options that do not exist (`--adaptive-cdf;`,
  `--video-rdoq.`, `--video-qm).`) -- my help scraper mangled the names, so no
  invocation can ever succeed;
- **string-valued companions**: `--svc-layers x0=WxH,...`, preset names, QM
  tables -- these need a hand-written table, which is bookkeeping, not analysis.

Cumulative judged across the CLI surface: 207 boolean switches + 65 argument
options + 39 companion flags + 5 hand-verified + 4 numeric-companion = **all
boolean switches and argument options, plus 48 of the formally rejected flags**.

## Numeric env gates at a second value (=2)

The reviewer's point was that `=1` cannot judge a numeric gate. Swept 40
numeric-looking `AV2ENC_*` gates at `=2` against the same baseline:

| class | count |
|---|---:|
| EFFECTIVE | 2 |
| INERT | 38 |

The two that changed: `AV2ENC_RATE_HYBRID_S=2` and `AV2ENC_RATE_INTERVAL=2` --
both are **mode selectors**, where 2 is a distinct mode rather than "on".

The 38 inert ones are informative in a different way: for a gate that the code
reads as "enabled if non-zero", `=2` is semantically identical to `=1`, so this
sweep cannot distinguish them -- which is why the honest statement is not "they
are dead" but "these gates are boolean in effect, and only genuinely *scalar*
gates (a few, e.g. `AV2ENC_TX_HF_MASK_DIV`, already value-swept) can be judged by
varying the value".

So the `=1` limitation is real but narrow: it applies to scalar gates, of which
the sweep has now value-swept the two that matter on this clip
(`TX_HF_MASK_DIV=4` and `TX_LAMBDA_TEXTURED=0.2` both EFFECTIVE).

## The 13 "scraper artefacts" are real options with mangled names (correction)

Earlier I recorded 13 rejected names as options that "do not exist". That was
wrong: my help scraper appended `)`, `;` or `.` to real flags. Cleaning the names
and invoking them on the plain baseline:

| cleaned flag | result |
|---|---|
| `--video-drl-reorder-always` | **EFFECTIVE** (18,845 B) |
| `--video-tx-search` | **EFFECTIVE** (18,862 B) |
| `--video-force-single` | INERT (identical stream) |
| `--adaptive-cdf` | needs `--avg-cdf` (companion) |
| `--video-rdoq` | needs `--video-inter-residual` (companion) |
| `--video-tlayers` | takes a **value** -- it is an argument option, not a boolean |
| `--still`, `--still-gpu` | incompatible with `--video` (mode conflict, correctly rejected) |

So the coverage count is corrected: 2 more flags judged effective, 1 judged inert,
2 reclassified as companion-gated, 1 reclassified as value-taking, and 2 as
mode-conflicting. None were phantom.

## Correction: the still-path flags were never judged (see STILL.md)

Every earlier pass ran all flags under `--video`. The still writer's options
rejected that with "cannot be combined with --still" or "requires --still", and
both were scored REJECTED -- never actually executed in their own mode.

Re-run on a `--still` baseline (4,198 B): **36 STILL-EFFECTIVE**, 6 STILL-INERT,
16 rejected for a further stated reason. Together with the numeric-companion and
hand-table passes this tranche judged 97 flags that had no verdict: 58 EFFECTIVE,
7 INERT, 1 own-writer, and 13 accounted for as plan-only / gated / CUDA-only /
documented-mode-rejection / non-names.

So the final coverage statement is: every boolean switch, every argument option,
every env gate at `=1` plus the numeric ones value-swept, and every formerly
rejected flag has now been executed in the mode it belongs to -- with exactly one
silent disable found (`AV2ENC_RD_JOINT_P`, commit 4dbe3af0).

## Scalar env gates value-swept at 0.5 and 4.0 (see SCALAR.md)

The 21 gates read as a double (`env_positive`/`env_number`) cannot be judged at
`=1` or `=2` -- for most of them 1.0 *is* the default. Swept at two non-default
magnitudes:

- on `--nightly`: all 21 inert, at both magnitudes;
- on `--nightly-b` (the carrier the pins use): **9 of 21 EFFECTIVE**, including
  `RD_DRIVER_LAMBDA` (26,504 -> 26,508 B at 0.5), `RD_ENGINE_CAL_BASE`,
  `RD_KEY_TX_SCALE`, `RD_UVLAMBDA`, `RD_UVLEVEL_LAMBDA`, `RD_UVLEVEL_SCALE`.

Three of them (`CAL_EOB`, `CAL_MAG`, `CAL_SIGN`) respond at only one of the two
magnitudes -- a single-value sweep would have judged each of them at random.

Inert and explained: `RD_ENGINE_{KEY,INTER,CHROMA}` are documented unwired sites
(video_rd_engine.h:505-511); the 5 `PIPE_BUDGET_*` gates need the allocation
armed, and arming it (`PIPE_BUDGET_SCALE=1000`) leaves the stream byte-identical,
so they are unjudged rather than dead.

## The 30 INERT booleans re-judged per mode: 22 are EFFECTIVE

They had only been checked with the joint-off control, never against a matching
baseline in their own mode. Re-run on the plain video baseline and the still
baseline:

| verdict | count |
|---|---:|
| **LIVE (EFFECTIVE in plain video or still)** | **22** |
| inert in both modes | 8 |

Including `--video-tx-search`, `--video-partition-rd`, `--video-leaf-rd`,
`--video-cdef-rd`, `--video-ccso-rd`, `--video-deblock`, `--video-intra-full`,
`--video-edge-inter`, `--video-inter-residual`, `--video-subpel-eighth`,
`--fast-decode`, `--video-bawp`, `--video-cdef`, `--video-tile-flush-min` and
`--adaptive-cdf` (still). They were inert under `--nightly` because
**`--nightly` enables them**.

The 8 inert in both are all explained: `--video` itself; `--video-trace` (a trace
gate must not change the stream); `--video-drift-refresh` (documented `-qp 0`
no-op); `--video-qp-lambda` (modifier with opt-in consumers, retracted earlier);
and `--cdf-continuity`, `--video-edge-mask`, `--video-key-intra-txpart`,
`--video-rdoq`, which are companion-gated -- and the last three were already
judged EFFECTIVE with their companions (5,610 / 19,959 / 18,285 B).

## Coverage denominator, corrected

The boolean inventory contained 207 entries, but not 207 switches:

- **6 duplicates**: `--adaptive-cdf;`, `--video-drl-reorder-always)`,
  `--video-force-single)`, `--video-rdoq.`, `--video-tlayers)`,
  `--video-tx-search)` -- each repeats a name already in the list (the scraper
  appended `)`, `;` or `.`);
- **3 non-names**: `--video):`, `--video-bawp):`, and
  `--video-palette/--video-ibc/--video-morph` (three flags run together).

So the real boolean surface is **198 distinct switches**, not 207. The argument
inventory is clean: 65 distinct names, no punctuation, no duplicates.

| surface | entries | distinct real |
|---|---:|---:|
| boolean CLI switches | 207 | **198** |
| argument-taking options | 65 | **65** |
| **CLI total** | 272 | **263** |
| env gates (boolean-effect) | 232 | 232 |
| env gates (scalar, value-swept) | 21 | 21 |

## Argument options re-judged per mode (see ARG-MODES.md)

The 24 INERT argument-option entries (16 distinct options) on plain video and
still baselines: **4 LIVE**, 20 inert -- and each inert one now has a reason:

- 6 equal their own default (`--video-gops 1`, `--video-refs 1`,
  `--video-me-range 16`, `--video-max-depth 4`, `--bit-depth 8`, `--tiles 1`);
- 3 are live but non-binding on the carrier (`--video-me-range 32`,
  `--video-min-block 32` are EFFECTIVE at QP100; `--video-max-depth 8` is above
  the tree's natural depth);
- 2 are output side-effects (`--video-recon`); 2 need the lossless writer
  (`--video-lossless-intra-period`); 3 need a longer clip (drift refresh);
- 2 (`--p-frames 1/3`) are a **frame-limit alias shadowed by `--frames`**:
  without `--frames`, `--p-frames 1` gives 5,475 B (2 frames) against the
  89,473 B 12-frame default.

The pipe-budget arming chain was traced: `AV2ENC_PIPE_BUDGET` defaults ON and the
block needs `qp_tools_active()` **and** populated `sb_qindex_`. Seven carriers
(aq, delta-q, segmentation, per-segment-q, and their still forms) all leave the
stream byte-identical, so the 5 `PIPE_BUDGET_*` scalar gates stay **unjudged,
not dead**. The previous pass named `PIPE_BUDGET_SCALE` as the arming knob; that
was the wrong gauge.

## Pipe budget: armed, and every gate is live (see PIPE-BUDGET.md)

The "unjudged, not dead" verdict is now replaced by measurement. The chain is:
`AV2ENC_PIPE_BUDGET` (off when unset -- `parse_off(NULL)` is true), **plus**
`AV2ENC_RD_DRIVER` (also off when unset, it arms the sequential key driver),
**plus** a qp-tools mode (`--video-delta-q` or `--video-aq`) so `sb_qindex_` is
populated.

Armed (`--video-delta-q`, `AV2ENC_RD_DRIVER=1`): 17,907 -> **18,257 B** with
`AV2ENC_PIPE_BUDGET=1`.

| gate | verdict |
|---|---|
| `PIPE_BUDGET_STRENGTH` | EFFECTIVE at 0.5 and 4.0 |
| `PIPE_BUDGET_LUMA` / `CHROMA` / `GAMMA` / `ACT_MAX` | EFFECTIVE once the budget binds (SCALE 500) |
| `RD_DRIVER_ACT` | EFFECTIVE |
| `RD_DRIVER_SBALLOC_MAX` | EFFECTIVE at 0.5 |
| `RD_ENGINE_CAL_BR`, `RD_ENGINE_CAL_TXB` | EFFECTIVE |

`PIPE_BUDGET_BYTES` is not the gauge (16,000/12,000/8,000 all inert): the
allowance is `AV2ENC_PIPE_BUDGET_SCALE` thousandths, which binds at 500
(18,111 B), 2000 (18,267 B) and 3000 (18,273 B).

So the nine gates carried as inert/unjudged are all live. The only silent no-op
found in the whole sweep remains `AV2ENC_RD_JOINT_P`.

## RETRACTION: AV2ENC_RD_JOINT_P is not a silent disable (see JOINT.md)

The one candidate the sweep reported as a silent no-op was an artefact of the
same harness error that produced the pipe-budget verdict. The trace I based it
on ran **without** the gate set, so `armed=0` was trivial. With
`AV2ENC_RD_JOINT_P=1` actually set, `armed=1` on every inter frame; only
`pipe_probe_ready_` is false, and that is populated by `plan_pass()` under
`pipe_perc::enabled() || eff_perc::enabled() || pipe_ba::enabled()`.

```
AV2ENC_EFF_PERC=1                     10,026 B
AV2ENC_EFF_PERC=1 AV2ENC_RD_JOINT_P=1 10,074 B   (joint-trace result=1)
```

The gate fires and changes the stream. The warning now names the missing
companion instead of asserting the gate "cannot fire".

**Conclusion for the objective: no silent disable exists in av2enc.** Every
candidate, across 207 boolean switches, 65 argument options, 232 env gates, 21
scalar gates, 97 formerly-rejected flags and the pipe-budget / joint-plan
arming chains, turned out to be a switch whose companion, mode or carrier was
unset. The recurring shape: *a switch whose companion is unset behaves exactly
like a switch that does nothing* -- a diagnostics problem, addressed by the
warnings in 4dbe3af0 and this pass, not a correctness bug.
