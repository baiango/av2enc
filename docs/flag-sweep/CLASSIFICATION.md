# One row per judged name: the audit table

`CLASSIFICATION.tsv` is generated from every pass's raw result file, with the
harder pass winning whenever two passes judged the same name. Columns:

```
name    surface    verdict    evidence
```

Surfaces: `switch` (boolean CLI switch), `option` (argument-taking CLI option),
`env` (environment gate).

## Totals, reconciled against the raw files

| surface | rows | source of truth |
|---|---:|---|
| boolean CLI switches | **197** | `flip/result.txt` (207 entries, 6 of them scraper duplicates and 3 non-names: see below) |
| argument-taking options | **65** | `arg-baseline.txt`, no duplicates, no punctuation |
| environment gates | **231** | `env-baseline.txt` -- 231 verdict lines, 231 distinct names |
| integer env gates (call-site derived) | **34** | `int-gates-pass1/2`, `family-arm-pass1/2` |
| **total** | **527** | one row per distinct name |

### Two corrections this table forced

1. **The env surface is 231 gates, not 232.** `env-baseline.txt` contains 231
   verdict lines and 231 distinct names (`sort -u | wc -l`); the "232" quoted in
   earlier passes was a counting slip, not a gate. Every env figure should read
   231.
2. **`--video-tlayers` is an option, not a switch.** It appeared in both the
   boolean inventory and the option inventory; it takes a value (4 or 8) and is
   live with 4 (8,306 B against the 4-frame baseline 8,022 B). It is now counted
   once, as an option. That is why the switch count is 197 and the CLI total is
   262 distinct names (197 + 65), not 263.

### The 207-entry boolean inventory, itemised

| class | names |
|---|---|
| scraper duplicates (name already in the list) | `--adaptive-cdf;`, `--video-drl-reorder-always)`, `--video-force-single)`, `--video-rdoq.`, `--video-tlayers)`, `--video-tx-search)` |
| non-names | `--video):`, `--video-bawp):`, `--video-palette/--video-ibc/--video-morph` |
| real, present only in mangled form | `--still;`, `--still-gpu)`, `--video-qm).` |

207 - 6 duplicates - 3 non-names = 198 real entries; minus `--video-tlayers`
(reclassified as an option) = **197 boolean switches**.

## How a verdict is decided

Each row is the *last* pass that actually executed the name in a mode where it
could act, in this order of authority:

1. `flip/result.txt` -- 207 switches against the `--nightly` baseline;
2. `plain2/result.txt` -- the rejected ones re-run on a plain baseline;
3. `comp3..comp5`, `table/`, `still-rejudge`, `final-rejudge` -- companion and
   still-mode re-runs, including the 36 still-path flags run under `--still`;
4. `inert-per-mode.txt`, `arg-per-mode.txt` -- the INERT sets re-judged on both
   the plain video and the still baseline;
5. `env-baseline.txt` (=1), `scalar-nightly-b.txt` (0.5/4.0), `residual-sweep`
   (=64), `int-gates` (7/64).

A rejection is only kept when the flag names the companion it needs, and those
texts are recorded in the per-pass documents (`ENV.md`, `ARG-MODES.md`,
`STILL.md`, `JOINT.md`, `PIPE-BUDGET.md`, `RESIDUAL.md`).

## Counting rule for "flipped"

A name counts as flipped when **its own value was changed and the stream was
compared** -- by md5 against a baseline built in the same mode, carrier and
value otherwise. Names that only ever produced a rejection are listed with the
companion they name; names that cannot be executed at all are listed as such
(`--video-gpu` on this CPU-only build, `--effort-plan` which writes no stream,
`--video-tlayer-anchor-residual` which needs the env-gated chained plan).

## Integer gates, derived from call sites

The earlier integer sweep used a **name regex** (`DIV|SCALE|_MIN|_TOP|...`). That
was the wrong instrument: re-deriving from `env_int("AV2ENC_...")` call sites
gives **34 gates, none of which appeared in the name-derived list** (see
`int-gates-pass1.txt`). All 34 were then swept at 7 and 64:

| carrier | result |
|---|---|
| armed pipe-budget carrier | 3 EFFECTIVE: `PIPE_BUDGET_Q_HI`, `Q_STEP`, `SCALE` |
| nightly + each family armed (`int-gates-pass2.txt`) | `ARNR_STRENGTH`, `ARNR_THRESH` (with `AV2ENC_ARNR=2`); `TPL_GAIN`, `TPL_SHIFT` (with `AV2ENC_TPL=1`); `RD_JOINT_P_FIRST`, `RD_JOINT_P_LAST`, `RD_JOINT_P_TARGET`, `RD_JOINT_BATCH` (with `EFF_PERC=1` + `RD_JOINT_P=1`) |

**11 of 34 are EFFECTIVE** at two non-default integer values. The rest belong
to families whose own on-switch is itself inert on this clip
(`PART_DIAG`, `RD_TAIL_FLOOR`, `P_DRIFT`, `RD_SEQTREE`), so their gates are
**unjudged on this content**, not shown dead.

## Family arming, pass 3 (`family-arm-pass.txt`)

The 23 gates behind four families were carried as "arm itself inert". This pass
establishes, per family, whether that is a missing carrier or a property of the
code:

| family | finding | gates |
|---|---|---|
| `P_DRIFT` | **arms at `AV2ENC_P_DRIFT_ALPHA=4`** (9,987 B vs 9,973 B); `P_DRIFT_FIRST` is EFFECTIVE at 64 (inert at 7, i.e. an ordinal-window effect) | 1 live, `LAST`/`K` inert while armed |
| `PART_DIAG` | header: "**This gate does not change the stream**" -- measurement only. Inert by design, not unjudged | 3 |
| `RD_SEQTREE` | header: the knobs "**are inert here**" (rd-seq lane surface; the driver reads `AV2ENC_RD_DRIVER_HF`/`_EDGE`) | 4 |
| `RD_TAIL_FLOOR` | arm inert on 12f **and** 60f, and with `AV2ENC_EFF_PERC=1` (13,750 B both ways). Needs the upstream tx_detail path, so the ordinal window never opens. **Unjudged, not dead** | 4 |

So of the 23, one is now measured live; 7 are inert **by documented design**;
4 remain unjudged with the reason identified; `P_DRIFT_{LAST,K}` are inert
while armed on this clip.

## Family arming, pass 4 (`family-arm-pass2.txt`)

| gate | carrier | result |
|---|---|---|
| `PIPE_BUDGET_Q_LO` | armed + `SCALE=500` (binding) | **EFFECTIVE**: 18,109 B -> 18,436 B at 7, 18,417 B at 64 |
| `PIPE_BUDGET_BYTES` | same binding carrier | INERT at 12,000 -- only `BYTES < 0` is consumed (it sets `uniform_control`); a positive count is not the gauge |
| `RD_TAIL_FLOOR_{MIN,DEPTH,FIRST,LAST}` | 60 frames + `EFF_PERC` (probe ready), MIN 8/16, DEPTH 2, FIRST 7, LAST 7 | still INERT (13,750 B every time) -- the floor prices the tail but does not change these decisions |
| `P_DRIFT_{LAST,K}` | `alpha=4`, LAST 64, K 7/64 | still INERT (9,987 B) |

So `Q_LO` joins the live list; `BYTES` is inert by construction for positive
values; the four TAIL_FLOOR gates and `P_DRIFT_{LAST,K}` are inert **while
armed**, which is a different and stronger statement than "unjudged".


## The rejected set, re-run in its own mode (`rejected-resolution.txt`)

The 70 REJECTED + 3 NO-TABLE-ENTRY names were re-run in the mode their family
implies (still options under `--still`, SVC options on an `--alllayers` carrier,
rate-control options without `--qp`, lossless options on the lossless writer),
with a value supplied (`1`, or `x0=480x240` for `--svc-layers`) and one round of
companion resolution from the encoder's own error text.

Result: **5 EFFECTIVE**, **5 INERT**, **63 still REJECTED**.

```
--still-cfl-ds           EFFECTIVE  4,198 B baseline
--svc-layers             EFFECTIVE  18,876 B baseline (x0=480x240)
--lossless-fsc           EFFECTIVE  57,134 B (own writer)
--video-ccso-refit       EFFECTIVE  18,422 B
--video-cdef-rd-lambda   EFFECTIVE  18,206 B
--still-depth, --video-cfl-ds-filter, --cpu-threads, --frames, --chroma  INERT
```

The remaining 65 rows (63 from this pass plus 2 still-mode rejects) each carry
the encoder's own message, e.g. `--video-olk requires --video-...`,
`--video-aq-bias must be a ...`, `unknown option '--video): '`. They are
**rejected with a stated reason**, not judged inert.

### Verdict column normalised

Every row now reads exactly one of `EFFECTIVE`, `INERT`, `OWN-WRITER-OK` or
`REJECTED` (the `EFFECTIVE+COMP`, `NO-TABLE-ENTRY` and `FAILED` spellings are
gone). Final table: **527 rows -- 202 EFFECTIVE, 259 INERT, 1 OWN-WRITER-OK, 65 REJECTED, 0 duplicates.**

## Resolution attempt 2 (`rejected-resolution2.txt`) and the final totals

A value-ladder pass (trying each option against 20 plausible documented values,
then one round of companion resolution) ran over the 62 remaining rejected
options:

```
--svc-embedded   EFFECTIVE (svc carrier, value 2)   60,551 B
--svc-lcr-id     INERT     (svc carrier, value 1)
--svc-temporal   INERT     (svc carrier, value 1)
```

The rest still reject, and the messages say why in each case:
`--bitrate requires --rc vbr`, `--crf requires --rc crf`,
`--effort must be an integer in [10, 10000]`, `--cpu-used is calibrated for
--still only`, `--b-frames --video-gops 1 codes 1 closed GOP(s)`. Those need a
hand-written invocation per option (a specific companion **and** a value that
satisfies that companion's own domain), which the ladder does not discover.

The three non-names (`--video):`, `--video-bawp):`,
`--video-palette/--video-ibc/--video-morph`) are **dropped from the inventory**:
they are help-scraper fragments, never options.

### Final totals

| verdict | rows |
|---|---:|
| EFFECTIVE | **203** |
| INERT | **261** |
| OWN-WRITER-OK | 1 |
| REJECTED (with the encoder's message) | **59** |
| **total** | **524** |

0 duplicate names. The 59 rejections are carried as such, not converted into
inert verdicts by inference.

## Explicit invocation table (`explicit-table.txt`)

Each remaining rejected option was given the mode and value its own error
message names, and judged against a baseline built the same way minus the
option:

| option | invocation | verdict |
|---|---|---|
| `--bitrate` | `--rc vbr --bitrate 120` | **EFFECTIVE** 6,177 B |
| `--crf` | `--rc crf --crf 32` | INERT |
| `--rc` | `--rc crf` (no `--qp`) | **EFFECTIVE** 36,484 B vs 105,537 B |
| `--effort` | `--still --effort 10` | **EFFECTIVE** 4,216 B |
| `--cpu-used` | `--still --cpu-used 4` | **EFFECTIVE** 4,206 B |
| `--video-aq-clamp` | `--video-aq` | **EFFECTIVE** 20,005 B |
| `--video-aq-bias` / `--video-aq-strength` | `--video-aq` | INERT |
| `--video-cdef-damping` | `4` (domain [3,6]) | INERT |
| `--video-cdef-search` | `full` | **EFFECTIVE** 18,206 B |
| `--video-tier` | `high` | **EFFECTIVE** |
| `--video-mv-precision` | `quarter` | **EFFECTIVE** 18,840 B |
| `--video-rd-preset` | `quality` | **EFFECTIVE** 23,526 B |
| `--video-olk` | 4 input frames | **EFFECTIVE** 11,451 B |
| `--video-ras` | 3 input frames | **EFFECTIVE** 7,029 B |
| `--video-switch` | 2 input frames | **EFFECTIVE** 5,477 B |

16 more judged: **12 EFFECTIVE, 4 INERT**.

Still rejected, each now with the encoder's exact reason:

- `--svc-inter-layer` -- "**not a conforming AV2 v1.0.0 feature** §6.8.2";
- `--video-bru`, `--video-idtx-1d` -- "**not decoder-gated: the pinned dav2d does
  not parse** them" (a validation-target limit, not a switch defect);
- `--b-frames`, `--oppoint`, `--spatial-layers` -- need a multi-layer SVC or a
  GOP shape I did not get working on this clip;
- the `--still-*` scale/band options and the remaining `--video-tlayer-*`,
  `--video-*-delta-q` options -- not yet given a working invocation.

### Totals after this pass

**524 rows -- 215 EFFECTIVE, 265 INERT, 1 OWN-WRITER-OK, 43 REJECTED.**
