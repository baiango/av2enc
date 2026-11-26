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

## Domain parsing from the error text (`resolve3.txt`)

A pass that reads the option's own message and takes the value from the domain
it states (a range `[lo, hi]`, an enumeration, a required companion, a mode)
resolved 19 more: **12 EFFECTIVE, 7 INERT**.

| option | mode | verdict |
|---|---|---|
| `--still-ccso-bands`, `--still-cdef-strengths`, `--still-delta-q-res`, `--still-ibc-search`, `--still-qm-level`, `--still-seg-delta` | `--still` | **EFFECTIVE** |
| `--video-partition-rd-depth` | video | **EFFECTIVE** |
| `--video-delta-q-res` | `--video-delta-q` | **EFFECTIVE** 18,524 B vs 18,574 B |
| `--video-rdoq-scale` | `--video-inter-residual` | **EFFECTIVE** 18,266 B vs 19,583 B |
| `--video-seg-delta` | `--video-per-segment-q` | **EFFECTIVE** 18,667 B vs 17,579 B |
| `--video-uv-delta-q` / `--video-y-dc-delta-q` | video | **EFFECTIVE** 18,874 / 19,890 B |
| `--video-bru`, `--video-idtx-1d`, `--video-segments` | video (bare) | INERT |
| `--still-ddt`, `--still-sdp` | `--still` | INERT |
| `--nightly-b`, `--effort-plan` | own mode | INERT |

`--video-bru` and `--video-idtx-1d` being INERT is informative: the earlier
"not decoder-gated / the pinned dav2d does not parse them" messages are
*warnings* under the pinned validator, not refusals -- the switches execute and
do not change this stream.

### Totals after this pass

**524 rows -- 227 EFFECTIVE, 272 INERT, 1 OWN-WRITER-OK, 24 REJECTED.**

The 24 that remain reject for a stated structural reason, e.g.
`--svc-inter-layer` ("not a conforming AV2 v1.0.0 feature"), the SVC layer
selectors (`spatial layer x1 is smaller than x0` -- my 2-layer geometry is
invalid), `--recon` (needs a 10-bit core), `--vram-cap` ("no effect in this
CPU-only build"), and the `--video-tlayer-*` family (needs the chained
`--video-tlayers` plan).

## SVC geometry, tlayer plan, and the final scope split (`resolve4.txt`)

The 2-layer carrier was inverted: the **larger** layer is `x1`. With
`--svc-layers x0=240x120,x1=480x240` the SVC path runs (31,056 B) and the layer
selectors are judged on it:

| option | verdict |
|---|---|
| `--oppoint 1` | INERT on the SVC carrier |
| `--spatial-layers 2` | INERT |
| `--video-segments` | INERT |
| `--video-uv-delta-q`, `--video-y-dc-delta-q` | INERT here, **EFFECTIVE** on plain video (18,874 / 19,890 B) |

With the tlayer family armed (`--video-tlayers 4`, 8,306 B):

| option | verdict |
|---|---|
| `--video-tlayer-drop 0` | **EFFECTIVE** 5,756 B |
| `--still-ist-scale` (with `--still-ist`) | INERT |
| `--still-rdo-scale` (with `--still-rdo-intra`) | **EFFECTIVE** 4,166 vs 4,176 B |
| `--still-tcq-scale` (with `--still-tcq`) | **EFFECTIVE** 1,018 vs 3,645 B |
| `--still-tx-type-scale` (with `--still-tx-type`) | INERT |
| `--video-num-ref-frames 4` (domain [4,8]) | **EFFECTIVE** 18,856 vs 18,845 B |

### Scope split, stated explicitly

Ten rows are marked **OUT-OF-SCOPE** rather than judged, because the reason is a
property of this environment or of the format, not of the switch:

| row | why |
|---|---|
| `--video-gpu`, `--still-gpu`, `--vram-cap` | CUDA-only; this is a CPU-only build |
| `--gpu-plan` | a host-planning dry run; it cannot be combined with an encode |
| `--recon` | needs a 10-bit core this build does not accept (tried 4:2:2 as well) |
| `--svc-inter-layer` | not a conforming AV2 v1.0.0 feature (§6.8.2) |
| `--video-tlayer-{qp-offsets,anchor-residual,refresh-budget,refresh-threshold}` | need the env-gated chained tlayers plan |

### Final totals

**524 rows -- 231 EFFECTIVE, 276 INERT, 1 OWN-WRITER-OK, 10 OUT-OF-SCOPE, 6 REJECTED.**

## The last six rows (`final6`)

Each was run with the exact shape its message demands, with an existence check
on both the option's output and its baseline before any md5 comparison:

| row | invocation | result |
|---|---|---|
| `--b-frames` | `--frames 4` (the message demands exactly 4), `--b-frames 2` | **EFFECTIVE** 8,317 vs 8,022 B |
| `--video-leading` | `--frames 4 --video-leading 1` (≤9 frames) | **EFFECTIVE** 11,090 B |
| `--still-rdoq-scale` | `--still --still-rdoq --still-rdoq-scale 1` | **EFFECTIVE** 968 vs 3,310 B |
| `AV2ENC_FRAME_QP_PLAN` | `=100:220` (it "expects <head>:<tail>") | **EFFECTIVE** 5,039 vs 18,845 B |
| `AV2ENC_PERCEPTUAL_FILTER` | `=1` | INERT (18,845 B = base) |
| `--video-lossless-b` | `--video --qp 0 --frames 4 --video-tlayers 4` | **still unresolved**: the encoder prints "Video lossless encode complete" but writes no file at the `-o` path, so there is nothing to compare |

Two of these were previously void and are now correct: `--still-rdoq-scale` had
been called EFFECTIVE from a missing file, and this run checks both files exist
before comparing (968 B vs 3,310 B).

### Final totals

**524 rows -- 235 EFFECTIVE, 277 INERT, 1 OWN-WRITER-OK, 10 OUT-OF-SCOPE, 1 REJECTED.**

## The chained tlayers plan located, and the last row (`final7`)

The four `--video-tlayer-*` rows marked OUT-OF-SCOPE were not a dead end: the
gate is `tlayers_chain_enabled()` (`options.h:549`) --

```cpp
if (options.video_tlayers_chain) return true;
const char* value = std::getenv("AV2ENC_TLAYERS_CHAIN");
return value != nullptr && value[0] != '\0' && value[0] != '0';
```

-- and the chained form needs `--video-tlayers != 0` **and** `--video-gops > 1`.
Armed (`AV2ENC_TLAYERS_CHAIN=1 --video-tlayers 4 --video-gops 2`, 13,669 B):

| option | verdict |
|---|---|
| `--video-tlayer-anchor-residual` | INERT (13,669 B = base) |
| `--video-tlayer-refresh-budget 1000` | INERT |
| `--video-tlayer-refresh-threshold 1` | INERT |
| `--video-tlayer-qp-offsets 0,0,0,0` | INERT on `--video-tlayers 8 --video-gops 2` (25,096 B = base) |

`--video-lossless-b`: the `-o` path is honoured (79,344 B written, "4 output
frame(s), base_q_idx 0"). It is a **mode enabler**, not a tune: `--qp 0` without
the flag is refused, so there is no like-for-like baseline to compare against.
Recorded as OWN-WRITER-OK beside `--lossless`.

### Final totals -- no row left without a verdict

**524 rows -- 235 EFFECTIVE, 281 INERT, 2 OWN-WRITER-OK, 6 OUT-OF-SCOPE, 0 REJECTED.**

The six OUT-OF-SCOPE rows are environment limits only: `--video-gpu`,
`--still-gpu`, `--vram-cap` (CUDA-only build), `--gpu-plan` (a dry run that
cannot be combined with an encode), `--recon` (needs a 10-bit core this build
refuses), `--svc-inter-layer` (not a conforming AV2 v1.0.0 feature).

## The six OUT-OF-SCOPE rows, executed (`final8`)

| row | how it was executed | verdict |
|---|---|---|
| `--recon` | `--still --tiles 2x2 --no-deblock --recon out.y4m` -- the multi-tile still path its help names | **INERT**: recon Y4M written (172,845 B), stream 4,210 B = baseline |
| `--vram-cap` | `--vram-cap 1024` on the video carrier | **INERT**: 18,845 B = baseline (the build also prints "no effect in a CPU-only build") |
| `--gpu-plan` | `av2enc <clip> --gpu-plan 1` with no `--video`/`-o` | **OWN-WRITER-OK**: prints a host tile plan ("4x2 (8 tiles), sb 64x64"), writes no stream |
| `--video-gpu` | -- | blocked: "requires a CUDA build; this is a CPU-only reference build" |
| `--still-gpu` | -- | blocked: same message |
| `--svc-inter-layer` | valid 2-layer carrier `x0=240x120,x1=480x240` | refused: "not a conforming AV2 v1.0.0 feature: §6.8.2" |

`--recon` is the notable one: it never needed a 10-bit core at all (the 10-bit
message was about the *input* coloursapce); it needs the **multi-tile still**
path, and there it writes its dump and leaves the stream byte-identical.

### Final totals

**524 rows -- 235 EFFECTIVE, 283 INERT, 3 OWN-WRITER-OK, 3 OUT-OF-SCOPE, 0 REJECTED.**

The three remaining rows are blocked by the build or the format, not by arming:
two CUDA-only switches on a CPU reference build, and one option for a syntax the
AV2 v1.0.0 spec does not define.

## Env gates derived from call sites, and the 454-name reconciliation

The env surface was rebuilt from **read sites in `src`**, not from the help or a
name pattern:

```python
getenv("AV2ENC_...")                       -> 190 names
env_int/env_positive/env_number("AV2ENC_...") ->  88 names
union (real read sites)                    -> 276 names
```

**The 454 figure counted non-gates.** Of the 454 distinct `AV2ENC_*` tokens in
`src`, 178 are not environment variables at all:

| token class | count | example |
|---|---:|---|
| include guards | 96 | `AV2ENC_CLI_OPTIONS_H`, `AV2ENC_CUDA_CFL_H_` |
| syntax/table constants | 65 | `AV2ENC_CDEF_HD`, `AV2ENC_ADD_CDF` |
| names only in comments | 17 | mentions in doc comments |

276 + 178 = 454. **Every one of the 276 real read sites now has a verdict**:
`env-baseline` covered 231, the scalar and integer passes covered 55, and this
pass swept the **34 that had never been run** at 0.5 and 4.0 on two carriers (a
plain nightly baseline and one with EFF_PERC, PIPE_BA, PIPE_PERC, RD_TAIL_FLOOR,
PART_DIAG, P_DRIFT, RD_SEQTREE, TPL and RD_DRIVER all armed):

| gate | verdict |
|---|---|
| `EFF_PERC_CHROMA` | **EFFECTIVE** (armed, 0.5 and 4.0) |
| `PIPE_BA_LAMBDA`, `PIPE_BA_WEIGHT` | **EFFECTIVE** |
| `PIPE_LPL_DEEP`, `PIPE_LPL_REFRESH` | **EFFECTIVE** (even on the plain carrier) |
| `P_DRIFT_ALPHA` | **EFFECTIVE** |
| `RD_DRIVER_EDGE`, `RD_DRIVER_HF` | **EFFECTIVE** |
| the other 26 (incl. `JOINT_TRACE`, `RDOQ_TRACE`, the `*_HF`/`*_EDGE`/`*_LAMBDA` knobs of unarmed families) | INERT at both values on both carriers |

**Eight more live gates**, all previously unwatched — the same pattern as every
other family in this sweep: the read site existed, the coverage list did not.

### Totals

**558 rows -- 243 EFFECTIVE, 309 INERT, 3 OWN-WRITER-OK, 3 OUT-OF-SCOPE.**

## The INERT set re-tested on other carriers (`inert-sample.txt`)

A random sample of 40 of the 309 INERT rows was re-run at 0.5 and 4.0 on two
carrier families the earlier sweeps did not use:

- **v60** -- 60 frames, `--nightly`, `AV2ENC_EFF_PERC=1` (13,750 B);
- **bf**  -- `--frames 4 --b-frames 2`, `AV2ENC_EFF_PERC=1` (8,117 B).

**8 of the 40 flipped to EFFECTIVE:**

```
AV2ENC_DEBLOCK_APPLY      v60 @0.5
AV2ENC_EFF_PERC_CSF       v60 and bf, both values
AV2ENC_EFF_PERC_KEY_CSF   v60, both values
AV2ENC_EFF_PERC_RATE      v60 @0.5
AV2ENC_EFF_PERC_SCALE     v60 and bf, both values
AV2ENC_EFF_PERC_TOP       v60 and bf, both values
AV2ENC_P_DRIFT            v60, both values
AV2ENC_RD_MODE_BITS_DIR   v60 @0.5
```

So a fifth of a random sample of "inert" gates were live on a carrier the sweep
had not tried -- the same failure mode as every family before it. The INERT
label in this table therefore means **"did not change the streams it was given"**,
not "cannot change any stream". The 8 above are corrected to EFFECTIVE.

### Totals

**558 rows -- 251 EFFECTIVE, 301 INERT, 3 OWN-WRITER-OK, 3 OUT-OF-SCOPE.**

## Scope decision for the last three rows

| row | decision |
|---|---|
| `--video-gpu` | **environment-blocked**: "requires a CUDA build; this is a CPU-only reference build". No CUDA toolchain or GPU is present in this workspace, and macOS is explicitly not an encoder target per `AGENTS.md` -- so it cannot be executed here by construction. |
| `--still-gpu` | same message, same reason. |
| `--svc-inter-layer` | **spec-refused**: "--svc-inter-layer is not a conforming AV2 v1.0.0 feature: §6.8.2 requires lcr_dependent_xlayers_flag". No configuration can make it encode. |

The first two need a decision from the operator (build on a CUDA host, or accept
them as out of scope for a CPU reference build); the third is settled by the
format itself.

## Scope decision, signed off by the operator (2026-11-27)

The operator was asked whether to obtain a CUDA host or accept the two
CUDA-only rows as permanently out of scope, and chose: **accept as out of
scope**. The decision is recorded here so the table is closed with an explicit
scope boundary rather than an open question:

- `--video-gpu`, `--still-gpu` -- **out of scope for a CPU reference build**.
  Evidence: "requires a CUDA build; this is a CPU-only reference build". The
  development and production targets are Linux + CUDA (`AGENTS.md`); macOS is
  explicitly not an encoder target, so no carrier exists in this workspace.
- `--svc-inter-layer` -- **refused by the format**, not out of scope: "not a
  conforming AV2 v1.0.0 feature: §6.8.2 requires lcr_dependent_xlayers_flag".

The sweep is closed at this boundary. 558 rows, 3 documented exclusions, and the
`INERT` column stated for what it is: "did not change the streams it was given"
(8 of 40 sampled re-tests moved on a different carrier).

## Every remaining INERT env gate re-run on the two flip carriers (`inert-all-c41.txt`)

After the 40-row sample showed 8 flips, **all 228 remaining INERT `AV2ENC_*`
gates** were re-run (staged, stopping at the first flip):

1. 60 frames, `--nightly`, `AV2ENC_EFF_PERC=1`, value 4.0 (13,750 B);
2. `--frames 4 --b-frames 2`, `AV2ENC_EFF_PERC=1`, value 4.0 (8,117 B);
3. 60 frames, value 0.5.

**14 more were live:**

```
EFF_PERC_CSF2   EFF_PERC_DSHAPE   PORT_HORIZON   RDOQ_ADAPT
RDOQ_CTX_REFINE RDOQ_DIST_WEIGHT  RDOQ_EXACT     RDOQ_TRELLIS_STEP_SCALE
RD_HEAD_TX      RD_JOINT_P        RD_TAIL_TX     RD_UVLEVEL
TX_DETAIL_SEARCH_TAIL             TX_HF_MASK_DIV
```

All fourteen flipped on the **60-frame** carrier at 4.0 -- they were inert on the
12-frame nightly carrier only because the tail pictures and the rate/level
planning they act on never appear in a 12-frame clip.

`AV2ENC_RD_JOINT_P` appears in that list, which is consistent with the earlier
retraction: with `EFF_PERC` armed it moves the stream, and its "inert" row was
an artefact of the carrier, never of the gate.

The remaining 214 gates are inert on **all three** of those runs. So the INERT
column of this table now rests on at least two independent carrier families
(12-frame and 60-frame, plus a b-frame carrier), not on a single sweep.

### Totals

**558 rows -- 265 EFFECTIVE, 287 INERT, 3 OWN-WRITER-OK, 3 OUT-OF-SCOPE.**

## CLI switches re-tested on a 60-frame carrier (`inert-cli-60f.txt`)

The 56 INERT CLI switches/options (the non-env rows) were re-run on the
60-frame carrier, which is the clip shape that uncovered the `RD_TAIL_*` and
`RDOQ_*` gates:

| result | count |
|---|---:|
| **EFFECTIVE** | **2** (`--video-me-range`, `--frames`) |
| INERT | 25 |
| not applicable in this mode (rejected by a stated companion) | 29 |

`--video-me-range` is the substantive one: it was inert on the 12-frame clip and
moves the 60-frame stream. `--frames` changes the clip length by construction,
so its EFFECTIVE verdict says only that the option acts.

### Totals so far

**558 rows -- 267 EFFECTIVE, 285 INERT, 3 OWN-WRITER-OK, 3 OUT-OF-SCOPE.**
