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
