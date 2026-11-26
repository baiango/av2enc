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
