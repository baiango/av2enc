# Environment-gate sweep (2026-11-27)

Every `AV2ENC_*` (and other) `getenv` gate in the tree, set to `=1`, against a
fixed baseline (`nasa_480x240`, 12 frames, QP150, `--nightly`, 1 thread).
Raw results: `docs/flag-sweep/env-baseline.txt`.

| class | count |
|---|---:|
| EFFECTIVE (stream changed) | **24** |
| INERT (identical md5) | 205 |
| FAILED (encode error) | 2 |

The 24 EFFECTIVE gates are the validity check for the method -- e.g.
`AV2ENC_SEF_DUPLICATES`, `AV2ENC_RATE_HYBRID_S`, `AV2ENC_PART_PROBE_CHROMA`,
`AV2ENC_RD_DRIVER`, `AV2ENC_RD_KEY_TX`, `AV2ENC_SKIP_DRIFT`, `AV2ENC_TPL`,
`AV2ENC_ARNR`, `AV2ENC_STILL_*_CTX`. So the sweep does move the stream when a
gate is live.

## INERT does not mean disabled

Three large groups are inert **by construction**, and counting them as bugs would
be wrong:

1. trace / dump / stats / debug gates (~40): they must not change a stream.
2. features not exercised on this input or target: `AV2ENC_STILL_*` (still path),
   `*GPU*`, `AV2ENC_TIP*`, `*TLAYER*`, `ARNR`, `TPL`, `VRAM`.
3. gates for experimental decision paths that `--nightly` does not run
   (`AV2ENC_RDO_*`, `AV2ENC_RD_ENGINE*`, `AV2ENC_VIDEO_RDO*`, `AV2ENC_COMPOUND_*`).

## Gates proved live elsewhere (inert only in this configuration)

- `AV2ENC_RDOQ_REFINE_RECON` -- inert because the intra RDOQ path was unreachable
  (fixed separately); measured inert again after the fix.
- `AV2ENC_KEY_INTRA_TX_PART` -- gated behind `AV2ENC_RD_KEY_TX`.
- `AV2ENC_SEF_DUPLICATES` -- EFFECTIVE, but only on an inter chain.

## Remaining suspects

`AV2ENC_TX_HF_MASK_DIV` / `AV2ENC_TX_LAMBDA_TEXTURED` show INERT at `=1`, yet they
are EFFECTIVE as CLI/env values in the stack arm measurements (they change stills
measurably). Value `1` is simply not distinguishable from the default for a
divisor, so the `=1` protocol cannot judge numeric knobs -- they need a
value-sweep, not an on/off flip.

## Value sweep for numeric gates (the =1 protocol's blind spot)

Same still/config as the stack arm (`src720_1f`, QP186, `--nightly-b --video-gdf
--video-uv-qp-offset -23`), md5 against the no-variable baseline (47,948 B):

| gate | value | bytes | verdict |
|---|---|---:|---|
| `AV2ENC_TX_HF_MASK_DIV` | 0 | 47,948 | INERT (0 ignored) |
| `AV2ENC_TX_HF_MASK_DIV` | **4** | **46,764** | **EFFECTIVE** |
| `AV2ENC_TX_LAMBDA_TEXTURED` | 0.05 | 47,948 | INERT (= default) |
| `AV2ENC_TX_LAMBDA_TEXTURED` | **0.2** | **47,849** | **EFFECTIVE** |

So these gates are live; the `=1` protocol could not judge them because 1 is
indistinguishable from the default for a divisor and a multiplier. **Numeric
gates require a value sweep** -- their INERT entries in the on/off table are
artefacts of the protocol, not evidence of a silent disable.

Process note: two earlier attempts at this sweep were void -- an unquoted `$C`
in zsh is not word-split, so every flag arrived as one argument and every encode
failed; the empty md5s then compared equal and reported INERT. Always check the
file exists and is non-empty before comparing.

## Corrected verdict on --video-qp-lambda: wired and effective, not a silent disable

Earlier this sweep listed `--video-qp-lambda` as the one genuinely silent switch.
That was wrong, and the trace proves it:

```
--video-qp-lambda off: scale=0 legacy=1 leaf_rd_lambda v=118.450885370
--video-qp-lambda on : scale=1 legacy=0 leaf_rd_lambda v=115.679063074
```

The option reaches `VideoOptions::qp_lambda_scale`, and the fitted lambda differs
by 2.3 %. It changes the stream as soon as a consumer that actually uses the fit
is enabled:

| flags | bytes | md5 relation |
|---|---:|---|
| `--video-tx-search` | 88,606 | baseline |
| `--video-tx-search --video-qp-lambda` | **88,633** | **DIFFER** |
| `--video-leaf-rd` / `+ --video-qp-lambda` | 26,506 | identical |

So it is a **modifier**, not a broken switch: it retunes lambdas that only some
features consume, and on a `--nightly` 240p 12-frame clip at QP150/100 no
decision flipped. No bug to fix.
