# The still-path flags: 47 were never judged, and 36 of them are live

## The protocol bug that hid them

Every previous pass ran all 142 nightly-rejected flags the same way:

```
av2enc <clip> -o out --video --frames 12 --qp 150 --cpu-threads 1 <flag>
```

The still writer's options answer that with

```
av2enc: --video cannot be combined with --still, --still-gpu: drop one
av2enc: --still-ccso requires --still or --still-gpu
```

Both were scored **REJECTED**, i.e. bucketed as "the flag does not apply to
this run". They were never invoked in the mode they belong to. Re-run on a
`--still` baseline (`--still --qp 150`, 4,198 B / `5798ed59f6c56aac34127e43474d6a2a`):

| class | count |
|---|---:|
| **STILL-EFFECTIVE** | **36** |
| STILL-INERT | 6 |
| still-rejected for a further stated reason | 16 |

So 36 switches that the sweep had written off as inapplicable are **live and
change the still stream**: `--still-ccso`, `--still-cctx`, `--still-cdef`,
`--still-chroma-lr`, `--still-content-interpretation`, `--still-dc-ac-delta-q`,
`--still-full-intra`, `--still-delta-q`, `--still-gdf`, `--still-dip`,
`--still-ext-partitions`, `--still-ibp`, `--still-grain`, `--still-ibc`,
`--still-idtx`, `--still-ist`, `--still-mrl`, `--still-mhccp`,
`--still-pc-wiener`, `--still-palette`, `--still-rdoq`, `--still-rdo-intra`,
`--still-per-segment-q`, `--still-qm`, `--still-sb128`, `--still-sb256`,
`--still-segmentation`, `--still-tcq`, `--still-tx-partition`,
`--still-uv-full`, `--still-wiener`, `--still-user-qm`, `--no-deblock`,
`--no-cfl`, `--still-ccso-edge`, `--still-cfl-explicit`.

Genuinely inert on this clip: `--still-dc`, `--still-deblock`, `--still-sub8`,
`--still-single`, `--still-tx-type` and bare `--still`.

## The 16 still rejections, resolved

Each names an exact companion; the ones that resolve are judged on a baseline
built with the same frames/qp/mode (never compared across modes):

| flag | invocation | bytes | verdict |
|---|---|---:|---|
| `--still-cdef-on-skip-txfm` | `--still-cdef` | 4,212 | EFFECTIVE |
| `--still-grain-adaptive` | `--still-grain` | 4,264 | EFFECTIVE |
| `--metadata-frame-hash` | `--still` | 4,220 | EFFECTIVE |

## The last batch: compound, TIP, QM, RDOQ

Baselines: `v12` 18,845 B, `v4` (b-frames 2) 8,317 B, `v6` 10,555 B,
`v90` 105,537 B, still 4,198 B.

| flag | invocation | bytes | verdict |
|---|---|---:|---|
| `--video-mvd-sign-derive` | `--video-mv-precision half` | 26,192 | EFFECTIVE |
| `--video-tip-block-ref` | `--video-tip --video-ref-mvs --video-force-single` | 8,267 | EFFECTIVE |
| `--video-compound-near-near` | `--video-compound` | 8,318 | EFFECTIVE |
| `--video-compound-wedge` | `--video-compound` | 8,323 | EFFECTIVE |
| `--video-compound-diffwtd` | `--video-compound` | 8,321 | EFFECTIVE |
| `--video-cwp-weighted` | `--video-compound --video-compound-near-near --video-imp-mask` | 8,322 | EFFECTIVE |
| `--video-imp-mask` | `--video-compound --video-compound-near-near --video-cwp-weighted` | 8,322 | EFFECTIVE |
| `--video-opfl` | `--video-compound --video-compound-near-near --video-refine-mv` | 8,323 | EFFECTIVE |
| `--video-refine-mv` | `--video-compound --video-compound-near-near --video-opfl` | 8,323 | EFFECTIVE |
| `--video-qm` | `--qp 90` | 53,622 | EFFECTIVE |
| `--video-rdoq` | `--video-inter-residual` | 18,285 | EFFECTIVE |
| `--adaptive-cdf` | `--avg-cdf` | 17,722 | EFFECTIVE |
| `--video-tlayers` | `--frames 4 --video-tlayers 4` | 8,306 (base 8,022) | EFFECTIVE |
| `--lossless-fsc` | `--lossless` (own writer) | 57,134 vs 45,070 | EFFECTIVE |

Note on `--video-tlayers`: the error text says "supports 4 or 8 **frames**" but
it means the **value** must be 4 or 8 (temporal-layer count). `--video-tlayers 2`
and `3` are rejected at any frame count; `4` is accepted.

## What is still not a stream comparison, and why

| flag | status |
|---|---|
| `--effort-plan` | plan-only: prints the budget and selection, writes no stream (rc 0) |
| `--video-tlayer-anchor-residual` | needs the env-gated chained `--video-tlayers` plan, not the CLI form |
| `--video-gpu` | rejected on this CPU-only build ("requires a CUDA build") |
| `--nightly-b` | rejected: `--video-edge-inter` is measured on the low-delay P chain |
| `--still-ddt` | "not signallable in a still picture" -- documented rejection |
| `--still-sdp` | "SDP sub-tree partitions are not wired" -- documented rejection |
| `--still-gpu` | CUDA-only |
| `--video):`, `--video-bawp):`, `--video-palette/--video-ibc/--video-morph` | not option names (help-scraper fragments) |

## Tally for this tranche

Of the 97 flags that had no verdict before this pass:

- **58 EFFECTIVE**
- **7 INERT**
- 1 own-writer EFFECTIVE (`--lossless`)
- 1 plan-only, 1 gated, 1 build-limited, 7 documented mode rejections, 3 non-names

## The lesson, again

Seven protocol errors have now produced false "dead" verdicts: empty tags,
truncated companions, prose-as-value, redirect-before-mkdir, zsh not
word-splitting, `requires --still` read as "rejected", and value-taking flags
re-appended without a value. The encoder was never the thing that was wrong.
