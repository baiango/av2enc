# R19: range-coder interval-width rate model

Branch `exp/rate-interval`. Gate `AV2ENC_RATE_INTERVAL`, default off:

| Value | Model |
|---|---|
| unset / `0` | historical heuristics (pins unchanged) |
| `1` | interval-width estimate (the model under test) |
| `exact` | ablation: the same symbol sequence through a real §8.2 encoder (R05 "full trial coding") |

`AV2ENC_RATE_INTERVAL_STATS=1` enables the per-leaf predicted-vs-actual report
(and the `-log2 p` symbol meter it compares against).  `AV2ENC_RATE_INTERVAL_TRACE=1`
also prints one line per reported leaf.

## What the model does

The leaf decision prices a residual with `estimate_tx_bits`
(`8 + 2.5 * nonzero + 0.5 * eob` per plane, `+4` for inter luma).  That proxy
cannot see level magnitudes, the EOB symbol, the base/BR split, escapes or
signs.  R19 instead walks the exact §8.3 coefficient symbol sequence of the
decided TX plan and charges every symbol through the §8.2.6 coder interval:

```
bits += log2(SymbolRange / interval_width)
SymbolRange = interval_width << renorm_bits        // 15 - floor_log2(width)
```

That is `-log2` of the interval the real coder would hold after the symbols,
computed without emitting a bit.  Two differences from R01 (`-log2 p` from the
CDF) are the point of the experiment:

* the §8.2.6 boundaries round to an 8-multiple and use `Prob_Inc`, so a rare
  symbol is priced like the coder floors it, not at a theoretical `-log2 p`;
* the CDF rows are a scratch copy of the **frame-start §8.2 image**
  (`cdf_seed`, i.e. the carried live state; §9.3 defaults on the first frame or
  with the chain off) and adapt *inside the leaf* whenever the frame adapts.

`AV2ENC_RATE_INTERVAL=exact` runs the same sequence through
`av2::Encoder` into a scratch buffer and reports the unmaterialized prefix
bits.  That is the R05 reference this branch compares against (R05's own
branch was empty when this experiment ran, so the reference is the documented
model, implemented here as an ablation value).

### Coverage

Priced: TX partition symbols (§5.20.6.3, through the reference module), luma
inter TX type (§5.20.8.2), the full §5.20.7.27 coefficient syntax of every
plane (`txb_skip`, EOB prefix/extra, `coeff_base`/`coeff_base_eob`,
`coeff_br`, `read_quant`, `dc_sign`/sign bits), and the inter mode group
(`is_inter`, `skip`, `single_mode`, `use_bawp`, DRL, shell-coded MVD).

Not priced: intra `y_mode` set/index/offset and the intra luma TX type (the
neighbour-derived mode-list inversion lives in the writer), CDEF/CCSO/delta-Q
filter symbols, and compound/TIP/AMVD syntax.  The error report compares
against the same groups, so an omission cannot masquerade as accuracy.

Coder cost is one `fast_log2` table lookup per symbol instead of libm; the
microbenchmark below is the measured trade-off against the trial encoder.

### Where it is used

`InterFrameEncoder::plan_rate_bits()` replaces `luma_plan_bits` /
`estimate_tx_bits` at:

* the inter skip-vs-residual decision,
* the intra/inter leaf decision (both the `--video-leaf-rd` path and the
  opt-in residual path),
* the measured leaf cost of the exact partition refinement,
* the partition probe's leaf price.

The state is frame-start, not leaf-start: decisions run before the tile
writer (and the wavefront decides many superblocks before the serial tile
stream exists).  The resulting error is the measured quantity below, not a
hidden assumption.

## Model error (protocol requirement)

`--nightly --qp 110` on `nasa_480x240_30fps_60f.y4m`, 60 frames, 8 threads,
gate on with `AV2ENC_RATE_INTERVAL_STATS=1`.  Error = predicted minus the
writer's symbol information cost for the same groups (mode + residual for an
inter leaf, residual only for an intra leaf; the intra mode group is excluded
on both sides).  `legacy` is the historical heuristic on the same leaf, so the
pair answers whether the model reduces error.

| Class | Leaves | Actual mean (bit) | Legacy median / p90 / bias | R19 median / p90 / bias | R05 exact median / p90 / bias |
|---|---:|---:|---:|---:|---:|
| inter, all | 2878 | 6.1 | 0.00 / 3.23 / +3.38 | 1.43 / 8.57 / +3.55 | 1.91 / 9.19 / +4.13 |
| inter, coded residual | 358 | 36.5 | 17.45 / 58.63 / +27.20 | **9.72 / 26.78 / +12.89** | 10.28 / 27.43 / +13.39 |
| intra, keyframe (coded residual) | 256 | 360.5 | 52.38 / 93.85 / +51.26 | 189.63 / 564.43 / +232.13 | 190.56 / 565.17 / +232.60 |
| intra, P frame (coded residual) | 50 | 0.3 | 2.79 / 2.93 / +2.66 | **0.45 / 0.71 / +0.51** | 0.89 / 1.31 / +0.92 |
| bsize 8x8 | 391 | 9.8 | 0.00 / 7.53 / +2.54 | 0.00 / 5.51 / +5.18 | 0.00 / 5.76 / +5.26 |
| bsize 16x16 | 422 | 4.8 | 0.00 / 1.06 / +2.84 | 0.00 / 2.86 / +1.70 | 0.00 / 3.28 / +1.77 |
| bsize 32x32 | 1114 | 7.1 | 0.00 / 19.42 / +5.90 | 0.00 / 9.25 / +2.38 | 0.00 / 9.74 / +2.47 |
| bsize 64x64 | 222 | 389.7 | 56.19 / 93.85 / +56.19 | 210.62 / 564.43 / +248.88 | 211.57 / 565.17 / +249.35 |

Reading:

* On the decision-relevant P-frame residuals the model roughly halves the
  error: median +9.7 vs +17.5 bits, p90 26.8 vs 58.6 bits, bias +12.9 vs
  +27.2 on a 36.5-bit mean (relative +35% vs +75%).  Every large P-frame block
  size (32x32, 64x64, 16x16) improves.
* On the keyframe the model is much worse than the heuristic: median error
  +190 vs +52 bits and mean bias +232 vs +51.  The reason is the state, not
  the arithmetic: a keyframe starts from the §9.3 defaults and its coefficient
  CDFs sharpen over the whole tile, while the model restarts from the
  frame-start image at every decision and only adapts inside one leaf.  The
  model is not consumed by any keyframe decision (`use_intra` is forced on a
  keyframe and its leaves never enter the intra/inter comparison), so this
  column is a reported limitation rather than a decision input.  On P-frame
  intra leaves, where the previous frame carries the CDF image, the model is
  better than the heuristic (bias +0.51 vs +2.66).
* R19 vs R05: the interval width is within ~1 bit/leaf of the real encoder's
  prefix bits (median difference 0.2-0.8 bits, bias +0.5 bits), i.e. the
  arithmetic is not the error term.  The CDF-state approximation is.

## Decisions changed

Same run: 3,184 coded leaves, 29,353 `plan_rate_bits` calls, 18 skip/residual
flips (all skip -> residual) and 50 intra/inter flips (all inter -> intra)
against the historical heuristic on the same candidates (counted inside the
decision, not by diffing streams).  At QP106 the block mix moves from 2,906
inter / 256 intra to 2,941 / 267, and the drift-refresh schedule moves from 9
to 10 frames because a refresh trigger reads the decoded reference quality,
which the decisions changed.

The flips are the mechanism of the K loss, and they are one-sided:

* The model predicts P-frame intra residuals much better than the heuristic
  (50 leaves: actual 0.3 bit mean, legacy bias +2.66, R19 bias +0.51).  On
  those same leaves the intra *mode* syntax is not priced by the model (the
  writer's neighbour-derived y_mode inversion is out of scope) and the
  heuristic charges ~5 bits where the writer spends 9-15.  Making one term
  accurate moved 50 marginal leaves across the threshold with the other term
  still under-priced.
* The same asymmetry pushes skip -> residual: the model's inter residual bias
  is +12.9 bits instead of +27.2, so residuals look ~14 bits cheaper and 18
  borderline leaves start coding a residual.

In other words: the better estimate is real, but the leaf objective around it
(lambda, mode syntax) was calibrated against the old biased estimate.  That is
the finding about the decisions, not about the coder arithmetic.

## Reproduce

```sh
BIN=build-dev/av2enc
REF=testdata/nasa_480x240_30fps_60f.y4m

# model error + decision counters (one 60-frame run)
AV2ENC_RATE_INTERVAL=1 AV2ENC_RATE_INTERVAL_STATS=1 \
  "$BIN" "$REF" -o /tmp/x.ivf --video --frames 60 --qp 110 \
  --cpu-threads 8 --nightly

# one K point (run the protocol's five commands, gate on)
AV2ENC_RATE_INTERVAL=1 "$BIN" "$REF" -o /tmp/x.ivf --video --frames 60 \
  --cpu-threads 8 --qp Q --nightly
tools/bin/dav2d-macos --strict 1 -i /tmp/x.ivf -o /tmp/x.y4m --muxer yuv4mpeg2
DAV2D=tools/bin/dav2d-macos python3 tools/quality.py --ref "$REF" --ivf /tmp/x.ivf --json /tmp/q.json
python3 tools/extra_metrics.py --ref "$REF" --test /tmp/x.y4m --json /tmp/e.json
python3 tools/kpi.py --quality-json /tmp/q.json --extra-json /tmp/e.json
```

Drift adds `--video-drift-refresh-threshold 1.5 --video-drift-refresh-budget
3250`; the 720p pin uses `space8k_1440x720_60f.y4m` with
`--video-key-qp-offset -57 --video-leaf-rd-lambda 250 --video-ext-partitions`.
R05 is the same run with `AV2ENC_RATE_INTERVAL=exact`.

## Validation

Gate off: all five pins byte-identical (25,009 `a1d0dba1`, 25,141 `3984677b`,
23,690 `1b99127a`, 24,113 `7f03eb6e`, 720p 36,572 `c4ba2813`).
Gate on: `dav2d --strict 1` decodes 60/60 at 240p and 720p; the visible
encoder recon (`--video-recon`) is payload-identical to the decoded stream
(4-frame 640x360 nightly clip); `tools/test.sh` 10/10; `gate_video_nightly`
and `gate_video_qp_lambda` pass.

## Encode cost vs R05

Coder-only microbenchmark (`bench/rate_interval/bench_rate_interval.cpp`,
one fresh sink per 512-symbol leaf, adaptation on, same symbol sequence):

| Model | ns/symbol (loaded host, 3 runs) | bits/leaf |
|---|---:|---:|
| interval (R19) | 60.8 / 98.9 / 107.2 | 272.3 |
| exact trial encode (R05) | 135.7 / 257.1 / 244.9 | 273.0 |

The trial encoder is **2.3-2.6x** slower per symbol on the same sequence and
agrees to 0.7 bit/leaf (0.26%).  Absolute times on this host are inflated by
the other branches' encodes; only the ratio is quoted.  Inside the encoder
both are dominated by the shared §8.3 context derivation (0.17-0.51 us per
residual leaf across runs, too noisy on this host to separate the coder).

## K at the bars

All numbers are the gate-on arm on the same host, `--frames 60 --cpu-threads 8`.
Rate is interpolated linearly in bytes (30 fps for 240p, and the pin's own
36,572-byte axis for 720p).  Bars are the protocol's fixed gate-off references
(240p plain K 82.15 at 100.0 kbps; drift K 83.80; 720p K 45.61 / JOD 7.3236).

240p plain (`--nightly`, QP 105/106/107):

| QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|
| 105 | 25,380 | 101.52 | 9.6912 | 3.3968 | 80.091 | 81.86 |
| 106 | 25,170 | 100.68 | 9.7165 | 4.0524 | 80.291 | 80.93 |
| 107 | 24,704 | 98.816 | 9.7137 | 3.9167 | 80.378 | 81.14 |
| **iso 100.0** | 25,000 | 100.0 | **9.7155** | **4.0029** | **80.323** | **81.01** |
| bar | | | 9.7334 | 3.4202 | 80.837 | 82.15 |
| delta | | +0.65% @ QP106 | -0.018 | +0.58 | -0.51 | **-1.14** |

240p drift (`--nightly --video-drift-refresh-threshold 1.5
--video-drift-refresh-budget 3250`, QP 109..113; the gate-on arm spends more
bits per QP, so the 100 kbps interpolation needs two QPs below the protocol's
top bracket):

| QP | bytes | kbps | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|---:|
| 109 | 25,927 | 103.71 | 9.7379 | 2.9276 | 81.099 | 83.19 |
| 110 | 25,489 | 101.96 | 9.7324 | 3.2855 | 80.741 | 82.38 |
| 111 | 25,384 | 101.54 | 9.7445 | 2.9015 | 81.018 | 83.27 |
| 112 | 24,670 | 98.68 | 9.7149 | 2.8556 | 80.585 | 83.15 |
| 113 | 24,602 | 98.41 | 9.7542 | 2.8491 | 81.320 | 83.48 |
| **iso 100.0** | 25,000 | 100.0 | **9.7286** | **2.8768** | **80.785** | **83.21** |
| bar | | | 9.7605 | 2.6562 | 81.25 | 83.80 |
| delta | | | -0.032 | +0.22 | -0.47 | **-0.59** |

720p pin recipe (QP 216/217/218):

| QP | bytes | JOD | BA | SS2 | K |
|---|---:|---:|---:|---:|---:|
| 216 | 37,723 | 7.3702 | 29.6860 | -8.836 | 46.01 |
| 217 | 36,907 | 7.3311 | 29.8218 | -10.013 | 45.61 |
| 218 | 36,030 | 7.2981 | 30.6467 | -10.851 | 45.24 |
| **36,572 B** | 36,572 | **7.3185** | **30.137** | **-10.333** | **45.47** |
| bar | 36,572 | 7.3236 | 28.8713 | -10.146 | 45.61 |
| delta | | -0.005 | +1.27 | -0.19 | **-0.14** |

At a **fixed QP** the gate-on stream is dominated at all three points:

| Point | control | gate on | delta |
|---|---|---|---|
| 240p QP106 | 25,008 B / K 82.2 | 25,170 B / K 80.93 | +0.65% rate, -1.27 K |
| drift QP110 | 25,146 B / K 83.94 | 25,489 B / K 82.38 | +1.36% rate, -1.56 K |
| 720p QP217 | 36,572 B / K 45.61 | 36,907 B / K 45.61 | +0.92% rate, 0.00 K |

Every arm loses at the bar.  The 240p losses (-1.14 / -0.59 K) are inside the
documented +/-1.6 K adjacent-QP ripple, so the defensible statement is "no
gain", not "a measured regression"; the 720p point is flat (-0.14 K,
-0.005 JOD) with a clear Butteraugli loss (+1.27 BA at the pin rate).

## Verdict

**Negative at the bars.**  The model is a good *coder*: interval width agrees
with a full §8.2 trial encode to ~0.5-0.7 bit/leaf and is 2.3-2.6x cheaper per
symbol, and on the decision-relevant P-frame residuals it halves the heuristic's
error (median +9.7 vs +17.5 bits, bias +12.9 vs +27.2 on a 36.5-bit mean).
That is not enough.  At matched rate it loses **-1.14 K** at the plain 240p
point, **-0.59 K** on the tuned drift recipe, and is flat at the 720p pin
(**-0.14 K**, JOD -0.005) while Butteraugli degrades at every point.  At fixed
QP every arm spends more rate for equal or worse quality.

The mechanism is decision-side, not coder-side: accurate residual pricing
flips 68 marginal leaves into coding a residual or into intra, while the rest
of the leaf objective around it (the leaf lambda and, for intra, the mode
syntax, which the model does not price) was calibrated against the old biased
estimate.  A follow-up that wants this model to move K must re-calibrate the
leaf objective around an accurate rate term, and must solve the keyframe
CDF-state problem before the intra side is trustworthy (replaying the frame's
own adaptation in raster order conflicts with the wavefront's parallel
superblocks; a per-superblock replay is the obvious next experiment).

Do not ship this by default.  Reusable pieces: `IntervalSink` (the §8.2.6
interval arithmetic without bit emission), the §8.3 symbol enumeration through
`rdoq::av2::SymbolSink`, and the per-leaf error harness
(`AV2ENC_RATE_INTERVAL_STATS=1`).
