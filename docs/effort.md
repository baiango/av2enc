# Effort presets (`--effort` / `--cpu-used`)

Status: **opt-in, Still path only.** The default `--still`, `--still-gpu`,
Basic and `--video` streams are untouched: the selector is never called unless
`--effort` or `--cpu-used` is present.

The table has **one measured row per feature**. The runtime ranks eligible
features by score ascending (efficient first), walks the ranking and takes a
prefix while the cumulative score fits the budget. It never exceeds the budget.
Presets are just rank cutoffs at geometric budgets; there is no separately
calibrated bundle ladder.

## Features

Measured on **Apple M4**, 7 paired repeats, fixture
`testdata/shine_640x360_1f.y4m` (640x360, 1 frame), CVVDP 0.5.7 with
`standard_fhd`, pinned `dav2d` f9dafbc. Quality is interpolated at the 1000 kbps
anchor from the QP grid `110,130,150,170,190,210,230`; the 300/1500/2500 kbps
points are recorded in `tools/effort/effort_table.json`.

| Rank | Feature | Score | Δtime | ΔCVVDP @ 1000 | Equal-CVVDP rate | Marginal | Confidence |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `--still-dip` | **10.0** | +18.3% | +0.266% | −1.90% | 10.0 | medium |
| 2 | `--still-mhccp` | **44.4** | +66.5% | +0.220% | −1.56% | 74.6 | low |
| — | `--still-pc-wiener` | excluded | +249.6% | −0.030% | +0.21% | — | medium |
| — | `--still-rdo-intra` | excluded | +819.3% | −0.080% | +0.55% | — | low |
| — | `--still-full-intra` | excluded | +108.0% | −0.577% | +4.03% | — | medium |
| — | `--still-mrl` | excluded | +828.5% | −0.753% | +5.30% | — | medium |
| — | `--still-tx-type` | excluded | +10.4% | 0.000% | 0.00% | — | medium |
| — | `--still-ist` | excluded | +8.5% | −0.009% | +0.07% | — | medium |
| — | `--still-rdoq` | excluded | +177.6% | −2.239% | +14.97% | — | medium |
| — | `--still-uv-full` | excluded | +10.3% | −0.038% | +0.28% | — | medium |
| — | `--still-cdef` | excluded | +386.7% | −0.269% | +1.88% | — | medium |
| — | `--still-ccso` | excluded | +11.6% | −0.663% | +4.38% | — | medium |
| — | `--still-ext-partitions` | excluded | +685.1% | 0.000% | 0.00% | — | medium |
| — | `--still-ibp` | excluded | +1.0% | −0.181% | +1.34% | — | high |

A feature is **eligible** only when it improves CVVDP at the fixed bitrate and
has a positive equal-CVVDP rate gain. Everything else is excluded from
selection and listed with the measured reason. This is why only two features are
enabled: the rest are neutral or lose quality at iso-rate on this corpus.

### Marginal pass and interactions

The top-K prefix is measured once with the feature on top of the already-ranked
prefix (`--marginal 5`). The incremental cost is stored as `marginal_score` and
is what the runtime charges; the ranking stays the solo score. This catches the
worst non-additivity without a combinatorial search.

For the current table the pair is `--still-dip --still-mhccp`: additive
marginal score **84.6**, measured prefix score **28.8** (**+75.6% time,
−2.72% rate at equal CVVDP**). The budget charges the additive marginal sum
(84.6), which is conservative; the measured prefix score (28.8) is the pair's
own efficiency and is reported for validation. `--still-mhccp` costs 44.4 solo
but 74.6 on top of DIP, because its rate gain overlaps with DIP's. The runtime
uses the 74.6 marginal cost. The measured prefix is stored as
`prefix_validation`; if a future prefix measures worse than its additive
estimate, the tail can be truncated without an exhaustive search.

## Budget and mapping

`--effort N` takes an integer in `[10, 10000]` on a log scale. Features are
added in rank order while `used + cost <= N`; the first feature that would
overshoot stops the prefix and the budget is never exceeded.

`--cpu-used K` is sugar for the budget:

```text
budget(K) = clamp(round(10 * 1000 ** ((8 - K) / 8)), 10, 10000)
```

| `--cpu-used` | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `--effort` | 10 | 24 | 56 | 133 | 316 | 750 | 1778 | 4217 | 10000 |

With the current table:

| Budget | Enabled prefix | Groups |
|---:|---|---|
| 10–84 | `--still-dip` | rank 1 |
| 85–10000 | `--still-dip --still-mhccp` | rank 1–2 |

So `--cpu-used 8,7,6` select DIP, and `--cpu-used 5..0` select DIP+MHCCP. No
budget can buy the excluded features.

### Safety margin

Scores are conservative: encode cost is the **p75 of paired A/B CPU-time
ratios** (the baseline and the candidate are run alternately), not a single
wall-clock sample. That is the "upper-bound score" alternative to a 90% budget
cap; the runtime therefore uses the full budget and the report prints the used
percentage. A noisy host cannot make a feature look cheaper than the measured
p75; if timing is too noisy the calibrator refuses a non-positive top-feature
cost instead of writing a fabricated score.

## Score and scale

The contract formula is

```text
score_raw = 1000 * max(0, d_time_pct) / max(0.25, quality_gain_pct)
```

On this corpus ΔCVVDP is 0.0–0.3%, below the 0.25% floor, so the literal JOD
denominator gives `score_raw` in the 1e4–1e6 range for every feature; nothing
would fit the fixed `[10, 10000]` budget. The table therefore uses the measured
**bitrate change at equal CVVDP** (`d_rate_pct`, negative = fewer bits) as the
quality term:

```text
quality_gain_pct = -d_rate_pct
score_raw        = 1000 * max(0, d_time_pct) / max(0.25, quality_gain_pct)
score            = round(score_raw * score_scale, 1)
```

`score_scale` maps the most efficient measured feature to the minimum budget
(10.0), so the published range spans the table without changing the ordering or
the ratios. `d_cvvdp_pct`, `score_raw`, `score_scale` and every anchor point
are recorded. This is the only measurement substitution; it is documented
because the literal formula cannot be budgeted.

## Calibration tool

```sh
# Re-measure on the current host and regenerate the runtime table
python3 tools/effort/effort_calibrate.py --bin build-int/av2enc --repeats 7 --marginal 5

# Plan only
python3 tools/effort/effort_calibrate.py --bin build-int/av2enc --dry-run

# Validate the generated include
python3 tools/effort/gen_table.py --check
```

`tools/effort/effort_calibrate.py`:

- reads `tools/effort/features.json` (one row per flag, plus the conflict list);
- encodes the baseline and every feature solo on the QP grid;
- records child **CPU time** and re-measures each feature with an alternating
  paired A/B at its anchor QP; the p75 ratio is the conservative score;
- measures the top-K cumulative prefix (`--marginal K`) for the marginal cost;
- measures CVVDP through `tools/quality.py` and `dav2d`; if CVVDP is unavailable
  it refuses to write unless `--allow-psnr-fallback` is passed, and then marks
  every row `psnr_fallback`;
- refuses a row whose 1000 kbps anchor is not bracketed, whose encode failed, or
  whose bitstream differs across repeats;
- records CPU/GPU model, decoder, metric version/display, encoder SHA-256, git
  commit and fixture MD5 (`repeats`, `confidence` and `time_spread_pct` are in
  every row);
- writes `tools/effort/effort_table.json` and regenerates
  `src/pipeline/planner/effort_table.inc` (the compiled-in runtime table).

Deterministic means the command, QP grid, rank order and output schema are fixed
and every bitstream is checked identical across repeats. CPU-time scores are
measurements and vary with host load; the table records the paired-ratio spread
and a `confidence` label. Re-run on the target host before trusting the scores
there.

## Runtime behaviour

When a preset is active the CLI prints the budget, the enabled prefix and the
used budget:

```text
av2enc: effort: budget 85.0 (--effort 85); selected 2 feature(s); used 84.6 of 85.0
  #1 --still-dip (marginal 10.0; 18.3% time, -1.9% rate at equal CVVDP; medium confidence)
  #2 --still-mhccp (marginal 74.6; 66.5% time, -1.6% rate at equal CVVDP; low confidence)
  excluded:
    --still-rdoq: CVVDP does not improve at equal bitrate and the feature costs time
    ...
  top-K prefix calibration: additive 84.6, measured score 28.8 (75.6% time / -2.7% rate at equal CVVDP)
```

Selection is deterministic and offline: no A/B encode happens at encode time.

Explicit feature flags win. If the caller passes a measured flag, the ranking
leaves it to the caller and does not charge it. A feature whose flag conflicts
with an explicit or higher-ranked flag is skipped and reported (`--no-cfl`
skips MHCCP, `--still-mrl` skips IBP/rdo-intra, `--still-sb128` skips
CDEF/CCSO, and so on).

`--effort-plan` prints the report and exits 0 without opening the input or
output. It is what `tools/effort/gate_effort.sh` uses for the budget invariant.

## Presets as cutoffs

There is no separately calibrated preset bundle. `--cpu-used` is only the
geometric budget above; the enabled set is the same ranked prefix the numeric
`--effort` would produce. The old A..H ladder and `--effort-level` were removed:
the exhaustive bundle calibration did not outperform the ranked greedy prefix.

## Gate

```sh
tools/effort/gate_effort.sh --bin build-int/av2enc
tools/effort/gate_effort.sh --bin build-int/av2enc --quick   # skip 720p video
```

The gate checks:

1. `gen_table.py --check` (the compiled-in table matches the JSON).
2. Every feature has measured fields; the ranking is score-ascending.
3. Budget invariant for every `--cpu-used` value and a budget sweep: the
   selected prefix and used score match the table, and `used <= budget`.
4. Explicit flags are kept/not charged; conflicts are skipped.
5. The default still/basic streams are byte-identical to the pins in
   `refactor/baseline.json` when no preset is passed.
6. A still ranked prefix decodes with `dav2d --strict 1` and is measured
   (CVVDP, or a labelled PSNR fallback) against the no-preset baseline.
7. A representative `--video` stream decodes with `--strict 1` and
   `--effort --video` is rejected (no video rows are calibrated yet).

## Remaining uncalibrated flags

Not in `features.json` and therefore never selected:

`--still-wiener`, `--still-chroma-lr`, `--still-gdf`, `--still-grain`,
`--still-cfl-ds`, `--still-cfl-explicit`, `--still-segmentation`,
`--still-delta-q`, `--still-sb128`, `--still-sb256`, `--still-qm`,
`--still-dc-ac-delta-q`, `--still-content-interpretation`, `--still-palette`
(rejected), and the whole `--video` / `--still-gpu` surface.
