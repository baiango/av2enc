# The four residual inference-based verdicts, measured

Every item here was previously settled by inference: a source comment, a
single-value sweep, or a 2-frame carrier. Each is now measured.

## A. `--video-drift-refresh-{threshold,budget,qp}` on a 60-frame carrier

The feature engages check first -- 60 frames, `--nightly` (which enables drift
refresh), versus `--video-drift-refresh-off`:

```
off 12,885 B   on 12,885 B   -> identical
```

Identical, so the five option runs are all inert:

```
--video-drift-refresh-threshold 1   12,885 B INERT
--video-drift-refresh-threshold 3   12,885 B INERT
--video-drift-refresh-budget    1   12,885 B INERT
--video-drift-refresh-budget    3   12,885 B INERT
--video-drift-refresh-qp        3   12,885 B INERT
```

But "inert" is not the interesting part -- the mechanism is. `--video-trace`
prints the per-frame drift decision, and the path **is reached**, 59 times:

```
drift frame=1 slot=0 anchor=39.6698 ref_psnr=39.6698 spent=0 refresh=0
drift frame=2 slot=0 anchor=39.6698 ref_psnr=39.7542 spent=0 refresh=0
drift frame=3 slot=0 anchor=39.6698 ref_psnr=39.729  spent=0 refresh=0
```

The trigger is (`video_frame.cpp:2990`):

```cpp
drift_refresh_frame = ref_psnr < drift_anchor_psnr - options.drift_refresh_threshold_db
                      && drift_spent_bytes < budget;
```

On this clip the reference PSNR stays *above* the anchor (39.75 vs 39.67), so no
frame ever qualifies -- and raising the threshold makes it strictly harder, not
easier (`anchor - threshold` shrinks). Verified: threshold 10 and 25 both leave
the stream at 12,885 B with `refresh=1` count 0.

**Verdict:** the feature is wired, reached, and content-gated. The three options
are inert on any clip without drift; their liveness is **not demonstrated**, and
I did not manufacture a drifting clip to force it. That is the honest state.

## B. The integer gates at a third value (=64)

The 38 gates called "boolean-equivalent" from `=2` alone, re-run at `=64`
against the 12-frame nightly baseline (9,973 B):

| value | EFFECTIVE | INERT |
|---|---:|---:|
| `=2` | 2 | 38 |
| `=64` | **2** | **38** |

The same two respond at both values and they are mode selectors, where the
value carries meaning:

```
AV2ENC_RATE_HYBRID_S=64   9,941 B  EFFECTIVE
AV2ENC_RATE_INTERVAL=64   9,971 B  EFFECTIVE
```

The other 38 are inert at two distinct non-default values, which is a
measurement now rather than an inference from one. They read the variable as
"enabled if non-zero" (or are not reached on this carrier), and no third value
changes that.

## C. `RD_ENGINE_{KEY,INTER,CHROMA}` -- inert by construction, not by comment

Measured on the nightly-b pin carrier (26,504 B): all three INERT at 0.5 and
4.0.

The structural reason, which makes this a proof rather than an observation:
`site_scale()` -- the only function that reads these three variables -- is
referenced in exactly one place, `print_stats()`:

```
video_rd_engine.h:510  inline double site_scale(Site site) {   // reads the env
video_rd_engine.h:718     site_scale(Site::kKeyIntra), ...      // inside print_stats
video_rd_engine.h:719     site_scale(Site::kChroma));
```

and the three site counters those stats print are never incremented anywhere in
`src/`. So no code path multiplies a price by these scales: they can only ever
change a diagnostic line, never a stream. That is deliberate --
`video_rd_engine.h:505-511` says the unwired sites are held at 1.0 "so enabling
the gate never silently re-tunes them".

## D. The other two joint-plan companions

`AV2ENC_RD_JOINT_P=1` fires with either of the companions the warning names,
not just the one tested earlier:

```
AV2ENC_PIPE_PERC=1                9,981 B   +JOINT  9,983 B   -> JOINT FIRES
AV2ENC_PIPE_BA=1                  9,997 B   +JOINT  9,982 B   -> JOINT FIRES
AV2ENC_EFF_PERC=1                10,026 B   +JOINT 10,074 B   -> JOINT FIRES
```

All three verified, so the warning's message is correct as written.

## Revised status of the four

| item | before | now |
|---|---|---|
| drift-refresh options | "inert, never triggers on 2 frames" (inference) | measured inert on 60 frames; path reached 59x; content-gated, liveness unproven |
| 38 integer gates | "boolean-equivalent" from `=2` | inert at `=2` **and** `=64`; 2 mode selectors live at both |
| `RD_ENGINE_{KEY,INTER,CHROMA}` | unwired "per the comment" | proven: `site_scale()` is only ever called from `print_stats()`; counters never incremented |
| joint companions | one companion tested | all three fire the solve |
