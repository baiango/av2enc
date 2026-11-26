# Scalar env gates: the `=1` and `=2` sweeps could never judge them

## Which gates are scalar

21 `AV2ENC_*` gates are read as a **double**, via `env_positive(name, fallback)`
(`video_frame_internal.h:205`, `video_rd_engine.h:83`) or
`pb::env_number(name, fallback)` (`video_pipe_budget.h:67`):

```
AV2ENC_PIPE_BUDGET_{ACT_MAX,CHROMA,GAMMA,LUMA,STRENGTH}
AV2ENC_RD_DRIVER_{ACT,LAMBDA,SBALLOC_MAX}
AV2ENC_RD_ENGINE_{CAL_BASE,CAL_BR,CAL_EOB,CAL_MAG,CAL_SIGN,CAL_TXB,CHROMA,INTER,KEY}
AV2ENC_RD_KEY_TX_SCALE
AV2ENC_RD_{UVLAMBDA,UVLEVEL_LAMBDA,UVLEVEL_SCALE}
```

For these, `=1` is the default of most of them (`env_positive("...", 1.0)`) and
`=2` is inside the range where the value is clamped or simply does not change a
decision. Sweeping at `0.5` and `4.0` is the only way to see whether the value
reaches anything.

## Take 1 -- `--nightly`: 21 of 21 inert at both magnitudes

Baseline `--video --frames 12 --qp 150 --cpu-threads 1 --nightly`, 9,973 B /
`2f4f4b39bb202e813337fa1d6c050339`:

```
0.5:INERT  4.0:INERT      (all 21 gates)
```

Not a single gate moved. That is **not** a verdict, and take 2 shows why.

## Take 2 -- `--nightly-b`, the mode these gates actually serve

`AV2ENC_RD_DRIVER_LAMBDA` is in the pinned commands (`=256` on the src720 arm),
so the gates are live somewhere. Re-run on the pin carrier
(`--video --frames 1 --qp 186 --cpu-threads 1 --nightly-b --video-gdf
--video-uv-qp-offset -23`, `AV2ENC_RD_DRIVER_MIN=16`), baseline 26,504 B /
`3eb12dbd13960422e2aee25f4f102946`:

| gate | 0.5 | 4.0 |
|---|---|---|
| `AV2ENC_RD_DRIVER_LAMBDA` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_RD_ENGINE_CAL_BASE` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_RD_KEY_TX_SCALE` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_RD_UVLAMBDA` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_RD_UVLEVEL_LAMBDA` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_RD_UVLEVEL_SCALE` | **EFFECTIVE** | **EFFECTIVE** |
| `AV2ENC_RD_ENGINE_CAL_EOB` | INERT | **EFFECTIVE** |
| `AV2ENC_RD_ENGINE_CAL_MAG` | **EFFECTIVE** | INERT |
| `AV2ENC_RD_ENGINE_CAL_SIGN` | INERT | **EFFECTIVE** |
| `AV2ENC_RD_DRIVER_ACT` | INERT | INERT |
| `AV2ENC_RD_DRIVER_SBALLOC_MAX` | INERT | INERT |
| `AV2ENC_RD_ENGINE_CAL_BR` | INERT | INERT |
| `AV2ENC_RD_ENGINE_CAL_TXB` | INERT | INERT |
| `AV2ENC_RD_ENGINE_CHROMA` | INERT | INERT |
| `AV2ENC_RD_ENGINE_INTER` | INERT | INERT |
| `AV2ENC_RD_ENGINE_KEY` | INERT | INERT |
| `AV2ENC_PIPE_BUDGET_{ACT_MAX,CHROMA,GAMMA,LUMA,STRENGTH}` | INERT | INERT |

**9 of 21 are EFFECTIVE.** Direct check of the headline case:

```
baseline (no LAMBDA set)   26,504 B  3eb12dbd13960422e2aee25f4f102946
AV2ENC_RD_DRIVER_LAMBDA=0.5 26,508 B  de87798166a6f6b1a72dda473ba8df48
```

The three `CAL_*` gates that move at only one magnitude are the interesting
ones: `CAL_MAG` responds at 0.5 but not 4.0, while `CAL_EOB` and `CAL_SIGN`
respond at 4.0 but not 0.5. A single-value sweep would have called each of them
inert or live at random.

## The 12 that stay inert, and why that is not a silent disable

- `RD_ENGINE_KEY`, `RD_ENGINE_INTER`, `RD_ENGINE_CHROMA` are **documented unwired
  sites** -- `video_rd_engine.h:505-511` says the scales are "1.0 for the sites
  this milestone does not wire, so enabling the gate never silently re-tunes
  them". Inert by design, stated in the code.
- The 5 `PIPE_BUDGET_*` gates need the allocation armed. Arming it with
  `AV2ENC_PIPE_BUDGET_SCALE=1000` produced **26,504 B / 3eb12dbd -- byte-identical
  to the unarmed baseline**, so the allocation block is not entered on this
  carrier at all. These 5 are **unjudged**, not dead; the arming knob I tried
  does not reach them.
- `RD_DRIVER_ACT`, `RD_DRIVER_SBALLOC_MAX`, `CAL_BR`, `CAL_TXB` trade against
  quantities that this one-frame carrier does not move.

## A void attempt, recorded

The first shell version of the pipe-budget run passed the flag set as `$A`
unquoted. zsh does not word-split, so every invocation died with "unknown
option '--video --frames 1 ...'" and no file was written -- and the two empty
md5s compared **equal**, printing INERT for all five gates. Re-run in Python
(no shell splitting). This is the eighth harness error of the sweep to produce
a false verdict, and the same shape as the others: an all-inert result is a
harness smell, not a measurement.
