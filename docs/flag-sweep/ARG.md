# Argument-taking option sweep (2026-11-27)

All 65 CLI options that take a value, each run at two values (its documented
default and a different one) against a fixed baseline (`nasa_480x240`, 12 frames,
QP150, `--nightly`, 1 thread). Raw: `docs/flag-sweep/arg-baseline.txt`.

| class | count |
|---|---:|
| EFFECTIVE | 13 |
| INERT | 24 |
| REJECTED (invalid for that option's type/range) | 93 |

So the argument surface is now judged rather than skipped: 130 runs, and the
options that change the stream are separated from those that cannot at the values
tested.

## INERT entries and why

Most are inert because the value equals the default or the feature is off:

- `--bit-depth=8`, `--video-gops=1`, `--video-refs=1`, `--video-cdef-strength=1`,
  `--video-me-range=16`, `--video-max-depth=4` -- the documented defaults.
- `--video-recon=*` -- writes a recon side file; the stream is correctly
  unchanged.
- `--video-drift-refresh-*` -- the drift-refresh feature is a measured no-op
  (default `-qp 0`).
- `--video-lossless-intra-period` -- lossless is off.
- `--p-frames`, `--tiles=1` -- not the controlling option for this mode.

## Shortlist worth a targeted re-test (not yet judged properly)

These stayed inert at a value that should matter, so they are the only remaining
"suspicious" entries:

- `--video-min-block=32` (default 16; 32 inert on this clip)
- `--video-me-range=32` (default 16; 32 inert)
- `--video-max-depth=8` (default 4; 8 inert -- earlier a still also showed
  depth 6 identical, consistent with the tree already reaching its cap)
- `--video-key-qp-offset=1|3` (should shift the key QP)

A single-QP, single-clip test cannot separate "clamped/ignored" from "no decision
flipped", so these need a two-QP ladder before any call is made.
