# Ceiling allocation: measured closed

Status: diagnostic, 2026-11-19. The f0-32 source-replacement ceiling is the
project oracle (90.197 at 240p/100 kbps). This note records five allocation
attempts to move it. None did, so the remaining gap is coding efficiency, not
where the bits go.

## Setup

`AV2ENC_EFF_PERC=1`, drift recipe 1.5/3250, 240p, 60 frames, AVM oracle,
matched to 100.000 kbps. `AV2ENC_FRAME_QP_PLAN=<head>:<tail>` is the new
default-off probe: it adds a per-frame `base_q_idx` offset to frames 1-32 and
33-59. A per-frame base_q_idx is legal with the carried CDF image that
`--fast-decode` enables (`--cdf-continuity`); the 2.9% reset applies to
delta-Q/segmentation, not to this path.

## Results

| variant | matched K @100 | f0-32 | f44-59 |
|---|---:|---:|---:|
| eff-perc baseline | **85.099** | **90.197** | 88.995 |
| drift budget 3,250 → 8,000 | 84.868 | 89.971 | 88.656 |
| head -3 / tail +1 | 85.091 | 90.161 | 89.048 |
| head -4 / tail +2 | 85.030 | 89.899 | 88.835 |
| key -30 / tail +2 | 84.832 | 89.928 | 88.856 |
| key -30 / head -1 / tail +2 | 84.746 | 89.817 | 88.740 |

Reference: global QP116 (not an iso-rate stream) reaches f0-32 **90.519** at
**109.57 kbps**. The +0.32 ceiling costs 9.6 kbps; every attempt to fund it
from the tail at 100 kbps loses more than it gains, because the head inherits
the key reconstruction and the tail QP raise costs the tail row.

## Verdict

- The f0-32 ceiling at 100 kbps is **90.197** and is not allocation-limited.
- Do not retry head/tail QP splits, key-offset spends, or refresh-budget
  increases as a ceiling lever.
- The ceiling is limited by head prediction/residual coding efficiency. That is
  the only open lever for lanes 1/3/5.
- `AV2ENC_FRAME_QP_PLAN` stays default-off; unset keeps every pin byte-identical
  (verified: nightly QP106 25,008 B `0639b43630abb454f7be5ac494d2efb4`).

## Direction of the ceiling

The ceiling replaces the named segment with source, so it measures the **rest**
of the stream. A head tool can only raise the f44-59 ceiling; a tail tool can
only raise the f0-32 ceiling. Passing a head tool on f0-32 is a logic error:
the head is the segment being replaced. Lane 1's head allocator demonstrates
it: f0-32 86.365, but f44-59 **88.685 vs nightly 88.066 (+0.62)**.
