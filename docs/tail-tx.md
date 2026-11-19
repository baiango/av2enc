# Tail frequency-aware TX: verified ceiling gain

Status: **verified on main, 2026-11-19.** The tool came from lane 4
(`exp/agent-loop-oracle-4`, commit `6ea67476`), independently re-measured on
main with a 3-point ladder and the pins intact.

## What it does

`AV2ENC_TX_DETAIL_SEARCH_TAIL=1` with `AV2ENC_TX_DETAIL_MIN=0.50`: on frames
33-59 only, admit the finer 4x4/8x8-class inter TX plan where the source 2-4 px
band energy is high, priced by the exact plan rate. Default off.

## Measurement (240p, 4:2:0, 60f, AVM oracle, QP 119/120/121)

| | matched K @100 | f0-32 ceiling |
|---|---:|---:|
| eff-perc base | 85.099 | 90.197 |
| tail TX, threshold 0.50 | **85.297** (+0.198) | **90.959** (+0.762) |

Segment detail: f33-43 87.300 -> 87.058, f44-59 88.995 -> 88.566. The gain is
the head-replacement row, i.e. the tail got better; the tail's own ceiling
drops because the head is unchanged.

## Validity

- Pins: 25,009 `a1d0dba1`, 25,141 `3984677b`, 23,690 `1b99127a`.
- QP119 candidate stream 25,569 B / 102.3 kbps; `dav2d --strict 1` decodes and
  the decoded payload equals `--video-recon` byte for byte.
- The lane's 2-point gate printed K 85.385; the 3-point ladder gives 85.297.
  Both clear the +0.15 bar; the ceiling is 90.959 in both.

## Status

The tool and the BA pipeline module it was committed with are in main. The
reported number is the 3-point one. This is the first verified gain above both
the 85.099 K and the 90.197 ceiling.
