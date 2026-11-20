# RD-engine continuous loop

Status: **active**. One agent iteration at a time on the RD-engine queue. The
runner (`tools/agent_loop.py`) measures each iteration with `tools/envelope.py`
against the 86.013 base and keeps or reverts `src/`.

## Base

K = **86.013** (bracket) / **85.580** (fit) at 240p/100 kbps, L3+L5 merged.

Runner base env (merged into every arm; do not re-specify):

```
AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1
AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50
AV2ENC_RD_KEY_TX=1 AV2ENC_RD_UVLEVEL=1
```

Base args:

```
--video --frames 60 --cpu-threads 1 --fast-decode
--video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
```

Pins (gate off): QP106 25,008 B / `0639b436…`; QP110 23,690 B / `1b99127a…`.

## Oracle ceiling target (f0-32 > 91.068)

The f0-32 source-replacement ceiling is limited by the tail f33-59. The
tail-TX threshold is saturated. The next lever is the exact-rate TX/level RD
(the L3 class) applied to the tail frames: fidelity per bit, not rate cuts.
Report the f0-32 and f44-59 ceilings with every result, not only matched K.


## Fidelity-preserving tools (priority)

Rule: a tool that cuts rate must be fidelity-neutral at fixed QP. The closed
rate-cut tools (RDOQ, TCQ, dead zone, TX band) lost because they traded
high-frequency fidelity for bytes. These replace that trade with explicit
fidelity pricing.

1. Perceptual loop-filter RD: decide CDEF/deblock with BA/SS2/CSF, not SSE;
   skip filtering where it removes detail.
2. Per-part D in the key TX/level trellis: use the TX search's per-part
   reconstruction, not a block-level CSF re-weighting (lane 2's D fix).
3. Per-coefficient perceptual quantisation: coefficient-level CSF cost instead
   of a global AC dead zone.
4. Per-block MC filter: EIGHTTAP/SHARP/SMOOTH per block by RD (the frame-global
   forms were negative; per-block unmeasured).
5. Reconstruction-domain quantisation: account for the decoder's integer
   inverse/rounding in the trellis.

## Queue (pick the first unblocked item, one per iteration)

1. **Head residual/QP budget** — f1–32 is 45 B/frame vs SVT's 82 and is the
   f0–32 ceiling limiter. Try a head-frame QP/residual offset funded by the key
   or tail; measure the f0–32/f44–59 ceilings as well as K.
2. **Per-block MC filter / GME/warp** — the frame-global forms were negative;
   the per-block switchable form is unmeasured at 100 kbps.
3. **DDT / IST / parity hiding at the KPI point** — implemented and opt-in,
   never scored in the 100 kbps aggregate.
4. **Key partition with sequential recon** — only if `rd-seq` misses; the
   candidate must be measured in its own reconstruction.
5. **Live-context rate model** — only if `rd-rate` misses.
6. Next target from the census; record why.

## In flight (do not duplicate)

- `rd-rate` live-context rate model
- `rd-seq` sequential-recon tree
- `rd-chroma-ctx` chroma live contexts
- `rd-repair` net-of-repair propagation

## Closed (do not repeat)

Key rate cuts (RDOQ/TX band/bypass/EOB/TCQ/CCTX/exact cost/TX plan/mode RD),
inter λ rescaling, reference structure (B/ARF/hierarchical), tail-TX knob
sweeps, filters/AQ/dead zones/QM, horizon/skip propagation in the current
units.

## Ledger

| # | target | patch | probe | K@100 | verdict |
|---:|---|---|---|---:|---|
