
## Composition result (do not re-run)

Matched K does NOT stack across the merged KEEPs. Dense 6-point ladders
(QP113-118, docs/rd-compose.md): base driver 86.041, +MC 86.184, +tail 86.266,
+both 86.160 (robust fit, R2 0.012). MC's early +0.341 was a 2-point bracket
artefact; the dense-ladder value is +0.143. They compose on rate (both is 2-4%
smaller at every QP), not on matched K.

Gap decomposition vs AVM: CVVDP 9.836 vs 9.862 (equal); BA -2.82 K (70%), SS2
-1.06 K (26%). Split 48% head f1-32 / 48% tail f33-59 / 1% key. The head
spends 45-65 B/frame vs AVM ~197.
# RD-engine lane 2 — P coding-block floor

Focus: Tail/P coding-block 16x16 -> 8x8/4x4 with a measured partition (BA-aware D + exact rate, sequential recon). This is the last structural blocker named by the tail lane.

Base K 86.013; opt-in best 86.283 (driver + per-block MC); ceiling f0-32 91.51.
Base env (merged by the runner): AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1
AV2ENC_TX_DETAIL_SEARCH_TAIL=1 AV2ENC_TX_DETAIL_MIN=0.50 AV2ENC_RD_KEY_TX=1
AV2ENC_RD_UVLEVEL=1 AV2ENC_RD_DRIVER=1. Gate +0.15 K, 3+ arms, keep/revert.

## Closed (do not repeat)

Key rate cuts (RDOQ/TX band/bypass/EOB/TCQ/CCTX/exact cost/TX plan/mode RD),
inter lambda rescaling, reference structure, tail-TX knob sweeps, filters/AQ/
dead zones/QM, horizon/skip propagation in the old units, head QP plan.

## Merged KEEPs

- key chroma UV search (+0.337), key TX+chroma levels (86.013), BA-aware
  sequential key driver (bracket +0.03, fit +0.26, ceiling 91.608), tail
  exact-rate residual (ceiling +0.11), per-block MC (+0.341).

## Ledger

| # | target | patch | probe | K@100 | verdict |
|---:|---|---|---|---:|---|
