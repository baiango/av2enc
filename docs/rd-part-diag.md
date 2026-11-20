# P-frame partition diagnostic (`AV2ENC_PART_DIAG`)

Status: **NEGATIVE — the P-partition pool is closed.** The measured partition
probe clamps the P tree to `partition_probe_depth` = 1, so a 64x64 / 64x32
leaf's children are never priced by the coder's own leaf path. This gate
measures both sides of every emitted P leaf and answers whether that clamp is
leaving splits on the table. On the production recipe it is not: the head's
splits cost rate, and the tail's depth-1 splits are worth ~0.3 % of the
stream, below the +0.15 K bar by an order of magnitude.

## What it measures

Default off. `AV2ENC_PART_DIAG=1` runs the same whole-picture sequential walk
as the tail floor and, at every emitted leaf with a side >= `MIN` (default 16),
encodes

* the leaf itself, and
* each legal one-level split (`HORZ`, `VERT`, `SPLIT`) with the coder's own
  leaf path,

into the live reconstruction, and scores both sides with one objective:

```text
D = seq_rd::leaf_distortion_gains() on the leaf's own reconstruction
    (eff-perc inter low-band CSF + optional HF / block-edge terms)
  + weighted chroma SSE
R = rate_interval::estimate() on the emitted §8.3 symbols
  + the live inter-syntax probe (is_inter / skip / mode / ref / DRL / MVD)
  + 2 bits per partition symbol
J = D + lambda * R,  lambda = the leaf's own pipe lambda
```

The reference leaf is then committed through the normal path, so **the stream
is byte-identical with the gate on and off** (verified: QP116 `/tmp` stream
`775d76acd6ebdf52c9baa9928807cd26`, 23,798 B, both ways). Prediction for an
inter leaf comes from the reference picture, so a trial is exact for it; the
intra leaves inside P (17.7 % of blocks) read the current reconstruction and
are the approximation the diagnostic accepts.

`AV2ENC_PART_DIAG_COMMIT=1` commits the measured winner instead of the
reference. It exists to validate the measurement end to end; it is not part of
the recipe.

| Variable | Default | Meaning |
|---|---|---|
| `AV2ENC_PART_DIAG` | off | master gate |
| `AV2ENC_PART_DIAG_FIRST/LAST` | 1 / 32 | picture window (head); 33 / 59 for the tail |
| `AV2ENC_PART_DIAG_MIN` | 16 | smallest measured leaf side |
| `AV2ENC_PART_DIAG_LAMBDA` | 1.0 | scale on the leaf pipe lambda |
| `AV2ENC_PART_DIAG_HF/EDGE` | 0 / 0 | measured-D high-pass / block-edge weights |
| `AV2ENC_PART_DIAG_MAX` | 0 | cap on measured nodes (0 = all) |
| `AV2ENC_PART_DIAG_COMMIT` | off | commit the winner (validation only) |
| `AV2ENC_PART_DIAG_TRACE` | off | first 12 skip reasons |

Recipe (the ladder base plus the merged head/tail exact-rate arms):

```sh
AV2ENC_EFF_PERC=1 AV2ENC_RD_UVFULL=1 AV2ENC_TX_DETAIL_SEARCH_TAIL=1 \
AV2ENC_TX_DETAIL_MIN=0.50 AV2ENC_RD_KEY_TX=1 AV2ENC_RD_UVLEVEL=1 \
AV2ENC_RD_DRIVER=1 AV2ENC_RD_HEAD_TX=1 AV2ENC_RD_TAIL_TX=1 \
AV2ENC_PART_DIAG=1 av2enc testdata/nasa_480x240_30fps_60f.y4m \
  -o /tmp/d.ivf --video --frames 60 --qp 116 --cpu-threads 8 --fast-decode \
  --video-drift-refresh-threshold 1.5 --video-drift-refresh-budget 3250
```

## Measured (production recipe, QP 114 / 116)

| picture window | QP | nodes | split wins | D leaf → split | rate on winners | verdict |
|---|---:|---:|---:|---|---:|---|
| head 1-32 | 116 | 780 | 63 (8.1 %) | 5.875 M → 5.825 M (−0.85 %) | **+80.8 B** | costs rate |
| head 1-32 | 114 | 817 | 79 (9.7 %) | 5.854 M → 5.811 M (−0.73 %) | **+49.0 B** | costs rate |
| tail 33-59 | 116 | 5,275 | 970 (18.4 %) | 10.090 M → 9.842 M (−2.5 %) | −289.3 B | of which depth-1 **−67.1 B** |
| tail 33-59 | 114 | 5,365 | 1,071 (20.0 %) | 9.845 M → 9.593 M (−2.6 %) | −445.4 B | of which depth-1 **−112.6 B** |

Tail QP116 by shape:

| leaf | nodes | split wins | rate on winners |
|---|---:|---:|---:|
| 64x32 | 179 | 26 | −18.0 B |
| 32x32 | 182 | 37 | −5.1 B |
| 32x16 | 500 | 105 | −44.0 B |
| 16x16 | 4,183 | 759 | −196.7 B |

The 16x16 row is the already-measured tail floor (`AV2ENC_RD_TAIL_FLOOR`), not
a new pool. The **depth-1** leaves — the ones the probe clamp hides — are worth
**67 B at QP116 / 113 B at QP114 on the tail**, i.e. 0.3-0.5 % of the stream,
and the head's splits are rate-positive. At the measured exchange rate
(~0.045 K per 1 % of stream) that is **+0.01…0.02 K**, far below the +0.15 KEEP
bar.

Composition check: arming the measured tail floor on top of the head/tail
exact-rate arms gives 23,915 B at QP116 versus 23,798 B without it — the floor
*adds* 117 B at the same QP. Its measured KEEP was +0.033 ceiling / +0.023 fit,
which is inside the same noise band. The pools overlap and do not stack.

## Why the earlier number was wrong

A first run measured the tail at **−604.7 B** and 44 % split wins. That run
used `env $BASE …` from zsh, which does not word-split `$BASE`: only
`AV2ENC_EFF_PERC` was set and the rest of the recipe was silently dropped, so
the tree under test was plain `--fast-decode` (27,393 B at QP116 instead of the
recipe's 24,843 B). On the real recipe the depth-1 pool is 67 B. Any future
lane must set the environment with `env ${=BASE}` (zsh), an array, or `bash`.

## Verdict

* The P tree is priced to within ~1 % on the production recipe. The depth-1
  clamp is not the missing RD gain; the leaves are large because splitting a
  smooth P leaf costs more syntax than it saves.
* Do not build a measured P-frame partition driver. The remaining K gap is
  **BA (−2.82 K, 70 %) and SS2 (−1.06 K, 26 %)** at equal JOD, not tree
  decisions: the error is in the artifact structure the D model and the
  prediction/tool set produce, not in which leaves are chosen.
* Keep the gate as the regression test for that claim: if a future leaf coder
  or tree change makes the splits cheap, `dR_win` will move off zero here
  before it moves K.

## Reproduce

```sh
# head and tail at the two rungs, production recipe
tools/lane-part-diag/diag.sh 114 116
# stream identity with the gate on/off
md5 -q /tmp/partdiag_head116.ivf /tmp/partdiag_head116_off.ivf
```

Conformance: gate off reproduces the pins (QP106 25,008 B /
`0639b43630abb454f7be5ac494d2efb4`; QP110 23,690 B /
`1b99127a4cd1d386fa5600fb63c57a83`); gate on reproduces the recipe stream
byte for byte; release `-Werror` build clean.
