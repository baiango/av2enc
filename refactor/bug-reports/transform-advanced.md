# transform-advanced bug scout

Repo: `/Users/arin/work/av2enc` @ `527738a` (read-only). Lane: TCQ/parity
hiding/CCTX, IST, IDTX, DDT/DDTX/FDDT, per-part TX partition; focus on
TCQ+CCTX, IST+RDOQ/TX-type, TX-partition+palette/IBC, state/context resets,
per-part recon ordering. Counts: **2 confirmed (recon mismatch), 1 confirmed
(accepted-and-ignored), 2 duplicate**.

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-TXADV-01 | high | confirmed | `--still-tx-partition` + directional luma (`--still-full-intra`/`--still-rdo-intra`) gives a wrong encoder recon and a decoder/encoder mismatch | see Details | QP20 640x360: 25,955/345,600 samples differ (Y 4,819, U 14,788, V 6,348, max 10); `--no-deblock` same; `AV2ENC_STILL_NO_EDGE=1` makes it exact; tx-partition alone and full-intra alone exact | `src/pipeline/still/still_encoder.cpp:1027-1047,1102,2140,2703-2711`; spec §7.13.2.15/.16 | new |
| SCOUT-TXADV-02 | med | confirmed | `--still-ist --still-mrl` recon mismatch: IST hardcodes `mrl_index = 0` while §7.15.3 selects the kernel from the real MRL angle | see Details | `testdata/shine_640x360_1f.y4m` qp40: 258 Y diffs, qp60: 371 Y diffs, max 1; each flag alone exact; NO_EDGE/no-deblock do not change it | `src/pipeline/still/still_encoder.cpp:1570,1896`; `src/pipeline/still/still_syntax.cpp:1841` | new |
| SCOUT-TXADV-03 | low | confirmed | `--still-cctx` silently skips RDOQ on the chroma planes (accepted-and-ignored; CCTX gate claims the composition) | see Details | with `--no-cfl`, `--still-cctx` vs `+--still-rdoq` changes Y (2827 samples) but U/V are 0/0; RDOQ alone does change chroma on a flat-luma/noisy-chroma fixture | `src/pipeline/still/still_encoder.cpp:1749-1790,2311-2316` | new |
| SCOUT-TXADV-04 | med | confirmed | TX partition aborts on 4x4 coding blocks (`MiSize 0`) with `--still-sub8` | `build/av2enc <64x64>.y4m -o /tmp/t.ivf --still --qp 16 --still-tx-partition --still-sub8 --still-depth 8` | `tx partition geometry failed: ... TX_PARTITION_NONE is not signallable for MiSize 0`; sub8 alone passes | `src/codec/transform/tx_partition/tx_partition.cpp:265,317-330`; `still_encoder.cpp:2110` | duplicate — SCOUT-PARTITION-08 (`bug-reports/partition.md`) |
| SCOUT-TXADV-05 | low | doc/gap | `--still-tcq-scale` accepted and ignored without `--still-tcq` | `build/av2enc tests/data/basic_64x64_1f.y4m -o /tmp/a.ivf --frames 1 --still --still-tcq-scale 4` | exit 0; MD5 `0ef98250…` identical to plain `--still`; with `--still-tcq` the same scale changes it (`53f69411…`) | `src/main.cpp:1994-1996` (guard list omits tcq-scale) | duplicate — SCOUT-CHROMA-01 (`formats-chroma.md`) / SCOUT-FORMATS-HBD-01 |

## Details

### SCOUT-TXADV-01 — per-part prediction uses part-origin smooth flags, decoder uses coding-block origin

Repro:
```sh
cd /Users/arin/work/av2enc
B=build/av2enc; I=tests/data/basic_640x360_1f.y4m
$B $I -o /tmp/txp.ivf --recon /tmp/txp.rec.y4m --frames 1 --still \
    --qp 20 --still-tx-partition --still-full-intra
tools/bin/dav2d-macos -q -i /tmp/txp.ivf -o /tmp/txp.dec.y4m --threads 1 --strict 1
python3 - /tmp/txp.dec.y4m /tmp/txp.rec.y4m <<'PY'
import sys
def p(f):
    d=open(f,'rb').read(); i=d.find(b'FRAME\n'); return d[i+6:]
a,b=p(sys.argv[1]),p(sys.argv[2]); print(sum(x!=y for x,y in zip(a,b)))
PY
```
Observed: 25,955 differing samples (Y 4,819 / U 14,788 / V 6,348, max 10).
`--still-rdo-intra` instead of `--still-full-intra` is worse (48,994 at QP20);
`--still-tx-partition` alone and `--still-full-intra` alone are exact.
`--no-deblock` does not change it; `AV2ENC_STILL_NO_EDGE=1` makes this QP20
case exact (also QP20/QP32 on `testdata/src_640x360.y4m`), so the intra edge
filter is the divergence. Forced-mode
sweep: forcing every block directional (`AV2ENC_STILL_FORCE_Y=2:-3`) is exact;
forcing the first block column SMOOTH (`AV2ENC_STILL_FORCE_Y=9:0
AV2ENC_STILL_FORCE_Y2=2:-3`) reproduces 4,954 diffs. The mismatch is luma
first and propagates to chroma through CfL.

Expected: decoder reconstruction equals the encoder reconstruction; the TX
partition gate explicitly claims the full/rdo-intra composition
(`still_tx_partition_gate.sh`).

Likely cause: `encode_luma_parts()` calls `predict_block()` once per transform
part at the part origin (`still_encoder.cpp:2140`), and `predict_block()` then
derives `above/left_neighbor_is_smooth` from `intra_joint_` at
`(y0>>2, x0>>2)` (`still_encoder.cpp:1027-1047,1102`). The spec derives both
flags from the **coding block** origin (`get_filter_type_above` uses
`MiRow/MiCol`; §7.13.2.15/.16). Because `intra_joint_` was just filled for the
whole coding block (`still_encoder.cpp:2703-2711`), every part after the first
row/column reads the current block's own joint mode instead of the adjacent
coding block the decoder reads. With a SMOOTH/SMOOTH_V/SMOOTH_H neighbour, the
decoder uses `filterType=1` (strength 3 for these sizes) while the encoder uses
0, so the filtered edge and the prediction differ. Only directional modes use
the filter, hence the `--still-full-intra`/`--still-rdo-intra` dependence. The
gate's only directional composition case is 64x64 at QP90 with forced SPLIT,
which does not hit a smooth neighbour.

### SCOUT-TXADV-02 — IST + MRL: `mrl_index` forced to 0

Repro:
```sh
cd /Users/arin/work/av2enc
B=build/av2enc; I=testdata/shine_640x360_1f.y4m
$B $I -o /tmp/im.ivf --recon /tmp/im.rec.y4m --frames 1 --still \
    --qp 60 --still-ist --still-mrl
tools/bin/dav2d-macos -q -i /tmp/im.ivf -o /tmp/im.dec.y4m --threads 1 --strict 1
# body compare (same python as above): 371 Y diffs, U/V 0, max 1
```
Observed: `--still-ist` alone and `--still-mrl` alone are exact, as is
`--still-mrl` with `--still-tx-type`/`--still-rdoq`. The combination differs at
QP40 (258 Y samples) and QP60 (371 Y samples, max 1); `--no-deblock` and
`AV2ENC_STILL_NO_EDGE=1` do not change the result.

Expected: decoder-exact, like the IST gate's other compositions.

Likely cause: the encoder's IST path hardcodes `stx.mrl_index = 0` in the
candidate search (`still_encoder.cpp:1896`), the inverse used for recon
(`still_encoder.cpp:1570`) and the syntax applicability/validation
(`still_syntax.cpp:1841`). §7.15.3 computes
`pAngle = Mode_To_Angle[mode] + AngleDeltaY*ANGLE_STEP + Mrl_Index_To_Delta[MrlIndex]`
before `wide_angle_mapping()` and then picks
`Inv_Most_Probable_Stx_Mapping[mode][set]`, so a block with `mrl_index > 0`
gets a different kernel/transpose at the decoder. `--still-ist --still-mrl` is
accepted by the CLI (no rejection in `still_encoder.cpp:3729-3795`). Fix by
passing the block's real `mrl_index` through the IST decision/recon/syntax, or
reject the pair. The IST gate and MRL gates never cross-compose.

### SCOUT-TXADV-03 — CCTX chroma never runs RDOQ

Repro:
```sh
cd /Users/arin/work/av2enc
B=build/av2enc; I=tests/data/basic_64x64_1f.y4m
$B $I -o /tmp/cc.ivf  --recon /tmp/cc.rec.y4m  --frames 1 --still --still-cctx --no-cfl
$B $I -o /tmp/ccr.ivf --recon /tmp/ccr.rec.y4m --frames 1 --still --still-cctx --still-rdoq --no-cfl
# compare /tmp/cc.rec.y4m vs /tmp/ccr.rec.y4m per plane
```
Observed: with `--no-cfl` (chroma prediction independent of luma) the RDOQ flag
changes 2,827 luma samples but **0** U and **0** V samples. RDOQ is real on the
non-CCTX path: on a flat-luma/noisy-chroma fixture, `--still-rdoq` alone
changes the stream at QP 12/24/40/60, while `--still-cctx --still-rdoq` is
byte-identical to `--still-cctx` at every QP.

Expected: either RDOQ is applied to the CCTX chroma quantisation or the
combination is rejected/documented. `still_cctx_gate.sh` says CCTX "composes
with RDOQ"; its case only compares decoder-exactness, so it cannot see the
flag being dropped.

Likely cause: `compute_chroma_cctx()` quantises the rotated U/V pair with
`cctx_enumerate_scalar()` and scalar `quant_step_u/v`
(`still_encoder.cpp:2311-2316`); it never invokes the `av2enc::rdoq` path that
`compute_tx()` uses for non-CCTX chroma (`still_encoder.cpp:1749-1790`).

## Dedup notes

- `refactor/bugs.csv` has no transform/tx-partition/IST/TCQ rows; `git log --all
  --grep` for `smooth`, `tx partition`, `edge filter` shows only merges.
- ADV-01 is distinct from `intra-modes.md` SCOUT-INTRA-02: that one needs
  `--still-ext-partitions`/`--still-sub8` thin leaves and is unaffected by
  `AV2ENC_STILL_NO_EDGE`; this one needs only `--still-tx-partition`, first
  diverges in normal 16x16/32x32 transform parts, and disappears with NO_EDGE.
- ADV-02 is distinct from `intra-tools.md`/`partition.md` MRL rows (MRL alone
  is exact in this repro); the trigger is adding IST.
- ADV-04 already reported as SCOUT-PARTITION-08 (`partition.md`); independently
  reproduced on `noise_64x64`/`mix_128x96` at QP16 (missing CLI rejection and
  missing `BLOCK_4X4` skip in `choose_luma_tx_partition`).
- ADV-05 already noted in SCOUT-CHROMA-01 (`formats-chroma.md`) and
  SCOUT-FORMATS-HBD-01; the other `*-scale` guards exist, `--still-tcq-scale`
  is the one missing.
