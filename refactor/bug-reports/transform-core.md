# transform-core bug scout

Lane: transform variants and TX_TYPE (intra sets, long-side sets, inter TX sets,
`--still-tx-type`, TX-type search). Repo read-only; probes built from the real
sources into `/tmp` (`c++ -std=c++20 -I src ...`, no repo edits).

| id | severity | class | title | repro | evidence | files | status |
|---|---|---|---|---|---|---|---|
| SCOUT-TRANSFORM-01 | med | confirmed | `TX_Type_In_Set_Inter[TX_SET_INTER_2]` allows 8 types, spec allows 12 | probe below | `in_set(8..11)=0`; spec row = 12 ones | `src/codec/transform/inter_residual/inter_tx_type.cpp:54-64` | new |
| SCOUT-TRANSFORM-02 | med | confirmed | TX_TYPE 7/8 enum names swapped vs spec; inter `Inv_Set1/2` entries numerically swapped, workaround drops both types | patched probe below | type 7→(1,7), type 8→(1,6); spec Inv[15]=ADST_FLIPADST(7), Inv[14]=FLIPADST_ADST(8) | `src/codec/transform/variants/tx_variants.h:61-64`, `tx_variants.cpp:20-45`, `inter_residual/inter_tx_type.cpp:29-43,103-105` | new |
| SCOUT-TRANSFORM-03 | med | confirmed | `enable_inter_ddt=0` filter removes 7 spec-legal identity/1-D inter types; 32x32 DCT_IDTX set loses IDTX | probe below | 32x32 ddt=0 count=1 (spec: IDTX+DCT); 8x8 ddt=0 count=7 (spec set: 16) | `inter_tx_type.cpp:228-241`, callers `inter_coeff_syntax.cpp:264-267`, `inter_residual.cpp:185-190` | new |
| SCOUT-TRANSFORM-04 | low | confirmed | `kMaxInterTxCandidates=10` silently truncates INTER_1 candidate list when DDT is on (14 legal) | probe below | 8x8 ddt=1 count=10; drops FLIPADST_DCT, DCT_FLIPADST, ADST_ADST, FLIPADST_FLIPADST | `inter_tx_type.cpp:271-299,286,329`, `inter_tx_type.h:74` | new |
| SCOUT-TRANSFORM-05 | low | doc | intra TX_TYPE search scores candidates with the scalar quantiser even when TCQ/QM/DC-AC-delta-Q changes the real reconstruction | `--still-tx-type --still-tcq` trace | search path uses `txv::inverse_transform`; committed path uses `dequant_inverse_qindex` | `src/pipeline/still/still_encoder.cpp:1631-1633,1510-1521`, `src/codec/transform/tx_type/tx_type.cpp:379-381` | new |

## Details

### SCOUT-TRANSFORM-01 — INTER_2 membership table drops four legal types
Repro:
```sh
cd /Users/arin/work/av2enc
cat > /tmp/p1.cpp <<'CPP'
#include "codec/transform/inter_residual/inter_tx_type.h"
#include <cstdio>
using namespace av2enc::video::residual; using av2enc::txv::TxType;
int main(){ for(int t=7;t<=11;++t)
  printf("INTER_2 type %d in_set=%d\n", t,
         inter_tx_type_in_set(InterTxSet::kInter2,(TxType)t)); }
CPP
c++ -std=c++20 -O0 -I src /tmp/p1.cpp src/codec/transform/inter_residual/inter_tx_type.cpp \
    src/codec/transform/variants/tx_variants.cpp -o /tmp/p1 && /tmp/p1
```
Observed:
```
INTER_2 type 7 in_set=1
INTER_2 type 8 in_set=0   # ADST_FLIPADST (=8 in spec constants)
INTER_2 type 9 in_set=0   # IDTX
INTER_2 type 10 in_set=0  # V_DCT
INTER_2 type 11 in_set=0  # H_DCT
```
Expected: spec §5.20.7.29 (PDF p233) `Tx_Type_In_Set_Inter` row `TX_SET_INTER_2` is
`{1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0}` — types 0..11 are in set.
Likely cause: `kInSetInter[6]` was written `{1×8,0×8}` instead of `{1×12,0×4}`.
Impact: `inter_tx_symbol()` refuses four types the decoder accepts for 16x16 inter
TX; `inter_chroma_tx_type()` aliases them to DCT while §5.20.7.29
`compute_tx_type()` would return them when `enable_chroma_dctonly=0`. Component
only today: production video writes DCT_DCT and `enable_chroma_dctonly=1`.

### SCOUT-TRANSFORM-02 — TX_TYPE enum 7/8 names are swapped; inter tables inherit it
Spec constants (PDF p31) and §7.15.4.1 (p527): 7 = ADST_FLIPADST = `{FDST,ADST}`,
8 = FLIPADST_ADST = `{ADST,FDST}`. Repo names them `kFlipadstAdst = 7`,
`kAdstFlipadst = 8` (`tx_variants.h:63-64`) and `kTxTypeNames[7]="FLIPADST_ADST"`,
`[8]="ADST_FLIPADST"` (`tx_variants.cpp:39-45`); `tx_type_test.cpp:344-353` pins
the wrong names. The numeric `kTransform1dType` rows are correct (verified: intra
streams selecting type 7 and 8 decode sample-exactly against dav2d), so only the
identifiers/names are wrong.
Consequence: `kInterInvSet1[14..15]` = `{8,7}` and `kInterInvSet2[10..11]` = `{8,7}`
but the spec tables are `{7,8}`. `txv_1d_mapping_matches_spec()` then removes both
types from every inter candidate list and `write_inter_tx_type()` cannot code them.
Repro: copy the source to /tmp, comment out the two `txv_1d_mapping_matches_spec`
guards, and call `inter_tx_symbol(8x8,0,ddt=1,TxType(7/8))`:
```
INTER_1 type=7 -> symbol=1 offset=7   (InvSet1[15] = FLIPADST_ADST(8))
INTER_1 type=8 -> symbol=1 offset=6   (InvSet1[14] = ADST_FLIPADST(7))
```
Expected: 7 -> (1,6), 8 -> (1,7). The `inter_residual_test.cpp:325-345`
workaround ("swapped 1D types") is masking this table/naming bug; fixing the enum
and the two table entries re-enables two legal inter types.

### SCOUT-TRANSFORM-03 — `enable_inter_ddt=0` removes identity/1-D inter types
`inter_tx_type_is_ddt()` returns true for IDTX, V/H_DCT, V/H_ADST, V/H_FLIPADST,
and `inter_tx_candidates`/`inter_tx_symbol` drop them when `enable_inter_ddt` is
false. §7.15.4.1 `get_transform_1d_type()` only substitutes DDTX/FDDT for
`ADST`/`FDST` when `useDdt`; identity rows are unaffected, and the sequence bit
does not change `Tx_Type_In_Set_Inter`. Repro (same probe pattern as 01):
```
TX_32X32 ddt=0 set=TX_SET_DCT_IDTX count=1: DCT_DCT      # spec set: IDTX, DCT_DCT
TX_8X8   ddt=0 set=TX_SET_INTER_1 count=7                # spec set: 16 types
32x32 symbol(IDTX,ddt=0) fails
```
Impact: `inter_residual` with default `enable_inter_ddt=false` (its own default
and the only value any production/gate path uses) never searches IDTX/V/H_DCT or
the one-axis ADST/FDST types; the `--all-tx-types` gate therefore does not cover
them. `write_inter_tx_type()`/`tx_type_symbol_bits()` hardcode `false`
(`inter_coeff_syntax.cpp:264-267`, `inter_residual.cpp:185-190`), so a decision
made with `enable_inter_ddt=true` can select a candidate the writer then rejects.

### SCOUT-TRANSFORM-04 — candidate array too small for INTER_1 with DDT on
`kMaxInterTxCandidates = 10`, but INTER_1 has 16 symbol positions (14 codable even
with types 7/8 withheld). With `enable_inter_ddt=true` the enumeration stops at
10: `TX_8X8 ddt=1 count=10: IDTX V_DCT H_DCT V_ADST H_ADST V_FLIPADST H_FLIPADST
DCT_DCT ADST_DCT DCT_ADST`. Symbol order means FLIPADST_DCT, DCT_FLIPADST,
ADST_ADST and FLIPADST_FLIPADST are never evaluated. Truncation is silent (no
error), so an RDO caller cannot tell the set is incomplete.

### SCOUT-TRANSFORM-05 — TX_TYPE search cost uses a different quantiser than the encode
`compute_tx()` chooses the type with `evaluate_intra_tx_type()`, which quantises
with `p.quant_factor` and inverts with `txv::inverse_transform` (no TCQ extra
`dq_denom`, no QM `q2`, no delta-Q). The committed path re-quantises and
reconstructs with `dequant_inverse_qindex()` (TCQ shift at
`still_encoder.cpp:1518-1520`, QM/`q2` and delta-Q below). With
`--still-tx-type --still-tcq` (or `--still-qm`, `--still-dc-ac-delta-q`) every
candidate's reported SSE/cost is measured on a different reconstruction than the
one that is finally written; the winner is re-quantised so the stream stays
correct, but the search is not minimising the real cost. Quality/RDO gap, no
desync (the `eob<=1` DCT fallback is re-run after the real quantiser).
