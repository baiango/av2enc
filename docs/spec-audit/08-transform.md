# AV2 v1.0.0 §7.14–§7.15 — Dequantization and transform audit

Scope: **§7.14 Reconstruction and dequantization** and **§7.15 Inverse
transform process**, spec pages **515–528**. Normative text extracted from
`spec/AV2_Spec_v1.0.0.pdf` with pypdf (PDF page index = spec page − 1).
Signalling rules from §5.4.8 (p63), §5.20.7.29 (p232) and §5.20.8.2
(pp237–239) are cited where they gate reachability, because §7.14/§7.15
processes that cannot be signalled are not encoder capability.

This is a **code audit**, not a restatement of component READMEs. Where a
README and code disagree, the code is reported. All component tests listed
below were executed for this audit; the ad-hoc decoder comparisons are marked
as such because they are not committed gates.

## Status vocabulary

| Status | Meaning |
|---|---|
| **WIRED** | A shipped encoder path emits and reconstructs this process, and a decoder gate (or a bit-exact per-sample comparison against `dav2d`) exists. |
| **PARTIAL** | Implemented and reachable, but only through a constrained/gate-only writer, only a subset of signalled forms is exercised, or the shipped CLI cannot reach it while the library API can. |
| **COMPONENT** | Complete reference implementation with unit/parity tests, but **no encoder bitstream path uses it** (header bit hard-coded 0, or no caller). |
| **ABSENT** | No implementation, or the syntax cannot carry it at all. |

### Encoder paths that matter here

| Path | What it is | Transform/quant relevance |
|---|---|---|
| **M0** (`src/encoder/m0_still.cpp`, `src/cuda/transform.cu`) | default CLI (`av2enc file.y4m -o out.ivf`) | DCT only, `TX_MODE_LARGEST`, `q = 528` only, 8-bit. Leaf sizes 64×64, 32×32, 32×16, 16×32, 32×8 plus chroma 32×32, 16×16, 16×8, 8×16, 16×4. |
| **M1 CPU** (`src/encoder/m1/m1_still.cpp`, `src/encoder/tx_variants/`) | `--m1` | all 16 TX_TYPE values implemented; intra luma selects among the 9 non-identity types (opt-in `--m1-tx-type`), all partition-reachable sizes, QP 1..255 through `qlookup`, 8-bit. This is the main transform path audited. |
| **M1 GPU** (`src/cuda/m1/tx/`, `src/cuda/tx_variants/`) | `--m1-gpu` | same `TxParams`/kernels as M1 CPU, but `base_q_idx` is pinned to 90 (`src/cuda/m1/m1_gpu.cu:1022`). |
| **M2** (`src/encoder/m2/`) | standalone gate `src/encoder/m2/build_and_test.sh`; not linked into the CLI | `TX_MODE_LARGEST`, DCT_DCT only in the syntax writer, 8-bit. |
| **Lossless** (`src/encoder/lossless/`) | gate-only, not linked into the CLI (`README.md`: deliberately not wired) | Q=0, WHT / lossless IDTX / DPCM, 8×8 and 4×4 only. |
| **HBD** (`src/cuda/m1/hbd/`) | 10-bit component, no stream path (`gate_10bit_still.sh` says the headers cannot produce a 10-bit stream yet) | 10-bit dequant/inverse clips, all 25 sizes, all 16 types. |

### Gate inventory (short names used in Evidence)

| Name | What it proves / result |
|---|---|
| **G-TXV** | `src/encoder/tx_variants/build_and_test.sh` → **21,497,851 checks, 0 failures**. All 25 sizes, 16 TX types, shifts, `dq_denom`, odd-log2 pre-scale, 64→32 duplication, IDTX scales, length-32 ADST/FDST alias. |
| **G-TXTYPE** | `src/encoder/tx_type/build_and_test.sh` → **382,552 checks, 0 failures**. TX sets, symbols, all 25 sizes, all 16 types incl. 32-length aliasing. |
| **G-IST** | `src/encoder/tx_secondary/build_and_test.sh` → **158,153,173 checks, 0 failures**. 4×4/8×8 kernels, scans, kernel mapping, syntax legality, forward projection. |
| **G-QM** | `src/encoder/qm/build_test.sh` → **258,446 checks, 0 failures**. QM tables, `q2`, `dqDenom`, user-QM serialization. |
| **G-QM-GATE** | `src/encoder/qm/qm_gate.sh` → `qm gate: OK`. `dav2d` accepts `using_qmatrix=1` with default matrices; `avmdec` accepts user-QM OBUs. Streams are **bit-spliced**, not produced by the encoder quantiser. |
| **G-TCQ** | `src/encoder/tcq/build_and_test.sh` → PASS. Trellis, TCQ level mapping, dequant shift helper, CCTX search; README notes no decoder gate because core headers hard-code the flags to 0. |
| **G-DDT-U** | `src/encoder/m2/ddt/build_and_test.sh` → **7,764 checks, 0 failures**. |
| **G-DDT-GATE** | `build_and_test.sh --gate` → **DDT GATE PASS**: 4 variants (DDTX column/row, FDDT column/row), 62/45/62/45 DDT directions, `dav2d` recon exact, splice byte-identical, flag-0 negative control. Uses **build-directory patched copies** of `m2_syntax.cpp`, `inter_residual.cpp`, `m2_headers.cpp`. |
| **G-LOSSLESS** | `src/encoder/lossless/build_and_test.sh` → `22 decoder checks passed` on `dav2d` **and** `avmdec` (WHT, lossless IDTX/FSC, DPCM). |
| **G-HBD** | `src/cuda/m1/hbd/build_hbd_test.sh` → **19,312 checks, 0 failures** (CPU 10-bit reference; CUDA half skipped, no nvcc on this host). |
| **G-M2** | `src/encoder/m2/build_and_test.sh` → **M2 GATE PASS**, 2-frame + 8-frame 720p recon exact (DCT_DCT only). |
| **G-M1-FEATURE** | `tools/m1_verify/feature_gate.sh`; the `--m1 --m1-tx-type` 64×64 golden MD5 `15681508a144eeea88bb8223185d75e9`. This audit reproduced that MD5 from `build-m1/av2enc`. |
| **A-M1-TX** | *Ad-hoc audit run, not a committed gate.* `m1_dump` + `tools/m1_verify/verify.py` on `dav2d`: `--m1 --m1-tx-type --qp 20/30`, depths 0–4, sizes 256×256, 64×16, 16×64, 64×8, 8×64, 64×32, 32×8, 8×32, 16×32, 16×8, 8×16, 8×8, 16×16 → **every plane exact, max\|diff\|=0**. Trace showed `tx=8x32, 32x8, 16x32, 32x16, 64x32, 64x64, 16x64, 64x16, 8x64, 64x8`. |

---

## 1. §7.14.2 — Dequantization functions

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| Q1 — `qlookup(q)`: `Ac_Qlookup[q]` for q<25, else `Ac_Qlookup[((q-1)%24)+1] << ((q-1)/24)`; 25-entry table | §7.14.2 / p515 | **WIRED** | `src/encoder/rc/rc_tables.h:17-29` precomputed 0..255 table; used by M1 at `src/encoder/m1/m1_still.cpp:196`; formula independently implemented at `src/encoder/hbd/hbd_depth.cpp:67-73`; `src/encoder/quant/dcdeltaq/quant.cpp:38-41` | M0 pins q=528 and never uses the table. Table values were spot-checked against the recurrence (q=25 → 80); no exhaustive PDF-table diff in this audit. |
| Q2 — `get_q(qindex,delta)`: lossless `qindex==0 && delta<=0 → 64`; else `qlookup(Clip3(1,MaxQ,qindex+delta))` | §7.14.2 / p515-516 | **PARTIAL** | `dcdeltaq` full form at `quant.cpp:43-47`, tested in `dcdeltaq_test.cpp:96-130`; M1 only ever calls the delta=0 case (`m1_still.cpp:196,758-787`) | Non-zero `DeltaQ*`/`Base*DeltaQ` deltas are never used by M1/M2; the wired path is the degenerate delta=0 case. No lossless q=0 path in M1 (rejected in the CLI). |
| Q3 — `get_qindex(ignoreDeltaQ, segmentId)`: `SEG_LVL_ALT_Q`, `base_q_idx`, `CurrentQIndex` under `delta_q_present`, clip `[0,MaxQ]` | §7.14.2 / p516 | **COMPONENT** | `dcdeltaq::get_qindex` `quant.cpp:63-75`; ALT_Q/`CurrentQIndex`/lossless-segment cases pinned in `dcdeltaq_test.cpp:132-158,672`; lossless variant `lossless.cpp:258-265` | Not wired: M1 writes `segmentation_enabled=0` and `delta_q_present=0` (`src/encoder/m1/m1_headers.cpp:176-178`); M2 the same (`m2_headers.cpp:283-285`). |
| Q4 — `get_dc_quant(plane)` / `get_ac_quant(plane)` with separate `DeltaQYDc`, `BaseYDcDeltaQ`, `DeltaQU/V Dc/Ac`, `BaseUV*DeltaQ` | §7.14.2 / p516 | **PARTIAL** | M1 collapses DC and AC to one `q = qlookup(base_q_idx)`: `m1_still.cpp:196,758-787`; complete derivation in `dcdeltaq::derive_quant` `quant.cpp:77-90` (DC/AC per plane) | The wired stream is conformant only because `equal_ac_dc_q=1` and every delta is zero (`m1_headers.cpp:139-142`). It cannot exercise a distinct DC step, a luma DC delta, or separate UV DC/AC. |
| Q5 — `MaxQ` = 255 (8-bit) / 303 (10-bit); `base_q_idx` width | §7.14.2 p515 / §6.4.1 | **PARTIAL** | 8-bit: `m1_headers.cpp:175` writes `f(8)` and `m1_still.cpp:196` indexes 0..255; 10-bit: `hbd::depth::max_q()` and `base_q_idx_bits()` `src/encoder/hbd/hbd_depth.h`/`.cpp:90-93`, tested by **G-HBD** | 10-bit has no stream path: headers hard-code 8-bit (`src/cuda/m1/hbd/gate_10bit_still.sh:6-7`). |

---

## 2. §7.14.3 — Reconstruct process

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| R1 — Reconstruct order: dequant → secondary transform when `plane==0 && sec_tx_type!=0` → 2D inverse into `Residual` → add to `CurrFrame` | §7.14.3 / p516-517 | **PARTIAL** | M1 order is dequant → 2D inverse → add (`m1_still.cpp:765-787,978`); the IST step exists only in `txsec::apply_inverse_secondary` (`tx_secondary.cpp:324-354`) and has no caller | No M1/M2 path invokes the secondary transform, so step 1 of the ordered list is never executed. |
| R2 — Residual add: `CurrFrame = Clip1(CurrFrame + Residual)`; residual representable in `1+BitDepth` bits when `Lossless` | §7.14.3 / p517 | **WIRED** | `m1_still.cpp:984-1002` (`clip1(p + r)`); 8-bit `clip1`; M0 equivalent; **A-M1-TX** exact recon | 8-bit only in the wired path. The conformance requirement is not asserted for lossless (lossless module is separate). |
| R3 — `Dequant` / `Residual` clip ranges are `BitDepth`-dependent: `±(1<<(7+BitDepth))`, row `±(1<<(BitDepth+7))`, column `±(1<<BitDepth)` | §7.14.3/§7.14.4/§7.15.2.1 | **PARTIAL** | 8-bit: `tx_variants.cpp:258-271` and `276-360` (`±(1<<15)`, `[-256,255]`); 10-bit: `src/cuda/m1/hbd/hbd_cpu.cpp:154-230` + `hbd_depth.h:45-70`; tested by **G-HBD** | 10-bit only component. `qm_core.h` omits the §7.14.4 dequant clip entirely (see D10). |

---

## 3. §7.14.4 — Dequantization process

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| D1 — `shift = (pels>256) + (pels>1024)` with `pels = Tx_Width*Tx_Height`, `dqDenom = 1<<shift` | §7.14.4 / p518 | **WIRED** | `tx_variants.cpp:158-163` (coded pels, not adjusted); formula test all 25 sizes in `tx_variants_test.cpp:117-120`; M0 `m0_still.cpp:1719-1721`; 10-bit twin `hbd_cpu.cpp:154-165` | None for 8-bit. |
| D2 — `useFsc = enable_fsc && PlaneTxType==IDTX && plane==0 && (fsc_mode \|\| is_inter)` | §7.14.4 / p518 | **PARTIAL** | `use_fsc` exists as an input in `qm::core::CoeffParams` (`qm_core.h:159`) and `dcdeltaq::QuantBlock` (`dcdeltaq.h:261`); real FSC coding lives only in the gate-only lossless module (`lossless.cpp`, **G-LOSSLESS**) | No lossy FSC path: M1/M2 write `enable_fsc=0` (`m1_headers.cpp:130`, `m2_headers.cpp:190`). |
| D3 — TCQ extra shift: `allow_tcq && plane==0 && !Lossless && get_tx_class(PlaneTxType)==TX_CLASS_2D && !useFsc → shift+=1` | §7.14.4 / p518 | **COMPONENT** | Correct helper `tcq::dequant_shift_extra` `src/encoder/tcq/tcq.h:101-114`; correct in `dcdeltaq::extra_tcq_shift` `quant.cpp:24-35` (IDTX counts as 2D); tested in **G-TCQ**/`dcdeltaq_test` | Headers hard-code `enable_tcq=0` (`m1_headers.cpp:137`, `m2_headers.cpp:198`). **`qm_core.h` has the wrong predicate**: `dequant_denom` uses `tx_uses_qm(tx_type) = tx_type<9` (`qm_core.h:231-238,108`), so IDTX (type 9, `TX_CLASS_2D`) silently loses the TCQ shift. `qm_test.cpp:192-200,404-450` pins that wrong behaviour. |
| D4 — `segLvl`: `tw>8 \|\| th>8` → `qm_y[0]`/`qm_u[0]`/`qm_v[0]`; else `SegQMLevel[plane][segment_id]` | §7.14.4 / p518 | **COMPONENT** | `qm::resolve_seg_level` `src/encoder/qm/qm.cpp:212-228`; shared device twin `qm_core.h:171-181`; tested **G-QM** | No M1/M2 path sets `using_qmatrix=1` (`m1_headers.cpp:176`, `m2_headers.cpp:284`). Only the gate splice can. |
| D5 — `useQm = using_qmatrix==1 && PlaneTxType<IDTX && segLvl<NUM_CUSTOM_QMS` | §7.14.4 / p518 | **COMPONENT** | `qm_core.h:108,183-186`; `use_qm` uses `tx_type<9` and `segLvl<15`; **G-QM** | Correct predicate, but no encoder path uses it. `qm_gate.sh` splices only identity user matrices, so encoder reconstruction is unaffected (see D7). |
| D6 — `useUserQm = useQm && tw<=8 && th<=8 && QmDataPresent[segLvl]`; user matrix indexing `[2]` for `tw<th`, `[1]` for `tw>th`, `[0]` with `8/th`,`8/tw` for square | §7.14.4 / p518 | **COMPONENT** | `qm_core.h:188-218`; table extraction/serialization in `qm.cpp`; **G-QM** 258k checks; **G-QM-GATE** level 0/5/14 identity | No encoder reconstruction uses `UserQm`; the gate only proves decodability, not encoder-side q2. |
| D7 — `q2 = Round2(q*m,5)` when QM active | §7.14.4 / p518 | **COMPONENT** | `qm_core.h:220-226`; `dcdeltaq::build_q2_matrix` `quant.cpp:150-177`; tested **G-QM** | Same as D4–D6: no stream uses a non-identity QM. A default-matrix splice changes decoder pixels (`qm_gate.sh` output: `default QM differs`) while encoder recon stays on `q`; so recon is exact only for the identity case. |
| D8 — `q2 = q` when QM is off | §7.14.4 / p518 | **WIRED** | M1 `m1_still.cpp:765-787`; `tx_variants.cpp:258-271`; M0 `m0_still.cpp:1729-1731` (quant factor uses `kQuantStep`) and `:2004-2007`; **G-TXV**, **A-M1-TX** | None 8-bit. |
| D9 — `dqHigh = Abs(qc)*q2`; mask `0xFFFFFF`; `dq = Round2(., QUANT_TABLE_BITS=3)`; `dq2 = sign*(dq/dqDenom)` | §7.14.4 / p518-519 | **WIRED** | `tx_variants.cpp:265-270`; M1 alternative for `q != 528` at `m1_still.cpp:771-783`; `hbd_cpu.cpp:154-165`; all-25-size formula test `tx_variants_test.cpp:304-340`; M0 `m0_still.cpp:2004-2007` | `tx_variants` computes `abs_qc * 528` in `int32`; a level above ~4.06 M would overflow (no such level observed). |
| D10 — `Dequant = Clip3(-(1<<(7+BitDepth)), (1<<(7+BitDepth))-1, dq2)` | §7.14.4 / p519 | **PARTIAL** | 8-bit: `tx_variants.cpp:269-271`, `m1_still.cpp:783-784`; 10-bit: `hbd_cpu.cpp:163` via `clip_dequant`; tested **G-HBD** | 10-bit is component-only. `qm::core::dequant_coeff` (`qm_core.h:242-255`) returns `sign*(dq/denom)` with **no Clip3**, and `qm_test.cpp:181-190` omits it from its reference too. `dcdeltaq::dequantize_block` does clip (`quant.cpp:240-245`) but at fixed 8-bit bounds only. |
| S1 — `save_dequant`: `SaveDequant[plane][i][j] = Dequant[i][j]` for `Min(32,W)×Min(32,H)` | §7.14.5 / p519 | **COMPONENT** | No explicit `SaveDequant` buffer in M1/M2; CCTX pair is modelled by `tcq::CctxPair`; `dcdeltaq` keeps per-plane dequant arrays | The process is trivial, but the only consumer (CCTX) is component-only (S2). |
| S2 — `get_dequant`: for `CCTX_NONE` copy; else rotate U/V with `Cctx_Mtx[angle]`, `Round2Signed(.,8)`, clip `±(1<<(BitDepth+7))` | §7.14.6 / p519-520 | **COMPONENT** | `tcq::cctx_dequant_component` `src/encoder/tcq/cctx.cpp:64-84`; `cctx_dequant_pair` `:86-91`; matrix `cctx.h:31-50`; search at `cctx.cpp:126-217`; **G-TCQ** | `enable_cctx=0` in every writer (`m1_headers.cpp:136`, `m2_headers.cpp:197`). No stream signals `cctx_type`. |
| S3 — `Cctx_Mtx[6][2] = {{181,181},{222,128},{128,222},{181,-181},{222,-128},{128,-222}}` and the six angles | §7.14.6 / p520 | **COMPONENT** | `cctx.h:31-50`; forward inverse rotation with `256/(c²+s²)` in `cctx.cpp:105-124`; tested **G-TCQ** (`test_cctx` at `tcq_test.cpp:473-520`) | No decoder-gated CCTX stream. |

---

## 4. §7.15.2.1 — 1D inverse transforms

Kernel tables are `src/encoder/tx_variants/tx_tables.h` (DCT4/8/16/32, ADST4/8/16, FDST4/8/16; §9.6) and `src/encoder/m2/ddt/ddt_tables.h` (DDTX8/16). The 16-entry `Transform_1d_Type` mapping is `tx_variants.cpp:20-35`.

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| T1 — DCT, `sz==4` | §7.15.2.1 / p520 | **WIRED** | `tx_variants.cpp:146-172`; `kDct4` `tx_tables.h`; **G-TXV**; used as chroma of every 8×8 luma block; **A-M1-TX** 8×8→4×4 chroma exact | Luma 4×4 is not selectable from the CLI (see X1). |
| T2 — DCT, `sz==8` | p520-521 | **WIRED** | same; `kDct8`; 25-size sweep | — |
| T3 — DCT, `sz==16` | p521 | **WIRED** | same; `kDct16`; observed `tx=16x16/16x32/32x16` in **A-M1-TX** | — |
| T4 — DCT, `sz==32` | p521 | **WIRED** | same; `kDct32`; observed 64→32 adjusted blocks; **A-M1-TX** exact | — |
| T5 — ADST, `sz==4` | p520 | **PARTIAL** | `kAdst4` and `tx_kernel` `tx_variants.cpp:153-157`; **G-TXV** | Cannot be signalled by the shipped CLI: 4-sample transforms occur only on chroma (8×8 luma → 4×4 chroma) and `enable_chroma_dctonly=1` forces chroma DCT_DCT (`m1_still.cpp:1815`, `m2_headers.cpp:194`). Reachable only via the M1 library API with `min_block_size=4` (not a CLI option). |
| T6 — ADST, `sz==8` | p520-521 | **WIRED** | `kAdst8`; intra sets via `tx_type.cpp`+`tx_type_tables.h`; M1 selects at `m1_still.cpp:810-875`; **G-TXTYPE**, **A-M1-TX** | Identity combinations excluded (X4). |
| T7 — ADST, `sz==16` | p521 | **WIRED** | `kAdst16`; same path | — |
| T8 — FDST (FLIPADST), `sz==4` | p520 | **PARTIAL** | `kFdst4`; **G-TXV** | Same 4-sample/chroma-only limitation as T5. |
| T9 — FDST, `sz==8` | p520-521 | **WIRED** | `kFdst8`; FLIPADST_* intra symbols; **G-TXTYPE** | — |
| T10 — FDST, `sz==16` | p521 | **WIRED** | `kFdst16`; same | — |
| T11 — DDTX, `sz==8` | p520-521 | **PARTIAL** | `ddt::inverse_1d` `ddt_transform.cpp:194-216`; `effective_kernel` `:172-192`; **G-DDT-U**, **G-DDT-GATE** (62 DDTX directions) | Only via the **patched build-dir gate**; production `m2_syntax.cpp` writes DCT_DCT for every inter block and `enable_inter_ddt=0`. See M8/X5/X8. |
| T12 — DDTX, `sz==16` | p521 | **PARTIAL** | same; DDTX16 table | Same. |
| T13 — FDDT, `sz==8` | p520-521 | **PARTIAL** | `kernel_is_flipped` `ddt_transform.cpp:130-133`; **G-DDT-GATE** (45 FDDT directions in each row case) | Same. |
| T14 — FDDT, `sz==16` | p521 | **PARTIAL** | same | Same. |
| T15 — `sz==32` branch ignores `txType1D` and always uses `Dct_Kernel32`; so ADST/FDST/DDTX/FDDT at length 32 alias DCT | §7.15.2.1 / p521 | **WIRED** | `tx_kernel` `tx_variants.cpp:153-171` returns `kDct32` for ADST/FDST at n=32; `ddt_transform.cpp:177-188`; tests `tx_variants_test.cpp:277-302`, `tx_type_test.cpp:803-900` (G-TXV/G-TXTYPE) | — |
| T16 — Output clip `±(1<<(BitDepth + (colTx?0:7)))`; `Round2(s, shift)` | §7.15.2.1 / p521 | **WIRED** | `tx_variants.cpp:283-335` (row `±(1<<15)`, col `[-256,255]`); 10-bit `hbd_cpu.cpp:167-230`; `ddt_transform.cpp:200-216`; **A-M1-TX** exact | 10-bit component only (R3). |

---

## 5. §7.15.2.2 / §7.15.2.3 — WHT and identity

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| W1 — Inverse WHT (4-point, reversible lifting, `shift=3` row / `0` column) for lossless `DCT_DCT` | §7.15.2.2 / p521 | **PARTIAL** | `lossless.cpp:122-139` (`inverse_wht4`), `182-200` (`inverse_lossless_4x4`); forward exact inverse `:143-160`; **G-LOSSLESS** 22 decoder checks | Gate-only lossless module; not linked into the CLI, and CLI rejects `--qp 0` (`main.cpp:535`). 8-bit only. |
| I1 — Inverse identity: `Round2(src*scale, shift)` with scale 128/181/256/362 from `get_identity_scale(log2Sz)` and `colTx` clip | §7.15.2.3 / p521-522; scale table p528 | **COMPONENT** | `tx_variants.cpp:286-335` (null kernel path) + `tx_tables.h:196-201`; `identity_scale` goldens `tx_variants_test.cpp:78-82`; **G-TXV** | No lossy stream signals IDTX: intra `enable_idtx_intra=0` and candidates filtered (`m1_still.cpp:831-851,1816`); inter syntax writes DCT_DCT (`m2_syntax.cpp:1049-1084`). Lossless IDTX is a different process (M3), not §7.15.2.3. |

---

## 6. §7.15.3 — Secondary transform (IST)

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| IST1 — `w=Min(32,Tx_Width)`, `h=Min(32,Tx_Height)`, `bwl=Min(5,Tx_Width_Log2)`, `large=w>=8&&h>=8`, `n=8/20/32` (`TX_8X8` or `ADST_ADST` → reduced) | §7.15.3 / p522 | **COMPONENT** | `txsec::secondary_size_info` `tx_secondary.cpp:59-91`; constants `tx_secondary_tables.h:10-19`; **G-IST** | No caller. Headers hard-code `enable_intra_ist=0`, `enable_inter_ist=0` (`m1_headers.cpp:132-134`, `m2_headers.cpp:192-193`). |
| IST2 — Kernel selection: inter→0; intra `ADST_ADST && W>=8 && H>=8` → `Inv_Most_Probable_Stx_Mapping_Adst`, else `Inv_Most_Probable_Stx_Mapping`; `ADST_ADST → kernel+=7` | §7.15.3 / p522-523 | **COMPONENT** | `kernel_for_syntax` `tx_secondary.cpp:142-173`; `map_y_mode` `:107-140` (wide-angle + MRL delta); **G-IST** 158M checks | Same; the `most_probable_stx_set` symbol is never written. |
| IST3 — `transpose = H_PRED \|\| D157_PRED \|\| D67_PRED \|\| SMOOTH_H_PRED` | p523 | **COMPONENT** | `transpose_for_mode` `tx_secondary.cpp:175-179`; test `tx_secondary_test.cpp` | Same. |
| IST4 — Scan-in `get_scan(txSz,TX_CLASS_2D)` over adjusted `w,h`; scan-out `Stx_Scan_Order_4x4` / `Stx_Scan_Order_8x8` through `Stx_Scan_Map` | p523-524 | **COMPONENT** | `diagonal_scan` `tx_secondary.cpp:28-45`; `kStxScanOrder*`/`kStxScanMap` in `tx_secondary_tables.h`; inverse remap `:273-321`; **G-IST** | Same. |
| IST5 — Syntax applicability/validity: only `plane 0`, `!Lossless`, type `DCT_DCT`/`ADST_ADST`, `YMode!=PAETH`, `eob<=eobLim`, inter `eob>3 && W,H>=16`, intra `eob!=1` | p522 / §5.20.8.2 p239 | **COMPONENT** | `secondary_syntax_applicable` `tx_secondary.cpp:181-210`; `secondary_syntax_valid` `:212-238`; `enumerate_secondary_candidates` `:240-271`; **G-IST** | No syntax writer reads the result; `sec_tx_type` is never emitted. |
| IST6 — Inverse: read `n` scan-in coeffs, `Round2Signed(t,7)`, clip `±(1<<(BD+7))`, write `n2` outputs with scan map/transpose | p523-524 | **COMPONENT** | `inverse_secondary_dequant_kernel` `tx_secondary.cpp:273-321`; `apply_inverse_secondary` `:324-354`; **G-IST** | Not integrated into the reconstruct order (R1). |
| IST7 — Encoder-side forward projection (transpose of the inverse, same shift) | not normative | **COMPONENT** | `forward_secondary_coeffs_kernel` `tx_secondary.cpp:357-410`; **G-IST** (`forward_projection`) | Not used by any RDO or quantiser path. |
| IST8 — Sequence header bits `enable_intra_ist`, `enable_inter_ist`; per-block `sec_tx_type`, `most_probable_stx_set` | §5.4.8 p63 / §5.20.8.2 p239 | **ABSENT** | Writers force both bits to 0 (`m1_headers.cpp:132-134`, `m2_headers.cpp:192-193`); no `sec_tx_type` in `m1_syntax.cpp`/`m2_syntax.cpp`; `txsec::write_secondary_transform_header_bits` (`tx_secondary.cpp:455-460`) has no caller | The tool cannot be enabled from any encoder path. |

---

## 7. §7.15.4 / §7.15.4.1 — 2D inverse and matrix transform

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| M1 — `adjTxSz = Adjusted_Tx_Size[txSz]` for all 25 sizes (64×64/64×32/32×64→32×32; 16×64/64×16→16×32/32×16; 8×64/64×8→8×32/32×8; 4×64/64×4→4×32/32×4) | §7.15.4 / p524 | **WIRED** | `kTxSizes[25]` `tx_tables.h:168-194`; `make_tx_params` `tx_variants.cpp:135-173`; test all 25 sizes `tx_variants_test.cpp:94-140`; 10-bit twin uses same table; **A-M1-TX** observed 64→32, 16×64/64×16, 8×64/64×8 | Correct table; no gap. |
| M2 — `pels = adjW*adjH`, `shift = (pels>256)+(pels>1024)` for the lossy 2D process | §7.15.4 / p525 | **WIRED** | `tx_variants.cpp:273-281`; observed 64×64 (shift 2), 32×32 (1), 16×16/8×32 (0/1) blocks in **A-M1-TX**, all exact | The `Lossless && IDTX → Residual = Dequant >> (3-shift)` variant is not part of this row; it is M3. |
| M3 — Lossless IDTX `Dequant >> (3-shift)` | §7.15.4 / p525 | **PARTIAL** | `lossless.cpp:685-700` `idtx` path copies levels; `inverse_lossless_4x4` for DCT_DCT; **G-LOSSLESS** | Only 4×4 and 8×8 (shift 0), so the shift term is never exercised. No 16×16/32×32/64×64 lossless IDTX. |
| M4 — DPCM cumulative sum (`V_PRED` columns, else rows) when `use_dpcm_*` | §7.15.4 / p525 | **PARTIAL** | `lossless.cpp` `dpcm_cumsum_4x4/8x8` and `forward/inverse` helpers; tests `lossless_test.cpp:186-238`; **G-LOSSLESS** | Only in the gate-only lossless module; M1/M2 never set `use_dpcm`. |
| M5 — Adjusted-size expansion by sample duplication (`w!=w2`, `h!=h2`) | §7.15.4 / p525 | **WIRED** | `tx_variants.cpp:341-359`; `ddt_transform.cpp:303-322`; `hbd_cpu.cpp`; tests `tx_variants_test.cpp:442-476`; **A-M1-TX** exact for 64×64/64×32/32×8/8×32/16×64/64×16/64×8/8×64 | — |
| M6 — Row pass then column pass; `txRowIn`/`txColIn`; residual written from `txColOut` | §7.15.4.1 / p527-528 | **WIRED** | `tx_variants.cpp:295-335`; `ddt_transform.cpp:241-300`; 10-bit `hbd_cpu.cpp:167-230`; **G-TXV**, **A-M1-TX** | — |
| M7 — `Transform_Shift[TX_SIZES_ALL][2]` (25 entries) | §7.15.4.1 / p526 | **WIRED** | `kTxSizes` row/col shifts `tx_tables.h:168-194`; test all 25 (`tx_variants_test.cpp:117-119`); 10-bit and DDT reuse the same `TxParams`; M0 has a hand-written subset (`m0_still.cpp:1281-1305`) | M0's table is only correct for the sizes M0 codes (64×64, 32×32, 32×16/16×32, 32×8, and chroma 32×32/16×16/16×8/8×16/16×4); it is not a general 25-size table. |
| M8 — `get_transform_1d_type(dir,sz)`: `useDdt = enable_inter_ddt && !use_intrabc && is_inter`; rewrite ADST→DDTX / FDST→FDDT when `sz!=4` | §7.15.4.1 / p526 | **PARTIAL** | `ddt::effective_1d` `ddt_transform.cpp:149-159`; `make_plan` `:161-170`; **G-DDT-U**, **G-DDT-GATE** | Only the patched gate build sets `enable_inter_ddt=1`; production headers write 0 (`m1_headers.cpp` has no DDT bit for single-picture; `m2_headers.cpp:195` writes 0). |
| M9 — `Transform_1d_Type[TX_TYPES][2]` 16 entries | §7.15.4.1 / p527 | **WIRED** | `kTransform1dType` `tx_variants.cpp:20-35`; test `tx_variants_test.cpp:141-170`; 10-bit/DDT reuse | Table is correct; identity rows are unreachable (I1/X4). |
| M10 — Rectangular pre-scale `Round2(Dequant*2896,12)` when `Abs(log2W-log2H)` is odd; encoder forward uses `4096/2896` | §7.15.4.1 / p527 | **WIRED** | inverse `tx_variants.cpp:281-287`; forward `:164,242-246`; test `tx_variants_test.cpp:122-124`; observed odd-log2 sizes 32×8/8×32/16×32/32×16/64×32 with **A-M1-TX** exact | — |
| M11 — Lossless branch invokes Inverse WHT for both row and column | §7.15.4.1 / p527-528 | **PARTIAL** | `lossless.cpp:682-700`; **G-LOSSLESS** | Only 4×4 WHT (lossless `TxMode=ONLY_4X4`); gate-only module. |
| M12 — Identity branch (`rowType/colType == IDT`) inside the 2D matrix process | §7.15.4.1 / p527-528 | **COMPONENT** | `tx_variants.cpp:295-335` (null kernel); `identity_scale` `tx_tables.h:196-201`; **G-TXV** | Same reachability gap as I1: no lossy stream signals IDTX. |
| M13 — 4×4, 8×8, 16×16, 32×32 and rectangular sizes through `w=1<<adjLog2W`, `h=1<<adjLog2H` | §7.15.4.1 / p525-526 | **WIRED** | `tx_variants.cpp` generic `w,h` loops; `tx_tables.h:168-194`; **G-TXV** all 25; **A-M1-TX** observed 8×32, 32×8, 16×32, 32×16, 16×16, 16×64, 64×16, 8×64, 64×8, 64×32, 64×64 | Luma 4-sample dimensions need `min_block_size=4` (library only); chroma 4-sample dimensions are reachable and exact (T1). |
| M14 — 64-coded sizes use an adjusted 32×32/32×16/16×32/... transform and duplicate the residual | §7.15.4 / p525 | **WIRED** | `tx_variants.cpp:341-359`; observed `64x64`, `64x32`, `16x64`, `64x16`, `8x64`, `64x8`; **A-M1-TX** all planes exact | `TX_4X64`/`TX_64X4` are not signallable (X1); no other 64-size gap. |

Note on the 16 TX_TYPE values: V_DCT/H_DCT, V_ADST/H_ADST and
V_FLIPADST/H_FLIPADST are not separate kernels. They are the
`Transform_1d_Type` rows that pair `IDT` on one axis with DCT/ADST/FDST on the
other (M9), so they are covered by I1/M12 plus T1–T10. `tx_variants` implements
and tests all 16 rows (**G-TXV**); in the shipped encoder the 9 non-identity
rows are the only ones intra can select, and inter cannot select any of them
(X4/X5).

---

## 8. Signalling gates for the above processes

| Item | §/page | Status | Evidence | Gap |
|---|---|---|---|---|
| X1 — `tx_size` signalling / `TX_MODE_LARGEST`: tx size is the coding-block size; no explicit `tx_size` bits | §5.18.8.1 p135; §5.20.8.2 p237 | **PARTIAL** | M1 writes `tx_mode_select=false` (`m1_headers.cpp:258`, `m1_still.cpp:1817`); M2 same (`m2_headers.cpp:289`); M1 syntax asserts `tx_size == block size` (`m1_syntax.cpp:1399-1409`) | Cannot signal any transform smaller than its coding block, or a different shape. `TX_4X64`/`TX_64X4` have no block size at all and are therefore never signallable. Luma 4-sample blocks need the library-only `min_block_size=4` (CLI hard-codes 8, `src/encoder/m1/m1_still.h:28`). |
| X2 — `tx_mode_select=1` → `TX_MODE_SELECT`, `tx_size` and `tx_split` symbols | §5.18.8.1 / §5.20.8.2 | **ABSENT** | No writer emits `tx_mode_select=1`; `txv::select_tx_partition`/`tx_block_cost` (`src/encoder/tx_variants/tx_select.cpp`) is used only by tests | The encoder cannot signal TX partitioning (NONE/SPLIT/HORZ/VERT) or an explicit `tx_size`. |
| X3 — Intra `transform_type()` sets: INTRA_1 (7 mode-dependent types), INTRA_2, WIDE/HIGH_32 (long flag + 4), WIDE/HIGH_64 (implied long + 4) | §5.20.8.2 pp237-239; `get_tx_set` §5.20.8.3 | **WIRED** | `txtype::intra_tx_candidates`/`intra_tx_set` `tx_type.cpp:99-213`; M1 selection `m1_still.cpp:810-875`; syntax `m1_syntax.cpp:1062-1114`; **G-TXTYPE**, **G-M1-FEATURE**, **A-M1-TX** | Identity entries of the long sets are filtered because `enable_idtx_intra=0` (X4). `reduced_tx_set` is always 0 (X7). |
| X4 — Intra identity types: IDTX, V_DCT, H_DCT, V_ADST, H_ADST, V_FLIPADST, H_FLIPADST | §5.20.8.2 / §5.4.8 | **ABSENT** | `m1_still.cpp:1816` writes `enable_idtx_intra=false`; candidates filtered at `:831-851`; M1 header config default false (`m1_headers.h:69-71`) | No intra block can signal an identity-based TX type, even though §7.15.2.3 kernels exist. |
| X5 — Inter TX types: INTER_1/INTER_2/DCT_IDTX/DCT_IDTX_IDDCT/long inter sets | §5.20.8.2 p238, p240 | **PARTIAL** | Component selection `m2/residual/inter_tx_type.cpp:51-95,251-360`; decision searches them `inter_residual.cpp:524-575`; **but** producer syntax writes DCT_DCT only (`m2_syntax.cpp:1049-1084`); **G-DDT-GATE** uses patched `m2_syntax.cpp` | Production M2 signals DCT_DCT for every codable inter block. All other inter types (incl. IDTX and DDTX/FDDT-derived types) are unsignallable outside the DDT gate build. |
| X6 — Chroma TX type: `compute_tx_type()` returns DCT_DCT when `enable_chroma_dctonly=1` | §5.20.7.29 p232 | **WIRED** | `m1_still.cpp:1815` sets true; `m2_headers.cpp:194` writes true; chroma `compute_tx` never selects a type (`m1_still.cpp:1124-1129`) | Conformant but chroma never exercises ADST/FDST/IDTX; 4-sample chroma blocks are always DCT. |
| X7 — `reduced_tx_set` 0..3 selects INTRA_1/INTRA_2/DCT_IDTX | §5.18.8.1 p135 / §5.20.8.3 | **PARTIAL** | M1/M2 write 0 (`m1_headers.cpp:259`, `m2_headers.cpp:295`); `txtype` supports 1/2/3 (`tx_type.cpp:126-136`); **G-TXTYPE** | No stream uses reduced sets 1/2/3; INTRA_2 and the 2-type reduced set are never exercised. |
| X8 — `enable_inter_ddt` sequence bit (inter ADST/FDST → DDTX/FDDT) | §5.4.8 p63 | **ABSENT** | M2 writes 0 (`m2_headers.cpp:195`); M1 single-picture stream has no bit; `ddt_syntax.cpp` writer exists for non-single-picture only | The production encoder cannot enable DDT. Only **G-DDT-GATE**'s patched build can. |
| X9 — `enable_tcq` (+ `choose_tcq_per_frame`) | §5.4.8 p63; §5.20.7.27 | **ABSENT** | All writers hard-code 0 (`m1_headers.cpp:137`, `m2_headers.cpp:198`, `m0_headers.cpp:244`); `tcq` README lists the blocked call sites | No TCQ stream. |
| X10 — `enable_cctx` + per-block `cctx_type` | §5.4.8 / §5.20.7.27 / §7.14.6 | **ABSENT** | All writers hard-code 0 (`m1_headers.cpp:136`, `m2_headers.cpp:197`); no `cctx_type` symbol in M1/M2 syntax | No CCTX stream. |
| X11 — `using_qmatrix`, `setup_qm_params()`, QM OBU | §5.18.6.2 / §5.13 | **ABSENT** | M1/M2 write `using_qmatrix=false` (`m1_headers.cpp:176`, `m2_headers.cpp:284`); `qm_gate.cpp` bit-splices a header/OBU into an existing stream (**G-QM-GATE**) | Encoder quantisation/reconstruction never applies QM. The gate only proves the decoder accepts the syntax; only identity user matrices keep recon exact. |
| X12 — Lossless (`base_q_idx=0`), `CodedLossless`, `TxMode=ONLY_4X4` | §5.18.2 / §5.18.6.1 / §5.18.8.1 | **PARTIAL** | CLI rejects `--qp 0` (`main.cpp:535-538`); separate gate-only writer `src/encoder/lossless/lossless.cpp` with **G-LOSSLESS** | Not available from the shipped CLI/M1/M2; the lossless module codes only 8×8 luma coding blocks (4×4 transforms) and 8×8 FSC IDTX. |

---

## 9. TX size reachability matrix

`BlockSize` comes from `src/encoder/partition/partition.h:63-94`; M1 uses
`TX_MODE_LARGEST`, so the TX size equals the luma coding block size; chroma is
the 4:2:0 subsize (`partition.cpp:64-68,944-957`). "CLI" means the shipped
`--m1` path (`min_block_size=8`); 4-sample sizes are reachable **as chroma**
even when the luma block is ≥8.

| TX size (index) | Luma block via CLI? | Reachable as chroma? | Signallable? | Reconstructs bit-exactly? |
|---|---|---|---|---|
| TX_4X4 (0) | No (needs min 4) | Yes: 8×8 luma → 4×4 chroma; **A-M1-TX** exact | Yes (chroma only; DCT_DCT) | Yes |
| TX_8X8 (1) | Yes | Yes: 16×16 luma | Yes | Yes |
| TX_16X16 (2) | Yes | Yes: 32×32 luma | Yes | Yes |
| TX_32X32 (3) | Yes | Yes: 64×64 luma | Yes (DCTONLY, no type symbol) | Yes |
| TX_64X64 (4) | Yes (64×64 block) | No | Yes (DCTONLY, no type symbol) | Yes (**A-M1-TX**) |
| TX_4X8 (5) | No | Yes: 8×16 luma → 4×8 chroma; exact | Yes (chroma only; DCT) | Yes |
| TX_8X4 (6) | No | Yes: 16×8 luma → 8×4 chroma; exact | Yes (chroma only; DCT) | Yes |
| TX_8X16 (7) | Yes | Yes: 16×32 luma | Yes | Yes |
| TX_16X8 (8) | Yes | Yes: 32×16 luma | Yes | Yes |
| TX_16X32 (9) | Yes | Yes: 32×64 luma | Yes (WIDE/HIGH_32 set) | Yes (**A-M1-TX**) |
| TX_32X16 (10) | Yes | Yes: 64×32 luma | Yes (WIDE/HIGH_32 set) | Yes (**A-M1-TX**) |
| TX_32X64 (11) | Yes | No | Yes (DCTONLY, no type symbol) | Yes |
| TX_64X32 (12) | Yes | No | Yes (DCTONLY, no type symbol) | Yes (**A-M1-TX**) |
| TX_4X16 (13) | No | Yes: 8×32 luma → 4×16; exact | Yes (chroma only; DCT) | Yes |
| TX_16X4 (14) | No | Yes: 32×8 luma → 16×4; exact | Yes (chroma only; DCT) | Yes |
| TX_8X32 (15) | Yes | Yes: 16×64 luma | Yes (WIDE/HIGH_32) | Yes (**A-M1-TX**) |
| TX_32X8 (16) | Yes | Yes: 64×16 luma | Yes (WIDE/HIGH_32) | Yes (**A-M1-TX**) |
| TX_16X64 (17) | Yes | No | Yes (WIDE/HIGH_64) | Yes (**A-M1-TX**) |
| TX_64X16 (18) | Yes | No | Yes (WIDE/HIGH_64) | Yes (**A-M1-TX**) |
| TX_4X32 (19) | No | Yes: 8×64 luma → 4×32; exact | Yes (chroma only; DCT) | Yes |
| TX_32X4 (20) | No | Yes: 64×8 luma → 32×4; exact | Yes (chroma only; DCT) | Yes |
| TX_8X64 (21) | Yes | No | Yes (WIDE/HIGH_64) | Yes (**A-M1-TX**) |
| TX_64X8 (22) | Yes | No | Yes (WIDE/HIGH_64) | Yes (**A-M1-TX**) |
| TX_4X64 (23) | No | No (luma 8×128 impossible) | **No** | **No** |
| TX_64X4 (24) | No | No (luma 128×8 impossible) | **No** | **No** |

Notes:
- The M1 library (`M1Options::min_block_size = 4`) can create luma 4-sample
  blocks in principle; the CLI offers no such flag and the partition search did
  not select any 4-sample luma block in spot checks, so ADST/FDST at length 4
  (T5/T8) remain library-only.
- M0 (default CLI) supports only 64×64, 32×32, 32×16, 16×32, 32×8 and chroma
  32×32/16×16/16×8/8×16/16×4, all DCT_DCT (`src/encoder/m0_still.cpp:98-112`;
  README M0 section). It cannot signal any TX type.

---

## 10. Code-level deviations found while auditing

These are not "unimplemented tools"; they are places where the component
arithmetic disagrees with §7.14 even if the tool were wired:

1. **`qm::core::dequant_denom` uses the wrong TX-class test for TCQ.**
   §7.14.4 keys the extra shift on `get_tx_class(PlaneTxType)==TX_CLASS_2D`,
   which includes IDTX (type 9). `qm_core.h:231-238` uses
   `tx_uses_qm(tx_type) = tx_type < 9` (`qm_core.h:108`), so IDTX loses the
   shift. `qm_test.cpp:192-200,404-450` asserts the wrong result. The
   `dcdeltaq` implementation is correct (`quant.cpp:24-35` uses
   `tx_type >= 10` as the 1-D class test).
2. **`qm::core::dequant_coeff` omits the §7.14.4 Clip3.** `qm_core.h:242-255`
   returns `sign*(dq/denom)`; step 8 of §7.14.4 clips to
   `±(1<<(7+BitDepth))`. `qm_test.cpp:181-190` also omits the clip. The
   `dcdeltaq` path clips (`quant.cpp:240-245`), but at fixed 8-bit bounds.
3. **Lossless IDTX shift is not modelled for `pels >= 1024`.**
   `lossless.cpp:685-700` copies levels instead of `Dequant >> (3-shift)`;
   correct only while shift is 0 (4×4/8×8). The module never codes larger
   sizes, so this is latent.
4. **M0 `transform_shifts()` is a hand-written subset, not
   `Transform_Shift[txSz]`.** `m0_still.cpp:1281-1305` is correct for the
   coded sizes M0 emits, but a 64×32/32×64 leaf (debug partition modes) would
   use adjusted 32×32 shifts `{6,13}` instead of `TX_64X32`/`TX_32X64`
   `{6,12}`. The default M0 tree does not code those leaves (README:
   "partition nodes only"), so no shipped stream is affected.

---

## 11. Counts

**73 audited items** (excluding the 25-row size matrix, which is a
cross-check).

| Status | Count | Items |
|---|---:|---|
| **WIRED** | **26** | Q1, D1, D8, D9, R2, T1–T4, T6, T7, T9, T10, T15, T16, M1, M2, M5, M6, M7, M9, M10, M13, M14, X3, X6 |
| **PARTIAL** | **22** | Q2, Q4, Q5, D2, D10, R1, R3, T5, T8, T11–T14, W1, M3, M4, M8, M11, X1, X5, X7, X12 |
| **COMPONENT** | **18** | Q3, D3–D7, S1–S3, I1, IST1–IST7, M12 |
| **ABSENT** | **7** | IST8, X2, X4, X8, X9, X10, X11 |
| **Total** | **73** | |

By area: §7.14.2 dequant functions 5, §7.14.3 reconstruct 3, §7.14.4/§7.14.5/§7.14.6 dequant+save+CCTX 13, §7.15.2.1 1D 16, §7.15.2.2/.3 WHT/identity 2, §7.15.3 IST 8, §7.15.4/.4.1 2D/matrix 14, signalling 12.

---

## 12. TX type / size combinations the encoder cannot signal or cannot reconstruct bit-exactly

### 12.1 Cannot be signalled at all

1. **TX_4X64 and TX_64X4** — no corresponding `BlockSize` in AV2 and no
   explicit `tx_size` syntax (`TX_MODE_LARGEST`). Never emitted by any path.
2. **Any transform smaller than its coding block** — e.g. TX_4X4 in an 8×8
   block, TX_8X8 in 16×16, TX_16X16 in 32×32, TX_32X32 in 64×64, TX_8X16 in
   16×32. `tx_mode_select` is always 0 (X1/X2), and `tx_split` is never
   written. The `tx_select` component cannot change this.
3. **Luma 4-sample transforms** (TX_4X4, TX_4X8, TX_8X4, TX_4X16, TX_16X4,
   TX_4X32, TX_32X4) — the CLI hard-codes `min_block_size=8` and exposes no
   flag. They are signalled only as **chroma** transforms (DCT_DCT).
   Consequently **ADST/FDST at length 4 (T5/T8) are never signalled**: all
   4-sample blocks are chroma, and chroma is forced to DCT_DCT.
4. **All intra identity TX types** — IDTX, V_DCT, H_DCT, V_ADST, H_ADST,
   V_FLIPADST, H_FLIPADST: `enable_idtx_intra=0` and the M1 candidate list
   filters them (`X4`). This also removes the identity entries from
   WIDE_64/HIGH_64 and WIDE_32/HIGH_32 sets.
5. **All inter TX types except DCT_DCT** — production `m2_syntax.cpp` writes
   the DCT_DCT symbol for INTER_1, INTER_2, DCT_IDTX and the long sets (`X5`).
6. **DDTX / FDDT** (derived from inter ADST/FDST at adjusted length 8/16) —
   requires `enable_inter_ddt=1`; no production stream sets it (`X8`).
7. **`sec_tx_type != 0` (IST)** — both sequence enable bits are 0 and no
   per-block symbol is written (`IST8`).
8. **`reduced_tx_set` 1/2/3** — always 0 (`X7`); INTRA_2 and the two-type
   reduced sets never occur.
9. **Chroma non-DCT types** — `enable_chroma_dctonly=1` forces DCT_DCT (X6).
10. **`using_qmatrix=1`, `enable_tcq=1`, `enable_cctx=1`** — hard-coded 0 in
    every writer (`X9`–`X11`). (`enable_parity_hiding=1` is likewise hard-coded
    0 but is outside §7.14/§7.15.)

### 12.2 Can be signalled only by a non-production/gate writer

| Tool | Writer | What it proves |
|---|---|---|
| Lossless WHT / lossless IDTX / DPCM / FSC | `src/encoder/lossless` | 22 `dav2d`+`avmdec` checks, but not linked into the CLI; only 8×8 coding blocks (4×4 transforms) and 8×8 FSC IDTX. |
| DDTX / FDDT | `src/encoder/m2/ddt/gate` (patched build-dir copies) | 4 variants, recon exact, but production source still hard-codes DCT_DCT and `enable_inter_ddt=0`. |
| QM | `src/encoder/qm/qm_gate.cpp` (header/OBU splice) | Decoder accepts default matrices and user-QM OBUs; encoder recon remains QM-agnostic, so only identity user QM is recon-exact. |
| 10-bit transforms | `src/cuda/m1/hbd` | CPU/CUDA component + 19,312 checks; no 10-bit stream can be produced (`gate_10bit_still.sh:6-7`). |

### 12.3 Can be signalled (or spliced) but the encoder cannot reconstruct bit-exactly

1. **`enable_inter_ddt=1` with an inter ADST/FDST block at adjusted length
   8/16.** The production reconstruction calls `txv::inverse_transform`
   (`m2/residual/inter_residual.cpp:338`), which has no DDTX/FDDT branch; the
   decoder derives DDTX/FDDT from the same `inter_tx_type`, so recon drifts.
   The gate works only because it substitutes `ddt::inverse_block` at that call
   site (`src/encoder/m2/ddt/gate/README.md` "Blocked core edits" 1–4).
2. **Any `sec_tx_type != 0` stream.** The reconstruct order in M1/M2 never
   invokes §7.15.3; the 2D inverse runs on the un-rotated dequant coefficients
   (R1).
3. **`enable_tcq=1` streams.** M1's quantiser emits scalar/RDOQ levels
   (`m1_still.cpp:882-950`) and `txv::dequantize` has no TCQ extra shift; the
   decoder applies both the level mapping and the extra `dqDenom` shift.
4. **`using_qmatrix=1` with a non-identity matrix.** The encoder dequant uses
   `q2=q` (D8); the decoder uses `Round2(q*m,5)`. `qm_gate.sh` demonstrates the
   pixel difference for default matrices.
5. **`enable_cctx=1` with `cctx_type != 0`.** No U/V rotation is applied in
   the encoder reconstruction (S2/S3).
6. **`base_q_idx=0` lossless streams.** The CLI rejects them; the separate
   lossless module covers only 4×4 WHT / 8×8 FSC IDTX.
7. **10-bit streams (`bit_depth_idc=0`).** All wired paths clip at 8-bit; HBD
   is component-only.
8. **Lossy FSC IDTX streams.** No `enable_fsc=1` path exists in M1/M2; the
   `useFsc` term only exists as an input to the component dequant.

---
