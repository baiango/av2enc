# Intra identity / non-DCT transform-type coverage (`src/codec/transform/idtx/`)

Status: **component + integrated still path (IDTX-only).** The CPU
`--still-idtx` path (`src/pipeline/still/still_encoder.cpp` +
`still_syntax.cpp` + `still_headers.cpp`) writes `enable_idtx_intra = 1`,
codes the §5.20.5.5 `fsc_mode = 0` symbol for every luma coding block <=
32x32, and adds `IDTX` to the intra luma TX-type candidates. IDTX is
`TX_CLASS_2D`, so the existing diagonal scan and 2-D reference contexts are
correct; the one-axis identity types (`V/H_DCT`, `V/H_ADST`,
`V/H_FLIPADST`) stay disabled because they need the class-specific scans.
The decoder gate is `still_idtx_gate.sh`: default-off, pinned base streams
unchanged, IDTX selected on long-side transforms at 640x360 (5 blocks), and
every stream sample-exact with `dav2d --strict 1`. IDTX composes with TX
partitioning, TCQ, IST, CCTX, RDOQ, QM and the TX_TYPE search.

The rest of this document describes the standalone component.

Owning and self-contained; the component itself is not wired into
`CMakeLists.txt` or `src/main.cpp`, and it does not commit. It implements the
transform-type coverage that the audit marks ABSENT in
`docs/spec-audit/08-transform.md` (X4, X7, X6) and the `enable_idtx_intra`
conditional in `docs/spec-audit/02-sequence-header.md` §5.4.8.

Normative source: AV2 v1.0.0 (commit `38f28e7`) §5.4.8, §5.20.7.29,
§5.20.8.2, §5.20.8.3, §7.14.4, §7.15.4.1. No encoder code from other projects;
`dav2d` is used only as a black-box validator.

## What was absent, and what this adds

| Gap (audit) | This component |
|---|---|
| X4 — no intra block can signal IDTX/V_DCT/H_DCT/V_ADST/H_ADST/V_FLIPADST/H_FLIPADST (`enable_idtx_intra=0`, candidates filtered) | `is_identity_type()`, the effective-flag derivation, candidate filtering, and a decoder-gated/byte-pinned writer path |
| X7 — `reduced_tx_set` always 0 | §5.20.8.3 sets for 1/2/3, including the `reduced_tx_set == 2` intra DCT override and INTRA_2's two types |
| X6 — chroma always DCT (`enable_chroma_dctonly=1`) | §5.20.7.29 chroma derivation through `Mode_To_Txfm` + `is_tx_type_in_set`, decoder-gated as ADST_DCT |
| D5/D3 — QM and TCQ predicates keyed on the type | `derive_tool_use()`: `useQm` requires `PlaneTxType < IDTX`; the TCQ extra shift uses `TX_CLASS_2D` (which includes IDTX) |
| IST applicability | `use_ist` excludes identity types through the DCT_DCT/ADST_ADST rule |
| §5.4.8 `enable_fsc` interaction | `enable_fsc=1` infers `enable_idtx_intra=1` and removes the bit; `useFsc` and `allow_fsc_intra()` predicates |

## Files

| File | Contents |
|---|---|
| `idtx_intra.h` / `.cpp` | planner: sequence-flag inference, `get_tx_set`, legal candidate enumeration, `plan_type`, chroma derivation, tool interaction |
| `type_syntax.h` / `.cpp` | §5.20.8.2 `transform_type()` writer/reader over the normative §8.2 range coder with the §9.3 CDFs |
| `seq_bits.h` / `.cpp` | §5.4.8 `sequence_transform_quant_entropy_config()` subset writer/reader (the `enable_fsc` conditional lives here) |
| `idtx_intra_test.cpp` | unit, exhaustive legality, chroma derivation, reduced sets, QM/IST, identity parity, codec round trip |
| `idtx_intra_gate.cpp` | `dav2d` stream gate: feature (non-DCT), all-off control, identity |
| `patch_still_syntax.py` | build-dir patch adding the §5.20.5.5 `fsc_mode` symbol (see gate) |
| `idtx_goldens.h` | pinned hashes / bit strings |
| `build_and_test.sh` | standalone build + tests + gate (no CMake) |

## API

```cpp
namespace av2enc::idtx {

struct SeqFlags { bool enable_fsc, enable_idtx_intra, enable_intra_ist, ...;
                  int reduced_tx_set; };
DerivedSeqFlags derive_sequence_flags(const SeqFlags&);
//   idtx_bit_coded  = !enable_fsc
//   enable_idtx_intra = enable_fsc || enable_idtx_intra

bool is_identity_type(TxType);      // type >= IDTX (9)
int  tx_class(TxType);              // 0 2D, 1 HORIZ, 2 VERT (IDTX is 2D)
bool intra_set_allows(Set, TxType); // Tx_Type_In_Set_Intra row

struct Query { int coded_w, coded_h, plane, mapped_mode, y_mode,
               uv_mode, uv_angle_delta, eob; SeqFlags flags; };

bool get_tx_set(const Query&, Set*);            // §5.20.8.3, plane-dependent
bool enumerate_types(const Query&, TypeSet*);   // §5.20.8.2 symbol order
bool plan_type(const Query&, TxType, TypePlan*);// validate + exact symbols
TxType default_type(const Query&);              // DCT_DCT (or derived chroma)

bool derive_chroma_type(int uv_mode, int uv_angle_delta, int w, int h,
                        const SeqFlags&, TxType*, Set* = nullptr);
TxType mode_to_txfm(int uv_mode);               // 14-entry table

struct ToolUse { bool use_qm, use_ist, use_fsc, allow_fsc_intra,
                 tcq_extra_shift, parity_hiding; int tx_class, ist_eob_limit; };
ToolUse derive_tool_use(const Query&, TxType plane_type, int seg_level,
                        bool qm_data_present, bool lossless, bool fsc_mode);
}
```

`TypePlan::type_syntax` is false when no `intra_tx_type` symbol is coded
(`eob <= 1`, DCT-only size, `reduced_tx_set == 2` on plane 0, or chroma). For
plane > 0 `plan_type` accepts only the derived type, because
`compute_tx_type()` ignores `TxTypes[]`.

The type codec:

```cpp
TypeSyntaxWriter w;      // CDFs adapt unless `adapt == false`
w.write(set, tx_size_index, type, symbol, flag);
w.finish(&bytes);
TypeSyntaxReader r;
r.init(bytes, bytes.size());
r.read(type_set, tx_size_index, adapt, &TypeSymbols{...});
r.finish();
```

`TypeSyntaxReader` maps the symbol back through the caller's `TypeSet`, so
INTRA_1/INTRA_2 use `Md_Idx_To_Type` exactly like the decoder (the symbol alone
does not name a type).

### Legality rules implemented

* `enable_idtx_intra` (effective) is required for every type `>= IDTX` on every
  plane and in every set, including the long-32/long-64 sets that always list
  identity entries.
* `eob == 1` forces DCT_DCT with no symbol (`eob == 0` is txb_skip).
* `reduced_tx_set == 2` skips the entire type block for plane 0 at every size.
* Plane-dependent reduced set: plane 0 uses `reduced_tx_set`, chroma uses
  `enable_chroma_dctonly` (`1` → INTRA_2, then `compute_tx_type` returns
  DCT_DCT; `0` → INTRA_1/INTRA_2).
* Chroma 32x32 and larger luma-relative sizes are DCTONLY (§5.20.8.3).
* Long-64 sets imply `is_long_side_dct = 1`; only WIDE_32/HIGH_32 code it.
* `useQm` is false for every identity type (`PlaneTxType < IDTX`).
* IST is only possible for DCT_DCT/ADST_ADST, so never for an identity type.
* With `enable_fsc == 1`, the effective `enable_idtx_intra` is 1 even though
  its raw value is 0, and `allow_fsc_intra()`/`useFsc` are derived from it.

## Tests

`./build_and_test.sh` → **62,524,206 checks, 0 failures**, clean under
`SANITIZE=1` and `WERROR=1`.

| Test | What it pins |
|---|---|
| sequence flags | `enable_fsc` inference, bit presence, 32-combination round trip, §5.4.8 bits golden, and the incumbent M1 header changing in exactly one bit for the flag |
| exhaustive legality | 25 sizes × 3 planes × 4 reduced sets × idtx on/off × chroma-DCT-only on/off × 13 modes × 14 UV modes × 5 EOBs = **1,092,000** enumerations; every candidate set-legal, identity-gated, symbol-unique, DCT always present, every non-candidate rejected |
| chroma derivation | `Mode_To_Txfm` table, V/H/D45/D135/PAETH/UV_CFL, wide-angle remap at 8x16, DCT-only sizes, never an identity type |
| reduced sets | 16x16: 0→INTRA_1(7), 1/3→INTRA_2(2), 2→DCT; long-32 with reduced 2 still forced DCT; identity-off drops exactly the identity entries (8→3) |
| QM/IST | `useQm == type < IDTX` for all 16 types; IST only DCT_DCT/ADST_ADST, eob limits, PAETH exclusion, lossless off |
| identity parity | independent §7.15.4.1 process vs `txv::inverse_block` for all 7 identity types × 25 sizes: **107,632 samples exact** |
| type codec | 67-decision multi-set stream, adaptation, escape-free long sets, byte + round-trip golden |

Goldens are FNV-1a 64 over deterministic bytes (no floating point hashed) in
`idtx_goldens.h`; the bytes are reproducible on any host.

## Decoder gate

`./build_and_test.sh` builds the plan with
`cuda/m1/plan/plan.cpp` (the same partition search the CPU M1 reference uses)
and writes three streams from `tests/data/basic_64x64_1f.y4m`. All three are
decoded with `dav2d --strict 1 --threads 1` and the visible samples must match
the encoder's own reconstruction exactly.

```
feature   stream: 1039 bytes, fnv 5ba50465798c1951, type symbols 11
all-off   stream: 1015 bytes, fnv 4f377f66c3651e11
all-off   control: byte-identical to the DCT-only writer
feature   types  : 6 non-DCT luma blocks, 20 chroma ADST_DCT
feature   recon: max|diff| Y=0 U=0 V=0
all-off   recon: max|diff| Y=0 U=0 V=0
identity  stream: 1637 bytes, fnv 08a999e7cc62c347, 8 identity blocks
identity  recon: max|diff| Y=0 U=0 V=0
quality         : feature 42.915 dB (1039 B)  all-off 42.809 dB (1015 B)  identity 42.239 dB (1637 B)
IDTX GATE PASS
```

* **feature** — `enable_idtx_intra=0`, `enable_chroma_dctonly=0`. Luma signals
  planner-chosen non-DCT `TX_CLASS_2D` types (ADST_DCT / DCT_ADST / ADST_ADST /
  FLIPADST_*); chroma derives ADST_DCT from `UVMode = V_PRED` with no type
  symbol. The decoded pixels are bit-exact.
* **all-off** — `enable_idtx_intra=0`, `enable_chroma_dctonly=1`, DCT_DCT on
  every plane; bit-exact and **byte-identical** to a second encode that bypasses
  the planner entirely (pre-feature writer path). FNV-pinned.
* **identity** — `enable_idtx_intra=1`, `enable_chroma_dctonly=0`; 8 luma blocks
  signal IDTX (the only identity type `TX_CLASS_2D`), bit-exact. FNV-pinned.

The streams also pass `tools/validate_dav2d.sh` (PSNR 42.914 / 42.809 dB on the
64x64 source), and the AVM reference decoder accepts both decoder-gated streams
(`avmdec --md5`: `24d9c399…` feature, `76f2695d…` identity), so the identity
tile is not a dav2d-specific quirk.

### The `enable_idtx_intra = 1` decoder rejection was a missing symbol

`docs/m1-features.md` records that signalling `enable_idtx_intra=1` made
`dav2d` reject otherwise-valid streams and calls it a sequence-header
bit-position/open item. It is not a header problem: §5.20.5.5 ends
`read_intra_y_mode()` with

```
if ( TreeType != CHROMA_PART && allow_fsc_intra() ) { fsc_mode S() }
```

and `allow_fsc_intra()` is `enable_idtx_intra && w <= FSC_MAX && h <= FSC_MAX`.
The core writer never codes that symbol, so every intra luma block <= 32x32 is
one symbol short as soon as the flag is 1.

`patch_still_syntax.py` adds the symbol to a build-directory copy of
`src/encoder/m1/still_syntax.cpp` (always `fsc_mode = 0`, so the coefficient path
is unchanged; the `FSC=1` `coeff_base_bob`/`coeff_base_idtx` path stays out of
scope). With it, the identity stream decodes bit-exactly. The permanent fix
belongs in `still_syntax.cpp` (see call sites); the patch is the same
build-dir-patched-copy technique as `src/encoder/m2/ddt/gate`.

Streaming the six one-axis identity types (`V/H_DCT`, `V/H_ADST`,
`V/H_FLIPADST`) additionally needs `code_tx()` to use the class-specific scan
and `Mag_Ref_Offset_With_Tx_Class` contexts; it currently hard-codes the
diagonal scan and 2D reference offsets (`src/encoder/m1/still_syntax.cpp:1119-1135`
and the `kSigRef`/`kMagRef` tables). The planner and the txv parity test cover
all seven types today; the stream proves IDTX end to end.

## Quality / rate

Measured on the pinned 64x64 frame at `base_q_idx = 90`, one fixed partition
plan, DC luma prediction:

| Stream | Bytes | PSNR |
|---|---:|---:|
| all-off (DCT_DCT) | 1015 | 42.809 dB |
| feature (planner non-DCT luma + derived ADST_DCT chroma) | 1039 (+2.4%) | 42.915 dB (+0.106 dB) |
| identity (IDTX luma) | 1637 (+61%) | 42.239 dB (−0.570 dB) |

The identity result is the expected one: IDTX is a poor RD choice on natural
content at this QP, which is why the planner must keep it optional and the
selection is SSE+rate driven. This is a fixed-tree measurement, not a BD-rate
claim.

## Call sites (for the owner of the M1 core)

Nothing was edited outside this directory. To wire the coverage:

| Point | Current code | Suggested use |
|---|---|---|
| `src/encoder/m1/still_headers.cpp:131` | writes `c.enable_idtx_intra` | keep the bit; it is already conditional on `enable_fsc` per §5.4.8 (M1 never sets fsc) |
| `src/encoder/m1/still_headers.cpp:134` | writes `c.enable_chroma_dctonly` | expose a config flag; `idtx::derive_chroma_type()` gives the encoder-side type |
| `src/encoder/m1/still_encoder.cpp:1815-1816` | forces chroma DCT-only and `enable_idtx_intra=0` | replace with the derived flags from `SeqFlags` |
| `src/encoder/m1/still_encoder.cpp:831-848` | filters identity candidates by hand | replace with `idtx::enumerate_types()` / `plan_type()` |
| `src/encoder/m1/still_syntax.cpp:149-160` | `luma_tx_type_syntax()` hard-codes reduced 0 / idtx 0 | use `idtx::get_tx_set()` and emit INTRA_2 / long-set symbols from `TypePlan` |
| `src/encoder/m1/still_syntax.cpp:1064-1112` | symbol emission | needs the `reduced_tx_set == 2` DCT override, the §5.20.5.5 `fsc_mode` symbol when `enable_idtx_intra=1`, and the class-specific scan/contexts before the axis types can stream |
| `src/encoder/qm/qm_core.h:108,231-238` | `tx_uses_qm = tx_type < 9` reused as the TCQ 2D test | `idtx::tx_class()` is the correct predicate (IDTX is 2D) |
| `src/encoder/tx_secondary/tx_secondary.cpp:181-238` | IST applicability | identity types are already excluded by the DCT_DCT/ADST_ADST rule; `derive_tool_use()` returns the same answer |

The planner is stateless and allocation-free; one call per TX block at decision
time (`enumerate_types`) and one per chosen block at write time (`plan_type`).
