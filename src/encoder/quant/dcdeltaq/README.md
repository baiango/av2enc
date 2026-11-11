# DC/AC delta-Q (`src/encoder/quant/dcdeltaq/`)

Status: **component only.** It is not wired into `CMakeLists.txt`, `src/main.cpp`
or the M1/M2 core, and it edits no file outside this directory.  Nothing is
committed.

Normative source: AV2 Bitstream & Decoding Process Specification v1.0.0.

| Clause | Implemented here |
|---|---|
| §5.4.8 | `sequence_transform_quant_entropy_config()`: `separate_uv_delta_q`, `equal_ac_dc_q`, `base_y_dc_delta_q`, `base_uv_dc_delta_q`, `base_uv_ac_delta_q`, `y_dc_delta_q_enabled`, `uv_dc_delta_q_enabled`, `uv_ac_delta_q_enabled` (plus the tool flags needed to reach them) |
| §5.18.6.1 | complete `quantization_params()`: `base_q_idx` and all `DeltaQ*` fields |
| §5.18.6.3 | `read_delta_q()` (`f(1) delta_coded`, `su(7) delta_q`) |
| §5.18.2 | `LosslessArray[]` / `CodedLossless` |
| §7.14.2 | `qlookup()`, `get_q()`, `get_qindex()`, `get_dc_quant()`, `get_ac_quant()` |
| §7.14.4 | forward/inverse quantization with the QM `q2` and TCQ `shift += 1` interactions |
| §5.20.7.27 | TCQ `q0` / level mapping helpers used by the interplay tests |

## Why this is not just "add three fields"

* With `equal_ac_dc_q = 1` **no Y-DC or UV-DC base field is coded at all**:
  `BaseYDcDeltaQ` is inferred 0, `BaseUVDcDeltaQ` is copied from
  `BaseUVAcDeltaQ`, and both DC enables stay 0.  A non-zero Y-DC delta is
  impossible without `equal_ac_dc_q = 0`, so the current M1/M2 streams cannot
  express one.
* The sequence base deltas have only 32 legal values each:
  `DELTA_DCQUANT_BITS = 5`, `Base* = DELTA_DCQUANT_MIN + raw` with
  `DELTA_DCQUANT_MIN = -23`, `DELTA_DCQUANT_MAX = 8`.
* The frame deltas are `su(7)`, i.e. `[-64, 63]`, and are only present when the
  sequence-level enable bit is 1.
* `diff_uv_delta` exists only when `separate_uv_delta_q = 1`.  With it at 0 the
  decoder copies `DeltaQV* = DeltaQU*`.
* `get_q(0, delta <= 0)` is the §7.14.2 special case `64`; at
  `base_q_idx = 0`, `CodedLossless` additionally requires every
  `DeltaQ* + Base*` to be `<= 0`.  A positive total delta leaves lossless mode
  even though the qindex is still 0.
* The coefficient CDF slice is selected from the **raw** `base_q_idx`
  (§6.17.2), never from the effective DC/AC qindex.

## Files

| File | Purpose |
|---|---|
| `dcdeltaq.h` | public API |
| `syntax.cpp` | §5.4.8 and §5.18.6.1 writers/readers |
| `quant.cpp` | §7.14.2 derivation, §7.14.4 quantizer/dequantizer, TCQ helpers |
| `search.cpp` | deterministic encoder-side frame-delta RD search |
| `dcdeltaq_test.cpp` | 10,915,148 checks |
| `dcdeltaq_bench.cpp` | quality/rate measurement harness |
| `dcdeltaq_gate.cpp` | header-splice decoder gate helper |
| `build_and_test.sh` | build + unit tests (`WERROR=1`, `SANITIZE=1`) |
| `build_bench.sh` | build + run the RD harness |
| `gate_dav2d.sh` | pinned `dav2d --strict 1 --threads 1` gate |

## API

### Sequence header

```cpp
#include "encoder/quant/dcdeltaq/dcdeltaq.h"

av2enc::dcdeltaq::SequenceDeltaQ seq;
seq.equal_ac_dc_q = false;              // required for a Y-DC delta
seq.separate_uv_delta_q = true;
seq.y_dc_delta_q_enabled = true;
seq.uv_dc_delta_q_enabled = true;
seq.uv_ac_delta_q_enabled = true;
seq.base_y_dc_delta_q = -3;             // [-23, 8]
seq.base_uv_dc_delta_q = -5;
seq.base_uv_ac_delta_q = 0;

// Delta-only splice (starts at separate_uv_delta_q):
av2enc::BitWriter w;
av2enc::dcdeltaq::write_sequence_delta_q_fields(w, seq, &error);

// Complete §5.4.8 group (starts at enable_fsc):
av2enc::dcdeltaq::SeqTransformQuantEntropyConfig cfg;
cfg.single_picture_header_flag = true;
cfg.enable_chroma_dctonly = true;      // match the base stream
av2enc::dcdeltaq::write_sequence_transform_quant_entropy_config(
    w, cfg, seq, &error);
```

`normalized_sequence_delta_q()` is the view the decoder derives:
monochrome forces UV fields off, and `equal_ac_dc_q = 1` forces
`BaseYDcDeltaQ = 0` / `BaseUVDcDeltaQ = BaseUVAcDeltaQ`.

### Frame header

```cpp
av2enc::dcdeltaq::FrameDeltaQ f;
f.delta_q_y_dc = -3;
f.delta_q_u_dc = -5;
f.delta_q_v_dc = -6;                    // separate_uv_delta_q = 1
f.delta_q_u_ac = -1;
f.delta_q_v_ac = -2;

av2enc::dcdeltaq::write_quantization_params(w, /*base_q_idx=*/90, seq, f,
                                            &error);
// reader:
av2enc::deltaq::BitReader r(w.bytes());
av2enc::dcdeltaq::QuantizationParams qp;
av2enc::dcdeltaq::read_quantization_params(r, seq, /*bit_depth=*/8, qp,
                                           &error);
```

The writer canonicalizes the syntax:

* zero values are written as `delta_coded = 0` (one bit), not `1 + su(7)`;
* `diff_uv_delta` is derived from the U/V values (and only emitted when
  `separate_uv_delta_q = 1`);
* with `equal_ac_dc_q = 1` the caller must pass
  `DeltaQU/VDc == DeltaQU/VAc`; only the AC field is written;
* with `separate_uv_delta_q = 0` the caller must pass `DeltaQV* = DeltaQU*`.

### Derived quantizers

```cpp
av2enc::dcdeltaq::QuantFrameInputs in;
in.base_q_idx = 90;
in.seq = seq;
in.frame = f;

const auto q = av2enc::dcdeltaq::derive_quant(in, /*segment_id=*/0);
// q.plane[0].dc / .ac, q.plane[1], q.plane[2]
// q.lossless is §5.18.2 Lossless for this segment.
```

`derive_lossless_flags(in)` returns the full `LosslessArray`, `CodedLossless`
and `HasLosslessSegment` for all `MaxSegments`.

### Quantizer / dequantizer

```cpp
av2enc::dcdeltaq::QuantBlock block;
block.plane = 0;
block.coded_w = 8;
block.coded_h = 8;
block.tx_size = 1;
block.tx_type = 0;
block.q = q.plane[0];
block.coefficient_scale = params.quant_factor;  // tx_variants at q2 = 528
block.using_qmatrix = false;                     // or a qm::QmFrameState
block.allow_tcq = false;

av2enc::dcdeltaq::quantize_block(coeff, block, levels, &error);
av2enc::dcdeltaq::dequantize_block(levels, block, dequant, &error);
```

`quantize_block` uses `q2` per coefficient (`q.dc` at `(0,0)`, `q.ac`
elsewhere), applies QM as `Round2(q * m, 5)`, and multiplies the forward target
by `2^extra_shift` so the decoder's TCQ shift is compensated.
`dequantize_block` is the exact §7.14.4 path:

```
shift = (pels > 256) + (pels > 1024)
        + (allow_tcq && plane == 0 && !lossless && tx_class == 2D && !use_fsc)
dequant = Clip3(-(1<<15), (1<<15)-1,
                sign * (Round2((abs(level) * q2) & 0xFFFFFF, 3) / (1<<shift)))
```

When `lossless` is set, both functions bypass the §7.14.4 scaling (the
reversible WHT/IDTX path does not invoke dequantization).

## Encoder-side search

`search_frame_delta_q()` is a deterministic, bounded coordinate descent over
the independent frame-delta knobs (max 5: Y-DC, U/V DC and AC), with:

* a 19-value candidate ladder bounded by `max_delta_magnitude`;
* a hard `max_luma_blocks` cap (deterministic block subsampling);
* a deterministic tie-break (smaller `|delta|`, then smaller value);
* at `base_q_idx = 0`, positive frame deltas are excluded by default so
  `CodedLossless` stays 1 (the §5.18.2 interaction);
* legal output by construction (`validate_frame_delta_q`).

The search fixes the sequence base deltas at 0 and enables all three frame
fields.  The frame range `[-64, 63]` is a superset of the sequence base range
`[-23, 8]`, so the search can reach every effective `qindex + delta`; base
deltas only change how a value is split between the sequence and frame headers.

The cost model is 8x8 luma / 4x4 chroma DC-predicted DCT blocks, the module's
own §7.14.4 quantizer, SSE, and a deterministic entropy proxy.  It is **not** a
real entropy coder and `lambda_scale` is calibrated for the proxy
(`0.002`), not for AV1.

Measured with `build_bench.sh` (modelled rate, not bytes):

`testdata/m0_64x64_1f.y4m`, `separate_uv_delta_q = 0` unless noted:

| QP | ΔPSNR | Δbits | Σrate |
|---:|---:|---:|---:|
| 32 | **+0.801 dB** | +529.7 | +3.7% |
| 64 | **+0.446 dB** | +255.0 | +2.6% |
| 90 (sep=1) | +0.072 dB | +5.0 | +0.07% |
| 120 | −0.174 dB | −120.4 | −3.3% |
| 141 (sep=1) | −0.097 dB | −85.6 | −3.8% |
| 191 | **+0.521 dB** | −12.0 | −1.4% |
| 255 | **+1.729 dB** | −4.4 | −0.7% |

`testdata/shine_256x256_1f.y4m`:

| QP | ΔPSNR | Δbits | Σrate |
|---:|---:|---:|---:|
| 32 | **+0.983 dB** | +3958.7 | +4.8% |
| 64 | **+0.350 dB** | +1167.7 | +2.0% |
| 90 | **+0.376 dB** | +369.9 | +0.9% |
| 120 (sep=1) | **+0.314 dB** | +120.5 | +0.4% |
| 141 (sep=1) | **+0.280 dB** | −32.4 | −0.2% |
| 191 (sep=1) | **+0.339 dB** | +69.5 | +0.6% |
| 255 | **+2.414 dB** | +124.4 | +1.3% |

Interpretation: the search is most useful at high QP (large PSNR gain for a
small rate increase) and at low QP it buys quality with a few percent more
rate.  It never changes more than the requested bounded delta ladder.

## Tests

```sh
./build_and_test.sh                 # 10,915,148 checks, 0 failures
WERROR=1 ./build_and_test.sh
SANITIZE=1 ./build_and_test.sh      # clean under ASan + UBSan
```

Coverage:

* `qlookup()` for all 0..255, `get_q()` special cases and clamps,
  `get_qindex()` with ALT_Q and `CurrentQIndex`;
* **all 32³ sequence base-delta combinations** for `equal_ac_dc_q = 0`, both
  `separate_uv_delta_q` values, plus every UV-AC value for
  `equal_ac_dc_q = 1` and monochrome;
* complete §5.4.8 round-trips over the tool-flag combinations, including the
  `enable_fsc -> enable_idtx_intra`, `enable_tcq -> choose_tcq_per_frame ->
  enable_parity_hiding`, and `single_picture_header_flag -> enable_avg_cdf`
  inferences;
* **every `su(7)` frame value** for each of the five fields in both
  `separate_uv_delta_q` modes; **all 128² pairs**; and the exhaustive **128³**
  `(Y-DC, UV-DC, UV-AC)` cross product;
* QP sweep `{0,1,2,7,8,16,32,64,90,120,140,141,190,191,254,255}` with every
  combined delta in `{-87,-64,-24,-8,-1,0,1,8,24,63,71}`, both UV modes and
  both `equal_ac_dc_q` values;
* QP 0 lossless: `CodedLossless`, `delta_q_present`, ALT_Q = −255 segments,
  positive-delta escape, and the dequant pass-through;
* QM identity/default matrices (`q2 = Round2(q*m, 5)`), the TCQ shift, and the
  TCQ level mapping;
* search determinism, legality, evaluation bound, RD ≤ zero-delta baseline,
  and QP-0 preservation.

### Pinned goldens

| Item | Bits | Value |
|---|---:|---|
| Sequence delta fields, `equal_ac_dc_q=1`, UV AC disabled | 8 | `01101110` |
| Sequence delta fields, `separate=1, equal=0`, bases −3/−5/0 | 20 | `10101001100101101111` |
| Frame `quantization_params()`, base 90, separate, YDC −3, UDC −5, VDC −6, UAC −1, VAC −2 | 49 | `0101101011111101111111011111111111111101011111110` |
| Frame, base 90, non-separate, YDC −3, UDC −5, UAC −1 | 32 | `5afdfbff` |
| Frame, base 90, `equal_ac_dc_q=1`, UV AC −1 | 16 | `5aff` |
| Full §5.4.8, default tools, delta fields disabled | 17 | `00000000001101110` |

## Decoder gate

```sh
./gate_dav2d.sh
DAV2D=/path/to/dav2d AV2ENC_BIN=/path/to/av2enc ./gate_dav2d.sh
KEEP=1 ./gate_dav2d.sh
```

The harness (`dcdeltaq_gate.cpp`) bit-splices the sequence-header §5.4.8 group
and the CLK `quantization_params()` into a real M1 still, leaving the tile
payload untouched.  The base stream is a flat 64×64 picture encoded with
`--m1 --qp 8 --m1-single --m1-dc --no-deblock`; a flat plane has DC-only
coefficients.  Variants use `base_q_idx = 1` and deltas `+7`, so every
quantizer the gate frame actually uses is `qlookup(8) = 49`, the same as the
reference stream.  The coefficient CDF slice is 0 for both `base_q_idx = 1`
and `8`, so the tile entropy decoder is unchanged.

Passing checks:

1. `--identity` rebuild is byte-identical to the input IVF;
2. `equal_ac_dc_q = 0`, shared UV deltas → `dav2d --strict 1 --threads 1`
   recon **0 diffs**;
3. `separate_uv_delta_q = 1` with `diff_uv_delta = 1` (U = +7, V = −3;
   V's flat 128 plane has no residual) → recon **0 diffs**;
4. non-zero sequence base deltas (Y −7 / UV −5) combined with frame deltas
   (Y +14 / UV +12) → the same +7 total → recon **0 diffs**;
5. a Y-DC delta that changes the effective quantizer still decodes and changes
   pixels (proves the deltas are applied, not ignored).

## Exact M1/M2 call sites (not edited here)

### Sequence header

`src/encoder/m1/m1_headers.cpp:139-142` (inside `write_sequence_header()`):

```cpp
writer.put_bit(false);  // separate_uv_delta_q
writer.put_bit(true);   // equal_ac_dc_q
writer.put_bits(23U, 5U);  // base_uv_ac_delta_q
writer.put_bit(false);     // uv_ac_delta_q_enabled
```

replace with:

```cpp
dcdeltaq::write_sequence_delta_q_fields(writer, config.delta_q, &error);
```

`src/encoder/m2/m2_headers.cpp:201-204` is the same four lines.  `M1Config` /
`M2Config` need a `dcdeltaq::SequenceDeltaQ delta_q` field.  If a complete
§5.4.8 group is preferred, replace the whole `enable_fsc` .. `uv_ac_delta_q_enabled`
run with `dcdeltaq::write_sequence_transform_quant_entropy_config(writer, cfg,
config.delta_q, &error)`; the M2 writer has the extra `enable_inter_ddt` bit at
line 195, so pass `single_picture_header_flag = false` there.

`src/encoder/lossless/lossless.cpp:2847-2850` has the same four bits in
`write_sequence_header()`.  That writer is a separate lossless syntax path; the
QP-0 interaction is covered by the unit tests, not this gate.

### Frame header

`src/encoder/m1/m1_headers.cpp:175-178`:

```cpp
writer.put_bits(c.base_q_idx, 8U);  // base_q_idx
writer.put_bit(false);              // segmentation_enabled
writer.put_bit(false);              // using_qmatrix
writer.put_bit(false);              // delta_q_present
```

replace with:

```cpp
dcdeltaq::write_quantization_params(writer, c.base_q_idx, c.delta_q,
                                    c.frame_delta_q, &error);
writer.put_bit(false);  // segmentation_enabled
writer.put_bit(false);  // using_qmatrix
writer.put_bit(false);  // delta_q_present
```

`src/encoder/m2/m2_headers.cpp:282-285` is the same base_q_idx + three bits.
`write_quantization_params()` writes base_q_idx itself; keep the following
three bits in the caller.

The callers must also validate the sequence enables against the range: M1 uses
`base_q_idx >= 1`; `M2Config::base_q_idx` is 1..255.

### Quantizer / reconstruction

`src/encoder/m1/m1_still.cpp:194-196` derives one `quant_step_` from
`rc::kQlookupQ0To255[base_q_idx]`.  Replace it with a per-frame
`dcdeltaq::QuantFrameState` (one `PlaneQuant` per plane) and pass
`quant.plane[plane]` into `apply_qindex_step()` at lines 758-762 and
`dequant_inverse_qindex()` at lines 765-784.  Those two functions are the exact
sites where the current uniform `quant_step_` enters forward quantization
(`compute_tx`, lines 797/872/965) and inverse dequantization (line 978).

`src/encoder/m2/m2_frame.cpp:145-150` (`frame_quant_step()`),
`:301` (`quant_step_ = frame_quant_step(...)`), and
`:205-232` (`compute_tx()`'s forward `quant_factor` and §7.14.4 dequant loop)
are the M2 equivalents.  Replace `int quant_step` with the two plane-specific
q values and use `dcdeltaq::quantize_block()` / `dequantize_block()` or the same
formulas.  `src/encoder/m2/m2_frame.cpp:748` (`sample_step`) and the
`estimate_tx_bits()` model at `:260` should use the plane's AC q for luma and
the plane's DC/AC q for chroma.

## Limits

* 8-bit qindex domain only (`MaxQ = 255`); the 9-bit `bit_depth` reader form
  parses `base_q_idx` but the quantizer arithmetic is the 8-bit path.
* No TIP frames (`TipFrameMode == TIP_FRAME_AS_OUTPUT`).
* QM support consumes an existing `qm::QmFrameState`; this module does not
  write `quantizer_matrix_obu()` or user QM payloads.
* The search's rate model is a proxy, not the §8.3 entropy coder; its
  `lambda_scale` is calibrated for that proxy.
* The search requires width/height multiples of 8 and 4:2:0 input.
* `get_q()` clips `qindex + delta` to `[1, MaxQ]` and keeps the §7.14.2
  `qindex == 0 && delta <= 0 -> 64` special case.
