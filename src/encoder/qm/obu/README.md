# QM OBU state machine (§5.13 / §6.12)

Encoder-side mirror of the AV2 quantizer-matrix OBU and the decoder state
variables it updates. This directory is additive: it does not edit
`src/encoder/qm/qm.{h,cpp}` or any M1/M2 header writer. The exact merge is
described at the end.

Normative sources (spec/AV2_Spec_v1.0.0.pdf):

| § | Item |
|---|---|
| §5.2.1 | `open_bitstream_unit()` / Annex B.2 `leb128` framing |
| §5.2.2 | `obu_header()`: extension flag, `obu_type`, `obu_tlayer_id`, `obu_mlayer_id`, `obu_xlayer_id` |
| §5.2.3 | `trailing_bits()` |
| §5.4.11 | `user_defined_qm()` / `Fundamental_Tx_Size[3]` |
| §5.13 | `quantizer_matrix_obu()` |
| §5.18.6.2 | `setup_qm_params()` |
| §6.4.11 | user-QM semantics (`quant_delta` range, no zero entries) |
| §6.12 | QM OBU semantics (`QmSeen`/`QmProtected`/`QmDataPresent`/layer ids) |
| §6.5 + §5.5 | temporal delimiter clears `QmProtected` |

## Files

| File | Purpose |
|---|---|
| `qm_obu.h` / `qm_obu.cpp` | OBU framing, §5.13 payload writer/reader, §5.18.6.2 frame-header writer/reader, `QmState` |
| `qm_obu_goldens.h` | pinned golden bytes and frame-header bit strings |
| `qm_obu_test.cpp` | 413 checks: round-trips, goldens, state, conformance rejections |
| `build_test.sh` | standalone build/run (optional ASan+UBSan) |
| `qm_obu_gate.cpp` | bit-splice helper: injects QM OBUs + `using_qmatrix=1` into a one-frame M1 IVF |
| `qm_obu_gate.sh` | unit tests + `dav2d` default-matrix check + `avmdec` OBU checks |

## API

```cpp
#include "encoder/qm/obu/qm_obu.h"

using namespace av2enc::qm::obu;

// 1. Encoder builds one or more OBUs between coded frames.
QuantizerMatrixObu obu;
obu.bit_map = 1u << 3;                 // f(15); level 3
obu.chroma_info_present = true;        // numPlanes = 3
obu.levels[3].is_default = false;      // qm_is_default_flag = 0
std::memset(&obu.levels[3].user, 32, sizeof(obu.levels[3].user));

std::vector<std::uint8_t> bytes;
std::string error;
ObuHeader header;                       // type 22, tlayer/mlayer as needed
if (!make_quantizer_matrix_obu(obu, &header, &bytes, &error)) { /* ... */ }

// 2. Drive the state machine in stream order.
QmState state;
state.on_temporal_delimiter();          // QmProtected[*] = 0
state.apply_obu(header, obu, &error);   // QmSeen/Protected/DataPresent/layers
state.on_key_frame(/*first_picture_in_tu=*/true, /*NumPlanes=*/3);
                                       // QmSeen = 0, then reset_qm()
bool present = state.data_present(3);   // QmDataPresent[3]
const UserQmLevel* matrix = state.user_qm(3);

// 3. Write the frame header fields.
FrameQm qm;
qm.using_qmatrix = true;
qm.qm_y[0] = 3; qm.qm_u[0] = 3; qm.qm_v[0] = 3;
qm.qm_uv_same_as_y[0] = true;
BitWriter writer;
write_frame_qm_params(writer, qm, /*NumPlanes=*/3,
                      /*segmentation_enabled=*/false, &error);
```

Reading mirrors the writers:

* `read_quantizer_matrix_payload()` parses exactly the bytes after the OBU
  header and enforces §5.2.3 trailing bits.
* `parse_obu_sequence()` splits an Annex B byte sequence into
  `{header, payload, raw}` entries.
* `read_frame_qm_params()` takes `separate_uv_delta_q` from the surrounding
  `quantizer_params()` (it is coded before `setup_qm_params()`) and reports the
  exact syntax bit count.
* `apply_obu_bytes()` parses one complete QM OBU and applies it.

## State semantics implemented

`QmState` has 15 entries (`NUM_CUSTOM_QMS = 15`, levels 0..14) with:

| Variable | Meaning |
|---|---|
| `qm_seen` | `QmSeen[level]`; cleared at each `frame_header_info()` |
| `qm_protected` | `QmProtected[level]`; cleared by a temporal delimiter, set by every QM OBU |
| `qm_data_present` | `QmDataPresent[level]`; `0` for `qm_is_default_flag == 1` |
| `qm_num_planes` | `QmNumPlanes[level]` |
| `qm_mlayer_id` / `qm_tlayer_id` | `QmMLayerId[level]` / `QmTLayerId[level]`, `-1` after a reset |
| `has_user_qm` / `user` | last `user_defined_qm()` matrices for the level |

Rules:

* **Start of bitstream** — every `QmProtected`, `QmSeen`, `QmDataPresent` is 0
  and both layer ids are `-1` (§6.12).
* **Temporal delimiter** — `QmProtected[level] = 0` for all levels. `QmSeen`
  and `QmDataPresent` are untouched. The multi-OBU window is *not* reset (a TD
  is not a coded frame).
* **Frame header** — `QmSeen[level] = 0` and the "between coded frames" QM-OBU
  window restarts.
* **`reset_qm()`** — called by `on_key_frame(first_picture_in_tu, planes)` for
  CLK/OLK. Only levels with `QmProtected == 0` lose `QmDataPresent`, keep
  `-1` layer ids and take the new `QmNumPlanes`. Protected levels survive a key
  frame. The SWITCH/RAS branch of `reset_qm()` depends on
  `MLayerPresenceMap`; this module models a single embedded layer, so callers
  that emit those OBU types must not reuse `on_key_frame()`.
* **`qm_bit_map == 0`** — sets `QmProtected = 1`, `QmDataPresent = 0`,
  `QmMLayerId = QmTLayerId = -1` for all levels. Rejected unless it is the
  first QM OBU in the current window.
* **Bits set in `qm_bit_map`** — set `QmSeen = QmProtected = 1`, store the OBU's
  `obu_mlayer_id`/`obu_tlayer_id` and `NumPlanes`, then set
  `QmDataPresent = !qm_is_default_flag`. A level already in `QmSeen` is
  rejected with *"specified twice between coded frames"*.
* **User data** — present iff `qm_is_default_flag == 0`; only the fundamental
  extents are coded (8x8, 8x4, 4x8) and a zero coefficient is rejected.

## Conformance rejection coverage

`qm_obu_test.cpp` and `qm_obu_gate --selftest` reject:

* a second `qm_bit_map == 0` OBU between coded frames;
* a level specified twice in that window (including inside one frame window);
* `qm_bit_map` bit 15 (not representable by f(15));
* user-QM coefficient 0 (§6.4.11);
* `quant_delta` outside `[-128, 127]` on read;
* missing trailing one bit / non-zero trailing bits;
* truncated OBU header, leb128, payload, or frame-header fields;
* `qm_uv_same_as_y == 1` with `qm_u != qm_y` or `qm_v != qm_y`;
* `qm_v != qm_u` when `separate_uv_delta_q == 0`;
* `pic_qm_num_minus_1 != 0` when segmentation is disabled, or outside f(2);
* a non-QM OBU type passed to `apply_obu()`.

## Pinned goldens

`qm_obu_goldens.h` pins complete Annex B OBUs:

| Golden | Content |
|---|---|
| `kResetChroma` | `qm_bit_map = 0`, chroma present |
| `kResetMono` | `qm_bit_map = 0`, mono |
| `kLevel0DefaultChroma` | level 0, default |
| `kLevel5DefaultMono` | level 5, default, mono |
| `kLevel5User32Chroma` | level 5, all-32 identity user matrix, chroma |
| `kLevels0And14Default` | levels 0 and 14, both default |
| `kLevel3User32Mono` | level 3, identity, mono |
| `kLevel2DefaultMonoNoChroma` | level 2, default, no chroma info |
| `kLevel7DefaultMlayer2Tlayer1` | level 7, default, mlayer 2 / tlayer 1 |

plus the §5.18.6.2 bit strings `kFrameOffBits`, `kFrameY5UvSameBits`,
`kFrameY3U7Bits`, `kFrameSeg2SeparateBits`, `kFrameMonoY14Bits`.

## Tests and gate

```
./build_test.sh                 # 413 checks, 0 failures
SANITIZE=1 ./build_test.sh      # 413 checks, 0 failures (ASan+UBSan)
./qm_obu_gate.sh
```

Gate output on this host:

```
qm obu: 413 checks, 0 failures
qm_obu selftest: OK
dav2d: using_qmatrix default matrices OK (b5ca… != b28f…)
avmdec: level 0 identity == base
avmdec: level 5 identity == base
avmdec: level 14 identity == base
avmdec: reset-identity == base (identity matrix)
avmdec: multi == base (identity matrix)
avmdec: level 0 default OBU == no-OBU default, differs from base
avmdec: level 5 default OBU == no-OBU default, differs from base
avmdec: level 14 default OBU == no-OBU default, differs from base
qm obu gate: OK
```

The gate splices a §5.18.6.2 `using_qmatrix = 1` header into an M1 still
picture, because M1/M2 currently hardcode `using_qmatrix = 0`. The pinned
`dav2d` rejects OBU type 22, so the default-matrix check uses the no-OBU path
where the frame header selects the level and `QmDataPresent` stays 0. `avmdec`
parses QM OBUs and checks:

* all-32 user matrices are the identity, so identity/reset/multi variants must
  decode bit-identically to the base stream;
* `qm_is_default_flag = 1` must decode identically to the implicit default
  (no OBU) and differently from the base, proving the QM scaling is applied.

## Exact merge into `qm` and `m1`/`m2_headers`

The merge below is intentionally not applied here (file ownership).

### 1. `src/encoder/qm/qm.h` / `qm.cpp`

* Add `#include "encoder/qm/obu/qm_obu.h"` and `obu::QmState qm_state;` to
  `QmFrameState`. Keep `qm_data_present[]` as the fast dequant mirror, or
  replace its uses with `qm_state.data_present(level)`.
* `make_core_params()` (`qm.cpp`, around lines 240–285): source
  `core_params.qm_data_present` from `qm_state.data_present(seg_level)` and
  `core_params.user_qm` from `qm_state.user_qm(seg_level)`. This makes
  `qm_is_default_flag` and `reset_qm()` visible to §7.14.4 dequantization.
* `write_setup_qm_params()` (`qm.cpp`, around line 326): keep the signature but
  convert `QmFrameState` to `obu::FrameQm` (`using_qmatrix`,
  `pic_qm_num_minus_1`, `qm_y/u/v`, `qm_uv_same_as_y[i] =
  (qm_u[i] == qm_y[i] && qm_v[i] == qm_y[i])`, `separate_uv_delta_q`) and call
  `obu::write_frame_qm_params()`. That adds the missing reader and the
  `qm_uv_same_as_y` representation while keeping the existing call sites.
* Replace the low-level body of `make_quantizer_matrix_obu()` with
  `obu::make_quantizer_matrix_obu()` so `qm_is_default_flag`, the f(15)
  validation and the §5.4.11 inference paths share one implementation.

### 2. `src/encoder/m1/m1_headers.h` / `m1_headers.cpp`

* `M1Config` (line ~58): add `obu::FrameQm qm;` and optionally
  `std::vector<obu::QuantizerMatrixObu> qm_obus;`.
* `write_frame_header_syntax()` line 177: replace
  `writer.put_bit(false); // using_qmatrix` with
  `obu::write_frame_qm_params(writer, c.qm, 3, /*segmentation_enabled=*/false,
  &error)`. The function returns `size_t` and has no error channel, so either
  (a) validate `c.qm` at `M1Config` construction time, or (b) render the QM
  fields into a scratch `BitWriter` first and copy the bits into `writer`.
* `assemble_m1_still_picture()` line 284: after `seq`, before `clk`, append
  `obu::make_quantizer_matrix_obu()` for each planned level. The OBU must
  precede the frame header that references the level (§7.3.8.9). Also emit a
  reset OBU (`qm_bit_map = 0`) when the level set must survive `reset_qm()`.
* `src/encoder/m1/m1_still.cpp`: populate `config.qm` from the chosen segment
  levels, and mirror the same `obu::QmState` updates when the stream plan is
  built: TD → `on_temporal_delimiter()`; each QM OBU → `apply_obu()`; each CLK
  → `on_key_frame(first_picture_in_tu, 3)`. Set `SegQMLevel = 15` for lossless
  segments.

### 3. `src/encoder/m2/m2_headers.h` / `m2_headers.cpp`

* `M2Config` (line 21): add `obu::FrameQm qm;`.
* `write_frame_header_syntax()` line 284: same replacement as M1. Use
  `NumPlanes = 3` for 4:2:0.
* `m2_frame.cpp` stream assembly (lines ~1007 and ~1044): emit the QM OBU(s)
  between `m2::make_sequence_header_obu()` and `make_frame_obu()`.
* `src/encoder/m2/deltaq/m2_deltaq.cpp:39` currently forces
  `out.using_qmatrix = false`; make that derive from `qm_state` /
  `config.qm`. This is the natural M2 owner of `QmDataPresent`.
* Other hardcoded sites to update when their gates gain QM:
  `src/encoder/m2/post/tiled_frame.cpp:445`,
  `src/encoder/m2/multiref/gate/compound_stream.cpp:234`,
  `src/encoder/pipeline/tiled_still_gate.cpp:263`.

### 4. `src/bitstream/m0_headers.cpp`

* Line 310 (`using_qmatrix` in the M0 still header): replace with
  `obu::write_frame_qm_params()`. `bitstream/m0_headers.h` must then carry an
  `obu::FrameQm` in `StillPictureConfig`, and `m0_still.cpp` must emit the QM
  OBU after the sequence header.

### 5. Decoder/harness state

* Any tool that walks a real stream should call, in order:
  `on_temporal_delimiter()` for type 2, `apply_obu()` for type 22,
  `on_key_frame(first_picture_in_tu, NumPlanes)` for CLK/OLK, and
  `on_frame_header()` for every other frame OBU before its tile group.
* Keep `QmDataPresent` and `UserQm` derived from `QmState`; do not duplicate
  the conformance bookkeeping in the header writers.
