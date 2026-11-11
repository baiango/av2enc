# SVC layer pipeline

Status: **encoder-side component + standalone compose/gate tool.** It is not
wired into `src/main.cpp`, the root `CMakeLists.txt`, or the M1/M2 core. It
builds a validated layer plan, deterministic emission order, OBU/layer framing,
and Annex F.3.1/F.3.2 selection/extraction. The gate in `gate_dav2d.sh` composes
real 2- and 3-layer streams from independent single-layer encodes and validates
them with `dav2d` and `avmdec`.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(commit `38f28e7`), §5.2.2, §5.4.1, §5.8, §5.10, §5.11, §7.3.6, §7.3.7,
Annex F.3.1 and Annex F.3.2. No AVM, SVT, clave, or av2-turbo encoder code was
consulted. `dav2d` / `avmdec` are used as black-box decoders only.

## Files

| File | Purpose |
|---|---|
| `svc_config.h` / `.cpp` | Spatial/temporal/embedded layer config, per-layer sequence headers, global LCR, cumulative operating points, deterministic layer list, frame assignment, retention maps, selection resolution |
| `svc_pipeline.h` / `.cpp` | Per-temporal-unit emission order, custom GOP ordering, inter-layer reference policy, reference-before-target validation |
| `svc_framing.h` / `.cpp` | §5.2.2 OBU header, leb128 framing, LCR/OPS payload writers, stream parsing, layered composition, Annex F.3.2 extraction |
| `svc_test.cpp` | Standalone unit tests (no CUDA, no encoder core, no decoder) |
| `svc_mux.cpp` | `svc_mux` CLI: rewrap single-layer IVF units into a layered IVF, or extract one operating point |
| `build_and_test.sh` | Build/run the unit tests, optional ASan+UBSan and `-Werror` |
| `gate_dav2d.sh` | Real 2/3-layer decoder gate: compose, `dav2d`, `avmdec --select-ops`, extraction |
| `INTEGRATION.md` | Exact core wiring steps, decoder support matrix, and blockers |

## Build and test

```sh
cd src/encoder/svc
./build_and_test.sh
WERROR=1 ./build_and_test.sh
SANITIZE=1 ./build_and_test.sh
```

No CMake and no CUDA. The test binary links the pure `bitstream/bitwriter.cpp`
and the existing `tiles/svc_layers.cpp` validator (read-only; it is not edited).

The decoder gate needs an existing encoder build plus decoder binaries:

```sh
AV2ENC=/path/to/av2enc ./gate_dav2d.sh
```

It skips with exit code 77 when the encoder or `dav2d` is missing.

## Layer model

```cpp
av2enc::svc::LayerConfig config;
config.spatial_layers = {{0, 64, 64}, {1, 128, 128}, {2, 256, 256}};
config.temporal_layers = 2;   // tlayer 0..1
config.embedded_layers = 1;   // mlayer 0
config.seq_lcr_id = 1;        // associated global LCR id
config.single_picture_header = false;
config.still_picture = false;

av2enc::svc::LayerPlan plan;
std::string error;
if (!av2enc::svc::build_layer_plan(config, plan, error)) {
  // error says exactly which invariant failed
}
```

`build_layer_plan()` produces:

- `sequences[i]`: one sequence-header config per spatial layer, with that
  layer's `max_width`/`max_height`, `max_mlayer_id = M-1`,
  `max_tlayer_id = T-1`, and `seq_lcr_id`. `single_picture_header_flag`
  forces one spatial layer, `M = T = 1`, and `seq_lcr_id = 0` (§5.4.1).
  Set `LayerConfig::dependencies` to override the default reflexive/transitive
  dependency chain; custom maps are validated before use.
- `lcr`: one global LCR (`obu_xlayer_id = 31`) with `lcr_xlayer_map`, a complete
  `mlayer_map`/`tlayer_map` per extended layer, and embedded descriptors.
  `dependent_xlayers` is **false**: this pipeline declares no inter-layer
  prediction.
- `ops`: one cumulative operating point per spatial layer. OP `i` selects
  xlayers `0..i` with all `M` embedded and all `T` temporal layers. `ops_cnt` is
  `f(3)`, so more than 7 default points (more than 7 spatial layers) is
  rejected instead of silently truncated.
- `layers`: the deterministic coded-layer list, ordered by xlayer, then mlayer,
  then tlayer, all ascending.
- `operating_points`: decoder-facing `{op_index, xlayer_map, counts, label}`.

`validate_layer_plan()` re-runs every sequence/LCR/OPS check plus plan-level
consistency (one SH per spatial layer, cumulative OPs, exact layer order).

### Spatial-only path (no inter-layer prediction)

AV2 keeps a separate decode context per extended layer (`load_xlayer_context()`
/ `save_xlayer_context()`), and this encoder does not implement cross-layer
sample, MV, or coefficient prediction. Every extended layer is independently
decodable. `ReferencePolicy` makes that explicit:

```cpp
av2enc::svc::ReferencePolicy policy;  // all cross-layer prediction = false
av2enc::svc::reference_allowed(plan, policy, target, reference, error);
```

A cross-xlayer reference is rejected with a message naming the spatial-only
path. Lower same-xlayer tlayers and (same-xlayer, same-mlayer) temporal
references are allowed; forward-in-time and cross-mlayer references are
rejected by default. `validate_frame_references()` additionally requires the
reference to be emitted before the target in the schedule.

### Deterministic assignment and order

```cpp
av2enc::svc::FrameSource source{/*tu=*/0, /*tlayer=*/1, 128, 128, /*index=*/3};
std::vector<av2enc::tiles::svc::LayerId> assigned;
av2enc::svc::assign_layers(plan, sources, assigned, error);
```

A source frame maps to the **unique** spatial layer with the same coded size,
at the requested embedded/temporal layer. Two spatial layers with the same
resolution make the assignment ambiguous and are rejected; submission order
never breaks a tie. (The plan and `compose_layered_stream` still support
same-resolution layers, e.g. quality scalability; only the size-based helper
needs distinct sizes. A scheduler that keeps several same-size layers must
assign by explicit layer id.)

`plan_emission_order(plan, temporal_units, order, error)` emits, per temporal
unit:

```
TD (global)
[global LCR, global OPS]        first temporal unit only
for each xlayer ascending:
  sequence header                first temporal unit only
  for each mlayer ascending:
    for each tlayer ascending:
      frame
```

`order_frames()` takes a GOP scheduler's `FrameRequest` list and sorts it by
`(temporal unit, xlayer, mlayer, tlayer, non-output before output)`. Duplicate
layer/TU requests and absent layers are rejected.

### Selection and retention

```cpp
av2enc::svc::LayerSelection selection;
selection.mode = av2enc::svc::SelectionMode::kOperatingPoint;  // --oppoint N
selection.op_index = 1;
av2enc::tiles::svc::RetentionMap retention;
av2enc::svc::resolve_selection(plan, selection, retention, error);
```

Modes: `kAllLayers` (`--alllayers`), `kOperatingPoint` (`--oppoint N`), and
`kUpToSpatialLayer` (`--spatial-layers N`, cumulative). `resolve_selection()`
uses Annex F.3.1 retention expansion, marks the global OBU bucket
(`GLOBAL_XLAYER_ID`, m0/t0) when the plan carries global metadata, and then
checks dependency closure. Retaining a layer while dropping one of its
dependencies, or retaining a layer absent from the plan, is rejected.

## OBU / layer framing

`encode_obu_header()` implements §5.2.2 exactly:

```
obu_header_extension_flag f(1)
obu_type                  f(5)
obu_tlayer_id             f(2)
if (obu_header_extension_flag) {
  obu_mlayer_id f(3)
  obu_xlayer_id f(5)
}
```

The extension byte is emitted only for a non-zero mlayer/xlayer. TD and MSDO
infer `GLOBAL_XLAYER_ID = 31` with no extension byte. Example headers:

| OBU | Bytes |
|---|---|
| TD (global) | `08` |
| sequence header, x0 | `04` |
| global LCR/OPS (x31) | `C0 1F` |
| regular tile group, x2/t1 | `9D 02` |

`frame_obu()` prepends `leb128(num_bytes_in_obu)` (Annex B.2). The LCR/OPS
writers emit the complete §5.8 / §5.10 payload and the extensible-OBU tail
(`obu_extension_flag = 0` + trailing bits). Extraction copies retained OBUs
byte-for-byte, so an all-layers extraction is identical to the input.

### `svc_mux` (ungated bridge)

```sh
svc_mux -o out.ivf --layer base.ivf --layer enh.ivf [--layer ...]
        [--no-lcr] [--no-ops] [--extract OP | --oppoint OP] [--alllayers]
```

`--extract` and `--oppoint` are aliases (Annex F.3.2 extraction); `--alllayers`
keeps the full stream and is the default.

Each input must be a complete single-layer temporal unit from the existing
encoder: exactly one TD, one sequence header, and one coded frame. `svc_mux`
rewrites only OBU layer ids, adds one global TD/LCR/OPS, and optionally runs
Annex F.3.2 extraction. It never re-encodes sequence headers or tile payloads.
It rejects inputs that already carry LCR/OPS, missing/duplicate frames, wrong
layer order, and `--temporal`/`--embedded` greater than 1 (one unit cannot fill
extra m/t layers).

`svc_mux` is the bridge that makes a real decoder gate possible **without core
edits**. It is not a substitute for per-layer encoding of video.

## Tests

`svc_test.cpp` covers:

- 2-layer (x0/x1, t0/t1) and 3-layer (x0..x2, t0..t2) plans; per-layer sequence
  header dimensions and limits; deterministic layer list.
- spatial + embedded (M=2) + temporal (T=2) configuration and maps.
- cumulative OP extraction (OP0/OP1/OP2), `kAllLayers`,
  `kUpToSpatialLayer`, retention counts and global OBU retention.
- custom dependency maps (accepted when reflexive/transitive, rejected
  otherwise) and Annex F.3.1 transitive retention expansion (t2 → t1 → t0).
- retention closure failures (missing dependency, absent layer, selected xlayer
  with no retained pair).
- frame-to-layer assignment (unique, ambiguous, unmatched, tlayer overflow,
  duplicate TU/layer).
- emission order (TD/LCR/OPS/SEQ/frame sequence, per-TU TD, xlayer ordering,
  custom GOP order, non-output before output, duplicate/absent/empty rejection).
- reference policy (cross-xlayer default rejection, declared-but-unimplemented
  dependency, lower temporal reference, forward reference, cross-mlayer,
  self-reference, reference-after-target).
- framing (header bytes for TD/SEQ/LCR/tile group, extension byte, truncated
  headers, leb128 boundaries, size/truncation errors, parse round-trip).
- LCR global and local payload bit layouts, OPS global and local payload bit
  layouts, `ops_reset_flag`, per-point sizes and map bits.
- compose 2/3 layers, deterministic repeat, no-metadata compose, extraction
  equality for all layers and OP subsets, compose rejections.
- single-picture still plan constraints.

All checks pass with `-Werror` and under ASan+UBSan.

## Decoder gate results

`gate_dav2d.sh` composes:

- 2 layers: 64x64 QP90 + 64x64 QP120 (same-resolution xlayers).
- 3 layers: 64x64 QP90 + 64x64 QP120 + 640x360 QP90 (mixed resolution).

and compares decoder output against the concatenation of independently decoded
single-layer inputs.

| Check | Result |
|---|---|
| `dav2d` 2-layer `--alllayers 1` | byte-exact |
| `dav2d` 3-layer `--alllayers 1` | byte-exact |
| `avmdec` 2-layer `--all-layers` | byte-exact |
| `avmdec` 3-layer `--all-layers` | byte-exact |
| `avmdec --select-ops=0,0/1/2` | byte-exact per operating point |
| `svc_mux --extract 0/1/2` → `dav2d` and `avmdec` | byte-exact |
| `avmdec` on `--no-lcr` 2-layer stream | **fails** (exit 1, no output) |

Decoder support notes, from the pinned builds:

- **`dav2d` `f9dafbc` decodes multiple xlayers and honours `--alllayers`, but
  does not implement OBU type 16 (LCR) or type 18 (OPS).** It prints
  `Unknown OBU type 16` / `18` and ignores them. `--oppoint 0/1/2` on the
  3-layer stream produces the same 357,888-byte full output for all three
  values, so `dav2d --oppoint` cannot gate OPS selection.
- **`avmdec` (macOS bundle, `tools/bin/avm-macos/avmdec`)** implements LCR/OPS,
  `--select-ops=ops_id,op_index`, and `--all-layers`. It decodes every composed
  and extracted stream byte-exactly, and it rejects a multi-xlayer stream that
  lacks LCR/MSDO. This is the OPS gate for the pipeline.

The composed stream reuses the existing still-picture sequence headers
(`single_picture_header_flag = 1`, `seq_lcr_id = 0`), so those headers are not
formally associated with the global LCR. `avmdec` accepts this and selects the
OPs correctly; the real layered path sets `seq_lcr_id` and `max_mlayer_id` /
`max_tlayer_id` from the plan (see `INTEGRATION.md`).

## Blockers

1. **No core integration.** `src/main.cpp` still emits one still picture and the
   M0/M1 sequence-header writers pin `single_picture_header_flag = 1`. A
   multi-frame, multi-layer real-video stream needs a non-single-picture
   sequence header and the scheduler wiring in `INTEGRATION.md`. This component
   does not edit those files.
2. **No inter-layer prediction.** Layers are independent. The LCR deliberately
   declares no `lcr_num_dependent_xlayer_map`, so no cross-layer gain is
   claimed or possible in this path.
3. **`dav2d` cannot validate LCR/OPS.** Use `avmdec --select-ops` for that;
   keep `dav2d` for layer decode/framing. Do not report an SVC result from
   `dav2d --oppoint` on this build.
4. **Default OP count.** More than 7 spatial layers needs an explicit OP list
   (not exposed yet); the plan rejects instead of truncating.
5. **`svc_mux` is still-only.** It composes one-frame units; multi-frame video
   goes through the encoder scheduler, not the mux.
