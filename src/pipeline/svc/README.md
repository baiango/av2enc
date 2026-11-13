# SVC layer pipeline

Status: **wired into the product CLI** (`--video --svc-layers`). `src/main.cpp`
parses the layer flags, `svc_stream.cpp` builds the plan and drives per-layer
encode + §7.3.6 emission, and the root `CMakeLists.txt` links the module. The
gate in `gate_cli_svc.sh` decodes real 2-layer streams with `dav2d` and
`avmdec`, checks decoded picture counts and compares every picture against the
encoder reconstruction; `gate_dav2d.sh` still covers the still-picture
`svc_mux` composition path.

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`), §5.2.2, §5.4.1, §5.8, §5.10, §5.11, §6.8.2,
§7.3.6, §7.3.7, Annex A.2, Annex F.3.1 and Annex F.3.2. No AVM, SVT, clave, or
av2-turbo encoder code was consulted. `dav2d` / `avmdec` are used as black-box
decoders only.

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

### Cross-layer prediction: embedded refinement, not xlayers

AV2 keeps a separate decode context per extended layer (`load_xlayer_context()`
/ `save_xlayer_context()`), and **cross-xlayer prediction cannot be signalled at
all** in a conforming v1.0.0 stream: §6.8.2 requires
`lcr_dependent_xlayers_flag == 0`. `--svc-inter-layer` is therefore a named
rejection and `validate_layer_plan()` rejects any LCR that declares a
dependency. There is no reference-resampling process to implement.

The decodable cross-layer form is **embedded (mlayer) refinement**
(`--svc-embedded N`). Within one xlayer:

* mlayer 0 is the normal CLK + P chain and is independently decodable;
* mlayer `m > 0` is coded by `encode_video_refinement_sequence()`: every picture
  is a regular inter picture that predicts from mlayer `m-1`'s decoded picture
  at the same display index, references slot `m-1` and refreshes slot `m`;
* the sequence header raises `max_mlayer_id` and `seq_max_mlayer_cnt`, and
  Annex A caps the ladder at three embedded layers for 8-bit 4:2:0
  (`seq_profile_idc` 0/1/2), which is why `--svc-embedded 4..8` is rejected;
* the base chain pins slot 0 (`VideoOptions::fixed_write_slot`) so the rotating
  base slot never collides with a refinement slot — a collision makes an m0
  frame reference a slot holding m1, which §6.17.2 forbids.

`ReferencePolicy` encodes the rule: cross-xlayer is always rejected with the
§6.8.2 citation, cross-mlayer is allowed for a higher mlayer, and forward
temporal references are rejected. `validate_frame_references()` additionally
requires the reference to be emitted before the target in the schedule.

### Hierarchical temporal layers

`--svc-temporal N` (N = 2..4) codes one closed hierarchical-P GOP of
`1 << N` frames per xlayer/mlayer (the product `inter::tlayers` planner), and
emits each frame into the temporal unit that owns its coding position with the
plan's `obu_tlayer_id`. All layers share the plan, so every coded output frame
unit in a TU carries the same `OrderHint` (§7.3.7). Because the plan clears
`monotonic_output_order_flag`, the global LCR sets
`lcr_doh_constraint_flag = 1`; `avmdec` rejects the stream otherwise
("lcr_doh_constraint_flag must be 1 when monotonic_output_order_flag is 0").
`--svc-temporal` and `--svc-embedded` cannot be combined: the embedded path
codes a flat chain.

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
formally associated with the global LCR. The real layered path sets
`seq_lcr_id` and `max_mlayer_id` / `max_tlayer_id` from the plan (see
`INTEGRATION.md`).

**Product CLI (wired).** `src/main.cpp` exposes
`--video --svc-layers x0=WxH[,x1=WxH,...]` (plus `--svc-lcr-id`,
`--svc-temporal 1..4`, `--svc-embedded 1..3`, `--oppoint`, `--alllayers`,
`--spatial-layers`, and the `--svc-inter-layer` named rejection).
`svc_stream.cpp` performs the INTEGRATION.md steps:

1. the video sequence header is parameterised by
   `seq_lcr_id`/`max_mlayer_id`/`max_tlayer_id`/`seq_max_mlayer_cnt_minus_1`
   (plus the §5.4.1 dependency-map rows);
2. `make_layer_config_record_obu()` / `make_operating_point_set_obu()` wrap the
   payload writers as OBU types 16/18;
3. each spatial layer is box-scaled and encoded as an independent video
   sequence with its own sequence header and OBU `xlayer_id`; embedded layers
   add the cross-mlayer refinement sequences described above;
4. temporal units follow the shared schedule (one TD per TU, global LCR/OPS
   and per-layer sequence headers in TU 0 only);
5. `resolve_selection()` + `extract_substream()` implement `--oppoint`,
   `--spatial-layers` and `--alllayers`;
6. `--video-recon` writes one reconstruction Y4M per (xlayer, mlayer) so the
   gate can compare every decoded picture sample-exactly.

`gate_cli_svc.sh` result on the pinned builds (27/27):

| Check | Result |
|---|---|
| 2-layer mixed-resolution `avmdec --all-layers` | byte-exact vs per-layer decodes |
| `avmdec --select-ops=0,0/0,1` | byte-exact cumulative OPs |
| `--oppoint 0` / `--spatial-layers 1` extraction | byte-exact |
| base layer vs plain `--video` encode of the scaled source | byte-exact |
| `--svc-temporal 2` (4-frame GOP, 2 xlayers) | 8/8 pictures, all == recon |
| `--svc-embedded 2` (3 TUs, 2 xlayers, 2 mlayers) | 12/12 pictures, all == recon |
| `--svc-embedded 2` mlayer 1 vs mlayer 0 | distinct refinement picture |
| `dav2d --strict 1` single-layer flat stream | 3/3 pictures |
| `dav2d --strict 1` single-layer `--svc-temporal 2` | 4/4 pictures |
| `dav2d --strict 1` single-layer `--svc-embedded 2` | **parse failure** (below) |

Measured embedded ladder (3 frames, x0 160x144, `--qp 90`, step 12):
mlayer 0 (QP 102) 36.95 dB PSNR-Y, mlayer 1 (QP 90) 45.95 dB PSNR-Y. The
refinement is a real quality step, not a duplicate picture.

Pinned-decoder limitations (`dav2d` `f9dafbc`, `avmdec` macOS build):

- Single-layer SVC streams decode fully under `dav2d --strict 1`, including the
  hierarchical temporal GOP. **Multi-xlayer streams that contain inter (P)
  frames do not**: `dav2d` emits fewer pictures than layers x TUs and still
  exits 0 (same-resolution 2-layer 3-TU: 4 pictures, mixed-resolution: 2), so
  an exit-code-only check is a false pass. Use `avmdec` and count pictures.
- `dav2d` cannot parse a sequence header with `max_mlayer_id > 0` ("Error
  parsing sequence header / Overrun in OBU bit buffer"), so embedded streams
  are `avmdec`-only. `avmdec` implements LCR/OPS, `--select-ops`,
  `--all-layers` and the mlayer dependency rules.
- `dav2d` ignores OBU type 16 (LCR) and type 18 (OPS) ("Unknown OBU type
  16/18"); its `--oppoint` has no effect on this build, so OPS selection is
  `avmdec`-only.

## Remaining limits

1. **Cross-xlayer prediction does not exist in AV2 v1.0.0.** §6.8.2 requires
   `lcr_dependent_xlayers_flag = 0`; the encoder rejects the flag and the
   `--svc-inter-layer` flag by name (see `docs/negative-results.md`).
   Compression across layers is only the embedded (mlayer) ladder.
2. **Temporal + embedded cannot be combined.** `--svc-temporal > 1` with
   `--svc-embedded > 1` is rejected: the refinement sequence would need a
   cross-mlayer reference schedule inside the hierarchical GOP. The flat chain
   and the GOP are each complete.
3. **Embedded ladder depth.** Annex A caps `seq_max_mlayer_cnt` at 3 for 8-bit
   4:2:0 (`seq_profile_idc` 0/1/2); `--svc-embedded 4..8` is rejected by name.
4. **`dav2d` cannot validate multi-xlayer inter, LCR/OPS, or
   `max_mlayer_id > 0`.** Use `avmdec` for those; keep `dav2d` for single-layer
   decode/framing.
5. **Default OP count.** More than 7 spatial layers needs an explicit OP list
   (not exposed yet); the plan rejects instead of truncating.
6. **`svc_mux` is still-only.** It composes one-frame units; multi-frame video
   goes through `svc_stream.cpp`, not the mux.
