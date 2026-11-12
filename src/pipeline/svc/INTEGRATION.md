# SVC layer pipeline — integration notes

This file is the hand-off for wiring `src/encoder/svc/` into the encoder core.
The component and its own tests are complete; no file outside
`src/encoder/svc/` was modified. The block below names the exact seams, the
values that must come from the layer plan, and the decoder gate that must pass
after each step.

## What already exists

- `LayerConfig` → `LayerPlan` (`build_layer_plan`) with per-layer sequence
  header configs, a global LCR, cumulative operating points, and a deterministic
  layer list.
- `plan_emission_order` / `order_frames` for per-temporal-unit ordering.
- `ReferencePolicy` for the spatial-only reference rules.
- OBU header encode/decode, leb128 framing, LCR/OPS payload writers, stream
  parsing, and Annex F.3.2 extraction.
- `svc_mux`: composes pre-encoded single-layer units into a layered IVF. This is
  the gate bridge, not the product path.
- Real decoder gate: `gate_dav2d.sh` passes 2- and 3-layer composition,
  `avmdec --select-ops`, and extraction. See the README table.

## Integration steps

### 1. CLI surface (`src/main.cpp`, CLI owner)

Add flags and parse them into `av2enc::svc::LayerConfig`:

| Flag | Meaning |
|---|---|
| `--svc-layers x0=WxH[,x1=WxH,...]` | spatial layers, ids contiguous from 0 |
| `--svc-temporal N` | `temporal_layers` (1..4) |
| `--svc-embedded N` | `embedded_layers` (1..8; product default 1) |
| `--svc-lcr-id N` | `seq_lcr_id` (1..7; 0 only for stills) |
| `--oppoint N` | select OP N for output/extraction |
| `--alllayers` | select every layer |
| `--spatial-layers N` | cumulative selection through xlayer N-1 |

Reject conflicting selections (`--oppoint` with `--alllayers`) and any
`--svc-*` flag combined with a still-picture path (`single_picture_header_flag`
cannot describe layers). Do not silently drop unknown layer flags.

### 2. Build the plan once per sequence

```cpp
av2enc::svc::LayerPlan plan;
std::string error;
if (!av2enc::svc::build_layer_plan(config, plan, error)) {
  // print `error`; do not emit a partial stream
}
```

The plan owns:
- `plan.sequences[x]`: max dimensions, `max_mlayer_id`, `max_tlayer_id`,
  `seq_lcr_id`, dependency maps for spatial layer `x`;
- `plan.lcr`: global LCR (tlayer maps, embedded descriptors, no xlayer
  dependency);
- `plan.ops`: cumulative OPS (one point per spatial layer).

### 3. Encode each spatial layer

Scale the source to each spatial layer and run the existing single-layer
encoder for that layer. Each layer gets its own sequence header with:

- `max_frame_width/height` = that spatial layer's coded size;
- `single_picture_header_flag = 0`, `still_picture = 0` (real video);
- `seq_lcr_id = plan.config.seq_lcr_id`;
- `max_tlayer_id = plan.config.temporal_layers - 1`;
- `max_mlayer_id = plan.config.embedded_layers - 1`;
- `seq_max_mlayer_cnt_minus_1 = max_mlayer_id` (or the SeqMaxMlayerCnt the
  scheduler actually uses; it must be ≤ `max_mlayer_id + 1` and ≥ the distinct
  mlayers emitted);
- `monotonic_output_order_flag` per the GOP scheduler.

The M2 writer already emits the non-single-picture form at
`src/encoder/m2/video_headers.cpp:98-110`, but pins `seq_lcr_id = 0`,
`max_tlayer_id = 0`, `max_mlayer_id = 0`. Parameterize those five values from
`plan.sequences[x]`; do not edit the M1/M0 still writers (they are forced by
`single_picture_header_flag = 1`). Add a `make_layer_config_record_obu()` and
`make_operating_point_set_obu()` wrapper next to `video_headers.*` (or in the
caller) that calls `svc::write_layer_config_record_payload()` /
`svc::write_operating_point_set_payload()` and `svc::frame_obu()`. The payload
writers already include the extensible-OBU tail.

For each coded frame, set the OBU header from the layer id:

```cpp
av2enc::svc::ObuHeader header;
header.type = av2enc::svc::ObuType::kRegularTileGroup;  // or CLK/OLK
header.xlayer_id = id.xlayer_id;
header.mlayer_id = id.mlayer_id;
header.tlayer_id = id.tlayer_id;
```

Sequence headers are layer-specific; keep `mlayer_id = 0`,
`tlayer_id = 0` for them (§7.3.6).

### 4. Emit temporal units

Use `plan_emission_order(plan, temporal_units, order, error)` for the simple
"every layer every TU" schedule, or `order_frames()` with a GOP scheduler's
`FrameRequest` list. The emission order is already the §7.3.6/§7.3.7 order:

```
TD(global)
global LCR, global OPS          (first temporal unit only)
for each xlayer ascending:
  sequence header               (first temporal unit only)
  for each mlayer ascending:
    non-output units
    output unit
```

Write exactly one `OBU_TEMPORAL_DELIMITER` per temporal unit
(`obu_xlayer_id = GLOBAL_XLAYER_ID`). Do not copy a TD from each layer's
single-layer unit. A multi-xlayer stream without an LCR (or MSDO) is rejected
by `avmdec`, even though `dav2d` tolerates it; emit the global LCR.

### 5. Frame assignment and references

If the scheduler produces `(width, height, tlayer)` sources, call
`assign_layers()` and reject on error. Ambiguous sizes are a configuration bug,
not something to resolve by submission order.

Use `reference_allowed()` with the default `ReferencePolicy` (all cross-layer
flags false). Do not emit an inter-layer reference: there is no encoder-side
cross-layer predictor and the LCR declares no xlayer dependency. Keep
`lcr.dependent_xlayers = false`; if a future design adds a predictor, the
dependency map, the LCR, and the reference validity check must change together.

### 6. Output selection

For an encode that should emit a subset:

```cpp
av2enc::svc::LayerSelection selection;
selection.mode = av2enc::svc::SelectionMode::kOperatingPoint;  // --oppoint
selection.op_index = requested_op;
av2enc::tiles::svc::RetentionMap retention;
if (!av2enc::svc::resolve_selection(plan, selection, retention, error)) { ... }
std::vector<std::uint8_t> sub;
if (!av2enc::svc::extract_substream(full_stream, retention, sub, error)) { ... }
```

`--alllayers` uses `kAllLayers` and returns the input byte-for-byte. The IVF
header stays the base-layer size; keep the timestamp as the temporal-unit
index.

### 7. Decoder gates per step

1. Single-layer still (existing gate): unchanged reference MD5s.
2. Compose two single-layer units with `svc_mux`, decode with `dav2d
   --alllayers 1` and `avmdec --all-layers`; both must equal the concatenated
   independent decodes.
3. Add the global LCR/OPS; `avmdec --select-ops=0,N` must equal the cumulative
   prefix. `dav2d` will log `Unknown OBU type 16/18` and ignore the metadata;
   that is expected for `f9dafbc`.
4. Only after 1–3, wire per-layer encode callbacks. Gate each new layer count
   and each temporal-layer count with the same comparison.
5. Run `tools/validate_dav2d.sh out.ivf` on every emitted stream. Keep
   `--strict 1`.

## What the pinned decoders support

| Feature | `dav2d` `f9dafbc` | `avmdec` (macOS bundle) |
|---|---|---|
| Multiple xlayers in one temporal unit | yes | yes |
| `--alllayers` / `--all-layers` | yes | yes |
| OBU type 16 (LCR) | **no** (`Unknown OBU type 16`) | yes |
| OBU type 18 (OPS) | **no** (`Unknown OBU type 18`) | yes |
| `--oppoint N` / `--select-ops=ops,op` | no effect on this build | yes, byte-exact |
| Multi-xlayer stream without LCR/MSDO | decodes | **rejects** |

Consequences:

- `dav2d --oppoint` is not an SVC gate on this build. Use
  `avmdec --select-ops`.
- Keep the global LCR in every multi-xlayer stream even though `dav2d` ignores
  it; `avmdec` and the spec require it.
- The composed gate stream reuses still-picture sequence headers
  (`seq_lcr_id = 0`). `avmdec` accepts this; the real path must set
  `seq_lcr_id` from the plan so the LCR is formally associated.

## Remaining blockers / limits

- **Core writer.** A real multi-frame, multi-layer stream needs the M2 (or a new
  video) sequence-header writer parameterized for layers, plus scheduler
  wiring. That is outside `src/encoder/svc/`.
- **Inter-layer prediction.** Not implemented and not claimed. The spatial-only
  rules are enforced at reference-check time.
- **Default OP cap.** More than 7 spatial layers exceeds `ops_cnt` `f(3)`; the
  plan rejects. Add an explicit OP list before supporting >7.
- **Temporal-layer presence.** `plan_emission_order` emits every configured
  layer every TU. A hierarchical GOP must use `order_frames()` and must keep
  each layer's `OrderHint` consistent within a temporal unit
  (`lcr_doh_constraint_flag = 1` is not written today).
- **Still path.** `svc_mux` is one-frame-per-layer and must not be shipped as
  the encoder. It exists to make the decoder gate real before core wiring.
- **No CMake target.** The component builds via `build_and_test.sh`; the root
  `CMakeLists.txt` was intentionally not edited. Add `src/encoder/svc/*.cpp`
  and link `tiles/svc_layers.cpp` + `bitstream/bitwriter.cpp` when the
  integration owner wires it.

---

## Wave-3 status (pipelines integration): not wired

SVC is **not** wired into the product CLI. The module is complete and the
standalone gate is kept, but an opt-in CLI path would have to ship `svc_mux`,
which is explicitly a one-frame-per-layer gate bridge, not an encoder. The
composed stream cannot yet be produced end-to-end from the product encoder.

Current standalone-gate state (`AV2ENC=build-int/av2enc
src/pipeline/svc/gate_dav2d.sh`):

- the gate exits **77 (skip)** at the first `svc_mux` compose step:
  `svc_mux` must associate the global LCR, which needs a sequence header with
  `single_picture_header_flag = 0` and `seq_lcr_id != 0`; the only layer sources
  the encoder can produce today are `--still` single-picture headers;
- composing with `--no-lcr` avoids that error, but `avmdec` rejects a
  multi-xlayer stream without LCR/MSDO (and `dav2d` ignores LCR/OPS), so that is
  not a pass either.

Missing plumbing, in dependency order:

1. **Non-single-picture sequence header.** `pipeline/video/video_headers.cpp`
   hardcodes `seq_lcr_id = 0`, `max_mlayer_id = 0`, `max_tlayer_id = 0` and
   `seq_max_mlayer_cnt_minus_1 = 0`. Parameterize those from
   `svc::LayerPlan::sequences[x]` and add `still_picture = 0`,
   `single_picture_header_flag = 0`.
2. **LCR/OPS OBU wrappers.** Call `svc::write_layer_config_record_payload()` /
   `svc::write_operating_point_set_payload()` and wrap them with
   `svc::frame_obu()` (types 16/18) once per stream, in the first temporal unit
   after the TD. `dav2d f9dafbc` ignores both; `avmdec` requires them.
3. **Per-layer encode.** Scale the source to each spatial layer, run the video
   encoder per layer with the layer's own sequence header and OBU
   `xlayer_id`/`mlayer_id`/`tlayer_id`, and emit temporal units through
   `plan_emission_order()` (one TD per TU, xlayer ascending, mlayer ascending).
4. **Selection/extraction.** `resolve_selection()` + `extract_substream()` for
   `--oppoint`/`--alllayers`; `dav2d --oppoint` is a no-op on this build, so the
   OPS gate must use `avmdec --select-ops=ops_id,op`.
5. **Gate.** Re-run `gate_dav2d.sh` with the real per-layer sources and require
   byte-exact equality with the concatenation of independent decodes, plus
   `avmdec --select-ops` per operating point. Until then, do not claim SVC in
   the CLI or a compression gain (layers are delivery packaging, not a BD-rate
   tool).
