# M2 integration sites

No M2 file was edited by this component. These are the exact places a future
M2 integration (or a separate random-access unit assembler) must touch, with
the current line numbers and the data that has to cross the boundary.

The simplest integration keeps M2's tile payloads and replaces only the unit
framing; that is what `ras_gate.cpp` does. A tighter integration moves the
decision into `encode_video_sequence()`.

## 1. Sequence header: enable long-term ids

`src/encoder/m2/video_headers.cpp:163`

```cpp
writer.put_bits(0U, 3U);  // long_term_frame_id_bits
```

Replace the literal with `c.long_term_frame_id_bits` and add that field to
`VideoConfig` (`src/encoder/m2/video_headers.h:23-40`). `0` keeps every existing
stream byte-identical; `1` makes LongTermId 0 codable. With the current writer
`ra::make_sequence_header_obu(config)` produces a byte-identical OBU when the
field is 0, as asserted by `random_access_test.cpp`.

The sequence header also fixes `max_mlayer_id = 0`, which is required for the
RAS derived-refresh branch in §5.18.2 and for `OBU_CLOSED_LOOP_KEY`
`refresh_frame_flags = allFrames`.

## 2. Frame header: add SWITCH/RAS syntax

`src/encoder/m2/video_headers.cpp:224-310` — `write_frame_header_syntax()`
currently has only the keyframe and inter branches:

```cpp
if (!key) {
  writer.put_bit(true);  // frame_is_inter = 1
}
```

A SWITCH/RAS frame must instead code:

```cpp
if (obu_type == ObuType::kSwitch || obu_type == ObuType::kRas) {
  writer.put_bit(restricted_prediction_switch);  // replaces frame_is_inter
  frame_type = SWITCH_FRAME;                     // no primary-ref bits later
}
if (frame_type == KEY_FRAME && long_term_frame_id_bits > 0)
  writer.put_bits(long_term_id_plus_1, long_term_frame_id_bits);
if (obu_type == ObuType::kRas && long_term_frame_id_bits > 0) {
  writer.put_bits(num_key_ref_frames, 3);
  for (i < num_key_ref_frames)
    writer.put_bits(ref_long_term_id[i], long_term_frame_id_bits);
}
// frame_size_override_flag = 1 is inferred for SWITCH_FRAME
// signal_primary_ref_frame and disable_cross_frame_cdf_init are absent
// refresh_frame_flags: SWITCH explicit, RAS derived (no bits)
// explicitRefFrameMap = 1 is inferred (no frame_explicit_ref_frame_map bit)
// skip_mode_present is absent for SWITCH_FRAME
```

Rather than duplicating the writer, an integration can call
`ra::write_frame_header_info()` for the SWITCH/RAS frames and M2's
`write_tile_group_header()` for key/regular frames. Both start at the same
point in the OBU payload (`is_first_tile_group` first), so the callers are
interchangeable.

`src/encoder/m2/video_headers.cpp:328-339` — `make_frame_obu()` picks the OBU type
from `VideoFrameHeader::keyframe`:

```cpp
const ObuType type =
    header.keyframe ? ObuType::kClosedLoopKey : ObuType::kRegularTileGroup;
```

Add an `obu_type`/`frame_kind` field (or a new overload) so the same tile
payload can be framed as `OBU_SWITCH` (10) or `OBU_RAS_FRAME` (21). The OBU
writer already codes raw 5-bit types, so no container change is needed.

## 3. Reference state and frame planning

`src/encoder/m2/video_frame.cpp:988-1005` — sequence/key setup. Add
`long_term_frame_id_bits`, and for the long-term keyframe:

```cpp
key_header.long_term_id_plus_1 = 1;   // LongTermId = 0
key_header.immediate_output_frame = true;
```

Keep `refresh_frame_flags = allFrames` so every slot starts with LongTermId 0.
The key tile payload is unchanged; only the frame header gains the
`long_term_id_plus_1` field.

`src/encoder/m2/video_frame.cpp:1027-1050` — the P-frame loop. This is where a
random-access decision goes:

```cpp
const VideoReconFrame& reference = result.recon.back();   // line 1028
InterFrameEncoder encoder(frames[i], reference, ...);  // line 1028
...
VideoFrameHeader header;
header.order_hint = i;      // line 1040
header.ref_slot = 0;        // line 1041
header.refresh_frame_flags = 1;  // line 1042
```

Needed changes:

- Maintain `ra::RefState` alongside `result.recon`.
- For a SWITCH at frame `i`, use the existing tile payload (encoded against
  slot 0) and `ra::plan_switch_frame(config, refs, candidate, true)`. Choose
  `refresh_frame_flags` so the long-term slot (slot 0) is **not** refreshed;
  refresh another slot (e.g. `1 << 1`) or the RAS will no longer find the
  long-term frame.
- For a RAS at frame `j`, the payload must predict from the long-term slot, not
  from `result.recon.back()`. Two options:
  1. encode a separate P payload against `result.recon[lt_frame_index]` (this
     is what the gate does with a two-frame `[frame0, frame2]` encode), or
  2. add a reference override to `InterFrameEncoder` so the loop can keep one
     reference frame per frame kind.
  Then call `ra::plan_ras_frame(config, refs, candidate, ras_long_term_ids)`.
  `plan_ras_frame` returns `kReencodeRequired` when the payload's slot is not
  a named long-term slot.
- `ra::validate_header()` before emitting the OBU; it catches the §6.17/§7.4.5
  reference violations, tlayer constraints and dimension ratio.
- Update `ra::RefState` with `ra::update_reference_state()` after each frame so
  the next plan sees the correct `RefLongTermId`/`RefValid`/restricted state.

## 4. Tile payload: no syntax change

`src/encoder/m2/video_frame.cpp:394`

```cpp
plan.inter_frame = !keyframe_;
```

SWITCH and RAS have `FrameIsIntra = 0`, so the existing
`inter_frame_mode_info` writer (`src/encoder/m2/video_syntax.cpp:1904`,
`code_is_inter()`) is already correct. `assemble_video_frame()` at
`src/encoder/m2/video_syntax.cpp:2045` does not need a new mode. The payload
assumes:

- single reference `LAST` with `ref_frame_idx[0]` = the candidate slot;
- `skip_mode_present = 0` (SWITCH/RAS have no such bit; M2 P writes 0);
- the same `disable_cdf_update = 1` / `disable_cross_frame_cdf_init = 1`
  default-CDF path (SWITCH/RAS use `PRIMARY_REF_NONE`, which also initialises
  default CDFs).

## 5. Unit assembly

`src/encoder/m2/video_frame.cpp:1008-1020` (key unit) and `1044-1050` (P unit)
build `TD + [SEQ] + frame OBU`. A random-access unit uses:

```cpp
unit = TD + our_sequence_header
     + frame_obu(type=CLK,   key_header,    key_tile)
// next temporal unit
unit = TD + frame_obu(type=SWITCH, switch_header, switch_tile)
// next temporal unit
unit = TD + frame_obu(type=RAS,   ras_header,    ras_tile)
```

The sequence header is emitted once (key unit), exactly as M2 does. The gate
uses `ra::make_temporal_delimiter_obu()` + `ra::make_sequence_header_obu()` +
`ra::make_frame_obu()` for this.

## 6. Reconstruction bookkeeping

`src/encoder/m2/video_frame.cpp:901-925` (`to_recon_frame`), `1017` (key recon)
and `1052` (P recon) are unchanged. For random-access frames append:

- SWITCH recon = the P encoder's recon for that frame (predicted from the
  long-term/previous slot);
- RAS recon = the recon of the payload encoded against the long-term slot.

The order must match the decoded output order. The gate compares
`{seq3.recon[0], seq3.recon[1], seq2.recon[1]}`.

## 7. Validation gate

`src/encoder/m2/video_gate.cpp`:

- `write_ivf()` at line 257 and `compare_y4m()` at line 136 are reusable as-is.
- `run_phase()` at line 273 is the closest template: encode, write IVF, run
  `dav2d`, compare. Add a phase that builds the three-frame SWITCH/RAS stream
  with this component.
- The pinned decoder command in the gate is
  `dav2d --strict 1 --threads 1 -q -i in.ivf -o out.y4m`. Note the existing M2
  gate uses `--threads 0`; the random-access gate deliberately pins
  `--threads 1` and `--strict 1`.

## 8. Build lists

If the component becomes part of the default build, add:

```text
src/encoder/inter/random_access/random_access.cpp
src/encoder/inter/random_access/ras_gate.cpp   # gate target only
src/encoder/inter/random_access/random_access_test.cpp  # test target only
```

to the relevant `CMakeLists.txt` target(s). This component deliberately did not
edit `CMakeLists.txt`; `build_and_test.sh` compiles the same source list
directly. The gate links the M2 core: `video_frame.cpp`, `video_syntax.cpp`,
`video_headers.cpp`, `merge.cpp`, `partition.cpp`, `intra_modes.cpp`,
`tx_variants.cpp`, `tx_select.cpp`, `av2_entropy.cpp`, `me.cpp`, `mc.cpp`,
`bitwriter.cpp`, `basic_headers.cpp`, `io/y4m.cpp`, `io/ivf.cpp`.

## 9. Caveats

- The pinned dav2d build (`f9dafbc`) reads explicit refresh bits for
  `OBU_RAS_FRAME` instead of deriving them. Set
  `SequenceConfig::ras_codes_refresh_frame_flags = true` for that decoder; the
  bits carry the §5.18.2 derived mask, so the normative state is unchanged.
- The pinned decoder rejects `num_total_refs = 0`. A RAS must therefore name
  and reference at least one long-term slot. Set `long_term_frame_id_bits > 0`
  and keep the long-term frame in a slot the SWITCH does not refresh.
- SWITCH `restricted_prediction_switch = 1` marks every currently valid slot
  restricted. The named long-term slot remains valid and keeps its
  `RefLongTermId`, so a following RAS may still use its samples; it may not use
  its other state.
