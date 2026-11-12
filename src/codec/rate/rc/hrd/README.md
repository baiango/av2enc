# HRD / timing: sequence timing, decoder model info and buffer removal timing

Status: **spec-only component.** This directory is not wired into the root
`CMakeLists.txt`, `src/main.cpp`, the M1/M2 header writers or the frame loop.
It produces the AV2 v1.0.0 syntax for the decoder model and a deterministic
schedule planner, but it does not change any reference stream. The integration
sites are listed below.

Normative references (AV2 Bitstream & Decoding Process Specification v1.0.0,
commit `38f28e7`):

| Section | Syntax / semantics |
|---|---|
| §4.11.3 `uvlc()`, §4.11.10 `rg(n)` | descriptors |
| §5.2.1/§5.2.2/§5.2.3 | OBU framing, header, trailing bits |
| §5.4.1 | `seq_initial_display_delay_present_flag`, `decoder_model_info_present_flag`, `num_units_in_decoding_tick`, `seq_decoder_model_info_present_flag` |
| §5.4.12 | `timing_info()` |
| §5.4.13 | `seq_decoder_model_info()` |
| §5.12 | `buffer_removal_timing_obu()` |
| §5.15 | `content_interpretation_obu()` (the carrier of `timing_info()`) |
| §6.4.12 / §6.4.13 / §6.11 | semantics |
| §7.3.3 / §7.3.6 / §7.3.7 | OBU ordering |
| Annex E.5.4 / E.7.2 / E.7.3 / E.7.4 | decoding-schedule model, delay consistency, overflow, underflow |

## Files

| File | Purpose |
|---|---|
| `hrd.h` / `hrd.cpp` | Syntax structs, writers/readers, validation, OBU framing and the sequence-header timing splice |
| `hrd_buffer.h` / `hrd_buffer.cpp` | Clock derivation, two-pass/VBV → `br_time` planner, per-operating-point planner, `FramePlan` adapter |
| `bit_reader.h` | Self-contained MSB-first bit reader (inverse of `BitWriter` for f/uvlc/rg) |
| `hrd_test.cpp` | Pinned bit goldens, round-trips, validation, clock arithmetic, planner tests |
| `hrd_stream_gate.cpp` | M2 stream rewrite + pinned `dav2d` gate |
| `build_and_test.sh` | Standalone build + test driver |

## Build and test

```sh
cd src/encoder/rc/hrd
./build_and_test.sh
HRD_WERROR=1   ./build_and_test.sh
HRD_SANITIZE=1 ./build_and_test.sh
HRD_GATE=0     ./build_and_test.sh   # unit tests only
CXX=g++-16     ./build_and_test.sh
```

No CMake, no CUDA, no external dependencies. The gate links the unmodified M2
CPU reference (`src/encoder/m2/`) and looks for `tools/bin/dav2d` or
`tools/bin/dav2d-macos` (or `$DAV2D`).

Local result on the macOS dev host with Apple clang 21:

- **510 checks, 0 failures** with Apple clang 21 and Homebrew g++ 16,
  `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`, and with ASan +
  UBSan.
- The M2 decoder gate passes: decoded MD5
  `ea90e2e6a9be74a5fda88faa0c433134` for both baseline and timing-carrying
  streams, pixels byte-identical, encoder recon exact.

## Syntax mapping

### `timing_info()` (§5.4.12)

```text
num_units_in_display_tick f(32)
time_scale                f(32)
equal_picture_interval    f(1)
if equal_picture_interval:
  num_ticks_per_picture_minus_1 uvlc()
```

`TimingInfo` is emitted inside a Content Interpretation OBU (§5.15), which is
otherwise written in full:

```cpp
ContentInterpretation ci;
ci.ci_timing_info_present_flag = true;
ci.timing.num_units_in_display_tick = 1000;
ci.timing.time_scale = 30000;
ci.timing.equal_picture_interval = true;
ci.timing.num_ticks_per_picture_minus_1 = 0;

std::vector<std::uint8_t> obu;
std::string error;
hrd::build_content_interpretation_obu(ci, obu, &error);
```

### `seq_decoder_model_info()` (§5.4.13)

```text
decoder_buffer_delay uvlc()
encoder_buffer_delay uvlc()
low_delay_mode_flag  f(1)
```

Both delays are in units of 1/90000 s. In strict mode (`low_delay_mode_flag =
0`) underflow is not allowed; in low-delay mode late removal is permitted and
the planner reports it.

### Sequence-header timing block (§5.4.1)

```text
seq_initial_display_delay_present_flag f(1)
if present: seq_initial_display_delay_minus_1 f(4)
decoder_model_info_present_flag f(1)
if present:
  num_units_in_decoding_tick f(32)
  seq_decoder_model_info_present_flag f(1)
  if present: seq_decoder_model_info()
```

`write_sequence_timing()`/`read_sequence_timing()` cover exactly this block.
`splice_sequence_timing()` rewrites an existing M2/M0 sequence header in place:
it parses the §5.4.1 prefix through `seq_cropping_window_present_flag`, checks
that the baseline has no timing block, inserts the new block byte-for-byte, and
copies the remaining syntax bits unchanged. This is the zero-edit integration
path while M1/M2 own their header writers.

### `buffer_removal_timing_obu()` (§5.12)

```text
br_ops_dependent_flag f(1)
if dependent:
  br_ops_id f(4)
  br_ops_cnt[br_ops_id] f(3)
  for i in 0..br_ops_cnt-1:
    br_decoder_model_present_op_flag[br_ops_id][i] f(1)
    if present: br_time_op[br_ops_id][i] rg(4)
else:
  br_time rg(4)
```

Both forms round-trip. `BufferRemovalTiming` carries either the single
`br_time` or the per-operating-point `op_times` vector. `rg(4)` limits every
time to **0..511** DecCT ticks; `validate_brt_tick_ranges()` enforces it and the
planner errors with an actionable message if a schedule would exceed it.

## Clock-tick arithmetic

The planner uses integer units of `1 / (90000 * time_scale)` seconds:

```text
DecCT units = num_units_in_decoding_tick * 90000
DispCT units = num_units_in_display_tick * 90000        (per display tick)
frame interval units = (num_ticks_per_picture_minus_1 + 1)
                       * num_units_in_display_tick * 90000
DecCT ticks per picture = ceil(frame interval units / DecCT units)
delay in 1/90000 s = delay * time_scale units
```

`derive_clock()` exposes the exact values and whether the display interval is
an integral number of DecCT ticks. Tested cases include 30 fps (`1000/30000`
with `num_units_in_decoding_tick = 1000` → exactly 1 tick), 24000/1001, and
`num_units_in_decoding_tick = 900` (2 ticks, inexact).

## Buffer planner

`plan_buffer_removal()` implements Annex E.5.4 decoding-schedule mode:

```text
ScheduledRemoval[0] = DecoderBufferDelay / 90000
ScheduledRemoval[i] = ScheduledRemoval[PrevRap] + br_time[i] * DecCT
FirstBitArrival[i] = max(LastBitArrival[i-1], ScheduledRemoval[i] - T)
T = (DecoderBufferDelay + EncoderBufferDelay) / 90000
LastBitArrival[i] = FirstBitArrival[i] + CodedBits[i] / BitRate
```

The schedule starts at the nominal display interval and is pushed later in
whole DecCT ticks whenever `LastBitArrival` would exceed the scheduled removal.
`br_time` is relative to the applicable random access point (the previous RAP
for a RAP frame, the current RAP otherwise), exactly as in §6.11.

Checks enforced:

- `frames[0]` must be a RAP; `equal_picture_interval` must be 1.
- Strict mode: `ScheduledRemoval[i] >= LastBitArrival[i]` for every frame.
- Low-delay mode: underflow is permitted; removal is deferred to the first
  DecCT tick after the DFG is complete, and one tick of decode time is reserved
  between successive actual removals.
- `T * BitRate <= BufferSize`; a simulated occupancy sweep rejects overflow.
- At a RAP after the first, `ScheduledRemoval - LastBitArrival[i-1] >=
  DecoderBufferDelay` so §E.7.2 holds.
- `DecoderBufferDelay != 0` when BRT is signalled.
- `br_time <= 511`.

`HrdBufferConfig::allow_delay_adjustment = true` (the default) raises the
decoder/encoder delays to the smallest whole-1/90000-s values that fit frame 0
and the largest frame. The plan reports the values to signal in
`seq_decoder_model_info()`. With adjustment disabled the planner fails rather
than emitting an illegal `br_time`.

For per-operating-point BRT, every timeline must cover the same coded frames.
The planner first resolves one common delay pair across all operating points,
then emits one `BufferRemovalTiming` per frame with
`br_ops_dependent_flag = 1` and one `br_time_op` entry per point.

## Two-pass / VBV adapter

```cpp
#include "encoder/rc/hrd/hrd_buffer.h"

std::vector<av2enc::hrd::FrameBits> frames;
std::string error;
// Prefer the real coded payload bits; kTarget uses FramePlan::target_bits,
// which is the effective budget after the twopass VBV cap.
if (!av2enc::hrd::frames_from_twopass_plans(
        plans, actual_bits, av2enc::hrd::TwopassBitSource::kActual,
        frames, &error)) { /* ... */ }

av2enc::hrd::HrdBufferConfig cfg;
// Derive the clock/rate/buffer fields from the existing two-pass config and
// TVBV state, then override anything the integration knows better.
if (!av2enc::hrd::hrd_config_from_twopass(tp_config, rc.vbv(), 1000, cfg,
                                          &error)) { /* ... */ }
cfg.decoder_buffer_delay = 70000;          // 1/90000 s
cfg.encoder_buffer_delay = 20000;

av2enc::hrd::BufferTimingPlan buffer;
if (!av2enc::hrd::plan_buffer_removal(cfg, frames, buffer, &error)) { /* ... */ }
for (std::size_t i = 0; i < frames.size(); ++i) {
  av2enc::hrd::BufferRemovalTiming brt =
      av2enc::hrd::br_time_for_frame(buffer, i);
  std::vector<std::uint8_t> obu;
  av2enc::hrd::build_buffer_removal_timing_obu(brt, obu, &error);
  // write obu before the frame OBU in coded output frame unit i
}
```

`actual_bits` is the coded frame payload/DFG size, not container bytes. If
the two-pass value excludes OBU framing, add the OBU header bytes before
planning: Annex E counts the complete decodable frame group. The gate includes
the OBU headers in the DFG size and iterates the BRT construction to a fixed
point because the BRT OBU itself is part of the DFG.

## Validation

- `validate_brt_against_ops(brt, brt_xlayer_id, ops_summaries)`:
  - `br_ops_id` must match an **active** OPS (`ops_cnt != 0`) for the same
    `obu_xlayer_id`;
  - `br_ops_cnt` must equal `ops_cnt[obu_xlayer_id][br_ops_id]`;
  - a reset OPS (`ops_cnt == 0`) does not satisfy the correspondence rule.
- `validate_brt_tick_ranges()`: every present `br_time`/`br_time_op` in
  `0..511`, `br_ops_id <= 15`, `br_ops_cnt` in `1..7`.
- `sequence_timing_valid()`: `num_units_in_decoding_tick > 0` when the decoder
  model is present; `seq_decoder_model_info_present_flag` requires
  `decoder_model_info_present_flag`; `seq_initial_display_delay_minus_1 <= 15`.
- `timing_info_valid()`: both clock fields non-zero.
- `content_interpretation_valid()`: `f(2)` scan type, `rg(2)` colour idc,
  chroma sample position <= 5, aspect ratio idc <= 16 or 255.
- `validate_temporal_unit_order()`: TD first; LCR/OPS/atlas allowed before
  SEQ; CI requires a preceding SEQ; at most one SEQ/CI/BRT; CI and BRT before
  the frame OBU(s); one frame OBU per unit in the M2 shape.

## Pinned bit goldens

`hrd_test.cpp` pins these byte strings (no generated tables):

| Vector | Bytes |
|---|---|
| `timing_info` 30 fps (`1000/30000`, 1 tick) | `00 00 03 e8 00 00 75 30 c0` |
| `timing_info` 24000/1001 | `00 00 03 e9 00 00 5d c0 c0` |
| `seq_decoder_model_info(70000, 20000, 0)` | `00 00 88 b8 80 01 38 84` |
| sequence timing block (`delay=2`, `nutdt=1000`, 70000/20000/0) | `94 00 00 0f a2 00 01 11 71 00 02 71 08` |
| BRT non-dependent `br_time = 0` (full OBU) | `02 3c 02` |
| BRT non-dependent `br_time = 511` (full OBU) | `06 3c 7f ff ff ff 7c` |
| BRT dependent `ops_id=3, cnt=2, times=0,5` (full OBU) | `04 3c 9a 82 58` |
| CI OBU with `timing_info` 24000/1001 | `0b 60 04 00 00 03 e9 00 00 5d c0 d0` |
| M2 64x64 baseline sequence header OBU | `13 04 80 75 00 2a bf fe 00 28 00 6e 23 80 00 00 10 1b 80 82` |
| spliced 64x64 sequence header (full decoder model) | `20 04 80 75 00 2a bf fe 94 00 00 0f a2 00 01 11 71 00 02 71 08 02 80 06 e2 38 00 00 01 01 b8 08 20` |
| spliced 64x64 sequence header (`seq_decoder_model_info` absent) | `18 04 80 75 00 2a bf fe 94 00 00 0f a0 01 40 03 71 1c 00 00 00 80 dc 04 10` |

The same vectors are also tested as writer→reader round-trips.

## Decoder gate result

`hrd_stream_gate` encodes a 3-frame 64×64 M2 stream, then builds:

1. the baseline stream;
2. a **spec stream** with the full sequence timing block (including
   `seq_decoder_model_info()`), a CI OBU with `timing_info()`, and one BRT OBU
   per frame;
3. a **decoder-compatible stream** with the same `timing_info`, the same
   `decoder_model_info_present_flag`/`num_units_in_decoding_tick`, but without
   the nested `seq_decoder_model_info()` and without BRT.

Observed with the pinned `dav2d` (`f9dafbc`, nightly 2026-11-10):

- The decoder-compatible stream decodes under `--strict 1 --threads 1` with an
  exit code of 0. Decoded Y4M MD5 is
  `ea90e2e6a9be74a5fda88faa0c433134`, identical to the baseline; the decoded
  frames still match the encoder reconstruction byte-for-byte. `dav2d`
  **does** process `timing_info`: `--frametimes` changes when the CI timing
  block is present.
- The pinned `dav2d` **rejects `OBU_BUFFER_REMOVAL_TIMING` (type 15)** as an
  unknown OBU (`Unknown OBU type 15`). It cannot be used to gate BRT.
- The pinned `dav2d` **does not parse the spec v1.0.0
  `seq_decoder_model_info()` block**: the stream with that block fails in
  sequence/frame-header parsing (with `low_delay_mode_flag = 1` it reports
  `Overrun in OBU bit buffer`). This is a decoder/spec gap, not a change to the
  syntax implemented here.

Therefore BRT conformance is gated by:

1. the buffer model (no strict-mode underflow, no overflow, `br_time <= 511`,
   OPS correspondence, low-delay behavior), and
2. self-parse of the full spec stream (OBU order and exact `br_time` values).

The decoder gate itself is pixel invariance + recon on the compatible stream.
A future `dav2d` that implements §5.12/§5.4.13 will exercise the full stream;
the gate automatically takes the stronger path if `dav2d` accepts it.

## Exact integration sites

These are the edits the integration pass owns. This component does not make
them.

1. **`src/encoder/m2/video_headers.h`** — add
   `av2enc::hrd::SequenceTimingConfig sequence_timing;` to `VideoConfig` (and
   include `encoder/rc/hrd/hrd.h`).

2. **`src/encoder/m2/video_headers.cpp`, `write_sequence_header()`** — currently
   writes (lines 126–127):

   ```cpp
   writer.put_bit(false);  // seq_initial_display_delay_present_flag
   writer.put_bit(false);  // decoder_model_info_present_flag
   ```

   Replace those two bits with:

   ```cpp
   if (!av2enc::hrd::write_sequence_timing(writer, c.sequence_timing, &error)) {
     return 0;  // or propagate a status
   }
   ```

   `write_sequence_header()` returns `std::size_t`; the integration must
   validate `sequence_timing_valid()` before calling it (or widen the return
   path). `make_sequence_header_obu()` already appends the extensible-OBU
   `obu_extension_flag` and trailing bits after the syntax bits, so no other
   change is needed there.

   Alternative zero-edit path used by the gate:
   `hrd::splice_sequence_timing(baseline_obu, config, out, &error)` before
   `make_obu`, if the caller prefers to keep the baseline writer untouched.

3. **`src/encoder/m2/video_frame.cpp`, unit assembly** — the key unit is built
   around lines 1007–1016 (`TD + SEQ + frame`) and P units around 1044–1050
   (`TD + frame`). The required order is:

   ```text
   TD, [SEQ], [CI (first coded extended layer unit only)],
   [BRT], frame OBU(s)
   ```

   Insert the CI OBU only in the first unit. Insert one BRT OBU per unit
   immediately before the frame OBU. Compute its `br_time` from the actual
   coded DFG bits. Because the BRT OBU is part of the DFG, iterate the unit
   size + plan to a fixed point (the gate does this in
   `build_injected_units()`). If the stream is planned offline, the sequence
   header can be emitted from the plan's `decoder_buffer_delay`/
   `encoder_buffer_delay` before the first frame.

4. **`src/encoder/rc/twopass/` integration point** — after
   `TwoPassController::end_frame(actual_payload_bits)` (or from the offline
   `FramePlan` vector), call `frames_from_twopass_plans()` and
   `plan_buffer_removal()` as shown above. For streaming, the sequence header
   needs fixed delays; use `allow_delay_adjustment = true` with a conservative
   first pass, or re-plan the prefix and keep the sequence-level delay pair
   constant (the spec requires a constant sum across RAPs).

5. **`src/main.cpp` / CLI (owned by the integration pass)** — add options such
   as `--timing`, `--decoder-model`, `--brt`, `--fps`, `--bitrate`, and
   `--buffer-size`; the defaults must stay off so reference streams do not
   change. `base_q_idx`/M2 behaviour is untouched.

6. **`CMakeLists.txt` (owned by the integration pass)** — add
   `src/encoder/rc/hrd/hrd.cpp` and `src/encoder/rc/hrd/hrd_buffer.cpp` to the
   CPU target. No CUDA source is needed; this is per-sequence/per-frame policy,
   not a hot pixel stage.

7. **OPS correspondence** — `src/encoder/svc/svc_framing.cpp` already writes
   OPS OBUs and knows `ops_id`/`ops_cnt`. Feed its summaries to
   `validate_brt_against_ops(brt, obu_xlayer_id, summaries)` before emitting a
   per-operating-point BRT.

## Limitations / next steps

- The buffer model covers Annex E.5.4 decoding-schedule scheduling and the
  arrival/overflow/underflow inequalities. It does **not** simulate
  presentation times, frame-buffer reuse, `TimeToDecode`/`InitialDisplayDelay`
  (E.5.6/E.5.7/E.6), or the full decoder model error-code path. Those need the
  level tables and a presentation schedule.
- No OPS/level-table writer. `BufferSize` and `BitRate` are caller-supplied
  (level `MaxBufferSize`/`MaxBitrate` × `BitrateProfileFactor`).
- No padding or frame dropping. If a target cannot be scheduled, the planner
  fails with a specific error instead of emitting an illegal BRT.
- `rg(4)` gives only 511 ticks per RAP. Long GOPs at a small
  `num_units_in_decoding_tick` must either raise the tick size or insert RAPs;
  the planner reports the required tick count.
- The pinned `dav2d` cannot validate BRT or the spec `seq_decoder_model_info()`;
  see the decoder gate section. Do not weaken the syntax to match the decoder.
- No quality/BD-rate claim. This component only makes the timing stream legal.
