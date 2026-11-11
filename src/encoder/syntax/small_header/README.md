# encoder/syntax/small_header — small sequence-header features

Spec-only implementation of five AV2 v1.0.0 sequence-header features that the
M1/M2 writers currently hardcode off. This directory is **additive**: it does
not modify `src/encoder/m1/m1_*`, `src/encoder/m2/m2_*`,
`src/bitstream/`, `src/main.cpp`, or `CMakeLists.txt`.

Normative reference:
**AV2 Bitstream & Decoding Process Specification v1.0.0**
(2026-05-28, commit `38f28e7`).

| Feature | Spec |
|---|---|
| Cropping window + display size | §5.4.1, §5.18.4.4 |
| `reduce_pb_aspect_ratio` / `MaxPbAspectRatio` | §5.4.3, §6.19.3.2, §6.19.4.1 |
| `df_par_bits_minus_2` | §5.4.10, §5.18.5.2 |
| Sequence tile info / `tile_params()` | §5.4.2, §5.18.7.2–§5.18.7.7 |
| Sequence segments / `seg_info(MaxSegments)` | §5.4.4, §5.4.9, §5.18.7.1 |

## Files

| File | Purpose |
|---|---|
| `small_header.h` | public API and POD configuration/result types |
| `small_header.cpp` | exact writers, independent readers, validation |
| `small_header_test.cpp` | 364 checks: pinned bit goldens, round-trips, errors |
| `gate.cpp` | rewrites a real M1 stream and emits feature-on IVF streams |
| `build_and_test.sh` | standalone build, tests, pinned-`dav2d` gate |

No CLI is added. The only entry points are the `av2enc::syntax::small_header`
functions and the test/gate binaries built by the script.

```sh
./src/encoder/syntax/small_header/build_and_test.sh
SANITIZE=1 WERROR=1 ./src/encoder/syntax/small_header/build_and_test.sh
SKIP_GATE=1 ./src/encoder/syntax/small_header/build_and_test.sh
AV2ENC=/path/to/av2enc DAV2D=/path/to/dav2d \
  ./src/encoder/syntax/small_header/build_and_test.sh
```

The build script uses the existing `build/av2enc` (or `build-m1/av2enc`) only
to produce real M1 tile payloads. It does not rebuild or reconfigure them.

## API

All writers validate before writing and return `false` with a human-readable
`std::string* error` on bad input. All readers are independent parsers of the
same MSB-first bitstream that `av2enc::BitWriter` produces.

### §5.4.1 cropping

```cpp
struct CroppingWindow { bool present; uint32_t left, right, top, bottom; };
struct DisplaySize { uint32_t width, height; };

bool validate_cropping_window(window, max_frame_width, max_frame_height, error);
bool write_cropping_window(BitWriter&, const CroppingWindow&, error);
bool read_cropping_window(BitReader&, CroppingWindow*, error);
bool derive_display_size(window, max_w, max_h, frame_w, frame_h,
                         DisplaySize*, error);
```

`derive_display_size()` is §5.18.4.4 `compute_image_size()`: offsets are scaled
by `FrameWidth / maxFrameWidth`, and the result is `CropWidth × CropHeight`.
Offsets are constrained to `max_frame_width_minus_1` /
`max_frame_height_minus_1`, and an empty derived rectangle is rejected.

### §5.4.3 PB aspect ratio

```cpp
struct PbAspectRatio {
  bool reduce;
  uint8_t max_pb_aspect_ratio_log2_minus_1;  // 0 or 1 when reduce
};
int max_pb_aspect_ratio(const PbAspectRatio&);  // 2, 4, or 8

bool write_pb_aspect_ratio(BitWriter&, const PbAspectRatio&, error);
bool read_pb_aspect_ratio(BitReader&, PbAspectRatio*, error);

bool pb_leaf_aspect_allowed(int bw4, int bh4, int max_ratio);
bool pb_partition_none_allowed_by_aspect(int bw4, int bh4, int max_ratio);
bool pb_non_none_partition_allowed_by_aspect(int bw4, int bh4, int max_ratio);
```

`MaxPbAspectRatio` is 8 when `reduce_pb_aspect_ratio == 0`, otherwise
`1 << (max_pb_aspect_ratio_log2_minus_1 + 1)` (2 or 4). The helpers mirror
§6.19.3.2 `is_partition_allowed()`: a too-thin candidate forbids
`PARTITION_NONE`; an 8:1-or-worse candidate forbids every partition. §6.19.4.1
requires every decoded leaf to satisfy both `bw4 <= bh4 * MaxPbAspectRatio` and
`bh4 <= bw4 * MaxPbAspectRatio`.

The existing partition modules already expose `max_pb_aspect_ratio` in their
geometry (`partition`, `partition_ext`, `sdp`, `sb128`). Feed that field from
this config instead of hardcoding 8; the header and the partition search then
agree.

### §5.4.10 deblocking-filter parallel bits

```cpp
struct DfPar { uint8_t df_par_bits_minus_2; int bits() const; /* 2..5 */ };

bool validate_df_par(const DfPar&, error);
bool write_df_par(BitWriter&, const DfPar&, error);
bool read_df_par(BitReader&, DfPar*, error);
bool df_delta_q_representable(int value, const DfPar&);
bool validate_df_delta_q(const DfPar&, const DfDeltaQSet&, error);
```

`dfParBits = df_par_bits_minus_2 + 2` is the width of each
`df_delta_q[i] f(dfParBits)` in §5.18.5.2. The coded unsigned value maps to
`DfDeltaQ[i] = df_delta_q[i] - (1 << (dfParBits - 1))`, so the representable
range is `[-(1 << (dfParBits-1)), (1 << (dfParBits-1)) - 1]`. An encoder that
emits any non-zero `DfDeltaQ` must therefore choose a field wide enough for it;
`validate_df_delta_q()` rejects overflow. When `df_delta_q_present[1] == 0`,
`DfDeltaQ[1]` is inferred equal to `DfDeltaQ[0]`, which is already representable.

### §5.4.2 / §5.18.7 tile info

```cpp
enum class SbSize { k64x64 = 64, k128x128 = 128, k256x256 = 256 };
SbSize sequence_superblock_size(use_256x256_superblock,
                                use_128x128_superblock);  // get_seq_sb_size()

struct TileParamsRequest {
  int tile_cols_log2 = -1;   // -1 = level minimum
  int tile_rows_log2 = -1;
  std::vector<int> width_in_sbs;   // non-empty -> non-uniform spacing
  std::vector<int> height_in_sbs;
};
struct TileParams { /* starts, counts, log2s, sb_shift, limits, bits */ };

bool write_tile_params(...);
bool read_tile_params(...);

struct SequenceTileConfig {
  bool present, allow_change;
  TileParamsRequest request;
  TileParams params;
};
bool write_sequence_tile_config(...);
bool read_sequence_tile_config(...);

struct FrameTileRequest { bool reuse_sequence; TileParamsRequest params; };
struct FrameTileInfo { bool reused_sequence; TileParams params; size_t syntax_bits; };
bool write_frame_tile_info(...);
bool read_frame_tile_info(...);
```

`sequence_superblock_size()` is the §5.18.7.6 `get_seq_sb_size()`. Call it
**after** parsing `sequence_partition_config()` and pass the result to both the
sequence tile writer (where it is `uniformSbSize == sbSize`) and the frame tile
writer (where `uniformSbSize = seqSbSize` and `sbSize` is the frame's
`SbSize`).

`write_tile_params()` implements the complete §5.18.7.3 function, including
both uniform and non-uniform spacing, level/tier `maxTileWidthSb` /
`maxTileAreaSb` derivation, `tile_log2()`, and `uniform_spacing()`. The reader
parses the non-uniform `ns(n)` widths/heights too.

`write_frame_tile_info()` implements §5.18.7.2 `tile_info()` reuse exactly:

- `haveTileParams = seq_tile_info_present_flag`;
- uniform sequence params are eligible when `uniform_eligible()` holds for the
  frame's superblock grid (they need not have identical dimensions);
- non-uniform sequence params are eligible only when `SeqSbCols == sbCols` and
  `SeqSbRows == sbRows`;
- if eligible and `allow_tile_info_change == 0`, no bits are written and the
  frame infers `reuse_tile_info = 1`;
- if eligible and `allow_tile_info_change == 1`, one `reuse_tile_info` bit is
  written, followed by full `tile_params()` when it is 0;
- if not eligible, full `tile_params()` is always written.

### §5.4.4 / §5.4.9 segments

```cpp
struct SegmentInfo {
  bool enabled[16][3];
  int value[16][3];
};
struct SequenceSegmentConfig {
  bool enable_ext_seg;  // MaxSegments = 16, else 8
  bool present;         // seq_seg_info_present_flag
  bool allow_change;    // seq_allow_seg_info_change
  SegmentInfo info;
  int max_segments() const;
};

bool write_segment_info(BitWriter&, const SegmentInfo&, int num_segments, error);
bool read_segment_info(BitReader&, int num_segments, SegmentInfo*, error);
bool write_sequence_segment_config(...);
bool read_sequence_segment_config(...);

bool write_frame_segmentation_params(BitWriter&, const SequenceSegmentConfig&,
                                     const FrameSegmentRequest&,
                                     bool derived_primary_ref_none,
                                     bool frame_type_switch,
                                     FrameSegmentState*, error);
bool read_frame_segmentation_params(...);
```

`seg_info()` uses the normative tables from §5.4.9:
`Segmentation_Feature_Bits = {9,0,0}`, `_Signed = {1,0,0}`,
`_Max = {MAXQ_BITS,0,0}`. `SEG_LVL_ALT_Q` is therefore `su(10)` clipped by the
decoder to `[-351,351]`; the writer rejects values outside that range rather
than relying on the clip. `SEG_LVL_SKIP` and `SEG_LVL_GLOBALMV` have zero value
bits, so an enabled flag there still writes only the `feature_enabled` bit.

Persistence and change rules (§5.18.7.1):

| Sequence state | `allow_change` | Frame syntax |
|---|---:|---|
| `seq_seg_info_present_flag = 1` | 0 | `reuse_seg_info = 1` inferred; **no** `reuse_seg_info` bit, no `seg_info()` |
| `seq_seg_info_present_flag = 1` | 1 | `reuse_seg_info f(1)`; 1 copies `SeqFeature*`, 0 codes `seg_info(MaxSegments)` |
| `seq_seg_info_present_flag = 0` | — | frame codes `seg_info(MaxSegments)` when segmentation is enabled |

`write_frame_segmentation_params()` rejects a frame override when the sequence
header forbids it, and derives `MaxSegments` from `enable_ext_seg` even when
sequence info is absent (the frame then signals 8 or 16 segments). For
`DerivedPrimaryRefFrame == PRIMARY_REF_NONE` it infers
`segmentation_update_map = 1` and writes no map-update bits; otherwise it codes
`segmentation_update_map` and, when set and `FrameType != SWITCH_FRAME`,
`segmentation_temporal_update`.

## Pinned bit goldens

`small_header_test.cpp` pins exact bits, not just round-trips. Selected cases:

| Structure | Bits | Bytes |
|---|---|---|
| cropping absent | `0` | `00` |
| `left=1,right=2,top=3,bottom=4` | `1 010 011 00100 00101` | `a64280` |
| `reduce_pb_aspect_ratio = 0` | `0` | `00` |
| reduce, `log2_minus_1 = 0` | `10` | `80` |
| reduce, `log2_minus_1 = 1` | `11` | `c0` |
| `df_par_bits_minus_2 = 0..3` | `00,01,10,11` | `00,40,80,c0` |
| sequence segment absent | `00` | `00` |
| 8 segments, present, all disabled | `010` + 24 zeros | 27 bits |
| 16 segments, present+allow | `111` + 48 zeros | 51 bits |
| seq tile 64×64, allow 0 | `101` | 3 bits |
| seq tile 640×360, allow 1 | `11100` | 5 bits |
| frame tile reuse, no change allowed | *(no bits)* | 0 bits |
| frame tile reuse bit | `1` | 1 bit |
| frame tile fresh 64×64 tile_params | `01` | 2 bits |

Additional round-trip coverage includes the §5.18.7.3 forced minimum for an
8192-wide level-5.1 frame (`tile_cols_log2 = tile_rows_log2 = 1`) and
non-uniform 3×3 tile spacing, plus the same frame at `seqSbSize = 128×128`
(superblock grid 5×3, `sb_shift = 5`).

Validation/error tests cover: off-limit crop offsets and empty cropped
rectangles; reduce/log2 inconsistencies; `df_par` outside 0..3 and
unrepresentable `DfDeltaQ`; segment values outside `±MAXQ_BITS`, non-zero
SKIP/GLOBALMV values, `allow_change` without `present`, and forbidden frame
overrides; tile log2 outside level bounds, bad non-uniform coverage, reserved
level/tier combinations, wrong `seqSbSize`, and `allow_change` without
`present`.

## dav2d gate

`build_and_test.sh` generates two real M1 still-picture baselines with the
existing encoder (`--m1 --no-deblock --m1-single`) at 64×64 and 640×360. The
gate:

1. parses TD/SEQ/CLK OBUs from the IVF;
2. reconstructs the baseline frame header and sequence OBU with features **off**
   and byte-compares both with the input (`header parity OK`);
3. strips the original tile payload after that header and re-emits CLK with the
   same payload and the feature-on headers;
4. writes `crop`, `aspect`, `dfpar`, `segment`, `tile`, and `all` IVF streams.

`--m1-single` is used so the baseline prediction blocks are square and already
respect the reduced `MaxPbAspectRatio`. The script also runs a negative control:
the same feature on the full M1 tree must be **rejected** by `dav2d`, proving
the decoder is parsing and enforcing `reduce_pb_aspect_ratio`.

Observed output on the pinned `dav2d` `f9dafbc`:

```text
small_header_test: PASS (364 checks)
gate: header parity OK (SEQ 11 bytes, CLK header 3 bytes, tile 697 bytes), 6 feature streams
gate: header parity OK (SEQ 12 bytes, CLK header 4 bytes, tile 9862 bytes), 6 feature streams
m1_64s/crop:    output 64x64,  recon diffs Y=0 U=0 V=0
m1_64s/aspect:  output 64x64,  recon diffs Y=0 U=0 V=0
m1_64s/dfpar:   output 64x64,  recon diffs Y=0 U=0 V=0
m1_64s/segment: output 64x64,  recon diffs Y=0 U=0 V=0
m1_64s/tile:    output 64x64,  recon diffs Y=0 U=0 V=0
m1_64s/all:     output 64x64,  recon diffs Y=0 U=0 V=0
m1_360s/crop:    output 640x360, recon diffs Y=0 U=0 V=0
m1_360s/aspect:  output 640x360, recon diffs Y=0 U=0 V=0
m1_360s/dfpar:   output 640x360, recon diffs Y=0 U=0 V=0
m1_360s/segment: output 640x360, recon diffs Y=0 U=0 V=0
m1_360s/tile:    output 640x360, recon diffs Y=0 U=0 V=0
m1_360s/all:     output 640x360, recon diffs Y=0 U=0 V=0
gate: all feature streams decoded with 0 reconstruction diffs
gate: negative control rejected as expected (reduce_pb_aspect_ratio parsed)
gate: PASS
```

All decodes use exactly:

```sh
dav2d --strict 1 --threads 1
```

Implementation note: pinned `dav2d` emits the coded frame dimensions even when
`seq_cropping_window_present_flag` is 1; it parses and constrains the window but
does not crop its Y4M output. The comparison therefore crops the baseline when
the decoder output is smaller than the coded frame and otherwise compares the
full frame. Cropping semantics (`CropWidth`/`CropHeight`) are pinned by the
`s64280` bit golden and the `derive_display_size()` tests.

The segment gate enables `enable_ext_seg` (16 segments) and non-zero
`SEG_LVL_ALT_Q` values in the sequence header while leaving the frame's
`segmentation_enabled` bit at 0. That exercises the full 16-segment
`seg_info()` bit count in the sequence header and proves the following frame
header stays synchronized. Enabling frame-level segmentation on an existing M1
tile would require rewriting every block's segment-id syntax, which is outside
this header-only module; persistence/change rules are covered by
`test_frame_segmentation_persistence()`.

## Exact core call sites

These are the edits the M1/M2 owners need to make. No core file is changed by
this module.

### `src/encoder/m1/m1_headers.cpp`

| Current site | Replacement |
|---|---|
| L100 `writer.put_bit(false); // seq_cropping_window_present_flag` | `write_cropping_window(writer, c.cropping, &error);` (after `seq_allow_seg_info_change`-style config plumbing) |
| L107 `writer.put_bit(false); // reduce_pb_aspect_ratio` | `write_pb_aspect_ratio(writer, c.pb_aspect_ratio, &error);` |
| L110–111 `enable_ext_seg` / `seq_seg_info_present_flag` | `write_sequence_segment_config(writer, c.segment, &error);` |
| L154 `writer.put_bits(0U, 2U); // df_par_bits_minus_2` | `write_df_par(writer, c.df_par, &error);` |
| L157 `writer.put_bit(false); // seq_tile_info_present_flag` | `write_sequence_tile_config(writer, &c.tile, width, height, sequence_superblock_size(...), level, 0, &error);` |
| L173 `write_tile_info(writer, c);` | `write_frame_tile_info(writer, c.tile, width, height, SbSize, level, 0, frame_tile_request, &tile_info, &error);` with `c.tile` from the sequence header |
| L176 `writer.put_bit(false); // segmentation_enabled` | `write_frame_segmentation_params(writer, c.segment, frame_seg_request, /*derived_primary_ref_none=*/true, /*switch=*/false, &seg_state, &error);` |
| L191–198 `df_delta_q_present` per plane | validate `DfDeltaQ` against `c.df_par` with `validate_df_delta_q()` before writing any `df_delta_q f(dfParBits)` value |

`M1Config` gains `CroppingWindow cropping; PbAspectRatio pb_aspect_ratio;
DfPar df_par; SequenceSegmentConfig segment; SequenceTileConfig tile;`.
`sequence_superblock_size(use256, use128)` must be evaluated from the partition
config; M1 currently always passes `SbSize::k64x64`.

### `src/encoder/m2/m2_headers.cpp`

Same replacements at:

| Feature | Line |
|---|---:|
| cropping | 125 |
| `reduce_pb_aspect_ratio` | 134 |
| `enable_ext_seg` / `seq_seg_info_present_flag` | 137–138 |
| `df_par_bits_minus_2` | 214 |
| `seq_tile_info_present_flag` | 217 |
| frame `write_tile_info` | 280 |
| frame `segmentation_enabled` | 283 |
| frame deblock `df_delta_q_present` | 284–289 |

M2's sequence has `single_picture_header_flag = 0`, so the segment and tile
functions are called identically; only the surrounding baseline fields differ.

For a format where the sequence header already encodes a feature, the reader
`read_*` functions can parse the emitted OBU without linking the writer, which
is how the unit tests cross-check the bit counts.
