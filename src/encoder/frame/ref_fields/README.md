# Frame-header reference fields — §5.18.1/§5.18.2/§5.18.3.4/§5.18.4.3

Status: **spec-complete writer + independent reader + pinned goldens + a
`dav2d --strict 1 --threads 1` gate**, with no edits to the M1/M2 core,
`src/main.cpp` or `CMakeLists.txt`.

This module implements the five frame-header mechanisms that
`docs/spec-audit/03-frame-header.md` lists as **ABSENT**:

| Mechanism | § | Audit rows closed |
|---|---|---|
| `primary_ref_frame` + `disable_cross_frame_cdf_init` and the CDF-init consequence | §5.18.2 / §6.17.2 | `signal_primary_ref_frame`, `primary_ref_frame` |
| `max_drl_bits_minus_1` / `change_drl` | §5.18.2 / §6.17.2 | `max_drl_bits_minus_1` / `change_drl` |
| `change_bvp_drl` / `max_bvp_drl_bits_minus_1` | §5.18.3.4 / §6.17.3.3 | `change_bvp_drl` / `max_bvp_drl_bits_minus_1` |
| `has_refresh_frame_flags` / `frame_to_refresh` | §5.18.2 / §6.17.2 | `has_refresh_frame_flags` / `frame_to_refresh` |
| `frame_size_with_refs()` / `found_ref` | §5.18.4.3 / §6.17.4.3 | `frame_size_with_refs()` / `found_ref` |
| `frame_header_copy()` / `header_bit[i]` | §5.18.1 / §6.17.1 | `frame_header_copy` / `header_bit[i]` |

Normative reference: AV2 Bitstream & Decoding Process Specification v1.0.0
(2026-05-28, commit `38f28e7`), `spec/AV2_Spec_v1.0.0.pdf`.

```text
src/encoder/frame/ref_fields/
  ref_fields.h            public API: syntax structs, writer, reader, validators
  ref_fields.cpp          writer + independent bit reader + semantics
  ref_fields_m2_adapter.h M2 sequence/frame-header mirror used by the gate
  ref_fields_m2_adapter.cpp
  ref_fields_test.cpp     305 checks: goldens, round-trips, every error path
  ref_fields_gate.cpp     3-stream dav2d gate + header re-parse assertions
  build_and_test.sh       standalone build/run (no CMake changes)
  README.md               this file
```

## Spec map

| Syntax / rule | Section |
|---|---|
| `frame_header( isFirst )`, `frame_header_copy()`, `header_bit[i]` | §5.18.1 |
| `frame_header_present_flag`, `is_first_tile_group` | §5.19 |
| `frame_header_info()` order: `cur_mfh_id`, `seq_header_id_in_frame_header`, `frame_is_inter`, output flags, `frame_size_override_flag`, `order_hint` | §5.18.2 |
| primary-ref block, refresh block, explicit map, `frame_size_with_refs()` | §5.18.2 |
| `intrabc_params()`, `screen_content_params()` | §5.18.3.3, §5.18.3.4 |
| `max_drl_bits_minus_1` / `change_drl` | §5.18.2 |
| `change_bvp_drl` / `max_bvp_drl_bits_minus_1` | §5.18.3.4 |
| `max_drl_bits_minus_1` semantics, `has_refresh_frame_flags`, `frame_to_refresh`, `refresh_frame_flags` | §6.17.2 |
| `found_ref` semantics + 1/2..16 size ratio | §6.17.4.3 |
| `compute_image_size()` (`MiCols`, `MiRows`) | §5.18.4.4 / §6.17.4.4 |
| `set_primary_ref_frame_and_ctx()`, `choose_primary_secondary_ref_frame()` | §5.18.2 (CDF-init consequence) |
| `uvlc()`, `ns(n)`, `su(n)` | §4.11.3, §4.11.7, §4.11.6 |
| `PRIMARY_REF_NONE = 7`, `PRIMARY_REF_CHOOSE = 8`, `RESTRICTED_OH = -1`, `MAX_REF_MV_STACK_SIZE = 6`, `MAX_REF_BV_STACK_SIZE = 4`, `NUM_REF_FRAMES = 16`, `REFS_PER_FRAME = 7` | §3 |

## API

All syntax functions write/read *bits only* — no OBU framing and no
`byte_alignment()`. `BitWriter` is the repo's existing MSB-first writer; the
reader is `RefBitReader`, private to this module, so the read path never calls
writer-side code.

### Primary reference (§5.18.2)

```c++
struct PrimaryRefSignal {
  bool signal_primary_ref_frame = false;
  std::uint32_t primary_ref_frame = kPrimaryRefChoose;  // f(3) when signalled
  bool disable_cross_frame_cdf_init = true;
};

struct PrimaryRefResolved {
  std::uint32_t primary_ref_frame;            // after the §5.18.2 forcing
  std::uint32_t derived_primary_ref_frame;
  std::uint32_t secondary_ref_frame;
  bool coded_disable_cross_frame_cdf_init;
  bool disable_cross_frame_cdf_init;          // effective
  bool load_cdfs_from_reference;              // false => init_coeff/non_coeff
  std::uint8_t cdf_reference_slot;            // ref_frame_idx[primary]
  bool load_previous_state;                   // load_previous()
};

bool write_primary_ref_signal(BitWriter&, FrameType, bool is_tip,
                              const PrimaryRefSignal&, std::string*);
bool read_primary_ref_signal(RefBitReader&, FrameType, bool is_tip,
                             PrimaryRefSignal*, std::string*);
bool resolve_primary_ref(FrameType, const PrimaryRefSignal&, const RefList&,
                         const FrameRefState&, std::uint8_t base_q_idx,
                         PrimaryRefResolved*, std::string*);
void choose_primary_secondary_ref_frame(const RefList&, const FrameRefState&,
                                        std::uint8_t base_q_idx,
                                        std::uint32_t* primary,
                                        std::uint32_t* secondary);
```

`resolve_primary_ref` implements the whole §5.18.2
`set_primary_ref_frame_and_ctx()` consequence:

```c
(DerivedPrimaryRefFrame, derivedSecondaryRefFrame) =
    choose_primary_secondary_ref_frame();
if (primary_ref_frame == PRIMARY_REF_CHOOSE) primary_ref_frame = Derived;
if (Derived == PRIMARY_REF_NONE || primary_ref_frame == PRIMARY_REF_NONE) {
  primary_ref_frame = PRIMARY_REF_NONE;
  DerivedPrimaryRefFrame = PRIMARY_REF_NONE;
  disable_cross_frame_cdf_init = 1;         // forced
}
if (primary_ref_frame == PRIMARY_REF_NONE || disable_cross_frame_cdf_init)
  init_non_coeff_cdfs(); init_coeff_cdfs();  // load_cdfs_from_reference == false
else
  load_cdfs(ref_frame_idx[primary_ref_frame]); load_previous();
```

The derivation (closest `RefBaseQIdx`, then `is_ref_better()` with
`FloorLog2(RefFrameWidth * RefFrameHeight)` and `get_relative_dist()`) is
modelled in full. `first_slot_with_ref()` is a decoder-internal `RefCounter`
dedup rule that this module does not model; every valid slot is treated as the
first slot holding its frame.

Enforced: `primary_ref_frame < NumTotalRefs`, the slot must be `RefValid` and
must not have `RESTRICTED_OH`, `f(3)` cannot encode `PRIMARY_REF_NONE`, and an
intra/SWITCH frame cannot signal a primary.

### DRL depth (§5.18.2, §5.18.3.4)

```c++
struct DrlConfig { bool allow_frame_max_drl_bits; std::uint32_t seq_max_drl_bits_minus_1; };
struct DrlSignal { bool change_drl; std::uint32_t max_drl_bits_minus_1; };
bool write_drl(BitWriter&, const DrlConfig&, const DrlSignal&, std::string*);
bool read_drl(RefBitReader&, const DrlConfig&, DrlSignal*, std::string*);

struct BvDrlConfig { bool allow_frame_max_bvp_drl_bits; std::uint32_t seq_max_bvp_drl_bits_minus_1; };
struct BvDrlSignal { bool change_bvp_drl; std::uint32_t max_bvp_drl_bits_minus_1; };
bool write_bvp_drl(BitWriter&, const BvDrlConfig&, const BvDrlSignal&, std::string*);
bool read_bvp_drl(RefBitReader&, const BvDrlConfig&, BvDrlSignal*, std::string*);
```

The writer takes the *resolved* value and inverts the decoder's
`if (value >= seq) value += 1` step. Enforced: the change flag requires the
matching sequence flag, the value is representable in `ns(MAX_REF_MV_STACK_SIZE-2)`
resp. `ns(2)`, and the result stays `<= MAX_REF_MV_STACK_SIZE - 2` (= 4) resp.
`<= MAX_REF_BV_STACK_SIZE - 2` (= 2). `change_bvp_drl` is only reachable inside
`intrabc_params()` when `allow_intrabc == 1`; the adapter keeps that ordering.

### Refresh mask (§5.18.2)

```c++
struct RefreshConfig { bool enable_short_refresh_frame_flags; std::uint32_t num_ref_frames;
                       bool obu_is_closed_loop_key; bool max_mlayer_id_zero; };
struct RefreshSignal { bool has_refresh_frame_flags; std::uint32_t frame_to_refresh;
                       std::uint32_t refresh_frame_flags; };
bool write_refresh_frame_flags(BitWriter&, FrameType, const RefreshConfig&,
                               const RefreshSignal&, bool immediate_output_frame,
                               std::string*);
bool read_refresh_frame_flags(RefBitReader&, FrameType, const RefreshConfig&,
                              bool immediate_output_frame, RefreshSignal*,
                              std::string*);
bool validate_refresh_frame_flags(FrameType, const RefreshConfig&,
                                  const RefreshSignal&,
                                  bool immediate_output_frame, std::string*);
```

Every branch of §5.18.2 is covered: CLK `allFrames` inference (no bits),
short-refresh KEY (`frame_to_refresh` only), generic KEY (`f(NumRefFrames)`),
SWITCH (`f(NumRefFrames)`), short-refresh inter (`has_refresh_frame_flags` +
`frame_to_refresh`, or a zero mask), and the generic inter mask. Enforced:
`frame_to_refresh < NumRefFrames`, `has == 0` implies a zero mask (and vice
versa), `INTRA_ONLY_FRAME` must not refresh every slot, and a frame that is not
output immediately must refresh at least one slot.

### Frame size from references (§5.18.4.3)

```c++
struct SizeConfig { std::uint32_t num_total_refs, max_frame_width, max_frame_height;
                    std::uint32_t frame_width_bits, frame_height_bits; };
struct FrameSizeSignal { std::uint32_t width, height; bool prefer_found_ref; };
struct FrameSizeResolved { std::array<bool,7> found_ref; std::uint32_t inherited_from;
                           bool explicit_size; std::uint32_t width, height, mi_cols, mi_rows; };
bool write_frame_size_with_refs(BitWriter&, const SizeConfig&, const RefList&,
                                const FrameRefState&, const FrameSizeSignal&,
                                FrameSizeResolved*, std::string*);
bool read_frame_size_with_refs(RefBitReader&, const SizeConfig&, const RefList&,
                               const FrameRefState&, FrameSizeResolved*,
                               std::string*);
bool validate_frame_size_against_refs(const FrameSizeResolved&, const RefList&,
                                      const FrameRefState&, std::string*);
```

`prefer_found_ref` picks the first reference in the list whose live size equals
the requested size; otherwise `found_ref = 0` for every reference and
`frame_size()` is coded with the sequence `frame_width_bits_minus_1 + 1` /
`frame_height_bits_minus_1 + 1`. `NumTotalRefs == 0` codes `frame_size()`
directly (the `found_ref` loop body never runs), which the tests pin.

Enforced (§6.17.4.3): the selected reference must be `RefValid` and must not
have `RESTRICTED_OH`; the final size must fit `max_frame_*`; and for every
reference in the list `2*FrameWidth >= RefFrameWidth`,
`FrameWidth <= 16*RefFrameWidth` (same for height). `MiCols`/`MiRows` follow
§5.18.4.4.

### Shared header / `frame_header_copy` (§5.18.1)

```c++
struct HeaderBits { std::size_t num_bits; std::vector<std::uint8_t> bytes; };
HeaderBits capture_bits(std::span<const std::uint8_t> writer_bytes,
                        std::size_t start_bit, std::size_t end_bit);
std::size_t write_frame_header_copy(BitWriter&, const HeaderBits&, std::string*);
bool read_frame_header_copy(RefBitReader&, const HeaderBits& expected, std::string*);

class FrameHeaderStore {
  void reset();
  void record(std::span<const std::uint8_t> writer_bytes,
              std::size_t start_bit, std::size_t end_bit);  // NumFrameHeaderBits
  bool present() const;  std::size_t num_bits() const;  const HeaderBits& bits() const;
  std::size_t write_copy(BitWriter&, std::string*) const;
  bool verify_copy(RefBitReader&, std::string*) const;
};

bool write_nonfirst_tile_group_header(BitWriter&, const FrameHeaderStore&,
                                      bool frame_header_present, std::string*);
bool read_nonfirst_tile_group_header(RefBitReader&, const FrameHeaderStore&,
                                     bool* frame_header_present, std::string*);
```

`capture_bits` extracts `[start_bit, end_bit)` of a writer buffer into a packed
MSB-first buffer, so a caller never re-runs the header writer to learn
`NumFrameHeaderBits`. The copy is replayed bit for bit and is *not*
byte-aligned (§6.17.1 NOTE). The reader rejects any mismatch, any truncation and
an empty header.

Layering: the copy lives in `tile_group_obu()`, not in `frame_header_info()`.
`is_first_tile_group = 1` records the header; a later tile group sets
`frame_header_present_flag = 1` and repeats it. The M2 adapter exposes the full
path as `make_m2_ref_frame_obus(..., split_tile_groups = true)`.

### M2 adapter

```c++
struct M2RefConfig { /* sequence + frame config, three gate flags */ };
struct M2RefFrameHeader { /* key, order_hint, refresh, primary, size, drl, intrabc */ };
struct M2RefParsedHeader { /* everything the independent reader recovered */ };

std::vector<std::uint8_t> make_m2_ref_sequence_header_obu(const M2RefConfig&, std::string*);
std::size_t write_m2_ref_frame_header(BitWriter&, const M2RefConfig&,
                                      const M2RefFrameHeader&, const RefList&,
                                      const FrameRefState&, const M2TileInfo&,
                                      std::size_t* header_bits, std::string*);
std::vector<std::vector<std::uint8_t>> make_m2_ref_frame_obus(
    const M2RefConfig&, const M2RefFrameHeader&, const RefList&,
    const FrameRefState&, const M2TileInfo&,
    std::span<const std::vector<std::uint8_t>> tile_payloads,
    bool split_tile_groups, std::string*);
bool read_m2_ref_frame_header(RefBitReader&, bool keyframe, const M2RefConfig&,
                              const RefList&, const FrameRefState&,
                              const M2TileInfo&, M2RefParsedHeader*, std::string*);
bool read_m2_ref_nonfirst_tile_group_header(RefBitReader&,
                                            const FrameHeaderStore&,
                                            bool* frame_header_present,
                                            std::string*);
```

With every new option off the adapter is **byte-identical** to
`m2::make_sequence_header_obu()` and `m2::write_frame_header()`; the unit test
pins that for both a key and an inter frame. The three sequence flags that gate
the elements are `allow_frame_max_drl_bits`, `allow_frame_max_bvp_drl_bits` and
`enable_short_refresh_frame_flags`.

`M2RefFrameHeader::{change_drl, max_drl_bits_minus_1, change_bvp_drl,
max_bvp_drl_bits_minus_1}` are only consulted when the matching sequence flag
is 1; otherwise the sequence value is inferred and no bits are coded.

## Pinned bit goldens

Written by the writer, asserted verbatim in `ref_fields_test.cpp`.

| Case | Bits | Bytes |
|---|---|---|
| primary signalled, `primary_ref_frame = 5`, `disable_cross = 1` | `1 1 101` | `e8` |
| primary not signalled, `disable_cross = 0` | `0 0` | `00` |
| primary signalled on a TIP frame (no disable bit), `primary = 0` | `1 000` | `80` |
| `change_drl = 1`, `seq = 4`, value `3` | `1` + `ns(4,3)`=`11` | `e0` |
| `change_drl = 1`, `seq = 2`, value `3` | `1` + `ns(4,2)`=`10` | `c0` |
| `change_drl = 0`, `seq = 4`, value `4` | `0` | `00` |
| `allow_frame_max_drl_bits = 0` | *(none)* | *(empty)* |
| `change_bvp_drl = 1`, `seq = 0`, value `1` | `1` + `ns(2,0)`=`0` | `80` |
| short refresh, inter, `frame_to_refresh = 0` | `1` + `0` | `80` |
| short refresh, inter, no refresh | `0` | `00` |
| generic refresh, `NumRefFrames = 2`, mask `1` | `01` | `40` |
| CLK with `max_mlayer_id = 0` | *(none)* | *(empty)* |
| short-refresh KEY (non-CLK), slot 1 | `1` | `80` |
| `found_ref` on ref 0 | `1` | `80` |
| `found_ref` on ref 1 | `0 1` | `40` |
| explicit `frame_size()`, 128x64, 7+7 bits | `0` + `1111111` + `0111111` | `7f7e` |
| `NumTotalRefs = 0` (no found_ref bits), 128x64 | `1111111` + `0111111` | `fefc` |
| 13-bit header captured from `10101 1100110 1` | — | `ae68` |
| that header copied after a `101` prefix | `101` + `1010111001101` | `b5cd` |
| non-first group, `frame_header_present_flag = 1`, header `0x5a` | `1` + `01011010` | `ad00` |

## Gate

`ref_fields_gate.cpp` builds three 3-frame, 128x64, 2x1-tile streams from the
same two-tile payloads (one 64x64 intra/inter leaf per tile, so the model
reconstruction is a constant 128 and any header desync shows up immediately):

| Stream | Frame 1 size | Frame 2 tile groups | Other elements |
|---|---|---|---|
| `baseline` | `frame_size_override_flag = 0` (sequence size) | one group, all tiles | generic refresh mask, no `change_drl`, `disable_cross = 1`, no primary signal |
| `found_ref` | `override = 1`, `found_ref = 1` (inherits 128x64) | **two groups**, second repeats the header | `primary_ref_frame = 0` with `disable_cross = 0`, `has_refresh_frame_flags = 1` / `frame_to_refresh = 0`, `change_drl` 4→3, frame 2 `allow_intrabc = 1` and `change_bvp_drl` 0→1 |
| `explicit` | `override = 1`, `found_ref = 0` + explicit 128x64 | two groups | same as `found_ref` (the control) |

Every stream is decoded with `dav2d --strict 1 --threads 1` and must produce
exactly three frames whose Y/U/V samples are all 128, and the three decoded
Y4M files must be byte-identical. The gate also re-parses each produced header
with the independent reader and asserts the fields (inherited size, signalled
primary, CDF source slot, refresh index, DRL values, the copied header).

Observed on the pinned `tools/bin/dav2d-macos` (commit `f9dafbc`):

```text
  stream bytes: baseline 72, found_ref 92, explicit 99
  baseline   decoded 3 frames, all samples 128
  found_ref  decoded 3 frames, all samples 128
  explicit   decoded 3 frames, all samples 128
ref_fields_gate: 36 checks, 0 failure(s)
REF FIELDS GATE PASS
```

This gate is the decoder-gated evidence added for these elements: the pinned
`dav2d` accepts `primary_ref_frame` with `disable_cross_frame_cdf_init = 0`,
`enable_short_refresh_frame_flags` + `has_refresh_frame_flags`,
`allow_frame_max_drl_bits` + `change_drl`, `allow_frame_max_bvp_drl_bits` +
`change_bvp_drl`, `frame_size_with_refs()` / `found_ref`, and a
`frame_header_present_flag = 1` tile group that repeats the header.

Run it with:

```sh
src/encoder/frame/ref_fields/build_and_test.sh            # tests + gate
SANITIZE=1 src/encoder/frame/ref_fields/build_and_test.sh # ASan/UBSan
```

`--features <mask>` on the gate is a diagnostic: bit 0 short refresh,
1 `change_drl`, 2 `change_bvp_drl`, 3 `primary_ref_frame`, 4
`frame_size_with_refs`, 5 tile-group copy, 6 explicit size instead of
`found_ref`.

## Conformance / validation

| Check | Writer | Reader |
|---|---|---|
| `primary_ref_frame` ∈ `[0, NumTotalRefs)` when signalled | reject | reject |
| `primary_ref_frame` is not a `RESTRICTED_OH` reference | reject | reject |
| primary frame is `RefValid` | reject | reject |
| `f(3)` cannot code `PRIMARY_REF_NONE` | reject | reject |
| intra / SWITCH frames cannot signal a primary | reject | structure |
| `change_drl` requires `allow_frame_max_drl_bits` | reject | structure |
| DRL value representable and `<= MAX_REF_MV_STACK_SIZE - 2` | reject | reject |
| `change_bvp_drl` requires `allow_intrabc` and the sequence flag | structure | structure |
| BVP value `<= MAX_REF_BV_STACK_SIZE - 2` | reject | reject |
| `frame_to_refresh < NumRefFrames` | reject | reject |
| `has_refresh_frame_flags` only for non-KEY/non-SWITCH short-refresh frames | reject | structure |
| `has == 0` ⇒ zero mask; `has == 1` ⇒ single-bit mask | reject | reject |
| non-output frame must refresh ≥ 1 slot | reject | reject |
| `INTRA_ONLY` must not refresh all slots | reject | reject |
| CLK must infer `allFrames` | reject | derive |
| `found_ref` selects `RefValid`, non-restricted reference | reject | reject |
| `frame_size()` fits `max_frame_*` and the coded bit width | reject | reject |
| 1/2 .. 16× size ratio against every reference | reject | reject |
| `NumTotalRefs <= REFS_PER_FRAME` | reject | reject |
| `frame_header_copy` length and every bit | emit | reject mismatch/truncation |
| empty header copy | reject | reject |

## Integration call sites (not edited here)

The M2 core and MFH adapter are another owner's files. The exact sites this
module is meant to replace:

| Site | Current | Replacement |
|---|---|---|
| `src/encoder/m2/m2_headers.cpp:165` | `allow_frame_max_drl_bits = 0` | `M2RefConfig::allow_frame_max_drl_bits` |
| `src/encoder/m2/m2_headers.cpp:167` | `allow_frame_max_bvp_drl_bits = 0` | `M2RefConfig::allow_frame_max_bvp_drl_bits` |
| `src/encoder/m2/m2_headers.cpp:182` | `enable_short_refresh_frame_flags = 0` | `M2RefConfig::enable_short_refresh_frame_flags` |
| `src/encoder/m2/m2_headers.cpp:244` | `frame_size_override_flag = 0` | `write_frame_size_with_refs()` |
| `src/encoder/m2/m2_headers.cpp:254-255` | `signal_primary_ref_frame = 0`, `disable_cross_frame_cdf_init = 1` | `write_primary_ref_signal()` |
| `src/encoder/m2/m2_headers.cpp:258` | `refresh_frame_flags f(NumRefFrames)` | `write_refresh_frame_flags()` |
| `src/encoder/m2/m2_headers.cpp:269` | `allow_intrabc = 0` | `write_bvp_drl()` after the bit |
| `src/encoder/m2/m2_headers.cpp:270` | *no bits* for `max_drl_bits_minus_1` | `write_drl()` |
| `src/encoder/frame/mfh/mfh_m2_adapter.cpp:140-146` | rejects inter-frame `frame_size_override` | `write_frame_size_with_refs()` (the README gap closes) |
| `src/encoder/tiles/tile_group.cpp:77-91` | always one group, `frame_header_present_flag` never coded | `write_nonfirst_tile_group_header()` for later groups |
| `src/encoder/m2/post/tiled_frame.cpp:642-700` | one group, all tiles | `make_m2_ref_frame_obus(..., split=true)` for one OBU per group |
| `src/encoder/m2/m2_frame.cpp:1005-1050` | one tile group per frame | carry a `FrameHeaderStore` per frame if tile groups are split |

Every other tile-group writer hard-codes `is_first_tile_group = 1`
(`m1_headers.cpp:287`, `m0_headers.cpp:332`, `longterm_headers.cpp:492`,
`random_access.cpp:1334`, `leading.cpp:518`, `m2_headers.cpp:321`,
`tiled_frame.cpp:680`); those are the sites a `frame_header_present_flag = 1`
path would extend.

## Out of scope

* Bridge frames (§5.18.4.2) and the `IsBridge` header special cases.
* TIP frame headers: the `is_tip` flag is threaded through the primary-ref
  block (the disable bit is absent), but TIP itself is not written.
* The implicit `get_ref_frames()` reference map: the M2 sequence header always
  sets `explicit_ref_frame_map = 1`, so the adapter rejects the implicit form.
* `RefCounter`-based `first_slot_with_ref()` dedup in the primary-ref ranking.
* SWITCH frames (`frame_size_override_flag` is forced and
  `frame_size_with_refs()` is skipped for them).
* Multi-tile-group decoding of the last group's `tile_size_minus_1`: the
  module only writes one tile per group, where the field is absent by syntax.
