# M1 screen-content signalling — §5.4.7 / §5.18.3.3 / §5.18.3.4

Single source of truth for the **header** side of the AV2 screen-content
tools. This module writes and independently reads the sequence- and frame-level
SCC syntax, validates the combinations the spec forbids, and exposes the
derived rules that the palette and intra-block-copy coders must obey.

It owns **only** this directory. Neither `palette_stream`, `ibc_stream`,
`m1_headers`, `m1_syntax`, nor `main.cpp` was modified.

```sh
./src/encoder/m1/scc_flags/build_and_test.sh
WERROR=1 SKIP_GATE=1 ./src/encoder/m1/scc_flags/build_and_test.sh
SANITIZE=1 WERROR=1 SKIP_GATE=1 ./src/encoder/m1/scc_flags/build_and_test.sh
```

Status: **537 unit checks / 0 failures** and a four-stream `dav2d` gate:

| mode | frame | header | tile | frame | result |
|---|---|---|---:|---:|---|
| `off` | 768x128 | 25 bits / 4 B | 29 B | 51 B | byte-identical to `m1::assemble_m1_still_picture` |
| `integer` | 768x128 | 26 bits / 4 B | 30 B | 52 B | `dav2d` recon exact |
| `intrabc` | 768x128 | 28 bits / 4 B | 34 B | 56 B | `dav2d` recon exact |
| `palette` | 64x64 | 24 bits / 3 B | 167 B | 188 B | `dav2d` recon exact |

Gate command: `dav2d --strict 1 --threads 1` (`f9dafbc`).

## Spec map

| Clause | Page | What it owns here |
|---|---|---|
| §5.4.7 | p63–64 | `sequence_scc_config()`: all eight reachable branches |
| §6.4.7 | p270 | `SELECT_SCREEN_CONTENT_TOOLS` / `SELECT_INTEGER_MV` inference |
| §5.18.3.3 | p113–114 | `screen_content_params()` |
| §5.18.3.4 | p114 | `intrabc_params()` incl. `change_bvp_drl` |
| §6.17.3.2 | p335 | `allow_screen_content_tools` / `force_integer_mv` semantics |
| §6.17.3.3 | p335 | `allow_intrabc` / global / local semantics |
| §5.18.2 | p99, p104–105 | frame SCC region presence; `FrameMvPrecision` |
| §5.20.5.3 | p171 | `use_intrabc` availability |
| §5.20.5.4 | p172 | `intrabc_precision`, IBC `MvPrecision` |
| §5.20.5.5 | p172 | `palette_mode_info()` availability |
| §5.20.7.3 | p189 | inter-frame `use_intrabc` needs `MIXED_REGION` |

## API (`scc_flags.h`)

```cpp
namespace av2enc::m1::scc_flags {

struct SequenceSccConfig;         // exact §5.4.7 syntax selector
struct SequenceSccState;          // decoded values + which bits were present
struct FrameSccContext;           // FrameIsIntra / TIP-as-output / BRU / bridge
struct FrameSccConfig;            // exact §5.18.3.3/.4 selector
struct FrameSccState;             // decoded values + derived rules
struct StillSccConfig, StillSccResult;

bool validate_sequence_scc_config(const SequenceSccConfig&, std::string&);
SequenceSccState derive_sequence_scc_state(const SequenceSccConfig&);
bool write_sequence_scc_config(BitWriter&, const SequenceSccConfig&,
                               SequenceSccState*, std::size_t*, std::string&);
bool read_sequence_scc_config(BitReader&, bool single_picture_header_flag,
                              SequenceSccState&, std::string&);

bool validate_frame_scc_config(const SequenceSccState&, const FrameSccContext&,
                               const FrameSccConfig&, int seq_max_bvp,
                               bool allow_frame_max_bvp, std::string&);
FrameSccState derive_frame_scc_state(...);
bool write_frame_scc_params(BitWriter&, const SequenceSccState&,
                            const FrameSccContext&, const FrameSccConfig&,
                            int seq_max_bvp, bool allow_frame_max_bvp,
                            FrameSccState*, std::size_t*, std::string&);
bool read_frame_scc_params(BitReader&, ...);

bool palette_allowed_for_block(const FrameSccState&, int plane, int y_mode,
                               int bsize, int block_w, int block_h);
bool intrabc_allowed_for_block(const FrameSccState&, int bsize, int block_w,
                               int block_h, bool tree_is_chroma_part,
                               bool region_is_mixed = true,
                               bool frame_is_intra = true);
bool morph_pred_allowed(const FrameSccState&, bool frame_is_intra,
                        bool enable_bawp);
int  intrabc_mv_precision(const FrameSccState&);

bool assemble_still_picture(const M1Config&, const StillSccConfig&,
                            int seq_max_bvp, bool allow_frame_max_bvp,
                            std::span<const std::uint8_t> tile,
                            StillSccResult&, std::string&);
bool still_frame_header_size(...);
bool read_still_frame_scc(std::span<const std::uint8_t> td_seq_clk, ...);

}
```

`BitReader` is an independent MSB-first bit reader with `uvlc()` and `ns(n)`.
The writer is `av2enc::BitWriter`; the reader never calls writer code.

## Exact §5.4.7 syntax

```c
if ( single_picture_header_flag ) {
  seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS   // 2
  seq_force_integer_mv = SELECT_INTEGER_MV                       // 2
} else {
  seq_choose_screen_content_tools f(1)
  if ( seq_choose_screen_content_tools ) seq_force_screen_content_tools = 2
  else                                   seq_force_screen_content_tools f(1)
  if ( seq_force_screen_content_tools > 0 ) {
    seq_choose_integer_mv f(1)
    if ( seq_choose_integer_mv ) seq_force_integer_mv = 2
    else                         seq_force_integer_mv f(1)
  } else {
    seq_force_integer_mv = 2
  }
}
```

All eight reachable branches are pinned (bit string → byte golden):

| Branch | bits | bytes | `force_sct` | `force_imv` |
|---|---|---|---:|---:|
| single picture | `` | `` | 2 | 2 |
| choose sct, choose imv | `11` | `c0` | 2 | 2 |
| choose sct, imv=0 | `100` | `80` | 2 | 0 |
| choose sct, imv=1 | `101` | `a0` | 2 | 1 |
| force sct=0 | `00` | `00` | 0 | 2 |
| force sct=1, choose imv | `011` | `60` | 1 | 2 |
| force sct=1, imv=0 | `0100` | `40` | 1 | 0 |
| force sct=1, imv=1 | `0101` | `50` | 1 | 1 |

Note the spec's non-obvious case: `force_sct = 0` still infers
`seq_force_integer_mv = SELECT_INTEGER_MV` (§5.4.7 p64).

## Exact §5.18.3.3 + §5.18.3.4 syntax

```c
// screen_content_params()
if ( seq_force_screen_content_tools == 2 ) allow_screen_content_tools f(1)
else                                       allow_screen_content_tools = seq_force_screen_content_tools
if ( allow_screen_content_tools ) {
  if ( seq_force_integer_mv == 2 ) force_integer_mv f(1)
  else                             force_integer_mv = seq_force_integer_mv
} else {
  force_integer_mv = 0
}

// intrabc_params()
allow_intrabc f(1)
if ( allow_intrabc ) {
  if ( FrameIsIntra ) {
    allow_global_intrabc f(1)
    if ( allow_global_intrabc ) allow_local_intrabc f(1)
    else                        allow_local_intrabc = 1
  } else {
    allow_global_intrabc = 0
    allow_local_intrabc  = 1
  }
  max_bvp_drl_bits_minus_1 = seq_max_bvp_drl_bits_minus_1
  if ( allow_frame_max_bvp_drl_bits ) {
    change_bvp_drl f(1)
    if ( change_bvp_drl ) {
      max_bvp_drl_bits_minus_1 ns(2)
      if ( max_bvp_drl_bits_minus_1 >= seq_max_bvp_drl_bits_minus_1 )
        max_bvp_drl_bits_minus_1 += 1
    }
  }
}
```

Region presence (§5.18.2): both functions are parsed for `FrameIsIntra`.
For an inter frame they are parsed only when
`TipFrameMode != TIP_FRAME_AS_OUTPUT && !bru_inactive && !IsBridge`;
otherwise the whole region is absent and every flag is inferred 0. The gate's
unit tests pin the TIP-as-output and BRU-inactive cases (0 bits).

Pinned frame branches (single-picture sequence, `force_sct = force_imv = 2`):

| Case | bits | bytes |
|---|---|---|
| all off | `00` | `00` |
| allow screen, fractional | `100` | `80` |
| allow screen, integer | `110` | `c0` |
| IBC global only | `11110` | `f0` |
| IBC local inferred (`global=0`) | `1110` | `e0` |
| IBC global + local | `11111` | `f8` |
| inter SCC present, IBC | `111` | `e0` |
| inter TIP-as-output / BRU-inactive | `` | `` |
| seq forces screen+integer | `0` | `00` |
| seq forces screen off, IBC local | `10` | `80` |
| `change_bvp_drl` seq0 → 1 / 2 | `111010` / `111011` | `e8` / `ec` |
| `allow_frame_max` set, no change | `11100` | `e0` |

## Validation (combinations the spec forbids)

`validate_sequence_scc_config()` rejects:

* any explicit `seq_*` field with `single_picture_header_flag = 1`
  (all are inferred);
* `seq_choose_screen_content_tools` together with
  `seq_force_screen_content_tools`;
* integer-MV fields when `seq_force_screen_content_tools = 0`;
* `seq_choose_integer_mv` together with `seq_force_integer_mv`.

`validate_frame_scc_config()` rejects:

* active SCC flags when the region is absent (TIP-as-output, BRU-inactive
  inter, bridge);
* `force_integer_mv` when `allow_screen_content_tools = 0`;
* a frame value that disagrees with a non-`SELECT` `seq_force_*`;
* `allow_global_intrabc` / `allow_local_intrabc` when `allow_intrabc = 0`;
* `allow_global_intrabc = 1` on an inter frame (inferred 0);
* `allow_local_intrabc = 0` when `allow_intrabc = 1` and
  `allow_global_intrabc = 0` (inferred 1);
* `change_bvp_drl` without `allow_frame_max_bvp_drl_bits`;
* `max_bvp_drl_bits_minus_1 != seq_max_bvp_drl_bits_minus_1` when
  `change_bvp_drl = 0`;
* a `change_bvp_drl` target not reachable by `ns(2)` + the `>= seq` bump
  (for example `seq_max = 0, target = 0`).

`allow_intrabc = 1` is **not** rejected when `allow_screen_content_tools = 0`:
AV2 v1.0.0 does not couple them, and the branch is exercised by the
`seq forces screen tools off; intrabc local inferred` test. The palette and IBC
adapters impose the stronger encoder policy (see below).

## Derived rules

* `FrameSccState::IsIntraBC = allow_intrabc`.
* `force_integer_mv = 1` →
  `FrameMvPrecision = MV_PRECISION_ONE_PEL` and the inter
  `use_qtr_precision_mv` / `allow_high_precision_mv` bits are absent
  (`codes_frame_mv_precision = false`).
* Intra frames never code frame MV precision
  (`FrameMvPrecision` is initialised to one-pel in §5.18.2).
* IBC block MV precision:
  `force_integer_mv ? MV_PRECISION_ONE_PEL : MV_PRECISION_QUARTER_PEL`;
  `intrabc_precision` is coded only for `NEWMV && !force_integer_mv`.
* Palette (§5.20.5.5): luma only (`PlaneStart == 0`), `YMode == DC_PRED`,
  `MiSize >= BLOCK_8X8`, block width/height ≤ 64, and
  `allow_screen_content_tools`.
* IntraBC (§5.20.5.3): `allow_intrabc`, not a chroma part, width/height ≤ 64,
  and `MiSize != BLOCK_64X64`. Inter frames also need
  `RegionType == MIXED_REGION`.
* `morph_pred` (§5.20.5.4): `FrameIsIntra && allow_screen_content_tools &&
  enable_bawp`.

## Interaction with the existing modules

`scc_flags_bridge.h/.cpp` contains the only coupling. Both adapters run the
source module's own validation, translate its flags, then run this module's
validation:

```cpp
palette_stream::ScreenContentFlags -> from_palette_flags(...)
ibc_stream::StreamConfig           -> from_ibc_config(...)
```

Neither module was edited. The gate calls their public encoders:

* `ibc_stream::encode_ibc_stream()` for `off`, `integer`, `intrabc`;
* `palette_stream::encode_minimal_palette_frame()` for `palette`.

For each stream the gate extracts the tile payload, re-assembles the frame with
`scc_flags::assemble_still_picture()`, and requires **byte identity** with the
original frame. It then parses the produced bytes with the independent
`read_still_frame_scc()` reader and checks the recovered state. This proves the
header bits do not drift from either tile coder.

## Production call sites (not applied)

1. **`src/encoder/m1/m1_headers.h` `M1Config`**
   add `scc_flags::FrameSccConfig frame_scc;` and
   `scc_flags::SequenceSccConfig sequence_scc;` (or the four bools and the
   `intrabc_params` fields).
2. **`src/encoder/m1/m1_headers.cpp` `write_frame_header_syntax()`**
   replace the two hard-coded bits at lines 170–171 with
   `scc_flags::write_frame_scc_params()`. With all flags off it writes exactly
   the same `00`; the spliced-tail assembler proves byte identity. For a
   non-single sequence header, call
   `scc_flags::write_sequence_scc_config()` at the §5.4.7 position.
3. **`src/encoder/m1/m1_syntax.cpp` `Writer::write_block()`**
   use `palette_allowed_for_block()` before the palette syntax position and
   `intrabc_allowed_for_block()` before `use_intrabc`. The existing
   `palette_stream::PaletteSyntaxWriter` / `ibc_stream` writers remain the tile
   coders.
4. **`src/encoder/m1/m1_still.cpp`**
   feed `FrameSccState` to the mode decision so palette/IBC candidates are
   only considered where the header allows them.
5. **`src/main.cpp`**
   map user tool flags (e.g. `--m1-palette`, `--m1-ibc`,
   `--m1-force-integer-mv`) onto the config and reject invalid combinations with
   the error strings from `validate_*`.

## Limits

* The assembler supports `single_picture_header_flag = 1` only, because
  `m1::make_sequence_header_obu()` hardcodes it. The §5.4.7 bit writer/reader
  covers the non-single branches for the M2 header writer.
* 8-bit 4:2:0 still-picture header subset; CDEF/CCSO/deblock tails are
  inherited verbatim from `m1::write_frame_header()`.
* No CUDA mirror; this is header syntax only.
