# DRL reorder (§5.4.6 / §7.12.2 / §7.12.2.19)

Spec-exact DRL reordering for AV2 v1.0.0 (2026-05-28, commit `38f28e7`).
This directory is additive: it does not modify the M1/M2 core, the shared
`merge` module, `src/main.cpp`, or `CMakeLists.txt`.

## API

`drl_reorder.h` — stack transform, independent of the M2 types.

```cpp
namespace av2enc::inter::drl {

enum class Mode { kDisabled, kConstraint, kAlways };      // Table 6.4
struct SequenceFlags { bool disable_drl_reorder, constrain_drl_reorder; };
Mode mode_from_sequence_header(const SequenceFlags&);
int  derive_frame_max_drl_bits(const FrameDrlParams&, bool* valid);
int  drl_index_count(int max_drl_bits_minus_1, bool* valid);   // m

struct StackCandidate { int32_t mv[2][2]; int weight; int row_offset;
                        int col_offset; int32_t cwp; };
struct Stack { StackCandidate cand[6]; int count; int num_nearest; };

bool use_sort(Mode, bool use_temporal_first, int num_nearest);      // step 16
int  reorder_permutation(..., int* perm);                           // step 17
int  reorder_stack(Stack*, Mode, bool use_temporal_first);
bool pre_to_post_index(const int* perm, int count, int pre, int* post);
bool post_to_pre_index(const int* perm, int count, int post, int* pre);
IndexChoice choose_index_for_mv(const Stack&, int32_t y, int32_t x,
                                int start, int m);
int  signal_index_for_pre(...);       // pre-reorder -> signalled index
}
```

`drl_syntax.h` — §5.20.7.8 writer and **independent** reader.

```cpp
struct DrlCdfState;                    // TileDrlMode / Skip / Tip tables
void init_default_drl_cdfs(DrlCdfState*);
enum class DrlCdfSource { kInter, kSkipMode, kTip };
struct DrlSymbolContext { DrlCdfSource source; int new_mv_context; }; // 0..4
Status write_drl_index(av2::Encoder*, DrlCdfState*, ctx, index, start, m);
Status read_drl_index (av2::Decoder*, DrlCdfState*, ctx, start, m, int*);
int    drl_cdf_row(int idx);           // Min(idx, 2)
```

`video_stack_adapter.h` — the only file that includes `encoder/m2/merge/merge.h`.

```cpp
Mode mode_from_m2(merge::DrlReorder);
merge::Status build_reordered_stack(const merge::StackParams&, Mode,
                                    bool use_temporal_first,
                                    merge::MvStack*);
int reorder_video_stack(merge::MvStack*, Mode, bool use_temporal_first);
int signal_index_for_pre(const merge::MvStack&, Mode, bool, int pre_index);
```

`build_reordered_stack()` copies the parameters, forces
`merge::DrlReorder::kDisabled`, calls `merge::build_mv_stack()` and then applies
the normative transform. It never calls the shared component's selection sort.

## Normative behaviour

* `disable_drl_reorder = 1` → `DRL_REORDER_DISABLED`.
  Otherwise `constrain_drl_reorder` selects `DRL_REORDER_CONSTRAINT` (1) or
  `DRL_REORDER_ALWAYS` (2). There is no frame-level
  `drl_reorder_constrained` in v1.0.0; the frame header only carries
  `max_drl_bits_minus_1`/`change_drl`, gated by `allow_frame_max_drl_bits`.
* Frame stack step 14 snapshots `numNearest`; step 16:
  `useSort = ALWAYS || (CONSTRAINT && !useTemporalFirst && numNearest >= 4)`.
* Step 17 invokes §7.12.2.19 over `[0, numNearest)`:
  **one selection step**, i.e. swap the complete record (weight, CWP, row/col
  offsets, both MV lists) at `start` with the first maximum-weight entry.
  Ties keep the earliest index (`maxWeight < WeightStack[idx]` is strict).
  It is **not** a full selection sort and it never touches derived-SMVP or
  extra-search candidates appended at steps 18–22.
* `read_drl_idx(start, m)` reads `drl_mode` for `idx = start .. m-1`, stops on
  0 and returns `m` on exhaustion. `m = max_drl_bits_minus_1 + 1`.
  A second index starts at `RefMvIdx0 + 1` for same-reference `NEAR_NEARMV`.
* CDF rows: `TileDrlModeCdf[Min(idx, 2)][NewMvContext]` for normal inter,
  `TileSkipDrlModeCdf[Min(idx, 2)]` for `skip_mode`, and
  `TileTipDrlModeCdf[Min(idx, 2)]` for TIP. The default tables are the §9.3
  values; `NewMvContext` is §7.11.2.

## The previous desync

The earlier implementation (still present in `av2enc::video::merge::build_mv_stack` and
`inter::mc::build_mv_stack`) calls a **full selection sort** over
`[0, numNearest)`. For weights `{1,2,3,4}`:

| transform | post order | pre=1 maps to | decoder index 2 is |
|---|---|---|---|
| normative single swap | `{4,2,3,1}` | post 1 | MV 20 |
| legacy selection sort | `{4,3,2,1}` | post 2 | MV 30 |

The encoder believes index 2 is MV 20; the decoder's index 2 is MV 30. Both
streams parse, so the failure appears as a reconstruction/MV mismatch —
exactly the historical `DRL_REORDER_CONSTRAINT` desync. `drl_reorder_test`
pins this case (`test_legacy_desync`) and the shared `merge` test's
`{1,1,1,2,1,0}` fixture is also covered
(`test_real_video_scan_fixture`).

With `DRL_REORDER_CONSTRAINT` the same bug also fires for every stack where
more than one non-maximum weight is out of order; `numNearest >= 4` is the
gate, so only larger inter blocks were affected.

## Tests

`./build_and_test.sh` (add `--no-gate`, `SANITIZE=1`).

* `test_sequence_derivation` / `test_use_sort` — §5.4.6, §5.5, step 16.
* `test_single_move_to_front` — strict ties, complete-record swap,
  `[0,numNearest)` only, CONSTRAINT threshold, ALWAYS.
* `test_legacy_desync` — the failing case above.
* `test_second_drl_start` — every legal `(start, m, index)` round-trips,
  including `start == m`, and illegal indices are rejected.
* `run_writer_reader_roundtrip` — 20,000 randomized stacks × modes ×
  `useTemporalFirst` × CDF sources × NewMvContext; writer → entropy coder →
  independent reader → same index; index mapping preserves the candidate.
* `test_real_video_blocks` — 128×128 M2 frame geometry (64×64 SBs, 16×16 leaves,
  two reference slots), every block's `merge::build_mv_stack()` output over
  all three modes, all signalable indices.
* `test_real_video_scan_fixture` — the M2 merge module's own 8×8 fixture
  (six nearest candidates, weights `{1,1,1,2,1,0}`).
* `test_goldens` — parses `goldens/drl_reorder_goldens.txt`.
* `test_invalid_ranges` — `m > 5`, `start > m`, uninitialized reader.

81,720 checks, 0 failures; ASan/UBSan clean.

## Decoder gate

`gate/` contains an independent P-frame writer for the decoder gate. The M2
core supplies the key and P1 tiles/headers; the gate writes P2 with
`disable_drl_reorder = 0` and `constrain_drl_reorder = 0` (ALWAYS), 8×8 leaves,
two reference slots, five distinct MVs, NEARMV and NEWMV blocks, and DRL
indices chosen from the reordered stack.

```sh
DAV2D=tools/bin/dav2d-macos ./build_and_test.sh
```

The pinned `dav2d --strict 1 --threads 1` decodes the stream and every sample
of the encoder reconstruction matches (0 diffs). Results:

| stream | leaves | distinct MVs | max stack | reordered blocks | recon diffs |
|---|---|---|---|---|---|
| synthetic 128×128 | 256 | 5 | 3 | 14 | 0 |
| `shine` 1280×704 | 14,080 | 5 | 4 | 707 | 0 |
| corrupted-index control | 256 | 5 | 3 | 14 | 1,873 |

The negative control (`--corrupt-index`) signals the wrong DRL index and
produces non-zero diffs, so the gate is sensitive to the exact failure mode.
Key and P1 come from the M2 core with `force_global_mv` so they contain no
`read_drl_idx()`; the shared sequence header (reordering enabled) therefore
cannot desynchronise the reference frames.

Notes:
* AV2 v1.0.0 `Partition_Subsize[PARTITION_SPLIT]` is only valid for
  `BLOCK_128X128`/`BLOCK_256X256`; a 64×64 SB reaches 8×8 through alternating
  `PARTITION_HORZ`/`PARTITION_VERT`. `PARTITION_SPLIT` below 128 is invalid.
* In a `MIXED_REGION` inter frame `is_partition_allowed()` rejects any subsize
  of `BLOCK_4X4`, so the smallest inter leaves are 4×8/8×4 (and 8×8). The gate
  uses the conservative 8×8 corpus; on that corpus the legacy and normative
  transforms happen to agree, so the decoder negative control is the corrupted
  index rather than `--mode legacy`. The exact old-logic failing case is
  pinned by `test_legacy_desync` and `goldens/drl_reorder_goldens.txt`.

## M2 integration sites (not applied)

1. `src/encoder/m2/video_headers.cpp:159`
   `writer.put_bit(true); // disable_drl_reorder`
   → write `Flags.disable_drl_reorder`; when 0 write
   `Flags.constrain_drl_reorder`. Add both to `VideoConfig`.
2. `src/encoder/m2/video_frame.cpp:473`
   `params.reorder = merge::DrlReorder::kDisabled;`
   → `drl::build_reordered_stack(params, mode, /*use_temporal_first=*/false,
   &stack)` (or keep `kDisabled` and call `drl::reorder_video_stack()` after
   `merge::build_stack_checked`). `useTemporalFirst` is false while
   `enable_ref_frame_mvs = 0`.
3. `src/encoder/m2/video_syntax.cpp:1657` — same replacement in
   `write_inter_mode_info()`; it is the writer-side mirror of (2).
4. `src/encoder/m2/merge/merge.cpp:455` — replace `sort_nearest()`'s full
   selection sort with the single move-to-front (or never enable the merge
   module's sort and use `drl::reorder_video_stack`). This is the actual bug.
5. `src/encoder/m2/merge/merge.cpp:1436` (`choose_drl_index_for_mv`,
declared in `merge.h:415`)
   — search the **post-reorder** stack; the signalled index is
   `perm[pre_index]` (`drl::signal_index_for_pre`). Candidates whose post
   index is `>= m` are not signalable.
6. `src/encoder/m2/video_syntax.cpp:1521` (`code_drl_index`) — already uses
   `Min(idx,2)` × `NewMvContext`; route it through
   `drl::write_drl_index()` so writer and reader share the §5.20.7.8 loop.
7. Frame header `max_drl_bits_minus_1`: use
   `drl::derive_frame_max_drl_bits()` and `drl::drl_index_count()`; clamp
   `change_drl` to `ns(MAX_REF_MV_STACK_SIZE - 2)`.
8. `src/encoder/m2/README.md` feature status: replace
   "DRL reordering ... BLOCKED / stays DISABLED" with the mode/flag pair and
   the `merge::build_mv_stack` replacement above.

`inter::mc::build_mv_stack` (`src/encoder/inter/mc/mc.cpp:389-393`) has the same
full-sort bug and is used by `merge::audit_mc_stack`; it must not be used with
reordering enabled until it calls this module.

## Files

| File | Purpose |
|---|---|
| `drl_reorder.{h,cpp}` | mode/limit derivation, transform, permutation, index mapping |
| `drl_syntax.{h,cpp}` | CDF tables, §5.20.7.8 writer and independent reader |
| `video_stack_adapter.{h,cpp}` | M2 merge bridge; forces the shared sort off |
| `drl_reorder_test.cpp` | unit, property, real-M2-block and golden tests |
| `goldens/drl_reorder_goldens.txt` | pinned post-reorder orders and signalled indices |
| `gate/drl_tile.{h,cpp}` | 8×8-leaf P-frame writer (partition/mode/DRL/recon grid) |
| `gate/drl_stream.{h,cpp}` | §5.4/§5.5 headers with `disable_drl_reorder = 0` |
| `gate/drl_gate.cpp` | key+P1 from M2 core, P2 gate, dav2d decode/recon compare |
| `build_and_test.sh` | self-contained build, tests and decoder gate |
