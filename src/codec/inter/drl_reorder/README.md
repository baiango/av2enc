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

`build_reordered_stack()` copies the parameters, passes the requested mode
into `merge::build_mv_stack()` (which applies §7.12.2.19 at step 17, before the
step-20 extra search), and never sorts a second time. `use_temporal_first` is
the caller's step-16 "do not sort" term; the product path passes the pinned
decoder's high-priority-TMVP decision (see below).

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

## The previous desync (resolved)

The first product integration looked exact with zero swaps but desynchronised
from frame 2 as soon as a swap moved a candidate. Two independent causes were
found against the pinned decoder (`dav2d` `f9dafbc`, `src/refmvs.c`):

1. **The transform ran after the whole stack was built.** `merge::
   build_mv_stack()` runs the §7.12.2.19 sort at step 17, *before* the
   §7.12.2.20 extra search derives the MVP-combination candidates from the
   sorted `[0, numNearest)` prefix. Building with the sort disabled and
   applying the transform to the finished stack leaves those tail candidates
   in pre-sort order, so every signalled index past `numNearest` names a
   different MV. Measured case: 64x64 at `mi(144,176)` on the 720p clip, post
   index 4 is `(−2,−2)` with the late transform and `(0,6)` in the decoder.
   The fix passes the mode into `merge::build_mv_stack()`; `merge::
   sort_nearest()` was the historical **full selection sort** and now performs
   the normative single move-to-front.
2. **The pinned decoder suppresses CONSTRAINT for close references.** dav2d
   computes its step-16 "high-priority TMVP" term as
   `drl_reorder != ALWAYS && !has_bothside_refs && |POC diff| <= 2` for a
   single-reference block, **without** the spec's `use_ref_frame_mvs` guard.
   For a low-delay P frame the reference is one POC away, so
   `DRL_REORDER_CONSTRAINT` never sorts. The encoder mirrors that condition
   (`drl_constraint_sort_suppressed()` in `video_frame.cpp`), which makes the
   CONSTRAINT stream bit-exact but inert; `DRL_REORDER_ALWAYS` exercises the
   sort. Reproducer for the first cause before the fix:
   `--video-drl-reorder-always --video-max-depth 3` (3 frames, 720p) differed
   from frame 2 at `(704,576)`.

The legacy full-selection-sort divergence is still pinned for the record. For
weights `{1,2,3,4}`:

| transform | post order | pre=1 maps to | decoder index 2 is |
|---|---|---|---|
| normative single swap | `{4,2,3,1}` | post 1 | MV 20 |
| legacy selection sort | `{4,3,2,1}` | post 2 | MV 30 |

The shared `merge` module now performs the normative single swap, so these
two transforms agree. `drl_reorder_test` still pins the legacy case
(`test_legacy_desync`) and the `{1,1,1,2,1,0}` fixture
(`test_real_video_scan_fixture`).

`inter::mc::build_mv_stack` (`src/codec/inter/mc/mc.cpp`) is a separate
audit-only stack builder and still uses the full selection sort; the wired
video path uses `merge`.

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

81,728 checks, 0 failures; ASan/UBSan clean.

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

## Product integration (applied)

The CPU video path (`src/pipeline/video/`) now carries the pair:

1. `video_headers.cpp` writes `disable_drl_reorder` and, when it is 0,
   `constrain_drl_reorder` from `VideoConfig`.
2. `video_frame.cpp::load_stack()` calls `drl::build_reordered_stack()` with
   the requested mode and the pinned decoder's step-16 suppression term
   (`drl_constraint_sort_suppressed()`).
3. `video_syntax.cpp` builds the same stack when it validates a block's
   signalled index (the writer-side mirror).
4. `merge::sort_nearest()` performs the normative single move-to-front at
   step 17, so the step-20 MVP-combination candidates are derived from the
   sorted prefix.
5. `--video-drl-reorder` / `--video-drl-reorder-always` select CONSTRAINT /
   ALWAYS; the CLI composition rules still reject DRL with `--b-frames`,
   `--video-compound`, `--video-gme`, SVC and multi-tile streams.

`--video-trace` prints one `drl reorder mi=(row,col) WxH ref=N nn=M pre=...
post=...` line per block whose `[0, numNearest)` prefix the transform changed;
`gate_video_drl_reorder.sh` counts those lines to prove the transform is
exercised.

`inter::mc::build_mv_stack` (`src/codec/inter/mc/mc.cpp`) is a separate
audit-only stack builder, still uses the full selection sort, and must not be
used with reordering enabled until it calls this module.

## Product gate

`src/pipeline/video/gate_video_drl_reorder.sh` (registered as
`gate_video_drl_reorder` in `tools/run_gates.sh`) generates a deterministic
640x360 pan fixture and pins:

| case | swaps | pictures | recon |
|---|---|---|---|
| depth 3, CONSTRAINT | 0 | 6/6 | exact |
| depth 3, ALWAYS | 18 | 6/6 | exact |
| depth 4 min-block 8, ALWAYS | 74 | 6/6 | exact |

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
