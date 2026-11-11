// M2 CDEF adapter — post-deblock strength planning, §5.18.7.10 header bits and
// §5.20.10.1 per-block index syntax.
//
// Normative placement (AV2 v1.0.0 §7.17/§7.18/§7.19/§7.20):
//
//   reconstruct -> deblock -> **CDEF** -> CCSO -> (loop restoration)
//
// This adapter owns the CDEF slot of the M2 post-chain:
//
//   1. decide one strength set per frame (CdefStrengths == 1) or a per-64x64
//      index grid (CdefStrengths 2..8), using the exact §7.18 filter as the SSE
//      metric;
//   2. produce the filter options and apply §7.18 to the post-deblock planes
//      (the result is the reference frame for later pictures);
//   3. produce the §5.18.7.10 `cdef_params()` fields and the exact header bits;
//   4. write the §5.20.10.1 `cdef_idx` symbols in coding order.
//
// It reuses, rather than reimplements:
//
//   * `src/encoder/filters/cdef/cdef.h` + `cdef_search.h` — the §7.18 pixel
//     core and the SSE search;
//   * `src/cuda/m1/cdef/m1_cdef.{h,cpp}` — the M1 stage planner that wraps the
//     search and the header packing (host-only, no CUDA);
//   * `src/cuda/m1/cdef/m1_cdef_index_syntax.h` — the single implementation of
//     the §8.3.1 `cdef_index0` context;
//   * `src/encoder/m1/filters_stream/` — the verified §5.4.10/§5.18.7.10
//     header writers and the §5.20.10.1 tile-symbol wrappers.
//
// Scope: 8-bit 4:2:0, dimensions divisible by 8 (the M2 core requires that).
// The adapter allocates nothing in the filter/plan calls except the returned
// plan's vectors; it never touches device memory and never synchronises.
//
// Emitting a real M2 stream
// -------------------------
// `CdefStrengths == 1` codes **no** tile symbol (§5.20.10.1 forces index 0),
// so CDEF can be enabled with header bits only; `Plan::header_only_emittable`
// reports that.  The current M2 tile writer has no `read_cdef()` hook, so
// `CdefStrengths >= 2` is a component-only decision until that hook lands.
// The exact hook order is documented in README.md; `TileIndexWriter` below is
// the hook.
#ifndef AV2ENC_ENCODER_M2_CDEF_M2_CDEF_H_
#define AV2ENC_ENCODER_M2_CDEF_M2_CDEF_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "bitstream/bitwriter.h"
#include "cuda/m1/cdef/m1_cdef_index_syntax.h"
#include "encoder/filters/cdef/cdef.h"
#include "encoder/m1/filters_stream/filters_stream.h"
#include "entropy/av2/av2_entropy.h"

namespace av2enc::m2::cdef_adapter {

namespace filters_stream = av2enc::m1::filters_stream;

// One host plane binding.  `stride` is in samples; the plane is MI-aligned
// (`mi_cols*4` luma / `mi_cols*2` chroma wide).
struct PlaneView {
  const std::uint8_t* data = nullptr;
  int stride = 0;
};

// Everything the CDEF slot needs for one frame.
//
// `recon` is the **post-deblock** reconstruction (the CDEF source); `source` is
// the original picture used only by the strength search.  `source` may use a
// tight visible stride.
//
// `skip` / `lossless` / `lossless_chroma` are the §7.18.1 MI grids on the luma
// MI grid; null means "all zero" (the M2 intra case).  Tile starts are luma MI
// units; `disable_loopfilters_across_tiles` in the options decides whether they
// clip the filter region.
struct FrameView {
  int width = 0;
  int height = 0;
  int mi_cols = 0;
  int mi_rows = 0;
  PlaneView recon[3];
  PlaneView source[3];
  const std::uint8_t* skip = nullptr;
  const std::uint8_t* lossless = nullptr;
  const std::uint8_t* lossless_chroma = nullptr;
  const std::uint16_t* tile_col_starts = nullptr;
  int num_tile_cols = 0;
  const std::uint16_t* tile_row_starts = nullptr;
  int num_tile_rows = 0;
};

// Caller-owned MI grids for the all-zero case.  The M2 intra path uses
// zero_grids(); an inter path with skip_flag must fill `skip` from
// `Skips[mi]` and `lossless` from `LosslessArray[SegmentIds[mi]]`.
struct FrameGrids {
  std::vector<std::uint8_t> skip;
  std::vector<std::uint8_t> lossless;
  std::vector<std::uint8_t> lossless_chroma;

  void zero(int mi_cols, int mi_rows);
  void bind(FrameView* view) const;
};

struct Options {
  // CdefDamping = cdef_damping_minus_3 + 3, 3..6.
  int damping = 3;
  // CdefStrengths.  1 is the header-only (zero tile symbol) mode.
  int num_strengths = 1;
  // cdef_on_skip_txfm_frame_enable.  The filter uses it for the MI skip
  // derivation (§7.18.1).
  bool on_skip_txfm = false;
  // Sequence CdefOnSkipTxfm mode: 0 = DISABLED, 1 = ALWAYS_ON, 2 = ADAPTIVE
  // (§5.4.10).  Must agree with `on_skip_txfm`.
  int cdef_on_skip_txfm = 0;
  bool disable_loopfilters_across_tiles = false;
  // Keep an all-zero strength set so a block can be left unfiltered.  Required
  // when num_strengths >= 2 (there is no -1 cdef_idx syntax).
  bool include_off = true;
  std::int64_t chroma_weight = 1;
  // Optional candidate ladder (syntax values); null uses the default ladder.
  const av2enc::cdef::CdefStrength* candidates = nullptr;
  int candidate_count = 0;
};

// Result of the per-frame decision.
struct Plan {
  av2enc::cdef::CdefOptions options{};  // ready for cdef_frame()/the GPU kernel
  int num_strengths = 0;
  // MI geometry the decision was made for (the §5.20.10.1 grid bounds).
  int mi_cols = 0;
  int mi_rows = 0;
  int n64_cols = 0;
  int n64_rows = 0;
  // One entry per 64x64 luma block, raster order, in [0, num_strengths).
  std::vector<std::int8_t> cdef_idx;
  std::int64_t sse_before = 0;  // unfiltered
  std::int64_t sse_after = 0;   // chosen plan
  int blocks_off = 0;
  bool forced_single_strength = false;
  bool header_only_emittable = false;

  // §5.18.7.10 fields, decoded domain, frame_enable = true.
  filters_stream::CdefParams params{};
  // Exact bits write_frame_header_cdef() writes in the M2 header context.
  int header_bits = 0;
  // Exact bits write_index_symbols() writes (including the §8.2.4 exit bits).
  int index_bits = 0;
};

// Runs the strength search.  Does not touch pixels.
bool plan(const FrameView& frame, const Options& options, Plan* out,
          std::string* error);

// Applies §7.18 to `frame.recon` and writes the filtered frame.  `dst` is the
// CDEF output (`CdefFrame`); `src` and `dst` must not alias.  When no CDEF
// stage runs, the caller keeps the post-deblock frame as the reference.
bool filter(const FrameView& frame, const Plan& plan,
            std::uint8_t* const dst[3], const int dst_stride[3],
            std::string* error);

// §5.18.7.10 cdef_params() bits for this plan.
//
// `single_picture_header` selects the M1 (true) or M2 (false) header form.
// `cdef_on_skip_txfm` is the sequence mode for the non-single-picture case;
// with a single picture header the syntax forces ADAPTIVE.
std::size_t write_frame_header_cdef(BitWriter& writer, const Plan& plan,
                                    bool single_picture_header,
                                    int cdef_on_skip_txfm,
                                    std::string* error);

// §5.4.10 sequence_filter_config() row for M2.  `enable_cdef` is the only bit
// this adapter sets; everything else is off.  Returns the bit count.
std::size_t write_sequence_filter_config(BitWriter& writer, bool enable_cdef,
                                         bool on_skip_always_on,
                                         bool on_skip_disabled,
                                         std::string* error);

// The per-tile `read_cdef()` state.  One instance per tile; never shared
// between tiles or frames.  `sb_width_log2` is 4 for 64x64 superblocks.
struct TileIndexWriter {
  filters_stream::CdefTileState state{};
  int tile_mi_col_start = 0;
  int tile_mi_row_start = 0;

  bool init(int mi_cols, int mi_rows, int strengths, int sb_width_log2,
            std::string* error);
  // §5.20.9.6 clear_cdef(): call once at every superblock start, before the
  // first coding block of that superblock.
  void clear_superblock(int mi_row, int mi_col);
  // §5.20.10.1: call from intra_frame_mode_info()/inter_frame_mode_info() for
  // every coding block.  A repeat call for an already-coded 64x64 is a no-op.
  bool write(av2::Encoder& encoder, int mi_row, int mi_col, int idx,
             std::string* error);
};

// Decoder mirror for round-trip tests.
bool read_tile_index(av2::Decoder& decoder, TileIndexWriter* tile, int mi_row,
                     int mi_col, int* idx, std::string* error);

// Writes the whole frame's cdef_idx grid as a standalone §8.2 stream, in the
// exact coding order (64x64 raster).  This is the byte/bit cost of the index
// syntax and the reference for the tile-writer hook.
bool write_index_symbols(const Plan& plan, std::vector<std::uint8_t>* payload,
                         std::string* error);

// Decodes a standalone stream produced by write_index_symbols() and checks it
// against `plan.cdef_idx`.
bool read_index_symbols(std::span<const std::uint8_t> payload,
                        const Plan& plan, std::vector<std::int8_t>* grid,
                        std::string* error);

// Best-effort §7.18.1 grid materialisation from optional per-MI inputs.
// Returns false when a segment id falls outside the lossless array.
bool build_mi_grids(int mi_cols, int mi_rows, const std::uint8_t* skip,
                    const std::uint8_t* segment_id,
                    const std::uint8_t* chroma_segment_id,
                    const std::uint8_t* lossless_array,
                    int lossless_array_size, FrameGrids* out,
                    std::string* error);

}  // namespace av2enc::m2::cdef_adapter

#endif  // AV2ENC_ENCODER_M2_CDEF_M2_CDEF_H_
