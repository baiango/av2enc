// Chroma intra decision for the M2 video path — AV2 v1.0.0 §7.13.5.
//
// The M2 writer codes intra chroma with the same DC / CfL candidate set the M1
// still product uses (`choose_uv_mode()` in src/pipeline/still/still_encoder.cpp).
// This header is the video-side mirror of that subset and is shared by the CPU
// reference (`video_frame.cpp`) and the CPU-emulation backend, so both make the
// identical decision; the CUDA backend runs `still::chroma_decide()` from
// src/cuda/pipeline/still/chroma/, which is bit-exact against the same policy.
//
// Scope: 8-bit 4:2:0, derived alpha (CFL_DERIVED_ALPHA, cfl_index = 1).  The
// chroma block is at most 32x32 (64x64 luma), so the §7.13.2.11 DC-subsampled
// dispatch never applies.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace av2enc::video::chroma_search {

// One chroma plane view.  `recon` holds the reconstructed plane up to (not
// including) the current block and receives the transient prediction; `source`
// is the visible frame used for SSE scoring.
struct PlaneView {
  std::uint8_t* recon = nullptr;
  int recon_stride = 0;
  const std::uint8_t* source = nullptr;
  int source_stride = 0;
  int source_max_x = 0;  // inclusive, visible chroma width - 1
  int source_max_y = 0;  // inclusive, visible chroma height - 1
  // Allocated reconstruction plane size.  A bottom/right-edge coding block can
  // extend past the coded MI-aligned plane (e.g. a 64x64 leaf whose PARTITION_NONE
  // is legal because has_rows is true); the decoder reconstructs it into its
  // padded frame buffer, so the search must use a padded scratch plane instead
  // of writing past `recon`.
  int recon_width = 0;
  int recon_height = 0;
};

struct Options {
  // §5.4.5 enable_cfl_intra (always 1 in the M2 sequence header) and the
  // opt-in --video-cfl decision.  With this false the helper returns DC.
  bool enable_cfl = false;
  // `cfl_ds_filter_index` (0 for 4:2:0).
  int ds_filter = 0;
  // §7.13.6-.8 MHCCP: also score the three CFL_MULTI directions.  Implies
  // enable_cfl.
  bool enable_mhccp = false;
};

struct Choice {
  bool is_cfl = false;
  // CFL_DERIVED_ALPHA.  The writer codes is_cfl + this index only; derived
  // alpha is not signalled.
  int cfl_index = 1;
  // §7.13.7 CFL_MULTI direction when cfl_index == 2.
  int cfl_mh_dir = 0;
  int alpha_u_q8 = 0;
  int alpha_v_q8 = 0;
  // Winning prediction, w*h samples, tightly packed.
  std::vector<std::uint8_t> u;
  std::vector<std::uint8_t> v;
  // Winning decision score (U-only for DC, U+V for CfL), matching the M1
  // `UvChoice::sse`.
  std::int64_t sse = 0;
};

// `dc_u`/`dc_v` are the DC predictions the caller already computed (w*h).  The
// reconstructed luma plane must already contain this block's samples: CfL reads
// the co-located luma and the above/left reconstructed luma window.
//
// On return `recon_u`/`recon_v` hold the winning prediction over the block
// (the caller overwrites them with the final reconstruction).  `dc_sse_u` is the
// DC candidate score the caller measured on the U plane.
bool choose(const PlaneView& u_plane, const PlaneView& v_plane,
            const std::uint8_t* recon_luma, int luma_stride,
            int luma_coded_max_x, int luma_coded_max_y, int sb_top_y,
            int x, int y, int w, int h, int coded_width, int coded_height,
            int num4_above_right, int num4_below_left,
            const std::vector<std::uint8_t>& dc_u,
            const std::vector<std::uint8_t>& dc_v, std::int64_t dc_sse_u,
            const Options& options, Choice& out, std::string& error);

}  // namespace av2enc::video::chroma_search
