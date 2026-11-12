// Intra block copy CPU tests (§5.20.5.4, §6.19.7.12, §7.13.3.17/.18).
#include "codec/intra/ibc/core/ibc.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

using av2enc::ibc::BlockVector;
using av2enc::ibc::IbcBufferState;
using av2enc::ibc::IbcConfig;
using av2enc::ibc::Match;
using av2enc::ibc::NeighbourBvs;
using av2enc::ibc::SearchOptions;

void test_plane_offset() {
  check(av2enc::ibc::plane_offset(-2560, 0) == -320, "luma offset even");
  check(av2enc::ibc::plane_offset(-2560, 1) == -160, "chroma offset even");
  check(av2enc::ibc::plane_offset(0, 0) == 0, "zero offset luma");
  check(av2enc::ibc::plane_offset(0, 1) == 0, "zero offset chroma");
  // Odd BVs use the §7.13.3.17 rounding, not a truncating divide.
  const int odd = -2561;
  const int luma = av2enc::ibc::plane_offset(odd, 0);
  const int chroma = av2enc::ibc::plane_offset(odd, 1);
  check(luma == -321, "odd luma offset rounds down");
  check(chroma == -161, "odd chroma offset");
}

void test_stack_order() {
  const int w = 512;
  const int h = 128;
  auto screen = av2enc::ibc::make_synthetic_screen(w, h);
  const auto view = screen.view();
  IbcConfig config;
  config.allow_local = false;
  config.allow_global = true;
  IbcBufferState state;
  state.reset(w / 4, h / 4);
  const auto stack = av2enc::ibc::build_stack(
      view, config, state, 0, 5 * 16, static_cast<int>(
          av2enc::partition::BlockSize::k64x32), NeighbourBvs{});
  check(stack.count == 2, "stack capped by max_bvp_drl_bits_minus_1+2");
  check(stack.candidates[0].bv == (BlockVector{-64 * 8, 0}),
        "default 0 is (0,-SbHeight)");
  check(stack.candidates[1].bv == (BlockVector{0, -320 * 8}),
        "default 1 is (-SbWidth-256,0)");

  config.max_bvp_drl_bits_minus_1 = 3;
  const auto wide = av2enc::ibc::build_stack(
      view, config, state, 0, 5 * 16,
      static_cast<int>(av2enc::partition::BlockSize::k64x32),
      NeighbourBvs{});
  check(wide.count == 4, "four defaults fit when DRL bits == 3");
  check(wide.candidates[2].bv == (BlockVector{-32 * 8, 0}),
        "third default is (0,-BlockHeight)");
  check(wide.candidates[3].bv == (BlockVector{0, -64 * 8}),
        "fourth default is (-BlockWidth,0)");
  // Neighbour candidates are inserted before the defaults.
  const BlockVector left{0, -64 * 8};
  NeighbourBvs nb;
  nb.left = &left;
  const auto with_nb = av2enc::ibc::build_stack(
      view, config, state, 0, 5 * 16,
      static_cast<int>(av2enc::partition::BlockSize::k64x32), nb);
  check(with_nb.count >= 1 && with_nb.candidates[0].bv == left,
        "left IBC BV is stack index 0");
  check(av2enc::ibc::find_stack_index(with_nb, left) == 0,
        "find_stack_index finds the neighbour");
}

void test_validity() {
  const int w = 384;
  const int h = 128;
  auto screen = av2enc::ibc::make_synthetic_screen(w, h);
  const auto view = screen.view();
  IbcConfig config;
  config.allow_global = true;
  config.allow_local = false;
  IbcBufferState state;
  state.reset(w / 4, h / 4);
  const int bsize = static_cast<int>(av2enc::partition::BlockSize::k32x64);
  std::string reason;
  // Interior, 5 SB columns back: valid global source.
  check(av2enc::ibc::is_bv_valid(view, config, state, 16, 80, bsize,
                                 BlockVector{0, -320 * 8}, &reason),
        "global source 5 SBs back is valid: " + reason);
  // Current block: inside the delay window.
  check(!av2enc::ibc::is_bv_valid(view, config, state, 16, 80, bsize,
                                  BlockVector{0, 0}, &reason),
        "zero BV is rejected");
  // Source outside the tile.
  check(!av2enc::ibc::is_bv_valid(view, config, state, 16, 80, bsize,
                                  BlockVector{0, -400 * 8}, &reason),
        "source left of the tile is rejected");
  // Too close: active col 3, source col 0 is inside the 4-SB delay window.
  check(!av2enc::ibc::is_bv_valid(view, config, state, 16, 48, bsize,
                                  BlockVector{0, -192 * 8}, &reason),
        "source inside INTRABC_DELAY_SB64 is rejected");
  // Above source at row 1 is valid.
  check(av2enc::ibc::is_bv_valid(view, config, state, 16, 80, bsize,
                                 BlockVector{-64 * 8, 0}, &reason),
        "above source row is valid");
  // A source that extends past the bottom of the tile is rejected.
  check(!av2enc::ibc::is_bv_valid(
            view, config, state, 16, 80,
            static_cast<int>(av2enc::partition::BlockSize::k64x64),
            BlockVector{64 * 8, 0}, &reason),
        "source block crossing the tile edge is rejected");
}

void test_local_buffer() {
  const int w = 384;
  const int h = 64;
  auto screen = av2enc::ibc::make_synthetic_screen(w, h);
  const auto view = screen.view();
  IbcConfig config;
  config.allow_global = false;
  config.allow_local = true;
  IbcBufferState state;
  state.reset(w / 4, h / 4);
  const int bsize = static_cast<int>(av2enc::partition::BlockSize::k32x64);
  // Decode four 64x64 regions (two 32x64 leaves each).
  for (int sb = 0; sb < 4; ++sb) {
    state.update_buffers(0, sb * 16);
    state.mark_coded(0, sb * 16, 64, 64);
  }
  state.update_buffers(0, 4 * 16);
  state.mark_coded(0, 4 * 16, 64, 64);
  state.update_buffers(0, 5 * 16);
  state.mark_coded(0, 5 * 16, 64, 64);
  // At SB5 the four most recent regions are cols 2..5; col 0 and 1 have been
  // evicted (col 0 first, then col 1).
  std::string reason;
  check(!av2enc::ibc::is_bv_valid(view, config, state, 0, 5 * 16, bsize,
                                  BlockVector{0, -5 * 64 * 8}, &reason),
        "evicted local buffer is rejected: " + reason);
  check(av2enc::ibc::is_bv_valid(view, config, state, 0, 5 * 16, bsize,
                                 BlockVector{0, -3 * 64 * 8}, &reason),
        "recent local buffer is accepted: " + reason);
  // A source in the current block is rejected.
  check(!av2enc::ibc::is_bv_valid(view, config, state, 0, 5 * 16, bsize,
                                  BlockVector{0, 0}, &reason),
        "current-block overlap is rejected");
}

void test_search_and_copy() {
  const int w = 384;
  const int h = 192;
  auto screen = av2enc::ibc::make_synthetic_screen(w, h);
  const auto view = screen.view();
  IbcConfig config;
  config.allow_global = true;
  config.allow_local = false;
  IbcBufferState state;
  state.reset(w / 4, h / 4);

  // Repeated 64x32 tile row: at row 1, column 5 the horizontal default
  // candidate is exactly the same tile.
  const int bsize = static_cast<int>(av2enc::partition::BlockSize::k64x32);
  const int mi_row = 8;       // y = 32
  const int mi_col = 5 * 16;  // x = 320
  SearchOptions options;
  options.stack_only = true;
  const Match stack_match = av2enc::ibc::search_block(
      view, config, state, mi_row, mi_col, bsize, options, nullptr);
  check(stack_match.found, "stack search finds a candidate");
  check(stack_match.sad == 0, "stack search finds an exact tile");
  check(stack_match.stack_index == 1, "exact tile is the (-320,0) default");

  options.stack_only = false;
  options.step = 4;
  const Match full = av2enc::ibc::search_block(view, config, state, mi_row,
                                               mi_col, bsize, options,
                                               nullptr);
  check(full.found && full.sad == 0, "full search finds an exact tile");

  // Block copy is exact for the chosen BV.
  av2enc::ibc::CopyResult copy;
  std::string error;
  check(av2enc::ibc::predict_copy(view, mi_row, mi_col, bsize,
                                  stack_match.bv, 64, 32, copy, &error),
        "predict_copy: " + error);
  const int x = mi_col * 4;
  const int y = mi_row * 4;
  const int dx = av2enc::ibc::plane_offset(stack_match.bv.col, 0);
  const int dy = av2enc::ibc::plane_offset(stack_match.bv.row, 0);
  int diff = 0;
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 64; ++c) {
      const int src = view.y.data[static_cast<std::size_t>(y + dy + r) *
                                      static_cast<std::size_t>(view.y.stride) +
                                  static_cast<std::size_t>(x + dx + c)];
      if (src != copy.y[static_cast<std::size_t>(r) * static_cast<std::size_t>(64) +
                     static_cast<std::size_t>(c)]) ++diff;
    }
  }
  check(diff == 0, "IBC copy is byte-exact");
  // Transform-skip residual is zero for an exact copy.
  std::vector<std::int16_t> residual(static_cast<std::size_t>(64) * 32);
  av2enc::ibc::copy_then_residual(view, mi_row, mi_col, bsize, stack_match.bv,
                                  residual.data(), 64, nullptr, 0, nullptr, 0,
                                  64, 32);
  int nz = 0;
  for (std::int16_t v : residual) nz += v != 0 ? 1 : 0;
  check(nz == 0, "IBC transform-skip residual is zero");

  // Flat region: any valid source is exact, but the delay still applies.
  const Match flat = av2enc::ibc::search_block(view, config, state, 0, 5 * 16,
                                               bsize, options, nullptr);
  check(flat.found && flat.sad == 0, "flat region copy");
}

void test_chroma_copy() {
  const int w = 256;
  const int h = 128;
  auto screen = av2enc::ibc::make_synthetic_screen(w, h);
  const auto view = screen.view();
  // Chroma planes are flat in the generator; compare a copy that is exact.
  const int mi_row = 16;
  const int mi_col = 80;
  const BlockVector bv{0, -320 * 8};
  av2enc::ibc::CopyResult copy;
  std::string error;
  check(av2enc::ibc::predict_copy(view, mi_row, mi_col,
                                  static_cast<int>(
                                      av2enc::partition::BlockSize::k32x64),
                                  bv, 32, 64, copy, &error),
        "chroma predict_copy: " + error);
  const int cdx = av2enc::ibc::plane_offset(bv.col, 1);
  const int cdy = av2enc::ibc::plane_offset(bv.row, 1);
  const int cx = (mi_col * 4) / 2;
  const int cy = (mi_row * 4) / 2;
  int diff = 0;
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 16; ++c) {
      const int src = view.u.data[static_cast<std::size_t>(cy + cdy + r) *
                                      static_cast<std::size_t>(view.u.stride) +
                                  static_cast<std::size_t>(cx + cdx + c)];
      if (src != copy.u[static_cast<std::size_t>(r) * static_cast<std::size_t>(16) +
                     static_cast<std::size_t>(c)]) ++diff;
    }
  }
  check(diff == 0, "4:2:0 chroma copy is byte-exact");
}

void test_synthetic_quality() {
  // Report the IBC skip error on the repeated row: an exact copy is lossless.
  const int w = 384;
  const int h = 192;
  auto screen = av2enc::ibc::make_synthetic_screen(w, h);
  const auto view = screen.view();
  IbcConfig config;
  config.allow_global = true;
  config.allow_local = false;
  IbcBufferState state;
  state.reset(w / 4, h / 4);
  SearchOptions options;
  options.stack_only = true;
  const Match m = av2enc::ibc::search_block(
      view, config, state, 8, 80,
      static_cast<int>(av2enc::partition::BlockSize::k64x32), options, nullptr);
  check(m.found, "quality search");
  av2enc::ibc::CopyResult copy;
  std::string error;
  check(av2enc::ibc::predict_copy(
            view, 8, 80,
            static_cast<int>(av2enc::partition::BlockSize::k64x32), m.bv, 64,
            32, copy, &error),
        "quality copy");
  const std::int64_t sse = av2enc::ibc::block_sse(
      copy.y.data(), 64,
      view.y.data +
          static_cast<std::size_t>(32) *
              static_cast<std::size_t>(view.y.stride) +
          static_cast<std::size_t>(320),
      view.y.stride, 64, 32);
  std::printf("synthetic IBC: bv=(%d,%d), sad=%lld, sse=%lld\n", m.bv.row,
              m.bv.col, static_cast<long long>(m.sad),
              static_cast<long long>(sse));
  check(sse == 0, "IBC skip is lossless on repeated screen content");
}

}  // namespace

int main() {
  test_plane_offset();
  test_stack_order();
  test_validity();
  test_local_buffer();
  test_search_and_copy();
  test_chroma_copy();
  test_synthetic_quality();
  std::printf("ibc_test: %d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
