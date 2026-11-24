// §6.17.2 loop-restoration CDF rows in the §8.2 frame image.
//
// init_non_coeff_cdfs() names UseWienerNsCdf, WienerNsLengthCdf,
// WienerNsUvSymCdf, WienerNsBaseCdf, UsePcWienerCdf and FlexRestorationTypeCdf.
// §7.4.4 load_cdfs()/§7.5 save_cdfs() copy every array named by
// init_non_coeff_cdfs(), so a frame that carries the §8.2 image must carry
// these six rows too.  When they were omitted, --cdf-continuity restarted the
// tile LR CDFs at the §9.3 defaults every frame while the decoder started from
// the previous frame's adapted rows: the first LR-coded frame after the first
// adaptation (frame 3 of the 60-frame nightly stream) failed to decode.
//
// This test is decoder-free: it checks that
//   * the packed CDF image transports the six rows (and defaults to §9.3),
//   * filters_stream::LrState seed/store round-trips them,
//   * pc_wiener_stream::TileState seed/store round-trips them, including the
//     §5.20.10.5 use_pc_wiener row.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "codec/filters/loop_restoration/lr_cdf_image.h"
#include "codec/filters/pc_wiener/pc_wiener_stream.h"
#include "codec/filters/stream/filters_stream.h"
#include "pipeline/video/video_syntax.h"
#include "pipeline/video/video_syntax_writer.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what, int line) {
  if (!ok) {
    std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
    ++g_failures;
  }
}

#define CHECK(cond) Check((cond), #cond, __LINE__)

using av2enc::filters::lr_cdf::Image;

// A value that cannot be a §9.3 default for the touched rows.
constexpr std::uint16_t kMarker = 0x1234;

void TestImageCarriesLrRows() {
  std::string error;
  std::vector<std::uint16_t> image;
  CHECK(av2enc::video::syntax::cdf_image_defaults_for_base_q_idx(
      90, image, error));
  CHECK(error.empty());
  CHECK(!image.empty());

  av2enc::video::syntax::detail::CdfState state{};
  av2enc::video::syntax::detail::init_cdfs(state, /*qctx_slice=*/0);
  av2enc::video::syntax::detail::import_cdf_image(state, image.data());
  // The default image must reproduce the §9.3 LR rows.
  Image default_rows;
  av2enc::filters::lr_cdf::SetDefaults(&default_rows);
  const Image& imported = state.lr_cdfs;
  CHECK(imported.use_wiener_ns[0] == default_rows.use_wiener_ns[0]);
  CHECK(imported.wiener_ns_length[1][0] ==
        default_rows.wiener_ns_length[1][0]);
  CHECK(imported.wiener_ns_base[4] == default_rows.wiener_ns_base[4]);

  // Mutate every row, export, import into a fresh state, and require the
  // adapted values to survive.  Without the rows in cdf_arrays() the fresh
  // state keeps its §9.3 defaults and this fails.
  state.lr_cdfs.use_wiener_ns[0] = kMarker;
  state.lr_cdfs.wiener_ns_length[0][2] = kMarker + 1;
  state.lr_cdfs.wiener_ns_uv_sym[2] = kMarker + 2;
  state.lr_cdfs.wiener_ns_base[3] = kMarker + 3;
  state.lr_cdfs.use_pc_wiener[1] = kMarker + 4;
  state.lr_cdfs.flex_restoration_type[1][2][0] = kMarker + 5;
  std::vector<std::uint16_t> adapted(image.size());
  av2enc::video::syntax::detail::export_cdf_image(state, adapted.data());
  CHECK(adapted != image);

  av2enc::video::syntax::detail::CdfState round{};
  av2enc::video::syntax::detail::init_cdfs(round, /*qctx_slice=*/0);
  av2enc::video::syntax::detail::import_cdf_image(round, adapted.data());
  CHECK(round.lr_cdfs.use_wiener_ns[0] == kMarker);
  CHECK(round.lr_cdfs.wiener_ns_length[0][2] == kMarker + 1);
  CHECK(round.lr_cdfs.wiener_ns_uv_sym[2] == kMarker + 2);
  CHECK(round.lr_cdfs.wiener_ns_base[3] == kMarker + 3);
  CHECK(round.lr_cdfs.use_pc_wiener[1] == kMarker + 4);
  CHECK(round.lr_cdfs.flex_restoration_type[1][2][0] == kMarker + 5);
}

void TestFiltersStreamSeedStore() {
  std::string error;
  av2enc::still::filters_stream::LrParams params;
  av2enc::still::filters_stream::LrState state;
  CHECK(av2enc::still::filters_stream::lr_state_init(
      &state, params, /*mi_cols=*/120, /*mi_rows=*/60, /*sub_x=*/1,
      /*sub_y=*/1, /*sb_size=*/64, error));
  CHECK(error.empty());

  Image image;
  av2enc::filters::lr_cdf::SetDefaults(&image);
  image.use_wiener_ns[1] = kMarker;
  image.wiener_ns_base[0] = kMarker + 1;
  av2enc::still::filters_stream::lr_state_seed_cdfs(&state, image);
  CHECK(state.use_wiener_ns[1] == kMarker);
  CHECK(state.wiener_ns_base[0] == kMarker + 1);

  state.use_wiener_ns[2] = kMarker + 2;
  state.flex_restoration_type[2][1][1] = kMarker + 3;
  Image stored;
  av2enc::filters::lr_cdf::SetDefaults(&stored);
  av2enc::still::filters_stream::lr_state_store_cdfs(state, &stored);
  CHECK(stored.use_wiener_ns[1] == kMarker);
  CHECK(stored.use_wiener_ns[2] == kMarker + 2);
  CHECK(stored.wiener_ns_base[0] == kMarker + 1);
  CHECK(stored.flex_restoration_type[2][1][1] == kMarker + 3);
}

void TestPcWienerSeedStore() {
  std::string error;
  av2enc::filters::pc_wiener_stream::LrParams params;
  params.frame_type[0] =
      av2enc::filters::pc_wiener_stream::kRestorePcWiener;
  av2enc::filters::pc_wiener_stream::TileState state;
  CHECK(av2enc::filters::pc_wiener_stream::InitTileState(
      &state, params, /*mi_cols=*/120, /*mi_rows=*/60, /*sub_x=*/1,
      /*sub_y=*/1, error));
  CHECK(error.empty());

  Image image;
  av2enc::filters::lr_cdf::SetDefaults(&image);
  image.use_pc_wiener[0] = kMarker;
  image.wiener_ns_uv_sym[1] = kMarker + 1;
  av2enc::filters::pc_wiener_stream::TileStateSeedCdfs(&state, image);
  CHECK(state.use_pc_wiener[0] == kMarker);
  CHECK(state.wiener_ns_uv_sym[1] == kMarker + 1);

  state.use_pc_wiener[2] = kMarker + 2;
  state.wiener_ns_length[1][1] = kMarker + 3;
  Image stored;
  av2enc::filters::lr_cdf::SetDefaults(&stored);
  av2enc::filters::pc_wiener_stream::TileStateStoreCdfs(state, &stored);
  CHECK(stored.use_pc_wiener[0] == kMarker);
  CHECK(stored.use_pc_wiener[2] == kMarker + 2);
  CHECK(stored.wiener_ns_uv_sym[1] == kMarker + 1);
  CHECK(stored.wiener_ns_length[1][1] == kMarker + 3);
}

}  // namespace

int main() {
  TestImageCarriesLrRows();
  TestFiltersStreamSeedStore();
  TestPcWienerSeedStore();
  if (g_failures != 0) {
    std::fprintf(stderr, "lr_cdf_image_test: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("lr_cdf_image_test: PASS\n");
  return 0;
}
