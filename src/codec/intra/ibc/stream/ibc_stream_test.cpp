// Tests for the M1 IBC syntax adapter, stack, border rule and minimal stream.
#include "codec/intra/ibc/stream/ibc_stack.h"
#include "codec/intra/ibc/stream/ibc_syntax.h"
#include "codec/intra/ibc/stream/still_ibc_stream.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace av2 = av2enc::av2;

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

using namespace av2enc::still::ibc_stream;
using av2enc::ibc::BlockVector;
using av2enc::ibc::FrameView;
using av2enc::ibc::IbcBufferState;
using av2enc::ibc::IbcConfig;

constexpr int kBs64x32 = 11;
constexpr int kBs64x64 = 12;

FrameView view_of(const av2enc::ibc::SyntheticScreen& s) { return s.view(); }

void test_cdf_init() {
  IbcCdfs cdfs{};
  init_ibc_cdfs(cdfs);
  check(cdfs.intrabc[0][0] == 32085, "intrabc CDF copied");
  check(cdfs.intrabc_mode[0] == 29993, "intrabc_mode CDF copied");
  check(cdfs.skip[5][0] == 3320, "skip CDF copied");
  check(cdfs.mv.shell_class0[3][0] == 8680,
        "shell class0 P=ONE_PEL CDF copied");
  check(cdfs.mv.shell_class1[6][7] == 55,
        "shell class1 P=EIGHTH_PEL CDF copied");
  check(cdfs.mv.shell_offset_low_class[1][1][0] == 20966,
        "shell low-class context copied");
}

struct DecodedMv {
  int ctx = 0;
  int precision = 0;
  BlockVector diff{};
};

void test_mv_roundtrip() {
  const std::array<int, 6> precisions{0, 1, 3, 4, 5, 6};
  const std::array<int, 15> mags{0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31,
                                 63, 64, 255, 1000};
  const std::array<std::array<int, 2>, 6> signs{
      {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {0, 1}, {1, 0}}};
  for (int freeze = 0; freeze < 2; ++freeze) {
    IbcCdfs enc_cdfs{};
    IbcCdfs dec_cdfs{};
    init_ibc_cdfs(enc_cdfs);
    init_ibc_cdfs(dec_cdfs);
    std::vector<std::uint8_t> payload(1U << 20);
    av2::Encoder enc;
    enc.init_symbol(payload.data(), payload.size());
    enc.disable_cdf_update = freeze == 1 ? 1 : 0;
    std::vector<DecodedMv> expected;
    std::string error;
    int n = 0;
    for (int p : precisions) {
      const int quantum = 1 << (kMvPrecisionEighthPel - p);
      for (int mr : mags) {
        for (int mc : mags) {
          for (const auto& s : signs) {
            BlockVector diff{s[0] * mr * quantum, s[1] * mc * quantum};
            const int ctx = n & 1;
            if (!write_mv(enc, enc_cdfs.mv, ctx, p, diff, false, error)) {
              check(false, "write_mv: " + error);
              return;
            }
            expected.push_back(DecodedMv{ctx, p, diff});
            ++n;
          }
        }
      }
    }
    std::size_t sz = 0;
    check(enc.exit_symbol(&sz) == av2::Status::kOk, "mv exit_symbol");
    av2::Decoder dec;
    check(dec.init_symbol(payload.data(), payload.size(), sz) ==
              av2::Status::kOk,
          "mv init_symbol");
    dec.disable_cdf_update = freeze == 1 ? 1 : 0;
    for (const DecodedMv& e : expected) {
      BlockVector got{};
      if (!read_mv(dec, dec_cdfs.mv, e.ctx, e.precision, false, got, error)) {
        check(false, "read_mv: " + error);
        return;
      }
      check(got == e.diff, "MV roundtrip p=" + std::to_string(e.precision) +
                               " diff=(" + std::to_string(e.diff.row) + "," +
                               std::to_string(e.diff.col) + ")");
    }
    check(dec.exit_symbol() == av2::Status::kOk, "mv decoder exit_symbol");
  }

  IbcCdfs cdfs{};
  init_ibc_cdfs(cdfs);
  std::vector<std::uint8_t> payload(1024);
  av2::Encoder enc;
  enc.init_symbol(payload.data(), payload.size());
  std::string error;
  check(!write_mv(enc, cdfs.mv, 1, kMvPrecisionOnePel, BlockVector{1, 0},
                  false, error),
        "MV rejects a value below the one-pel quantum");
  check(!write_mv(enc, cdfs.mv, 2, kMvPrecisionOnePel, BlockVector{0, 0},
                  false, error),
        "MV rejects an invalid context");
}

BvStack make_test_stack() {
  BvStack stack;
  const std::array<BlockVector, 5> bvs{
      BlockVector{0, -320 * 8}, BlockVector{-64 * 8, 0},
      BlockVector{0, -384 * 8}, BlockVector{-128 * 8, 0},
      BlockVector{0, -256 * 8}};
  for (int i = 0; i < 4; ++i) {
    stack.entries[static_cast<std::size_t>(i)].bv =
        bvs[static_cast<std::size_t>(i)];
    stack.entries[static_cast<std::size_t>(i)].valid = true;
  }
  stack.count = 4;
  return stack;
}

void test_intrabc_info_roundtrip() {
  for (int force = 0; force < 2; ++force) {
    for (int freeze = 0; freeze < 2; ++freeze) {
      IbcCdfs enc_cdfs{};
      IbcCdfs dec_cdfs{};
      init_ibc_cdfs(enc_cdfs);
      init_ibc_cdfs(dec_cdfs);
      ScreenContentConfig config;
      config.force_integer_mv = force == 1;
      config.max_bvp_drl_bits_minus_1 = force == 1 ? 2 : 0;
      const BvStack stack = make_test_stack();
      std::vector<std::uint8_t> payload(1U << 18);
      av2::Encoder enc;
      enc.init_symbol(payload.data(), payload.size());
      enc.disable_cdf_update = freeze == 1 ? 1 : 0;
      struct Record {
        BvSyntax syntax;
        BlockVector bv;
      };
      std::vector<Record> records;
      std::string error;
      const int max_ref = force == 1 ? 3 : 1;
      for (int ref = 0; ref <= max_ref; ++ref) {
        BvSyntax near;
        near.mode = IntrabcMode::kNearMv;
        near.ref_mv_idx = ref;
        const BlockVector base = stack_bv_at(stack, ref);
        if (!write_intrabc_info(enc, enc_cdfs, config, stack, near, base,
                                error)) {
          check(false, "write NEARMV: " + error);
          return;
        }
        records.push_back(Record{near, base});

        const std::array<BlockVector, 3> diffs{
            BlockVector{512, -512}, BlockVector{-1024, 256},
            BlockVector{0, 0}};
        for (const BlockVector& diff : diffs) {
          BvSyntax nw;
          nw.mode = IntrabcMode::kNewMv;
          nw.ref_mv_idx = ref;
          nw.diff = diff;
          if (!force) {
            nw.intrabc_precision_quarter = (ref & 1) != 0;
          }
          const int quantum =
              (!force && nw.intrabc_precision_quarter) ? 2 : 8;
          BlockVector quantised = diff;
          quantised.row =
              (diff.row / quantum) * quantum;
          quantised.col = (diff.col / quantum) * quantum;
          nw.diff = quantised;
          const BlockVector target{base.row + quantised.row,
                                   base.col + quantised.col};
          if (!write_intrabc_info(enc, enc_cdfs, config, stack, nw, target,
                                  error)) {
            check(false, "write NEWMV: " + error);
            return;
          }
          records.push_back(Record{nw, target});
        }
      }
      std::size_t sz = 0;
      check(enc.exit_symbol(&sz) == av2::Status::kOk, "intrabc exit_symbol");
      av2::Decoder dec;
      check(dec.init_symbol(payload.data(), payload.size(), sz) ==
                av2::Status::kOk,
            "intrabc init_symbol");
      dec.disable_cdf_update = freeze == 1 ? 1 : 0;
      for (const Record& r : records) {
        BvSyntax got;
        BlockVector bv{};
        if (!read_intrabc_info(dec, dec_cdfs, config, stack, got, bv, error)) {
          check(false, "read_intrabc_info: " + error);
          return;
        }
        check(got.mode == r.syntax.mode, "intrabc mode roundtrip");
        check(got.ref_mv_idx == r.syntax.ref_mv_idx,
              "RefMvIdx roundtrip (force=" + std::to_string(force) + ")");
        check(bv == r.bv, "intrabc BV roundtrip");
        check(got.diff == r.syntax.diff, "intrabc MVD roundtrip");
        check(got.intrabc_precision_quarter ==
                  r.syntax.intrabc_precision_quarter,
              "intrabc_precision roundtrip");
      }
      check(dec.exit_symbol() == av2::Status::kOk,
            "intrabc decoder exit_symbol");
    }
  }

  ScreenContentConfig config;
  config.force_integer_mv = true;
  config.max_bvp_drl_bits_minus_1 = 0;
  const BvStack stack = make_test_stack();
  IbcCdfs cdfs{};
  init_ibc_cdfs(cdfs);
  std::vector<std::uint8_t> payload(1024);
  av2::Encoder enc;
  enc.init_symbol(payload.data(), payload.size());
  BvSyntax near;
  near.mode = IntrabcMode::kNearMv;
  near.ref_mv_idx = 1;
  std::string error;
  check(!write_intrabc_info(enc, cdfs, config, stack, near,
                            BlockVector{0, 0}, error),
        "NEARMV rejects a target that is not the stack candidate");
}

void test_use_intrabc_and_skip() {
  for (int freeze = 0; freeze < 2; ++freeze) {
    IbcCdfs enc_cdfs{};
    IbcCdfs dec_cdfs{};
    init_ibc_cdfs(enc_cdfs);
    init_ibc_cdfs(dec_cdfs);
    std::vector<std::uint8_t> payload(4096);
    av2::Encoder enc;
    enc.init_symbol(payload.data(), payload.size());
    enc.disable_cdf_update = freeze == 1 ? 1 : 0;
    std::string error;
    const std::array<bool, 8> bits{false, true, true, false, true,
                                   false, true, true};
    for (std::size_t i = 0; i < bits.size(); ++i) {
      const int ctx = static_cast<int>(i % 3);
      check(write_use_intrabc(enc, enc_cdfs, ctx, bits[i], error),
            "write use_intrabc");
      check(write_skip_flag(enc, enc_cdfs, ctx, bits[(i + 1) % bits.size()],
                            error),
            "write skip_flag");
    }
    std::size_t sz = 0;
    check(enc.exit_symbol(&sz) == av2::Status::kOk, "uexit");
    av2::Decoder dec;
    check(dec.init_symbol(payload.data(), payload.size(), sz) ==
              av2::Status::kOk,
          "uinit");
    dec.disable_cdf_update = freeze == 1 ? 1 : 0;
    for (std::size_t i = 0; i < bits.size(); ++i) {
      const int ctx = static_cast<int>(i % 3);
      bool got = false;
      check(read_use_intrabc(dec, dec_cdfs, ctx, got, error) && got == bits[i],
            "use_intrabc roundtrip");
      check(read_skip_flag(dec, dec_cdfs, ctx, got, error) &&
                got == bits[(i + 1) % bits.size()],
            "skip_flag roundtrip");
    }
    check(dec.exit_symbol() == av2::Status::kOk, "udec exit");
  }
  IbcCdfs cdfs{};
  init_ibc_cdfs(cdfs);
  std::vector<std::uint8_t> payload(64);
  av2::Encoder enc;
  enc.init_symbol(payload.data(), payload.size());
  std::string error;
  check(!write_use_intrabc(enc, cdfs, 3, true, error),
        "use_intrabc rejects an invalid context");
  check(!write_skip_flag(enc, cdfs, 6, true, error),
        "skip_flag rejects an invalid context");
}


void test_mode_info_neighbours() {
  const int mi_cols = 192;
  const int mi_rows = 32;
  std::vector<std::uint8_t> ibc(
      static_cast<std::size_t>(mi_cols * mi_rows), 0);
  std::vector<std::uint8_t> skips(ibc.size(), 0);
  // One IBC block at superblock row 1, column 6 (64x32).
  for (int r = 16; r < 24; ++r) {
    for (int c = 96; c < 112; ++c) {
      ibc[static_cast<std::size_t>(r * mi_cols + c)] = 1;
      skips[static_cast<std::size_t>(r * mi_cols + c)] = 1;
    }
  }

  // The next block to the right is at the top of the superblock row.  NPos
  // visits its bottom-left and left positions, both inside the same wide IBC
  // block, so the context is 2 (the old above+left approximation gave 1).
  const MiNeighbours n =
      mode_info_neighbours(16, 112, kBs64x32, mi_cols, mi_rows, 64, false);
  check(n.count == 2, "NPos visits two positions at a superblock top row");
  check(n.row[0] == 23 && n.col[0] == 111, "NPos bottom-left position");
  check(n.row[1] == 16 && n.col[1] == 111, "NPos left position");
  check(ibc_symbol_context(ibc, mi_cols, mi_rows, 16, 112, kBs64x32, 64) == 2,
        "use_intrabc context counts both NPos visits");
  check(skip_symbol_context(skips, mi_cols, mi_rows, 16, 112, kBs64x32, 64) ==
            1,
        "skip_flag uses the NPosBuf list (above-right is not the same block)");

  // NPos stops after two entries; above positions across a superblock border
  // are excluded.  A block below the IBC block sees bottom-left/above-right
  // first, not the IBC block directly above it.
  const MiNeighbours below =
      mode_info_neighbours(24, 96, kBs64x32, mi_cols, mi_rows, 64, false);
  check(below.count == 2, "NPos is capped at two positions");
  check(below.row[0] == 31 && below.col[0] == 95,
        "NPos below-left comes first");
  check(below.row[1] == 23 && below.col[1] == 111,
        "NPos above-right comes second");

  // NPosBuf allows the same positions across superblock rows.
  const MiNeighbours buffered =
      mode_info_neighbours(24, 96, kBs64x32, mi_cols, mi_rows, 64, true);
  check(buffered.count == 2, "NPosBuf is capped at two positions");
  check(buffered.row[0] == 31 && buffered.col[0] == 95,
        "NPosBuf below-left comes first");
  check(buffered.row[1] == 23 && buffered.col[1] == 111,
        "NPosBuf above-right comes second");
}

void test_stack_and_border() {
  const int w = 768;
  const int h = 256;
  auto screen = av2enc::ibc::make_synthetic_screen(w, h);
  const FrameView view = view_of(screen);
  IbcConfig config;
  config.allow_global = true;
  config.allow_local = false;
  config.max_bvp_drl_bits_minus_1 = 2;
  IbcBufferState state;
  state.reset(w / 4, h / 4);
  BvNeighbourMap map;
  map.reset(w / 4, h / 4);
  std::string error;

  // Empty map: the four defaults, in add_to_ref_bv() order.
  const BvStack defaults =
      build_bv_stack(view, config, state, 16, 176, kBs64x32, map);
  check(defaults.count == 4, "four defaults fit when max_bvp=2");
  check(defaults.entries[0].bv == (BlockVector{-64 * 8, 0}),
        "default 0 is (0,-SbHeight)");
  check(defaults.entries[1].bv == (BlockVector{0, -320 * 8}),
        "default 1 is (-SbWidth-256,0)");
  check(defaults.entries[2].bv == (BlockVector{-32 * 8, 0}),
        "default 2 is (0,-BlockHeight)");
  check(defaults.entries[3].bv == (BlockVector{0, -64 * 8}),
        "default 3 is (-BlockWidth,0)");

  // A left IBC block and an above IBC block enter before the defaults.
  const BlockVector left_bv{0, -448 * 8};
  const BlockVector above_bv{-64 * 8, -384 * 8};
  map.mark_block(16, 160, 64, 32, true, left_bv);
  map.mark_block(8, 176, 64, 32, true, above_bv);
  const BvStack neighbours =
      build_bv_stack(view, config, state, 16, 176, kBs64x32, map);
  check(neighbours.count == 4, "neighbour stack is capped");
  check(neighbours.entries[0].bv == left_bv,
        "bottom-left scan inserts the left IBC BV first");
  check(neighbours.entries[1].bv == above_bv,
        "above-right scan inserts the above IBC BV second");
  check(find_bv_index(neighbours, left_bv) == 0, "find_bv_index left");
  check(find_bv_index(neighbours, BlockVector{1, 2}) == -1,
        "find_bv_index missing");

  // Duplicate neighbour BVs are pruned (weights are not modelled).
  BvNeighbourMap dup;
  dup.reset(w / 4, h / 4);
  dup.mark_block(16, 160, 64, 32, true, left_bv);
  dup.mark_block(8, 176, 64, 32, true, left_bv);
  const BvStack dedup =
      build_bv_stack(view, config, state, 16, 176, kBs64x32, dup);
  int occurrences = 0;
  for (int i = 0; i < dedup.count; ++i) {
    if (dedup.entries[static_cast<std::size_t>(i)].bv == left_bv) {
      ++occurrences;
    }
  }
  check(occurrences == 1, "duplicate neighbour BV is inserted once");

  // §6.19.7.12 validity: a valid global source and an invalid one.
  IbcConfig global;
  global.allow_global = true;
  global.allow_local = false;
  check(av2enc::ibc::is_bv_valid(view, global, state, 16, 96, kBs64x32,
                                 BlockVector{0, -320 * 8}, &error),
        "interior global BV is valid: " + error);
  check(!av2enc::ibc::is_bv_valid(view, global, state, 16, 96, kBs64x32,
                                  BlockVector{0, 0}, &error),
        "zero BV is invalid");
  check(!av2enc::ibc::is_bv_valid(view, global, state, 16, 96, kBs64x32,
                                  BlockVector{0, -2000 * 8}, &error),
        "source outside the tile is invalid");

  // ±1 reference border: source at x=0 (col 5) fails, x=64 (col 6) passes.
  check(av2enc::ibc::is_bv_valid(view, global, state, 16, 80, kBs64x32,
                                 BlockVector{0, -320 * 8}, nullptr),
        "source at x=0 passes plain geometry");
  check(!is_bv_border_valid(view, global, state, 16, 80, kBs64x32,
                            BlockVector{0, -320 * 8}, &error),
        "source at x=0 fails the ±1 border");
  check(is_bv_border_valid(view, global, state, 16, 96, kBs64x32,
                           BlockVector{0, -320 * 8}, &error),
        "source at x=64 passes the ±1 border: " + error);
  check(!is_bv_border_valid(view, global, state, 16, 96, kBs64x32,
                            BlockVector{-64 * 8, 0}, &error),
        "source at y=0 fails the ±1 top border");
  check(is_bv_encodable(view, global, state, 16, 96, kBs64x32,
                        BlockVector{0, -320 * 8}, &error),
        "encodable combines geometry and border: " + error);
  check(!is_bv_encodable(view, global, state, 16, 80, kBs64x32,
                         BlockVector{0, -320 * 8}, &error),
        "encoder rejects the border-violating BV");

  // Range and delay messages.
  check(!bv_in_range(BlockVector{kMvUp, 0}), "MV_UP is out of range");
  check(!bv_in_range(BlockVector{kMvLow, 0}), "MV_LOW is out of range");
  check(bv_in_range(BlockVector{kMvLow + 1, kMvUp - 1}),
        "open MV range endpoints are accepted");
  check(!av2enc::ibc::is_bv_valid(view, global, state, 16, 80, kBs64x32,
                                  BlockVector{0, -192 * 8}, &error),
        "source inside INTRABC_DELAY_SB64 is rejected");
  check(error.find("delay") != std::string::npos,
        "delay rejection carries the delay reason: " + error);

  // Wavefront: block at row 3, col 0 pointing at row 0, col 11.
  check(!av2enc::ibc::is_bv_valid(view, global, state, 48, 0, kBs64x32,
                                  BlockVector{-3 * 64 * 8, 11 * 64 * 8},
                                  &error),
        "wavefront violation is rejected");
  check(error.find("wavefront") != std::string::npos,
        "wavefront rejection carries the reason: " + error);

  // Local buffer path: a source not in the ring is rejected.
  IbcConfig local;
  local.allow_global = false;
  local.allow_local = true;
  IbcBufferState local_state;
  local_state.reset(w / 4, h / 4);
  for (int sb = 0; sb < 6; ++sb) {
    local_state.update_buffers(0, sb * 16);
    local_state.mark_coded(0, sb * 16, 64, 64);
  }
  local_state.update_buffers(0, 6 * 16);
  check(!av2enc::ibc::is_bv_valid(view, local, local_state, 0, 6 * 16,
                                  kBs64x32, BlockVector{0, -6 * 64 * 8},
                                  &error),
        "evicted local buffer source is rejected");

  // intrabc_available mirrors §5.20.5.3.
  check(intrabc_available(true, kBs64x32), "64x32 is IBC-signallable");
  check(!intrabc_available(true, kBs64x64), "64x64 is not IBC-signallable");
  check(!intrabc_available(true, kBs64x32, true),
        "CHROMA_PART is not IBC-signallable");
  check(!intrabc_available(false, kBs64x32),
        "allow_intrabc=0 disables use_intrabc");
}

void test_partial_blocks_and_residual() {
  const int w = 240;
  const int h = 128;
  auto screen = av2enc::ibc::make_synthetic_screen(w, h);
  const FrameView view = view_of(screen);
  IbcConfig config;
  config.allow_global = true;
  config.allow_local = false;
  IbcBufferState state;
  state.reset(w / 4, h / 4);
  std::string error;

  // A 64x32 block whose right 16 columns are outside the visible frame.
  // Source x=32..95, y=64..95 is fully interior.
  const int mi_row = 16;
  const int mi_col = 48;  // x=192
  const BlockVector bv{-32 * 8, (128 - 192) * 8};
  check(is_bv_encodable(view, config, state, mi_row, mi_col, kBs64x32, bv,
                        &error),
        "partial block source is encodable: " + error);
  std::vector<std::uint8_t> dst(64 * 32, 0);
  check(av2enc::ibc::copy_block(view, mi_row, mi_col, kBs64x32, bv,
                                dst.data(), 64, nullptr, 0, nullptr, 0, 48,
                                32),
        "partial-width copy_block");
  int diff = 0;
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 48; ++c) {
      const std::uint8_t src =
          view.y.data[static_cast<std::size_t>(32 + r) *
                          static_cast<std::size_t>(view.y.stride) +
                      static_cast<std::size_t>(128 + c)];
      if (src != dst[static_cast<std::size_t>(r) * 64 +
                      static_cast<std::size_t>(c)]) {
        ++diff;
      }
    }
  }
  check(diff == 0, "partial-width copy is byte-exact");

  // A 64x32 block whose bottom 16 rows are outside the visible frame.
  const int h2 = 96;
  auto screen2 = av2enc::ibc::make_synthetic_screen(w, h2);
  const FrameView view2 = view_of(screen2);
  IbcBufferState state2;
  state2.reset(w / 4, h2 / 4);
  const BlockVector bv2{-32 * 8, (64 - 128) * 8};
  check(is_bv_encodable(view2, config, state2, 16, 32, kBs64x32, bv2,
                        &error),
        "partial-height source is encodable: " + error);
  std::vector<std::uint8_t> dst2(64 * 32, 0);
  check(av2enc::ibc::copy_block(view2, 16, 32, kBs64x32, bv2, dst2.data(), 64,
                                nullptr, 0, nullptr, 0, 64, 16),
        "partial-height copy_block");
  diff = 0;
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 64; ++c) {
      const std::uint8_t src =
          view2.y.data[static_cast<std::size_t>(32 + r) *
                           static_cast<std::size_t>(view2.y.stride) +
                       static_cast<std::size_t>(64 + c)];
      if (src != dst2[static_cast<std::size_t>(r) * 64 +
                       static_cast<std::size_t>(c)]) {
        ++diff;
      }
    }
  }
  check(diff == 0, "partial-height copy is byte-exact");

  // Transform-skip residual: an exact copy has zero residual; a different
  // source region has a nonzero residual.
  const int fw = 128;
  const int fh = 128;
  std::vector<std::uint8_t> y(static_cast<std::size_t>(fw) * fh, 77);
  av2enc::ibc::FrameView flat;
  flat.y = av2enc::ibc::PlaneView{y.data(), fw, fw, fh};
  flat.has_chroma = false;
  std::vector<std::int16_t> residual(64 * 32, 0);
  av2enc::ibc::copy_then_residual(flat, 16, 16, kBs64x32,
                                  BlockVector{0, -64 * 8}, residual.data(), 64,
                                  nullptr, 0, nullptr, 0, 64, 32);
  int nonzero = 0;
  for (std::int16_t v : residual) nonzero += v != 0 ? 1 : 0;
  check(nonzero == 0, "transform-skip residual is zero for an exact copy");
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 64; ++c) {
      y[static_cast<std::size_t>(64 + r) * static_cast<std::size_t>(fw) +
        static_cast<std::size_t>(64 + c)] = 9;
    }
  }
  av2enc::ibc::copy_then_residual(flat, 16, 16, kBs64x32,
                                  BlockVector{0, -64 * 8}, residual.data(), 64,
                                  nullptr, 0, nullptr, 0, 64, 32);
  nonzero = 0;
  for (std::int16_t v : residual) nonzero += v != 0 ? 1 : 0;
  check(nonzero == 64 * 32, "non-skip residual covers the visible block");
}


StreamPlan make_global_ibc_plan() {
  StreamPlan plan;
  plan.config.width = 768;
  plan.config.height = 128;
  plan.config.allow_screen_content_tools = true;
  plan.config.force_integer_mv = true;
  plan.config.allow_intrabc = true;
  plan.config.allow_global_intrabc = true;
  plan.config.allow_local_intrabc = false;
  const int sb_cols = 12;
  const int sb_rows = 2;
  for (int sb = 0; sb < sb_cols * sb_rows; ++sb) {
    const int sb_row = sb / sb_cols;
    const int sb_col = sb % sb_cols;
    StreamSb s;
    s.partition = StreamSb::Partition::kHorz;
    for (int half = 0; half < 2; ++half) {
      StreamBlock leaf;
      leaf.mi_row = sb_row * 16 + half * 8;
      leaf.mi_col = sb_col * 16;
      leaf.bsize = kBs64x32;
      if (sb_row == 1 && half == 0 && sb_col >= 8 && sb_col <= 11) {
        leaf.ibc = true;
        leaf.skip = true;
        leaf.syntax.mode =
            sb_col <= 9 ? IntrabcMode::kNewMv : IntrabcMode::kNearMv;
        if (sb_col == 8) {
          leaf.syntax.ref_mv_idx = 1;
          leaf.bv = BlockVector{0, -384 * 8};
        } else if (sb_col == 9) {
          leaf.syntax.ref_mv_idx = 0;
          leaf.bv = BlockVector{0, -320 * 8};
        } else {
          leaf.syntax.ref_mv_idx = 0;
          leaf.bv = BlockVector{0, -320 * 8};
        }
      }
      s.leaves.push_back(leaf);
    }
    plan.sbs.push_back(s);
  }
  return plan;
}

StreamPlan make_near_only_plan() {
  StreamPlan plan = make_global_ibc_plan();
  for (StreamSb& sb : plan.sbs) {
    for (StreamBlock& leaf : sb.leaves) {
      leaf.ibc = false;
      leaf.syntax = BvSyntax{};
      leaf.bv = BlockVector{0, 0};
    }
  }
  const int sb_cols = 12;
  const int sb_rows = 2;
  for (int sb = 0; sb < sb_cols * sb_rows; ++sb) {
    const int sb_row = sb / sb_cols;
    const int sb_col = sb % sb_cols;
    if (sb_row == 1 && (sb_col == 6 || sb_col == 11)) {
      StreamBlock& leaf = plan.sbs[static_cast<std::size_t>(sb)]
                              .leaves[0];
      leaf.ibc = true;
      leaf.skip = true;
      leaf.syntax.mode = IntrabcMode::kNearMv;
      leaf.syntax.ref_mv_idx = 1;
      leaf.bv = BlockVector{0, -320 * 8};
    }
  }
  return plan;
}


StreamPlan make_residual_plan() {
  StreamPlan plan = make_global_ibc_plan();
  for (StreamSb& sb : plan.sbs) {
    for (StreamBlock& leaf : sb.leaves) {
      if (leaf.ibc) leaf.skip = false;
    }
  }
  return plan;
}

void test_stream_encode(const std::string& out_prefix) {
  StreamPlan plan = make_global_ibc_plan();
  StreamResult result;
  std::string error;
  const bool ok = encode_ibc_stream(plan, result, error);
  check(ok, "global IBC stream encodes: " + error);
  if (!ok) return;
  check(result.ibc_blocks == 4, "stream contains four IBC blocks");
  check(result.newmv_blocks == 2, "stream contains two NEWMV blocks");
  check(result.tile_bytes > 0, "tile payload is non-empty");
  int non128 = 0;
  for (std::uint8_t v : result.recon_y) non128 += v != 128 ? 1 : 0;
  check(non128 == 0, "DC/IBC reconstruction is flat 128");

  if (!out_prefix.empty()) {
    check(write_ibc_ivf(result, plan.config, out_prefix + ".ivf", error),
          "write IVF: " + error);
    check(write_ibc_y4m(result, plan.config, out_prefix + ".y4m", error),
          "write Y4M: " + error);
  }

  StreamPlan residual_plan = make_residual_plan();
  StreamResult residual_result;
  const bool residual_ok = encode_ibc_stream(residual_plan, residual_result,
                                             error);
  check(residual_ok, "skip_flag=0 IBC residual stream encodes: " + error);
  if (residual_ok && !out_prefix.empty()) {
    check(write_ibc_ivf(residual_result, residual_plan.config,
                        out_prefix + ".resid.ivf", error),
          "write residual IVF: " + error);
    check(write_ibc_y4m(residual_result, residual_plan.config,
                        out_prefix + ".resid.y4m", error),
          "write residual Y4M: " + error);
  }

  StreamPlan near_plan = make_near_only_plan();
  StreamResult near_result;
  const bool near_ok = encode_ibc_stream(near_plan, near_result, error);
  check(near_ok, "NEARMV-only stream encodes: " + error);
  if (near_ok && !out_prefix.empty()) {
    check(write_ibc_ivf(near_result, near_plan.config,
                        out_prefix + ".near.ivf", error),
          "write NEARMV IVF: " + error);
    check(write_ibc_y4m(near_result, near_plan.config,
                        out_prefix + ".near.y4m", error),
          "write NEARMV Y4M: " + error);
  }

  // Invalid BV is rejected before/while writing.
  StreamPlan bad = make_global_ibc_plan();
  bool found = false;
  for (StreamSb& sb : bad.sbs) {
    for (StreamBlock& leaf : sb.leaves) {
      if (leaf.ibc && !found) {
        leaf.bv = BlockVector{0, 0};
        found = true;
      }
    }
  }
  StreamResult bad_result;
  check(!encode_ibc_stream(bad, bad_result, error),
        "invalid IBC BV fails the stream encode");
  check(error.find("invalid") != std::string::npos ||
            error.find("delay") != std::string::npos ||
            error.find("IBC") != std::string::npos,
        "invalid BV error is diagnostic: " + error);

  // A second geometry (NONE 64x64 DC + implied HORZ bottom row).
  StreamPlan small;
  small.config.width = 320;
  small.config.height = 96;
  small.config.allow_intrabc = false;
  small.config.allow_global_intrabc = false;
  small.config.allow_local_intrabc = false;
  for (int sb_row = 0; sb_row < 2; ++sb_row) {
    for (int sb_col = 0; sb_col < 5; ++sb_col) {
      StreamSb s;
      StreamBlock leaf;
      leaf.mi_row = sb_row * 16;
      leaf.mi_col = sb_col * 16;
      if (sb_row == 0) {
        s.partition = StreamSb::Partition::kNone;
        leaf.bsize = kBs64x64;
        s.leaves.push_back(leaf);
      } else if (sb_col < 4) {
        s.partition = StreamSb::Partition::kHorz;
        leaf.bsize = kBs64x32;
        s.leaves.push_back(leaf);
      } else {
        s.partition = StreamSb::Partition::kNone;
        leaf.bsize = kBs64x64;
        s.leaves.push_back(leaf);
      }
      small.sbs.push_back(s);
    }
  }
  StreamResult small_result;
  // The last row's final 64x64 root is implied HORZ; build the plan with that
  // geometry so the test also exercises the boundary cut.
  small.sbs.back().partition = StreamSb::Partition::kHorz;
  small.sbs.back().leaves.clear();
  StreamBlock tail;
  tail.mi_row = 16;
  tail.mi_col = 64;
  tail.bsize = kBs64x32;
  small.sbs.back().leaves.push_back(tail);
  const bool small_ok = encode_ibc_stream(small, small_result, error);
  check(small_ok, "320x96 boundary-partition stream encodes: " + error);
  if (small_ok && !out_prefix.empty()) {
    check(write_ibc_ivf(small_result, small.config,
                        out_prefix + ".small.ivf", error),
          "write small IVF: " + error);
    check(write_ibc_y4m(small_result, small.config,
                        out_prefix + ".small.y4m", error),
          "write small Y4M: " + error);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_prefix;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out-prefix" && i + 1 < argc) {
      out_prefix = argv[++i];
    }
  }
  test_cdf_init();
  test_mv_roundtrip();
  test_intrabc_info_roundtrip();
  test_use_intrabc_and_skip();
  test_mode_info_neighbours();
  test_stack_and_border();
  test_partial_blocks_and_residual();
  test_stream_encode(out_prefix);
  std::printf("ibc_stream_test: %d checks, %d failures\n", g_checks,
              g_failures);
  return g_failures == 0 ? 0 : 1;
}
