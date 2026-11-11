// Standalone multi-frame / multi-tile CDF-continuity stream.
//
// This is the proof the production writers cannot currently provide: two
// independent `FrameCdfContinuity` instances, each with its own CDF bank,
// drive an `av2::Encoder` and an `av2::Decoder` through
//
//   frame 0: CLK-style reset, 2x2 tiles, copyCdf (context_update_tile_id=2)
//   frame 1: load_cdfs(slot 0), 2x2 tiles, copyCdf (context_update_tile_id=1)
//   frame 2: load_cdfs(slot 0) + blend_cdfs(slot 1), 2x2 tiles,
//            enable_avg_cdf=1 / avg_cdf_type=0, copyCdf (id=3)
//
// Every tile payload in frames 1 and 2 is entropy-coded against the CDFs that
// frame 0/1 installed at `frame_end_update_cdf()`.  The decoder starts each
// frame from the same reference context and must reproduce every symbol and
// every CDF array.  The test also encodes the same symbol sequence against a
// reset context and against no blend, and asserts the bytes differ: that is
// what "carries CDFs across a frame boundary" means at the bitstream level.
//
// Build:
//   ./src/encoder/entropy/continuity/build_and_test.sh
//
// No dav2d here: the payloads are raw §8.2 tile payloads, not OBU-wrapped M1/M2
// syntax.  `gate_dav2d_continuity.cpp` covers the OBU-wrapped stream.

#include "encoder/entropy/continuity/cdf_continuity.h"
#include "entropy/av2/av2_entropy.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace cont = av2enc::entropy::continuity;
namespace acdf = av2enc::entropy::avg_cdf;
using av2enc::av2::Decoder;
using av2enc::av2::Encoder;
using av2enc::av2::Status;
using cont::ContinuityStatus;
using cont::FrameCdfContinuity;
using cont::FrameCdfInput;

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    ++g_checks;                                                            \
    if (!(cond)) {                                                         \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                      \
  } while (false)

#define CHECK_STATUS(expr, expected)                                       \
  do {                                                                     \
    ++g_checks;                                                            \
    const ContinuityStatus check_status = (expr);                          \
    if (check_status != (expected)) {                                      \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %s (expected %s)\n",         \
                   __FILE__, __LINE__, #expr,                              \
                   cont::continuity_status_name(check_status),             \
                   cont::continuity_status_name(expected));                \
    }                                                                      \
  } while (false)

#define CHECK_AV2(expr)                                                    \
  do {                                                                     \
    ++g_checks;                                                            \
    const Status check_status = (expr);                                    \
    if (check_status != Status::kOk) {                                     \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__,   \
                   #expr, av2enc::av2::status_name(check_status));         \
    }                                                                      \
  } while (false)

#define CHECK_EQ_U64(a, b)                                                 \
  do {                                                                     \
    ++g_checks;                                                            \
    const std::uint64_t check_a = (a);                                     \
    const std::uint64_t check_b = (b);                                     \
    if (check_a != check_b) {                                              \
      ++g_failures;                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s (%016llx vs %016llx)\n",  \
                   __FILE__, __LINE__, #a, #b,                             \
                   static_cast<unsigned long long>(check_a),               \
                   static_cast<unsigned long long>(check_b));              \
    }                                                                      \
  } while (false)

constexpr std::uint16_t kDefaultA[4] = {10000, 20000, 56, 0};
constexpr std::uint16_t kDefaultB[5] = {8000, 16000, 24000, 40, 0};
constexpr std::uint32_t kFrameCount = 3;
constexpr std::uint32_t kTileCols = 2;
constexpr std::uint32_t kTileRows = 2;
constexpr std::uint32_t kTileCount = kTileCols * kTileRows;
constexpr int kSymbolsPerTile = 64;

void init_controller(FrameCdfContinuity* ctl) {
  const acdf::CdfSpec specs[] = {
      acdf::CdfSpec{3, kDefaultA},
      acdf::CdfSpec{4, kDefaultB},
  };
  std::string error;
  CHECK(ctl->init_owned(specs, cont::kNumRefFrames, &error));
  CHECK(error.empty());
}

struct FrameConfig {
  std::uint32_t context_update_tile_id = 0;
  bool enable_avg_cdf = false;
  bool avg_cdf_type = false;
  bool blend = false;
  int raw_base_q_idx = 90;
};

FrameConfig frame_config(std::uint32_t frame, bool blend) {
  FrameConfig config;
  if (frame == 0u) {
    config.context_update_tile_id = 2u;
    config.raw_base_q_idx = 90;
  } else if (frame == 1u) {
    config.context_update_tile_id = 1u;
    config.raw_base_q_idx = 120;  // qctx slice 1
  } else {
    config.context_update_tile_id = 3u;
    config.enable_avg_cdf = true;
    config.avg_cdf_type = false;  // copyCdf + blend path
    config.blend = blend;
    config.raw_base_q_idx = 191;  // qctx slice 3
  }
  return config;
}

FrameCdfInput make_input(std::uint32_t frame, bool carry, bool blend) {
  const FrameConfig config = frame_config(frame, blend);
  FrameCdfInput in;
  in.tile_cols = kTileCols;
  in.tile_rows = kTileRows;
  in.context_update_tile_id = config.context_update_tile_id;
  in.enable_avg_cdf = config.enable_avg_cdf;
  in.avg_cdf_type = config.avg_cdf_type;
  in.raw_base_q_idx = config.raw_base_q_idx;
  in.disable_cdf_update = false;
  if (frame == 0u) {
    in.keyframe = true;  // CLK/OLK: defaults
  } else {
    in.primary_ref_slot = 0u;
    in.disable_cross_frame_cdf_init = !carry;
    if (config.blend) {
      in.blend_ref_present = true;
      in.blend_ref_slot = 1u;
    }
  }
  return in;
}

int symbol_a(std::uint32_t frame, std::uint32_t tile, int index) {
  return static_cast<int>((frame * 7u + tile * 3u +
                           static_cast<std::uint32_t>(index)) %
                          3u);
}

int symbol_b(std::uint32_t frame, std::uint32_t tile, int index) {
  return static_cast<int>((frame * 5u + tile * 7u +
                           static_cast<std::uint32_t>(index) * 3u) %
                          4u);
}

using FramePayloads = std::vector<std::vector<std::uint8_t>>;
using Stream = std::array<FramePayloads, kFrameCount>;

std::uint64_t fnv1a(const std::uint8_t* data, std::size_t size,
                    std::uint64_t hash = 1469598103934665603ull) {
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

std::uint64_t cdf_digest(const FrameCdfContinuity& ctl) {
  std::uint64_t hash = 1469598103934665603ull;
  for (std::size_t t = 0; t < ctl.bank().table_count(); ++t) {
    const acdf::CdfTable& table = ctl.bank().table(t);
    const std::size_t entries =
        (static_cast<std::size_t>(table.symbols) + 1u) *
        sizeof(std::uint16_t);
    hash = fnv1a(reinterpret_cast<const std::uint8_t*>(table.frame), entries,
                 hash);
  }
  return hash;
}

bool arrays_equal(const std::uint16_t* a, const std::uint16_t* b,
                  std::size_t entries) {
  return std::memcmp(a, b, entries * sizeof(std::uint16_t)) == 0;
}

bool compare_banks(const FrameCdfContinuity& enc,
                   const FrameCdfContinuity& dec, const char* where) {
  bool ok = true;
  for (std::size_t t = 0; t < enc.bank().table_count(); ++t) {
    const acdf::CdfTable& a = enc.bank().table(t);
    const acdf::CdfTable& b = dec.bank().table(t);
    const std::size_t entries = static_cast<std::size_t>(a.symbols) + 1u;
    if (!arrays_equal(a.tile, b.tile, entries) ||
        !arrays_equal(a.saved, b.saved, entries) ||
        !arrays_equal(a.frame, b.frame, entries)) {
      std::fprintf(stderr, "CDF lockstep divergence (%s), table %zu\n", where,
                   t);
      ok = false;
    }
  }
  ++g_checks;
  return ok;
}

bool encode_tile(FrameCdfContinuity* ctl, std::uint32_t frame,
                 std::uint32_t tile, FramePayloads* payloads) {
  CHECK_STATUS(ctl->begin_tile(), ContinuityStatus::kOk);
  std::array<std::uint8_t, 1u << 16> buffer{};
  Encoder encoder;
  CHECK_AV2(encoder.init_symbol(buffer.data(), buffer.size()));
  encoder.disable_cdf_update = 0;
  for (int i = 0; i < kSymbolsPerTile; ++i) {
    CHECK_AV2(encoder.encode_symbol(ctl->bank().tile_cdf(0), 3,
                                    symbol_a(frame, tile, i)));
    CHECK_AV2(encoder.encode_symbol(ctl->bank().tile_cdf(1), 4,
                                    symbol_b(frame, tile, i)));
  }
  std::size_t bytes = 0;
  CHECK_AV2(encoder.exit_symbol(&bytes));
  payloads->emplace_back(buffer.begin(),
                         buffer.begin() + static_cast<std::ptrdiff_t>(bytes));
  CHECK_STATUS(ctl->end_tile(tile), ContinuityStatus::kOk);
  return true;
}

bool decode_tile(FrameCdfContinuity* ctl, std::uint32_t frame,
                 std::uint32_t tile, const std::vector<std::uint8_t>& payload) {
  CHECK_STATUS(ctl->begin_tile(), ContinuityStatus::kOk);
  Decoder decoder;
  CHECK_AV2(decoder.init_symbol(payload.data(), payload.size(),
                                payload.size()));
  decoder.disable_cdf_update = 0;
  for (int i = 0; i < kSymbolsPerTile; ++i) {
    int got_a = -1;
    int got_b = -1;
    CHECK_AV2(decoder.decode_symbol(ctl->bank().tile_cdf(0), 3, &got_a));
    CHECK_AV2(decoder.decode_symbol(ctl->bank().tile_cdf(1), 4, &got_b));
    CHECK(got_a == symbol_a(frame, tile, i));
    CHECK(got_b == symbol_b(frame, tile, i));
  }
  CHECK_AV2(decoder.exit_symbol());
  CHECK_STATUS(ctl->end_tile(tile), ContinuityStatus::kOk);
  return true;
}

// Runs the whole sequence.  `carry=false` replaces frames 1/2 with the
// §5.18.2 reset branch; `blend=false` removes the §7.4.4 blend from frame 2.
// `lockstep` also runs the independent decoder and compares every CDF array.
bool run_sequence(bool carry, bool blend, bool lockstep, Stream* stream,
                  std::uint64_t* final_digest) {
  FrameCdfContinuity encoder_ctl;
  FrameCdfContinuity decoder_ctl;
  init_controller(&encoder_ctl);
  if (lockstep) init_controller(&decoder_ctl);

  for (std::uint32_t frame = 0; frame < kFrameCount; ++frame) {
    const FrameCdfInput input = make_input(frame, carry, blend);
    std::string error;
    CHECK_STATUS(encoder_ctl.begin_frame(input, &error),
                 ContinuityStatus::kOk);
    if (lockstep) {
      CHECK_STATUS(decoder_ctl.begin_frame(input, &error),
                   ContinuityStatus::kOk);
      CHECK(compare_banks(encoder_ctl, decoder_ctl, "frame begin"));
    }
    if (carry && frame > 0u) {
      // The seed must not be the §9.3 defaults: this frame's symbols are
      // entropy-coded against state carried across the frame boundary.
      for (std::size_t t = 0; t < encoder_ctl.bank().table_count(); ++t) {
        const acdf::CdfTable& table = encoder_ctl.bank().table(t);
        const std::size_t entries =
            static_cast<std::size_t>(table.symbols) + 1u;
        CHECK(std::memcmp(table.tile, table.defaults,
                          entries * sizeof(std::uint16_t)) != 0);
      }
    }

    FramePayloads& payloads = (*stream)[frame];
    payloads.clear();
    for (std::uint32_t tile = 0; tile < kTileCount; ++tile) {
      encode_tile(&encoder_ctl, frame, tile, &payloads);
      if (lockstep && payloads.size() == static_cast<std::size_t>(tile) + 1u) {
        decode_tile(&decoder_ctl, frame, tile, payloads[tile]);
        CHECK(compare_banks(encoder_ctl, decoder_ctl, "tile end"));
      }
    }

    CHECK_STATUS(encoder_ctl.end_frame(&error), ContinuityStatus::kOk);
    if (lockstep) {
      CHECK_STATUS(decoder_ctl.end_frame(&error), ContinuityStatus::kOk);
      CHECK(compare_banks(encoder_ctl, decoder_ctl, "frame end"));
      CHECK_EQ_U64(cdf_digest(encoder_ctl), cdf_digest(decoder_ctl));
    }

    // §7.23 save_cdfs(i): frame 0 refreshes slots 0 and 1; frame 1 refreshes
    // slot 0; frame 2 refreshes slot 1.
    if (frame == 0u) {
      CHECK_STATUS(encoder_ctl.save_reference(0, &error),
                   ContinuityStatus::kOk);
      CHECK_STATUS(encoder_ctl.save_reference(1, &error),
                   ContinuityStatus::kOk);
      if (lockstep) {
        CHECK_STATUS(decoder_ctl.save_reference(0, &error),
                     ContinuityStatus::kOk);
        CHECK_STATUS(decoder_ctl.save_reference(1, &error),
                     ContinuityStatus::kOk);
      }
    } else if (frame == 1u) {
      CHECK_STATUS(encoder_ctl.save_reference(0, &error),
                   ContinuityStatus::kOk);
      if (lockstep) {
        CHECK_STATUS(decoder_ctl.save_reference(0, &error),
                     ContinuityStatus::kOk);
      }
    } else {
      CHECK_STATUS(encoder_ctl.save_reference(1, &error),
                   ContinuityStatus::kOk);
      if (lockstep) {
        CHECK_STATUS(decoder_ctl.save_reference(1, &error),
                     ContinuityStatus::kOk);
      }
    }
    if (lockstep) CHECK(compare_banks(encoder_ctl, decoder_ctl, "save_cdfs"));
    std::printf("  frame %u: tiles %zu/%zu/%zu/%zu B, qctx slice %d\n", frame,
                payloads[0].size(), payloads[1].size(), payloads[2].size(),
                payloads[3].size(), encoder_ctl.plan().qctx_slice);
  }
  if (final_digest != nullptr) *final_digest = cdf_digest(encoder_ctl);
  return true;
}

bool payloads_differ(const FramePayloads& a, const FramePayloads& b) {
  if (a.size() != b.size()) return true;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return true;
  }
  return false;
}

std::size_t payload_bytes(const FramePayloads& p) {
  std::size_t total = 0;
  for (const auto& tile : p) total += tile.size();
  return total;
}

bool write_stream(const std::string& path, const Stream& stream) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  const char magic[4] = {'C', 'D', 'F', 'C'};
  out.write(magic, 4);
  const std::uint32_t frames = kFrameCount;
  out.write(reinterpret_cast<const char*>(&frames), sizeof(frames));
  for (const FramePayloads& frame : stream) {
    const std::uint32_t tiles = static_cast<std::uint32_t>(frame.size());
    out.write(reinterpret_cast<const char*>(&tiles), sizeof(tiles));
    for (const auto& tile : frame) {
      const std::uint32_t size = static_cast<std::uint32_t>(tile.size());
      out.write(reinterpret_cast<const char*>(&size), sizeof(size));
      out.write(reinterpret_cast<const char*>(tile.data()),
                static_cast<std::streamsize>(tile.size()));
    }
  }
  return out.good();
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--help") {
      std::printf("usage: lockstep_stream_test [--out stream.bin]\n");
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  std::printf("carried sequence (2x2 tiles, 3 frames):\n");
  Stream carried{};
  std::uint64_t carried_digest = 0;
  run_sequence(/*carry=*/true, /*blend=*/true, /*lockstep=*/true, &carried,
               &carried_digest);

  std::printf("reset control (same symbols, no cross-frame carry):\n");
  Stream reset{};
  std::uint64_t reset_digest = 0;
  run_sequence(/*carry=*/false, /*blend=*/true, /*lockstep=*/false, &reset,
               &reset_digest);

  std::printf("no-blend control for frame 2:\n");
  Stream no_blend{};
  std::uint64_t no_blend_digest = 0;
  run_sequence(/*carry=*/true, /*blend=*/false, /*lockstep=*/false, &no_blend,
               &no_blend_digest);

  CHECK(payloads_differ(carried[1], reset[1]));
  CHECK(payloads_differ(carried[2], reset[2]));
  CHECK(payloads_differ(carried[2], no_blend[2]));

  std::printf("  carried frame1 bytes=%zu reset frame1 bytes=%zu\n",
              payload_bytes(carried[1]), payload_bytes(reset[1]));
  std::printf("  carried frame2 bytes=%zu reset frame2 bytes=%zu\n",
              payload_bytes(carried[2]), payload_bytes(reset[2]));
  std::printf("  carried frame2 bytes=%zu no-blend frame2 bytes=%zu\n",
              payload_bytes(carried[2]), payload_bytes(no_blend[2]));
  std::printf("  carried final CDF digest=%016llx reset=%016llx\n",
              static_cast<unsigned long long>(carried_digest),
              static_cast<unsigned long long>(reset_digest));
  std::printf("  no-blend final CDF digest=%016llx\n",
              static_cast<unsigned long long>(no_blend_digest));
  CHECK(carried_digest != reset_digest);
  CHECK(carried_digest != no_blend_digest);

  if (!out_path.empty()) {
    CHECK(write_stream(out_path, carried));
    std::printf("  wrote %s\n", out_path.c_str());
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "FAIL: %d/%d checks failed\n", g_failures, g_checks);
    return 1;
  }
  std::printf(
      "PASS: %d checks; frame boundary carried an adapted CDF state and the "
      "decoder reproduced every symbol and array\n",
      g_checks);
  return 0;
}
