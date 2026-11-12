#include "codec/intra/palette/stream/palette_stream.h"

#include "codec/intra/palette/core/scc_stream.h"
#include "codec/partition/tree/partition.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace av2enc::still::palette_stream {
namespace {

// Count cache entries reused by `colors`, using the same greedy scan as
// palette::split_cache_and_run() (the analysis result may come from a caller
// that did not pre-compute PaletteResult::cache_used).
int count_cache_reuse(const std::uint8_t* colors, int size,
                      const std::uint8_t* cache, int cache_n) noexcept {
  bool used[16] = {};
  int idx = 0;
  for (int i = 0; i < cache_n && idx < size; ++i) {
    for (int c = idx; c < size; ++c) {
      if (colors[c] == cache[i]) {
        used[i] = true;
        ++idx;
        break;
      }
    }
  }
  int count = 0;
  for (int i = 0; i < cache_n; ++i) count += used[i] ? 1 : 0;
  return count;
}

}  // namespace

// ---------------------------------------------------------------------------
// Screen-content flags and header bits
// ---------------------------------------------------------------------------

bool validate_screen_content_flags(const ScreenContentFlags& flags,
                                   std::string& error) {
  error.clear();
  if (flags.allow_palette && !flags.allow_screen_content_tools) {
    error = "allow_palette requires allow_screen_content_tools";
    return false;
  }
  if (flags.force_integer_mv && !flags.allow_screen_content_tools) {
    error = "force_integer_mv requires allow_screen_content_tools";
    return false;
  }
  if (flags.allow_intrabc && !(flags.allow_global_intrabc ||
                               flags.allow_local_intrabc)) {
    error = "allow_intrabc requires a local or global reference region";
    return false;
  }
  // allow_global_intrabc / allow_local_intrabc are only coded when
  // allow_intrabc is set; leaving them true with IBC off is harmless.
  return true;
}

ScreenContentFlags palette_screen_content_flags() {
  ScreenContentFlags flags;
  flags.single_picture_header = true;
  flags.allow_screen_content_tools = true;
  flags.allow_palette = true;
  flags.force_integer_mv = true;
  flags.allow_intrabc = false;
  flags.allow_global_intrabc = false;
  flags.allow_local_intrabc = false;
  return flags;
}

std::size_t write_sequence_screen_content_params(
    BitWriter& writer, const ScreenContentFlags& flags) {
  if (flags.single_picture_header) {
    // §5.4.7: both values are inferred; no bits are present.
    return 0;
  }
  const std::size_t start = writer.bit_size();
  const bool choose_screen = flags.allow_screen_content_tools;
  writer.put_bit(choose_screen);  // seq_choose_screen_content_tools
  if (!choose_screen) {
    writer.put_bit(false);  // seq_force_screen_content_tools = 0
  } else {
    // PRESENT: the caller wants the frame header to decide both switches.
    const bool choose_integer = flags.force_integer_mv;
    writer.put_bit(choose_integer);  // seq_choose_integer_mv
    if (!choose_integer) writer.put_bit(false);  // seq_force_integer_mv = 0
  }
  return writer.bit_size() - start;
}

std::size_t write_frame_screen_content_params(
    BitWriter& writer, const ScreenContentFlags& flags) {
  const std::size_t start = writer.bit_size();
  // Under single_picture_header_flag, seq_force_screen_content_tools is
  // SELECT_SCREEN_CONTENT_TOOLS, so this bit is always present.
  writer.put_bit(flags.allow_screen_content_tools);
  if (flags.allow_screen_content_tools) {
    // seq_force_integer_mv is SELECT_INTEGER_MV.
    writer.put_bit(flags.force_integer_mv);
  }
  return writer.bit_size() - start;
}

bool palette_allowed_for_block(const ScreenContentFlags& flags, int bsize,
                               int block_w, int block_h) noexcept {
  if (!flags.allow_screen_content_tools || !flags.allow_palette) return false;
  // §5.20.5.5: MiSize >= BLOCK_8X8 (numeric enum order) and w,h <= 64.
  if (bsize < kBlock8x8 || bsize >= 128) return false;
  if (block_w <= 0 || block_h <= 0) return false;
  if (block_w > kMaxPaletteBlockDim || block_h > kMaxPaletteBlockDim) {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// §8.3.2 CDF accessors
// ---------------------------------------------------------------------------

const std::uint16_t* y_mode_cdf(const pal::PaletteCdfState& cdfs) noexcept {
  return cdfs.y_mode;
}

const std::uint16_t* y_size_cdf(const pal::PaletteCdfState& cdfs) noexcept {
  return cdfs.y_size;
}

const std::uint16_t* color_cdf(const pal::PaletteCdfState& cdfs, int size,
                               int context_index) noexcept {
  if (size < 2 || size > pal::kPaletteMaxSize) return nullptr;
  if (context_index < 0 || context_index >= pal::kPaletteColorContexts) {
    return nullptr;
  }
  switch (size) {
    case 2:
      return cdfs.color2[context_index];
    case 3:
      return cdfs.color3[context_index];
    case 4:
      return cdfs.color4[context_index];
    case 5:
      return cdfs.color5[context_index];
    case 6:
      return cdfs.color6[context_index];
    case 7:
      return cdfs.color7[context_index];
    case 8:
      return cdfs.color8[context_index];
    default:
      return nullptr;
  }
}

const std::uint16_t* identity_cdf(const pal::PaletteCdfState& cdfs,
                                  int prev_identity) noexcept {
  if (prev_identity < 0 || prev_identity >= pal::kPaletteRowFlagContexts) {
    return nullptr;
  }
  return cdfs.identity[prev_identity];
}

int palette_size_symbol(int size) noexcept { return size - 2; }

bool color_order_at(const pal::PaletteResult& result, int block_w, int r,
                    int c, ColorOrderSample& out) noexcept {
  out = ColorOrderSample{};
  if (!result.has_palette || block_w <= 0) return false;
  if (result.size < 2 || result.size > pal::kPaletteMaxSize) return false;
  if (r < 0 || c < 0 || c >= block_w) return false;
  if (result.color_map.size() <
      static_cast<std::size_t>(r + 1) * static_cast<std::size_t>(block_w)) {
    return false;
  }
  out.r = r;
  out.c = c;
  out.hash = pal::palette_color_context(result.color_map.data(), block_w, r, c,
                                        result.size, out.order.data());
  out.context_index = pal::palette_color_context_index(out.hash);
  if (out.context_index < 0) return false;
  const int colour = static_cast<int>(
      result.color_map[static_cast<std::size_t>(r) *
                           static_cast<std::size_t>(block_w) +
                       static_cast<std::size_t>(c)]);
  out.symbol =
      pal::palette_symbol_for_color(colour, out.order.data(), result.size);
  return out.symbol >= 0;
}

// ---------------------------------------------------------------------------
// PaletteSyntaxWriter
// ---------------------------------------------------------------------------

PaletteSyntaxWriter::PaletteSyntaxWriter(const ScreenContentFlags& flags)
    : flags_(flags) {}

bool PaletteSyntaxWriter::reset(int mi_cols, int mi_rows, std::string& error) {
  error.clear();
  if (mi_cols <= 0 || mi_rows <= 0) {
    error = "palette frame state needs positive MI dimensions";
    return false;
  }
  if (!validate_screen_content_flags(flags_, error)) return false;
  state_.reset(mi_cols, mi_rows);
  return true;
}

int PaletteSyntaxWriter::cache(int mi_row, int mi_col, bool avail_l,
                               std::uint8_t out[16]) const noexcept {
  return pal::get_palette_cache(state_, mi_row, mi_col, avail_l, out);
}

bool PaletteSyntaxWriter::write(av2::Encoder& enc, pal::PaletteCdfState& cdfs,
                                int mi_row, int mi_col, int bsize, int block_w,
                                int block_h, int visible_w, int visible_h,
                                const pal::PaletteResult* result, int y_mode,
                                PaletteModeInfo* info, std::string& error) {
  error.clear();
  if (!state_.valid()) {
    error = "PaletteSyntaxWriter::write before reset()";
    return false;
  }
  if (mi_row < 0 || mi_col < 0 || mi_row >= state_.mi_rows ||
      mi_col >= state_.mi_cols) {
    error = "palette block origin is outside the MI grid";
    return false;
  }
  if (info != nullptr) *info = PaletteModeInfo{};

  const bool want_palette = result != nullptr && result->has_palette;
  // §5.20.8.1: the syntax is only present for PlaneStart == 0 (luma) and
  // YMode == DC_PRED, even when screen-content tools are allowed.
  const bool reachable =
      y_mode == 0 &&
      palette_allowed_for_block(flags_, bsize, block_w, block_h);
  if (want_palette && !reachable) {
    error = "palette is not signallable at this syntax position/size";
    return false;
  }
  if (!reachable) return true;  // no symbol at all

  if (visible_w <= 0 || visible_h <= 0 || visible_w > block_w ||
      visible_h > block_h) {
    error = "invalid onscreen palette dimensions";
    return false;
  }
  if (want_palette) {
    if (result->size < 2 || result->size > pal::kPaletteMaxSize) {
      error = "palette size out of range";
      return false;
    }
    if (result->color_map.size() <
        static_cast<std::size_t>(block_w) *
            static_cast<std::size_t>(block_h)) {
      error = "palette colour map is smaller than the coding block";
      return false;
    }
    // Direction is only coded when both dimensions are below 64.
    if ((block_w >= kMaxPaletteBlockDim || block_h >= kMaxPaletteBlockDim) &&
        result->direction != 0) {
      error = "palette_direction is not coded at 64 samples";
      return false;
    }
    if (result->onscreen_width != 0 &&
        result->onscreen_width != visible_w) {
      error = "palette result onscreen width disagrees with the block";
      return false;
    }
    if (result->onscreen_height != 0 &&
        result->onscreen_height != visible_h) {
      error = "palette result onscreen height disagrees with the block";
      return false;
    }
  }

  std::uint8_t cache[16] = {};
  const int cache_n = this->cache(mi_row, mi_col, mi_col > 0, cache);
  const pal::PaletteResult empty;
  const pal::PaletteResult& payload = result != nullptr ? *result : empty;

  if (!pal::encode_palette_y(enc, cdfs, payload, block_w, block_h, visible_w,
                             visible_h, cache, cache_n, error)) {
    return false;
  }

  if (payload.has_palette) {
    // Store for every MI the block covers, exactly like the decoder's
    // §5.20.8.1 PaletteSizes/PaletteColors update.
    pal::store_palette_state(state_, mi_row, mi_col, block_w, block_h, payload);
  }

  if (info != nullptr) {
    info->has_palette_y = payload.has_palette;
    info->cache_len = cache_n;
    info->onscreen_width = visible_w;
    info->onscreen_height = visible_h;
    if (payload.has_palette) {
      info->size = payload.size;
      info->direction = payload.direction;
      info->direction_signalled =
          block_w < kMaxPaletteBlockDim && block_h < kMaxPaletteBlockDim;
      const int outer = payload.direction ? visible_w : visible_h;
      const int inner = payload.direction ? visible_h : visible_w;
      info->identity_rows.resize(static_cast<std::size_t>(outer));
      for (int i = 0; i < outer; ++i) {
        info->identity_rows[static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>(pal::palette_classify_line(
                payload.color_map.data(), block_w, payload.direction, i, inner,
                i > 0, payload.size));
      }
      info->cache_used =
          count_cache_reuse(payload.colors.data(), payload.size, cache,
                            cache_n);
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// PaletteSyntaxReader
// ---------------------------------------------------------------------------

PaletteSyntaxReader::PaletteSyntaxReader(const ScreenContentFlags& flags)
    : flags_(flags) {}

bool PaletteSyntaxReader::reset(int mi_cols, int mi_rows, std::string& error) {
  error.clear();
  if (mi_cols <= 0 || mi_rows <= 0) {
    error = "palette frame state needs positive MI dimensions";
    return false;
  }
  if (!validate_screen_content_flags(flags_, error)) return false;
  state_.reset(mi_cols, mi_rows);
  return true;
}

int PaletteSyntaxReader::cache(int mi_row, int mi_col, bool avail_l,
                               std::uint8_t out[16]) const noexcept {
  return pal::get_palette_cache(state_, mi_row, mi_col, avail_l, out);
}

bool PaletteSyntaxReader::read(av2::Decoder& dec, pal::PaletteCdfState& cdfs,
                               int mi_row, int mi_col, int bsize, int block_w,
                               int block_h, int visible_w, int visible_h,
                               int y_mode, pal::PaletteResult& out,
                               PaletteModeInfo* info, std::string& error) {
  error.clear();
  out = pal::PaletteResult{};
  if (info != nullptr) *info = PaletteModeInfo{};
  if (!state_.valid()) {
    error = "PaletteSyntaxReader::read before reset()";
    return false;
  }
  if (mi_row < 0 || mi_col < 0 || mi_row >= state_.mi_rows ||
      mi_col >= state_.mi_cols) {
    error = "palette block origin is outside the MI grid";
    return false;
  }
  const bool reachable =
      y_mode == 0 &&
      palette_allowed_for_block(flags_, bsize, block_w, block_h);
  if (!reachable) return true;

  if (visible_w <= 0 || visible_h <= 0 || visible_w > block_w ||
      visible_h > block_h) {
    error = "invalid onscreen palette dimensions";
    return false;
  }
  std::uint8_t cache[16] = {};
  const int cache_n = this->cache(mi_row, mi_col, mi_col > 0, cache);
  if (!pal::decode_palette_y(dec, cdfs, block_w, block_h, visible_w, visible_h,
                             cache, cache_n, out, error)) {
    return false;
  }
  out.onscreen_width = visible_w;
  out.onscreen_height = visible_h;
  if (out.has_palette) {
    pal::store_palette_state(state_, mi_row, mi_col, block_w, block_h, out);
  }

  if (info != nullptr) {
    info->has_palette_y = out.has_palette;
    info->cache_len = cache_n;
    info->onscreen_width = visible_w;
    info->onscreen_height = visible_h;
    if (out.has_palette) {
      info->size = out.size;
      info->direction = out.direction;
      info->direction_signalled =
          block_w < kMaxPaletteBlockDim && block_h < kMaxPaletteBlockDim;
      const int outer = out.direction ? visible_w : visible_h;
      const int inner = out.direction ? visible_h : visible_w;
      info->identity_rows.resize(static_cast<std::size_t>(outer));
      for (int i = 0; i < outer; ++i) {
        info->identity_rows[static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>(pal::palette_classify_line(
                out.color_map.data(), block_w, out.direction, i, inner, i > 0,
                out.size));
      }
      info->cache_used =
          count_cache_reuse(out.colors.data(), out.size, cache, cache_n);
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Minimal M1-compatible palette stream
// ---------------------------------------------------------------------------

namespace {

scc::SccConfig make_scc_config(const MinimalPaletteFramePlan& plan) {
  scc::SccConfig config;
  config.width = plan.width;
  config.height = plan.height;
  config.base_q_idx = plan.base_q_idx;
  config.allow_screen_content_tools = plan.flags.allow_screen_content_tools;
  config.allow_intrabc = plan.flags.allow_intrabc;
  config.allow_global_intrabc = plan.flags.allow_global_intrabc;
  config.allow_local_intrabc = plan.flags.allow_local_intrabc;
  config.enable_cfl_intra = plan.enable_cfl_intra;
  config.enable_intra_edge_filter = plan.enable_intra_edge_filter;
  config.apply_deblock_y_vert = plan.apply_deblock_y_vert;
  config.apply_deblock_y_horz = plan.apply_deblock_y_horz;
  config.apply_deblock_u = plan.apply_deblock_u;
  config.apply_deblock_v = plan.apply_deblock_v;
  return config;
}

}  // namespace

bool encode_minimal_palette_frame(const MinimalPaletteFramePlan& plan,
                                  MinimalPaletteFrameResult& out,
                                  std::string& error) {
  error.clear();
  out = MinimalPaletteFrameResult{};
  if (!validate_screen_content_flags(plan.flags, error)) return false;
  if (!plan.flags.allow_palette) {
    error = "minimal palette stream requires allow_palette";
    return false;
  }
  if (plan.flags.allow_intrabc) {
    error = "the minimal palette stream is palette-only (no IBC)";
    return false;
  }
  if (plan.width <= 0 || plan.height <= 0 || (plan.width & 1) != 0 ||
      (plan.height & 1) != 0) {
    error = "minimal palette stream needs even positive dimensions";
    return false;
  }

  scc::SccFramePlan scc_plan;
  scc_plan.config = make_scc_config(plan);
  scc_plan.sbs.reserve(plan.sbs.size());
  for (const MinimalPaletteSuperblock& sb : plan.sbs) {
    scc::SbPlan scc_sb;
    switch (sb.partition) {
      case MinimalPaletteSuperblock::Partition::kNone:
        scc_sb.partition = scc::SbPlan::Partition::kNone;
        break;
      case MinimalPaletteSuperblock::Partition::kHorz:
        scc_sb.partition = scc::SbPlan::Partition::kHorz;
        break;
      case MinimalPaletteSuperblock::Partition::kVert:
        scc_sb.partition = scc::SbPlan::Partition::kVert;
        break;
    }
    for (const MinimalPaletteLeaf& leaf : sb.leaves) {
      if (!leaf.palette.has_palette) {
        error = "minimal palette stream leaf has no palette";
        return false;
      }
      const auto block_size =
          static_cast<partition::BlockSize>(leaf.bsize);
      const int bw = partition::block_width(block_size);
      const int bh = partition::block_height(block_size);
      if (!palette_allowed_for_block(plan.flags, leaf.bsize, bw, bh)) {
        error = "minimal palette stream leaf is not palette-signallable";
        return false;
      }
      scc::BlockPlan block;
      block.mi_row = leaf.mi_row;
      block.mi_col = leaf.mi_col;
      block.bsize = leaf.bsize;
      block.kind = scc::BlockKind::kPalette;
      block.palette = leaf.palette;
      scc_sb.leaves.push_back(std::move(block));
    }
    scc_plan.sbs.push_back(std::move(scc_sb));
  }

  scc::SccEncodeResult scc_out;
  if (!scc::encode_scc_frame(scc_plan, scc_out, error)) return false;
  out.frame = std::move(scc_out.frame);
  out.recon_y = std::move(scc_out.recon_y);
  out.recon_u = std::move(scc_out.recon_u);
  out.recon_v = std::move(scc_out.recon_v);
  out.tile_bytes = scc_out.tile_bytes;
  out.transform_skip_eob0 = true;
  return true;
}

bool write_minimal_palette_ivf(const MinimalPaletteFrameResult& result,
                               const MinimalPaletteFramePlan& plan,
                               const std::string& path, std::string& error) {
  scc::SccEncodeResult scc_out;
  scc_out.frame = result.frame;
  scc_out.recon_y = result.recon_y;
  scc_out.recon_u = result.recon_u;
  scc_out.recon_v = result.recon_v;
  scc_out.tile_bytes = result.tile_bytes;
  return scc::write_scc_ivf(scc_out, make_scc_config(plan), path, error);
}

bool write_minimal_palette_y4m(const MinimalPaletteFrameResult& result,
                               const MinimalPaletteFramePlan& plan,
                               const std::string& path, std::string& error) {
  scc::SccEncodeResult scc_out;
  scc_out.frame = result.frame;
  scc_out.recon_y = result.recon_y;
  scc_out.recon_u = result.recon_u;
  scc_out.recon_v = result.recon_v;
  scc_out.tile_bytes = result.tile_bytes;
  return scc::write_scc_y4m(scc_out, make_scc_config(plan), path, error);
}

}  // namespace av2enc::still::palette_stream
