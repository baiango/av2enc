// AV2 v1.0.0 average-CDF bank implementation.  See avg_cdf_bank.h for the
// normative section map.  No allocation or I/O in the per-tile operations.

#include "encoder/entropy/avg_cdf/avg_cdf_bank.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace av2enc::entropy::avg_cdf {
namespace {

bool fail(std::string* error, const char* text) {
  if (error != nullptr) *error = text;
  return false;
}

bool add_overflow(std::size_t a, std::size_t b, std::size_t* out) {
  if (a > std::numeric_limits<std::size_t>::max() - b) return false;
  *out = a + b;
  return true;
}

}  // namespace

std::uint32_t average_log2(std::uint32_t tile_count) noexcept {
  // Spec assumes TileCols * TileRows >= 1.  Treat 0 as a single tile so a
  // caller that has not finished planning cannot shift by a bogus amount.
  if (tile_count <= 1u) return 0u;
  std::uint32_t log2 = 0u;
  while (tile_count > 1u) {
    tile_count >>= 1u;
    ++log2;
  }
  return std::min(log2, kMaxAverageLog2);
}

std::uint32_t average_tile_limit(std::uint32_t tile_count) noexcept {
  return 1u << average_log2(tile_count);
}

void average_cdf(std::uint16_t* cdf, const std::uint16_t* tile_cdf,
                 std::uint16_t symbols, std::uint32_t num_log2,
                 bool first_tile) noexcept {
  if (cdf == nullptr || tile_cdf == nullptr || symbols < 2 ||
      num_log2 > kMaxAverageLog2) {
    return;
  }
  const int n = static_cast<int>(symbols);
  if (first_tile) {
    // §8.2.4: cdf[i] = 1 << 15 for i < sz-2; rate from tile 0; count = 0.
    for (int i = 0; i < n - 1; ++i) {
      cdf[i] = static_cast<std::uint16_t>(kCdfScale);
    }
    cdf[n - 1] = tile_cdf[n - 1];
    cdf[n] = 0;
  }
  for (int i = 0; i < n - 1; ++i) {
    const std::uint32_t value = cdf[i];
    const std::uint32_t tile_value = tile_cdf[i];
    cdf[i] = static_cast<std::uint16_t>(
        value - ((kCdfScale - tile_value) >> num_log2));
  }
  cdf[n] = static_cast<std::uint16_t>(
      cdf[n] + static_cast<std::uint16_t>(tile_cdf[n] >> num_log2));
}

void blend_cdf(std::uint16_t* cdf, const std::uint16_t* saved_cdf,
               std::uint16_t symbols) noexcept {
  if (cdf == nullptr || saved_cdf == nullptr || symbols < 2) return;
  const int n = static_cast<int>(symbols);
  for (int i = 0; i < n - 1; ++i) {
    const std::uint32_t saved = saved_cdf[i];
    const std::uint32_t current = cdf[i];
    // cdf[i] = (1<<15) - (((1<<15)-saved) + 7*((1<<15)-cur) + 4) >> 3
    const std::uint32_t omitted_saved = kCdfScale - saved;
    const std::uint32_t omitted_current = kCdfScale - current;
    cdf[i] = static_cast<std::uint16_t>(
        kCdfScale - ((omitted_saved + 7u * omitted_current + 4u) >> 3));
  }
  cdf[n] = static_cast<std::uint16_t>(
      (static_cast<std::uint32_t>(saved_cdf[n]) +
       7u * static_cast<std::uint32_t>(cdf[n]) + 4u) >>
      3);
}

bool CdfBank::init(std::span<const CdfTable> tables,
                   const CdfBankConfig& config, std::string* error) {
  if (error != nullptr) error->clear();
  if (tables.empty()) return fail(error, "avg_cdf: empty CDF table list");
  config_ = config;
  tables_.assign(tables.begin(), tables.end());
  storage_.clear();
  owning_ = false;
  return validate(error);
}

bool CdfBank::init_owned(std::span<const CdfSpec> specs,
                         const CdfBankConfig& config, std::string* error) {
  if (error != nullptr) error->clear();
  if (specs.empty()) return fail(error, "avg_cdf: empty CDF spec list");
  if (config.reference_contexts > 64u) {
    return fail(error, "avg_cdf: reference_contexts must be <= 64");
  }

  config_ = config;
  tables_.clear();
  storage_.clear();
  owning_ = false;

  const std::size_t contexts = config.reference_contexts;
  std::size_t total = 0;
  for (const CdfSpec& spec : specs) {
    if (spec.symbols < 2 || spec.symbols > kMaxSymbols) {
      return fail(error, "avg_cdf: symbols must be in [2, 8]");
    }
    if (spec.defaults == nullptr) {
      return fail(error, "avg_cdf: null default CDF");
    }
    const std::size_t entries = static_cast<std::size_t>(spec.symbols) + 1u;
    std::size_t per_table = 0;
    if (!add_overflow(entries, entries, &per_table) ||          // frame + tile
        !add_overflow(per_table, entries, &per_table) ||        // + saved
        !add_overflow(per_table, entries, &per_table)) {        // + defaults
      return fail(error, "avg_cdf: CDF storage size overflow");
    }
    for (std::size_t c = 0; c < contexts; ++c) {
      if (!add_overflow(per_table, entries, &per_table)) {
        return fail(error, "avg_cdf: CDF context storage size overflow");
      }
    }
    if (!add_overflow(total, per_table, &total)) {
      return fail(error, "avg_cdf: CDF storage size overflow");
    }
  }

  storage_.assign(total, 0);
  tables_.reserve(specs.size());
  std::size_t offset = 0;
  for (const CdfSpec& spec : specs) {
    const std::size_t entries = static_cast<std::size_t>(spec.symbols) + 1u;
    CdfTable table;
    table.symbols = spec.symbols;
    table.frame = storage_.data() + offset;
    offset += entries;
    table.tile = storage_.data() + offset;
    offset += entries;
    table.saved = storage_.data() + offset;
    offset += entries;
    table.defaults = storage_.data() + offset;
    std::memcpy(storage_.data() + offset, spec.defaults,
                entries * sizeof(std::uint16_t));
    offset += entries;
    if (contexts != 0u) {
      table.reference_base = storage_.data() + offset;
      table.reference_stride = static_cast<std::uint32_t>(entries);
      offset += entries * contexts;
    }
    tables_.push_back(table);
  }
  owning_ = true;
  if (!validate(error)) {
    tables_.clear();
    storage_.clear();
    owning_ = false;
    return false;
  }
  reset();
  return true;
}

bool CdfBank::validate(std::string* error) const {
  for (const CdfTable& table : tables_) {
    if (table.symbols < 2 || table.symbols > kMaxSymbols) {
      return fail(error, "avg_cdf: symbols must be in [2, 8]");
    }
    if (table.frame == nullptr || table.tile == nullptr ||
        table.saved == nullptr || table.defaults == nullptr) {
      return fail(error, "avg_cdf: null CDF array pointer");
    }
    if (config_.reference_contexts != 0u) {
      if (table.reference_base == nullptr ||
          table.reference_stride !=
              static_cast<std::uint32_t>(table.symbols) + 1u) {
        return fail(error,
                    "avg_cdf: reference contexts need reference_base and "
                    "reference_stride == N+1");
      }
    }
  }
  if (!config_.enable_avg_cdf && !config_.avg_cdf_type) {
    // This is the `copyCdf` branch, not an error.  context_update_tile_id is
    // validated when save_tile_cdfs() sees the tile count.
  }
  return true;
}

void CdfBank::reset() noexcept {
  for (const CdfTable& table : tables_) {
    const std::size_t entries =
        static_cast<std::size_t>(table.symbols) + 1u;
    std::memcpy(table.frame, table.defaults,
                entries * sizeof(std::uint16_t));
    std::memcpy(table.tile, table.defaults,
                entries * sizeof(std::uint16_t));
    std::memcpy(table.saved, table.defaults,
                entries * sizeof(std::uint16_t));
  }
}

void CdfBank::begin_frame(bool reset_to_defaults) noexcept {
  if (reset_to_defaults) {
    reset();
    return;
  }
  // Keep the current frame CDFs.  `tile` and `saved` are refreshed at tile 0
  // and at frame end; do not carry a previous tile's working copy into the new
  // frame.
  for (const CdfTable& table : tables_) {
    const std::size_t entries =
        static_cast<std::size_t>(table.symbols) + 1u;
    std::memcpy(table.tile, table.frame,
                entries * sizeof(std::uint16_t));
  }
}

void CdfBank::load_tile_cdfs() noexcept {
  for (const CdfTable& table : tables_) {
    const std::size_t entries =
        static_cast<std::size_t>(table.symbols) + 1u;
    std::memcpy(table.tile, table.frame,
                entries * sizeof(std::uint16_t));
  }
}

bool CdfBank::save_tile_cdfs(std::uint32_t tile_num,
                             std::uint32_t tile_count) noexcept {
  if (tile_count == 0u || tile_num >= tile_count) return false;

  if (config_.enable_avg_cdf && config_.avg_cdf_type) {
    const std::uint32_t num_log2 = average_log2(tile_count);
    if (tile_num >= (1u << num_log2)) {
      // §8.2.4: avgCdf is 0 for tiles outside the first 1<<numLog2.  Neither
      // copyCdf nor avgCdf runs for those tiles.
      return true;
    }
    const bool first = tile_num == 0u;
    for (const CdfTable& table : tables_) {
      average_cdf(table.saved, table.tile, table.symbols, num_log2, first);
    }
    return true;
  }

  // copyCdf branch: only the selected tile becomes Saved*.
  if (tile_num != config_.context_update_tile_id) return true;
  for (const CdfTable& table : tables_) {
    const std::size_t entries =
        static_cast<std::size_t>(table.symbols) + 1u;
    std::memcpy(table.saved, table.tile,
                entries * sizeof(std::uint16_t));
  }
  return true;
}

void CdfBank::average_at_frame_end() noexcept {
  for (const CdfTable& table : tables_) {
    const std::size_t entries =
        static_cast<std::size_t>(table.symbols) + 1u;
    std::memcpy(table.frame, table.saved,
                entries * sizeof(std::uint16_t));
    // §7.5: the last entry is the use count; scale it by 3/4 after the copy.
    table.frame[table.symbols] = static_cast<std::uint16_t>(
        (3u * static_cast<std::uint32_t>(table.saved[table.symbols])) >> 2);
  }
}

bool CdfBank::save_reference(std::uint32_t ctx) noexcept {
  if (!owning_ || config_.reference_contexts == 0u ||
      ctx >= config_.reference_contexts) {
    return false;
  }
  for (const CdfTable& table : tables_) {
    const std::size_t entries =
        static_cast<std::size_t>(table.symbols) + 1u;
    std::memcpy(table.reference_base + entries * ctx, table.frame,
                entries * sizeof(std::uint16_t));
  }
  return true;
}

bool CdfBank::load_reference(std::uint32_t ctx) noexcept {
  if (!owning_ || config_.reference_contexts == 0u ||
      ctx >= config_.reference_contexts) {
    return false;
  }
  for (const CdfTable& table : tables_) {
    const std::size_t entries =
        static_cast<std::size_t>(table.symbols) + 1u;
    std::memcpy(table.frame, table.reference_base + entries * ctx,
                entries * sizeof(std::uint16_t));
  }
  return true;
}

bool CdfBank::blend_reference(std::uint32_t ctx) noexcept {
  if (!owning_ || config_.reference_contexts == 0u ||
      ctx >= config_.reference_contexts) {
    return false;
  }
  for (const CdfTable& table : tables_) {
    const std::size_t entries =
        static_cast<std::size_t>(table.symbols) + 1u;
    blend_cdf(table.frame, table.reference_base + entries * ctx,
              table.symbols);
  }
  return true;
}

}  // namespace av2enc::entropy::avg_cdf
