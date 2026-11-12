#include "codec/inter/longterm/longterm_stream.h"

#include <cstddef>

namespace av2enc::inter::longterm {
namespace {

bool set_error(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
  return false;
}

const char* role_name(FrameRole role) {
  switch (role) {
    case FrameRole::kClosedLoopKey:
      return "CLK";
    case FrameRole::kOpenLoopKey:
      return "OLK";
    case FrameRole::kLeading:
      return "LEADING";
    case FrameRole::kRegular:
      return "REGULAR";
  }
  return "?";
}

}  // namespace

bool extract_video_tiles(std::span<const std::vector<std::uint8_t>> units,
                      std::span<const std::uint32_t> tile_bytes,
                      std::vector<std::vector<std::uint8_t>>* tiles,
                      std::string* error) {
  if (tiles == nullptr) return set_error(error, "extract_video_tiles: null out");
  if (units.size() != tile_bytes.size()) {
    return set_error(error, "M2 unit/tile count mismatch");
  }
  tiles->clear();
  tiles->reserve(units.size());
  for (std::size_t i = 0; i < units.size(); ++i) {
    const std::size_t count = tile_bytes[i];
    if (count == 0 || count > units[i].size()) {
      return set_error(error, "M2 tile size " + std::to_string(count) +
                                  " does not fit unit " + std::to_string(i));
    }
    const auto begin = units[i].end() - static_cast<std::ptrdiff_t>(count);
    tiles->emplace_back(begin, units[i].end());
  }
  return true;
}

bool assemble_sequence(const SequenceConfig& config,
                       const std::vector<TemporalUnitPlan>& units,
                       AssembledSequence* out, std::string* error) {
  if (out == nullptr) return set_error(error, "assemble_sequence: null out");
  if (units.empty()) return set_error(error, "no temporal units");
  for (const TemporalUnitPlan& unit : units) {
    if (unit.frames.empty()) return set_error(error, "empty temporal unit");
  }

  PoolState pool;
  reset_pool(config, &pool);
  AssembledSequence result;
  bool first_picture = true;

  for (std::size_t u = 0; u < units.size(); ++u) {
    const TemporalUnitPlan& unit = units[u];
    std::vector<FrameHeader> headers;
    headers.reserve(unit.frames.size());
    for (const PlannedFrame& frame : unit.frames) {
      headers.push_back(frame.header);
    }
    if (!validate_temporal_unit(config, headers, error)) {
      return set_error(error, "temporal unit " + std::to_string(u) + ": " +
                                 (error != nullptr ? *error : "invalid"));
    }
    if (first_picture && unit.frames.front().header.role !=
                             FrameRole::kClosedLoopKey) {
      return set_error(error,
                       "a coded video sequence must start at a closed random "
                       "access point (OBU_CLOSED_LOOP_KEY)");
    }

    std::vector<std::uint8_t> bytes;
    if (u == 0) {
      const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
      const std::vector<std::uint8_t> seq = make_sequence_header_obu(config);
      bytes.reserve(td.size() + seq.size() + 1024U);
      bytes.insert(bytes.end(), td.begin(), td.end());
      bytes.insert(bytes.end(), seq.begin(), seq.end());
    } else {
      const std::vector<std::uint8_t> td = make_temporal_delimiter_obu();
      bytes.insert(bytes.end(), td.begin(), td.end());
    }

    for (std::size_t i = 0; i < unit.frames.size(); ++i) {
      const PlannedFrame& frame = unit.frames[i];
      const bool first_in_tu = i == 0;
      if (!validate_frame_header(config, frame.header, error)) {
        return set_error(error, "unit " + std::to_string(u) + " frame " +
                                   std::to_string(i) + ": " +
                                   (error != nullptr ? *error : "invalid"));
      }
      if (frame.header.role == FrameRole::kClosedLoopKey &&
          !first_in_tu) {
        return set_error(error, "a CLK must be the first frame of its TU");
      }
      PoolEvents events;
      if (!apply_frame(config, frame.header, first_in_tu, &pool, &events,
                       error)) {
        return set_error(error, "unit " + std::to_string(u) + " frame " +
                                   std::to_string(i) + ": " +
                                   (error != nullptr ? *error : "invalid"));
      }
      const std::vector<std::uint8_t> obu =
          make_frame_obu(config, frame.header, frame.tile_payload);
      FrameEvent record;
      record.pool = events;
      record.order_hint = frame.header.order_hint;
      record.role = role_name(frame.header.role);
      record.obu_type = obu_type_of(frame.header.role);
      record.obu_bytes = obu.size();
      result.frames.push_back(std::move(record));
      bytes.insert(bytes.end(), obu.begin(), obu.end());
      first_picture = false;
    }
    result.units.push_back(std::move(bytes));
  }

  result.valid_slots_at_end = 0;
  for (int i = 0; i < config.num_ref_frames; ++i) {
    if (pool.slots[static_cast<std::size_t>(i)].valid) {
      result.valid_slots_at_end |= 1U << i;
    }
  }
  result.final_pool = pool;
  *out = std::move(result);
  return true;
}

}  // namespace av2enc::inter::longterm
