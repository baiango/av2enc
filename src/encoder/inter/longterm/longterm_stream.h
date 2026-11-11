// Stream assembly for long-term-reference / open-random-access sequences.
//
// The M2 core emits one intra keyframe followed by P frames that reference one
// slot. This module keeps those tile payloads byte-for-byte and replaces the
// headers, so the long-term-reference layer is a header-level splice:
//
//   [TD][SEQ][CLK A]                       temporal unit 0 (closed RA)
//   [TD][OLK K][P ...]                     temporal unit 1 (open RA)
//   [TD][P ...]                            temporal unit 2..N
//
// The tile payload of every frame is produced by m2::encode_m2_sequence(); the
// caller is responsible for encoding each tile against the reference picture
// that its rewritten header will name. See longterm_gate.cpp for the worked
// example and README.md for the integration contract.
#ifndef AV2ENC_ENCODER_INTER_LONGTERM_LONGTERM_STREAM_H_
#define AV2ENC_ENCODER_INTER_LONGTERM_LONGTERM_STREAM_H_

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "encoder/inter/longterm/longterm.h"

namespace av2enc::inter::longterm {

// One frame inside a temporal unit.
struct PlannedFrame {
  FrameHeader header;
  // §5.20 tile payload including the §8.2.4 exit_symbol() bits. Taken
  // verbatim from an M2 sequence unit.
  std::vector<std::uint8_t> tile_payload;
};

struct TemporalUnitPlan {
  std::vector<PlannedFrame> frames;
};

struct FrameEvent {
  PoolEvents pool;   // events of apply_frame()
  std::uint32_t order_hint = 0;
  std::string role;
  std::uint8_t obu_type = 0;
  std::size_t obu_bytes = 0;
};

struct AssembledSequence {
  // One entry per temporal unit, ready for the IVF writer.
  std::vector<std::vector<std::uint8_t>> units;
  std::vector<FrameEvent> frames;   // flattened, coding order
  std::uint32_t valid_slots_at_end = 0;
  PoolState final_pool;
};

// Extracts the trailing tile payload from each M2 temporal unit. The M2 writer
// places the tile last, so the payload is the trailing `tile_bytes[i]` bytes.
bool extract_m2_tiles(std::span<const std::vector<std::uint8_t>> units,
                      std::span<const std::uint32_t> tile_bytes,
                      std::vector<std::vector<std::uint8_t>>* tiles,
                      std::string* error);

// Validates the whole plan (temporal-unit rules, pool state, reference
// validity, open/closed random access) and emits one OBU per frame wrapped in
// its temporal unit. `sequence_header_in_first_unit` controls whether the
// sequence header OBU is repeated ahead of the first frame of unit 0.
bool assemble_sequence(const SequenceConfig& config,
                       const std::vector<TemporalUnitPlan>& units,
                       AssembledSequence* out, std::string* error);

}  // namespace av2enc::inter::longterm

#endif  // AV2ENC_ENCODER_INTER_LONGTERM_LONGTERM_STREAM_H_
