// Deterministic serialization of a §7.12.2 MV stack (the encoder-side record).
//
// The format is a flat little-endian record with an explicit magic, version and
// a trailing FNV-1a checksum.  It is *not* a bitstream format: it exists so the
// stack the encoder used can be archived, diffed and re-parsed by an
// independent mirror (stack_mirror.h).  Every field is written in a fixed order
// and the record is self-delimiting.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "encoder/inter/mv_stack_full/mv_stack_full.h"

namespace av2enc::inter::mv_stack_full {

inline constexpr std::uint8_t kStackRecordVersion = 1;
inline constexpr std::uint64_t kStackRecordMagic = 0x4653564d;  // "MVSF" LE

// `num_lists` is 1 for a single-prediction stack and 2 for a compound one; it
// is part of the record because a single stack's list-1 entries are unused.
std::vector<std::uint8_t> serialize_stack(const Stack& stack, int num_lists);

// FNV-1a 64 over the record.  Used by the pinned goldens.
std::uint64_t fnv1a64(std::span<const std::uint8_t> bytes);

// Convenience: serialized record hash.
std::uint64_t stack_record_hash(const Stack& stack, int num_lists);

bool write_stack_record(const std::string& path, const Stack& stack,
                        int num_lists, std::string* error);

}  // namespace av2enc::inter::mv_stack_full
