#include "encoder/inter/mv_stack_full/stack_io.h"

#include <cstdio>
#include <string>

namespace av2enc::inter::mv_stack_full {
namespace {

void put_u8(std::vector<std::uint8_t>& out, std::uint8_t value) {
  out.push_back(value);
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  out.push_back(static_cast<std::uint8_t>(value & 0xffu));
  out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xffu));
  out.push_back(static_cast<std::uint8_t>((value >> 16) & 0xffu));
  out.push_back(static_cast<std::uint8_t>((value >> 24) & 0xffu));
}

void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xffu));
  }
}

void put_i32(std::vector<std::uint8_t>& out, std::int32_t value) {
  put_u32(out, static_cast<std::uint32_t>(value));
}

}  // namespace

std::vector<std::uint8_t> serialize_stack(const Stack& stack, int num_lists) {
  if (num_lists != 1 && num_lists != 2) num_lists = 1;
  std::vector<std::uint8_t> out;
  out.reserve(64 + static_cast<std::size_t>(stack.count) * 32);
  put_u32(out, static_cast<std::uint32_t>(kStackRecordMagic));
  put_u8(out, kStackRecordVersion);
  put_u8(out, static_cast<std::uint8_t>(num_lists));
  put_u8(out, static_cast<std::uint8_t>(stack.sorted ? 1 : 0));
  put_u8(out, 0);  // reserved
  put_u32(out, static_cast<std::uint32_t>(stack.count));
  put_u32(out, static_cast<std::uint32_t>(stack.num_nearest));
  put_u32(out, static_cast<std::uint32_t>(stack.num_warp_found));
  put_i32(out, stack.prune_count);
  put_i32(out, stack.single_prune_count);
  put_i32(out, stack.derived_prune_count);
  put_i32(out, stack.derived_count);
  for (int i = 0; i < stack.count; ++i) {
    for (int list = 0; list < num_lists; ++list) {
      put_i32(out, stack.cand[i].mv[list].y);
      put_i32(out, stack.cand[i].mv[list].x);
    }
    put_i32(out, stack.cand[i].weight);
    put_i32(out, stack.cand[i].row_offset);
    put_i32(out, stack.cand[i].col_offset);
    put_u8(out, static_cast<std::uint8_t>(stack.cand[i].cwp));
    put_u8(out, 0);
    put_u8(out, 0);
    put_u8(out, 0);
  }
  for (int i = 0; i < stack.num_warp_found; ++i) {
    for (int j = 0; j < 6; ++j) put_i32(out, stack.warp_params[i].p[j]);
  }
  put_u64(out, fnv1a64(out));
  return out;
}

std::uint64_t fnv1a64(std::span<const std::uint8_t> bytes) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::uint8_t byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::uint64_t stack_record_hash(const Stack& stack, int num_lists) {
  const std::vector<std::uint8_t> bytes = serialize_stack(stack, num_lists);
  return fnv1a64(bytes);
}

bool write_stack_record(const std::string& path, const Stack& stack,
                        int num_lists, std::string* error) {
  const std::vector<std::uint8_t> bytes = serialize_stack(stack, num_lists);
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    if (error != nullptr) *error = "cannot open " + path;
    return false;
  }
  const std::size_t written =
      std::fwrite(bytes.data(), 1, bytes.size(), file);
  std::fclose(file);
  if (written != bytes.size()) {
    if (error != nullptr) *error = "short write to " + path;
    return false;
  }
  return true;
}

}  // namespace av2enc::inter::mv_stack_full
