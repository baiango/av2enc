#include "codec/inter/mv_stack/stack_mirror.h"

#include <cstddef>
#include <string>

#include "codec/inter/mv_stack/stack_io.h"

namespace av2enc::inter::mv_stack_full {
namespace {

// Independent little-endian cursor.  Every read is bounds-checked.
class Cursor {
 public:
  explicit Cursor(std::span<const std::uint8_t> record) : record_(record) {}

  bool u8(std::uint8_t* out) {
    if (out == nullptr || pos_ + 1 > record_.size()) return false;
    *out = record_[pos_++];
    return true;
  }
  bool u32(std::uint32_t* out) {
    if (out == nullptr || pos_ + 4 > record_.size()) return false;
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      v |= static_cast<std::uint32_t>(
               record_[pos_ + static_cast<std::size_t>(i)])
           << (8 * i);
    }
    pos_ += 4;
    *out = v;
    return true;
  }
  bool u64(std::uint64_t* out) {
    if (out == nullptr || pos_ + 8 > record_.size()) return false;
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
      v |= static_cast<std::uint64_t>(
               record_[pos_ + static_cast<std::size_t>(i)])
           << (8 * i);
    }
    pos_ += 8;
    *out = v;
    return true;
  }
  bool i32(std::int32_t* out) {
    std::uint32_t raw = 0;
    if (!u32(&raw)) return false;
    *out = static_cast<std::int32_t>(raw);
    return true;
  }
  std::size_t pos() const { return pos_; }
  std::size_t size() const { return record_.size(); }

 private:
  std::span<const std::uint8_t> record_;
  std::size_t pos_ = 0;
};

bool fail(std::string* error, const char* message) {
  if (error != nullptr) *error = message;
  return false;
}

bool mv_in_range(Mv mv) {
  return mv.y > kMvLow && mv.y < kMvUpp && mv.x > kMvLow && mv.x < kMvUpp;
}

}  // namespace

bool StackMirror::parse(std::span<const std::uint8_t> record,
                        std::string* error) {
  if (record.size() < 8 + 32) return fail(error, "record too short");

  // The checksum covers everything before the final 8 bytes.
  const std::size_t payload_size = record.size() - 8;
  const std::uint64_t expected =
      fnv1a64(record.subspan(0, payload_size));
  Cursor check(record.subspan(payload_size));
  std::uint64_t stored = 0;
  if (!check.u64(&stored)) return fail(error, "truncated checksum");
  if (stored != expected) return fail(error, "record checksum mismatch");

  Cursor cursor(record.subspan(0, payload_size));
  std::uint32_t magic = 0;
  if (!cursor.u32(&magic)) return fail(error, "truncated magic");
  if (magic != static_cast<std::uint32_t>(kStackRecordMagic)) {
    return fail(error, "bad stack-record magic");
  }
  std::uint8_t version = 0;
  std::uint8_t num_lists = 1;
  std::uint8_t sorted = 0;
  std::uint8_t reserved = 0;
  if (!cursor.u8(&version) || !cursor.u8(&num_lists) || !cursor.u8(&sorted) ||
      !cursor.u8(&reserved)) {
    return fail(error, "truncated header");
  }
  if (version != kStackRecordVersion) return fail(error, "unsupported version");
  if (num_lists != 1 && num_lists != 2) {
    return fail(error, "NumLists outside 1..2");
  }
  if (sorted > 1) return fail(error, "sorted flag outside 0..1");
  if (reserved != 0) return fail(error, "reserved byte is not zero");

  MirrorStack stack;
  stack.num_lists = num_lists == 2 ? 2 : 1;
  stack.sorted = sorted != 0;
  std::uint32_t count = 0;
  std::uint32_t num_nearest = 0;
  std::uint32_t num_warp = 0;
  if (!cursor.u32(&count) || !cursor.u32(&num_nearest) ||
      !cursor.u32(&num_warp) || !cursor.i32(&stack.prune_count) ||
      !cursor.i32(&stack.single_prune_count) ||
      !cursor.i32(&stack.derived_prune_count) ||
      !cursor.i32(&stack.derived_count)) {
    return fail(error, "truncated counts");
  }
  if (count > static_cast<std::uint32_t>(kMaxRefMvStackSize)) {
    return fail(error, "NumMvFound exceeds MAX_REF_MV_STACK_SIZE");
  }
  if (num_nearest > count) return fail(error, "numNearest exceeds NumMvFound");
  if (num_warp > static_cast<std::uint32_t>(kMaxWarpRefCandidates)) {
    return fail(error, "NumWarpFound exceeds MAX_WARP_REF_CANDIDATES");
  }
  stack.count = static_cast<int>(count);
  stack.num_nearest = static_cast<int>(num_nearest);
  stack.num_warp_found = static_cast<int>(num_warp);

  for (int i = 0; i < stack.count; ++i) {
    MirrorCandidate& c = stack.cand[i];
    for (int list = 0; list < stack.num_lists; ++list) {
      if (!cursor.i32(&c.mv[list].y) || !cursor.i32(&c.mv[list].x)) {
        return fail(error, "truncated candidate MV");
      }
    }
    if (stack.num_lists == 1) c.mv[1] = c.mv[0];
    if (!cursor.i32(&c.weight) || !cursor.i32(&c.row_offset) ||
        !cursor.i32(&c.col_offset)) {
      return fail(error, "truncated candidate fields");
    }
    std::uint8_t cwp = 0;
    std::uint8_t pad[3] = {0, 0, 0};
    if (!cursor.u8(&cwp) || !cursor.u8(&pad[0]) || !cursor.u8(&pad[1]) ||
        !cursor.u8(&pad[2])) {
      return fail(error, "truncated cwp field");
    }
    if (pad[0] != 0 || pad[1] != 0 || pad[2] != 0) {
      return fail(error, "non-zero candidate padding");
    }
    c.cwp = static_cast<std::int8_t>(cwp);
  }
  for (int i = 0; i < stack.num_warp_found; ++i) {
    for (int j = 0; j < 6; ++j) {
      if (!cursor.i32(&stack.warp_params[i].p[j])) {
        return fail(error, "truncated warp parameter");
      }
    }
  }
  if (cursor.pos() != payload_size) {
    return fail(error, "trailing bytes after the stack record");
  }
  stack_ = stack;
  return validate(error);
}

bool StackMirror::validate(std::string* error) const {
  const MirrorStack& s = stack_;
  if (s.count < 0 || s.count > kMaxRefMvStackSize) {
    return fail(error, "count outside the stack size");
  }
  if (s.num_nearest < 0 || s.num_nearest > s.count) {
    return fail(error, "numNearest outside 0..count");
  }
  if (s.num_warp_found < 0 || s.num_warp_found > kMaxWarpRefCandidates) {
    return fail(error, "NumWarpFound outside the warp stack size");
  }
  // The spec's guard is outside the pruning loop, so one inner sweep can push
  // the counter past the constant: at most MAX_PR_NUM - 1 + NumMvFound.
  if (s.prune_count < 0 ||
      s.prune_count > kMaxPrNum - 1 + kMaxRefMvStackSize) {
    return fail(error, "PruneCount outside its reachable range");
  }
  if (s.single_prune_count < 0 ||
      s.single_prune_count > kMaxDrPrNum - 1 + kMaxDrStackSize) {
    return fail(error, "SinglePruneCount outside its reachable range");
  }
  if (s.derived_prune_count < 0 ||
      s.derived_prune_count > kMaxDrPrNum - 1 + kMaxDrStackSize) {
    return fail(error, "DerivedPruneCount outside its reachable range");
  }
  if (s.derived_count < 0 || s.derived_count > kMaxDrStackSize) {
    return fail(error, "DerivedMvCount outside 0..MAX_DR_STACK_SIZE");
  }
  for (int i = 0; i < s.count; ++i) {
    for (int list = 0; list < s.num_lists; ++list) {
      if (!mv_in_range(s.cand[i].mv[list])) {
        return fail(error, "candidate MV outside (MV_LOW, MV_UPP)");
      }
    }
    if (s.cand[i].weight < 0) return fail(error, "negative candidate weight");
    if (s.cand[i].cwp < -64 || s.cand[i].cwp > 64) {
      return fail(error, "cwp outside a plausible range");
    }
  }
  // §7.12.2.19 moves the highest-weight entry to the front of [0, numNearest).
  if (s.sorted && s.num_nearest > 0) {
    for (int i = 1; i < s.num_nearest; ++i) {
      if (s.cand[0].weight < s.cand[i].weight) {
        return fail(error, "sorted prefix violates the move-to-front invariant");
      }
    }
  }
  return true;
}

bool StackMirror::resolve_mv(int drl_index, int mode, Mv global, Mv mvd,
                             Mv* out, std::string* error) const {
  return mirror_resolve_mv(stack_, drl_index, mode, global, mvd, out, error);
}

bool mirror_resolve_mv(const MirrorStack& stack, int drl_index, int mode,
                       Mv global, Mv mvd, Mv* out, std::string* error) {
  if (out == nullptr) return fail(error, "null output");
  if (mode == 1) {
    *out = clamp_mv_to_integer(global);
    return true;
  }
  if (mode != 0 && mode != 2) return fail(error, "mode outside 0..2");
  if (drl_index < 0 || drl_index > kMaxDrlIndex || drl_index >= stack.count) {
    return fail(error, "DRL index outside the mirrored stack");
  }
  const Mv pred = stack.cand[drl_index].mv[0];
  if (mode == 0) {
    *out = clamp_mv_to_integer(pred);
    return true;
  }
  Mv mv;
  mv.y = clamp_mv_to_integer(static_cast<std::int32_t>(
      static_cast<std::int64_t>(pred.y) + static_cast<std::int64_t>(mvd.y)));
  mv.x = clamp_mv_to_integer(static_cast<std::int32_t>(
      static_cast<std::int64_t>(pred.x) + static_cast<std::int64_t>(mvd.x)));
  *out = mv;
  return true;
}

bool StackMirror::equals(const Stack& encoder, std::string* why) const {
  const MirrorStack& m = stack_;
  if (m.count != encoder.count) {
    if (why != nullptr) *why = "candidate count differs";
    return false;
  }
  if (m.num_nearest != encoder.num_nearest) {
    if (why != nullptr) *why = "numNearest differs";
    return false;
  }
  if (m.sorted != encoder.sorted) {
    if (why != nullptr) *why = "sorted flag differs";
    return false;
  }
  if (m.num_warp_found != encoder.num_warp_found) {
    if (why != nullptr) *why = "NumWarpFound differs";
    return false;
  }
  for (int i = 0; i < m.count; ++i) {
    for (int list = 0; list < m.num_lists; ++list) {
      if (!(m.cand[i].mv[list] == encoder.cand[i].mv[list])) {
        if (why != nullptr) *why = "candidate MV differs";
        return false;
      }
    }
    if (m.cand[i].weight != encoder.cand[i].weight ||
        m.cand[i].row_offset != encoder.cand[i].row_offset ||
        m.cand[i].col_offset != encoder.cand[i].col_offset ||
        m.cand[i].cwp != encoder.cand[i].cwp) {
      if (why != nullptr) *why = "candidate metadata differs";
      return false;
    }
  }
  for (int i = 0; i < m.num_warp_found; ++i) {
    for (int j = 0; j < 6; ++j) {
      if (m.warp_params[i].p[j] != encoder.warp_params[i].p[j]) {
        if (why != nullptr) *why = "warp parameter differs";
        return false;
      }
    }
  }
  for (int i = 0; i < m.count; ++i) {
    for (int list = m.num_lists; list < 2; ++list) {
      if (!(m.cand[i].mv[list] == encoder.cand[i].mv[list])) {
        if (why != nullptr) *why = "unused list MV differs";
        return false;
      }
    }
  }
  return true;
}

bool lockstep(const Stack& encoder, int num_lists, Mv global,
              const MirrorStack& mirror, LockstepReport* report) {
  if (report == nullptr) return false;
  *report = LockstepReport{};
  std::string why;
  // Re-parse the record built from the encoder to get a class instance; the
  // caller already parsed `mirror`, so compare with it directly first.
  if (mirror.count != encoder.count || mirror.num_nearest != encoder.num_nearest ||
      mirror.sorted != encoder.sorted ||
      mirror.num_warp_found != encoder.num_warp_found) {
    report->fields_equal = false;
    report->why = "mirror header differs";
    return false;
  }
  for (int i = 0; i < mirror.count; ++i) {
    for (int list = 0; list < mirror.num_lists; ++list) {
      if (!(mirror.cand[i].mv[list] == encoder.cand[i].mv[list])) {
        report->fields_equal = false;
        report->why = "mirror candidate MV differs";
        return false;
      }
    }
    if (mirror.cand[i].weight != encoder.cand[i].weight ||
        mirror.cand[i].row_offset != encoder.cand[i].row_offset ||
        mirror.cand[i].col_offset != encoder.cand[i].col_offset ||
        mirror.cand[i].cwp != encoder.cand[i].cwp) {
      report->fields_equal = false;
      report->why = "mirror candidate metadata differs";
      return false;
    }
  }
  for (int i = 0; i < mirror.num_warp_found; ++i) {
    for (int j = 0; j < 6; ++j) {
      if (mirror.warp_params[i].p[j] != encoder.warp_params[i].p[j]) {
        report->fields_equal = false;
        report->why = "mirror warp parameter differs";
        return false;
      }
    }
  }

  const int last = std::min(mirror.count - 1, kMaxDrlIndex);
  const Mv mvd{3, -5};
  for (int drl = 0; drl <= last; ++drl) {
    for (int mode = 0; mode <= 2; ++mode) {
      Mv from_encoder;
      if (mode == 1) {
        from_encoder = clamp_mv_to_integer(global);
      } else {
        const Mv pred = encoder.cand[drl].mv[0];
        if (mode == 0) {
          from_encoder = clamp_mv_to_integer(pred);
        } else {
          from_encoder.y = clamp_mv_to_integer(static_cast<std::int32_t>(
              static_cast<std::int64_t>(pred.y) + mvd.y));
          from_encoder.x = clamp_mv_to_integer(static_cast<std::int32_t>(
              static_cast<std::int64_t>(pred.x) + mvd.x));
        }
      }
      Mv from_mirror;
      if (!mirror_resolve_mv(mirror, drl, mode, global, mvd, &from_mirror,
                             &why)) {
        report->resolved_equal = false;
        report->first_bad_drl = drl;
        report->first_bad_mode = mode;
        report->why = why;
        return false;
      }
      ++report->drl_checked;
      if (!(from_encoder == from_mirror)) {
        report->resolved_equal = false;
        report->first_bad_drl = drl;
        report->first_bad_mode = mode;
        report->why = "resolved MV differs";
        return false;
      }
    }
  }
  (void)num_lists;
  return true;
}

}  // namespace av2enc::inter::mv_stack_full
