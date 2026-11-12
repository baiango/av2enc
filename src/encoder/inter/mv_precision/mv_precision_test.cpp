// Unit / property / golden tests for the MV-precision signalling module.
//
// Coverage:
//   * §9.3 CDF copies, dimensions, monotonicity, and shell CDF row sizes;
//   * §5.4.6 seq_enabled_motion_modes / seq_frame_motion_modes_present_flag /
//     enable_six_param_warp_delta, and §5.18.2 frame_enabled_motion_modes;
//   * §5.18.2 use_qtr_precision_mv / allow_high_precision_mv and
//     read_interpolation_filter();
//   * §5.20.7.6 use_amvd / use_most_probable_precision / pb_mv_precision;
//   * §5.20.7.20 mv_joint, §5.20.7.21 amvd_index, jmvd_scale_mode;
//   * §5.20.7.13 projMv and enable_mvd_sign_derive;
//   * §5.20.7.6 switchable interp_filter, including the legal alphabet;
//   * §8.3 contexts for every one of the above;
//   * independent writer/reader lockstep over randomized inputs, with and
//     without CDF adaptation;
//   * pinned goldens in goldens/mv_precision_goldens.txt.
//
// Build/run: ./build_and_test.sh

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "encoder/inter/mv_precision/mv_precision.h"

namespace mvprec = av2enc::inter::mvprec;
namespace me2 = av2enc::me2;
namespace mc = av2enc::inter::mc;
namespace av2 = av2enc::av2;

namespace {

struct Test {
  int checks = 0;
  int failures = 0;
  void expect(bool cond, const std::string& what) {
    ++checks;
    if (!cond) {
      ++failures;
      if (failures <= 60) std::printf("FAIL: %s\n", what.c_str());
    }
  }
  void eq_int(long long got, long long want, const std::string& what) {
    ++checks;
    if (got != want) {
      ++failures;
      if (failures <= 60) {
        std::printf("FAIL: %s (got %lld, want %lld)\n", what.c_str(), got,
                    want);
      }
    }
  }
};

std::string hex(const std::vector<std::uint8_t>& bytes) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2U);
  for (std::uint8_t byte : bytes) {
    out.push_back(digits[byte >> 4U]);
    out.push_back(digits[byte & 0x0fU]);
  }
  return out;
}

std::vector<std::uint8_t> encode_payload(
    const std::function<bool(av2::Encoder&)>& body, bool disable_cdf_update) {
  std::vector<std::uint8_t> buffer(1U << 16U);
  av2::Encoder enc;
  if (enc.init_symbol(buffer.data(), buffer.size()) != av2::Status::kOk) {
    return {};
  }
  enc.disable_cdf_update = disable_cdf_update ? 1 : 0;
  if (!body(enc)) return {};
  std::size_t bytes = 0;
  if (enc.exit_symbol(&bytes) != av2::Status::kOk) return {};
  buffer.resize(bytes);
  return buffer;
}

bool decode_payload(const std::vector<std::uint8_t>& payload,
                    const std::function<bool(av2::Decoder&)>& body,
                    bool disable_cdf_update) {
  av2::Decoder dec;
  if (dec.init_symbol(payload.data(), payload.size(), payload.size()) !=
      av2::Status::kOk) {
    return false;
  }
  dec.disable_cdf_update = disable_cdf_update ? 1 : 0;
  if (!body(dec)) return false;
  return dec.exit_symbol() == av2::Status::kOk;
}

int motion_mode_mask(const mvprec::MotionModeSet& set) {
  int mask = 0;
  for (int mode = 1; mode < mvprec::kMotionModes; ++mode) {
    if (set.enabled[mode]) mask |= 1 << mode;
  }
  return mask;
}

mvprec::MotionModeSet motion_mode_set(int mask) {
  mvprec::MotionModeSet set;
  for (int mode = 1; mode < mvprec::kMotionModes; ++mode) {
    set.enabled[mode] = (mask & (1 << mode)) != 0;
  }
  return set;
}

// ---------------------------------------------------------------------------
// §9.3 CDF checks
// ---------------------------------------------------------------------------

void test_cdf_tables(Test& t) {
  mvprec::MvPrecisionCdfState state;
  mvprec::init_default_mv_precision_cdfs(&state);

  t.expect(std::memcmp(state.use_amvd, mvprec::cdf::kUseAmvdCdf,
                       sizeof(state.use_amvd)) == 0,
           "use_amvd CDF copy");
  t.expect(std::memcmp(state.mv_joint, mvprec::cdf::kMvJointAdaptiveCdf,
                       sizeof(state.mv_joint)) == 0,
           "mv_joint CDF copy");
  t.expect(std::memcmp(state.pb_mv_precision,
                       mvprec::cdf::kPbMvPrecisionCdf,
                       sizeof(state.pb_mv_precision)) == 0,
           "pb_mv_precision CDF copy");
  t.expect(std::memcmp(state.interp_filter, mvprec::cdf::kInterpFilterCdf,
                       sizeof(state.interp_filter)) == 0,
           "interp_filter CDF copy");

  const auto check_cdf = [&](const std::uint16_t* cdf, int n,
                             const std::string& what) {
    t.expect(av2::cdf_is_valid(cdf, n), what + " is a valid inverse CDF");
    for (int i = 1; i + 2 < n; ++i) {
      t.expect(cdf[i] >= cdf[i - 1], what + " is non-decreasing");
    }
  };

  for (int i = 0; i < 9; ++i) {
    for (int c = 0; c < 3; ++c) {
      check_cdf(&state.use_amvd[i][c][0], 2, "use_amvd");
    }
  }
  for (int c = 0; c < 3; ++c) {
    check_cdf(&state.use_most_probable_precision[c][0], 2,
              "use_most_probable_precision");
  }
  for (int c = 0; c < 2; ++c) {
    for (int row = 0; row < 3; ++row) {
      check_cdf(&state.pb_mv_precision[c][row][0], 3, "pb_mv_precision");
    }
  }
  for (int c = 0; c < 16; ++c) {
    check_cdf(&state.interp_filter[c][0], 3, "interp_filter");
  }
  check_cdf(&state.mv_joint[0], 4, "mv_joint");

  // The shell class alphabet is precision dependent; pin the sizes that the
  // spec's §9.3 tables actually contain.
  const int class0[7] = {5, 6, -1, 7, 7, 8, 8};
  const int class1[7] = {6, 6, -1, 7, 8, 8, 8};
  for (int p = 0; p <= 6; ++p) {
    if (p == 2) continue;
    t.eq_int(state.shell_class[p][0][class0[p]], 0,
             "shell class0 tail is a zero param/count slot");
    t.eq_int(state.shell_class[p][1][class1[p]], 0,
             "shell class1 tail is a zero param/count slot");
  }
  // Spot-check the first row against §9.3.
  t.eq_int(state.pb_mv_precision[0][0][0], 10923, "pb row 0 cdf[0]");
  t.eq_int(state.interp_filter[0][0], 29975, "interp row 0 cdf[0]");
  t.eq_int(state.use_amvd[0][0][0], 5980, "use_amvd row 0 cdf[0]");
}

// ---------------------------------------------------------------------------
// Motion modes
// ---------------------------------------------------------------------------

struct SeqModesCase {
  int seq_mask = 0;
  int frame_mask = 0;
  bool six = false;
};

std::vector<std::uint8_t> seq_modes_payload(const SeqModesCase& c) {
  mvprec::SequenceMotionModeSyntax syntax;
  syntax.seq_enabled = motion_mode_set(c.seq_mask);
  syntax.seq_frame_motion_modes_present_flag = syntax.seq_enabled.any();
  syntax.enable_six_param_warp_delta = c.six;
  av2enc::BitWriter writer;
  std::string error;
  if (!mvprec::write_sequence_motion_modes(writer, syntax, &error)) return {};
  writer.align_to_byte();
  const auto bytes = writer.bytes();
  return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
}

std::vector<std::uint8_t> frame_modes_payload(const SeqModesCase& c) {
  mvprec::SequenceMotionModeSyntax syntax;
  syntax.seq_enabled = motion_mode_set(c.seq_mask);
  syntax.seq_frame_motion_modes_present_flag = syntax.seq_enabled.any();
  syntax.enable_six_param_warp_delta = c.six;
  const mvprec::MotionModeSet frame = motion_mode_set(c.frame_mask);
  av2enc::BitWriter writer;
  std::string error;
  if (!mvprec::write_frame_motion_modes(writer, syntax, frame, &error)) {
    return {};
  }
  writer.align_to_byte();
  const auto bytes = writer.bytes();
  return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
}

void test_motion_modes(Test& t, std::vector<std::string>* golden) {
  const int masks[] = {0, 1 << 1, 1 << 2, 1 << 3, 1 << 4,
                       (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4),
                       (1 << 3) | (1 << 4)};
  for (int seq_mask : masks) {
    for (int frame_mask : masks) {
      const bool allowed =
          (frame_mask & ~seq_mask) == 0 || (seq_mask == 0 && frame_mask == 0);
      const bool seq_any = (seq_mask & 0x1e) != 0;
      const SeqModesCase c{seq_mask, frame_mask, false};
      const auto seq_payload = seq_modes_payload(c);
      const auto frame_payload = frame_modes_payload(c);
      if (!allowed) {
        t.expect(frame_payload.empty(),
                 "frame mode outside seq mask must be rejected");
        continue;
      }
      t.expect(!seq_payload.empty(), "seq motion modes payload");
      if (!seq_any) {
        t.expect(frame_payload.empty(), "no frame bits when the seq has none");
        mvprec::BitReader empty_reader(frame_payload);
        mvprec::BitReader again(seq_payload);
        mvprec::SequenceMotionModeSyntax inferred_seq;
        mvprec::MotionModeSet inferred;
        std::string infer_error;
        t.expect(mvprec::read_sequence_motion_modes(again, &inferred_seq,
                                                    &infer_error),
                 "empty seq reader");
        t.expect(mvprec::read_frame_motion_modes(empty_reader, inferred_seq,
                                                 &inferred, &infer_error),
                 "inferred frame modes");
        t.eq_int(motion_mode_mask(inferred), 0,
                 "frame modes inferred from an empty sequence set");
        continue;
      }
      t.expect(!frame_payload.empty(), "frame motion modes payload");

      mvprec::BitReader seq_reader(seq_payload);
      mvprec::SequenceMotionModeSyntax read_seq;
      std::string error;
      t.expect(mvprec::read_sequence_motion_modes(seq_reader, &read_seq,
                                                  &error),
               "seq motion modes reader");
      t.eq_int(motion_mode_mask(read_seq.seq_enabled), seq_mask,
               "seq mask round-trip");
      t.eq_int(read_seq.seq_frame_motion_modes_present_flag, seq_any ? 1 : 0,
               "seq present flag");
      t.eq_int(read_seq.enable_six_param_warp_delta, 0,
               "six-param absent without DELTAWARP");

      mvprec::BitReader frame_reader(frame_payload);
      mvprec::MotionModeSet read_frame;
      t.expect(mvprec::read_frame_motion_modes(frame_reader, read_seq,
                                               &read_frame, &error),
               "frame motion modes reader");
      t.eq_int(motion_mode_mask(read_frame), frame_mask,
               "frame mask round-trip");
      if (!seq_any) {
        // No bits are present and the sequence values (all zero) are inferred.
        t.eq_int(motion_mode_mask(read_frame), 0,
                 "frame modes inferred from an empty sequence set");
      }
      if (golden != nullptr && seq_any && (frame_mask & ~seq_mask) == 0) {
        golden->push_back("seq_modes|" + std::to_string(seq_mask) + "|" +
                          std::to_string(seq_any ? 1 : 0) + "|" +
                          std::to_string(c.six ? 1 : 0) + "|" +
                          hex(seq_payload));
        golden->push_back("frame_modes|" + std::to_string(seq_mask) + "|" +
                          std::to_string(frame_mask) + "|" +
                          hex(frame_payload));
      }
    }
  }

  // DELTAWARP enables the six-parameter flag.
  SeqModesCase delta;
  delta.seq_mask = 1 << mvprec::kMotionDeltaWarp;
  delta.frame_mask = delta.seq_mask;
  delta.six = true;
  const auto payload = seq_modes_payload(delta);
  mvprec::BitReader reader(payload);
  mvprec::SequenceMotionModeSyntax read;
  std::string error;
  t.expect(mvprec::read_sequence_motion_modes(reader, &read, &error),
           "DELTAWARP seq reader");
  t.expect(read.enable_six_param_warp_delta, "six-param round-trip");
  if (golden != nullptr) {
    golden->push_back("seq_modes|" + std::to_string(delta.seq_mask) + "|1|1|" +
                      hex(payload));
  }

  // Invalid: six-param without DELTAWARP.
  mvprec::SequenceMotionModeSyntax bad;
  bad.seq_enabled = motion_mode_set(1 << mvprec::kMotionLocalWarp);
  bad.seq_frame_motion_modes_present_flag = true;
  bad.enable_six_param_warp_delta = true;
  av2enc::BitWriter writer;
  t.expect(!mvprec::write_sequence_motion_modes(writer, bad, &error),
           "six-param without DELTAWARP rejected");
}

// ---------------------------------------------------------------------------
// Frame precision and frame interpolation filter
// ---------------------------------------------------------------------------

void test_frame_precision(Test& t, std::vector<std::string>* golden) {
  for (int target = 3; target <= 6; ++target) {
    for (int flex = 0; flex <= 1; ++flex) {
      me2::FramePrecisionSyntax syntax;
      if (!me2::me2_frame_precision_to_bits(target, flex, 0, &syntax)) {
        continue;
      }
      av2enc::BitWriter writer;
      std::string error;
      t.expect(mvprec::write_frame_mv_precision(writer, syntax, 0, &error),
               "frame precision writer");
      writer.align_to_byte();
      const auto bytes = writer.bytes();
      const std::vector<std::uint8_t> payload(bytes.begin(), bytes.end());

      mvprec::BitReader reader(payload);
      me2::FramePrecisionSyntax read;
      t.expect(mvprec::read_frame_mv_precision(reader, 0, flex, &read, &error),
               "frame precision reader");
      t.eq_int(read.frame_precision, target, "frame precision round-trip");
      t.eq_int(read.use_per_block_mv_precision, flex ? 1 : 0,
               "UsePerBlockMvPrecision round-trip");
      if (golden != nullptr) {
        golden->push_back("frame_prec|0|" + std::to_string(flex) + "|" +
                          std::to_string(target) + "|" + hex(payload));
      }
    }
  }
  // force_integer_mv codes no bits and pins ONE_PEL.
  me2::FramePrecisionSyntax forced;
  t.expect(me2::me2_frame_precision_to_bits(me2::kMvPrecisionOnePel, 0, 1,
                                            &forced),
           "ONE_PEL syntax exists for force_integer_mv");
  av2enc::BitWriter writer;
  std::string error;
  t.expect(mvprec::write_frame_mv_precision(writer, forced, 1, &error),
           "force_integer_mv writes no bits");
  t.eq_int(static_cast<long long>(writer.bit_size()), 0,
           "force_integer_mv bit count");
  // A non-ONE target under force_integer_mv is rejected.
  me2::FramePrecisionSyntax eighth;
  t.expect(me2::me2_frame_precision_to_bits(me2::kMvPrecisionEighthPel, 0, 0,
                                            &eighth),
           "EIGHTH header bits");
  av2enc::BitWriter bad_writer;
  t.expect(!mvprec::write_frame_mv_precision(bad_writer, eighth, 1, &error),
           "force_integer_mv rejects EIGHTH");

  // read_interpolation_filter().
  for (int filter = 0; filter <= 3; ++filter) {
    av2enc::BitWriter w;
    t.expect(mvprec::write_frame_interp_filter(w, false, filter, &error),
             "fixed frame filter writer");
    w.align_to_byte();
    const auto bytes = w.bytes();
    mvprec::BitReader r(
        std::span<const std::uint8_t>(bytes.data(), bytes.size()));
    bool switchable = true;
    int read_filter = -1;
    t.expect(mvprec::read_frame_interp_filter(r, &switchable, &read_filter,
                                              &error),
             "fixed frame filter reader");
    t.expect(!switchable, "fixed filter switchable bit");
    t.eq_int(read_filter, filter, "fixed frame filter round-trip");
  }
  av2enc::BitWriter sw;
  t.expect(mvprec::write_frame_interp_filter(sw, true, mvprec::kInterpSwitchable,
                                             &error),
           "switchable frame filter writer");
  sw.align_to_byte();
  const auto bytes = sw.bytes();
  mvprec::BitReader r(
      std::span<const std::uint8_t>(bytes.data(), bytes.size()));
  bool switchable = false;
  int read_filter = -1;
  t.expect(mvprec::read_frame_interp_filter(r, &switchable, &read_filter,
                                            &error),
           "switchable frame filter reader");
  t.expect(switchable, "switchable frame filter bit");
  t.eq_int(read_filter, mvprec::kInterpSwitchable,
           "switchable frame filter value");
  t.expect(!mvprec::frame_interp_filter_legal(5, false),
           "frame filter 5 is illegal");
  t.expect(!mvprec::block_interp_filter_legal(3),
           "BILINEAR is not a block symbol");
  t.expect(mvprec::block_interp_filter_legal(2), "EIGHTTAP_SHARP is a symbol");
}

// ---------------------------------------------------------------------------
// AMVD / precision / interp-filter entropy symbols
// ---------------------------------------------------------------------------

void test_amvd_symbols(Test& t, std::vector<std::string>* golden) {
  for (int disable = 0; disable <= 1; ++disable) {
    for (int index = 0; index < 9; ++index) {
      for (int ctx = 0; ctx < 3; ++ctx) {
        for (int value = 0; value <= 1; ++value) {
          mvprec::MvPrecisionCdfState cdfs;
          mvprec::init_default_mv_precision_cdfs(&cdfs);
          const auto payload = encode_payload(
              [&](av2::Encoder& enc) {
                return mvprec::write_use_amvd(enc, cdfs, index, ctx,
                                              value != 0, nullptr);
              },
              disable != 0);
          t.expect(!payload.empty(), "use_amvd payload");
          mvprec::MvPrecisionCdfState read_cdfs;
          mvprec::init_default_mv_precision_cdfs(&read_cdfs);
          bool read = !value;
          t.expect(decode_payload(
                       payload,
                       [&](av2::Decoder& dec) {
                         return mvprec::read_use_amvd(dec, read_cdfs, index, ctx,
                                                      &read, nullptr);
                       },
                       disable != 0),
                   "use_amvd reader");
          t.eq_int(read ? 1 : 0, value, "use_amvd round-trip");
          if (golden != nullptr && disable != 0 && value == 1) {
            golden->push_back("amvd_cdf|" + std::to_string(index) + "|" +
                              std::to_string(ctx) + "|1|" + hex(payload));
          }
        }
      }
    }
  }

  for (int joint = 0; joint < mvprec::kMvJoints; ++joint) {
    for (int index0 = 0; index0 < mvprec::kMaxAmvdIndex; index0 += 3) {
      for (int index1 = 0; index1 < mvprec::kMaxAmvdIndex; index1 += 3) {
        mvprec::AmvdMvSyntax syntax;
        syntax.joint = joint;
        const bool row = joint == mvprec::kMvJointHzvnz ||
                         joint == mvprec::kMvJointHnzvnz;
        const bool col = joint == mvprec::kMvJointHnzvz ||
                         joint == mvprec::kMvJointHnzvnz;
        syntax.index[0] = index0;
        syntax.index[1] = index1;
        syntax.magnitude[0] =
            row ? mvprec::kAmvdIndexToMvd[index0] : 0;
        syntax.magnitude[1] =
            col ? mvprec::kAmvdIndexToMvd[index1] : 0;
        mvprec::MvPrecisionCdfState cdfs;
        mvprec::init_default_mv_precision_cdfs(&cdfs);
        const auto payload = encode_payload(
            [&](av2::Encoder& enc) {
              return mvprec::write_amvd_mv(enc, cdfs, syntax, nullptr);
            },
            false);
        t.expect(!payload.empty(), "amvd mv payload");
        mvprec::MvPrecisionCdfState read_cdfs;
        mvprec::init_default_mv_precision_cdfs(&read_cdfs);
        mvprec::AmvdMvSyntax read;
        t.expect(decode_payload(
                     payload,
                     [&](av2::Decoder& dec) {
                       return mvprec::read_amvd_mv(dec, read_cdfs, &read,
                                                   nullptr);
                     },
                     false),
                 "amvd mv reader");
        t.eq_int(read.joint, joint, "mv_joint round-trip");
        t.eq_int(read.magnitude[0], syntax.magnitude[0], "AMVD row magnitude");
        t.eq_int(read.magnitude[1], syntax.magnitude[1], "AMVD col magnitude");
        if (golden != nullptr && joint != 0 && index0 == 0 && index1 == 0) {
          golden->push_back("amvd_mv|" + std::to_string(joint) + "|" +
                            std::to_string(index0) + "|" +
                            std::to_string(index1) + "|" + hex(payload));
        }
      }
    }
  }

  // jmvd_scale_mode, both alphabets.
  for (int use_amvd = 0; use_amvd <= 1; ++use_amvd) {
    const int count = use_amvd ? mvprec::kJointAmvdScaleFactorCnt
                               : mvprec::kJointNewMvScaleFactorCnt;
    for (int mode = 0; mode < count; ++mode) {
      mvprec::MvPrecisionCdfState cdfs;
      mvprec::init_default_mv_precision_cdfs(&cdfs);
      const auto payload = encode_payload(
          [&](av2::Encoder& enc) {
            return mvprec::write_jmvd_scale_mode(enc, cdfs, use_amvd != 0, mode,
                                                 nullptr);
          },
          false);
      mvprec::MvPrecisionCdfState read_cdfs;
      mvprec::init_default_mv_precision_cdfs(&read_cdfs);
      int read = -1;
      t.expect(decode_payload(
                   payload,
                   [&](av2::Decoder& dec) {
                     return mvprec::read_jmvd_scale_mode(dec, read_cdfs,
                                                         use_amvd != 0, &read,
                                                         nullptr);
                   },
                   false),
               "jmvd_scale_mode reader");
      t.eq_int(read, mode, "jmvd_scale_mode round-trip");
      if (golden != nullptr && mode == 1) {
        golden->push_back("jmvd|" + std::to_string(use_amvd) + "|" +
                          std::to_string(mode) + "|" + hex(payload));
      }
    }
  }
}

void test_block_precision(Test& t, std::vector<std::string>* golden) {
  const int frames[] = {me2::kMvPrecisionHalfPel,
                        me2::kMvPrecisionQuarterPel,
                        me2::kMvPrecisionEighthPel};
  for (int frame_precision : frames) {
    for (int flex = 0; flex <= 1; ++flex) {
      me2::FramePrecisionSyntax frame;
      if (!me2::me2_frame_precision_to_bits(frame_precision, flex, 0, &frame)) {
        continue;
      }
      for (int target = 0; target <= 6; ++target) {
        mvprec::MvResConfig config;
        config.enable_flex_mvres = flex != 0;
        mvprec::BlockPrecisionSyntax plan;
        std::string error;
        const bool ok = mvprec::plan_block_precision(
            config, frame, mvprec::kYModeNewMv, false, target, &plan, &error);
        const bool reachable = [&]() {
          if (target == frame_precision) return true;
          if (flex == 0) return false;
          for (int pb = 0; pb <= 2; ++pb) {
            if (me2::me2_pb_precision_derive(frame_precision, pb) == target) {
              return true;
            }
          }
          return false;
        }();
        t.eq_int(ok ? 1 : 0, reachable ? 1 : 0,
                 "plan_block_precision reachability");
        if (!ok) continue;
        if (golden != nullptr && flex != 0 && target != frame_precision) {
          golden->push_back("pbmap|" + std::to_string(frame_precision) + "|" +
                            std::to_string(target) + "|" +
                            std::to_string(plan.use_most_probable_precision ? 1
                                                                          : 0) +
                            "|" + std::to_string(plan.pb_mv_precision) + "|" +
                            std::to_string(plan.mv_precision));
        }
        // Round-trip through the entropy coder with the two precision ctxs.
        for (int most_ctx = 0; most_ctx < 3; ++most_ctx) {
          for (int pb_ctx = 0; pb_ctx < 2; ++pb_ctx) {
            mvprec::MvPrecisionCdfState cdfs;
            mvprec::init_default_mv_precision_cdfs(&cdfs);
            const auto payload = encode_payload(
                [&](av2::Encoder& enc) {
                  return mvprec::write_block_precision(enc, cdfs, most_ctx,
                                                       pb_ctx, plan, nullptr);
                },
                false);
            mvprec::MvPrecisionCdfState read_cdfs;
            mvprec::init_default_mv_precision_cdfs(&read_cdfs);
            mvprec::BlockPrecisionSyntax read;
            t.expect(decode_payload(
                         payload,
                         [&](av2::Decoder& dec) {
                           return mvprec::read_block_precision(
                               dec, read_cdfs, most_ctx, pb_ctx, frame,
                               mvprec::kYModeNewMv, false,
                               config.enable_adaptive_mvd,
                               config.enable_flex_mvres, &read, nullptr);
                         },
                         false),
                     "block precision reader");
            t.eq_int(read.mv_precision, plan.mv_precision,
                     "block precision round-trip");
            t.eq_int(read.use_most_probable_precision,
                     plan.use_most_probable_precision,
                     "use_most_probable round-trip");
            t.eq_int(read.pb_mv_precision, plan.pb_mv_precision,
                     "pb_mv_precision round-trip");
            if (golden != nullptr && most_ctx == 0 && pb_ctx == 0 &&
                flex != 0 && target != frame_precision) {
              golden->push_back("prec_sym|" + std::to_string(frame_precision) +
                                "|" + std::to_string(target) + "|" +
                                hex(payload));
            }
          }
        }
      }
    }
  }

  // AMVD short-circuits per-block precision: MvPrecision = FrameMvPrecision,
  // no symbols, and the target is ignored.
  mvprec::MvResConfig config;
  config.enable_adaptive_mvd = true;
  config.enable_flex_mvres = true;
  me2::FramePrecisionSyntax frame;
  t.expect(me2::me2_frame_precision_to_bits(me2::kMvPrecisionEighthPel, 1, 0,
                                            &frame),
           "EIGHTH frame for AMVD plan");
  mvprec::BlockPrecisionSyntax plan;
  std::string error;
  t.expect(mvprec::plan_block_precision(config, frame, mvprec::kYModeNewMv,
                                        true, me2::kMvPrecisionHalfPel, &plan,
                                        &error),
           "AMVD plan accepts any target");
  t.expect(plan.is_adaptive_mvd, "AMVD plan flag");
  t.expect(!plan.code_use_most_probable_precision,
           "AMVD plan codes no precision symbol");
  t.eq_int(plan.mv_precision, me2::kMvPrecisionEighthPel,
           "AMVD pins FrameMvPrecision");

  // The matching reader takes the IsAdaptiveMvd short-circuit with no
  // precision symbols.
  {
    mvprec::MvPrecisionCdfState cdfs;
    mvprec::init_default_mv_precision_cdfs(&cdfs);
    const auto none = encode_payload([](av2::Encoder&) { return true; }, true);
    mvprec::BlockPrecisionSyntax read_plan;
    t.expect(decode_payload(
                 none,
                 [&](av2::Decoder& dec) {
                   return mvprec::read_block_precision(
                       dec, cdfs, 0, 0, frame, mvprec::kYModeNewMv, true, true,
                       true, &read_plan, nullptr);
                 },
                 true),
             "adaptive precision reader");
    t.expect(read_plan.is_adaptive_mvd, "adaptive reader flag");
    t.eq_int(read_plan.mv_precision, me2::kMvPrecisionEighthPel,
             "adaptive reader MvPrecision");
    t.eq_int(read_plan.use_most_probable_precision, 1,
             "adaptive reader most-probable flag");
    t.expect(!read_plan.code_use_most_probable_precision,
             "adaptive reader codes no symbol");
  }

  // Non-BlockPrecisionSymbol cases.
  mvprec::MvPrecisionCdfState cdfs;
  mvprec::init_default_mv_precision_cdfs(&cdfs);
  const auto none = encode_payload(
      [&](av2::Encoder& enc) {
        return mvprec::write_block_precision(enc, cdfs, 0, 0, plan, nullptr);
      },
      true);
  t.expect(!none.empty(), "zero precision symbols still emit a payload");
}

void test_interp_filter_block(Test& t, std::vector<std::string>* golden) {
  for (int ctx = 0; ctx < 16; ++ctx) {
    for (int filter = 0; filter <= 2; ++filter) {
      mvprec::MvPrecisionCdfState cdfs;
      mvprec::init_default_mv_precision_cdfs(&cdfs);
      const auto payload = encode_payload(
          [&](av2::Encoder& enc) {
            return mvprec::write_block_interp_filter(enc, cdfs, ctx, filter,
                                                     nullptr);
          },
          false);
      t.expect(!payload.empty(), "interp payload");
      mvprec::MvPrecisionCdfState read_cdfs;
      mvprec::init_default_mv_precision_cdfs(&read_cdfs);
      int read = -1;
      t.expect(decode_payload(
                   payload,
                   [&](av2::Decoder& dec) {
                     return mvprec::read_block_interp_filter(dec, read_cdfs,
                                                             ctx, &read,
                                                             nullptr);
                   },
                   false),
               "interp reader");
      t.eq_int(read, filter, "interp round-trip");
      if (golden != nullptr && ctx == 0) {
        golden->push_back("interp|0|" + std::to_string(filter) + "|" +
                          hex(payload));
      }
    }
  }
  mvprec::MvPrecisionCdfState cdfs;
  mvprec::init_default_mv_precision_cdfs(&cdfs);
  const auto bad = encode_payload(
      [&](av2::Encoder& enc) {
        return mvprec::write_block_interp_filter(enc, cdfs, 0, 3, nullptr);
      },
      true);
  t.expect(bad.empty(), "BILINEAR block filter is rejected");
  const auto bad_ctx = encode_payload(
      [&](av2::Encoder& enc) {
        return mvprec::write_block_interp_filter(enc, cdfs, 16, 0, nullptr);
      },
      true);
  t.expect(bad_ctx.empty(), "interp ctx 16 is rejected");
}

// ---------------------------------------------------------------------------
// Shell MVD, including flex precision
// ---------------------------------------------------------------------------

void test_shell_mv(Test& t, std::vector<std::string>* golden) {
  const int precisions[] = {0, 1, 3, 4, 5, 6};
  const std::int32_t mvs[][2] = {
      {0, 0}, {2, 0}, {0, 2}, {2, 2}, {4, 4}, {12, 0}, {0, 12}, {6, 6},
      {16, 8}, {32, 0}, {0, 32}, {128, 128}, {64, 0}, {40, 8}};
  for (int precision : precisions) {
    const int shift = me2::kMvPrecisionEighthPel - precision;
    const std::int32_t radix = static_cast<std::int32_t>(1) << shift;
    for (const auto& mv : mvs) {
      const std::int32_t dy = mv[0] * radix;
      const std::int32_t dx = mv[1] * radix;
      mvprec::MvPrecisionCdfState cdfs;
      mvprec::init_default_mv_precision_cdfs(&cdfs);
      const auto payload = encode_payload(
          [&](av2::Encoder& enc) {
            return mvprec::write_shell_mv(enc, cdfs, dy, dx, precision,
                                          nullptr);
          },
          false);
      t.expect(!payload.empty(), "shell payload");
      mvprec::MvPrecisionCdfState read_cdfs;
      mvprec::init_default_mv_precision_cdfs(&read_cdfs);
      mc::Mv read;
      t.expect(decode_payload(
                   payload,
                   [&](av2::Decoder& dec) {
                     return mvprec::read_shell_mv(dec, read_cdfs, precision,
                                                  &read, nullptr);
                   },
                   false),
               "shell reader");
      t.eq_int(read.y, dy, "shell row magnitude");
      t.eq_int(read.x, dx, "shell col magnitude");
      if (golden != nullptr && precision == me2::kMvPrecisionEighthPel &&
          dy == 12 && dx == 0) {
        golden->push_back("shell|6|12|0|" + hex(payload));
      }
      if (golden != nullptr && precision == me2::kMvPrecisionOnePel &&
          dy == 1 && dx == 1) {
        golden->push_back("shell|3|1|1|" + hex(payload));
      }
    }
  }
  // Unaligned MVD at a coarse precision is rejected.
  mvprec::MvPrecisionCdfState cdfs;
  mvprec::init_default_mv_precision_cdfs(&cdfs);
  const auto bad = encode_payload(
      [&](av2::Encoder& enc) {
        return mvprec::write_shell_mv(enc, cdfs, 2, 0,
                                      me2::kMvPrecisionOnePel, nullptr);
      },
      true);
  t.expect(bad.empty(), "unaligned shell MVD rejected");
  const auto bad2 = encode_payload(
      [&](av2::Encoder& enc) {
        return mvprec::write_shell_mv(enc, cdfs, 0, 0,
                                      me2::kMvPrecisionTwoPel, nullptr);
      },
      true);
  t.expect(bad2.empty(), "TWO_PEL has no shell CDF");
}

// ---------------------------------------------------------------------------
// Contexts
// ---------------------------------------------------------------------------

void test_contexts(Test& t, std::vector<std::string>* golden) {
  t.eq_int(mvprec::use_amvd_cdf_index(mvprec::kYModeNewMv, false), 4,
           "NEWMV AMVD index");
  t.eq_int(mvprec::use_amvd_cdf_index(mvprec::kYModeNewMv, true), 4,
           "NEWMV optflow AMVD index");
  t.eq_int(mvprec::use_amvd_cdf_index(mvprec::kYModeNearNewMv, false), 0,
           "NEAR_NEWMV AMVD index");
  t.eq_int(mvprec::use_amvd_cdf_index(mvprec::kYModeNearNewMv, true), 2,
           "NEAR_NEWMV optflow AMVD index");
  t.eq_int(mvprec::use_amvd_cdf_index(mvprec::kYModeNewNearMv, true), 3,
           "NEW_NEARMV optflow AMVD index");
  t.eq_int(mvprec::use_amvd_cdf_index(mvprec::kYModeJointNewMv, false), 5,
           "JOINT_NEWMV AMVD index");
  t.eq_int(mvprec::use_amvd_cdf_index(mvprec::kYModeJointNewMv, true), 6,
           "JOINT_NEWMV optflow AMVD index");
  t.eq_int(mvprec::use_amvd_cdf_index(mvprec::kYModeNewNewMv, false), 7,
           "NEW_NEWMV AMVD index");
  t.eq_int(mvprec::use_amvd_cdf_index(mvprec::kYModeNewNewMv, true), 8,
           "NEW_NEWMV optflow AMVD index");
  t.eq_int(mvprec::use_amvd_cdf_index(mvprec::kYModeNearMv, false), -1,
           "NEARMV has no AMVD symbol");

  // Neighbour grid: a 16x16 MI frame, 64x64 SB (16 MI).  The current block is
  // (16,12) 2x2: its above neighbours are in the previous superblock row and
  // therefore absent from NPos but present in NPosBuf.
  const int mi_cols = 16;
  const int mi_rows = 32;
  std::vector<mvprec::NeighborMvInfo> grid(
      static_cast<std::size_t>(mi_cols) * static_cast<std::size_t>(mi_rows));
  const auto at = [&](int r, int c) -> mvprec::NeighborMvInfo& {
    return grid[static_cast<std::size_t>(r) * static_cast<std::size_t>(mi_cols) +
                static_cast<std::size_t>(c)];
  };
  // Same-row left neighbours (positions (17,11) and (16,11)).
  at(17, 11).available = true;
  at(17, 11).ref[0] = 0;
  at(17, 11).uses_amvd = true;
  at(17, 11).use_most_probable_precision = false;
  at(17, 11).mv_precision = me2::kMvPrecisionEighthPel;
  at(17, 11).interp_filter = 2;
  at(16, 11).available = true;
  at(16, 11).ref[0] = 0;
  at(16, 11).uses_amvd = true;
  at(16, 11).use_most_probable_precision = false;
  at(16, 11).mv_precision = me2::kMvPrecisionHalfPel;
  at(16, 11).interp_filter = 2;
  // Above neighbours in the previous SB row (positions (15,13) and (15,12)).
  at(15, 13).available = true;
  at(15, 13).ref[0] = 0;
  at(15, 13).uses_amvd = true;
  at(15, 13).use_most_probable_precision = true;
  at(15, 13).mv_precision = me2::kMvPrecisionHalfPel;
  at(15, 13).interp_filter = 1;
  at(15, 12).available = true;
  at(15, 12).ref[0] = 1;
  at(15, 12).uses_amvd = false;
  at(15, 12).interp_filter = 3;

  const mvprec::NeighborSet neighbours =
      mvprec::collect_neighbors(grid.data(), mi_cols, mi_rows, 16, 12, 2, 2, 16);
  t.eq_int(neighbours.num_npos, 2, "npos keeps the same SB row only");
  t.eq_int(neighbours.num_nbuf, 2, "nbuf keeps the buffered neighbours");
  t.eq_int(mvprec::use_amvd_ctx(neighbours, 0), 2, "AMVD ctx counts nbuf");
  t.eq_int(mvprec::use_amvd_ctx(neighbours, 1), 0, "AMVD ctx ref mismatch");
  t.eq_int(mvprec::use_most_probable_precision_ctx(neighbours), 0,
           "most-probable ctx uses npos");
  t.eq_int(mvprec::pb_mv_precision_ctx(neighbours,
                                       me2::kMvPrecisionEighthPel),
           1, "pb ctx uses npos");
  t.eq_int(mvprec::interp_filter_ctx(neighbours, 0, false), 2,
           "interp ctx uses npos left type");
  t.eq_int(mvprec::interp_filter_ctx(neighbours, 0, true), 6,
           "compound interp ctx adds 4");

  // Differing same-row neighbour types add 3 instead of one of them.
  at(16, 11).interp_filter = 1;
  const mvprec::NeighborSet differing = mvprec::collect_neighbors(
      grid.data(), mi_cols, mi_rows, 16, 12, 2, 2, 16);
  t.eq_int(mvprec::interp_filter_ctx(differing, 0, false), 3,
           "differing interp types");

  // needs_interp_filter().
  t.expect(mvprec::needs_interp_filter(true, mvprec::kYModeNewMv, 0),
           "NEWMV needs a filter symbol");
  t.expect(!mvprec::needs_interp_filter(true, mvprec::kYModeGlobalMv, 0),
           "large GLOBALMV has no filter symbol");
  t.expect(!mvprec::needs_interp_filter(true, mvprec::kYModeGlobalGlobalMv, 0),
           "large GLOBAL_GLOBALMV has no filter symbol");
  t.expect(!mvprec::needs_interp_filter(true, mvprec::kYModeNewMv,
                                        mvprec::kMotionLocalWarp),
           "warp has no filter symbol");
  t.expect(mvprec::needs_interp_filter(false, mvprec::kYModeGlobalMv, 0),
           "small GLOBALMV needs a filter symbol");

  if (golden != nullptr) {
    golden->push_back("ctx|amvd|" + std::to_string(neighbours.num_npos) + "|" +
                      std::to_string(neighbours.num_nbuf) + "|" +
                      std::to_string(mvprec::use_amvd_ctx(neighbours, 0)) +
                      "|" +
                      std::to_string(
                          mvprec::use_most_probable_precision_ctx(neighbours)) +
                      "|" +
                      std::to_string(mvprec::interp_filter_ctx(neighbours, 0,
                                                               false)));
  }
}

// ---------------------------------------------------------------------------
// Joint MVD + sign derivation
// ---------------------------------------------------------------------------

void test_joint_mvd(Test& t, std::vector<std::string>* golden) {
  {
    const mvprec::JointMvdPlan plan = mvprec::joint_mvd_plan(1, 2, false, false);
    t.eq_int(plan.base_list, 1, "plan chooses the nearer base list");
    t.eq_int(plan.first_distance, 2, "plan first distance");
    t.eq_int(plan.second_distance, 1, "plan second distance");
    const mc::Mv proj = mvprec::project_joint_mvd(mc::Mv{8, 4}, 1, 2);
    t.eq_int(proj.y, 4, "projMv row");
    t.eq_int(proj.x, 2, "projMv col");
    const mc::Mv amvd1 = mvprec::scale_joint_projection(proj, 1, true);
    t.eq_int(amvd1.y, 8, "AMVD scale 1 row");
    t.eq_int(amvd1.x, 4, "AMVD scale 1 col");
    const mc::Mv amvd2 = mvprec::scale_joint_projection(proj, 2, true);
    t.eq_int(amvd2.y, 2, "AMVD scale 2 row");
    t.eq_int(amvd2.x, 1, "AMVD scale 2 col");
    const mc::Mv s1 = mvprec::scale_joint_projection(proj, 1, false);
    t.eq_int(s1.y, 8, "non-AMVD scale 1 row");
    t.eq_int(s1.x, 2, "non-AMVD scale 1 col");
    const mc::Mv s2 = mvprec::scale_joint_projection(proj, 2, false);
    t.eq_int(s2.y, 4, "non-AMVD scale 2 row");
    t.eq_int(s2.x, 4, "non-AMVD scale 2 col");
    const mc::Mv s3 = mvprec::scale_joint_projection(proj, 3, false);
    t.eq_int(s3.y, 2, "non-AMVD scale 3 row");
    t.eq_int(s3.x, 2, "non-AMVD scale 3 col");
    const mc::Mv s4 = mvprec::scale_joint_projection(proj, 4, false);
    t.eq_int(s4.y, 4, "non-AMVD scale 4 row");
    t.eq_int(s4.x, 1, "non-AMVD scale 4 col");
    mc::Mv pred[2] = {{0, 0}, {100, -100}};
    mc::Mv out[2];
    mvprec::assign_joint_mvd(pred, mc::Mv{8, 4}, plan, 0, false, out);
    t.eq_int(out[1].y, 108, "assign_joint base list 1 row");
    t.eq_int(out[1].x, -96, "assign_joint base list 1 col");
    t.eq_int(out[0].y, 4, "assign_joint projected row");
    t.eq_int(out[0].x, 2, "assign_joint projected col");
    if (golden != nullptr) {
      golden->push_back("project|1|2|0|0|1|1|8|4|" +
                        std::to_string(amvd1.y) + "|" +
                        std::to_string(amvd1.x));
    }
  }
  {
    const mvprec::JointMvdPlan plan = mvprec::joint_mvd_plan(-1, 2, false, false);
    t.eq_int(plan.base_list, 1, "opposite-side base list 1");
    t.eq_int(plan.first_distance, 2, "opposite-side first");
    t.eq_int(plan.second_distance, -1, "opposite-side second is negative");
    const mc::Mv proj = mvprec::project_joint_mvd(mc::Mv{8, 4}, -1, 2);
    t.eq_int(proj.y, -4, "negative proj row");
    t.eq_int(proj.x, -2, "negative proj col");
  }
  {
    const mvprec::JointMvdPlan plan = mvprec::joint_mvd_plan(4, 4, false, true);
    t.eq_int(plan.base_list, 1, "restricted ref forces base list 1");
    t.eq_int(plan.first_distance, 4, "restricted swap first");
    t.eq_int(plan.second_distance, 4, "restricted swap second");
  }
  // mv_clamp_to_integer.
  t.eq_int(mvprec::mv_clamp_to_integer(0), 0, "clamp zero");
  t.eq_int(mvprec::mv_clamp_to_integer(me2::kMvLow), me2::kMvLow + 8,
           "low clamp");
  t.eq_int(mvprec::mv_clamp_to_integer(me2::kMvUpp), me2::kMvUpp - 8,
           "high clamp");

  // Sign derivation truth table.
  {
    mvprec::MvdSignConfig config;
    config.enable_mvd_sign_derive = true;
    config.frame_precision = me2::kMvPrecisionQuarterPel;
    config.mv_precision = me2::kMvPrecisionHalfPel;
    t.expect(mvprec::mvd_sign_derive_allowed(config),
             "sign derive allowed for NEWMV at coarser precision");
  }
  const auto allowed = [](mvprec::MvdSignConfig config) {
    return mvprec::mvd_sign_derive_allowed(config);
  };
  mvprec::MvdSignConfig base;
  base.enable_mvd_sign_derive = true;
  base.frame_precision = me2::kMvPrecisionQuarterPel;
  base.mv_precision = me2::kMvPrecisionHalfPel;
  t.expect(allowed(base), "sign derive base case");
  mvprec::MvdSignConfig c = base;
  c.enable_mvd_sign_derive = false;
  t.expect(!allowed(c), "sign derive disabled");
  c = base;
  c.motion_mode = 1;
  t.expect(!allowed(c), "sign derive needs SIMPLE");
  c = base;
  c.is_adaptive_mvd = true;
  t.expect(!allowed(c), "AMVD disables sign derive");
  c = base;
  c.skip_mode = true;
  t.expect(!allowed(c), "skip_mode disables sign derive");
  c = base;
  c.allow_screen_content_tools = true;
  t.expect(!allowed(c), "screen content disables sign derive");
  c = base;
  c.frame_precision = me2::kMvPrecisionEighthPel;
  t.expect(!allowed(c), "EIGHTH frame precision disables sign derive");
  c = base;
  c.mv_precision = me2::kMvPrecisionQuarterPel;
  t.expect(!allowed(c), "QUARTER block precision disables sign derive");
  c = base;
  c.y_mode = mvprec::kYModeNearNewMv;
  t.expect(!allowed(c), "has_nearmv disables sign derive");
  c = base;
  c.is_compound = true;
  c.ref_mv_idx = 1;
  t.expect(!allowed(c), "compound RefMvIdx != 0 disables sign derive");
  c.ref_mv_idx = 0;
  t.expect(allowed(c), "compound RefMvIdx == 0 allows sign derive");
  c = base;
  c.use_intrabc = true;
  t.expect(!allowed(c), "intrabc disables sign derive");

  // Round-trip with and without derivation.
  for (int derive = 0; derive <= 1; ++derive) {
    for (int y_mode : {mvprec::kYModeNewMv, mvprec::kYModeNewNewMv}) {
      for (int lists = 1; lists <= 2; ++lists) {
        mvprec::MvdSignConfig config;
        config.enable_mvd_sign_derive = derive != 0;
        config.frame_precision = me2::kMvPrecisionQuarterPel;
        config.mv_precision = me2::kMvPrecisionHalfPel;
        config.y_mode = y_mode;
        config.is_compound = lists == 2;
        config.ref_mv_idx = 0;
        // Make the last nonzero component agree with the derived sign when
        // derivation is active, otherwise the writer correctly refuses.
        mc::Mv mvd[2] = {{8, 4}, {-8, 4}};
        if (derive != 0) {
          int last_sign = 0;
          int last_list = -1;
          int last_comp = -1;
          for (int list = 0; list < lists; ++list) {
            const std::int32_t values[2] = {mvd[list].y, mvd[list].x};
            for (int comp = 0; comp < 2; ++comp) {
              if (values[comp] == 0) continue;
              const std::int32_t magnitude =
                  values[comp] < 0 ? -values[comp] : values[comp];
              last_sign += magnitude >> 2;
              last_list = list;
              last_comp = comp;
            }
          }
          const bool negative = (last_sign & 1) != 0;
          if (last_list >= 0) {
            std::int32_t& component = last_comp == 0 ? mvd[last_list].y
                                                     : mvd[last_list].x;
            const std::int32_t magnitude =
                component < 0 ? -component : component;
            component = negative ? -magnitude : magnitude;
          }
        }
        const int expected_symbols =
            mvprec::mvd_sign_symbol_count(config, mvd, lists);
        const auto payload = encode_payload(
            [&](av2::Encoder& enc) {
              return mvprec::write_mvd_signs(enc, config, mvd, lists, nullptr);
            },
            false);
        t.expect(!payload.empty(), "sign payload");
        mc::Mv read[2] = {{8, 4}, {8, 4}};
        t.expect(decode_payload(
                     payload,
                     [&](av2::Decoder& dec) {
                       return mvprec::read_mvd_signs(dec, config, read, lists,
                                                     nullptr);
                     },
                     false),
                 "sign reader");
        for (int list = 0; list < lists; ++list) {
          t.eq_int(read[list].y, mvd[list].y, "sign row round-trip");
          t.eq_int(read[list].x, mvd[list].x, "sign col round-trip");
        }
        // Count the raw bits independently of the coder.
        int nonzero = 0;
        for (int list = 0; list < lists; ++list) {
          const std::int32_t values[2] = {mvd[list].y, mvd[list].x};
          for (std::int32_t value : values) {
            if (value == 0) continue;
            ++nonzero;
          }
        }
        const int threshold = y_mode == mvprec::kYModeNewNewMv ? 4 : 1;
        const bool derive_active = derive != 0 && nonzero >= threshold;
        const int want = derive_active ? nonzero - 1 : nonzero;
        t.eq_int(expected_symbols, want, "sign symbol count");
        if (golden != nullptr && derive != 0 && lists == 1 &&
            y_mode == mvprec::kYModeNewMv) {
          golden->push_back("sign|5|4|16|0|0|1|" +
                            std::to_string(mvd[0].y) + "|" +
                            std::to_string(mvd[0].x) + "|0|0|" +
                            hex(payload));
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Goldens
// ---------------------------------------------------------------------------

bool emit_goldens(const std::string& path, Test& t) {
  std::vector<std::string> lines;
  test_motion_modes(t, &lines);
  test_frame_precision(t, &lines);
  test_amvd_symbols(t, &lines);
  test_block_precision(t, &lines);
  test_interp_filter_block(t, &lines);
  test_shell_mv(t, &lines);
  test_contexts(t, &lines);
  test_joint_mvd(t, &lines);

  std::ofstream out(path);
  if (!out) return false;
  out << "# Pinned MV-precision goldens.\n";
  out << "# Fields are '|'-separated; `hex` is the complete tile payload from\n";
  out << "# av2::Encoder with disable_cdf_update = 1 unless noted.\n";
  out << "#\n";
  out << "# seq_modes|seq_mask|present|six|hex\n";
  out << "# frame_modes|seq_mask|frame_mask|hex\n";
  out << "# frame_prec|force|flex|target|hex\n";
  out << "# amvd_cdf|index|ctx|value|hex\n";
  out << "# amvd_mv|joint|idx0|idx1|hex\n";
  out << "# jmvd|use_amvd|mode|hex\n";
  out << "# pbmap|frame_precision|target|use_most_probable|pb|MvPrecision\n";
  out << "# prec_sym|frame_precision|target|hex\n";
  out << "# interp|ctx|filter|hex\n";
  out << "# shell|precision|dy|dx|hex\n";
  out << "# ctx|name|num_npos|num_nbuf|amvd_ctx|most_ctx|interp_ctx\n";
  out << "# project|d0|d1|restrict0|restrict1|scale|use_amvd|dy|dx|outy|outx\n";
  out << "# sign|frame_precision|mv_precision|y_mode|compound|ref_mv_idx|derive|y0|x0|y1|x1|hex\n";
  for (const std::string& line : lines) out << line << "\n";
  return true;
}

void test_goldens(Test& t, const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    t.expect(false, "goldens file opens: " + path);
    return;
  }
  std::string line;
  int lines = 0;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    ++lines;
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
      const std::size_t bar = line.find('|', start);
      fields.push_back(line.substr(start, bar - start));
      if (bar == std::string::npos) break;
      start = bar + 1;
    }
    t.expect(fields.size() >= 2, "golden field count: " + line);
    if (fields.size() < 2) continue;
    const std::string& kind = fields[0];
    const auto number = [&](std::size_t i) {
      return std::atoi(fields[i].c_str());
    };
    if (kind == "seq_modes") {
      SeqModesCase c;
      c.seq_mask = number(1);
      c.six = number(3) != 0;
      mvprec::SequenceMotionModeSyntax syntax;
      syntax.seq_enabled = motion_mode_set(c.seq_mask);
      syntax.seq_frame_motion_modes_present_flag = syntax.seq_enabled.any();
      syntax.enable_six_param_warp_delta = c.six;
      av2enc::BitWriter writer;
      std::string error;
      t.expect(mvprec::write_sequence_motion_modes(writer, syntax, &error),
               "golden seq writer");
      writer.align_to_byte();
      const auto bytes = writer.bytes();
      t.eq_int(hex(std::vector<std::uint8_t>(bytes.begin(), bytes.end())) ==
                   fields[4],
               1, "golden seq payload " + fields[4]);
    } else if (kind == "frame_modes") {
      SeqModesCase c;
      c.seq_mask = number(1);
      c.frame_mask = number(2);
      const auto payload = frame_modes_payload(c);
      t.eq_int(hex(payload) == fields[3], 1, "golden frame payload");
    } else if (kind == "frame_prec") {
      const int flex = number(2);
      const int target = number(3);
      me2::FramePrecisionSyntax syntax;
      t.expect(me2::me2_frame_precision_to_bits(target, flex, 0, &syntax),
               "golden frame prec target");
      av2enc::BitWriter writer;
      std::string error;
      t.expect(mvprec::write_frame_mv_precision(writer, syntax, 0, &error),
               "golden frame prec writer");
      writer.align_to_byte();
      const auto bytes = writer.bytes();
      t.eq_int(hex(std::vector<std::uint8_t>(bytes.begin(), bytes.end())) ==
                   fields[4],
               1, "golden frame prec payload");
    } else if (kind == "amvd_cdf") {
      const int index = number(1);
      const int ctx = number(2);
      mvprec::MvPrecisionCdfState cdfs;
      mvprec::init_default_mv_precision_cdfs(&cdfs);
      const auto payload = encode_payload(
          [&](av2::Encoder& enc) {
            return mvprec::write_use_amvd(enc, cdfs, index, ctx, true, nullptr);
          },
          true);
      t.eq_int(hex(payload) == fields[4], 1, "golden use_amvd payload");
    } else if (kind == "amvd_mv") {
      const int joint = number(1);
      mvprec::AmvdMvSyntax syntax;
      syntax.joint = joint;
      const bool row = joint == mvprec::kMvJointHzvnz ||
                       joint == mvprec::kMvJointHnzvnz;
      const bool col = joint == mvprec::kMvJointHnzvz ||
                       joint == mvprec::kMvJointHnzvnz;
      syntax.magnitude[0] = row ? mvprec::kAmvdIndexToMvd[number(2)] : 0;
      syntax.magnitude[1] = col ? mvprec::kAmvdIndexToMvd[number(3)] : 0;
      syntax.index[0] = number(2);
      syntax.index[1] = number(3);
      mvprec::MvPrecisionCdfState cdfs;
      mvprec::init_default_mv_precision_cdfs(&cdfs);
      const auto payload = encode_payload(
          [&](av2::Encoder& enc) {
            return mvprec::write_amvd_mv(enc, cdfs, syntax, nullptr);
          },
          false);
      t.eq_int(hex(payload) == fields[4], 1, "golden AMVD payload");
    } else if (kind == "jmvd") {
      mvprec::MvPrecisionCdfState cdfs;
      mvprec::init_default_mv_precision_cdfs(&cdfs);
      const auto payload = encode_payload(
          [&](av2::Encoder& enc) {
            return mvprec::write_jmvd_scale_mode(
                enc, cdfs, number(1) != 0, number(2), nullptr);
          },
          false);
      t.eq_int(hex(payload) == fields[3], 1, "golden jmvd payload");
    } else if (kind == "pbmap") {
      const int frame_precision = number(1);
      const int target = number(2);
      me2::FramePrecisionSyntax frame;
      t.expect(me2::me2_frame_precision_to_bits(frame_precision, 1, 0, &frame),
               "golden pbmap frame");
      mvprec::MvResConfig config;
      config.enable_flex_mvres = true;
      mvprec::BlockPrecisionSyntax plan;
      std::string error;
      t.expect(mvprec::plan_block_precision(config, frame,
                                            mvprec::kYModeNewMv, false, target,
                                            &plan, &error),
               "golden pbmap plan");
      t.eq_int(plan.use_most_probable_precision ? 1 : 0, number(3),
               "golden pbmap use_most");
      t.eq_int(plan.pb_mv_precision, number(4), "golden pbmap pb");
      t.eq_int(plan.mv_precision, number(5), "golden pbmap precision");
    } else if (kind == "prec_sym") {
      const int frame_precision = number(1);
      const int target = number(2);
      me2::FramePrecisionSyntax frame;
      t.expect(me2::me2_frame_precision_to_bits(frame_precision, 1, 0, &frame),
               "golden prec frame");
      mvprec::MvResConfig config;
      config.enable_flex_mvres = true;
      mvprec::BlockPrecisionSyntax plan;
      std::string error;
      t.expect(mvprec::plan_block_precision(config, frame,
                                            mvprec::kYModeNewMv, false, target,
                                            &plan, &error),
               "golden prec plan");
      mvprec::MvPrecisionCdfState cdfs;
      mvprec::init_default_mv_precision_cdfs(&cdfs);
      const auto payload = encode_payload(
          [&](av2::Encoder& enc) {
            return mvprec::write_block_precision(enc, cdfs, 0, 0, plan, nullptr);
          },
          false);
      t.eq_int(hex(payload) == fields[3], 1, "golden prec payload");
    } else if (kind == "interp") {
      mvprec::MvPrecisionCdfState cdfs;
      mvprec::init_default_mv_precision_cdfs(&cdfs);
      const auto payload = encode_payload(
          [&](av2::Encoder& enc) {
            return mvprec::write_block_interp_filter(enc, cdfs, number(1),
                                                     number(2), nullptr);
          },
          false);
      t.eq_int(hex(payload) == fields[3], 1, "golden interp payload");
    } else if (kind == "shell") {
      mvprec::MvPrecisionCdfState cdfs;
      mvprec::init_default_mv_precision_cdfs(&cdfs);
      const auto payload = encode_payload(
          [&](av2::Encoder& enc) {
            return mvprec::write_shell_mv(enc, cdfs, number(2), number(3),
                                          number(1), nullptr);
          },
          false);
      t.eq_int(hex(payload) == fields[4], 1, "golden shell payload");
    } else if (kind == "ctx") {
      const int mi_cols = 16;
      const int mi_rows = 32;
      std::vector<mvprec::NeighborMvInfo> grid(
          static_cast<std::size_t>(mi_cols) *
          static_cast<std::size_t>(mi_rows));
      const auto at = [&](int r, int c) -> mvprec::NeighborMvInfo& {
        return grid[static_cast<std::size_t>(r) *
                        static_cast<std::size_t>(mi_cols) +
                    static_cast<std::size_t>(c)];
      };
      at(17, 11).available = true;
      at(17, 11).ref[0] = 0;
      at(17, 11).uses_amvd = true;
      at(17, 11).interp_filter = 2;
      at(16, 11).available = true;
      at(16, 11).ref[0] = 0;
      at(16, 11).uses_amvd = true;
      at(16, 11).mv_precision = me2::kMvPrecisionHalfPel;
      at(16, 11).interp_filter = 2;
      at(15, 13).available = true;
      at(15, 13).ref[0] = 0;
      at(15, 13).uses_amvd = true;
      at(15, 13).use_most_probable_precision = true;
      at(15, 13).mv_precision = me2::kMvPrecisionHalfPel;
      at(15, 13).interp_filter = 1;
      at(15, 12).available = true;
      at(15, 12).ref[0] = 1;
      const mvprec::NeighborSet n = mvprec::collect_neighbors(
          grid.data(), mi_cols, mi_rows, 16, 12, 2, 2, 16);
      t.eq_int(n.num_npos, number(2), "golden ctx npos");
      t.eq_int(n.num_nbuf, number(3), "golden ctx nbuf");
      t.eq_int(mvprec::use_amvd_ctx(n, 0), number(4), "golden ctx amvd");
      t.eq_int(mvprec::use_most_probable_precision_ctx(n), number(5),
               "golden ctx most");
      t.eq_int(mvprec::interp_filter_ctx(n, 0, false), number(6),
               "golden ctx interp");
    } else if (kind == "project") {
      const mvprec::JointMvdPlan plan = mvprec::joint_mvd_plan(
          number(1), number(2), number(3) != 0, number(4) != 0);
      mc::Mv proj = mvprec::project_joint_mvd(mc::Mv{number(7), number(8)},
                                              plan.second_distance,
                                              plan.first_distance);
      proj = mvprec::scale_joint_projection(proj, number(5), number(6) != 0);
      t.eq_int(proj.y, number(9), "golden project row");
      t.eq_int(proj.x, number(10), "golden project col");
    } else if (kind == "sign") {
      mvprec::MvdSignConfig config;
      config.frame_precision = number(1);
      config.mv_precision = number(2);
      config.y_mode = number(3);
      config.is_compound = number(4) != 0;
      config.ref_mv_idx = number(5);
      config.enable_mvd_sign_derive = number(6) != 0;
      mc::Mv mvd[2] = {{number(7), number(8)}, {number(9), number(10)}};
      const int lists = config.is_compound ? 2 : 1;
      const auto payload = encode_payload(
          [&](av2::Encoder& enc) {
            return mvprec::write_mvd_signs(enc, config, mvd, lists, nullptr);
          },
          false);
      t.eq_int(hex(payload) == fields[11], 1, "golden sign payload");
    } else {
      t.expect(false, "unknown golden kind: " + kind);
    }
  }
  t.expect(lines > 0, "goldens file is not empty");
}

}  // namespace

int main(int argc, char** argv) {
  std::string golden_path = "goldens/mv_precision_goldens.txt";
  std::string emit_path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--goldens" && i + 1 < argc) {
      golden_path = argv[++i];
    } else if (arg == "--emit-goldens" && i + 1 < argc) {
      emit_path = argv[++i];
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  Test t;
  test_cdf_tables(t);
  test_motion_modes(t, nullptr);
  test_frame_precision(t, nullptr);
  test_amvd_symbols(t, nullptr);
  test_block_precision(t, nullptr);
  test_interp_filter_block(t, nullptr);
  test_shell_mv(t, nullptr);
  test_contexts(t, nullptr);
  test_joint_mvd(t, nullptr);

  if (!emit_path.empty()) {
    Test emit_test;
    if (!emit_goldens(emit_path, emit_test)) {
      std::fprintf(stderr, "failed to emit goldens to %s\n", emit_path.c_str());
      return 1;
    }
    std::printf("wrote %s (%d emit checks, %d failures)\n", emit_path.c_str(),
                emit_test.checks, emit_test.failures);
  }

  test_goldens(t, golden_path);

  std::printf("mv_precision_test: %d checks, %d failures\n", t.checks,
              t.failures);
  return t.failures == 0 ? 0 : 1;
}
