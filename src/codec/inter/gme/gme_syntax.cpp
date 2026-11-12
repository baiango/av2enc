#include "gme_syntax.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace av2enc::inter::gme {
namespace {

constexpr int kParamShift[6] = {kGmTransPrecDiff, kGmTransPrecDiff, kGmAlphaPrecDiff,
                                kGmAlphaPrecDiff, kGmAlphaPrecDiff, kGmAlphaPrecDiff};
constexpr int kParamMin[6] = {kGmTransMin, kGmTransMin, kGmAlphaMin,
                              kGmAlphaMin, kGmAlphaMin, kGmAlphaMin};
constexpr int kParamMax[6] = {kGmTransMax, kGmTransMax, kGmAlphaMax,
                              kGmAlphaMax, kGmAlphaMax, kGmAlphaMax};

unsigned ceil_log2_u32(std::uint32_t value) {
  if (value <= 1U) return 0U;
  unsigned bits = 0;
  std::uint32_t v = value - 1U;
  while (v != 0U) {
    v >>= 1U;
    ++bits;
  }
  return bits;
}

void put_ns(BitWriter& writer, std::uint32_t n, std::uint32_t value) {
  if (n <= 1U) return;  // ns(1) codes nothing; value must be 0.
  const unsigned w = ceil_log2_u32(n);
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

bool fail(std::string* error, const std::string& text) {
  if (error != nullptr) *error = text;
  return false;
}

int inverse_recenter(int r, int v) {
  if (v > 2 * r) {
    return v;
  }
  if ((v & 1) != 0) {
    return r - ((v + 1) >> 1);
  }
  return r + (v >> 1);
}

int recenter_inverse(int r, int x) {
  if (x > 2 * r) return x;
  if (x > r) return 2 * (x - r);
  return 2 * (r - x) - (x == r ? 0 : 1);
}

bool write_subexp(BitWriter& writer, int num_syms, int k, int value) {
  if (num_syms <= 0 || value < 0 || value >= num_syms) return false;
  int i = 0;
  int mk = 0;
  while (true) {
    const int b2 = i != 0 ? k + i - 1 : k;
    const int a = 1 << b2;
    if (num_syms <= mk + 3 * a) {
      const int n = num_syms - mk;
      put_ns(writer, static_cast<std::uint32_t>(n),
             static_cast<std::uint32_t>(value - mk));
      return true;
    }
    if (value >= mk + a) {
      writer.put_bit(true);
      ++i;
      mk += a;
    } else {
      writer.put_bit(false);
      writer.put_bits(static_cast<std::uint64_t>(value - mk),
                      static_cast<unsigned>(b2));
      return true;
    }
  }
}

bool read_subexp(GmBitReader& reader, int num_syms, int k, int* value) {
  if (num_syms <= 0) return false;
  int i = 0;
  int mk = 0;
  while (true) {
    const int b2 = i != 0 ? k + i - 1 : k;
    const int a = 1 << b2;
    if (num_syms <= mk + 3 * a) {
      const int n = num_syms - mk;
      std::uint32_t payload = 0;
      if (!reader.read_ns(static_cast<std::uint32_t>(n), &payload)) return false;
      *value = static_cast<int>(payload) + mk;
      return true;
    }
    bool more = false;
    if (!reader.read_bit(&more)) return false;
    if (more) {
      ++i;
      mk += a;
    } else {
      std::uint32_t payload = 0;
      if (!reader.read_bits(static_cast<unsigned>(b2), &payload)) return false;
      *value = static_cast<int>(payload) + mk;
      return true;
    }
  }
}

bool write_unsigned_subexp_with_ref(BitWriter& writer, int num_syms, int r, int k, int x) {
  if (num_syms <= 0) return false;
  int v = 0;
  if ((r << 1) <= num_syms) {
    v = recenter_inverse(r, x);
  } else {
    const int z = num_syms - 1 - x;
    v = recenter_inverse(num_syms - 1 - r, z);
  }
  return write_subexp(writer, num_syms, k, v);
}

bool read_unsigned_subexp_with_ref(GmBitReader& reader, int num_syms, int r, int k, int* value) {
  int v = 0;
  if (!read_subexp(reader, num_syms, k, &v)) return false;
  if ((r << 1) <= num_syms) {
    *value = inverse_recenter(r, v);
  } else {
    *value = num_syms - 1 - inverse_recenter(num_syms - 1 - r, v);
  }
  return true;
}

bool write_signed_subexp_with_ref(BitWriter& writer, int low, int high, int r, int k, int value) {
  if (value < low || value >= high) return false;
  return write_unsigned_subexp_with_ref(writer, high - low, r - low, k, value - low);
}

bool read_signed_subexp_with_ref(GmBitReader& reader, int low, int high, int r, int k,
                                 int* value) {
  int x = 0;
  if (!read_unsigned_subexp_with_ref(reader, high - low, r - low, k, &x)) return false;
  *value = x + low;
  return true;
}

int precision_bits(int index) { return index < 2 ? kGmTransPrecBits : kGmAlphaPrecBits; }

int precision_max(int index) { return index < 2 ? kGmTransMax : kGmAlphaMax; }

}  // namespace

void write_sequence_enable_global_motion(BitWriter& writer, bool enable) {
  writer.put_bit(enable);
}

bool GmBitReader::read_bit(bool* bit) {
  if (bit == nullptr || bit_pos_ >= bytes_.size() * 8U) return false;
  const std::size_t byte_index = bit_pos_ >> 3U;
  const unsigned offset = 7U - static_cast<unsigned>(bit_pos_ & 7U);
  *bit = ((bytes_[byte_index] >> offset) & 1U) != 0U;
  ++bit_pos_;
  return true;
}

bool GmBitReader::read_bits(unsigned num_bits, std::uint32_t* value) {
  if (value == nullptr || num_bits > 32U) return false;
  std::uint32_t out = 0;
  for (unsigned i = 0; i < num_bits; ++i) {
    bool bit = false;
    if (!read_bit(&bit)) return false;
    out = (out << 1U) | (bit ? 1U : 0U);
  }
  *value = out;
  return true;
}

bool GmBitReader::read_ns(std::uint32_t n, std::uint32_t* value) {
  if (value == nullptr || n < 1U) return false;
  if (n == 1U) {
    *value = 0;
    return true;
  }
  const unsigned w = ceil_log2_u32(n);
  const std::uint32_t m = (1U << w) - n;
  std::uint32_t v = 0;
  if (!read_bits(w - 1U, &v)) return false;
  if (v < m) {
    *value = v;
    return true;
  }
  bool extra = false;
  if (!read_bit(&extra)) return false;
  *value = (v << 1U) + (extra ? 1U : 0U) - m;
  return *value < n;
}

void scale_warp_model(const std::int32_t base[6], int base_distance, int distance,
                      std::int32_t out[6]) {
  for (int i = 0; i < 6; ++i) out[i] = kDefaultWarpParams[i];
  if (base_distance == 0) return;
  if (base_distance < 0) {
    base_distance = -base_distance;
    distance = -distance;
  }
  const DivScaleShift ds = resolve_divisor(base_distance);
  constexpr int kLimit = (1 << 22) - 1;
  for (int i = 0; i < 6; ++i) {
    const int center = kDefaultWarpParams[i];
    const std::int64_t diff = static_cast<std::int64_t>(base[i]) - center;
    const std::int64_t clipped =
        diff < -static_cast<std::int64_t>(kLimit)
            ? -static_cast<std::int64_t>(kLimit)
            : (diff > static_cast<std::int64_t>(kLimit) ? kLimit : diff);
    const int input = static_cast<int>(clipped);
    const std::int64_t scaled = round2_signed64(
        static_cast<std::int64_t>(input) * ds.factor, ds.shift);
    const std::int64_t rounded = round2_signed64(scaled * distance, kParamShift[i]);
    const int output = clip3(kParamMin[i], kParamMax[i], static_cast<int>(rounded));
    out[i] = static_cast<std::int32_t>(
        center + (static_cast<std::int64_t>(output) *
                  (static_cast<std::int64_t>(1) << kParamShift[i])));
  }
}

bool write_global_param(BitWriter& writer, int /*ref*/, int index, const std::int32_t prev[6],
                        const GmParams& params, std::string* error) {
  if (index < 0 || index >= 6) return fail(error, "global param index out of range");
  const int prec_bits = precision_bits(index);
  const int mx = precision_max(index);
  const int prec_diff = kWarpedModelPrecBits - prec_bits;
  const int round = (index % 3 == 2) ? (1 << kWarpedModelPrecBits) : 0;
  const int sub = (index % 3 == 2) ? (1 << prec_bits) : 0;
  const std::int64_t shifted =
      (static_cast<std::int64_t>(params.p[static_cast<std::size_t>(index)]) - round) >>
      prec_diff;
  const std::int64_t step = static_cast<std::int64_t>(1) << prec_diff;
  if (shifted * step + round != params.p[static_cast<std::size_t>(index)]) {
    return fail(error, "global param is not on the syntax quantisation grid");
  }
  if (shifted < -static_cast<std::int64_t>(mx) - 1 ||
      shifted > static_cast<std::int64_t>(mx) + 1) {
    return fail(error, "global param outside the syntax range");
  }
  const int r = (prev[index] >> prec_diff) - sub;
  if (!write_signed_subexp_with_ref(writer, -mx, mx + 1, r, 3,
                                    static_cast<int>(shifted))) {
    return fail(error, "global param subexp encode failed");
  }
  return true;
}

bool read_global_param(GmBitReader& reader, int /*ref*/, int index, const std::int32_t prev[6],
                       std::int32_t out[6], std::string* error) {
  if (index < 0 || index >= 6) return fail(error, "global param index out of range");
  const int prec_bits = precision_bits(index);
  const int mx = precision_max(index);
  const int prec_diff = kWarpedModelPrecBits - prec_bits;
  const int round = (index % 3 == 2) ? (1 << kWarpedModelPrecBits) : 0;
  const int sub = (index % 3 == 2) ? (1 << prec_bits) : 0;
  const int r = (prev[index] >> prec_diff) - sub;
  int value = 0;
  if (!read_signed_subexp_with_ref(reader, -mx, mx + 1, r, 3, &value)) {
    return fail(error, "global param subexp decode failed");
  }
  const std::int64_t decoded =
      static_cast<std::int64_t>(value) *
          (static_cast<std::int64_t>(1) << prec_diff) +
      round;
  if (decoded < std::numeric_limits<std::int32_t>::min() ||
      decoded > std::numeric_limits<std::int32_t>::max()) {
    return fail(error, "decoded global param exceeds int32");
  }
  out[index] = static_cast<std::int32_t>(decoded);
  return true;
}

std::size_t write_global_motion_params(BitWriter& writer, const GmSyntaxContext& context,
                                       std::string* error) {
  const std::size_t start = writer.bit_size();
  auto fail_write = [&](const std::string& text) {
    fail(error, text);
    return writer.bit_size() - start;
  };

  if (context.frame_is_intra || !context.enable_global_motion) return 0;

  writer.put_bit(context.use_global_motion);
  if (!context.use_global_motion) return writer.bit_size() - start;

  if (context.num_total_refs < 1 || context.num_total_refs > kMaxRefsPerFrame) {
    return fail_write("global motion: invalid num_total_refs");
  }

  std::int32_t base_params[6];
  for (int i = 0; i < 6; ++i) base_params[i] = kDefaultWarpParams[i];
  int base_distance = 1;
  int our_ref = context.num_total_refs;
  if (!context.switch_frame) {
    our_ref = context.our_ref >= 0 ? context.our_ref : context.num_total_refs;
    if (our_ref > context.num_total_refs) {
      return fail_write("global motion: our_ref out of range");
    }
    put_ns(writer, static_cast<std::uint32_t>(context.num_total_refs + 1),
           static_cast<std::uint32_t>(our_ref));
  } else if (context.our_ref >= 0 && context.our_ref != context.num_total_refs) {
    return fail_write("global motion: SWITCH_FRAME forces our_ref = NumTotalRefs");
  }
  if (our_ref != context.num_total_refs) {
    const int ref_idx = context.ref_frame_idx[static_cast<std::size_t>(our_ref)];
    if (ref_idx < 0 || ref_idx >= kNumRefFrames) {
      return fail_write("global motion: base reference slot out of range");
    }
    const int n = context.history.ref_num_total_refs[static_cast<std::size_t>(ref_idx)];
    if (n > 0) {
      if (context.their_ref < 0 || context.their_ref >= n) {
        return fail_write("global motion: their_ref out of range");
      }
      put_ns(writer, static_cast<std::uint32_t>(n),
             static_cast<std::uint32_t>(context.their_ref));
      for (int i = 0; i < 6; ++i) {
        base_params[i] = static_cast<std::int32_t>(
            context.history.saved_gm_params[static_cast<std::size_t>(ref_idx)]
                                           [static_cast<std::size_t>(context.their_ref)]
                                           [static_cast<std::size_t>(i)]);
      }
      base_distance = get_relative_dist(
          context.ref_order_hint[static_cast<std::size_t>(our_ref)],
          context.history.saved_order_hints[static_cast<std::size_t>(ref_idx)]
                                          [static_cast<std::size_t>(context.their_ref)]);
    }
  }

  for (int ref = 0; ref < context.num_total_refs; ++ref) {
    const int ref_order_hint = context.ref_order_hint[static_cast<std::size_t>(ref)];
    const int dist = get_relative_dist(context.order_hint, ref_order_hint);
    const GmParams& target = context.params[static_cast<std::size_t>(ref)];
    if (dist == 0 || ref_order_hint == kRestrictedOrderHint) {
      if (!is_identity_model(target)) {
        return fail_write("global motion: non-identity model for zero/restricted distance");
      }
      continue;
    }
    std::int32_t prev[6];
    scale_warp_model(base_params, base_distance, dist, prev);

    if (target.type == GmType::kIdentity) {
      writer.put_bit(false);
      continue;
    }
    writer.put_bit(true);
    const bool rot_zoom = target.type == GmType::kRotZoom;
    writer.put_bit(rot_zoom);
    if (!rot_zoom && target.type != GmType::kAffine) {
      return fail_write("global motion: unsupported GmType");
    }
    if (rot_zoom) {
      if (target.p[4] != -target.p[3] || target.p[5] != target.p[2]) {
        return fail_write("global motion: ROTZOOM model is not symmetric");
      }
    }
    if (!write_global_param(writer, ref, 2, prev, target, error)) return writer.bit_size() - start;
    if (!write_global_param(writer, ref, 3, prev, target, error)) return writer.bit_size() - start;
    if (!rot_zoom) {
      if (!write_global_param(writer, ref, 4, prev, target, error)) {
        return writer.bit_size() - start;
      }
      if (!write_global_param(writer, ref, 5, prev, target, error)) {
        return writer.bit_size() - start;
      }
    }
    if (!write_global_param(writer, ref, 0, prev, target, error)) return writer.bit_size() - start;
    if (!write_global_param(writer, ref, 1, prev, target, error)) return writer.bit_size() - start;
  }
  return writer.bit_size() - start;
}

bool read_global_motion_params(GmBitReader& reader, const GmSyntaxContext& context,
                               std::array<GmParams, kMaxRefsPerFrame>* params_out,
                               std::string* error, bool* use_global_motion_out) {
  if (params_out == nullptr) return fail(error, "global motion: null output");
  for (int ref = 0; ref < kMaxRefsPerFrame; ++ref) {
    (*params_out)[static_cast<std::size_t>(ref)] = make_identity_gm();
  }
  if (use_global_motion_out != nullptr) *use_global_motion_out = false;
  if (context.frame_is_intra || !context.enable_global_motion) return true;

  bool use_global = false;
  if (!reader.read_bit(&use_global)) return fail(error, "global motion: truncated use bit");
  if (use_global_motion_out != nullptr) *use_global_motion_out = use_global;
  if (!use_global) return true;

  if (context.num_total_refs < 1 || context.num_total_refs > kMaxRefsPerFrame) {
    return fail(error, "global motion: invalid num_total_refs");
  }

  std::int32_t base_params[6];
  for (int i = 0; i < 6; ++i) base_params[i] = kDefaultWarpParams[i];
  int base_distance = 1;
  int our_ref = context.num_total_refs;
  if (!context.switch_frame) {
    std::uint32_t value = 0;
    if (!reader.read_ns(static_cast<std::uint32_t>(context.num_total_refs + 1), &value)) {
      return fail(error, "global motion: truncated our_ref");
    }
    our_ref = static_cast<int>(value);
  }
  if (our_ref != context.num_total_refs) {
    if (our_ref < 0 || our_ref >= kMaxRefsPerFrame) {
      return fail(error, "global motion: our_ref out of range");
    }
    const int ref_idx = context.ref_frame_idx[static_cast<std::size_t>(our_ref)];
    if (ref_idx < 0 || ref_idx >= kNumRefFrames) {
      return fail(error, "global motion: base reference slot out of range");
    }
    const int n = context.history.ref_num_total_refs[static_cast<std::size_t>(ref_idx)];
    if (n > 0) {
      std::uint32_t their_ref = 0;
      if (!reader.read_ns(static_cast<std::uint32_t>(n), &their_ref)) {
        return fail(error, "global motion: truncated their_ref");
      }
      if (static_cast<int>(their_ref) >= n) {
        return fail(error, "global motion: their_ref out of range");
      }
      for (int i = 0; i < 6; ++i) {
        base_params[i] = static_cast<std::int32_t>(
            context.history.saved_gm_params[static_cast<std::size_t>(ref_idx)]
                                           [their_ref][static_cast<std::size_t>(i)]);
      }
      base_distance = get_relative_dist(
          context.ref_order_hint[static_cast<std::size_t>(our_ref)],
          context.history.saved_order_hints[static_cast<std::size_t>(ref_idx)][their_ref]);
    }
  }

  for (int ref = 0; ref < context.num_total_refs; ++ref) {
    const int ref_order_hint = context.ref_order_hint[static_cast<std::size_t>(ref)];
    const int dist = get_relative_dist(context.order_hint, ref_order_hint);
    if (dist == 0 || ref_order_hint == kRestrictedOrderHint) {
      (*params_out)[static_cast<std::size_t>(ref)] = make_identity_gm();
      continue;
    }
    std::int32_t prev[6];
    scale_warp_model(base_params, base_distance, dist, prev);
    bool is_global = false;
    if (!reader.read_bit(&is_global)) return fail(error, "global motion: truncated is_global");
    if (!is_global) {
      (*params_out)[static_cast<std::size_t>(ref)] = make_identity_gm();
      continue;
    }
    bool is_rot_zoom = false;
    if (!reader.read_bit(&is_rot_zoom)) {
      return fail(error, "global motion: truncated is_rot_zoom");
    }
    GmParams decoded = make_identity_gm();
    decoded.type = is_rot_zoom ? GmType::kRotZoom : GmType::kAffine;
    std::int32_t values[6];
    for (int i = 0; i < 6; ++i) values[i] = kDefaultWarpParams[i];
    if (!read_global_param(reader, ref, 2, prev, values, error)) return false;
    if (!read_global_param(reader, ref, 3, prev, values, error)) return false;
    if (!is_rot_zoom) {
      if (!read_global_param(reader, ref, 4, prev, values, error)) return false;
      if (!read_global_param(reader, ref, 5, prev, values, error)) return false;
    } else {
      values[4] = -values[3];
      values[5] = values[2];
    }
    if (!read_global_param(reader, ref, 0, prev, values, error)) return false;
    if (!read_global_param(reader, ref, 1, prev, values, error)) return false;
    for (int i = 0; i < 6; ++i) decoded.p[static_cast<std::size_t>(i)] = values[i];
    (*params_out)[static_cast<std::size_t>(ref)] = decoded;
  }
  return true;
}

}  // namespace av2enc::inter::gme
