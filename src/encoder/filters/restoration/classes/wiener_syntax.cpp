#include "encoder/filters/restoration/classes/wiener_syntax.h"

#include "encoder/filters/restoration/classes/wiener_cdfs.h"
#include "encoder/filters/restoration/wiener_core.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace av2enc::restoration::classes {
namespace {

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr && error->empty()) {
    *error = message;
  }
  return false;
}

int ceil_log2(int v) {
  int r = 0;
  int p = 1;
  while (p < v) {
    p <<= 1;
    ++r;
  }
  return r;
}

void zero_coeffs(std::int16_t* coeffs) {
  for (int i = 0; i < kWienerNsCoeffStride; ++i) {
    coeffs[i] = 0;
  }
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

int unsigned_subexp_with_ref_value(int mx, int r, int k, int v) {
  (void)k;
  if ((r << 1) <= mx) {
    return inverse_recenter(r, v);
  }
  return mx - 1 - inverse_recenter(mx - 1 - r, v);
}

bool encode_subexp(BitWriter& writer, int num_syms, int k, int value,
                   std::string* error) {
  if (num_syms <= 0 || value < 0 || value >= num_syms) {
    return fail(error, "subexp value out of range");
  }
  int i = 0;
  int mk = 0;
  for (;;) {
    const int b2 = (i != 0) ? k + i - 1 : k;
    if (b2 < 0 || b2 > 30) {
      return fail(error, "subexp bit width out of range");
    }
    const int a = 1 << b2;
    if (num_syms <= mk + 3 * a) {
      const int n = num_syms - mk;
      if (value < mk || value >= num_syms) {
        return fail(error, "subexp final value out of range");
      }
      if (!wiener_put_ns(writer, static_cast<unsigned>(n),
                         static_cast<std::uint32_t>(value - mk), error)) {
        return false;
      }
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

bool decode_subexp(WienerBitReader& reader, int num_syms, int k, int* value,
                   std::string* error) {
  if (value == nullptr || num_syms <= 0) {
    return fail(error, "invalid subexp argument");
  }
  int i = 0;
  int mk = 0;
  for (;;) {
    const int b2 = (i != 0) ? k + i - 1 : k;
    if (b2 < 0 || b2 > 30) {
      return fail(error, "subexp bit width out of range");
    }
    const int a = 1 << b2;
    if (num_syms <= mk + 3 * a) {
      const int n = num_syms - mk;
      std::uint32_t bits = 0;
      if (!reader.read_ns(static_cast<unsigned>(n), &bits)) {
        return fail(error, "truncated subexp final bits");
      }
      *value = mk + static_cast<int>(bits);
      return true;
    }
    bool more = false;
    if (!reader.read_bit(&more)) {
      return fail(error, "truncated subexp more bit");
    }
    if (more) {
      ++i;
      mk += a;
    } else {
      std::uint32_t bits = 0;
      if (!reader.read_bits(static_cast<unsigned>(b2), &bits)) {
        return fail(error, "truncated subexp bits");
      }
      *value = mk + static_cast<int>(bits);
      return true;
    }
  }
}

bool decode_4part_value(int num, int base, int rem, int* value, int* bits_out) {
  if (base < 0 || base > 3) {
    return false;
  }
  const int bits = 2 - num + std::max(1, base);
  if (bits < 1 || bits > 15) {
    return false;
  }
  const int offset = (base == 0) ? 0 : (1 << bits);
  if (rem < 0 || rem >= (1 << bits)) {
    return false;
  }
  if (value != nullptr) {
    *value = offset + rem;
  }
  if (bits_out != nullptr) {
    *bits_out = bits;
  }
  return true;
}

int unsigned_4part_value(int k, int r, int v) {
  const int mx = 1 << k;
  if ((r << 1) <= mx) {
    return inverse_recenter(r, v);
  }
  return mx - 1 - inverse_recenter(mx - 1 - r, v);
}

bool raw_bit(av2::Encoder& encoder, bool bit, std::string* error) {
  const av2::Status status = encoder.encode_bool(bit);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("raw bit encode failed: ") +
                           av2::status_name(status));
  }
  return true;
}

bool raw_bit(av2::Decoder& decoder, bool* bit, std::string* error) {
  const av2::Status status = decoder.read_bool(bit);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("raw bit decode failed: ") +
                           av2::status_name(status));
  }
  return true;
}

void copy_cdf(std::uint16_t* dst, const std::uint16_t* src, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = src[i];
  }
}

template <std::size_t N>
void copy_cdf(std::uint16_t (&dst)[N], const std::uint16_t (&src)[N]) {
  copy_cdf(dst, src, static_cast<int>(N));
}

bool write_lr_size_flags(BitWriter& writer, int shift, int sb_size,
                         std::string* error) {
  if (shift < 0 || shift > 3) {
    return fail(error, "LoopRestorationSize shift outside 0..3");
  }
  if (sb_size == 256) {
    if (shift != 0 && shift != 1) {
      return fail(error, "256x256 SB supports shifts 0 and 1");
    }
    writer.put_bit(shift == 1);
    return true;
  }
  writer.put_bit(shift == 1);
  if (shift == 1) {
    return true;
  }
  writer.put_bit(shift == 0);
  if (shift == 0) {
    return true;
  }
  if (sb_size == 128) {
    if (shift != 2) {
      return fail(error, "128x128 SB supports shifts 0, 1 and 2");
    }
    return true;
  }
  if (sb_size != 64) {
    return fail(error, "unsupported superblock size");
  }
  if (shift != 2 && shift != 3) {
    return fail(error, "64x64 SB supports shifts 0..3");
  }
  writer.put_bit(shift == 2);
  return true;
}

bool read_lr_size_flags(WienerBitReader& reader, int sb_size, int* shift,
                        std::string* error) {
  bool half = false;
  if (!reader.read_bit(&half)) {
    return fail(error, "truncated lr size flag");
  }
  if (half) {
    *shift = 1;
    return true;
  }
  if (sb_size == 256) {
    *shift = 0;
    return true;
  }
  bool max_size = false;
  if (!reader.read_bit(&max_size)) {
    return fail(error, "truncated lr max size flag");
  }
  if (max_size) {
    *shift = 0;
    return true;
  }
  if (sb_size == 128) {
    *shift = 2;
    return true;
  }
  if (sb_size != 64) {
    return fail(error, "unsupported superblock size");
  }
  bool quarter = false;
  if (!reader.read_bit(&quarter)) {
    return fail(error, "truncated lr quarter size flag");
  }
  *shift = quarter ? 2 : 3;
  return true;
}

// Resolves match index m to the filter the decoder will use.
void resolve_match_filter(int plane, int num_classes, int m,
                          const WienerLrHeaderParams& params,
                          const WienerReferenceSet& refs,
                          const WienerSearchResult& search,
                          std::int16_t out[kWienerNsCoeffStride]) {
  zero_coeffs(out);
  if (m == 0) {
    return;
  }
  if (m < num_classes) {
    const std::int16_t* src = params.frame_coeffs[plane][m - 1];
    for (int j = 0; j < kWienerNsCoeffStride; ++j) {
      out[j] = src[j];
    }
    return;
  }
  const int ref_index = m - num_classes;
  if (ref_index < 0 ||
      ref_index >= static_cast<int>(search.ref_filters.size())) {
    return;
  }
  const WienerRefFilterEntry& e =
      search.ref_filters[static_cast<std::size_t>(ref_index)];
  const std::int16_t* src =
      refs.ref[e.ref_idx].coeffs[e.ref_plane][e.ref_class];
  for (int j = 0; j < kWienerNsCoeffStride; ++j) {
    out[j] = src[j];
  }
}

// Forward declaration of the shared group/count setup.
struct GroupSetup {
  int num_classes = 1;
  int num_ref_filters = 0;
  int group_counts[3] = {1, 0, 0};
  int group_base[3] = {0, 0, 0};
  int group_hits[3] = {0, 0, 0};
  WienerSearchResult search;
};

bool prepare_groups(int plane, const WienerLrHeaderParams& params,
                    const WienerReferenceSet& refs, bool nopcw,
                    GroupSetup* out, std::string* error) {
  if (out == nullptr) {
    return fail(error, "null group setup");
  }
  out->num_classes = (plane == 0) ? params.num_filter_classes : 1;
  if (out->num_classes < 1 || out->num_classes > kWienerNsClasses) {
    return fail(error, "NumFilterClasses outside 1..16");
  }
  if (out->num_classes != 1 &&
      !valid_num_filter_classes(out->num_classes)) {
    return fail(error, "NumFilterClasses is not in Decode_Num_Filter_Classes");
  }
  out->search =
      wiener_search_frame_filters(plane, -1, params, refs, nopcw);
  out->num_ref_filters = out->search.num_ref_filters;
  out->group_counts[0] = out->num_classes;
  out->group_counts[1] = out->num_ref_filters;
  out->group_counts[2] = 0;  // This encoder keeps PC-Wiener disabled.
  out->group_base[0] = 0;
  for (int i = 1; i < 3; ++i) {
    out->group_base[i] = out->group_base[i - 1] + out->group_counts[i - 1];
  }
  for (int i = 0; i < 3; ++i) {
    out->group_hits[i] = 0;
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Bit reader / descriptors
// ---------------------------------------------------------------------------

bool WienerBitReader::read_bit(bool* bit) {
  if (bit == nullptr || data_ == nullptr || bit_pos_ >= bit_size_) {
    return false;
  }
  const std::size_t byte = bit_pos_ >> 3;
  const unsigned shift = 7U - static_cast<unsigned>(bit_pos_ & 7U);
  *bit = ((data_[byte] >> shift) & 1U) != 0U;
  ++bit_pos_;
  return true;
}

bool WienerBitReader::read_bits(unsigned n, std::uint32_t* value) {
  if (value == nullptr || n > 32U) {
    return false;
  }
  std::uint32_t v = 0;
  for (unsigned i = 0; i < n; ++i) {
    bool b = false;
    if (!read_bit(&b)) {
      return false;
    }
    v = (v << 1U) | (b ? 1U : 0U);
  }
  *value = v;
  return true;
}

bool WienerBitReader::read_ns(unsigned n, std::uint32_t* value) {
  if (value == nullptr || n == 0U) {
    return false;
  }
  if (n == 1U) {
    *value = 0;
    return true;
  }
  unsigned w = 0;
  while ((1U << w) < n) {
    ++w;
  }
  const std::uint32_t m = (1U << w) - n;
  std::uint32_t v = 0;
  if (!read_bits(w - 1U, &v)) {
    return false;
  }
  if (v < m) {
    *value = v;
    return true;
  }
  bool extra = false;
  if (!read_bit(&extra)) {
    return false;
  }
  *value = (v << 1U) - m + (extra ? 1U : 0U);
  return *value < n;
}

bool wiener_put_ns(BitWriter& writer, unsigned n, std::uint32_t value,
                   std::string* error) {
  if (n == 0U || value >= n) {
    return fail(error, "ns() value out of range");
  }
  if (n == 1U) {
    return true;
  }
  unsigned w = 0;
  while ((1U << w) < n) {
    ++w;
  }
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
    return true;
  }
  const std::uint32_t y = value + m;
  const std::uint32_t v = y >> 1U;
  const bool extra = (y & 1U) != 0U;
  writer.put_bits(v, w - 1U);
  writer.put_bit(extra);
  return true;
}

bool wiener_encode_signed_subexp_with_ref(BitWriter& writer, int low,
                                          int high, int r, int k,
                                          int target, std::string* error) {
  if (high <= low || target < low || target >= high) {
    return fail(error, "signed subexp target out of range");
  }
  const int mx = high - low;
  const int r_off = r - low;
  const int x_off = target - low;
  for (int v = 0; v < mx; ++v) {
    if (unsigned_subexp_with_ref_value(mx, r_off, k, v) == x_off) {
      return encode_subexp(writer, mx, k, v, error);
    }
  }
  return fail(error, "signed subexp has no encoding (bad reference)");
}

bool wiener_decode_signed_subexp_with_ref(WienerBitReader& reader, int low,
                                          int high, int r, int k,
                                          int* target, std::string* error) {
  if (target == nullptr || high <= low) {
    return fail(error, "invalid signed subexp argument");
  }
  int v = 0;
  if (!decode_subexp(reader, high - low, k, &v, error)) {
    return false;
  }
  *target = low + unsigned_subexp_with_ref_value(high - low, r - low, k, v);
  return true;
}

bool wiener_encode_signed_4part(av2::Encoder& encoder,
                                std::uint16_t* base_cdf, int low, int k,
                                int r, int target, std::string* error) {
  if (base_cdf == nullptr || k < 0 || k > 15) {
    return fail(error, "invalid 4part argument");
  }
  const int r_off = r - low;
  const int x_off = target - low;
  int chosen_base = 0;
  int chosen_rem = 0;
  int chosen_bits = 0;
  bool found = false;
  for (int base = 0; base < 4 && !found; ++base) {
    const int bits = 2 - (6 - k) + std::max(1, base);
    if (bits < 1 || bits > 15) {
      continue;
    }
    for (int rem = 0; rem < (1 << bits); ++rem) {
      int value = 0;
      if (!decode_4part_value(6 - k, base, rem, &value, nullptr)) {
        continue;
      }
      if (unsigned_4part_value(k, r_off, value) == x_off) {
        chosen_base = base;
        chosen_rem = rem;
        chosen_bits = bits;
        found = true;
        break;
      }
    }
  }
  if (!found) {
    return fail(error, "signed 4part has no encoding");
  }
  const av2::Status status =
      encoder.encode_symbol(base_cdf, 4, chosen_base);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("wiener_ns_base encode failed: ") +
                           av2::status_name(status));
  }
  for (int i = chosen_bits - 1; i >= 0; --i) {
    if (!raw_bit(encoder, ((chosen_rem >> i) & 1) != 0, error)) {
      return false;
    }
  }
  return true;
}

bool wiener_decode_signed_4part(av2::Decoder& decoder,
                                std::uint16_t* base_cdf, int low, int k,
                                int r, int* target, std::string* error) {
  if (base_cdf == nullptr || target == nullptr) {
    return fail(error, "invalid 4part argument");
  }
  int base = 0;
  av2::Status status = decoder.decode_symbol(base_cdf, 4, &base);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("wiener_ns_base decode failed: ") +
                           av2::status_name(status));
  }
  const int bits = 2 - (6 - k) + std::max(1, base);
  if (bits < 1 || bits > 15) {
    return fail(error, "invalid wiener_ns_base");
  }
  std::uint32_t rem = 0;
  status = decoder.read_literal(bits, &rem);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("wiener_ns_rem decode failed: ") +
                           av2::status_name(status));
  }
  int value = 0;
  if (!decode_4part_value(6 - k, base, static_cast<int>(rem), &value,
                          nullptr)) {
    return fail(error, "invalid wiener_ns_base/rem combination");
  }
  *target = low + unsigned_4part_value(k, r - low, value);
  return true;
}

// ---------------------------------------------------------------------------
// search_frame_filters
// ---------------------------------------------------------------------------

WienerSearchResult wiener_search_frame_filters(
    const int plane, const int target, const WienerLrHeaderParams& params,
    const WienerReferenceSet& refs, const bool nopcw) {
  WienerSearchResult out;
  out.num_classes = (plane == 0) ? params.num_filter_classes : 1;
  const int min_pc = (plane > 0 || nopcw) ? 0 : 16;
  const int max_ref =
      (nopcw ? 16 : 64) - out.num_classes - min_pc;
  int num_ref = 0;
  const int num_check_planes = (plane > 0) ? 2 : 1;
  const int num_total_refs =
      std::min(refs.num_total_refs, kMaxLrRefs);
  for (int ref = 0; ref < num_total_refs; ++ref) {
    if (refs.is_switch_frame ||
        refs.order_hint[ref] == kRestrictedOrderHint) {
      continue;
    }
    const int idx = refs.ref_frame_idx[ref];
    if (idx < 0 || idx >= kMaxLrRefs) {
      continue;
    }
    for (int check = 0; check < num_check_planes; ++check) {
      const int check_plane =
          (check == 0) ? plane : ((plane == 1) ? 2 : 1);
      if (!refs.ref[idx].filters_on[check_plane]) {
        continue;
      }
      const int num_ref_classes =
          (plane == 0) ? refs.ref[idx].num_filter_classes : 1;
      if (num_ref_classes < 1 || num_ref_classes > kWienerNsClasses) {
        continue;
      }
      for (int i = 0; i < num_ref_classes; ++i) {
        if (num_ref < max_ref) {
          if (num_ref + out.num_classes == target) {
            out.match_idx = idx;
            out.match_cls = i;
            out.match_plane = check_plane;
          }
          WienerRefFilterEntry entry;
          entry.ref_idx = idx;
          entry.ref_plane = check_plane;
          entry.ref_class = i;
          out.ref_filters.push_back(entry);
          ++num_ref;
        }
      }
    }
  }
  out.num_ref_filters = num_ref;
  return out;
}

// ---------------------------------------------------------------------------
// Frame filters (§5.20.10.6 with readFrameFilters = 1)
// ---------------------------------------------------------------------------

namespace {

bool write_frame_filters_impl(BitWriter& writer,
                              const WienerLrHeaderConfig& config,
                              WienerLrHeaderParams* params, int plane,
                              const WienerReferenceSet& refs,
                              std::string* error) {
  const bool nopcw = config.lr_tools_disable[0][kRestorePcWiener];
  if (!nopcw) {
    return fail(error,
                "frame-level PC-Wiener translation needs Pc_Wiener_Filters, "
                "which is not in the spec text");
  }
  GroupSetup g;
  if (!prepare_groups(plane, *params, refs, nopcw, &g, error)) {
    return false;
  }

  const int num_classes = g.num_classes;
  for (int c = 0; c < num_classes; ++c) {
    g.group_counts[0] = c + 1;
    const int pred_group =
        (c == 0) ? ((g.group_counts[1] > 2)
                        ? 1
                        : wiener_predict_group(g.group_counts))
                 : wiener_predict_group(g.group_hits);
    const int m = params->frame_filter[plane][c].match_index;
    if (m < 0 || m >= g.num_classes + g.num_ref_filters) {
      return fail(error, "frame filter match_index outside the match range");
    }
    int chosen_group = (m < g.num_classes) ? 0 : 1;
    if (m < g.num_classes && m > c) {
      return fail(error, "frame filter refers to a later class");
    }
    int num_zeros = 0;
    int alt_group = 0;
    for (int i = 0; i < 3; ++i) {
      if (i != pred_group) {
        if (g.group_counts[i] == 0) {
          ++num_zeros;
        } else {
          alt_group = i;
        }
      }
    }
    if (num_zeros != 2) {
      if (chosen_group == pred_group) {
        writer.put_bit(false);  // use_alt_group
      } else {
        writer.put_bit(true);
        if (num_zeros == 1) {
          if (chosen_group != alt_group) {
            return fail(error, "frame filter group is not reachable");
          }
        } else {
          int group_bit = -1;
          for (int gb = 0; gb <= 1; ++gb) {
            const int mapped =
                (pred_group <= gb) ? gb + 1 : gb;
            if (mapped == chosen_group) {
              group_bit = gb;
              break;
            }
          }
          if (group_bit < 0) {
            return fail(error, "frame filter group has no group_bit");
          }
          writer.put_bit(group_bit != 0);
        }
      }
    } else if (chosen_group != pred_group) {
      return fail(error, "frame filter group is not reachable (numZeros==2)");
    }
    const int n = g.group_counts[chosen_group];
    const int base = g.group_base[chosen_group];
    if (n <= 0) {
      return fail(error, "empty frame filter group");
    }
    if (n == 1) {
      if (m != base) {
        return fail(error, "single-entry frame filter group mismatch");
      }
    } else {
      const int ref = base + (n >> 1);
      if (!wiener_encode_signed_subexp_with_ref(writer, base, base + n, ref,
                                                4, m, error)) {
        return false;
      }
    }
    ++g.group_hits[chosen_group];
  }

  const int n_coeffs = (plane == 0) ? kWienerNsLumaCoeffs
                                    : kWienerNsChromaCoeffs;
  const int num_subsets = wiener_num_subsets(plane);
  const bool chroma = plane > 0;

  // §5.20.10.6: the merged flags for *all* classes are signalled before any
  // class's subset/coefficient data (the second and third loops in the spec).
  for (int c = 0; c < num_classes; ++c) {
    const WienerFrameFilterSyntax& f = params->frame_filter[plane][c];
    if (f.match_index < 0 ||
        f.match_index >= num_classes + g.num_ref_filters) {
      return fail(error, "frame filter match_index out of range");
    }
    writer.put_bit(f.merged);
  }

  for (int c = 0; c < num_classes; ++c) {
    WienerFrameFilterSyntax& f = params->frame_filter[plane][c];
    std::int16_t ref[kWienerNsCoeffStride];
    resolve_match_filter(plane, num_classes, f.match_index, *params, refs,
                         g.search, ref);
    std::int16_t final_coeffs[kWienerNsCoeffStride];
    zero_coeffs(final_coeffs);
    if (f.merged) {
      for (int j = 0; j < kWienerNsCoeffStride; ++j) {
        final_coeffs[j] = ref[j];
      }
    } else {
      if (f.subset < 0 || f.subset >= num_subsets) {
        return fail(error, "frame filter subset out of range");
      }
      if (chroma && f.subset > 0 && f.uv_sym) {
        for (int j = kWienerNsShortCoeffs; j + 1 < n_coeffs; j += 2) {
          if (f.coeffs[j] != f.coeffs[j + 1]) {
            return fail(error, "wiener_ns_uv_sym requires symmetric pairs");
          }
        }
      } else if (!(chroma && f.subset > 0)) {
        f.uv_sym = false;
      }
      for (int s = 0; s < num_subsets - 1; ++s) {
        const bool len = s < f.subset;
        writer.put_bit(len);
        if (!len) {
          break;
        }
      }
      if (chroma && f.subset > 0) {
        writer.put_bit(f.uv_sym);
      }
      for (int j = 0; j < n_coeffs; ++j) {
        const int min = kWienerNsTapsMin[chroma ? 1 : 0][j];
        const int k = kWienerNsTapsK[chroma ? 1 : 0][j];
        std::int16_t v = ref[j];
        if (wiener_tap_present(plane, f.subset, j)) {
          const int target = f.coeffs[j];
          if (target < min || target >= min + (1 << k)) {
            return fail(error,
                        "frame Wiener coefficient outside its tap range");
          }
          if (!wiener_encode_signed_subexp_with_ref(
                  writer, min, min + (1 << k), ref[j], k - 3, target,
                  error)) {
            return false;
          }
          v = static_cast<std::int16_t>(target);
        } else {
          v = 0;
        }
        final_coeffs[j] = v;
        if (chroma && f.subset > 0 && f.uv_sym &&
            j >= kWienerNsShortCoeffs) {
          final_coeffs[j + 1] = v;
          ++j;
        }
      }
      for (int j = n_coeffs; j < kWienerNsCoeffStride; ++j) {
        final_coeffs[j] = 0;
      }
    }
    for (int j = 0; j < kWienerNsCoeffStride; ++j) {
      params->frame_coeffs[plane][c][j] = final_coeffs[j];
      f.coeffs[j] = final_coeffs[j];
    }
  }
  return true;
}

bool read_frame_filters_impl(WienerBitReader& reader,
                             const WienerLrHeaderConfig& config,
                             WienerLrHeaderParams* params, int plane,
                             const WienerReferenceSet& refs,
                             std::string* error) {
  const bool nopcw = config.lr_tools_disable[0][kRestorePcWiener];
  if (!nopcw) {
    return fail(error,
                "frame-level PC-Wiener translation needs Pc_Wiener_Filters, "
                "which is not in the spec text");
  }
  GroupSetup g;
  if (!prepare_groups(plane, *params, refs, nopcw, &g, error)) {
    return false;
  }
  const int num_classes = g.num_classes;
  for (int c = 0; c < num_classes; ++c) {
    g.group_counts[0] = c + 1;
    const int pred_group =
        (c == 0) ? ((g.group_counts[1] > 2)
                        ? 1
                        : wiener_predict_group(g.group_counts))
                 : wiener_predict_group(g.group_hits);
    int num_zeros = 0;
    int alt_group = 0;
    for (int i = 0; i < 3; ++i) {
      if (i != pred_group) {
        if (g.group_counts[i] == 0) {
          ++num_zeros;
        } else {
          alt_group = i;
        }
      }
    }
    bool use_alt_group = false;
    if (num_zeros != 2) {
      if (!reader.read_bit(&use_alt_group)) {
        return fail(error, "truncated use_alt_group");
      }
    }
    int chosen_group = pred_group;
    if (use_alt_group) {
      if (num_zeros == 1) {
        chosen_group = alt_group;
      } else {
        bool group_bit = false;
        if (!reader.read_bit(&group_bit)) {
          return fail(error, "truncated group_bit");
        }
        const int gb = group_bit ? 1 : 0;
        chosen_group = (pred_group <= gb) ? gb + 1 : gb;
      }
    }
    if (chosen_group < 0 || chosen_group > 2) {
      return fail(error, "frame filter group outside 0..2");
    }
    const int n = g.group_counts[chosen_group];
    const int base = g.group_base[chosen_group];
    if (n <= 0) {
      return fail(error, "empty frame filter group");
    }
    int m = base;
    if (n > 1) {
      const int ref = base + (n >> 1);
      if (!wiener_decode_signed_subexp_with_ref(reader, base, base + n, ref,
                                                4, &m, error)) {
        return false;
      }
    }
    params->frame_filter[plane][c].match_index = m;
    ++g.group_hits[chosen_group];
  }

  const int n_coeffs = (plane == 0) ? kWienerNsLumaCoeffs
                                    : kWienerNsChromaCoeffs;
  const int num_subsets = wiener_num_subsets(plane);
  const bool chroma = plane > 0;

  // §5.20.10.6 second loop: merged flags for every class first.
  for (int c = 0; c < num_classes; ++c) {
    WienerFrameFilterSyntax& f = params->frame_filter[plane][c];
    if (f.match_index < 0 ||
        f.match_index >= num_classes + g.num_ref_filters) {
      return fail(error, "frame filter match_index out of range");
    }
    if (!reader.read_bit(&f.merged)) {
      return fail(error, "truncated merged_param");
    }
  }

  for (int c = 0; c < num_classes; ++c) {
    WienerFrameFilterSyntax& f = params->frame_filter[plane][c];
    const bool merged = f.merged;
    std::int16_t ref[kWienerNsCoeffStride];
    resolve_match_filter(plane, num_classes, f.match_index, *params, refs,
                         g.search, ref);
    std::int16_t final_coeffs[kWienerNsCoeffStride];
    zero_coeffs(final_coeffs);
    if (merged) {
      for (int j = 0; j < kWienerNsCoeffStride; ++j) {
        final_coeffs[j] = ref[j];
      }
      f.subset = 0;
      f.uv_sym = false;
    } else {
      int subset = 0;
      for (int s = 0; s < num_subsets - 1; ++s) {
        bool len = false;
        if (!reader.read_bit(&len)) {
          return fail(error, "truncated wiener_ns_length");
        }
        if (!len) {
          break;
        }
        ++subset;
      }
      f.subset = subset;
      f.uv_sym = false;
      if (chroma && subset > 0) {
        if (!reader.read_bit(&f.uv_sym)) {
          return fail(error, "truncated wiener_ns_uv_sym");
        }
      }
      for (int j = 0; j < n_coeffs; ++j) {
        const int min = kWienerNsTapsMin[chroma ? 1 : 0][j];
        const int k = kWienerNsTapsK[chroma ? 1 : 0][j];
        std::int16_t v = ref[j];
        if (wiener_tap_present(plane, subset, j)) {
          int target = 0;
          if (!wiener_decode_signed_subexp_with_ref(
                  reader, min, min + (1 << k), ref[j], k - 3, &target,
                  error)) {
            return false;
          }
          v = static_cast<std::int16_t>(target);
        } else {
          v = 0;
        }
        final_coeffs[j] = v;
        if (chroma && subset > 0 && f.uv_sym &&
            j >= kWienerNsShortCoeffs) {
          final_coeffs[j + 1] = v;
          ++j;
        }
      }
      for (int j = n_coeffs; j < kWienerNsCoeffStride; ++j) {
        final_coeffs[j] = 0;
      }
    }
    for (int j = 0; j < kWienerNsCoeffStride; ++j) {
      params->frame_coeffs[plane][c][j] = final_coeffs[j];
      f.coeffs[j] = final_coeffs[j];
    }
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// lr_params()
// ---------------------------------------------------------------------------

bool wiener_write_lr_params(BitWriter& writer,
                            const WienerLrHeaderConfig& config,
                            WienerLrHeaderParams* params,
                            const WienerReferenceSet& refs,
                            std::string* error) {
  if (params == nullptr) {
    return fail(error, "null LrParams");
  }
  if (config.coded_lossless || !config.enable_restoration) {
    for (int p = 0; p < 3; ++p) {
      params->frame_type[p] = kRestoreNone;
      params->frame_filters_on[p] = false;
      params->temporal_pred_flag[p] = false;
    }
    return true;
  }
  if (!config.lr_tools_disable[0][kRestorePcWiener]) {
    return fail(error,
                "encoder cannot write PC-Wiener: Pc_Wiener_* tables are not "
                "in the spec text");
  }
  if (params->num_filter_classes != 1 &&
      !valid_num_filter_classes(params->num_filter_classes)) {
    return fail(error, "NumFilterClasses is not in Decode_Num_Filter_Classes");
  }
  params->num_filter_classes_idx =
      encode_num_filter_classes_idx(params->num_filter_classes);
  if (params->num_filter_classes_idx < 0) {
    return fail(error, "cannot encode NumFilterClasses");
  }

  bool uses_luma_lr = false;
  bool uses_chroma_lr = false;
  for (int plane = 0; plane < config.num_planes; ++plane) {
    const bool chroma = plane > 0;
    int index_to_tool[4] = {kRestoreNone, 0, 0, 0};
    int tools_count = 1;
    for (int i = 1; i < kRestoreSwitchable; ++i) {
      if (!config.lr_tools_disable[chroma ? 1 : 0][i]) {
        index_to_tool[tools_count++] = i;
      }
    }
    index_to_tool[tools_count] = kRestoreSwitchable;
    const bool allow_switchable = tools_count > 2;
    const int n = tools_count + (allow_switchable ? 1 : 0);
    const int target = params->frame_type[plane];
    int tool_index = -1;
    for (int i = 0; i < n; ++i) {
      if (index_to_tool[i] == target) {
        tool_index = i;
        break;
      }
    }
    if (tool_index < 0) {
      return fail(error, "FrameRestorationType is not signallable");
    }
    if (!wiener_put_ns(writer, static_cast<unsigned>(n),
                       static_cast<std::uint32_t>(tool_index), error)) {
      return false;
    }
    if (target != kRestoreNone) {
      if (plane == 0) {
        uses_luma_lr = true;
      } else {
        uses_chroma_lr = true;
      }
    }
    if (target == kRestorePcWiener) {
      return fail(error,
                  "PC-Wiener is not implementable without the Pc_Wiener_* "
                  "tables");
    }
    if (target == kRestoreWienerNs || target == kRestoreSwitchable) {
      writer.put_bit(params->frame_filters_on[plane]);
      if (params->frame_filters_on[plane]) {
        const int num_ref_frames =
            (refs.frame_is_intra || config.is_switch_frame)
                ? 0
                : config.num_total_refs;
        if (num_ref_frames > kMaxLrRefs) {
          return fail(error, "NumTotalRefs exceeds the reference array");
        }
        if (num_ref_frames > 0) {
          writer.put_bit(params->temporal_pred_flag[plane]);
        } else {
          params->temporal_pred_flag[plane] = false;
        }
        if (params->temporal_pred_flag[plane] && num_ref_frames > 1) {
          const int bits = ceil_log2(num_ref_frames);
          if (params->rst_ref_pic_idx[plane] < 0 ||
              params->rst_ref_pic_idx[plane] >= num_ref_frames) {
            return fail(error, "rst_ref_pic_idx outside NumTotalRefs");
          }
          writer.put_bits(
              static_cast<std::uint64_t>(params->rst_ref_pic_idx[plane]),
              static_cast<unsigned>(bits));
        } else {
          params->rst_ref_pic_idx[plane] = 0;
        }
        if (params->temporal_pred_flag[plane]) {
          const int rst = params->rst_ref_pic_idx[plane];
          if (rst < 0 || rst >= num_ref_frames || rst >= kMaxLrRefs) {
            return fail(error, "rst_ref_pic_idx outside the reference set");
          }
          const int ref_idx = refs.ref_frame_idx[rst];
          if (ref_idx < 0 || ref_idx >= kMaxLrRefs) {
            return fail(error, "ref_frame_idx outside NUM_REF_FRAMES");
          }
          int ref_plane = plane;
          if (plane > 0 && !refs.ref[ref_idx].filters_on[plane]) {
            ref_plane = (plane == 1) ? 2 : 1;
          }
          if (!refs.ref[ref_idx].filters_on[ref_plane]) {
            return fail(error,
                        "temporal prediction references a frame without "
                        "frame filters on the required plane");
          }
          if (plane == 0) {
            params->num_filter_classes =
                refs.ref[ref_idx].num_filter_classes;
            params->num_filter_classes_idx =
                encode_num_filter_classes_idx(params->num_filter_classes);
            if (params->num_filter_classes_idx < 0) {
              return fail(error, "reference NumFilterClasses is invalid");
            }
          }
          for (int c = 0; c < kWienerNsClasses; ++c) {
            for (int j = 0; j < kWienerNsCoeffStride; ++j) {
              params->frame_coeffs[plane][c][j] =
                  refs.ref[ref_idx].coeffs[ref_plane][c][j];
            }
          }
        } else if (plane == 0) {
          writer.put_bits(
              static_cast<std::uint64_t>(params->num_filter_classes_idx), 3U);
        }
      }
    } else {
      params->frame_filters_on[plane] = false;
      params->temporal_pred_flag[plane] = false;
    }
  }

  if (uses_luma_lr &&
      !write_lr_size_flags(writer, params->luma_shift, config.sb_size,
                           error)) {
    return false;
  }
  if (uses_chroma_lr &&
      !write_lr_size_flags(writer, params->chroma_shift, config.sb_size,
                           error)) {
    return false;
  }

  if (!params->frame_filters_on[0]) {
    // §5.18.7.11 initialises NumFilterClasses = 1 for the luma plane and only
    // changes it when luma frame filters are present.
    params->num_filter_classes = 1;
    params->num_filter_classes_idx = 0;
  }

  for (int plane = 0; plane < config.num_planes; ++plane) {
    if (!params->frame_filters_on[plane] ||
        params->temporal_pred_flag[plane]) {
      continue;
    }
    if (!write_frame_filters_impl(writer, config, params, plane, refs,
                                  error)) {
      return false;
    }
  }
  return true;
}

bool wiener_read_lr_params(WienerBitReader& reader,
                           const WienerLrHeaderConfig& config,
                           WienerLrHeaderParams* params,
                           const WienerReferenceSet& refs,
                           std::string* error) {
  if (params == nullptr) {
    return fail(error, "null LrParams");
  }
  *params = WienerLrHeaderParams{};
  if (config.coded_lossless || !config.enable_restoration) {
    for (int p = 0; p < 3; ++p) {
      params->frame_type[p] = kRestoreNone;
      params->frame_filters_on[p] = false;
      params->temporal_pred_flag[p] = false;
    }
    return true;
  }
  params->num_filter_classes = 1;
  bool uses_luma_lr = false;
  bool uses_chroma_lr = false;
  for (int plane = 0; plane < config.num_planes; ++plane) {
    const bool chroma = plane > 0;
    int index_to_tool[4] = {kRestoreNone, 0, 0, 0};
    int tools_count = 1;
    for (int i = 1; i < kRestoreSwitchable; ++i) {
      if (!config.lr_tools_disable[chroma ? 1 : 0][i]) {
        index_to_tool[tools_count++] = i;
      }
    }
    index_to_tool[tools_count] = kRestoreSwitchable;
    const bool allow_switchable = tools_count > 2;
    const int n = tools_count + (allow_switchable ? 1 : 0);
    std::uint32_t tool_index = 0;
    if (!reader.read_ns(static_cast<unsigned>(n), &tool_index) ||
        static_cast<int>(tool_index) >= n) {
      return fail(error, "truncated FrameRestorationType tool_index");
    }
    params->frame_type[plane] = index_to_tool[tool_index];
    if (params->frame_type[plane] != kRestoreNone) {
      if (plane == 0) {
        uses_luma_lr = true;
      } else {
        uses_chroma_lr = true;
      }
    }
    const int type = params->frame_type[plane];
    if (type == kRestorePcWiener) {
      return fail(error,
                  "PC-Wiener is not implementable without the Pc_Wiener_* "
                  "tables");
    }
    if (type == kRestoreWienerNs || type == kRestoreSwitchable) {
      if (!reader.read_bit(&params->frame_filters_on[plane])) {
        return fail(error, "truncated frame_filters_on");
      }
      if (params->frame_filters_on[plane]) {
        const int num_ref_frames =
            (refs.frame_is_intra || config.is_switch_frame)
                ? 0
                : config.num_total_refs;
        if (num_ref_frames > kMaxLrRefs) {
          return fail(error, "NumTotalRefs exceeds the reference array");
        }
        if (num_ref_frames > 0) {
          if (!reader.read_bit(&params->temporal_pred_flag[plane])) {
            return fail(error, "truncated temporal_pred_flag");
          }
        }
        if (params->temporal_pred_flag[plane] && num_ref_frames > 1) {
          const int bits = ceil_log2(num_ref_frames);
          std::uint32_t rst = 0;
          if (!reader.read_bits(static_cast<unsigned>(bits), &rst) ||
              static_cast<int>(rst) >= num_ref_frames) {
            return fail(error, "truncated rst_ref_pic_idx");
          }
          params->rst_ref_pic_idx[plane] = static_cast<int>(rst);
        } else {
          params->rst_ref_pic_idx[plane] = 0;
        }
        if (params->temporal_pred_flag[plane]) {
          const int rst = params->rst_ref_pic_idx[plane];
          if (rst < 0 || rst >= num_ref_frames || rst >= kMaxLrRefs) {
            return fail(error, "rst_ref_pic_idx outside the reference set");
          }
          const int ref_idx = refs.ref_frame_idx[rst];
          if (ref_idx < 0 || ref_idx >= kMaxLrRefs) {
            return fail(error, "ref_frame_idx outside NUM_REF_FRAMES");
          }
          int ref_plane = plane;
          if (plane > 0 && !refs.ref[ref_idx].filters_on[plane]) {
            ref_plane = (plane == 1) ? 2 : 1;
          }
          if (!refs.ref[ref_idx].filters_on[ref_plane]) {
            return fail(error,
                        "temporal prediction references a frame without "
                        "frame filters on the required plane");
          }
          if (plane == 0) {
            params->num_filter_classes =
                refs.ref[ref_idx].num_filter_classes;
            params->num_filter_classes_idx =
                encode_num_filter_classes_idx(params->num_filter_classes);
            if (params->num_filter_classes_idx < 0) {
              return fail(error, "reference NumFilterClasses is invalid");
            }
          }
          for (int c = 0; c < kWienerNsClasses; ++c) {
            for (int j = 0; j < kWienerNsCoeffStride; ++j) {
              params->frame_coeffs[plane][c][j] =
                  refs.ref[ref_idx].coeffs[ref_plane][c][j];
            }
          }
        } else if (plane == 0) {
          std::uint32_t idx = 0;
          if (!reader.read_bits(3, &idx)) {
            return fail(error, "truncated num_filter_classes_idx");
          }
          params->num_filter_classes_idx = static_cast<int>(idx);
          params->num_filter_classes =
              decode_num_filter_classes(params->num_filter_classes_idx);
          if (params->num_filter_classes == 0) {
            return fail(error, "invalid num_filter_classes_idx");
          }
        }
      }
    } else {
      params->frame_filters_on[plane] = false;
      params->temporal_pred_flag[plane] = false;
    }
  }

  if (uses_luma_lr &&
      !read_lr_size_flags(reader, config.sb_size, &params->luma_shift,
                          error)) {
    return false;
  }
  if (uses_chroma_lr &&
      !read_lr_size_flags(reader, config.sb_size, &params->chroma_shift,
                          error)) {
    return false;
  }

  for (int plane = 0; plane < config.num_planes; ++plane) {
    if (!params->frame_filters_on[plane] ||
        params->temporal_pred_flag[plane]) {
      continue;
    }
    if (!read_frame_filters_impl(reader, config, params, plane, refs,
                                 error)) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Tile state / read_lr_unit()
// ---------------------------------------------------------------------------

bool wiener_tile_init(WienerTileState* state,
                      const WienerLrHeaderParams& params, int mi_cols,
                      int mi_rows, int sub_x, int sub_y,
                      std::string* error) {
  if (state == nullptr || mi_cols <= 0 || mi_rows <= 0) {
    return fail(error, "invalid LR tile geometry");
  }
  *state = WienerTileState{};
  state->params = params;
  state->mi_cols = mi_cols;
  state->mi_rows = mi_rows;
  state->sub_x = sub_x;
  state->sub_y = sub_y;

  copy_cdf(state->use_wiener_ns, kUseWienerNsCdf);
  copy_cdf(state->use_pc_wiener, kUsePcWienerCdf);
  for (int tool = 0; tool < 3; ++tool) {
    for (int plane = 0; plane < 3; ++plane) {
      copy_cdf(state->flex_restoration_type[tool][plane],
               kFlexRestorationTypeCdf[tool][plane]);
    }
  }
  copy_cdf(state->wiener_ns_length[0], kWienerNsLengthCdf[0]);
  copy_cdf(state->wiener_ns_length[1], kWienerNsLengthCdf[1]);
  copy_cdf(state->wiener_ns_uv_sym, kWienerNsUvSymCdf);
  copy_cdf(state->wiener_ns_base, kWienerNsBaseCdf);

  const int sub = std::max(sub_x, sub_y);
  state->unit_size[0] = kRestorationTileSizeMax >> params.luma_shift;
  state->unit_size[1] =
      (kRestorationTileSizeMax >> sub) >> params.chroma_shift;
  state->unit_size[2] = state->unit_size[1];
  for (int plane = 0; plane < 3; ++plane) {
    const int size = state->unit_size[plane];
    if (size <= 0 || (size & (size - 1)) != 0) {
      return fail(error, "LoopRestorationSize is not a power of two");
    }
    if (plane == 0 && size < 64) {
      return fail(error, "luma LoopRestorationSize below the stripe height");
    }
    if (plane > 0 && size < (64 >> sub_y)) {
      return fail(error, "chroma LoopRestorationSize below the stripe height");
    }
    const int sx = (plane == 0) ? 0 : sub_x;
    const int sy = (plane == 0) ? 0 : sub_y;
    state->unit_cols[plane] =
        lr_count_units(size, (mi_cols * kMiSize) >> sx);
    state->unit_rows[plane] =
        lr_count_units(size, (mi_rows * kMiSize) >> sy);
    const std::size_t units =
        static_cast<std::size_t>(state->unit_cols[plane]) *
        static_cast<std::size_t>(state->unit_rows[plane]);
    state->unit_type[plane].assign(units, static_cast<std::uint8_t>(kRestoreNone));
    state->unit_coeffs[plane].assign(units * kWienerNsCoeffStride, 0);
  }

  for (int plane = 0; plane < 3; ++plane) {
    if (params.frame_type[plane] != kRestoreNone &&
        params.frame_type[plane] != kRestoreWienerNs &&
        params.frame_type[plane] != kRestoreSwitchable) {
      return fail(error, "unsupported FrameRestorationType");
    }
    for (int c = 0; c < kWienerNsClasses; ++c) {
      state->bank_ptr[plane][c] = 0;
      state->bank_size[plane][c] = 0;
      for (int bank = 0; bank < kLrBankSize; ++bank) {
        for (int j = 0; j < kWienerNsCoeffStride; ++j) {
          const int min = kWienerNsTapsMin[plane == 0 ? 0 : 1][j];
          const int k = kWienerNsTapsK[plane == 0 ? 0 : 1][j];
          state->ref_bank[plane][c][bank][j] =
              static_cast<std::int16_t>(min + ((1 << k) >> 1));
        }
      }
    }
  }
  if (params.frame_filters_on[0] && params.num_filter_classes > 1) {
    // The class map is supplied by the caller; tile syntax itself only needs
    // the coefficients.  No table is required here.
  }
  return true;
}

namespace {

bool write_unit_filter(av2::Encoder& encoder, WienerTileState& state,
                       int plane, int unit_row, int unit_col,
                       const WienerUnitSyntax& unit, std::string* error) {
  const bool chroma = plane > 0;
  const int n_coeffs = (plane == 0) ? kWienerNsLumaCoeffs
                                    : kWienerNsChromaCoeffs;
  const int num_subsets = wiener_num_subsets(plane);
  const int cls = 0;
  const int old_size = state.bank_size[plane][cls];
  const int old_ptr = state.bank_ptr[plane][cls];

  if (!raw_bit(encoder, unit.merged, error)) {
    return false;
  }
  int k_sel = unit.use_bank_k;
  if (k_sel < 0) {
    k_sel = (old_size <= 1) ? 0 : old_size - 1;
  }
  const int max_k = std::max(1, old_size);
  if (k_sel < 0 || k_sel >= max_k) {
    return fail(error, "use_bank_k outside the current bank");
  }
  for (int k = 0; k < old_size - 1; ++k) {
    const bool bit = k == k_sel;
    if (!raw_bit(encoder, bit, error)) {
      return false;
    }
    if (bit) {
      break;
    }
  }
  const int ref_slot = (old_ptr - k_sel + kLrBankSize) & (kLrBankSize - 1);
  const std::size_t base =
      (static_cast<std::size_t>(unit_row) *
           static_cast<std::size_t>(state.unit_cols[plane]) +
       static_cast<std::size_t>(unit_col)) *
      kWienerNsCoeffStride;

  if (unit.merged) {
    if (state.bank_size[plane][cls] == 0) {
      state.bank_size[plane][cls] = 1;
    }
    for (int j = 0; j < n_coeffs; ++j) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] =
          state.ref_bank[plane][cls][ref_slot][j];
    }
    return true;
  }

  int new_ptr = old_ptr;
  int new_size = old_size;
  if (old_size < kLrBankSize) {
    new_ptr = old_size;
    new_size = old_size + 1;
  } else {
    new_ptr = (old_ptr + 1) & (kLrBankSize - 1);
  }

  if (unit.subset < 0 || unit.subset >= num_subsets) {
    return fail(error, "per-unit Wiener subset out of range");
  }
  for (int s = 0; s < num_subsets - 1; ++s) {
    const bool len = s < unit.subset;
    const av2::Status status = encoder.encode_symbol(
        &state.wiener_ns_length[std::min(plane, 1)][0], 2, len ? 1 : 0);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("wiener_ns_length encode failed: ") +
                             av2::status_name(status));
    }
    if (!len) {
      break;
    }
  }
  bool uv_sym = false;
  if (chroma && unit.subset > 0) {
    uv_sym = unit.uv_sym;
    const av2::Status status =
        encoder.encode_symbol(&state.wiener_ns_uv_sym[0], 2,
                              uv_sym ? 1 : 0);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("wiener_ns_uv_sym encode failed: ") +
                             av2::status_name(status));
    }
    if (uv_sym) {
      for (int j = kWienerNsShortCoeffs; j + 1 < n_coeffs; j += 2) {
        if (unit.coeffs[j] != unit.coeffs[j + 1]) {
          return fail(error, "wiener_ns_uv_sym requires symmetric pairs");
        }
      }
    }
  }

  state.bank_ptr[plane][cls] = new_ptr;
  state.bank_size[plane][cls] = new_size;

  for (int j = 0; j < n_coeffs; ++j) {
    const int min = kWienerNsTapsMin[chroma ? 1 : 0][j];
    const int k = kWienerNsTapsK[chroma ? 1 : 0][j];
    std::int16_t v = state.ref_bank[plane][cls][ref_slot][j];
    if (wiener_tap_present(plane, unit.subset, j)) {
      const int target = unit.coeffs[j];
      if (target < min || target >= min + (1 << k)) {
        return fail(error, "per-unit Wiener coefficient outside its tap range");
      }
      if (!wiener_encode_signed_4part(
              encoder, state.wiener_ns_base, min, k,
              state.ref_bank[plane][cls][ref_slot][j], target, error)) {
        return false;
      }
      v = static_cast<std::int16_t>(target);
    } else {
      v = 0;
    }
    state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = v;
    state.ref_bank[plane][cls][new_ptr][j] = v;
    if (chroma && unit.subset > 0 && uv_sym &&
        j >= kWienerNsShortCoeffs) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j) + 1] = v;
      state.ref_bank[plane][cls][new_ptr][j + 1] = v;
      ++j;
    }
  }
  for (int j = n_coeffs; j < kWienerNsCoeffStride; ++j) {
    state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = 0;
    state.ref_bank[plane][cls][new_ptr][j] = 0;
  }
  return true;
}

bool read_unit_filter(av2::Decoder& decoder, WienerTileState& state,
                      int plane, int unit_row, int unit_col,
                      WienerUnitSyntax* unit, std::string* error) {
  const bool chroma = plane > 0;
  const int n_coeffs = (plane == 0) ? kWienerNsLumaCoeffs
                                    : kWienerNsChromaCoeffs;
  const int num_subsets = wiener_num_subsets(plane);
  const int cls = 0;
  const int old_size = state.bank_size[plane][cls];
  const int old_ptr = state.bank_ptr[plane][cls];

  bool merged = false;
  if (!raw_bit(decoder, &merged, error)) {
    return false;
  }
  int k_sel = 0;
  for (int k = 0; k < old_size - 1; ++k) {
    bool bit = false;
    if (!raw_bit(decoder, &bit, error)) {
      return false;
    }
    if (bit) {
      k_sel = k;
      break;
    }
    k_sel = k + 1;
  }
  const int ref_slot = (old_ptr - k_sel + kLrBankSize) & (kLrBankSize - 1);
  const std::size_t base =
      (static_cast<std::size_t>(unit_row) *
           static_cast<std::size_t>(state.unit_cols[plane]) +
       static_cast<std::size_t>(unit_col)) *
      kWienerNsCoeffStride;
  if (unit != nullptr) {
    unit->merged = merged;
    unit->use_bank_k = k_sel;
  }

  if (merged) {
    if (state.bank_size[plane][cls] == 0) {
      state.bank_size[plane][cls] = 1;
    }
    for (int j = 0; j < n_coeffs; ++j) {
      const std::int16_t v = state.ref_bank[plane][cls][ref_slot][j];
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = v;
      if (unit != nullptr) {
        unit->coeffs[j] = v;
      }
    }
    return true;
  }

  int new_ptr = old_ptr;
  int new_size = old_size;
  if (old_size < kLrBankSize) {
    new_ptr = old_size;
    new_size = old_size + 1;
  } else {
    new_ptr = (old_ptr + 1) & (kLrBankSize - 1);
  }

  int subset = 0;
  for (int s = 0; s < num_subsets - 1; ++s) {
    int len = 0;
    const av2::Status status = decoder.decode_symbol(
        &state.wiener_ns_length[std::min(plane, 1)][0], 2, &len);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("wiener_ns_length decode failed: ") +
                             av2::status_name(status));
    }
    if (len == 0) {
      break;
    }
    ++subset;
  }
  bool uv_sym = false;
  if (chroma && subset > 0) {
    int sym = 0;
    const av2::Status status =
        decoder.decode_symbol(&state.wiener_ns_uv_sym[0], 2, &sym);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("wiener_ns_uv_sym decode failed: ") +
                             av2::status_name(status));
    }
    uv_sym = sym != 0;
  }

  state.bank_ptr[plane][cls] = new_ptr;
  state.bank_size[plane][cls] = new_size;
  if (unit != nullptr) {
    unit->subset = subset;
    unit->uv_sym = uv_sym;
  }

  for (int j = 0; j < n_coeffs; ++j) {
    const int min = kWienerNsTapsMin[chroma ? 1 : 0][j];
    const int k = kWienerNsTapsK[chroma ? 1 : 0][j];
    std::int16_t v = state.ref_bank[plane][cls][ref_slot][j];
    if (wiener_tap_present(plane, subset, j)) {
      int target = 0;
      if (!wiener_decode_signed_4part(
              decoder, state.wiener_ns_base, min, k,
              state.ref_bank[plane][cls][ref_slot][j], &target, error)) {
        return false;
      }
      v = static_cast<std::int16_t>(target);
    } else {
      v = 0;
    }
    state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = v;
    state.ref_bank[plane][cls][new_ptr][j] = v;
    if (unit != nullptr) {
      unit->coeffs[j] = v;
    }
    if (chroma && subset > 0 && uv_sym &&
        j >= kWienerNsShortCoeffs) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j) + 1] = v;
      state.ref_bank[plane][cls][new_ptr][j + 1] = v;
      if (unit != nullptr) {
        unit->coeffs[j + 1] = v;
      }
      ++j;
    }
  }
  for (int j = n_coeffs; j < kWienerNsCoeffStride; ++j) {
    state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = 0;
    state.ref_bank[plane][cls][new_ptr][j] = 0;
  }
  return true;
}

}  // namespace

bool wiener_write_unit(av2::Encoder& encoder, WienerTileState& state,
                       int plane, int unit_row, int unit_col,
                       const WienerUnitSyntax& unit, std::string* error) {
  if (plane < 0 || plane > 2 || unit_row < 0 || unit_col < 0 ||
      unit_row >= state.unit_rows[plane] ||
      unit_col >= state.unit_cols[plane]) {
    return fail(error, "LR unit outside the frame grid");
  }
  const int frame_type = state.params.frame_type[plane];
  if (frame_type == kRestoreNone) {
    if (unit.type != kRestoreNone) {
      return fail(error, "LR unit type set on a RESTORE_NONE plane");
    }
    return true;
  }
  if (frame_type == kRestorePcWiener) {
    return fail(error, "PC-Wiener is not supported");
  }
  int type = kRestoreNone;
  if (frame_type == kRestoreWienerNs) {
    if (unit.type != kRestoreNone && unit.type != kRestoreWienerNs) {
      return fail(error, "WIENER_NONSEP unit must be on or off");
    }
    const bool on = unit.type == kRestoreWienerNs;
    const av2::Status status =
        encoder.encode_symbol(&state.use_wiener_ns[0], 2, on ? 1 : 0);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("use_wiener_ns encode failed: ") +
                             av2::status_name(status));
    }
    type = on ? kRestoreWienerNs : kRestoreNone;
  } else if (frame_type == kRestoreSwitchable) {
    if (unit.type != kRestoreNone && unit.type != kRestoreWienerNs) {
      return fail(error, "switchable unit must be NONE or WIENER_NONSEP");
    }
    type = kRestoreWienerNs;
    for (int tool = 0; tool < kRestoreSwitchable - 1; ++tool) {
      const bool on = unit.type == tool;
      const av2::Status status = encoder.encode_symbol(
          &state.flex_restoration_type[tool][plane][0], 2, on ? 1 : 0);
      if (status != av2::Status::kOk) {
        return fail(error, std::string("flex_restoration_type encode failed: ") +
                               av2::status_name(status));
      }
      if (on) {
        type = tool;
        break;
      }
    }
  } else {
    return fail(error, "unsupported FrameRestorationType");
  }

  const std::size_t index =
      static_cast<std::size_t>(unit_row) *
          static_cast<std::size_t>(state.unit_cols[plane]) +
      static_cast<std::size_t>(unit_col);
  state.unit_type[plane][index] = static_cast<std::uint8_t>(type);
  if (type == kRestoreWienerNs) {
    if (state.params.frame_filters_on[plane]) {
      return true;  // read_wienerns_filter() returns immediately.
    }
    return write_unit_filter(encoder, state, plane, unit_row, unit_col, unit,
                             error);
  }
  return true;
}

bool wiener_read_unit(av2::Decoder& decoder, WienerTileState& state,
                      int plane, int unit_row, int unit_col,
                      WienerUnitSyntax* unit, std::string* error) {
  if (plane < 0 || plane > 2 || unit_row < 0 || unit_col < 0 ||
      unit_row >= state.unit_rows[plane] ||
      unit_col >= state.unit_cols[plane]) {
    return fail(error, "LR unit outside the frame grid");
  }
  if (unit != nullptr) {
    *unit = WienerUnitSyntax{};
  }
  const int frame_type = state.params.frame_type[plane];
  if (frame_type == kRestoreNone) {
    return true;
  }
  if (frame_type == kRestorePcWiener) {
    return fail(error, "PC-Wiener is not supported");
  }
  int type = kRestoreNone;
  if (frame_type == kRestoreWienerNs) {
    int on = 0;
    const av2::Status status =
        decoder.decode_symbol(&state.use_wiener_ns[0], 2, &on);
    if (status != av2::Status::kOk) {
      return fail(error, std::string("use_wiener_ns decode failed: ") +
                             av2::status_name(status));
    }
    type = on != 0 ? kRestoreWienerNs : kRestoreNone;
  } else if (frame_type == kRestoreSwitchable) {
    type = kRestoreWienerNs;
    for (int tool = 0; tool < kRestoreSwitchable - 1; ++tool) {
      int on = 0;
      const av2::Status status = decoder.decode_symbol(
          &state.flex_restoration_type[tool][plane][0], 2, &on);
      if (status != av2::Status::kOk) {
        return fail(error, std::string("flex_restoration_type decode failed: ") +
                               av2::status_name(status));
      }
      if (on != 0) {
        type = tool;
        break;
      }
    }
    if (type == kRestorePcWiener) {
      return fail(error, "switchable PC-Wiener unit is not supported");
    }
  } else {
    return fail(error, "unsupported FrameRestorationType");
  }

  const std::size_t index =
      static_cast<std::size_t>(unit_row) *
          static_cast<std::size_t>(state.unit_cols[plane]) +
      static_cast<std::size_t>(unit_col);
  state.unit_type[plane][index] = static_cast<std::uint8_t>(type);
  if (unit != nullptr) {
    unit->type = type;
  }
  if (type == kRestoreWienerNs) {
    if (state.params.frame_filters_on[plane]) {
      return true;
    }
    return read_unit_filter(decoder, state, plane, unit_row, unit_col, unit,
                            error);
  }
  return true;
}

namespace {

bool walk_tile(av2::Encoder* encoder, av2::Decoder* decoder,
               WienerTileState& state,
               const std::vector<WienerUnitSyntax> write_plans[3],
               std::vector<WienerUnitSyntax> read_plans[3], int sb_size,
               int tile_mi_row_start, int tile_mi_col_start,
               std::string* error) {
  if (sb_size <= 0 || (sb_size % kMiSize) != 0) {
    return fail(error, "invalid superblock size");
  }
  const int sb4 = sb_size / kMiSize;
  if (tile_mi_row_start < 0 || tile_mi_col_start < 0 ||
      tile_mi_row_start >= state.mi_rows || tile_mi_col_start >= state.mi_cols) {
    return fail(error, "tile origin outside the frame");
  }
  std::vector<std::uint8_t> written[3];
  for (int plane = 0; plane < 3; ++plane) {
    const std::size_t units =
        static_cast<std::size_t>(state.unit_cols[plane]) *
        static_cast<std::size_t>(state.unit_rows[plane]);
    written[plane].assign(units, 0);
    if (state.params.frame_type[plane] != kRestoreNone &&
        write_plans != nullptr &&
        static_cast<int>(write_plans[plane].size()) <
            static_cast<int>(units)) {
      return fail(error, "write plan grid is smaller than the unit grid");
    }
    if (read_plans != nullptr) {
      read_plans[plane].assign(units, WienerUnitSyntax{});
    }
  }
  for (int mi_row = tile_mi_row_start; mi_row < state.mi_rows;
       mi_row += sb4) {
    for (int mi_col = tile_mi_col_start; mi_col < state.mi_cols;
         mi_col += sb4) {
      for (int plane = 0; plane < 3; ++plane) {
        if (state.params.frame_type[plane] == kRestoreNone) {
          continue;
        }
        WienerUnitRange range;
        if (!wiener_unit_range(state, plane, mi_row, mi_col, sb4, sb4,
                               tile_mi_row_start, tile_mi_col_start, &range,
                               error)) {
          return false;
        }
        for (int r = range.row_begin; r < range.row_end; ++r) {
          for (int c = range.col_begin; c < range.col_end; ++c) {
            if (r < 0 || c < 0 || r >= state.unit_rows[plane] ||
                c >= state.unit_cols[plane]) {
              return fail(error, "read_lr unit outside the unit grid");
            }
            const std::size_t index =
                static_cast<std::size_t>(r) *
                    static_cast<std::size_t>(state.unit_cols[plane]) +
                static_cast<std::size_t>(c);
            if (written[plane][index]) {
              return fail(error, "read_lr visits a unit twice");
            }
            written[plane][index] = 1;
            if (encoder != nullptr) {
              if (!wiener_write_unit(*encoder, state, plane, r, c,
                                     write_plans[plane][index], error)) {
                return false;
              }
            } else {
              WienerUnitSyntax unit;
              if (!wiener_read_unit(*decoder, state, plane, r, c, &unit,
                                    error)) {
                return false;
              }
              if (read_plans != nullptr) {
                read_plans[plane][index] = unit;
              }
            }
          }
        }
      }
    }
  }
  for (int plane = 0; plane < 3; ++plane) {
    if (state.params.frame_type[plane] == kRestoreNone) {
      continue;
    }
    const int sub_x = (plane == 0) ? 0 : state.sub_x;
    const int sub_y = (plane == 0) ? 0 : state.sub_y;
    const int unit_size = state.unit_size[plane];
    const int lr_row_offset =
        (tile_mi_row_start * kMiSize >> sub_y) / unit_size;
    const int lr_col_offset =
        (tile_mi_col_start * kMiSize >> sub_x) / unit_size;
    const int tile_unit_rows = lr_count_units(
        unit_size, ((state.mi_rows - tile_mi_row_start) * kMiSize) >> sub_y);
    const int tile_unit_cols = lr_count_units(
        unit_size, ((state.mi_cols - tile_mi_col_start) * kMiSize) >> sub_x);
    for (int r = lr_row_offset; r < lr_row_offset + tile_unit_rows; ++r) {
      for (int c = lr_col_offset; c < lr_col_offset + tile_unit_cols; ++c) {
        if (r < 0 || c < 0 || r >= state.unit_rows[plane] ||
            c >= state.unit_cols[plane]) {
          return fail(error, "tile unit range outside the unit grid");
        }
        const std::size_t index =
            static_cast<std::size_t>(r) *
                static_cast<std::size_t>(state.unit_cols[plane]) +
            static_cast<std::size_t>(c);
        if (!written[plane][index]) {
          return fail(error, "read_lr did not visit every restoration unit");
        }
      }
    }
  }
  return true;
}

}  // namespace

bool wiener_write_tile(av2::Encoder& encoder, WienerTileState& state,
                       const std::vector<WienerUnitSyntax> plans[3],
                       int sb_size, int tile_mi_row_start,
                       int tile_mi_col_start, std::string* error) {
  if (plans == nullptr) {
    return fail(error, "null tile plans");
  }
  return walk_tile(&encoder, nullptr, state, plans, nullptr, sb_size,
                   tile_mi_row_start, tile_mi_col_start, error);
}

bool wiener_read_tile(av2::Decoder& decoder, WienerTileState& state,
                      std::vector<WienerUnitSyntax> plans[3], int sb_size,
                      int tile_mi_row_start, int tile_mi_col_start,
                      std::string* error) {
  return walk_tile(nullptr, &decoder, state, nullptr, plans, sb_size,
                   tile_mi_row_start, tile_mi_col_start, error);
}

bool wiener_unit_range(const WienerTileState& state, int plane, int mi_row,
                       int mi_col, int w4, int h4, int tile_mi_row_start,
                       int tile_mi_col_start, WienerUnitRange* range,
                       std::string* error) {
  if (range == nullptr || plane < 0 || plane > 2) {
    return fail(error, "invalid read_lr argument");
  }
  *range = WienerUnitRange{};
  const int type = state.params.frame_type[plane];
  if (type == kRestoreNone) {
    return true;
  }
  const int sub_x = (plane == 0) ? 0 : state.sub_x;
  const int sub_y = (plane == 0) ? 0 : state.sub_y;
  const int unit_size = state.unit_size[plane];
  if (unit_size <= 0) {
    return fail(error, "invalid LoopRestorationSize");
  }
  const int mi_cols = state.mi_cols - tile_mi_col_start;
  const int mi_rows = state.mi_rows - tile_mi_row_start;
  const int c = mi_col - tile_mi_col_start;
  const int r = mi_row - tile_mi_row_start;
  if (c < 0 || r < 0) {
    return fail(error, "read_lr outside the tile");
  }
  const int unit_rows =
      lr_count_units(unit_size, (mi_rows * kMiSize) >> sub_y);
  const int unit_cols =
      lr_count_units(unit_size, (mi_cols * kMiSize) >> sub_x);
  const int lr_row_offset =
      (tile_mi_row_start * kMiSize >> sub_y) / unit_size;
  const int lr_col_offset =
      (tile_mi_col_start * kMiSize >> sub_x) / unit_size;
  const int step_y = kMiSize >> sub_y;
  const int step_x = kMiSize >> sub_x;
  if (step_y <= 0 || step_x <= 0) {
    return fail(error, "invalid subsampling in read_lr");
  }
  const int unit_row_start = (r * step_y + unit_size - 1) / unit_size;
  const int unit_row_end =
      std::min(unit_rows, ((r + h4) * step_y + unit_size - 1) / unit_size);
  const int unit_col_start = (c * step_x + unit_size - 1) / unit_size;
  const int unit_col_end =
      std::min(unit_cols, ((c + w4) * step_x + unit_size - 1) / unit_size);
  range->row_begin = unit_row_start + lr_row_offset;
  range->row_end = unit_row_end + lr_row_offset;
  range->col_begin = unit_col_start + lr_col_offset;
  range->col_end = unit_col_end + lr_col_offset;
  return true;
}

}  // namespace av2enc::restoration::classes
