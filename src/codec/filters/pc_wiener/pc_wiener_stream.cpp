#include "codec/filters/pc_wiener/pc_wiener_stream.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#include "codec/filters/loop_restoration/classes/wiener_cdfs.h"
#include "codec/filters/loop_restoration/restoration_common.h"
#include "codec/filters/loop_restoration/wiener_core.h"
#include "codec/filters/tables/pc_wiener.h"
#include "codec/filters/tables/pc_wiener_tables.h"

namespace av2enc::filters::pc_wiener_stream {
namespace {

namespace rest = av2enc::restoration;
namespace spec = av2enc::filters::spec;
namespace st = av2enc::filters::spec_tables;
namespace rcls = av2enc::restoration::classes;

bool fail(std::string* error, std::string message) {
  if (error != nullptr && error->empty()) *error = std::move(message);
  return false;
}

// §5.20.10.6 Wiener_Ns_Taps_Present[2][4][18].
constexpr std::uint8_t kTapsPresent[2][4][18] = {
    {
        {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
    },
    {
        {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
};

int PredictGroup(const int counts[3]) {
  int pred = 0;
  for (int i = 1; i <= 2; ++i) {
    if (counts[i] > counts[pred]) pred = i;
  }
  return pred;
}

// §5.18.9.6 inverse_recenter().
int InverseRecenter(int r, int v) {
  if (v > 2 * r) return v;
  if ((v & 1) != 0) return r - ((v + 1) >> 1);
  return r + (v >> 1);
}

// §5.18.9.4 decode_unsigned_subexp_with_ref(), value form.
int UnsignedSubexpValue(int mx, int r, int v) {
  if ((r << 1) <= mx) return InverseRecenter(r, v);
  return mx - 1 - InverseRecenter(mx - 1 - r, v);
}

// §5.18.9.5 decode_subexp() inverse.
bool EncodeSubexp(BitWriter& writer, int num_syms, int k, int value,
                  std::string* error) {
  if (num_syms <= 0 || value < 0 || value >= num_syms) {
    return fail(error, "subexp value out of range");
  }
  int i = 0;
  int mk = 0;
  for (;;) {
    const int b = i != 0 ? k + i - 1 : k;
    if (b < 0 || b > 30) return fail(error, "subexp bit width out of range");
    const int a = 1 << b;
    if (num_syms <= mk + 3 * a) {
      PutNs(writer, static_cast<unsigned>(num_syms - mk),
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
                      static_cast<unsigned>(b));
      return true;
    }
  }
}

bool DecodeSubexp(Reader& reader, int num_syms, int k, int* value,
                  std::string* error) {
  if (value == nullptr || num_syms <= 0) {
    return fail(error, "invalid subexp argument");
  }
  int i = 0;
  int mk = 0;
  for (;;) {
    const int b = i != 0 ? k + i - 1 : k;
    if (b < 0 || b > 30) return fail(error, "subexp bit width out of range");
    const int a = 1 << b;
    if (num_syms <= mk + 3 * a) {
      std::uint32_t bits = 0;
      if (!reader.ReadNs(static_cast<unsigned>(num_syms - mk), &bits)) {
        return fail(error, "truncated subexp final bits");
      }
      *value = mk + static_cast<int>(bits);
      return true;
    }
    bool more = false;
    if (!reader.ReadBit(&more)) return fail(error, "truncated subexp bit");
    if (more) {
      ++i;
      mk += a;
    } else {
      std::uint32_t bits = 0;
      if (!reader.ReadBits(static_cast<unsigned>(b), &bits)) {
        return fail(error, "truncated subexp bits");
      }
      *value = mk + static_cast<int>(bits);
      return true;
    }
  }
}

bool EncodeSignedSubexpWithRef(BitWriter& writer, int low, int high, int r,
                               int k, int target, std::string* error) {
  if (high <= low || target < low || target >= high) {
    return fail(error, "signed subexp target out of range");
  }
  const int mx = high - low;
  const int r_off = r - low;
  const int x_off = target - low;
  for (int v = 0; v < mx; ++v) {
    if (UnsignedSubexpValue(mx, r_off, v) == x_off) {
      return EncodeSubexp(writer, mx, k, v, error);
    }
  }
  return fail(error, "signed subexp has no encoding");
}

bool DecodeSignedSubexpWithRef(Reader& reader, int low, int high, int r, int k,
                               int* target, std::string* error) {
  if (target == nullptr || high <= low) {
    return fail(error, "invalid signed subexp argument");
  }
  int v = 0;
  if (!DecodeSubexp(reader, high - low, k, &v, error)) return false;
  *target = low + UnsignedSubexpValue(high - low, r - low, v);
  return true;
}

struct FourPart {
  int base = 0;
  int rem = 0;
  int bits = 0;
  int value = 0;
};

// decode_4part(num) for the per-unit syntax.  `num` is 6 - k.
bool FourPartCode(int num, int base, int rem, FourPart* out) {
  if (out == nullptr) return false;
  const int bits = 2 - num + std::max(1, base);
  if (bits < 1 || bits > 15) return false;
  const int offset = base == 0 ? 0 : (1 << bits);
  if (rem < 0 || rem >= (1 << bits)) return false;
  out->base = base;
  out->rem = rem;
  out->bits = bits;
  out->value = offset + rem;
  return true;
}

int Unsigned4PartValue(int k, int r, int v) {
  const int mx = 1 << k;
  if ((r << 1) <= mx) return InverseRecenter(r, v);
  return mx - 1 - InverseRecenter(mx - 1 - r, v);
}

bool RawBit(av2::Encoder& encoder, bool bit, std::string* error) {
  const av2::Status status = encoder.encode_bool(bit);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("raw bit encode failed: ") +
                           av2::status_name(status));
  }
  return true;
}

bool RawBit(av2::Decoder& decoder, bool* bit, std::string* error) {
  const av2::Status status = decoder.read_bool(bit);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("raw bit decode failed: ") +
                           av2::status_name(status));
  }
  return true;
}

bool Symbol(av2::Encoder& encoder, std::uint16_t* cdf, int n, int symbol,
            const char* name, std::string* error) {
  const av2::Status status = encoder.encode_symbol(cdf, n, symbol);
  if (status != av2::Status::kOk) {
    return fail(error, std::string(name) + " encode failed: " +
                           av2::status_name(status));
  }
  return true;
}

bool Symbol(av2::Decoder& decoder, std::uint16_t* cdf, int n, int* symbol,
            const char* name, std::string* error) {
  const av2::Status status = decoder.decode_symbol(cdf, n, symbol);
  if (status != av2::Status::kOk) {
    return fail(error, std::string(name) + " decode failed: " +
                           av2::status_name(status));
  }
  return true;
}

bool EncodeSigned4Part(av2::Encoder& encoder, std::uint16_t* base_cdf, int low,
                       int k, int r, int target, std::string* error) {
  const int r_off = r - low;
  const int x_off = target - low;
  FourPart found;
  bool have = false;
  for (int base = 0; base < 4 && !have; ++base) {
    const int bits = 2 - (6 - k) + std::max(1, base);
    for (int rem = 0; rem < (1 << bits); ++rem) {
      FourPart code;
      if (!FourPartCode(6 - k, base, rem, &code)) continue;
      if (Unsigned4PartValue(k, r_off, code.value) == x_off) {
        found = code;
        have = true;
        break;
      }
    }
  }
  if (!have) return fail(error, "signed 4part has no encoding");
  if (!Symbol(encoder, base_cdf, 4, found.base, "wiener_ns_base", error)) {
    return false;
  }
  for (int i = found.bits - 1; i >= 0; --i) {
    if (!RawBit(encoder, ((found.rem >> i) & 1) != 0, error)) return false;
  }
  return true;
}

bool DecodeSigned4Part(av2::Decoder& decoder, std::uint16_t* base_cdf, int low,
                       int k, int r, int* target, std::string* error) {
  int base = 0;
  if (!Symbol(decoder, base_cdf, 4, &base, "wiener_ns_base", error)) {
    return false;
  }
  const int bits = 2 - (6 - k) + std::max(1, base);
  if (bits < 1 || bits > 15) return fail(error, "invalid wiener_ns_base");
  std::uint32_t rem = 0;
  const av2::Status status = decoder.read_literal(bits, &rem);
  if (status != av2::Status::kOk) {
    return fail(error, std::string("wiener_ns_rem decode failed: ") +
                           av2::status_name(status));
  }
  FourPart code;
  if (!FourPartCode(6 - k, base, static_cast<int>(rem), &code)) {
    return fail(error, "invalid 4part code");
  }
  *target = low + Unsigned4PartValue(k, r - low, code.value);
  return true;
}

void CopyCdf(std::uint16_t* dst, const std::uint16_t* src, int n) {
  for (int i = 0; i < n; ++i) dst[i] = src[i];
}

int NumCoeffs(int plane) {
  return plane == 0 ? kWienerNsLumaCoeffs : kWienerNsChromaCoeffs;
}

int NumSubsets(int plane) { return plane == 0 ? 4 : 3; }

bool ValidNumFilterClasses(int value) {
  for (int i = 0; i < 8; ++i) {
    if (kDecodeNumFilterClasses[i] == value) return true;
  }
  return false;
}

int EncodeNumFilterClassesIndex(int value) {
  for (int i = 0; i < 8; ++i) {
    if (kDecodeNumFilterClasses[i] == value) return i;
  }
  return -1;
}

// §5.20.10.6 fill_first_slot...() reference for `match`.
void ResolveFrameFilterMatch(const LrParams& params, int plane, int num_classes,
                             int match, std::int16_t ref[kWienerNsCoeffStride]) {
  for (int j = 0; j < kWienerNsCoeffStride; ++j) ref[j] = 0;
  if (match == 0) return;
  if (match < num_classes) {
    for (int j = 0; j < kWienerNsCoeffStride; ++j) {
      ref[j] = params.frame_filter[plane][match - 1].coeffs[j];
    }
    return;
  }
  // numRefFilters is zero for the intra writer/reader; match beyond the
  // class group is the translated PC-Wiener bank.
  TranslatedPcWienerSet(match - num_classes, ref);
}

bool WriteFrameFilters(BitWriter& writer, const LrHeaderContext& context,
                       LrParams* params, int plane, std::string* error) {
  const bool nopcw = context.lr_tools_disable[0][kRestorePcWiener];
  const int num_classes = plane == 0 ? params->num_filter_classes : 1;
  const int num_ref_filters = 0;
  const int num_subsets = NumSubsets(plane);
  const int n_coeffs = NumCoeffs(plane);
  const bool chroma = plane > 0;

  int group_counts[3] = {num_classes, num_ref_filters, 0};
  if (!nopcw && plane == 0) {
    group_counts[2] = kPcWienerBank - num_classes - num_ref_filters;
  }
  if (group_counts[2] < 0) {
    return fail(error, "PC-Wiener frame-filter group is too small");
  }
  int group_hits[3] = {0, 0, 0};
  int group_base[3] = {0, 0, 0};
  for (int i = 1; i < 3; ++i) {
    group_base[i] = group_base[i - 1] + group_counts[i - 1];
  }

  for (int c = 0; c < num_classes; ++c) {
    group_counts[0] = c + 1;
    const int pred_group =
        c == 0 ? (group_counts[1] > 2 ? 1 : PredictGroup(group_counts))
               : PredictGroup(group_hits);
    const int match = params->frame_filter[plane][c].match_index;
    int chosen = -1;
    if (match >= 0 && match < group_base[1]) {
      chosen = 0;
    } else if (match >= group_base[1] && match < group_base[2]) {
      chosen = 1;
    } else if (match >= group_base[2] && match < kPcWienerBank) {
      chosen = 2;
    }
    if (chosen < 0 || group_counts[chosen] <= 0) {
      return fail(error, "frame filter match_index outside the groups");
    }
    int num_zeros = 0;
    int alt_group = 0;
    for (int i = 0; i < 3; ++i) {
      if (i != pred_group) {
        if (group_counts[i] == 0) {
          ++num_zeros;
        } else {
          alt_group = i;
        }
      }
    }
    if (num_zeros != 2) {
      if (chosen == pred_group) {
        writer.put_bit(false);
      } else {
        writer.put_bit(true);
        if (num_zeros == 1) {
          if (chosen != alt_group) {
            return fail(error, "frame filter group is not reachable");
          }
        } else {
          int group_bit = -1;
          for (int gb = 0; gb <= 1; ++gb) {
            const int mapped = pred_group <= gb ? gb + 1 : gb;
            if (mapped == chosen) {
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
    } else if (chosen != pred_group) {
      return fail(error, "frame filter group is not reachable (numZeros==2)");
    }
    const int n = group_counts[chosen];
    const int base = group_base[chosen];
    if (n == 1) {
      if (match != base) {
        return fail(error, "single-entry frame filter group mismatch");
      }
    } else {
      const int ref = base + (n >> 1);
      if (!EncodeSignedSubexpWithRef(writer, base, base + n, ref, 4, match,
                                     error)) {
        return false;
      }
    }
    ++group_hits[chosen];
  }

  for (int c = 0; c < num_classes; ++c) {
    writer.put_bit(params->frame_filter[plane][c].merged);
  }

  for (int c = 0; c < num_classes; ++c) {
    FrameFilter& f = params->frame_filter[plane][c];
    std::int16_t ref[kWienerNsCoeffStride];
    ResolveFrameFilterMatch(*params, plane, num_classes, f.match_index, ref);
    std::int16_t final_coeffs[kWienerNsCoeffStride] = {};
    if (f.merged) {
      f.subset = 0;
      f.uv_sym = false;
      for (int j = 0; j < kWienerNsCoeffStride; ++j) final_coeffs[j] = ref[j];
    } else {
      if (f.match_index > 0 && f.match_index < num_classes &&
          f.match_index > c) {
        return fail(error, "frame filter refers to a later class");
      }
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
        if (!len) break;
      }
      if (chroma && f.subset > 0) writer.put_bit(f.uv_sym);
      for (int j = 0; j < n_coeffs; ++j) {
        const int min = rest::kWienerNsTapsMin[chroma ? 1 : 0][j];
        const int k = rest::kWienerNsTapsK[chroma ? 1 : 0][j];
        std::int16_t v = ref[j];
        if (kTapsPresent[chroma ? 1 : 0][f.subset][j] != 0) {
          const int target = f.coeffs[j];
          if (target < min || target >= min + (1 << k)) {
            return fail(error, "frame Wiener coefficient outside its tap range");
          }
          if (!EncodeSignedSubexpWithRef(writer, min, min + (1 << k), ref[j],
                                         k - 3, target, error)) {
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

bool ReadFrameFilters(Reader& reader, const LrHeaderContext& context,
                      LrParams* params, int plane, std::string* error) {
  const bool nopcw = context.lr_tools_disable[0][kRestorePcWiener];
  const int num_classes = plane == 0 ? params->num_filter_classes : 1;
  const int num_ref_filters = 0;
  const int num_subsets = NumSubsets(plane);
  const int n_coeffs = NumCoeffs(plane);
  const bool chroma = plane > 0;

  int group_counts[3] = {num_classes, num_ref_filters, 0};
  if (!nopcw && plane == 0) {
    group_counts[2] = kPcWienerBank - num_classes - num_ref_filters;
  }
  int group_hits[3] = {0, 0, 0};
  int group_base[3] = {0, 0, 0};
  for (int i = 1; i < 3; ++i) {
    group_base[i] = group_base[i - 1] + group_counts[i - 1];
  }

  for (int c = 0; c < num_classes; ++c) {
    group_counts[0] = c + 1;
    const int pred_group =
        c == 0 ? (group_counts[1] > 2 ? 1 : PredictGroup(group_counts))
               : PredictGroup(group_hits);
    int num_zeros = 0;
    int alt_group = 0;
    for (int i = 0; i < 3; ++i) {
      if (i != pred_group) {
        if (group_counts[i] == 0) {
          ++num_zeros;
        } else {
          alt_group = i;
        }
      }
    }
    bool use_alt = false;
    if (num_zeros != 2 && !reader.ReadBit(&use_alt)) {
      return fail(error, "truncated use_alt_group");
    }
    int group = pred_group;
    if (use_alt) {
      if (num_zeros == 1) {
        group = alt_group;
      } else {
        bool group_bit = false;
        if (!reader.ReadBit(&group_bit)) {
          return fail(error, "truncated group_bit");
        }
        const int gb = group_bit ? 1 : 0;
        group = pred_group <= gb ? gb + 1 : gb;
      }
    }
    if (group < 0 || group > 2 || group_counts[group] <= 0) {
      return fail(error, "frame-filter group outside the table");
    }
    const int n = group_counts[group];
    const int base = group_base[group];
    int match = base;
    if (n > 1) {
      const int ref = base + (n >> 1);
      if (!DecodeSignedSubexpWithRef(reader, base, base + n, ref, 4, &match,
                                     error)) {
        return false;
      }
    }
    params->frame_filter[plane][c].match_index = match;
    ++group_hits[group];
  }

  for (int c = 0; c < num_classes; ++c) {
    if (!reader.ReadBit(&params->frame_filter[plane][c].merged)) {
      return fail(error, "truncated merged_param");
    }
  }

  for (int c = 0; c < num_classes; ++c) {
    FrameFilter& f = params->frame_filter[plane][c];
    if (f.match_index < 0 || f.match_index >= kPcWienerBank) {
      return fail(error, "frame filter match_index out of range");
    }
    if (f.match_index > 0 && f.match_index < num_classes &&
        f.match_index > c) {
      return fail(error, "frame filter refers to a later class");
    }
    std::int16_t ref[kWienerNsCoeffStride];
    ResolveFrameFilterMatch(*params, plane, num_classes, f.match_index, ref);
    std::int16_t final_coeffs[kWienerNsCoeffStride] = {};
    if (f.merged) {
      for (int j = 0; j < kWienerNsCoeffStride; ++j) final_coeffs[j] = ref[j];
    } else {
      int subset = 0;
      for (int s = 0; s < num_subsets - 1; ++s) {
        bool len = false;
        if (!reader.ReadBit(&len)) {
          return fail(error, "truncated wiener_ns_length");
        }
        if (!len) break;
        ++subset;
      }
      f.subset = subset;
      f.uv_sym = false;
      if (chroma && subset > 0) {
        if (!reader.ReadBit(&f.uv_sym)) {
          return fail(error, "truncated wiener_ns_uv_sym");
        }
      }
      for (int j = 0; j < n_coeffs; ++j) {
        const int min = rest::kWienerNsTapsMin[chroma ? 1 : 0][j];
        const int k = rest::kWienerNsTapsK[chroma ? 1 : 0][j];
        std::int16_t v = ref[j];
        if (kTapsPresent[chroma ? 1 : 0][subset][j] != 0) {
          int target = 0;
          if (!DecodeSignedSubexpWithRef(reader, min, min + (1 << k), ref[j],
                                         k - 3, &target, error)) {
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

bool WriteLrSizeFlags(BitWriter& writer, int shift, int sb_size,
                      std::string* error) {
  if (shift < 0 || shift > 3) {
    return fail(error, "LoopRestorationSize shift outside 0..3");
  }
  if (sb_size == 256) {
    if (shift != 0) return fail(error, "256x256 SB only supports shift 0");
    return true;
  }
  writer.put_bit(shift == 1);
  if (shift == 1) return true;
  writer.put_bit(shift == 0);
  if (shift == 0) return true;
  if (sb_size == 128) {
    if (shift != 2) {
      return fail(error, "128x128 SB only supports shift 0,1,2");
    }
    return true;
  }
  if (shift != 2 && shift != 3) {
    return fail(error, "64x64 SB supports shift 0,1,2,3");
  }
  writer.put_bit(shift == 2);
  return true;
}

bool ReadLrSizeFlags(Reader& reader, int sb_size, int* shift,
                     std::string* error) {
  bool half = false;
  if (!reader.ReadBit(&half)) return fail(error, "truncated lr size flag");
  if (half) {
    *shift = 1;
    return true;
  }
  if (sb_size == 256) {
    *shift = 0;
    return true;
  }
  bool max_size = false;
  if (!reader.ReadBit(&max_size)) {
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
  bool quarter = false;
  if (!reader.ReadBit(&quarter)) {
    return fail(error, "truncated lr quarter size flag");
  }
  *shift = quarter ? 2 : 3;
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public small helpers
// ---------------------------------------------------------------------------

int FilterSetIndex(int base_q_idx) { return spec::PcWienerFilterSetIndex(base_q_idx); }

const std::uint8_t kShuffledIndex[64] = {
    16, 7,  58, 21, 12, 61, 26, 38, 18, 30, 50, 45, 23, 49, 43, 62,
    42, 54, 27, 36, 17, 44, 32, 34, 4,  24, 52, 31, 37, 11, 33, 19,
    35, 6,  22, 53, 63, 25, 41, 47, 1,  59, 0,  28, 40, 55, 48, 8,
    5,  51, 9,  46, 56, 60, 15, 2,  13, 14, 57, 29, 3,  20, 39, 10,
};

int TranslatedPcWiener(int m, int j) {
  if (j >= 12 || m < 0 || m >= kPcWienerBank) return 0;
  const int filt = kShuffledIndex[m];
  // Round2Signed(x, PC_WIENER_PREC_BITS - WIENER_NS_PREC_BITS) is the identity
  // (both precisions are 7).
  int coeff = st::kPcWienerFilters[0][filt][j];
  const int min = rest::kWienerNsTapsMin[0][j];
  const int max = min + (1 << rest::kWienerNsTapsK[0][j]) - 1;
  coeff = std::max(min, std::min(max, coeff));
  return coeff;
}

void TranslatedPcWienerSet(int m, std::int16_t out[kWienerNsCoeffStride]) {
  for (int j = 0; j < kWienerNsCoeffStride; ++j) {
    out[j] = static_cast<std::int16_t>(TranslatedPcWiener(m, j));
  }
}

const std::uint8_t* PcWienerSubclassLookup(int filter_set, int num_classes_idx) {
  if (filter_set < 0 || filter_set >= kPcWienerFilterSets ||
      num_classes_idx < 0 || num_classes_idx >= 8) {
    return nullptr;
  }
  return st::kPcWienerSubClassify2[filter_set][num_classes_idx];
}

bool Reader::ReadBit(bool* value) {
  if (value == nullptr || bit_pos_ >= bit_size_) return false;
  const std::size_t byte = bit_pos_ >> 3U;
  const unsigned shift = 7U - static_cast<unsigned>(bit_pos_ & 7U);
  *value = ((data_[byte] >> shift) & 1U) != 0U;
  ++bit_pos_;
  return true;
}

bool Reader::ReadBits(unsigned count, std::uint32_t* value) {
  if (value == nullptr || count > 32U || bit_size_ - bit_pos_ < count) {
    return false;
  }
  std::uint32_t out = 0;
  for (unsigned i = 0; i < count; ++i) {
    bool bit = false;
    if (!ReadBit(&bit)) return false;
    out = (out << 1U) | (bit ? 1U : 0U);
  }
  *value = out;
  return true;
}

bool Reader::ReadNs(unsigned n, std::uint32_t* value) {
  if (n == 0U || value == nullptr) return false;
  if (n == 1U) {
    *value = 0;
    return true;
  }
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  std::uint32_t v = 0;
  if (!ReadBits(w - 1U, &v)) return false;
  if (v < m) {
    *value = v;
    return true;
  }
  bool extra = false;
  if (!ReadBit(&extra)) return false;
  *value = (v << 1U) - m + (extra ? 1U : 0U);
  return *value < n;
}

void PutNs(BitWriter& writer, unsigned n, std::uint32_t value) {
  if (n <= 1U) return;
  unsigned w = 0;
  while ((1U << w) < n) ++w;
  const std::uint32_t m = (1U << w) - n;
  if (value < m) {
    writer.put_bits(value, w - 1U);
  } else {
    const std::uint32_t x = value + m;
    writer.put_bits(x >> 1U, w - 1U);
    writer.put_bit((x & 1U) != 0U);
  }
}

// ---------------------------------------------------------------------------
// §5.4.10 sequence_filter_config()
// ---------------------------------------------------------------------------

std::size_t WriteSequenceFilterConfig(BitWriter& writer,
                                      const SequenceFilterConfig& config,
                                      bool single_picture_header_flag,
                                      std::string* error) {
  if (config.enable_gdf) {
    return fail(error, "PC-Wiener stream module does not implement GDF"), 0;
  }
  if (!single_picture_header_flag && config.cdef_on_skip_txfm_always_on &&
      config.cdef_on_skip_txfm_disabled) {
    return fail(error, "invalid cdef_on_skip_txfm flags"), 0;
  }
  const std::size_t start = writer.bit_size();
  writer.put_bit(config.disable_loopfilters_across_tiles);
  writer.put_bit(config.enable_cdef);
  writer.put_bit(config.enable_gdf);
  if (config.enable_gdf && config.gdf_unit_matches_sb_size) {
    writer.put_bit(true);
  }
  writer.put_bit(config.enable_restoration);
  if (config.enable_restoration) {
    writer.put_bit(config.lr_tools_disable[0][kRestorePcWiener]);
    writer.put_bit(config.lr_tools_disable[0][kRestoreWienerNs]);
    // lr_tools_disable[1][RESTORE_PC_WIENER] is inferred 1.
    writer.put_bit(config.lr_tools_uv_present);
    if (config.lr_tools_uv_present) {
      writer.put_bit(config.lr_tools_disable[1][kRestoreWienerNs]);
    }
    // else: chroma Wiener NS inherits the luma flag.
  }
  writer.put_bit(config.enable_ccso);
  if (config.enable_ccso) {
    writer.put_bit(config.ccso_unit_matches_sb_size);
  }
  if (single_picture_header_flag) {
    // CdefOnSkipTxfm = ADAPTIVE, no bits.
  } else {
    writer.put_bit(config.cdef_on_skip_txfm_always_on);
    if (!config.cdef_on_skip_txfm_always_on) {
      writer.put_bit(config.cdef_on_skip_txfm_disabled);
    }
  }
  writer.put_bits(config.df_par_bits_minus_2, 2U);
  return writer.bit_size() - start;
}

bool ReadSequenceFilterConfig(Reader& reader, bool single_picture_header_flag,
                              SequenceFilterConfig* config,
                              std::string* error) {
  if (config == nullptr) return fail(error, "null sequence filter config");
  SequenceFilterConfig out;
  if (!reader.ReadBit(&out.disable_loopfilters_across_tiles) ||
      !reader.ReadBit(&out.enable_cdef) ||
      !reader.ReadBit(&out.enable_gdf)) {
    return fail(error, "truncated sequence_filter_config");
  }
  if (out.enable_gdf && !reader.ReadBit(&out.gdf_unit_matches_sb_size)) {
    return fail(error, "truncated gdf_unit_matches_sb_size");
  }
  if (!reader.ReadBit(&out.enable_restoration)) {
    return fail(error, "truncated enable_restoration");
  }
  out.lr_tools_disable[0][kRestoreNone] = false;
  out.lr_tools_disable[1][kRestoreNone] = false;
  out.lr_tools_disable[1][kRestorePcWiener] = true;
  out.lr_tools_disable[1][kRestoreSwitchable] = false;
  if (out.enable_restoration) {
    if (!reader.ReadBit(&out.lr_tools_disable[0][kRestorePcWiener]) ||
        !reader.ReadBit(&out.lr_tools_disable[0][kRestoreWienerNs]) ||
        !reader.ReadBit(&out.lr_tools_uv_present)) {
      return fail(error, "truncated lr_tools_disable");
    }
    if (out.lr_tools_uv_present) {
      if (!reader.ReadBit(&out.lr_tools_disable[1][kRestoreWienerNs])) {
        return fail(error, "truncated lr_tools_disable[1]");
      }
    } else {
      out.lr_tools_disable[1][kRestoreWienerNs] =
          out.lr_tools_disable[0][kRestoreWienerNs];
    }
  }
  if (!reader.ReadBit(&out.enable_ccso)) {
    return fail(error, "truncated enable_ccso");
  }
  if (out.enable_ccso &&
      !reader.ReadBit(&out.ccso_unit_matches_sb_size)) {
    return fail(error, "truncated ccso_unit_matches_sb_size");
  }
  if (single_picture_header_flag) {
    out.cdef_on_skip_txfm_always_on = false;
    out.cdef_on_skip_txfm_disabled = false;
  } else {
    if (!reader.ReadBit(&out.cdef_on_skip_txfm_always_on)) {
      return fail(error, "truncated cdef_on_skip_txfm_always_on");
    }
    if (!out.cdef_on_skip_txfm_always_on &&
        !reader.ReadBit(&out.cdef_on_skip_txfm_disabled)) {
      return fail(error, "truncated cdef_on_skip_txfm_disabled");
    }
  }
  std::uint32_t df = 0;
  if (!reader.ReadBits(2, &df)) return fail(error, "truncated df_par_bits");
  out.df_par_bits_minus_2 = static_cast<std::uint8_t>(df);
  if (out.enable_gdf) return fail(error, "GDF is outside this module");
  *config = out;
  return true;
}

std::size_t SequenceFilterConfigBits(const SequenceFilterConfig& config,
                                     bool single_picture_header_flag,
                                     std::string* error) {
  BitWriter writer;
  return WriteSequenceFilterConfig(writer, config, single_picture_header_flag,
                                   error);
}

// ---------------------------------------------------------------------------
// §5.18.7.11 lr_params()
// ---------------------------------------------------------------------------

int CountUnits(int unit_size, int frame_size) {
  if (unit_size <= 0) return 1;
  const int n = (frame_size + (unit_size >> 1)) / unit_size;
  return n > 1 ? n : 1;
}

std::size_t WriteLrParams(BitWriter& writer, LrParams* params,
                          const LrHeaderContext& context, std::string* error) {
  if (params == nullptr) return fail(error, "null LrParams"), 0;
  if (context.coded_lossless || !context.enable_restoration) {
    for (int p = 0; p < 3; ++p) {
      params->frame_type[p] = kRestoreNone;
      params->frame_filters_on[p] = false;
      params->temporal_pred_flag[p] = false;
    }
    return 0;
  }
  const std::size_t start = writer.bit_size();
  if (params->num_filter_classes != 1 &&
      !ValidNumFilterClasses(params->num_filter_classes)) {
    return fail(error, "NumFilterClasses is not in Decode_Num_Filter_Classes"), 0;
  }
  params->num_filter_classes_idx =
      EncodeNumFilterClassesIndex(params->num_filter_classes);
  if (params->num_filter_classes_idx < 0 && params->num_filter_classes != 1) {
    return fail(error, "cannot encode NumFilterClasses"), 0;
  }

  bool uses_luma_lr = false;
  bool uses_chroma_lr = false;
  for (int plane = 0; plane < context.num_planes; ++plane) {
    const bool chroma = plane > 0;
    int index_to_tool[4] = {kRestoreNone, 0, 0, 0};
    int tools_count = 1;
    for (int i = 1; i < kRestoreSwitchable; ++i) {
      if (!context.lr_tools_disable[chroma ? 1 : 0][i]) {
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
      return fail(error, "FrameRestorationType is not signallable"), 0;
    }
    PutNs(writer, static_cast<unsigned>(n),
          static_cast<std::uint32_t>(tool_index));
    if (target != kRestoreNone) {
      if (chroma) {
        uses_chroma_lr = true;
      } else {
        uses_luma_lr = true;
      }
    }
    if (target == kRestorePcWiener) {
      // §5.18.7.11 codes frame_filters_on only for WIENER_NONSEP/SWITCHABLE.
      if (params->frame_filters_on[plane]) {
        return fail(error, "PC-Wiener frame type has no frame_filters_on bit"), 0;
      }
    } else if (target == kRestoreWienerNs || target == kRestoreSwitchable) {
      writer.put_bit(params->frame_filters_on[plane]);
      if (params->frame_filters_on[plane]) {
        const int num_ref_frames =
            (context.frame_is_intra || context.is_switch_frame)
                ? 0
                : context.num_total_refs;
        if (num_ref_frames > 0) {
          writer.put_bit(params->temporal_pred_flag[plane]);
        } else {
          params->temporal_pred_flag[plane] = false;
        }
        if (params->temporal_pred_flag[plane]) {
          return fail(error,
                      "temporal frame-filter prediction is outside this module"),
                 0;
        }
        params->rst_ref_pic_idx[plane] = 0;
        if (plane == 0) {
          if (params->num_filter_classes_idx < 0 ||
              params->num_filter_classes_idx > 7) {
            return fail(error, "num_filter_classes_idx outside f(3)"), 0;
          }
          writer.put_bits(
              static_cast<std::uint64_t>(params->num_filter_classes_idx), 3U);
        }
      }
    } else {
      params->frame_filters_on[plane] = false;
      params->temporal_pred_flag[plane] = false;
    }
  }

  if (uses_luma_lr && !WriteLrSizeFlags(writer, params->luma_shift,
                                        context.sb_size, error)) {
    return 0;
  }
  if (uses_chroma_lr && !WriteLrSizeFlags(writer, params->chroma_shift,
                                          context.sb_size, error)) {
    return 0;
  }

  if (!params->frame_filters_on[0]) {
    params->num_filter_classes = 1;
    params->num_filter_classes_idx = 0;
    params->subclass_lookup_count = 0;
  } else if (params->num_filter_classes > 1) {
    const std::uint8_t* lookup = PcWienerSubclassLookup(
        FilterSetIndex(context.base_q_idx), params->num_filter_classes_idx);
    if (lookup == nullptr) {
      return fail(error, "invalid PC-Wiener subclass lookup"), 0;
    }
    for (int i = 0; i < kNumLutClasses; ++i) params->subclass_lookup[i] = lookup[i];
    params->subclass_lookup_count = kNumLutClasses;
  } else {
    params->subclass_lookup_count = 0;
  }

  for (int plane = 0; plane < context.num_planes; ++plane) {
    if (!params->frame_filters_on[plane] ||
        params->temporal_pred_flag[plane]) {
      continue;
    }
    if (!WriteFrameFilters(writer, context, params, plane, error)) return 0;
  }
  return writer.bit_size() - start;
}

bool ReadLrParams(Reader& reader, const LrHeaderContext& context,
                  LrParams* params, std::string* error) {
  if (params == nullptr) return fail(error, "null LrParams");
  *params = LrParams{};
  if (context.coded_lossless || !context.enable_restoration) {
    for (int p = 0; p < 3; ++p) {
      params->frame_type[p] = kRestoreNone;
      params->frame_filters_on[p] = false;
      params->temporal_pred_flag[p] = false;
    }
    return true;
  }
  bool uses_luma_lr = false;
  bool uses_chroma_lr = false;
  params->num_filter_classes = 1;
  params->num_filter_classes_idx = 0;
  for (int plane = 0; plane < context.num_planes; ++plane) {
    const bool chroma = plane > 0;
    int index_to_tool[4] = {kRestoreNone, 0, 0, 0};
    int tools_count = 1;
    for (int i = 1; i < kRestoreSwitchable; ++i) {
      if (!context.lr_tools_disable[chroma ? 1 : 0][i]) {
        index_to_tool[tools_count++] = i;
      }
    }
    index_to_tool[tools_count] = kRestoreSwitchable;
    const bool allow_switchable = tools_count > 2;
    const int n = tools_count + (allow_switchable ? 1 : 0);
    std::uint32_t tool_index = 0;
    if (!reader.ReadNs(static_cast<unsigned>(n), &tool_index) ||
        static_cast<int>(tool_index) >= n) {
      return fail(error, "truncated FrameRestorationType tool_index");
    }
    params->frame_type[plane] = index_to_tool[tool_index];
    if (params->frame_type[plane] != kRestoreNone) {
      if (chroma) {
        uses_chroma_lr = true;
      } else {
        uses_luma_lr = true;
      }
    }
    const int type = params->frame_type[plane];
    if (type == kRestoreWienerNs || type == kRestoreSwitchable) {
      if (!reader.ReadBit(&params->frame_filters_on[plane])) {
        return fail(error, "truncated frame_filters_on");
      }
      if (params->frame_filters_on[plane]) {
        const int num_ref_frames =
            (context.frame_is_intra || context.is_switch_frame)
                ? 0
                : context.num_total_refs;
        if (num_ref_frames > 0) {
          if (!reader.ReadBit(&params->temporal_pred_flag[plane])) {
            return fail(error, "truncated temporal_pred_flag");
          }
        }
        if (params->temporal_pred_flag[plane]) {
          return fail(error,
                      "temporal frame-filter prediction is outside this module");
        }
        params->rst_ref_pic_idx[plane] = 0;
        if (plane == 0) {
          std::uint32_t idx = 0;
          if (!reader.ReadBits(3, &idx)) {
            return fail(error, "truncated num_filter_classes_idx");
          }
          params->num_filter_classes_idx = static_cast<int>(idx);
          params->num_filter_classes = kDecodeNumFilterClasses[idx];
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
  if (uses_luma_lr && !ReadLrSizeFlags(reader, context.sb_size,
                                       &params->luma_shift, error)) {
    return false;
  }
  if (uses_chroma_lr && !ReadLrSizeFlags(reader, context.sb_size,
                                         &params->chroma_shift, error)) {
    return false;
  }

  if (!params->frame_filters_on[0]) {
    params->num_filter_classes = 1;
    params->num_filter_classes_idx = 0;
    params->subclass_lookup_count = 0;
  } else if (params->num_filter_classes > 1) {
    const std::uint8_t* lookup = PcWienerSubclassLookup(
        FilterSetIndex(context.base_q_idx), params->num_filter_classes_idx);
    if (lookup == nullptr) return fail(error, "invalid PC-Wiener lookup");
    for (int i = 0; i < kNumLutClasses; ++i) params->subclass_lookup[i] = lookup[i];
    params->subclass_lookup_count = kNumLutClasses;
  }

  for (int plane = 0; plane < context.num_planes; ++plane) {
    if (!params->frame_filters_on[plane] ||
        params->temporal_pred_flag[plane]) {
      continue;
    }
    if (!ReadFrameFilters(reader, context, params, plane, error)) return false;
  }
  return true;
}

std::size_t LrParamsBits(LrParams* params, const LrHeaderContext& context,
                         std::string* error) {
  BitWriter writer;
  return WriteLrParams(writer, params, context, error);
}

// ---------------------------------------------------------------------------
// §5.20.10.5 read_lr_unit() / §5.20.10.6 read_wienerns_filter()
// ---------------------------------------------------------------------------

bool InitTileState(TileState* state, const LrParams& params, int mi_cols,
                   int mi_rows, int sub_x, int sub_y, std::string* error) {
  if (state == nullptr || mi_cols <= 0 || mi_rows <= 0) {
    return fail(error, "invalid LR tile geometry");
  }
  *state = TileState{};
  state->params = params;
  state->mi_cols = mi_cols;
  state->mi_rows = mi_rows;
  state->sub_x = sub_x;
  state->sub_y = sub_y;

  CopyCdf(state->use_wiener_ns, rcls::kUseWienerNsCdf, 3);
  CopyCdf(state->use_pc_wiener, rcls::kUsePcWienerCdf, 3);
  for (int tool = 0; tool < 3; ++tool) {
    for (int plane = 0; plane < 3; ++plane) {
      CopyCdf(state->flex_restoration_type[tool][plane],
              rcls::kFlexRestorationTypeCdf[tool][plane], 3);
    }
  }
  CopyCdf(state->wiener_ns_length[0], rcls::kWienerNsLengthCdf[0], 3);
  CopyCdf(state->wiener_ns_length[1], rcls::kWienerNsLengthCdf[1], 3);
  CopyCdf(state->wiener_ns_uv_sym, rcls::kWienerNsUvSymCdf, 3);
  CopyCdf(state->wiener_ns_base, rcls::kWienerNsBaseCdf, 5);

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
    const int sx = plane == 0 ? 0 : sub_x;
    const int sy = plane == 0 ? 0 : sub_y;
    state->unit_cols[plane] =
        CountUnits(size, (mi_cols * kMiSize) >> sx);
    state->unit_rows[plane] =
        CountUnits(size, (mi_rows * kMiSize) >> sy);
    const std::size_t units =
        static_cast<std::size_t>(state->unit_cols[plane]) *
        static_cast<std::size_t>(state->unit_rows[plane]);
    state->unit_type[plane].assign(units,
                                   static_cast<std::uint8_t>(kRestoreNone));
    state->unit_coeffs[plane].assign(units * kWienerNsCoeffStride, 0);
    for (int c = 0; c < kWienerNsClasses; ++c) {
      state->bank_ptr[plane][c] = 0;
      state->bank_size[plane][c] = 0;
      for (int bank = 0; bank < kLrBankSize; ++bank) {
        for (int j = 0; j < kWienerNsCoeffStride; ++j) {
          const int min = rest::kWienerNsTapsMin[plane == 0 ? 0 : 1][j];
          const int k = rest::kWienerNsTapsK[plane == 0 ? 0 : 1][j];
          state->ref_bank[plane][c][bank][j] =
              static_cast<std::int16_t>(min + ((1 << k) >> 1));
        }
      }
    }
  }
  return true;
}

namespace {

bool WriteUnitFilter(av2::Encoder& encoder, TileState& state, int plane,
                     int unit_row, int unit_col, const UnitDecision& unit,
                     std::string* error) {
  const bool chroma = plane > 0;
  const int n_coeffs = NumCoeffs(plane);
  const int num_subsets = NumSubsets(plane);
  const int cls = 0;
  const int old_size = state.bank_size[plane][cls];
  const int old_ptr = state.bank_ptr[plane][cls];

  if (!RawBit(encoder, unit.merged, error)) return false;
  int k_sel = unit.use_bank_k;
  if (k_sel < 0) k_sel = old_size <= 1 ? 0 : old_size - 1;
  const int max_k = std::max(1, old_size);
  if (k_sel < 0 || k_sel >= max_k) {
    return fail(error, "use_bank_k outside the current bank");
  }
  for (int k = 0; k < old_size - 1; ++k) {
    const bool bit = k == k_sel;
    if (!RawBit(encoder, bit, error)) return false;
    if (bit) break;
  }
  const int ref_slot = (old_ptr - k_sel + kLrBankSize) & (kLrBankSize - 1);
  const std::size_t base =
      (static_cast<std::size_t>(unit_row) *
           static_cast<std::size_t>(state.unit_cols[plane]) +
       static_cast<std::size_t>(unit_col)) *
      kWienerNsCoeffStride;

  if (unit.merged) {
    if (state.bank_size[plane][cls] == 0) state.bank_size[plane][cls] = 1;
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
    if (!Symbol(encoder, &state.wiener_ns_length[std::min(plane, 1)][0], 2,
                len ? 1 : 0, "wiener_ns_length", error)) {
      return false;
    }
    if (!len) break;
  }
  bool uv_sym = false;
  if (chroma && unit.subset > 0) {
    uv_sym = unit.uv_sym;
    if (!Symbol(encoder, &state.wiener_ns_uv_sym[0], 2, uv_sym ? 1 : 0,
                "wiener_ns_uv_sym", error)) {
      return false;
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
    const int min = rest::kWienerNsTapsMin[chroma ? 1 : 0][j];
    const int k = rest::kWienerNsTapsK[chroma ? 1 : 0][j];
    std::int16_t v = state.ref_bank[plane][cls][ref_slot][j];
    if (kTapsPresent[chroma ? 1 : 0][unit.subset][j] != 0) {
      const int target = unit.coeffs[j];
      if (target < min || target >= min + (1 << k)) {
        return fail(error, "per-unit Wiener coefficient outside its tap range");
      }
      if (!EncodeSigned4Part(encoder, state.wiener_ns_base, min, k,
                             state.ref_bank[plane][cls][ref_slot][j], target,
                             error)) {
        return false;
      }
      v = static_cast<std::int16_t>(target);
    } else {
      v = 0;
    }
    state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = v;
    state.ref_bank[plane][cls][new_ptr][j] = v;
    if (chroma && unit.subset > 0 && uv_sym && j >= kWienerNsShortCoeffs) {
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

bool ReadUnitFilter(av2::Decoder& decoder, TileState& state, int plane,
                    int unit_row, int unit_col, UnitDecision* unit,
                    std::string* error) {
  const bool chroma = plane > 0;
  const int n_coeffs = NumCoeffs(plane);
  const int num_subsets = NumSubsets(plane);
  const int cls = 0;
  const int old_size = state.bank_size[plane][cls];
  const int old_ptr = state.bank_ptr[plane][cls];

  bool merged = false;
  if (!RawBit(decoder, &merged, error)) return false;
  int k_sel = 0;
  for (int k = 0; k < old_size - 1; ++k) {
    bool bit = false;
    if (!RawBit(decoder, &bit, error)) return false;
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
    if (state.bank_size[plane][cls] == 0) state.bank_size[plane][cls] = 1;
    for (int j = 0; j < n_coeffs; ++j) {
      const std::int16_t v = state.ref_bank[plane][cls][ref_slot][j];
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = v;
      if (unit != nullptr) unit->coeffs[j] = v;
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
    if (!Symbol(decoder, &state.wiener_ns_length[std::min(plane, 1)][0], 2,
                &len, "wiener_ns_length", error)) {
      return false;
    }
    if (len == 0) break;
    ++subset;
  }
  bool uv_sym = false;
  if (chroma && subset > 0) {
    int sym = 0;
    if (!Symbol(decoder, &state.wiener_ns_uv_sym[0], 2, &sym,
                "wiener_ns_uv_sym", error)) {
      return false;
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
    const int min = rest::kWienerNsTapsMin[chroma ? 1 : 0][j];
    const int k = rest::kWienerNsTapsK[chroma ? 1 : 0][j];
    std::int16_t v = state.ref_bank[plane][cls][ref_slot][j];
    if (kTapsPresent[chroma ? 1 : 0][subset][j] != 0) {
      int target = 0;
      if (!DecodeSigned4Part(decoder, state.wiener_ns_base, min, k,
                             state.ref_bank[plane][cls][ref_slot][j], &target,
                             error)) {
        return false;
      }
      v = static_cast<std::int16_t>(target);
    } else {
      v = 0;
    }
    state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] = v;
    state.ref_bank[plane][cls][new_ptr][j] = v;
    if (unit != nullptr) unit->coeffs[j] = v;
    if (chroma && subset > 0 && uv_sym && j >= kWienerNsShortCoeffs) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j) + 1] = v;
      state.ref_bank[plane][cls][new_ptr][j + 1] = v;
      if (unit != nullptr) unit->coeffs[j + 1] = v;
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

bool WriteUnit(av2::Encoder& encoder, TileState& state, int plane,
               int unit_row, int unit_col, const UnitDecision& decision,
               std::string* error) {
  if (plane < 0 || plane > 2 || unit_row < 0 || unit_col < 0 ||
      unit_row >= state.unit_rows[plane] ||
      unit_col >= state.unit_cols[plane]) {
    return fail(error, "LR unit outside the frame grid");
  }
  const int frame_type = state.params.frame_type[plane];
  int type = kRestoreNone;
  if (frame_type == kRestoreNone) {
    if (decision.type != kRestoreNone) {
      return fail(error, "LR unit written for a RESTORE_NONE plane");
    }
  } else if (frame_type == kRestorePcWiener) {
    if (decision.type != kRestoreNone &&
        decision.type != kRestorePcWiener) {
      return fail(error, "PC-Wiener unit must be on or off");
    }
    const bool on = decision.type == kRestorePcWiener;
    if (!Symbol(encoder, &state.use_pc_wiener[0], 2, on ? 1 : 0,
                "use_pc_wiener", error)) {
      return false;
    }
    type = on ? kRestorePcWiener : kRestoreNone;
  } else if (frame_type == kRestoreWienerNs) {
    if (decision.type != kRestoreNone &&
        decision.type != kRestoreWienerNs) {
      return fail(error, "WIENER_NONSEP unit must be on or off");
    }
    const bool on = decision.type == kRestoreWienerNs;
    if (!Symbol(encoder, &state.use_wiener_ns[0], 2, on ? 1 : 0,
                "use_wiener_ns", error)) {
      return false;
    }
    type = on ? kRestoreWienerNs : kRestoreNone;
  } else if (frame_type == kRestoreSwitchable) {
    if (decision.type != kRestoreNone &&
        decision.type != kRestorePcWiener &&
        decision.type != kRestoreWienerNs) {
      return fail(error, "invalid switchable LR unit type");
    }
    type = kRestoreWienerNs;
    for (int tool = 0; tool < kRestoreSwitchable - 1; ++tool) {
      const bool on = decision.type == tool;
      if (!Symbol(encoder, &state.flex_restoration_type[tool][plane][0], 2,
                  on ? 1 : 0, "flex_restoration_type", error)) {
        return false;
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
  if (type != kRestoreWienerNs) return true;
  if (state.params.frame_filters_on[plane]) {
    const std::size_t base = index * kWienerNsCoeffStride;
    const int n = NumCoeffs(plane);
    for (int j = 0; j < n; ++j) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] =
          state.params.frame_coeffs[plane][0][j];
    }
    return true;
  }
  return WriteUnitFilter(encoder, state, plane, unit_row, unit_col, decision,
                         error);
}

bool ReadUnit(av2::Decoder& decoder, TileState& state, int plane,
              int unit_row, int unit_col, UnitDecision* decision,
              std::string* error) {
  if (plane < 0 || plane > 2 || unit_row < 0 || unit_col < 0 ||
      unit_row >= state.unit_rows[plane] ||
      unit_col >= state.unit_cols[plane]) {
    return fail(error, "LR unit outside the frame grid");
  }
  UnitDecision local;
  if (decision == nullptr) decision = &local;
  *decision = UnitDecision{};
  const int frame_type = state.params.frame_type[plane];
  int type = kRestoreNone;
  if (frame_type == kRestoreNone) {
    type = kRestoreNone;
  } else if (frame_type == kRestorePcWiener) {
    int on = 0;
    if (!Symbol(decoder, &state.use_pc_wiener[0], 2, &on, "use_pc_wiener",
                error)) {
      return false;
    }
    type = on != 0 ? kRestorePcWiener : kRestoreNone;
  } else if (frame_type == kRestoreWienerNs) {
    int on = 0;
    if (!Symbol(decoder, &state.use_wiener_ns[0], 2, &on, "use_wiener_ns",
                error)) {
      return false;
    }
    type = on != 0 ? kRestoreWienerNs : kRestoreNone;
  } else if (frame_type == kRestoreSwitchable) {
    type = kRestoreWienerNs;
    for (int tool = 0; tool < kRestoreSwitchable - 1; ++tool) {
      int on = 0;
      if (!Symbol(decoder, &state.flex_restoration_type[tool][plane][0], 2, &on,
                  "flex_restoration_type", error)) {
        return false;
      }
      if (on != 0) {
        type = tool;
        break;
      }
    }
  } else {
    return fail(error, "unsupported FrameRestorationType");
  }
  decision->type = type;
  const std::size_t index =
      static_cast<std::size_t>(unit_row) *
          static_cast<std::size_t>(state.unit_cols[plane]) +
      static_cast<std::size_t>(unit_col);
  state.unit_type[plane][index] = static_cast<std::uint8_t>(type);
  if (type != kRestoreWienerNs) return true;
  if (state.params.frame_filters_on[plane]) {
    const std::size_t base = index * kWienerNsCoeffStride;
    const int n = NumCoeffs(plane);
    for (int j = 0; j < n; ++j) {
      state.unit_coeffs[plane][base + static_cast<std::size_t>(j)] =
          state.params.frame_coeffs[plane][0][j];
      decision->coeffs[j] = state.params.frame_coeffs[plane][0][j];
    }
    return true;
  }
  return ReadUnitFilter(decoder, state, plane, unit_row, unit_col, decision,
                        error);
}

bool UnitRangeFor(TileState& state, int plane, int mi_row, int mi_col, int w4,
                  int h4, int tile_mi_row_start, int tile_mi_col_start,
                  UnitRange* range, std::string* error) {
  if (range == nullptr || plane < 0 || plane > 2) {
    return fail(error, "invalid read_lr argument");
  }
  *range = UnitRange{};
  if (state.params.frame_type[plane] == kRestoreNone) return true;
  const int sub_x = plane == 0 ? 0 : state.sub_x;
  const int sub_y = plane == 0 ? 0 : state.sub_y;
  const int unit_size = state.unit_size[plane];
  if (unit_size <= 0) return fail(error, "invalid LoopRestorationSize");
  const int mi_cols = state.mi_cols - tile_mi_col_start;
  const int mi_rows = state.mi_rows - tile_mi_row_start;
  const int c = mi_col - tile_mi_col_start;
  const int r = mi_row - tile_mi_row_start;
  if (c < 0 || r < 0) return fail(error, "read_lr outside the tile");
  const int step_y = kMiSize >> sub_y;
  const int step_x = kMiSize >> sub_x;
  if (step_y <= 0 || step_x <= 0) return fail(error, "invalid subsampling");
  const int unit_rows = CountUnits(unit_size, (mi_rows * kMiSize) >> sub_y);
  const int unit_cols = CountUnits(unit_size, (mi_cols * kMiSize) >> sub_x);
  const int lr_row_offset =
      (tile_mi_row_start * kMiSize >> sub_y) / unit_size;
  const int lr_col_offset =
      (tile_mi_col_start * kMiSize >> sub_x) / unit_size;
  const int row_start = (r * step_y + unit_size - 1) / unit_size;
  const int row_end =
      std::min(unit_rows, ((r + h4) * step_y + unit_size - 1) / unit_size);
  const int col_start = (c * step_x + unit_size - 1) / unit_size;
  const int col_end =
      std::min(unit_cols, ((c + w4) * step_x + unit_size - 1) / unit_size);
  range->row_begin = row_start + lr_row_offset;
  range->row_end = row_end + lr_row_offset;
  range->col_begin = col_start + lr_col_offset;
  range->col_end = col_end + lr_col_offset;
  return true;
}

bool WriteBlock(av2::Encoder& encoder, TileState& state, int mi_row,
                int mi_col, int w4, int h4, int tile_mi_row_start,
                int tile_mi_col_start,
                const std::vector<UnitDecision>& decisions,
                std::string* error) {
  int cursor = 0;
  for (int plane = 0; plane < 3; ++plane) {
    UnitRange range;
    if (!UnitRangeFor(state, plane, mi_row, mi_col, w4, h4, tile_mi_row_start,
                      tile_mi_col_start, &range, error)) {
      return false;
    }
    for (int r = range.row_begin; r < range.row_end; ++r) {
      for (int c = range.col_begin; c < range.col_end; ++c) {
        if (cursor >= static_cast<int>(decisions.size())) {
          return fail(error, "read_lr decision count too small");
        }
        if (!WriteUnit(encoder, state, plane, r, c, decisions[cursor],
                       error)) {
          return false;
        }
        ++cursor;
      }
    }
  }
  if (cursor != static_cast<int>(decisions.size())) {
    return fail(error, "read_lr decision count too large");
  }
  return true;
}

bool ReadBlock(av2::Decoder& decoder, TileState& state, int mi_row, int mi_col,
               int w4, int h4, int tile_mi_row_start, int tile_mi_col_start,
               std::vector<UnitDecision>* decisions, std::string* error) {
  if (decisions != nullptr) decisions->clear();
  for (int plane = 0; plane < 3; ++plane) {
    UnitRange range;
    if (!UnitRangeFor(state, plane, mi_row, mi_col, w4, h4, tile_mi_row_start,
                      tile_mi_col_start, &range, error)) {
      return false;
    }
    for (int r = range.row_begin; r < range.row_end; ++r) {
      for (int c = range.col_begin; c < range.col_end; ++c) {
        UnitDecision decision;
        if (!ReadUnit(decoder, state, plane, r, c, &decision, error)) {
          return false;
        }
        if (decisions != nullptr) decisions->push_back(decision);
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// §7.20.4 pixel classification + filtering helpers
// ---------------------------------------------------------------------------

namespace {

void StripeBounds(int row, int mi_rows, int height, int* start, int* end) {
  const int luma_y = row * kMiSize;
  const int stripe_num = (luma_y + 8) / 64;
  const int unclipped_start = stripe_num * 64 - 8;
  const int unclipped_end = unclipped_start + 64;
  *start = std::max(0, unclipped_start);
  *end = std::min(height - 1, unclipped_end - 1);
  (void)mi_rows;
}

struct PcLumaBuffers {
  std::vector<std::uint16_t> curr;
  std::vector<std::uint16_t> cdef;
  int width = 0;
  int height = 0;
};

bool MakePcLuma(const DecisionPlanes& planes, PcLumaBuffers* out,
                std::string* error) {
  if (out == nullptr || planes.width <= 0 || planes.height <= 0) {
    return fail(error, "invalid decision planes");
  }
  if (planes.bit_depth != 8) {
    return fail(error, "PC-Wiener stream module is 8-bit only");
  }
  out->width = planes.width;
  out->height = planes.height;
  const std::size_t count =
      static_cast<std::size_t>(out->width) * out->height;
  out->curr.assign(count, 0);
  out->cdef.assign(count, 0);
  if (planes.curr[0] == nullptr || planes.cdef[0] == nullptr) {
    return fail(error, "missing luma planes");
  }
  for (int y = 0; y < out->height; ++y) {
    for (int x = 0; x < out->width; ++x) {
      out->curr[static_cast<std::size_t>(y) * out->width + x] =
          planes.curr[0][static_cast<std::ptrdiff_t>(y) *
                              planes.curr_stride[0] +
                          x];
      out->cdef[static_cast<std::size_t>(y) * out->width + x] =
          planes.cdef[0][static_cast<std::ptrdiff_t>(y) *
                              planes.cdef_stride[0] +
                          x];
    }
  }
  return true;
}

void SetupPcFrame(const DecisionPlanes& planes, PcLumaBuffers* luma,
                  int mi_cols, int mi_rows, std::uint8_t* filter_class,
                  int filter_class_stride, std::uint16_t* dst, int dst_stride,
                  std::vector<std::uint8_t>* zero_skip,
                  spec::PcWienerFrame* frame, spec::PcWienerConfig* cfg) {
  *frame = spec::PcWienerFrame{};
  frame->src.curr = luma->curr.data();
  frame->src.cdef = luma->cdef.data();
  frame->src.curr_stride = luma->width;
  frame->src.cdef_stride = luma->width;
  frame->src.luma_start_x = 0;
  frame->src.luma_end_x = luma->width - 1;
  frame->src.luma_start_y = 0;
  frame->src.luma_end_y = luma->height - 1;
  frame->lr = dst;
  frame->lr_stride = dst_stride;
  frame->filter_class = filter_class;
  frame->filter_class_stride = filter_class_stride;

  zero_skip->assign(static_cast<std::size_t>(mi_cols) * mi_rows, 0);
  *cfg = spec::PcWienerConfig{};
  cfg->base_q_idx = planes.base_q_idx;
  cfg->bit_depth = planes.bit_depth;
  cfg->mi_col_end = mi_cols;
  cfg->mi_row_start = 0;
  cfg->mi_row_end = mi_rows;
  cfg->lr_tx_skip = planes.lr_tx_skip != nullptr ? planes.lr_tx_skip
                                                 : zero_skip->data();
  cfg->lr_tx_skip_stride =
      planes.lr_tx_skip != nullptr ? planes.lr_tx_skip_stride : mi_cols;
}

int UnitIndexForBlock(int row, int col, int unit_size, int unit_rows,
                      int unit_cols) {
  const int unit_row =
      std::min(unit_rows - 1, ((row * kMiSize + 8)) / unit_size);
  const int unit_col =
      std::min(unit_cols - 1, (col * kMiSize) / unit_size);
  return unit_row * unit_cols + unit_col;
}

void FilterAllBlocks(const DecisionPlanes& planes, PcLumaBuffers* luma,
                     int mi_cols, int mi_rows, std::uint16_t* dst,
                     int dst_stride) {
  std::vector<std::uint8_t> zero_skip;
  std::vector<std::uint8_t> class_grid(
      static_cast<std::size_t>(mi_cols) * mi_rows, 0);
  spec::PcWienerFrame frame;
  spec::PcWienerConfig cfg;
  SetupPcFrame(planes, luma, mi_cols, mi_rows, class_grid.data(), mi_cols, dst,
               dst_stride, &zero_skip, &frame, &cfg);
  for (int row = 0; row < mi_rows; ++row) {
    int stripe_start = 0;
    int stripe_end = 0;
    StripeBounds(row, mi_rows, luma->height, &stripe_start, &stripe_end);
    frame.src.luma_stripe_start_y = stripe_start;
    frame.src.luma_stripe_end_y = stripe_end;
    for (int col = 0; col < mi_cols; ++col) {
      const int x = col * kMiSize;
      const int y = row * kMiSize;
      const int w = std::min(kMiSize, luma->width - x);
      const int h = std::min(kMiSize, luma->height - y);
      if (w <= 0 || h <= 0) continue;
      spec::PcWienerFilterBlock(frame, cfg, x, y, w, h);
    }
  }
}

// §7.20.3 luma evaluation for one 4x4 block with a 16-entry coefficient set.
std::int64_t WienerBlockSse(const rest::LrCoreFrame& core, int row, int col,
                            const std::int16_t coeffs[kWienerNsCoeffStride],
                            const std::uint8_t* target, int target_stride,
                            int width, int height) {
  rest::LrBlock block;
  const int x = col * kMiSize;
  const int y = row * kMiSize;
  if (!rest::lr_block_context(core, 0, row, col, &block)) return 0;
  int w = std::min(kMiSize, width - x);
  int h = std::min(kMiSize, height - y);
  if (w <= 0 || h <= 0) return 0;
  std::int64_t sse = 0;
  for (int r = 0; r < h; ++r) {
    for (int c = 0; c < w; ++c) {
      const int sx = x + c;
      const int sy = y + r;
      const int m = rest::lr_get_source_sample(core, block, 0, sx, sy);
      int s = m << rest::kWienerNsPrecBits;
      for (int i = 0; i < rest::kWienerNsTapsY; ++i) {
        const int dy = rest::kWienerNsConfigY[i][0];
        const int dx = rest::kWienerNsConfigY[i][1];
        const int idx = rest::kWienerNsConfigY[i][2];
        const int diff =
            rest::lr_get_source_sample(core, block, 0, sx + dx, sy + dy) - m;
        s += diff * coeffs[idx];
      }
      const int v = rest::round2(s, rest::kWienerNsPrecBits);
      const int clipped = rest::clip3(0, 255, v);
      const int t =
          target[static_cast<std::ptrdiff_t>(sy) * target_stride + sx];
      const int d = clipped - t;
      sse += static_cast<std::int64_t>(d) * d;
    }
  }
  return sse;
}

void BuildCoreForEval(const DecisionPlanes& planes, rest::LrCoreFrame* core,
                      int mi_cols, int mi_rows, int luma_shift,
                      const std::uint8_t* filter_class,
                      const std::uint8_t* subclass_lookup,
                      int num_classes) {
  *core = rest::LrCoreFrame{};
  core->mi_cols = mi_cols;
  core->mi_rows = mi_rows;
  core->sub_x = planes.sub_x;
  core->sub_y = planes.sub_y;
  core->bit_depth = planes.bit_depth;
  core->unit_size[0] = kRestorationTileSizeMax >> luma_shift;
  core->frame_type[0] = kRestoreWienerNs;
  core->frame_filters_on[0] = 1;
  core->num_filter_classes = num_classes > 1 ? num_classes : 1;
  core->filter_class_grid = filter_class;
  core->subclass_lookup = subclass_lookup;
  core->filter_class_count = kNumLutClasses;
  core->plane[0].curr = planes.curr[0];
  core->plane[0].curr_stride = planes.curr_stride[0];
  core->plane[0].cdef = planes.cdef[0];
  core->plane[0].cdef_stride = planes.cdef_stride[0];
  core->plane[0].unit_cols =
      CountUnits(core->unit_size[0], planes.width);
  core->plane[0].unit_rows =
      CountUnits(core->unit_size[0], planes.height);
}

}  // namespace

void BuildFilterClassGrid(const DecisionPlanes& planes,
                          std::vector<std::uint8_t>* filter_class) {
  if (filter_class == nullptr) return;
  PcLumaBuffers luma;
  std::string error;
  if (!MakePcLuma(planes, &luma, &error)) {
    filter_class->clear();
    return;
  }
  const int mi_cols = (planes.width + kMiSize - 1) / kMiSize;
  const int mi_rows = (planes.height + kMiSize - 1) / kMiSize;
  filter_class->assign(static_cast<std::size_t>(mi_cols) * mi_rows, 0);
  std::vector<std::uint8_t> zero_skip;
  spec::PcWienerFrame frame;
  spec::PcWienerConfig cfg;
  SetupPcFrame(planes, &luma, mi_cols, mi_rows, filter_class->data(), mi_cols,
               nullptr, 0, &zero_skip, &frame, &cfg);
  for (int row = 0; row < mi_rows; ++row) {
    int stripe_start = 0;
    int stripe_end = 0;
    StripeBounds(row, mi_rows, luma.height, &stripe_start, &stripe_end);
    frame.src.luma_stripe_start_y = stripe_start;
    frame.src.luma_stripe_end_y = stripe_end;
    for (int col = 0; col < mi_cols; ++col) {
      const int x = col * kMiSize;
      const int y = row * kMiSize;
      const int cls = spec::PcWienerClass(frame, cfg, x, y);
      (*filter_class)[static_cast<std::size_t>(row) * mi_cols + col] =
          static_cast<std::uint8_t>(cls);
    }
  }
}

int ApplyPerUnitPcWiener(const DecisionPlanes& planes,
                         const std::vector<std::uint8_t>& unit_type,
                         int unit_size, std::uint8_t* dst, int dst_stride) {
  if (dst == nullptr || unit_size <= 0) return 0;
  PcLumaBuffers luma;
  std::string error;
  if (!MakePcLuma(planes, &luma, &error)) return 0;
  const int mi_cols = (planes.width + kMiSize - 1) / kMiSize;
  const int mi_rows = (planes.height + kMiSize - 1) / kMiSize;
  const int unit_cols = CountUnits(unit_size, planes.width);
  const int unit_rows = CountUnits(unit_size, planes.height);
  if (static_cast<int>(unit_type.size()) != unit_cols * unit_rows) return 0;

  std::vector<std::uint16_t> out(
      static_cast<std::size_t>(planes.width) * planes.height, 0);
  for (int y = 0; y < planes.height; ++y) {
    for (int x = 0; x < planes.width; ++x) {
      out[static_cast<std::size_t>(y) * planes.width + x] =
          luma.cdef[static_cast<std::size_t>(y) * planes.width + x];
    }
  }
  std::vector<std::uint8_t> zero_skip;
  std::vector<std::uint8_t> class_grid(
      static_cast<std::size_t>(mi_cols) * mi_rows, 0);
  spec::PcWienerFrame frame;
  spec::PcWienerConfig cfg;
  SetupPcFrame(planes, &luma, mi_cols, mi_rows, class_grid.data(), mi_cols,
               out.data(), planes.width, &zero_skip, &frame, &cfg);
  int filtered = 0;
  for (int row = 0; row < mi_rows; ++row) {
    int stripe_start = 0;
    int stripe_end = 0;
    StripeBounds(row, mi_rows, planes.height, &stripe_start, &stripe_end);
    frame.src.luma_stripe_start_y = stripe_start;
    frame.src.luma_stripe_end_y = stripe_end;
    for (int col = 0; col < mi_cols; ++col) {
      const int unit_index =
          UnitIndexForBlock(row, col, unit_size, unit_rows, unit_cols);
      if (unit_index < 0 ||
          unit_index >= static_cast<int>(unit_type.size()) ||
          unit_type[unit_index] != kRestorePcWiener) {
        continue;
      }
      const int x = col * kMiSize;
      const int y = row * kMiSize;
      const int w = std::min(kMiSize, planes.width - x);
      const int h = std::min(kMiSize, planes.height - y);
      if (w <= 0 || h <= 0) continue;
      spec::PcWienerFilterBlock(frame, cfg, x, y, w, h);
      ++filtered;
    }
  }
  for (int y = 0; y < planes.height; ++y) {
    for (int x = 0; x < planes.width; ++x) {
      const int v = out[static_cast<std::size_t>(y) * planes.width + x];
      dst[static_cast<std::ptrdiff_t>(y) * dst_stride + x] =
          static_cast<std::uint8_t>(std::max(0, std::min(255, v)));
    }
  }
  return filtered;
}

int ApplyFrameClassWiener(const DecisionPlanes& planes,
                          const std::int16_t frame_coeffs[kWienerNsClasses]
                                                    [kWienerNsCoeffStride],
                          const std::uint8_t* filter_class,
                          const std::uint8_t* subclass_lookup,
                          int subclass_count, std::uint8_t* dst,
                          int dst_stride) {
  if (dst == nullptr || filter_class == nullptr) return 0;
  const int mi_cols = (planes.width + kMiSize - 1) / kMiSize;
  const int mi_rows = (planes.height + kMiSize - 1) / kMiSize;
  rest::LrCoreFrame core;
  BuildCoreForEval(planes, &core, mi_cols, mi_rows, planes.luma_shift,
                   filter_class, subclass_lookup, subclass_count);
  core.plane[0].dst = dst;
  core.plane[0].dst_stride = dst_stride;
  core.plane[0].frame_coeffs = &frame_coeffs[0][0];
  rest::lr_copy_plane(core, 0);
  return rest::lr_plane(core, 0);
}

namespace {

double LumaSse(const std::uint8_t* a, int stride_a, const std::uint8_t* b,
               int stride_b, int width, int height) {
  double sse = 0.0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int d = static_cast<int>(a[static_cast<std::ptrdiff_t>(y) *
                                          stride_a +
                                      x]) -
                    static_cast<int>(b[static_cast<std::ptrdiff_t>(y) *
                                          stride_b +
                                      x]);
      sse += static_cast<double>(d) * d;
    }
  }
  return sse;
}

}  // namespace

bool DecidePerUnitPcWiener(const DecisionPlanes& planes, double lambda,
                           DecisionResult* result, std::string* error) {
  if (result == nullptr) return fail(error, "null decision result");
  *result = DecisionResult{};
  if (planes.width <= 0 || planes.height <= 0 || planes.curr[0] == nullptr ||
      planes.cdef[0] == nullptr || planes.target[0] == nullptr) {
    return fail(error, "invalid PC-Wiener decision input");
  }
  if (planes.bit_depth != 8) {
    return fail(error, "PC-Wiener stream module is 8-bit only");
  }
  const int mi_cols = (planes.width + kMiSize - 1) / kMiSize;
  const int mi_rows = (planes.height + kMiSize - 1) / kMiSize;
  BuildFilterClassGrid(planes, &result->filter_class);
  if (static_cast<int>(result->filter_class.size()) != mi_cols * mi_rows) {
    return fail(error, "classifier failed");
  }

  const int unit_size = kRestorationTileSizeMax >> planes.luma_shift;
  const int unit_cols = CountUnits(unit_size, planes.width);
  const int unit_rows = CountUnits(unit_size, planes.height);
  result->total_units = unit_cols * unit_rows;

  PcLumaBuffers luma;
  if (!MakePcLuma(planes, &luma, error)) return false;

  std::vector<std::uint16_t> filtered(
      static_cast<std::size_t>(planes.width) * planes.height, 0);
  FilterAllBlocks(planes, &luma, mi_cols, mi_rows, filtered.data(),
                  planes.width);

  std::vector<double> sse_on(static_cast<std::size_t>(result->total_units), 0.0);
  std::vector<double> sse_off(static_cast<std::size_t>(result->total_units),
                              0.0);
  result->identity_sse = 0.0;
  for (int row = 0; row < mi_rows; ++row) {
    for (int col = 0; col < mi_cols; ++col) {
      const int unit_index =
          UnitIndexForBlock(row, col, unit_size, unit_rows, unit_cols);
      const int x = col * kMiSize;
      const int y = row * kMiSize;
      const int w = std::min(kMiSize, planes.width - x);
      const int h = std::min(kMiSize, planes.height - y);
      for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
          const int sx = x + c;
          const int sy = y + r;
          const double t = planes.target[0][static_cast<std::ptrdiff_t>(sy) *
                                                planes.target_stride[0] +
                                            sx];
          const double base =
              planes.cdef[0][static_cast<std::ptrdiff_t>(sy) *
                                  planes.cdef_stride[0] +
                              sx];
          const double f =
              filtered[static_cast<std::size_t>(sy) * planes.width + sx];
          sse_off[static_cast<std::size_t>(unit_index)] +=
              (base - t) * (base - t);
          sse_on[static_cast<std::size_t>(unit_index)] +=
              (f - t) * (f - t);
        }
      }
    }
  }

  result->params = LrParams{};
  result->params.frame_type[0] = kRestorePcWiener;
  result->params.frame_type[1] = kRestoreNone;
  result->params.frame_type[2] = kRestoreNone;
  result->params.frame_filters_on[0] = false;
  result->params.luma_shift = planes.luma_shift;
  result->params.chroma_shift = planes.chroma_shift;

  result->unit_type[0].assign(static_cast<std::size_t>(result->total_units),
                              static_cast<std::uint8_t>(kRestoreNone));
  result->unit_type[1].assign(
      static_cast<std::size_t>(CountUnits(
          (kRestorationTileSizeMax >> std::max(planes.sub_x, planes.sub_y)) >>
              planes.chroma_shift,
          planes.width >> planes.sub_x)) *
          CountUnits(
              (kRestorationTileSizeMax >> std::max(planes.sub_x, planes.sub_y)) >>
                  planes.chroma_shift,
              planes.height >> planes.sub_y),
      0);
  result->unit_type[2] = result->unit_type[1];

  int active = 0;
  for (int i = 0; i < result->total_units; ++i) {
    const double bits_est = 1.0;
    const bool on = sse_on[static_cast<std::size_t>(i)] + lambda * bits_est <
                    sse_off[static_cast<std::size_t>(i)];
    if (on) {
      result->unit_type[0][static_cast<std::size_t>(i)] = kRestorePcWiener;
      ++active;
      result->filtered_sse += sse_on[static_cast<std::size_t>(i)];
    } else {
      result->filtered_sse += sse_off[static_cast<std::size_t>(i)];
    }
    result->identity_sse += sse_off[static_cast<std::size_t>(i)];
  }
  result->active_units = active;

  LrHeaderContext context;
  context.coded_lossless = false;
  context.enable_restoration = true;
  context.single_picture_header_flag = false;
  context.frame_is_intra = true;
  context.sb_size = 64;
  context.num_planes = 3;
  context.sub_x = planes.sub_x;
  context.sub_y = planes.sub_y;
  context.base_q_idx = planes.base_q_idx;
  context.lr_tools_disable[0][kRestorePcWiener] = false;
  context.lr_tools_disable[0][kRestoreWienerNs] = true;
  context.lr_tools_disable[1][kRestorePcWiener] = true;
  context.lr_tools_disable[1][kRestoreWienerNs] = true;

  LrParams params = result->params;
  result->header_bits = static_cast<int>(
      LrParamsBits(&params, context, error));
  if (result->header_bits == 0 && error != nullptr && !error->empty()) {
    return false;
  }
  result->params = params;

  std::vector<std::uint8_t> unit_type = result->unit_type[0];
  TileState state;
  if (!InitTileState(&state, params, mi_cols, mi_rows, planes.sub_x,
                     planes.sub_y, error)) {
    return false;
  }
  constexpr std::size_t kScratchBytes = 1U << 20;
  std::vector<std::uint8_t> scratch(kScratchBytes, 0);
  av2::Encoder encoder;
  if (encoder.init_symbol(scratch.data(), scratch.size()) != av2::Status::kOk) {
    return fail(error, "scratch encoder init failed");
  }
  for (int r = 0; r < unit_rows; ++r) {
    for (int c = 0; c < unit_cols; ++c) {
      UnitDecision decision;
      decision.type = unit_type[static_cast<std::size_t>(r) * unit_cols + c];
      if (!WriteUnit(encoder, state, 0, r, c, decision, error)) return false;
    }
  }
  std::size_t bytes = 0;
  if (encoder.exit_symbol(&bytes) != av2::Status::kOk) {
    return fail(error, "scratch encoder exit failed");
  }
  result->tile_bits = static_cast<int>(bytes * 8);
  return true;
}

bool DecideFrameClassPcWiener(const DecisionPlanes& planes, int num_classes,
                              double lambda, DecisionResult* result,
                              std::string* error) {
  if (result == nullptr) return fail(error, "null decision result");
  *result = DecisionResult{};
  if (planes.width <= 0 || planes.height <= 0 || planes.curr[0] == nullptr ||
      planes.cdef[0] == nullptr || planes.target[0] == nullptr) {
    return fail(error, "invalid PC-Wiener decision input");
  }
  if (num_classes < 2 || !ValidNumFilterClasses(num_classes)) {
    return fail(error, "NumFilterClasses must be 2..16 in the decode table");
  }
  const int num_classes_idx = EncodeNumFilterClassesIndex(num_classes);
  if (num_classes_idx < 0) return fail(error, "cannot encode NumFilterClasses");

  const int mi_cols = (planes.width + kMiSize - 1) / kMiSize;
  const int mi_rows = (planes.height + kMiSize - 1) / kMiSize;
  BuildFilterClassGrid(planes, &result->filter_class);
  if (static_cast<int>(result->filter_class.size()) != mi_cols * mi_rows) {
    return fail(error, "classifier failed");
  }
  const std::uint8_t* lookup = PcWienerSubclassLookup(
      FilterSetIndex(planes.base_q_idx), num_classes_idx);
  if (lookup == nullptr) return fail(error, "missing subclass table");
  for (int i = 0; i < kNumLutClasses; ++i) result->subclass_lookup[i] = lookup[i];
  result->subclass_lookup_count = kNumLutClasses;

  std::vector<std::vector<std::pair<int, int>>> cells(
      static_cast<std::size_t>(num_classes));
  for (int row = 0; row < mi_rows; ++row) {
    for (int col = 0; col < mi_cols; ++col) {
      const int cls =
          result->filter_class[static_cast<std::size_t>(row) * mi_cols + col];
      const int sub = lookup[cls];
      if (sub >= 0 && sub < num_classes) {
        cells[static_cast<std::size_t>(sub)].emplace_back(row, col);
      }
    }
  }

  LrParams params;
  params.frame_type[0] = kRestoreWienerNs;
  params.frame_type[1] = kRestoreNone;
  params.frame_type[2] = kRestoreNone;
  params.frame_filters_on[0] = true;
  params.num_filter_classes = num_classes;
  params.num_filter_classes_idx = num_classes_idx;
  params.luma_shift = planes.luma_shift;
  params.chroma_shift = planes.chroma_shift;
  for (int i = 0; i < kNumLutClasses; ++i) params.subclass_lookup[i] = lookup[i];
  params.subclass_lookup_count = kNumLutClasses;

  const int pc_group = kPcWienerBank - num_classes;
  if (pc_group <= 0) return fail(error, "empty translated PC-Wiener group");

  rest::LrCoreFrame core;
  BuildCoreForEval(planes, &core, mi_cols, mi_rows, planes.luma_shift,
                   result->filter_class.data(), lookup, num_classes);

  // Pick a translated filter per class.  Evaluate all candidates on a spread
  // of at most 96 cells, then compute the winning SSE exactly.
  for (int c = 0; c < num_classes; ++c) {
    const auto& class_cells = cells[static_cast<std::size_t>(c)];
    int best_k = 0;
    double best_sse = std::numeric_limits<double>::max();
    const int step = std::max(1, static_cast<int>(class_cells.size()) / 96);
    for (int k = 0; k < pc_group; ++k) {
      std::int16_t coeffs[kWienerNsCoeffStride];
      TranslatedPcWienerSet(k, coeffs);
      double sse = 0.0;
      for (int i = 0; i < static_cast<int>(class_cells.size()); i += step) {
        const int row = class_cells[static_cast<std::size_t>(i)].first;
        const int col = class_cells[static_cast<std::size_t>(i)].second;
        sse += static_cast<double>(WienerBlockSse(
            core, row, col, coeffs, planes.target[0],
            planes.target_stride[0], planes.width, planes.height));
      }
      if (sse < best_sse) {
        best_sse = sse;
        best_k = k;
      }
    }
    std::int16_t best_coeffs[kWienerNsCoeffStride];
    TranslatedPcWienerSet(best_k, best_coeffs);
    double exact = 0.0;
    for (const auto& cell : class_cells) {
      exact += static_cast<double>(
          WienerBlockSse(core, cell.first, cell.second, best_coeffs,
                         planes.target[0], planes.target_stride[0],
                         planes.width, planes.height));
    }
    params.frame_filter[0][c].match_index = num_classes + best_k;
    params.frame_filter[0][c].merged = true;
    for (int j = 0; j < kWienerNsCoeffStride; ++j) {
      params.frame_filter[0][c].coeffs[j] = best_coeffs[j];
    }
    (void)exact;
  }

  LrHeaderContext context;
  context.coded_lossless = false;
  context.enable_restoration = true;
  context.single_picture_header_flag = false;
  context.frame_is_intra = true;
  context.sb_size = 64;
  context.num_planes = 3;
  context.sub_x = planes.sub_x;
  context.sub_y = planes.sub_y;
  context.base_q_idx = planes.base_q_idx;
  context.lr_tools_disable[0][kRestorePcWiener] = false;
  context.lr_tools_disable[0][kRestoreWienerNs] = false;
  context.lr_tools_disable[1][kRestorePcWiener] = true;
  // lr_tools_uv_present == 0: chroma Wiener inherits the luma flag.
  context.lr_tools_disable[1][kRestoreWienerNs] =
      context.lr_tools_disable[0][kRestoreWienerNs];

  LrParams written = params;
  result->header_bits = static_cast<int>(
      LrParamsBits(&written, context, error));
  if (result->header_bits == 0 && error != nullptr && !error->empty()) {
    return false;
  }
  params = written;

  // Exact identity/filtered SSE for the chosen class filters.
  result->identity_sse = 0.0;
  result->filtered_sse = 0.0;
  std::vector<std::uint8_t> filtered(
      static_cast<std::size_t>(planes.width) * planes.height, 0);
  ApplyFrameClassWiener(planes, params.frame_coeffs[0], result->filter_class.data(),
                        lookup, num_classes, filtered.data(), planes.width);
  result->identity_sse = LumaSse(planes.cdef[0], planes.cdef_stride[0],
                                 planes.target[0], planes.target_stride[0],
                                 planes.width, planes.height);
  result->filtered_sse = LumaSse(filtered.data(), planes.width, planes.target[0],
                                 planes.target_stride[0], planes.width,
                                 planes.height);

  const int unit_size = kRestorationTileSizeMax >> planes.luma_shift;
  const int unit_cols = CountUnits(unit_size, planes.width);
  const int unit_rows = CountUnits(unit_size, planes.height);
  result->total_units = unit_cols * unit_rows;
  result->unit_type[0].assign(static_cast<std::size_t>(result->total_units),
                              static_cast<std::uint8_t>(kRestoreNone));
  const bool turn_on = result->filtered_sse + lambda * 1.0 <
                       result->identity_sse;
  if (turn_on) {
    for (auto& type : result->unit_type[0]) {
      type = static_cast<std::uint8_t>(kRestoreWienerNs);
    }
    result->active_units = result->total_units;
  }
  // Chroma is unused in this mode (frame type NONE).
  result->unit_type[1].clear();
  result->unit_type[2].clear();

  TileState state;
  if (!InitTileState(&state, params, mi_cols, mi_rows, planes.sub_x,
                     planes.sub_y, error)) {
    return false;
  }
  constexpr std::size_t kScratchBytes = 1U << 20;
  std::vector<std::uint8_t> scratch(kScratchBytes, 0);
  av2::Encoder encoder;
  if (encoder.init_symbol(scratch.data(), scratch.size()) != av2::Status::kOk) {
    return fail(error, "scratch encoder init failed");
  }
  for (int r = 0; r < unit_rows; ++r) {
    for (int c = 0; c < unit_cols; ++c) {
      UnitDecision decision;
      decision.type =
          result->unit_type[0][static_cast<std::size_t>(r) * unit_cols + c];
      if (!WriteUnit(encoder, state, 0, r, c, decision, error)) return false;
    }
  }
  std::size_t bytes = 0;
  if (encoder.exit_symbol(&bytes) != av2::Status::kOk) {
    return fail(error, "scratch encoder exit failed");
  }
  result->tile_bits = static_cast<int>(bytes * 8);
  result->params = params;
  result->used_frame_class_mode = true;
  result->num_filter_classes = num_classes;
  return true;
}

}  // namespace av2enc::filters::pc_wiener_stream
