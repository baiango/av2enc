#include "codec/filters/tables/film_grain.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "codec/filters/tables/film_grain_tables.h"
#include "codec/filters/tables/spec_math.h"

namespace av2enc::filters::spec {
namespace {

using namespace av2enc::filters::spec_tables;

// §7.21.7.2 random number process.
class RandomRegister {
 public:
  explicit RandomRegister(std::uint16_t seed) : reg_(seed) {}

  void Reset(std::uint16_t value) { reg_ = value; }
  void XorWith(std::uint16_t value) {
    reg_ = static_cast<std::uint16_t>(reg_ ^ value);
  }

  int GetRandomNumber(int bits) {
    const int r = reg_;
    const int bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    const int next = (r >> 1) | (bit << 15);
    const int result = (next >> (16 - bits)) & ((1 << bits) - 1);
    reg_ = static_cast<std::uint16_t>(next);
    return result;
  }

 private:
  std::uint16_t reg_;
};

struct ConstrainedImage {
  int width = 0;
  int height = 0;
  std::vector<int> samples;

  ConstrainedImage(int w, int h) : width(w), height(h) {
    samples.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
  }
  int& At(int x, int y) {
    return samples[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)];
  }
  int At(int x, int y) const {
    return samples[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                   static_cast<std::size_t>(x)];
  }
};

// §7.21.7.3 generate grain process.
void GenerateGrain(const FilmGrainModel& m, std::uint16_t grain_seed, int sub_x,
                   int sub_y, int bit_depth, ConstrainedImage* luma,
                   ConstrainedImage* cb, ConstrainedImage* cr) {
  const int grain_min = -(1 << (bit_depth - 1));
  const int grain_max = (1 << (bit_depth - 1)) - 1;
  const int noise_shift = 12 - bit_depth + m.grain_scale_shift;
  const int ar_shift = m.ar_coeff_shift_minus_6 + 6;

  RandomRegister rng(grain_seed);

  for (int y = 0; y < 73; ++y) {
    for (int x = 0; x < 82; ++x) {
      const int g =
          m.num_y_points > 0 ? kGaussianSequence[rng.GetRandomNumber(11)] : 0;
      luma->At(x, y) = static_cast<int>(Round2(g, noise_shift));
    }
  }

  for (int y = 3; y < 73; ++y) {
    for (int x = 3; x < 82 - 3; ++x) {
      std::int64_t s = 0;
      int pos = 0;
      for (int delta_row = -m.ar_coeff_lag; delta_row <= 0; ++delta_row) {
        for (int delta_col = -m.ar_coeff_lag; delta_col <= m.ar_coeff_lag;
             ++delta_col) {
          if (delta_row == 0 && delta_col == 0) break;
          s += static_cast<std::int64_t>(
                   luma->At(x + delta_col, y + delta_row)) *
               m.ar_coeffs_y[pos];
          ++pos;
        }
      }
      luma->At(x, y) =
          Clip3(grain_min, grain_max,
                luma->At(x, y) + static_cast<int>(Round2(s, ar_shift)));
    }
  }

  const int chroma_w = sub_x ? 44 : 82;
  const int chroma_h = sub_y ? 38 : 73;
  const int chroma_shift = 12 - bit_depth + m.grain_scale_shift;

  rng.Reset(grain_seed);
  rng.XorWith(0xb524);
  for (int y = 0; y < chroma_h; ++y) {
    for (int x = 0; x < chroma_w; ++x) {
      const int g = (m.num_cb_points > 0 || m.chroma_scaling_from_luma)
                        ? kGaussianSequence[rng.GetRandomNumber(11)]
                        : 0;
      cb->At(x, y) = static_cast<int>(Round2(g, chroma_shift));
    }
  }

  rng.Reset(grain_seed);
  rng.XorWith(0x49d8);
  for (int y = 0; y < chroma_h; ++y) {
    for (int x = 0; x < chroma_w; ++x) {
      const int g = (m.num_cr_points > 0 || m.chroma_scaling_from_luma)
                        ? kGaussianSequence[rng.GetRandomNumber(11)]
                        : 0;
      cr->At(x, y) = static_cast<int>(Round2(g, chroma_shift));
    }
  }

  for (int y = 3; y < chroma_h; ++y) {
    for (int x = 3; x < chroma_w - 3; ++x) {
      std::int64_t s0 = 0;
      std::int64_t s1 = 0;
      int pos = 0;
      for (int delta_row = -m.ar_coeff_lag; delta_row <= 0; ++delta_row) {
        for (int delta_col = -m.ar_coeff_lag; delta_col <= m.ar_coeff_lag;
             ++delta_col) {
          const int c0 = m.ar_coeffs_cb[pos];
          const int c1 = m.ar_coeffs_cr[pos];
          if (delta_row == 0 && delta_col == 0) {
            if (m.num_y_points > 0) {
              int luma_sum = 0;
              const int luma_x = ((x - 3) << sub_x) + 3;
              const int luma_y = ((y - 3) << sub_y) + 3;
              for (int i = 0; i <= sub_y; ++i) {
                for (int j = 0; j <= sub_x; ++j) {
                  luma_sum += luma->At(luma_x + j, luma_y + i);
                }
              }
              luma_sum = static_cast<int>(Round2(luma_sum, sub_x + sub_y));
              s0 += static_cast<std::int64_t>(luma_sum) * c0;
              s1 += static_cast<std::int64_t>(luma_sum) * c1;
            }
            break;
          }
          s0 += static_cast<std::int64_t>(cb->At(x + delta_col, y + delta_row)) *
                c0;
          s1 += static_cast<std::int64_t>(cr->At(x + delta_col, y + delta_row)) *
                c1;
          ++pos;
        }
      }
      cb->At(x, y) =
          Clip3(grain_min, grain_max,
                cb->At(x, y) + static_cast<int>(Round2(s0, ar_shift)));
      cr->At(x, y) =
          Clip3(grain_min, grain_max,
                cr->At(x, y) + static_cast<int>(Round2(s1, ar_shift)));
    }
  }
}

// §7.21.7.4 scaling lookup initialization process.
void BuildScalingLut(const FilmGrainModel& m, int num_planes,
                     int lut[3][256]) {
  auto get_x = [&](int plane, int i) {
    if (plane == 0 || m.chroma_scaling_from_luma) return m.point_y_value[i];
    return plane == 1 ? m.point_cb_value[i] : m.point_cr_value[i];
  };
  auto get_y = [&](int plane, int i) {
    if (plane == 0 || m.chroma_scaling_from_luma) return m.point_y_scaling[i];
    return plane == 1 ? m.point_cb_scaling[i] : m.point_cr_scaling[i];
  };

  for (int plane = 0; plane < num_planes; ++plane) {
    int num_points = 0;
    if (plane == 0 || m.chroma_scaling_from_luma) {
      num_points = m.num_y_points;
    } else if (plane == 1) {
      num_points = m.num_cb_points;
    } else {
      num_points = m.num_cr_points;
    }

    if (num_points == 0) {
      for (int x = 0; x < 256; ++x) lut[plane][x] = 0;
      continue;
    }

    for (int x = 0; x < get_x(plane, 0); ++x) lut[plane][x] = get_y(plane, 0);
    for (int i = 0; i < num_points - 1; ++i) {
      const int delta_y = get_y(plane, i + 1) - get_y(plane, i);
      const int delta_x = get_x(plane, i + 1) - get_x(plane, i);
      const int delta = delta_y * ((65536 + (delta_x >> 1)) / delta_x);
      for (int x = 0; x < delta_x; ++x) {
        const int v =
            get_y(plane, i) + static_cast<int>((x * delta + 32768) >> 16);
        lut[plane][get_x(plane, i) + x] = v;
      }
    }
    for (int x = get_x(plane, num_points - 1); x < 256; ++x) {
      lut[plane][x] = get_y(plane, num_points - 1);
    }
  }
}

// §7.21.7.5 scale_lut(plane, index, bitDepth).
int ScaleLut(const int lut[3][256], int plane, int index, int bit_depth) {
  const int shift = bit_depth - 8;
  const int x = index >> shift;
  const int rem = index - (x << shift);
  if (x == 255) return lut[plane][255];
  const int start = lut[plane][x];
  const int end = lut[plane][x + 1];
  return start + static_cast<int>(Round2(static_cast<std::int64_t>(end - start) * rem,
                                         shift));
}

// §7.21.7.5 add noise synthesis process.
void AddNoise(const FilmGrainModel& m, const ConstrainedImage& luma_grain,
              const ConstrainedImage& cb_grain, const ConstrainedImage& cr_grain,
              FilmGrainFrame& frame, int lut[3][256],
              std::uint16_t grain_seed) {
  const int w = frame.w;
  const int h = frame.h;
  const int bit_depth = frame.bit_depth;
  const int sub_x = frame.sub_x;
  const int sub_y = frame.sub_y;
  const int num_planes = frame.num_planes;
  const int grain_min = -(1 << (bit_depth - 1));
  const int grain_max = (1 << (bit_depth - 1)) - 1;

  const int luma_size = m.film_grain_block_size ? 32 : 16;
  const int step = luma_size >> 1;

  const int stripe_h[3] = {34, 34 >> sub_y, 34 >> sub_y};
  // Chroma uses the luma-style x*2+j indexing when planeSubX == 0 (4:4:4),
  // so only subsampled chroma gets the narrower allocation.
  const int chroma_stripe_w = ((w + 1) >> 1) + luma_size + 4;
  const int stripe_w[3] = {
      w + 2 * luma_size + 4,
      sub_x ? chroma_stripe_w : w + 2 * luma_size + 4,
      sub_x ? chroma_stripe_w : w + 2 * luma_size + 4,
  };

  const int half_h = (h + 1) / 2;
  const int num_stripes = half_h > 0 ? (half_h + step - 1) / step : 1;

  std::vector<int> noise_stripe[3];
  for (int plane = 0; plane < 3; ++plane) {
    noise_stripe[plane].assign(
        static_cast<std::size_t>(num_stripes) *
            static_cast<std::size_t>(stripe_h[plane]) *
            static_cast<std::size_t>(stripe_w[plane]),
        0);
  }
  auto stripe_at = [&](int plane, int num, int i, int col) -> int& {
    return noise_stripe[plane]
                       [(static_cast<std::size_t>(num) *
                             static_cast<std::size_t>(stripe_h[plane]) +
                         static_cast<std::size_t>(i)) *
                            static_cast<std::size_t>(stripe_w[plane]) +
                        static_cast<std::size_t>(col)];
  };

  RandomRegister rng(grain_seed);

  int luma_num = 0;
  for (int y = 0; y < half_h; y += step) {
    rng.Reset(grain_seed);
    const int luma_rand = y >> 3;
    rng.XorWith(static_cast<std::uint16_t>(((luma_rand * 37 + 178) & 255) << 8));
    rng.XorWith(static_cast<std::uint16_t>((luma_rand * 173 + 105) & 255));

    for (int x = 0; x < (w + 1) / 2; x += step) {
      const int offset_y =
          rng.GetRandomNumber(9) * (3 - static_cast<int>(m.film_grain_block_size)) >> 6;
      rng.GetRandomNumber(1);
      rng.GetRandomNumber(1);
      rng.GetRandomNumber(1);
      const int offset_x =
          rng.GetRandomNumber(9) * (3 - static_cast<int>(m.film_grain_block_size)) >> 6;
      rng.GetRandomNumber(1);
      rng.GetRandomNumber(1);
      rng.GetRandomNumber(1);

      for (int plane = 0; plane < num_planes; ++plane) {
        const int plane_sub_x = plane > 0 ? sub_x : 0;
        const int plane_sub_y = plane > 0 ? sub_y : 0;
        const int plane_offset_x =
            plane_sub_x ? 6 + offset_x : 9 + offset_x * 2;
        const int plane_offset_y =
            plane_sub_y ? 6 + offset_y : 9 + offset_y * 2;

        for (int i = 0; i < (luma_size + 2) >> plane_sub_y; ++i) {
          for (int j = 0; j < (luma_size + 2) >> plane_sub_x; ++j) {
            int g = 0;
            if (plane == 0) {
              g = luma_grain.At(plane_offset_x + j, plane_offset_y + i);
            } else if (plane == 1) {
              g = cb_grain.At(plane_offset_x + j, plane_offset_y + i);
            } else {
              g = cr_grain.At(plane_offset_x + j, plane_offset_y + i);
            }

            if (plane_sub_x == 0) {
              if (j < 2 && m.overlap_flag && x > 0) {
                const int old = stripe_at(plane, luma_num, i, x * 2 + j);
                g = j == 0 ? old * 27 + g * 17 : old * 17 + g * 27;
                g = Clip3(grain_min, grain_max,
                          static_cast<int>(Round2(g, 5)));
              }
              stripe_at(plane, luma_num, i, x * 2 + j) = g;
            } else {
              if (j == 0 && m.overlap_flag && x > 0) {
                const int old = stripe_at(plane, luma_num, i, x + j);
                g = old * 23 + g * 22;
                g = Clip3(grain_min, grain_max,
                          static_cast<int>(Round2(g, 5)));
              }
              stripe_at(plane, luma_num, i, x + j) = g;
            }
          }
        }
      }
    }
    ++luma_num;
  }

  // Blend noise stripes into a full noise image.
  std::vector<int> noise_image[3];
  for (int plane = 0; plane < 3; ++plane) {
    const int plane_sub_x = plane > 0 ? sub_x : 0;
    const int plane_sub_y = plane > 0 ? sub_y : 0;
    const int iw = (w + plane_sub_x) >> plane_sub_x;
    const int ih = (h + plane_sub_y) >> plane_sub_y;
    noise_image[plane].assign(
        static_cast<std::size_t>(iw) * static_cast<std::size_t>(ih), 0);
  }
  auto image_at = [&](int plane, int x, int y, int width) -> int& {
    return noise_image[plane][static_cast<std::size_t>(y) *
                              static_cast<std::size_t>(width) +
                          static_cast<std::size_t>(x)];
  };

  for (int plane = 0; plane < num_planes; ++plane) {
    const int plane_sub_x = plane > 0 ? sub_x : 0;
    const int plane_sub_y = plane > 0 ? sub_y : 0;
    const int iw = (w + plane_sub_x) >> plane_sub_x;
    const int ih = (h + plane_sub_y) >> plane_sub_y;
    const int shift = 4 + static_cast<int>(m.film_grain_block_size) - plane_sub_y;

    for (int y = 0; y < ih; ++y) {
      const int num = y >> shift;
      const int i = y - (num << shift);
      for (int x = 0; x < iw; ++x) {
        int g = stripe_at(plane, num, i, x);
        if (plane_sub_y == 0) {
          if (i < 2 && num > 0 && m.overlap_flag) {
            const int old = stripe_at(plane, num - 1, i + luma_size, x);
            g = i == 0 ? old * 27 + g * 17 : old * 17 + g * 27;
            g = Clip3(grain_min, grain_max, static_cast<int>(Round2(g, 5)));
          }
        } else {
          if (i < 1 && num > 0 && m.overlap_flag) {
            const int old = stripe_at(plane, num - 1, i + (luma_size >> 1), x);
            g = old * 23 + g * 22;
            g = Clip3(grain_min, grain_max, static_cast<int>(Round2(g, 5)));
          }
        }
        image_at(plane, x, y, iw) = g;
      }
    }
  }

  // Blend noise with the image.
  int min_value = 0;
  int max_luma = 0;
  int max_chroma = 0;
  if (m.clip_to_restricted_range) {
    min_value = 16 << (bit_depth - 8);
    max_luma = 235 << (bit_depth - 8);
    max_chroma = m.fg_mc_identity ? max_luma : 240 << (bit_depth - 8);
  } else {
    min_value = 0;
    max_luma = (256 << (bit_depth - 8)) - 1;
    max_chroma = max_luma;
  }
  const int scaling_shift = m.grain_scaling_minus_8 + 8;

  const int chroma_w = (w + sub_x) >> sub_x;
  const int chroma_h = (h + sub_y) >> sub_y;

  if (num_planes > 1) {
    for (int y = 0; y < chroma_h; ++y) {
      for (int x = 0; x < chroma_w; ++x) {
        const int luma_x = x << sub_x;
        const int luma_y = y << sub_y;
        const int luma_next_x = std::min(luma_x + 1, w - 1);
        const int average_luma =
            sub_x ? static_cast<int>(Round2(
                        frame.plane[0].data[static_cast<std::ptrdiff_t>(luma_y) *
                                                 frame.plane[0].stride +
                                             luma_x] +
                            frame.plane[0].data[static_cast<std::ptrdiff_t>(luma_y) *
                                                     frame.plane[0].stride +
                                                 luma_next_x],
                        1))
                  : frame.plane[0].data[static_cast<std::ptrdiff_t>(luma_y) *
                                            frame.plane[0].stride +
                                        luma_x];

        if (m.num_cb_points > 0 || m.chroma_scaling_from_luma) {
          const int orig = frame.plane[1].data[static_cast<std::ptrdiff_t>(y) *
                                                   frame.plane[1].stride +
                                               x];
          int merged = average_luma;
          if (!m.chroma_scaling_from_luma) {
            const int combined = average_luma * (m.cb_luma_mult - 128) +
                                 orig * (m.cb_mult - 128);
            merged = Clip3(0, (1 << bit_depth) - 1,
                           (combined >> 6) +
                               ((m.cb_offset - 256) << (bit_depth - 8)));
          }
          const int noise = image_at(1, x, y, chroma_w);
          const int scaled = static_cast<int>(Round2(
              static_cast<std::int64_t>(ScaleLut(lut, 1, merged, bit_depth)) * noise,
              scaling_shift));
          frame.plane[1].data[static_cast<std::ptrdiff_t>(y) *
                                  frame.plane[1].stride +
                              x] = static_cast<std::uint16_t>(
              Clip3(min_value, max_chroma, orig + scaled));
        }

        if (num_planes > 2 &&
            (m.num_cr_points > 0 || m.chroma_scaling_from_luma)) {
          const int orig = frame.plane[2].data[static_cast<std::ptrdiff_t>(y) *
                                                   frame.plane[2].stride +
                                               x];
          int merged = average_luma;
          if (!m.chroma_scaling_from_luma) {
            const int combined = average_luma * (m.cr_luma_mult - 128) +
                                 orig * (m.cr_mult - 128);
            merged = Clip3(0, (1 << bit_depth) - 1,
                           (combined >> 6) +
                               ((m.cr_offset - 256) << (bit_depth - 8)));
          }
          const int noise = image_at(2, x, y, chroma_w);
          const int scaled = static_cast<int>(Round2(
              static_cast<std::int64_t>(ScaleLut(lut, 2, merged, bit_depth)) * noise,
              scaling_shift));
          frame.plane[2].data[static_cast<std::ptrdiff_t>(y) *
                                  frame.plane[2].stride +
                              x] = static_cast<std::uint16_t>(
              Clip3(min_value, max_chroma, orig + scaled));
        }
      }
    }
  }

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (m.num_y_points == 0) continue;
      const int orig = frame.plane[0].data[static_cast<std::ptrdiff_t>(y) *
                                               frame.plane[0].stride +
                                           x];
      const int noise = image_at(0, x, y, w);
      const int scaled = static_cast<int>(Round2(
          static_cast<std::int64_t>(ScaleLut(lut, 0, orig, bit_depth)) * noise,
          scaling_shift));
      frame.plane[0].data[static_cast<std::ptrdiff_t>(y) *
                              frame.plane[0].stride +
                          x] =
          static_cast<std::uint16_t>(Clip3(min_value, max_luma, orig + scaled));
    }
  }
}

}  // namespace

bool FilmGrainModelValid(const FilmGrainModel& m, int sub_x, int sub_y,
                         int bit_depth, int num_planes, std::string* why) {
  auto fail = [&](const char* reason) {
    if (why != nullptr) *why = reason;
    return false;
  };
  if (bit_depth < 8 || bit_depth > 12) return fail("bit_depth outside 8..12");
  if (num_planes < 1 || num_planes > 3) return fail("num_planes outside 1..3");
  if (sub_x < 0 || sub_x > 1 || sub_y < 0 || sub_y > 1) {
    return fail("subsampling outside 4:4:4..4:2:0");
  }
  auto check_points = [&](int n, const int* values, const char* name) {
    if (n < 0 || n > kMaxFilmGrainPoints) return fail(name);
    if (n > 0 && values[0] >= 256) return fail(name);
    for (int i = 1; i < n; ++i) {
      if (values[i] <= values[i - 1] || values[i] >= 256) return fail(name);
    }
    return true;
  };
  if (!check_points(m.num_y_points, m.point_y_value, "point_y_value")) return false;
  if (!check_points(m.num_cb_points, m.point_cb_value, "point_cb_value")) return false;
  if (!check_points(m.num_cr_points, m.point_cr_value, "point_cr_value")) return false;
  if (m.chroma_scaling_from_luma && (m.num_cb_points != 0 || m.num_cr_points != 0)) {
    return fail("chroma_scaling_from_luma requires zero chroma points");
  }
  if (sub_x == 1 && sub_y == 1 && (m.num_cb_points == 0) != (m.num_cr_points == 0)) {
    return fail("4:2:0 requires both or neither chroma point set");
  }
  if (m.grain_scaling_minus_8 < 0 || m.grain_scaling_minus_8 > 3) {
    return fail("grain_scaling_minus_8 outside 0..3");
  }
  if (m.ar_coeff_lag < 0 || m.ar_coeff_lag > 3) return fail("ar_coeff_lag outside 0..3");
  if (m.ar_coeff_shift_minus_6 < 0 || m.ar_coeff_shift_minus_6 > 3) {
    return fail("ar_coeff_shift_minus_6 outside 0..3");
  }
  if (m.grain_scale_shift < 0 || m.grain_scale_shift > 3) {
    return fail("grain_scale_shift outside 0..3");
  }
  if (m.cb_mult < 0 || m.cb_mult > 255 || m.cb_luma_mult < 0 ||
      m.cb_luma_mult > 255 || m.cb_offset < 0 || m.cb_offset > 511) {
    return fail("cb_* outside signaled range");
  }
  if (m.cr_mult < 0 || m.cr_mult > 255 || m.cr_luma_mult < 0 ||
      m.cr_luma_mult > 255 || m.cr_offset < 0 || m.cr_offset > 511) {
    return fail("cr_* outside signaled range");
  }
  return true;
}

void ApplyFilmGrain(const FilmGrainModel& model, FilmGrainFrame& frame,
                    std::uint16_t grain_seed) {
  if (!FilmGrainModelValid(model, frame.sub_x, frame.sub_y, frame.bit_depth,
                           frame.num_planes, nullptr)) {
    return;
  }
  if (frame.plane[0].data == nullptr) return;
  if (frame.num_planes > 1 && frame.plane[1].data == nullptr) return;
  if (frame.num_planes > 2 && frame.plane[2].data == nullptr) return;

  ConstrainedImage luma_grain(82, 73);
  ConstrainedImage cb_grain(82, 73);
  ConstrainedImage cr_grain(82, 73);

  GenerateGrain(model, grain_seed, frame.sub_x, frame.sub_y, frame.bit_depth,
                &luma_grain, &cb_grain, &cr_grain);

  int lut[3][256] = {};
  BuildScalingLut(model, frame.num_planes, lut);

  AddNoise(model, luma_grain, cb_grain, cr_grain, frame, lut, grain_seed);
}

}  // namespace av2enc::filters::spec
