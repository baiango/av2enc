#include "frame/metadata/cli/metadata_cli.h"

#include <array>
#include <cstring>
#include <span>
namespace av2enc::metadata::cli {
namespace {

// Minimal RFC 1321 MD5 (§6.16.13 hash_type 0).
class Md5 {
 public:
  Md5() { reset(); }

  void update(std::span<const std::uint8_t> data) {
    for (const std::uint8_t byte : data) {
      buffer_[used_++] = byte;
      if (used_ == 64) {
        transform(buffer_.data());
        used_ = 0;
      }
    }
    total_ += static_cast<std::uint64_t>(data.size());
  }

  std::array<std::uint8_t, 16> finish() {
    const std::uint64_t bit_length = total_ * 8U;
    std::uint8_t pad = 0x80;
    const std::uint8_t zero = 0;
    update(std::span<const std::uint8_t>(&pad, 1));
    while (used_ != 56) update(std::span<const std::uint8_t>(&zero, 1));
    std::array<std::uint8_t, 8> length{};
    for (int i = 0; i < 8; ++i) {
      length[static_cast<std::size_t>(i)] =
          static_cast<std::uint8_t>(bit_length >> (8 * i));
    }
    update(std::span<const std::uint8_t>(length.data(), length.size()));
    std::array<std::uint8_t, 16> digest{};
    for (int i = 0; i < 4; ++i) {
      for (int b = 0; b < 4; ++b) {
        digest[static_cast<std::size_t>(i * 4 + b)] =
            static_cast<std::uint8_t>(state_[i] >> (8 * b));
      }
    }
    reset();
    return digest;
  }

 private:
  static std::uint32_t rotate(std::uint32_t x, std::uint32_t c) {
    return (x << c) | (x >> (32 - c));
  }

  void reset() {
    state_ = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u};
    used_ = 0;
    total_ = 0;
  }

  void transform(const std::uint8_t* block) {
    static constexpr std::uint32_t kK[64] = {
        0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu, 0xf57c0fafu,
        0x4787c62au, 0xa8304613u, 0xfd469501u, 0x698098d8u, 0x8b44f7afu,
        0xffff5bb1u, 0x895cd7beu, 0x6b901122u, 0xfd987193u, 0xa679438eu,
        0x49b40821u, 0xf61e2562u, 0xc040b340u, 0x265e5a51u, 0xe9b6c7aau,
        0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u, 0x21e1cde6u,
        0xc33707d6u, 0xf4d50d87u, 0x455a14edu, 0xa9e3e905u, 0xfcefa3f8u,
        0x676f02d9u, 0x8d2a4c8au, 0xfffa3942u, 0x8771f681u, 0x6d9d6122u,
        0xfde5380cu, 0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u,
        0x289b7ec6u, 0xeaa127fau, 0xd4ef3085u, 0x04881d05u, 0xd9d4d039u,
        0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u, 0xf4292244u, 0x432aff97u,
        0xab9423a7u, 0xfc93a039u, 0x655b59c3u, 0x8f0ccc92u, 0xffeff47du,
        0x85845dd1u, 0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u,
        0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u};
    static constexpr std::uint32_t kShift[64] = {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
    std::array<std::uint32_t, 16> m{};
    for (int i = 0; i < 16; ++i) {
      m[static_cast<std::size_t>(i)] =
          static_cast<std::uint32_t>(block[i * 4]) |
          (static_cast<std::uint32_t>(block[i * 4 + 1]) << 8) |
          (static_cast<std::uint32_t>(block[i * 4 + 2]) << 16) |
          (static_cast<std::uint32_t>(block[i * 4 + 3]) << 24);
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    for (std::uint32_t i = 0; i < 64; ++i) {
      std::uint32_t f = 0;
      std::uint32_t g = 0;
      if (i < 16) {
        f = (b & c) | (~b & d);
        g = i;
      } else if (i < 32) {
        f = (d & b) | (~d & c);
        g = (5 * i + 1) % 16;
      } else if (i < 48) {
        f = b ^ c ^ d;
        g = (3 * i + 5) % 16;
      } else {
        f = c ^ (b | ~d);
        g = (7 * i) % 16;
      }
      const std::uint32_t temp = d;
      d = c;
      c = b;
      b = b + rotate(a + f + kK[i] + m[g], kShift[i]);
      a = temp;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
  }

  std::array<std::uint32_t, 4> state_{};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t used_ = 0;
  std::uint64_t total_ = 0;
};

hdr::MetadataObuConfig suffix_config() {
  hdr::MetadataObuConfig config;
  config.suffix = true;
  config.layer = hdr::MetadataLayer::kCurrent;
  config.persistence = hdr::MetadataPersistence::kGlobal;
  return config;
}

bool append_obu(std::vector<std::uint8_t>& unit,
                const std::vector<std::uint8_t>& obu) {
  if (obu.empty()) return false;
  unit.insert(unit.end(), obu.begin(), obu.end());
  return true;
}

DecodedFrameHash compute_frame_hash(const Options::FramePlanes& planes,
                                    bool per_plane) {
  DecodedFrameHash value;
  value.hash_type = 0;  // MD5
  value.per_plane = per_plane;
  value.has_grain = false;
  value.is_monochrome = false;
  if (per_plane) {
    const std::span<const std::uint8_t> plane_data[3] = {planes.y, planes.u,
                                                         planes.v};
    for (int i = 0; i < 3; ++i) {
      Md5 md5;
      md5.update(plane_data[i]);
      value.plane_hash[static_cast<std::size_t>(i)] = md5.finish();
    }
  } else {
    Md5 md5;
    md5.update(planes.y);
    md5.update(planes.u);
    md5.update(planes.v);
    value.frame_hash = md5.finish();
  }
  return value;
}

}  // namespace

std::vector<std::uint8_t> default_icc_profile() {
  std::vector<std::uint8_t> profile(132, 0);
  const auto put32 = [&profile](std::size_t at, std::uint32_t value) {
    profile[at + 0] = static_cast<std::uint8_t>((value >> 24) & 0xFF);
    profile[at + 1] = static_cast<std::uint8_t>((value >> 16) & 0xFF);
    profile[at + 2] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
    profile[at + 3] = static_cast<std::uint8_t>(value & 0xFF);
  };
  put32(0, 132);          // profile size
  put32(4, 0);            // preferred CMM
  put32(8, 0x04300000);   // version 4.3
  put32(12, 0x6D6E7472);  // 'mntr' display device class
  put32(16, 0x52474220);  // 'RGB ' data colour space
  put32(20, 0x58595A20);  // 'XYZ ' PCS
  put32(36, 0x61637370);  // 'acsp'
  // Byte 48 (D50 illuminant XYZ), offsets 64 (intent), 80 (creator), 100
  // (profile ID) and 128 (zero tag count) stay zero: a valid minimal header.
  put32(68, 0x0000F6D6);  // illuminant X = 0.9642
  put32(72, 0x00010000);  // illuminant Y = 1.0
  put32(76, 0x0000D32D);  // illuminant Z = 0.8249
  return profile;
}

bool append_suffix_metadata(std::vector<std::uint8_t>& unit,
                            const Options& options,
                            const Options::FramePlanes* planes,
                            std::string& error) {
  if (!options.any()) return true;
  const hdr::MetadataObuConfig config = suffix_config();
  std::vector<std::uint8_t> extra;
  if (options.frame_hash) {
    if (planes == nullptr) {
      error = "--metadata-frame-hash needs the encoder reconstruction";
      return false;
    }
    const auto payload = make_decoded_frame_hash_unit(
        compute_frame_hash(*planes, options.frame_hash_per_plane), error);
    if (!payload.has_value()) return false;
    const std::vector<std::uint8_t> obu =
        make_short_obu(*payload, config, error);
    if (obu.empty() || !append_obu(extra, obu)) return false;
  }
  if (options.cll) {
    const std::vector<std::uint8_t> obu =
        hdr::make_cll_short_obu(options.cll_value, config, error);
    if (obu.empty() || !append_obu(extra, obu)) return false;
  }
  if (options.mdcv) {
    const std::vector<std::uint8_t> obu =
        hdr::make_mdcv_short_obu(options.mdcv_value, config, error);
    if (obu.empty() || !append_obu(extra, obu)) return false;
  }
  if (options.banding) {
    const auto payload = make_banding_hints_unit(options.banding_value, error);
    if (!payload.has_value()) return false;
    const std::vector<std::uint8_t> obu =
        make_short_obu(*payload, config, error);
    if (obu.empty() || !append_obu(extra, obu)) return false;
  }
  if (options.icc) {
    IccProfile value;
    value.data = options.icc_data;
    const auto payload = make_icc_profile_unit(value, error);
    if (!payload.has_value()) return false;
    const std::vector<std::uint8_t> obu =
        make_short_obu(*payload, config, error);
    if (obu.empty() || !append_obu(extra, obu)) return false;
  }
  if (options.user_data) {
    const auto payload = make_user_data_unregistered_unit(
        options.user_data_value, error);
    if (!payload.has_value()) return false;
    const std::vector<std::uint8_t> obu =
        make_short_obu(*payload, config, error);
    if (obu.empty() || !append_obu(extra, obu)) return false;
  }
  if (options.temporal_point) {
    TemporalPointInfo value;
    value.frame_presentation_time = options.frame_presentation_time;
    const auto payload = make_temporal_point_info_unit(value, error);
    if (!payload.has_value()) return false;
    const std::vector<std::uint8_t> obu =
        make_short_obu(*payload, config, error);
    if (obu.empty() || !append_obu(extra, obu)) return false;
  }
  unit.insert(unit.end(), extra.begin(), extra.end());
  return true;
}

}  // namespace av2enc::metadata::cli
