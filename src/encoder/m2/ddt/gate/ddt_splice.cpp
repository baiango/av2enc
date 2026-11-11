// DDT gate: splice the §5.4.8 enable_inter_ddt bit into an existing M2 stream.
//
// Inputs are two IVF files produced by the same gate binary and the same
// encoder inputs, differing only in DDT_GATE_ENABLE.  The tool:
//   1. locates the sequence-header OBU in frame 0,
//   2. asserts that exactly one bit differs between the files and that it is
//      inside that OBU's payload,
//   3. flips the bit in the first file and writes the spliced stream,
//   4. asserts the spliced stream is byte-identical to the second file.
//
// Usage: ddt_splice <base.ivf> <ddt.ivf> <spliced.ivf>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "encoder/m2/ddt/ddt_syntax.h"

namespace {

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

bool write_file(const std::string& path, const std::vector<std::uint8_t>& data) {
  std::ofstream output(path, std::ios::binary);
  if (!output) return false;
  output.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(output);
}

struct Span {
  std::size_t begin = 0;
  std::size_t end = 0;
};

// Parse the OBU framing of the first IVF frame and return the sequence-header
// payload byte range.
Span find_sequence_payload(const std::vector<std::uint8_t>& file) {
  if (file.size() < 44) {
    std::fprintf(stderr, "file too small for IVF header\n");
    std::exit(2);
  }
  std::size_t pos = 32 + 12;  // IVF file header + first frame header
  const std::size_t frame_end =
      pos + (static_cast<std::size_t>(file[32]) |
             (static_cast<std::size_t>(file[33]) << 8) |
             (static_cast<std::size_t>(file[34]) << 16) |
             (static_cast<std::size_t>(file[35]) << 24));
  if (frame_end > file.size()) {
    std::fprintf(stderr, "truncated IVF frame\n");
    std::exit(2);
  }
  while (pos < frame_end) {
    std::uint64_t size = 0;
    int shift = 0;
    std::size_t cursor = pos;
    while (cursor < frame_end) {
      const std::uint8_t byte = file[cursor++];
      size |= static_cast<std::uint64_t>(byte & 0x7fU) << shift;
      shift += 7;
      if ((byte & 0x80U) == 0) break;
    }
    const std::size_t obu_begin = cursor;
    if (obu_begin >= frame_end) break;
    const std::uint8_t header = file[obu_begin];
    const unsigned type = (header >> 2) & 0x1fU;
    const std::size_t obu_end = obu_begin + static_cast<std::size_t>(size);
    if (obu_end > frame_end) {
      std::fprintf(stderr, "truncated OBU\n");
      std::exit(2);
    }
    if (type == 1U) {  // OBU_SEQUENCE_HEADER
      if (obu_end <= obu_begin + 1) break;
      return Span{obu_begin + 1, obu_end};
    }
    pos = obu_end;
  }
  std::fprintf(stderr, "sequence header OBU not found\n");
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <base.ivf> <ddt.ivf> <spliced.ivf>\n",
                 argv[0]);
    return 2;
  }
  std::vector<std::uint8_t> base = read_file(argv[1]);
  const std::vector<std::uint8_t> ddt = read_file(argv[2]);
  if (base.size() != ddt.size()) {
    std::fprintf(stderr,
                 "FAIL: streams differ in length (%zu vs %zu); the DDT bit "
                 "must not change the payload size for this config\n",
                 base.size(), ddt.size());
    return 1;
  }

  std::size_t bit = 0;
  int differences = 0;
  for (std::size_t i = 0; i < base.size(); ++i) {
    const std::uint8_t diff = static_cast<std::uint8_t>(base[i] ^ ddt[i]);
    if (diff == 0) continue;
    for (int b = 0; b < 8; ++b) {
      if ((diff & (0x80U >> b)) != 0) {
        bit = i * 8 + static_cast<std::size_t>(b);
        ++differences;
      }
    }
  }
  if (differences != 1) {
    std::fprintf(stderr, "FAIL: expected exactly one differing bit, got %d\n",
                 differences);
    return 1;
  }

  const Span seq = find_sequence_payload(base);
  const std::size_t seq_begin_bit = seq.begin * 8;
  const std::size_t seq_end_bit = seq.end * 8;
  if (bit < seq_begin_bit || bit >= seq_end_bit) {
    std::fprintf(stderr,
                 "FAIL: differing bit %zu is outside the sequence payload "
                 "[%zu, %zu)\n",
                 bit, seq_begin_bit, seq_end_bit);
    return 1;
  }

  const std::size_t value = (ddt[bit / 8] >> (7 - (bit % 8))) & 1U;
  const std::size_t syntax_bit = bit - seq_begin_bit;
  // Exercise the component's reader on both real headers.
  const std::span<const std::uint8_t> base_payload(
      base.data() + seq.begin, seq.end - seq.begin);
  const std::span<const std::uint8_t> ddt_payload(
      ddt.data() + seq.begin, seq.end - seq.begin);
  bool base_value = true;
  bool ddt_value = false;
  std::size_t base_next = 0;
  std::size_t ddt_next = 0;
  if (!av2enc::m2::ddt::read_enable_inter_ddt(
          base_payload, syntax_bit, /*single_picture_header_flag=*/false,
          &base_value, &base_next) ||
      !av2enc::m2::ddt::read_enable_inter_ddt(
          ddt_payload, syntax_bit, /*single_picture_header_flag=*/false,
          &ddt_value, &ddt_next) ||
      base_value || !ddt_value || base_next != syntax_bit + 1 ||
      ddt_next != syntax_bit + 1) {
    std::fprintf(stderr, "FAIL: ddt_syntax reader disagrees with the splice\n");
    return 1;
  }
  std::printf("DDT splice: bit=%zu seq_payload=[%zu,%zu) syntax_bit=%zu "
              "reader=%d->%d\n",
              bit, seq_begin_bit, seq_end_bit, syntax_bit,
              base_value ? 1 : 0, ddt_value ? 1 : 0);
  std::printf("DDT splice: enable_inter_ddt=%zu, stream bytes=%zu, "
              "payload bytes=%zu\n",
              value, base.size(), seq.end - seq.begin);

  base[bit / 8] = static_cast<std::uint8_t>(
      base[bit / 8] ^ static_cast<std::uint8_t>(0x80U >> (bit % 8)));
  if (base != ddt) {
    std::fprintf(stderr, "FAIL: spliced stream != DDT stream\n");
    return 1;
  }
  if (!write_file(argv[3], base)) {
    std::fprintf(stderr, "cannot write %s\n", argv[3]);
    return 1;
  }
  std::printf("DDT splice: PASS (spliced == DDT stream, one bit)\n");
  return 0;
}
