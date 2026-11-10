#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace test {

inline int& failure_count() {
  static int count = 0;
  return count;
}

inline void check(bool condition, const char* expression, const char* file, int line) {
  if (!condition) {
    ++failure_count();
    std::cerr << file << ':' << line << ": CHECK failed: " << expression << '\n';
  }
}

inline int finish(const char* name) {
  if (failure_count() != 0) {
    std::cerr << name << ": " << failure_count() << " failure(s)\n";
    return 1;
  }
  std::cout << name << ": PASS\n";
  return 0;
}

inline std::filesystem::path temp_path(std::string_view name) {
  return std::filesystem::temp_directory_path() /
         (std::string("av2enc_test_") + std::string(name));
}

inline void remove_file(const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::remove(path, error);
}

inline bool write_file(const std::filesystem::path& path,
                       const std::vector<std::uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return false;
  }
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(out);
}

inline std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return {};
  }
  std::vector<std::uint8_t> bytes;
  char byte = 0;
  while (in.get(byte)) {
    bytes.push_back(static_cast<std::uint8_t>(byte));
  }
  return bytes;
}

}  // namespace test

#define CHECK(expression) ::test::check((expression), #expression, __FILE__, __LINE__)
