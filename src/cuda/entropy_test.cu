// CUDA AV2 §8.2 entropy port — correctness and performance tests.
//
// The CPU reference is av2enc::av2::Encoder/Decoder (src/entropy/av2/), the
// same state machine the normative module tests.  Every GPU case is compared
// byte-for-byte, plus final CDFs, counters, and the bytes past `sz` (the GPU
// path must not write beyond the payload).
//
// Build/run:
//   ./src/cuda/build_entropy_test.sh
//   ARCH=sm_120 ./src/cuda/build_entropy_test.sh
//   ./build/av2_cuda_entropy_test --no-bench

#include "entropy.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace {

using av2enc::av2::Decoder;
using av2enc::av2::Encoder;
using av2enc::av2::Status;
using av2enc::av2::status_name;
using av2enc::cuda::EntropyDecodeDesc;
using av2enc::cuda::EntropyEncodeDesc;
using av2enc::cuda::EntropyOp;
using av2enc::cuda::EntropyTapeEntry;
using av2enc::cuda::EntropyTileResult;

constexpr uint32_t kCdfEntries = 10;  // n <= 8, plus rate/count; room for n=9 test
constexpr uint8_t kGarbageFill = 0xCD;

// ---------------------------------------------------------------------------
// Test plumbing
// ---------------------------------------------------------------------------

int g_checks = 0;
int g_failures = 0;
int g_printed = 0;

void report_failure(const std::string& label, const char* what) {
  ++g_failures;
  if (g_printed < 40) {
    std::fprintf(stderr, "FAIL [%s] %s\n", label.c_str(), what);
    ++g_printed;
  }
}

void expect_true(bool condition, const std::string& label, const char* what) {
  ++g_checks;
  if (!condition) {
    report_failure(label, what);
  }
}

void expect_status(Status actual, Status expected, const std::string& label,
                   const char* what) {
  ++g_checks;
  if (actual != expected) {
    if (g_printed < 40) {
      std::fprintf(stderr, "FAIL [%s] %s: got %s, want %s\n", label.c_str(),
                   what, status_name(actual), status_name(expected));
      ++g_printed;
    }
    ++g_failures;
  }
}

void expect_u64(uint64_t actual, uint64_t expected, const std::string& label,
                const char* what) {
  ++g_checks;
  if (actual != expected) {
    if (g_printed < 40) {
      std::fprintf(stderr, "FAIL [%s] %s: got %" PRIu64 ", want %" PRIu64 "\n",
                   label.c_str(), what, actual, expected);
      ++g_printed;
    }
    ++g_failures;
  }
}

void expect_bytes(const uint8_t* actual, const uint8_t* expected, size_t count,
                  const std::string& label, const char* what) {
  ++g_checks;
  if (count != 0u && std::memcmp(actual, expected, count) != 0) {
    size_t first = 0;
    while (first < count && actual[first] == expected[first]) {
      ++first;
    }
    if (g_printed < 40) {
      std::fprintf(stderr,
                   "FAIL [%s] %s: first difference at byte %zu "
                   "(got 0x%02x, want 0x%02x), %zu bytes\n",
                   label.c_str(), what, first, actual[first], expected[first],
                   count);
      ++g_printed;
    }
    ++g_failures;
  }
}

void expect_word_vectors(const std::vector<uint16_t>& actual,
                         const std::vector<uint16_t>& expected,
                         const std::string& label, const char* what) {
  expect_u64(actual.size(), expected.size(), label, what);
  if (actual.size() != expected.size()) {
    return;
  }
  expect_bytes(reinterpret_cast<const uint8_t*>(actual.data()),
               reinterpret_cast<const uint8_t*>(expected.data()),
               actual.size() * sizeof(uint16_t), label, what);
}

void cuda_check(cudaError_t error, const char* what) {
  ++g_checks;
  if (error != cudaSuccess) {
    std::fprintf(stderr, "FATAL cuda %s: %s\n", what, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr)

std::string hex_string(const std::vector<uint8_t>& bytes) {
  static const char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (uint8_t byte : bytes) {
    out.push_back(kHex[byte >> 4u]);
    out.push_back(kHex[byte & 0x0fu]);
  }
  return out;
}

uint32_t rnd_below(std::mt19937_64& rng, uint32_t bound) {
  if (bound == 0u) {
    return 0u;
  }
  return static_cast<uint32_t>(rng() % bound);
}

// ---------------------------------------------------------------------------
// CDF helpers (same shapes as src/entropy/av2/av2_entropy_test.cpp)
// ---------------------------------------------------------------------------

struct Cdf {
  std::array<uint16_t, kCdfEntries> v{};
  int n = 0;

  uint16_t* data() { return v.data(); }
  const uint16_t* data() const { return v.data(); }
};

Cdf make_uniform_cdf(int n, uint16_t rate, uint16_t count) {
  Cdf c;
  c.n = n;
  for (int i = 0; i < n - 1; ++i) {
    c.v[static_cast<size_t>(i)] = static_cast<uint16_t>(
        (32768u * static_cast<uint32_t>(i + 1)) /
        static_cast<uint32_t>(n));
  }
  c.v[static_cast<size_t>(n - 1)] = rate;
  c.v[static_cast<size_t>(n)] = count;
  return c;
}

Cdf make_list_cdf(std::initializer_list<uint16_t> values) {
  Cdf c;
  int i = 0;
  for (uint16_t value : values) {
    c.v[static_cast<size_t>(i++)] = value;
  }
  c.n = i - 1;
  return c;
}

Cdf make_random_cdf(std::mt19937_64& rng, int n, uint16_t rate,
                    uint16_t count) {
  Cdf c;
  c.n = n;
  uint32_t previous = 0;
  for (int i = 0; i < n - 1; ++i) {
    const uint32_t lower = previous + 1u;
    const uint32_t remaining = static_cast<uint32_t>(n - 2 - i);
    const uint32_t upper = 32767u - remaining;
    const uint32_t span = upper - lower + 1u;
    const uint32_t value = lower + rnd_below(rng, span);
    c.v[static_cast<size_t>(i)] = static_cast<uint16_t>(value);
    previous = value;
  }
  c.v[static_cast<size_t>(n - 1)] = rate;
  c.v[static_cast<size_t>(n)] = count;
  return c;
}

// ---------------------------------------------------------------------------
// CPU reference: same logical tape, driven by av2::Encoder / av2::Decoder
// ---------------------------------------------------------------------------

struct HostEncoded {
  Status status = Status::kOk;
  std::vector<uint8_t> buffer;  // full capacity, 0xCD fill
  size_t payload_bytes = 0;
  std::vector<Cdf> cdfs;
  uint64_t symbol_count = 0;
  uint64_t bool_count = 0;
  uint32_t tape_used = 0;
};

// Same policy as the kernel: default frozen, explicit opt-in to adapt.
bool cdf_updates_enabled(uint32_t flags) {
  return ((flags & av2enc::cuda::kEntropyEnableCdfUpdate) != 0u) &&
         ((flags & av2enc::cuda::kEntropyDisableCdfUpdate) == 0u);
}

HostEncoded host_encode(const std::vector<EntropyTapeEntry>& tape,
                        const std::vector<Cdf>& cdfs, size_t capacity,
                        uint32_t flags) {
  HostEncoded result;
  result.cdfs = cdfs;
  result.buffer.assign(capacity, kGarbageFill);

  Encoder encoder;
  encoder.disable_cdf_update = cdf_updates_enabled(flags) ? 0u : 1u;

  Status status = encoder.init_symbol(result.buffer.data(), capacity);
  uint32_t used = 0;
  if (status == Status::kOk) {
    for (size_t i = 0; i < tape.size(); ++i) {
      used = static_cast<uint32_t>(i + 1u);
      const EntropyTapeEntry& entry = tape[i];
      const uint8_t op = entry.op;
      if (op == static_cast<uint8_t>(EntropyOp::kEnd)) {
        break;
      }
      if (op == static_cast<uint8_t>(EntropyOp::kSymbol)) {
        if (static_cast<uint32_t>(entry.cdf_id) >= cdfs.size() ||
            static_cast<uint32_t>(entry.n) + 1u > kCdfEntries) {
          status = Status::kInvalidArgument;
          break;
        }
        status = encoder.encode_symbol(
            result.cdfs[entry.cdf_id].data(), static_cast<int>(entry.n),
            static_cast<int>(entry.symbol));
      } else if (op == static_cast<uint8_t>(EntropyOp::kBool)) {
        status = encoder.encode_bool(entry.symbol != 0u);
      } else if (op == static_cast<uint8_t>(EntropyOp::kLiteral)) {
        if (entry.n > 32u) {
          status = Status::kInvalidArgument;
          break;
        }
        for (uint32_t k = 0; k < entry.n; ++k) {
          const uint32_t bit = (entry.literal >> (entry.n - 1u - k)) & 1u;
          status = encoder.encode_bool(bit != 0u);
          if (status != Status::kOk) {
            break;
          }
        }
      } else {
        status = Status::kInvalidArgument;
      }
      if (status != Status::kOk) {
        break;
      }
    }
  }

  size_t payload_bytes = 0;
  if (status == Status::kOk) {
    status = encoder.exit_symbol(&payload_bytes);
  }

  result.status = status;
  result.payload_bytes = (status == Status::kOk) ? payload_bytes : 0u;
  result.symbol_count = encoder.symbol_count;
  result.bool_count = encoder.bool_count;
  result.tape_used = used;
  return result;
}

struct HostDecoded {
  Status status = Status::kOk;
  std::vector<EntropyTapeEntry> tape;
  std::vector<Cdf> cdfs;
  uint64_t symbol_count = 0;
  uint64_t bool_count = 0;
  uint32_t tape_used = 0;
};

HostDecoded host_decode(const std::vector<EntropyTapeEntry>& tape,
                        const std::vector<Cdf>& cdfs,
                        const std::vector<uint8_t>& backing, size_t sz,
                        uint32_t flags) {
  HostDecoded result;
  result.tape = tape;
  result.cdfs = cdfs;

  Decoder decoder;
  decoder.disable_cdf_update = cdf_updates_enabled(flags) ? 0u : 1u;

  Status status =
      decoder.init_symbol(backing.data(), backing.size(), sz);
  uint32_t used = 0;
  if (status == Status::kOk) {
    for (size_t i = 0; i < result.tape.size(); ++i) {
      used = static_cast<uint32_t>(i + 1u);
      EntropyTapeEntry& entry = result.tape[i];
      const uint8_t op = entry.op;
      if (op == static_cast<uint8_t>(EntropyOp::kEnd)) {
        break;
      }
      if (op == static_cast<uint8_t>(EntropyOp::kSymbol)) {
        if (static_cast<uint32_t>(entry.cdf_id) >= cdfs.size() ||
            static_cast<uint32_t>(entry.n) + 1u > kCdfEntries) {
          status = Status::kInvalidArgument;
          break;
        }
        int symbol = 0;
        status = decoder.decode_symbol(
            result.cdfs[entry.cdf_id].data(), static_cast<int>(entry.n),
            &symbol);
        if (status == Status::kOk) {
          entry.symbol = static_cast<uint16_t>(symbol);
        }
      } else if (op == static_cast<uint8_t>(EntropyOp::kBool)) {
        bool bit = false;
        status = decoder.read_bool(&bit);
        if (status == Status::kOk) {
          entry.symbol = bit ? 1u : 0u;
        }
      } else if (op == static_cast<uint8_t>(EntropyOp::kLiteral)) {
        uint32_t literal = 0;
        status = decoder.read_literal(static_cast<int>(entry.n), &literal);
        if (status == Status::kOk) {
          entry.literal = literal;
        }
      } else {
        status = Status::kInvalidArgument;
      }
      if (status != Status::kOk) {
        break;
      }
    }
  }

  if (status == Status::kOk) {
    status = decoder.exit_symbol();
  }

  result.status = status;
  result.symbol_count = decoder.symbol_count;
  result.bool_count = decoder.bool_count;
  result.tape_used = used;
  return result;
}

// ---------------------------------------------------------------------------
// Device buffers and launch plans
// ---------------------------------------------------------------------------

template <typename T>
class DeviceArray {
 public:
  DeviceArray() = default;
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  void alloc(size_t count) {
    count_ = count;
    const size_t bytes = (count != 0u ? count : 1u) * sizeof(T);
    void* ptr = nullptr;
    cuda_check(cudaMalloc(&ptr, bytes), "cudaMalloc");
    ptr_ = static_cast<T*>(ptr);
  }

  void upload(const std::vector<T>& host) {
    if (!host.empty()) {
      cuda_check(cudaMemcpy(ptr_, host.data(), host.size() * sizeof(T),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy H2D");
    }
  }

  void download(std::vector<T>& host) const {
    host.resize(count_);
    if (count_ != 0u) {
      cuda_check(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy D2H");
    }
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

struct TileCase {
  std::vector<EntropyTapeEntry> tape;
  std::vector<Cdf> cdfs;
  size_t capacity = 4096;
  uint32_t flags = 0;
};

class EncodePlan {
 public:
  explicit EncodePlan(const std::vector<TileCase>& cases) {
    tiles_ = static_cast<uint32_t>(cases.size());
    std::vector<EntropyTapeEntry> flat_tape;
    std::vector<uint16_t> flat_cdfs;
    std::vector<uint8_t> flat_out;

    uint64_t tape_total = 0;
    uint64_t cdf_total = 0;
    uint64_t out_total = 0;
    for (const TileCase& c : cases) {
      tape_off_.push_back(static_cast<uint32_t>(tape_total));
      tape_len_.push_back(static_cast<uint32_t>(c.tape.size()));
      cdf_off_.push_back(static_cast<uint32_t>(cdf_total));
      cdf_count_.push_back(static_cast<uint32_t>(c.cdfs.size()));
      out_off_.push_back(static_cast<uint32_t>(out_total));
      out_cap_.push_back(static_cast<uint32_t>(c.capacity));
      tape_total += c.tape.size();
      cdf_total += c.cdfs.size() * kCdfEntries;
      out_total += (c.capacity != 0u) ? c.capacity : 1u;
    }
    flat_tape.reserve(static_cast<size_t>(tape_total));
    flat_cdfs.reserve(static_cast<size_t>(cdf_total));
    flat_out.assign(static_cast<size_t>(out_total), kGarbageFill);
    for (const TileCase& c : cases) {
      flat_tape.insert(flat_tape.end(), c.tape.begin(), c.tape.end());
      for (const Cdf& cdf : c.cdfs) {
        flat_cdfs.insert(flat_cdfs.end(), cdf.v.begin(), cdf.v.end());
      }
    }

    initial_cdfs_ = flat_cdfs;
    d_tape_.alloc(static_cast<size_t>(tape_total));
    d_cdfs_.alloc(static_cast<size_t>(cdf_total));
    d_out_.alloc(static_cast<size_t>(out_total));
    d_desc_.alloc(cases.size());
    d_results_.alloc(cases.size());
    d_tape_.upload(flat_tape);
    d_cdfs_.upload(flat_cdfs);
    d_out_.upload(flat_out);

    std::vector<EntropyEncodeDesc> descs(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) {
      EntropyEncodeDesc& d = descs[i];
      d = EntropyEncodeDesc{};
      d.tape = d_tape_.get() + tape_off_[i];
      d.tape_entries = tape_len_[i];
      d.flags = cases[i].flags;
      d.cdfs = d_cdfs_.get() + cdf_off_[i];
      d.cdf_count = cdf_count_[i];
      d.cdf_entries = kCdfEntries;
      d.out = d_out_.get() + out_off_[i];
      d.out_capacity = out_cap_[i];
    }
    d_desc_.upload(descs);
  }

  // Every launch adapts the caller's CDF copies in place.  A benchmark that
  // relaunches the same plan must restore the initial CDFs first, otherwise
  // each launch starts from the previous launch's final model.
  void reset_cdfs() { d_cdfs_.upload(initial_cdfs_); }

  cudaError_t launch(cudaStream_t stream = nullptr) {
    return av2enc::cuda::launch_entropy_encode(d_desc_.get(), d_results_.get(),
                                               tiles_, stream);
  }

  void download_results(std::vector<EntropyTileResult>& out) const {
    d_results_.download(out);
  }

  void download_bytes(std::vector<std::vector<uint8_t>>& out) const {
    out.resize(tiles_);
    for (uint32_t i = 0; i < tiles_; ++i) {
      out[i].resize(out_cap_[i]);
      if (out_cap_[i] != 0u) {
        cuda_check(cudaMemcpy(out[i].data(), d_out_.get() + out_off_[i],
                              out_cap_[i], cudaMemcpyDeviceToHost),
                   "cudaMemcpy out D2H");
      }
    }
  }

  void download_cdfs(std::vector<std::vector<uint16_t>>& out) const {
    out.resize(tiles_);
    for (uint32_t i = 0; i < tiles_; ++i) {
      const uint32_t words = cdf_count_[i] * kCdfEntries;
      out[i].resize(words);
      if (words != 0u) {
        cuda_check(cudaMemcpy(out[i].data(), d_cdfs_.get() + cdf_off_[i],
                              words * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy cdf D2H");
      }
    }
  }

 private:
  uint32_t tiles_ = 0;
  std::vector<uint32_t> tape_off_;
  std::vector<uint32_t> tape_len_;
  std::vector<uint32_t> cdf_off_;
  std::vector<uint32_t> cdf_count_;
  std::vector<uint32_t> out_off_;
  std::vector<uint32_t> out_cap_;
  std::vector<uint16_t> initial_cdfs_;
  DeviceArray<EntropyTapeEntry> d_tape_;
  DeviceArray<uint16_t> d_cdfs_;
  DeviceArray<uint8_t> d_out_;
  DeviceArray<EntropyEncodeDesc> d_desc_;
  DeviceArray<EntropyTileResult> d_results_;
};

struct DecodeTile {
  std::vector<EntropyTapeEntry> tape;
  std::vector<Cdf> cdfs;
  std::vector<uint8_t> backing;
  size_t sz = 0;
  uint32_t flags = 0;
};

class DecodePlan {
 public:
  explicit DecodePlan(const std::vector<DecodeTile>& tiles) {
    tiles_ = static_cast<uint32_t>(tiles.size());
    std::vector<EntropyTapeEntry> flat_tape;
    std::vector<uint16_t> flat_cdfs;
    std::vector<uint8_t> flat_in;

    uint64_t tape_total = 0;
    uint64_t cdf_total = 0;
    uint64_t in_total = 0;
    for (const DecodeTile& t : tiles) {
      tape_off_.push_back(static_cast<uint32_t>(tape_total));
      tape_len_.push_back(static_cast<uint32_t>(t.tape.size()));
      cdf_off_.push_back(static_cast<uint32_t>(cdf_total));
      cdf_count_.push_back(static_cast<uint32_t>(t.cdfs.size()));
      in_off_.push_back(static_cast<uint32_t>(in_total));
      in_len_.push_back(static_cast<uint32_t>(t.backing.size()));
      sizes_.push_back(static_cast<uint32_t>(t.sz));
      tape_total += t.tape.size();
      cdf_total += t.cdfs.size() * kCdfEntries;
      in_total += (t.backing.size() != 0u) ? t.backing.size() : 1u;
    }
    flat_tape.reserve(static_cast<size_t>(tape_total));
    flat_cdfs.reserve(static_cast<size_t>(cdf_total));
    flat_in.assign(static_cast<size_t>(in_total), 0);
    for (const DecodeTile& t : tiles) {
      flat_tape.insert(flat_tape.end(), t.tape.begin(), t.tape.end());
      for (const Cdf& cdf : t.cdfs) {
        flat_cdfs.insert(flat_cdfs.end(), cdf.v.begin(), cdf.v.end());
      }
    }
    size_t cursor = 0;
    for (const DecodeTile& t : tiles) {
      std::copy(t.backing.begin(), t.backing.end(),
                flat_in.begin() + static_cast<ptrdiff_t>(cursor));
      cursor += (t.backing.size() != 0u) ? t.backing.size() : 1u;
    }

    initial_cdfs_ = flat_cdfs;
    d_tape_.alloc(static_cast<size_t>(tape_total));
    d_cdfs_.alloc(static_cast<size_t>(cdf_total));
    d_in_.alloc(static_cast<size_t>(in_total));
    d_desc_.alloc(tiles.size());
    d_results_.alloc(tiles.size());
    d_tape_.upload(flat_tape);
    d_cdfs_.upload(flat_cdfs);
    d_in_.upload(flat_in);

    std::vector<EntropyDecodeDesc> descs(tiles.size());
    for (size_t i = 0; i < tiles.size(); ++i) {
      EntropyDecodeDesc& d = descs[i];
      d = EntropyDecodeDesc{};
      d.in = d_in_.get() + in_off_[i];
      d.payload_bytes = sizes_[i];
      d.backing_bytes = in_len_[i];
      d.tape = d_tape_.get() + tape_off_[i];
      d.tape_entries = tape_len_[i];
      d.flags = tiles[i].flags;
      d.cdfs = d_cdfs_.get() + cdf_off_[i];
      d.cdf_count = cdf_count_[i];
      d.cdf_entries = kCdfEntries;
    }
    d_desc_.upload(descs);
  }

  void reset_cdfs() { d_cdfs_.upload(initial_cdfs_); }

  cudaError_t launch(cudaStream_t stream = nullptr) {
    return av2enc::cuda::launch_entropy_decode(d_desc_.get(), d_results_.get(),
                                               tiles_, stream);
  }

  void download_results(std::vector<EntropyTileResult>& out) const {
    d_results_.download(out);
  }

  void download_tapes(std::vector<std::vector<EntropyTapeEntry>>& out) const {
    out.resize(tiles_);
    for (uint32_t i = 0; i < tiles_; ++i) {
      out[i].resize(tape_len_[i]);
      if (tape_len_[i] != 0u) {
        cuda_check(cudaMemcpy(out[i].data(), d_tape_.get() + tape_off_[i],
                              tape_len_[i] * sizeof(EntropyTapeEntry),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy tape D2H");
      }
    }
  }

  void download_cdfs(std::vector<std::vector<uint16_t>>& out) const {
    out.resize(tiles_);
    for (uint32_t i = 0; i < tiles_; ++i) {
      const uint32_t words = cdf_count_[i] * kCdfEntries;
      out[i].resize(words);
      if (words != 0u) {
        cuda_check(cudaMemcpy(out[i].data(), d_cdfs_.get() + cdf_off_[i],
                              words * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy cdf D2H");
      }
    }
  }

 private:
  uint32_t tiles_ = 0;
  std::vector<uint32_t> tape_off_;
  std::vector<uint32_t> tape_len_;
  std::vector<uint32_t> cdf_off_;
  std::vector<uint32_t> cdf_count_;
  std::vector<uint32_t> in_off_;
  std::vector<uint32_t> in_len_;
  std::vector<uint32_t> sizes_;
  std::vector<uint16_t> initial_cdfs_;
  DeviceArray<EntropyTapeEntry> d_tape_;
  DeviceArray<uint16_t> d_cdfs_;
  DeviceArray<uint8_t> d_in_;
  DeviceArray<EntropyDecodeDesc> d_desc_;
  DeviceArray<EntropyTileResult> d_results_;
};

struct GpuEncoded {
  std::vector<EntropyTileResult> results;
  std::vector<std::vector<uint8_t>> bytes;
  std::vector<std::vector<uint16_t>> cdf_words;
};

GpuEncoded gpu_encode(const std::vector<TileCase>& cases) {
  EncodePlan plan(cases);
  CUDA_CHECK(plan.launch());
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuEncoded out;
  plan.download_results(out.results);
  plan.download_bytes(out.bytes);
  plan.download_cdfs(out.cdf_words);
  return out;
}

struct GpuDecoded {
  std::vector<EntropyTileResult> results;
  std::vector<std::vector<EntropyTapeEntry>> tapes;
  std::vector<std::vector<uint16_t>> cdf_words;
};

GpuDecoded gpu_decode(const std::vector<DecodeTile>& tiles) {
  DecodePlan plan(tiles);
  CUDA_CHECK(plan.launch());
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuDecoded out;
  plan.download_results(out.results);
  plan.download_tapes(out.tapes);
  plan.download_cdfs(out.cdf_words);
  return out;
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

void compare_cdf_words(const std::vector<Cdf>& expected,
                       const std::vector<uint16_t>& actual,
                       const std::string& label) {
  expect_u64(actual.size(),
             expected.size() * static_cast<size_t>(kCdfEntries),
             label, "cdf word count");
  if (actual.size() != expected.size() * kCdfEntries) {
    return;
  }
  for (size_t c = 0; c < expected.size(); ++c) {
    for (int i = 0; i <= expected[c].n; ++i) {
      expect_u64(actual[c * kCdfEntries + static_cast<size_t>(i)],
                 expected[c].v[static_cast<size_t>(i)], label, "final cdf");
    }
  }
}

void check_encode_case(const TileCase& c, const GpuEncoded& g, size_t index,
                       const std::string& label) {
  const HostEncoded host =
      host_encode(c.tape, c.cdfs, c.capacity, c.flags);
  const EntropyTileResult& r = g.results[index];

  expect_status(static_cast<Status>(r.status), host.status, label, "status");
  expect_u64(r.symbol_count, host.symbol_count, label, "symbol_count");
  expect_u64(r.bool_count, host.bool_count, label, "bool_count");
  expect_u64(r.tape_entries_used, host.tape_used, label, "tape used");

  if (host.status != Status::kOk) {
    return;
  }

  expect_u64(r.payload_bytes, host.payload_bytes, label, "payload_bytes");
  if (r.payload_bytes != host.payload_bytes) {
    return;
  }
  const size_t sz = host.payload_bytes;
  expect_bytes(g.bytes[index].data(), host.buffer.data(), sz, label, "payload");
  // The coder must not touch a byte past `sz`.
  for (size_t i = sz; i < c.capacity; ++i) {
    if (g.bytes[index][i] != kGarbageFill) {
      char what[96];
      std::snprintf(what, sizeof what, "wrote past payload at byte %zu", i);
      expect_true(false, label, what);
      break;
    }
  }
  for (size_t i = sz; i < c.capacity; ++i) {
    if (host.buffer[i] != kGarbageFill) {
      char what[96];
      std::snprintf(what, sizeof what, "CPU wrote past payload at byte %zu", i);
      expect_true(false, label, what);
      break;
    }
  }
  compare_cdf_words(host.cdfs, g.cdf_words[index], label);
}

void check_decode_case(const DecodeTile& t, const GpuDecoded& g, size_t index,
                       const std::string& label) {
  const HostDecoded host =
      host_decode(t.tape, t.cdfs, t.backing, t.sz, t.flags);
  const EntropyTileResult& r = g.results[index];

  expect_status(static_cast<Status>(r.status), host.status, label, "status");
  expect_u64(r.symbol_count, host.symbol_count, label, "symbol_count");
  expect_u64(r.bool_count, host.bool_count, label, "bool_count");
  expect_u64(r.tape_entries_used, host.tape_used, label, "tape used");
  compare_cdf_words(host.cdfs, g.cdf_words[index], label);

  const std::vector<EntropyTapeEntry>& got = g.tapes[index];
  expect_u64(got.size(), host.tape.size(), label, "decoded tape size");
  if (got.size() != host.tape.size()) {
    return;
  }
  for (size_t i = 0; i < got.size(); ++i) {
    expect_u64(got[i].op, host.tape[i].op, label, "decoded tape op");
    expect_u64(got[i].n, host.tape[i].n, label, "decoded tape n");
    expect_u64(got[i].cdf_id, host.tape[i].cdf_id, label, "decoded tape cdf_id");
    expect_u64(got[i].symbol, host.tape[i].symbol, label, "decoded symbol");
    expect_u64(got[i].literal, host.tape[i].literal, label, "decoded literal");
  }
}

// ---------------------------------------------------------------------------
// Tape builders
// ---------------------------------------------------------------------------

std::vector<EntropyTapeEntry> symbol_tape(const std::vector<int>& symbols,
                                          int n) {
  std::vector<EntropyTapeEntry> tape;
  tape.reserve(symbols.size());
  for (int symbol : symbols) {
    tape.push_back(av2enc::cuda::make_symbol_entry(
        0, static_cast<uint8_t>(n), static_cast<uint16_t>(symbol)));
  }
  return tape;
}

std::vector<EntropyTapeEntry> random_tape(std::mt19937_64& rng,
                                          uint32_t entries,
                                          const std::vector<Cdf>& cdfs,
                                          bool allow_literals) {
  std::vector<EntropyTapeEntry> tape;
  tape.reserve(entries);
  for (uint32_t i = 0; i < entries; ++i) {
    const uint32_t pick = rnd_below(rng, 100u);
    if (pick < 62u && !cdfs.empty()) {
      const uint32_t cdf_id = rnd_below(rng, static_cast<uint32_t>(cdfs.size()));
      const uint8_t n = static_cast<uint8_t>(cdfs[cdf_id].n);
      const uint16_t symbol =
          static_cast<uint16_t>(rnd_below(rng, static_cast<uint32_t>(n)));
      tape.push_back(av2enc::cuda::make_symbol_entry(
          static_cast<uint16_t>(cdf_id), n, symbol));
    } else if (pick < 82u || !allow_literals) {
      tape.push_back(av2enc::cuda::make_bool_entry(
          rnd_below(rng, 2u) != 0u));
    } else {
      const uint8_t width = static_cast<uint8_t>(rnd_below(rng, 33u));
      const uint32_t value = static_cast<uint32_t>(rng());
      tape.push_back(av2enc::cuda::make_literal_entry(width, value));
    }
  }
  return tape;
}

std::vector<Cdf> make_cdf_set(std::mt19937_64& rng, uint32_t count,
                              bool randomize) {
  std::vector<Cdf> cdfs;
  cdfs.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    const int n = 2 + static_cast<int>(rnd_below(rng, 7u));
    const uint16_t rate = static_cast<uint16_t>(rnd_below(rng, 125u));
    const uint16_t count_value = static_cast<uint16_t>(rnd_below(rng, 33u));
    if (randomize) {
      cdfs.push_back(make_random_cdf(rng, n, rate, count_value));
    } else {
      cdfs.push_back(make_uniform_cdf(n, rate, count_value));
    }
  }
  return cdfs;
}

// ---------------------------------------------------------------------------
// Golden vectors pinned by src/entropy/av2/av2_entropy_test.cpp
// ---------------------------------------------------------------------------

void test_golden_vectors() {
  std::printf("== golden vectors ==\n");
  struct Golden {
    const char* name;
    Cdf cdf;
    std::vector<int> symbols;
    bool disable;
    const char* hex;
    std::vector<uint16_t> final_cdf;
  };

  std::vector<Golden> cases;
  cases.push_back({"mv_joint_adapt", make_list_cdf({4, 17705, 32748, 1, 0}),
                   {0, 1, 2, 3, 3, 2, 1, 0, 3, 0, 2, 1, 1, 0, 3, 2}, false,
                   "002357f288ec788000", {3243, 17149, 29519, 1, 16}});
  cases.push_back({"mv_joint_static", make_list_cdf({4, 17705, 32748, 1, 0}),
                   {0, 1, 2, 3, 3, 2, 1, 0, 3, 0, 2, 1, 1, 0, 3, 2}, true,
                   "002297ff998df6015ceffc0000", {4, 17705, 32748, 1, 0}});
  cases.push_back({"delta_q_adapt",
                   make_list_cdf(
                       {16594, 23325, 26424, 28225, 29358, 30099, 30613, 56, 0}),
                   {0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                    0, 0, 7, 7, 3, 4, 2, 1, 5, 6},
                   false, "5cf061685f61fa8346e55168200000",
                   {6377, 10984, 14988, 18614, 22047, 25415, 28717, 56, 26}});
  cases.push_back({"delta_q_static",
                   make_list_cdf(
                       {16594, 23325, 26424, 28225, 29358, 30099, 30613, 56, 0}),
                   {0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
                    0, 0, 7, 7, 3, 4, 2, 1, 5, 6},
                   true, "566bfde33de90f9507f8a60b600000",
                   {16594, 23325, 26424, 28225, 29358, 30099, 30613, 56, 0}});
  cases.push_back({"binary_adapt", make_list_cdf({16384, 0, 0}),
                   {1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0},
                   false, "9aeefc0000", {16780, 0, 20}});
  cases.push_back({"binary_static", make_list_cdf({16384, 0, 0}),
                   {1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0},
                   true, "9cced80000", {16384, 0, 0}});
  cases.push_back({"n3_adapt", make_list_cdf({10000, 20000, 5, 0}),
                   {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                    0, 1, 2, 0, 1},
                   false, "2badb29b390000", {10859, 22283, 5, 23}});
  cases.push_back({"n3_static", make_list_cdf({10000, 20000, 5, 0}),
                   {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                    0, 1, 2, 0, 1},
                   true, "285f92a4740000", {10000, 20000, 5, 0}});

  for (const Golden& golden : cases) {
    TileCase c;
    c.tape = symbol_tape(golden.symbols, golden.cdf.n);
    c.cdfs.push_back(golden.cdf);
    c.capacity = 4096;
    c.flags = golden.disable ? 0u : av2enc::cuda::kEntropyEnableCdfUpdate;

    const HostEncoded host =
        host_encode(c.tape, c.cdfs, c.capacity, c.flags);
    expect_status(host.status, Status::kOk, golden.name, "CPU status");
    std::vector<uint8_t> cpu_payload(host.buffer.begin(),
                                     host.buffer.begin() +
                                         static_cast<ptrdiff_t>(
                                             host.payload_bytes));
    expect_true(hex_string(cpu_payload) == golden.hex, golden.name,
                "CPU golden hex");

    const GpuEncoded gpu = gpu_encode({c});
    const std::string label = std::string("golden/") + golden.name;
    check_encode_case(c, gpu, 0, label);

    if (gpu.results[0].status == static_cast<uint32_t>(Status::kOk)) {
      std::vector<uint8_t> payload(
          gpu.bytes[0].begin(),
          gpu.bytes[0].begin() +
              static_cast<ptrdiff_t>(gpu.results[0].payload_bytes));
      expect_true(hex_string(payload) == golden.hex, label, "GPU golden hex");
    }

    // Decode the GPU payload with the GPU decoder and compare the tape.
    DecodeTile d;
    d.tape = c.tape;
    d.cdfs = c.cdfs;
    d.backing = gpu.bytes[0];
    d.sz = gpu.results[0].payload_bytes;
    d.flags = c.flags;
    const GpuDecoded dec = gpu_decode({d});
    check_decode_case(d, dec, 0, label + "/decode");
  }

  // Boolean-only golden vector, generated by the same §8.2.3 state machine.
  {
    const bool bits[] = {true,  false, false, true,  true,  true,  false,
                         true,  false, false, true,  true,  false, false,
                         false, true,  true,  false, true,  false};
    TileCase c;
    for (bool bit : bits) {
      c.tape.push_back(av2enc::cuda::make_bool_entry(bit));
    }
    c.capacity = 64;
    const GpuEncoded gpu = gpu_encode({c});
    std::vector<uint8_t> payload(
        gpu.bytes[0].begin(),
        gpu.bytes[0].begin() +
            static_cast<ptrdiff_t>(gpu.results[0].payload_bytes));
    expect_true(hex_string(payload) == "9d31a80000", "golden/bools",
                "bools hex");
    check_encode_case(c, gpu, 0, "golden/bools");
  }

  // Interleaved symbols and bools: pinned hex and pinned final CDFs.
  {
    const int symbols[] = {0, 3, 2, 1, 1, 0, 3, 2, 2, 0, 1, 3};
    const bool bools[] = {true,  false, true,  true,  false, false,
                          true,  true,  false, false, true,  false};
    TileCase c;
    c.cdfs.push_back(make_uniform_cdf(4, 56, 0));
    for (size_t i = 0; i < 12; ++i) {
      c.tape.push_back(av2enc::cuda::make_symbol_entry(
          0, 4, static_cast<uint16_t>(symbols[i])));
      c.tape.push_back(av2enc::cuda::make_bool_entry(bools[i]));
    }
    c.capacity = 256;
    // The pinned CPU vector was generated with adaptation enabled.
    c.flags = av2enc::cuda::kEntropyEnableCdfUpdate;
    const GpuEncoded gpu = gpu_encode({c});
    std::vector<uint8_t> payload(
        gpu.bytes[0].begin(),
        gpu.bytes[0].begin() +
            static_cast<ptrdiff_t>(gpu.results[0].payload_bytes));
    expect_true(hex_string(payload) == "3a77858ac88000", "golden/interleaved",
                "interleaved hex");
    check_encode_case(c, gpu, 0, "golden/interleaved");
  }

  // Empty stream is 80 00; a one-byte 80 is also a conformant empty stream.
  {
    TileCase c;
    c.capacity = 16;
    const GpuEncoded gpu = gpu_encode({c});
    std::vector<uint8_t> payload(
        gpu.bytes[0].begin(),
        gpu.bytes[0].begin() +
            static_cast<ptrdiff_t>(gpu.results[0].payload_bytes));
    expect_true(hex_string(payload) == "8000", "golden/empty", "empty hex");
    check_encode_case(c, gpu, 0, "golden/empty");

    DecodeTile one;
    one.backing = {0x80};
    one.sz = 1;
    const GpuDecoded dec = gpu_decode({one});
    expect_status(static_cast<Status>(dec.results[0].status), Status::kOk,
                  "golden/empty80", "decode sz=1 status");
    check_decode_case(one, dec, 0, "golden/empty80");
  }
}

// ---------------------------------------------------------------------------
// Random symbol streams, alphabets 2..8, adaptation on/off
// ---------------------------------------------------------------------------

void test_random_symbols() {
  std::printf("== random symbols (n=2..8, adaptation on/off) ==\n");
  std::mt19937_64 rng(0x51A7E5EEDULL);
  const uint16_t rates[] = {0, 5, 56, 124};
  const uint16_t counts[] = {0, 5, 16, 31, 32};

  uint32_t case_id = 0;
  for (int n = 2; n <= 8; ++n) {
    for (uint16_t rate : rates) {
      for (uint16_t count : counts) {
        for (int disable = 0; disable <= 1; ++disable) {
          for (int randomize = 0; randomize <= 1; ++randomize) {
            TileCase c;
            Cdf cdf = randomize != 0
                          ? make_random_cdf(rng, n, rate, count)
                          : make_uniform_cdf(n, rate, count);
            c.cdfs.push_back(cdf);
            c.tape.reserve(400);
            for (int i = 0; i < 400; ++i) {
              c.tape.push_back(av2enc::cuda::make_symbol_entry(
                  0, static_cast<uint8_t>(n),
                  static_cast<uint16_t>(rnd_below(rng, static_cast<uint32_t>(n)))));
            }
            c.capacity = 4096;
            c.flags =
                (disable == 0) ? av2enc::cuda::kEntropyEnableCdfUpdate : 0u;
            const GpuEncoded gpu = gpu_encode({c});
            char label[128];
            std::snprintf(label, sizeof label,
                          "random n=%d rate=%u count=%u disable=%d rand=%d", n,
                          static_cast<unsigned>(rate),
                          static_cast<unsigned>(count), disable, randomize);
            check_encode_case(c, gpu, 0, label);
            ++case_id;
          }
        }
      }
    }
  }
  std::printf("   %u encode cases\n", case_id);
}

// ---------------------------------------------------------------------------
// Bool/literal interleaving, long tapes, staging overflow
// ---------------------------------------------------------------------------

void test_interleaved_and_staging() {
  std::printf("== interleaved bool/literal/symbol tapes ==\n");
  std::mt19937_64 rng(0xB0017A9EULL);
  for (int variant = 0; variant < 24; ++variant) {
    TileCase c;
    const uint32_t cdf_count = 1u + rnd_below(rng, 8u);
    c.cdfs = make_cdf_set(rng, cdf_count, (variant & 1) != 0);
    c.tape = random_tape(rng, 2000u, c.cdfs, true);
    c.capacity = 65536;
    c.flags = ((variant & 2) != 0) ? 0u
                                   : av2enc::cuda::kEntropyEnableCdfUpdate;
    const GpuEncoded gpu = gpu_encode({c});
    char label[64];
    std::snprintf(label, sizeof label, "interleaved/%d", variant);
    check_encode_case(c, gpu, 0, label);
  }

  // Literal widths 0..32, including 0 and 32, with and without symbols.
  {
    TileCase c;
    c.cdfs.push_back(make_uniform_cdf(3, 56, 0));
    for (uint8_t width = 0; width <= 32; ++width) {
      c.tape.push_back(av2enc::cuda::make_symbol_entry(
          0, 3, static_cast<uint16_t>(width % 3u)));
      c.tape.push_back(av2enc::cuda::make_literal_entry(
          width, static_cast<uint32_t>(rng())));
      c.tape.push_back(av2enc::cuda::make_bool_entry((width & 1u) != 0u));
    }
    c.capacity = 4096;
    const GpuEncoded gpu = gpu_encode({c});
    check_encode_case(c, gpu, 0, "literal widths 0..32");
  }

  // n == 1 alphabet: consumes no bits, no CDF update.
  {
    TileCase c;
    c.cdfs.push_back(make_list_cdf({0, 0}));
    for (int i = 0; i < 100; ++i) {
      c.tape.push_back(av2enc::cuda::make_symbol_entry(0, 1, 0));
    }
    c.tape.push_back(av2enc::cuda::make_bool_entry(true));
    c.capacity = 64;
    const GpuEncoded gpu = gpu_encode({c});
    check_encode_case(c, gpu, 0, "n=1 alphabet");
    expect_u64(gpu.results[0].payload_bytes, 2, "n=1 alphabet", "empty payload");
  }

  // Staging overflow: payload larger than kEntropyStageBytes (4 KiB), with
  // several flushed windows and carries that can cross a flush boundary.
  {
    const uint32_t lengths[] = {60000u, 200000u};
    for (size_t variant = 0; variant < 2; ++variant) {
      TileCase c;
      c.cdfs.push_back(make_uniform_cdf(8, 56, 0));
      c.tape.reserve(lengths[variant]);
      for (uint32_t i = 0; i < lengths[variant]; ++i) {
        c.tape.push_back(av2enc::cuda::make_symbol_entry(
            0, 8, static_cast<uint16_t>(rnd_below(rng, 8u))));
      }
      c.capacity = 262144;
      const GpuEncoded gpu = gpu_encode({c});
      char label[64];
      std::snprintf(label, sizeof label, "staging/%zu", variant);
      expect_true(gpu.results[0].payload_bytes >
                      av2enc::cuda::kEntropyStageBytes,
                  label, "payload must exceed one staging window");
      check_encode_case(c, gpu, 0, label);

      // Same stream with staging disabled: direct global bit writes.
      TileCase direct = c;
      direct.flags |= av2enc::cuda::kEntropyDisableStaging;
      const GpuEncoded gpu_direct = gpu_encode({direct});
      check_encode_case(direct, gpu_direct, 0,
                        std::string(label) + "/direct");
      expect_bytes(gpu_direct.bytes[0].data(), gpu.bytes[0].data(),
                   gpu.results[0].payload_bytes, label, "staged==direct");
    }
  }
}

// ---------------------------------------------------------------------------
// Empty and single-symbol streams
// ---------------------------------------------------------------------------

void test_empty_and_single() {
  std::printf("== empty / single-symbol streams ==\n");

  // Empty tape for every capacity from 0..3.
  for (uint32_t capacity = 0; capacity <= 3; ++capacity) {
    TileCase c;
    c.capacity = capacity;
    const GpuEncoded gpu = gpu_encode({c});
    char label[64];
    std::snprintf(label, sizeof label, "empty/cap=%u", capacity);
    check_encode_case(c, gpu, 0, label);
  }

  // One symbol for each alphabet size, and all N symbols.
  for (int n = 1; n <= 8; ++n) {
    for (int symbol = 0; symbol < n; ++symbol) {
      TileCase c;
      c.cdfs.push_back(make_uniform_cdf(n, 56, 0));
      c.tape.push_back(av2enc::cuda::make_symbol_entry(
          0, static_cast<uint8_t>(n),
          static_cast<uint16_t>(symbol)));
      c.capacity = 64;
      const GpuEncoded gpu = gpu_encode({c});
      char label[64];
      std::snprintf(label, sizeof label, "single n=%d sym=%d", n, symbol);
      check_encode_case(c, gpu, 0, label);
      expect_true(gpu.results[0].payload_bytes >= 2u, label, "at least 2 bytes");
    }
  }

  // Single literal of width 0 and width 32.
  for (uint32_t width : {0u, 32u}) {
    TileCase c;
    c.tape.push_back(av2enc::cuda::make_literal_entry(
        static_cast<uint8_t>(width), (width == 0u) ? 0u : 0xDEADBEEFu));
    c.capacity = 64;
    const GpuEncoded gpu = gpu_encode({c});
    char label[64];
    std::snprintf(label, sizeof label, "literal width=%u",
                  static_cast<unsigned>(width));
    check_encode_case(c, gpu, 0, label);
  }

  // Empty tape decoded from 0..2 declared bytes over a 2-byte backing store.
  for (size_t sz = 0; sz <= 2; ++sz) {
    DecodeTile d;
    d.backing = {0x80, 0x00};
    d.sz = sz;
    const GpuDecoded dec = gpu_decode({d});
    char label[64];
    std::snprintf(label, sizeof label, "empty decode sz=%zu", sz);
    check_decode_case(d, dec, 0, label);
  }
  // Zero-length payload whose backing first byte is not 0x80: the decoder
  // reads that byte in exit_symbol() (same behaviour as the CPU decoder).
  {
    DecodeTile d;
    d.backing = {0x00};
    d.sz = 0;
    const GpuDecoded dec = gpu_decode({d});
    check_decode_case(d, dec, 0, "empty decode sz=0 zero byte");
  }
}

// ---------------------------------------------------------------------------
// Multiple tiles: 4x2 and 8x2, with and without CDF staging
// ---------------------------------------------------------------------------

void test_multi_tile(uint32_t cols, uint32_t rows, bool staging_off) {
  std::printf("== multi-tile %ux%u (%u tiles%s) ==\n", cols, rows, cols * rows,
              staging_off ? ", staging off" : "");
  const uint32_t tiles = cols * rows;
  std::mt19937_64 rng(0x7E5EED00ULL + cols * 100u + rows);

  std::vector<TileCase> cases;
  cases.reserve(tiles);
  for (uint32_t t = 0; t < tiles; ++t) {
    TileCase c;
    const uint32_t cdf_count = 1u + rnd_below(rng, 6u);
    c.cdfs = make_cdf_set(rng, cdf_count, (t & 1u) != 0u);
    c.tape = random_tape(rng, 500u + t * 37u, c.cdfs, true);
    c.capacity = 32768;
    c.flags = ((t & 2u) != 0u) ? 0u
                               : av2enc::cuda::kEntropyEnableCdfUpdate;
    if (staging_off) {
      c.flags |= av2enc::cuda::kEntropyDisableStaging;
    }
    cases.push_back(c);
  }

  const GpuEncoded gpu = gpu_encode(cases);
  for (uint32_t t = 0; t < tiles; ++t) {
    char label[96];
    std::snprintf(label, sizeof label, "multi %ux%u tile %u", cols, rows, t);
    check_encode_case(cases[t], gpu, t, label);
  }

  // Tile independence: each tile run alone must match its batched output.
  for (uint32_t t = 0; t < tiles; ++t) {
    const GpuEncoded single = gpu_encode({cases[t]});
    char label[96];
    std::snprintf(label, sizeof label, "multi %ux%u solo %u", cols, rows, t);
    expect_bytes(single.bytes[0].data(), gpu.bytes[t].data(),
                 single.results[0].payload_bytes, label, "batch == solo bytes");
    expect_word_vectors(single.cdf_words[0], gpu.cdf_words[t],
                        std::string(label) + "/solo",
                        "batch == solo final cdf");
  }

  // Decode every tile from the batched payloads.
  std::vector<DecodeTile> decodes;
  decodes.reserve(tiles);
  for (uint32_t t = 0; t < tiles; ++t) {
    DecodeTile d;
    d.tape = cases[t].tape;
    d.cdfs = cases[t].cdfs;
    d.backing = gpu.bytes[t];
    d.sz = gpu.results[t].payload_bytes;
    d.flags = cases[t].flags & ~av2enc::cuda::kEntropyDisableStaging;
    decodes.push_back(d);
  }
  const GpuDecoded dec = gpu_decode(decodes);
  for (uint32_t t = 0; t < tiles; ++t) {
    char label[96];
    std::snprintf(label, sizeof label, "multi %ux%u decode %u", cols, rows, t);
    check_decode_case(decodes[t], dec, t, label);
  }
}

// ---------------------------------------------------------------------------
// Truncation and corruption
// ---------------------------------------------------------------------------

void test_truncation_and_corruption() {
  std::printf("== truncation / corruption ==\n");
  std::mt19937_64 rng(0xC0FFEEULL);

  TileCase base;
  base.cdfs.push_back(make_uniform_cdf(6, 56, 0));
  base.cdfs.push_back(make_random_cdf(rng, 5, 12, 4));
  for (int i = 0; i < 300; ++i) {
    const uint32_t pick = rnd_below(rng, 100u);
    if (pick < 70u) {
      const uint32_t cdf_id = rnd_below(rng, 2u);
      const uint8_t n = static_cast<uint8_t>(base.cdfs[cdf_id].n);
      base.tape.push_back(av2enc::cuda::make_symbol_entry(
          static_cast<uint16_t>(cdf_id), n,
          static_cast<uint16_t>(rnd_below(rng, static_cast<uint32_t>(n)))));
    } else if (pick < 90u) {
      base.tape.push_back(
          av2enc::cuda::make_bool_entry(rnd_below(rng, 2u) != 0u));
    } else {
      base.tape.push_back(av2enc::cuda::make_literal_entry(
          static_cast<uint8_t>(rnd_below(rng, 17u)),
          static_cast<uint32_t>(rng())));
    }
  }
  base.capacity = 4096;
  const GpuEncoded encoded = gpu_encode({base});
  expect_status(static_cast<Status>(encoded.results[0].status), Status::kOk,
                "truncation/base", "base encode");
  const size_t sz = encoded.results[0].payload_bytes;
  expect_true(sz > 8u, "truncation/base", "base payload size");

  // Every declared sz from 0..sz+2 against the same backing store.
  {
    std::vector<DecodeTile> decodes;
    for (size_t declared = 0; declared <= sz + 2; ++declared) {
      DecodeTile d;
      d.tape = base.tape;
      d.cdfs = base.cdfs;
      d.backing = encoded.bytes[0];
      d.sz = declared;
      decodes.push_back(d);
    }
    const GpuDecoded dec = gpu_decode(decodes);
    for (size_t i = 0; i < decodes.size(); ++i) {
      char label[80];
      std::snprintf(label, sizeof label, "truncation sz=%zu", decodes[i].sz);
      check_decode_case(decodes[i], dec, i, label);
    }
  }

  // sz > backing: kTruncatedInput on both sides.
  {
    std::vector<DecodeTile> decodes;
    for (size_t extra = 1; extra <= 4; ++extra) {
      DecodeTile d;
      d.tape = base.tape;
      d.cdfs = base.cdfs;
      d.backing.assign(sz, 0);
      std::copy(encoded.bytes[0].begin(),
                encoded.bytes[0].begin() + static_cast<ptrdiff_t>(sz),
                d.backing.begin());
      d.sz = sz + extra;
      decodes.push_back(d);
    }
    const GpuDecoded dec = gpu_decode(decodes);
    for (size_t i = 0; i < decodes.size(); ++i) {
      char label[80];
      std::snprintf(label, sizeof label, "truncation over sz=%zu",
                    decodes[i].sz);
      check_decode_case(decodes[i], dec, i, label);
      expect_status(static_cast<Status>(dec.results[i].status),
                    Status::kTruncatedInput, label, "truncated status");
    }
  }

  // 200 random single-bit flips of the payload.
  {
    std::vector<DecodeTile> decodes;
    for (int trial = 0; trial < 200; ++trial) {
      std::vector<uint8_t> corrupt(encoded.bytes[0].begin(),
                                   encoded.bytes[0].begin() +
                                       static_cast<ptrdiff_t>(sz));
      const uint32_t bit =
          rnd_below(rng, static_cast<uint32_t>(sz * 8u));
      corrupt[bit >> 3u] = static_cast<uint8_t>(
          corrupt[bit >> 3u] ^ (1u << (7u - (bit & 7u))));
      DecodeTile d;
      d.tape = base.tape;
      d.cdfs = base.cdfs;
      d.backing = corrupt;
      d.sz = sz;
      decodes.push_back(d);
    }
    const GpuDecoded dec = gpu_decode(decodes);
    for (size_t i = 0; i < decodes.size(); ++i) {
      char label[80];
      std::snprintf(label, sizeof label, "corrupt bit %zu", i);
      check_decode_case(decodes[i], dec, i, label);
    }
  }

  // Random garbage buffers: the two implementations must agree on the result.
  {
    std::vector<DecodeTile> decodes;
    for (int trial = 0; trial < 64; ++trial) {
      const size_t length = 1u + rnd_below(rng, 64u);
      DecodeTile d;
      d.tape = base.tape;
      d.cdfs = base.cdfs;
      d.backing.resize(length);
      for (size_t i = 0; i < length; ++i) {
        d.backing[i] = static_cast<uint8_t>(rnd_below(rng, 256u));
      }
      d.sz = length;
      decodes.push_back(d);
    }
    const GpuDecoded dec = gpu_decode(decodes);
    for (size_t i = 0; i < decodes.size(); ++i) {
      char label[80];
      std::snprintf(label, sizeof label, "garbage %zu", i);
      check_decode_case(decodes[i], dec, i, label);
    }
  }
}

// ---------------------------------------------------------------------------
// Buffer exhaustion
// ---------------------------------------------------------------------------

void test_buffer_exhaustion() {
  std::printf("== buffer exhaustion ==\n");
  std::mt19937_64 rng(0xB0FFEEULL);
  TileCase base;
  base.cdfs.push_back(make_uniform_cdf(5, 40, 0));
  base.tape = random_tape(rng, 500u, base.cdfs, false);
  base.capacity = 4096;
  const GpuEncoded full = gpu_encode({base});
  expect_status(static_cast<Status>(full.results[0].status), Status::kOk,
                "exhaustion/base", "base encode");
  const size_t sz = full.results[0].payload_bytes;

  const size_t capacities[] = {0u, 1u, 2u, sz / 2u, sz - 1u, sz, sz + 1u};
  for (size_t capacity : capacities) {
    TileCase c = base;
    c.capacity = capacity;
    // Keep a valid backing allocation even at capacity 0 (tests use size 0).
    const GpuEncoded gpu = gpu_encode({c});
    char label[80];
    std::snprintf(label, sizeof label, "exhaustion cap=%zu", capacity);
    check_encode_case(c, gpu, 0, label);
    if (capacity < sz) {
      expect_status(static_cast<Status>(gpu.results[0].status),
                    Status::kBufferTooSmall, label, "buffer too small");
    } else {
      expect_status(static_cast<Status>(gpu.results[0].status), Status::kOk,
                    label, "capacity fits");
    }
  }
}

// ---------------------------------------------------------------------------
// Invalid inputs
// ---------------------------------------------------------------------------

void test_invalid_inputs() {
  std::printf("== invalid inputs ==\n");
  const std::vector<Cdf> bad_cdfs = {
      make_list_cdf({100, 200, 300, 125, 0}),   // rate too large
      make_list_cdf({100, 200, 300, 124, 33}),  // count too large
      make_list_cdf({200, 100, 300, 5, 0}),     // decreasing: not monotonic
      make_list_cdf({100, 200, 32768, 5, 0}),   // out of range
  };
  for (size_t i = 0; i < bad_cdfs.size(); ++i) {
    TileCase c;
    c.cdfs.push_back(bad_cdfs[i]);
    c.tape.push_back(av2enc::cuda::make_symbol_entry(
        0, static_cast<uint8_t>(bad_cdfs[i].n), 0));
    c.capacity = 64;
    const GpuEncoded gpu = gpu_encode({c});
    char label[80];
    std::snprintf(label, sizeof label, "invalid cdf %zu", i);
    check_encode_case(c, gpu, 0, label);
    expect_status(static_cast<Status>(gpu.results[0].status),
                  Status::kInvalidCdf, label, "invalid cdf status");
  }

  // Alphabet size 9 is outside Prob_Inc; cdf_entries=10 keeps the tape valid
  // so the coder, not the tape layer, rejects it.
  {
    TileCase c;
    Cdf cdf;
    cdf.n = 9;
    for (int i = 0; i < 8; ++i) {
      cdf.v[static_cast<size_t>(i)] = static_cast<uint16_t>(
          (32768u * static_cast<uint32_t>(i + 1)) / 9u);
    }
    cdf.v[8] = 56;
    cdf.v[9] = 0;
    c.cdfs.push_back(cdf);
    c.tape.push_back(av2enc::cuda::make_symbol_entry(0, 9, 0));
    c.capacity = 64;
    const GpuEncoded gpu = gpu_encode({c});
    check_encode_case(c, gpu, 0, "invalid n=9");
    expect_status(static_cast<Status>(gpu.results[0].status),
                  Status::kInvalidSymbol, "invalid n=9", "n=9 status");
  }

  // Symbol out of range.
  {
    TileCase c;
    c.cdfs.push_back(make_uniform_cdf(4, 56, 0));
    c.tape.push_back(av2enc::cuda::make_symbol_entry(0, 4, 4));
    c.capacity = 64;
    const GpuEncoded gpu = gpu_encode({c});
    check_encode_case(c, gpu, 0, "invalid symbol");
    expect_status(static_cast<Status>(gpu.results[0].status),
                  Status::kInvalidSymbol, "invalid symbol", "symbol status");
  }

  // cdf_id out of range, unknown op, literal width 33.
  {
    std::vector<TileCase> cases(3);
    cases[0].cdfs.push_back(make_uniform_cdf(3, 56, 0));
    cases[0].tape.push_back(av2enc::cuda::make_symbol_entry(5, 3, 0));
    cases[0].capacity = 64;

    cases[1].cdfs.push_back(make_uniform_cdf(3, 56, 0));
    EntropyTapeEntry unknown{};
    unknown.op = 7;
    cases[1].tape.push_back(unknown);
    cases[1].capacity = 64;

    cases[2].cdfs.push_back(make_uniform_cdf(3, 56, 0));
    cases[2].tape.push_back(av2enc::cuda::make_literal_entry(33, 0));
    cases[2].capacity = 64;

    const GpuEncoded gpu = gpu_encode(cases);
    for (size_t i = 0; i < cases.size(); ++i) {
      char label[80];
      std::snprintf(label, sizeof label, "invalid tape %zu", i);
      check_encode_case(cases[i], gpu, i, label);
      expect_status(static_cast<Status>(gpu.results[i].status),
                    Status::kInvalidArgument, label, "tape status");
    }
  }

  // Null tape with a non-zero entry count: the kernel reports invalid argument
  // instead of dereferencing it.
  {
    TileCase c;
    c.cdfs.push_back(make_uniform_cdf(2, 56, 0));
    c.tape.push_back(av2enc::cuda::make_symbol_entry(0, 2, 0));
    c.capacity = 64;
    EntropyEncodeDesc desc{};
    desc.tape = nullptr;
    desc.tape_entries = 1;
    desc.cdfs = nullptr;
    desc.cdf_count = 0;
    desc.cdf_entries = kCdfEntries;
    std::vector<uint8_t> out(64, 0);
    DeviceArray<uint8_t> d_out;
    d_out.alloc(out.size());
    d_out.upload(out);
    desc.out = d_out.get();
    desc.out_capacity = static_cast<uint32_t>(out.size());
    DeviceArray<EntropyEncodeDesc> d_desc;
    d_desc.alloc(1);
    std::vector<EntropyEncodeDesc> descs = {desc};
    d_desc.upload(descs);
    DeviceArray<EntropyTileResult> d_result;
    d_result.alloc(1);
    CUDA_CHECK(av2enc::cuda::launch_entropy_encode(d_desc.get(), d_result.get(),
                                                   1));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<EntropyTileResult> results;
    d_result.download(results);
    expect_status(static_cast<Status>(results[0].status),
                  Status::kInvalidArgument, "null tape", "null tape status");
  }
}

// ---------------------------------------------------------------------------
// CDF staging modes: a set larger than shared staging stays correct
// ---------------------------------------------------------------------------

void test_cdf_staging_modes() {
  std::printf("== CDF staging modes ==\n");
  std::mt19937_64 rng(0x5EEDCDFULL);

  // 250 CDFs * 10 entries = 2500 > kEntropyCdfStageEntries (2048): the kernel
  // must fall back to global-memory CDFs.
  {
    TileCase c;
    c.cdfs.reserve(250);
    for (int i = 0; i < 250; ++i) {
      c.cdfs.push_back(make_random_cdf(rng, 2 + (i % 7), 56, 0));
    }
    c.tape.reserve(2500);
    for (int i = 0; i < 2500; ++i) {
      const uint32_t cdf_id = rnd_below(rng, 250u);
      const uint8_t n = static_cast<uint8_t>(c.cdfs[cdf_id].n);
      c.tape.push_back(av2enc::cuda::make_symbol_entry(
          static_cast<uint16_t>(cdf_id), n,
          static_cast<uint16_t>(rnd_below(rng, n))));
    }
    c.capacity = 65536;
    const GpuEncoded gpu = gpu_encode({c});
    check_encode_case(c, gpu, 0, "cdf staging/global");

    TileCase direct = c;
    direct.flags |= av2enc::cuda::kEntropyDisableStaging;
    const GpuEncoded gpu_direct = gpu_encode({direct});
    check_encode_case(direct, gpu_direct, 0, "cdf staging/global direct");
  }

  // Exactly at the staging limit: 204 CDFs * 10 = 2040 entries.
  {
    TileCase c;
    c.cdfs.reserve(204);
    for (int i = 0; i < 204; ++i) {
      c.cdfs.push_back(make_uniform_cdf(2, 56, 0));
    }
    c.tape.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
      c.tape.push_back(av2enc::cuda::make_symbol_entry(
          static_cast<uint16_t>(rnd_below(rng, 204u)), 2,
          static_cast<uint16_t>(rnd_below(rng, 2u))));
    }
    c.capacity = 16384;
    const GpuEncoded gpu = gpu_encode({c});
    check_encode_case(c, gpu, 0, "cdf staging/limit");
  }
}

// ---------------------------------------------------------------------------
// Zero-probability CDF entries (equal adjacent entries), opt-in adaptive CDFs
// ---------------------------------------------------------------------------
//
// CPU contract since "entropy: opt-in adaptive CDFs with zero-width symbols":
//   * cdf_is_valid() accepts equal adjacent entries;
//   * compute_boundaries() allows a non-increasing boundary sequence;
//   * the §8.2.6 decoder do/while skips a repeated boundary;
//   * encode_symbol() rejects a requested zero-width range interval with
//     kInvalidSymbol and leaves the coder untouched (defensive: Prob_Inc keeps
//     a positive interval for the equal-entry table below, so it is encodable).
// The CUDA port reuses those helpers and must match byte-for-byte.

void test_zero_width_symbols() {
  std::printf("== zero-probability CDF entries (frozen + opt-in adaptive) ==\n");

  // Exact adapted table from the 640x360 CPU encode: cdf[1] == cdf[2], so
  // symbol 2 has zero probability in the model.  Prob_Inc still gives it a
  // positive range interval, so it stays encodable.
  const Cdf table = make_list_cdf({27718, 32641, 32641, 90, 32});
  expect_true(table.n == 4, "zero-width/table", "table n");
  expect_true(av2enc::av2::cdf_is_valid(table.data(), table.n),
              "zero-width/table", "cdf_is_valid accepts equal entries");
  av2enc::av2::Boundaries boundaries{};
  expect_status(av2enc::av2::compute_boundaries(
                    32768u, table.data(), table.n, boundaries),
                Status::kOk, "zero-width/table", "compute_boundaries");
  expect_true(boundaries.cur[0] >= boundaries.cur[1] &&
                  boundaries.cur[1] >= boundaries.cur[2] &&
                  boundaries.cur[2] >= boundaries.cur[3],
              "zero-width/table", "boundaries are non-increasing");

  // Same symbol stream as the CPU regression: every symbol, including the
  // zero-probability symbol 2.
  const int stream[] = {2, 1, 2, 3, 0, 2, 3, 3, 1, 2,
                        0, 1, 3, 2, 2, 0, 3, 1, 0, 2};
  for (int adaptive = 0; adaptive <= 1; ++adaptive) {
    TileCase c;
    c.cdfs.push_back(table);
    for (int symbol : stream) {
      c.tape.push_back(av2enc::cuda::make_symbol_entry(
          0, 4, static_cast<uint16_t>(symbol)));
    }
    c.capacity = 256;
    // Default is frozen; adaptation is an explicit opt-in.
    c.flags = (adaptive != 0) ? av2enc::cuda::kEntropyEnableCdfUpdate : 0u;
    const std::string label =
        std::string("zero-width/") + (adaptive != 0 ? "adaptive" : "frozen");
    const GpuEncoded gpu = gpu_encode({c});
    check_encode_case(c, gpu, 0, label);
    expect_status(static_cast<Status>(gpu.results[0].status), Status::kOk,
                  label, "encode status");

    DecodeTile d;
    d.tape = c.tape;
    d.cdfs = c.cdfs;
    d.backing = gpu.bytes[0];
    d.sz = gpu.results[0].payload_bytes;
    d.flags = c.flags;
    const GpuDecoded dec = gpu_decode({d});
    check_decode_case(d, dec, 0, label + "/decode");
    expect_status(static_cast<Status>(dec.results[0].status), Status::kOk,
                  label + "/decode", "decode status");
    for (size_t i = 0; i < dec.tapes[0].size(); ++i) {
      expect_u64(dec.tapes[0][i].symbol, static_cast<uint16_t>(stream[i]),
                 label + "/decode", "decoded symbol including symbol 2");
    }
    if (adaptive == 0) {
      // Frozen: the equal cdf[1]/cdf[2] entries stay untouched.
      expect_u64(dec.cdf_words[0][0], table.v[0], label, "frozen cdf[0]");
      expect_u64(dec.cdf_words[0][1], table.v[1], label, "frozen cdf[1]");
      expect_u64(dec.cdf_words[0][2], table.v[2], label, "frozen cdf[2]");
    }
  }

  // Adaptation equalises cdf[0] and cdf[1] after 500 symbol-0 codes: symbol 1
  // becomes zero-probability mid-stream.  This is the CPU regression
  // test_zero_width_cdf_entries() case, exercised through the GPU tape path.
  {
    const Cdf initial = make_list_cdf({100, 200, 300, 90, 0});
    Cdf probe = initial;
    for (int i = 0; i < 500; ++i) {
      av2enc::av2::adapt_cdf(probe.data(), 4, 0);
    }
    expect_u64(probe.v[0], probe.v[1], "zero-width/adaptation",
               "cdf[0] == cdf[1] after 500 updates");

    TileCase c;
    c.cdfs.push_back(initial);
    for (int i = 0; i < 500; ++i) {
      c.tape.push_back(av2enc::cuda::make_symbol_entry(0, 4, 0));
    }
    c.capacity = 2048;
    c.flags = av2enc::cuda::kEntropyEnableCdfUpdate;  // explicit opt-in
    const GpuEncoded gpu = gpu_encode({c});
    check_encode_case(c, gpu, 0, "zero-width/adaptation");
    expect_u64(gpu.cdf_words[0][0], probe.v[0], "zero-width/adaptation",
               "final cdf[0]");
    expect_u64(gpu.cdf_words[0][1], probe.v[1], "zero-width/adaptation",
               "final cdf[1]");

    DecodeTile d;
    d.tape = c.tape;
    d.cdfs = c.cdfs;
    d.backing = gpu.bytes[0];
    d.sz = gpu.results[0].payload_bytes;
    d.flags = c.flags;
    const GpuDecoded dec = gpu_decode({d});
    check_decode_case(d, dec, 0, "zero-width/adaptation/decode");
    expect_u64(dec.cdf_words[0][0], dec.cdf_words[0][1],
               "zero-width/adaptation/decode", "cdf[0] == cdf[1] at the end");
    expect_u64(dec.cdf_words[0][0], probe.v[0],
               "zero-width/adaptation/decode", "final cdf[0]");
  }
}

// ---------------------------------------------------------------------------
// Benchmark-stream regression: highly skewed 2048-symbol adaptive stream
// ---------------------------------------------------------------------------

void test_benchmark_stream_regression() {
  std::printf("== benchmark stream regression ==\n");
  std::mt19937_64 rng(0xDEC0DEULL);
  const double weights[] = {0.35, 0.25, 0.15, 0.10,
                            0.07, 0.04, 0.025, 0.015};
  std::discrete_distribution<int> dist(std::begin(weights),
                                       std::end(weights));
  TileCase c;
  c.cdfs.push_back(make_uniform_cdf(8, 56, 0));
  for (int i = 0; i < 2048; ++i) {
    c.tape.push_back(av2enc::cuda::make_symbol_entry(
        0, 8, static_cast<uint16_t>(dist(rng))));
  }
  c.capacity = (2048u * 3u) / 2u + 4096u;
  c.flags = av2enc::cuda::kEntropyEnableCdfUpdate;
  const GpuEncoded gpu = gpu_encode({c});
  check_encode_case(c, gpu, 0, "benchmark-stream/encode");

  DecodeTile d;
  d.tape = c.tape;
  d.cdfs = c.cdfs;
  d.backing = gpu.bytes[0];
  d.sz = gpu.results[0].payload_bytes;
  d.flags = c.flags;
  const GpuDecoded dec = gpu_decode({d});
  check_decode_case(d, dec, 0, "benchmark-stream/decode");

  // Same stream as a 16-tile 8x2-style batch.
  std::vector<DecodeTile> batch(16, d);
  const GpuDecoded dec16 = gpu_decode(batch);
  for (size_t i = 0; i < batch.size(); ++i) {
    check_decode_case(batch[i], dec16, i, "benchmark-stream/decode16");
  }
}

// ---------------------------------------------------------------------------
// Performance: per-tile latency, occupancy, serial tail
// ---------------------------------------------------------------------------

struct BenchResult {
  uint32_t tiles = 0;
  double launch_ms_min = 0.0;   // min over per-launch event samples
  double launch_ms_med = 0.0;
  double per_tile_cycles_min = 0.0;   // min over all tiles in the last launchs
  double per_tile_cycles_med = 0.0;
  double per_tile_cycles_max = 0.0;
  double payload_bytes = 0.0;
};

std::vector<EntropyTapeEntry> make_bench_tape(std::mt19937_64& rng,
                                              uint32_t symbols) {
  // Skewed distribution roughly like quantized coefficients.
  const double weights[] = {0.35, 0.25, 0.15, 0.10, 0.07, 0.04, 0.025, 0.015};
  std::discrete_distribution<int> dist(std::begin(weights), std::end(weights));
  std::vector<EntropyTapeEntry> tape;
  tape.reserve(symbols);
  for (uint32_t i = 0; i < symbols; ++i) {
    tape.push_back(av2enc::cuda::make_symbol_entry(
        0, 8, static_cast<uint16_t>(dist(rng))));
  }
  return tape;
}

// Per-launch samples, min/median.  A single averaged back-to-back loop is
// sensitive to the SM clock state and to other tenants on the box; the min of
// many individually timed launches is the stable "uncontended" latency.
BenchResult bench_encode(const char* name, uint32_t tiles, uint32_t symbols,
                         uint32_t flags, uint32_t reps, double clock_khz) {
  (void)clock_khz;
  std::mt19937_64 rng(0xFEEDFACEULL + tiles * 17u + flags);
  const std::vector<EntropyTapeEntry> tape = make_bench_tape(rng, symbols);
  std::vector<TileCase> cases;
  cases.reserve(tiles);
  for (uint32_t t = 0; t < tiles; ++t) {
    TileCase c;
    c.cdfs.push_back(make_uniform_cdf(8, 56, 0));
    c.tape = tape;
    c.capacity = (symbols * 3u) / 2u + 4096u;
    c.flags = flags;
    cases.push_back(c);
  }

  EncodePlan plan(cases);
  for (int warmup = 0; warmup < 20; ++warmup) {
    plan.reset_cdfs();
    CUDA_CHECK(plan.launch());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<double> samples;
  samples.reserve(reps);
  std::vector<EntropyTileResult> results;
  for (uint32_t i = 0; i < reps; ++i) {
    plan.reset_cdfs();
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(plan.launch());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    samples.push_back(static_cast<double>(elapsed_ms));
    plan.download_results(results);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::sort(samples.begin(), samples.end());

  std::vector<uint64_t> cycles;
  cycles.reserve(results.size());
  double payload = 0.0;
  for (const EntropyTileResult& r : results) {
    if (r.status == static_cast<uint32_t>(Status::kOk)) {
      cycles.push_back(r.cycles);
      payload += static_cast<double>(r.payload_bytes);
    }
  }
  std::sort(cycles.begin(), cycles.end());
  BenchResult out;
  out.tiles = tiles;
  out.launch_ms_min = samples.front();
  out.launch_ms_med = samples[samples.size() / 2];
  out.per_tile_cycles_min =
      cycles.empty() ? 0.0 : static_cast<double>(cycles.front());
  out.per_tile_cycles_med =
      cycles.empty() ? 0.0
                     : static_cast<double>(cycles[cycles.size() / 2]);
  out.per_tile_cycles_max =
      cycles.empty() ? 0.0 : static_cast<double>(cycles.back());
  out.payload_bytes =
      cycles.empty() ? 0.0 : payload / static_cast<double>(cycles.size());
  (void)name;
  return out;
}

void benchmark() {
  std::printf("== benchmarks ==\n");
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int clock_khz_i = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz_i, cudaDevAttrClockRate, device));
  const double clock_khz = static_cast<double>(clock_khz_i);  // kHz
  const int sm_count = prop.multiProcessorCount;

  int enc_blocks = 0;
  int dec_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &enc_blocks, av2enc::cuda::entropy_encode_kernel,
      static_cast<int>(av2enc::cuda::kEntropyBlockThreads), 0));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &dec_blocks, av2enc::cuda::entropy_decode_kernel,
      static_cast<int>(av2enc::cuda::kEntropyBlockThreads), 0));
  const int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / 32;

  std::printf("GPU: %s, %d SMs, clock %.3f GHz (max boost), shared/SM %zu B\n",
              prop.name, sm_count, clock_khz / 1.0e6,
              static_cast<size_t>(prop.sharedMemPerMultiprocessor));
  std::printf(
      "occupancy: encode %d block/SM (%d active warps, %.1f%% of %d warps), "
      "decode %d block/SM (%.1f%%)\n",
      enc_blocks, enc_blocks,
      100.0 * static_cast<double>(enc_blocks) /
          static_cast<double>(max_warps_per_sm),
      max_warps_per_sm, dec_blocks,
      100.0 * static_cast<double>(dec_blocks) /
          static_cast<double>(max_warps_per_sm));

  const uint32_t symbols = 2048;
  const uint32_t layouts[] = {1, 4, 8, 16, 32, 64, 128, 256};
  const uint32_t reps[] = {50, 50, 50, 40, 30, 20, 10, 6};
  // Worst case: the opt-in adaptive path.  The frozen default does less work.
  const uint32_t bench_flags = av2enc::cuda::kEntropyEnableCdfUpdate;
  double single_tile_ms = 0.0;

  std::printf(
      "%-6s %8s %12s %14s %12s %12s %10s\n", "tiles", "bytes",
      "launch_ms", "tile_cyc_med", "tile_cyc_min", "tile_cyc_max", "tiles/s");
  for (size_t i = 0; i < std::size(layouts); ++i) {
    const BenchResult r = bench_encode("encode", layouts[i], symbols,
                                       bench_flags, reps[i], clock_khz);
    if (layouts[i] == 1) {
      single_tile_ms = r.launch_ms_min;
    }
    std::printf("%-6u %8.0f %12.4f %14.0f %12.0f %12.0f %10.0f\n", r.tiles,
                r.payload_bytes, r.launch_ms_min, r.per_tile_cycles_med,
                r.per_tile_cycles_min, r.per_tile_cycles_max,
                static_cast<double>(r.tiles) / r.launch_ms_min * 1000.0);
  }

  const BenchResult staged = bench_encode("encode", 1, symbols, bench_flags,
                                          30, clock_khz);
  const BenchResult direct = bench_encode(
      "encode-direct", 1, symbols,
      bench_flags | av2enc::cuda::kEntropyDisableStaging, 30, clock_khz);
  const BenchResult staged16 = bench_encode("encode", 16, symbols,
                                            bench_flags, 30, clock_khz);
  const BenchResult direct16 = bench_encode(
      "encode-direct", 16, symbols,
      bench_flags | av2enc::cuda::kEntropyDisableStaging, 30, clock_khz);
  const BenchResult frozen16 = bench_encode("encode-frozen", 16, symbols, 0,
                                            30, clock_khz);

  std::printf(
      "staging: 1 tile %.0f -> %.0f cycles (%.2fx), 16 tiles %.0f -> %.0f "
      "(%.2fx)\n",
      direct.per_tile_cycles_med, staged.per_tile_cycles_med,
      direct.per_tile_cycles_med /
          (staged.per_tile_cycles_med > 0.0
               ? staged.per_tile_cycles_med
               : 1.0),
      direct16.per_tile_cycles_med, staged16.per_tile_cycles_med,
      direct16.per_tile_cycles_med /
          (staged16.per_tile_cycles_med > 0.0
               ? staged16.per_tile_cycles_med
               : 1.0));
  std::printf("cdf mode: 16 tiles adaptive %.0f cycles, frozen %.0f cycles "
              "(%.2fx)\n",
              staged16.per_tile_cycles_med,
              frozen16.per_tile_cycles_med,
              staged16.per_tile_cycles_med /
                  (frozen16.per_tile_cycles_med > 0.0
                       ? frozen16.per_tile_cycles_med
                       : 1.0));

  // CPU reference on the same 2048-symbol tape.  This is the serial-tail
  // evidence: one GPU tile is far slower than one CPU thread on this box.
  {
    std::mt19937_64 rng(0xFEEDFACEULL + 16u * 17u + bench_flags);
    const std::vector<EntropyTapeEntry> tape = make_bench_tape(rng, symbols);
    const Cdf cdf = make_uniform_cdf(8, 56, 0);
    double best_us = 0.0;
    for (int trial = 0; trial < 5; ++trial) {
      std::vector<uint8_t> buffer((symbols * 3u) / 2u + 4096u, 0xCD);
      std::vector<Cdf> cdfs{cdf};
      av2enc::av2::Encoder encoder;
      encoder.init_symbol(buffer.data(), buffer.size());
      const auto begin = std::chrono::steady_clock::now();
      for (const EntropyTapeEntry& entry : tape) {
        encoder.encode_symbol(cdfs[0].data(), 8,
                              static_cast<int>(entry.symbol));
      }
      encoder.exit_symbol(nullptr);
      const auto end = std::chrono::steady_clock::now();
      const double us =
          std::chrono::duration<double, std::micro>(end - begin).count();
      if (trial == 0 || us < best_us) {
        best_us = us;
      }
    }
    std::printf(
        "cpu reference: %.1f us/tile (%.1f ns/symbol), %.2fx faster than one "
        "GPU tile\n",
        best_us, best_us * 1000.0 / static_cast<double>(symbols),
        (single_tile_ms * 1000.0) / best_us);
  }

  // Decode timing for one tile and for the 8x2 layout.
  {
    std::mt19937_64 rng(0xDEC0DEULL);
    const std::vector<EntropyTapeEntry> tape = make_bench_tape(rng, symbols);
    TileCase c;
    c.cdfs.push_back(make_uniform_cdf(8, 56, 0));
    c.tape = tape;
    c.capacity = (symbols * 3u) / 2u + 4096u;
    c.flags = bench_flags;
    const GpuEncoded encoded = gpu_encode({c});
    std::vector<DecodeTile> decodes;
    for (int i = 0; i < 16; ++i) {
      DecodeTile d;
      d.tape = tape;
      d.cdfs = c.cdfs;
      d.backing = encoded.bytes[0];
      d.sz = encoded.results[0].payload_bytes;
      d.flags = bench_flags;
      decodes.push_back(d);
    }
    DecodePlan plan(decodes);
    for (int warmup = 0; warmup < 20; ++warmup) {
      plan.reset_cdfs();
      CUDA_CHECK(plan.launch());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> samples;
    for (int i = 0; i < 30; ++i) {
      plan.reset_cdfs();
      CUDA_CHECK(cudaEventRecord(start));
      CUDA_CHECK(plan.launch());
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
      samples.push_back(static_cast<double>(elapsed_ms));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    std::vector<EntropyTileResult> results;
    plan.download_results(results);
    std::vector<uint64_t> cycles;
    for (const EntropyTileResult& r : results) {
      if (r.status == static_cast<uint32_t>(Status::kOk)) {
        cycles.push_back(r.cycles);
      }
    }
    std::sort(cycles.begin(), cycles.end());
    std::printf(
        "decode: 16 tiles %.4f ms/launch (min), payload %.0f bytes, "
        "tile cycles med %.0f\n",
        samples.front(),
        static_cast<double>(encoded.results[0].payload_bytes),
        cycles.empty() ? 0.0
                       : static_cast<double>(cycles[cycles.size() / 2]));
  }

  std::printf(
      "serial tail: one tile is %.2f us of serial lane work; with %d SMs an "
      "8x2 frame (16 tiles) is one wave, so the entropy stage cannot be "
      "shorter than ~%.2f us regardless of SM count.\n",
      single_tile_ms * 1000.0, sm_count, single_tile_ms * 1000.0);
  std::printf(
      "             beyond %d tiles, extra tiles queue in later waves; "
      "batch frames to keep every SM fed.\n",
      sm_count);
}

}  // namespace

int main(int argc, char** argv) {
  bool run_bench = true;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--no-bench") == 0) {
      run_bench = false;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return EXIT_FAILURE;
    }
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("CUDA AV2 §8.2 entropy port tests\n");
  std::printf("device: %s, cc %d.%d, %d SMs, CUDA %d\n", prop.name,
              prop.major, prop.minor, prop.multiProcessorCount,
              CUDART_VERSION);

  test_golden_vectors();
  test_random_symbols();
  test_interleaved_and_staging();
  test_empty_and_single();
  test_multi_tile(4, 2, false);
  test_multi_tile(8, 2, false);
  test_multi_tile(4, 2, true);
  test_truncation_and_corruption();
  test_buffer_exhaustion();
  test_invalid_inputs();
  test_cdf_staging_modes();
  test_zero_width_symbols();
  test_benchmark_stream_regression();
  if (run_bench) {
    benchmark();
  }

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  if (g_failures == 0) {
    std::printf("PASS\n");
    return EXIT_SUCCESS;
  }
  std::printf("FAIL\n");
  return EXIT_FAILURE;
}
