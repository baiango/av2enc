// av2::Encoder / av2::Decoder adapters for the transform-partition syntax.
//
// Kept out of tx_partition.h so the pure syntax/geometry component does not
// depend on the entropy module.  These adapters are the ones used by the
// component test goldens and by the dav2d gate.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "encoder/m2/tx_partition/tx_partition.h"
#include "entropy/av2/av2_entropy.h"

namespace av2enc::m2::txpartition {

class Av2SymbolSink final : public SymbolSink {
 public:
  explicit Av2SymbolSink(av2::Encoder* encoder) : enc_(encoder) {}

  bool symbol(std::uint16_t* cdf, int n, int value,
              std::string& error) override {
    if (enc_ == nullptr) {
      error = "null entropy encoder";
      return false;
    }
    const av2::Status status = enc_->encode_symbol(cdf, n, value);
    if (status != av2::Status::kOk) {
      error = std::string("encode_symbol failed: ") +
              av2::status_name(status) + " n=" + std::to_string(n) +
              " symbol=" + std::to_string(value);
      return false;
    }
    return true;
  }

 private:
  av2::Encoder* enc_ = nullptr;
};

class Av2SymbolSource final : public SymbolSource {
 public:
  explicit Av2SymbolSource(av2::Decoder* decoder) : dec_(decoder) {}

  bool symbol(std::uint16_t* cdf, int n, int* value,
              std::string& error) override {
    if (dec_ == nullptr) {
      error = "null entropy decoder";
      return false;
    }
    const av2::Status status = dec_->decode_symbol(cdf, n, value);
    if (status != av2::Status::kOk) {
      error = std::string("decode_symbol failed: ") +
              av2::status_name(status);
      return false;
    }
    return true;
  }

 private:
  av2::Decoder* dec_ = nullptr;
};

// Convenience: encode `count` symbols into a fresh tile and return the
// exit_symbol() byte count.  Used by the bit-cost report and the goldens.
class TileBitWriter {
 public:
  explicit TileBitWriter(std::size_t capacity = 4096)
      : buffer_(capacity, 0) {
    (void)enc_.init_symbol(buffer_.data(), buffer_.size());
  }

  SymbolSink& sink() { return sink_; }
  av2::Encoder& encoder() { return enc_; }
  void disable_cdf_update(bool disable) {
    enc_.disable_cdf_update = disable ? 1 : 0;
  }

  bool finish(std::span<const std::uint8_t>* bytes, std::string& error) {
    std::size_t payload = 0;
    const av2::Status status = enc_.exit_symbol(&payload);
    if (status != av2::Status::kOk) {
      error = std::string("exit_symbol failed: ") + av2::status_name(status);
      return false;
    }
    out_.assign(buffer_.begin(),
                buffer_.begin() + static_cast<std::ptrdiff_t>(payload));
    *bytes = std::span<const std::uint8_t>(out_);
    return true;
  }

 private:
  std::vector<std::uint8_t> buffer_;
  std::vector<std::uint8_t> out_;
  av2::Encoder enc_{};
  Av2SymbolSink sink_{&enc_};
};

class TileBitReader {
 public:
  bool init(std::span<const std::uint8_t> bytes, std::string& error) {
    const av2::Status status =
        dec_.init_symbol(bytes.data(), bytes.size(), bytes.size());
    if (status != av2::Status::kOk) {
      error = std::string("init_symbol failed: ") + av2::status_name(status);
      return false;
    }
    return true;
  }

  SymbolSource& source() { return source_; }
  av2::Decoder& decoder() { return dec_; }

  bool finish(std::string& error) {
    const av2::Status status = dec_.exit_symbol();
    if (status != av2::Status::kOk) {
      error = std::string("exit_symbol failed: ") + av2::status_name(status);
      return false;
    }
    return true;
  }

 private:
  av2::Decoder dec_{};
  Av2SymbolSource source_{&dec_};
};

}  // namespace av2enc::m2::txpartition
