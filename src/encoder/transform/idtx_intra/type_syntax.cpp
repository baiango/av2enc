// See type_syntax.h.
#include "encoder/transform/idtx_intra/type_syntax.h"

#include "encoder/tx_type/tx_type_tables.h"

#include <cstring>
#include <utility>

namespace av2enc::idtx {
namespace {

using txtype::kIntraTxTypeLongCdf;
using txtype::kIntraTxTypeSet1Cdf;
using txtype::kIntraTxTypeSet2Cdf;
using txtype::kIsLongSideDctCdf;

bool set_sqr(Set set, int tx_size_index, int* sqr) {
  if (tx_size_index < 0 || tx_size_index >= 25) return false;
  switch (set) {
    case Set::kIntra1:
    case Set::kIntra2:
    case Set::kWide64:
    case Set::kHigh64:
    case Set::kWide32:
    case Set::kHigh32:
      *sqr = txtype::kTxSizeSqr[tx_size_index];
      return true;
    case Set::kDctOnly:
    case Set::kDctIdtx:
      return false;
  }
  return false;
}

bool fail(std::string* error, const char* message) {
  if (error != nullptr) *error = message;
  return false;
}

template <std::size_t R, std::size_t C>
void copy_table(std::uint16_t (&dst)[R][C],
                const std::uint16_t (&src)[R][C]) {
  std::memcpy(dst, src, sizeof(dst));
}

}  // namespace

int cdf_symbol_count(Set set) {
  switch (set) {
    case Set::kIntra1:
      return 7;
    case Set::kIntra2:
      return 2;
    case Set::kWide64:
    case Set::kHigh64:
    case Set::kWide32:
    case Set::kHigh32:
      return 4;
    case Set::kDctOnly:
    case Set::kDctIdtx:
      return 0;
  }
  return 0;
}

const std::uint16_t* default_type_cdf(Set set, int tx_size_index) {
  int sqr = 0;
  if (!set_sqr(set, tx_size_index, &sqr)) return nullptr;
  switch (set) {
    case Set::kIntra1:
      return kIntraTxTypeSet1Cdf[sqr];
    case Set::kIntra2:
      return kIntraTxTypeSet2Cdf[sqr];
    case Set::kWide64:
    case Set::kHigh64:
    case Set::kWide32:
    case Set::kHigh32:
      return kIntraTxTypeLongCdf[sqr];
    default:
      return nullptr;
  }
}

void TypeSyntaxWriter::ensure_cdfs() {
  if (cdf_ready_) return;
  copy_table(set1_, kIntraTxTypeSet1Cdf);
  copy_table(set2_, kIntraTxTypeSet2Cdf);
  copy_table(long_, kIntraTxTypeLongCdf);
  copy_table(flag_, kIsLongSideDctCdf);
  cdf_ready_ = true;
}

void TypeSyntaxWriter::reset() {
  failed_ = false;
  finished_ = false;
  cdf_ready_ = false;
  encoder_ = av2enc::av2::Encoder{};
  buffer_.clear();
}

bool TypeSyntaxWriter::write(Set set, int tx_size_index, TxType type, int symbol,
                             bool flag, bool adapt, std::string* error) {
  if (failed_) return fail(error, "writer is already poisoned");
  if (finished_) return fail(error, "writer already finished");
  const int nsym = cdf_symbol_count(set);
  if (nsym == 0) return true;  // no intra_tx_type syntax for this set
  int sqr = 0;
  if (!set_sqr(set, tx_size_index, &sqr)) {
    return fail(error, "invalid TX size index for this set");
  }
  if (symbol < 0 || symbol >= nsym) return fail(error, "symbol out of range");
  const int type_value = static_cast<int>(type);
  if (type_value < 0 || type_value >= 16) {
    return fail(error, "TX type out of range");
  }

  ensure_cdfs();
  if (encoder_.out == nullptr) {
    buffer_.assign(1u << 14, 0);
    if (encoder_.init_symbol(buffer_.data(), buffer_.size()) !=
        av2enc::av2::Status::kOk) {
      failed_ = true;
      return fail(error, "range coder init failed");
    }
  }
  encoder_.disable_cdf_update = adapt ? 0 : 1;

  const bool codes_flag =
      set == Set::kWide32 || set == Set::kHigh32;
  if (codes_flag) {
    if (encoder_.encode_symbol(flag_[0], 2, flag ? 1 : 0) !=
        av2enc::av2::Status::kOk) {
      failed_ = true;
      return fail(error, "is_long_side_dct coding failed");
    }
  } else if ((set == Set::kWide64 || set == Set::kHigh64) && !flag) {
    return fail(error, "long-64 sets imply is_long_side_dct = 1");
  }

  std::uint16_t* cdf = nullptr;
  switch (set) {
    case Set::kIntra1:
      cdf = set1_[sqr];
      break;
    case Set::kIntra2:
      cdf = set2_[sqr];
      break;
    default:
      cdf = long_[sqr];
      break;
  }
  if (encoder_.encode_symbol(cdf, nsym, symbol) != av2enc::av2::Status::kOk) {
    failed_ = true;
    return fail(error, "intra_tx_type coding failed");
  }
  return true;
}

bool TypeSyntaxWriter::finish(std::vector<std::uint8_t>* bytes,
                              std::string* error) {
  if (failed_) return fail(error, "writer is already poisoned");
  if (!finished_) {
    if (encoder_.out == nullptr) {
      // No symbol was written: emit an empty payload.
      if (bytes != nullptr) bytes->clear();
      finished_ = true;
      return true;
    }
    std::size_t payload = 0;
    if (encoder_.exit_symbol(&payload) != av2enc::av2::Status::kOk) {
      failed_ = true;
      return fail(error, "range coder finish failed");
    }
    buffer_.resize(payload);
    finished_ = true;
  }
  if (bytes != nullptr) *bytes = buffer_;
  return true;
}

void TypeSyntaxReader::ensure_cdfs() {
  if (cdf_ready_) return;
  copy_table(set1_, kIntraTxTypeSet1Cdf);
  copy_table(set2_, kIntraTxTypeSet2Cdf);
  copy_table(long_, kIntraTxTypeLongCdf);
  copy_table(flag_, kIsLongSideDctCdf);
  cdf_ready_ = true;
}

void TypeSyntaxReader::reset() {
  failed_ = false;
  started_ = false;
  cdf_ready_ = false;
  decoder_ = av2enc::av2::Decoder{};
  storage_.clear();
}

bool TypeSyntaxReader::init(std::vector<std::uint8_t> bytes,
                            std::size_t payload_bytes, std::string* error) {
  reset();
  if (payload_bytes == 0 || payload_bytes > bytes.size()) {
    return fail(error, "payload size is out of range");
  }
  storage_ = std::move(bytes);
  if (decoder_.init_symbol(storage_.data(), storage_.size(), payload_bytes) !=
      av2enc::av2::Status::kOk) {
    return fail(error, "range decoder init failed");
  }
  started_ = true;
  return true;
}

bool TypeSyntaxReader::read(const TypeSet& types, int tx_size_index, bool adapt,
                            TypeSymbols* out, std::string* error) {
  const Set set = types.set;
  if (out == nullptr) return fail(error, "null output");
  if (!started_) return fail(error, "reader is not initialized");
  if (failed_) return fail(error, "reader is already poisoned");
  out->set = set;
  out->codes_flag = false;
  out->flag = false;
  out->symbol = -1;
  out->type = TxType::kDctDct;

  const int nsym = cdf_symbol_count(set);
  if (nsym == 0) {
    // No intra_tx_type syntax: the effective type is the planner's only entry.
    if (types.count > 0) out->type = types.candidates[0].type;
    return true;
  }
  int sqr = 0;
  if (!set_sqr(set, tx_size_index, &sqr)) {
    return fail(error, "invalid TX size index for this set");
  }
  ensure_cdfs();
  decoder_.disable_cdf_update = adapt ? 0 : 1;

  const bool codes_flag = set == Set::kWide32 || set == Set::kHigh32;
  if (codes_flag) {
    int bit = 0;
    if (decoder_.decode_symbol(flag_[0], 2, &bit) !=
        av2enc::av2::Status::kOk) {
      failed_ = true;
      return fail(error, "is_long_side_dct decode failed");
    }
    out->codes_flag = true;
    out->flag = bit != 0;
  } else if (set == Set::kWide64 || set == Set::kHigh64) {
    out->flag = true;  // implied
  }

  std::uint16_t* cdf = nullptr;
  switch (set) {
    case Set::kIntra1:
      cdf = set1_[sqr];
      break;
    case Set::kIntra2:
      cdf = set2_[sqr];
      break;
    default:
      cdf = long_[sqr];
      break;
  }
  int symbol = 0;
  if (decoder_.decode_symbol(cdf, nsym, &symbol) != av2enc::av2::Status::kOk) {
    failed_ = true;
    return fail(error, "intra_tx_type decode failed");
  }
  out->symbol = symbol;

  // Map the decoded symbol (and flag) back through the legal set so INTRA_1 /
  // INTRA_2 use Md_Idx_To_Type exactly like the decoder.
  for (int i = 0; i < types.count; ++i) {
    const Candidate& c = types.candidates[i];
    if (c.symbol != symbol) continue;
    if (codes_flag && c.is_long_side_dct != out->flag) continue;
    out->type = c.type;
    return true;
  }
  failed_ = true;
  return fail(error, "decoded symbol is not in the queried set");
}

bool TypeSyntaxReader::finish(std::string* error) {
  if (failed_) return fail(error, "reader is already poisoned");
  if (!started_) return fail(error, "reader is not initialized");
  if (decoder_.exit_symbol() != av2enc::av2::Status::kOk) {
    failed_ = true;
    return fail(error, "range decoder exit failed");
  }
  return true;
}

}  // namespace av2enc::idtx
