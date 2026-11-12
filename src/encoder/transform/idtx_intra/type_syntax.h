// §5.20.8.2 transform_type() writer/reader for the intra type symbols.
//
// Only the transform-type part of transform_type() is modelled: the
// `is_long_side_dct` flag (WIDE_32/HIGH_32), the `intra_tx_type` symbol and
// the implied flag for WIDE_64/HIGH_64.  The caller decides the type with
// `idtx_intra.h`; this pair turns that decision into the exact §8.2 symbol
// bits and back.
//
// The CDFs are the §9.3 defaults (`kIntraTxTypeSet1Cdf`, `kIntraTxTypeSet2Cdf`,
// `kIntraTxTypeLongCdf`, `kIsLongSideDctCdf`).  The coder is the normative
// §8.2 range coder in `src/entropy/av2`, so the reader runs the same maths a
// decoder runs.  CDF adaptation is on by default (`disable_cdf_update == 0`);
// pass `adapt = false` to model `disable_cdf_update == 1`.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "encoder/transform/idtx_intra/idtx_intra.h"
#include "entropy/av2/av2_entropy.h"

namespace av2enc::idtx {

// What a reader recovered for one block.
struct TypeSymbols {
  Set set = Set::kDctOnly;
  bool codes_flag = false;
  bool flag = false;
  int symbol = -1;  // -1 when the set codes no intra_tx_type
  TxType type = TxType::kDctDct;
};

class TypeSyntaxWriter {
 public:
  // `tx_size_index` is the AV2 TX_SIZE index (0..24).  `flag` is the
  // is_long_side_dct value; it is coded only when the set codes it.
  bool write(Set set, int tx_size_index, TxType type, int symbol, bool flag,
             bool adapt = true, std::string* error = nullptr);
  bool finish(std::vector<std::uint8_t>* bytes, std::string* error = nullptr);
  void reset();

 private:
  void ensure_cdfs();
  bool failed_ = false;
  bool finished_ = false;
  bool cdf_ready_ = false;
  av2enc::av2::Encoder encoder_;
  std::vector<std::uint8_t> buffer_;
  std::uint16_t set1_[3][8] = {};
  std::uint16_t set2_[3][3] = {};
  std::uint16_t long_[4][5] = {};
  std::uint16_t flag_[2][3] = {};
};

class TypeSyntaxReader {
 public:
  // `payload_bytes` is the §8.2 payload size (sz).  `bytes` may be larger; the
  // reader never looks past `payload_bytes`.
  bool init(std::vector<std::uint8_t> bytes, std::size_t payload_bytes,
            std::string* error = nullptr);
  // `types` is the planner's legal set for this block; the symbol is mapped
  // back to a TX_TYPE through it (INTRA_1/INTRA_2 depend on size class and
  // mode, so the symbol alone is not enough to name the type).
  bool read(const TypeSet& types, int tx_size_index, bool adapt,
            TypeSymbols* out, std::string* error = nullptr);
  bool finish(std::string* error = nullptr);
  void reset();

 private:
  void ensure_cdfs();
  bool failed_ = false;
  bool started_ = false;
  bool cdf_ready_ = false;
  av2enc::av2::Decoder decoder_;
  std::vector<std::uint8_t> storage_;
  std::uint16_t set1_[3][8] = {};
  std::uint16_t set2_[3][3] = {};
  std::uint16_t long_[4][5] = {};
  std::uint16_t flag_[2][3] = {};
};

// The §9.3 default CDF for a set and TX size: 7 symbols for INTRA_1, 2 for
// INTRA_2, 4 for the long sets.  Returns nullptr for a no-syntax set.
int cdf_symbol_count(Set set);
const std::uint16_t* default_type_cdf(Set set, int tx_size_index);

}  // namespace av2enc::idtx
