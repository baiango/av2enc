// Default CDEF index CDFs — AV2 v1.0.0 §9.3.
//
// Values copied verbatim from spec/tables/default_cdf.json
// (`Default_Cdef_Index0_Cdf`, `Default_Cdef_Index_Minus1_With{3..8}_Cdf`).
// Generate with src/encoder/m1/generate_still_tables.py; do not hand-edit.
//
// Shape notes:
//   * index0 has CDEF_STRENGTH_INDEX0_CTX = 4 contexts, each with 2 symbols,
//     stored as 3 inverse-CDF entries (cdf[n] = 0).
//   * `minus1_withN` has N entries, i.e. N-1 symbols: index_minus_1 in
//     [0, CdefStrengths-2].
#ifndef AV2ENC_CUDA_STILL_CDEF_CDF_H_
#define AV2ENC_CUDA_STILL_CDEF_CDF_H_

#include <cstdint>

namespace av2enc::cuda::still {

inline constexpr std::uint16_t kCdefIndex0Cdf[4][3] = {
    {29034, 32, 0},
    {16472, 32, 0},
    {5751, 32, 0},
    {3115, 31, 0},
};

inline constexpr std::uint16_t kCdefIndexMinus1With3Cdf[3] = {17708, 32, 0};
inline constexpr std::uint16_t kCdefIndexMinus1With4Cdf[4] = {13413, 24899, 32,
                                                              0};
inline constexpr std::uint16_t kCdefIndexMinus1With5Cdf[5] = {10588, 19866,
                                                              26664, 37, 0};
inline constexpr std::uint16_t kCdefIndexMinus1With6Cdf[6] = {
    10131, 17874, 23876, 28766, 37, 0};
inline constexpr std::uint16_t kCdefIndexMinus1With7Cdf[7] = {
    8363, 15451, 20811, 25453, 29393, 32, 0};
inline constexpr std::uint16_t kCdefIndexMinus1With8Cdf[8] = {
    7372, 13867, 18969, 23278, 26977, 30156, 32, 0};

// Returns the default index_minus_1 CDF for a CdefStrengths value, or null
// outside 3..8 (2 strengths uses no minus-1 symbol at all).
inline const std::uint16_t* cdef_index_minus1_default(int strengths) {
  switch (strengths) {
    case 3:
      return kCdefIndexMinus1With3Cdf;
    case 4:
      return kCdefIndexMinus1With4Cdf;
    case 5:
      return kCdefIndexMinus1With5Cdf;
    case 6:
      return kCdefIndexMinus1With6Cdf;
    case 7:
      return kCdefIndexMinus1With7Cdf;
    case 8:
      return kCdefIndexMinus1With8Cdf;
    default:
      return nullptr;
  }
}

}  // namespace av2enc::cuda::still

#endif  // AV2ENC_CUDA_STILL_CDEF_CDF_H_
