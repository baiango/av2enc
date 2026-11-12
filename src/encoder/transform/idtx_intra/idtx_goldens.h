// Pinned goldens for the intra identity-transform coverage.
//
// Every value is a deterministic hash of bytes produced by this module and is
// stable across machines (FNV-1a 64, no floating point in the hashed data).
//
//   kSeqBitsAllOffBits / kSeqBitsAllOff
//       §5.4.8 block for the M1-shaped all-off config (fsc=0, idtx=0,
//       chroma-dct-only=1, equal_ac_dc_q=1, base_uv_ac=23).
//   kLegalityHash
//       FNV over the exhaustive type legality enumeration (25 sizes x 3 planes
//       x 4 reduced sets x 2 idtx x 2 chroma-dct-only x 13 modes x 14 UV modes
//       x 5 EOBs), hashing each candidate symbol in order.
//   kIdentityParityHash
//       FNV over the reconstructed residuals of all 7 identity types at all 25
//       sizes from the independent §7.15.4.1 reference.
//   kTypeSyntaxBytes / kTypeSyntaxHash
//       §5.20.8.2 symbol stream for the fixed 67-decision sequence.
//   kFeatureStreamFnv / kAllOffStreamFnv / kIdentityStreamFnv
//       Encoded temporal units from `idtx_intra_gate`: the non-DCT feature
//       stream, the all-off control and the identity-type stream.
#pragma once

#include <cstddef>
#include <cstdint>

namespace {

constexpr std::size_t kSeqBitsAllOffBits = 17;
constexpr std::uint8_t kSeqBitsAllOff[3] = {0x08, 0x37, 0x00};

constexpr std::uint64_t kLegalityHash = 0x11776b9d9a04f1efULL;
constexpr std::uint64_t kIdentityParityHash = 0x1628402f894dec3dULL;

constexpr std::size_t kTypeSyntaxBytes = 32;
constexpr std::uint64_t kTypeSyntaxHash = 0x934b150af67ff167ULL;

constexpr std::uint64_t kFeatureStreamFnv = 0x5ba50465798c1951ULL;
constexpr std::uint64_t kAllOffStreamFnv = 0x4f377f66c3651e11ULL;
constexpr std::uint64_t kIdentityStreamFnv = 0x08a999e7cc62c347ULL;

}  // namespace
