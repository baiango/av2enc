// AV2 DDT 1D kernels (Ddtx_Kernel8 / Ddtx_Kernel16, §9.6).
//
// Provenance: AV2 Specification v1.0.0, §9.6 "1d transform tables", PDF
// pp. 848-849.  The PDF renders the numeric tables that the HTML spec leaves
// in attachment headers; both were cross-checked against
//   https://av2.aomedia.org/v1.0.0/attachments/ddtx_kernel8.h
//   https://av2.aomedia.org/v1.0.0/attachments/ddtx_kernel16.h
// and are identical to the values below.
//
// Matrix convention (§7.15.2.1): Ddtx_KernelN[j][i] multiplies input
// coefficient src[j] to produce output sample i.  Each row is one basis
// vector; row norms are 64*sqrt(N) like the DCT/ADST/FDST kernels.
#pragma once

#include <cstdint>

namespace av2enc::m2::ddt {

inline constexpr std::int16_t kDdtxKernel8[8][8] = {
    {   4,    7,   15,   33,   65,   98,  106,   80},
    {   6,   14,   36,   77,  100,   45,  -57,  -98},
    {  22,   48,   85,   88,    0,  -86,  -23,   82},
    {  57,   94,   76,  -26,  -73,   34,   54,  -66},
    {  96,   73,  -43,  -69,   55,   20,  -71,   53},
    { 103,  -17,  -80,   56,   15,  -66,   75,  -41},
    {  78,  -79,    7,   56,  -82,   79,  -56,   26},
    {  56,  -96,   98,  -77,   54,  -33,   19,   -6},
};

inline constexpr std::int16_t kDdtxKernel16[16][16] = {
    {  12,   15,   19,   23,   30,   39,   51,   66,   78,   88,   94,   97,   93,   83,   68,   50},
    {  17,   23,   30,   38,   48,   61,   76,   87,   83,   59,   19,  -30,  -73,  -99,  -99,  -76},
    {  37,   49,   60,   69,   75,   75,   61,   29,  -18,  -67,  -96,  -83,  -28,   40,   84,   83},
    {  45,   60,   69,   73,   66,   40,   -8,  -65,  -91,  -57,   21,   86,   81,    8,  -69,  -90},
    {  47,   60,   61,   49,   19,  -29,  -77,  -83,  -16,   75,   93,    3,  -92,  -74,   32,   97},
    {  60,   74,   64,   28,  -31,  -87,  -82,    4,   88,   54,  -55,  -77,   29,   88,    3,  -86},
    {  64,   70,   40,  -19,  -79,  -78,   11,   92,   28,  -85,  -41,   82,   39,  -83,  -37,   83},
    {  82,   73,    3,  -80,  -91,   10,   94,   18,  -84,   -5,   80,  -17,  -70,   47,   55,  -68},
    {  89,   48,  -53,  -96,   -5,   89,   16,  -83,   12,   75,  -51,  -43,   81,  -14,  -75,   67},
    { 100,    9,  -99,  -45,   84,   36,  -81,    4,   73,  -60,  -17,   76,  -55,  -21,   81,  -56},
    {  92,  -35,  -91,   42,   71,  -69,  -22,   85,  -60,  -17,   77,  -70,   11,   56,  -83,   49},
    {  84,  -71,  -46,   88,  -16,  -67,   79,  -22,  -46,   84,  -68,   15,   46,  -83,   82,  -40},
    {  69,  -83,    2,   75,  -78,   18,   50,  -85,   81,  -43,   -6,   53,  -81,   88,  -69,   32},
    {  50,  -79,   47,   14,  -60,   67,  -37,   -6,   49,  -80,   98,  -99,   90,  -71,   48,  -19},
    {  51,  -89,   73,  -17,  -45,   89, -103,   97,  -83,   71,  -56,   44,  -31,   22,  -11,    5},
    {  44,  -95,  124, -126,  108,  -81,   54,  -30,   16,   -6,    1,    3,   -4,    5,   -3,    2},
};

// Returns the Ddtx_KernelN matrix (N == 8 or 16), or nullptr for any other N.
// The matrix is returned in the §7.15.2.1 [coefficient][sample] orientation;
// FDDT is the same matrix with the sample (second) index mirrored, which the
// transform code applies explicitly.
inline const std::int16_t* ddt_kernel(int n) {
  if (n == 8) return &kDdtxKernel8[0][0];
  if (n == 16) return &kDdtxKernel16[0][0];
  return nullptr;
}

}  // namespace av2enc::m2::ddt
