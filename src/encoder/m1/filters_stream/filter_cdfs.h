// Default non-coefficient CDFs for the M1 filter/restoration syntax.
//
// Generated from spec/tables/default_cdf.json (§9.3).  Do not hand-edit.
// The CDEF CDFs already live in src/cuda/m1/cdef/m1_cdef_cdf.h and are
// included directly by filters_stream.cpp.
#ifndef AV2ENC_ENCODER_M1_FILTERS_STREAM_FILTER_CDFS_H_
#define AV2ENC_ENCODER_M1_FILTERS_STREAM_FILTER_CDFS_H_

#include <cstdint>

namespace av2enc::m1::filters_stream {

// §8.3 ccso_blk: TileCcsoBlkCdf[plane][ctx], 4 contexts per plane.
inline constexpr std::uint16_t kCcsoBlkCdf[3][4][3] = 
    {{{18469, 62, 0},
            {16384, 0, 0},
            {4949, 37, 0},
            {16384, 0, 0}},
        {{23470, 37, 0},
            {16384, 0, 0},
            {6666, 37, 0},
            {16384, 0, 0}},
        {{22914, 37, 0},
            {16384, 0, 0},
            {6993, 37, 0},
            {16384, 0, 0}}};

// §5.20.10.5 use_wiener_ns.
inline constexpr std::uint16_t kUseWienerNsCdf[3] = 
    {9051, 32, 0};

// §8.3 wiener_ns_length: TileWienerNsLengthCdf[Min(plane, 1)].
inline constexpr std::uint16_t kWienerNsLengthCdf[2][3] = 
    {{4898, 56, 0},
        {6088, 26, 0}};

// §5.20.10.6 wiener_ns_uv_sym.
inline constexpr std::uint16_t kWienerNsUvSymCdf[3] = 
    {29286, 65, 0};

// §8.3 wiener_ns_base (decode_4part).
inline constexpr std::uint16_t kWienerNsBaseCdf[5] = 
    {16557, 24352, 29677, 6, 0};

// §8.3 flex_restoration_type: TileFlexRestorationTypeCdf[tool][plane].
inline constexpr std::uint16_t kFlexRestorationTypeCdf[3][3][3] = 
    {{{25542, 62, 0},
            {16384, 0, 0},
            {16384, 0, 0}},
        {{25550, 37, 0},
            {16384, 0, 0},
            {16384, 0, 0}},
        {{16384, 0, 0},
            {16384, 0, 0},
            {16384, 0, 0}}};

}  // namespace av2enc::m1::filters_stream

#endif  // AV2ENC_ENCODER_M1_FILTERS_STREAM_FILTER_CDFS_H_
