// Pinned §5.20.8.3 / Tx_Type_In_Set_Inter masks for the DDT tests.
//
// Test-only header shared by ddt_device_emulation_test.cpp and ddt_gpu_test.cu.
// The masks are the same ones pinned in src/encoder/m2/ddt/ddt_test.cpp:
//   legal_mask     = Tx_Type_In_Set_Inter for inter blocks, reduced_tx_set = 0
//   component_mask = any ADST/FDST at adjusted length 8 or 16
//   ddt_mask       = legal_mask & component_mask  (the 100 legal DDT pairs)
#pragma once

#include <cstdint>

#include "codec/transform/variants/tx_variants.h"

namespace av2enc::cuda::ddt::test {

struct SizeRow {
  av2enc::txv::TxSize size;
  std::uint16_t legal_mask;
  std::uint16_t component_mask;
  std::uint16_t ddt_mask;
};

inline constexpr SizeRow kSizeRows[25] = {
    {av2enc::txv::TxSize::k4x4, 0xffff, 0x0000, 0x0000},
    {av2enc::txv::TxSize::k8x8, 0xffff, 0xf1fe, 0xf1fe},
    {av2enc::txv::TxSize::k16x16, 0x0fff, 0xf1fe, 0x01fe},
    {av2enc::txv::TxSize::k32x32, 0x0201, 0x0000, 0x0000},
    {av2enc::txv::TxSize::k64x64, 0x0001, 0x0000, 0x0000},
    {av2enc::txv::TxSize::k4x8, 0xffff, 0x51da, 0x51da},
    {av2enc::txv::TxSize::k8x4, 0xffff, 0xa1ec, 0xa1ec},
    {av2enc::txv::TxSize::k8x16, 0xffff, 0xf1fe, 0xf1fe},
    {av2enc::txv::TxSize::k16x8, 0xffff, 0xf1fe, 0xf1fe},
    {av2enc::txv::TxSize::k16x32, 0xae25, 0xa1ec, 0xa024},
    {av2enc::txv::TxSize::k32x16, 0x5e13, 0x51da, 0x5012},
    {av2enc::txv::TxSize::k32x64, 0x0001, 0x0000, 0x0000},
    {av2enc::txv::TxSize::k64x32, 0x0001, 0x0000, 0x0000},
    {av2enc::txv::TxSize::k4x16, 0xffff, 0x51da, 0x51da},
    {av2enc::txv::TxSize::k16x4, 0xffff, 0xa1ec, 0xa1ec},
    {av2enc::txv::TxSize::k8x32, 0xae25, 0xa1ec, 0xa024},
    {av2enc::txv::TxSize::k32x8, 0x5e13, 0x51da, 0x5012},
    {av2enc::txv::TxSize::k16x64, 0x0425, 0xa1ec, 0x0024},
    {av2enc::txv::TxSize::k64x16, 0x0813, 0x51da, 0x0012},
    {av2enc::txv::TxSize::k4x32, 0xae25, 0x0000, 0x0000},
    {av2enc::txv::TxSize::k32x4, 0x5e13, 0x0000, 0x0000},
    {av2enc::txv::TxSize::k8x64, 0x0425, 0xa1ec, 0x0024},
    {av2enc::txv::TxSize::k64x8, 0x0813, 0x51da, 0x0012},
    {av2enc::txv::TxSize::k4x64, 0x0425, 0x0000, 0x0000},
    {av2enc::txv::TxSize::k64x4, 0x0813, 0x0000, 0x0000},
};

inline constexpr int kLegalDdtPairCount = 100;

}  // namespace av2enc::cuda::ddt::test
