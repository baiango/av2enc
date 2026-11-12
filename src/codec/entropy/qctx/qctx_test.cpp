// Coefficient-CDF qctx table tests.
//
// Covers:
//   * §8.3.1 slice selection from raw base_q_idx (all boundaries, 8/10-bit);
//   * every generated table, every qctx slice, every context distribution:
//     shape metadata and §8.2.6 CDF validity (monotonic boundaries, rate,
//     adaptation counter);
//   * slice 0 byte-for-byte against the shipping slice-0 tables
//     (basic_cdf_tables.h and rdoq_cdf_tables.h; still_syntax.cpp's local
//     Eob_Pt_16/32 are covered through the rdoq header);
//   * distinct aggregate fingerprints per slice;
//   * the typed view/`copy_cdf` plumbing the M1/M2 writers will use.
//
// Standalone: build with run_tests.sh (no CMakeLists edit required).
#include "codec/entropy/qctx/qctx.h"

#include "pipeline/still/basic_cdf_tables.h"
#include "codec/rate/rdoq/rdoq_cdf_tables.h"
#include "codec/entropy/av2/av2_entropy.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

int g_failures = 0;
int g_checks = 0;

void fail(const char* file, int line, const std::string& what) {
  ++g_failures;
  std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what.c_str());
}

#define CHECK(cond)                                                       \
  do {                                                                    \
    ++g_checks;                                                           \
    if (!(cond)) fail(__FILE__, __LINE__, #cond);                         \
  } while (0)

#define CHECK_EQ(a, b)                                                    \
  do {                                                                    \
    ++g_checks;                                                           \
    const auto check_a = (a);                                             \
    const auto check_b = (b);                                             \
    if (!(check_a == check_b)) {                                          \
      fail(__FILE__, __LINE__,                                            \
           std::string(#a " == " #b " (") + std::to_string(check_a) +     \
               " vs " + std::to_string(check_b) + ")");                   \
    }                                                                     \
  } while (0)

// Byte-for-byte comparison of our qctx slice 0 with a shipping slice-0 array.
// The inner dimensions match, so sizeof and memcmp are exact.
#define CHECK_SLICE0_SHIPPING(name, shipping)                             \
  do {                                                                    \
    ++g_checks;                                                           \
    if (sizeof(name[0]) != sizeof(shipping)) {                            \
      fail(__FILE__, __LINE__, #name " inner size mismatch vs " #shipping); \
    } else if (std::memcmp(name[0], shipping, sizeof(name[0])) != 0) {    \
      fail(__FILE__, __LINE__, #name " slice 0 differs from " #shipping); \
    }                                                                     \
  } while (0)

using namespace av2enc::still::qctx;
using av2enc::still::qctx::kCoeffCdfSliceCount;
using av2enc::still::qctx::kCoeffCdfTableCount;
using av2enc::still::qctx::kCoeffCdfTables;
using av2enc::still::qctx::kCoeffCdfSlices;
using av2enc::still::qctx::slice_for_base_q_idx;
using av2enc::still::qctx::tables_for_base_q_idx;
using av2enc::still::qctx::tables_for_slice;

void test_slice_selection() {
  // Every boundary, both sides, plus out-of-range defensive values.
  CHECK_EQ(slice_for_base_q_idx(-1000), 0);
  CHECK_EQ(slice_for_base_q_idx(-1), 0);
  CHECK_EQ(slice_for_base_q_idx(0), 0);
  CHECK_EQ(slice_for_base_q_idx(1), 0);
  CHECK_EQ(slice_for_base_q_idx(89), 0);
  CHECK_EQ(slice_for_base_q_idx(90), 0);
  CHECK_EQ(slice_for_base_q_idx(91), 1);
  CHECK_EQ(slice_for_base_q_idx(139), 1);
  CHECK_EQ(slice_for_base_q_idx(140), 1);
  CHECK_EQ(slice_for_base_q_idx(141), 2);
  CHECK_EQ(slice_for_base_q_idx(189), 2);
  CHECK_EQ(slice_for_base_q_idx(190), 2);
  CHECK_EQ(slice_for_base_q_idx(191), 3);
  CHECK_EQ(slice_for_base_q_idx(255), 3);
  CHECK_EQ(slice_for_base_q_idx(256), 3);
  // 10-bit MaxQ is 303 (§6.4.1); the raw value must not be clamped to 255.
  CHECK_EQ(slice_for_base_q_idx(303), 3);
  CHECK_EQ(slice_for_base_q_idx(999), 3);

  // Clamped slice-index access.
  CHECK(&tables_for_slice(-1) == &kCoeffCdfSlices[0]);
  CHECK(&tables_for_slice(0) == &kCoeffCdfSlices[0]);
  CHECK(&tables_for_slice(3) == &kCoeffCdfSlices[3]);
  CHECK(&tables_for_slice(4) == &kCoeffCdfSlices[3]);

  // and the frame-level API resolves to the same view.
  CHECK(tables_for_base_q_idx(90).coeff_base == &kCoeffBaseCdf[0]);
  CHECK(tables_for_base_q_idx(91).coeff_base == &kCoeffBaseCdf[1]);
  CHECK(tables_for_base_q_idx(140).coeff_base == &kCoeffBaseCdf[1]);
  CHECK(tables_for_base_q_idx(141).coeff_base == &kCoeffBaseCdf[2]);
  CHECK(tables_for_base_q_idx(190).coeff_base == &kCoeffBaseCdf[2]);
  CHECK(tables_for_base_q_idx(191).coeff_base == &kCoeffBaseCdf[3]);
  CHECK(tables_for_base_q_idx(303).coeff_base == &kCoeffBaseCdf[3]);
  CHECK(tables_for_base_q_idx(90).txb_skip == &kTxbSkipCdf[0]);
  CHECK(tables_for_base_q_idx(191).txb_skip == &kTxbSkipCdf[3]);
  CHECK(tables_for_base_q_idx(91).eob_pt_16 == &kEobPt16Cdf[1]);
  CHECK(tables_for_base_q_idx(191).dc_sign == &kDcSignCdf[3]);
}

void test_metadata_and_validity() {
  CHECK_EQ(kCoeffCdfTableCount, 27);
  CHECK_EQ(kCoeffCdfSliceCount, 4);

  std::size_t total_scalars = 0;
  std::size_t total_distributions = 0;
  for (int table = 0; table < kCoeffCdfTableCount; ++table) {
    const auto& info = kCoeffCdfTables[table];
    CHECK(info.name != nullptr);
    CHECK(info.slice0 != nullptr);
    CHECK(info.symbols >= 2 && info.symbols <= 8);
    const int stride = static_cast<int>(info.slice_stride);
    CHECK(stride > 0);
    // A slice is an integral number of N+1-entry distributions.
    CHECK_EQ(stride % (info.symbols + 1), 0);
    total_scalars += info.slice_stride;
    total_distributions += info.slice_stride /
                           static_cast<std::size_t>(info.symbols + 1);
  }
  // 27 tables * 4 slices = 18,868 scalars and 3,424 distributions.  If the
  // pinned spec changes these counts, the header must be regenerated and the
  // qctx module re-reviewed; fail loudly instead of silently changing models.
  CHECK_EQ(total_scalars, std::size_t{4717});
  CHECK_EQ(total_distributions, std::size_t{856});

  // Every distribution of every table in every qctx slice must be a valid
  // §8.2.6 CDF.  cdf_is_valid checks monotonic boundaries, the rate index and
  // the adaptation counter.
  int checked_distributions = 0;
  int checked_scalars = 0;
  for (int table = 0; table < kCoeffCdfTableCount; ++table) {
    const auto& info = kCoeffCdfTables[table];
    const int symbols = info.symbols;
    const int step = symbols + 1;
    for (int slice = 0; slice < kCoeffCdfSliceCount; ++slice) {
      const std::uint16_t* base =
          info.slice0 + static_cast<std::size_t>(slice) * info.slice_stride;
      for (std::size_t offset = 0; offset < info.slice_stride;
           offset += static_cast<std::size_t>(step)) {
        ++checked_distributions;
        if (!av2enc::av2::cdf_is_valid(base + offset, symbols)) {
          fail(__FILE__, __LINE__,
               std::string("invalid CDF: ") + info.name + " slice " +
                   std::to_string(slice) + " offset " +
                   std::to_string(offset));
        }
      }
      checked_scalars += static_cast<int>(info.slice_stride);
    }
  }
  CHECK_EQ(checked_scalars, 18868);
  CHECK_EQ(checked_distributions, 3424);
}

// FNV-1a over a slice's scalars, used only to prove the four slices are not
// accidental copies of one another.
std::uint32_t fingerprint(int slice) {
  std::uint32_t hash = 2166136261U;
  for (int table = 0; table < kCoeffCdfTableCount; ++table) {
    const auto& info = kCoeffCdfTables[table];
    const std::uint16_t* base =
        info.slice0 + static_cast<std::size_t>(slice) * info.slice_stride;
    for (std::size_t i = 0; i < info.slice_stride; ++i) {
      hash ^= base[i];
      hash *= 16777619U;
    }
  }
  return hash;
}

void test_slice_content_distinctness() {
  const std::uint32_t f0 = fingerprint(0);
  const std::uint32_t f1 = fingerprint(1);
  const std::uint32_t f2 = fingerprint(2);
  const std::uint32_t f3 = fingerprint(3);
  CHECK(f0 != f1);
  CHECK(f0 != f2);
  CHECK(f0 != f3);
  CHECK(f1 != f2);
  CHECK(f1 != f3);
  CHECK(f2 != f3);

  // Major coefficient tables must actually change at every boundary.  The
  // spec has at least one table whose four slices are identical
  // (Default_Coeff_Base_Ph_Cdf), so this is intentionally not a generic
  // property.
  auto differs = [](const void* a, const void* b, std::size_t bytes) {
    return std::memcmp(a, b, bytes) != 0;
  };
  const std::size_t stride = kCoeffCdfTables[1].slice_stride;  // Coeff_Base
  CHECK(differs(kCoeffBaseCdf[0], kCoeffBaseCdf[1],
                stride * sizeof(std::uint16_t)));
  CHECK(differs(kCoeffBaseCdf[1], kCoeffBaseCdf[2],
                stride * sizeof(std::uint16_t)));
  CHECK(differs(kCoeffBaseCdf[2], kCoeffBaseCdf[3],
                stride * sizeof(std::uint16_t)));
  CHECK(differs(kTxbSkipCdf[0], kTxbSkipCdf[1],
                300 * sizeof(std::uint16_t)));
  CHECK(differs(kEobPt64Cdf[0], kEobPt64Cdf[1],
                24 * sizeof(std::uint16_t)));
  CHECK(differs(kCoeffBrCdf[0], kCoeffBrCdf[1],
                35 * sizeof(std::uint16_t)));
}

void test_slice0_matches_shipping() {
  // basic_cdf_tables.h: 20 coefficient tables, slice 0 only.
  CHECK_SLICE0_SHIPPING(kCoeffBaseCdf, av2enc::basic::kCoeffBaseCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseEobCdf, av2enc::basic::kCoeffBaseEobCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseEobUvCdf, av2enc::basic::kCoeffBaseEobUvCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseLfCdf, av2enc::basic::kCoeffBaseLfCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseLfEobCdf, av2enc::basic::kCoeffBaseLfEobCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseLfEobUvCdf,
                        av2enc::basic::kCoeffBaseLfEobUvCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseLfUvCdf, av2enc::basic::kCoeffBaseLfUvCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseUvCdf, av2enc::basic::kCoeffBaseUvCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBrCdf, av2enc::basic::kCoeffBrCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBrLfCdf, av2enc::basic::kCoeffBrLfCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBrUvCdf, av2enc::basic::kCoeffBrUvCdf);
  CHECK_SLICE0_SHIPPING(kDcSignCdf, av2enc::basic::kDcSignCdf);
  CHECK_SLICE0_SHIPPING(kEobExtraCdf, av2enc::basic::kEobExtraCdf);
  CHECK_SLICE0_SHIPPING(kEobPt1024Cdf, av2enc::basic::kEobPt1024Cdf);
  CHECK_SLICE0_SHIPPING(kEobPt128Cdf, av2enc::basic::kEobPt128Cdf);
  CHECK_SLICE0_SHIPPING(kEobPt256Cdf, av2enc::basic::kEobPt256Cdf);
  CHECK_SLICE0_SHIPPING(kEobPt512Cdf, av2enc::basic::kEobPt512Cdf);
  CHECK_SLICE0_SHIPPING(kEobPt64Cdf, av2enc::basic::kEobPt64Cdf);
  CHECK_SLICE0_SHIPPING(kTxbSkipCdf, av2enc::basic::kTxbSkipCdf);
  CHECK_SLICE0_SHIPPING(kVTxbSkipCdf, av2enc::basic::kVTxbSkipCdf);

  // rdoq_cdf_tables.h: the same 20 plus Eob_Pt_16/32 (the two tables the M1
  // writer currently defines locally in still_syntax.cpp).
  CHECK_SLICE0_SHIPPING(kEobPt16Cdf, av2enc::rdoq::cdf::kEobPt16Cdf);
  CHECK_SLICE0_SHIPPING(kEobPt32Cdf, av2enc::rdoq::cdf::kEobPt32Cdf);
  CHECK_SLICE0_SHIPPING(kEobPt64Cdf, av2enc::rdoq::cdf::kEobPt64Cdf);
  CHECK_SLICE0_SHIPPING(kEobPt128Cdf, av2enc::rdoq::cdf::kEobPt128Cdf);
  CHECK_SLICE0_SHIPPING(kEobPt256Cdf, av2enc::rdoq::cdf::kEobPt256Cdf);
  CHECK_SLICE0_SHIPPING(kEobPt512Cdf, av2enc::rdoq::cdf::kEobPt512Cdf);
  CHECK_SLICE0_SHIPPING(kEobPt1024Cdf, av2enc::rdoq::cdf::kEobPt1024Cdf);
  CHECK_SLICE0_SHIPPING(kEobExtraCdf, av2enc::rdoq::cdf::kEobExtraCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseCdf, av2enc::rdoq::cdf::kCoeffBaseCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseLfCdf, av2enc::rdoq::cdf::kCoeffBaseLfCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseEobCdf,
                        av2enc::rdoq::cdf::kCoeffBaseEobCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseLfEobCdf,
                        av2enc::rdoq::cdf::kCoeffBaseLfEobCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBrCdf, av2enc::rdoq::cdf::kCoeffBrCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBrLfCdf, av2enc::rdoq::cdf::kCoeffBrLfCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseUvCdf, av2enc::rdoq::cdf::kCoeffBaseUvCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseLfUvCdf,
                        av2enc::rdoq::cdf::kCoeffBaseLfUvCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseEobUvCdf,
                        av2enc::rdoq::cdf::kCoeffBaseEobUvCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBaseLfEobUvCdf,
                        av2enc::rdoq::cdf::kCoeffBaseLfEobUvCdf);
  CHECK_SLICE0_SHIPPING(kCoeffBrUvCdf, av2enc::rdoq::cdf::kCoeffBrUvCdf);
  CHECK_SLICE0_SHIPPING(kTxbSkipCdf, av2enc::rdoq::cdf::kTxbSkipCdf);
  CHECK_SLICE0_SHIPPING(kVTxbSkipCdf, av2enc::rdoq::cdf::kVTxbSkipCdf);
  CHECK_SLICE0_SHIPPING(kDcSignCdf, av2enc::rdoq::cdf::kDcSignCdf);
}

void test_copy_api() {
  // The copy helper must accept exactly the pointer-to-inner-array the view
  // struct exposes, with a compile-time shape check.
  std::uint16_t txb_skip[2][5][10][3] = {};
  av2enc::still::qctx::copy_cdf(txb_skip, *tables_for_base_q_idx(191).txb_skip);
  CHECK(std::memcmp(txb_skip, kTxbSkipCdf[3], sizeof(txb_skip)) == 0);

  std::uint16_t dc_sign[2][2][3][3] = {};
  av2enc::still::qctx::copy_cdf(dc_sign, *tables_for_base_q_idx(120).dc_sign);
  CHECK(std::memcmp(dc_sign, kDcSignCdf[1], sizeof(dc_sign)) == 0);

  #define CHECK_VIEW(table_field, table_name)                              \
    do {                                                                   \
      CHECK(tables_for_slice(0).table_field == &table_name[0]);            \
      CHECK(tables_for_slice(1).table_field == &table_name[1]);            \
      CHECK(tables_for_slice(2).table_field == &table_name[2]);            \
      CHECK(tables_for_slice(3).table_field == &table_name[3]);            \
    } while (0)
  CHECK_VIEW(txb_skip, kTxbSkipCdf);
  CHECK_VIEW(v_txb_skip, kVTxbSkipCdf);
  CHECK_VIEW(eob_pt_16, kEobPt16Cdf);
  CHECK_VIEW(eob_pt_32, kEobPt32Cdf);
  CHECK_VIEW(eob_pt_64, kEobPt64Cdf);
  CHECK_VIEW(eob_pt_128, kEobPt128Cdf);
  CHECK_VIEW(eob_pt_256, kEobPt256Cdf);
  CHECK_VIEW(eob_pt_512, kEobPt512Cdf);
  CHECK_VIEW(eob_pt_1024, kEobPt1024Cdf);
  CHECK_VIEW(eob_extra, kEobExtraCdf);
  CHECK_VIEW(coeff_base, kCoeffBaseCdf);
  CHECK_VIEW(coeff_base_lf, kCoeffBaseLfCdf);
  CHECK_VIEW(coeff_base_eob, kCoeffBaseEobCdf);
  CHECK_VIEW(coeff_base_lf_eob, kCoeffBaseLfEobCdf);
  CHECK_VIEW(coeff_br, kCoeffBrCdf);
  CHECK_VIEW(coeff_br_lf, kCoeffBrLfCdf);
  CHECK_VIEW(coeff_base_uv, kCoeffBaseUvCdf);
  CHECK_VIEW(coeff_base_lf_uv, kCoeffBaseLfUvCdf);
  CHECK_VIEW(coeff_base_eob_uv, kCoeffBaseEobUvCdf);
  CHECK_VIEW(coeff_base_lf_eob_uv, kCoeffBaseLfEobUvCdf);
  CHECK_VIEW(coeff_br_uv, kCoeffBrUvCdf);
  CHECK_VIEW(dc_sign, kDcSignCdf);
  CHECK_VIEW(coeff_base_bob, kCoeffBaseBobCdf);
  CHECK_VIEW(coeff_base_idtx, kCoeffBaseIdtxCdf);
  CHECK_VIEW(coeff_br_idtx, kCoeffBrIdtxCdf);
  CHECK_VIEW(coeff_base_ph, kCoeffBasePhCdf);
  CHECK_VIEW(idtx_sign, kIdtxSignCdf);
  #undef CHECK_VIEW
}

}  // namespace

int main() {
  test_slice_selection();
  test_metadata_and_validity();
  test_slice_content_distinctness();
  test_slice0_matches_shipping();
  test_copy_api();

  if (g_failures != 0) {
    std::fprintf(stderr, "qctx_test: %d failure(s) in %d checks\n",
                 g_failures, g_checks);
    return 1;
  }
  std::printf("qctx_test: PASS (%d checks)\n", g_checks);
  return 0;
}
