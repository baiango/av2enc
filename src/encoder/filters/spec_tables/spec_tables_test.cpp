// Structural checks for the recovered normative tables.
//
// Byte provenance is enforced by generate_spec_tables.py (raw SHA-256 pins plus
// `--check` on the generated headers, run by build_and_test.sh).  This test
// checks the invariants the filter implementations rely on: extents, index
// ranges, and the constraints printed in the specification.

#include <cstdint>
#include <cstdio>

#include "encoder/filters/spec_tables/film_grain_tables.h"
#include "encoder/filters/spec_tables/gdf_tables.h"
#include "encoder/filters/spec_tables/pc_wiener_tables.h"
#include "encoder/filters/spec_tables/spec_tables_provenance.h"
#include "encoder/filters/spec_tables/spec_test_common.h"

namespace {

using namespace av2enc::filters::spec::test;
using namespace av2enc::filters::spec_tables;

void TestProvenance() {
  const int n = static_cast<int>(sizeof(kTableProvenance) / sizeof(kTableProvenance[0]));
  CHECK(n == 10);
  for (int i = 0; i < n; ++i) {
    const char* sha = kTableProvenance[i].sha256;
    int len = 0;
    while (sha[len] != '\0') ++len;
    CHECK(len == 64);
    CHECK(kTableProvenance[i].filename != nullptr);
    CHECK(kTableProvenance[i].spec_name != nullptr);
  }
  CHECK(kAttachmentCommit[0] != '\0');
  CHECK(kSpecVersion[0] != '\0');
}

void TestGdfTables() {
  int alpha_max = 0;
  for (auto& a : kGdfAlpha) {
    for (auto& b : a) {
      for (auto& c : b) {
        for (int v : c) {
          CHECK(v <= 511);
          if (v > alpha_max) alpha_max = v;
        }
      }
    }
  }
  CHECK(alpha_max == 511);

  for (auto& a : kGdfWeight) {
    for (auto& b : a) {
      for (auto& c : b) {
        for (auto& d : c) {
          for (int v : d) CHECK(v >= -512 && v <= 511);
        }
      }
    }
  }

  for (auto& a : kGdfBias) {
    for (auto& b : a) {
      for (int v : b) CHECK(v > -32768 && v < 32768);
    }
  }

  for (auto& a : kGdfIntraError) {
    for (int v : a) CHECK(v >= -128 && v <= 127);
  }
  for (auto& a : kGdfInterError) {
    for (auto& b : a) {
      for (int v : b) CHECK(v >= -128 && v <= 127);
    }
  }

  // Gdf_Coords: 18 distinct non-zero offsets.
  for (int i = 0; i < 18; ++i) {
    CHECK(!(kGdfCoords[i].dy == 0 && kGdfCoords[i].dx == 0));
    for (int j = i + 1; j < 18; ++j) {
      CHECK(!(kGdfCoords[i].dy == kGdfCoords[j].dy &&
              kGdfCoords[i].dx == kGdfCoords[j].dx));
    }
  }
}

void TestPcWienerTables() {
  for (auto& set : kPcWienerFilters) {
    for (auto& filt : set) {
      for (int v : filt) CHECK(v >= -512 && v <= 511);
    }
  }
  for (auto& set : kPcWienerSubClassify) {
    for (int v : set) CHECK(v < kPcWienerFiltersPerSet);
  }

  // §5.18.7.11 Decode_Num_Filter_Classes[8].
  const int decode_num_classes[8] = {1, 2, 3, 4, 6, 8, 12, 16};
  for (auto& set : kPcWienerSubClassify2) {
    for (int j = 0; j < 8; ++j) {
      for (int c = 0; c < kPcWienerLutClasses; ++c) {
        CHECK(set[j][c] < decode_num_classes[j]);
      }
    }
  }

  // §7.14.2 qlookup(q) = Ac_Qlookup[q] for q < 25, else the printed recurrence.
  CHECK(kQlookup[0] == kAcQlookup[0]);
  for (int q = 0; q < 25; ++q) CHECK(kQlookup[q] == kAcQlookup[q]);
  for (int q = 25; q < 352; ++q) {
    const int idx = ((q - 1) % 24) + 1;
    const int shift = (q - 1) / 24;
    CHECK(kQlookup[q] == (kAcQlookup[idx] << shift));
    CHECK(kQlookup[q] >= kQlookup[q - 1]);
  }
  CHECK(kQlookup[255] == 61440);
}

void TestFilmGrainTables() {
  for (int v : kGaussianSequence) {
    CHECK(v >= -2048 && v <= 2047);
    CHECK((v % 4) == 0);
  }
  CHECK(kGaussianSequenceSize == 2048);
  CHECK(kGaussianSequence[0] == 56);
  CHECK(kGaussianSequence[2047] == -484);
}

}  // namespace

int main() {
  TestProvenance();
  TestGdfTables();
  TestPcWienerTables();
  TestFilmGrainTables();
  return Finish("spec_tables_test");
}
