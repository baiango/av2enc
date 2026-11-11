// AV2 v1.0.0 §5.18.8.1 / §5.20.6.1 / §5.20.6.2 / §5.20.6.3
// Transform-mode and transform-partition signalling.
//
// The shipped M1/M2 paths hard-code `tx_mode_select = 0` (TX_MODE_LARGEST):
// one transform per plane whose size equals the coding block.  This component
// implements the missing TX_MODE_SELECT syntax:
//
//   read_tx_mode()            tx_mode_select, TxMode
//   read_tx_size(allowSelect) maxRectTxSize / 64x64 chunking / TX partition
//   read_block_tx_size()      inter non-skip driver (per-64x64 chunk)
//   read_tx_partition()       tx_do_partition / tx_partition_type /
//                             tx_2or3_partition_type + set_tx_size()
//
// The writer mirrors the reader, and both use §9.3 default CDFs copied from
// the pinned spec tables.  The geometry is the normative `find_tx_size()` /
// `set_tx_size()` pair, so a caller can enumerate legal splits per block size
// (`legal_splits()`), signal one, and reconstruct the exact LumaTxSizes grid.
//
// Not implemented here: the lossless `lossless_tx_size` symbol (§5.20.6.1,
// outside this component's scope).  The lossless *inference* branches are
// modelled exactly; a stream that would need the symbol is rejected with a
// clear error instead of being mis-signalled.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace av2enc::m2::txpartition {

// §6.19.6.1 enum_tx_size.
inline constexpr int kTxSize4x4 = 0;
inline constexpr int kTxSize8x8 = 1;
inline constexpr int kTxSize16x16 = 2;
inline constexpr int kTxSize32x32 = 3;
inline constexpr int kTxSize64x64 = 4;
inline constexpr int kTxSize4x64 = 23;
inline constexpr int kTxSize64x4 = 24;
inline constexpr int kTxSizeCount = 25;

// §6.19.3.1 enum_block_size.
inline constexpr int kBlock4x4 = 0;
inline constexpr int kBlockCount = 29;

// §9.3 constants.
inline constexpr int kTxPartitionTypeNum = 7;
inline constexpr int kTxPartitionTypeNumVertAndHorz = 14;
inline constexpr int kTxPartitionTypeNumVertOrHorz = 3;
inline constexpr int kTxFmSplitGroup = 9;
inline constexpr int kFscModes = 2;

// Table 6.23 txPartition values.  HORZ5/VERT5 are in the set_tx_size()
// semantics table but are not reachable from the signalling syntax in
// §5.20.6.3 (tx_2or3_partition_type only selects 2-way / 4-way).
enum class TxPartition : std::uint8_t {
  kNone = 0,
  kSplit = 1,
  kHorz = 2,
  kVert = 3,
  kHorz4 = 4,
  kVert4 = 5,
  kHorz5 = 6,
  kVert5 = 7,
};

const char* tx_partition_name(TxPartition part);

enum class TxMode : std::uint8_t {
  kLargest = 0,   // TX_MODE_LARGEST
  kSelect = 1,    // TX_MODE_SELECT
  kOnly4x4 = 2,   // ONLY_4X4 (CodedLossless)
};

// ---------------------------------------------------------------------------
// §9.2 geometry tables (generated from spec/tables/block_tables.json).
// ---------------------------------------------------------------------------

int max_tx_size_rect(int mi_size);       // Max_Tx_Size_Rect[MiSize]
int tx_part_group(int mi_size);          // Size_To_Tx_Part_Group_Lookup
int tx_type_group_vert_and_horz(int mi_size);  // 255 == BLOCK_INVALID
int tx_type_group_vert_or_horz(int mi_size);   // 255 == BLOCK_INVALID
int block_width_mi(int mi_size);         // Num_4x4_Blocks_Wide
int block_height_mi(int mi_size);        // Num_4x4_Blocks_High
int block_width_px(int mi_size);
int block_height_px(int mi_size);
int tx_width(int tx_size);
int tx_height(int tx_size);
int tx_width_log2(int tx_size);
int tx_height_log2(int tx_size);
int adjusted_tx_size(int tx_size);       // Adjusted_Tx_Size

// §5.20.6.3 find_tx_size(w, h).  Returns -1 for TX_INVALID.
int find_tx_size(int w, int h);

// ---------------------------------------------------------------------------
// Legal split enumeration
// ---------------------------------------------------------------------------

struct SplitPart {
  int mi_row = 0;
  int mi_col = 0;
  int tx_size = -1;
};

struct SplitGeometry {
  int count = 0;
  // Parts are in *decode order* (the order residual() visits them): row-major
  // over the block for SPLIT/HORZ/VERT, top-to-bottom for HORZ4 and
  // left-to-right for VERT4.
  SplitPart parts[4];
};

struct LegalSplits {
  // True when §5.20.6.3 can code a partition at all: MiSize > BLOCK_4X4 and
  // Block_Width/Block_Height <= 64.  Larger blocks are chunked into 64x64
  // transforms by read_tx_size()/read_block_tx_size() and code no bits here.
  bool applicable = false;
  bool chunked = false;
  bool allow_horz = false;
  bool allow_vert = false;
  // Both directions legal -> tx_partition_type (3 useful symbols of 7).
  bool has_partition_type = false;
  // Exactly one direction legal and Size_To_Tx_Type_Group_Vert_Or_Horz > 0 ->
  // tx_2or3_partition_type (2 or 4 parts).
  bool has_2or3 = false;
  bool reduced_tx_part_set = false;
  int option_count = 0;
  TxPartition options[8] = {};
};

LegalSplits legal_splits(int mi_size, bool reduced_tx_part_set = false);

// Resolves one TxPartition into concrete TX blocks.  Returns false when the
// partition is not legal for this block size (including HORZ4/VERT4 in the
// reduced set and HORZ5/VERT5, which the syntax cannot signal).
bool tx_partition_geometry(int mi_size, int mi_row, int mi_col,
                           TxPartition part, SplitGeometry* out,
                           std::string* error = nullptr);

// True when the *syntax* can represent this partition for this block size,
// independent of the caller's reduced-set configuration.
bool tx_partition_signallable(int mi_size, TxPartition part);

// ---------------------------------------------------------------------------
// §9.3 CDF state (one copy per tile/frame; the entropy coder adapts in place).
// ---------------------------------------------------------------------------

struct TxPartitionCdfs {
  std::uint16_t tx_do_partition[kFscModes][2][kTxFmSplitGroup][3];
  std::uint16_t tx_partition_type[kFscModes][2]
                                 [kTxPartitionTypeNumVertAndHorz]
                                 [kTxPartitionTypeNum + 1];
  std::uint16_t tx_partition_type_reduced[kFscModes][2]
                                         [kTxPartitionTypeNumVertAndHorz]
                                         [kTxPartitionTypeNum + 1];
  std::uint16_t tx_2or3_partition_type[kFscModes][2]
                                      [kTxPartitionTypeNumVertOrHorz - 1][3];

  // True when reset() selected Default_Tx_Partition_Type_Reduced_Cdf.
  bool reduced = false;

  // Copies the §9.3 defaults.  `reduced_tx_part_set` selects the uniform
  // TxPartitionTypeCdf variant exactly as §9.3 / init_tile() do.
  void reset(bool reduced_tx_part_set);

  std::uint16_t* do_partition_cdf(bool fsc_mode, bool is_inter, int mi_size);
  std::uint16_t* partition_type_cdf(bool fsc_mode, bool is_inter,
                                    int mi_size);
  std::uint16_t* partition_2or3_cdf(bool fsc_mode, bool is_inter, int mi_size);
};

// FNV-1a over the initial CDF bytes; pinned by the test as a table golden.
std::uint64_t cdf_fingerprint(const TxPartitionCdfs& cdfs);

// ---------------------------------------------------------------------------
// Symbol plumbing (the entropy coder is the caller's; this component owns the
// syntax and contexts, not the range coder).
// ---------------------------------------------------------------------------

class SymbolSink {
 public:
  virtual ~SymbolSink() = default;
  virtual bool symbol(std::uint16_t* cdf, int n, int value,
                      std::string& error) = 0;
};

class SymbolSource {
 public:
  virtual ~SymbolSource() = default;
  virtual bool symbol(std::uint16_t* cdf, int n, int* value,
                      std::string& error) = 0;
};

// ---------------------------------------------------------------------------
// Codable decision description
// ---------------------------------------------------------------------------

struct BlockTxContext {
  int mi_size = kBlock4x4;
  bool is_inter = false;
  bool fsc_mode = false;
  bool reduced_tx_part_set = false;
};

struct TxSizeContext {
  int mi_size = kBlock4x4;
  int mi_row = 0;
  int mi_col = 0;
  int mi_rows = 1;
  int mi_cols = 1;
  bool is_inter = false;
  bool skip_flag = false;
  bool lossless = false;
  bool fsc_mode = false;
  TxMode tx_mode = TxMode::kLargest;
  bool reduced_tx_part_set = false;
};

// LumaTxSizes / LumaTxMiddle / LumaTxScanOrder (§5.20.6.2), indexed in 4x4
// units.  Only cells inside [0, mi_rows) x [0, mi_cols) are written.
struct LumaTxGrid {
  int mi_rows = 0;
  int mi_cols = 0;
  std::vector<std::uint8_t> tx_size;
  std::vector<std::uint8_t> middle;
  std::vector<std::uint8_t> scan_order;

  void reset(int rows, int cols);
  bool contains(int row, int col) const;
  std::uint8_t at(int row, int col) const;
  std::uint8_t middle_at(int row, int col) const;
  std::uint8_t scan_order_at(int row, int col) const;
  void set(int row, int col, int h4, int w4, int tx_size, int mid = 0,
           int order = 0);
};

// Writes the syntax for one read_tx_partition() call and returns the resolved
// geometry.  `decision` must be one of legal_splits(mi_size).options.
// The returned part origins are relative to (0,0); use the driver functions
// (or tx_partition_geometry() with the real coding-block origin) when the
// block is not at the tile origin.
bool write_tx_partition(SymbolSink& sink, TxPartitionCdfs& cdfs,
                        const BlockTxContext& ctx, TxPartition decision,
                        SplitGeometry* geometry, std::string& error);

// Independent reader for one read_tx_partition() call.
bool read_tx_partition(SymbolSource& source, TxPartitionCdfs& cdfs,
                       const BlockTxContext& ctx, int mi_row, int mi_col,
                       TxPartition* decision, SplitGeometry* geometry,
                       std::string& error);

// Callback returning the partition decision for one 64x64 chunk of a coding
// block.  `chunk_index` is raster order over (heightChunks, widthChunks).
using PartitionDecider = std::function<bool(
    int chunk_index, int chunk_mi_row, int chunk_mi_col,
    TxPartition* out, std::string& error)>;

// Mirrors read_tx_size(allowSelect).  Returns true and fills `grid`; also
// reports whether partition syntax was coded (`partitioned`).
bool write_tx_size(SymbolSink& sink, TxPartitionCdfs& cdfs,
                   const TxSizeContext& ctx, bool allow_select,
                   const PartitionDecider& decide, LumaTxGrid& grid,
                   bool* partitioned, std::string& error);

// Mirrors read_block_tx_size().
bool write_block_tx_size(SymbolSink& sink, TxPartitionCdfs& cdfs,
                         const TxSizeContext& ctx,
                         const PartitionDecider& decide, LumaTxGrid& grid,
                         std::string& error);

// Independent readers for the two drivers.
bool read_tx_size(SymbolSource& source, TxPartitionCdfs& cdfs,
                  const TxSizeContext& ctx, bool allow_select,
                  LumaTxGrid& grid, bool* partitioned, std::string& error);
bool read_block_tx_size(SymbolSource& source, TxPartitionCdfs& cdfs,
                        const TxSizeContext& ctx, LumaTxGrid& grid,
                        std::string& error);

// read_tx_mode() (§5.18.8.1) with CodedLossless equivalent.
TxMode tx_mode_from_syntax(bool coded_lossless, bool tx_mode_select);

// §5.20.6.2/§5.20.6.3 chunk counts.
int tx_chunk_cols(int mi_size);  // Max(1, Block_Width >> 6)
int tx_chunk_rows(int mi_size);  // Max(1, Block_Height >> 6)

}  // namespace av2enc::m2::txpartition
