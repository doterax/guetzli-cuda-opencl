// Per-block search for the order in which to zero coefficients (docs/design/modernization.md §5.3).
#ifndef GUETZLI_BLOCK_ZEROING_H_
#define GUETZLI_BLOCK_ZEROING_H_

#include <cstdint>
#include <vector>

#include "guetzli/comparator.h"
#include "guetzli/jpeg_data.h"
#include "guetzli/output_image.h"

namespace guetzli {

// Number of coefficients of a block across the three components.
constexpr int kBlockZeroingBlockSize = 3 * kDCTBlockSize;

struct CoeffData {
  int idx;
  float block_err;
};

struct BlockZeroingInput {
  const JPEGData& jpg;     // the original coefficients (quantization removed)
  OutputImage& img;        // the current image; unchanged when run() returns
  uint8_t comp_mask;       // components to search; all have the subsampling of the last one
  Comparator& comparator;  // StartBlockComparisons() has been called
  int greedy_lookahead;    // Params::zeroing_greedy_lookahead
  bool new_zeroing_model;  // Params::new_zeroing_model
};

class IBlockZeroingSearch {
 public:
  virtual ~IBlockZeroingSearch() = default;

  // For each block (row-major, at the subsampling of the components in comp_mask), kBlockZeroingBlockSize
  // entries: the coefficient indexes to zero in order, each with the block error after zeroing it and all
  // previous ones. Only entries with 0 < block_err <= comparator.BlockErrorLimit() are meaningful.
  virtual std::vector<CoeffData> run(const BlockZeroingInput& in) = 0;
};

// The CPU search: greedy, scoring candidates with Comparator::CompareBlock(). A block's search stops at
// the first step whose error reaches the block error limit.
class CpuBlockZeroingSearch final : public IBlockZeroingSearch {
 public:
  std::vector<CoeffData> run(const BlockZeroingInput& in) override;
};

}  // namespace guetzli

#endif  // GUETZLI_BLOCK_ZEROING_H_
