// --check for the block-zeroing search: runs the tested (GPU) and the reference (CPU) search, reports
// differing entries as a "CHK" error and returns the reference result.
#ifndef GUETZLI_BACKEND_CHECK_CHECKING_BLOCK_ZEROING_H_
#define GUETZLI_BACKEND_CHECK_CHECKING_BLOCK_ZEROING_H_

#include <memory>
#include <vector>

#include "core/encoder/block_zeroing.h"
#include "logging/logger.h"

namespace guetzli {

class CheckingBlockZeroingSearch final : public IBlockZeroingSearch {
 public:
  CheckingBlockZeroingSearch(std::unique_ptr<IBlockZeroingSearch> tested,
                             std::unique_ptr<IBlockZeroingSearch> reference, ILogger& logger)
      : tested_(std::move(tested)), reference_(std::move(reference)), logger_(logger) {}

  std::vector<CoeffData> run(const BlockZeroingInput& in) override;

 private:
  std::unique_ptr<IBlockZeroingSearch> tested_;
  std::unique_ptr<IBlockZeroingSearch> reference_;
  ILogger& logger_;
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_CHECK_CHECKING_BLOCK_ZEROING_H_
