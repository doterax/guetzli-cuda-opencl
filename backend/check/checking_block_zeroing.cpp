#include "backend/check/checking_block_zeroing.h"

#include <cmath>
#include <cstddef>

namespace guetzli {

std::vector<CoeffData> CheckingBlockZeroingSearch::run(const BlockZeroingInput& in) {
  const std::vector<CoeffData> tested = tested_->run(in);
  std::vector<CoeffData> reference = reference_->run(in);
  if (tested.size() != reference.size()) {
    logger_.log(LogLevel::Error, "CHK BlockZeroing size %zu:%zu\n", tested.size(), reference.size());
    return reference;
  }
  size_t count = 0;
  for (size_t i = 0; i < reference.size(); ++i) {
    if (tested[i].idx != reference[i].idx || std::fabs(tested[i].block_err - reference[i].block_err) > 0.001) {
      ++count;
    }
  }
  if (count > 0) logger_.log(LogLevel::Error, "CHK BlockZeroing order %zu:%zu\n", count, reference.size());
  return reference;
}

}  // namespace guetzli
