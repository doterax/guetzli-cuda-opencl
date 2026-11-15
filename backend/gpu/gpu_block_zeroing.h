// IBlockZeroingSearch on a GPU (one kernel launch for all blocks). Needs the comparator's
// blockReferenceData(), i.e. a GpuButteraugliComparator (directly or inside a CheckingComparator).
#ifndef GUETZLI_BACKEND_GPU_GPU_BLOCK_ZEROING_H_
#define GUETZLI_BACKEND_GPU_GPU_BLOCK_ZEROING_H_

#include <memory>
#include <utility>
#include <vector>

#include "backend/gpu/gpu_kernels.h"
#include "guetzli/block_zeroing.h"

namespace guetzli {

class GpuBlockZeroingSearch final : public IBlockZeroingSearch {
 public:
  explicit GpuBlockZeroingSearch(std::shared_ptr<GpuBlockZeroingKernel> kernel) : kernel_(std::move(kernel)) {}

  std::vector<CoeffData> run(const BlockZeroingInput& in) override;

 private:
  std::shared_ptr<GpuBlockZeroingKernel> kernel_;
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_GPU_GPU_BLOCK_ZEROING_H_
