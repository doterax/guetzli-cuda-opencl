// IBlockZeroingSearch on a GPU (one kernel launch for all blocks). Needs the comparator's
// blockReferenceData(), i.e. a GpuButteraugliComparator (directly or inside a CheckingComparator).
#ifndef GUETZLI_BACKEND_GPU_GPU_BLOCK_ZEROING_H_
#define GUETZLI_BACKEND_GPU_GPU_BLOCK_ZEROING_H_

#include <vector>

#include "clguetzli/clguetzli.cl.h"
#include "guetzli/block_zeroing.h"

namespace guetzli {

// clComputeBlockZeroingOrder / cuComputeBlockZeroingOrder.
using GpuBlockZeroingKernel = void (*)(CoeffData* output_order_batch, const channel_info orig_channel[3],
                                       const float* orig_image_batch, const float* mask_scale, int image_width,
                                       int image_height, const channel_info mayout_channel[3], int factor,
                                       int comp_mask, float block_error_limit);

class GpuBlockZeroingSearch final : public IBlockZeroingSearch {
 public:
  explicit GpuBlockZeroingSearch(GpuBlockZeroingKernel kernel) : kernel_(kernel) {}

  std::vector<CoeffData> run(const BlockZeroingInput& in) override;

 private:
  GpuBlockZeroingKernel kernel_;
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_GPU_GPU_BLOCK_ZEROING_H_
