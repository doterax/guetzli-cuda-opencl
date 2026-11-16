#include "backend/gpu/gpu_block_zeroing.h"

#include <optional>
#include <stdexcept>

#include "core/util/fast_log.h"

namespace guetzli {

std::vector<CoeffData> GpuBlockZeroingSearch::run(const BlockZeroingInput& in) {
  const std::optional<BlockReferenceData> reference = in.comparator.blockReferenceData();
  if (!reference) {
    throw std::logic_error("GPU block zeroing needs a GPU comparator");
  }

  const OutputImage& img = in.img;
  const int width = img.width();
  const int height = img.height();
  const int last_c = log2FloorNonZero(in.comp_mask);
  const int factor_x = img.component(last_c).factorX();
  const int factor_y = img.component(last_c).factorY();
  const int block_width = (width + 8 * factor_x - 1) / (8 * factor_x);
  const int block_height = (height + 8 * factor_y - 1) / (8 * factor_y);
  const int num_blocks = block_width * block_height;

  ChannelInfo orig_channel[3];
  ChannelInfo mayout_channel[3];
  for (int c = 0; c < 3; c++) {
    mayout_channel[c].factor = img.component(c).factorX();
    mayout_channel[c].block_width = img.component(c).widthInBlocks();
    mayout_channel[c].block_height = img.component(c).heightInBlocks();
    mayout_channel[c].coeff = img.component(c).coeffs();
    mayout_channel[c].pixel = img.component(c).pixels();

    orig_channel[c].factor = in.jpg.components[c].v_samp_factor;
    orig_channel[c].block_width = in.jpg.components[c].width_in_blocks;
    orig_channel[c].block_height = in.jpg.components[c].height_in_blocks;
    orig_channel[c].coeff = in.jpg.components[c].coeffs.data();
  }

  std::vector<CoeffData> output_order(static_cast<size_t>(num_blocks) * kBlockZeroingBlockSize);
  kernel_->computeBlockZeroingOrder(output_order.data(), orig_channel, reference->opsin.data(),
                                    reference->mask_scale.data(), width, height, mayout_channel, factor_x, in.comp_mask,
                                    in.comparator.blockErrorLimit());
  return output_order;
}

}  // namespace guetzli
