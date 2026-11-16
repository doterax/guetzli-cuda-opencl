#include "backend/gpu/gpu_dct_ops.h"

#include <cassert>

namespace guetzli {

void GpuDctOps::copyFromJpegComponent(OutputImageComponent& c, const JpegComponent& comp, int factor_x, int factor_y,
                                      const int* quant) {
  c.reset(factor_x, factor_y);
  assert(c.width_in_blocks() <= comp.width_in_blocks);
  assert(c.height_in_blocks() <= comp.height_in_blocks);

  const int bw = c.widthInBlocks();
  const int bh = c.heightInBlocks();
  std::vector<uint8_t> idct(static_cast<size_t>(bw) * bh * kDctBlockSize);
  kernels_->copyFromJpegComponent(c.mutableCoeffs(), idct.data(), comp.coeffs.data(), quant, comp.width_in_blocks,
                                  comp.height_in_blocks, bw, bh, c.width(), c.height());
  for (int block_y = 0; block_y < bh; ++block_y) {
    for (int block_x = 0; block_x < bw; ++block_x) {
      const int offset = (block_y * bw + block_x) * kDctBlockSize;
      c.updatePixelsForBlock(block_x, block_y, &idct[offset]);
    }
  }
  c.setQuant(quant);
}

void GpuDctOps::applyGlobalQuantization(OutputImageComponent& c, const int q[kDctBlockSize]) {
  const int bw = c.widthInBlocks();
  const int bh = c.heightInBlocks();
  std::vector<uint8_t> idct(static_cast<size_t>(bw) * bh * kDctBlockSize);
  std::vector<uint8_t> changed(static_cast<size_t>(bw) * bh);
  kernels_->applyGlobalQuantization(c.mutableCoeffs(), idct.data(), changed.data(), q, bw, bh);
  for (int block_y = 0; block_y < bh; ++block_y) {
    for (int block_x = 0; block_x < bw; ++block_x) {
      const int block_ix = block_y * bw + block_x;
      if (changed[block_ix] != 0u) {
        c.updatePixelsForBlock(block_x, block_y, &idct[block_ix * kDctBlockSize]);
      }
    }
  }
  c.setQuant(q);
}

void GpuDctOps::toSrgb(const OutputImage& img, int xmin, int ymin, int xsize, int ysize, std::vector<uint8_t>& rgb) {
  kernels_->componentsToPixels(rgb.data(), xmin, ymin, xsize, ysize, img.components());
}

}  // namespace guetzli
