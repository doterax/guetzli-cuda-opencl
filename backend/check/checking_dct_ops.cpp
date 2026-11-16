#include "backend/check/checking_dct_ops.h"

#include <cstddef>

#include "backend/check/mismatch.h"
#include "core/image/output_image.h"

namespace guetzli {

void CheckingDctOps::compare(const char* what, const OutputImageComponent& reference,
                             const OutputImageComponent& tested) {
  const size_t num_coeffs = static_cast<size_t>(reference.widthInBlocks()) * reference.heightInBlocks() * kDctBlockSize;
  reportMismatches(logger_, what, "coeff", reference.coeffs(), tested.coeffs(), num_coeffs);
  reportMismatches(logger_, what, "pixel", reference.pixels(), tested.pixels(), reference.pixelsSize());
}

void CheckingDctOps::copyFromJpegComponent(OutputImageComponent& c, const JpegComponent& comp, int factor_x,
                                           int factor_y, const int* quant) {
  OutputImageComponent tested = c;
  tested_->copyFromJpegComponent(tested, comp, factor_x, factor_y, quant);
  reference_->copyFromJpegComponent(c, comp, factor_x, factor_y, quant);
  compare("CopyFromJpegComponent", c, tested);
}

void CheckingDctOps::applyGlobalQuantization(OutputImageComponent& c, const int q[kDctBlockSize]) {
  OutputImageComponent tested = c;
  tested_->applyGlobalQuantization(tested, q);
  reference_->applyGlobalQuantization(c, q);
  compare("ApplyGlobalQuantization", c, tested);
}

void CheckingDctOps::toSrgb(const OutputImage& img, int xmin, int ymin, int xsize, int ysize,
                            std::vector<uint8_t>& rgb) {
  reference_->toSrgb(img, xmin, ymin, xsize, ysize, rgb);
  if (xsize * ysize <= kDctBlockSize) return;
  std::vector<uint8_t> tested(rgb.size());
  tested_->toSrgb(img, xmin, ymin, xsize, ysize, tested);
  reportMismatches(logger_, "ToSRGB", "rgb", rgb.data(), tested.data(), rgb.size());
}

}  // namespace guetzli
