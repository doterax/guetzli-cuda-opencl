// Coefficient/pixel operations on an OutputImage that a backend may run on a device
// (docs/design/modernization.md §5.3). OutputImage calls the instance it was created with
// and knows nothing about backends.
#ifndef GUETZLI_DCT_OPS_H_
#define GUETZLI_DCT_OPS_H_

#include <cstdint>
#include <vector>

#include "guetzli/jpeg_data.h"

namespace guetzli {

class OutputImage;
class OutputImageComponent;

class IDctOps {
 public:
  virtual ~IDctOps() = default;

  // Resets c to the given subsampling and loads the dequantized coefficients of comp
  // (which must not be downsampled) and their pixels. quant becomes c.quant().
  virtual void copyFromJpegComponent(OutputImageComponent& c, const JPEGComponent& comp, int factor_x, int factor_y,
                                     const int* quant) = 0;

  // Quantizes every block of c with q and updates the pixels of the changed blocks. q becomes c.quant().
  virtual void applyGlobalQuantization(OutputImageComponent& c, const int q[kDCTBlockSize]) = 0;

  // 8-bit sRGB view of the window (see OutputImage::ToSRGB); rgb has 3 * xsize * ysize bytes.
  virtual void toSrgb(const OutputImage& img, int xmin, int ymin, int xsize, int ysize, std::vector<uint8_t>& rgb) = 0;
};

// The CPU implementation; also the reference the GPU implementations are checked against.
class CpuDctOps final : public IDctOps {
 public:
  void copyFromJpegComponent(OutputImageComponent& c, const JPEGComponent& comp, int factor_x, int factor_y,
                             const int* quant) override;
  void applyGlobalQuantization(OutputImageComponent& c, const int q[kDCTBlockSize]) override;
  void toSrgb(const OutputImage& img, int xmin, int ymin, int xsize, int ysize, std::vector<uint8_t>& rgb) override;
};

}  // namespace guetzli

#endif  // GUETZLI_DCT_OPS_H_
