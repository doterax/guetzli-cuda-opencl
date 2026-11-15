// IDctOps on a GPU. OpenCL and CUDA differ only in the kernel launchers, which are passed in.
#ifndef GUETZLI_BACKEND_GPU_GPU_DCT_OPS_H_
#define GUETZLI_BACKEND_GPU_GPU_DCT_OPS_H_

#include <cstdint>
#include <vector>

#include "guetzli/dct_ops.h"
#include "guetzli/output_image.h"

namespace guetzli {

// The cl*/cu* host launchers (clguetzli.h, cuguetzli.h).
struct GpuDctKernels {
  void (*copy_from_jpeg_component)(coeff_t* output_batch, uint8_t* output_idct, const coeff_t* jpeg_batch,
                                   const int* quant, int jpeg_block_width, int jpeg_block_height,
                                   int output_block_width, int output_block_height, int output_width,
                                   int output_height);
  void (*apply_global_quantization)(coeff_t* output_batch, uint8_t* output_idct, uint8_t* output_bool, const int* q,
                                    int block_width, int block_height);
  void (*components_to_pixels)(uint8_t* rgb, int xmin, int ymin, int xsize, int ysize,
                               const std::vector<OutputImageComponent>& components);
};

class GpuDctOps final : public IDctOps {
 public:
  explicit GpuDctOps(const GpuDctKernels& kernels) : kernels_(kernels) {}

  void copyFromJpegComponent(OutputImageComponent& c, const JPEGComponent& comp, int factor_x, int factor_y,
                             const int* quant) override;
  void applyGlobalQuantization(OutputImageComponent& c, const int q[kDCTBlockSize]) override;
  void toSrgb(const OutputImage& img, int xmin, int ymin, int xsize, int ysize, std::vector<uint8_t>& rgb) override;

 private:
  GpuDctKernels kernels_;
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_GPU_GPU_DCT_OPS_H_
