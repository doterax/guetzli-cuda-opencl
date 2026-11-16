// IDctOps on a GPU. OpenCL and CUDA differ only in the kernel launchers (GpuDctKernels).
#ifndef GUETZLI_BACKEND_GPU_GPU_DCT_OPS_H_
#define GUETZLI_BACKEND_GPU_GPU_DCT_OPS_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "backend/gpu/gpu_kernels.h"
#include "core/image/dct_ops.h"
#include "core/image/output_image.h"

namespace guetzli {

class GpuDctOps final : public IDctOps {
 public:
  explicit GpuDctOps(std::shared_ptr<GpuDctKernels> kernels) : kernels_(std::move(kernels)) {}

  void copyFromJpegComponent(OutputImageComponent& c, const JPEGComponent& comp, int factor_x, int factor_y,
                             const int* quant) override;
  void applyGlobalQuantization(OutputImageComponent& c, const int q[kDCTBlockSize]) override;
  void toSrgb(const OutputImage& img, int xmin, int ymin, int xsize, int ysize, std::vector<uint8_t>& rgb) override;

 private:
  std::shared_ptr<GpuDctKernels> kernels_;
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_GPU_GPU_DCT_OPS_H_
