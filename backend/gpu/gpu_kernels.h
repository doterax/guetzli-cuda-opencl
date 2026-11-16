// What a GPU backend (backend/opencl, backend/cuda) provides to the shared GPU code in backend/gpu: typed launchers
// for each pipeline stage. Implementations create their device context on first use, so an image that never needs
// the device (too small) never initializes it. Failures throw GpuError.
#ifndef GUETZLI_BACKEND_GPU_GPU_KERNELS_H_
#define GUETZLI_BACKEND_GPU_GPU_KERNELS_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "backend/gpu/kernel_types.h"
#include "core/encoder/block_zeroing.h"
#include "core/image/output_image.h"

namespace guetzli {

// Device state of one comparator: the original's opsin-dynamics image stays resident between compare() calls.
class GpuCompareSession {
 public:
  virtual ~GpuCompareSession() = default;
  // Distance map of img against the original, xsize * ysize values (out).
  virtual void compare(const OutputImage& img, int step, float* distmap) = 0;
};

class GpuButteraugliKernels {
 public:
  virtual ~GpuButteraugliKernels() = default;
  // In place, three planes of xsize * ysize values.
  virtual void opsinDynamicsImage(float* r, float* g, float* b, unsigned int xsize, unsigned int ysize) = 0;
  virtual void mask(float* mask_r, float* mask_g, float* mask_b, float* maskdc_r, float* maskdc_g, float* maskdc_b,
                    unsigned int xsize, unsigned int ysize, const float* r, const float* g, const float* b,
                    const float* r2, const float* g2, const float* b2) = 0;
  // r, g, b: the original's opsin-dynamics image (xsize * ysize values each), uploaded once.
  virtual std::unique_ptr<GpuCompareSession> startCompare(const float* r, const float* g, const float* b, int xsize,
                                                          int ysize) = 0;
};

class GpuDctKernels {
 public:
  virtual ~GpuDctKernels() = default;
  virtual void copyFromJpegComponent(coeff_t* output_batch, uint8_t* output_idct, const coeff_t* jpeg_batch,
                                     const int* quant, int jpeg_block_width, int jpeg_block_height,
                                     int output_block_width, int output_block_height, int output_width,
                                     int output_height) = 0;
  virtual void applyGlobalQuantization(coeff_t* output_batch, uint8_t* output_idct, uint8_t* output_bool, const int* q,
                                       int block_width, int block_height) = 0;
  virtual void componentsToPixels(uint8_t* rgb, int xmin, int ymin, int xsize, int ysize,
                                  const std::vector<OutputImageComponent>& components) = 0;
};

class GpuBlockZeroingKernel {
 public:
  virtual ~GpuBlockZeroingKernel() = default;
  virtual void computeBlockZeroingOrder(CoeffData* output_order_batch, const channel_info orig_channel[3],
                                        const float* orig_image_batch, const float* mask_scale, int image_width,
                                        int image_height, const channel_info mayout_channel[3], int factor,
                                        int comp_mask, float block_error_limit) = 0;
};

// A GPU device: all launchers.
class GpuKernels : public GpuButteraugliKernels, public GpuDctKernels, public GpuBlockZeroingKernel {};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_GPU_GPU_KERNELS_H_
