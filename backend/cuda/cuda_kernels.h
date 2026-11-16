// The CUDA device of one run. The butteraugli pipeline is shared with OpenCL (GpuKernelsBase); this class provides the
// context (the driver is loaded and the context created on first use or by probe(), and owned here), one launcher
// per kernel, and the DCT / block-zeroing round trips.
//
// Author: strongtu@tencent.com
//         ianhuang@tencent.com
//         chriskzhou@tencent.com
#ifndef GUETZLI_BACKEND_CUDA_CUDA_KERNELS_H_
#define GUETZLI_BACKEND_CUDA_CUDA_KERNELS_H_

#ifdef __USE_CUDA__

#include <cstdint>
#include <memory>

#include "backend/cuda/cuda_context.h"
#include "backend/cuda/cuda_driver.h"
#include "backend/gpu/gpu_kernels_base.h"
#include "backend/gpu/gpu_options.h"
#include "logging/logger.h"

namespace guetzli {

struct CudaTypes {
  using Mem = cu_mem;
  using View = OcuChannels;
  using Buffer = CuBuffer;
  using Channels = CuChannels;
  using Memory = CuMemory;
};

class CudaKernels final : public GpuKernelsBase<CudaKernels, CudaTypes> {
 public:
  CudaKernels(const GpuOptions& options, ILogger& logger);
  ~CudaKernels() override;
  CudaKernels(const CudaKernels&) = delete;
  CudaKernels& operator=(const CudaKernels&) = delete;
  CudaKernels(CudaKernels&&) = delete;
  CudaKernels& operator=(CudaKernels&&) = delete;

  // Loads the driver and creates the context now. Returns false if CUDA is not usable (logged, unless the driver is
  // simply not installed).
  [[nodiscard]] bool probe();

  void copyFromJpegComponent(coeff_t* output_batch, uint8_t* output_idct, const coeff_t* jpeg_batch, const int* quant,
                             int jpeg_block_width, int jpeg_block_height, int output_block_width,
                             int output_block_height, int output_width, int output_height) override;
  void applyGlobalQuantization(coeff_t* output_batch, uint8_t* output_idct, uint8_t* output_bool, const int* q,
                               int block_width, int block_height) override;
  void computeBlockZeroingOrder(CoeffData* output_order_batch, const ChannelInfo orig_channel[3],
                                const float* orig_image_batch, const float* mask_scale, int image_width,
                                int image_height, const ChannelInfo mayout_channel[3], int factor, int comp_mask,
                                float block_error_limit) override;

 private:
  friend class GpuKernelsBase<CudaKernels, CudaTypes>;

  CudaContext& context();
  void createContext(const CudaDriver& driver);

  // One launcher per kernel (GpuKernelsBase).
  void convolutionScaleKernel(cu_mem scale, int n, cu_mem multipliers, int len, int step, int offset,
                              float border_ratio, int count);
  void convolutionXKernel(cu_mem result, cu_mem inp, unsigned int xsize, unsigned int ysize, cu_mem multipliers,
                          unsigned int len, int xstep, int offset, cu_mem scale);
  void convolutionYKernel(cu_mem result, cu_mem inp, unsigned int xsize, unsigned int ysize, cu_mem multipliers,
                          unsigned int len, int xstep, int offset, cu_mem scale);
  void squareSampleKernel(cu_mem result, cu_mem image, unsigned int xsize, unsigned int ysize, unsigned int xstep,
                          unsigned int ystep);
  void opsinDynamicsKernel(const OcuChannels& rgb, int size, const OcuChannels& blurred);
  void maskHighIntensityChangeKernel(const OcuChannels& xyb0, const OcuChannels& xyb1, unsigned int xsize,
                                     unsigned int ysize, const OcuChannels& c0, const OcuChannels& c1);
  void edgeDetectorKernel(cu_mem result, unsigned int res_xsize, unsigned int res_ysize, const OcuChannels& rgb,
                          const OcuChannels& rgb2, unsigned int xsize, unsigned int ysize, unsigned int step);
  void edgeDetectorLowFreqKernel(cu_mem block_diff_ac, unsigned int res_xsize, unsigned int res_ysize,
                                 const OcuChannels& rgb, const OcuChannels& rgb2, unsigned int xsize,
                                 unsigned int ysize, unsigned int step);
  void blockDiffMapKernel(cu_mem block_diff_dc, cu_mem block_diff_ac, unsigned int res_xsize, unsigned int res_ysize,
                          const OcuChannels& rgb, const OcuChannels& rgb2, unsigned int xsize, unsigned int ysize,
                          unsigned int step);
  void diffPrecomputeKernel(const OcuChannels& mask, const OcuChannels& xyb0, const OcuChannels& xyb1,
                            unsigned int xsize, unsigned int ysize);
  void scaleImageKernel(cu_mem img, unsigned int size, float w);
  void average5x5Kernel(cu_mem img, unsigned int xsize, unsigned int ysize, cu_mem img_org);
  void minSquareValKernel(cu_mem result, unsigned int xsize, unsigned int ysize, cu_mem img, unsigned int square_size,
                          unsigned int offset);
  void doMaskKernel(const OcuChannels& mask, const OcuChannels& mask_dc, unsigned int xsize, unsigned int ysize,
                    const OcuChannels& lut, const OcuChannels& lut_dc);
  void combineChannelsKernel(cu_mem result, const OcuChannels& mask, const OcuChannels& mask_dc, unsigned int xsize,
                             unsigned int ysize, cu_mem block_diff_dc, cu_mem block_diff_ac, cu_mem edge_detector_map,
                             unsigned int res_xsize, unsigned int step, unsigned int work_xsize,
                             unsigned int work_ysize);
  void upsampleSquareRootKernel(cu_mem diffmap_out, cu_mem diffmap, unsigned int xsize, unsigned int ysize, int step,
                                int res_xsize, int res_ysize);
  void removeBorderKernel(cu_mem out, int out_xsize, int out_ysize, cu_mem in, int cls, int cls2);
  void addBorderKernel(cu_mem out, unsigned int xsize, unsigned int ysize, int cls, int cls2, cu_mem in);
  void srgbToLinearKernel(cu_mem srgb, const OcuChannels& out, cu_mem lut, int size);
  void componentsToPixelsKernel(cu_mem out, int xmin, int ymin, int xsize, int ysize, int c, cu_mem pixels, int width,
                                int height, int range_x, int range_y);
  void componentsToPixelsEx1Kernel(cu_mem out, int xmin, int ymin, int xsize, int ysize, int c, int width, int height,
                                   int range_x, int range_y);
  void componentsToPixelsEx2Kernel(cu_mem out, int xmin, int ymin, int xsize, int ysize, int c, int width, int height,
                                   int range_y);
  void colorTransformKernel(cu_mem out, int pixel_count);
  void blockMax8x8Kernel(cu_mem distmap, int xsize, int ysize, cu_mem block_max, int width8, int height8);

  GpuOptions options_;
  ILogger& logger_;
  std::unique_ptr<CudaContext> context_;
};

}  // namespace guetzli

#endif  // __USE_CUDA__

#endif  // GUETZLI_BACKEND_CUDA_CUDA_KERNELS_H_
