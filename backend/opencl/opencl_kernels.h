// The OpenCL device of one run. The butteraugli pipeline is shared with CUDA (GpuKernelsBase); this class provides the
// context (created on first use or by probe() and owned here), one launcher per kernel, and the DCT / block-zeroing
// round trips.
//
// Author: strongtu@tencent.com
//         ianhuang@tencent.com
//         chriskzhou@tencent.com
#ifndef GUETZLI_BACKEND_OPENCL_OPENCL_KERNELS_H_
#define GUETZLI_BACKEND_OPENCL_OPENCL_KERNELS_H_

#ifdef __USE_OPENCL__

#include <cstddef>
#include <cstdint>
#include <memory>

#include "backend/gpu/gpu_kernels_base.h"
#include "backend/gpu/gpu_options.h"
#include "backend/opencl/opencl_context.h"
#include "logging/logger.h"

namespace guetzli {

struct OpenClTypes {
  using Mem = cl_mem;
  using View = OclChannels;
  using Buffer = ClBuffer;
  using Channels = ClChannels;
  using Memory = ClMemory;
};

class OpenClKernels final : public GpuKernelsBase<OpenClKernels, OpenClTypes> {
 public:
  OpenClKernels(const GpuOptions& options, ILogger& logger);
  ~OpenClKernels() override;
  OpenClKernels(const OpenClKernels&) = delete;
  OpenClKernels& operator=(const OpenClKernels&) = delete;
  OpenClKernels(OpenClKernels&&) = delete;
  OpenClKernels& operator=(OpenClKernels&&) = delete;

  // Creates the context now (device selection, kernel compile). Logs the reason and returns false if OpenCL is not
  // usable.
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
  friend class GpuKernelsBase<OpenClKernels, OpenClTypes>;

  OpenClContext& context();

  // Enqueues a w x h (or n) range padded up to whole work-groups; every kernel bounds-checks its global id.
  void enqueueRange2D(cl_kernel kernel, size_t w, size_t h, size_t lx = 16, size_t ly = 16);
  void enqueueRange1D(cl_kernel kernel, size_t n, size_t l = 256);

  // One launcher per kernel (GpuKernelsBase).
  void convolutionScaleKernel(cl_mem scale, int n, cl_mem multipliers, int len, int step, int offset,
                              float border_ratio, int count);
  void convolutionXKernel(cl_mem result, cl_mem inp, unsigned int xsize, unsigned int ysize, cl_mem multipliers,
                          unsigned int len, int xstep, int offset, cl_mem scale);
  void convolutionYKernel(cl_mem result, cl_mem inp, unsigned int xsize, unsigned int ysize, cl_mem multipliers,
                          unsigned int len, int xstep, int offset, cl_mem scale);
  void squareSampleKernel(cl_mem result, cl_mem image, unsigned int xsize, unsigned int ysize, unsigned int xstep,
                          unsigned int ystep);
  void opsinDynamicsKernel(const OclChannels& rgb, int size, const OclChannels& blurred);
  void maskHighIntensityChangeKernel(const OclChannels& xyb0, const OclChannels& xyb1, unsigned int xsize,
                                     unsigned int ysize, const OclChannels& c0, const OclChannels& c1);
  void edgeDetectorKernel(cl_mem result, unsigned int res_xsize, unsigned int res_ysize, const OclChannels& rgb,
                          const OclChannels& rgb2, unsigned int xsize, unsigned int ysize, unsigned int step);
  void edgeDetectorLowFreqKernel(cl_mem block_diff_ac, unsigned int res_xsize, unsigned int res_ysize,
                                 const OclChannels& rgb, const OclChannels& rgb2, unsigned int xsize,
                                 unsigned int ysize, unsigned int step);
  void blockDiffMapKernel(cl_mem block_diff_dc, cl_mem block_diff_ac, unsigned int res_xsize, unsigned int res_ysize,
                          const OclChannels& rgb, const OclChannels& rgb2, unsigned int xsize, unsigned int ysize,
                          unsigned int step);
  void diffPrecomputeKernel(const OclChannels& mask, const OclChannels& xyb0, const OclChannels& xyb1,
                            unsigned int xsize, unsigned int ysize);
  void scaleImageKernel(cl_mem img, unsigned int size, float w);
  void average5x5Kernel(cl_mem img, unsigned int xsize, unsigned int ysize, cl_mem img_org);
  void minSquareValKernel(cl_mem result, unsigned int xsize, unsigned int ysize, cl_mem img, unsigned int square_size,
                          unsigned int offset);
  void doMaskKernel(const OclChannels& mask, const OclChannels& mask_dc, unsigned int xsize, unsigned int ysize,
                    const OclChannels& lut, const OclChannels& lut_dc);
  void combineChannelsKernel(cl_mem result, const OclChannels& mask, const OclChannels& mask_dc, unsigned int xsize,
                             unsigned int ysize, cl_mem block_diff_dc, cl_mem block_diff_ac, cl_mem edge_detector_map,
                             unsigned int res_xsize, unsigned int step, unsigned int work_xsize,
                             unsigned int work_ysize);
  void upsampleSquareRootKernel(cl_mem diffmap_out, cl_mem diffmap, unsigned int xsize, unsigned int ysize, int step,
                                int res_xsize, int res_ysize);
  void removeBorderKernel(cl_mem out, int out_xsize, int out_ysize, cl_mem in, int cls, int cls2);
  void addBorderKernel(cl_mem out, unsigned int xsize, unsigned int ysize, int cls, int cls2, cl_mem in);
  void srgbToLinearKernel(cl_mem srgb, const OclChannels& out, cl_mem lut, int size);
  void componentsToPixelsKernel(cl_mem out, int xmin, int ymin, int xsize, int ysize, int c, cl_mem pixels, int width,
                                int height, int range_x, int range_y);
  void componentsToPixelsEx1Kernel(cl_mem out, int xmin, int ymin, int xsize, int ysize, int c, int width, int height,
                                   int range_x, int range_y);
  void componentsToPixelsEx2Kernel(cl_mem out, int xmin, int ymin, int xsize, int ysize, int c, int width, int height,
                                   int range_y);
  void colorTransformKernel(cl_mem out, int pixel_count);

  GpuOptions options_;
  ILogger& logger_;
  std::unique_ptr<OpenClContext> context_;
};

}  // namespace guetzli

#endif  // __USE_OPENCL__

#endif  // GUETZLI_BACKEND_OPENCL_OPENCL_KERNELS_H_
