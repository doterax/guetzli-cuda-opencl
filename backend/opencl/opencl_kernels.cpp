// OpenCL launchers for the kernels in kernels/guetzli.cl.
//
// Author: strongtu@tencent.com
//         ianhuang@tencent.com
//         chriskzhou@tencent.com
//         stephendeng@tencent.com
#ifdef __USE_OPENCL__

#include "backend/opencl/opencl_kernels.h"

#include <exception>

#include "backend/opencl/kernel_args.h"

namespace guetzli {

namespace {

size_t roundUp(size_t v, size_t m) {
  return (v + m - 1) / m * m;
}

}  // namespace

OpenClKernels::OpenClKernels(const GpuOptions& options, ILogger& logger) : options_(options), logger_(logger) {}

OpenClKernels::~OpenClKernels() = default;

OpenClContext& OpenClKernels::context() {
  // A failed initialization throws and leaves context_ empty.
  if (!context_) {
    context_ = std::make_unique<OpenClContext>(options_, logger_);
  }
  return *context_;
}

bool OpenClKernels::probe() {
  try {
    context();
  } catch (const std::exception& e) {
    logger_.log(LogLevel::Error, "OpenCL not usable: %s\n", e.what());
    return false;
  }
  return true;
}

// Enqueues a w x h range padded up to whole lx x ly work-groups. Every kernel
// bounds-checks its global id, so padding threads just return. (Was: NULL local
// size, which lets drivers pick tiny groups for odd sizes -- a prime width gives
// groups of one -- or an AMD-only divisor search.)
void OpenClKernels::enqueueRange2D(cl_kernel kernel, size_t w, size_t h, size_t lx, size_t ly) {
  if (w == 0 || h == 0) {
    return;
  }
  OpenClContext& ocl = context();
  const size_t max_wg = ocl.maxWorkGroupSize(kernel);
  while (lx * ly > max_wg && ly > 1) {
    ly /= 2;
  }
  while (lx * ly > max_wg && lx > 1) {
    lx /= 2;
  }
  // Short ranges (e.g. a few rows): trade group height for width.
  while (ly > 1 && ly / 2 >= h) {
    ly /= 2;
    lx *= 2;
  }
  size_t global[2] = {roundUp(w, lx), roundUp(h, ly)};
  size_t local[2] = {lx, ly};
  CL_CHECK(clEnqueueNDRangeKernel(ocl.queue(), kernel, 2, nullptr, global, local, 0, nullptr, nullptr));
}

void OpenClKernels::enqueueRange1D(cl_kernel kernel, size_t n, size_t l) {
  if (n == 0) {
    return;
  }
  OpenClContext& ocl = context();
  const size_t max_wg = ocl.maxWorkGroupSize(kernel);
  while (l > max_wg && l > 1) {
    l /= 2;
  }
  size_t global[1] = {roundUp(n, l)};
  size_t local[1] = {l};
  CL_CHECK(clEnqueueNDRangeKernel(ocl.queue(), kernel, 1, nullptr, global, local, 0, nullptr, nullptr));
}

// --- Kernel launchers ---------------------------------------------------------------------------------------------

void OpenClKernels::convolutionScaleKernel(cl_mem scale, int n, cl_mem multipliers, int len, int step, int offset,
                                           float border_ratio, int count) {
  cl_kernel kernel = context().kernel(KERNEL_CONVOLUTIONSCALE);
  setKernelArgs(kernel, &scale, &n, &multipliers, &len, &step, &offset, &border_ratio, &count);
  enqueueRange1D(kernel, count, 64);
}

void OpenClKernels::convolutionXKernel(cl_mem result, cl_mem inp, unsigned int xsize, unsigned int ysize,
                                       cl_mem multipliers, unsigned int len, int xstep, int offset, cl_mem scale) {
  cl_kernel kernel = context().kernel(KERNEL_CONVOLUTIONX);
  setKernelArgs(kernel, &result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &scale);
  enqueueRange2D(kernel, (xsize + xstep - 1) / xstep, ysize);
}

void OpenClKernels::convolutionYKernel(cl_mem result, cl_mem inp, unsigned int xsize, unsigned int ysize,
                                       cl_mem multipliers, unsigned int len, int xstep, int offset, cl_mem scale) {
  cl_kernel kernel = context().kernel(KERNEL_CONVOLUTIONY);
  setKernelArgs(kernel, &result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &scale);
  enqueueRange2D(kernel, (xsize + xstep - 1) / xstep, (ysize + xstep - 1) / xstep);
}

void OpenClKernels::squareSampleKernel(cl_mem result, cl_mem image, unsigned int xsize, unsigned int ysize,
                                       unsigned int xstep, unsigned int ystep) {
  cl_kernel kernel = context().kernel(KERNEL_SQUARESAMPLE);
  setKernelArgs(kernel, &result, &xsize, &ysize, &image, &xstep, &ystep);
  enqueueRange2D(kernel, xsize, ysize);
}

void OpenClKernels::opsinDynamicsKernel(const OclChannels& rgb, int size, const OclChannels& blurred) {
  cl_kernel kernel = context().kernel(KERNEL_OPSINDYNAMICSIMAGE);
  setKernelArgs(kernel, &rgb.r, &rgb.g, &rgb.b, &size, &blurred.r, &blurred.g, &blurred.b);
  enqueueRange1D(kernel, size);
}

void OpenClKernels::maskHighIntensityChangeKernel(const OclChannels& xyb0, const OclChannels& xyb1, unsigned int xsize,
                                                  unsigned int ysize, const OclChannels& c0, const OclChannels& c1) {
  cl_kernel kernel = context().kernel(KERNEL_MASKHIGHINTENSITYCHANGE);
  setKernelArgs(kernel, &xyb0.r, &xyb0.g, &xyb0.b, &xsize, &ysize, &xyb1.r, &xyb1.g, &xyb1.b, &c0.r, &c0.g, &c0.b,
                &c1.r, &c1.g, &c1.b);
  enqueueRange2D(kernel, xsize, ysize);
}

void OpenClKernels::edgeDetectorKernel(cl_mem result, unsigned int res_xsize, unsigned int res_ysize,
                                       const OclChannels& rgb, const OclChannels& rgb2, unsigned int xsize,
                                       unsigned int ysize, unsigned int step) {
  cl_kernel kernel = context().kernel(KERNEL_EDGEDETECTOR);
  setKernelArgs(kernel, &result, &res_xsize, &res_ysize, &rgb.r, &rgb.g, &rgb.b, &rgb2.r, &rgb2.g, &rgb2.b, &xsize,
                &ysize, &step);
  enqueueRange2D(kernel, res_xsize, res_ysize);
}

void OpenClKernels::edgeDetectorLowFreqKernel(cl_mem block_diff_ac, unsigned int res_xsize, unsigned int res_ysize,
                                              const OclChannels& rgb, const OclChannels& rgb2, unsigned int xsize,
                                              unsigned int ysize, unsigned int step) {
  cl_kernel kernel = context().kernel(KERNEL_EDGEDETECTORLOWFREQ);
  setKernelArgs(kernel, &block_diff_ac, &res_xsize, &res_ysize, &rgb.r, &rgb.g, &rgb.b, &rgb2.r, &rgb2.g, &rgb2.b,
                &xsize, &ysize, &step);
  enqueueRange2D(kernel, res_xsize, res_ysize);
}

void OpenClKernels::blockDiffMapKernel(cl_mem block_diff_dc, cl_mem block_diff_ac, unsigned int res_xsize,
                                       unsigned int res_ysize, const OclChannels& rgb, const OclChannels& rgb2,
                                       unsigned int xsize, unsigned int ysize, unsigned int step) {
  cl_kernel kernel = context().kernel(KERNEL_BLOCKDIFFMAP);
  setKernelArgs(kernel, &block_diff_dc, &block_diff_ac, &res_xsize, &res_ysize, &rgb.r, &rgb.g, &rgb.b, &rgb2.r,
                &rgb2.g, &rgb2.b, &xsize, &ysize, &step);
  enqueueRange2D(kernel, res_xsize, res_ysize);
}

void OpenClKernels::diffPrecomputeKernel(const OclChannels& mask, const OclChannels& xyb0, const OclChannels& xyb1,
                                         unsigned int xsize, unsigned int ysize) {
  cl_kernel kernel = context().kernel(KERNEL_DIFFPRECOMPUTE);
  setKernelArgs(kernel, &mask.x, &mask.y, &mask.b, &xsize, &ysize, &xyb0.x, &xyb0.y, &xyb0.b, &xyb1.x, &xyb1.y,
                &xyb1.b);
  enqueueRange2D(kernel, xsize, ysize);
}

void OpenClKernels::scaleImageKernel(cl_mem img, unsigned int size, float w) {
  cl_kernel kernel = context().kernel(KERNEL_SCALEIMAGE);
  setKernelArgs(kernel, &img, &size, &w);
  enqueueRange1D(kernel, size);
}

void OpenClKernels::average5x5Kernel(cl_mem img, unsigned int xsize, unsigned int ysize, cl_mem img_org) {
  cl_kernel kernel = context().kernel(KERNEL_AVERAGE5X5);
  setKernelArgs(kernel, &img, &xsize, &ysize, &img_org);
  enqueueRange2D(kernel, xsize, ysize);
}

void OpenClKernels::minSquareValKernel(cl_mem result, unsigned int xsize, unsigned int ysize, cl_mem img,
                                       unsigned int square_size, unsigned int offset) {
  cl_kernel kernel = context().kernel(KERNEL_MINSQUAREVAL);
  setKernelArgs(kernel, &result, &xsize, &ysize, &img, &square_size, &offset);
  enqueueRange2D(kernel, xsize, ysize);
}

void OpenClKernels::doMaskKernel(const OclChannels& mask, const OclChannels& mask_dc, unsigned int xsize,
                                 unsigned int ysize, const OclChannels& lut, const OclChannels& lut_dc) {
  cl_kernel kernel = context().kernel(KERNEL_DOMASK);
  setKernelArgs(kernel, &mask.r, &mask.g, &mask.b, &xsize, &ysize, &mask_dc.r, &mask_dc.g, &mask_dc.b, &lut.x, &lut.y,
                &lut.b, &lut_dc.x, &lut_dc.y, &lut_dc.b);
  enqueueRange2D(kernel, xsize, ysize);
}

void OpenClKernels::combineChannelsKernel(cl_mem result, const OclChannels& mask, const OclChannels& mask_dc,
                                          unsigned int xsize, unsigned int ysize, cl_mem block_diff_dc,
                                          cl_mem block_diff_ac, cl_mem edge_detector_map, unsigned int res_xsize,
                                          unsigned int step, unsigned int work_xsize, unsigned int work_ysize) {
  cl_kernel kernel = context().kernel(KERNEL_COMBINECHANNELS);
  setKernelArgs(kernel, &result, &mask.r, &mask.g, &mask.b, &mask_dc.r, &mask_dc.g, &mask_dc.b, &xsize, &ysize,
                &block_diff_dc, &block_diff_ac, &edge_detector_map, &res_xsize, &step);
  enqueueRange2D(kernel, work_xsize, work_ysize);
}

void OpenClKernels::upsampleSquareRootKernel(cl_mem diffmap_out, cl_mem diffmap, unsigned int xsize, unsigned int ysize,
                                             int step, int res_xsize, int res_ysize) {
  cl_kernel kernel = context().kernel(KERNEL_UPSAMPLESQUAREROOT);
  setKernelArgs(kernel, &diffmap_out, &diffmap, &xsize, &ysize, &step, &res_xsize, &res_ysize);
  enqueueRange2D(kernel, res_xsize, res_ysize);
}

void OpenClKernels::removeBorderKernel(cl_mem out, int out_xsize, int out_ysize, cl_mem in, int cls, int cls2) {
  cl_kernel kernel = context().kernel(KERNEL_REMOVEBORDER);
  setKernelArgs(kernel, &out, &out_xsize, &out_ysize, &in, &cls, &cls2);
  enqueueRange2D(kernel, out_xsize, out_ysize);
}

void OpenClKernels::addBorderKernel(cl_mem out, unsigned int xsize, unsigned int ysize, int cls, int cls2, cl_mem in) {
  cl_kernel kernel = context().kernel(KERNEL_ADDBORDER);
  setKernelArgs(kernel, &out, &xsize, &ysize, &cls, &cls2, &in);
  enqueueRange2D(kernel, xsize, ysize);
}

void OpenClKernels::srgbToLinearKernel(cl_mem srgb, const OclChannels& out, cl_mem lut, int size) {
  cl_kernel kernel = context().kernel(KERNEL_SRGBTOLINEAR);
  setKernelArgs(kernel, &srgb, &out.r, &out.g, &out.b, &lut, &size);
  enqueueRange1D(kernel, size);
}

void OpenClKernels::componentsToPixelsKernel(cl_mem out, int xmin, int ymin, int xsize, int ysize, int c, cl_mem pixels,
                                             int width, int height, int range_x, int range_y) {
  cl_kernel kernel = context().kernel(KERNEL_COMPONENTSTOPIXELS);
  setKernelArgs(kernel, &out, &xmin, &ymin, &xsize, &ysize, &c, &pixels, &width, &height);
  enqueueRange2D(kernel, range_x, range_y);
}

void OpenClKernels::componentsToPixelsEx1Kernel(cl_mem out, int xmin, int ymin, int xsize, int ysize, int c, int width,
                                                int height, int range_x, int range_y) {
  cl_kernel kernel = context().kernel(KERNEL_COMPONENTSTOPIXELS_EX1);
  setKernelArgs(kernel, &out, &xmin, &ymin, &xsize, &ysize, &c, &width, &height);
  enqueueRange2D(kernel, range_x, range_y);
}

void OpenClKernels::componentsToPixelsEx2Kernel(cl_mem out, int xmin, int ymin, int xsize, int ysize, int c, int width,
                                                int height, int range_y) {
  cl_kernel kernel = context().kernel(KERNEL_COMPONENTSTOPIXELS_EX2);
  setKernelArgs(kernel, &out, &xmin, &ymin, &xsize, &ysize, &c, &width, &height);
  enqueueRange2D(kernel, xsize, range_y);
}

void OpenClKernels::colorTransformKernel(cl_mem out, int pixel_count) {
  cl_kernel kernel = context().kernel(KERNEL_COLORTRANSFORMYCBCRTORGB);
  setKernelArgs(kernel, &out, &pixel_count);
  enqueueRange1D(kernel, pixel_count);
}

// --- DCT and block-zeroing round trips ----------------------------------------------------------------------------

void OpenClKernels::computeBlockZeroingOrder(CoeffData* output_order_batch, const ChannelInfo orig_channel[3],
                                             const float* orig_image_batch, const float* mask_scale,
                                             const int image_width, const int image_height,
                                             const ChannelInfo mayout_channel[3], const int factor, const int comp_mask,
                                             const float block_error_limit) {
  // Validate factor to prevent division by zero
  if (factor <= 0) {
    logger_.log(LogLevel::Info, "Error: factor is %d, must be > 0. Skipping clComputeBlockZeroingOrder.\n", factor);
    return;
  }

  const int block8_width = (image_width + 8 - 1) / 8;
  const int block8_height = (image_height + 8 - 1) / 8;
  const int blockf_width = (image_width + (8 * factor) - 1) / (8 * factor);
  const int blockf_height = (image_height + (8 * factor) - 1) / (8 * factor);

  OpenClContext& ocl = context();

  const size_t orig_image_size = sizeof(float) * 3 * kDctBlockSize * block8_width * block8_height;
  ClBuffer mem_orig_image = ocl.upload(orig_image_size, orig_image_batch);

  const size_t mask_scale_size = sizeof(float) * 3 * block8_width * block8_height;
  ClBuffer mem_mask_scale = ocl.upload(mask_scale_size, mask_scale);

  const size_t output_order_batch_size = sizeof(CoeffData) * 3 * kDctBlockSize * blockf_width * blockf_height;
  // Zero-filled (the kernel writes only each block's valid entries).
  ClBuffer mem_output_order_batch = ocl.zeroed(output_order_batch_size);

  ClBuffer mem_orig_coeff[3];
  ClBuffer mem_mayout_coeff[3];
  ClBuffer mem_mayout_pixel[3];
  for (int c = 0; c < 3; c++) {
    int block_count = orig_channel[c].block_width * orig_channel[c].block_height;
    mem_orig_coeff[c] = ocl.upload(block_count * sizeof(::coeff_t) * kDctBlockSize, orig_channel[c].coeff);

    block_count = mayout_channel[c].block_width * mayout_channel[c].block_height;
    mem_mayout_coeff[c] = ocl.upload(block_count * sizeof(::coeff_t) * kDctBlockSize, mayout_channel[c].coeff);

    mem_mayout_pixel[c] =
        ocl.upload(static_cast<size_t>(image_width) * image_height * sizeof(uint16_t), mayout_channel[c].pixel);
  }

  cl_kernel kernel = ocl.kernel(KERNEL_COMPUTEBLOCKZEROINGORDER);
  setKernelArgs(kernel, &mem_orig_coeff[0].get(), &mem_orig_coeff[1].get(), &mem_orig_coeff[2].get(),
                &mem_orig_image.get(), &mem_mask_scale.get(), &blockf_width, &blockf_height, &image_width,
                &image_height, &mem_mayout_coeff[0].get(), &mem_mayout_coeff[1].get(), &mem_mayout_coeff[2].get(),
                &mem_mayout_pixel[0].get(), &mem_mayout_pixel[1].get(), &mem_mayout_pixel[2].get(),
                &orig_channel[0].block_width, &orig_channel[1].block_width, &orig_channel[2].block_width,
                &mayout_channel[0].factor, &mayout_channel[1].factor, &mayout_channel[2].factor,
                &mayout_channel[0].block_width, &mayout_channel[1].block_width, &mayout_channel[2].block_width,
                &mayout_channel[0].block_height, &mayout_channel[1].block_height, &mayout_channel[2].block_height,
                &factor, &comp_mask, &block_error_limit, &mem_output_order_batch.get());

  // Heavy per-thread state limits occupancy anyway: one-warp groups so small
  // images (few blocks) still spread across all compute units.
  enqueueRange2D(kernel, blockf_width, blockf_height, 8, 4);

  ocl.readAsync(output_order_batch, mem_output_order_batch.get(), output_order_batch_size);
  ocl.finish();
}

void OpenClKernels::copyFromJpegComponent(coeff_t* output_batch /*in,out*/, uint8_t* output_idct /*out*/,
                                          const coeff_t* jpeg_batch /*in*/, const int* quant,
                                          const int jpeg_block_width, const int jpeg_block_height,
                                          const int output_block_width, const int output_block_height,
                                          const int output_width, const int output_height) {
  OpenClContext& ocl = context();

  const int src_block_count = jpeg_block_width * jpeg_block_height;
  ClBuffer src_coeff = ocl.upload(src_block_count * sizeof(::coeff_t) * kDctBlockSize, jpeg_batch);

  const size_t dst_coeff_size = output_block_width * output_block_height * sizeof(::coeff_t) * kDctBlockSize;
  ClBuffer dst_coeff = ocl.upload(dst_coeff_size, output_batch);

  const size_t dst_idct_size = output_block_width * output_block_height * sizeof(uint8_t) * kDctBlockSize;
  ClBuffer dst_idct = ocl.upload(dst_idct_size, output_idct);

  ClBuffer src_quant = ocl.upload(kDctBlockSize * sizeof(int), quant);

  cl_kernel kernel = ocl.kernel(KERNEL_COPYFROMJPEGCOMPONENT);
  setKernelArgs(kernel, &dst_coeff.get(), &dst_idct.get(), &src_coeff.get(), &src_quant.get(), &jpeg_block_width,
                &jpeg_block_height, &output_block_width, &output_block_height, &output_width, &output_height);

  enqueueRange2D(kernel, output_block_width, output_block_height);

  ocl.readAsync(output_batch, dst_coeff.get(), dst_coeff_size);
  ocl.readAsync(output_idct, dst_idct.get(), dst_idct_size);
  ocl.finish();
}

void OpenClKernels::applyGlobalQuantization(coeff_t* output_batch /*in,out*/, uint8_t* output_idct /*out*/,
                                            uint8_t* output_bool /*out*/, const int* q /*in*/, const int block_width,
                                            const int block_height) {
  OpenClContext& ocl = context();

  const size_t dst_coeff_size = block_width * block_height * sizeof(::coeff_t) * kDctBlockSize;
  ClBuffer dst_coeff = ocl.upload(dst_coeff_size, output_batch);

  const size_t dst_idct_size = block_width * block_height * sizeof(uint8_t) * kDctBlockSize;
  ClBuffer dst_idct = ocl.upload(dst_idct_size, output_idct);

  const size_t dst_bool_size = block_width * block_height * sizeof(uchar);
  ClBuffer dst_bool = ocl.upload(dst_bool_size, output_bool);

  ClBuffer src_q = ocl.upload(kDctBlockSize * sizeof(int), q);

  cl_kernel kernel = ocl.kernel(KERNEL_APPLYGLOBALQUANTIZATION);
  setKernelArgs(kernel, &dst_coeff.get(), &dst_idct.get(), &dst_bool.get(), &src_q.get(), &block_width, &block_height);

  enqueueRange2D(kernel, block_width, block_height);

  ocl.readAsync(output_batch, dst_coeff.get(), dst_coeff_size);
  ocl.readAsync(output_idct, dst_idct.get(), dst_idct_size);
  ocl.readAsync(output_bool, dst_bool.get(), dst_bool_size);
  ocl.finish();
}

}  // namespace guetzli

#endif  // __USE_OPENCL__
