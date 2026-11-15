// OpenCL launchers for the kernels in clguetzli.cl.
//
// Author: strongtu@tencent.com
//         ianhuang@tencent.com
//         chriskzhou@tencent.com
//         stephendeng@tencent.com
#ifdef __USE_OPENCL__

#include "backend/opencl/opencl_kernels.h"

#include <math.h>
#include <string.h>

#include <algorithm>
#include <exception>
#include <utility>
#include <vector>

#include "backend/opencl/kernel_args.h"
#include "guetzli/gamma_correct.h"

namespace guetzli {

namespace {

size_t roundUp(size_t v, size_t m) {
  return (v + m - 1) / m * m;
}

void MakeMask(float extmul, float extoff, float mul, float offset, float scaler, float* result) {
  for (size_t i = 0; i < 512; ++i) {
    const float c = mul / ((0.01 * scaler * i) + offset);
    result[i] = 1.0 + extmul * (c + extoff);
    result[i] *= result[i];
  }
}

const float kInternalGoodQualityThreshold = 14.921561160295326;
const float kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

}  // namespace

// Per-comparator state: the original's opsin-dynamics image stays resident on the device.
class OpenClKernels::CompareSession final : public GpuCompareSession {
 public:
  CompareSession(OpenClKernels& kernels, ClChannels ref, int xsize, int ysize)
      : kernels_(kernels), ref_(std::move(ref)), xsize_(xsize), ysize_(ysize) {}

  void compare(const OutputImage& img, int step, float* distmap) override {
    kernels_.compare(ref_.view(), xsize_, ysize_, img, step, distmap);
  }

 private:
  OpenClKernels& kernels_;
  ClChannels ref_;  // never modified
  int xsize_;
  int ysize_;
};

OpenClKernels::OpenClKernels(const GpuOptions& options, ILogger& logger) : options_(options), logger_(logger) {}

OpenClKernels::~OpenClKernels() = default;

OpenClContext& OpenClKernels::context() {
  // A failed initialization throws and leaves context_ empty.
  if (!context_) context_ = std::make_unique<OpenClContext>(options_, logger_);
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
  if (w == 0 || h == 0) return;
  OpenClContext& ocl = context();
  const size_t max_wg = ocl.maxWorkGroupSize(kernel);
  while (lx * ly > max_wg && ly > 1) ly /= 2;
  while (lx * ly > max_wg && lx > 1) lx /= 2;
  // Short ranges (e.g. a few rows): trade group height for width.
  while (ly > 1 && ly / 2 >= h) {
    ly /= 2;
    lx *= 2;
  }
  size_t global[2] = {roundUp(w, lx), roundUp(h, ly)};
  size_t local[2] = {lx, ly};
  CL_CHECK(clEnqueueNDRangeKernel(ocl.queue(), kernel, 2, NULL, global, local, 0, NULL, NULL));
}

void OpenClKernels::enqueueRange1D(cl_kernel kernel, size_t n, size_t l) {
  if (n == 0) return;
  OpenClContext& ocl = context();
  const size_t max_wg = ocl.maxWorkGroupSize(kernel);
  while (l > max_wg && l > 1) l /= 2;
  size_t global[1] = {roundUp(n, l)};
  size_t local[1] = {l};
  CL_CHECK(clEnqueueNDRangeKernel(ocl.queue(), kernel, 1, NULL, global, local, 0, NULL, NULL));
}

void OpenClKernels::opsinDynamicsImage(float* r, float* g, float* b, const unsigned int xsize,
                                       const unsigned int ysize) {
  size_t elements = xsize * ysize;
  size_t channel_size = elements * sizeof(float);

  OpenClContext& ocl = context();

  ClChannels rgb = ocl.uploadChannels(channel_size, r, g, b);

  opsinDynamicsOnDevice(rgb.view(), xsize, ysize);

  CL_CHECK(clEnqueueReadBuffer(ocl.queue(), rgb.ch[0].get(), false, 0, channel_size, r, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(ocl.queue(), rgb.ch[1].get(), false, 0, channel_size, g, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(ocl.queue(), rgb.ch[2].get(), false, 0, channel_size, b, 0, NULL, NULL));
  CL_CHECK(clFinish(ocl.queue()));
}

void OpenClKernels::computeBlockZeroingOrder(CoeffData* output_order_batch, const channel_info orig_channel[3],
                                             const float* orig_image_batch, const float* mask_scale,
                                             const int image_width, const int image_height,
                                             const channel_info mayout_channel[3], const int factor,
                                             const int comp_mask, const float BlockErrorLimit) {
  // Validate factor to prevent division by zero
  if (factor <= 0) {
    logger_.log(LogLevel::Info, "Error: factor is %d, must be > 0. Skipping clComputeBlockZeroingOrder.\n", factor);
    return;
  }

  const int block8_width = (image_width + 8 - 1) / 8;
  const int block8_height = (image_height + 8 - 1) / 8;
  const int blockf_width = (image_width + 8 * factor - 1) / (8 * factor);
  const int blockf_height = (image_height + 8 * factor - 1) / (8 * factor);

  OpenClContext& ocl = context();

  size_t orig_image_size = sizeof(float) * 3 * kDCTBlockSize * block8_width * block8_height;
  ClBuffer mem_orig_image = ocl.upload(orig_image_size, orig_image_batch);

  size_t mask_scale_size = sizeof(float) * 3 * block8_width * block8_height;
  ClBuffer mem_mask_scale = ocl.upload(mask_scale_size, mask_scale);

  size_t output_order_batch_size = sizeof(CoeffData) * 3 * kDCTBlockSize * blockf_width * blockf_height;
  // Zero-filled (the kernel writes only each block's valid entries).
  ClBuffer mem_output_order_batch = ocl.zeroed(output_order_batch_size);

  ClBuffer mem_orig_coeff[3];
  ClBuffer mem_mayout_coeff[3];
  ClBuffer mem_mayout_pixel[3];
  for (int c = 0; c < 3; c++) {
    int block_count = orig_channel[c].block_width * orig_channel[c].block_height;
    size_t orig_coeff_size = block_count * sizeof(::coeff_t) * kDCTBlockSize;
    mem_orig_coeff[c] = ocl.upload(orig_coeff_size, orig_channel[c].coeff);

    block_count = mayout_channel[c].block_width * mayout_channel[c].block_height;
    size_t mayout_coeff_size = block_count * sizeof(::coeff_t) * kDCTBlockSize;
    mem_mayout_coeff[c] = ocl.upload(mayout_coeff_size, mayout_channel[c].coeff);

    size_t mayout_pixel_size = image_width * image_height * sizeof(uint16_t);
    mem_mayout_pixel[c] = ocl.upload(mayout_pixel_size, mayout_channel[c].pixel);
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
                &factor, &comp_mask, &BlockErrorLimit, &mem_output_order_batch.get());

  // Heavy per-thread state limits occupancy anyway: one-warp groups so small
  // images (few blocks) still spread across all compute units.
  enqueueRange2D(kernel, blockf_width, blockf_height, 8, 4);

  CL_CHECK(clEnqueueReadBuffer(ocl.queue(), mem_output_order_batch.get(), false, 0, output_order_batch_size,
                               output_order_batch, 0, NULL, NULL));
  CL_CHECK(clFinish(ocl.queue()));
}

void OpenClKernels::mask(float* mask_r, float* mask_g, float* mask_b, float* maskdc_r, float* maskdc_g, float* maskdc_b,
                         const unsigned int xsize, const unsigned int ysize, const float* r, const float* g,
                         const float* b, const float* r2, const float* g2, const float* b2) {
  OpenClContext& ocl = context();

  size_t channel_size = xsize * ysize * sizeof(float);

  ClChannels rgb = ocl.uploadChannels(channel_size, r, g, b);
  ClChannels rgb2 = ocl.uploadChannels(channel_size, r2, g2, b2);
  ClChannels mask = ocl.zeroedChannels(channel_size);
  ClChannels mask_dc = ocl.zeroedChannels(channel_size);

  maskOnDevice(mask.view(), mask_dc.view(), rgb.view(), rgb2.view(), xsize, ysize);

  float* const mask_out[3] = {mask_r, mask_g, mask_b};
  float* const mask_dc_out[3] = {maskdc_r, maskdc_g, maskdc_b};
  for (int i = 0; i < 3; i++) {
    CL_CHECK(clEnqueueReadBuffer(ocl.queue(), mask.ch[i].get(), false, 0, channel_size, mask_out[i], 0, NULL, NULL));
  }
  for (int i = 0; i < 3; i++) {
    CL_CHECK(
        clEnqueueReadBuffer(ocl.queue(), mask_dc.ch[i].get(), false, 0, channel_size, mask_dc_out[i], 0, NULL, NULL));
  }
  CL_CHECK(clFinish(ocl.queue()));
}

void OpenClKernels::diffmapOpsinDynamicsImage(cl_mem result, ocl_channels xyb0, ocl_channels xyb1,
                                              const unsigned int xsize, const unsigned int ysize,
                                              const unsigned int step) {
  const size_t res_xsize = (xsize + step - 1) / step;
  const size_t res_ysize = (ysize + step - 1) / step;

  size_t channel_size = xsize * ysize * sizeof(float);
  size_t channel_step_size = res_xsize * res_ysize * sizeof(float);

  OpenClContext& ocl = context();

  ClBuffer edge_detector_map = ocl.zeroed(3 * channel_step_size);
  ClBuffer block_diff_dc = ocl.zeroed(3 * channel_step_size);
  ClBuffer block_diff_ac = ocl.zeroed(3 * channel_step_size);

  maskHighIntensityChange(xyb0, xyb1, xsize, ysize);

  edgeDetectorMap(edge_detector_map.get(), xyb0, xyb1, xsize, ysize, step);
  blockDiffMap(block_diff_dc.get(), block_diff_ac.get(), xyb0, xyb1, xsize, ysize, step);
  edgeDetectorLowFreq(block_diff_ac.get(), xyb0, xyb1, xsize, ysize, step);
  {
    // Every pixel of mask (DiffPrecompute) and mask_dc (DoMask) is written.
    ClChannels mask = ocl.uninitializedChannels(channel_size);
    ClChannels mask_dc = ocl.uninitializedChannels(channel_size);
    maskOnDevice(mask.view(), mask_dc.view(), xyb0, xyb1, xsize, ysize);
    combineChannels(result, mask.view(), mask_dc.view(), xsize, ysize, block_diff_dc.get(), block_diff_ac.get(),
                    edge_detector_map.get(), res_xsize, step);
  }

  calculateDiffmap(result, xsize, ysize, step);
}

// Scale table for a convolution with the (constant, cached) weights buffer
// multipliers over n samples: clConvolutionScaleEx is run once per
// {multipliers, n, step, border_ratio} and the result kept on the device.
cl_mem OpenClKernels::convolutionScale(const cl_mem multipliers, int len, int n, int step, int offset,
                                       float border_ratio) {
  OpenClContext& ocl = context();

  unsigned int br_bits;
  memcpy(&br_bits, &border_ratio, sizeof(br_bits));
  const ConstantCache<ClMemory>::ConvolutionKey key = {(size_t)multipliers, (size_t)n, (size_t)step, (size_t)br_bits};
  return ocl.constants().convolutionScale(key, [&] {
    const int count = (n + step - 1) / step;
    ClMemory scale = ocl.createMemory(count * sizeof(float));
    const cl_mem scale_mem = scale.get();

    cl_kernel kernel = ocl.kernel(KERNEL_CONVOLUTIONSCALE);
    setKernelArgs(kernel, &scale_mem, &n, &multipliers, &len, &step, &offset, &border_ratio, &count);
    enqueueRange1D(kernel, count, 64);
    return scale;
  });
}

void OpenClKernels::convolutionX(cl_mem result /*out*/, const cl_mem inp, unsigned int xsize, unsigned int ysize,
                                 const cl_mem multipliers, unsigned int len, int xstep, int offset,
                                 float border_ratio) {
  OpenClContext& ocl = context();

  cl_mem scale = convolutionScale(multipliers, len, xsize, xstep, offset, border_ratio);

  cl_kernel kernel = ocl.kernel(KERNEL_CONVOLUTIONX);
  setKernelArgs(kernel, &result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &scale);

  size_t x_count = (xsize + xstep - 1) / xstep;
  enqueueRange2D(kernel, x_count, ysize);
}

void OpenClKernels::convolutionY(cl_mem result /*out*/, const cl_mem inp, unsigned int xsize, unsigned int ysize,
                                 const cl_mem multipliers, unsigned int len, int xstep, int offset,
                                 float border_ratio) {
  OpenClContext& ocl = context();

  cl_mem scale = convolutionScale(multipliers, len, ysize, xstep, offset, border_ratio);

  cl_kernel kernel = ocl.kernel(KERNEL_CONVOLUTIONY);
  setKernelArgs(kernel, &result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &scale);

  size_t x_count = (xsize + xstep - 1) / xstep;
  size_t y_count = (ysize + xstep - 1) / xstep;
  enqueueRange2D(kernel, x_count, y_count);
}

void OpenClKernels::squareSample(cl_mem result /*out*/, const cl_mem image, unsigned int xsize, unsigned int ysize,
                                 unsigned int xstep, unsigned int ystep) {
  cl_kernel kernel = context().kernel(KERNEL_SQUARESAMPLE);
  setKernelArgs(kernel, &result, &xsize, &ysize, &image, &xstep, &ystep);

  enqueueRange2D(kernel, xsize, ysize);
}

void OpenClKernels::blur(cl_mem image, const unsigned int xsize, const unsigned int ysize, const float sigma,
                         const float border_ratio, cl_mem result) {
  float m = 2.25;  // Accuracy increases when m is increased.
  const float scaler = -1.0 / (2 * sigma * sigma);
  // For m = 9.0: exp(-scaler * diff * diff) < 2^ {-52}
  const int diff = std::max<int>(1, m * fabs(sigma));
  const int expn_size = 2 * diff + 1;
  std::vector<float> expn(expn_size);
  for (int i = -diff; i <= diff; ++i) {
    expn[i + diff] = static_cast<float>(exp(scaler * i * i));
  }

  const int xstep = std::max<int>(1, int(sigma / 3));

  OpenClContext& ocl = context();

  // Weights depend only on sigma: uploaded once per sigma and kept on the device.
  const cl_mem mem_expn =
      ocl.constants().blurWeights(sigma, [&] { return ocl.uploadMemory(expn.size() * sizeof(float), expn.data()); });

  // Every element ConvolutionY reads is written by ConvolutionX: no zero-fill needed.
  ClBuffer temp = ocl.uninitialized(xsize * ysize * sizeof(float));
  convolutionX(temp.get(), image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
  convolutionY(result, temp.get(), xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
  if (xstep > 1) {
    squareSample(result, result, xsize, ysize, xstep, xstep);
  }
}

void OpenClKernels::opsinDynamicsOnDevice(const ocl_channels& rgb, const unsigned int xsize, const unsigned int ysize) {
  static const float kSigma = 1.1;
  const size_t size = xsize * ysize;
  size_t channel_size = size * sizeof(float);

  OpenClContext& ocl = context();

  // Fully overwritten by the blurs.
  ClChannels blurred_buffers = ocl.uninitializedChannels(channel_size);
  const ocl_channels blurred = blurred_buffers.view();

  blur(rgb.r, xsize, ysize, kSigma, 0.0, blurred.r);
  blur(rgb.g, xsize, ysize, kSigma, 0.0, blurred.g);
  blur(rgb.b, xsize, ysize, kSigma, 0.0, blurred.b);

  cl_kernel kernel = ocl.kernel(KERNEL_OPSINDYNAMICSIMAGE);
  const int size_arg = (int)size;
  setKernelArgs(kernel, &rgb.r, &rgb.g, &rgb.b, &size_arg, &blurred.r, &blurred.g, &blurred.b);

  // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetKernelSuggestedLocalWorkSizeKHR.html
  enqueueRange1D(kernel, size);
}

void OpenClKernels::maskHighIntensityChange(const ocl_channels& xyb0 /*in,out*/, const ocl_channels& xyb1 /*in,out*/,
                                            const unsigned int xsize, const unsigned int ysize) {
  size_t channel_size = xsize * ysize * sizeof(float);

  OpenClContext& ocl = context();

  ClChannels c0_buffers = ocl.uninitializedChannels(channel_size);  // full copies below
  ClChannels c1_buffers = ocl.uninitializedChannels(channel_size);
  const ocl_channels c0 = c0_buffers.view();
  const ocl_channels c1 = c1_buffers.view();

  CL_CHECK(clEnqueueCopyBuffer(ocl.queue(), xyb0.r, c0.r, 0, 0, channel_size, 0, NULL, NULL));
  CL_CHECK(clEnqueueCopyBuffer(ocl.queue(), xyb0.g, c0.g, 0, 0, channel_size, 0, NULL, NULL));
  CL_CHECK(clEnqueueCopyBuffer(ocl.queue(), xyb0.b, c0.b, 0, 0, channel_size, 0, NULL, NULL));
  CL_CHECK(clEnqueueCopyBuffer(ocl.queue(), xyb1.r, c1.r, 0, 0, channel_size, 0, NULL, NULL));
  CL_CHECK(clEnqueueCopyBuffer(ocl.queue(), xyb1.g, c1.g, 0, 0, channel_size, 0, NULL, NULL));
  CL_CHECK(clEnqueueCopyBuffer(ocl.queue(), xyb1.b, c1.b, 0, 0, channel_size, 0, NULL, NULL));

  cl_kernel kernel = ocl.kernel(KERNEL_MASKHIGHINTENSITYCHANGE);
  setKernelArgs(kernel, &xyb0.r, &xyb0.g, &xyb0.b, &xsize, &ysize, &xyb1.r, &xyb1.g, &xyb1.b, &c0.r, &c0.g, &c0.b,
                &c1.r, &c1.g, &c1.b);

  enqueueRange2D(kernel, xsize, ysize);
}

void OpenClKernels::edgeDetectorMap(cl_mem result /*out*/, const ocl_channels& rgb, const ocl_channels& rgb2,
                                    const unsigned int xsize, const unsigned int ysize, const unsigned int step) {
  size_t channel_size = xsize * ysize * sizeof(float);

  OpenClContext& ocl = context();

  ClChannels rgb_blured_buffers = ocl.uninitializedChannels(channel_size);  // fully written by the blurs
  ClChannels rgb2_blured_buffers = ocl.uninitializedChannels(channel_size);
  const ocl_channels rgb_blured = rgb_blured_buffers.view();
  const ocl_channels rgb2_blured = rgb2_blured_buffers.view();

  static const float kSigma[3] = {1.5, 0.586, 0.4};

  for (int i = 0; i < 3; i++) {
    blur(rgb.ch[i], xsize, ysize, kSigma[i], 0.0, rgb_blured.ch[i]);
    blur(rgb2.ch[i], xsize, ysize, kSigma[i], 0.0, rgb2_blured.ch[i]);
  }

  const size_t res_xsize = (xsize + step - 1) / step;
  const size_t res_ysize = (ysize + step - 1) / step;

  cl_kernel kernel = ocl.kernel(KERNEL_EDGEDETECTOR);
  setKernelArgs(kernel, &result, &res_xsize, &res_ysize, &rgb_blured.r, &rgb_blured.g, &rgb_blured.b, &rgb2_blured.r,
                &rgb2_blured.g, &rgb2_blured.b, &xsize, &ysize, &step);

  enqueueRange2D(kernel, res_xsize, res_ysize);
}

void OpenClKernels::blockDiffMap(cl_mem block_diff_dc /*out*/, cl_mem block_diff_ac /*out*/, const ocl_channels& rgb,
                                 const ocl_channels& rgb2, const unsigned int xsize, const unsigned int ysize,
                                 const unsigned int step) {
  const size_t res_xsize = (xsize + step - 1) / step;
  const size_t res_ysize = (ysize + step - 1) / step;

  cl_kernel kernel = context().kernel(KERNEL_BLOCKDIFFMAP);
  setKernelArgs(kernel, &block_diff_dc, &block_diff_ac, &res_xsize, &res_ysize, &rgb.r, &rgb.g, &rgb.b, &rgb2.r,
                &rgb2.g, &rgb2.b, &xsize, &ysize, &step);

  enqueueRange2D(kernel, res_xsize, res_ysize);
}

void OpenClKernels::edgeDetectorLowFreq(cl_mem block_diff_ac /*in,out*/, const ocl_channels& rgb,
                                        const ocl_channels& rgb2, const unsigned int xsize, const unsigned int ysize,
                                        const unsigned int step) {
  size_t channel_size = xsize * ysize * sizeof(float);

  static const float kSigma = 14;
  OpenClContext& ocl = context();
  ClChannels rgb_blured_buffers = ocl.uninitializedChannels(channel_size);  // fully written by the blurs
  ClChannels rgb2_blured_buffers = ocl.uninitializedChannels(channel_size);
  const ocl_channels rgb_blured = rgb_blured_buffers.view();
  const ocl_channels rgb2_blured = rgb2_blured_buffers.view();

  for (int i = 0; i < 3; i++) {
    blur(rgb.ch[i], xsize, ysize, kSigma, 0.0, rgb_blured.ch[i]);
    blur(rgb2.ch[i], xsize, ysize, kSigma, 0.0, rgb2_blured.ch[i]);
  }

  const size_t res_xsize = (xsize + step - 1) / step;
  const size_t res_ysize = (ysize + step - 1) / step;

  cl_kernel kernel = ocl.kernel(KERNEL_EDGEDETECTORLOWFREQ);
  setKernelArgs(kernel, &block_diff_ac, &res_xsize, &res_ysize, &rgb_blured.r, &rgb_blured.g, &rgb_blured.b,
                &rgb2_blured.r, &rgb2_blured.g, &rgb2_blured.b, &xsize, &ysize, &step);

  enqueueRange2D(kernel, res_xsize, res_ysize);
}

void OpenClKernels::diffPrecompute(const ocl_channels& mask /*out*/, const ocl_channels& xyb0, const ocl_channels& xyb1,
                                   const unsigned int xsize, const unsigned int ysize) {
  cl_kernel kernel = context().kernel(KERNEL_DIFFPRECOMPUTE);
  setKernelArgs(kernel, &mask.x, &mask.y, &mask.b, &xsize, &ysize, &xyb0.x, &xyb0.y, &xyb0.b, &xyb1.x, &xyb1.y,
                &xyb1.b);

  enqueueRange2D(kernel, xsize, ysize);
}

void OpenClKernels::scaleImage(cl_mem img /*in, out*/, unsigned int size, float w) {
  cl_kernel kernel = context().kernel(KERNEL_SCALEIMAGE);
  setKernelArgs(kernel, &img, &size, &w);

  enqueueRange1D(kernel, size);
}

void OpenClKernels::average5x5(cl_mem img /*in,out*/, const unsigned int xsize, const unsigned int ysize) {
  if (xsize < 4 || ysize < 4) {
    // TODO: Make this work for small dimensions as well.
    return;
  }

  OpenClContext& ocl = context();

  size_t len = xsize * ysize * sizeof(float);
  ClBuffer img_org = ocl.uninitialized(len);  // full copy below

  CL_CHECK(clEnqueueCopyBuffer(ocl.queue(), img, img_org.get(), 0, 0, len, 0, NULL, NULL));

  cl_kernel kernel = ocl.kernel(KERNEL_AVERAGE5X5);
  setKernelArgs(kernel, &img, &xsize, &ysize, &img_org.get());

  enqueueRange2D(kernel, xsize, ysize);
}

void OpenClKernels::minSquareVal(cl_mem img /*in,out*/, const unsigned int xsize, const unsigned int ysize,
                                 const unsigned int square_size, const unsigned int offset) {
  OpenClContext& ocl = context();

  size_t result_size = sizeof(cl_float) * xsize * ysize;
  ClBuffer result = ocl.uninitialized(result_size);  // every pixel written

  cl_kernel kernel = ocl.kernel(KERNEL_MINSQUAREVAL);
  setKernelArgs(kernel, &result.get(), &xsize, &ysize, &img, &square_size, &offset);

  enqueueRange2D(kernel, xsize, ysize);
  CL_CHECK(clEnqueueCopyBuffer(ocl.queue(), result.get(), img, 0, 0, sizeof(cl_float) * xsize * ysize, 0, NULL, NULL));
}

void OpenClKernels::doMask(ocl_channels mask /*in, out*/, ocl_channels mask_dc /*in, out*/, size_t xsize,
                           size_t ysize) {
  OpenClContext& ocl = context();

  // Constant LUTs: computed and uploaded once per context, kept on the device.
  const auto lut = [&ocl](int i, float extmul, float extoff, float offset, float scaler, float mul) {
    return ocl.constants().maskLut(i, [&] {
      float table[512];
      MakeMask(extmul, extoff, mul, offset, scaler, table);
      return ocl.uploadMemory(sizeof(table), table);
    });
  };
  ocl_channels xyb, xyb_dc;
  xyb.ch[0] = lut(0, 0.975741017749, -4.25328244168, 0.454909521427, 0.0738288224836, 20.8029176447);
  xyb.ch[1] = lut(1, 0.373995618954, 1.5307267433, 0.911952641929, 1.1731667845, 16.2447033988);
  xyb.ch[2] = lut(2, 0.61582234137, -4.25376118646, 1.05105070921, 0.47434643535, 31.1444967089);
  xyb_dc.ch[0] = lut(3, 1.79116943438, -3.86797479189, 0.670960225853, 0.486575865525, 20.4563479139);
  xyb_dc.ch[1] = lut(4, 0.212223514236, -3.65647120524, 1.73396799447, 0.170392660501, 21.6566724788);
  xyb_dc.ch[2] = lut(5, 0.349376011816, -0.894711072781, 0.901647926679, 0.380086095024, 18.0373825149);

  cl_kernel kernel = ocl.kernel(KERNEL_DOMASK);
  setKernelArgs(kernel, &mask.r, &mask.g, &mask.b, &xsize, &ysize, &mask_dc.r, &mask_dc.g, &mask_dc.b, &xyb.x, &xyb.y,
                &xyb.b, &xyb_dc.x, &xyb_dc.y, &xyb_dc.b);

  enqueueRange2D(kernel, xsize, ysize);
}

void OpenClKernels::maskOnDevice(ocl_channels mask /*out*/, ocl_channels mask_dc /*out*/, const ocl_channels& rgb,
                                 const ocl_channels& rgb2, const unsigned int xsize, const unsigned int ysize) {
  diffPrecompute(mask, rgb, rgb2, xsize, ysize);
  for (int i = 0; i < 3; i++) {
    average5x5(mask.ch[i], xsize, ysize);
    minSquareVal(mask.ch[i], xsize, ysize, 4, 0);

    static const float sigma[3] = {
        9.65781083553,
        14.2644604355,
        4.53358927369,
    };

    blur(mask.ch[i], xsize, ysize, sigma[i], 0.0, mask.ch[i]);
  }

  doMask(mask, mask_dc, xsize, ysize);

  for (int i = 0; i < 3; i++) {
    scaleImage(mask.ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
    scaleImage(mask_dc.ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
  }
}

void OpenClKernels::combineChannels(cl_mem result /*out*/, const ocl_channels& mask, const ocl_channels& mask_dc,
                                    const unsigned int xsize, const unsigned int ysize, const cl_mem block_diff_dc,
                                    const cl_mem block_diff_ac, const cl_mem edge_detector_map,
                                    const unsigned int res_xsize, const unsigned int step) {
  const size_t work_xsize = ((xsize - 8 + step) + step - 1) / step;
  const size_t work_ysize = ((ysize - 8 + step) + step - 1) / step;

  cl_kernel kernel = context().kernel(KERNEL_COMBINECHANNELS);
  setKernelArgs(kernel, &result, &mask.r, &mask.g, &mask.b, &mask_dc.r, &mask_dc.g, &mask_dc.b, &xsize, &ysize,
                &block_diff_dc, &block_diff_ac, &edge_detector_map, &res_xsize, &step);

  enqueueRange2D(kernel, work_xsize, work_ysize);
}

void OpenClKernels::upsampleSquareRoot(cl_mem diffmap, const unsigned int xsize, const unsigned int ysize,
                                       const int step) {
  OpenClContext& ocl = context();

  size_t diffmap_out_size = xsize * ysize * sizeof(float);
  ClBuffer diffmap_out = ocl.zeroed(diffmap_out_size);

  cl_kernel kernel = ocl.kernel(KERNEL_UPSAMPLESQUAREROOT);
  const size_t res_xsize = (xsize + step - 1) / step;
  const size_t res_ysize = (ysize + step - 1) / step;
  const int res_xsize_arg = (int)res_xsize, res_ysize_arg = (int)res_ysize;
  setKernelArgs(kernel, &diffmap_out.get(), &diffmap, &xsize, &ysize, &step, &res_xsize_arg, &res_ysize_arg);

  enqueueRange2D(kernel, res_xsize, res_ysize);
  CL_CHECK(
      clEnqueueCopyBuffer(ocl.queue(), diffmap_out.get(), diffmap, 0, 0, xsize * ysize * sizeof(float), 0, NULL, NULL));
}

void OpenClKernels::removeBorder(cl_mem out, const cl_mem in, const unsigned int xsize, const unsigned int ysize,
                                 const int step) {
  cl_int cls = 8 - step;
  cl_int cls2 = (8 - step) / 2;

  int out_xsize = xsize - cls;
  int out_ysize = ysize - cls;

  cl_kernel kernel = context().kernel(KERNEL_REMOVEBORDER);
  setKernelArgs(kernel, &out, &out_xsize, &out_ysize, &in, &cls, &cls2);

  enqueueRange2D(kernel, out_xsize, out_ysize);
}

void OpenClKernels::addBorder(cl_mem out, const unsigned int xsize, const unsigned int ysize, const int step,
                              const cl_mem in) {
  cl_int cls = 8 - step;
  cl_int cls2 = (8 - step) / 2;
  cl_kernel kernel = context().kernel(KERNEL_ADDBORDER);
  setKernelArgs(kernel, &out, &xsize, &ysize, &cls, &cls2, &in);

  enqueueRange2D(kernel, xsize, ysize);
}

void OpenClKernels::calculateDiffmap(cl_mem diffmap /*in,out*/, const unsigned int xsize, const unsigned int ysize,
                                     const int step) {
  upsampleSquareRoot(diffmap, xsize, ysize, step);

  static const float kSigma = 8.8510880283;
  static const float mul1 = 24.8235314874;
  static const float scale = 1.0 / (1.0 + mul1);

  const int s = 8 - step;

  OpenClContext& ocl = context();
  size_t blurred_size = (xsize - s) * (ysize - s) * sizeof(float);
  ClBuffer blurred = ocl.uninitialized(blurred_size);  // RemoveBorder writes every element
  removeBorder(blurred.get(), diffmap, xsize, ysize, step);

  static const float border_ratio = 0.03027655136;
  blur(blurred.get(), xsize - s, ysize - s, kSigma, border_ratio, blurred.get());

  addBorder(diffmap, xsize, ysize, step, blurred.get());
  scaleImage(diffmap, xsize * ysize, scale);
}

void OpenClKernels::copyFromJpegComponent(coeff_t* output_batch /*in,out*/, uint8_t* output_idct /*out*/,
                                          const coeff_t* jpeg_batch /*in*/, const int* quant,
                                          const int jpeg_block_width, const int jpeg_block_height,
                                          const int output_block_width, const int output_block_height,
                                          const int output_width, const int output_height) {
  OpenClContext& ocl = context();

  int src_block_count = jpeg_block_width * jpeg_block_height;
  size_t src_coeff_size = src_block_count * sizeof(::coeff_t) * kDCTBlockSize;
  ClBuffer src_coeff = ocl.upload(src_coeff_size, jpeg_batch);

  size_t dst_coeff_size = output_block_width * output_block_height * sizeof(::coeff_t) * kDCTBlockSize;
  ClBuffer dst_coeff = ocl.upload(dst_coeff_size, output_batch);

  size_t dst_idct_size = output_block_width * output_block_height * sizeof(uint8_t) * kDCTBlockSize;
  ClBuffer dst_idct = ocl.upload(dst_idct_size, output_idct);

  size_t src_quant_size = kDCTBlockSize * sizeof(int);
  ClBuffer src_quant = ocl.upload(src_quant_size, quant);

  cl_kernel kernel = ocl.kernel(KERNEL_COPYFROMJPEGCOMPONENT);
  setKernelArgs(kernel, &dst_coeff.get(), &dst_idct.get(), &src_coeff.get(), &src_quant.get(), &jpeg_block_width,
                &jpeg_block_height, &output_block_width, &output_block_height, &output_width, &output_height);

  enqueueRange2D(kernel, output_block_width, output_block_height);

  CL_CHECK(clEnqueueReadBuffer(ocl.queue(), dst_coeff.get(), false, 0, dst_coeff_size, output_batch, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(ocl.queue(), dst_idct.get(), false, 0, dst_idct_size, output_idct, 0, NULL, NULL));
  CL_CHECK(clFinish(ocl.queue()));
}

void OpenClKernels::applyGlobalQuantization(coeff_t* output_batch /*in,out*/, uint8_t* output_idct /*out*/,
                                            uint8_t* output_bool /*out*/, const int* q /*in*/, const int block_width,
                                            const int block_height) {
  OpenClContext& ocl = context();

  size_t dst_coeff_size = block_width * block_height * sizeof(::coeff_t) * kDCTBlockSize;
  ClBuffer dst_coeff = ocl.upload(dst_coeff_size, output_batch);

  size_t dst_idct_size = block_width * block_height * sizeof(uint8_t) * kDCTBlockSize;
  ClBuffer dst_idct = ocl.upload(dst_idct_size, output_idct);

  size_t dst_bool_size = block_width * block_height * sizeof(uchar);
  ClBuffer dst_bool = ocl.upload(dst_bool_size, output_bool);

  size_t src_q_size = kDCTBlockSize * sizeof(int);
  ClBuffer src_q = ocl.upload(src_q_size, q);

  cl_kernel kernel = ocl.kernel(KERNEL_APPLYGLOBALQUANTIZATION);
  setKernelArgs(kernel, &dst_coeff.get(), &dst_idct.get(), &dst_bool.get(), &src_q.get(), &block_width, &block_height);

  enqueueRange2D(kernel, block_width, block_height);

  CL_CHECK(clEnqueueReadBuffer(ocl.queue(), dst_coeff.get(), false, 0, dst_coeff_size, output_batch, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(ocl.queue(), dst_idct.get(), false, 0, dst_idct_size, output_idct, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(ocl.queue(), dst_bool.get(), false, 0, dst_bool_size, output_bool, 0, NULL, NULL));
  CL_CHECK(clFinish(ocl.queue()));
}

// Enqueues the component -> 8-bit pixel conversion of a region plus the YCbCr -> RGB
// transform into the device buffer out (xsize * ysize * 3 bytes, every byte written).
// pixels[c] holds the device copy of components[c].pixels().
void OpenClKernels::componentsToPixelsDevice(cl_mem out, const cl_mem pixels[3], const int xmin, const int ymin,
                                             const int xsize, const int ysize,
                                             const OutputImageComponent* const components[3]) {
  OpenClContext& ocl = context();

  for (int c = 0; c < 3; ++c) {
    const int width = components[c]->width();
    const int height = components[c]->height();

    const int yend1 = ymin + ysize;
    const int yend0 = std::min(yend1, height);
    const int xend1 = xmin + xsize;
    const int xend0 = std::min(xend1, width);

    {
      cl_kernel kernel = ocl.kernel(KERNEL_COMPONENTSTOPIXELS);
      setKernelArgs(kernel, &out, &xmin, &ymin, &xsize, &ysize, &c, &pixels[c], &width, &height);
      enqueueRange2D(kernel, xend0 - xmin, yend0 - ymin);
    }

    if (xend1 - xend0 > 0) {
      cl_kernel kernel = ocl.kernel(KERNEL_COMPONENTSTOPIXELS_EX1);
      setKernelArgs(kernel, &out, &xmin, &ymin, &xsize, &ysize, &c, &width, &height);
      enqueueRange2D(kernel, xend1 - xend0, yend0 - ymin);
    }

    if (yend1 - yend0 > 0) {
      cl_kernel kernel = ocl.kernel(KERNEL_COMPONENTSTOPIXELS_EX2);
      setKernelArgs(kernel, &out, &xmin, &ymin, &xsize, &ysize, &c, &width, &height);
      enqueueRange2D(kernel, xsize, yend1 - yend0);
    }
  }

  cl_kernel kernel = ocl.kernel(KERNEL_COLORTRANSFORMYCBCRTORGB);
  const int pixel_count = xsize * ysize;
  setKernelArgs(kernel, &out, &pixel_count);
  enqueueRange1D(kernel, pixel_count);
}

void OpenClKernels::componentsToPixels(uint8_t* rgb, /*out*/
                                       const int xmin, const int ymin, const int xsize, const int ysize,
                                       const std::vector<OutputImageComponent>& components /*in*/) {
  OpenClContext& ocl = context();

  const OutputImageComponent* comps[3] = {&components[0], &components[1], &components[2]};
  ClBuffer pixels[3];
  cl_mem pixel_mems[3];
  for (int c = 0; c < 3; ++c) {
    const size_t pixels_size = (size_t)comps[c]->width() * comps[c]->height() * sizeof(ushort);
    pixels[c] = ocl.upload(pixels_size, comps[c]->pixels());
    pixel_mems[c] = pixels[c].get();
  }

  const size_t out_size = (size_t)xsize * ysize * 3 * sizeof(uchar);
  ClBuffer cl_out = ocl.uninitialized(out_size);  // every byte is written

  componentsToPixelsDevice(cl_out.get(), pixel_mems, xmin, ymin, xsize, ysize, comps);

  CL_CHECK(clEnqueueReadBuffer(ocl.queue(), cl_out.get(), CL_TRUE, 0, out_size, rgb, 0, NULL, NULL));
}

std::unique_ptr<GpuCompareSession> OpenClKernels::startCompare(const float* r, const float* g, const float* b,
                                                               int xsize, int ysize) {
  const size_t channel_size = (size_t)xsize * ysize * sizeof(float);
  return std::make_unique<CompareSession>(*this, context().uploadChannels(channel_size, r, g, b), xsize, ysize);
}

void OpenClKernels::compare(const ocl_channels& ref, const int xsize, const int ysize, const OutputImage& img, int step,
                            float* distmap /*out*/) {
  OpenClContext& ocl = context();

  const int size = xsize * ysize;
  const size_t channel_size = (size_t)size * sizeof(float);

  // 1. Output image -> linear RGB, entirely on the device (was: download 8-bit RGB,
  //    CPU LUT, re-upload three float planes).
  ClChannels xyb1_buffers;
  {
    const OutputImageComponent* comps[3] = {&img.component(0), &img.component(1), &img.component(2)};
    ClBuffer pixels[3];
    cl_mem pixel_mems[3];
    for (int c = 0; c < 3; ++c) {
      const size_t pixels_size = (size_t)comps[c]->width() * comps[c]->height() * sizeof(ushort);
      pixels[c] = ocl.uninitialized(pixels_size);
      pixel_mems[c] = pixels[c].get();
      // Non-blocking: img outlives this call, and the blocking read at the end
      // completes all queued work before we return.
      CL_CHECK(clEnqueueWriteBuffer(ocl.queue(), pixel_mems[c], CL_FALSE, 0, pixels_size, comps[c]->pixels(), 0, NULL,
                                    NULL));
    }

    ClBuffer srgb = ocl.uninitialized((size_t)size * 3);
    componentsToPixelsDevice(srgb.get(), pixel_mems, 0, 0, xsize, ysize, comps);

    xyb1_buffers = ocl.uninitializedChannels(channel_size);
    const ocl_channels xyb1 = xyb1_buffers.view();
    // Same values as OutputImage::ToLinearRGB (the double table cast to float).
    cl_mem lut = ocl.constants().srgbToLinear([&] {
      const double* table = Srgb8ToLinearTable();
      float lut_f[256];
      for (int i = 0; i < 256; i++) lut_f[i] = static_cast<float>(table[i]);
      return ocl.uploadMemory(sizeof(lut_f), lut_f);
    });
    cl_kernel kernel = ocl.kernel(KERNEL_SRGBTOLINEAR);
    setKernelArgs(kernel, &srgb.get(), &xyb1.r, &xyb1.g, &xyb1.b, &lut, &size);
    enqueueRange1D(kernel, size);
  }

  const ocl_channels xyb1 = xyb1_buffers.view();

  // 2. Reference stays resident; the diffmap mutates its input, so work on a copy.
  ClChannels xyb0_buffers = ocl.uninitializedChannels(channel_size);
  const ocl_channels xyb0 = xyb0_buffers.view();
  for (int c = 0; c < 3; ++c) {
    CL_CHECK(clEnqueueCopyBuffer(ocl.queue(), ref.ch[c], xyb0.ch[c], 0, 0, channel_size, 0, NULL, NULL));
  }

  // Zero-filled: CombineChannels / UpsampleSquareRoot leave the borders untouched.
  ClBuffer mem_result = ocl.zeroed(channel_size);

  opsinDynamicsOnDevice(xyb1, xsize, ysize);
  diffmapOpsinDynamicsImage(mem_result.get(), xyb0, xyb1, xsize, ysize, step);

  CL_CHECK(clEnqueueReadBuffer(ocl.queue(), mem_result.get(), CL_TRUE, 0, channel_size, distmap, 0, NULL, NULL));
}

}  // namespace guetzli

#endif  // __USE_OPENCL__
