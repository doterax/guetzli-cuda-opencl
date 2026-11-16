// CUDA launchers for the kernels in kernels/guetzli.cl (compiled to PTX via kernels/guetzli.cu).
//
// Author: strongtu@tencent.com
//         ianhuang@tencent.com
//         chriskzhou@tencent.com
//         stephendeng@tencent.com
#ifdef __USE_CUDA__

#include "backend/cuda/cuda_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <optional>
#include <utility>
#include <vector>

#include "core/image/gamma_correct.h"

namespace guetzli {

namespace {

constexpr int kBlockSize = 128;  // 1D launches
constexpr int kBlockSizeX = 16;
constexpr int kBlockSizeY = 16;

constexpr unsigned int blockCount(unsigned int items, unsigned int per_block) {
  return (items + per_block - 1) / per_block;
}

// Block shape for a width x height 2D launch (grid = ceil; every kernel bounds-checks).
// Fixed 32x8 = 256 threads: full warps along x for coalescing. Short ranges trade
// height for width. (Was: the 96-thread shape that minimised padding -- 96x1,
// 48x2, ..., 3x32 -- i.e. partial warps and poor coalescing for most sizes.)
void calcBlock(int width, int height, int& x, int& y) {
  (void)width;
  y = 8;
  while (y > 1 && y / 2 >= height) {
    y /= 2;
  }
  x = 256 / y;
}

// Launches a width x height range with calcBlock()'s block shape.
void launch2D(CudaContext& ocu, CUfunction kernel, int width, int height, const void* args[]) {
  int x = kBlockSizeX;
  int y = kBlockSizeY;
  calcBlock(width, height, x, y);
  ocu.launch(kernel, blockCount(width, x), blockCount(height, y), x, y, args);
}

void makeMask(float extmul, float extoff, float mul, float offset, float scaler, float* result) {
  for (unsigned int i = 0; i < 512; ++i) {
    const float c = mul / ((0.01f * scaler * i) + offset);
    result[i] = 1.0f + extmul * (c + extoff);
    result[i] *= result[i];
  }
}

const float kInternalGoodQualityThreshold = 14.921561160295326f;
const float kGlobalScale = 1.0f / kInternalGoodQualityThreshold;

}  // namespace

// Per-comparator state: the original's opsin-dynamics image stays resident on the device.
class CudaKernels::CompareSession final : public GpuCompareSession {
 public:
  CompareSession(CudaKernels& kernels, CuChannels ref, int xsize, int ysize)
      : kernels_(kernels), ref_(std::move(ref)), xsize_(xsize), ysize_(ysize) {}

  void compare(const OutputImage& img, int step, float* distmap) override {
    kernels_.compare(ref_.view(), xsize_, ysize_, img, step, distmap);
  }

 private:
  CudaKernels& kernels_;
  CuChannels ref_;  // never modified
  int xsize_;
  int ysize_;
};

CudaKernels::CudaKernels(const GpuOptions& options, ILogger& logger) : options_(options), logger_(logger) {}

CudaKernels::~CudaKernels() = default;

void CudaKernels::createContext(const CudaDriver& driver) {
  context_ = std::make_unique<CudaContext>(driver, options_, logger_);
}

CudaContext& CudaKernels::context() {
  // A failed initialization throws and leaves context_ empty.
  if (!context_) {
    maybeInjectGpuFailure("CUDA");
    const std::optional<CudaDriver> driver = CudaDriver::load(logger_);
    if (!driver) {
      throw GpuError("failed to load CUDA driver library");
    }
    createContext(*driver);
  }
  return *context_;
}

bool CudaKernels::probe() {
  if (context_) {
    return true;
  }
  const std::optional<CudaDriver> driver = CudaDriver::load(logger_);
  if (!driver) {
    return false;
  }
  try {
    maybeInjectGpuFailure("CUDA");
    createContext(*driver);
  } catch (const std::exception& e) {
    logger_.log(LogLevel::Error, "CUDA not usable: %s\n", e.what());
    return false;
  }
  return true;
}

void CudaKernels::opsinDynamicsImage(float* r, float* g, float* b, const unsigned int xsize, const unsigned int ysize) {
  size_t channel_size = xsize * ysize * sizeof(float);

  CudaContext& ocu = context();
  const CudaDriver& drv = ocu.driver();
  CuChannels rgb = ocu.uploadChannels(channel_size, r, g, b);

  opsinDynamicsOnDevice(rgb.view(), xsize, ysize);

  CU_CHECK(drv.memcpy_dtoh_async(r, rgb.ch[0].get(), channel_size, ocu.stream()));
  CU_CHECK(drv.memcpy_dtoh_async(g, rgb.ch[1].get(), channel_size, ocu.stream()));
  CU_CHECK(drv.memcpy_dtoh_async(b, rgb.ch[2].get(), channel_size, ocu.stream()));
  ocu.finish();
}

void CudaKernels::computeBlockZeroingOrder(CoeffData* output_order_batch, const ChannelInfo orig_channel[3],
                                           const float* orig_image_batch, const float* mask_scale,
                                           const int image_width, const int image_height,
                                           const ChannelInfo mayout_channel[3], const int factor, const int comp_mask,
                                           const float block_error_limit) {
  // Validate factor to prevent division by zero
  if (factor <= 0) {
    logger_.log(LogLevel::Info, "Error: factor is %d, must be > 0. Skipping cuComputeBlockZeroingOrder.\n", factor);
    return;
  }

  const int block8_width = (image_width + 8 - 1) / 8;
  const int block8_height = (image_height + 8 - 1) / 8;
  const int blockf_width = (image_width + 8 * factor - 1) / (8 * factor);
  const int blockf_height = (image_height + 8 * factor - 1) / (8 * factor);

  CudaContext& ocu = context();

  CuBuffer mem_orig_coeff[3];
  CuBuffer mem_mayout_coeff[3];
  CuBuffer mem_mayout_pixel[3];
  for (int c = 0; c < 3; c++) {
    int block_count = orig_channel[c].block_width * orig_channel[c].block_height;
    mem_orig_coeff[c] = ocu.upload(block_count * sizeof(::coeff_t) * kDctBlockSize, orig_channel[c].coeff);

    block_count = mayout_channel[c].block_width * mayout_channel[c].block_height;
    mem_mayout_coeff[c] = ocu.upload(block_count * sizeof(::coeff_t) * kDctBlockSize, mayout_channel[c].coeff);

    mem_mayout_pixel[c] = ocu.upload(image_width * image_height * sizeof(uint16_t), mayout_channel[c].pixel);
  }
  CuBuffer mem_orig_image =
      ocu.upload(sizeof(float) * 3 * kDctBlockSize * block8_width * block8_height, orig_image_batch);
  CuBuffer mem_mask_scale = ocu.upload(sizeof(float) * 3 * block8_width * block8_height, mask_scale);

  size_t output_order_batch_size = sizeof(CoeffData) * 3 * kDctBlockSize * blockf_width * blockf_height;
  // Zero-filled (the kernel writes only each block's valid entries).
  CuBuffer mem_output_order_batch = ocu.zeroed(output_order_batch_size);

  CUfunction kernel = ocu.kernel(KERNEL_COMPUTEBLOCKZEROINGORDER);
  const void* args[] = {&mem_orig_coeff[0].get(),
                        &mem_orig_coeff[1].get(),
                        &mem_orig_coeff[2].get(),
                        &mem_orig_image.get(),
                        &mem_mask_scale.get(),
                        &blockf_width,
                        &blockf_height,
                        &image_width,
                        &image_height,
                        &mem_mayout_coeff[0].get(),
                        &mem_mayout_coeff[1].get(),
                        &mem_mayout_coeff[2].get(),
                        &mem_mayout_pixel[0].get(),
                        &mem_mayout_pixel[1].get(),
                        &mem_mayout_pixel[2].get(),
                        &orig_channel[0].block_width,
                        &orig_channel[1].block_width,
                        &orig_channel[2].block_width,
                        &mayout_channel[0].factor,
                        &mayout_channel[1].factor,
                        &mayout_channel[2].factor,
                        &mayout_channel[0].block_width,
                        &mayout_channel[1].block_width,
                        &mayout_channel[2].block_width,
                        &mayout_channel[0].block_height,
                        &mayout_channel[1].block_height,
                        &mayout_channel[2].block_height,
                        &factor,
                        &comp_mask,
                        &block_error_limit,
                        &mem_output_order_batch.get()};

  // Heavy per-thread state (registers + local memory) limits occupancy anyway: use
  // one-warp blocks so small images (few blocks) still spread across all SMs.
  int x = 8;
  int y = 4;
  ocu.launch(kernel, blockCount(blockf_width, x), blockCount(blockf_height, y), x, y, args);

  CU_CHECK(ocu.driver().memcpy_dtoh(output_order_batch, mem_output_order_batch.get(), output_order_batch_size));
}

void CudaKernels::mask(float* mask_r, float* mask_g, float* mask_b, float* maskdc_r, float* maskdc_g, float* maskdc_b,
                       const unsigned int xsize, const unsigned int ysize, const float* r, const float* g,
                       const float* b, const float* r2, const float* g2, const float* b2) {
  CudaContext& ocu = context();
  const CudaDriver& drv = ocu.driver();

  size_t channel_size = xsize * ysize * sizeof(float);

  CuChannels rgb = ocu.uploadChannels(channel_size, r, g, b);
  CuChannels rgb2 = ocu.uploadChannels(channel_size, r2, g2, b2);
  CuChannels mask = ocu.zeroedChannels(channel_size);
  CuChannels mask_dc = ocu.zeroedChannels(channel_size);

  maskOnDevice(mask.view(), mask_dc.view(), rgb.view(), rgb2.view(), xsize, ysize);

  float* const mask_out[3] = {mask_r, mask_g, mask_b};
  float* const mask_dc_out[3] = {maskdc_r, maskdc_g, maskdc_b};
  for (int i = 0; i < 3; i++) {
    CU_CHECK(drv.memcpy_dtoh_async(mask_out[i], mask.ch[i].get(), channel_size, ocu.stream()));
  }
  for (int i = 0; i < 3; i++) {
    CU_CHECK(drv.memcpy_dtoh_async(mask_dc_out[i], mask_dc.ch[i].get(), channel_size, ocu.stream()));
  }
  ocu.finish();
}

void CudaKernels::diffmapOpsinDynamicsImage(cu_mem result, OcuChannels xyb0, OcuChannels xyb1, const unsigned int xsize,
                                            const unsigned int ysize, const unsigned int step) {
  const unsigned int res_xsize = (xsize + step - 1) / step;
  const unsigned int res_ysize = (ysize + step - 1) / step;

  size_t channel_size = xsize * ysize * sizeof(float);
  size_t channel_step_size = res_xsize * res_ysize * sizeof(float);

  CudaContext& ocu = context();

  CuBuffer edge_detector_map = ocu.zeroed(3 * channel_step_size);
  CuBuffer block_diff_dc = ocu.zeroed(3 * channel_step_size);
  CuBuffer block_diff_ac = ocu.zeroed(3 * channel_step_size);

  maskHighIntensityChange(xyb0, xyb1, xsize, ysize);

  edgeDetectorMap(edge_detector_map.get(), xyb0, xyb1, xsize, ysize, step);
  blockDiffMap(block_diff_dc.get(), block_diff_ac.get(), xyb0, xyb1, xsize, ysize, step);
  edgeDetectorLowFreq(block_diff_ac.get(), xyb0, xyb1, xsize, ysize, step);
  {
    // Every pixel of mask (DiffPrecompute) and mask_dc (DoMask) is written.
    CuChannels mask = ocu.uninitializedChannels(channel_size);
    CuChannels mask_dc = ocu.uninitializedChannels(channel_size);
    maskOnDevice(mask.view(), mask_dc.view(), xyb0, xyb1, xsize, ysize);
    combineChannels(result, mask.view(), mask_dc.view(), xsize, ysize, block_diff_dc.get(), block_diff_ac.get(),
                    edge_detector_map.get(), res_xsize, step);
  }

  calculateDiffmap(result, xsize, ysize, step);
}

// Scale table for a convolution with the (constant, cached) weights buffer
// multipliers over n samples: clConvolutionScaleEx is run once per
// {multipliers, n, step, border_ratio} and the result kept on the device.
cu_mem CudaKernels::convolutionScale(const cu_mem multipliers, int len, int n, int step, int offset,
                                     float border_ratio) {
  CudaContext& ocu = context();

  unsigned int br_bits = 0;
  memcpy(&br_bits, &border_ratio, sizeof(br_bits));
  const ConstantCache<CuMemory>::ConvolutionKey key = {(size_t)multipliers, (size_t)n, (size_t)step, (size_t)br_bits};
  return ocu.constants().convolutionScale(key, [&] {
    const int count = (n + step - 1) / step;
    CuMemory scale = ocu.allocateMemory(count * sizeof(float));
    const cu_mem scale_mem = scale.get();

    CUfunction kernel = ocu.kernel(KERNEL_CONVOLUTIONSCALE);
    const void* args[] = {&scale_mem, &n, &multipliers, &len, &step, &offset, &border_ratio, &count};
    ocu.launch(kernel, blockCount(count, 64), 1, 64, 1, args);
    return scale;
  });
}

void CudaKernels::convolutionX(cu_mem result /*out*/, const cu_mem inp, unsigned int xsize, unsigned int ysize,
                               const cu_mem multipliers, unsigned int len, int xstep, int offset, float border_ratio) {
  CudaContext& ocu = context();

  cu_mem scale = convolutionScale(multipliers, len, xsize, xstep, offset, border_ratio);

  CUfunction kernel = ocu.kernel(KERNEL_CONVOLUTIONX);
  const void* args[] = {&result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &scale};

  unsigned int x_count = (xsize + xstep - 1) / xstep;
  launch2D(ocu, kernel, x_count, ysize, args);
}

void CudaKernels::convolutionY(cu_mem result /*out*/, const cu_mem inp, unsigned int xsize, unsigned int ysize,
                               const cu_mem multipliers, unsigned int len, int xstep, int offset, float border_ratio) {
  CudaContext& ocu = context();

  cu_mem scale = convolutionScale(multipliers, len, ysize, xstep, offset, border_ratio);

  CUfunction kernel = ocu.kernel(KERNEL_CONVOLUTIONY);
  const void* args[] = {&result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &scale};

  unsigned int x_count = (xsize + xstep - 1) / xstep;
  unsigned int y_count = (ysize + xstep - 1) / xstep;
  launch2D(ocu, kernel, x_count, y_count, args);
}

void CudaKernels::squareSample(cu_mem result /*out*/, const cu_mem image, unsigned int xsize, unsigned int ysize,
                               unsigned int xstep, unsigned int ystep) {
  CudaContext& ocu = context();

  CUfunction kernel = ocu.kernel(KERNEL_SQUARESAMPLE);
  const void* args[] = {&result, &xsize, &ysize, &image, &xstep, &ystep};

  launch2D(ocu, kernel, xsize, ysize, args);
}

void CudaKernels::blur(cu_mem image /*out, opt*/, const unsigned int xsize, const unsigned int ysize, const float sigma,
                       const float border_ratio, cu_mem result /*out, opt*/) {
  float m = 2.25;  // Accuracy increases when m is increased.
  const float scaler = -1.0f / (2 * sigma * sigma);
  // For m = 9.0: exp(-scaler * diff * diff) < 2^ {-52}
  const int diff = std::max<int>(1, m * std::fabs(sigma));
  const int expn_size = 2 * diff + 1;
  std::vector<float> expn(expn_size);
  for (int i = -diff; i <= diff; ++i) {
    expn[i + diff] = std::exp(scaler * i * i);
  }

  const int xstep = std::max<int>(1, int(sigma / 3));

  CudaContext& ocu = context();
  // Weights depend only on sigma: uploaded once per sigma and kept on the device.
  const cu_mem mem_expn =
      ocu.constants().blurWeights(sigma, [&] { return ocu.uploadMemory(expn.size() * sizeof(float), expn.data()); });

  // Every element ConvolutionY reads is written by ConvolutionX: no zero-fill needed.
  CuBuffer temp = ocu.uninitialized(sizeof(float) * xsize * ysize);
  convolutionX(temp.get(), image, xsize, ysize, mem_expn, expn_size, xstep, diff, border_ratio);
  convolutionY((result != 0u) ? result : image, temp.get(), xsize, ysize, mem_expn, expn_size, xstep, diff,
               border_ratio);
  if (xstep > 1) {
    squareSample((result != 0u) ? result : image, (result != 0u) ? result : image, xsize, ysize, xstep, xstep);
  }
}

void CudaKernels::opsinDynamicsOnDevice(const OcuChannels& rgb, const unsigned int xsize, const unsigned int ysize) {
  static const float kSigma = 1.1f;

  size_t channel_size = xsize * ysize * sizeof(float);

  CudaContext& ocu = context();
  CuChannels rgb_blurred_buffers = ocu.uninitializedChannels(channel_size);  // fully written by the blurs
  const OcuChannels rgb_blurred = rgb_blurred_buffers.view();

  const int size = xsize * ysize;

  blur(rgb.r, xsize, ysize, kSigma, 0.0, rgb_blurred.r);
  blur(rgb.g, xsize, ysize, kSigma, 0.0, rgb_blurred.g);
  blur(rgb.b, xsize, ysize, kSigma, 0.0, rgb_blurred.b);

  CUfunction kernel = ocu.kernel(KERNEL_OPSINDYNAMICSIMAGE);
  const void* args[] = {&rgb.r, &rgb.g, &rgb.b, &size, &rgb_blurred.r, &rgb_blurred.g, &rgb_blurred.b};

  ocu.launch(kernel, blockCount(size, kBlockSize), 1, kBlockSize, 1, args);
}

void CudaKernels::maskHighIntensityChange(const OcuChannels& xyb0 /*in,out*/, const OcuChannels& xyb1 /*in,out*/,
                                          const unsigned int xsize, const unsigned int ysize) {
  size_t channel_size = xsize * ysize * sizeof(float);

  CudaContext& ocu = context();
  const CudaDriver& drv = ocu.driver();

  CuChannels c0_buffers = ocu.uninitializedChannels(channel_size);  // full copies below
  CuChannels c1_buffers = ocu.uninitializedChannels(channel_size);
  const OcuChannels c0 = c0_buffers.view();
  const OcuChannels c1 = c1_buffers.view();

  CU_CHECK(drv.memcpy_dtod_async(c0.r, xyb0.r, channel_size, ocu.stream()));
  CU_CHECK(drv.memcpy_dtod_async(c0.g, xyb0.g, channel_size, ocu.stream()));
  CU_CHECK(drv.memcpy_dtod_async(c0.b, xyb0.b, channel_size, ocu.stream()));
  CU_CHECK(drv.memcpy_dtod_async(c1.r, xyb1.r, channel_size, ocu.stream()));
  CU_CHECK(drv.memcpy_dtod_async(c1.g, xyb1.g, channel_size, ocu.stream()));
  CU_CHECK(drv.memcpy_dtod_async(c1.b, xyb1.b, channel_size, ocu.stream()));

  CUfunction kernel = ocu.kernel(KERNEL_MASKHIGHINTENSITYCHANGE);
  const void* args[] = {&xyb0.r, &xyb0.g, &xyb0.b, &xsize, &ysize, &xyb1.r, &xyb1.g,
                        &xyb1.b, &c0.r,   &c0.g,   &c0.b,  &c1.r,  &c1.g,   &c1.b};

  launch2D(ocu, kernel, xsize, ysize, args);
}

void CudaKernels::edgeDetectorMap(cu_mem result /*out*/, const OcuChannels& rgb, const OcuChannels& rgb2,
                                  const unsigned int xsize, const unsigned int ysize, const unsigned int step) {
  size_t channel_size = xsize * ysize * sizeof(float);

  CudaContext& ocu = context();

  CuChannels rgb_blured_buffers = ocu.uninitializedChannels(channel_size);  // fully written by the blurs
  CuChannels rgb2_blured_buffers = ocu.uninitializedChannels(channel_size);
  const OcuChannels rgb_blured = rgb_blured_buffers.view();
  const OcuChannels rgb2_blured = rgb2_blured_buffers.view();

  static const float kSigma[3] = {1.5f, 0.586f, 0.4f};

  for (int i = 0; i < 3; i++) {
    blur(rgb.ch[i], xsize, ysize, kSigma[i], 0.0, rgb_blured.ch[i]);
    blur(rgb2.ch[i], xsize, ysize, kSigma[i], 0.0, rgb2_blured.ch[i]);
  }

  const unsigned int res_xsize = (xsize + step - 1) / step;
  const unsigned int res_ysize = (ysize + step - 1) / step;

  CUfunction kernel = ocu.kernel(KERNEL_EDGEDETECTOR);
  const void* args[] = {&result,        &res_xsize,     &res_ysize,     &rgb_blured.r, &rgb_blured.g, &rgb_blured.b,
                        &rgb2_blured.r, &rgb2_blured.g, &rgb2_blured.b, &xsize,        &ysize,        &step};

  launch2D(ocu, kernel, res_xsize, res_ysize, args);
}

void CudaKernels::blockDiffMap(cu_mem block_diff_dc /*out*/, cu_mem block_diff_ac /*out*/, const OcuChannels& rgb,
                               const OcuChannels& rgb2, const unsigned int xsize, const unsigned int ysize,
                               const unsigned int step) {
  CudaContext& ocu = context();

  const unsigned int res_xsize = (xsize + step - 1) / step;
  const unsigned int res_ysize = (ysize + step - 1) / step;

  CUfunction kernel = ocu.kernel(KERNEL_BLOCKDIFFMAP);
  const void* args[] = {&block_diff_dc, &block_diff_ac, &res_xsize, &res_ysize, &rgb.r, &rgb.g, &rgb.b,
                        &rgb2.r,        &rgb2.g,        &rgb2.b,    &xsize,     &ysize, &step};

  launch2D(ocu, kernel, res_xsize, res_ysize, args);
}

void CudaKernels::edgeDetectorLowFreq(cu_mem block_diff_ac /*in,out*/, const OcuChannels& rgb, const OcuChannels& rgb2,
                                      const unsigned int xsize, const unsigned int ysize, const unsigned int step) {
  size_t channel_size = xsize * ysize * sizeof(float);

  static const float kSigma = 14;

  CudaContext& ocu = context();
  CuChannels rgb_blured_buffers = ocu.uninitializedChannels(channel_size);  // fully written by the blurs
  CuChannels rgb2_blured_buffers = ocu.uninitializedChannels(channel_size);
  const OcuChannels rgb_blured = rgb_blured_buffers.view();
  const OcuChannels rgb2_blured = rgb2_blured_buffers.view();

  for (int i = 0; i < 3; i++) {
    blur(rgb.ch[i], xsize, ysize, kSigma, 0.0, rgb_blured.ch[i]);
    blur(rgb2.ch[i], xsize, ysize, kSigma, 0.0, rgb2_blured.ch[i]);
  }

  const unsigned int res_xsize = (xsize + step - 1) / step;
  const unsigned int res_ysize = (ysize + step - 1) / step;

  CUfunction kernel = ocu.kernel(KERNEL_EDGEDETECTORLOWFREQ);
  const void* args[] = {&block_diff_ac, &res_xsize,     &res_ysize,     &rgb_blured.r, &rgb_blured.g, &rgb_blured.b,
                        &rgb2_blured.r, &rgb2_blured.g, &rgb2_blured.b, &xsize,        &ysize,        &step};

  launch2D(ocu, kernel, res_xsize, res_ysize, args);
}

void CudaKernels::diffPrecompute(const OcuChannels& mask /*out*/, const OcuChannels& xyb0, const OcuChannels& xyb1,
                                 const unsigned int xsize, const unsigned int ysize) {
  CudaContext& ocu = context();

  CUfunction kernel = ocu.kernel(KERNEL_DIFFPRECOMPUTE);
  const void* args[] = {&mask.x, &mask.y, &mask.b, &xsize,  &ysize, &xyb0.x,
                        &xyb0.y, &xyb0.b, &xyb1.x, &xyb1.y, &xyb1.b};

  launch2D(ocu, kernel, xsize, ysize, args);
}

void CudaKernels::scaleImage(cu_mem img /*in, out*/, unsigned int size, float w) {
  CudaContext& ocu = context();
  float fw = w;

  CUfunction kernel = ocu.kernel(KERNEL_SCALEIMAGE);
  const void* args[] = {&img, &size, &fw};

  ocu.launch(kernel, (size + 511) / 512, 1, 512, 1, args);
}

void CudaKernels::average5x5(cu_mem img /*in,out*/, const unsigned int xsize, const unsigned int ysize) {
  if (xsize < 4 || ysize < 4) {
    // TODO: Make this work for small dimensions as well.
    return;
  }

  CudaContext& ocu = context();

  size_t len = xsize * ysize * sizeof(float);
  CuBuffer img_org = ocu.uninitialized(len);  // full copy below

  CU_CHECK(ocu.driver().memcpy_dtod_async(img_org.get(), img, len, ocu.stream()));

  CUfunction kernel = ocu.kernel(KERNEL_AVERAGE5X5);
  const void* args[] = {&img, &xsize, &ysize, &img_org.get()};

  launch2D(ocu, kernel, xsize, ysize, args);
}

void CudaKernels::minSquareVal(cu_mem img /*in,out*/, const unsigned int xsize, const unsigned int ysize,
                               const unsigned int square_size, const unsigned int offset) {
  CudaContext& ocu = context();

  CuBuffer result = ocu.uninitialized(sizeof(float) * xsize * ysize);  // every pixel written

  CUfunction kernel = ocu.kernel(KERNEL_MINSQUAREVAL);
  const void* args[] = {&result.get(), &xsize, &ysize, &img, &square_size, &offset};

  launch2D(ocu, kernel, xsize, ysize, args);
  CU_CHECK(ocu.driver().memcpy_dtod_async(img, result.get(), sizeof(float) * xsize * ysize, ocu.stream()));
}

void CudaKernels::doMask(OcuChannels mask /*in, out*/, OcuChannels mask_dc /*in, out*/, unsigned int xsize,
                         unsigned int ysize) {
  CudaContext& ocu = context();

  // Constant LUTs: computed and uploaded once per context, kept on the device.
  const auto lut = [&ocu](int i, float extmul, float extoff, float offset, float scaler, float mul) {
    return ocu.constants().maskLut(i, [&] {
      float table[512];
      makeMask(extmul, extoff, mul, offset, scaler, table);
      return ocu.uploadMemory(sizeof(table), table);
    });
  };
  OcuChannels xyb{};
  OcuChannels xyb_dc{};
  xyb.ch[0] = lut(0, 0.975741017749f, -4.25328244168f, 0.454909521427f, 0.0738288224836f, 20.8029176447f);
  xyb.ch[1] = lut(1, 0.373995618954f, 1.5307267433f, 0.911952641929f, 1.1731667845f, 16.2447033988f);
  xyb.ch[2] = lut(2, 0.61582234137f, -4.25376118646f, 1.05105070921f, 0.47434643535f, 31.1444967089f);
  xyb_dc.ch[0] = lut(3, 1.79116943438f, -3.86797479189f, 0.670960225853f, 0.486575865525f, 20.4563479139f);
  xyb_dc.ch[1] = lut(4, 0.212223514236f, -3.65647120524f, 1.73396799447f, 0.170392660501f, 21.6566724788f);
  xyb_dc.ch[2] = lut(5, 0.349376011816f, -0.894711072781f, 0.901647926679f, 0.380086095024f, 18.0373825149f);

  CUfunction kernel = ocu.kernel(KERNEL_DOMASK);
  const void* args[] = {&mask.r,    &mask.g, &mask.b, &xsize, &ysize,    &mask_dc.r, &mask_dc.g,
                        &mask_dc.b, &xyb.x,  &xyb.y,  &xyb.b, &xyb_dc.x, &xyb_dc.y,  &xyb_dc.b};

  launch2D(ocu, kernel, xsize, ysize, args);
}

void CudaKernels::maskOnDevice(OcuChannels mask /*out*/, OcuChannels mask_dc /*out*/, const OcuChannels& rgb,
                               const OcuChannels& rgb2, const unsigned int xsize, const unsigned int ysize) {
  diffPrecompute(mask, rgb, rgb2, xsize, ysize);
  for (int i = 0; i < 3; i++) {
    average5x5(mask.ch[i], xsize, ysize);
    minSquareVal(mask.ch[i], xsize, ysize, 4, 0);

    static const float kSigma[3] = {
        9.65781083553f,
        14.2644604355f,
        4.53358927369f,
    };

    blur(mask.ch[i], xsize, ysize, kSigma[i], 0.0);
  }

  doMask(mask, mask_dc, xsize, ysize);

  for (int i = 0; i < 3; i++) {
    scaleImage(mask.ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
    scaleImage(mask_dc.ch[i], xsize * ysize, kGlobalScale * kGlobalScale);
  }
}

void CudaKernels::combineChannels(cu_mem result /*out*/, const OcuChannels& mask, const OcuChannels& mask_dc,
                                  const unsigned int xsize, const unsigned int ysize, const cu_mem block_diff_dc,
                                  const cu_mem block_diff_ac, const cu_mem edge_detector_map,
                                  const unsigned int res_xsize, const unsigned int step) {
  CudaContext& ocu = context();

  const unsigned int work_xsize = ((xsize - 8 + step) + step - 1) / step;
  const unsigned int work_ysize = ((ysize - 8 + step) + step - 1) / step;

  CUfunction kernel = ocu.kernel(KERNEL_COMBINECHANNELS);
  const void* args[] = {&result, &mask.r, &mask.g,        &mask.b,        &mask_dc.r,         &mask_dc.g, &mask_dc.b,
                        &xsize,  &ysize,  &block_diff_dc, &block_diff_ac, &edge_detector_map, &res_xsize, &step};

  // 2D blocks; the kernel returns for padded threads (was 1x1-thread blocks).
  ocu.launch(kernel, blockCount(work_xsize, 16), blockCount(work_ysize, 16), 16, 16, args);
}

void CudaKernels::upsampleSquareRoot(cu_mem diffmap, const unsigned int xsize, const unsigned int ysize,
                                     const int step) {
  CudaContext& ocu = context();

  CuBuffer diffmap_out = ocu.zeroed(xsize * ysize * sizeof(float));

  CUfunction kernel = ocu.kernel(KERNEL_UPSAMPLESQUAREROOT);
  const unsigned int res_xsize = (xsize + step - 1) / step;
  const unsigned int res_ysize = (ysize + step - 1) / step;
  const int res_xsize_arg = (int)res_xsize;
  const int res_ysize_arg = (int)res_ysize;
  const void* args[] = {&diffmap_out.get(), &diffmap, &xsize, &ysize, &step, &res_xsize_arg, &res_ysize_arg};

  // 2D blocks with bounds guard in the kernel (was 1x1-thread blocks).
  ocu.launch(kernel, blockCount(res_xsize, 16), blockCount(res_ysize, 16), 16, 16, args);
  CU_CHECK(ocu.driver().memcpy_dtod_async(diffmap, diffmap_out.get(), xsize * ysize * sizeof(float), ocu.stream()));
}

void CudaKernels::removeBorder(cu_mem out, const cu_mem in, const unsigned int xsize, const unsigned int ysize,
                               const int step) {
  CudaContext& ocu = context();

  int cls = 8 - step;
  int cls2 = (8 - step) / 2;

  int out_xsize = xsize - cls;
  int out_ysize = ysize - cls;

  CUfunction kernel = ocu.kernel(KERNEL_REMOVEBORDER);
  const void* args[] = {&out, &out_xsize, &out_ysize, &in, &cls, &cls2};

  launch2D(ocu, kernel, out_xsize, out_ysize, args);
}

void CudaKernels::addBorder(cu_mem out, unsigned int xsize, unsigned int ysize, int step, cu_mem in) {
  CudaContext& ocu = context();

  int cls = 8 - step;
  int cls2 = (8 - step) / 2;
  CUfunction kernel = ocu.kernel(KERNEL_ADDBORDER);
  const void* args[] = {&out, &xsize, &ysize, &cls, &cls2, &in};

  launch2D(ocu, kernel, xsize, ysize, args);
}

void CudaKernels::calculateDiffmap(cu_mem diffmap /*in,out*/, const unsigned int xsize, const unsigned int ysize,
                                   const int step) {
  upsampleSquareRoot(diffmap, xsize, ysize, step);

  static const float kSigma = 8.8510880283;
  static const float kMul1 = 24.8235314874;
  static const float kScale = 1.0 / (1.0 + kMul1);

  const int s = 8 - step;

  CudaContext& ocu = context();
  CuBuffer blurred = ocu.uninitialized((xsize - s) * (ysize - s) * sizeof(float));  // RemoveBorder writes every element
  removeBorder(blurred.get(), diffmap, xsize, ysize, step);

  static const float kBorderRatio = 0.03027655136f;
  blur(blurred.get(), xsize - s, ysize - s, kSigma, kBorderRatio);

  addBorder(diffmap, xsize, ysize, step, blurred.get());
  scaleImage(diffmap, xsize * ysize, kScale);
}

void CudaKernels::copyFromJpegComponent(coeff_t* output_batch /*in,out*/, uint8_t* output_idct /*in,out*/,
                                        const coeff_t* jpeg_batch /*in*/, const int* quant, const int jpeg_block_width,
                                        const int jpeg_block_height, const int output_block_width,
                                        const int output_block_height, const int output_width,
                                        const int output_height) {
  CudaContext& ocu = context();
  const CudaDriver& drv = ocu.driver();

  int src_block_count = jpeg_block_width * jpeg_block_height;
  CuBuffer src_coeff = ocu.upload(src_block_count * sizeof(::coeff_t) * kDctBlockSize, jpeg_batch);

  size_t dst_coeff_size = output_block_width * output_block_height * sizeof(::coeff_t) * kDctBlockSize;
  CuBuffer dst_coeff = ocu.upload(dst_coeff_size, output_batch);

  size_t dst_idct_size = output_block_width * output_block_height * sizeof(uint8_t) * kDctBlockSize;
  CuBuffer dst_idct = ocu.upload(dst_idct_size, output_idct);

  size_t src_quant_size = kDctBlockSize * sizeof(int);
  CuBuffer src_quant = ocu.upload(src_quant_size, quant);

  CUfunction kernel = ocu.kernel(KERNEL_COPYFROMJPEGCOMPONENT);
  const void* args[] = {&dst_coeff.get(),  &dst_idct.get(),    &src_coeff.get(),    &src_quant.get(),
                        &jpeg_block_width, &jpeg_block_height, &output_block_width, &output_block_height,
                        &output_width,     &output_height};

  launch2D(ocu, kernel, output_block_width, output_block_height, args);

  CU_CHECK(drv.memcpy_dtoh_async(output_batch, dst_coeff.get(), dst_coeff_size, ocu.stream()));
  CU_CHECK(drv.memcpy_dtoh_async(output_idct, dst_idct.get(), dst_idct_size, ocu.stream()));
  ocu.finish();
}

void CudaKernels::applyGlobalQuantization(coeff_t* output_batch /*in,out*/, uint8_t* output_idct /*out*/,
                                          uint8_t* output_bool /*out*/, const int* q /*in*/, const int block_width,
                                          const int block_height) {
  CudaContext& ocu = context();
  const CudaDriver& drv = ocu.driver();

  size_t dst_coeff_size = block_width * block_height * sizeof(::coeff_t) * kDctBlockSize;
  CuBuffer dst_coeff = ocu.upload(dst_coeff_size, output_batch);

  size_t dst_idct_size = block_width * block_height * sizeof(uint8_t) * kDctBlockSize;
  CuBuffer dst_idct = ocu.upload(dst_idct_size, output_idct);

  size_t dst_bool_size = block_width * block_height * sizeof(uchar);
  CuBuffer dst_bool = ocu.upload(dst_bool_size, output_bool);

  size_t src_q_size = kDctBlockSize * sizeof(int);
  CuBuffer src_q = ocu.upload(src_q_size, q);

  CUfunction kernel = ocu.kernel(KERNEL_APPLYGLOBALQUANTIZATION);
  const void* args[] = {&dst_coeff.get(), &dst_idct.get(), &dst_bool.get(), &src_q.get(), &block_width, &block_height};

  launch2D(ocu, kernel, block_width, block_height, args);

  CU_CHECK(drv.memcpy_dtoh_async(output_batch, dst_coeff.get(), dst_coeff_size, ocu.stream()));
  CU_CHECK(drv.memcpy_dtoh_async(output_idct, dst_idct.get(), dst_idct_size, ocu.stream()));
  CU_CHECK(drv.memcpy_dtoh_async(output_bool, dst_bool.get(), dst_bool_size, ocu.stream()));
  ocu.finish();
}

// Enqueues the component -> 8-bit pixel conversion of a region plus the YCbCr -> RGB
// transform into the device buffer out (xsize * ysize * 3 bytes, every byte written).
// pixels[c] holds the device copy of components[c].pixels().
void CudaKernels::componentsToPixelsDevice(cu_mem out, const cu_mem pixels[3], const int xmin, const int ymin,
                                           const int xsize, const int ysize,
                                           const OutputImageComponent* const components[3]) {
  CudaContext& ocu = context();

  for (int c = 0; c < 3; ++c) {
    const int width = components[c]->width();
    const int height = components[c]->height();

    const int yend1 = ymin + ysize;
    const int yend0 = std::min(yend1, height);
    const int xend1 = xmin + xsize;
    const int xend0 = std::min(xend1, width);

    {
      CUfunction kernel = ocu.kernel(KERNEL_COMPONENTSTOPIXELS);
      const void* args[] = {&out, &xmin, &ymin, &xsize, &ysize, &c, &pixels[c], &width, &height};
      launch2D(ocu, kernel, xend0 - xmin, yend0 - ymin, args);
    }

    if (xend1 - xend0 > 0) {
      CUfunction kernel = ocu.kernel(KERNEL_COMPONENTSTOPIXELS_EX1);
      const void* args[] = {&out, &xmin, &ymin, &xsize, &ysize, &c, &width, &height};
      launch2D(ocu, kernel, xend1 - xend0, yend0 - ymin, args);
    }

    if (yend1 - yend0 > 0) {
      CUfunction kernel = ocu.kernel(KERNEL_COMPONENTSTOPIXELS_EX2);
      const void* args[] = {&out, &xmin, &ymin, &xsize, &ysize, &c, &width, &height};
      ocu.launch(kernel, blockCount(xsize, kBlockSize), yend1 - yend0, kBlockSize, 1, args);
    }
  }

  CUfunction kernel = ocu.kernel(KERNEL_COLORTRANSFORMYCBCRTORGB);
  const int pixel_count = xsize * ysize;
  const void* args[] = {&out, &pixel_count};
  ocu.launch(kernel, blockCount(pixel_count, kBlockSize), 1, kBlockSize, 1, args);
}

void CudaKernels::componentsToPixels(uint8_t* rgb, /*out*/
                                     const int xmin, const int ymin, const int xsize, const int ysize,
                                     const std::vector<OutputImageComponent>& components /*in*/) {
  CudaContext& ocu = context();

  const OutputImageComponent* comps[3] = {components.data(), &components[1], &components[2]};
  CuBuffer pixels[3];
  cu_mem pixel_mems[3];
  for (int c = 0; c < 3; ++c) {
    const size_t pixels_size = (size_t)comps[c]->width() * comps[c]->height() * sizeof(ushort);
    pixels[c] = ocu.upload(pixels_size, comps[c]->pixels());
    pixel_mems[c] = pixels[c].get();
  }

  const size_t out_size = (size_t)xsize * ysize * 3 * sizeof(uchar);
  CuBuffer cl_out = ocu.uninitialized(out_size);  // every byte is written

  componentsToPixelsDevice(cl_out.get(), pixel_mems, xmin, ymin, xsize, ysize, comps);

  CU_CHECK(ocu.driver().memcpy_dtoh(rgb, cl_out.get(), out_size));
}

std::unique_ptr<GpuCompareSession> CudaKernels::startCompare(const float* r, const float* g, const float* b, int xsize,
                                                             int ysize) {
  const size_t channel_size = (size_t)xsize * ysize * sizeof(float);
  return std::make_unique<CompareSession>(*this, context().uploadChannels(channel_size, r, g, b), xsize, ysize);
}

void CudaKernels::compare(const OcuChannels& ref, const int xsize, const int ysize, const OutputImage& img, int step,
                          float* distmap /*out*/) {
  CudaContext& ocu = context();
  const CudaDriver& drv = ocu.driver();

  const int size = xsize * ysize;
  const size_t channel_size = (size_t)size * sizeof(float);

  // 1. Output image -> linear RGB, entirely on the device (was: download 8-bit RGB,
  //    CPU LUT, re-upload three float planes).
  CuChannels xyb1_buffers;
  {
    const OutputImageComponent* comps[3] = {&img.component(0), &img.component(1), &img.component(2)};
    CuBuffer pixels[3];
    cu_mem pixel_mems[3];
    for (int c = 0; c < 3; ++c) {
      const size_t pixels_size = (size_t)comps[c]->width() * comps[c]->height() * sizeof(ushort);
      pixels[c] = ocu.upload(pixels_size, comps[c]->pixels());
      pixel_mems[c] = pixels[c].get();
    }

    CuBuffer srgb = ocu.uninitialized((size_t)size * 3);
    componentsToPixelsDevice(srgb.get(), pixel_mems, 0, 0, xsize, ysize, comps);

    xyb1_buffers = ocu.uninitializedChannels(channel_size);
    const OcuChannels xyb1 = xyb1_buffers.view();
    // Same values as OutputImage::ToLinearRGB (the double table cast to float).
    const cu_mem lut = ocu.constants().srgbToLinear([&] {
      const double* table = srgb8ToLinearTable();
      float lut_f[256];
      for (int i = 0; i < 256; i++) {
        lut_f[i] = static_cast<float>(table[i]);
      }
      return ocu.uploadMemory(sizeof(lut_f), lut_f);
    });
    CUfunction kernel = ocu.kernel(KERNEL_SRGBTOLINEAR);
    const void* args[] = {&srgb.get(), &xyb1.r, &xyb1.g, &xyb1.b, &lut, &size};
    ocu.launch(kernel, blockCount(size, 256), 1, 256, 1, args);
  }
  const OcuChannels xyb1 = xyb1_buffers.view();

  // 2. Reference stays resident; the diffmap mutates its input, so work on a copy.
  CuChannels xyb0_buffers = ocu.uninitializedChannels(channel_size);
  const OcuChannels xyb0 = xyb0_buffers.view();
  for (int c = 0; c < 3; ++c) {
    CU_CHECK(drv.memcpy_dtod_async(xyb0.ch[c], ref.ch[c], channel_size, ocu.stream()));
  }

  // Zero-filled: CombineChannels / UpsampleSquareRoot leave the borders untouched.
  CuBuffer mem_result = ocu.zeroed(channel_size);

  opsinDynamicsOnDevice(xyb1, xsize, ysize);
  diffmapOpsinDynamicsImage(mem_result.get(), xyb0, xyb1, xsize, ysize, step);

  CU_CHECK(drv.memcpy_dtoh_async(distmap, mem_result.get(), channel_size, ocu.stream()));
  ocu.finish();
}

}  // namespace guetzli

#endif  // __USE_CUDA__
