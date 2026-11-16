// CUDA launchers for the kernels in kernels/guetzli.cl (compiled to PTX via kernels/guetzli.cu).
//
// Author: strongtu@tencent.com
//         ianhuang@tencent.com
//         chriskzhou@tencent.com
//         stephendeng@tencent.com
#ifdef __USE_CUDA__

#include "backend/cuda/cuda_kernels.h"

#include <exception>
#include <optional>

namespace guetzli {

namespace {

constexpr int kBlockSize = 128;  // 1D launches

constexpr unsigned int blockCount(unsigned int items, unsigned int per_block) {
  return (items + per_block - 1) / per_block;
}

// Launches a width x height range (grid = ceil; every kernel bounds-checks) with 256-thread blocks: full warps along
// x for coalescing; short ranges trade height for width. (Was: the 96-thread shape that minimised padding -- 96x1,
// 48x2, ..., 3x32 -- i.e. partial warps and poor coalescing for most sizes.)
void launch2D(CudaContext& ocu, CUfunction kernel, int width, int height, const void* args[]) {
  int y = 8;
  while (y > 1 && y / 2 >= height) {
    y /= 2;
  }
  const int x = 256 / y;
  ocu.launch(kernel, blockCount(width, x), blockCount(height, y), x, y, args);
}

}  // namespace

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

// --- Kernel launchers ---------------------------------------------------------------------------------------------

void CudaKernels::convolutionScaleKernel(cu_mem scale, int n, cu_mem multipliers, int len, int step, int offset,
                                         float border_ratio, int count) {
  CudaContext& ocu = context();
  const void* args[] = {&scale, &n, &multipliers, &len, &step, &offset, &border_ratio, &count};
  ocu.launch(ocu.kernel(KERNEL_CONVOLUTIONSCALE), blockCount(count, 64), 1, 64, 1, args);
}

void CudaKernels::convolutionXKernel(cu_mem result, cu_mem inp, unsigned int xsize, unsigned int ysize,
                                     cu_mem multipliers, unsigned int len, int xstep, int offset, cu_mem scale) {
  CudaContext& ocu = context();
  const void* args[] = {&result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &scale};
  launch2D(ocu, ocu.kernel(KERNEL_CONVOLUTIONX), (xsize + xstep - 1) / xstep, ysize, args);
}

void CudaKernels::convolutionYKernel(cu_mem result, cu_mem inp, unsigned int xsize, unsigned int ysize,
                                     cu_mem multipliers, unsigned int len, int xstep, int offset, cu_mem scale) {
  CudaContext& ocu = context();
  const void* args[] = {&result, &xsize, &ysize, &inp, &multipliers, &len, &xstep, &offset, &scale};
  launch2D(ocu, ocu.kernel(KERNEL_CONVOLUTIONY), (xsize + xstep - 1) / xstep, (ysize + xstep - 1) / xstep, args);
}

void CudaKernels::squareSampleKernel(cu_mem result, cu_mem image, unsigned int xsize, unsigned int ysize,
                                     unsigned int xstep, unsigned int ystep) {
  CudaContext& ocu = context();
  const void* args[] = {&result, &xsize, &ysize, &image, &xstep, &ystep};
  launch2D(ocu, ocu.kernel(KERNEL_SQUARESAMPLE), xsize, ysize, args);
}

void CudaKernels::opsinDynamicsKernel(const OcuChannels& rgb, int size, const OcuChannels& blurred) {
  CudaContext& ocu = context();
  const void* args[] = {&rgb.r, &rgb.g, &rgb.b, &size, &blurred.r, &blurred.g, &blurred.b};
  ocu.launch(ocu.kernel(KERNEL_OPSINDYNAMICSIMAGE), blockCount(size, kBlockSize), 1, kBlockSize, 1, args);
}

void CudaKernels::maskHighIntensityChangeKernel(const OcuChannels& xyb0, const OcuChannels& xyb1, unsigned int xsize,
                                                unsigned int ysize, const OcuChannels& c0, const OcuChannels& c1) {
  CudaContext& ocu = context();
  const void* args[] = {&xyb0.r, &xyb0.g, &xyb0.b, &xsize, &ysize, &xyb1.r, &xyb1.g,
                        &xyb1.b, &c0.r,   &c0.g,   &c0.b,  &c1.r,  &c1.g,   &c1.b};
  launch2D(ocu, ocu.kernel(KERNEL_MASKHIGHINTENSITYCHANGE), xsize, ysize, args);
}

void CudaKernels::edgeDetectorKernel(cu_mem result, unsigned int res_xsize, unsigned int res_ysize,
                                     const OcuChannels& rgb, const OcuChannels& rgb2, unsigned int xsize,
                                     unsigned int ysize, unsigned int step) {
  CudaContext& ocu = context();
  const void* args[] = {&result, &res_xsize, &res_ysize, &rgb.r, &rgb.g, &rgb.b,
                        &rgb2.r, &rgb2.g,    &rgb2.b,    &xsize, &ysize, &step};
  launch2D(ocu, ocu.kernel(KERNEL_EDGEDETECTOR), res_xsize, res_ysize, args);
}

void CudaKernels::edgeDetectorLowFreqKernel(cu_mem block_diff_ac, unsigned int res_xsize, unsigned int res_ysize,
                                            const OcuChannels& rgb, const OcuChannels& rgb2, unsigned int xsize,
                                            unsigned int ysize, unsigned int step) {
  CudaContext& ocu = context();
  const void* args[] = {&block_diff_ac, &res_xsize, &res_ysize, &rgb.r, &rgb.g, &rgb.b,
                        &rgb2.r,        &rgb2.g,    &rgb2.b,    &xsize, &ysize, &step};
  launch2D(ocu, ocu.kernel(KERNEL_EDGEDETECTORLOWFREQ), res_xsize, res_ysize, args);
}

void CudaKernels::blockDiffMapKernel(cu_mem block_diff_dc, cu_mem block_diff_ac, unsigned int res_xsize,
                                     unsigned int res_ysize, const OcuChannels& rgb, const OcuChannels& rgb2,
                                     unsigned int xsize, unsigned int ysize, unsigned int step) {
  CudaContext& ocu = context();
  const void* args[] = {&block_diff_dc, &block_diff_ac, &res_xsize, &res_ysize, &rgb.r, &rgb.g, &rgb.b,
                        &rgb2.r,        &rgb2.g,        &rgb2.b,    &xsize,     &ysize, &step};
  launch2D(ocu, ocu.kernel(KERNEL_BLOCKDIFFMAP), res_xsize, res_ysize, args);
}

void CudaKernels::diffPrecomputeKernel(const OcuChannels& mask, const OcuChannels& xyb0, const OcuChannels& xyb1,
                                       unsigned int xsize, unsigned int ysize) {
  CudaContext& ocu = context();
  const void* args[] = {&mask.x, &mask.y, &mask.b, &xsize,  &ysize, &xyb0.x,
                        &xyb0.y, &xyb0.b, &xyb1.x, &xyb1.y, &xyb1.b};
  launch2D(ocu, ocu.kernel(KERNEL_DIFFPRECOMPUTE), xsize, ysize, args);
}

void CudaKernels::scaleImageKernel(cu_mem img, unsigned int size, float w) {
  CudaContext& ocu = context();
  const void* args[] = {&img, &size, &w};
  ocu.launch(ocu.kernel(KERNEL_SCALEIMAGE), (size + 511) / 512, 1, 512, 1, args);
}

void CudaKernels::average5x5Kernel(cu_mem img, unsigned int xsize, unsigned int ysize, cu_mem img_org) {
  CudaContext& ocu = context();
  const void* args[] = {&img, &xsize, &ysize, &img_org};
  launch2D(ocu, ocu.kernel(KERNEL_AVERAGE5X5), xsize, ysize, args);
}

void CudaKernels::minSquareValKernel(cu_mem result, unsigned int xsize, unsigned int ysize, cu_mem img,
                                     unsigned int square_size, unsigned int offset) {
  CudaContext& ocu = context();
  const void* args[] = {&result, &xsize, &ysize, &img, &square_size, &offset};
  launch2D(ocu, ocu.kernel(KERNEL_MINSQUAREVAL), xsize, ysize, args);
}

void CudaKernels::doMaskKernel(const OcuChannels& mask, const OcuChannels& mask_dc, unsigned int xsize,
                               unsigned int ysize, const OcuChannels& lut, const OcuChannels& lut_dc) {
  CudaContext& ocu = context();
  const void* args[] = {&mask.r,    &mask.g, &mask.b, &xsize, &ysize,    &mask_dc.r, &mask_dc.g,
                        &mask_dc.b, &lut.x,  &lut.y,  &lut.b, &lut_dc.x, &lut_dc.y,  &lut_dc.b};
  launch2D(ocu, ocu.kernel(KERNEL_DOMASK), xsize, ysize, args);
}

void CudaKernels::combineChannelsKernel(cu_mem result, const OcuChannels& mask, const OcuChannels& mask_dc,
                                        unsigned int xsize, unsigned int ysize, cu_mem block_diff_dc,
                                        cu_mem block_diff_ac, cu_mem edge_detector_map, unsigned int res_xsize,
                                        unsigned int step, unsigned int work_xsize, unsigned int work_ysize) {
  CudaContext& ocu = context();
  const void* args[] = {&result, &mask.r, &mask.g,        &mask.b,        &mask_dc.r,         &mask_dc.g, &mask_dc.b,
                        &xsize,  &ysize,  &block_diff_dc, &block_diff_ac, &edge_detector_map, &res_xsize, &step};
  // 2D blocks; the kernel returns for padded threads.
  ocu.launch(ocu.kernel(KERNEL_COMBINECHANNELS), blockCount(work_xsize, 16), blockCount(work_ysize, 16), 16, 16, args);
}

void CudaKernels::upsampleSquareRootKernel(cu_mem diffmap_out, cu_mem diffmap, unsigned int xsize, unsigned int ysize,
                                           int step, int res_xsize, int res_ysize) {
  CudaContext& ocu = context();
  const void* args[] = {&diffmap_out, &diffmap, &xsize, &ysize, &step, &res_xsize, &res_ysize};
  // 2D blocks with a bounds guard in the kernel.
  ocu.launch(ocu.kernel(KERNEL_UPSAMPLESQUAREROOT), blockCount(res_xsize, 16), blockCount(res_ysize, 16), 16, 16, args);
}

void CudaKernels::removeBorderKernel(cu_mem out, int out_xsize, int out_ysize, cu_mem in, int cls, int cls2) {
  CudaContext& ocu = context();
  const void* args[] = {&out, &out_xsize, &out_ysize, &in, &cls, &cls2};
  launch2D(ocu, ocu.kernel(KERNEL_REMOVEBORDER), out_xsize, out_ysize, args);
}

void CudaKernels::addBorderKernel(cu_mem out, unsigned int xsize, unsigned int ysize, int cls, int cls2, cu_mem in) {
  CudaContext& ocu = context();
  const void* args[] = {&out, &xsize, &ysize, &cls, &cls2, &in};
  launch2D(ocu, ocu.kernel(KERNEL_ADDBORDER), xsize, ysize, args);
}

void CudaKernels::srgbToLinearKernel(cu_mem srgb, const OcuChannels& out, cu_mem lut, int size) {
  CudaContext& ocu = context();
  const void* args[] = {&srgb, &out.r, &out.g, &out.b, &lut, &size};
  ocu.launch(ocu.kernel(KERNEL_SRGBTOLINEAR), blockCount(size, 256), 1, 256, 1, args);
}

void CudaKernels::componentsToPixelsKernel(cu_mem out, int xmin, int ymin, int xsize, int ysize, int c, cu_mem pixels,
                                           int width, int height, int range_x, int range_y) {
  CudaContext& ocu = context();
  const void* args[] = {&out, &xmin, &ymin, &xsize, &ysize, &c, &pixels, &width, &height};
  launch2D(ocu, ocu.kernel(KERNEL_COMPONENTSTOPIXELS), range_x, range_y, args);
}

void CudaKernels::componentsToPixelsEx1Kernel(cu_mem out, int xmin, int ymin, int xsize, int ysize, int c, int width,
                                              int height, int range_x, int range_y) {
  CudaContext& ocu = context();
  const void* args[] = {&out, &xmin, &ymin, &xsize, &ysize, &c, &width, &height};
  launch2D(ocu, ocu.kernel(KERNEL_COMPONENTSTOPIXELS_EX1), range_x, range_y, args);
}

void CudaKernels::componentsToPixelsEx2Kernel(cu_mem out, int xmin, int ymin, int xsize, int ysize, int c, int width,
                                              int height, int range_y) {
  CudaContext& ocu = context();
  const void* args[] = {&out, &xmin, &ymin, &xsize, &ysize, &c, &width, &height};
  ocu.launch(ocu.kernel(KERNEL_COMPONENTSTOPIXELS_EX2), blockCount(xsize, kBlockSize), range_y, kBlockSize, 1, args);
}

void CudaKernels::colorTransformKernel(cu_mem out, int pixel_count) {
  CudaContext& ocu = context();
  const void* args[] = {&out, &pixel_count};
  ocu.launch(ocu.kernel(KERNEL_COLORTRANSFORMYCBCRTORGB), blockCount(pixel_count, kBlockSize), 1, kBlockSize, 1, args);
}

// --- DCT and block-zeroing round trips ----------------------------------------------------------------------------

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
  const int blockf_width = (image_width + (8 * factor) - 1) / (8 * factor);
  const int blockf_height = (image_height + (8 * factor) - 1) / (8 * factor);

  CudaContext& ocu = context();

  CuBuffer mem_orig_coeff[3];
  CuBuffer mem_mayout_coeff[3];
  CuBuffer mem_mayout_pixel[3];
  for (int c = 0; c < 3; c++) {
    int block_count = orig_channel[c].block_width * orig_channel[c].block_height;
    mem_orig_coeff[c] = ocu.upload(block_count * sizeof(::coeff_t) * kDctBlockSize, orig_channel[c].coeff);

    block_count = mayout_channel[c].block_width * mayout_channel[c].block_height;
    mem_mayout_coeff[c] = ocu.upload(block_count * sizeof(::coeff_t) * kDctBlockSize, mayout_channel[c].coeff);

    mem_mayout_pixel[c] =
        ocu.upload(static_cast<size_t>(image_width) * image_height * sizeof(uint16_t), mayout_channel[c].pixel);
  }
  CuBuffer mem_orig_image =
      ocu.upload(sizeof(float) * 3 * kDctBlockSize * block8_width * block8_height, orig_image_batch);
  CuBuffer mem_mask_scale = ocu.upload(sizeof(float) * 3 * block8_width * block8_height, mask_scale);

  const size_t output_order_batch_size = sizeof(CoeffData) * 3 * kDctBlockSize * blockf_width * blockf_height;
  // Zero-filled (the kernel writes only each block's valid entries).
  CuBuffer mem_output_order_batch = ocu.zeroed(output_order_batch_size);

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
  const int x = 8;
  const int y = 4;
  ocu.launch(ocu.kernel(KERNEL_COMPUTEBLOCKZEROINGORDER), blockCount(blockf_width, x), blockCount(blockf_height, y), x,
             y, args);

  CU_CHECK(ocu.driver().memcpy_dtoh(output_order_batch, mem_output_order_batch.get(), output_order_batch_size));
}

void CudaKernels::copyFromJpegComponent(coeff_t* output_batch /*in,out*/, uint8_t* output_idct /*in,out*/,
                                        const coeff_t* jpeg_batch /*in*/, const int* quant, const int jpeg_block_width,
                                        const int jpeg_block_height, const int output_block_width,
                                        const int output_block_height, const int output_width,
                                        const int output_height) {
  CudaContext& ocu = context();

  const int src_block_count = jpeg_block_width * jpeg_block_height;
  CuBuffer src_coeff = ocu.upload(src_block_count * sizeof(::coeff_t) * kDctBlockSize, jpeg_batch);

  const size_t dst_coeff_size = output_block_width * output_block_height * sizeof(::coeff_t) * kDctBlockSize;
  CuBuffer dst_coeff = ocu.upload(dst_coeff_size, output_batch);

  const size_t dst_idct_size = output_block_width * output_block_height * sizeof(uint8_t) * kDctBlockSize;
  CuBuffer dst_idct = ocu.upload(dst_idct_size, output_idct);

  CuBuffer src_quant = ocu.upload(kDctBlockSize * sizeof(int), quant);

  const void* args[] = {&dst_coeff.get(),  &dst_idct.get(),    &src_coeff.get(),    &src_quant.get(),
                        &jpeg_block_width, &jpeg_block_height, &output_block_width, &output_block_height,
                        &output_width,     &output_height};
  launch2D(ocu, ocu.kernel(KERNEL_COPYFROMJPEGCOMPONENT), output_block_width, output_block_height, args);

  ocu.readAsync(output_batch, dst_coeff.get(), dst_coeff_size);
  ocu.readAsync(output_idct, dst_idct.get(), dst_idct_size);
  ocu.finish();
}

void CudaKernels::applyGlobalQuantization(coeff_t* output_batch /*in,out*/, uint8_t* output_idct /*out*/,
                                          uint8_t* output_bool /*out*/, const int* q /*in*/, const int block_width,
                                          const int block_height) {
  CudaContext& ocu = context();

  const size_t dst_coeff_size = block_width * block_height * sizeof(::coeff_t) * kDctBlockSize;
  CuBuffer dst_coeff = ocu.upload(dst_coeff_size, output_batch);

  const size_t dst_idct_size = block_width * block_height * sizeof(uint8_t) * kDctBlockSize;
  CuBuffer dst_idct = ocu.upload(dst_idct_size, output_idct);

  const size_t dst_bool_size = block_width * block_height * sizeof(uchar);
  CuBuffer dst_bool = ocu.upload(dst_bool_size, output_bool);

  CuBuffer src_q = ocu.upload(kDctBlockSize * sizeof(int), q);

  const void* args[] = {&dst_coeff.get(), &dst_idct.get(), &dst_bool.get(), &src_q.get(), &block_width, &block_height};
  launch2D(ocu, ocu.kernel(KERNEL_APPLYGLOBALQUANTIZATION), block_width, block_height, args);

  ocu.readAsync(output_batch, dst_coeff.get(), dst_coeff_size);
  ocu.readAsync(output_idct, dst_idct.get(), dst_idct_size);
  ocu.readAsync(output_bool, dst_bool.get(), dst_bool_size);
  ocu.finish();
}

}  // namespace guetzli

#endif  // __USE_CUDA__
