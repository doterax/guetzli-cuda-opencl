// The butteraugli pipeline on the device, written once for OpenCL and CUDA.
//
// GpuKernelsBase<Derived, Types> implements GpuKernels' butteraugli part (opsin dynamics, mask, compare sessions) and
// componentsToPixels from the composite steps below (blur, edge detectors, masks, diffmap, ...). A backend derives
// from it (CRTP) and provides:
//   Types: Mem (device pointer: cl_mem / cu_mem), View (three-plane view: OclChannels / OcuChannels), Buffer and
//          Channels (pooled RAII buffers), Memory (unpooled constant table).
//   Derived::context(): zeroed / uninitialized / upload / uploadAsync / *Channels / allocateMemory / uploadMemory /
//          constants() / copy / readAsync / finish (see OpenClContext, CudaContext).
//   Derived::*Kernel(...): one launcher per kernel (argument packing and launch shape only).
// Every float constant and expression of the pipeline lives here, so both backends compute the same host-side values
// (the kernels themselves still compile with different toolchains and fast-math flags, see todo 22).
#ifndef GUETZLI_BACKEND_GPU_GPU_KERNELS_BASE_H_
#define GUETZLI_BACKEND_GPU_GPU_KERNELS_BASE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "backend/gpu/constant_cache.h"
#include "backend/gpu/gpu_kernels.h"
#include "core/image/gamma_correct.h"

namespace guetzli {

namespace gpu_detail {

// DoMask lookup table (512 entries), in double like the CPU reference.
inline void makeMask(float extmul, float extoff, float mul, float offset, float scaler, float* result) {
  for (size_t i = 0; i < 512; ++i) {
    const float c = mul / ((0.01 * scaler * static_cast<double>(i)) + offset);
    result[i] = 1.0 + (extmul * (c + extoff));
    result[i] *= result[i];
  }
}

inline constexpr float kInternalGoodQualityThreshold = 14.921561160295326;
inline constexpr float kGlobalScale = 1.0 / kInternalGoodQualityThreshold;

// Key component for a device pointer (cl_mem is a pointer, cu_mem an integer).
template <class Mem>
size_t memKey(Mem mem) {
  if constexpr (std::is_pointer_v<Mem>) {
    return reinterpret_cast<size_t>(mem);
  } else {
    return static_cast<size_t>(mem);
  }
}

}  // namespace gpu_detail

template <class Derived, class Types>
class GpuKernelsBase : public GpuKernels {
 public:
  using Mem = typename Types::Mem;
  using View = typename Types::View;
  using Buffer = typename Types::Buffer;
  using Channels = typename Types::Channels;
  using Memory = typename Types::Memory;

  // In place, three planes of xsize * ysize values.
  void opsinDynamicsImage(float* r, float* g, float* b, unsigned int xsize, unsigned int ysize) override {
    auto& c = ctx();
    const size_t channel_size = static_cast<size_t>(xsize) * ysize * sizeof(float);
    Channels rgb = c.uploadChannels(channel_size, r, g, b);
    opsinDynamicsOnDevice(rgb.view(), xsize, ysize);
    float* const out[3] = {r, g, b};
    for (int i = 0; i < 3; ++i) {
      c.readAsync(out[i], rgb.ch[i].get(), channel_size);
    }
    c.finish();
  }

  void mask(float* mask_r, float* mask_g, float* mask_b, float* maskdc_r, float* maskdc_g, float* maskdc_b,
            unsigned int xsize, unsigned int ysize, const float* r, const float* g, const float* b, const float* r2,
            const float* g2, const float* b2) override {
    auto& c = ctx();
    const size_t channel_size = static_cast<size_t>(xsize) * ysize * sizeof(float);
    Channels rgb = c.uploadChannels(channel_size, r, g, b);
    Channels rgb2 = c.uploadChannels(channel_size, r2, g2, b2);
    Channels mask = c.zeroedChannels(channel_size);
    Channels mask_dc = c.zeroedChannels(channel_size);
    maskOnDevice(mask.view(), mask_dc.view(), rgb.view(), rgb2.view(), xsize, ysize);
    float* const mask_out[3] = {mask_r, mask_g, mask_b};
    float* const mask_dc_out[3] = {maskdc_r, maskdc_g, maskdc_b};
    for (int i = 0; i < 3; ++i) {
      c.readAsync(mask_out[i], mask.ch[i].get(), channel_size);
    }
    for (int i = 0; i < 3; ++i) {
      c.readAsync(mask_dc_out[i], mask_dc.ch[i].get(), channel_size);
    }
    c.finish();
  }

  std::unique_ptr<GpuCompareSession> startCompare(const float* r, const float* g, const float* b, int xsize,
                                                  int ysize) override {
    const size_t channel_size = static_cast<size_t>(xsize) * ysize * sizeof(float);
    return std::make_unique<CompareSession>(*this, ctx().uploadChannels(channel_size, r, g, b), xsize, ysize);
  }

  void componentsToPixels(uint8_t* rgb, int xmin, int ymin, int xsize, int ysize,
                          const std::vector<OutputImageComponent>& components) override {
    auto& c = ctx();
    const OutputImageComponent* comps[3] = {components.data(), &components[1], &components[2]};
    Buffer pixels[3];
    Mem pixel_mems[3];
    for (int i = 0; i < 3; ++i) {
      const size_t pixels_size = static_cast<size_t>(comps[i]->width()) * comps[i]->height() * sizeof(uint16_t);
      pixels[i] = c.upload(pixels_size, comps[i]->pixels());
      pixel_mems[i] = pixels[i].get();
    }
    const size_t out_size = static_cast<size_t>(xsize) * ysize * 3;
    Buffer out = c.uninitialized(out_size);  // every byte is written
    componentsToPixelsDevice(out.get(), pixel_mems, xmin, ymin, xsize, ysize, comps);
    c.readAsync(rgb, out.get(), out_size);
    c.finish();
  }

 private:
  friend Derived;  // CRTP: only the backend constructs its base
  GpuKernelsBase() = default;

  // Device state of one comparator: the original's opsin-dynamics image stays resident.
  class CompareSession final : public GpuCompareSession {
   public:
    CompareSession(GpuKernelsBase& kernels, Channels ref, int xsize, int ysize)
        : kernels_(kernels), ref_(std::move(ref)), xsize_(xsize), ysize_(ysize) {}

    void compare(const OutputImage& img, int step, float* distmap) override {
      kernels_.compare(ref_.view(), xsize_, ysize_, img, step, distmap);
    }

   private:
    GpuKernelsBase& kernels_;
    Channels ref_;  // never modified
    int xsize_;
    int ysize_;
  };

  Derived& self() { return static_cast<Derived&>(*this); }
  auto& ctx() { return self().context(); }

  // Distance map of img against the resident reference (xsize * ysize values, out).
  void compare(const View& ref, int xsize, int ysize, const OutputImage& img, int step, float* distmap) {
    auto& c = ctx();
    const int size = xsize * ysize;
    const size_t channel_size = static_cast<size_t>(size) * sizeof(float);

    // 1. Output image -> linear RGB, entirely on the device.
    Channels xyb1_buffers;
    {
      const OutputImageComponent* comps[3] = {&img.component(0), &img.component(1), &img.component(2)};
      Buffer pixels[3];
      Mem pixel_mems[3];
      for (int i = 0; i < 3; ++i) {
        const size_t pixels_size = static_cast<size_t>(comps[i]->width()) * comps[i]->height() * sizeof(uint16_t);
        // img outlives this call, and the read at the end completes all queued work.
        pixels[i] = c.uploadAsync(pixels_size, comps[i]->pixels());
        pixel_mems[i] = pixels[i].get();
      }
      Buffer srgb = c.uninitialized(static_cast<size_t>(size) * 3);
      componentsToPixelsDevice(srgb.get(), pixel_mems, 0, 0, xsize, ysize, comps);

      xyb1_buffers = c.uninitializedChannels(channel_size);
      // Same values as OutputImage::toLinearRgb (the double table cast to float).
      const Mem lut = c.constants().srgbToLinear([&] {
        const double* table = srgb8ToLinearTable();
        float lut_f[256];
        for (int i = 0; i < 256; ++i) {
          lut_f[i] = static_cast<float>(table[i]);
        }
        return c.uploadMemory(sizeof(lut_f), lut_f);
      });
      self().srgbToLinearKernel(srgb.get(), xyb1_buffers.view(), lut, size);
    }
    const View xyb1 = xyb1_buffers.view();

    // 2. The reference stays resident; the diffmap mutates its input, so work on a copy.
    Channels xyb0_buffers = c.uninitializedChannels(channel_size);
    const View xyb0 = xyb0_buffers.view();
    for (int i = 0; i < 3; ++i) {
      c.copy(xyb0.ch[i], ref.ch[i], channel_size);
    }

    // Zero-filled: combineChannels / upsampleSquareRoot leave the borders untouched.
    Buffer result = c.zeroed(channel_size);
    opsinDynamicsOnDevice(xyb1, xsize, ysize);
    diffmapOpsinDynamicsImage(result.get(), xyb0, xyb1, xsize, ysize, step);
    c.readAsync(distmap, result.get(), channel_size);
    c.finish();
  }

  // Component -> 8-bit pixels of a region, then YCbCr -> RGB, into out (xsize * ysize * 3 bytes, all written).
  // pixels[i] holds the device copy of components[i]->pixels().
  void componentsToPixelsDevice(Mem out, const Mem pixels[3], int xmin, int ymin, int xsize, int ysize,
                                const OutputImageComponent* const components[3]) {
    for (int i = 0; i < 3; ++i) {
      const int width = components[i]->width();
      const int height = components[i]->height();
      const int yend1 = ymin + ysize;
      const int yend0 = std::min(yend1, height);
      const int xend1 = xmin + xsize;
      const int xend0 = std::min(xend1, width);
      self().componentsToPixelsKernel(out, xmin, ymin, xsize, ysize, i, pixels[i], width, height, xend0 - xmin,
                                      yend0 - ymin);
      if (xend1 - xend0 > 0) {
        self().componentsToPixelsEx1Kernel(out, xmin, ymin, xsize, ysize, i, width, height, xend1 - xend0,
                                           yend0 - ymin);
      }
      if (yend1 - yend0 > 0) {
        self().componentsToPixelsEx2Kernel(out, xmin, ymin, xsize, ysize, i, width, height, yend1 - yend0);
      }
    }
    self().colorTransformKernel(out, xsize * ysize);
  }

  void diffmapOpsinDynamicsImage(Mem result, const View& xyb0, const View& xyb1, unsigned int xsize, unsigned int ysize,
                                 unsigned int step) {
    auto& c = ctx();
    const unsigned int res_xsize = (xsize + step - 1) / step;
    const unsigned int res_ysize = (ysize + step - 1) / step;
    const size_t channel_size = static_cast<size_t>(xsize) * ysize * sizeof(float);
    const size_t channel_step_size = static_cast<size_t>(res_xsize) * res_ysize * sizeof(float);

    Buffer edge_detector_map = c.zeroed(3 * channel_step_size);
    Buffer block_diff_dc = c.zeroed(3 * channel_step_size);
    Buffer block_diff_ac = c.zeroed(3 * channel_step_size);

    maskHighIntensityChange(xyb0, xyb1, xsize, ysize);
    edgeDetectorMap(edge_detector_map.get(), xyb0, xyb1, xsize, ysize, step);
    self().blockDiffMapKernel(block_diff_dc.get(), block_diff_ac.get(), res_xsize, res_ysize, xyb0, xyb1, xsize, ysize,
                              step);
    edgeDetectorLowFreq(block_diff_ac.get(), xyb0, xyb1, xsize, ysize, step);
    {
      // Every pixel of mask (diffPrecompute) and mask_dc (doMask) is written.
      Channels mask = c.uninitializedChannels(channel_size);
      Channels mask_dc = c.uninitializedChannels(channel_size);
      maskOnDevice(mask.view(), mask_dc.view(), xyb0, xyb1, xsize, ysize);
      const unsigned int work_xsize = ((xsize - 8 + step) + step - 1) / step;
      const unsigned int work_ysize = ((ysize - 8 + step) + step - 1) / step;
      self().combineChannelsKernel(result, mask.view(), mask_dc.view(), xsize, ysize, block_diff_dc.get(),
                                   block_diff_ac.get(), edge_detector_map.get(), res_xsize, step, work_xsize,
                                   work_ysize);
    }
    calculateDiffmap(result, xsize, ysize, step);
  }

  // Scale table for a convolution with the (constant, cached) weights multipliers over n samples: computed once per
  // {multipliers, n, step, border_ratio} and kept on the device.
  Mem convolutionScale(Mem multipliers, int len, int n, int step, int offset, float border_ratio) {
    auto& c = ctx();
    uint32_t br_bits = 0;
    std::memcpy(&br_bits, &border_ratio, sizeof(br_bits));
    const typename ConstantCache<Memory>::ConvolutionKey key = {gpu_detail::memKey(multipliers), static_cast<size_t>(n),
                                                                static_cast<size_t>(step), br_bits};
    return c.constants().convolutionScale(key, [&] {
      const int count = (n + step - 1) / step;
      Memory scale = c.allocateMemory(count * sizeof(float));
      self().convolutionScaleKernel(scale.get(), n, multipliers, len, step, offset, border_ratio, count);
      return scale;
    });
  }

  void blur(Mem image, unsigned int xsize, unsigned int ysize, float sigma, float border_ratio, Mem result) {
    const float m = 2.25;  // Accuracy increases when m is increased.
    const float scaler = -1.0 / (2 * sigma * sigma);
    const int diff = std::max<int>(1, m * std::fabs(sigma));
    const int expn_size = (2 * diff) + 1;
    std::vector<float> expn(expn_size);
    for (int i = -diff; i <= diff; ++i) {
      expn[i + diff] = std::exp(scaler * i * i);
    }
    const int xstep = std::max<int>(1, static_cast<int>(sigma / 3));

    auto& c = ctx();
    // Weights depend only on sigma: uploaded once per sigma and kept on the device.
    const Mem weights =
        c.constants().blurWeights(sigma, [&] { return c.uploadMemory(expn.size() * sizeof(float), expn.data()); });

    // Every element the Y pass reads is written by the X pass: no zero-fill needed.
    Buffer temp = c.uninitialized(static_cast<size_t>(xsize) * ysize * sizeof(float));
    self().convolutionXKernel(temp.get(), image, xsize, ysize, weights, expn_size, xstep, diff,
                              convolutionScale(weights, expn_size, xsize, xstep, diff, border_ratio));
    self().convolutionYKernel(result, temp.get(), xsize, ysize, weights, expn_size, xstep, diff,
                              convolutionScale(weights, expn_size, ysize, xstep, diff, border_ratio));
    if (xstep > 1) {
      self().squareSampleKernel(result, result, xsize, ysize, xstep, xstep);
    }
  }

  void opsinDynamicsOnDevice(const View& rgb, unsigned int xsize, unsigned int ysize) {
    static const float kSigma = 1.1;
    const size_t channel_size = static_cast<size_t>(xsize) * ysize * sizeof(float);
    Channels blurred_buffers = ctx().uninitializedChannels(channel_size);  // fully written by the blurs
    const View blurred = blurred_buffers.view();
    for (int i = 0; i < 3; ++i) {
      blur(rgb.ch[i], xsize, ysize, kSigma, 0.0, blurred.ch[i]);
    }
    self().opsinDynamicsKernel(rgb, static_cast<int>(xsize * ysize), blurred);
  }

  void maskHighIntensityChange(const View& xyb0 /*in,out*/, const View& xyb1 /*in,out*/, unsigned int xsize,
                               unsigned int ysize) {
    auto& c = ctx();
    const size_t channel_size = static_cast<size_t>(xsize) * ysize * sizeof(float);
    Channels c0_buffers = c.uninitializedChannels(channel_size);  // full copies below
    Channels c1_buffers = c.uninitializedChannels(channel_size);
    const View c0 = c0_buffers.view();
    const View c1 = c1_buffers.view();
    for (int i = 0; i < 3; ++i) {
      c.copy(c0.ch[i], xyb0.ch[i], channel_size);
    }
    for (int i = 0; i < 3; ++i) {
      c.copy(c1.ch[i], xyb1.ch[i], channel_size);
    }
    self().maskHighIntensityChangeKernel(xyb0, xyb1, xsize, ysize, c0, c1);
  }

  // Blurs both images with sigma[i] per channel into fresh buffers.
  void blurBoth(const View& rgb, const View& rgb2, unsigned int xsize, unsigned int ysize, const float sigma[3],
                Channels& rgb_blurred, Channels& rgb2_blurred) {
    auto& c = ctx();
    const size_t channel_size = static_cast<size_t>(xsize) * ysize * sizeof(float);
    rgb_blurred = c.uninitializedChannels(channel_size);  // fully written by the blurs
    rgb2_blurred = c.uninitializedChannels(channel_size);
    for (int i = 0; i < 3; ++i) {
      blur(rgb.ch[i], xsize, ysize, sigma[i], 0.0, rgb_blurred.ch[i].get());
      blur(rgb2.ch[i], xsize, ysize, sigma[i], 0.0, rgb2_blurred.ch[i].get());
    }
  }

  void edgeDetectorMap(Mem result /*out*/, const View& rgb, const View& rgb2, unsigned int xsize, unsigned int ysize,
                       unsigned int step) {
    static const float kSigma[3] = {1.5, 0.586, 0.4};
    Channels rgb_blurred;
    Channels rgb2_blurred;
    blurBoth(rgb, rgb2, xsize, ysize, kSigma, rgb_blurred, rgb2_blurred);
    const unsigned int res_xsize = (xsize + step - 1) / step;
    const unsigned int res_ysize = (ysize + step - 1) / step;
    self().edgeDetectorKernel(result, res_xsize, res_ysize, rgb_blurred.view(), rgb2_blurred.view(), xsize, ysize,
                              step);
  }

  void edgeDetectorLowFreq(Mem block_diff_ac /*in,out*/, const View& rgb, const View& rgb2, unsigned int xsize,
                           unsigned int ysize, unsigned int step) {
    static const float kSigma[3] = {14, 14, 14};
    Channels rgb_blurred;
    Channels rgb2_blurred;
    blurBoth(rgb, rgb2, xsize, ysize, kSigma, rgb_blurred, rgb2_blurred);
    const unsigned int res_xsize = (xsize + step - 1) / step;
    const unsigned int res_ysize = (ysize + step - 1) / step;
    self().edgeDetectorLowFreqKernel(block_diff_ac, res_xsize, res_ysize, rgb_blurred.view(), rgb2_blurred.view(),
                                     xsize, ysize, step);
  }

  void average5x5(Mem img /*in,out*/, unsigned int xsize, unsigned int ysize) {
    if (xsize < 4 || ysize < 4) {
      return;  // not implemented for small dimensions (as in the CPU code)
    }
    auto& c = ctx();
    const size_t len = static_cast<size_t>(xsize) * ysize * sizeof(float);
    Buffer img_org = c.uninitialized(len);  // full copy below
    c.copy(img_org.get(), img, len);
    self().average5x5Kernel(img, xsize, ysize, img_org.get());
  }

  void minSquareVal(Mem img /*in,out*/, unsigned int xsize, unsigned int ysize, unsigned int square_size,
                    unsigned int offset) {
    auto& c = ctx();
    const size_t len = static_cast<size_t>(xsize) * ysize * sizeof(float);
    Buffer result = c.uninitialized(len);  // every pixel written
    self().minSquareValKernel(result.get(), xsize, ysize, img, square_size, offset);
    c.copy(img, result.get(), len);
  }

  void doMask(const View& mask /*in,out*/, const View& mask_dc /*in,out*/, unsigned int xsize, unsigned int ysize) {
    auto& c = ctx();
    // Constant LUTs: computed and uploaded once per context, kept on the device.
    const auto lut = [&c](int i, float extmul, float extoff, float offset, float scaler, float mul) {
      return c.constants().maskLut(i, [&] {
        float table[512];
        gpu_detail::makeMask(extmul, extoff, mul, offset, scaler, table);
        return c.uploadMemory(sizeof(table), table);
      });
    };
    View xyb{};
    View xyb_dc{};
    xyb.ch[0] = lut(0, 0.975741017749, -4.25328244168, 0.454909521427, 0.0738288224836, 20.8029176447);
    xyb.ch[1] = lut(1, 0.373995618954, 1.5307267433, 0.911952641929, 1.1731667845, 16.2447033988);
    xyb.ch[2] = lut(2, 0.61582234137, -4.25376118646, 1.05105070921, 0.47434643535, 31.1444967089);
    xyb_dc.ch[0] = lut(3, 1.79116943438, -3.86797479189, 0.670960225853, 0.486575865525, 20.4563479139);
    xyb_dc.ch[1] = lut(4, 0.212223514236, -3.65647120524, 1.73396799447, 0.170392660501, 21.6566724788);
    xyb_dc.ch[2] = lut(5, 0.349376011816, -0.894711072781, 0.901647926679, 0.380086095024, 18.0373825149);
    self().doMaskKernel(mask, mask_dc, xsize, ysize, xyb, xyb_dc);
  }

  void maskOnDevice(const View& mask /*out*/, const View& mask_dc /*out*/, const View& rgb, const View& rgb2,
                    unsigned int xsize, unsigned int ysize) {
    static const float kSigma[3] = {9.65781083553, 14.2644604355, 4.53358927369};
    self().diffPrecomputeKernel(mask, rgb, rgb2, xsize, ysize);
    for (int i = 0; i < 3; ++i) {
      average5x5(mask.ch[i], xsize, ysize);
      minSquareVal(mask.ch[i], xsize, ysize, 4, 0);
      blur(mask.ch[i], xsize, ysize, kSigma[i], 0.0, mask.ch[i]);
    }
    doMask(mask, mask_dc, xsize, ysize);
    const unsigned int size = xsize * ysize;
    for (int i = 0; i < 3; ++i) {
      self().scaleImageKernel(mask.ch[i], size, gpu_detail::kGlobalScale * gpu_detail::kGlobalScale);
      self().scaleImageKernel(mask_dc.ch[i], size, gpu_detail::kGlobalScale * gpu_detail::kGlobalScale);
    }
  }

  void upsampleSquareRoot(Mem diffmap /*in,out*/, unsigned int xsize, unsigned int ysize, int step) {
    auto& c = ctx();
    const size_t len = static_cast<size_t>(xsize) * ysize * sizeof(float);
    Buffer diffmap_out = c.zeroed(len);
    const int res_xsize = static_cast<int>((xsize + step - 1) / step);
    const int res_ysize = static_cast<int>((ysize + step - 1) / step);
    self().upsampleSquareRootKernel(diffmap_out.get(), diffmap, xsize, ysize, step, res_xsize, res_ysize);
    c.copy(diffmap, diffmap_out.get(), len);
  }

  void calculateDiffmap(Mem diffmap /*in,out*/, unsigned int xsize, unsigned int ysize, int step) {
    static const float kSigma = 8.8510880283;
    static const float kMul1 = 24.8235314874;
    static const float kScale = 1.0 / (1.0 + kMul1);
    static const float kBorderRatio = 0.03027655136;

    upsampleSquareRoot(diffmap, xsize, ysize, step);

    const int s = 8 - step;
    const int cls = 8 - step;
    const int cls2 = (8 - step) / 2;
    const int out_xsize = static_cast<int>(xsize) - s;
    const int out_ysize = static_cast<int>(ysize) - s;
    Buffer blurred = ctx().uninitialized(static_cast<size_t>(out_xsize) * out_ysize * sizeof(float));  // all written
    self().removeBorderKernel(blurred.get(), out_xsize, out_ysize, diffmap, cls, cls2);
    blur(blurred.get(), out_xsize, out_ysize, kSigma, kBorderRatio, blurred.get());
    self().addBorderKernel(diffmap, xsize, ysize, cls, cls2, blurred.get());
    self().scaleImageKernel(diffmap, xsize * ysize, kScale);
  }
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_GPU_GPU_KERNELS_BASE_H_
