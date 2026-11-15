// The kernels of clguetzli.cl, as looked up by the OpenCL and CUDA contexts.
#ifndef GUETZLI_BACKEND_GPU_KERNEL_NAMES_H_
#define GUETZLI_BACKEND_GPU_KERNEL_NAMES_H_

namespace guetzli {

enum KernelName {
  KERNEL_CONVOLUTIONX = 0,
  KERNEL_CONVOLUTIONY,
  KERNEL_SQUARESAMPLE,
  KERNEL_OPSINDYNAMICSIMAGE,
  KERNEL_MASKHIGHINTENSITYCHANGE,
  KERNEL_EDGEDETECTOR,
  KERNEL_BLOCKDIFFMAP,
  KERNEL_EDGEDETECTORLOWFREQ,
  KERNEL_DIFFPRECOMPUTE,
  KERNEL_SCALEIMAGE,
  KERNEL_AVERAGE5X5,
  KERNEL_MINSQUAREVAL,
  KERNEL_DOMASK,
  KERNEL_COMBINECHANNELS,
  KERNEL_UPSAMPLESQUAREROOT,
  KERNEL_REMOVEBORDER,
  KERNEL_ADDBORDER,
  KERNEL_COMPUTEBLOCKZEROINGORDER,
  KERNEL_COPYFROMJPEGCOMPONENT,
  KERNEL_APPLYGLOBALQUANTIZATION,
  KERNEL_COMPONENTSTOPIXELS,
  KERNEL_COMPONENTSTOPIXELS_EX1,
  KERNEL_COMPONENTSTOPIXELS_EX2,
  KERNEL_COLORTRANSFORMYCBCRTORGB,
  KERNEL_SRGBTOLINEAR,
  KERNEL_CONVOLUTIONSCALE,
  KERNEL_COUNT,
};

// Entry point names in clguetzli.cl, indexed by KernelName.
inline constexpr const char* kKernelEntryPoints[KERNEL_COUNT] = {
    "clConvolutionXEx",
    "clConvolutionYEx",
    "clSquareSampleEx",
    "clOpsinDynamicsImageEx",
    "clMaskHighIntensityChangeEx",
    "clEdgeDetectorMapEx",
    "clBlockDiffMapEx",
    "clEdgeDetectorLowFreqEx",
    "clDiffPrecomputeEx",
    "clScaleImageEx",
    "clAverage5x5Ex",
    "clMinSquareValEx",
    "clDoMaskEx",
    "clCombineChannelsEx",
    "clUpsampleSquareRootEx",
    "clRemoveBorderEx",
    "clAddBorderEx",
    "clComputeBlockZeroingOrderEx",
    "clCopyFromJpegComponentEx",
    "clApplyGlobalQuantizationEx",
    "clComponentsToPixels",
    "clComponentsToPixelsEx1",
    "clComponentsToPixelsEx2",
    "clColorTransformYCbCrToRGB",
    "clSrgbToLinearEx",
    "clConvolutionScaleEx",
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_GPU_KERNEL_NAMES_H_
