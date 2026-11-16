// The GPU kernels embedded in the executable (LZO-compressed, generated headers in kernels/).
#pragma once

#include <string>

namespace guetzli::kernels {

#ifdef __USE_OPENCL__
// The OpenCL C source of kernels/guetzli.cl. Throws GpuError if the embedded data is damaged.
std::string openClSource();
#endif

#ifdef __USE_CUDA__
// The CUDA fatbin built from kernels/guetzli.cu (native code for common GPUs + PTX). Throws GpuError if the
// embedded data is damaged.
std::string cudaFatbin();
// What the fatbin contains (toolkit, architectures).
const char* cudaFatbinInfo();
#endif

}  // namespace guetzli::kernels
