// The GPU kernels embedded in the executable (LZO-compressed, generated headers in kernels/).
#pragma once

#include <string>

namespace guetzli::kernels {

#ifdef __USE_OPENCL__
// The OpenCL C source of kernels/guetzli.cl. Throws GpuError if the embedded data is damaged.
std::string openClSource();
#endif

#ifdef __USE_CUDA__
// The PTX built from kernels/guetzli.cu. Throws GpuError if the embedded data is damaged.
std::string cudaPtx();
#endif

}  // namespace guetzli::kernels
