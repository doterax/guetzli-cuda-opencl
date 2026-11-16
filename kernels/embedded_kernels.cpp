#include "kernels/embedded_kernels.h"

#include <cstddef>
#include <span>
#include <string>

#include "backend/gpu_error.h"
#include "minilzo.h"

#ifdef __USE_OPENCL__
#include "kernels/guetzli_cl_src.h"
#endif
#ifdef __USE_CUDA__
#include "kernels/guetzli_cu_ptx.h"
#endif

namespace guetzli::kernels {

namespace {

[[maybe_unused]] std::string decompress(std::span<const unsigned char> lzo, std::size_t size, const char* what) {
  std::string out(size, '\0');
  lzo_uint out_size = size;
  const int result =
      lzo1x_decompress_safe(lzo.data(), lzo.size(), reinterpret_cast<unsigned char*>(out.data()), &out_size, nullptr);
  if (result != LZO_E_OK || out_size != size) {
    throw GpuError(std::string("failed to decompress the embedded ") + what);
  }
  return out;
}

}  // namespace

#ifdef __USE_OPENCL__
std::string openClSource() {
  return decompress(kOpenClSourceLzo, kOpenClSourceSize, "OpenCL source");
}
#endif

#ifdef __USE_CUDA__
std::string cudaPtx() {
  return decompress(kCudaPtxLzo, kCudaPtxSize, "CUDA PTX");
}
#endif

}  // namespace guetzli::kernels
