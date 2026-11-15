// Options for GPU backend initialization, from the configuration (--device). Verbosity comes from the logger.
#ifndef GUETZLI_BACKEND_GPU_GPU_OPTIONS_H_
#define GUETZLI_BACKEND_GPU_GPU_OPTIONS_H_

namespace guetzli {

struct GpuOptions {
  int device = -1;  // device index within the backend; < 0 = default selection
};

}  // namespace guetzli

#endif  // GUETZLI_BACKEND_GPU_GPU_OPTIONS_H_
