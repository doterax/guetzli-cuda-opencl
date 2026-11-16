// Sets OpenCL kernel arguments from pointers to host values. int and size_t values are passed as cl_int (the kernels
// take int). Throws GpuError if an argument is rejected.
#ifndef GUETZLI_BACKEND_OPENCL_KERNEL_ARGS_H_
#define GUETZLI_BACKEND_OPENCL_KERNEL_ARGS_H_

#ifdef __USE_OPENCL__

#include <cstddef>
#include <string>

#include "backend/opencl/opencl_context.h"

namespace guetzli {

namespace kernel_args {

template <typename T>
inline cl_int set(cl_kernel k, int idx, const T* t) {
  return clSetKernelArg(k, idx, sizeof(T), t);
}

inline cl_int set(cl_kernel k, int idx, const int* t) {
  const cl_int c = *t;
  return clSetKernelArg(k, idx, sizeof(cl_int), &c);
}

inline cl_int set(cl_kernel k, int idx, const size_t* t) {
  const auto c = static_cast<cl_int>(*t);
  return clSetKernelArg(k, idx, sizeof(cl_int), &c);
}

template <typename T>
inline void setChecked(cl_kernel k, int idx, const T* t) {
  const cl_int err = set(k, idx, t);
  if (err != CL_SUCCESS) throwOpenClError(err, ("clSetKernelArg(" + std::to_string(idx) + ")").c_str());
}

}  // namespace kernel_args

// Sets arguments 0..N-1 of kernel k.
template <typename... Args>
inline void setKernelArgs(cl_kernel k, const Args*... args) {
  int idx = 0;
  (kernel_args::setChecked(k, idx++, args), ...);
}

}  // namespace guetzli

#endif  // __USE_OPENCL__

#endif  // GUETZLI_BACKEND_OPENCL_KERNEL_ARGS_H_
