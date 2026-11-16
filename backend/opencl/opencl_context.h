// OpenCL device context: device, queue, program, kernels, buffer pool and constant tables of one backend instance.
//
// Author: strongtu@tencent.com
//         ianhuang@tencent.com
#ifndef GUETZLI_BACKEND_OPENCL_OPENCL_CONTEXT_H_
#define GUETZLI_BACKEND_OPENCL_OPENCL_CONTEXT_H_

#ifdef __USE_OPENCL__

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "backend/gpu/constant_cache.h"
#include "backend/gpu/gpu_options.h"
#include "backend/gpu/kernel_names.h"
#include "backend/gpu_error.h"
#include "backend/gpu/kernel_types.h"
#include "logging/logger.h"
#include "third_party/OpenCL/include/CL/cl.h"

class Device;  // third_party/OpenCL-Wrapper

namespace guetzli {

const char* TranslateOpenCLError(cl_int errorCode);

// Throws GpuError("OpenCL call failed in <where>: <error>").
[[noreturn]] void throwOpenClError(cl_int err, const char* where);

// Evaluates e once; on failure throws GpuError (Application falls back to the CPU).
#define CL_CHECK(e)                                            \
  do {                                                         \
    const cl_int cl_check_result_ = (e);                       \
    if (CL_SUCCESS != cl_check_result_) {                      \
      ::guetzli::throwOpenClError(cl_check_result_, __func__); \
    }                                                          \
  } while (0)

class OpenClContext;

// A device buffer from the context's pool; returned to the pool when destroyed. The in-order queue orders any later
// reuse of the buffer after the commands already enqueued on it.
class ClBuffer {
 public:
  ClBuffer() = default;
  ClBuffer(ClBuffer&& other) noexcept;
  ClBuffer& operator=(ClBuffer&& other) noexcept;
  ClBuffer(const ClBuffer&) = delete;
  ClBuffer& operator=(const ClBuffer&) = delete;
  ~ClBuffer();

  // A reference, so that &buffer.get() can be passed as a kernel argument.
  [[nodiscard]] const cl_mem& get() const { return mem_; }

 private:
  friend class OpenClContext;
  ClBuffer(OpenClContext* owner, cl_mem mem) : owner_(owner), mem_(mem) {}

  OpenClContext* owner_ = nullptr;
  cl_mem mem_ = nullptr;
};

// Three planes from the pool; view() gives the handles to pass to the launchers.
struct ClChannels {
  ClBuffer ch[3];
  [[nodiscard]] ocl_channels view() const {
    ocl_channels v;
    for (int i = 0; i < 3; i++) v.ch[i] = ch[i].get();
    return v;
  }
};

// A device allocation outside the pool (constant tables); released when destroyed.
class ClMemory {
 public:
  explicit ClMemory(cl_mem mem) : mem_(mem) {}
  ClMemory(ClMemory&& other) noexcept : mem_(other.mem_) { other.mem_ = nullptr; }
  ClMemory& operator=(ClMemory&&) = delete;
  ClMemory(const ClMemory&) = delete;
  ClMemory& operator=(const ClMemory&) = delete;
  ~ClMemory();

  [[nodiscard]] cl_mem get() const { return mem_; }

 private:
  cl_mem mem_;
};

class OpenClContext {
 public:
  // Selects the device (options.device, else the one with the most FLOPs), compiles the embedded kernel source and
  // creates the kernels. Throws GpuError.
  OpenClContext(const GpuOptions& options, ILogger& logger);
  ~OpenClContext();
  OpenClContext(const OpenClContext&) = delete;
  OpenClContext& operator=(const OpenClContext&) = delete;

  [[nodiscard]] ILogger& logger() const { return logger_; }
  [[nodiscard]] cl_context context() const { return context_; }
  [[nodiscard]] cl_command_queue queue() const { return queue_; }
  [[nodiscard]] cl_kernel kernel(KernelName name) const { return kernels_[name]; }
  // Largest work-group the kernel can be launched with on this device.
  [[nodiscard]] size_t maxWorkGroupSize(cl_kernel kernel);

  // Pooled buffers of `bytes` bytes.
  [[nodiscard]] ClBuffer zeroed(size_t bytes);
  // Contents undefined: only for buffers that are fully overwritten before being read.
  [[nodiscard]] ClBuffer uninitialized(size_t bytes);
  // Blocking upload: data may go away right after the call.
  [[nodiscard]] ClBuffer upload(size_t bytes, const void* data);
  [[nodiscard]] ClChannels zeroedChannels(size_t bytes);
  [[nodiscard]] ClChannels uninitializedChannels(size_t bytes);
  [[nodiscard]] ClChannels uploadChannels(size_t bytes, const void* c0, const void* c1, const void* c2);

  // Unpooled allocations for constant tables.
  [[nodiscard]] ClMemory createMemory(size_t bytes);
  [[nodiscard]] ClMemory uploadMemory(size_t bytes, const void* data);
  [[nodiscard]] ConstantCache<ClMemory>& constants() { return constants_; }

 private:
  friend class ClBuffer;

  cl_mem createBuffer(size_t bytes);
  void release(cl_mem mem) noexcept;  // back to the pool
  void drainPool() noexcept;          // frees all idle pooled buffers
  void printDeviceCapabilities() const;

  ILogger& logger_;
  std::unique_ptr<Device> device_;  // owns context, queue and program
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_kernel kernels_[KERNEL_COUNT] = {};
  std::map<cl_kernel, size_t> max_work_group_;

  std::map<size_t, std::vector<cl_mem>> free_buffers_;  // idle pooled buffers by size
  std::map<cl_mem, size_t> buffer_sizes_;               // every pooled buffer (busy or idle) -> size
  ConstantCache<ClMemory> constants_;                   // destroyed first
};

// Prints the OpenCL devices with their indices for --version (stdout).
void enumOpenClDevices();

}  // namespace guetzli

#endif  // __USE_OPENCL__

#endif  // GUETZLI_BACKEND_OPENCL_OPENCL_CONTEXT_H_
