// CUDA device context: driver, device context, module, kernels, stream, memory pool and constant tables of one
// backend instance.
//
// Author: strongtu@tencent.com
#ifndef GUETZLI_BACKEND_CUDA_CUDA_CONTEXT_H_
#define GUETZLI_BACKEND_CUDA_CUDA_CONTEXT_H_

#ifdef __USE_CUDA__

#include <cuda.h>

#include <cstddef>

#include "backend/cuda/cuda_driver.h"
#include "backend/cuda/cuda_memory_pool.h"
#include "backend/gpu/constant_cache.h"
#include "backend/gpu/gpu_options.h"
#include "backend/gpu/kernel_names.h"
#include "backend/gpu_error.h"
#include "backend/gpu/kernel_types.h"
#include "logging/logger.h"

namespace guetzli {

const char* translateCudaError(CUresult error_code);

// Throws GpuError("CUDA call failed in <where>: <error>").
[[noreturn]] void throwCudaError(CUresult err, const char* where);

// Evaluates e once; on failure throws GpuError (Application falls back to the CPU).
#define CU_CHECK(e)                                          \
  do {                                                       \
    const CUresult cu_check_result_ = (e);                   \
    if (CUDA_SUCCESS != cu_check_result_) {                  \
      ::guetzli::throwCudaError(cu_check_result_, __func__); \
    }                                                        \
  } while (0)

// A device buffer from the context's pool; returned to the pool when destroyed. The single stream orders any later
// reuse of the buffer after the work already queued on it.
class CuBuffer {
 public:
  CuBuffer() = default;
  CuBuffer(CuBuffer&& other) noexcept;
  CuBuffer& operator=(CuBuffer&& other) noexcept;
  CuBuffer(const CuBuffer&) = delete;
  CuBuffer& operator=(const CuBuffer&) = delete;
  ~CuBuffer();

  // A reference, so that &buffer.get() can be passed as a kernel argument.
  [[nodiscard]] const cu_mem& get() const { return mem_; }

 private:
  friend class CudaContext;
  CuBuffer(CudaMemoryPool* pool, cu_mem mem) : pool_(pool), mem_(mem) {}

  CudaMemoryPool* pool_ = nullptr;
  cu_mem mem_ = 0;
};

// Three planes from the pool; view() gives the handles to pass to the launchers.
struct CuChannels {
  CuBuffer ch[3];
  [[nodiscard]] OcuChannels view() const {
    OcuChannels v{};
    for (int i = 0; i < 3; i++) {
      v.ch[i] = ch[i].get();
    }
    return v;
  }
};

// A device allocation outside the pool (constant tables); freed when destroyed.
class CuMemory {
 public:
  CuMemory(const CudaDriver& driver, cu_mem mem) : driver_(&driver), mem_(mem) {}
  CuMemory(CuMemory&& other) noexcept : driver_(other.driver_), mem_(other.mem_) { other.mem_ = 0; }
  CuMemory& operator=(CuMemory&&) = delete;
  CuMemory(const CuMemory&) = delete;
  CuMemory& operator=(const CuMemory&) = delete;
  ~CuMemory() {
    if (mem_ != 0u) {
      driver_->mem_free(mem_);
    }
  }

  [[nodiscard]] cu_mem get() const { return mem_; }

 private:
  const CudaDriver* driver_;
  cu_mem mem_;
};

class CudaContext {
 public:
  // Creates a context on the selected device (options.device, else device 0), JIT-compiles the embedded PTX and
  // looks up the kernels. Throws GpuError.
  CudaContext(const CudaDriver& driver, const GpuOptions& options, ILogger& logger);
  ~CudaContext();
  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  [[nodiscard]] const CudaDriver& driver() const { return driver_; }
  [[nodiscard]] CUstream stream() const { return stream_; }
  [[nodiscard]] CUfunction kernel(KernelName name) const { return kernels_[name]; }
  // Launches f on a gx x gy grid of bx x by blocks on the stream.
  void launch(CUfunction f, unsigned int gx, unsigned int gy, unsigned int bx, unsigned int by, const void* args[]);
  // Waits for all queued work.
  void finish();

  // Pooled buffers of `bytes` bytes.
  [[nodiscard]] CuBuffer zeroed(size_t bytes);
  // Contents undefined: only for buffers that are fully overwritten before being read.
  [[nodiscard]] CuBuffer uninitialized(size_t bytes);
  // Queued upload from pageable host memory (staged before the call returns).
  [[nodiscard]] CuBuffer upload(size_t bytes, const void* data);
  // Same as upload (pageable memory is staged before the call returns).
  [[nodiscard]] CuBuffer uploadAsync(size_t bytes, const void* data) { return upload(bytes, data); }
  [[nodiscard]] CuChannels zeroedChannels(size_t bytes);
  [[nodiscard]] CuChannels uninitializedChannels(size_t bytes);
  [[nodiscard]] CuChannels uploadChannels(size_t bytes, const void* c0, const void* c1, const void* c2);

  // Unpooled allocations for constant tables.
  [[nodiscard]] CuMemory allocateMemory(size_t bytes);
  [[nodiscard]] CuMemory uploadMemory(size_t bytes, const void* data);  // synchronous
  [[nodiscard]] ConstantCache<CuMemory>& constants() { return constants_; }

  // Queued device-to-device copy.
  void copy(cu_mem dst, cu_mem src, size_t bytes);
  // Queued download: dst is valid after finish().
  void readAsync(void* dst, cu_mem src, size_t bytes);

 private:
  void destroy() noexcept;
  // JIT-compiles the embedded PTX for the device (compute capability major.minor). On failure the GpuError names
  // the driver version, the PTX ISA and target, and the JIT log (e.g. a PTX too new for the driver, issue #15).
  void loadModule(int cap_major, int cap_minor);

  CudaDriver driver_;
  ILogger& logger_;
  CUcontext context_ = nullptr;
  CUmodule module_ = nullptr;
  CUstream stream_ = nullptr;
  CUfunction kernels_[KERNEL_COUNT] = {};
  CudaMemoryPool pool_;
  ConstantCache<CuMemory> constants_;
};

// Prints the CUDA devices with their indices for --version (stdout).
void enumCudaDevices();

}  // namespace guetzli

#endif  // __USE_CUDA__

#endif  // GUETZLI_BACKEND_CUDA_CUDA_CONTEXT_H_
