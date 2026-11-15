// Memory pool for CUDA: allocations are slow, so released blocks are kept and reused for requests of the same or a
// smaller size.
//
// Author: ianhuang@tencent.com
#ifndef GUETZLI_BACKEND_CUDA_CUDA_MEMORY_POOL_H_
#define GUETZLI_BACKEND_CUDA_CUDA_MEMORY_POOL_H_

#ifdef __USE_CUDA__

#include <cuda.h>

#include <cstddef>
#include <list>

#include "backend/cuda/cuda_driver.h"
#include "logging/logger.h"

namespace guetzli {

class CudaMemoryPool {
 public:
  CudaMemoryPool(const CudaDriver& driver, ILogger& logger) : driver_(driver), logger_(logger) {}
  ~CudaMemoryPool();
  CudaMemoryPool(const CudaMemoryPool&) = delete;
  CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

  void setStream(CUstream stream) { stream_ = stream; }

  // init != nullptr: upload; init == nullptr: zero-fill, unless zero == false (contents undefined). Throws GpuError.
  CUdeviceptr allocate(size_t size, const void* init, bool zero);
  void release(CUdeviceptr mem) noexcept;
  // Frees every idle block.
  void drain() noexcept;

 private:
  enum class Status { Idle, Busy };
  struct Block {
    Status status;
    size_t size;
    size_t used;
    CUdeviceptr mem;
  };

  const CudaDriver& driver_;
  ILogger& logger_;
  CUstream stream_ = nullptr;
  std::list<Block> blocks_;  // sorted by size
  size_t alloc_count_ = 0;
  size_t total_mem_request_ = 0;
};

}  // namespace guetzli

#endif  // __USE_CUDA__

#endif  // GUETZLI_BACKEND_CUDA_CUDA_MEMORY_POOL_H_
