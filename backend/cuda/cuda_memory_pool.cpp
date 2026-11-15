// Memory pool for CUDA.
//
// Author: ianhuang@tencent.com
#ifdef __USE_CUDA__

#include "backend/cuda/cuda_memory_pool.h"

#include "backend/cuda/cuda_context.h"

namespace guetzli {

CudaMemoryPool::~CudaMemoryPool() {
  // Busy blocks cannot exist here (every buffer returns its block when destroyed); free everything regardless.
  for (const Block& block : blocks_) driver_.memFree(block.mem);
}

CUdeviceptr CudaMemoryPool::allocate(size_t s, const void* init, bool zero) {
  alloc_count_++;
  total_mem_request_ += s;
  Block* candidate = nullptr;
  for (Block& block : blocks_) {
    if (block.status == Status::Idle && block.size >= s) {
      candidate = &block;
      break;
    }
  }
  CUdeviceptr mem = 0;
  if (candidate != nullptr) {
    candidate->status = Status::Busy;
    candidate->used = s;
    mem = candidate->mem;
  } else {
    CUdeviceptr new_mem = 0;
    CUresult err = driver_.memAlloc(&new_mem, s);
    if (err == CUDA_ERROR_OUT_OF_MEMORY) {
      // Idle pooled blocks are only freed at shutdown; release them and retry once.
      drain();
      err = driver_.memAlloc(&new_mem, s);
    }
    if (err != CUDA_SUCCESS) {
      throw GpuError("CUDA allocation of " + std::to_string(s) + " bytes failed: " + TranslateCUDAError(err));
    }
    blocks_.push_back(Block{Status::Busy, s, s, new_mem});
    blocks_.sort([](const Block& a, const Block& b) { return a.size < b.size; });
    mem = new_mem;
  }
  if (init) {
    CU_CHECK(driver_.memcpyHtoDAsync(mem, init, s, stream_));
  } else if (zero) {
    CU_CHECK(driver_.memsetD8Async(mem, 0, s, stream_));
  }
  return mem;
}

void CudaMemoryPool::release(CUdeviceptr mem) noexcept {
  for (Block& block : blocks_) {
    if (block.mem == mem) {
      block.status = Status::Idle;
      block.used = 0;
      return;
    }
  }
}

void CudaMemoryPool::drain() noexcept {
  size_t total_mem = 0;
  const size_t total_block = blocks_.size();
  for (auto it = blocks_.begin(); it != blocks_.end();) {
    if (it->status == Status::Idle) {
      total_mem += it->size;
      driver_.memFree(it->mem);
      it = blocks_.erase(it);
    } else {
      ++it;
    }
  }

  logger_.log(LogLevel::Debug,
              "mem_pool has %zu blocks, and total pool memory is:%f kb, total memory request:%f kb, total alloc "
              "count:%zu.\n",
              total_block, (float)(total_mem) / 1024, (float)(total_mem_request_) / 1024, alloc_count_);
}

}  // namespace guetzli

#endif  // __USE_CUDA__
