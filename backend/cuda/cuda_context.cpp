// CUDA device context.
//
// Author: strongtu@tencent.com
#ifdef __USE_CUDA__

#include "backend/cuda/cuda_context.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

#include "kernels/embedded_kernels.h"

namespace guetzli {

namespace {

// The first line of text starting with prefix (e.g. ".version 7.8"), or "?".
std::string firstLineStartingWith(const std::string& text, const std::string& prefix) {
  size_t pos = 0;
  while (pos < text.size()) {
    size_t end = text.find('\n', pos);
    if (end == std::string::npos) {
      end = text.size();
    }
    if (text.compare(pos, prefix.size(), prefix) == 0) {
      std::string line = text.substr(pos, end - pos);
      while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
        line.pop_back();
      }
      return line;
    }
    pos = end + 1;
  }
  return "?";
}

}  // namespace

void throwCudaError(CUresult err, const char* where) {
  throw GpuError(std::string("CUDA call failed in ") + where + ": " + translateCudaError(err));
}

void CudaContext::loadModule(int cap_major, int cap_minor) {
  // c_str() is NUL-terminated, as cuModuleLoadDataEx expects for PTX text.
  const std::string ptx = kernels::cudaPtx();

  char error_log[8192] = {};
  // The driver API passes option values as void*.
  // NOLINTBEGIN(performance-no-int-to-ptr)
  CUjit_option options[] = {CU_JIT_CACHE_MODE, CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void* values[] = {reinterpret_cast<void*>(static_cast<uintptr_t>(CU_JIT_CACHE_OPTION_CA)), error_log,
                    reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof(error_log)))};
  // NOLINTEND(performance-no-int-to-ptr)
  const auto start = std::chrono::steady_clock::now();
  const CUresult err = driver_.module_load_data_ex(&module_, ptx.c_str(), 3, options, values);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  if (err == CUDA_SUCCESS) {
    logger_.log(LogLevel::Debug, "CUDA module loaded in %.0f ms (JIT-compiled unless cached by the driver)\n", ms);
    return;
  }
  int driver_version = 0;
  driver_.driver_get_version(&driver_version);
  std::string message = std::string("cuModuleLoadDataEx: ") + translateCudaError(err) + "; driver CUDA " +
                        std::to_string(driver_version / 1000) + "." + std::to_string((driver_version % 1000) / 10) +
                        ", device sm_" + std::to_string(cap_major) + std::to_string(cap_minor) + ", PTX " +
                        firstLineStartingWith(ptx, ".version") + " " + firstLineStartingWith(ptx, ".target");
  if (error_log[0] != '\0') {
    error_log[sizeof(error_log) - 1] = '\0';
    std::string log(error_log);
    while (!log.empty() && (log.back() == '\n' || log.back() == '\r')) {
      log.pop_back();
    }
    message += "; JIT log: " + log;
  }
  throw GpuError(message);
}

CuBuffer::CuBuffer(CuBuffer&& other) noexcept : pool_(other.pool_), mem_(other.mem_) {
  other.pool_ = nullptr;
  other.mem_ = 0;
}

CuBuffer& CuBuffer::operator=(CuBuffer&& other) noexcept {
  if (this != &other) {
    if (pool_ != nullptr) {
      pool_->release(mem_);
    }
    pool_ = std::exchange(other.pool_, nullptr);
    mem_ = std::exchange(other.mem_, 0);
  }
  return *this;
}

CuBuffer::~CuBuffer() {
  if (pool_ != nullptr) {
    pool_->release(mem_);
  }
}

CudaContext::CudaContext(const CudaDriver& driver, const GpuOptions& options, ILogger& logger)
    : driver_(driver), logger_(logger), pool_(driver_, logger) {
  try {
    CU_CHECK(driver_.init(0));
    CUdevice dev = 0;

    // Use user-specified device index if valid
    if (options.device >= 0) {
      int device_count = 0;
      CU_CHECK(driver_.device_get_count(&device_count));
      if (options.device < device_count) {
        CU_CHECK(driver_.device_get(&dev, options.device));
      } else {
        logger_.log(LogLevel::Warning,
                    "Warning: CUDA device %d not found (only %d devices available), using default device 0\n",
                    options.device, device_count);
        CU_CHECK(driver_.device_get(&dev, 0));
      }
    } else {
      CU_CHECK(driver_.device_get(&dev, 0));
    }

    CU_CHECK(driver_.ctx_create(&context_, CU_CTX_SCHED_AUTO, dev));

    char name[1024] = {0};
    int proc_count = 0;
    int thread_count = 0;
    int cap_major = 0;
    int cap_minor = 0;
    driver_.device_get_name(name, sizeof(name), dev);
    driver_.device_get_attribute(&cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    driver_.device_get_attribute(&cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    driver_.device_get_attribute(&proc_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    driver_.device_get_attribute(&thread_count, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
    // Always printed, to stderr (as before the logger existed).
    logger_.log(LogLevel::Warning, "CUDA Adapter:%s Ver%d.%d MP %d MaxThread Per MP %d)\r\n", name, cap_major,
                cap_minor, proc_count, thread_count);
    if ((proc_count == 0) || (thread_count == 0)) {
      throw GpuError("CUDA device reports no multiprocessors");
    }

    {
      loadModule(cap_major, cap_minor);
    }

    for (int i = 0; i < KERNEL_COUNT; i++) {
      const CUresult err = driver_.module_get_function(&kernels_[i], module_, kKernelEntryPoints[i]);
      if (err != CUDA_SUCCESS) {
        throw GpuError(std::string("CUDA kernel not found: ") + kKernelEntryPoints[i] + " (" + translateCudaError(err) +
                       ")");
      }
    }

    driver_.ctx_set_cache_config(CU_FUNC_CACHE_PREFER_SHARED);
    // cuCtxSetSharedMemConfig is deprecated and no longer needed on modern GPUs
    // as they have a fixed shared memory bank size

    CU_CHECK(driver_.stream_create(&stream_, 0));
    pool_.setStream(stream_);
  } catch (...) {
    destroy();
    throw;
  }
}

CudaContext::~CudaContext() {
  destroy();
}

void CudaContext::destroy() noexcept {
  // Teardown order: free device memory, unload module, destroy stream, then context.
  constants_.clear();
  pool_.drain();
  if (module_ != nullptr) {
    driver_.module_unload(module_);
  }
  if (stream_ != nullptr) {
    driver_.stream_destroy(stream_);
  }
  if (context_ != nullptr) {
    driver_.ctx_destroy(context_);
  }
  module_ = nullptr;
  stream_ = nullptr;
  context_ = nullptr;
}

void CudaContext::launch(CUfunction f, unsigned int gx, unsigned int gy, unsigned int bx, unsigned int by,
                         const void* args[]) {
  CU_CHECK(driver_.launch_kernel(f, gx, gy, 1, bx, by, 1, 0, stream_, const_cast<void**>(args), nullptr));
}

void CudaContext::finish() {
  CU_CHECK(driver_.stream_synchronize(stream_));
}

void CudaContext::copy(cu_mem dst, cu_mem src, size_t bytes) {
  CU_CHECK(driver_.memcpy_dtod_async(dst, src, bytes, stream_));
}

void CudaContext::readAsync(void* dst, cu_mem src, size_t bytes) {
  CU_CHECK(driver_.memcpy_dtoh_async(dst, src, bytes, stream_));
}

CuBuffer CudaContext::zeroed(size_t bytes) {
  return {&pool_, pool_.allocate(bytes, nullptr, true)};
}

CuBuffer CudaContext::uninitialized(size_t bytes) {
  return {&pool_, pool_.allocate(bytes, nullptr, false)};
}

CuBuffer CudaContext::upload(size_t bytes, const void* data) {
  return {&pool_, pool_.allocate(bytes, data, true)};
}

CuChannels CudaContext::zeroedChannels(size_t bytes) {
  CuChannels c;
  for (CuBuffer& b : c.ch) {
    b = zeroed(bytes);
  }
  return c;
}

CuChannels CudaContext::uninitializedChannels(size_t bytes) {
  CuChannels c;
  for (CuBuffer& b : c.ch) {
    b = uninitialized(bytes);
  }
  return c;
}

CuChannels CudaContext::uploadChannels(size_t bytes, const void* c0, const void* c1, const void* c2) {
  CuChannels c;
  c.ch[0] = upload(bytes, c0);
  c.ch[1] = upload(bytes, c1);
  c.ch[2] = upload(bytes, c2);
  return c;
}

CuMemory CudaContext::allocateMemory(size_t bytes) {
  cu_mem mem = 0;
  CUresult err = driver_.mem_alloc(&mem, bytes);
  if (err == CUDA_ERROR_OUT_OF_MEMORY) {
    pool_.drain();
    err = driver_.mem_alloc(&mem, bytes);
  }
  CU_CHECK(err);
  return {driver_, mem};
}

CuMemory CudaContext::uploadMemory(size_t bytes, const void* data) {
  CuMemory memory = allocateMemory(bytes);
  // Synchronous: data may be a caller's local.
  CU_CHECK(driver_.memcpy_htod_async(memory.get(), data, bytes, stream_));
  CU_CHECK(driver_.stream_synchronize(stream_));
  return memory;
}

void enumCudaDevices() {
  NullLogger quiet;
  const std::optional<CudaDriver> driver = CudaDriver::load(quiet);
  if (!driver) {
    fprintf(stdout, "CUDA:\n  (driver not available)\n\n");
    return;
  }

  CUresult err = driver->init(0);
  if (err != CUDA_SUCCESS) {
    fprintf(stdout, "CUDA:\n  (initialization failed)\n\n");
    return;
  }

  int device_count = 0;
  err = driver->device_get_count(&device_count);
  if (err != CUDA_SUCCESS || device_count == 0) {
    fprintf(stdout, "CUDA:\n  (no devices found)\n\n");
    return;
  }

  fprintf(stdout, "CUDA:\n");
  for (int i = 0; i < device_count; i++) {
    CUdevice dev = 0;
    err = driver->device_get(&dev, i);
    if (err != CUDA_SUCCESS) {
      continue;
    }

    char name[256] = {0};
    driver->device_get_name(name, sizeof(name), dev);
    fprintf(stdout, "  %d: %s\n", i, name);
  }
  fprintf(stdout, "\n");
}

const char* translateCudaError(CUresult error_code) {
  switch (error_code) {
    case CUDA_SUCCESS:
      return "CUDA_SUCCESS";
    case CUDA_ERROR_INVALID_VALUE:
      return "CUDA_ERROR_INVALID_VALUE";
    case CUDA_ERROR_OUT_OF_MEMORY:
      return "CUDA_ERROR_OUT_OF_MEMORY";
    case CUDA_ERROR_NOT_INITIALIZED:
      return "CUDA_ERROR_NOT_INITIALIZED";
    case CUDA_ERROR_DEINITIALIZED:
      return "CUDA_ERROR_DEINITIALIZED";
    case CUDA_ERROR_PROFILER_DISABLED:
      return "CUDA_ERROR_PROFILER_DISABLED";
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
      return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:
      return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
      return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
    case CUDA_ERROR_NO_DEVICE:
      return "CUDA_ERROR_NO_DEVICE";
    case CUDA_ERROR_INVALID_DEVICE:
      return "CUDA_ERROR_INVALID_DEVICE";
    case CUDA_ERROR_INVALID_IMAGE:
      return "CUDA_ERROR_INVALID_IMAGE";
    case CUDA_ERROR_INVALID_CONTEXT:
      return "CUDA_ERROR_INVALID_CONTEXT";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
      return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
    case CUDA_ERROR_MAP_FAILED:
      return "CUDA_ERROR_MAP_FAILED";
    case CUDA_ERROR_UNMAP_FAILED:
      return "CUDA_ERROR_UNMAP_FAILED";
    case CUDA_ERROR_ARRAY_IS_MAPPED:
      return "CUDA_ERROR_ARRAY_IS_MAPPED";
    case CUDA_ERROR_ALREADY_MAPPED:
      return "CUDA_ERROR_ALREADY_MAPPED";
    case CUDA_ERROR_NO_BINARY_FOR_GPU:
      return "CUDA_ERROR_NO_BINARY_FOR_GPU";
    case CUDA_ERROR_ALREADY_ACQUIRED:
      return "CUDA_ERROR_ALREADY_ACQUIRED";
    case CUDA_ERROR_NOT_MAPPED:
      return "CUDA_ERROR_NOT_MAPPED";
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
      return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
      return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
    case CUDA_ERROR_ECC_UNCORRECTABLE:
      return "CUDA_ERROR_ECC_UNCORRECTABLE";
    case CUDA_ERROR_UNSUPPORTED_LIMIT:
      return "CUDA_ERROR_UNSUPPORTED_LIMIT";
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
      return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
    case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
      return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
    case CUDA_ERROR_INVALID_PTX:
      return "CUDA_ERROR_INVALID_PTX";
    case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
      return "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
    // case CUDA_ERROR_NVLINK_UNCORRECTABLE: return "CUDA_ERROR_NVLINK_UNCORRECTABLE";
    case CUDA_ERROR_INVALID_SOURCE:
      return "CUDA_ERROR_INVALID_SOURCE";
    case CUDA_ERROR_FILE_NOT_FOUND:
      return "CUDA_ERROR_FILE_NOT_FOUND";
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
      return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
      return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
    case CUDA_ERROR_OPERATING_SYSTEM:
      return "CUDA_ERROR_OPERATING_SYSTEM";
    case CUDA_ERROR_INVALID_HANDLE:
      return "CUDA_ERROR_INVALID_HANDLE";
    case CUDA_ERROR_NOT_FOUND:
      return "CUDA_ERROR_NOT_FOUND";
    case CUDA_ERROR_NOT_READY:
      return "CUDA_ERROR_NOT_READY";
    case CUDA_ERROR_ILLEGAL_ADDRESS:
      return "CUDA_ERROR_ILLEGAL_ADDRESS";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
      return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
    case CUDA_ERROR_LAUNCH_TIMEOUT:
      return "CUDA_ERROR_LAUNCH_TIMEOUT";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
      return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
      return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
      return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
      return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
    case CUDA_ERROR_CONTEXT_IS_DESTROYED:
      return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
    case CUDA_ERROR_ASSERT:
      return "CUDA_ERROR_ASSERT";
    case CUDA_ERROR_TOO_MANY_PEERS:
      return "CUDA_ERROR_TOO_MANY_PEERS";
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
      return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
      return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
    case CUDA_ERROR_HARDWARE_STACK_ERROR:
      return "CUDA_ERROR_HARDWARE_STACK_ERROR";
    case CUDA_ERROR_ILLEGAL_INSTRUCTION:
      return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
    case CUDA_ERROR_MISALIGNED_ADDRESS:
      return "CUDA_ERROR_MISALIGNED_ADDRESS";
    case CUDA_ERROR_INVALID_ADDRESS_SPACE:
      return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
    case CUDA_ERROR_INVALID_PC:
      return "CUDA_ERROR_INVALID_PC";
    case CUDA_ERROR_LAUNCH_FAILED:
      return "CUDA_ERROR_LAUNCH_FAILED";
    case CUDA_ERROR_NOT_PERMITTED:
      return "CUDA_ERROR_NOT_PERMITTED";
    case CUDA_ERROR_NOT_SUPPORTED:
      return "CUDA_ERROR_NOT_SUPPORTED";
    case CUDA_ERROR_UNKNOWN:
    default:
      return "CUDA_ERROR_UNKNOWN";
  }
}

}  // namespace guetzli

#endif  // __USE_CUDA__
