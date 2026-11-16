// OpenCL device context.
//
// Author: strongtu@tencent.com
//         ianhuang@tencent.com
#ifdef __USE_OPENCL__

#include "backend/opencl/opencl_context.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "kernels/embedded_kernels.h"
#include "third_party/OpenCL-Wrapper/opencl.hpp"

namespace guetzli {

void throwOpenClError(cl_int err, const char* where) {
  throw GpuError(std::string("OpenCL call failed in ") + where + ": " + TranslateOpenCLError(err));
}

ClBuffer::ClBuffer(ClBuffer&& other) noexcept : owner_(other.owner_), mem_(other.mem_) {
  other.owner_ = nullptr;
  other.mem_ = nullptr;
}

ClBuffer& ClBuffer::operator=(ClBuffer&& other) noexcept {
  if (this != &other) {
    if (owner_) owner_->release(mem_);
    owner_ = std::exchange(other.owner_, nullptr);
    mem_ = std::exchange(other.mem_, nullptr);
  }
  return *this;
}

ClBuffer::~ClBuffer() {
  if (owner_) owner_->release(mem_);
}

ClMemory::~ClMemory() {
  if (mem_) clReleaseMemObject(mem_);
}

OpenClContext::OpenClContext(const GpuOptions& options, ILogger& logger) : logger_(logger) {
  MaybeInjectGpuFailure("OpenCL");

  std::vector<Device_Info> devices = get_devices(true);
  if (devices.empty()) {
    throw GpuError("No OpenCL devices found");
  }

  // Select device: use user-specified index if valid, otherwise pick best by flops
  Device_Info best_device;
  if (options.device >= 0 && options.device < (int)devices.size()) {
    best_device = select_device_with_id((uint)options.device, devices);
  } else {
    if (options.device >= 0) {
      logger_.log(LogLevel::Warning,
                  "Warning: OpenCL device %d not found (only %d devices available), using best device\n",
                  options.device, (int)devices.size());
    }
    best_device = select_device_with_most_flops(devices);
  }

  const std::string source = kernels::openClSource();
  Clock clock;
  device_ = std::make_unique<Device>(best_device, source);
  print_info("OpenCL C code compilation time: ~" + to_string(clock.stop() * 1000, 3) + "ms\n");

  queue_ = device_->get_cl_queue().get();
  context_ = device_->get_cl_context().get();
  const cl_program program = device_->get_cl_program().get();

  for (int i = 0; i < KERNEL_COUNT; i++) {
    cl_int err = CL_SUCCESS;
    kernels_[i] = clCreateKernel(program, kKernelEntryPoints[i], &err);
    CL_CHECK(err);
  }

  print_info("OpenCL created\n");

  if (logger_.enabled(LogLevel::Debug)) printDeviceCapabilities();
}

OpenClContext::~OpenClContext() {
  for (cl_kernel kernel : kernels_) {
    if (kernel) clReleaseKernel(kernel);
  }
  // Pooled buffers; constants_ and device_ (context, queue, program) are released by their destructors.
  for (const auto& [mem, size] : buffer_sizes_) clReleaseMemObject(mem);
}

size_t OpenClContext::maxWorkGroupSize(cl_kernel kernel) {
  auto it = max_work_group_.find(kernel);
  if (it != max_work_group_.end()) return it->second;

  cl_device_id dev = nullptr;
  CL_CHECK(clGetCommandQueueInfo(queue_, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr));
  size_t wg = 0;
  CL_CHECK(clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg), &wg, nullptr));
  if (wg == 0) wg = 1;
  max_work_group_[kernel] = wg;
  return wg;
}

cl_mem OpenClContext::createBuffer(size_t bytes) {
  cl_int err = CL_SUCCESS;
  cl_mem mem = clCreateBuffer(context_, CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE || err == CL_OUT_OF_RESOURCES) {
    // Idle pooled buffers still hold device memory: free them and retry once.
    drainPool();
    mem = clCreateBuffer(context_, CL_MEM_READ_WRITE, bytes, nullptr, &err);
  }
  CL_CHECK(err);
  return mem;
}

ClBuffer OpenClContext::uninitialized(size_t bytes) {
  cl_mem mem = nullptr;
  auto it = free_buffers_.find(bytes);
  if (it != free_buffers_.end() && !it->second.empty()) {
    mem = it->second.back();
    it->second.pop_back();
  } else {
    mem = createBuffer(bytes);
    buffer_sizes_[mem] = bytes;
  }
  return ClBuffer(this, mem);
}

ClBuffer OpenClContext::zeroed(size_t bytes) {
  ClBuffer buffer = uninitialized(bytes);
  const cl_char zero = 0;
  // No clFinish: the pattern is copied at enqueue time and the in-order queue orders the fill before any later use.
  CL_CHECK(clEnqueueFillBuffer(queue_, buffer.get(), &zero, sizeof(zero), 0, bytes, 0, nullptr, nullptr));
  return buffer;
}

ClBuffer OpenClContext::upload(size_t bytes, const void* data) {
  ClBuffer buffer = uninitialized(bytes);
  CL_CHECK(clEnqueueWriteBuffer(queue_, buffer.get(), CL_TRUE, 0, bytes, data, 0, nullptr, nullptr));
  return buffer;
}

ClChannels OpenClContext::zeroedChannels(size_t bytes) {
  ClChannels c;
  for (ClBuffer& b : c.ch) b = zeroed(bytes);
  return c;
}

ClChannels OpenClContext::uninitializedChannels(size_t bytes) {
  ClChannels c;
  for (ClBuffer& b : c.ch) b = uninitialized(bytes);
  return c;
}

ClChannels OpenClContext::uploadChannels(size_t bytes, const void* c0, const void* c1, const void* c2) {
  ClChannels c;
  c.ch[0] = upload(bytes, c0);
  c.ch[1] = upload(bytes, c1);
  c.ch[2] = upload(bytes, c2);
  return c;
}

void OpenClContext::release(cl_mem mem) noexcept {
  auto it = buffer_sizes_.find(mem);
  if (it != buffer_sizes_.end()) free_buffers_[it->second].push_back(mem);
}

void OpenClContext::drainPool() noexcept {
  for (auto& [size, idle] : free_buffers_) {
    for (cl_mem mem : idle) {
      buffer_sizes_.erase(mem);
      clReleaseMemObject(mem);
    }
  }
  free_buffers_.clear();
}

ClMemory OpenClContext::createMemory(size_t bytes) {
  return ClMemory(createBuffer(bytes));
}

ClMemory OpenClContext::uploadMemory(size_t bytes, const void* data) {
  ClMemory memory = createMemory(bytes);
  CL_CHECK(clEnqueueWriteBuffer(queue_, memory.get(), CL_TRUE, 0, bytes, data, 0, nullptr, nullptr));
  return memory;
}

void OpenClContext::printDeviceCapabilities() const {
  const Device_Info& deviceInfo = device_->info;
  // Get platform information from the device's context
  cl_platform_id platform = nullptr;
  cl_int err = clGetContextInfo(deviceInfo.cl_context(), CL_CONTEXT_PLATFORM, sizeof(platform), &platform, nullptr);

  std::string platformName = "Unknown";
  std::string platformVersion = "Unknown";
  std::string platformVendor = "Unknown";

  if (err == CL_SUCCESS && platform) {
    size_t nameLen = 0;
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &nameLen);
    std::vector<char> platformNameVec(nameLen + 1);
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, nameLen, &platformNameVec[0], NULL);
    platformName = std::string(platformNameVec.data());

    size_t versionLen = 0;
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, NULL, &versionLen);
    std::vector<char> platformVersionVec(versionLen);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, versionLen, &platformVersionVec[0], NULL);
    platformVersion = std::string(platformVersionVec.data());

    size_t vendorLen = 0;
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, NULL, &vendorLen);
    std::vector<char> platformVendorVec(vendorLen);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorLen, &platformVendorVec[0], NULL);
    platformVendor = std::string(platformVendorVec.data());
  }

  // Get additional device information not available in Device_Info
  cl_device_id device = deviceInfo.cl_device();

  size_t maxWorkGroupSize = 0;  // CL_DEVICE_MAX_WORK_GROUP_SIZE is a size_t
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);

  size_t maxWorkItemSizes[3] = {0};
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), &maxWorkItemSizes, NULL);

  cl_bool compilerAvailable = CL_FALSE;
  clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compilerAvailable), &compilerAvailable, NULL);

  cl_bool linkerAvailable = CL_FALSE;
  clGetDeviceInfo(device, CL_DEVICE_LINKER_AVAILABLE, sizeof(linkerAvailable), &linkerAvailable, NULL);

  cl_device_fp_config fpConfig = 0;
  clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fpConfig), &fpConfig, NULL);

  char deviceVersion[256] = {0};
  clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(deviceVersion) - 1, deviceVersion, NULL);

  cl_device_exec_capabilities execCapabilities = 0;
  clGetDeviceInfo(device, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(execCapabilities), &execCapabilities, NULL);

  // Print comprehensive device report
  logger_.log(LogLevel::Debug, "=== OpenCL Device Capabilities Report ===\n");
  logger_.log(LogLevel::Debug, "Platform: %s\n", platformName.c_str());
  logger_.log(LogLevel::Debug, "Platform Version: %s\n", platformVersion.c_str());
  logger_.log(LogLevel::Debug, "Platform Vendor: %s\n", platformVendor.c_str());
  logger_.log(LogLevel::Debug, "Device: %s\n", deviceInfo.name.c_str());
  logger_.log(LogLevel::Debug, "Device Vendor: %s\n", deviceInfo.vendor.c_str());
  logger_.log(LogLevel::Debug, "Device Version: %s\n", deviceVersion);
  logger_.log(LogLevel::Debug, "OpenCL C Version: %s\n", deviceInfo.opencl_c_version.c_str());
  logger_.log(LogLevel::Debug, "Driver Version: %s\n", deviceInfo.driver_version.c_str());
  logger_.log(LogLevel::Debug, "\n=== Memory Information ===\n");
  logger_.log(LogLevel::Debug, "Global Memory Size: %u MB\n", deviceInfo.memory);
  logger_.log(LogLevel::Debug, "Local Memory Size: %u KB\n", deviceInfo.local_cache);
  logger_.log(LogLevel::Debug, "Max Memory Allocation: %u MB\n", deviceInfo.max_global_buffer);
  logger_.log(LogLevel::Debug, "\n=== Compute Capabilities ===\n");
  logger_.log(LogLevel::Debug, "Max Compute Units: %u\n", deviceInfo.compute_units);
  logger_.log(LogLevel::Debug, "Max Work Group Size: %zu\n", maxWorkGroupSize);
  logger_.log(LogLevel::Debug, "Max Work Item Sizes: [%zu, %zu, %zu]\n", maxWorkItemSizes[0], maxWorkItemSizes[1],
              maxWorkItemSizes[2]);
  logger_.log(LogLevel::Debug, "Clock Frequency: %u MHz\n", deviceInfo.clock_frequency);
  logger_.log(LogLevel::Debug, "Compiler Available: %s\n", compilerAvailable ? "Yes" : "No");
  logger_.log(LogLevel::Debug, "Linker Available: %s\n", linkerAvailable ? "Yes" : "No");
  logger_.log(LogLevel::Debug, "Double Precision Support: %s\n",
              fpConfig != 0 ? "Yes" : "No");  // non-zero DOUBLE_FP_CONFIG => fp64 supported
  logger_.log(LogLevel::Debug, "FP64 Capable: %s\n", deviceInfo.is_fp64_capable ? "Yes" : "No");
  logger_.log(LogLevel::Debug, "FP32 Capable: %s\n", deviceInfo.is_fp32_capable ? "Yes" : "No");
  logger_.log(LogLevel::Debug, "FP16 Capable: %s\n", deviceInfo.is_fp16_capable ? "Yes" : "No");
  logger_.log(LogLevel::Debug, "Device Type: %s%s\n", deviceInfo.is_cpu ? "CPU " : "", deviceInfo.is_gpu ? "GPU" : "");
  logger_.log(LogLevel::Debug, "Estimated Performance: %.2f TFLOPs\n", deviceInfo.tflops);
  logger_.log(LogLevel::Debug, "Execution Capabilities: %s%s%s\n", (execCapabilities & CL_EXEC_KERNEL) ? "Kernel " : "",
              (execCapabilities & CL_EXEC_NATIVE_KERNEL) ? "Native " : "", "");
  logger_.log(LogLevel::Debug, "==========================================\n\n");
}

const char* TranslateOpenCLError(cl_int errorCode) {
  switch (errorCode) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "CL_MISALIGNED_SUB_BUFFER_OFFSET";  //-13
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";  //-14
    case CL_COMPILE_PROGRAM_FAILURE:
      return "CL_COMPILE_PROGRAM_FAILURE";  //-15
    case CL_LINKER_NOT_AVAILABLE:
      return "CL_LINKER_NOT_AVAILABLE";  //-16
    case CL_LINK_PROGRAM_FAILURE:
      return "CL_LINK_PROGRAM_FAILURE";  //-17
    case CL_DEVICE_PARTITION_FAILED:
      return "CL_DEVICE_PARTITION_FAILED";  //-18
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";  //-19
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";  //-63
    case CL_INVALID_PROPERTY:
      return "CL_INVALID_PROPERTY";  //-64
    case CL_INVALID_IMAGE_DESCRIPTOR:
      return "CL_INVALID_IMAGE_DESCRIPTOR";  //-65
    case CL_INVALID_COMPILER_OPTIONS:
      return "CL_INVALID_COMPILER_OPTIONS";  //-66
    case CL_INVALID_LINKER_OPTIONS:
      return "CL_INVALID_LINKER_OPTIONS";  //-67
    case CL_INVALID_DEVICE_PARTITION_COUNT:
      return "CL_INVALID_DEVICE_PARTITION_COUNT";  //-68
      //    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
      //    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70

    default:
      return "UNKNOWN ERROR CODE";
  }
}

void enumOpenClDevices() {
  std::vector<cl::Platform> cl_platforms;
  cl::Platform::get(&cl_platforms);

  if (cl_platforms.empty()) {
    fprintf(stdout, "OpenCL:\n  (no platforms found)\n\n");
    return;
  }

  fprintf(stdout, "OpenCL:\n");
  uint id = 0;
  for (uint i = 0; i < (uint)cl_platforms.size(); i++) {
    std::vector<cl::Device> cl_devices;
    cl_platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &cl_devices);
    for (uint j = 0; j < (uint)cl_devices.size(); j++) {
      std::string name = cl_devices[j].getInfo<CL_DEVICE_NAME>();
      // trim whitespace
      size_t start = name.find_first_not_of(" \t\r\n");
      size_t end = name.find_last_not_of(" \t\r\n");
      if (start != std::string::npos) name = name.substr(start, end - start + 1);
      fprintf(stdout, "  %u: %s\n", id, name.c_str());
      id++;
    }
  }
  fprintf(stdout, "\n");
}

}  // namespace guetzli

#endif  // __USE_OPENCL__
