/*
* OpenCL Manager
*
* Author: strongtu@tencent.com
*         ianhuang@tencent.com
*/

#include "ocl.h"
#include <string.h>
#include <string>
#include <vector>
#include <string>
#include "clguetzli/clguetzli_cl_src.h"
#include "lzodec.h"
#include <stdexcept>
//#define LOG
//#define UTILITIES_FILE
#include "third_party/OpenCL-Wrapper/opencl.hpp"

using std::string;

#ifdef __USE_OPENCL__

void PrintDeviceCapabilities(cl_platform_id platform, cl_device_id device);

string opencl_c_container() {
	LzoDec decompressed(clguetzli_cl_src_lzo, sizeof(clguetzli_cl_src_lzo));
	return string((char*)decompressed.getData(), decompressed.getSize());
}

//inline Device_Info select_device_oclgrind_or_with_most_flops(const vector<Device_Info>& devices = get_devices()) { // returns device with best floating-point performance
//	string simulator = "Oclgrind Simulator";
//	for (uint i = 0u; i < (uint)devices.size(); i++) { // find device with highest (estimated) floating point performance
//		if (devices[i].name.find(simulator) != std::string::npos) {
//			return devices[i];
//		}
//	}
//	float best_value = 0.0f;
//	uint best_i = 0u;
//	for (uint i = 0u; i < (uint)devices.size(); i++) { // find device with highest (estimated) floating point performance
//		if (devices[i].tflops > best_value) {
//			best_value = devices[i].tflops;
//			best_i = i;
//		}
//	}
//	return devices[best_i];
//}

ocl_args_d_t& getOcl()
{
    static bool bInit = false;
    static ocl_args_d_t ocl;

    if (bInit == true) return ocl;

    bInit = true;

	vector<Device_Info> devices = get_devices(true);

	if (devices.empty()) {
		LogError("No OpenCL devices found");
		throw std::runtime_error("Failed to create OpenCL program: No OpenCL devices found");
	}

	// Select the best device (highest performance)
	Device_Info best_device = select_device_with_most_flops(devices);
	

	// Check if device is AMD for optimization
	bool isAmd = contains(to_lower(best_device.vendor), "amd") ||
		contains(to_lower(best_device.vendor), "advanced micro devices");

	string opencl_source = opencl_c_container();
	Clock clock;
	static Device device(best_device, opencl_source);
	std::string info = "OpenCL C code compilation time: ~" + to_string(clock.stop() * 1000, 3) + "ms\n";
	print_info(info);

    
	ocl.isAmd = isAmd;

	ocl.program = device.get_cl_program().get();
	ocl.commandQueue = device.get_cl_queue().get();
	ocl.context = device.get_cl_context().get();
	ocl.device = &device;

	cl_int  err = 0;

    ocl.kernel[KERNEL_CONVOLUTION] = clCreateKernel(ocl.program, "clConvolutionEx", &err);
	LOG_CL_RESULT(err);
	
    ocl.kernel[KERNEL_CONVOLUTIONX] = clCreateKernel(ocl.program, "clConvolutionXEx", &err);
	LOG_CL_RESULT(err);
    ocl.kernel[KERNEL_CONVOLUTIONY] = clCreateKernel(ocl.program, "clConvolutionYEx", &err);
	LOG_CL_RESULT(err);
    ocl.kernel[KERNEL_SQUARESAMPLE] = clCreateKernel(ocl.program, "clSquareSampleEx", &err);
	LOG_CL_RESULT(err);
    ocl.kernel[KERNEL_OPSINDYNAMICSIMAGE] = clCreateKernel(ocl.program, "clOpsinDynamicsImageEx", &err);
	LOG_CL_RESULT(err);
    ocl.kernel[KERNEL_MASKHIGHINTENSITYCHANGE] = clCreateKernel(ocl.program, "clMaskHighIntensityChangeEx", &err);
	LOG_CL_RESULT(err);
    ocl.kernel[KERNEL_EDGEDETECTOR] = clCreateKernel(ocl.program, "clEdgeDetectorMapEx", &err);
	LOG_CL_RESULT(err);
    ocl.kernel[KERNEL_BLOCKDIFFMAP] = clCreateKernel(ocl.program, "clBlockDiffMapEx", &err);
	LOG_CL_RESULT(err);
    ocl.kernel[KERNEL_EDGEDETECTORLOWFREQ] = clCreateKernel(ocl.program, "clEdgeDetectorLowFreqEx", &err);
	LOG_CL_RESULT(err);
    ocl.kernel[KERNEL_DIFFPRECOMPUTE] = clCreateKernel(ocl.program, "clDiffPrecomputeEx", &err);
	LOG_CL_RESULT(err);
	ocl.kernel[KERNEL_SCALEIMAGE] = clCreateKernel(ocl.program, "clScaleImageEx", &err);
	LOG_CL_RESULT(err);
	ocl.kernel[KERNEL_AVERAGE5X5] = clCreateKernel(ocl.program, "clAverage5x5Ex", &err);
	LOG_CL_RESULT(err);
	ocl.kernel[KERNEL_MINSQUAREVAL] = clCreateKernel(ocl.program, "clMinSquareValEx", &err);
	LOG_CL_RESULT(err);
	ocl.kernel[KERNEL_DOMASK] = clCreateKernel(ocl.program, "clDoMaskEx", &err);
	LOG_CL_RESULT(err);
	ocl.kernel[KERNEL_COMBINECHANNELS] = clCreateKernel(ocl.program, "clCombineChannelsEx", &err);
	LOG_CL_RESULT(err);
	ocl.kernel[KERNEL_UPSAMPLESQUAREROOT] = clCreateKernel(ocl.program, "clUpsampleSquareRootEx", &err);
	LOG_CL_RESULT(err);
	ocl.kernel[KERNEL_REMOVEBORDER] = clCreateKernel(ocl.program, "clRemoveBorderEx", &err);
	LOG_CL_RESULT(err);
    ocl.kernel[KERNEL_ADDBORDER] = clCreateKernel(ocl.program, "clAddBorderEx", &err);
    ocl.kernel[KERNEL_COMPUTEBLOCKZEROINGORDER] = clCreateKernel(ocl.program, "clComputeBlockZeroingOrderEx", &err);
	LOG_CL_RESULT(err);
	ocl.kernel[KERNEL_COPYFROMJPEGCOMPONENT] = clCreateKernel(ocl.program, "clCopyFromJpegComponentEx", &err);
	LOG_CL_RESULT(err);
	ocl.kernel[KERNEL_APPLYGLOBALQUANTIZATION] = clCreateKernel(ocl.program, "clApplyGlobalQuantizationEx", &err);
	LOG_CL_RESULT(err);
	ocl.kernel[KERNEL_COMPONENTSTOPIXELS] = clCreateKernel(ocl.program, "clComponentsToPixels", &err);
	LOG_CL_RESULT(err);
	ocl.kernel[KERNEL_COMPONENTSTOPIXELS_EX1] = clCreateKernel(ocl.program, "clComponentsToPixelsEx1", &err);
	LOG_CL_RESULT(err);
	ocl.kernel[KERNEL_COMPONENTSTOPIXELS_EX2] = clCreateKernel(ocl.program, "clComponentsToPixelsEx2", &err);
	LOG_CL_RESULT(err);
	ocl.kernel[KERNEL_COLORTRANSFORMYCBCRTORGB] = clCreateKernel(ocl.program, "clColorTransformYCbCrToRGB", &err);
	LOG_CL_RESULT(err);

	print_info("OpenCL created\n");

    return ocl;
}

ocl_args_d_t::ocl_args_d_t() :
	context(NULL),
	commandQueue(NULL),
	device(NULL),
	program(NULL)
{
	for (int i = 0; i < KERNEL_COUNT; i++)
	{
		kernel[i] = NULL;
	}
}

ocl_args_d_t::~ocl_args_d_t()
{
	cl_int err = CL_SUCCESS;
	for (int i = 0; i < KERNEL_COUNT; i++)
	{
		err = clReleaseKernel(kernel[i]);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
		}
	}

	if (program)
	{
		err = clReleaseProgram(program);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseProgram returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
	if (commandQueue)
	{
		err = clReleaseCommandQueue(commandQueue);
		if (CL_SUCCESS != err)
		{
			LogError("Error: clReleaseCommandQueue returned '%s'.\n", TranslateOpenCLError(err));
		}
	}
}

cl_mem ocl_args_d_t::allocMem(size_t s, const void *init)
{
	cl_int err = 0;
	cl_mem mem = clCreateBuffer(this->context, CL_MEM_READ_WRITE, s, nullptr, &err);
    LOG_CL_RESULT(err);
    if (!mem) return NULL;
    
    // init memory
    if (init)
    {
        err = clEnqueueWriteBuffer(this->commandQueue, mem, CL_FALSE, 0, s, init, 0, NULL, NULL);
        LOG_CL_RESULT(err);
        err = clFinish(this->commandQueue);
        LOG_CL_RESULT(err);
    }
    else
    {
        cl_char cc = 0;
        err = clEnqueueFillBuffer(this->commandQueue, mem, &cc, sizeof(cc), 0, s / sizeof(cc), 0, NULL, NULL);
        LOG_CL_RESULT(err);
        err = clFinish(this->commandQueue);
        LOG_CL_RESULT(err);
    }

	return mem;
}

ocl_channels ocl_args_d_t::allocMemChannels(size_t s, const void *c0, const void *c1, const void *c2)
{
	const void *c[3] = { c0, c1, c2 };

	ocl_channels img;
    for (int i = 0; i < 3; i++)
    {
        img.ch[i] = allocMem(s, c[i]);
    }

	return img;
}

void ocl_args_d_t::releaseMemChannels(ocl_channels &rgb)
{
    for (int i = 0; i < 3; i++)
    {
        clReleaseMemObject(rgb.ch[i]);
        rgb.ch[i] = NULL;
    }
}

const char* TranslateOpenCLError(cl_int errorCode)
{
	switch (errorCode)
	{
	case CL_SUCCESS:                            return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
	case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
	case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
	case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
	case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
	case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
	case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
	case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
	case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
	case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
	case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
																												//    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
																												//    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70    

	default:
		return "UNKNOWN ERROR CODE";
	}
}


/*
* Check whether an OpenCL platform is the required platform
* (based on the platform's name)
*/
bool CheckPreferredPlatformMatch(cl_platform_id platform, const char* preferredPlatform)
{
	size_t stringLength = 0;
	cl_int err = CL_SUCCESS;
	bool match = false;

	// In order to read the platform's name, we first read the platform's name string length (param_value is NULL).
	// The value returned in stringLength
	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &stringLength);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME length returned '%s'.\n", TranslateOpenCLError(err));
		return false;
	}

	// Now, that we know the platform's name string length, we can allocate enough space before read it
	std::vector<char> platformName(stringLength);

	// Read the platform's name string
	// The read value returned in platformName
	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, stringLength, &platformName[0], NULL);
	if (CL_SUCCESS != err)
	{
		LogError("Error: clGetplatform_ids() to get CL_PLATFORM_NAME returned %s.\n", TranslateOpenCLError(err));
		return false;
	}

	// Now check if the platform's name is the required one
	if (strstr(&platformName[0], preferredPlatform) != 0)
	{
		// The checked platform is the one we're looking for
		match = true;
	}

	return match;
}

/*
* Print detailed device capabilities and information
*/
void PrintDeviceCapabilities(cl_platform_id platform, cl_device_id device)
{
	cl_int err = CL_SUCCESS;
	size_t stringLength = 0;
	
	// Platform information
	size_t nameLen = 0;
	clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &nameLen);
	std::vector<char> platformName(nameLen + 1);
	clGetPlatformInfo(platform, CL_PLATFORM_NAME, nameLen, &platformName[0], NULL);
	platformName[nameLen] = 0;

	clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, NULL, &stringLength);
	std::vector<char> platformVersion(stringLength);
	clGetPlatformInfo(platform, CL_PLATFORM_VERSION, stringLength, &platformVersion[0], NULL);

	clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, NULL, &stringLength);
	std::vector<char> platformVendor(stringLength);
	clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, stringLength, &platformVendor[0], NULL);

	// Device information
	clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &stringLength);
	std::vector<char> deviceName(stringLength);
	clGetDeviceInfo(device, CL_DEVICE_NAME, stringLength, &deviceName[0], NULL);

	clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, NULL, &stringLength);
	std::vector<char> deviceVendor(stringLength);
	clGetDeviceInfo(device, CL_DEVICE_VENDOR, stringLength, &deviceVendor[0], NULL);

	clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &stringLength);
	std::vector<char> deviceVersion(stringLength);
	clGetDeviceInfo(device, CL_DEVICE_VERSION, stringLength, &deviceVersion[0], NULL);

	clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &stringLength);
	std::vector<char> deviceOpenCLCVersion(stringLength);
	clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, stringLength, &deviceOpenCLCVersion[0], NULL);

	clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &stringLength);
	std::vector<char> driverVersion(stringLength);
	clGetDeviceInfo(device, CL_DRIVER_VERSION, stringLength, &driverVersion[0], NULL);

	// Device capabilities
	cl_ulong globalMemSize = 0;
	clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, NULL);

	cl_ulong localMemSize = 0;
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);

	cl_ulong maxMemAllocSize = 0;
	clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxMemAllocSize), &maxMemAllocSize, NULL);

	cl_uint maxComputeUnits = 0;
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);

	cl_uint maxWorkGroupSize = 0;
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);

	size_t maxWorkItemSizes[3] = {0};
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), &maxWorkItemSizes, NULL);

	cl_bool compilerAvailable = CL_FALSE;
	clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compilerAvailable), &compilerAvailable, NULL);

	cl_bool linkerAvailable = CL_FALSE;
	clGetDeviceInfo(device, CL_DEVICE_LINKER_AVAILABLE, sizeof(linkerAvailable), &linkerAvailable, NULL);

	cl_device_fp_config fpConfig = 0;
	clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fpConfig), &fpConfig, NULL);

	cl_device_exec_capabilities execCapabilities = 0;
	clGetDeviceInfo(device, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(execCapabilities), &execCapabilities, NULL);

	// Print comprehensive device report
	LogError("=== OpenCL Device Capabilities Report ===\n");
	LogError("Platform: %s\n", platformName.data());
	LogError("Platform Version: %s\n", platformVersion.data());
	LogError("Platform Vendor: %s\n", platformVendor.data());
	LogError("Device: %s\n", deviceName.data());
	LogError("Device Vendor: %s\n", deviceVendor.data());
	LogError("Device Version: %s\n", deviceVersion.data());
	LogError("OpenCL C Version: %s\n", deviceOpenCLCVersion.data());
	LogError("Driver Version: %s\n", driverVersion.data());
	LogError("\n=== Memory Information ===\n");
	LogError("Global Memory Size: %llu MB\n", globalMemSize / (1024 * 1024));
	LogError("Local Memory Size: %llu KB\n", localMemSize / 1024);
	LogError("Max Memory Allocation: %llu MB\n", maxMemAllocSize / (1024 * 1024));
	LogError("\n=== Compute Capabilities ===\n");
	LogError("Max Compute Units: %u\n", maxComputeUnits);
	LogError("Max Work Group Size: %u\n", maxWorkGroupSize);
	LogError("Max Work Item Sizes: [%zu, %zu, %zu]\n", maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);
	LogError("Compiler Available: %s\n", compilerAvailable ? "Yes" : "No");
	LogError("Linker Available: %s\n", linkerAvailable ? "Yes" : "No");
	LogError("Double Precision Support: %s\n", (fpConfig & CL_FP_FMA) ? "Yes" : "No");
	LogError("Execution Capabilities: %s%s%s\n", 
		(execCapabilities & CL_EXEC_KERNEL) ? "Kernel " : "",
		(execCapabilities & CL_EXEC_NATIVE_KERNEL) ? "Native " : "",
		"");
	LogError("==========================================\n\n");
}

bool supportsOpenCl()
{
	vector<Device_Info> devices = get_devices(true);

	if (devices.empty()) {
		return false;
	}

	return true;
}

#endif