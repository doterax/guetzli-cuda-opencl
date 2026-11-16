# Embedded kernel headers (docs/design/modernization.md §6.2 and §7).
#
# kernels/guetzli_cl_src.h, kernels/guetzli_cu_fatbin.h and kernels/guetzli_cu_info.h are committed and consumed as-is; the normal
# build never regenerates them. Explicit targets:
#   check-kernel-headers  the committed OpenCL header matches kernels/guetzli.cl (run in CI)
#   kernel-headers        regenerate all of them (the fatbin needs the pinned CUDA toolkit)
#
# Kernel-header options:
#   GUETZLI_KERNEL_NVCC_VERSION  required nvcc release (default 11.8; issue #15)
#   GUETZLI_CUDA_PTX_ARCH        PTX in the fatbin, JIT-compiled for GPUs without native code (default compute_52:
#                                Maxwell and newer)
#   GUETZLI_CUDA_SASS_ARCHS      native code in the fatbin (default 61;75;86;89: Pascal, Turing, Ampere, Ada), so
#                                those GPUs skip the multi-second first-run JIT
#   GUETZLI_NVCC_CCBIN           host compiler directory for nvcc (-ccbin), e.g. an MSVC 14.29 bin dir
#   GUETZLI_NVCC_EXTRA_FLAGS     extra nvcc flags, e.g. -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH

set(GUETZLI_KERNEL_NVCC_VERSION "11.8" CACHE STRING "nvcc release required to build the embedded PTX")
set(GUETZLI_CUDA_PTX_ARCH "compute_52" CACHE STRING "PTX target of the embedded CUDA fatbin")
set(GUETZLI_CUDA_SASS_ARCHS "61;75;86;89" CACHE STRING "Native (SASS) architectures of the embedded CUDA fatbin")
set(GUETZLI_NVCC_CCBIN "" CACHE PATH "Host compiler (directory) passed to nvcc -ccbin")
set(GUETZLI_NVCC_EXTRA_FLAGS "" CACHE STRING "Extra nvcc flags for the PTX build")

# Host tool that LZO-compresses a kernel into a header (not part of the default build).
add_executable(embed_kernel EXCLUDE_FROM_ALL kernels/tools/embed_kernel.cpp third_party/minilzo/minilzo.c)
target_include_directories(embed_kernel PRIVATE third_party/minilzo)
set_target_properties(embed_kernel PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

set(_kh_script "${CMAKE_CURRENT_SOURCE_DIR}/cmake/kernel_headers_script.cmake")
set(_kh_common
  -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
  -DWORK_DIR=${CMAKE_BINARY_DIR}/kernel_headers
  -DEMBED=$<TARGET_FILE:embed_kernel>)

add_custom_target(check-kernel-headers
  COMMAND "${CMAKE_COMMAND}" ${_kh_common} -DMODE=check -P "${_kh_script}"
  DEPENDS embed_kernel
  COMMENT "Checking the committed OpenCL kernel header against kernels/guetzli.cl"
  VERBATIM)

file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _kh_cuda_path)
file(TO_CMAKE_PATH "$ENV{CUDA_HOME}" _kh_cuda_home)
find_program(GUETZLI_NVCC nvcc HINTS "${_kh_cuda_path}/bin" "${_kh_cuda_home}/bin" /usr/local/cuda/bin)
add_custom_target(kernel-headers
  COMMAND "${CMAKE_COMMAND}" ${_kh_common} -DMODE=generate
          -DNVCC=${GUETZLI_NVCC}
          -DNVCC_VERSION=${GUETZLI_KERNEL_NVCC_VERSION}
          -DCUDA_PTX_ARCH=${GUETZLI_CUDA_PTX_ARCH}
          "-DCUDA_SASS_ARCHS=${GUETZLI_CUDA_SASS_ARCHS}"
          -DNVCC_CCBIN=${GUETZLI_NVCC_CCBIN}
          "-DNVCC_EXTRA_FLAGS=${GUETZLI_NVCC_EXTRA_FLAGS}"
          -P "${_kh_script}"
  DEPENDS embed_kernel
  COMMENT "Regenerating the embedded kernel headers (commit both headers afterwards)"
  VERBATIM)
