# Embedded kernel headers (docs/design/modernization.md §6.2 and §7).
#
# kernels/guetzli_cl_src.h and kernels/guetzli_cu_ptx.h are committed and consumed as-is; the normal
# build never regenerates them. Explicit targets:
#   check-kernel-headers  the committed OpenCL header matches kernels/guetzli.cl (run in CI)
#   kernel-headers        regenerate both headers (PTX needs the pinned CUDA toolkit)
#
# Kernel-header options:
#   GUETZLI_KERNEL_NVCC_VERSION  required nvcc release (default 11.8; issue #15)
#   GUETZLI_CUDA_ARCH            PTX target (default compute_75)
#   GUETZLI_NVCC_CCBIN           host compiler directory for nvcc (-ccbin), e.g. an MSVC 14.29 bin dir
#   GUETZLI_NVCC_EXTRA_FLAGS     extra nvcc flags, e.g. -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH

set(GUETZLI_KERNEL_NVCC_VERSION "11.8" CACHE STRING "nvcc release required to build the embedded PTX")
set(GUETZLI_CUDA_ARCH "compute_75" CACHE STRING "PTX target architecture of the embedded kernels")
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
          -DCUDA_ARCH=${GUETZLI_CUDA_ARCH}
          -DNVCC_CCBIN=${GUETZLI_NVCC_CCBIN}
          "-DNVCC_EXTRA_FLAGS=${GUETZLI_NVCC_EXTRA_FLAGS}"
          -P "${_kh_script}"
  DEPENDS embed_kernel
  COMMENT "Regenerating the embedded kernel headers (commit both headers afterwards)"
  VERBATIM)
