# Third-party libraries for guetzli.
#
# Outputs:
#   GUETZLI_DEP_INCLUDE_DIRS  extra include directories (after the in-tree ones)
#   GUETZLI_DEP_LIBRARIES     link items, in link order
#
# Libraries are taken from GUETZLI_EXTERNAL_PREFIX (the external/ superbuild) when present,
# preferring static archives there (a dylib/.so next to them would otherwise win and add a
# runtime dependency), and from the system otherwise.

set(GUETZLI_DEP_INCLUDE_DIRS "")
set(GUETZLI_DEP_LIBRARIES "")

set(_ext_lib "${GUETZLI_EXTERNAL_PREFIX}/lib")
if(EXISTS "${GUETZLI_EXTERNAL_PREFIX}/include")
  list(APPEND GUETZLI_DEP_INCLUDE_DIRS "${GUETZLI_EXTERNAL_PREFIX}/include")
  message(STATUS "guetzli: external dependencies from ${GUETZLI_EXTERNAL_PREFIX}")
else()
  message(STATUS "guetzli: ${GUETZLI_EXTERNAL_PREFIX} not found, using system libraries")
endif()

# guetzli_find_library(<var> STATIC <archive names...> SHARED <-l names...>)
function(guetzli_find_library var)
  cmake_parse_arguments(ARG "" "" "STATIC;SHARED" ${ARGN})
  foreach(name IN LISTS ARG_STATIC)
    if(EXISTS "${_ext_lib}/${name}")
      set(${var} "${_ext_lib}/${name}" CACHE FILEPATH "" FORCE)
      return()
    endif()
  endforeach()
  find_library(${var} NAMES ${ARG_SHARED})
  if(NOT ${var})
    message(FATAL_ERROR "guetzli: library ${ARG_SHARED} not found (build external/ or install it)")
  endif()
endfunction()

# Static archives must come before the archives they depend on: tiff -> jpeg, zlib; png -> zlib.
guetzli_find_library(GUETZLI_TIFF_LIBRARY STATIC libtiff.a SHARED tiff)
guetzli_find_library(GUETZLI_PNG_LIBRARY STATIC libpng16.a SHARED png16 png)
list(APPEND GUETZLI_DEP_LIBRARIES "${GUETZLI_TIFF_LIBRARY}" "${GUETZLI_PNG_LIBRARY}")
if(GUETZLI_WITH_FULL_JPEG)
  guetzli_find_library(GUETZLI_TURBOJPEG_LIBRARY STATIC libturbojpeg.a SHARED turbojpeg)
  list(APPEND GUETZLI_DEP_LIBRARIES "${GUETZLI_TURBOJPEG_LIBRARY}")
endif()
# libtiff's JPEG codec needs libjpeg even without FULL_JPEG.
guetzli_find_library(GUETZLI_JPEG_LIBRARY STATIC libjpeg.a SHARED jpeg)
guetzli_find_library(GUETZLI_ZLIB_LIBRARY STATIC libz.a libzlibstatic.a SHARED z zlib)
list(APPEND GUETZLI_DEP_LIBRARIES "${GUETZLI_JPEG_LIBRARY}" "${GUETZLI_ZLIB_LIBRARY}")

# System headers for png/tiff/jpeg when not using the superbuild.
if(NOT EXISTS "${GUETZLI_EXTERNAL_PREFIX}/include/png.h")
  find_path(GUETZLI_PNG_INCLUDE_DIR png.h PATH_SUFFIXES libpng16 libpng)
  if(GUETZLI_PNG_INCLUDE_DIR)
    list(APPEND GUETZLI_DEP_INCLUDE_DIRS "${GUETZLI_PNG_INCLUDE_DIR}")
  endif()
endif()

# CUDA: only cuda.h is needed. The driver is loaded at runtime (cuda_dynload.cpp); never link it.
if(GUETZLI_WITH_CUDA)
  file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _cuda_path)  # Windows env vars use backslashes
  file(TO_CMAKE_PATH "$ENV{CUDA_HOME}" _cuda_home)
  find_path(GUETZLI_CUDA_INCLUDE_DIR cuda.h
    HINTS "${_cuda_path}/include" "${_cuda_home}/include"
    PATHS /usr/local/cuda/include /opt/cuda/include)
  if(NOT GUETZLI_CUDA_INCLUDE_DIR)
    message(FATAL_ERROR "guetzli: cuda.h not found. Install the CUDA toolkit headers, set CUDA_PATH, "
                        "or configure with -DGUETZLI_WITH_CUDA=OFF.")
  endif()
  list(APPEND GUETZLI_DEP_INCLUDE_DIRS "${GUETZLI_CUDA_INCLUDE_DIR}")
  list(APPEND GUETZLI_DEP_LIBRARIES ${CMAKE_DL_LIBS})  # dlopen on Linux
endif()

# OpenCL: Apple framework; else the static Khronos ICD loader from the superbuild (preferred: the binary
# then needs no OpenCL.dll and finds the vendor drivers through the registry / ICD files at runtime);
# else the bundled import library for OpenCL.dll (Windows) or the system loader.
if(GUETZLI_WITH_OPENCL)
  if(APPLE)
    list(APPEND GUETZLI_DEP_LIBRARIES "-framework OpenCL")
  elseif(EXISTS "${_ext_lib}/OpenCL.a")
    list(APPEND GUETZLI_DEP_LIBRARIES "${_ext_lib}/OpenCL.a")
  elseif(EXISTS "${_ext_lib}/libOpenCL.a")
    list(APPEND GUETZLI_DEP_LIBRARIES "${_ext_lib}/libOpenCL.a" ${CMAKE_DL_LIBS})
  elseif(WIN32)
    # Import library bundled in third_party/OpenCL/lib.
    list(APPEND GUETZLI_DEP_LIBRARIES "-L${CMAKE_CURRENT_SOURCE_DIR}/third_party/OpenCL/lib" -Wl,-Bdynamic
         -lOpenCL)
  else()
    find_library(GUETZLI_OPENCL_LIBRARY OpenCL)
    if(NOT GUETZLI_OPENCL_LIBRARY)
      message(FATAL_ERROR "guetzli: OpenCL library not found (install an ICD loader, e.g. ocl-icd-opencl-dev)")
    endif()
    list(APPEND GUETZLI_DEP_LIBRARIES "${GUETZLI_OPENCL_LIBRARY}")
  endif()
  if(WIN32)
    # The ICD loader needs these system DLLs.
    list(APPEND GUETZLI_DEP_LIBRARIES -Wl,-Bdynamic -lcfgmgr32 -lole32)
  endif()
endif()
if(WIN32 AND GUETZLI_STATIC)
  # Back to static for the libraries the toolchain appends implicitly (libstdc++, winpthread, ...).
  list(APPEND GUETZLI_DEP_LIBRARIES -Wl,-Bstatic)
endif()
