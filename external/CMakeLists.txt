cmake_minimum_required(VERSION 3.18)
project(guetzli_external_deps NONE)

# =============================================================================
# Guetzli CUDA/OpenCL — External Dependency Superbuild
#
# Downloads, builds, and installs all third-party libraries into
#   external/install/{include,lib}
#
# Usage:
#   cd external
#   cmake -B build -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_INSTALL_PREFIX=<abs-path>/external/install
#   cmake --build build --config Release
#
# ExternalProject_Add is used (not FetchContent) because libjpeg-turbo
# explicitly forbids add_subdirectory() integration.
# =============================================================================

include(ExternalProject)

# Default install prefix inside external/install
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "" FORCE)
endif()
set(EP_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")

# Build type passthrough
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Common CMake args for all external projects
set(EP_CMAKE_ARGS
  -DCMAKE_INSTALL_PREFIX:PATH=${EP_INSTALL_DIR}
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
  -DBUILD_SHARED_LIBS:BOOL=OFF
)
# Propagate compilers if set
if(CMAKE_C_COMPILER)
  list(APPEND EP_CMAKE_ARGS -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER})
endif()
if(CMAKE_CXX_COMPILER)
  list(APPEND EP_CMAKE_ARGS -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER})
endif()
# Propagate generator
if(CMAKE_GENERATOR)
  set(EP_GENERATOR -G "${CMAKE_GENERATOR}")
else()
  set(EP_GENERATOR "")
endif()

# Options to select which deps to build
option(BUILD_ZLIB          "Build zlib"                ON)
option(BUILD_LIBPNG        "Build libpng"              ON)
option(BUILD_LIBTIFF       "Build libtiff"             ON)
option(BUILD_LIBJPEG_TURBO "Build libjpeg-turbo"       ON)
option(BUILD_OPENCL        "Build OpenCL headers+ICD"  ON)

# =============================================================================
# zlib
# =============================================================================
if(BUILD_ZLIB)
  ExternalProject_Add(ep_zlib
    URL      https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
    URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
    CMAKE_ARGS ${EP_CMAKE_ARGS}
      -DZLIB_BUILD_EXAMPLES:BOOL=OFF
      -DSKIP_INSTALL_FILES:BOOL=ON
    ${EP_GENERATOR}
  )
  set(ZLIB_DEP ep_zlib)
else()
  set(ZLIB_DEP "")
endif()

# =============================================================================
# libpng (depends on zlib)
# =============================================================================
if(BUILD_LIBPNG)
  ExternalProject_Add(ep_libpng
    DEPENDS  ${ZLIB_DEP}
    URL      https://github.com/glennrp/libpng/archive/refs/tags/v1.6.43.tar.gz
    URL_HASH SHA256=fecc95b46cf05e8e3fc8a414750e0ba5aad00d89e9fdf175e94ff041caf1a03a
    CMAKE_ARGS ${EP_CMAKE_ARGS}
      -DPNG_SHARED:BOOL=OFF
      -DPNG_STATIC:BOOL=ON
      -DPNG_TESTS:BOOL=OFF
      -DPNG_TOOLS:BOOL=OFF
      -DCMAKE_PREFIX_PATH:PATH=${EP_INSTALL_DIR}
    ${EP_GENERATOR}
  )
  set(LIBPNG_DEP ep_libpng)
else()
  set(LIBPNG_DEP "")
endif()

# =============================================================================
# libjpeg-turbo (requires ExternalProject — forbids add_subdirectory)
# =============================================================================
if(BUILD_LIBJPEG_TURBO)
  ExternalProject_Add(ep_libjpeg_turbo
    URL      https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.0.4/libjpeg-turbo-3.0.4.tar.gz
    URL_HASH SHA256=99130559e7d62e8d695f2c0eaeef912c5828d5b84a0537dcb24c9678c9d5b76b
    CMAKE_ARGS ${EP_CMAKE_ARGS}
      -DENABLE_SHARED:BOOL=OFF
      -DENABLE_STATIC:BOOL=ON
      -DWITH_TURBOJPEG:BOOL=ON
      -DWITH_JAVA:BOOL=OFF
    ${EP_GENERATOR}
  )
  set(LIBJPEG_DEP ep_libjpeg_turbo)
else()
  set(LIBJPEG_DEP "")
endif()

# =============================================================================
# libtiff (depends on zlib and libjpeg-turbo)
# =============================================================================
if(BUILD_LIBTIFF)
  ExternalProject_Add(ep_libtiff
    DEPENDS  ${ZLIB_DEP} ${LIBJPEG_DEP}
    URL      https://download.osgeo.org/libtiff/tiff-4.6.0.tar.gz
    URL_HASH SHA256=88b3979e6d5c7e32b50d7ec72fb15af724f6ab2cbf7e10880c360a77e4b5d99a
    CMAKE_ARGS ${EP_CMAKE_ARGS}
      -Dtiff-tools:BOOL=OFF
      -Dtiff-tests:BOOL=OFF
      -Dtiff-contrib:BOOL=OFF
      -Dtiff-docs:BOOL=OFF
      -Dtiff-install:BOOL=ON
      -DCMAKE_PREFIX_PATH:PATH=${EP_INSTALL_DIR}
    ${EP_GENERATOR}
  )
  set(LIBTIFF_DEP ep_libtiff)
else()
  set(LIBTIFF_DEP "")
endif()

# =============================================================================
# OpenCL Headers + ICD Loader
# =============================================================================
if(BUILD_OPENCL)
  ExternalProject_Add(ep_opencl_headers
    URL https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2024.05.08.tar.gz
    URL_HASH SHA256=3c3dd236d35f4960028f4f58ce8d963fb63f3d50251d1e9854b76f1caab9a309
    CMAKE_ARGS ${EP_CMAKE_ARGS}
    ${EP_GENERATOR}
  )

  ExternalProject_Add(ep_opencl_icd_loader
    DEPENDS  ep_opencl_headers
    URL https://github.com/KhronosGroup/OpenCL-ICD-Loader/archive/refs/tags/v2024.05.08.tar.gz
    URL_HASH SHA256=eb2c9fde125ffc58f418d62ad83131ba686cccedcb390cc7e6bb81cc5ef2bd4f
    CMAKE_ARGS ${EP_CMAKE_ARGS}
      -DCMAKE_PREFIX_PATH:PATH=${EP_INSTALL_DIR}
    ${EP_GENERATOR}
  )
endif()

# =============================================================================
message(STATUS "=== External dependencies configured ===")
message(STATUS "  Install prefix: ${EP_INSTALL_DIR}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
