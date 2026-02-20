# External Dependencies

This directory contains the CMake superbuild that downloads and builds all
third-party libraries required by Guetzli CUDA/OpenCL.

## What gets built

| Library | Version | Purpose |
|---------|---------|---------|
| zlib | 1.3.1 | Compression (required by libpng & libtiff) |
| libpng | 1.6.43 | PNG image reading |
| libjpeg-turbo | 3.0.4 | JPEG decoding (`__SUPPORT_FULL_JPEG__`) |
| libtiff | 4.6.0 | TIFF image reading |
| OpenCL-Headers | 2024.05.08 | Khronos OpenCL C headers |
| OpenCL-ICD-Loader | 2024.05.08 | OpenCL ICD Loader (libOpenCL) |

## Quick Start

```bash
cd external
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

After the build completes, headers and libraries are installed into:

```
external/install/
├── include/   (png.h, zlib.h, tiffio.h, jpeglib.h, turbojpeg.h, CL/cl.h, ...)
└── lib/       (libpng16.a, libz.a, libtiff.a, libjpeg.a, libturbojpeg.a, libOpenCL.a, ...)
```

The main Makefile automatically picks these up via `-Iexternal/install/include -Lexternal/install/lib`.

## Selective builds

You can disable individual dependencies:

```bash
cmake -B build -DBUILD_LIBTIFF=OFF -DBUILD_OPENCL=OFF ...
```

## Cleaning

```bash
rm -rf build install
```
