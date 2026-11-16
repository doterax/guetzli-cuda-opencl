<p align="center"><img src="https://cloud.githubusercontent.com/assets/203457/24553916/1f3f88b6-162c-11e7-990a-731b2560f15c.png" alt="Guetzli" width="64"></p>

# Guetzli CUDA/OpenCL

[![CI](https://github.com/doterax/guetzli-cuda-opencl/actions/workflows/ci.yml/badge.svg)](https://github.com/doterax/guetzli-cuda-opencl/actions/workflows/ci.yml)

GPU-accelerated JPEG encoder based on [Google Guetzli](https://github.com/google/guetzli). Guetzli generates JPEG images that are typically 20-30% smaller than libjpeg at equivalent visual quality.

This fork adds **CUDA** and **OpenCL** support to dramatically speed up encoding:

| Method | Example (750×400) |
|--------|-------------------|
| Original CPU | 14.7 s |
| Procedure Optimised (`--c`) | 8.2 s |
| OpenCL (`--opencl`) | 1.5 s |
| CUDA (`--cuda`) | 0.8 s |

The binary auto-detects the best available backend at runtime. On systems
without an NVIDIA GPU, CUDA is simply skipped — no driver required to start
the program.

## Quick Start

```bash
# Build dependencies (one-time)
cmake -S external -B external/build -DCMAKE_BUILD_TYPE=Release
cmake --build external/build

# Build and test (Windows/MinGW: add -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++)
cmake --preset release
cmake --build --preset release
ctest --preset release

# Run (build/release/bin/guetzli)
guetzli [--quality Q] [--verbose] input.png output.jpg
```

See [BUILD.md](BUILD.md) for full build instructions, prerequisites, and
CI/CD documentation.

## Usage

```bash
guetzli [--quality Q] [--verbose] original.png output.jpg
guetzli [--quality Q] [--verbose] original.jpg output.jpg   # requires FULL_JPEG
guetzli [--quality Q] [--verbose] original.tiff output.jpg
```

You can also force a specific backend:

```bash
guetzli --cuda   input.png output.jpg   # Force CUDA
guetzli --opencl input.png output.jpg   # Force OpenCL
guetzli --c      input.png output.jpg   # CPU with procedure optimisation
guetzli --auto   input.png output.jpg   # Auto-detect best backend (default)
```

**Notes:**
- Guetzli uses ~300 MB of memory per 1 MPix of input.
- Guetzli assumes sRGB input with a gamma of 2.2.
- Prefer uncompressed (PNG/TIFF) input for best results.
- JPEG alpha channels are not supported; PNG alpha is composited on black.

## Supported Platforms

| Platform | CUDA | OpenCL | Tested |
|----------|------|--------|--------|
| Windows x64 (MinGW-w64 / MSYS2) | ✅ | ✅ | CI |
| Linux x64 (GCC) | ✅ | ✅ | CI |
| macOS arm64 (Apple Silicon) | — | ✅ | CI |

## Building

| Step | Command |
|------|---------|
| Build external deps | `cmake -S external -B external/build -DCMAKE_BUILD_TYPE=Release && cmake --build external/build` |
| Release build | `cmake --preset release && cmake --build --preset release` |
| Debug build | `cmake --preset debug && cmake --build --preset debug` |
| OpenCL-only | `cmake --preset release -DGUETZLI_WITH_CUDA=OFF` |
| Run tests | `ctest --preset release` |

Full details (prerequisites, GitHub Actions setup, troubleshooting) are in
[BUILD.md](BUILD.md).

## GitHub Actions CI/CD

Every pull request is built and tested on Linux, Windows (MSYS2), and macOS.
Pushing a `v*` tag (e.g. `v1.3.0`) creates a GitHub Release with pre-built
binaries for all platforms.

See [BUILD.md § GitHub Actions CI/CD](BUILD.md#github-actions-cicd) for setup
instructions.

## Project Structure

```
app/              Command line and composition root
config/, logging/ Configuration value types, logger
core/             Encoder pipeline, image I/O
guetzli/          JPEG data, reader/writer, output image, butteraugli comparator
backend/          CPU, OpenCL, CUDA and check-mode backends (GPU host code)
kernels/          The GPU kernel source (OpenCL C, also compiled to PTX) and its embedded headers
clguetzli/        Optimized CPU butteraugli, host-side kernel types
third_party/      Bundled: butteraugli, minilzo, OpenCL headers, CLI11, doctest
external/         CMake superbuild for zlib, libpng, libtiff, libjpeg-turbo,
                  OpenCL ICD loader
tests/            Smoke-test suite + sample images
.github/          CI/CD workflow
```

## Contributing

1. Fork and create a feature branch.
2. Ensure `ctest --preset release` passes on at least one platform.
3. Open a pull request — CI will build on all three platforms.
4. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgements

- [Google Guetzli](https://github.com/google/guetzli) — original JPEG
  encoder.
- Original CUDA/OpenCL acceleration by strongtu, ianhuang, chriskzhou,
  stephendeng @ Tencent.
