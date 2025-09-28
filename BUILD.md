# Guetzli CUDA/OpenCL Build Guide

This document provides comprehensive instructions for building the Guetzli CUDA/OpenCL project using the portable Makefile system.

## Quick Start

### Windows
```cmd
# Build with all features (CUDA + OpenCL)
build.bat

# Build debug version
build.bat debug

# Build static library
build.bat static

# Clean build files
build.bat clean
```

### Linux
```bash
# Build with all features (CUDA + OpenCL)
./build.sh

# Build debug version
./build.sh debug

# Build static library
./build.sh static

# Clean build files
./build.sh clean
```

## Prerequisites

### Required Tools

#### Windows
- **MinGW-w64** or **MSYS2** - Download from [msys2.org](https://www.msys2.org/)
- **Python 3** - Download from [python.org](https://www.python.org/downloads/)
- **Git** (optional) - For cloning the repository

#### Linux
- **GCC/G++** - Install build-essential package
- **Python 3** - Usually pre-installed
- **Git** (optional) - For cloning the repository

### Optional Dependencies

#### CUDA Support
- **CUDA Toolkit** - Download from [NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit)
- Set `CUDA_PATH` environment variable to CUDA installation directory

#### OpenCL Support
- **OpenCL SDK** - Choose from:
  - Intel OpenCL SDK
  - AMD OpenCL SDK
  - NVIDIA OpenCL SDK
- Set `OPENCL_SDK_PATH` environment variable to SDK installation directory

#### PNG Support
- **libpng development files**
  - Windows: Install via vcpkg or download pre-built libraries
  - Linux: Install via package manager (libpng-dev)

## Installation Instructions

### Windows Setup

1. **Install MSYS2**
   ```cmd
   # Download and install MSYS2 from https://www.msys2.org/
   # Open MSYS2 terminal and install MinGW-w64
   pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make
   pacman -S mingw-w64-x86_64-pkg-config
   ```

2. **Add to PATH**
   - Add `C:\msys64\mingw64\bin` to your Windows PATH
   - Add `C:\msys64\usr\bin` to your Windows PATH

3. **Install Python**
   - Download Python 3 from python.org
   - Make sure to check "Add Python to PATH" during installation

4. **Install CUDA (Optional)**
   ```cmd
   # Download and install CUDA Toolkit
   # Set environment variable
   set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0
   ```

5. **Install OpenCL (Optional)**
   ```cmd
   # Download Intel OpenCL SDK or AMD OpenCL SDK
   # Set environment variable
   set OPENCL_SDK_PATH=C:\Program Files (x86)\Intel\OpenCL SDK
   ```

### Linux Setup

1. **Install Build Tools**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install build-essential python3 pkg-config libpng-dev
   
   # CentOS/RHEL
   sudo yum groupinstall "Development Tools"
   sudo yum install python3 pkgconfig libpng-devel
   
   # Fedora
   sudo dnf groupinstall "Development Tools"
   sudo dnf install python3 pkgconfig libpng-devel
   ```

2. **Install CUDA (Optional)**
   ```bash
   # Download CUDA Toolkit from NVIDIA
   # Follow installation instructions
   # Add to PATH
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

3. **Install OpenCL (Optional)**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install opencl-headers ocl-icd-opencl-dev
   
   # CentOS/RHEL
   sudo yum install opencl-headers ocl-icd-devel
   
   # Fedora
   sudo dnf install opencl-headers ocl-icd-devel
   ```

## Build Options

### Makefile Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `CONFIG` | Build configuration | `release` | `debug`, `release` |
| `FEATURES` | Enabled features | `CUDA OPENCL` | `CUDA`, `OPENCL`, or both |

### Build Targets

| Target | Description |
|--------|-------------|
| `all` | Build executable (default) |
| `static` | Build static library |
| `cuda-headers` | Generate CUDA header files |
| `clean` | Remove all build files |
| `help` | Show help information |

### Examples

#### Basic Builds
```bash
# Build release executable with all features
make

# Build debug executable
make CONFIG=debug

# Build static library
make static

# Build with OpenCL only
make FEATURES=OPENCL

# Build with CUDA only
make FEATURES=CUDA

# Build without GPU acceleration
make FEATURES=
```

#### Advanced Builds
```bash
# Debug build with OpenCL only
make CONFIG=debug FEATURES=OPENCL

# Release static library with CUDA
make CONFIG=release FEATURES=CUDA static

# Clean and rebuild
make clean && make
```

## Output Files

### Executable
- **Windows**: `bin/release/guetzli.exe` or `bin/debug/guetzli.exe`
- **Linux**: `bin/release/guetzli` or `bin/debug/guetzli`

### Static Library
- **Windows**: `bin/release/libguetzli_static.a` or `bin/debug/libguetzli_static.a`
- **Linux**: `bin/release/libguetzli_static.a` or `bin/debug/libguetzli_static.a`

## Troubleshooting

### Common Issues

#### "make: command not found"
- **Windows**: Install MSYS2 and add MinGW-w64 to PATH
- **Linux**: Install build-essential package

#### "python: command not found"
- **Windows**: Install Python and add to PATH
- **Linux**: Install python3 package

#### "nvcc: command not found"
- Install CUDA Toolkit
- Set CUDA_PATH environment variable
- Add CUDA bin directory to PATH

#### "OpenCL headers not found"
- Install OpenCL SDK
- Set OPENCL_SDK_PATH environment variable
- Install OpenCL development packages

#### "libpng not found"
- **Windows**: Install libpng via vcpkg or download pre-built libraries
- **Linux**: Install libpng-dev package

#### Build errors related to CUDA/OpenCL
- Check that CUDA_PATH and OPENCL_SDK_PATH are set correctly
- Verify that the respective SDKs are properly installed
- Try building without GPU features: `make FEATURES=`

### Debug Information

#### Enable Verbose Output
```bash
# Show detailed compilation commands
make VERBOSE=1
```

#### Check Dependencies
```bash
# Windows
build.bat help

# Linux
./build.sh help
```

## Performance Notes

- **CUDA**: Provides the best performance for NVIDIA GPUs
- **OpenCL**: Works with AMD, Intel, and NVIDIA GPUs
- **CPU-only**: Slowest but most compatible
- **Memory**: Guetzli requires ~300MB per 1MPix of input image

## Cross-Platform Compatibility

The Makefile is designed to work on both Windows and Linux with minimal changes:

- **Windows**: Uses MinGW-w64 GCC and Windows-specific paths
- **Linux**: Uses system GCC and standard Unix paths
- **Portable**: Uses environment variables for SDK paths

## Contributing

When modifying the build system:

1. Test on both Windows and Linux
2. Update this documentation
3. Ensure backward compatibility
4. Add appropriate error messages

## License

This build system follows the same license as the main Guetzli project.

