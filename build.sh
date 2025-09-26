#!/bin/bash
# Guetzli CUDA/OpenCL Build Script for Linux
# This script helps set up the environment and build the project

set -e

echo "Guetzli CUDA/OpenCL Build Script"
echo "================================="
echo

# Check if make is available (skip if help is requested)
if [ "$1" != "help" ]; then
    if ! command -v make &> /dev/null; then
        echo "ERROR: make is not found in PATH"
        echo "Please install build-essential package:"
        echo "  Ubuntu/Debian: sudo apt-get install build-essential"
        echo "  CentOS/RHEL: sudo yum groupinstall 'Development Tools'"
        echo "  Fedora: sudo dnf groupinstall 'Development Tools'"
        exit 1
    fi
fi

# Check if Python is available (skip if help is requested)
if [ "$1" != "help" ]; then
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        echo "ERROR: python is not found in PATH"
        echo "Please install Python:"
        echo "  Ubuntu/Debian: sudo apt-get install python3"
        echo "  CentOS/RHEL: sudo yum install python3"
        echo "  Fedora: sudo dnf install python3"
        exit 1
    fi
fi

# Use python3 if python is not available
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Check for CUDA
if [ -n "$CUDA_PATH" ]; then
    echo "Found CUDA_PATH: $CUDA_PATH"
    if [ ! -f "$CUDA_PATH/bin/nvcc" ]; then
        echo "WARNING: nvcc not found in $CUDA_PATH/bin/"
        echo "CUDA support may not work properly"
    fi
else
    if [ -f "/usr/local/cuda/bin/nvcc" ]; then
        echo "Found CUDA in /usr/local/cuda/"
        export CUDA_PATH="/usr/local/cuda"
    else
        echo "WARNING: CUDA not found"
        echo "CUDA support will be disabled"
        echo "To enable CUDA, install CUDA Toolkit and set CUDA_PATH"
    fi
fi

# Check for OpenCL
if [ -n "$OPENCL_SDK_PATH" ]; then
    echo "Found OPENCL_SDK_PATH: $OPENCL_SDK_PATH"
else
    if [ -f "/usr/include/CL/cl.h" ]; then
        echo "Found OpenCL headers in /usr/include/CL/"
    else
        echo "WARNING: OpenCL headers not found"
        echo "OpenCL support may not work properly"
        echo "Install OpenCL development package:"
        echo "  Ubuntu/Debian: sudo apt-get install opencl-headers ocl-icd-opencl-dev"
        echo "  CentOS/RHEL: sudo yum install opencl-headers ocl-icd-devel"
        echo "  Fedora: sudo dnf install opencl-headers ocl-icd-devel"
    fi
fi

# Check for libpng
if ! pkg-config --exists libpng; then
    echo "WARNING: libpng development files not found"
    echo "Install libpng development package:"
    echo "  Ubuntu/Debian: sudo apt-get install libpng-dev"
    echo "  CentOS/RHEL: sudo yum install libpng-devel"
    echo "  Fedora: sudo dnf install libpng-devel"
fi

echo

# Show help function
show_help() {
    echo "Usage: ./build.sh [options] [target]"
    echo
    echo "Options:"
    echo "  debug          - Build in debug mode"
    echo "  release        - Build in release mode (default)"
    echo "  --no-cuda      - Disable CUDA support"
    echo "  --no-opencl    - Disable OpenCL support"
    echo "  --no-gpu       - Disable all GPU support"
    echo
    echo "Targets:"
    echo "  all            - Build executable (default)"
    echo "  static         - Build static library"
    echo "  clean          - Clean build files"
    echo "  help           - Show this help"
    echo
    echo "Examples:"
    echo "  ./build.sh                     # Build release executable with all features"
    echo "  ./build.sh debug               # Build debug executable"
    echo "  ./build.sh static              # Build static library"
    echo "  ./build.sh --no-cuda           # Build with OpenCL only"
    echo "  ./build.sh clean               # Clean build files"
    echo
    echo "Dependencies:"
    echo "  - GCC/G++ (build-essential package)"
    echo "  - Python 3"
    echo "  - CUDA Toolkit (optional, for CUDA support)"
    echo "  - OpenCL SDK (optional, for OpenCL support)"
    echo "  - libpng development files"
    echo
    echo "Environment Variables:"
    echo "  CUDA_PATH      - CUDA installation path"
    echo "  OPENCL_SDK_PATH - OpenCL SDK installation path"
    echo
}

# Parse command line arguments
CONFIG="release"
FEATURES="CUDA OPENCL"
TARGET="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        debug)
            CONFIG="debug"
            shift
            ;;
        release)
            CONFIG="release"
            shift
            ;;
        static)
            TARGET="static"
            shift
            ;;
        clean)
            TARGET="clean"
            shift
            ;;
        help)
            show_help
            exit 0
            ;;
        --no-cuda)
            FEATURES="OPENCL"
            shift
            ;;
        --no-opencl)
            FEATURES="CUDA"
            shift
            ;;
        --no-gpu)
            FEATURES=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build
echo "Building with configuration: $CONFIG"
echo "Features: $FEATURES"
echo "Target: $TARGET"
echo

# Run make
echo "Running make..."
make CONFIG="$CONFIG" FEATURES="$FEATURES" $TARGET

if [ $? -eq 0 ]; then
    echo
    echo "Build completed successfully!"
    if [ "$TARGET" = "all" ]; then
        echo "Executable: bin/$CONFIG/guetzli"
    fi
    if [ "$TARGET" = "static" ]; then
        echo "Static library: bin/$CONFIG/libguetzli_static.a"
    fi
else
    echo
    echo "Build failed with error code $?"
    echo
    echo "Common issues:"
    echo "- Make sure all dependencies are installed"
    echo "- Check that CUDA_PATH and OPENCL_SDK_PATH are set correctly"
    echo "- Run './build.sh help' for more information"
    exit 1
fi
