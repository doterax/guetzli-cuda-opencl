@echo off
REM Guetzli CUDA/OpenCL Build Script for Windows
REM This script helps set up the environment and build the project

setlocal enabledelayedexpansion

echo Guetzli CUDA/OpenCL Build Script
echo ================================
echo.

REM Check if make is available (skip if help is requested)
if "%~1" neq "help" (
    where make >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: make is not found in PATH
        echo Please install MinGW-w64 or MSYS2 and add it to your PATH
        echo Download from: https://www.msys2.org/
        echo.
        pause
        exit /b 1
    )
)

REM Check if Python is available
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: python is not found in PATH
    echo Please install Python and add it to your PATH
    echo Download from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Check for CUDA
if defined CUDA_PATH (
    echo Found CUDA_PATH: %CUDA_PATH%
    if not exist "%CUDA_PATH%\bin\nvcc.exe" (
        echo WARNING: nvcc.exe not found in %CUDA_PATH%\bin\
        echo CUDA support may not work properly
    )
) else (
    echo WARNING: CUDA_PATH not set
    echo CUDA support will be disabled
    echo To enable CUDA, set CUDA_PATH environment variable
)

REM Check for OpenCL
if defined OPENCL_SDK_PATH (
    echo Found OPENCL_SDK_PATH: %OPENCL_SDK_PATH%
) else (
    echo WARNING: OPENCL_SDK_PATH not set
    echo OpenCL support may not work properly
    echo To enable OpenCL, set OPENCL_SDK_PATH environment variable
)

echo.

REM Parse command line arguments
set CONFIG=release
set FEATURES=CUDA OPENCL
set TARGET=all

:parse_args
if "%~1"=="" goto :build
if "%~1"=="debug" set CONFIG=debug
if "%~1"=="release" set CONFIG=release
if "%~1"=="static" set TARGET=static
if "%~1"=="clean" set TARGET=clean
if "%~1"=="help" goto :show_help
if "%~1"=="--no-cuda" set FEATURES=OPENCL
if "%~1"=="--no-opencl" set FEATURES=CUDA
if "%~1"=="--no-gpu" set FEATURES=
shift
goto :parse_args

:build
echo Building with configuration: %CONFIG%
echo Features: %FEATURES%
echo Target: %TARGET%
echo.

REM Set up environment for Visual Studio tools if available
if exist "vcvars64.bat" (
    echo Setting up Visual Studio environment...
    call vcvars64.bat
)

REM Run make
echo Running make...
make CONFIG=%CONFIG% FEATURES="%FEATURES%" %TARGET%

if %errorlevel% equ 0 (
    echo.
    echo Build completed successfully!
    if "%TARGET%"=="all" (
        echo Executable: bin\%CONFIG%\guetzli.exe
    )
    if "%TARGET%"=="static" (
        echo Static library: bin\%CONFIG%\libguetzli_static.a
    )
) else (
    echo.
    echo Build failed with error code %errorlevel%
    echo.
    echo Common issues:
    echo - Make sure MinGW-w64 is installed and in PATH
    echo - Check that all dependencies are installed
    echo - Verify CUDA_PATH and OPENCL_SDK_PATH if using GPU features
    echo - Run 'build.bat help' for more information
)

echo.
pause
exit /b %errorlevel%

:show_help
echo Usage: build.bat [options] [target]
echo.
echo Options:
echo   debug          - Build in debug mode
echo   release        - Build in release mode (default)
echo   --no-cuda      - Disable CUDA support
echo   --no-opencl    - Disable OpenCL support
echo   --no-gpu       - Disable all GPU support
echo.
echo Targets:
echo   all            - Build executable (default)
echo   static         - Build static library
echo   clean          - Clean build files
echo   help           - Show this help
echo.
echo Examples:
echo   build.bat                     # Build release executable with all features
echo   build.bat debug               # Build debug executable
echo   build.bat static              # Build static library
echo   build.bat --no-cuda           # Build with OpenCL only
echo   build.bat clean               # Clean build files
echo.
echo Dependencies:
echo   - MinGW-w64 (https://www.msys2.org/)
echo   - Python (https://www.python.org/downloads/)
echo   - CUDA Toolkit (optional, for CUDA support)
echo   - OpenCL SDK (optional, for OpenCL support)
echo   - libpng development files
echo.
echo Environment Variables:
echo   CUDA_PATH      - CUDA installation path
echo   OPENCL_SDK_PATH - OpenCL SDK installation path
echo.
pause
exit /b 0
