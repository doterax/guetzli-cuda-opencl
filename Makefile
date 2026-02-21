# Guetzli CUDA/OpenCL Makefile
# Supports: Windows (LLVM/Clang), Linux, macOS
# Build external deps first:  make external
# Then build:                 make
# Run tests:                  make test

# Detect OS
# Check if we're in a Unix-like shell environment (Git Bash, MSYS2, WSL, Cygwin)
SHELL_TYPE := $(shell echo $$SHELL)
# Detect if we have a Unix shell even on Windows
HAS_UNIX_SHELL := $(shell which sh 2>/dev/null || echo "")

ifeq ($(OS),Windows_NT)
    # Check if we're running in a Unix-like environment (Git Bash, MSYS2, etc.)
    ifneq ($(HAS_UNIX_SHELL),)
        # Running in Unix-like shell on Windows (Git Bash, MSYS2, Cygwin)
        DETECTED_OS := WindowsUnix
        RM := rm -f
        RMDIR := rm -rf
        MKDIR := mkdir -p
        COPY := cp
        SEP := /
        EXE_EXT := .exe
        OBJ_EXT := .o
        LIB_EXT := .a
        # Default to LLVM/Clang on Windows if available
        ifeq ($(CXX),)
            CXX := $(shell which clang++ 2>/dev/null || which g++ 2>/dev/null || echo g++)
        else
            # Normalize backslashes from Windows env vars for Unix shell
            CXX := $(subst \,/,$(CXX))
        endif
        ifeq ($(AR),)
            AR := $(shell which llvm-ar 2>/dev/null || which ar 2>/dev/null || echo ar)
        else
            AR := $(subst \,/,$(AR))
        endif
        # Windows-specific paths (normalize backslashes)
        ifdef CUDA_PATH
            CUDA_INC := $(subst \,/,$(CUDA_PATH))/include
            CUDA_LIB := $(subst \,/,$(CUDA_PATH))/lib/x64
        else
            CUDA_INC :=
            CUDA_LIB :=
        endif
        ifdef OPENCL_SDK_PATH
            OPENCL_INC := $(OPENCL_SDK_PATH)/include
            OPENCL_LIB := $(OPENCL_SDK_PATH)/lib/x64
        else
            OPENCL_INC :=
            OPENCL_LIB :=
        endif
    else
        # Native Windows cmd.exe environment
        DETECTED_OS := Windows
        RM := del /Q /F
        RMDIR := rmdir /S /Q
        MKDIR := mkdir
        COPY := copy /Y
        SEP := /
        EXE_EXT := .exe
        OBJ_EXT := .o
        LIB_EXT := .a
        # Default to LLVM/Clang on Windows if available
        ifeq ($(CXX),)
            CXX := clang++
            # Fallback to g++ if clang++ not found
            ifeq ($(shell where clang++ 2>NUL 2>&1),)
                CXX := g++
            endif
        endif
        ifeq ($(AR),)
            AR := llvm-ar
            ifeq ($(shell where llvm-ar 2>NUL 2>&1),)
                AR := ar
            endif
        endif
        # Windows-specific paths
        ifdef CUDA_PATH
            CUDA_INC := $(CUDA_PATH)/include
            CUDA_LIB := $(CUDA_PATH)/lib/x64
        else
            CUDA_INC :=
            CUDA_LIB :=
        endif
        ifdef OPENCL_SDK_PATH
            OPENCL_INC := $(OPENCL_SDK_PATH)/include
            OPENCL_LIB := $(OPENCL_SDK_PATH)/lib/x64
        else
            OPENCL_INC :=
            OPENCL_LIB :=
        endif
    endif
else
    # Detect Unix-like OS (Linux, macOS, etc.)
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        DETECTED_OS := Linux
        RM := rm -f
        RMDIR := rm -rf
        MKDIR := mkdir -p
        COPY := cp
        SEP := /
        EXE_EXT :=
        OBJ_EXT := .o
        LIB_EXT := .a
        CUDA_INC := /usr/local/cuda/include
        CUDA_LIB := /usr/local/cuda/lib64
        OPENCL_INC := /usr/include
        OPENCL_LIB := /usr/lib/x86_64-linux-gnu
    endif
    ifeq ($(UNAME_S),Darwin)
        DETECTED_OS := macOS
        RM := rm -f
        RMDIR := rm -rf
        MKDIR := mkdir -p
        COPY := cp
        SEP := /
        EXE_EXT :=
        OBJ_EXT := .o
        LIB_EXT := .a
        CUDA_INC := /usr/local/cuda/include
        CUDA_LIB := /usr/local/cuda/lib
        OPENCL_INC := /usr/local/include
        OPENCL_LIB := /usr/local/lib
    endif
    # Default compilers for Unix
    ifeq ($(CXX),)
        CXX := g++
        # Check for clang++ and prefer it if available
        ifneq ($(shell which clang++ 2>/dev/null),)
            CXX := clang++
        endif
    endif
    ifeq ($(AR),)
        AR := ar
    endif
endif

# Python detection
PYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)
ifeq ($(PYTHON),)
    PYTHON := python
endif

# NVCC detection
ifeq ($(DETECTED_OS),WindowsUnix)
    # On Windows with Unix shell, use Unix-style paths
    ifdef CUDA_PATH
        NVCC := $(CUDA_PATH)/bin/nvcc
    else
        NVCC := $(shell which nvcc 2>/dev/null || echo nvcc)
    endif
else ifeq ($(DETECTED_OS),Windows)
    # On Windows cmd.exe, try CUDA_PATH first, then check PATH
    ifdef CUDA_PATH
        NVCC := $(CUDA_PATH)/bin/nvcc.exe
    else
        NVCC := nvcc
    endif
else
    # On Unix, check PATH first, then CUDA_PATH
    ifdef CUDA_PATH
        NVCC := $(CUDA_PATH)/bin/nvcc
    else
        # Try to find nvcc in PATH, default to 'nvcc' if not found
        NVCC := $(shell which nvcc 2>/dev/null || echo nvcc)
    endif
endif

# Directories
SRC_DIR := .
GUETZLI_DIR := guetzli
CLGUETZLI_DIR := clguetzli
THIRD_PARTY_DIR := third_party
BUTTERAUGLI_DIR := $(THIRD_PARTY_DIR)/butteraugli/butteraugli
MINILZO_DIR := $(THIRD_PARTY_DIR)/minilzo
OPENCL_INCLUDE_DIR := $(THIRD_PARTY_DIR)/OpenCL/include
OPENCL_WRAPPER_DIR := $(THIRD_PARTY_DIR)/OpenCL-Wrapper
BUILD_DIR := build
BIN_DIR := bin
OBJ_DIR := $(BUILD_DIR)/obj

# External dependency install directory (built via CMake superbuild)
EXTERNAL_DIR := external
EXT_INSTALL  := $(EXTERNAL_DIR)/install
EXT_INC      := $(EXT_INSTALL)/include
EXT_LIB      := $(EXT_INSTALL)/lib

# Configurable CUDA architecture
CUDA_ARCH ?= compute_75

# Target configuration
CONFIG ?= release
TARGET_NAME := guetzli
STATIC_LIB_NAME := libguetzli_static$(LIB_EXT)

# Build directories
TARGET_DIR := $(BIN_DIR)/$(CONFIG)
OBJ_TARGET_DIR := $(OBJ_DIR)/$(CONFIG)

# Define space for use in conditions
space :=
space +=

# Compiler flags base
CXXFLAGS_BASE := -std=c++11
CXXFLAGS_BASE += -I$(SRC_DIR)
CXXFLAGS_BASE += -I$(BUTTERAUGLI_DIR)/..
CXXFLAGS_BASE += -I$(CLGUETZLI_DIR)
CXXFLAGS_BASE += -I$(OPENCL_INCLUDE_DIR)
CXXFLAGS_BASE += -I$(OPENCL_WRAPPER_DIR)
CXXFLAGS_BASE += -I$(MINILZO_DIR)

# External dependency includes/libs (from CMake superbuild)
ifneq ($(wildcard $(EXT_INC)/png.h),)
    CXXFLAGS_BASE += -I$(EXT_INC)
    LDFLAGS += -L$(EXT_LIB)
endif

# Platform-specific includes
# CUDA includes
ifdef CUDA_PATH
    # Normalize backslashes and escape spaces for shell compatibility
    CUDA_INCLUDE_PATH := $(subst \,/,$(CUDA_PATH))/include
    space := $(subst ,, )
    CUDA_INC_ESC := $(subst $(space),\ ,$(CUDA_INCLUDE_PATH))
    CXXFLAGS_BASE += -I$(CUDA_INC_ESC)
    NVCC_INCLUDES := -I"$(CUDA_INCLUDE_PATH)"
else
    # Try to find CUDA in common locations
    ifeq ($(DETECTED_OS),WindowsUnix)
        ifneq ($(wildcard /usr/local/cuda/include/cuda.h),)
            CXXFLAGS_BASE += -I/usr/local/cuda/include
        endif
        ifneq ($(wildcard /opt/cuda/include/cuda.h),)
            CXXFLAGS_BASE += -I/opt/cuda/include
        endif
    else ifeq ($(DETECTED_OS),Linux)
        ifneq ($(wildcard /usr/local/cuda/include/cuda.h),)
            CXXFLAGS_BASE += -I/usr/local/cuda/include
        endif
        ifneq ($(wildcard /opt/cuda/include/cuda.h),)
            CXXFLAGS_BASE += -I/opt/cuda/include
        endif
    else ifeq ($(DETECTED_OS),macOS)
        ifneq ($(wildcard /usr/local/cuda/include/cuda.h),)
            CXXFLAGS_BASE += -I/usr/local/cuda/include
        endif
    endif
    # Also check if CUDA_INC is set directly
    ifdef CUDA_INC
        ifneq ($(CUDA_INC),)
            CXXFLAGS_BASE += -I$(CUDA_INC)
        endif
    endif
endif

# OpenCL includes
ifdef OPENCL_SDK_PATH
    CXXFLAGS_BASE += -I$(OPENCL_SDK_PATH)/include
else ifdef OPENCL_INC
    ifneq ($(OPENCL_INC),)
        CXXFLAGS_BASE += -I$(OPENCL_INC)
    endif
endif

# Configuration-specific flags
ifeq ($(CONFIG),debug)
    CXXFLAGS := $(CXXFLAGS_BASE) -g -O0 -DDEBUG
    NVCCFLAGS := -g -O0 -DDEBUG --device-debug
else
    CXXFLAGS := $(CXXFLAGS_BASE) -O3 -DNDEBUG -ffast-math
    NVCCFLAGS := -O3 -DNDEBUG
    # Link-time optimisation (set LTO=1 to enable; disabled by default because
    # it increases build time significantly)
    ifeq ($(LTO),1)
        CXXFLAGS += -flto
        LDFLAGS  += -flto
    endif
endif

# Windows-specific compiler flags (apply after CXXFLAGS is set from CXXFLAGS_BASE)
# On Windows (MinGW) we produce a fully self-contained binary:
#   -static-libgcc -static-libstdc++  → no libgcc_s_seh-1.dll / libstdc++-6.dll
#   -Wl,-Bstatic                      → force .a for jpeg, png, zlib, tiff, pthread
# Libraries that *must* remain dynamic (OpenCL, cfgmgr32, ole32) go into
# LDFLAGS_DYNAMIC and are appended after -Wl,-Bdynamic at link time.
LDFLAGS_DYNAMIC :=
ifeq ($(DETECTED_OS),Windows)
    CXXFLAGS += -Wno-unknown-pragmas -Wno-microsoft-template
    # Use forward slashes for paths in compiler flags
    CXXFLAGS := $(subst \,/,$(CXXFLAGS))
    # Produce a fully self-contained binary: -static makes the linker prefer
    # .a archives for ALL libraries (including implicit libgcc, libstdc++,
    # winpthread).  Libraries that must stay dynamic (OpenCL, Windows system
    # DLLs) go into LDFLAGS_DYNAMIC with -Wl,-Bdynamic.
    LDFLAGS += -static
endif
ifeq ($(DETECTED_OS),WindowsUnix)
    CXXFLAGS += -Wno-unknown-pragmas
    LDFLAGS += -static
endif

# macOS-specific compiler flags
ifeq ($(DETECTED_OS),macOS)
    CXXFLAGS += -stdlib=libc++
    LDFLAGS += -stdlib=libc++
endif

# Feature flags
FEATURES ?= CUDA OPENCL FULL_JPEG
ifneq (,$(findstring CUDA,$(FEATURES)))
    CXXFLAGS += -D__USE_CUDA__
    # CUDA driver library is loaded at runtime (cuda_dynload.cpp) so we do NOT
    # link -lcuda or -L<cuda_lib>.  On Linux we need -ldl for dlopen/dlsym.
    ifneq ($(DETECTED_OS),$(filter $(DETECTED_OS),Windows WindowsUnix macOS))
        LDFLAGS += -ldl
    endif
endif

ifneq (,$(findstring OPENCL,$(FEATURES)))
    CXXFLAGS += -D__USE_OPENCL__
    CXXFLAGS += -DCL_TARGET_OPENCL_VERSION=300
    CXXFLAGS += -DCL_HPP_MINIMUM_OPENCL_VERSION=200
    ifeq ($(DETECTED_OS),macOS)
        LDFLAGS += -framework OpenCL
    else
        # Check for library naming: CMake ICD-Loader installs as OpenCL.a (no lib prefix)
        ifneq ($(wildcard $(EXT_LIB)/OpenCL.a),)
            LDFLAGS += $(EXT_LIB)/OpenCL.a
        else ifneq ($(wildcard $(EXT_LIB)/libOpenCL.a),)
            LDFLAGS += -lOpenCL
        else
            # On Windows, OpenCL.dll is a system-provided DLL — keep it dynamic
            ifeq ($(DETECTED_OS),$(filter $(DETECTED_OS),Windows WindowsUnix))
                LDFLAGS_DYNAMIC += -lOpenCL
            else
                LDFLAGS += -lOpenCL
            endif
        endif
        # Windows: OpenCL ICD Loader needs cfgmgr32 and ole32 (system DLLs)
        ifeq ($(DETECTED_OS),$(filter $(DETECTED_OS),Windows WindowsUnix))
            LDFLAGS_DYNAMIC += -lcfgmgr32 -lole32
        endif
    endif
    ifdef OPENCL_LIB
        ifneq ($(OPENCL_LIB),)
            LDFLAGS += -L$(OPENCL_LIB)
        endif
    endif
endif

# JPEG support via libjpeg-turbo
ifneq (,$(findstring FULL_JPEG,$(FEATURES)))
    CXXFLAGS += -D__SUPPORT_FULL_JPEG__
    # On macOS, prefer static archives to avoid @rpath dylib issues
    ifeq ($(DETECTED_OS),macOS)
        ifneq ($(wildcard $(EXT_LIB)/libjpeg.a),)
            LDFLAGS += $(EXT_LIB)/libjpeg.a
        else
            LDFLAGS += -ljpeg
        endif
        ifneq ($(wildcard $(EXT_LIB)/libturbojpeg.a),)
            LDFLAGS += $(EXT_LIB)/libturbojpeg.a
        else
            LDFLAGS += -lturbojpeg
        endif
    else
        LDFLAGS += -ljpeg -lturbojpeg
    endif
endif

# PNG + zlib support — prefer external/install, then third_party, then system
# Check external/install first (from CMake superbuild)
ifneq ($(wildcard $(EXT_INC)/png.h),)
    # Already added -I/-L above; just add link flags
    # On macOS the linker prefers .dylib over .a when both exist in the same
    # directory, even though the CMake superbuild passes BUILD_SHARED_LIBS=OFF.
    # zlib's own CMakeLists always builds both; linking the dylib produces an
    # @rpath reference that fails at runtime.  Explicitly pass the static
    # archive paths on macOS to avoid this.
    ifeq ($(DETECTED_OS),macOS)
        ifneq ($(wildcard $(EXT_LIB)/libpng16.a),)
            PNG_LIBS := $(EXT_LIB)/libpng16.a
        else
            PNG_LIBS := -lpng16
        endif
        ifneq ($(wildcard $(EXT_LIB)/libz.a),)
            PNG_LIBS += $(EXT_LIB)/libz.a
        else
            PNG_LIBS += -lz
        endif
    else
    # CMake zlib on MinGW installs as libzlibstatic.a, not libz.a
    ifneq ($(wildcard $(EXT_LIB)/libz.a),)
        PNG_LIBS := -lpng16 -lz
    else
        PNG_LIBS := -lpng16 -lzlibstatic
    endif
    endif
else
    # Check third_party/libpng
    PNG_THIRD_PARTY_DIR := $(THIRD_PARTY_DIR)/libpng
    PNG_THIRD_PARTY_INCLUDE := $(PNG_THIRD_PARTY_DIR)/include
    PNG_THIRD_PARTY_LIB := $(PNG_THIRD_PARTY_DIR)/lib
    PNG_IN_THIRD_PARTY := $(wildcard $(PNG_THIRD_PARTY_INCLUDE)/png.h)

    ifneq ($(PNG_IN_THIRD_PARTY),)
        CXXFLAGS_BASE += -I$(PNG_THIRD_PARTY_INCLUDE)
        LDFLAGS += -L$(PNG_THIRD_PARTY_LIB)
        PNG_LIBS := -lpng -lz
    else
        # Fallback to system libpng
        ifeq ($(DETECTED_OS),$(filter $(DETECTED_OS),Windows WindowsUnix))
            PNG_LIBS := -lpng -lz
            ifdef PNG_PATH
                CXXFLAGS_BASE += -I$(PNG_PATH)/include
                LDFLAGS += -L$(PNG_PATH)/lib
            endif
        else
            PNG_CFLAGS := $(shell pkg-config --cflags libpng 2>/dev/null || echo "")
            PNG_LIBS   := $(shell pkg-config --libs libpng 2>/dev/null || echo "-lpng -lz")
            CXXFLAGS_BASE += $(PNG_CFLAGS)
        endif
    endif
endif
LDFLAGS += $(PNG_LIBS)

# TIFF support — prefer external/install, then system
ifneq ($(wildcard $(EXT_INC)/tiffio.h),)
    # On macOS, prefer static archive to avoid @rpath dylib issues
    ifeq ($(DETECTED_OS),macOS)
        ifneq ($(wildcard $(EXT_LIB)/libtiff.a),)
            TIFF_LIBS := $(EXT_LIB)/libtiff.a
        else
            TIFF_LIBS := -ltiff
        endif
    else
        TIFF_LIBS := -ltiff
    endif
else
    ifeq ($(DETECTED_OS),$(filter $(DETECTED_OS),Windows WindowsUnix))
        TIFF_LIBS := -ltiff
    else
        TIFF_LIBS := $(shell pkg-config --libs libtiff-4 2>/dev/null || echo "-ltiff")
    endif
endif
LDFLAGS += $(TIFF_LIBS)

# Source files
GUETZLI_SOURCES := $(wildcard $(GUETZLI_DIR)/*.cc)
CLGUETZLI_SOURCES := $(wildcard $(CLGUETZLI_DIR)/*.cpp)
BUTTERAUGLI_SOURCES := $(BUTTERAUGLI_DIR)/butteraugli.cc
MINILZO_SOURCES := $(MINILZO_DIR)/minilzo.c

# Exclude main from static library
STATIC_SOURCES := $(filter-out $(GUETZLI_DIR)/guetzli.cc,$(GUETZLI_SOURCES))
STATIC_SOURCES += $(CLGUETZLI_SOURCES)
STATIC_SOURCES += $(BUTTERAUGLI_SOURCES)
STATIC_SOURCES += $(MINILZO_SOURCES)

EXECUTABLE_SOURCES := $(GUETZLI_SOURCES)
EXECUTABLE_SOURCES += $(CLGUETZLI_SOURCES)
EXECUTABLE_SOURCES += $(BUTTERAUGLI_SOURCES)
EXECUTABLE_SOURCES += $(MINILZO_SOURCES)

# Object files - handle both .cc/.cpp and .c files
# Convert source paths to object paths preserving directory structure
STATIC_OBJECTS := $(patsubst $(GUETZLI_DIR)/%.cc,$(OBJ_TARGET_DIR)/$(GUETZLI_DIR)/%.o,$(filter $(GUETZLI_DIR)/%.cc,$(STATIC_SOURCES)))
STATIC_OBJECTS += $(patsubst $(CLGUETZLI_DIR)/%.cpp,$(OBJ_TARGET_DIR)/$(CLGUETZLI_DIR)/%.o,$(filter $(CLGUETZLI_DIR)/%.cpp,$(STATIC_SOURCES)))
STATIC_OBJECTS += $(patsubst $(BUTTERAUGLI_DIR)/%.cc,$(OBJ_TARGET_DIR)/$(BUTTERAUGLI_DIR)/%.o,$(filter $(BUTTERAUGLI_DIR)/%.cc,$(STATIC_SOURCES)))
STATIC_OBJECTS += $(patsubst $(MINILZO_DIR)/%.c,$(OBJ_TARGET_DIR)/$(MINILZO_DIR)/%.o,$(filter $(MINILZO_DIR)/%.c,$(STATIC_SOURCES)))

EXECUTABLE_OBJECTS := $(patsubst $(GUETZLI_DIR)/%.cc,$(OBJ_TARGET_DIR)/$(GUETZLI_DIR)/%.o,$(filter $(GUETZLI_DIR)/%.cc,$(EXECUTABLE_SOURCES)))
EXECUTABLE_OBJECTS += $(patsubst $(CLGUETZLI_DIR)/%.cpp,$(OBJ_TARGET_DIR)/$(CLGUETZLI_DIR)/%.o,$(filter $(CLGUETZLI_DIR)/%.cpp,$(EXECUTABLE_SOURCES)))
EXECUTABLE_OBJECTS += $(patsubst $(BUTTERAUGLI_DIR)/%.cc,$(OBJ_TARGET_DIR)/$(BUTTERAUGLI_DIR)/%.o,$(filter $(BUTTERAUGLI_DIR)/%.cc,$(EXECUTABLE_SOURCES)))
EXECUTABLE_OBJECTS += $(patsubst $(MINILZO_DIR)/%.c,$(OBJ_TARGET_DIR)/$(MINILZO_DIR)/%.o,$(filter $(MINILZO_DIR)/%.c,$(EXECUTABLE_SOURCES)))

# CUDA object files (use .cu.o suffix to avoid collision with .cpp → .o)
CUDA_SOURCES := $(CLGUETZLI_DIR)/clguetzli.cu
CUDA_OBJECTS := $(patsubst $(CLGUETZLI_DIR)/%.cu,$(OBJ_TARGET_DIR)/$(CLGUETZLI_DIR)/%.cu.o,$(CUDA_SOURCES))

# Generated header files
GENERATED_HEADERS := $(CLGUETZLI_DIR)/clguetzli_cu_ptx.h $(CLGUETZLI_DIR)/clguetzli_cl_src.h

# Targets
TARGET := $(TARGET_DIR)/$(TARGET_NAME)$(EXE_EXT)
STATIC_TARGET := $(TARGET_DIR)/$(STATIC_LIB_NAME)

# Default target
.PHONY: all clean help static executable cuda-headers external test install dist

all: $(TARGET)

static: $(STATIC_TARGET)

executable: $(TARGET)

# Build external dependencies via CMake superbuild
external:
	@echo "Building external dependencies..."
ifeq ($(DETECTED_OS),$(filter $(DETECTED_OS),WindowsUnix Linux macOS))
	@cd $(EXTERNAL_DIR) && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON 2>&1
	@cd $(EXTERNAL_DIR) && cmake --build build --config Release 2>&1
else
	@cd $(EXTERNAL_DIR) && cmake -B build -DCMAKE_BUILD_TYPE=Release 2>&1
	@cd $(EXTERNAL_DIR) && cmake --build build --config Release 2>&1
endif
	@echo "External dependencies built successfully."

# Run tests
test: $(TARGET)
ifeq ($(DETECTED_OS),$(filter $(DETECTED_OS),WindowsUnix Linux macOS))
	@bash tests/run_tests.sh $(TARGET)
else
	@tests\run_tests.bat $(subst /,\,$(TARGET))
endif

# Install
PREFIX ?= /usr/local
install: $(TARGET)
ifeq ($(DETECTED_OS),$(filter $(DETECTED_OS),Linux macOS))
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/
else
	@echo "Install target is only supported on Linux/macOS. Copy $(TARGET) manually."
endif

# Create distribution archive
dist: $(TARGET)
	@echo "Creating distribution archive..."
ifeq ($(DETECTED_OS),$(filter $(DETECTED_OS),WindowsUnix Windows))
	@7z a guetzli-cuda-opencl-win-x64.zip $(TARGET) README.md LICENSE 2>/dev/null || echo "7z not found; skipping archive"
else ifeq ($(DETECTED_OS),macOS)
	@tar czf guetzli-cuda-opencl-macos-arm64.tar.gz -C $(TARGET_DIR) $(TARGET_NAME)$(EXE_EXT) -C $(SRC_DIR) README.md LICENSE
else
	@tar czf guetzli-cuda-opencl-linux-x64.tar.gz -C $(TARGET_DIR) $(TARGET_NAME)$(EXE_EXT) -C $(SRC_DIR) README.md LICENSE
endif

# Create directories
$(TARGET_DIR):
	@echo Creating $(TARGET_DIR)
ifeq ($(DETECTED_OS),WindowsUnix)
	@$(MKDIR) $(TARGET_DIR) 2>/dev/null || true
else ifeq ($(DETECTED_OS),Windows)
	@if not exist "$(subst /,\,$(TARGET_DIR))" $(MKDIR) "$(subst /,\,$(TARGET_DIR))" 2>NUL || true
else
	@$(MKDIR) $(TARGET_DIR) 2>/dev/null || true
endif

$(OBJ_TARGET_DIR):
	@echo Creating $(OBJ_TARGET_DIR)
ifeq ($(DETECTED_OS),WindowsUnix)
	@test ! -f $(BUILD_DIR) || ($(RM) $(BUILD_DIR) && echo Removed file $(BUILD_DIR)) || true
	@test -d $(BUILD_DIR) || $(MKDIR) $(BUILD_DIR) 2>/dev/null || true
	@test ! -f $(OBJ_DIR) || ($(RM) $(OBJ_DIR) && echo Removed file $(OBJ_DIR)) || true
	@test -d $(OBJ_DIR) || $(MKDIR) $(OBJ_DIR) 2>/dev/null || true
	@$(MKDIR) $(OBJ_TARGET_DIR) 2>/dev/null || true
else ifeq ($(DETECTED_OS),Windows)
	@if exist "$(subst /,\,$(BUILD_DIR))" ( \
		if not exist "$(subst /,\,$(BUILD_DIR))\\*" del /F /Q "$(subst /,\,$(BUILD_DIR))" 2>NUL || true \
	) else ( \
		if not exist "$(subst /,\,$(OBJ_TARGET_DIR))" $(MKDIR) "$(subst /,\,$(OBJ_TARGET_DIR))" 2>NUL || true \
	)
else
	@test ! -f $(BUILD_DIR) || ($(RM) $(BUILD_DIR) && echo Removed file $(BUILD_DIR)) || true
	@test -d $(BUILD_DIR) || $(MKDIR) $(BUILD_DIR) 2>/dev/null || true
	@test ! -f $(OBJ_DIR) || ($(RM) $(OBJ_DIR) && echo Removed file $(OBJ_DIR)) || true
	@test -d $(OBJ_DIR) || $(MKDIR) $(OBJ_DIR) 2>/dev/null || true
	@$(MKDIR) $(OBJ_TARGET_DIR) 2>/dev/null || true
endif

# Prevent Make's implicit %: %.cpp rule from trying to compile .cl.cpp -> .cl
$(CLGUETZLI_DIR)/clguetzli.cl: ;

# Build minilzoc tool (needed for LZO compression of embedded GPU kernels)
# GPU kernel source/PTX MUST be LZO-compressed before embedding to avoid
# antivirus false positives from C-like code in the executable's text segment.
MINILZOC_DIR := minilzoc
MINILZOC := $(MINILZOC_DIR)/minilzoc$(EXE_EXT)

$(MINILZOC): $(MINILZOC_DIR)/minilzoc.cpp $(MINILZO_DIR)/minilzo.c
	@echo Building minilzoc...
	@$(CXX) -O2 -std=c++11 -I$(MINILZO_DIR) -o $@ $^

# Generate CUDA headers with mandatory LZO compression
cuda-headers: $(GENERATED_HEADERS)

$(CLGUETZLI_DIR)/clguetzli_cu_ptx.h: $(CLGUETZLI_DIR)/clguetzli.cu | $(MINILZOC)
	@echo Generating CUDA PTX header...
	@$(NVCC) $(NVCCFLAGS) -Xcompiler "/wd 4819" -I"$(SRC_DIR)" -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -arch=$(CUDA_ARCH) -ptx -o $(CLGUETZLI_DIR)/clguetzli.cu.ptx64 $(CLGUETZLI_DIR)/clguetzli.cu || echo "Warning: CUDA PTX generation failed - CUDA may not be available"
	@$(MINILZOC) $(CLGUETZLI_DIR)/clguetzli.cu.ptx64 $(CLGUETZLI_DIR)/clguetzli.cu.ptx64.lzo
	@$(PYTHON) format_header.py $(CLGUETZLI_DIR)/clguetzli.cu.ptx64.lzo $(CLGUETZLI_DIR)/clguetzli_cu_ptx.h clguetzli_cu64_lzo || echo "Warning: Header generation failed"

$(CLGUETZLI_DIR)/clguetzli_cl_src.h: $(CLGUETZLI_DIR)/clguetzli.cl | $(MINILZOC)
	@echo Generating OpenCL source header...
	@$(MINILZOC) $(CLGUETZLI_DIR)/clguetzli.cl $(CLGUETZLI_DIR)/clguetzli.cl.lzo
	@$(PYTHON) format_header.py $(CLGUETZLI_DIR)/clguetzli.cl.lzo $(CLGUETZLI_DIR)/clguetzli_cl_src.h clguetzli_cl_src_lzo || echo "Warning: OpenCL header generation failed"

# Build executable - ensure object files are built first
# Note: CUDA_OBJECTS not linked — .cu kernels are compiled to PTX and loaded at
# runtime via the driver API (cuModuleLoadDataEx). The PTX is embedded in
# clguetzli_cu_ptx.h which is generated by the cuda-headers target.
$(TARGET): $(TARGET_DIR) $(OBJ_TARGET_DIR) cuda-headers $(EXECUTABLE_OBJECTS)
	@echo Linking $(TARGET_NAME)...
	@echo "  CXX      = $(CXX)"
	@echo "  LDFLAGS  = $(LDFLAGS)"
	@echo "  OBJECTS  = $(words $(EXECUTABLE_OBJECTS)) object file(s)"
ifeq ($(DETECTED_OS),WindowsUnix)
	$(CXX) $(EXECUTABLE_OBJECTS) -o $@ $(LDFLAGS) -Wl,-Bdynamic $(LDFLAGS_DYNAMIC) -Wl,-Bstatic
else ifeq ($(DETECTED_OS),Windows)
	$(CXX) $(EXECUTABLE_OBJECTS) -o "$@" $(LDFLAGS) -Wl,-Bdynamic $(LDFLAGS_DYNAMIC) -Wl,-Bstatic
else
	$(CXX) $(EXECUTABLE_OBJECTS) -o $@ $(LDFLAGS) $(LDFLAGS_DYNAMIC)
endif

# Build static library
$(STATIC_TARGET): $(TARGET_DIR) $(OBJ_TARGET_DIR) cuda-headers $(STATIC_OBJECTS)
	@echo Creating static library...
ifeq ($(DETECTED_OS),WindowsUnix)
	@$(AR) rcs $@ $(STATIC_OBJECTS)
else ifeq ($(DETECTED_OS),Windows)
	@$(AR) rcs "$@" $(STATIC_OBJECTS) || echo "Static library creation failed"
else
	@$(AR) rcs $@ $(STATIC_OBJECTS)
endif

# Compile C++ source files (.cc) from guetzli directory
$(OBJ_TARGET_DIR)/$(GUETZLI_DIR)/%.o: $(GUETZLI_DIR)/%.cc
	@echo Compiling $<...
ifeq ($(DETECTED_OS),WindowsUnix)
	@$(MKDIR) $(dir $@) 2>/dev/null || true
else ifeq ($(DETECTED_OS),Windows)
	@if not exist "$(subst /,\,$(dir $@))" $(MKDIR) "$(subst /,\,$(dir $@))" 2>NUL || true
else
	@$(MKDIR) $(dir $@) 2>/dev/null || true
endif
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile C++ source files (.cpp) from clguetzli directory
$(OBJ_TARGET_DIR)/$(CLGUETZLI_DIR)/%.o: $(CLGUETZLI_DIR)/%.cpp
	@echo Compiling $<...
ifeq ($(DETECTED_OS),WindowsUnix)
	@$(MKDIR) $(dir $@) 2>/dev/null || true
else ifeq ($(DETECTED_OS),Windows)
	@if not exist "$(subst /,\,$(dir $@))" $(MKDIR) "$(subst /,\,$(dir $@))" 2>NUL || true
else
	@$(MKDIR) $(dir $@) 2>/dev/null || true
endif
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile C++ source files (.cc) from butteraugli directory
$(OBJ_TARGET_DIR)/$(BUTTERAUGLI_DIR)/%.o: $(BUTTERAUGLI_DIR)/%.cc
	@echo Compiling $<...
ifeq ($(DETECTED_OS),WindowsUnix)
	@$(MKDIR) $(dir $@) 2>/dev/null || true
else ifeq ($(DETECTED_OS),Windows)
	@if not exist "$(subst /,\,$(dir $@))" $(MKDIR) "$(subst /,\,$(dir $@))" 2>NUL || true
else
	@$(MKDIR) $(dir $@) 2>/dev/null || true
endif
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile C source files (.c) from minilzo directory
$(OBJ_TARGET_DIR)/$(MINILZO_DIR)/%.o: $(MINILZO_DIR)/%.c
	@echo Compiling $<...
ifeq ($(DETECTED_OS),WindowsUnix)
	@$(MKDIR) $(dir $@) 2>/dev/null || true
else ifeq ($(DETECTED_OS),Windows)
	@if not exist "$(subst /,\,$(dir $@))" $(MKDIR) "$(subst /,\,$(dir $@))" 2>NUL || true
else
	@$(MKDIR) $(dir $@) 2>/dev/null || true
endif
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files (output .cu.o to avoid collision with .cpp → .o)
$(OBJ_TARGET_DIR)/$(CLGUETZLI_DIR)/%.cu.o: $(CLGUETZLI_DIR)/%.cu
	@echo Compiling CUDA $<...
ifeq ($(DETECTED_OS),WindowsUnix)
	@$(MKDIR) $(dir $@) 2>/dev/null || true
	@$(NVCC) $(NVCCFLAGS) $(NVCC_INCLUDES) -Xcompiler "-Wno-unknown-pragmas" -I"$(SRC_DIR)" -I$(BUTTERAUGLI_DIR)/.. -I$(CLGUETZLI_DIR) -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -arch=$(CUDA_ARCH) -c $< -o $@ 2>/dev/null || echo "Warning: CUDA compilation failed - CUDA may not be available"
else ifeq ($(DETECTED_OS),Windows)
	@if not exist "$(subst /,\,$(dir $@))" $(MKDIR) "$(subst /,\,$(dir $@))" 2>NUL || true
	@$(NVCC) $(NVCCFLAGS) $(NVCC_INCLUDES) -Xcompiler "/wd 4819" -I"$(SRC_DIR)" -I$(BUTTERAUGLI_DIR)/.. -I$(CLGUETZLI_DIR) -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -arch=$(CUDA_ARCH) -c $< -o $@ 2>NUL || echo Warning: CUDA compilation failed - CUDA may not be available
else
	@$(MKDIR) $(dir $@) 2>/dev/null || true
	@$(NVCC) $(NVCCFLAGS) $(NVCC_INCLUDES) -Xcompiler "-Wno-unknown-pragmas" -I"$(SRC_DIR)" -I$(BUTTERAUGLI_DIR)/.. -I$(CLGUETZLI_DIR) -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -arch=$(CUDA_ARCH) -c $< -o $@ 2>/dev/null || echo "Warning: CUDA compilation failed - CUDA may not be available"
endif

# Clean targets
clean:
	@echo Cleaning build files...
ifeq ($(DETECTED_OS),WindowsUnix)
	@test ! -f $(BUILD_DIR) || ($(RM) $(BUILD_DIR) && echo Removed file $(BUILD_DIR)) || true
	@test ! -f $(BIN_DIR) || ($(RM) $(BIN_DIR) && echo Removed file $(BIN_DIR)) || true
	@$(RMDIR) $(BUILD_DIR) $(BIN_DIR) 2>/dev/null || true
	@$(RM) $(CLGUETZLI_DIR)/clguetzli.cu.ptx64 $(CLGUETZLI_DIR)/clguetzli.cu.ptx32 $(CLGUETZLI_DIR)/clguetzli.cu.ptx 2>/dev/null || true
	@$(RM) $(CLGUETZLI_DIR)/clguetzli.cu.ptx64.lzo 2>/dev/null || true
	@$(RM) $(MINILZOC) 2>/dev/null || true
else ifeq ($(DETECTED_OS),Windows)
	@if exist "$(subst /,\,$(BUILD_DIR))" $(RMDIR) "$(subst /,\,$(BUILD_DIR))" 2>NUL || if exist "$(subst /,\,$(BUILD_DIR))" $(RM) "$(subst /,\,$(BUILD_DIR))" 2>NUL || true
	@if exist "$(subst /,\,$(BIN_DIR))" $(RMDIR) "$(subst /,\,$(BIN_DIR))" 2>NUL || if exist "$(subst /,\,$(BIN_DIR))" $(RM) "$(subst /,\,$(BIN_DIR))" 2>NUL || true
	@if exist "$(subst /,\,$(CLGUETZLI_DIR))\clguetzli.cu.ptx64" $(RM) "$(subst /,\,$(CLGUETZLI_DIR))\clguetzli.cu.ptx64"
	@if exist "$(subst /,\,$(CLGUETZLI_DIR))\clguetzli.cu.ptx32" $(RM) "$(subst /,\,$(CLGUETZLI_DIR))\clguetzli.cu.ptx32"
	@if exist "$(subst /,\,$(CLGUETZLI_DIR))\clguetzli.cu.ptx" $(RM) "$(subst /,\,$(CLGUETZLI_DIR))\clguetzli.cu.ptx"
	@if exist "$(subst /,\,$(CLGUETZLI_DIR))\clguetzli.cu.ptx64.lzo" $(RM) "$(subst /,\,$(CLGUETZLI_DIR))\clguetzli.cu.ptx64.lzo"
	@if exist "$(subst /,\,$(MINILZOC))" $(RM) "$(subst /,\,$(MINILZOC))"
else
	@test ! -f $(BUILD_DIR) || ($(RM) $(BUILD_DIR) && echo Removed file $(BUILD_DIR)) || true
	@test ! -f $(BIN_DIR) || ($(RM) $(BIN_DIR) && echo Removed file $(BIN_DIR)) || true
	@$(RMDIR) $(BUILD_DIR) $(BIN_DIR) 2>/dev/null || true
	@$(RM) $(CLGUETZLI_DIR)/clguetzli.cu.ptx64 $(CLGUETZLI_DIR)/clguetzli.cu.ptx32 $(CLGUETZLI_DIR)/clguetzli.cu.ptx 2>/dev/null || true
	@$(RM) $(CLGUETZLI_DIR)/clguetzli.cu.ptx64.lzo 2>/dev/null || true
	@$(RM) $(MINILZOC) 2>/dev/null || true
endif

# Help target
help:
	@echo "Guetzli CUDA/OpenCL Build System"
	@echo "================================"
	@echo ""
	@echo "Detected OS: $(DETECTED_OS)"
	@echo "C++ Compiler: $(CXX)"
	@echo "CUDA Compiler: $(NVCC)"
	@echo "Python: $(PYTHON)"
	@echo ""
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build executable (default)"
	@echo "  static       - Build static library"
	@echo "  executable   - Build executable"
	@echo "  cuda-headers - Generate CUDA header files"
	@echo "  clean        - Remove all build files"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Options:"
	@echo "  CONFIG=debug|release  - Build configuration (default: release)"
	@echo "  FEATURES=CUDA OPENCL FULL_JPEG - Enable features (default: CUDA OPENCL FULL_JPEG)"
	@echo "  CXX=compiler          - Override C++ compiler"
	@echo ""
	@echo "Examples:"
	@echo "  make external                  # Build external dependencies first"
	@echo "  make                           # Build release executable with CUDA and OpenCL"
	@echo "  make CONFIG=debug              # Build debug executable"
	@echo "  make static                    # Build static library"
	@echo "  make FEATURES=OPENCL           # Build with OpenCL only"
	@echo "  make CXX=clang++               # Use clang++ compiler"
	@echo "  make test                      # Run tests"
	@echo "  make clean                     # Clean build files"
	@echo ""
	@echo "Dependencies:"
	@echo "  - C++ Compiler (GCC, Clang, or MSVC)"
	@echo "  - CUDA Toolkit (optional, for CUDA support)"
	@echo "  - OpenCL SDK (optional, for OpenCL support)"
	@echo "  - libpng development files"
	@echo "  - Python (for header generation)"
	@echo ""
	@echo "Environment Variables:"
	@echo "  CUDA_PATH      - CUDA installation path"
	@echo "  OPENCL_SDK_PATH - OpenCL SDK installation path"
	@echo "  PNG_PATH       - libpng installation path (Windows)"
	@echo ""
	@echo "Platform-specific notes:"
	@echo "  Windows: Uses LLVM/Clang if available, falls back to MinGW-w64 GCC"
	@echo "  Linux: Uses GCC or Clang"
	@echo "  macOS: Uses Clang with libc++"

# Include dependency files (for automatic header dependencies)
# Note: Commented out to avoid makefile parsing issues with non-existent files
# To enable, uncomment these lines and add -MD flags to CXXFLAGS
# -include $(EXECUTABLE_OBJECTS:.o=.d)
# -include $(STATIC_OBJECTS:.o=.d)
# -include $(CUDA_OBJECTS:.o=.d)
