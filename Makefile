# Guetzli CUDA/OpenCL Makefile
# Portable Makefile for Windows (MinGW-w64) and Linux
# Author: AI Assistant

# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL := cmd.exe
    RM := del /Q
    RMDIR := rmdir /S /Q
    MKDIR := mkdir
    COPY := copy
    SEP := \\
    EXE_EXT := .exe
    OBJ_EXT := .o
    LIB_EXT := .a
    # Windows-specific paths
    ifdef CUDA_PATH
        CUDA_INC := $(CUDA_PATH)\include
        CUDA_LIB := $(CUDA_PATH)\lib\x64
    endif
    ifdef OPENCL_SDK_PATH
        OPENCL_INC := $(OPENCL_SDK_PATH)\include
        OPENCL_LIB := $(OPENCL_SDK_PATH)\lib\x64
    endif
else
    DETECTED_OS := Linux
    RM := rm -f
    RMDIR := rm -rf
    MKDIR := mkdir -p
    COPY := cp
    SEP := /
    EXE_EXT :=
    OBJ_EXT := .o
    LIB_EXT := .a
    # Linux-specific paths
    CUDA_INC := /usr/local/cuda/include
    CUDA_LIB := /usr/local/cuda/lib64
    OPENCL_INC := /usr/include
    OPENCL_LIB := /usr/lib/x86_64-linux-gnu
endif

# Compiler settings
CXX := g++
NVCC := nvcc
AR := ar
PYTHON := python

# Directories
SRC_DIR := .
GUETZLI_DIR := guetzli
CLGUETZLI_DIR := clguetzli
THIRD_PARTY_DIR := third_party/butteraugli
BUILD_DIR := build
BIN_DIR := bin
OBJ_DIR := $(BUILD_DIR)/obj

# Target configuration
CONFIG ?= release
TARGET_NAME := guetzli
STATIC_LIB_NAME := libguetzli_static$(LIB_EXT)

# Build directories
TARGET_DIR := $(BIN_DIR)/$(CONFIG)
OBJ_TARGET_DIR := $(OBJ_DIR)/$(CONFIG)

# Compiler flags
CXXFLAGS := -std=c++11 -I$(SRC_DIR) -I$(THIRD_PARTY_DIR) -I$(CLGUETZLI_DIR)
LDFLAGS :=

# Platform-specific includes
ifdef CUDA_INC
    CXXFLAGS += -I$(CUDA_INC)
endif
ifdef OPENCL_INC
    CXXFLAGS += -I$(OPENCL_INC)
endif

# Configuration-specific flags
ifeq ($(CONFIG),debug)
    CXXFLAGS += -g -O0 -DDEBUG
    NVCCFLAGS := -g -O0 -DDEBUG
else
    CXXFLAGS += -O3 -DNDEBUG
    NVCCFLAGS := -O3 -DNDEBUG
endif

# Feature flags
FEATURES ?= CUDA OPENCL
ifneq (,$(findstring CUDA,$(FEATURES)))
    CXXFLAGS += -D__USE_CUDA__
    LDFLAGS += -lcuda
    ifdef CUDA_LIB
        LDFLAGS += -L$(CUDA_LIB)
    endif
endif

ifneq (,$(findstring OPENCL,$(FEATURES)))
    CXXFLAGS += -D__USE_OPENCL__
    LDFLAGS += -lOpenCL
    ifdef OPENCL_LIB
        LDFLAGS += -L$(OPENCL_LIB)
    endif
endif

# PNG support
PNG_CFLAGS := $(shell pkg-config --cflags libpng 2>/dev/null || echo "")
PNG_LIBS := $(shell pkg-config --libs libpng 2>/dev/null || echo "-lpng")
CXXFLAGS += $(PNG_CFLAGS)
LDFLAGS += $(PNG_LIBS)

# Source files
GUETZLI_SOURCES := $(wildcard $(GUETZLI_DIR)/*.cc)
CLGUETZLI_SOURCES := $(wildcard $(CLGUETZLI_DIR)/*.cpp)
BUTTERAUGLI_SOURCES := $(THIRD_PARTY_DIR)/butteraugli/butteraugli.cc

# Exclude main from static library
STATIC_SOURCES := $(filter-out $(GUETZLI_DIR)/guetzli.cc,$(GUETZLI_SOURCES)) $(CLGUETZLI_SOURCES) $(BUTTERAUGLI_SOURCES)
EXECUTABLE_SOURCES := $(GUETZLI_SOURCES) $(CLGUETZLI_SOURCES) $(BUTTERAUGLI_SOURCES)

# Object files
STATIC_OBJECTS := $(STATIC_SOURCES:$(SRC_DIR)/%.cc=$(OBJ_TARGET_DIR)/%.o)
STATIC_OBJECTS := $(STATIC_OBJECTS:$(SRC_DIR)/%.cpp=$(OBJ_TARGET_DIR)/%.o)
EXECUTABLE_OBJECTS := $(EXECUTABLE_SOURCES:$(SRC_DIR)/%.cc=$(OBJ_TARGET_DIR)/%.o)
EXECUTABLE_OBJECTS := $(EXECUTABLE_OBJECTS:$(SRC_DIR)/%.cpp=$(OBJ_TARGET_DIR)/%.o)

# CUDA object files
CUDA_SOURCES := $(CLGUETZLI_DIR)/clguetzli.cu
CUDA_OBJECTS := $(CUDA_SOURCES:$(SRC_DIR)/%.cu=$(OBJ_TARGET_DIR)/%.o)

# Generated header files
GENERATED_HEADERS := $(CLGUETZLI_DIR)/clguetzli_cu_ptx.h $(CLGUETZLI_DIR)/clguetzli_cl_src.h

# Targets
TARGET := $(TARGET_DIR)/$(TARGET_NAME)$(EXE_EXT)
STATIC_TARGET := $(TARGET_DIR)/$(STATIC_LIB_NAME)

# Default target
.PHONY: all clean help static executable cuda-headers

all: $(TARGET)

static: $(STATIC_TARGET)

executable: $(TARGET)

# Create directories
$(TARGET_DIR):
	@echo Creating $(TARGET_DIR)
	$(MKDIR) $(subst /,$(SEP),$(TARGET_DIR))

$(OBJ_TARGET_DIR):
	@echo Creating $(OBJ_TARGET_DIR)
	$(MKDIR) $(subst /,$(SEP),$(OBJ_TARGET_DIR))

# Generate CUDA headers
cuda-headers: $(GENERATED_HEADERS)

$(CLGUETZLI_DIR)/clguetzli_cu_ptx.h: $(CLGUETZLI_DIR)/clguetzli.cu
	@echo Generating CUDA PTX header...
	$(NVCC) $(NVCCFLAGS) -Xcompiler "/wd 4819" -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -arch=compute_75 -ptx -o $(CLGUETZLI_DIR)/clguetzli.cu.ptx64 $(CLGUETZLI_DIR)/clguetzli.cu
	$(PYTHON) format_header.py $(CLGUETZLI_DIR)/clguetzli.cu.ptx64 $(CLGUETZLI_DIR)/clguetzli_cu_ptx.h clguetzli_cu64

$(CLGUETZLI_DIR)/clguetzli_cl_src.h: $(CLGUETZLI_DIR)/clguetzli.cl
	@echo Generating OpenCL source header...
	$(PYTHON) format_header.py $(CLGUETZLI_DIR)/clguetzli.cl $(CLGUETZLI_DIR)/clguetzli_cl_src.h clguetzli_cl_src

# Build executable
$(TARGET): $(TARGET_DIR) $(OBJ_TARGET_DIR) cuda-headers $(EXECUTABLE_OBJECTS) $(CUDA_OBJECTS)
	@echo Linking $(TARGET_NAME)...
	$(CXX) $(EXECUTABLE_OBJECTS) $(CUDA_OBJECTS) -o $@ $(LDFLAGS)

# Build static library
$(STATIC_TARGET): $(TARGET_DIR) $(OBJ_TARGET_DIR) cuda-headers $(STATIC_OBJECTS) $(CUDA_OBJECTS)
	@echo Creating static library...
	$(AR) rcs $@ $(STATIC_OBJECTS) $(CUDA_OBJECTS)

# Compile C++ source files
$(OBJ_TARGET_DIR)/%.o: $(SRC_DIR)/%.cc
	@echo Compiling $<...
	@$(MKDIR) $(subst /,$(SEP),$(dir $@))
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_TARGET_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo Compiling $<...
	@$(MKDIR) $(subst /,$(SEP),$(dir $@))
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files
$(OBJ_TARGET_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo Compiling CUDA $<...
	@$(MKDIR) $(subst /,$(SEP),$(dir $@))
	$(NVCC) $(NVCCFLAGS) -Xcompiler "/wd 4819" -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -arch=compute_75 -c $< -o $@

# Clean targets
clean:
	@echo Cleaning build files...
ifeq ($(DETECTED_OS),Windows)
	@if exist $(subst /,$(SEP),$(BUILD_DIR)) $(RMDIR) $(subst /,$(SEP),$(BUILD_DIR))
	@if exist $(subst /,$(SEP),$(BIN_DIR)) $(RMDIR) $(subst /,$(SEP),$(BIN_DIR))
	@if exist $(subst /,$(SEP),$(CLGUETZLI_DIR)/clguetzli.cu.ptx64) $(RM) $(subst /,$(SEP),$(CLGUETZLI_DIR)/clguetzli.cu.ptx64)
	@if exist $(subst /,$(SEP),$(CLGUETZLI_DIR)/clguetzli.cu.ptx32) $(RM) $(subst /,$(SEP),$(CLGUETZLI_DIR)/clguetzli.cu.ptx32)
else
	$(RMDIR) $(BUILD_DIR) $(BIN_DIR)
	$(RM) $(CLGUETZLI_DIR)/clguetzli.cu.ptx64 $(CLGUETZLI_DIR)/clguetzli.cu.ptx32
endif

# Help target
help:
	@echo "Guetzli CUDA/OpenCL Build System"
	@echo "================================="
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
	@echo "  FEATURES=CUDA OPENCL  - Enable features (default: CUDA OPENCL)"
	@echo ""
	@echo "Examples:"
	@echo "  make                           # Build release executable with CUDA and OpenCL"
	@echo "  make CONFIG=debug              # Build debug executable"
	@echo "  make static                    # Build static library"
	@echo "  make FEATURES=OPENCL           # Build with OpenCL only"
	@echo "  make clean                     # Clean build files"
	@echo ""
	@echo "Dependencies:"
	@echo "  - MinGW-w64 (Windows) or GCC (Linux)"
	@echo "  - CUDA Toolkit (for CUDA support)"
	@echo "  - OpenCL SDK (for OpenCL support)"
	@echo "  - libpng development files"
	@echo "  - Python (for header generation)"
	@echo ""
	@echo "Environment Variables:"
	@echo "  CUDA_PATH      - CUDA installation path"
	@echo "  OPENCL_SDK_PATH - OpenCL SDK installation path"

# Include dependency files
-include $(EXECUTABLE_OBJECTS:.o=.d)
-include $(STATIC_OBJECTS:.o=.d)
-include $(CUDA_OBJECTS:.o=.d)