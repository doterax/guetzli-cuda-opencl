# Makefile for minilzoc compression tool

CXX = g++
CXXFLAGS = -O2 -Wall -std=c++11
INCLUDES = -I../third_party/minilzo
LIBS = 

# Source files
SOURCES = minilzoc.cpp ../third_party/minilzo/minilzo.c
TARGET = minilzoc

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES) $(LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Install to bin directory
install: $(TARGET)
	mkdir -p ./bin
	mv $(TARGET) ./bin/

.PHONY: all clean install


