/*
 * minilzoc - MiniLZO Compression Tool
 * 
 * A simple command-line tool to compress files using the MiniLZO library.
 * Usage: minilzoc INPUT_FILE OUTPUT_COMPRESSED_FILE
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdlib>

// Include MiniLZO headers
#include "../third_party/minilzo/minilzo.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " INPUT_FILE OUTPUT_COMPRESSED_FILE" << std::endl;
        std::cerr << "Compress INPUT_FILE using LZO compression and save to OUTPUT_COMPRESSED_FILE" << std::endl;
        return 1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = argv[2];

    // Open input file
    std::ifstream input(inputFile, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Error: Cannot open input file '" << inputFile << "'" << std::endl;
        return 1;
    }

    // Get file size
    input.seekg(0, std::ios::end);
    size_t inputSize = input.tellg();
    input.seekg(0, std::ios::beg);

    if (inputSize == 0) {
        std::cerr << "Error: Input file is empty" << std::endl;
        return 1;
    }

    // Read input data
    std::vector<unsigned char> inputData(inputSize);
    input.read(reinterpret_cast<char*>(inputData.data()), inputSize);
    input.close();

    // Prepare compression
    // LZO1X-1 compression requires a work memory buffer
    lzo_bytep workmem = new lzo_byte[LZO1X_1_MEM_COMPRESS];
    
    // Allocate output buffer (worst case: input size + some overhead)
    lzo_uint outputSize = inputSize + inputSize / 16 + 64 + 3;
    std::vector<unsigned char> outputData(outputSize);

    // Compress the data
    int result = lzo1x_1_compress(
        inputData.data(), inputSize,
        outputData.data(), &outputSize,
        workmem
    );

    delete[] workmem;

    if (result != LZO_E_OK) {
        std::cerr << "Error: Compression failed with code " << result << std::endl;
        return 1;
    }

    // Open output file
    std::ofstream output(outputFile, std::ios::binary);
    if (!output.is_open()) {
        std::cerr << "Error: Cannot create output file '" << outputFile << "'" << std::endl;
        return 1;
    }

    // Write compressed data
    output.write(reinterpret_cast<char*>(outputData.data()), outputSize);
    output.close();

    // Print compression statistics
    double compressionRatio = (double)outputSize / inputSize;
    double compressionPercent = (1.0 - compressionRatio) * 100.0;

    std::cout << "Compression completed successfully!" << std::endl;
    std::cout << "Input size:  " << inputSize << " bytes" << std::endl;
    std::cout << "Output size: " << outputSize << " bytes" << std::endl;
    std::cout << "Compression ratio: " << compressionRatio << " (" << compressionPercent << "% reduction)" << std::endl;

    return 0;
}


