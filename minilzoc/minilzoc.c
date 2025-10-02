/*
 * minilzoc - MiniLZO Compression Tool (C version)
 * 
 * A simple command-line tool to compress files using the MiniLZO library.
 * Usage: minilzoc INPUT_FILE OUTPUT_COMPRESSED_FILE
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include MiniLZO headers
#include "../third_party/minilzo/minilzo.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s INPUT_FILE OUTPUT_COMPRESSED_FILE\n", argv[0]);
        fprintf(stderr, "Compress INPUT_FILE using LZO compression and save to OUTPUT_COMPRESSED_FILE\n");
        return 1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = argv[2];

    // Open input file
    FILE* input = fopen(inputFile, "rb");
    if (!input) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", inputFile);
        return 1;
    }

    // Get file size
    fseek(input, 0, SEEK_END);
    size_t inputSize = ftell(input);
    fseek(input, 0, SEEK_SET);

    if (inputSize == 0) {
        fprintf(stderr, "Error: Input file is empty\n");
        fclose(input);
        return 1;
    }

    // Read input data
    unsigned char* inputData = (unsigned char*)malloc(inputSize);
    if (!inputData) {
        fprintf(stderr, "Error: Cannot allocate memory for input data\n");
        fclose(input);
        return 1;
    }

    size_t bytesRead = fread(inputData, 1, inputSize, input);
    fclose(input);

    if (bytesRead != inputSize) {
        fprintf(stderr, "Error: Failed to read input file completely\n");
        free(inputData);
        return 1;
    }

    // Prepare compression
    // LZO1X-1 compression requires a work memory buffer
    lzo_bytep workmem = (lzo_bytep)malloc(LZO1X_1_MEM_COMPRESS);
    if (!workmem) {
        fprintf(stderr, "Error: Cannot allocate work memory\n");
        free(inputData);
        return 1;
    }
    
    // Allocate output buffer (worst case: input size + some overhead)
    lzo_uint outputSize = inputSize + inputSize / 16 + 64 + 3;
    unsigned char* outputData = (unsigned char*)malloc(outputSize);
    if (!outputData) {
        fprintf(stderr, "Error: Cannot allocate memory for output data\n");
        free(inputData);
        free(workmem);
        return 1;
    }

    // Compress the data
    int result = lzo1x_1_compress(
        inputData, inputSize,
        outputData, &outputSize,
        workmem
    );

    free(workmem);
    free(inputData);

    if (result != LZO_E_OK) {
        fprintf(stderr, "Error: Compression failed with code %d\n", result);
        free(outputData);
        return 1;
    }

    // Open output file
    FILE* output = fopen(outputFile, "wb");
    if (!output) {
        fprintf(stderr, "Error: Cannot create output file '%s'\n", outputFile);
        free(outputData);
        return 1;
    }

    // Write compressed data
    size_t bytesWritten = fwrite(outputData, 1, outputSize, output);
    fclose(output);
    free(outputData);

    if (bytesWritten != outputSize) {
        fprintf(stderr, "Error: Failed to write output file completely\n");
        return 1;
    }

    // Print compression statistics
    double compressionRatio = (double)outputSize / inputSize;
    double compressionPercent = (1.0 - compressionRatio) * 100.0;

    printf("Compression completed successfully!\n");
    printf("Input size:  %zu bytes\n", inputSize);
    printf("Output size: %u bytes\n", outputSize);
    printf("Compression ratio: %.3f (%.1f%% reduction)\n", compressionRatio, compressionPercent);

    return 0;
}


