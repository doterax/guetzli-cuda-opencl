/*
 * LzoDec - LZO Decompression Class Implementation
 */

#include "lzodec.h"
#include <cstring>
#include <stdexcept>

LzoDec::LzoDec(const unsigned char* compressed_data, size_t compressed_size)
    : decompressed_size_(0), valid_(false) {
    
    if (!compressed_data || compressed_size == 0) {
        return;
    }
    
    // Estimate decompressed size (worst case: compressed_size * 4)
    // This is a conservative estimate for LZO compression
    size_t estimated_size = compressed_size * 4;
    
    // Allocate buffer for decompressed data
    decompressed_data_.resize(estimated_size);
    
    // Decompress the data
    lzo_uint decompressed_size = estimated_size;
    int result = lzo1x_decompress_safe(
        compressed_data, compressed_size,
        decompressed_data_.data(), &decompressed_size,
        nullptr  // workmem not needed for decompression
    );
    
    if (result == LZO_E_OK) {
        // Resize to actual decompressed size
        decompressed_data_.resize(decompressed_size);
        decompressed_size_ = decompressed_size;
        valid_ = true;
    } else {
        // Decompression failed, clear the buffer
        decompressed_data_.clear();
        decompressed_size_ = 0;
        valid_ = false;
    }
}

LzoDec::~LzoDec() {
    // Vector will automatically clean up memory
}


