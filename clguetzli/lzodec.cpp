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
    
    // The decompressed size is not stored in the stream: start with a 4x
    // estimate and grow on LZO_E_OUTPUT_OVERRUN (PTX compresses ~3.3:1, but
    // there is no hard upper bound on the ratio).
    size_t capacity = compressed_size * 4;
    const size_t max_capacity = compressed_size * 1024 + 1024;
    for (;;) {
        // +1: room for a NUL terminator (PTX / OpenCL source are used as C strings).
        decompressed_data_.resize(capacity + 1);
        lzo_uint out_size = capacity;
        int result = lzo1x_decompress_safe(
            compressed_data, compressed_size,
            decompressed_data_.data(), &out_size,
            nullptr  // workmem not needed for decompression
        );
        if (result == LZO_E_OK) {
            decompressed_data_.resize(out_size + 1);
            decompressed_data_[out_size] = 0;
            decompressed_size_ = out_size;
            valid_ = true;
            return;
        }
        if (result != LZO_E_OUTPUT_OVERRUN || capacity >= max_capacity) {
            break;
        }
        capacity *= 2;
    }

    // Decompression failed
    decompressed_data_.assign(1, 0);
    decompressed_size_ = 0;
    valid_ = false;
}

LzoDec::~LzoDec() {
    // Vector will automatically clean up memory
}


