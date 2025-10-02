/*
 * LzoDec - LZO Decompression Class
 * 
 * A simple class to decompress LZO-compressed data at runtime.
 * Usage: LzoDec decompressed(compressed_data, compressed_data_size)
 */

#ifndef LZODEC_H
#define LZODEC_H

#include <vector>
#include <memory>

// Include MiniLZO headers
#include "../third_party/minilzo/minilzo.h"

class LzoDec {
public:
    // Constructor: decompress data immediately
    LzoDec(const unsigned char* compressed_data, size_t compressed_size);
    
    // Destructor
    ~LzoDec();
    
    // Get decompressed data
    const unsigned char* getData() const { return decompressed_data_.data(); }
    
    // Get decompressed data size
    size_t getSize() const { return decompressed_size_; }
    
    // Check if decompression was successful
    bool isValid() const { return valid_; }

private:
    std::vector<unsigned char> decompressed_data_;
    size_t decompressed_size_;
    bool valid_;
    
    // Disable copy constructor and assignment operator
    LzoDec(const LzoDec&) = delete;
    LzoDec& operator=(const LzoDec&) = delete;
};

#endif // LZODEC_H


