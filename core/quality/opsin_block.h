/*
 * Host (CPU) copies of kernel helpers used by host code. See opsin_block.cc.
 */
#ifndef GUETZLI_OPSIN_BLOCK_H_
#define GUETZLI_OPSIN_BLOCK_H_

namespace guetzli {

constexpr int kOpsinBlockSize = 64;

// sRGB 8-bit -> linear table, single precision; same values as the kernels'
// kSrgb8ToLinearTable (not the double table from gamma_correct.h).
const float* Srgb8ToLinearTableF();

// In-place opsin dynamics of one 8x8 block given as linear RGB planes;
// identical to the kernel's CalcOpsinDynamicsImage.
void OpsinDynamicsBlock8x8(float rgb[3][kOpsinBlockSize]);

}  // namespace guetzli

#endif  // GUETZLI_OPSIN_BLOCK_H_
