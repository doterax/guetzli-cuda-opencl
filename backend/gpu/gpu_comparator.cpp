#include "backend/gpu/gpu_comparator.h"

#include <algorithm>

#include "butteraugli/butteraugli.h"
#include "clguetzli/opsin_block.h"

namespace guetzli {

namespace {
constexpr size_t kMinDeviceSize = 100;  // both dimensions must be larger
}

void GpuButteraugliOps::OpsinDynamicsImage(size_t xsize, size_t ysize, std::vector<std::vector<float>>& rgb) {
  if (xsize > kMinDeviceSize && ysize > kMinDeviceSize) {
    kernels_.opsin_dynamics_image(rgb[0].data(), rgb[1].data(), rgb[2].data(), xsize, ysize);
  } else {
    ::butteraugli::OpsinDynamicsImage(xsize, ysize, rgb);
  }
}

void GpuButteraugliOps::Mask(const std::vector<std::vector<float>>& xyb0, const std::vector<std::vector<float>>& xyb1,
                             size_t xsize, size_t ysize, std::vector<std::vector<float>>* mask,
                             std::vector<std::vector<float>>* mask_dc) {
  if (xsize > kMinDeviceSize && ysize > kMinDeviceSize) {
    mask->resize(3);
    mask_dc->resize(3);
    for (int i = 0; i < 3; i++) {
      (*mask)[i].resize(xsize * ysize);
      (*mask_dc)[i].resize(xsize * ysize);
    }
    kernels_.mask((*mask)[0].data(), (*mask)[1].data(), (*mask)[2].data(), (*mask_dc)[0].data(), (*mask_dc)[1].data(),
                  (*mask_dc)[2].data(), xsize, ysize, xyb0[0].data(), xyb0[1].data(), xyb0[2].data(), xyb1[0].data(),
                  xyb1[1].data(), xyb1[2].data());
  } else {
    ::butteraugli::Mask(xyb0, xyb1, xsize, ysize, mask, mask_dc);
  }
}

void GpuButteraugliOps::MaskHighIntensityChange(size_t xsize, size_t ysize, const std::vector<std::vector<float>>& c0,
                                                const std::vector<std::vector<float>>& c1,
                                                std::vector<std::vector<float>>& xyb0,
                                                std::vector<std::vector<float>>& xyb1) {
  ::butteraugli::MaskHighIntensityChange(xsize, ysize, c0, c1, xyb0, xyb1);
}

void GpuButteraugliComparator::Compare(const OutputImage& img) {
  distmap_.resize(static_cast<size_t>(width_) * height_);
  compareOnDevice(img, distmap_.data());
  distance_ = ::butteraugli::ButteraugliScoreFromDiffmap(distmap_);
}

void GpuButteraugliComparator::StartBlockComparisons() {
  ButteraugliComparator::StartBlockComparisons();
  if (block_lists_ready_) return;

  const int block_width = (width_ + 7) / 8;
  const int block_height = (height_ + 7) / 8;
  const int num_blocks = block_width * block_height;

  const float* lut = Srgb8ToLinearTableF();

  opsin_blocks_.resize(num_blocks * 3 * kDCTBlockSize);
  mask_scale_blocks_.resize(num_blocks * 3);
  for (int block_y = 0, block_ix = 0; block_y < block_height; ++block_y) {
    for (int block_x = 0; block_x < block_width; ++block_x, ++block_ix) {
      float* cur_r = &opsin_blocks_[block_ix * 3 * kDCTBlockSize];
      float* cur_g = cur_r + kDCTBlockSize;
      float* cur_b = cur_g + kDCTBlockSize;

      for (int iy = 0, i = 0; iy < 8; ++iy) {
        for (int ix = 0; ix < 8; ++ix, ++i) {
          int x = std::min(8 * block_x + ix, width_ - 1);
          int y = std::min(8 * block_y + iy, height_ - 1);
          int px = y * width_ + x;

          cur_r[i] = lut[rgb_orig_[3 * px]];
          cur_g[i] = lut[rgb_orig_[3 * px + 1]];
          cur_b[i] = lut[rgb_orig_[3 * px + 2]];
        }
      }

      OpsinDynamicsBlock8x8((float (*)[kOpsinBlockSize])cur_r);

      int xmin = block_x * 8;
      int ymin = block_y * 8;

      mask_scale_blocks_[block_ix * 3] = mask_xyz_[0][ymin * width_ + xmin];
      mask_scale_blocks_[block_ix * 3 + 1] = mask_xyz_[1][ymin * width_ + xmin];
      mask_scale_blocks_[block_ix * 3 + 2] = mask_xyz_[2][ymin * width_ + xmin];
    }
  }
  block_lists_ready_ = true;
}

std::optional<BlockReferenceData> GpuButteraugliComparator::blockReferenceData() const {
  if (!block_lists_ready_) return std::nullopt;
  return BlockReferenceData{opsin_blocks_, mask_scale_blocks_};
}

}  // namespace guetzli
