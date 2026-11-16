/*
 * Copyright 2016 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GUETZLI_OUTPUT_IMAGE_H_
#define GUETZLI_OUTPUT_IMAGE_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/image/dct_ops.h"
#include "core/jpeg/jpeg_data.h"

namespace guetzli {

class OutputImageComponent {
 public:
  OutputImageComponent(int w, int h);

  void reset(int factor_x, int factor_y);

  [[nodiscard]] int width() const { return width_; }
  [[nodiscard]] int height() const { return height_; }
  [[nodiscard]] int factorX() const { return factor_x_; }
  [[nodiscard]] int factorY() const { return factor_y_; }
  [[nodiscard]] int widthInBlocks() const { return width_in_blocks_; }
  [[nodiscard]] int heightInBlocks() const { return height_in_blocks_; }
  [[nodiscard]] const coeff_t* coeffs() const { return coeffs_.data(); }
  [[nodiscard]] const uint16_t* pixels() const { return pixels_.data(); }
  [[nodiscard]] size_t pixelsSize() const { return pixels_.size(); }
  [[nodiscard]] const int* quant() const { return quant_.data(); }
  [[nodiscard]] bool isAllZero() const;

  // Fills in block[] with the 8x8 coefficient block with block coordinates
  // (block_x, block_y).
  // NOTE: If the component is 2x2 subsampled, this corresponds to the 16x16
  // pixel area with upper-left corner (16 * block_x, 16 * block_y).
  void getCoeffBlock(int block_x, int block_y, coeff_t block[kDctBlockSize]) const;

  // Fills in out[] array with the 8-bit pixel view of this component cropped
  // to the specified window. The window's upper-left corner, (xmin, ymin) must
  // be within the image, but the window may extend past the image. In that
  // case the edge pixels are duplicated.
  void toPixels(int xmin, int ymin, int xsize, int ysize, uint8_t* out, int stride) const;

  // Fills in out[] array with the floating-point precision pixel view of the
  // component.
  // REQUIRES: factor_x() == 1 and factor_y() == 1.
  void toFloatPixels(float* out, int stride) const;

  // Sets the 8x8 coefficient block with block coordinates (block_x, block_y)
  // to block[].
  // NOTE: If the component is 2x2 subsampled, this corresponds to the 16x16
  // pixel area with upper-left corner (16 * block_x, 16 * block_y).
  // REQUIRES: block[k] % quant()[k] == 0 for each coefficient index k.
  void setCoeffBlock(int block_x, int block_y, const coeff_t block[kDctBlockSize]);

  // CPU implementations of IDctOps (OutputImage goes through its IDctOps).
  // Requires that comp is not downsampled.
  void copyFromJpegComponent(const JpegComponent& comp, int factor_x, int factor_y, const int* quant);
  void applyGlobalQuantization(const int q[kDctBlockSize]);

  // Low-level access for the device implementations of IDctOps.
  coeff_t* mutableCoeffs() { return coeffs_.data(); }
  void updatePixelsForBlock(int block_x, int block_y, const uint8_t idct[kDctBlockSize]);
  void setQuant(const int q[kDctBlockSize]);

 private:
  const int width_;
  const int height_;
  int factor_x_;
  int factor_y_;
  int width_in_blocks_;
  int height_in_blocks_;
  int num_blocks_;
  std::vector<coeff_t> coeffs_;
  std::vector<uint16_t> pixels_;
  // Same as last argument of ApplyGlobalQuantization() (default is all 1s).
  std::array<int, kDctBlockSize> quant_;
};

class OutputImage {
 public:
  // ops runs the coefficient/pixel operations; it must outlive the image (and its copies).
  OutputImage(int w, int h, IDctOps& ops);

  [[nodiscard]] int width() const { return width_; }
  [[nodiscard]] int height() const { return height_; }

  OutputImageComponent& component(int c) { return components_[c]; }
  [[nodiscard]] const OutputImageComponent& component(int c) const { return components_[c]; }
  [[nodiscard]] const std::vector<OutputImageComponent>& components() const { return components_; }

  // Requires that jpg is in YUV444 format.
  void copyFromJpegData(const JpegData& jpg);

  void applyGlobalQuantization(const int q[3][kDctBlockSize]);

  // If sharpen or blur are enabled, preprocesses image before downsampling U or
  // V to improve butteraugli score and/or reduce file size.
  // u_sharpen: sharpen the u channel in red areas to improve score (not as
  // effective as v_sharpen, blue is not so important)
  // u_blur: blur the u channel in some areas to reduce file size
  // v_sharpen: sharpen the v channel in red areas to improve score
  // v_blur: blur the v channel in some areas to reduce file size
  struct DownsampleConfig {  // default: YUV420
    int u_factor_x = 2;
    int u_factor_y = 2;
    int v_factor_x = 2;
    int v_factor_y = 2;
    bool u_sharpen = true;
    bool u_blur = true;
    bool v_sharpen = true;
    bool v_blur = true;
    bool use_silver_screen = false;
  };

  void downsample(const DownsampleConfig& cfg);

  void saveToJpegData(JpegData* jpg) const;

  [[nodiscard]] std::vector<uint8_t> toSrgb() const;

  [[nodiscard]] std::vector<uint8_t> toSrgb(int xmin, int ymin, int xsize, int ysize) const;

  void toLinearRgb(std::vector<std::vector<float>>* rgb) const;

  void toLinearRgb(int xmin, int ymin, int xsize, int ysize, std::vector<std::vector<float>>* rgb) const;

  [[nodiscard]] std::string frameTypeStr() const;

 private:
  const int width_;
  const int height_;
  IDctOps* ops_;
  std::vector<OutputImageComponent> components_;
};

}  // namespace guetzli

#endif  // GUETZLI_OUTPUT_IMAGE_H_
