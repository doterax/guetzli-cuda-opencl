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

#include "core/image/preprocess_downsample.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <cmath>

using std::size_t;

namespace {

// convolve with size*size kernel
std::vector<float> convolve2D(const std::vector<float>& image, int w, int h, const double* kernel, int size) {
  auto result = image;
  int size2 = size / 2;
  for (size_t i = 0; i < image.size(); i++) {
    int x = i % w;
    int y = i / w;
    // Avoid non-normalized results at boundary by skipping edges.
    if (x < size2 || x + size - size2 - 1 >= w || y < size2 || y + size - size2 - 1 >= h) {
      continue;
    }
    float v = 0;
    for (int j = 0; j < size * size; j++) {
      int x2 = x + j % size - size2;
      int y2 = y + j / size - size2;
      v += static_cast<float>(kernel[j]) * image[y2 * w + x2];
    }
    result[i] = v;
  }
  return result;
}

// convolve horizontally and vertically with 1D kernel
std::vector<float> convolve2X(const std::vector<float>& image, int w, int h, const double* kernel, int size,
                              double mul) {
  auto temp = image;
  int size2 = size / 2;
  for (size_t i = 0; i < image.size(); i++) {
    int x = i % w;
    int y = i / w;
    // Avoid non-normalized results at boundary by skipping edges.
    if (x < size2 || x + size - size2 - 1 >= w) {
      continue;
    }
    float v = 0;
    for (int j = 0; j < size; j++) {
      int x2 = x + j - size2;
      v += static_cast<float>(kernel[j]) * image[y * w + x2];
    }
    temp[i] = v * static_cast<float>(mul);
  }
  auto result = temp;
  for (size_t i = 0; i < temp.size(); i++) {
    int x = i % w;
    int y = i / w;
    // Avoid non-normalized results at boundary by skipping edges.
    if (y < size2 || y + size - size2 - 1 >= h) {
      continue;
    }
    float v = 0;
    for (int j = 0; j < size; j++) {
      int y2 = y + j - size2;
      v += static_cast<float>(kernel[j]) * temp[y2 * w + x];
    }
    result[i] = v * static_cast<float>(mul);
  }
  return result;
}

double normal(double x, double sigma) {
  static const double kInvSqrt2Pi = 0.3989422804014327;
  return std::exp(-x * x / (2 * sigma * sigma)) * kInvSqrt2Pi / sigma;
}

std::vector<float> sharpen(const std::vector<float>& image, int w, int h, float sigma, float amount) {
  // This is only made for small sigma, e.g. 1.3.
  std::vector<double> kernel(5);
  const size_t center = kernel.size() / 2;
  for (size_t i = 0; i < kernel.size(); i++) {
    kernel[i] = normal(1.0 * i - static_cast<double>(center), sigma);
  }

  double sum = 0;
  for (double i : kernel) {
    sum += i;
  }
  const double mul = 1.0 / sum;

  std::vector<float> result = convolve2X(image, w, h, kernel.data(), kernel.size(), mul);
  for (size_t i = 0; i < image.size(); i++) {
    result[i] = image[i] + (image[i] - result[i]) * amount;
  }
  return result;
}

void erode(int w, int h, std::vector<bool>* image) {
  std::vector<bool> temp = *image;
  for (int y = 1; y + 1 < h; y++) {
    for (int x = 1; x + 1 < w; x++) {
      size_t index = y * w + x;
      if (!(temp[index] && temp[index - 1] && temp[index + 1] && temp[index - w] && temp[index + w])) {
        (*image)[index] = false;
      }
    }
  }
}

void dilate(int w, int h, std::vector<bool>* image) {
  std::vector<bool> temp = *image;
  for (int y = 1; y + 1 < h; y++) {
    for (int x = 1; x + 1 < w; x++) {
      size_t index = y * w + x;
      if (temp[index] || temp[index - 1] || temp[index + 1] || temp[index - w] || temp[index + w]) {
        (*image)[index] = true;
      }
    }
  }
}

std::vector<float> blur(const std::vector<float>& image, int w, int h) {
  // This is only made for small sigma, e.g. 1.3.
  static const double kSigma = 1.3;
  std::vector<double> kernel(5);
  const size_t center = kernel.size() / 2;
  for (size_t i = 0; i < kernel.size(); i++) {
    kernel[i] = normal(1.0 * i - static_cast<double>(center), kSigma);
  }

  double sum = 0;
  for (double i : kernel) {
    sum += i;
  }
  const double mul = 1.0 / sum;

  return convolve2X(image, w, h, kernel.data(), kernel.size(), mul);
}

}  // namespace

namespace guetzli {

// Do the sharpening to the v channel, but only in areas where it will help
// channel should be 2 for v sharpening, or 1 for less effective u sharpening
std::vector<std::vector<float>> preProcessChannel(int w, int h, int channel, float sigma, float amount, bool do_blur,
                                                  bool do_sharpen, const std::vector<std::vector<float>>& image) {
  if (!do_blur && !do_sharpen) {
    return image;
  }

  // Bring in range 0.0-1.0 for Y, -0.5 - 0.5 for U and V
  auto yuv = image;
  for (size_t i = 0; i < yuv[0].size(); i++) {
    yuv[0][i] /= 255.0;
    yuv[1][i] = yuv[1][i] / 255.0f - 0.5f;
    yuv[2][i] = yuv[2][i] / 255.0f - 0.5f;
  }

  // Map of areas where the image is not too bright to apply the effect.
  std::vector<bool> darkmap(image[0].size(), false);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      size_t index = y * w + x;
      float y = yuv[0][index];
      float u = yuv[1][index];
      float v = yuv[2][index];

      float r = y + 1.402f * v;
      float g = y - 0.34414f * u - 0.71414f * v;
      float b = y + 1.772f * u;

      // Parameters tuned to avoid sharpening in too bright areas, where the
      // effect makes it worse instead of better.
      if (channel == 2 && g < 0.85 && b < 0.85 && r < 0.9) {
        darkmap[index] = true;
      }
      if (channel == 1 && r < 0.85 && g < 0.85 && b < 0.9) {
        darkmap[index] = true;
      }
    }
  }

  erode(w, h, &darkmap);
  erode(w, h, &darkmap);
  erode(w, h, &darkmap);

  // Map of areas where the image is red enough (blue in case of u channel).
  std::vector<bool> redmap(image[0].size(), false);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      size_t index = y * w + x;
      float u = yuv[1][index];
      float v = yuv[2][index];

      // Parameters tuned to allow only colors on which sharpening is useful.
      if (channel == 2 && 2.116 * v > -0.34414 * u + 0.2 && 1.402 * v > 1.772 * u + 0.2) {
        redmap[index] = true;
      }
      if (channel == 1 && v < 1.263 * u - 0.1 && u > -0.33741 * v) {
        redmap[index] = true;
      }
    }
  }

  dilate(w, h, &redmap);
  dilate(w, h, &redmap);
  dilate(w, h, &redmap);

  // Map of areas where to allow sharpening by combining red and dark areas
  std::vector<bool> sharpenmap(image[0].size(), false);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      size_t index = y * w + x;
      sharpenmap[index] = redmap[index] && darkmap[index];
    }
  }

  // Threshold for where considered an edge.
  const double threshold = (channel == 2 ? 0.02 : 1.0) * 127.5;

  static const double kEdgeMatrix[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};

  // Map of areas where to allow blurring, only where it is not too sharp
  std::vector<bool> blurmap(image[0].size(), false);
  std::vector<float> edge = convolve2D(yuv[channel], w, h, kEdgeMatrix, 3);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      size_t index = y * w + x;
      float u = yuv[1][index];
      float v = yuv[2][index];
      if (sharpenmap[index]) {
        continue;
      }
      if (!darkmap[index]) {
        continue;
      }
      if (std::fabs(edge[index]) < threshold && v < -0.162 * u) {
        blurmap[index] = true;
      }
    }
  }
  erode(w, h, &blurmap);
  erode(w, h, &blurmap);

  // Choose sharpened, blurred or original per pixel
  std::vector<float> sharpened = sharpen(yuv[channel], w, h, sigma, amount);
  std::vector<float> blurred = blur(yuv[channel], w, h);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      size_t index = y * w + x;

      if (sharpenmap[index]) {
        if (do_sharpen) {
          yuv[channel][index] = sharpened[index];
        }
      } else if (blurmap[index]) {
        if (do_blur) {
          yuv[channel][index] = blurred[index];
        }
      }
    }
  }

  // Bring back to range 0-255
  for (size_t i = 0; i < yuv[0].size(); i++) {
    yuv[0][i] *= 255.0;
    yuv[1][i] = (yuv[1][i] + 0.5f) * 255.0f;
    yuv[2][i] = (yuv[2][i] + 0.5f) * 255.0f;
  }
  return yuv;
}

namespace {

inline float clip(float val) {
  return std::max(0.0f, std::min(255.0f, val));
}

inline float rgbToY(float r, float g, float b) {
  return 0.299f * r + 0.587f * g + 0.114f * b;
}

inline float rgbToU(float r, float g, float b) {
  return -0.16874f * r - 0.33126f * g + 0.5f * b + 128.0f;
}

inline float rgbToV(float r, float g, float b) {
  return 0.5f * r - 0.41869f * g - 0.08131f * b + 128.0f;
}

inline float yuvToR(float y, float u, float v) {
  return y + 1.402f * (v - 128.0f);
}

inline float yuvToG(float y, float u, float v) {
  return y - 0.344136f * (u - 128.0f) - 0.714136f * (v - 128.0f);
}

inline float yuvToB(float y, float u, float v) {
  return y + 1.772f * (u - 128.0f);
}

// TODO(user) Use SRGB->linear conversion and a lookup-table.
inline float gammaToLinear(float x) {
  return static_cast<float>(std::pow(x / 255.0f, 2.2));
}

// TODO(user) Use linear->SRGB conversion and a lookup-table.
inline float linearToGamma(float x) {
  return 255.0 * std::pow(x, 1.0 / 2.2);
}

std::vector<float> linearlyAveragedLuma(const std::vector<float>& rgb) {
  assert(rgb.size() % 3 == 0);
  std::vector<float> y(rgb.size() / 3);
  for (size_t i = 0, p = 0; p < rgb.size(); ++i, p += 3) {
    y[i] = linearToGamma(rgbToY(gammaToLinear(rgb[p + 0]), gammaToLinear(rgb[p + 1]), gammaToLinear(rgb[p + 2])));
  }
  return y;
}

std::vector<float> linearlyDownsample2x2(const std::vector<float>& rgb_in, const int width, const int height) {
  assert(rgb_in.size() == 3 * width * height);
  int w = (width + 1) / 2;
  int h = (height + 1) / 2;
  std::vector<float> rgb_out(3 * w * h);
  for (int y = 0, p = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int i = 0; i < 3; ++i, ++p) {
        rgb_out[p] = 0.0;
        for (int iy = 0; iy < 2; ++iy) {
          for (int ix = 0; ix < 2; ++ix) {
            int yy = std::min(height - 1, 2 * y + iy);
            int xx = std::min(width - 1, 2 * x + ix);
            rgb_out[p] += gammaToLinear(rgb_in[3 * (yy * width + xx) + i]);
          }
        }
        rgb_out[p] = linearToGamma(0.25f * rgb_out[p]);
      }
    }
  }
  return rgb_out;
}

std::vector<std::vector<float>> rgbToYuv(const std::vector<float>& rgb) {
  std::vector<std::vector<float>> yuv(3, std::vector<float>(rgb.size() / 3));
  for (size_t i = 0, p = 0; p < rgb.size(); ++i, p += 3) {
    const float r = rgb[p + 0];
    const float g = rgb[p + 1];
    const float b = rgb[p + 2];
    yuv[0][i] = rgbToY(r, g, b);
    yuv[1][i] = rgbToU(r, g, b);
    yuv[2][i] = rgbToV(r, g, b);
  }
  return yuv;
}

std::vector<float> yuvToRgb(const std::vector<std::vector<float>>& yuv) {
  std::vector<float> rgb(3 * yuv[0].size());
  for (size_t i = 0, p = 0; p < rgb.size(); ++i, p += 3) {
    const float y = yuv[0][i];
    const float u = yuv[1][i];
    const float v = yuv[2][i];
    rgb[p + 0] = clip(yuvToR(y, u, v));
    rgb[p + 1] = clip(yuvToG(y, u, v));
    rgb[p + 2] = clip(yuvToB(y, u, v));
  }
  return rgb;
}

// Upsamples img_in with a box-filter, and returns an image with output
// dimensions width x height.
std::vector<float> upsample2x2(const std::vector<float>& img_in, const int width, const int height) {
  int w = (width + 1) / 2;
  int h = (height + 1) / 2;
  assert(img_in.size() == w * h);
  std::vector<float> img_out(width * height);
  for (int y = 0, p = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x, ++p) {
      for (int iy = 0; iy < 2; ++iy) {
        for (int ix = 0; ix < 2; ++ix) {
          int yy = std::min(height - 1, 2 * y + iy);
          int xx = std::min(width - 1, 2 * x + ix);
          img_out[yy * width + xx] = img_in[p];
        }
      }
    }
  }
  return img_out;
}

// Apply the "fancy upsample" filter used by libjpeg.
std::vector<float> blur(const std::vector<float>& img, const int width, const int height) {
  std::vector<float> img_out(width * height);
  for (int y0 = 0; y0 < height; y0 += 2) {
    for (int x0 = 0; x0 < width; x0 += 2) {
      for (int iy = 0; iy < 2 && y0 + iy < height; ++iy) {
        for (int ix = 0; ix < 2 && x0 + ix < width; ++ix) {
          int dy = 4 * iy - 2;
          int dx = 4 * ix - 2;
          int x1 = std::min(width - 1, std::max(0, x0 + dx));
          int y1 = std::min(height - 1, std::max(0, y0 + dy));
          img_out[(y0 + iy) * width + x0 + ix] = (9.0f * img[y0 * width + x0] + 3.0f * img[y0 * width + x1] +
                                                  3.0f * img[y1 * width + x0] + 1.0f * img[y1 * width + x1]) /
                                                 16.0f;
        }
      }
    }
  }
  return img_out;
}

std::vector<float> yuv420ToRgb(const std::vector<std::vector<float>>& yuv420, const int width, const int height) {
  std::vector<std::vector<float>> yuv;
  yuv.push_back(yuv420[0]);
  std::vector<float> u = upsample2x2(yuv420[1], width, height);
  std::vector<float> v = upsample2x2(yuv420[2], width, height);
  yuv.push_back(blur(u, width, height));
  yuv.push_back(blur(v, width, height));
  return yuvToRgb(yuv);
}

void updateGuess(const std::vector<float>& target, const std::vector<float>& reconstructed, std::vector<float>* guess) {
  assert(reconstructed.size() == guess->size());
  assert(target.size() == guess->size());
  for (size_t i = 0; i < guess->size(); ++i) {
    // TODO(user): Evaluate using a decaying constant here.
    (*guess)[i] = clip((*guess)[i] - (reconstructed[i] - target[i]));
  }
}

}  // namespace

std::vector<std::vector<float>> rgbToYuv420(const std::vector<uint8_t>& rgb_in, const int width, const int height) {
  std::vector<float> rgbf(rgb_in.size());
  for (size_t i = 0; i < rgb_in.size(); ++i) {
    rgbf[i] = static_cast<float>(rgb_in[i]);
  }
  std::vector<float> y_target = linearlyAveragedLuma(rgbf);
  std::vector<std::vector<float>> yuv_target = rgbToYuv(linearlyDownsample2x2(rgbf, width, height));
  std::vector<std::vector<float>> yuv_guess = yuv_target;
  yuv_guess[0] = upsample2x2(yuv_guess[0], width, height);
  // TODO(user): Stop early if the error is small enough.
  for (int iter = 0; iter < 20; ++iter) {
    std::vector<float> rgb_rec = yuv420ToRgb(yuv_guess, width, height);
    std::vector<float> y_rec = linearlyAveragedLuma(rgb_rec);
    std::vector<std::vector<float>> yuv_rec = rgbToYuv(linearlyDownsample2x2(rgb_rec, width, height));
    updateGuess(y_target, y_rec, yuv_guess.data());
    updateGuess(yuv_target[1], yuv_rec[1], &yuv_guess[1]);
    updateGuess(yuv_target[2], yuv_rec[2], &yuv_guess[2]);
  }
  yuv_guess[1] = upsample2x2(yuv_guess[1], width, height);
  yuv_guess[2] = upsample2x2(yuv_guess[2], width, height);
  return yuv_guess;
}

}  // namespace guetzli
