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

#include "core/encoder/encoder.h"

#include <cstdlib>
#include <cstring>
#include <memory>

#include "core/encoder/frequency_masking.h"
#include "core/encoder/output_selector.h"
#include "core/encoder/quant_matrix_search.h"
#include "core/io/jpeg_decode.h"
#include "core/encoder/debug_print.h"
#include "core/image/jpeg_data_decoder.h"
#include "core/image/jpeg_data_encoder.h"
#include "core/jpeg/jpeg_data_reader.h"
#include "core/image/output_image.h"
#include "core/jpeg/quantize.h"

namespace guetzli {

namespace {

// Butteraugli needs at least this many pixels in each dimension.
constexpr int kMinComparatorSize = 32;

// Multiplies the coefficients by their quantizers, stores the quantizers in q_in and sets all
// quantization tables to 1.
void removeOriginalQuantization(JpegData* jpg, int q_in[3][kDctBlockSize]) {
  for (int i = 0; i < 3; ++i) {
    JpegComponent& c = jpg->components[i];
    const int* q = &jpg->quant[c.quant_idx].values[0];
    std::memcpy(&q_in[i][0], q, kDctBlockSize * sizeof(q[0]));
    for (size_t j = 0; j < c.coeffs.size(); ++j) {
      c.coeffs[j] *= q[j % kDctBlockSize];
    }
  }
  int q[3][kDctBlockSize];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < kDctBlockSize; ++j) q[i][j] = 1;
  saveQuantTables(q, jpg);
}

// False if a dequantized coefficient is out of the range the encoder supports.
bool checkJpegSanity(const JpegData& jpg) {
  const int max_component = 1 << 12;
  for (const JpegComponent& comp : jpg.components) {
    const JpegQuantTable& quant_table = jpg.quant[comp.quant_idx];
    for (size_t i = 0; i < comp.coeffs.size(); i++) {
      coeff_t coeff = comp.coeffs[i];
      int quant = quant_table.values[i % kDctBlockSize];
      if (std::abs(static_cast<int64_t>(coeff) * quant) > max_component) {
        return false;
      }
    }
  }
  return true;
}

bool isGrayscale(const JpegData& jpg) {
  for (int c = 1; c < 3; ++c) {
    const JpegComponent& comp = jpg.components[c];
    for (size_t i = 0; i < comp.coeffs.size(); ++i) {
      if (comp.coeffs[i] != 0) return false;
    }
  }
  return true;
}

void downsampleImage(OutputImage* img, bool use_silver_screen) {
  if (img->component(1).factorX() > 1 || img->component(1).factorY() > 1) {
    return;
  }
  OutputImage::DownsampleConfig cfg;
  cfg.use_silver_screen = use_silver_screen;
  img->downsample(cfg);
}

}  // namespace

Encoder::Encoder(const EncoderParams& params, Backend& backend, ILogger& logger)
    : params_(params), backend_(&backend), logger_(&logger) {
  stats_.logger = &logger;
}

std::optional<std::string> Encoder::encodeJpegData(const JpegData& jpg_in, Comparator* comparator) {
  ProcessStats* stats = &stats_;
  if (jpg_in.components.size() != 3 || !hasYCbCrColorSpace(jpg_in)) {
    logger_->log(LogLevel::Error, "Only YUV color space input jpeg is supported\n");
    return std::nullopt;
  }
  bool input_is_420;
  if (jpg_in.is444()) {
    input_is_420 = false;
  } else if (jpg_in.is420()) {
    input_is_420 = true;
  } else {
    std::string factors;
    for (const JpegComponent& comp : jpg_in.components) {
      factors += " " + std::to_string(comp.h_samp_factor) + "x" + std::to_string(comp.v_samp_factor);
    }
    logger_->log(LogLevel::Error, "Unsupported sampling factors:%s\n", factors.c_str());
    return std::nullopt;
  }

  OutputSelector selector(params_.clear_metadata, stats_);
  int q_in[3][kDctBlockSize];
  // Output the original image, in case we do not manage to create anything
  // with a good enough quality.
  const std::string original = selector.serialize(jpg_in);
  GUETZLI_LOG(stats, "Original Out[%7zd]", original.size());
  if (comparator == nullptr) {
    GUETZLI_LOG(stats, " <image too small for Butteraugli>\n");
    // Butteraugli doesn't work with images this small.
    return original;
  }
  {
    JpegData jpg = jpg_in;
    removeOriginalQuantization(&jpg, q_in);
    OutputImage img(jpg.width, jpg.height, *backend_->dct);
    img.copyFromJpegData(jpg);
    comparator->compare(img);
  }
  selector.offer(original, *comparator);

  QuantMatrixSearch quant_search(*comparator, selector, stats_);
  FrequencyMasking masking(params_, *backend_->block_zeroing, *comparator, selector, stats_);
  const int try_420 = (input_is_420 || params_.force_420 || (params_.try_420 && !isGrayscale(jpg_in))) ? 1 : 0;
  const int force_420 = (input_is_420 || params_.force_420) ? 1 : 0;
  for (int downsample = force_420; downsample <= try_420; ++downsample) {
    JpegData jpg = jpg_in;
    removeOriginalQuantization(&jpg, q_in);
    OutputImage img(jpg.width, jpg.height, *backend_->dct);
    img.copyFromJpegData(jpg);
    if (downsample) {
      downsampleImage(&img, params_.use_silver_screen);
      img.saveToJpegData(&jpg);
    }
    int best_q[3][kDctBlockSize];
    std::memcpy(best_q, q_in, sizeof(best_q));
    if (!quant_search.run(jpg, downsample != 0, best_q, img)) {
      for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < kDctBlockSize; ++i) {
          best_q[c][i] = 1;
        }
      }
    }
    img.copyFromJpegData(jpg);
    img.applyGlobalQuantization(best_q);

    if (!downsample) {
      masking.run(jpg, img, 7, 1.0, false);
    } else {
      const float ymul = jpg.components.size() == 1 ? 1.0f : 0.97f;
      masking.run(jpg, img, 1, ymul, false);
      masking.run(jpg, img, 6, 1.0, true);
    }
  }

  return selector.takeBest();
}

std::optional<std::string> Encoder::encodeWithComparator(const JpegData& jpg, const std::vector<uint8_t>& rgb) {
  std::unique_ptr<Comparator> comparator;
  if (jpg.width >= kMinComparatorSize && jpg.height >= kMinComparatorSize) {
    comparator = backend_->comparators->create(jpg.width, jpg.height, &rgb, params_.butteraugli_target, &stats_);
  }
  return encodeJpegData(jpg, comparator.get());
}

std::optional<std::string> Encoder::encodeJpeg(const std::string& data) {
  JpegData jpg;
  if (!readJpeg(data, JPEG_READ_ALL, &jpg)) {
    logger_->log(LogLevel::Error, "Can't read jpg data from input file\n");
    return std::nullopt;
  }
  if (!checkJpegSanity(jpg)) {
    logger_->log(LogLevel::Error, "Unsupported input JPEG (unexpectedly large coefficient values).\n");
    return std::nullopt;
  }
  std::vector<uint8_t> rgb = decodeJpegToRgb(jpg, *backend_->dct);
  if (rgb.empty()) {
    // Not something the encoder can take as coefficients: decode to pixels and encode those.
    std::optional<RgbImage> image = decodeJpegPixels(data, *logger_);
    if (!image) return std::nullopt;
    return encodeRgb(image->rgb, image->width, image->height);
  }
  return encodeWithComparator(jpg, rgb);
}

std::optional<std::string> Encoder::encodeRgb(const std::vector<uint8_t>& rgb, int width, int height) {
  JpegData jpg;
  if (!encodeRgbToJpeg(rgb, width, height, &jpg)) {
    logger_->log(LogLevel::Error, "Could not create jpg data from rgb pixels\n");
    return std::nullopt;
  }
  return encodeWithComparator(jpg, rgb);
}

}  // namespace guetzli
