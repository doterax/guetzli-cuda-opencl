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
#include "guetzli/debug_print.h"
#include "guetzli/jpeg_data_decoder.h"
#include "guetzli/jpeg_data_encoder.h"
#include "guetzli/jpeg_data_reader.h"
#include "guetzli/output_image.h"
#include "guetzli/quantize.h"

namespace guetzli {

namespace {

// Butteraugli needs at least this many pixels in each dimension.
constexpr int kMinComparatorSize = 32;

// Multiplies the coefficients by their quantizers, stores the quantizers in q_in and sets all
// quantization tables to 1.
void RemoveOriginalQuantization(JPEGData* jpg, int q_in[3][kDCTBlockSize]) {
  for (int i = 0; i < 3; ++i) {
    JPEGComponent& c = jpg->components[i];
    const int* q = &jpg->quant[c.quant_idx].values[0];
    std::memcpy(&q_in[i][0], q, kDCTBlockSize * sizeof(q[0]));
    for (size_t j = 0; j < c.coeffs.size(); ++j) {
      c.coeffs[j] *= q[j % kDCTBlockSize];
    }
  }
  int q[3][kDCTBlockSize];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < kDCTBlockSize; ++j) q[i][j] = 1;
  SaveQuantTables(q, jpg);
}

// False if a dequantized coefficient is out of the range the encoder supports.
bool CheckJpegSanity(const JPEGData& jpg) {
  const int kMaxComponent = 1 << 12;
  for (const JPEGComponent& comp : jpg.components) {
    const JPEGQuantTable& quant_table = jpg.quant[comp.quant_idx];
    for (size_t i = 0; i < comp.coeffs.size(); i++) {
      coeff_t coeff = comp.coeffs[i];
      int quant = quant_table.values[i % kDCTBlockSize];
      if (std::abs(static_cast<int64_t>(coeff) * quant) > kMaxComponent) {
        return false;
      }
    }
  }
  return true;
}

bool IsGrayscale(const JPEGData& jpg) {
  for (int c = 1; c < 3; ++c) {
    const JPEGComponent& comp = jpg.components[c];
    for (size_t i = 0; i < comp.coeffs.size(); ++i) {
      if (comp.coeffs[i] != 0) return false;
    }
  }
  return true;
}

void DownsampleImage(OutputImage* img, bool use_silver_screen) {
  if (img->component(1).factor_x() > 1 || img->component(1).factor_y() > 1) {
    return;
  }
  OutputImage::DownsampleConfig cfg;
  cfg.use_silver_screen = use_silver_screen;
  img->Downsample(cfg);
}

}  // namespace

Encoder::Encoder(const EncoderParams& params, Backend& backend, ILogger& logger)
    : params_(params), backend_(&backend), logger_(&logger) {
  stats_.logger = &logger;
}

std::optional<std::string> Encoder::encodeJpegData(const JPEGData& jpg_in, Comparator* comparator) {
  ProcessStats* stats = &stats_;
  if (jpg_in.components.size() != 3 || !HasYCbCrColorSpace(jpg_in)) {
    logger_->log(LogLevel::Error, "Only YUV color space input jpeg is supported\n");
    return std::nullopt;
  }
  bool input_is_420;
  if (jpg_in.Is444()) {
    input_is_420 = false;
  } else if (jpg_in.Is420()) {
    input_is_420 = true;
  } else {
    std::string factors;
    for (const JPEGComponent& comp : jpg_in.components) {
      factors += " " + std::to_string(comp.h_samp_factor) + "x" + std::to_string(comp.v_samp_factor);
    }
    logger_->log(LogLevel::Error, "Unsupported sampling factors:%s\n", factors.c_str());
    return std::nullopt;
  }

  OutputSelector selector(params_.clear_metadata, stats_);
  int q_in[3][kDCTBlockSize];
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
    JPEGData jpg = jpg_in;
    RemoveOriginalQuantization(&jpg, q_in);
    OutputImage img(jpg.width, jpg.height, *backend_->dct);
    img.CopyFromJpegData(jpg);
    comparator->Compare(img);
  }
  selector.offer(original, *comparator);

  QuantMatrixSearch quant_search(*comparator, selector, stats_);
  FrequencyMasking masking(params_, *backend_->block_zeroing, *comparator, selector, stats_);
  const int try_420 = (input_is_420 || params_.force_420 || (params_.try_420 && !IsGrayscale(jpg_in))) ? 1 : 0;
  const int force_420 = (input_is_420 || params_.force_420) ? 1 : 0;
  for (int downsample = force_420; downsample <= try_420; ++downsample) {
    JPEGData jpg = jpg_in;
    RemoveOriginalQuantization(&jpg, q_in);
    OutputImage img(jpg.width, jpg.height, *backend_->dct);
    img.CopyFromJpegData(jpg);
    if (downsample) {
      DownsampleImage(&img, params_.use_silver_screen);
      img.SaveToJpegData(&jpg);
    }
    int best_q[3][kDCTBlockSize];
    std::memcpy(best_q, q_in, sizeof(best_q));
    if (!quant_search.run(jpg, downsample != 0, best_q, img)) {
      for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < kDCTBlockSize; ++i) {
          best_q[c][i] = 1;
        }
      }
    }
    img.CopyFromJpegData(jpg);
    img.ApplyGlobalQuantization(best_q);

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

std::optional<std::string> Encoder::encodeWithComparator(const JPEGData& jpg, const std::vector<uint8_t>& rgb) {
  std::unique_ptr<Comparator> comparator;
  if (jpg.width >= kMinComparatorSize && jpg.height >= kMinComparatorSize) {
    comparator = backend_->comparators->create(jpg.width, jpg.height, &rgb, params_.butteraugli_target, &stats_);
  }
  return encodeJpegData(jpg, comparator.get());
}

std::optional<std::string> Encoder::encodeJpeg(const std::string& data) {
  JPEGData jpg;
  if (!ReadJpeg(data, JPEG_READ_ALL, &jpg)) {
    logger_->log(LogLevel::Error, "Can't read jpg data from input file\n");
    return std::nullopt;
  }
  if (!CheckJpegSanity(jpg)) {
    logger_->log(LogLevel::Error, "Unsupported input JPEG (unexpectedly large coefficient values).\n");
    return std::nullopt;
  }
  std::vector<uint8_t> rgb = DecodeJpegToRGB(jpg, *backend_->dct);
  if (rgb.empty()) {
    // Not something the encoder can take as coefficients: decode to pixels and encode those.
    std::optional<RgbImage> image = decodeJpegPixels(data, *logger_);
    if (!image) return std::nullopt;
    return encodeRgb(image->rgb, image->width, image->height);
  }
  return encodeWithComparator(jpg, rgb);
}

std::optional<std::string> Encoder::encodeRgb(const std::vector<uint8_t>& rgb, int width, int height) {
  JPEGData jpg;
  if (!EncodeRGBToJpeg(rgb, width, height, &jpg)) {
    logger_->log(LogLevel::Error, "Could not create jpg data from rgb pixels\n");
    return std::nullopt;
  }
  return encodeWithComparator(jpg, rgb);
}

}  // namespace guetzli
