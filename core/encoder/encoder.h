// Encodes one image: quantization-matrix search, then frequency masking, for 4:4:4 and (when asked
// or when the input is 4:2:0) 4:2:0, keeping the smallest output within the distance target.
#ifndef GUETZLI_CORE_ENCODER_ENCODER_H_
#define GUETZLI_CORE_ENCODER_ENCODER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "core/encoder/encoder_params.h"
#include "core/encoder/backend.h"
#include "core/quality/comparator.h"
#include "core/jpeg/jpeg_data.h"
#include "core/encoder/stats.h"
#include "logging/logger.h"

namespace guetzli {

class Encoder {
 public:
  // backend and logger must outlive the encoder. Errors go to logger (LogLevel::Error), the
  // --verbose trace to LogLevel::Debug.
  Encoder(const EncoderParams& params, Backend& backend, ILogger& logger);

  // Re-encodes a JPEG file. nullopt on failure (logged).
  [[nodiscard]] std::optional<std::string> encodeJpeg(const std::string& data);

  // Encodes RGB pixels (3 bytes per pixel, row-major). nullopt on failure (logged).
  [[nodiscard]] std::optional<std::string> encodeRgb(const std::vector<uint8_t>& rgb, int width, int height);

  // The encoder pipeline on jpg; comparator is null for images too small for Butteraugli (the input
  // is then returned unchanged). nullopt if jpg is not 4:4:4 or 4:2:0 YCbCr.
  [[nodiscard]] std::optional<std::string> encodeJpegData(const JpegData& jpg, Comparator* comparator);

  [[nodiscard]] const ProcessStats& stats() const { return stats_; }

 private:
  [[nodiscard]] std::optional<std::string> encodeWithComparator(const JpegData& jpg, const std::vector<uint8_t>& rgb);

  EncoderParams params_;
  Backend* backend_;
  ILogger* logger_;
  ProcessStats stats_;
};

}  // namespace guetzli

#endif  // GUETZLI_CORE_ENCODER_ENCODER_H_
