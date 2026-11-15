// Decoding of JPEG files the encoder cannot take as coefficients (other subsampling modes, CMYK, ...).
#ifndef GUETZLI_CORE_IO_JPEG_DECODE_H_
#define GUETZLI_CORE_IO_JPEG_DECODE_H_

#include <optional>
#include <string>

#include "core/io/image_io.h"
#include "logging/logger.h"

namespace guetzli {

// Decodes data to RGB pixels with libjpeg-turbo (CMYK/YCCK converted to RGB). nullopt on a decoding
// error (logged), and always in builds without full JPEG support (__SUPPORT_FULL_JPEG__), where it
// logs that the input should be given as PNG.
[[nodiscard]] std::optional<RgbImage> decodeJpegPixels(const std::string& data, ILogger& logger);

}  // namespace guetzli

#endif  // GUETZLI_CORE_IO_JPEG_DECODE_H_
