// Input decoding (PNG, TIFF) and file I/O. JPEG input is handled by the encoder itself.
#ifndef GUETZLI_CORE_IO_IMAGE_IO_H_
#define GUETZLI_CORE_IO_IMAGE_IO_H_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "logging/logger.h"

namespace guetzli {

class IoError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Reads a whole file; "-" reads stdin. Throws IoError.
[[nodiscard]] std::string readFile(const std::filesystem::path& path);
// Writes a whole file; "-" writes stdout. Throws IoError.
void writeFile(const std::filesystem::path& path, std::string_view contents);

struct RgbImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgb;  // 3 bytes per pixel, row-major
};

enum class AlphaBackground { Black, White };

[[nodiscard]] bool isPng(std::string_view data);
[[nodiscard]] bool isTiff(std::string_view data);
// JPEG start-of-image marker; the encoder parses the rest.
[[nodiscard]] bool isJpeg(std::string_view data);

// nullopt on a decoding error.
[[nodiscard]] std::optional<RgbImage> decodePng(const std::string& data, AlphaBackground background);
// nullopt on a decoding error (details logged). libtiff warnings are shown only when the logger
// has LogLevel::Debug enabled.
[[nodiscard]] std::optional<RgbImage> decodeTiff(const std::string& data, AlphaBackground background, ILogger& logger);

}  // namespace guetzli

#endif  // GUETZLI_CORE_IO_IMAGE_IO_H_
