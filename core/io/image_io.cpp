#include "core/io/image_io.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>

#include "png.h"
#include "tiffio.h"

namespace guetzli {

namespace {

std::string errnoMessage(const char* what) {
  return std::string(what) + ": " + std::strerror(errno);
}

bool isStdStream(const std::filesystem::path& path) {
  return path == "-";
}

}  // namespace

std::string readFile(const std::filesystem::path& path) {
  const bool from_stdin = isStdStream(path);
  std::FILE* f = from_stdin ? stdin : std::fopen(path.string().c_str(), "rb");
  if (f == nullptr) {
    throw IoError(errnoMessage("Can't open input file"));
  }
  std::unique_ptr<std::FILE, int (*)(std::FILE*)> closer(from_stdin ? nullptr : f, &std::fclose);

  std::string result;
  long buffer_size = 8192;
  if (std::fseek(f, 0, SEEK_END) == 0) {
    const long size = std::ftell(f);
    buffer_size = size > 0 ? size : 1;
    if (std::fseek(f, 0, SEEK_SET) != 0) {
      throw IoError(errnoMessage("fseek"));
    }
  } else if (std::ferror(f) != 0) {
    throw IoError(errnoMessage("fseek"));
  }

  std::unique_ptr<char[]> buf(new char[buffer_size]);
  while (std::feof(f) == 0) {
    const size_t read_bytes = std::fread(buf.get(), 1, static_cast<size_t>(buffer_size), f);
    if (std::ferror(f) != 0) {
      throw IoError(errnoMessage("fread"));
    }
    result.append(buf.get(), read_bytes);
  }
  return result;
}

void writeFile(const std::filesystem::path& path, std::string_view contents) {
  const bool to_stdout = isStdStream(path);
  std::FILE* f = to_stdout ? stdout : std::fopen(path.string().c_str(), "wb");
  if (f == nullptr) {
    throw IoError(errnoMessage("Can't open output file for writing"));
  }
  const bool written = std::fwrite(contents.data(), 1, contents.size(), f) == contents.size();
  const int saved_errno = errno;
  const bool closed = std::fclose(f) == 0;
  if (!written) {
    errno = saved_errno;
    throw IoError(errnoMessage("fwrite"));
  }
  if (!closed) {
    throw IoError(errnoMessage("fclose"));
  }
}

// ---------------------------------------------------------------------------------------------
// Alpha blending
// ---------------------------------------------------------------------------------------------
namespace {

inline uint8_t blendOnBlack(const uint8_t val, const uint8_t alpha) {
  return (static_cast<int>(val) * static_cast<int>(alpha) + 128) / 255;
}
inline uint8_t blendOnWhite(const uint8_t val, const uint8_t alpha) {
  const int a = static_cast<int>(alpha);
  return (static_cast<int>(val) * a + 255 * (255 - a) + 128) / 255;
}
inline uint8_t blend(AlphaBackground background, const uint8_t val, const uint8_t alpha) {
  return background == AlphaBackground::Black ? blendOnBlack(val, alpha) : blendOnWhite(val, alpha);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// PNG
// ---------------------------------------------------------------------------------------------
bool isPng(std::string_view data) {
  static const unsigned char kPngMagicBytes[] = {
      0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n',
  };
  return data.size() >= 8 && std::memcmp(data.data(), kPngMagicBytes, sizeof(kPngMagicBytes)) == 0;
}

std::optional<RgbImage> decodePng(const std::string& data, AlphaBackground background) {
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (png_ptr == nullptr) {
    return std::nullopt;
  }

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == nullptr) {
    png_destroy_read_struct(&png_ptr, nullptr, nullptr);
    return std::nullopt;
  }

  // Declared before setjmp: libpng errors longjmp back here.
  RgbImage image;
  std::istringstream memstream(data, std::ios::in | std::ios::binary);

  if (setjmp(png_jmpbuf(png_ptr)) != 0) {
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    return std::nullopt;
  }

  png_set_read_fn(png_ptr, static_cast<void*>(&memstream),
                  [](png_structp png_ptr, png_bytep out_bytes, png_size_t byte_count_to_read) {
                    std::istringstream& memstream = *static_cast<std::istringstream*>(png_get_io_ptr(png_ptr));

                    memstream.read(reinterpret_cast<char*>(out_bytes), byte_count_to_read);

                    if (memstream.eof()) {
                      png_error(png_ptr, "unexpected end of data");
                    }
                    if (memstream.fail()) {
                      png_error(png_ptr, "read from memory error");
                    }
                  });

  // packing == convert 1,2,4 bit images, strip == 16 -> 8 bits / channel,
  // expand == palettes -> rgb, grayscale -> 8 bit images, tRNS -> alpha.
  const unsigned int png_transforms = PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND | PNG_TRANSFORM_STRIP_16;

  png_read_png(png_ptr, info_ptr, png_transforms, nullptr);

  png_bytep* row_pointers = png_get_rows(png_ptr, info_ptr);

  const int xsize = image.width = static_cast<int>(png_get_image_width(png_ptr, info_ptr));
  const int ysize = image.height = static_cast<int>(png_get_image_height(png_ptr, info_ptr));
  std::vector<uint8_t>& rgb = image.rgb;
  rgb.resize(3 * xsize * ysize);

  const int components = png_get_channels(png_ptr, info_ptr);
  switch (components) {
    case 1: {
      // GRAYSCALE
      for (int y = 0; y < ysize; ++y) {
        const uint8_t* row_in = row_pointers[y];
        uint8_t* row_out = &rgb[3 * y * xsize];
        for (int x = 0; x < xsize; ++x) {
          const uint8_t gray = row_in[x];
          row_out[3 * x + 0] = gray;
          row_out[3 * x + 1] = gray;
          row_out[3 * x + 2] = gray;
        }
      }
      break;
    }
    case 2: {
      // GRAYSCALE + ALPHA
      for (int y = 0; y < ysize; ++y) {
        const uint8_t* row_in = row_pointers[y];
        uint8_t* row_out = &rgb[3 * y * xsize];
        for (int x = 0; x < xsize; ++x) {
          const uint8_t gray = blend(background, row_in[2 * x], row_in[2 * x + 1]);
          row_out[3 * x + 0] = gray;
          row_out[3 * x + 1] = gray;
          row_out[3 * x + 2] = gray;
        }
      }
      break;
    }
    case 3: {
      // RGB
      for (int y = 0; y < ysize; ++y) {
        const uint8_t* row_in = row_pointers[y];
        uint8_t* row_out = &rgb[3 * y * xsize];
        std::memcpy(row_out, row_in, 3 * xsize);
      }
      break;
    }
    case 4: {
      // RGBA
      for (int y = 0; y < ysize; ++y) {
        const uint8_t* row_in = row_pointers[y];
        uint8_t* row_out = &rgb[3 * y * xsize];
        for (int x = 0; x < xsize; ++x) {
          const uint8_t alpha = row_in[4 * x + 3];
          row_out[3 * x + 0] = blend(background, row_in[4 * x + 0], alpha);
          row_out[3 * x + 1] = blend(background, row_in[4 * x + 1], alpha);
          row_out[3 * x + 2] = blend(background, row_in[4 * x + 2], alpha);
        }
      }
      break;
    }
    default:
      png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
      return std::nullopt;
  }
  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
  return image;
}

// ---------------------------------------------------------------------------------------------
// TIFF
// ---------------------------------------------------------------------------------------------
namespace {

struct TiffIo {
  const char* data;
  const size_t size;
  char* fp;
};

tsize_t tiffRead(thandle_t fd, tdata_t buf, tsize_t size) {
  auto* data = static_cast<TiffIo*>(fd);
  const tsize_t bytes_to_eof = (data->data + data->size) - data->fp;
  size = std::min(size, bytes_to_eof);
  std::memcpy(buf, data->fp, size);
  data->fp += size;
  return size;
}
tsize_t tiffDummyWrite(thandle_t /*unused*/, tdata_t /*unused*/, tsize_t /*unused*/) {
  std::fprintf(stderr, "[TIFF] tiff_Write not supported\n");
  return 0;
}
toff_t tiffSeek(thandle_t fd, toff_t off, int whence) {
  auto* data = static_cast<TiffIo*>(fd);
  switch (whence) {
    case SEEK_SET:
      data->fp = const_cast<char*>(data->data + off);
      break;
    case SEEK_CUR:
      data->fp = data->fp + off;
      break;
    case SEEK_END:
      data->fp = const_cast<char*>((data->data + data->size) - off);
      break;
  }
  return data->fp - data->data;
}
toff_t tiffSize(thandle_t fd) {
  return static_cast<TiffIo*>(fd)->size;
}
int tiffDummyClose(thandle_t /*unused*/) {
  return 0;
}
int tiffDummyMap(thandle_t /*unused*/, void** /*unused*/, toff_t* /*unused*/) {
  std::fprintf(stderr, "[TIFF] tiff_Map is not supported\n");
  return 0;
}
void tiffDummyUnmap(thandle_t /*unused*/, tdata_t /*unused*/, toff_t /*unused*/) {
  std::fprintf(stderr, "[TIFF] tiff_Unmap is not supported\n");
}
void tiffEmptyWarningHandler(const char* /*unused*/, const char* /*unused*/, va_list /*unused*/) {}

}  // namespace

bool isJpeg(std::string_view data) {
  return data.size() >= 2 && static_cast<unsigned char>(data[0]) == 0xFF && static_cast<unsigned char>(data[1]) == 0xD8;
}

bool isTiff(std::string_view data) {
  static const uint16_t kTiffMagickBe = TIFF_BIGENDIAN;
  static const uint16_t kTiffMagickLe = TIFF_LITTLEENDIAN;
  return data.size() >= 2 && (std::memcmp(data.data(), &kTiffMagickBe, sizeof(kTiffMagickBe)) == 0 ||
                              std::memcmp(data.data(), &kTiffMagickLe, sizeof(kTiffMagickLe)) == 0);
}

std::optional<RgbImage> decodeTiff(const std::string& data, AlphaBackground background, ILogger& logger) {
  TiffIo input = {.data = data.data(), .size = data.size(), .fp = const_cast<char*>(data.data())};

  if (!logger.enabled(LogLevel::Debug)) {
    TIFFSetWarningHandler(tiffEmptyWarningHandler);
  }

  TIFF* tif = TIFFClientOpen("Memory", "rm", static_cast<thandle_t>(&input), tiffRead, tiffDummyWrite, tiffSeek,
                             tiffDummyClose, tiffSize, tiffDummyMap, tiffDummyUnmap);
  if (tif == nullptr) {
    logger.log(LogLevel::Error, "[TIFF] TIFFClientOpen failed\n");
    return std::nullopt;
  }

  uint32_t width = 0;
  uint32_t height = 0;
  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);

  const uint64_t npixels = static_cast<uint64_t>(width) * height;
  uint32_t* pixels = npixels == 0 || npixels > (SIZE_MAX / sizeof(uint32_t))
                         ? nullptr
                         : static_cast<uint32_t*>(_TIFFmalloc(static_cast<tmsize_t>(npixels * sizeof(uint32_t))));
  if (pixels == nullptr) {
    logger.log(LogLevel::Error, "[TIFF] cannot allocate a %ux%u raster\n", static_cast<unsigned>(width),
               static_cast<unsigned>(height));
    TIFFClose(tif);
    return std::nullopt;
  }

  if (TIFFReadRGBAImageOriented(tif, width, height, pixels, ORIENTATION_TOPLEFT) == 0) {
    logger.log(LogLevel::Error, "[TIFF] TIFFReadRGBAImage failed\n");
    _TIFFfree(pixels);
    TIFFClose(tif);
    return std::nullopt;
  }

  RgbImage image;
  image.width = static_cast<int>(width);
  image.height = static_cast<int>(height);
  image.rgb.resize(3 * npixels);

  // TIFFReadRGBAImage returns 8-bit samples with *premultiplied* alpha (unassociated alpha is
  // converted; images without alpha get A = 255), packed as TIFFGetR/G/B/A expect. Compositing
  // a premultiplied pixel: on black it is the sample itself, on white sample + (255 - A).
  for (uint64_t i = 0; i < npixels; ++i) {
    const uint32_t p = pixels[i];
    const uint32_t a = TIFFGetA(p);
    const uint32_t add = background == AlphaBackground::White ? 255 - a : 0;
    const uint32_t rgb[3] = {TIFFGetR(p), TIFFGetG(p), TIFFGetB(p)};
    for (int c = 0; c < 3; ++c) {
      image.rgb[3 * i + c] = static_cast<uint8_t>(std::min<uint32_t>(255, rgb[c] + add));
    }
  }

  _TIFFfree(pixels);
  TIFFClose(tif);
  return image;
}

}  // namespace guetzli
