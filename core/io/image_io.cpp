#include "core/io/image_io.h"

#include <cerrno>
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
  if (!f) throw IoError(errnoMessage("Can't open input file"));
  std::unique_ptr<std::FILE, int (*)(std::FILE*)> closer(from_stdin ? nullptr : f, &std::fclose);

  std::string result;
  long buffer_size = 8192;
  if (std::fseek(f, 0, SEEK_END) == 0) {
    const long size = std::ftell(f);
    buffer_size = size > 0 ? size : 1;
    if (std::fseek(f, 0, SEEK_SET) != 0) throw IoError(errnoMessage("fseek"));
  } else if (std::ferror(f)) {
    throw IoError(errnoMessage("fseek"));
  }

  std::unique_ptr<char[]> buf(new char[buffer_size]);
  while (!std::feof(f)) {
    const size_t read_bytes = std::fread(buf.get(), 1, static_cast<size_t>(buffer_size), f);
    if (std::ferror(f)) throw IoError(errnoMessage("fread"));
    result.append(buf.get(), read_bytes);
  }
  return result;
}

void writeFile(const std::filesystem::path& path, std::string_view contents) {
  const bool to_stdout = isStdStream(path);
  std::FILE* f = to_stdout ? stdout : std::fopen(path.string().c_str(), "wb");
  if (!f) throw IoError(errnoMessage("Can't open output file for writing"));
  const bool written = std::fwrite(contents.data(), 1, contents.size(), f) == contents.size();
  const int saved_errno = errno;
  const bool closed = std::fclose(f) == 0;
  if (!written) {
    errno = saved_errno;
    throw IoError(errnoMessage("fwrite"));
  }
  if (!closed) throw IoError(errnoMessage("fclose"));
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
  static const unsigned char kPNGMagicBytes[] = {
      0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n',
  };
  return data.size() >= 8 && std::memcmp(data.data(), kPNGMagicBytes, sizeof(kPNGMagicBytes)) == 0;
}

std::optional<RgbImage> decodePng(const std::string& data, AlphaBackground background) {
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png_ptr) return std::nullopt;

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
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
                  [](png_structp png_ptr, png_bytep outBytes, png_size_t byteCountToRead) {
                    std::istringstream& memstream = *static_cast<std::istringstream*>(png_get_io_ptr(png_ptr));

                    memstream.read(reinterpret_cast<char*>(outBytes), byteCountToRead);

                    if (memstream.eof()) png_error(png_ptr, "unexpected end of data");
                    if (memstream.fail()) png_error(png_ptr, "read from memory error");
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
  TiffIo* data = static_cast<TiffIo*>(fd);
  const tsize_t bytes_to_eof = (data->data + data->size) - data->fp;
  if (size > bytes_to_eof) size = bytes_to_eof;
  std::memcpy(buf, data->fp, size);
  data->fp += size;
  return size;
}
tsize_t tiffDummyWrite(thandle_t, tdata_t, tsize_t) {
  std::fprintf(stderr, "[TIFF] tiff_Write not supported\n");
  return 0;
}
toff_t tiffSeek(thandle_t fd, toff_t off, int whence) {
  TiffIo* data = static_cast<TiffIo*>(fd);
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
int tiffDummyClose(thandle_t) {
  return 0;
}
int tiffDummyMap(thandle_t, void**, toff_t*) {
  std::fprintf(stderr, "[TIFF] tiff_Map is not supported\n");
  return 0;
}
void tiffDummyUnmap(thandle_t, tdata_t, toff_t) {
  std::fprintf(stderr, "[TIFF] tiff_Unmap is not supported\n");
}
void tiffEmptyWarningHandler(const char*, const char*, va_list) {}

}  // namespace

bool isTiff(std::string_view data) {
  static const uint16_t kTIFFMagickBE = TIFF_BIGENDIAN;
  static const uint16_t kTIFFMagickLE = TIFF_LITTLEENDIAN;
  return data.size() >= 2 && (std::memcmp(data.data(), &kTIFFMagickBE, sizeof(kTIFFMagickBE)) == 0 ||
                              std::memcmp(data.data(), &kTIFFMagickLE, sizeof(kTIFFMagickLE)) == 0);
}

std::optional<RgbImage> decodeTiff(const std::string& data, AlphaBackground background, ILogger& logger) {
  TiffIo input = {data.data(), data.size(), const_cast<char*>(data.data())};

  if (!logger.enabled(LogLevel::Debug)) TIFFSetWarningHandler(tiffEmptyWarningHandler);

  TIFF* tif = TIFFClientOpen("Memory", "rm", static_cast<thandle_t>(&input), tiffRead, tiffDummyWrite, tiffSeek,
                             tiffDummyClose, tiffSize, tiffDummyMap, tiffDummyUnmap);
  if (!tif) {
    logger.log(LogLevel::Error, "[TIFF] TIFFClientOpen failed\n");
    return std::nullopt;
  }

  unsigned int width(0), height(0), components(0);
  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &components);

  const unsigned int npixels = width * height;
  auto* pixels = static_cast<unsigned int*>(_TIFFmalloc(npixels * sizeof(unsigned int)));

  if (!TIFFReadRGBAImageOriented(tif, width, height, pixels, ORIENTATION_TOPLEFT)) {
    logger.log(LogLevel::Error, "[TIFF] TIFFReadRGBAImage failed\n");
    _TIFFfree(pixels);
    TIFFClose(tif);
    return std::nullopt;
  }

  RgbImage image;
  image.width = static_cast<int>(width);
  image.height = static_cast<int>(height);
  image.rgb.resize(3 * image.width * image.height);

  for (unsigned int y = 0; y < height; ++y) {
    const unsigned int* row_in = pixels + (y * width);
    uint8_t* row_out = &image.rgb[3 * y * image.width];
    for (int x = 0; x < image.width; ++x) {
      const uint8_t* pixel = reinterpret_cast<const uint8_t*>(&row_in[x]);
      if (components != 4) {
        row_out[3 * x + 0] = pixel[0];
        row_out[3 * x + 1] = pixel[1];
        row_out[3 * x + 2] = pixel[2];
      } else {
        // FIXME: the alpha byte of an ABGR pixel is pixel[3]; pixel[4] is the next pixel's red
        // (and past the end for the last pixel). Kept as-is because fixing it changes output
        // (CONTRIBUTING.md); to be fixed in a separate, discussed change.
        const uint8_t alpha = pixel[4];
        row_out[3 * x + 0] = blend(background, pixel[0], alpha);
        row_out[3 * x + 1] = blend(background, pixel[1], alpha);
        row_out[3 * x + 2] = blend(background, pixel[2], alpha);
      }
    }
  }

  _TIFFfree(pixels);
  TIFFClose(tif);
  return image;
}

}  // namespace guetzli
