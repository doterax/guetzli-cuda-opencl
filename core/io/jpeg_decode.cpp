#include "core/io/jpeg_decode.h"

#ifdef __SUPPORT_FULL_JPEG__
#include <csetjmp>
#include <cstdio>
#include <vector>

#include "jpeglib.h"
#include "turbojpeg.h"
#endif

namespace guetzli {

#ifdef __SUPPORT_FULL_JPEG__

namespace {

void cmykToRgb(const unsigned char* src, unsigned char* dst, size_t size) {
  for (size_t offset = 0; offset < size; offset += 4) {
    unsigned char c = *src++;
    unsigned char m = *src++;
    unsigned char y = *src++;
    unsigned char k = *src++;
    *dst++ = (c * k + c + 128) >> 8;
    *dst++ = (m * k + m + 128) >> 8;
    *dst++ = (y * k + y + 128) >> 8;
  }
}

// libjpeg reports fatal errors through error_exit, which must not return; jump back instead of exiting.
struct JpegErrorManager {
  jpeg_error_mgr pub;
  std::jmp_buf jump;
  char message[JMSG_LENGTH_MAX];
};

void onJpegError(j_common_ptr cinfo) {
  auto* err = reinterpret_cast<JpegErrorManager*>(cinfo->err);
  (*cinfo->err->format_message)(cinfo, err->message);
  std::longjmp(err->jump, 1);
}

// No C++ objects with destructors may live in this function's frame across setjmp.
bool decompressWithLibjpeg(const std::string& data, RgbImage& image, ILogger& logger) {
  jpeg_decompress_struct cinfo;
  JpegErrorManager jerr;
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = onJpegError;
  jerr.message[0] = '\0';
  if (setjmp(jerr.jump)) {
    jpeg_destroy_decompress(&cinfo);
    logger.log(LogLevel::Error, "Could not decode the JPEG file: %s\n", jerr.message);
    return false;
  }
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, reinterpret_cast<const unsigned char*>(data.data()), static_cast<unsigned long>(data.size()));
  if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
    jpeg_destroy_decompress(&cinfo);
    logger.log(LogLevel::Error, "File does not seem to be a normal JPEG\n");
    return false;
  }
  cinfo.out_color_space = JCS_RGB;  // force RGB output
  jpeg_start_decompress(&cinfo);
  image.width = static_cast<int>(cinfo.output_width);
  image.height = static_cast<int>(cinfo.output_height);
  const size_t row_stride = static_cast<size_t>(cinfo.output_width) * cinfo.output_components;
  image.rgb.resize(row_stride * cinfo.output_height);
  while (cinfo.output_scanline < cinfo.output_height) {
    unsigned char* row = image.rgb.data() + cinfo.output_scanline * row_stride;
    jpeg_read_scanlines(&cinfo, &row, 1);
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  return true;
}

bool decompressWithTurbo(const std::string& data, tjhandle handle, int width, int height, int colorspace,
                         RgbImage& image, ILogger& logger) {
  auto* src = reinterpret_cast<unsigned char*>(const_cast<char*>(data.data()));
  const auto src_size = static_cast<unsigned long>(data.size());
  image.width = width;
  image.height = height;
  image.rgb.resize(static_cast<size_t>(tjPixelSize[TJPF_RGB]) * width * height);
  int result;
  if (colorspace == TJCS_CMYK || colorspace == TJCS_YCCK) {
    const int pitch = tjPixelSize[TJPF_CMYK] * width;
    std::vector<uint8_t> cmyk(static_cast<size_t>(pitch) * height);
    result = tjDecompress2(handle, src, src_size, cmyk.data(), width, pitch, height, TJPF_CMYK, 0);
    if (result == 0) {
      cmykToRgb(cmyk.data(), image.rgb.data(), cmyk.size());
    }
  } else {
    const int pitch = tjPixelSize[TJPF_RGB] * width;
    result = tjDecompress2(handle, src, src_size, image.rgb.data(), width, pitch, height, TJPF_RGB, 0);
  }
  if (result != 0) {
    logger.log(LogLevel::Error, "tjDecompress2() failed: %s\n for source color space %i\n", tjGetErrorStr(),
               colorspace);
    return false;
  }
  return true;
}

}  // namespace

std::optional<RgbImage> decodeJpegPixels(const std::string& data, ILogger& logger) {
  tjhandle handle = tjInitDecompress();
  if (handle == nullptr) {
    logger.log(LogLevel::Error, "tjInitDecompress() failed: %s\n", tjGetErrorStr());
    return std::nullopt;
  }
  RgbImage image;
  int width;
  int height;
  int subsampling;
  int colorspace;
  bool ok;
  if (tjDecompressHeader3(handle, reinterpret_cast<unsigned char*>(const_cast<char*>(data.data())),
                          static_cast<unsigned long>(data.size()), &width, &height, &subsampling, &colorspace) != 0) {
    ok = decompressWithLibjpeg(data, image, logger);
  } else {
    ok = decompressWithTurbo(data, handle, width, height, colorspace, image, logger);
  }
  tjDestroy(handle);
  if (!ok) {
    return std::nullopt;
  }
  return image;
}

#else

std::optional<RgbImage> decodeJpegPixels(const std::string&, ILogger& logger) {
  logger.log(LogLevel::Error,
             "Unsupported input JPEG file (e.g. unsupported "
             "downsampling mode).\nPlease provide the input image as "
             "a PNG file.\n");
  return std::nullopt;
}

#endif

}  // namespace guetzli
