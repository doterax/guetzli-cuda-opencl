#include "app/application.h"

#include <exception>
#include <new>
#include <optional>

#include "clguetzli/clguetzli.h"
#include "core/io/image_io.h"
#include "guetzli/jpeg_data.h"
#include "guetzli/jpeg_data_reader.h"
#include "guetzli/processor.h"
#include "guetzli/quality.h"
#include "guetzli/stats.h"

namespace guetzli {

namespace {

// An upper estimate of memory usage of Guetzli. The bound is
// max(kLowestMemusageMB * 1<<20, pixel_count * kBytesPerPixel)
constexpr int kBytesPerPixel = 110;
constexpr int kLowestMemusageMB = 100;  // in MB

constexpr const char* kSupportMessage =
    "Support this project:\n"
    "  If you find this tool useful, please consider supporting the development\n"
    "  by buying me a coffee: https://www.buymeacoffee.com/doterax \n"
    "  Your support helps maintain and improve this project!\n\n";

MATH_MODE mathModeFor(const BackendOptions& backend) {
  switch (backend.kind) {
    case BackendKind::Cpu:
      return MODE_CPU_OPT;
    case BackendKind::OpenCl:
      return backend.check_against_cpu ? MODE_CHECKCL : MODE_OPENCL;
    case BackendKind::Cuda:
      return backend.check_against_cpu ? MODE_CHECKCUDA : MODE_CUDA;
    case BackendKind::Auto:
      break;
  }
  return MODE_AUTO;
}

}  // namespace

bool Application::withinMemoryLimit(double pixels) const {
  if (!config_.io.memlimit_mb) return true;
  const double limit = static_cast<double>(*config_.io.memlimit_mb);
  return !(pixels * kBytesPerPixel / (1 << 20) > limit || limit < kLowestMemusageMB);
}

void Application::selectBackend() {
  GpuOptions gpu;
  gpu.device = config_.backend.device.value_or(-1);
  gpu.verbose = logger_.enabled(LogLevel::Debug);
#ifdef __USE_OPENCL__
  configureOpenCl(gpu);
#endif
#ifdef __USE_CUDA__
  configureCuda(gpu);
#endif

  g_mathMode = mathModeFor(config_.backend);
  if (g_mathMode != MODE_AUTO) return;

  logger_.log(LogLevel::Info, "Autodetect:\n");
#ifdef __USE_CUDA__
  logger_.log(LogLevel::Info, "  Try CUDA...\n");
  if (supportsCuda()) {
    logger_.log(LogLevel::Info, "  CUDA detected.\n");
    g_mathMode = MODE_CUDA;
    return;
  }
  logger_.log(LogLevel::Info, "    CUDA is not supported.\n");
#endif
#ifdef __USE_OPENCL__
  logger_.log(LogLevel::Info, "  Try OpenCL...\n");
  if (supportsOpenCl()) {
    logger_.log(LogLevel::Info, "  OpenCL detected.\n");
    g_mathMode = MODE_OPENCL;
    return;
  }
  logger_.log(LogLevel::Info, "    OpenCL is not supported.\n");
#endif
  logger_.log(LogLevel::Info, "  Using optimized CPU implementation (--c).\n");
  g_mathMode = MODE_CPU_OPT;
}

Application::Result Application::encode(const std::string& in_data, std::string* out_data) {
  Params params;
  params.butteraugli_target = static_cast<float>(ButteraugliScoreForQuality(config_.encoder.quality));
  ProcessStats stats;
  stats.logger = &logger_;
  const AlphaBackground background = config_.encoder.blend_on_white ? AlphaBackground::White : AlphaBackground::Black;

  std::optional<RgbImage> image;
  if (isPng(in_data)) {
    image = decodePng(in_data, background);
    if (!image) {
      logger_.log(LogLevel::Error, "Error reading PNG data from input file\n");
      return Result::Failed;
    }
  } else if (isTiff(in_data)) {
    image = decodeTiff(in_data, background, logger_);
    if (!image) {
      logger_.log(LogLevel::Error, "Error reading TIFF data from input file\n");
      return Result::Failed;
    }
  }

  if (image) {
    if (!withinMemoryLimit(static_cast<double>(image->width) * image->height)) {
      logger_.log(LogLevel::Error, "Memory limit would be exceeded. Failing.\n");
      return Result::Failed;
    }
    if (!Process(params, &stats, image->rgb, image->width, image->height, out_data)) {
      logger_.log(LogLevel::Error, "Guetzli processing failed\n");
      return Result::Failed;
    }
    return Result::Success;
  }

  // Anything else is tried as JPEG.
  JPEGData jpg_header;
  if (!ReadJpeg(in_data, JPEG_READ_HEADER, &jpg_header)) {
    logger_.log(LogLevel::Error, "Error reading JPG data from input file\n");
    return Result::NotSupported;
  }
  if (!withinMemoryLimit(static_cast<double>(jpg_header.width) * jpg_header.height)) {
    logger_.log(LogLevel::Error, "Memory limit would be exceeded. Failing.\n");
    return Result::Failed;
  }
  if (!Process(params, &stats, in_data, out_data)) {
    logger_.log(LogLevel::Error, "Guetzli processing failed\n");
    return Result::Failed;
  }
  return Result::Success;
}

int Application::run() {
  selectBackend();

  std::string in_data;
  try {
    in_data = readFile(config_.io.input);
  } catch (const IoError& e) {
    logger_.log(LogLevel::Error, "%s\n", e.what());
    return 1;
  }

  std::string out_data;
  Result result = Result::Failed;
  for (int attempt = 0; attempt < 2; ++attempt) {
    try {
      result = encode(in_data, &out_data);
      break;
    } catch (const std::bad_alloc&) {
      throw;  // host OOM: retrying on the CPU would not help (main's terminate handler reports it)
    } catch (const std::exception& e) {
      if (g_mathMode == MODE_CPU_OPT || g_mathMode == MODE_CPU) throw;
      // GPU backend failure (no device, kernel build/load error, device OOM, ...):
      // redo the whole job on the CPU.
      logger_.log(LogLevel::Error, "GPU processing failed (%s); falling back to CPU (--c).\n", e.what());
      g_mathMode = MODE_CPU_OPT;
      out_data.clear();
    }
  }

  if (result != Result::Success) {
    // Exit code and message as before for every failure (not only unsupported formats).
    logger_.log(LogLevel::Error, "Unknown file format: %s\n", config_.io.input.string().c_str());
    return 2;
  }

  logger_.log(LogLevel::Info, "Guetzli process suceeded, writing ... ");
  try {
    writeFile(config_.io.output, out_data);
  } catch (const IoError& e) {
    logger_.log(LogLevel::Error, "%s\n", e.what());
    return 1;
  }
  logger_.log(LogLevel::Info, "Done.\n\n");
  logger_.log(LogLevel::Info, "%s", kSupportMessage);
  return 0;
}

}  // namespace guetzli
