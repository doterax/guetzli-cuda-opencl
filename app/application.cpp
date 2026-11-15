#include "app/application.h"

#include <exception>
#include <new>
#include <optional>

#include "backend/factory.h"
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

// Transitional: the parts of the encoder that do not take the Backend yet still read g_mathMode.
MATH_MODE mathModeFor(BackendKind kind, bool check) {
  switch (kind) {
    case BackendKind::OpenCl:
      return check ? MODE_CHECKCL : MODE_OPENCL;
    case BackendKind::Cuda:
      return check ? MODE_CHECKCUDA : MODE_CUDA;
    case BackendKind::Cpu:
    case BackendKind::Auto:
      break;
  }
  return MODE_CPU_OPT;
}

}  // namespace

void Application::logMemoryLimitExceeded() {
  logger_.log(LogLevel::Error, "Memory limit (%zu MB) would be exceeded. Failing. Use --memlimit or --nomemlimit.\n",
              *config_.io.memlimit_mb);
}

bool Application::withinMemoryLimit(double pixels) const {
  if (!config_.io.memlimit_mb) return true;
  const double limit = static_cast<double>(*config_.io.memlimit_mb);
  return !(pixels * kBytesPerPixel / (1 << 20) > limit || limit < kLowestMemusageMB);
}

Application::Result Application::encode(Backend& backend, const std::string& in_data, std::string* out_data) {
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
      logMemoryLimitExceeded();
      return Result::Failed;
    }
    if (!Process(params, backend, &stats, image->rgb, image->width, image->height, out_data)) {
      logger_.log(LogLevel::Error, "Guetzli processing failed\n");
      return Result::Failed;
    }
    return Result::Success;
  }

  if (!isJpeg(in_data)) return Result::NotSupported;
  JPEGData jpg_header;
  if (!ReadJpeg(in_data, JPEG_READ_HEADER, &jpg_header)) {
    logger_.log(LogLevel::Error, "Error reading JPEG data from input file\n");
    return Result::Failed;
  }
  if (!withinMemoryLimit(static_cast<double>(jpg_header.width) * jpg_header.height)) {
    logMemoryLimitExceeded();
    return Result::Failed;
  }
  if (!Process(params, backend, &stats, in_data, out_data)) {
    logger_.log(LogLevel::Error, "Guetzli processing failed\n");
    return Result::Failed;
  }
  return Result::Success;
}

int Application::run() {
  configureGpuBackends(config_.backend, logger_);
  BackendKind kind = resolveBackendKind(config_.backend.kind, compiledInProbes(), logger_);

  std::string in_data;
  try {
    in_data = readFile(config_.io.input);
  } catch (const IoError& e) {
    logger_.log(LogLevel::Error, "%s\n", e.what());
    return kExitUsageOrIoError;
  }

  std::string out_data;
  Result result = Result::Failed;
  for (int attempt = 0; attempt < 2; ++attempt) {
    try {
      Backend backend = makeBackend(kind, config_.backend.check_against_cpu, logger_);
      g_mathMode = mathModeFor(kind, config_.backend.check_against_cpu);
      result = encode(backend, in_data, &out_data);
      break;
    } catch (const std::bad_alloc&) {
      throw;  // host OOM: retrying on the CPU would not help (main's terminate handler reports it)
    } catch (const std::exception& e) {
      if (kind == BackendKind::Cpu) throw;
      // GPU backend failure (no device, kernel build/load error, device OOM, ...):
      // redo the whole job on the CPU.
      logger_.log(LogLevel::Error, "GPU processing failed (%s); falling back to CPU (--c).\n", e.what());
      kind = BackendKind::Cpu;
      out_data.clear();
    }
  }

  switch (result) {
    case Result::NotSupported:
      logger_.log(LogLevel::Error, "Unknown file format: %s (expected PNG, TIFF or JPEG)\n",
                  config_.io.input.string().c_str());
      return kExitUnsupportedInput;
    case Result::Failed:
      return kExitProcessingFailed;  // the reason was logged where it happened
    case Result::Success:
      break;
  }

  logger_.log(LogLevel::Info, "Guetzli process succeeded, writing ... ");
  try {
    writeFile(config_.io.output, out_data);
  } catch (const IoError& e) {
    logger_.log(LogLevel::Error, "%s\n", e.what());
    return kExitUsageOrIoError;
  }
  logger_.log(LogLevel::Info, "Done.\n\n");
  logger_.log(LogLevel::Info, "%s", kSupportMessage);
  return kExitSuccess;
}

}  // namespace guetzli
