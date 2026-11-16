#include "core/encoder/output_selector.h"

#include <stdexcept>

#include "core/jpeg/jpeg_data_writer.h"

namespace guetzli {

namespace {

int appendToString(void* data, const uint8_t* buf, size_t count) {
  static_cast<std::string*>(data)->append(reinterpret_cast<const char*>(buf), count);
  return static_cast<int>(count);
}

}  // namespace

std::string OutputSelector::serialize(const JpegData& jpg) const {
  std::string out;
  JpegOutput output(appendToString, &out);
  if (!writeJpeg(jpg, clear_metadata_, output)) {
    throw std::logic_error("could not serialize the JPEG data");
  }
  return out;
}

std::string OutputSelector::serialize(const OutputImage& img, const JpegData& jpg, JpegData* scratch) const {
  // A different component count means saveToJpegData() dropped or added components: start from jpg again so the
  // fields it does not set are jpg's.
  if (scratch->components.size() != jpg.components.size()) {
    *scratch = jpg;
  }
  img.saveToJpegData(scratch);
  return serialize(*scratch);
}

void OutputSelector::offer(const std::string& encoded, const Comparator& comparator) {
  const double score = comparator.scoreOutputSize(static_cast<int>(encoded.size()));
  logger_->log(LogLevel::Debug, " Score[%.4f]", score);
  if (score < score_ || score_ < 0) {
    best_ = encoded;
    score_ = score;
    logger_->log(LogLevel::Debug, " (*)");
  }
  logger_->log(LogLevel::Debug, "\n");
}

}  // namespace guetzli
