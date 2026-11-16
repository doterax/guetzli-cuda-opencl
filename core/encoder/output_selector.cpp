#include "core/encoder/output_selector.h"

#include <stdexcept>

#include "core/encoder/debug_print.h"
#include "core/jpeg/jpeg_data_writer.h"

namespace guetzli {

namespace {

int appendToString(void* data, const uint8_t* buf, size_t count) {
  static_cast<std::string*>(data)->append(reinterpret_cast<const char*>(buf), count);
  return static_cast<int>(count);
}

}  // namespace

std::string OutputSelector::serialize(const JPEGData& jpg) const {
  std::string out;
  JPEGOutput output(appendToString, &out);
  if (!WriteJpeg(jpg, clear_metadata_, output)) {
    throw std::logic_error("could not serialize the JPEG data");
  }
  return out;
}

void OutputSelector::offer(const std::string& encoded, const Comparator& comparator) {
  const double score = comparator.ScoreOutputSize(static_cast<int>(encoded.size()));
  GUETZLI_LOG(stats_, " Score[%.4f]", score);
  if (score < score_ || score_ < 0) {
    best_ = encoded;
    score_ = score;
    GUETZLI_LOG(stats_, " (*)");
  }
  GUETZLI_LOG(stats_, "\n");
}

}  // namespace guetzli
