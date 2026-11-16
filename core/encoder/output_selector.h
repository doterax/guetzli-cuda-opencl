// Serializes candidate JPEGs and keeps the best one seen so far (by the comparator's score).
#ifndef GUETZLI_CORE_ENCODER_OUTPUT_SELECTOR_H_
#define GUETZLI_CORE_ENCODER_OUTPUT_SELECTOR_H_

#include <string>
#include <utility>

#include "guetzli/comparator.h"
#include "guetzli/jpeg_data.h"
#include "guetzli/stats.h"

namespace guetzli {

class OutputSelector {
 public:
  // stats must outlive the selector; its logger receives the score trace.
  OutputSelector(bool clear_metadata, ProcessStats& stats) : clear_metadata_(clear_metadata), stats_(&stats) {}

  // The JPEG file for jpg. Throws std::logic_error if jpg cannot be written.
  [[nodiscard]] std::string serialize(const JPEGData& jpg) const;

  // Scores encoded (the comparator's last Compare() must have been on its image) and keeps it if
  // it beats the best so far. Logs " Score[...]", " (*)" if kept, and the end of the line.
  void offer(const std::string& encoded, const Comparator& comparator);

  [[nodiscard]] bool hasOutput() const { return score_ >= 0; }
  [[nodiscard]] const std::string& best() const { return best_; }
  [[nodiscard]] double bestScore() const { return score_; }
  [[nodiscard]] std::string takeBest() { return std::move(best_); }

 private:
  bool clear_metadata_;
  ProcessStats* stats_;
  std::string best_;
  double score_ = -1;  // < 0: nothing kept yet
};

}  // namespace guetzli

#endif  // GUETZLI_CORE_ENCODER_OUTPUT_SELECTOR_H_
