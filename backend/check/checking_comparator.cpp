#include "backend/check/checking_comparator.h"

#include "backend/check/mismatch.h"

namespace guetzli {

namespace {
// GPU kernels and the optimized CPU code differ in the last bits (up to ~0.0047 seen on 4:2:0 input);
// same absolute tolerance as the per-kernel checks this replaces.
constexpr double kDistmapTolerance = 0.005;
}  // namespace

void CheckingComparator::compare(const OutputImage& img) {
  tested_->compare(img);
  reference_->compare(img);
  const std::vector<float> reference = reference_->distmap();
  const std::vector<float> tested = tested_->distmap();
  if (tested.size() != reference.size()) {
    logger_.log(LogLevel::Error, "CHK Compare distmap size %zu:%zu\n", tested.size(), reference.size());
    return;
  }
  reportMismatches(logger_, "Compare", "distmap", reference.data(), tested.data(), reference.size(), kDistmapTolerance);
  const float distance[2] = {reference_->distmapAggregate(), tested_->distmapAggregate()};
  reportMismatches(logger_, "Compare", "distance", &distance[0], &distance[1], 1, kDistmapTolerance);
}

void CheckingComparator::startBlockComparisons() {
  tested_->startBlockComparisons();
  reference_->startBlockComparisons();
}

void CheckingComparator::finishBlockComparisons() {
  tested_->finishBlockComparisons();
  reference_->finishBlockComparisons();
}

std::unique_ptr<Comparator> CheckingComparatorFactory::create(int width, int height, const std::vector<uint8_t>* rgb,
                                                              float target_distance, ProcessStats* stats) {
  return std::make_unique<CheckingComparator>(tested_->create(width, height, rgb, target_distance, stats),
                                              reference_->create(width, height, rgb, target_distance, stats), logger_);
}

}  // namespace guetzli
