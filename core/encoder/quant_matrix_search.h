// Search for the global quantization matrix (the first stage of the encoder).
#ifndef GUETZLI_CORE_ENCODER_QUANT_MATRIX_SEARCH_H_
#define GUETZLI_CORE_ENCODER_QUANT_MATRIX_SEARCH_H_

#include <cstddef>
#include <vector>

#include "core/encoder/output_selector.h"
#include "core/quality/comparator.h"
#include "core/jpeg/jpeg_data.h"
#include "core/image/output_image.h"
#include "core/encoder/stats.h"
#include "logging/logger.h"

namespace guetzli {

// A tried quantization matrix and its result.
struct QuantData {
  int q[3][kDctBlockSize];
  size_t jpg_size;
  bool dist_ok;
};

// True if a is a better result than b: within the distance target first, then smaller.
[[nodiscard]] bool compareQuantData(const QuantData& a, const QuantData& b);

// Compares a[0..3*kDCTBlockSize) and b[0..3*kDCTBlockSize) and returns
//   0 : if they are equal
//  -1 : if a is everywhere <= than b and in at least one coordinate <
//   1 : if a is everywhere >= than b and in at least one coordinate >
//   2 : if a and b are uncomparable (some coordinate smaller and some greater)
[[nodiscard]] int compareQuantMatrices(const int* a, const int* b);

// Sum of the quantizers weighted by contrast sensitivity: grows with coarser quantization.
[[nodiscard]] double quantMatrixHeuristicScore(const int q[3][kDctBlockSize]);

// Proposes quantization matrices for a binary search on the heuristic score: the lower bound is the
// coarsest matrix within the distance target, the upper bound the finest one outside it.
class QuantMatrixGenerator {
 public:
  explicit QuantMatrixGenerator(bool downsample);

  // Fills q with the next matrix to try; false when the search is done.
  bool getNext(int q[3][kDctBlockSize]);
  // Records the result of a tried matrix.
  void add(const QuantData& data);

 private:
  void getQuantMatrixWithHeuristicScore(double score, int q[3][kDctBlockSize]) const;

  const bool downsample_;
  // Lower bound for quant matrix heuristic score used in binary search.
  double hscore_a_ = -1.0;
  // Upper bound for quant matrix heuristic score used in binary search, or 0.0
  // if no upper bound is found yet.
  double hscore_b_ = -1.0;
  // Cached value of the sum of all ContrastSensitivity() values over all
  // quant matrix elements.
  double total_csf_ = 0.0;
  std::vector<QuantData> quants_;
};

// Tries the matrices of a QuantMatrixGenerator on the image; every try is offered to the selector.
class QuantMatrixSearch {
 public:
  // All arguments must outlive the search.
  QuantMatrixSearch(Comparator& comparator, OutputSelector& selector, ProcessStats& stats, ILogger& logger)
      : comparator_(&comparator), selector_(&selector), stats_(&stats), logger_(&logger) {}

  // jpg: the image with its quantization removed; img: scratch image of the same size (overwritten).
  // best_q: in, the first matrix to try; out, the selected matrix. Returns whether it meets the
  // (slightly stricter than final) distance target.
  bool run(const JpegData& jpg, bool downsample, int best_q[3][kDctBlockSize], OutputImage& img);

 private:
  QuantData tryMatrix(const JpegData& jpg, float target_mul, int q[3][kDctBlockSize], OutputImage& img);

  Comparator* comparator_;
  OutputSelector* selector_;
  ProcessStats* stats_;
  ILogger* logger_;
};

}  // namespace guetzli

#endif  // GUETZLI_CORE_ENCODER_QUANT_MATRIX_SEARCH_H_
