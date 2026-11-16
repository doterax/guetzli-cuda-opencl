// Estimated JPEG size while the frequency masking changes AC coefficients (todo S7).
#ifndef GUETZLI_CORE_ENCODER_SIZE_ESTIMATOR_H_
#define GUETZLI_CORE_ENCODER_SIZE_ESTIMATOR_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/jpeg/jpeg_data.h"
#include "core/jpeg/jpeg_data_writer.h"

namespace guetzli {

// Entropy-coded size in bytes of the symbols in histograms[i] coded with depths[i * JpegHistogram::kSize ...].
[[nodiscard]] size_t EntropyCodedDataSize(const std::vector<JpegHistogram>& histograms,
                                          const std::vector<uint8_t>& depths);

// Clusters histograms, stores the code length of each symbol of histogram i in
// (*depths)[i * JpegHistogram::kSize ...] and returns the size of the Huffman table headers in bytes.
size_t ComputeEntropyCodes(const std::vector<JpegHistogram>& histograms, std::vector<uint8_t>* depths);

// Estimated size of the DC data of jpg in bytes (including its Huffman tables).
[[nodiscard]] size_t EstimateDCSize(const JPEGData& jpg);

// Header size + DC size + AC Huffman tables + AC entropy-coded data. The headers, DC data and AC code
// lengths are fixed until refreshEntropyCodes(); the AC data size follows every block change in
// constant time per changed symbol, and equals EntropyCodedDataSize() over the current histograms.
class JpegSizeEstimator {
 public:
  // jpg: the image as it would be written; num_histograms >= jpg.components.size() AC histograms are
  // kept (the extra ones start empty).
  JpegSizeEstimator(const JPEGData& jpg, size_t num_histograms, bool clear_metadata);

  // Component c's block old_block (dequantized coefficients, quantizers quant) becomes new_block.
  void replaceBlock(int c, const coeff_t* old_block, const coeff_t* new_block, const int* quant);

  // Recomputes the AC Huffman codes from the current histograms.
  void refreshEntropyCodes();

  [[nodiscard]] size_t estimate() const;

  [[nodiscard]] const std::vector<JpegHistogram>& histograms() const { return histograms_; }
  [[nodiscard]] const std::vector<uint8_t>& depths() const { return depths_; }

 private:
  void addBlock(int c, const coeff_t* coeffs, const int* quant, int weight);
  void addSymbol(int c, int symbol, int weight);
  void recomputeBits();

  size_t fixed_size_;  // header + DC data
  size_t ac_histogram_size_ = 0;
  std::vector<JpegHistogram> histograms_;
  std::vector<uint8_t> depths_;
  // Per histogram: its code bits before the escape-byte estimate; and the sum of HistogramEntropyCost().
  std::vector<size_t> raw_bits_;
  size_t total_bits_ = 0;
};

}  // namespace guetzli

#endif  // GUETZLI_CORE_ENCODER_SIZE_ESTIMATOR_H_
