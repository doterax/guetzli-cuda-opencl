// Process exit codes of the guetzli tool (also listed in --help).
#ifndef GUETZLI_APP_EXIT_CODES_H_
#define GUETZLI_APP_EXIT_CODES_H_

namespace guetzli {

inline constexpr int kExitSuccess = 0;
inline constexpr int kExitUsageOrIoError = 1;    // bad command line, unreadable input, unwritable output
inline constexpr int kExitUnsupportedInput = 2;  // input is not PNG, TIFF or JPEG
inline constexpr int kExitProcessingFailed = 3;  // corrupt input, memory limit exceeded, encoder failure

}  // namespace guetzli

#endif  // GUETZLI_APP_EXIT_CODES_H_
