/*
 * Copyright 2016 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "core/image/jpeg_data_decoder.h"

#include "core/image/output_image.h"

namespace guetzli {

// Mimic libjpeg's heuristics to guess jpeg color space.
// Requires that the jpg has 3 components.
bool hasYCbCrColorSpace(const JpegData& jpg) {
  bool has_adobe_marker = false;
  uint8_t adobe_transform = 0;
  for (const std::string& app : jpg.app_data) {
    if (static_cast<uint8_t>(app[0]) == 0xe0) {
      return true;
    }
    if (static_cast<uint8_t>(app[0]) == 0xee && app.size() >= 15) {
      has_adobe_marker = true;
      adobe_transform = app[14];
    }
  }
  if (has_adobe_marker) {
    return (adobe_transform != 0);
  }
  const int cid0 = jpg.components[0].id;
  const int cid1 = jpg.components[1].id;
  const int cid2 = jpg.components[2].id;
  return (cid0 != 'R' || cid1 != 'G' || cid2 != 'B');
}

std::vector<uint8_t> decodeJpegToRgb(const JpegData& jpg, IDctOps& dct) {
  if (jpg.components.size() == 3 && hasYCbCrColorSpace(jpg) && (jpg.is420() || jpg.is444())) {
    OutputImage img(jpg.width, jpg.height, dct);
    img.copyFromJpegData(jpg);
    return img.toSrgb();
  }
  return {};
}

}  // namespace guetzli
