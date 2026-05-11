/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
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

#ifndef MODELBOX_FLOWUNIT_PADDLE_OCR_CROP_ROTATE_CPU_H_
#define MODELBOX_FLOWUNIT_PADDLE_OCR_CROP_ROTATE_CPU_H_

#include <modelbox/flowunit.h>
#include <opencv2/opencv.hpp>

#include <atomic>

constexpr const char *FLOWUNIT_NAME = "paddle_ocr_crop_rotate";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Perspective-warp polygons from a frame into N crops.\n"
    "\t@Inputs: in_image (uint8 BGR with polygons meta).\n"
    "\t@Outputs: out_crop (uint8 BGR, N per input frame, tagged with\n"
    "\t  frame_id/box_id/box_count/polygon meta),\n"
    "\t  out_image (uint8 BGR forwarded with frame_id/box_count meta).\n";

class PaddleOcrCropRotateFlowUnit : public modelbox::FlowUnit {
 public:
  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> ctx) override;

 private:
  std::atomic<int64_t> next_frame_id_{0};
};
#endif  // MODELBOX_FLOWUNIT_PADDLE_OCR_CROP_ROTATE_CPU_H_
