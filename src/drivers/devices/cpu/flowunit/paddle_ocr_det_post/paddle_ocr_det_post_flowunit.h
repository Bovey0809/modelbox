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

#ifndef MODELBOX_FLOWUNIT_PADDLE_OCR_DET_POST_CPU_H_
#define MODELBOX_FLOWUNIT_PADDLE_OCR_DET_POST_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <opencv2/opencv.hpp>

#include "modelbox/buffer.h"

constexpr const char *FLOWUNIT_NAME = "paddle_ocr_det_post";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: PP-OCR DBNet postprocess. Probmap -> polygons.\n"
    "\t@Inputs: in_image (uint8 BGR, with resized_w/resized_h meta),\n"
    "\t  in_prob (float [1,1,H,W]).\n"
    "\t@Outputs: out_image (uint8 BGR) with meta polygons (float[]),\n"
    "\t  polygon_count (int32).\n";

class PaddleOcrDetPostFlowUnit : public modelbox::FlowUnit {
 public:
  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> ctx) override;

 private:
  float db_thresh_{0.3F};
  float box_thresh_{0.6F};
  float unclip_ratio_{1.5F};
  int max_candidates_{1000};
  int min_size_{3};
};
#endif  // MODELBOX_FLOWUNIT_PADDLE_OCR_DET_POST_CPU_H_
