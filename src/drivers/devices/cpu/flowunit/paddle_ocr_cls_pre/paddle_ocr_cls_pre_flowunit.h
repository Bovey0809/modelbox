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

#ifndef MODELBOX_FLOWUNIT_PADDLE_OCR_CLS_PRE_CPU_H_
#define MODELBOX_FLOWUNIT_PADDLE_OCR_CLS_PRE_CPU_H_

#include <modelbox/flowunit.h>
#include <opencv2/opencv.hpp>

constexpr const char *FLOWUNIT_NAME = "paddle_ocr_cls_pre";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: PP-OCR angle classifier preprocess: fixed 192x48 + mean/std "
    "0.5 normalize.\n"
    "\t@Inputs: in_crop (uint8 BGR).\n"
    "\t@Outputs: out_tensor (float [1,3,48,192] one per crop),\n"
    "\t  out_crop (uint8 BGR passthrough with frame_id/box_id/box_count/polygon).\n";

class PaddleOcrClsPreFlowUnit : public modelbox::FlowUnit {
 public:
  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> ctx) override;
};
#endif  // MODELBOX_FLOWUNIT_PADDLE_OCR_CLS_PRE_CPU_H_
