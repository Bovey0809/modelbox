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

#ifndef MODELBOX_FLOWUNIT_PADDLE_OCR_CLS_POST_CPU_H_
#define MODELBOX_FLOWUNIT_PADDLE_OCR_CLS_POST_CPU_H_

#include <modelbox/flowunit.h>
#include <opencv2/opencv.hpp>

constexpr const char *FLOWUNIT_NAME = "paddle_ocr_cls_post";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: PP-OCR angle classifier postprocess. Rotates crops 180.\n"
    "\t@Inputs: in_logits (float [N,2]), in_crop (uint8 BGR).\n"
    "\t@Outputs: out_crop (uint8 BGR) with angle meta.\n";

class PaddleOcrClsPostFlowUnit : public modelbox::FlowUnit {
 public:
  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> ctx) override;

 private:
  float cls_thresh_{0.9F};
};
#endif  // MODELBOX_FLOWUNIT_PADDLE_OCR_CLS_POST_CPU_H_
