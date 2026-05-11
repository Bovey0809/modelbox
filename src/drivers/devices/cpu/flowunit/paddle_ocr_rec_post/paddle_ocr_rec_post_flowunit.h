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

#ifndef MODELBOX_FLOWUNIT_PADDLE_OCR_REC_POST_CPU_H_
#define MODELBOX_FLOWUNIT_PADDLE_OCR_REC_POST_CPU_H_

#include <modelbox/flowunit.h>

#include <string>
#include <vector>

constexpr const char *FLOWUNIT_NAME = "paddle_ocr_rec_post";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: PP-OCR recognition postprocess. CTC greedy decode.\n"
    "\t@Inputs: in_logits (float [N,T,C]), in_crop (uint8 BGR passthrough).\n"
    "\t@Outputs: out_result (uint8 placeholder) with text/score/frame_id/"
    "box_id/box_count/polygon meta.\n";

class PaddleOcrRecPostFlowUnit : public modelbox::FlowUnit {
 public:
  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(std::shared_ptr<modelbox::DataContext> ctx) override;

 private:
  std::vector<std::string> charset_;
  float score_thresh_{0.5F};
};

#endif  // MODELBOX_FLOWUNIT_PADDLE_OCR_REC_POST_CPU_H_
