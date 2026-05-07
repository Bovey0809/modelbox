/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
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

#ifndef MODELBOX_FLOWUNIT_PERSON_DET_CROP_CPU_H_
#define MODELBOX_FLOWUNIT_PERSON_DET_CROP_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <opencv2/opencv.hpp>

#include "modelbox/buffer.h"

constexpr const char *FLOWUNIT_NAME = "person_det_crop";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Filter person detections and crop the top-scoring person.\n"
    "\t@Inputs: in_image (BGR uint8 full frame) + in_det_feat (float, post-NMS\n"
    "\t  [K, 6] rows of [x1, y1, x2, y2, score, class_id] in net coordinates).\n"
    "\t@Outputs: out_crop (BGR uint8) — highest-score person crop, clamped to\n"
    "\t  image bounds.  When no person passes the threshold the full original\n"
    "\t  frame is forwarded unchanged so downstream nodes stay in sync.\n"
    "\t@Params: conf_threshold (float, default 0.40), person_class_id (int,\n"
    "\t  default 0 = COCO person), net_w / net_h (int, default 640).\n";

class PersonDetCropFlowUnit : public modelbox::FlowUnit {
 public:
  PersonDetCropFlowUnit() = default;
  ~PersonDetCropFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  struct Detection {
    float x1, y1, x2, y2, score;
    int label;
  };

  int net_h_{640};
  int net_w_{640};
  int person_class_id_{0};
  float conf_threshold_{0.40F};
};

#endif  // MODELBOX_FLOWUNIT_PERSON_DET_CROP_CPU_H_
