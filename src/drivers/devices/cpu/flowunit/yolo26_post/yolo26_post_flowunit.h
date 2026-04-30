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

#ifndef MODELBOX_FLOWUNIT_YOLO26_POST_CPU_H_
#define MODELBOX_FLOWUNIT_YOLO26_POST_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <opencv2/opencv.hpp>

#include "modelbox/buffer.h"

constexpr const char *FLOWUNIT_NAME = "yolo26_post";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Anchor-free YOLO postprocess for Ultralytics outputs.\n"
    "\t@Inputs: in_image (BGR uint8) + in_feat (float, layout [B, 4+nc, N]).\n"
    "\t@Outputs: out_image (BGR uint8) with red bounding boxes drawn around\n"
    "\t  detected objects after sigmoid+argmax class selection and NMS.\n"
    "\t@Constraint: Designed for the YOLO26n / YOLO11n / YOLOv8n head; the\n"
    "\t  feature tensor format must be Ultralytics anchor-free (no objectness,\n"
    "\t  scores already sigmoid-applied by the export head).";

class Yolo26PostFlowUnit : public modelbox::FlowUnit {
 public:
  Yolo26PostFlowUnit() = default;
  ~Yolo26PostFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  struct Detection {
    float x1, y1, x2, y2;
    float score;
    int label;
  };

  std::vector<Detection> Decode(const float *feat, int channels, int num_anchors,
                                float scale_x, float scale_y) const;
  std::vector<Detection> Nms(std::vector<Detection> dets) const;
  void DrawBoxes(cv::Mat &img, const std::vector<Detection> &dets) const;

  int net_h_{640};
  int net_w_{640};
  int num_classes_{80};
  float conf_threshold_{0.25F};
  float iou_threshold_{0.45F};
};

#endif  // MODELBOX_FLOWUNIT_YOLO26_POST_CPU_H_
