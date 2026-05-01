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

#ifndef MODELBOX_FLOWUNIT_YOLO_SEG_POST_CPU_H_
#define MODELBOX_FLOWUNIT_YOLO_SEG_POST_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <opencv2/opencv.hpp>
#include <vector>

#include "modelbox/buffer.h"

constexpr const char *FLOWUNIT_NAME = "yolo_seg_post";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Anchor-free YOLO11/26 segmentation postprocess.\n"
    "\t@Inputs: in_image (BGR uint8), in_feat (float [B, 4+nc+nm, N]),\n"
    "\t  in_proto (float [B, nm, ph, pw]).\n"
    "\t@Outputs: out_data (BGR uint8) with per-instance colored mask overlay\n"
    "\t  + red bounding boxes drawn after sigmoid + argmax + NMS.\n";

class YoloSegPostFlowUnit : public modelbox::FlowUnit {
 public:
  YoloSegPostFlowUnit() = default;
  ~YoloSegPostFlowUnit() override = default;

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
    int anchor_idx;  // index into output0's anchor dimension
  };

  std::vector<Detection> Decode(const float *feat, int channels, int num_anchors,
                                float scale_x, float scale_y) const;
  std::vector<Detection> Nms(std::vector<Detection> dets) const;
  void DrawMaskAndBoxes(cv::Mat &img, const std::vector<Detection> &dets,
                        const float *feat, int feat_channels, int num_anchors,
                        const float *proto, int nm, int ph, int pw) const;

  int net_h_{640};
  int net_w_{640};
  int num_classes_{80};
  int num_masks_{32};
  float conf_threshold_{0.25F};
  float iou_threshold_{0.45F};
  float mask_threshold_{0.5F};
  float overlay_alpha_{0.5F};
};

#endif  // MODELBOX_FLOWUNIT_YOLO_SEG_POST_CPU_H_
