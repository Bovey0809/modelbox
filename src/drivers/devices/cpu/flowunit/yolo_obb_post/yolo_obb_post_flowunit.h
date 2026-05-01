/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef MODELBOX_FLOWUNIT_YOLO_OBB_POST_CPU_H_
#define MODELBOX_FLOWUNIT_YOLO_OBB_POST_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <opencv2/opencv.hpp>
#include <vector>

#include "modelbox/buffer.h"

constexpr const char *FLOWUNIT_NAME = "yolo_obb_post";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Anchor-free YOLO oriented-bounding-box post-processor.\n"
    "\t@Inputs: in_image (BGR uint8) + in_feat (float [B, 4+nc+1, N]).\n"
    "\t  Channels: 4 cxcywh + nc class scores + 1 rotation angle (rad).\n"
    "\t@Outputs: out_data (BGR uint8) with rotated rectangles drawn via\n"
    "\t  cv::RotatedRect, palette-colored per class, plus class:score text.\n";

class YoloObbPostFlowUnit : public modelbox::FlowUnit {
 public:
  YoloObbPostFlowUnit() = default;
  ~YoloObbPostFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  struct OBox {
    float cx, cy, w, h, angle;  // angle in radians, original-image units
    float score;
    int label;
  };

  std::vector<OBox> Decode(const float *feat, int channels, int num_anchors,
                           float scale_x, float scale_y) const;
  std::vector<OBox> Nms(std::vector<OBox> dets) const;
  void DrawObbs(cv::Mat &img, const std::vector<OBox> &dets) const;

  int net_h_{640};
  int net_w_{640};
  int num_classes_{15};
  float conf_threshold_{0.25F};
  float iou_threshold_{0.45F};
};

#endif  // MODELBOX_FLOWUNIT_YOLO_OBB_POST_CPU_H_
