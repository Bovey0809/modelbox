/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef MODELBOX_FLOWUNIT_YOLO_POSE_POST_CPU_H_
#define MODELBOX_FLOWUNIT_YOLO_POSE_POST_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <opencv2/opencv.hpp>
#include <vector>

#include "modelbox/buffer.h"

constexpr const char *FLOWUNIT_NAME = "yolo_pose_post";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Anchor-free YOLO pose-estimation post-processor.\n"
    "\t@Inputs: in_image (BGR uint8) + in_feat (float [B, 5+17*3, N]).\n"
    "\t@Outputs: out_data (BGR uint8) with bbox + COCO skeleton drawn.\n"
    "\t@Layout: feat[c, n] is column-major channel/anchor; channels 0..3\n"
    "\t  are cxcywh, channel 4 is person score, 5..55 are 17*(x,y,conf).\n";

class YoloPosePostFlowUnit : public modelbox::FlowUnit {
 public:
  YoloPosePostFlowUnit() = default;
  ~YoloPosePostFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  struct Person {
    float x1, y1, x2, y2;
    float score;
    float kpts[17 * 3];  // (x, y, conf) per keypoint, in original image coords
  };

  std::vector<Person> Decode(const float *feat, int num_anchors,
                             float scale_x, float scale_y) const;
  std::vector<Person> Nms(std::vector<Person> dets) const;
  void DrawSkeleton(cv::Mat &img, const std::vector<Person> &people) const;

  int net_h_{640};
  int net_w_{640};
  int num_kpts_{17};
  float conf_threshold_{0.25F};
  float iou_threshold_{0.45F};
  float kpt_threshold_{0.5F};
};

#endif  // MODELBOX_FLOWUNIT_YOLO_POSE_POST_CPU_H_
