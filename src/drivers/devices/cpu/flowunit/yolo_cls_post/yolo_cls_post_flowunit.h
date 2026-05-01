/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef MODELBOX_FLOWUNIT_YOLO_CLS_POST_CPU_H_
#define MODELBOX_FLOWUNIT_YOLO_CLS_POST_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <opencv2/opencv.hpp>

#include "modelbox/buffer.h"

constexpr const char *FLOWUNIT_NAME = "yolo_cls_post";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: ImageNet-style classification post-processor.\n"
    "\t@Inputs: in_image (BGR uint8) + in_feat (float [B, num_classes]).\n"
    "\t@Outputs: out_data (BGR uint8) with the top-K predictions overlaid\n"
    "\t  as text in the upper-left corner. Scores are softmaxed over the\n"
    "\t  raw logits coming out of the model.\n";

class YoloClsPostFlowUnit : public modelbox::FlowUnit {
 public:
  YoloClsPostFlowUnit() = default;
  ~YoloClsPostFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  int top_k_{5};
  int num_classes_{1000};
};

#endif  // MODELBOX_FLOWUNIT_YOLO_CLS_POST_CPU_H_
