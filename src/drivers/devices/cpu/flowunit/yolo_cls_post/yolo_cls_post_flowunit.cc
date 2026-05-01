/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "yolo_cls_post_flowunit.h"

#include <securec.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

modelbox::Status YoloClsPostFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  top_k_ = opts->GetInt32("top_k", 5);
  num_classes_ = opts->GetInt32("num_classes", 1000);
  MBLOG_INFO << "yolo_cls_post: top_k=" << top_k_
             << " num_classes=" << num_classes_;
  return modelbox::STATUS_OK;
}

modelbox::Status YoloClsPostFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status YoloClsPostFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto in_image = data_ctx->Input("in_image");
  auto in_feat = data_ctx->Input("in_feat");
  auto out_image = data_ctx->Output("out_data");
  if (!in_image || !in_feat || !out_image) {
    return {modelbox::STATUS_FAULT, "yolo_cls_post: missing port"};
  }
  if (in_image->Size() != in_feat->Size() || in_image->Size() == 0) {
    return modelbox::STATUS_OK;
  }

  std::vector<size_t> shapes;
  for (size_t i = 0; i < in_image->Size(); ++i) {
    shapes.push_back(in_image->At(i)->GetBytes());
  }
  auto build = out_image->Build(shapes);
  if (!build) return build;

  for (size_t i = 0; i < in_image->Size(); ++i) {
    auto img_buf = in_image->At(i);
    auto feat_buf = in_feat->At(i);

    int32_t width = 0, height = 0, channel = 3;
    img_buf->Get("width", width);
    img_buf->Get("height", height);
    img_buf->Get("channel", channel);
    if (width <= 0 || height <= 0 || channel != 3) {
      return {modelbox::STATUS_FAULT, "yolo_cls_post: bad image meta"};
    }

    const auto feat_bytes = feat_buf->GetBytes();
    const int n = static_cast<int>(feat_bytes / sizeof(float));
    if (n != num_classes_) {
      MBLOG_WARN << "yolo_cls_post: feat has " << n << " logits, expected "
                 << num_classes_ << " -- using actual size";
    }
    const auto *logits = static_cast<const float *>(feat_buf->ConstData());

    // Softmax (numerically stable) so the displayed scores sum to 1.
    float maxv = logits[0];
    for (int c = 1; c < n; ++c) maxv = std::max(maxv, logits[c]);
    std::vector<float> probs(n);
    float sum = 0.0F;
    for (int c = 0; c < n; ++c) {
      probs[c] = std::exp(logits[c] - maxv);
      sum += probs[c];
    }
    if (sum > 0) {
      for (int c = 0; c < n; ++c) probs[c] /= sum;
    }

    // Partial sort to find top_k indices.
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    const int k = std::min(top_k_, n);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });

    auto out_buf = out_image->At(i);
    auto *dst = static_cast<uint8_t *>(out_buf->MutableData());
    const auto *src = static_cast<const uint8_t *>(img_buf->ConstData());
    if (memcpy_s(dst, img_buf->GetBytes(), src, img_buf->GetBytes()) != EOK) {
      return {modelbox::STATUS_FAULT, "yolo_cls_post: image copy failed"};
    }
    cv::Mat out_mat(height, width, CV_8UC3, dst);

    // Write top-K predictions in the upper-left corner. Black drop-shadow
    // behind white text so it remains legible on bright frames.
    for (int j = 0; j < k; ++j) {
      char line[64];
      snprintf(line, sizeof(line), "%d. cls=%d  %.2f%%", j + 1, idx[j],
               probs[idx[j]] * 100.0F);
      const int y = 24 + j * 22;
      cv::putText(out_mat, line, cv::Point(11, y + 1),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
      cv::putText(out_mat, line, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                  0.6, cv::Scalar(255, 255, 255), 1);
    }

    out_buf->CopyMeta(img_buf);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(YoloClsPostFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_image"});
  desc.AddFlowUnitInput({"in_feat"});
  desc.AddFlowUnitOutput({"out_data"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "top_k", "int", true, "5", "number of top predictions to overlay"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "num_classes", "int", true, "1000", "expected logit length"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
