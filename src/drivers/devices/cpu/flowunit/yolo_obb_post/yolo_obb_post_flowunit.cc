/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "yolo_obb_post_flowunit.h"

#include <securec.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

namespace {
const cv::Scalar &PaletteColor(int idx) {
  static const cv::Scalar palette[] = {
      {0, 0, 255},   {0, 85, 255},  {0, 170, 255}, {0, 255, 255},
      {0, 255, 170}, {0, 255, 85},  {0, 255, 0},   {85, 255, 0},
      {170, 255, 0}, {255, 255, 0}, {255, 170, 0}, {255, 85, 0},
      {255, 0, 0},   {255, 0, 85},  {255, 0, 170}};
  return palette[idx % (int)(sizeof(palette) / sizeof(palette[0]))];
}
}  // namespace

modelbox::Status YoloObbPostFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  net_h_ = opts->GetInt32("net_h", 640);
  net_w_ = opts->GetInt32("net_w", 640);
  num_classes_ = opts->GetInt32("num_classes", 15);
  conf_threshold_ = opts->GetFloat("conf_threshold", 0.25F);
  iou_threshold_ = opts->GetFloat("iou_threshold", 0.45F);
  MBLOG_INFO << "yolo_obb_post: net=" << net_w_ << "x" << net_h_
             << " nc=" << num_classes_ << " conf=" << conf_threshold_
             << " iou=" << iou_threshold_;
  return modelbox::STATUS_OK;
}

modelbox::Status YoloObbPostFlowUnit::Close() { return modelbox::STATUS_OK; }

std::vector<YoloObbPostFlowUnit::OBox> YoloObbPostFlowUnit::Decode(
    const float *feat, int channels, int num_anchors, float scale_x,
    float scale_y) const {
  std::vector<OBox> out;
  out.reserve(64);
  const int nc = num_classes_;
  const int angle_idx = 4 + nc;  // last channel
  for (int n = 0; n < num_anchors; ++n) {
    int best = 0;
    float best_score = feat[4 * num_anchors + n];
    for (int c = 1; c < nc; ++c) {
      float s = feat[(4 + c) * num_anchors + n];
      if (s > best_score) {
        best_score = s;
        best = c;
      }
    }
    if (best_score < conf_threshold_) continue;
    OBox b;
    b.cx = feat[0 * num_anchors + n] * scale_x;
    b.cy = feat[1 * num_anchors + n] * scale_y;
    b.w = feat[2 * num_anchors + n] * scale_x;
    b.h = feat[3 * num_anchors + n] * scale_y;
    b.angle = feat[angle_idx * num_anchors + n];
    b.score = best_score;
    b.label = best;
    out.push_back(b);
  }
  return out;
}

std::vector<YoloObbPostFlowUnit::OBox> YoloObbPostFlowUnit::Nms(
    std::vector<OBox> dets) const {
  // Use axis-aligned bounding boxes of the rotated rectangles for NMS.
  // True rotated-IoU is more correct but more expensive; AABB-IoU is a
  // perfectly fine first approximation for typical aerial imagery where
  // boxes do not overlap heavily after rotation.
  std::sort(dets.begin(), dets.end(),
            [](const OBox &a, const OBox &b) { return a.score > b.score; });
  auto aabb = [](const OBox &b) {
    cv::RotatedRect rr(cv::Point2f(b.cx, b.cy), cv::Size2f(b.w, b.h),
                       b.angle * 180.0F / static_cast<float>(CV_PI));
    return rr.boundingRect2f();
  };
  std::vector<OBox> kept;
  std::vector<bool> sup(dets.size(), false);
  for (size_t i = 0; i < dets.size(); ++i) {
    if (sup[i]) continue;
    kept.push_back(dets[i]);
    auto ai = aabb(dets[i]);
    float a_area = ai.width * ai.height;
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (sup[j]) continue;
      auto bj = aabb(dets[j]);
      cv::Rect2f inter_rect = ai & bj;
      float inter = inter_rect.width * inter_rect.height;
      float b_area = bj.width * bj.height;
      float uni = a_area + b_area - inter;
      if (uni > 0 && inter / uni > iou_threshold_) sup[j] = true;
    }
  }
  return kept;
}

void YoloObbPostFlowUnit::DrawObbs(cv::Mat &img,
                                   const std::vector<OBox> &dets) const {
  for (const auto &b : dets) {
    cv::RotatedRect rr(cv::Point2f(b.cx, b.cy), cv::Size2f(b.w, b.h),
                       b.angle * 180.0F / static_cast<float>(CV_PI));
    cv::Point2f pts[4];
    rr.points(pts);
    const cv::Scalar &color = PaletteColor(b.label);
    for (int e = 0; e < 4; ++e) {
      cv::line(img, pts[e], pts[(e + 1) % 4], color, 2);
    }
    char label[32];
    snprintf(label, sizeof(label), "%d:%.2f", b.label, b.score);
    cv::putText(img, label, pts[1], cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
  }
}

modelbox::Status YoloObbPostFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto in_image = data_ctx->Input("in_image");
  auto in_feat = data_ctx->Input("in_feat");
  auto out_image = data_ctx->Output("out_data");
  if (!in_image || !in_feat || !out_image) {
    return {modelbox::STATUS_FAULT, "yolo_obb_post: missing port"};
  }
  if (in_image->Size() != in_feat->Size()) {
    return {modelbox::STATUS_FAULT, "yolo_obb_post: size mismatch"};
  }
  if (in_image->Size() == 0) return modelbox::STATUS_OK;

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
      return {modelbox::STATUS_FAULT, "yolo_obb_post: bad image meta"};
    }

    const int channels = 4 + num_classes_ + 1;
    const auto feat_bytes = feat_buf->GetBytes();
    if (feat_bytes % (sizeof(float) * channels) != 0) {
      return {modelbox::STATUS_FAULT, "yolo_obb_post: bad feat size"};
    }
    const int num_anchors =
        static_cast<int>(feat_bytes / (sizeof(float) * channels));
    const auto *feat = static_cast<const float *>(feat_buf->ConstData());
    const float scale_x = static_cast<float>(width) / static_cast<float>(net_w_);
    const float scale_y = static_cast<float>(height) / static_cast<float>(net_h_);

    auto raw = Decode(feat, channels, num_anchors, scale_x, scale_y);
    auto kept = Nms(std::move(raw));

    auto out_buf = out_image->At(i);
    auto *dst = static_cast<uint8_t *>(out_buf->MutableData());
    const auto *src = static_cast<const uint8_t *>(img_buf->ConstData());
    if (memcpy_s(dst, img_buf->GetBytes(), src, img_buf->GetBytes()) != EOK) {
      return {modelbox::STATUS_FAULT, "yolo_obb_post: image copy failed"};
    }
    cv::Mat out_mat(height, width, CV_8UC3, dst);
    DrawObbs(out_mat, kept);
    out_buf->CopyMeta(img_buf);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(YoloObbPostFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_image"});
  desc.AddFlowUnitInput({"in_feat"});
  desc.AddFlowUnitOutput({"out_data"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "net_h", "int", true, "640", "network input height"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "net_w", "int", true, "640", "network input width"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "num_classes", "int", true, "15", "DOTA-class count"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "conf_threshold", "float", true, "0.25", "score threshold"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "iou_threshold", "float", true, "0.45", "NMS IoU threshold"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
