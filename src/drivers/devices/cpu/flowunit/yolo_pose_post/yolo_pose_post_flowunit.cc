/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "yolo_pose_post_flowunit.h"

#include <securec.h>

#include <algorithm>
#include <cstring>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

namespace {
// COCO 17-keypoint skeleton edges, 0-indexed.
//   0 nose, 1 L-eye, 2 R-eye, 3 L-ear, 4 R-ear,
//   5 L-shldr, 6 R-shldr, 7 L-elbow, 8 R-elbow, 9 L-wrist, 10 R-wrist,
//   11 L-hip, 12 R-hip, 13 L-knee, 14 R-knee, 15 L-ankle, 16 R-ankle.
const int kSkeleton[][2] = {
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
    {5, 11},  {6, 12},  {5, 6},   {5, 7},   {6, 8},
    {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},
    {1, 3},   {2, 4},   {3, 5},   {4, 6}};
constexpr int kNumEdges = sizeof(kSkeleton) / sizeof(kSkeleton[0]);

// Limb color (BGR). Matches Ultralytics' default palette ordering.
const cv::Scalar kLimbColor(255, 128, 0);
const cv::Scalar kKptColor(0, 255, 255);
const cv::Scalar kBoxColor(0, 0, 255);
}  // namespace

modelbox::Status YoloPosePostFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  net_h_ = opts->GetInt32("net_h", 640);
  net_w_ = opts->GetInt32("net_w", 640);
  num_kpts_ = opts->GetInt32("num_kpts", 17);
  conf_threshold_ = opts->GetFloat("conf_threshold", 0.25F);
  iou_threshold_ = opts->GetFloat("iou_threshold", 0.45F);
  kpt_threshold_ = opts->GetFloat("kpt_threshold", 0.5F);
  MBLOG_INFO << "yolo_pose_post: net=" << net_w_ << "x" << net_h_
             << " kpts=" << num_kpts_ << " conf=" << conf_threshold_
             << " iou=" << iou_threshold_ << " kpt_thr=" << kpt_threshold_;
  return modelbox::STATUS_OK;
}

modelbox::Status YoloPosePostFlowUnit::Close() { return modelbox::STATUS_OK; }

std::vector<YoloPosePostFlowUnit::Person> YoloPosePostFlowUnit::Decode(
    const float *feat, int num_anchors, float scale_x, float scale_y) const {
  std::vector<Person> out;
  out.reserve(8);
  const int channels = 5 + num_kpts_ * 3;
  for (int n = 0; n < num_anchors; ++n) {
    const float score = feat[4 * num_anchors + n];
    if (score < conf_threshold_) continue;
    Person p;
    const float cx = feat[0 * num_anchors + n];
    const float cy = feat[1 * num_anchors + n];
    const float w = feat[2 * num_anchors + n];
    const float h = feat[3 * num_anchors + n];
    p.x1 = (cx - w / 2.0F) * scale_x;
    p.y1 = (cy - h / 2.0F) * scale_y;
    p.x2 = (cx + w / 2.0F) * scale_x;
    p.y2 = (cy + h / 2.0F) * scale_y;
    p.score = score;
    for (int k = 0; k < num_kpts_; ++k) {
      const int xc = 5 + k * 3;
      p.kpts[k * 3 + 0] = feat[xc * num_anchors + n] * scale_x;
      p.kpts[k * 3 + 1] = feat[(xc + 1) * num_anchors + n] * scale_y;
      p.kpts[k * 3 + 2] = feat[(xc + 2) * num_anchors + n];
    }
    out.push_back(p);
  }
  (void)channels;
  return out;
}

std::vector<YoloPosePostFlowUnit::Person> YoloPosePostFlowUnit::Nms(
    std::vector<Person> dets) const {
  std::sort(dets.begin(), dets.end(),
            [](const Person &a, const Person &b) { return a.score > b.score; });
  std::vector<Person> kept;
  std::vector<bool> suppressed(dets.size(), false);
  for (size_t i = 0; i < dets.size(); ++i) {
    if (suppressed[i]) continue;
    kept.push_back(dets[i]);
    const auto &a = dets[i];
    float a_area = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (suppressed[j]) continue;
      const auto &b = dets[j];
      float ix1 = std::max(a.x1, b.x1);
      float iy1 = std::max(a.y1, b.y1);
      float ix2 = std::min(a.x2, b.x2);
      float iy2 = std::min(a.y2, b.y2);
      float iw = std::max(0.0F, ix2 - ix1);
      float ih = std::max(0.0F, iy2 - iy1);
      float inter = iw * ih;
      float b_area = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
      float uni = a_area + b_area - inter;
      if (uni > 0 && inter / uni > iou_threshold_) {
        suppressed[j] = true;
      }
    }
  }
  return kept;
}

void YoloPosePostFlowUnit::DrawSkeleton(
    cv::Mat &img, const std::vector<Person> &people) const {
  for (const auto &p : people) {
    int x1 = std::max(0, std::min(img.cols - 1, static_cast<int>(p.x1)));
    int y1 = std::max(0, std::min(img.rows - 1, static_cast<int>(p.y1)));
    int x2 = std::max(0, std::min(img.cols - 1, static_cast<int>(p.x2)));
    int y2 = std::max(0, std::min(img.rows - 1, static_cast<int>(p.y2)));
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), kBoxColor, 2);
    char label[16];
    snprintf(label, sizeof(label), "%.2f", p.score);
    cv::putText(img, label, cv::Point(x1, std::max(0, y1 - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, kBoxColor, 1);

    // Limbs first, then keypoints on top so circles are visible.
    for (int e = 0; e < kNumEdges; ++e) {
      int i = kSkeleton[e][0];
      int j = kSkeleton[e][1];
      if (i >= num_kpts_ || j >= num_kpts_) continue;
      float ci = p.kpts[i * 3 + 2];
      float cj = p.kpts[j * 3 + 2];
      if (ci < kpt_threshold_ || cj < kpt_threshold_) continue;
      cv::Point pi(static_cast<int>(p.kpts[i * 3 + 0]),
                   static_cast<int>(p.kpts[i * 3 + 1]));
      cv::Point pj(static_cast<int>(p.kpts[j * 3 + 0]),
                   static_cast<int>(p.kpts[j * 3 + 1]));
      cv::line(img, pi, pj, kLimbColor, 2);
    }
    for (int k = 0; k < num_kpts_; ++k) {
      float c = p.kpts[k * 3 + 2];
      if (c < kpt_threshold_) continue;
      cv::Point pk(static_cast<int>(p.kpts[k * 3 + 0]),
                   static_cast<int>(p.kpts[k * 3 + 1]));
      cv::circle(img, pk, 3, kKptColor, -1);
    }
  }
}

modelbox::Status YoloPosePostFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto in_image = data_ctx->Input("in_image");
  auto in_feat = data_ctx->Input("in_feat");
  auto out_image = data_ctx->Output("out_data");
  if (!in_image || !in_feat || !out_image) {
    return {modelbox::STATUS_FAULT, "yolo_pose_post: missing port"};
  }
  if (in_image->Size() != in_feat->Size()) {
    return {modelbox::STATUS_FAULT, "yolo_pose_post: size mismatch"};
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
      return {modelbox::STATUS_FAULT, "yolo_pose_post: bad image meta"};
    }

    const int channels = 5 + num_kpts_ * 3;
    const auto feat_bytes = feat_buf->GetBytes();
    if (feat_bytes % (sizeof(float) * channels) != 0) {
      return {modelbox::STATUS_FAULT, "yolo_pose_post: bad feat size"};
    }
    const int num_anchors =
        static_cast<int>(feat_bytes / (sizeof(float) * channels));

    const auto *feat = static_cast<const float *>(feat_buf->ConstData());
    const float scale_x = static_cast<float>(width) / static_cast<float>(net_w_);
    const float scale_y = static_cast<float>(height) / static_cast<float>(net_h_);

    auto raw = Decode(feat, num_anchors, scale_x, scale_y);
    auto kept = Nms(std::move(raw));

    auto out_buf = out_image->At(i);
    auto *dst = static_cast<uint8_t *>(out_buf->MutableData());
    const auto *src = static_cast<const uint8_t *>(img_buf->ConstData());
    if (memcpy_s(dst, img_buf->GetBytes(), src, img_buf->GetBytes()) != EOK) {
      return {modelbox::STATUS_FAULT, "yolo_pose_post: image copy failed"};
    }
    cv::Mat out_mat(height, width, CV_8UC3, dst);
    DrawSkeleton(out_mat, kept);
    out_buf->CopyMeta(img_buf);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(YoloPosePostFlowUnit, desc) {
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
      "num_kpts", "int", true, "17", "number of keypoints"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "conf_threshold", "float", true, "0.25", "person score threshold"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "iou_threshold", "float", true, "0.45", "NMS IoU threshold"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "kpt_threshold", "float", true, "0.5", "per-keypoint visibility threshold"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
