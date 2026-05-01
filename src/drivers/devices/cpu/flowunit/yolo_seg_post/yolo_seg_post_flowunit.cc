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

#include "yolo_seg_post_flowunit.h"

#include <securec.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

namespace {
inline float Sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

// Deterministic palette so the same class index always renders in the same
// color across frames. 21 distinct hues works well for COCO's 80 classes
// (re-used cyclically) and keeps adjacent labels visually separable.
const cv::Scalar &PaletteColor(int idx) {
  static const cv::Scalar palette[] = {
      {0, 0, 255},   {0, 85, 255},  {0, 170, 255}, {0, 255, 255},
      {0, 255, 170}, {0, 255, 85},  {0, 255, 0},   {85, 255, 0},
      {170, 255, 0}, {255, 255, 0}, {255, 170, 0}, {255, 85, 0},
      {255, 0, 0},   {255, 0, 85},  {255, 0, 170}, {255, 0, 255},
      {170, 0, 255}, {85, 0, 255},  {255, 85, 85}, {170, 170, 85},
      {85, 170, 170}};
  return palette[idx % (int)(sizeof(palette) / sizeof(palette[0]))];
}
}  // namespace

modelbox::Status YoloSegPostFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  net_h_ = opts->GetInt32("net_h", 640);
  net_w_ = opts->GetInt32("net_w", 640);
  num_classes_ = opts->GetInt32("num_classes", 80);
  num_masks_ = opts->GetInt32("num_masks", 32);
  conf_threshold_ = opts->GetFloat("conf_threshold", 0.25F);
  iou_threshold_ = opts->GetFloat("iou_threshold", 0.45F);
  mask_threshold_ = opts->GetFloat("mask_threshold", 0.5F);
  overlay_alpha_ = opts->GetFloat("overlay_alpha", 0.5F);
  MBLOG_INFO << "yolo_seg_post: net=" << net_w_ << "x" << net_h_ << " nc="
             << num_classes_ << " nm=" << num_masks_
             << " conf=" << conf_threshold_ << " iou=" << iou_threshold_
             << " mask=" << mask_threshold_ << " alpha=" << overlay_alpha_;
  return modelbox::STATUS_OK;
}

modelbox::Status YoloSegPostFlowUnit::Close() { return modelbox::STATUS_OK; }

std::vector<YoloSegPostFlowUnit::Detection> YoloSegPostFlowUnit::Decode(
    const float *feat, int channels, int num_anchors, float scale_x,
    float scale_y) const {
  std::vector<Detection> out;
  out.reserve(64);
  const int nc = num_classes_;
  for (int n = 0; n < num_anchors; ++n) {
    int best_label = 0;
    float best_score = feat[4 * num_anchors + n];
    for (int c = 1; c < nc; ++c) {
      float s = feat[(4 + c) * num_anchors + n];
      if (s > best_score) {
        best_score = s;
        best_label = c;
      }
    }
    if (best_score < conf_threshold_) {
      continue;
    }
    float cx = feat[0 * num_anchors + n];
    float cy = feat[1 * num_anchors + n];
    float w = feat[2 * num_anchors + n];
    float h = feat[3 * num_anchors + n];
    Detection d;
    d.x1 = (cx - w / 2.0F) * scale_x;
    d.y1 = (cy - h / 2.0F) * scale_y;
    d.x2 = (cx + w / 2.0F) * scale_x;
    d.y2 = (cy + h / 2.0F) * scale_y;
    d.score = best_score;
    d.label = best_label;
    d.anchor_idx = n;
    out.push_back(d);
  }
  return out;
}

std::vector<YoloSegPostFlowUnit::Detection> YoloSegPostFlowUnit::Nms(
    std::vector<Detection> dets) const {
  std::sort(dets.begin(), dets.end(), [](const Detection &a, const Detection &b) {
    return a.score > b.score;
  });
  std::vector<Detection> kept;
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

void YoloSegPostFlowUnit::DrawMaskAndBoxes(
    cv::Mat &img, const std::vector<Detection> &dets, const float *feat,
    int feat_channels, int num_anchors, const float *proto, int nm, int ph,
    int pw) const {
  const int img_w = img.cols;
  const int img_h = img.rows;

  // Collect per-detection mask coefficients into a row-matrix
  // [num_dets, nm], multiply by proto reshaped as [nm, ph*pw], yields
  // [num_dets, ph*pw] = the raw masks in proto space.
  if (dets.empty()) return;
  cv::Mat coefs(static_cast<int>(dets.size()), nm, CV_32F);
  for (size_t i = 0; i < dets.size(); ++i) {
    int anchor = dets[i].anchor_idx;
    for (int c = 0; c < nm; ++c) {
      coefs.at<float>(static_cast<int>(i), c) =
          feat[(4 + num_classes_ + c) * num_anchors + anchor];
    }
  }
  cv::Mat proto_mat(nm, ph * pw, CV_32F, const_cast<float *>(proto));
  cv::Mat masks_proto = coefs * proto_mat;  // [num_dets, ph*pw]

  for (size_t i = 0; i < dets.size(); ++i) {
    const auto &d = dets[i];
    cv::Mat mask_proto(ph, pw, CV_32F);
    std::memcpy(mask_proto.data, masks_proto.ptr<float>(static_cast<int>(i)),
                ph * pw * sizeof(float));

    // sigmoid in-place
    cv::Mat mask;
    cv::exp(-mask_proto, mask);
    mask = 1.0F / (1.0F + mask);

    // Resize the mask from proto space (160x160) to image space, then crop
    // to the detection bbox so we don't leak outside it.
    cv::Mat mask_img;
    cv::resize(mask, mask_img, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);

    int x1 = std::max(0, std::min(img_w - 1, static_cast<int>(d.x1)));
    int y1 = std::max(0, std::min(img_h - 1, static_cast<int>(d.y1)));
    int x2 = std::max(0, std::min(img_w - 1, static_cast<int>(d.x2)));
    int y2 = std::max(0, std::min(img_h - 1, static_cast<int>(d.y2)));
    if (x2 <= x1 || y2 <= y1) continue;
    cv::Rect box(x1, y1, x2 - x1, y2 - y1);

    cv::Mat mask_crop = mask_img(box);
    cv::Mat mask_bin = mask_crop > mask_threshold_;

    cv::Mat roi = img(box);
    const cv::Scalar &color = PaletteColor(d.label);
    cv::Mat color_layer(roi.size(), roi.type(), color);
    cv::Mat blended;
    cv::addWeighted(color_layer, overlay_alpha_, roi, 1.0F - overlay_alpha_,
                    0.0, blended);
    blended.copyTo(roi, mask_bin);

    cv::rectangle(img, box, color, 2);
    char label[32];
    snprintf(label, sizeof(label), "%d:%.2f", d.label, d.score);
    cv::putText(img, label, cv::Point(x1, std::max(0, y1 - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
  }
}

modelbox::Status YoloSegPostFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto in_image = data_ctx->Input("in_image");
  auto in_feat = data_ctx->Input("in_feat");
  auto in_proto = data_ctx->Input("in_proto");
  auto out_image = data_ctx->Output("out_data");
  if (!in_image || !in_feat || !in_proto || !out_image) {
    return {modelbox::STATUS_FAULT, "yolo_seg_post: missing port"};
  }
  if (in_image->Size() != in_feat->Size() ||
      in_image->Size() != in_proto->Size()) {
    return {modelbox::STATUS_FAULT, "yolo_seg_post: input size mismatch"};
  }
  if (in_image->Size() == 0) return modelbox::STATUS_OK;

  std::vector<size_t> shapes;
  for (size_t i = 0; i < in_image->Size(); ++i) {
    shapes.push_back(in_image->At(i)->GetBytes());
  }
  auto build_status = out_image->Build(shapes);
  if (!build_status) return build_status;

  for (size_t i = 0; i < in_image->Size(); ++i) {
    auto img_buf = in_image->At(i);
    auto feat_buf = in_feat->At(i);
    auto proto_buf = in_proto->At(i);

    int32_t width = 0, height = 0, channel = 3;
    img_buf->Get("width", width);
    img_buf->Get("height", height);
    img_buf->Get("channel", channel);
    if (width <= 0 || height <= 0 || channel != 3) {
      return {modelbox::STATUS_FAULT, "yolo_seg_post: bad image meta"};
    }

    const int feat_channels = 4 + num_classes_ + num_masks_;
    const auto feat_bytes = feat_buf->GetBytes();
    if (feat_bytes % (sizeof(float) * feat_channels) != 0) {
      return {modelbox::STATUS_FAULT, "yolo_seg_post: bad feat size"};
    }
    const int num_anchors =
        static_cast<int>(feat_bytes / (sizeof(float) * feat_channels));

    // Proto shape comes from upstream meta (set by openvino_inference). Default
    // to 160x160 (yolo11n-seg @ 640) if shape meta is absent.
    std::vector<size_t> proto_shape;
    proto_buf->Get("shape", proto_shape);
    int nm = num_masks_, ph = 160, pw = 160;
    if (proto_shape.size() == 4) {
      nm = static_cast<int>(proto_shape[1]);
      ph = static_cast<int>(proto_shape[2]);
      pw = static_cast<int>(proto_shape[3]);
    }
    const auto *feat = static_cast<const float *>(feat_buf->ConstData());
    const auto *proto = static_cast<const float *>(proto_buf->ConstData());

    const float scale_x = static_cast<float>(width) / static_cast<float>(net_w_);
    const float scale_y = static_cast<float>(height) / static_cast<float>(net_h_);

    auto raw = Decode(feat, feat_channels, num_anchors, scale_x, scale_y);
    auto kept = Nms(std::move(raw));

    auto out_buf = out_image->At(i);
    auto *dst = static_cast<uint8_t *>(out_buf->MutableData());
    const auto *src = static_cast<const uint8_t *>(img_buf->ConstData());
    if (memcpy_s(dst, img_buf->GetBytes(), src, img_buf->GetBytes()) != EOK) {
      return {modelbox::STATUS_FAULT, "yolo_seg_post: image copy failed"};
    }
    cv::Mat out_mat(height, width, CV_8UC3, dst);
    DrawMaskAndBoxes(out_mat, kept, feat, feat_channels, num_anchors, proto,
                     nm, ph, pw);

    out_buf->CopyMeta(img_buf);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(YoloSegPostFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_image"});
  desc.AddFlowUnitInput({"in_feat"});
  desc.AddFlowUnitInput({"in_proto"});
  desc.AddFlowUnitOutput({"out_data"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "net_h", "int", true, "640", "network input height"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "net_w", "int", true, "640", "network input width"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "num_classes", "int", true, "80", "number of class channels"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "num_masks", "int", true, "32", "mask coefficient count"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "conf_threshold", "float", true, "0.25", "minimum class score"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "iou_threshold", "float", true, "0.45", "NMS IoU threshold"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "mask_threshold", "float", true, "0.5", "mask binarisation threshold"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "overlay_alpha", "float", true, "0.5",
      "alpha for the colored mask overlay"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
