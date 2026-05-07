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

#include "person_det_crop_flowunit.h"

#include <algorithm>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

modelbox::Status PersonDetCropFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  net_h_ = opts->GetInt32("net_h", 640);
  net_w_ = opts->GetInt32("net_w", 640);
  person_class_id_ = opts->GetInt32("person_class_id", 0);
  conf_threshold_ = opts->GetFloat("conf_threshold", 0.40F);
  MBLOG_INFO << "person_det_crop: net=" << net_w_ << "x" << net_h_
             << " class=" << person_class_id_ << " conf=" << conf_threshold_;
  return modelbox::STATUS_OK;
}

modelbox::Status PersonDetCropFlowUnit::Close() {
  return modelbox::STATUS_OK;
}

modelbox::Status PersonDetCropFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto in_image = data_ctx->Input("in_image");
  auto in_det_feat = data_ctx->Input("in_det_feat");
  auto out_crop = data_ctx->Output("out_crop");

  if (!in_image || !in_det_feat || !out_crop) {
    return {modelbox::STATUS_FAULT, "person_det_crop: missing port"};
  }
  if (in_image->Size() != in_det_feat->Size()) {
    return {modelbox::STATUS_FAULT, "person_det_crop: in_image/in_det_feat size mismatch"};
  }

  for (size_t i = 0; i < in_image->Size(); ++i) {
    auto img_buf = in_image->At(i);
    auto feat_buf = in_det_feat->At(i);

    int32_t width = 0;
    int32_t height = 0;
    int32_t channel = 3;
    std::string pix_fmt = "bgr";
    img_buf->Get("width", width);
    img_buf->Get("height", height);
    img_buf->Get("channel", channel);
    img_buf->Get("pix_fmt", pix_fmt);

    if (width <= 0 || height <= 0) {
      return {modelbox::STATUS_FAULT, "person_det_crop: bad image meta"};
    }

    // Parse post-NMS detection tensor: rows of [x1, y1, x2, y2, score, class]
    // in net_w_ x net_h_ coordinate space.
    const auto *feat = static_cast<const float *>(feat_buf->ConstData());
    const size_t feat_bytes = feat_buf->GetBytes();
    const int num_dets = static_cast<int>(feat_bytes / (sizeof(float) * 6));

    float scale_x = static_cast<float>(width) / static_cast<float>(net_w_);
    float scale_y = static_cast<float>(height) / static_cast<float>(net_h_);

    Detection best;
    best.score = -1.0F;
    for (int d = 0; d < num_dets; ++d) {
      const float *row = feat + d * 6;
      float score = row[4];
      int class_id = static_cast<int>(row[5]);
      if (score < conf_threshold_ || class_id != person_class_id_) {
        continue;
      }
      if (score > best.score) {
        best.x1 = row[0];
        best.y1 = row[1];
        best.x2 = row[2];
        best.y2 = row[3];
        best.score = score;
        best.label = class_id;
      }
    }

    auto *raw = const_cast<void *>(img_buf->ConstData());
    cv::Mat frame(height, width, CV_8UC3, raw);

    cv::Mat region;
    if (best.score >= conf_threshold_) {
      int x1 = std::max(0, std::min(width - 1,
                                    static_cast<int>(best.x1 * scale_x)));
      int y1 = std::max(0, std::min(height - 1,
                                    static_cast<int>(best.y1 * scale_y)));
      int x2 = std::max(0, std::min(width - 1,
                                    static_cast<int>(best.x2 * scale_x)));
      int y2 = std::max(0, std::min(height - 1,
                                    static_cast<int>(best.y2 * scale_y)));
      int rw = std::max(1, x2 - x1);
      int rh = std::max(1, y2 - y1);
      frame(cv::Rect(x1, y1, rw, rh)).copyTo(region);
      MBLOG_DEBUG << "person_det_crop: person score=" << best.score
                  << " crop=[" << x1 << "," << y1 << "," << rw << "x" << rh << "]";
    } else {
      // No person above threshold — forward the full frame so downstream
      // nodes (resize → pose_det → pose_post → encoder) stay in sync.
      frame.copyTo(region);
      MBLOG_DEBUG << "person_det_crop: no person detected, forwarding full frame";
    }

    auto region_mat = std::make_shared<cv::Mat>(region);
    const size_t out_bytes = region_mat->total() * region_mat->elemSize();
    out_crop->EmplaceBack(region_mat->data, out_bytes,
                          [region_mat](void * /*ptr*/) {});
    auto out_buf = out_crop->Back();
    const int32_t out_w = region_mat->cols;
    const int32_t out_h = region_mat->rows;
    out_buf->Set("width", out_w);
    out_buf->Set("height", out_h);
    out_buf->Set("channel", 3);
    out_buf->Set("pix_fmt", pix_fmt);
    out_buf->Set("width_stride", static_cast<int32_t>(out_w * 3));
    out_buf->Set("height_stride", out_h);
    out_buf->Set("shape", std::vector<size_t>{static_cast<size_t>(out_h),
                                              static_cast<size_t>(out_w), 3});
    out_buf->Set("layout", std::string("hwc"));
    out_buf->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(PersonDetCropFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_image"});
  desc.AddFlowUnitInput({"in_det_feat"});
  desc.AddFlowUnitOutput({"out_crop"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
