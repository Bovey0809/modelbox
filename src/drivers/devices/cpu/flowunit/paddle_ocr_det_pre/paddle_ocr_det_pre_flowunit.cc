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

#include "paddle_ocr_det_pre_flowunit.h"

#include <securec.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

namespace {
constexpr float kMean[3] = {0.485F, 0.456F, 0.406F};
constexpr float kStd[3] = {0.229F, 0.224F, 0.225F};
constexpr float kScale = 1.0F / 255.0F;

struct ResizeSpec {
  int rw;
  int rh;
};

ResizeSpec ComputeResize(int w, int h, int max_side_len) {
  float ratio = 1.0F;
  int max_side = std::max(w, h);
  if (max_side > max_side_len) {
    ratio = static_cast<float>(max_side_len) / static_cast<float>(max_side);
  }
  int rw = static_cast<int>(std::round(static_cast<float>(w) * ratio / 32.0F)) * 32;
  int rh = static_cast<int>(std::round(static_cast<float>(h) * ratio / 32.0F)) * 32;
  rw = std::max(32, rw);
  rh = std::max(32, rh);
  return {rw, rh};
}
}  // namespace

modelbox::Status PaddleOcrDetPreFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  max_side_len_ = opts->GetInt32("max_side_len", 960);
  MBLOG_INFO << "paddle_ocr_det_pre: max_side_len=" << max_side_len_;
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrDetPreFlowUnit::Close() {
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrDetPreFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto in_image = ctx->Input("in_image");
  auto out_tensor = ctx->Output("out_tensor");
  auto out_image = ctx->Output("out_image");
  if (!in_image || !out_tensor || !out_image) {
    return {modelbox::STATUS_FAULT, "paddle_ocr_det_pre: missing port"};
  }
  const size_t n = in_image->Size();
  if (n == 0) return modelbox::STATUS_OK;

  std::vector<int32_t> widths(n);
  std::vector<int32_t> heights(n);
  std::vector<int32_t> channels(n);
  std::vector<ResizeSpec> specs(n);
  std::vector<size_t> tensor_sizes(n);
  std::vector<size_t> image_sizes(n);

  for (size_t i = 0; i < n; ++i) {
    auto buf = in_image->At(i);
    int32_t w = 0, h = 0, c = 3;
    buf->Get("width", w);
    buf->Get("height", h);
    buf->Get("channel", c);
    if (w <= 0 || h <= 0 || c != 3) {
      return {modelbox::STATUS_FAULT, "paddle_ocr_det_pre: bad image meta"};
    }
    widths[i] = w;
    heights[i] = h;
    channels[i] = c;
    specs[i] = ComputeResize(w, h, max_side_len_);
    tensor_sizes[i] = static_cast<size_t>(1) * 3 * specs[i].rh * specs[i].rw *
                      sizeof(float);
    image_sizes[i] = static_cast<size_t>(w) * h * c;
  }

  auto bs = out_tensor->Build(tensor_sizes);
  if (!bs) return bs;
  bs = out_image->Build(image_sizes);
  if (!bs) return bs;

  for (size_t i = 0; i < n; ++i) {
    auto img_buf = in_image->At(i);
    const int w = widths[i];
    const int h = heights[i];
    const int rw = specs[i].rw;
    const int rh = specs[i].rh;
    const auto *src = static_cast<const uint8_t *>(img_buf->ConstData());

    cv::Mat in_mat(h, w, CV_8UC3, const_cast<uint8_t *>(src));
    cv::Mat resized;
    cv::resize(in_mat, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    auto tensor_buf = out_tensor->At(i);
    auto *dst = static_cast<float *>(tensor_buf->MutableData());
    const size_t plane = static_cast<size_t>(rh) * rw;
    // HWC u8 -> CHW fp32 ImageNet-normalized
    for (int y = 0; y < rh; ++y) {
      const auto *row = rgb.ptr<uint8_t>(y);
      for (int x = 0; x < rw; ++x) {
        const size_t off = static_cast<size_t>(y) * rw + x;
        for (int c = 0; c < 3; ++c) {
          float v = static_cast<float>(row[x * 3 + c]) * kScale;
          dst[c * plane + off] = (v - kMean[c]) / kStd[c];
        }
      }
    }
    std::vector<size_t> shape = {1, 3, static_cast<size_t>(rh),
                                 static_cast<size_t>(rw)};
    tensor_buf->Set("shape", shape);
    tensor_buf->Set("type", modelbox::ModelBoxDataType::MODELBOX_FLOAT);

    auto img_out_buf = out_image->At(i);
    auto *img_dst = static_cast<uint8_t *>(img_out_buf->MutableData());
    if (memcpy_s(img_dst, img_out_buf->GetBytes(), src,
                 img_buf->GetBytes()) != EOK) {
      return {modelbox::STATUS_FAULT, "paddle_ocr_det_pre: image copy failed"};
    }
    img_out_buf->CopyMeta(img_buf);
    img_out_buf->Set("width", widths[i]);
    img_out_buf->Set("height", heights[i]);
    img_out_buf->Set("channel", channels[i]);
    img_out_buf->Set("resized_w", static_cast<int32_t>(rw));
    img_out_buf->Set("resized_h", static_cast<int32_t>(rh));
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(PaddleOcrDetPreFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_image"});
  desc.AddFlowUnitOutput({"out_tensor"});
  desc.AddFlowUnitOutput({"out_image"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "max_side_len", "int", true, "960",
      "max side length after resize (rounded to multiple of 32)"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
