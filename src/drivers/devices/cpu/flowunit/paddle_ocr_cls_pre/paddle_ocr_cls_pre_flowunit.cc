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

#include "paddle_ocr_cls_pre_flowunit.h"

#include <securec.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

namespace {
constexpr int kClsH = 48;
constexpr int kClsW = 192;
// PaddleOCR angle classifier normalization: mean=std=0.5.
constexpr float kClsMean = 0.5F;
constexpr float kClsStd = 0.5F;
constexpr float kScale = 1.0F / 255.0F;
}  // namespace

modelbox::Status PaddleOcrClsPreFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  MBLOG_INFO << "paddle_ocr_cls_pre: fixed output 192x48, mean/std=0.5";
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrClsPreFlowUnit::Close() {
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrClsPreFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto in_crop = ctx->Input("in_crop");
  auto out_tensor = ctx->Output("out_tensor");
  auto out_crop = ctx->Output("out_crop");
  if (!in_crop || !out_tensor || !out_crop) {
    return {modelbox::STATUS_FAULT, "paddle_ocr_cls_pre: missing port"};
  }
  const size_t n = in_crop->Size();
  if (n == 0) return modelbox::STATUS_OK;

  std::vector<int32_t> widths(n);
  std::vector<int32_t> heights(n);
  std::vector<int> rws(n);
  std::vector<size_t> csizes(n);

  for (size_t i = 0; i < n; ++i) {
    auto buf = in_crop->At(i);
    int32_t w = 0, h = 0, c = 3;
    buf->Get("width", w);
    buf->Get("height", h);
    buf->Get("channel", c);
    if (w <= 0 || h <= 0 || c != 3) {
      return {modelbox::STATUS_FAULT, "paddle_ocr_cls_pre: bad crop meta"};
    }
    const size_t need = static_cast<size_t>(w) * h * 3;
    if (buf->GetBytes() != need) {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_cls_pre: crop buffer size mismatch"};
    }
    widths[i] = w;
    heights[i] = h;
    int rw = static_cast<int>(
        std::round(static_cast<float>(w) * kClsH / static_cast<float>(h)));
    if (rw < 1) rw = 1;
    if (rw > kClsW) rw = kClsW;
    rws[i] = rw;
    csizes[i] = need;
  }

  const size_t per = static_cast<size_t>(3) * kClsH * kClsW * sizeof(float);

  std::vector<size_t> tensor_sizes(n, per);
  auto bs = out_tensor->Build(tensor_sizes);
  if (!bs) return bs;
  bs = out_crop->Build(csizes);
  if (!bs) return bs;

  for (size_t i = 0; i < n; ++i) {
    auto src_buf = in_crop->At(i);
    const int w = widths[i];
    const int h = heights[i];
    const int rw = rws[i];
    const auto *src = static_cast<const uint8_t *>(src_buf->ConstData());

    cv::Mat in_mat(h, w, CV_8UC3, const_cast<uint8_t *>(src));
    cv::Mat resized;
    cv::resize(in_mat, resized, cv::Size(rw, kClsH), 0, 0, cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    auto tensor_buf = out_tensor->At(i);
    auto *dst = static_cast<float *>(tensor_buf->MutableData());
    std::memset(dst, 0, per);

    // HWC u8 -> CHW fp32 normalized with mean=std=0.5; zero-pad to 192.
    for (int y = 0; y < kClsH; ++y) {
      const auto *row = rgb.ptr<uint8_t>(y);
      for (int x = 0; x < rw; ++x) {
        for (int c = 0; c < 3; ++c) {
          float v = static_cast<float>(row[x * 3 + c]) * kScale;
          dst[c * kClsH * kClsW + y * kClsW + x] = (v - kClsMean) / kClsStd;
        }
      }
    }
    std::vector<size_t> tshape = {1, 3, static_cast<size_t>(kClsH),
                                  static_cast<size_t>(kClsW)};
    tensor_buf->Set("shape", tshape);
    tensor_buf->Set("type", modelbox::ModelBoxDataType::MODELBOX_FLOAT);

    auto dst_buf = out_crop->At(i);
    auto *crop_dst = static_cast<uint8_t *>(dst_buf->MutableData());
    if (memcpy_s(crop_dst, dst_buf->GetBytes(), src, src_buf->GetBytes()) !=
        EOK) {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_cls_pre: crop passthrough copy failed"};
    }

    dst_buf->Set("width", w);
    dst_buf->Set("height", h);
    dst_buf->Set("channel", 3);
    dst_buf->Set("pix_fmt", std::string("bgr"));
    dst_buf->Set("width_stride", static_cast<int32_t>(w * 3));
    dst_buf->Set("height_stride", h);
    dst_buf->Set("shape",
                 std::vector<size_t>{static_cast<size_t>(h),
                                     static_cast<size_t>(w), 3});
    dst_buf->Set("layout", std::string("hwc"));
    dst_buf->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);

    int64_t frame_id = 0;
    int32_t box_id = 0;
    int32_t box_count = 0;
    std::vector<float> polygon;
    if (src_buf->Get("frame_id", frame_id)) {
      dst_buf->Set("frame_id", frame_id);
    }
    if (src_buf->Get("box_id", box_id)) {
      dst_buf->Set("box_id", box_id);
    }
    if (src_buf->Get("box_count", box_count)) {
      dst_buf->Set("box_count", box_count);
    }
    if (src_buf->Get("polygon", polygon)) {
      dst_buf->Set("polygon", polygon);
    }
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(PaddleOcrClsPreFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_crop"});
  desc.AddFlowUnitOutput({"out_tensor"});
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
