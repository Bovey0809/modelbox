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

#include "paddle_ocr_cls_post_flowunit.h"

#include <securec.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

modelbox::Status PaddleOcrClsPostFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  cls_thresh_ = opts->GetFloat("cls_thresh", 0.9F);
  MBLOG_INFO << "paddle_ocr_cls_post: cls_thresh=" << cls_thresh_;
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrClsPostFlowUnit::Close() {
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrClsPostFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto in_logits = ctx->Input("in_logits");
  auto in_crop = ctx->Input("in_crop");
  auto out_crop = ctx->Output("out_crop");
  if (!in_logits || !in_crop || !out_crop) {
    return {modelbox::STATUS_FAULT, "paddle_ocr_cls_post: missing port"};
  }

  const size_t n_crop = in_crop->Size();
  const size_t n_logit = in_logits->Size();
  if (n_crop == 0) {
    return modelbox::STATUS_OK;
  }

  // Gather logits into a flat float array of length 2*N.
  // Expected scheme: one buffer holding the whole batch [N,2]; tolerate
  // the alternative of N separate [2] buffers.
  std::vector<float> logits_flat;
  logits_flat.reserve(n_crop * 2);

  if (n_logit == 1) {
    auto lbuf = in_logits->At(0);
    const size_t bytes = lbuf->GetBytes();
    const size_t need_bytes = n_crop * 2 * sizeof(float);
    if (bytes != need_bytes) {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_cls_post: logits bytes " + std::to_string(bytes) +
                  " != expected " + std::to_string(need_bytes)};
    }
    const auto *src = static_cast<const float *>(lbuf->ConstData());
    logits_flat.assign(src, src + n_crop * 2);
  } else if (n_logit == n_crop) {
    for (size_t i = 0; i < n_crop; ++i) {
      auto lbuf = in_logits->At(i);
      if (lbuf->GetBytes() != 2 * sizeof(float)) {
        return {modelbox::STATUS_FAULT,
                "paddle_ocr_cls_post: per-crop logits buffer not [2] floats"};
      }
      const auto *src = static_cast<const float *>(lbuf->ConstData());
      logits_flat.push_back(src[0]);
      logits_flat.push_back(src[1]);
    }
  } else {
    return {modelbox::STATUS_FAULT,
            "paddle_ocr_cls_post: in_logits size " + std::to_string(n_logit) +
                " incompatible with in_crop size " + std::to_string(n_crop)};
  }

  // Build out_crop sizes (same as input crop bytes -- rotation preserves size).
  std::vector<size_t> out_sizes(n_crop);
  for (size_t i = 0; i < n_crop; ++i) {
    out_sizes[i] = in_crop->At(i)->GetBytes();
  }
  auto bs = out_crop->Build(out_sizes);
  if (!bs) return bs;

  for (size_t i = 0; i < n_crop; ++i) {
    auto src_buf = in_crop->At(i);
    auto dst_buf = out_crop->At(i);

    int32_t w = 0;
    int32_t h = 0;
    int32_t c = 3;
    src_buf->Get("width", w);
    src_buf->Get("height", h);
    src_buf->Get("channel", c);
    if (w <= 0 || h <= 0 || c != 3) {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_cls_post: bad crop meta"};
    }

    const size_t need = static_cast<size_t>(w) * h * 3;
    if (src_buf->GetBytes() != need || dst_buf->GetBytes() != need) {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_cls_post: crop buffer size mismatch"};
    }

    const float l0 = logits_flat[i * 2];
    const float l1 = logits_flat[i * 2 + 1];
    const float m = std::max(l0, l1);
    const float e0 = std::exp(l0 - m);
    const float e1 = std::exp(l1 - m);
    const float p180 = e1 / (e0 + e1);

    auto *src_raw = const_cast<void *>(src_buf->ConstData());
    cv::Mat src(h, w, CV_8UC3, src_raw);
    auto *dst = static_cast<uint8_t *>(dst_buf->MutableData());

    int32_t angle = 0;
    if (p180 >= cls_thresh_) {
      cv::Mat rot;
      cv::rotate(src, rot, cv::ROTATE_180);
      if (rot.isContinuous()) {
        if (memcpy_s(dst, need, rot.data, need) != EOK) {
          return {modelbox::STATUS_FAULT,
                  "paddle_ocr_cls_post: rotated copy failed"};
        }
      } else {
        const size_t row_bytes = static_cast<size_t>(w) * 3;
        for (int r = 0; r < h; ++r) {
          if (memcpy_s(dst + r * row_bytes, row_bytes, rot.ptr(r), row_bytes) !=
              EOK) {
            return {modelbox::STATUS_FAULT,
                    "paddle_ocr_cls_post: rotated row copy failed"};
          }
        }
      }
      angle = 180;
    } else {
      if (memcpy_s(dst, need, src_buf->ConstData(), need) != EOK) {
        return {modelbox::STATUS_FAULT,
                "paddle_ocr_cls_post: passthrough copy failed"};
      }
      angle = 0;
    }

    // Meta passthrough (match paddle_ocr_crop_rotate's written keys).
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
    dst_buf->Set("angle", angle);
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(PaddleOcrClsPostFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_logits"});
  desc.AddFlowUnitInput({"in_crop"});
  desc.AddFlowUnitOutput({"out_crop"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "cls_thresh", "float", true, "0.9",
      "minimum 180-degree softmax probability required to rotate"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
