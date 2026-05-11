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

#include "paddle_ocr_draw_flowunit.h"

#include <securec.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

modelbox::Status PaddleOcrDrawFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  text_scale_ = opts->GetFloat("text_scale", 0.6F);
  line_thickness_ = opts->GetInt32("line_thickness", 2);
  MBLOG_INFO << "paddle_ocr_draw: text_scale=" << text_scale_
             << " line_thickness=" << line_thickness_;
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrDrawFlowUnit::Close() {
  std::lock_guard<std::mutex> lk(mu_);
  frames_.clear();
  pending_.clear();
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrDrawFlowUnit::Emit(
    const std::shared_ptr<modelbox::BufferList> &out, size_t idx,
    const FrameEntry &frame, const std::vector<ResultEntry> &results) {
  const int w = frame.width;
  const int h = frame.height;
  const size_t need = static_cast<size_t>(w) * h * 3;
  auto buf = out->At(idx);
  if (buf->GetBytes() != need) {
    return {modelbox::STATUS_FAULT,
            "paddle_ocr_draw: out buffer size mismatch"};
  }

  // Copy source bytes into a writable cv::Mat we own.
  cv::Mat img(h, w, CV_8UC3);
  if (memcpy_s(img.data, need, frame.bytes.data(), frame.bytes.size()) != EOK) {
    return {modelbox::STATUS_FAULT, "paddle_ocr_draw: frame copy failed"};
  }

  for (const auto &r : results) {
    std::vector<cv::Point> poly(4);
    for (int k = 0; k < 4; ++k) {
      int px = std::max(
          0, std::min(w - 1, static_cast<int>(std::lround(r.polygon[k].x))));
      int py = std::max(
          0, std::min(h - 1, static_cast<int>(std::lround(r.polygon[k].y))));
      poly[k] = cv::Point(px, py);
    }
    std::vector<std::vector<cv::Point>> polys = {poly};
    cv::polylines(img, polys, true, cv::Scalar(0, 255, 0), line_thickness_);

    if (!r.text.empty()) {
      char label[256];
      std::snprintf(label, sizeof(label), "%s %.2f", r.text.c_str(),
                    static_cast<double>(r.score));
      cv::Point anchor(poly[0].x, std::max(12, poly[0].y));
      cv::putText(img, label, anchor, cv::FONT_HERSHEY_SIMPLEX,
                  static_cast<double>(text_scale_), cv::Scalar(0, 255, 0), 1);
    }
  }

  auto *dst = static_cast<uint8_t *>(buf->MutableData());
  if (memcpy_s(dst, need, img.data, need) != EOK) {
    return {modelbox::STATUS_FAULT, "paddle_ocr_draw: out copy failed"};
  }

  buf->Set("width", static_cast<int32_t>(w));
  buf->Set("height", static_cast<int32_t>(h));
  buf->Set("channel", static_cast<int32_t>(3));
  buf->Set("pix_fmt", std::string("bgr"));
  buf->Set("width_stride", static_cast<int32_t>(w * 3));
  buf->Set("height_stride", static_cast<int32_t>(h));
  buf->Set("shape", std::vector<size_t>{static_cast<size_t>(h),
                                        static_cast<size_t>(w), 3});
  buf->Set("layout", std::string("hwc"));
  buf->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrDrawFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto in_image = ctx->Input("in_image");
  auto in_result = ctx->Input("in_result");
  auto out_image = ctx->Output("out_image");
  if (!out_image) {
    return {modelbox::STATUS_FAULT, "paddle_ocr_draw: missing out_image port"};
  }

  // Pairs queued for emission this call: (frame, results).
  std::vector<std::pair<FrameEntry, std::vector<ResultEntry>>> ready;

  {
    std::lock_guard<std::mutex> lk(mu_);

    // A. Ingest in_image.
    if (in_image) {
      const size_t n_img = in_image->Size();
      for (size_t i = 0; i < n_img; ++i) {
        auto buf = in_image->At(i);
        int64_t frame_id = 0;
        int32_t w = 0;
        int32_t h = 0;
        int32_t c = 3;
        int32_t box_count = 0;
        buf->Get("frame_id", frame_id);
        buf->Get("width", w);
        buf->Get("height", h);
        buf->Get("channel", c);
        buf->Get("box_count", box_count);
        if (w <= 0 || h <= 0 || c != 3) {
          return {modelbox::STATUS_FAULT,
                  "paddle_ocr_draw: bad image meta on in_image"};
        }
        const size_t need = static_cast<size_t>(w) * h * 3;
        if (buf->GetBytes() < need) {
          return {modelbox::STATUS_FAULT,
                  "paddle_ocr_draw: in_image buffer smaller than w*h*3"};
        }

        FrameEntry fe;
        fe.width = w;
        fe.height = h;
        fe.channel = c;
        fe.box_count = box_count;
        fe.bytes.resize(need);
        if (memcpy_s(fe.bytes.data(), need, buf->ConstData(), need) != EOK) {
          return {modelbox::STATUS_FAULT,
                  "paddle_ocr_draw: in_image snapshot failed"};
        }

        if (box_count == 0) {
          // Emit immediately; no join needed.
          ready.emplace_back(std::move(fe), std::vector<ResultEntry>{});
          continue;
        }

        // Check if pending already has enough results.
        auto it = pending_.find(frame_id);
        if (it != pending_.end() &&
            static_cast<int32_t>(it->second.size()) >= box_count) {
          std::vector<ResultEntry> rs = std::move(it->second);
          pending_.erase(it);
          ready.emplace_back(std::move(fe), std::move(rs));
        } else {
          frames_[frame_id] = std::move(fe);
        }
      }
    }

    // B. Ingest in_result.
    if (in_result) {
      const size_t n_res = in_result->Size();
      for (size_t i = 0; i < n_res; ++i) {
        auto buf = in_result->At(i);
        int64_t frame_id = 0;
        int32_t box_id = 0;
        int32_t box_count = 0;
        std::vector<float> poly_flat;
        std::string text;
        float score = 0.0F;
        buf->Get("frame_id", frame_id);
        buf->Get("box_id", box_id);
        buf->Get("box_count", box_count);
        buf->Get("polygon", poly_flat);
        buf->Get("text", text);
        buf->Get("score", score);

        ResultEntry re;
        re.text = std::move(text);
        re.score = score;
        if (poly_flat.size() >= 8) {
          for (int k = 0; k < 4; ++k) {
            re.polygon[k] = cv::Point2f(poly_flat[2 * k],
                                        poly_flat[2 * k + 1]);
          }
        }

        // If the frame is already cached and this completes the set, emit.
        auto fit = frames_.find(frame_id);
        if (fit != frames_.end()) {
          auto &slot = pending_[frame_id];
          slot.push_back(std::move(re));
          if (static_cast<int32_t>(slot.size()) >= fit->second.box_count) {
            FrameEntry fe = std::move(fit->second);
            frames_.erase(fit);
            std::vector<ResultEntry> rs = std::move(slot);
            pending_.erase(frame_id);
            ready.emplace_back(std::move(fe), std::move(rs));
          }
        } else {
          pending_[frame_id].push_back(std::move(re));
        }
      }
    }

    if (frames_.size() > 256) {
      MBLOG_WARN << "paddle_ocr_draw: frames_ cache size=" << frames_.size()
                 << " (upstream may be dropping frames)";
    }
  }

  if (ready.empty()) {
    // No completed frames to emit this call. Build empty output.
    auto bs = out_image->Build(std::vector<size_t>{});
    (void)bs;
    return modelbox::STATUS_OK;
  }

  std::vector<size_t> sizes;
  sizes.reserve(ready.size());
  for (const auto &p : ready) {
    sizes.push_back(static_cast<size_t>(p.first.width) * p.first.height * 3);
  }
  auto bs = out_image->Build(sizes);
  if (!bs) return bs;

  for (size_t i = 0; i < ready.size(); ++i) {
    auto st = Emit(out_image, i, ready[i].first, ready[i].second);
    if (!st) return st;
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(PaddleOcrDrawFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_image"});
  desc.AddFlowUnitInput({"in_result"});
  desc.AddFlowUnitOutput({"out_image"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetInputContiguous(false);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "text_scale", "float", false, "0.6",
      "OpenCV putText fontScale for OCR label rendering"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "line_thickness", "int", false, "2",
      "OpenCV polylines line thickness for polygon rendering"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
