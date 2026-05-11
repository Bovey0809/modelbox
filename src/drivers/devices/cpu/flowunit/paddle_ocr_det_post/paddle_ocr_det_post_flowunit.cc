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

#include "paddle_ocr_det_post_flowunit.h"

#include <securec.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

namespace {

struct Pt {
  float x;
  float y;
};

// Unclip a 4-point box by scaling away from its centroid by `ratio`.
// Simple scalar dilation around centroid (Vatti offset is overkill for V1).
void UnclipBox(cv::Point2f pts[4], float ratio, std::array<Pt, 4> &out) {
  float cx = 0.0F;
  float cy = 0.0F;
  for (int i = 0; i < 4; ++i) {
    cx += pts[i].x;
    cy += pts[i].y;
  }
  cx *= 0.25F;
  cy *= 0.25F;
  for (int i = 0; i < 4; ++i) {
    out[i].x = cx + (pts[i].x - cx) * ratio;
    out[i].y = cy + (pts[i].y - cy) * ratio;
  }
}

}  // namespace

modelbox::Status PaddleOcrDetPostFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  db_thresh_ = opts->GetFloat("db_thresh", 0.3F);
  box_thresh_ = opts->GetFloat("box_thresh", 0.6F);
  unclip_ratio_ = opts->GetFloat("unclip_ratio", 1.5F);
  max_candidates_ = opts->GetInt32("max_candidates", 1000);
  min_size_ = opts->GetInt32("min_size", 3);
  MBLOG_INFO << "paddle_ocr_det_post: db_thresh=" << db_thresh_
             << " box_thresh=" << box_thresh_
             << " unclip_ratio=" << unclip_ratio_
             << " max_candidates=" << max_candidates_
             << " min_size=" << min_size_;
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrDetPostFlowUnit::Close() {
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrDetPostFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto in_image = ctx->Input("in_image");
  auto in_prob = ctx->Input("in_prob");
  auto out_image = ctx->Output("out_image");
  if (!in_image || !in_prob || !out_image) {
    return {modelbox::STATUS_FAULT, "paddle_ocr_det_post: missing port"};
  }
  const size_t n = in_image->Size();
  if (n == 0) return modelbox::STATUS_OK;
  if (in_prob->Size() != n) {
    return {modelbox::STATUS_FAULT,
            "paddle_ocr_det_post: image/prob batch size mismatch"};
  }

  std::vector<size_t> image_sizes(n);
  std::vector<int32_t> widths(n);
  std::vector<int32_t> heights(n);
  for (size_t i = 0; i < n; ++i) {
    auto buf = in_image->At(i);
    int32_t w = 0;
    int32_t h = 0;
    int32_t c = 3;
    buf->Get("width", w);
    buf->Get("height", h);
    buf->Get("channel", c);
    if (w <= 0 || h <= 0 || c != 3) {
      return {modelbox::STATUS_FAULT, "paddle_ocr_det_post: bad image meta"};
    }
    widths[i] = w;
    heights[i] = h;
    image_sizes[i] = static_cast<size_t>(w) * h * c;
  }

  auto bs = out_image->Build(image_sizes);
  if (!bs) return bs;

  for (size_t i = 0; i < n; ++i) {
    auto img_buf = in_image->At(i);
    auto prob_buf = in_prob->At(i);

    std::vector<size_t> shape;
    if (!prob_buf->Get("shape", shape)) {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_det_post: prob shape meta missing"};
    }
    int ph = 0;
    int pw = 0;
    if (shape.size() == 4) {
      ph = static_cast<int>(shape[2]);
      pw = static_cast<int>(shape[3]);
    } else if (shape.size() == 2) {
      ph = static_cast<int>(shape[0]);
      pw = static_cast<int>(shape[1]);
    } else {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_det_post: unsupported prob shape rank"};
    }
    if (ph <= 0 || pw <= 0) {
      return {modelbox::STATUS_FAULT, "paddle_ocr_det_post: bad prob shape"};
    }

    const int orig_w = widths[i];
    const int orig_h = heights[i];

    // Wrap prob data as cv::Mat (no copy).
    const auto *prob_data = static_cast<const float *>(prob_buf->ConstData());
    cv::Mat prob(ph, pw, CV_32F, const_cast<void *>(
                                       static_cast<const void *>(prob_data)));

    cv::Mat bin;
    cv::threshold(prob, bin, db_thresh_, 1.0, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U, 255);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    if (static_cast<int>(contours.size()) > max_candidates_) {
      contours.resize(static_cast<size_t>(max_candidates_));
    }

    const float sx = static_cast<float>(orig_w) / static_cast<float>(pw);
    const float sy = static_cast<float>(orig_h) / static_cast<float>(ph);

    std::vector<float> polygons;
    int32_t polygon_count = 0;
    polygons.reserve(contours.size() * 8);

    for (const auto &contour : contours) {
      if (static_cast<int>(contour.size()) < min_size_) continue;

      cv::RotatedRect rect = cv::minAreaRect(contour);
      if (rect.size.width < static_cast<float>(min_size_) ||
          rect.size.height < static_cast<float>(min_size_)) {
        continue;
      }
      cv::Point2f pts[4];
      rect.points(pts);

      // Build polygon mask and compute mean prob inside.
      cv::Mat mask = cv::Mat::zeros(ph, pw, CV_8U);
      std::vector<cv::Point> poly(4);
      for (int k = 0; k < 4; ++k) {
        int px = std::max(0, std::min(pw - 1, static_cast<int>(pts[k].x)));
        int py = std::max(0, std::min(ph - 1, static_cast<int>(pts[k].y)));
        poly[k] = cv::Point(px, py);
      }
      std::vector<std::vector<cv::Point>> poly_wrap = {poly};
      cv::fillPoly(mask, poly_wrap, cv::Scalar(1));
      double score = cv::mean(prob, mask)[0];
      if (score < box_thresh_) continue;

      // Unclip by scalar dilation about centroid.
      std::array<Pt, 4> uc{};
      UnclipBox(pts, unclip_ratio_, uc);

      // Scale back to original image space and clamp.
      for (int k = 0; k < 4; ++k) {
        float x = uc[k].x * sx;
        float y = uc[k].y * sy;
        x = std::max(0.0F, std::min(static_cast<float>(orig_w - 1), x));
        y = std::max(0.0F, std::min(static_cast<float>(orig_h - 1), y));
        polygons.push_back(x);
        polygons.push_back(y);
      }
      ++polygon_count;
    }

    // Copy original image bytes through.
    auto img_out_buf = out_image->At(i);
    const auto *src = static_cast<const uint8_t *>(img_buf->ConstData());
    auto *dst = static_cast<uint8_t *>(img_out_buf->MutableData());
    if (memcpy_s(dst, img_out_buf->GetBytes(), src, img_buf->GetBytes()) !=
        EOK) {
      return {modelbox::STATUS_FAULT, "paddle_ocr_det_post: image copy failed"};
    }
    img_out_buf->CopyMeta(img_buf);
    img_out_buf->Set("width", widths[i]);
    img_out_buf->Set("height", heights[i]);
    img_out_buf->Set("channel", 3);
    img_out_buf->Set("polygon_count", polygon_count);
    img_out_buf->Set("polygons", polygons);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(PaddleOcrDetPostFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_image"});
  desc.AddFlowUnitInput({"in_prob"});
  desc.AddFlowUnitOutput({"out_image"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "db_thresh", "float", true, "0.3", "binarization threshold on probmap"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "box_thresh", "float", true, "0.6",
      "minimum mean-prob score to keep a polygon"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "unclip_ratio", "float", true, "1.5",
      "scalar dilation factor around centroid"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "max_candidates", "int", true, "1000",
      "max number of contours to consider per image"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "min_size", "int", true, "3",
      "minimum contour size / box edge in probmap pixels"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
