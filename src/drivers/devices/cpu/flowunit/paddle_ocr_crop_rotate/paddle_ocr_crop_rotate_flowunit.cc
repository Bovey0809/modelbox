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

#include "paddle_ocr_crop_rotate_flowunit.h"

#include <securec.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

namespace {

struct CropJob {
  size_t in_idx;
  std::vector<cv::Point2f> poly;  // 4 points
  int rw;
  int rh;
  int32_t box_id;
  int32_t box_count;
  int64_t frame_id;
};

struct FrameMeta {
  int64_t frame_id;
  int32_t box_count;
  int32_t width;
  int32_t height;
  size_t bytes;
};

inline float Dist(const cv::Point2f &a, const cv::Point2f &b) {
  float dx = a.x - b.x;
  float dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

}  // namespace

modelbox::Status PaddleOcrCropRotateFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> & /*opts*/) {
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrCropRotateFlowUnit::Close() {
  return modelbox::STATUS_OK;
}

modelbox::Status PaddleOcrCropRotateFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> ctx) {
  auto in_image = ctx->Input("in_image");
  auto out_crop = ctx->Output("out_crop");
  auto out_image = ctx->Output("out_image");
  if (!in_image || !out_crop || !out_image) {
    return {modelbox::STATUS_FAULT, "paddle_ocr_crop_rotate: missing port"};
  }
  const size_t n = in_image->Size();
  if (n == 0) return modelbox::STATUS_OK;

  std::vector<CropJob> jobs;
  std::vector<FrameMeta> frame_metas(n);
  std::vector<size_t> image_sizes(n);

  for (size_t i = 0; i < n; ++i) {
    auto buf = in_image->At(i);
    int32_t w = 0;
    int32_t h = 0;
    int32_t c = 3;
    buf->Get("width", w);
    buf->Get("height", h);
    buf->Get("channel", c);
    if (w <= 0 || h <= 0 || c != 3) {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_crop_rotate: bad image meta"};
    }
    int32_t polygon_count = 0;
    buf->Get("polygon_count", polygon_count);
    std::vector<float> polygons;
    buf->Get("polygons", polygons);
    if (polygon_count < 0) polygon_count = 0;
    if (static_cast<size_t>(polygon_count) * 8 > polygons.size()) {
      polygon_count = static_cast<int32_t>(polygons.size() / 8);
    }

    int64_t frame_id = next_frame_id_.fetch_add(1);
    frame_metas[i] = {frame_id, polygon_count, w, h,
                      static_cast<size_t>(w) * h * c};
    image_sizes[i] = frame_metas[i].bytes;

    for (int32_t k = 0; k < polygon_count; ++k) {
      const float *p = polygons.data() + (static_cast<size_t>(k) * 8);
      std::vector<cv::Point2f> pts(4);
      for (int j = 0; j < 4; ++j) {
        pts[j] = cv::Point2f(p[j * 2], p[j * 2 + 1]);
      }
      float w0 = Dist(pts[1], pts[0]);
      float w1 = Dist(pts[3], pts[2]);
      float h0 = Dist(pts[2], pts[1]);
      float h1 = Dist(pts[0], pts[3]);
      int rw = static_cast<int>(std::round(std::max(w0, w1)));
      int rh = static_cast<int>(std::round(std::max(h0, h1)));
      if (rw < 1) rw = 1;
      if (rh < 1) rh = 1;
      jobs.push_back({i, std::move(pts), rw, rh, k, polygon_count, frame_id});
    }
  }

  // Build out_image buffers (one per input frame).
  auto bs = out_image->Build(image_sizes);
  if (!bs) return bs;

  // Build out_crop buffers (one per polygon across all frames).
  std::vector<size_t> crop_sizes;
  crop_sizes.reserve(jobs.size());
  for (const auto &job : jobs) {
    crop_sizes.push_back(static_cast<size_t>(job.rw) * job.rh * 3);
  }
  if (!crop_sizes.empty()) {
    auto bs2 = out_crop->Build(crop_sizes);
    if (!bs2) return bs2;
  }

  // Fill crops.
  for (size_t j = 0; j < jobs.size(); ++j) {
    const auto &job = jobs[j];
    auto src_buf = in_image->At(job.in_idx);
    const int sw = frame_metas[job.in_idx].width;
    const int sh = frame_metas[job.in_idx].height;
    auto *src_raw = const_cast<void *>(src_buf->ConstData());
    cv::Mat src(sh, sw, CV_8UC3, src_raw);

    cv::Point2f src_pts[4] = {job.poly[0], job.poly[1], job.poly[2],
                              job.poly[3]};
    cv::Point2f dst_pts[4] = {
        cv::Point2f(0.0F, 0.0F),
        cv::Point2f(static_cast<float>(job.rw - 1), 0.0F),
        cv::Point2f(static_cast<float>(job.rw - 1),
                    static_cast<float>(job.rh - 1)),
        cv::Point2f(0.0F, static_cast<float>(job.rh - 1))};
    cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat crop;
    cv::warpPerspective(src, crop, M, cv::Size(job.rw, job.rh),
                        cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    int out_w = job.rw;
    int out_h = job.rh;
    if (static_cast<float>(out_h) > 1.5F * static_cast<float>(out_w)) {
      cv::Mat rotated;
      cv::rotate(crop, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
      crop = rotated;
      std::swap(out_w, out_h);
    }

    auto crop_buf = out_crop->At(j);
    const size_t need = static_cast<size_t>(out_w) * out_h * 3;
    if (crop_buf->GetBytes() != need) {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_crop_rotate: crop buffer size mismatch"};
    }
    auto *dst = static_cast<uint8_t *>(crop_buf->MutableData());
    if (crop.isContinuous()) {
      if (memcpy_s(dst, need, crop.data, need) != EOK) {
        return {modelbox::STATUS_FAULT,
                "paddle_ocr_crop_rotate: crop copy failed"};
      }
    } else {
      size_t row_bytes = static_cast<size_t>(out_w) * 3;
      for (int r = 0; r < out_h; ++r) {
        if (memcpy_s(dst + r * row_bytes, row_bytes, crop.ptr(r), row_bytes) !=
            EOK) {
          return {modelbox::STATUS_FAULT,
                  "paddle_ocr_crop_rotate: crop row copy failed"};
        }
      }
    }
    std::vector<float> poly_flat(8);
    for (int k = 0; k < 4; ++k) {
      poly_flat[k * 2] = job.poly[k].x;
      poly_flat[k * 2 + 1] = job.poly[k].y;
    }
    crop_buf->Set("width", static_cast<int32_t>(out_w));
    crop_buf->Set("height", static_cast<int32_t>(out_h));
    crop_buf->Set("channel", 3);
    crop_buf->Set("pix_fmt", std::string("bgr"));
    crop_buf->Set("width_stride", static_cast<int32_t>(out_w * 3));
    crop_buf->Set("height_stride", static_cast<int32_t>(out_h));
    crop_buf->Set("shape",
                  std::vector<size_t>{static_cast<size_t>(out_h),
                                      static_cast<size_t>(out_w), 3});
    crop_buf->Set("layout", std::string("hwc"));
    crop_buf->Set("type", modelbox::ModelBoxDataType::MODELBOX_UINT8);
    crop_buf->Set("frame_id", job.frame_id);
    crop_buf->Set("box_id", job.box_id);
    crop_buf->Set("box_count", job.box_count);
    crop_buf->Set("polygon", poly_flat);
  }

  // Forward original frame on side output port.
  for (size_t i = 0; i < n; ++i) {
    auto src_buf = in_image->At(i);
    auto img_out_buf = out_image->At(i);
    const auto *src = static_cast<const uint8_t *>(src_buf->ConstData());
    auto *dst = static_cast<uint8_t *>(img_out_buf->MutableData());
    if (memcpy_s(dst, img_out_buf->GetBytes(), src, src_buf->GetBytes()) !=
        EOK) {
      return {modelbox::STATUS_FAULT,
              "paddle_ocr_crop_rotate: image copy failed"};
    }
    img_out_buf->CopyMeta(src_buf);
    img_out_buf->Set("width", frame_metas[i].width);
    img_out_buf->Set("height", frame_metas[i].height);
    img_out_buf->Set("channel", 3);
    img_out_buf->Set("frame_id", frame_metas[i].frame_id);
    img_out_buf->Set("box_count", frame_metas[i].box_count);
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(PaddleOcrCropRotateFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_image"});
  desc.AddFlowUnitOutput({"out_crop"});
  desc.AddFlowUnitOutput({"out_image"});
  desc.SetFlowType(modelbox::STREAM);
  desc.SetOutputType(modelbox::EXPAND);
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
