/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "tracknet_post_flowunit.h"

#include <algorithm>
#include <cstring>
#include <sstream>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

extern "C" bool TrackNetExtractCentroid(const float *heatmap, int h, int w,
                                        float score_thr, float mask_ratio,
                                        float *cx_out, float *cy_out,
                                        float *peak_out) {
  float peak = 0.f;
  int n = h * w;
  for (int i = 0; i < n; ++i) {
    if (heatmap[i] > peak) peak = heatmap[i];
  }
  if (peak_out) *peak_out = peak;
  if (peak < score_thr) return false;

  float thr = peak * mask_ratio;
  double wx = 0;
  double wy = 0;
  double ws = 0;
  for (int y = 0; y < h; ++y) {
    const float *row = heatmap + y * w;
    for (int x = 0; x < w; ++x) {
      float v = row[x];
      if (v >= thr) {
        wx += static_cast<double>(x) * v;
        wy += static_cast<double>(y) * v;
        ws += v;
      }
    }
  }
  if (ws <= 0) return false;
  *cx_out = static_cast<float>(wx / ws);
  *cy_out = static_cast<float>(wy / ws);
  return true;
}

namespace {

cv::Scalar ParseBGR(const std::string &s, cv::Scalar def) {
  std::stringstream ss(s);
  int b = 0;
  int g = 0;
  int r = 0;
  char comma = 0;
  if (!(ss >> b >> comma >> g >> comma >> r)) return def;
  return cv::Scalar(b, g, r);
}

}  // namespace

modelbox::Status TrackNetPostFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  net_h_ = opts->GetInt32("net_h", net_h_);
  net_w_ = opts->GetInt32("net_w", net_w_);
  heatmap_channel_ = opts->GetInt32("heatmap_channel", heatmap_channel_);
  score_thr_ = opts->GetFloat("score_thr", score_thr_);
  mask_ratio_ = opts->GetFloat("mask_ratio", mask_ratio_);
  circle_radius_ = opts->GetInt32("circle_radius", circle_radius_);
  draw_heatmap_ = opts->GetBool("draw_heatmap", draw_heatmap_);
  circle_color_ =
      ParseBGR(opts->GetString("circle_color", "0,0,255"), circle_color_);
  return modelbox::STATUS_OK;
}

modelbox::Status TrackNetPostFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status TrackNetPostFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto hm_list = data_ctx->Input("heatmaps");
  auto src_list = data_ctx->Input("source_frame");
  auto out_list = data_ctx->Output("frame_out");
  if (hm_list->Size() != src_list->Size()) {
    return {modelbox::STATUS_FAULT, "heatmaps/source_frame size mismatch"};
  }

  const int H = net_h_;
  const int W = net_w_;
  const size_t plane = static_cast<size_t>(H) * static_cast<size_t>(W);
  const size_t n = hm_list->Size();

  // Pre-compute output shapes from source-frame metadata so the BufferList
  // allocates contiguous storage we then memcpy into.
  std::vector<size_t> shapes(n, 0);
  std::vector<int> src_ws(n, 0);
  std::vector<int> src_hs(n, 0);
  for (size_t i = 0; i < n; ++i) {
    auto src = src_list->At(i);
    int sw = 0;
    int sh = 0;
    src->Get("width", sw);
    src->Get("height", sh);
    if (sw <= 0 || sh <= 0) {
      return {modelbox::STATUS_FAULT,
              "source_frame missing width/height meta"};
    }
    src_ws[i] = sw;
    src_hs[i] = sh;
    shapes[i] = static_cast<size_t>(sw) * static_cast<size_t>(sh) * 3u;
  }

  auto build = out_list->Build(shapes);
  if (!build) return build;

  for (size_t i = 0; i < n; ++i) {
    auto hm = hm_list->At(i);
    auto src = src_list->At(i);
    auto out = out_list->At(i);

    const int src_w = src_ws[i];
    const int src_h = src_hs[i];

    std::memcpy(out->MutableData(), src->ConstData(), shapes[i]);
    cv::Mat draw(src_h, src_w, CV_8UC3, out->MutableData());

    const auto *hmap = static_cast<const float *>(hm->ConstData());
    const float *channel =
        hmap + static_cast<size_t>(heatmap_channel_) * plane;

    float cx_lr = 0;
    float cy_lr = 0;
    float peak = 0;
    bool ok = TrackNetExtractCentroid(channel, H, W, score_thr_, mask_ratio_,
                                      &cx_lr, &cy_lr, &peak);
    if (ok) {
      float cx = cx_lr * static_cast<float>(src_w) / static_cast<float>(W);
      float cy = cy_lr * static_cast<float>(src_h) / static_cast<float>(H);
      cv::circle(draw, cv::Point(static_cast<int>(cx), static_cast<int>(cy)),
                 circle_radius_, circle_color_, cv::FILLED, cv::LINE_AA);
    }

    if (draw_heatmap_) {
      cv::Mat hm_lr(H, W, CV_32FC1, const_cast<float *>(channel));
      cv::Mat hm_resized;
      cv::resize(hm_lr, hm_resized, cv::Size(src_w, src_h));
      cv::Mat hm_u8;
      hm_resized.convertTo(hm_u8, CV_8UC1, 255.0);
      cv::Mat hm_color;
      cv::applyColorMap(hm_u8, hm_color, cv::COLORMAP_JET);
      cv::addWeighted(draw, 0.7, hm_color, 0.3, 0, draw);
    }

    out->Set("width", src_w);
    out->Set("height", src_h);
    out->Set("pix_fmt", std::string("bgr"));
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(TrackNetPostFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"heatmaps"});
  desc.AddFlowUnitInput({"source_frame"});
  desc.AddFlowUnitOutput({"frame_out"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "net_h", "int", true, "288", "TrackNet input/heatmap height"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "net_w", "int", true, "512", "TrackNet input/heatmap width"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "heatmap_channel", "int", true, "2",
      "channel index of the ball heatmap in the CHW output"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "score_thr", "float", true, "0.3",
      "min peak heatmap value to emit a detection"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "mask_ratio", "float", true, "0.5",
      "weighted-centroid mask threshold = peak * mask_ratio"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "circle_radius", "int", true, "6", "drawn circle radius in pixels"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "circle_color", "string", true, "0,0,255",
      "BGR triplet, comma-separated"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "draw_heatmap", "bool", true, "false",
      "overlay JET-colored heatmap on the frame"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
