/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "yolo_track_post_flowunit.h"

#include <securec.h>

#include <algorithm>
#include <cstring>
#include <memory>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

namespace {
constexpr const char *kTrackState = "yolo_track_state";

struct TrackState {
  int next_id{1};
  std::vector<YoloTrackPostFlowUnit::Track> tracks;
};

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

float BoxIoU(float ax1, float ay1, float ax2, float ay2, float bx1, float by1,
             float bx2, float by2) {
  float ix1 = std::max(ax1, bx1);
  float iy1 = std::max(ay1, by1);
  float ix2 = std::min(ax2, bx2);
  float iy2 = std::min(ay2, by2);
  float iw = std::max(0.0F, ix2 - ix1);
  float ih = std::max(0.0F, iy2 - iy1);
  float inter = iw * ih;
  float a_area = std::max(0.0F, ax2 - ax1) * std::max(0.0F, ay2 - ay1);
  float b_area = std::max(0.0F, bx2 - bx1) * std::max(0.0F, by2 - by1);
  float uni = a_area + b_area - inter;
  return uni > 0 ? inter / uni : 0.0F;
}
}  // namespace

modelbox::Status YoloTrackPostFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  net_h_ = opts->GetInt32("net_h", 640);
  net_w_ = opts->GetInt32("net_w", 640);
  num_classes_ = opts->GetInt32("num_classes", 80);
  conf_threshold_ = opts->GetFloat("conf_threshold", 0.35F);
  iou_threshold_ = opts->GetFloat("iou_threshold", 0.45F);
  track_iou_ = opts->GetFloat("track_iou", 0.3F);
  max_lost_ = opts->GetInt32("max_lost", 20);
  MBLOG_INFO << "yolo_track_post: net=" << net_w_ << "x" << net_h_
             << " nc=" << num_classes_ << " conf=" << conf_threshold_
             << " iou=" << iou_threshold_ << " track_iou=" << track_iou_
             << " max_lost=" << max_lost_;
  return modelbox::STATUS_OK;
}

modelbox::Status YoloTrackPostFlowUnit::Close() { return modelbox::STATUS_OK; }

std::vector<YoloTrackPostFlowUnit::Detection> YoloTrackPostFlowUnit::Decode(
    const float *feat, int channels, int num_anchors, float scale_x,
    float scale_y) const {
  std::vector<Detection> out;
  out.reserve(64);
  const int nc = num_classes_;
  for (int n = 0; n < num_anchors; ++n) {
    int best = 0;
    float best_score = feat[4 * num_anchors + n];
    for (int c = 1; c < nc; ++c) {
      float s = feat[(4 + c) * num_anchors + n];
      if (s > best_score) {
        best_score = s;
        best = c;
      }
    }
    if (best_score < conf_threshold_) continue;
    Detection d;
    float cx = feat[0 * num_anchors + n];
    float cy = feat[1 * num_anchors + n];
    float w = feat[2 * num_anchors + n];
    float h = feat[3 * num_anchors + n];
    d.x1 = (cx - w / 2.0F) * scale_x;
    d.y1 = (cy - h / 2.0F) * scale_y;
    d.x2 = (cx + w / 2.0F) * scale_x;
    d.y2 = (cy + h / 2.0F) * scale_y;
    d.score = best_score;
    d.label = best;
    out.push_back(d);
  }
  return out;
}

std::vector<YoloTrackPostFlowUnit::Detection> YoloTrackPostFlowUnit::Nms(
    std::vector<Detection> dets) const {
  std::sort(dets.begin(), dets.end(),
            [](const Detection &a, const Detection &b) { return a.score > b.score; });
  std::vector<Detection> kept;
  std::vector<bool> sup(dets.size(), false);
  for (size_t i = 0; i < dets.size(); ++i) {
    if (sup[i]) continue;
    kept.push_back(dets[i]);
    const auto &a = dets[i];
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (sup[j]) continue;
      const auto &b = dets[j];
      if (BoxIoU(a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2) >
          iou_threshold_) {
        sup[j] = true;
      }
    }
  }
  return kept;
}

void YoloTrackPostFlowUnit::UpdateTracks(const std::vector<Detection> &dets,
                                         std::vector<Track> &tracks,
                                         int &next_id) const {
  // Greedy match: for each track (highest-confidence first by current score),
  // find the unassigned detection with the highest IoU >= track_iou_; mark
  // both as assigned. Birth new tracks for unmatched detections; age out
  // unmatched tracks that exceed max_lost_.
  std::vector<bool> det_used(dets.size(), false);
  std::sort(tracks.begin(), tracks.end(), [](const Track &a, const Track &b) {
    return a.score > b.score;
  });

  for (auto &t : tracks) {
    int best_j = -1;
    float best_iou = track_iou_;
    for (size_t j = 0; j < dets.size(); ++j) {
      if (det_used[j]) continue;
      if (dets[j].label != t.label) continue;
      float iou = BoxIoU(t.x1, t.y1, t.x2, t.y2, dets[j].x1, dets[j].y1,
                         dets[j].x2, dets[j].y2);
      if (iou > best_iou) {
        best_iou = iou;
        best_j = static_cast<int>(j);
      }
    }
    if (best_j >= 0) {
      const auto &d = dets[best_j];
      t.x1 = d.x1;
      t.y1 = d.y1;
      t.x2 = d.x2;
      t.y2 = d.y2;
      t.score = d.score;
      t.hits++;
      t.lost = 0;
      det_used[best_j] = true;
    } else {
      t.lost++;
    }
  }

  // Drop tracks lost too long.
  tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
                              [&](const Track &t) { return t.lost > max_lost_; }),
               tracks.end());

  // Birth new tracks for unmatched detections.
  for (size_t j = 0; j < dets.size(); ++j) {
    if (det_used[j]) continue;
    Track t;
    t.id = next_id++;
    t.label = dets[j].label;
    t.score = dets[j].score;
    t.x1 = dets[j].x1;
    t.y1 = dets[j].y1;
    t.x2 = dets[j].x2;
    t.y2 = dets[j].y2;
    t.hits = 1;
    t.lost = 0;
    tracks.push_back(t);
  }
}

void YoloTrackPostFlowUnit::DrawTracks(cv::Mat &img,
                                       const std::vector<Track> &tracks) const {
  for (const auto &t : tracks) {
    if (t.lost > 0) continue;  // only draw confirmed-this-frame tracks
    int x1 = std::max(0, std::min(img.cols - 1, static_cast<int>(t.x1)));
    int y1 = std::max(0, std::min(img.rows - 1, static_cast<int>(t.y1)));
    int x2 = std::max(0, std::min(img.cols - 1, static_cast<int>(t.x2)));
    int y2 = std::max(0, std::min(img.rows - 1, static_cast<int>(t.y2)));
    // Color is keyed on track id, not class -- visually following an ID is
    // the whole point of tracking.
    const cv::Scalar &color = PaletteColor(t.id);
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    char label[40];
    snprintf(label, sizeof(label), "ID%d c%d %.2f", t.id, t.label, t.score);
    cv::putText(img, label, cv::Point(x1, std::max(0, y1 - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
  }
}

modelbox::Status YoloTrackPostFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto in_image = data_ctx->Input("in_image");
  auto in_feat = data_ctx->Input("in_feat");
  auto out_image = data_ctx->Output("out_data");
  if (!in_image || !in_feat || !out_image) {
    return {modelbox::STATUS_FAULT, "yolo_track_post: missing port"};
  }
  if (in_image->Size() != in_feat->Size()) {
    return {modelbox::STATUS_FAULT, "yolo_track_post: size mismatch"};
  }
  if (in_image->Size() == 0) return modelbox::STATUS_OK;

  // Per-session tracker state, kept on the data_ctx so multiple concurrent
  // streams don't clobber each other.
  auto state = std::static_pointer_cast<TrackState>(
      data_ctx->GetPrivate(kTrackState));
  if (state == nullptr) {
    state = std::make_shared<TrackState>();
    data_ctx->SetPrivate(kTrackState, state);
  }

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
      return {modelbox::STATUS_FAULT, "yolo_track_post: bad image meta"};
    }

    const int channels = 4 + num_classes_;
    const auto feat_bytes = feat_buf->GetBytes();
    if (feat_bytes % (sizeof(float) * channels) != 0) {
      return {modelbox::STATUS_FAULT, "yolo_track_post: bad feat size"};
    }
    const int num_anchors =
        static_cast<int>(feat_bytes / (sizeof(float) * channels));
    const auto *feat = static_cast<const float *>(feat_buf->ConstData());
    const float scale_x = static_cast<float>(width) / static_cast<float>(net_w_);
    const float scale_y = static_cast<float>(height) / static_cast<float>(net_h_);

    auto raw = Decode(feat, channels, num_anchors, scale_x, scale_y);
    auto kept = Nms(std::move(raw));
    UpdateTracks(kept, state->tracks, state->next_id);

    auto out_buf = out_image->At(i);
    auto *dst = static_cast<uint8_t *>(out_buf->MutableData());
    const auto *src = static_cast<const uint8_t *>(img_buf->ConstData());
    if (memcpy_s(dst, img_buf->GetBytes(), src, img_buf->GetBytes()) != EOK) {
      return {modelbox::STATUS_FAULT, "yolo_track_post: image copy failed"};
    }
    cv::Mat out_mat(height, width, CV_8UC3, dst);
    DrawTracks(out_mat, state->tracks);
    out_buf->CopyMeta(img_buf);
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(YoloTrackPostFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"in_image"});
  desc.AddFlowUnitInput({"in_feat"});
  desc.AddFlowUnitOutput({"out_data"});
  desc.SetFlowType(modelbox::STREAM);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "net_h", "int", true, "640", "network input height"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "net_w", "int", true, "640", "network input width"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "num_classes", "int", true, "80", "number of class channels"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "conf_threshold", "float", true, "0.35", "minimum class score"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "iou_threshold", "float", true, "0.45", "NMS IoU threshold"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "track_iou", "float", true, "0.3",
      "minimum IoU to associate detection to existing track"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "max_lost", "int", true, "20",
      "frames a track may go unmatched before being dropped"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
