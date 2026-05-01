/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef MODELBOX_FLOWUNIT_YOLO_TRACK_POST_CPU_H_
#define MODELBOX_FLOWUNIT_YOLO_TRACK_POST_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <opencv2/opencv.hpp>
#include <vector>

#include "modelbox/buffer.h"

constexpr const char *FLOWUNIT_NAME = "yolo_track_post";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Anchor-free YOLO detection + greedy-IoU SORT tracker.\n"
    "\t@Inputs: in_image (BGR uint8) + in_feat (float [B, 4+nc, N]).\n"
    "\t@Outputs: out_data (BGR uint8) with per-track stable-ID boxes drawn.\n"
    "\t  Detections are passed through the same anchor-free decode + NMS as\n"
    "\t  yolo26_post; matching to existing tracks is greedy by IoU; tracks\n"
    "\t  unmatched for max_lost frames are dropped.\n";

class YoloTrackPostFlowUnit : public modelbox::FlowUnit {
 public:
  YoloTrackPostFlowUnit() = default;
  ~YoloTrackPostFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

  // Public so the per-stream state struct (in anonymous namespace) can hold
  // a vector of these.
  struct Detection {
    float x1, y1, x2, y2;
    float score;
    int label;
  };

  struct Track {
    int id;
    int label;
    float score;
    float x1, y1, x2, y2;
    int hits;
    int lost;
  };

 private:
  std::vector<Detection> Decode(const float *feat, int channels, int num_anchors,
                                float scale_x, float scale_y) const;
  std::vector<Detection> Nms(std::vector<Detection> dets) const;
  void UpdateTracks(const std::vector<Detection> &dets,
                    std::vector<Track> &tracks, int &next_id) const;
  void DrawTracks(cv::Mat &img, const std::vector<Track> &tracks) const;

  int net_h_{640};
  int net_w_{640};
  int num_classes_{80};
  float conf_threshold_{0.35F};
  float iou_threshold_{0.45F};
  float track_iou_{0.3F};
  int max_lost_{20};
};

#endif  // MODELBOX_FLOWUNIT_YOLO_TRACK_POST_CPU_H_
