/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef MODELBOX_FLOWUNIT_TRACKNET_POST_CPU_H_
#define MODELBOX_FLOWUNIT_TRACKNET_POST_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/flowunit.h>

#include <opencv2/opencv.hpp>

#include <string>

constexpr const char *FLOWUNIT_NAME = "tracknet_post";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: TrackNet heatmap post-process: peak -> centroid -> draw circle.\n"
    "\t@Inputs:  heatmaps (float CHW 3x288x512), source_frame (uint8 BGR HxWx3)\n"
    "\t@Outputs: frame_out (uint8 BGR HxWx3) with red ball circle\n";

extern "C" bool TrackNetExtractCentroid(const float *heatmap, int h, int w,
                                        float score_thr, float mask_ratio,
                                        float *cx_out, float *cy_out,
                                        float *peak_out);

class TrackNetPostFlowUnit : public modelbox::FlowUnit {
 public:
  TrackNetPostFlowUnit() = default;
  ~TrackNetPostFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  int net_h_{288};
  int net_w_{512};
  int heatmap_channel_{2};
  float score_thr_{0.3f};
  float mask_ratio_{0.5f};
  int circle_radius_{6};
  cv::Scalar circle_color_{0, 0, 255};  // BGR red
  bool draw_heatmap_{false};
};

#endif  // MODELBOX_FLOWUNIT_TRACKNET_POST_CPU_H_
