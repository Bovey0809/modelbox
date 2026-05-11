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

#ifndef MODELBOX_FLOWUNIT_PADDLE_OCR_DRAW_CPU_H_
#define MODELBOX_FLOWUNIT_PADDLE_OCR_DRAW_CPU_H_

#include <modelbox/flowunit.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

constexpr const char *FLOWUNIT_NAME = "paddle_ocr_draw";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Render OCR polygons + text on frames, joining per frame_id.\n"
    "\t@Inputs: in_image (uint8 BGR with frame_id + box_count meta),\n"
    "\t  in_result (uint8 dummy with text/score/frame_id/box_id/box_count/polygon meta).\n"
    "\t@Outputs: out_image (uint8 BGR annotated).\n";

class PaddleOcrDrawFlowUnit : public modelbox::FlowUnit {
 public:
  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> ctx) override;

 private:
  struct ResultEntry {
    std::array<cv::Point2f, 4> polygon{};
    std::string text;
    float score{0.0F};
  };
  struct FrameEntry {
    int32_t width{0};
    int32_t height{0};
    int32_t channel{3};
    int32_t box_count{0};
    std::vector<uint8_t> bytes;
  };

  // Renders one frame (with its joined results) into out[idx].
  modelbox::Status Emit(const std::shared_ptr<modelbox::BufferList> &out,
                        size_t idx, const FrameEntry &frame,
                        const std::vector<ResultEntry> &results);

  std::mutex mu_;
  std::unordered_map<int64_t, FrameEntry> frames_;
  std::unordered_map<int64_t, std::vector<ResultEntry>> pending_;
  float text_scale_{0.6F};
  int line_thickness_{2};
};

#endif  // MODELBOX_FLOWUNIT_PADDLE_OCR_DRAW_CPU_H_
