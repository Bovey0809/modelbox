/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef MODELBOX_FLOWUNIT_TRACKNET_FRAME_STACKER_CPU_H_
#define MODELBOX_FLOWUNIT_TRACKNET_FRAME_STACKER_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/buffer.h>
#include <modelbox/flowunit.h>

#include <cstddef>
#include <vector>

constexpr const char *FLOWUNIT_NAME = "tracknet_frame_stacker";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Sliding 3-frame stacker for TrackNet.\n"
    "\t@Inputs:  frame  (float CHW 3x288x512)\n"
    "\t@Outputs: stacked (float CHW 9x288x512), concat[t-2,t-1,t]\n"
    "\t  First two frames pad by repeating the available frames.\n";

// Pure helper -- C linkage so tests link without C++ name mangling.
extern "C" void TrackNetStackFrames(const float *cur, const float *prev1,
                                    const float *prev2, bool have_prev1,
                                    bool have_prev2, size_t per_frame,
                                    float *out);

class TrackNetFrameStackerFlowUnit : public modelbox::FlowUnit {
 public:
  TrackNetFrameStackerFlowUnit() = default;
  ~TrackNetFrameStackerFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;
  modelbox::Status DataPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  size_t per_frame_floats_{3 * 288 * 512};
};

#endif  // MODELBOX_FLOWUNIT_TRACKNET_FRAME_STACKER_CPU_H_
