/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef MODELBOX_FLOWUNIT_HIT_INTERVALS_SINK_CPU_H_
#define MODELBOX_FLOWUNIT_HIT_INTERVALS_SINK_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flowunit.h>

#include <memory>
#include <string>
#include <vector>

constexpr const char *FLOWUNIT_NAME = "hit_intervals_sink";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Sink flowunit. Consumes the per-window (1,2) logit stream "
    "from hit_sound_infer_coreml, applies majority-vote smoothing + "
    "duration filtering + buffered-rally detection, and writes the resulting "
    "hit/rally intervals to CSV.\n";

class HitIntervalsSinkFlowUnit : public modelbox::FlowUnit {
 public:
  HitIntervalsSinkFlowUnit() = default;
  ~HitIntervalsSinkFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;
  modelbox::Status DataPre(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;
  modelbox::Status DataPost(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  struct StreamState;
  std::shared_ptr<StreamState> GetOrCreateState(
      std::shared_ptr<modelbox::DataContext> data_ctx);
  modelbox::Status FlushStream(StreamState &state);

  float step_sec_{0.02f};
  float window_sec_{0.5f};
  float threshold_{0.5f};
  int smooth_win_{5};
  float max_hit_dur_{1.0f};
  float max_gap_{3.0f};
  float buffer_time_{4.0f};
  std::string interval_mode_{"hit"};   // hit | rally
  std::string intervals_csv_{};
  std::string summary_json_{};
};

#endif  // MODELBOX_FLOWUNIT_HIT_INTERVALS_SINK_CPU_H_
