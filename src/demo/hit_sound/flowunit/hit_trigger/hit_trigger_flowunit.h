/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef MODELBOX_FLOWUNIT_HIT_TRIGGER_CPU_H_
#define MODELBOX_FLOWUNIT_HIT_TRIGGER_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

constexpr const char *FLOWUNIT_NAME = "hit_trigger";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: One-shot trigger flowunit. Emits a single 1-byte buffer on "
    "'out' via CreateExternalData() so a downstream expand flowunit can fire "
    "exactly once. Replaces video_input in opencv-free pipelines.\n";

class HitTriggerFlowUnit : public modelbox::FlowUnit {
 public:
  HitTriggerFlowUnit() = default;
  ~HitTriggerFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;
};

#endif  // MODELBOX_FLOWUNIT_HIT_TRIGGER_CPU_H_
