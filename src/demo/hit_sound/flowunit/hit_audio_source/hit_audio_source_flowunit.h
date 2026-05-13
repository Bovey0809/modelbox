/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef MODELBOX_FLOWUNIT_HIT_AUDIO_SOURCE_CPU_H_
#define MODELBOX_FLOWUNIT_HIT_AUDIO_SOURCE_CPU_H_

#include <modelbox/base/device.h>
#include <modelbox/base/status.h>
#include <modelbox/flow.h>
#include <modelbox/flowunit.h>

#include <string>

constexpr const char *FLOWUNIT_NAME = "hit_audio_source";
constexpr const char *FLOWUNIT_TYPE = "cpu";
constexpr const char *FLOWUNIT_DESC =
    "\n\t@Brief: Source flowunit for the hit_sound pipeline. Reads a binary "
    "file of precomputed mel-spectrogram windows produced by "
    "scripts/precompute_mels.py and emits one float32 (1,1,n_mels,T) buffer "
    "per window on 'mel'.\n";

class HitAudioSourceFlowUnit : public modelbox::FlowUnit {
 public:
  HitAudioSourceFlowUnit() = default;
  ~HitAudioSourceFlowUnit() override = default;

  modelbox::Status Open(
      const std::shared_ptr<modelbox::Configuration> &opts) override;
  modelbox::Status Close() override;
  modelbox::Status Process(
      std::shared_ptr<modelbox::DataContext> data_ctx) override;

 private:
  std::string mel_bin_path_;
};

#endif  // MODELBOX_FLOWUNIT_HIT_AUDIO_SOURCE_CPU_H_
