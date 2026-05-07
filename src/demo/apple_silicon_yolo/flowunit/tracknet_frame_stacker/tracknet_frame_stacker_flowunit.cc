/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "tracknet_frame_stacker_flowunit.h"

#include <cstring>
#include <memory>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

namespace {

struct StreamState {
  std::vector<float> prev1;
  std::vector<float> prev2;
  bool have_prev1 = false;
  bool have_prev2 = false;
};

constexpr const char *kStateKey = "tracknet_frame_stacker_state";

}  // namespace

extern "C" void TrackNetStackFrames(const float *cur, const float *prev1,
                                    const float *prev2, bool have_prev1,
                                    bool have_prev2, size_t per_frame,
                                    float *out) {
  const float *slot0;
  const float *slot1;
  if (have_prev2 && have_prev1) {
    slot0 = prev2;
    slot1 = prev1;
  } else if (have_prev1) {
    slot0 = prev1;
    slot1 = prev1;
  } else {
    slot0 = cur;
    slot1 = cur;
  }
  std::memcpy(out + 0 * per_frame, slot0, per_frame * sizeof(float));
  std::memcpy(out + 1 * per_frame, slot1, per_frame * sizeof(float));
  std::memcpy(out + 2 * per_frame, cur, per_frame * sizeof(float));
}

modelbox::Status TrackNetFrameStackerFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  per_frame_floats_ = static_cast<size_t>(
      opts->GetUint64("per_frame_floats", 3 * 288 * 512));
  return modelbox::STATUS_OK;
}

modelbox::Status TrackNetFrameStackerFlowUnit::Close() {
  return modelbox::STATUS_OK;
}

modelbox::Status TrackNetFrameStackerFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto state = std::make_shared<StreamState>();
  state->prev1.resize(per_frame_floats_);
  state->prev2.resize(per_frame_floats_);
  data_ctx->SetPrivate(kStateKey, state);
  return modelbox::STATUS_OK;
}

modelbox::Status TrackNetFrameStackerFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto state = std::static_pointer_cast<StreamState>(
      data_ctx->GetPrivate(kStateKey));
  if (!state) {
    return {modelbox::STATUS_FAULT, "stream state missing"};
  }

  auto in_list = data_ctx->Input("frame");
  auto out_list = data_ctx->Output("stacked");
  if (!in_list || !out_list) {
    return {modelbox::STATUS_FAULT, "ports missing"};
  }

  const size_t per = per_frame_floats_;
  const size_t per_bytes = per * sizeof(float);

  std::vector<size_t> shapes(in_list->Size(), 3 * per_bytes);
  auto build = out_list->Build(shapes);
  if (!build) return build;

  for (size_t i = 0; i < in_list->Size(); ++i) {
    auto in = in_list->At(i);
    if (in->GetBytes() < per_bytes) {
      return {modelbox::STATUS_FAULT, "frame buffer too small"};
    }
    const auto *cur = static_cast<const float *>(in->ConstData());

    auto out = out_list->At(i);
    auto *dst = static_cast<float *>(out->MutableData());

    TrackNetStackFrames(cur,
                        state->have_prev1 ? state->prev1.data() : nullptr,
                        state->have_prev2 ? state->prev2.data() : nullptr,
                        state->have_prev1, state->have_prev2, per, dst);

    // Carry source-resolution metadata through if upstream attached any.
    out->CopyMeta(in);

    // Shift state.
    if (state->have_prev1) {
      std::memcpy(state->prev2.data(), state->prev1.data(), per_bytes);
      state->have_prev2 = true;
    }
    std::memcpy(state->prev1.data(), cur, per_bytes);
    state->have_prev1 = true;
  }

  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(TrackNetFrameStackerFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Image");
  desc.AddFlowUnitInput({"frame"});
  desc.AddFlowUnitOutput({"stacked"});
  desc.SetFlowType(modelbox::STREAM);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "per_frame_floats", "int", true, "442368",
      "floats per single CHW frame (default 3*288*512)"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
