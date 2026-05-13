/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "hit_trigger_flowunit.h"

#include <cstdint>
#include <thread>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

modelbox::Status HitTriggerFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  // Spawn a background thread that fires exactly one buffer down the pipe via
  // the modelbox external-data path. Same idiom as video_input but with no
  // OpenCV dependency.
  auto fire = [this]() {
    auto ext = this->CreateExternalData();
    if (!ext) {
      MBLOG_ERROR << "hit_trigger: CreateExternalData returned null";
      return;
    }
    auto buf_list = ext->CreateBufferList();
    auto build = buf_list->Build({1});
    if (!build) {
      MBLOG_ERROR << "hit_trigger: BufferList::Build failed: "
                  << build.WrapErrormsgs();
      return;
    }
    auto buf = buf_list->At(0);
    *static_cast<uint8_t *>(buf->MutableData()) = 1;
    auto send = ext->Send(buf_list);
    if (!send) {
      MBLOG_ERROR << "hit_trigger: Send failed: " << send.WrapErrormsgs();
    }
    auto close = ext->Close();
    if (!close) {
      MBLOG_ERROR << "hit_trigger: Close failed: " << close.WrapErrormsgs();
    }
  };
  std::thread t(fire);
  t.detach();
  return modelbox::STATUS_OK;
}

modelbox::Status HitTriggerFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status HitTriggerFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto out = data_ctx->Output("out");
  out->Build({1});
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(HitTriggerFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Generic");
  desc.AddFlowUnitOutput({"out"});
  desc.SetFlowType(modelbox::NORMAL);
  desc.SetDescription(FLOWUNIT_DESC);
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
