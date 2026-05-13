/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "hit_audio_source_flowunit.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

namespace {

// File layout (little-endian):
//   bytes  0..3   : magic "MELS"
//   bytes  4..7   : uint32 count        (# windows)
//   bytes  8..11  : uint32 mel_h        (typically 128)
//   bytes 12..15  : uint32 mel_w        (typically 48)
//   bytes 16..    : count * mel_h * mel_w * float32  (row-major, C-contiguous)
constexpr const char *kMagic = "MELS";

struct MelBinHeader {
  char magic[4];
  uint32_t count;
  uint32_t h;
  uint32_t w;
};

}  // namespace

modelbox::Status HitAudioSourceFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  mel_bin_path_ = opts->GetString("mel_bin_path");
  if (mel_bin_path_.empty()) {
    return {modelbox::STATUS_BADCONF,
            "hit_audio_source: 'mel_bin_path' is required"};
  }
  return modelbox::STATUS_OK;
}

modelbox::Status HitAudioSourceFlowUnit::Close() { return modelbox::STATUS_OK; }

modelbox::Status HitAudioSourceFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  // Expand: each trigger input expands into N mel windows on `mel`.
  auto in = data_ctx->Input("trigger");
  auto out = data_ctx->Output("mel");
  if (in == nullptr || in->Size() == 0) {
    return modelbox::STATUS_OK;
  }

  std::ifstream f(mel_bin_path_, std::ios::binary);
  if (!f) {
    return {modelbox::STATUS_FAULT,
            "hit_audio_source: cannot open mel_bin_path: " + mel_bin_path_};
  }
  MelBinHeader hdr{};
  f.read(reinterpret_cast<char *>(&hdr), sizeof(hdr));
  if (f.gcount() != static_cast<std::streamsize>(sizeof(hdr)) ||
      std::memcmp(hdr.magic, kMagic, 4) != 0) {
    return {modelbox::STATUS_FAULT,
            "hit_audio_source: bad header in " + mel_bin_path_};
  }
  const size_t per_floats = static_cast<size_t>(hdr.h) * hdr.w;
  const size_t per_bytes = per_floats * sizeof(float);
  const uint32_t count = hdr.count;

  MBLOG_INFO << "hit_audio_source: expanding into " << count << " windows of ("
             << "1,1," << hdr.h << "," << hdr.w << ") from " << mel_bin_path_;

  std::vector<size_t> shapes(count, per_bytes);
  auto build = out->Build(shapes);
  if (!build) {
    return {modelbox::STATUS_FAULT,
            "hit_audio_source: BufferList::Build failed: " +
                build.WrapErrormsgs()};
  }
  std::vector<float> chunk(per_floats);
  for (uint32_t i = 0; i < count; ++i) {
    f.read(reinterpret_cast<char *>(chunk.data()),
           static_cast<std::streamsize>(per_bytes));
    if (f.gcount() != static_cast<std::streamsize>(per_bytes)) {
      return {modelbox::STATUS_FAULT,
              "hit_audio_source: short read on window " + std::to_string(i)};
    }
    auto buf = out->At(i);
    std::memcpy(buf->MutableData(), chunk.data(), per_bytes);
    buf->Set("index", static_cast<int32_t>(i));
  }
  return modelbox::STATUS_OK;
}

MODELBOX_FLOWUNIT(HitAudioSourceFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Audio");
  desc.AddFlowUnitInput({"trigger"});
  desc.AddFlowUnitOutput({"mel"});
  desc.SetOutputType(modelbox::EXPAND);
  desc.SetFlowType(modelbox::STREAM);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "mel_bin_path", "string", true, "",
      "path to a MELS-magic binary of precomputed mel windows"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
