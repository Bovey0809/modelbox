/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "hit_intervals_sink_flowunit.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

#include "modelbox/flowunit.h"
#include "modelbox/flowunit_api_helper.h"

namespace {

// IEEE 754 half-precision decode. CoreML may produce fp16 output tensors when
// the model was converted with compute_precision=FLOAT16 (our case).
float DecodeHalf(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1u;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t f = 0;
  if (exp == 0) {
    if (mant == 0) {
      f = sign << 31;
    } else {
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      ++exp;
      mant &= ~0x400u;
      f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f = (sign << 31) | 0x7F800000u | (mant << 13);
  } else {
    f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  }
  float out = 0.f;
  std::memcpy(&out, &f, 4);
  return out;
}

float SoftmaxPos(float a, float b) {
  float m = std::max(a, b);
  float ea = std::exp(a - m);
  float eb = std::exp(b - m);
  return eb / (ea + eb);
}

}  // namespace

struct HitIntervalsSinkFlowUnit::StreamState {
  std::deque<int> smooth_buf;
  int smooth_win{5};
  float run_start{-1.f};
  float last_hit{-1.f};
  bool in_rally{false};
  std::vector<std::pair<float, float>> pos_windows;
  std::vector<std::pair<float, int>> rally_states;
  size_t window_count{0};
};

modelbox::Status HitIntervalsSinkFlowUnit::Open(
    const std::shared_ptr<modelbox::Configuration> &opts) {
  step_sec_ = opts->GetFloat("step_sec", step_sec_);
  window_sec_ = opts->GetFloat("window_sec", window_sec_);
  threshold_ = opts->GetFloat("threshold", threshold_);
  smooth_win_ = opts->GetInt32("smooth_win", smooth_win_);
  max_hit_dur_ = opts->GetFloat("max_hit_dur", max_hit_dur_);
  max_gap_ = opts->GetFloat("max_gap", max_gap_);
  buffer_time_ = opts->GetFloat("buffer_time", buffer_time_);
  interval_mode_ = opts->GetString("interval_mode", "hit");
  intervals_csv_ = opts->GetString("intervals_csv", "");
  summary_json_ = opts->GetString("summary_json", "");
  if (intervals_csv_.empty()) {
    return {modelbox::STATUS_BADCONF,
            "hit_intervals_sink: 'intervals_csv' is required"};
  }
  return modelbox::STATUS_OK;
}

modelbox::Status HitIntervalsSinkFlowUnit::Close() {
  return modelbox::STATUS_OK;
}

std::shared_ptr<HitIntervalsSinkFlowUnit::StreamState>
HitIntervalsSinkFlowUnit::GetOrCreateState(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto priv = data_ctx->GetPrivate("state");
  if (priv != nullptr) {
    return std::static_pointer_cast<StreamState>(priv);
  }
  auto state = std::make_shared<StreamState>();
  state->smooth_win = smooth_win_;
  data_ctx->SetPrivate("state", state);
  return state;
}

modelbox::Status HitIntervalsSinkFlowUnit::DataPre(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  GetOrCreateState(data_ctx);
  return modelbox::STATUS_OK;
}

modelbox::Status HitIntervalsSinkFlowUnit::Process(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto state = GetOrCreateState(data_ctx);
  auto in = data_ctx->Input("logits");
  if (in == nullptr) {
    return modelbox::STATUS_OK;
  }
  const size_t n = in->Size();
  for (size_t i = 0; i < n; ++i) {
    auto buf = in->At(i);
    size_t bytes = buf->GetBytes();
    const void *data = buf->ConstData();
    float lo0 = 0.f;
    float lo1 = 0.f;
    if (bytes == 2 * sizeof(float)) {
      const auto *f = static_cast<const float *>(data);
      lo0 = f[0];
      lo1 = f[1];
    } else if (bytes == 2 * sizeof(uint16_t)) {
      const auto *h = static_cast<const uint16_t *>(data);
      lo0 = DecodeHalf(h[0]);
      lo1 = DecodeHalf(h[1]);
    } else {
      MBLOG_ERROR << "hit_intervals_sink: unexpected logit buffer size "
                  << bytes;
      return {modelbox::STATUS_FAULT, "bad logit buffer size"};
    }
    float prob = SoftmaxPos(lo0, lo1);
    float center =
        static_cast<float>(state->window_count) * step_sec_ + window_sec_ / 2.f;
    int pred_raw = prob >= threshold_ ? 1 : 0;

    // Majority-vote smoothing.
    state->smooth_buf.push_back(pred_raw);
    if (static_cast<int>(state->smooth_buf.size()) > state->smooth_win) {
      state->smooth_buf.pop_front();
    }
    int sum = 0;
    for (int v : state->smooth_buf) sum += v;
    int pred_smooth =
        sum * 2 > static_cast<int>(state->smooth_buf.size()) ? 1 : 0;

    // Duration filter.
    int pred_filt = 0;
    if (pred_smooth == 1) {
      if (state->run_start < 0) state->run_start = center;
      pred_filt = (center - state->run_start) > max_hit_dur_ ? 0 : 1;
    } else {
      state->run_start = -1.f;
      pred_filt = 0;
    }

    // Rally detector.
    if (pred_filt == 1) {
      state->last_hit = center;
      state->in_rally = true;
    } else if (state->in_rally && state->last_hit >= 0 &&
               (center - state->last_hit) > max_gap_) {
      state->in_rally = false;
    }
    state->rally_states.emplace_back(center, state->in_rally ? 1 : 0);

    if (pred_filt == 1) {
      float start = center - window_sec_ / 2.f;
      state->pos_windows.emplace_back(start, start + window_sec_);
    }
    ++state->window_count;
  }
  return modelbox::STATUS_OK;
}

modelbox::Status HitIntervalsSinkFlowUnit::FlushStream(StreamState &state) {
  // Merge near-adjacent positive windows into hit intervals.
  std::vector<std::pair<float, float>> hits;
  auto pos = state.pos_windows;
  std::sort(pos.begin(), pos.end());
  for (const auto &w : pos) {
    if (!hits.empty() && w.first - hits.back().second <= step_sec_ + 1e-6f) {
      hits.back().second = std::max(hits.back().second, w.second);
    } else {
      hits.push_back(w);
    }
  }

  // Rally intervals from state trace.
  std::vector<std::pair<float, float>> rallies;
  float start_t = -1.f;
  float prev_t = -1.f;
  for (const auto &kv : state.rally_states) {
    float t = kv.first;
    int s = kv.second;
    if (s == 1 && start_t < 0) {
      start_t = t;
    } else if (s == 0 && start_t >= 0) {
      float end = (prev_t >= 0 ? prev_t : t) + step_sec_;
      rallies.emplace_back(start_t, end);
      start_t = -1.f;
    }
    prev_t = t;
  }
  if (start_t >= 0 && prev_t >= 0) {
    rallies.emplace_back(start_t, prev_t + step_sec_);
  }

  const auto &selected =
      (interval_mode_ == "rally" || interval_mode_ == "both") ? rallies : hits;

  {
    std::ofstream out(intervals_csv_);
    if (!out) {
      return {modelbox::STATUS_FAULT,
              "hit_intervals_sink: cannot open intervals_csv: " +
                  intervals_csv_};
    }
    for (size_t i = 0; i < selected.size(); ++i) {
      if (i) out << "\n";
      out << selected[i].first << "," << selected[i].second;
    }
  }
  if (!summary_json_.empty()) {
    std::ofstream out(summary_json_);
    if (out) {
      out << "{\"windows\":" << state.window_count
          << ",\"hits\":" << hits.size()
          << ",\"rallies\":" << rallies.size()
          << ",\"mode\":\"" << interval_mode_ << "\""
          << ",\"intervals_csv\":\"" << intervals_csv_ << "\"}\n";
    }
  }
  MBLOG_INFO << "hit_intervals_sink: windows=" << state.window_count
             << " hits=" << hits.size() << " rallies=" << rallies.size()
             << " -> " << intervals_csv_;
  return modelbox::STATUS_OK;
}

modelbox::Status HitIntervalsSinkFlowUnit::DataPost(
    std::shared_ptr<modelbox::DataContext> data_ctx) {
  auto state = GetOrCreateState(data_ctx);
  if (state->window_count == 0) return modelbox::STATUS_OK;
  return FlushStream(*state);
}

MODELBOX_FLOWUNIT(HitIntervalsSinkFlowUnit, desc) {
  desc.SetFlowUnitName(FLOWUNIT_NAME);
  desc.SetFlowUnitGroupType("Audio");
  desc.AddFlowUnitInput({"logits"});
  desc.SetCollapseAll(true);
  desc.SetOutputType(modelbox::COLLAPSE);
  desc.SetFlowType(modelbox::STREAM);
  desc.SetDescription(FLOWUNIT_DESC);
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "step_sec", "float", true, "0.02", "sliding step in seconds"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "window_sec", "float", true, "0.5", "window length in seconds"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "threshold", "float", true, "0.5", "probability threshold"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "smooth_win", "int", true, "5", "majority-vote smoothing window"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "max_hit_dur", "float", true, "1.0",
      "drop continuous positive runs longer than this"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "max_gap", "float", true, "3.0",
      "max silence within a rally before it closes"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "buffer_time", "float", true, "4.0",
      "rally buffer time (must be > max_gap)"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "interval_mode", "string", true, "hit", "hit | rally | both"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "intervals_csv", "string", true, "", "output csv path"));
  desc.AddFlowUnitOption(modelbox::FlowUnitOption(
      "summary_json", "string", true, "", "optional summary json path"));
}

MODELBOX_DRIVER_FLOWUNIT(desc) {
  desc.Desc.SetName(FLOWUNIT_NAME);
  desc.Desc.SetClass(modelbox::DRIVER_CLASS_FLOWUNIT);
  desc.Desc.SetType(FLOWUNIT_TYPE);
  desc.Desc.SetDescription(FLOWUNIT_DESC);
  desc.Desc.SetVersion("1.0.0");
}
