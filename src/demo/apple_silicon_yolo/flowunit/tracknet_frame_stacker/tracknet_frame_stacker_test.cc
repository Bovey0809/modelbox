/*
 * Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <gtest/gtest.h>

#include <vector>

extern "C" void TrackNetStackFrames(const float *cur, const float *prev1,
                                    const float *prev2, bool have_prev1,
                                    bool have_prev2, size_t per_frame,
                                    float *out);

namespace {

constexpr size_t H = 2;
constexpr size_t W = 2;
constexpr size_t C = 3;
constexpr size_t PER = C * H * W;  // 12

std::vector<float> Filled(float v) { return std::vector<float>(PER, v); }

}  // namespace

TEST(StackFrames, FirstFrameDuplicatesCurrent) {
  auto cur = Filled(7.f);
  std::vector<float> out(3 * PER, -1.f);
  TrackNetStackFrames(cur.data(), nullptr, nullptr, false, false, PER,
                      out.data());
  for (size_t i = 0; i < 3 * PER; ++i) EXPECT_EQ(out[i], 7.f) << "i=" << i;
}

TEST(StackFrames, SecondFrameUsesPrev1Twice) {
  auto cur = Filled(2.f);
  auto prev1 = Filled(1.f);
  std::vector<float> out(3 * PER, -1.f);
  TrackNetStackFrames(cur.data(), prev1.data(), nullptr, true, false, PER,
                      out.data());
  for (size_t i = 0; i < PER; ++i) EXPECT_EQ(out[i], 1.f);
  for (size_t i = PER; i < 2 * PER; ++i) EXPECT_EQ(out[i], 1.f);
  for (size_t i = 2 * PER; i < 3 * PER; ++i) EXPECT_EQ(out[i], 2.f);
}

TEST(StackFrames, ThirdAndAfterUsePrev2Prev1Cur) {
  auto cur = Filled(3.f);
  auto prev1 = Filled(2.f);
  auto prev2 = Filled(1.f);
  std::vector<float> out(3 * PER, -1.f);
  TrackNetStackFrames(cur.data(), prev1.data(), prev2.data(), true, true, PER,
                      out.data());
  for (size_t i = 0; i < PER; ++i) EXPECT_EQ(out[i], 1.f);
  for (size_t i = PER; i < 2 * PER; ++i) EXPECT_EQ(out[i], 2.f);
  for (size_t i = 2 * PER; i < 3 * PER; ++i) EXPECT_EQ(out[i], 3.f);
}
