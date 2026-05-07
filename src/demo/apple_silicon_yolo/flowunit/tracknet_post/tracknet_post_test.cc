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

#include <cmath>
#include <vector>

extern "C" bool TrackNetExtractCentroid(const float *heatmap, int h, int w,
                                        float score_thr, float mask_ratio,
                                        float *cx_out, float *cy_out,
                                        float *peak_out);

namespace {

std::vector<float> GaussianHeatmap(int h, int w, float cx, float cy,
                                   float sigma) {
  std::vector<float> hm(static_cast<size_t>(h) * static_cast<size_t>(w), 0.f);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float dx = static_cast<float>(x) - cx;
      float dy = static_cast<float>(y) - cy;
      hm[y * w + x] = std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
    }
  }
  return hm;
}

}  // namespace

TEST(ExtractCentroid, RecoversGaussianPeakWithinOnePixel) {
  const int H = 288;
  const int W = 512;
  auto hm = GaussianHeatmap(H, W, /*cx=*/123.4f, /*cy=*/77.8f, /*sigma=*/5.f);
  float cx = 0;
  float cy = 0;
  float peak = 0;
  ASSERT_TRUE(TrackNetExtractCentroid(hm.data(), H, W, 0.3f, 0.5f, &cx, &cy,
                                      &peak));
  EXPECT_NEAR(cx, 123.4f, 1.0f);
  EXPECT_NEAR(cy, 77.8f, 1.0f);
  EXPECT_GT(peak, 0.99f);
}

TEST(ExtractCentroid, RecoversCornerPeak) {
  const int H = 288, W = 512;
  auto hm = GaussianHeatmap(H, W, /*cx=*/0.f, /*cy=*/0.f, /*sigma=*/3.f);
  float cx = 999, cy = 999, peak = 0;
  ASSERT_TRUE(TrackNetExtractCentroid(hm.data(), H, W, 0.3f, 0.5f, &cx, &cy, &peak));
  EXPECT_LT(cx, 1.5f);
  EXPECT_LT(cy, 1.5f);
}

TEST(ExtractCentroid, FlatBelowThresholdReturnsFalse) {
  const int H = 288;
  const int W = 512;
  std::vector<float> hm(static_cast<size_t>(H) * static_cast<size_t>(W), 0.1f);
  float cx = 0;
  float cy = 0;
  float peak = 0;
  EXPECT_FALSE(TrackNetExtractCentroid(hm.data(), H, W, 0.3f, 0.5f, &cx, &cy,
                                       &peak));
}
