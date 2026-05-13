#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
"""tracknet_ball_emitter — consume a per-frame TrackNet heatmap (CHW float),
extract the ball centroid, and emit a structured (cx, cy, peak, frame_idx)
buffer in source-frame coordinates.

Mirrors the algorithm in
src/demo/apple_silicon_yolo/flowunit/tracknet_post/tracknet_post_flowunit.cc
(extern "C" TrackNetExtractCentroid). C++ port deferred to v2.
"""

from __future__ import annotations

import numpy as np
import _flowunit as modelbox


def extract_centroid(heat: np.ndarray, score_thr: float, mask_ratio: float,
                     net_h: int, net_w: int) -> tuple[float, float, float]:
    """Weighted-centroid of the foreground heatmap channel.

    Mirrors TrackNetExtractCentroid() in tracknet_post_flowunit.cc:
      - Take max over foreground channels (channels 1+) to get a single plane.
      - Find the global peak.
      - If peak < score_thr: return (-1, -1, peak).
      - Build weighted centroid over all pixels >= peak * mask_ratio.

    Args:
        heat:       CHW float32 array (C >= 2; channel 0 is background).
        score_thr:  Minimum peak value to emit a detection.
        mask_ratio: Weighted-centroid threshold = peak * mask_ratio.
        net_h:      Heatmap height (not used in computation, kept for API parity).
        net_w:      Heatmap width  (not used in computation, kept for API parity).

    Returns:
        (cx, cy, peak) in net-resolution (pixel) coordinates, or
        (-1.0, -1.0, peak) when peak < score_thr.
    """
    if heat.ndim != 3 or heat.shape[0] < 2:
        return (-1.0, -1.0, 0.0)

    # Max over foreground channels (channels 1+), matching heatmap_channel=2
    # default in the C++ unit which uses a single configurable channel; here
    # we take the element-wise max so any active channel contributes.
    fg = heat[1:].max(axis=0)  # shape (H, W)

    peak = float(fg.max())
    if peak < score_thr:
        return (-1.0, -1.0, peak)

    thr = peak * mask_ratio
    mask = fg >= thr
    ys, xs = np.where(mask)
    weights = fg[ys, xs].astype(np.float64)
    ws = weights.sum()
    if ws <= 0:
        peak_idx = int(np.argmax(fg))
        peak_y, peak_x = divmod(peak_idx, fg.shape[1])
        return (float(peak_x), float(peak_y), peak)

    cx = float((xs.astype(np.float64) * weights).sum() / ws)
    cy = float((ys.astype(np.float64) * weights).sum() / ws)
    return (cx, cy, peak)


class TracknetBallEmitter(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self.frame_idx = 0

    def open(self, config):
        self.score_thr = config.get_float("score_thr", 0.3)
        self.mask_ratio = config.get_float("mask_ratio", 0.5)
        self.net_h = config.get_int("net_h", 288)
        self.net_w = config.get_int("net_w", 512)
        self.source_width = config.get_int("source_width", 1280)
        self.source_height = config.get_int("source_height", 720)
        self.scale_x = self.source_width / self.net_w
        self.scale_y = self.source_height / self.net_h
        self.frame_idx = 0
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        heatmaps = data_context.input("heatmaps")
        ball_out = data_context.output("ball_pos")
        for buf in heatmaps:
            arr = np.frombuffer(buf.as_object(), dtype=np.float32)
            try:
                arr = arr.reshape(3, self.net_h, self.net_w)
            except ValueError:
                modelbox.error(
                    f"tracknet_ball_emitter: cannot reshape heatmap of size "
                    f"{arr.size} into (3,{self.net_h},{self.net_w})")
                return modelbox.Status.StatusCode.STATUS_FAULT
            cx, cy, peak = extract_centroid(
                arr, self.score_thr, self.mask_ratio, self.net_h, self.net_w
            )
            if cx >= 0:
                cx *= self.scale_x
                cy *= self.scale_y
            out = np.array([cx, cy, peak, float(self.frame_idx)],
                           dtype=np.float32)
            ob = modelbox.Buffer(self.get_bind_device(), out.tobytes())
            ob.set("frame_idx", int(self.frame_idx))
            ball_out.push_back(ob)
            self.frame_idx += 1
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()

    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
