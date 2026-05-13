#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""hit_centers_emitter — collapse the per-window audio logit stream into a
single list of hit-center video-frame indices for the session.

Smoothing + duration filtering constants mirror hit_intervals_sink_flowunit.cc.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import _flowunit as modelbox


def smooth_majority(binary: np.ndarray, window: int) -> np.ndarray:
    """Majority-vote smoother. window must be odd; even windows are rounded up."""
    if window <= 1:
        return binary.astype(np.int32)
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.zeros_like(binary, dtype=np.int32)
    pad = np.pad(binary.astype(np.int32), half, mode="edge")
    for i in range(len(binary)):
        win = pad[i:i + window]
        out[i] = 1 if win.sum() * 2 > window else 0
    return out


def intervals_to_centers(logits: np.ndarray, video_fps: float,
                         step_sec: float, window_sec: float,
                         threshold: float, smooth_win: int,
                         max_hit_dur: float) -> list[dict[str, Any]]:
    """Find hit intervals from per-window 2-class logits and map their centers
    to source-video frame indices.

    Each window represents the audio segment [k*step_sec, k*step_sec+window_sec].
    A hit's audio-time center is approximated as midpoint(interval) + window_sec/2.
    """
    if logits.ndim != 2 or logits.shape[1] != 2:
        return []
    # Class 1 dominant => hit (no softmax needed for argmax decisions).
    binary = (logits[:, 1] > logits[:, 0]).astype(np.int32)
    smoothed = smooth_majority(binary, max(1, smooth_win | 1))
    centers: list[dict[str, Any]] = []
    in_run = False
    run_start = 0
    for k, val in enumerate(smoothed):
        if val and not in_run:
            in_run = True
            run_start = k
        elif not val and in_run:
            in_run = False
            run_end = k - 1
            _maybe_add_center(logits, run_start, run_end, step_sec,
                              window_sec, max_hit_dur, video_fps, centers)
    if in_run:
        _maybe_add_center(logits, run_start, len(smoothed) - 1, step_sec,
                          window_sec, max_hit_dur, video_fps, centers)
    return centers


def _maybe_add_center(logits, run_start, run_end, step_sec, window_sec,
                      max_hit_dur, video_fps, centers):
    t_start = run_start * step_sec
    t_end = run_end * step_sec + window_sec
    if (t_end - t_start) > max_hit_dur:
        return
    t_center = (t_start + t_end) / 2.0
    frame_idx = int(round(t_center * video_fps))
    audio_conf = float(logits[run_start:run_end + 1, 1].mean())
    centers.append({"frame_idx": frame_idx, "audio_conf": audio_conf,
                    "t_center_sec": t_center})


class HitCentersEmitter(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self._logits: list[np.ndarray] = []

    def open(self, config):
        self.video_fps = config.get_float("video_fps", 30.0)
        self.step_sec = config.get_float("step_sec", 0.02)
        self.window_sec = config.get_float("window_sec", 0.5)
        self.threshold = config.get_float("threshold", 0.5)
        self.smooth_win = config.get_int("smooth_win", 5)
        self.max_hit_dur = config.get_float("max_hit_dur", 1.0)
        self._logits = []
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        logits_in = data_context.input("logits")
        for buf in logits_in:
            arr = np.frombuffer(buf.as_object(), dtype=np.float32)
            self._logits.append(arr.reshape(-1, 2))
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_pre(self, data_context):
        self._logits = []
        return modelbox.Status()

    def data_post(self, data_context):
        out = data_context.output("hit_centers")
        if not self._logits:
            payload = json.dumps([]).encode("utf-8")
            ob = modelbox.Buffer(self.get_bind_device(), payload)
            out.push_back(ob)
            return modelbox.Status()
        merged = np.concatenate(self._logits, axis=0)
        centers = intervals_to_centers(
            merged, self.video_fps, self.step_sec, self.window_sec,
            self.threshold, self.smooth_win, self.max_hit_dur
        )
        payload = json.dumps(centers).encode("utf-8")
        ob = modelbox.Buffer(self.get_bind_device(), payload)
        ob.set("num_hits", len(centers))
        out.push_back(ob)
        return modelbox.Status()

    def close(self):
        return modelbox.Status()
