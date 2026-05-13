#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""hit_postprocess — collapse flowunit that consumes the per-window logits
stream from a hit_sound inference flowunit and emits hit/rally intervals.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import _flowunit as modelbox

_HERE = os.path.dirname(os.path.abspath(__file__))
_UTILS = os.path.normpath(os.path.join(_HERE, "..", "hit_sound_detect"))
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)
from hit_sound_utils import (  # noqa: E402
    BufferedRallyDetector,
    DurationFilter,
    OutputSmoother,
    merge_positive_windows,
    write_overlay_video,
)


def _softmax_pos(logits: np.ndarray) -> float:
    x = logits.astype(np.float64).reshape(-1)
    x = x - x.max()
    e = np.exp(x)
    return float(e[1] / e.sum())


class HitPostprocess(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()

    def open(self, config):
        self.step_sec = config.get_float("step_sec", 0.02)
        self.window_sec = config.get_float("window_sec", 0.5)
        self.threshold = config.get_float("threshold", 0.5)
        self.smooth_win = config.get_int("smooth_win", 5)
        self.max_hit_dur = config.get_float("max_hit_dur", 1.0)
        self.max_gap = config.get_float("max_gap", 3.0)
        self.buffer_time = config.get_float("buffer_time", 4.0)
        self.interval_mode = config.get_string("interval_mode", "hit")
        self.intervals_csv = config.get_string("intervals_csv", "/tmp/hit_sound_intervals.csv")
        self.overlay_mp4 = config.get_string("overlay_mp4", "")
        self.video_path = config.get_string("video_path", "")
        self.write_overlay = config.get_bool("write_overlay", True)
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        logits_list = data_context.input("logits")
        out_data = data_context.output("out_data")

        smoother = OutputSmoother(window_size=self.smooth_win)
        dur_filter = DurationFilter(max_duration=self.max_hit_dur)
        rally = BufferedRallyDetector(
            max_inter_hit_time=self.max_gap,
            buffer_time=self.buffer_time,
            step=self.step_sec,
        )

        pos_windows: List[Tuple[float, float]] = []
        n = 0
        for buf in logits_list:
            raw = np.array(buf.as_object(), copy=False)
            logits = raw.view(np.float32) if raw.dtype != np.float32 else raw
            prob = _softmax_pos(logits)
            # The emitter sets `time` on each buffer; the inference flowunit's
            # native impl drops custom meta, so fall back to the index when
            # missing (time = idx * step + window/2).
            try:
                center = float(buf.get("time"))
            except Exception:
                center = n * self.step_sec + self.window_sec / 2.0
            pred_raw = 1 if prob >= self.threshold else 0
            pred_smooth = smoother.smooth(pred_raw)
            pred_filt = dur_filter.filter(center, pred_smooth)
            rally.process(center, pred_filt)
            if pred_filt == 1:
                start = center - self.window_sec / 2.0
                pos_windows.append((start, start + self.window_sec))
            n += 1

        hit_intervals = merge_positive_windows(pos_windows, join_gap=self.step_sec)
        ts, states = rally.finalize()
        rally_intervals: List[Tuple[float, float]] = []
        start_t = None
        prev_t = None
        for t, s in zip(ts.tolist(), states.tolist()):
            t = float(t); s = int(s)
            if s == 1 and start_t is None:
                start_t = t
            elif s == 0 and start_t is not None:
                rally_intervals.append((start_t, (prev_t if prev_t is not None else t) + self.step_sec))
                start_t = None
            prev_t = t
        if start_t is not None and prev_t is not None:
            rally_intervals.append((start_t, prev_t + self.step_sec))

        selected = rally_intervals if self.interval_mode in ("rally", "both") else hit_intervals

        if self.intervals_csv:
            Path(self.intervals_csv).parent.mkdir(parents=True, exist_ok=True)
            Path(self.intervals_csv).write_text(
                "\n".join([f"{s:.3f},{e:.3f}" for s, e in selected]),
                encoding="utf-8",
            )

        if self.write_overlay and self.overlay_mp4 and self.video_path and os.path.exists(self.video_path):
            try:
                Path(self.overlay_mp4).parent.mkdir(parents=True, exist_ok=True)
                write_overlay_video(self.video_path, self.overlay_mp4, selected)
            except Exception as exc:
                modelbox.error(f"hit_postprocess: overlay write failed: {exc}")

        payload = {
            "windows": int(n),
            "mode": self.interval_mode,
            "hit_intervals": [[round(s, 3), round(e, 3)] for s, e in hit_intervals],
            "rally_intervals": [[round(s, 3), round(e, 3)] for s, e in rally_intervals],
            "intervals_csv": self.intervals_csv,
            "overlay_mp4": self.overlay_mp4 if self.write_overlay else "",
            "video": self.video_path,
        }
        body = (json.dumps(payload) + chr(0)).encode("utf-8").strip()
        out_data.push_back(modelbox.Buffer(self.get_bind_device(), body))
        modelbox.info(
            f"hit_postprocess: windows={n} hits={len(hit_intervals)} "
            f"rallies={len(rally_intervals)} mode={self.interval_mode}"
        )
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()

    def data_pre(self, data_context):
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
