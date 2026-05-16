#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""ball_history_collector — collapse per-frame ball_pos float32[4] buffers
(cx, cy, peak, frame_idx) into a single JSON buffer emitted at session end.

Output payload: {"frame_idx_str": [cx, cy, peak], ...}

String keys are used so the dict survives JSON serialisation which requires
all keys to be strings.
"""

from __future__ import annotations

import json

import numpy as np
import _flowunit as modelbox


class BallHistoryCollector(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self._ball: dict[str, list[float]] = {}

    def open(self, config):
        self._ball = {}
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_pre(self, data_context):
        self._ball = {}
        return modelbox.Status()

    def process(self, data_context):
        for buf in data_context.input("ball_pos"):
            arr = np.frombuffer(buf.as_object(), dtype=np.float32)
            if arr.size < 4:
                modelbox.error(
                    f"ball_history_collector: ball_pos buffer too small "
                    f"({arr.size} floats, need 4)")
                return modelbox.Status.StatusCode.STATUS_FAULT
            cx = float(arr[0])
            cy = float(arr[1])
            peak = float(arr[2])
            fi = int(arr[3])
            self._ball[str(fi)] = [cx, cy, peak]
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_post(self, data_context):
        out = data_context.output("ball_history")
        payload = json.dumps(self._ball).encode("utf-8")
        ob = modelbox.Buffer(self.get_bind_device(), payload)
        ob.set("num_frames", len(self._ball))
        out.push_back(ob)
        return modelbox.Status()

    def close(self):
        return modelbox.Status()
