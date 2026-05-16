#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
"""poses_history_collector — collapse per-frame tracked_poses JSON buffers
into a single JSON buffer emitted at session end.

Input buffer body (per yolo_pose_track_post): JSON bytes with schema
  {"frame_idx": int, "tracks": [{track_id, bbox, kpts, score}, ...]}

Output payload: {"frame_idx_str": [{track_id, bbox, kpts, score}, ...], ...}

String keys are used so the dict survives JSON serialisation.
"""

from __future__ import annotations

import json

import _flowunit as modelbox


class PosesHistoryCollector(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self._poses: dict[str, list] = {}

    def open(self, config):
        self._poses = {}
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_pre(self, data_context):
        self._poses = {}
        return modelbox.Status()

    def process(self, data_context):
        for buf in data_context.input("tracked_poses"):
            try:
                payload = json.loads(bytes(buf.as_object()).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                modelbox.error(
                    f"poses_history_collector: failed to parse tracked_poses JSON: {exc}")
                return modelbox.Status.StatusCode.STATUS_FAULT
            fi = int(payload.get("frame_idx", -1))
            if fi < 0:
                modelbox.error(
                    "poses_history_collector: tracked_poses buffer missing frame_idx")
                return modelbox.Status.StatusCode.STATUS_FAULT
            self._poses[str(fi)] = payload.get("tracks", [])
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def data_post(self, data_context):
        out = data_context.output("poses_history")
        payload = json.dumps(self._poses).encode("utf-8")
        ob = modelbox.Buffer(self.get_bind_device(), payload)
        ob.set("num_frames", len(self._poses))
        out.push_back(ob)
        return modelbox.Status()

    def close(self):
        return modelbox.Status()
