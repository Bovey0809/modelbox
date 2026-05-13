#
# Copyright 2026 The Modelbox Project Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
"""yolo_pose_track_post — pose post-processing + greedy-IoU SORT tracker.

Algorithm mirrors:
  - src/drivers/devices/cpu/flowunit/yolo_pose_post/yolo_pose_post_flowunit.cc
    (Decode + NMS for the [1, 5+17*3, N] anchor-free YOLOv8-pose tensor).
  - src/drivers/devices/cpu/flowunit/yolo_track_post/yolo_track_post_flowunit.cc
    (greedy-IoU SORT-style tracker UpdateTracks).

Emits a structured `tracked_poses` JSON port. C++ port deferred to v2.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import _flowunit as modelbox


def _bbox_iou(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def decode_pose(feat: np.ndarray, conf_threshold: float,
                scale_x: float, scale_y: float) -> list[dict[str, Any]]:
    """Decode [1, 56, N] anchor-free YOLOv8-pose output to per-person dicts.

    The C++ Decode() in yolo_pose_post_flowunit.cc lays the feature tensor out
    as [channels, N] in row-major order (i.e. feat[channel * N + anchor]).
    We expect the same layout here after reshape to (1, 56, N).
    """
    assert feat.ndim == 3 and feat.shape[1] == 56, f"got shape {feat.shape}"
    arr = feat[0]  # shape: (56, N)
    scores = arr[4]
    keep = np.where(scores >= conf_threshold)[0]
    persons = []
    for idx in keep:
        cx, cy, w, h = arr[0, idx], arr[1, idx], arr[2, idx], arr[3, idx]
        x1 = (cx - w / 2.0) * scale_x
        y1 = (cy - h / 2.0) * scale_y
        x2 = (cx + w / 2.0) * scale_x
        y2 = (cy + h / 2.0) * scale_y
        kpts = []
        for k in range(17):
            kx = arr[5 + 3 * k + 0, idx] * scale_x
            ky = arr[5 + 3 * k + 1, idx] * scale_y
            kc = arr[5 + 3 * k + 2, idx]
            kpts.append([float(kx), float(ky), float(kc)])
        persons.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(scores[idx]),
            "kpts": kpts,
        })
    return persons


def nms_pose(persons: list[dict[str, Any]], iou_threshold: float) \
        -> list[dict[str, Any]]:
    """NMS on pose detections.

    Matches yolo_pose_post_flowunit.cc Nms(): sort descending by score, then
    greedy suppress with a boolean mask — O(n^2) but n is small.
    """
    if not persons:
        return []
    ordered = sorted(persons, key=lambda p: -p["score"])
    suppressed = [False] * len(ordered)
    kept: list[dict[str, Any]] = []
    for i, head in enumerate(ordered):
        if suppressed[i]:
            continue
        kept.append(head)
        for j in range(i + 1, len(ordered)):
            if suppressed[j]:
                continue
            if _bbox_iou(head["bbox"], ordered[j]["bbox"]) > iou_threshold:
                suppressed[j] = True
    return kept


class GreedyIoUTracker:
    """Greedy IoU SORT tracker matching yolo_track_post_flowunit.cc UpdateTracks().

    Key behavioral choices matching the C++ reference:
    - next_id starts at 1 (not 0).
    - Tracks are sorted by descending score before matching each frame.
    - A track is pruned when lost > max_lost (strictly greater), matching
      C++: tracks.erase(...[&](const Track& t){ return t.lost > max_lost_; }).
    - New tracks are appended after updating existing ones (birth after death
      check), same order as C++.
    """

    def __init__(self, track_iou: float, max_lost: int):
        self.track_iou = track_iou
        self.max_lost = max_lost
        self.tracks: list[dict[str, Any]] = []
        self.next_id = 1  # matches C++ int next_id{1}

    def update(self, persons: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Sort tracks by descending score, matching C++ UpdateTracks pre-sort.
        self.tracks.sort(key=lambda t: -t["score"])

        matched_dets: set[int] = set()
        for tr in self.tracks:
            best_iou = self.track_iou  # threshold acts as the floor (>=)
            best_idx = -1
            for di, det in enumerate(persons):
                if di in matched_dets:
                    continue
                iou = _bbox_iou(tr["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = di
            if best_idx >= 0:
                tr["bbox"] = persons[best_idx]["bbox"]
                tr["score"] = persons[best_idx]["score"]
                tr["kpts"] = persons[best_idx]["kpts"]
                tr["hits"] += 1
                tr["lost"] = 0
                matched_dets.add(best_idx)
            else:
                tr["lost"] += 1

        # Prune lost tracks — strictly greater than max_lost, matching C++.
        self.tracks = [t for t in self.tracks if t["lost"] <= self.max_lost]

        # Spawn new tracks for unmatched detections.
        for di, det in enumerate(persons):
            if di in matched_dets:
                continue
            self.tracks.append({
                "track_id": self.next_id,
                "bbox": det["bbox"],
                "score": det["score"],
                "kpts": det["kpts"],
                "hits": 1,
                "lost": 0,
            })
            self.next_id += 1

        # Return only tracks matched this frame (lost == 0).
        return [{"track_id": t["track_id"], "bbox": t["bbox"],
                 "score": t["score"], "kpts": t["kpts"]}
                for t in self.tracks if t["lost"] == 0]


class YoloPoseTrackPost(modelbox.FlowUnit):
    def __init__(self):
        super().__init__()
        self.tracker = None
        self.frame_idx = 0
        # Set default attribute values so open() is idempotent and the
        # config-defaults test can call open() on a fresh instance.
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.kpt_threshold = 0.5
        self.track_iou = 0.3
        self.max_lost = 20
        self.net_h = 640
        self.net_w = 640
        self.source_width = 1280
        self.source_height = 720
        self.emit_overlay = False
        self.scale_x = self.source_width / self.net_w
        self.scale_y = self.source_height / self.net_h

    def open(self, config):
        self.conf_threshold = config.get_float("conf_threshold", 0.25)
        self.iou_threshold = config.get_float("iou_threshold", 0.45)
        self.kpt_threshold = config.get_float("kpt_threshold", 0.5)
        self.track_iou = config.get_float("track_iou", 0.3)
        self.max_lost = config.get_int("max_lost", 20)
        self.net_h = config.get_int("net_h", 640)
        self.net_w = config.get_int("net_w", 640)
        self.source_width = config.get_int("source_width", 1280)
        self.source_height = config.get_int("source_height", 720)
        self.emit_overlay = bool(config.get_bool("emit_overlay", False)) \
            if hasattr(config, "get_bool") else False
        self.scale_x = self.source_width / self.net_w
        self.scale_y = self.source_height / self.net_h
        self.tracker = GreedyIoUTracker(self.track_iou, self.max_lost)
        self.frame_idx = 0
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def process(self, data_context):
        feat_in = data_context.input("in_feat")
        poses_out = data_context.output("tracked_poses")
        for buf in feat_in:
            arr = np.frombuffer(buf.as_object(), dtype=np.float32)
            try:
                feat = arr.reshape(1, 56, -1)
            except ValueError:
                modelbox.error(
                    f"yolo_pose_track_post: cannot reshape feat of size "
                    f"{arr.size} into (1, 56, N)")
                return modelbox.Status.StatusCode.STATUS_FAULT
            persons = decode_pose(feat, self.conf_threshold,
                                  self.scale_x, self.scale_y)
            persons = nms_pose(persons, self.iou_threshold)
            tracked = self.tracker.update(persons)
            payload = json.dumps({"frame_idx": self.frame_idx,
                                  "tracks": tracked}).encode("utf-8")
            ob = modelbox.Buffer(self.get_bind_device(), payload)
            ob.set("frame_idx", int(self.frame_idx))
            poses_out.push_back(ob)
            self.frame_idx += 1
        return modelbox.Status.StatusCode.STATUS_SUCCESS

    def close(self):
        return modelbox.Status()

    def data_pre(self, data_context):
        # Reset per-stream state so multi-stream sessions don't carry over
        # tracker state or frame indices.
        if self.tracker is not None:
            self.tracker = GreedyIoUTracker(self.track_iou, self.max_lost)
        self.frame_idx = 0
        return modelbox.Status()

    def data_post(self, data_context):
        return modelbox.Status()
